"""Build the exact teacher-forced Stage 3F-A anchor cache.

The Stage 3C state sequence is replayed with the same seed and spatial split;
every accepted state is checked against ``sample_index.jsonl`` before the
original ``Path.generate_target_maps`` implementation is called.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import default_collate


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.trajectory_anchor_fusion import (  # noqa: E402
    collect_anchor_prehead_features,
)
from model.trajectory_evidence_encoder import (  # noqa: E402
    TrajectoryEvidenceEncoder,
)
from model.trajectory_support_head import TrajectorySupportHead  # noqa: E402
from scripts.prepare_stage3c_branch_dataset import (  # noqa: E402
    _advance_teacher_forced_path,
    _make_paths,
    _new_path,
)
from train_branch_aux import (  # noqa: E402
    _build_auxiliary_modules,
    _load_config,
    _load_frozen_rpnet,
    _move_nested,
)
from utils.branch_targets import build_immediate_branch_targets  # noqa: E402
from utils.stage3c_branch_dataset import Stage3CBranchDataset  # noqa: E402
from utils.stage3c_checkpoint import load_stage3c_checkpoint  # noqa: E402
from utils.stage3fa_loss import original_vecroad_anchor_losses  # noqa: E402
from utils.stage3d_checkpoint import load_stage3d_support_checkpoint  # noqa: E402
from utils.stage3e0_checkpoint import load_stage3e0_checkpoint  # noqa: E402
from utils.stage3fa_anchor_cache import (  # noqa: E402
    array_schema,
    write_stage3fa_manifest,
    write_stage3fa_shard,
)
from utils.trajectory_evidence_robustness import (  # noqa: E402
    deterministic_fragment_thinning,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for block in iter(lambda: input_file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _module_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(memoryview(value.detach().cpu().contiguous().numpy()))
    return digest.hexdigest()


def _resolve(path: Any) -> Path:
    value = Path(str(path))
    return value if value.is_absolute() else ROOT / value


def _load_index(dataset_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    records = {"train": [], "val": []}
    with (dataset_dir / "sample_index.jsonl").open(
            "r", encoding="utf-8") as input_file:
        for line in input_file:
            record = json.loads(line)
            records[record["split"]].append(record)
    return records


def _assert_record(record: Mapping[str, Any], *, path_index: int, state,
                   branch_count: int, center_xy) -> None:
    checks = {
        "subtile_index": (int(path_index), int(record["subtile_index"])),
        "vertex_id": (int(state.vertex.id), int(record["vertex_id"])),
        "is_key_point": (bool(state.is_key_point), bool(record["is_key_point"])),
        "branch_count": (int(branch_count), int(record["branch_count"])),
    }
    failed = {key: value for key, value in checks.items() if value[0] != value[1]}
    if not np.allclose(center_xy, record["center_xy"], rtol=0.0, atol=1e-6):
        failed["center_xy"] = (list(center_xy), record["center_xy"])
    if failed:
        raise RuntimeError("Stage 3C replay diverged: {}".format(failed))


def _replay_split(
    *, split: str, records: Sequence[Mapping[str, Any]],
    path_indices: Sequence[int], paths, subtiles, cfg,
    forbidden_centers: set, collected_centers: set,
) -> Iterable[Dict[str, Any]]:
    active = list(path_indices)
    cursor = 0
    sample_index = 0
    attempts = 0
    while sample_index < len(records):
        attempts += 1
        if attempts > max(len(records) * 50, 1000):
            raise RuntimeError("teacher-forced replay exceeded attempt budget")
        active_position = cursor % len(active)
        path_index = active[active_position]
        path = paths[path_index]
        state = path.pop_state(
            follow_order=False, probs=[0.15, 0.8, 0.05],
            WINDOW_SIZE=cfg.TRAIN.WINDOW_SIZE)
        if state is None or len(path.graph.vertices) >= cfg.TRAIN.MAX_PATH_LENGTH:
            paths[path_index] = _new_path(path_index, subtiles[path_index])
            continue
        cursor += 1
        local_input = path.make_path_input(
            extension_vertex=state.vertex,
            fetch_list=["aerial_image_chw", "walked_path_small", "road_seg_thick3"],
            traj_filter=False, is_key_point=state.is_key_point,
            WINDOW_SIZE=cfg.TRAIN.WINDOW_SIZE)
        road_segmentation = (
            local_input["road_seg_thick3"][0]
            if state.vertex.edge_pos is not None else None)
        target_poses = path.get_target_poses(
            extension_vertex=state.vertex,
            road_segmentation=road_segmentation,
            STEP_LENGTH=cfg.TRAIN.STEP_LENGTH,
            is_key_point=state.is_key_point,
            NUM_TARGETS=cfg.TRAIN.NUM_TARGETS,
            RECT_RADIUS=cfg.TRAIN.RECT_RADIUS,
            WINDOW_SIZE=cfg.TRAIN.WINDOW_SIZE)
        center_xy = (float(state.vertex.point.x), float(state.vertex.point.y))
        center_key = (round(center_xy[0], 6), round(center_xy[1], 6))
        if center_key in forbidden_centers:
            _advance_teacher_forced_path(
                path=path, state=state, target_poses=target_poses,
                road_segmentation=road_segmentation, cfg=cfg)
            continue
        branch_targets = build_immediate_branch_targets(
            target_poses=target_poses, current_vertex=state.vertex,
            graph=path.gc.graph, window_size=cfg.TRAIN.WINDOW_SIZE)
        record = records[sample_index]
        _assert_record(
            record, path_index=path_index, state=state,
            branch_count=branch_targets.branch_count, center_xy=center_xy)
        target = path.generate_target_maps(
            extension_vertex=state.vertex, target_poses=target_poses,
            NUM_TARGETS=cfg.TRAIN.NUM_TARGETS,
            WINDOW_SIZE=cfg.TRAIN.WINDOW_SIZE,
            is_key_point=state.is_key_point).astype(np.float32)
        max_branches = int(cfg.STAGE3C.DATASET.MAX_BRANCHES)
        next_node_xy = np.zeros((max_branches, 2), dtype=np.float32)
        next_node_mask = np.zeros(max_branches, dtype=np.bool_)
        count = int(branch_targets.branch_count)
        if count:
            offsets = branch_targets.branch_offsets_rel.numpy()
            next_node_xy[:count] = offsets + np.asarray(center_xy, dtype=np.float32)
            next_node_mask[:count] = True
        end_index = (
            1 if state.is_key_point
            else target_poses.get_supervision_end_index())
        sample = {
            "sample_id": int(record["sample_index"]),
            "dataset_index": int(sample_index),
            "subtile_index": int(path_index),
            "vertex_id": int(state.vertex.id),
            "center_xy": np.asarray(center_xy, dtype=np.float32),
            "is_key_point": bool(state.is_key_point),
            "branch_count": count,
            "category_id": 1 if count == 1 else 2 if count == 2 else 3 if count >= 3 else 0,
            "supervision_end_index": int(end_index),
            "next_node_xy": next_node_xy,
            "next_node_mask": next_node_mask,
            "anchor_target": target,
            "aerial_image": np.asarray(local_input["aerial_image_chw"], dtype=np.float32),
            "walked_path": np.asarray(local_input["walked_path_small"], dtype=np.float32),
        }
        collected_centers.add(center_key)
        _advance_teacher_forced_path(
            path=path, state=state, target_poses=target_poses,
            road_segmentation=road_segmentation, cfg=cfg)
        sample_index += 1
        yield sample


def _loss_per_sample(logits: torch.Tensor, targets: torch.Tensor,
                     end_indices: torch.Tensor) -> torch.Tensor:
    values = []
    for row in range(logits.shape[0]):
        end = int(end_indices[row])
        values.append(original_vecroad_anchor_losses(
            logits[row:row + 1], logits[row:row + 1],
            targets[row:row + 1], end_indices[row:row + 1])[
                "anchor_loss"])
    return torch.stack(values)


def _process_batch(
    replay_samples: Sequence[Mapping[str, Any]], dataset, *, rpnet,
    trajectory_encoder, evidence_encoder, device, num_targets: int,
) -> tuple[Dict[str, np.ndarray], Dict[str, float]]:
    trajectory_cpu = default_collate([
        dataset[int(sample["dataset_index"])] for sample in replay_samples])
    trajectory_batch = _move_nested(trajectory_cpu, device)["trajectory_batch"]
    aerial = torch.from_numpy(np.stack([
        sample["aerial_image"] for sample in replay_samples])).to(device)
    walked = torch.from_numpy(np.stack([
        sample["walked_path"] for sample in replay_samples])).to(device)
    targets = torch.from_numpy(np.stack([
        sample["anchor_target"] for sample in replay_samples])).to(device)
    end_indices = torch.tensor([
        sample["supervision_end_index"] for sample in replay_samples],
        device=device, dtype=torch.long)
    sample_ids = torch.tensor([
        sample["sample_id"] for sample in replay_samples], dtype=torch.long,
        device=device)
    with torch.no_grad():
        outputs = rpnet(
            aerial_image=aerial, traj_image=None, aerial_traj_image=None,
            neighborhood_trajectory_norm=None, valid_mask=None,
            walked_path=walked, NUM_TARGETS=num_targets, test=False,
            model="origin", use_traj=False)
        prehead = collect_anchor_prehead_features(
            outputs["feature_maps"], num_targets)
        trajectory = trajectory_encoder(trajectory_batch)
        evidence = evidence_encoder(
            trajectory["fragment_tokens"], trajectory["fragment_mask"],
            return_attention=True)
        retained_mask = deterministic_fragment_thinning(
            fragment_mask=trajectory_batch["fragment_mask"],
            sample_ids=sample_ids,
            track_indices=trajectory_batch["track_indices"],
            start_point_indices=trajectory_batch["start_point_indices"],
            end_point_indices=trajectory_batch["end_point_indices"],
            retain_ratio=0.25)
        retained_input = dict(trajectory_batch)
        retained_input["fragment_mask"] = retained_mask
        retained_input["point_mask"] = (
            trajectory_batch["point_mask"] & retained_mask.unsqueeze(-1))
        retained_trajectory = trajectory_encoder(retained_input)
        retained_evidence = evidence_encoder(
            retained_trajectory["fragment_tokens"],
            retained_trajectory["fragment_mask"], return_attention=True)

        reconstructed_anchor = torch.cat([
            rpnet.conv_final(prehead["anchor_features"][:, step])
            for step in range(num_targets)], dim=1)
        reconstructed_lowrs = torch.cat([
            torch.nn.functional.interpolate(
                rpnet.next_step_final(
                    prehead["anchor_lowrs_features"][:, step]),
                scale_factor=4, mode="bilinear", align_corners=True)
            for step in range(num_targets)], dim=1)
        output_diff = max(
            float((reconstructed_anchor - outputs["anchor"]).abs().max()),
            float((reconstructed_lowrs - outputs["anchor_lowrs"]).abs().max()))
        dynamic_loss = _loss_per_sample(
            outputs["anchor"], targets, end_indices)
        dynamic_lowrs_loss = _loss_per_sample(
            outputs["anchor_lowrs"], targets, end_indices)
        cached_loss = _loss_per_sample(
            reconstructed_anchor, targets, end_indices)
        cached_lowrs_loss = _loss_per_sample(
            reconstructed_lowrs, targets, end_indices)
        loss_diff = max(
            float((dynamic_loss - cached_loss).abs().max()),
            float((dynamic_lowrs_loss - cached_lowrs_loss).abs().max()),
            float(((dynamic_loss + dynamic_lowrs_loss)
                   - (cached_loss + cached_lowrs_loss)).abs().max()))

    # Both pre-head features remain float32.  Float16 overflowed for the
    # recursive low-resolution state, and even the finite full-resolution
    # feature changed reconstructed anchor logits by about 1e-2 on the
    # canonical checkpoint.  Stage 3F-A requires <=1e-6 cache equivalence.
    anchor_features = (
        prehead["anchor_features"].cpu().numpy().astype(np.float32))
    anchor_lowrs_features = (
        prehead["anchor_lowrs_features"].cpu().numpy().astype(np.float32))
    for name, value in (
            ("anchor_features", anchor_features),
            ("anchor_lowrs_features", anchor_lowrs_features)):
        if not np.isfinite(value).all():
            raise RuntimeError(
                "{} contains NaN/Inf after cache dtype conversion".format(name))
    # Validate the actual serialized representation, not merely the dynamic
    # tensors from which it was produced.
    with torch.no_grad():
        cached_anchor_features = torch.from_numpy(
            anchor_features).to(device=device)
        cached_lowrs_features = torch.from_numpy(
            anchor_lowrs_features).to(device=device)
        serialized_anchor = torch.cat([
            rpnet.conv_final(cached_anchor_features[:, step])
            for step in range(num_targets)], dim=1)
        serialized_lowrs = torch.cat([
            torch.nn.functional.interpolate(
                rpnet.next_step_final(cached_lowrs_features[:, step]),
                scale_factor=4, mode="bilinear", align_corners=True)
            for step in range(num_targets)], dim=1)
        serialized_output_diff = max(
            float((serialized_anchor - outputs["anchor"]).abs().max()),
            float((serialized_lowrs - outputs["anchor_lowrs"]).abs().max()))
        serialized_loss = _loss_per_sample(
            serialized_anchor, targets, end_indices)
        serialized_lowrs_loss = _loss_per_sample(
            serialized_lowrs, targets, end_indices)
        serialized_loss_diff = max(
            float((dynamic_loss - serialized_loss).abs().max()),
            float((dynamic_lowrs_loss - serialized_lowrs_loss).abs().max()),
            float(((dynamic_loss + dynamic_lowrs_loss)
                   - (serialized_loss + serialized_lowrs_loss)).abs().max()))
        output_diff = max(output_diff, serialized_output_diff)
        loss_diff = max(loss_diff, serialized_loss_diff)
    values = {
        "anchor_features": anchor_features,
        "anchor_lowrs_features": anchor_lowrs_features,
        "original_anchor_logits": outputs["anchor"].cpu().numpy().astype(np.float32),
        "original_anchor_lowrs_logits": outputs["anchor_lowrs"].cpu().numpy().astype(np.float32),
        "trajectory_evidence": evidence["trajectory_evidence_tokens"].cpu().numpy().astype(np.float32),
        "trajectory_evidence_retain25": retained_evidence["trajectory_evidence_tokens"].cpu().numpy().astype(np.float32),
        "trajectory_available": evidence["trajectory_evidence_mask"].any(dim=1).cpu().numpy().astype(np.bool_),
        "evidence_attention": evidence["fragment_attention_weights"].cpu().numpy().astype(np.float32),
        "evidence_attention_retain25": retained_evidence["fragment_attention_weights"].cpu().numpy().astype(np.float32),
    }
    scalar_types = {
        "sample_id": np.int64, "dataset_index": np.int64,
        "subtile_index": np.int64, "vertex_id": np.int64,
        "is_key_point": np.bool_, "branch_count": np.int64,
        "category_id": np.int64, "supervision_end_index": np.int64,
    }
    for name, dtype in scalar_types.items():
        values[name] = np.asarray([
            sample[name] for sample in replay_samples], dtype=dtype)
    for name, dtype in (
            ("center_xy", np.float32), ("next_node_xy", np.float32),
            ("next_node_mask", np.bool_), ("anchor_target", np.float32)):
        values[name] = np.stack([
            sample[name] for sample in replay_samples]).astype(dtype)
    values["anchor_lowrs_target"] = values["anchor_target"].copy()
    return values, {"output_max_abs_diff": output_diff,
                    "loss_max_abs_diff": loss_diff}


class _Accumulator:
    def __init__(self, cache_dir: Path, split: str, shard_size: int) -> None:
        self.cache_dir = cache_dir
        self.split = split
        self.shard_size = int(shard_size)
        self.pending: Dict[str, List[np.ndarray]] = {}
        self.count = 0
        self.shards = []
        self.schema = None

    def add(self, arrays: Mapping[str, np.ndarray]) -> None:
        for row in range(next(iter(arrays.values())).shape[0]):
            for name, value in arrays.items():
                self.pending.setdefault(name, []).append(value[row])
            self.count += 1
            if len(next(iter(self.pending.values()))) >= self.shard_size:
                self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        arrays = {name: np.stack(values) for name, values in self.pending.items()}
        if self.schema is None:
            self.schema = array_schema(arrays)
        elif self.schema != array_schema(arrays):
            raise RuntimeError("Stage 3F-A cache schema changed")
        name = "{}_{:04d}.npz".format(self.split, len(self.shards))
        self.shards.append(write_stage3fa_shard(
            self.cache_dir / self.split / name, arrays))
        self.pending = {}


def build_cache(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = _load_config(args.config)
    stage = cfg.STAGE3FA
    cache_dir = args.output_dir or _resolve(stage.CACHE_DIR)
    if cache_dir.exists() and any(cache_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError("non-empty cache; pass --overwrite: {}".format(cache_dir))
        shutil.rmtree(str(cache_dir))
    cache_dir.mkdir(parents=True, exist_ok=True)
    seed = 20260724  # Must replay the immutable Stage 3C teacher-forced split.
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(args.device)
    dataset_dir = _resolve(cfg.STAGE3C.DATASET_DIR)
    records = _load_index(dataset_dir)
    train_dataset = Stage3CBranchDataset(dataset_dir, "train")
    val_dataset = Stage3CBranchDataset(dataset_dir, "val")
    image_checkpoint = _resolve(cfg.STAGE3C.IMAGE_CHECKPOINT)
    e4_checkpoint = _resolve(stage.get("E4_CHECKPOINT", cfg.STAGE3E0.E4_CHECKPOINT))
    evidence_checkpoint = _resolve(stage.CANONICAL_EVIDENCE_CHECKPOINT)
    support_checkpoint = _resolve(stage.SUPPORT_CHECKPOINT)
    evidence_sha = _sha256(evidence_checkpoint)
    if evidence_sha != str(stage.EXPECTED_EVIDENCE_SHA256):
        raise RuntimeError("canonical evidence checkpoint SHA-256 mismatch")

    rpnet, _ = _load_frozen_rpnet(cfg, image_checkpoint, device)
    trajectory_encoder, graph_encoder, branch_decoder = _build_auxiliary_modules(cfg, device)
    load_stage3c_checkpoint(
        e4_checkpoint, trajectory_encoder=trajectory_encoder,
        graph_state_encoder=graph_encoder, branch_decoder=branch_decoder,
        map_location="cpu")
    evidence_encoder = TrajectoryEvidenceEncoder(
        hidden_dim=128, num_evidence_tokens=1, num_heads=4,
        dropout=0.1, aggregation_mode="latent_attention").to(device)
    load_stage3e0_checkpoint(
        evidence_checkpoint, evidence_encoder=evidence_encoder,
        map_location="cpu", expected_e4_sha256=_sha256(e4_checkpoint))
    support_head = TrajectorySupportHead(hidden_dim=128).to(device)
    load_stage3d_support_checkpoint(
        support_checkpoint, support_head=support_head, map_location="cpu")
    frozen_modules = {
        "rpnet": rpnet, "trajectory_encoder": trajectory_encoder,
        "graph_state_encoder": graph_encoder,
        "branch_decoder": branch_decoder,
        "trajectory_evidence_encoder": evidence_encoder,
        "support_head": support_head,
    }
    for module in frozen_modules.values():
        module.eval().requires_grad_(False)
    frozen_sha = {name: _module_sha256(module) for name, module in frozen_modules.items()}

    paths, subtiles = _make_paths(cfg)
    val_indices = sorted(set(int(value) for value in cfg.STAGE3C.DATASET.VAL_SUBTILE_INDICES))
    train_indices = [index for index in range(len(paths)) if index not in val_indices]
    train_centers, val_centers = set(), set()
    reports, schema = {}, None
    max_output_diff = max_loss_diff = 0.0
    started = time.perf_counter()
    for split, indices, dataset, forbidden, collected in (
            ("train", train_indices, train_dataset, set(), train_centers),
            ("val", val_indices, val_dataset, train_centers, val_centers)):
        accumulator = _Accumulator(
            cache_dir, split, int(stage.CACHE.SHARD_SIZE))
        batch = []
        for sample in _replay_split(
                split=split, records=records[split], path_indices=indices,
                paths=paths, subtiles=subtiles, cfg=cfg,
                forbidden_centers=forbidden, collected_centers=collected):
            batch.append(sample)
            if len(batch) >= int(stage.CACHE.FORWARD_BATCH_SIZE):
                arrays, equivalence = _process_batch(
                    batch, dataset, rpnet=rpnet,
                    trajectory_encoder=trajectory_encoder,
                    evidence_encoder=evidence_encoder, device=device,
                    num_targets=int(cfg.TRAIN.NUM_TARGETS))
                accumulator.add(arrays); batch = []
                max_output_diff = max(max_output_diff, equivalence["output_max_abs_diff"])
                max_loss_diff = max(max_loss_diff, equivalence["loss_max_abs_diff"])
                if accumulator.count % 32 == 0:
                    print("{} cache: {}/{}".format(
                        split, accumulator.count, len(records[split])), flush=True)
        if batch:
            arrays, equivalence = _process_batch(
                batch, dataset, rpnet=rpnet,
                trajectory_encoder=trajectory_encoder,
                evidence_encoder=evidence_encoder, device=device,
                num_targets=int(cfg.TRAIN.NUM_TARGETS))
            accumulator.add(arrays)
            max_output_diff = max(max_output_diff, equivalence["output_max_abs_diff"])
            max_loss_diff = max(max_loss_diff, equivalence["loss_max_abs_diff"])
        accumulator.flush()
        schema = accumulator.schema if schema is None else schema
        reports[split] = {"sample_count": accumulator.count,
                          "shards": accumulator.shards}
    tolerance = float(stage.CACHE.STRICT_EQUIVALENCE_TOLERANCE)
    if max_output_diff > tolerance or max_loss_diff > tolerance:
        raise RuntimeError("dynamic/cache equivalence failed")
    frozen_after = {name: _module_sha256(module) for name, module in frozen_modules.items()}
    if frozen_after != frozen_sha:
        raise RuntimeError("a frozen module changed while building cache")
    manifest = {
        "source_stage3c_dataset": str(dataset_dir),
        "sample_ids_identical_to_stage3c": True,
        "teacher_forced_replay_seed": seed,
        "feature_dtype": {
            "anchor_features": "float32",
            "anchor_lowrs_features": "float32",
        },
        "array_schema": schema,
        "splits": reports,
        "checkpoint_paths": {
            "image": str(image_checkpoint), "e4": str(e4_checkpoint),
            "evidence": str(evidence_checkpoint), "support": str(support_checkpoint)},
        "checkpoint_sha256": {
            "image": _sha256(image_checkpoint), "e4": _sha256(e4_checkpoint),
            "evidence": evidence_sha, "support": _sha256(support_checkpoint)},
        "frozen_module_sha256": frozen_sha,
        "dynamic_cache_equivalence": {
            "anchor_output_max_abs_diff": max_output_diff,
            "anchor_loss_max_abs_diff": max_loss_diff,
            "tolerance": tolerance, "passed": True},
        "loss_definition": {
            "name": "official_vecroad_binary_cross_entropy_with_logits",
            "prediction_target_order": "prediction_then_target",
            "spatial_and_recursive_reduction": "sum_per_sample",
            "batch_reduction": "mean",
            "anchor_lowrs_weight": 1.0,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_stage3fa_manifest(cache_dir, manifest)
    with (cache_dir / "build_report.json").open("w", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2, sort_keys=True); output.write("\n")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/stage3fa_seed20260724.yml"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    result = build_cache(_parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
