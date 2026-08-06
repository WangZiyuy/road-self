"""Evaluate Stage 3F-A without changing VecRoad graph growth."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.trajectory_evidence_encoder import TrajectoryEvidenceEncoder  # noqa: E402
from model.trajectory_support_head import TrajectorySupportHead  # noqa: E402
from train_branch_aux import (  # noqa: E402
    _build_auxiliary_modules, _load_config, _load_frozen_rpnet,
    _move_nested, _stage_fuse_for_batch,
)
from train_stage3fa_anchor_fusion import (  # noqa: E402
    _forward, _loss, _module_sha256, _move, _resolve, build_fusion,
)
from utils.stage3c_branch_dataset import Stage3CBranchDataset  # noqa: E402
from utils.stage3c_checkpoint import load_stage3c_checkpoint  # noqa: E402
from utils.stage3d_checkpoint import load_stage3d_support_checkpoint  # noqa: E402
from utils.stage3e0_checkpoint import load_stage3e0_checkpoint  # noqa: E402
from utils.stage3fa_anchor_cache import (  # noqa: E402
    Stage3FAAnchorDataset, stage3fa_collate,
)
from utils.stage3fa_checkpoint import load_stage3fa_checkpoint  # noqa: E402
from utils.stage3fa_metrics import (  # noqa: E402
    PixelHistogramMetrics, aggregate_localization, decode_immediate_nodes,
    localization_record,
)
from utils.trajectory_evidence_robustness import (  # noqa: E402
    global_wrong_sample_donor_indices,
)


MODES = (
    "original_anchor", "fused_initialization", "fused_full_trajectory",
    "fused_no_trajectory", "fused_wrong_sample_trajectory",
    "fused_retain_25",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for block in iter(lambda: input_file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(_plain(value), output, indent=2, sort_keys=True)
        output.write("\n")


def _all_evidence(dataset: Stage3FAAnchorDataset, cyclic_shift: int):
    sample_ids, evidence, available = [], [], []
    for index in range(len(dataset)):
        sample = dataset[index]
        sample_ids.append(sample["sample_id"])
        evidence.append(sample["trajectory_evidence"])
        available.append(sample["trajectory_available"])
    sample_ids = torch.stack(sample_ids).long()
    evidence = torch.stack(evidence).float()
    available = torch.stack(available).bool()
    donor = global_wrong_sample_donor_indices(
        sample_ids, cyclic_shift=int(cyclic_shift))
    return {
        int(sample_ids[index]): {
            "evidence": evidence[int(donor[index])],
            "available": available[int(donor[index])],
            "donor_sample_id": int(sample_ids[int(donor[index])]),
        }
        for index in range(len(dataset))
    }


def _group_names(category: int, branch_count: int):
    category_name = {
        0: "other", 1: "ordinary", 2: "t_junction", 3: "multi_branch"
    }[int(category)]
    count_name = (
        "count_0" if branch_count == 0 else
        "count_1" if branch_count == 1 else
        "count_2" if branch_count == 2 else "count_ge3")
    return ("all", category_name, count_name)


def _evaluate_mode(
    *, mode: str, dataset, trained_fusion, initial_fusion,
    anchor_weight, lowrs_weight, cfg, device, wrong_evidence,
    collect_examples: bool = False,
) -> tuple[Dict[str, Any], Dict[int, Any]]:
    if mode not in MODES:
        raise ValueError("unknown Stage 3F-A mode")
    loader = DataLoader(
        dataset, batch_size=int(cfg.STAGE3FA.TRAINING.VAL_BATCH_SIZE),
        shuffle=False, num_workers=0, collate_fn=stage3fa_collate)
    groups = ("all", "ordinary", "t_junction", "multi_branch", "other",
              "count_0", "count_1", "count_2", "count_ge3")
    heads = ("anchor", "anchor_lowrs")
    pixel = {head: {group: PixelHistogramMetrics() for group in groups}
             for head in heads}
    localization = {head: {group: [] for group in groups} for head in heads}
    responses = {head: {group: {"peak": [], "gt": [], "background": []}
                        for group in groups} for head in heads}
    losses = {"anchor_loss": 0.0, "anchor_lowrs_loss": 0.0,
              "anchor_total_loss": 0.0}
    examples: Dict[int, Any] = {}
    sample_count = 0
    original_logit_max_abs_diff = 0.0
    with torch.no_grad():
        for cpu_batch in loader:
            batch = _move(cpu_batch, device)
            if mode == "original_anchor":
                outputs = {"anchor": batch["original_anchor_logits"].float(),
                           "anchor_lowrs": batch["original_anchor_lowrs_logits"].float()}
            else:
                fusion = initial_fusion if mode == "fused_initialization" else trained_fusion
                availability = None
                evidence_key = "trajectory_evidence"
                if mode == "fused_no_trajectory":
                    availability = torch.zeros_like(
                        batch["trajectory_available"], dtype=torch.float32)
                elif mode == "fused_retain_25":
                    evidence_key = "trajectory_evidence_retain25"
                elif mode == "fused_wrong_sample_trajectory":
                    replacement = torch.stack([
                        wrong_evidence[int(value)]["evidence"]
                        for value in batch["sample_id"].detach().cpu().tolist()
                    ]).to(device)
                    availability = torch.stack([
                        wrong_evidence[int(value)]["available"]
                        for value in batch["sample_id"].detach().cpu().tolist()
                    ]).to(device)
                    batch = dict(batch); batch["trajectory_evidence"] = replacement
                outputs = _forward(
                    fusion, batch, anchor_weight, lowrs_weight,
                    availability_override=availability,
                    evidence_key=evidence_key)
            if mode in ("fused_initialization", "fused_no_trajectory"):
                original_logit_max_abs_diff = max(
                    original_logit_max_abs_diff,
                    float((outputs["anchor"] - batch[
                        "original_anchor_logits"].float()).abs().max()),
                    float((outputs["anchor_lowrs"] - batch[
                        "original_anchor_lowrs_logits"].float()).abs().max()))
            batch_losses = _loss(
                outputs["anchor"], outputs["anchor_lowrs"],
                batch["anchor_target"].float(),
                batch["supervision_end_index"])
            count = int(batch["sample_id"].shape[0]); sample_count += count
            for name in losses:
                losses[name] += float(batch_losses[name]) * count
            probabilities = {
                "anchor": torch.sigmoid(outputs["anchor"]).cpu().numpy(),
                "anchor_lowrs": torch.sigmoid(outputs["anchor_lowrs"]).cpu().numpy(),
            }
            target = batch["anchor_target"].cpu().numpy()
            centers = batch["center_xy"].cpu().numpy()
            gt_xy = batch["next_node_xy"].cpu().numpy()
            gt_mask = batch["next_node_mask"].cpu().numpy()
            for row in range(count):
                group_names = _group_names(
                    int(batch["category_id"][row]),
                    int(batch["branch_count"][row]))
                end = int(batch["supervision_end_index"][row])
                for head in heads:
                    probs = probabilities[head][row]
                    record = localization_record(
                        probabilities=probs, center_xy=centers[row],
                        gt_xy=gt_xy[row], gt_mask=gt_mask[row],
                        threshold=float(cfg.STAGE3FA.EVALUATION.ROAD_SEG_THRESHOLD),
                        step_length=float(cfg.TRAIN.STEP_LENGTH),
                        junction_max_region_area=int(
                            cfg.STAGE3FA.EVALUATION.JUNCTION_MAX_REGION_AREA),
                        match_threshold=float(
                            cfg.STAGE3FA.EVALUATION.COORDINATE_MATCH_THRESHOLD_PIXELS),
                        topk_oracle_k=int(
                            cfg.STAGE3FA.EVALUATION.TOPK_ORACLE_K))
                    for group in group_names:
                        pixel[head][group].update(probs[:end], target[row, :end])
                        localization[head][group].append(record)
                        responses[head][group]["peak"].append(float(probs[0].max()))
                        gt_pixels = target[row, 0] >= 0.95
                        bg_pixels = target[row, 0] < 0.01
                        if gt_pixels.any():
                            responses[head][group]["gt"].append(float(probs[0][gt_pixels].mean()))
                        responses[head][group]["background"].append(float(probs[0][bg_pixels].mean()))
                category = int(batch["category_id"][row])
                if collect_examples and category in (1, 2, 3) and category not in examples:
                    examples[category] = {
                        "sample_id": int(batch["sample_id"][row]),
                        "dataset_index": int(batch["dataset_index"][row]),
                        "anchor": probabilities["anchor"][row],
                        "target": target[row], "center_xy": centers[row],
                        "gt_xy": gt_xy[row], "gt_mask": gt_mask[row],
                    }
    result = {"mode": mode, "sample_count": sample_count,
              "original_logit_max_abs_diff": original_logit_max_abs_diff,
              "losses": {name: value / max(sample_count, 1)
                         for name, value in losses.items()}, "heads": {}}
    for head in heads:
        result["heads"][head] = {}
        for group in groups:
            if not localization[head][group]:
                continue
            response = responses[head][group]
            result["heads"][head][group] = {
                **pixel[head][group].compute(),
                **aggregate_localization(localization[head][group]),
                "peak_value_mean": float(np.mean(response["peak"])),
                "gt_peak_response_mean": float(np.mean(response["gt"])) if response["gt"] else None,
                "background_false_positive_response_mean": float(np.mean(response["background"])),
            }
    return result, examples


def _branch_regression_check(cfg, *, rpnet, trajectory_encoder,
                             graph_encoder, branch_decoder, fusion,
                             anchor_dataset, device) -> Dict[str, Any]:
    dataset = Stage3CBranchDataset(_resolve(cfg.STAGE3C.DATASET_DIR), "val")
    cpu_batch = default_collate([dataset[index] for index in range(min(4, len(dataset)))])
    batch = _move_nested(cpu_batch, device)
    with torch.no_grad():
        stage_fuse = _stage_fuse_for_batch(
            rpnet=rpnet, batch=batch, cache=None, device=device)
        trajectory = trajectory_encoder(batch["trajectory_batch"])
        state = graph_encoder(batch["graph_state"])
        before = branch_decoder(
            stage_fuse=stage_fuse, state_token=state,
            fragment_tokens=trajectory["fragment_tokens"],
            fragment_mask=trajectory["fragment_mask"],
            walked_path=batch["walked_path"])
        anchor_cpu = stage3fa_collate([
            anchor_dataset[index] for index in range(min(4, len(anchor_dataset)))])
        anchor = _move(anchor_cpu, device)
        _forward(
            fusion, anchor, rpnet.conv_final.weight,
            rpnet.next_step_final.weight)
        after = branch_decoder(
            stage_fuse=stage_fuse, state_token=state,
            fragment_tokens=trajectory["fragment_tokens"],
            fragment_mask=trajectory["fragment_mask"],
            walked_path=batch["walked_path"])
    differences = {
        key: float((before[key] - after[key]).abs().max())
        for key in ("branch_exist_logits", "branch_offsets_norm", "branch_directions")}
    return {"max_abs_differences": differences,
            "maximum": max(differences.values()),
            "passed": max(differences.values()) <= 1e-6}


def _visualize(output_dir: Path, stage3c_dataset, original, fused,
               cache_dataset, cfg) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = []
    labels = {1: "ordinary", 2: "t_junction", 3: "multi_branch"}
    for category in (1, 2, 3):
        if category not in original or category not in fused:
            continue
        item = fused[category]
        raw = stage3c_dataset[item["dataset_index"]]
        cache = cache_dataset[item["dataset_index"]]
        image = raw["aerial_image"].permute(1, 2, 0).numpy()
        figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        axes[0].imshow(image)
        mask = raw["trajectory_batch"]["fragment_mask"].numpy().astype(bool)
        points = raw["trajectory_batch"]["traj_xy_norm"].numpy()
        point_mask = raw["trajectory_batch"]["point_mask"].numpy().astype(bool)
        attention = cache["evidence_attention"].numpy()[0]
        for index in np.flatnonzero(mask):
            xy = points[index, point_mask[index]] * 128.0 + 128.0
            if len(xy):
                axes[0].plot(xy[:, 0], xy[:, 1], linewidth=0.6,
                             alpha=0.15 + 0.85 * float(attention[index] / max(attention.max(), 1e-8)),
                             color="cyan")
        axes[0].set_title("trajectory fragments + evidence attention")
        for axis, value, title in (
                (axes[1], original[category]["anchor"][0], "original anchor"),
                (axes[2], fused[category]["anchor"][0], "fused anchor")):
            axis.imshow(value.T, origin="lower", cmap="magma", vmin=0, vmax=1)
            gt = item["gt_xy"][item["gt_mask"]] - item["center_xy"] + 128.0
            if len(gt):
                axis.scatter(gt[:, 0], gt[:, 1], marker="x", c="cyan", s=55,
                             label="GT")
            predictions = decode_immediate_nodes(
                value[None], item["center_xy"],
                threshold=float(cfg.STAGE3FA.EVALUATION.ROAD_SEG_THRESHOLD),
                step_length=float(cfg.TRAIN.STEP_LENGTH),
                junction_max_region_area=int(
                    cfg.STAGE3FA.EVALUATION.JUNCTION_MAX_REGION_AREA))
            if predictions:
                predicted = np.stack(predictions) - item["center_xy"] + 128.0
                axis.scatter(predicted[:, 0], predicted[:, 1], marker="o",
                             facecolors="none", edgecolors="lime", s=55,
                             label="decoded")
            if len(gt) or predictions:
                axis.legend(loc="upper right", fontsize=7)
            axis.set_title(title)
        for axis in axes:
            axis.set_xlim(0, 256); axis.set_ylim(256, 0)
        figure.suptitle("{} sample {}".format(labels[category], item["sample_id"]))
        path = output_dir / "visualizations" / (labels[category] + ".png")
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(figure); paths.append(str(path.resolve()))
    return paths


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = _load_config(args.config)
    device = torch.device(args.device or cfg.STAGE3C.DEVICE)
    resident_setting = cfg.STAGE3FA.TRAINING.RESIDENT_CACHE_SHARDS
    cache_shards = (
        None if str(resident_setting).lower() == "all"
        else int(resident_setting))
    cache_dataset = Stage3FAAnchorDataset(
        _resolve(cfg.STAGE3FA.CACHE_DIR), "val",
        cache_shards=cache_shards)
    manifest = cache_dataset.manifest
    rpnet, _ = _load_frozen_rpnet(
        cfg, Path(manifest["checkpoint_paths"]["image"]), device)
    trajectory_encoder, graph_encoder, branch_decoder = _build_auxiliary_modules(cfg, device)
    load_stage3c_checkpoint(
        Path(manifest["checkpoint_paths"]["e4"]),
        trajectory_encoder=trajectory_encoder,
        graph_state_encoder=graph_encoder, branch_decoder=branch_decoder,
        map_location="cpu")
    evidence_encoder = TrajectoryEvidenceEncoder(
        hidden_dim=128, num_evidence_tokens=1, num_heads=4,
        dropout=0.1, aggregation_mode="latent_attention").to(device)
    load_stage3e0_checkpoint(
        Path(manifest["checkpoint_paths"]["evidence"]),
        evidence_encoder=evidence_encoder, map_location="cpu",
        expected_e4_sha256=manifest["checkpoint_sha256"]["e4"])
    support_head = TrajectorySupportHead(hidden_dim=128).to(device)
    load_stage3d_support_checkpoint(
        Path(manifest["checkpoint_paths"]["support"]),
        support_head=support_head, map_location="cpu")
    modules = {"rpnet": rpnet, "trajectory_encoder": trajectory_encoder,
               "graph_state_encoder": graph_encoder,
               "branch_decoder": branch_decoder,
               "trajectory_evidence_encoder": evidence_encoder,
               "support_head": support_head}
    for module in modules.values():
        module.eval().requires_grad_(False)
    frozen_hashes = {name: _module_sha256(module) for name, module in modules.items()}
    if frozen_hashes != manifest["frozen_module_sha256"]:
        raise RuntimeError("frozen module SHA differs from cache manifest")

    trained = build_fusion(cfg, device)
    checkpoint = args.checkpoint or (
        _resolve(cfg.STAGE3FA.OUTPUT_DIR) / "checkpoints" /
        "stage3fa_anchor_fusion.best.pth.tar")
    payload = load_stage3fa_checkpoint(
        checkpoint, fusion=trained, map_location=device)
    trained.eval()
    initial = build_fusion(cfg, device).eval()
    wrong = _all_evidence(
        cache_dataset,
        int(cfg.STAGE3FA.EVALUATION.WRONG_SAMPLE_CYCLIC_SHIFT))
    results, example_sets = {}, {}
    for mode in MODES:
        result, examples = _evaluate_mode(
            mode=mode, dataset=cache_dataset, trained_fusion=trained,
            initial_fusion=initial, anchor_weight=rpnet.conv_final.weight,
            lowrs_weight=rpnet.next_step_final.weight, cfg=cfg,
            device=device, wrong_evidence=wrong,
            collect_examples=mode in ("original_anchor", "fused_full_trajectory"))
        results[mode] = result; example_sets[mode] = examples
        print("{} AP={:.6f} hit5={:.6f} error={}".format(
            mode, result["heads"]["anchor"]["all"]["pixel_ap"],
            result["heads"]["anchor"]["all"]["hit_at_5_px"],
            result["heads"]["anchor"]["all"]["top1_endpoint_error_mean"]), flush=True)
    strict_tolerance = float(cfg.STAGE3FA.ACCEPTANCE.STRICT_TOLERANCE)
    strict = {}
    for mode in ("fused_initialization", "fused_no_trajectory"):
        differences = []
        for head in ("anchor", "anchor_lowrs"):
            original_metrics = results["original_anchor"]["heads"][head]["all"]
            mode_metrics = results[mode]["heads"][head]["all"]
            for name in ("pixel_ap", "pixel_auroc", "peak_value_mean",
                         "gt_peak_response_mean", "background_false_positive_response_mean"):
                differences.append(abs(float(original_metrics[name]) - float(mode_metrics[name])))
        logit_difference = results[mode]["original_logit_max_abs_diff"]
        strict[mode] = {"logit_max_abs_diff": logit_difference,
                        "metric_max_abs_diff": max(differences),
                        "tolerance": strict_tolerance,
                        "passed": max(logit_difference, max(differences)) <= strict_tolerance}
    branch_regression = _branch_regression_check(
        cfg, rpnet=rpnet, trajectory_encoder=trajectory_encoder,
        graph_encoder=graph_encoder, branch_decoder=branch_decoder,
        fusion=trained, anchor_dataset=cache_dataset, device=device)
    if not branch_regression["passed"]:
        raise RuntimeError("branch outputs changed in Stage 3F-A")
    stage3c_val = Stage3CBranchDataset(_resolve(cfg.STAGE3C.DATASET_DIR), "val")
    output_dir = _resolve(cfg.STAGE3FA.OUTPUT_DIR)
    visualizations = _visualize(
        output_dir, stage3c_val, example_sets["original_anchor"],
        example_sets["fused_full_trajectory"], cache_dataset, cfg)
    available_count = sum(
        int(bool(cache_dataset[index]["trajectory_available"]))
        for index in range(len(cache_dataset)))
    report = {
        "stage": "3F-A", "seed": int(cfg.STAGE3C.SEED),
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_epoch": int(payload["epoch"]),
        "modes": results, "strict_equivalence": strict,
        "branch_regression": branch_regression,
        "frozen_module_sha256": frozen_hashes,
        "frozen_sha_unchanged_from_cache": True,
        "visualizations": visualizations,
        "teacher_forced_anchor_validation": True,
        "closed_loop_road_graph_extraction": False,
        "path_push_reads_fusion_or_branch_output": False,
        "trajectory_availability": {
            "sample_count": len(cache_dataset),
            "available_count": available_count,
            "natural_empty_count": len(cache_dataset) - available_count,
            "no_trajectory_mode_is_synthetic_ablation": True,
        },
    }
    _write(output_dir / "evaluation.json", report)
    _write(output_dir / "anchor_metrics.json", {
        "seed": report["seed"], "modes": results})
    _write(output_dir / "robustness_results.json", {
        "full": results["fused_full_trajectory"],
        "no_trajectory": results["fused_no_trajectory"],
        "wrong_sample": results["fused_wrong_sample_trajectory"],
        "retain_25": results["fused_retain_25"]})
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(_plain(evaluate(_parse_args())), indent=2, sort_keys=True))
