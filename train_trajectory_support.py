"""Train and validate the frozen-E4 Stage 3D-A trajectory support head.

This is a side experiment.  Support scores are never fed back into the E4
trajectory attention, branch outputs, RPNet anchor head, or Path.push.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from easydict import EasyDict
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.trajectory_support_head import (  # noqa: E402
    TrajectorySupportHead,
    trajectory_support_bce_loss,
)
from train_branch_aux import (  # noqa: E402
    _build_auxiliary_modules,
    _build_branch_criterion,
    _load_config,
    _load_frozen_rpnet,
    _move_nested,
    _resolve_device,
    _set_module_mode,
    _set_seed,
    _stage_fuse_for_batch,
)
from utils.stage3c_branch_dataset import (  # noqa: E402
    Stage3CBranchDataset,
)
from utils.stage3c_checkpoint import (  # noqa: E402
    load_stage3c_checkpoint,
)
from utils.stage3d_checkpoint import (  # noqa: E402
    build_stage3d_support_checkpoint_payload,
    load_stage3d_support_checkpoint,
    save_stage3d_support_checkpoint,
)
from utils.trajectory_support_metrics import (  # noqa: E402
    TrajectorySupportMetricAccumulator,
    support_label_diagnostics,
)
from utils.trajectory_support_features import (  # noqa: E402
    build_pre_trajectory_branch_tokens,
)
from utils.trajectory_support_targets import (  # noqa: E402
    build_trajectory_support_targets,
)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(
            _plain(value),
            output_file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        output_file.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while True:
            block = input_file.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _target_parameters(cfg: EasyDict) -> Dict[str, float]:
    target = cfg.STAGE3D.SUPPORT_TARGET
    return {
        "window_size": float(cfg.TRAIN.WINDOW_SIZE),
        "step_length": float(cfg.TRAIN.STEP_LENGTH),
        "distance_sigma_pixels": float(
            target.DISTANCE_SIGMA_PIXELS),
        "axis_gamma": float(target.AXIS_GAMMA),
        "positive_threshold": float(target.POSITIVE_THRESHOLD),
        "epsilon": float(target.EPSILON),
    }


def _label_report_for_dataset(
    dataset: Stage3CBranchDataset,
    cfg: EasyDict,
    *,
    batch_size: int,
) -> Dict[str, Any]:
    chunks = {
        "support_positive_mask": [],
        "support_valid": [],
        "branch_mask": [],
        "segment_only_positive_mask": [],
    }
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    started_at = time.perf_counter()
    for batch in loader:
        support = build_trajectory_support_targets(
            batch["trajectory_batch"],
            batch["branch_targets"],
            **_target_parameters(cfg),
        )
        chunks["support_positive_mask"].append(
            support["support_positive_mask"].numpy())
        chunks["support_valid"].append(
            support["support_valid"].numpy())
        chunks["branch_mask"].append(
            batch["branch_targets"]["branch_mask"].numpy())
        chunks["segment_only_positive_mask"].append(
            support["segment_only_positive_mask"].numpy())
    report = support_label_diagnostics(**chunks)
    report["sample_count"] = len(dataset)
    report["elapsed_seconds"] = float(
        time.perf_counter() - started_at)
    return report


def run_label_diagnostics(
    *,
    train_dataset: Stage3CBranchDataset,
    val_dataset: Stage3CBranchDataset,
    cfg: EasyDict,
) -> Dict[str, Any]:
    batch_size = int(cfg.STAGE3D.TRAINING.CACHE_BATCH_SIZE)
    train = _label_report_for_dataset(
        train_dataset, cfg, batch_size=batch_size)
    validation = _label_report_for_dataset(
        val_dataset, cfg, batch_size=batch_size)

    # Recompute the combined aggregate from branch-level counts.  The exact
    # grouping rates are weighted by their branch counts, not sample counts.
    combined_groups = {}
    for group_name in ("0", "1", "2", ">=3", ">=2"):
        members = [
            train["by_gt_branch_count"][group_name],
            validation["by_gt_branch_count"][group_name],
        ]
        branch_count = sum(
            int(member["branch_count"]) for member in members)
        available = sum(
            int(member["available_branch_count"]) for member in members)
        combined_groups[group_name] = {
            "sample_count": sum(
                int(member["sample_count"]) for member in members),
            "branch_count": branch_count,
            "available_branch_count": available,
            "support_available_rate": (
                float(available) / branch_count
                if branch_count else None),
        }
    total_branches = (
        int(train["gt_branch_count"])
        + int(validation["gt_branch_count"]))
    available_branches = (
        int(train["support_available_branch_count"])
        + int(validation["support_available_branch_count"]))
    combined = {
        "sample_count": len(train_dataset) + len(val_dataset),
        "gt_branch_count": total_branches,
        "support_available_branch_count": available_branches,
        "support_available_rate": (
            float(available_branches) / max(total_branches, 1)),
        "bounded_64_branch_support_hit_rate": (
            float(available_branches) / max(total_branches, 1)),
        # Compatibility alias for the already published Stage 3D-A report.
        # This is a branch-level hit rate, not candidate-level recall.
        "bounded_64_oracle_support_recall": (
            float(available_branches) / max(total_branches, 1)),
        "by_gt_branch_count": combined_groups,
        "positive_fragment_pair_count": (
            int(train["positive_fragment_pair_count"])
            + int(validation["positive_fragment_pair_count"])),
        "segment_only_positive_pair_count": (
            int(train["segment_only_positive_pair_count"])
            + int(validation["segment_only_positive_pair_count"])),
    }
    positive_count = combined["positive_fragment_pair_count"]
    combined["segment_only_positive_ratio"] = (
        float(combined["segment_only_positive_pair_count"])
        / max(positive_count, 1))
    threshold = float(
        cfg.STAGE3D.SUPPORT_TARGET.MIN_MULTIBRANCH_AVAILABLE_RATE)
    multibranch_rate = combined_groups[">=2"][
        "support_available_rate"]
    passed = bool(
        multibranch_rate is not None
        and multibranch_rate >= threshold)
    return {
        "parameters": _target_parameters(cfg),
        "train": train,
        "validation": validation,
        "combined": combined,
        "gate": {
            "minimum_multibranch_support_available_rate": threshold,
            "observed_multibranch_support_available_rate":
                multibranch_rate,
            "passed": passed,
            "training_allowed": passed,
        },
    }


class FrozenSupportDataset(Dataset):
    """Small in-memory cache of frozen E4 tokens and geometry labels."""

    def __init__(self, tensors: Mapping[str, torch.Tensor]) -> None:
        self.tensors = dict(tensors)
        lengths = {int(value.shape[0]) for value in self.tensors.values()}
        if len(lengths) != 1:
            raise ValueError("frozen support cache has inconsistent lengths")
        self.length = lengths.pop() if lengths else 0

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            key: value[index]
            for key, value in self.tensors.items()
        }


def _freeze_e4_modules(
    modules: Sequence[torch.nn.Module],
) -> None:
    _set_module_mode(modules, False)
    for module in modules:
        module.requires_grad_(False)
    if any(
            parameter.requires_grad
            for module in modules
            for parameter in module.parameters()):
        raise RuntimeError("all E4 auxiliary modules must be frozen")


def build_frozen_support_cache(
    *,
    dataset: Stage3CBranchDataset,
    rpnet: torch.nn.Module,
    modules: Sequence[torch.nn.Module],
    criterion: torch.nn.Module,
    cfg: EasyDict,
    device: torch.device,
) -> Tuple[FrozenSupportDataset, Dict[str, Any]]:
    _freeze_e4_modules(modules)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.STAGE3D.TRAINING.CACHE_BATCH_SIZE),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    trajectory_encoder, graph_state_encoder, branch_decoder = modules
    chunks: Dict[str, List[torch.Tensor]] = {}
    started_at = time.perf_counter()
    with torch.no_grad():
        for cpu_batch in loader:
            batch = _move_nested(cpu_batch, device)
            stage_fuse = _stage_fuse_for_batch(
                rpnet=rpnet,
                batch=batch,
                cache=None,
                device=device,
            )
            trajectory_output = trajectory_encoder(
                batch["trajectory_batch"])
            state_token = graph_state_encoder(batch["graph_state"])
            branch_output = branch_decoder(
                stage_fuse=stage_fuse,
                state_token=state_token,
                fragment_tokens=trajectory_output["fragment_tokens"],
                fragment_mask=trajectory_output["fragment_mask"],
                walked_path=batch["walked_path"],
                return_attention=True,
                return_debug_states=True,
            )
            branch_losses = criterion(
                branch_output, batch["branch_targets"])
            support = build_trajectory_support_targets(
                batch["trajectory_batch"],
                batch["branch_targets"],
                **_target_parameters(cfg),
            )
            batch_size, query_count = branch_output[
                "branch_exist_logits"].shape
            matched_target_indices = torch.full(
                (batch_size, query_count),
                -1,
                dtype=torch.long,
                device=device,
            )
            for batch_index, (
                    query_indices, target_indices) in enumerate(
                        branch_losses["matches"]):
                matched_target_indices[
                    batch_index, query_indices] = target_indices
            values = {
                "branch_tokens": branch_output["branch_tokens"],
                "graph_conditioned_queries": branch_output[
                    "debug_graph_conditioned_queries"],
                "image_cross_attention_context": branch_output[
                    "debug_image_cross_attention_output"],
                "graph_state_contribution": branch_output[
                    "debug_graph_state_contribution"],
                "pre_trajectory_branch_tokens": (
                    build_pre_trajectory_branch_tokens(
                        branch_output[
                            "debug_graph_conditioned_queries"],
                        branch_output[
                            "debug_image_cross_attention_output"],
                    )
                ),
                "fragment_tokens": trajectory_output["fragment_tokens"],
                "fragment_mask": trajectory_output["fragment_mask"],
                "trajectory_attention_weights": branch_output[
                    "trajectory_attention_weights"],
                "support_targets": support["support_targets"],
                "support_positive_mask": support[
                    "support_positive_mask"],
                "support_valid": support["support_valid"],
                "segment_only_positive_mask": support[
                    "segment_only_positive_mask"],
                "segment_only": batch["trajectory_batch"]["segment_only"],
                "matched_target_indices": matched_target_indices,
                "sample_ids": batch["metadata"]["dataset_index"],
                "branch_count": batch["branch_targets"]["branch_count"],
                "branch_mask": batch["branch_targets"]["branch_mask"],
                "branch_offsets_norm": batch[
                    "branch_targets"]["branch_offsets_norm"],
                "branch_directions": batch[
                    "branch_targets"]["branch_directions"],
                "traj_xy_norm": batch["trajectory_batch"]["traj_xy_norm"],
                "point_mask": batch["trajectory_batch"]["point_mask"],
            }
            for key, value in values.items():
                chunks.setdefault(key, []).append(value.detach().cpu())
    tensors = {
        key: torch.cat(values, dim=0)
        for key, values in chunks.items()
    }
    cache = FrozenSupportDataset(tensors)
    finite = all(
        bool(torch.isfinite(value).all())
        for value in tensors.values()
        if value.is_floating_point()
    )
    return cache, {
        "sample_count": len(cache),
        "elapsed_seconds": float(time.perf_counter() - started_at),
        "finite": finite,
        "e4_modules_frozen": True,
        "branch_predictions_feed_path_push": False,
        "support_changes_trajectory_attention": False,
        "tensor_shapes": {
            key: list(value.shape)
            for key, value in tensors.items()
        },
        "size_bytes": int(sum(
            value.numel() * value.element_size()
            for value in tensors.values()
        )),
    }


def _matches_from_batch(
    matched_target_indices: torch.Tensor,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    matches = []
    for row in matched_target_indices:
        query_indices = torch.nonzero(
            row >= 0, as_tuple=False).flatten()
        target_indices = row.index_select(0, query_indices)
        matches.append((query_indices, target_indices))
    return matches


def _soft_target_entropy_floor(
    batch: Mapping[str, torch.Tensor],
    matches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
) -> float:
    """Return the minimum achievable BCE for the supervised soft labels."""

    entropy_sum = 0.0
    pair_count = 0
    targets = batch["support_targets"]
    support_valid = batch["support_valid"].to(dtype=torch.bool)
    fragment_mask = batch["fragment_mask"].to(dtype=torch.bool)
    for batch_index, (
            prediction_indices, target_indices) in enumerate(matches):
        del prediction_indices
        for target_index in target_indices.tolist():
            if not bool(support_valid[batch_index, target_index]):
                continue
            selected = targets[
                batch_index, target_index, fragment_mask[batch_index]]
            if selected.numel() == 0:
                continue
            selected = selected.clamp(0.0, 1.0)
            entropy = -(
                torch.special.xlogy(selected, selected)
                + torch.special.xlogy(1.0 - selected, 1.0 - selected)
            )
            entropy_sum += float(entropy.sum().detach().cpu())
            pair_count += int(selected.numel())
    return entropy_sum / max(pair_count, 1)


def _reducible_loss_reduction(
    initial_loss: float,
    achieved_loss: float,
    entropy_floor: float,
) -> float:
    reducible = float(initial_loss) - float(entropy_floor)
    if reducible <= 0.0:
        return 1.0 if achieved_loss <= initial_loss else 0.0
    remaining = max(float(achieved_loss) - float(entropy_floor), 0.0)
    return float(1.0 - remaining / reducible)


def _move_flat_batch(
    batch: Mapping[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device=device, non_blocking=True)
        for key, value in batch.items()
    }


def evaluate_support_head(
    *,
    support_head: TrajectorySupportHead,
    dataset: Dataset,
    cfg: EasyDict,
    device: torch.device,
    batch_size: int,
) -> Dict[str, Any]:
    support_head.eval()
    metrics = TrajectorySupportMetricAccumulator(
        recall_ks=tuple(
            int(value)
            for value in cfg.STAGE3D.EVALUATION.RECALL_KS),
        jaccard_k=int(
            cfg.STAGE3D.EVALUATION.TOP_K_JACCARD),
    )
    loss_sum = 0.0
    pair_count = 0
    started_at = time.perf_counter()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    with torch.no_grad():
        for cpu_batch in loader:
            batch = _move_flat_batch(cpu_batch, device)
            logits = support_head(
                batch["branch_tokens"],
                batch["fragment_tokens"],
                batch["fragment_mask"],
            )
            matches = _matches_from_batch(
                batch["matched_target_indices"])
            losses = trajectory_support_bce_loss(
                logits,
                batch["support_targets"],
                batch["support_valid"],
                batch["fragment_mask"],
                matches,
            )
            pairs = int(losses["supervised_pair_count"])
            loss_sum += float(losses["loss"]) * pairs
            pair_count += pairs
            metrics.update(
                support_logits=logits,
                attention_weights=batch[
                    "trajectory_attention_weights"],
                support_targets=batch["support_targets"],
                support_positive_mask=batch[
                    "support_positive_mask"],
                support_valid=batch["support_valid"],
                fragment_mask=batch["fragment_mask"],
                segment_only=batch["segment_only"],
                matches=matches,
                sample_ids=batch["sample_ids"],
            )
    result = metrics.compute()
    result["loss"] = loss_sum / max(pair_count, 1)
    result["supervised_pair_count"] = pair_count
    result["elapsed_seconds"] = float(
        time.perf_counter() - started_at)
    return result


def _checkpoint_payload(
    *,
    support_head: TrajectorySupportHead,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    e4_checkpoint: Path,
    e4_sha256: str,
    cfg: EasyDict,
    metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    return build_stage3d_support_checkpoint_payload(
        support_head=support_head,
        optimizer=optimizer,
        epoch=epoch,
        e4_checkpoint=str(e4_checkpoint.resolve()),
        e4_checkpoint_sha256=e4_sha256,
        config_snapshot=_plain(cfg),
        metrics=metrics,
    )


def run_support_overfit_sanity(
    *,
    cache: FrozenSupportDataset,
    cfg: EasyDict,
    device: torch.device,
    output_dir: Path,
    e4_checkpoint: Path,
    e4_sha256: str,
) -> Dict[str, Any]:
    sanity_cfg = cfg.STAGE3D.SANITY
    eligible = torch.nonzero(
        cache.tensors["support_valid"].any(dim=1),
        as_tuple=False,
    ).flatten()
    sample_count = min(int(sanity_cfg.SAMPLE_COUNT), eligible.numel())
    if sample_count == 0:
        raise RuntimeError("no support-valid samples for sanity check")
    selected = eligible[:sample_count]
    subset = FrozenSupportDataset({
        key: value.index_select(0, selected)
        for key, value in cache.tensors.items()
    })
    model_cfg = cfg.STAGE3D.MODEL
    hidden_dim = int(cfg.STAGE3C.MODEL.HIDDEN_DIM)
    support_head = TrajectorySupportHead(
        hidden_dim=hidden_dim,
        projection_dim=int(model_cfg.PROJECTION_DIM),
    ).to(device=device)
    optimizer = torch.optim.Adam(
        support_head.parameters(),
        lr=float(sanity_cfg.LEARNING_RATE),
    )
    batch = _move_flat_batch({
        key: value
        for key, value in subset.tensors.items()
    }, device)
    matches = _matches_from_batch(
        batch["matched_target_indices"])
    initial = evaluate_support_head(
        support_head=support_head,
        dataset=subset,
        cfg=cfg,
        device=device,
        batch_size=sample_count,
    )
    entropy_floor = _soft_target_entropy_floor(batch, matches)
    history = [{"epoch": 0, "metrics": initial}]
    best = copy.deepcopy(initial)
    best_epoch = 0
    best_state = copy.deepcopy(support_head.state_dict())
    started_at = time.perf_counter()
    for epoch in range(1, int(sanity_cfg.MAX_EPOCHS) + 1):
        support_head.train()
        optimizer.zero_grad(set_to_none=True)
        logits = support_head(
            batch["branch_tokens"],
            batch["fragment_tokens"],
            batch["fragment_mask"],
        )
        losses = trajectory_support_bce_loss(
            logits,
            batch["support_targets"],
            batch["support_valid"],
            batch["fragment_mask"],
            matches,
        )
        losses["loss"].backward()
        optimizer.step()
        if (
                epoch % int(sanity_cfg.EVAL_EVERY_EPOCHS) == 0
                or epoch == int(sanity_cfg.MAX_EPOCHS)):
            evaluation = evaluate_support_head(
                support_head=support_head,
                dataset=subset,
                cfg=cfg,
                device=device,
                batch_size=sample_count,
            )
            history.append({"epoch": epoch, "metrics": evaluation})
            if (
                    evaluation["support_ap"] > best["support_ap"]
                    or (
                        evaluation["support_ap"] == best["support_ap"]
                        and evaluation["loss"] < best["loss"]
                    )):
                best = copy.deepcopy(evaluation)
                best_epoch = epoch
                best_state = copy.deepcopy(support_head.state_dict())
    final = history[-1]["metrics"]
    loss_reduction = (
        1.0 - float(best["loss"]) / max(float(initial["loss"]), 1e-12)
    )
    reducible_loss_reduction = _reducible_loss_reduction(
        initial["loss"], best["loss"], entropy_floor)
    passed = bool(
        reducible_loss_reduction >= float(
            sanity_cfg.MIN_REDUCIBLE_LOSS_REDUCTION)
        and float(best["support_ap"]) >= float(
            sanity_cfg.MIN_SUPPORT_AP)
    )
    support_head.load_state_dict(best_state, strict=True)
    best_payload = _checkpoint_payload(
        support_head=support_head,
        optimizer=optimizer,
        epoch=best_epoch,
        e4_checkpoint=e4_checkpoint,
        e4_sha256=e4_sha256,
        cfg=cfg,
        metrics=best,
    )
    save_stage3d_support_checkpoint(
        output_dir / "checkpoints" / "support_sanity.best.pth.tar",
        best_payload,
    )
    report = {
        "sample_count": sample_count,
        "selected_dataset_indices": selected.tolist(),
        "initial": initial,
        "best": best,
        "best_epoch": best_epoch,
        "final": final,
        "final_epoch": int(sanity_cfg.MAX_EPOCHS),
        "loss_reduction": loss_reduction,
        "soft_target_entropy_floor": entropy_floor,
        "reducible_loss_reduction": reducible_loss_reduction,
        "minimum_reducible_loss_reduction": float(
            sanity_cfg.MIN_REDUCIBLE_LOSS_REDUCTION),
        "passed": passed,
        "history": history,
        "elapsed_seconds": float(time.perf_counter() - started_at),
    }
    _write_json(output_dir / "support_sanity_report.json", report)
    return report


def run_formal_support_training(
    *,
    train_cache: FrozenSupportDataset,
    val_cache: FrozenSupportDataset,
    cfg: EasyDict,
    device: torch.device,
    output_dir: Path,
    e4_checkpoint: Path,
    e4_sha256: str,
) -> Tuple[TrajectorySupportHead, Dict[str, Any]]:
    training_cfg = cfg.STAGE3D.TRAINING
    hidden_dim = int(cfg.STAGE3C.MODEL.HIDDEN_DIM)
    support_head = TrajectorySupportHead(
        hidden_dim=hidden_dim,
        projection_dim=int(
            cfg.STAGE3D.MODEL.PROJECTION_DIM),
    ).to(device=device)
    optimizer = torch.optim.Adam(
        support_head.parameters(),
        lr=float(training_cfg.LEARNING_RATE),
        weight_decay=float(training_cfg.WEIGHT_DECAY),
    )
    generator = torch.Generator().manual_seed(
        int(cfg.STAGE3C.SEED))
    loader = DataLoader(
        train_cache,
        batch_size=int(training_cfg.BATCH_SIZE),
        shuffle=True,
        generator=generator,
        num_workers=int(training_cfg.NUM_WORKERS),
        pin_memory=device.type == "cuda",
    )
    history = []
    best_ap = -1.0
    best_epoch = -1
    checkpoint_dir = output_dir / "checkpoints"
    started_at = time.perf_counter()
    for epoch in range(1, int(training_cfg.EPOCHS) + 1):
        support_head.train()
        loss_sum = 0.0
        pair_count = 0
        for cpu_batch in loader:
            batch = _move_flat_batch(cpu_batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = support_head(
                batch["branch_tokens"],
                batch["fragment_tokens"],
                batch["fragment_mask"],
            )
            matches = _matches_from_batch(
                batch["matched_target_indices"])
            losses = trajectory_support_bce_loss(
                logits,
                batch["support_targets"],
                batch["support_valid"],
                batch["fragment_mask"],
                matches,
            )
            pairs = int(losses["supervised_pair_count"])
            if pairs == 0:
                continue
            losses["loss"].backward()
            optimizer.step()
            loss_sum += float(losses["loss"].detach()) * pairs
            pair_count += pairs
        validation = evaluate_support_head(
            support_head=support_head,
            dataset=val_cache,
            cfg=cfg,
            device=device,
            batch_size=int(training_cfg.VAL_BATCH_SIZE),
        )
        record = {
            "epoch": epoch,
            "training_loss": loss_sum / max(pair_count, 1),
            "training_supervised_pair_count": pair_count,
            "validation": validation,
        }
        history.append(record)
        payload = _checkpoint_payload(
            support_head=support_head,
            optimizer=optimizer,
            epoch=epoch,
            e4_checkpoint=e4_checkpoint,
            e4_sha256=e4_sha256,
            cfg=cfg,
            metrics=validation,
        )
        save_stage3d_support_checkpoint(
            checkpoint_dir / "stage3d_support.latest.pth.tar",
            payload,
        )
        if float(validation["support_ap"]) > best_ap:
            best_ap = float(validation["support_ap"])
            best_epoch = epoch
            save_stage3d_support_checkpoint(
                checkpoint_dir / "stage3d_support.best.pth.tar",
                payload,
            )
        print(
            "support epoch {:03d}: train_loss={:.6f} "
            "val_loss={:.6f} AP={:.4f} attention_AP={:.4f}".format(
                epoch,
                record["training_loss"],
                validation["loss"],
                validation["support_ap"],
                validation["attention_support_ap"],
            ),
            flush=True,
        )
    best_path = checkpoint_dir / "stage3d_support.best.pth.tar"
    best_optimizer = torch.optim.Adam(support_head.parameters(), lr=1e-3)
    best_payload = load_stage3d_support_checkpoint(
        best_path,
        support_head=support_head,
        optimizer=best_optimizer,
        map_location=device,
    )
    best_validation = evaluate_support_head(
        support_head=support_head,
        dataset=val_cache,
        cfg=cfg,
        device=device,
        batch_size=int(training_cfg.VAL_BATCH_SIZE),
    )
    report = {
        "train_sample_count": len(train_cache),
        "validation_sample_count": len(val_cache),
        "epochs": int(training_cfg.EPOCHS),
        "best_epoch": int(best_payload["epoch"]),
        "best_validation": best_validation,
        "history": history,
        "elapsed_seconds": float(time.perf_counter() - started_at),
        "only_support_head_trainable": True,
        "trajectory_attention_unchanged": True,
        "branch_outputs_unchanged": True,
        "branch_predictions_feed_path_push": False,
        "best_checkpoint": str(best_path.resolve()),
    }
    _write_json(output_dir / "support_training_report.json", report)
    return support_head, report


def render_support_visualizations(
    *,
    support_head: TrajectorySupportHead,
    cache: FrozenSupportDataset,
    cfg: EasyDict,
    device: torch.device,
    output_dir: Path,
) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    branch_counts = cache.tensors["branch_count"].numpy()
    categories = {
        "ordinary": np.flatnonzero(branch_counts == 1),
        "t_junction": np.flatnonzero(branch_counts == 2),
        "multi_branch": np.flatnonzero(branch_counts >= 3),
    }
    paths = []
    cases_per_type = int(
        cfg.STAGE3D.EVALUATION.VISUALIZATION_CASES_PER_TYPE)
    top_k = int(cfg.STAGE3D.EVALUATION.TOP_K_JACCARD)
    half_window = float(cfg.TRAIN.WINDOW_SIZE) / 2.0
    support_head.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    for category, indices in categories.items():
        for case_number, dataset_index in enumerate(
                indices[:cases_per_type]):
            sample = {
                key: value[dataset_index:dataset_index + 1].to(
                    device=device)
                for key, value in cache.tensors.items()
            }
            with torch.no_grad():
                logits = support_head(
                    sample["branch_tokens"],
                    sample["fragment_tokens"],
                    sample["fragment_mask"],
                )
            probabilities = torch.sigmoid(logits)[0].cpu().numpy()
            matched = sample["matched_target_indices"][0].cpu().numpy()
            target_to_query = {
                int(target): query
                for query, target in enumerate(matched)
                if target >= 0
            }
            valid_fragments = sample["fragment_mask"][0].cpu().numpy()
            points = (
                sample["traj_xy_norm"][0].cpu().numpy() * half_window)
            point_mask = sample["point_mask"][0].cpu().numpy()
            branch_mask = sample["branch_mask"][0].cpu().numpy()
            branch_endpoints = (
                sample["branch_offsets_norm"][0].cpu().numpy()
                * half_window)
            soft_targets = sample["support_targets"][0].cpu().numpy()
            valid_targets = sample["support_valid"][0].cpu().numpy()
            valid_target_indices = np.flatnonzero(
                branch_mask & valid_targets)
            column_count = max(1 + len(valid_target_indices), 2)
            figure, axes = plt.subplots(
                1, column_count,
                figsize=(5 * column_count, 5),
                squeeze=False,
            )
            all_axis = axes[0, 0]
            for fragment_index in np.flatnonzero(valid_fragments):
                fragment_points = points[
                    fragment_index, point_mask[fragment_index]]
                all_axis.plot(
                    fragment_points[:, 0],
                    fragment_points[:, 1],
                    color="0.75",
                    linewidth=0.7,
                )
            for target_index in np.flatnonzero(branch_mask):
                endpoint = branch_endpoints[target_index]
                all_axis.arrow(
                    0.0, 0.0, endpoint[0], endpoint[1],
                    width=0.4, head_width=3.0, length_includes_head=True)
            all_axis.scatter([0.0], [0.0], c="red", s=30)
            all_axis.set_title("all 64 candidates + GT branches")

            for axis_index, target_index in enumerate(
                    valid_target_indices, start=1):
                axis = axes[0, axis_index]
                query_index = target_to_query.get(int(target_index))
                if query_index is None:
                    axis.set_title(
                        "GT {} has no matched query".format(target_index))
                    continue
                valid_indices = np.flatnonzero(valid_fragments)
                order = valid_indices[np.argsort(
                    -probabilities[query_index, valid_indices],
                    kind="mergesort",
                )[:top_k]]
                for fragment_index in valid_indices:
                    fragment_points = points[
                        fragment_index, point_mask[fragment_index]]
                    axis.plot(
                        fragment_points[:, 0],
                        fragment_points[:, 1],
                        color="0.85",
                        linewidth=0.5,
                    )
                for rank, fragment_index in enumerate(order):
                    fragment_points = points[
                        fragment_index, point_mask[fragment_index]]
                    prediction = probabilities[
                        query_index, fragment_index]
                    target = soft_targets[
                        target_index, fragment_index]
                    axis.plot(
                        fragment_points[:, 0],
                        fragment_points[:, 1],
                        linewidth=1.0 + 2.5 * prediction,
                        label=(
                            "#{:d} f{:d} pred={:.2f} gt={:.2f}"
                            .format(rank + 1, fragment_index,
                                    prediction, target)),
                    )
                endpoint = branch_endpoints[target_index]
                axis.arrow(
                    0.0, 0.0, endpoint[0], endpoint[1],
                    color="black", width=0.5, head_width=3.0,
                    length_includes_head=True,
                )
                axis.scatter([0.0], [0.0], c="red", s=30)
                axis.set_title(
                    "GT branch {} / query {} top-{}".format(
                        target_index, query_index, top_k))
                axis.legend(fontsize=6, loc="upper right")
            for axis in axes[0]:
                axis.set_xlim(-half_window, half_window)
                axis.set_ylim(half_window, -half_window)
                axis.set_aspect("equal")
                axis.grid(alpha=0.15)
            figure.suptitle(
                "{} sample {}: support score vs soft target".format(
                    category, int(sample["sample_ids"][0])))
            figure.tight_layout()
            path = output_dir / "{}_{:02d}.png".format(
                category, case_number)
            figure.savefig(str(path), dpi=160)
            plt.close(figure)
            paths.append(str(path.resolve()))
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage3d_a_support.yml"),
    )
    parser.add_argument(
        "--mode",
        choices=("labels", "sanity", "train", "evaluate"),
        default="train",
    )
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--e4-checkpoint", type=Path)
    parser.add_argument("--image-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--support-checkpoint", type=Path)
    parser.add_argument("--device")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = _load_config(args.config)
    if "STAGE3D" not in cfg:
        raise ValueError("config must define STAGE3D")
    seed = int(cfg.STAGE3C.SEED)
    _set_seed(seed)
    device = _resolve_device(
        args.device or str(cfg.STAGE3C.DEVICE))
    dataset_dir = (
        args.dataset_dir or Path(cfg.STAGE3C.DATASET_DIR))
    output_dir = (
        args.output_dir or Path(cfg.STAGE3D.OUTPUT_DIR))
    e4_checkpoint = (
        args.e4_checkpoint or Path(cfg.STAGE3D.E4_CHECKPOINT))
    image_checkpoint = (
        args.image_checkpoint or Path(cfg.STAGE3C.IMAGE_CHECKPOINT))
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = Stage3CBranchDataset(
        dataset_dir, "train", preload=True)
    val_dataset = Stage3CBranchDataset(
        dataset_dir, "val", preload=True)
    label_report = run_label_diagnostics(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        cfg=cfg,
    )
    _write_json(output_dir / "label_diagnostics.json", label_report)
    print(json.dumps(label_report["gate"], indent=2, sort_keys=True))
    if args.mode == "labels":
        return
    if not label_report["gate"]["passed"]:
        raise RuntimeError(
            "multi-branch support availability is below the configured "
            "80% gate; support training was not started")

    e4_checkpoint = e4_checkpoint.resolve(strict=False)
    if not e4_checkpoint.is_file():
        raise FileNotFoundError(
            "E4 checkpoint not found: {}".format(e4_checkpoint))
    e4_sha256 = _sha256(e4_checkpoint)
    rpnet, _ = _load_frozen_rpnet(
        cfg, image_checkpoint, device)
    modules = _build_auxiliary_modules(cfg, device)
    e4_payload = load_stage3c_checkpoint(
        e4_checkpoint,
        trajectory_encoder=modules[0],
        graph_state_encoder=modules[1],
        branch_decoder=modules[2],
        optimizer=None,
        map_location=device,
    )
    _freeze_e4_modules(modules)
    if int(e4_payload.get("epoch", -1)) < 0:
        raise ValueError("E4 checkpoint has no valid epoch")
    criterion = _build_branch_criterion(cfg)
    train_cache, train_cache_report = build_frozen_support_cache(
        dataset=train_dataset,
        rpnet=rpnet,
        modules=modules,
        criterion=criterion,
        cfg=cfg,
        device=device,
    )
    val_cache, val_cache_report = build_frozen_support_cache(
        dataset=val_dataset,
        rpnet=rpnet,
        modules=modules,
        criterion=criterion,
        cfg=cfg,
        device=device,
    )
    _write_json(output_dir / "frozen_cache_report.json", {
        "train": train_cache_report,
        "validation": val_cache_report,
        "e4_checkpoint": str(e4_checkpoint),
        "e4_checkpoint_sha256": e4_sha256,
        "e4_checkpoint_epoch": int(e4_payload["epoch"]),
        "e4_strict_load": True,
        "e4_modules_frozen": True,
        "rpnet_strict_and_frozen": True,
        "support_changes_branch_outputs": False,
        "support_changes_trajectory_attention": False,
        "support_feeds_path_push": False,
    })

    sanity = run_support_overfit_sanity(
        cache=train_cache,
        cfg=cfg,
        device=device,
        output_dir=output_dir,
        e4_checkpoint=e4_checkpoint,
        e4_sha256=e4_sha256,
    )
    if args.mode == "sanity":
        if not sanity["passed"]:
            raise RuntimeError("support-head sanity overfit failed")
        return
    if not sanity["passed"]:
        raise RuntimeError(
            "32-sample support head did not clearly overfit; formal "
            "training was not started")

    _set_seed(seed)
    if args.mode == "evaluate":
        if args.support_checkpoint is None:
            raise ValueError(
                "--support-checkpoint is required for evaluate")
        support_head = TrajectorySupportHead(
            hidden_dim=int(cfg.STAGE3C.MODEL.HIDDEN_DIM),
            projection_dim=int(
                cfg.STAGE3D.MODEL.PROJECTION_DIM),
        ).to(device=device)
        load_stage3d_support_checkpoint(
            args.support_checkpoint,
            support_head=support_head,
            map_location=device,
        )
        validation = evaluate_support_head(
            support_head=support_head,
            dataset=val_cache,
            cfg=cfg,
            device=device,
            batch_size=int(
                cfg.STAGE3D.TRAINING.VAL_BATCH_SIZE),
        )
        _write_json(output_dir / "support_evaluation.json", validation)
        return

    support_head, training = run_formal_support_training(
        train_cache=train_cache,
        val_cache=val_cache,
        cfg=cfg,
        device=device,
        output_dir=output_dir,
        e4_checkpoint=e4_checkpoint,
        e4_sha256=e4_sha256,
    )
    visualizations = render_support_visualizations(
        support_head=support_head,
        cache=val_cache,
        cfg=cfg,
        device=device,
        output_dir=output_dir / "visualizations",
    )
    final_report = {
        "schema_version": "stage3d-a-v1",
        "config": str(args.config.resolve(strict=False)),
        "dataset_dir": str(dataset_dir.resolve(strict=False)),
        "e4_checkpoint": str(e4_checkpoint),
        "e4_checkpoint_sha256": e4_sha256,
        "e4_checkpoint_epoch": int(e4_payload["epoch"]),
        "label_diagnostics": label_report,
        "sanity": sanity,
        "training": training,
        "visualizations": visualizations,
        "constraints": {
            "e4_strict_load": True,
            "e4_modules_frozen": True,
            "only_support_head_trained": True,
            "trajectory_attention_unchanged": True,
            "branch_offset_existence_unchanged": True,
            "anchor_unchanged": True,
            "path_push_unchanged": True,
        },
    }
    _write_json(output_dir / "summary.json", final_report)
    print(json.dumps({
        "support_available_rate": label_report[
            "combined"]["support_available_rate"],
        "multibranch_support_available_rate": label_report[
            "combined"]["by_gt_branch_count"][">=2"][
                "support_available_rate"],
        "sanity_passed": sanity["passed"],
        "validation_support_ap": training[
            "best_validation"]["support_ap"],
        "attention_support_ap": training[
            "best_validation"]["attention_support_ap"],
        "checkpoint": training["best_checkpoint"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
