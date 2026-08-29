"""Frozen training-protocol helpers for Seg-Raster Stage S3B.

This module contains no model architecture.  It makes the loss, checkpoint,
selection and resource-cap rules independently testable before remote CUDA
training starts.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


STAGE_S3B_SEED = 20260827
MAX_OPTIMIZER_STEPS = 20480
EVAL_INTERVAL = 2560
CHECKPOINT_STEPS = tuple(range(0, MAX_OPTIMIZER_STEPS + 1, EVAL_INTERVAL))
LR_MULTIPLIERS = (1.0, 0.3, 0.1)
LOSS_LEGACY = "legacy_exact"
LOSS_BALANCED = "class_balanced_bce"
LOSS_BALANCED_DICE = "class_balanced_bce_plus_dice"
LOSS_KINDS = (LOSS_LEGACY, LOSS_BALANCED, LOSS_BALANCED_DICE)
GRAPH_CAPS = {
    "max_iterations": 3000,
    "max_vertices": 5000,
    "max_directed_edges": 10000,
    "max_wall_time_seconds": 900,
}


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def frozen_plan_batch_identities(
    sample_order: Sequence[Mapping[str, Any]], *, count: int = 100
) -> list[str]:
    """Recompute authoritative batch identities from frozen sample records.

    Historical S3 artifacts contain an aggregate first-100 hash produced by
    an older hash protocol.  The explicit sample_order rows are the data
    contract, so Stage S3B compares their canonical content batch by batch.
    """
    if len(sample_order) < count:
        raise ValueError("frozen sample order is shorter than parity gate")
    required = (
        "region", "crop_origin_xy", "extension_vertex_xy", "is_key_point",
        "target_count", "end_index", "augmentation")
    result = []
    for batch in sample_order[:count]:
        samples = batch.get("samples", [])
        canonical_samples = []
        for sample in samples:
            missing = [key for key in required if key not in sample]
            if missing:
                raise ValueError("sample plan row is missing: {}".format(missing))
            canonical_samples.append({key: sample[key] for key in required})
        per_sample = [canonical_sha256(value) for value in canonical_samples]
        result.append(canonical_sha256(per_sample))
    return result


def repair_composite(metrics: Mapping[str, float]) -> float:
    return float(
        (float(metrics["road_f1"]) + float(metrics["road_iou"])
         + float(metrics["junction_auprc"])) / 3.0)


def legacy_composite(metrics: Mapping[str, float]) -> float:
    return float(
        (float(metrics["road_f1"]) + float(metrics["road_iou"])
         + float(metrics["junction_f1"])) / 3.0)


def soft_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Mean soft Dice loss; no-positive targets remain finite by definition."""
    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have identical shape")
    probabilities = torch.sigmoid(logits)
    flat_probabilities = probabilities.reshape(probabilities.shape[0], -1)
    flat_targets = targets.reshape(targets.shape[0], -1)
    intersection = (flat_probabilities * flat_targets).sum(dim=1)
    denominator = flat_probabilities.sum(dim=1) + flat_targets.sum(dim=1)
    dice = (2.0 * intersection + float(smooth)) / (
        denominator + float(smooth))
    return (1.0 - dice).mean()


def junction_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    kind: str = LOSS_LEGACY,
    pos_weight: float = 1.0,
    alpha: float = 1.0,
    dice_weight: float = 1.0,
    dice_smooth: float = 1.0,
) -> torch.Tensor:
    if kind not in LOSS_KINDS:
        raise ValueError("unknown junction loss: {}".format(kind))
    if kind == LOSS_LEGACY:
        return F.binary_cross_entropy_with_logits(
            logits, targets, reduction="sum")
    weight = torch.as_tensor(
        float(pos_weight), dtype=logits.dtype, device=logits.device)
    balanced = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=weight, reduction="sum")
    if kind == LOSS_BALANCED:
        candidate = balanced
    else:
        candidate = balanced + float(dice_weight) * soft_dice_loss(
            logits, targets, smooth=dice_smooth)
    return float(alpha) * candidate


def positive_weight_from_counts(
    positive_count: int,
    negative_count: int,
    *,
    cap: float = 32.0,
) -> dict[str, float | int]:
    positive_count = int(positive_count)
    negative_count = int(negative_count)
    if positive_count < 0 or negative_count < 0:
        raise ValueError("pixel counts cannot be negative")
    raw = negative_count / max(positive_count, 1)
    return {
        "positive_count": positive_count,
        "negative_count": negative_count,
        "raw_pos_weight": float(raw),
        "capped_pos_weight": float(min(raw, float(cap))),
        "cap": float(cap),
    }


def gradient_matching_alpha(
    legacy_norms: Sequence[float], candidate_norms: Sequence[float]
) -> float:
    if len(legacy_norms) != len(candidate_norms) or not legacy_norms:
        raise ValueError("gradient norm series must be non-empty and aligned")
    legacy = float(np.mean(np.asarray(legacy_norms, dtype=np.float64)))
    candidate = float(np.mean(np.asarray(candidate_norms, dtype=np.float64)))
    if not math.isfinite(legacy) or not math.isfinite(candidate) or candidate <= 0:
        raise ValueError("gradient norms must be finite and candidate positive")
    return legacy / candidate


def best_step_by_repair(validation_by_step: Mapping[int | str, Mapping[str, float]]) -> int:
    if not validation_by_step:
        raise ValueError("validation series is empty")
    rows = [(int(step), repair_composite(metrics))
            for step, metrics in validation_by_step.items()]
    return max(rows, key=lambda item: (item[1], -item[0]))[0]


def retention(validation_by_step: Mapping[int | str, Mapping[str, float]]) -> float:
    latest_step = max(int(step) for step in validation_by_step)
    latest = repair_composite(validation_by_step[
        latest_step if latest_step in validation_by_step else str(latest_step)])
    peak = max(repair_composite(row) for row in validation_by_step.values())
    return latest / peak if peak > 0 else 0.0


def select_learning_rate(
    image_only_candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select only from explicitly image-only Phase-A candidates."""
    if not image_only_candidates:
        raise ValueError("no LR candidates")
    if any(row.get("input_kind") != "image_only" for row in image_only_candidates):
        raise ValueError("LR selection may only inspect image-only candidates")
    valid = [row for row in image_only_candidates
             if row.get("status") == "PASS" and bool(row.get("finite", True))]
    if not valid:
        return {"status": "FAIL", "reason": "no_finite_candidate"}
    stable = [row for row in valid if float(row["retention"]) >= 0.70]
    pool = stable or valid
    best_score = max(float(row["best_repair_composite"]) for row in pool)
    near = [row for row in pool
            if best_score - float(row["best_repair_composite"]) < 0.001]
    selected = min(near, key=lambda row: float(row["lr_multiplier"]))
    return {
        "status": "PASS" if stable else "FAIL",
        "lr_stability_gate": "PASS" if stable else "FAIL",
        "selected_run": selected["run_key"],
        "selected_lr_multiplier": float(selected["lr_multiplier"]),
        "selected_base_lr": float(selected["base_lr"]),
        "phase_b_allowed": bool(stable),
        "selection_scope": [str(row["run_key"]) for row in image_only_candidates],
    }


def select_junction_loss(
    image_only_candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the preregistered loss rule to image-only candidates only."""
    if not image_only_candidates:
        raise ValueError("no loss candidates")
    if any(row.get("input_kind") != "image_only" for row in image_only_candidates):
        raise ValueError("loss selection may only inspect image-only candidates")
    by_kind = {str(row["loss_kind"]): row for row in image_only_candidates}
    legacy = by_kind.get(LOSS_LEGACY)
    if legacy is None:
        raise ValueError("legacy control is required")
    eligible = []
    for row in image_only_candidates:
        road_ok = not (
            float(row["best_metrics"]["road_f1"])
            < 0.95 * float(legacy["best_metrics"]["road_f1"])
            and float(row["best_metrics"]["road_iou"])
            < 0.95 * float(legacy["best_metrics"]["road_iou"]))
        f1_ok = float(row["best_metrics"]["junction_shared_f1"]) >= float(
            legacy["best_metrics"]["junction_shared_f1"])
        if (row.get("status") == "PASS" and bool(row.get("finite", True))
                and not bool(row.get("gradient_explosion", False))
                and road_ok and f1_ok):
            eligible.append(row)
    improved = [row for row in eligible if row["loss_kind"] != LOSS_LEGACY
                and float(row["best_metrics"]["junction_auprc"])
                > float(legacy["best_metrics"]["junction_auprc"])]
    pool = improved or [legacy]
    stable = [row for row in pool if float(row["retention"]) >= 0.70]
    selected = max(
        stable or pool,
        key=lambda row: (float(row["best_metrics"]["junction_auprc"]),
                         float(row["retention"])))
    return {
        "status": "PASS",
        "junction_loss_repair": (
            "PASS" if selected["loss_kind"] != LOSS_LEGACY
            else "NO_EVIDENCE"),
        "selected_run": selected["run_key"],
        "selected_loss_kind": selected["loss_kind"],
        "selection_scope": [str(row["run_key"]) for row in image_only_candidates],
    }


def simulate_early_stop(
    validation_by_step: Mapping[int | str, Mapping[str, float]],
    *,
    min_steps: int = 5120,
    patience: int = 3,
    min_delta: float = 0.0005,
) -> dict[str, Any]:
    best = -math.inf
    stale = 0
    stop_step = None
    history = []
    for step in sorted(int(value) for value in validation_by_step):
        metrics = validation_by_step[
            step if step in validation_by_step else str(step)]
        score = repair_composite(metrics)
        improved = score > best + float(min_delta)
        if improved:
            best = score
            stale = 0
        elif step >= min_steps:
            stale += 1
        history.append({"step": step, "repair_composite": score,
                        "improved": improved, "stale_intervals": stale})
        if step >= min_steps and stale >= patience:
            stop_step = step
            break
    return {
        "status": "PASS", "simulated_stop_step": stop_step,
        "would_stop": stop_step is not None, "min_steps": min_steps,
        "patience_intervals": patience, "min_delta": min_delta,
        "selection_source": "image_only_validation_control_only",
        "history": history,
    }


def choose_shared_threshold(
    logits: np.ndarray,
    targets: np.ndarray,
    thresholds: Iterable[float] | None = None,
) -> dict[str, Any]:
    """Choose one threshold from the image-only calibration subset."""
    probability = 1.0 / (1.0 + np.exp(-np.clip(
        np.asarray(logits, dtype=np.float64), -80.0, 80.0)))
    truth = np.asarray(targets) > 0
    candidates = list(thresholds if thresholds is not None
                      else np.linspace(0.01, 0.99, 99))
    rows = []
    for threshold in candidates:
        prediction = probability >= float(threshold)
        tp = int(np.count_nonzero(prediction & truth))
        fp = int(np.count_nonzero(prediction & ~truth))
        fn = int(np.count_nonzero(~prediction & truth))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"threshold": float(threshold), "precision": precision,
                     "recall": recall, "f1": f1,
                     "predicted_positive_count": tp + fp})
    selected = max(rows, key=lambda row: (row["f1"], -row["threshold"]))
    return {
        "status": "PASS", "selected_threshold": selected["threshold"],
        "selection_source": "selected_image_only_calibration_subset",
        "frozen_for_controls": True,
        "target_positive_count": int(np.count_nonzero(truth)),
        "curve": rows,
    }


def save_versioned_model_checkpoint(
    path: str | Path,
    state_dict: Mapping[str, torch.Tensor],
    *,
    step: int,
    code_sha: str,
    config_sha: str,
    metric_code_sha: str,
) -> None:
    """Atomically create (never overwrite) one model-only checkpoint."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    temporary = destination.with_name(destination.name + ".tmp-{}".format(os.getpid()))
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        torch.save({
            "stage": "seg_raster_stage_s3b",
            "kind": "versioned_model_only",
            "optimizer_step": int(step),
            "code_sha": code_sha,
            "config_sha": config_sha,
            "metric_code_sha": metric_code_sha,
            "state_dict": state_dict,
        }, temporary)
        if destination.exists():
            raise FileExistsError(destination)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_common_step_checkpoint(
    path: str | Path, *, expected_step: int, expected_code_sha: str
) -> Mapping[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("kind") != "versioned_model_only":
        raise ValueError("not a versioned model-only checkpoint")
    if int(payload.get("optimizer_step", -1)) != int(expected_step):
        raise ValueError("checkpoint step mismatch")
    if payload.get("code_sha") != expected_code_sha:
        raise ValueError("checkpoint code SHA mismatch")
    if "optimizer" in payload:
        raise ValueError("model-only checkpoint contains optimizer state")
    return payload["state_dict"]


@dataclass(frozen=True)
class GraphResourceSnapshot:
    iterations: int
    vertices: int
    directed_edges: int
    elapsed_seconds: float


def graph_resource_status(
    snapshot: GraphResourceSnapshot,
    caps: Mapping[str, int | float] = GRAPH_CAPS,
) -> dict[str, Any]:
    reached = []
    if snapshot.iterations >= int(caps["max_iterations"]):
        reached.append("MAX_GRAPH_ITERATIONS")
    if snapshot.vertices >= int(caps["max_vertices"]):
        reached.append("MAX_GRAPH_VERTICES")
    if snapshot.directed_edges >= int(caps["max_directed_edges"]):
        reached.append("MAX_DIRECTED_EDGES")
    if snapshot.elapsed_seconds >= float(caps["max_wall_time_seconds"]):
        reached.append("MAX_GRAPH_WALL_TIME_SECONDS")
    return {
        "status": "RESOURCE_CAP_REACHED" if reached else "WITHIN_RESOURCE_CAP",
        "reached_caps": reached,
        "snapshot": {
            "iterations": snapshot.iterations,
            "vertices": snapshot.vertices,
            "directed_edges": snapshot.directed_edges,
            "elapsed_seconds": snapshot.elapsed_seconds,
        },
        "caps": dict(caps),
        "natural_termination": False if reached else None,
    }


def assert_json_finite(value: Any) -> None:
    if isinstance(value, Mapping):
        for child in value.values():
            assert_json_finite(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            assert_json_finite(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON contains NaN or Infinity")


def validate_commit_paths(paths: Sequence[str]) -> None:
    forbidden = {
        "checkpoint", "checkpoints", "dataset", "datasets", "raster",
        "tensorboard", "cache", "weights", "__pycache__", ".pytest_cache"}
    for path in paths:
        parts = {part.lower() for part in Path(path).parts}
        if parts & forbidden:
            raise ValueError("forbidden Stage S3B commit path: " + path)
