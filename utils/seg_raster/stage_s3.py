"""Deterministic controls and audit helpers for Seg-Raster Stage S3.

The functions in this module are deliberately independent from the training
entry point.  They make the six-run comparison, replay identity, strict
initialization, spatial split, and GPU eligibility rules directly testable.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml


CONTROL_ALIGNED = "aligned"
CONTROL_ZERO = "zero"
CONTROL_SHIFT_FIXED = "shift_fixed"
CONTROLS = (CONTROL_ALIGNED, CONTROL_ZERO, CONTROL_SHIFT_FIXED)
STAGE_S3_SEED = 20260827


@dataclass(frozen=True)
class ExperimentSpec:
    key: str
    run_id: str
    trajectory_mode: str
    raster_control: str | None
    anchor_grad_to_seg: bool


EXPERIMENT_MATRIX = (
    ExperimentSpec("C0", "C0_image_detach_seed20260827", "none", None, False),
    ExperimentSpec(
        "C1", "C1_aligned_detach_seed20260827", "raster_seg_only",
        CONTROL_ALIGNED, False),
    ExperimentSpec(
        "C2", "C2_zero_detach_seed20260827", "raster_seg_only",
        CONTROL_ZERO, False),
    ExperimentSpec(
        "C3", "C3_shifted_detach_seed20260827", "raster_seg_only",
        CONTROL_SHIFT_FIXED, False),
    ExperimentSpec("J0", "J0_image_joint_seed20260827", "none", None, True),
    ExperimentSpec(
        "J1", "J1_aligned_joint_seed20260827", "raster_seg_only",
        CONTROL_ALIGNED, True),
)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("utf-8")


def identity_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_stage_s3_config(path: str | Path) -> dict[str, Any]:
    """Load one explicit S3 overlay and its repository-relative base file."""
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as handle:
        overlay = yaml.safe_load(handle) or {}
    base_name = overlay.pop("BASE_CONFIG", None)
    if base_name is None:
        resolved = overlay
    else:
        base_path = (source.parent / str(base_name)).resolve()
        with base_path.open("r", encoding="utf-8") as handle:
            base = yaml.safe_load(handle) or {}
        resolved = _deep_merge(base, overlay)
    validate_experiment_config(resolved)
    return resolved


def validate_experiment_config(config: Mapping[str, Any]) -> None:
    s3 = config.get("S3", {})
    traj = config.get("TRAJ", {})
    train = config.get("TRAIN", {})
    mode = traj.get("MODE")
    anchor_grad = bool(traj.get("RASTER", {}).get("ANCHOR_GRAD_TO_SEG", True))
    if mode not in ("none", "raster_seg_only"):
        raise ValueError("S3 supports only none or raster_seg_only")
    if traj.get("SEQUENCE", {}).get("ENABLED", False):
        raise ValueError("S3 forbids trajectory sequence input")
    if train.get("MODEL", "origin") != "origin":
        raise ValueError("S3 forbids legacy DSF as a formal path")
    if bool(train.get("DATA_PARALLEL", False)):
        raise ValueError("S3 requires one process per physical GPU")
    if int(train.get("NUM_TARGETS", 0)) != 4:
        raise ValueError("S3 requires NUM_TARGETS=4")
    if int(train.get("WINDOW_SIZE", 0)) != 256:
        raise ValueError("S3 requires 256x256 crops")
    control = traj.get("RASTER", {}).get("CONTROL")
    if mode == "none" and control is not None:
        raise ValueError("image-only runs cannot declare a raster control")
    if mode == "raster_seg_only" and control not in CONTROLS:
        raise ValueError("raster_seg_only requires an S3 control")
    declared = s3.get("ANCHOR_GRAD_TO_SEG")
    if declared is not None and bool(declared) != anchor_grad:
        raise ValueError("S3 and TRAJ anchor-gradient declarations differ")


def apply_raster_control(
    raster: np.ndarray,
    valid_mask: np.ndarray,
    control: str,
    *,
    shift_xy: tuple[int, int] = (128, 128),
) -> tuple[np.ndarray, np.ndarray]:
    """Apply an S3 control without changing shape or the real valid mask.

    Shift is performed in the canonical full-canvas coordinate system with
    zero fill.  There is no circular wrap.
    """
    raster = np.asarray(raster)
    valid_mask = np.asarray(valid_mask)
    if raster.shape != valid_mask.shape:
        raise ValueError("raster and valid_mask shapes must match")
    if control not in CONTROLS:
        raise ValueError("unknown raster control: {!r}".format(control))
    binary = (raster > 0).astype(np.float32, copy=False)
    mask = (valid_mask > 0).astype(np.float32, copy=True)
    binary = binary * mask
    if control == CONTROL_ALIGNED:
        result = binary.copy()
    elif control == CONTROL_ZERO:
        result = np.zeros_like(binary, dtype=np.float32)
    else:
        shift_x, shift_y = map(int, shift_xy)
        result = np.zeros_like(binary, dtype=np.float32)
        height, width = binary.shape[-2:]
        src_x0 = max(0, -shift_x)
        src_x1 = min(width, width - shift_x)
        src_y0 = max(0, -shift_y)
        src_y1 = min(height, height - shift_y)
        if src_x1 > src_x0 and src_y1 > src_y0:
            dst_x0, dst_x1 = src_x0 + shift_x, src_x1 + shift_x
            dst_y0, dst_y1 = src_y0 + shift_y, src_y1 + shift_y
            result[..., dst_y0:dst_y1, dst_x0:dst_x1] = (
                binary[..., src_y0:src_y1, src_x0:src_x1])
        result *= mask
    return np.ascontiguousarray(result), np.ascontiguousarray(mask)


def apply_synchronized_augmentation(
    arrays: Mapping[str, np.ndarray],
    *,
    rot90_k: int = 0,
    flip_x: bool = False,
    flip_y: bool = False,
) -> dict[str, np.ndarray]:
    """Apply one declared spatial transform to every aligned HxW field."""
    transformed: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        array = np.asarray(value)
        if array.ndim < 2:
            raise ValueError("{} has no spatial dimensions".format(key))
        if array.shape[-2] != array.shape[-1]:
            raise ValueError("synchronized S3 crops must be square")
        out = np.rot90(array, k=int(rot90_k) % 4, axes=(-2, -1))
        if flip_x:
            out = np.flip(out, axis=-1)
        if flip_y:
            out = np.flip(out, axis=-2)
        transformed[key] = np.ascontiguousarray(out)
    return transformed


def sample_identity(record: Mapping[str, Any]) -> str:
    """Hash only the fields that define one replayable training sample."""
    required = (
        "region", "crop_origin_xy", "extension_vertex_xy", "is_key_point",
        "target_count", "end_index", "augmentation")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError("sample identity is missing: {}".format(missing))
    return identity_sha256({key: record[key] for key in required})


def audit_batch_parity(
    plans: Mapping[str, Sequence[Mapping[str, Any]]],
    count: int = 100,
) -> dict[str, Any]:
    if not plans:
        raise ValueError("at least one plan is required")
    plan_names = sorted(plans)
    compared = min(count, *(len(plans[name]) for name in plan_names))
    identities = {
        name: [sample_identity(row) for row in plans[name][:compared]]
        for name in plan_names
    }
    reference = identities[plan_names[0]]
    mismatches = {
        name: [idx for idx, pair in enumerate(zip(reference, values))
               if pair[0] != pair[1]]
        for name, values in identities.items()
    }
    passed = compared >= count and not any(mismatches.values())
    return {
        "status": "PASS" if passed else "FAIL",
        "requested_batch_count": count,
        "compared_batch_count": compared,
        "plan_names": plan_names,
        "mismatch_indices": mismatches,
        "reference_identity_sha256": identity_sha256(reference),
    }


@dataclass(frozen=True)
class SpatialExtent:
    x0: int
    y0: int
    x1: int
    y1: int

    def validate(self) -> None:
        if self.x0 < 0 or self.y0 < 0 or self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("invalid spatial extent: {}".format(self))

    def intersects(self, other: "SpatialExtent") -> bool:
        return not (
            self.x1 <= other.x0 or other.x1 <= self.x0
            or self.y1 <= other.y0 or other.y1 <= self.y0)

    def contains_crop(self, x: int, y: int, size: int) -> bool:
        return (self.x0 <= x and self.y0 <= y
                and x + size <= self.x1 and y + size <= self.y1)


def build_spatial_split(
    *,
    canvas_wh: tuple[int, int],
    crop_size: int = 256,
    boundary_buffer: int = 256,
) -> dict[str, Any]:
    """Create a deterministic left/right split separated by a crop buffer."""
    width, height = map(int, canvas_wh)
    mid = width // 2
    half_buffer = int(math.ceil(boundary_buffer / 2))
    train = SpatialExtent(0, 0, mid - half_buffer, height)
    validation = SpatialExtent(mid + half_buffer, 0, width, height)
    train.validate()
    validation.validate()
    if train.intersects(validation):
        raise AssertionError("train and validation extents overlap")
    if min(train.x1 - train.x0, validation.x1 - validation.x0, height) < crop_size:
        raise ValueError("split is too small for the requested crop")
    payload = {
        "kind": "deterministic_xian_spatial_holdout",
        "canvas_wh": [width, height],
        "crop_size": crop_size,
        "excluded_boundary_buffer": boundary_buffer,
        "train_extent": asdict(train),
        "validation_extent": asdict(validation),
        "overlap_check": "PASS",
    }
    payload["manifest_sha256"] = identity_sha256(payload)
    return payload


def parse_gpu_inventory_csv(text: str) -> list[dict[str, Any]]:
    fields = (
        "index", "uuid", "name", "driver_version", "memory_total_mb",
        "memory_used_mb", "memory_free_mb", "utilization_percent",
        "temperature_c")
    rows = []
    for raw in csv.reader(line for line in text.splitlines() if line.strip()):
        if len(raw) != len(fields):
            raise ValueError("unexpected nvidia-smi GPU row: {!r}".format(raw))
        row = dict(zip(fields, (item.strip() for item in raw)))
        for key in (
                "index", "memory_total_mb", "memory_used_mb", "memory_free_mb",
                "utilization_percent", "temperature_c"):
            row[key] = int(row[key])
        rows.append(row)
    return rows


def parse_compute_apps_csv(text: str) -> list[dict[str, Any]]:
    rows = []
    for raw in csv.reader(line for line in text.splitlines() if line.strip()):
        if len(raw) != 4:
            raise ValueError("unexpected compute-app row: {!r}".format(raw))
        used = raw[2].strip()
        rows.append({
            "pid": int(raw[0].strip()),
            "gpu_uuid": raw[1].strip(),
            "used_memory_mb": None if used in ("N/A", "[N/A]") else int(used),
            "process_name": PureWindowsPath(raw[3].strip()).name,
        })
    return rows


def required_free_memory_mb(
    max_allocated_mb: float, max_reserved_mb: float) -> int:
    return max(
        int(math.ceil(max_reserved_mb * 1.30)),
        int(math.ceil(max_allocated_mb + 2048.0)))


def evaluate_gpu_eligibility(
    samples: Sequence[Mapping[str, Any]],
    compute_apps: Sequence[Mapping[str, Any]],
    *,
    required_free_mb: int,
    excluded_indices: Iterable[int] = (),
    own_pids: Iterable[int] = (),
    max_utilization: int = 15,
) -> list[dict[str, Any]]:
    """Apply the S3 hard GPU gate to three or more inventory snapshots."""
    if len(samples) < 3:
        raise ValueError("GPU eligibility requires at least three samples")
    excluded = {int(value) for value in excluded_indices}
    own = {int(value) for value in own_pids}
    by_index: dict[int, list[Mapping[str, Any]]] = {}
    for sample in samples:
        for gpu in sample["gpus"]:
            by_index.setdefault(int(gpu["index"]), []).append(gpu)
    result = []
    for index, observations in sorted(by_index.items()):
        reasons = []
        if len(observations) != len(samples):
            reasons.append("missing_inventory_sample")
        first = observations[0]
        uuid = first["uuid"]
        if index in excluded:
            reasons.append("excluded_by_environment")
        external = [
            app for app in compute_apps
            if app["gpu_uuid"] == uuid and int(app["pid"]) not in own]
        if external:
            reasons.append("external_compute_process")
        if any(int(obs["utilization_percent"]) > max_utilization
               for obs in observations):
            reasons.append("utilization_above_limit")
        if any(int(obs["memory_free_mb"]) <= required_free_mb
               for obs in observations):
            reasons.append("insufficient_free_memory")
        result.append({
            "index": index,
            "uuid": uuid,
            "name": first["name"],
            "eligible": not reasons,
            "reasons": reasons,
            "external_compute_processes": external,
            "sample_count": len(observations),
        })
    return result


class FifoGpuScheduler:
    """Pure FIFO allocation state; it never terminates external processes."""

    def __init__(self, gpu_indices: Iterable[int]):
        self.available = list(dict.fromkeys(int(value) for value in gpu_indices))
        self.running: dict[int, str] = {}

    def allocate(self, run_id: str) -> int | None:
        if run_id in self.running.values():
            raise ValueError("run is already scheduled")
        if not self.available:
            return None
        index = self.available.pop(0)
        if index in self.running:
            raise AssertionError("GPU double allocation")
        self.running[index] = run_id
        return index

    def release(self, index: int, run_id: str) -> None:
        index = int(index)
        if self.running.get(index) != run_id:
            raise ValueError("GPU/run allocation mismatch")
        del self.running[index]
        self.available.append(index)


def strict_shared_state_audit(
    image_only_state: Mapping[str, torch.Tensor],
    raster_state: Mapping[str, torch.Tensor],
    *,
    raster_prefix: str = "segmentation_raster_fusion.",
) -> dict[str, Any]:
    """Require every shared VecRoad tensor to exist with the same shape."""
    image_keys = set(image_only_state)
    raster_keys = set(raster_state)
    missing_shared = sorted(image_keys - raster_keys)
    extra_non_raster = sorted(
        key for key in raster_keys - image_keys if not key.startswith(raster_prefix))
    raster_only = sorted(
        key for key in raster_keys - image_keys if key.startswith(raster_prefix))
    shape_mismatch = sorted(
        key for key in image_keys & raster_keys
        if tuple(image_only_state[key].shape) != tuple(raster_state[key].shape))
    status = "PASS" if not (
        missing_shared or extra_non_raster or shape_mismatch) else "FAIL"
    return {
        "status": status,
        "shared_key_count": len(image_keys & raster_keys),
        "missing_shared_keys": missing_shared,
        "unexpected_non_raster_keys": extra_non_raster,
        "shape_mismatch_keys": shape_mismatch,
        "raster_only_key_count": len(raster_only),
        "raster_only_keys": raster_only,
    }


def assert_zero_initialized_raster_residual(
    state: Mapping[str, torch.Tensor],
) -> list[str]:
    candidates = [
        key for key in state
        if key.startswith("segmentation_raster_fusion.")
        and ("output_projection" in key or "residual_projection" in key
             or "delta_projection" in key)
    ]
    nonzero = [key for key in candidates if torch.count_nonzero(state[key]).item()]
    if nonzero:
        raise ValueError("raster residual is not zero initialized: {}".format(nonzero))
    return candidates


def stitch_tiles(
    tiles: Sequence[np.ndarray],
    origins_xy: Sequence[tuple[int, int]],
    canvas_hw: tuple[int, int],
) -> np.ndarray:
    """Average overlapping CHW/HW logits into a fixed full canvas."""
    if len(tiles) != len(origins_xy) or not tiles:
        raise ValueError("tiles and origins must be non-empty and equal length")
    first = np.asarray(tiles[0])
    prefix = first.shape[:-2]
    height, width = map(int, canvas_hw)
    total = np.zeros(prefix + (height, width), dtype=np.float64)
    count = np.zeros((height, width), dtype=np.float64)
    for tile, (x, y) in zip(tiles, origins_xy):
        tile = np.asarray(tile)
        if tile.shape[:-2] != prefix:
            raise ValueError("tile channel shapes differ")
        tile_h, tile_w = tile.shape[-2:]
        if x < 0 or y < 0 or x + tile_w > width or y + tile_h > height:
            raise ValueError("tile exceeds canvas")
        total[..., y:y + tile_h, x:x + tile_w] += tile
        count[y:y + tile_h, x:x + tile_w] += 1
    if np.any(count == 0):
        raise ValueError("stitch plan leaves uncovered canvas pixels")
    return (total / count).astype(first.dtype, copy=False)


def binary_segmentation_metrics(
    logits: np.ndarray,
    target: np.ndarray,
    *,
    threshold: float = 0.3,
) -> dict[str, float]:
    logits = np.asarray(logits, dtype=np.float64)
    target = np.asarray(target) > 0
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    prediction = probability >= float(threshold)
    tp = int(np.count_nonzero(prediction & target))
    fp = int(np.count_nonzero(prediction & ~target))
    fn = int(np.count_nonzero(~prediction & target))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    flat_target = target.reshape(-1).astype(np.int64)
    flat_probability = probability.reshape(-1)
    positive_count = int(flat_target.sum())
    if positive_count:
        order = np.argsort(-flat_probability, kind="mergesort")
        sorted_target = flat_target[order]
        cumulative_tp = np.cumsum(sorted_target)
        precision_curve = cumulative_tp / np.arange(1, len(sorted_target) + 1)
        auprc = float(np.sum(precision_curve * sorted_target) / positive_count)
    else:
        auprc = 0.0
    return {
        "precision": precision, "recall": recall, "f1": f1, "iou": iou,
        "auprc": auprc,
        "true_positive": tp, "false_positive": fp, "false_negative": fn,
    }


def anchor_metrics(
    logits: np.ndarray,
    target: np.ndarray,
    end_indices: Sequence[int],
    *,
    threshold: float = 0.3,
    top_k: int = 10,
) -> dict[str, Any]:
    logits = np.asarray(logits, dtype=np.float64)
    target = np.asarray(target) > 0
    if logits.shape != target.shape or logits.ndim != 4:
        raise ValueError("anchor logits and target must be equal BCHW arrays")
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    prediction = probability >= threshold
    per_step_hits = np.zeros(logits.shape[1], dtype=np.int64)
    per_step_total = np.zeros(logits.shape[1], dtype=np.int64)
    top_k_hits = 0
    target_count = 0
    localization_errors = []
    false_positives = 0
    missed_branches = 0
    for batch_index, end_index in enumerate(end_indices):
        for step in range(min(int(end_index), logits.shape[1])):
            truth = target[batch_index, step]
            if not np.any(truth):
                continue
            target_count += 1
            per_step_total[step] += 1
            hit = bool(np.any(prediction[batch_index, step] & truth))
            per_step_hits[step] += int(hit)
            missed_branches += int(not hit)
            false_positives += int(np.count_nonzero(
                prediction[batch_index, step] & ~truth))
            flat = probability[batch_index, step].reshape(-1)
            k = min(int(top_k), flat.size)
            top_indices = np.argpartition(flat, -k)[-k:]
            truth_flat = truth.reshape(-1)
            top_k_hits += int(np.any(truth_flat[top_indices]))
            peak = np.unravel_index(int(np.argmax(flat)), truth.shape)
            truth_yx = np.argwhere(truth)
            distances = np.sqrt(np.sum(
                (truth_yx - np.asarray(peak)[None, :]) ** 2, axis=1))
            localization_errors.append(float(np.min(distances)))
    diversity = []
    for left in range(logits.shape[1]):
        for right in range(left + 1, logits.shape[1]):
            diversity.append(float(np.mean(np.abs(
                probability[:, left] - probability[:, right]))))
    return {
        "per_step_recall": [
            float(per_step_hits[index] / per_step_total[index])
            if per_step_total[index] else 0.0
            for index in range(logits.shape[1])],
        "top_k": int(top_k),
        "top_k_recall": float(top_k_hits / target_count) if target_count else 0.0,
        "localization_error": float(np.mean(localization_errors))
        if localization_errors else 0.0,
        "false_positive_count": false_positives,
        "missed_branch_count": missed_branches,
        "channel_diversity_mean_absolute_difference": float(np.mean(diversity))
        if diversity else 0.0,
        "evaluated_target_count": target_count,
        "fixed_threshold": threshold,
    }


def coverage_bin(value: float) -> str:
    if value < 0.01:
        return "low"
    if value < 0.05:
        return "medium"
    return "high"


def segmentation_causal_screen(
    metrics: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    required = ("C0", "C1", "C2", "C3")
    if any(key not in metrics for key in required):
        return {"status": "INCONCLUSIVE", "reason": "missing_required_run"}
    names = ("road_f1", "road_iou", "junction_f1")
    deltas = {
        run: {name: float(metrics[run][name]) - float(metrics["C0"][name])
              for name in names}
        for run in ("C1", "C2", "C3")
    }
    improved_count = sum(deltas["C1"][name] > 0 for name in names)
    means = {run: sum(values.values()) / len(names) for run, values in deltas.items()}
    collapse = any(
        float(metrics["C1"].get(name, 0.0))
        < float(metrics["C0"].get(name, 0.0)) - 0.10
        for name in ("road_precision", "road_recall",
                     "junction_precision", "junction_recall"))
    if improved_count >= 2 and means["C1"] > max(means["C2"], means["C3"]) and not collapse:
        status = "PROMISING"
    elif means["C1"] < 0:
        status = "REGRESSION"
    elif means["C1"] <= max(means["C2"], means["C3"]):
        status = "NO_EVIDENCE"
    else:
        status = "INCONCLUSIVE"
    return {
        "status": status,
        "metric_deltas_vs_C0": deltas,
        "mean_deltas_vs_C0": means,
        "c1_improved_metric_count": improved_count,
        "precision_recall_collapse": collapse,
    }


def indirect_anchor_screen(
    c0: Mapping[str, float] | None,
    c1: Mapping[str, float] | None,
) -> dict[str, Any]:
    if c0 is None or c1 is None:
        return {"status": "INCONCLUSIVE", "reason": "missing_required_run"}
    recall_delta = float(c1["top_k_recall"]) - float(c0["top_k_recall"])
    improvements = {
        "top_k_recall": recall_delta,
        "localization_error": (
            float(c0["localization_error"]) - float(c1["localization_error"])),
        "missed_branch_count": (
            float(c0["missed_branch_count"]) - float(c1["missed_branch_count"])),
    }
    if recall_delta < -0.005:
        status = "REGRESSION"
    elif any(value > 0 for value in improvements.values()):
        status = "PROMISING"
    else:
        status = "NO_EVIDENCE"
    return {"status": status, "deltas": improvements}


def experiment_matrix_payload() -> dict[str, Any]:
    rows = [asdict(spec) for spec in EXPERIMENT_MATRIX]
    payload = {
        "stage": "seg_raster_stage_s3",
        "seed": STAGE_S3_SEED,
        "runs": rows,
        "shared_constraints": {
            "data_parallel": False,
            "num_targets": 4,
            "crop_size": 256,
            "raster_lr_multiplier": 1.0,
            "one_physical_gpu_per_run": True,
        },
    }
    payload["matrix_sha256"] = identity_sha256(payload)
    return payload
