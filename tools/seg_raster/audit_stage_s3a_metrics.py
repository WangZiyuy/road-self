"""Independent metric and protocol primitives for the Stage S3A audit.

This module deliberately does not import the Stage S3 metric implementation.
It is used as the NumPy reference against which the frozen evaluator is
checked.  It contains no model or training mutation code.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REFERENCE_SCHEMA_VERSION = "1.0.0"
DEFAULT_THRESHOLD_GRID = np.unique(np.concatenate([
    np.geomspace(0.001, 0.1, 25),
    np.linspace(0.11, 0.5, 40),
])).astype(np.float64)


def sigmoid_once(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))


def average_precision_reference(
    probability: np.ndarray,
    target: np.ndarray,
) -> float:
    """Non-interpolated AP with tied scores evaluated as one threshold group."""
    scores = np.asarray(probability, dtype=np.float64).reshape(-1)
    truth = (np.asarray(target).reshape(-1) > 0).astype(np.int64)
    positives = int(truth.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_truth = truth[order]
    cumulative_tp = np.cumsum(sorted_truth)
    cumulative_fp = np.cumsum(1 - sorted_truth)
    group_end = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), len(scores) - 1]
    tp = cumulative_tp[group_end].astype(np.float64)
    fp = cumulative_fp[group_end].astype(np.float64)
    recall = tp / positives
    precision = tp / np.maximum(tp + fp, 1.0)
    recall_increment = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_increment * precision))


def binary_reference_metrics(
    logits: np.ndarray,
    target: np.ndarray,
    *,
    threshold: float = 0.3,
) -> dict[str, Any]:
    logits = np.asarray(logits, dtype=np.float64)
    target_bool = np.asarray(target) > 0
    if logits.shape != target_bool.shape:
        raise ValueError("logits and target must have identical shapes")
    probability = sigmoid_once(logits)
    prediction = probability >= float(threshold)
    tp = int(np.count_nonzero(prediction & target_bool))
    fp = int(np.count_nonzero(prediction & ~target_bool))
    fn = int(np.count_nonzero(~prediction & target_bool))
    tn = int(np.count_nonzero(~prediction & ~target_bool))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "auprc": average_precision_reference(probability, target_bool),
        "confusion_matrix": {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
        },
        "predicted_positive_count": int(tp + fp),
        "target_positive_count": int(tp + fn),
        "total_pixel_count": int(target_bool.size),
        "threshold": float(threshold),
        "aggregation": "micro_over_all_pixels",
    }


def _summary(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not array.size:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
    }


def probability_forensics(
    logits: np.ndarray,
    target: np.ndarray,
    *,
    count_thresholds: Sequence[float] = (0.01, 0.03, 0.05, 0.1, 0.2, 0.3),
) -> dict[str, Any]:
    values = np.asarray(logits, dtype=np.float64)
    truth = np.asarray(target) > 0
    if values.shape != truth.shape:
        raise ValueError("logits and target must have identical shapes")
    probability = sigmoid_once(values)
    quantile_levels = np.asarray(
        [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0])
    quantile_names = ["p0", "p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "p100"]
    flat_probability = probability.reshape(-1)
    if flat_probability.size:
        quantiles = np.quantile(flat_probability, quantile_levels)
        quantile_payload = {
            name: float(value) for name, value in zip(quantile_names, quantiles)
        }
    else:
        quantile_payload = {name: None for name in quantile_names}
    imagewise = probability.reshape(probability.shape[0], -1).max(axis=1)
    return {
        "logits_all": _summary(values),
        "logits_target_positive": _summary(values[truth]),
        "logits_target_negative": _summary(values[~truth]),
        "probability_quantiles": quantile_payload,
        "probability_max": float(flat_probability.max()) if flat_probability.size else None,
        "pixels_above_probability": {
            format(float(threshold), ".6g"): int(np.count_nonzero(
                flat_probability >= float(threshold)))
            for threshold in count_thresholds
        },
        "imagewise_max_probability": [float(value) for value in imagewise],
        "target_positive_pixel_count": int(np.count_nonzero(truth)),
        "target_negative_pixel_count": int(np.count_nonzero(~truth)),
        "positive_pixel_rate": float(np.mean(truth)),
    }


def calibration_metrics(
    logits: np.ndarray,
    target: np.ndarray,
    *,
    bin_count: int = 15,
) -> dict[str, Any]:
    probability = sigmoid_once(logits).reshape(-1)
    truth = (np.asarray(target).reshape(-1) > 0).astype(np.float64)
    if probability.size != truth.size:
        raise ValueError("logits and target must have identical sizes")
    brier = float(np.mean((probability - truth) ** 2)) if truth.size else 0.0
    edges = np.linspace(0.0, 1.0, int(bin_count) + 1)
    bins = []
    ece = 0.0
    for index in range(int(bin_count)):
        lower, upper = float(edges[index]), float(edges[index + 1])
        mask = ((probability >= lower) & (probability < upper))
        if index == bin_count - 1:
            mask = (probability >= lower) & (probability <= upper)
        count = int(np.count_nonzero(mask))
        confidence = float(probability[mask].mean()) if count else 0.0
        accuracy = float(truth[mask].mean()) if count else 0.0
        if truth.size:
            ece += (count / truth.size) * abs(confidence - accuracy)
        bins.append({
            "lower": lower,
            "upper": upper,
            "count": count,
            "mean_probability": confidence,
            "empirical_positive_rate": accuracy,
        })
    prevalence = float(truth.mean()) if truth.size else 0.0
    return {
        "brier_score": brier,
        "ece": float(ece),
        "ece_bin_count": int(bin_count),
        "bins": bins,
        "prevalence": prevalence,
        "prevalence_auprc_baseline": prevalence,
    }


def threshold_sweep(
    logits: np.ndarray,
    target: np.ndarray,
    thresholds: Iterable[float] = DEFAULT_THRESHOLD_GRID,
) -> list[dict[str, Any]]:
    probability = sigmoid_once(logits)
    truth = np.asarray(target) > 0
    rows = []
    for value in thresholds:
        threshold = float(value)
        prediction = probability >= threshold
        tp = int(np.count_nonzero(prediction & truth))
        fp = int(np.count_nonzero(prediction & ~truth))
        fn = int(np.count_nonzero(~prediction & truth))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "threshold": threshold,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "predicted_positive_count": int(tp + fp),
        })
    return rows


def detect_double_sigmoid_input(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    finite = bool(np.isfinite(array).all())
    probability_like = bool(
        finite and array.size and float(array.min()) >= 0.0 and float(array.max()) <= 1.0)
    return {
        "finite": finite,
        "input_range": [float(array.min()), float(array.max())] if array.size else None,
        "probability_like_input": probability_like,
        "double_sigmoid_risk": probability_like,
    }


def anchor_reference_metrics(
    logits: np.ndarray,
    target: np.ndarray,
    end_indices: Sequence[int],
    *,
    threshold: float = 0.3,
    top_k: int = 10,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    values = np.asarray(logits, dtype=np.float64)
    truth_all = np.asarray(target) > 0
    if values.shape != truth_all.shape or values.ndim != 4:
        raise ValueError("anchor logits and target must be equal BCHW arrays")
    if len(end_indices) != values.shape[0]:
        raise ValueError("end_indices length must equal batch dimension")
    probability = sigmoid_once(values)
    prediction = probability >= float(threshold)
    step_hits = np.zeros(values.shape[1], dtype=np.int64)
    step_totals = np.zeros(values.shape[1], dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for sample_index, end_index in enumerate(end_indices):
        for step in range(min(int(end_index), values.shape[1])):
            truth = truth_all[sample_index, step]
            if not np.any(truth):
                continue
            flat = probability[sample_index, step].reshape(-1)
            k = min(int(top_k), flat.size)
            top_indices = np.argpartition(flat, -k)[-k:]
            top_hit = bool(np.any(truth.reshape(-1)[top_indices]))
            threshold_hit = bool(np.any(prediction[sample_index, step] & truth))
            peak = np.unravel_index(int(np.argmax(flat)), truth.shape)
            truth_yx = np.argwhere(truth)
            distance = float(np.min(np.sqrt(np.sum(
                (truth_yx - np.asarray(peak)[None, :]) ** 2, axis=1))))
            false_positive_pixels = int(np.count_nonzero(
                prediction[sample_index, step] & ~truth))
            predicted_positive = int(np.count_nonzero(prediction[sample_index, step]))
            rows.append({
                "target_index": len(rows),
                "sample_index": int(sample_index),
                "anchor_step": int(step),
                "target_pixel_count": int(np.count_nonzero(truth)),
                "predicted_positive_count": predicted_positive,
                "threshold_hit": threshold_hit,
                "top_k_hit": top_hit,
                "peak_probability": float(flat.max()),
                "peak_yx": [int(peak[0]), int(peak[1])],
                "localization_error": distance,
                "false_positive_pixel_count": false_positive_pixels,
            })
            step_totals[step] += 1
            step_hits[step] += int(threshold_hit)
    diversity = [
        float(np.mean(np.abs(probability[:, left] - probability[:, right])))
        for left in range(values.shape[1])
        for right in range(left + 1, values.shape[1])
    ]
    count = len(rows)
    metrics = {
        "per_step_recall": [
            float(step_hits[index] / step_totals[index]) if step_totals[index] else 0.0
            for index in range(values.shape[1])
        ],
        "per_step_target_count": [int(value) for value in step_totals],
        "top_k": int(top_k),
        "top_k_recall": float(sum(row["top_k_hit"] for row in rows) / count) if count else 0.0,
        "localization_error": float(np.mean([
            row["localization_error"] for row in rows])) if count else 0.0,
        "false_positive_count": int(sum(
            row["false_positive_pixel_count"] for row in rows)),
        "missed_branch_count": int(sum(not row["threshold_hit"] for row in rows)),
        "channel_diversity_mean_absolute_difference": float(np.mean(diversity)) if diversity else 0.0,
        "evaluated_target_count": count,
        "fixed_threshold": float(threshold),
        "top_k_bypasses_fixed_threshold": True,
        "localization_error_scope": "all evaluated targets using global heatmap argmax",
        "false_positive_unit": "thresholded pixels outside target mask",
    }
    return metrics, rows


def numeric_max_abs_difference(left: Any, right: Any) -> float:
    differences: list[float] = []
    def visit(a: Any, b: Any) -> None:
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            for key in set(a) & set(b):
                visit(a[key], b[key])
        elif isinstance(a, Sequence) and not isinstance(a, (str, bytes)) \
                and isinstance(b, Sequence) and not isinstance(b, (str, bytes)):
            for av, bv in zip(a, b):
                visit(av, bv)
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            differences.append(abs(float(a) - float(b)))
    visit(left, right)
    return max(differences, default=0.0)


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def pixel_array_record(path: str | Path) -> dict[str, Any]:
    from PIL import Image
    source = Path(path)
    with source.open("rb") as handle:
        byte_sha = hashlib.sha256(handle.read()).hexdigest()
    with Image.open(source) as image:
        metadata = {
            "format": image.format,
            "mode": image.mode,
            "info": {str(key): str(value) for key, value in sorted(image.info.items())},
        }
        array = np.asarray(image).copy()
    return {
        "size_bytes": source.stat().st_size,
        "byte_sha256": byte_sha,
        "decoded_shape": [int(value) for value in array.shape],
        "decoded_dtype": str(array.dtype),
        "decoded_pixel_array_sha256": array_sha256(array),
        "metadata": metadata,
    }


def select_best_validation_record(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    candidates = [row for row in records if row.get("kind") == "frozen_validation"]
    if not candidates:
        raise ValueError("no frozen validation records")
    return max(candidates, key=lambda row: (
        float(row["metrics"]["segmentation_composite"]),
        -int(row["step"]),
    ))


def validate_recorded_best_step(
    records: Sequence[Mapping[str, Any]],
    checkpoint_step: int,
) -> dict[str, Any]:
    selected = select_best_validation_record(records)
    expected = int(selected["step"])
    return {
        "status": "PASS" if expected == int(checkpoint_step) else "FAIL",
        "recomputed_best_step": expected,
        "checkpoint_step": int(checkpoint_step),
        "recomputed_best_composite": float(
            selected["metrics"]["segmentation_composite"]),
    }


def classify_final_metrics_scope(trace: Mapping[str, Any]) -> str:
    """Classify the S3 summary final_metrics field from explicit trace facts."""
    if trace.get("target_prediction_alias"):
        return "TARGET_PREDICTION_ALIAS_BUG"
    if trace.get("accumulator_reset"):
        return "ACCUMULATION_RESET_BUG"
    if trace.get("computation_bug"):
        return "OTHER_COMPUTATION_BUG"
    if (trace.get("dataset_split") == "train"
            and int(trace.get("batch_count", 0)) == 1
            and trace.get("cross_batch_accumulation") is False):
        return "LAST_TRAIN_BATCH_METRICS"
    if trace.get("dataset_split") == "validation":
        return "VALID_VALIDATION_METRICS"
    if trace.get("dataset_split") == "train":
        return "VALID_TRAINING_METRICS_BUT_MISNAMED"
    return "UNRESOLVED"


def protocol_checkpoint(
    inventory: Mapping[str, Sequence[Mapping[str, Any]]],
    run_key: str,
    step: int,
) -> dict[str, Any]:
    matches = [row for row in inventory.get(run_key, []) if int(row["step"]) == int(step)]
    if not matches:
        return {
            "status": "UNAVAILABLE_MISSING_CHECKPOINT",
            "run_key": run_key,
            "requested_step": int(step),
        }
    preferred = sorted(matches, key=lambda row: row.get("kind") != "latest")[0]
    return {"status": "AVAILABLE", **dict(preferred)}


def graph_control_matrix(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    indexed = {
        (str(row["run_key"]), str(row["checkpoint_kind"])): row
        for row in records if row.get("status") == "PASS"
    }
    protocols = {}
    for kind in ("best", "latest"):
        missing = [key for key in ("C0", "C1", "C2", "C3")
                   if (key, kind) not in indexed]
        if missing:
            protocols[kind] = {"status": "INCOMPLETE", "missing_runs": missing}
            continue
        rows = {key: indexed[(key, kind)] for key in ("C0", "C1", "C2", "C3")}
        protocols[kind] = {
            "status": "PASS",
            "runs": rows,
            "c1_deltas": {
                baseline: {
                    metric: float(rows["C1"][metric]) - float(rows[baseline][metric])
                    for metric in ("apls", "topo")
                }
                for baseline in ("C0", "C2", "C3")
            },
        }
    return {"status": "PASS" if all(
        value["status"] == "PASS" for value in protocols.values()) else "INCOMPLETE",
        "protocols": protocols}


def paired_bootstrap_delta(
    left: Sequence[float],
    right: Sequence[float],
    *,
    seed: int = 20260827,
    iterations: int = 10000,
) -> dict[str, Any]:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or not a.size:
        raise ValueError("paired samples must be equal non-empty vectors")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, a.size, size=(int(iterations), a.size))
    deltas = np.mean(b[indices] - a[indices], axis=1)
    return {
        "paired_sample_count": int(a.size),
        "iterations": int(iterations),
        "seed": int(seed),
        "mean_delta": float(np.mean(b - a)),
        "bootstrap_mean_delta": float(deltas.mean()),
        "percentile_95_interval": [
            float(np.quantile(deltas, 0.025)),
            float(np.quantile(deltas, 0.975)),
        ],
    }


def leave_one_out_deltas(
    left: Sequence[float],
    right: Sequence[float],
) -> dict[str, Any]:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or a.size < 2:
        raise ValueError("leave-one-out requires equal vectors with at least two samples")
    values = [float(np.mean(np.delete(b - a, index))) for index in range(a.size)]
    full = float(np.mean(b - a))
    return {
        "full_mean_delta": full,
        "leave_one_out_deltas": values,
        "minimum": min(values),
        "maximum": max(values),
        "sign_reversal": any(value == 0 or math.copysign(1.0, value) != math.copysign(1.0, full)
                             for value in values) if full else any(value != 0 for value in values),
    }


def finite_json_dumps(value: Any, *, indent: int = 2) -> str:
    return json.dumps(value, indent=indent, ensure_ascii=False, allow_nan=False) + "\n"
