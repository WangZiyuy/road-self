"""Streaming heatmap and original-decoder localization metrics for Stage 3F-A."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from lib import geom
from utils.model_utils import map_to_coordinate


class PixelHistogramMetrics:
    """Bounded-memory pixel AP/AUROC/calibration accumulator."""

    def __init__(self, bins: int = 4096) -> None:
        self.bins = int(bins)
        self.positive = np.zeros(self.bins, dtype=np.int64)
        self.negative = np.zeros(self.bins, dtype=np.int64)
        self.positive_probability_sum = 0.0
        self.negative_probability_sum = 0.0
        self.positive_count = 0
        self.negative_count = 0

    def update(self, probabilities: np.ndarray, targets: np.ndarray) -> None:
        scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        labels = np.asarray(targets).reshape(-1) >= 0.5
        indices = np.minimum(
            (np.clip(scores, 0.0, 1.0) * self.bins).astype(np.int64),
            self.bins - 1)
        self.positive += np.bincount(
            indices[labels], minlength=self.bins)
        self.negative += np.bincount(
            indices[~labels], minlength=self.bins)
        self.positive_probability_sum += float(scores[labels].sum())
        self.negative_probability_sum += float(scores[~labels].sum())
        self.positive_count += int(labels.sum())
        self.negative_count += int((~labels).sum())

    def compute(self) -> Dict[str, float]:
        tp = np.cumsum(self.positive[::-1], dtype=np.float64)
        fp = np.cumsum(self.negative[::-1], dtype=np.float64)
        recall = tp / max(float(self.positive_count), 1.0)
        precision = tp / np.maximum(tp + fp, 1.0)
        previous = np.concatenate(([0.0], recall[:-1]))
        ap = float(np.sum((recall - previous) * precision))
        tpr = recall
        fpr = fp / max(float(self.negative_count), 1.0)
        auroc = float(np.trapz(tpr, fpr))
        calibration = []
        boundaries = np.linspace(0, self.bins, 16, dtype=np.int64)
        for index in range(15):
            start, end = boundaries[index], boundaries[index + 1]
            positives = int(self.positive[start:end].sum())
            negatives = int(self.negative[start:end].sum())
            count = positives + negatives
            calibration.append({
                "lower": start / self.bins,
                "upper": end / self.bins,
                "count": count,
                "positive_fraction": positives / max(count, 1),
                "mean_confidence_approx": (start + end - 1) / (2 * self.bins),
            })
        return {
            "pixel_ap": ap,
            "pixel_auroc": auroc,
            "positive_probability_mean": (
                self.positive_probability_sum / max(self.positive_count, 1)),
            "negative_probability_mean": (
                self.negative_probability_sum / max(self.negative_count, 1)),
            "positive_pixel_count": self.positive_count,
            "negative_pixel_count": self.negative_count,
            "histogram_bins": self.bins,
            "calibration_curve_15_bins": calibration,
        }


def _point_xy(point) -> np.ndarray:
    return np.asarray([float(point.x), float(point.y)], dtype=np.float64)


def decode_immediate_nodes(
    probabilities: np.ndarray,
    center_xy: np.ndarray,
    *, threshold: float, step_length: float,
    junction_max_region_area: int,
) -> List[np.ndarray]:
    """Reuse map_to_coordinate while limiting evaluation to immediate channel 0."""

    vertex = SimpleNamespace(point=geom.Point(
        float(center_xy[0]), float(center_xy[1])))
    results = map_to_coordinate(
        np.array(probabilities[None, :1], copy=True),
        [True], [vertex], ROAD_SEG_THRESHOLE=float(threshold),
        STEP_LENGTH=float(step_length),
        JUNC_MAX_REGION_AREA=int(junction_max_region_area))[0]
    return [_point_xy(point) for point in results]


def localization_record(
    *, probabilities: np.ndarray, center_xy: np.ndarray,
    gt_xy: np.ndarray, gt_mask: np.ndarray, threshold: float,
    step_length: float, junction_max_region_area: int,
    match_threshold: float, topk_oracle_k: int = 4,
) -> Dict[str, float]:
    predictions = decode_immediate_nodes(
        probabilities, center_xy, threshold=threshold,
        step_length=step_length,
        junction_max_region_area=junction_max_region_area)
    gt = np.asarray(gt_xy)[np.asarray(gt_mask, dtype=bool)]
    predicted = np.stack(predictions) if predictions else np.zeros((0, 2))
    if len(predicted):
        local = np.rint(predicted - center_xy + probabilities.shape[-1] / 2.0)
        local = np.clip(local.astype(np.int64), 0, probabilities.shape[-1] - 1)
        scores = probabilities[0, local[:, 0], local[:, 1]]
        predicted = predicted[np.argsort(scores, kind="stable")[::-1]]
    distances = (
        np.linalg.norm(predicted[:, None] - gt[None, :], axis=-1)
        if len(predicted) and len(gt) else np.zeros((len(predicted), len(gt))))
    matched = 0
    if distances.size:
        rows, columns = linear_sum_assignment(distances)
        matched = int(np.sum(distances[rows, columns] <= match_threshold))
    top1_error = float("nan")
    if len(predicted) and len(gt):
        top1_error = float(np.min(np.linalg.norm(gt - predicted[0], axis=1)))
    oracle_error = float("nan")
    if len(predicted) and len(gt):
        oracle_distances = distances[:max(1, int(topk_oracle_k))]
        oracle_error = float(np.mean(np.min(oracle_distances, axis=0)))
    return {
        "gt_count": int(len(gt)), "predicted_count": int(len(predicted)),
        "matched_count": matched,
        "exact_count": float(len(gt) == len(predicted)),
        "top1_endpoint_error": top1_error,
        "topk_oracle_endpoint_error": oracle_error,
        "hit_3": float(len(gt) > 0 and np.isfinite(top1_error) and top1_error <= 3.0),
        "hit_5": float(len(gt) > 0 and np.isfinite(top1_error) and top1_error <= 5.0),
        "hit_10": float(len(gt) > 0 and np.isfinite(top1_error) and top1_error <= 10.0),
        "missed_count": max(0, len(gt) - matched),
        "extra_count": max(0, len(predicted) - matched),
    }


def aggregate_localization(records: Sequence[Dict[str, float]]) -> Dict[str, float]:
    gt = sum(int(record["gt_count"]) for record in records)
    predicted = sum(int(record["predicted_count"]) for record in records)
    matched = sum(int(record["matched_count"]) for record in records)
    top1 = np.asarray([record["top1_endpoint_error"] for record in records], dtype=float)
    oracle = np.asarray([record["topk_oracle_endpoint_error"] for record in records], dtype=float)
    valid_top1 = top1[np.isfinite(top1)]
    valid_oracle = oracle[np.isfinite(oracle)]
    gt_records = [record for record in records if record["gt_count"] > 0]
    precision = matched / max(predicted, 1)
    recall = matched / max(gt, 1)
    return {
        "sample_count": len(records), "gt_next_node_count": gt,
        "predicted_anchor_count": predicted,
        "exact_next_node_count_accuracy": float(np.mean([
            record["exact_count"] for record in records])) if records else 0.0,
        "top1_endpoint_error_mean": float(valid_top1.mean()) if len(valid_top1) else float("nan"),
        "top1_endpoint_error_median": float(np.median(valid_top1)) if len(valid_top1) else float("nan"),
        "topk_oracle_endpoint_error_mean": float(valid_oracle.mean()) if len(valid_oracle) else float("nan"),
        "hit_at_3_px": float(np.mean([record["hit_3"] for record in gt_records])) if gt_records else 0.0,
        "hit_at_5_px": float(np.mean([record["hit_5"] for record in gt_records])) if gt_records else 0.0,
        "hit_at_10_px": float(np.mean([record["hit_10"] for record in gt_records])) if gt_records else 0.0,
        "missed_node_rate": sum(record["missed_count"] for record in records) / max(gt, 1),
        "extra_node_rate": sum(record["extra_count"] for record in records) / max(predicted, 1),
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }
