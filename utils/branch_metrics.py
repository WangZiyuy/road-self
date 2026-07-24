"""Hungarian-matched validation metrics for unordered branch predictions."""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from utils.branch_diagnostics import branch_precision_recall_curve


class BranchMetricAccumulator:
    """Accumulate thresholded detection and continuous geometry metrics."""

    def __init__(
        self,
        *,
        window_size: float,
        existence_threshold: float = 0.5,
        endpoint_match_threshold_pixels: float = 20.0,
        direction_match_threshold_degrees: float = 45.0,
        duplicate_endpoint_threshold_pixels: float = 12.0,
        duplicate_direction_threshold_degrees: float = 25.0,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if not 0.0 <= existence_threshold <= 1.0:
            raise ValueError("existence_threshold must be in [0, 1]")
        thresholds = (
            endpoint_match_threshold_pixels,
            direction_match_threshold_degrees,
            duplicate_endpoint_threshold_pixels,
            duplicate_direction_threshold_degrees,
        )
        if any(value < 0.0 for value in thresholds):
            raise ValueError("metric thresholds must be non-negative")
        self.half_window = float(window_size) / 2.0
        self.existence_threshold = float(existence_threshold)
        self.endpoint_match_threshold_pixels = float(
            endpoint_match_threshold_pixels)
        self.direction_match_threshold_degrees = float(
            direction_match_threshold_degrees)
        self.duplicate_endpoint_threshold_pixels = float(
            duplicate_endpoint_threshold_pixels)
        self.duplicate_direction_threshold_degrees = float(
            duplicate_direction_threshold_degrees)
        self.sample_count = 0
        self.exact_count_correct = 0
        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0
        self.predicted_branch_count = 0
        self.gt_branch_count = 0
        self.endpoint_errors: List[float] = []
        self.direction_errors: List[float] = []
        self.duplicate_pair_count = 0
        self.predicted_pair_count = 0
        self.nodes_with_duplicates = 0
        self.nodes_with_multiple_predictions = 0

    @staticmethod
    def _direction_error_degrees(
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> torch.Tensor:
        first = F.normalize(first, p=2, dim=-1, eps=1e-6)
        second = F.normalize(second, p=2, dim=-1, eps=1e-6)
        cosine = (first * second).sum(dim=-1).clamp(-1.0, 1.0)
        return torch.rad2deg(torch.acos(cosine))

    def update(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> None:
        probabilities = torch.sigmoid(
            predictions["branch_exist_logits"]).detach().cpu()
        pred_offsets = predictions[
            "branch_offsets_norm"].detach().cpu()
        pred_directions = predictions[
            "branch_directions"].detach().cpu()
        target_offsets = targets["branch_offsets_norm"].detach().cpu()
        target_directions = targets["branch_directions"].detach().cpu()
        target_mask = targets["branch_mask"].detach().cpu().to(
            dtype=torch.bool)
        batch_size = probabilities.shape[0]

        for batch_index in range(batch_size):
            predicted_indices = torch.nonzero(
                probabilities[batch_index] >= self.existence_threshold,
                as_tuple=False,
            ).flatten()
            target_indices = torch.nonzero(
                target_mask[batch_index], as_tuple=False).flatten()
            predicted_count = int(predicted_indices.numel())
            target_count = int(target_indices.numel())
            self.sample_count += 1
            self.predicted_branch_count += predicted_count
            self.gt_branch_count += target_count
            self.exact_count_correct += int(
                predicted_count == target_count)

            selected_pred_offsets = pred_offsets[
                batch_index].index_select(0, predicted_indices)
            selected_pred_directions = pred_directions[
                batch_index].index_select(0, predicted_indices)
            selected_target_offsets = target_offsets[
                batch_index].index_select(0, target_indices)
            selected_target_directions = target_directions[
                batch_index].index_select(0, target_indices)

            node_duplicate_count = 0
            if predicted_count >= 2:
                self.nodes_with_multiple_predictions += 1
                for first_index in range(predicted_count):
                    for second_index in range(
                            first_index + 1, predicted_count):
                        self.predicted_pair_count += 1
                        endpoint_error = torch.linalg.vector_norm(
                            selected_pred_offsets[first_index]
                            - selected_pred_offsets[second_index]
                        ) * self.half_window
                        direction_error = self._direction_error_degrees(
                            selected_pred_directions[first_index],
                            selected_pred_directions[second_index],
                        )
                        if (
                                float(endpoint_error)
                                <= self.duplicate_endpoint_threshold_pixels
                                and float(direction_error)
                                <= self.duplicate_direction_threshold_degrees):
                            self.duplicate_pair_count += 1
                            node_duplicate_count += 1
                self.nodes_with_duplicates += int(
                    node_duplicate_count > 0)

            if predicted_count == 0 or target_count == 0:
                self.false_positive += predicted_count
                self.false_negative += target_count
                continue

            endpoint_cost = torch.cdist(
                selected_pred_offsets,
                selected_target_offsets,
                p=1,
            )
            normalized_pred = F.normalize(
                selected_pred_directions, dim=-1, eps=1e-6)
            normalized_target = F.normalize(
                selected_target_directions, dim=-1, eps=1e-6)
            direction_cost = 1.0 - torch.matmul(
                normalized_pred, normalized_target.transpose(0, 1))
            rows, columns = linear_sum_assignment(
                (endpoint_cost + direction_cost).numpy())
            matched_pred = torch.as_tensor(rows, dtype=torch.long)
            matched_target = torch.as_tensor(columns, dtype=torch.long)
            endpoint_errors = torch.linalg.vector_norm(
                selected_pred_offsets.index_select(0, matched_pred)
                - selected_target_offsets.index_select(0, matched_target),
                dim=-1,
            ) * self.half_window
            direction_errors = self._direction_error_degrees(
                selected_pred_directions.index_select(0, matched_pred),
                selected_target_directions.index_select(0, matched_target),
            )
            self.endpoint_errors.extend(
                float(value) for value in endpoint_errors)
            self.direction_errors.extend(
                float(value) for value in direction_errors)
            valid_matches = (
                (endpoint_errors <= self.endpoint_match_threshold_pixels)
                & (
                    direction_errors
                    <= self.direction_match_threshold_degrees
                )
            )
            true_positive = int(valid_matches.sum())
            self.true_positive += true_positive
            self.false_positive += predicted_count - true_positive
            self.false_negative += target_count - true_positive

    @staticmethod
    def _safe_ratio(numerator: int, denominator: int) -> float:
        return (
            float(numerator) / float(denominator)
            if denominator > 0
            else 0.0
        )

    @staticmethod
    def _distribution(values: List[float]):
        if not values:
            return {"mean": None, "median": None, "count": 0}
        array = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(np.mean(array)),
            "median": float(np.median(array)),
            "count": int(array.size),
        }

    def compute(self) -> Dict[str, float]:
        precision = self._safe_ratio(
            self.true_positive,
            self.true_positive + self.false_positive,
        )
        recall = self._safe_ratio(
            self.true_positive,
            self.true_positive + self.false_negative,
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0.0
            else 0.0
        )
        endpoint = self._distribution(self.endpoint_errors)
        direction = self._distribution(self.direction_errors)
        return {
            "sample_count": int(self.sample_count),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "exact_branch_count_accuracy": self._safe_ratio(
                self.exact_count_correct, self.sample_count),
            "endpoint_error_mean_pixels": endpoint["mean"],
            "endpoint_error_median_pixels": endpoint["median"],
            "endpoint_error_pair_count": endpoint["count"],
            "direction_error_mean_degrees": direction["mean"],
            "direction_error_median_degrees": direction["median"],
            "direction_error_pair_count": direction["count"],
            "missed_branch_rate": self._safe_ratio(
                self.false_negative, self.gt_branch_count),
            "extra_branch_rate": self._safe_ratio(
                self.false_positive, self.predicted_branch_count),
            "duplicate_query_pair_ratio": self._safe_ratio(
                self.duplicate_pair_count, self.predicted_pair_count),
            "nodes_with_duplicate_ratio": self._safe_ratio(
                self.nodes_with_duplicates, self.sample_count),
            "true_positive": int(self.true_positive),
            "false_positive": int(self.false_positive),
            "false_negative": int(self.false_negative),
            "predicted_branch_count": int(self.predicted_branch_count),
            "gt_branch_count": int(self.gt_branch_count),
        }


class BranchAveragePrecisionAccumulator:
    """Accumulate predictions for threshold-free branch AP/PR."""

    def __init__(
        self,
        *,
        window_size: float,
        endpoint_match_threshold_pixels: float,
        direction_match_threshold_degrees: float,
    ) -> None:
        self.window_size = float(window_size)
        self.endpoint_match_threshold_pixels = float(
            endpoint_match_threshold_pixels)
        self.direction_match_threshold_degrees = float(
            direction_match_threshold_degrees)
        self.scores = []
        self.pred_offsets = []
        self.pred_directions = []
        self.target_offsets = []
        self.target_directions = []
        self.target_masks = []

    def update(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> None:
        self.scores.append(torch.sigmoid(
            predictions["branch_exist_logits"]
        ).detach().cpu().numpy())
        self.pred_offsets.append(
            predictions["branch_offsets_norm"].detach().cpu().numpy())
        self.pred_directions.append(
            predictions["branch_directions"].detach().cpu().numpy())
        self.target_offsets.append(
            targets["branch_offsets_norm"].detach().cpu().numpy())
        self.target_directions.append(
            targets["branch_directions"].detach().cpu().numpy())
        self.target_masks.append(
            targets["branch_mask"].detach().cpu().numpy())

    def compute(self) -> Dict[str, object]:
        if not self.scores:
            return {
                "average_precision": 0.0,
                "gt_branch_count": 0,
                "precision": np.zeros(0, dtype=np.float64),
                "recall": np.zeros(0, dtype=np.float64),
                "scores": np.zeros(0, dtype=np.float64),
            }
        return branch_precision_recall_curve(
            np.concatenate(self.scores, axis=0),
            np.concatenate(self.pred_offsets, axis=0),
            np.concatenate(self.pred_directions, axis=0),
            np.concatenate(self.target_offsets, axis=0),
            np.concatenate(self.target_directions, axis=0),
            np.concatenate(self.target_masks, axis=0),
            window_size=self.window_size,
            endpoint_threshold_pixels=(
                self.endpoint_match_threshold_pixels),
            direction_threshold_degrees=(
                self.direction_match_threshold_degrees),
        )
