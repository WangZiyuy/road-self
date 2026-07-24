"""Threshold-free and collapse diagnostics for Stage 3C branch queries."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def distribution_statistics(values: Iterable[float]) -> Dict[str, object]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0, "mean": None, "median": None,
            "p10": None, "p90": None,
        }
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p90": float(np.quantile(array, 0.90)),
    }


def binary_average_precision(
    scores: np.ndarray,
    labels: np.ndarray,
) -> float:
    scores, labels = _binary_inputs(scores, labels)
    positive_count = int(labels.sum())
    if positive_count == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order].astype(np.float64)
    precision = np.cumsum(sorted_labels) / np.arange(
        1, sorted_labels.size + 1, dtype=np.float64)
    return float(np.sum(precision * sorted_labels) / positive_count)


def binary_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC with exact tie handling via grouped trapezoids."""

    scores, labels = _binary_inputs(scores, labels)
    positive_count = int(labels.sum())
    negative_count = int(labels.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ordered_scores = scores[order]
    ordered_labels = labels[order]
    true_positive = [0.0]
    false_positive = [0.0]
    cursor = 0
    while cursor < ordered_scores.size:
        end = cursor + 1
        while (
                end < ordered_scores.size
                and ordered_scores[end] == ordered_scores[cursor]):
            end += 1
        group = ordered_labels[cursor:end]
        true_positive.append(
            true_positive[-1] + float(group.sum()))
        false_positive.append(
            false_positive[-1] + float(group.size - group.sum()))
        cursor = end
    tpr = np.asarray(true_positive) / positive_count
    fpr = np.asarray(false_positive) / negative_count
    return float(np.trapz(tpr, fpr))


def calibration_curve(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    bin_count: int = 15,
) -> Dict[str, object]:
    probabilities, labels = _binary_inputs(probabilities, labels)
    if bin_count <= 0:
        raise ValueError("bin_count must be positive")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("calibration probabilities must be in [0, 1]")
    edges = np.linspace(0.0, 1.0, bin_count + 1)
    # A probability of exactly one belongs to the final bin.
    indices = np.minimum(
        np.searchsorted(edges, probabilities, side="right") - 1,
        bin_count - 1,
    )
    indices = np.maximum(indices, 0)
    counts = np.zeros(bin_count, dtype=np.int64)
    confidence = np.full(bin_count, np.nan, dtype=np.float64)
    accuracy = np.full(bin_count, np.nan, dtype=np.float64)
    for bin_index in range(bin_count):
        selected = indices == bin_index
        counts[bin_index] = int(selected.sum())
        if counts[bin_index]:
            confidence[bin_index] = float(
                probabilities[selected].mean())
            accuracy[bin_index] = float(labels[selected].mean())
    nonempty = counts > 0
    ece = float(np.sum(
        counts[nonempty] / max(probabilities.size, 1)
        * np.abs(confidence[nonempty] - accuracy[nonempty])
    ))
    return {
        "bin_edges": edges,
        "count": counts,
        "mean_probability": confidence,
        "positive_fraction": accuracy,
        "ece": ece,
    }


def branch_precision_recall_curve(
    scores: np.ndarray,
    pred_offsets_norm: np.ndarray,
    pred_directions: np.ndarray,
    target_offsets_norm: np.ndarray,
    target_directions: np.ndarray,
    target_mask: np.ndarray,
    *,
    window_size: float,
    endpoint_threshold_pixels: float,
    direction_threshold_degrees: float,
) -> Dict[str, object]:
    """Compute score-ranked branch AP with per-node one-to-one GT use."""

    scores = np.asarray(scores, dtype=np.float64)
    pred_offsets_norm = np.asarray(
        pred_offsets_norm, dtype=np.float64)
    pred_directions = np.asarray(pred_directions, dtype=np.float64)
    target_offsets_norm = np.asarray(
        target_offsets_norm, dtype=np.float64)
    target_directions = np.asarray(
        target_directions, dtype=np.float64)
    target_mask = np.asarray(target_mask, dtype=bool)
    if scores.ndim != 2:
        raise ValueError("scores must have shape [S, Q]")
    sample_count, query_count = scores.shape
    if pred_offsets_norm.shape != (sample_count, query_count, 2):
        raise ValueError("pred_offsets_norm must have shape [S, Q, 2]")
    if pred_directions.shape != pred_offsets_norm.shape:
        raise ValueError("pred_directions shape differs from offsets")
    if (
            target_offsets_norm.ndim != 3
            or target_offsets_norm.shape[0] != sample_count
            or target_offsets_norm.shape[2] != 2):
        raise ValueError(
            "target_offsets_norm must have shape [S, M, 2]")
    if target_directions.shape != target_offsets_norm.shape:
        raise ValueError("target direction shape differs from offsets")
    if target_mask.shape != target_offsets_norm.shape[:2]:
        raise ValueError("target_mask must have shape [S, M]")
    if window_size <= 0.0:
        raise ValueError("window_size must be positive")

    sample_ids = np.repeat(np.arange(sample_count), query_count)
    query_ids = np.tile(np.arange(query_count), sample_count)
    flat_scores = scores.reshape(-1)
    # Stable deterministic ordering: score, sample, then query.
    order = np.lexsort(
        (query_ids, sample_ids, -flat_scores))
    used = [
        np.zeros(target_mask[index].shape, dtype=bool)
        for index in range(sample_count)
    ]
    true_positive = np.zeros(order.size, dtype=np.float64)
    half_window = float(window_size) / 2.0
    for rank, flat_index in enumerate(order):
        sample_index = int(sample_ids[flat_index])
        query_index = int(query_ids[flat_index])
        available = np.flatnonzero(
            target_mask[sample_index] & ~used[sample_index])
        if available.size == 0:
            continue
        endpoint_error = np.linalg.norm(
            pred_offsets_norm[sample_index, query_index]
            - target_offsets_norm[sample_index, available],
            axis=-1,
        ) * half_window
        direction_error = _direction_errors_degrees(
            np.broadcast_to(
                pred_directions[sample_index, query_index],
                (available.size, 2),
            ),
            target_directions[sample_index, available],
        )
        valid = (
            (endpoint_error <= endpoint_threshold_pixels)
            & (direction_error <= direction_threshold_degrees)
        )
        if not np.any(valid):
            continue
        valid_positions = np.flatnonzero(valid)
        normalized_cost = (
            endpoint_error[valid_positions]
            / max(endpoint_threshold_pixels, np.finfo(float).eps)
            + direction_error[valid_positions]
            / max(direction_threshold_degrees, np.finfo(float).eps)
        )
        chosen_position = valid_positions[
            int(np.argmin(normalized_cost))]
        chosen_target = int(available[chosen_position])
        used[sample_index][chosen_target] = True
        true_positive[rank] = 1.0

    false_positive = 1.0 - true_positive
    cumulative_tp = np.cumsum(true_positive)
    cumulative_fp = np.cumsum(false_positive)
    total_gt = int(target_mask.sum())
    precision = cumulative_tp / np.maximum(
        cumulative_tp + cumulative_fp, 1.0)
    recall = (
        cumulative_tp / total_gt
        if total_gt > 0
        else np.zeros_like(cumulative_tp)
    )
    # Precision-envelope AP makes the metric independent of a chosen score
    # threshold while retaining the existing TP geometry thresholds.
    if total_gt > 0 and precision.size:
        envelope = np.maximum.accumulate(precision[::-1])[::-1]
        recall_previous = np.concatenate(([0.0], recall[:-1]))
        average_precision = float(np.sum(
            (recall - recall_previous) * envelope))
    else:
        average_precision = 0.0
    return {
        "average_precision": average_precision,
        "precision": precision,
        "recall": recall,
        "scores": flat_scores[order],
        "true_positive": true_positive.astype(np.int8),
        "sample_indices": sample_ids[order],
        "query_indices": query_ids[order],
        "gt_branch_count": total_gt,
    }


def duplicate_statistics(
    pred_offsets_norm: np.ndarray,
    pred_directions: np.ndarray,
    selected_query_indices: Sequence[np.ndarray],
    *,
    window_size: float,
    endpoint_threshold_pixels: float,
    direction_threshold_degrees: float,
) -> Dict[str, object]:
    pred_offsets_norm = np.asarray(
        pred_offsets_norm, dtype=np.float64)
    pred_directions = np.asarray(pred_directions, dtype=np.float64)
    pair_count = 0
    duplicate_pair_count = 0
    nodes_with_duplicates = 0
    nodes_with_multiple = 0
    for sample_index, raw_indices in enumerate(selected_query_indices):
        indices = np.asarray(raw_indices, dtype=np.int64)
        if indices.size < 2:
            continue
        nodes_with_multiple += 1
        node_has_duplicate = False
        for first_position in range(indices.size):
            for second_position in range(
                    first_position + 1, indices.size):
                pair_count += 1
                first = int(indices[first_position])
                second = int(indices[second_position])
                endpoint_error = float(np.linalg.norm(
                    pred_offsets_norm[sample_index, first]
                    - pred_offsets_norm[sample_index, second]
                ) * (float(window_size) / 2.0))
                direction_error = float(_direction_errors_degrees(
                    pred_directions[sample_index, first][None, :],
                    pred_directions[sample_index, second][None, :],
                )[0])
                if (
                        endpoint_error <= endpoint_threshold_pixels
                        and direction_error
                        <= direction_threshold_degrees):
                    duplicate_pair_count += 1
                    node_has_duplicate = True
        nodes_with_duplicates += int(node_has_duplicate)
    sample_count = len(selected_query_indices)
    return {
        "duplicate_pair_count": duplicate_pair_count,
        "pair_count": pair_count,
        "duplicate_pair_ratio": (
            duplicate_pair_count / pair_count
            if pair_count else 0.0
        ),
        "nodes_with_duplicates": nodes_with_duplicates,
        "nodes_with_multiple_predictions": nodes_with_multiple,
        "nodes_with_duplicates_ratio": (
            nodes_with_duplicates / sample_count
            if sample_count else 0.0
        ),
    }


def oracle_k_metrics(
    scores: np.ndarray,
    pred_offsets_norm: np.ndarray,
    pred_directions: np.ndarray,
    target_offsets_norm: np.ndarray,
    target_directions: np.ndarray,
    target_mask: np.ndarray,
    *,
    window_size: float,
    endpoint_threshold_pixels: float,
    direction_threshold_degrees: float,
    duplicate_endpoint_threshold_pixels: float,
    duplicate_direction_threshold_degrees: float,
) -> Dict[str, object]:
    """Evaluate the top-GT-count query slots without a score threshold."""

    scores = np.asarray(scores, dtype=np.float64)
    pred_offsets_norm = np.asarray(
        pred_offsets_norm, dtype=np.float64)
    pred_directions = np.asarray(pred_directions, dtype=np.float64)
    target_offsets_norm = np.asarray(
        target_offsets_norm, dtype=np.float64)
    target_directions = np.asarray(
        target_directions, dtype=np.float64)
    target_mask = np.asarray(target_mask, dtype=bool)
    selected = []
    endpoint_errors = []
    direction_errors = []
    true_positive = 0
    total_gt = int(target_mask.sum())
    covered_gt = 0
    for sample_index in range(scores.shape[0]):
        valid_targets = np.flatnonzero(target_mask[sample_index])
        count = int(valid_targets.size)
        order = np.lexsort((
            np.arange(scores.shape[1]),
            -scores[sample_index],
        ))
        query_indices = order[:count]
        selected.append(query_indices)
        if count == 0:
            continue
        endpoint_l1 = np.abs(
            pred_offsets_norm[sample_index, query_indices, None, :]
            - target_offsets_norm[
                sample_index, valid_targets[None, :], :]
        ).sum(axis=-1)
        pred_axis = _normalize_vectors(
            pred_directions[sample_index, query_indices])
        target_axis = _normalize_vectors(
            target_directions[sample_index, valid_targets])
        direction_cost = 1.0 - np.matmul(
            pred_axis, target_axis.T)
        rows, columns = linear_sum_assignment(
            endpoint_l1 + direction_cost)
        chosen_queries = query_indices[rows]
        chosen_targets = valid_targets[columns]
        endpoint = np.linalg.norm(
            pred_offsets_norm[sample_index, chosen_queries]
            - target_offsets_norm[sample_index, chosen_targets],
            axis=-1,
        ) * (float(window_size) / 2.0)
        direction = _direction_errors_degrees(
            pred_directions[sample_index, chosen_queries],
            target_directions[sample_index, chosen_targets],
        )
        endpoint_errors.extend(endpoint.tolist())
        direction_errors.extend(direction.tolist())
        valid = (
            (endpoint <= endpoint_threshold_pixels)
            & (direction <= direction_threshold_degrees)
        )
        true_positive += int(valid.sum())
        covered_gt += int(np.unique(chosen_targets[valid]).size)
    duplicates = duplicate_statistics(
        pred_offsets_norm,
        pred_directions,
        selected,
        window_size=window_size,
        endpoint_threshold_pixels=duplicate_endpoint_threshold_pixels,
        direction_threshold_degrees=(
            duplicate_direction_threshold_degrees),
    )
    return {
        "endpoint_error": distribution_statistics(endpoint_errors),
        "direction_error": distribution_statistics(direction_errors),
        "true_positive": true_positive,
        "gt_branch_count": total_gt,
        "recall": true_positive / total_gt if total_gt else 0.0,
        "distinct_gt_coverage": (
            covered_gt / total_gt if total_gt else 0.0),
        "duplicates": duplicates,
        "selected_query_indices": selected,
    }


def query_pairwise_statistics(
    hidden_states: np.ndarray,
) -> Dict[str, object]:
    hidden_states = np.asarray(hidden_states, dtype=np.float64)
    if hidden_states.ndim != 3:
        raise ValueError("hidden states must have shape [B, Q, D]")
    norms = np.linalg.norm(hidden_states, axis=-1)
    normalized = hidden_states / np.maximum(
        norms[..., None], np.finfo(float).eps)
    similarities = np.matmul(
        normalized, np.swapaxes(normalized, 1, 2))
    query_count = hidden_states.shape[1]
    upper = np.triu_indices(query_count, k=1)
    pair_values = similarities[:, upper[0], upper[1]].reshape(-1)
    return {
        "pairwise_cosine": distribution_statistics(pair_values),
        "hidden_norm": distribution_statistics(norms.reshape(-1)),
    }


def _binary_inputs(
    scores: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).reshape(-1).astype(bool)
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must have the same size")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores contain NaN or Inf")
    return scores, labels


def _direction_errors_degrees(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first_norm = first / np.maximum(
        np.linalg.norm(first, axis=-1, keepdims=True),
        1e-12,
    )
    second_norm = second / np.maximum(
        np.linalg.norm(second, axis=-1, keepdims=True),
        1e-12,
    )
    cosine = np.clip(np.sum(first_norm * second_norm, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def _normalize_vectors(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(
        np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)
