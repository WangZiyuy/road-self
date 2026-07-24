"""Diagnose Stage 3C existence separation and query collapse.

This script is evaluation-only.  It never calls Path.push and never changes
RPNet, branch targets, trajectory recall, or any checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.branch_set_loss import (  # noqa: E402
    branch_matching_cost_components,
    hungarian_match_branches,
)
from train_branch_aux import (  # noqa: E402
    MODALITY_FULL,
    MODALITY_NO_TRAJECTORY,
    MODALITY_TRAJECTORY_GRAPH,
    _build_auxiliary_modules,
    _build_branch_criterion,
    _build_optimizer,
    _forward_auxiliary,
    _load_config,
    _load_frozen_rpnet,
    _metric_accumulator,
    _move_nested,
    _resolve_device,
    _set_module_mode,
    _set_seed,
    _stage_fuse_for_batch,
)
from utils.branch_diagnostics import (  # noqa: E402
    binary_average_precision,
    binary_auroc,
    branch_precision_recall_curve,
    calibration_curve,
    distribution_statistics,
    duplicate_statistics,
    oracle_k_metrics,
    query_pairwise_statistics,
)
from utils.stage3c_branch_dataset import Stage3CBranchDataset  # noqa: E402
from utils.stage3c_checkpoint import load_stage3c_checkpoint  # noqa: E402


MODALITIES = (
    MODALITY_FULL,
    MODALITY_NO_TRAJECTORY,
    MODALITY_TRAJECTORY_GRAPH,
)
DEBUG_STATE_KEYS = (
    "debug_learned_query_embedding",
    "debug_pre_graph_queries",
    "debug_graph_conditioned_queries",
    "debug_pre_cross_attention_queries",
    "debug_image_cross_attention_output",
    "debug_trajectory_cross_attention_output",
    "debug_final_fused_queries",
    "debug_graph_state_contribution",
)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/stage3c_e0_current_checkpoint_diagnostics.yml"),
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--image-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--split", choices=("train", "val"), default="val")
    return parser.parse_args()


def _numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _concatenate(chunks: Sequence[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def _stats(values: Iterable[float]) -> Dict[str, object]:
    return distribution_statistics(np.asarray(list(values)).reshape(-1))


def _json_ready(value):
    if isinstance(value, Mapping):
        return {
            str(key): _json_ready(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(
            _json_ready(value),
            output,
            indent=2,
            sort_keys=True,
        )
        output.write("\n")


def _matches_to_mask(
    matches,
    batch_size: int,
    query_count: int,
) -> np.ndarray:
    mask = np.zeros((batch_size, query_count), dtype=bool)
    for sample_index, (queries, _) in enumerate(matches):
        mask[sample_index, _numpy(queries).astype(np.int64)] = True
    return mask


def _selected_from_matches(matches) -> List[np.ndarray]:
    return [
        _numpy(query_indices).astype(np.int64)
        for query_indices, _ in matches
    ]


def _cost_statistics(
    chunks: Mapping[str, List[np.ndarray]],
) -> Dict[str, object]:
    return {
        name: _stats(
            np.concatenate(values) if values else [])
        for name, values in chunks.items()
    }


def _group_name(count: int) -> str:
    if count == 0:
        return "count_0"
    if count == 1:
        return "count_1"
    if count == 2:
        return "count_2"
    if count >= 3:
        return "count_ge_3"
    return "count_ge_2"


def _subset_metrics(
    indices: np.ndarray,
    *,
    probabilities: np.ndarray,
    offsets: np.ndarray,
    directions: np.ndarray,
    target_offsets: np.ndarray,
    target_directions: np.ndarray,
    target_mask: np.ndarray,
    cfg,
    threshold: float = 0.5,
    actual_labels: np.ndarray = None,
) -> Dict[str, object]:
    if indices.size == 0:
        return {"sample_count": 0}
    evaluation = cfg.STAGE3C.EVALUATION
    oracle = oracle_k_metrics(
        probabilities[indices],
        offsets[indices],
        directions[indices],
        target_offsets[indices],
        target_directions[indices],
        target_mask[indices],
        window_size=float(cfg.TRAIN.WINDOW_SIZE),
        endpoint_threshold_pixels=float(
            evaluation.ENDPOINT_MATCH_THRESHOLD_PIXELS),
        direction_threshold_degrees=float(
            evaluation.DIRECTION_MATCH_THRESHOLD_DEGREES),
        duplicate_endpoint_threshold_pixels=float(
            evaluation.DUPLICATE_ENDPOINT_THRESHOLD_PIXELS),
        duplicate_direction_threshold_degrees=float(
            evaluation.DUPLICATE_DIRECTION_THRESHOLD_DEGREES),
    )
    oracle.pop("selected_query_indices")
    branch_pr = branch_precision_recall_curve(
        probabilities[indices],
        offsets[indices],
        directions[indices],
        target_offsets[indices],
        target_directions[indices],
        target_mask[indices],
        window_size=float(cfg.TRAIN.WINDOW_SIZE),
        endpoint_threshold_pixels=float(
            evaluation.ENDPOINT_MATCH_THRESHOLD_PIXELS),
        direction_threshold_degrees=float(
            evaluation.DIRECTION_MATCH_THRESHOLD_DEGREES),
    )
    threshold_evaluator = _metric_accumulator(
        cfg, existence_threshold=threshold)
    threshold_evaluator.update(
        {
            "branch_exist_logits": torch.from_numpy(
                np.log(
                    np.clip(
                        probabilities[indices],
                        1e-7,
                        1.0 - 1e-7,
                    )
                    / np.clip(
                        1.0 - probabilities[indices],
                        1e-7,
                        1.0,
                    )
                )
            ),
            "branch_offsets_norm": torch.from_numpy(
                offsets[indices]),
            "branch_directions": torch.from_numpy(
                directions[indices]),
        },
        {
            "branch_offsets_norm": torch.from_numpy(
                target_offsets[indices]),
            "branch_directions": torch.from_numpy(
                target_directions[indices]),
            "branch_mask": torch.from_numpy(
                target_mask[indices]),
        },
    )
    threshold_metrics = threshold_evaluator.compute()
    result = {
        "sample_count": int(indices.size),
        "gt_branch_count": int(target_mask[indices].sum()),
        "branch_ap": float(branch_pr["average_precision"]),
        "oracle_k": oracle,
        "thresholded_metrics": threshold_metrics,
    }
    if actual_labels is not None:
        labels = actual_labels[indices]
        subset_probabilities = probabilities[indices]
        result.update({
            "slot_ap": binary_average_precision(
                subset_probabilities, labels),
            "matched_probability": _stats(
                subset_probabilities[labels]),
            "unmatched_probability": _stats(
                subset_probabilities[~labels]),
        })
    return result


def _query_similarity_summary(debug_chunks) -> Dict[str, object]:
    result = {}
    for modality, stages in debug_chunks.items():
        result[modality] = {}
        for stage, chunks in stages.items():
            result[modality][stage] = query_pairwise_statistics(
                np.concatenate(chunks, axis=0))
    return result


def _modality_diagnostic_summary(
    *,
    values: Mapping[str, np.ndarray],
    actual_labels: np.ndarray,
    actual_selected: Sequence[np.ndarray],
    target_offsets: np.ndarray,
    target_directions: np.ndarray,
    target_mask: np.ndarray,
    gt_counts: np.ndarray,
    cfg,
    threshold: float,
) -> Dict[str, object]:
    """Compute the same validation metrics independently per modality."""

    probabilities = values["probability"]
    logits = values["logit"]
    offsets = values["offsets"]
    directions = values["directions"]
    evaluation = cfg.STAGE3C.EVALUATION
    branch_pr = branch_precision_recall_curve(
        probabilities,
        offsets,
        directions,
        target_offsets,
        target_directions,
        target_mask,
        window_size=float(cfg.TRAIN.WINDOW_SIZE),
        endpoint_threshold_pixels=float(
            evaluation.ENDPOINT_MATCH_THRESHOLD_PIXELS),
        direction_threshold_degrees=float(
            evaluation.DIRECTION_MATCH_THRESHOLD_DEGREES),
    )
    oracle = oracle_k_metrics(
        probabilities,
        offsets,
        directions,
        target_offsets,
        target_directions,
        target_mask,
        window_size=float(cfg.TRAIN.WINDOW_SIZE),
        endpoint_threshold_pixels=float(
            evaluation.ENDPOINT_MATCH_THRESHOLD_PIXELS),
        direction_threshold_degrees=float(
            evaluation.DIRECTION_MATCH_THRESHOLD_DEGREES),
        duplicate_endpoint_threshold_pixels=float(
            evaluation.DUPLICATE_ENDPOINT_THRESHOLD_PIXELS),
        duplicate_direction_threshold_degrees=float(
            evaluation.DUPLICATE_DIRECTION_THRESHOLD_DEGREES),
    )
    oracle_selected = oracle.pop("selected_query_indices")
    threshold_selected = [
        np.flatnonzero(probabilities[index] >= threshold)
        for index in range(probabilities.shape[0])
    ]
    duplicate_arguments = {
        "window_size": float(cfg.TRAIN.WINDOW_SIZE),
        "endpoint_threshold_pixels": float(
            evaluation.DUPLICATE_ENDPOINT_THRESHOLD_PIXELS),
        "direction_threshold_degrees": float(
            evaluation.DUPLICATE_DIRECTION_THRESHOLD_DEGREES),
    }
    duplicates = {
        "thresholded": duplicate_statistics(
            offsets, directions, threshold_selected,
            **duplicate_arguments),
        "oracle_k": duplicate_statistics(
            offsets, directions, oracle_selected,
            **duplicate_arguments),
        "actual_matched": duplicate_statistics(
            offsets, directions, actual_selected,
            **duplicate_arguments),
    }
    threshold_evaluator = _metric_accumulator(
        cfg, existence_threshold=threshold)
    threshold_evaluator.update(
        {
            "branch_exist_logits": torch.from_numpy(logits),
            "branch_offsets_norm": torch.from_numpy(offsets),
            "branch_directions": torch.from_numpy(directions),
        },
        {
            "branch_offsets_norm": torch.from_numpy(target_offsets),
            "branch_directions": torch.from_numpy(target_directions),
            "branch_mask": torch.from_numpy(target_mask),
        },
    )
    threshold_metrics = threshold_evaluator.compute()
    groups = {
        "count_0": np.flatnonzero(gt_counts == 0),
        "count_1": np.flatnonzero(gt_counts == 1),
        "count_2": np.flatnonzero(gt_counts == 2),
        "count_ge_3": np.flatnonzero(gt_counts >= 3),
    }
    metrics_by_count = {}
    for name, indices in groups.items():
        group = _subset_metrics(
            indices,
            probabilities=probabilities,
            offsets=offsets,
            directions=directions,
            target_offsets=target_offsets,
            target_directions=target_directions,
            target_mask=target_mask,
            cfg=cfg,
            threshold=threshold,
            actual_labels=actual_labels,
        )
        if indices.size:
            group["duplicates"] = {
                "thresholded": duplicate_statistics(
                    offsets[indices],
                    directions[indices],
                    [threshold_selected[int(index)]
                     for index in indices],
                    **duplicate_arguments),
                "oracle_k": duplicate_statistics(
                    offsets[indices],
                    directions[indices],
                    [oracle_selected[int(index)]
                     for index in indices],
                    **duplicate_arguments),
                "actual_matched": duplicate_statistics(
                    offsets[indices],
                    directions[indices],
                    [actual_selected[int(index)]
                     for index in indices],
                    **duplicate_arguments),
            }
        metrics_by_count[name] = group
    matched = probabilities[actual_labels]
    unmatched = probabilities[~actual_labels]
    return {
        "branch_ap": float(branch_pr["average_precision"]),
        "slot_ap": binary_average_precision(
            probabilities, actual_labels),
        "matched_probability": _stats(matched),
        "unmatched_probability": _stats(unmatched),
        "probability_separation_mean": (
            float(np.mean(matched) - np.mean(unmatched))
            if matched.size and unmatched.size
            else None
        ),
        "exact_count_accuracy": threshold_metrics[
            "exact_branch_count_accuracy"],
        "missed_branch_rate": threshold_metrics[
            "missed_branch_rate"],
        "extra_branch_rate": threshold_metrics[
            "extra_branch_rate"],
        "predicted_branch_count": threshold_metrics[
            "predicted_branch_count"],
        "gt_branch_count": threshold_metrics["gt_branch_count"],
        "oracle_k": oracle,
        "oracle_k_duplicate_ratio": duplicates[
            "oracle_k"]["duplicate_pair_ratio"],
        "actual_matched_duplicate_ratio": duplicates[
            "actual_matched"]["duplicate_pair_ratio"],
        "duplicates": duplicates,
        "metrics_by_gt_count": metrics_by_count,
    }


def run_diagnostics(
    *,
    cfg,
    checkpoint: Path,
    image_checkpoint: Path,
    dataset_dir: Path,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    max_samples: int = None,
    split: str = "val",
    dataset_indices: Sequence[int] = None,
) -> Dict[str, object]:
    _set_seed(int(cfg.STAGE3C.SEED))
    base_dataset = Stage3CBranchDataset(
        dataset_dir, split, preload=True)
    if dataset_indices is not None and max_samples is not None:
        raise ValueError(
            "dataset_indices and max_samples are mutually exclusive")
    if dataset_indices is not None:
        normalized_indices = [int(index) for index in dataset_indices]
        if len(set(normalized_indices)) != len(normalized_indices):
            raise ValueError("dataset_indices must be unique")
        if any(
                index < 0 or index >= len(base_dataset)
                for index in normalized_indices):
            raise IndexError("dataset index is outside the split")
        dataset = torch.utils.data.Subset(
            base_dataset, normalized_indices)
    elif max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        dataset = torch.utils.data.Subset(
            base_dataset,
            list(range(min(max_samples, len(base_dataset)))),
        )
    else:
        dataset = base_dataset
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg.STAGE3C.TRAINING.NUM_WORKERS),
        pin_memory=device.type == "cuda",
    )
    rpnet, image_payload = _load_frozen_rpnet(
        cfg, image_checkpoint, device)
    modules = _build_auxiliary_modules(cfg, device)
    optimizer = _build_optimizer(cfg, modules)
    checkpoint_payload = load_stage3c_checkpoint(
        checkpoint,
        trajectory_encoder=modules[0],
        graph_state_encoder=modules[1],
        branch_decoder=modules[2],
        optimizer=optimizer,
        map_location=device,
    )
    _set_module_mode(modules, False)
    criterion = _build_branch_criterion(cfg)
    diagnostics_cfg = cfg.STAGE3C.get("DIAGNOSTICS", {})
    threshold = float(diagnostics_cfg.get(
        "EXISTENCE_THRESHOLD",
        cfg.STAGE3C.EVALUATION.EXISTENCE_THRESHOLD,
    ))
    calibration_bins = int(diagnostics_cfg.get(
        "CALIBRATION_BINS", 15))

    modality_chunks = {
        modality: defaultdict(list) for modality in MODALITIES
    }
    target_chunks = defaultdict(list)
    actual_label_chunks_by_modality = {
        modality: [] for modality in MODALITIES
    }
    reference_label_chunks = []
    sample_id_chunks = []
    actual_selected_by_modality = {
        modality: [] for modality in MODALITIES
    }
    reference_selected: List[np.ndarray] = []
    match_frequencies = np.zeros(
        int(cfg.STAGE3C.MODEL.NUM_QUERIES), dtype=np.int64)
    match_frequency_by_count = defaultdict(
        lambda: np.zeros(
            int(cfg.STAGE3C.MODEL.NUM_QUERIES), dtype=np.int64))
    cost_chunks = defaultdict(list)
    matched_cost_chunks = defaultdict(list)
    assignment_margins = []
    matched_probability_ranks = []
    actual_match_is_highest = []
    debug_chunks = {
        modality: defaultdict(list) for modality in MODALITIES
    }
    started_at = time.perf_counter()
    sample_offset = 0
    with torch.no_grad():
        for cpu_batch in loader:
            batch = _move_nested(cpu_batch, device)
            stage_fuse = _stage_fuse_for_batch(
                rpnet=rpnet,
                batch=batch,
                cache=None,
                device=device,
            )
            targets = batch["branch_targets"]
            batch_count = int(stage_fuse.shape[0])
            target_chunks["offsets"].append(_numpy(
                targets["branch_offsets_norm"]))
            target_chunks["directions"].append(_numpy(
                targets["branch_directions"]))
            target_chunks["mask"].append(_numpy(
                targets["branch_mask"]).astype(bool))
            target_chunks["count"].append(_numpy(
                targets["branch_count"]).astype(np.int64))
            if "dataset_index" in batch["metadata"]:
                sample_ids = _numpy(
                    batch["metadata"]["dataset_index"]).astype(np.int64)
            else:
                sample_ids = np.arange(
                    sample_offset, sample_offset + batch_count)
            sample_id_chunks.append(sample_ids)
            sample_offset += batch_count

            predictions_by_modality = {}
            for modality in MODALITIES:
                predictions = _forward_auxiliary(
                    modules=modules,
                    batch=batch,
                    stage_fuse=stage_fuse,
                    modality=modality,
                    return_debug_states=True,
                )
                predictions_by_modality[modality] = predictions
                modality_chunks[modality]["probability"].append(
                    _numpy(torch.sigmoid(
                        predictions["branch_exist_logits"])))
                modality_chunks[modality]["logit"].append(
                    _numpy(predictions["branch_exist_logits"]))
                modality_chunks[modality]["offsets"].append(
                    _numpy(predictions["branch_offsets_norm"]))
                modality_chunks[modality]["directions"].append(
                    _numpy(predictions["branch_directions"]))
                for key in DEBUG_STATE_KEYS:
                    debug_chunks[modality][key].append(
                        _numpy(predictions[key]))
                modality_matches = criterion(
                    predictions, targets)["matches"]
                actual_label_chunks_by_modality[modality].append(
                    _matches_to_mask(
                        modality_matches,
                        batch_count,
                        int(predictions[
                            "branch_exist_logits"].shape[1]),
                    )
                )
                actual_selected_by_modality[modality].extend(
                    _selected_from_matches(modality_matches))

            predictions = predictions_by_modality[MODALITY_FULL]
            losses = criterion(predictions, targets)
            actual_matches = losses["matches"]
            reference_matches = hungarian_match_branches(
                predictions,
                targets,
                endpoint_cost_weight=criterion.endpoint_cost_weight,
                direction_cost_weight=criterion.direction_cost_weight,
                existence_cost_weight=0.0,
            )
            reference_label_chunks.append(_matches_to_mask(
                reference_matches, batch_count,
                int(predictions["branch_exist_logits"].shape[1])))
            reference_selected.extend(
                _selected_from_matches(reference_matches))

            probabilities = _numpy(torch.sigmoid(
                predictions["branch_exist_logits"]))
            counts = _numpy(targets["branch_count"]).astype(np.int64)
            components = branch_matching_cost_components(
                predictions, targets)
            target_mask = targets["branch_mask"].to(dtype=torch.bool)
            valid_matrix = target_mask.unsqueeze(1).expand_as(
                components["endpoint"])
            total_cost = (
                criterion.endpoint_cost_weight * components["endpoint"]
                + criterion.direction_cost_weight
                * components["direction"]
                + criterion.match_cost_exist_weight
                * components["existence"]
            )
            for name, component in components.items():
                cost_chunks[name].append(
                    _numpy(component[valid_matrix]).reshape(-1))
            for batch_index, (
                    query_indices, target_indices) in enumerate(
                        actual_matches):
                query_np = _numpy(query_indices).astype(np.int64)
                target_np = _numpy(target_indices).astype(np.int64)
                actual_selected_count = int(counts[batch_index])
                group = _group_name(actual_selected_count)
                for query_index in query_np:
                    match_frequencies[query_index] += 1
                    match_frequency_by_count[group][query_index] += 1
                    probability_order = np.lexsort((
                        np.arange(probabilities.shape[1]),
                        -probabilities[batch_index],
                    ))
                    rank = int(np.flatnonzero(
                        probability_order == query_index)[0]) + 1
                    matched_probability_ranks.append(rank)
                    actual_match_is_highest.append(rank == 1)
                for name, component in components.items():
                    if query_np.size:
                        matched_cost_chunks[name].append(_numpy(
                            component[
                                batch_index,
                                query_indices,
                                target_indices,
                            ]
                        ).reshape(-1))
                valid_targets = np.flatnonzero(_numpy(
                    target_mask[batch_index]))
                for target_index in valid_targets:
                    column = _numpy(total_cost[
                        batch_index, :, target_index])
                    ordered = np.sort(column)
                    if ordered.size >= 2:
                        assignment_margins.append(
                            float(ordered[1] - ordered[0]))

    target_offsets = _concatenate(target_chunks["offsets"])
    target_directions = _concatenate(target_chunks["directions"])
    target_mask = _concatenate(target_chunks["mask"]).astype(bool)
    gt_counts = _concatenate(target_chunks["count"]).astype(np.int64)
    sample_ids = _concatenate(sample_id_chunks).astype(np.int64)
    actual_labels_by_modality = {
        modality: _concatenate(chunks).astype(bool)
        for modality, chunks
        in actual_label_chunks_by_modality.items()
    }
    actual_labels = actual_labels_by_modality[MODALITY_FULL]
    actual_selected = actual_selected_by_modality[MODALITY_FULL]
    reference_labels = _concatenate(
        reference_label_chunks).astype(bool)
    full = {
        key: _concatenate(values)
        for key, values in modality_chunks[MODALITY_FULL].items()
    }
    probabilities = full["probability"]
    logits = full["logit"]
    offsets = full["offsets"]
    directions = full["directions"]
    evaluation = cfg.STAGE3C.EVALUATION

    calibration_actual = calibration_curve(
        probabilities, actual_labels, bin_count=calibration_bins)
    calibration_reference = calibration_curve(
        probabilities, reference_labels, bin_count=calibration_bins)
    branch_pr_by_modality = {}
    modality_values = {}
    for modality in MODALITIES:
        values = {
            key: _concatenate(chunks)
            for key, chunks in modality_chunks[modality].items()
        }
        modality_values[modality] = values
        branch_pr_by_modality[modality] = (
            branch_precision_recall_curve(
                values["probability"],
                values["offsets"],
                values["directions"],
                target_offsets,
                target_directions,
                target_mask,
                window_size=float(cfg.TRAIN.WINDOW_SIZE),
                endpoint_threshold_pixels=float(
                    evaluation.ENDPOINT_MATCH_THRESHOLD_PIXELS),
                direction_threshold_degrees=float(
                    evaluation.DIRECTION_MATCH_THRESHOLD_DEGREES),
            )
        )

    oracle = oracle_k_metrics(
        probabilities,
        offsets,
        directions,
        target_offsets,
        target_directions,
        target_mask,
        window_size=float(cfg.TRAIN.WINDOW_SIZE),
        endpoint_threshold_pixels=float(
            evaluation.ENDPOINT_MATCH_THRESHOLD_PIXELS),
        direction_threshold_degrees=float(
            evaluation.DIRECTION_MATCH_THRESHOLD_DEGREES),
        duplicate_endpoint_threshold_pixels=float(
            evaluation.DUPLICATE_ENDPOINT_THRESHOLD_PIXELS),
        duplicate_direction_threshold_degrees=float(
            evaluation.DUPLICATE_DIRECTION_THRESHOLD_DEGREES),
    )
    oracle_selected = oracle.pop("selected_query_indices")
    threshold_selected = [
        np.flatnonzero(probabilities[index] >= threshold)
        for index in range(probabilities.shape[0])
    ]
    duplicate_arguments = {
        "window_size": float(cfg.TRAIN.WINDOW_SIZE),
        "endpoint_threshold_pixels": float(
            evaluation.DUPLICATE_ENDPOINT_THRESHOLD_PIXELS),
        "direction_threshold_degrees": float(
            evaluation.DUPLICATE_DIRECTION_THRESHOLD_DEGREES),
    }
    duplicates = {
        "thresholded": duplicate_statistics(
            offsets, directions, threshold_selected,
            **duplicate_arguments),
        "oracle_k": duplicate_statistics(
            offsets, directions, oracle_selected,
            **duplicate_arguments),
        "actual_matched": duplicate_statistics(
            offsets, directions, actual_selected,
            **duplicate_arguments),
        "geometry_reference_matched": duplicate_statistics(
            offsets, directions, reference_selected,
            **duplicate_arguments),
    }

    # Reuse the existing thresholded evaluator, including its exact TP
    # endpoint/direction criteria and missed/extra definitions.
    threshold_evaluator = _metric_accumulator(
        cfg, existence_threshold=threshold)
    threshold_evaluator.update(
        {
            "branch_exist_logits": torch.from_numpy(logits),
            "branch_offsets_norm": torch.from_numpy(offsets),
            "branch_directions": torch.from_numpy(directions),
        },
        {
            "branch_offsets_norm": torch.from_numpy(target_offsets),
            "branch_directions": torch.from_numpy(target_directions),
            "branch_mask": torch.from_numpy(target_mask),
        },
    )
    threshold_metrics = threshold_evaluator.compute()

    matched_probabilities = probabilities[actual_labels]
    unmatched_probabilities = probabilities[~actual_labels]
    reference_matched_probabilities = probabilities[reference_labels]
    reference_unmatched_probabilities = probabilities[~reference_labels]
    query_similarity = _query_similarity_summary(debug_chunks)
    modality_summaries = {
        modality: _modality_diagnostic_summary(
            values=modality_values[modality],
            actual_labels=actual_labels_by_modality[modality],
            actual_selected=actual_selected_by_modality[modality],
            target_offsets=target_offsets,
            target_directions=target_directions,
            target_mask=target_mask,
            gt_counts=gt_counts,
            cfg=cfg,
            threshold=threshold,
        )
        for modality in MODALITIES
    }
    for modality in MODALITIES:
        modality_summaries[modality]["query_similarity"] = (
            query_similarity[modality])
    offset_norm = np.linalg.norm(offsets, axis=-1)
    direction_finite = np.isfinite(directions).all(axis=-1)
    offset_stats = {
        "all_offset_norm": _stats(offset_norm.reshape(-1)),
        "matched_offset_norm": _stats(offset_norm[actual_labels]),
        "unmatched_offset_norm": _stats(offset_norm[~actual_labels]),
        "near_zero_offset_ratio": float(
            np.mean(offset_norm < 1e-6)),
        "absolute_tanh_output_gt_0_95_ratio": float(
            np.mean(np.abs(offsets) > 0.95)),
        "matched_absolute_tanh_output_gt_0_95_ratio": float(
            np.mean(np.abs(offsets[actual_labels]) > 0.95)
            if np.any(actual_labels) else 0.0),
        "unmatched_absolute_tanh_output_gt_0_95_ratio": float(
            np.mean(np.abs(offsets[~actual_labels]) > 0.95)
            if np.any(~actual_labels) else 0.0),
        "direction_nan_count": int(np.isnan(directions).sum()),
        "direction_inf_count": int(np.isinf(directions).sum()),
        "finite_direction_slot_ratio": float(
            np.mean(direction_finite)),
    }
    group_indices = {
        "count_0": np.flatnonzero(gt_counts == 0),
        "count_1": np.flatnonzero(gt_counts == 1),
        "count_ge_2": np.flatnonzero(gt_counts >= 2),
        "count_2": np.flatnonzero(gt_counts == 2),
        "count_ge_3": np.flatnonzero(gt_counts >= 3),
    }
    metrics_by_count = {
        name: _subset_metrics(
            indices,
            probabilities=probabilities,
            offsets=offsets,
            directions=directions,
            target_offsets=target_offsets,
            target_directions=target_directions,
            target_mask=target_mask,
            cfg=cfg,
        )
        for name, indices in group_indices.items()
    }
    for name, indices in group_indices.items():
        if indices.size == 0:
            continue
        metrics_by_count[name]["duplicates"] = {}
        for selection_name, selection in (
                ("thresholded", threshold_selected),
                ("oracle_k", oracle_selected),
                ("actual_matched", actual_selected),
                ("geometry_reference_matched", reference_selected)):
            metrics_by_count[name]["duplicates"][selection_name] = (
                duplicate_statistics(
                    offsets[indices],
                    directions[indices],
                    [selection[int(index)] for index in indices],
                    **duplicate_arguments,
                )
            )

    probability_rank_stats = _stats(matched_probability_ranks)
    per_query_probability = {
        str(query_index): _stats(probabilities[:, query_index])
        for query_index in range(probabilities.shape[1])
    }
    histogram_edges = np.linspace(0.0, 1.0, 21)
    matched_histogram, _ = np.histogram(
        matched_probabilities, bins=histogram_edges)
    unmatched_histogram, _ = np.histogram(
        unmatched_probabilities, bins=histogram_edges)
    empty_probabilities = probabilities[gt_counts == 0]
    assignment_stats = {
        "per_query_match_frequency": {
            str(index): int(value)
            for index, value in enumerate(match_frequencies)
        },
        "per_query_match_frequency_by_gt_count": {
            group: {
                str(index): int(value)
                for index, value in enumerate(frequencies)
            }
            for group, frequencies in match_frequency_by_count.items()
        },
        "best_second_best_cost_margin": _stats(
            assignment_margins),
        "cost_component_statistics": {
            "all_candidate_to_gt": _cost_statistics(cost_chunks),
            "actual_matches": _cost_statistics(matched_cost_chunks),
        },
        "actual_matched_query_is_highest_probability_ratio": (
            float(np.mean(actual_match_is_highest))
            if actual_match_is_highest else None
        ),
        "empty_gt_query_probability": _stats(
            empty_probabilities.reshape(-1)),
    }

    summary = {
        "schema_version": "stage3c-diagnostics-v1",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_epoch": int(
            checkpoint_payload.get("epoch", -1)),
        "checkpoint_state_dict_key_counts": {
            "trajectory_encoder": len(
                checkpoint_payload["trajectory_encoder"]),
            "graph_state_encoder": len(
                checkpoint_payload["graph_state_encoder"]),
            "branch_decoder": len(
                checkpoint_payload["branch_decoder"]),
        },
        "image_checkpoint": str(image_checkpoint.resolve()),
        "rpnet_checkpoint_metadata": {
            key: image_payload.get(key)
            for key in (
                "outer_it", "path_it", "trajectory_mode",
                "model_name", "num_targets", "step_length",
                "window_size",
            )
        },
        "sample_count": int(probabilities.shape[0]),
        "dataset_split": split,
        "dataset_indices": (
            [int(index) for index in dataset_indices]
            if dataset_indices is not None
            else None
        ),
        "query_count": int(probabilities.shape[1]),
        "diagnostic_threshold": threshold,
        "branch_ap": float(
            branch_pr_by_modality[
                MODALITY_FULL]["average_precision"]),
        "branch_ap_by_modality": {
            modality: float(values["average_precision"])
            for modality, values in branch_pr_by_modality.items()
        },
        "metrics_by_modality": modality_summaries,
        "slot_ap": binary_average_precision(
            probabilities, actual_labels),
        "slot_ap_geometry_reference": binary_average_precision(
            probabilities, reference_labels),
        "slot_auroc": binary_auroc(
            probabilities, actual_labels),
        "slot_auroc_geometry_reference": binary_auroc(
            probabilities, reference_labels),
        "ece": float(calibration_actual["ece"]),
        "ece_geometry_reference": float(
            calibration_reference["ece"]),
        "matched_logit": _stats(logits[actual_labels]),
        "unmatched_logit": _stats(logits[~actual_labels]),
        "matched_probability": _stats(matched_probabilities),
        "unmatched_probability": _stats(
            unmatched_probabilities),
        "geometry_reference_matched_probability": _stats(
            reference_matched_probabilities),
        "geometry_reference_unmatched_probability": _stats(
            reference_unmatched_probabilities),
        "matched_prob_mean": _stats(
            matched_probabilities)["mean"],
        "matched_prob_median": _stats(
            matched_probabilities)["median"],
        "unmatched_prob_mean": _stats(
            unmatched_probabilities)["mean"],
        "unmatched_prob_median": _stats(
            unmatched_probabilities)["median"],
        "probability_rank_of_matched": probability_rank_stats,
        "probability_gt_0_1_given_matched": float(
            np.mean(matched_probabilities > 0.1)),
        "probability_gt_0_1_given_unmatched": float(
            np.mean(unmatched_probabilities > 0.1)),
        "per_query_average_probability": per_query_probability,
        "exact_count_accuracy": threshold_metrics[
            "exact_branch_count_accuracy"],
        "predicted_branch_count": threshold_metrics[
            "predicted_branch_count"],
        "GT_branch_count": threshold_metrics["gt_branch_count"],
        "missed_branch_rate": threshold_metrics[
            "missed_branch_rate"],
        "extra_branch_rate": threshold_metrics[
            "extra_branch_rate"],
        "thresholded_duplicate_ratio": duplicates[
            "thresholded"]["duplicate_pair_ratio"],
        "oracle_k_duplicate_ratio": duplicates[
            "oracle_k"]["duplicate_pair_ratio"],
        "matched_duplicate_ratio": duplicates[
            "actual_matched"]["duplicate_pair_ratio"],
        "geometry_reference_matched_duplicate_ratio": duplicates[
            "geometry_reference_matched"]["duplicate_pair_ratio"],
        "oracle_k": oracle,
        "duplicates": duplicates,
        "metrics_by_GT_count": metrics_by_count,
        "per_query_match_frequency": assignment_stats[
            "per_query_match_frequency"],
        "cost_component_statistics": assignment_stats[
            "cost_component_statistics"],
        "offset_saturation_statistics": offset_stats,
        "elapsed_seconds": float(time.perf_counter() - started_at),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "summary.json", summary)
    _write_json(
        output_dir / "assignment_stats.json", assignment_stats)
    _write_json(
        output_dir / "query_similarity.json", query_similarity)
    _write_json(output_dir / "offset_stats.json", offset_stats)
    np.savez(
        str(output_dir / "calibration.npz"),
        bin_edges=calibration_actual["bin_edges"],
        actual_count=calibration_actual["count"],
        actual_mean_probability=calibration_actual[
            "mean_probability"],
        actual_positive_fraction=calibration_actual[
            "positive_fraction"],
        geometry_reference_count=calibration_reference["count"],
        geometry_reference_mean_probability=calibration_reference[
            "mean_probability"],
        geometry_reference_positive_fraction=calibration_reference[
            "positive_fraction"],
    )
    pr_payload = {}
    for modality, values in branch_pr_by_modality.items():
        for key in (
                "precision", "recall", "scores", "true_positive",
                "sample_indices", "query_indices"):
            pr_payload["{}_{}".format(modality, key)] = values[key]
        pr_payload["{}_average_precision".format(modality)] = (
            np.asarray(values["average_precision"]))
    np.savez(str(output_dir / "branch_pr_curve.npz"), **pr_payload)
    np.savez(
        str(output_dir / "existence_histograms.npz"),
        bin_edges=histogram_edges,
        matched_count=matched_histogram,
        unmatched_count=unmatched_histogram,
    )
    with (output_dir / "per_query.csv").open(
            "w", encoding="utf-8", newline="") as output:
        field_names = (
            "sample_id", "query_id", "gt_branch_count",
            "exist_logit", "exist_probability",
            "actual_matched", "geometry_reference_matched",
            "endpoint_x_norm", "endpoint_y_norm",
            "direction_x", "direction_y",
        )
        writer = csv.DictWriter(output, fieldnames=field_names)
        writer.writeheader()
        for sample_index in range(probabilities.shape[0]):
            for query_index in range(probabilities.shape[1]):
                writer.writerow({
                    "sample_id": int(sample_ids[sample_index]),
                    "query_id": query_index,
                    "gt_branch_count": int(gt_counts[sample_index]),
                    "exist_logit": float(
                        logits[sample_index, query_index]),
                    "exist_probability": float(
                        probabilities[sample_index, query_index]),
                    "actual_matched": int(
                        actual_labels[sample_index, query_index]),
                    "geometry_reference_matched": int(
                        reference_labels[sample_index, query_index]),
                    "endpoint_x_norm": float(
                        offsets[sample_index, query_index, 0]),
                    "endpoint_y_norm": float(
                        offsets[sample_index, query_index, 1]),
                    "direction_x": float(
                        directions[sample_index, query_index, 0]),
                    "direction_y": float(
                        directions[sample_index, query_index, 1]),
                })
    return summary


def main() -> None:
    args = _parse_args()
    cfg = _load_config(args.config)
    diagnostics_cfg = cfg.STAGE3C.get("DIAGNOSTICS", {})
    checkpoint = args.checkpoint or Path(
        diagnostics_cfg.get("CHECKPOINT", ""))
    if not str(checkpoint):
        raise ValueError(
            "checkpoint is required in config or --checkpoint")
    dataset_dir = args.dataset_dir or Path(cfg.STAGE3C.DATASET_DIR)
    image_checkpoint = (
        args.image_checkpoint
        or Path(cfg.STAGE3C.IMAGE_CHECKPOINT)
    )
    output_dir = args.output_dir or Path(cfg.STAGE3C.OUTPUT_DIR)
    device = _resolve_device(
        args.device or str(cfg.STAGE3C.DEVICE))
    batch_size = args.batch_size or int(
        cfg.STAGE3C.TRAINING.VAL_BATCH_SIZE)
    summary = run_diagnostics(
        cfg=cfg,
        checkpoint=checkpoint,
        image_checkpoint=image_checkpoint,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        device=device,
        batch_size=batch_size,
        max_samples=args.max_samples,
        split=args.split,
    )
    print(json.dumps(_json_ready({
        "summary": str((output_dir / "summary.json").resolve()),
        "branch_ap": summary["branch_ap"],
        "slot_ap": summary["slot_ap"],
        "slot_auroc": summary["slot_auroc"],
        "oracle_k_duplicate_ratio": summary[
            "oracle_k_duplicate_ratio"],
        "matched_duplicate_ratio": summary[
            "matched_duplicate_ratio"],
    }), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
