"""Offline Stage 3D-C0 support-guided trajectory aggregation evaluation.

This script is inference-only.  It strictly loads and freezes RPNet, E4, and
the Stage 3D-A post-fusion support head.  Support-derived contexts are passed
through E4's existing frozen fusion/output layers but never into Path.push.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from easydict import EasyDict
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.trajectory_support_head import TrajectorySupportHead  # noqa: E402
from train_branch_aux import (  # noqa: E402
    _build_auxiliary_modules,
    _build_branch_criterion,
    _load_config,
    _load_frozen_rpnet,
    _move_nested,
    _resolve_device,
    _set_seed,
    _stage_fuse_for_batch,
)
from utils.branch_diagnostics import (  # noqa: E402
    binary_average_precision,
    distribution_statistics,
    oracle_k_metrics,
    query_pairwise_statistics,
)
from utils.branch_metrics import (  # noqa: E402
    BranchAveragePrecisionAccumulator,
    BranchMetricAccumulator,
)
from utils.stage3c_branch_dataset import (  # noqa: E402
    Stage3CBranchDataset,
)
from utils.stage3c_checkpoint import load_stage3c_checkpoint  # noqa: E402
from utils.stage3d_checkpoint import (  # noqa: E402
    load_stage3d_support_checkpoint,
)
from utils.trajectory_support_aggregation import (  # noqa: E402
    decoder_fragment_values,
    freeze_modules,
    permute_valid_fragment_values,
    recompute_branch_predictions,
    support_weighted_trajectory_context,
)
from utils.trajectory_support_ranking import (  # noqa: E402
    TrajectorySupportRankingAccumulator,
)
from utils.trajectory_support_targets import (  # noqa: E402
    build_trajectory_support_targets,
)


VARIANT_ORIGINAL = "original_attention"
VARIANT_SUPPORT = "support_aggregation"
VARIANT_NO_TRAJECTORY = "no_trajectory"
VARIANT_RANDOM = "random_aggregation"


def _topk_variant_name(top_k: int) -> str:
    return "support_topk_{}".format(int(top_k))


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
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


def _metric_accumulator(cfg: EasyDict) -> BranchMetricAccumulator:
    evaluation = cfg.STAGE3C.EVALUATION
    return BranchMetricAccumulator(
        window_size=float(cfg.TRAIN.WINDOW_SIZE),
        existence_threshold=float(evaluation.EXISTENCE_THRESHOLD),
        endpoint_match_threshold_pixels=float(
            evaluation.ENDPOINT_MATCH_THRESHOLD_PIXELS),
        direction_match_threshold_degrees=float(
            evaluation.DIRECTION_MATCH_THRESHOLD_DEGREES),
        duplicate_endpoint_threshold_pixels=float(
            evaluation.DUPLICATE_ENDPOINT_THRESHOLD_PIXELS),
        duplicate_direction_threshold_degrees=float(
            evaluation.DUPLICATE_DIRECTION_THRESHOLD_DEGREES),
    )


def _ap_accumulator(cfg: EasyDict) -> BranchAveragePrecisionAccumulator:
    evaluation = cfg.STAGE3C.EVALUATION
    return BranchAveragePrecisionAccumulator(
        window_size=float(cfg.TRAIN.WINDOW_SIZE),
        endpoint_match_threshold_pixels=float(
            evaluation.ENDPOINT_MATCH_THRESHOLD_PIXELS),
        direction_match_threshold_degrees=float(
            evaluation.DIRECTION_MATCH_THRESHOLD_DEGREES),
    )


class BranchVariantAccumulator:
    """Collect thresholded, AP, and oracle-K metrics for one variant."""

    def __init__(self, cfg: EasyDict) -> None:
        self.cfg = cfg
        self.thresholded = _metric_accumulator(cfg)
        self.ap = _ap_accumulator(cfg)
        self.prediction_chunks: Dict[str, List[np.ndarray]] = {
            "logits": [],
            "scores": [],
            "offsets": [],
            "directions": [],
            "slot_labels": [],
        }
        self.target_chunks: Dict[str, List[np.ndarray]] = {
            "offsets": [],
            "directions": [],
            "mask": [],
            "counts": [],
        }

    def update(
        self,
        predictions: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
        matches: Sequence,
    ) -> None:
        prediction_dict = dict(predictions)
        target_dict = dict(targets)
        self.thresholded.update(prediction_dict, target_dict)
        self.ap.update(prediction_dict, target_dict)
        slot_labels = torch.zeros_like(
            predictions["branch_exist_logits"],
            dtype=torch.bool,
        )
        if len(matches) != slot_labels.shape[0]:
            raise ValueError(
                "matches must contain one assignment per sample")
        for batch_index, (
                prediction_indices, _) in enumerate(matches):
            slot_labels[batch_index, prediction_indices] = True
        self.prediction_chunks["logits"].append(
            predictions["branch_exist_logits"].detach().cpu().numpy())
        self.prediction_chunks["scores"].append(torch.sigmoid(
            predictions["branch_exist_logits"]
        ).detach().cpu().numpy())
        self.prediction_chunks["offsets"].append(
            predictions["branch_offsets_norm"].detach().cpu().numpy())
        self.prediction_chunks["directions"].append(
            predictions["branch_directions"].detach().cpu().numpy())
        self.prediction_chunks["slot_labels"].append(
            slot_labels.detach().cpu().numpy())
        self.target_chunks["offsets"].append(
            targets["branch_offsets_norm"].detach().cpu().numpy())
        self.target_chunks["directions"].append(
            targets["branch_directions"].detach().cpu().numpy())
        self.target_chunks["mask"].append(
            targets["branch_mask"].detach().cpu().numpy())
        self.target_chunks["counts"].append(
            targets["branch_mask"].sum(dim=1).detach().cpu().numpy())

    def _subset_metrics(
        self,
        arrays: Mapping[str, np.ndarray],
        target_arrays: Mapping[str, np.ndarray],
        indices: np.ndarray,
    ) -> Dict[str, Any]:
        if indices.size == 0:
            return {
                "sample_count": 0,
                "gt_branch_count": 0,
                "branch_ap": 0.0,
                "slot_ap": 0.0,
            }
        predictions = {
            "branch_exist_logits": torch.from_numpy(
                arrays["logits"][indices]),
            "branch_offsets_norm": torch.from_numpy(
                arrays["offsets"][indices]),
            "branch_directions": torch.from_numpy(
                arrays["directions"][indices]),
        }
        targets = {
            "branch_offsets_norm": torch.from_numpy(
                target_arrays["offsets"][indices]),
            "branch_directions": torch.from_numpy(
                target_arrays["directions"][indices]),
            "branch_mask": torch.from_numpy(
                target_arrays["mask"][indices]),
        }
        thresholded_accumulator = _metric_accumulator(self.cfg)
        ap_accumulator = _ap_accumulator(self.cfg)
        thresholded_accumulator.update(predictions, targets)
        ap_accumulator.update(predictions, targets)
        evaluation = self.cfg.STAGE3C.EVALUATION
        oracle = oracle_k_metrics(
            arrays["scores"][indices],
            arrays["offsets"][indices],
            arrays["directions"][indices],
            target_arrays["offsets"][indices],
            target_arrays["directions"][indices],
            target_arrays["mask"][indices],
            window_size=float(self.cfg.TRAIN.WINDOW_SIZE),
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
        return {
            "sample_count": int(indices.size),
            "gt_branch_count": int(
                target_arrays["mask"][indices].sum()),
            "branch_ap": float(
                ap_accumulator.compute()["average_precision"]),
            "slot_ap": float(binary_average_precision(
                arrays["scores"][indices],
                arrays["slot_labels"][indices],
            )),
            "thresholded": thresholded_accumulator.compute(),
            "oracle_k": oracle,
        }

    def compute(self) -> Dict[str, Any]:
        thresholded = self.thresholded.compute()
        ap = self.ap.compute()
        if not self.prediction_chunks["scores"]:
            raise RuntimeError("branch variant accumulator is empty")
        evaluation = self.cfg.STAGE3C.EVALUATION
        oracle = oracle_k_metrics(
            np.concatenate(self.prediction_chunks["scores"], axis=0),
            np.concatenate(self.prediction_chunks["offsets"], axis=0),
            np.concatenate(self.prediction_chunks["directions"], axis=0),
            np.concatenate(self.target_chunks["offsets"], axis=0),
            np.concatenate(self.target_chunks["directions"], axis=0),
            np.concatenate(self.target_chunks["mask"], axis=0),
            window_size=float(self.cfg.TRAIN.WINDOW_SIZE),
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
        arrays = {
            key: np.concatenate(chunks, axis=0)
            for key, chunks in self.prediction_chunks.items()
        }
        target_arrays = {
            key: np.concatenate(chunks, axis=0)
            for key, chunks in self.target_chunks.items()
        }
        categories = {
            "ordinary": np.flatnonzero(
                target_arrays["counts"] == 1),
            "t_junction": np.flatnonzero(
                target_arrays["counts"] == 2),
            "multi_branch": np.flatnonzero(
                target_arrays["counts"] >= 3),
        }
        return {
            "branch_ap": float(ap["average_precision"]),
            "slot_ap": float(binary_average_precision(
                arrays["scores"], arrays["slot_labels"])),
            "thresholded": thresholded,
            "oracle_k": oracle,
            "by_category": {
                name: self._subset_metrics(
                    arrays, target_arrays, indices)
                for name, indices in categories.items()
            },
        }


def _max_prediction_difference(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> float:
    return max(
        float(torch.max(torch.abs(left[key] - right[key])).detach().cpu())
        for key in (
            "branch_exist_logits",
            "branch_offsets_norm",
            "branch_directions",
        )
    )


def _mean_std(values: Sequence[float]) -> Dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()) if array.size else None,
        "std": float(array.std()) if array.size else None,
        "values": array.tolist(),
    }


def _random_summary(
    results: Mapping[int, Mapping[str, Any]],
) -> Dict[str, Any]:
    paths = {
        "branch_ap": lambda value: value["branch_ap"],
        "slot_ap": lambda value: value["slot_ap"],
        "exact_branch_count_accuracy": lambda value: value[
            "thresholded"]["exact_branch_count_accuracy"],
        "endpoint_error_mean_pixels": lambda value: value[
            "thresholded"]["endpoint_error_mean_pixels"],
        "direction_error_mean_degrees": lambda value: value[
            "thresholded"]["direction_error_mean_degrees"],
        "oracle_k_duplicate_pair_ratio": lambda value: value[
            "oracle_k"]["duplicates"]["duplicate_pair_ratio"],
        "oracle_k_distinct_gt_coverage": lambda value: value[
            "oracle_k"]["distinct_gt_coverage"],
    }
    return {
        key: _mean_std([
            float(getter(value))
            for value in results.values()
        ])
        for key, getter in paths.items()
    }


def _context_similarity_metrics(
    original_contexts: np.ndarray,
    support_contexts: np.ndarray,
) -> Dict[str, Any]:
    if original_contexts.shape != support_contexts.shape:
        raise ValueError(
            "original and support contexts must share one shape")
    original_norm = np.linalg.norm(original_contexts, axis=-1)
    support_norm = np.linalg.norm(support_contexts, axis=-1)
    valid = (original_norm > 1e-8) & (support_norm > 1e-8)
    cosine = np.zeros_like(original_norm, dtype=np.float64)
    cosine[valid] = (
        np.sum(
            original_contexts * support_contexts,
            axis=-1,
        )[valid]
        / (original_norm[valid] * support_norm[valid])
    )
    return {
        "original_inter_query": query_pairwise_statistics(
            original_contexts),
        "support_inter_query": query_pairwise_statistics(
            support_contexts),
        "aligned_original_support": distribution_statistics(
            cosine[valid].tolist()),
    }


def assess_stage3d_c0(
    *,
    original_branch_ap: float,
    support_branch_ap: float,
    no_trajectory_branch_ap: float,
    support_selection_ap: float,
    cfg: EasyDict,
    random_branch_ap_mean: Optional[float] = None,
    random_branch_ap_std: Optional[float] = None,
) -> Dict[str, Any]:
    decision = cfg.STAGE3D_C0.DECISION
    support_delta = support_branch_ap - original_branch_ap
    full_delta = original_branch_ap - no_trajectory_branch_ap
    support_improved = (
        support_delta
        > float(decision.MIN_SUPPORT_OVER_ORIGINAL_BRANCH_AP)
    )
    full_gain_meaningful = (
        full_delta
        >= float(decision.MIN_FULL_OVER_NO_TRAJECTORY_BRANCH_AP)
    )
    support_selection_high = (
        support_selection_ap
        >= float(decision.MIN_HIGH_SUPPORT_SELECTION_AP)
    )
    can_enter_next_stage = bool(
        support_improved and full_gain_meaningful)
    support_random_delta = (
        None
        if random_branch_ap_mean is None
        else support_branch_ap - random_branch_ap_mean
    )
    support_random_z = (
        None
        if (
            support_random_delta is None
            or random_branch_ap_std is None
            or random_branch_ap_std <= 0.0
        )
        else support_random_delta / random_branch_ap_std
    )
    if can_enter_next_stage:
        diagnosis = (
            "offline support aggregation improves frozen E4 branch AP; "
            "the aggregation mechanism merits a non-circular training-stage "
            "experiment, but the Stage 3D-A post-fusion head cannot be used "
            "as an online selector")
    elif support_selection_high and not support_improved:
        diagnosis = (
            "support ranking is strong but frozen aggregation does not "
            "materially improve branch AP; support labels are not changed, "
            "and the "
            "branch-token/decoder fusion plus post-fusion circular dependency "
            "must be analyzed before any integration")
    elif support_delta < 0.0:
        diagnosis = (
            "support aggregation lowers frozen E4 branch AP; keep support "
            "labels unchanged and analyze post-fusion circularity and fusion "
            "distribution mismatch")
    else:
        diagnosis = (
            "offline aggregation evidence is inconclusive and does not meet "
            "the configured full-versus-no-trajectory gain")
    return {
        "support_minus_original_branch_ap": float(support_delta),
        "full_minus_no_trajectory_branch_ap": float(full_delta),
        "support_selection_ap": float(support_selection_ap),
        "support_minus_random_mean_branch_ap": (
            None if support_random_delta is None
            else float(support_random_delta)),
        "support_vs_random_standard_deviations": (
            None if support_random_z is None
            else float(support_random_z)),
        "support_aggregation_improves_original": support_improved,
        "full_gain_over_no_trajectory_is_meaningful":
            full_gain_meaningful,
        "support_selection_is_high": support_selection_high,
        "can_enter_next_stage_training": can_enter_next_stage,
        "post_fusion_support_is_circular_for_online_use": True,
        "diagnosis": diagnosis,
        "thresholds": _plain(decision),
    }


def _capture_visualization_case(
    cases: Dict[str, List[Dict[str, Any]]],
    *,
    batch: Mapping[str, Any],
    batch_index: int,
    support_probabilities: torch.Tensor,
    matches: Sequence,
    cfg: EasyDict,
) -> None:
    count = int(
        batch["branch_targets"]["branch_count"][batch_index].item())
    category = (
        "ordinary" if count == 1
        else "t_junction" if count == 2
        else "multi_branch" if count >= 3
        else None
    )
    if category is None:
        return
    limit = int(
        cfg.STAGE3D_C0.EVALUATION.VISUALIZATION_CASES_PER_TYPE)
    if len(cases[category]) >= limit:
        return
    prediction_indices, target_indices = matches[batch_index]
    cases[category].append({
        "sample_id": int(
            batch["metadata"]["dataset_index"][batch_index].item()),
        "aerial_image": batch["aerial_image"][
            batch_index].detach().cpu(),
        "traj_xy_norm": batch["trajectory_batch"]["traj_xy_norm"][
            batch_index].detach().cpu(),
        "point_mask": batch["trajectory_batch"]["point_mask"][
            batch_index].detach().cpu(),
        "fragment_mask": batch["trajectory_batch"]["fragment_mask"][
            batch_index].detach().cpu(),
        "branch_offsets_norm": batch["branch_targets"][
            "branch_offsets_norm"][batch_index].detach().cpu(),
        "branch_mask": batch["branch_targets"]["branch_mask"][
            batch_index].detach().cpu(),
        "support_probabilities": support_probabilities[
            batch_index].detach().cpu(),
        "prediction_indices": prediction_indices.detach().cpu(),
        "target_indices": target_indices.detach().cpu(),
    })


def render_visualizations(
    cases: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    output_dir: Path,
    cfg: EasyDict,
) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    half_window = float(cfg.TRAIN.WINDOW_SIZE) / 2.0
    top_k = int(cfg.STAGE3D_C0.EVALUATION.VISUALIZATION_TOP_K)
    paths = []
    for category, records in cases.items():
        for case_index, sample in enumerate(records):
            valid_fragments = sample["fragment_mask"].numpy().astype(bool)
            valid_indices = np.flatnonzero(valid_fragments)
            points = sample["traj_xy_norm"].numpy() * half_window
            point_mask = sample["point_mask"].numpy().astype(bool)
            branch_mask = sample["branch_mask"].numpy().astype(bool)
            endpoints = (
                sample["branch_offsets_norm"].numpy() * half_window)
            probabilities = sample["support_probabilities"].numpy()
            target_to_query = {
                int(target): int(query)
                for query, target in zip(
                    sample["prediction_indices"].tolist(),
                    sample["target_indices"].tolist(),
                )
            }
            valid_targets = np.flatnonzero(branch_mask)
            column_count = 1 + max(len(valid_targets), 1)
            figure, axes = plt.subplots(
                1,
                column_count,
                figsize=(5 * column_count, 5),
                squeeze=False,
            )
            image = np.clip(
                sample["aerial_image"].permute(1, 2, 0).numpy(),
                0.0,
                1.0,
            )
            for axis in axes[0]:
                axis.imshow(
                    image,
                    extent=(-half_window, half_window,
                            half_window, -half_window),
                    alpha=0.80,
                )
                axis.scatter([0.0], [0.0], c="red", s=28, zorder=5)
                axis.set_xlim(-half_window, half_window)
                axis.set_ylim(half_window, -half_window)
                axis.set_aspect("equal")
                axis.grid(alpha=0.15)
            all_axis = axes[0, 0]
            for fragment_index in valid_indices:
                fragment = points[
                    fragment_index, point_mask[fragment_index]]
                all_axis.plot(
                    fragment[:, 0], fragment[:, 1],
                    color="white", alpha=0.55, linewidth=0.7)
            for target_index in valid_targets:
                endpoint = endpoints[target_index]
                all_axis.arrow(
                    0.0, 0.0, endpoint[0], endpoint[1],
                    color="yellow", width=0.5, head_width=3.0,
                    length_includes_head=True)
            all_axis.set_title("64 fragments + GT immediate branches")

            for panel_index, target_index in enumerate(
                    valid_targets, start=1):
                axis = axes[0, panel_index]
                query_index = target_to_query.get(int(target_index))
                for fragment_index in valid_indices:
                    fragment = points[
                        fragment_index, point_mask[fragment_index]]
                    axis.plot(
                        fragment[:, 0], fragment[:, 1],
                        color="white", alpha=0.22, linewidth=0.5)
                if query_index is None:
                    axis.set_title(
                        "GT {}: no matched query".format(target_index))
                    continue
                order = valid_indices[np.argsort(
                    -probabilities[query_index, valid_indices],
                    kind="mergesort",
                )[:top_k]]
                colors = plt.cm.tab10(
                    np.linspace(0.0, 0.9, max(len(order), 1)))
                for rank, fragment_index in enumerate(order):
                    fragment = points[
                        fragment_index, point_mask[fragment_index]]
                    probability = probabilities[
                        query_index, fragment_index]
                    axis.plot(
                        fragment[:, 0], fragment[:, 1],
                        color=colors[rank],
                        linewidth=1.0 + 2.5 * probability,
                        label="#{} f{} p={:.2f}".format(
                            rank + 1, fragment_index, probability),
                    )
                endpoint = endpoints[target_index]
                axis.arrow(
                    0.0, 0.0, endpoint[0], endpoint[1],
                    color="yellow", width=0.6, head_width=3.0,
                    length_includes_head=True)
                axis.set_title(
                    "GT branch {} / query {} top-{}".format(
                        target_index, query_index, top_k))
                axis.legend(fontsize=6, loc="upper right")
            figure.suptitle(
                "{} sample {}".format(category, sample["sample_id"]))
            figure.tight_layout()
            path = output_dir / "{}_{:02d}.png".format(
                category, case_index)
            figure.savefig(str(path), dpi=170, bbox_inches="tight")
            plt.close(figure)
            paths.append(str(path.resolve()))
    return paths


def _write_readme(output_dir: Path, report: Mapping[str, Any]) -> None:
    variants = report["branch_metrics"]
    random_mean = report["random_aggregation_stability"]
    support_selection = report["trajectory_selection"]
    context = report["context_similarity"]
    decision = report["decision"]
    best_support = variants[decision["best_support_variant"]]
    original = variants[VARIANT_ORIGINAL]
    lines = [
        "# road_self Stage 3D-C0",
        "",
        "Frozen offline validation of support-guided trajectory context "
        "aggregation. No parameter was trained or modified.",
        "",
        "## Branch comparison",
        "",
        "| variant | branch AP | slot AP | endpoint mean px | "
        "direction mean deg | exact count | oracle duplicate | "
        "distinct coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    topk_names = [
        _topk_variant_name(top_k)
        for top_k in report["aggregation_top_ks"]
    ]
    for name in (
            VARIANT_ORIGINAL,
            VARIANT_NO_TRAJECTORY,
            VARIANT_SUPPORT,
            *topk_names):
        value = variants[name]
        thresholded = value["thresholded"]
        oracle = value["oracle_k"]
        lines.append(
            "| {name} | {ap:.4f} | {slot:.4f} | {endpoint:.2f} | "
            "{direction:.2f} | {count:.4f} | {duplicate:.4f} | "
            "{coverage:.4f} |".format(
                name=name,
                ap=value["branch_ap"],
                slot=value["slot_ap"],
                endpoint=thresholded["endpoint_error_mean_pixels"],
                direction=thresholded["direction_error_mean_degrees"],
                count=thresholded["exact_branch_count_accuracy"],
                duplicate=oracle["duplicates"]["duplicate_pair_ratio"],
                coverage=oracle["distinct_gt_coverage"],
            )
        )
    lines.append(
        "| random_aggregation (mean) | {ap:.4f} | {slot:.4f} | "
        "{endpoint:.2f} | {direction:.2f} | {count:.4f} | "
        "{duplicate:.4f} | {coverage:.4f} |".format(
            ap=random_mean["branch_ap"]["mean"],
            slot=random_mean["slot_ap"]["mean"],
            endpoint=random_mean[
                "endpoint_error_mean_pixels"]["mean"],
            direction=random_mean[
                "direction_error_mean_degrees"]["mean"],
            count=random_mean[
                "exact_branch_count_accuracy"]["mean"],
            duplicate=random_mean[
                "oracle_k_duplicate_pair_ratio"]["mean"],
            coverage=random_mean[
                "oracle_k_distinct_gt_coverage"]["mean"],
        )
    )
    lines.extend([
        "",
        "`full` is an explicit alias of frozen E4 `original_attention`.",
        "",
        "## Branch AP by road category",
        "",
        "| category | no trajectory | original | full support | "
        + " | ".join(topk_names) + " |",
        "| --- | " + " | ".join(
            ["---:"] * (3 + len(topk_names))) + " |",
    ])
    for category in (
            "ordinary", "t_junction", "multi_branch"):
        category_values = [
            variants[VARIANT_NO_TRAJECTORY][
                "by_category"][category]["branch_ap"],
            variants[VARIANT_ORIGINAL][
                "by_category"][category]["branch_ap"],
            variants[VARIANT_SUPPORT][
                "by_category"][category]["branch_ap"],
            *[
                variants[name]["by_category"][
                    category]["branch_ap"]
                for name in topk_names
            ],
        ]
        lines.append(
            "| {} | {} |".format(
                category,
                " | ".join(
                    "{:.4f}".format(value)
                    for value in category_values),
            )
        )
    lines.extend([
        "",
        "## Trajectory selection",
        "",
        "- support AP: **{:.4f}**".format(
            support_selection["support_ap"]),
        "- Precision@8 / Recall@8 / nDCG@8: **{:.4f} / {:.4f} / "
        "{:.4f}**".format(
            support_selection["precision_at"]["8"],
            support_selection["recall_at"]["8"],
            support_selection["ndcg_at"]["8"],
        ),
        "- query top-8 Jaccard median: **{}**".format(
            support_selection["predicted_top_k_jaccard"]["median"]),
        "- query overlap by top-k: {}".format(
            ", ".join(
                "k={} median={}".format(
                    top_k,
                    support_selection[
                        "query_topk_fragment_overlap"
                    ][str(top_k)]["median"],
                )
                for top_k in report["aggregation_top_ks"])),
        "",
        "## Context similarity",
        "",
        "- original inter-query cosine mean: **{}**".format(
            context["original_inter_query"]["pairwise_cosine"]["mean"]),
        "- support inter-query cosine mean: **{}**".format(
            context["support_inter_query"]["pairwise_cosine"]["mean"]),
        "- aligned original/support cosine mean: **{}**".format(
            context["aligned_original_support"]["mean"]),
        "- by category aligned cosine: {}".format(
            ", ".join(
                "{}={}".format(
                    category,
                    value["aligned_original_support"]["mean"],
                )
                for category, value in
                context["by_category"].items())),
        "",
        "## Decision",
        "",
        "- support - original branch AP: **{:+.4f}**".format(
            decision["support_minus_original_branch_ap"]),
        "- best support aggregation: **{}**".format(
            decision["best_support_variant"]),
        "- full-support - original branch AP: **{:+.4f}**".format(
            decision["full_support_minus_original_branch_ap"]),
        "- support - random mean branch AP: **{:+.4f}** "
        "({:.2f} random-baseline std)".format(
            decision["support_minus_random_mean_branch_ap"],
            decision["support_vs_random_standard_deviations"]),
        "- full - no-trajectory branch AP: **{:+.4f}**".format(
            decision["full_minus_no_trajectory_branch_ap"]),
        "- best-support slot AP change: **{:+.4f}**".format(
            best_support["slot_ap"] - original["slot_ap"]),
        "- best-support oracle distinct-coverage change: "
        "**{:+.4f}**".format(
            best_support["oracle_k"]["distinct_gt_coverage"]
            - original["oracle_k"]["distinct_gt_coverage"]),
        "- best-support oracle duplicate ratio: **{:.4f}**".format(
            best_support["oracle_k"]["duplicates"][
                "duplicate_pair_ratio"]),
        "- best-support category AP changes: {}".format(
            ", ".join(
                "{}={:+.4f}".format(
                    category,
                    best_support["by_category"][category]["branch_ap"]
                    - original["by_category"][category]["branch_ap"],
                )
                for category in (
                    "ordinary", "t_junction", "multi_branch"))),
        "- enter next-stage training: **{}**".format(
            "yes" if decision[
                "can_enter_next_stage_training"] else "no"),
        "- diagnosis: {}".format(decision["diagnosis"]),
        "",
        "The Stage 3D-A head reads final branch tokens and is therefore "
        "circular for online use. Its result is an offline diagnostic upper "
        "bound, not an implementation to feed into Path.push.",
        "",
        "RPNet, E4, the support head, anchor, branch GT, compression, and "
        "Path.push remained unchanged.",
    ])
    (output_dir / "README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/stage3d_c0_support_aggregation.yml"),
    )
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--e4-checkpoint", type=Path)
    parser.add_argument("--support-checkpoint", type=Path)
    parser.add_argument("--image-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = _load_config(args.config)
    if "STAGE3D" not in cfg or "STAGE3D_C0" not in cfg:
        raise ValueError("config must define STAGE3D and STAGE3D_C0")
    seed = int(cfg.STAGE3C.SEED)
    _set_seed(seed)
    device = _resolve_device(
        args.device or str(cfg.STAGE3C.DEVICE))
    dataset_dir = (
        args.dataset_dir or Path(cfg.STAGE3C.DATASET_DIR))
    output_dir = (
        args.output_dir or Path(cfg.STAGE3D_C0.OUTPUT_DIR))
    e4_checkpoint = (
        args.e4_checkpoint or Path(cfg.STAGE3D.E4_CHECKPOINT)
    ).resolve(strict=False)
    support_checkpoint = (
        args.support_checkpoint
        or Path(cfg.STAGE3D_C0.SUPPORT_CHECKPOINT)
    ).resolve(strict=False)
    image_checkpoint = (
        args.image_checkpoint or Path(cfg.STAGE3C.IMAGE_CHECKPOINT)
    ).resolve(strict=False)
    for name, path in (
            ("E4", e4_checkpoint),
            ("Stage 3D-A support", support_checkpoint),
            ("image-only RPNet", image_checkpoint)):
        if not path.is_file():
            raise FileNotFoundError(
                "{} checkpoint not found: {}".format(name, path))
    output_dir.mkdir(parents=True, exist_ok=True)

    val_dataset = Stage3CBranchDataset(
        dataset_dir, "val", preload=True)
    loader = DataLoader(
        val_dataset,
        batch_size=int(
            cfg.STAGE3D_C0.EVALUATION.BATCH_SIZE),
        shuffle=False,
        num_workers=int(
            cfg.STAGE3D_C0.EVALUATION.NUM_WORKERS),
        pin_memory=device.type == "cuda",
    )
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
    hidden_dim = int(cfg.STAGE3C.MODEL.HIDDEN_DIM)
    support_head = TrajectorySupportHead(
        hidden_dim=hidden_dim,
        projection_dim=int(cfg.STAGE3D.MODEL.PROJECTION_DIM),
    ).to(device=device)
    support_payload = load_stage3d_support_checkpoint(
        support_checkpoint,
        support_head=support_head,
        optimizer=None,
        map_location=device,
    )
    e4_sha256 = _sha256(e4_checkpoint)
    if support_payload["e4_checkpoint_sha256"] != e4_sha256:
        raise RuntimeError(
            "support checkpoint was not trained against the supplied E4: "
            "{} != {}".format(
                support_payload["e4_checkpoint_sha256"],
                e4_sha256,
            ))
    freeze_modules([rpnet, *modules, support_head])
    criterion = _build_branch_criterion(cfg)
    trajectory_encoder, graph_state_encoder, branch_decoder = modules

    aggregation_top_ks = tuple(sorted(set(
        int(value)
        for value in cfg.STAGE3D_C0.EVALUATION.AGGREGATION_TOP_KS
    )))
    if not aggregation_top_ks or any(
            value <= 0 for value in aggregation_top_ks):
        raise ValueError(
            "AGGREGATION_TOP_KS must contain positive integers")
    variants = {
        VARIANT_ORIGINAL: BranchVariantAccumulator(cfg),
        VARIANT_SUPPORT: BranchVariantAccumulator(cfg),
        VARIANT_NO_TRAJECTORY: BranchVariantAccumulator(cfg),
        **{
            _topk_variant_name(top_k):
                BranchVariantAccumulator(cfg)
            for top_k in aggregation_top_ks
        },
    }
    random_seeds = [
        int(value)
        for value in cfg.STAGE3D_C0.EVALUATION.RANDOM_SEEDS
    ]
    if not random_seeds or len(set(random_seeds)) != len(random_seeds):
        raise ValueError("RANDOM_SEEDS must be non-empty and unique")
    random_variants = {
        random_seed: BranchVariantAccumulator(cfg)
        for random_seed in random_seeds
    }
    ranking_ks = tuple(
        int(value)
        for value in cfg.STAGE3D_C0.EVALUATION.RANKING_KS)
    support_rankings = {
        top_k: TrajectorySupportRankingAccumulator(
            ranking_ks=ranking_ks,
            jaccard_k=top_k,
        )
        for top_k in aggregation_top_ks
    }
    context_chunks = {
        "original": [],
        "support": [],
        "gt_counts": [],
    }
    visual_cases: Dict[str, List[Dict[str, Any]]] = {
        "ordinary": [],
        "t_junction": [],
        "multi_branch": [],
    }
    max_original_recompute_diff = 0.0
    max_no_trajectory_recompute_diff = 0.0
    started_at = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
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
            original = branch_decoder(
                stage_fuse=stage_fuse,
                state_token=state_token,
                fragment_tokens=trajectory_output["fragment_tokens"],
                fragment_mask=trajectory_output["fragment_mask"],
                walked_path=batch["walked_path"],
                return_attention=True,
                return_debug_states=True,
            )
            original_recomputed = recompute_branch_predictions(
                branch_decoder,
                graph_conditioned_queries=original[
                    "debug_graph_conditioned_queries"],
                image_context=original[
                    "debug_image_cross_attention_output"],
                trajectory_context=original[
                    "debug_trajectory_cross_attention_output"],
                graph_state_contribution=original[
                    "debug_graph_state_contribution"],
            )
            max_original_recompute_diff = max(
                max_original_recompute_diff,
                _max_prediction_difference(
                    original, original_recomputed),
            )
            empty_mask = torch.zeros_like(
                trajectory_output["fragment_mask"])
            no_trajectory = branch_decoder(
                stage_fuse=stage_fuse,
                state_token=state_token,
                fragment_tokens=trajectory_output["fragment_tokens"],
                fragment_mask=empty_mask,
                walked_path=batch["walked_path"],
                return_attention=False,
                return_debug_states=False,
            )
            zero_context = torch.zeros_like(
                original["debug_trajectory_cross_attention_output"])
            no_trajectory_recomputed = recompute_branch_predictions(
                branch_decoder,
                graph_conditioned_queries=original[
                    "debug_graph_conditioned_queries"],
                image_context=original[
                    "debug_image_cross_attention_output"],
                trajectory_context=zero_context,
                graph_state_contribution=original[
                    "debug_graph_state_contribution"],
            )
            max_no_trajectory_recompute_diff = max(
                max_no_trajectory_recompute_diff,
                _max_prediction_difference(
                    no_trajectory, no_trajectory_recomputed),
            )

            support_logits = support_head(
                original["branch_tokens"],
                trajectory_output["fragment_tokens"],
                trajectory_output["fragment_mask"],
            )
            fragment_values = decoder_fragment_values(
                branch_decoder,
                trajectory_output["fragment_tokens"],
                trajectory_output["fragment_mask"],
            )
            support_pool = support_weighted_trajectory_context(
                support_logits,
                fragment_values,
                trajectory_output["fragment_mask"],
                epsilon=float(
                    cfg.STAGE3D_C0.EVALUATION.SUPPORT_EPSILON),
                output_projection=branch_decoder.
                    trajectory_cross_attention.out_proj,
            )
            support_predictions = recompute_branch_predictions(
                branch_decoder,
                graph_conditioned_queries=original[
                    "debug_graph_conditioned_queries"],
                image_context=original[
                    "debug_image_cross_attention_output"],
                trajectory_context=support_pool["context"],
                graph_state_contribution=original[
                    "debug_graph_state_contribution"],
            )
            targets = batch["branch_targets"]
            original_matches = criterion(
                original, targets)["matches"]
            support_matches = criterion(
                support_predictions, targets)["matches"]
            no_trajectory_matches = criterion(
                no_trajectory, targets)["matches"]
            variants[VARIANT_ORIGINAL].update(
                original, targets, original_matches)
            variants[VARIANT_SUPPORT].update(
                support_predictions, targets, support_matches)
            variants[VARIANT_NO_TRAJECTORY].update(
                no_trajectory, targets, no_trajectory_matches)

            for top_k in aggregation_top_ks:
                topk_context = support_weighted_trajectory_context(
                    support_logits,
                    fragment_values,
                    trajectory_output["fragment_mask"],
                    epsilon=float(
                        cfg.STAGE3D_C0.EVALUATION.SUPPORT_EPSILON),
                    output_projection=branch_decoder.
                        trajectory_cross_attention.out_proj,
                    top_k=top_k,
                )["context"]
                topk_predictions = recompute_branch_predictions(
                    branch_decoder,
                    graph_conditioned_queries=original[
                        "debug_graph_conditioned_queries"],
                    image_context=original[
                        "debug_image_cross_attention_output"],
                    trajectory_context=topk_context,
                    graph_state_contribution=original[
                        "debug_graph_state_contribution"],
                )
                topk_matches = criterion(
                    topk_predictions, targets)["matches"]
                variants[_topk_variant_name(top_k)].update(
                    topk_predictions, targets, topk_matches)

            sample_ids = batch["metadata"]["dataset_index"].to(
                dtype=torch.long)
            for random_seed, accumulator in random_variants.items():
                random_values = permute_valid_fragment_values(
                    fragment_values,
                    trajectory_output["fragment_mask"],
                    sample_ids,
                    seed=random_seed,
                )
                random_context = support_weighted_trajectory_context(
                    support_logits,
                    random_values,
                    trajectory_output["fragment_mask"],
                    epsilon=float(
                        cfg.STAGE3D_C0.EVALUATION.SUPPORT_EPSILON),
                    output_projection=branch_decoder.
                        trajectory_cross_attention.out_proj,
                )["context"]
                random_predictions = recompute_branch_predictions(
                    branch_decoder,
                    graph_conditioned_queries=original[
                        "debug_graph_conditioned_queries"],
                    image_context=original[
                        "debug_image_cross_attention_output"],
                    trajectory_context=random_context,
                    graph_state_contribution=original[
                        "debug_graph_state_contribution"],
                )
                random_matches = criterion(
                    random_predictions, targets)["matches"]
                accumulator.update(
                    random_predictions, targets, random_matches)

            matches = original_matches
            support_targets = build_trajectory_support_targets(
                batch["trajectory_batch"],
                targets,
                **_target_parameters(cfg),
            )
            for support_ranking in support_rankings.values():
                support_ranking.update(
                    scores=support_pool["support_probabilities"],
                    support_targets=support_targets["support_targets"],
                    support_positive_mask=support_targets[
                        "support_positive_mask"],
                    support_valid=support_targets["support_valid"],
                    branch_mask=targets["branch_mask"],
                    fragment_mask=trajectory_output["fragment_mask"],
                    matches=matches,
                    branch_count=targets["branch_count"],
                    sample_ids=sample_ids,
                )
            original_context = original[
                "debug_trajectory_cross_attention_output"]
            support_context = support_pool["context"]
            context_chunks["original"].append(
                original_context.detach().cpu().numpy())
            context_chunks["support"].append(
                support_context.detach().cpu().numpy())
            context_chunks["gt_counts"].append(
                targets["branch_count"].detach().cpu().numpy())
            for batch_index in range(stage_fuse.shape[0]):
                _capture_visualization_case(
                    visual_cases,
                    batch=batch,
                    batch_index=batch_index,
                    support_probabilities=support_pool[
                        "support_probabilities"],
                    matches=matches,
                    cfg=cfg,
                )

    tolerance = float(
        cfg.STAGE3D_C0.EVALUATION.NUMERICAL_EQUIVALENCE_TOLERANCE)
    if max_original_recompute_diff > tolerance:
        raise RuntimeError(
            "external original-context recompute is not numerically "
            "equivalent: {} > {}".format(
                max_original_recompute_diff, tolerance))
    if max_no_trajectory_recompute_diff > tolerance:
        raise RuntimeError(
            "external zero-context recompute is not numerically equivalent: "
            "{} > {}".format(
                max_no_trajectory_recompute_diff, tolerance))

    branch_results = {
        name: accumulator.compute()
        for name, accumulator in variants.items()
    }
    random_results = {
        str(random_seed): accumulator.compute()
        for random_seed, accumulator in random_variants.items()
    }
    random_stability = _random_summary({
        int(key): value
        for key, value in random_results.items()
    })
    support_ranking_results = {
        str(top_k): accumulator.compute()
        for top_k, accumulator in support_rankings.items()
    }
    primary_jaccard_k = int(
        cfg.STAGE3D_C0.EVALUATION.TOP_K_JACCARD)
    if str(primary_jaccard_k) not in support_ranking_results:
        raise ValueError(
            "TOP_K_JACCARD must be included in AGGREGATION_TOP_KS")
    trajectory_selection = support_ranking_results[
        str(primary_jaccard_k)]
    trajectory_selection["query_topk_fragment_overlap"] = {
        str(top_k): result["predicted_top_k_jaccard"]
        for top_k, result in support_ranking_results.items()
    }
    original_contexts = np.concatenate(
        context_chunks["original"], axis=0)
    support_contexts = np.concatenate(
        context_chunks["support"], axis=0)
    gt_counts = np.concatenate(
        context_chunks["gt_counts"], axis=0)
    context_similarity = _context_similarity_metrics(
        original_contexts, support_contexts)
    context_similarity["by_category"] = {
        name: _context_similarity_metrics(
            original_contexts[indices],
            support_contexts[indices],
        )
        for name, indices in {
            "ordinary": np.flatnonzero(gt_counts == 1),
            "t_junction": np.flatnonzero(gt_counts == 2),
            "multi_branch": np.flatnonzero(gt_counts >= 3),
        }.items()
    }
    support_candidate_names = [
        VARIANT_SUPPORT,
        *[
            _topk_variant_name(top_k)
            for top_k in aggregation_top_ks
        ],
    ]
    best_support_variant = max(
        support_candidate_names,
        key=lambda name: branch_results[name]["branch_ap"],
    )
    decision = assess_stage3d_c0(
        original_branch_ap=branch_results[
            VARIANT_ORIGINAL]["branch_ap"],
        support_branch_ap=branch_results[
            best_support_variant]["branch_ap"],
        no_trajectory_branch_ap=branch_results[
            VARIANT_NO_TRAJECTORY]["branch_ap"],
        support_selection_ap=float(
            trajectory_selection["support_ap"]),
        cfg=cfg,
        random_branch_ap_mean=float(
            random_stability["branch_ap"]["mean"]),
        random_branch_ap_std=float(
            random_stability["branch_ap"]["std"]),
    )
    decision.update({
        "best_support_variant": best_support_variant,
        "best_support_branch_ap": float(
            branch_results[best_support_variant]["branch_ap"]),
        "full_support_minus_original_branch_ap": float(
            branch_results[VARIANT_SUPPORT]["branch_ap"]
            - branch_results[VARIANT_ORIGINAL]["branch_ap"]),
    })
    visualizations = render_visualizations(
        visual_cases,
        output_dir=output_dir / "visualizations",
        cfg=cfg,
    )
    elapsed_seconds = float(time.perf_counter() - started_at)
    peak_cuda_bytes = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    report = {
        "schema_version": "stage3d-c0-v1",
        "config": str(args.config.resolve(strict=False)),
        "dataset_dir": str(dataset_dir.resolve(strict=False)),
        "validation_sample_count": len(val_dataset),
        "aggregation_top_ks": list(aggregation_top_ks),
        "checkpoints": {
            "image": str(image_checkpoint),
            "e4": str(e4_checkpoint),
            "e4_sha256": e4_sha256,
            "e4_epoch": int(e4_payload["epoch"]),
            "stage3d_a_support": str(support_checkpoint),
            "support_epoch": int(support_payload["epoch"]),
            "support_e4_sha256": support_payload[
                "e4_checkpoint_sha256"],
        },
        "branch_metrics": {
            "full": branch_results[VARIANT_ORIGINAL],
            **branch_results,
        },
        "full_is_original_attention_alias": True,
        "random_aggregation_by_seed": random_results,
        "random_aggregation_stability": random_stability,
        "trajectory_selection": trajectory_selection,
        "context_similarity": context_similarity,
        "numerical_equivalence": {
            "tolerance": tolerance,
            "original_context_recompute_max_abs_diff":
                max_original_recompute_diff,
            "zero_context_recompute_max_abs_diff":
                max_no_trajectory_recompute_diff,
            "passed": True,
        },
        "decision": decision,
        "visualizations": visualizations,
        "runtime": {
            "elapsed_seconds": elapsed_seconds,
            "device": str(device),
            "peak_cuda_memory_bytes": peak_cuda_bytes,
        },
        "constraints": {
            "all_modules_eval_and_frozen": True,
            "optimizer_created": False,
            "parameters_modified": False,
            "e4_decoder_source_modified": False,
            "branch_gt_modified": False,
            "trajectory_compression_modified": False,
            "anchor_modified": False,
            "path_push_modified": False,
            "support_context_fed_to_path_push": False,
            "stage3d_a_support_reads_final_branch_tokens": True,
        },
    }
    _write_json(output_dir / "summary.json", report)
    _write_readme(output_dir, report)
    print(json.dumps({
        "branch_ap": {
            name: value["branch_ap"]
            for name, value in branch_results.items()
        },
        "random_branch_ap": report[
            "random_aggregation_stability"]["branch_ap"],
        "support_selection_ap": trajectory_selection["support_ap"],
        "support_minus_original_branch_ap": decision[
            "support_minus_original_branch_ap"],
        "full_minus_no_trajectory_branch_ap": decision[
            "full_minus_no_trajectory_branch_ap"],
        "can_enter_next_stage_training": decision[
            "can_enter_next_stage_training"],
        "output_dir": str(output_dir.resolve(strict=False)),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
