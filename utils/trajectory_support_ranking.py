"""Ranking diagnostics for branch-conditioned trajectory support.

The accumulator accepts arbitrary fragment scores.  It is therefore shared
by the frozen E4 attention baseline, the Stage 3D-A post-fusion support head,
and the Stage 3D-B0 pre-trajectory support head.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from utils.branch_diagnostics import (
    binary_average_precision,
    binary_auroc,
    distribution_statistics,
)


MatchIndices = Tuple[torch.Tensor, torch.Tensor]


def _finite_or_none(value: float) -> Optional[float]:
    return float(value) if np.isfinite(value) else None


def _jaccard(left: set, right: set) -> float:
    union = left | right
    return float(len(left & right)) / max(len(union), 1)


def _mean_or_none(values: Sequence[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


class TrajectorySupportRankingAccumulator:
    """Accumulate threshold-free support rankings and invalid-branch scores."""

    def __init__(
        self,
        *,
        ranking_ks: Sequence[int] = (1, 4, 8, 16),
        jaccard_k: int = 8,
    ) -> None:
        if not ranking_ks or any(int(value) <= 0 for value in ranking_ks):
            raise ValueError("ranking_ks must contain positive integers")
        if jaccard_k <= 0:
            raise ValueError("jaccard_k must be positive")
        self.ranking_ks = tuple(
            sorted(set(int(value) for value in ranking_ks)))
        self.jaccard_k = int(jaccard_k)
        self._valid_records: List[Dict[str, object]] = []
        self._invalid_records: List[Dict[str, object]] = []

    def update(
        self,
        *,
        scores: torch.Tensor,
        support_targets: torch.Tensor,
        support_positive_mask: torch.Tensor,
        support_valid: torch.Tensor,
        branch_mask: torch.Tensor,
        fragment_mask: torch.Tensor,
        matches: Sequence[MatchIndices],
        branch_count: torch.Tensor,
        sample_ids: Optional[torch.Tensor] = None,
    ) -> None:
        """Add matched query/GT-branch records.

        ``scores`` must already be probabilities or ranking scores with shape
        ``[B, K, N]``.  Support-invalid GT branches are retained for
        diagnostics but never enter AP, ranking metrics, or training loss.
        """

        if scores.ndim != 3:
            raise ValueError("scores must have shape [B, K, N]")
        batch_size, _, fragment_count = scores.shape
        if (
                support_targets.ndim != 3
                or support_targets.shape[0] != batch_size
                or support_targets.shape[2] != fragment_count):
            raise ValueError(
                "support_targets must have shape [B, M, N]")
        if tuple(support_positive_mask.shape) != tuple(
                support_targets.shape):
            raise ValueError(
                "support_positive_mask must match support_targets")
        if tuple(support_valid.shape) != tuple(
                support_targets.shape[:2]):
            raise ValueError("support_valid must have shape [B, M]")
        if tuple(branch_mask.shape) != tuple(
                support_targets.shape[:2]):
            raise ValueError("branch_mask must have shape [B, M]")
        if tuple(fragment_mask.shape) != (
                batch_size, fragment_count):
            raise ValueError("fragment_mask must have shape [B, N]")
        if tuple(branch_count.shape) != (batch_size,):
            raise ValueError("branch_count must have shape [B]")
        if len(matches) != batch_size:
            raise ValueError("matches must contain one item per sample")

        scores = scores.detach().cpu()
        support_targets = support_targets.detach().cpu()
        support_positive_mask = support_positive_mask.detach().cpu().to(
            dtype=torch.bool)
        support_valid = support_valid.detach().cpu().to(dtype=torch.bool)
        branch_mask = branch_mask.detach().cpu().to(dtype=torch.bool)
        fragment_mask = fragment_mask.detach().cpu().to(dtype=torch.bool)
        branch_count = branch_count.detach().cpu().to(dtype=torch.long)
        if sample_ids is None:
            sample_ids = torch.arange(batch_size)
        sample_ids = sample_ids.detach().cpu().to(dtype=torch.long)

        for batch_index, (
                prediction_indices, target_indices) in enumerate(matches):
            for prediction_index, target_index in zip(
                    prediction_indices.detach().cpu().tolist(),
                    target_indices.detach().cpu().tolist()):
                if not bool(branch_mask[batch_index, target_index]):
                    continue
                valid_fragments = fragment_mask[batch_index]
                if not bool(valid_fragments.any()):
                    continue
                original_indices = np.flatnonzero(
                    valid_fragments.numpy())
                record = {
                    "sample_id": int(sample_ids[batch_index]),
                    "gt_branch_count": int(branch_count[batch_index]),
                    "target_index": int(target_index),
                    "scores": scores[
                        batch_index,
                        prediction_index,
                        valid_fragments,
                    ].numpy(),
                    "labels": support_positive_mask[
                        batch_index,
                        target_index,
                        valid_fragments,
                    ].numpy(),
                    "soft_targets": support_targets[
                        batch_index,
                        target_index,
                        valid_fragments,
                    ].numpy(),
                    "original_indices": original_indices,
                }
                if bool(support_valid[batch_index, target_index]):
                    self._valid_records.append(record)
                else:
                    self._invalid_records.append(record)

    def _ranking_metrics(
        self,
        records: Sequence[Mapping[str, object]],
    ) -> Dict[str, object]:
        if not records:
            return self._empty_ranking_metrics()
        all_scores = np.concatenate([
            np.asarray(record["scores"], dtype=np.float64)
            for record in records
        ])
        all_labels = np.concatenate([
            np.asarray(record["labels"], dtype=bool)
            for record in records
        ])
        metrics: Dict[str, object] = {
            "branch_count": len(records),
            "fragment_pair_count": int(all_labels.size),
            "positive_fragment_pair_count": int(all_labels.sum()),
            "support_ap": float(binary_average_precision(
                all_scores, all_labels)),
            "support_auroc": _finite_or_none(
                binary_auroc(all_scores, all_labels)),
            "positive_fragments_per_branch": distribution_statistics([
                int(np.asarray(record["labels"], dtype=bool).sum())
                for record in records
            ]),
        }
        precision_at: Dict[str, float] = {}
        recall_at: Dict[str, float] = {}
        hit_at: Dict[str, float] = {}
        mass_recall_at: Dict[str, float] = {}
        ndcg_at: Dict[str, float] = {}
        for k in self.ranking_ks:
            precisions = []
            recalls = []
            hits = []
            mass_recalls = []
            ndcgs = []
            for record in records:
                branch_scores = np.asarray(
                    record["scores"], dtype=np.float64)
                labels = np.asarray(record["labels"], dtype=bool)
                soft = np.asarray(
                    record["soft_targets"], dtype=np.float64)
                limit = min(k, branch_scores.size)
                order = np.argsort(
                    -branch_scores, kind="mergesort")[:limit]
                selected_positive = int(labels[order].sum())
                precisions.append(
                    float(selected_positive) / max(limit, 1))
                recalls.append(
                    float(selected_positive) / max(int(labels.sum()), 1))
                hits.append(float(selected_positive > 0))
                mass_recalls.append(
                    float(soft[order].sum())
                    / max(float(soft.sum()), 1e-12))
                discount = 1.0 / np.log2(
                    np.arange(limit, dtype=np.float64) + 2.0)
                selected_gain = np.exp2(soft[order]) - 1.0
                ideal_order = np.argsort(
                    -soft, kind="mergesort")[:limit]
                ideal_gain = np.exp2(soft[ideal_order]) - 1.0
                dcg = float((selected_gain * discount).sum())
                idcg = float((ideal_gain * discount).sum())
                ndcgs.append(dcg / max(idcg, 1e-12))
            key = str(k)
            precision_at[key] = float(np.mean(precisions))
            recall_at[key] = float(np.mean(recalls))
            hit_at[key] = float(np.mean(hits))
            mass_recall_at[key] = float(np.mean(mass_recalls))
            ndcg_at[key] = float(np.mean(ndcgs))
        metrics.update({
            "precision_at": precision_at,
            "recall_at": recall_at,
            "hit_at": hit_at,
            "soft_support_mass_recall_at": mass_recall_at,
            "ndcg_at": ndcg_at,
        })
        return metrics

    def _empty_ranking_metrics(self) -> Dict[str, object]:
        zeros = {str(k): 0.0 for k in self.ranking_ks}
        return {
            "branch_count": 0,
            "fragment_pair_count": 0,
            "positive_fragment_pair_count": 0,
            "support_ap": 0.0,
            "support_auroc": None,
            "positive_fragments_per_branch": distribution_statistics([]),
            "precision_at": dict(zeros),
            "recall_at": dict(zeros),
            "hit_at": dict(zeros),
            "soft_support_mass_recall_at": dict(zeros),
            "ndcg_at": dict(zeros),
        }

    def _pairwise_jaccards(
        self,
        *,
        predicted: bool,
    ) -> Dict[str, object]:
        by_sample: Dict[int, List[set]] = defaultdict(list)
        for record in self._valid_records:
            indices = np.asarray(
                record["original_indices"], dtype=np.int64)
            if predicted:
                scores = np.asarray(record["scores"], dtype=np.float64)
                limit = min(self.jaccard_k, scores.size)
                selection = np.argsort(
                    -scores, kind="mergesort")[:limit]
                selected_set = set(indices[selection].tolist())
            else:
                labels = np.asarray(record["labels"], dtype=bool)
                selected_set = set(indices[labels].tolist())
            by_sample[int(record["sample_id"])].append(selected_set)
        values = []
        for sets_for_sample in by_sample.values():
            for left_index in range(len(sets_for_sample)):
                for right_index in range(
                        left_index + 1, len(sets_for_sample)):
                    values.append(_jaccard(
                        sets_for_sample[left_index],
                        sets_for_sample[right_index],
                    ))
        return {
            "k": self.jaccard_k if predicted else None,
            "pair_count": len(values),
            "mean": _mean_or_none(values),
            "median": (
                float(np.median(values)) if values else None),
            "p90": (
                float(np.percentile(values, 90)) if values else None),
        }

    def _invalid_metrics(self) -> Dict[str, object]:
        if not self._invalid_records:
            return {
                "branch_count": 0,
                "fragment_pair_count": 0,
                "max_probability": None,
                "mean_probability": None,
                "top1_probability": distribution_statistics([]),
            }
        arrays = [
            np.asarray(record["scores"], dtype=np.float64)
            for record in self._invalid_records
        ]
        concatenated = np.concatenate(arrays)
        top1 = [float(array.max()) for array in arrays if array.size]
        return {
            "branch_count": len(arrays),
            "fragment_pair_count": int(concatenated.size),
            "max_probability": float(concatenated.max()),
            "mean_probability": float(concatenated.mean()),
            "top1_probability": distribution_statistics(top1),
        }

    def compute(self) -> Dict[str, object]:
        result = self._ranking_metrics(self._valid_records)
        grouped = {}
        for name, predicate in (
                ("1", lambda count: count == 1),
                ("2", lambda count: count == 2),
                (">=3", lambda count: count >= 3),
                (">=2", lambda count: count >= 2)):
            grouped[name] = self._ranking_metrics([
                record
                for record in self._valid_records
                if predicate(int(record["gt_branch_count"]))
            ])
        result.update({
            "by_gt_branch_count": grouped,
            "predicted_top_k_jaccard": self._pairwise_jaccards(
                predicted=True),
            "gt_positive_set_jaccard": self._pairwise_jaccards(
                predicted=False),
            "support_invalid": self._invalid_metrics(),
        })
        return result
