"""Metrics for branch-conditioned trajectory-fragment support."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from utils.branch_diagnostics import (
    binary_average_precision,
    binary_auroc,
    distribution_statistics,
)


class TrajectorySupportMetricAccumulator:
    """Accumulate matched, support-valid branch/fragment predictions."""

    def __init__(
        self,
        *,
        recall_ks: Sequence[int] = (1, 4, 8, 16),
        jaccard_k: int = 8,
    ) -> None:
        if not recall_ks or any(int(value) <= 0 for value in recall_ks):
            raise ValueError("recall_ks must contain positive integers")
        if jaccard_k <= 0:
            raise ValueError("jaccard_k must be positive")
        self.recall_ks = tuple(sorted(set(int(value) for value in recall_ks)))
        self.jaccard_k = int(jaccard_k)
        self._scores: List[np.ndarray] = []
        self._attention: List[np.ndarray] = []
        self._labels: List[np.ndarray] = []
        self._soft_targets: List[np.ndarray] = []
        self._segment_only: List[np.ndarray] = []
        self._branch_records: List[Dict[str, object]] = []

    def update(
        self,
        *,
        support_logits: torch.Tensor,
        attention_weights: torch.Tensor,
        support_targets: torch.Tensor,
        support_positive_mask: torch.Tensor,
        support_valid: torch.Tensor,
        fragment_mask: torch.Tensor,
        segment_only: torch.Tensor,
        matches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        sample_ids: Optional[torch.Tensor] = None,
    ) -> None:
        probabilities = torch.sigmoid(support_logits).detach().cpu()
        attention = attention_weights.detach().cpu()
        targets = support_targets.detach().cpu()
        positive = support_positive_mask.detach().cpu().to(dtype=torch.bool)
        valid_branch = support_valid.detach().cpu().to(dtype=torch.bool)
        valid_fragment = fragment_mask.detach().cpu().to(dtype=torch.bool)
        segment_only = segment_only.detach().cpu().to(dtype=torch.bool)
        if sample_ids is None:
            sample_ids = torch.arange(probabilities.shape[0])
        sample_ids = sample_ids.detach().cpu().to(dtype=torch.long)

        for batch_index, (
                prediction_indices, target_indices) in enumerate(matches):
            for prediction_index, target_index in zip(
                    prediction_indices.detach().cpu().tolist(),
                    target_indices.detach().cpu().tolist()):
                if not bool(valid_branch[batch_index, target_index]):
                    continue
                mask = valid_fragment[batch_index]
                scores = probabilities[
                    batch_index, prediction_index, mask].numpy()
                attention_scores = attention[
                    batch_index, prediction_index, mask].numpy()
                labels = positive[
                    batch_index, target_index, mask].numpy()
                soft = targets[
                    batch_index, target_index, mask].numpy()
                segment = segment_only[batch_index, mask].numpy()
                if labels.size == 0 or not bool(labels.any()):
                    continue
                original_indices = np.flatnonzero(mask.numpy())
                self._scores.append(scores)
                self._attention.append(attention_scores)
                self._labels.append(labels)
                self._soft_targets.append(soft)
                self._segment_only.append(segment)
                self._branch_records.append({
                    "sample_id": int(sample_ids[batch_index]),
                    "target_index": int(target_index),
                    "scores": scores,
                    "labels": labels,
                    "original_indices": original_indices,
                })

    def compute(self) -> Dict[str, object]:
        if not self._scores:
            return {
                "branch_count": 0,
                "fragment_pair_count": 0,
                "support_ap": 0.0,
                "support_auroc": None,
                "attention_support_ap": 0.0,
                "attention_support_auroc": None,
                "recall_at": {
                    str(value): 0.0 for value in self.recall_ks},
                "positive_fragments_per_branch": distribution_statistics([]),
                "segment_only_positive_ratio": 0.0,
                "top_k_fragment_jaccard": {
                    "k": self.jaccard_k,
                    "pair_count": 0,
                    "mean": None,
                    "median": None,
                },
            }
        scores = np.concatenate(self._scores)
        attention = np.concatenate(self._attention)
        labels = np.concatenate(self._labels).astype(bool)
        segment_only = np.concatenate(self._segment_only).astype(bool)
        positive_counts = [
            int(values.sum()) for values in self._labels]

        recalls = {}
        for k in self.recall_ks:
            branch_recalls = []
            for branch_scores, branch_labels in zip(
                    self._scores, self._labels):
                limit = min(k, branch_scores.size)
                order = np.argsort(
                    -branch_scores, kind="mergesort")[:limit]
                branch_recalls.append(
                    float(branch_labels[order].sum())
                    / max(int(branch_labels.sum()), 1))
            recalls[str(k)] = float(np.mean(branch_recalls))

        by_sample = defaultdict(list)
        for record in self._branch_records:
            scores_for_branch = record["scores"]
            original_indices = record["original_indices"]
            limit = min(self.jaccard_k, scores_for_branch.size)
            selected = np.argsort(
                -scores_for_branch, kind="mergesort")[:limit]
            by_sample[record["sample_id"]].append(set(
                original_indices[selected].tolist()))
        jaccards = []
        for branch_sets in by_sample.values():
            for left_index in range(len(branch_sets)):
                for right_index in range(left_index + 1, len(branch_sets)):
                    union = branch_sets[left_index] | branch_sets[right_index]
                    intersection = (
                        branch_sets[left_index] & branch_sets[right_index])
                    jaccards.append(
                        float(len(intersection)) / max(len(union), 1))
        positive_segment_only = segment_only & labels
        return {
            "branch_count": len(self._branch_records),
            "fragment_pair_count": int(labels.size),
            "positive_fragment_pair_count": int(labels.sum()),
            "support_ap": float(binary_average_precision(scores, labels)),
            "support_auroc": _finite_or_none(
                binary_auroc(scores, labels)),
            "attention_support_ap": float(
                binary_average_precision(attention, labels)),
            "attention_support_auroc": _finite_or_none(
                binary_auroc(attention, labels)),
            "recall_at": recalls,
            "positive_fragments_per_branch":
                distribution_statistics(positive_counts),
            "segment_only_positive_ratio": float(
                positive_segment_only.sum() / max(int(labels.sum()), 1)),
            "top_k_fragment_jaccard": {
                "k": self.jaccard_k,
                "pair_count": len(jaccards),
                "mean": (
                    float(np.mean(jaccards)) if jaccards else None),
                "median": (
                    float(np.median(jaccards)) if jaccards else None),
            },
        }


def support_label_diagnostics(
    *,
    support_positive_mask: Iterable[np.ndarray],
    support_valid: Iterable[np.ndarray],
    branch_mask: Iterable[np.ndarray],
    segment_only_positive_mask: Iterable[np.ndarray],
) -> Dict[str, object]:
    positive = np.concatenate(
        [np.asarray(value, dtype=bool) for value in support_positive_mask],
        axis=0,
    )
    valid = np.concatenate(
        [np.asarray(value, dtype=bool) for value in support_valid],
        axis=0,
    )
    branches = np.concatenate(
        [np.asarray(value, dtype=bool) for value in branch_mask],
        axis=0,
    )
    segment_positive = np.concatenate(
        [
            np.asarray(value, dtype=bool)
            for value in segment_only_positive_mask
        ],
        axis=0,
    )
    valid = valid & branches
    gt_count = branches.sum(axis=1)
    branch_total = int(branches.sum())
    available = int(valid.sum())
    positive_counts = positive.sum(axis=-1)[branches]
    grouped = {}
    groups = {
        "0": gt_count == 0,
        "1": gt_count == 1,
        "2": gt_count == 2,
        ">=3": gt_count >= 3,
        ">=2": gt_count >= 2,
    }
    for name, sample_selection in groups.items():
        branch_selection = branches & sample_selection[:, None]
        denominator = int(branch_selection.sum())
        grouped[name] = {
            "sample_count": int(sample_selection.sum()),
            "branch_count": denominator,
            "available_branch_count": int(
                (valid & sample_selection[:, None]).sum()),
            "support_available_rate": (
                float((valid & sample_selection[:, None]).sum())
                / denominator if denominator else None),
        }
    segment_positive_count = int(segment_positive.sum())
    positive_count = int(positive.sum())
    return {
        "gt_branch_count": branch_total,
        "support_available_branch_count": available,
        "support_available_rate": (
            float(available) / branch_total if branch_total else 0.0),
        "bounded_64_branch_support_hit_rate": (
            float(available) / branch_total if branch_total else 0.0),
        # Retained for Stage 3D-A report compatibility.  This value only
        # states whether each branch has at least one positive fragment after
        # bounded-64 compression; it is not full-candidate recall.
        "bounded_64_oracle_support_recall": (
            float(available) / branch_total if branch_total else 0.0),
        "positive_fragments_per_gt_branch":
            distribution_statistics(positive_counts),
        "positive_fragment_pair_count": positive_count,
        "segment_only_positive_pair_count": segment_positive_count,
        "segment_only_positive_ratio": (
            float(segment_positive_count) / positive_count
            if positive_count else 0.0),
        "by_gt_branch_count": grouped,
    }


def _finite_or_none(value: float) -> Optional[float]:
    return float(value) if np.isfinite(value) else None
