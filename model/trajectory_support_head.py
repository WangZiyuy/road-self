"""Independent branch-conditioned trajectory-fragment support head."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


MatchIndices = Tuple[torch.Tensor, torch.Tensor]


class TrajectorySupportHead(nn.Module):
    """Score every branch-query/fragment pair with projected dot products."""

    def __init__(
        self,
        hidden_dim: int = 128,
        projection_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        projection_dim = (
            hidden_dim if projection_dim is None else int(projection_dim))
        if projection_dim <= 0:
            raise ValueError("projection_dim must be positive")
        self.hidden_dim = int(hidden_dim)
        self.projection_dim = projection_dim
        self.branch_projection = nn.Linear(
            hidden_dim, projection_dim)
        self.fragment_projection = nn.Linear(
            hidden_dim, projection_dim)

    def forward(
        self,
        branch_tokens: torch.Tensor,
        fragment_tokens: torch.Tensor,
        fragment_mask: torch.Tensor,
    ) -> torch.Tensor:
        if branch_tokens.ndim != 3:
            raise ValueError("branch_tokens must have shape [B, K, D]")
        if fragment_tokens.ndim != 3:
            raise ValueError("fragment_tokens must have shape [B, N, D]")
        if branch_tokens.shape[0] != fragment_tokens.shape[0]:
            raise ValueError("branch and fragment batch sizes differ")
        if (
                branch_tokens.shape[-1] != self.hidden_dim
                or fragment_tokens.shape[-1] != self.hidden_dim):
            raise ValueError("token dimensions do not match hidden_dim")
        if tuple(fragment_mask.shape) != tuple(
                fragment_tokens.shape[:2]):
            raise ValueError("fragment_mask must have shape [B, N]")
        if (
                branch_tokens.device != fragment_tokens.device
                or fragment_tokens.device != fragment_mask.device):
            raise ValueError("support-head inputs must share one device")

        branch = self.branch_projection(branch_tokens)
        fragment = self.fragment_projection(fragment_tokens)
        logits = torch.matmul(
            branch, fragment.transpose(1, 2)
        ) / math.sqrt(float(self.projection_dim))
        return logits.masked_fill(
            ~fragment_mask.to(dtype=torch.bool).unsqueeze(1), 0.0)


def trajectory_support_bce_loss(
    support_logits: torch.Tensor,
    support_targets: torch.Tensor,
    support_valid: torch.Tensor,
    fragment_mask: torch.Tensor,
    matches: List[MatchIndices],
) -> Dict[str, torch.Tensor]:
    """Supervise matched queries only when their GT branch has support.

    Unmatched branch queries deliberately receive no support supervision in
    Stage 3D-A.  This keeps support learning separate from branch existence.
    """

    if support_logits.ndim != 3:
        raise ValueError("support_logits must have shape [B, K, N]")
    batch_size, _, fragment_count = support_logits.shape
    if (
            support_targets.ndim != 3
            or support_targets.shape[0] != batch_size
            or support_targets.shape[2] != fragment_count):
        raise ValueError(
            "support_targets must have shape [B, M, N]")
    if tuple(support_valid.shape) != tuple(
            support_targets.shape[:2]):
        raise ValueError("support_valid must have shape [B, M]")
    if tuple(fragment_mask.shape) != (
            batch_size, fragment_count):
        raise ValueError("fragment_mask must have shape [B, N]")
    if len(matches) != batch_size:
        raise ValueError("matches must contain one entry per sample")

    weighted_loss = support_logits.sum() * 0.0
    supervised_pair_count = 0
    supervised_branch_count = 0
    for batch_index, (
            prediction_indices, target_indices) in enumerate(matches):
        if prediction_indices.numel() != target_indices.numel():
            raise ValueError("match index lengths differ")
        for prediction_index, target_index in zip(
                prediction_indices.tolist(), target_indices.tolist()):
            if not bool(support_valid[batch_index, target_index]):
                continue
            valid_fragments = fragment_mask[batch_index].to(
                dtype=torch.bool)
            count = int(valid_fragments.sum().item())
            if count == 0:
                continue
            logits = support_logits[
                batch_index, prediction_index, valid_fragments]
            targets = support_targets[
                batch_index, target_index, valid_fragments].to(
                    device=logits.device, dtype=logits.dtype)
            weighted_loss = weighted_loss + (
                F.binary_cross_entropy_with_logits(
                    logits, targets, reduction="sum"))
            supervised_pair_count += count
            supervised_branch_count += 1
    if supervised_pair_count:
        loss = weighted_loss / float(supervised_pair_count)
    else:
        loss = support_logits.sum() * 0.0
    return {
        "loss": loss,
        "supervised_branch_count": torch.tensor(
            supervised_branch_count,
            device=support_logits.device,
            dtype=torch.int64,
        ),
        "supervised_pair_count": torch.tensor(
            supervised_pair_count,
            device=support_logits.device,
            dtype=torch.int64,
        ),
    }
