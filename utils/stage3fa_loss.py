"""Original VecRoad anchor supervision used by Stage 3F-A.

The official VecRoad training loop applies binary cross entropy with logits
to each valid recursive anchor map with ``reduction="sum"``. Stage 3F-A
keeps that per-sample spatial reduction, then averages samples so the update
scale does not depend on the cache batch size.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def original_vecroad_anchor_losses(
    anchor_logits: torch.Tensor,
    anchor_lowrs_logits: torch.Tensor,
    targets: torch.Tensor,
    end_indices: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Return official-logit BCE losses for valid recursive target maps."""
    if anchor_logits.shape != anchor_lowrs_logits.shape:
        raise ValueError("anchor and anchor_lowrs logits must have equal shape")
    if anchor_logits.shape != targets.shape:
        raise ValueError("anchor logits and targets must have equal shape")
    if anchor_logits.shape[0] != end_indices.shape[0]:
        raise ValueError("end_indices must contain one value per sample")
    if not bool(torch.isfinite(anchor_logits).all()) or not bool(
            torch.isfinite(anchor_lowrs_logits).all()):
        raise ValueError("anchor logits must be finite")
    if not bool(torch.isfinite(targets).all()):
        raise ValueError("anchor targets must be finite")
    if bool((targets < 0).any()) or bool((targets > 1).any()):
        raise ValueError(
            "anchor targets must be in [0, 1]; prediction/target order may be reversed")

    full_losses = []
    lowrs_losses = []
    for row in range(anchor_logits.shape[0]):
        end = int(end_indices[row])
        if end <= 0 or end > anchor_logits.shape[1]:
            raise ValueError("invalid Stage 3F-A supervision end index")
        target = targets[row, :end]
        full_losses.append(F.binary_cross_entropy_with_logits(
            anchor_logits[row, :end], target, reduction="sum"))
        lowrs_losses.append(F.binary_cross_entropy_with_logits(
            anchor_lowrs_logits[row, :end], target, reduction="sum"))

    full = torch.stack(full_losses).mean()
    lowrs = torch.stack(lowrs_losses).mean()
    return {
        "anchor_loss": full,
        "anchor_lowrs_loss": lowrs,
        "anchor_total_loss": full + lowrs,
    }
