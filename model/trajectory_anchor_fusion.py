"""Zero-initialized trajectory-evidence residuals for VecRoad anchors.

This module is deliberately external to :class:`RPNet`: it consumes the two
frozen anchor-only pre-head features exposed by the original forward pass and
adds residual logits through the original frozen output convolutions.  It
cannot affect road/junction heads, recursive anchor feedback, or Path.push.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _availability_tensor(
    available: Optional[torch.Tensor],
    evidence: torch.Tensor,
) -> torch.Tensor:
    if available is None:
        return torch.ones(
            evidence.shape[0], device=evidence.device,
            dtype=evidence.dtype)
    if available.ndim != 1 or available.shape[0] != evidence.shape[0]:
        raise ValueError("trajectory_available must have shape [B]")
    return available.to(device=evidence.device, dtype=evidence.dtype)


class TrajectorySpatialResidualAdapter(nn.Module):
    """Turn one projected trajectory token into a spatial residual."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(2 * channels, channels, 1)
        self.output = nn.Conv2d(channels, channels, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        feature: torch.Tensor,
        projected_trajectory: torch.Tensor,
    ) -> torch.Tensor:
        if feature.ndim != 4:
            raise ValueError("anchor feature must have shape [B,C,H,W]")
        if projected_trajectory.shape != feature.shape[:2]:
            raise ValueError("projected trajectory must have shape [B,C]")
        token_map = projected_trajectory[:, :, None, None].expand(
            -1, -1, feature.shape[-2], feature.shape[-1])
        hidden = F.gelu(self.reduce(torch.cat([feature, token_map], dim=1)))
        return self.output(hidden)


class ZeroInitializedTrajectoryAnchorFusion(nn.Module):
    """Shared trajectory projection/gate with two zero-init anchor adapters."""

    def __init__(
        self,
        evidence_dim: int = 128,
        anchor_channels: int = 32,
        gate_hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.evidence_dim = int(evidence_dim)
        self.anchor_channels = int(anchor_channels)
        self.trajectory_projection = nn.Linear(
            self.evidence_dim, self.anchor_channels)
        self.trajectory_norm = nn.LayerNorm(self.anchor_channels)
        self.gate = nn.Sequential(
            nn.Linear(
                self.evidence_dim + self.anchor_channels,
                int(gate_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(gate_hidden_dim), 1),
        )
        self.anchor_adapter = TrajectorySpatialResidualAdapter(
            self.anchor_channels)
        self.anchor_lowrs_adapter = TrajectorySpatialResidualAdapter(
            self.anchor_channels)

    def project(self, evidence: torch.Tensor) -> torch.Tensor:
        if evidence.ndim == 3:
            if evidence.shape[1] != 1:
                raise ValueError(
                    "Stage 3F-A requires exactly one evidence token")
            evidence = evidence[:, 0]
        if evidence.ndim != 2 or evidence.shape[1] != self.evidence_dim:
            raise ValueError("trajectory evidence must have shape [B,1,128]")
        return F.gelu(self.trajectory_norm(
            self.trajectory_projection(evidence)))

    def _fuse_one(
        self,
        feature: torch.Tensor,
        evidence_vector: torch.Tensor,
        projected: torch.Tensor,
        available: torch.Tensor,
        adapter: TrajectorySpatialResidualAdapter,
    ) -> Dict[str, torch.Tensor]:
        pooled = feature.mean(dim=(-2, -1))
        gate = torch.sigmoid(self.gate(torch.cat(
            [evidence_vector, pooled], dim=-1)))[:, :, None, None]
        delta = adapter(feature, projected)
        fused = feature + available[:, None, None, None] * gate * delta
        return {"feature": fused, "delta": delta, "gate": gate}

    def forward(
        self,
        *,
        anchor_feature: torch.Tensor,
        anchor_lowrs_feature: torch.Tensor,
        trajectory_evidence: torch.Tensor,
        trajectory_available: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        evidence = (
            trajectory_evidence[:, 0]
            if trajectory_evidence.ndim == 3 else trajectory_evidence)
        projected = self.project(trajectory_evidence)
        available = _availability_tensor(
            trajectory_available, evidence)
        anchor = self._fuse_one(
            anchor_feature, evidence, projected, available,
            self.anchor_adapter)
        lowrs = self._fuse_one(
            anchor_lowrs_feature, evidence, projected, available,
            self.anchor_lowrs_adapter)
        return {
            "anchor_feature": anchor["feature"],
            "anchor_delta": anchor["delta"],
            "anchor_gate": anchor["gate"],
            "anchor_lowrs_feature": lowrs["feature"],
            "anchor_lowrs_delta": lowrs["delta"],
            "anchor_lowrs_gate": lowrs["gate"],
        }


def fuse_cached_anchor_logits(
    *,
    fusion: ZeroInitializedTrajectoryAnchorFusion,
    anchor_features: torch.Tensor,
    anchor_lowrs_features: torch.Tensor,
    original_anchor_logits: torch.Tensor,
    original_anchor_lowrs_logits: torch.Tensor,
    trajectory_evidence: torch.Tensor,
    trajectory_available: torch.Tensor,
    anchor_head_weight: torch.Tensor,
    anchor_lowrs_head_weight: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Apply pre-head feature residuals while preserving exact base logits.

    Features have shape ``[B,S,C,H,W]``.  The frozen head bias is omitted for
    the residual term because it is already present in ``original_*``.
    """

    if anchor_features.ndim != 5 or anchor_lowrs_features.ndim != 5:
        raise ValueError("cached anchor features must have shape [B,S,C,H,W]")
    batch_size, steps = anchor_features.shape[:2]
    if original_anchor_logits.shape[:2] != (batch_size, steps):
        raise ValueError("original anchor logits do not match cached features")
    evidence = trajectory_evidence[:, None].expand(
        -1, steps, -1, -1).reshape(batch_size * steps, 1, -1)
    available = trajectory_available[:, None].expand(
        -1, steps).reshape(batch_size * steps)
    full = anchor_features.reshape(
        batch_size * steps, *anchor_features.shape[2:])
    lowrs = anchor_lowrs_features.reshape(
        batch_size * steps, *anchor_lowrs_features.shape[2:])
    fused = fusion(
        anchor_feature=full,
        anchor_lowrs_feature=lowrs,
        trajectory_evidence=evidence,
        trajectory_available=available,
    )
    full_delta = fused["anchor_feature"] - full
    lowrs_delta = fused["anchor_lowrs_feature"] - lowrs
    full_logit_delta = F.conv2d(
        full_delta, anchor_head_weight, bias=None, padding=1)
    lowrs_logit_delta = F.conv2d(
        lowrs_delta, anchor_lowrs_head_weight, bias=None)
    lowrs_logit_delta = F.interpolate(
        lowrs_logit_delta, scale_factor=4, mode="bilinear",
        align_corners=True)
    anchor = original_anchor_logits + full_logit_delta.reshape_as(
        original_anchor_logits)
    anchor_lowrs = (
        original_anchor_lowrs_logits
        + lowrs_logit_delta.reshape_as(original_anchor_lowrs_logits))
    return {
        "anchor": anchor,
        "anchor_lowrs": anchor_lowrs,
        "anchor_feature_delta": full_delta,
        "anchor_lowrs_feature_delta": lowrs_delta,
        "anchor_gate": fused["anchor_gate"].reshape(
            batch_size, steps, 1, 1, 1),
        "anchor_lowrs_gate": fused["anchor_lowrs_gate"].reshape(
            batch_size, steps, 1, 1, 1),
    }


def collect_anchor_prehead_features(
    feature_maps: Mapping[str, torch.Tensor],
    num_targets: int,
) -> Dict[str, torch.Tensor]:
    """Stack exact frozen pre-head tensors exposed by RPNet."""

    full = [feature_maps["decoded_ft_1_step_{}".format(index)]
            for index in range(int(num_targets))]
    lowrs = [feature_maps["next_step_step_{}".format(index)]
             for index in range(int(num_targets))]
    return {
        "anchor_features": torch.stack(full, dim=1),
        "anchor_lowrs_features": torch.stack(lowrs, dim=1),
    }
