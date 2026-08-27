"""Typed output contract for isolated segmentation raster fusion."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RasterFusionOutput:
    stage_fuse_seg: torch.Tensor
    traj_feature: torch.Tensor
    projected_traj: torch.Tensor
    valid_mask_downsampled: torch.Tensor | None
    delta: torch.Tensor
