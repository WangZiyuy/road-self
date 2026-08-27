"""Residual fusion whose output is consumed only by segmentation heads."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .contracts import RasterFusionOutput
from .encoder import TrajectoryRasterEncoder


class SegmentationOnlyRasterFusion(nn.Module):
    def __init__(
        self,
        image_channels: int = 128,
        trajectory_channels: int = 32,
        use_valid_mask: bool = True,
    ) -> None:
        super().__init__()
        self.image_channels = int(image_channels)
        self.use_valid_mask = bool(use_valid_mask)
        self.raster_encoder = TrajectoryRasterEncoder(trajectory_channels)
        self.traj_projection = nn.Conv2d(
            trajectory_channels, trajectory_channels, 1, bias=False)
        fusion_channels = self.image_channels + trajectory_channels
        if self.use_valid_mask:
            fusion_channels += 1
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_channels, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
        )
        self.delta_projection = nn.Conv2d(64, self.image_channels, 1)
        nn.init.zeros_(self.delta_projection.weight)
        nn.init.zeros_(self.delta_projection.bias)

    def forward(
        self,
        stage_fuse_img: torch.Tensor,
        traj_binary: torch.Tensor,
        traj_valid_mask: torch.Tensor | None = None,
    ) -> RasterFusionOutput:
        if stage_fuse_img.ndim != 4 or stage_fuse_img.shape[1] != self.image_channels:
            raise ValueError(
                "stage_fuse_img must be [B,{},H/4,W/4]; got {}"
                .format(self.image_channels, tuple(stage_fuse_img.shape))
            )
        if traj_binary.ndim != 4 or traj_binary.shape[1] != 1:
            raise ValueError("traj_binary must have shape [B,1,H,W]")
        if traj_binary.shape[0] != stage_fuse_img.shape[0]:
            raise ValueError("image and trajectory raster batch sizes differ")
        if not torch.all((traj_binary == 0) | (traj_binary == 1)):
            raise ValueError("traj_binary values must belong to {0,1}")

        if traj_valid_mask is None:
            traj_valid_mask = torch.ones_like(traj_binary)
        elif tuple(traj_valid_mask.shape) != tuple(traj_binary.shape):
            raise ValueError("traj_valid_mask shape must match traj_binary")
        traj_valid_mask = (traj_valid_mask > 0).to(
            device=traj_binary.device, dtype=traj_binary.dtype)
        traj_binary = traj_binary * traj_valid_mask

        traj_feature = self.raster_encoder(traj_binary)
        if traj_feature.shape[-2:] != stage_fuse_img.shape[-2:]:
            traj_feature = F.interpolate(
                traj_feature,
                size=stage_fuse_img.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        projected_traj = self.traj_projection(traj_feature)
        inputs = [stage_fuse_img, projected_traj]
        valid_mask_downsampled = None
        if self.use_valid_mask:
            valid_mask_downsampled = F.interpolate(
                traj_valid_mask,
                size=stage_fuse_img.shape[-2:],
                mode="nearest",
            )
            inputs.append(valid_mask_downsampled)
        delta = self.delta_projection(self.fusion(torch.cat(inputs, dim=1)))
        return RasterFusionOutput(
            stage_fuse_seg=stage_fuse_img + delta,
            traj_feature=traj_feature,
            projected_traj=projected_traj,
            valid_mask_downsampled=valid_mask_downsampled,
            delta=delta,
        )
