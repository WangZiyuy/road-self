"""Strict zero-preserving trajectory-raster adapter for the road head only.

The module deliberately excludes image features from every trainable raster
transform.  Its learned residual is multiplied by a non-learned support mask
derived from the binary raster and by the valid mask.  Consequently a zero
raster produces a bitwise-zero residual for every parameter value, including
after optimizer updates.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ZeroPreservingRoadOutput:
    stage_fuse_road: torch.Tensor
    stage_fuse_junction: torch.Tensor
    raster_feature: torch.Tensor
    projected_raster: torch.Tensor
    support_mask: torch.Tensor
    valid_mask_downsampled: torch.Tensor
    residual: torch.Tensor


class StrictZeroPreservingRoadAdapter(nn.Module):
    """Map binary raster support to an additive road-only residual.

    No convolution has a bias and no affine normalization is used.  The
    ``bypass`` flag is reserved for the N0 null control.  It retains a
    differentiable zero dependency on every adapter parameter so N0 and N2
    receive identical zero gradients under deterministic training.
    """

    def __init__(
        self,
        image_channels: int = 128,
        hidden_channels: int = 32,
        projection_init: str = "default",
        use_support_multiplier: bool = True,
    ) -> None:
        super().__init__()
        self.image_channels = int(image_channels)
        self.hidden_channels = int(hidden_channels)
        if projection_init not in ("default", "zero"):
            raise ValueError("projection_init must be 'default' or 'zero'")
        self.projection_init = projection_init
        self.use_support_multiplier = bool(use_support_multiplier)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(16, self.hidden_channels, 3, stride=2,
                      padding=1, bias=False),
            nn.ReLU(inplace=False),
        )
        self.projection = nn.Conv2d(
            self.hidden_channels, self.image_channels, 1, bias=False)
        if self.projection_init == "zero":
            nn.init.zeros_(self.projection.weight)

    def _parameter_zero(self, reference: torch.Tensor) -> torch.Tensor:
        zero = reference.new_zeros(())
        for parameter in self.parameters():
            zero = zero + parameter.sum() * 0.0
        return zero

    def forward(
        self,
        stage_fuse_img: torch.Tensor,
        traj_binary: torch.Tensor,
        traj_valid_mask: torch.Tensor | None = None,
        *,
        bypass: bool = False,
    ) -> ZeroPreservingRoadOutput:
        if stage_fuse_img.ndim != 4 or stage_fuse_img.shape[1] != self.image_channels:
            raise ValueError(
                "stage_fuse_img must be [B,{},H,W]".format(
                    self.image_channels))
        if traj_binary.ndim != 4 or traj_binary.shape[1] != 1:
            raise ValueError("traj_binary must be [B,1,H,W]")
        if traj_binary.shape[0] != stage_fuse_img.shape[0]:
            raise ValueError("image and raster batch sizes differ")
        if not torch.all((traj_binary == 0) | (traj_binary == 1)):
            raise ValueError("traj_binary values must belong to {0,1}")
        if traj_valid_mask is None:
            traj_valid_mask = torch.ones_like(traj_binary)
        elif tuple(traj_valid_mask.shape) != tuple(traj_binary.shape):
            raise ValueError("traj_valid_mask shape must match traj_binary")

        valid = (traj_valid_mask > 0).to(
            device=traj_binary.device, dtype=traj_binary.dtype)
        masked_raster = traj_binary * valid
        target_size = stage_fuse_img.shape[-2:]
        support = F.adaptive_max_pool2d(masked_raster, target_size)
        valid_down = F.adaptive_max_pool2d(valid, target_size)
        feature = self.encoder(masked_raster)
        if feature.shape[-2:] != target_size:
            feature = F.interpolate(
                feature, size=target_size, mode="bilinear",
                align_corners=False)
        projected = self.projection(feature)
        multiplier = support if self.use_support_multiplier else torch.ones_like(support)
        residual = multiplier * valid_down * projected
        if bypass:
            residual = torch.zeros_like(residual) + self._parameter_zero(residual)
        stage_fuse_road = stage_fuse_img + residual
        return ZeroPreservingRoadOutput(
            stage_fuse_road=stage_fuse_road,
            stage_fuse_junction=stage_fuse_img,
            raster_feature=feature,
            projected_raster=projected,
            support_mask=support,
            valid_mask_downsampled=valid_down,
            residual=residual,
        )


def validate_zero_preserving_contract(module: nn.Module) -> None:
    """Fail fast if a future edit introduces a bias or affine normalization."""
    for name, child in module.named_modules():
        if isinstance(child, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            if child.bias is not None:
                raise ValueError("zero-preserving transform has bias: " + name)
        if isinstance(child, (
            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
            nn.GroupNorm, nn.LayerNorm,
        )):
            raise ValueError(
                "zero-preserving transform may not contain normalization: "
                + name)
