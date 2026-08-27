"""Lightweight quarter-resolution encoder for binary trajectory presence."""

from __future__ import annotations

import torch
from torch import nn


class _ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int) -> None:
        groups = 4 if out_channels % 4 == 0 else 1
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride,
                padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )


class TrajectoryRasterEncoder(nn.Module):
    """Encode ``[B,1,H,W]`` binary presence to ``[B,C,H/4,W/4]``."""

    def __init__(self, output_channels: int = 32) -> None:
        super().__init__()
        self.output_channels = int(output_channels)
        self.downsample_1 = _ConvNormAct(1, 16, stride=2)
        self.downsample_2 = _ConvNormAct(16, 24, stride=2)
        self.depthwise = nn.Sequential(
            nn.Conv2d(24, 24, 3, padding=1, groups=24, bias=False),
            nn.GroupNorm(4, 24),
            nn.SiLU(inplace=True),
            nn.Conv2d(24, self.output_channels, 1, bias=False),
            nn.GroupNorm(4, self.output_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, raster: torch.Tensor) -> torch.Tensor:
        if raster.ndim != 4 or raster.shape[1] != 1:
            raise ValueError(
                "TrajectoryRasterEncoder expects [B,1,H,W]; got {}"
                .format(tuple(raster.shape))
            )
        return self.depthwise(
            self.downsample_2(self.downsample_1(raster)))
