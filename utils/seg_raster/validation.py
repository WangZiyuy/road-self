"""Fail-fast validation for canonical raster batches."""

from __future__ import annotations

import torch


def validate_binary_raster_tensor(
    raster: torch.Tensor,
    valid_mask: torch.Tensor,
) -> None:
    if raster.ndim != 4 or raster.shape[1] != 1:
        raise ValueError("canonical raster must have shape [B,1,H,W]")
    if tuple(valid_mask.shape) != tuple(raster.shape):
        raise ValueError("canonical raster and valid mask shapes must match")
    if raster.dtype != torch.float32:
        raise TypeError("canonical raster must use float32")
    if not torch.all((raster == 0) | (raster == 1)):
        raise ValueError("canonical raster values must belong to {0,1}")
    if not torch.all((valid_mask == 0) | (valid_mask == 1)):
        raise ValueError("valid-mask values must belong to {0,1}")
    if torch.any(raster[valid_mask == 0] != 0):
        raise ValueError("canonical raster must be zero outside valid extent")


def validate_raster_pair_shape(
    aerial: torch.Tensor,
    raster: torch.Tensor,
) -> None:
    if aerial.ndim != 4 or raster.ndim != 4:
        raise ValueError("aerial and raster inputs must be BCHW tensors")
    if aerial.shape[0] != raster.shape[0] or aerial.shape[-2:] != raster.shape[-2:]:
        raise ValueError(
            "aerial {} and trajectory raster {} are not spatially aligned"
            .format(tuple(aerial.shape), tuple(raster.shape))
        )
