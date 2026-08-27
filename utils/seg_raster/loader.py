"""One canonical loader for Stage S2 trajectory rasters.

The trusted Xi'an source may contain multiple non-zero intensity levels, but
the Stage S2 model contract has exactly one interpretation: non-zero means
trajectory presence.  This module is shared by training and inference adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class TrajectoryRasterInput:
    raster: torch.Tensor
    valid_mask: torch.Tensor
    region_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


def _as_single_channel(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[-1] == 1:
        return array[..., 0]
    raise ValueError(
        "trajectory raster must be HxW or HxWx1; got shape {}"
        .format(tuple(array.shape))
    )


def build_valid_mask(
    height: int,
    width: int,
    valid_extent_wh: tuple[int, int] | None = None,
) -> np.ndarray:
    """Build an upper-left valid-data mask for a padded canonical canvas."""
    if height <= 0 or width <= 0:
        raise ValueError("raster height and width must be positive")
    mask = np.ones((height, width), dtype=np.float32)
    if valid_extent_wh is None:
        return mask
    valid_width, valid_height = map(int, valid_extent_wh)
    if not (0 <= valid_width <= width and 0 <= valid_height <= height):
        raise ValueError(
            "valid extent {} exceeds raster canvas {}"
            .format((valid_width, valid_height), (width, height))
        )
    mask.fill(0.0)
    mask[:valid_height, :valid_width] = 1.0
    return mask


def canonicalize_raster_array(
    raw: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    valid_extent_wh: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return float32 binary presence and valid mask arrays in HxW order."""
    raw_2d = _as_single_channel(raw)
    if valid_mask is not None and valid_extent_wh is not None:
        raise ValueError("provide valid_mask or valid_extent_wh, not both")
    if valid_mask is None:
        mask = build_valid_mask(
            raw_2d.shape[0], raw_2d.shape[1], valid_extent_wh)
    else:
        mask = _as_single_channel(np.asarray(valid_mask)).astype(
            np.float32, copy=False)
        if mask.shape != raw_2d.shape:
            raise ValueError(
                "trajectory valid mask shape {} does not match raster shape {}"
                .format(mask.shape, raw_2d.shape)
            )
        mask = (mask > 0).astype(np.float32, copy=False)
    binary = (raw_2d > 0).astype(np.float32, copy=False)
    binary *= mask
    return np.ascontiguousarray(binary), np.ascontiguousarray(mask)


def canonicalize_raster_tensor(
    raw: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Canonicalize HxW/CHW/BCHW tensors while preserving device and layout."""
    if raw.ndim == 2:
        raw = raw.unsqueeze(0).unsqueeze(0)
    elif raw.ndim == 3:
        raw = raw.unsqueeze(0)
    if raw.ndim != 4 or raw.shape[1] != 1:
        raise ValueError(
            "trajectory raster tensor must resolve to [B,1,H,W]; got {}"
            .format(tuple(raw.shape))
        )
    binary = (raw > 0).to(dtype=torch.float32)
    if valid_mask is None:
        mask = torch.ones_like(binary)
    else:
        if valid_mask.ndim == 2:
            valid_mask = valid_mask.unsqueeze(0).unsqueeze(0)
        elif valid_mask.ndim == 3:
            valid_mask = valid_mask.unsqueeze(0)
        if tuple(valid_mask.shape) != tuple(binary.shape):
            raise ValueError(
                "trajectory valid mask shape {} does not match raster shape {}"
                .format(tuple(valid_mask.shape), tuple(binary.shape))
            )
        mask = (valid_mask > 0).to(device=raw.device, dtype=torch.float32)
    return binary * mask, mask


def load_trajectory_raster(
    path: str | Path,
    *,
    region_id: str,
    expected_hw: tuple[int, int] | None = None,
    valid_extent_wh: tuple[int, int] | None = None,
    device: torch.device | str | None = None,
) -> TrajectoryRasterInput:
    """Load one raster as canonical BCHW float32 binary presence."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError("trajectory raster not found: {}".format(source))
    raw = np.asarray(Image.open(source).convert("L"))
    if expected_hw is not None and tuple(raw.shape) != tuple(expected_hw):
        raise ValueError(
            "trajectory raster shape {} does not match expected HxW {}"
            .format(tuple(raw.shape), tuple(expected_hw))
        )
    binary, mask = canonicalize_raster_array(
        raw, valid_extent_wh=valid_extent_wh)
    raster_tensor = torch.from_numpy(binary)[None, None]
    mask_tensor = torch.from_numpy(mask)[None, None]
    if device is not None:
        raster_tensor = raster_tensor.to(device)
        mask_tensor = mask_tensor.to(device)
    return TrajectoryRasterInput(
        raster=raster_tensor,
        valid_mask=mask_tensor,
        region_ids=[str(region_id)],
        metadata={
            "path": str(source),
            "raw_shape_hw": list(raw.shape),
            "canonical_shape_bchw": list(raster_tensor.shape),
            "input_semantics": "binary_presence",
            "conversion": "traj_binary = (traj_raw > 0).astype(float32)",
            "valid_extent_wh": (
                list(valid_extent_wh) if valid_extent_wh is not None else None),
        },
    )
