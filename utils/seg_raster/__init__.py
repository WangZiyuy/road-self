"""Canonical trajectory-raster loading and validation."""

from .loader import (
    TrajectoryRasterInput,
    build_valid_mask,
    canonicalize_raster_array,
    canonicalize_raster_tensor,
    load_trajectory_raster,
)
from .manifest import describe_canonical_raster, sha256_file
from .validation import validate_binary_raster_tensor, validate_raster_pair_shape

__all__ = [
    "TrajectoryRasterInput",
    "build_valid_mask",
    "canonicalize_raster_array",
    "canonicalize_raster_tensor",
    "describe_canonical_raster",
    "load_trajectory_raster",
    "sha256_file",
    "validate_binary_raster_tensor",
    "validate_raster_pair_shape",
]
