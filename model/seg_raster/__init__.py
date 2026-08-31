"""Isolated trajectory-raster modules for road/junction segmentation."""

from .contracts import RasterFusionOutput
from .encoder import TrajectoryRasterEncoder
from .fusion import SegmentationOnlyRasterFusion
from .zero_preserving_road_adapter import (
    StrictZeroPreservingRoadAdapter,
    ZeroPreservingRoadOutput,
    validate_zero_preserving_contract,
)

__all__ = [
    "RasterFusionOutput",
    "SegmentationOnlyRasterFusion",
    "TrajectoryRasterEncoder",
    "StrictZeroPreservingRoadAdapter",
    "ZeroPreservingRoadOutput",
    "validate_zero_preserving_contract",
]
