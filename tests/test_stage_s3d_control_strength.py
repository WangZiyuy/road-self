from __future__ import annotations

import numpy as np

from utils.seg_raster.stage_s3d import translate_zero_fill


def test_large_shift_reduces_aligned_overlap() -> None:
    raster = np.zeros((1, 1, 2048, 2048), dtype=np.uint8)
    raster[..., 500:800, 500:800] = 1
    shifted = translate_zero_fill(raster, (512, 512))
    intersection = np.logical_and(raster, shifted).sum()
    union = np.logical_or(raster, shifted).sum()
    assert intersection / union == 0
    assert shifted.sum() == raster.sum()
