from __future__ import annotations

import numpy as np
import torch

from utils.seg_raster.stage_s3d import tensor_statistics, translate_zero_fill


def test_large_shift_is_zero_fill_without_wrap() -> None:
    raster = np.zeros((1, 1, 1024, 1024), dtype=np.float32)
    raster[..., 10, 20] = 1
    raster[..., 900, 900] = 1
    shifted = translate_zero_fill(raster, (512, 512))
    assert shifted[..., 522, 532] == 1
    assert shifted[..., 388, 388] == 0
    assert shifted.sum() == 1


def test_forensic_statistics_are_finite_and_checksummed() -> None:
    stats = tensor_statistics(torch.arange(16, dtype=torch.float32).reshape(
        1, 1, 4, 4))
    assert stats["nonzero_ratio"] == 15 / 16
    assert stats["l2_norm"] > 0
    assert len(stats["sha256"]) == 64
