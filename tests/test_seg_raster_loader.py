from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from pathlib import Path

from utils.seg_raster import (
    build_valid_mask,
    canonicalize_raster_array,
    load_trajectory_raster,
    validate_binary_raster_tensor,
    validate_raster_pair_shape,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _artifact_test_path(name: str) -> Path:
    return REPO_ROOT / "artifacts" / (name + "_stage_s2_pytest.png")


def test_raw_nonzero_values_become_binary_presence() -> None:
    raw = np.array([[0, 1, 128], [255, 0, 7]], dtype=np.uint8)
    path = _artifact_test_path("loader_values")
    try:
        Image.fromarray(raw).save(path)
        loaded = load_trajectory_raster(
            path, region_id="synthetic", expected_hw=(2, 3))
        assert loaded.raster.dtype == torch.float32
        assert set(torch.unique(loaded.raster).tolist()) == {0.0, 1.0}
        assert loaded.raster[0, 0].tolist() == [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]
        validate_binary_raster_tensor(loaded.raster, loaded.valid_mask)
    finally:
        path.unlink(missing_ok=True)


def test_full_canvas_valid_mask_distinguishes_padding() -> None:
    raw = np.full((8, 7), 128, dtype=np.uint8)
    binary, mask = canonicalize_raster_array(
        raw, valid_extent_wh=(4, 5))
    assert binary[:5, :4].all()
    assert not binary[5:, :].any()
    assert not binary[:, 4:].any()
    assert mask.sum() == 20


def test_upper_left_tile_of_valid_extent_has_all_valid_mask() -> None:
    full = build_valid_mask(8, 8, valid_extent_wh=(6, 7))
    assert np.all(full[:4, :4] == 1)


def test_loader_shape_and_pair_alignment_fail_fast() -> None:
    path = _artifact_test_path("loader_shape")
    try:
        Image.fromarray(np.zeros((4, 5), dtype=np.uint8)).save(path)
        with pytest.raises(ValueError, match="expected"):
            load_trajectory_raster(
                path, region_id="synthetic", expected_hw=(5, 4))
        with pytest.raises(ValueError, match="not spatially aligned"):
            validate_raster_pair_shape(
                torch.zeros(1, 3, 4, 5), torch.zeros(1, 1, 5, 4))
    finally:
        path.unlink(missing_ok=True)
