from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model.model import RPNet
from utils.seg_raster import canonicalize_raster_array


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_infer_wiring_uses_canonical_loader_without_sequence_requirement() -> None:
    source = (REPO_ROOT / "infer.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "load_trajectory_raster" in source
    assert "trajectory_mode=TRAJECTORY_MODE" in source
    assert "traj_valid_mask=" in source
    assert "enable_trajectory_modules=USE_SEQUENCE" in source
    assert "enable_raster_segmentation=USE_SEG_RASTER" in source
    assert isinstance(tree, ast.Module)


def test_xian_canonical_raster_crop_runs_segmentation_infer_smoke() -> None:
    raster_path = REPO_ROOT / "data_self/input/traj_test/xian.png"
    if raster_path.is_file():
        raw = np.asarray(Image.open(raster_path).convert("L").crop((0, 0, 128, 128)))
    else:
        raw = np.zeros((128, 128), dtype=np.uint8)
        raw[16:112, 60:68] = 255
    binary, valid = canonicalize_raster_array(raw)
    raster = torch.from_numpy(binary)[None, None]
    mask = torch.from_numpy(valid)[None, None]
    net = RPNet(
        num_targets=1,
        backbone_pretrained=False,
        enable_raster_segmentation=True,
    ).eval()
    with torch.no_grad():
        output = net(
            aerial_image=torch.zeros(1, 3, 128, 128),
            traj_image=raster,
            aerial_traj_image=None,
            neighborhood_trajectory_norm=None,
            valid_mask=None,
            walked_path=None,
            test=True,
            model="origin",
            use_traj=False,
            trajectory_mode="raster_seg_only",
            traj_valid_mask=mask,
        )
    assert output["road"].shape == (1, 1, 128, 128)
    assert output["junc"].shape == (1, 1, 128, 128)
