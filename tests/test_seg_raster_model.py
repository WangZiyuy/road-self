from __future__ import annotations

import torch

from model.model import RPNet
from model.seg_raster import (
    SegmentationOnlyRasterFusion,
    TrajectoryRasterEncoder,
)


def test_encoder_and_fusion_shapes_and_zero_init_parity() -> None:
    encoder = TrajectoryRasterEncoder(output_channels=32)
    raster = (torch.rand(2, 1, 64, 64) > 0.8).float()
    encoded = encoder(raster)
    assert encoded.shape == (2, 32, 16, 16)

    fusion = SegmentationOnlyRasterFusion()
    stage_fuse_img = torch.randn(2, 128, 16, 16)
    output = fusion(stage_fuse_img, raster, torch.ones_like(raster))
    assert output.stage_fuse_seg.shape == stage_fuse_img.shape
    assert torch.equal(output.stage_fuse_seg, stage_fuse_img)
    assert sum(p.numel() for p in fusion.parameters()) < 150_000


def test_rpnet_raster_segmentation_and_anchor_shapes() -> None:
    torch.set_num_threads(1)
    net = RPNet(
        num_targets=1,
        backbone_pretrained=False,
        enable_raster_segmentation=True,
    ).eval()
    image = torch.randn(1, 3, 128, 128)
    raster = (torch.rand(1, 1, 128, 128) > 0.9).float()
    walked = torch.zeros(1, 1, 32, 32)
    with torch.no_grad():
        output = net(
            image,
            raster,
            None,
            None,
            None,
            walked,
            NUM_TARGETS=1,
            model="origin",
            trajectory_mode="raster_seg_only",
            traj_valid_mask=torch.ones_like(raster),
        )
    assert output["road"].shape == (1, 1, 32, 32)
    assert output["junc"].shape == (1, 1, 32, 32)
    assert output["anchor"].shape == (1, 1, 128, 128)
    assert output["anchor_lowrs"].shape == (1, 1, 128, 128)
    assert output["feature_maps"]["stage_fuse_img"].shape == (1, 128, 32, 32)
    assert output["feature_maps"]["stage_fuse_seg"].shape == (1, 128, 32, 32)


def test_raster_mode_requires_aligned_present_raster() -> None:
    net = RPNet(
        num_targets=1,
        backbone_pretrained=False,
        enable_raster_segmentation=True,
    ).eval()
    image = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        try:
            net(
                aerial_image=image,
                traj_image=None,
                aerial_traj_image=None,
                neighborhood_trajectory_norm=None,
                valid_mask=None,
                walked_path=None,
                test=True,
                model="origin",
                trajectory_mode="raster_seg_only",
            )
        except ValueError as exc:
            assert "requires a trajectory raster" in str(exc)
        else:
            raise AssertionError("missing raster did not fail fast")
