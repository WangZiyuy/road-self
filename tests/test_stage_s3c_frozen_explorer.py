import inspect

import torch
from torch import nn

from model.model import RPNet
from utils.seg_raster.stage_s3c import (
    configure_frozen_explorer, trainable_parameters,
)


class _Explorer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stage_1 = nn.Conv2d(3, 4, 1)
        self.conv_fuse = nn.Conv2d(4, 4, 1)
        self.road_seg = nn.Sequential(nn.Conv2d(4, 4, 1), nn.BatchNorm2d(4))
        self.conv_road_final = nn.Conv2d(4, 1, 1)
        self.junc_seg = nn.Sequential(nn.Conv2d(4, 4, 1), nn.BatchNorm2d(4))
        self.conv_junc_final = nn.Conv2d(4, 1, 1)
        self.fuse_module = nn.Conv2d(12, 4, 1)
        self.decoders = nn.ModuleList([nn.Conv2d(4, 4, 1)])
        self.next_step_final = nn.Conv2d(4, 1, 1)
        self.segmentation_raster_fusion = nn.Conv2d(1, 4, 1)


def test_backbone_and_anchor_frozen_segmentation_and_raster_trainable() -> None:
    model = _Explorer()
    contract = configure_frozen_explorer(model, raster_enabled=True)
    by_name = {row["name"]: row for row in contract}
    assert not by_name["stage_1.weight"]["requires_grad"]
    assert not by_name["fuse_module.weight"]["requires_grad"]
    assert not by_name["decoders.0.weight"]["requires_grad"]
    assert by_name["road_seg.0.weight"]["requires_grad"]
    assert by_name["junc_seg.0.weight"]["requires_grad"]
    assert by_name["segmentation_raster_fusion.weight"]["requires_grad"]
    optimizer_ids = {id(value) for value in trainable_parameters(model)}
    assert optimizer_ids == {id(parameter) for parameter in model.parameters()
                             if parameter.requires_grad}


def test_image_only_does_not_train_absent_or_disabled_raster_adapter() -> None:
    model = _Explorer()
    configure_frozen_explorer(model, raster_enabled=False)
    assert not model.segmentation_raster_fusion.weight.requires_grad
    assert model.road_seg[0].weight.requires_grad


def test_segmentation_only_forward_is_explicit_and_default_path_unchanged() -> None:
    signature = inspect.signature(RPNet.forward)
    assert signature.parameters["segmentation_only"].default is False
    source = inspect.getsource(RPNet.forward)
    assert "if segmentation_only" in source
    assert "return {'road': road_final, 'junc': junc_final}" in source
    assert "traj_binary" not in source[source.index("next_points_placeholder"):]
