from __future__ import annotations

import torch
from torch import nn

from model.model import RPNet
from model.seg_raster import (
    StrictZeroPreservingRoadAdapter, validate_zero_preserving_contract)


def test_zero_raster_is_bitwise_identity_before_and_after_optimizer_step() -> None:
    torch.manual_seed(7)
    module = StrictZeroPreservingRoadAdapter(8, 4)
    validate_zero_preserving_contract(module)
    image = torch.randn(2, 8, 8, 8)
    zero = torch.zeros(2, 1, 32, 32)
    valid = torch.ones_like(zero)
    first = module(image, zero, valid)
    assert torch.equal(first.stage_fuse_road, image)
    assert torch.count_nonzero(first.residual).item() == 0
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-2,
                                 weight_decay=2e-4)
    first.stage_fuse_road.square().sum().backward()
    optimizer.step()
    second = module(image, zero, valid)
    assert torch.equal(second.stage_fuse_road, image)
    assert torch.equal(second.stage_fuse_junction, image)
    assert torch.count_nonzero(second.residual).item() == 0


def test_aligned_raster_has_nonzero_finite_residual_and_gradient() -> None:
    torch.manual_seed(9)
    module = StrictZeroPreservingRoadAdapter(8, 4)
    image = torch.randn(2, 8, 8, 8)
    raster = (torch.rand(2, 1, 32, 32) > 0.7).float()
    output = module(image, raster, torch.ones_like(raster))
    assert torch.count_nonzero(output.residual).item() > 0
    output.stage_fuse_road.square().mean().backward()
    gradients = [parameter.grad for parameter in module.parameters()]
    assert all(value is not None and torch.isfinite(value).all()
               for value in gradients)
    assert any(torch.count_nonzero(value).item() > 0 for value in gradients)


def _forward(model: RPNet, image: torch.Tensor, raster: torch.Tensor) -> dict:
    return model(
        image, raster, None, None, None,
        torch.zeros(image.shape[0], 1, image.shape[2] // 4,
                    image.shape[3] // 4),
        NUM_TARGETS=1, model="origin",
        trajectory_mode="raster_road_zero_preserving",
        traj_valid_mask=torch.ones_like(raster),
    )


def test_junction_and_image_backbone_are_raster_invariant() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(13)
    model = RPNet(
        num_targets=1, backbone_pretrained=False,
        enable_zero_preserving_road_adapter=True).eval()
    image = torch.randn(1, 3, 64, 64)
    zero = torch.zeros(1, 1, 64, 64)
    aligned = (torch.rand_like(zero) > 0.7).float()
    with torch.no_grad():
        left, right = _forward(model, image, zero), _forward(model, image, aligned)
    assert torch.equal(left["junc"], right["junc"])
    assert torch.equal(left["feature_maps"]["stage_fuse_img"],
                       right["feature_maps"]["stage_fuse_img"])
    assert torch.equal(left["feature_maps"]["stage_fuse_junction"],
                       right["feature_maps"]["stage_fuse_junction"])


def test_anchor_has_no_raw_raster_channel() -> None:
    torch.set_num_threads(1)
    model = RPNet(
        num_targets=1, backbone_pretrained=False,
        enable_zero_preserving_road_adapter=True).eval()
    shapes = []
    handle = model.fuse_module.register_forward_pre_hook(
        lambda _module, inputs: shapes.append(tuple(inputs[0].shape)))
    image = torch.randn(1, 3, 64, 64)
    raster = torch.ones(1, 1, 64, 64)
    with torch.no_grad():
        _forward(model, image, raster)
    handle.remove()
    assert shapes == [(1, 257, 16, 16)]
