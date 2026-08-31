from __future__ import annotations

import torch
from torch import nn

from model.seg_raster import StrictZeroPreservingRoadAdapter
from utils.seg_raster.stage_s3e import weighted_road_loss


def test_zero_init_projection_is_exactly_zero() -> None:
    module = StrictZeroPreservingRoadAdapter(
        image_channels=8, hidden_channels=4, projection_init="zero")
    assert torch.count_nonzero(module.projection.weight).item() == 0


def test_zero_init_aligned_sample0_is_image_identity() -> None:
    torch.manual_seed(9)
    module = StrictZeroPreservingRoadAdapter(
        image_channels=8, hidden_channels=4, projection_init="zero")
    image = torch.randn(2, 8, 8, 8)
    raster = (torch.rand(2, 1, 32, 32) > .7).float()
    output = module(image, raster, torch.ones_like(raster))
    assert torch.equal(output.stage_fuse_road, image)
    assert torch.count_nonzero(output.residual).item() == 0


def test_zero_init_sample0_logits_equal_null_logits() -> None:
    torch.manual_seed(11)
    module = StrictZeroPreservingRoadAdapter(
        image_channels=8, hidden_channels=4, projection_init="zero")
    head = nn.Conv2d(8, 1, 1)
    image = torch.randn(2, 8, 8, 8)
    raster = (torch.rand(2, 1, 32, 32) > .4).float()
    aligned = module(image, raster, torch.ones_like(raster))
    assert torch.equal(head(aligned.stage_fuse_road), head(image))


def test_zero_init_first_step_head_gradient_matches_null() -> None:
    torch.manual_seed(13)
    default = StrictZeroPreservingRoadAdapter(8, 4, projection_init="default")
    zero = StrictZeroPreservingRoadAdapter(8, 4, projection_init="zero")
    zero.encoder.load_state_dict(default.encoder.state_dict())
    head_a, head_b = nn.Conv2d(8, 1, 1), nn.Conv2d(8, 1, 1)
    head_b.load_state_dict(head_a.state_dict())
    image = torch.randn(2, 8, 8, 8)
    raster = (torch.rand(2, 1, 32, 32) > .5).float()
    target = torch.rand(2, 1, 8, 8)
    null = default(image, raster, torch.ones_like(raster), bypass=True)
    aligned = zero(image, raster, torch.ones_like(raster))
    weighted_road_loss(head_a(null.stage_fuse_road), target).backward()
    weighted_road_loss(head_b(aligned.stage_fuse_road), target).backward()
    assert torch.equal(head_a.weight.grad, head_b.weight.grad)
    assert torch.equal(head_a.bias.grad, head_b.bias.grad)
    assert torch.count_nonzero(zero.projection.weight.grad).item() > 0
    assert all(parameter.grad is not None and torch.count_nonzero(parameter.grad) == 0
               for parameter in zero.encoder.parameters())


def test_encoder_weight_decay_update_is_distinct_from_loss_gradient() -> None:
    torch.manual_seed(17)
    module = StrictZeroPreservingRoadAdapter(8, 4, projection_init="zero")
    before = [parameter.detach().clone() for parameter in module.encoder.parameters()]
    image = torch.randn(2, 8, 8, 8)
    raster = torch.ones(2, 1, 32, 32)
    module(image, raster, torch.ones_like(raster)).stage_fuse_road.square().sum().backward()
    assert all(torch.count_nonzero(parameter.grad).item() == 0
               for parameter in module.encoder.parameters())
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-2, weight_decay=2e-4)
    optimizer.step()
    assert any(not torch.equal(old, new) for old, new in zip(
        before, module.encoder.parameters()))


def test_default_init_remains_backward_compatible_and_nonzero() -> None:
    module = StrictZeroPreservingRoadAdapter(8, 4)
    assert module.projection_init == "default"
    assert torch.count_nonzero(module.projection.weight).item() > 0


def test_default_init_adapter_checkpoint_round_trip_is_exact() -> None:
    torch.manual_seed(31)
    source = StrictZeroPreservingRoadAdapter(8, 4)
    target = StrictZeroPreservingRoadAdapter(8, 4, projection_init="default")
    target.load_state_dict(source.state_dict(), strict=True)
    image = torch.randn(1, 8, 8, 8)
    raster = (torch.rand(1, 1, 32, 32) > .5).float()
    valid = torch.ones_like(raster)
    left, right = source(image, raster, valid), target(image, raster, valid)
    assert torch.equal(left.residual, right.residual)
    assert torch.equal(left.stage_fuse_road, right.stage_fuse_road)
