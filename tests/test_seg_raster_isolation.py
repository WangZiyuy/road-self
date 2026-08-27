from __future__ import annotations

import torch
from torch import nn

from model.model import RPNet


class _FixedSegFeatures(nn.Module):
    def forward(self, stage_fuse_seg: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            stage_fuse_seg.shape[0],
            64,
            stage_fuse_seg.shape[2],
            stage_fuse_seg.shape[3],
            device=stage_fuse_seg.device,
            dtype=stage_fuse_seg.dtype,
        )


def _forward(net: RPNet, image: torch.Tensor, raster: torch.Tensor) -> dict:
    return net(
        aerial_image=image,
        traj_image=raster,
        aerial_traj_image=None,
        neighborhood_trajectory_norm=None,
        valid_mask=None,
        walked_path=torch.zeros(image.shape[0], 1, image.shape[2] // 4, image.shape[3] // 4),
        NUM_TARGETS=1,
        model="origin",
        use_traj=False,
        trajectory_mode="raster_seg_only",
        traj_valid_mask=torch.ones_like(raster),
    )


def test_anchor_receives_no_direct_raster_channels_and_fixed_seg_parity() -> None:
    torch.set_num_threads(1)
    net = RPNet(
        num_targets=1,
        backbone_pretrained=False,
        enable_raster_segmentation=True,
    ).eval()
    nn.init.normal_(net.segmentation_raster_fusion.delta_projection.weight, std=0.01)
    net.road_seg = _FixedSegFeatures()
    net.junc_seg = _FixedSegFeatures()
    seen_shapes = []
    handle = net.fuse_module.register_forward_pre_hook(
        lambda _module, inputs: seen_shapes.append(tuple(inputs[0].shape)))
    image = torch.randn(1, 3, 128, 128)
    first = torch.zeros(1, 1, 128, 128)
    second = torch.ones_like(first)
    with torch.no_grad():
        output_first = _forward(net, image, first)
        output_second = _forward(net, image, second)
    handle.remove()
    assert seen_shapes == [(1, 257, 32, 32), (1, 257, 32, 32)]
    assert torch.equal(output_first["anchor"], output_second["anchor"])
    assert torch.equal(
        output_first["feature_maps"]["stage_fuse_img"],
        output_second["feature_maps"]["stage_fuse_img"],
    )


def test_anchor_grad_to_seg_switch_blocks_or_allows_raster_gradient() -> None:
    torch.set_num_threads(1)
    net = RPNet(
        num_targets=1,
        backbone_pretrained=False,
        enable_raster_segmentation=True,
        anchor_grad_to_seg=False,
    ).train()
    image = torch.randn(2, 3, 128, 128)
    raster = (torch.rand(2, 1, 128, 128) > 0.8).float()

    detached_output = _forward(net, image, raster)
    detached_output["anchor"].square().mean().backward()
    assert all(
        parameter.grad is None
        for parameter in net.segmentation_raster_fusion.parameters())

    net.zero_grad(set_to_none=True)
    net.anchor_grad_to_seg = True
    joint_output = _forward(net, image, raster)
    joint_output["anchor"].square().mean().backward()
    gradients = [
        parameter.grad
        for parameter in net.segmentation_raster_fusion.parameters()
    ]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients)


def test_none_mode_exact_origin_parity_and_no_new_registration() -> None:
    torch.manual_seed(17)
    baseline = RPNet(
        num_targets=1, backbone_pretrained=False).eval()
    torch.manual_seed(17)
    raster_capable = RPNet(
        num_targets=1,
        backbone_pretrained=False,
        enable_raster_segmentation=True,
    ).eval()
    raster_state = raster_capable.state_dict()
    for key, value in baseline.state_dict().items():
        assert torch.equal(value, raster_state[key]), key
    assert not hasattr(baseline, "segmentation_raster_fusion")

    image = torch.randn(1, 3, 128, 128)
    walked = torch.zeros(1, 1, 32, 32)
    with torch.no_grad():
        first = baseline(
            image, None, None, None, None, walked,
            NUM_TARGETS=1, model="origin", trajectory_mode="none")
        second = raster_capable(
            image, None, None, None, None, walked,
            NUM_TARGETS=1, model="origin", trajectory_mode="none")
    for key in ("road", "junc", "anchor", "anchor_lowrs"):
        assert torch.equal(first[key], second[key]), key
