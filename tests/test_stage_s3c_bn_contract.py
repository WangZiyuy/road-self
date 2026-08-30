import torch
from torch import nn

from utils.seg_raster.stage_s3c import (
    configure_frozen_explorer, original_batch_norm_checksum,
    set_frozen_explorer_train_mode,
)


class _BnModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stage_1 = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4))
        self.road_seg = nn.Sequential(nn.Conv2d(4, 4, 1), nn.BatchNorm2d(4))
        self.conv_road_final = nn.Conv2d(4, 1, 1)
        self.junc_seg = nn.Sequential(nn.Conv2d(4, 4, 1), nn.BatchNorm2d(4))
        self.conv_junc_final = nn.Conv2d(4, 1, 1)
        self.segmentation_raster_fusion = nn.Sequential(
            nn.Conv2d(1, 4, 1), nn.GroupNorm(2, 4))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feature = self.stage_1(image)
        return self.conv_road_final(self.road_seg(feature)).sum()


def test_original_bn_stats_stay_frozen_but_head_affine_remains_trainable() -> None:
    model = _BnModel()
    configure_frozen_explorer(model, raster_enabled=True)
    before = original_batch_norm_checksum(model)
    set_frozen_explorer_train_mode(model)
    assert model.stage_1[1].training is False
    assert model.road_seg[1].training is False
    assert model.road_seg[1].weight.requires_grad
    assert not model.stage_1[1].weight.requires_grad
    model(torch.randn(10, 3, 8, 8)).backward()
    after = original_batch_norm_checksum(model)
    assert after == before
    assert model.road_seg[1].weight.grad is not None


def test_adapter_groupnorm_has_no_running_stats_by_design() -> None:
    model = _BnModel()
    configure_frozen_explorer(model, raster_enabled=True)
    set_frozen_explorer_train_mode(model)
    adapter_norm = model.segmentation_raster_fusion[1]
    assert isinstance(adapter_norm, nn.GroupNorm)
    assert adapter_norm.training is True
    assert not hasattr(adapter_norm, "running_mean")
