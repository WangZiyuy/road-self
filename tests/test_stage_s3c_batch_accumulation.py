import copy

import torch
from torch import nn

from utils.seg_raster.stage_s3c import (
    SampleBudgetCounter, checkpoint_name, segmentation_losses,
)


def _loss(model: nn.Module, x: torch.Tensor, road: torch.Tensor,
          junction: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    output = {"road": logits[:, :1], "junc": logits[:, 1:]}
    return segmentation_losses(output, road, junction)["total"]


def test_microbatch10_accum2_sum_gradient_equals_batch20_reference() -> None:
    torch.manual_seed(7)
    reference = nn.Conv2d(3, 2, 1)
    accumulated = copy.deepcopy(reference)
    x = torch.randn(20, 3, 4, 4)
    road = torch.randint(0, 2, (20, 1, 4, 4)).float()
    junction = torch.randint(0, 2, (20, 1, 4, 4)).float()
    _loss(reference, x, road, junction).backward()
    _loss(accumulated, x[:10], road[:10], junction[:10]).backward()
    _loss(accumulated, x[10:], road[10:], junction[10:]).backward()
    assert torch.allclose(reference.weight.grad, accumulated.weight.grad,
                          rtol=1e-5, atol=1e-6)
    assert torch.allclose(reference.bias.grad, accumulated.bias.grad,
                          rtol=1e-5, atol=1e-6)


def test_samples_seen_and_versioned_checkpoint_grid() -> None:
    counter = SampleBudgetCounter()
    assert counter.record_micro_batch(10) is False
    assert counter.record_micro_batch(10) is True
    assert counter.samples_seen == 20
    assert counter.micro_batches == 2
    assert counter.optimizer_updates == 1
    assert checkpoint_name(2560) == "samples_002560.pth.tar"
