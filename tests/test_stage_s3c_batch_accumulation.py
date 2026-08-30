import copy

import torch
from torch import nn
import yaml

from utils.OSMDataset import replicate_subtiles_for_independent_paths
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


def test_xian_path_replication_provides_independent_local_batch_slots() -> None:
    source = [{"region": "xian", "slot": 0}, {"region": "xian", "slot": 1}]
    replicated = replicate_subtiles_for_independent_paths(source, 5)
    assert len(replicated) == 10
    assert all(row is not source[index % 2]
               for index, row in enumerate(replicated))
    replicated[0]["slot"] = 99
    assert replicated[2]["slot"] == 0


def test_stage_s3c_uses_five_independent_path_replicas() -> None:
    config = yaml.safe_load(open("configs/stage_s3c_common.yml", encoding="utf-8"))
    assert config["TRAIN"]["SPATIAL_PATH_REPLICAS"] == 5
