import pytest
import torch
from torch import nn

from utils.seg_raster.stage_s3c import strict_load_official_checkpoint


class _Baseline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Linear(3, 2)


class _Raster(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Linear(3, 2)
        self.segmentation_raster_fusion = nn.Linear(2, 2)


def test_strict_shared_key_loading_and_explicit_raster_new_keys() -> None:
    source = _Baseline()
    payload = {"state_dict": source.state_dict()}
    baseline = _Baseline()
    exact = strict_load_official_checkpoint(baseline, payload)
    assert exact["shared_key_count"] == 2
    assert exact["missing_new_key_count"] == 0
    raster = _Raster()
    audit = strict_load_official_checkpoint(
        raster, payload, allowed_new_prefixes=("segmentation_raster_fusion.",))
    assert audit["shared_key_count"] == 2
    assert audit["missing_new_key_count"] == 2
    assert all(key.startswith("segmentation_raster_fusion.")
               for key in audit["missing_new_keys"])


def test_data_parallel_prefix_is_normalized_but_trajectory_keys_are_rejected() -> None:
    source = _Baseline()
    prefixed = {"module." + key: value for key, value in source.state_dict().items()}
    audit = strict_load_official_checkpoint(_Baseline(), {"state_dict": prefixed})
    assert audit["data_parallel_prefix_policy"] == "STRIP_MODULE_PREFIX"
    bad = dict(source.state_dict())
    bad["transformer.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="trajectory_related_keys"):
        strict_load_official_checkpoint(_Baseline(), {"state_dict": bad})


def test_shape_mismatch_is_never_silently_loaded() -> None:
    state = dict(_Baseline().state_dict())
    state["shared.weight"] = torch.zeros(9, 9)
    with pytest.raises(ValueError, match="shape_mismatch_keys"):
        strict_load_official_checkpoint(_Baseline(), {"state_dict": state})
