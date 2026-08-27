from __future__ import annotations

from easydict import EasyDict
import pytest

from model.model import RPNet
from utils.trajectory_mode import (
    TRAJ_MODE_RASTER_SEG_ONLY,
    load_region_trajectory_inputs_for_mode,
    prepare_trajectory_sequence_batch,
    resolve_trajectory_mode,
    trajectory_fetch_fields,
    validate_trajectory_model_compatibility,
)


def _cfg(*, model: str = "origin", sequence: bool = False) -> EasyDict:
    return EasyDict({
        "TRAJ": {
            "MODE": TRAJ_MODE_RASTER_SEG_ONLY,
            "SEQUENCE": {"ENABLED": sequence},
            "RASTER": {"INPUT_SEMANTICS": "binary_presence"},
        },
        "TRAIN": {"MODEL": model},
    })


def test_raster_mode_resolves_and_fetches_no_sequence_field() -> None:
    cfg = _cfg()
    assert resolve_trajectory_mode(cfg) == TRAJ_MODE_RASTER_SEG_ONLY
    fields = trajectory_fetch_fields(
        TRAJ_MODE_RASTER_SEG_ONLY, include_raster=True)
    assert "traj_image_chw" in fields
    assert "traj_valid_mask_chw" in fields
    assert "valid_trajectories" not in fields


def test_raster_mode_never_calls_region_sequence_loader() -> None:
    def forbidden_loader(*_args):
        raise AssertionError("sequence loader was called")

    assert load_region_trajectory_inputs_for_mode(
        TRAJ_MODE_RASTER_SEG_ONLY, "xian", _cfg(), forbidden_loader
    ) == (None, [], None, None)


def test_raster_mode_never_calls_sequence_padding_or_normalization() -> None:
    def forbidden(*_args):
        raise AssertionError("sequence preprocessing was called")

    assert prepare_trajectory_sequence_batch(
        TRAJ_MODE_RASTER_SEG_ONLY, None, forbidden, forbidden
    ) == (None, None)


def test_raster_mode_rejects_dsf_and_sequence_combinations() -> None:
    with pytest.raises(ValueError, match="DSFNet"):
        validate_trajectory_model_compatibility(_cfg(model="DSFNet"))
    with pytest.raises(ValueError, match="sequence"):
        validate_trajectory_model_compatibility(_cfg(sequence=True))


def test_raster_model_constructs_no_legacy_transformer_or_fuse_module() -> None:
    net = RPNet(
        num_targets=1,
        backbone_pretrained=False,
        enable_raster_segmentation=True,
    )
    assert hasattr(net, "segmentation_raster_fusion")
    assert not hasattr(net, "transformer")
    assert not hasattr(net, "fuse_module_traj")
    assert not hasattr(net, "DSF")
