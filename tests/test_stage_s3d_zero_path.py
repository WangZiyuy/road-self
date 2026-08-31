from __future__ import annotations

from model.seg_raster import StrictZeroPreservingRoadAdapter
from utils.seg_raster.stage_s3d import classify_current_zero_path


def test_adapter_has_no_bias_or_normalization() -> None:
    module = StrictZeroPreservingRoadAdapter()
    names = [child.__class__.__name__ for child in module.modules()]
    assert not any("Norm" in name for name in names)
    for child in module.modules():
        if hasattr(child, "bias"):
            assert child.bias is None


def test_current_s3c_zero_path_classifies_multiple_runtime_causes() -> None:
    result = classify_current_zero_path(
        image_enters_trainable_fusion=True,
        valid_mask_enters_trainable_fusion=True,
        normalization_affine=True,
        bias_present=True,
        runtime_residual_nonzero=True,
    )
    assert result == "MULTIPLE_CAUSES"
