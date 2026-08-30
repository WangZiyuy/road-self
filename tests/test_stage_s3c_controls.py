import numpy as np

from utils.seg_raster.stage_s3 import (
    apply_raster_control, gpu_eligibility_overrides_from_environment,
)
from utils.seg_raster.stage_s3c import canonical_sha256


def test_zero_and_shift_controls_preserve_shape_mask_and_never_wrap() -> None:
    raster = np.zeros((8, 8), dtype=np.uint8)
    raster[1, 1] = 1
    raster[7, 7] = 1
    mask = np.ones_like(raster)
    aligned, aligned_mask = apply_raster_control(raster, mask, "aligned")
    zero, zero_mask = apply_raster_control(raster, mask, "zero")
    shifted, shifted_mask = apply_raster_control(
        raster, mask, "shift_fixed", shift_xy=(2, 2))
    assert aligned.shape == zero.shape == shifted.shape == raster.shape
    assert np.array_equal(aligned_mask, zero_mask)
    assert np.array_equal(aligned_mask, shifted_mask)
    assert np.count_nonzero(zero) == 0
    assert shifted[3, 3] == 1
    assert shifted[1, 1] == 0
    assert shifted[7, 7] == 0


def test_replay_plan_hash_changes_on_any_common_sample_identity_change() -> None:
    plan = [{"crop_origin_xy": [1, 2], "end_index": 4},
            {"crop_origin_xy": [3, 4], "end_index": 1}]
    changed = [dict(row) for row in plan]
    changed[1] = dict(changed[1], end_index=2)
    assert canonical_sha256(plan) != canonical_sha256(changed)


def test_low_memory_gpu_colocation_requires_explicit_bounded_opt_in(
        monkeypatch) -> None:
    monkeypatch.setenv("S3_ALLOW_LOW_MEMORY_EXTERNAL_PROCESSES", "1")
    monkeypatch.setenv("S3_MAX_EXTERNAL_COMPUTE_MEM_MB", "4096")
    monkeypatch.setenv("S3_MAX_UTILIZATION", "20")
    monkeypatch.setenv("S3_MAX_FREE_MEMORY_DROP_MB", "2048")
    monkeypatch.delenv("S3_EXCLUDE_GPUS", raising=False)
    assert gpu_eligibility_overrides_from_environment() == {
        "allow_external_compute": True,
        "max_external_compute_memory_mb": 4096,
        "max_utilization": 20,
        "max_free_memory_drop_mb": 2048,
    }


def test_stage_s3c_launchers_do_not_hardcode_gpu_indices() -> None:
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    for name in (
        "launch_stage_s3c_audit.py",
        "launch_stage_s3c_phase.py", "launch_stage_s3c_graph.py",
        "launch_stage_s3c_anchor.py",
    ):
        source = (root / "tools" / "seg_raster" / name).read_text(
            encoding="utf-8")
        assert "CUDA_VISIBLE_DEVICES" in source
        assert "collect_inventory(" in source
        assert "evaluate_gpu_eligibility(" in source
        assert "excluded_indices()" in source
