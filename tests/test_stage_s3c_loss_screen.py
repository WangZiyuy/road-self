import pytest

from utils.seg_raster.stage_s3c import (
    select_baseline_controlled_common_samples, select_phase_a_loss,
)


def _metrics(road: float, junction: float) -> dict:
    return {"road_f1": road, "road_iou": road - 0.1,
            "road_auprc": road + 0.02, "junction_f1": junction,
            "junction_auprc": junction + 0.1}


def test_phase_a_selects_only_from_image_only_loss_candidates() -> None:
    rows = [
        {"run_key": "P0", "input_kind": "image_only",
         "loss_kind": "legacy_exact", "status": "PASS", "finite": True,
         "best_metrics": _metrics(0.7, 0.2)},
        {"run_key": "P1", "input_kind": "image_only",
         "loss_kind": "class_balanced_bce", "status": "PASS", "finite": True,
         "best_metrics": _metrics(0.71, 0.3)},
    ]
    result = select_phase_a_loss(rows)
    assert result["selected_run"] == "P1"
    assert result["raster_results_read_for_selection"] is False
    bad = list(rows) + [dict(rows[0], run_key="R1", input_kind="raster")]
    with pytest.raises(ValueError, match="image-only"):
        select_phase_a_loss(bad)


def test_common_sample_count_is_selected_from_r0_with_early_tie_rule() -> None:
    r0 = {0: _metrics(0.5, 0.1), 2560: _metrics(0.7, 0.2),
          5120: _metrics(0.7005, 0.2)}
    assert select_baseline_controlled_common_samples(r0) == 2560
