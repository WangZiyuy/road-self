from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from utils.seg_raster.stage_s3 import apply_raster_control, audit_batch_parity
from utils.seg_raster.stage_s3b import (
    CHECKPOINT_STEPS, GRAPH_CAPS, LOSS_BALANCED, LOSS_BALANCED_DICE,
    LOSS_LEGACY, GraphResourceSnapshot, assert_json_finite,
    choose_shared_threshold, gradient_matching_alpha, graph_resource_status,
    frozen_plan_batch_identities, junction_loss, load_common_step_checkpoint,
    positive_weight_from_counts,
    repair_composite, save_versioned_model_checkpoint, select_junction_loss,
    select_learning_rate, simulate_early_stop, soft_dice_loss,
    validate_commit_paths)


def metrics(value: float, junction: float | None = None) -> dict:
    return {"road_f1": value, "road_iou": value, "road_auprc": value,
            "junction_f1": value, "junction_auprc": junction if junction is not None else value,
            "junction_shared_f1": value}


def sample(index: int) -> dict:
    return {"region": "xian", "crop_origin_xy": [index, 0],
            "extension_vertex_xy": [index + 1, 1], "is_key_point": False,
            "target_count": 1, "end_index": 1,
            "augmentation": {"rot90_k": 0, "flip_x": False, "flip_y": False}}


def test_legacy_loss_exactly_matches_old_sum_bce() -> None:
    logits = torch.tensor([[[[0.2, -1.0]]]])
    target = torch.tensor([[[[1.0, 0.0]]]])
    assert torch.equal(junction_loss(logits, target),
                       F.binary_cross_entropy_with_logits(logits, target, reduction="sum"))


def test_balanced_bce_matches_reference() -> None:
    logits = torch.tensor([0.0, 1.0, -1.0])
    target = torch.tensor([1.0, 0.0, 1.0])
    expected = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=torch.tensor(3.0), reduction="sum")
    actual = junction_loss(logits, target, kind=LOSS_BALANCED, pos_weight=3.0)
    assert torch.allclose(actual, expected)


def test_soft_dice_known_answer() -> None:
    logits = torch.zeros((1, 1, 1, 2))
    target = torch.tensor([[[[1.0, 0.0]]]])
    assert torch.allclose(soft_dice_loss(logits, target, smooth=1.0),
                          torch.tensor(1.0 - 2.0 / 3.0))


def test_no_positive_dice_is_finite() -> None:
    value = junction_loss(torch.zeros((2, 1, 4, 4)), torch.zeros((2, 1, 4, 4)),
                          kind=LOSS_BALANCED_DICE, pos_weight=32)
    assert torch.isfinite(value)


def test_pos_weight_is_reproducible_and_capped() -> None:
    assert positive_weight_from_counts(10, 1000) == positive_weight_from_counts(10, 1000)
    assert positive_weight_from_counts(10, 1000)["capped_pos_weight"] == 32.0


def test_gradient_matching_alpha_is_reproducible() -> None:
    assert gradient_matching_alpha([2, 4], [4, 8]) == 0.5


def test_lr_grid_and_no_scheduler_contract() -> None:
    assert CHECKPOINT_STEPS == (0, 2560, 5120, 7680, 10240, 12800, 15360, 17920, 20480)
    config = Path("configs/stage_s3b_common.yml").read_text(encoding="utf-8")
    assert "SCHEDULER: none" in config and "WARMUP_STEPS: 0" in config


def test_lr_selection_uses_only_image_candidates() -> None:
    rows = [{"run_key": "A0", "input_kind": "image_only", "status": "PASS",
             "finite": True, "retention": 0.8, "best_repair_composite": 0.5,
             "lr_multiplier": 1.0, "base_lr": 1e-4},
            {"run_key": "A2", "input_kind": "image_only", "status": "PASS",
             "finite": True, "retention": 0.9, "best_repair_composite": 0.5005,
             "lr_multiplier": 0.3, "base_lr": 3e-5}]
    assert select_learning_rate(rows)["selected_run"] == "A2"
    with pytest.raises(ValueError):
        select_learning_rate([{**rows[0], "input_kind": "raster"}])


def test_loss_selection_uses_only_image_candidates() -> None:
    legacy = {"run_key": "B0", "input_kind": "image_only", "loss_kind": LOSS_LEGACY,
              "status": "PASS", "finite": True, "gradient_explosion": False,
              "retention": 0.8, "best_metrics": metrics(0.5, 0.1)}
    balanced = {**legacy, "run_key": "B2", "loss_kind": LOSS_BALANCED,
                "best_metrics": metrics(0.5, 0.2)}
    assert select_junction_loss([legacy, balanced])["selected_loss_kind"] == LOSS_BALANCED
    with pytest.raises(ValueError):
        select_junction_loss([legacy, {**balanced, "input_kind": "raster"}])


def test_sample_plan_replay_parity() -> None:
    plan = [sample(index) for index in range(100)]
    assert audit_batch_parity({"A0": plan, "A1": list(plan)}, 100)["status"] == "PASS"


def test_sample_gate_recomputes_explicit_rows_not_stale_aggregate() -> None:
    order = [{"samples": [sample(index)]} for index in range(100)]
    identities = frozen_plan_batch_identities(order, count=100)
    assert len(identities) == 100
    assert identities[0] == frozen_plan_batch_identities(order, count=1)[0]


def test_versioned_checkpoint_refuses_overwrite() -> None:
    path = Path("artifacts/_stage_s3b_test_checkpoint_0.pth.tar")
    path.unlink(missing_ok=True)
    try:
        save_versioned_model_checkpoint(path, {"x": torch.ones(1)}, step=0,
                                        code_sha="a", config_sha="b", metric_code_sha="c")
        with pytest.raises(FileExistsError):
            save_versioned_model_checkpoint(path, {"x": torch.ones(1)}, step=0,
                                            code_sha="a", config_sha="b", metric_code_sha="c")
    finally:
        path.unlink(missing_ok=True)


def test_common_step_checkpoint_loads_strict_metadata() -> None:
    path = Path("artifacts/_stage_s3b_test_checkpoint_2560.pth.tar")
    path.unlink(missing_ok=True)
    try:
        save_versioned_model_checkpoint(path, {"x": torch.ones(1)}, step=2560,
                                        code_sha="sha", config_sha="b", metric_code_sha="c")
        assert "x" in load_common_step_checkpoint(path, expected_step=2560,
                                                    expected_code_sha="sha")
    finally:
        path.unlink(missing_ok=True)


def test_last_train_batch_metrics_cannot_enter_selection() -> None:
    validation = {0: metrics(0.2), 2560: metrics(0.3)}
    result = simulate_early_stop(validation)
    assert result["selection_source"] == "image_only_validation_control_only"


def test_repair_composite_uses_junction_auprc() -> None:
    row = metrics(0.3, 0.9)
    row["junction_f1"] = 0.0
    assert repair_composite(row) == pytest.approx(0.5)


def test_shared_threshold_is_one_frozen_value() -> None:
    result = choose_shared_threshold(np.array([-4, 4]), np.array([0, 1]))
    assert result["frozen_for_controls"] is True
    assert result["selection_source"] == "selected_image_only_calibration_subset"


def test_controls_preserve_shape_and_mask() -> None:
    raster = np.ones((1, 1, 5, 6), dtype=np.uint8)
    mask = np.ones_like(raster)
    for control in ("aligned", "zero", "shift_fixed"):
        value, valid = apply_raster_control(raster, mask, control, shift_xy=(1, 1))
        assert value.shape == raster.shape and np.array_equal(valid, mask)


def test_shift_control_zero_fills_without_wrap() -> None:
    raster = np.zeros((1, 1, 3, 3), dtype=np.uint8)
    raster[..., 0, 0] = 1
    shifted, _ = apply_raster_control(raster, np.ones_like(raster),
                                      "shift_fixed", shift_xy=(-1, 0))
    assert shifted.sum() == 0


def test_graph_caps_are_identical_for_every_control() -> None:
    snapshots = [GraphResourceSnapshot(3000, 1, 1, 1) for _ in range(4)]
    assert all(graph_resource_status(value)["caps"] == GRAPH_CAPS for value in snapshots)


def test_resource_cap_never_reports_pass() -> None:
    result = graph_resource_status(GraphResourceSnapshot(3000, 1, 1, 1))
    assert result["status"] == "RESOURCE_CAP_REACHED"
    assert result["natural_termination"] is False


def test_formal_config_forbids_sequence_transformer_and_dsf() -> None:
    text = Path("configs/stage_s3b_common.yml").read_text(encoding="utf-8")
    assert "ENABLED: false" in text
    worker = Path("tools/seg_raster/train_stage_s3b.py").read_text(encoding="utf-8")
    assert "enable_trajectory_modules=False" in worker
    assert 'model="origin"' in worker


def test_raw_raster_is_not_direct_anchor_input() -> None:
    source = Path("model/model.py").read_text(encoding="utf-8")
    assert "stage_fuse_seg" in source
    assert "segmentation_raster_fusion" in source
    assert "anchor_grad_to_seg" in source


def test_json_finite_rejects_nan() -> None:
    assert_json_finite({"value": 1.0})
    with pytest.raises(ValueError):
        assert_json_finite({"value": float("nan")})


def test_commit_manifest_forbids_large_run_data() -> None:
    validate_commit_paths(["artifacts/stage_s3b_conclusion.json",
                           "docs/audits/stage_s3b_final_report.md"])
    for path in ("checkpoints/x.pth", "dataset/x", "cache/x", "raster/x.png"):
        with pytest.raises(ValueError):
            validate_commit_paths([path])


def test_graph_caps_are_frozen_to_requested_values() -> None:
    assert GRAPH_CAPS == {"max_iterations": 3000, "max_vertices": 5000,
                          "max_directed_edges": 10000,
                          "max_wall_time_seconds": 900}
