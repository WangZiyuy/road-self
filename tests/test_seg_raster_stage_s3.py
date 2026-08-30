from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from model.model import RPNet
from tools.seg_raster.launch_stage_s3_parallel import logical_run_path
from utils.seg_raster.stage_s3 import (
    EXPERIMENT_MATRIX,
    FifoGpuScheduler,
    SpatialExtent,
    anchor_metrics,
    apply_raster_control,
    apply_synchronized_augmentation,
    audit_batch_parity,
    build_spatial_split,
    evaluate_gpu_eligibility,
    load_stage_s3_config,
    parse_compute_apps_csv,
    parse_gpu_inventory_csv,
    required_free_memory_mb,
    sample_identity,
    segmentation_causal_screen,
    stitch_tiles,
    strict_shared_state_audit,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_six_experiment_configs_are_complete_and_fair() -> None:
    assert [spec.key for spec in EXPERIMENT_MATRIX] == [
        "C0", "C1", "C2", "C3", "J0", "J1"]
    configs = {}
    for spec in EXPERIMENT_MATRIX:
        paths = sorted((REPO_ROOT / "configs").glob(
            "stage_s3_{}_*yml".format(spec.key)))
        assert len(paths) == 1
        configs[spec.key] = load_stage_s3_config(paths[0])
    shared = (
        "SEED", "OPTIMIZER_STEPS", "WINDOW_SIZE", "NUM_TARGETS",
        "BATCH_SIZE")
    for key in shared:
        assert len({cfg["TRAIN"].get(key, cfg["S3"].get(key))
                    for cfg in configs.values()}) == 1
    assert configs["C0"]["TRAJ"]["MODE"] == "none"
    assert configs["C1"]["TRAJ"]["RASTER"]["CONTROL"] == "aligned"
    assert configs["C2"]["TRAJ"]["RASTER"]["CONTROL"] == "zero"
    assert configs["C3"]["TRAJ"]["RASTER"]["CONTROL"] == "shift_fixed"
    assert configs["C0"]["TRAJ"]["RASTER"]["ANCHOR_GRAD_TO_SEG"] is False
    assert configs["J0"]["TRAJ"]["RASTER"]["ANCHOR_GRAD_TO_SEG"] is True


def test_zero_and_shift_controls_preserve_shape_and_valid_mask_without_wrap() -> None:
    raster = np.zeros((8, 9), dtype=np.uint8)
    raster[1, 1] = 255
    raster[6, 7] = 128
    mask = np.ones_like(raster, dtype=np.uint8)
    mask[:, -1] = 0
    aligned, aligned_mask = apply_raster_control(raster, mask, "aligned")
    zero, zero_mask = apply_raster_control(raster, mask, "zero")
    shifted, shifted_mask = apply_raster_control(
        raster, mask, "shift_fixed", shift_xy=(2, 2))
    assert aligned.shape == zero.shape == shifted.shape == raster.shape
    assert np.array_equal(aligned_mask, zero_mask)
    assert np.array_equal(aligned_mask, shifted_mask)
    assert not zero.any()
    assert shifted[3, 3] == 1
    assert shifted[0, 0] == 0
    assert shifted[6, 7] == 0  # zero fill, not circular wrap


def test_synchronized_augmentation_handles_image_and_quarter_scale_labels() -> None:
    image = np.arange(3 * 8 * 8).reshape(3, 8, 8)
    raster = np.arange(8 * 8).reshape(1, 8, 8)
    road = np.arange(2 * 2).reshape(1, 2, 2)
    result = apply_synchronized_augmentation(
        {"image": image, "raster": raster, "road": road},
        rot90_k=1, flip_x=True, flip_y=True)
    assert np.array_equal(
        result["image"], np.flip(np.flip(np.rot90(image, axes=(-2, -1)), -1), -2))
    assert np.array_equal(
        result["road"], np.flip(np.flip(np.rot90(road, axes=(-2, -1)), -1), -2))


def _sample(index: int) -> dict:
    return {
        "region": "xian", "crop_origin_xy": [index, 0],
        "extension_vertex_xy": [index + 128, 128],
        "is_key_point": bool(index % 2), "target_count": 4,
        "end_index": 4,
        "augmentation": {"rot90_k": 0, "flip_x": False, "flip_y": False},
    }


def test_sample_plan_is_replayable_and_first_100_identity_must_match() -> None:
    rows = [_sample(index) for index in range(100)]
    plans = {key: list(rows) for key in ("C0", "C1", "C2", "C3", "J0", "J1")}
    assert audit_batch_parity(plans, 100)["status"] == "PASS"
    plans["C3"] = list(rows)
    plans["C3"][17] = _sample(999)
    result = audit_batch_parity(plans, 100)
    assert result["status"] == "FAIL"
    assert result["mismatch_indices"]["C3"] == [17]
    assert len(sample_identity(rows[0])) == 64


def test_spatial_split_has_buffer_and_crops_cannot_cross() -> None:
    split = build_spatial_split(
        canvas_wh=(4096, 4096), crop_size=256, boundary_buffer=256)
    train = SpatialExtent(**split["train_extent"])
    validation = SpatialExtent(**split["validation_extent"])
    assert not train.intersects(validation)
    assert validation.x0 - train.x1 == 256
    assert train.contains_crop(0, 0, 256)
    assert not train.contains_crop(train.x1 - 128, 0, 256)


def test_gpu_inventory_and_compute_process_are_hard_eligibility_gates() -> None:
    gpu_text = "0, GPU-a, RTX 4050, 566.24, 6141, 500, 5641, 0, 50"
    gpu = parse_gpu_inventory_csv(gpu_text)[0]
    samples = [{"gpus": [dict(gpu)]} for _ in range(3)]
    apps = parse_compute_apps_csv("123, GPU-a, 512, external.exe")
    result = evaluate_gpu_eligibility(
        samples, apps, required_free_mb=3000)
    assert result[0]["eligible"] is False
    assert "external_compute_process" in result[0]["reasons"]
    assert result[0]["external_compute_processes"][0]["pid"] == 123
    assert required_free_memory_mb(4000, 4500) == 6048


def test_explicit_low_memory_external_process_allowance_is_bounded() -> None:
    gpu_text = "0, GPU-a, RTX 4090, 566.24, 24564, 700, 23864, 12, 50"
    gpu = parse_gpu_inventory_csv(gpu_text)[0]
    samples = [{"gpus": [dict(gpu)]} for _ in range(3)]
    apps = parse_compute_apps_csv("123, GPU-a, 700, external.exe")
    allowed = evaluate_gpu_eligibility(
        samples, apps, required_free_mb=12000, max_utilization=20,
        allow_external_compute=True, max_external_compute_memory_mb=4096)
    assert allowed[0]["eligible"] is True
    assert allowed[0]["external_compute_allowed"] is True
    assert allowed[0]["external_compute_memory_mb"] == 700
    rejected = evaluate_gpu_eligibility(
        samples, apps, required_free_mb=12000, max_utilization=20,
        allow_external_compute=True, max_external_compute_memory_mb=512)
    assert rejected[0]["eligible"] is False
    assert "external_compute_process" in rejected[0]["reasons"]


def test_gpu_scheduler_queues_and_never_double_assigns() -> None:
    scheduler = FifoGpuScheduler([3])
    assert scheduler.allocate("C0") == 3
    assert scheduler.allocate("C1") is None
    scheduler.release(3, "C0")
    assert scheduler.allocate("C1") == 3
    with pytest.raises(ValueError, match="already scheduled"):
        scheduler.allocate("C1")


def test_launcher_logical_run_path_preserves_redacted_placeholder() -> None:
    assert logical_run_path("C0_image_detach", "stdout.log") == (
        "${RUN_ROOT}/C0_image_detach/stdout.log")


def test_strict_shared_checkpoint_loader_allows_only_raster_prefix() -> None:
    image = {"shared.weight": torch.zeros(2, 3)}
    raster = {
        "shared.weight": torch.ones(2, 3),
        "segmentation_raster_fusion.delta_projection.weight": torch.zeros(1),
    }
    audit = strict_shared_state_audit(image, raster)
    assert audit["status"] == "PASS"
    assert audit["raster_only_key_count"] == 1
    raster["shared.weight"] = torch.ones(3, 2)
    assert strict_shared_state_audit(image, raster)["status"] == "FAIL"


def test_full_canvas_stitching_has_exact_shape_and_overlap_average() -> None:
    tiles = [np.ones((1, 2, 2), dtype=np.float32),
             np.full((1, 2, 2), 3, dtype=np.float32)]
    stitched = stitch_tiles(tiles, [(0, 0), (1, 0)], (2, 3))
    assert stitched.shape == (1, 2, 3)
    assert np.all(stitched[:, :, 1] == 2)


def test_anchor_metric_contract_and_causal_screen_rules() -> None:
    target = np.zeros((1, 4, 8, 8), dtype=np.float32)
    target[0, 0, 3, 3] = 1
    logits = np.full_like(target, -10)
    logits[0, 0, 3, 3] = 10
    metrics = anchor_metrics(logits, target, [1], threshold=0.3)
    assert metrics["top_k_recall"] == 1.0
    assert metrics["missed_branch_count"] == 0
    base = {
        "road_f1": .50, "road_iou": .40, "junction_f1": .30,
        "road_precision": .50, "road_recall": .50,
        "junction_precision": .30, "junction_recall": .30}
    values = {
        "C0": base,
        "C1": {key: value + .02 for key, value in base.items()},
        "C2": {key: value + .005 for key, value in base.items()},
        "C3": dict(base),
    }
    assert segmentation_causal_screen(values)["status"] == "PROMISING"


def test_image_only_detach_blocks_anchor_gradient_to_segmentation_heads() -> None:
    torch.set_num_threads(1)
    model = RPNet(
        num_targets=1, backbone_pretrained=False,
        enable_raster_segmentation=False, anchor_grad_to_seg=False).train()
    output = model(
        torch.randn(2, 3, 128, 128), None, None, None, None,
        torch.zeros(2, 1, 32, 32), trajectory_mode="none", model="origin")
    output["anchor"].square().mean().backward()
    assert all(parameter.grad is None for parameter in model.road_seg.parameters())
    assert all(parameter.grad is None for parameter in model.junc_seg.parameters())


def test_production_256_num_targets_four_forward_backward_optimizer_step() -> None:
    torch.set_num_threads(1)
    model = RPNet(
        num_targets=4, backbone_pretrained=False,
        enable_raster_segmentation=True, anchor_grad_to_seg=True).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    image = torch.randn(1, 3, 256, 256)
    raster = (torch.rand(1, 1, 256, 256) > 0.9).float()
    output = model(
        image, raster, None, None, None,
        torch.zeros(1, 1, 64, 64), trajectory_mode="raster_seg_only",
        traj_valid_mask=torch.ones_like(raster), model="origin")
    assert output["road"].shape == (1, 1, 64, 64)
    assert output["junc"].shape == (1, 1, 64, 64)
    assert output["anchor"].shape == (1, 4, 256, 256)
    assert output["anchor_lowrs"].shape == (1, 4, 256, 256)
    loss = sum(output[key].square().mean() for key in (
        "road", "junc", "anchor", "anchor_lowrs"))
    loss.backward()
    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in model.segmentation_raster_fusion.parameters())
    optimizer.step()
