from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch
from torch import nn

from model.seg_raster import StrictZeroPreservingRoadAdapter
from tools.seg_raster.build_stage_s3e_commit_manifest import allowed
from tools.seg_raster.train_stage_s3e import RUN_PROFILES, save_checkpoint
from utils.seg_raster.stage_s3 import load_stage_s3_config
from utils.seg_raster.stage_s3e import (
    configure_stage_s3e_training, finite_tree, weighted_road_loss)


class Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.road_seg = nn.Conv2d(2, 2, 1)
        self.conv_road_final = nn.Conv2d(2, 1, 1)
        self.zero_preserving_road_adapter = StrictZeroPreservingRoadAdapter(2, 2)
        self.other = nn.Conv2d(2, 2, 1)


def test_road_head_lr_zero_freezes_head_and_keeps_adapter_trainable() -> None:
    model = Toy()
    contract, groups = configure_stage_s3e_training(
        model, road_head_lr=0.0, freeze_encoder=False)
    by_name = {row["name"]: row for row in contract}
    assert not by_name["road_seg.weight"]["requires_grad"]
    assert by_name[
        "zero_preserving_road_adapter.projection.weight"]["requires_grad"]
    assert all(group.get("name") != "road_head" for group in groups)


def test_encoder_freeze_removes_it_from_optimizer_and_weight_decay() -> None:
    model = Toy()
    before = {name: value.detach().clone() for name, value in model.named_parameters()
              if "adapter.encoder" in name}
    _, groups = configure_stage_s3e_training(
        model, road_head_lr=1e-5, freeze_encoder=True)
    optimizer = torch.optim.Adam(groups, lr=1e-2, weight_decay=2e-4)
    optimizer.zero_grad(set_to_none=True)
    sum(parameter.square().sum() for parameter in model.parameters()
        if parameter.requires_grad).backward()
    optimizer.step()
    current = dict(model.named_parameters())
    assert all(torch.equal(value, current[name]) for name, value in before.items())


def test_support_ablation_changes_only_multiplier_behavior() -> None:
    torch.manual_seed(21)
    with_support = StrictZeroPreservingRoadAdapter(4, 2, use_support_multiplier=True)
    without_support = StrictZeroPreservingRoadAdapter(4, 2, use_support_multiplier=False)
    without_support.load_state_dict(with_support.state_dict())
    image = torch.randn(1, 4, 4, 4)
    raster = torch.zeros(1, 1, 16, 16)
    raster[..., :4, :4] = 1
    valid = torch.ones_like(raster)
    left = with_support(image, raster, valid)
    right = without_support(image, raster, valid)
    assert torch.equal(left.raster_feature, right.raster_feature)
    assert torch.equal(left.projected_raster, right.projected_raster)
    assert not torch.equal(left.residual, right.residual)


def test_weighted_loss_is_reproducible_and_scale_matches_gradient_norm() -> None:
    logits_a = torch.tensor([[[[-1.0, 1.0]]]], requires_grad=True)
    logits_b = logits_a.detach().clone().requires_grad_(True)
    target = torch.tensor([[[[0.0, 1.0]]]])
    legacy = weighted_road_loss(logits_a, target)
    legacy.backward()
    unscaled = weighted_road_loss(logits_b, target, negative_weight=.25)
    unscaled.backward()
    scale = torch.linalg.vector_norm(logits_a.grad) / torch.linalg.vector_norm(logits_b.grad)
    logits_c = logits_a.detach().clone().requires_grad_(True)
    weighted_road_loss(logits_c, target, negative_weight=.25,
                       scale=float(scale)).backward()
    assert torch.allclose(torch.linalg.vector_norm(logits_a.grad),
                          torch.linalg.vector_norm(logits_c.grad))


def test_dense_checkpoint_refuses_overwrite() -> None:
    model = Toy()
    path = Path(__file__).with_name("_stage_s3e_checkpoint_test.pth.tar")
    if path.exists():
        path.unlink()
    try:
        save_checkpoint(path, model, optimizer_updates=1, samples_seen=20,
                        code_sha="a" * 40, config_sha="b" * 64)
        try:
            save_checkpoint(path, model, optimizer_updates=1, samples_seen=20,
                            code_sha="a" * 40, config_sha="b" * 64)
        except FileExistsError:
            pass
        else:
            raise AssertionError("dense checkpoint overwrite was accepted")
    finally:
        if path.exists():
            path.unlink()


def test_profiles_change_one_declared_factor_each() -> None:
    assert RUN_PROFILES["Z2"] == {
        "control": "aligned", "projection_init": "zero"}
    assert set(RUN_PROFILES["C1"]) == {
        "control", "projection_init", "road_head_lr"}
    assert set(RUN_PROFILES["C2"]) == {
        "control", "projection_init", "use_support_multiplier"}
    assert set(RUN_PROFILES["C3"]) == {
        "control", "projection_init", "freeze_encoder"}


def test_json_finite_rejects_nonfinite() -> None:
    finite_tree({"ok": [1.0, 2.0]})
    for value in (float("nan"), float("inf")):
        try:
            finite_tree({"bad": value})
        except ValueError:
            pass
        else:
            raise AssertionError("non-finite JSON value accepted")


def test_training_worker_keeps_sample_and_seed_parity_checks() -> None:
    from tools.seg_raster import train_stage_s3e
    source = inspect.getsource(train_stage_s3e.main)
    assert "sample identity mismatch" in source
    assert "common tensor mismatch" in source
    assert "set_seed(STAGE_S3D_SEED)" in source


def test_stage_s3e_config_resolves_complete_s3d_training_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_stage_s3_config(root / "configs/stage_s3e_common.yml")
    assert config["S3"]["SPLIT_MANIFEST"] == "artifacts/stage_s3_split_manifest.json"
    assert config["S3"]["EFFECTIVE_BATCH_SIZE"] == 20
    assert config["TRAIN"]["BATCH_SIZE"] == 10
    assert config["S3D"]["MAX_SAMPLES_SEEN"] == 40960
    assert config["TRAJ"]["RASTER"]["PROJECTION_INIT"] == "default"


def test_formal_checkpoint_grid_is_merged_with_dense_early_grid() -> None:
    from tools.seg_raster import train_stage_s3e
    source = inspect.getsource(train_stage_s3e.main)
    assert "dense_updates |= formal_update_grid" in source


def test_stage_s3e_commit_manifest_rejects_runtime_data_and_weights() -> None:
    for path in (
        "data_self/stage_s3e_seg_raster/checkpoints/latest.pth.tar",
        "artifacts/stage_s3e_raster.png",
        "artifacts/stage_s3e_model.pt",
        "tools/seg_raster/__pycache__/worker.pyc",
    ):
        try:
            allowed(path)
        except ValueError:
            pass
        else:
            raise AssertionError("forbidden commit path accepted: " + path)


def test_readonly_launcher_is_dynamic_and_never_hardcodes_gpu_indices() -> None:
    from tools.seg_raster import launch_stage_s3e_readonly
    source = inspect.getsource(launch_stage_s3e_readonly.main)
    assert "collect_inventory" in source
    assert "evaluate_gpu_eligibility" in source
    assert "CUDA_VISIBLE_DEVICES" in source
    assert '"0,2,3"' not in source
    assert "terminate(" not in source
    assert "kill(" not in source


def test_c4_calibration_has_predeclared_acceptance_gates() -> None:
    from tools.seg_raster import calibrate_stage_s3e_gradient_balance
    source = inspect.getsource(calibrate_stage_s3e_gradient_balance.main)
    assert "ratio_error <= 0.20" in source
    assert "gradient_relative_error <= 0.01" in source
    assert '"optimizer_steps_executed": 0' in source
    grid = calibrate_stage_s3e_gradient_balance.NEGATIVE_WEIGHT_CANDIDATE_GRID
    assert min(grid) <= 0.005
    assert max(grid) == 1.0


def test_phase_c_plan_rejects_failed_calibration() -> None:
    from tools.seg_raster.control_stage_s3e import plan
    sha = "a" * 40
    calibration = Path(__file__).with_name("_stage_s3e_failed_calibration.json")
    output = Path(__file__).with_name("_stage_s3e_phase_c_plan.json")
    try:
        calibration.write_text(json.dumps({
            "status": "FAIL", "run_code_sha": sha,
            "optimizer_steps_executed": 0,
        }), encoding="utf-8")
        try:
            plan("C", sha, output, calibration)
        except RuntimeError as error:
            assert "non-PASS" in str(error)
        else:
            raise AssertionError("Phase C accepted a failed calibration")
        assert not output.exists()
    finally:
        calibration.unlink(missing_ok=True)
        output.unlink(missing_ok=True)


def test_phase_c_reducer_preserves_infeasible_c4_status() -> None:
    from tools.seg_raster import control_stage_s3e
    source = inspect.getsource(control_stage_s3e.reduce_c)
    assert '"C1", "C2", "C3"' in source
    assert "NOT_EXECUTED_CALIBRATION_TARGET_INFEASIBLE" in source
    assert "INCONCLUSIVE_CALIBRATION_TARGET_INFEASIBLE" in source
    assert "functional_degradation_persists_with_frozen_head" in source
