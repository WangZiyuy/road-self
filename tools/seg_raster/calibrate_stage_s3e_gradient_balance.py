"""Predeclared read-only calibration for the conditional S3E C4 loss."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from model.model import RPNet
from tools.seg_raster.train_stage_s3c import common_batch_sha, set_seed, to_cuda
from tools.seg_raster.train_stage_s3d import build_batches, forward_model
from utils import model_utils
from utils.seg_raster.stage_s3 import load_stage_s3_config, sha256_file
from utils.seg_raster.stage_s3d import STAGE_S3D_SEED, strict_load_stage_s3d_baseline
from utils.seg_raster.stage_s3e import (
    ROAD_HEAD_PREFIXES, configure_stage_s3e_training, finite_tree,
    named_gradient_vector, weighted_road_loss)


# This grid is part of the frozen C4 protocol.  Its lower bound is deliberately
# below the failed 0.05 boundary from the invalidated Phase C attempt so the
# predeclared search can bracket a unit residual-gradient mass ratio.
NEGATIVE_WEIGHT_CANDIDATE_GRID = tuple(
    float(value) for value in np.geomspace(0.005, 1.0, 61))


def write(path: Path, value: object) -> None:
    finite_tree(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def new_model(checkpoint: Path) -> RPNet:
    set_seed(STAGE_S3D_SEED)
    model = RPNet(
        num_targets=4, backbone_pretrained=False,
        enable_trajectory_modules=False,
        enable_zero_preserving_road_adapter=True,
        anchor_grad_to_seg=False, raster_projection_init="zero")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    strict_load_stage_s3d_baseline(model, payload)
    configure_stage_s3e_training(
        model, road_head_lr=1e-5, freeze_encoder=False)
    model.cuda().eval()
    return model


def measure(model: RPNet, batch, negative_weight: float, scale: float) -> dict:
    model.zero_grad(set_to_none=True)
    output = forward_model(model, batch, "aligned")
    residual = output["feature_maps"]["strict_raster_residual"]
    residual.retain_grad()
    target = to_cuda(batch.batch_road_segmentation).to(output["road"].dtype)
    loss = weighted_road_loss(
        output["road"], target, negative_weight=negative_weight, scale=scale)
    loss.backward()
    vector, _ = named_gradient_vector(model, ROAD_HEAD_PREFIXES)
    expanded = (target > 0.5).expand_as(residual)
    grad = residual.grad.detach().abs()
    positive = float(grad[expanded].sum())
    negative = float(grad[~expanded].sum())
    return {
        "negative_weight": float(negative_weight), "loss_scale": float(scale),
        "road_head_gradient_l2": float(torch.linalg.vector_norm(vector.double())),
        "adapter_residual_negative_positive_gradient_mass_ratio":
            negative / max(positive, 1e-30),
        "loss": float(loss.detach()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-code-sha", required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("C4 calibration requires remote CUDA")
    if os.environ.get("S3E_RUN_CODE_SHA") != args.run_code_sha:
        raise RuntimeError("S3E calibration SHA mismatch")
    os.environ["S3D_CONTROL_ROOT"] = os.fspath(args.control_root)
    config = load_stage_s3_config(REPO_ROOT / "configs/stage_s3e_common.yml")
    split = json.loads((REPO_ROOT / config["S3"]["SPLIT_MANIFEST"])
                       .read_text(encoding="utf-8"))
    model_utils.Path.visualize_and_save_path = lambda *a, **k: None
    set_seed(STAGE_S3D_SEED + 1)
    batches, validation_sha, _ = build_batches(config, split, "aligned")
    batch = batches[0]
    baseline = measure(new_model(args.baseline_checkpoint), batch, 1.0, 1.0)
    candidates = []
    for weight in NEGATIVE_WEIGHT_CANDIDATE_GRID:
        candidates.append(measure(
            new_model(args.baseline_checkpoint), batch, float(weight), 1.0))
    selected = min(candidates, key=lambda row: (
        abs(np.log(max(row[
            "adapter_residual_negative_positive_gradient_mass_ratio"], 1e-30))),
        abs(row["negative_weight"] - 1.0)))
    loss_scale = (baseline["road_head_gradient_l2"]
                  / max(selected["road_head_gradient_l2"], 1e-30))
    verified_a = measure(
        new_model(args.baseline_checkpoint), batch,
        selected["negative_weight"], loss_scale)
    verified_b = measure(
        new_model(args.baseline_checkpoint), batch,
        selected["negative_weight"], loss_scale)
    reproducible = verified_a == verified_b
    ratio_error = abs(float(verified_a[
        "adapter_residual_negative_positive_gradient_mass_ratio"]) - 1.0)
    gradient_relative_error = abs(
        float(verified_a["road_head_gradient_l2"])
        - float(baseline["road_head_gradient_l2"])) / max(
            abs(float(baseline["road_head_gradient_l2"])), 1e-30)
    calibrated = ratio_error <= 0.20 and gradient_relative_error <= 0.01
    report = {
        "stage": "seg_raster_stage_s3e", "phase": "C4_calibration",
        "status": "PASS" if reproducible and calibrated else "FAIL",
        "execution_environment": "REMOTE_TRAINING_SERVER",
        "run_code_sha": args.run_code_sha,
        "optimizer_steps_executed": 0,
        "calibration_batch_sha256": common_batch_sha(batch),
        "validation_plan_sha256": validation_sha,
        "baseline_checkpoint_sha256": sha256_file(args.baseline_checkpoint),
        "candidate_grid": list(NEGATIVE_WEIGHT_CANDIDATE_GRID),
        "legacy": baseline, "selected_unscaled": selected,
        "negative_weight": selected["negative_weight"],
        "loss_scale": float(loss_scale), "verified": verified_a,
        "reproducible": reproducible,
        "gradient_ratio_absolute_error_from_one": ratio_error,
        "road_head_gradient_l2_relative_error": gradient_relative_error,
        "calibration_acceptance": calibrated,
        "predeclared_tolerances": {
            "gradient_ratio_absolute_error_max": 0.20,
            "road_head_gradient_l2_relative_error_max": 0.01},
        "selection_rule": "minimize absolute log residual-gradient mass ratio to 1",
        "total_gradient_matching_rule": "match road-head gradient L2 to legacy BCE",
    }
    write(args.output, report)
    return 0 if reproducible and calibrated else 1


if __name__ == "__main__":
    raise SystemExit(main())
