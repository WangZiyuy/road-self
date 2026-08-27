"""Freeze the S3 optimizer-step budget from a 100-step CUDA timing probe."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from tools.seg_raster.train_stage_s3 import (
    _cfg_for_dataset, _forward, _load_initialization, _losses, _model_for,
    frozen_checkout)
from utils.OSMDataset import OSMDataset
from utils.seg_raster.stage_s3 import load_stage_s3_config


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-code-sha", required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--probe-steps", type=int, default=100)
    parser.add_argument("--eligible-gpu-count", type=int, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=(REPO_ROOT / "data_self/stage_s3_seg_raster/runtime/audits/"
                 "stage_s3_budget_manifest.json"))
    args = parser.parse_args()
    frozen_checkout(args.run_code_sha)
    if json.loads(args.preflight.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("budget probe requires preflight PASS")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("budget probe requires one visible CUDA GPU")
    config = load_stage_s3_config(
        REPO_ROOT / "configs/stage_s3_J1_aligned_joint.yml")
    split = json.loads((
        REPO_ROOT / "artifacts/stage_s3_split_manifest.json"
    ).read_text(encoding="utf-8"))
    extent = split["train_extent"]
    extent_list = [extent["x0"], extent["y0"], extent["x1"], extent["y1"]]
    random.seed(20260827)
    np.random.seed(20260827)
    torch.manual_seed(20260827)
    torch.cuda.manual_seed_all(20260827)
    model = _model_for(config)
    _load_initialization(model, config)
    model.cuda().train()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["TRAIN"]["SOLVER"]["LEARNING_RATE"]),
        betas=(0.9, 0.99),
        weight_decay=float(config["TRAIN"]["SOLVER"]["WEIGHT_DECAY"]))
    dataset = OSMDataset(_cfg_for_dataset(config, extent_list), net=None, training=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for step in range(1, args.probe_steps + 1):
        batch = dataset.get_batch()
        optimizer.zero_grad(set_to_none=True)
        output = _forward(model, batch, "raster_seg_only")
        losses = _losses(output, batch)
        losses["total"].backward()
        optimizer.step()
        batch.batch_output_road = torch.sigmoid(output["road"]).detach().cpu().numpy()
        batch.batch_output_junc = torch.sigmoid(output["junc"]).detach().cpu().numpy()
        batch.batch_output_anchor_maps = torch.sigmoid(output["anchor"]).detach().cpu().numpy()
        dataset.push_and_vis_batch(batch, 0, step)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    seconds_per_step = elapsed / args.probe_steps
    baseline_steps = 50 * 2048
    parallelism = max(1, min(6, int(args.eligible_gpu_count)))
    estimated_hours = baseline_steps * seconds_per_step * 6 / parallelism / 3600
    evaluation_interval = int(config["S3"]["EVALUATION_INTERVAL"])
    if estimated_hours <= 12.0:
        kind = "FULL_BASELINE"
        optimizer_steps = baseline_steps
    else:
        kind = "SCREENING"
        optimizer_steps = min(
            baseline_steps,
            max(math.ceil(0.25 * baseline_steps), 3 * evaluation_interval))
    evaluation_checkpoints = optimizer_steps // evaluation_interval
    if evaluation_checkpoints < 5:
        raise RuntimeError("frozen budget would provide fewer than five evaluations")
    payload = {
        "stage": "seg_raster_stage_s3", "status": "PASS",
        "run_code_sha": args.run_code_sha,
        "n_baseline_steps": baseline_steps,
        "timing_probe": {
            "mode": "aligned_raster_joint", "steps": args.probe_steps,
            "elapsed_seconds": elapsed, "seconds_per_step": seconds_per_step,
            "crop_size": 256, "num_targets": 4, "includes_data_generation": True,
        },
        "eligible_gpu_count": args.eligible_gpu_count,
        "assumed_parallelism": parallelism,
        "estimated_full_matrix_wall_clock_hours": estimated_hours,
        "kind": kind, "optimizer_steps": optimizer_steps,
        "evaluation_interval": evaluation_interval,
        "evaluation_checkpoint_count": evaluation_checkpoints,
        "single_budget_for_all_six_runs": True,
        "frozen_before_comparative_results": True,
    }
    write_json(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
