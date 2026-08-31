"""Dynamically launch one read-only Stage S3E CUDA audit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.seg_raster.launch_stage_s3b_phase import (
    collect_inventory, excluded_indices, inventory_sample, utc_now, write_json)
from tools.seg_raster.launch_stage_s3e_phase import assert_frozen
from utils.seg_raster.stage_s3 import (
    evaluate_gpu_eligibility, gpu_eligibility_overrides_from_environment,
    gpu_prelaunch_evidence, update_post_launch_contention)


def command(args: argparse.Namespace) -> list[str]:
    if args.job == "phase-a":
        required = (
            args.checkpoint_root, args.checkpoint_inventory,
            args.control_root, args.source_stage_s3d_sha)
        if any(value is None for value in required):
            raise ValueError("phase-a source arguments are required")
        return [
            sys.executable,
            str(REPO_ROOT / "tools/seg_raster/evaluate_stage_s3e_cross_transplant.py"),
            "--checkpoint-root", str(args.checkpoint_root),
            "--checkpoint-inventory", str(args.checkpoint_inventory),
            "--control-root", str(args.control_root),
            "--output-root", str(args.output_root),
            "--run-code-sha", args.run_code_sha,
            "--source-stage-s3d-sha", args.source_stage_s3d_sha,
        ]
    required = (args.baseline_checkpoint, args.control_root, args.calibration_output)
    if any(value is None for value in required):
        raise ValueError("C4 calibration source arguments are required")
    return [
        sys.executable,
        str(REPO_ROOT / "tools/seg_raster/calibrate_stage_s3e_gradient_balance.py"),
        "--run-code-sha", args.run_code_sha,
        "--baseline-checkpoint", str(args.baseline_checkpoint),
        "--control-root", str(args.control_root),
        "--output", str(args.calibration_output),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", choices=("phase-a", "c4-calibration"), required=True)
    parser.add_argument("--run-code-sha", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--checkpoint-inventory", type=Path)
    parser.add_argument("--control-root", type=Path)
    parser.add_argument("--source-stage-s3d-sha")
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--calibration-output", type=Path)
    parser.add_argument("--required-free-memory-mb", type=int, default=12000)
    parser.add_argument("--sample-interval-seconds", type=float, default=7.0)
    args = parser.parse_args()
    assert_frozen(args.run_code_sha)
    if os.environ.get("S3_EXCLUDE_GPUS", ""):
        raise RuntimeError("formal Stage S3E keeps S3_EXCLUDE_GPUS empty")

    samples, apps = collect_inventory(args.sample_interval_seconds)
    policy = gpu_eligibility_overrides_from_environment()
    required = max(args.required_free_memory_mb,
                   int(os.environ.get("S3_MIN_FREE_MEM_MB", "0") or 0))
    eligibility = evaluate_gpu_eligibility(
        samples, apps, required_free_mb=required,
        excluded_indices=excluded_indices(), **policy)
    eligible = [row for row in eligibility if row["eligible"]]
    args.output_root.mkdir(parents=True, exist_ok=True)
    label = "a" if args.job == "phase-a" else "c4_calibration"
    inventory_path = args.output_root / (
        "stage_s3e_gpu_inventory_{}.json".format(label))
    schedule_path = args.output_root / (
        "stage_s3e_gpu_schedule_{}.json".format(label))
    write_json(inventory_path, {
        "stage": "seg_raster_stage_s3e", "job": args.job,
        "status": "PASS" if eligible else "BLOCKED_NO_ELIGIBLE_GPU",
        "execution_environment": "REMOTE_TRAINING_SERVER",
        "run_code_sha": args.run_code_sha, "samples": samples,
        "compute_apps": apps, "eligibility": eligibility,
        "eligibility_policy": policy, "eligible_gpu_count": len(eligible),
        "s3_exclude_gpus": "", "required_free_memory_mb": required,
    })
    if not eligible:
        write_json(schedule_path, {
            "stage": "seg_raster_stage_s3e", "job": args.job,
            "status": "BLOCKED_NO_ELIGIBLE_GPU", "jobs": [],
            "external_processes_terminated": False})
        return 3

    by_model: dict[str, list[dict]] = {}
    for row in eligible:
        by_model.setdefault(str(row["name"]), []).append(row)
    _, pool = max(by_model.items(), key=lambda pair: (len(pair[1]), pair[0]))
    gpu = pool[0]
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu["index"])
    environment["S3E_RUN_CODE_SHA"] = args.run_code_sha
    stdout_path = args.output_root / ("stage_s3e_{}.stdout.log".format(label))
    stderr_path = args.output_root / ("stage_s3e_{}.stderr.log".format(label))
    record = {
        "job": args.job, "physical_index": gpu["index"],
        "gpu_uuid": gpu["uuid"], "gpu_name": gpu["name"],
        "prelaunch_gpu_samples": gpu_prelaunch_evidence(
            samples, gpu["index"], gpu["uuid"]),
        "start_time": utc_now(), "end_time": None, "exit_code": None,
        "post_launch_contention": {
            "observed": False, "sample_count": 0,
            "min_free_memory_mb": None,
            "max_external_compute_memory_mb": None,
            "new_external_pids": [], "gpu_query_error_count": 0,
            "last_sampled_at": None},
        "optimizer_steps_executed": 0,
    }
    with stdout_path.open("w", encoding="utf-8") as stdout, \
            stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command(args), cwd=REPO_ROOT, env=environment,
            stdout=stdout, stderr=stderr)
        record["pid"] = process.pid
        while process.poll() is None:
            time.sleep(5)
            try:
                sample, current_apps = inventory_sample()
                update_post_launch_contention(
                    record, sample, current_apps, own_pid=process.pid,
                    required_free_memory_mb=required,
                    max_external_compute_memory_mb=policy[
                        "max_external_compute_memory_mb"])
            except Exception as error:
                contention = record["post_launch_contention"]
                contention["observed"] = True
                contention["gpu_query_error_count"] += 1
                contention["last_query_error"] = type(error).__name__
        record["exit_code"] = process.returncode
    record["end_time"] = utc_now()
    write_json(schedule_path, {
        "stage": "seg_raster_stage_s3e", "job": args.job,
        "status": "PASS" if record["exit_code"] == 0 else "FAIL",
        "run_code_sha": args.run_code_sha, "jobs": [record],
        "parallel_job_peak": 1, "external_processes_terminated": False})
    return 0 if record["exit_code"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
