"""Dynamically select one remote GPU for Stage S3C preflight/baseline."""

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
    collect_inventory, excluded_indices, inventory_sample, utc_now, write_json,
)
from utils.seg_raster.stage_s3 import (
    evaluate_gpu_eligibility, gpu_eligibility_overrides_from_environment,
    gpu_prelaunch_evidence, update_post_launch_contention,
)


def assert_frozen(expected_sha: str) -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True,
        capture_output=True).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"], check=True,
        text=True, capture_output=True).stdout.strip()
    if head != expected_sha or status:
        raise RuntimeError("Stage S3C audit requires a clean frozen checkout")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-code-sha", required=True)
    parser.add_argument("--action", choices=("preflight", "baseline"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-plan", type=Path, required=True)
    parser.add_argument("--required-free-memory-mb", type=int, default=12000)
    parser.add_argument("--sample-interval-seconds", type=float, default=7.0)
    args = parser.parse_args()
    assert_frozen(args.run_code_sha)

    samples, apps = collect_inventory(args.sample_interval_seconds)
    required = max(args.required_free_memory_mb,
                   int(os.environ.get("S3_MIN_FREE_MEM_MB", "0") or 0))
    eligibility_policy = gpu_eligibility_overrides_from_environment()
    eligibility = evaluate_gpu_eligibility(
        samples, apps, required_free_mb=required,
        excluded_indices=excluded_indices(), **eligibility_policy)
    eligible = [row for row in eligibility if row["eligible"]]
    phase = args.action.upper()
    args.output_root.mkdir(parents=True, exist_ok=True)
    inventory_path = args.output_root / ("gpu_inventory_phase_" + phase + ".json")
    schedule_path = args.output_root / ("gpu_schedule_phase_" + phase + ".json")
    write_json(inventory_path, {
        "stage": "seg_raster_stage_s3c", "phase": phase,
        "execution_environment": "REMOTE_TRAINING_SERVER",
        "remote_host_label": "exp-237-tunnel",
        "run_code_sha": args.run_code_sha,
        "required_free_memory_mb": required,
        "s3_exclude_gpus": os.environ.get("S3_EXCLUDE_GPUS", ""),
        "eligibility_policy": eligibility_policy,
        "samples": samples, "compute_apps": apps,
        "eligibility": eligibility, "eligible_gpu_count": len(eligible),
        "status": "PASS" if eligible else "BLOCKED_NO_ELIGIBLE_GPU",
    })
    if not eligible:
        write_json(schedule_path, {
            "stage": "seg_raster_stage_s3c", "phase": phase,
            "status": "BLOCKED_NO_ELIGIBLE_GPU", "jobs": [],
            "external_processes_terminated": False,
        })
        return 3

    # Pick from the current three-sample evidence, not a persistent GPU list.
    gpu = max(eligible, key=lambda row: (
        min(row["free_memory_series_mb"]),
        -max(row["utilization_series_percent"]),
        str(row["uuid"]),
    ))
    logs = args.output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout_handle = (logs / (args.action + ".stdout.log")).open(
        "w", encoding="utf-8")
    stderr_handle = (logs / (args.action + ".stderr.log")).open(
        "w", encoding="utf-8")
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu["index"])
    command = [
        sys.executable,
        str(REPO_ROOT / "tools/seg_raster/audit_stage_s3c_remote.py"),
        args.action, "--run-code-sha", args.run_code_sha,
        "--output-root", str(args.output_root),
        "--sample-plan", str(args.sample_plan),
    ]
    process = subprocess.Popen(
        command, cwd=REPO_ROOT, env=environment,
        stdout=stdout_handle, stderr=stderr_handle)
    record = {
        "action": args.action, "physical_index": gpu["index"],
        "gpu_uuid": gpu["uuid"], "gpu_name": gpu["name"],
        "pid": process.pid, "start_time": utc_now(), "end_time": None,
        "exit_code": None,
        "prelaunch_gpu_samples": gpu_prelaunch_evidence(
            samples, gpu["index"], gpu["uuid"]),
        "eligibility_policy": eligibility_policy,
        "post_launch_contention": {
            "observed": False, "sample_count": 0,
            "min_free_memory_mb": None,
            "max_external_compute_memory_mb": None,
            "new_external_pids": [], "gpu_query_error_count": 0,
            "last_sampled_at": None,
        },
        "stdout_path": "${S3C_REMOTE_OUTPUT}/logs/" + args.action + ".stdout.log",
        "stderr_path": "${S3C_REMOTE_OUTPUT}/logs/" + args.action + ".stderr.log",
    }
    write_json(schedule_path, {
        "stage": "seg_raster_stage_s3c", "phase": phase,
        "status": "RUNNING", "run_code_sha": args.run_code_sha,
        "jobs": [record], "external_processes_terminated": False,
    })
    while process.poll() is None:
        time.sleep(5)
        try:
            contention_sample, contention_apps = inventory_sample()
            update_post_launch_contention(
                record, contention_sample, contention_apps,
                own_pid=process.pid, required_free_memory_mb=required,
                max_external_compute_memory_mb=eligibility_policy[
                    "max_external_compute_memory_mb"],
            )
        except Exception as error:
            record["post_launch_contention"]["observed"] = True
            record["post_launch_contention"]["gpu_query_error_count"] += 1
            record["post_launch_contention"]["last_query_error"] = (
                type(error).__name__)
    stdout_handle.close()
    stderr_handle.close()
    record["exit_code"] = int(process.returncode)
    record["end_time"] = utc_now()
    result_name = ("stage_s3c_remote_preflight.json" if args.action == "preflight"
                   else "stage_s3c_baseline_evaluation.json")
    result_path = args.output_root / result_name
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        record["peak_allocated_memory_mb"] = result.get(
            "peak_allocated_memory_mb")
        record["peak_reserved_memory_mb"] = result.get(
            "peak_reserved_memory_mb")
    status = "PASS" if process.returncode == 0 else "FAIL"
    write_json(schedule_path, {
        "stage": "seg_raster_stage_s3c", "phase": phase,
        "status": status, "run_code_sha": args.run_code_sha,
        "jobs": [record], "external_processes_terminated": False,
    })
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
