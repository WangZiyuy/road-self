"""Select one idle remote GPU and run the frozen-anchor control audit."""

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
        raise RuntimeError("Stage S3C anchor requires a clean frozen checkout")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-code-sha", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--required-free-memory-mb", type=int, default=12000)
    args = parser.parse_args()
    assert_frozen(args.run_code_sha)
    plan_payload = json.loads(args.plan.read_text(encoding="utf-8"))
    stage_s3d = plan_payload.get("stage") == "seg_raster_stage_s3d"
    stage_label = "seg_raster_stage_s3d" if stage_s3d else "seg_raster_stage_s3c"
    output_name = ("stage_s3d_anchor_raw.json" if stage_s3d
                   else "stage_s3c_anchor_raw.json")
    samples, apps = collect_inventory(7.0)
    eligibility_policy = gpu_eligibility_overrides_from_environment()
    eligibility = evaluate_gpu_eligibility(
        samples, apps, required_free_mb=args.required_free_memory_mb,
        excluded_indices=excluded_indices(), **eligibility_policy)
    eligible = [row for row in eligibility if row["eligible"]]
    write_json(args.output_root / "gpu_inventory_phase_ANCHOR.json", {
        "stage": stage_label, "phase": "ANCHOR",
        "execution_environment": "REMOTE_TRAINING_SERVER",
        "remote_host_label": "exp-237-tunnel",
        "run_code_sha": args.run_code_sha,
        "required_free_memory_mb": args.required_free_memory_mb,
        "s3_exclude_gpus": os.environ.get("S3_EXCLUDE_GPUS", ""),
        "samples": samples, "compute_apps": apps,
        "eligibility_policy": eligibility_policy,
        "eligibility": eligibility, "eligible_gpu_count": len(eligible),
        "status": "PASS" if eligible else "BLOCKED_NO_ELIGIBLE_GPU"})
    if not eligible:
        return 3
    gpu = eligible[0]
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu["index"])
    command = [
        sys.executable,
        str(REPO_ROOT / "tools/seg_raster" / (
            "evaluate_stage_s3d_anchor.py" if stage_s3d
            else "evaluate_stage_s3c_anchor.py")),
        "--run-code-sha", args.run_code_sha,
        "--plan", str(args.plan), "--run-root", str(args.run_root),
        "--output", str(args.output_root / output_name),
        "--physical-gpu", str(gpu["index"]),
    ]
    started = utc_now()
    process = subprocess.Popen(command, cwd=REPO_ROOT, env=environment)
    record = {
        "physical_index": gpu["index"], "gpu_uuid": gpu["uuid"],
        "gpu_name": gpu["name"], "pid": process.pid,
        "start_time": started, "end_time": None, "exit_code": None,
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
    }
    while process.poll() is None:
        time.sleep(5)
        try:
            contention_sample, contention_apps = inventory_sample()
            update_post_launch_contention(
                record, contention_sample, contention_apps,
                own_pid=process.pid,
                required_free_memory_mb=args.required_free_memory_mb,
                max_external_compute_memory_mb=eligibility_policy[
                    "max_external_compute_memory_mb"],
            )
        except Exception as error:
            record["post_launch_contention"]["observed"] = True
            record["post_launch_contention"]["gpu_query_error_count"] += 1
            record["post_launch_contention"]["last_query_error"] = (
                type(error).__name__)
    completed_code = int(process.returncode)
    record["end_time"] = utc_now()
    record["exit_code"] = completed_code
    result_path = args.output_root / output_name
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        record["peak_allocated_memory_mb"] = result.get(
            "peak_gpu_memory_allocated_mb")
        record["peak_reserved_memory_mb"] = result.get(
            "peak_gpu_memory_reserved_mb")
    write_json(args.output_root / "gpu_schedule_phase_ANCHOR.json", {
        "stage": stage_label, "phase": "ANCHOR",
        "run_code_sha": args.run_code_sha,
        "status": "PASS" if completed_code == 0 else "FAIL",
        "jobs": [record],
        "external_processes_terminated": False})
    return completed_code


if __name__ == "__main__":
    raise SystemExit(main())
