"""Select one idle remote GPU and run the frozen-anchor control audit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.seg_raster.launch_stage_s3b_phase import (
    collect_inventory, excluded_indices, utc_now, write_json,
)
from utils.seg_raster.stage_s3 import evaluate_gpu_eligibility


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-code-sha", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--required-free-memory-mb", type=int, default=12000)
    args = parser.parse_args()
    samples, apps = collect_inventory(7.0)
    eligibility = evaluate_gpu_eligibility(
        samples, apps, required_free_mb=args.required_free_memory_mb,
        excluded_indices=excluded_indices())
    eligible = [row for row in eligibility if row["eligible"]]
    write_json(args.output_root / "gpu_inventory_phase_ANCHOR.json", {
        "stage": "seg_raster_stage_s3c", "phase": "ANCHOR",
        "execution_environment": "REMOTE_TRAINING_SERVER",
        "samples": samples, "compute_apps": apps,
        "eligibility": eligibility, "eligible_gpu_count": len(eligible),
        "status": "PASS" if eligible else "BLOCKED_NO_ELIGIBLE_GPU"})
    if not eligible:
        return 3
    gpu = eligible[0]
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu["index"])
    command = [
        sys.executable,
        str(REPO_ROOT / "tools/seg_raster/evaluate_stage_s3c_anchor.py"),
        "--run-code-sha", args.run_code_sha,
        "--plan", str(args.plan), "--run-root", str(args.run_root),
        "--output", str(args.output_root / "stage_s3c_anchor_raw.json"),
        "--physical-gpu", str(gpu["index"]),
    ]
    started = utc_now()
    completed = subprocess.run(command, cwd=REPO_ROOT, env=environment)
    write_json(args.output_root / "gpu_schedule_phase_ANCHOR.json", {
        "stage": "seg_raster_stage_s3c", "phase": "ANCHOR",
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "jobs": [{"physical_index": gpu["index"], "gpu_uuid": gpu["uuid"],
                  "gpu_name": gpu["name"], "start_time": started,
                  "end_time": utc_now(), "exit_code": completed.returncode}],
        "external_processes_terminated": False})
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
