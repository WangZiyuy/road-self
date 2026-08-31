"""Dynamic FIFO multi-GPU launcher for a frozen Stage S3D phase."""

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
from utils.seg_raster.stage_s3 import (
    FifoGpuScheduler, evaluate_gpu_eligibility,
    gpu_eligibility_overrides_from_environment, gpu_prelaunch_evidence,
    update_post_launch_contention)


def assert_frozen(expected_sha: str) -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True,
        capture_output=True).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"], check=True,
        text=True, capture_output=True).stdout.strip()
    if head != expected_sha or status:
        raise RuntimeError("Stage S3D phase requires a clean frozen checkout")


def worker_command(job: dict, code_sha: str) -> list[str]:
    return [
        sys.executable, str(REPO_ROOT / "tools/seg_raster/train_stage_s3d.py"),
        "--run-code-sha", code_sha, "--run-key", job["run_key"],
        "--run-id", job["run_id"], "--control", job["control"],
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-code-sha", required=True)
    parser.add_argument("--phase-plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--required-free-memory-mb", type=int, default=12000)
    parser.add_argument("--sample-interval-seconds", type=float, default=7.0)
    args = parser.parse_args()
    assert_frozen(args.run_code_sha)
    plan = json.loads(args.phase_plan.read_text(encoding="utf-8"))
    if plan.get("run_code_sha") != args.run_code_sha:
        raise RuntimeError("phase plan code SHA mismatch")
    samples, apps = collect_inventory(args.sample_interval_seconds)
    required = max(args.required_free_memory_mb,
                   int(os.environ.get("S3_MIN_FREE_MEM_MB", "0") or 0))
    policy = gpu_eligibility_overrides_from_environment()
    eligibility = evaluate_gpu_eligibility(
        samples, apps, required_free_mb=required,
        excluded_indices=excluded_indices(), **policy)
    eligible = [row for row in eligibility if row["eligible"]]
    args.output_root.mkdir(parents=True, exist_ok=True)
    inventory_path = args.output_root / "stage_s3d_gpu_inventory.json"
    schedule_path = args.output_root / "stage_s3d_gpu_schedule.json"
    write_json(inventory_path, {
        "stage": "seg_raster_stage_s3d", "phase": plan["phase"],
        "execution_environment": "REMOTE_TRAINING_SERVER",
        "remote_host_label": "exp-237-tunnel",
        "run_code_sha": args.run_code_sha,
        "required_free_memory_mb": required,
        "s3_exclude_gpus": os.environ.get("S3_EXCLUDE_GPUS", ""),
        "eligibility_policy": policy, "samples": samples,
        "compute_apps": apps, "eligibility": eligibility,
        "eligible_gpu_count": len(eligible),
        "status": "PASS" if eligible else "BLOCKED_NO_ELIGIBLE_GPU",
    })
    if not eligible:
        write_json(schedule_path, {
            "stage": "seg_raster_stage_s3d", "phase": plan["phase"],
            "status": "BLOCKED_NO_ELIGIBLE_GPU", "jobs": [],
            "external_processes_terminated": False})
        return 3
    by_model: dict[str, list[dict]] = {}
    for row in eligible:
        by_model.setdefault(str(row["name"]), []).append(row)
    preferred_model, pool = max(
        by_model.items(), key=lambda pair: (len(pair[1]), pair[0]))
    maximum = min(len(pool), max(1, int(os.environ.get(
        "S3_MAX_PARALLEL", "5"))))
    selected = pool[:maximum]
    scheduler = FifoGpuScheduler(row["index"] for row in selected)
    gpu_by_index = {row["index"]: row for row in selected}
    queue, running, records, peak = list(plan["jobs"]), {}, [], 0
    while queue or running:
        while queue and scheduler.available:
            job = queue.pop(0)
            gpu_index = scheduler.allocate(job["run_id"])
            run_dir = (REPO_ROOT / "data_self/stage_s3d_seg_raster"
                       / job["run_id"])
            run_dir.mkdir(parents=True, exist_ok=True)
            stdout = (run_dir / "stdout.log").open("w", encoding="utf-8")
            stderr = (run_dir / "stderr.log").open("w", encoding="utf-8")
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
            environment["S3D_RUN_CODE_SHA"] = args.run_code_sha
            process = subprocess.Popen(
                worker_command(job, args.run_code_sha), cwd=REPO_ROOT,
                env=environment, stdout=stdout, stderr=stderr)
            gpu = gpu_by_index[gpu_index]
            record = {
                "run_key": job["run_key"], "run_id": job["run_id"],
                "physical_index": gpu_index, "gpu_uuid": gpu["uuid"],
                "gpu_name": gpu["name"], "pid": process.pid,
                "start_time": utc_now(), "end_time": None,
                "exit_code": None,
                "prelaunch_gpu_samples": gpu_prelaunch_evidence(
                    samples, gpu_index, gpu["uuid"]),
                "eligibility_policy": policy,
                "post_launch_contention": {
                    "observed": False, "sample_count": 0,
                    "min_free_memory_mb": None,
                    "max_external_compute_memory_mb": None,
                    "new_external_pids": [], "gpu_query_error_count": 0,
                    "last_sampled_at": None},
                "stdout_path": "${S3D_RUN_ROOT}/" + job["run_id"] + "/stdout.log",
                "stderr_path": "${S3D_RUN_ROOT}/" + job["run_id"] + "/stderr.log",
            }
            records.append(record)
            running[gpu_index] = {
                "process": process, "job": job, "record": record,
                "stdout": stdout, "stderr": stderr}
        peak = max(peak, len(running))
        write_json(schedule_path, {
            "stage": "seg_raster_stage_s3d", "phase": plan["phase"],
            "status": "RUNNING", "run_code_sha": args.run_code_sha,
            "jobs": records, "parallel_job_peak": peak,
            "preferred_homogeneous_gpu_model": preferred_model})
        time.sleep(5)
        if running:
            try:
                sample, current_apps = inventory_sample()
                for state in running.values():
                    update_post_launch_contention(
                        state["record"], sample, current_apps,
                        own_pid=state["process"].pid,
                        required_free_memory_mb=required,
                        max_external_compute_memory_mb=policy[
                            "max_external_compute_memory_mb"])
            except Exception as error:
                for state in running.values():
                    summary = state["record"]["post_launch_contention"]
                    summary["observed"] = True
                    summary["gpu_query_error_count"] += 1
                    summary["last_query_error"] = type(error).__name__
        for gpu_index, state in list(running.items()):
            code = state["process"].poll()
            if code is None:
                continue
            state["stdout"].close()
            state["stderr"].close()
            state["record"]["end_time"] = utc_now()
            state["record"]["exit_code"] = code
            summary_path = (REPO_ROOT / "data_self/stage_s3d_seg_raster"
                            / state["job"]["run_id"] / "summary.json")
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                state["record"]["peak_allocated_memory_mb"] = summary.get(
                    "peak_allocated_memory_mb")
                state["record"]["peak_reserved_memory_mb"] = summary.get(
                    "peak_reserved_memory_mb")
            scheduler.release(gpu_index, state["job"]["run_id"])
            del running[gpu_index]
    status = "PASS" if all(row["exit_code"] == 0 for row in records) else "FAIL"
    write_json(schedule_path, {
        "stage": "seg_raster_stage_s3d", "phase": plan["phase"],
        "status": status, "run_code_sha": args.run_code_sha,
        "jobs": records, "parallel_job_peak": peak,
        "preferred_homogeneous_gpu_model": preferred_model,
        "external_processes_terminated": False})
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
