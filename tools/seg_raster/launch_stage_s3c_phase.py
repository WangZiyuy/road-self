"""FIFO multi-GPU launcher for one frozen Stage S3C training phase."""

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
    FifoGpuScheduler, evaluate_gpu_eligibility,
    gpu_eligibility_overrides_from_environment, gpu_prelaunch_evidence,
    update_post_launch_contention,
)


def assert_frozen(expected_sha: str) -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True,
        capture_output=True).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"], check=True,
        text=True, capture_output=True).stdout.strip()
    if head != expected_sha or status:
        raise RuntimeError("Stage S3C phase requires a clean frozen checkout")


def worker_command(job: dict, code_sha: str) -> list[str]:
    command = [
        sys.executable, str(REPO_ROOT / "tools/seg_raster/train_stage_s3c.py"),
        "--run-code-sha", code_sha,
        "--phase", str(job["phase"]),
        "--run-key", str(job["run_key"]),
        "--run-id", str(job["run_id"]),
        "--input-kind", str(job["input_kind"]),
        "--loss-kind", str(job["loss_kind"]),
        "--pos-weight", str(job.get("pos_weight", 1.0)),
        "--loss-alpha", str(job.get("loss_alpha", 1.0)),
    ]
    if job.get("control") is not None:
        command.extend(["--control", str(job["control"])])
    return command


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
    queue = list(plan["jobs"])
    samples, apps = collect_inventory(args.sample_interval_seconds)
    required = max(args.required_free_memory_mb,
                   int(os.environ.get("S3_MIN_FREE_MEM_MB", "0") or 0))
    eligibility_policy = gpu_eligibility_overrides_from_environment()
    eligibility = evaluate_gpu_eligibility(
        samples, apps, required_free_mb=required,
        excluded_indices=excluded_indices(), **eligibility_policy)
    eligible = [row for row in eligibility if row["eligible"]]
    inventory = {
        "stage": "seg_raster_stage_s3c", "phase": plan["phase"],
        "execution_environment": "REMOTE_TRAINING_SERVER",
        "remote_host_label": "exp-237-tunnel",
        "run_code_sha": args.run_code_sha,
        "required_free_memory_mb": required,
        "s3_exclude_gpus": os.environ.get("S3_EXCLUDE_GPUS", ""),
        "eligibility_policy": eligibility_policy,
        "samples": samples, "compute_apps": apps,
        "eligibility": eligibility, "eligible_gpu_count": len(eligible),
        "status": "PASS" if eligible else "BLOCKED_NO_ELIGIBLE_GPU",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.output_root / ("gpu_inventory_phase_" + plan["phase"] + ".json"),
               inventory)
    schedule_path = args.output_root / (
        "gpu_schedule_phase_" + plan["phase"] + ".json")
    if not eligible:
        write_json(schedule_path, {
            "stage": "seg_raster_stage_s3c", "phase": plan["phase"],
            "status": "BLOCKED_NO_ELIGIBLE_GPU", "jobs": [],
            "run_code_sha": args.run_code_sha,
            "external_processes_terminated": False})
        return 3
    by_model: dict[str, list[dict]] = {}
    for row in eligible:
        by_model.setdefault(str(row["name"]), []).append(row)
    preferred_model, pool = max(
        by_model.items(), key=lambda pair: (len(pair[1]), pair[0]))
    maximum = min(len(pool), max(1, int(os.environ.get("S3_MAX_PARALLEL", "4"))))
    selected = pool[:maximum]
    scheduler = FifoGpuScheduler(row["index"] for row in selected)
    gpu_by_index = {row["index"]: row for row in selected}
    running, records, peak = {}, [], 0
    while queue or running:
        while queue and scheduler.available:
            job = queue.pop(0)
            gpu_index = scheduler.allocate(job["run_id"])
            run_dir = REPO_ROOT / "data_self/stage_s3c_seg_raster" / job["run_id"]
            run_dir.mkdir(parents=True, exist_ok=True)
            stdout_handle = (run_dir / "stdout.log").open("w", encoding="utf-8")
            stderr_handle = (run_dir / "stderr.log").open("w", encoding="utf-8")
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
            environment["S3C_RUN_CODE_SHA"] = args.run_code_sha
            process = subprocess.Popen(
                worker_command(job, args.run_code_sha), cwd=REPO_ROOT,
                env=environment, stdout=stdout_handle, stderr=stderr_handle)
            gpu = gpu_by_index[gpu_index]
            record = {
                "run_key": job["run_key"], "run_id": job["run_id"],
                "phase": job["phase"], "physical_index": gpu_index,
                "gpu_uuid": gpu["uuid"], "gpu_name": gpu["name"],
                "pid": process.pid, "start_time": utc_now(), "end_time": None,
                "exit_code": None,
                "prelaunch_gpu_samples": gpu_prelaunch_evidence(
                    samples, gpu_index, gpu["uuid"]),
                "eligibility_policy": eligibility_policy,
                "post_launch_contention": {
                    "observed": False, "sample_count": 0,
                    "min_free_memory_mb": None,
                    "max_external_compute_memory_mb": None,
                    "new_external_pids": [], "gpu_query_error_count": 0,
                    "last_sampled_at": None,
                },
                "stdout_path": "${S3C_RUN_ROOT}/" + job["run_id"] + "/stdout.log",
                "stderr_path": "${S3C_RUN_ROOT}/" + job["run_id"] + "/stderr.log",
            }
            records.append(record)
            running[gpu_index] = {
                "process": process, "job": job, "record": record,
                "stdout": stdout_handle, "stderr": stderr_handle}
        peak = max(peak, len(running))
        write_json(schedule_path, {
            "stage": "seg_raster_stage_s3c", "phase": plan["phase"],
            "status": "RUNNING", "run_code_sha": args.run_code_sha,
            "jobs": records, "parallel_job_peak": peak,
            "preferred_homogeneous_gpu_model": preferred_model})
        time.sleep(5)
        if running:
            try:
                contention_sample, contention_apps = inventory_sample()
                for state in running.values():
                    update_post_launch_contention(
                        state["record"], contention_sample, contention_apps,
                        own_pid=state["process"].pid,
                        required_free_memory_mb=required,
                        max_external_compute_memory_mb=eligibility_policy[
                            "max_external_compute_memory_mb"],
                    )
            except Exception as error:
                for state in running.values():
                    summary = state["record"]["post_launch_contention"]
                    summary["observed"] = True
                    summary["gpu_query_error_count"] += 1
                    summary["last_query_error"] = type(error).__name__
        for gpu_index, state in list(running.items()):
            exit_code = state["process"].poll()
            if exit_code is None:
                continue
            state["stdout"].close()
            state["stderr"].close()
            state["record"]["end_time"] = utc_now()
            state["record"]["exit_code"] = exit_code
            summary_path = (REPO_ROOT / "data_self/stage_s3c_seg_raster"
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
        "stage": "seg_raster_stage_s3c", "phase": plan["phase"],
        "status": status, "run_code_sha": args.run_code_sha,
        "jobs": records, "parallel_job_peak": peak,
        "preferred_homogeneous_gpu_model": preferred_model,
        "external_processes_terminated": False})
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
