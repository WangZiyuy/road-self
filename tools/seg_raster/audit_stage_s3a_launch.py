"""FIFO multi-GPU launcher for immutable Stage S3A checkpoint evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.seg_raster.stage_s3 import (
    EXPERIMENT_MATRIX,
    FifoGpuScheduler,
    evaluate_gpu_eligibility,
    parse_compute_apps_csv,
    parse_gpu_inventory_csv,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")


def run_text(command: list[str]) -> str:
    return subprocess.run(
        command, check=True, text=True, capture_output=True).stdout


def gpu_snapshot() -> dict:
    text = run_text([
        "nvidia-smi",
        "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,"
        "memory.free,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ])
    return {"sampled_at_utc": utc_now(), "gpus": parse_gpu_inventory_csv(text)}


def compute_apps() -> list[dict]:
    text = run_text([
        "nvidia-smi",
        "--query-compute-apps=pid,gpu_uuid,used_memory,process_name",
        "--format=csv,noheader,nounits",
    ])
    return parse_compute_apps_csv(text)


def collect_inventory(interval_seconds: float) -> tuple[list[dict], list[dict]]:
    samples = []
    apps = {}
    for index in range(3):
        samples.append(gpu_snapshot())
        for app in compute_apps():
            apps[(app["pid"], app["gpu_uuid"])] = app
        if index != 2:
            time.sleep(float(interval_seconds))
    return samples, list(apps.values())


def config_for(key: str) -> Path:
    paths = sorted((REPO_ROOT / "configs").glob(
        "stage_s3_{}_*yml".format(key)))
    if len(paths) != 1:
        raise RuntimeError("expected one config for {}".format(key))
    return paths[0]


def verify_checkout(expected_sha: str) -> None:
    head = run_text(["git", "rev-parse", "HEAD"]).strip()
    status = run_text([
        "git", "status", "--short", "--untracked-files=no"]).strip()
    if head != expected_sha or status:
        raise RuntimeError("S3A launcher requires the clean audit code commit")


def build_jobs(source_run_root: Path, output_root: Path) -> list[dict]:
    jobs = []
    for spec in EXPERIMENT_MATRIX:
        for kind in ("best", "latest"):
            checkpoint = source_run_root / spec.run_id / "checkpoints" / (
                kind + ".pth.tar")
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            jobs.append({
                "job_id": "{}_{}_repeat0".format(spec.key, kind),
                "run_key": spec.key,
                "run_id": spec.run_id,
                "checkpoint_kind": kind,
                "checkpoint": checkpoint,
                "repeat_index": 0,
                "output": output_root / "checkpoint_results" / (
                    "{}_{}_repeat0.json".format(spec.key, kind)),
            })
    for key in ("C0", "C1"):
        spec = next(value for value in EXPERIMENT_MATRIX if value.key == key)
        for kind in ("best", "latest"):
            jobs.append({
                "job_id": "{}_{}_repeat1".format(key, kind),
                "run_key": key,
                "run_id": spec.run_id,
                "checkpoint_kind": kind,
                "checkpoint": source_run_root / spec.run_id / "checkpoints" / (
                    kind + ".pth.tar"),
                "repeat_index": 1,
                "output": output_root / "checkpoint_results" / (
                    "{}_{}_repeat1.json".format(key, kind)),
            })
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-code-sha", required=True)
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-interval-seconds", type=float, default=5.0)
    parser.add_argument("--required-free-memory-mb", type=int, default=4504)
    args = parser.parse_args()
    verify_checkout(args.audit_code_sha)
    args.output_root.mkdir(parents=True, exist_ok=True)
    samples, apps = collect_inventory(args.sample_interval_seconds)
    excluded = {
        int(value.strip()) for value in os.environ.get(
            "S3_EXCLUDE_GPUS", "").split(",") if value.strip()
    }
    eligible_rows = evaluate_gpu_eligibility(
        samples, apps,
        required_free_mb=max(
            int(args.required_free_memory_mb),
            int(os.environ.get("S3_MIN_FREE_MEM_MB", "0") or 0)),
        excluded_indices=excluded,
    )
    eligible = [row for row in eligible_rows if row["eligible"]]
    max_parallel = max(1, int(os.environ.get("S3_MAX_PARALLEL", "8")))
    by_model: dict[str, list[dict]] = {}
    for row in eligible:
        by_model.setdefault(str(row["name"]), []).append(row)
    selected = []
    preferred_model = None
    if by_model:
        preferred_model, pool = max(
            by_model.items(), key=lambda item: (len(item[1]), item[0]))
        selected = pool[:max_parallel]
    inventory = {
        "stage": "seg_raster_stage_s3a",
        "execution_environment": "REMOTE_TRAINING_SERVER",
        "audit_code_sha": args.audit_code_sha,
        "required_free_memory_mb": int(args.required_free_memory_mb),
        "samples": samples,
        "compute_apps": apps,
        "eligibility": eligible_rows,
        "eligible_gpu_count": len(eligible),
        "selected_gpu_count": len(selected),
        "preferred_homogeneous_gpu_model": preferred_model,
        "status": "PASS" if selected else "BLOCKED_NO_ELIGIBLE_GPU",
        "external_process_policy": "never terminate or preempt",
    }
    write_json(args.output_root / "stage_s3a_gpu_inventory.json", inventory)
    if not selected:
        write_json(args.output_root / "stage_s3a_gpu_schedule.json", {
            "stage": "seg_raster_stage_s3a",
            "status": "BLOCKED_NO_ELIGIBLE_GPU",
            "jobs": [],
            "parallel_job_peak": 0,
        })
        return 3
    jobs = build_jobs(args.source_run_root, args.output_root)
    scheduler = FifoGpuScheduler(row["index"] for row in selected)
    gpu_by_index = {row["index"]: row for row in selected}
    queue = list(jobs)
    running: dict[int, dict] = {}
    records = []
    peak = 0
    logs = args.output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    while queue or running:
        while queue and scheduler.available:
            job = queue.pop(0)
            gpu_index = scheduler.allocate(job["job_id"])
            assert gpu_index is not None
            stdout_path = logs / (job["job_id"] + ".stdout")
            stderr_path = logs / (job["job_id"] + ".stderr")
            stdout_handle = stdout_path.open("w", encoding="utf-8")
            stderr_handle = stderr_path.open("w", encoding="utf-8")
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
            command = [
                sys.executable,
                str(REPO_ROOT / "tools/seg_raster/audit_stage_s3a_remote_eval.py"),
                "--config", str(config_for(job["run_key"])),
                "--run-key", job["run_key"],
                "--run-id", job["run_id"],
                "--checkpoint", str(job["checkpoint"]),
                "--checkpoint-kind", job["checkpoint_kind"],
                "--repeat-index", str(job["repeat_index"]),
                "--audit-code-sha", args.audit_code_sha,
                "--output", str(job["output"]),
            ]
            process = subprocess.Popen(
                command, cwd=REPO_ROOT, env=environment,
                stdout=stdout_handle, stderr=stderr_handle)
            gpu = gpu_by_index[gpu_index]
            record = {
                "job_id": job["job_id"],
                "run_key": job["run_key"],
                "checkpoint_kind": job["checkpoint_kind"],
                "repeat_index": job["repeat_index"],
                "physical_gpu_index": gpu_index,
                "gpu_uuid": gpu["uuid"],
                "gpu_name": gpu["name"],
                "pid": process.pid,
                "start_time": utc_now(),
                "end_time": None,
                "exit_code": None,
                "output_logical_path": "${S3A_REMOTE_OUTPUT}/checkpoint_results/" + job["output"].name,
                "stdout_logical_path": "${S3A_REMOTE_OUTPUT}/logs/" + stdout_path.name,
                "stderr_logical_path": "${S3A_REMOTE_OUTPUT}/logs/" + stderr_path.name,
            }
            records.append(record)
            running[gpu_index] = {
                "job": job, "process": process, "record": record,
                "stdout": stdout_handle, "stderr": stderr_handle,
            }
        peak = max(peak, len(running))
        write_json(args.output_root / "stage_s3a_gpu_schedule.json", {
            "stage": "seg_raster_stage_s3a", "status": "RUNNING",
            "audit_code_sha": args.audit_code_sha, "jobs": records,
            "parallel_job_peak": peak,
        })
        time.sleep(2.0)
        for gpu_index, state in list(running.items()):
            code = state["process"].poll()
            if code is None:
                continue
            state["stdout"].close()
            state["stderr"].close()
            state["record"]["end_time"] = utc_now()
            state["record"]["exit_code"] = int(code)
            scheduler.release(gpu_index, state["job"]["job_id"])
            del running[gpu_index]
    status = "PASS" if all(row["exit_code"] == 0 for row in records) else "FAIL"
    write_json(args.output_root / "stage_s3a_gpu_schedule.json", {
        "stage": "seg_raster_stage_s3a",
        "execution_environment": "REMOTE_TRAINING_SERVER",
        "status": status,
        "audit_code_sha": args.audit_code_sha,
        "jobs": records,
        "parallel_job_peak": peak,
        "preferred_homogeneous_gpu_model": preferred_model,
    })
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
