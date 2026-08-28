"""FIFO launcher for the Stage S3A detach graph-control matrix."""

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

from tools.seg_raster.audit_stage_s3a_launch import (
    collect_inventory,
    verify_checkout,
    write_json,
)
from utils.seg_raster.stage_s3 import (
    EXPERIMENT_MATRIX,
    FifoGpuScheduler,
    evaluate_gpu_eligibility,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-code-sha", required=True)
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument("--source-runtime", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--sample-interval-seconds", type=float, default=5.0)
    parser.add_argument("--required-free-memory-mb", type=int, default=4504)
    args = parser.parse_args()
    verify_checkout(args.audit_code_sha)
    samples, apps = collect_inventory(args.sample_interval_seconds)
    excluded = {
        int(value.strip()) for value in os.environ.get(
            "S3_EXCLUDE_GPUS", "").split(",") if value.strip()
    }
    rows = evaluate_gpu_eligibility(
        samples, apps, required_free_mb=int(args.required_free_memory_mb),
        excluded_indices=excluded)
    eligible = [row for row in rows if row["eligible"]]
    max_parallel = max(1, int(os.environ.get("S3_MAX_PARALLEL", "8")))
    by_model: dict[str, list[dict]] = {}
    for row in eligible:
        by_model.setdefault(str(row["name"]), []).append(row)
    if not by_model:
        write_json(args.output_root / "stage_s3a_graph_gpu_schedule.json", {
            "stage": "seg_raster_stage_s3a", "status": "BLOCKED_NO_ELIGIBLE_GPU",
            "jobs": [], "parallel_job_peak": 0, "eligibility": rows})
        return 3
    preferred_model, pool = max(
        by_model.items(), key=lambda item: (len(item[1]), item[0]))
    selected = pool[:max_parallel]
    specs = {spec.key: spec for spec in EXPERIMENT_MATRIX}
    jobs = []
    for key in ("C0", "C1", "C2", "C3"):
        spec = specs[key]
        for kind in ("best", "latest"):
            checkpoint = args.source_run_root / spec.run_id / "checkpoints" / (
                kind + ".pth.tar")
            jobs.append({
                "job_id": "graph_{}_{}".format(key, kind),
                "run_key": key,
                "run_id": spec.run_id,
                "checkpoint_kind": kind,
                "checkpoint": checkpoint,
                "output_dir": args.output_root / "graph" / key / kind,
                "result": args.output_root / "graph_results" / (
                    "{}_{}.json".format(key, kind)),
            })
    scheduler = FifoGpuScheduler(row["index"] for row in selected)
    gpu_by_index = {row["index"]: row for row in selected}
    queue, running, records = list(jobs), {}, []
    peak = 0
    logs = args.output_root / "graph_logs"
    logs.mkdir(parents=True, exist_ok=True)
    while queue or running:
        while queue and scheduler.available:
            job = queue.pop(0)
            index = scheduler.allocate(job["job_id"])
            assert index is not None
            stdout_path = logs / (job["job_id"] + ".stdout")
            stderr_path = logs / (job["job_id"] + ".stderr")
            stdout = stdout_path.open("w", encoding="utf-8")
            stderr = stderr_path.open("w", encoding="utf-8")
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = str(index)
            command = [
                sys.executable,
                str(REPO_ROOT / "tools/seg_raster/audit_stage_s3a_graph.py"),
                "--audit-code-sha", args.audit_code_sha,
                "--run-key", job["run_key"],
                "--run-id", job["run_id"],
                "--checkpoint", str(job["checkpoint"]),
                "--checkpoint-kind", job["checkpoint_kind"],
                "--source-runtime", str(args.source_runtime),
                "--base-config", str(args.base_config),
                "--output-dir", str(job["output_dir"]),
                "--result", str(job["result"]),
                "--physical-gpu", str(index),
            ]
            process = subprocess.Popen(
                command, cwd=REPO_ROOT, env=environment,
                stdout=stdout, stderr=stderr)
            gpu = gpu_by_index[index]
            record = {
                "job_id": job["job_id"], "run_key": job["run_key"],
                "checkpoint_kind": job["checkpoint_kind"],
                "physical_gpu_index": index, "gpu_uuid": gpu["uuid"],
                "gpu_name": gpu["name"], "pid": process.pid,
                "start_time": utc_now(), "end_time": None, "exit_code": None,
                "result_logical_path": "${S3A_REMOTE_OUTPUT}/graph_results/" + job["result"].name,
            }
            records.append(record)
            running[index] = {
                "job": job, "process": process, "record": record,
                "stdout": stdout, "stderr": stderr,
            }
        peak = max(peak, len(running))
        write_json(args.output_root / "stage_s3a_graph_gpu_schedule.json", {
            "stage": "seg_raster_stage_s3a", "status": "RUNNING",
            "audit_code_sha": args.audit_code_sha, "jobs": records,
            "parallel_job_peak": peak})
        time.sleep(5.0)
        for index, state in list(running.items()):
            code = state["process"].poll()
            if code is None:
                continue
            state["stdout"].close()
            state["stderr"].close()
            state["record"]["end_time"] = utc_now()
            state["record"]["exit_code"] = int(code)
            scheduler.release(index, state["job"]["job_id"])
            del running[index]
    status = "PASS" if all(row["exit_code"] == 0 for row in records) else "FAIL"
    write_json(args.output_root / "stage_s3a_graph_gpu_schedule.json", {
        "stage": "seg_raster_stage_s3a",
        "execution_environment": "REMOTE_TRAINING_SERVER",
        "status": status, "audit_code_sha": args.audit_code_sha,
        "jobs": records, "parallel_job_peak": peak,
        "preferred_homogeneous_gpu_model": preferred_model,
    })
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
