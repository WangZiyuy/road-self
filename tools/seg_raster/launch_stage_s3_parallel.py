"""Discover idle GPUs and FIFO-launch the frozen Stage S3 run matrix."""

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


GPU_QUERY = (
    "index,uuid,name,driver_version,memory.total,memory.used,memory.free,"
    "utilization.gpu,temperature.gpu")
APP_QUERY = "pid,gpu_uuid,used_memory,process_name"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_text(command: list[str]) -> str:
    return subprocess.run(
        command, check=True, text=True, encoding="utf-8",
        errors="replace", capture_output=True).stdout


def query_gpu_snapshot() -> dict[str, object]:
    text = _run_text([
        "nvidia-smi", "--query-gpu={}".format(GPU_QUERY),
        "--format=csv,noheader,nounits"])
    return {"sampled_at": utc_now(), "gpus": parse_gpu_inventory_csv(text)}


def query_compute_apps() -> list[dict[str, object]]:
    text = _run_text([
        "nvidia-smi", "--query-compute-apps={}".format(APP_QUERY),
        "--format=csv,noheader,nounits"])
    return parse_compute_apps_csv(text)


def collect_three_samples(interval_seconds: float = 10.0) -> tuple[list, list]:
    samples = []
    apps_by_identity = {}
    for index in range(3):
        samples.append(query_gpu_snapshot())
        for app in query_compute_apps():
            apps_by_identity[(app["pid"], app["gpu_uuid"])] = app
        if index != 2:
            time.sleep(interval_seconds)
    return samples, list(apps_by_identity.values())


def assert_frozen_checkout(expected_sha: str) -> None:
    branch = _run_text(["git", "branch", "--show-current"]).strip()
    head = _run_text(["git", "rev-parse", "HEAD"]).strip()
    status = _run_text(["git", "status", "--short", "--untracked-files=all"])
    if branch != "feat/seg-raster-only":
        raise RuntimeError("unexpected branch: {}".format(branch))
    if head != expected_sha:
        raise RuntimeError("HEAD differs from frozen run code SHA")
    if status.strip():
        raise RuntimeError("training requires a clean frozen checkout")


def _excluded_indices() -> set[int]:
    raw = os.environ.get("S3_EXCLUDE_GPUS", "")
    return {int(value.strip()) for value in raw.split(",") if value.strip()}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")


def _config_path(key: str) -> Path:
    matches = sorted((REPO_ROOT / "configs").glob("stage_s3_{}_*yml".format(key)))
    if len(matches) != 1:
        raise RuntimeError("expected exactly one config for {}".format(key))
    return matches[0]


def logical_run_path(run_id: str, filename: str) -> str:
    """Return a redacted run path without formatting the placeholder."""
    return "${RUN_ROOT}/" + run_id + "/" + filename


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-code-sha", required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument(
        "--inventory-output", type=Path,
        default=(REPO_ROOT / "data_self/stage_s3_seg_raster/runtime/audits/"
                 "stage_s3_gpu_inventory.json"))
    parser.add_argument(
        "--schedule-output", type=Path,
        default=(REPO_ROOT / "data_self/stage_s3_seg_raster/runtime/audits/"
                 "stage_s3_gpu_schedule.json"))
    parser.add_argument("--sample-interval-seconds", type=float, default=10.0)
    args = parser.parse_args()

    assert_frozen_checkout(args.run_code_sha)
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS":
        raise RuntimeError("parallel training is forbidden until preflight PASS")
    budget = json.loads(args.budget.read_text(encoding="utf-8"))
    if budget.get("status") != "PASS" or budget.get("run_code_sha") != args.run_code_sha:
        raise RuntimeError("parallel training requires a frozen matching budget")
    required_free = int(preflight["memory_measurement"]["required_free_memory_mb"])
    env_floor = int(os.environ.get("S3_MIN_FREE_MEM_MB", "0") or 0)
    required_free = max(required_free, env_floor)
    max_parallel = max(1, int(os.environ.get("S3_MAX_PARALLEL", "6")))
    max_wait_minutes = max(0.0, float(os.environ.get("S3_MAX_WAIT_MINUTES", "30")))
    deadline = time.monotonic() + max_wait_minutes * 60
    inventory_rounds = []
    eligible = []
    while True:
        try:
            samples, apps = collect_three_samples(args.sample_interval_seconds)
            evaluated = evaluate_gpu_eligibility(
                samples, apps, required_free_mb=required_free,
                excluded_indices=_excluded_indices())
            inventory_rounds.append({
                "samples": samples,
                "compute_apps": apps,
                "eligibility": evaluated,
            })
            eligible = [item for item in evaluated if item["eligible"]]
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            inventory_rounds.append({
                "sampled_at": utc_now(), "query_error": str(error)})
            eligible = []
        inventory = {
            "stage": "seg_raster_stage_s3",
            "required_free_memory_mb": required_free,
            "rounds": inventory_rounds,
            "eligible_gpu_count": len(eligible),
            "status": "PASS" if eligible else "WAITING",
        }
        _write_json(args.inventory_output, inventory)
        if eligible or time.monotonic() >= deadline:
            break
        time.sleep(min(30.0, max(0.0, deadline - time.monotonic())))

    if not eligible:
        inventory["status"] = "BLOCKED_NO_ELIGIBLE_GPU"
        _write_json(args.inventory_output, inventory)
        _write_json(args.schedule_output, {
            "stage": "seg_raster_stage_s3",
            "status": "BLOCKED_NO_ELIGIBLE_GPU",
            "run_code_sha": args.run_code_sha,
            "jobs": [],
            "parallel_job_peak": 0,
            "policy": "no external process was terminated or preempted",
        })
        return 3

    # Use the largest homogeneous pool for directly comparable jobs.  This
    # may trade peak parallelism for causal fairness on heterogeneous hosts.
    by_model: dict[str, list[dict[str, object]]] = {}
    for item in eligible:
        by_model.setdefault(str(item["name"]), []).append(item)
    preferred_model, homogeneous_pool = max(
        by_model.items(), key=lambda pair: (len(pair[1]), pair[0]))
    selected = homogeneous_pool[:max_parallel]
    scheduler = FifoGpuScheduler(item["index"] for item in selected)
    gpu_by_index = {item["index"]: item for item in selected}
    queue = list(EXPERIMENT_MATRIX)
    running: dict[int, dict[str, object]] = {}
    jobs = []
    peak = 0
    run_root = REPO_ROOT / "data_self/stage_s3_seg_raster"
    while queue or running:
        while queue and scheduler.available:
            spec = queue.pop(0)
            gpu_index = scheduler.allocate(spec.run_id)
            assert gpu_index is not None
            run_dir = run_root / spec.run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = run_dir / "stdout.log"
            stderr_path = run_dir / "stderr.log"
            stdout_handle = stdout_path.open("w", encoding="utf-8")
            stderr_handle = stderr_path.open("w", encoding="utf-8")
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
            env["S3_RUN_CODE_SHA"] = args.run_code_sha
            env["S3_OPTIMIZER_STEPS"] = str(budget["optimizer_steps"])
            command = [
                sys.executable,
                str(REPO_ROOT / "tools/seg_raster/train_stage_s3.py"),
                "--config", str(_config_path(spec.key)),
                "--run-code-sha", args.run_code_sha,
            ]
            process = subprocess.Popen(
                command, cwd=str(REPO_ROOT), env=env,
                stdout=stdout_handle, stderr=stderr_handle)
            gpu = gpu_by_index[gpu_index]
            record = {
                "run_id": spec.run_id,
                "run_key": spec.key,
                "physical_index": gpu_index,
                "gpu_uuid": gpu["uuid"],
                "gpu_name": gpu["name"],
                "pid": process.pid,
                "start_time": utc_now(),
                "end_time": None,
                "exit_code": None,
                "peak_allocated_memory_mb": None,
                "peak_reserved_memory_mb": None,
                "stdout_path": logical_run_path(spec.run_id, "stdout.log"),
                "stderr_path": logical_run_path(spec.run_id, "stderr.log"),
            }
            jobs.append(record)
            running[gpu_index] = {
                "process": process, "spec": spec, "record": record,
                "stdout": stdout_handle, "stderr": stderr_handle}
        peak = max(peak, len(running))
        _write_json(args.schedule_output, {
            "stage": "seg_raster_stage_s3", "status": "RUNNING",
            "run_code_sha": args.run_code_sha, "jobs": jobs,
            "parallel_job_peak": peak})
        time.sleep(5.0)
        for gpu_index, state in list(running.items()):
            process = state["process"]
            exit_code = process.poll()
            if exit_code is None:
                continue
            state["stdout"].close()
            state["stderr"].close()
            record = state["record"]
            record["end_time"] = utc_now()
            record["exit_code"] = exit_code
            summary_path = run_root / state["spec"].run_id / "summary.json"
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                record["peak_allocated_memory_mb"] = summary.get("peak_allocated_memory_mb")
                record["peak_reserved_memory_mb"] = summary.get("peak_reserved_memory_mb")
            scheduler.release(gpu_index, state["spec"].run_id)
            del running[gpu_index]

    status = "PASS" if all(job["exit_code"] == 0 for job in jobs) else "FAIL"
    _write_json(args.schedule_output, {
        "stage": "seg_raster_stage_s3", "status": status,
        "run_code_sha": args.run_code_sha, "jobs": jobs,
        "parallel_job_peak": peak,
        "preferred_homogeneous_gpu_model": preferred_model})
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
