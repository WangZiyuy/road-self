"""FIFO launcher for Stage S3C baseline or conditional control graphs."""

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
        raise RuntimeError("Stage S3C graph requires a clean frozen checkout")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-code-sha", required=True)
    parser.add_argument("--graph-plan", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--required-free-memory-mb", type=int, default=12000)
    args = parser.parse_args()
    assert_frozen(args.run_code_sha)
    plan = json.loads(args.graph_plan.read_text(encoding="utf-8"))
    stage_s3d = plan.get("stage") == "seg_raster_stage_s3d"
    stage_label = "seg_raster_stage_s3d" if stage_s3d else "seg_raster_stage_s3c"
    if plan.get("run_code_sha") != args.run_code_sha:
        raise RuntimeError("graph plan code SHA mismatch")
    samples, apps = collect_inventory(7.0)
    eligibility_policy = gpu_eligibility_overrides_from_environment()
    eligibility = evaluate_gpu_eligibility(
        samples, apps, required_free_mb=args.required_free_memory_mb,
        excluded_indices=excluded_indices(), **eligibility_policy)
    eligible = [row for row in eligibility if row["eligible"]]
    phase = str(plan.get("phase", "GRAPH"))
    write_json(args.output_root / ("gpu_inventory_phase_" + phase + ".json"), {
        "stage": stage_label, "phase": phase,
        "execution_environment": "REMOTE_TRAINING_SERVER",
        "remote_host_label": "exp-237-tunnel",
        "run_code_sha": args.run_code_sha,
        "required_free_memory_mb": args.required_free_memory_mb,
        "s3_exclude_gpus": os.environ.get("S3_EXCLUDE_GPUS", ""),
        "samples": samples, "compute_apps": apps,
        "eligibility_policy": eligibility_policy,
        "eligibility": eligibility, "eligible_gpu_count": len(eligible),
        "status": "PASS" if eligible else "BLOCKED_NO_ELIGIBLE_GPU"})
    schedule_path = args.output_root / ("gpu_schedule_phase_" + phase + ".json")
    if not eligible:
        return 3
    by_model: dict[str, list[dict]] = {}
    for row in eligible:
        by_model.setdefault(str(row["name"]), []).append(row)
    preferred_model, pool = max(
        by_model.items(), key=lambda pair: (len(pair[1]), pair[0]))
    pool = pool[:max(1, min(len(pool), int(os.environ.get("S3_MAX_PARALLEL", "4"))))]
    scheduler = FifoGpuScheduler(row["index"] for row in pool)
    gpu_by_index = {row["index"]: row for row in pool}
    queue = list(plan["runs"].items())
    running, records, peak = {}, [], 0
    while queue or running:
        while queue and scheduler.available:
            key, spec = queue.pop(0)
            gpu_index = scheduler.allocate(key)
            kind = spec.get("checkpoint_kind", "stage_s3c")
            source_run_id = spec.get("source_run_id", spec.get("run_id"))
            checkpoint = (Path(spec["checkpoint"])
                          if kind == "official_release" else
                          args.run_root / source_run_id / "checkpoints"
                          / spec["checkpoint"])
            output_dir = args.output_root / "graph" / key
            output_dir.mkdir(parents=True, exist_ok=True)
            stdout_handle = (output_dir / "stdout.log").open("w", encoding="utf-8")
            stderr_handle = (output_dir / "stderr.log").open("w", encoding="utf-8")
            result_path = args.output_root / "graph_results" / (key + ".json")
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
            command = [
                sys.executable,
                str(REPO_ROOT / "tools/seg_raster/evaluate_stage_s3c_graph.py"),
                "--run-code-sha", args.run_code_sha, "--run-key", key,
                "--checkpoint", str(checkpoint), "--checkpoint-kind", kind,
                "--control-root", str(args.control_root),
                "--threshold", str(plan.get("fixed_threshold", 0.3)),
                "--output-dir", str(output_dir), "--result", str(result_path),
                "--physical-gpu", str(gpu_index)]
            process = subprocess.Popen(
                command, cwd=REPO_ROOT, env=environment,
                stdout=stdout_handle, stderr=stderr_handle)
            gpu = gpu_by_index[gpu_index]
            record = {
                "run_key": key, "physical_index": gpu_index,
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
                "result_path": "${S3C_REMOTE_OUTPUT}/graph_results/" + key + ".json"}
            records.append(record)
            running[gpu_index] = {
                "process": process, "key": key, "record": record,
                "stdout": stdout_handle, "stderr": stderr_handle}
        peak = max(peak, len(running))
        write_json(schedule_path, {
            "stage": stage_label, "phase": phase,
            "status": "RUNNING", "run_code_sha": args.run_code_sha,
            "jobs": records,
            "parallel_job_peak": peak,
            "preferred_homogeneous_gpu_model": preferred_model})
        time.sleep(5)
        if running:
            try:
                contention_sample, contention_apps = inventory_sample()
                for state in running.values():
                    update_post_launch_contention(
                        state["record"], contention_sample, contention_apps,
                        own_pid=state["process"].pid,
                        required_free_memory_mb=args.required_free_memory_mb,
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
            code = state["process"].poll()
            if code is None:
                continue
            state["stdout"].close(); state["stderr"].close()
            state["record"]["exit_code"] = code
            state["record"]["end_time"] = utc_now()
            result_path = args.output_root / "graph_results" / (
                state["key"] + ".json")
            if result_path.is_file():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                state["record"]["peak_allocated_memory_mb"] = result.get(
                    "peak_gpu_memory_allocated_mb")
                state["record"]["peak_reserved_memory_mb"] = result.get(
                    "peak_gpu_memory_reserved_mb")
            scheduler.release(gpu_index, state["key"])
            del running[gpu_index]
    status = "PASS" if all(row["exit_code"] == 0 for row in records) else "FAIL"
    write_json(schedule_path, {
        "stage": stage_label, "phase": phase,
        "status": status, "run_code_sha": args.run_code_sha,
        "jobs": records, "parallel_job_peak": peak,
        "preferred_homogeneous_gpu_model": preferred_model,
        "external_processes_terminated": False})
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
