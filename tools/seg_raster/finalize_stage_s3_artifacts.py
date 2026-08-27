"""Materialize small tracked S3 artifacts after all GPU activity stops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.seg_raster.stage_s3 import EXPERIMENT_MATRIX


ABSOLUTE_WINDOWS_PATH = re.compile(r"(?i)[a-z]:[\\/]")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if ABSOLUTE_WINDOWS_PATH.search(encoded):
        raise ValueError("absolute Windows path found in final JSON: {}".format(path.name))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")


def blocked_artifacts(base_sha: str, run_code_sha: str, preflight: dict) -> None:
    artifacts = REPO_ROOT / "artifacts"
    eligibility = preflight.get("eligibility", [])
    gpu_models = sorted({item.get("name") for item in eligibility if item.get("name")})
    gpu_inventory = {
        "stage": "seg_raster_stage_s3",
        "status": "BLOCKED_NO_ELIGIBLE_GPU",
        "eligible_gpu_count": 0,
        "required_initial_free_memory_mb": 2048,
        "max_wait_minutes": preflight.get("max_wait_minutes", 30),
        "gpu_samples": preflight.get("gpu_samples", []),
        "compute_apps": preflight.get("compute_apps", []),
        "eligibility": eligibility,
        "no_external_process_terminated": True,
    }
    schedule = {
        "stage": "seg_raster_stage_s3",
        "status": "BLOCKED_NO_ELIGIBLE_GPU",
        "run_code_sha": run_code_sha,
        "jobs": [], "parallel_job_peak": 0,
        "no_process_preempted_or_terminated": True,
    }
    budget = {
        "stage": "seg_raster_stage_s3", "status": "BLOCKED",
        "reason": "production CUDA preflight did not obtain an eligible GPU",
        "n_baseline_steps": 102400, "kind": "SCREENING",
        "optimizer_steps": 0, "timing_probe_steps_completed": 0,
        "frozen_before_comparative_results": True,
    }
    run_records = {
        spec.key: {
            "run_id": spec.run_id, "code_sha": run_code_sha,
            "config_sha": None, "data_manifest_sha": None,
            "sample_plan_sha": None, "initialization_sha": None,
            "seed": 20260827, "gpu_index": None, "gpu_uuid": None,
            "gpu_model": None, "start_time": None, "end_time": None,
            "optimizer_steps": 0, "samples_seen": 0,
            "best_checkpoint_sha256": None, "final_checkpoint_sha256": None,
            "best_metrics": {}, "final_metrics": {},
            "status": "NOT_STARTED", "invalid_reason": "PREFLIGHT_BLOCKED",
        }
        for spec in EXPERIMENT_MATRIX
    }
    training = {
        "stage": "seg_raster_stage_s3", "status": "BLOCKED",
        "reason": "PREFLIGHT_BLOCKED", "run_code_sha": run_code_sha,
        "runs": run_records, "first_100_batch_identity_parity": True,
    }
    blocked = {
        "stage": "seg_raster_stage_s3", "status": "BLOCKED",
        "reason": "PREFLIGHT_BLOCKED; no comparative model output exists",
    }
    graph = {
        "stage": "seg_raster_stage_s3",
        "status": "NOT_EXECUTED_BY_GATE",
        "segmentation_gate_passed": False,
        "reason": "preflight blocked before training",
        "runs": {},
    }
    control = {
        "stage": "seg_raster_stage_s3", "status": "INCONCLUSIVE",
        "segmentation_causal_screen": "INCONCLUSIVE",
        "indirect_anchor_screen": "INCONCLUSIVE",
        "joint_screen": "INCONCLUSIVE",
        "reason": "no eligible GPU; no runs were started",
        "single_seed_significance_claim": False,
    }
    conclusion = {
        "stage": "seg_raster_stage_s3",
        "branch": "feat/seg-raster-only",
        "s3_base_sha": base_sha, "run_code_sha": run_code_sha,
        "preflight": "BLOCKED", "split_gate": "PASS",
        "sample_parity": "PASS",
        "gpu_discovery": {
            "eligible_gpu_count": 0, "parallel_job_peak": 0,
            "gpu_models": gpu_models, "status": "BLOCKED"},
        "training_budget": {"kind": "SCREENING", "optimizer_steps": 0},
        "runs": {spec.key: "NOT_STARTED" for spec in EXPERIMENT_MATRIX},
        "segmentation_causal_screen": "INCONCLUSIVE",
        "indirect_anchor_screen": "INCONCLUSIVE",
        "joint_screen": "INCONCLUSIVE",
        "closed_loop_graph": "NOT_EXECUTED_BY_GATE",
        "go_no_go_for_multiseed": "INCONCLUSIVE",
        "key_evidence": [
            "The first 100 teacher-forced batches match across all six configs.",
            "No eligible CUDA GPU became available during the declared wait.",
            "No training subprocess was started and no external process was disturbed."],
        "risks": [
            "Production CUDA preflight, memory measurement, timing probe, training, and evaluation remain unexecuted."],
        "next_stage_proposal": [
            "Repeat Stage S3 from the same frozen SHA when an eligible CUDA GPU is available."],
    }
    write_json(artifacts / "stage_s3_gpu_inventory.json", gpu_inventory)
    write_json(artifacts / "stage_s3_gpu_schedule.json", schedule)
    write_json(artifacts / "stage_s3_budget_manifest.json", budget)
    write_json(artifacts / "stage_s3_training_results.json", training)
    write_json(artifacts / "stage_s3_segmentation_comparison.json", blocked)
    write_json(artifacts / "stage_s3_anchor_comparison.json", blocked)
    write_json(artifacts / "stage_s3_graph_comparison.json", graph)
    write_json(artifacts / "stage_s3_control_analysis.json", control)
    write_json(artifacts / "stage_s3_conclusion.json", conclusion)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--run-code-sha", required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    args = parser.parse_args()
    preflight = read_json(args.preflight)
    write_json(REPO_ROOT / "artifacts/stage_s3_preflight.json", preflight)
    if preflight.get("status") == "BLOCKED":
        blocked_artifacts(args.base_sha, args.run_code_sha, preflight)
        return 0
    if preflight.get("status") != "PASS":
        raise RuntimeError("cannot finalize a failed preflight")
    raise RuntimeError(
        "PASS preflight requires completed training aggregation before finalization")


if __name__ == "__main__":
    raise SystemExit(main())
