"""Aggregate frozen S3 run outputs and apply the predeclared causal gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.seg_raster.stage_s3 import (
    EXPERIMENT_MATRIX,
    indirect_anchor_screen,
    segmentation_causal_screen,
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")


def metric_deltas(candidate: dict, baseline: dict) -> dict:
    return {
        key: float(candidate[key]) - float(baseline[key])
        for key in candidate.keys() & baseline.keys()
        if isinstance(candidate[key], (int, float))
        and isinstance(baseline[key], (int, float))
        and not key.endswith("count")
    }


def joint_screen(segmentation: dict, anchors: dict) -> dict:
    if "J0" not in segmentation or "J1" not in segmentation:
        return {"status": "INCONCLUSIVE", "reason": "missing_required_run"}
    seg_names = ("road_f1", "road_iou", "junction_f1")
    seg_improvements = sum(
        segmentation["J1"][name] > segmentation["J0"][name]
        for name in seg_names)
    anchor = indirect_anchor_screen(anchors.get("J0"), anchors.get("J1"))
    if seg_improvements >= 2 or anchor["status"] == "PROMISING":
        status = "PROMISING"
    elif seg_improvements == 0 and anchor["status"] == "REGRESSION":
        status = "REGRESSION"
    else:
        status = "INCONCLUSIVE"
    return {
        "status": status, "segmentation_improved_metric_count": seg_improvements,
        "anchor_screen": anchor,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-code-sha", required=True)
    parser.add_argument(
        "--run-root", type=Path,
        default=REPO_ROOT / "data_self/stage_s3_seg_raster")
    parser.add_argument(
        "--schedule", type=Path,
        default=(REPO_ROOT / "data_self/stage_s3_seg_raster/runtime/audits/"
                 "stage_s3_gpu_schedule.json"))
    args = parser.parse_args()
    summaries, segmentation, anchors = {}, {}, {}
    schedule = read_json(args.schedule) if args.schedule.is_file() else {"jobs": []}
    assignments = {job["run_key"]: job for job in schedule.get("jobs", [])}
    missing = []
    for spec in EXPERIMENT_MATRIX:
        run_dir = args.run_root / spec.run_id
        paths = {
            "summary": run_dir / "summary.json",
            "segmentation": run_dir / "evaluation/segmentation.json",
            "anchor": run_dir / "evaluation/anchor.json",
        }
        absent = [name for name, path in paths.items() if not path.is_file()]
        if absent:
            missing.append({"run_key": spec.key, "missing": absent})
            continue
        summary = read_json(paths["summary"])
        assignment = assignments.get(spec.key, {})
        summary.update({
            "gpu_index": assignment.get("physical_index"),
            "gpu_uuid": assignment.get("gpu_uuid"),
            "gpu_model": assignment.get("gpu_name"),
            "process_start_time": assignment.get("start_time"),
            "process_end_time": assignment.get("end_time"),
            "process_exit_code": assignment.get("exit_code"),
        })
        summaries[spec.key] = summary
        segmentation[spec.key] = read_json(paths["segmentation"])
        anchors[spec.key] = read_json(paths["anchor"])
    complete = not missing and all(
        summary.get("status") == "PASS"
        and summary.get("code_sha") == args.run_code_sha
        for summary in summaries.values())
    sample_shas = {
        summary.get("first_100_batch_identity_sha256")
        for summary in summaries.values()}
    sample_parity = complete and len(sample_shas) == 1

    training_results = {
        "stage": "seg_raster_stage_s3",
        "status": "PASS" if complete and sample_parity else (
            "BLOCKED" if missing else "FAIL"),
        "run_code_sha": args.run_code_sha,
        "runs": summaries, "missing_outputs": missing,
        "first_100_batch_identity_parity": sample_parity,
    }
    write_json(REPO_ROOT / "artifacts/stage_s3_training_results.json", training_results)
    if not complete:
        placeholder = {
            "stage": "seg_raster_stage_s3", "status": "BLOCKED",
            "reason": "six valid run outputs are required", "missing": missing}
        for name in (
                "stage_s3_segmentation_comparison.json",
                "stage_s3_anchor_comparison.json",
                "stage_s3_graph_comparison.json",
                "stage_s3_control_analysis.json"):
            write_json(REPO_ROOT / "artifacts" / name, placeholder)
        return 3

    segmentation_comparison = {
        "stage": "seg_raster_stage_s3", "status": "PASS",
        "fixed_threshold_shared": True, "runs": segmentation,
        "deltas": {
            "C1_minus_C0": metric_deltas(segmentation["C1"], segmentation["C0"]),
            "C2_minus_C0": metric_deltas(segmentation["C2"], segmentation["C0"]),
            "C3_minus_C0": metric_deltas(segmentation["C3"], segmentation["C0"]),
            "J1_minus_J0": metric_deltas(segmentation["J1"], segmentation["J0"]),
        },
    }
    anchor_comparison = {
        "stage": "seg_raster_stage_s3", "status": "PASS", "runs": anchors,
        "C1_vs_C0_screen": indirect_anchor_screen(anchors["C0"], anchors["C1"]),
        "J1_vs_J0_screen": indirect_anchor_screen(anchors["J0"], anchors["J1"]),
    }
    causal = segmentation_causal_screen(segmentation)
    joint = joint_screen(segmentation, anchors)
    graph_gate = (
        causal["status"] == "PROMISING" or joint["status"] == "PROMISING")
    graph_runs = {}
    if graph_gate:
        for key in ("C0", "C1", "J0", "J1"):
            spec = next(item for item in EXPERIMENT_MATRIX if item.key == key)
            path = args.run_root / spec.run_id / "evaluation/graph.json"
            if path.is_file():
                graph_runs[key] = read_json(path)
        graph_status = "PASS" if len(graph_runs) == 4 else "FAIL"
    else:
        graph_status = "NOT_EXECUTED_BY_GATE"
    graph_comparison = {
        "stage": "seg_raster_stage_s3", "status": graph_status,
        "segmentation_gate_passed": graph_gate, "runs": graph_runs,
    }
    control = {
        "stage": "seg_raster_stage_s3", "status": "PASS",
        "segmentation_causal_screen": causal,
        "indirect_anchor_screen": anchor_comparison["C1_vs_C0_screen"],
        "joint_screen": joint,
        "interpretation_constraint": (
            "A J1-only improvement is joint optimization evidence and is not "
            "evidence of an indirect segmentation-to-anchor causal effect."),
    }
    write_json(REPO_ROOT / "artifacts/stage_s3_segmentation_comparison.json", segmentation_comparison)
    write_json(REPO_ROOT / "artifacts/stage_s3_anchor_comparison.json", anchor_comparison)
    write_json(REPO_ROOT / "artifacts/stage_s3_graph_comparison.json", graph_comparison)
    write_json(REPO_ROOT / "artifacts/stage_s3_control_analysis.json", control)
    go = "GO" if (
        causal["status"] == "PROMISING" or joint["status"] == "PROMISING") else "NO_GO"
    conclusion = {
        "stage": "seg_raster_stage_s3", "branch": "feat/seg-raster-only",
        "run_code_sha": args.run_code_sha,
        "preflight": "PASS", "split_gate": "PASS",
        "sample_parity": "PASS" if sample_parity else "FAIL",
        "runs": {key: summaries[key]["status"] for key in summaries},
        "segmentation_causal_screen": causal["status"],
        "indirect_anchor_screen": anchor_comparison["C1_vs_C0_screen"]["status"],
        "joint_screen": joint["status"],
        "closed_loop_graph": graph_status,
        "go_no_go_for_multiseed": go,
        "key_evidence": [
            "single fixed seed only; no statistical significance claim",
            "raw raster direct-to-anchor path remains forbidden by isolation tests"],
        "risks": ["single-seed screening"],
        "next_stage_proposal": [
            "Run a separately authorized three-seed replication only if GO."],
    }
    write_json(REPO_ROOT / "artifacts/stage_s3_conclusion.json", conclusion)
    return 0 if graph_status != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
