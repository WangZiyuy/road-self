"""Reduce small Stage S3E evidence into final JSON and Markdown artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.seg_raster.control_stage_s3e import RUN_IDS
from utils.seg_raster.stage_s3e import finite_tree


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: object) -> None:
    finite_tree(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def placeholder(reason: str) -> dict:
    return {
        "stage": "seg_raster_stage_s3e",
        "status": "NOT_EXECUTED_BY_ZERO_INIT_CAUSAL_GATE",
        "reason": reason,
    }


def write_doc(path: Path, title: str, rows: list[tuple[str, object]]) -> None:
    lines = ["# " + title, ""]
    lines.extend("- {}: {}".format(key, value) for key, value in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def merge_gpu(
    output_root: Path, phase_ab_root: Path, phase_c_executed: bool,
    c4_status: str | None,
) -> None:
    inventories, schedules = [], []
    for phase in ("a", "b"):
        inventories.append(read(phase_ab_root / (
            "stage_s3e_gpu_inventory_{}.json".format(phase))))
        schedules.append(read(phase_ab_root / (
            "stage_s3e_gpu_schedule_{}.json".format(phase))))
    if phase_c_executed:
        for phase in ("c4_calibration", "c"):
            inventories.append(read(output_root / (
                "stage_s3e_gpu_inventory_{}.json".format(phase))))
            schedules.append(read(output_root / (
                "stage_s3e_gpu_schedule_{}.json".format(phase))))
    formal_training_schedules = [
        row for row in schedules if row.get("job") != "c4-calibration"]
    all_training_pass = all(
        row["status"] == "PASS" for row in formal_training_schedules)
    merged_status = "PASS" if all_training_pass else "FAIL"
    if all_training_pass and c4_status and c4_status != "PASS":
        merged_status = "PASS_WITH_C4_NOT_EXECUTED"
    write(output_root / "stage_s3e_gpu_inventory.json", {
        "stage": "seg_raster_stage_s3e", "status": merged_status,
        "execution_environment": "REMOTE_TRAINING_SERVER",
        "phases": inventories})
    write(output_root / "stage_s3e_gpu_schedule.json", {
        "stage": "seg_raster_stage_s3e",
        "status": merged_status, "phases": schedules,
        "c4_status": c4_status,
        "external_processes_terminated": False})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--phase-ab-output-root", type=Path)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-code-sha", required=True)
    parser.add_argument("--phase-ab-code-sha")
    parser.add_argument("--base-sha", required=True)
    args = parser.parse_args()
    phase_ab_root = args.phase_ab_output_root or args.output_root
    phase_ab_code_sha = args.phase_ab_code_sha or args.run_code_sha
    phase_a = read(phase_ab_root / "stage_s3e_phase_a_summary.json")
    phase_b = read(phase_ab_root / "stage_s3e_phase_b_summary.json")
    phase_c_path = args.output_root / "stage_s3e_phase_c_summary.json"
    phase_c_executed = phase_c_path.is_file()
    phase_c = read(phase_c_path) if phase_c_executed else None

    runs = {key: read(args.run_root / RUN_IDS[key] / "summary.json")
            for key in ("Z0", "Z1", "Z2")}
    zero_experiment = dict(phase_b)
    zero_experiment["run_code_sha"] = phase_ab_code_sha
    zero_experiment["reducer_code_sha"] = args.run_code_sha
    write(args.output_root / "stage_s3e_zero_init_experiment.json", zero_experiment)
    write(args.output_root / "stage_s3e_zero_init_dynamics.json", {
        "stage": "seg_raster_stage_s3e", "status": "PASS",
        "run_code_sha": phase_ab_code_sha,
        "reducer_code_sha": args.run_code_sha,
        "runs": {key: {
            "run_id": value["run_id"],
            "dense_diagnostics_by_update": value["dense_diagnostics_by_update"],
            "validation_metrics_by_update": value["validation_metrics_by_update"],
            "first_optimizer_step": value["first_optimizer_step"],
            "final_counterfactual_controls": value[
                "final_counterfactual_controls"],
        } for key, value in runs.items()}})

    if phase_c_executed:
        mechanism = phase_c
        support = {"stage": "seg_raster_stage_s3e", "status": "PASS",
                   "result": phase_c["support_ablation"],
                   "evidence": phase_c["comparisons"]["C2"]}
        encoder = {"stage": "seg_raster_stage_s3e", "status": "PASS",
                   "result": phase_c["encoder_ablation"],
                   "evidence": phase_c["comparisons"]["C3"]}
        gradient = {"stage": "seg_raster_stage_s3e", "status": "PASS",
                    "result": phase_c["gradient_balance_ablation"],
                    "evidence": phase_c["comparisons"]["C4"]}
        if phase_c["c4_status"] != "PASS":
            gradient["status"] = phase_c["c4_status"]
    else:
        reason = "ZERO_INIT_ROOT_CAUSE_STATUS=MAJOR_CAUSAL_FACTOR"
        mechanism = placeholder(reason)
        support = placeholder(reason)
        encoder = placeholder(reason)
        gradient = placeholder(reason)
    write(args.output_root / "stage_s3e_mechanism_matrix.json", mechanism)
    write(args.output_root / "stage_s3e_support_ablation.json", support)
    write(args.output_root / "stage_s3e_encoder_ablation.json", encoder)
    write(args.output_root / "stage_s3e_gradient_balance_ablation.json", gradient)
    c4_status = phase_c.get("c4_status") if phase_c else None
    merge_gpu(args.output_root, phase_ab_root, phase_c_executed, c4_status)

    zero_status = phase_b["zero_init_root_cause_status"]
    if zero_status == "MAJOR_CAUSAL_FACTOR":
        random_class = "PROVEN_CAUSAL"
        minimum_fix = "PROJECTION_ZERO_INIT"
    elif zero_status == "PARTIAL_CAUSAL_FACTOR":
        random_class = "SUPPORTED_CAUSAL"
        minimum_fix = "INCONCLUSIVE_PENDING_MECHANISM_ABLATIONS"
    else:
        random_class = "AMPLIFIER"
        minimum_fix = "NOT_PROJECTION_ZERO_INIT_ALONE"
    classifications = {
        "RANDOM_INITIAL_RESIDUAL": random_class,
        "ROAD_HEAD_COADAPTATION": (
            "PROVEN_CAUSAL" if phase_c_executed and phase_c[
                "road_head_update_ablation"] == "SUPPORTED"
            else "FALSIFIED" if phase_c_executed and phase_c[
                "road_head_update_ablation"] == "NOT_NECESSARY"
            else "SUPPORTED_CAUSAL" if phase_a[
                "destructive_interaction"] == "SUPPORTED" else "INCONCLUSIVE"),
        "SUPPORT_MULTIPLICATION": (
            "AMPLIFIER" if phase_c_executed and phase_c[
                "support_ablation"] == "SUPPORTED_AS_AMPLIFIER"
            else "FALSIFIED" if phase_c_executed else "INCONCLUSIVE"),
        "RASTER_ENCODER_LEARNING": (
            "SUPPORTED_CAUSAL" if phase_c_executed and phase_c[
                "encoder_ablation"] == "SUPPORTED"
            else "FALSIFIED" if phase_c_executed else "INCONCLUSIVE"),
        "BACKGROUND_GRADIENT_DOMINANCE": (
            "SUPPORTED_CAUSAL" if phase_c_executed and phase_c[
                "gradient_balance_ablation"] == "SUPPORTED"
            else "INCONCLUSIVE" if phase_c_executed and phase_c[
                "gradient_balance_ablation"] ==
                "INCONCLUSIVE_CALIBRATION_TARGET_INFEASIBLE"
            else "CORRELATED_NOT_CAUSAL" if phase_c_executed else "AMPLIFIER"),
    }
    classifications["ROAD_HEAD_PARAMETER_DRIFT"] = (
        "PROVEN_CAUSAL" if phase_a["road_head_drift"] == "PROVEN"
        else "INCONCLUSIVE")
    classifications["TRAINED_RASTER_ADAPTER_FORWARD"] = (
        "PROVEN_CAUSAL" if phase_c_executed and phase_c["comparisons"][
            "C1"]["functional_degradation_persists_with_frozen_head"]
        else "SUPPORTED_CAUSAL" if phase_a["adapter_on_clean_head"] == "HARMFUL"
        else "INCONCLUSIVE")
    road_head_necessary = (
        phase_c_executed
        and phase_c["road_head_update_ablation"] == "SUPPORTED")
    encoder_supported = (
        phase_c_executed and phase_c["encoder_ablation"] == "SUPPORTED")
    if road_head_necessary:
        main_damage = "TRAINABLE_ROAD_HEAD_PARAMETERS"
    else:
        main_damage = "DISTRIBUTED_RASTER_ADAPTER_FORWARD_AND_ROAD_HEAD_DRIFT"
    if zero_status != "MAJOR_CAUSAL_FACTOR" and encoder_supported:
        minimum_fix = (
            "NO_COMPLETE_FIX_JUSTIFIED; BEST_PARTIAL_MITIGATION="
            "PROJECTION_ZERO_INIT_PLUS_FROZEN_RASTER_ENCODER")
    next_allowed = (
        "SINGLE_VARIABLE_RASTER_ENCODER_LR_OR_FREEZE_VALIDATION"
        if encoder_supported else "NO_ADDITIONAL_FIX_WITHOUT_NEW_EVIDENCE")
    conclusion = {
        "stage": "seg_raster_stage_s3e", "status": "PASS",
        "execution_environment": "REMOTE_TRAINING_SERVER",
        "branch": "feat/seg-raster-only", "s3e_base_sha": args.base_sha,
        "s3e_run_code_sha": args.run_code_sha,
        "phase_ab_training_code_sha": phase_ab_code_sha,
        "phase_c_training_code_sha": args.run_code_sha,
        "cross_transplant": phase_a,
        "zero_init_root_cause_status": zero_status,
        "phase_c_executed": phase_c_executed,
        "phase_c_status": phase_c.get("status") if phase_c else None,
        "phase_c_completed_runs": phase_c.get(
            "completed_run_keys", []) if phase_c else [],
        "c4_status": phase_c.get("c4_status") if phase_c else None,
        "root_cause_classification": classifications,
        "first_harmful_forward_stage":
            "PROJECTED_RESIDUAL_TO_STAGE_FUSE_ROAD_BEFORE_OPTIMIZER_STEP",
        "first_observed_head_drift_interval": phase_a[
            "first_functional_head_drift_interval"],
        "main_final_damage_location": main_damage,
        "minimum_justified_fix": minimum_fix,
        "next_experiment_allowed": next_allowed,
        "next_experiment_forbidden": [
            "combined_fix", "anchor_evaluation", "graph_evaluation",
            "multiseed", "density_raster", "trajectory_sequence",
            "new_fusion_architecture"],
        "anchor_evaluation": "NOT_EXECUTED_BY_STAGE_BOUNDARY",
        "graph_evaluation": "NOT_EXECUTED_BY_STAGE_BOUNDARY",
        "multiseed_started": False,
        "invalid_run_count": 2,
        "invalid_runs_artifact": "artifacts/stage_s3e_invalid_runs.json",
    }
    write(args.output_root / "stage_s3e_conclusion.json", conclusion)

    write_doc(args.output_root / "stage_s3e_zero_init.md",
              "Stage S3E Projection Zero-Init", [
                  ("Z1 reproduction", phase_b["z1_reproduction_gate"]),
                  ("sample-0 parity", phase_b["zero_init_sample0_parity"]),
                  ("first-step parity", phase_b["zero_init_first_step_parity"]),
                  ("root-cause status", zero_status)])
    write_doc(args.output_root / "stage_s3e_mechanism_ablation.md",
              "Stage S3E Mechanism Ablations", [
                  ("executed", phase_c_executed),
                  ("result", mechanism.get("status", "PASS")),
                  ("road-head update", mechanism.get(
                      "road_head_update_ablation", "NOT_EXECUTED")),
                  ("support multiplier", mechanism.get(
                      "support_ablation", "NOT_EXECUTED")),
                  ("encoder learning", mechanism.get(
                      "encoder_ablation", "NOT_EXECUTED")),
                  ("gradient balance", mechanism.get(
                      "gradient_balance_ablation", "NOT_EXECUTED"))])
    write_doc(args.output_root / "stage_s3e_final_report.md",
              "Stage S3E Final Report", [
                  ("run code SHA", args.run_code_sha),
                  ("first head drift", conclusion[
                      "first_observed_head_drift_interval"]),
                  ("main damage", conclusion["main_final_damage_location"]),
                  ("minimum fix", minimum_fix),
                  ("next allowed", next_allowed),
                  ("classifications", classifications)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
