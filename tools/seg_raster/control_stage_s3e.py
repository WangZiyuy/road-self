"""Plans and reducers for the frozen Stage S3E causal matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.seg_raster.stage_s3 import sha256_file
from utils.seg_raster.stage_s3e import finite_tree


RUN_IDS = {
    "Z0": "Z0_null_seed20260827",
    "Z1": "Z1_default_aligned_seed20260827",
    "Z2": "Z2_zero_init_aligned_seed20260827",
    "C1": "C1_frozen_road_head_seed20260827",
    "C2": "C2_no_support_multiplier_seed20260827",
    "C3": "C3_frozen_encoder_seed20260827",
    "C4": "C4_gradient_balance_seed20260827",
}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: object) -> None:
    finite_tree(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def plan(phase: str, sha: str, output: Path, calibration: Path | None) -> None:
    keys = ("Z0", "Z1", "Z2") if phase == "B" else ("C1", "C2", "C3", "C4")
    calibration_record = None
    if phase == "C":
        if calibration is None:
            raise ValueError("Phase C requires the C4 calibration manifest")
        calibration_record = read(calibration)
        if calibration_record.get("status") != "PASS":
            raise RuntimeError("Phase C refuses a non-PASS C4 calibration")
        if calibration_record.get("run_code_sha") != sha:
            raise RuntimeError("Phase C calibration SHA does not match run SHA")
        if calibration_record.get("optimizer_steps_executed") != 0:
            raise RuntimeError("Phase C calibration must be read-only")
    jobs = []
    for key in keys:
        row = {"run_key": key, "run_id": RUN_IDS[key]}
        if key == "C4":
            row["calibration_manifest"] = str(calibration)
            row["calibration_manifest_sha256"] = sha256_file(calibration)
        jobs.append(row)
    write(output, {"stage": "seg_raster_stage_s3e", "phase": phase,
                   "status": "READY", "run_code_sha": sha,
                   "calibration_status": (
                       calibration_record["status"]
                       if calibration_record is not None else None),
                   "jobs": jobs})


def classify_adapter(row: dict) -> str:
    values = [row[name]["adapter_on_clean_head"]
              for name in ("road_f1", "road_iou", "road_auprc")]
    positives = sum(value > 0 for value in values)
    if positives == 3:
        return "BENEFICIAL"
    if positives == 0:
        return "HARMFUL"
    return "MIXED"


def reduce_a(root: Path, output: Path) -> None:
    cross = read(root / "stage_s3e_cross_transplant.json")
    drift = read(root / "stage_s3e_head_parameter_drift.json")
    gradients = read(root / "stage_s3e_gradient_field.json")
    if any(row.get("status") != "PASS" for row in (cross, drift, gradients)):
        raise RuntimeError("Phase A inputs are not PASS")
    samples = cross["common_sample_counts"]
    final = cross["checkpoints"][str(samples[-1])]["decomposition"]
    head_proven = (final["road_auprc"]["head_drift"] < -0.002
                   or final["road_f1"]["head_drift"] < -0.005)
    interaction_series = {
        str(sample): cross["checkpoints"][str(sample)]["decomposition"][
            "road_auprc"]["interaction"] for sample in samples}
    negative = [sample for sample in samples if interaction_series[str(sample)] < 0]
    result = {
        "stage": "seg_raster_stage_s3e", "phase": "A",
        "status": "PASS",
        "road_head_drift": "PROVEN" if head_proven else "NOT_PROVEN",
        "adapter_on_clean_head": classify_adapter(final),
        "destructive_interaction": (
            "SUPPORTED" if final["road_auprc"]["interaction"] < 0
            else "NOT_SUPPORTED"),
        "first_functional_head_drift_interval":
            drift["first_functional_head_drift_interval"],
        "first_negative_auprc_interaction_checkpoint": (
            negative[0] if negative else "NOT_OBSERVED"),
        "final_decomposition": final,
        "gradient_field": gradients["by_samples"],
    }
    write(output, result)


def summary(run_root: Path, key: str) -> dict:
    value = read(run_root / RUN_IDS[key] / "summary.json")
    if value.get("status") != "PASS":
        raise RuntimeError(key + " summary is not PASS")
    return value


def metric_delta(left: dict, right: dict) -> dict:
    names = ("road_precision", "road_recall", "road_f1", "road_iou", "road_auprc")
    return {name: float(right[name]) - float(left[name]) for name in names}


def close(left: float, right: float, atol: float = 1e-7) -> bool:
    return abs(float(left) - float(right)) <= atol


def reduce_b(
    run_root: Path, legacy_dynamics: Path, phase_a_cross: Path, output: Path,
) -> None:
    runs = {key: summary(run_root, key) for key in ("Z0", "Z1", "Z2")}
    legacy = read(legacy_dynamics)["runs"]["N1"]["validation_metrics_by_samples"]
    cross = read(phase_a_cross)
    reproduction = {}
    for update, sample in ((0, 0), (128, 2560)):
        current = runs["Z1"]["validation_metrics_by_update"][str(update)]
        expected = legacy[str(sample)]
        exact = cross["checkpoints"][str(sample)]["metrics"]["T11"]
        fields = ("road_f1", "road_iou", "road_auprc", "road_precision", "road_recall")
        reproduction[str(sample)] = {
            name: {"current": current[name], "expected": expected[name],
                   "equal": close(current[name], expected[name])}
            for name in fields}
        reproduction[str(sample)]["prediction_sha256"] = {
            "current": current["road_prediction_sha256"],
            "expected": exact["road_prediction_sha256"],
            "equal": current["road_prediction_sha256"]
                     == exact["road_prediction_sha256"],
        }
        reproduction[str(sample)]["residual_to_image_l2_ratio"] = {
            "current": current["residual_to_image_l2_ratio"],
            "expected": exact["residual_to_image_l2_ratio"],
            "equal": close(current["residual_to_image_l2_ratio"],
                           exact["residual_to_image_l2_ratio"]),
        }
    z1_reproduction = all(row["equal"] for point in reproduction.values()
                          for row in point.values())

    z0_0 = runs["Z0"]["validation_metrics_by_update"]["0"]
    z2_0 = runs["Z2"]["validation_metrics_by_update"]["0"]
    sample0_fields = ("road_f1", "road_iou", "road_auprc", "road_precision",
                      "road_recall", "gt_mean_probability",
                      "background_mean_probability")
    sample0_parity = all(close(z0_0[name], z2_0[name]) for name in sample0_fields)
    sample0_parity &= z0_0["road_prediction_sha256"] == z2_0["road_prediction_sha256"]
    sample0_parity &= abs(float(z2_0["residual_to_image_l2_ratio"])) <= 1e-12

    z0_step = runs["Z0"]["first_optimizer_step"]
    z2_step = runs["Z2"]["first_optimizer_step"]
    first_step_parity = (
        z0_step["loss_gradient"]["road_head"]["sha256"]
        == z2_step["loss_gradient"]["road_head"]["sha256"]
        and z0_step["road_head_optimizer_delta"]["after_sha256"]
        == z2_step["road_head_optimizer_delta"]["after_sha256"])
    projection_gradient_nonzero = not z2_step["loss_gradient"]["projection"]["all_zero"]
    encoder_loss_gradient_zero = z2_step["loss_gradient"]["encoder"]["all_zero"]

    final = {key: runs[key]["final_counterfactual_controls"] for key in runs}
    z0_null = final["Z0"]["null"]
    z1_null, z2_null = final["Z1"]["null"], final["Z2"]["null"]
    z1_aligned, z2_aligned = final["Z1"]["aligned"], final["Z2"]["aligned"]
    gaps, reductions = {}, {}
    for name in ("road_f1", "road_iou", "road_auprc"):
        gap1 = float(z1_null[name]) - float(z0_null[name])
        gap2 = float(z2_null[name]) - float(z0_null[name])
        gaps[name] = {"z1": gap1, "z2": gap2}
        reductions[name] = 1.0 - abs(gap2) / max(abs(gap1), 1e-30)
    auprc_reduction = reductions["road_auprc"]
    improved = sum(float(z2_aligned[name]) > float(z1_aligned[name])
                   for name in ("road_f1", "road_iou", "road_auprc"))
    if auprc_reduction >= 0.50 and improved >= 2:
        classification = "MAJOR_CAUSAL_FACTOR"
    elif auprc_reduction < 0.20:
        classification = "INITIAL_HARM_CONFIRMED_BUT_NOT_MAIN_FINAL_CAUSE"
    else:
        classification = "PARTIAL_CAUSAL_FACTOR"
    if not z1_reproduction:
        classification = "NOT_INTERPRETABLE_Z1_REPRODUCTION_FAILED"

    result = {
        "stage": "seg_raster_stage_s3e", "phase": "B",
        "status": "PASS" if z1_reproduction else "FAIL",
        "z1_reproduction_gate": "PASS" if z1_reproduction else "FAIL",
        "z1_reproduction": reproduction,
        "zero_init_sample0_parity": "PASS" if sample0_parity else "FAIL",
        "zero_init_first_step_parity": "PASS" if first_step_parity else "FAIL",
        "projection_loss_gradient_nonzero": projection_gradient_nonzero,
        "encoder_loss_gradient_zero_at_first_step": encoder_loss_gradient_zero,
        "head_drift_gap": gaps, "head_drift_gap_reduction": reductions,
        "z2_final_aligned_vs_z1": metric_delta(z1_aligned, z2_aligned),
        "z2_metrics_improved_count": improved,
        "zero_init_root_cause_status": classification,
        "phase_c_required": classification != "MAJOR_CAUSAL_FACTOR"
                            or improved < 2,
        "runs": {key: {"run_id": value["run_id"],
                       "status": value["status"],
                       "final_counterfactual_controls":
                           value["final_counterfactual_controls"]}
                 for key, value in runs.items()},
    }
    write(output, result)


def reduce_c(run_root: Path, phase_b: Path, output: Path) -> None:
    image = summary(run_root, "Z0")
    reference = summary(run_root, "Z2")
    runs = {key: summary(run_root, key) for key in ("C1", "C2", "C3", "C4")}
    ref_aligned = reference["final_counterfactual_controls"]["aligned"]
    image_null = image["final_counterfactual_controls"]["null"]
    ref_null = reference["final_counterfactual_controls"]["null"]
    comparisons = {}
    for key, value in runs.items():
        aligned = value["final_counterfactual_controls"]["aligned"]
        comparisons[key] = {
            "metrics_vs_z2": metric_delta(ref_aligned, aligned),
            "final_null": value["final_counterfactual_controls"]["null"],
            "final_aligned": aligned,
            "road_head_checksum_unchanged": value.get(
                "road_head_checksum_unchanged"),
            "encoder_checksum_unchanged": value.get("encoder_checksum_unchanged"),
            "run_profile": value["run_profile"],
        }
        gap_ref = float(ref_null["road_auprc"]) - float(image_null["road_auprc"])
        gap_run = float(value["final_counterfactual_controls"]["null"][
            "road_auprc"]) - float(image_null["road_auprc"])
        comparisons[key]["head_drift_auprc_gap"] = gap_run
        comparisons[key]["head_drift_auprc_gap_reduction_vs_z2"] = (
            1.0 - abs(gap_run) / max(abs(gap_ref), 1e-30))
    c1 = comparisons["C1"]
    c1_adapter_gain = (
        float(c1["final_aligned"]["road_auprc"])
        - float(c1["final_null"]["road_auprc"]))
    road_head_result = (
        "SUPPORTED" if c1["road_head_checksum_unchanged"]
        and c1_adapter_gain > 0 else "NOT_NECESSARY")
    c2 = comparisons["C2"]
    support_result = (
        "SUPPORTED_AS_AMPLIFIER"
        if c2["head_drift_auprc_gap_reduction_vs_z2"] >= 0.20
        and c2["metrics_vs_z2"]["road_auprc"] >= 0.002
        else "NOT_PRIMARY")
    c3 = comparisons["C3"]
    encoder_result = (
        "SUPPORTED"
        if c3["encoder_checksum_unchanged"]
        and c3["head_drift_auprc_gap_reduction_vs_z2"] >= 0.20
        and c3["metrics_vs_z2"]["road_auprc"] >= 0.002
        else "NOT_NECESSARY")
    c4 = comparisons["C4"]
    calibrated_ratio = float(runs["C4"]["run_profile"].get(
        "calibrated_initial_gradient_ratio", 1.0))
    gradient_result = (
        "SUPPORTED"
        if abs(calibrated_ratio - 1.0) <= 0.20
        and c4["head_drift_auprc_gap_reduction_vs_z2"] >= 0.20
        and c4["metrics_vs_z2"]["road_recall"] >= 0.01
        else "CORRELATED_NOT_CAUSAL")
    result = {
        "stage": "seg_raster_stage_s3e", "phase": "C", "status": "PASS",
        "phase_b_zero_init_status": read(phase_b)["zero_init_root_cause_status"],
        "comparisons": comparisons,
        "road_head_update_ablation": road_head_result,
        "support_ablation": support_result,
        "encoder_ablation": encoder_result,
        "gradient_balance_ablation": gradient_result,
    }
    write(output, result)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan-b", "plan-c"):
        p = sub.add_parser(name)
        p.add_argument("--run-code-sha", required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--calibration", type=Path)
    p = sub.add_parser("reduce-a")
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("reduce-b")
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--legacy-dynamics", type=Path, required=True)
    p.add_argument("--phase-a-cross", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("reduce-c")
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--phase-b", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan-b":
        plan("B", args.run_code_sha, args.output, None)
    elif args.command == "plan-c":
        plan("C", args.run_code_sha, args.output, args.calibration)
    elif args.command == "reduce-a":
        reduce_a(args.input_root, args.output)
    elif args.command == "reduce-b":
        reduce_b(args.run_root, args.legacy_dynamics, args.phase_a_cross,
                 args.output)
    else:
        reduce_c(args.run_root, args.phase_b, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
