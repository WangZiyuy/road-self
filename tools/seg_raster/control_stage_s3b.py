"""Create Stage S3B phase plans and reduce frozen run summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from utils.seg_raster.stage_s3b import (
    LOSS_BALANCED, LOSS_BALANCED_DICE, LOSS_LEGACY, best_step_by_repair,
    repair_composite, retention, select_junction_loss, select_learning_rate,
    simulate_early_stop)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")


def run_id(key: str) -> str:
    return key + "_seed20260827"


def job(key: str, phase: str, input_kind: str, lr: float, loss: str,
        *, control: str | None = None, pos_weight: float = 1.0,
        alpha: float = 1.0) -> dict:
    return {"run_key": key, "run_id": run_id(key), "phase": phase,
            "input_kind": input_kind, "control": control,
            "lr_multiplier": lr, "loss_kind": loss,
            "pos_weight": pos_weight, "loss_alpha": alpha}


def summary(run_root: Path, run_id_value: str, code_sha: str) -> dict:
    path = run_root / run_id_value / "summary.json"
    value = read_json(path)
    if value.get("status") != "PASS" or value.get("code_sha") != code_sha:
        raise RuntimeError("invalid run summary: " + run_id_value)
    expected = {str(step) for step in range(0, 20481, 2560)}
    if set(value["validation_metrics_by_step"]) != expected:
        raise RuntimeError("incomplete common checkpoint grid: " + run_id_value)
    if len(value.get("checkpoint_inventory", [])) != 9:
        raise RuntimeError("incomplete versioned checkpoint inventory")
    return value


def candidate(key: str, value: dict, input_kind: str) -> dict:
    validation = value["validation_metrics_by_step"]
    best_step = best_step_by_repair(validation)
    best = validation[str(best_step)]
    return {
        "run_key": key, "input_kind": input_kind, "status": value["status"],
        "finite": value.get("invalid_reason") is None,
        "lr_multiplier": value.get("validation_metrics_by_step", {})
                         and read_lr(value),
        "base_lr": 0.0001 * read_lr(value),
        "loss_kind": value.get("run_id", "").split("_")[0],
        "best_step": best_step, "best_repair_composite": repair_composite(best),
        "retention": retention(validation), "best_metrics": best,
    }


def read_lr(value: dict) -> float:
    first = value["validation_metrics_by_step"]["0"]
    return float(first["optimizer_learning_rate"]) / 0.0001


def plan_a(code_sha: str) -> dict:
    jobs = []
    for index, lr in enumerate((1.0, 0.3, 0.1)):
        jobs.extend([
            job("A{}".format(index * 2), "A", "image_only", lr, LOSS_LEGACY),
            job("A{}".format(index * 2 + 1), "A", "raster", lr, LOSS_LEGACY,
                control="aligned")])
    return {"stage": "seg_raster_stage_s3b", "phase": "A",
            "run_code_sha": code_sha, "selection_uses": ["A0", "A2", "A4"],
            "jobs": jobs}


def reduce_a(args) -> int:
    rows = {key: summary(args.run_root, run_id(key), args.run_code_sha)
            for key in ("A0", "A1", "A2", "A3", "A4", "A5")}
    controls = []
    for key in ("A0", "A2", "A4"):
        row = candidate(key, rows[key], "image_only")
        row["loss_kind"] = LOSS_LEGACY
        controls.append(row)
    selection = select_learning_rate(controls)
    payload = {"stage": "seg_raster_stage_s3b", "status": "PASS",
               "run_code_sha": args.run_code_sha, "image_only_candidates": controls,
               "aligned_runs_excluded_from_selection": ["A1", "A3", "A5"],
               "selection": selection}
    write_json(args.output_root / "stage_s3b_lr_screen.json", payload)
    selected_key = selection.get("selected_run")
    if selected_key:
        write_json(args.output_root / "stage_s3b_early_stop_simulation.json",
                   simulate_early_stop(rows[selected_key]["validation_metrics_by_step"]))
    if not selection.get("phase_b_allowed"):
        return 3
    lr = float(selection["selected_lr_multiplier"])
    pair_index = {1.0: 0, 0.3: 2, 0.1: 4}[lr]
    audit = read_json(args.output_root / "stage_s3b_loss_gradient_audit.json")
    pos_weight = audit["junction_pos_weight"]["capped_pos_weight"]
    alpha_l1 = audit["junction_gradient_matching"][LOSS_BALANCED]["alpha"]
    alpha_l2 = audit["junction_gradient_matching"][LOSS_BALANCED_DICE]["alpha"]
    plan = {"stage": "seg_raster_stage_s3b", "phase": "B",
            "run_code_sha": args.run_code_sha,
            "selected_lr_multiplier": lr,
            "reused_runs": {
                "B0": run_id("A{}".format(pair_index)),
                "B1": run_id("A{}".format(pair_index + 1))},
            "jobs": [
                job("B2", "B", "image_only", lr, LOSS_BALANCED,
                    pos_weight=pos_weight, alpha=alpha_l1),
                job("B3", "B", "raster", lr, LOSS_BALANCED, control="aligned",
                    pos_weight=pos_weight, alpha=alpha_l1),
                job("B4", "B", "image_only", lr, LOSS_BALANCED_DICE,
                    pos_weight=pos_weight, alpha=alpha_l2),
                job("B5", "B", "raster", lr, LOSS_BALANCED_DICE,
                    control="aligned", pos_weight=pos_weight, alpha=alpha_l2)]}
    write_json(args.output_root / "phase_b_plan.json", plan)
    return 0


def curve_f1(metrics: dict, threshold: float) -> float:
    return min(metrics["junction_threshold_curve"],
               key=lambda row: abs(float(row["threshold"]) - threshold))["f1"]


def reduce_b(args) -> int:
    plan = read_json(args.output_root / "phase_b_plan.json")
    lr_screen = read_json(args.output_root / "stage_s3b_lr_screen.json")
    lr = float(lr_screen["selection"]["selected_lr_multiplier"])
    values = {
        "B0": summary(args.run_root, plan["reused_runs"]["B0"], args.run_code_sha),
        "B1": summary(args.run_root, plan["reused_runs"]["B1"], args.run_code_sha),
        "B2": summary(args.run_root, run_id("B2"), args.run_code_sha),
        "B3": summary(args.run_root, run_id("B3"), args.run_code_sha),
        "B4": summary(args.run_root, run_id("B4"), args.run_code_sha),
        "B5": summary(args.run_root, run_id("B5"), args.run_code_sha)}
    kinds = {"B0": LOSS_LEGACY, "B2": LOSS_BALANCED,
             "B4": LOSS_BALANCED_DICE}
    candidates = []
    for key, kind in kinds.items():
        row = candidate(key, values[key], "image_only")
        row["loss_kind"] = kind
        row["best_metrics"] = dict(row["best_metrics"])
        row["best_metrics"]["junction_shared_f1"] = row["best_metrics"]["junction_f1"]
        row["gradient_explosion"] = False
        candidates.append(row)
    selection = select_junction_loss(candidates)
    selected_loss = selection["selected_loss_kind"]
    selected_image = {LOSS_LEGACY: "B0", LOSS_BALANCED: "B2",
                      LOSS_BALANCED_DICE: "B4"}[selected_loss]
    selected_raster = {LOSS_LEGACY: "B1", LOSS_BALANCED: "B3",
                       LOSS_BALANCED_DICE: "B5"}[selected_loss]
    image_summary = values[selected_image]
    best_step = best_step_by_repair(image_summary["validation_metrics_by_step"])
    best_metrics = image_summary["validation_metrics_by_step"][str(best_step)]
    selected_curve = max(best_metrics["junction_threshold_curve"],
                         key=lambda row: (row["f1"], -row["threshold"]))
    shared_threshold = {
        "stage": "seg_raster_stage_s3b", "status": "PASS",
        "selection_run": selected_image, "selection_step": best_step,
        "selection_scope": "image_only_calibration_subset",
        "fixed_0_3_continues_to_be_reported": True,
        "selected_threshold": selected_curve["threshold"],
        "selected_threshold_metrics": selected_curve,
        "frozen_for_controls": ["R0", "R1", "R2", "R3"],
        "per_run_threshold_tuning_allowed": False}
    write_json(args.output_root / "stage_s3b_shared_threshold.json", shared_threshold)
    write_json(args.output_root / "stage_s3b_junction_loss_screen.json", {
        "stage": "seg_raster_stage_s3b", "status": "PASS",
        "run_code_sha": args.run_code_sha, "selected_lr_multiplier": lr,
        "image_only_candidates": candidates,
        "aligned_runs_excluded_from_selection": ["B1", "B3", "B5"],
        "selection": selection})
    selected_pair = {
        "R0": image_summary["run_id"], "R1": values[selected_raster]["run_id"]}
    loss_job = next((row for row in plan["jobs"] if row["run_key"] == selected_image), None)
    if selected_loss == LOSS_LEGACY:
        pos_weight, alpha = 1.0, 1.0
    else:
        pos_weight, alpha = loss_job["pos_weight"], loss_job["loss_alpha"]
    phase_c = {"stage": "seg_raster_stage_s3b", "phase": "C",
               "run_code_sha": args.run_code_sha, "selected_lr_multiplier": lr,
               "selected_loss_kind": selected_loss,
               "shared_threshold": shared_threshold["selected_threshold"],
               "reused_runs": selected_pair,
               "jobs": [
                   job("R2", "C", "raster", lr, selected_loss, control="zero",
                       pos_weight=pos_weight, alpha=alpha),
                   job("R3", "C", "raster", lr, selected_loss,
                       control="shift_fixed", pos_weight=pos_weight, alpha=alpha)]}
    write_json(args.output_root / "phase_c_plan.json", phase_c)
    return 0


def metric_at(summary_value: dict, step: int, shared_threshold: float) -> dict:
    metrics = dict(summary_value["validation_metrics_by_step"][str(step)])
    metrics["junction_shared_threshold"] = shared_threshold
    metrics["junction_shared_f1"] = curve_f1(metrics, shared_threshold)
    return metrics


def sensitivity(r0: dict, r1: dict) -> dict:
    left = r0["per_sample"]
    right = r1["per_sample"]
    if len(left) != len(right) or len(left) < 2:
        return {"status": "INCONCLUSIVE"}
    values = np.asarray([
        np.mean([right[index][name] - left[index][name]
                 for name in ("road_f1", "road_iou", "road_auprc")])
        for index in range(len(left))], dtype=np.float64)
    loo = [float(np.mean(np.delete(values, index))) for index in range(len(values))]
    rng = np.random.default_rng(20260827)
    bootstrap = [float(np.mean(values[rng.integers(0, len(values), len(values))]))
                 for _ in range(1000)]
    return {"status": "PASS" if min(loo) > 0 and np.percentile(bootstrap, 2.5) > 0
            else "FAIL", "leave_one_out_min_delta": min(loo),
            "bootstrap_95_percentile_interval": [float(np.percentile(bootstrap, 2.5)),
                                                  float(np.percentile(bootstrap, 97.5))],
            "sample_count": len(values)}


def reduce_c(args) -> int:
    plan = read_json(args.output_root / "phase_c_plan.json")
    values = {
        "R0": summary(args.run_root, plan["reused_runs"]["R0"], args.run_code_sha),
        "R1": summary(args.run_root, plan["reused_runs"]["R1"], args.run_code_sha),
        "R2": summary(args.run_root, run_id("R2"), args.run_code_sha),
        "R3": summary(args.run_root, run_id("R3"), args.run_code_sha)}
    shared_threshold = float(plan["shared_threshold"])
    common_step = best_step_by_repair(values["R0"]["validation_metrics_by_step"])
    rows = {key: metric_at(value, common_step, shared_threshold)
            for key, value in values.items()}
    names = ("road_f1", "road_iou", "road_auprc")
    deltas = {key: {name: rows[key][name] - rows["R0"][name] for name in names}
              for key in ("R1", "R2", "R3")}
    means = {key: float(np.mean(list(value.values()))) for key, value in deltas.items()}
    improved = sum(value > 0 for value in deltas["R1"].values())
    collapse = any(rows["R1"][name] < rows["R0"][name] - 0.10
                   for name in ("road_precision", "road_recall",
                                "junction_precision", "junction_recall"))
    steps = sorted(int(step) for step in values["R0"]["validation_metrics_by_step"])
    center = steps.index(common_step)
    start = max(0, min(center - 1, len(steps) - 3))
    trend_steps = steps[start:start + 3]
    trend = []
    for step in trend_steps:
        left = values["R0"]["validation_metrics_by_step"][str(step)]
        right = values["R1"]["validation_metrics_by_step"][str(step)]
        signal = float(np.mean([right[name] - left[name] for name in names]))
        trend.append({"step": step, "road_repair_signal": signal})
    robust = sensitivity(rows["R0"], rows["R1"])
    gate_pass = (improved >= 2 and means["R1"] > max(means["R2"], means["R3"])
                 and rows["R1"]["junction_auprc"] >= rows["R0"]["junction_auprc"]
                 and not collapse and len(trend) == 3
                 and all(row["road_repair_signal"] > 0 for row in trend)
                 and robust["status"] == "PASS")
    comparison = {
        "stage": "seg_raster_stage_s3b", "status": "PASS",
        "baseline_controlled_common_step": common_step,
        "common_step_selected_from": "R0_image_only_repair_composite_only",
        "shared_junction_threshold": shared_threshold, "fixed_threshold": 0.3,
        "runs": rows, "deltas_vs_R0": deltas, "mean_deltas_vs_R0": means,
        "three_checkpoint_direction": trend,
        "small_sample_sensitivity": robust,
        "segmentation_causal_gate": "PASS" if gate_pass else "FAIL"}
    write_json(args.output_root / "stage_s3b_segmentation_comparison.json", comparison)
    write_json(args.output_root / "stage_s3b_control_matrix.json", {
        "stage": "seg_raster_stage_s3b", "status": "PASS",
        "selected_lr_multiplier": plan["selected_lr_multiplier"],
        "selected_loss_kind": plan["selected_loss_kind"],
        "shared_threshold": shared_threshold, "common_step": common_step,
        "runs": {key: value["run_id"] for key, value in values.items()}})
    checkpoint_inventory = {
        "stage": "seg_raster_stage_s3b", "status": "PASS",
        "expected_steps": steps, "versioned_model_checkpoint_count": 36,
        "all_controls_have_identical_step_grid": True,
        "runs": {key: value["checkpoint_inventory"] for key, value in values.items()}}
    write_json(args.output_root / "stage_s3b_checkpoint_inventory.json",
               checkpoint_inventory)
    parity = {key: {"sample_identity_sha256": value["first_100_batch_identity_sha256"],
                    "common_tensor_sha256": value["first_100_common_tensor_sha256"],
                    "valid_mask_sha256": value.get("first_100_valid_mask_sha256")}
              for key, value in values.items()}
    sample_pass = len({row["sample_identity_sha256"] for row in parity.values()}) == 1
    common_pass = len({row["common_tensor_sha256"] for row in parity.values()}) == 1
    raster_mask_pass = len({parity[key]["valid_mask_sha256"]
                            for key in ("R1", "R2", "R3")}) == 1
    write_json(args.output_root / "stage_s3b_sample_parity.json", {
        "stage": "seg_raster_stage_s3b",
        "status": "PASS" if sample_pass and common_pass and raster_mask_pass else "FAIL",
        "first_100_batches": parity, "sample_identity_parity": sample_pass,
        "common_tensor_parity": common_pass,
        "raster_control_valid_mask_parity": raster_mask_pass,
        "image_only_valid_mask": "NOT_APPLICABLE_NOT_LOADED"})
    anchor_comparison = {
        "stage": "seg_raster_stage_s3b",
        "status": "NOT_EXECUTED_BY_GATE" if not gate_pass else "PENDING",
        "segmentation_causal_gate": comparison["segmentation_causal_gate"],
        "baseline_controlled_common_step": common_step}
    graph_comparison = dict(anchor_comparison)
    write_json(args.output_root / "stage_s3b_anchor_comparison.json", anchor_comparison)
    write_json(args.output_root / "stage_s3b_graph_comparison.json", graph_comparison)
    if gate_pass:
        graph_plan = {"stage": "seg_raster_stage_s3b", "phase": "G",
                      "run_code_sha": args.run_code_sha, "common_step": common_step,
                      "shared_threshold": shared_threshold,
                      "runs": {key: {"source_run_id": value["run_id"],
                                     "checkpoint": "model_step_{:06d}.pth.tar".format(common_step)}
                               for key, value in values.items()}}
        write_json(args.output_root / "phase_g_plan.json", graph_plan)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("plan-a", "reduce-a", "reduce-b", "reduce-c"))
    parser.add_argument("--run-code-sha", required=True)
    parser.add_argument("--run-root", type=Path,
                        default=REPO_ROOT / "data_self/stage_s3b_seg_raster")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "plan-a":
        write_json(args.output_root / "phase_a_plan.json", plan_a(args.run_code_sha))
        return 0
    return {"reduce-a": reduce_a, "reduce-b": reduce_b,
            "reduce-c": reduce_c}[args.action](args)


if __name__ == "__main__":
    raise SystemExit(main())
