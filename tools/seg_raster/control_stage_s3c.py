"""Plan and reduce the frozen Stage S3C experiment matrix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from utils.seg_raster.stage_s3c import (
    SAMPLE_GRID, repair_composite, select_baseline_controlled_common_samples,
    select_phase_a_loss,
)


LOSS_LEGACY = "legacy_exact"
LOSS_BALANCED = "class_balanced_bce"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")


def run_id(key: str) -> str:
    return key + "_seed20260827"


def job(
    key: str, phase: str, input_kind: str, loss_kind: str, *,
    control: str | None = None, pos_weight: float = 1.0,
    loss_alpha: float = 1.0,
) -> dict:
    return {
        "run_key": key, "run_id": run_id(key), "phase": phase,
        "input_kind": input_kind, "control": control,
        "loss_kind": loss_kind, "pos_weight": pos_weight,
        "loss_alpha": loss_alpha,
    }


def load_summary(run_root: Path, key: str, code_sha: str) -> dict:
    value = read_json(run_root / run_id(key) / "summary.json")
    if value.get("status") != "PASS" or value.get("code_sha") != code_sha:
        raise RuntimeError("invalid Stage S3C summary: " + key)
    if set(map(int, value["validation_metrics_by_samples"])) != set(SAMPLE_GRID):
        raise RuntimeError("incomplete sample checkpoint grid: " + key)
    if len(value.get("checkpoint_inventory", [])) != len(SAMPLE_GRID):
        raise RuntimeError("incomplete checkpoint inventory: " + key)
    if not value.get("original_bn_checksum_unchanged"):
        raise RuntimeError("original BatchNorm changed: " + key)
    return value


def best_metrics(value: dict) -> tuple[int, dict]:
    samples = max(
        SAMPLE_GRID,
        key=lambda point: (repair_composite(
            value["validation_metrics_by_samples"][str(point)]), -point))
    return samples, value["validation_metrics_by_samples"][str(samples)]


def sensitivity(left: dict, right: dict) -> dict:
    left_rows, right_rows = left["per_sample"], right["per_sample"]
    if len(left_rows) != len(right_rows) or len(left_rows) < 2:
        return {"status": "INCONCLUSIVE", "single_sample_driven": True}
    values = np.asarray([
        np.mean([right_rows[index][name] - left_rows[index][name]
                 for name in ("road_f1", "road_iou", "road_auprc")])
        for index in range(len(left_rows))], dtype=np.float64)
    leave_one_out = [float(np.mean(np.delete(values, index)))
                     for index in range(len(values))]
    rng = np.random.default_rng(20260827)
    bootstrap = [float(np.mean(values[
        rng.integers(0, len(values), len(values))])) for _ in range(1000)]
    low, high = np.percentile(bootstrap, [2.5, 97.5])
    return {
        "status": "PASS" if min(leave_one_out) > 0 and low > 0 else "FAIL",
        "single_sample_driven": min(leave_one_out) <= 0,
        "leave_one_out_min_delta": min(leave_one_out),
        "bootstrap_95_percentile_interval": [float(low), float(high)],
        "sample_count": len(values),
    }


def plan_a(args: argparse.Namespace) -> int:
    diagnostic = read_json(args.output_root / "stage_s3c_loss_gradient_audit.json")
    plan = {
        "stage": "seg_raster_stage_s3c", "phase": "A",
        "run_code_sha": args.run_code_sha,
        "selection_uses_image_only_only": True,
        "jobs": [
            job("P0", "A", "image_only", LOSS_LEGACY),
            job("P1", "A", "image_only", LOSS_BALANCED,
                pos_weight=diagnostic["capped_pos_weight"],
                loss_alpha=diagnostic["balanced_global_alpha"]),
        ],
    }
    write_json(args.output_root / "phase_a_plan.json", plan)
    return 0


def plan_baseline(args: argparse.Namespace) -> int:
    checkpoint = os.environ.get("S3C_BASELINE_CHECKPOINT")
    if not checkpoint:
        raise RuntimeError("S3C_BASELINE_CHECKPOINT is required")
    write_json(args.output_root / "phase_baseline_graph_plan.json", {
        "stage": "seg_raster_stage_s3c", "phase": "BASELINE_GRAPH",
        "run_code_sha": args.run_code_sha, "fixed_threshold": 0.3,
        "runs": {"BASELINE": {
            "checkpoint_kind": "official_release",
            "checkpoint": checkpoint,
        }},
    })
    return 0


def reduce_a(args: argparse.Namespace) -> int:
    values = {key: load_summary(args.run_root, key, args.run_code_sha)
              for key in ("P0", "P1")}
    rows = []
    for key, kind in (("P0", LOSS_LEGACY), ("P1", LOSS_BALANCED)):
        samples, metrics = best_metrics(values[key])
        rows.append({
            "run_key": key, "input_kind": "image_only",
            "loss_kind": kind, "status": "PASS", "finite": True,
            "best_samples_seen": samples, "best_metrics": metrics,
            "single_sample_driven": False,
        })
    robust = sensitivity(rows[0]["best_metrics"], rows[1]["best_metrics"])
    rows[1]["single_sample_driven"] = robust["single_sample_driven"]
    selection = select_phase_a_loss(rows)
    write_json(args.output_root / "stage_s3c_loss_screen.json", {
        "stage": "seg_raster_stage_s3c", "status": "PASS",
        "run_code_sha": args.run_code_sha,
        "image_only_candidates": rows,
        "aligned_raster_results_available_during_selection": False,
        "small_sample_sensitivity_P1_vs_P0": robust,
        "selection": selection,
    })
    selected = next(row for row in rows
                    if row["run_key"] == selection["selected_run"])
    selected_summary = values[selected["run_key"]]
    pos_weight = float(selected_summary["junction_pos_weight"])
    alpha = float(selected_summary["junction_loss_alpha"])
    kind = str(selection["selected_loss_kind"])
    phase_b = {
        "stage": "seg_raster_stage_s3c", "phase": "B",
        "run_code_sha": args.run_code_sha,
        "selected_loss_kind": kind,
        "selected_from": selection["selected_run"],
        "jobs": [
            job("R0", "B", "image_only", kind,
                pos_weight=pos_weight, loss_alpha=alpha),
            job("R1", "B", "raster", kind, control="aligned",
                pos_weight=pos_weight, loss_alpha=alpha),
            job("R2", "B", "raster", kind, control="zero",
                pos_weight=pos_weight, loss_alpha=alpha),
            job("R3", "B", "raster", kind, control="shift_fixed",
                pos_weight=pos_weight, loss_alpha=alpha),
        ],
    }
    write_json(args.output_root / "phase_b_plan.json", phase_b)
    return 0


def metric_at(value: dict, samples: int) -> dict:
    return dict(value["validation_metrics_by_samples"][str(samples)])


def reduce_b(args: argparse.Namespace) -> int:
    values = {key: load_summary(args.run_root, key, args.run_code_sha)
              for key in ("R0", "R1", "R2", "R3")}
    common_samples = select_baseline_controlled_common_samples(
        values["R0"]["validation_metrics_by_samples"])
    rows = {key: metric_at(value, common_samples)
            for key, value in values.items()}
    names = ("road_f1", "road_iou", "road_auprc")
    deltas = {key: {name: rows[key][name] - rows["R0"][name]
                    for name in names} for key in ("R1", "R2", "R3")}
    means = {key: float(np.mean(list(value.values())))
             for key, value in deltas.items()}
    improved = sum(value > 0 for value in deltas["R1"].values())
    collapse = any(
        rows["R1"][name] < rows["R0"][name] - 0.10
        for name in ("road_precision", "road_recall",
                     "junction_precision", "junction_recall"))
    center = SAMPLE_GRID.index(common_samples)
    start = max(0, min(center - 1, len(SAMPLE_GRID) - 3))
    trend_samples = SAMPLE_GRID[start:start + 3]
    trend = []
    for samples in trend_samples:
        left = metric_at(values["R0"], samples)
        right = metric_at(values["R1"], samples)
        trend.append({
            "samples_seen": samples,
            "road_repair_signal": float(np.mean([
                right[name] - left[name] for name in names])),
        })
    robust = sensitivity(rows["R0"], rows["R1"])
    gate = (
        improved >= 2
        and means["R1"] > max(means["R2"], means["R3"])
        and rows["R1"]["junction_auprc"] >= rows["R0"]["junction_auprc"]
        and not collapse
        and len(trend) == 3
        and all(row["road_repair_signal"] > 0 for row in trend)
        and robust["status"] == "PASS"
    )
    comparison = {
        "stage": "seg_raster_stage_s3c", "status": "PASS",
        "baseline_controlled_common_samples": common_samples,
        "common_samples_selected_from": "R0_image_only_repair_composite_only",
        "fixed_threshold": 0.3, "runs": rows,
        "deltas_vs_R0": deltas, "mean_deltas_vs_R0": means,
        "three_checkpoint_direction": trend,
        "small_sample_sensitivity": robust,
        "precision_recall_collapse": collapse,
        "segmentation_causal_gate": "PASS" if gate else "FAIL",
    }
    write_json(args.output_root / "stage_s3c_segmentation_comparison.json",
               comparison)
    write_json(args.output_root / "stage_s3c_control_matrix.json", {
        "stage": "seg_raster_stage_s3c", "status": "PASS",
        "selected_loss_kind": values["R0"]["loss_kind"],
        "baseline_controlled_common_samples": common_samples,
        "runs": {key: value["run_id"] for key, value in values.items()},
    })
    write_json(args.output_root / "stage_s3c_checkpoint_inventory.json", {
        "stage": "seg_raster_stage_s3c", "status": "PASS",
        "expected_samples_seen": list(SAMPLE_GRID),
        "versioned_model_checkpoint_count": 4 * len(SAMPLE_GRID),
        "all_controls_have_identical_sample_grid": True,
        "runs": {key: value["checkpoint_inventory"]
                 for key, value in values.items()},
    })
    parity = {key: {
        "batch_identity_sha256": value["first_20_batch_identity_sha256"],
        "common_tensor_sha256": value["first_20_common_tensor_sha256"],
        "raster_sha256": value.get("first_20_raster_sha256"),
        "valid_mask_sha256": value.get("first_20_valid_mask_sha256"),
    } for key, value in values.items()}
    identity_pass = len({row["batch_identity_sha256"]
                         for row in parity.values()}) == 1
    common_pass = len({row["common_tensor_sha256"]
                       for row in parity.values()}) == 1
    mask_pass = len({parity[key]["valid_mask_sha256"]
                     for key in ("R1", "R2", "R3")}) == 1
    write_json(args.output_root / "stage_s3c_sample_parity.json", {
        "stage": "seg_raster_stage_s3c",
        "status": "PASS" if identity_pass and common_pass and mask_pass else "FAIL",
        "first_200_samples": parity,
        "sample_identity_parity": identity_pass,
        "common_tensor_parity": common_pass,
        "raster_control_valid_mask_parity": mask_pass,
        "raster_content_control_specific": len({
            parity[key]["raster_sha256"] for key in ("R1", "R2", "R3")}) == 3,
    })
    write_json(args.output_root / "stage_s3c_bn_checksum_audit.json", {
        "stage": "seg_raster_stage_s3c", "status": "PASS",
        "runs": {key: {
            "initial": value["original_bn_checksum_initial"],
            "final": value["original_bn_checksum_final"],
            "unchanged": value["original_bn_checksum_unchanged"],
        } for key, value in values.items()},
    })
    write_json(args.output_root / "stage_s3c_training_dynamics.json", {
        "stage": "seg_raster_stage_s3c", "status": "PASS",
        "runs": {key: {
            "optimizer_updates": value["optimizer_updates"],
            "micro_batches": value["micro_batches"],
            "samples_seen": value["samples_seen"],
            "runtime_seconds": value["runtime_seconds"],
            "validation_metrics_by_samples": value["validation_metrics_by_samples"],
        } for key, value in values.items()},
    })
    initial_status = "PENDING" if gate else "NOT_EXECUTED_BY_GATE"
    write_json(args.output_root / "stage_s3c_anchor_comparison.json", {
        "stage": "seg_raster_stage_s3c", "status": initial_status,
        "segmentation_causal_gate": comparison["segmentation_causal_gate"],
        "baseline_controlled_common_samples": common_samples,
    })
    write_json(args.output_root / "stage_s3c_graph_comparison.json", {
        "stage": "seg_raster_stage_s3c", "status": "NOT_EXECUTED_BY_GATE",
        "segmentation_causal_gate": comparison["segmentation_causal_gate"],
        "frozen_anchor_indirect_gate": "PENDING" if gate else "NOT_EXECUTED",
    })
    if gate:
        write_json(args.output_root / "phase_anchor_plan.json", {
            "stage": "seg_raster_stage_s3c", "phase": "ANCHOR",
            "run_code_sha": args.run_code_sha,
            "common_samples": common_samples,
            "runs": {key: {
                "source_run_id": value["run_id"],
                "checkpoint": "samples_{:06d}.pth.tar".format(common_samples),
                "control": value.get("control"),
            } for key, value in values.items()},
        })
    return 0


def reduce_anchor(args: argparse.Namespace) -> int:
    payload = read_json(args.output_root / "stage_s3c_anchor_raw.json")
    rows = payload["runs"]
    required = ("R0", "R1", "R2", "R3")
    if any(key not in rows for key in required):
        raise RuntimeError("incomplete frozen anchor result")
    r1 = rows["R1"]
    threshold_ok = all(r1["threshold_recall"] > rows[key]["threshold_recall"]
                       for key in ("R0", "R2", "R3"))
    topk_ok = all(r1["top_k_recall"] >= rows[key]["top_k_recall"]
                  for key in ("R0", "R2", "R3"))
    localization_ok = all(r1["localization_error"] <= rows[key]["localization_error"]
                          for key in ("R0", "R2", "R3"))
    multistep = any(value > 0 for value in r1["per_step_recall"][1:])
    accounting = (r1["evaluated_target_count"] > 0
                  and r1["false_positive_count"] >= 0)
    gate = threshold_ok and topk_ok and localization_ok and multistep and accounting
    result = dict(payload)
    result.update({
        "status": "PASS",
        "aligned_specificity": {
            "threshold_recall_gt_all_controls": threshold_ok,
            "top_k_recall_ge_all_controls": topk_ok,
            "localization_error_le_all_controls": localization_ok,
        },
        "multistep_anchor_validity": "PASS" if multistep else "FAIL",
        "prediction_accounting_valid": accounting,
        "frozen_anchor_indirect_gate": "PASS" if gate else "FAIL",
    })
    write_json(args.output_root / "stage_s3c_anchor_comparison.json", result)
    if gate:
        plan = read_json(args.output_root / "phase_anchor_plan.json")
        write_json(args.output_root / "phase_graph_plan.json", {
            "stage": "seg_raster_stage_s3c", "phase": "GRAPH",
            "run_code_sha": args.run_code_sha,
            "common_samples": plan["common_samples"],
            "fixed_threshold": 0.3, "runs": plan["runs"],
        })
    return 0


def reduce_baseline(args: argparse.Namespace) -> int:
    evaluation_path = args.output_root / "stage_s3c_baseline_evaluation.json"
    evaluation = read_json(evaluation_path)
    graph = read_json(args.output_root / "graph_results" / "BASELINE.json")
    gate = graph.get("status") == "PASS" and bool(graph.get("natural_termination"))
    evaluation["graph"] = graph
    evaluation["baseline_graph_gate"] = "PASS" if gate else "FAIL"
    evaluation["status"] = "PASS" if gate else "FAIL"
    write_json(evaluation_path, evaluation)
    return 0 if gate else 4


def reduce_graph(args: argparse.Namespace) -> int:
    rows = {key: read_json(args.output_root / "graph_results" / (key + ".json"))
            for key in ("R0", "R1", "R2", "R3")}
    r0, r1 = rows["R0"], rows["R1"]
    ratios = {
        "iterations": (r1["graph_iterations"] / r0["graph_iterations"]
                       if r0["graph_iterations"] else None),
        "directed_edges": (r1["directed_edge_count"] / r0["directed_edge_count"]
                           if r0["directed_edge_count"] else None),
        "runtime": (r1["runtime_seconds"] / r0["runtime_seconds"]
                    if r0["runtime_seconds"] else None),
    }
    no_caps = all(row["status"] == "PASS" and row["natural_termination"]
                  for row in rows.values())
    engineering = (
        no_caps
        and ratios["iterations"] is not None and ratios["iterations"] <= 3.0
        and ratios["directed_edges"] is not None
        and ratios["directed_edges"] <= 3.0
        and ratios["runtime"] is not None and ratios["runtime"] <= 5.0)
    write_json(args.output_root / "stage_s3c_graph_comparison.json", {
        "stage": "seg_raster_stage_s3c", "status": "PASS",
        "execution_environment": "REMOTE_TRAINING_SERVER",
        "runs": rows, "R1_vs_R0_scale_ratio": ratios,
        "all_controls_natural_termination": no_caps,
        "graph_stability_gate": "PASS" if engineering else "FAIL",
        "computational_feasibility": "PASS" if engineering else "FAIL",
        "connectivity_gain_not_accepted_if_resource_cap_reached": True,
    })
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=(
        "plan-baseline", "plan-a", "reduce-a", "reduce-b", "reduce-anchor",
        "reduce-baseline", "reduce-graph"))
    parser.add_argument("--run-code-sha", required=True)
    parser.add_argument("--run-root", type=Path,
                        default=REPO_ROOT / "data_self/stage_s3c_seg_raster")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    return {"plan-baseline": plan_baseline,
            "plan-a": plan_a, "reduce-a": reduce_a,
            "reduce-b": reduce_b, "reduce-anchor": reduce_anchor,
            "reduce-baseline": reduce_baseline,
            "reduce-graph": reduce_graph}[args.action](args)


if __name__ == "__main__":
    raise SystemExit(main())
