"""Plan and reduce the frozen Stage S3D N0--N4 experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.seg_raster.stage_s3c import SAMPLE_GRID
from utils.seg_raster.stage_s3d import (
    model_contract_payload, null_parity_audit, select_n0_common_samples,
    spatial_causal_gate,
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")


def run_id(key: str) -> str:
    names = {
        "N0": "N0_null_seed20260827", "N1": "N1_aligned_seed20260827",
        "N2": "N2_zero_seed20260827",
        "N3": "N3_shift512_seed20260827",
        "N4": "N4_permuted_seed20260827",
    }
    return names[key]


def plan(args: argparse.Namespace) -> int:
    controls = {"N0": "null", "N1": "aligned", "N2": "zero",
                "N3": "shift_large", "N4": "permuted"}
    write_json(args.output_root / "phase_c_plan.json", {
        "stage": "seg_raster_stage_s3d", "phase": "C",
        "run_code_sha": args.run_code_sha, "seed": 20260827,
        "jobs": [{"run_key": key, "run_id": run_id(key),
                  "control": control}
                 for key, control in controls.items()],
    })
    write_json(args.output_root / "stage_s3d_model_contract.json",
               model_contract_payload())
    return 0


def load_summary(root: Path, key: str, sha: str) -> dict:
    value = read_json(root / run_id(key) / "summary.json")
    if value.get("status") != "PASS" or value.get("code_sha") != sha:
        raise RuntimeError("invalid Stage S3D run: " + key)
    if set(map(int, value["validation_metrics_by_samples"])) != set(SAMPLE_GRID):
        raise RuntimeError("incomplete checkpoint grid: " + key)
    if not value.get("original_bn_checksum_unchanged"):
        raise RuntimeError("original BatchNorm changed: " + key)
    return value


def at(value: dict, samples: int) -> dict:
    return dict(value["validation_metrics_by_samples"][str(samples)])


def reduce(args: argparse.Namespace) -> int:
    values = {key: load_summary(args.run_root, key, args.run_code_sha)
              for key in ("N0", "N1", "N2", "N3", "N4")}
    parity_rows = {}
    parity_pass = True
    for samples in SAMPLE_GRID:
        audit = null_parity_audit(at(values["N0"], samples),
                                  at(values["N2"], samples), atol=0.0)
        parity_rows[str(samples)] = audit
        parity_pass &= audit["status"] == "PASS"
    write_json(args.output_root / "stage_s3d_null_parity.json", {
        "stage": "seg_raster_stage_s3d",
        "status": "PASS" if parity_pass else "FAIL",
        "all_common_checkpoints_compared": True,
        "checkpoint_results": parity_rows,
        "whole_checkpoint_file_sha_equality_required": False,
    })
    common = select_n0_common_samples(
        values["N0"]["validation_metrics_by_samples"])
    rows = {key: at(value, common) for key, value in values.items()}
    center = SAMPLE_GRID.index(common)
    start = max(0, min(center - 1, len(SAMPLE_GRID) - 3))
    trend_samples = SAMPLE_GRID[start:start + 3]
    trend = []
    for samples in trend_samples:
        aligned = at(values["N1"], samples)
        controls = [at(values[key], samples)
                    for key in ("N0", "N2", "N3", "N4")]
        trend.append({
            "samples_seen": samples,
            "aligned_gt_all_controls": all(
                aligned[name] > control[name]
                for control in controls
                for name in ("road_f1", "road_iou", "road_auprc")),
        })
    input_swap = read_json(args.output_root / "stage_s3d_input_swap_matrix.json")
    input_dependence = input_swap.get("current_raster_input_dependence")
    input_swap_aligned_best = input_swap.get(
        "r1_trained_aligned_strictly_best", False)
    junction_shas = {key: rows[key]["junction_prediction_sha256"]
                     for key in rows}
    junction_parity = len(set(junction_shas.values())) == 1
    gate = spatial_causal_gate(
        rows, null_parity="PASS" if parity_pass else "FAIL",
        three_checkpoint_direction=[row["aligned_gt_all_controls"]
                                    for row in trend],
        input_swap_aligned_best=bool(input_swap_aligned_best),
        junction_parity=junction_parity)
    comparison = {
        "stage": "seg_raster_stage_s3d", "status": "PASS",
        "baseline_controlled_common_samples": common,
        "common_samples_selected_from": "N0_road_composite_only",
        "fixed_threshold": 0.3, "runs": rows,
        "three_checkpoint_direction": trend,
        "junction_prediction_sha256": junction_shas,
        "junction_parity": "PASS" if junction_parity else "FAIL",
        "current_raster_input_dependence": input_dependence,
        "segmentation_spatial_causal_gate": gate["status"],
        "gate_evidence": gate,
    }
    write_json(args.output_root / "stage_s3d_segmentation_comparison.json",
               comparison)
    identity_shas = {key: value["first_20_batch_identity_sha256"]
                     for key, value in values.items()}
    common_shas = {key: value["first_20_common_tensor_sha256"]
                   for key, value in values.items()}
    mask_shas = {key: value["first_20_valid_mask_sha256"]
                 for key, value in values.items()}
    sample_pass = (len(set(identity_shas.values())) == 1
                   and len(set(common_shas.values())) == 1
                   and len(set(mask_shas.values())) == 1)
    write_json(args.output_root / "stage_s3d_sample_parity.json", {
        "stage": "seg_raster_stage_s3d",
        "status": "PASS" if sample_pass else "FAIL",
        "sample_identity_sha256": identity_shas,
        "common_tensor_sha256": common_shas,
        "valid_mask_sha256": mask_shas,
        "permutation_donor_mapping_sha256": values["N4"][
            "training_donor_mapping_sha256"],
    })
    write_json(args.output_root / "stage_s3d_checkpoint_inventory.json", {
        "stage": "seg_raster_stage_s3d", "status": "PASS",
        "expected_samples_seen": list(SAMPLE_GRID),
        "versioned_model_checkpoint_count": 5 * len(SAMPLE_GRID),
        "runs": {key: value["checkpoint_inventory"]
                 for key, value in values.items()},
    })
    write_json(args.output_root / "stage_s3d_training_dynamics.json", {
        "stage": "seg_raster_stage_s3d", "status": "PASS",
        "runs": {key: {
            "optimizer_updates": value["optimizer_updates"],
            "micro_batches": value["micro_batches"],
            "samples_seen": value["samples_seen"],
            "runtime_seconds": value["runtime_seconds"],
            "validation_metrics_by_samples": value[
                "validation_metrics_by_samples"],
        } for key, value in values.items()},
    })
    anchor_status = "PENDING" if gate["status"] == "PASS" else "NOT_EXECUTED_BY_GATE"
    write_json(args.output_root / "stage_s3d_anchor_comparison.json", {
        "stage": "seg_raster_stage_s3d", "status": anchor_status,
        "segmentation_spatial_causal_gate": gate["status"],
        "baseline_controlled_common_samples": common,
    })
    write_json(args.output_root / "stage_s3d_graph_comparison.json", {
        "stage": "seg_raster_stage_s3d", "status": "NOT_EXECUTED_BY_GATE",
        "segmentation_spatial_causal_gate": gate["status"],
        "frozen_anchor_indirect_gate": (
            "PENDING" if gate["status"] == "PASS" else "NOT_EXECUTED"),
    })
    if gate["status"] == "PASS":
        write_json(args.output_root / "phase_anchor_plan.json", {
            "stage": "seg_raster_stage_s3d", "phase": "ANCHOR",
            "run_code_sha": args.run_code_sha, "common_samples": common,
            "runs": {key: {"run_id": value["run_id"],
                           "control": value["control"],
                           "checkpoint": "samples_{:06d}.pth.tar".format(common)}
                     for key, value in values.items()},
        })
    return 0 if parity_pass and sample_pass else 4


def reduce_anchor(args: argparse.Namespace) -> int:
    raw = read_json(args.output_root / "stage_s3d_anchor_raw.json")
    rows = raw["runs"]
    n1 = rows["N1"]
    gate = all(
        n1["threshold_recall"] > rows[key]["threshold_recall"]
        and n1["top_k_recall"] >= rows[key]["top_k_recall"]
        and n1["localization_error"] <= rows[key]["localization_error"]
        for key in ("N0", "N2", "N3", "N4"))
    multistep = any(value > 0 for value in n1["per_step_recall"][1:])
    result = dict(raw)
    result.update({
        "status": "PASS", "frozen_anchor_indirect_gate": (
            "PASS" if gate and multistep else "FAIL"),
        "multistep_anchor_validity": "PASS" if multistep else "FAIL",
    })
    write_json(args.output_root / "stage_s3d_anchor_comparison.json", result)
    if gate and multistep:
        plan = read_json(args.output_root / "phase_anchor_plan.json")
        write_json(args.output_root / "phase_graph_plan.json", {
            "stage": "seg_raster_stage_s3d", "phase": "GRAPH",
            "run_code_sha": args.run_code_sha,
            "common_samples": plan["common_samples"],
            "fixed_threshold": 0.3,
            "runs": {key: {
                "run_id": spec["run_id"], "checkpoint": spec["checkpoint"],
                "checkpoint_kind": "stage_s3d", "control": spec["control"],
            } for key, spec in plan["runs"].items()},
        })
    return 0


def reduce_graph(args: argparse.Namespace) -> int:
    rows = {key: read_json(args.output_root / "graph_results" / (key + ".json"))
            for key in ("N0", "N1", "N2", "N3", "N4")}
    n1 = rows["N1"]
    stable = n1.get("status") == "PASS" and bool(n1.get("natural_termination"))
    write_json(args.output_root / "stage_s3d_graph_comparison.json", {
        "stage": "seg_raster_stage_s3d", "status": "PASS",
        "execution_environment": "REMOTE_TRAINING_SERVER",
        "runs": rows,
        "all_runs_under_uniform_resource_caps": True,
        "graph_stability_gate": "PASS" if stable else "FAIL",
        "computational_feasibility": "PASS" if stable else "FAIL",
    })
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=(
        "plan", "reduce", "reduce-anchor", "reduce-graph"))
    parser.add_argument("--run-code-sha", required=True)
    parser.add_argument("--run-root", type=Path,
                        default=Path("data_self/stage_s3d_seg_raster"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    return {
        "plan": plan, "reduce": reduce,
        "reduce-anchor": reduce_anchor, "reduce-graph": reduce_graph,
    }[args.action](args)


if __name__ == "__main__":
    raise SystemExit(main())
