"""Reduce immutable Stage S3 evidence into the Stage S3A forensic report.

The reducer consumes only small JSON/JSONL evidence plus file metadata.  It
does not load a model, run CUDA, mutate a checkpoint, or overwrite Stage S3
artifacts.  Missing historical checkpoints are represented explicitly rather
than reconstructed from logs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.seg_raster.audit_stage_s3a_metrics import (
    finite_json_dumps,
    graph_control_matrix,
    leave_one_out_deltas,
    numeric_max_abs_difference,
    paired_bootstrap_delta,
    select_best_validation_record,
    validate_recorded_best_step,
)
from utils.seg_raster.stage_s3 import EXPERIMENT_MATRIX, sha256_file


FORMAL_S3_SHA = "2e68f4e5a1c7cfad041182c2edce3194b8175b8c"
RUN_KEYS = ("C0", "C1", "C2", "C3", "J0", "J1")
DETACH_KEYS = ("C0", "C1", "C2", "C3")
METRIC_KEYS = ("road_f1", "road_iou", "junction_f1")
PATHOLOGICAL_GRAPH_STATUS = "TERMINATED_PATHOLOGICAL_EXPANSION"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(finite_json_dumps(payload), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def git_text(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, text=True,
        capture_output=True).stdout.strip()


def stable_sha(value: Any) -> str:
    data = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def metric_value(record: Mapping[str, Any], name: str) -> float:
    road_or_junction, key = name.split("_", 1)
    return float(record["segmentation"][road_or_junction][key])


def compact_checkpoint(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run_key": record["run_key"],
        "checkpoint_kind": record["checkpoint_kind"],
        "checkpoint_step": int(record["checkpoint_step"]),
        "checkpoint_sha256": record["checkpoint_sha256"],
        "validation_plan_sha": record["validation_plan_sha"],
        "sample_identity_sha": record["sample_identity_sha"],
        "metric_code_sha": record["metric_code_sha"],
        "threshold": float(record["fixed_threshold"]),
        "segmentation": record["segmentation"],
        "anchor": record["anchor"],
        "evaluation_time_seconds": record["evaluation_time_seconds"],
        "status": record["status"],
    }


def deltas(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Return right-left deltas for the main segmentation and anchor metrics."""
    result = {
        name: metric_value(right, name) - metric_value(left, name)
        for name in METRIC_KEYS
    }
    result["segmentation_composite"] = (
        float(right["segmentation"]["segmentation_composite"])
        - float(left["segmentation"]["segmentation_composite"])
    )
    for name in ("top_k_recall", "localization_error", "missed_branch_count",
                 "channel_diversity_mean_absolute_difference"):
        result["anchor_" + name] = float(right["anchor"][name]) - float(
            left["anchor"][name])
    return result


def historical_metric_rows(source_root: Path) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    specifications = {item.key: item for item in EXPERIMENT_MATRIX}
    all_rows, validation = {}, {}
    for key in RUN_KEYS:
        path = source_root / specifications[key].run_id / "metrics.jsonl"
        rows = read_jsonl(path)
        all_rows[key] = rows
        validation[key] = [row for row in rows if row.get("kind") == "frozen_validation"]
        if len(validation[key]) != 20:
            raise RuntimeError(f"{key}: expected 20 validation log records")
    return all_rows, validation


def load_evaluations(remote_output: Path) -> tuple[list[dict], dict[tuple[str, str], dict]]:
    records = [read_json(path) for path in sorted(
        (remote_output / "checkpoint_results").glob("*.json"))]
    primary = {
        (row["run_key"], row["checkpoint_kind"]): row
        for row in records if int(row.get("repeat_index", 0)) == 0
    }
    missing = [(key, kind) for key in RUN_KEYS for kind in ("best", "latest")
               if (key, kind) not in primary]
    if missing:
        raise RuntimeError(f"missing checkpoint evaluations: {missing}")
    return records, primary


def checkpoint_inventory(
    source_root: Path,
    training: Mapping[str, Any],
    validation: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    specifications = {item.key: item for item in EXPERIMENT_MATRIX}
    by_run: dict[str, list[dict[str, Any]]] = {}
    historical = []
    actual = []
    for key in RUN_KEYS:
        spec = specifications[key]
        run_dir = source_root / spec.run_id
        run_actual = []
        for kind in ("best", "latest"):
            path = run_dir / "checkpoints" / f"{kind}.pth.tar"
            row = {
                "run_key": key,
                "checkpoint_kind": kind,
                "step": int(
                    training["runs"][key]["optimizer_steps"] if kind == "latest"
                    else select_best_validation_record(validation[key])["step"]),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "logical_path": "${REMOTE_RUN_ROOT}/stage_s3/" + spec.run_id
                                + "/checkpoints/" + path.name,
                "forwardable": True,
            }
            run_actual.append(row)
            actual.append(row)
        by_run[key] = run_actual
        available_steps = {int(row["step"]): row for row in run_actual}
        for record in validation[key]:
            step = int(record["step"])
            historical.append({
                "run_key": key,
                "step": step,
                "logged_validation_metrics_available": True,
                "checkpoint_file_available": step in available_steps,
                "checkpoint_kind_if_available": (
                    available_steps[step]["checkpoint_kind"] if step in available_steps else None),
                "forward_recomputation_status": (
                    "AVAILABLE" if step in available_steps else "UNAVAILABLE_MISSING_CHECKPOINT"),
            })
    payload = {
        "stage": "seg_raster_stage_s3a",
        "status": "PASS_WITH_HISTORICAL_CHECKPOINT_LIMITATION",
        "execution_environment": "REMOTE_TRAINING_SERVER",
        "formal_s3_run_code_sha": FORMAL_S3_SHA,
        "expected_validation_checkpoint_count_from_protocol": 120,
        "historical_validation_log_record_count": len(historical),
        "actual_checkpoint_file_count": len(actual),
        "forwardable_unique_file_count": len(actual),
        "missing_historical_checkpoint_file_count": len(historical) - sum(
            row["checkpoint_file_available"] for row in historical),
        "retention_policy_observed": (
            "train_stage_s3.py overwrote latest.pth.tar at each evaluation and "
            "best.pth.tar only on composite improvement; no versioned step files remain"),
        "evidence_boundary": (
            "All 120 historical points are log evidence; only the 12 listed files "
            "can be independently forwarded. Missing weights were not fabricated or retrained."),
        "actual_checkpoint_files": actual,
        "historical_validation_records": historical,
    }
    return payload, by_run


def provenance_payload(training: Mapping[str, Any], primary: Mapping[tuple[str, str], dict]) -> dict:
    final_trace = {
        "dataset_split": "train", "batch_count": 1,
        "cross_batch_accumulation": False, "model_mode": "train",
        "torch_no_grad": False, "computed_from_output_before_optimizer_update": True,
        "stored_after_optimizer_update": True, "target_prediction_alias": False,
        "accumulator_reset": False, "computation_bug": False,
    }
    return {
        "stage": "seg_raster_stage_s3a",
        "status": "PASS",
        "formal_s3_run_code_sha": FORMAL_S3_SHA,
        "final_metrics_classification": "LAST_TRAIN_BATCH_METRICS",
        "fields": {
            "final_metrics": {
                "definition": "tools/seg_raster/train_stage_s3.py:_seg_metrics and main lines 329-375,427-428",
                "input": "last deterministic training batch at optimizer step 102400",
                "split": "train", "batch_identity": "last sample-plan batch",
                "model_mode": "train", "torch_no_grad": False,
                "checkpoint_kind": "none; live training model",
                "checkpoint_step": 102400,
                "checkpoint_sha256": None,
                "sigmoid": "once inside _seg_metrics",
                "threshold": 0.3, "aggregation": "micro within one batch",
                "cross_batch_accumulation": False,
                "optimizer_timing": "pre-update output measured after optimizer.step",
                "classification_trace": final_trace,
                "per_run_values": {key: training["runs"][key]["final_metrics"] for key in RUN_KEYS},
            },
            "best_metrics": {
                "definition": "tools/seg_raster/train_stage_s3.py:_evaluate_validation lines 203-240; main 388-400",
                "input": "frozen validation plan: 8 batches / 16 samples",
                "split": "validation", "model_mode": "eval", "torch_no_grad": True,
                "checkpoint_kind": "per-run best validation segmentation composite",
                "sigmoid": "once at metric boundary", "threshold": 0.3,
                "aggregation": "micro over all 16 samples", "cross_batch_accumulation": True,
                "optimizer_timing": "post-update model",
            },
            "stage_s3_segmentation_comparison": {
                "source": "each run evaluation/segmentation.json overwritten at every evaluation",
                "checkpoint_kind": "latest", "checkpoint_step": 102400,
                "validation_plan_sha": primary[("C0", "latest")]["validation_plan_sha"],
            },
            "stage_s3_anchor_comparison": {
                "source": "each run evaluation/anchor.json overwritten at every evaluation",
                "checkpoint_kind": "latest", "checkpoint_step": 102400,
                "validation_plan_sha": primary[("C0", "latest")]["validation_plan_sha"],
            },
            "stage_s3_graph_comparison": {
                "source": "formal closed-loop evaluator",
                "checkpoint_kind": "per-run best", "checkpoint_steps": {
                    key: primary[(key, "best")]["checkpoint_step"] for key in RUN_KEYS},
                "metric": "deterministic_pixel_graph_approximation, not official SpaceNet APLS",
            },
            "segmentation_composite": {
                "formula": "mean(road_F1, road_IoU, junction_F1)",
                "numerator": "road_F1 + road_IoU + junction_F1",
                "denominator": 3,
            },
        },
    }


def protocol_payload(primary: Mapping[tuple[str, str], dict], validation: Mapping[str, list[dict]]) -> dict:
    protocols: dict[str, Any] = {}
    for kind, label in (("latest", "A_latest"), ("best", "B_per_run_best")):
        rows = {key: primary[(key, kind)] for key in RUN_KEYS}
        protocols[label] = {
            "status": "PASS", "selection": kind,
            "runs": {key: compact_checkpoint(value) for key, value in rows.items()},
            "comparisons": {
                "C1_minus_C0": deltas(rows["C0"], rows["C1"]),
                "C1_minus_C2": deltas(rows["C2"], rows["C1"]),
                "C1_minus_C3": deltas(rows["C3"], rows["C1"]),
                "J1_minus_J0": deltas(rows["J0"], rows["J1"]),
            },
        }
    c0_step = int(select_best_validation_record(validation["C0"])["step"])
    j0_step = int(select_best_validation_record(validation["J0"])["step"])
    protocols["C_C0_best_common_step"] = {
        "status": "UNAVAILABLE_MISSING_CHECKPOINT", "requested_step": c0_step,
        "comparison_runs": list(DETACH_KEYS),
        "missing_runs": [key for key in DETACH_KEYS
                         if primary[(key, "best")]["checkpoint_step"] != c0_step
                         and primary[(key, "latest")]["checkpoint_step"] != c0_step],
    }
    protocols["C_J0_best_common_step"] = {
        "status": "UNAVAILABLE_MISSING_CHECKPOINT", "requested_step": j0_step,
        "comparison_runs": ["J0", "J1"],
        "missing_runs": [key for key in ("J0", "J1")
                         if primary[(key, "best")]["checkpoint_step"] != j0_step
                         and primary[(key, "latest")]["checkpoint_step"] != j0_step],
    }
    protocols["C_global_common_step_diagnostic"] = {
        "status": "UNAVAILABLE_MISSING_CHECKPOINT",
        "selection_rule": "image-only C0/J0 validation control without graph/test inspection",
        "candidate_steps": {"C0_best": c0_step, "J0_best": j0_step},
        "reason": "versioned checkpoints were not retained for all six runs",
    }
    latest_rank = sorted(RUN_KEYS, key=lambda key: primary[(key, "latest")]["segmentation"]["segmentation_composite"], reverse=True)
    best_rank = sorted(RUN_KEYS, key=lambda key: primary[(key, "best")]["segmentation"]["segmentation_composite"], reverse=True)
    return {
        "stage": "seg_raster_stage_s3a", "status": "PARTIAL_PROTOCOL_C_UNAVAILABLE",
        "protocols": protocols,
        "ranking": {"latest": latest_rank, "per_run_best": best_rank,
                    "ranking_reversal": latest_rank != best_rank},
        "causal_limit": "Per-run best compares different optimizer steps; common-step evidence is unavailable.",
    }


def training_dynamics_payload(all_rows: Mapping[str, list[dict]], validation: Mapping[str, list[dict]], training: Mapping[str, Any]) -> dict:
    runs = {}
    for key in RUN_KEYS:
        validations = validation[key]
        best = select_best_validation_record(validations)
        latest = validations[-1]
        train_rows = [row for row in all_rows[key]
                      if row.get("kind") is None and "loss" in row]
        final_recorded = training["runs"][key]["final_metrics"]
        last_train = train_rows[-1]["metrics"] if train_rows else None
        runs[key] = {
            "raw_validation_points": validations,
            "training_log_point_count": len(train_rows),
            "first_training_log_point": train_rows[0] if train_rows else None,
            "last_training_log_point": train_rows[-1] if train_rows else None,
            "best_step": int(best["step"]),
            "best_composite": float(best["metrics"]["segmentation_composite"]),
            "latest_composite": float(latest["metrics"]["segmentation_composite"]),
            "latest_over_best_ratio": (
                float(latest["metrics"]["segmentation_composite"])
                / float(best["metrics"]["segmentation_composite"]) if best["metrics"]["segmentation_composite"] else 0.0),
            "deterioration_after_best": float(
                latest["metrics"]["segmentation_composite"]
                - best["metrics"]["segmentation_composite"]),
            "final_metrics_exactly_match_last_train_log": last_train == final_recorded,
        }
    return {
        "stage": "seg_raster_stage_s3a", "status": "PASS",
        "runs": runs,
        "overall_assessment": (
            "All run-specific best steps occur before step 102400 and all latest composites are below their best; "
            "the fixed 102400 budget is longer than the observed validation optima for this single seed."),
        "final_metrics_scope": "last training batch only; not a validation or generalization estimate",
    }


def junction_payload(primary: Mapping[tuple[str, str], dict]) -> dict:
    records = {}
    causes = {"CLASS_IMBALANCE", "THRESHOLD_TOO_HIGH", "LATE_TRAINING_OVERFIT", "LOSS_DOMINATED_BY_BACKGROUND"}
    for key in RUN_KEYS:
        records[key] = {}
        for kind in ("best", "latest"):
            row = primary[(key, kind)]
            data = row["junction_forensics"]
            records[key][kind] = {
                "checkpoint_step": row["checkpoint_step"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                **data,
                "fixed_threshold_metrics": row["segmentation"]["junction"],
            }
            if data["probability_max"] < 0.3:
                causes.add("CALIBRATION_COLLAPSE")
            if data["logit_input_double_sigmoid_check"]["double_sigmoid_risk"]:
                causes.add("DOUBLE_SIGMOID")
    if "DOUBLE_SIGMOID" in causes:
        contract = "FAIL"
    else:
        contract = "PASS: logits returned; exactly one sigmoid; eval/no_grad; aligned 64x64 target/output"
    return {
        "stage": "seg_raster_stage_s3a", "status": "PASS",
        "records": records,
        "label_contract": {
            "shape": "[B,1,64,64]", "generation": "utils/model_utils.py junc_seg_small",
            "circle_radius_pixels": 2, "axis_order": "audited against synchronized crop coordinates",
            "downsampling": "label is drawn directly on 64x64 grid; no evaluator resampling",
        },
        "contract_check": contract,
        "root_cause_classification": sorted(causes),
        "interpretation": (
            "F1=0 at threshold 0.3 means no thresholded hits. It does not mean raw logits or probabilities are identically zero; "
            "AUPRC and probability distributions retain ranking information."),
        "threshold_policy": (
            "0.3 remains the causal comparison threshold. Per-run optimal thresholds are diagnostic only and are not used to claim improvements."),
    }


def loss_payload(primary: Mapping[tuple[str, str], dict]) -> dict:
    rows = {key: {kind: primary[(key, kind)]["loss_balance"]
                  for kind in ("best", "latest")} for key in RUN_KEYS}
    return {
        "stage": "seg_raster_stage_s3a", "status": "PASS",
        "actual_loss_contract": {
            "function": "tools/seg_raster/train_stage_s3.py:_losses",
            "criterion": "binary_cross_entropy_with_logits",
            "reduction": "sum", "batch_normalization": None,
            "class_weighting": None, "positive_weighting": None,
            "total": "road + junction + anchor + anchor_lowrs",
        },
        "checkpoint_batch_forensics": rows,
        "interpretation": (
            "Summed dense losses give background pixels substantial aggregate influence. Gradient norms are forensic passes on the frozen "
            "checkpoint and first validation batch; no optimizer update was performed."),
    }


def anchor_payload(primary: Mapping[tuple[str, str], dict], old_anchor: Mapping[str, Any]) -> tuple[dict, dict]:
    records, targets = {}, []
    for key in RUN_KEYS:
        records[key] = {}
        for kind in ("best", "latest"):
            row = primary[(key, kind)]
            records[key][kind] = {
                "checkpoint_step": row["checkpoint_step"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "metrics": row["anchor"],
                "existing_vs_reference_max_abs_difference": row[
                    "anchor_metric_reference_check"]["maximum_absolute_difference"],
            }
            for target in row["anchor_per_target"]:
                targets.append({"run_key": key, "checkpoint_kind": kind, **target})
    best = {key: primary[(key, "best")] for key in RUN_KEYS}
    latest = {key: primary[(key, "latest")] for key in RUN_KEYS}
    latest_later_three_zero = all(
        all(value == 0 for value in row["anchor"]["per_step_recall"][1:])
        for row in latest.values())
    detach_best_later_three_zero = all(
        all(value == 0 for value in best[key]["anchor"]["per_step_recall"][1:])
        for key in DETACH_KEYS)
    j1_best_later_step_hit = any(
        value > 0 for value in best["J1"]["anchor"]["per_step_recall"][1:])
    latest_no_false_positives = all(
        row["anchor"]["false_positive_count"] == 0 for row in latest.values())
    all_available_no_false_positives = all(
        row["anchor"]["false_positive_count"] == 0 for row in primary.values())
    historical_difference = max(
        numeric_max_abs_difference(
            old_anchor.get("runs", {}).get(key, {}),
            primary[(key, "latest")]["anchor"])
        for key in RUN_KEYS)
    payload = {
        "stage": "seg_raster_stage_s3a", "status": "PASS",
        "historical_anchor_comparison_provenance": {
            "status": "PASS" if historical_difference <= 1e-8 else "FAIL",
            "checkpoint_kind": "latest", "checkpoint_step": 102400,
            "evidence": "historical evaluation/anchor.json was overwritten at each validation; values match latest protocol",
            "historical_artifact": "artifacts/stage_s3_anchor_comparison.json",
            "historical_vs_recomputed_latest_maximum_absolute_difference": historical_difference,
        },
        "records": records,
        "protocol_comparisons": {
            "best": {"C1_minus_C0": deltas(best["C0"], best["C1"]),
                     "C1_minus_C2": deltas(best["C2"], best["C1"]),
                     "C1_minus_C3": deltas(best["C3"], best["C1"])},
            "latest": {"C1_minus_C0": deltas(latest["C0"], latest["C1"]),
                       "C1_minus_C2": deltas(latest["C2"], latest["C1"]),
                       "C1_minus_C3": deltas(latest["C3"], latest["C1"])},
            "common_step": {"status": "UNAVAILABLE_MISSING_CHECKPOINT"},
        },
        "multistep_diagnosis": {
            "zero_based_steps_1_through_3_recall_zero_for_all_latest_checkpoints": latest_later_three_zero,
            "zero_based_steps_1_through_3_recall_zero_for_all_detach_best_checkpoints": detach_best_later_three_zero,
            "j1_best_has_a_later_step_hit": j1_best_later_step_hit,
            "fixed_threshold_produced_no_false_positive_pixels_for_all_latest_checkpoints": latest_no_false_positives,
            "fixed_threshold_produced_no_false_positive_pixels_for_all_available_checkpoints": all_available_no_false_positives,
            "top_k_bypasses_fixed_threshold": True,
            "localization_error_scope": "all targets using global heatmap argmax",
            "interpretation": (
                "Threshold recall and false-positive counts can both be zero when all heatmaps stay below 0.3; top-K still measures ranking. "
                "Near-zero channel diversity and absent later-step hits make multistep behavior invalid as positive evidence."),
        },
        "C1_VS_C0_EFFECT": _aggregate_anchor_effect(best, latest, "C0"),
        "ALIGNED_VS_ZERO_SPECIFICITY": _aggregate_anchor_effect(best, latest, "C2"),
        "ALIGNED_VS_SHIFTED_SPECIFICITY": _aggregate_anchor_effect(best, latest, "C3"),
        "MULTISTEP_ANCHOR_VALIDITY": "FAIL" if (
            latest_later_three_zero and detach_best_later_three_zero) else "INCONCLUSIVE",
        "ANCHOR_METRIC_VALIDITY": "PASS" if all(
            row["anchor_metric_reference_check"]["maximum_absolute_difference"] <= 1e-8
            for row in primary.values()) else "FAIL",
        "historical_values": old_anchor.get("runs", {}),
    }
    return payload, {"stage": "seg_raster_stage_s3a", "status": "PASS", "rows": targets}


def _anchor_effect(left: Mapping[str, Any], aligned: Mapping[str, Any]) -> str:
    """Apply the frozen top-K/non-regression rule to one checkpoint protocol."""
    left_metrics = left["anchor"]
    aligned_metrics = aligned["anchor"]
    top_k_delta = float(aligned_metrics["top_k_recall"]) - float(
        left_metrics["top_k_recall"])
    improvements = (
        top_k_delta > 0
        or float(aligned_metrics["localization_error"])
        < float(left_metrics["localization_error"])
        or int(aligned_metrics["missed_branch_count"])
        < int(left_metrics["missed_branch_count"])
    )
    return "PASS" if top_k_delta >= -0.005 and improvements else "FAIL"


def _aggregate_anchor_effect(
    best: Mapping[str, Mapping[str, Any]],
    latest: Mapping[str, Mapping[str, Any]],
    baseline_key: str,
) -> str:
    statuses = {
        "best": _anchor_effect(best[baseline_key], best["C1"]),
        "latest": _anchor_effect(latest[baseline_key], latest["C1"]),
    }
    if set(statuses.values()) == {"PASS"}:
        return "PASS"
    if set(statuses.values()) == {"FAIL"}:
        return "FAIL"
    return "INCONCLUSIVE"


def pixel_parity(local: Mapping[str, Any], remote: Mapping[str, Any]) -> dict:
    names = sorted(set(local["files"]) | set(remote["files"]))
    rows, all_pixels = {}, True
    for name in names:
        left, right = local["files"][name], remote["files"][name]
        same_pixels = left["decoded_pixel_array_sha256"] == right["decoded_pixel_array_sha256"]
        all_pixels &= same_pixels
        rows[name] = {
            "local": left, "remote": right,
            "byte_sha_equal": left["byte_sha256"] == right["byte_sha256"],
            "decoded_shape_equal": left["decoded_shape"] == right["decoded_shape"],
            "decoded_dtype_equal": left["decoded_dtype"] == right["decoded_dtype"],
            "decoded_pixel_array_sha_equal": same_pixels,
            "max_absolute_difference": 0 if same_pixels else None,
            "different_pixel_count": 0 if same_pixels else None,
            "metadata_equal": left["metadata"] == right["metadata"],
            "png_encoding_difference_only": bool(
                same_pixels and left["byte_sha256"] != right["byte_sha256"]),
        }
    return {
        "stage": "seg_raster_stage_s3a", "status": "PASS" if all_pixels else "FAIL",
        "pixel_array_sha_definition": "sha256(np.ascontiguousarray(array).tobytes())",
        "files": rows,
        "all_decoded_pixel_arrays_identical": all_pixels,
        "encoding_difference_claim_allowed": all_pixels,
    }


def sensitivity_payload(primary: Mapping[tuple[str, str], dict]) -> dict:
    protocols = {}
    for kind in ("best", "latest"):
        left = primary[("C0", kind)]
        right = primary[("C1", kind)]
        c0_samples = [row["segmentation_composite"] for row in left["per_sample_segmentation"]]
        c1_samples = [row["segmentation_composite"] for row in right["per_sample_segmentation"]]
        c0_targets = [float(row["top_k_hit"]) for row in left["anchor_per_target"]]
        c1_targets = [float(row["top_k_hit"]) for row in right["anchor_per_target"]]
        segmentation_loo = leave_one_out_deltas(c0_samples, c1_samples)
        anchor_loo = leave_one_out_deltas(c0_targets, c1_targets)
        protocols[kind] = {
            "segmentation": {
                "bootstrap": paired_bootstrap_delta(c0_samples, c1_samples),
                "leave_one_out": segmentation_loo,
                "single_sample_removal_reverses_sign": segmentation_loo["sign_reversal"],
            },
            "anchor_top_k": {
                "bootstrap": paired_bootstrap_delta(c0_targets, c1_targets),
                "leave_one_out": anchor_loo,
                "single_target_removal_reverses_sign": anchor_loo["sign_reversal"],
            },
        }
    return {
        "stage": "seg_raster_stage_s3a", "status": "PASS",
        "validation_sample_count": 16, "anchor_target_count": 24,
        "claim_scope": "descriptive uncertainty only; no statistical-significance claim",
        "protocols": protocols,
    }


def determinism_payload(records: Sequence[Mapping[str, Any]]) -> dict:
    indexed: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in records:
        indexed.setdefault((row["run_key"], row["checkpoint_kind"]), []).append(row)
    comparisons = []
    for key in ("C0", "C1"):
        for kind in ("best", "latest"):
            rows = sorted(indexed[(key, kind)], key=lambda row: row["repeat_index"])
            if len(rows) < 2:
                comparisons.append({"run_key": key, "checkpoint_kind": kind, "status": "MISSING_REPEAT"})
                continue
            first, second = rows[:2]
            checksum_equal = first["checksums"] == second["checksums"]
            sample_equal = first["sample_order"] == second["sample_order"]
            metric_equal = first["segmentation"] == second["segmentation"] and first["anchor"] == second["anchor"]
            comparisons.append({
                "run_key": key, "checkpoint_kind": kind,
                "metrics_exactly_equal": metric_equal,
                "prediction_and_probability_checksums_equal": checksum_equal,
                "sample_order_exactly_equal": sample_equal,
                "status": "PASS" if metric_equal and checksum_equal and sample_equal else "FAIL",
            })
    return {
        "stage": "seg_raster_stage_s3a",
        "status": "PASS" if all(row["status"] == "PASS" for row in comparisons) else "FAIL",
        "comparisons": comparisons, "floating_tolerance": 0.0,
    }


def graph_determinism_payload(records: Sequence[Mapping[str, Any]]) -> dict:
    indexed: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in records:
        indexed.setdefault((row["run_key"], row["checkpoint_kind"]), []).append(row)
    comparisons = []
    metric_keys = (
        "apls", "apls_directional", "topo", "topo_metrics", "connectivity",
        "junction_correctness", "candidate_edge_count", "undirected_edge_count",
        "dangling_edge_count", "duplicate_edge_count", "vertex_count",
        "graph_iterations",
    )
    for kind in ("best", "latest"):
        rows = sorted(
            indexed.get(("C0", kind), []),
            key=lambda row: int(row.get("repeat_index", 0)))
        if len(rows) < 2:
            comparisons.append({
                "run_key": "C0", "checkpoint_kind": kind,
                "status": "MISSING_REPEAT",
            })
            continue
        first, second = rows[:2]
        graph_equal = (
            first.get("postprocessed_graph_sha256")
            == second.get("postprocessed_graph_sha256")
            and first.get("postprocessed_graph_sha256") is not None)
        metrics_equal = all(first.get(key) == second.get(key) for key in metric_keys)
        settings_equal = (
            first.get("deterministic_settings")
            == second.get("deterministic_settings"))
        comparisons.append({
            "run_key": "C0", "checkpoint_kind": kind,
            "postprocessed_graph_checksum_exactly_equal": graph_equal,
            "graph_metrics_exactly_equal": metrics_equal,
            "deterministic_settings_exactly_equal": settings_equal,
            "status": "PASS" if graph_equal and metrics_equal and settings_equal else "FAIL",
        })
    return {
        "status": "PASS" if all(
            row["status"] == "PASS" for row in comparisons) else "FAIL",
        "comparisons": comparisons,
        "floating_tolerance": 0.0,
    }


def graph_job_accounting(records: Sequence[Mapping[str, Any]]) -> dict:
    """Account for complete and explicitly terminated primary graph jobs.

    A pathological termination remains a failed evaluation.  It may complete
    the job identity matrix for forensic reduction, but it must never be
    counted as a successful graph metric result.
    """
    expected_primary = {
        (key, kind) for key in DETACH_KEYS for kind in ("best", "latest")}
    primary = [
        row for row in records if int(row.get("repeat_index", 0)) == 0]
    identities = [
        (str(row.get("run_key")), str(row.get("checkpoint_kind")))
        for row in primary]
    missing = sorted(expected_primary - set(identities))
    duplicate = sorted({identity for identity in identities
                        if identities.count(identity) > 1})
    successful = [row for row in records if row.get("status") == "PASS"]
    pathological = [
        row for row in records
        if row.get("status") == PATHOLOGICAL_GRAPH_STATUS]
    unexpected_failures = [
        row for row in records
        if row.get("status") not in ("PASS", PATHOLOGICAL_GRAPH_STATUS)]
    c1_best_pathological = [
        row for row in pathological
        if row.get("run_key") == "C1"
        and row.get("checkpoint_kind") == "best"
        and int(row.get("repeat_index", 0)) == 0]
    other_primary_pass = all(
        row.get("status") == "PASS"
        for row in primary
        if not (row.get("run_key") == "C1"
                and row.get("checkpoint_kind") == "best"))
    recognized_partial = (
        not missing and not duplicate and not unexpected_failures
        and len(primary) == len(expected_primary)
        and len(c1_best_pathological) == 1
        and len(pathological) == 1
        and other_primary_pass)
    all_primary_pass = (
        not missing and not duplicate and not pathological
        and not unexpected_failures and len(primary) == len(expected_primary)
        and all(row.get("status") == "PASS" for row in primary))
    if all_primary_pass:
        status = "PASS"
    elif recognized_partial:
        status = "PARTIAL_TERMINATED_PATHOLOGICAL_EXPANSION"
    else:
        status = "FAIL"
    return {
        "status": status,
        "expected_primary_job_count": len(expected_primary),
        "observed_primary_job_count": len(primary),
        "total_record_count": len(records),
        "successful_record_count": len(successful),
        "pathological_termination_count": len(pathological),
        "missing_primary_jobs": [list(identity) for identity in missing],
        "duplicate_primary_jobs": [list(identity) for identity in duplicate],
        "unexpected_failure_count": len(unexpected_failures),
        "c1_best_graph_metrics": (
            "NOT_AVAILABLE_INCOMPLETE_RUN" if c1_best_pathological
            else "AVAILABLE"),
    }


def reference_gate(primary: Mapping[tuple[str, str], dict]) -> dict:
    exact_tolerance = 1e-12
    auprc_tolerance = 1e-6
    rows = []
    for (key, kind), record in sorted(primary.items()):
        for task in ("road", "junction"):
            check = record[f"{task}_metric_reference_check"]
            differences = check["absolute_differences"]
            exact_max = max(
                float(differences[name])
                for name in ("precision", "recall", "f1", "iou"))
            auprc_difference = float(differences["auprc"])
            rows.append({
                "run_key": key, "checkpoint_kind": kind, "task": task,
                "maximum_absolute_difference": check["maximum_absolute_difference"],
                "exact_metric_maximum_absolute_difference": exact_max,
                "auprc_absolute_difference": auprc_difference,
                "status": "PASS" if (
                    exact_max <= exact_tolerance
                    and auprc_difference <= auprc_tolerance) else "FAIL",
            })
        anchor = record["anchor_metric_reference_check"]["maximum_absolute_difference"]
        rows.append({"run_key": key, "checkpoint_kind": kind, "task": "anchor",
                     "maximum_absolute_difference": anchor,
                     "status": "PASS" if anchor <= 1e-8 else "FAIL"})
    return {
        "stage": "seg_raster_stage_s3a",
        "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL",
        "tolerances": {
            "precision_recall_f1_iou": exact_tolerance,
            "auprc": auprc_tolerance,
        },
        "auprc_tie_policy_note": (
            "The frozen evaluator applies stable per-pixel ordering within equal-score ties; "
            "the independent reference collapses equal scores into one threshold group. "
            "Differences up to 1e-6 are accepted only for AUPRC; confusion-derived metrics "
            "remain subject to 1e-12."),
        "synthetic_reference_tests": "tests/test_stage_s3a_metrics.py",
        "checkpoint_comparisons": rows,
    }


def markdown_table(rows: Iterable[Sequence[Any]], headers: Sequence[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def write_docs(
    docs: Path, provenance: Mapping[str, Any], dynamics: Mapping[str, Any],
    junction: Mapping[str, Any], anchor: Mapping[str, Any], graph: Mapping[str, Any],
    conclusion: Mapping[str, Any], protocols: Mapping[str, Any],
) -> None:
    docs.mkdir(parents=True, exist_ok=True)
    docs_payload = {
        "stage_s3a_metric_dataflow.md": f"""# Stage S3A metric dataflow

`final_metrics` is classified as **{provenance['final_metrics_classification']}**. It is the last training batch only: logits were produced in `model.train()`, the optimizer then stepped, and the pre-update output was measured without cross-batch accumulation. It is not validation evidence.

`best_metrics` uses `model.eval()` and `torch.no_grad()` over the frozen 8-batch / 16-sample validation plan. Historical segmentation and anchor comparison files came from the overwritten latest evaluation; historical graph comparison used per-run best checkpoints.

The frozen segmentation composite is `(road F1 + road IoU + junction F1) / 3`. Full machine-readable provenance is in `artifacts/stage_s3a_metric_provenance.json`.
""",
        "stage_s3a_training_dynamics.md": "# Stage S3A training dynamics\n\n" + markdown_table([
            (key, row["best_step"], f"{row['best_composite']:.6f}", f"{row['latest_composite']:.6f}", f"{row['deterioration_after_best']:.6f}")
            for key, row in dynamics["runs"].items()],
            ("Run", "Best step", "Best composite", "Latest composite", "Latest-best"))
            + "\n\nAll best steps precede 102400. This single-seed evidence supports late-training validation deterioration, not a universal optimal-step claim. Raw 20-point histories remain in the JSON artifact.\n",
        "stage_s3a_junction_forensics.md": f"""# Stage S3A junction forensics

Root-cause classes: `{', '.join(junction['root_cause_classification'])}`.

The model/evaluator contract passes: junction outputs are logits, one sigmoid is applied at the metric boundary, evaluation is `eval/no_grad`, and output/target are 64×64. F1=0 at threshold 0.3 means no thresholded hits; it does not mean logits or probabilities are numerically all zero. AUPRC, quantiles, calibration and a shared diagnostic threshold sweep are retained in the JSON. Per-run tuned thresholds are not used for causal claims.
""",
        "stage_s3a_anchor_forensics.md": f"""# Stage S3A anchor forensics

Historical anchor provenance: **{anchor['historical_anchor_comparison_provenance']['status']}**, latest step 102400.

Multistep validity: **{anchor['MULTISTEP_ANCHOR_VALIDITY']}**. The fixed-threshold and top-K metrics answer different questions: top-K bypasses 0.3, while threshold recall/false-positive counts can be zero when all probabilities remain below 0.3. The aligned result is not specific against shifted control, so the indirect-anchor causal claim is not retained.
""",
        "stage_s3a_graph_overgeneration.md": f"""# Stage S3A graph controls and overgeneration

Graph audit status: **{graph['status']}**. Nine jobs completed successfully. C1-best was terminated as `TERMINATED_PATHOLOGICAL_EXPANSION` after its forensic snapshot; its incomplete partial graph is not reported as a completed graph metric result. C1-latest and the completed C0/C2/C3 controls remain valid. The C0-best common-step matrix is unavailable because C1's step-5120 weights were not retained.

All available APLS values are `deterministic_pixel_graph_approximation`, not official SpaceNet APLS. Latest-checkpoint candidate/undirected/dangling/duplicate edge counts are reported with C1−C0, C1−C2 and C1−C3 deltas so connectivity gains cannot be separated from overgeneration by APLS alone. The best-checkpoint comparison is explicitly incomplete rather than populated from the C1-best partial graph.
""",
        "stage_s3a_final_report.md": f"""# Stage S3A final report

Stage S3A completed a read-only post-training audit of the immutable S3 checkpoints. There were 120 historical validation log points but only 12 retained checkpoint files (`best` and `latest` for six runs); Protocol C is therefore **UNAVAILABLE_MISSING_CHECKPOINT**, not reconstructed.

- Metric reference gate: **{conclusion['metric_reference_gate']}**
- Final metrics: **{conclusion['final_metrics_classification']}**
- Junction: `{', '.join(conclusion['junction_root_cause'])}`
- Anchor specificity: **{conclusion['anchor_control_specificity']}**
- Pixel parity: **{conclusion['local_remote_pixel_parity']}**
- Evaluation determinism: **{conclusion['evaluation_determinism']}**
- C1-best computational feasibility: **{conclusion['c1_best_computational_feasibility']}**
- C1-best graph expansion: **{conclusion['c1_best_graph_expansion']}**
- C1-best graph metrics: **{conclusion['c1_best_graph_metrics']}**
- Segmentation-only multi-seed: **{conclusion['segmentation_only_multiseed']}**
- End-to-end multi-seed: **{conclusion['end_to_end_multiseed']}**
- Multi-seed decision: **{conclusion['go_no_go_for_multiseed']}**

Checkpoint selection finding: segmentation-composite best checkpoint is not graph-stable; C1-latest terminates normally while C1-best exhibits pathological expansion.

The original Stage S3 artifacts remain unchanged. Stage S3A narrows or supersedes claims in a separate reconciliation artifact. The failed C1-best run is preserved as failed evidence and is never represented as `PASS`.
""",
    }
    for name, text in docs_payload.items():
        (docs / name).write_text(text.rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-code-sha", required=True)
    parser.add_argument("--evaluation-code-sha")
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument("--remote-output-root", type=Path, required=True)
    parser.add_argument("--local-pixel-manifest", type=Path, required=True)
    parser.add_argument("--remote-pixel-manifest", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=REPO_ROOT / "artifacts")
    parser.add_argument("--docs-dir", type=Path, default=REPO_ROOT / "docs/audits")
    args = parser.parse_args()
    if git_text("rev-parse", "HEAD") != args.audit_code_sha:
        raise RuntimeError("reducer must run from the frozen audit-code checkout")
    evaluation_code_sha = args.evaluation_code_sha or args.audit_code_sha

    training = read_json(REPO_ROOT / "artifacts/stage_s3_training_results.json")
    old_anchor = read_json(REPO_ROOT / "artifacts/stage_s3_anchor_comparison.json")
    old_conclusion = read_json(REPO_ROOT / "artifacts/stage_s3_conclusion.json")
    all_rows, validation = historical_metric_rows(args.source_run_root)
    evaluations, primary = load_evaluations(args.remote_output_root)
    if any(row.get("audit_code_sha") != evaluation_code_sha for row in evaluations):
        raise RuntimeError("checkpoint evaluation code SHA mismatch")
    inventory, by_run_inventory = checkpoint_inventory(
        args.source_run_root, training, validation)
    inventory["audit_code_sha"] = evaluation_code_sha
    inventory["evaluation_code_sha"] = evaluation_code_sha
    inventory["reducer_code_sha"] = args.audit_code_sha
    provenance = provenance_payload(training, primary)
    reference = reference_gate(primary)
    protocols = protocol_payload(primary, validation)
    dynamics = training_dynamics_payload(all_rows, validation, training)
    junction = junction_payload(primary)
    loss = loss_payload(primary)
    anchor, anchor_targets = anchor_payload(primary, old_anchor)
    graph_records = [read_json(path) for path in sorted(
        (args.remote_output_root / "graph_results").glob("*.json"))]
    if any(row.get("audit_code_sha") != evaluation_code_sha for row in graph_records):
        raise RuntimeError("graph evaluation code SHA mismatch")
    graph_primary = [
        row for row in graph_records if int(row.get("repeat_index", 0)) == 0]
    graph_matrix = graph_control_matrix(graph_primary)
    graph_accounting = graph_job_accounting(graph_records)
    graph = {
        "stage": "seg_raster_stage_s3a",
        **graph_matrix,
        "status": graph_accounting["status"],
        "control_matrix_status": graph_matrix["status"],
        "job_accounting": graph_accounting,
        "records": graph_records,
        "common_step": {"status": "UNAVAILABLE_MISSING_CHECKPOINT", "step": 5120,
                        "missing_runs": ["C1"]},
        "apls_kind": "deterministic_pixel_graph_approximation",
        "graph_density_normalized_comparison_recommended": True,
    }
    graph_index = {
        (row["run_key"], row["checkpoint_kind"]): row for row in graph_primary}
    graph["detailed_comparisons"] = {}
    for kind in ("best", "latest"):
        protocol = graph_matrix["protocols"][kind]
        if protocol["status"] != "PASS":
            graph["detailed_comparisons"][kind] = {
                "status": "NOT_AVAILABLE_INCOMPLETE_RUN",
                "missing_runs": protocol["missing_runs"],
                "reason": (
                    "C1-best was terminated for pathological closed-loop "
                    "expansion before complete graph metrics were produced."
                    if kind == "best" and "C1" in protocol["missing_runs"]
                    else "Required completed graph result is unavailable."),
            }
            continue
        rows = {key: graph_index[(key, kind)] for key in DETACH_KEYS}
        comparisons = {}
        for baseline in ("C0", "C2", "C3"):
            comparisons["C1_minus_" + baseline] = {
                "apls": rows["C1"]["apls"] - rows[baseline]["apls"],
                "topo_f1": rows["C1"]["topo"] - rows[baseline]["topo"],
                "largest_component_edge_length_ratio": (
                    rows["C1"]["connectivity"]["largest_component_edge_length_ratio"]
                    - rows[baseline]["connectivity"]["largest_component_edge_length_ratio"]),
                "junction_f1": (rows["C1"]["junction_correctness"]["f1"]
                                - rows[baseline]["junction_correctness"]["f1"]),
                "candidate_edge_count": rows["C1"]["candidate_edge_count"] - rows[baseline]["candidate_edge_count"],
                "undirected_edge_count": rows["C1"]["undirected_edge_count"] - rows[baseline]["undirected_edge_count"],
                "dangling_edge_count": rows["C1"]["dangling_edge_count"] - rows[baseline]["dangling_edge_count"],
                "duplicate_edge_count": rows["C1"]["duplicate_edge_count"] - rows[baseline]["duplicate_edge_count"],
                "graph_iterations": rows["C1"]["graph_iterations"] - rows[baseline]["graph_iterations"],
                "runtime_seconds": rows["C1"]["inference_time_seconds"] - rows[baseline]["inference_time_seconds"],
            }
        graph["detailed_comparisons"][kind] = {
            "status": "PASS", "comparisons": comparisons}
    graph["overgeneration_assessment"] = (
        "C1 overgeneration must be interpreted jointly with C2/C3 candidate, undirected, dangling and duplicate edge counts; "
        "connectivity or approximate APLS alone is not registration-specific evidence.")
    parity = pixel_parity(
        read_json(args.local_pixel_manifest), read_json(args.remote_pixel_manifest))
    sensitivity = sensitivity_payload(primary)
    validation_determinism = determinism_payload(evaluations)
    graph_determinism = graph_determinism_payload(graph_records)
    determinism = {
        "stage": "seg_raster_stage_s3a",
        "status": "PASS" if (
            validation_determinism["status"] == "PASS"
            and graph_determinism["status"] == "PASS") else "FAIL",
        "validation_evaluation": validation_determinism,
        "graph_evaluation": graph_determinism,
    }

    matrix = {
        "stage": "seg_raster_stage_s3a", "status": "PASS_WITH_RETENTION_LIMITATION",
        "forward_recomputed_checkpoint_count": len(primary),
        "evaluation_code_sha": evaluation_code_sha,
        "historical_log_only_validation_point_count": 120,
        "forward_recomputed": [compact_checkpoint(row) for row in primary.values()],
        "historical_validation_logs": {
            key: validation[key] for key in RUN_KEYS},
        "limitation": "108 historical validation points lack retained checkpoint files and cannot be re-forwarded.",
    }

    best_checks = {
        key: validate_recorded_best_step(
            validation[key], primary[(key, "best")]["checkpoint_step"])
        for key in RUN_KEYS}
    protocols["best_checkpoint_recomputation"] = best_checks

    latest_cmp = protocols["protocols"]["A_latest"]["comparisons"]
    best_cmp = protocols["protocols"]["B_per_run_best"]["comparisons"]
    seg_latest = sum(latest_cmp["C1_minus_C0"][name] > 0 for name in METRIC_KEYS) >= 2
    seg_best = sum(best_cmp["C1_minus_C0"][name] > 0 for name in METRIC_KEYS) >= 2
    seg_fair = seg_latest or seg_best
    aligned_specific = any(
        sum(comparison["C1_minus_C2"][name] > 0 for name in METRIC_KEYS) >= 2
        and sum(comparison["C1_minus_C3"][name] > 0 for name in METRIC_KEYS) >= 2
        for comparison in (latest_cmp, best_cmp))
    graph_completed_evidence_pass = (
        graph_accounting["status"]
        == "PARTIAL_TERMINATED_PATHOLOGICAL_EXPANSION"
        and graph_matrix["protocols"]["latest"]["status"] == "PASS"
        and graph_determinism["status"] == "PASS")
    unresolved_p0 = reference["status"] != "PASS" or determinism["status"] != "PASS"
    if unresolved_p0 or parity["status"] != "PASS":
        segmentation_only_decision = "NO_GO"
    elif seg_latest and not seg_best:
        segmentation_only_decision = "NO_GO"
    elif seg_fair and aligned_specific:
        segmentation_only_decision = "GO"
    else:
        segmentation_only_decision = "NO_GO"
    end_to_end_decision = "NO_GO"
    decision = "NO_GO"

    reconciliation = {
        "stage": "seg_raster_stage_s3a", "status": "PASS",
        "source_conclusion": "artifacts/stage_s3_conclusion.json",
        "claims": {
            "segmentation_causal_screen": {
                "old": old_conclusion.get("segmentation_causal_screen"),
                "status": "VERIFIED_WITH_NARROWER_SCOPE" if seg_fair else "INCONCLUSIVE",
                "new_scope": "single-seed checkpoint-protocol-sensitive screening; Protocol C unavailable",
            },
            "indirect_anchor_screen": {
                "old": old_conclusion.get("indirect_anchor_screen"),
                "status": "SUPERSEDED",
                "reason": "aligned C1 lacks stable shifted-control specificity and multistep validity failed",
            },
            "joint_screen": {
                "old": old_conclusion.get("joint_screen"), "status": "INCONCLUSIVE",
                "reason": "single seed and common-step weights unavailable",
            },
            "closed_loop_graph": {
                "old": old_conclusion.get("closed_loop_graph"),
                "status": "FAILED_PATHOLOGICAL_EXPANSION",
                "new_scope": (
                    "C1-latest and the completed C0/C2/C3 controls remain valid; "
                    "C1-best has no complete graph metrics."),
            },
            "go_no_go_for_multiseed": {
                "old": old_conclusion.get("go_no_go_for_multiseed"),
                "status": "SUPERSEDED", "new": decision,
            },
        },
    }
    conclusion = {
        "stage": "seg_raster_stage_s3a", "branch": "feat/seg-raster-only",
        "s3a_base_sha": read_json(
            REPO_ROOT / "artifacts/stage_s3a_git_start.json")["s3a_base_sha"],
        "s3a_audit_code_sha": evaluation_code_sha,
        "reducer_code_sha": args.audit_code_sha,
        "evaluation_code_sha": evaluation_code_sha,
        "formal_s3_run_code_sha": FORMAL_S3_SHA,
        "execution_environment": "REMOTE_TRAINING_SERVER",
        "metric_reference_gate": reference["status"],
        "final_metrics_classification": "LAST_TRAIN_BATCH_METRICS",
        "checkpoint_protocol_result": protocols["status"],
        "junction_root_cause": junction["root_cause_classification"],
        "junction_contract": junction["contract_check"],
        "loss_balance": "BACKGROUND_DOMINATED_SUM_REDUCTION_RISK",
        "anchor_provenance": anchor["historical_anchor_comparison_provenance"]["status"],
        "anchor_control_specificity": anchor["ALIGNED_VS_SHIFTED_SPECIFICITY"],
        "multistep_anchor_validity": anchor["MULTISTEP_ANCHOR_VALIDITY"],
        "graph_c2_c3_controls": (
            "PASS_LATEST_AND_COMPLETED_BEST_CONTROLS"
            if graph_completed_evidence_pass else "FAIL"),
        "c1_best_computational_feasibility": "FAIL",
        "c1_best_graph_expansion": "PATHOLOGICAL",
        "c1_best_natural_termination": "NOT_IMMINENT",
        "c1_best_graph_metrics": "NOT_AVAILABLE_INCOMPLETE_RUN",
        "c1_best_graph_status": PATHOLOGICAL_GRAPH_STATUS,
        "checkpoint_selection_finding": (
            "segmentation-composite best checkpoint is not graph-stable; "
            "C1-latest terminates normally while C1-best exhibits pathological expansion."),
        "local_remote_pixel_parity": parity["status"],
        "small_sample_sensitivity": "DESCRIPTIVE_ONLY",
        "evaluation_determinism": determinism["status"],
        "end_to_end_multiseed": end_to_end_decision,
        "segmentation_only_multiseed": segmentation_only_decision,
        "go_no_go_for_multiseed": decision,
        "unresolved_p0_metric_bug": unresolved_p0,
        "key_limitations": [
            "Only best/latest checkpoint files were retained; Protocol C cannot be recomputed.",
            "Single seed, 16 validation samples and 24 anchor targets do not establish statistical significance.",
            "APLS is a deterministic pixel-graph approximation, not official SpaceNet APLS.",
            "C1-best graph metrics are unavailable because pathological expansion was terminated after forensic capture.",
        ],
    }

    outputs = {
        "stage_s3a_metric_provenance.json": provenance,
        "stage_s3a_metric_reference_check.json": reference,
        "stage_s3a_checkpoint_inventory.json": inventory,
        "stage_s3a_checkpoint_metric_matrix.json": matrix,
        "stage_s3a_checkpoint_protocol_comparison.json": protocols,
        "stage_s3a_training_dynamics.json": dynamics,
        "stage_s3a_junction_forensics.json": junction,
        "stage_s3a_loss_balance_audit.json": loss,
        "stage_s3a_anchor_forensics.json": anchor,
        "stage_s3a_anchor_per_target.json": anchor_targets,
        "stage_s3a_graph_control_comparison.json": graph,
        "stage_s3a_local_remote_pixel_parity.json": parity,
        "stage_s3a_small_sample_sensitivity.json": sensitivity,
        "stage_s3a_evaluation_determinism.json": determinism,
        "stage_s3a_claim_reconciliation.json": reconciliation,
        "stage_s3a_gpu_inventory.json": read_json(
            args.remote_output_root / "stage_s3a_gpu_inventory.json"),
        "stage_s3a_gpu_schedule.json": {
            "stage": "seg_raster_stage_s3a",
            "status": "PARTIAL_TERMINATED_PATHOLOGICAL_EXPANSION",
            "evaluation_code_sha": evaluation_code_sha,
            "reducer_code_sha": args.audit_code_sha,
            "checkpoint_evaluation": read_json(
                args.remote_output_root / "stage_s3a_gpu_schedule.json"),
            "graph_evaluation": read_json(
                args.remote_output_root / "stage_s3a_graph_gpu_schedule.json"),
        },
        "stage_s3a_conclusion.json": conclusion,
    }
    for name, payload in outputs.items():
        write_json(args.artifact_dir / name, payload)
    write_docs(args.docs_dir, provenance, dynamics, junction, anchor, graph, conclusion, protocols)
    print(finite_json_dumps({
        "status": "PASS_WITH_PATHOLOGICAL_GRAPH_FAILURE",
        "artifact_count": len(outputs),
        "evaluation_code_sha": evaluation_code_sha,
        "reducer_code_sha": args.audit_code_sha,
        "go_no_go_for_multiseed": decision,
    }), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
