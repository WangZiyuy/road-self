from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tools.seg_raster.audit_stage_s3a_graph import (
    prepare_config,
    select_postprocessed_graph,
    set_graph_seed,
)
from tools.seg_raster.audit_stage_s3a_metrics import (
    graph_control_matrix,
    leave_one_out_deltas,
    paired_bootstrap_delta,
    protocol_checkpoint,
    select_best_validation_record,
    validate_recorded_best_step,
)
from tools.seg_raster.audit_stage_s3a_reduce import (
    anchor_payload,
    checkpoint_inventory,
    graph_determinism_payload,
    graph_job_accounting,
    reference_gate,
    stable_sha,
)
from utils.seg_raster.stage_s3 import EXPERIMENT_MATRIX


def validation(step: int, composite: float) -> dict:
    return {
        "kind": "frozen_validation",
        "step": step,
        "metrics": {"segmentation_composite": composite},
    }


def test_best_checkpoint_recomputation_prefers_earliest_exact_tie() -> None:
    rows = [validation(5120, 0.7), validation(10240, 0.7), validation(15360, 0.6)]
    assert select_best_validation_record(rows)["step"] == 5120
    assert validate_recorded_best_step(rows, 5120)["status"] == "PASS"
    assert validate_recorded_best_step(rows, 10240)["status"] == "FAIL"


def test_common_step_protocol_reports_missing_checkpoint_instead_of_substitution() -> None:
    inventory = {
        "C0": [{"kind": "best", "step": 5120, "sha256": "a"}],
        "C1": [{"kind": "best", "step": 10240, "sha256": "b"},
               {"kind": "latest", "step": 102400, "sha256": "c"}],
    }
    assert protocol_checkpoint(inventory, "C0", 5120)["status"] == "AVAILABLE"
    result = protocol_checkpoint(inventory, "C1", 5120)
    assert result == {
        "status": "UNAVAILABLE_MISSING_CHECKPOINT",
        "run_key": "C1",
        "requested_step": 5120,
    }


def test_graph_control_matrix_requires_all_detach_controls_for_each_protocol() -> None:
    rows = [
        {"status": "PASS", "run_key": key, "checkpoint_kind": kind,
         "apls": index / 10, "topo": index / 20}
        for kind in ("best", "latest")
        for index, key in enumerate(("C0", "C1", "C2", "C3"))
    ]
    matrix = graph_control_matrix(rows)
    assert matrix["status"] == "PASS"
    assert set(matrix["protocols"]["best"]["c1_deltas"]) == {"C0", "C2", "C3"}
    incomplete = graph_control_matrix(rows[:-1])
    assert incomplete["status"] == "INCOMPLETE"
    assert incomplete["protocols"]["latest"]["missing_runs"] == ["C3"]


def test_deterministic_bootstrap_and_leave_one_out_sign_sensitivity() -> None:
    left = [0.0, 0.0, 0.0, 0.0]
    right = [1.0, 1.0, -1.0, 1.0]
    first = paired_bootstrap_delta(left, right, seed=7, iterations=1000)
    second = paired_bootstrap_delta(left, right, seed=7, iterations=1000)
    assert first == second
    assert first["mean_delta"] == pytest.approx(0.5)
    loo = leave_one_out_deltas(left, right)
    assert len(loo["leave_one_out_deltas"]) == 4
    assert loo["sign_reversal"] is False


def test_reducer_stable_sha_is_key_order_independent() -> None:
    assert stable_sha({"a": 1, "b": [2, 3]}) == stable_sha({"b": [2, 3], "a": 1})


def test_checkpoint_inventory_distinguishes_logs_from_retained_files(tmp_path) -> None:
    training = {"runs": {}}
    histories = {}
    for spec in EXPERIMENT_MATRIX:
        checkpoint_dir = tmp_path / spec.run_id / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "best.pth.tar").write_bytes((spec.key + "best").encode())
        (checkpoint_dir / "latest.pth.tar").write_bytes((spec.key + "latest").encode())
        histories[spec.key] = [validation(5, 0.9), validation(10, 0.8), validation(15, 0.7)]
        training["runs"][spec.key] = {"optimizer_steps": 15}
    result, per_run = checkpoint_inventory(tmp_path, training, histories)
    assert result["actual_checkpoint_file_count"] == 12
    assert result["historical_validation_log_record_count"] == 18
    assert result["missing_historical_checkpoint_file_count"] == 6
    assert all(len(per_run[key]) == 2 for key in per_run)


def _checkpoint_record(key: str, kind: str) -> dict:
    metrics = {
        "per_step_recall": [0.0, 0.0, 0.0, 0.0],
        "top_k_recall": 0.25,
        "localization_error": 10.0,
        "false_positive_count": 0,
        "missed_branch_count": 3,
        "channel_diversity_mean_absolute_difference": 0.0,
    }
    return {
        "checkpoint_step": 5 if kind == "best" else 15,
        "checkpoint_sha256": key + kind,
        "anchor": metrics,
        "anchor_metric_reference_check": {"maximum_absolute_difference": 0.0},
        "anchor_per_target": [],
        "segmentation": {
            "road": {"f1": 0.1, "iou": 0.05},
            "junction": {"f1": 0.0},
            "segmentation_composite": 0.05,
        },
    }


def test_anchor_checkpoint_provenance_is_explicit() -> None:
    primary = {(key, kind): _checkpoint_record(key, kind)
               for key in ("C0", "C1", "C2", "C3", "J0", "J1")
               for kind in ("best", "latest")}
    result, targets = anchor_payload(primary, {"runs": {}})
    provenance = result["historical_anchor_comparison_provenance"]
    assert provenance["status"] == "PASS"
    assert provenance["checkpoint_kind"] == "latest"
    assert provenance["checkpoint_step"] == 102400
    assert provenance["historical_vs_recomputed_latest_maximum_absolute_difference"] == 0.0
    assert targets["rows"] == []


def test_reference_gate_uses_strict_confusion_tolerance_and_documented_auprc_tie_tolerance() -> None:
    primary = {}
    for key in ("C0", "C1", "C2", "C3", "J0", "J1"):
        for kind in ("best", "latest"):
            row = _checkpoint_record(key, kind)
            segmentation_check = {
                "absolute_differences": {
                    "precision": 0.0, "recall": 0.0, "f1": 0.0,
                    "iou": 0.0, "auprc": 4.7e-7,
                },
                "maximum_absolute_difference": 4.7e-7,
            }
            row["road_metric_reference_check"] = segmentation_check
            row["junction_metric_reference_check"] = segmentation_check
            primary[(key, kind)] = row
    result = reference_gate(primary)
    assert result["status"] == "PASS"
    assert result["tolerances"]["auprc"] == 1e-6
    primary[("C0", "best")]["road_metric_reference_check"][
        "absolute_differences"]["f1"] = 1e-9
    assert reference_gate(primary)["status"] == "FAIL"


def test_anchor_forensics_flags_detach_multistep_failure_and_latest_false_positive_scope() -> None:
    primary = {(key, kind): _checkpoint_record(key, kind)
               for key in ("C0", "C1", "C2", "C3", "J0", "J1")
               for kind in ("best", "latest")}
    primary[("J1", "best")]["anchor"]["per_step_recall"] = [0.5, 0.2, 0.25, 1 / 3]
    primary[("C1", "best")]["anchor"]["false_positive_count"] = 3
    payload, _ = anchor_payload(primary, {"runs": {}})
    assert payload["MULTISTEP_ANCHOR_VALIDITY"] == "FAIL"
    diagnosis = payload["multistep_diagnosis"]
    assert diagnosis["j1_best_has_a_later_step_hit"] is True
    assert diagnosis[
        "fixed_threshold_produced_no_false_positive_pixels_for_all_latest_checkpoints"] is True
    assert diagnosis[
        "fixed_threshold_produced_no_false_positive_pixels_for_all_available_checkpoints"] is False


def test_graph_seed_restarts_python_numpy_and_torch_rngs() -> None:
    import random
    import numpy as np
    import torch

    first_settings = set_graph_seed(73)
    first = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    second_settings = set_graph_seed(73)
    second = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert first_settings == second_settings
    assert first == second


def test_graph_determinism_requires_equal_postprocessed_graph_and_metrics() -> None:
    records = []
    for kind in ("best", "latest"):
        for repeat in (0, 1):
            records.append({
                "run_key": "C0", "checkpoint_kind": kind,
                "repeat_index": repeat,
                "postprocessed_graph_sha256": kind + "-sha",
                "deterministic_settings": {"seed": 1},
                "apls": 0.2, "apls_directional": {}, "topo": 0.1,
                "topo_metrics": {}, "connectivity": {},
                "junction_correctness": {}, "candidate_edge_count": 1,
                "undirected_edge_count": 1, "dangling_edge_count": 0,
                "duplicate_edge_count": 0, "vertex_count": 2,
                "graph_iterations": 3,
            })
    assert graph_determinism_payload(records)["status"] == "PASS"
    records[-1]["postprocessed_graph_sha256"] = "different"
    assert graph_determinism_payload(records)["status"] == "FAIL"


def test_graph_accounting_preserves_pathological_c1_best_as_failure() -> None:
    records = []
    for kind in ("best", "latest"):
        for key in ("C0", "C1", "C2", "C3"):
            status = (
                "TERMINATED_PATHOLOGICAL_EXPANSION"
                if key == "C1" and kind == "best" else "PASS")
            records.append({
                "run_key": key, "checkpoint_kind": kind,
                "repeat_index": 0, "status": status,
            })
    for kind in ("best", "latest"):
        records.append({
            "run_key": "C0", "checkpoint_kind": kind,
            "repeat_index": 1, "status": "PASS",
        })
    accounting = graph_job_accounting(records)
    assert accounting["status"] == (
        "PARTIAL_TERMINATED_PATHOLOGICAL_EXPANSION")
    assert accounting["successful_record_count"] == 9
    assert accounting["pathological_termination_count"] == 1
    assert accounting["c1_best_graph_metrics"] == (
        "NOT_AVAILABLE_INCOMPLETE_RUN")

    records[1]["status"] = "USER_CANCELLED"
    accounting = graph_job_accounting(records)
    assert accounting["status"] == "FAIL"
    assert accounting["unexpected_failure_count"] == 1


def test_graph_audit_resolves_inherited_stage_s3_config(tmp_path) -> None:
    output = tmp_path / "graph_output"
    (output / "checkpoint").mkdir(parents=True)
    (output / "checkpoint" / "selected.pth.tar").write_bytes(b"placeholder")
    (output / "graph_trajectory").mkdir(parents=True)
    (output / "graph_trajectory" / "xian.png").write_bytes(b"placeholder")
    source_runtime = tmp_path / "runtime"
    (source_runtime / "controls" / "aligned").mkdir(parents=True)
    args = SimpleNamespace(
        base_config=Path("configs/stage_s3_C0_image_detach.yml"),
        run_key="C0",
        checkpoint=tmp_path / "source.pth.tar",
        checkpoint_kind="best",
        output_dir=output,
        source_runtime=source_runtime,
    )
    path = prepare_config(args, checkpoint_step=5120)
    with path.open("r", encoding="utf-8") as handle:
        resolved = yaml.safe_load(handle)
    assert "DIR" in resolved
    assert resolved["TRAIN"]["NUM_TARGETS"] == 4
    assert resolved["TEST"]["CROP_SZ"] == 256
    assert resolved["TEST"]["CPU_WORKER"] == 5
    assert resolved["TEST"]["BATCH_SIZE_ANCHOR"] == 15
    assert resolved["DIR"]["PRE_JUNC_NMS_DIR"] == "data_self/input/junction_nms/"


def test_graph_audit_selects_postprocessed_graph_not_raw_graph(tmp_path) -> None:
    raw = tmp_path / "graphs" / "selected_4" / "graphs_junc" / "xian.graph"
    post = tmp_path / "graphs" / "selected_4" / "post" / "xian.graph"
    raw.parent.mkdir(parents=True)
    post.parent.mkdir(parents=True)
    raw.write_text("raw", encoding="utf-8")
    post.write_text("post", encoding="utf-8")
    assert select_postprocessed_graph(tmp_path) == post
