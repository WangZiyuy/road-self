from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tools.seg_raster.audit_stage_s3a_graph import prepare_config
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
    assert result["historical_anchor_comparison_provenance"] == {
        "status": "PASS",
        "checkpoint_kind": "latest",
        "checkpoint_step": 102400,
        "evidence": "historical evaluation/anchor.json was overwritten at each validation; values match latest protocol",
        "historical_artifact": "artifacts/stage_s3_anchor_comparison.json",
    }
    assert targets["rows"] == []


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
