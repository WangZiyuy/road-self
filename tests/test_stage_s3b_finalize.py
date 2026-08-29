import argparse
import json
from pathlib import Path

from tools.seg_raster.finalize_stage_s3b import (
    LR_GATE_NOT_EXECUTED, finalize_lr_stability_failure)


STEPS = list(range(0, 20481, 2560))


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _summary(run_key: str, code_sha: str, *, raster: bool) -> dict:
    metrics = {
        str(step): {
            "road_f1": 0.1, "road_iou": 0.05, "junction_auprc": 0.02,
            "checkpoint_step": step, "optimizer_learning_rate": 1e-5}
        for step in STEPS}
    return {
        "run_key": run_key, "run_id": run_key + "_seed20260827",
        "status": "PASS", "code_sha": code_sha, "optimizer_steps": 20480,
        "elapsed_seconds": 1.0, "validation_metrics_by_step": metrics,
        "best_validation_metrics": metrics["0"],
        "latest_validation_metrics": metrics["20480"],
        "last_train_batch_metrics": {"scope": "train"},
        "checkpoint_inventory": [{"step": step} for step in STEPS],
        "first_100_batch_identity_sha256": "identity",
        "first_100_common_tensor_sha256": "common",
        "first_100_valid_mask_sha256": "mask" if raster else None}


def test_lr_gate_failure_finalizer_does_not_require_later_phases(tmp_path):
    run_code_sha = "a" * 40
    reducer_code_sha = "b" * 40
    run_root = tmp_path / "runs"
    output_root = tmp_path / "out"
    jobs = []
    for index in range(6):
        key = "A{}".format(index)
        run_id = key + "_seed20260827"
        jobs.append({"run_key": key, "run_id": run_id})
        _write(run_root / run_id / "summary.json",
               _summary(key, run_code_sha, raster=index % 2 == 1))
    _write(output_root / "gpu_inventory_phase_A.json", {"status": "PASS"})
    _write(output_root / "gpu_schedule_phase_A.json", {"status": "PASS"})
    lr = {
        "selection": {"lr_stability_gate": "FAIL",
                      "selected_lr_multiplier": 0.1},
        "image_only_candidates": [
            {"run_key": "A0", "lr_multiplier": 1.0,
             "best_repair_composite": 0.1, "retention": 0.2},
            {"run_key": "A2", "lr_multiplier": 0.3,
             "best_repair_composite": 0.1, "retention": 0.3},
            {"run_key": "A4", "lr_multiplier": 0.1,
             "best_repair_composite": 0.1, "retention": 0.4}]}
    args = argparse.Namespace(run_code_sha=run_code_sha,
                              reducer_code_sha=reducer_code_sha,
                              run_root=run_root, output_root=output_root)

    assert finalize_lr_stability_failure(args, lr, {"jobs": jobs}) == 0

    conclusion = json.loads(
        (output_root / "stage_s3b_conclusion.json").read_text())
    assert conclusion["training_protocol_repair"] == "FAIL"
    assert conclusion["phase_b_status"] == LR_GATE_NOT_EXECUTED
    assert conclusion["go_for_segmentation_multi_seed"] == "NO_GO"
    assert conclusion["go_for_end_to_end_multi_seed"] == "NO_GO"
    assert conclusion["s3b_run_code_sha"] == run_code_sha
    assert conclusion["reducer_code_sha"] == reducer_code_sha
    for name in ("stage_s3b_junction_loss_screen.json",
                 "stage_s3b_segmentation_comparison.json",
                 "stage_s3b_anchor_comparison.json",
                 "stage_s3b_graph_comparison.json"):
        payload = json.loads((output_root / name).read_text())
        assert payload["status"] == LR_GATE_NOT_EXECUTED
