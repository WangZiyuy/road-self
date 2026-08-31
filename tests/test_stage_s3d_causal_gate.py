from __future__ import annotations

from utils.seg_raster.stage_s3d import (
    select_n0_common_samples, spatial_causal_gate, validate_commit_paths)


def _metrics(offset: float) -> dict:
    return {
        "road_precision": 0.7 + offset, "road_recall": 0.7 + offset,
        "road_f1": 0.7 + offset, "road_iou": 0.6 + offset,
        "road_auprc": 0.8 + offset,
        "junction_f1": 0.4, "junction_auprc": 0.5,
        "per_sample": [
            {"road_f1": 0.60 + offset + index * .001,
             "road_iou": 0.50 + offset + index * .001,
             "road_auprc": 0.70 + offset + index * .001}
            for index in range(8)],
    }


def test_common_samples_are_selected_from_n0_with_early_near_tie() -> None:
    rows = {
        "2560": _metrics(0.0), "5120": _metrics(0.0004),
        "7680": _metrics(-0.1),
    }
    assert select_n0_common_samples(rows) == 2560


def test_gate_requires_aligned_to_beat_every_control() -> None:
    rows = {"N0": _metrics(0), "N1": _metrics(.05),
            "N2": _metrics(0), "N3": _metrics(-.01),
            "N4": _metrics(-.02)}
    passed = spatial_causal_gate(
        rows, null_parity="PASS",
        three_checkpoint_direction=[True, True, True],
        input_swap_aligned_best=True, junction_parity=True)
    assert passed["status"] == "PASS"
    rows["N4"] = _metrics(.06)
    failed = spatial_causal_gate(
        rows, null_parity="PASS",
        three_checkpoint_direction=[True, True, True],
        input_swap_aligned_best=True, junction_parity=True)
    assert failed["status"] == "FAIL"


def test_commit_manifest_rejects_runtime_binaries() -> None:
    validate_commit_paths(["artifacts/stage_s3d_model_contract.json"])
    for path in ("data_self/run/summary.json", "checkpoints/model.pth.tar",
                 "artifacts/view.png"):
        try:
            validate_commit_paths([path])
        except ValueError:
            pass
        else:
            raise AssertionError(path)
