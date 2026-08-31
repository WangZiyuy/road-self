from __future__ import annotations

from utils.seg_raster.stage_s3d import null_parity_audit


def _row() -> dict:
    return {
        "shared_trainable_tensor_sha256": "a",
        "road_prediction_sha256": "b",
        "junction_prediction_sha256": "c",
        "road_feature_sha256": "d",
        "gradient_sha256": "e",
        "road_f1": 0.1, "road_iou": 0.2, "road_auprc": 0.3,
        "junction_f1": 0.4, "junction_auprc": 0.5,
    }


def test_exact_null_parity_passes_and_any_mismatch_fails() -> None:
    row = _row()
    assert null_parity_audit(row, dict(row))["status"] == "PASS"
    changed = dict(row, road_prediction_sha256="different")
    assert null_parity_audit(row, changed)["status"] == "FAIL"
