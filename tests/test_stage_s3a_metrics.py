from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools.seg_raster.audit_stage_s3a_metrics import (
    anchor_reference_metrics,
    array_sha256,
    average_precision_reference,
    binary_reference_metrics,
    calibration_metrics,
    classify_final_metrics_scope,
    detect_double_sigmoid_input,
    finite_json_dumps,
    pixel_array_record,
    threshold_sweep,
)


def logits_from_probability(values: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(values, dtype=np.float64), 1e-9, 1 - 1e-9)
    return np.log(probability / (1.0 - probability))


@pytest.mark.parametrize(
    ("probability", "target", "expected"),
    [
        ([0.1, 0.2], [0, 0], (0.0, 0.0, 0.0, 0.0)),
        ([0.8, 0.9], [1, 1], (1.0, 1.0, 1.0, 1.0)),
        ([0.9, 0.1], [1, 0], (1.0, 1.0, 1.0, 1.0)),
        ([0.1, 0.9], [1, 0], (0.0, 0.0, 0.0, 0.0)),
        ([0.1, 0.2], [1, 0], (0.0, 0.0, 0.0, 0.0)),
    ],
)
def test_reference_binary_known_answers(probability, target, expected) -> None:
    result = binary_reference_metrics(
        logits_from_probability(np.asarray(probability)), np.asarray(target),
        threshold=0.5)
    assert tuple(result[key] for key in ("precision", "recall", "f1", "iou")) == expected
    assert result["predicted_positive_count"] == sum(value >= 0.5 for value in probability)
    assert result["target_positive_count"] == sum(target)


def test_reference_ap_handles_perfect_opposite_sparse_and_no_target() -> None:
    assert average_precision_reference(np.array([0.9, 0.8, 0.1]), np.array([1, 1, 0])) == 1.0
    assert average_precision_reference(np.array([0.9, 0.8, 0.1]), np.array([0, 0, 1])) == pytest.approx(1 / 3)
    sparse = np.zeros(1000, dtype=np.int64)
    sparse[777] = 1
    probability = np.linspace(0, 1, 1000)
    assert 0 < average_precision_reference(probability, sparse) <= 1
    assert average_precision_reference(np.array([0.9]), np.array([0])) == 0.0


def test_micro_aggregation_is_not_macro_average() -> None:
    probability = np.array([[0.9] * 10, [0.1] * 10])
    target = np.array([[1] * 10, [0] * 9 + [1]])
    logits = logits_from_probability(probability)
    micro = binary_reference_metrics(logits, target, threshold=0.5)["f1"]
    macro = np.mean([
        binary_reference_metrics(logits[i:i + 1], target[i:i + 1], threshold=0.5)["f1"]
        for i in range(2)
    ])
    assert micro != macro


def test_threshold_sweep_and_calibration_include_no_prediction_case() -> None:
    logits = logits_from_probability(np.array([0.01, 0.02, 0.03]))
    target = np.array([0, 1, 0])
    sweep = threshold_sweep(logits, target, [0.001, 0.5])
    assert sweep[0]["predicted_positive_count"] == 3
    assert sweep[1]["predicted_positive_count"] == 0
    calibration = calibration_metrics(logits, target, bin_count=5)
    assert 0 <= calibration["brier_score"] <= 1
    assert calibration["prevalence"] == pytest.approx(1 / 3)


def test_double_sigmoid_detector_distinguishes_logits_from_probability() -> None:
    assert detect_double_sigmoid_input(np.array([-2.0, 3.0]))["double_sigmoid_risk"] is False
    assert detect_double_sigmoid_input(np.array([0.1, 0.9]))["double_sigmoid_risk"] is True


def test_anchor_false_positive_topk_and_per_target_accounting() -> None:
    target = np.zeros((1, 4, 8, 8), dtype=np.float32)
    target[0, 0, 3, 3] = 1
    logits = np.full_like(target, -10)
    logits[0, 0, 3, 3] = 10
    logits[0, 0, 0, 0] = 10
    metrics, rows = anchor_reference_metrics(logits, target, [1], threshold=0.3, top_k=2)
    assert metrics["per_step_recall"] == [1.0, 0.0, 0.0, 0.0]
    assert metrics["false_positive_count"] == 1
    assert metrics["top_k_recall"] == 1.0
    assert metrics["top_k_bypasses_fixed_threshold"] is True
    assert len(rows) == metrics["evaluated_target_count"] == 1
    assert rows[0]["target_index"] == 0


def test_final_metrics_scope_classification_is_evidence_driven() -> None:
    assert classify_final_metrics_scope({
        "dataset_split": "train", "batch_count": 1,
        "cross_batch_accumulation": False,
    }) == "LAST_TRAIN_BATCH_METRICS"
    assert classify_final_metrics_scope({
        "dataset_split": "validation", "batch_count": 8,
    }) == "VALID_VALIDATION_METRICS"
    assert classify_final_metrics_scope({"target_prediction_alias": True}) == (
        "TARGET_PREDICTION_ALIAS_BUG")


def test_pixel_array_hash_is_decoded_content_hash(tmp_path: Path) -> None:
    array = np.arange(36, dtype=np.uint8).reshape(6, 6)
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.fromarray(array).save(first, compress_level=0)
    Image.fromarray(array).save(second, compress_level=9)
    left = pixel_array_record(first)
    right = pixel_array_record(second)
    assert left["byte_sha256"] != right["byte_sha256"]
    assert left["decoded_pixel_array_sha256"] == right["decoded_pixel_array_sha256"]
    assert left["decoded_pixel_array_sha256"] == array_sha256(array)


def test_json_finite_serialization_rejects_nan() -> None:
    assert json.loads(finite_json_dumps({"value": 1.0})) == {"value": 1.0}
    with pytest.raises(ValueError):
        finite_json_dumps({"value": float("nan")})
