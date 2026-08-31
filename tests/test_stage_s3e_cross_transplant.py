from __future__ import annotations

import inspect

import torch

from utils.seg_raster.stage_s3e import (
    ADAPTER_PREFIX, ROAD_HEAD_PREFIXES, build_cross_transplant_state,
    metric_decomposition, parameter_drift, tensor_map_sha256)


def _state(offset: float) -> dict[str, torch.Tensor]:
    return {
        "stage_1.weight": torch.tensor([1.0]),
        "road_seg.0.weight": torch.tensor([2.0 + offset]),
        "road_seg.0.bias": torch.tensor([3.0 + offset]),
        "road_seg.0.bn.weight": torch.tensor([1.0 + offset]),
        "road_seg.0.bn.bias": torch.tensor([0.5 + offset]),
        "road_seg.1.bn.weight": torch.tensor([1.5 + offset]),
        "road_seg.1.bn.bias": torch.tensor([0.25 + offset]),
        "road_seg.0.running_mean": torch.tensor([99.0 + offset]),
        "conv_road_final.weight": torch.tensor([4.0 + offset]),
        "zero_preserving_road_adapter.encoder.0.weight":
            torch.tensor([5.0 + offset]),
        "zero_preserving_road_adapter.projection.weight":
            torch.tensor([6.0 + offset]),
    }


def test_t00_t01_t10_t11_submodule_loading() -> None:
    n0, n1 = _state(0), _state(10)
    expected = {
        "T00": ("N0", "BYPASSED"), "T01": ("N0", "N1"),
        "T10": ("N1", "BYPASSED"), "T11": ("N1", "N1"),
    }
    for key, (head, adapter) in expected.items():
        state, audit = build_cross_transplant_state(n0, n1, key)
        assert audit["sources"]["road_head"] == head
        assert audit["sources"]["adapter"] == adapter
        assert torch.equal(state["stage_1.weight"], n0["stage_1.weight"])


def test_adapter_swap_does_not_change_road_head_or_backbone() -> None:
    n0, n1 = _state(0), _state(10)
    state, _ = build_cross_transplant_state(n0, n1, "T01")
    assert tensor_map_sha256(state, prefixes=ROAD_HEAD_PREFIXES) == tensor_map_sha256(
        n0, prefixes=ROAD_HEAD_PREFIXES)
    assert tensor_map_sha256(state, prefixes=(ADAPTER_PREFIX,)) == tensor_map_sha256(
        n1, prefixes=(ADAPTER_PREFIX,))
    assert torch.equal(state["stage_1.weight"], n0["stage_1.weight"])


def test_road_head_selection_does_not_change_adapter() -> None:
    n0, n1 = _state(0), _state(10)
    state, _ = build_cross_transplant_state(n0, n1, "T10")
    assert tensor_map_sha256(state, prefixes=ROAD_HEAD_PREFIXES) == tensor_map_sha256(
        n1, prefixes=ROAD_HEAD_PREFIXES)
    assert tensor_map_sha256(state, prefixes=(ADAPTER_PREFIX,)) == tensor_map_sha256(
        n1, prefixes=(ADAPTER_PREFIX,))


def test_metric_decomposition_matches_factorial_definition() -> None:
    rows = {}
    for key, value in {"T00": 1.0, "T01": 1.2, "T10": 0.7, "T11": 0.8}.items():
        rows[key] = {name: value for name in (
            "road_f1", "road_iou", "road_auprc", "road_precision",
            "road_recall", "gt_mean_probability", "background_mean_probability")}
    result = metric_decomposition(rows)["road_auprc"]
    assert abs(result["head_drift"] + 0.3) < 1e-12
    assert abs(result["adapter_on_clean_head"] - 0.2) < 1e-12
    assert abs(result["adapter_on_adapted_head"] - 0.1) < 1e-12
    assert abs(result["interaction"] + 0.1) < 1e-12


def test_parameter_drift_excludes_batch_norm_running_buffers() -> None:
    initial, n0, n1 = _state(0), _state(0), _state(1)
    n1["road_seg.0.running_mean"] = torch.tensor([1e9])
    result = parameter_drift(initial, n0, n1)
    assert result["road_seg"]["n1_vs_n0"]["delta_l2"] < 10
    assert result["road_head_bn_affine"]["n1_vs_n0"]["delta_l2"] > 0


def test_cross_transplant_evaluator_verifies_checkpoint_provenance() -> None:
    from tools.seg_raster import evaluate_stage_s3e_cross_transplant
    source = inspect.getsource(evaluate_stage_s3e_cross_transplant)
    assert "actual != expected[run][samples][\"sha256\"]" in source
    assert "source_checkpoint_sha256" in source
    assert "source_stage_s3d_run_code_sha" in source
