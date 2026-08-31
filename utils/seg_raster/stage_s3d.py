"""Pure contracts for Stage S3D raster-specificity experiments."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .stage_s3c import (
    GRAPH_CAPS, MAX_SAMPLES_SEEN, SAMPLE_GRID,
    strict_load_official_checkpoint,
)


STAGE_S3D_SEED = 20260827
STRICT_MODE = "raster_road_zero_preserving"
ADAPTER_PREFIX = "zero_preserving_road_adapter."
ROAD_PREFIXES = ("road_seg.", "conv_road_final.")
CONTROLS = ("null", "aligned", "zero", "shift_large", "permuted")
INPUT_SWAP_CONTROLS = (
    "aligned", "zero", "shift_fixed", "shift_large", "permuted", "all_one")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("utf-8")).hexdigest()


def array_sha256(array: np.ndarray | torch.Tensor) -> str:
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().contiguous().numpy()
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def named_tensor_sha256(
    values: Sequence[tuple[str, torch.Tensor | None]],
) -> str:
    digest = hashlib.sha256()
    for name, tensor in values:
        digest.update(name.encode("utf-8"))
        if tensor is None:
            digest.update(b"<NONE>")
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def road_parameter_sha256(model: nn.Module) -> str:
    return named_tensor_sha256([
        (name, parameter) for name, parameter in model.named_parameters()
        if name.startswith(ROAD_PREFIXES)
    ])


def trainable_gradient_sha256(model: nn.Module) -> str:
    return named_tensor_sha256([
        (name, parameter.grad) for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ])


def translate_zero_fill(array: np.ndarray, shift_xy: tuple[int, int]) -> np.ndarray:
    """Translate the last two dimensions without circular wrap."""
    source = np.asarray(array)
    result = np.zeros_like(source)
    shift_x, shift_y = map(int, shift_xy)
    height, width = source.shape[-2:]
    src_x0, src_x1 = max(0, -shift_x), min(width, width - shift_x)
    src_y0, src_y1 = max(0, -shift_y), min(height, height - shift_y)
    if src_x1 > src_x0 and src_y1 > src_y0:
        result[..., src_y0 + shift_y:src_y1 + shift_y,
               src_x0 + shift_x:src_x1 + shift_x] = source[
                   ..., src_y0:src_y1, src_x0:src_x1]
    return np.ascontiguousarray(result)


def density_strata(values: Sequence[float], strata: int = 4) -> list[int]:
    values_np = np.asarray(values, dtype=np.float64)
    if values_np.ndim != 1 or not len(values_np):
        raise ValueError("density values must be a non-empty vector")
    if strata < 1:
        raise ValueError("strata must be positive")
    if len(values_np) == 1:
        return [0]
    boundaries = np.quantile(values_np, np.linspace(0, 1, strata + 1)[1:-1])
    return np.searchsorted(boundaries, values_np, side="right").astype(int).tolist()


def density_stratified_derangement(
    positive_ratios: Sequence[float], *, seed: int = STAGE_S3D_SEED,
) -> list[int]:
    """Create a deterministic no-self donor mapping with density preference."""
    ratios = np.asarray(positive_ratios, dtype=np.float64)
    count = len(ratios)
    if count < 2:
        raise ValueError("a derangement requires at least two samples")
    strata = density_strata(ratios)
    rng = np.random.default_rng(seed)
    mapping = [-1] * count
    singleton_indices = []
    for stratum in sorted(set(strata)):
        group = [index for index, value in enumerate(strata)
                 if value == stratum]
        if len(group) == 1:
            singleton_indices.extend(group)
            continue
        shift = int(rng.integers(1, len(group)))
        for position, index in enumerate(group):
            mapping[index] = group[(position + shift) % len(group)]
    if singleton_indices:
        # Quantile ties can create singleton strata.  Fall back to the best
        # global cyclic derangement so the mapping remains a true permutation.
        offsets = list(range(1, count))
        rng.shuffle(offsets)
        best = min(offsets, key=lambda offset: (
            sum(strata[i] != strata[(i + offset) % count]
                for i in range(count)),
            sum(abs(ratios[i] - ratios[(i + offset) % count])
                for i in range(count)),
            offset,
        ))
        mapping = [(index + best) % count for index in range(count)]
    if any(index == donor for index, donor in enumerate(mapping)):
        raise AssertionError("permutation contains a self-match")
    return mapping


def permute_rasters(
    rasters: np.ndarray, mapping: Sequence[int],
) -> np.ndarray:
    array = np.asarray(rasters)
    if array.shape[0] != len(mapping):
        raise ValueError("mapping length differs from raster sample count")
    if sorted(map(int, mapping)) != list(range(len(mapping))):
        raise ValueError("mapping is not a permutation")
    if any(index == int(donor) for index, donor in enumerate(mapping)):
        raise ValueError("permutation contains a self-match")
    return np.ascontiguousarray(array[np.asarray(mapping, dtype=np.int64)])


def tensor_statistics(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().float().cpu().contiguous()
    flat = value.reshape(-1)
    spatial_variance = (
        value.var(dim=(-2, -1), unbiased=False).mean()
        if value.ndim >= 2 else value.new_zeros(()))
    return {
        "shape": list(value.shape), "min": float(flat.min()),
        "max": float(flat.max()), "mean": float(flat.mean()),
        "std": float(flat.std(unbiased=False)),
        "l1_norm": float(flat.abs().sum()),
        "l2_norm": float(torch.linalg.vector_norm(flat)),
        "spatial_variance": float(spatial_variance),
        "nonzero_ratio": float(torch.count_nonzero(flat) / flat.numel()),
        "sha256": array_sha256(value),
    }


def configure_road_only_training(model: nn.Module) -> list[dict[str, Any]]:
    contract = []
    for name, parameter in model.named_parameters():
        trainable = name.startswith(ROAD_PREFIXES + (ADAPTER_PREFIX,))
        parameter.requires_grad_(trainable)
        contract.append({
            "name": name, "requires_grad": trainable,
            "optimizer_included": trainable,
            "gradient_source": (
                "road_segmentation_loss" if trainable else "FROZEN"),
        })
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    return contract


def set_road_only_train_mode(model: nn.Module) -> None:
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def road_only_parameters(model: nn.Module) -> list[nn.Parameter]:
    values = [parameter for parameter in model.parameters()
              if parameter.requires_grad]
    if not values:
        raise ValueError("road-only optimizer has no parameters")
    return values


def road_loss(output: Mapping[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.binary_cross_entropy_with_logits(
        output["road"], target, reduction="sum")


def strict_load_stage_s3d_baseline(
    model: nn.Module, payload: Mapping[str, Any],
) -> dict[str, Any]:
    return strict_load_official_checkpoint(
        model, payload, allowed_new_prefixes=(ADAPTER_PREFIX,))


def road_composite(metrics: Mapping[str, float]) -> float:
    return float(np.mean([
        metrics["road_f1"], metrics["road_iou"], metrics["road_auprc"]]))


def select_n0_common_samples(
    metrics_by_samples: Mapping[int | str, Mapping[str, float]],
    *, near_tie: float = 0.001,
) -> int:
    rows = sorted((int(key), road_composite(value))
                  for key, value in metrics_by_samples.items())
    if not rows:
        raise ValueError("N0 validation grid is empty")
    best = max(score for _, score in rows)
    return min(samples for samples, score in rows if best - score < near_tie)


def null_parity_audit(
    n0: Mapping[str, Any], n2: Mapping[str, Any],
    *, atol: float = 0.0,
) -> dict[str, Any]:
    fields = (
        "shared_trainable_tensor_sha256", "road_prediction_sha256",
        "junction_prediction_sha256", "road_feature_sha256",
        "gradient_sha256",
    )
    differences = {}
    passed = True
    for field in fields:
        equal = n0.get(field) == n2.get(field)
        differences[field] = {"equal": equal, "n0": n0.get(field),
                              "n2": n2.get(field)}
        passed &= equal
    for field in ("road_f1", "road_iou", "road_auprc",
                  "junction_f1", "junction_auprc"):
        left, right = float(n0[field]), float(n2[field])
        delta = abs(left - right)
        differences[field] = {"abs_difference": delta,
                              "within_tolerance": delta <= atol}
        passed &= delta <= atol
    return {"status": "PASS" if passed else "FAIL",
            "absolute_tolerance": atol, "comparisons": differences}


def paired_sensitivity(
    baseline_rows: Sequence[Mapping[str, float]],
    aligned_rows: Sequence[Mapping[str, float]],
    *, seed: int = STAGE_S3D_SEED,
) -> dict[str, Any]:
    if len(baseline_rows) != len(aligned_rows) or len(baseline_rows) < 2:
        return {"status": "INCONCLUSIVE", "reason": "INSUFFICIENT_SAMPLES"}
    values = np.asarray([
        np.mean([aligned_rows[index][name] - baseline_rows[index][name]
                 for name in ("road_f1", "road_iou", "road_auprc")])
        for index in range(len(baseline_rows))], dtype=np.float64)
    leave_one_out = [float(np.mean(np.delete(values, index)))
                     for index in range(len(values))]
    rng = np.random.default_rng(seed)
    bootstrap = [float(np.mean(values[
        rng.integers(0, len(values), len(values))])) for _ in range(2000)]
    low, high = np.percentile(bootstrap, [2.5, 97.5])
    passed = low > 0 and min(leave_one_out) > 0
    return {
        "status": "PASS" if passed else "FAIL",
        "paired_mean_delta": float(np.mean(values)),
        "bootstrap_95_percentile_interval": [float(low), float(high)],
        "leave_one_out_min_delta": min(leave_one_out),
        "sample_count": len(values),
    }


def spatial_causal_gate(
    metrics: Mapping[str, Mapping[str, Any]], *,
    null_parity: str, three_checkpoint_direction: Sequence[bool],
    input_swap_aligned_best: bool, junction_parity: bool,
) -> dict[str, Any]:
    required = ("N0", "N1", "N2", "N3", "N4")
    if any(key not in metrics for key in required):
        return {"status": "INCONCLUSIVE", "reason": "MISSING_RUN"}
    n1 = metrics["N1"]
    names = ("road_f1", "road_iou", "road_auprc")
    strict_beats = {
        key: all(float(n1[name]) > float(metrics[key][name])
                 for name in names)
        for key in ("N0", "N2", "N3", "N4")}
    improved_vs_n0 = sum(
        float(n1[name]) > float(metrics["N0"][name]) for name in names)
    collapse = (
        float(n1["road_precision"]) < float(metrics["N0"]["road_precision"]) - 0.10
        or float(n1["road_recall"]) < float(metrics["N0"]["road_recall"]) - 0.10)
    sensitivity = paired_sensitivity(
        metrics["N0"]["per_sample"], n1["per_sample"])
    passed = (
        null_parity == "PASS" and all(strict_beats.values())
        and improved_vs_n0 >= 2 and sensitivity["status"] == "PASS"
        and len(three_checkpoint_direction) >= 3
        and all(three_checkpoint_direction) and not collapse
        and input_swap_aligned_best and junction_parity)
    return {
        "status": "PASS" if passed else "FAIL",
        "aligned_strictly_beats_each_control_all_three_metrics": strict_beats,
        "metrics_improved_vs_n0": improved_vs_n0,
        "paired_sensitivity": sensitivity,
        "three_checkpoint_direction": list(three_checkpoint_direction),
        "precision_recall_collapse": collapse,
        "input_swap_aligned_best": input_swap_aligned_best,
        "junction_parity": junction_parity,
    }


def classify_current_zero_path(
    *, image_enters_trainable_fusion: bool,
    valid_mask_enters_trainable_fusion: bool,
    normalization_affine: bool,
    bias_present: bool,
    runtime_residual_nonzero: bool,
) -> str:
    causes = []
    if bias_present:
        causes.append("ZERO_PATH_BIAS")
    if normalization_affine:
        causes.append("NORMALIZATION_AFFINE_PATH")
    if image_enters_trainable_fusion:
        causes.append("IMAGE_SIDE_EXTRA_CAPACITY")
    if valid_mask_enters_trainable_fusion:
        causes.append("VALID_MASK_CONSTANT_PATH")
    if not runtime_residual_nonzero:
        return "ZERO_PATH_ALREADY_STRICT" if not causes else "UNRESOLVED"
    if len(causes) == 1:
        return causes[0]
    if len(causes) > 1:
        return "MULTIPLE_CAUSES"
    return "UNRESOLVED"


def assert_json_finite(value: Any) -> None:
    if isinstance(value, Mapping):
        for child in value.values():
            assert_json_finite(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            assert_json_finite(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON contains NaN or Infinity")


def validate_commit_paths(paths: Sequence[str]) -> None:
    forbidden_parts = {
        "data_self", "checkpoints", "datasets", "tensorboard", "cache",
        "__pycache__", ".pytest_cache",
    }
    forbidden_suffixes = (
        ".pth", ".pth.tar", ".pt", ".ckpt", ".png", ".tif", ".tiff")
    for raw in paths:
        path = Path(raw)
        if any(part.lower() in forbidden_parts for part in path.parts):
            raise ValueError("forbidden Stage S3D commit path: " + raw)
        if raw.lower().endswith(forbidden_suffixes):
            raise ValueError("forbidden Stage S3D binary: " + raw)


def model_contract_payload() -> dict[str, Any]:
    return {
        "stage": "seg_raster_stage_s3d", "status": "PASS",
        "mode": STRICT_MODE, "max_samples_seen": MAX_SAMPLES_SEEN,
        "checkpoint_grid": list(SAMPLE_GRID), "seed": STAGE_S3D_SEED,
        "road_only": True, "junction_strict_image_only": True,
        "raw_raster_direct_to_anchor": False,
        "zero_property": "F(stage_fuse_img, zero)=stage_fuse_img",
        "graph_caps": dict(GRAPH_CAPS),
    }
