"""Stage S3E root-cause closure contracts.

The helpers in this module deliberately operate on the existing strict road
adapter.  They do not introduce a second fusion architecture.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


ROAD_HEAD_PREFIXES = ("road_seg.", "conv_road_final.")
ADAPTER_PREFIX = "zero_preserving_road_adapter."
ENCODER_PREFIX = ADAPTER_PREFIX + "encoder."
PROJECTION_PREFIX = ADAPTER_PREFIX + "projection."


def belongs_to(name: str, prefixes: Sequence[str]) -> bool:
    return name.startswith(tuple(prefixes))


def tensor_map_sha256(
    values: Mapping[str, torch.Tensor], *, prefixes: Sequence[str],
) -> str:
    digest = hashlib.sha256()
    matched = 0
    for name in sorted(values):
        if not belongs_to(name, prefixes):
            continue
        value = values[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
        matched += 1
    if not matched:
        raise ValueError("no tensors matched prefixes: {!r}".format(prefixes))
    return digest.hexdigest()


def copy_prefixed_state(
    destination: dict[str, torch.Tensor],
    source: Mapping[str, torch.Tensor],
    *, prefixes: Sequence[str],
) -> list[str]:
    changed = []
    for name in destination:
        if not belongs_to(name, prefixes):
            continue
        if name not in source:
            raise KeyError("source state lacks " + name)
        if tuple(destination[name].shape) != tuple(source[name].shape):
            raise ValueError("shape mismatch for " + name)
        destination[name] = source[name].detach().clone()
        changed.append(name)
    if not changed:
        raise ValueError("no state tensors selected for transplant")
    return changed


def build_cross_transplant_state(
    n0_state: Mapping[str, torch.Tensor],
    n1_state: Mapping[str, torch.Tensor],
    combination: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Build T00/T01/T10/T11 while preserving every unrelated tensor."""
    if combination not in ("T00", "T01", "T10", "T11"):
        raise ValueError("unknown transplant combination: " + combination)
    base = n0_state if combination in ("T00", "T01") else n1_state
    state = {name: value.detach().clone() for name, value in base.items()}
    changed: list[str] = []
    if combination == "T01":
        changed = copy_prefixed_state(
            state, n1_state, prefixes=(ADAPTER_PREFIX,))
    source = {
        "road_head": "N0" if combination in ("T00", "T01") else "N1",
        "adapter": ("N1" if combination in ("T01", "T11")
                    else "BYPASSED"),
        "raster_control": "aligned" if combination in ("T01", "T11") else "null",
    }
    audit = {
        "combination": combination,
        "sources": source,
        "transplanted_tensor_names": changed,
        "road_head_sha256": tensor_map_sha256(
            state, prefixes=ROAD_HEAD_PREFIXES),
        "adapter_sha256": tensor_map_sha256(
            state, prefixes=(ADAPTER_PREFIX,)),
        "backbone_sha256": tensor_map_sha256(
            state, prefixes=("stage_1.", "stage_2.", "stage_3.",
                             "stage_4.", "stage_5.", "conv_fuse.",
                             "conv_2_side.", "conv_3_side.",
                             "conv_4_side.", "conv_5_side.")),
    }
    return state, audit


def metric_decomposition(rows: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    required = ("T00", "T01", "T10", "T11")
    if any(key not in rows for key in required):
        raise ValueError("cross-transplant metrics are incomplete")
    names = (
        "road_f1", "road_iou", "road_auprc", "road_precision",
        "road_recall", "gt_mean_probability", "background_mean_probability",
    )
    result: dict[str, Any] = {}
    for name in names:
        t00, t01 = float(rows["T00"][name]), float(rows["T01"][name])
        t10, t11 = float(rows["T10"][name]), float(rows["T11"][name])
        result[name] = {
            "head_drift": t10 - t00,
            "adapter_on_clean_head": t01 - t00,
            "adapter_on_adapted_head": t11 - t10,
            "interaction": t11 - t10 - t01 + t00,
        }
    return result


def flat_named_tensors(
    state: Mapping[str, torch.Tensor], prefixes: Sequence[str],
) -> torch.Tensor:
    values = [state[name].detach().float().cpu().reshape(-1)
              for name in sorted(state) if belongs_to(name, prefixes)]
    if not values:
        raise ValueError("empty tensor selection")
    return torch.cat(values)


def vector_comparison(
    left: torch.Tensor, right: torch.Tensor,
) -> dict[str, float]:
    left = left.detach().double().reshape(-1).cpu()
    right = right.detach().double().reshape(-1).cpu()
    delta = right - left
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    delta_norm = torch.linalg.vector_norm(delta)
    denominator = left_norm * right_norm
    cosine = ((left @ right) / denominator).item() if denominator > 0 else 1.0
    return {
        "left_l2": float(left_norm), "right_l2": float(right_norm),
        "delta_l2": float(delta_norm),
        "normalized_delta_l2": float(delta_norm / left_norm.clamp_min(1e-30)),
        "cosine": float(cosine),
    }


def parameter_drift(
    initial: Mapping[str, torch.Tensor],
    n0: Mapping[str, torch.Tensor],
    n1: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    excluded_suffixes = (".running_mean", ".running_var", ".num_batches_tracked")
    initial = {name: value for name, value in initial.items()
               if not name.endswith(excluded_suffixes)}
    n0 = {name: value for name, value in n0.items()
          if not name.endswith(excluded_suffixes)}
    n1 = {name: value for name, value in n1.items()
          if not name.endswith(excluded_suffixes)}
    groups = {
        "road_head_all": ROAD_HEAD_PREFIXES,
        "road_seg": ("road_seg.",),
        "conv_road_final": ("conv_road_final.",),
        "road_head_bn_affine": (
            "road_seg.0.bn.", "road_seg.1.bn."),
    }
    result = {}
    for key, prefixes in groups.items():
        base = flat_named_tensors(initial, prefixes)
        left = flat_named_tensors(n0, prefixes)
        right = flat_named_tensors(n1, prefixes)
        result[key] = {
            "n1_vs_n0": vector_comparison(left, right),
            "n0_vs_initial": vector_comparison(base, left),
            "n1_vs_initial": vector_comparison(base, right),
            "n0_sha256": tensor_map_sha256(n0, prefixes=prefixes),
            "n1_sha256": tensor_map_sha256(n1, prefixes=prefixes),
        }
    return result


def configure_stage_s3e_training(
    model: nn.Module, *, road_head_lr: float,
    freeze_encoder: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Freeze every unrelated parameter and return Adam parameter groups."""
    contract, head, adapter = [], [], []
    for name, parameter in model.named_parameters():
        is_head = belongs_to(name, ROAD_HEAD_PREFIXES)
        is_adapter = name.startswith(ADAPTER_PREFIX)
        frozen_encoder = freeze_encoder and name.startswith(ENCODER_PREFIX)
        trainable = (is_adapter and not frozen_encoder) or (
            is_head and road_head_lr > 0)
        parameter.requires_grad_(trainable)
        if trainable:
            (head if is_head else adapter).append(parameter)
        contract.append({
            "name": name, "requires_grad": trainable,
            "optimizer_included": trainable,
            "group": "road_head" if is_head else (
                "raster_adapter" if is_adapter else "frozen"),
            "frozen_encoder": bool(frozen_encoder),
        })
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    groups = []
    if head:
        groups.append({"params": head, "lr": float(road_head_lr),
                       "name": "road_head"})
    if adapter:
        groups.append({"params": adapter, "name": "raster_adapter"})
    if not groups:
        raise ValueError("Stage S3E optimizer has no parameters")
    return contract, groups


def weighted_road_loss(
    logits: torch.Tensor, target: torch.Tensor, *,
    negative_weight: float = 1.0, scale: float = 1.0,
) -> torch.Tensor:
    if negative_weight <= 0 or scale <= 0:
        raise ValueError("loss weights must be positive")
    element = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weights = torch.where(target > 0.5, torch.ones_like(target),
                          target.new_full((), float(negative_weight)))
    return element.mul(weights).sum().mul(float(scale))


def named_gradient_vector(
    model: nn.Module, prefixes: Sequence[str],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    per_name = {}
    for name, parameter in model.named_parameters():
        if not belongs_to(name, prefixes):
            continue
        value = (torch.zeros_like(parameter) if parameter.grad is None
                 else parameter.grad).detach().float().cpu().reshape(-1)
        per_name[name] = value
    if not per_name:
        raise ValueError("no gradients selected")
    return torch.cat([per_name[name] for name in sorted(per_name)]), per_name


def gradient_comparison(
    null: torch.Tensor, aligned: torch.Tensor,
) -> dict[str, float]:
    left = null.double().reshape(-1)
    right = aligned.double().reshape(-1)
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    denominator = left_norm * right_norm
    cosine = ((left @ right) / denominator).item() if denominator > 0 else 1.0
    return {
        "null_l2": float(left_norm), "aligned_l2": float(right_norm),
        "dot_product": float(left @ right), "cosine": float(cosine),
    }


def layerwise_gradient_comparison(
    null: Mapping[str, torch.Tensor], aligned: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    if set(null) != set(aligned):
        raise ValueError("gradient tensor names differ")
    return {name: gradient_comparison(null[name], aligned[name])
            for name in sorted(null)}


def optimizer_parameter_delta(
    before: Mapping[str, torch.Tensor], model: nn.Module,
    prefixes: Sequence[str],
) -> dict[str, Any]:
    after = dict(model.named_parameters())
    selected_before = {name: value for name, value in before.items()
                       if belongs_to(name, prefixes)}
    selected_after = {name: after[name].detach() for name in selected_before}
    return {
        "comparison": vector_comparison(
            flat_named_tensors(selected_before, prefixes),
            flat_named_tensors(selected_after, prefixes)),
        "before_sha256": tensor_map_sha256(selected_before, prefixes=prefixes),
        "after_sha256": tensor_map_sha256(selected_after, prefixes=prefixes),
    }


def clone_named_parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()}


def finite_tree(value: Any) -> None:
    if isinstance(value, Mapping):
        for child in value.values():
            finite_tree(child)
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        for child in value:
            finite_tree(child)
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise ValueError("non-finite value")
