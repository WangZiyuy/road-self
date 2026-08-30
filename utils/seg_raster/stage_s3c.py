"""Contracts for Stage S3C frozen-explorer raster adaptation.

This module deliberately contains no remote orchestration.  It centralizes the
strict official-checkpoint policy, parameter/BatchNorm freeze contract, sample
budget, control selection, and graph caps so they can be tested on CPU before
formal remote execution.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn


STAGE_S3C_SEED = 20260827
MAX_SAMPLES_SEEN = 40960
MICRO_BATCH_PER_GPU = 10
GRADIENT_ACCUMULATION = 2
EFFECTIVE_SAMPLES_PER_UPDATE = 20
SAMPLE_GRID = (
    0, 2560, 5120, 7680, 10240, 12800, 15360, 17920, 20480,
    25600, 30720, 35840, 40960,
)
GRAPH_CAPS = {
    "max_iterations": 3000,
    "max_vertices": 5000,
    "max_directed_edges": 10000,
    "max_wall_time_seconds": 900,
}
OFFICIAL_SOURCE_SHA = "ffcb47e50e48ced717b2ac0e0f8c720ffc083441"
OFFICIAL_STATE_KEY_COUNT = 648
TRAJECTORY_KEY_MARKERS = (
    "trajectory", "transformer", "fuse_module_traj", "dsf",
    "traj_to_img", "cross_attention", "stage_1_traj",
    "segmentation_raster_fusion",
)
TRAINABLE_BASE_PREFIXES = (
    "road_seg.", "conv_road_final.", "junc_seg.", "conv_junc_final.",
)
RASTER_PREFIX = "segmentation_raster_fusion."


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def extract_state_dict(payload: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    state = payload.get("state_dict", payload)
    if not isinstance(state, Mapping) or not state:
        raise ValueError("checkpoint does not contain a non-empty state_dict")
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("checkpoint state_dict contains non-tensor values")
    return state


def _strip_or_add_module_prefix(
    state: Mapping[str, torch.Tensor], model_keys: Sequence[str]
) -> tuple[dict[str, torch.Tensor], str]:
    loaded_keys = list(state)
    model_keys = list(model_keys)
    loaded_module = bool(loaded_keys) and all(
        key.startswith("module.") for key in loaded_keys)
    model_module = bool(model_keys) and all(
        key.startswith("module.") for key in model_keys)
    if loaded_module and not model_module:
        return ({key[len("module."):]: value for key, value in state.items()},
                "STRIP_MODULE_PREFIX")
    if model_module and not loaded_module:
        return ({"module." + key: value for key, value in state.items()},
                "ADD_MODULE_PREFIX")
    return dict(state), "UNCHANGED"


def strict_load_official_checkpoint(
    model: nn.Module,
    payload: Mapping[str, Any],
    *,
    allowed_new_prefixes: Sequence[str] = (),
) -> dict[str, Any]:
    """Load all shared keys exactly, preserving only explicitly new modules.

    The final call is strict=True.  Missing keys are permitted only before the
    merge and only when every one belongs to an explicitly allowed new module.
    This prevents a permissive ``strict=False`` load from hiding shared-key
    incompatibility.
    """
    expected = model.state_dict()
    loaded_raw = extract_state_dict(payload)
    loaded, prefix_policy = _strip_or_add_module_prefix(
        loaded_raw, list(expected))
    expected_keys, loaded_keys = set(expected), set(loaded)
    shared = expected_keys & loaded_keys
    missing = sorted(expected_keys - loaded_keys)
    unexpected = sorted(loaded_keys - expected_keys)
    shape_mismatch = sorted(
        key for key in shared
        if tuple(expected[key].shape) != tuple(loaded[key].shape))
    disallowed_missing = [
        key for key in missing
        if not any(key.startswith(prefix) for prefix in allowed_new_prefixes)
    ]
    trajectory_keys = sorted(
        key for key in loaded
        if any(marker in key.lower() for marker in TRAJECTORY_KEY_MARKERS))
    if unexpected or shape_mismatch or disallowed_missing or trajectory_keys:
        raise ValueError(json.dumps({
            "unexpected_keys": unexpected,
            "shape_mismatch_keys": shape_mismatch,
            "disallowed_missing_keys": disallowed_missing,
            "trajectory_related_keys": trajectory_keys,
        }, sort_keys=True))
    merged = {key: value for key, value in expected.items()}
    merged.update(loaded)
    result = model.load_state_dict(merged, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("strict merged checkpoint load failed")
    return {
        "checkpoint_state_key_count": len(loaded),
        "model_state_key_count": len(expected),
        "shared_key_count": len(shared),
        "missing_new_key_count": len(missing),
        "missing_new_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatch_keys": shape_mismatch,
        "trajectory_related_key_count": len(trajectory_keys),
        "data_parallel_prefix_policy": prefix_policy,
        "strict_merged_load": True,
    }


def is_trainable_parameter(name: str, *, raster_enabled: bool) -> bool:
    return (
        any(name.startswith(prefix) for prefix in TRAINABLE_BASE_PREFIXES)
        or (raster_enabled and name.startswith(RASTER_PREFIX))
    )


def expected_gradient_source(name: str, *, raster_enabled: bool) -> str:
    if any(name.startswith(prefix) for prefix in TRAINABLE_BASE_PREFIXES):
        return "road_or_junction_segmentation_loss"
    if raster_enabled and name.startswith(RASTER_PREFIX):
        return "road_and_junction_segmentation_losses"
    return "FROZEN_NO_BACKWARD"


def configure_frozen_explorer(
    model: nn.Module, *, raster_enabled: bool
) -> list[dict[str, Any]]:
    """Freeze the original explorer and expose only segmentation parameters."""
    contract = []
    for name, parameter in model.named_parameters():
        trainable = is_trainable_parameter(name, raster_enabled=raster_enabled)
        parameter.requires_grad_(trainable)
        contract.append({
            "name": name,
            "module": name.rsplit(".", 1)[0] if "." in name else name,
            "requires_grad": trainable,
            "optimizer_included": trainable,
            "expected_gradient_source": expected_gradient_source(
                name, raster_enabled=raster_enabled),
        })
    enforce_original_batch_norm_eval(model)
    return contract


def is_original_batch_norm(name: str, module: nn.Module) -> bool:
    return (
        isinstance(module, nn.modules.batchnorm._BatchNorm)
        and not name.startswith(RASTER_PREFIX.rstrip("."))
    )


def enforce_original_batch_norm_eval(model: nn.Module) -> None:
    for name, module in model.named_modules():
        if is_original_batch_norm(name, module):
            module.eval()


def set_frozen_explorer_train_mode(model: nn.Module) -> None:
    model.train()
    enforce_original_batch_norm_eval(model)


def original_batch_norm_checksum(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, module in model.named_modules():
        if not is_original_batch_norm(name, module):
            continue
        for buffer_name in ("running_mean", "running_var", "num_batches_tracked"):
            value = getattr(module, buffer_name, None)
            if value is None:
                continue
            tensor = value.detach().cpu().contiguous()
            digest.update((name + "." + buffer_name).encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(bytes(tensor.numpy().tobytes()))
    return digest.hexdigest()


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    parameters = [parameter for parameter in model.parameters()
                  if parameter.requires_grad]
    if not parameters:
        raise ValueError("frozen-explorer optimizer has no trainable parameters")
    return parameters


def segmentation_losses(
    output: Mapping[str, torch.Tensor],
    road_target: torch.Tensor,
    junction_target: torch.Tensor,
    *,
    junction_pos_weight: float = 1.0,
    junction_alpha: float = 1.0,
) -> dict[str, torch.Tensor]:
    road = F.binary_cross_entropy_with_logits(
        output["road"], road_target, reduction="sum")
    weight = torch.as_tensor(
        float(junction_pos_weight), dtype=output["junc"].dtype,
        device=output["junc"].device)
    junction = float(junction_alpha) * F.binary_cross_entropy_with_logits(
        output["junc"], junction_target, pos_weight=weight, reduction="sum")
    return {"road": road, "junction": junction, "total": road + junction}


def checkpoint_name(samples_seen: int) -> str:
    samples_seen = int(samples_seen)
    if samples_seen not in SAMPLE_GRID:
        raise ValueError("samples_seen is outside the frozen checkpoint grid")
    return "samples_{:06d}.pth.tar".format(samples_seen)


@dataclass
class SampleBudgetCounter:
    micro_batch_size: int = MICRO_BATCH_PER_GPU
    accumulation_steps: int = GRADIENT_ACCUMULATION
    micro_batches: int = 0
    optimizer_updates: int = 0
    samples_seen: int = 0

    def record_micro_batch(self, actual_size: int) -> bool:
        if int(actual_size) != self.micro_batch_size:
            raise ValueError("micro-batch size differs from frozen contract")
        self.micro_batches += 1
        self.samples_seen += int(actual_size)
        should_step = self.micro_batches % self.accumulation_steps == 0
        if should_step:
            self.optimizer_updates += 1
        return should_step


def repair_composite(metrics: Mapping[str, float]) -> float:
    return float((float(metrics["road_f1"]) + float(metrics["road_iou"])
                  + float(metrics["junction_auprc"])) / 3.0)


def select_phase_a_loss(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not candidates or any(row.get("input_kind") != "image_only"
                             for row in candidates):
        raise ValueError("Phase A loss selection accepts image-only runs only")
    by_kind = {str(row["loss_kind"]): row for row in candidates}
    legacy = by_kind.get("legacy_exact")
    balanced = by_kind.get("class_balanced_bce")
    if legacy is None or balanced is None:
        raise ValueError("Phase A requires legacy and balanced BCE controls")
    def eligible(row: Mapping[str, Any]) -> bool:
        metrics = row["best_metrics"]
        legacy_metrics = legacy["best_metrics"]
        return (
            row.get("status") == "PASS"
            and bool(row.get("finite", True))
            and float(metrics["road_auprc"])
                >= float(legacy_metrics["road_auprc"]) - 0.01
            and float(metrics["junction_f1"])
                >= float(legacy_metrics["junction_f1"])
            and not bool(row.get("single_sample_driven", False))
        )
    selected = balanced if (
        eligible(balanced)
        and float(balanced["best_metrics"]["junction_auprc"])
            > float(legacy["best_metrics"]["junction_auprc"])
        and repair_composite(balanced["best_metrics"])
            > repair_composite(legacy["best_metrics"])
    ) else legacy
    return {
        "status": "PASS",
        "selected_run": selected["run_key"],
        "selected_loss_kind": selected["loss_kind"],
        "selection_scope": [row["run_key"] for row in candidates],
        "raster_results_read_for_selection": False,
    }


def select_baseline_controlled_common_samples(
    r0_validation: Mapping[int | str, Mapping[str, float]],
    *,
    near_tie: float = 0.001,
) -> int:
    if not r0_validation:
        raise ValueError("R0 validation grid is empty")
    rows = sorted((int(step), repair_composite(metrics))
                  for step, metrics in r0_validation.items())
    best = max(score for _, score in rows)
    return min(step for step, score in rows if best - score < float(near_tie))


@dataclass(frozen=True)
class GraphResourceSnapshot:
    iterations: int
    vertices: int
    directed_edges: int
    elapsed_seconds: float


def graph_resource_status(
    snapshot: GraphResourceSnapshot,
    caps: Mapping[str, int | float] = GRAPH_CAPS,
) -> dict[str, Any]:
    reached = []
    if snapshot.iterations >= int(caps["max_iterations"]):
        reached.append("MAX_GRAPH_ITERATIONS")
    if snapshot.vertices >= int(caps["max_vertices"]):
        reached.append("MAX_GRAPH_VERTICES")
    if snapshot.directed_edges >= int(caps["max_directed_edges"]):
        reached.append("MAX_DIRECTED_EDGES")
    if snapshot.elapsed_seconds >= float(caps["max_wall_time_seconds"]):
        reached.append("MAX_GRAPH_WALL_TIME_SECONDS")
    return {
        "status": "RESOURCE_CAP_REACHED" if reached else "WITHIN_RESOURCE_CAP",
        "reached_caps": reached,
        "natural_termination": False if reached else None,
        "caps": dict(caps),
    }


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
    forbidden_suffixes = (
        ".pth", ".pth.tar", ".pt", ".ckpt", ".tif", ".tiff", ".png",
    )
    forbidden_parts = {
        "data_self", "datasets", "checkpoints", "tensorboard", "cache",
        "__pycache__", ".pytest_cache",
    }
    for raw in paths:
        path = Path(raw)
        if any(part.lower() in forbidden_parts for part in path.parts):
            raise ValueError("forbidden Stage S3C commit path: " + raw)
        if raw.lower().endswith(forbidden_suffixes):
            raise ValueError("forbidden Stage S3C binary artifact: " + raw)
