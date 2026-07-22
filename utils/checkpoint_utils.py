"""Checkpoint naming, metadata, saving, and loading for road_self."""

from __future__ import annotations

import os
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from utils.trajectory_mode import resolve_trajectory_mode


_MISSING = object()
_WARNED_INFERENCE_CONFLICTS: set[tuple[str, str]] = set()


def _get_value(container: Any, key: str, default: Any = _MISSING) -> Any:
    if container is None:
        return default
    if isinstance(container, Mapping):
        return container.get(key, default)
    return getattr(container, key, default)


def _checkpoint_directory(cfg: Any) -> Path:
    directory = _get_value(_get_value(cfg, "DIR", None), "CHECK_POINT_DIR")
    if directory is _MISSING or not str(directory).strip():
        raise ValueError("DIR.CHECK_POINT_DIR must be configured")
    return Path(os.path.expanduser(str(directory)))


def _resolve_under_checkpoint_dir(value: str, checkpoint_dir: Path) -> Path:
    path = Path(os.path.expanduser(value))
    return path if path.is_absolute() else checkpoint_dir / path


def _legacy_inference_path(test_cfg: Any, checkpoint_dir: Path) -> Path | None:
    legacy_name = _get_value(test_cfg, "CKPT", _MISSING)
    if legacy_name is _MISSING or not str(legacy_name).strip():
        return None
    value = str(legacy_name)
    if not value.endswith(".pth.tar"):
        value += ".pth.tar"
    return _resolve_under_checkpoint_dir(value, checkpoint_dir)


@dataclass(frozen=True)
class TrainingCheckpointPaths:
    versioned: Path
    latest: Path | None


def has_training_checkpoint_config(cfg: Any) -> bool:
    train_cfg = _get_value(cfg, "TRAIN", None)
    return _get_value(train_cfg, "CHECKPOINT", _MISSING) is not _MISSING


def resolve_training_checkpoint_paths(
    cfg: Any,
    *,
    outer_it: int,
    path_it: int,
) -> TrainingCheckpointPaths:
    """Resolve one-based, versioned naming while retaining zero-based metadata."""
    train_cfg = _get_value(cfg, "TRAIN", None)
    checkpoint_cfg = _get_value(train_cfg, "CHECKPOINT", _MISSING)
    if checkpoint_cfg is _MISSING:
        raise ValueError("TRAIN.CHECKPOINT is required for the new checkpoint lifecycle")

    prefix = str(_get_value(checkpoint_cfg, "PREFIX", "checkpoint")).strip()
    if not prefix:
        raise ValueError("TRAIN.CHECKPOINT.PREFIX must not be empty")
    if Path(prefix).name != prefix:
        raise ValueError("TRAIN.CHECKPOINT.PREFIX must be a file-name prefix")
    if outer_it < 0 or path_it < 0:
        raise ValueError("outer_it and path_it must be non-negative")

    checkpoint_dir = _checkpoint_directory(cfg)
    versioned = checkpoint_dir / (
        "{}.outer_{:03d}.path_{:04d}.pth.tar".format(
            prefix, outer_it, path_it + 1
        )
    )
    save_latest = bool(_get_value(checkpoint_cfg, "SAVE_LATEST", True))
    latest = checkpoint_dir / "{}.latest.pth.tar".format(prefix) if save_latest else None
    return TrainingCheckpointPaths(versioned=versioned, latest=latest)


def should_save_training_checkpoint(
    cfg: Any,
    *,
    outer_it: int,
    path_it: int,
    path_iterations: int,
) -> bool:
    if not has_training_checkpoint_config(cfg):
        return False
    checkpoint_cfg = _get_value(_get_value(cfg, "TRAIN", None), "CHECKPOINT")
    every_outer = int(_get_value(checkpoint_cfg, "SAVE_EVERY_OUTER", 1))
    if every_outer <= 0:
        raise ValueError("TRAIN.CHECKPOINT.SAVE_EVERY_OUTER must be positive")
    return path_it + 1 == path_iterations and outer_it % every_outer == 0


def resolve_inference_checkpoint_path(
    cfg: Any,
    *,
    require_exists: bool = False,
) -> Path:
    """Resolve TEST.CKPT_FILE first, then the legacy TEST.CKPT name."""
    checkpoint_dir = _checkpoint_directory(cfg)
    test_cfg = _get_value(cfg, "TEST", None)
    exact_file = _get_value(test_cfg, "CKPT_FILE", _MISSING)
    legacy_path = _legacy_inference_path(test_cfg, checkpoint_dir)

    if exact_file is not _MISSING and str(exact_file).strip():
        resolved = _resolve_under_checkpoint_dir(str(exact_file), checkpoint_dir)
        if legacy_path is not None and legacy_path != resolved:
            conflict = (os.fspath(resolved), os.fspath(legacy_path))
            if conflict not in _WARNED_INFERENCE_CONFLICTS:
                warnings.warn(
                    "TEST.CKPT_FILE resolves to {!r}, while legacy TEST.CKPT "
                    "resolves to {!r}; TEST.CKPT_FILE takes precedence.".format(
                        os.fspath(resolved), os.fspath(legacy_path)
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )
                _WARNED_INFERENCE_CONFLICTS.add(conflict)
    elif legacy_path is not None:
        resolved = legacy_path
    else:
        raise ValueError("configure TEST.CKPT_FILE or legacy TEST.CKPT")

    resolved = resolved.resolve(strict=False)
    if require_exists and not resolved.is_file():
        raise FileNotFoundError(
            "checkpoint not found at resolved path: {}".format(resolved)
        )
    return resolved


def _serializable_snapshot(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _serializable_snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable_snapshot(item) for item in value]
    if isinstance(value, Path):
        return os.fspath(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def build_checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: Any,
    outer_it: int,
    path_it: int,
    config_path: str | os.PathLike[str] | None = None,
    random_seed: int | None = None,
) -> dict[str, Any]:
    train_cfg = _get_value(cfg, "TRAIN", None)
    return {
        "format_version": 1,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "outer_it": int(outer_it),
        "path_it": int(path_it),
        "trajectory_mode": resolve_trajectory_mode(cfg),
        "config_path": os.fspath(config_path) if config_path is not None else None,
        "config_snapshot": _serializable_snapshot(cfg),
        "random_seed": random_seed,
        "model_name": str(_get_value(train_cfg, "MODEL", "origin")),
        "num_targets": int(_get_value(train_cfg, "NUM_TARGETS")),
        "step_length": int(_get_value(train_cfg, "STEP_LENGTH")),
        "window_size": int(_get_value(train_cfg, "WINDOW_SIZE")),
    }


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_training_checkpoint(
    payload: dict[str, Any], paths: TrainingCheckpointPaths
) -> TrainingCheckpointPaths:
    _atomic_torch_save(payload, paths.versioned)
    if paths.latest is not None:
        _atomic_torch_save(payload, paths.latest)
    return paths


def load_checkpoint_payload(
    path: str | os.PathLike[str], *, map_location: Any = "cpu"
) -> dict[str, Any]:
    checkpoint_path = Path(path).resolve(strict=False)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "checkpoint not found at resolved path: {}".format(checkpoint_path)
        )
    payload = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(
            "checkpoint does not contain a state_dict: {}".format(checkpoint_path)
        )
    return payload


def _state_dict_for_model(
    state_dict: Mapping[str, torch.Tensor], model: torch.nn.Module
) -> dict[str, torch.Tensor]:
    loaded = dict(state_dict)
    model_keys = list(model.state_dict().keys())
    loaded_keys = list(loaded.keys())
    model_has_module = bool(model_keys) and all(
        key.startswith("module.") for key in model_keys
    )
    loaded_has_module = bool(loaded_keys) and all(
        key.startswith("module.") for key in loaded_keys
    )
    if loaded_has_module and not model_has_module:
        return {key[len("module."):]: value for key, value in loaded.items()}
    if model_has_module and not loaded_has_module:
        return {"module." + key: value for key, value in loaded.items()}
    return loaded


def load_checkpoint_into_model(
    model: torch.nn.Module,
    path: str | os.PathLike[str],
    *,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: Any = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    payload = load_checkpoint_payload(path, map_location=map_location)
    state_dict = payload["state_dict"]
    if not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint state_dict is not a mapping: {}".format(path))
    model.load_state_dict(_state_dict_for_model(state_dict, model), strict=strict)
    if optimizer is not None:
        if "optimizer" not in payload:
            raise ValueError("checkpoint does not contain optimizer state: {}".format(path))
        optimizer.load_state_dict(payload["optimizer"])
    return payload
