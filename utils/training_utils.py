"""Small configuration helpers for the road_self training loop."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


DEFAULT_PATH_ITERATIONS = 2048


def _get_value(container: Any, key: str, default: Any) -> Any:
    if container is None:
        return default
    if isinstance(container, Mapping):
        return container.get(key, default)
    return getattr(container, key, default)


def resolve_path_iterations(cfg: Any) -> int:
    train_cfg = _get_value(cfg, "TRAIN", None)
    value = int(_get_value(train_cfg, "PATH_ITERATIONS", DEFAULT_PATH_ITERATIONS))
    if value <= 0:
        raise ValueError("TRAIN.PATH_ITERATIONS must be positive")
    return value


def training_global_step(outer_it: int, path_it: int, path_iterations: int) -> int:
    if outer_it < 0 or path_it < 0 or path_iterations <= 0:
        raise ValueError("training step values must be non-negative and non-zero")
    return outer_it * path_iterations + path_it
