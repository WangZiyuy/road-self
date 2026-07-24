"""Stage 3C YAML loading with small, explicit experiment inheritance."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml
from easydict import EasyDict


def _deep_merge(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> Dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if (
                key in merged
                and isinstance(merged[key], Mapping)
                and isinstance(value, Mapping)):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_stage3c_config(path: Path) -> EasyDict:
    """Load a config and recursively merge an optional ``BASE_CONFIG``.

    Relative base paths are resolved next to the child YAML.  Cycles are
    rejected.  The resolved mapping is returned so checkpoint snapshots are
    self-contained rather than depending on a later base-file change.
    """

    def load_mapping(current: Path, stack) -> Dict[str, Any]:
        current = current.resolve(strict=False)
        if current in stack:
            raise ValueError(
                "cyclic Stage 3C BASE_CONFIG reference: {}".format(
                    current))
        if not current.is_file():
            raise FileNotFoundError(
                "Stage 3C config not found: {}".format(current))
        with current.open("r", encoding="utf-8") as config_file:
            raw = yaml.load(config_file, Loader=yaml.UnsafeLoader)
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise ValueError(
                "Stage 3C config root must be a mapping: {}".format(
                    current))
        raw = dict(raw)
        base_reference = raw.pop("BASE_CONFIG", None)
        if base_reference is None:
            return raw
        base_path = Path(str(base_reference))
        if not base_path.is_absolute():
            base_path = current.parent / base_path
        base = load_mapping(base_path, stack + (current,))
        return _deep_merge(base, raw)

    return EasyDict(load_mapping(Path(path), ()))
