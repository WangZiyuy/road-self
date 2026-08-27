"""Checksum and metadata helpers for canonical raster artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_canonical_raster(
    path: str | Path,
    *,
    valid_extent_wh: tuple[int, int] | None = None,
    intended_for_commit: bool = False,
) -> dict[str, Any]:
    source = Path(path)
    raw = np.asarray(Image.open(source).convert("L"))
    unique, counts = np.unique(raw, return_counts=True)
    return {
        "path": str(source),
        "size_bytes": source.stat().st_size,
        "sha256": sha256_file(source),
        "shape_hw": list(raw.shape),
        "dtype": str(raw.dtype),
        "raw_unique_value_counts": {
            str(int(value)): int(count)
            for value, count in zip(unique, counts)
        },
        "canonical_semantics": "binary_presence",
        "canonical_conversion": "traj_binary = (traj_raw > 0).astype(float32)",
        "valid_extent_wh": (
            list(valid_extent_wh) if valid_extent_wh is not None else None),
        "intended_for_commit": bool(intended_for_commit),
    }
