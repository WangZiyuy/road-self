"""Derive deterministic binary tiles from a registered full-canvas raster."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.seg_raster import canonicalize_raster_array, sha256_file


def derive_upper_left_tile(
    full_canvas_path: Path,
    output_path: Path,
    *,
    tile_size: int = 4096,
    valid_extent_wh: tuple[int, int] = (4300, 5000),
) -> dict[str, object]:
    raw = np.asarray(Image.open(full_canvas_path).convert("L"))
    binary, valid_mask = canonicalize_raster_array(
        raw, valid_extent_wh=valid_extent_wh)
    if binary.shape[0] < tile_size or binary.shape[1] < tile_size:
        raise ValueError("full canvas is smaller than requested tile")
    tile = (binary[:tile_size, :tile_size] * 255).astype(np.uint8)
    tile_mask = valid_mask[:tile_size, :tile_size]
    if not np.all(tile_mask == 1):
        raise ValueError("upper-left training tile contains invalid padding")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tile, mode="L").save(
        output_path, format="PNG", optimize=False, compress_level=9)
    return {
        "source_path": str(full_canvas_path),
        "source_sha256": sha256_file(full_canvas_path),
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "shape_hw": [tile_size, tile_size],
        "conversion": "traj_binary = (traj_raw > 0).astype(float32)",
        "intended_for_commit": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-canvas", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=4096)
    args = parser.parse_args()
    result = derive_upper_left_tile(
        args.full_canvas, args.output, tile_size=args.tile_size)
    for key, value in result.items():
        print("{}={}".format(key, value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
