"""Normalize grouped xian trajectories into VecRoad trajectory-piece CSVs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("traj_id", "data_time", "lon", "lat")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_metadata(path: Path):
    metadata = json.loads(path.read_text(encoding="utf-8"))
    bbox = metadata.get("bbox_gcj02") or metadata.get("bbox")
    size = metadata.get("original_size")
    if not isinstance(bbox, dict) or not isinstance(size, list) or len(size) != 2:
        raise ValueError(f"metadata must contain bbox_gcj02 and original_size: {path}")
    bbox = {
        key: float(bbox[key])
        for key in ("lat_min", "lon_min", "lat_max", "lon_max")
    }
    width, height = map(int, size)
    return metadata, bbox, width, height


def _quantiles(series: pd.Series) -> dict[str, float]:
    quantiles = series.quantile([0.0, 0.25, 0.5, 0.75, 0.95, 1.0])
    return {
        "min": float(quantiles.loc[0.0]),
        "p25": float(quantiles.loc[0.25]),
        "p50": float(quantiles.loc[0.5]),
        "p75": float(quantiles.loc[0.75]),
        "p95": float(quantiles.loc[0.95]),
        "max": float(quantiles.loc[1.0]),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a traj_id/lon/lat grouped CSV into one VecRoad-compatible "
            "CSV per trajectory."
        )
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data_self/input/regions/xian_metadata.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_self/input/traj_piece/xian"),
    )
    parser.add_argument("--min-points", type=int, default=2)
    parser.add_argument("--grid-cell-size", type=int, default=256)
    parser.add_argument("--tile-size", type=int, default=4096)
    return parser.parse_args()


def prepare_trajectory_pieces(
        source: Path,
        metadata_path: Path,
        output_dir: Path,
        *,
        min_points: int = 2,
        grid_cell_size: int = 256,
        tile_size: int = 4096) -> dict:
    source = source.resolve(strict=True)
    metadata_path = metadata_path.resolve(strict=True)
    output_dir = output_dir.resolve(strict=False)
    if min_points < 2:
        raise ValueError("min_points must be at least 2")
    if grid_cell_size <= 0 or tile_size <= 0:
        raise ValueError("grid_cell_size and tile_size must be positive")
    remove_empty_output = False
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise FileExistsError(
                f"output directory is not empty; refusing to overwrite: {output_dir}"
            )
        remove_empty_output = True

    metadata, bbox, width, height = _load_metadata(metadata_path)
    data = pd.read_csv(source, dtype={"traj_id": str})
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"source is missing required columns: {missing_columns}")
    data = data.loc[:, list(REQUIRED_COLUMNS)].copy()
    data["data_time"] = pd.to_datetime(data["data_time"], errors="coerce")
    data["lon"] = pd.to_numeric(data["lon"], errors="coerce")
    data["lat"] = pd.to_numeric(data["lat"], errors="coerce")
    invalid_rows = data.isna().any(axis=1)
    if invalid_rows.any():
        raise ValueError(f"source contains {int(invalid_rows.sum())} invalid rows")
    finite = np.isfinite(data[["lon", "lat"]].to_numpy()).all(axis=1)
    if not finite.all():
        raise ValueError(f"source contains {int((~finite).sum())} non-finite points")

    inside = (
        data["lat"].between(bbox["lat_min"], bbox["lat_max"], inclusive="both")
        & data["lon"].between(bbox["lon_min"], bbox["lon_max"], inclusive="both")
    )
    if not inside.all():
        outside = data.loc[~inside, ["traj_id", "data_time", "lon", "lat"]]
        raise ValueError(
            "trajectory points fall outside metadata bbox; count={} sample={}".format(
                len(outside), outside.head(5).to_dict("records")))

    data.sort_values(["traj_id", "data_time"], kind="stable", inplace=True)
    group_sizes = data.groupby("traj_id", sort=False).size()
    too_short = group_sizes[group_sizes < min_points]
    if len(too_short):
        raise ValueError(
            f"{len(too_short)} trajectories contain fewer than {min_points} points"
        )
    same_trajectory = data["traj_id"].eq(data["traj_id"].shift())
    time_delta = data["data_time"].diff().dt.total_seconds().where(same_trajectory)
    if (time_delta <= 0).any():
        raise ValueError(
            f"source contains {int((time_delta <= 0).sum())} non-increasing timestamps"
        )

    pixel_x = (
        (data["lon"] - bbox["lon_min"])
        * width / (bbox["lon_max"] - bbox["lon_min"])
    )
    pixel_y = (
        height
        - (data["lat"] - bbox["lat_min"])
        * height / (bbox["lat_max"] - bbox["lat_min"])
    )
    grid_cells = set(zip(
        np.floor(pixel_x / grid_cell_size).astype(int),
        np.floor(pixel_y / grid_cell_size).astype(int),
    ))
    tile_counts = pd.DataFrame({
        "tile_x": np.floor(pixel_x / tile_size).astype(int),
        "tile_y": np.floor(pixel_y / tile_size).astype(int),
    }).value_counts().sort_index()
    expected_tiles = {
        (tile_x, tile_y)
        for tile_x in range(math.ceil(width / tile_size))
        for tile_y in range(math.ceil(height / tile_size))
    }
    observed_tiles = set(tile_counts.index.tolist())
    missing_tiles = sorted(expected_tiles - observed_tiles)
    if missing_tiles:
        raise ValueError(f"trajectory data does not cover image tiles: {missing_tiles}")
    boundary_gaps = {
        "left": float(pixel_x.min()),
        "top": float(pixel_y.min()),
        "right": float(width - pixel_x.max()),
        "bottom": float(height - pixel_y.max()),
    }
    excessive_gaps = {
        side: gap for side, gap in boundary_gaps.items()
        if gap > grid_cell_size
    }
    if excessive_gaps:
        raise ValueError(
            "trajectory extent does not reach metadata boundaries within one "
            f"{grid_cell_size}px grid cell: {excessive_gaps}"
        )

    if remove_empty_output:
        output_dir.rmdir()
    temporary_dir = output_dir.parent / f"{output_dir.name}.tmp-{os.getpid()}"
    if temporary_dir.exists():
        raise FileExistsError(f"temporary output already exists: {temporary_dir}")
    temporary_dir.mkdir(parents=True)
    index_path = temporary_dir / "trajectory_index.jsonl"
    trajectory_count = 0
    written_points = 0
    try:
        with index_path.open("w", encoding="utf-8", newline="\n") as index_file:
            for trajectory_count, (trajectory_id, group) in enumerate(
                    data.groupby("traj_id", sort=False), start=1):
                file_name = f"traj_{trajectory_count - 1:05d}.csv"
                piece_path = temporary_dir / file_name
                with piece_path.open("w", encoding="utf-8", newline="") as piece_file:
                    writer = csv.writer(piece_file, lineterminator="\n")
                    writer.writerow(("data_time", "lat", "lon"))
                    for row in group.itertuples(index=False):
                        writer.writerow((
                            row.data_time.strftime("%Y-%m-%d %H:%M:%S"),
                            f"{row.lat:.7f}",
                            f"{row.lon:.7f}",
                        ))
                written_points += len(group)
                index_file.write(json.dumps({
                    "file": file_name,
                    "source_traj_id": trajectory_id,
                    "point_count": len(group),
                    "time_start": group["data_time"].iloc[0].isoformat(),
                    "time_end": group["data_time"].iloc[-1].isoformat(),
                }, ensure_ascii=False) + "\n")

        if written_points != len(data) or trajectory_count != len(group_sizes):
            raise AssertionError("written trajectory totals do not match source totals")

        manifest = {
            "region": metadata.get("region", "xian"),
            "source": os.fspath(source),
            "source_sha256": _sha256(source),
            "metadata": os.fspath(metadata_path),
            "metadata_sha256": _sha256(metadata_path),
            "coordinate_source": "raw GCJ02 lon/lat",
            "output_layout": "one trajectory per CSV: data_time,lat,lon",
            "bbox": bbox,
            "original_size": [width, height],
            "trajectory_count": trajectory_count,
            "point_count": written_points,
            "trajectory_length": _quantiles(group_sizes),
            "time_gap_over_one_hour_count": int((time_delta > 3600).sum()),
            "point_bounds": {
                "lon_min": float(data["lon"].min()),
                "lat_min": float(data["lat"].min()),
                "lon_max": float(data["lon"].max()),
                "lat_max": float(data["lat"].max()),
            },
            "pixel_bounds": {
                "x_min": float(pixel_x.min()),
                "y_min": float(pixel_y.min()),
                "x_max": float(pixel_x.max()),
                "y_max": float(pixel_y.max()),
            },
            "boundary_gap_pixels": boundary_gaps,
            "grid_coverage": {
                "cell_size": grid_cell_size,
                "occupied_cells": len(grid_cells),
                "total_bbox_cells": (
                    math.ceil(width / grid_cell_size)
                    * math.ceil(height / grid_cell_size)
                ),
            },
            "tile_point_counts": {
                f"{tile_x}_{tile_y}": int(count)
                for (tile_x, tile_y), count in tile_counts.items()
            },
        }
        (temporary_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary_dir, output_dir)
        return manifest
    except Exception:
        # Keep a failed temporary directory for forensic inspection. It is
        # never promoted to the configured dataset path.
        raise


def main() -> None:
    args = _parse_args()
    report = prepare_trajectory_pieces(
        args.source,
        args.metadata,
        args.output_dir,
        min_points=args.min_points,
        grid_cell_size=args.grid_cell_size,
        tile_size=args.tile_size,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
