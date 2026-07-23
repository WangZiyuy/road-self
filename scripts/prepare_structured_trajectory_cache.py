"""Build the Stage 1A structured trajectory cache from one-CSV-per-track data."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import shutil
import sys
import time
import uuid
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.gis_to_graph import latlng_to_pixel
from utils.structured_trajectory_store import (
    SCHEMA_VERSION,
    open_structured_trajectory_store,
)
from utils.trajectory_fragments import (
    SEGMENT_GRID_INDEX_BASIS,
    trajectory_grid_cells,
)


@dataclass(frozen=True)
class TrackSource:
    """Stable input identity and optional manifest expectations for one track."""

    source_path: Path
    source_file: str
    source_traj_id: str
    expected_point_count: Optional[int] = None
    expected_time_start_ns: Optional[int] = None
    expected_time_end_ns: Optional[int] = None


@dataclass(frozen=True)
class RegionTransform:
    """Geographic bounds and image geometry used by GisToGraphConverter."""

    width: int
    height: int
    lat_min: float
    lon_min: float
    lat_max: float
    lon_max: float

    @property
    def xscale(self) -> float:
        return self.width / (self.lon_max - self.lon_min)

    @property
    def yscale(self) -> float:
        return self.height / (self.lat_max - self.lat_min)


def _read_json_object(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object in {}".format(path))
    return value


def _load_region_transform(metadata_path: Path) -> Tuple[Dict[str, Any], RegionTransform]:
    metadata = _read_json_object(metadata_path)
    bbox = (
        metadata.get("bbox_gcj02")
        or metadata.get("bbox_wgs84")
        or metadata.get("bbox")
    )
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        bbox = {
            "lat_min": bbox[0],
            "lon_min": bbox[1],
            "lat_max": bbox[2],
            "lon_max": bbox[3],
        }
    if not isinstance(bbox, dict):
        raise ValueError(
            "metadata must contain bbox_gcj02, bbox_wgs84, or bbox")

    image_size = (
        metadata.get("original_size")
        or metadata.get("image_size")
        or metadata.get("valid_size")
    )
    if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
        raise ValueError(
            "metadata must contain a two-element image size")

    required_bbox = ("lat_min", "lon_min", "lat_max", "lon_max")
    missing_bbox = [name for name in required_bbox if name not in bbox]
    if missing_bbox:
        raise ValueError(
            "metadata bbox is missing fields {}".format(missing_bbox))

    transform = RegionTransform(
        width=int(image_size[0]),
        height=int(image_size[1]),
        lat_min=float(bbox["lat_min"]),
        lon_min=float(bbox["lon_min"]),
        lat_max=float(bbox["lat_max"]),
        lon_max=float(bbox["lon_max"]),
    )
    numeric_values = (
        transform.width,
        transform.height,
        transform.lat_min,
        transform.lon_min,
        transform.lat_max,
        transform.lon_max,
    )
    if not all(math.isfinite(float(value)) for value in numeric_values):
        raise ValueError("metadata contains a non-finite image size or bbox")
    if transform.width <= 0 or transform.height <= 0:
        raise ValueError("metadata image dimensions must be positive")
    if (
        transform.lat_max <= transform.lat_min
        or transform.lon_max <= transform.lon_min
    ):
        raise ValueError("metadata bbox must have positive width and height")
    return metadata, transform


def _parse_timestamp_ns(value: str, source: str) -> int:
    text = value.strip()
    if not text:
        raise ValueError("empty timestamp in {}".format(source))
    try:
        timestamp = np.datetime64(text, "ns")
    except (TypeError, ValueError) as error:
        raise ValueError(
            "invalid timestamp {!r} in {}".format(text, source)) from error
    timestamp_ns = int(timestamp.astype(np.int64))
    if timestamp_ns == np.iinfo(np.int64).min:
        raise ValueError("NaT timestamp in {}".format(source))
    return timestamp_ns


def _timestamp_ns_to_text(timestamp_ns: int) -> str:
    return np.datetime_as_string(
        np.datetime64(int(timestamp_ns), "ns"),
        unit="ns",
    )


def _safe_source_path(piece_dir: Path, source_file: str) -> Path:
    relative_path = Path(source_file.replace("\\", "/"))
    if relative_path.is_absolute():
        raise ValueError(
            "trajectory index source_file must be relative: {!r}".format(
                source_file))
    candidate = (piece_dir / relative_path).resolve()
    try:
        candidate.relative_to(piece_dir)
    except ValueError as error:
        raise ValueError(
            "trajectory index source_file escapes piece directory: {!r}"
            .format(source_file)) from error
    if not candidate.is_file():
        raise FileNotFoundError(
            "trajectory CSV from index does not exist: {}".format(candidate))
    return candidate


def _optional_index_timestamp(
    record: Mapping[str, Any],
    field: str,
    source: str,
) -> Optional[int]:
    value = record.get(field)
    if value is None or str(value).strip() == "":
        return None
    return _parse_timestamp_ns(str(value), "{} field {}".format(source, field))


def _discover_track_sources(piece_dir: Path) -> Tuple[List[TrackSource], str]:
    index_path = piece_dir / "trajectory_index.jsonl"
    csv_paths = sorted(
        (path.resolve() for path in piece_dir.rglob("*.csv")),
        key=lambda path: path.relative_to(piece_dir).as_posix(),
    )
    if not csv_paths:
        raise ValueError("no trajectory CSV files found in {}".format(piece_dir))

    if not index_path.is_file():
        return [
            TrackSource(
                source_path=path,
                source_file=path.relative_to(piece_dir).as_posix(),
                source_traj_id=path.stem,
            )
            for path in csv_paths
        ], "lexical_relative_csv_path"

    sources: List[TrackSource] = []
    indexed_paths = set()
    with index_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                raise ValueError(
                    "blank line in {} at line {}".format(
                        index_path, line_number))
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "invalid JSON in {} at line {}".format(
                        index_path, line_number)) from error
            if not isinstance(record, dict):
                raise ValueError(
                    "{} line {} must contain a JSON object".format(
                        index_path, line_number))
            source_file_value = record.get("file", record.get("source_file"))
            if not source_file_value:
                raise ValueError(
                    "{} line {} is missing file/source_file".format(
                        index_path, line_number))
            source_path = _safe_source_path(
                piece_dir, str(source_file_value))
            if source_path in indexed_paths:
                raise ValueError(
                    "trajectory index references a CSV more than once: {}"
                    .format(source_path))
            indexed_paths.add(source_path)
            source_file = source_path.relative_to(piece_dir).as_posix()
            source_traj_id = str(
                record.get("source_traj_id", source_path.stem))
            if not source_traj_id:
                raise ValueError(
                    "empty source_traj_id for {}".format(source_file))
            expected_point_count = record.get("point_count")
            if expected_point_count is not None:
                expected_point_count = int(expected_point_count)
                if expected_point_count <= 0:
                    raise ValueError(
                        "point_count must be positive for {}".format(
                            source_file))
            record_source = "{} line {}".format(index_path, line_number)
            sources.append(
                TrackSource(
                    source_path=source_path,
                    source_file=source_file,
                    source_traj_id=source_traj_id,
                    expected_point_count=expected_point_count,
                    expected_time_start_ns=_optional_index_timestamp(
                        record, "time_start", record_source),
                    expected_time_end_ns=_optional_index_timestamp(
                        record, "time_end", record_source),
                )
            )

    extra_csvs = [
        path.relative_to(piece_dir).as_posix()
        for path in csv_paths
        if path not in indexed_paths
    ]
    if extra_csvs:
        preview = extra_csvs[:5]
        raise ValueError(
            "trajectory_index.jsonl omits {} CSV files; examples: {}".format(
                len(extra_csvs), preview))
    if len(indexed_paths) != len(csv_paths):
        raise ValueError(
            "trajectory index and CSV file counts do not match")
    return sources, "trajectory_index_jsonl_line_order"


def _read_track_csv(source: TrackSource) -> Tuple[np.ndarray, np.ndarray]:
    lat_lon_rows: List[Tuple[float, float]] = []
    timestamp_rows: List[int] = []
    with source.source_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        try:
            raw_header = next(reader)
        except StopIteration as error:
            raise ValueError(
                "trajectory CSV is empty: {}".format(source.source_path)
            ) from error
        header = [name.strip() for name in raw_header]
        required = ("data_time", "lat", "lon")
        missing = [name for name in required if name not in header]
        if missing:
            raise ValueError(
                "trajectory CSV {} is missing columns {}".format(
                    source.source_path, missing))
        column = {name: header.index(name) for name in required}
        required_width = max(column.values()) + 1

        for line_number, row in enumerate(reader, start=2):
            if not row or all(not value.strip() for value in row):
                raise ValueError(
                    "blank trajectory row in {} at line {}".format(
                        source.source_path, line_number))
            if len(row) < required_width:
                raise ValueError(
                    "short trajectory row in {} at line {}".format(
                        source.source_path, line_number))
            row_source = "{} line {}".format(
                source.source_path, line_number)
            timestamp_ns = _parse_timestamp_ns(
                row[column["data_time"]], row_source)
            try:
                latitude = float(row[column["lat"]])
                longitude = float(row[column["lon"]])
            except ValueError as error:
                raise ValueError(
                    "invalid latitude/longitude in {}".format(
                        row_source)) from error
            if not math.isfinite(latitude) or not math.isfinite(longitude):
                raise ValueError(
                    "non-finite latitude/longitude in {}".format(row_source))
            timestamp_rows.append(timestamp_ns)
            lat_lon_rows.append((latitude, longitude))

    if not lat_lon_rows:
        raise ValueError(
            "trajectory CSV contains no points: {}".format(source.source_path))
    lat_lon = np.asarray(lat_lon_rows, dtype=np.float64)
    timestamps_ns = np.asarray(timestamp_rows, dtype=np.int64)
    return lat_lon, timestamps_ns


def _validate_source_expectations(
    source: TrackSource,
    timestamps_ns: np.ndarray,
) -> None:
    point_count = int(timestamps_ns.shape[0])
    if (
        source.expected_point_count is not None
        and source.expected_point_count != point_count
    ):
        raise ValueError(
            "point_count mismatch for {}: index={}, CSV={}".format(
                source.source_file,
                source.expected_point_count,
                point_count,
            ))
    if (
        source.expected_time_start_ns is not None
        and source.expected_time_start_ns != int(timestamps_ns[0])
    ):
        raise ValueError(
            "time_start mismatch for {}".format(source.source_file))
    if (
        source.expected_time_end_ns is not None
        and source.expected_time_end_ns != int(timestamps_ns[-1])
    ):
        raise ValueError(
            "time_end mismatch for {}".format(source.source_file))


def _lat_lon_to_pixel(
    lat_lon: np.ndarray,
    transform: RegionTransform,
) -> np.ndarray:
    """Use the exact scalar conversion helper shared with GisToGraphConverter."""

    points_xy = np.empty((lat_lon.shape[0], 2), dtype=np.float32)
    for point_index, (latitude, longitude) in enumerate(lat_lon):
        points_xy[point_index] = latlng_to_pixel(
            float(latitude),
            float(longitude),
            transform.height,
            transform.lat_min,
            transform.lon_min,
            transform.xscale,
            transform.yscale,
        )
    return points_xy


def _array_description(array: np.ndarray) -> Dict[str, Any]:
    return {
        "shape": [int(value) for value in array.shape],
        "dtype": np.dtype(array.dtype).name,
    }


def _write_deterministic_npz(
    path: Path,
    arrays: Sequence[Tuple[str, np.ndarray]],
) -> None:
    """Write an uncompressed NPZ with stable member order and timestamps."""

    with zipfile.ZipFile(
        str(path), mode="w", compression=zipfile.ZIP_STORED
    ) as archive:
        for name, array in arrays:
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            member = zipfile.ZipInfo(
                filename="{}.npy".format(name),
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            member.compress_type = zipfile.ZIP_STORED
            member.external_attr = 0o600 << 16
            archive.writestr(member, buffer.getvalue())


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(
            value,
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")


def _write_track_index(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            file.write("\n")


def _prepare_output_destination(output_dir: Path, overwrite: bool) -> None:
    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise FileExistsError(
            "output path exists and is not a directory: {}".format(
                output_dir))
    has_contents = next(output_dir.iterdir(), None) is not None
    if has_contents and not overwrite:
        raise FileExistsError(
            "refusing to overwrite non-empty output directory without "
            "--overwrite: {}".format(output_dir))


def _promote_temporary_output(
    temporary_dir: Path,
    output_dir: Path,
    overwrite: bool,
) -> None:
    backup_dir: Optional[Path] = None
    try:
        if output_dir.exists():
            if next(output_dir.iterdir(), None) is None:
                output_dir.rmdir()
            elif overwrite:
                backup_dir = output_dir.with_name(
                    ".{}.backup-{}".format(
                        output_dir.name, os.getpid()))
                if backup_dir.exists():
                    raise FileExistsError(
                        "temporary backup path already exists: {}".format(
                            backup_dir))
                output_dir.replace(backup_dir)
        temporary_dir.replace(output_dir)
    except BaseException:
        if backup_dir is not None and backup_dir.exists() and not output_dir.exists():
            backup_dir.replace(output_dir)
        raise
    if backup_dir is not None:
        shutil.rmtree(str(backup_dir))


def build_structured_trajectory_cache(
    region: str,
    piece_dir: Path,
    metadata_path: Path,
    output_dir: Path,
    cell_size: int = 256,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Build, atomically publish, reopen, and validate a structured cache."""

    piece_dir = Path(piece_dir).resolve(strict=True)
    metadata_path = Path(metadata_path).resolve(strict=True)
    output_dir = Path(output_dir).resolve()
    if not piece_dir.is_dir():
        raise NotADirectoryError(
            "piece directory is not a directory: {}".format(piece_dir))
    if not metadata_path.is_file():
        raise FileNotFoundError(
            "metadata file does not exist: {}".format(metadata_path))
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")
    if not region.strip():
        raise ValueError("region must not be empty")
    _prepare_output_destination(output_dir, overwrite)

    metadata, transform = _load_region_transform(metadata_path)
    metadata_region = metadata.get("region")
    if metadata_region is not None and str(metadata_region) != str(region):
        raise ValueError(
            "requested region {!r} does not match metadata region {!r}".format(
                region, metadata_region))
    sources, source_order = _discover_track_sources(piece_dir)

    offsets = np.zeros((len(sources) + 1,), dtype=np.int64)
    timestamp_ranges: List[Tuple[int, int]] = []
    for track_index, source in enumerate(sources):
        _, timestamps_ns = _read_track_csv(source)
        _validate_source_expectations(source, timestamps_ns)
        offsets[track_index + 1] = (
            offsets[track_index] + timestamps_ns.shape[0])
        timestamp_ranges.append(
            (int(timestamps_ns[0]), int(timestamps_ns[-1])))
    point_count = int(offsets[-1])

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = output_dir.parent / ".{}.tmp-{}".format(
        output_dir.name, uuid.uuid4().hex
    )
    temporary_dir.mkdir()
    try:
        points_xy = np.lib.format.open_memmap(
            str(temporary_dir / "points_xy.npy"),
            mode="w+",
            dtype=np.float32,
            shape=(point_count, 2),
        )
        timestamps_ns_file = np.lib.format.open_memmap(
            str(temporary_dir / "timestamps_ns.npy"),
            mode="w+",
            dtype=np.int64,
            shape=(point_count,),
        )
        np.save(
            str(temporary_dir / "track_offsets.npy"),
            offsets,
            allow_pickle=False,
        )

        track_records: List[Dict[str, Any]] = []
        cell_to_track_ids: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for track_index, source in enumerate(sources):
            lat_lon, timestamps_ns = _read_track_csv(source)
            _validate_source_expectations(source, timestamps_ns)
            start = int(offsets[track_index])
            end = int(offsets[track_index + 1])
            if end - start != lat_lon.shape[0]:
                raise RuntimeError(
                    "trajectory changed between build passes: {}".format(
                        source.source_path))
            pixel_points = _lat_lon_to_pixel(lat_lon, transform)
            points_xy[start:end] = pixel_points
            timestamps_ns_file[start:end] = timestamps_ns

            occupied_cells = trajectory_grid_cells(
                pixel_points,
                cell_size,
                include_segments=True,
            )
            for cell_x, cell_y in occupied_cells:
                cell_to_track_ids[(int(cell_x), int(cell_y))].append(
                    track_index)

            time_start_ns, time_end_ns = timestamp_ranges[track_index]
            track_records.append(
                {
                    "track_index": track_index,
                    "source_traj_id": source.source_traj_id,
                    "source_file": source.source_file,
                    "point_count": end - start,
                    "time_start": _timestamp_ns_to_text(time_start_ns),
                    "time_end": _timestamp_ns_to_text(time_end_ns),
                }
            )

        points_xy.flush()
        timestamps_ns_file.flush()
        del points_xy
        del timestamps_ns_file

        sorted_cells = sorted(cell_to_track_ids)
        cells = np.asarray(sorted_cells, dtype=np.int32).reshape((-1, 2))
        grid_cell_offsets = np.zeros(
            (len(sorted_cells) + 1,), dtype=np.int64)
        flattened_track_ids: List[int] = []
        for cell_index, cell in enumerate(sorted_cells):
            track_ids = sorted(set(cell_to_track_ids[cell]))
            flattened_track_ids.extend(track_ids)
            grid_cell_offsets[cell_index + 1] = len(flattened_track_ids)
        grid_track_ids = np.asarray(flattened_track_ids, dtype=np.int32)
        _write_deterministic_npz(
            temporary_dir / "grid_index.npz",
            (
                ("cells", cells),
                ("cell_offsets", grid_cell_offsets),
                ("track_ids", grid_track_ids),
            ),
        )
        _write_track_index(
            temporary_dir / "track_index.jsonl", track_records)

        meta = {
            "schema_version": SCHEMA_VERSION,
            "region": str(region),
            "trajectory_count": len(sources),
            "point_count": point_count,
            "image_size": [transform.width, transform.height],
            "geographic_bbox": {
                "lat_min": transform.lat_min,
                "lon_min": transform.lon_min,
                "lat_max": transform.lat_max,
                "lon_max": transform.lon_max,
            },
            "cell_size": int(cell_size),
            "grid_cell_count": int(cells.shape[0]),
            "grid_membership_count": int(grid_track_ids.shape[0]),
            "grid_index_basis": SEGMENT_GRID_INDEX_BASIS,
            "track_order": source_order,
            "coordinate_order": ["x", "y"],
            "timestamp_unit": "nanoseconds_since_unix_epoch",
            "arrays": {
                "points_xy": {
                    "shape": [point_count, 2],
                    "dtype": "float32",
                },
                "timestamps_ns": {
                    "shape": [point_count],
                    "dtype": "int64",
                },
                "track_offsets": _array_description(offsets),
                "grid_cells": _array_description(cells),
                "grid_cell_offsets": _array_description(grid_cell_offsets),
                "grid_track_ids": _array_description(grid_track_ids),
            },
        }
        if "canvas_size" in metadata:
            canvas_size = metadata["canvas_size"]
            if isinstance(canvas_size, (list, tuple)):
                meta["canvas_size"] = [
                    int(value) for value in canvas_size
                ]
            else:
                meta["canvas_size"] = int(canvas_size)
        _write_json(temporary_dir / "meta.json", meta)

        temporary_store = open_structured_trajectory_store(
            str(temporary_dir))
        temporary_store.validate()
        del temporary_store
        _promote_temporary_output(
            temporary_dir, output_dir, overwrite=overwrite)
        return meta
    except BaseException:
        if temporary_dir.exists():
            shutil.rmtree(str(temporary_dir))
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic, memory-mapped structured trajectory "
            "cache from one CSV per trajectory."
        )
    )
    parser.add_argument("--region", required=True)
    parser.add_argument("--piece-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cell-size", type=int, default=256)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing non-empty output directory atomically.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    started = time.perf_counter()
    meta = build_structured_trajectory_cache(
        region=args.region,
        piece_dir=args.piece_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        cell_size=args.cell_size,
        overwrite=args.overwrite,
    )
    report = dict(meta)
    report["output_dir"] = str(args.output_dir.resolve())
    report["build_seconds"] = round(time.perf_counter() - started, 6)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
