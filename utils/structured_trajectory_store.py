"""Memory-mapped reader for the Stage 1A structured trajectory cache."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np


SCHEMA_VERSION = "1.0"
REQUIRED_CACHE_FILES = (
    "points_xy.npy",
    "timestamps_ns.npy",
    "track_offsets.npy",
    "track_index.jsonl",
    "grid_index.npz",
    "meta.json",
)


@dataclass(frozen=True)
class StructuredTrack:
    """A zero-copy view of one trajectory and its provenance."""

    track_index: int
    source_traj_id: str
    source_file: str
    points_xy: np.ndarray
    timestamps_ns: np.ndarray
    metadata: Mapping[str, Any]

    def __len__(self) -> int:
        return int(self.points_xy.shape[0])


class StructuredTrajectoryStore:
    """Read-only structured trajectory store backed by NumPy mmap arrays."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir.resolve(strict=True)
        if not self.cache_dir.is_dir():
            raise NotADirectoryError(
                "structured trajectory cache is not a directory: {}".format(
                    self.cache_dir))

        missing = [
            name for name in REQUIRED_CACHE_FILES
            if not (self.cache_dir / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "structured trajectory cache is missing files {}: {}".format(
                    missing, self.cache_dir))

        self.meta = self._read_json(self.cache_dir / "meta.json")
        self.track_records = self._read_track_index(
            self.cache_dir / "track_index.jsonl")
        self.points_xy = np.load(
            self.cache_dir / "points_xy.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        self.timestamps_ns = np.load(
            self.cache_dir / "timestamps_ns.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        self.track_offsets = np.load(
            self.cache_dir / "track_offsets.npy",
            mmap_mode="r",
            allow_pickle=False,
        )

        with np.load(
                self.cache_dir / "grid_index.npz",
                allow_pickle=False) as grid:
            required_grid_arrays = {"cells", "cell_offsets", "track_ids"}
            missing_grid = sorted(required_grid_arrays - set(grid.files))
            if missing_grid:
                raise ValueError(
                    "grid_index.npz is missing arrays: {}".format(
                        missing_grid))
            self.grid_cells = grid["cells"]
            self.grid_cell_offsets = grid["cell_offsets"]
            self.grid_track_ids = grid["track_ids"]

        self.cell_size = int(self.meta.get("cell_size", 0))
        self._cell_to_index = {
            (int(cell[0]), int(cell[1])): index
            for index, cell in enumerate(self.grid_cells)
        }
        self._validate_cheap_invariants()

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object in {}".format(path))
        return value

    @staticmethod
    def _read_track_index(path: Path) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    raise ValueError(
                        "blank line in {} at line {}".format(
                            path, line_number))
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(
                        "track index line {} is not an object".format(
                            line_number))
                records.append(record)
        return records

    def _validate_cheap_invariants(self) -> None:
        if self.meta.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                "unsupported structured trajectory schema {!r}; expected {!r}"
                .format(self.meta.get("schema_version"), SCHEMA_VERSION))
        if self.cell_size <= 0:
            raise ValueError("cell_size must be positive")
        if self.points_xy.dtype != np.dtype(np.float32):
            raise TypeError("points_xy.npy must have dtype float32")
        if self.timestamps_ns.dtype != np.dtype(np.int64):
            raise TypeError("timestamps_ns.npy must have dtype int64")
        if self.track_offsets.dtype != np.dtype(np.int64):
            raise TypeError("track_offsets.npy must have dtype int64")
        if self.points_xy.ndim != 2 or self.points_xy.shape[1] != 2:
            raise ValueError("points_xy.npy must have shape [point_count, 2]")
        if self.timestamps_ns.shape != (self.points_xy.shape[0],):
            raise ValueError(
                "timestamps_ns.npy length does not match points_xy.npy")
        if self.track_offsets.shape != (len(self.track_records) + 1,):
            raise ValueError(
                "track_offsets.npy must contain trajectory_count + 1 entries")
        if self.grid_cells.dtype != np.dtype(np.int32):
            raise TypeError("grid cells must have dtype int32")
        if self.grid_cell_offsets.dtype != np.dtype(np.int64):
            raise TypeError("grid cell offsets must have dtype int64")
        if self.grid_track_ids.dtype != np.dtype(np.int32):
            raise TypeError("grid track ids must have dtype int32")
        if self.grid_cells.ndim != 2 or self.grid_cells.shape[1] != 2:
            raise ValueError("grid cells must have shape [cell_count, 2]")
        if self.grid_cell_offsets.shape != (len(self.grid_cells) + 1,):
            raise ValueError(
                "grid cell offsets must contain cell_count + 1 entries")

    @property
    def trajectory_count(self) -> int:
        return len(self.track_records)

    @property
    def point_count(self) -> int:
        return int(self.points_xy.shape[0])

    def get_track(self, track_index: int) -> StructuredTrack:
        """Return a read-only, zero-copy view of one trajectory."""

        if isinstance(track_index, bool) or not isinstance(
                track_index, (int, np.integer)):
            raise TypeError("track_index must be an integer")
        track_index = int(track_index)
        if track_index < 0 or track_index >= self.trajectory_count:
            raise IndexError(
                "track_index {} is outside [0, {})".format(
                    track_index, self.trajectory_count))
        start = int(self.track_offsets[track_index])
        end = int(self.track_offsets[track_index + 1])
        record = self.track_records[track_index]
        return StructuredTrack(
            track_index=track_index,
            source_traj_id=str(record["source_traj_id"]),
            source_file=str(record["source_file"]),
            points_xy=self.points_xy[start:end],
            timestamps_ns=self.timestamps_ns[start:end],
            metadata=record,
        )

    def candidate_track_ids_for_rect(
            self,
            xmin: float,
            ymin: float,
            xmax: float,
            ymax: float) -> np.ndarray:
        """Return sorted track IDs whose indexed point cells overlap a rectangle."""

        bounds = (xmin, ymin, xmax, ymax)
        if not all(math.isfinite(float(value)) for value in bounds):
            raise ValueError("rectangle bounds must be finite")
        xmin, ymin, xmax, ymax = map(float, bounds)
        if xmin > xmax or ymin > ymax:
            raise ValueError(
                "rectangle minimum bounds must not exceed maximum bounds")

        min_cell_x = math.floor(xmin / self.cell_size)
        max_cell_x = math.floor(xmax / self.cell_size)
        min_cell_y = math.floor(ymin / self.cell_size)
        max_cell_y = math.floor(ymax / self.cell_size)
        candidate_parts = []
        for cell_x in range(min_cell_x, max_cell_x + 1):
            for cell_y in range(min_cell_y, max_cell_y + 1):
                cell_index = self._cell_to_index.get((cell_x, cell_y))
                if cell_index is None:
                    continue
                start = int(self.grid_cell_offsets[cell_index])
                end = int(self.grid_cell_offsets[cell_index + 1])
                candidate_parts.append(self.grid_track_ids[start:end])

        if not candidate_parts:
            return np.empty((0,), dtype=np.int32)
        return np.unique(np.concatenate(candidate_parts)).astype(
            np.int32, copy=False)

    def validate(self) -> Dict[str, Any]:
        """Fully validate array, track, metadata, and grid-index consistency."""

        point_count = self.point_count
        trajectory_count = self.trajectory_count
        offsets = np.asarray(self.track_offsets)
        if offsets[0] != 0:
            raise ValueError("track_offsets must start at zero")
        if offsets[-1] != point_count:
            raise ValueError(
                "last track offset must equal the total point count")
        if np.any(offsets[1:] < offsets[:-1]):
            raise ValueError("track_offsets must be non-decreasing")

        for track_index, record in enumerate(self.track_records):
            required = {
                "track_index",
                "source_traj_id",
                "source_file",
                "point_count",
                "time_start",
                "time_end",
            }
            missing = sorted(required - set(record))
            if missing:
                raise ValueError(
                    "track record {} is missing fields {}".format(
                        track_index, missing))
            if int(record["track_index"]) != track_index:
                raise ValueError(
                    "track record order does not match track_index at {}".format(
                        track_index))
            expected_count = int(offsets[track_index + 1] - offsets[track_index])
            if int(record["point_count"]) != expected_count:
                raise ValueError(
                    "track {} point_count does not match offsets".format(
                        track_index))
            if expected_count <= 0:
                raise ValueError(
                    "track {} must contain at least one point".format(
                        track_index))
            if not str(record["source_traj_id"]):
                raise ValueError(
                    "track {} has an empty source_traj_id".format(
                        track_index))
            if not str(record["source_file"]):
                raise ValueError(
                    "track {} has an empty source_file".format(track_index))
            start = int(offsets[track_index])
            end = int(offsets[track_index + 1])
            time_start_ns = self._parse_record_timestamp(
                record["time_start"], track_index, "time_start")
            time_end_ns = self._parse_record_timestamp(
                record["time_end"], track_index, "time_end")
            if time_start_ns != int(self.timestamps_ns[start]):
                raise ValueError(
                    "track {} time_start does not match its first point"
                    .format(track_index))
            if time_end_ns != int(self.timestamps_ns[end - 1]):
                raise ValueError(
                    "track {} time_end does not match its last point"
                    .format(track_index))

        chunk_size = 1_000_000
        nat_value = np.iinfo(np.int64).min
        for start in range(0, point_count, chunk_size):
            end = min(point_count, start + chunk_size)
            if not np.isfinite(self.points_xy[start:end]).all():
                raise ValueError("points_xy contains NaN or Inf")
            if np.any(self.timestamps_ns[start:end] == nat_value):
                raise ValueError("timestamps_ns contains NaT")

        meta_trajectory_count = int(self.meta.get("trajectory_count", -1))
        meta_point_count = int(self.meta.get("point_count", -1))
        if meta_trajectory_count != trajectory_count:
            raise ValueError("meta trajectory_count does not match track index")
        if meta_point_count != point_count:
            raise ValueError("meta point_count does not match points array")
        self._validate_meta_array(
            "points_xy", self.points_xy.shape, self.points_xy.dtype)
        self._validate_meta_array(
            "timestamps_ns",
            self.timestamps_ns.shape,
            self.timestamps_ns.dtype,
        )
        self._validate_meta_array(
            "track_offsets",
            self.track_offsets.shape,
            self.track_offsets.dtype,
        )
        self._validate_meta_array(
            "grid_cells", self.grid_cells.shape, self.grid_cells.dtype)
        self._validate_meta_array(
            "grid_cell_offsets",
            self.grid_cell_offsets.shape,
            self.grid_cell_offsets.dtype,
        )
        self._validate_meta_array(
            "grid_track_ids",
            self.grid_track_ids.shape,
            self.grid_track_ids.dtype,
        )

        cell_offsets = np.asarray(self.grid_cell_offsets)
        if cell_offsets[0] != 0:
            raise ValueError("grid cell offsets must start at zero")
        if cell_offsets[-1] != len(self.grid_track_ids):
            raise ValueError(
                "last grid cell offset must equal track_ids length")
        if np.any(cell_offsets[1:] < cell_offsets[:-1]):
            raise ValueError("grid cell offsets must be non-decreasing")
        if len(self.grid_cells) > 1:
            previous = self.grid_cells[:-1].astype(np.int64)
            following = self.grid_cells[1:].astype(np.int64)
            sorted_pairs = (
                (following[:, 0] > previous[:, 0])
                | (
                    (following[:, 0] == previous[:, 0])
                    & (following[:, 1] > previous[:, 1])
                )
            )
            if not sorted_pairs.all():
                raise ValueError("grid cells must be sorted and unique")
        for cell_index in range(len(self.grid_cells)):
            start = int(cell_offsets[cell_index])
            end = int(cell_offsets[cell_index + 1])
            track_ids = self.grid_track_ids[start:end]
            if len(track_ids) > 1 and np.any(track_ids[1:] <= track_ids[:-1]):
                raise ValueError(
                    "track ids for each grid cell must be sorted and unique")
        if len(self.grid_track_ids):
            if int(self.grid_track_ids.min()) < 0:
                raise ValueError("grid index contains a negative track id")
            if int(self.grid_track_ids.max()) >= trajectory_count:
                raise ValueError("grid index contains an out-of-range track id")
        if int(self.meta.get("grid_cell_count", -1)) != len(self.grid_cells):
            raise ValueError("meta grid_cell_count does not match grid index")
        if int(self.meta.get("grid_membership_count", -1)) != len(
                self.grid_track_ids):
            raise ValueError(
                "meta grid_membership_count does not match grid index")

        expected_grid_memberships = 0
        for track_index in range(trajectory_count):
            start = int(offsets[track_index])
            end = int(offsets[track_index + 1])
            occupied_cells = np.unique(
                np.floor(self.points_xy[start:end] / self.cell_size).astype(
                    np.int32),
                axis=0,
            )
            expected_grid_memberships += len(occupied_cells)
            for cell_x, cell_y in occupied_cells:
                cell_index = self._cell_to_index.get(
                    (int(cell_x), int(cell_y)))
                if cell_index is None:
                    raise ValueError(
                        "grid index is missing cell ({}, {}) for track {}"
                        .format(cell_x, cell_y, track_index))
                cell_start = int(cell_offsets[cell_index])
                cell_end = int(cell_offsets[cell_index + 1])
                cell_track_ids = self.grid_track_ids[cell_start:cell_end]
                position = int(np.searchsorted(
                    cell_track_ids, track_index))
                if (
                    position >= len(cell_track_ids)
                    or int(cell_track_ids[position]) != track_index
                ):
                    raise ValueError(
                        "grid cell ({}, {}) is missing track {}".format(
                            cell_x, cell_y, track_index))
        if expected_grid_memberships != len(self.grid_track_ids):
            raise ValueError(
                "grid index contains unexpected or missing memberships")

        return {
            "passed": True,
            "schema_version": SCHEMA_VERSION,
            "trajectory_count": trajectory_count,
            "point_count": point_count,
            "grid_cell_count": int(self.grid_cells.shape[0]),
            "grid_membership_count": int(self.grid_track_ids.shape[0]),
            "memory_mapped": {
                "points_xy": isinstance(self.points_xy, np.memmap),
                "timestamps_ns": isinstance(self.timestamps_ns, np.memmap),
                "track_offsets": isinstance(self.track_offsets, np.memmap),
            },
        }

    def _validate_meta_array(
            self,
            name: str,
            shape: tuple,
            dtype: np.dtype) -> None:
        arrays = self.meta.get("arrays")
        if not isinstance(arrays, dict) or name not in arrays:
            raise ValueError("meta is missing array description {!r}".format(name))
        description = arrays[name]
        if list(description.get("shape", [])) != list(shape):
            raise ValueError(
                "meta shape for {} does not match file".format(name))
        if str(description.get("dtype")) != np.dtype(dtype).name:
            raise ValueError(
                "meta dtype for {} does not match file".format(name))

    @staticmethod
    def _parse_record_timestamp(
            value: Any,
            track_index: int,
            field: str) -> int:
        try:
            timestamp = np.datetime64(str(value), "ns")
        except (TypeError, ValueError) as error:
            raise ValueError(
                "track {} has invalid {} {!r}".format(
                    track_index, field, value)) from error
        timestamp_ns = int(timestamp.astype(np.int64))
        if timestamp_ns == np.iinfo(np.int64).min:
            raise ValueError(
                "track {} has NaT in {}".format(track_index, field))
        return timestamp_ns


def open_structured_trajectory_store(
        cache_dir: str) -> StructuredTrajectoryStore:
    """Open a Stage 1A cache without copying its large arrays into memory."""

    return StructuredTrajectoryStore(Path(cache_dir))
