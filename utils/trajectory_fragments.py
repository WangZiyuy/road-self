"""Exact, identity-preserving local trajectory fragment retrieval."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from utils.structured_trajectory_store import StructuredTrajectoryStore


POINT_GRID_INDEX_BASIS = "trajectory_sample_point_cells"
SEGMENT_GRID_INDEX_BASIS = (
    "trajectory_sample_points_and_segments_supercover_cells"
)


@dataclass(frozen=True)
class TrajectoryFragment:
    """One continuous visit of a source trajectory to a query window.

    ``start_point_index`` is inclusive and ``end_point_index`` is exclusive.
    Both indices are local to the original track, not the flattened cache.
    Context points are included in these bounds and returned arrays.
    """

    track_index: int
    source_traj_id: str
    source_file: str
    start_point_index: int
    end_point_index: int
    points_global_xy: np.ndarray
    points_relative_xy: np.ndarray
    timestamps_ns: np.ndarray

    def __len__(self) -> int:
        return int(self.points_global_xy.shape[0])


def segment_rectangle_interval(
    point_a: Sequence[float],
    point_b: Sequence[float],
    bounds: Tuple[float, float, float, float],
) -> Optional[Tuple[float, float]]:
    """Return the closed segment parameter interval inside a rectangle."""

    x_min, y_min, x_max, y_max = bounds
    point_a_array = np.asarray(point_a, dtype=np.float64)
    point_b_array = np.asarray(point_b, dtype=np.float64)
    if point_a_array.shape != (2,) or point_b_array.shape != (2,):
        raise ValueError("segment endpoints must each contain x and y")
    if not np.isfinite(point_a_array).all() or not np.isfinite(
        point_b_array
    ).all():
        raise ValueError("segment endpoints must be finite")

    delta = point_b_array - point_a_array
    parameter_min = 0.0
    parameter_max = 1.0
    for origin, direction, lower, upper in (
        (point_a_array[0], delta[0], x_min, x_max),
        (point_a_array[1], delta[1], y_min, y_max),
    ):
        if direction == 0.0:
            if origin < lower or origin > upper:
                return None
            continue
        axis_min = (lower - origin) / direction
        axis_max = (upper - origin) / direction
        if axis_min > axis_max:
            axis_min, axis_max = axis_max, axis_min
        parameter_min = max(parameter_min, float(axis_min))
        parameter_max = min(parameter_max, float(axis_max))
        if parameter_min > parameter_max:
            return None
    return parameter_min, parameter_max


def _point_in_rectangle(
    point: np.ndarray,
    bounds: Tuple[float, float, float, float],
) -> bool:
    x_min, y_min, x_max, y_max = bounds
    return bool(
        x_min <= float(point[0]) <= x_max
        and y_min <= float(point[1]) <= y_max
    )


def _lower_supercover_cell(value: float, cell_size: int) -> int:
    """Include both cells when a closed segment lies on a cell boundary."""

    scaled = value / cell_size
    cell = math.floor(scaled)
    nearest_integer = round(scaled)
    if math.isclose(
        scaled,
        nearest_integer,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        cell -= 1
    return cell


def trajectory_grid_cells(
    points_xy: np.ndarray,
    cell_size: int,
    include_segments: bool = True,
) -> np.ndarray:
    """Return sorted unique cells occupied by points or traversed segments.

    The segment mode is a geometric supercover: a cell is registered whenever
    its closed rectangle intersects a consecutive trajectory segment. It is
    deliberately independent of query-time gap thresholds so that the index
    remains a high-recall candidate source.
    """

    points = np.asarray(points_xy)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_xy must have shape [point_count, 2]")
    if points.shape[0] == 0:
        return np.empty((0, 2), dtype=np.int32)
    if not np.isfinite(points).all():
        raise ValueError("points_xy must be finite")
    if isinstance(cell_size, bool) or int(cell_size) != cell_size:
        raise TypeError("cell_size must be an integer")
    cell_size = int(cell_size)
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")

    occupied = {
        (
            math.floor(float(point[0]) / cell_size),
            math.floor(float(point[1]) / cell_size),
        )
        for point in points
    }
    if include_segments:
        for point_index in range(points.shape[0] - 1):
            point_a = points[point_index].astype(np.float64, copy=False)
            point_b = points[point_index + 1].astype(
                np.float64, copy=False)
            x_min = min(float(point_a[0]), float(point_b[0]))
            x_max = max(float(point_a[0]), float(point_b[0]))
            y_min = min(float(point_a[1]), float(point_b[1]))
            y_max = max(float(point_a[1]), float(point_b[1]))
            min_cell_x = _lower_supercover_cell(x_min, cell_size)
            max_cell_x = math.floor(x_max / cell_size)
            min_cell_y = _lower_supercover_cell(y_min, cell_size)
            max_cell_y = math.floor(y_max / cell_size)
            for cell_x in range(min_cell_x, max_cell_x + 1):
                cell_x_min = cell_x * cell_size
                cell_x_max = (cell_x + 1) * cell_size
                for cell_y in range(min_cell_y, max_cell_y + 1):
                    cell_bounds = (
                        float(cell_x_min),
                        float(cell_y * cell_size),
                        float(cell_x_max),
                        float((cell_y + 1) * cell_size),
                    )
                    if segment_rectangle_interval(
                        point_a, point_b, cell_bounds
                    ) is not None:
                        occupied.add((cell_x, cell_y))
    return np.asarray(sorted(occupied), dtype=np.int32).reshape((-1, 2))


def _parse_center(center_xy: Sequence[float]) -> np.ndarray:
    center = np.asarray(center_xy, dtype=np.float64)
    if center.shape != (2,):
        raise ValueError("center_xy must contain exactly x and y")
    if not np.isfinite(center).all():
        raise ValueError("center_xy must be finite")
    return center


def _parse_window_size(
    window_size: Union[float, Sequence[float]],
) -> Tuple[float, float]:
    if np.isscalar(window_size):
        width = height = float(window_size)
    else:
        values = np.asarray(window_size, dtype=np.float64)
        if values.shape != (2,):
            raise ValueError(
                "window_size must be a scalar or [width, height]")
        width, height = map(float, values)
    if not math.isfinite(width) or not math.isfinite(height):
        raise ValueError("window_size must be finite")
    if width <= 0.0 or height <= 0.0:
        raise ValueError("window_size must be positive")
    return width, height


def _parse_optional_threshold(
    value: Optional[float],
    name: str,
) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("{} must be finite and non-negative".format(name))
    return value


def _continuous_piece_ranges(
    points_xy: np.ndarray,
    timestamps_ns: np.ndarray,
    max_time_gap_seconds: Optional[float],
    max_spatial_gap_pixels: Optional[float],
) -> List[Tuple[int, int]]:
    point_count = int(points_xy.shape[0])
    if point_count == 0:
        return []
    if point_count == 1:
        return [(0, 1)]

    edge_is_continuous = np.ones((point_count - 1,), dtype=np.bool_)
    if max_time_gap_seconds is not None:
        max_time_gap_ns = max_time_gap_seconds * 1_000_000_000.0
        for edge_index in range(point_count - 1):
            time_gap_ns = abs(
                int(timestamps_ns[edge_index + 1])
                - int(timestamps_ns[edge_index])
            )
            if time_gap_ns > max_time_gap_ns:
                edge_is_continuous[edge_index] = False
    if max_spatial_gap_pixels is not None:
        deltas = (
            points_xy[1:].astype(np.float64)
            - points_xy[:-1].astype(np.float64)
        )
        distances = np.hypot(deltas[:, 0], deltas[:, 1])
        edge_is_continuous &= distances <= max_spatial_gap_pixels

    pieces: List[Tuple[int, int]] = []
    piece_start = 0
    for edge_index, is_continuous in enumerate(edge_is_continuous):
        if not bool(is_continuous):
            pieces.append((piece_start, edge_index + 1))
            piece_start = edge_index + 1
    pieces.append((piece_start, point_count))
    return pieces


def _window_visit_intervals(
    points_xy: np.ndarray,
    piece_start: int,
    piece_end: int,
    bounds: Tuple[float, float, float, float],
) -> List[Tuple[float, float]]:
    if piece_end - piece_start == 1:
        if _point_in_rectangle(points_xy[piece_start], bounds):
            position = float(piece_start)
            return [(position, position)]
        return []

    point_a = points_xy[piece_start:piece_end - 1].astype(
        np.float64, copy=False)
    point_b = points_xy[piece_start + 1:piece_end].astype(
        np.float64, copy=False)
    direction = point_b - point_a
    segment_count = point_a.shape[0]
    parameter_min = np.zeros((segment_count,), dtype=np.float64)
    parameter_max = np.ones((segment_count,), dtype=np.float64)
    is_valid = np.ones((segment_count,), dtype=np.bool_)
    for axis, lower, upper in (
        (0, bounds[0], bounds[2]),
        (1, bounds[1], bounds[3]),
    ):
        origin = point_a[:, axis]
        axis_direction = direction[:, axis]
        is_parallel = axis_direction == 0.0
        is_valid &= ~(
            is_parallel & ((origin < lower) | (origin > upper))
        )
        is_nonparallel = ~is_parallel
        nonparallel_indices = np.flatnonzero(is_nonparallel)
        if nonparallel_indices.size:
            axis_min = (
                lower - origin[nonparallel_indices]
            ) / axis_direction[nonparallel_indices]
            axis_max = (
                upper - origin[nonparallel_indices]
            ) / axis_direction[nonparallel_indices]
            lower_parameter = np.minimum(axis_min, axis_max)
            upper_parameter = np.maximum(axis_min, axis_max)
            parameter_min[nonparallel_indices] = np.maximum(
                parameter_min[nonparallel_indices], lower_parameter)
            parameter_max[nonparallel_indices] = np.minimum(
                parameter_max[nonparallel_indices], upper_parameter)
        is_valid &= parameter_min <= parameter_max

    visits: List[Tuple[float, float]] = []
    for local_edge_index in np.flatnonzero(is_valid):
        point_index = piece_start + int(local_edge_index)
        interval_start = (
            point_index + parameter_min[local_edge_index])
        interval_end = (
            point_index + parameter_max[local_edge_index])
        if visits and interval_start <= visits[-1][1] + 1e-12:
            visits[-1] = (
                visits[-1][0],
                max(visits[-1][1], interval_end),
            )
        else:
            visits.append((interval_start, interval_end))
    return visits


def query_trajectory_fragments(
    store: "StructuredTrajectoryStore",
    center_xy: Sequence[float],
    window_size: Union[float, Sequence[float]] = 256,
    context_points: int = 2,
    max_time_gap_seconds: Optional[float] = None,
    max_spatial_gap_pixels: Optional[float] = None,
) -> List[TrajectoryFragment]:
    """Retrieve exact, continuous visits to a local window.

    Results are deterministic: tracks are ordered by ``track_index`` and
    multiple visits from one track retain original point/time order.
    """

    center = _parse_center(center_xy)
    width, height = _parse_window_size(window_size)
    if isinstance(context_points, bool) or not isinstance(
        context_points, (int, np.integer)
    ):
        raise TypeError("context_points must be an integer")
    context_points = int(context_points)
    if context_points < 0:
        raise ValueError("context_points must be non-negative")
    max_time_gap_seconds = _parse_optional_threshold(
        max_time_gap_seconds, "max_time_gap_seconds")
    max_spatial_gap_pixels = _parse_optional_threshold(
        max_spatial_gap_pixels, "max_spatial_gap_pixels")

    bounds = (
        float(center[0] - width / 2.0),
        float(center[1] - height / 2.0),
        float(center[0] + width / 2.0),
        float(center[1] + height / 2.0),
    )
    grid_basis = store.meta.get(
        "grid_index_basis", POINT_GRID_INDEX_BASIS)
    if grid_basis == SEGMENT_GRID_INDEX_BASIS:
        candidate_track_ids = store.candidate_track_ids_for_rect(*bounds)
    elif grid_basis == POINT_GRID_INDEX_BASIS:
        # Stage 1A point-only caches cannot guarantee segment recall. A full
        # scan preserves correctness until the cache is rebuilt by Stage 1B.
        candidate_track_ids = np.arange(
            store.trajectory_count, dtype=np.int32)
    else:
        raise ValueError(
            "unsupported grid_index_basis {!r}".format(grid_basis))

    fragments: List[TrajectoryFragment] = []
    center_float32 = center.astype(np.float32)
    for track_index_value in candidate_track_ids:
        track = store.get_track(int(track_index_value))
        if (
            max_time_gap_seconds is None
            and max_spatial_gap_pixels is None
        ):
            pieces = [(0, len(track))]
        else:
            pieces = _continuous_piece_ranges(
                track.points_xy,
                track.timestamps_ns,
                max_time_gap_seconds,
                max_spatial_gap_pixels,
            )
        for piece_start, piece_end in pieces:
            visits = _window_visit_intervals(
                track.points_xy,
                piece_start,
                piece_end,
                bounds,
            )
            for visit_start, visit_end in visits:
                core_start = max(
                    piece_start, int(math.floor(visit_start)))
                core_end = min(
                    piece_end, int(math.ceil(visit_end)) + 1)
                fragment_start = max(
                    piece_start, core_start - context_points)
                fragment_end = min(
                    piece_end, core_end + context_points)
                points_global_xy = track.points_xy[
                    fragment_start:fragment_end
                ]
                timestamps_ns = track.timestamps_ns[
                    fragment_start:fragment_end
                ]
                points_relative_xy = (
                    np.asarray(points_global_xy, dtype=np.float32)
                    - center_float32
                )
                points_relative_xy.setflags(write=False)
                fragments.append(
                    TrajectoryFragment(
                        track_index=track.track_index,
                        source_traj_id=track.source_traj_id,
                        source_file=track.source_file,
                        start_point_index=fragment_start,
                        end_point_index=fragment_end,
                        points_global_xy=points_global_xy,
                        points_relative_xy=points_relative_xy,
                        timestamps_ns=timestamps_ns,
                    )
                )
    return fragments
