"""Deterministic non-learned compression of high-recall trajectory fragments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from utils.trajectory_fragments import (
    TrajectoryFragment,
    segment_rectangle_interval,
)


VALID_COMPRESSION_STRATEGIES = ("nearest", "near_diverse")
GEOMETRY_DESCRIPTOR_NAMES = (
    "nearest_x_norm",
    "nearest_y_norm",
    "center_x_norm",
    "center_y_norm",
    "axis_cos_2theta",
    "axis_sin_2theta",
    "coverage_length_norm",
    "minimum_distance_norm",
    "segment_only",
)
GEOMETRY_DESCRIPTOR_WEIGHTS = np.asarray(
    [1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.75, 0.25],
    dtype=np.float32,
)


@dataclass(frozen=True)
class CompressionResult:
    """Selected real fragments and their high-recall support assignments."""

    selected_fragments: Tuple[TrajectoryFragment, ...]
    source_fragment_indices: np.ndarray
    support_count: np.ndarray
    total_fragment_count: int
    kept_fragment_count: int
    truncated_fragment_count: int
    selection_strategy: str
    selected_geometry_descriptors: np.ndarray
    center_xy: Tuple[float, float]
    window_size_xy: Tuple[float, float]
    max_fragments: Optional[int]
    near_fraction: float

    def __post_init__(self) -> None:
        selected_fragments = tuple(self.selected_fragments)
        source_indices = np.asarray(
            self.source_fragment_indices, dtype=np.int64).copy()
        support_count = np.asarray(
            self.support_count, dtype=np.int64).copy()
        descriptors = np.asarray(
            self.selected_geometry_descriptors, dtype=np.float32).copy()
        kept_count = len(selected_fragments)
        if source_indices.shape != (kept_count,):
            raise ValueError(
                "source_fragment_indices must match selected fragments")
        if support_count.shape != (kept_count,):
            raise ValueError(
                "support_count must match selected fragments")
        if descriptors.shape != (
            kept_count,
            len(GEOMETRY_DESCRIPTOR_NAMES),
        ):
            raise ValueError(
                "selected_geometry_descriptors has an invalid shape")
        if np.any(support_count < 0):
            raise ValueError("support_count must be non-negative")
        if int(support_count.sum()) != int(self.total_fragment_count):
            raise ValueError(
                "support_count must sum to total_fragment_count")
        if int(self.kept_fragment_count) != kept_count:
            raise ValueError(
                "kept_fragment_count does not match selected fragments")
        if (
            int(self.truncated_fragment_count)
            != int(self.total_fragment_count) - kept_count
        ):
            raise ValueError("truncated_fragment_count is inconsistent")
        source_indices.setflags(write=False)
        support_count.setflags(write=False)
        descriptors.setflags(write=False)
        object.__setattr__(self, "selected_fragments", selected_fragments)
        object.__setattr__(self, "source_fragment_indices", source_indices)
        object.__setattr__(self, "support_count", support_count)
        object.__setattr__(
            self, "selected_geometry_descriptors", descriptors)


def _parse_center(center_xy: Sequence[float]) -> np.ndarray:
    center = np.asarray(center_xy, dtype=np.float64)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("center_xy must contain finite x and y")
    return center


def _parse_window_size(
    window_size: Union[float, Sequence[float]],
) -> np.ndarray:
    if np.isscalar(window_size):
        size_xy = np.asarray(
            [float(window_size), float(window_size)],
            dtype=np.float64,
        )
    else:
        size_xy = np.asarray(window_size, dtype=np.float64)
        if size_xy.shape != (2,):
            raise ValueError(
                "window_size must be a scalar or [width, height]")
    if not np.isfinite(size_xy).all() or np.any(size_xy <= 0.0):
        raise ValueError("window_size must be finite and positive")
    return size_xy


def _validate_fragment(fragment: TrajectoryFragment) -> np.ndarray:
    points = np.asarray(fragment.points_global_xy, dtype=np.float64)
    timestamps = np.asarray(fragment.timestamps_ns)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        raise ValueError(
            "each fragment must contain at least one [x, y] point")
    if not np.isfinite(points).all():
        raise ValueError("fragment points must be finite")
    if timestamps.shape != (points.shape[0],):
        raise ValueError(
            "fragment timestamps must match its point count")
    if int(fragment.start_point_index) < 0:
        raise ValueError(
            "fragment start_point_index must be non-negative")
    if int(fragment.end_point_index) <= int(
        fragment.start_point_index
    ):
        raise ValueError(
            "fragment end_point_index must exceed start_point_index")
    if (
        int(fragment.end_point_index)
        - int(fragment.start_point_index)
        != points.shape[0]
    ):
        raise ValueError(
            "fragment point-index range must match its point count")
    return points


def fragment_minimum_distance(
    fragment: TrajectoryFragment,
    center_xy: Sequence[float],
) -> float:
    """Return exact minimum distance from a node to a fragment polyline."""

    center = _parse_center(center_xy)
    points = _validate_fragment(fragment) - center
    minimum = float(np.min(np.hypot(points[:, 0], points[:, 1])))
    if points.shape[0] == 1:
        return minimum

    segment_start = points[:-1]
    segment_delta = points[1:] - segment_start
    squared_length = np.sum(segment_delta * segment_delta, axis=1)
    nonzero = squared_length > 0.0
    if np.any(nonzero):
        projection = np.zeros_like(squared_length)
        projection[nonzero] = np.clip(
            -np.sum(
                segment_start[nonzero] * segment_delta[nonzero],
                axis=1,
            )
            / squared_length[nonzero],
            0.0,
            1.0,
        )
        nearest = segment_start + projection[:, None] * segment_delta
        segment_distances = np.hypot(nearest[:, 0], nearest[:, 1])
        minimum = min(minimum, float(np.min(segment_distances)))
    return minimum


def _nearest_position_axis(
    relative_points: np.ndarray,
) -> Tuple[np.ndarray, float, np.ndarray]:
    point_distances = np.hypot(
        relative_points[:, 0], relative_points[:, 1])
    nearest_point_index = int(np.argmin(point_distances))
    nearest_position = relative_points[nearest_point_index].copy()
    minimum_distance = float(point_distances[nearest_point_index])
    axis = np.zeros((2,), dtype=np.float64)
    if relative_points.shape[0] == 1:
        return nearest_position, minimum_distance, axis

    segment_start = relative_points[:-1]
    segment_delta = relative_points[1:] - segment_start
    squared_length = np.sum(segment_delta * segment_delta, axis=1)
    nonzero_indices = np.flatnonzero(squared_length > 0.0)
    if nonzero_indices.size == 0:
        return nearest_position, minimum_distance, axis
    nonzero_start = segment_start[nonzero_indices]
    nonzero_delta = segment_delta[nonzero_indices]
    nonzero_squared_length = squared_length[nonzero_indices]
    projection = np.clip(
        -np.sum(nonzero_start * nonzero_delta, axis=1)
        / nonzero_squared_length,
        0.0,
        1.0,
    )
    projected_positions = (
        nonzero_start + projection[:, None] * nonzero_delta)
    projected_distances = np.hypot(
        projected_positions[:, 0], projected_positions[:, 1])
    nearest_local_index = int(np.argmin(projected_distances))
    if float(projected_distances[nearest_local_index]) <= minimum_distance:
        nearest_position = projected_positions[nearest_local_index]
        minimum_distance = float(
            projected_distances[nearest_local_index])
        tangent = nonzero_delta[nearest_local_index]
        tangent = tangent / np.linalg.norm(tangent)
        axis = np.asarray(
            [
                tangent[0] * tangent[0] - tangent[1] * tangent[1],
                2.0 * tangent[0] * tangent[1],
            ],
            dtype=np.float64,
        )
    return nearest_position, minimum_distance, axis


def _coverage_length_inside_window(
    relative_points: np.ndarray,
    size_xy: np.ndarray,
) -> float:
    if relative_points.shape[0] < 2:
        return 0.0
    half_size = size_xy / 2.0
    bounds = (
        -float(half_size[0]),
        -float(half_size[1]),
        float(half_size[0]),
        float(half_size[1]),
    )
    coverage_length = 0.0
    for point_index in range(relative_points.shape[0] - 1):
        point_a = relative_points[point_index]
        point_b = relative_points[point_index + 1]
        interval = segment_rectangle_interval(
            point_a, point_b, bounds)
        if interval is None:
            continue
        segment_length = float(np.linalg.norm(point_b - point_a))
        coverage_length += (
            segment_length * max(0.0, interval[1] - interval[0]))
    return coverage_length


def trajectory_fragment_geometry_descriptor(
    fragment: TrajectoryFragment,
    center_xy: Sequence[float],
    window_size: Union[float, Sequence[float]],
) -> np.ndarray:
    """Return a cheap continuous, direction-axis-aware geometry descriptor."""

    center = _parse_center(center_xy)
    size_xy = _parse_window_size(window_size)
    half_size = size_xy / 2.0
    points = _validate_fragment(fragment)
    relative_points = points - center
    nearest_position, minimum_distance, axis = _nearest_position_axis(
        relative_points)
    center_position = np.mean(relative_points, axis=0)
    coverage_length = _coverage_length_inside_window(
        relative_points, size_xy)
    inside = (
        (relative_points[:, 0] >= -half_size[0])
        & (relative_points[:, 0] <= half_size[0])
        & (relative_points[:, 1] >= -half_size[1])
        & (relative_points[:, 1] <= half_size[1])
    )
    segment_only = not bool(inside.any())
    half_diagonal = float(np.linalg.norm(half_size))
    window_diagonal = float(np.linalg.norm(size_xy))
    descriptor = np.asarray(
        [
            nearest_position[0] / half_size[0],
            nearest_position[1] / half_size[1],
            center_position[0] / half_size[0],
            center_position[1] / half_size[1],
            axis[0],
            axis[1],
            coverage_length / window_diagonal,
            minimum_distance / half_diagonal,
            float(segment_only),
        ],
        dtype=np.float64,
    )
    descriptor[0:4] = np.clip(descriptor[0:4], -2.0, 2.0)
    descriptor[6] = np.clip(descriptor[6], 0.0, 4.0)
    descriptor[7] = np.clip(descriptor[7], 0.0, 2.0)
    return descriptor.astype(np.float32)


def build_trajectory_geometry_descriptors(
    fragments: Sequence[TrajectoryFragment],
    center_xy: Sequence[float],
    window_size: Union[float, Sequence[float]],
) -> np.ndarray:
    descriptors = np.empty(
        (len(fragments), len(GEOMETRY_DESCRIPTOR_NAMES)),
        dtype=np.float32,
    )
    for fragment_index, fragment in enumerate(fragments):
        descriptors[fragment_index] = (
            trajectory_fragment_geometry_descriptor(
                fragment,
                center_xy,
                window_size,
            )
        )
    return descriptors


def _fragment_identity_key(
    fragment: TrajectoryFragment,
) -> Tuple:
    return (
        int(fragment.track_index),
        int(fragment.start_point_index),
        int(fragment.end_point_index),
        str(fragment.source_traj_id),
        str(fragment.source_file),
    )


def _nearest_indices(
    fragments: Sequence[TrajectoryFragment],
    minimum_distances: np.ndarray,
    kept_count: int,
) -> List[int]:
    return sorted(
        range(len(fragments)),
        key=lambda fragment_index: (
            float(minimum_distances[fragment_index]),
            int(fragments[fragment_index].track_index),
            int(fragments[fragment_index].start_point_index),
            int(fragments[fragment_index].end_point_index),
            fragment_index,
        ),
    )[:kept_count]


def _near_diverse_indices(
    fragments: Sequence[TrajectoryFragment],
    descriptors: np.ndarray,
    minimum_distances: np.ndarray,
    kept_count: int,
    near_fraction: float,
) -> List[int]:
    total_count = len(fragments)
    canonical_keys = [
        _fragment_identity_key(fragment) for fragment in fragments
    ]
    if kept_count >= total_count:
        return sorted(
            range(total_count),
            key=lambda fragment_index: canonical_keys[fragment_index],
        )

    near_count = min(
        kept_count,
        max(1, int(math.ceil(kept_count * near_fraction))),
    )
    selected = sorted(
        range(total_count),
        key=lambda fragment_index: (
            float(minimum_distances[fragment_index]),
            canonical_keys[fragment_index],
        ),
    )[:near_count]
    selected_set = set(selected)
    weighted_descriptors = (
        descriptors * GEOMETRY_DESCRIPTOR_WEIGHTS[None, :])
    selected_descriptors = weighted_descriptors[selected]
    differences = (
        weighted_descriptors[:, None, :]
        - selected_descriptors[None, :, :]
    )
    minimum_squared_distance = np.min(
        np.sum(differences * differences, axis=2), axis=1)
    minimum_squared_distance[selected] = -np.inf

    while len(selected) < kept_count:
        diversity = np.sqrt(
            np.maximum(minimum_squared_distance, 0.0))
        proximity_weight = 1.0 / (
            1.0 + descriptors[:, 7].astype(np.float64))
        score = diversity * proximity_weight
        if selected_set:
            score[np.fromiter(
                selected_set,
                dtype=np.int64,
                count=len(selected_set),
            )] = -np.inf
        maximum_score = float(np.max(score))
        tied_indices = np.flatnonzero(score == maximum_score)
        next_index = min(
            (int(index) for index in tied_indices),
            key=lambda fragment_index: canonical_keys[fragment_index],
        )
        selected.append(next_index)
        selected_set.add(next_index)
        difference = (
            weighted_descriptors
            - weighted_descriptors[next_index][None, :]
        )
        squared_distance = np.sum(
            difference * difference, axis=1)
        minimum_squared_distance = np.minimum(
            minimum_squared_distance, squared_distance)
        minimum_squared_distance[next_index] = -np.inf
    return selected


def _support_counts(
    descriptors: np.ndarray,
    selected_indices: Sequence[int],
) -> np.ndarray:
    total_count = descriptors.shape[0]
    kept_count = len(selected_indices)
    if total_count == 0:
        return np.empty((0,), dtype=np.int64)
    support_count = np.ones((kept_count,), dtype=np.int64)
    selected_mask = np.zeros((total_count,), dtype=np.bool_)
    selected_mask[np.asarray(selected_indices, dtype=np.int64)] = True
    unselected_indices = np.flatnonzero(~selected_mask)
    if unselected_indices.size == 0:
        return support_count

    weighted = descriptors * GEOMETRY_DESCRIPTOR_WEIGHTS[None, :]
    representative_descriptors = weighted[
        np.asarray(selected_indices, dtype=np.int64)]
    chunk_size = 512
    for chunk_start in range(0, unselected_indices.size, chunk_size):
        chunk_indices = unselected_indices[
            chunk_start:chunk_start + chunk_size]
        differences = (
            weighted[chunk_indices, None, :]
            - representative_descriptors[None, :, :]
        )
        squared_distances = np.sum(
            differences * differences, axis=2)
        assignments = np.argmin(squared_distances, axis=1)
        support_count += np.bincount(
            assignments, minlength=kept_count).astype(np.int64)
    return support_count


def compress_trajectory_fragments(
    fragments: Sequence[TrajectoryFragment],
    center_xy: Sequence[float],
    window_size: Union[float, Sequence[float]],
    max_fragments: Optional[int],
    strategy: str = "near_diverse",
    near_fraction: float = 0.25,
    geometry_descriptors: Optional[np.ndarray] = None,
    minimum_distances: Optional[np.ndarray] = None,
) -> CompressionResult:
    """Compress full Stage 1B candidates without using image or GT evidence."""

    if not isinstance(fragments, (list, tuple)):
        raise TypeError("fragments must be a list or tuple")
    center = _parse_center(center_xy)
    size_xy = _parse_window_size(window_size)
    if strategy not in VALID_COMPRESSION_STRATEGIES:
        raise ValueError(
            "unknown compression strategy {!r}; expected {}".format(
                strategy, ", ".join(VALID_COMPRESSION_STRATEGIES)))
    near_fraction = float(near_fraction)
    if not math.isfinite(near_fraction) or not 0.0 <= near_fraction <= 1.0:
        raise ValueError("near_fraction must be in [0, 1]")
    if max_fragments is not None:
        if isinstance(max_fragments, bool) or not isinstance(
            max_fragments, (int, np.integer)
        ):
            raise TypeError("max_fragments must be an integer or None")
        max_fragments = int(max_fragments)
        if max_fragments < 0:
            raise ValueError("max_fragments must be non-negative")

    total_count = len(fragments)
    if total_count > 0 and max_fragments == 0:
        raise ValueError(
            "max_fragments must retain at least one representative "
            "when candidates are non-empty")
    if geometry_descriptors is None:
        descriptors = build_trajectory_geometry_descriptors(
            fragments, center, size_xy)
    else:
        descriptors = np.asarray(
            geometry_descriptors, dtype=np.float32)
        if descriptors.shape != (
            total_count,
            len(GEOMETRY_DESCRIPTOR_NAMES),
        ):
            raise ValueError(
                "geometry_descriptors has an invalid shape")
        if not np.isfinite(descriptors).all():
            raise ValueError(
                "geometry_descriptors must contain finite values")
    if minimum_distances is None:
        distance_values = np.asarray(
            [
                fragment_minimum_distance(fragment, center)
                for fragment in fragments
            ],
            dtype=np.float64,
        )
    else:
        distance_values = np.asarray(
            minimum_distances, dtype=np.float64)
        if distance_values.shape != (total_count,):
            raise ValueError("minimum_distances has an invalid shape")
        if not np.isfinite(distance_values).all():
            raise ValueError(
                "minimum_distances must contain finite values")

    kept_count = (
        total_count
        if max_fragments is None
        else min(total_count, max_fragments)
    )
    if kept_count == 0:
        selected_indices: List[int] = []
    elif max_fragments is None:
        selected_indices = list(range(total_count))
    elif strategy == "nearest":
        selected_indices = _nearest_indices(
            fragments, distance_values, kept_count)
    else:
        selected_indices = _near_diverse_indices(
            fragments,
            descriptors,
            distance_values,
            kept_count,
            near_fraction,
        )
    support_count = _support_counts(
        descriptors, selected_indices)
    selected_indices_array = np.asarray(
        selected_indices, dtype=np.int64)
    return CompressionResult(
        selected_fragments=tuple(
            fragments[index] for index in selected_indices),
        source_fragment_indices=selected_indices_array,
        support_count=support_count,
        total_fragment_count=total_count,
        kept_fragment_count=kept_count,
        truncated_fragment_count=total_count - kept_count,
        selection_strategy=strategy,
        selected_geometry_descriptors=descriptors[
            selected_indices_array
        ],
        center_xy=(float(center[0]), float(center[1])),
        window_size_xy=(float(size_xy[0]), float(size_xy[1])),
        max_fragments=max_fragments,
        near_fraction=near_fraction,
    )
