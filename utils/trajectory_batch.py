"""Build explicit padded PyTorch batches from Stage 1B trajectory fragments."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from utils.trajectory_fragments import TrajectoryFragment


def _as_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _parse_centers(center_xy, batch_size: int) -> np.ndarray:
    centers = _as_numpy(center_xy).astype(np.float64, copy=False)
    if centers.shape == (2,):
        centers = np.repeat(centers[None, :], batch_size, axis=0)
    if centers.shape != (batch_size, 2):
        raise ValueError(
            "center_xy must have shape [2] or [batch_size, 2]")
    if not np.isfinite(centers).all():
        raise ValueError("center_xy must contain finite values")
    return centers


def _parse_window_size(
    window_size: Union[float, Sequence[float]],
) -> np.ndarray:
    if np.isscalar(window_size):
        size = np.asarray(
            [float(window_size), float(window_size)], dtype=np.float64)
    else:
        size = _as_numpy(window_size).astype(np.float64, copy=False)
        if size.shape != (2,):
            raise ValueError(
                "window_size must be a scalar or [width, height]")
    if not np.isfinite(size).all() or np.any(size <= 0.0):
        raise ValueError("window_size must be finite and positive")
    return size


def _parse_max_fragments(max_fragments: Optional[int]) -> Optional[int]:
    if max_fragments is None:
        return None
    if isinstance(max_fragments, bool) or not isinstance(
        max_fragments, (int, np.integer)
    ):
        raise TypeError("max_fragments must be an integer or None")
    max_fragments = int(max_fragments)
    if max_fragments < 0:
        raise ValueError("max_fragments must be non-negative")
    return max_fragments


def _validate_fragment(fragment: TrajectoryFragment) -> None:
    points = np.asarray(fragment.points_global_xy)
    timestamps = np.asarray(fragment.timestamps_ns)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        raise ValueError(
            "each fragment must contain at least one [x, y] point")
    if timestamps.shape != (points.shape[0],):
        raise ValueError(
            "fragment timestamps must match its point count")
    if not np.isfinite(points).all():
        raise ValueError("fragment points must be finite")
    if fragment.start_point_index < 0:
        raise ValueError("fragment start_point_index must be non-negative")
    if fragment.end_point_index <= fragment.start_point_index:
        raise ValueError(
            "fragment end_point_index must exceed start_point_index")
    if (
        fragment.end_point_index - fragment.start_point_index
        != points.shape[0]
    ):
        raise ValueError(
            "fragment point-index range must match its point count")


def fragment_minimum_distance(
    fragment: TrajectoryFragment,
    center_xy: Sequence[float],
) -> float:
    """Return the exact minimum distance from a node to a fragment polyline."""

    _validate_fragment(fragment)
    center = np.asarray(center_xy, dtype=np.float64)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("center_xy must contain finite x and y")
    points = (
        np.asarray(fragment.points_global_xy, dtype=np.float64) - center
    )
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


def _select_fragments(
    fragments: Sequence[TrajectoryFragment],
    center_xy: np.ndarray,
    max_fragments: Optional[int],
) -> Tuple[List[Tuple[int, TrajectoryFragment, float]], int]:
    candidates = []
    for source_index, fragment in enumerate(fragments):
        distance = fragment_minimum_distance(fragment, center_xy)
        candidates.append((source_index, fragment, distance))
    if max_fragments is not None:
        candidates.sort(
            key=lambda item: (
                item[2],
                int(item[1].track_index),
                int(item[1].start_point_index),
                int(item[1].end_point_index),
                item[0],
            )
        )
        candidates = candidates[:max_fragments]
    return candidates, len(fragments)


def build_trajectory_batch(
    fragment_lists: Sequence[Sequence[TrajectoryFragment]],
    center_xy,
    window_size: Union[float, Sequence[float]],
    max_fragments: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Pad variable fragment sets without flattening fragment identity.

    All returned tensors are CPU tensors. Padding is represented exclusively
    by ``point_mask`` and ``fragment_mask``; zero coordinates remain valid
    observations when their masks are true.
    """

    if not isinstance(fragment_lists, (list, tuple)):
        raise TypeError("fragment_lists must be a sequence per batch sample")
    batch_size = len(fragment_lists)
    centers = _parse_centers(center_xy, batch_size)
    size_xy = _parse_window_size(window_size)
    half_size_xy = size_xy / 2.0
    max_fragments = _parse_max_fragments(max_fragments)

    selected_per_sample = []
    total_counts = np.zeros((batch_size,), dtype=np.int64)
    kept_counts = np.zeros((batch_size,), dtype=np.int64)
    max_points = 0
    max_kept_fragments = 0
    for batch_index, fragments in enumerate(fragment_lists):
        if not isinstance(fragments, (list, tuple)):
            raise TypeError(
                "each fragment_lists item must be a fragment sequence")
        selected, total_count = _select_fragments(
            fragments,
            centers[batch_index],
            max_fragments,
        )
        selected_per_sample.append(selected)
        total_counts[batch_index] = total_count
        kept_counts[batch_index] = len(selected)
        max_kept_fragments = max(max_kept_fragments, len(selected))
        for _, fragment, _ in selected:
            max_points = max(max_points, len(fragment))

    batch_shape = (
        batch_size,
        max_kept_fragments,
        max_points,
    )
    traj_xy_rel = np.zeros(batch_shape + (2,), dtype=np.float32)
    traj_xy_norm = np.zeros(batch_shape + (2,), dtype=np.float32)
    traj_time_delta = np.zeros(batch_shape, dtype=np.float32)
    point_mask = np.zeros(batch_shape, dtype=np.bool_)
    fragment_mask = np.zeros(
        (batch_size, max_kept_fragments), dtype=np.bool_)
    point_inside_mask = np.zeros(batch_shape, dtype=np.bool_)
    segment_only = np.zeros(
        (batch_size, max_kept_fragments), dtype=np.bool_)
    track_indices = np.full(
        (batch_size, max_kept_fragments), -1, dtype=np.int64)
    start_point_indices = np.full(
        (batch_size, max_kept_fragments), -1, dtype=np.int64)
    end_point_indices = np.full(
        (batch_size, max_kept_fragments), -1, dtype=np.int64)
    source_fragment_indices = np.full(
        (batch_size, max_kept_fragments), -1, dtype=np.int64)
    fragment_min_distance = np.full(
        (batch_size, max_kept_fragments), np.inf, dtype=np.float32)

    for batch_index, selected in enumerate(selected_per_sample):
        center = centers[batch_index]
        for fragment_index, (
            source_index,
            fragment,
            minimum_distance,
        ) in enumerate(selected):
            points = np.asarray(
                fragment.points_global_xy, dtype=np.float64)
            timestamps = np.asarray(fragment.timestamps_ns, dtype=np.int64)
            point_count = points.shape[0]
            relative = points - center
            inside = (
                (relative[:, 0] >= -half_size_xy[0])
                & (relative[:, 0] <= half_size_xy[0])
                & (relative[:, 1] >= -half_size_xy[1])
                & (relative[:, 1] <= half_size_xy[1])
            )
            first_timestamp = int(timestamps[0])
            time_delta_ns = np.fromiter(
                (int(timestamp) - first_timestamp for timestamp in timestamps),
                dtype=np.int64,
                count=point_count,
            )
            time_delta = (
                time_delta_ns.astype(np.float64) / 1_000_000_000.0
            )

            traj_xy_rel[
                batch_index, fragment_index, :point_count
            ] = relative.astype(np.float32)
            traj_xy_norm[
                batch_index, fragment_index, :point_count
            ] = (relative / half_size_xy).astype(np.float32)
            traj_time_delta[
                batch_index, fragment_index, :point_count
            ] = time_delta.astype(np.float32)
            point_mask[
                batch_index, fragment_index, :point_count
            ] = True
            point_inside_mask[
                batch_index, fragment_index, :point_count
            ] = inside
            fragment_mask[batch_index, fragment_index] = True
            segment_only[batch_index, fragment_index] = not bool(
                inside.any())
            track_indices[
                batch_index, fragment_index
            ] = int(fragment.track_index)
            start_point_indices[
                batch_index, fragment_index
            ] = int(fragment.start_point_index)
            end_point_indices[
                batch_index, fragment_index
            ] = int(fragment.end_point_index)
            source_fragment_indices[
                batch_index, fragment_index
            ] = source_index
            fragment_min_distance[
                batch_index, fragment_index
            ] = minimum_distance

    truncated_counts = total_counts - kept_counts
    return {
        "traj_xy_rel": torch.from_numpy(traj_xy_rel),
        "traj_xy_norm": torch.from_numpy(traj_xy_norm),
        "traj_time_delta": torch.from_numpy(traj_time_delta),
        "point_mask": torch.from_numpy(point_mask),
        "fragment_mask": torch.from_numpy(fragment_mask),
        "point_inside_mask": torch.from_numpy(point_inside_mask),
        "segment_only": torch.from_numpy(segment_only),
        "track_indices": torch.from_numpy(track_indices),
        "start_point_indices": torch.from_numpy(start_point_indices),
        "end_point_indices": torch.from_numpy(end_point_indices),
        "source_fragment_indices": torch.from_numpy(
            source_fragment_indices),
        "fragment_min_distance": torch.from_numpy(
            fragment_min_distance),
        "total_fragment_count": torch.from_numpy(total_counts),
        "kept_fragment_count": torch.from_numpy(kept_counts),
        "truncated_fragment_count": torch.from_numpy(
            truncated_counts),
    }
