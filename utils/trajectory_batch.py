"""Build explicit padded PyTorch batches from Stage 1B trajectory fragments."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch

from utils.trajectory_compression import (
    CompressionResult,
    compress_trajectory_fragments,
    fragment_minimum_distance,
)
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


def _prepare_sample_fragments(
    sample,
    center_xy: np.ndarray,
    size_xy: np.ndarray,
    max_fragments: Optional[int],
):
    if isinstance(sample, CompressionResult):
        if not np.allclose(
            np.asarray(sample.center_xy, dtype=np.float64),
            center_xy,
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(
                "CompressionResult center does not match batch center")
        if not np.allclose(
            np.asarray(sample.window_size_xy, dtype=np.float64),
            size_xy,
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(
                "CompressionResult window does not match batch window")
        prepared = [
            (
                int(source_index),
                fragment,
                fragment_minimum_distance(fragment, center_xy),
                int(support_count),
            )
            for source_index, fragment, support_count in zip(
                sample.source_fragment_indices,
                sample.selected_fragments,
                sample.support_count,
            )
        ]
        return (
            prepared,
            int(sample.total_fragment_count),
            int(sample.kept_fragment_count),
            int(sample.truncated_fragment_count),
        )

    if not isinstance(sample, (list, tuple)):
        raise TypeError(
            "each fragment_lists item must be a fragment sequence or "
            "CompressionResult")
    total_count = len(sample)
    if max_fragments is None:
        prepared = [
            (
                source_index,
                fragment,
                fragment_minimum_distance(fragment, center_xy),
                1,
            )
            for source_index, fragment in enumerate(sample)
        ]
        return prepared, total_count, total_count, 0
    if max_fragments == 0:
        return [], total_count, 0, total_count

    compression = compress_trajectory_fragments(
        fragments=sample,
        center_xy=center_xy,
        window_size=size_xy,
        max_fragments=max_fragments,
        strategy="nearest",
    )
    return _prepare_sample_fragments(
        compression,
        center_xy,
        size_xy,
        max_fragments=None,
    )


def build_trajectory_batch(
    fragment_lists: Sequence[
        Union[Sequence[TrajectoryFragment], CompressionResult]
    ],
    center_xy,
    window_size: Union[float, Sequence[float]],
    max_fragments: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Pad variable fragment sets without flattening fragment identity.

    All returned tensors are CPU tensors. Padding is represented exclusively
    by ``point_mask`` and ``fragment_mask``; zero coordinates remain valid
    observations when their masks are true. A ``CompressionResult`` is used
    exactly as supplied, including representative support counts and original
    source indices; ``max_fragments`` is not applied to it a second time.
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
        (
            selected,
            total_count,
            kept_count,
            _truncated_count,
        ) = _prepare_sample_fragments(
            fragments,
            centers[batch_index],
            size_xy,
            max_fragments,
        )
        selected_per_sample.append(selected)
        total_counts[batch_index] = total_count
        kept_counts[batch_index] = kept_count
        max_kept_fragments = max(max_kept_fragments, len(selected))
        for _, fragment, _, _ in selected:
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
    fragment_support_count = np.zeros(
        (batch_size, max_kept_fragments), dtype=np.int64)

    for batch_index, selected in enumerate(selected_per_sample):
        center = centers[batch_index]
        for fragment_index, (
            source_index,
            fragment,
            minimum_distance,
            support_count,
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
            fragment_support_count[
                batch_index, fragment_index
            ] = support_count

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
        "fragment_support_count": torch.from_numpy(
            fragment_support_count),
        "total_fragment_count": torch.from_numpy(total_counts),
        "kept_fragment_count": torch.from_numpy(kept_counts),
        "truncated_fragment_count": torch.from_numpy(
            truncated_counts),
        "compression_total_count": torch.from_numpy(
            total_counts.copy()),
        "compression_kept_count": torch.from_numpy(
            kept_counts.copy()),
        "compression_truncated_count": torch.from_numpy(
            truncated_counts.copy()),
    }
