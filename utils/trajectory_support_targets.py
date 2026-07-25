"""Soft branch-conditioned supervision for structured trajectory fragments.

The labels in this module are geometry-only teacher targets.  They do not
change trajectory recall, branch matching, RPNet, or the graph-growth state
machine.
"""

from __future__ import annotations

from typing import Dict

import torch


def _cross_2d(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]


def _point_segment_distance(
    point: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    segment = end - start
    denominator = (segment * segment).sum(dim=-1).clamp_min(epsilon)
    projection = ((point - start) * segment).sum(dim=-1) / denominator
    projection = projection.clamp(0.0, 1.0)
    closest = start + projection.unsqueeze(-1) * segment
    return torch.linalg.vector_norm(point - closest, dim=-1)


def _segment_distance_to_branch(
    segment_start: torch.Tensor,
    segment_end: torch.Tensor,
    branch_end: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    """Return distance from fragment segments to origin-to-endpoint segments.

    Shapes are broadcastable and end in ``[..., 2]``.  Explicit crossing
    detection is needed because endpoint-to-segment distances alone are
    positive for two segments that cross in their interiors.
    """

    origin = torch.zeros_like(branch_end)
    distances = torch.stack(
        (
            _point_segment_distance(
                segment_start, origin, branch_end, epsilon=epsilon),
            _point_segment_distance(
                segment_end, origin, branch_end, epsilon=epsilon),
            _point_segment_distance(
                origin, segment_start, segment_end, epsilon=epsilon),
            _point_segment_distance(
                branch_end, segment_start, segment_end, epsilon=epsilon),
        ),
        dim=-1,
    )
    minimum = distances.amin(dim=-1)

    fragment_vector = segment_end - segment_start
    branch_vector = branch_end
    denominator = _cross_2d(fragment_vector, branch_vector)
    origin_delta = -segment_start
    non_parallel = denominator.abs() > epsilon
    safe_denominator = torch.where(
        non_parallel, denominator, torch.ones_like(denominator))
    fragment_parameter = (
        _cross_2d(origin_delta, branch_vector) / safe_denominator)
    branch_parameter = (
        _cross_2d(origin_delta, fragment_vector) / safe_denominator)
    intersects = (
        non_parallel
        & (fragment_parameter >= 0.0)
        & (fragment_parameter <= 1.0)
        & (branch_parameter >= 0.0)
        & (branch_parameter <= 1.0)
    )
    return torch.where(intersects, torch.zeros_like(minimum), minimum)


def build_trajectory_support_targets(
    trajectory_batch: Dict[str, torch.Tensor],
    branch_targets: Dict[str, torch.Tensor],
    *,
    window_size: float,
    step_length: float,
    distance_sigma_pixels: float,
    axis_gamma: float,
    positive_threshold: float,
    epsilon: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """Build soft support labels with shape ``[B, M, N]``.

    A fragment is scored against the local segment from the current node to
    one immediate GT endpoint.  The score is the product of Gaussian
    distance, undirected local-axis agreement, and projected coverage.
    ``support_valid`` is true only when a real GT branch has at least one
    fragment whose soft target reaches ``positive_threshold``.
    """

    if window_size <= 0.0 or step_length <= 0.0:
        raise ValueError("window_size and step_length must be positive")
    if distance_sigma_pixels <= 0.0:
        raise ValueError("distance_sigma_pixels must be positive")
    if axis_gamma < 0.0:
        raise ValueError("axis_gamma must be non-negative")
    if not 0.0 <= positive_threshold <= 1.0:
        raise ValueError("positive_threshold must be in [0, 1]")

    required_trajectory = (
        "traj_xy_norm",
        "point_mask",
        "fragment_mask",
        "segment_only",
    )
    required_branches = ("branch_offsets_norm", "branch_mask")
    for key in required_trajectory:
        if key not in trajectory_batch:
            raise KeyError("trajectory batch is missing {!r}".format(key))
    for key in required_branches:
        if key not in branch_targets:
            raise KeyError("branch targets are missing {!r}".format(key))

    xy_norm = trajectory_batch["traj_xy_norm"]
    point_mask = trajectory_batch["point_mask"].to(dtype=torch.bool)
    fragment_mask = trajectory_batch["fragment_mask"].to(dtype=torch.bool)
    segment_only = trajectory_batch["segment_only"].to(dtype=torch.bool)
    branch_offsets_norm = branch_targets["branch_offsets_norm"].to(
        device=xy_norm.device, dtype=xy_norm.dtype)
    branch_mask = branch_targets["branch_mask"].to(
        device=xy_norm.device, dtype=torch.bool)

    if xy_norm.ndim != 4 or xy_norm.shape[-1] != 2:
        raise ValueError("traj_xy_norm must have shape [B, N, T, 2]")
    batch_size, fragment_count, point_count, _ = xy_norm.shape
    if tuple(point_mask.shape) != (
            batch_size, fragment_count, point_count):
        raise ValueError("point_mask must have shape [B, N, T]")
    if tuple(fragment_mask.shape) != (batch_size, fragment_count):
        raise ValueError("fragment_mask must have shape [B, N]")
    if tuple(segment_only.shape) != (batch_size, fragment_count):
        raise ValueError("segment_only must have shape [B, N]")
    if (
            branch_offsets_norm.ndim != 3
            or branch_offsets_norm.shape[0] != batch_size
            or branch_offsets_norm.shape[-1] != 2):
        raise ValueError(
            "branch_offsets_norm must have shape [B, M, 2]")
    branch_count = branch_offsets_norm.shape[1]
    if tuple(branch_mask.shape) != (batch_size, branch_count):
        raise ValueError("branch_mask must have shape [B, M]")

    output_shape = (batch_size, branch_count, fragment_count)
    if point_count < 2 or fragment_count == 0 or branch_count == 0:
        zeros = xy_norm.new_zeros(output_shape)
        valid = torch.zeros(
            (batch_size, branch_count),
            dtype=torch.bool,
            device=xy_norm.device,
        )
        return {
            "support_targets": zeros,
            "support_positive_mask": torch.zeros_like(
                zeros, dtype=torch.bool),
            "support_valid": valid,
            "distance_score": zeros.clone(),
            "axis_score": zeros.clone(),
            "coverage_score": zeros.clone(),
            "minimum_distance_pixels": torch.full_like(
                zeros, float("inf")),
            "segment_only_positive_mask": torch.zeros_like(
                zeros, dtype=torch.bool),
        }

    half_window = float(window_size) / 2.0
    points = xy_norm * half_window
    segment_start = points[:, :, :-1, :]
    segment_end = points[:, :, 1:, :]
    segment_vector = segment_end - segment_start
    segment_length = torch.linalg.vector_norm(
        segment_vector, dim=-1)
    valid_segments = (
        point_mask[:, :, :-1]
        & point_mask[:, :, 1:]
        & (segment_length > epsilon)
        & fragment_mask.unsqueeze(-1)
    )
    segment_direction = segment_vector / segment_length.clamp_min(
        epsilon).unsqueeze(-1)

    branch_end = branch_offsets_norm * half_window
    branch_length = torch.linalg.vector_norm(branch_end, dim=-1)
    valid_branches = branch_mask & (branch_length > epsilon)
    branch_direction = branch_end / branch_length.clamp_min(
        epsilon).unsqueeze(-1)

    # [B, M, N, S, 2]
    expanded_start = segment_start[:, None, :, :, :]
    expanded_end = segment_end[:, None, :, :, :]
    expanded_branch_end = branch_end[:, :, None, None, :]
    per_segment_distance = _segment_distance_to_branch(
        expanded_start,
        expanded_end,
        expanded_branch_end,
        epsilon=epsilon,
    )
    expanded_valid_segments = valid_segments[:, None, :, :]
    infinity = torch.full_like(per_segment_distance, float("inf"))
    masked_distance = torch.where(
        expanded_valid_segments, per_segment_distance, infinity)
    minimum_distance, closest_segment_index = masked_distance.min(dim=-1)
    has_axis = valid_segments.any(dim=-1)

    expanded_segment_direction = segment_direction[:, None, :, :, :]
    expanded_branch_direction = branch_direction[:, :, None, None, :]
    per_segment_axis = (
        expanded_segment_direction * expanded_branch_direction
    ).sum(dim=-1).abs().clamp(0.0, 1.0)
    gathered_axis = torch.gather(
        per_segment_axis,
        dim=-1,
        index=closest_segment_index.unsqueeze(-1),
    ).squeeze(-1)
    axis_score = gathered_axis.pow(float(axis_gamma))
    axis_score = torch.where(
        has_axis[:, None, :], axis_score, torch.zeros_like(axis_score))

    # Coverage is the fragment's span along the finite immediate branch,
    # clipped to [current node, immediate endpoint].
    projections = torch.einsum(
        "bntd,bmd->bmnt", points, branch_direction)
    point_valid = point_mask[:, None, :, :] & fragment_mask[
        :, None, :, None]
    projection_low = torch.where(
        point_valid,
        projections,
        torch.full_like(projections, float("inf")),
    ).amin(dim=-1)
    projection_high = torch.where(
        point_valid,
        projections,
        torch.full_like(projections, float("-inf")),
    ).amax(dim=-1)
    zero = torch.zeros_like(projection_low)
    clipped_low = torch.maximum(projection_low, zero)
    clipped_high = torch.minimum(
        projection_high, branch_length.unsqueeze(-1))
    projected_coverage = (
        clipped_high - clipped_low).clamp_min(0.0)
    coverage_score = (
        projected_coverage / float(step_length)).clamp(0.0, 1.0)

    distance_score = torch.exp(
        -minimum_distance.square()
        / (2.0 * float(distance_sigma_pixels) ** 2)
    )
    support = distance_score * axis_score * coverage_score
    valid_pair = (
        valid_branches[:, :, None]
        & fragment_mask[:, None, :]
        & has_axis[:, None, :]
        & torch.isfinite(minimum_distance)
    )
    support = torch.where(
        valid_pair, support, torch.zeros_like(support))
    positive = valid_pair & (support >= float(positive_threshold))
    support_valid = valid_branches & positive.any(dim=-1)
    segment_only_positive = (
        positive & segment_only[:, None, :])

    return {
        "support_targets": support,
        "support_positive_mask": positive,
        "support_valid": support_valid,
        "distance_score": torch.where(
            valid_pair, distance_score, torch.zeros_like(distance_score)),
        "axis_score": torch.where(
            valid_pair, axis_score, torch.zeros_like(axis_score)),
        "coverage_score": torch.where(
            valid_pair, coverage_score, torch.zeros_like(coverage_score)),
        "minimum_distance_pixels": minimum_distance,
        "segment_only_positive_mask": segment_only_positive,
    }
