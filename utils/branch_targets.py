"""Immediate GT road-branch supervision derived from VecRoad targets."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class ImmediateBranchTargets:
    """Variable-length branch targets for one current graph node."""

    branch_offsets_rel: torch.Tensor
    branch_offsets_norm: torch.Tensor
    branch_directions: torch.Tensor
    branch_mask: torch.Tensor
    branch_count: int


def _xy_from_point_like(value, graph=None) -> Tuple[float, float]:
    if value is None:
        raise ValueError("point value must not be None")

    point_member = getattr(value, "point", None)
    if callable(point_member):
        if graph is None:
            raise ValueError(
                "graph is required to resolve target positions")
        value = point_member(graph)
    elif point_member is not None:
        value = point_member

    if hasattr(value, "x") and hasattr(value, "y"):
        return float(value.x), float(value.y)

    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2,):
        raise TypeError(
            "point must expose x/y or be a length-two coordinate")
    return float(array[0]), float(array[1])


def _immediate_positions(target_poses) -> Iterable:
    if target_poses is None:
        return ()
    slots = getattr(target_poses, "target_poses", target_poses)
    if len(slots) == 0:
        return ()
    # VecRoad semantics: only slot zero describes branches that leave the
    # current node.  Later slots are recursive future points on a branch.
    return slots[0]


def _deduplicate_endpoints(
        endpoints: Sequence[Tuple[float, float]],
        merge_distance: float,
) -> List[Tuple[float, float]]:
    ordered = sorted(
        enumerate(endpoints),
        key=lambda item: (item[1][0], item[1][1], item[0]),
    )
    selected: List[Tuple[float, float]] = []
    for _, endpoint in ordered:
        candidate = np.asarray(endpoint, dtype=np.float64)
        if any(
                float(np.linalg.norm(
                    candidate - np.asarray(existing, dtype=np.float64)))
                <= merge_distance
                for existing in selected):
            continue
        selected.append(endpoint)
    return selected


def build_immediate_branch_targets(
        target_poses,
        current_vertex,
        graph,
        window_size: float = 256.0,
        merge_distance: float = 1e-3,
) -> ImmediateBranchTargets:
    """Build the immediate branch set from ``target_poses[0]`` only.

    This deliberately does not map ``NUM_TARGETS`` to branch slots.
    ``target_poses[1:]`` remain VecRoad recursive future supervision and are
    ignored by this interface.
    """

    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if merge_distance < 0:
        raise ValueError("merge_distance must be non-negative")

    current_xy = np.asarray(
        _xy_from_point_like(current_vertex, graph=None),
        dtype=np.float64,
    )
    endpoints = []
    for target_position in _immediate_positions(target_poses):
        endpoint = _xy_from_point_like(target_position, graph=graph)
        if np.isfinite(endpoint).all():
            endpoints.append(endpoint)
    endpoints = _deduplicate_endpoints(endpoints, merge_distance)

    offsets = []
    directions = []
    for endpoint in endpoints:
        offset = np.asarray(endpoint, dtype=np.float64) - current_xy
        distance = float(np.linalg.norm(offset))
        # A target coincident with the current node has no branch direction
        # and is therefore not usable immediate-branch supervision.
        if not np.isfinite(distance) or distance <= 0.0:
            continue
        offsets.append(offset.astype(np.float32))
        directions.append((offset / distance).astype(np.float32))

    branch_count = len(offsets)
    if branch_count == 0:
        offsets_rel = np.zeros((0, 2), dtype=np.float32)
        branch_dirs = np.zeros((0, 2), dtype=np.float32)
    else:
        offsets_rel = np.stack(offsets, axis=0)
        branch_dirs = np.stack(directions, axis=0)
    offsets_norm = offsets_rel / np.float32(window_size / 2.0)
    branch_mask = np.ones(branch_count, dtype=np.bool_)

    return ImmediateBranchTargets(
        branch_offsets_rel=torch.from_numpy(offsets_rel),
        branch_offsets_norm=torch.from_numpy(offsets_norm),
        branch_directions=torch.from_numpy(branch_dirs),
        branch_mask=torch.from_numpy(branch_mask),
        branch_count=branch_count,
    )


def build_branch_target_batch(
        target_sets: Sequence[ImmediateBranchTargets],
) -> Dict[str, torch.Tensor]:
    """Pad variable immediate branch sets without inventing fake branches."""

    batch_size = len(target_sets)
    max_branches = max(
        (targets.branch_count for targets in target_sets), default=0)

    offsets_norm = torch.zeros(
        (batch_size, max_branches, 2), dtype=torch.float32)
    directions = torch.zeros(
        (batch_size, max_branches, 2), dtype=torch.float32)
    mask = torch.zeros(
        (batch_size, max_branches), dtype=torch.bool)
    counts = torch.zeros(batch_size, dtype=torch.int64)

    for batch_index, targets in enumerate(target_sets):
        count = int(targets.branch_count)
        if targets.branch_offsets_norm.shape != (count, 2):
            raise ValueError(
                "branch_offsets_norm shape does not match branch_count")
        if targets.branch_directions.shape != (count, 2):
            raise ValueError(
                "branch_directions shape does not match branch_count")
        if targets.branch_mask.shape != (count,):
            raise ValueError(
                "branch_mask shape does not match branch_count")
        counts[batch_index] = count
        if count == 0:
            continue
        offsets_norm[batch_index, :count] = (
            targets.branch_offsets_norm.to(dtype=torch.float32))
        directions[batch_index, :count] = (
            targets.branch_directions.to(dtype=torch.float32))
        mask[batch_index, :count] = targets.branch_mask.to(
            dtype=torch.bool)

    return {
        "branch_offsets_norm": offsets_norm,
        "branch_directions": directions,
        "branch_mask": mask,
        "branch_count": counts,
    }
