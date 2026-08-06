"""Deterministic Stage 3E-3 trajectory robustness transformations."""

from __future__ import annotations

import hashlib
import math
from typing import Mapping, Sequence

import numpy as np
import torch


def _fragment_key(
    sample_id: int,
    track_index: int,
    start_point_index: int,
    end_point_index: int,
) -> int:
    """Return a process-independent 64-bit key for one fragment identity."""

    payload = "{}:{}:{}:{}".format(
        int(sample_id),
        int(track_index),
        int(start_point_index),
        int(end_point_index),
    ).encode("ascii")
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(),
        byteorder="little",
        signed=False,
    )


def deterministic_fragment_thinning(
    *,
    fragment_mask: torch.Tensor,
    sample_ids: torch.Tensor,
    track_indices: torch.Tensor,
    start_point_indices: torch.Tensor,
    end_point_indices: torch.Tensor,
    retain_ratio: float,
) -> torch.Tensor:
    """Thin valid fragments using only stable sample/fragment identities.

    Selection is independent of batch order and never considers padding. An
    originally non-empty sample keeps ``ceil(valid_count * retain_ratio)``
    fragments, with a minimum of one.
    """

    ratio = float(retain_ratio)
    if not 0.0 < ratio <= 1.0:
        raise ValueError("retain_ratio must be in (0, 1]")
    mask = fragment_mask.to(dtype=torch.bool)
    if mask.ndim != 2:
        raise ValueError("fragment_mask must have shape [B, N]")
    batch_size, fragment_count = mask.shape
    if tuple(sample_ids.shape) != (batch_size,):
        raise ValueError("sample_ids must have shape [B]")
    identities = (
        track_indices,
        start_point_indices,
        end_point_indices,
    )
    if any(tuple(value.shape) != (batch_size, fragment_count)
           for value in identities):
        raise ValueError("fragment identity tensors must have shape [B, N]")

    cpu_mask = mask.detach().cpu().numpy()
    cpu_sample_ids = sample_ids.detach().cpu().numpy()
    cpu_identities = [value.detach().cpu().numpy() for value in identities]
    selected = np.zeros_like(cpu_mask, dtype=np.bool_)
    for row in range(batch_size):
        valid = np.flatnonzero(cpu_mask[row])
        if not valid.size:
            continue
        keep_count = max(1, int(math.ceil(valid.size * ratio)))
        ranked = sorted(
            valid.tolist(),
            key=lambda column: (
                _fragment_key(
                    int(cpu_sample_ids[row]),
                    int(cpu_identities[0][row, column]),
                    int(cpu_identities[1][row, column]),
                    int(cpu_identities[2][row, column]),
                ),
                int(cpu_identities[0][row, column]),
                int(cpu_identities[1][row, column]),
                int(cpu_identities[2][row, column]),
                int(column),
            ),
        )
        selected[row, ranked[:keep_count]] = True
    return torch.from_numpy(selected).to(device=fragment_mask.device)


def global_wrong_sample_donor_indices(
    sample_ids: torch.Tensor,
    *,
    cyclic_shift: int = 1,
) -> torch.Tensor:
    """Map every row to a donor using globally sorted unique sample IDs."""

    if sample_ids.ndim != 1:
        raise ValueError("sample_ids must have shape [S]")
    count = int(sample_ids.numel())
    if count < 2:
        raise ValueError("wrong-sample evaluation needs at least two samples")
    shift = int(cyclic_shift) % count
    if shift == 0:
        raise ValueError("cyclic_shift must not map samples to themselves")
    values = sample_ids.detach().cpu().numpy().astype(np.int64, copy=False)
    if np.unique(values).size != count:
        raise ValueError("sample_ids must be unique for global donor mapping")
    sorted_rows = np.argsort(values, kind="stable")
    donor_rows = np.empty(count, dtype=np.int64)
    for sorted_position, row in enumerate(sorted_rows):
        donor_rows[row] = sorted_rows[(sorted_position + shift) % count]
    if np.any(donor_rows == np.arange(count)):
        raise RuntimeError("wrong-sample mapping contains a self donor")
    return torch.from_numpy(donor_rows).to(device=sample_ids.device)


def replace_trajectory_with_global_donors(
    tensors: Mapping[str, torch.Tensor],
    *,
    trajectory_keys: Sequence[str],
    cyclic_shift: int = 1,
) -> dict:
    """Return a shallow cache copy with trajectory fields globally remapped."""

    if "sample_ids" not in tensors:
        raise KeyError("tensors must contain sample_ids")
    donor_indices = global_wrong_sample_donor_indices(
        tensors["sample_ids"], cyclic_shift=cyclic_shift)
    result = dict(tensors)
    for key in trajectory_keys:
        if key not in tensors:
            raise KeyError("trajectory field is missing: {}".format(key))
        result[key] = tensors[key].index_select(0, donor_indices)
    result["trajectory_source_sample_ids"] = tensors[
        "sample_ids"].index_select(0, donor_indices)
    return result
