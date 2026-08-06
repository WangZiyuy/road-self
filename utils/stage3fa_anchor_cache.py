"""Pickle-free fixed-shape cache for Stage 3F-A anchor validation."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, Subset


SCHEMA_VERSION = "stage3fa-anchor-cache-v1"
REQUIRED_ARRAYS = (
    "sample_id", "dataset_index", "subtile_index", "vertex_id",
    "center_xy", "is_key_point", "branch_count", "category_id",
    "supervision_end_index", "next_node_xy", "next_node_mask",
    "anchor_features", "anchor_lowrs_features", "anchor_target",
    "anchor_lowrs_target", "original_anchor_logits",
    "original_anchor_lowrs_logits", "trajectory_evidence",
    "trajectory_evidence_retain25", "trajectory_available",
    "evidence_attention", "evidence_attention_retain25",
)


def validate_stage3fa_arrays(arrays: Mapping[str, np.ndarray]) -> int:
    missing = [name for name in REQUIRED_ARRAYS if name not in arrays]
    if missing:
        raise KeyError("Stage 3F-A cache is missing: {}".format(
            ", ".join(missing)))
    count = None
    for name in REQUIRED_ARRAYS:
        value = arrays[name]
        if not isinstance(value, np.ndarray) or value.dtype.hasobject:
            raise TypeError("cache field {!r} must be a non-object ndarray".format(
                name))
        if value.ndim == 0:
            raise ValueError("cache fields require a sample dimension")
        count = value.shape[0] if count is None else count
        if value.shape[0] != count:
            raise ValueError("cache fields have inconsistent sample counts")
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError("cache field {!r} contains NaN/Inf".format(name))
    return int(count or 0)


def write_stage3fa_shard(path: Path, arrays: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    count = validate_stage3fa_arrays(arrays)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), **{name: arrays[name] for name in REQUIRED_ARRAYS})
    return {
        "file": path.name,
        "sample_count": count,
        "size_bytes": int(path.stat().st_size),
    }


def array_schema(arrays: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    validate_stage3fa_arrays(arrays)
    return {
        name: {
            "shape_per_sample": list(arrays[name].shape[1:]),
            "dtype": str(arrays[name].dtype),
        }
        for name in REQUIRED_ARRAYS
    }


def write_stage3fa_manifest(cache_dir: Path, manifest: Mapping[str, Any]) -> Path:
    path = Path(cache_dir) / "meta.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(manifest)
    payload["schema_version"] = SCHEMA_VERSION
    with path.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")
    return path


class Stage3FAAnchorDataset(Dataset):
    """Lazy fixed-shape reader for Stage 3F-A NPZ shards."""

    def __init__(self, cache_dir: Path, split: str,
                 cache_shards: Optional[int] = 1) -> None:
        self.cache_dir = Path(cache_dir)
        with (self.cache_dir / "meta.json").open(
                "r", encoding="utf-8") as input_file:
            self.manifest = json.load(input_file)
        if self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported Stage 3F-A cache schema")
        if split not in self.manifest["splits"]:
            raise KeyError("unknown Stage 3F-A split {!r}".format(split))
        self.split = split
        self.shards = []
        self.index = []
        for shard_index, metadata in enumerate(
                self.manifest["splits"][split]["shards"]):
            path = self.cache_dir / split / metadata["file"]
            self.shards.append(path)
            self.index.extend(
                (shard_index, local_index)
                for local_index in range(int(metadata["sample_count"])))
        self._limit = (
            len(self.shards) if cache_shards is None
            else max(1, int(cache_shards)))
        self._cache: OrderedDict[int, Dict[str, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.index)

    def _load(self, shard_index: int) -> Dict[str, np.ndarray]:
        if shard_index in self._cache:
            arrays = self._cache.pop(shard_index)
            self._cache[shard_index] = arrays
            return arrays
        with np.load(str(self.shards[shard_index]), allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        validate_stage3fa_arrays(arrays)
        self._cache[shard_index] = arrays
        while len(self._cache) > self._limit:
            self._cache.popitem(last=False)
        return arrays

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        shard_index, local_index = self.index[index]
        arrays = self._load(shard_index)
        return {
            name: torch.from_numpy(np.array(arrays[name][local_index], copy=True))
            for name in REQUIRED_ARRAYS
        }


def stage3fa_collate(batch):
    return {
        key: torch.stack([sample[key] for sample in batch], dim=0)
        for key in REQUIRED_ARRAYS
    }


class ShardLocalShuffleSampler(Sampler[int]):
    """Shuffle deterministically while keeping reads local to one shard.

    A Stage 3F-A sample contains large spatial feature maps.  A fully random
    sample permutation combined with the lazy one-shard cache reloads the same
    NPZ files many times per epoch.  Randomizing shard order and then sample
    order within each shard preserves stochastic iteration while ensuring each
    shard is normally opened once per epoch.
    """

    def __init__(self, dataset: Dataset, seed: int) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0
        if isinstance(dataset, Stage3FAAnchorDataset):
            source = dataset
            source_indices = range(len(dataset))
        elif (isinstance(dataset, Subset)
              and isinstance(dataset.dataset, Stage3FAAnchorDataset)):
            source = dataset.dataset
            source_indices = dataset.indices
        else:
            raise TypeError(
                "ShardLocalShuffleSampler requires a Stage3FAAnchorDataset view")
        by_shard: Dict[int, list[int]] = {}
        for dataset_index, source_index in enumerate(source_indices):
            shard_index, _ = source.index[int(source_index)]
            by_shard.setdefault(int(shard_index), []).append(dataset_index)
        self.by_shard = by_shard

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        shard_ids = list(self.by_shard)
        shard_order = torch.randperm(
            len(shard_ids), generator=generator).tolist()
        ordered = []
        for position in shard_order:
            indices = self.by_shard[shard_ids[position]]
            local_order = torch.randperm(
                len(indices), generator=generator).tolist()
            ordered.extend(indices[index] for index in local_order)
        return iter(ordered)

    def __len__(self) -> int:
        return len(self.dataset)
