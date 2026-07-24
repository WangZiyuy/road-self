"""Portable fixed-shape Stage 3C teacher-forced branch dataset."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


STAGE3C_DATASET_SCHEMA_VERSION = "1.0"

REQUIRED_STAGE3C_ARRAYS = (
    "aerial_image",
    "walked_path",
    "incoming_dir",
    "incoming_valid",
    "explored_edge_dirs",
    "explored_edge_mask",
    "explored_is_incoming",
    "is_key_point",
    "branch_offsets_norm",
    "branch_directions",
    "branch_mask",
    "branch_count",
    "traj_xy_norm",
    "traj_time_delta",
    "point_mask",
    "fragment_mask",
    "point_inside_mask",
    "segment_only",
    "track_indices",
    "start_point_indices",
    "end_point_indices",
    "total_fragment_count",
    "kept_fragment_count",
    "truncated_fragment_count",
    "trajectory_point_truncated_count",
    "center_xy",
    "subtile_index",
    "vertex_id",
)


def _validate_array_mapping(
    arrays: Mapping[str, np.ndarray],
) -> int:
    missing = [
        key for key in REQUIRED_STAGE3C_ARRAYS if key not in arrays
    ]
    if missing:
        raise KeyError(
            "Stage 3C shard is missing: {}".format(", ".join(missing)))

    sample_count = None
    for key in REQUIRED_STAGE3C_ARRAYS:
        value = arrays[key]
        if not isinstance(value, np.ndarray):
            raise TypeError(
                "Stage 3C array {!r} must be a numpy array".format(key))
        if value.dtype.hasobject:
            raise TypeError(
                "Stage 3C array {!r} must not use object dtype".format(key))
        if value.ndim == 0:
            raise ValueError(
                "Stage 3C array {!r} must have a sample dimension".format(
                    key))
        if sample_count is None:
            sample_count = int(value.shape[0])
        elif value.shape[0] != sample_count:
            raise ValueError(
                "Stage 3C arrays have inconsistent sample counts")
    return int(sample_count or 0)


def array_schema_from_mapping(
    arrays: Mapping[str, np.ndarray],
) -> Dict[str, Dict[str, Any]]:
    _validate_array_mapping(arrays)
    return {
        key: {
            "shape_per_sample": list(arrays[key].shape[1:]),
            "dtype": str(arrays[key].dtype),
        }
        for key in REQUIRED_STAGE3C_ARRAYS
    }


def write_stage3c_shard(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    *,
    compressed: bool = True,
) -> Dict[str, Any]:
    """Write one pickle-free NPZ shard after strict shape validation."""

    path = Path(path)
    sample_count = _validate_array_mapping(arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = np.savez_compressed if compressed else np.savez
    writer(
        str(path),
        **{key: arrays[key] for key in REQUIRED_STAGE3C_ARRAYS},
    )
    return {
        "file": path.name,
        "sample_count": sample_count,
        "size_bytes": int(path.stat().st_size),
    }


def write_stage3c_manifest(
    dataset_dir: Path,
    manifest: Mapping[str, Any],
) -> Path:
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    serializable = dict(manifest)
    serializable["schema_version"] = STAGE3C_DATASET_SCHEMA_VERSION
    manifest_path = dataset_dir / "meta.json"
    with manifest_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            serializable,
            output_file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        output_file.write("\n")
    return manifest_path


def _load_shard(path: Path) -> Dict[str, np.ndarray]:
    with np.load(str(path), allow_pickle=False) as archive:
        arrays = {
            key: np.asarray(archive[key])
            for key in archive.files
        }
    _validate_array_mapping(arrays)
    return arrays


def validate_stage3c_dataset(dataset_dir: Path) -> Dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "meta.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Stage 3C manifest not found: {}".format(manifest_path))
    with manifest_path.open("r", encoding="utf-8") as input_file:
        manifest = json.load(input_file)
    if manifest.get("schema_version") != STAGE3C_DATASET_SCHEMA_VERSION:
        raise ValueError(
            "unsupported Stage 3C schema version: {!r}".format(
                manifest.get("schema_version")))
    if "splits" not in manifest or not isinstance(
            manifest["splits"], dict):
        raise ValueError("Stage 3C manifest must define splits")
    expected_schema = manifest.get("array_schema")
    if not isinstance(expected_schema, dict):
        raise ValueError("Stage 3C manifest must define array_schema")

    split_reports = {}
    for split_name, split_metadata in manifest["splits"].items():
        total = 0
        shard_reports = []
        for shard_metadata in split_metadata.get("shards", []):
            shard_path = dataset_dir / split_name / shard_metadata["file"]
            if not shard_path.is_file():
                raise FileNotFoundError(
                    "Stage 3C shard not found: {}".format(shard_path))
            arrays = _load_shard(shard_path)
            sample_count = _validate_array_mapping(arrays)
            if sample_count != int(shard_metadata["sample_count"]):
                raise ValueError(
                    "manifest count differs for {}".format(shard_path))
            schema = array_schema_from_mapping(arrays)
            if schema != expected_schema:
                raise ValueError(
                    "array schema differs for {}".format(shard_path))
            total += sample_count
            shard_reports.append({
                "file": shard_metadata["file"],
                "sample_count": sample_count,
                "size_bytes": int(shard_path.stat().st_size),
            })
        if total != int(split_metadata.get("sample_count", -1)):
            raise ValueError(
                "split sample count differs for {!r}".format(split_name))
        split_reports[split_name] = {
            "sample_count": total,
            "shards": shard_reports,
        }
    return {
        "schema_version": STAGE3C_DATASET_SCHEMA_VERSION,
        "splits": split_reports,
    }


class Stage3CBranchDataset(Dataset):
    """Read fixed-shape NPZ shards without pickle or Python objects."""

    def __init__(
        self,
        dataset_dir: Path,
        split: str,
        *,
        preload: bool = False,
        cache_shards: int = 2,
    ) -> None:
        super().__init__()
        if cache_shards <= 0:
            raise ValueError("cache_shards must be positive")
        self.dataset_dir = Path(dataset_dir)
        manifest_path = self.dataset_dir / "meta.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                "Stage 3C manifest not found: {}".format(manifest_path))
        with manifest_path.open("r", encoding="utf-8") as input_file:
            self.manifest = json.load(input_file)
        if (
                self.manifest.get("schema_version")
                != STAGE3C_DATASET_SCHEMA_VERSION):
            raise ValueError(
                "unsupported Stage 3C dataset schema: {!r}".format(
                    self.manifest.get("schema_version")))
        if split not in self.manifest.get("splits", {}):
            raise KeyError(
                "unknown Stage 3C split {!r}".format(split))
        self.split = split
        split_metadata = self.manifest["splits"][split]
        self.shards = []
        self.index = []
        for shard_index, shard_metadata in enumerate(
                split_metadata.get("shards", [])):
            path = self.dataset_dir / split / shard_metadata["file"]
            sample_count = int(shard_metadata["sample_count"])
            self.shards.append(path)
            self.index.extend(
                (shard_index, local_index)
                for local_index in range(sample_count)
            )
        if len(self.index) != int(split_metadata["sample_count"]):
            raise ValueError(
                "split index count differs from manifest")
        self._cache_limit = (
            len(self.shards) if preload else int(cache_shards))
        self._cache = OrderedDict()
        if preload:
            for shard_index in range(len(self.shards)):
                self._get_shard(shard_index)

    def __len__(self) -> int:
        return len(self.index)

    def _get_shard(self, shard_index: int) -> Dict[str, np.ndarray]:
        if shard_index in self._cache:
            arrays = self._cache.pop(shard_index)
            self._cache[shard_index] = arrays
            return arrays
        arrays = _load_shard(self.shards[shard_index])
        expected_schema = self.manifest["array_schema"]
        if array_schema_from_mapping(arrays) != expected_schema:
            raise ValueError(
                "shard schema differs from manifest: {}".format(
                    self.shards[shard_index]))
        self._cache[shard_index] = arrays
        while len(self._cache) > self._cache_limit:
            self._cache.popitem(last=False)
        return arrays

    @staticmethod
    def _tensor(
        arrays: Mapping[str, np.ndarray],
        key: str,
        local_index: int,
    ) -> torch.Tensor:
        return torch.from_numpy(
            np.array(arrays[key][local_index], copy=True))

    def __getitem__(self, index: int) -> Dict[str, Any]:
        shard_index, local_index = self.index[index]
        arrays = self._get_shard(shard_index)
        tensor = lambda key: self._tensor(
            arrays, key, local_index)
        return {
            "aerial_image": tensor("aerial_image").to(
                dtype=torch.float32).div_(255.0),
            "walked_path": tensor("walked_path").to(
                dtype=torch.float32),
            "graph_state": {
                "incoming_dir": tensor("incoming_dir"),
                "incoming_valid": tensor("incoming_valid"),
                "explored_edge_dirs": tensor("explored_edge_dirs"),
                "explored_edge_mask": tensor("explored_edge_mask"),
                "explored_is_incoming": tensor(
                    "explored_is_incoming"),
                "is_key_point": tensor("is_key_point"),
            },
            "branch_targets": {
                "branch_offsets_norm": tensor("branch_offsets_norm"),
                "branch_directions": tensor("branch_directions"),
                "branch_mask": tensor("branch_mask"),
                "branch_count": tensor("branch_count"),
            },
            "trajectory_batch": {
                "traj_xy_norm": tensor("traj_xy_norm"),
                "traj_time_delta": tensor("traj_time_delta"),
                "point_mask": tensor("point_mask"),
                "fragment_mask": tensor("fragment_mask"),
                "point_inside_mask": tensor("point_inside_mask"),
                "segment_only": tensor("segment_only"),
                "track_indices": tensor("track_indices"),
                "start_point_indices": tensor(
                    "start_point_indices"),
                "end_point_indices": tensor("end_point_indices"),
                "total_fragment_count": tensor(
                    "total_fragment_count"),
                "kept_fragment_count": tensor(
                    "kept_fragment_count"),
                "truncated_fragment_count": tensor(
                    "truncated_fragment_count"),
            },
            "metadata": {
                "center_xy": tensor("center_xy"),
                "subtile_index": tensor("subtile_index"),
                "vertex_id": tensor("vertex_id"),
                "trajectory_point_truncated_count": tensor(
                    "trajectory_point_truncated_count"),
                "dataset_index": torch.tensor(index, dtype=torch.int64),
            },
        }
