"""Build pickle-free teacher-forced shards for Stage 3C training."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from easydict import EasyDict


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.branch_targets import build_immediate_branch_targets
from utils.graph_state import build_graph_state
from utils.model_utils import Path as VecRoadPath
from utils.stage3c_branch_dataset import (
    array_schema_from_mapping,
    validate_stage3c_dataset,
    write_stage3c_manifest,
    write_stage3c_shard,
)
from utils.structured_trajectory_store import (
    open_structured_trajectory_store,
)
from utils.tileloader import Tiles
from utils.trajectory_batch import build_trajectory_batch
from utils.trajectory_compression import compress_trajectory_fragments
from utils.trajectory_mode import TRAJ_MODE_NONE, resolve_trajectory_mode


def _load_config(config_path: Path) -> EasyDict:
    with config_path.open("r", encoding="utf-8") as config_file:
        cfg = EasyDict(yaml.load(
            config_file, Loader=yaml.UnsafeLoader))
    mode = resolve_trajectory_mode(cfg)
    if mode != TRAJ_MODE_NONE:
        raise ValueError(
            "Stage 3C dataset preparation requires TRAJ.MODE=none; "
            "resolved {!r}".format(mode))
    if "STAGE3C" not in cfg or "DATASET" not in cfg.STAGE3C:
        raise ValueError(
            "config must define STAGE3C.DATASET")
    return cfg


def _prepare_output_directory(path: Path, overwrite: bool) -> None:
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                "Stage 3C output directory is non-empty; pass "
                "--overwrite to replace it: {}".format(path))
        shutil.rmtree(str(path))
    path.mkdir(parents=True, exist_ok=True)


def _make_paths(cfg: EasyDict):
    tiles = Tiles(
        training_regions=cfg.TRAIN.TRAINING_REGIONS,
        parallel_tiles=cfg.TRAIN.PARALLEL_TILES,
        region_path=cfg.DIR.ALL_REGION_PATH,
        graph_dir=cfg.DIR.GRAPH_DIR,
        tile_dir=cfg.DIR.TILE_DIR,
        traj_dir=None,
    )
    subtiles = tiles.prepare_training()
    paths = [
        _new_path(index, subtile)
        for index, subtile in enumerate(subtiles)
    ]
    return paths, subtiles


def _new_path(index: int, subtile: Mapping[str, Any]) -> VecRoadPath:
    return VecRoadPath(
        idx=index,
        training=True,
        gc=subtile["gc"].clone(),
        tile_data=subtile,
        all_trajectories=[],
        all_pixel_trajectories=[],
    )


def _subtile_metadata(index: int, subtile: Mapping[str, Any]):
    rect = subtile["search_rect"]
    return {
        "subtile_index": int(index),
        "region": str(subtile["region"]),
        "bounds_xyxy": [
            float(rect.start.x),
            float(rect.start.y),
            float(rect.end.x),
            float(rect.end.y),
        ],
    }


def _fixed_branch_arrays(
    targets,
    max_branches: int,
) -> Dict[str, np.ndarray]:
    count = int(targets.branch_count)
    if count > max_branches:
        raise ValueError(
            "branch_count {} exceeds configured max_branches {}".format(
                count, max_branches))
    offsets = np.zeros((max_branches, 2), dtype=np.float32)
    directions = np.zeros((max_branches, 2), dtype=np.float32)
    mask = np.zeros(max_branches, dtype=np.bool_)
    if count:
        offsets[:count] = targets.branch_offsets_norm.numpy()
        directions[:count] = targets.branch_directions.numpy()
        mask[:count] = targets.branch_mask.numpy()
    return {
        "branch_offsets_norm": offsets,
        "branch_directions": directions,
        "branch_mask": mask,
        "branch_count": np.asarray(count, dtype=np.int64),
    }


def _ordered_subsample_indices(
    point_count: int,
    max_points: int,
) -> np.ndarray:
    if point_count <= max_points:
        return np.arange(point_count, dtype=np.int64)
    indices = np.linspace(
        0, point_count - 1, num=max_points, dtype=np.float64)
    indices = np.rint(indices).astype(np.int64)
    if len(np.unique(indices)) != max_points:
        raise RuntimeError(
            "ordered trajectory subsampling produced duplicate indices")
    return indices


def _fixed_trajectory_arrays(
    trajectory_batch: Mapping[str, torch.Tensor],
    max_fragments: int,
    max_points: int,
) -> Dict[str, np.ndarray]:
    source_xy = trajectory_batch["traj_xy_norm"][0].numpy()
    source_time = trajectory_batch["traj_time_delta"][0].numpy()
    source_point_mask = trajectory_batch["point_mask"][0].numpy()
    source_inside = trajectory_batch["point_inside_mask"][0].numpy()
    source_fragment_mask = trajectory_batch["fragment_mask"][0].numpy()
    source_segment_only = trajectory_batch["segment_only"][0].numpy()
    source_track_indices = trajectory_batch["track_indices"][0].numpy()
    source_start_indices = trajectory_batch[
        "start_point_indices"][0].numpy()
    source_end_indices = trajectory_batch[
        "end_point_indices"][0].numpy()

    source_fragment_count = int(source_fragment_mask.sum())
    if source_fragment_count > max_fragments:
        raise ValueError(
            "compressed fragment count exceeds configured budget")
    xy = np.zeros(
        (max_fragments, max_points, 2), dtype=np.float32)
    time_delta = np.zeros(
        (max_fragments, max_points), dtype=np.float32)
    point_mask = np.zeros(
        (max_fragments, max_points), dtype=np.bool_)
    inside = np.zeros(
        (max_fragments, max_points), dtype=np.bool_)
    fragment_mask = np.zeros(max_fragments, dtype=np.bool_)
    segment_only = np.zeros(max_fragments, dtype=np.bool_)
    track_indices = np.full(max_fragments, -1, dtype=np.int64)
    start_indices = np.full(max_fragments, -1, dtype=np.int64)
    end_indices = np.full(max_fragments, -1, dtype=np.int64)
    truncated_points = 0

    valid_fragment_indices = np.flatnonzero(source_fragment_mask)
    for output_index, source_index in enumerate(valid_fragment_indices):
        valid_points = np.flatnonzero(source_point_mask[source_index])
        selected = _ordered_subsample_indices(
            len(valid_points), max_points)
        selected_points = valid_points[selected]
        kept_point_count = len(selected_points)
        truncated_points += len(valid_points) - kept_point_count
        xy[output_index, :kept_point_count] = source_xy[
            source_index, selected_points]
        time_delta[output_index, :kept_point_count] = source_time[
            source_index, selected_points]
        point_mask[output_index, :kept_point_count] = True
        inside[output_index, :kept_point_count] = source_inside[
            source_index, selected_points]
        fragment_mask[output_index] = True
        segment_only[output_index] = source_segment_only[source_index]
        track_indices[output_index] = source_track_indices[source_index]
        start_indices[output_index] = source_start_indices[source_index]
        end_indices[output_index] = source_end_indices[source_index]

    return {
        "traj_xy_norm": xy,
        "traj_time_delta": time_delta,
        "point_mask": point_mask,
        "fragment_mask": fragment_mask,
        "point_inside_mask": inside,
        "segment_only": segment_only,
        "track_indices": track_indices,
        "start_point_indices": start_indices,
        "end_point_indices": end_indices,
        "total_fragment_count": np.asarray(
            int(trajectory_batch["total_fragment_count"][0]),
            dtype=np.int64,
        ),
        "kept_fragment_count": np.asarray(
            int(trajectory_batch["kept_fragment_count"][0]),
            dtype=np.int64,
        ),
        "truncated_fragment_count": np.asarray(
            int(trajectory_batch["truncated_fragment_count"][0]),
            dtype=np.int64,
        ),
        "trajectory_point_truncated_count": np.asarray(
            truncated_points, dtype=np.int64),
    }


def _sample_arrays(
    *,
    path,
    state,
    local_input,
    graph_state,
    branch_targets,
    trajectory_batch,
    subtile_index: int,
    max_branches: int,
    max_fragments: int,
    max_points: int,
) -> Dict[str, np.ndarray]:
    aerial = np.asarray(
        local_input["aerial_image_chw"], dtype=np.float32)
    aerial_uint8 = np.rint(
        np.clip(aerial, 0.0, 1.0) * 255.0).astype(np.uint8)
    walked_path = np.asarray(
        local_input["walked_path_small"], dtype=np.float32)
    walked_uint8 = (walked_path > 0.5).astype(np.uint8)
    arrays = {
        "aerial_image": aerial_uint8,
        "walked_path": walked_uint8,
        "incoming_dir": graph_state["incoming_dir"].numpy().astype(
            np.float32),
        "incoming_valid": graph_state["incoming_valid"].numpy().astype(
            np.bool_),
        "explored_edge_dirs": graph_state[
            "explored_edge_dirs"].numpy().astype(np.float32),
        "explored_edge_mask": graph_state[
            "explored_edge_mask"].numpy().astype(np.bool_),
        "explored_is_incoming": graph_state[
            "explored_is_incoming"].numpy().astype(np.bool_),
        "is_key_point": np.asarray(
            bool(state.is_key_point), dtype=np.bool_),
        "center_xy": np.asarray(
            [state.vertex.point.x, state.vertex.point.y],
            dtype=np.float32,
        ),
        "subtile_index": np.asarray(
            subtile_index, dtype=np.int64),
        "vertex_id": np.asarray(
            int(state.vertex.id), dtype=np.int64),
    }
    arrays.update(_fixed_branch_arrays(
        branch_targets, max_branches))
    arrays.update(_fixed_trajectory_arrays(
        trajectory_batch, max_fragments, max_points))
    return arrays


class _ShardAccumulator:
    def __init__(
        self,
        output_dir: Path,
        split: str,
        shard_size: int,
        compressed: bool,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.split = split
        self.shard_size = int(shard_size)
        self.compressed = bool(compressed)
        self.samples: List[Dict[str, np.ndarray]] = []
        self.shards = []
        self.array_schema = None

    def add(self, sample: Dict[str, np.ndarray]) -> None:
        self.samples.append(sample)
        if len(self.samples) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.samples:
            return
        keys = tuple(self.samples[0])
        if any(tuple(sample) != keys for sample in self.samples):
            raise ValueError("Stage 3C samples have inconsistent fields")
        arrays = {
            key: np.stack(
                [sample[key] for sample in self.samples], axis=0)
            for key in keys
        }
        if self.array_schema is None:
            self.array_schema = array_schema_from_mapping(arrays)
        elif array_schema_from_mapping(arrays) != self.array_schema:
            raise ValueError("Stage 3C shard schema changed")
        shard_name = "{}_{:04d}.npz".format(
            self.split, len(self.shards))
        shard_metadata = write_stage3c_shard(
            self.output_dir / self.split / shard_name,
            arrays,
            compressed=self.compressed,
        )
        self.shards.append(shard_metadata)
        self.samples = []


def _advance_teacher_forced_path(
    *,
    path,
    state,
    target_poses,
    road_segmentation,
    cfg: EasyDict,
) -> None:
    if state.vertex.edge_pos is None or len(target_poses) == 0:
        return
    if state.is_key_point:
        immediate_count = len(target_poses.target_poses[0])
        if immediate_count > 0:
            target_poses.target_poses[0] = random.sample(
                target_poses.target_poses[0],
                random.randint(1, immediate_count),
            )
    path.push(
        extension_vertex=state.vertex,
        is_key_point=state.is_key_point,
        follow_mode="follow_target",
        target_poses=target_poses,
        output_points=None,
        RECT_RADIUS=cfg.TRAIN.RECT_RADIUS,
        road_segmentation=road_segmentation,
        NUM_TARGETS=cfg.TRAIN.NUM_TARGETS,
        WINDOW_SIZE=cfg.TRAIN.WINDOW_SIZE,
        STEP_LENGTH=cfg.TRAIN.STEP_LENGTH,
        AVG_CONFIDENCE_THRESHOLD=(
            cfg.TRAIN.AVG_CONFIDENCE_THRESHOLD),
    )


def _collect_split(
    *,
    split: str,
    path_indices: Sequence[int],
    paths: List[VecRoadPath],
    subtiles: Sequence[Mapping[str, Any]],
    store,
    cfg: EasyDict,
    requested_samples: int,
    max_attempts: int,
    accumulator: _ShardAccumulator,
    index_file,
    max_explored_edges: int,
    max_branches: int,
    max_fragments: int,
    max_points: int,
    context_points: int,
    max_time_gap_seconds: Optional[float],
    max_spatial_gap_pixels: Optional[float],
    progress_every: int,
    forbidden_center_keys,
    collected_center_keys,
) -> Dict[str, Any]:
    active = list(path_indices)
    cursor = 0
    attempts = 0
    sample_count = 0
    query_seconds = 0.0
    compression_seconds = 0.0
    full_fragment_counts = []
    kept_fragment_counts = []
    branch_counts = []
    point_truncated_count = 0
    path_reset_count = 0
    spatial_overlap_skip_count = 0

    while (
            sample_count < requested_samples
            and active
            and attempts < max_attempts):
        attempts += 1
        active_position = cursor % len(active)
        path_index = active[active_position]
        path = paths[path_index]
        state = path.pop_state(
            follow_order=False,
            probs=[0.15, 0.8, 0.05],
            WINDOW_SIZE=cfg.TRAIN.WINDOW_SIZE,
        )
        if (
                state is None
                or len(path.graph.vertices) >= cfg.TRAIN.MAX_PATH_LENGTH):
            # Match OSMDataset's teacher-forced training lifecycle: a path
            # is restarted inside the same spatial subtile when exhausted or
            # when MAX_PATH_LENGTH is reached. This never crosses the
            # train/validation subtile boundary.
            paths[path_index] = _new_path(
                path_index, subtiles[path_index])
            path_reset_count += 1
            continue
        cursor += 1

        local_input = path.make_path_input(
            extension_vertex=state.vertex,
            fetch_list=[
                "aerial_image_chw",
                "walked_path_small",
                "road_seg_thick3",
            ],
            traj_filter=False,
            is_key_point=state.is_key_point,
            WINDOW_SIZE=cfg.TRAIN.WINDOW_SIZE,
        )
        road_segmentation = (
            local_input["road_seg_thick3"][0]
            if state.vertex.edge_pos is not None
            else None
        )
        target_poses = path.get_target_poses(
            extension_vertex=state.vertex,
            road_segmentation=road_segmentation,
            STEP_LENGTH=cfg.TRAIN.STEP_LENGTH,
            is_key_point=state.is_key_point,
            NUM_TARGETS=cfg.TRAIN.NUM_TARGETS,
            RECT_RADIUS=cfg.TRAIN.RECT_RADIUS,
            WINDOW_SIZE=cfg.TRAIN.WINDOW_SIZE,
        )
        graph_state = build_graph_state(
            path, state, max_explored_edges=max_explored_edges)
        branch_targets = build_immediate_branch_targets(
            target_poses=target_poses,
            current_vertex=state.vertex,
            graph=path.gc.graph,
            window_size=cfg.TRAIN.WINDOW_SIZE,
        )
        center_xy = (
            float(state.vertex.point.x),
            float(state.vertex.point.y),
        )
        center_key = (
            round(center_xy[0], 6),
            round(center_xy[1], 6),
        )
        if center_key in forbidden_center_keys:
            spatial_overlap_skip_count += 1
            _advance_teacher_forced_path(
                path=path,
                state=state,
                target_poses=target_poses,
                road_segmentation=road_segmentation,
                cfg=cfg,
            )
            continue
        query_start = time.perf_counter()
        fragments = store.query_trajectory_fragments(
            center_xy=center_xy,
            window_size=cfg.TRAIN.WINDOW_SIZE,
            context_points=context_points,
            max_time_gap_seconds=max_time_gap_seconds,
            max_spatial_gap_pixels=max_spatial_gap_pixels,
        )
        query_seconds += time.perf_counter() - query_start
        compression_start = time.perf_counter()
        compressed = compress_trajectory_fragments(
            fragments=fragments,
            center_xy=center_xy,
            window_size=cfg.TRAIN.WINDOW_SIZE,
            max_fragments=max_fragments,
            strategy="bounded_near_diverse",
            prepool_multiplier=8,
            near_fraction=0.5,
        )
        compression_seconds += time.perf_counter() - compression_start
        trajectory_batch = build_trajectory_batch(
            [compressed],
            center_xy=[center_xy],
            window_size=cfg.TRAIN.WINDOW_SIZE,
            max_fragments=None,
        )
        arrays = _sample_arrays(
            path=path,
            state=state,
            local_input=local_input,
            graph_state=graph_state,
            branch_targets=branch_targets,
            trajectory_batch=trajectory_batch,
            subtile_index=path_index,
            max_branches=max_branches,
            max_fragments=max_fragments,
            max_points=max_points,
        )
        accumulator.add(arrays)
        point_truncated_count += int(
            arrays["trajectory_point_truncated_count"])
        full_fragment_counts.append(len(fragments))
        kept_fragment_counts.append(compressed.kept_fragment_count)
        branch_counts.append(branch_targets.branch_count)
        collected_center_keys.add(center_key)
        index_file.write(json.dumps({
            "split": split,
            "sample_index": sample_count,
            "subtile_index": int(path_index),
            "vertex_id": int(state.vertex.id),
            "center_xy": list(center_xy),
            "is_key_point": bool(state.is_key_point),
            "branch_count": int(branch_targets.branch_count),
            "full_fragment_count": len(fragments),
            "kept_fragment_count": int(
                compressed.kept_fragment_count),
        }, sort_keys=True) + "\n")
        sample_count += 1
        if progress_every > 0 and sample_count % progress_every == 0:
            print(
                "{}: {}/{} samples, attempts={}, mean fragments={:.1f}".format(
                    split,
                    sample_count,
                    requested_samples,
                    attempts,
                    float(np.mean(full_fragment_counts)),
                ),
                flush=True,
            )

        _advance_teacher_forced_path(
            path=path,
            state=state,
            target_poses=target_poses,
            road_segmentation=road_segmentation,
            cfg=cfg,
        )

    accumulator.flush()
    if sample_count != requested_samples:
        raise RuntimeError(
            "{} split produced {}/{} samples after {} attempts".format(
                split, sample_count, requested_samples, attempts))
    return {
        "sample_count": sample_count,
        "attempt_count": attempts,
        "path_indices": [int(index) for index in path_indices],
        "shards": accumulator.shards,
        "mean_full_fragment_count": float(
            np.mean(full_fragment_counts)) if full_fragment_counts else 0.0,
        "mean_kept_fragment_count": float(
            np.mean(kept_fragment_counts)) if kept_fragment_counts else 0.0,
        "branch_count_histogram": {
            str(value): int(branch_counts.count(value))
            for value in sorted(set(branch_counts))
        },
        "trajectory_query_seconds": float(query_seconds),
        "trajectory_compression_seconds": float(compression_seconds),
        "trajectory_point_truncated_count": int(
            point_truncated_count),
        "path_reset_count": int(path_reset_count),
        "spatial_overlap_skip_count": int(
            spatial_overlap_skip_count),
        "unique_center_count": int(len(collected_center_keys)),
        "duplicate_center_sample_count": int(
            sample_count - len(collected_center_keys)),
    }


def prepare_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = _load_config(args.config)
    _prepare_output_directory(args.output_dir, args.overwrite)
    paths, subtiles = _make_paths(cfg)
    val_indices = sorted(set(args.val_subtile_indices))
    invalid = [
        index for index in val_indices
        if index < 0 or index >= len(paths)
    ]
    if invalid:
        raise ValueError(
            "validation subtile indices out of range: {}".format(invalid))
    train_indices = [
        index for index in range(len(paths))
        if index not in val_indices
    ]
    if not train_indices or not val_indices:
        raise ValueError(
            "both training and validation subtile sets must be non-empty")
    store = open_structured_trajectory_store(str(args.cache_dir))

    started_at = time.perf_counter()
    split_reports = {}
    array_schema = None
    train_center_keys = set()
    val_center_keys = set()
    index_path = args.output_dir / "sample_index.jsonl"
    with index_path.open("w", encoding="utf-8") as index_file:
        for split, path_indices, requested in (
                ("train", train_indices, args.train_samples),
                ("val", val_indices, args.val_samples)):
            accumulator = _ShardAccumulator(
                output_dir=args.output_dir,
                split=split,
                shard_size=args.shard_size,
                compressed=not args.uncompressed,
            )
            split_reports[split] = _collect_split(
                split=split,
                path_indices=path_indices,
                paths=paths,
                subtiles=subtiles,
                store=store,
                cfg=cfg,
                requested_samples=requested,
                max_attempts=max(
                    requested * args.max_attempt_multiplier,
                    requested,
                ),
                accumulator=accumulator,
                index_file=index_file,
                max_explored_edges=args.max_explored_edges,
                max_branches=args.max_branches,
                max_fragments=64,
                max_points=args.max_points_per_fragment,
                context_points=args.context_points,
                max_time_gap_seconds=args.max_time_gap_seconds,
                max_spatial_gap_pixels=args.max_spatial_gap_pixels,
                progress_every=args.progress_every,
                forbidden_center_keys=(
                    set() if split == "train" else train_center_keys),
                collected_center_keys=(
                    train_center_keys
                    if split == "train"
                    else val_center_keys
                ),
            )
            if array_schema is None:
                array_schema = accumulator.array_schema
            elif array_schema != accumulator.array_schema:
                raise RuntimeError(
                    "training and validation shard schemas differ")

    elapsed_seconds = time.perf_counter() - started_at
    manifest = {
        "region": str(cfg.TRAIN.TRAINING_REGIONS[0]),
        "source_config": str(args.config),
        "structured_trajectory_cache": str(args.cache_dir),
        "array_schema": array_schema,
        "window_size": int(cfg.TRAIN.WINDOW_SIZE),
        "walked_path_size": int(cfg.TRAIN.WINDOW_SIZE // 4),
        "max_explored_edges": int(args.max_explored_edges),
        "max_branches": int(args.max_branches),
        "trajectory": {
            "strategy": "bounded_near_diverse",
            "max_fragments": 64,
            "prepool_multiplier": 8,
            "near_fraction": 0.5,
            "max_points_per_fragment": int(
                args.max_points_per_fragment),
            "context_points": int(args.context_points),
            "max_time_gap_seconds": args.max_time_gap_seconds,
            "max_spatial_gap_pixels": args.max_spatial_gap_pixels,
            "support_count_used": False,
        },
        "split_strategy": "disjoint_2048_pixel_subtiles",
        "cross_split_center_overlap_count": int(
            len(train_center_keys & val_center_keys)),
        "train_subtile_indices": train_indices,
        "val_subtile_indices": val_indices,
        "subtiles": [
            _subtile_metadata(index, subtile)
            for index, subtile in enumerate(subtiles)
        ],
        "seed": int(args.seed),
        "shard_size": int(args.shard_size),
        "compressed_npz": not args.uncompressed,
        "elapsed_seconds": float(elapsed_seconds),
        "splits": split_reports,
    }
    write_stage3c_manifest(args.output_dir, manifest)
    validation = validate_stage3c_dataset(args.output_dir)
    report = {
        "output_dir": str(args.output_dir.resolve()),
        "manifest": manifest,
        "validation": validation,
    }
    with (args.output_dir / "build_report.json").open(
            "w", encoding="utf-8") as output_file:
        json.dump(
            report,
            output_file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        output_file.write("\n")
    return report


def _parse_int_list(value: str) -> List[int]:
    values = [
        item.strip() for item in str(value).split(",")
        if item.strip()
    ]
    if not values:
        raise argparse.ArgumentTypeError(
            "at least one subtile index is required")
    try:
        return [int(item) for item in values]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "subtile indices must be integers") from error


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare fixed-shape Stage 3C teacher-forced branch shards."))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage3c_branch_aux.yml"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data_self/input/traj_structured/xian/v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--val-samples", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=None)
    parser.add_argument(
        "--val-subtile-indices",
        type=_parse_int_list,
        default=None,
    )
    parser.add_argument(
        "--max-explored-edges", type=int, default=None)
    parser.add_argument("--max-branches", type=int, default=None)
    parser.add_argument(
        "--max-points-per-fragment", type=int, default=None)
    parser.add_argument("--context-points", type=int, default=None)
    parser.add_argument(
        "--max-time-gap-seconds", type=float, default=None)
    parser.add_argument(
        "--max-spatial-gap-pixels", type=float, default=None)
    parser.add_argument(
        "--max-attempt-multiplier", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--uncompressed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = _load_config(args.config)
    dataset_cfg = cfg.STAGE3C.DATASET
    defaults = {
        "output_dir": Path(cfg.STAGE3C.DATASET_DIR),
        "train_samples": int(dataset_cfg.TRAIN_SAMPLES),
        "val_samples": int(dataset_cfg.VAL_SAMPLES),
        "shard_size": int(dataset_cfg.SHARD_SIZE),
        "val_subtile_indices": [
            int(value) for value in dataset_cfg.VAL_SUBTILE_INDICES
        ],
        "max_explored_edges": int(dataset_cfg.MAX_EXPLORED_EDGES),
        "max_branches": int(dataset_cfg.MAX_BRANCHES),
        "max_points_per_fragment": int(
            dataset_cfg.MAX_POINTS_PER_FRAGMENT),
        "context_points": int(dataset_cfg.CONTEXT_POINTS),
        "max_time_gap_seconds": float(
            dataset_cfg.MAX_TIME_GAP_SECONDS),
        "max_spatial_gap_pixels": float(
            dataset_cfg.MAX_SPATIAL_GAP_PIXELS),
    }
    for field, default in defaults.items():
        if getattr(args, field) is None:
            setattr(args, field, default)
    positive_fields = (
        "train_samples",
        "val_samples",
        "shard_size",
        "max_branches",
        "max_points_per_fragment",
        "max_attempt_multiplier",
    )
    for field in positive_fields:
        if getattr(args, field) <= 0:
            raise ValueError("{} must be positive".format(field))
    if args.max_explored_edges < 0 or args.context_points < 0:
        raise ValueError(
            "max_explored_edges and context_points must be non-negative")
    report = prepare_dataset(args)
    print(json.dumps({
        "output_dir": report["output_dir"],
        "elapsed_seconds": report["manifest"]["elapsed_seconds"],
        "split_strategy": report["manifest"]["split_strategy"],
        "splits": {
            key: {
                "sample_count": value["sample_count"],
                "shard_count": len(value["shards"]),
                "branch_count_histogram": value[
                    "branch_count_histogram"],
                "trajectory_point_truncated_count": value[
                    "trajectory_point_truncated_count"],
            }
            for key, value in report["manifest"]["splits"].items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
