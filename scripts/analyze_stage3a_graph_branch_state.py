"""Inspect Stage 3A state and immediate branches on the training path flow."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
import torch
import yaml
from easydict import EasyDict


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.branch_targets import build_immediate_branch_targets
from utils.graph_state import build_graph_state
from utils.model_utils import Path as VecRoadPath
from utils.tileloader import Tiles
from utils.trajectory_mode import (
    TRAJ_MODE_NONE,
    resolve_trajectory_mode,
)


NODE_TYPES = ("ordinary", "t_junction", "multi_branch", "other")


def _distribution(values: Sequence[float]) -> Dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def _integer_histogram(values: Sequence[int]) -> Dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(Counter(int(x) for x in values).items())
    }


def _load_config(config_path: Path) -> EasyDict:
    with config_path.open("r", encoding="utf-8") as config_file:
        cfg = EasyDict(yaml.load(
            config_file, Loader=yaml.UnsafeLoader))
    trajectory_mode = resolve_trajectory_mode(cfg)
    if trajectory_mode != TRAJ_MODE_NONE:
        raise ValueError(
            "Stage 3A analysis must use TRAJ.MODE=none; resolved {!r}".format(
                trajectory_mode))
    return cfg


def _make_training_paths(cfg: EasyDict) -> Tuple[List[VecRoadPath], int]:
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
        VecRoadPath(
            idx=index,
            training=True,
            gc=subtile["gc"].clone(),
            tile_data=subtile,
            all_trajectories=[],
            all_pixel_trajectories=[],
        )
        for index, subtile in enumerate(subtiles)
    ]
    return paths, len(subtiles)


def _gt_degree_and_type(path, search_state) -> Tuple[Optional[int], str]:
    vertex = search_state.vertex
    if vertex.edge_pos is None:
        return None, "other"
    edge = vertex.edge_pos.edge(path.gc.graph)
    gt_vertex = None
    if edge.src(path.gc.graph).point == vertex.point:
        gt_vertex = edge.src(path.gc.graph)
    elif edge.dst(path.gc.graph).point == vertex.point:
        gt_vertex = edge.dst(path.gc.graph)

    if gt_vertex is None:
        return 2, "ordinary"
    degree = len(set(gt_vertex.neighbors(path.gc.graph)))
    if degree == 2:
        return degree, "ordinary"
    if degree == 3:
        return degree, "t_junction"
    if degree >= 4:
        return degree, "multi_branch"
    return degree, "other"


def _record_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    branch_counts = [record["branch_count"] for record in records]
    explored_counts = [
        record["explored_edge_count"] for record in records]
    incoming_count = sum(
        bool(record["incoming_valid"]) for record in records)
    later_slot_nonempty = sum(
        any(count > 0 for count in record["target_slot_counts"][1:])
        for record in records)
    return {
        "sample_count": len(records),
        "immediate_branch_count": {
            "distribution": _distribution(branch_counts),
            "histogram": _integer_histogram(branch_counts),
        },
        "maximum_immediate_branch_count": (
            max(branch_counts) if branch_counts else 0),
        "incoming_valid_count": int(incoming_count),
        "incoming_valid_rate": (
            float(incoming_count / len(records)) if records else 0.0),
        "explored_edge_count": {
            "distribution": _distribution(explored_counts),
            "histogram": _integer_histogram(explored_counts),
        },
        "later_target_slots_nonempty_count": int(later_slot_nonempty),
        "later_target_slots_excluded_from_branch_count": True,
    }


def _sample_records(
        paths: Sequence[VecRoadPath],
        cfg: EasyDict,
        max_states: int,
        max_attempts: int,
        max_explored_edges: int,
) -> List[Dict[str, Any]]:
    records = []
    active_path_indices = list(range(len(paths)))
    path_cursor = 0
    attempts = 0

    while (
            len(records) < max_states
            and active_path_indices
            and attempts < max_attempts):
        attempts += 1
        active_position = path_cursor % len(active_path_indices)
        path_index = active_path_indices[active_position]
        path = paths[path_index]
        state = path.pop_state(
            follow_order=False,
            probs=[0.15, 0.8, 0.05],
            WINDOW_SIZE=cfg.TRAIN.WINDOW_SIZE,
        )
        if (
                state is None
                or len(path.graph.vertices) >= cfg.TRAIN.MAX_PATH_LENGTH):
            active_path_indices.pop(active_position)
            if active_path_indices:
                path_cursor %= len(active_path_indices)
            continue
        path_cursor += 1

        graph_features = build_graph_state(
            path,
            state,
            max_explored_edges=max_explored_edges,
        )
        road_segmentation = None
        if state.vertex.edge_pos is not None:
            local_input = path.make_path_input(
                extension_vertex=state.vertex,
                fetch_list=["road_seg_thick3"],
                traj_filter=False,
                is_key_point=state.is_key_point,
                WINDOW_SIZE=cfg.TRAIN.WINDOW_SIZE,
            )
            road_segmentation = local_input["road_seg_thick3"][0]

        target_poses = path.get_target_poses(
            extension_vertex=state.vertex,
            road_segmentation=road_segmentation,
            STEP_LENGTH=cfg.TRAIN.STEP_LENGTH,
            is_key_point=state.is_key_point,
            NUM_TARGETS=cfg.TRAIN.NUM_TARGETS,
            RECT_RADIUS=cfg.TRAIN.RECT_RADIUS,
            WINDOW_SIZE=cfg.TRAIN.WINDOW_SIZE,
        )
        branches = build_immediate_branch_targets(
            target_poses=target_poses,
            current_vertex=state.vertex,
            graph=path.gc.graph,
            window_size=cfg.TRAIN.WINDOW_SIZE,
        )
        gt_degree, node_type = _gt_degree_and_type(path, state)
        edge_mask = graph_features["explored_edge_mask"]
        explored_dirs = graph_features["explored_edge_dirs"][edge_mask]
        record = {
            "sample_index": len(records),
            "path_index": int(path_index),
            "vertex_id": int(state.vertex.id),
            "center_xy": [
                float(state.vertex.point.x),
                float(state.vertex.point.y),
            ],
            "is_key_point": bool(state.is_key_point),
            "parent_vertex_id": state.parent_vertex_id,
            "incoming_edge_id": state.incoming_edge_id,
            "incoming_valid": bool(
                graph_features["incoming_valid"].item()),
            "incoming_dir": (
                graph_features["incoming_dir"].tolist()),
            "explored_edge_count": int(edge_mask.sum().item()),
            "explored_edge_dirs": explored_dirs.tolist(),
            "explored_is_incoming": (
                graph_features["explored_is_incoming"][edge_mask].tolist()),
            "branch_count": int(branches.branch_count),
            "branch_offsets_rel": (
                branches.branch_offsets_rel.tolist()),
            "branch_directions": (
                branches.branch_directions.tolist()),
            "target_slot_counts": [
                len(slot) for slot in target_poses.target_poses
            ],
            "gt_degree": gt_degree,
            "node_type": node_type,
        }
        records.append(record)

        if state.vertex.edge_pos is None or len(target_poses) == 0:
            continue

        # Match the current follow_target training behavior after recording
        # the complete, pre-sampling immediate branch supervision.
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

    return records


def _display_crop(
        image: Image.Image,
        center_xy: Sequence[float],
        window_size: float,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    margin = max(32.0, window_size * 0.25)
    extent = window_size / 2.0 + margin
    left = max(0, int(math.floor(center_xy[0] - extent)))
    top = max(0, int(math.floor(center_xy[1] - extent)))
    right = min(image.width, int(math.ceil(center_xy[0] + extent)))
    bottom = min(image.height, int(math.ceil(center_xy[1] + extent)))
    if left >= right or top >= bottom:
        raise ValueError("state visualization lies outside background")
    return np.asarray(image.crop((left, top, right, bottom))), (
        left, top, right, bottom)


def _visualize_record(
        record: Dict[str, Any],
        output_path: Path,
        window_size: float,
        background: Optional[Image.Image],
) -> None:
    center = np.asarray(record["center_xy"], dtype=np.float64)
    half_window = window_size / 2.0
    margin = max(32.0, window_size * 0.25)
    figure, axis = plt.subplots(figsize=(8, 8))
    if background is not None:
        crop, (left, top, right, bottom) = _display_crop(
            background, center, window_size)
        axis.imshow(
            crop,
            extent=(left, right, bottom, top),
            origin="upper",
        )

    incoming = np.asarray(record["incoming_dir"], dtype=np.float64)
    if record["incoming_valid"]:
        start = center - incoming * 70.0
        delta = center - start
        axis.arrow(
            start[0], start[1], delta[0], delta[1],
            color="deepskyblue", width=1.4, head_width=10,
            length_includes_head=True, zorder=5,
            label="incoming direction",
        )

    for index, direction in enumerate(record["explored_edge_dirs"]):
        direction = np.asarray(direction, dtype=np.float64)
        end = center + direction * 65.0
        axis.plot(
            [center[0], end[0]], [center[1], end[1]],
            color="orange", linewidth=3.0, alpha=0.9,
            label="explored neighbor" if index == 0 else None,
            zorder=4,
        )

    for index, offset in enumerate(record["branch_offsets_rel"]):
        endpoint = center + np.asarray(offset, dtype=np.float64)
        axis.plot(
            [center[0], endpoint[0]], [center[1], endpoint[1]],
            color="magenta", linewidth=3.0, linestyle="--",
            label="immediate GT branch" if index == 0 else None,
            zorder=6,
        )
        axis.scatter(
            endpoint[0], endpoint[1], color="yellow",
            edgecolors="black", s=55, zorder=7)

    axis.add_patch(Rectangle(
        (center[0] - half_window, center[1] - half_window),
        window_size,
        window_size,
        fill=False,
        edgecolor="red",
        linewidth=2.0,
        zorder=3,
    ))
    axis.scatter(
        center[0], center[1], color="lime", marker="*",
        edgecolors="black", s=180, zorder=8, label="current node")
    axis.set_xlim(
        center[0] - half_window - margin,
        center[0] + half_window + margin,
    )
    axis.set_ylim(
        center[1] + half_window + margin,
        center[1] - half_window - margin,
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(
        "{} | branches={} incoming={} explored={}".format(
            record["node_type"],
            record["branch_count"],
            record["incoming_valid"],
            record["explored_edge_count"],
        )
    )
    axis.set_xlabel("global pixel x")
    axis.set_ylabel("global pixel y")
    axis.legend(loc="upper right", fontsize=8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(figure)


def analyze_stage3a(
        config_path: Path,
        output_dir: Path,
        max_states: int,
        max_attempts: int,
        max_explored_edges: int,
        seed: int,
        background_image: Optional[Path],
        visualizations_per_type: int,
) -> Dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cfg = _load_config(config_path)
    paths, subtile_count = _make_training_paths(cfg)

    started_at = time.perf_counter()
    records = _sample_records(
        paths=paths,
        cfg=cfg,
        max_states=max_states,
        max_attempts=max_attempts,
        max_explored_edges=max_explored_edges,
    )
    elapsed_seconds = time.perf_counter() - started_at

    by_node_type = {
        node_type: _record_summary([
            record for record in records
            if record["node_type"] == node_type
        ])
        for node_type in NODE_TYPES
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir = output_dir / "visualizations"
    background = None
    if background_image is not None:
        background = Image.open(background_image).convert("RGB")

    visualizations = []
    try:
        for node_type in NODE_TYPES:
            candidates = [
                record for record in records
                if record["node_type"] == node_type
                and record["branch_count"] > 0
            ]
            candidates.sort(key=lambda record: (
                0 if record["incoming_valid"] else 1,
                -record["branch_count"],
                record["sample_index"],
            ))
            for record in candidates[:visualizations_per_type]:
                output_path = visualization_dir / (
                    "{}_sample_{:04d}.png".format(
                        node_type, record["sample_index"]))
                _visualize_record(
                    record=record,
                    output_path=output_path,
                    window_size=float(cfg.TRAIN.WINDOW_SIZE),
                    background=background,
                )
                visualizations.append({
                    "node_type": node_type,
                    "sample_index": record["sample_index"],
                    "branch_count": record["branch_count"],
                    "incoming_valid": record["incoming_valid"],
                    "explored_edge_count": (
                        record["explored_edge_count"]),
                    "output_path": str(output_path.resolve()),
                })
    finally:
        if background is not None:
            background.close()

    report = {
        "stage": "3A",
        "config_path": str(config_path.resolve()),
        "trajectory_mode": "none",
        "target_source": "Path.get_target_poses training flow",
        "immediate_branch_source": "target_poses[0]",
        "later_target_slots_excluded": True,
        "num_targets_unchanged": int(cfg.TRAIN.NUM_TARGETS),
        "window_size": int(cfg.TRAIN.WINDOW_SIZE),
        "step_length": int(cfg.TRAIN.STEP_LENGTH),
        "seed": int(seed),
        "subtile_count": int(subtile_count),
        "requested_max_states": int(max_states),
        "sampled_state_count": len(records),
        "elapsed_seconds": float(elapsed_seconds),
        "max_explored_edges": int(max_explored_edges),
        "overall": _record_summary(records),
        "by_node_type": by_node_type,
        "node_type_histogram": {
            node_type: sum(
                record["node_type"] == node_type for record in records)
            for node_type in NODE_TYPES
        },
        "records": records,
        "visualizations": visualizations,
    }
    report_path = output_dir / "stage3a_graph_branch_state.json"
    report["report_path"] = str(report_path.resolve())
    with report_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            report,
            output_file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        output_file.write("\n")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Stage 3A graph exploration state and immediate branch "
            "targets on the existing VecRoad training target flow."))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline_image_only.yml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-states", type=int, default=512)
    parser.add_argument("--max-attempts", type=int, default=8192)
    parser.add_argument("--max-explored-edges", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--background-image", type=Path, default=None)
    parser.add_argument("--visualizations-per-type", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.max_states <= 0:
        raise ValueError("max_states must be positive")
    if args.max_attempts < args.max_states:
        raise ValueError("max_attempts must be at least max_states")
    if args.max_explored_edges < 0:
        raise ValueError("max_explored_edges must be non-negative")
    if args.visualizations_per_type < 0:
        raise ValueError(
            "visualizations_per_type must be non-negative")

    report = analyze_stage3a(
        config_path=args.config,
        output_dir=args.output_dir,
        max_states=args.max_states,
        max_attempts=args.max_attempts,
        max_explored_edges=args.max_explored_edges,
        seed=args.seed,
        background_image=args.background_image,
        visualizations_per_type=args.visualizations_per_type,
    )
    compact = {
        "sampled_state_count": report["sampled_state_count"],
        "node_type_histogram": report["node_type_histogram"],
        "overall": report["overall"],
        "by_node_type": report["by_node_type"],
        "elapsed_seconds": report["elapsed_seconds"],
        "report_path": report["report_path"],
        "visualization_count": len(report["visualizations"]),
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
