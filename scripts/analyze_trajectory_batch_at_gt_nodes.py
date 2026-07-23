"""Measure Stage 1B/1C behavior at real GT road-graph nodes."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib import graph as graph_helper
from scripts.visualize_trajectory_fragments import (
    verify_background_alignment,
    visualize_fragments,
)
from utils.structured_trajectory_store import (
    open_structured_trajectory_store,
)
from utils.trajectory_batch import (
    build_trajectory_batch,
    fragment_minimum_distance,
)
from utils.trajectory_fragments import SEGMENT_GRID_INDEX_BASIS


NODE_TYPES = ("ordinary", "t_junction", "multi_branch")
DISTANCE_RADII = (10.0, 20.0, 40.0, 80.0)


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


def _node_type(degree: int) -> Optional[str]:
    if degree == 2:
        return "ordinary"
    if degree == 3:
        return "t_junction"
    if degree >= 4:
        return "multi_branch"
    return None


def _graph_nodes(graph) -> Dict[str, List[Dict[str, Any]]]:
    nodes = {node_type: [] for node_type in NODE_TYPES}
    for vertex_id in sorted(graph.vertices):
        vertex = graph.vertices[vertex_id]
        neighbor_ids = sorted(set(vertex.neighbors(graph)))
        node_type = _node_type(len(neighbor_ids))
        if node_type is None:
            continue
        center_xy = [
            float(vertex.point.x),
            float(vertex.point.y),
        ]
        incident_segments = []
        for neighbor_id in neighbor_ids:
            neighbor = graph.vertices[neighbor_id]
            incident_segments.append(
                np.asarray(
                    [
                        center_xy,
                        [float(neighbor.point.x), float(neighbor.point.y)],
                    ],
                    dtype=np.float64,
                )
            )
        nodes[node_type].append(
            {
                "vertex_id": int(vertex_id),
                "degree": len(neighbor_ids),
                "center_xy": center_xy,
                "incident_segments": incident_segments,
            }
        )
    return nodes


def _deterministic_subset(
    nodes: Sequence[Dict[str, Any]],
    limit: Optional[int],
    seed: int,
) -> List[Dict[str, Any]]:
    if limit is None or limit >= len(nodes):
        return list(nodes)
    if limit <= 0:
        return []
    random = np.random.default_rng(seed)
    selected_indices = np.sort(
        random.choice(len(nodes), size=limit, replace=False))
    return [nodes[int(index)] for index in selected_indices]


def _is_segment_only(fragment, center_xy, window_size: float) -> bool:
    relative = (
        np.asarray(fragment.points_global_xy, dtype=np.float64)
        - np.asarray(center_xy, dtype=np.float64)
    )
    half_window = float(window_size) / 2.0
    inside = (
        (relative[:, 0] >= -half_window)
        & (relative[:, 0] <= half_window)
        & (relative[:, 1] >= -half_window)
        & (relative[:, 1] <= half_window)
    )
    return not bool(inside.any())


def _budget_summary(
    fragment_counts: Sequence[int],
    budget: int,
) -> Dict[str, Any]:
    total = int(sum(fragment_counts))
    truncated_by_node = [
        max(0, int(count) - budget) for count in fragment_counts
    ]
    truncated = int(sum(truncated_by_node))
    truncated_nodes = sum(value > 0 for value in truncated_by_node)
    return {
        "max_fragments": int(budget),
        "kept_fragment_count": total - truncated,
        "truncated_fragment_count": truncated,
        "fragment_truncation_rate": (
            float(truncated / total) if total else 0.0
        ),
        "truncated_node_count": int(truncated_nodes),
        "node_truncation_rate": (
            float(truncated_nodes / len(fragment_counts))
            if fragment_counts
            else 0.0
        ),
    }


def _summarize_records(
    records: Sequence[Dict[str, Any]],
    budgets: Sequence[int],
) -> Dict[str, Any]:
    fragment_counts = [
        int(record["fragment_count"]) for record in records
    ]
    point_counts = [
        int(record["point_count"]) for record in records
    ]
    per_fragment_point_counts = [
        int(point_count)
        for record in records
        for point_count in record["fragment_point_counts"]
    ]
    return {
        "node_count": len(records),
        "fragment_count_per_node": _distribution(fragment_counts),
        "track_count_per_node": _distribution(
            [record["track_count"] for record in records]),
        "point_count_per_node": _distribution(point_counts),
        "point_count_per_fragment": _distribution(
            per_fragment_point_counts),
        "candidate_track_count_per_node": _distribution(
            [record["candidate_track_count"] for record in records]),
        "segment_only_fragment_count_per_node": _distribution(
            [
                record["segment_only_fragment_count"]
                for record in records
            ]
        ),
        "fragment_counts_within_distance": {
            str(int(radius)): _distribution(
                [
                    record["fragment_counts_within_distance"][
                        str(int(radius))
                    ]
                    for record in records
                ]
            )
            for radius in DISTANCE_RADII
        },
        "candidate_index_query_ms": _distribution(
            [record["candidate_index_query_ms"] for record in records]),
        "fragment_query_ms": _distribution(
            [record["fragment_query_ms"] for record in records]),
        "batch_build_ms_at_largest_budget": _distribution(
            [
                record["batch_build_ms_at_largest_budget"]
                for record in records
            ]
        ),
        "truncation": {
            str(budget): _budget_summary(fragment_counts, budget)
            for budget in budgets
        },
    }


def _select_visualization_records(
    records: Sequence[Dict[str, Any]],
    cases_per_type: int,
) -> List[Dict[str, Any]]:
    if cases_per_type <= 0:
        return []
    selected = []
    for node_type in NODE_TYPES:
        candidates = [
            record
            for record in records
            if record["node_type"] == node_type
            and record["fragment_count"] > 0
        ]
        if not candidates:
            continue
        counts = np.asarray(
            [record["fragment_count"] for record in candidates],
            dtype=np.float64,
        )
        quantiles = np.linspace(0.5, 0.9, cases_per_type)
        used_vertex_ids = set()
        for quantile in quantiles:
            target = float(np.quantile(counts, quantile))
            ordered = sorted(
                candidates,
                key=lambda record: (
                    abs(record["fragment_count"] - target),
                    record["vertex_id"],
                ),
            )
            for record in ordered:
                if record["vertex_id"] not in used_vertex_ids:
                    selected.append(record)
                    used_vertex_ids.add(record["vertex_id"])
                    break
    return selected


def analyze_gt_nodes(
    cache_dir: Path,
    graph_path: Path,
    output_dir: Path,
    window_size: float,
    context_points: int,
    max_time_gap_seconds: Optional[float],
    max_spatial_gap_pixels: Optional[float],
    budgets: Sequence[int],
    max_nodes_per_type: Optional[int],
    seed: int,
    background_image: Optional[Path],
    visualization_cases_per_type: int,
    visualization_max_fragments: Optional[int],
) -> Dict[str, Any]:
    store = open_structured_trajectory_store(os.fspath(cache_dir))
    grid_basis = store.meta.get("grid_index_basis")
    if grid_basis != SEGMENT_GRID_INDEX_BASIS:
        raise RuntimeError(
            "Stage 1C requires segment-aware cache basis {!r}; found {!r}. "
            "Rebuild with prepare_structured_trajectory_cache.py "
            "--overwrite.".format(SEGMENT_GRID_INDEX_BASIS, grid_basis)
        )
    graph = graph_helper.read_graph(
        os.fspath(graph_path), merge_duplicates=True)
    all_nodes = _graph_nodes(graph)
    selected_nodes = {
        node_type: _deterministic_subset(
            all_nodes[node_type],
            max_nodes_per_type,
            seed + node_type_index,
        )
        for node_type_index, node_type in enumerate(NODE_TYPES)
    }
    largest_budget = max(budgets) if budgets else None
    half_window = window_size / 2.0
    records = []
    total_selected_nodes = sum(map(len, selected_nodes.values()))
    completed_nodes = 0

    for node_type in NODE_TYPES:
        for node in selected_nodes[node_type]:
            center_xy = node["center_xy"]
            bounds = (
                center_xy[0] - half_window,
                center_xy[1] - half_window,
                center_xy[0] + half_window,
                center_xy[1] + half_window,
            )
            index_start = time.perf_counter()
            candidate_ids = store.candidate_track_ids_for_rect(*bounds)
            index_time_ms = (
                time.perf_counter() - index_start) * 1000.0

            query_start = time.perf_counter()
            fragments = store.query_trajectory_fragments(
                center_xy=center_xy,
                window_size=window_size,
                context_points=context_points,
                max_time_gap_seconds=max_time_gap_seconds,
                max_spatial_gap_pixels=max_spatial_gap_pixels,
            )
            query_time_ms = (
                time.perf_counter() - query_start) * 1000.0

            batch_start = time.perf_counter()
            batch = build_trajectory_batch(
                [fragments],
                center_xy=[center_xy],
                window_size=window_size,
                max_fragments=largest_budget,
            )
            batch_time_ms = (
                time.perf_counter() - batch_start) * 1000.0
            if int(batch["total_fragment_count"][0]) != len(fragments):
                raise AssertionError(
                    "batch builder changed the total fragment count")

            distances = [
                fragment_minimum_distance(fragment, center_xy)
                for fragment in fragments
            ]
            record = {
                "node_type": node_type,
                "vertex_id": node["vertex_id"],
                "degree": node["degree"],
                "center_xy": center_xy,
                "candidate_track_count": int(candidate_ids.size),
                "track_count": len(
                    {fragment.track_index for fragment in fragments}),
                "fragment_count": len(fragments),
                "point_count": int(
                    sum(len(fragment) for fragment in fragments)),
                "fragment_point_counts": [
                    len(fragment) for fragment in fragments
                ],
                "segment_only_fragment_count": sum(
                    _is_segment_only(
                        fragment, center_xy, window_size)
                    for fragment in fragments
                ),
                "fragment_counts_within_distance": {
                    str(int(radius)): sum(
                        distance <= radius for distance in distances)
                    for radius in DISTANCE_RADII
                },
                "candidate_index_query_ms": index_time_ms,
                "fragment_query_ms": query_time_ms,
                "batch_build_ms_at_largest_budget": batch_time_ms,
            }
            records.append(record)
            completed_nodes += 1
            if completed_nodes % 50 == 0 or (
                completed_nodes == total_selected_nodes
            ):
                print(
                    "analyzed {}/{} GT nodes".format(
                        completed_nodes, total_selected_nodes),
                    flush=True,
                )

    background_alignment = None
    if background_image is not None:
        with Image.open(background_image) as image_file:
            background_alignment = verify_background_alignment(
                image_file.convert("RGB"), store.meta)

    node_lookup = {
        (node_type, node["vertex_id"]): node
        for node_type in NODE_TYPES
        for node in selected_nodes[node_type]
    }
    visualization_reports = []
    visualization_records = _select_visualization_records(
        records, visualization_cases_per_type)
    visualization_dir = output_dir / "visualizations"
    for record in visualization_records:
        node = node_lookup[(record["node_type"], record["vertex_id"])]
        output_path = visualization_dir / (
            "{}_vertex_{:04d}_fragments_{:04d}.png".format(
                record["node_type"],
                record["vertex_id"],
                record["fragment_count"],
            )
        )
        visual_report = visualize_fragments(
            cache_dir=cache_dir,
            center_xy=record["center_xy"],
            window_size=window_size,
            context_points=context_points,
            output_path=output_path,
            background_image=background_image,
            max_time_gap_seconds=max_time_gap_seconds,
            max_spatial_gap_pixels=max_spatial_gap_pixels,
            reference_segments_xy=node["incident_segments"],
            max_fragments=visualization_max_fragments,
        )
        visual_report.update(
            {
                "node_type": record["node_type"],
                "vertex_id": record["vertex_id"],
                "degree": record["degree"],
            }
        )
        visualization_reports.append(visual_report)

    category_reports = {
        node_type: _summarize_records(
            [
                record
                for record in records
                if record["node_type"] == node_type
            ],
            budgets,
        )
        for node_type in NODE_TYPES
    }
    report = {
        "cache_dir": str(cache_dir.resolve()),
        "graph_path": str(graph_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "grid_index_basis": grid_basis,
        "segment_aware_index_active": True,
        "query_uses_full_trajectory_scan": False,
        "trajectory_count": int(store.trajectory_count),
        "point_count": int(store.point_count),
        "window_size": float(window_size),
        "context_points": int(context_points),
        "max_time_gap_seconds": max_time_gap_seconds,
        "max_spatial_gap_pixels": max_spatial_gap_pixels,
        "seed": int(seed),
        "budgets": list(budgets),
        "visualization_max_fragments": visualization_max_fragments,
        "graph_vertex_count": len(graph.vertices),
        "graph_directed_edge_count": len(graph.edges),
        "node_population": {
            node_type: len(all_nodes[node_type])
            for node_type in NODE_TYPES
        },
        "analyzed_node_count": {
            node_type: len(selected_nodes[node_type])
            for node_type in NODE_TYPES
        },
        "overall": _summarize_records(records, budgets),
        "by_node_type": category_reports,
        "background_alignment": background_alignment,
        "visualizations": visualization_reports,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "gt_node_trajectory_batch_stats.json"
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


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Stage 1B fragments and Stage 1C batches at real GT "
            "road nodes."
        )
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--background-image", type=Path, default=None)
    parser.add_argument("--window-size", type=float, default=256.0)
    parser.add_argument("--context-points", type=int, default=2)
    parser.add_argument("--max-time-gap-seconds", type=float, default=None)
    parser.add_argument("--max-spatial-gap-pixels", type=float, default=None)
    parser.add_argument(
        "--budgets",
        type=_positive_int,
        nargs="+",
        default=[32, 64, 128, 256],
    )
    parser.add_argument("--max-nodes-per-type", type=_positive_int)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument(
        "--visualization-cases-per-type",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--visualization-max-fragments",
        type=_positive_int,
        default=64,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.window_size <= 0.0 or not math.isfinite(args.window_size):
        raise ValueError("window_size must be finite and positive")
    if args.context_points < 0:
        raise ValueError("context_points must be non-negative")
    if args.visualization_cases_per_type < 0:
        raise ValueError(
            "visualization_cases_per_type must be non-negative")
    budgets = sorted(set(args.budgets))
    report = analyze_gt_nodes(
        cache_dir=args.cache_dir,
        graph_path=args.graph,
        output_dir=args.output_dir,
        window_size=args.window_size,
        context_points=args.context_points,
        max_time_gap_seconds=args.max_time_gap_seconds,
        max_spatial_gap_pixels=args.max_spatial_gap_pixels,
        budgets=budgets,
        max_nodes_per_type=args.max_nodes_per_type,
        seed=args.seed,
        background_image=args.background_image,
        visualization_cases_per_type=(
            args.visualization_cases_per_type),
        visualization_max_fragments=args.visualization_max_fragments,
    )
    compact = {
        "grid_index_basis": report["grid_index_basis"],
        "node_population": report["node_population"],
        "analyzed_node_count": report["analyzed_node_count"],
        "overall": report["overall"],
        "background_alignment": report["background_alignment"],
        "report_path": report["report_path"],
        "visualization_count": len(report["visualizations"]),
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
