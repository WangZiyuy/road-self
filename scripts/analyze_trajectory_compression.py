"""Compare Stage 2B trajectory compression at real GT road-graph nodes."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib import graph as graph_helper
from scripts.analyze_trajectory_batch_at_gt_nodes import (
    NODE_TYPES,
    _distribution,
    _graph_nodes,
)
from scripts.visualize_trajectory_fragments import (
    _display_crop,
    verify_background_alignment,
)
from utils.structured_trajectory_store import (
    open_structured_trajectory_store,
)
from utils.trajectory_compression import (
    COMPRESSION_TIMING_PHASES,
    CompressionResult,
    compress_trajectory_fragments,
    fragment_minimum_distance,
)
from utils.trajectory_fragments import SEGMENT_GRID_INDEX_BASIS


STRATEGIES = (
    "nearest",
    "near_diverse",
    "bounded_near_diverse",
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _probe_points(
    center_xy: Sequence[float],
    incident_segments: Sequence[np.ndarray],
    probe_distance: float,
) -> np.ndarray:
    center = np.asarray(center_xy, dtype=np.float64)
    probes = []
    for segment in incident_segments:
        segment_array = np.asarray(segment, dtype=np.float64)
        direction = segment_array[1] - center
        length = float(np.linalg.norm(direction))
        if length <= 0.0:
            continue
        probes.append(
            center + direction / length * min(probe_distance, length))
    if not probes:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(probes, dtype=np.float64)


def _branch_coverage(
    result: CompressionResult,
    probe_points: np.ndarray,
    evidence_distance: float,
) -> Tuple[int, int, List[bool]]:
    covered = []
    for probe_point in probe_points:
        has_evidence = any(
            fragment_minimum_distance(fragment, probe_point)
            <= evidence_distance
            for fragment in result.selected_fragments
        )
        covered.append(bool(has_evidence))
    return sum(covered), len(covered), covered


def _pairwise_rms(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    differences = points[:, None, :] - points[None, :, :]
    squared_distance = np.sum(differences * differences, axis=2)
    upper = squared_distance[
        np.triu_indices(points.shape[0], k=1)]
    return float(np.sqrt(np.mean(upper)))


def _selection_metrics(
    result: CompressionResult,
    covered_branch_count: int,
    branch_count: int,
    center_xy: Sequence[float],
    window_size: float,
) -> Dict[str, Any]:
    descriptors = result.selected_geometry_descriptors
    axis = descriptors[:, 4:6].astype(np.float64)
    valid_axis = np.linalg.norm(axis, axis=1) > 0.5
    if np.any(valid_axis):
        mean_axis = np.mean(axis[valid_axis], axis=0)
        axis_dispersion = float(
            np.clip(1.0 - np.linalg.norm(mean_axis), 0.0, 1.0))
        axis_pairwise_rms = _pairwise_rms(axis[valid_axis])
    else:
        axis_dispersion = 0.0
        axis_pairwise_rms = 0.0
    nearest_positions = descriptors[:, 0:2].astype(np.float64)
    center = np.asarray(center_xy, dtype=np.float64)
    half_window = float(window_size) / 2.0
    segment_only_flags = []
    for fragment in result.selected_fragments:
        relative = (
            np.asarray(fragment.points_global_xy, dtype=np.float64)
            - center
        )
        inside = (
            (relative[:, 0] >= -half_window)
            & (relative[:, 0] <= half_window)
            & (relative[:, 1] >= -half_window)
            & (relative[:, 1] <= half_window)
        )
        segment_only_flags.append(not bool(inside.any()))
    support_values = (
        []
        if result.support_count is None
        else result.support_count.astype(int).tolist()
    )
    return {
        "compression_time_ms": float(
            result.compression_timing_ms["total"]),
        "compression_timing_ms": dict(
            result.compression_timing_ms),
        "prepool_count": int(result.prepool_count),
        "descriptor_evaluation_count": int(
            result.descriptor_evaluation_count),
        "unique_track_count": len({
            int(fragment.track_index)
            for fragment in result.selected_fragments
        }),
        "support_count_valid": bool(result.support_count_valid),
        "support_count": support_values,
        "support_count_max": (
            int(np.max(result.support_count))
            if result.support_count is not None
            and result.support_count.size
            else 0
        ),
        "support_count_mean": (
            float(np.mean(result.support_count))
            if result.support_count is not None
            and result.support_count.size
            else 0.0
        ),
        "segment_only_count": int(sum(segment_only_flags)),
        "segment_only_ratio": (
            float(np.mean(segment_only_flags))
            if segment_only_flags
            else 0.0
        ),
        "invalid_axis_count": int(np.sum(~valid_axis)),
        "invalid_axis_ratio": (
            float(np.mean(~valid_axis))
            if valid_axis.size
            else 0.0
        ),
        "fragment_min_distance": (
            result.selected_minimum_distances.astype(float).tolist()),
        "axis_dispersion": axis_dispersion,
        "axis_pairwise_rms": axis_pairwise_rms,
        "nearest_position_pairwise_rms_norm": _pairwise_rms(
            nearest_positions),
        "covered_branch_count": int(covered_branch_count),
        "branch_count": int(branch_count),
        "branch_coverage": (
            float(covered_branch_count / branch_count)
            if branch_count
            else 1.0
        ),
        "all_branches_covered": bool(
            branch_count == 0 or covered_branch_count == branch_count),
    }


def _summarize_results(
    results: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    support_counts = [
        support
        for result in results
        for support in result["support_count"]
    ]
    total_branches = sum(
        int(result["branch_count"]) for result in results)
    covered_branches = sum(
        int(result["covered_branch_count"]) for result in results)
    selected_minimum_distances = [
        distance
        for result in results
        for distance in result["fragment_min_distance"]
    ]
    return {
        "node_count": len(results),
        "compression_time_ms": _distribution([
            result["compression_time_ms"] for result in results
        ]),
        "compression_timing_ms": {
            phase: _distribution([
                result["compression_timing_ms"][phase]
                for result in results
            ])
            for phase in COMPRESSION_TIMING_PHASES
        },
        "prepool_count": _distribution([
            result["prepool_count"] for result in results
        ]),
        "descriptor_evaluation_count": _distribution([
            result["descriptor_evaluation_count"]
            for result in results
        ]),
        "unique_track_count": _distribution([
            result["unique_track_count"] for result in results
        ]),
        "support_count": _distribution(support_counts),
        "representative_max_support_count": _distribution([
            result["support_count_max"] for result in results
        ]),
        "support_count_valid_node_count": sum(
            bool(result["support_count_valid"])
            for result in results
        ),
        "segment_only_ratio": _distribution([
            result["segment_only_ratio"] for result in results
        ]),
        "invalid_axis_ratio": _distribution([
            result["invalid_axis_ratio"] for result in results
        ]),
        "fragment_min_distance": _distribution(
            selected_minimum_distances),
        "axis_dispersion": _distribution([
            result["axis_dispersion"] for result in results
        ]),
        "axis_pairwise_rms": _distribution([
            result["axis_pairwise_rms"] for result in results
        ]),
        "nearest_position_pairwise_rms_norm": _distribution([
            result["nearest_position_pairwise_rms_norm"]
            for result in results
        ]),
        "covered_branch_count": int(covered_branches),
        "branch_count": int(total_branches),
        "evidence_coverage_at_k": (
            float(covered_branches / total_branches)
            if total_branches
            else 1.0
        ),
        "all_branches_covered_node_count": sum(
            bool(result["all_branches_covered"])
            for result in results
        ),
        "all_branches_covered_node_rate": (
            float(np.mean([
                bool(result["all_branches_covered"])
                for result in results
            ]))
            if results
            else 0.0
        ),
    }


def _plot_fragments(
    axis,
    fragments,
    support_counts: Optional[np.ndarray],
    representative: bool,
    title: str,
    center_xy: Sequence[float],
    window_size: float,
    incident_segments: Sequence[np.ndarray],
    probe_points: np.ndarray,
    background_crop,
    background_extent,
) -> None:
    center_x, center_y = map(float, center_xy)
    half_window = window_size / 2.0
    if background_crop is not None:
        left, top, right, bottom = background_extent
        axis.imshow(
            background_crop,
            extent=(left, right, bottom, top),
            origin="upper",
        )
    for segment in incident_segments:
        segment_array = np.asarray(segment, dtype=np.float64)
        axis.plot(
            segment_array[:, 0],
            segment_array[:, 1],
            color="cyan",
            linewidth=3.0,
            linestyle="--",
            alpha=0.95,
            zorder=5,
        )
    if probe_points.size:
        axis.scatter(
            probe_points[:, 0],
            probe_points[:, 1],
            color="yellow",
            edgecolor="black",
            marker="D",
            s=35,
            zorder=7,
        )

    if not representative:
        for fragment in fragments:
            points = np.asarray(
                fragment.points_global_xy, dtype=np.float64)
            axis.plot(
                points[:, 0],
                points[:, 1],
                color="white" if background_crop is not None else "0.45",
                linewidth=0.7,
                alpha=0.18,
                zorder=2,
            )
    elif support_counts is None:
        for fragment in fragments:
            points = np.asarray(
                fragment.points_global_xy, dtype=np.float64)
            axis.plot(
                points[:, 0],
                points[:, 1],
                color="darkorange",
                linewidth=1.8,
                alpha=0.82,
                zorder=4,
            )
    else:
        maximum_support = max(1, int(np.max(support_counts)))
        color_map = plt.get_cmap("plasma")
        for fragment, support_count in zip(
            fragments, support_counts
        ):
            points = np.asarray(
                fragment.points_global_xy, dtype=np.float64)
            normalized_support = (
                math.log1p(int(support_count))
                / math.log1p(maximum_support)
            )
            color = color_map(normalized_support)
            axis.plot(
                points[:, 0],
                points[:, 1],
                color=color,
                linewidth=1.2 + 2.0 * normalized_support,
                alpha=0.9,
                zorder=4,
            )
            nearest_index = int(np.argmin(np.linalg.norm(
                points
                - np.asarray(center_xy, dtype=np.float64)[None, :],
                axis=1,
            )))
            axis.text(
                points[nearest_index, 0],
                points[nearest_index, 1],
                str(int(support_count)),
                fontsize=5,
                color="black",
                bbox={
                    "facecolor": color,
                    "edgecolor": "none",
                    "alpha": 0.75,
                    "pad": 0.4,
                },
                zorder=8,
            )

    axis.add_patch(Rectangle(
        (center_x - half_window, center_y - half_window),
        window_size,
        window_size,
        fill=False,
        edgecolor="red",
        linewidth=1.5,
        zorder=6,
    ))
    axis.scatter(
        [center_x],
        [center_y],
        color="lime",
        edgecolor="black",
        marker="*",
        s=100,
        zorder=9,
    )
    margin = max(32.0, window_size * 0.25)
    axis.set_xlim(
        center_x - half_window - margin,
        center_x + half_window + margin,
    )
    axis.set_ylim(
        center_y + half_window + margin,
        center_y - half_window - margin,
    )
    axis.set_aspect("equal")
    axis.set_title(title)


def _visualize_comparison(
    output_path: Path,
    fragments,
    nearest: CompressionResult,
    near_diverse: CompressionResult,
    bounded_near_diverse: CompressionResult,
    center_xy: Sequence[float],
    window_size: float,
    incident_segments: Sequence[np.ndarray],
    probe_points: np.ndarray,
    background_image: Optional[Path],
) -> None:
    background_crop = None
    background_extent = None
    if background_image is not None:
        with Image.open(background_image) as image_file:
            image = image_file.convert("RGB")
            background_crop, background_extent = _display_crop(
                image, center_xy, window_size)
    figure, axes = plt.subplots(1, 4, figsize=(25, 6.5))
    _plot_fragments(
        axes[0],
        fragments,
        None,
        False,
        "all candidates (n={})".format(len(fragments)),
        center_xy,
        window_size,
        incident_segments,
        probe_points,
        background_crop,
        background_extent,
    )
    _plot_fragments(
        axes[1],
        nearest.selected_fragments,
        nearest.support_count,
        True,
        "nearest 64",
        center_xy,
        window_size,
        incident_segments,
        probe_points,
        background_crop,
        background_extent,
    )
    _plot_fragments(
        axes[2],
        near_diverse.selected_fragments,
        near_diverse.support_count,
        True,
        "near_diverse 64",
        center_xy,
        window_size,
        incident_segments,
        probe_points,
        background_crop,
        background_extent,
    )
    _plot_fragments(
        axes[3],
        bounded_near_diverse.selected_fragments,
        bounded_near_diverse.support_count,
        True,
        "bounded_near_diverse 64",
        center_xy,
        window_size,
        incident_segments,
        probe_points,
        background_crop,
        background_extent,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def analyze_compression(
    cache_dir: Path,
    graph_path: Path,
    output_dir: Path,
    background_image: Optional[Path],
    budgets: Sequence[int],
    window_size: float,
    context_points: int,
    max_time_gap_seconds: Optional[float],
    max_spatial_gap_pixels: Optional[float],
    near_fraction: float,
    bounded_near_fraction: float,
    prepool_multiplier: int,
    probe_distance: float,
    evidence_distance: float,
    max_nodes: Optional[int],
    visualization_count: int,
) -> Dict[str, Any]:
    store = open_structured_trajectory_store(os.fspath(cache_dir))
    grid_basis = store.meta.get("grid_index_basis")
    if grid_basis != SEGMENT_GRID_INDEX_BASIS:
        raise RuntimeError(
            "Stage 2B requires segment-aware cache basis {!r}; found "
            "{!r}".format(SEGMENT_GRID_INDEX_BASIS, grid_basis))
    graph = graph_helper.read_graph(
        os.fspath(graph_path), merge_duplicates=True)
    nodes_by_type = _graph_nodes(graph)
    nodes = [
        dict(node, node_type=node_type)
        for node_type in NODE_TYPES
        for node in nodes_by_type[node_type]
    ]
    nodes.sort(key=lambda node: int(node["vertex_id"]))
    if max_nodes is not None:
        nodes = nodes[:max_nodes]

    alignment = None
    if background_image is not None:
        with Image.open(background_image) as image_file:
            alignment = verify_background_alignment(
                image_file.convert("RGB"), store.meta)

    node_records: List[Dict[str, Any]] = []
    query_times_ms = []
    analysis_start = time.perf_counter()
    for node_index, node in enumerate(nodes):
        center_xy = node["center_xy"]
        probe_points = _probe_points(
            center_xy,
            node["incident_segments"],
            probe_distance,
        )
        query_start = time.perf_counter()
        fragments = store.query_trajectory_fragments(
            center_xy=center_xy,
            window_size=window_size,
            context_points=context_points,
            max_time_gap_seconds=max_time_gap_seconds,
            max_spatial_gap_pixels=max_spatial_gap_pixels,
        )
        query_times_ms.append(
            (time.perf_counter() - query_start) * 1000.0)

        selections: Dict[str, Dict[str, Dict[str, Any]]] = {
            strategy: {} for strategy in STRATEGIES
        }
        for strategy in STRATEGIES:
            for budget in budgets:
                result = compress_trajectory_fragments(
                    fragments=fragments,
                    center_xy=center_xy,
                    window_size=window_size,
                    max_fragments=budget,
                    strategy=strategy,
                    near_fraction=(
                        bounded_near_fraction
                        if strategy == "bounded_near_diverse"
                        else near_fraction
                    ),
                    prepool_multiplier=prepool_multiplier,
                )
                covered_count, branch_count, covered_flags = (
                    _branch_coverage(
                        result,
                        probe_points,
                        evidence_distance,
                    )
                )
                metrics = _selection_metrics(
                    result,
                    covered_count,
                    branch_count,
                    center_xy,
                    window_size,
                )
                metrics["covered_branches"] = covered_flags
                metrics["selected_track_indices"] = [
                    int(fragment.track_index)
                    for fragment in result.selected_fragments
                ]
                metrics["source_fragment_indices"] = (
                    result.source_fragment_indices.astype(int).tolist())
                selections[strategy][str(budget)] = metrics

        node_records.append({
            "node_type": node["node_type"],
            "vertex_id": int(node["vertex_id"]),
            "degree": int(node["degree"]),
            "center_xy": list(map(float, center_xy)),
            "fragment_count": len(fragments),
            "track_count": len({
                int(fragment.track_index) for fragment in fragments
            }),
            "branch_probe_points": probe_points.tolist(),
            "selections": selections,
        })
        if (
            (node_index + 1) % 25 == 0
            or node_index + 1 == len(nodes)
        ):
            print(
                "analyzed {}/{} GT nodes".format(
                    node_index + 1, len(nodes)),
                flush=True,
            )

    summaries: Dict[str, Dict[str, Any]] = {}
    for strategy in STRATEGIES:
        summaries[strategy] = {}
        for budget in budgets:
            key = str(budget)
            all_results = [
                record["selections"][strategy][key]
                for record in node_records
            ]
            by_node_type = {
                node_type: _summarize_results([
                    record["selections"][strategy][key]
                    for record in node_records
                    if record["node_type"] == node_type
                ])
                for node_type in NODE_TYPES
            }
            summaries[strategy][key] = {
                "overall": _summarize_results(all_results),
                "by_node_type": by_node_type,
            }

    visualization_reports = []
    if visualization_count > 0 and 64 in budgets and node_records:
        ranked = sorted(
            node_records,
            key=lambda record: (
                -(
                    record["selections"]["bounded_near_diverse"]["64"][
                        "covered_branch_count"
                    ]
                    - record["selections"]["nearest"]["64"][
                        "covered_branch_count"
                    ]
                ),
                -record["fragment_count"],
                record["vertex_id"],
            ),
        )
        selected_records = ranked[:visualization_count]
        node_lookup = {
            int(node["vertex_id"]): node for node in nodes
        }
        for record in selected_records:
            node = node_lookup[record["vertex_id"]]
            fragments = store.query_trajectory_fragments(
                center_xy=node["center_xy"],
                window_size=window_size,
                context_points=context_points,
                max_time_gap_seconds=max_time_gap_seconds,
                max_spatial_gap_pixels=max_spatial_gap_pixels,
            )
            nearest = compress_trajectory_fragments(
                fragments,
                node["center_xy"],
                window_size,
                64,
                strategy="nearest",
                near_fraction=near_fraction,
                prepool_multiplier=prepool_multiplier,
            )
            diverse = compress_trajectory_fragments(
                fragments,
                node["center_xy"],
                window_size,
                64,
                strategy="near_diverse",
                near_fraction=near_fraction,
                prepool_multiplier=prepool_multiplier,
            )
            bounded = compress_trajectory_fragments(
                fragments,
                node["center_xy"],
                window_size,
                64,
                strategy="bounded_near_diverse",
                near_fraction=bounded_near_fraction,
                prepool_multiplier=prepool_multiplier,
            )
            output_path = (
                output_dir
                / "visualizations"
                / (
                    "{}_vertex_{:04d}_all_nearest64_"
                    "diverse64_bounded64.png"
                ).format(
                    node["node_type"], node["vertex_id"])
            )
            _visualize_comparison(
                output_path=output_path,
                fragments=fragments,
                nearest=nearest,
                near_diverse=diverse,
                bounded_near_diverse=bounded,
                center_xy=node["center_xy"],
                window_size=window_size,
                incident_segments=node["incident_segments"],
                probe_points=_probe_points(
                    node["center_xy"],
                    node["incident_segments"],
                    probe_distance,
                ),
                background_image=background_image,
            )
            visualization_reports.append({
                "vertex_id": int(node["vertex_id"]),
                "node_type": node["node_type"],
                "fragment_count": len(fragments),
                "output_path": str(output_path.resolve()),
            })

    elapsed_seconds = time.perf_counter() - analysis_start
    report = {
        "stage": "2B",
        "cache_dir": str(cache_dir.resolve()),
        "graph_path": str(graph_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "grid_index_basis": grid_basis,
        "segment_aware_index_active": True,
        "query_uses_full_trajectory_scan": False,
        "graph_vertex_count": len(graph.vertices),
        "graph_directed_edge_count": len(graph.edges),
        "node_population": {
            node_type: len(nodes_by_type[node_type])
            for node_type in NODE_TYPES
        },
        "analyzed_node_count": len(nodes),
        "budgets": list(map(int, budgets)),
        "strategies": list(STRATEGIES),
        "window_size": float(window_size),
        "context_points": int(context_points),
        "max_time_gap_seconds": max_time_gap_seconds,
        "max_spatial_gap_pixels": max_spatial_gap_pixels,
        "near_fraction": float(near_fraction),
        "bounded_near_fraction": float(bounded_near_fraction),
        "prepool_multiplier": int(prepool_multiplier),
        "probe_distance": float(probe_distance),
        "evidence_distance": float(evidence_distance),
        "query_time_ms": _distribution(query_times_ms),
        "elapsed_seconds": float(elapsed_seconds),
        "background_alignment": alignment,
        "summaries": summaries,
        "visualizations": visualization_reports,
        "nodes": node_records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "trajectory_compression_analysis.json"
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
            "Compare nearest and continuous near_diverse trajectory "
            "compression at real GT nodes."
        )
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--background-image", type=Path, default=None)
    parser.add_argument(
        "--budgets",
        type=_positive_int,
        nargs="+",
        default=[32, 64, 128],
    )
    parser.add_argument("--window-size", type=float, default=256.0)
    parser.add_argument("--context-points", type=_nonnegative_int, default=2)
    parser.add_argument("--max-time-gap-seconds", type=float, default=None)
    parser.add_argument("--max-spatial-gap-pixels", type=float, default=None)
    parser.add_argument("--near-fraction", type=float, default=0.25)
    parser.add_argument(
        "--bounded-near-fraction",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--prepool-multiplier",
        type=_positive_int,
        default=8,
    )
    parser.add_argument("--probe-distance", type=float, default=40.0)
    parser.add_argument("--evidence-distance", type=float, default=20.0)
    parser.add_argument("--max-nodes", type=_positive_int, default=None)
    parser.add_argument(
        "--visualization-count",
        type=_nonnegative_int,
        default=6,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    for name in (
        "window_size",
        "probe_distance",
        "evidence_distance",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("{} must be finite and positive".format(name))
    if (
        not math.isfinite(args.near_fraction)
        or not 0.0 <= args.near_fraction <= 1.0
    ):
        raise ValueError("near_fraction must be in [0, 1]")
    if (
        not math.isfinite(args.bounded_near_fraction)
        or not 0.0 <= args.bounded_near_fraction <= 1.0
    ):
        raise ValueError(
            "bounded_near_fraction must be in [0, 1]")
    budgets = sorted(set(args.budgets))
    report = analyze_compression(
        cache_dir=args.cache_dir,
        graph_path=args.graph,
        output_dir=args.output_dir,
        background_image=args.background_image,
        budgets=budgets,
        window_size=args.window_size,
        context_points=args.context_points,
        max_time_gap_seconds=args.max_time_gap_seconds,
        max_spatial_gap_pixels=args.max_spatial_gap_pixels,
        near_fraction=args.near_fraction,
        bounded_near_fraction=args.bounded_near_fraction,
        prepool_multiplier=args.prepool_multiplier,
        probe_distance=args.probe_distance,
        evidence_distance=args.evidence_distance,
        max_nodes=args.max_nodes,
        visualization_count=args.visualization_count,
    )
    coverage = {
        strategy: {
            str(budget): report["summaries"][strategy][str(budget)][
                "overall"
            ]["evidence_coverage_at_k"]
            for budget in budgets
        }
        for strategy in STRATEGIES
    }
    compression_time_ms = {
        strategy: {
            str(budget): report["summaries"][strategy][str(budget)][
                "overall"
            ]["compression_time_ms"]["mean"]
            for budget in budgets
        }
        for strategy in STRATEGIES
    }
    compact = {
        "analyzed_node_count": report["analyzed_node_count"],
        "elapsed_seconds": report["elapsed_seconds"],
        "query_time_ms": report["query_time_ms"],
        "evidence_coverage_at_k": coverage,
        "mean_compression_time_ms": compression_time_ms,
        "report_path": report["report_path"],
        "visualizations": report["visualizations"],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
