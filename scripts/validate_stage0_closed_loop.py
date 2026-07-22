"""Run a bounded real-data closed-loop equivalence check for Stage 0.

This validator keeps two independent ``Path`` states.  The legacy-disabled
call supplies trajectory-shaped tensors while ``use_traj=False``; the Stage-0
call supplies ``None`` for every trajectory argument.  Both paths use the same
checkpoint, aerial tile, fixed start point, and VecRoad state machine.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from easydict import EasyDict


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib import geom, graph as graph_helper  # noqa: E402
from model.model import RPNet  # noqa: E402
from scripts.validate_stage0_baseline import (  # noqa: E402
    canonical_graph_signature,
    id_based_graph_signature,
)
from utils import model_utils, tileloader  # noqa: E402
from utils.checkpoint_utils import (  # noqa: E402
    load_checkpoint_into_model,
    resolve_inference_checkpoint_path,
)
from utils.regions import get_regions  # noqa: E402
from utils.trajectory_mode import (  # noqa: E402
    TRAJ_MODE_NONE,
    resolve_trajectory_mode,
    validate_trajectory_model_compatibility,
)


OUTPUT_KEYS = ("road", "junc", "anchor", "anchor_lowrs")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate bounded legacy-disabled/Stage-0 graph growth."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "baseline_image_only.yml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint override; defaults to TEST.CKPT_FILE.",
    )
    parser.add_argument("--region", default=None)
    parser.add_argument("--start-x", type=int, default=None)
    parser.add_argument("--start-y", type=int, default=None)
    parser.add_argument(
        "--start-state",
        choices=("key_point", "non_key_point"),
        default="key_point",
        help="Initialize the fixed point as a junction or queued road vertex.",
    )
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--coordinate-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data_self" / "baseline_image_only" / "closed_loop",
    )
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def _load_config(path: Path) -> EasyDict:
    with path.open("r", encoding="utf-8") as config_file:
        cfg = EasyDict(yaml.load(config_file, Loader=yaml.UnsafeLoader))
    mode = resolve_trajectory_mode(cfg)
    validate_trajectory_model_compatibility(cfg, mode)
    if mode != TRAJ_MODE_NONE:
        raise ValueError("closed-loop Stage-0 validation requires TRAJ.MODE=none")
    return cfg


def _select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda was requested, but CUDA is unavailable")
    return torch.device(name)


def _select_start_point(
    cfg: EasyDict,
    region_name: str,
    search_rect: geom.Rectangle,
    start_x: int | None,
    start_y: int | None,
) -> tuple[geom.Point, dict[str, Any]]:
    if (start_x is None) != (start_y is None):
        raise ValueError("--start-x and --start-y must be supplied together")
    safe_rect = search_rect.add_tol(-int(cfg.TEST.WINDOW_SIZE) // 2)
    if start_x is not None and start_y is not None:
        point = geom.Point(start_x, start_y)
        if not safe_rect.contains(point):
            raise ValueError("the requested start point is outside the safe tile area")
        return point, {"source": "explicit", "vertex_id": None, "degree": None}

    graph_path = Path(cfg.DIR.GRAPH_DIR) / "{}.graph".format(region_name)
    graph = graph_helper.read_graph(os.fspath(graph_path), merge_duplicates=False)
    candidates = []
    for vertex in graph.vertices.values():
        degree = max(len(vertex.in_edges_id), len(vertex.out_edges_id))
        if degree >= 3 and safe_rect.contains(vertex.point):
            candidates.append((
                -degree,
                int(vertex.point.x),
                int(vertex.point.y),
                int(vertex.id),
                vertex,
            ))
    if not candidates:
        raise RuntimeError("no safe junction start point exists in the requested tile")
    candidates.sort(key=lambda item: item[:-1])
    selected = candidates[0][-1]
    return selected.point, {
        "source": "highest_degree_gt_junction",
        "vertex_id": int(selected.id),
        "degree": max(len(selected.in_edges_id), len(selected.out_edges_id)),
    }


def _tile_data(
    region_name: str,
    search_rect: geom.Rectangle,
    cache: tileloader.TileCache,
    start_point: geom.Point,
    *,
    include_starting_location: bool,
) -> dict[str, Any]:
    junctions = []
    if include_starting_location:
        junctions = [[{
            "point": start_point,
            "edge_pos": None,
            "key_point": True,
        }]]
    return {
        "region": region_name,
        "search_rect": search_rect,
        "cache": cache,
        "starting_locations": {
            "junction": junctions,
            "middle": [],
        },
        "gc": None,
    }


def _point_tuple(point: Any) -> tuple[float, float]:
    return float(point.x), float(point.y)


def _coordinate_lists(points: list[list[Any]]) -> list[list[tuple[float, float]]]:
    return [[_point_tuple(point) for point in sample] for sample in points]


def _model_outputs_cpu(
    model: RPNet,
    aerial: torch.Tensor,
    walked: torch.Tensor,
    *,
    legacy_disabled: bool,
) -> dict[str, torch.Tensor]:
    batch_size, _, height, width = aerial.shape
    if legacy_disabled:
        traj_image = torch.ones(
            batch_size, 1, height, width, device=aerial.device
        )
        aerial_traj = torch.ones(
            batch_size, 4, height, width, device=aerial.device
        )
        tracks = torch.ones(batch_size, 2, 3, 2, device=aerial.device)
        mask = torch.ones(
            batch_size, 2, 3, dtype=torch.bool, device=aerial.device
        )
    else:
        traj_image = aerial_traj = tracks = mask = None

    output = model(
        aerial_image=aerial,
        traj_image=traj_image,
        aerial_traj_image=aerial_traj,
        neighborhood_trajectory_norm=tracks,
        valid_mask=mask,
        walked_path=walked,
        NUM_TARGETS=model.num_targets,
        test=False,
        model="origin",
        use_traj=False,
    )
    result = {key: output[key].detach().cpu().clone() for key in OUTPUT_KEYS}
    del output
    return result


def _road_probability(road_logits: torch.Tensor, window_size: int) -> np.ndarray:
    if road_logits.shape[-1] != window_size:
        road_logits = F.interpolate(
            road_logits,
            size=(window_size, window_size),
            mode="bilinear",
            align_corners=False,
        )
    return torch.sigmoid(road_logits).numpy()


def _run(args: argparse.Namespace, cfg: EasyDict) -> dict[str, Any]:
    if args.max_iterations <= 0:
        raise ValueError("--max-iterations must be positive")
    if args.tolerance < 0 or args.coordinate_tolerance <= 0:
        raise ValueError("tolerances must be non-negative, with coordinate tolerance > 0")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _select_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    region_name = args.region or str(cfg.TEST.SINGLE_REGION)
    regions = get_regions(cfg.DIR.TEST_REGION_PATH)
    if region_name not in regions:
        raise KeyError("region {!r} is missing from TEST_REGION_PATH".format(region_name))
    region = regions[region_name]
    tile_size = int(cfg.TRAIN.IMG_SZ)
    window_size = int(cfg.TEST.WINDOW_SIZE)
    tile_start = geom.Point(region.radius_x, region.radius_y).scale(tile_size)
    search_rect = geom.Rectangle(
        tile_start, tile_start.add(geom.Point(tile_size, tile_size))
    )
    start_point, start_metadata = _select_start_point(
        cfg, region_name, search_rect, args.start_x, args.start_y
    )

    forbidden_traj_dir = args.output_dir / "trajectory_must_not_be_read"
    cache = tileloader.TileCache(
        tile_dir=cfg.DIR.IMAGERY_DIR,
        traj_dir=os.fspath(forbidden_traj_dir),
        tile_size=tile_size,
        window_size=window_size,
        limit=int(cfg.TRAIN.PARALLEL_TILES),
    )
    legacy_path = model_utils.Path(
        0,
        training=False,
        gc=None,
        tile_data=_tile_data(
            region_name,
            search_rect,
            cache,
            start_point,
            include_starting_location=args.start_state == "key_point",
        ),
        all_trajectories=[],
        all_pixel_trajectories=[],
        graph=None,
        road_seg=None,
        WINDOW_SIZE=window_size,
    )
    stage0_path = model_utils.Path(
        0,
        training=False,
        gc=None,
        tile_data=_tile_data(
            region_name,
            search_rect,
            cache,
            start_point,
            include_starting_location=args.start_state == "key_point",
        ),
        all_trajectories=None,
        all_pixel_trajectories=None,
        graph=None,
        road_seg=None,
        WINDOW_SIZE=window_size,
    )
    if args.start_state == "non_key_point":
        legacy_vertex = legacy_path.graph.add_vertex(start_point)
        stage0_vertex = stage0_path.graph.add_vertex(start_point)
        legacy_path.prepend_search_vertex(legacy_vertex, is_key_point=False)
        stage0_path.prepend_search_vertex(stage0_vertex, is_key_point=False)

    checkpoint = (
        args.checkpoint.resolve()
        if args.checkpoint is not None
        else resolve_inference_checkpoint_path(cfg, require_exists=True)
    )
    if not checkpoint.is_file():
        raise FileNotFoundError("checkpoint not found: {}".format(checkpoint))
    model = RPNet(
        num_targets=int(cfg.TEST.NUM_TARGETS),
        backbone_pretrained=False,
    )
    payload = load_checkpoint_into_model(
        model, checkpoint, map_location="cpu", strict=True
    )
    model = model.to(device).eval()

    trace = []
    stopped_reason = "max_iterations"
    with torch.no_grad():
        for iteration in range(args.max_iterations):
            legacy_extension, legacy_key = legacy_path.pop(follow_order=True)
            stage0_extension, stage0_key = stage0_path.pop(follow_order=True)
            if (legacy_extension is None) != (stage0_extension is None):
                raise AssertionError("the two Path.pop states diverged")
            if legacy_extension is None:
                stopped_reason = "path_exhausted"
                break
            if (
                _point_tuple(legacy_extension.point)
                != _point_tuple(stage0_extension.point)
                or bool(legacy_key) != bool(stage0_key)
            ):
                raise AssertionError("Path.pop returned different extension states")

            fetch_list = ["aerial_image_chw", "walked_path_small"]
            legacy_input = legacy_path.make_path_input(
                extension_vertex=legacy_extension,
                fetch_list=fetch_list,
                traj_filter=False,
                is_key_point=legacy_key,
                WINDOW_SIZE=window_size,
            )
            stage0_input = stage0_path.make_path_input(
                extension_vertex=stage0_extension,
                fetch_list=fetch_list,
                traj_filter=False,
                is_key_point=stage0_key,
                WINDOW_SIZE=window_size,
            )
            if not np.array_equal(
                legacy_input["aerial_image_chw"], stage0_input["aerial_image_chw"]
            ) or not np.array_equal(
                legacy_input["walked_path_small"],
                stage0_input["walked_path_small"]
            ):
                raise AssertionError("the two local image/graph-state inputs diverged")

            legacy_aerial = torch.from_numpy(
                legacy_input["aerial_image_chw"]
            ).unsqueeze(0).float().to(device)
            legacy_walked = torch.from_numpy(
                legacy_input["walked_path_small"]
            ).unsqueeze(0).float().to(device)
            stage0_aerial = torch.from_numpy(
                stage0_input["aerial_image_chw"]
            ).unsqueeze(0).float().to(device)
            stage0_walked = torch.from_numpy(
                stage0_input["walked_path_small"]
            ).unsqueeze(0).float().to(device)

            legacy_output = _model_outputs_cpu(
                model, legacy_aerial, legacy_walked, legacy_disabled=True
            )
            stage0_output = _model_outputs_cpu(
                model, stage0_aerial, stage0_walked, legacy_disabled=False
            )
            output_differences = {}
            output_statistics = {}
            for key in OUTPUT_KEYS:
                difference = (legacy_output[key] - stage0_output[key]).abs()
                maximum = float(difference.max())
                finite = bool(
                    torch.isfinite(legacy_output[key]).all()
                    and torch.isfinite(stage0_output[key]).all()
                )
                if not finite or maximum > args.tolerance:
                    raise AssertionError(
                        "iteration {} output {} diverged: {}".format(
                            iteration, key, maximum
                        )
                    )
                output_differences[key] = maximum
                output_statistics[key] = {
                    "logit_min": float(legacy_output[key].min()),
                    "logit_max": float(legacy_output[key].max()),
                    "logit_mean": float(legacy_output[key].mean()),
                }

            legacy_key_array = np.asarray([legacy_key], dtype=np.bool_)
            stage0_key_array = np.asarray([stage0_key], dtype=np.bool_)
            legacy_anchor_probability = torch.sigmoid(
                legacy_output["anchor"]
            ).numpy()
            stage0_anchor_probability = torch.sigmoid(
                stage0_output["anchor"]
            ).numpy()
            threshold_diagnostics = {}
            diagnostic_thresholds = sorted({
                float(cfg.TEST.BINARIZE_MAP.ROAD_SEG_THRESHOLE),
                0.05,
                0.1,
                0.2,
                0.5,
            })
            for threshold in diagnostic_thresholds:
                diagnostic_key_array = np.asarray([legacy_key], dtype=np.bool_)
                diagnostic_points = model_utils.map_to_coordinate(
                    legacy_anchor_probability.copy(),
                    diagnostic_key_array,
                    [legacy_extension],
                    ROAD_SEG_THRESHOLE=threshold,
                    STEP_LENGTH=cfg.TEST.STEP_LENGTH,
                    JUNC_MAX_REGION_AREA=cfg.TEST.BINARIZE_MAP.JUNC_MAX_REGION_AREA,
                )
                threshold_diagnostics[str(threshold)] = _coordinate_lists(
                    diagnostic_points
                )[0]
            legacy_points = model_utils.map_to_coordinate(
                legacy_anchor_probability.copy(),
                legacy_key_array,
                [legacy_extension],
                ROAD_SEG_THRESHOLE=cfg.TEST.BINARIZE_MAP.ROAD_SEG_THRESHOLE,
                STEP_LENGTH=cfg.TEST.STEP_LENGTH,
                JUNC_MAX_REGION_AREA=cfg.TEST.BINARIZE_MAP.JUNC_MAX_REGION_AREA,
            )
            stage0_points = model_utils.map_to_coordinate(
                stage0_anchor_probability.copy(),
                stage0_key_array,
                [stage0_extension],
                ROAD_SEG_THRESHOLE=cfg.TEST.BINARIZE_MAP.ROAD_SEG_THRESHOLE,
                STEP_LENGTH=cfg.TEST.STEP_LENGTH,
                JUNC_MAX_REGION_AREA=cfg.TEST.BINARIZE_MAP.JUNC_MAX_REGION_AREA,
            )
            legacy_coordinates = _coordinate_lists(legacy_points)
            stage0_coordinates = _coordinate_lists(stage0_points)
            if legacy_coordinates != stage0_coordinates:
                raise AssertionError("map_to_coordinate outputs diverged")

            legacy_road = _road_probability(
                legacy_output["road"], window_size
            )[0, 0]
            stage0_road = _road_probability(
                stage0_output["road"], window_size
            )[0, 0]
            legacy_path.push(
                extension_vertex=legacy_extension,
                is_key_point=bool(legacy_key_array[0]),
                follow_mode=cfg.TEST.FOLLOW_MODE,
                target_poses=None,
                output_points=legacy_points[0],
                RECT_RADIUS=cfg.TEST.RECT_RADIUS,
                road_segmentation=legacy_road,
                NUM_TARGETS=cfg.TEST.NUM_TARGETS,
                WINDOW_SIZE=window_size,
                STEP_LENGTH=cfg.TEST.STEP_LENGTH,
                AVG_CONFIDENCE_THRESHOLD=cfg.TEST.AVG_CONFIDENCE_THRESHOLD,
            )
            stage0_path.push(
                extension_vertex=stage0_extension,
                is_key_point=bool(stage0_key_array[0]),
                follow_mode=cfg.TEST.FOLLOW_MODE,
                target_poses=None,
                output_points=stage0_points[0],
                RECT_RADIUS=cfg.TEST.RECT_RADIUS,
                road_segmentation=stage0_road,
                NUM_TARGETS=cfg.TEST.NUM_TARGETS,
                WINDOW_SIZE=window_size,
                STEP_LENGTH=cfg.TEST.STEP_LENGTH,
                AVG_CONFIDENCE_THRESHOLD=cfg.TEST.AVG_CONFIDENCE_THRESHOLD,
            )
            legacy_counts = {
                "vertices": len(legacy_path.graph.vertices),
                "directed_edges": len(legacy_path.graph.edges),
            }
            stage0_counts = {
                "vertices": len(stage0_path.graph.vertices),
                "directed_edges": len(stage0_path.graph.edges),
            }
            if legacy_counts != stage0_counts:
                raise AssertionError("graph counts diverged after Path.push")
            trace.append({
                "iteration": iteration,
                "extension": _point_tuple(legacy_extension.point),
                "is_key_point": bool(legacy_key_array[0]),
                "output_max_abs_diff": output_differences,
                "output_statistics": output_statistics,
                "anchor_probability": {
                    "min": float(legacy_anchor_probability.min()),
                    "max": float(legacy_anchor_probability.max()),
                    "mean": float(legacy_anchor_probability.mean()),
                    "per_channel_max": [
                        float(legacy_anchor_probability[0, channel].max())
                        for channel in range(legacy_anchor_probability.shape[1])
                    ],
                },
                "threshold_diagnostics": threshold_diagnostics,
                "map_to_coordinate": legacy_coordinates[0],
                "graph_after_push": legacy_counts,
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    legacy_graph_path = args.output_dir / "legacy_use_traj_false.graph"
    stage0_graph_path = args.output_dir / "stage0_traj_mode_none.graph"
    legacy_path.graph.save(os.fspath(legacy_graph_path), clear_self=False)
    stage0_path.graph.save(os.fspath(stage0_graph_path), clear_self=False)
    legacy_signature = canonical_graph_signature(
        legacy_path.graph, coordinate_tolerance=args.coordinate_tolerance
    )
    stage0_signature = canonical_graph_signature(
        stage0_path.graph, coordinate_tolerance=args.coordinate_tolerance
    )
    canonical_equal = legacy_signature == stage0_signature
    id_equal = (
        id_based_graph_signature(legacy_path.graph)
        == id_based_graph_signature(stage0_path.graph)
    )
    if not canonical_equal:
        raise AssertionError("final canonical graph signatures diverged")
    if forbidden_traj_dir.exists():
        raise AssertionError("the forbidden trajectory path was accessed or created")

    return {
        "passed": True,
        "config": os.fspath(args.config.resolve()),
        "checkpoint": os.fspath(checkpoint),
        "checkpoint_metadata": {
            key: payload.get(key)
            for key in (
                "outer_it",
                "path_it",
                "trajectory_mode",
                "random_seed",
                "model_name",
                "num_targets",
                "step_length",
                "window_size",
            )
        },
        "device": str(device),
        "seed": args.seed,
        "region": region_name,
        "start_point": _point_tuple(start_point),
        "start_state": args.start_state,
        "start_metadata": start_metadata,
        "requested_max_iterations": args.max_iterations,
        "completed_model_iterations": len(trace),
        "stopped_reason": stopped_reason,
        "trajectory_dependency": {
            "mode": "none",
            "forbidden_path": os.fspath(forbidden_traj_dir),
            "forbidden_path_exists": False,
        },
        "trace": trace,
        "graphs": {
            "legacy": os.fspath(legacy_graph_path),
            "stage0": os.fspath(stage0_graph_path),
            "canonical_equal": canonical_equal,
            "id_based_equal": id_equal,
            "vertex_count": legacy_signature["vertex_count"],
            "directed_edge_count": legacy_signature["directed_edge_count"],
            "undirected_edge_count": legacy_signature["undirected_edge_count"],
        },
    }


def main() -> int:
    args = _parse_args()
    cfg = _load_config(args.config.resolve())
    report = _run(args, cfg)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    json_output = args.json_output or args.output_dir / "report.json"
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
