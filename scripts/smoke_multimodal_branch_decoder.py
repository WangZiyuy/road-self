"""Real teacher-forced Stage 3B auxiliary-head smoke test.

This script never feeds predicted branch endpoints back to VecRoad.  RPNet is
frozen and supplies only its existing ``feature_maps["stage_fuse"]`` tensor.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from easydict import EasyDict


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.branch_query_decoder import MultiModalBranchQueryDecoder
from model.branch_set_loss import BranchSetCriterion
from model.graph_state_encoder import GraphStateEncoder
from model.model import build_model
from model.trajectory_encoder import TrajectoryFragmentEncoder
from utils.branch_targets import (
    build_branch_target_batch,
    build_immediate_branch_targets,
)
from utils.checkpoint_utils import (
    load_checkpoint_into_model,
    resolve_inference_checkpoint_path,
)
from utils.graph_state import build_graph_state
from utils.model_utils import Path as VecRoadPath
from utils.structured_trajectory_store import (
    open_structured_trajectory_store,
)
from utils.tileloader import Tiles
from utils.trajectory_batch import build_trajectory_batch
from utils.trajectory_compression import compress_trajectory_fragments
from utils.trajectory_mode import TRAJ_MODE_NONE, resolve_trajectory_mode


NODE_TYPES = ("ordinary", "t_junction", "multi_branch")


def _load_config(config_path: Path) -> EasyDict:
    with config_path.open("r", encoding="utf-8") as config_file:
        cfg = EasyDict(yaml.load(
            config_file, Loader=yaml.UnsafeLoader))
    mode = resolve_trajectory_mode(cfg)
    if mode != TRAJ_MODE_NONE:
        raise ValueError(
            "Stage 3B RPNet feature extraction requires TRAJ.MODE=none; "
            "resolved {!r}".format(mode))
    return cfg


def _make_training_paths(cfg: EasyDict) -> List[VecRoadPath]:
    tiles = Tiles(
        training_regions=cfg.TRAIN.TRAINING_REGIONS,
        parallel_tiles=cfg.TRAIN.PARALLEL_TILES,
        region_path=cfg.DIR.ALL_REGION_PATH,
        graph_dir=cfg.DIR.GRAPH_DIR,
        tile_dir=cfg.DIR.TILE_DIR,
        traj_dir=None,
    )
    return [
        VecRoadPath(
            idx=index,
            training=True,
            gc=subtile["gc"].clone(),
            tile_data=subtile,
            all_trajectories=[],
            all_pixel_trajectories=[],
        )
        for index, subtile in enumerate(tiles.prepare_training())
    ]


def _gt_degree(path, search_state) -> Optional[int]:
    vertex = search_state.vertex
    if vertex.edge_pos is None:
        return None
    edge = vertex.edge_pos.edge(path.gc.graph)
    gt_vertex = None
    if edge.src(path.gc.graph).point == vertex.point:
        gt_vertex = edge.src(path.gc.graph)
    elif edge.dst(path.gc.graph).point == vertex.point:
        gt_vertex = edge.dst(path.gc.graph)
    if gt_vertex is None:
        return 2
    return len(set(gt_vertex.neighbors(path.gc.graph)))


def _node_type(degree: Optional[int]) -> Optional[str]:
    if degree == 2:
        return "ordinary"
    if degree == 3:
        return "t_junction"
    if degree is not None and degree >= 4:
        return "multi_branch"
    return None


def _eligible_case(node_type: Optional[str], branch_count: int) -> bool:
    return (
        (node_type == "ordinary" and branch_count == 1)
        or (node_type == "t_junction" and branch_count >= 2)
        or (node_type == "multi_branch" and branch_count >= 3)
    )


def _collect_teacher_forced_cases(
    cfg: EasyDict,
    max_attempts: int,
    max_explored_edges: int,
) -> List[Dict[str, Any]]:
    paths = _make_training_paths(cfg)
    active_path_indices = list(range(len(paths)))
    path_cursor = 0
    attempts = 0
    selected = {}

    while (
            len(selected) < len(NODE_TYPES)
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
        branch_targets = build_immediate_branch_targets(
            target_poses=target_poses,
            current_vertex=state.vertex,
            graph=path.gc.graph,
            window_size=cfg.TRAIN.WINDOW_SIZE,
        )
        degree = _gt_degree(path, state)
        node_type = _node_type(degree)
        if (
                node_type not in selected
                and _eligible_case(
                    node_type, branch_targets.branch_count)):
            selected[node_type] = {
                "node_type": node_type,
                "path_index": int(path_index),
                "vertex_id": int(state.vertex.id),
                "center_xy": (
                    float(state.vertex.point.x),
                    float(state.vertex.point.y),
                ),
                "gt_degree": int(degree),
                "aerial_image": torch.from_numpy(
                    np.asarray(
                        local_input["aerial_image_chw"],
                        dtype=np.float32,
                    ).copy()
                ),
                "walked_path": torch.from_numpy(
                    np.asarray(
                        local_input["walked_path_small"],
                        dtype=np.float32,
                    ).copy()
                ),
                "graph_state": build_graph_state(
                    path,
                    state,
                    max_explored_edges=max_explored_edges,
                ),
                "branch_targets": branch_targets,
            }

        if state.vertex.edge_pos is None or len(target_poses) == 0:
            continue
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

    missing = [node_type for node_type in NODE_TYPES
               if node_type not in selected]
    if missing:
        raise RuntimeError(
            "could not find eligible teacher-forced cases for {} after "
            "{} attempts".format(", ".join(missing), attempts))
    cases = [selected[node_type] for node_type in NODE_TYPES]
    for case in cases:
        case["selection_attempt_count"] = attempts
    return cases


def _move_tensor_dict(
    values: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    return {
        key: (
            value.to(device=device)
            if torch.is_tensor(value)
            else value
        )
        for key, value in values.items()
    }


def _stack_graph_states(
    cases: Sequence[Dict[str, Any]],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    keys = cases[0]["graph_state"].keys()
    return {
        key: torch.stack([
            case["graph_state"][key] for case in cases
        ]).to(device=device)
        for key in keys
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _checkpoint_status(
    cfg: EasyDict,
    checkpoint_argument: Optional[Path],
) -> Tuple[Optional[Path], str]:
    if checkpoint_argument is not None:
        checkpoint = checkpoint_argument.resolve(strict=False)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                "requested checkpoint does not exist: {}".format(
                    checkpoint))
        return checkpoint, "explicit_image_only_checkpoint"

    expected = resolve_inference_checkpoint_path(
        cfg, require_exists=False)
    if expected.is_file():
        return expected, "configured_image_only_checkpoint"
    return None, (
        "random_rpnet_interface_smoke; configured checkpoint was absent: "
        "{}".format(expected)
    )


def run_smoke(args: argparse.Namespace) -> Dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    cfg = _load_config(args.config)
    selection_start = time.perf_counter()
    cases = _collect_teacher_forced_cases(
        cfg=cfg,
        max_attempts=args.max_attempts,
        max_explored_edges=args.max_explored_edges,
    )
    selection_ms = (time.perf_counter() - selection_start) * 1000.0

    store = open_structured_trajectory_store(str(args.cache_dir))
    compressed_sets = []
    case_reports = []
    for case in cases:
        query_start = time.perf_counter()
        fragments = store.query_trajectory_fragments(
            center_xy=case["center_xy"],
            window_size=cfg.TRAIN.WINDOW_SIZE,
            context_points=args.context_points,
            max_time_gap_seconds=args.max_time_gap_seconds,
            max_spatial_gap_pixels=args.max_spatial_gap_pixels,
        )
        query_ms = (time.perf_counter() - query_start) * 1000.0
        compression = compress_trajectory_fragments(
            fragments=fragments,
            center_xy=case["center_xy"],
            window_size=cfg.TRAIN.WINDOW_SIZE,
            max_fragments=64,
            strategy="bounded_near_diverse",
            prepool_multiplier=8,
            near_fraction=0.5,
        )
        compressed_sets.append(compression)
        case_reports.append({
            "node_type": case["node_type"],
            "vertex_id": case["vertex_id"],
            "center_xy": list(case["center_xy"]),
            "gt_degree": case["gt_degree"],
            "branch_count": int(
                case["branch_targets"].branch_count),
            "trajectory_fragment_count": len(fragments),
            "trajectory_fragment_kept_count": int(
                compression.kept_fragment_count),
            "trajectory_query_ms": float(query_ms),
            "trajectory_compression_ms": float(
                compression.compression_timing_ms["total"]),
        })

    centers = [case["center_xy"] for case in cases]
    trajectory_batch_cpu = build_trajectory_batch(
        compressed_sets,
        center_xy=centers,
        window_size=cfg.TRAIN.WINDOW_SIZE,
        max_fragments=None,
    )
    trajectory_batch = _move_tensor_dict(
        trajectory_batch_cpu, device)
    graph_state = _stack_graph_states(cases, device)
    branch_target_batch = _move_tensor_dict(
        build_branch_target_batch([
            case["branch_targets"] for case in cases
        ]),
        device,
    )

    checkpoint, rpnet_source = _checkpoint_status(
        cfg, args.checkpoint)
    rpnet = build_model(
        num_targets=cfg.TRAIN.NUM_TARGETS,
        backbone_pretrained=False,
        enable_trajectory_modules=False,
    )
    checkpoint_metadata = None
    if checkpoint is not None:
        payload = load_checkpoint_into_model(
            rpnet,
            checkpoint,
            map_location="cpu",
            strict=True,
        )
        checkpoint_metadata = {
            key: payload.get(key)
            for key in (
                "outer_it",
                "path_it",
                "trajectory_mode",
                "model_name",
                "num_targets",
                "step_length",
                "window_size",
            )
        }
    rpnet.to(device=device).eval()
    rpnet.requires_grad_(False)

    aerial_images = torch.stack([
        case["aerial_image"] for case in cases
    ]).to(device=device)
    walked_paths = torch.stack([
        case["walked_path"] for case in cases
    ]).to(device=device)
    _synchronize(device)
    rpnet_start = time.perf_counter()
    stage_fuse_parts = []
    with torch.no_grad():
        for aerial_image, walked_path in zip(
                aerial_images, walked_paths):
            rpnet_output = rpnet(
                aerial_image.unsqueeze(0),
                None,
                None,
                None,
                None,
                walked_path.unsqueeze(0),
                NUM_TARGETS=None,
                test=False,
                model="origin",
                use_traj=False,
            )
            stage_fuse_parts.append(
                rpnet_output["feature_maps"]["stage_fuse"])
    stage_fuse = torch.cat(stage_fuse_parts, dim=0)
    _synchronize(device)
    rpnet_forward_ms = (
        time.perf_counter() - rpnet_start) * 1000.0

    trajectory_encoder = TrajectoryFragmentEncoder(
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
    ).to(device=device).train()
    graph_encoder = GraphStateEncoder(
        hidden_dim=128).to(device=device).train()
    decoder = MultiModalBranchQueryDecoder(
        image_channels=128,
        trajectory_dim=128,
        hidden_dim=128,
        num_queries=6,
        num_heads=4,
        image_pool_size=16,
        dropout=0.1,
    ).to(device=device).train()
    criterion = BranchSetCriterion()

    for module in (trajectory_encoder, graph_encoder, decoder):
        module.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    auxiliary_forward_start = time.perf_counter()
    trajectory_output = trajectory_encoder(trajectory_batch)
    state_token = graph_encoder(graph_state)
    branch_output = decoder(
        stage_fuse=stage_fuse,
        state_token=state_token,
        fragment_tokens=trajectory_output["fragment_tokens"],
        fragment_mask=trajectory_output["fragment_mask"],
        return_attention=True,
    )
    losses = criterion(branch_output, branch_target_batch)
    _synchronize(device)
    auxiliary_forward_ms = (
        time.perf_counter() - auxiliary_forward_start) * 1000.0

    backward_start = time.perf_counter()
    losses["loss"].backward()
    _synchronize(device)
    backward_ms = (time.perf_counter() - backward_start) * 1000.0

    named_auxiliary_parameters = []
    for module_name, module in (
            ("trajectory_encoder", trajectory_encoder),
            ("graph_state_encoder", graph_encoder),
            ("branch_decoder", decoder)):
        named_auxiliary_parameters.extend([
            ("{}.{}".format(module_name, name), parameter)
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        ])
    missing_gradients = [
        name for name, parameter in named_auxiliary_parameters
        if parameter.grad is None
    ]
    nonfinite_gradients = [
        name for name, parameter in named_auxiliary_parameters
        if parameter.grad is not None
        and not bool(torch.isfinite(parameter.grad).all())
    ]

    decoder.eval()
    with torch.no_grad():
        no_trajectory_output = decoder(
            stage_fuse=stage_fuse,
            state_token=state_token.detach(),
            fragment_tokens=stage_fuse.new_zeros(
                (len(cases), 0, 128)),
            fragment_mask=torch.zeros(
                (len(cases), 0),
                dtype=torch.bool,
                device=device,
            ),
        )
        no_trajectory_losses = criterion(
            no_trajectory_output, branch_target_batch)
    no_trajectory_finite = all(
        bool(torch.isfinite(value).all())
        for value in no_trajectory_output.values()
        if torch.is_tensor(value)
    ) and all(
        bool(torch.isfinite(no_trajectory_losses[key]).all())
        for key in (
            "loss",
            "existence_loss",
            "endpoint_loss",
            "direction_loss",
        )
    )

    peak_cuda_bytes = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None
    )
    report = {
        "stage": "3B",
        "purpose": "auxiliary_interface_smoke_only",
        "branch_predictions_feed_path_push": False,
        "num_targets_unchanged": int(cfg.TRAIN.NUM_TARGETS),
        "trajectory_compression": {
            "strategy": "bounded_near_diverse",
            "max_fragments": 64,
            "prepool_multiplier": 8,
            "near_fraction": 0.5,
            "support_count_used": False,
        },
        "device": str(device),
        "seed": int(args.seed),
        "case_selection_ms": float(selection_ms),
        "cases": case_reports,
        "rpnet": {
            "frozen": True,
            "source": rpnet_source,
            "checkpoint_path": (
                str(checkpoint) if checkpoint is not None else None),
            "checkpoint_metadata": checkpoint_metadata,
            "forward_ms": float(rpnet_forward_ms),
        },
        "shapes": {
            "stage_fuse": list(stage_fuse.shape),
            "fragment_tokens": list(
                trajectory_output["fragment_tokens"].shape),
            "fragment_mask": list(
                trajectory_output["fragment_mask"].shape),
            "state_token": list(state_token.shape),
            "branch_exist_logits": list(
                branch_output["branch_exist_logits"].shape),
            "branch_offsets_norm": list(
                branch_output["branch_offsets_norm"].shape),
            "branch_directions": list(
                branch_output["branch_directions"].shape),
        },
        "losses": {
            "total": float(losses["loss"].detach().cpu()),
            "existence": float(
                losses["existence_loss"].detach().cpu()),
            "endpoint": float(
                losses["endpoint_loss"].detach().cpu()),
            "direction": float(
                losses["direction_loss"].detach().cpu()),
            "matched_count": int(
                losses["matched_count"].detach().cpu()),
        },
        "timing_ms": {
            "auxiliary_forward": float(auxiliary_forward_ms),
            "auxiliary_backward": float(backward_ms),
        },
        "gradients": {
            "all_auxiliary_parameters_have_gradients": (
                not missing_gradients),
            "all_existing_gradients_finite": (
                not nonfinite_gradients),
            "missing": missing_gradients,
            "nonfinite": nonfinite_gradients,
            "rpnet_has_gradients": any(
                parameter.grad is not None
                for parameter in rpnet.parameters()),
        },
        "all_outputs_finite": all(
            bool(torch.isfinite(value).all())
            for value in branch_output.values()
            if torch.is_tensor(value)
        ),
        "no_trajectory_smoke": {
            "finite": bool(no_trajectory_finite),
            "total_loss": float(
                no_trajectory_losses["loss"].detach().cpu()),
        },
        "peak_cuda_memory_bytes": peak_cuda_bytes,
    }
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Stage 3B multimodal auxiliary branch-head smoke test."))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline_image_only.yml"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data_self/input/traj_structured/xian/v1"),
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data_self/output/stage3b_smoke/"
            "stage3b_multimodal_branch_smoke.json"),
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--max-attempts", type=int, default=4096)
    parser.add_argument("--max-explored-edges", type=int, default=8)
    parser.add_argument("--context-points", type=int, default=2)
    parser.add_argument(
        "--max-time-gap-seconds", type=float, default=None)
    parser.add_argument(
        "--max-spatial-gap-pixels", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    if args.max_explored_edges < 0:
        raise ValueError("max_explored_edges must be non-negative")
    if args.context_points < 0:
        raise ValueError("context_points must be non-negative")

    report = run_smoke(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report["report_path"] = str(args.output.resolve())
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(
            report,
            output_file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        output_file.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
