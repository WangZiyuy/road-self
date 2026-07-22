"""Diagnose the road_self anchor regression against original VecRoad.

This is an analysis-only entry point.  It does not change the production
``RPNet.forward`` implementation.  Instead, it evaluates the same real local
window with both the current road_self computation and an in-script replay of
the original VecRoad image-only computation.  Either a road_self checkpoint
or an original VecRoad checkpoint can be loaded by matching state-dict keys.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from easydict import EasyDict
from skimage import measure


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.model import RPNet  # noqa: E402
from scripts.validate_stage0_closed_loop import (  # noqa: E402
    _select_start_point,
    _tile_data,
)
from utils import model_utils, tileloader  # noqa: E402
from utils.regions import get_regions  # noqa: E402


OUTPUT_KEYS = ("road", "junc", "anchor", "anchor_lowrs")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare current and original VecRoad anchor computations."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "baseline_image_only.yml",
    )
    parser.add_argument("--current-checkpoint", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--region", default=None)
    parser.add_argument("--start-x", type=int, default=None)
    parser.add_argument("--start-y", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=(0.1, 0.2, 0.3, 0.5)
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data_self" / "baseline_image_only" / "anchor_regression.json",
    )
    return parser.parse_args()


def _load_config(path: Path) -> EasyDict:
    with path.open("r", encoding="utf-8") as handle:
        return EasyDict(yaml.load(handle, Loader=yaml.UnsafeLoader))


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def _upsample(value: torch.Tensor, scale: float) -> torch.Tensor:
    return F.interpolate(
        value, scale_factor=scale, mode="bilinear", align_corners=True
    )


def forward_original_vecroad(
    net: RPNet,
    aerial_image: torch.Tensor,
    walked_path_small: torch.Tensor,
    num_targets: int,
) -> Dict[str, torch.Tensor]:
    """Replay the original VecRoad image-only forward with road_self modules."""
    stage_1 = net.stage_1(aerial_image)
    stage_1_down = net.maxpool(stage_1)

    stage_2 = net.stage_2(stage_1_down)
    stage_2_side = net.conv_2_side(stage_2)

    stage_3 = net.stage_3(stage_2)
    stage_3_side = _upsample(net.conv_3_side(stage_3), 2)

    stage_4 = net.stage_4(stage_3)
    stage_4_side = _upsample(net.conv_4_side(stage_4), 2)

    stage_5 = net.stage_5(stage_4)
    stage_5_side = _upsample(net.conv_5_side(stage_5), 2)

    stage_fuse = net.conv_fuse(torch.cat(
        [stage_2_side, stage_3_side, stage_4_side, stage_5_side], dim=1
    ))
    road_fts = net.road_seg(stage_fuse)
    road_final = net.conv_road_final(road_fts)
    junc_fts = net.junc_seg(stage_fuse)
    junc_final = net.conv_junc_final(junc_fts)

    placeholder = torch.zeros(
        stage_fuse.shape[0],
        32 * (net.num_targets - 1),
        stage_fuse.shape[2],
        stage_fuse.shape[3],
        device=stage_fuse.device,
        dtype=stage_fuse.dtype,
    )
    recurrent = torch.cat(
        [stage_fuse, road_fts, junc_fts, walked_path_small, placeholder], dim=1
    )
    anchor_features = None
    anchors = []
    anchors_lowrs = []
    for index in range(num_targets):
        next_step = net.fuse_module(recurrent)
        anchors_lowrs.append(_upsample(net.next_step_final(next_step), 4))

        decoded_4 = net.decoders[0](_upsample(stage_4, 2), next_step)
        decoded_3 = net.decoders[1](_upsample(stage_3, 2), decoded_4)
        decoded_2 = net.decoders[2](_upsample(stage_2, 2), _upsample(decoded_3, 2))
        decoded_1 = net.decoders[3](_upsample(stage_1, 2), _upsample(decoded_2, 2))

        channel_index = -(net.num_targets - index - 1) * 32
        if index < net.num_targets - 1:
            pooled = net.avgpool4(decoded_1)
            anchor_features = pooled if anchor_features is None else anchor_features + pooled
            recurrent[
                :,
                channel_index:channel_index + 32 if channel_index + 32 != 0 else None,
                :,
                :,
            ] = anchor_features
        anchors.append(net.conv_final(decoded_1))

    return {
        "road": road_final,
        "junc": junc_final,
        "anchor": torch.cat(anchors, dim=1),
        "anchor_lowrs": torch.cat(anchors_lowrs, dim=1),
    }


def forward_current_road_self(
    net: RPNet,
    aerial_image: torch.Tensor,
    walked_path: torch.Tensor,
    num_targets: int,
) -> Dict[str, torch.Tensor]:
    output = net(
        aerial_image=aerial_image,
        traj_image=None,
        aerial_traj_image=None,
        neighborhood_trajectory_norm=None,
        valid_mask=None,
        walked_path=walked_path,
        NUM_TARGETS=num_targets,
        test=False,
        model="origin",
        use_traj=False,
    )
    return {key: output[key] for key in OUTPUT_KEYS}


def _state_dict(payload: Any) -> Mapping[str, torch.Tensor]:
    state = payload.get("state_dict", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint does not contain a state-dict mapping")
    keys = list(state)
    if keys and all(key.startswith("module.") for key in keys):
        return {key[len("module."):]: value for key, value in state.items()}
    return state


def _load_matching_checkpoint(
    net: RPNet, checkpoint_path: Path
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    payload = torch.load(os.fspath(checkpoint_path), map_location="cpu")
    loaded = _state_dict(payload)
    expected = net.state_dict()
    matching = {
        key: value
        for key, value in loaded.items()
        if key in expected and tuple(value.shape) == tuple(expected[key].shape)
    }
    incompatible = net.load_state_dict(matching, strict=False)
    mismatch = sorted(
        key for key, value in loaded.items()
        if key in expected and tuple(value.shape) != tuple(expected[key].shape)
    )
    metadata = {}
    if isinstance(payload, Mapping):
        for key in (
            "outer_it", "path_it", "trajectory_mode", "model_name",
            "num_targets", "step_length", "window_size", "random_seed",
        ):
            if key in payload:
                metadata[key] = payload[key]
    report = {
        "checkpoint": os.fspath(checkpoint_path.resolve()),
        "checkpoint_state_keys": len(loaded),
        "matching_keys": len(matching),
        "shape_mismatch_keys": mismatch,
        "missing_model_keys": list(incompatible.missing_keys),
        "unexpected_model_keys": list(incompatible.unexpected_keys),
        "metadata": metadata,
    }
    return metadata, report


def _components(frame: np.ndarray, threshold: float) -> Dict[str, Any]:
    labels = measure.label(frame >= threshold, connectivity=2)
    areas = sorted((int(region.area) for region in measure.regionprops(labels)), reverse=True)
    return {
        "pixels": int(np.count_nonzero(frame >= threshold)),
        "component_count": len(areas),
        "component_areas_desc": areas[:10],
    }


def _analyse_output(
    output: Mapping[str, torch.Tensor],
    extension_vertex: Any,
    is_key_point: bool,
    thresholds: Tuple[float, ...],
    step_length: int,
    max_region_area: int,
) -> Dict[str, Any]:
    detached = {key: value.detach().cpu() for key, value in output.items()}
    anchor_probability = torch.sigmoid(detached["anchor"]).numpy()
    report: Dict[str, Any] = {"tensors": {}, "thresholds": {}}
    for key, value in detached.items():
        finite = bool(torch.isfinite(value).all())
        report["tensors"][key] = {
            "shape": list(value.shape),
            "finite": finite,
            "logit_min": float(value.min()),
            "logit_max": float(value.max()),
            "logit_mean": float(value.mean()),
        }
    report["anchor_channel_probability"] = [
        {
            "min": float(channel.min()),
            "max": float(channel.max()),
            "mean": float(channel.mean()),
        }
        for channel in anchor_probability[0]
    ]
    for threshold in thresholds:
        points = model_utils.map_to_coordinate(
            batch_output_maps=anchor_probability.copy(),
            batch_is_key_point=np.asarray([is_key_point]),
            batch_extension_vertices=[extension_vertex],
            ROAD_SEG_THRESHOLE=float(threshold),
            STEP_LENGTH=step_length,
            JUNC_MAX_REGION_AREA=max_region_area,
        )[0]
        report["thresholds"][str(threshold)] = {
            "coordinate_count": len(points),
            "coordinates": [[float(point.x), float(point.y)] for point in points],
            "channels": [
                _components(channel, float(threshold))
                for channel in anchor_probability[0]
            ],
        }
    return report


def _build_input(
    cfg: EasyDict,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Any, bool, Dict[str, Any]]:
    region_name = args.region or str(cfg.TEST.SINGLE_REGION)
    regions = get_regions(cfg.DIR.TEST_REGION_PATH)
    if region_name not in regions:
        raise KeyError("region {!r} is missing from TEST_REGION_PATH".format(region_name))
    region = regions[region_name]
    tile_size = int(cfg.TRAIN.IMG_SZ)
    window_size = int(cfg.TEST.WINDOW_SIZE)
    tile_start = model_utils.geom.Point(region.radius_x, region.radius_y).scale(tile_size)
    search_rect = model_utils.geom.Rectangle(
        tile_start, tile_start.add(model_utils.geom.Point(tile_size, tile_size))
    )
    start_point, start_metadata = _select_start_point(
        cfg, region_name, search_rect, args.start_x, args.start_y
    )
    cache = tileloader.TileCache(
        tile_dir=cfg.DIR.IMAGERY_DIR,
        traj_dir=os.fspath(args.output.parent / "trajectory_must_not_be_read"),
        tile_size=tile_size,
        window_size=window_size,
        limit=int(cfg.TRAIN.PARALLEL_TILES),
    )
    path = model_utils.Path(
        0,
        training=False,
        gc=None,
        tile_data=_tile_data(
            region_name,
            search_rect,
            cache,
            start_point,
            include_starting_location=True,
        ),
        all_trajectories=None,
        all_pixel_trajectories=None,
        graph=None,
        road_seg=None,
        WINDOW_SIZE=window_size,
    )
    extension_vertex, is_key_point = path.pop(follow_order=True)
    if extension_vertex is None:
        raise RuntimeError("fixed-start Path unexpectedly contains no extension vertex")
    data = path.make_path_input(
        extension_vertex=extension_vertex,
        fetch_list=["aerial_image_chw", "walked_path", "walked_path_small"],
        traj_filter=False,
        is_key_point=is_key_point,
        WINDOW_SIZE=window_size,
    )
    input_metadata = {
        "region": region_name,
        "start": [float(start_point.x), float(start_point.y)],
        "start_metadata": start_metadata,
        "is_key_point": bool(is_key_point),
        "aerial_shape": list(data["aerial_image_chw"].shape),
        "walked_path_shape": list(data["walked_path"].shape),
        "walked_path_small_shape": list(data["walked_path_small"].shape),
    }
    return data, extension_vertex, bool(is_key_point), input_metadata


def _run_case(
    checkpoint: Path,
    architecture: str,
    aerial: torch.Tensor,
    walked: torch.Tensor,
    walked_small: torch.Tensor,
    extension_vertex: Any,
    is_key_point: bool,
    args: argparse.Namespace,
    cfg: EasyDict,
    device: torch.device,
) -> Dict[str, Any]:
    torch.manual_seed(args.seed)
    net = RPNet(num_targets=int(cfg.TEST.NUM_TARGETS), backbone_pretrained=False)
    _, load_report = _load_matching_checkpoint(net, checkpoint)
    net = net.to(device).eval()
    with torch.no_grad():
        if architecture == "original_vecroad":
            output = forward_original_vecroad(
                net, aerial, walked_small, int(cfg.TEST.NUM_TARGETS)
            )
        elif architecture == "current_road_self":
            output = forward_current_road_self(
                net, aerial, walked, int(cfg.TEST.NUM_TARGETS)
            )
        else:
            raise ValueError("unknown architecture: {}".format(architecture))
    analysis = _analyse_output(
        output,
        extension_vertex,
        is_key_point,
        tuple(args.thresholds),
        int(cfg.TEST.STEP_LENGTH),
        int(cfg.TEST.BINARIZE_MAP.JUNC_MAX_REGION_AREA),
    )
    del output, net
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"checkpoint_load": load_report, "output": analysis}


def main() -> int:
    args = _parse_args()
    cfg = _load_config(args.config)
    device = _device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    for checkpoint in (args.current_checkpoint, args.reference_checkpoint):
        if not checkpoint.is_file():
            raise FileNotFoundError("checkpoint not found: {}".format(checkpoint.resolve()))

    data, extension, is_key, input_metadata = _build_input(cfg, args)
    aerial = torch.from_numpy(data["aerial_image_chw"]).unsqueeze(0).float().to(device)
    walked = torch.from_numpy(data["walked_path"]).unsqueeze(0).float().to(device)
    walked_small = torch.from_numpy(data["walked_path_small"]).unsqueeze(0).float().to(device)

    cases = {}
    combinations = (
        ("current_checkpoint__current_architecture", args.current_checkpoint, "current_road_self"),
        ("current_checkpoint__original_architecture", args.current_checkpoint, "original_vecroad"),
        ("reference_checkpoint__current_architecture", args.reference_checkpoint, "current_road_self"),
        ("reference_checkpoint__original_architecture", args.reference_checkpoint, "original_vecroad"),
    )
    for name, checkpoint, architecture in combinations:
        print("running {}".format(name), flush=True)
        cases[name] = _run_case(
            checkpoint,
            architecture,
            aerial,
            walked,
            walked_small,
            extension,
            is_key,
            args,
            cfg,
            device,
        )

    result = {
        "purpose": "anchor regression diagnosis; not a performance benchmark",
        "config": os.fspath(args.config.resolve()),
        "device": str(device),
        "seed": args.seed,
        "input": input_metadata,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("saved {}".format(args.output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
