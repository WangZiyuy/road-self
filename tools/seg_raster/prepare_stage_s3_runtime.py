"""Prepare ignored Xi'an controls, common initialization, and frozen manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image
import torch

from model.model import RPNet
from utils.seg_raster import build_valid_mask
from utils.seg_raster.stage_s3 import (
    CONTROLS,
    STAGE_S3_SEED,
    apply_raster_control,
    assert_zero_initialized_raster_residual,
    build_spatial_split,
    experiment_matrix_payload,
    identity_sha256,
    sha256_file,
    strict_shared_state_audit,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")


def copy_input(source: Path, destination: Path) -> dict:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "destination_label": "${DATA_ROOT}/" + destination.relative_to(
            REPO_ROOT / "data_self").as_posix(),
        "size_bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "copied_from_read_only_source": True,
        "intended_for_commit": False,
    }


def save_binary_png(array: np.ndarray, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((array > 0).astype(np.uint8) * 255, mode="L").save(path)
    return {
        "path_label": "${RUN_ROOT}/" + path.relative_to(
            REPO_ROOT / "data_self/stage_s3_seg_raster").as_posix(),
        "shape_hw": list(array.shape), "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path), "intended_for_commit": False,
    }


def make_initialization(path: Path) -> dict:
    random.seed(STAGE_S3_SEED)
    np.random.seed(STAGE_S3_SEED)
    torch.manual_seed(STAGE_S3_SEED)
    image = RPNet(
        num_targets=4, backbone_pretrained=False,
        enable_trajectory_modules=False, enable_raster_segmentation=False)
    image_state = {key: value.detach().cpu() for key, value in image.state_dict().items()}
    random.seed(STAGE_S3_SEED)
    np.random.seed(STAGE_S3_SEED)
    torch.manual_seed(STAGE_S3_SEED)
    raster = RPNet(
        num_targets=4, backbone_pretrained=False,
        enable_trajectory_modules=False, enable_raster_segmentation=True)
    raster_state = {key: value.detach().cpu() for key, value in raster.state_dict().items()}
    audit = strict_shared_state_audit(image_state, raster_state)
    if audit["status"] != "PASS":
        raise RuntimeError("shared deterministic initialization mismatch")
    shared_value_mismatch = sorted(
        key for key in image_state.keys() & raster_state.keys()
        if not torch.equal(image_state[key], raster_state[key]))
    if shared_value_mismatch:
        raise RuntimeError(
            "shared deterministic values mismatch: {}".format(
                shared_value_mismatch[:10]))
    residual_keys = assert_zero_initialized_raster_residual(raster_state)
    if not residual_keys:
        raise RuntimeError("zero-initialized raster residual was not found")
    content_identity = {
        "seed": STAGE_S3_SEED,
        "image_keys": sorted(image_state), "raster_keys": sorted(raster_state),
        "shared_audit": audit, "zero_residual_keys": residual_keys,
    }
    payload = {
        "schema_version": "1.0.0", "stage": "seg_raster_stage_s3",
        "source": "deterministic_common_initialization_snapshot",
        "seed": STAGE_S3_SEED,
        "initialization_sha256": identity_sha256(content_identity),
        "image_only_state_dict": image_state,
        "raster_state_dict": raster_state,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {
        "status": "PASS", "source": payload["source"],
        "path_label": "${RUN_ROOT}/runtime/common_initialization.pth.tar",
        "file_sha256": sha256_file(path),
        "content_identity_sha256": payload["initialization_sha256"],
        "shared_key_count": audit["shared_key_count"],
        "shared_value_mismatch": shared_value_mismatch,
        "missing_key": audit["missing_shared_keys"],
        "unexpected_key": audit["unexpected_non_raster_keys"],
        "shape_mismatch": audit["shape_mismatch_keys"],
        "raster_only_key_count": audit["raster_only_key_count"],
        "zero_initialized_residual_keys": residual_keys,
        "training_source": "no trusted image-only checkpoint was available",
        "intended_for_commit": False,
    }


def coverage_summary(raster: np.ndarray, split: dict, stride: int = 256) -> dict:
    result = {}
    for name in ("train", "validation"):
        extent = split[name + "_extent"]
        values = []
        for y in range(extent["y0"], extent["y1"] - split["crop_size"] + 1, stride):
            for x in range(extent["x0"], extent["x1"] - split["crop_size"] + 1, stride):
                crop = raster[y:y + split["crop_size"], x:x + split["crop_size"]]
                values.append(float(np.count_nonzero(crop) / crop.size))
        result[name] = {
            "eligible_grid_crop_count": len(values),
            "trajectory_coverage_min": min(values) if values else 0.0,
            "trajectory_coverage_mean": float(np.mean(values)) if values else 0.0,
            "trajectory_coverage_max": max(values) if values else 0.0,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    original = Path(r"E:\Code\VecRoad_self\data_self\input")
    parser.add_argument("--aerial-tile-source", type=Path, default=original / "imagery/xian_0_0.png")
    parser.add_argument("--aerial-canvas-source", type=Path, default=original / "imagery_8192/xian.png")
    parser.add_argument("--graph-source", type=Path, default=original / "graphs/xian.graph")
    parser.add_argument("--all-regions-source", type=Path, default=original / "regions/all_regions.txt")
    parser.add_argument("--test-regions-source", type=Path, default=original / "regions/xian_regions.txt")
    parser.add_argument(
        "--raster-canvas", type=Path,
        default=REPO_ROOT / "data_self/input/traj_test/xian.png")
    parser.add_argument("--base-sha", required=True)
    args = parser.parse_args()

    current_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
        text=True, capture_output=True).stdout.strip()
    if current_head != args.base_sha:
        raise RuntimeError("runtime preparation HEAD differs from frozen run SHA")
    data_input = REPO_ROOT / "data_self/input"
    copied = {
        "aerial_tile": copy_input(args.aerial_tile_source, data_input / "imagery/xian_0_0.png"),
        "aerial_canvas": copy_input(args.aerial_canvas_source, data_input / "imagery_8192/xian.png"),
        "graph": copy_input(args.graph_source, data_input / "graphs/xian.graph"),
        "all_regions": copy_input(args.all_regions_source, data_input / "regions/all_regions.txt"),
        "test_regions": copy_input(args.test_regions_source, data_input / "regions/xian_regions.txt"),
    }

    raw = np.asarray(Image.open(args.raster_canvas).convert("L"))
    if raw.shape != (8192, 8192):
        raise ValueError("canonical Xi'an canvas must be 8192x8192")
    valid = build_valid_mask(8192, 8192, (4300, 5000))
    runtime = REPO_ROOT / "data_self/stage_s3_seg_raster/runtime"
    controls = {}
    valid_mask_sha256 = hashlib.sha256(
        valid.astype(np.uint8, copy=False).tobytes()).hexdigest()
    for control in CONTROLS:
        binary, unchanged_mask = apply_raster_control(raw, valid, control)
        if not np.array_equal(unchanged_mask, valid):
            raise AssertionError("control changed the real valid mask")
        canvas_info = save_binary_png(
            binary, runtime / "control_canvases" / control / "xian.png")
        # Tileloader consumes x,y-oriented 4096 tiles; the source files are
        # standard H,W, and its load path performs the historical swap once.
        tile_info = save_binary_png(
            binary[:4096, :4096], runtime / "controls" / control / "xian_0_0.png")
        controls[control] = {
            "canvas": canvas_info, "training_tile": tile_info,
            "valid_mask_sha256": valid_mask_sha256,
            "nonzero_pixel_count": int(np.count_nonzero(binary)),
        }

    split = build_spatial_split(canvas_wh=(4096, 4096), crop_size=256, boundary_buffer=256)
    split.update({
        "stage": "seg_raster_stage_s3", "region": "xian",
        "graph_source": "${DATA_ROOT}/input/graphs/xian.graph",
        "graph_sha256": copied["graph"]["sha256"],
        "raster_coverage_distribution": coverage_summary((raw > 0), split),
        "validation_samples_enter_training_plan": False,
    })
    split["manifest_sha256"] = identity_sha256({
        key: value for key, value in split.items() if key != "manifest_sha256"})
    write_json(REPO_ROOT / "artifacts/stage_s3_split_manifest.json", split)
    initialization = make_initialization(runtime / "common_initialization.pth.tar")
    write_json(REPO_ROOT / "artifacts/stage_s3_initialization_audit.json", initialization)
    write_json(REPO_ROOT / "artifacts/stage_s3_experiment_matrix.json", experiment_matrix_payload())
    write_json(runtime / "runtime_manifest.json", {
        "status": "PASS", "code_sha": current_head, "copied_inputs": copied,
        "controls": controls, "initialization": initialization,
        "split_manifest_sha": split["manifest_sha256"],
        "all_runtime_files_intended_for_commit": False,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
