"""Evaluate one Stage S3C checkpoint under the uniform graph resource caps."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import yaml

from tools.seg_raster.audit_stage_s3a_graph import (
    build_graph, connectivity, directional_apls, junction_metrics,
    parse_graph, raw_stats, select_postprocessed_graph, topology_metrics,
)
from tools.seg_raster.evaluate_stage_s3b_graph import deep_merge
from utils.seg_raster.stage_s3 import load_stage_s3_config, sha256_file
from utils.seg_raster.stage_s3c import GRAPH_CAPS


def verify_checkout(expected_sha: str) -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True,
        capture_output=True).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"], check=True,
        text=True, capture_output=True).stdout.strip()
    if head != expected_sha or status:
        raise RuntimeError("graph evaluator requires a clean frozen checkout")


def prepare_config(args: argparse.Namespace) -> Path:
    with (REPO_ROOT / "configs/default_self.yml").open(
            "r", encoding="utf-8") as handle:
        defaults = yaml.load(handle, Loader=yaml.UnsafeLoader)
    base = deep_merge(defaults, load_stage_s3_config(
        REPO_ROOT / "configs/stage_s3c_common.yml"))
    config = deepcopy(base)
    raster = args.run_key in ("R1", "R2", "R3")
    control = {"R1": "aligned", "R2": "zero", "R3": "shift_fixed"}.get(
        args.run_key)
    checkpoint_dir = args.output_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    link = checkpoint_dir / "selected.pth.tar"
    if not link.exists():
        link.symlink_to(args.checkpoint)
    trajectory_dir = args.output_dir / "graph_trajectory"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    if raster:
        source = args.control_root / control
        raster_link = trajectory_dir / "xian.png"
        if not raster_link.exists():
            raster_link.symlink_to(source / "xian_0_0.png")
    config["TASK"] = "stage_s3c_{}_graph".format(args.run_key)
    config["TRAJ"] = {
        "MODE": "raster_seg_only" if raster else "none",
        "SEQUENCE": {"ENABLED": False},
        "RASTER": {"CONTROL": control, "INPUT_SEMANTICS": "binary_presence",
                   "USE_VALID_MASK": True, "VALID_EXTENT_WH": [4096, 4096],
                   "ANCHOR_GRAD_TO_SEG": False, "SHIFT_X": 128, "SHIFT_Y": 128}}
    config["DIR"].update({
        "CHECK_POINT_DIR": str(checkpoint_dir),
        "TEST_TRAJ_DIR": str(trajectory_dir),
        "TRAJ_DIR": str(args.control_root / (control or "aligned")),
        "SAVE_SEG_DIR": str(args.output_dir / "segmentation"),
        "INFER_STEP_DIR": str(args.output_dir / "infer_step"),
        "SAVE_GRAPH_DIR": str(args.output_dir / "graphs"),
        "SHORTCUT_DIR": str(args.output_dir / "shortcuts")})
    config["TRAIN"].update({
        "NUM_TARGETS": 4, "IMG_SZ": 4096, "WINDOW_SIZE": 256,
        "BACKBONE_PRETRAINED": False, "DATA_PARALLEL": False,
        "MODEL": "origin"})
    config["TEST"].update({
        "CKPT": "selected", "CKPT_FILE": "selected.pth.tar", "GPU_ID": "0",
        "DATA_PARALLEL": False, "BATCH_SIZE_SEG": 32,
        "BATCH_SIZE_ANCHOR": 15, "TEST_IMG_SZ": 4096, "NUM_TILES": 1,
        "CROP_SZ": 256, "SAMPLE_STEP": 1, "SKIP_EMPTY_CROP": False,
        "WINDOW_SIZE": 256, "INFER_STEP": "start", "SINGLE_REGION": "xian",
        "START_FROM_ROAD_PEAK": False, "START_FROM_JUNC_PEAK": True,
        "NUM_TARGETS": 4, "STEP_LENGTH": 20, "RECT_RADIUS": 10,
        "FOLLOW_MODE": "follow_output", "AVG_CONFIDENCE_THRESHOLD": 0.2,
        "SAVE_EXAMPLES": False, "PRINT_ITERATION": 100,
        "BINARIZE_MAP": {"ROAD_SEG_THRESHOLE": args.threshold,
                         "JUNC_MAX_REGION_AREA": 200,
                         "JUNC_SEG_THRESHOLE": args.threshold,
                         "ANCHOR_MAX_REGION_AREA": 1000,
                         "MIN_BAD_ROAD_AREA": 200}})
    path = args.output_dir / "resolved_graph_config.yml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-code-sha", required=True)
    parser.add_argument("--run-key", choices=("BASELINE", "R0", "R1", "R2", "R3"),
                        required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-kind", choices=("official_release", "stage_s3c"),
                        required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    args = parser.parse_args()
    verify_checkout(args.run_code_sha)
    if not torch.cuda.is_available():
        raise RuntimeError("formal graph evaluation requires remote CUDA")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if args.checkpoint_kind == "stage_s3c":
        if payload.get("code_sha") != args.run_code_sha:
            raise RuntimeError("checkpoint/run-code SHA mismatch")
        if payload.get("kind") != "versioned_model_only":
            raise RuntimeError("Stage S3C graph requires a versioned checkpoint")
    elif set(payload) != {"state_dict"}:
        raise RuntimeError("official release checkpoint wrapper is unexpected")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = prepare_config(args)
    random.seed(20260827)
    np.random.seed(20260827)
    torch.manual_seed(20260827)
    torch.cuda.manual_seed_all(20260827)
    torch.use_deterministic_algorithms(True, warn_only=True)
    sys.argv = ["infer.py", "--config", str(config_path)]
    import infer
    statuses = []
    original = infer.infer_anchor

    def capped(*positional, **keywords):
        keywords["resource_limits"] = GRAPH_CAPS
        result = original(*positional, **keywords)
        statuses.append(dict(capped.last_resource_status))
        return result

    infer.infer_anchor = capped
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        infer.main()
    elapsed = time.monotonic() - started
    resource = statuses[-1] if statuses else {
        "status": "INVALID", "reached_caps": [], "natural_termination": None,
        "snapshot": {"iterations": 0}}
    predicted_path = select_postprocessed_graph(args.output_dir)
    reference_vertices, reference_edges = parse_graph(
        REPO_ROOT / "data_self/input/graphs/xian.graph")
    predicted_vertices, predicted_edges = parse_graph(predicted_path)
    reference = build_graph(reference_vertices, reference_edges)
    predicted = build_graph(predicted_vertices, predicted_edges)
    reference_dense = build_graph(reference_vertices, reference_edges, densify=True)
    predicted_dense = build_graph(predicted_vertices, predicted_edges, densify=True)
    ref_to_pred = directional_apls(reference_dense, predicted_dense)
    pred_to_ref = directional_apls(predicted_dense, reference_dense)
    topo = topology_metrics(reference, predicted)
    junction = junction_metrics(reference, predicted)
    duplicates, undirected = raw_stats(predicted_vertices, predicted_edges)
    status = ("RESOURCE_CAP_REACHED"
              if resource["status"] == "RESOURCE_CAP_REACHED" else "PASS")
    result = {
        "stage": "seg_raster_stage_s3c", "status": status,
        "natural_termination": status == "PASS",
        "execution_environment": "REMOTE_TRAINING_SERVER",
        "run_code_sha": args.run_code_sha, "run_key": args.run_key,
        "checkpoint_kind": args.checkpoint_kind,
        "checkpoint_samples_seen": payload.get("samples_seen", 0),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "fixed_probability_threshold": args.threshold,
        "resource_caps": GRAPH_CAPS, "resource_status": resource,
        "graph_iterations": int(resource["snapshot"]["iterations"]),
        "vertex_count": len(predicted_vertices),
        "directed_edge_count": len(predicted_edges),
        "undirected_edge_count": undirected,
        "candidate_edge_count": topo["candidate_edge_count"],
        "dangling_edge_count": sum(1 for node in predicted.nodes()
                                   if predicted.degree(node) == 1),
        "duplicate_edge_count": duplicates,
        "runtime_seconds": elapsed,
        "apls": float((ref_to_pred["score"] + pred_to_ref["score"]) / 2),
        "apls_protocol": "deterministic_pixel_graph_approximation",
        "topo": topo["f1"], "topo_metrics": topo,
        "connectivity": connectivity(predicted),
        "junction_correctness": junction,
        "peak_gpu_memory_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
        "peak_gpu_memory_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
        "physical_gpu_index": args.physical_gpu,
        "gpu_name": torch.cuda.get_device_name(0),
        "started_at_utc": started_at,
        "graph_sha256": sha256_file(predicted_path),
        "graph_logical_path": "${S3C_REMOTE_OUTPUT}/graph/" + args.run_key
                              + "/xian.graph"}
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(
        result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
