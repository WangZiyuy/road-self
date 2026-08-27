"""Frozen one-GPU worker for a single Stage S3 controlled run."""

from __future__ import annotations

import argparse
import json
import os
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
import torch.nn.functional as F
import yaml
from easydict import EasyDict

from model.model import RPNet
from utils.OSMDataset import OSMDataset
from utils.seg_raster.stage_s3 import (
    anchor_metrics,
    binary_segmentation_metrics,
    identity_sha256,
    load_stage_s3_config,
    sample_identity,
    sha256_file,
    strict_shared_state_audit,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")


def append_jsonl(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def frozen_checkout(expected_sha: str) -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True,
        capture_output=True).stdout.strip()
    branch = subprocess.run(
        ["git", "branch", "--show-current"], check=True, text=True,
        capture_output=True).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"], check=True,
        text=True, capture_output=True).stdout.strip()
    if head != expected_sha or branch != "feat/seg-raster-only" or status:
        raise RuntimeError("worker refused a non-frozen checkout")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _model_for(config: dict) -> RPNet:
    raster_cfg = config["TRAJ"]["RASTER"]
    raster_mode = config["TRAJ"]["MODE"] == "raster_seg_only"
    return RPNet(
        num_targets=4,
        backbone_pretrained=False,
        enable_trajectory_modules=False,
        enable_raster_segmentation=raster_mode,
        raster_use_valid_mask=bool(raster_cfg["USE_VALID_MASK"]),
        anchor_grad_to_seg=bool(raster_cfg["ANCHOR_GRAD_TO_SEG"]),
    )


def _load_initialization(model: RPNet, config: dict) -> dict:
    path = REPO_ROOT / config["S3"]["INITIALIZATION"]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    key = (
        "raster_state_dict" if config["TRAJ"]["MODE"] == "raster_seg_only"
        else "image_only_state_dict")
    state = payload[key]
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("strict initialization returned incompatible keys")
    return {
        "path_label": "${RUN_ROOT}/runtime/common_initialization.pth.tar",
        "sha256": sha256_file(path),
        "payload_sha256": payload["initialization_sha256"],
        "selected_state": key,
    }


def _cfg_for_dataset(config: dict, split_extent: list[int]) -> EasyDict:
    cfg = json.loads(json.dumps(config))
    cfg["TRAIN"]["SPATIAL_EXTENT_XYXY"] = split_extent
    cfg["TRAIN"]["BATCH_SIZE"] = int(
        os.environ.get("S3_BATCH_SIZE", cfg["TRAIN"]["BATCH_SIZE"]))
    control = cfg["TRAJ"]["RASTER"].get("CONTROL")
    if control is not None:
        cfg["DIR"]["TRAJ_DIR"] = (
            "data_self/stage_s3_seg_raster/runtime/controls/{}".format(control))
    return EasyDict(cfg)


def _to_cuda(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array)).float().cuda(non_blocking=False)


def _forward(model: RPNet, batch: EasyDict, mode: str) -> dict[str, torch.Tensor]:
    traj = _to_cuda(batch.batch_traj_inputs) if mode == "raster_seg_only" else None
    traj_mask = (
        _to_cuda(batch.batch_traj_valid_masks)
        if mode == "raster_seg_only" else None)
    return model(
        aerial_image=_to_cuda(batch.batch_inputs),
        traj_image=traj,
        aerial_traj_image=None,
        neighborhood_trajectory_norm=None,
        valid_mask=None,
        walked_path=_to_cuda(batch.batch_walked_path_small),
        NUM_TARGETS=4,
        test=False,
        model="origin",
        use_traj=False,
        trajectory_mode=mode,
        traj_valid_mask=traj_mask,
    )


def _losses(output: dict[str, torch.Tensor], batch: EasyDict) -> dict[str, torch.Tensor]:
    target_anchor = _to_cuda(batch.batch_target_maps)
    anchor = torch.zeros((), device="cuda")
    anchor_lowrs = torch.zeros((), device="cuda")
    for index, end_index in enumerate(batch.batch_end_index):
        end_index = int(end_index)
        anchor = anchor + F.binary_cross_entropy_with_logits(
            output["anchor"][index, :end_index],
            target_anchor[index, :end_index], reduction="sum")
        anchor_lowrs = anchor_lowrs + F.binary_cross_entropy_with_logits(
            output["anchor_lowrs"][index, :end_index],
            target_anchor[index, :end_index], reduction="sum")
    road = F.binary_cross_entropy_with_logits(
        output["road"], _to_cuda(batch.batch_road_segmentation), reduction="sum")
    junction = F.binary_cross_entropy_with_logits(
        output["junc"], _to_cuda(batch.batch_junction_segmentation), reduction="sum")
    return {
        "anchor": anchor + anchor_lowrs,
        "road": road,
        "junction": junction,
        "total": anchor + anchor_lowrs + road + junction,
    }


def _seg_metrics(output: dict[str, torch.Tensor], batch: EasyDict, threshold: float) -> dict:
    road = binary_segmentation_metrics(
        output["road"].detach().cpu().numpy(),
        batch.batch_road_segmentation, threshold=threshold)
    junction = binary_segmentation_metrics(
        output["junc"].detach().cpu().numpy(),
        batch.batch_junction_segmentation, threshold=threshold)
    return {
        "road_precision": road["precision"], "road_recall": road["recall"],
        "road_f1": road["f1"], "road_iou": road["iou"],
        "junction_precision": junction["precision"],
        "junction_recall": junction["recall"],
        "junction_f1": junction["f1"], "junction_iou": junction["iou"],
        "segmentation_composite": (
            road["f1"] + road["iou"] + junction["f1"]) / 3.0,
    }


def _build_validation_batches(config: dict, split: dict) -> tuple[list[EasyDict], str]:
    extent = split["validation_extent"]
    extent_list = [extent["x0"], extent["y0"], extent["x1"], extent["y1"]]
    cfg = _cfg_for_dataset(config, extent_list)
    dataset = OSMDataset(cfg, net=None, training=True)
    batches = []
    identities = []
    for index in range(int(config["S3"]["VALIDATION_BATCHES"])):
        batch = dataset.get_batch()
        identities.append(identity_sha256([
            sample_identity(row) for row in batch.batch_sample_metadata]))
        dataset.push_and_vis_batch(batch, 0, index)
        batches.append(batch)
    return batches, identity_sha256(identities)


def _evaluate_validation(
    model: RPNet,
    batches: list[EasyDict],
    mode: str,
    threshold: float,
) -> tuple[dict, dict]:
    road_logits, road_targets = [], []
    junc_logits, junc_targets = [], []
    anchor_logits, anchor_targets, end_indices = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in batches:
            output = _forward(model, batch, mode)
            road_logits.append(output["road"].cpu().numpy())
            road_targets.append(batch.batch_road_segmentation)
            junc_logits.append(output["junc"].cpu().numpy())
            junc_targets.append(batch.batch_junction_segmentation)
            anchor_logits.append(output["anchor"].cpu().numpy())
            anchor_targets.append(batch.batch_target_maps)
            end_indices.extend(int(value) for value in batch.batch_end_index)
    model.train()
    road = binary_segmentation_metrics(
        np.concatenate(road_logits), np.concatenate(road_targets),
        threshold=threshold)
    junction = binary_segmentation_metrics(
        np.concatenate(junc_logits), np.concatenate(junc_targets),
        threshold=threshold)
    segmentation = {
        "road_precision": road["precision"], "road_recall": road["recall"],
        "road_f1": road["f1"], "road_iou": road["iou"],
        "road_auprc": road["auprc"],
        "junction_precision": junction["precision"],
        "junction_recall": junction["recall"],
        "junction_f1": junction["f1"], "junction_iou": junction["iou"],
        "junction_auprc": junction["auprc"],
        "segmentation_composite": (
            road["f1"] + road["iou"] + junction["f1"]) / 3.0,
        "fixed_threshold": threshold,
        "validation_batch_count": len(batches),
    }
    anchors = anchor_metrics(
        np.concatenate(anchor_logits), np.concatenate(anchor_targets),
        end_indices, threshold=threshold)
    return segmentation, anchors


def _save_checkpoint(
    path: Path, model: RPNet, optimizer: torch.optim.Optimizer,
    step: int, code_sha: str, config_sha: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "stage": "seg_raster_stage_s3", "code_sha": code_sha,
        "config_sha": config_sha, "optimizer_step": step,
        "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
    }, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-code-sha", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage S3 worker requires CUDA")
    frozen_checkout(args.run_code_sha)
    config = load_stage_s3_config(args.config)
    if os.environ.get("S3_RUN_CODE_SHA") != args.run_code_sha:
        raise RuntimeError("launcher/worker code SHA mismatch")
    configured_sha = config["S3"].get("CODE_SHA_REQUIRED", "")
    if configured_sha and configured_sha != args.run_code_sha:
        raise RuntimeError("config was frozen for a different code SHA")

    seed = int(config["S3"]["SEED"])
    set_seed(seed)
    run_id = config["S3"]["RUN_ID"]
    run_dir = REPO_ROOT / config["S3"]["RUN_ROOT"] / run_id
    checkpoint_dir = run_dir / "checkpoints"
    evaluation_dir = run_dir / "evaluation"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    (run_dir / "resolved_config.yml").write_text(
        yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    config_sha = identity_sha256(config)

    split_path = REPO_ROOT / config["S3"]["SPLIT_MANIFEST"]
    split = json.loads(split_path.read_text(encoding="utf-8"))
    sample_plan = json.loads((
        REPO_ROOT / "artifacts/stage_s3_sample_plan.json"
    ).read_text(encoding="utf-8"))
    if sample_plan.get("status") != "PASS":
        raise RuntimeError("training requires a PASS sample plan")
    expected_first_100_sha = sample_plan[
        "first_100_batch_identity_sha256"][config["S3"]["RUN_KEY"]]
    extent = split["train_extent"]
    extent_list = [extent["x0"], extent["y0"], extent["x1"], extent["y1"]]
    model = _model_for(config)
    initialization = _load_initialization(model, config)
    model.cuda().train()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["TRAIN"]["SOLVER"]["LEARNING_RATE"]),
        betas=(0.9, 0.99),
        weight_decay=float(config["TRAIN"]["SOLVER"]["WEIGHT_DECAY"]))
    # Validation is generated from its frozen, disjoint extent before the
    # training RNG is reset.  It never enters the training sample stream.
    set_seed(seed + 1)
    validation_batches, validation_plan_sha = _build_validation_batches(
        config, split)
    set_seed(seed)
    cfg = _cfg_for_dataset(config, extent_list)
    dataset = OSMDataset(cfg, net=None, training=True)

    requested_steps = int(os.environ.get(
        "S3_OPTIMIZER_STEPS", config["TRAIN"]["OPTIMIZER_STEPS"]))
    evaluation_interval = int(config["S3"]["EVALUATION_INTERVAL"])
    metrics_interval = int(config["S3"]["METRICS_INTERVAL"])
    threshold = float(config["S3"]["FIXED_THRESHOLD"])
    heartbeat_seconds = int(config["S3"]["HEARTBEAT_SECONDS"])
    first_identities = []
    best_composite = -1.0
    start = time.time()
    last_heartbeat = 0.0
    status = "RUNNING"
    invalid_reason = None
    final_metrics = {}
    best_metrics = {}
    try:
        for step in range(1, requested_steps + 1):
            batch = dataset.get_batch()
            identities = [sample_identity(row) for row in batch.batch_sample_metadata]
            if len(first_identities) < 100:
                first_identities.append(identity_sha256(identities))
                if len(first_identities) == 100:
                    observed = identity_sha256(first_identities)
                    if observed != expected_first_100_sha:
                        raise RuntimeError("first 100 batch sample plan mismatch")
            optimizer.zero_grad(set_to_none=True)
            output = _forward(model, batch, config["TRAJ"]["MODE"])
            expected_shapes = {
                "road": (cfg.TRAIN.BATCH_SIZE, 1, 64, 64),
                "junc": (cfg.TRAIN.BATCH_SIZE, 1, 64, 64),
                "anchor": (cfg.TRAIN.BATCH_SIZE, 4, 256, 256),
                "anchor_lowrs": (cfg.TRAIN.BATCH_SIZE, 4, 256, 256),
            }
            for key, expected in expected_shapes.items():
                if tuple(output[key].shape) != expected:
                    raise RuntimeError("{} shape mismatch".format(key))
                if not torch.isfinite(output[key]).all():
                    raise FloatingPointError("non-finite {} logits".format(key))
            losses = _losses(output, batch)
            if not all(torch.isfinite(value) for value in losses.values()):
                raise FloatingPointError("non-finite loss")
            losses["total"].backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError("non-finite gradient: {}".format(name))
            torch.nn.utils.clip_grad_value_(model.parameters(), 1e4)
            optimizer.step()
            batch.batch_output_road = torch.sigmoid(output["road"]).detach().cpu().numpy()
            batch.batch_output_junc = torch.sigmoid(output["junc"]).detach().cpu().numpy()
            batch.batch_output_anchor_maps = torch.sigmoid(output["anchor"]).detach().cpu().numpy()
            dataset.push_and_vis_batch(batch, 1, step - 1)
            final_metrics = _seg_metrics(output, batch, threshold)
            if step == 1 or step % metrics_interval == 0:
                frozen_checkout(args.run_code_sha)
                append_jsonl(metrics_path, {
                    "time": utc_now(), "step": step,
                    "samples_seen": step * int(cfg.TRAIN.BATCH_SIZE),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "loss": {key: float(value.detach()) for key, value in losses.items()},
                    "metrics": final_metrics,
                    "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / 2**20,
                    "gpu_memory_reserved_mb": torch.cuda.memory_reserved() / 2**20,
                })
            if step % evaluation_interval == 0 or step == requested_steps:
                validation_metrics, validation_anchor_metrics = _evaluate_validation(
                    model, validation_batches, config["TRAJ"]["MODE"], threshold)
                _save_checkpoint(
                    checkpoint_dir / "latest.pth.tar", model, optimizer, step,
                    args.run_code_sha, config_sha)
                append_jsonl(metrics_path, {
                    "time": utc_now(), "step": step,
                    "kind": "frozen_validation", "metrics": validation_metrics,
                    "validation_plan_sha": validation_plan_sha,
                })
                write_json(evaluation_dir / "segmentation.json", validation_metrics)
                write_json(evaluation_dir / "anchor.json", validation_anchor_metrics)
                composite = validation_metrics["segmentation_composite"]
                if composite > best_composite:
                    best_composite = composite
                    best_metrics = validation_metrics
                    _save_checkpoint(
                        checkpoint_dir / "best.pth.tar", model, optimizer, step,
                        args.run_code_sha, config_sha)
            now = time.time()
            if now - last_heartbeat >= heartbeat_seconds:
                write_json(run_dir / "heartbeat.json", {
                    "status": "RUNNING", "time": utc_now(), "step": step,
                    "samples_seen": step * int(cfg.TRAIN.BATCH_SIZE),
                    "code_sha": args.run_code_sha})
                last_heartbeat = now
        status = "PASS"
    except Exception as error:
        status = "INVALID"
        invalid_reason = "{}: {}".format(type(error).__name__, error)
        raise
    finally:
        latest = checkpoint_dir / "latest.pth.tar"
        best = checkpoint_dir / "best.pth.tar"
        summary = {
            "stage": "seg_raster_stage_s3", "run_id": run_id,
            "code_sha": args.run_code_sha, "config_sha": config_sha,
            "data_manifest_sha": split["manifest_sha256"],
            "sample_plan_sha": sample_plan["plan_sha256"],
            "initialization_sha": initialization.get("payload_sha256"),
            "validation_plan_sha": validation_plan_sha,
            "seed": seed, "status": status, "invalid_reason": invalid_reason,
            "optimizer_steps": step if "step" in locals() else 0,
            "samples_seen": (step * int(cfg.TRAIN.BATCH_SIZE)) if "step" in locals() else 0,
            "first_100_batch_identity_sha256": identity_sha256(first_identities),
            "best_checkpoint_sha256": sha256_file(best) if best.is_file() else None,
            "final_checkpoint_sha256": sha256_file(latest) if latest.is_file() else None,
            "best_metrics": best_metrics,
            "final_metrics": final_metrics,
            "start_time": datetime.fromtimestamp(start, timezone.utc).isoformat(),
            "end_time": utc_now(),
            "elapsed_seconds": time.time() - start,
            "peak_allocated_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_memory_mb": torch.cuda.max_memory_reserved() / 2**20,
        }
        write_json(run_dir / "summary.json", summary)
        write_json(run_dir / "run_manifest.json", {
            "run_id": run_id, "code_sha": args.run_code_sha,
            "config_sha": config_sha, "split_sha": split["manifest_sha256"],
            "initialization": initialization,
            "trajectory_sequence_required": False,
            "data_parallel": False, "visible_cuda_device_count": torch.cuda.device_count(),
        })
        write_json(run_dir / "heartbeat.json", {
            "status": status, "time": utc_now(),
            "step": summary["optimizer_steps"], "code_sha": args.run_code_sha})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
