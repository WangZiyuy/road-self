"""One-GPU worker for Stage S3C frozen-explorer adaptation."""

from __future__ import annotations

import argparse
import hashlib
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
import yaml
from easydict import EasyDict

from model.model import RPNet
from utils.OSMDataset import OSMDataset
from utils.seg_raster.stage_s3 import (
    binary_segmentation_metrics, identity_sha256, load_stage_s3_config,
    sample_identity, sha256_file,
)
from utils.seg_raster.stage_s3c import (
    MAX_SAMPLES_SEEN, SAMPLE_GRID, SampleBudgetCounter,
    checkpoint_name, configure_frozen_explorer,
    enforce_original_batch_norm_eval, original_batch_norm_checksum,
    repair_composite, segmentation_losses, set_frozen_explorer_train_mode,
    strict_load_official_checkpoint, trainable_parameters,
)


LOSS_LEGACY = "legacy_exact"
LOSS_BALANCED = "class_balanced_bce"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")


def append_jsonl(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(
            value, ensure_ascii=False, allow_nan=False) + "\n")


def frozen_checkout(expected_sha: str) -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True,
        capture_output=True).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"], check=True,
        text=True, capture_output=True).stdout.strip()
    if head != expected_sha or status:
        raise RuntimeError("Stage S3C worker refused a non-frozen checkout")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def array_sha256(arrays: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def common_batch_sha(batch: EasyDict) -> str:
    return array_sha256([
        batch.batch_inputs, batch.batch_walked_path_small,
        batch.batch_road_segmentation, batch.batch_junction_segmentation,
        batch.batch_target_maps, batch.batch_is_key_point,
        np.asarray(batch.batch_end_index),
    ])


def configure(args: argparse.Namespace) -> dict:
    config = load_stage_s3_config(REPO_ROOT / "configs/stage_s3c_common.yml")
    raster = args.input_kind == "raster"
    config["S3"]["RUN_ID"] = args.run_id
    config["S3"]["RUN_KEY"] = args.run_key
    config["S3"]["CODE_SHA_REQUIRED"] = args.run_code_sha
    config["S3"]["PHASE"] = args.phase
    config["S3"]["INPUT_KIND"] = args.input_kind
    config["TRAJ"]["MODE"] = "raster_seg_only" if raster else "none"
    config["TRAJ"]["RASTER"]["CONTROL"] = args.control if raster else None
    config["TRAJ"]["RASTER"]["ANCHOR_GRAD_TO_SEG"] = False
    config["S3C"]["JUNCTION_LOSS"] = args.loss_kind
    config["S3C"]["JUNCTION_POS_WEIGHT"] = args.pos_weight
    config["S3C"]["JUNCTION_LOSS_ALPHA"] = args.loss_alpha
    return config


def cfg_for_dataset(config: dict, extent: list[int]) -> EasyDict:
    cfg = json.loads(json.dumps(config))
    cfg["TRAIN"]["SPATIAL_EXTENT_XYXY"] = extent
    control = cfg["TRAJ"]["RASTER"].get("CONTROL")
    if control is not None:
        root = os.environ.get("S3C_CONTROL_ROOT")
        if not root:
            raise RuntimeError("S3C_CONTROL_ROOT is required for raster runs")
        cfg["DIR"]["TRAJ_DIR"] = os.fspath(Path(root) / control)
    return EasyDict(cfg)


def to_cuda(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array)).float().cuda(non_blocking=False)


def forward_model(
    model: RPNet, batch: EasyDict, mode: str, *, segmentation_only: bool
) -> dict[str, torch.Tensor]:
    raster = mode == "raster_seg_only"
    return model(
        aerial_image=to_cuda(batch.batch_inputs),
        traj_image=to_cuda(batch.batch_traj_inputs) if raster else None,
        aerial_traj_image=None, neighborhood_trajectory_norm=None,
        valid_mask=None, walked_path=to_cuda(batch.batch_walked_path_small),
        NUM_TARGETS=4, test=False, model="origin", use_traj=False,
        trajectory_mode=mode,
        traj_valid_mask=(to_cuda(batch.batch_traj_valid_masks)
                         if raster else None),
        segmentation_only=segmentation_only,
    )


def load_baseline(model: RPNet, *, raster: bool) -> dict:
    raw_path = os.environ.get("S3C_BASELINE_CHECKPOINT")
    if not raw_path:
        raise RuntimeError("S3C_BASELINE_CHECKPOINT is required")
    path = Path(raw_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    audit = strict_load_official_checkpoint(
        model, payload,
        allowed_new_prefixes=("segmentation_raster_fusion.",) if raster else ())
    audit.update({
        "path_label": "${OFFICIAL_VECROAD_RELEASE}/data/ckpt/vecroad.pth.tar",
        "sha256": sha256_file(path),
        "baseline_provenance": "OFFICIAL_RELEASE",
    })
    return audit


def build_validation_batches(config: dict, split: dict) -> tuple[list[EasyDict], str]:
    extent = split["validation_extent"]
    xyxy = [extent["x0"], extent["y0"], extent["x1"], extent["y1"]]
    dataset = OSMDataset(cfg_for_dataset(config, xyxy), net=None, training=True)
    batches, identities = [], []
    for index in range(int(config["S3"]["VALIDATION_BATCHES"])):
        batch = dataset.get_batch()
        identities.append(identity_sha256([
            sample_identity(row) for row in batch.batch_sample_metadata]))
        dataset.push_and_vis_batch(batch, 0, index)
        batches.append(batch)
    return batches, identity_sha256(identities)


def evaluate_segmentation(
    model: RPNet, batches: list[EasyDict], mode: str, threshold: float
) -> tuple[dict, str]:
    road_logits, road_targets, junc_logits, junc_targets = [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in batches:
            output = forward_model(
                model, batch, mode, segmentation_only=True)
            road_logits.append(output["road"].cpu().numpy())
            road_targets.append(batch.batch_road_segmentation)
            junc_logits.append(output["junc"].cpu().numpy())
            junc_targets.append(batch.batch_junction_segmentation)
    road_logits_np = np.concatenate(road_logits)
    road_targets_np = np.concatenate(road_targets)
    junc_logits_np = np.concatenate(junc_logits)
    junc_targets_np = np.concatenate(junc_targets)
    road = binary_segmentation_metrics(
        road_logits_np, road_targets_np, threshold=threshold)
    junction = binary_segmentation_metrics(
        junc_logits_np, junc_targets_np, threshold=threshold)
    per_sample = []
    for index in range(road_logits_np.shape[0]):
        r = binary_segmentation_metrics(
            road_logits_np[index:index + 1], road_targets_np[index:index + 1],
            threshold=threshold)
        j = binary_segmentation_metrics(
            junc_logits_np[index:index + 1], junc_targets_np[index:index + 1],
            threshold=threshold)
        per_sample.append({
            "sample_index": index,
            "road_f1": r["f1"], "road_iou": r["iou"],
            "road_auprc": r["auprc"],
            "junction_f1": j["f1"], "junction_auprc": j["auprc"],
        })
    metrics = {
        "scope": "frozen_validation", "model_mode": "eval", "no_grad": True,
        "fixed_threshold": threshold,
        "validation_batch_count": len(batches),
        "validation_sample_count": int(road_logits_np.shape[0]),
        "road_precision": road["precision"], "road_recall": road["recall"],
        "road_f1": road["f1"], "road_iou": road["iou"],
        "road_auprc": road["auprc"],
        "junction_precision": junction["precision"],
        "junction_recall": junction["recall"],
        "junction_f1": junction["f1"], "junction_iou": junction["iou"],
        "junction_auprc": junction["auprc"], "per_sample": per_sample,
    }
    metrics["repair_composite"] = repair_composite(metrics)
    set_frozen_explorer_train_mode(model)
    return metrics, array_sha256([road_logits_np, junc_logits_np])


def save_versioned_checkpoint(
    path: Path, model: RPNet, *, samples_seen: int, optimizer_updates: int,
    code_sha: str, config_sha: str,
) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-{}".format(os.getpid()))
    torch.save({
        "stage": "seg_raster_stage_s3c",
        "kind": "versioned_model_only",
        "code_sha": code_sha,
        "config_sha": config_sha,
        "samples_seen": int(samples_seen),
        "optimizer_updates": int(optimizer_updates),
        "state_dict": model.state_dict(),
    }, temporary)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-code-sha", required=True)
    parser.add_argument("--phase", choices=("A", "B"), required=True)
    parser.add_argument("--run-key", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input-kind", choices=("image_only", "raster"),
                        required=True)
    parser.add_argument("--control", choices=("aligned", "zero", "shift_fixed"))
    parser.add_argument("--loss-kind", choices=(LOSS_LEGACY, LOSS_BALANCED),
                        required=True)
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument("--loss-alpha", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage S3C formal worker requires remote CUDA")
    frozen_checkout(args.run_code_sha)
    if os.environ.get("S3C_RUN_CODE_SHA") != args.run_code_sha:
        raise RuntimeError("launcher/worker run-code SHA mismatch")
    if args.input_kind == "raster" and args.control is None:
        raise ValueError("raster run requires a control")
    config = configure(args)
    seed = int(config["S3"]["SEED"])
    set_seed(seed)
    run_dir = REPO_ROOT / config["S3"]["RUN_ROOT"] / args.run_id
    checkpoint_dir = run_dir / "checkpoints"
    evaluation_dir = run_dir / "evaluation"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    (run_dir / "resolved_config.yml").write_text(
        yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    config_sha = identity_sha256(config)
    raster = args.input_kind == "raster"
    model = RPNet(
        num_targets=4, backbone_pretrained=False,
        enable_trajectory_modules=False,
        enable_raster_segmentation=raster,
        raster_use_valid_mask=True, anchor_grad_to_seg=False)
    baseline_audit = load_baseline(model, raster=raster)
    parameter_contract = configure_frozen_explorer(
        model, raster_enabled=raster)
    model.cuda()
    set_frozen_explorer_train_mode(model)
    bn_initial = original_batch_norm_checksum(model)
    parameters = trainable_parameters(model)
    optimizer = torch.optim.Adam(
        parameters, lr=float(config["TRAIN"]["SOLVER"]["LEARNING_RATE"]),
        betas=(0.9, 0.99),
        weight_decay=float(config["TRAIN"]["SOLVER"]["WEIGHT_DECAY"]))
    optimizer.zero_grad(set_to_none=True)

    split = json.loads((REPO_ROOT / config["S3"]["SPLIT_MANIFEST"])
                       .read_text(encoding="utf-8"))
    sample_plan_path = REPO_ROOT / config["S3"]["SAMPLE_PLAN"]
    sample_plan = json.loads(sample_plan_path.read_text(encoding="utf-8"))
    if sample_plan.get("status") != "PASS":
        raise RuntimeError("Stage S3C requires a frozen PASS sample plan")
    expected_batches = sample_plan["micro_batches"]
    expected_micro_batches = MAX_SAMPLES_SEEN // int(config["TRAIN"]["BATCH_SIZE"])
    if len(expected_batches) != expected_micro_batches:
        raise RuntimeError("sample plan length differs from frozen sample budget")

    set_seed(seed + 1)
    validation_batches, validation_plan_sha = build_validation_batches(
        config, split)
    set_seed(seed)
    extent = split["train_extent"]
    train_extent = [extent["x0"], extent["y0"], extent["x1"], extent["y1"]]
    dataset = OSMDataset(
        cfg_for_dataset(config, train_extent), net=None, training=True)
    threshold = float(config["S3"]["FIXED_THRESHOLD"])
    counter = SampleBudgetCounter(
        micro_batch_size=int(config["TRAIN"]["BATCH_SIZE"]),
        accumulation_steps=int(config["S3"]["GRADIENT_ACCUMULATION"]))
    validation_by_samples, checkpoint_inventory = {}, []
    first_twenty_identity, first_twenty_common = [], []
    first_twenty_raster, first_twenty_mask = [], []
    started = time.time()
    last_heartbeat = 0.0
    status, invalid_reason = "RUNNING", None

    def evaluate_and_save(samples_seen: int) -> None:
        before = original_batch_norm_checksum(model)
        metrics, prediction_sha = evaluate_segmentation(
            model, validation_batches, config["TRAJ"]["MODE"], threshold)
        after = original_batch_norm_checksum(model)
        if before != after or after != bn_initial:
            raise RuntimeError("original BatchNorm checksum changed")
        destination = checkpoint_dir / checkpoint_name(samples_seen)
        save_versioned_checkpoint(
            destination, model, samples_seen=samples_seen,
            optimizer_updates=counter.optimizer_updates,
            code_sha=args.run_code_sha, config_sha=config_sha)
        checkpoint_sha = sha256_file(destination)
        metrics.update({
            "samples_seen": samples_seen,
            "optimizer_updates": counter.optimizer_updates,
            "checkpoint_sha256": checkpoint_sha,
            "prediction_sha256": prediction_sha,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "original_bn_checksum_before": before,
            "original_bn_checksum_after": after,
        })
        validation_by_samples[str(samples_seen)] = metrics
        write_json(evaluation_dir / "validation_samples_{:06d}.json".format(
            samples_seen), {"segmentation": metrics})
        append_jsonl(metrics_path, {
            "time": utc_now(), "kind": "validation",
            "samples_seen": samples_seen, "metrics": metrics})
        checkpoint_inventory.append({
            "samples_seen": samples_seen,
            "optimizer_updates": counter.optimizer_updates,
            "logical_path": "${S3C_RUN_ROOT}/" + args.run_id
                            + "/checkpoints/" + destination.name,
            "size_bytes": destination.stat().st_size,
            "sha256": checkpoint_sha,
            "prediction_sha256": prediction_sha,
            "optimizer_included": False,
        })

    try:
        evaluate_and_save(0)
        for batch_index, expected in enumerate(expected_batches):
            batch = dataset.get_batch()
            identity = identity_sha256([
                sample_identity(row) for row in batch.batch_sample_metadata])
            common = common_batch_sha(batch)
            if identity != expected["batch_identity_sha256"]:
                raise RuntimeError("sample identity mismatch at micro-batch {}"
                                   .format(batch_index))
            if common != expected["common_tensor_sha256"]:
                raise RuntimeError("common tensor mismatch at micro-batch {}"
                                   .format(batch_index))
            if batch_index < 20:
                first_twenty_identity.append(identity)
                first_twenty_common.append(common)
                if raster:
                    first_twenty_raster.append(array_sha256([
                        batch.batch_traj_inputs]))
                    first_twenty_mask.append(array_sha256([
                        batch.batch_traj_valid_masks]))
            output = forward_model(
                model, batch, config["TRAJ"]["MODE"], segmentation_only=True)
            for key in ("road", "junc"):
                expected_shape = (int(config["TRAIN"]["BATCH_SIZE"]), 1, 64, 64)
                if tuple(output[key].shape) != expected_shape:
                    raise RuntimeError(key + " shape mismatch")
                if not bool(torch.isfinite(output[key]).all()):
                    raise FloatingPointError("non-finite " + key + " logits")
            losses = segmentation_losses(
                output, to_cuda(batch.batch_road_segmentation),
                to_cuda(batch.batch_junction_segmentation),
                junction_pos_weight=float(config["S3C"]["JUNCTION_POS_WEIGHT"]),
                junction_alpha=float(config["S3C"]["JUNCTION_LOSS_ALPHA"]))
            if not all(bool(torch.isfinite(value)) for value in losses.values()):
                raise FloatingPointError("non-finite segmentation loss")
            losses["total"].backward()
            # Restore the official follow_target state transition.  No anchor
            # prediction is required or consumed in this mode.
            dataset.push_and_vis_batch(batch, 0, batch_index)
            should_step = counter.record_micro_batch(
                int(config["TRAIN"]["BATCH_SIZE"]))
            if should_step:
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        if not parameter.requires_grad:
                            raise RuntimeError("frozen parameter received gradient: " + name)
                        if not bool(torch.isfinite(parameter.grad).all()):
                            raise FloatingPointError("non-finite gradient: " + name)
                torch.nn.utils.clip_grad_value_(parameters, 1e4)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                enforce_original_batch_norm_eval(model)
            if (should_step and counter.optimizer_updates %
                    int(config["S3"]["METRICS_INTERVAL"]) == 0):
                frozen_checkout(args.run_code_sha)
                append_jsonl(metrics_path, {
                    "time": utc_now(), "kind": "training",
                    "micro_batches": counter.micro_batches,
                    "optimizer_updates": counter.optimizer_updates,
                    "samples_seen": counter.samples_seen,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "loss": {key: float(value.detach())
                             for key, value in losses.items()},
                    "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / 2**20,
                    "gpu_memory_reserved_mb": torch.cuda.memory_reserved() / 2**20,
                })
            if counter.samples_seen in SAMPLE_GRID and should_step:
                evaluate_and_save(counter.samples_seen)
            now = time.time()
            if now - last_heartbeat >= int(config["S3"]["HEARTBEAT_SECONDS"]):
                write_json(run_dir / "heartbeat.json", {
                    "status": "RUNNING", "time": utc_now(),
                    "samples_seen": counter.samples_seen,
                    "optimizer_updates": counter.optimizer_updates,
                    "code_sha": args.run_code_sha})
                last_heartbeat = now
        if counter.samples_seen != MAX_SAMPLES_SEEN:
            raise RuntimeError("worker did not reach frozen sample budget")
        status = "PASS"
    except Exception as error:
        status = "INVALID"
        invalid_reason = "{}: {}".format(type(error).__name__, error)
        raise
    finally:
        final_bn = original_batch_norm_checksum(model)
        best_samples = None
        if validation_by_samples:
            best_samples = max(
                (int(samples) for samples in validation_by_samples),
                key=lambda value: (repair_composite(
                    validation_by_samples[str(value)]), -value))
        summary = {
            "stage": "seg_raster_stage_s3c", "phase": args.phase,
            "run_key": args.run_key, "run_id": args.run_id,
            "code_sha": args.run_code_sha, "config_sha": config_sha,
            "status": status, "invalid_reason": invalid_reason,
            "execution_environment": "REMOTE_TRAINING_SERVER",
            "seed": seed, "input_kind": args.input_kind,
            "control": args.control, "loss_kind": args.loss_kind,
            "junction_pos_weight": args.pos_weight,
            "junction_loss_alpha": args.loss_alpha,
            "optimizer_updates": counter.optimizer_updates,
            "micro_batches": counter.micro_batches,
            "samples_seen": counter.samples_seen,
            "micro_batch_per_gpu": int(config["TRAIN"]["BATCH_SIZE"]),
            "gradient_accumulation": int(config["S3"]["GRADIENT_ACCUMULATION"]),
            "sum_loss_divided_by_accumulation": False,
            "data_parallel": False, "ddp": False,
            "baseline_checkpoint": baseline_audit,
            "sample_plan_sha256": sample_plan["plan_sha256"],
            "split_manifest_sha256": split["manifest_sha256"],
            "validation_plan_sha256": validation_plan_sha,
            "first_20_batch_identity_sha256": identity_sha256(
                first_twenty_identity) if first_twenty_identity else None,
            "first_20_common_tensor_sha256": identity_sha256(
                first_twenty_common) if first_twenty_common else None,
            "first_20_raster_sha256": identity_sha256(
                first_twenty_raster) if first_twenty_raster else None,
            "first_20_valid_mask_sha256": identity_sha256(
                first_twenty_mask) if first_twenty_mask else None,
            "original_bn_checksum_initial": bn_initial,
            "original_bn_checksum_final": final_bn,
            "original_bn_checksum_unchanged": final_bn == bn_initial,
            "trainable_parameter_contract": parameter_contract,
            "validation_metrics_by_samples": validation_by_samples,
            "best_samples_seen": best_samples,
            "checkpoint_inventory": checkpoint_inventory,
            "runtime_seconds": time.time() - started,
            "peak_allocated_memory_mb": (
                torch.cuda.max_memory_allocated() / 2**20),
            "peak_reserved_memory_mb": (
                torch.cuda.max_memory_reserved() / 2**20),
        }
        write_json(run_dir / "summary.json", summary)
        write_json(run_dir / "heartbeat.json", {
            "status": status, "time": utc_now(),
            "samples_seen": counter.samples_seen,
            "optimizer_updates": counter.optimizer_updates,
            "invalid_reason": invalid_reason})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
