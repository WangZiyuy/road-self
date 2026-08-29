"""One-GPU frozen worker for a single Stage S3B protocol-repair run."""

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
import torch.nn.functional as F
import yaml
from easydict import EasyDict

from model.model import RPNet
from utils.OSMDataset import OSMDataset
from utils.seg_raster.stage_s3 import (
    anchor_metrics, binary_segmentation_metrics, identity_sha256,
    load_stage_s3_config, sample_identity, sha256_file)
from utils.seg_raster.stage_s3b import (
    CHECKPOINT_STEPS, LOSS_KINDS, frozen_plan_batch_identities,
    junction_loss, legacy_composite,
    repair_composite, save_versioned_model_checkpoint)


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
        raise RuntimeError("Stage S3B worker refused a non-frozen checkout")


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
    arrays = [
        batch.batch_inputs, batch.batch_walked_path_small,
        batch.batch_road_segmentation, batch.batch_junction_segmentation,
        batch.batch_target_maps, batch.batch_is_key_point,
        np.asarray(batch.batch_end_index),
    ]
    return array_sha256(arrays)


def configure(args: argparse.Namespace) -> dict:
    config = load_stage_s3_config(REPO_ROOT / "configs/stage_s3b_common.yml")
    raster = args.input_kind == "raster"
    config["S3"]["RUN_ID"] = args.run_id
    config["S3"]["RUN_KEY"] = args.run_key
    config["S3"]["CODE_SHA_REQUIRED"] = args.run_code_sha
    config["S3"]["PHASE"] = args.phase
    config["S3"]["INPUT_KIND"] = args.input_kind
    config["S3B"]["JUNCTION_LOSS"] = args.loss_kind
    config["S3B"]["JUNCTION_POS_WEIGHT"] = args.pos_weight
    config["S3B"]["JUNCTION_LOSS_ALPHA"] = args.loss_alpha
    config["TRAIN"]["SOLVER"]["LR_MULTIPLIER"] = args.lr_multiplier
    config["TRAIN"]["SOLVER"]["LEARNING_RATE"] *= args.lr_multiplier
    config["TRAJ"]["MODE"] = "raster_seg_only" if raster else "none"
    config["TRAJ"]["RASTER"]["CONTROL"] = args.control if raster else None
    config["TRAJ"]["RASTER"]["ANCHOR_GRAD_TO_SEG"] = False
    config["S3"]["ANCHOR_GRAD_TO_SEG"] = False
    return config


def model_for(config: dict) -> RPNet:
    raster_mode = config["TRAJ"]["MODE"] == "raster_seg_only"
    return RPNet(
        num_targets=4, backbone_pretrained=False,
        enable_trajectory_modules=False,
        enable_raster_segmentation=raster_mode,
        raster_use_valid_mask=bool(config["TRAJ"]["RASTER"]["USE_VALID_MASK"]),
        anchor_grad_to_seg=False)


def load_initialization(model: RPNet, config: dict) -> dict:
    path = REPO_ROOT / config["S3"]["INITIALIZATION"]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    key = ("raster_state_dict" if config["TRAJ"]["MODE"] == "raster_seg_only"
           else "image_only_state_dict")
    result = model.load_state_dict(payload[key], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("strict common initialization failed")
    return {
        "path_label": "${S3B_RUNTIME}/common_initialization.pth.tar",
        "file_sha256": sha256_file(path),
        "content_sha256": payload["initialization_sha256"],
        "selected_state": key,
    }


def cfg_for_dataset(config: dict, split_extent: list[int]) -> EasyDict:
    cfg = json.loads(json.dumps(config))
    cfg["TRAIN"]["SPATIAL_EXTENT_XYXY"] = split_extent
    control = cfg["TRAJ"]["RASTER"].get("CONTROL")
    if control is not None:
        cfg["DIR"]["TRAJ_DIR"] = (
            "data_self/stage_s3b_seg_raster/runtime/controls/" + control)
    return EasyDict(cfg)


def to_cuda(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array)).float().cuda(non_blocking=False)


def forward(model: RPNet, batch: EasyDict, mode: str) -> dict[str, torch.Tensor]:
    raster = mode == "raster_seg_only"
    return model(
        aerial_image=to_cuda(batch.batch_inputs),
        traj_image=to_cuda(batch.batch_traj_inputs) if raster else None,
        aerial_traj_image=None, neighborhood_trajectory_norm=None,
        valid_mask=None, walked_path=to_cuda(batch.batch_walked_path_small),
        NUM_TARGETS=4, test=False, model="origin", use_traj=False,
        trajectory_mode=mode,
        traj_valid_mask=(to_cuda(batch.batch_traj_valid_masks) if raster else None))


def losses(
    output: dict[str, torch.Tensor], batch: EasyDict, config: dict
) -> dict[str, torch.Tensor]:
    target_anchor = to_cuda(batch.batch_target_maps)
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
        output["road"], to_cuda(batch.batch_road_segmentation), reduction="sum")
    junction = junction_loss(
        output["junc"], to_cuda(batch.batch_junction_segmentation),
        kind=config["S3B"]["JUNCTION_LOSS"],
        pos_weight=float(config["S3B"]["JUNCTION_POS_WEIGHT"]),
        alpha=float(config["S3B"]["JUNCTION_LOSS_ALPHA"]),
        dice_weight=float(config["S3B"]["DICE_WEIGHT"]),
        dice_smooth=float(config["S3B"]["DICE_SMOOTH"]))
    combined_anchor = anchor + anchor_lowrs
    return {"anchor": combined_anchor, "road": road, "junction": junction,
            "total": combined_anchor + road + junction}


def parameter_gradient_groups(model: RPNet) -> dict[str, float]:
    sums = {"road": 0.0, "junction": 0.0, "anchor": 0.0,
            "raster": 0.0, "shared": 0.0}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        value = float(torch.sum(parameter.grad.detach().double() ** 2))
        if "segmentation_raster_fusion" in name:
            key = "raster"
        elif "road" in name:
            key = "road"
        elif "junc" in name:
            key = "junction"
        elif "anchor" in name or "decoder" in name:
            key = "anchor"
        else:
            key = "shared"
        sums[key] += value
    return {key: math_sqrt(value) for key, value in sums.items()}


def math_sqrt(value: float) -> float:
    return float(np.sqrt(value))


def build_validation_batches(config: dict, split: dict) -> tuple[list[EasyDict], str]:
    extent = split["validation_extent"]
    extent_list = [extent["x0"], extent["y0"], extent["x1"], extent["y1"]]
    dataset = OSMDataset(cfg_for_dataset(config, extent_list), net=None, training=True)
    batches, identities = [], []
    for index in range(int(config["S3"]["VALIDATION_BATCHES"])):
        batch = dataset.get_batch()
        identities.append(identity_sha256([
            sample_identity(row) for row in batch.batch_sample_metadata]))
        dataset.push_and_vis_batch(batch, 0, index)
        batches.append(batch)
    return batches, identity_sha256(identities)


def threshold_curve(probability: np.ndarray, target: np.ndarray) -> list[dict]:
    truth = np.asarray(target) > 0
    rows = []
    for threshold in np.linspace(0.01, 0.99, 99):
        prediction = probability >= threshold
        tp = int(np.count_nonzero(prediction & truth))
        fp = int(np.count_nonzero(prediction & ~truth))
        fn = int(np.count_nonzero(~prediction & truth))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"threshold": float(threshold), "precision": precision,
                     "recall": recall, "f1": f1,
                     "predicted_positive_count": tp + fp})
    return rows


def evaluate_validation(
    model: RPNet, batches: list[EasyDict], mode: str, threshold: float
) -> tuple[dict, dict, str]:
    road_logits, road_targets, junc_logits, junc_targets = [], [], [], []
    anchor_logits, anchor_targets, end_indices, per_sample = [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in batches:
            output = forward(model, batch, mode)
            road_logits.append(output["road"].cpu().numpy())
            road_targets.append(batch.batch_road_segmentation)
            junc_logits.append(output["junc"].cpu().numpy())
            junc_targets.append(batch.batch_junction_segmentation)
            anchor_logits.append(output["anchor"].cpu().numpy())
            anchor_targets.append(batch.batch_target_maps)
            end_indices.extend(int(value) for value in batch.batch_end_index)
    model.train()
    road_logits_np, road_targets_np = np.concatenate(road_logits), np.concatenate(road_targets)
    junc_logits_np, junc_targets_np = np.concatenate(junc_logits), np.concatenate(junc_targets)
    road = binary_segmentation_metrics(road_logits_np, road_targets_np, threshold=threshold)
    junction = binary_segmentation_metrics(junc_logits_np, junc_targets_np, threshold=threshold)
    for index in range(road_logits_np.shape[0]):
        r = binary_segmentation_metrics(
            road_logits_np[index:index + 1], road_targets_np[index:index + 1],
            threshold=threshold)
        j = binary_segmentation_metrics(
            junc_logits_np[index:index + 1], junc_targets_np[index:index + 1],
            threshold=threshold)
        per_sample.append({"sample_index": index, "road_f1": r["f1"],
                           "road_iou": r["iou"], "road_auprc": r["auprc"],
                           "junction_auprc": j["auprc"]})
    metrics = {
        "scope": "validation", "model_mode": "eval", "no_grad": True,
        "sigmoid_boundary": True, "fixed_threshold": threshold,
        "aggregation": "all_validation_pixels",
        "validation_batch_count": len(batches),
        "validation_sample_count": int(road_logits_np.shape[0]),
        "road_precision": road["precision"], "road_recall": road["recall"],
        "road_f1": road["f1"], "road_iou": road["iou"],
        "road_auprc": road["auprc"],
        "junction_precision": junction["precision"],
        "junction_recall": junction["recall"],
        "junction_f1": junction["f1"], "junction_iou": junction["iou"],
        "junction_auprc": junction["auprc"],
        "junction_predicted_positive_count": (
            junction["true_positive"] + junction["false_positive"]),
        "junction_target_positive_count": (
            junction["true_positive"] + junction["false_negative"]),
        "legacy_composite": 0.0, "repair_composite": 0.0,
        "junction_threshold_curve": threshold_curve(
            1.0 / (1.0 + np.exp(-np.clip(junc_logits_np, -80, 80))),
            junc_targets_np),
        "per_sample": per_sample,
    }
    metrics["legacy_composite"] = legacy_composite(metrics)
    metrics["repair_composite"] = repair_composite(metrics)
    anchors = anchor_metrics(
        np.concatenate(anchor_logits), np.concatenate(anchor_targets), end_indices,
        threshold=threshold)
    prediction_sha = array_sha256([road_logits_np, junc_logits_np,
                                   np.concatenate(anchor_logits)])
    return metrics, anchors, prediction_sha


def save_full_checkpoint(
    path: Path, model: RPNet, optimizer: torch.optim.Optimizer, *, step: int,
    code_sha: str, config_sha: str, metric_code_sha: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "stage": "seg_raster_stage_s3b", "kind": "optimizer_resume",
        "code_sha": code_sha, "config_sha": config_sha,
        "metric_code_sha": metric_code_sha, "optimizer_step": int(step),
        "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
    }, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-code-sha", required=True)
    parser.add_argument("--phase", choices=("A", "B", "C"), required=True)
    parser.add_argument("--run-key", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input-kind", choices=("image_only", "raster"), required=True)
    parser.add_argument("--control", choices=("aligned", "zero", "shift_fixed"))
    parser.add_argument("--lr-multiplier", type=float, required=True)
    parser.add_argument("--loss-kind", choices=LOSS_KINDS, required=True)
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument("--loss-alpha", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage S3B formal worker requires remote CUDA")
    frozen_checkout(args.run_code_sha)
    if os.environ.get("S3B_RUN_CODE_SHA") != args.run_code_sha:
        raise RuntimeError("launcher/worker run-code SHA mismatch")
    config = configure(args)
    if args.input_kind == "raster" and args.control is None:
        raise ValueError("raster run requires a control")
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
    metric_code_sha = sha256_file(Path(__file__))

    split = json.loads((REPO_ROOT / config["S3"]["SPLIT_MANIFEST"])
                       .read_text(encoding="utf-8"))
    sample_plan = json.loads((REPO_ROOT / "artifacts/stage_s3_sample_plan.json")
                             .read_text(encoding="utf-8"))
    if sample_plan.get("status") != "PASS":
        raise RuntimeError("Stage S3B requires the frozen PASS sample plan")
    model = model_for(config)
    initialization = load_initialization(model, config)
    model.cuda().train()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["TRAIN"]["SOLVER"]["LEARNING_RATE"]),
        betas=(0.9, 0.99),
        weight_decay=float(config["TRAIN"]["SOLVER"]["WEIGHT_DECAY"]))
    set_seed(seed + 1)
    validation_batches, validation_plan_sha = build_validation_batches(config, split)
    set_seed(seed)
    extent = split["train_extent"]
    extent_list = [extent["x0"], extent["y0"], extent["x1"], extent["y1"]]
    cfg = cfg_for_dataset(config, extent_list)
    dataset = OSMDataset(cfg, net=None, training=True)
    requested_steps = int(config["TRAIN"]["OPTIMIZER_STEPS"])
    if tuple(config["S3B"]["CHECKPOINT_STEPS"]) != CHECKPOINT_STEPS:
        raise RuntimeError("checkpoint grid differs from frozen S3B contract")
    threshold = float(config["S3"]["FIXED_THRESHOLD"])
    heartbeat_seconds = int(config["S3"]["HEARTBEAT_SECONDS"])
    first_identities, first_common_shas = [], []
    first_raster_shas, first_valid_mask_shas = [], []
    validation_by_step, checkpoint_inventory = {}, []
    best_score = -1.0
    best_validation_metrics = {}
    last_train_batch_metrics = {}
    start = time.time()
    last_heartbeat = 0.0
    status, invalid_reason, step = "RUNNING", None, 0

    # Replay the full parity gate before the first optimizer update, then
    # reset the deterministic dataset stream for comparable training.
    for parity_index in range(100):
        parity_batch = dataset.get_batch()
        first_identities.append(identity_sha256([
            sample_identity(row) for row in parity_batch.batch_sample_metadata]))
        first_common_shas.append(common_batch_sha(parity_batch))
        if hasattr(parity_batch, "batch_traj_valid_masks"):
            first_raster_shas.append(array_sha256([
                parity_batch.batch_traj_inputs]))
            first_valid_mask_shas.append(array_sha256([
                parity_batch.batch_traj_valid_masks]))
        dataset.push_and_vis_batch(parity_batch, 0, parity_index)
    expected_identities = frozen_plan_batch_identities(
        sample_plan["sample_order"], count=100)
    if first_identities != expected_identities:
        mismatches = [index for index, pair in enumerate(
            zip(first_identities, expected_identities)) if pair[0] != pair[1]]
        raise RuntimeError(
            "pre-training sample parity differs from frozen sample_order at "
            + str(mismatches[:10]))
    expected_common = [row["common_tensor_sha256"]
                       for row in sample_plan["sample_order"][:100]]
    if first_common_shas != expected_common:
        raise RuntimeError("pre-training common tensor parity mismatch")
    if config["TRAJ"]["MODE"] == "raster_seg_only":
        source_key = {"aligned": "C1", "zero": "C2",
                      "shift_fixed": "C3"}[config["TRAJ"]["RASTER"]["CONTROL"]]
        expected_raster = sample_plan["raster_control_batch_sha256"][source_key][:100]
        if first_raster_shas != [row["raster_sha256"] for row in expected_raster]:
            raise RuntimeError("pre-training raster control parity mismatch")
        if first_valid_mask_shas != [row["valid_mask_sha256"]
                                     for row in expected_raster]:
            raise RuntimeError("pre-training valid-mask parity mismatch")
    set_seed(seed)
    dataset = OSMDataset(cfg, net=None, training=True)

    def evaluate_and_save(current_step: int) -> None:
        nonlocal best_score, best_validation_metrics
        metrics, anchor_values, prediction_sha = evaluate_validation(
            model, validation_batches, config["TRAJ"]["MODE"], threshold)
        versioned = checkpoint_dir / "model_step_{:06d}.pth.tar".format(current_step)
        save_versioned_model_checkpoint(
            versioned, model.state_dict(), step=current_step,
            code_sha=args.run_code_sha, config_sha=config_sha,
            metric_code_sha=metric_code_sha)
        checkpoint_sha = sha256_file(versioned)
        metrics.update({"checkpoint_step": current_step,
                        "checkpoint_sha256": checkpoint_sha,
                        "prediction_sha256": prediction_sha,
                        "metric_code_sha": metric_code_sha,
                        "optimizer_learning_rate": optimizer.param_groups[0]["lr"]})
        validation_by_step[str(current_step)] = metrics
        write_json(evaluation_dir / "validation_step_{:06d}.json".format(current_step), {
            "segmentation": metrics, "anchor": anchor_values})
        append_jsonl(metrics_path, {"time": utc_now(), "step": current_step,
                                   "kind": "validation", "metrics": metrics})
        save_full_checkpoint(
            checkpoint_dir / "latest.pth.tar", model, optimizer,
            step=current_step, code_sha=args.run_code_sha,
            config_sha=config_sha, metric_code_sha=metric_code_sha)
        if metrics["repair_composite"] > best_score:
            best_score = metrics["repair_composite"]
            best_validation_metrics = metrics
            save_full_checkpoint(
                checkpoint_dir / "best.pth.tar", model, optimizer,
                step=current_step, code_sha=args.run_code_sha,
                config_sha=config_sha, metric_code_sha=metric_code_sha)
        checkpoint_inventory.append({
            "step": current_step,
            "logical_path": "${S3B_RUN_ROOT}/" + args.run_id
                            + "/checkpoints/" + versioned.name,
            "kind": "versioned_model_only", "size_bytes": versioned.stat().st_size,
            "sha256": checkpoint_sha, "prediction_sha256": prediction_sha,
            "optimizer_included": False})

    try:
        evaluate_and_save(0)
        for step in range(1, requested_steps + 1):
            batch = dataset.get_batch()
            optimizer.zero_grad(set_to_none=True)
            output = forward(model, batch, config["TRAJ"]["MODE"])
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
            loss_values = losses(output, batch, config)
            if not all(bool(torch.isfinite(value)) for value in loss_values.values()):
                raise FloatingPointError("non-finite loss")
            loss_values["total"].backward()
            maximum_gradient = 0.0
            finite_gradient_values = 0
            total_gradient_values = 0
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    continue
                total_gradient_values += parameter.grad.numel()
                finite = torch.isfinite(parameter.grad)
                finite_gradient_values += int(finite.sum())
                if not bool(finite.all()):
                    raise FloatingPointError("non-finite gradient: " + name)
                maximum_gradient = max(
                    maximum_gradient, float(parameter.grad.detach().abs().max()))
            gradient_groups = parameter_gradient_groups(model)
            torch.nn.utils.clip_grad_value_(model.parameters(), 1e4)
            optimizer.step()
            road_train = binary_segmentation_metrics(
                output["road"].detach().cpu().numpy(),
                batch.batch_road_segmentation, threshold=threshold)
            junc_train = binary_segmentation_metrics(
                output["junc"].detach().cpu().numpy(),
                batch.batch_junction_segmentation, threshold=threshold)
            last_train_batch_metrics = {
                "scope": "last_train_batch", "model_mode": "train",
                "no_grad": False, "checkpoint_step": step,
                "batch_count": 1, "sample_count": int(cfg.TRAIN.BATCH_SIZE),
                "sigmoid_boundary": True, "threshold": threshold,
                "aggregation": "single_last_training_batch",
                "road_f1": road_train["f1"], "road_iou": road_train["iou"],
                "junction_f1": junc_train["f1"]}
            if step == 1 or step % int(config["S3"]["METRICS_INTERVAL"]) == 0:
                frozen_checkout(args.run_code_sha)
                append_jsonl(metrics_path, {
                    "time": utc_now(), "step": step, "kind": "training",
                    "samples_seen": step * int(cfg.TRAIN.BATCH_SIZE),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "loss": {key: float(value.detach())
                             for key, value in loss_values.items()},
                    "gradient_norm_by_parameter_group": gradient_groups,
                    "gradient_max_abs": maximum_gradient,
                    "gradient_finite_ratio": (
                        finite_gradient_values / total_gradient_values
                        if total_gradient_values else 1.0),
                    "last_train_batch_metrics": last_train_batch_metrics,
                    "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / 2**20,
                    "gpu_memory_reserved_mb": torch.cuda.memory_reserved() / 2**20})
            if step in CHECKPOINT_STEPS:
                evaluate_and_save(step)
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
        latest_step = str(max(map(int, validation_by_step))) if validation_by_step else None
        latest_validation = validation_by_step.get(latest_step, {}) if latest_step else {}
        summary = {
            "stage": "seg_raster_stage_s3b", "phase": args.phase,
            "run_key": args.run_key, "run_id": args.run_id,
            "code_sha": args.run_code_sha, "config_sha": config_sha,
            "metric_code_sha": metric_code_sha,
            "data_manifest_sha": split["manifest_sha256"],
            "sample_plan_sha": sample_plan["plan_sha256"],
            "initialization_sha": initialization["content_sha256"],
            "validation_plan_sha": validation_plan_sha,
            "seed": seed, "status": status, "invalid_reason": invalid_reason,
            "optimizer_steps": step,
            "samples_seen": step * int(cfg.TRAIN.BATCH_SIZE),
            "first_100_batch_identity_sha256": identity_sha256(first_identities),
            "first_100_batch_identity_gate_source": (
                "recomputed_from_frozen_sample_order_rows"),
            "historical_aggregate_hash_used_for_gate": False,
            "first_100_common_tensor_sha256": identity_sha256(first_common_shas),
            "first_100_raster_sha256": (
                identity_sha256(first_raster_shas) if first_raster_shas else None),
            "first_100_valid_mask_sha256": (
                identity_sha256(first_valid_mask_shas)
                if first_valid_mask_shas else None),
            "last_train_batch_metrics": last_train_batch_metrics,
            "validation_metrics_by_step": validation_by_step,
            "best_validation_metrics": best_validation_metrics,
            "latest_validation_metrics": latest_validation,
            "common_step_validation_metrics": None,
            "checkpoint_inventory": checkpoint_inventory,
            "start_time": datetime.fromtimestamp(start, timezone.utc).isoformat(),
            "end_time": utc_now(), "elapsed_seconds": time.time() - start,
            "peak_allocated_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_memory_mb": torch.cuda.max_memory_reserved() / 2**20,
        }
        write_json(run_dir / "summary.json", summary)
        write_json(run_dir / "run_manifest.json", {
            "stage": "seg_raster_stage_s3b", "run_key": args.run_key,
            "run_id": args.run_id, "code_sha": args.run_code_sha,
            "config_sha": config_sha, "initialization": initialization,
            "trajectory_sequence_required": False,
            "transformer_constructed": False, "legacy_dsf_constructed": False,
            "data_parallel": False,
            "visible_cuda_device_count": torch.cuda.device_count()})
        write_json(run_dir / "heartbeat.json", {
            "status": status, "time": utc_now(), "step": step,
            "invalid_reason": invalid_reason})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
