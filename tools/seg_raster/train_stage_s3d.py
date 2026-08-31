"""One-GPU worker for the frozen Stage S3D N0--N4 matrix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from model.model import RPNet
from tools.seg_raster.train_stage_s3c import (
    append_jsonl, array_sha256, build_validation_batches, common_batch_sha,
    frozen_checkout, identity_sha256, sample_identity, set_seed, to_cuda,
    utc_now, write_json,
)
from utils.OSMDataset import OSMDataset
from utils.seg_raster.stage_s3 import (
    binary_segmentation_metrics, load_stage_s3_config, sha256_file)
from utils.seg_raster.stage_s3c import (
    MAX_SAMPLES_SEEN, SAMPLE_GRID, SampleBudgetCounter, checkpoint_name,
    enforce_original_batch_norm_eval, original_batch_norm_checksum,
)
from utils.seg_raster.stage_s3d import (
    STAGE_S3D_SEED, STRICT_MODE, configure_road_only_training,
    density_stratified_derangement, named_tensor_sha256, permute_rasters,
    road_composite, road_loss, road_only_parameters, road_parameter_sha256,
    set_road_only_train_mode, strict_load_stage_s3d_baseline,
    trainable_gradient_sha256,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def configure(args: argparse.Namespace) -> dict:
    config = load_stage_s3_config(REPO_ROOT / "configs/stage_s3d_common.yml")
    config["S3"].update({
        "RUN_ID": args.run_id, "RUN_KEY": args.run_key,
        "CODE_SHA_REQUIRED": args.run_code_sha, "PHASE": "C",
    })
    config["TRAJ"]["MODE"] = STRICT_MODE
    config["TRAJ"]["RASTER"]["CONTROL"] = args.control
    return config


def cfg_for_dataset(config: dict, extent: list[int], control: str):
    from easydict import EasyDict
    cfg = json.loads(json.dumps(config))
    cfg["TRAIN"]["SPATIAL_EXTENT_XYXY"] = extent
    root = os.environ.get("S3D_CONTROL_ROOT")
    if not root:
        raise RuntimeError("S3D_CONTROL_ROOT is required")
    disk_control = "shift_large" if control == "shift_large" else "aligned"
    cfg["DIR"]["TRAJ_DIR"] = os.fspath(Path(root) / disk_control)
    return EasyDict(cfg)


def apply_control(batch, control: str, batch_index: int) -> dict | None:
    raster = np.asarray(batch.batch_traj_inputs)
    donor = None
    if control == "zero":
        batch.batch_traj_inputs = np.zeros_like(raster)
    elif control == "permuted":
        ratios = np.count_nonzero(raster, axis=(1, 2, 3)) / np.prod(
            raster.shape[1:])
        mapping = density_stratified_derangement(
            ratios.tolist(), seed=STAGE_S3D_SEED + int(batch_index))
        batch.batch_traj_inputs = permute_rasters(raster, mapping)
        donor = {
            "batch_index": int(batch_index), "mapping": mapping,
            "source_positive_ratios": ratios.tolist(),
            "donor_mapping_sha256": identity_sha256(mapping),
        }
    elif control not in ("null", "aligned", "shift_large"):
        raise ValueError("unknown S3D control: " + control)
    batch.batch_traj_inputs = np.ascontiguousarray(batch.batch_traj_inputs)
    return donor


def forward_model(model: RPNet, batch, control: str) -> dict:
    return model(
        aerial_image=to_cuda(batch.batch_inputs),
        traj_image=to_cuda(batch.batch_traj_inputs),
        aerial_traj_image=None, neighborhood_trajectory_norm=None,
        valid_mask=None, walked_path=to_cuda(batch.batch_walked_path_small),
        NUM_TARGETS=4, test=False, model="origin", use_traj=False,
        trajectory_mode=STRICT_MODE,
        traj_valid_mask=to_cuda(batch.batch_traj_valid_masks),
        segmentation_only=True,
        raster_adapter_bypass=control == "null",
    )


def build_batches(config: dict, split: dict, control: str):
    extent = split["validation_extent"]
    xyxy = [extent["x0"], extent["y0"], extent["x1"], extent["y1"]]
    dataset = OSMDataset(cfg_for_dataset(config, xyxy, control),
                         net=None, training=True)
    batches, identities, donors = [], [], []
    for index in range(int(config["S3"]["VALIDATION_BATCHES"])):
        batch = dataset.get_batch()
        identities.append(identity_sha256([
            sample_identity(row) for row in batch.batch_sample_metadata]))
        donor = apply_control(batch, control, index)
        if donor is not None:
            donors.append(donor)
        dataset.push_and_vis_batch(batch, 0, index)
        batches.append(batch)
    return batches, identity_sha256(identities), donors


def evaluate(model: RPNet, batches: list, control: str, threshold: float) -> dict:
    roads, junctions, road_targets, junction_targets, road_features = [], [], [], [], []
    raster_ratios = []
    model.eval()
    with torch.no_grad():
        for batch in batches:
            output = forward_model(model, batch, control)
            roads.append(output["road"].cpu().numpy())
            junctions.append(output["junc"].cpu().numpy())
            road_features.append(
                output["feature_maps"]["road_fts"].cpu().numpy())
            road_targets.append(batch.batch_road_segmentation)
            junction_targets.append(batch.batch_junction_segmentation)
            raster_ratios.extend((np.count_nonzero(
                batch.batch_traj_inputs, axis=(1, 2, 3)) /
                np.prod(batch.batch_traj_inputs.shape[1:])).tolist())
    road_logits = np.concatenate(roads)
    junction_logits = np.concatenate(junctions)
    road_target = np.concatenate(road_targets)
    junction_target = np.concatenate(junction_targets)
    road_feature = np.concatenate(road_features)
    road = binary_segmentation_metrics(road_logits, road_target,
                                       threshold=threshold)
    junction = binary_segmentation_metrics(junction_logits, junction_target,
                                           threshold=threshold)
    per_sample = []
    for index in range(len(road_logits)):
        rm = binary_segmentation_metrics(
            road_logits[index:index + 1], road_target[index:index + 1],
            threshold=threshold)
        jm = binary_segmentation_metrics(
            junction_logits[index:index + 1],
            junction_target[index:index + 1], threshold=threshold)
        per_sample.append({
            "sample_index": index, "road_precision": rm["precision"],
            "road_recall": rm["recall"], "road_f1": rm["f1"],
            "road_iou": rm["iou"], "road_auprc": rm["auprc"],
            "junction_f1": jm["f1"], "junction_auprc": jm["auprc"],
            "raster_positive_ratio": raster_ratios[index],
        })
    result = {
        "scope": "frozen_validation", "fixed_threshold": threshold,
        "validation_sample_count": len(road_logits),
        "road_precision": road["precision"], "road_recall": road["recall"],
        "road_f1": road["f1"], "road_iou": road["iou"],
        "road_auprc": road["auprc"],
        "junction_precision": junction["precision"],
        "junction_recall": junction["recall"],
        "junction_f1": junction["f1"], "junction_iou": junction["iou"],
        "junction_auprc": junction["auprc"],
        "per_sample": per_sample,
        "road_prediction_sha256": array_sha256([road_logits]),
        "junction_prediction_sha256": array_sha256([junction_logits]),
        "prediction_sha256": array_sha256([road_logits, junction_logits]),
        "road_feature_sha256": array_sha256([road_feature]),
        "raster_positive_ratio": {
            "min": min(raster_ratios), "mean": float(np.mean(raster_ratios)),
            "max": max(raster_ratios),
        },
    }
    result["road_composite"] = road_composite(result)
    set_road_only_train_mode(model)
    return result


def save_checkpoint(path: Path, model: RPNet, *, samples_seen: int,
                    optimizer_updates: int, code_sha: str,
                    config_sha: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-{}".format(os.getpid()))
    torch.save({
        "stage": "seg_raster_stage_s3d", "kind": "versioned_model_only",
        "code_sha": code_sha, "config_sha": config_sha,
        "samples_seen": samples_seen, "optimizer_updates": optimizer_updates,
        "state_dict": model.state_dict(),
    }, temporary)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-code-sha", required=True)
    parser.add_argument("--run-key", choices=("N0", "N1", "N2", "N3", "N4"),
                        required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--control", choices=(
        "null", "aligned", "zero", "shift_large", "permuted"),
        required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage S3D formal worker requires remote CUDA")
    frozen_checkout(args.run_code_sha)
    if os.environ.get("S3D_RUN_CODE_SHA") != args.run_code_sha:
        raise RuntimeError("launcher/worker run-code SHA mismatch")
    config = configure(args)
    set_seed(STAGE_S3D_SEED)
    run_dir = REPO_ROOT / config["S3"]["RUN_ROOT"] / args.run_id
    checkpoint_dir = run_dir / "checkpoints"
    evaluation_dir = run_dir / "evaluation"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.yml").write_text(
        yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    config_sha = identity_sha256(config)
    model = RPNet(
        num_targets=4, backbone_pretrained=False,
        enable_trajectory_modules=False,
        enable_zero_preserving_road_adapter=True,
        anchor_grad_to_seg=False)
    checkpoint = os.environ.get("S3D_BASELINE_CHECKPOINT")
    if not checkpoint:
        raise RuntimeError("S3D_BASELINE_CHECKPOINT is required")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    baseline_audit = strict_load_stage_s3d_baseline(model, payload)
    baseline_audit.update({
        "path_label": "${OFFICIAL_VECROAD_RELEASE}/data/ckpt/vecroad.pth.tar",
        "sha256": sha256_file(checkpoint), "provenance": "OFFICIAL_RELEASE",
    })
    parameter_contract = configure_road_only_training(model)
    model.cuda()
    set_road_only_train_mode(model)
    bn_initial = original_batch_norm_checksum(model)
    parameters = road_only_parameters(model)
    optimizer = torch.optim.Adam(
        parameters, lr=float(config["TRAIN"]["SOLVER"]["LEARNING_RATE"]),
        betas=(0.9, 0.99),
        weight_decay=float(config["TRAIN"]["SOLVER"]["WEIGHT_DECAY"]))
    optimizer.zero_grad(set_to_none=True)
    split = json.loads((REPO_ROOT / config["S3"]["SPLIT_MANIFEST"])
                       .read_text(encoding="utf-8"))
    sample_plan = json.loads((REPO_ROOT / config["S3"]["SAMPLE_PLAN"])
                             .read_text(encoding="utf-8"))
    if sample_plan.get("status") != "PASS":
        raise RuntimeError("Stage S3D requires PASS sample plan")
    expected_batches = sample_plan["micro_batches"]
    if len(expected_batches) != MAX_SAMPLES_SEEN // int(config["TRAIN"]["BATCH_SIZE"]):
        raise RuntimeError("sample plan length differs from budget")
    set_seed(STAGE_S3D_SEED + 1)
    validation_batches, validation_sha, validation_donors = build_batches(
        config, split, args.control)
    set_seed(STAGE_S3D_SEED)
    extent = split["train_extent"]
    train_extent = [extent["x0"], extent["y0"], extent["x1"], extent["y1"]]
    dataset = OSMDataset(cfg_for_dataset(config, train_extent, args.control),
                         net=None, training=True)
    counter = SampleBudgetCounter(
        micro_batch_size=int(config["TRAIN"]["BATCH_SIZE"]),
        accumulation_steps=int(config["S3"]["GRADIENT_ACCUMULATION"]))
    threshold = float(config["S3"]["FIXED_THRESHOLD"])
    validations, inventory, donors = {}, [], []
    first_identity, first_common, first_mask = [], [], []
    last_gradient_sha = trainable_gradient_sha256(model)
    started, last_heartbeat = time.time(), 0.0
    status, invalid_reason = "RUNNING", None

    def evaluate_and_save(samples_seen: int) -> None:
        before = original_batch_norm_checksum(model)
        metrics = evaluate(model, validation_batches, args.control, threshold)
        after = original_batch_norm_checksum(model)
        if before != after or after != bn_initial:
            raise RuntimeError("original BatchNorm checksum changed")
        metrics.update({
            "samples_seen": samples_seen,
            "optimizer_updates": counter.optimizer_updates,
            "shared_trainable_tensor_sha256": road_parameter_sha256(model),
            "gradient_sha256": last_gradient_sha,
        })
        destination = checkpoint_dir / checkpoint_name(samples_seen)
        save_checkpoint(destination, model, samples_seen=samples_seen,
                        optimizer_updates=counter.optimizer_updates,
                        code_sha=args.run_code_sha, config_sha=config_sha)
        checkpoint_sha = sha256_file(destination)
        metrics["checkpoint_sha256"] = checkpoint_sha
        validations[str(samples_seen)] = metrics
        write_json(evaluation_dir / "validation_samples_{:06d}.json".format(
            samples_seen), {"segmentation": metrics})
        append_jsonl(metrics_path, {
            "time": utc_now(), "kind": "validation",
            "samples_seen": samples_seen, "metrics": metrics})
        inventory.append({
            "samples_seen": samples_seen,
            "optimizer_updates": counter.optimizer_updates,
            "logical_path": "${S3D_RUN_ROOT}/" + args.run_id
                            + "/checkpoints/" + destination.name,
            "size_bytes": destination.stat().st_size,
            "sha256": checkpoint_sha, "optimizer_included": False,
        })

    try:
        evaluate_and_save(0)
        for batch_index, expected in enumerate(expected_batches):
            batch = dataset.get_batch()
            identity = identity_sha256([
                sample_identity(row) for row in batch.batch_sample_metadata])
            common = common_batch_sha(batch)
            if identity != expected["batch_identity_sha256"]:
                raise RuntimeError("sample identity mismatch at {}".format(batch_index))
            if common != expected["common_tensor_sha256"]:
                raise RuntimeError("common tensor mismatch at {}".format(batch_index))
            donor = apply_control(batch, args.control, batch_index)
            if donor is not None:
                donors.append(donor)
            if batch_index < 20:
                first_identity.append(identity)
                first_common.append(common)
                first_mask.append(array_sha256([batch.batch_traj_valid_masks]))
            output = forward_model(model, batch, args.control)
            if not bool(torch.isfinite(output["road"]).all()):
                raise FloatingPointError("non-finite road logits")
            if not bool(torch.isfinite(output["junc"]).all()):
                raise FloatingPointError("non-finite junction logits")
            loss = road_loss(output, to_cuda(batch.batch_road_segmentation))
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite road loss")
            loss.backward()
            dataset.push_and_vis_batch(batch, 0, batch_index)
            should_step = counter.record_micro_batch(
                int(config["TRAIN"]["BATCH_SIZE"]))
            if should_step:
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        if not parameter.requires_grad:
                            raise RuntimeError("frozen parameter gradient: " + name)
                        if not bool(torch.isfinite(parameter.grad).all()):
                            raise FloatingPointError("non-finite gradient: " + name)
                last_gradient_sha = trainable_gradient_sha256(model)
                torch.nn.utils.clip_grad_value_(parameters, 1e4)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                enforce_original_batch_norm_eval(model)
            if should_step and counter.samples_seen in SAMPLE_GRID:
                evaluate_and_save(counter.samples_seen)
            if should_step and counter.optimizer_updates % int(
                    config["S3"]["METRICS_INTERVAL"]) == 0:
                frozen_checkout(args.run_code_sha)
                append_jsonl(metrics_path, {
                    "time": utc_now(), "kind": "training",
                    "samples_seen": counter.samples_seen,
                    "optimizer_updates": counter.optimizer_updates,
                    "loss": float(loss.detach()),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / 2**20,
                    "gpu_memory_reserved_mb": torch.cuda.memory_reserved() / 2**20,
                })
            now = time.time()
            if now - last_heartbeat >= int(config["S3"]["HEARTBEAT_SECONDS"]):
                write_json(run_dir / "heartbeat.json", {
                    "status": "RUNNING", "time": utc_now(),
                    "samples_seen": counter.samples_seen,
                    "optimizer_updates": counter.optimizer_updates,
                    "code_sha": args.run_code_sha})
                last_heartbeat = now
        status = "PASS"
    except Exception as error:
        status = "INVALID"
        invalid_reason = "{}: {}".format(type(error).__name__, error)
        raise
    finally:
        summary = {
            "stage": "seg_raster_stage_s3d", "run_key": args.run_key,
            "run_id": args.run_id, "code_sha": args.run_code_sha,
            "config_sha": config_sha, "status": status,
            "invalid_reason": invalid_reason,
            "execution_environment": "REMOTE_TRAINING_SERVER",
            "seed": STAGE_S3D_SEED, "control": args.control,
            "optimizer_updates": counter.optimizer_updates,
            "micro_batches": counter.micro_batches,
            "samples_seen": counter.samples_seen,
            "baseline_checkpoint": baseline_audit,
            "sample_plan_sha256": sample_plan["plan_sha256"],
            "split_manifest_sha256": split["manifest_sha256"],
            "validation_plan_sha256": validation_sha,
            "validation_donor_mapping_sha256": identity_sha256(validation_donors),
            "training_donor_mapping_sha256": identity_sha256(donors),
            "first_20_batch_identity_sha256": identity_sha256(first_identity),
            "first_20_common_tensor_sha256": identity_sha256(first_common),
            "first_20_valid_mask_sha256": identity_sha256(first_mask),
            "original_bn_checksum_initial": bn_initial,
            "original_bn_checksum_final": original_batch_norm_checksum(model),
            "trainable_parameter_contract": parameter_contract,
            "validation_metrics_by_samples": validations,
            "checkpoint_inventory": inventory,
            "runtime_seconds": time.time() - started,
            "peak_allocated_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_memory_mb": torch.cuda.max_memory_reserved() / 2**20,
        }
        summary["original_bn_checksum_unchanged"] = (
            summary["original_bn_checksum_initial"]
            == summary["original_bn_checksum_final"])
        write_json(run_dir / "summary.json", summary)
        write_json(run_dir / "heartbeat.json", {
            "status": status, "time": utc_now(),
            "samples_seen": counter.samples_seen,
            "optimizer_updates": counter.optimizer_updates,
            "invalid_reason": invalid_reason})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
