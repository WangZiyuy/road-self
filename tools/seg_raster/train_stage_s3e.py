"""One-GPU worker for Stage S3E single-variable root-cause runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import yaml

from model.model import RPNet
from tools.seg_raster.train_stage_s3c import (
    append_jsonl, array_sha256, common_batch_sha, frozen_checkout,
    identity_sha256, sample_identity, set_seed, to_cuda, utc_now, write_json)
from tools.seg_raster.train_stage_s3d import (
    apply_control, build_batches, cfg_for_dataset, forward_model)
from utils.OSMDataset import OSMDataset
from utils.seg_raster.stage_s3 import (
    binary_segmentation_metrics, load_stage_s3_config, sha256_file)
from utils.seg_raster.stage_s3c import (
    MAX_SAMPLES_SEEN, SAMPLE_GRID, SampleBudgetCounter,
    enforce_original_batch_norm_eval, original_batch_norm_checksum)
from utils.seg_raster.stage_s3d import (
    STAGE_S3D_SEED, STRICT_MODE, road_composite, road_loss,
    strict_load_stage_s3d_baseline, trainable_gradient_sha256)
from utils.seg_raster.stage_s3e import (
    ADAPTER_PREFIX, ENCODER_PREFIX, PROJECTION_PREFIX, ROAD_HEAD_PREFIXES,
    clone_named_parameters, configure_stage_s3e_training, finite_tree,
    named_gradient_vector, optimizer_parameter_delta, tensor_map_sha256,
    weighted_road_loss)


RUN_PROFILES = {
    "Z0": {"control": "null", "projection_init": "default"},
    "Z1": {"control": "aligned", "projection_init": "default"},
    "Z2": {"control": "aligned", "projection_init": "zero"},
    "C1": {"control": "aligned", "projection_init": "zero",
           "road_head_lr": 0.0},
    "C2": {"control": "aligned", "projection_init": "zero",
           "use_support_multiplier": False},
    "C3": {"control": "aligned", "projection_init": "zero",
           "freeze_encoder": True},
    "C4": {"control": "aligned", "projection_init": "zero",
           "calibrated_loss": True},
}


def profile(run_key: str, base_lr: float, calibration: dict | None) -> dict:
    if run_key not in RUN_PROFILES:
        raise ValueError("unknown S3E run key: " + run_key)
    value = {
        "control": "aligned", "projection_init": "default",
        "use_support_multiplier": True, "road_head_lr": float(base_lr),
        "freeze_encoder": False, "negative_weight": 1.0,
        "loss_scale": 1.0,
    }
    value.update(RUN_PROFILES[run_key])
    if value.pop("calibrated_loss", False):
        if calibration is None or calibration.get("status") != "PASS":
            raise RuntimeError("C4 requires a PASS calibration manifest")
        value["negative_weight"] = float(calibration["negative_weight"])
        value["loss_scale"] = float(calibration["loss_scale"])
        value["calibrated_initial_gradient_ratio"] = float(
            calibration["verified"][
                "adapter_residual_negative_positive_gradient_mass_ratio"])
    return value


def configure(args: argparse.Namespace, run_profile: dict) -> dict:
    config = load_stage_s3_config(REPO_ROOT / "configs/stage_s3e_common.yml")
    sample_plan = os.environ.get("S3E_SAMPLE_PLAN")
    if not sample_plan:
        raise RuntimeError("S3E_SAMPLE_PLAN is required")
    config["S3"].update({
        "RUN_ID": args.run_id, "RUN_KEY": args.run_key,
        "CODE_SHA_REQUIRED": args.run_code_sha,
        "PHASE": "B" if args.run_key.startswith("Z") else "C",
        "SAMPLE_PLAN": sample_plan,
    })
    config["TRAJ"]["MODE"] = STRICT_MODE
    config["TRAJ"]["RASTER"]["CONTROL"] = run_profile["control"]
    config["S3E"]["RUN_PROFILE"] = run_profile
    return config


def model_for_profile(run_profile: dict) -> RPNet:
    return RPNet(
        num_targets=4, backbone_pretrained=False,
        enable_trajectory_modules=False,
        enable_zero_preserving_road_adapter=True,
        anchor_grad_to_seg=False,
        raster_projection_init=run_profile["projection_init"],
        raster_use_support_multiplier=run_profile["use_support_multiplier"])


def evaluate(model: RPNet, batches: list, control: str, threshold: float) -> dict:
    roads, targets, residual_sq, image_sq = [], [], 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for batch in batches:
            output = forward_model(model, batch, control)
            road = output["road"]
            target = to_cuda(batch.batch_road_segmentation).to(road.dtype)
            roads.append(road.cpu().numpy())
            targets.append(target.cpu().numpy())
            residual_sq += float(torch.square(
                output["feature_maps"]["strict_raster_residual"].float()).sum())
            image_sq += float(torch.square(
                output["feature_maps"]["stage_fuse_img"].float()).sum())
    logits = np.concatenate(roads)
    target = np.concatenate(targets)
    metric = binary_segmentation_metrics(logits, target, threshold=threshold)
    probability = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    positive = target > 0.5
    result = {
        "road_precision": metric["precision"],
        "road_recall": metric["recall"], "road_f1": metric["f1"],
        "road_iou": metric["iou"], "road_auprc": metric["auprc"],
        "gt_mean_probability": float(probability[positive].mean()),
        "background_mean_probability": float(probability[~positive].mean()),
        "residual_to_image_l2_ratio": float(
            np.sqrt(residual_sq) / max(np.sqrt(image_sq), 1e-30)),
        "road_prediction_sha256": array_sha256([logits]),
        "validation_sample_count": int(logits.shape[0]),
    }
    result["road_composite"] = road_composite(result)
    set_train_mode(model)
    return result


def set_train_mode(model: RPNet) -> None:
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def parameter_group_snapshot(model: RPNet) -> dict[str, str]:
    state = model.state_dict()
    return {
        "road_head": tensor_map_sha256(state, prefixes=ROAD_HEAD_PREFIXES),
        "adapter": tensor_map_sha256(state, prefixes=(ADAPTER_PREFIX,)),
        "encoder": tensor_map_sha256(state, prefixes=(ENCODER_PREFIX,)),
        "projection": tensor_map_sha256(state, prefixes=(PROJECTION_PREFIX,)),
    }


def gradient_group_snapshot(model: RPNet) -> dict[str, object]:
    rows = {}
    for key, prefixes in {
        "road_head": ROAD_HEAD_PREFIXES,
        "encoder": (ENCODER_PREFIX,),
        "projection": (PROJECTION_PREFIX,),
    }.items():
        vector, per_name = named_gradient_vector(model, prefixes)
        rows[key] = {
            "l2": float(torch.linalg.vector_norm(vector.double())),
            "sha256": tensor_map_sha256(per_name, prefixes=prefixes),
            "all_zero": bool(torch.count_nonzero(vector) == 0),
        }
    return rows


def diagnostic_gradient_snapshot(
    model: RPNet, batches: list, control: str, *,
    negative_weight: float, loss_scale: float,
) -> dict[str, object]:
    model.zero_grad(set_to_none=True)
    negative, positive = 0.0, 0.0
    total_loss = 0.0
    for batch in batches:
        output = forward_model(model, batch, control)
        residual = output["feature_maps"]["strict_raster_residual"]
        residual.retain_grad()
        target = to_cuda(batch.batch_road_segmentation).to(output["road"].dtype)
        if negative_weight == 1.0 and loss_scale == 1.0:
            loss = road_loss(output, target)
        else:
            loss = weighted_road_loss(
                output["road"], target, negative_weight=negative_weight,
                scale=loss_scale)
        loss.backward()
        expanded = (target > 0.5).expand_as(residual)
        gradient = residual.grad.detach().abs()
        positive += float(gradient[expanded].sum())
        negative += float(gradient[~expanded].sum())
        total_loss += float(loss.detach())
    groups = gradient_group_snapshot(model)
    model.zero_grad(set_to_none=True)
    return {
        "loss": total_loss,
        "parameter_groups": groups,
        "adapter_residual_negative_positive_gradient_mass_ratio":
            negative / max(positive, 1e-30),
    }


def diagnostic_snapshot(
    model: RPNet, batches: list, control: str, threshold: float,
    initial_parameters: dict[str, torch.Tensor], last_gradients: dict,
    optimizer_updates: int, samples_seen: int, *,
    negative_weight: float, loss_scale: float,
) -> dict:
    metrics = evaluate(model, batches, control, threshold)
    diagnostic_gradient = diagnostic_gradient_snapshot(
        model, batches, control, negative_weight=negative_weight,
        loss_scale=loss_scale)
    current = dict(model.named_parameters())
    road_initial = {name: value for name, value in initial_parameters.items()
                    if name.startswith(ROAD_HEAD_PREFIXES)}
    road_current = {name: current[name].detach() for name in road_initial}
    return {
        "optimizer_updates": optimizer_updates, "samples_seen": samples_seen,
        "metrics": metrics, "parameter_sha256": parameter_group_snapshot(model),
        "road_head_delta": optimizer_parameter_delta(
            road_initial, model, ROAD_HEAD_PREFIXES),
        "last_training_gradient": last_gradients,
        "diagnostic_gradient_field": diagnostic_gradient,
    }


def save_checkpoint(
    path: Path, model: RPNet, *, optimizer_updates: int,
    samples_seen: int, code_sha: str, config_sha: str,
) -> dict:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-{}".format(os.getpid()))
    torch.save({
        "stage": "seg_raster_stage_s3e", "kind": "versioned_model_only",
        "code_sha": code_sha, "config_sha": config_sha,
        "optimizer_updates": optimizer_updates, "samples_seen": samples_seen,
        "state_dict": model.state_dict(),
    }, temporary)
    temporary.replace(path)
    return {
        "optimizer_updates": optimizer_updates, "samples_seen": samples_seen,
        "logical_path": "${S3E_RUN_ROOT}/" + path.parent.parent.name
                        + "/checkpoints/" + path.name,
        "size_bytes": path.stat().st_size, "sha256": sha256_file(path),
        "optimizer_included": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-code-sha", required=True)
    parser.add_argument("--run-key", choices=tuple(RUN_PROFILES), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--calibration-manifest", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage S3E formal worker requires remote CUDA")
    frozen_checkout(args.run_code_sha)
    if os.environ.get("S3E_RUN_CODE_SHA") != args.run_code_sha:
        raise RuntimeError("launcher/worker S3E SHA mismatch")

    calibration = None
    if args.calibration_manifest:
        calibration = json.loads(args.calibration_manifest.read_text(encoding="utf-8"))
    base_config = load_stage_s3_config(REPO_ROOT / "configs/stage_s3e_common.yml")
    base_lr = float(base_config["TRAIN"]["SOLVER"]["LEARNING_RATE"])
    run_profile = profile(args.run_key, base_lr, calibration)
    config = configure(args, run_profile)
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

    model = model_for_profile(run_profile)
    baseline_path = os.environ.get("S3D_BASELINE_CHECKPOINT")
    if not baseline_path:
        raise RuntimeError("S3D_BASELINE_CHECKPOINT is required")
    payload = torch.load(baseline_path, map_location="cpu", weights_only=False)
    baseline_audit = strict_load_stage_s3d_baseline(model, payload)
    baseline_audit.update({
        "path_label": "${OFFICIAL_VECROAD_RELEASE}/data/ckpt/vecroad.pth.tar",
        "sha256": sha256_file(baseline_path), "provenance": "OFFICIAL_RELEASE",
    })
    contract, parameter_groups = configure_stage_s3e_training(
        model, road_head_lr=run_profile["road_head_lr"],
        freeze_encoder=run_profile["freeze_encoder"])
    model.cuda()
    set_train_mode(model)
    bn_initial = original_batch_norm_checksum(model)
    initial_parameters = clone_named_parameters(model)
    initial_checksums = parameter_group_snapshot(model)
    optimizer = torch.optim.Adam(
        parameter_groups, lr=base_lr, betas=(0.9, 0.99),
        weight_decay=float(config["TRAIN"]["SOLVER"]["WEIGHT_DECAY"]))
    optimizer.zero_grad(set_to_none=True)

    split = json.loads((REPO_ROOT / config["S3"]["SPLIT_MANIFEST"])
                       .read_text(encoding="utf-8"))
    sample_plan_path = Path(config["S3"]["SAMPLE_PLAN"])
    sample_plan = json.loads(sample_plan_path.read_text(encoding="utf-8"))
    if sample_plan.get("status") != "PASS":
        raise RuntimeError("Stage S3E requires PASS sample plan")
    expected_batches = sample_plan["micro_batches"]
    if len(expected_batches) != MAX_SAMPLES_SEEN // int(config["TRAIN"]["BATCH_SIZE"]):
        raise RuntimeError("sample plan length differs from S3E budget")

    set_seed(STAGE_S3D_SEED + 1)
    validation_batches, validation_sha, validation_donors = build_batches(
        config, split, run_profile["control"])
    diagnostic_batches = validation_batches[:int(
        config["S3"]["DIAGNOSTIC_VALIDATION_BATCHES"])]
    set_seed(STAGE_S3D_SEED)
    extent = split["train_extent"]
    train_extent = [extent["x0"], extent["y0"], extent["x1"], extent["y1"]]
    dataset = OSMDataset(cfg_for_dataset(
        config, train_extent, run_profile["control"]), net=None, training=True)
    counter = SampleBudgetCounter(
        micro_batch_size=int(config["TRAIN"]["BATCH_SIZE"]),
        accumulation_steps=int(config["S3"]["GRADIENT_ACCUMULATION"]))
    threshold = float(config["S3"]["FIXED_THRESHOLD"])
    dense_updates = set(map(int, config["S3E"]["DENSE_UPDATE_GRID"]))
    full_updates = set(map(int, config["S3E"]["FULL_VALIDATION_UPDATE_GRID"]))
    formal_update_grid = {int(samples) // int(config["S3"]["EFFECTIVE_BATCH_SIZE"])
                          for samples in SAMPLE_GRID}
    dense_updates |= formal_update_grid
    full_updates |= formal_update_grid
    validations, diagnostics, inventory = {}, {}, []
    first_identity, first_common, first_mask = [], [], []
    last_gradients = gradient_group_snapshot(model)
    first_step = None
    started, last_heartbeat = time.time(), 0.0
    status, invalid_reason = "RUNNING", None

    def save_dense(update: int) -> None:
        key = str(update)
        diagnostics[key] = diagnostic_snapshot(
            model, diagnostic_batches, run_profile["control"], threshold,
            initial_parameters, last_gradients, update, counter.samples_seen,
            negative_weight=run_profile["negative_weight"],
            loss_scale=run_profile["loss_scale"])
        destination = checkpoint_dir / "updates_{:06d}.pth.tar".format(update)
        inventory.append(save_checkpoint(
            destination, model, optimizer_updates=update,
            samples_seen=counter.samples_seen, code_sha=args.run_code_sha,
            config_sha=config_sha))
        write_json(evaluation_dir / "diagnostic_updates_{:06d}.json".format(update),
                   diagnostics[key])

    def full_validation(update: int) -> None:
        metrics = evaluate(
            model, validation_batches, run_profile["control"], threshold)
        metrics.update({"optimizer_updates": update,
                        "samples_seen": counter.samples_seen})
        validations[str(update)] = metrics
        write_json(evaluation_dir / "validation_updates_{:06d}.json".format(update),
                   {"segmentation": metrics})
        append_jsonl(metrics_path, {"time": utc_now(), "kind": "validation",
                                    "metrics": metrics})

    try:
        save_dense(0)
        full_validation(0)
        for batch_index, expected in enumerate(expected_batches):
            batch = dataset.get_batch()
            identity = identity_sha256([
                sample_identity(row) for row in batch.batch_sample_metadata])
            common = common_batch_sha(batch)
            if identity != expected["batch_identity_sha256"]:
                raise RuntimeError("sample identity mismatch at {}".format(batch_index))
            if common != expected["common_tensor_sha256"]:
                raise RuntimeError("common tensor mismatch at {}".format(batch_index))
            apply_control(batch, run_profile["control"], batch_index)
            if batch_index < 20:
                first_identity.append(identity)
                first_common.append(common)
                first_mask.append(array_sha256([batch.batch_traj_valid_masks]))
            output = forward_model(model, batch, run_profile["control"])
            target = to_cuda(batch.batch_road_segmentation).to(output["road"].dtype)
            if (run_profile["negative_weight"] == 1.0
                    and run_profile["loss_scale"] == 1.0):
                loss = road_loss(output, target)
            else:
                loss = weighted_road_loss(
                    output["road"], target,
                    negative_weight=run_profile["negative_weight"],
                    scale=run_profile["loss_scale"])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite S3E road loss")
            loss.backward()
            dataset.push_and_vis_batch(batch, 0, batch_index)
            should_step = counter.record_micro_batch(
                int(config["TRAIN"]["BATCH_SIZE"]))
            if should_step:
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None and not bool(
                            torch.isfinite(parameter.grad).all()):
                        raise FloatingPointError("non-finite gradient: " + name)
                last_gradients = gradient_group_snapshot(model)
                before_step = clone_named_parameters(model)
                torch.nn.utils.clip_grad_value_(
                    [p for group in optimizer.param_groups for p in group["params"]],
                    1e4)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                enforce_original_batch_norm_eval(model)
                if counter.optimizer_updates == 1:
                    first_step = {
                        "loss_gradient": last_gradients,
                        "road_head_optimizer_delta": optimizer_parameter_delta(
                            before_step, model, ROAD_HEAD_PREFIXES),
                        "encoder_total_optimizer_delta": optimizer_parameter_delta(
                            before_step, model, (ENCODER_PREFIX,)),
                        "projection_total_optimizer_delta": optimizer_parameter_delta(
                            before_step, model, (PROJECTION_PREFIX,)),
                        "weight_decay_separated_from_loss_gradient": True,
                    }
                    first_step["encoder_weight_decay_update"] = (
                        first_step["encoder_total_optimizer_delta"]
                        if last_gradients["encoder"]["all_zero"]
                        else {"status": "NOT_ISOLATABLE_NONZERO_LOSS_GRADIENT"})
                update = counter.optimizer_updates
                if update in dense_updates:
                    save_dense(update)
                if update in full_updates:
                    full_validation(update)
                if update % int(config["S3"]["METRICS_INTERVAL"]) == 0:
                    frozen_checkout(args.run_code_sha)
                    append_jsonl(metrics_path, {
                        "time": utc_now(), "kind": "training",
                        "optimizer_updates": update,
                        "samples_seen": counter.samples_seen,
                        "loss": float(loss.detach()),
                        "learning_rates": {str(group.get("name")): group["lr"]
                                           for group in optimizer.param_groups},
                        "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / 2**20,
                        "gpu_memory_reserved_mb": torch.cuda.memory_reserved() / 2**20,
                    })
            now = time.time()
            if now - last_heartbeat >= int(config["S3"]["HEARTBEAT_SECONDS"]):
                write_json(run_dir / "heartbeat.json", {
                    "status": "RUNNING", "time": utc_now(),
                    "optimizer_updates": counter.optimizer_updates,
                    "samples_seen": counter.samples_seen,
                    "code_sha": args.run_code_sha})
                last_heartbeat = now
        status = "PASS"
    except Exception as error:
        status = "INVALID"
        invalid_reason = "{}: {}".format(type(error).__name__, error)
        raise
    finally:
        final_controls = {}
        if counter.samples_seen == MAX_SAMPLES_SEEN:
            final_controls["null"] = evaluate(
                model, validation_batches, "null", threshold)
            final_controls["aligned"] = evaluate(
                model, validation_batches, "aligned", threshold)
        summary = {
            "stage": "seg_raster_stage_s3e", "run_key": args.run_key,
            "run_id": args.run_id, "code_sha": args.run_code_sha,
            "config_sha": config_sha, "status": status,
            "invalid_reason": invalid_reason,
            "execution_environment": "REMOTE_TRAINING_SERVER",
            "seed": STAGE_S3D_SEED, "run_profile": run_profile,
            "optimizer_updates": counter.optimizer_updates,
            "micro_batches": counter.micro_batches,
            "samples_seen": counter.samples_seen,
            "baseline_checkpoint": baseline_audit,
            "sample_plan_sha256": sample_plan["plan_sha256"],
            "split_manifest_sha256": split["manifest_sha256"],
            "validation_plan_sha256": validation_sha,
            "validation_donor_mapping_sha256": identity_sha256(validation_donors),
            "first_20_batch_identity_sha256": identity_sha256(first_identity),
            "first_20_common_tensor_sha256": identity_sha256(first_common),
            "first_20_valid_mask_sha256": identity_sha256(first_mask),
            "original_bn_checksum_initial": bn_initial,
            "original_bn_checksum_final": original_batch_norm_checksum(model),
            "initial_parameter_sha256": initial_checksums,
            "final_parameter_sha256": parameter_group_snapshot(model),
            "trainable_parameter_contract": contract,
            "dense_diagnostics_by_update": diagnostics,
            "validation_metrics_by_update": validations,
            "first_optimizer_step": first_step,
            "final_counterfactual_controls": final_controls,
            "checkpoint_inventory": inventory,
            "runtime_seconds": time.time() - started,
            "peak_allocated_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_memory_mb": torch.cuda.max_memory_reserved() / 2**20,
        }
        summary["original_bn_checksum_unchanged"] = (
            summary["original_bn_checksum_initial"]
            == summary["original_bn_checksum_final"])
        if args.run_key == "C1":
            summary["road_head_checksum_unchanged"] = (
                initial_checksums["road_head"]
                == summary["final_parameter_sha256"]["road_head"])
        if args.run_key == "C3":
            summary["encoder_checksum_unchanged"] = (
                initial_checksums["encoder"]
                == summary["final_parameter_sha256"]["encoder"])
        finite_tree(summary)
        write_json(run_dir / "summary.json", summary)
        write_json(run_dir / "heartbeat.json", {
            "status": status, "time": utc_now(),
            "optimizer_updates": counter.optimizer_updates,
            "samples_seen": counter.samples_seen,
            "invalid_reason": invalid_reason})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
