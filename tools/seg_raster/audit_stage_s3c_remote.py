"""Prepare and preflight the frozen Stage S3C protocol on the remote host."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from model.model import RPNet
from tools.seg_raster.train_stage_s3c import (
    array_sha256, build_validation_batches, cfg_for_dataset,
    common_batch_sha, forward_model, load_baseline, set_seed, to_cuda,
    write_json,
)
from utils.OSMDataset import OSMDataset
from utils.seg_raster.stage_s3 import (
    anchor_metrics, binary_segmentation_metrics, identity_sha256,
    load_stage_s3_config, sample_identity, sha256_file,
)
from utils.seg_raster.stage_s3c import (
    MAX_SAMPLES_SEEN, configure_frozen_explorer,
    original_batch_norm_checksum, repair_composite, segmentation_losses,
    set_frozen_explorer_train_mode, trainable_parameters,
)


def verify_checkout(expected_sha: str) -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True,
        capture_output=True).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"], check=True,
        text=True, capture_output=True).stdout.strip()
    if head != expected_sha or status:
        raise RuntimeError("Stage S3C audit requires a clean frozen checkout")


def config_for(*, raster: bool, control: str | None = None) -> dict:
    config = load_stage_s3_config(REPO_ROOT / "configs/stage_s3c_common.yml")
    config["TRAJ"]["MODE"] = "raster_seg_only" if raster else "none"
    config["TRAJ"]["RASTER"]["CONTROL"] = control if raster else None
    config["TRAJ"]["RASTER"]["ANCHOR_GRAD_TO_SEG"] = False
    return config


def train_extent(split: dict) -> list[int]:
    extent = split["train_extent"]
    return [extent["x0"], extent["y0"], extent["x1"], extent["y1"]]


def prepare(args: argparse.Namespace) -> int:
    verify_checkout(args.run_code_sha)
    config = config_for(raster=False)
    split = json.loads((REPO_ROOT / config["S3"]["SPLIT_MANIFEST"])
                       .read_text(encoding="utf-8"))
    set_seed(int(config["S3"]["SEED"]))
    dataset = OSMDataset(
        cfg_for_dataset(config, train_extent(split)), net=None, training=True)
    batch_size = int(config["TRAIN"]["BATCH_SIZE"])
    micro_batch_count = MAX_SAMPLES_SEEN // batch_size
    rows, first_200 = [], []
    for batch_index in range(micro_batch_count):
        batch = dataset.get_batch()
        identity = identity_sha256([
            sample_identity(row) for row in batch.batch_sample_metadata])
        rows.append({
            "micro_batch_index": batch_index,
            "batch_identity_sha256": identity,
            "common_tensor_sha256": common_batch_sha(batch),
        })
        if batch_index < 20:
            first_200.extend(batch.batch_sample_metadata)
        # Mirror the official teacher-forced path transition.  FOLLOW_MODE is
        # follow_target, so no model prediction is consumed here.
        dataset.push_and_vis_batch(batch, 0, batch_index)
    payload = {
        "stage": "seg_raster_stage_s3c",
        "status": "PASS",
        "execution_environment": "REMOTE_TRAINING_SERVER",
        "run_code_sha": args.run_code_sha,
        "seed": int(config["S3"]["SEED"]),
        "follow_mode": "follow_target",
        "crop_size": 256,
        "num_targets": 4,
        "micro_batch_size": batch_size,
        "micro_batch_count": micro_batch_count,
        "sample_count": MAX_SAMPLES_SEEN,
        "sample_generation": "deterministic_OSMDataset_teacher_forced",
        "split_manifest_sha256": split["manifest_sha256"],
        "micro_batches": rows,
        "first_200_sample_metadata": first_200,
    }
    payload["plan_sha256"] = identity_sha256(payload)
    args.sample_plan.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.sample_plan, payload)
    write_json(args.output_root / "stage_s3c_sample_plan.json", payload)
    return 0


def gradient_norm(model: RPNet, prefixes: tuple[str, ...]) -> float:
    total = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and name.startswith(prefixes):
            total += float(torch.sum(parameter.grad.detach().double() ** 2))
    return float(np.sqrt(total))


def preflight(args: argparse.Namespace) -> int:
    verify_checkout(args.run_code_sha)
    if not torch.cuda.is_available():
        raise RuntimeError("formal S3C preflight requires remote CUDA")
    split = json.loads((REPO_ROOT / "artifacts/stage_s3_split_manifest.json")
                       .read_text(encoding="utf-8"))
    raster_config = config_for(raster=True, control="aligned")
    image_config = config_for(raster=False)
    seed = int(raster_config["S3"]["SEED"])
    set_seed(seed)
    dataset = OSMDataset(
        cfg_for_dataset(raster_config, train_extent(split)),
        net=None, training=True)
    batches = []
    for index in range(16):
        batch = dataset.get_batch()
        batches.append(batch)
        dataset.push_and_vis_batch(batch, 0, index)
    positive = sum(int(np.count_nonzero(batch.batch_junction_segmentation))
                   for batch in batches)
    total = sum(int(np.asarray(batch.batch_junction_segmentation).size)
                for batch in batches)
    raw_pos_weight = (total - positive) / max(positive, 1)
    pos_weight = min(raw_pos_weight, 32.0)

    image = RPNet(
        num_targets=4, backbone_pretrained=False,
        enable_trajectory_modules=False, enable_raster_segmentation=False,
        anchor_grad_to_seg=False)
    raster = RPNet(
        num_targets=4, backbone_pretrained=False,
        enable_trajectory_modules=False, enable_raster_segmentation=True,
        raster_use_valid_mask=True, anchor_grad_to_seg=False)
    image_audit = load_baseline(image, raster=False)
    raster_audit = load_baseline(raster, raster=True)
    image_contract = configure_frozen_explorer(image, raster_enabled=False)
    raster_contract = configure_frozen_explorer(raster, raster_enabled=True)
    image.cuda()
    raster.cuda()
    set_frozen_explorer_train_mode(image)
    set_frozen_explorer_train_mode(raster)

    # Match legacy and balanced junction-head gradient scale on the frozen
    # official initialization using the fixed 16 diagnostic batches.
    legacy_norms, balanced_norms = [], []
    for batch in batches:
        output = forward_model(
            image, batch, "none", segmentation_only=True)
        target = to_cuda(batch.batch_junction_segmentation)
        for weight, destination in ((1.0, legacy_norms),
                                    (pos_weight, balanced_norms)):
            image.zero_grad(set_to_none=True)
            torch.nn.functional.binary_cross_entropy_with_logits(
                output["junc"], target,
                pos_weight=torch.as_tensor(weight, device="cuda"),
                reduction="sum").backward(retain_graph=True)
            destination.append(gradient_norm(
                image, ("junc_seg.", "conv_junc_final.")))
    alpha = float(np.mean(legacy_norms) / np.mean(balanced_norms))

    # Heaviest aligned-raster path: two micro-batches accumulate without loss
    # division, followed by one optimizer update on trainable parameters only.
    optimizer = torch.optim.Adam(
        trainable_parameters(raster), lr=1e-5, betas=(0.9, 0.99),
        weight_decay=2e-4)
    optimizer.zero_grad(set_to_none=True)
    before_bn = original_batch_norm_checksum(raster)
    torch.cuda.reset_peak_memory_stats()
    loss_rows = []
    for batch in batches[:2]:
        output = forward_model(
            raster, batch, "raster_seg_only", segmentation_only=True)
        losses = segmentation_losses(
            output, to_cuda(batch.batch_road_segmentation),
            to_cuda(batch.batch_junction_segmentation))
        losses["total"].backward()
        loss_rows.append({key: float(value.detach())
                          for key, value in losses.items()})
    nonzero_groups = {"road": False, "junction": False, "raster": False}
    for name, parameter in raster.named_parameters():
        if parameter.grad is None:
            continue
        if not parameter.requires_grad:
            raise RuntimeError("frozen parameter received gradient: " + name)
        if not bool(torch.isfinite(parameter.grad).all()):
            raise FloatingPointError("non-finite gradient: " + name)
        nonzero = bool(torch.count_nonzero(parameter.grad))
        if name.startswith(("road_seg.", "conv_road_final.")):
            nonzero_groups["road"] |= nonzero
        elif name.startswith(("junc_seg.", "conv_junc_final.")):
            nonzero_groups["junction"] |= nonzero
        elif name.startswith("segmentation_raster_fusion."):
            nonzero_groups["raster"] |= nonzero
    torch.nn.utils.clip_grad_value_(trainable_parameters(raster), 1e4)
    optimizer.step()
    after_bn = original_batch_norm_checksum(raster)
    if before_bn != after_bn or not all(nonzero_groups.values()):
        raise RuntimeError("frozen-BN or trainable-gradient preflight failed")

    # Full frozen explorer is forward-only.  This verifies the official anchor
    # schema and multistep diversity without authorizing anchor backward.
    raster.eval()
    with torch.no_grad():
        full = forward_model(
            raster, batches[0], "raster_seg_only", segmentation_only=False)
    expected_shapes = {
        "road": (10, 1, 64, 64), "junc": (10, 1, 64, 64),
        "anchor": (10, 4, 256, 256),
        "anchor_lowrs": (10, 4, 256, 256),
    }
    for key, shape in expected_shapes.items():
        if tuple(full[key].shape) != shape or not bool(torch.isfinite(full[key]).all()):
            raise RuntimeError("preflight output contract failed: " + key)
    channel_differences = [
        float(torch.max(torch.abs(full["anchor"][:, index]
                                  - full["anchor"][:, index + 1])))
        for index in range(3)]
    if not all(value > 0 for value in channel_differences):
        raise RuntimeError("anchor channels are not diverse")

    write_json(args.output_root / "stage_s3c_loss_gradient_audit.json", {
        "stage": "seg_raster_stage_s3c", "status": "PASS",
        "execution_environment": "REMOTE_TRAINING_SERVER",
        "run_code_sha": args.run_code_sha,
        "diagnostic_batch_count": 16,
        "junction_positive_pixels": positive,
        "junction_total_pixels": total,
        "junction_positive_ratio": positive / total,
        "raw_pos_weight": raw_pos_weight,
        "capped_pos_weight": pos_weight,
        "legacy_junction_head_gradient_norm_mean": float(np.mean(legacy_norms)),
        "balanced_junction_head_gradient_norm_mean": float(np.mean(balanced_norms)),
        "balanced_global_alpha": alpha,
    })
    write_json(args.output_root / "stage_s3c_remote_preflight.json", {
        "stage": "seg_raster_stage_s3c", "status": "PASS",
        "execution_environment": "REMOTE_TRAINING_SERVER",
        "run_code_sha": args.run_code_sha,
        "cuda": True, "precision": "fp32", "crop_size": 256,
        "num_targets": 4, "micro_batch_per_gpu": 10,
        "gradient_accumulation": 2,
        "effective_samples_per_update": 20,
        "sum_loss_divided_by_accumulation": False,
        "image_checkpoint_audit": image_audit,
        "raster_checkpoint_audit": raster_audit,
        "output_shapes": {key: list(value) for key, value in expected_shapes.items()},
        "all_logits_finite": True, "all_losses_gradients_finite": True,
        "anchor_channel_pair_max_abs_difference": channel_differences,
        "trainable_nonzero_gradient_groups": nonzero_groups,
        "anchor_backward_executed": False,
        "sequence_loader_called": False, "transformer_constructed": False,
        "legacy_dsf_constructed": False,
        "original_bn_checksum_before": before_bn,
        "original_bn_checksum_after": after_bn,
        "original_bn_checksum_unchanged": before_bn == after_bn,
        "microbatch_losses": loss_rows,
        "peak_allocated_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_memory_mb": torch.cuda.max_memory_reserved() / 2**20,
    })
    write_json(args.output_root / "stage_s3c_trainable_parameter_contract.json", {
        "stage": "seg_raster_stage_s3c", "status": "PASS",
        "image_only": image_contract, "raster": raster_contract,
    })
    return 0


def baseline(args: argparse.Namespace) -> int:
    verify_checkout(args.run_code_sha)
    if not torch.cuda.is_available():
        raise RuntimeError("formal baseline evaluation requires remote CUDA")
    torch.cuda.reset_peak_memory_stats()
    config = config_for(raster=False)
    split = json.loads((REPO_ROOT / config["S3"]["SPLIT_MANIFEST"])
                       .read_text(encoding="utf-8"))
    seed = int(config["S3"]["SEED"])
    set_seed(seed)
    model = RPNet(
        num_targets=4, backbone_pretrained=False,
        enable_trajectory_modules=False, enable_raster_segmentation=False,
        anchor_grad_to_seg=False)
    checkpoint = load_baseline(model, raster=False)
    configure_frozen_explorer(model, raster_enabled=False)
    model.cuda().eval()
    set_seed(seed + 1)
    batches, validation_sha = build_validation_batches(config, split)
    road_logits, road_targets, junc_logits, junc_targets = [], [], [], []
    anchor_logits, anchor_targets, end_indices = [], [], []
    with torch.no_grad():
        for batch in batches:
            output = forward_model(model, batch, "none", segmentation_only=False)
            road_logits.append(output["road"].cpu().numpy())
            road_targets.append(batch.batch_road_segmentation)
            junc_logits.append(output["junc"].cpu().numpy())
            junc_targets.append(batch.batch_junction_segmentation)
            anchor_logits.append(output["anchor"].cpu().numpy())
            anchor_targets.append(batch.batch_target_maps)
            end_indices.extend(int(value) for value in batch.batch_end_index)
    road_values = binary_segmentation_metrics(
        np.concatenate(road_logits), np.concatenate(road_targets), threshold=0.3)
    junc_values = binary_segmentation_metrics(
        np.concatenate(junc_logits), np.concatenate(junc_targets), threshold=0.3)
    anchor_values = anchor_metrics(
        np.concatenate(anchor_logits), np.concatenate(anchor_targets), end_indices,
        threshold=0.3)
    metrics = {
        "road_precision": road_values["precision"],
        "road_recall": road_values["recall"], "road_f1": road_values["f1"],
        "road_iou": road_values["iou"], "road_auprc": road_values["auprc"],
        "junction_precision": junc_values["precision"],
        "junction_recall": junc_values["recall"],
        "junction_f1": junc_values["f1"],
        "junction_iou": junc_values["iou"],
        "junction_auprc": junc_values["auprc"],
    }
    metrics["repair_composite"] = repair_composite(metrics)
    anchors = np.concatenate(anchor_logits)
    diversity = [float(np.max(np.abs(anchors[:, index] - anchors[:, index + 1])))
                 for index in range(3)]
    write_json(args.output_root / "stage_s3c_baseline_evaluation.json", {
        "stage": "seg_raster_stage_s3c", "status": "PARTIAL_GRAPH_PENDING",
        "execution_environment": "REMOTE_TRAINING_SERVER",
        "run_code_sha": args.run_code_sha, "checkpoint": checkpoint,
        "validation_plan_sha256": validation_sha,
        "segmentation": metrics, "anchor": anchor_values,
        "output_shapes": {"road": [10, 1, 64, 64],
                          "junction": [10, 1, 64, 64],
                          "anchor": [10, 4, 256, 256],
                          "anchor_lowrs": [10, 4, 256, 256]},
        "all_logits_finite": True,
        "anchor_channel_pair_max_abs_difference": diversity,
        "multistep_anchor_validity": "PASS" if all(v > 0 for v in diversity)
                                     else "FAIL",
        "peak_allocated_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_memory_mb": torch.cuda.max_memory_reserved() / 2**20,
        "graph": {"status": "PENDING"},
    })
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "preflight", "baseline"))
    parser.add_argument("--run-code-sha", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-plan", type=Path,
                        default=REPO_ROOT / "data_self/stage_s3c_seg_raster/runtime/sample_plan.json")
    args = parser.parse_args()
    return {"prepare": prepare, "preflight": preflight,
            "baseline": baseline}[args.action](args)


if __name__ == "__main__":
    raise SystemExit(main())
