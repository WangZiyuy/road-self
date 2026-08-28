"""Re-evaluate one immutable Stage S3 checkpoint on the frozen validation plan.

This entry point is intended for REMOTE_TRAINING_SERVER CUDA execution.  It
never updates a checkpoint or optimizer and writes exactly one independent
JSON result, allowing a FIFO multi-GPU launcher to reduce results safely.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from tools.seg_raster.audit_stage_s3a_metrics import (
    anchor_reference_metrics,
    array_sha256,
    binary_reference_metrics,
    calibration_metrics,
    detect_double_sigmoid_input,
    finite_json_dumps,
    numeric_max_abs_difference,
    probability_forensics,
    sigmoid_once,
    threshold_sweep,
)
from tools.seg_raster.train_stage_s3 import (
    _cfg_for_dataset,
    _forward,
    _losses,
    _model_for,
    set_seed,
)
from utils.OSMDataset import OSMDataset
from utils.seg_raster.stage_s3 import (
    anchor_metrics,
    binary_segmentation_metrics,
    identity_sha256,
    load_stage_s3_config,
    sample_identity,
    sha256_file,
)


FORMAL_S3_SHA = "2e68f4e5a1c7cfad041182c2edce3194b8175b8c"


def _tracked_checkout(expected_sha: str) -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
        text=True, capture_output=True).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=no"], cwd=REPO_ROOT,
        check=True, text=True, capture_output=True).stdout.strip()
    if head != expected_sha or status:
        raise RuntimeError("S3A evaluator requires the clean audit-code checkout")


def _build_validation(config: dict, split: dict) -> tuple[list, list, str]:
    extent = split["validation_extent"]
    cfg = _cfg_for_dataset(config, [
        extent["x0"], extent["y0"], extent["x1"], extent["y1"]])
    dataset = OSMDataset(cfg, net=None, training=True)
    batches = []
    batch_hashes = []
    sample_rows = []
    for index in range(int(config["S3"]["VALIDATION_BATCHES"])):
        batch = dataset.get_batch()
        identities = [sample_identity(row) for row in batch.batch_sample_metadata]
        batch_hashes.append(identity_sha256(identities))
        for sample_index, (metadata, identity) in enumerate(zip(
                batch.batch_sample_metadata, identities)):
            sample_rows.append({
                "evaluation_index": len(sample_rows),
                "batch_index": index,
                "batch_member_index": sample_index,
                "sample_identity_sha256": identity,
                "region": metadata["region"],
                "crop_origin_xy": metadata["crop_origin_xy"],
                "extension_vertex_xy": metadata["extension_vertex_xy"],
                "end_index": int(metadata["end_index"]),
            })
        dataset.push_and_vis_batch(batch, 0, index)
        batches.append(batch)
    return batches, sample_rows, identity_sha256(batch_hashes)


def _metric_payload(existing: MappingLike, reference: MappingLike) -> dict:
    common = ("precision", "recall", "f1", "iou", "auprc")
    differences = {
        key: abs(float(existing[key]) - float(reference[key])) for key in common
    }
    return {
        "existing_evaluator": {key: existing[key] for key in existing},
        "numpy_reference": dict(reference),
        "absolute_differences": differences,
        "maximum_absolute_difference": max(differences.values(), default=0.0),
    }


MappingLike = dict[str, object]


def _per_sample_segmentation(
    road_logits: np.ndarray,
    road_target: np.ndarray,
    junction_logits: np.ndarray,
    junction_target: np.ndarray,
    sample_rows: list[dict],
    threshold: float,
) -> list[dict]:
    rows = []
    for index, sample in enumerate(sample_rows):
        road = binary_reference_metrics(
            road_logits[index:index + 1], road_target[index:index + 1],
            threshold=threshold)
        junction = binary_reference_metrics(
            junction_logits[index:index + 1], junction_target[index:index + 1],
            threshold=threshold)
        rows.append({
            **sample,
            "road": road,
            "junction": junction,
            "segmentation_composite": float(
                (road["f1"] + road["iou"] + junction["f1"]) / 3.0),
        })
    return rows


def _parameter_group(name: str) -> str:
    if name.startswith(("junc_seg.", "conv_junc_final.")):
        return "junction_head"
    if name.startswith(("road_seg.", "conv_road_final.")):
        return "road_head"
    if name.startswith("segmentation_raster_fusion."):
        return "raster_fusion"
    if name.startswith(("resnet.", "stage_", "fuse_")):
        return "image_backbone"
    return "anchor_or_other"


def _gradient_norms(model: torch.nn.Module) -> dict[str, float]:
    squared: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        group = _parameter_group(name)
        value = float(torch.sum(parameter.grad.detach().double() ** 2).cpu())
        squared[group] = squared.get(group, 0.0) + value
    return {key: math.sqrt(value) for key, value in sorted(squared.items())}


def _loss_and_gradient_audit(model, batch, mode: str) -> dict:
    model.train()
    output = _forward(model, batch, mode)
    losses = _losses(output, batch)
    order = ("road", "junction", "anchor", "total")
    gradient_norms = {}
    for index, key in enumerate(order):
        model.zero_grad(set_to_none=True)
        losses[key].backward(retain_graph=index != len(order) - 1)
        gradient_norms[key] = _gradient_norms(model)
    target = torch.from_numpy(np.asarray(batch.batch_junction_segmentation)).float().cuda()
    background_logits = torch.full_like(target, -80.0)
    pure_background = F.binary_cross_entropy_with_logits(
        background_logits, target, reduction="sum")
    one_correct_logits = background_logits.clone()
    positives = torch.nonzero(target > 0, as_tuple=False)
    if positives.numel():
        first = positives[0]
        one_correct_logits[tuple(first.tolist())] = 80.0
    one_correct = F.binary_cross_entropy_with_logits(
        one_correct_logits, target, reduction="sum")
    return {
        "mode": "model.train() forensic gradient pass; no optimizer step",
        "loss_reduction": "sum",
        "class_weighting": None,
        "positive_weighting": None,
        "loss_values": {key: float(value.detach().cpu()) for key, value in losses.items()},
        "gradient_l2_norm_by_parameter_group": gradient_norms,
        "junction_target_positive_count": int(torch.count_nonzero(target).cpu()),
        "junction_target_negative_count": int(target.numel() - torch.count_nonzero(target).cpu()),
        "junction_pure_background_logit_minus_80_loss_sum": float(pure_background.cpu()),
        "junction_one_positive_corrected_logit_plus_80_loss_sum": float(one_correct.cpu()),
        "one_correct_positive_loss_reduction": float((pure_background - one_correct).cpu()),
    }


def evaluate(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("formal S3A checkpoint evaluation requires CUDA")
    _tracked_checkout(args.audit_code_sha)
    started = time.monotonic()
    set_seed(20260828)
    config = load_stage_s3_config(args.config)
    split = json.loads((REPO_ROOT / "artifacts/stage_s3_split_manifest.json").read_text(
        encoding="utf-8"))
    set_seed(int(config["S3"]["SEED"]) + 1)
    batches, sample_rows, validation_plan_sha = _build_validation(config, split)
    checkpoint_sha = sha256_file(args.checkpoint)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("stage") != "seg_raster_stage_s3":
        raise RuntimeError("not a Stage S3 checkpoint")
    if payload.get("code_sha") != FORMAL_S3_SHA:
        raise RuntimeError("checkpoint code SHA differs from formal Stage S3 SHA")
    model = _model_for(config)
    incompatible = model.load_state_dict(payload["state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict checkpoint loading failed")
    model.cuda().eval()
    road_logits, road_targets = [], []
    junc_logits, junc_targets = [], []
    anchor_logits, anchor_targets, end_indices = [], [], []
    with torch.no_grad():
        for batch in batches:
            output = _forward(model, batch, config["TRAJ"]["MODE"])
            road_logits.append(output["road"].cpu().numpy())
            road_targets.append(np.asarray(batch.batch_road_segmentation))
            junc_logits.append(output["junc"].cpu().numpy())
            junc_targets.append(np.asarray(batch.batch_junction_segmentation))
            anchor_logits.append(output["anchor"].cpu().numpy())
            anchor_targets.append(np.asarray(batch.batch_target_maps))
            end_indices.extend(int(value) for value in batch.batch_end_index)
    road_logits_np = np.concatenate(road_logits)
    road_targets_np = np.concatenate(road_targets)
    junc_logits_np = np.concatenate(junc_logits)
    junc_targets_np = np.concatenate(junc_targets)
    anchor_logits_np = np.concatenate(anchor_logits)
    anchor_targets_np = np.concatenate(anchor_targets)
    threshold = float(config["S3"]["FIXED_THRESHOLD"])
    road_existing = binary_segmentation_metrics(
        road_logits_np, road_targets_np, threshold=threshold)
    junc_existing = binary_segmentation_metrics(
        junc_logits_np, junc_targets_np, threshold=threshold)
    road_reference = binary_reference_metrics(
        road_logits_np, road_targets_np, threshold=threshold)
    junc_reference = binary_reference_metrics(
        junc_logits_np, junc_targets_np, threshold=threshold)
    anchor_existing = anchor_metrics(
        anchor_logits_np, anchor_targets_np, end_indices, threshold=threshold)
    anchor_reference, per_target = anchor_reference_metrics(
        anchor_logits_np, anchor_targets_np, end_indices, threshold=threshold)
    loss_audit = _loss_and_gradient_audit(
        model, batches[0], config["TRAJ"]["MODE"])
    model.eval()
    label_counts = [int(np.count_nonzero(value)) for value in junc_targets_np]
    metric_script = Path(__file__).with_name("audit_stage_s3a_metrics.py")
    metric_code_sha = hashlib.sha256(
        Path(__file__).read_bytes() + metric_script.read_bytes()).hexdigest()
    return {
        "stage": "seg_raster_stage_s3a",
        "status": "PASS",
        "execution_environment": "REMOTE_TRAINING_SERVER",
        "audit_code_sha": args.audit_code_sha,
        "formal_s3_run_code_sha": FORMAL_S3_SHA,
        "run_key": args.run_key,
        "run_id": args.run_id,
        "checkpoint_kind": args.checkpoint_kind,
        "checkpoint_step": int(payload["optimizer_step"]),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_size_bytes": args.checkpoint.stat().st_size,
        "repeat_index": int(args.repeat_index),
        "validation_plan_sha": validation_plan_sha,
        "sample_identity_sha": identity_sha256([
            row["sample_identity_sha256"] for row in sample_rows]),
        "metric_code_sha": metric_code_sha,
        "model_mode": "eval",
        "torch_no_grad": True,
        "sigmoid_execution": "exactly once in metric boundary",
        "fixed_threshold": threshold,
        "validation_batch_count": len(batches),
        "validation_sample_count": len(sample_rows),
        "road_metric_reference_check": _metric_payload(road_existing, road_reference),
        "junction_metric_reference_check": _metric_payload(junc_existing, junc_reference),
        "anchor_metric_reference_check": {
            "existing_evaluator": anchor_existing,
            "numpy_reference": anchor_reference,
            "maximum_absolute_difference": numeric_max_abs_difference(
                anchor_existing, anchor_reference),
        },
        "segmentation": {
            "road": road_reference,
            "junction": junc_reference,
            "segmentation_composite": float(
                (road_reference["f1"] + road_reference["iou"]
                 + junc_reference["f1"]) / 3.0),
        },
        "junction_forensics": {
            **probability_forensics(junc_logits_np, junc_targets_np),
            "calibration": calibration_metrics(junc_logits_np, junc_targets_np),
            "threshold_sweep": threshold_sweep(junc_logits_np, junc_targets_np),
            "per_sample_target_positive_pixel_count": label_counts,
            "samples_without_positive_target": int(sum(value == 0 for value in label_counts)),
            "label_shape": [int(value) for value in junc_targets_np.shape],
            "logit_input_double_sigmoid_check": detect_double_sigmoid_input(junc_logits_np),
        },
        "anchor": anchor_reference,
        "anchor_per_target": per_target,
        "per_sample_segmentation": _per_sample_segmentation(
            road_logits_np, road_targets_np, junc_logits_np, junc_targets_np,
            sample_rows, threshold),
        "loss_balance": loss_audit,
        "checksums": {
            "road_logits": array_sha256(road_logits_np),
            "road_probability": array_sha256(sigmoid_once(road_logits_np)),
            "road_target": array_sha256(road_targets_np),
            "junction_logits": array_sha256(junc_logits_np),
            "junction_probability": array_sha256(sigmoid_once(junc_logits_np)),
            "junction_target": array_sha256(junc_targets_np),
            "anchor_logits": array_sha256(anchor_logits_np),
            "anchor_probability": array_sha256(sigmoid_once(anchor_logits_np)),
            "anchor_target": array_sha256(anchor_targets_np),
        },
        "sample_order": sample_rows,
        "evaluation_time_seconds": float(time.monotonic() - started),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-key", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-kind", choices=("best", "latest"), required=True)
    parser.add_argument("--repeat-index", type=int, default=0)
    parser.add_argument("--audit-code-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(finite_json_dumps(result), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({
        "status": result["status"],
        "run_key": result["run_key"],
        "checkpoint_kind": result["checkpoint_kind"],
        "checkpoint_step": result["checkpoint_step"],
    }, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
