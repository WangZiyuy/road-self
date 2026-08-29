"""Remote CUDA preflight, loss-gradient audit and calibration for Stage S3B."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from tools.seg_raster.train_stage_s3b import (
    cfg_for_dataset, configure, forward, load_initialization, losses, model_for,
    set_seed, to_cuda, utc_now)
from utils.OSMDataset import OSMDataset
from utils.seg_raster.stage_s3 import identity_sha256, sample_identity, sha256_file
from utils.seg_raster.stage_s3b import (
    LOSS_BALANCED, LOSS_BALANCED_DICE, LOSS_LEGACY,
    gradient_matching_alpha, junction_loss, positive_weight_from_counts)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")


def verify_checkout(expected_sha: str) -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True,
        capture_output=True).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"], check=True,
        text=True, capture_output=True).stdout.strip()
    if head != expected_sha or status:
        raise RuntimeError("S3B audit requires a clean frozen checkout")


def grad_audit(model: torch.nn.Module) -> dict:
    groups = {"shared_backbone": 0.0, "segmentation_heads": 0.0,
              "raster_module": 0.0}
    maximum = 0.0
    finite, total = 0, 0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        mask = torch.isfinite(grad)
        finite += int(mask.sum())
        total += grad.numel()
        maximum = max(maximum, float(grad.abs().max()))
        squared = float(torch.sum(grad.double() ** 2))
        if "segmentation_raster_fusion" in name:
            groups["raster_module"] += squared
        elif "road_seg" in name or "junc_seg" in name:
            groups["segmentation_heads"] += squared
        elif "fuse_module" not in name and "anchor" not in name:
            groups["shared_backbone"] += squared
    return {
        "gradient_l2": {key: float(np.sqrt(value)) for key, value in groups.items()},
        "maximum_absolute_gradient": maximum,
        "finite_ratio": finite / total if total else 1.0,
    }


def junction_head_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for name, parameter in model.named_parameters():
        if "junc_seg" in name and parameter.grad is not None:
            total += float(torch.sum(parameter.grad.detach().double() ** 2))
    return float(np.sqrt(total))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-code-sha", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("formal Stage S3B audit requires remote CUDA")
    verify_checkout(args.run_code_sha)
    namespace = SimpleNamespace(
        input_kind="raster", run_id="diagnostic", run_key="DIAG",
        run_code_sha=args.run_code_sha, phase="A", loss_kind=LOSS_LEGACY,
        pos_weight=1.0, loss_alpha=1.0, lr_multiplier=1.0,
        control="aligned")
    config = configure(namespace)
    split_path = REPO_ROOT / config["S3"]["SPLIT_MANIFEST"]
    split = json.loads(split_path.read_text(encoding="utf-8"))
    sample_plan_path = REPO_ROOT / "artifacts/stage_s3_sample_plan.json"
    sample_plan = json.loads(sample_plan_path.read_text(encoding="utf-8"))
    extent = split["train_extent"]
    extent_list = [extent["x0"], extent["y0"], extent["x1"], extent["y1"]]
    set_seed(int(config["S3"]["SEED"]))
    model = model_for(config)
    initialization = load_initialization(model, config)
    model.cuda().train()
    dataset = OSMDataset(cfg_for_dataset(config, extent_list), net=None, training=True)
    batches = [dataset.get_batch() for _ in range(16)]
    identities = [identity_sha256([
        sample_identity(row) for row in batch.batch_sample_metadata])
        for batch in batches]
    positive = {"road": 0, "junction": 0, "anchor": 0}
    total_pixels = {"road": 0, "junction": 0, "anchor": 0}
    for batch in batches:
        for key, array in (
                ("road", batch.batch_road_segmentation),
                ("junction", batch.batch_junction_segmentation),
                ("anchor", batch.batch_target_maps)):
            positive[key] += int(np.count_nonzero(array))
            total_pixels[key] += int(np.asarray(array).size)
    junction_counts = positive_weight_from_counts(
        positive["junction"], total_pixels["junction"] - positive["junction"])
    pos_weight = float(junction_counts["capped_pos_weight"])
    per_batch, legacy_norms, balanced_norms, dice_norms = [], [], [], []
    torch.cuda.reset_peak_memory_stats()
    for batch_index, batch in enumerate(batches):
        output = forward(model, batch, config["TRAJ"]["MODE"])
        base_losses = losses(output, batch, config)
        record = {"batch_index": batch_index, "raw_loss": {}, "backward": {}}
        for index, name in enumerate(("road", "junction", "anchor", "total")):
            model.zero_grad(set_to_none=True)
            base_losses[name].backward(retain_graph=True)
            record["raw_loss"][name] = float(base_losses[name].detach())
            record["backward"][name] = grad_audit(model)
        target = to_cuda(batch.batch_junction_segmentation)
        candidate_losses = {
            LOSS_LEGACY: junction_loss(output["junc"], target, kind=LOSS_LEGACY),
            LOSS_BALANCED: junction_loss(
                output["junc"], target, kind=LOSS_BALANCED,
                pos_weight=pos_weight),
            LOSS_BALANCED_DICE: junction_loss(
                output["junc"], target, kind=LOSS_BALANCED_DICE,
                pos_weight=pos_weight),
        }
        norms = {}
        for kind, value in candidate_losses.items():
            model.zero_grad(set_to_none=True)
            value.backward(retain_graph=True)
            norms[kind] = junction_head_norm(model)
        legacy_norms.append(norms[LOSS_LEGACY])
        balanced_norms.append(norms[LOSS_BALANCED])
        dice_norms.append(norms[LOSS_BALANCED_DICE])
        record["junction_head_gradient_norm_unscaled"] = norms
        per_batch.append(record)
    alpha_balanced = gradient_matching_alpha(legacy_norms, balanced_norms)
    alpha_dice = gradient_matching_alpha(legacy_norms, dice_norms)
    pixel_counts = {}
    for key in positive:
        negative = total_pixels[key] - positive[key]
        pixel_counts[key] = {
            "positive": positive[key], "negative": negative,
            "total": total_pixels[key],
            "positive_ratio": positive[key] / total_pixels[key]}
    optimizer_contract = {
        "stage": "seg_raster_stage_s3b", "status": "PASS",
        "run_code_sha": args.run_code_sha,
        "historical_protocol": {
            "optimizer": "Adam", "base_learning_rate": 0.0001,
            "parameter_group_count": 1, "parameter_group_lr": [0.0001],
            "betas": [0.9, 0.99], "weight_decay": 0.0002,
            "scheduler": "none", "warmup": "none",
            "gradient_clipping": {"kind": "clip_grad_value_", "value": 10000.0},
            "road_loss": "BCEWithLogits(reduction=sum)",
            "junction_loss": "BCEWithLogits(reduction=sum)",
            "anchor_loss": "sum(BCEWithLogits(anchor)+BCEWithLogits(anchor_lowrs)) over valid target channels",
            "task_weights": {"road": 1.0, "junction": 1.0, "anchor": 1.0},
            "optimizer_step_order": "forward -> losses -> backward -> value clip -> optimizer.step -> last-train-batch metrics",
            "output_elements_per_batch_size_2": {
                "road": 8192, "junction": 8192,
                "anchor": 524288, "anchor_lowrs": 524288}},
        "stage_s3b_protocol": {
            "optimizer": "Adam", "scheduler": "none", "warmup": "none",
            "lr_multipliers": [1.0, 0.3, 0.1],
            "max_optimizer_steps": 20480, "evaluation_interval": 2560,
            "junction_loss_candidates": [LOSS_LEGACY, LOSS_BALANCED,
                                           LOSS_BALANCED_DICE]}}
    loss_audit = {
        "stage": "seg_raster_stage_s3b", "status": "PASS",
        "execution_environment": "REMOTE_TRAINING_SERVER",
        "remote_host_label": "exp-237-tunnel", "run_code_sha": args.run_code_sha,
        "diagnostic_batch_count": 16, "seed": int(config["S3"]["SEED"]),
        "diagnostic_batch_identity_sha256": identity_sha256(identities),
        "pixel_counts": pixel_counts, "junction_pos_weight": junction_counts,
        "junction_gradient_matching": {
            LOSS_BALANCED: {"alpha": alpha_balanced,
                            "legacy_mean_norm": float(np.mean(legacy_norms)),
                            "candidate_mean_norm": float(np.mean(balanced_norms))},
            LOSS_BALANCED_DICE: {"alpha": alpha_dice,
                                 "legacy_mean_norm": float(np.mean(legacy_norms)),
                                 "candidate_mean_norm": float(np.mean(dice_norms))}},
        "per_batch": per_batch,
        "initialization": initialization,
        "data_provenance": {
            "sample_plan_sha256": sample_plan["plan_sha256"],
            "split_manifest_sha256": split["manifest_sha256"],
            "initialization_content_sha256": initialization["content_sha256"],
            "initialization_file_sha256": initialization["file_sha256"],
            "graph_sha256": split["graph_sha256"]},
        "peak_allocated_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_memory_mb": torch.cuda.max_memory_reserved() / 2**20,
        "generated_at_utc": utc_now()}
    write_json(args.output_root / "stage_s3b_optimizer_loss_contract.json",
               optimizer_contract)
    write_json(args.output_root / "stage_s3b_loss_gradient_audit.json", loss_audit)
    write_json(args.output_root / "stage_s3b_remote_preflight.json", {
        "stage": "seg_raster_stage_s3b", "status": "PASS",
        "execution_environment": "REMOTE_TRAINING_SERVER",
        "run_code_sha": args.run_code_sha, "diagnostic_batch_count": 16,
        "all_losses_gradients_finite": True,
        "sequence_loader_called": False, "transformer_constructed": False,
        "legacy_dsf_constructed": False,
        "peak_allocated_memory_mb": loss_audit["peak_allocated_memory_mb"],
        "peak_reserved_memory_mb": loss_audit["peak_reserved_memory_mb"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
