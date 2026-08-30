"""Evaluate the four common-sample frozen-anchor Stage S3C controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from model.model import RPNet
from tools.seg_raster.audit_stage_s3c_remote import config_for
from tools.seg_raster.train_stage_s3c import (
    array_sha256, build_validation_batches, forward_model, set_seed,
    write_json,
)
from utils.seg_raster.stage_s3 import anchor_metrics, identity_sha256, sha256_file
from utils.seg_raster.stage_s3c import configure_frozen_explorer


def verify_checkout(expected_sha: str) -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True,
        capture_output=True).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"], check=True,
        text=True, capture_output=True).stdout.strip()
    if head != expected_sha or status:
        raise RuntimeError("anchor evaluator requires a clean frozen checkout")


def frozen_state_sha(state: dict[str, torch.Tensor]) -> str:
    excluded = (
        "road_seg.", "conv_road_final.", "junc_seg.",
        "conv_junc_final.", "segmentation_raster_fusion.")
    digest = hashlib.sha256()
    for key in sorted(state):
        if key.startswith(excluded):
            continue
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-code-sha", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    args = parser.parse_args()
    verify_checkout(args.run_code_sha)
    if not torch.cuda.is_available():
        raise RuntimeError("formal frozen-anchor evaluation requires remote CUDA")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    split = json.loads((REPO_ROOT / "artifacts/stage_s3_split_manifest.json")
                       .read_text(encoding="utf-8"))
    rows, validation_hashes, common_shas, frozen_shas = {}, {}, {}, {}
    for key in ("R0", "R1", "R2", "R3"):
        spec = plan["runs"][key]
        raster = key != "R0"
        control = {"R1": "aligned", "R2": "zero", "R3": "shift_fixed"}.get(key)
        config = config_for(raster=raster, control=control)
        set_seed(int(config["S3"]["SEED"]) + 1)
        batches, validation_sha = build_validation_batches(config, split)
        validation_hashes[key] = validation_sha
        common_shas[key] = identity_sha256([
            array_sha256([
                batch.batch_inputs, batch.batch_walked_path_small,
                batch.batch_target_maps, np.asarray(batch.batch_end_index)])
            for batch in batches])
        model = RPNet(
            num_targets=4, backbone_pretrained=False,
            enable_trajectory_modules=False,
            enable_raster_segmentation=raster,
            raster_use_valid_mask=True, anchor_grad_to_seg=False)
        checkpoint = (args.run_root / spec["source_run_id"] / "checkpoints"
                      / spec["checkpoint"])
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("code_sha") != args.run_code_sha:
            raise RuntimeError("anchor checkpoint code SHA mismatch")
        result = model.load_state_dict(payload["state_dict"], strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError("strict anchor checkpoint load failed")
        frozen_shas[key] = frozen_state_sha(dict(model.state_dict()))
        configure_frozen_explorer(model, raster_enabled=raster)
        model.cuda().eval()
        logits, targets, end_indices = [], [], []
        with torch.no_grad():
            for batch in batches:
                output = forward_model(
                    model, batch, config["TRAJ"]["MODE"],
                    segmentation_only=False)
                logits.append(output["anchor"].cpu().numpy())
                targets.append(batch.batch_target_maps)
                end_indices.extend(int(value) for value in batch.batch_end_index)
        logits_np = np.concatenate(logits)
        target_np = np.concatenate(targets)
        metrics = anchor_metrics(
            logits_np, target_np, end_indices, threshold=0.3)
        metrics["threshold_recall"] = (
            1.0 - metrics["missed_branch_count"] / metrics["evaluated_target_count"]
            if metrics["evaluated_target_count"] else 0.0)
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits_np, -80, 80)))
        metrics["prediction_probability_quantiles"] = {
            str(value): float(np.quantile(probability, value))
            for value in (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)}
        metrics.update({
            "run_key": key, "control": control,
            "checkpoint_samples_seen": int(payload["samples_seen"]),
            "checkpoint_sha256": sha256_file(checkpoint),
            "prediction_sha256": array_sha256([logits_np]),
            "frozen_backbone_anchor_sha256": frozen_shas[key],
            "physical_gpu_index": args.physical_gpu,
            "gpu_name": torch.cuda.get_device_name(0),
        })
        rows[key] = metrics
        del model
        torch.cuda.empty_cache()
    if len(set(validation_hashes.values())) != 1 or len(set(common_shas.values())) != 1:
        raise RuntimeError("frozen-anchor validation sample parity failed")
    if len(set(frozen_shas.values())) != 1:
        raise RuntimeError("backbone/anchor weights differ across controls")
    write_json(args.output, {
        "stage": "seg_raster_stage_s3c", "status": "PASS",
        "execution_environment": "REMOTE_TRAINING_SERVER",
        "run_code_sha": args.run_code_sha,
        "baseline_controlled_common_samples": plan["common_samples"],
        "fixed_threshold": 0.3,
        "validation_plan_sha256": next(iter(validation_hashes.values())),
        "common_validation_tensor_sha256": next(iter(common_shas.values())),
        "frozen_backbone_anchor_sha256": next(iter(frozen_shas.values())),
        "anchor_loss_backward_executed": False,
        "raw_raster_direct_anchor_path": False,
        "runs": rows,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
