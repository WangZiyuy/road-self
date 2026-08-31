"""Conditional frozen-anchor evaluation for Stage S3D N0--N4."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from model.model import RPNet
from tools.seg_raster.train_stage_s3c import array_sha256, set_seed, to_cuda, write_json
from tools.seg_raster.train_stage_s3d import build_batches
from utils.seg_raster.stage_s3 import anchor_metrics, identity_sha256, load_stage_s3_config, sha256_file
from utils.seg_raster.stage_s3d import STAGE_S3D_SEED, configure_road_only_training


REPO_ROOT = Path(__file__).resolve().parents[2]


def verify_checkout(expected_sha: str) -> None:
    head = subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                          text=True, capture_output=True).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"], check=True,
        text=True, capture_output=True).stdout.strip()
    if head != expected_sha or status:
        raise RuntimeError("S3D anchor evaluator requires frozen checkout")


def frozen_state_sha(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        if key.startswith(("road_seg.", "conv_road_final.",
                           "zero_preserving_road_adapter.")):
            continue
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
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
        raise RuntimeError("formal S3D anchor evaluation requires remote CUDA")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    split = json.loads((REPO_ROOT / "artifacts/stage_s3_split_manifest.json")
                       .read_text(encoding="utf-8"))
    config = load_stage_s3_config(REPO_ROOT / "configs/stage_s3d_common.yml")
    rows, validation_shas, common_shas, frozen_shas = {}, {}, {}, {}
    for key in ("N0", "N1", "N2", "N3", "N4"):
        spec = plan["runs"][key]
        control = spec["control"]
        set_seed(STAGE_S3D_SEED + 1)
        batches, validation_sha, _ = build_batches(config, split, control)
        validation_shas[key] = validation_sha
        common_shas[key] = identity_sha256([
            array_sha256([batch.batch_inputs, batch.batch_walked_path_small,
                          batch.batch_target_maps,
                          np.asarray(batch.batch_end_index)])
            for batch in batches])
        model = RPNet(
            num_targets=4, backbone_pretrained=False,
            enable_zero_preserving_road_adapter=True,
            anchor_grad_to_seg=False)
        checkpoint = (args.run_root / spec["run_id"] / "checkpoints"
                      / spec["checkpoint"])
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("code_sha") != args.run_code_sha:
            raise RuntimeError("S3D anchor checkpoint code SHA mismatch")
        model.load_state_dict(payload["state_dict"], strict=True)
        frozen_shas[key] = frozen_state_sha(dict(model.state_dict()))
        configure_road_only_training(model)
        model.cuda().eval()
        logits, targets, end_indices, per_target = [], [], [], []
        with torch.no_grad():
            for batch in batches:
                output = model(
                    to_cuda(batch.batch_inputs),
                    to_cuda(batch.batch_traj_inputs), None, None, None,
                    to_cuda(batch.batch_walked_path_small), NUM_TARGETS=4,
                    model="origin", trajectory_mode="raster_road_zero_preserving",
                    traj_valid_mask=to_cuda(batch.batch_traj_valid_masks),
                    raster_adapter_bypass=control == "null")
                logits.append(output["anchor"].cpu().numpy())
                targets.append(batch.batch_target_maps)
                end_indices.extend(int(value) for value in batch.batch_end_index)
        logits_np, targets_np = np.concatenate(logits), np.concatenate(targets)
        metrics = anchor_metrics(logits_np, targets_np, end_indices, threshold=0.3)
        metrics["threshold_recall"] = (
            1 - metrics["missed_branch_count"] / metrics["evaluated_target_count"]
            if metrics["evaluated_target_count"] else 0.0)
        probabilities = 1 / (1 + np.exp(-np.clip(logits_np, -80, 80)))
        metrics["prediction_probability_quantiles"] = {
            str(value): float(np.quantile(probabilities, value))
            for value in (0, .25, .5, .75, .9, .99, 1)}
        metrics.update({
            "run_key": key, "control": control,
            "checkpoint_samples_seen": payload["samples_seen"],
            "checkpoint_sha256": sha256_file(checkpoint),
            "prediction_sha256": array_sha256([logits_np]),
            "frozen_backbone_anchor_junction_sha256": frozen_shas[key],
            "physical_gpu_index": args.physical_gpu,
            "gpu_name": torch.cuda.get_device_name(0),
        })
        rows[key] = metrics
        del model
        torch.cuda.empty_cache()
    if len(set(validation_shas.values())) != 1 or len(set(common_shas.values())) != 1:
        raise RuntimeError("S3D anchor validation parity failed")
    if len(set(frozen_shas.values())) != 1:
        raise RuntimeError("S3D frozen backbone/anchor/junction differs")
    n1 = rows["N1"]
    specificity = all(
        n1["threshold_recall"] > rows[key]["threshold_recall"]
        and n1["top_k_recall"] >= rows[key]["top_k_recall"]
        and n1["localization_error"] <= rows[key]["localization_error"]
        for key in ("N0", "N2", "N3", "N4"))
    multistep = any(value > 0 for value in n1["per_step_recall"][1:])
    write_json(args.output, {
        "stage": "seg_raster_stage_s3d", "status": "PASS",
        "execution_environment": "REMOTE_TRAINING_SERVER",
        "run_code_sha": args.run_code_sha,
        "baseline_controlled_common_samples": plan["common_samples"],
        "fixed_threshold": 0.3, "runs": rows,
        "validation_plan_sha256": next(iter(validation_shas.values())),
        "frozen_backbone_anchor_junction_sha256": next(iter(frozen_shas.values())),
        "raw_raster_direct_anchor_path": False,
        "frozen_anchor_indirect_gate": "PASS" if specificity and multistep else "FAIL",
        "multistep_anchor_validity": "PASS" if multistep else "FAIL",
        "peak_gpu_memory_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
        "peak_gpu_memory_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
