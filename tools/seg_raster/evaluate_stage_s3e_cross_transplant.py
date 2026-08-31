"""Read-only Stage S3E road-head / raster-adapter cross-transplant audit."""

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

from model.model import RPNet
from tools.seg_raster.train_stage_s3c import (
    array_sha256, set_seed, to_cuda, utc_now)
from tools.seg_raster.train_stage_s3d import build_batches, forward_model
from utils import model_utils
from utils.seg_raster.stage_s3 import (
    binary_segmentation_metrics, load_stage_s3_config, sha256_file)
from utils.seg_raster.stage_s3d import (
    STAGE_S3D_SEED, configure_road_only_training, set_road_only_train_mode)
from utils.seg_raster.stage_s3e import (
    ADAPTER_PREFIX, ROAD_HEAD_PREFIXES, build_cross_transplant_state,
    finite_tree, gradient_comparison, layerwise_gradient_comparison,
    metric_decomposition, named_gradient_vector, parameter_drift,
    tensor_map_sha256)


def write_json(path: Path, value: object) -> None:
    finite_tree(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def load_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("stage") != "seg_raster_stage_s3d":
        raise RuntimeError("not a Stage S3D checkpoint: " + path.name)
    return payload["state_dict"]


def checkpoint_path(root: Path, run: str, samples: int) -> Path:
    run_id = "N0_null_seed20260827" if run == "N0" else "N1_aligned_seed20260827"
    return root / run_id / "checkpoints" / "samples_{:06d}.pth.tar".format(samples)


def verify_inventory(root: Path, inventory: dict) -> list[int]:
    expected = {}
    for run in ("N0", "N1"):
        expected[run] = {int(row["samples_seen"]): row for row in inventory["runs"][run]}
    common = sorted(set(expected["N0"]) & set(expected["N1"]))
    if not common:
        raise RuntimeError("N0/N1 have no common checkpoints")
    for run in ("N0", "N1"):
        for samples in common:
            path = checkpoint_path(root, run, samples)
            if not path.is_file():
                raise FileNotFoundError(path)
            actual = sha256_file(path)
            if actual != expected[run][samples]["sha256"]:
                raise RuntimeError("checkpoint SHA mismatch: {} {}".format(run, samples))
    return common


def new_model(state: dict[str, torch.Tensor]) -> RPNet:
    model = RPNet(
        num_targets=4, backbone_pretrained=False,
        enable_trajectory_modules=False,
        enable_zero_preserving_road_adapter=True,
        anchor_grad_to_seg=False)
    model.load_state_dict(state, strict=True)
    model.cuda().eval()
    return model


def evaluate(model: RPNet, batches: list, control: str, threshold: float) -> dict:
    logits, targets = [], []
    residual_l2, image_l2 = 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for batch in batches:
            output = forward_model(model, batch, control)
            road = output["road"]
            target = to_cuda(batch.batch_road_segmentation).to(road.dtype)
            logits.append(road.cpu().numpy())
            targets.append(target.cpu().numpy())
            residual_l2 += float(torch.square(
                output["feature_maps"]["strict_raster_residual"].float()).sum())
            image_l2 += float(torch.square(
                output["feature_maps"]["stage_fuse_img"].float()).sum())
    road_logits = np.concatenate(logits)
    road_target = np.concatenate(targets)
    metrics = binary_segmentation_metrics(
        road_logits, road_target, threshold=threshold)
    probability = 1.0 / (1.0 + np.exp(-road_logits.astype(np.float64)))
    positive = road_target > 0.5
    result = {
        "road_precision": metrics["precision"],
        "road_recall": metrics["recall"], "road_f1": metrics["f1"],
        "road_iou": metrics["iou"], "road_auprc": metrics["auprc"],
        "gt_mean_probability": float(probability[positive].mean()),
        "background_mean_probability": float(probability[~positive].mean()),
        "residual_to_image_l2_ratio": float(
            np.sqrt(residual_l2) / max(np.sqrt(image_l2), 1e-30)),
        "road_prediction_sha256": array_sha256([road_logits]),
        "validation_sample_count": int(road_logits.shape[0]),
    }
    return result


def cross_transplant(
    checkpoint_root: Path, samples: list[int], batches: list,
    threshold: float, output: Path, inventory: dict, source_stage_s3d_sha: str,
) -> dict:
    report = {
        "stage": "seg_raster_stage_s3e", "phase": "A_cross_transplant",
        "status": "RUNNING", "execution_environment": "REMOTE_TRAINING_SERVER",
        "optimizer_steps_executed": 0, "common_sample_counts": samples,
        "checkpoint_inventory_sha256": inventory["inventory_sha256"],
        "source_stage_s3d_run_code_sha": source_stage_s3d_sha,
        "checkpoints": {}, "generated_at": utc_now(),
    }
    for sample_count in samples:
        n0_state = load_state(checkpoint_path(checkpoint_root, "N0", sample_count))
        n1_state = load_state(checkpoint_path(checkpoint_root, "N1", sample_count))
        if tensor_map_sha256(n0_state, prefixes=("stage_1.", "stage_2.",
                "stage_3.", "stage_4.", "stage_5.")) != tensor_map_sha256(
                    n1_state, prefixes=("stage_1.", "stage_2.", "stage_3.",
                                        "stage_4.", "stage_5.")):
            raise RuntimeError("backbone differs at {}".format(sample_count))
        rows, audits = {}, {}
        for combination in ("T00", "T01", "T10", "T11"):
            state, audit = build_cross_transplant_state(
                n0_state, n1_state, combination)
            model = new_model(state)
            rows[combination] = evaluate(
                model, batches, audit["sources"]["raster_control"], threshold)
            audits[combination] = audit
            del model
            torch.cuda.empty_cache()
        report["checkpoints"][str(sample_count)] = {
            "metrics": rows, "decomposition": metric_decomposition(rows),
            "transplant_audit": audits,
            "source_checkpoint_sha256": {
                run: next(row["sha256"] for row in inventory["runs"][run]
                          if int(row["samples_seen"]) == sample_count)
                for run in ("N0", "N1")},
        }
        write_json(output, report)
    report["status"] = "PASS"
    write_json(output, report)
    return report


def drift_report(
    checkpoint_root: Path, samples: list[int], output: Path,
) -> dict:
    initial = load_state(checkpoint_path(checkpoint_root, "N0", samples[0]))
    rows = {}
    first_interval = None
    previous = None
    for sample_count in samples:
        n0 = load_state(checkpoint_path(checkpoint_root, "N0", sample_count))
        n1 = load_state(checkpoint_path(checkpoint_root, "N1", sample_count))
        rows[str(sample_count)] = parameter_drift(initial, n0, n1)
        previous = sample_count
    report = {
        "stage": "seg_raster_stage_s3e", "phase": "A_parameter_drift",
        "status": "PASS", "optimizer_steps_executed": 0,
        "common_sample_counts": samples, "by_samples": rows,
        "first_functional_head_drift_interval": first_interval,
    }
    write_json(output, report)
    return report


def gradient_for_control(model: RPNet, batches: list, control: str) -> dict:
    configure_road_only_training(model)
    model.cuda().eval()
    model.zero_grad(set_to_none=True)
    negative, positive = 0.0, 0.0
    for batch in batches:
        output = forward_model(model, batch, control)
        road = output["road"]
        target = to_cuda(batch.batch_road_segmentation).to(road.dtype)
        residual = output["feature_maps"]["strict_raster_residual"]
        residual.retain_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(
            road, target, reduction="sum").backward()
        expanded = (target > 0.5).expand_as(residual)
        grad = residual.grad.detach().abs()
        positive += float(grad[expanded].sum())
        negative += float(grad[~expanded].sum())
    vector, per_name = named_gradient_vector(model, ROAD_HEAD_PREFIXES)
    return {
        "vector": vector, "per_name": per_name,
        "road_head_gradient_l2": float(torch.linalg.vector_norm(vector.double())),
        "adapter_residual_negative_positive_gradient_mass_ratio": (
            negative / max(positive, 1e-30)),
    }


def gradient_report(
    checkpoint_root: Path, samples: list[int], batches: list, output: Path,
) -> dict:
    selected = sorted(set((samples[0], samples[1], samples[-1])))
    rows = {}
    for sample_count in selected:
        state = load_state(checkpoint_path(checkpoint_root, "N1", sample_count))
        values = {}
        for control in ("null", "aligned"):
            model = new_model(state)
            values[control] = gradient_for_control(model, batches, control)
            del model
            torch.cuda.empty_cache()
        rows[str(sample_count)] = {
            "overall": gradient_comparison(
                values["null"]["vector"], values["aligned"]["vector"]),
            "layerwise": layerwise_gradient_comparison(
                values["null"]["per_name"], values["aligned"]["per_name"]),
            "aligned_adapter_residual_negative_positive_gradient_mass_ratio":
                values["aligned"][
                    "adapter_residual_negative_positive_gradient_mass_ratio"],
        }
    report = {
        "stage": "seg_raster_stage_s3e", "phase": "A_gradient_field",
        "status": "PASS", "optimizer_steps_executed": 0,
        "diagnostic_batch_count": len(batches), "by_samples": rows,
    }
    write_json(output, report)
    return report


def add_functional_drift_interval(cross: dict, drift: dict, output: Path) -> None:
    samples = cross["common_sample_counts"]
    previous = None
    finding = "NOT_OBSERVED"
    for sample_count in samples:
        decomposition = cross["checkpoints"][str(sample_count)]["decomposition"]
        auprc = decomposition["road_auprc"]["head_drift"]
        f1 = decomposition["road_f1"]["head_drift"]
        if auprc < -0.002 or f1 < -0.005:
            finding = ("BEFORE_OR_AT_{}".format(sample_count) if previous is None
                       else "{}_TO_{}".format(previous, sample_count))
            break
        previous = sample_count
    drift["first_functional_head_drift_interval"] = finding
    write_json(output, drift)


def write_markdown(path: Path, cross: dict, drift: dict, gradients: dict) -> None:
    lines = [
        "# Stage S3E Cross-Transplant", "",
        "- Status: {}".format(cross["status"]),
        "- Optimizer steps executed: 0",
        "- Common checkpoints: {}".format(
            ", ".join(map(str, cross["common_sample_counts"]))),
        "- First functional head-drift interval: {}".format(
            drift["first_functional_head_drift_interval"]), "",
        "| samples | head drift AUPRC | adapter clean AUPRC | interaction AUPRC |",
        "|---:|---:|---:|---:|",
    ]
    for sample_count in cross["common_sample_counts"]:
        row = cross["checkpoints"][str(sample_count)]["decomposition"]["road_auprc"]
        lines.append("| {} | {:.6f} | {:.6f} | {:.6f} |".format(
            sample_count, row["head_drift"], row["adapter_on_clean_head"],
            row["interaction"]))
    lines.extend(["", "Gradient-field sample counts: {}.".format(
        ", ".join(gradients["by_samples"]))])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--checkpoint-inventory", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-code-sha", required=True)
    parser.add_argument("--source-stage-s3d-sha", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage S3E Phase A requires remote CUDA")
    if os.environ.get("S3E_RUN_CODE_SHA") != args.run_code_sha:
        raise RuntimeError("S3E code SHA environment mismatch")
    inventory = json.loads(args.checkpoint_inventory.read_text(encoding="utf-8"))
    inventory = dict(inventory)
    inventory["inventory_sha256"] = sha256_file(args.checkpoint_inventory)
    samples = verify_inventory(args.checkpoint_root, inventory)
    os.environ["S3D_CONTROL_ROOT"] = os.fspath(args.control_root)
    config = load_stage_s3_config(REPO_ROOT / "configs/stage_s3e_common.yml")
    split = json.loads((REPO_ROOT / config["S3"]["SPLIT_MANIFEST"])
                       .read_text(encoding="utf-8"))
    model_utils.Path.visualize_and_save_path = lambda *a, **k: None
    set_seed(STAGE_S3D_SEED + 1)
    batches, validation_sha, _ = build_batches(config, split, "aligned")
    args.output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    cross = cross_transplant(
        args.checkpoint_root, samples, batches,
        float(config["S3"]["FIXED_THRESHOLD"]),
        args.output_root / "stage_s3e_cross_transplant.json", inventory,
        args.source_stage_s3d_sha)
    cross["run_code_sha"] = args.run_code_sha
    cross["validation_plan_sha256"] = validation_sha
    cross["runtime_seconds"] = time.time() - started
    write_json(args.output_root / "stage_s3e_cross_transplant.json", cross)
    drift = drift_report(
        args.checkpoint_root, samples,
        args.output_root / "stage_s3e_head_parameter_drift.json")
    add_functional_drift_interval(
        cross, drift, args.output_root / "stage_s3e_head_parameter_drift.json")
    gradients = gradient_report(
        args.checkpoint_root, samples,
        batches[:int(config["S3"]["DIAGNOSTIC_VALIDATION_BATCHES"])],
        args.output_root / "stage_s3e_gradient_field.json")
    gradients["run_code_sha"] = args.run_code_sha
    write_json(args.output_root / "stage_s3e_gradient_field.json", gradients)
    write_markdown(
        args.output_root / "stage_s3e_cross_transplant.md",
        cross, drift, gradients)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
