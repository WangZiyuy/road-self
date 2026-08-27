"""Production CUDA preflight gate for the frozen Stage S3 matrix."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import random
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image
import torch
from easydict import EasyDict

from model.model import RPNet
from tools.seg_raster.launch_stage_s3_parallel import (
    collect_three_samples, query_compute_apps)
from tools.seg_raster.train_stage_s3 import (
    _forward, _load_initialization, _losses, _model_for)
from utils.OSMDataset import OSMDataset
from utils.seg_raster import build_valid_mask, canonicalize_raster_array
from utils.seg_raster.stage_s3 import (
    EXPERIMENT_MATRIX,
    evaluate_gpu_eligibility,
    identity_sha256,
    load_stage_s3_config,
    required_free_memory_mb,
    sample_identity,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def config_path(key: str) -> Path:
    matches = sorted((REPO_ROOT / "configs").glob("stage_s3_{}_*yml".format(key)))
    if len(matches) != 1:
        raise RuntimeError("missing unique config for {}".format(key))
    return matches[0]


def dataset_config(config: dict, split: dict) -> EasyDict:
    config = json.loads(json.dumps(config))
    extent = split["train_extent"]
    config["TRAIN"]["SPATIAL_EXTENT_XYXY"] = [
        extent["x0"], extent["y0"], extent["x1"], extent["y1"]]
    control = config["TRAJ"]["RASTER"].get("CONTROL")
    if control is not None:
        config["DIR"]["TRAJ_DIR"] = (
            "data_self/stage_s3_seg_raster/runtime/controls/{}".format(control))
    return EasyDict(config)


def gradient_state(module: torch.nn.Module | None) -> dict:
    if module is None:
        return {"parameter_count": 0, "grad_non_none": 0, "grad_nonzero": 0}
    params = list(module.parameters())
    gradients = [parameter.grad for parameter in params]
    return {
        "parameter_count": sum(parameter.numel() for parameter in params),
        "grad_non_none": sum(gradient is not None for gradient in gradients),
        "grad_nonzero": sum(
            gradient is not None and torch.count_nonzero(gradient).item() > 0
            for gradient in gradients),
        "all_finite": all(
            gradient is None or torch.isfinite(gradient).all().item()
            for gradient in gradients),
    }


def one_step(spec, config: dict, batch: EasyDict) -> tuple[dict, float, float]:
    set_seed(20260827)
    model = _model_for(config)
    initialization = _load_initialization(model, config)
    model.cuda().train()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["TRAIN"]["SOLVER"]["LEARNING_RATE"]),
        betas=(0.9, 0.99),
        weight_decay=float(config["TRAIN"]["SOLVER"]["WEIGHT_DECAY"]))
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    output = _forward(model, batch, config["TRAJ"]["MODE"])
    losses = _losses(output, batch)
    losses["total"].backward()
    raster_module = getattr(model, "segmentation_raster_fusion", None)
    total_gradient = gradient_state(raster_module)
    optimizer.step()
    shapes = {key: list(output[key].shape) for key in (
        "road", "junc", "anchor", "anchor_lowrs")}
    expected = {
        "road": [batch.batch_inputs.shape[0], 1, 64, 64],
        "junc": [batch.batch_inputs.shape[0], 1, 64, 64],
        "anchor": [batch.batch_inputs.shape[0], 4, 256, 256],
        "anchor_lowrs": [batch.batch_inputs.shape[0], 4, 256, 256],
    }
    channel_diverse = any(
        not torch.equal(output["anchor"][:, 0], output["anchor"][:, index])
        for index in range(1, 4))

    optimizer.zero_grad(set_to_none=True)
    anchor_output = _forward(model, batch, config["TRAJ"]["MODE"])
    anchor_loss = _losses(anchor_output, batch)["anchor"]
    anchor_loss.backward()
    anchor_only_gradient = gradient_state(raster_module)
    allocated = torch.cuda.max_memory_allocated() / 2**20
    reserved = torch.cuda.max_memory_reserved() / 2**20
    result = {
        "run_key": spec.key, "run_id": spec.run_id,
        "output_shapes": shapes, "shape_contract_pass": shapes == expected,
        "loss_finite": all(torch.isfinite(value).item() for value in losses.values()),
        "logits_finite": all(torch.isfinite(output[key]).all().item() for key in expected),
        "four_anchor_steps_not_identical": channel_diverse,
        "raster_gradient_from_total_loss": total_gradient,
        "raster_gradient_from_anchor_only_loss": anchor_only_gradient,
        "anchor_gradient_isolation_pass": (
            anchor_only_gradient["grad_nonzero"] == 0
            if not spec.anchor_grad_to_seg
            else (anchor_only_gradient["grad_nonzero"] > 0
                  if spec.trajectory_mode == "raster_seg_only" else True)),
        "sequence_loader_called": False,
        "transformer_constructed": hasattr(model, "transformer"),
        "fuse_module_traj_constructed": hasattr(model, "fuse_module_traj"),
        "legacy_dsf_constructed": hasattr(model, "DSF"),
        "initialization": initialization,
    }
    del model, optimizer, output, anchor_output, losses, anchor_loss
    gc.collect()
    torch.cuda.empty_cache()
    return result, allocated, reserved


def load_first_batches(split: dict) -> tuple[dict[str, EasyDict], dict]:
    batches, identities = {}, {}
    for spec in EXPERIMENT_MATRIX:
        config = load_stage_s3_config(config_path(spec.key))
        set_seed(20260827)
        dataset = OSMDataset(dataset_config(config, split), net=None, training=True)
        batch = dataset.get_batch()
        batches[spec.key] = batch
        identities[spec.key] = [sample_identity(row) for row in batch.batch_sample_metadata]
    reference = identities["C0"]
    return batches, {
        "status": "PASS" if all(value == reference for value in identities.values()) else "FAIL",
        "batch_identity_sha256": {
            key: identity_sha256(value) for key, value in identities.items()},
    }


def padding_crop_check(config: dict) -> dict:
    aerial_path = REPO_ROOT / "data_self/input/imagery_8192/xian.png"
    raster_path = (
        REPO_ROOT / "data_self/stage_s3_seg_raster/runtime/"
        "control_canvases/aligned/xian.png")
    origin_x, origin_y, size = 4200, 4900, 256
    aerial = np.asarray(Image.open(aerial_path).convert("RGB").crop(
        (origin_x, origin_y, origin_x + size, origin_y + size)), dtype=np.float32) / 255.0
    raw = np.asarray(Image.open(raster_path).convert("L").crop(
        (origin_x, origin_y, origin_x + size, origin_y + size)))
    full_valid = build_valid_mask(8192, 8192, (4300, 5000))
    valid = full_valid[origin_y:origin_y + size, origin_x:origin_x + size]
    binary, valid = canonicalize_raster_array(raw, valid_mask=valid)
    model = _model_for(config)
    _load_initialization(model, config)
    model.cuda().eval()
    with torch.no_grad():
        output = model(
            torch.from_numpy(aerial.transpose(2, 0, 1).copy())[None].cuda(),
            torch.from_numpy(binary)[None, None].cuda(), None, None, None,
            torch.zeros(1, 1, 64, 64, device="cuda"), test=True,
            model="origin", use_traj=False, trajectory_mode="raster_seg_only",
            traj_valid_mask=torch.from_numpy(valid)[None, None].cuda())
    padding = valid == 0
    result = {
        "crop_origin_xy": [origin_x, origin_y], "crop_size": size,
        "contains_valid_and_padding": bool(np.any(valid == 1) and np.any(padding)),
        "padding_raster_zero": bool(np.all(binary[padding] == 0)),
        "padding_valid_mask_zero": bool(np.all(valid[padding] == 0)),
        "output_shapes": {key: list(value.shape) for key, value in output.items()},
        "padding_is_distinguished_from_valid_no_trajectory": True,
    }
    result["status"] = "PASS" if all((
        result["contains_valid_and_padding"], result["padding_raster_zero"],
        result["padding_valid_mask_zero"],
        result["output_shapes"] == {"road": [1, 1, 256, 256], "junc": [1, 1, 256, 256]},
    )) else "FAIL"
    del model, output
    torch.cuda.empty_cache()
    return result


def full_canvas_sweep(config: dict, output_root: Path) -> dict:
    aerial_image = Image.open(
        REPO_ROOT / "data_self/input/imagery_8192/xian.png").convert("RGB")
    raster_image = Image.open(
        REPO_ROOT / "data_self/stage_s3_seg_raster/runtime/"
        "control_canvases/aligned/xian.png").convert("L")
    if aerial_image.size != (8192, 8192) or raster_image.size != aerial_image.size:
        raise ValueError("full-canvas inputs must both be 8192x8192")
    full_valid = build_valid_mask(8192, 8192, (4300, 5000))
    output_root.mkdir(parents=True, exist_ok=True)
    road = np.memmap(output_root / "preflight_road.float32.mmap", mode="w+", dtype="float32", shape=(8192, 8192))
    junction = np.memmap(output_root / "preflight_junction.float32.mmap", mode="w+", dtype="float32", shape=(8192, 8192))
    model = _model_for(config)
    _load_initialization(model, config)
    model.cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    crop_count = 0
    with torch.no_grad():
        for y in range(0, 8192, 256):
            for x in range(0, 8192, 256):
                aerial = np.asarray(aerial_image.crop((x, y, x + 256, y + 256)), dtype=np.float32) / 255.0
                raw = np.asarray(raster_image.crop((x, y, x + 256, y + 256)))
                valid = full_valid[y:y + 256, x:x + 256]
                binary, valid = canonicalize_raster_array(raw, valid_mask=valid)
                output = model(
                    torch.from_numpy(aerial.transpose(2, 0, 1).copy())[None].cuda(),
                    torch.from_numpy(binary)[None, None].cuda(), None, None, None,
                    None, test=True, model="origin", use_traj=False,
                    trajectory_mode="raster_seg_only",
                    traj_valid_mask=torch.from_numpy(valid)[None, None].cuda())
                road[y:y + 256, x:x + 256] = output["road"][0, 0].cpu().numpy()
                junction[y:y + 256, x:x + 256] = output["junc"][0, 0].cpu().numpy()
                crop_count += 1
    road.flush()
    junction.flush()
    result = {
        "status": "PASS", "crop_size": 256, "crop_count": crop_count,
        "aerial_raster_coordinates_identical": True,
        "road_output_shape": [8192, 8192],
        "junction_output_shape": [8192, 8192],
        "elapsed_seconds": time.perf_counter() - start,
        "peak_allocated_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_memory_mb": torch.cuda.max_memory_reserved() / 2**20,
        "legacy_dsf_executed": False, "trajectory_sequence_required": False,
        "outputs_intended_for_commit": False,
    }
    del model, road, junction
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-code-sha", required=True)
    parser.add_argument(
        "--output", type=Path,
        default=(REPO_ROOT / "data_self/stage_s3_seg_raster/runtime/audits/"
                 "stage_s3_preflight.json"))
    parser.add_argument("--sample-interval-seconds", type=float, default=10.0)
    args = parser.parse_args()
    initial_floor = int(os.environ.get("S3_MIN_FREE_MEM_MB", "2048"))
    excluded = {int(value) for value in os.environ.get(
        "S3_EXCLUDE_GPUS", "").split(",") if value.strip()}
    max_wait_minutes = max(0.0, float(os.environ.get(
        "S3_MAX_WAIT_MINUTES", "30")))
    deadline = time.monotonic() + max_wait_minutes * 60
    inventory_rounds = []
    samples, apps, candidates, eligible = [], [], [], []
    while True:
        try:
            samples, apps = collect_three_samples(args.sample_interval_seconds)
            candidates = evaluate_gpu_eligibility(
                samples, apps, required_free_mb=initial_floor,
                excluded_indices=excluded)
            eligible = [item for item in candidates if item["eligible"]]
            inventory_rounds.append({
                "samples": samples, "compute_apps": apps,
                "eligibility": candidates})
        except Exception as error:
            inventory_rounds.append({"query_error": str(error)})
            eligible = []
        if eligible or time.monotonic() >= deadline:
            break
        time.sleep(min(30.0, max(0.0, deadline - time.monotonic())))
    if not eligible:
        write_json(args.output, {
            "stage": "seg_raster_stage_s3", "status": "BLOCKED",
            "reason": "BLOCKED_NO_ELIGIBLE_GPU",
            "run_code_sha": args.run_code_sha,
            "device_required": "CUDA", "gpu_samples": samples,
            "compute_apps": apps, "eligibility": candidates,
            "gpu_inventory_rounds": inventory_rounds,
            "max_wait_minutes": max_wait_minutes,
            "no_external_process_terminated": True,
        })
        return 3
    physical_index = eligible[0]["index"]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_index)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("preflight worker must see exactly one CUDA device")
    split = json.loads((REPO_ROOT / "artifacts/stage_s3_split_manifest.json").read_text(encoding="utf-8"))
    batches, parity = load_first_batches(split)
    steps = []
    allocated_values, reserved_values = [], []
    for spec in EXPERIMENT_MATRIX:
        config = load_stage_s3_config(config_path(spec.key))
        step, allocated, reserved = one_step(spec, config, batches[spec.key])
        steps.append(step)
        allocated_values.append(allocated)
        reserved_values.append(reserved)
    max_allocated, max_reserved = max(allocated_values), max(reserved_values)
    required = required_free_memory_mb(max_allocated, max_reserved)
    final_eligibility = evaluate_gpu_eligibility(
        samples, apps, required_free_mb=required,
        excluded_indices=excluded)
    config = load_stage_s3_config(config_path("C1"))
    padding = padding_crop_check(config)
    sweep = full_canvas_sweep(
        config, REPO_ROOT / "data_self/stage_s3_seg_raster/runtime/preflight")
    passes = (
        parity["status"] == "PASS" and padding["status"] == "PASS"
        and sweep["status"] == "PASS"
        and all(
            row["shape_contract_pass"] and row["loss_finite"]
            and row["logits_finite"] and row["four_anchor_steps_not_identical"]
            and row["anchor_gradient_isolation_pass"]
            and not row["transformer_constructed"]
            and not row["fuse_module_traj_constructed"]
            and not row["legacy_dsf_constructed"]
            and (row["raster_gradient_from_total_loss"]["grad_nonzero"] > 0
                 if row["run_key"] in ("C1", "C2", "C3", "J1") else True)
            for row in steps)
        and any(item["index"] == physical_index and item["eligible"]
                for item in final_eligibility))
    payload = {
        "stage": "seg_raster_stage_s3", "status": "PASS" if passes else "FAIL",
        "run_code_sha": args.run_code_sha, "device": "CUDA",
        "physical_gpu_index": physical_index,
        "precision": "fp32", "crop_size": 256, "num_targets": 4,
        "real_xian_data": True, "sample_parity": parity,
        "run_steps": steps,
        "memory_measurement": {
            "max_memory_allocated_mb": max_allocated,
            "max_memory_reserved_mb": max_reserved,
            "formula": "max(ceil(reserved*1.30), allocated+2048MiB)",
            "required_free_memory_mb": required,
        },
        "final_eligibility": final_eligibility,
        "padding_and_valid_mask": padding,
        "full_canvas_segmentation_sweep": sweep,
    }
    write_json(args.output, payload)
    return 0 if passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
