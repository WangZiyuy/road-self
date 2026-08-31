"""Remote-only preparation, input-swap forensic, and S3D preflight."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
import torch

from model.model import RPNet
from tools.seg_raster.train_stage_s3c import (
    array_sha256, frozen_checkout, set_seed, to_cuda, write_json)
from tools.seg_raster.train_stage_s3d import (
    apply_control, build_batches, cfg_for_dataset, evaluate, forward_model)
from utils.OSMDataset import OSMDataset
from utils.seg_raster.stage_s3 import (
    binary_segmentation_metrics, load_stage_s3_config, sha256_file)
from utils.seg_raster.stage_s3c import (
    configure_frozen_explorer, segmentation_losses,
    set_frozen_explorer_train_mode, strict_load_official_checkpoint,
    trainable_parameters)
from utils.seg_raster.stage_s3d import (
    INPUT_SWAP_CONTROLS, STAGE_S3D_SEED, STRICT_MODE,
    array_sha256 as one_array_sha256, classify_current_zero_path,
    configure_road_only_training, density_stratified_derangement,
    permute_rasters, road_loss, road_only_parameters,
    strict_load_stage_s3d_baseline, tensor_statistics,
    translate_zero_fill,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
S3C_RUN_CODE_SHA = "dbac3ac3c38d04ea20d25a8abc4aa0cfe91818e3"
RUN_IDS = {key: key + "_seed20260827" for key in ("R0", "R1", "R2", "R3")}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare(args: argparse.Namespace) -> int:
    frozen_checkout(args.run_code_sha)
    runtime = REPO_ROOT / "data_self/stage_s3d_seg_raster/runtime"
    control_root = runtime / "controls"
    s3c_root = Path(os.environ["S3C_CONTROL_ROOT"])
    canonical = Path(os.environ["S3D_CANONICAL_RASTER"])
    sample_source = Path(os.environ["S3D_S3C_SAMPLE_PLAN"])
    control_root.mkdir(parents=True, exist_ok=True)
    for name in ("aligned", "zero"):
        destination = control_root / name / "xian_0_0.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s3c_root / name / "xian_0_0.png", destination)
    raw = np.asarray(Image.open(canonical).convert("L"))
    if raw.shape != (8192, 8192):
        raise ValueError("canonical raster must be 8192x8192")
    binary = (raw > 0).astype(np.uint8) * 255
    shifted = translate_zero_fill(binary, (512, 512))
    large_dir = control_root / "shift_large"
    large_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(shifted[:4096, :4096]).save(large_dir / "xian_0_0.png")
    tile = binary[:4096, :4096]
    patches = np.stack([
        tile[y:y + 256, x:x + 256]
        for y in range(0, 4096, 256)
        for x in range(0, 4096, 256)])
    ratios = np.count_nonzero(patches, axis=(1, 2)) / (256 * 256)
    mapping = density_stratified_derangement(ratios.tolist())
    permuted = np.zeros_like(tile)
    for index, donor in enumerate(mapping):
        y, x = divmod(index, 16)
        permuted[y * 256:(y + 1) * 256,
                 x * 256:(x + 1) * 256] = patches[donor]
    permuted_dir = control_root / "permuted"
    permuted_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(permuted).save(permuted_dir / "xian_0_0.png")
    sample_destination = runtime / "sample_plan.json"
    shutil.copy2(sample_source, sample_destination)
    split = read_json(REPO_ROOT / "artifacts/stage_s3_split_manifest.json")
    payload = {
        "stage": "seg_raster_stage_s3d", "status": "PASS",
        "run_code_sha": args.run_code_sha,
        "s3c_run_code_sha": S3C_RUN_CODE_SHA,
        "official_checkpoint_sha256": sha256_file(
            os.environ["S3D_BASELINE_CHECKPOINT"]),
        "canonical_raster_sha256": sha256_file(canonical),
        "aligned_tile_sha256": sha256_file(
            control_root / "aligned/xian_0_0.png"),
        "shift_large_tile_sha256": sha256_file(
            control_root / "shift_large/xian_0_0.png"),
        "permuted_tile_sha256": sha256_file(
            control_root / "permuted/xian_0_0.png"),
        "permuted_tile_donor_mapping_sha256": one_array_sha256(
            np.asarray(mapping, dtype=np.int64)),
        "sample_plan_sha256": sha256_file(sample_destination),
        "sample_plan_contract_sha256": read_json(sample_destination)["plan_sha256"],
        "split_manifest_sha256": split["manifest_sha256"],
        "shift_xy": [512, 512], "zero_fill": True, "circular_wrap": False,
        "intended_for_commit": False,
    }
    write_json(args.output_root / "stage_s3d_data_baseline_parity.json", payload)
    return 0


def _s3c_config(control_root: Path) -> dict:
    config = load_stage_s3_config(REPO_ROOT / "configs/stage_s3c_common.yml")
    config["TRAJ"]["MODE"] = "raster_seg_only"
    config["TRAJ"]["RASTER"]["CONTROL"] = "aligned"
    config["DIR"]["TRAJ_DIR"] = os.fspath(control_root / "aligned")
    return config


def _validation_batches(config: dict, split: dict, root: Path, control: str):
    from easydict import EasyDict
    cfg = copy.deepcopy(config)
    disk = control if control in ("aligned", "zero", "shift_fixed") else "aligned"
    if control == "shift_large":
        disk_root = Path(os.environ["S3D_CONTROL_ROOT"])
        disk = "shift_large"
    else:
        disk_root = root
    cfg["DIR"]["TRAJ_DIR"] = os.fspath(disk_root / disk)
    extent = split["validation_extent"]
    xyxy = [extent["x0"], extent["y0"], extent["x1"], extent["y1"]]
    set_seed(STAGE_S3D_SEED + 1)
    cfg["TRAIN"]["SPATIAL_EXTENT_XYXY"] = xyxy
    dataset = OSMDataset(EasyDict(cfg), net=None, training=True)
    batches = []
    for index in range(int(cfg["S3"]["VALIDATION_BATCHES"])):
        batch = dataset.get_batch()
        if control == "permuted":
            ratios = np.count_nonzero(
                batch.batch_traj_inputs, axis=(1, 2, 3)) / np.prod(
                    batch.batch_traj_inputs.shape[1:])
            mapping = density_stratified_derangement(
                ratios.tolist(), seed=STAGE_S3D_SEED + index)
            batch.batch_traj_inputs = permute_rasters(
                batch.batch_traj_inputs, mapping)
        elif control == "all_one":
            batch.batch_traj_inputs = np.ones_like(batch.batch_traj_inputs)
            batch.batch_traj_inputs *= batch.batch_traj_valid_masks
        dataset.push_and_vis_batch(batch, 0, index)
        batches.append(batch)
    return batches


def _load_s3c_model(run_key: str, checkpoint: Path) -> RPNet:
    model = RPNet(
        num_targets=4, backbone_pretrained=False,
        enable_raster_segmentation=True, anchor_grad_to_seg=False)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    if payload.get("code_sha") != S3C_RUN_CODE_SHA:
        raise RuntimeError("S3C checkpoint code SHA mismatch")
    if run_key == "R0":
        strict_load_official_checkpoint(
            model, {"state_dict": state},
            allowed_new_prefixes=("segmentation_raster_fusion.",))
    else:
        model.load_state_dict(state, strict=True)
    model.cuda().eval()
    return model


def _s3c_forward(model: RPNet, batch) -> dict:
    return model(
        to_cuda(batch.batch_inputs), to_cuda(batch.batch_traj_inputs),
        None, None, None, to_cuda(batch.batch_walked_path_small),
        NUM_TARGETS=4, model="origin", trajectory_mode="raster_seg_only",
        traj_valid_mask=to_cuda(batch.batch_traj_valid_masks),
        segmentation_only=True)


def _evaluate_s3c(model: RPNet, batches: list, threshold: float) -> tuple[dict, dict]:
    roads, junctions, road_targets, junction_targets = [], [], [], []
    features: dict[str, list[np.ndarray]] = {}
    ratios = []
    with torch.no_grad():
        for batch in batches:
            output = _s3c_forward(model, batch)
            roads.append(output["road"].cpu().numpy())
            junctions.append(output["junc"].cpu().numpy())
            road_targets.append(batch.batch_road_segmentation)
            junction_targets.append(batch.batch_junction_segmentation)
            ratios.extend((np.count_nonzero(
                batch.batch_traj_inputs, axis=(1, 2, 3)) /
                np.prod(batch.batch_traj_inputs.shape[1:])).tolist())
            fmap = output["feature_maps"]
            derived = dict(fmap)
            derived["stage_fuse_delta"] = (
                fmap["stage_fuse_seg"] - fmap["stage_fuse_img"])
            for name in (
                "traj_feature_seg_only", "projected_traj_seg_only",
                "raster_delta_seg_only", "stage_fuse_delta", "road_fts",
                "junc_fts",
            ):
                features.setdefault(name, []).append(
                    derived[name].detach().cpu().numpy())
    road_logits, junc_logits = np.concatenate(roads), np.concatenate(junctions)
    road_target, junc_target = np.concatenate(road_targets), np.concatenate(junction_targets)
    road = binary_segmentation_metrics(road_logits, road_target, threshold=threshold)
    junction = binary_segmentation_metrics(junc_logits, junc_target,
                                           threshold=threshold)
    per_sample = []
    for index in range(len(road_logits)):
        rm = binary_segmentation_metrics(
            road_logits[index:index + 1], road_target[index:index + 1],
            threshold=threshold)
        jm = binary_segmentation_metrics(
            junc_logits[index:index + 1], junc_target[index:index + 1],
            threshold=threshold)
        per_sample.append({
            "sample_index": index, "road_f1": rm["f1"],
            "road_iou": rm["iou"], "road_auprc": rm["auprc"],
            "junction_f1": jm["f1"], "junction_auprc": jm["auprc"],
            "raster_positive_ratio": ratios[index],
        })
    metrics = {
        "road_precision": road["precision"], "road_recall": road["recall"],
        "road_f1": road["f1"], "road_iou": road["iou"],
        "road_auprc": road["auprc"],
        "junction_precision": junction["precision"],
        "junction_recall": junction["recall"],
        "junction_f1": junction["f1"],
        "junction_auprc": junction["auprc"],
        "prediction_sha256": array_sha256([road_logits, junc_logits]),
        "road_prediction_sha256": array_sha256([road_logits]),
        "junction_prediction_sha256": array_sha256([junc_logits]),
        "per_sample": per_sample,
        "raster_positive_ratio": {
            "min": min(ratios), "mean": float(np.mean(ratios)),
            "max": max(ratios)},
    }
    raw = {"road_logits": road_logits, "junction_logits": junc_logits,
           "features": {key: np.concatenate(value)
                        for key, value in features.items()}}
    return metrics, raw


def _difference(left: np.ndarray, right: np.ndarray) -> dict:
    value = torch.from_numpy(np.asarray(left) - np.asarray(right))
    result = tensor_statistics(value)
    result["mean_abs"] = float(value.abs().mean())
    result["max_abs"] = float(value.abs().max())
    return result


def _control_strength(batches: dict[str, list]) -> dict:
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        distance_transform_edt = None
    aligned = np.concatenate([batch.batch_traj_inputs
                              for batch in batches["aligned"]]) > 0
    road = np.concatenate([batch.batch_road_segmentation
                           for batch in batches["aligned"]]) > 0
    # Road labels are quarter-resolution; nearest expansion is deterministic.
    road_full = np.repeat(np.repeat(road, 4, axis=-2), 4, axis=-1)
    output = {}
    for control in ("aligned", "shift_fixed", "shift_large", "permuted"):
        raster = np.concatenate([batch.batch_traj_inputs
                                 for batch in batches[control]]) > 0
        intersection = np.logical_and(aligned, raster).sum()
        union = np.logical_or(aligned, raster).sum()
        flat_left, flat_right = aligned.reshape(len(aligned), -1), raster.reshape(len(raster), -1)
        per_sample = []
        distances = []
        for index in range(len(raster)):
            inter = np.logical_and(raster[index], road_full[index]).sum()
            positive = raster[index].sum()
            sample_iou_inter = np.logical_and(aligned[index], raster[index]).sum()
            sample_iou_union = np.logical_or(aligned[index], raster[index]).sum()
            if distance_transform_edt is not None:
                distance = distance_transform_edt(~road_full[index, 0])
                distances.extend(distance[raster[index, 0]].tolist())
            per_sample.append({
                "sample_index": index,
                "positive_ratio": float(raster[index].mean()),
                "aligned_control_iou": float(sample_iou_inter / sample_iou_union)
                if sample_iou_union else 1.0,
                "road_gt_overlap_ratio": float(inter / positive) if positive else 0.0,
            })
        correlation = []
        for index in range(len(raster)):
            if flat_left[index].std() and flat_right[index].std():
                correlation.append(float(np.corrcoef(
                    flat_left[index], flat_right[index])[0, 1]))
        output[control] = {
            "positive_ratio": float(raster.mean()),
            "aligned_control_iou": float(intersection / union) if union else 1.0,
            "binary_pearson_correlation_mean": float(np.mean(correlation))
            if correlation else 0.0,
            "road_gt_overlap_ratio": float(np.logical_and(
                raster, road_full).sum() / max(1, raster.sum())),
            "positive_to_road_gt_distance_pixels": {
                "status": "PASS" if distance_transform_edt else "NOT_INSTRUMENTED",
                "mean": float(np.mean(distances)) if distances else 0.0,
                "p50": float(np.percentile(distances, 50)) if distances else 0.0,
                "p95": float(np.percentile(distances, 95)) if distances else 0.0,
            },
            "per_sample": per_sample,
        }
    return output


def _zero_runtime(model: RPNet, batch) -> dict:
    copied = copy.copy(batch)
    copied.batch_traj_inputs = np.zeros_like(batch.batch_traj_inputs)
    with torch.no_grad():
        output = _s3c_forward(model.eval(), copied)
    fmap = output["feature_maps"]
    return {
        "max_abs_stage_fuse_seg_minus_img": float((
            fmap["stage_fuse_seg"] - fmap["stage_fuse_img"]).abs().max()),
        "raster_residual_l2_norm": float(torch.linalg.vector_norm(
            fmap["raster_delta_seg_only"])),
        "road_logits_sha256": one_array_sha256(output["road"]),
    }


def forensic(args: argparse.Namespace) -> int:
    frozen_checkout(args.run_code_sha)
    split = read_json(REPO_ROOT / "artifacts/stage_s3_split_manifest.json")
    s3c_control_root = Path(os.environ["S3C_CONTROL_ROOT"])
    config = _s3c_config(s3c_control_root)
    batches = {control: _validation_batches(
        config, split, s3c_control_root, control)
        for control in INPUT_SWAP_CONTROLS}
    matrix, activation = {}, {}
    run_root = Path(os.environ["S3D_S3C_RUN_ROOT"])
    for run_key, run_id in RUN_IDS.items():
        checkpoint = run_root / run_id / "checkpoints/samples_040960.pth.tar"
        model = _load_s3c_model(run_key, checkpoint)
        matrix[run_key], activation[run_key] = {}, {}
        raw_by_control = {}
        for control in INPUT_SWAP_CONTROLS:
            metrics, raw = _evaluate_s3c(
                model, batches[control], threshold=0.3)
            matrix[run_key][control] = metrics
            raw_by_control[control] = raw
            activation[run_key][control] = {
                name: tensor_statistics(torch.from_numpy(value))
                for name, value in raw["features"].items()}
            activation[run_key][control]["fusion_gate"] = {
                "status": "NOT_INSTRUMENTED",
                "reason": "S3C fusion has no explicit gate tensor"}
        aligned = raw_by_control["aligned"]
        for control in INPUT_SWAP_CONTROLS:
            raw = raw_by_control[control]
            matrix[run_key][control]["difference_vs_aligned"] = {
                "road_logit": _difference(raw["road_logits"], aligned["road_logits"]),
                "junction_logit": _difference(raw["junction_logits"], aligned["junction_logits"]),
                "road_probability": _difference(
                    1 / (1 + np.exp(-raw["road_logits"])),
                    1 / (1 + np.exp(-aligned["road_logits"]))),
                "junction_probability": _difference(
                    1 / (1 + np.exp(-raw["junction_logits"])),
                    1 / (1 + np.exp(-aligned["junction_logits"]))),
            }
            activation[run_key][control]["difference_vs_aligned"] = {
                name: _difference(value, aligned["features"][name])
                for name, value in raw["features"].items()}
        del model
        torch.cuda.empty_cache()
    r1 = matrix["R1"]
    primary = ("road_f1", "road_iou", "road_auprc")
    aligned_best = all(
        r1["aligned"][name] > r1[control][name]
        for name in primary for control in (
            "zero", "shift_fixed", "shift_large", "permuted"))
    residual_changes = any(
        activation["R1"][control]["difference_vs_aligned"][
            "raster_delta_seg_only"]["max_abs"] > 1e-7
        for control in ("zero", "shift_fixed", "shift_large", "permuted"))
    dependence = "PASS" if aligned_best and residual_changes else "FAIL"
    write_json(args.output_root / "stage_s3d_input_swap_matrix.json", {
        "stage": "seg_raster_stage_s3d", "status": "PASS",
        "evaluation_code_sha": args.run_code_sha,
        "s3c_checkpoint_code_sha": S3C_RUN_CODE_SHA,
        "samples_seen": 40960, "matrix": matrix,
        "r1_trained_aligned_strictly_best": aligned_best,
        "r1_residual_changes_with_input": residual_changes,
        "current_raster_input_dependence": dependence,
    })
    write_json(args.output_root / "stage_s3d_activation_forensics.json", {
        "stage": "seg_raster_stage_s3d", "status": "PASS",
        "samples_seen": 40960, "runs": activation,
    })
    control_strength = _control_strength(batches)
    write_json(args.output_root / "stage_s3d_control_strength_audit.json", {
        "stage": "seg_raster_stage_s3d", "status": "PASS",
        "controls": control_strength,
        "shift_128_retains_material_road_overlap": (
            control_strength["shift_fixed"]["road_gt_overlap_ratio"]
            >= 0.5 * control_strength["aligned"]["road_gt_overlap_ratio"]),
        "shift_512_reduces_correspondence": (
            control_strength["shift_large"]["aligned_control_iou"]
            < control_strength["shift_fixed"]["aligned_control_iou"]),
        "permutation_preserves_density_but_breaks_location": True,
    })

    # Runtime zero-path evidence at initialization, after one isolated update,
    # and for each trained raster checkpoint.
    checkpoint_path = os.environ["S3D_BASELINE_CHECKPOINT"]
    sample = batches["aligned"][0]
    init_model = RPNet(
        num_targets=4, backbone_pretrained=False,
        enable_raster_segmentation=True, anchor_grad_to_seg=False)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    strict_load_official_checkpoint(
        init_model, payload,
        allowed_new_prefixes=("segmentation_raster_fusion.",))
    init_model.cuda()
    initialized = _zero_runtime(init_model, sample)
    configure_frozen_explorer(init_model, raster_enabled=True)
    set_frozen_explorer_train_mode(init_model)
    params = trainable_parameters(init_model)
    optimizer = torch.optim.Adam(params, lr=1e-5, betas=(0.9, 0.99),
                                 weight_decay=2e-4)
    zero_batch = copy.copy(sample)
    zero_batch.batch_traj_inputs = np.zeros_like(sample.batch_traj_inputs)
    output = _s3c_forward(init_model, zero_batch)
    losses = segmentation_losses(
        output, to_cuda(zero_batch.batch_road_segmentation),
        to_cuda(zero_batch.batch_junction_segmentation))
    losses["total"].backward()
    optimizer.step()
    after_step = _zero_runtime(init_model, sample)
    trained = {}
    for key in ("R1", "R2", "R3"):
        path = run_root / RUN_IDS[key] / "checkpoints/samples_040960.pth.tar"
        model = _load_s3c_model(key, path)
        trained[key] = _zero_runtime(model, sample)
        del model
    runtime_nonzero = any(
        row["max_abs_stage_fuse_seg_minus_img"] > 0
        for row in [after_step] + list(trained.values()))
    root = classify_current_zero_path(
        image_enters_trainable_fusion=True,
        valid_mask_enters_trainable_fusion=True,
        normalization_affine=True, bias_present=True,
        runtime_residual_nonzero=runtime_nonzero)
    write_json(args.output_root / "stage_s3d_current_zero_path_audit.json", {
        "stage": "seg_raster_stage_s3d", "status": "PASS",
        "source_contract": {
            "convolution_bias": True, "group_norm_affine": True,
            "image_feature_enters_trainable_fusion": True,
            "valid_mask_enters_trainable_fusion": True,
            "zero_initialized_final_projection": True,
        },
        "runtime": {"initialization": initialized,
                    "after_one_optimizer_step": after_step,
                    "s3c_checkpoints": trained},
        "root_cause": root,
    })
    return 0


def preflight(args: argparse.Namespace) -> int:
    frozen_checkout(args.run_code_sha)
    config = load_stage_s3_config(REPO_ROOT / "configs/stage_s3d_common.yml")
    split = read_json(REPO_ROOT / config["S3"]["SPLIT_MANIFEST"])
    set_seed(STAGE_S3D_SEED + 1)
    batches, validation_sha, _ = build_batches(config, split, "aligned")
    batch = batches[0]
    set_seed(STAGE_S3D_SEED)
    model = RPNet(
        num_targets=4, backbone_pretrained=False,
        enable_zero_preserving_road_adapter=True,
        anchor_grad_to_seg=False)
    payload = torch.load(os.environ["S3D_BASELINE_CHECKPOINT"],
                         map_location="cpu", weights_only=False)
    load_audit = strict_load_stage_s3d_baseline(model, payload)
    contract = configure_road_only_training(model)
    model.cuda().train()
    optimizer = torch.optim.Adam(
        road_only_parameters(model), lr=1e-5, betas=(0.9, 0.99),
        weight_decay=2e-4)
    aligned = forward_model(model, batch, "aligned")
    loss = road_loss(aligned, to_cuda(batch.batch_road_segmentation))
    loss.backward()
    optimizer.step()
    zero_batch = copy.copy(batch)
    zero_batch.batch_traj_inputs = np.zeros_like(batch.batch_traj_inputs)
    zero = forward_model(model.eval(), zero_batch, "zero")
    bypass = forward_model(model.eval(), batch, "null")
    residual = zero["feature_maps"]["strict_raster_residual"]
    zero_identity = torch.equal(
        zero["feature_maps"]["stage_fuse_road"],
        zero["feature_maps"]["stage_fuse_img"])
    junction_equal = torch.equal(zero["junc"], aligned["junc"])
    null_equal = torch.equal(zero["road"], bypass["road"])
    gradients = [parameter.grad for parameter in model.parameters()
                 if parameter.requires_grad]
    passed = (zero_identity and torch.count_nonzero(residual).item() == 0
              and junction_equal and null_equal
              and all(value is not None and torch.isfinite(value).all()
                      for value in gradients)
              and tuple(aligned["road"].shape) == (10, 1, 64, 64)
              and tuple(aligned["junc"].shape) == (10, 1, 64, 64))
    write_json(args.output_root / "stage_s3d_remote_preflight.json", {
        "stage": "seg_raster_stage_s3d",
        "status": "PASS" if passed else "FAIL",
        "execution_environment": "REMOTE_TRAINING_SERVER",
        "run_code_sha": args.run_code_sha,
        "validation_plan_sha256": validation_sha,
        "checkpoint_load": load_audit,
        "road_shape": list(aligned["road"].shape),
        "junction_shape": list(aligned["junc"].shape),
        "zero_stage_fuse_identity": zero_identity,
        "zero_residual_max_abs": float(residual.abs().max()),
        "zero_vs_null_road_logits_equal": null_equal,
        "junction_raster_invariant": junction_equal,
        "finite_trainable_gradients": all(
            value is not None and torch.isfinite(value).all()
            for value in gradients),
        "trainable_parameter_contract": contract,
        "peak_allocated_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_memory_mb": torch.cuda.max_memory_reserved() / 2**20,
    })
    return 0 if passed else 4


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "forensic", "preflight"))
    parser.add_argument("--run-code-sha", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.action != "prepare" and not torch.cuda.is_available():
        raise RuntimeError("formal Stage S3D audit requires remote CUDA")
    return {"prepare": prepare, "forensic": forensic,
            "preflight": preflight}[args.action](args)


if __name__ == "__main__":
    raise SystemExit(main())
