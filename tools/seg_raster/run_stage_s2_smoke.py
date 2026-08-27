"""Run deterministic CPU smoke steps for Stage S2 model variants."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.model import RPNet
from utils.seg_raster import canonicalize_raster_array, sha256_file


def _load_aligned_crop(
    aerial_path: Path,
    raster_path: Path,
    crop_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    aerial_image = Image.open(aerial_path).convert("RGB")
    raster_image = Image.open(raster_path).convert("L")
    if aerial_image.size != raster_image.size:
        raise ValueError(
            "smoke aerial/raster canvas sizes differ: {} vs {}"
            .format(aerial_image.size, raster_image.size))
    aerial = np.asarray(
        aerial_image.crop((0, 0, crop_size, crop_size)), dtype=np.float32
    ) / 255.0
    raw_raster = np.asarray(
        raster_image.crop((0, 0, crop_size, crop_size)))
    binary, valid = canonicalize_raster_array(raw_raster)
    image_tensor = torch.from_numpy(
        np.ascontiguousarray(aerial.transpose(2, 0, 1)))[None]
    raster_tensor = torch.from_numpy(binary)[None, None]
    mask_tensor = torch.from_numpy(valid)[None, None]
    return image_tensor, raster_tensor, mask_tensor


def _gradient_summary(module: torch.nn.Module | None) -> dict[str, object]:
    if module is None:
        return {
            "parameter_count": 0,
            "all_grad_non_none": True,
            "all_grad_finite": True,
            "any_grad_nonzero": False,
        }
    parameters = list(module.parameters())
    gradients = [parameter.grad for parameter in parameters]
    return {
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "all_grad_non_none": all(gradient is not None for gradient in gradients),
        "all_grad_finite": all(
            gradient is not None and torch.isfinite(gradient).all().item()
            for gradient in gradients),
        "any_grad_nonzero": any(
            gradient is not None and torch.count_nonzero(gradient).item() > 0
            for gradient in gradients),
    }


def _train_step(
    name: str,
    image: torch.Tensor,
    raster: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    raster_mode: bool,
    anchor_grad_to_seg: bool,
) -> dict[str, object]:
    torch.manual_seed(20260827)
    model = RPNet(
        num_targets=1,
        backbone_pretrained=False,
        enable_raster_segmentation=raster_mode,
        anchor_grad_to_seg=anchor_grad_to_seg,
    ).train()
    batch_image = image.repeat(2, 1, 1, 1)
    batch_raster = raster.repeat(2, 1, 1, 1) if raster_mode else None
    batch_mask = valid_mask.repeat(2, 1, 1, 1) if raster_mode else None
    walked = torch.zeros(
        2, 1, image.shape[-2] // 4, image.shape[-1] // 4)
    start = time.perf_counter()
    output = model(
        aerial_image=batch_image,
        traj_image=batch_raster,
        aerial_traj_image=None,
        neighborhood_trajectory_norm=None,
        valid_mask=None,
        walked_path=walked,
        NUM_TARGETS=1,
        model="origin",
        use_traj=False,
        trajectory_mode=("raster_seg_only" if raster_mode else "none"),
        traj_valid_mask=batch_mask,
    )
    road_target = torch.zeros_like(output["road"])
    junc_target = torch.zeros_like(output["junc"])
    anchor_target = torch.zeros_like(output["anchor"])
    loss = (
        F.binary_cross_entropy_with_logits(output["road"], road_target)
        + F.binary_cross_entropy_with_logits(output["junc"], junc_target)
        + F.binary_cross_entropy_with_logits(output["anchor"], anchor_target)
    )
    loss.backward()
    raster_module = getattr(model, "segmentation_raster_fusion", None)
    result = {
        "name": name,
        "status": "PASS",
        "loss": float(loss.detach()),
        "loss_finite": bool(torch.isfinite(loss).item()),
        "output_shapes": {
            key: list(output[key].shape)
            for key in ("road", "junc", "anchor", "anchor_lowrs")
        },
        "raster_gradient": _gradient_summary(raster_module),
        "legacy_transformer_constructed": hasattr(model, "transformer"),
        "legacy_fuse_module_traj_constructed": hasattr(
            model, "fuse_module_traj"),
        "legacy_dsf_constructed": hasattr(model, "DSF"),
        "elapsed_seconds": round(time.perf_counter() - start, 4),
    }
    del model, output, loss
    gc.collect()
    return result


def _infer_step(
    image: torch.Tensor,
    raster: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, object]:
    torch.manual_seed(20260827)
    model = RPNet(
        num_targets=1,
        backbone_pretrained=False,
        enable_raster_segmentation=True,
    ).eval()
    start = time.perf_counter()
    with torch.no_grad():
        output = model(
            aerial_image=image,
            traj_image=raster,
            aerial_traj_image=None,
            neighborhood_trajectory_norm=None,
            valid_mask=None,
            walked_path=None,
            test=True,
            model="origin",
            use_traj=False,
            trajectory_mode="raster_seg_only",
            traj_valid_mask=valid_mask,
        )
    result = {
        "status": "PASS",
        "uses_xian_canonical_raster": True,
        "trajectory_sequence_required": False,
        "road_shape": list(output["road"].shape),
        "junction_shape": list(output["junc"].shape),
        "road_logits_finite": bool(torch.isfinite(output["road"]).all()),
        "junction_logits_finite": bool(torch.isfinite(output["junc"]).all()),
        "elapsed_seconds": round(time.perf_counter() - start, 4),
    }
    del model, output
    gc.collect()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aerial", type=Path, required=True)
    parser.add_argument("--raster", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(1)
    image, raster, valid_mask = _load_aligned_crop(
        args.aerial, args.raster, args.crop_size)
    result = {
        "schema_version": "1.0.0",
        "stage": "S2",
        "device": "cpu",
        "seed": 20260827,
        "input": {
            "aerial_path": "${READ_ONLY_XIAN_AERIAL_CANVAS}",
            "trajectory_raster_path": "${CANONICAL_XIAN_RASTER_CANVAS}",
            "aerial_sha256": sha256_file(args.aerial),
            "trajectory_raster_sha256": sha256_file(args.raster),
            "crop_origin_xy": [0, 0],
            "crop_size": args.crop_size,
            "raster_values": sorted(float(v) for v in torch.unique(raster)),
        },
        "train_steps": [
            _train_step(
                "image_only", image, raster, valid_mask,
                raster_mode=False, anchor_grad_to_seg=True),
            _train_step(
                "raster_seg_only_detach", image, raster, valid_mask,
                raster_mode=True, anchor_grad_to_seg=False),
            _train_step(
                "raster_seg_only_joint", image, raster, valid_mask,
                raster_mode=True, anchor_grad_to_seg=True),
        ],
        "infer_step": _infer_step(image, raster, valid_mask),
    }
    result["status"] = (
        "PASS" if all(
            item["status"] == "PASS" and item["loss_finite"]
            for item in result["train_steps"]
        ) and result["infer_step"]["status"] == "PASS" else "FAIL")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
