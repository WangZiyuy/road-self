#!/usr/bin/env python3
"""Validate road_self image-only RPNet against the official VecRoad flow."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.model import RPNet  # noqa: E402


OUTPUT_KEYS = ("road", "junc", "anchor", "anchor_lowrs")


def _upsample(value: torch.Tensor, scale: float) -> torch.Tensor:
    return F.interpolate(
        value, scale_factor=scale, mode="bilinear", align_corners=True)


def official_reference_forward(
        net: RPNet,
        aerial_image: torch.Tensor,
        walked_path_small: torch.Tensor,
        num_targets: int) -> Dict[str, torch.Tensor]:
    """Independent transcription of official VecRoad ``RPNet.forward``."""
    stage_1 = net.stage_1(aerial_image)
    stage_2 = net.stage_2(net.maxpool(stage_1))
    stage_3 = net.stage_3(stage_2)
    stage_4 = net.stage_4(stage_3)
    stage_5 = net.stage_5(stage_4)

    stage_fuse = net.conv_fuse(torch.cat([
        net.conv_2_side(stage_2),
        _upsample(net.conv_3_side(stage_3), 2),
        _upsample(net.conv_4_side(stage_4), 2),
        _upsample(net.conv_5_side(stage_5), 2)], dim=1))
    road_fts = net.road_seg(stage_fuse)
    road_final = net.conv_road_final(road_fts)
    junc_fts = net.junc_seg(stage_fuse)
    junc_final = net.conv_junc_final(junc_fts)

    placeholder = torch.zeros(
        stage_fuse.shape[0],
        32 * (net.num_targets - 1),
        stage_fuse.shape[2],
        stage_fuse.shape[3],
        device=stage_fuse.device,
        dtype=stage_fuse.dtype)
    recurrent = torch.cat([
        stage_fuse,
        road_fts,
        junc_fts,
        walked_path_small,
        placeholder], dim=1)

    anchor_features = None
    anchors = []
    anchors_lowrs = []
    for index in range(num_targets):
        next_step = net.fuse_module(recurrent)
        anchors_lowrs.append(_upsample(net.next_step_final(next_step), 4))
        decoded_4 = net.decoders[0](_upsample(stage_4, 2), next_step)
        decoded_3 = net.decoders[1](_upsample(stage_3, 2), decoded_4)
        decoded_2 = net.decoders[2](
            _upsample(stage_2, 2), _upsample(decoded_3, 2))
        decoded_1 = net.decoders[3](
            _upsample(stage_1, 2), _upsample(decoded_2, 2))

        channel_index = -(net.num_targets - index - 1) * 32
        if index < net.num_targets - 1:
            pooled = net.avgpool4(decoded_1)
            anchor_features = (
                pooled if anchor_features is None
                else anchor_features + pooled)
            recurrent[
                :,
                channel_index:channel_index + 32
                if channel_index + 32 != 0 else None,
                :,
                :,
            ] = anchor_features
        anchors.append(net.conv_final(decoded_1))

    return {
        "road": road_final,
        "junc": junc_final,
        "anchor": torch.cat(anchors, dim=1),
        "anchor_lowrs": torch.cat(anchors_lowrs, dim=1)}


def _extract_state_dict(payload: Any) -> Mapping[str, torch.Tensor]:
    state_dict = (
        payload.get("state_dict", payload)
        if isinstance(payload, Mapping) else payload)
    if not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint does not contain a state_dict mapping")
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {
            key[len("module."):]: value
            for key, value in state_dict.items()}
    return state_dict


def validate_alignment(
        model: RPNet,
        aerial: torch.Tensor,
        walked_small: torch.Tensor,
        tolerance: float) -> Dict[str, Any]:
    model.eval()
    with torch.no_grad():
        production = model(
            aerial_image=aerial,
            traj_image=None,
            aerial_traj_image=None,
            neighborhood_trajectory_norm=None,
            valid_mask=None,
            walked_path=walked_small,
            NUM_TARGETS=model.num_targets,
            test=False,
            model="origin",
            use_traj=False)
        reference = official_reference_forward(
            model, aerial, walked_small, model.num_targets)

    outputs = {}
    passed = True
    for key in OUTPUT_KEYS:
        difference = (production[key] - reference[key]).abs()
        maximum = float(difference.max().cpu())
        mean = float(difference.mean().cpu())
        finite = bool(
            torch.isfinite(production[key]).all()
            and torch.isfinite(reference[key]).all())
        item_passed = finite and maximum <= tolerance
        passed = passed and item_passed
        outputs[key] = {
            "shape": list(production[key].shape),
            "max_abs_diff": maximum,
            "mean_abs_diff": mean,
            "finite": finite,
            "passed": item_passed}
    return {"passed": passed, "outputs": outputs}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--input-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.input_size < 64 or args.input_size % 32 != 0:
        raise ValueError("--input-size must be >=64 and divisible by 32")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = RPNet(
        num_targets=4,
        backbone_pretrained=False,
        enable_trajectory_modules=False)
    model_keys = tuple(model.state_dict())
    report: Dict[str, Any] = {
        "checkpoint": None,
        "state_dict_key_count": len(model_keys),
        "trajectory_parameters_present": any(
            key.startswith((
                "transformer.",
                "fuse_module_traj.",
                "DSF.",
                "missing_traj_feature"))
            for key in model_keys)}

    if args.checkpoint is not None:
        checkpoint = args.checkpoint.resolve(strict=True)
        payload = torch.load(str(checkpoint), map_location="cpu")
        state_dict = _extract_state_dict(payload)
        model.load_state_dict(state_dict, strict=True)
        report["checkpoint"] = str(checkpoint)
        report["checkpoint_state_dict_key_count"] = len(state_dict)

    model = model.to(device)
    aerial = torch.randn(
        args.batch_size,
        3,
        args.input_size,
        args.input_size,
        device=device)
    walked = torch.randn(
        args.batch_size,
        1,
        args.input_size // 4,
        args.input_size // 4,
        device=device)
    report["forward"] = validate_alignment(
        model, aerial, walked, args.tolerance)
    report["passed"] = bool(
        report["forward"]["passed"]
        and not report["trajectory_parameters_present"])

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
