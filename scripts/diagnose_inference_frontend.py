"""Run only the segmentation/start-point frontend of road_self inference."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "baseline_image_only.yml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data_self" / "baseline_image_only" / "frontend_diagnosis.json",
    )
    parser.add_argument(
        "--batch-size-seg",
        type=int,
        default=None,
        help="Optional diagnostic-only override for TEST.BATCH_SIZE_SEG.",
    )
    parser.add_argument(
        "--gpu-id",
        default=None,
        help="Optional diagnostic-only TEST.GPU_ID override.",
    )
    return parser.parse_args()


def _stats(value: np.ndarray) -> dict:
    return {
        "shape": list(value.shape),
        "finite": bool(np.isfinite(value).all()),
        "min": float(value.min()),
        "max": float(value.max()),
        "mean": float(value.mean()),
        "nonzero": int(np.count_nonzero(value)),
    }


def main() -> int:
    args = _parse_args()
    infer_config = args.config
    if args.gpu_id is not None:
        with args.config.open("r", encoding="utf-8") as handle:
            runtime_config = yaml.load(handle, Loader=yaml.UnsafeLoader)
        runtime_config["TEST"]["GPU_ID"] = str(args.gpu_id)
        infer_config = args.output.with_suffix(".runtime.yml")
        infer_config.parent.mkdir(parents=True, exist_ok=True)
        with infer_config.open("w", encoding="utf-8") as handle:
            yaml.dump(runtime_config, handle, sort_keys=False)
    original_argv = list(sys.argv)
    sys.argv = [os.fspath(REPO_ROOT / "infer.py"), "--config", os.fspath(infer_config)]
    try:
        import infer
    finally:
        sys.argv = original_argv

    if args.batch_size_seg is not None:
        if args.batch_size_seg <= 0:
            raise ValueError("--batch-size-seg must be positive")
        infer.cfg.TEST.BATCH_SIZE_SEG = args.batch_size_seg

    regions = infer.get_regions(infer.cfg.DIR.TEST_REGION_PATH)
    if infer.cfg.TEST.SINGLE_REGION:
        name = str(infer.cfg.TEST.SINGLE_REGION)
        regions = {name: regions[name]}

    net = infer.prepare_net().eval()
    with torch.no_grad():
        road_maps, junction_maps = infer.infer_segmentation(net, list(regions))

    report = {
        "purpose": "inference frontend diagnosis; graph exploration not run",
        "config": os.fspath(args.config.resolve()),
        "checkpoint": os.fspath(infer.resolve_inference_checkpoint_path(
            infer.cfg, require_exists=True
        )),
        "regions": {},
    }
    for region_name in regions:
        junction_points = infer.junction_nms(region_name, junction_maps[region_name])
        road_filtered = infer.road_seg_region_filter(region_name, road_maps[region_name])
        report["regions"][region_name] = {
            "road_probability": _stats(road_maps[region_name]),
            "junction_probability": _stats(junction_maps[region_name]),
            "junction_start_count": len(junction_points),
            "junction_starts": [list(map(int, point)) for point in junction_points[:1000]],
            "road_start_filter": _stats(road_filtered),
            "thresholds": {
                "junction": float(infer.cfg.TEST.BINARIZE_MAP.JUNC_SEG_THRESHOLE),
                "road": float(infer.cfg.TEST.BINARIZE_MAP.ROAD_SEG_THRESHOLE),
                "min_bad_road_area": int(infer.cfg.TEST.BINARIZE_MAP.MIN_BAD_ROAD_AREA),
            },
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("saved {}".format(args.output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
