import argparse
import os

import cv2
import numpy as np


def read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.float32) / 255.0


def component_stats(prob_map, threshold, max_area=None):
    mask = (prob_map >= threshold).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    if max_area is not None:
        kept = areas <= max_area
    else:
        kept = np.ones_like(areas, dtype=bool)
    return {
        "pixels": int(mask.sum()),
        "ratio": float(mask.mean()),
        "components": int(len(areas)),
        "kept_components": int(kept.sum()),
        "area_min": int(areas.min()) if len(areas) else 0,
        "area_median": float(np.median(areas)) if len(areas) else 0.0,
        "area_max": int(areas.max()) if len(areas) else 0,
    }


def print_prob_stats(name, img):
    valid = img[img > 0]
    print(f"{name}: shape={img.shape}, min={img.min():.6f}, max={img.max():.6f}, mean={img.mean():.6f}")
    if valid.size:
        qs = np.percentile(valid, [50, 75, 90, 95, 99])
        print(
            f"  nonzero={valid.size}, nonzero_ratio={valid.size / img.size:.6f}, "
            f"p50={qs[0]:.4f}, p75={qs[1]:.4f}, p90={qs[2]:.4f}, p95={qs[3]:.4f}, p99={qs[4]:.4f}"
        )
    else:
        print("  nonzero=0")


def save_overlay(raw_path, road, junc, out_path, road_threshold, junc_threshold):
    raw = cv2.imread(raw_path, cv2.IMREAD_COLOR)
    if raw is None:
        raise FileNotFoundError(raw_path)
    if raw.shape[:2] != road.shape:
        raw = cv2.resize(raw, (road.shape[1], road.shape[0]), interpolation=cv2.INTER_LINEAR)
    overlay = raw.copy()
    road_mask = road >= road_threshold
    junc_mask = junc >= junc_threshold
    overlay[road_mask] = (0.35 * overlay[road_mask] + 0.65 * np.array([0, 255, 0])).astype(np.uint8)
    overlay[junc_mask] = (0.25 * overlay[junc_mask] + 0.75 * np.array([0, 0, 255])).astype(np.uint8)
    cv2.imwrite(out_path, overlay)
    print(f"overlay saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Inspect VecRoad segmentation outputs.")
    parser.add_argument("--seg-dir", default="data_self/segmentation/50.2047")
    parser.add_argument("--region", default="xian")
    parser.add_argument("--road-threshold", type=float, default=0.3)
    parser.add_argument("--junc-threshold", type=float, default=0.3)
    parser.add_argument("--junc-sweep", default="0.05,0.1,0.15,0.2,0.25,0.3")
    parser.add_argument("--anchor-max-area", type=int, default=1000)
    parser.add_argument("--raw-image", default=None)
    parser.add_argument("--overlay-out", default=None)
    args = parser.parse_args()

    road_path = os.path.join(args.seg_dir, "road", args.region + ".png")
    junc_path = os.path.join(args.seg_dir, "junction", args.region + ".png")
    nms_path = os.path.join(args.seg_dir, "junc_nms", args.region + ".png")

    road = read_gray(road_path)
    junc = read_gray(junc_path)

    print_prob_stats("road", road)
    print_prob_stats("junction", junc)
    print("road threshold stats:", component_stats(road, args.road_threshold))
    print("junction threshold stats:", component_stats(junc, args.junc_threshold, args.anchor_max_area))
    if args.junc_sweep:
        print("junction threshold sweep:")
        for threshold_text in args.junc_sweep.split(","):
            threshold = float(threshold_text)
            stats = component_stats(junc, threshold, args.anchor_max_area)
            print(
                f"  th={threshold:.3f}: pixels={stats['pixels']}, "
                f"components={stats['components']}, kept={stats['kept_components']}, "
                f"area_median={stats['area_median']:.1f}, area_max={stats['area_max']}"
            )

    if os.path.exists(nms_path):
        nms = read_gray(nms_path)
        print(f"junc_nms points: {int(np.count_nonzero(nms))}")
    else:
        print(f"junc_nms not found: {nms_path}")

    if args.raw_image and args.overlay_out:
        save_overlay(args.raw_image, road, junc, args.overlay_out, args.road_threshold, args.junc_threshold)


if __name__ == "__main__":
    main()
