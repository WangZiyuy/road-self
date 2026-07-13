import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import OSMDataset


def build_grid_index(pixel_trajectories, cell_size):
    cell_to_traj_ids = defaultdict(list)
    for traj_id, traj in enumerate(tqdm(pixel_trajectories, desc="build grid index")):
        traj = np.asarray(traj, dtype=np.float32)
        if len(traj) == 0:
            continue
        cells = np.floor(traj / cell_size).astype(np.int32)
        cells = np.unique(cells, axis=0)
        for cx, cy in cells:
            cell_to_traj_ids[(int(cx), int(cy))].append(traj_id)

    sorted_cells = sorted(cell_to_traj_ids)
    cell_array = np.asarray(sorted_cells, dtype=np.int32)
    offsets = [0]
    ids = []
    for cell in sorted_cells:
        cell_ids = sorted(set(cell_to_traj_ids[cell]))
        ids.extend(cell_ids)
        offsets.append(len(ids))

    return (
        cell_array,
        np.asarray(offsets, dtype=np.int64),
        np.asarray(ids, dtype=np.int32),
    )


def flatten_trajectories(pixel_trajectories):
    offsets = [0]
    parts = []
    for traj in pixel_trajectories:
        traj = np.asarray(traj, dtype=np.float32)
        parts.append(traj)
        offsets.append(offsets[-1] + len(traj))
    points = np.concatenate(parts, axis=0) if parts else np.zeros((0, 2), dtype=np.float32)
    return points, np.asarray(offsets, dtype=np.int64)


def load_flattened_trajectories(path):
    data = np.load(path, allow_pickle=False)
    points = data["points"].astype(np.float32, copy=False)
    offsets = data["offsets"].astype(np.int64, copy=False)
    return [
        points[offsets[i]:offsets[i + 1]]
        for i in range(len(offsets) - 1)
    ]


def main():
    parser = argparse.ArgumentParser(description="Prepare xian trajectory cache for VecRoad_self.")
    parser.add_argument("--region", default="xian")
    parser.add_argument("--data-root", default="data_self")
    parser.add_argument("--traj-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cell-size", type=int, default=256)
    parser.add_argument("--point-source", choices=["raw", "matched"], default="raw")
    parser.add_argument(
        "--reuse-pixel-cache",
        action="store_true",
        help="Reuse output pixel_trajs.npz if it already exists and only rebuild the grid index.",
    )
    args = parser.parse_args()

    traj_dir = args.traj_dir or os.path.join(args.data_root, "input", "traj_piece", args.region)
    output_dir = args.output_dir or os.path.join(args.data_root, "input", "traj_prepared", args.region)
    os.makedirs(output_dir, exist_ok=True)
    pixel_cache = os.path.join(output_dir, "pixel_trajs.npz")

    bbox = OSMDataset.get_region_bbox_for_traj(args.region, data_root=args.data_root)
    if args.reuse_pixel_cache and os.path.isfile(pixel_cache):
        pixel_trajectories = load_flattened_trajectories(pixel_cache)
        points = np.load(pixel_cache, allow_pickle=False)["points"]
    else:
        trajectories = OSMDataset.get_all_traj_pieces_from_txt(
            traj_dir,
            bbox=bbox,
            txt_point_source=args.point_source)
        pixel_trajectories = OSMDataset.all_traj_to_all_pixel_traj(
            trajectories,
            args.region,
            data_root=args.data_root)

        points, offsets = flatten_trajectories(pixel_trajectories)
        np.savez(pixel_cache, points=points, offsets=offsets)

    cells, cell_offsets, traj_ids = build_grid_index(pixel_trajectories, args.cell_size)
    np.savez(
        os.path.join(output_dir, "grid_index.npz"),
        cells=cells,
        cell_offsets=cell_offsets,
        traj_ids=traj_ids)

    meta = {
        "region": args.region,
        "data_root": args.data_root,
        "traj_dir": traj_dir,
        "point_source": args.point_source,
        "bbox": {
            "lat_min": bbox[0],
            "lon_min": bbox[1],
            "lat_max": bbox[2],
            "lon_max": bbox[3],
        },
        "cell_size": args.cell_size,
        "trajectory_count": len(pixel_trajectories),
        "point_count": int(points.shape[0]),
        "grid_cell_count": int(cells.shape[0]),
    }
    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
