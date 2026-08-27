#!/usr/bin/env python3
"""Audit real aerial/trajectory raster files and their loader contracts."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
from typing import Any


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def image_info(path: Path, *, enumerate_values: bool) -> dict[str, Any]:
    from PIL import Image
    import numpy as np
    with Image.open(path) as image:
        array = np.asarray(image)
        channels = 1 if array.ndim == 2 else int(array.shape[2])
        values = None
        counts = None
        if enumerate_values:
            unique, unique_counts = np.unique(array, return_counts=True)
            if len(unique) <= 512:
                values = [int(value) if np.issubdtype(unique.dtype, np.integer) else float(value) for value in unique]
                counts = [int(value) for value in unique_counts]
        return {
            "path": str(path.resolve()), "exists": True, "filename": path.name,
            "width": int(image.width), "height": int(image.height),
            "mode": image.mode, "channels": channels, "dtype": str(array.dtype),
            "min": float(array.min()), "max": float(array.max()),
            "unique_values": values, "unique_counts": counts,
            "strict_binary_0_255": values is not None and set(values).issubset({0, 255}),
            "pil_metadata": {key: str(value) for key, value in image.info.items()},
            "crs": None, "geotransform": None, "pixel_resolution": None,
            "pixel_origin": None, "nodata": None,
            "metadata_status": "UNAVAILABLE_FOR_PNG_UNLESS_SIDECAR_EXISTS",
            "sha256": sha256(path), "size_bytes": path.stat().st_size,
        }


def locate(repo: Path, path: str, needle: str) -> dict[str, Any]:
    target = repo / path
    if not target.is_file():
        return {"path": path, "line": None, "snippet": "", "status": "NOT_PRESENT"}
    for number, line in enumerate(target.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if needle in line:
            return {"path": path, "line": number, "snippet": line.strip(), "status": "FOUND"}
    return {"path": path, "line": None, "snippet": "", "status": "NOT_FOUND"}


def configured_paths(repo: Path) -> list[dict[str, Any]]:
    results = []
    regex = re.compile(r"^\s*([A-Z0-9_]*(?:TRAJ|IMAGERY)[A-Z0-9_]*)\s*:\s*[\"']?([^\"'#]+)", re.I)
    for path in sorted((repo / "configs").glob("*.yml")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            match = regex.search(line)
            if match:
                results.append({
                    "config": path.relative_to(repo).as_posix(), "line": line_number,
                    "key": match.group(1), "value": match.group(2).strip(),
                })
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--source-repo", type=Path, default=Path(r"E:\Code\VecRoad_self"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    source = args.source_repo.resolve()
    input_root = source / "data_self" / "input"
    directory_names = [
        "imagery", "imagery_test", "imagery_8192", "traj", "traj_test",
        "traj_piece", "traj_prepared", "traj_structured", "regions",
    ]
    directories = [{
        "name": name, "path": str((input_root / name).resolve()),
        "exists": (input_root / name).is_dir(),
    } for name in directory_names]

    aerial_files = sorted((input_root / "imagery").glob("*.png")) if (input_root / "imagery").is_dir() else []
    raster_files = sorted((input_root / "traj").glob("*.png")) if (input_root / "traj").is_dir() else []
    aerial = [image_info(path, enumerate_values=False) for path in aerial_files]
    rasters = [image_info(path, enumerate_values=True) for path in raster_files]
    aerial_by_name = {item["filename"]: item for item in aerial}
    raster_by_name = {item["filename"]: item for item in rasters}
    matches = []
    for name in sorted(set(aerial_by_name) | set(raster_by_name)):
        left = aerial_by_name.get(name)
        right = raster_by_name.get(name)
        matches.append({
            "filename": name, "aerial_exists": left is not None,
            "raster_exists": right is not None,
            "same_width_height": bool(left and right and left["width"] == right["width"] and left["height"] == right["height"]),
            "aerial_shape_wh": [left["width"], left["height"]] if left else None,
            "raster_shape_wh": [right["width"], right["height"]] if right else None,
            "status": "PASS" if left and right and left["width"] == right["width"] and left["height"] == right["height"]
            else "DATA_ALIGNMENT_BLOCKER" if left and right else "UNPAIRED",
        })

    metadata = []
    regions_dir = input_root / "regions"
    if regions_dir.is_dir():
        for path in sorted(regions_dir.glob("*metadata*.json")):
            try:
                parsed = json.loads(path.read_text(encoding="utf-8"))
                metadata.append({"path": str(path.resolve()), "status": "PARSED", "content": parsed})
            except Exception as exc:
                metadata.append({"path": str(path.resolve()), "status": "PARSE_FAIL", "error": str(exc)})

    configured = configured_paths(repo)
    hardcoded_generation = []
    generator = repo / "data_self" / "gen_dataset.py"
    if generator.is_file():
        for line_number, line in enumerate(generator.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if "D:/DataSet" in line:
                hardcoded_generation.append({"path": "data_self/gen_dataset.py", "line": line_number, "snippet": line.strip()})

    xian_raster = raster_by_name.get("xian_0_0.png")
    xian_aerial = aerial_by_name.get("xian_0_0.png")
    xian_values = xian_raster.get("unique_values") if xian_raster else None
    direct_normalization = [float(value) / 255.0 for value in xian_values] if xian_values else None
    mismatches = [item for item in matches if item["status"] == "DATA_ALIGNMENT_BLOCKER"]
    payload = {
        "stage": "seg_raster_stage_s0", "generated_at": now_iso(),
        "audited_repository": str(repo), "read_only_asset_source": str(source),
        "configured_paths": configured, "directories": directories,
        "aerial_files": aerial, "trajectory_raster_files": rasters,
        "filename_matches": matches, "region_metadata": metadata,
        "hardcoded_data_generation_paths": hardcoded_generation,
        "xian_finding": {
            "aerial": xian_aerial, "raster": xian_raster,
            "shape_match": bool(xian_aerial and xian_raster and xian_aerial["width"] == xian_raster["width"] and xian_aerial["height"] == xian_raster["height"]),
            "direct_divide_255_values": direct_normalization,
            "status": "DATA_ALIGNMENT_BLOCKER" if xian_aerial and xian_raster and (xian_aerial["width"] != xian_raster["width"] or xian_aerial["height"] != xian_raster["height"]) else "NOT_VERIFIED",
        },
        "coordinate_contract": {
            "train_aerial_and_raster_same_region_argument": True,
            "train_same_crop_indices": True,
            "train_both_loaders_swap_axes_0_1": True,
            "silent_resize_present": False,
            "silent_crop_present": True,
            "silent_crop_detail": "Both sources are sliced with the same tile_origin and WINDOW_SIZE, but unequal source extents are not validated.",
            "infer_aerial_loader": "PIL whole-region TEST_IMAGERY_DIR, swapaxes(0,1), ToTensor, crop",
            "infer_raster_loader": "PIL whole-region TEST_TRAJ_DIR, swapaxes(0,1), ToTensor, crop",
            "train_and_infer_same_loader": False,
            "geographic_registration_proven": False,
            "metadata_status": "UNAVAILABLE; shape/crop agreement alone cannot prove CRS/geotransform registration.",
        },
        "evidence": [
            locate(repo, "utils/tileloader.py", "sat_im = sat_im.swapaxes(0, 1)"),
            locate(repo, "utils/model_utils.py", "big_traj_img = self.tile_data['cache'].get_traj"),
            locate(repo, "utils/model_utils.py", ".astype('float32') / 255.0"),
            locate(repo, "infer.py", "cfg.DIR.TEST_TRAJ_DIR"),
            locate(repo, "infer.py", "traj_map = traj_map.swapaxes(0, 1)"),
        ],
        "answers": {
            "same_region": "PARTIAL: loader uses the same region key; geographic registration metadata is unavailable.",
            "shape_exact": "FAIL" if mismatches else "NOT_VERIFIED",
            "same_crop_coordinate_order": "PASS_STATIC_CODE",
            "strict_binary": "FAIL" if xian_values and not set(xian_values).issubset({0, 255}) else "NOT_VERIFIED",
            "0_128_255_divide_255": "PASS" if xian_values == [0, 128, 255] else "FAIL" if xian_values else "NOT_PRESENT",
            "silent_transform": "transpose and fixed-coordinate crop exist; no resize or stretch found",
            "same_train_infer_loader": "FAIL",
            "train_traj_dir_exists": next((item["exists"] for item in directories if item["name"] == "traj"), False),
            "test_traj_dir_exists": next((item["exists"] for item in directories if item["name"] == "traj_test"), False),
        },
        "overall_status": "DATA_ALIGNMENT_BLOCKER" if mismatches else "NOT_VERIFIED",
        "prohibited_actions_respected": ["no resize", "no center crop", "no asset copy", "no asset mutation"],
    }
    write_json(repo / "artifacts" / "stage_s0_raster_alignment_audit.json", payload)
    print(json.dumps({
        "aerial_files": len(aerial), "raster_files": len(rasters),
        "mismatches": len(mismatches), "overall_status": payload["overall_status"],
    }, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
