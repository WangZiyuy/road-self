"""Build the Stage S1 raster-pair inventory from a read-only source tree."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.seg_raster.contract import AXIS_ORDER_CONTRACT, inspect_png


def _facts(source_root: Path, relative_path: str) -> Dict[str, Any]:
    path = source_root / Path(relative_path)
    if not path.is_file():
        return {"exists": False}
    return inspect_png(path)


def _nullable(facts: Dict[str, Any], key: str) -> Optional[Any]:
    return facts.get(key) if facts.get("exists") else None


def _entry(
    source_root: Path,
    *,
    region: str,
    split: str,
    aerial_path: str,
    raster_path: str,
    source_identity: str,
    lineage: Dict[str, Any],
) -> Dict[str, Any]:
    aerial = _facts(source_root, aerial_path)
    raster = _facts(source_root, raster_path)
    shape_match = bool(
        aerial.get("exists")
        and raster.get("exists")
        and aerial.get("shape_hwc", [])[:2] == raster.get("shape_hwc", [])[:2]
    )
    if not aerial.get("exists") or not raster.get("exists"):
        status = "BLOCKED_MISSING_PAIR"
    elif not shape_match:
        status = "BLOCKED_SHAPE_MISMATCH"
    else:
        status = "BLOCKED_REGISTRATION_AND_VALUE_SEMANTICS_UNPROVEN"
    return {
        "region": region,
        "split": split,
        "aerial_path": aerial_path,
        "raster_path": raster_path,
        "source_identity": source_identity,
        "aerial": aerial,
        "raster": raster,
        "aerial_sha256": _nullable(aerial, "sha256"),
        "raster_sha256": _nullable(raster, "sha256"),
        "aerial_shape_hwc": _nullable(aerial, "shape_hwc"),
        "raster_shape_hwc": _nullable(raster, "shape_hwc"),
        "aerial_dtype": _nullable(aerial, "dtype"),
        "raster_dtype": _nullable(raster, "dtype"),
        "raster_unique_values": (
            sorted(int(value) for value in (raster.get("unique_value_counts") or {}))
            if raster.get("exists")
            else None
        ),
        "shape": {
            "aerial": _nullable(aerial, "shape_hwc"),
            "raster": _nullable(raster, "shape_hwc"),
        },
        "channels": {
            "aerial": _nullable(aerial, "channels"),
            "raster": _nullable(raster, "channels"),
        },
        "dtype": {
            "aerial": _nullable(aerial, "dtype"),
            "raster": _nullable(raster, "dtype"),
        },
        "value_range": {
            "aerial": _nullable(aerial, "value_range"),
            "raster": _nullable(raster, "value_range"),
        },
        "allowed_values": {
            "aerial": list(range(256)),
            "raster": [0, 128, 255] if raster.get("exists") else None,
        },
        "normalization": {
            "aerial": "float32(value)/255.0",
            "raster": "float32(value)/255.0",
        },
        "coordinate_reference": {
            "aerial": lineage.get("aerial_coordinate_reference", "UNKNOWN"),
            "raster": "UNKNOWN",
        },
        "crs": None,
        "geotransform": None,
        "pixel_size": None,
        "pixel_origin": None,
        "image_array_pixel_origin": "upper_left",
        "axis_order": AXIS_ORDER_CONTRACT,
        "coordinate_axis_contract": "disk image width,height -> decoded array height,width,channels -> legacy loader width,height,channels via swapaxes(0,1)",
        "y_direction": "down",
        "source_lineage": lineage,
        "checksum": {
            "aerial": _nullable(aerial, "sha256"),
            "raster": _nullable(raster, "sha256"),
        },
        "grid_registration_proven": False,
        "registration_evidence": [],
        "shape_match": shape_match,
        "value_semantics": {
            "status": "UNKNOWN",
            "observed_raw_values": (
                sorted(int(value) for value in raster.get("unique_value_counts", {}))
                if raster.get("exists")
                else None
            ),
            "post_normalization_values": (
                [0.0, 128.0 / 255.0, 1.0] if raster.get("exists") else None
            ),
            "semantic_assignments": None,
        },
        "status": status,
    }


def build_manifest(source_root: Path) -> Dict[str, Any]:
    """Build evidence without writing into ``source_root``."""

    entries = [
        _entry(
            source_root,
            region="xian",
            split="train",
            aerial_path="data_self/input/imagery/xian_0_0.png",
            raster_path="data_self/input/traj/xian_0_0.png",
            source_identity="xian-current-dirty-snapshot",
            lineage={
                "snapshot": "CURRENT_DIRTY_SNAPSHOT_READ_ONLY",
                "aerial_coordinate_reference": "GCJ-02 bbox from data_self/input/regions/xian_metadata.json",
                "aerial_recipe": "raw 4300x5000 RGB pasted at (0,0) on 8192x8192 zero canvas; xian_0_0 is the upper-left 4096x4096 crop",
                "aerial_recipe_source": "scripts/prepare_xian_image.py",
                "raster_recipe": "UNKNOWN",
                "raster_value_assignment_source": "NOT_FOUND",
            },
        ),
        _entry(
            source_root,
            region="chicago",
            split="validation",
            aerial_path="data_self/input/imagery/chicago_0_0.png",
            raster_path="data_self/input/traj/chicago_0_0.png",
            source_identity="chicago-validation-unresolved",
            lineage={
                "snapshot": "CURRENT_DIRTY_SNAPSHOT_READ_ONLY",
                "aerial_coordinate_reference": "UNKNOWN",
                "configuration_source": "utils/tileloader.py validation_regions=['chicago']",
                "raster_recipe": "UNKNOWN",
            },
        ),
        _entry(
            source_root,
            region="20",
            split="test",
            aerial_path="data_self/input/imagery_test/20.png",
            raster_path="data_self/input/traj_test/20.png",
            source_identity="region-20-test",
            lineage={
                "snapshot": "CURRENT_DIRTY_SNAPSHOT_READ_ONLY",
                "aerial_coordinate_reference": "UNKNOWN",
                "configuration_source": "configs/default_self.yml plus data_self/input/test_regions.txt",
                "raster_recipe": "UNKNOWN; traj_test directory absent",
            },
        ),
        _entry(
            source_root,
            region="xian",
            split="test",
            aerial_path="data_self/input/imagery_8192/xian.png",
            raster_path="data_self/input/traj_test/xian.png",
            source_identity="xian-test-profile",
            lineage={
                "snapshot": "CURRENT_DIRTY_SNAPSHOT_READ_ONLY",
                "aerial_coordinate_reference": "GCJ-02 bbox from data_self/input/regions/xian_metadata.json",
                "configuration_source": "configs/xian_self.yml",
                "raster_recipe": "UNKNOWN; traj_test directory absent",
            },
        ),
    ]
    return {
        "schema_version": "1.0.0",
        "manifest_kind": "trajectory_raster_pair_contract",
        "stage": "seg_raster_stage_s1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_root_label": "CURRENT_DIRTY_SNAPSHOT_READ_ONLY",
        "entries": entries,
        "split_separation": {
            "status": "UNPROVEN",
            "reason": "The configured validation source is absent and all required test trajectory rasters are absent; no source-backed non-leak proof exists.",
        },
        "test_data_contract": {
            "default_profile_configured_regions": ["20"],
            "xian_profile_configured_regions": ["xian"],
            "aerial_files_observed_in_default_test_directory": ["20.png", "653.png"],
            "unconfigured_observed_aerial_files": ["653.png"],
            "configured_trajectory_test_directory_exists": False,
            "test_raster_generation_source": "UNKNOWN",
            "same_deterministic_generation_flow_as_train": "UNPROVEN",
            "train_test_leakage": "UNPROVEN",
            "status": "BLOCKED_TEST_RASTER_SOURCE_MISSING",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
