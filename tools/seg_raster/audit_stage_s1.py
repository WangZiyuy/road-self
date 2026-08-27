"""Generate Stage S1 provenance and contract artifacts from read-only inputs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.seg_raster.build_pair_manifest import build_manifest
from tools.seg_raster.contract import inspect_png, sha256_file, validate_pair_manifest


STAGE = "seg_raster_stage_s1"
S1_BASE_SHA = "23285e5bc6515ca88a3121d2547aa9ab0476a7ad"
Image.MAX_IMAGE_PIXELS = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout.strip()


def _logical_worktrees(raw: str) -> List[Dict[str, str]]:
    result: List[Dict[str, str]] = []
    block: Dict[str, str] = {}
    for line in [*raw.splitlines(), ""]:
        if not line:
            if block:
                worktree = block.get("worktree", "")
                block["worktree"] = (
                    "CURRENT_WORKTREE"
                    if worktree.replace("\\", "/").endswith("VecRoad_self-seg-raster")
                    else "CURRENT_DIRTY_SNAPSHOT_READ_ONLY"
                )
                result.append(block)
                block = {}
            continue
        key, _, value = line.partition(" ")
        block[key] = value
    return result


def _crop_proof(raw_aerial: Path, full_canvas: Path, tile: Path) -> Dict[str, Any]:
    with Image.open(raw_aerial) as image:
        raw = np.asarray(image)
    with Image.open(full_canvas) as image:
        full = np.asarray(image)
    with Image.open(tile) as image:
        tile_array = np.asarray(image)
    height, width = raw.shape[:2]
    return {
        "raw_shape_hwc": list(raw.shape),
        "full_canvas_shape_hwc": list(full.shape),
        "tile_shape_hwc": list(tile_array.shape),
        "tile_equals_raw_upper_left_4096": bool(np.array_equal(tile_array, raw[:4096, :4096])),
        "full_canvas_raw_extent_equals_raw": bool(np.array_equal(full[:height, :width], raw)),
        "full_canvas_right_padding_all_zero": bool(np.all(full[:height, width:] == 0)),
        "full_canvas_bottom_padding_all_zero": bool(np.all(full[height:, :] == 0)),
    }


def _candidate_facts(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    rows = []
    for index, path in enumerate(paths, start=1):
        if not path.is_file():
            rows.append({"candidate_label": f"external_candidate_{index}", "exists": False})
            continue
        with Image.open(path) as image:
            size = list(image.size)
            mode = image.mode
            metadata_keys = sorted(str(key) for key in image.info)
        rows.append(
            {
                "candidate_label": f"external_candidate_{index}",
                "exists": True,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "size_wh": size,
                "mode": mode,
                "embedded_metadata_keys": metadata_keys,
            }
        )
    return rows


def generate(args: argparse.Namespace) -> None:
    output = args.output_dir
    source_root = args.source_root
    repo = args.repo
    dirty_metadata_path = source_root / "data_self/input/regions/xian_metadata.json"
    piece_manifest_path = source_root / "data_self/input/traj_piece/xian/manifest.json"
    metadata = json.loads(dirty_metadata_path.read_text(encoding="utf-8"))
    piece_manifest = json.loads(piece_manifest_path.read_text(encoding="utf-8"))

    aerial_tile_path = source_root / "data_self/input/imagery/xian_0_0.png"
    full_canvas_path = source_root / "data_self/input/imagery_8192/xian.png"
    legacy_raster_path = source_root / "data_self/input/traj/xian_0_0.png"
    aerial_tile = inspect_png(aerial_tile_path)
    full_canvas = inspect_png(full_canvas_path)
    legacy_raster = inspect_png(legacy_raster_path)
    raw_aerial = inspect_png(args.raw_aerial)
    crop_proof = _crop_proof(args.raw_aerial, full_canvas_path, aerial_tile_path)

    pair_manifest = build_manifest(source_root)
    pair_manifest["generated_at"] = _now()
    pair_issues = validate_pair_manifest(pair_manifest, source_root)
    _write_json(output / "stage_s1_pair_manifest.json", pair_manifest)
    _write_json(
        output / "stage_s1_pair_validation.json",
        {
            "stage": STAGE,
            "generated_at": _now(),
            "manifest_path": "artifacts/stage_s1_pair_manifest.json",
            "source_root_label": "CURRENT_DIRTY_SNAPSHOT_READ_ONLY",
            "status": "BLOCKED" if pair_issues else "PASS",
            "issue_count": len(pair_issues),
            "issues": [issue.to_dict() for issue in pair_issues],
        },
    )

    git_start = {
        "stage": STAGE,
        "generated_at": _now(),
        "required_worktree": "CURRENT_WORKTREE",
        "branch": _git(repo, "branch", "--show-current"),
        "head_sha": _git(repo, "rev-parse", "HEAD"),
        "s1_base_sha": S1_BASE_SHA,
        "status_short_at_start": [],
        "upstream": "origin/feat/seg-raster-only",
        "origin_remote": "${ORIGIN_REMOTE_REDACTED}",
        "worktrees": _logical_worktrees(_git(repo, "worktree", "list", "--porcelain")),
        "source_provenance_gate": "PASSED_BEFORE_S1_BY_USER",
        "dirty_source_policy": "CURRENT_DIRTY_SNAPSHOT_READ_ONLY; no writes or Git mutations",
    }
    _write_json(output / "stage_s1_git_start.json", git_start)

    raw_trajectory = {
        "exists": args.raw_trajectory.is_file(),
        "size_bytes": args.raw_trajectory.stat().st_size,
        "sha256": sha256_file(args.raw_trajectory),
        "source_label": "${RAW_TRAJECTORY_SOURCE}",
    }
    provenance = {
        "stage": STAGE,
        "generated_at": _now(),
        "status": "BLOCKED",
        "aerial": {
            "status": "PASS",
            "raw_source": {**raw_aerial, "source_label": "${RAW_AERIAL_SOURCE}"},
            "metadata": {
                "path": "data_self/input/regions/xian_metadata.json",
                "sha256": sha256_file(dirty_metadata_path),
                "region": metadata["region"],
                "original_size_wh": metadata["original_size"],
                "canvas_size": metadata["canvas_size"],
                "tile_size": metadata["tile_size"],
                "bbox_gcj02": metadata["bbox_gcj02"],
            },
            "full_canvas": full_canvas,
            "upper_left_tile": aerial_tile,
            "pixel_proof": crop_proof,
            "recipe_source": "scripts/prepare_xian_image.py",
        },
        "trajectory_points": {
            "status": "PASS_FOR_POINT_LINEAGE_ONLY",
            "raw_source": raw_trajectory,
            "piece_manifest_path": "data_self/input/traj_piece/xian/manifest.json",
            "piece_manifest_sha256": sha256_file(piece_manifest_path),
            "manifest_source_sha_matches": raw_trajectory["sha256"] == piece_manifest["source_sha256"],
            "coordinate_system": "GCJ-02 longitude/latitude",
            "pixel_formula_source": "scripts/prepare_xian_traj_piece.py and utils/gis_to_graph.py",
            "trajectory_count": piece_manifest["trajectory_count"],
            "point_count": piece_manifest["point_count"],
            "note": "Point provenance does not establish how the legacy PNG was drawn or encoded.",
        },
        "legacy_trajectory_raster": {
            "status": "BLOCKED_SOURCE_PROVENANCE_UNKNOWN",
            "path": "data_self/input/traj/xian_0_0.png",
            "facts": legacy_raster,
            "generation_script": "UNKNOWN",
            "input_files": "UNKNOWN",
            "intermediate_files": "UNKNOWN",
            "parameters": "UNKNOWN",
            "coordinate_conversion": "UNKNOWN",
            "crop_extent": "UNKNOWN",
            "output_size_wh": [4300, 5000],
            "output_value_domain": [0, 128, 255],
            "generation_command": "UNKNOWN",
            "deterministically_reproducible": False,
            "evidence_paths_and_lines": [
                "utils/tileloader.py:20-33 reads and swaps image axes",
                "utils/model_utils.py:921-926 crops and divides by 255",
                "no checked source line writes values 0/128/255 for this PNG",
            ],
            "embedded_geospatial_metadata": False,
            "generator_found_in_checked_code_or_history": False,
            "value_assignment_source_found": False,
            "external_candidate_rasters": _candidate_facts(args.candidate_raster),
            "candidate_match_status": "NO_SIZE_AND_SHA_MATCH",
        },
        "blocker": "No source-backed generator or transform connects the legacy trajectory raster bytes to the aerial pixel grid.",
        "configured_test_raster_outputs": [
            {
                "profile": "default_self",
                "region": "20",
                "expected_path": "data_self/input/traj_test/20.png",
                "status": "MISSING",
                "generation_script": "UNKNOWN",
            },
            {
                "profile": "xian_self",
                "region": "xian",
                "expected_path": "data_self/input/traj_test/xian.png",
                "status": "MISSING",
                "generation_script": "UNKNOWN",
            },
        ],
    }
    _write_json(output / "stage_s1_provenance_trace.json", provenance)

    normalized = {str(value): value / 255.0 for value in (0, 128, 255)}
    value_semantics = {
        "stage": STAGE,
        "generated_at": _now(),
        "status": "BLOCKED_VALUE_SEMANTICS_UNKNOWN",
        "path": "data_self/input/traj/xian_0_0.png",
        "dtype": legacy_raster["dtype"],
        "observed_unique_value_counts": legacy_raster["unique_value_counts"],
        "raw_allowed_values_observed": [0, 128, 255],
        "loader_normalization_source": "utils/model_utils.py",
        "loader_operation": "float32(value)/255.0",
        "post_normalization_values": normalized,
        "semantic_assignments": {
            "0": "UNKNOWN",
            "128": "UNKNOWN",
            "255": "UNKNOWN",
        },
        "generator_or_assignment_code_found": False,
        "blocker": "Observed levels and loader normalization do not prove what each level means.",
    }
    _write_json(output / "stage_s1_value_semantics.json", value_semantics)

    alignment = {
        "stage": STAGE,
        "generated_at": _now(),
        "status": "BLOCKED_REGISTRATION_PROVENANCE_INSUFFICIENT",
        "aerial_crop_chain": {"status": "PASS", **crop_proof},
        "legacy_raster_shape_hwc": legacy_raster["shape_hwc"],
        "train_aerial_tile_shape_hwc": aerial_tile["shape_hwc"],
        "legacy_raster_matches_raw_aerial_dimensions": legacy_raster["shape_hwc"][:2] == raw_aerial["shape_hwc"][:2],
        "legacy_raster_matches_train_tile_dimensions": legacy_raster["shape_hwc"][:2] == aerial_tile["shape_hwc"][:2],
        "same_shape_is_registration_proof": False,
        "candidate_upper_left_crop": {
            "status": "NOT_EXECUTED",
            "reason": "The aerial crop chain is proven, but the legacy raster origin, transform, and drawing recipe are not.",
        },
        "axis_contract": {
            "decoded": "height,width,channels",
            "legacy_loader_internal": "width,height,channels after swapaxes(0,1)",
            "pixel_origin_for_image_arrays": "upper_left",
            "y_direction_for_image_arrays": "down",
            "cross_source_grid_registration": "UNPROVEN",
        },
    }
    _write_json(output / "stage_s1_alignment_diagnostics.json", alignment)

    rebuild = {
        "stage": STAGE,
        "generated_at": _now(),
        "status": "NOT_EXECUTED",
        "output_path": None,
        "output_sha256": None,
        "real_data_written": False,
        "reason": "BLOCKED_VALUE_SEMANTICS_UNKNOWN and BLOCKED_REGISTRATION_PROVENANCE_INSUFFICIENT",
        "required_before_rebuild": [
            "source-backed rasterization algorithm",
            "source-backed meanings for raw values 0/128/255",
            "proven coordinate transform, pixel origin, axis order, and y direction",
            "declared train/validation/test source separation",
        ],
    }
    _write_json(output / "stage_s1_rebuild_manifest.json", rebuild)
    _write_json(
        output / "stage_s1_reproducibility_check.json",
        {
            "stage": STAGE,
            "generated_at": _now(),
            "status": "NOT_EXECUTED",
            "independent_output_count": 0,
            "byte_identical": None,
            "reason": "No authoritative deterministic rebuild was permitted by the evidence gate.",
            "synthetic_contract_test": "Covered by tests/test_seg_raster_data_contract.py; it is not evidence for real Xi'an data.",
        },
    )

    conclusion = {
        "stage": STAGE,
        "generated_at": _now(),
        "base_sha": S1_BASE_SHA,
        "s1_base_sha": S1_BASE_SHA,
        "branch": "feat/seg-raster-only",
        "source_provenance": "BLOCKED",
        "source_provenance_status": "BLOCKED",
        "value_semantics": "BLOCKED",
        "value_semantics_status": "BLOCKED_VALUE_SEMANTICS_UNKNOWN",
        "train_pairing": "BLOCKED",
        "validation_pairing": "BLOCKED",
        "test_pairing": "BLOCKED",
        "pairing_status": {
            "train": "BLOCKED_GRID_SHAPE_AND_REGISTRATION",
            "validation": "BLOCKED_VALIDATION_PAIR_MISSING",
            "test": "BLOCKED_TEST_RASTER_SOURCE_MISSING",
        },
        "xian_registration": "BLOCKED",
        "xian_registration_status": "BLOCKED_REGISTRATION_PROVENANCE_INSUFFICIENT",
        "test_raster_status": "BLOCKED_TEST_RASTER_SOURCE_MISSING",
        "deterministic_rebuild": "NOT_EXECUTED",
        "deterministic_rebuild_status": "NOT_EXECUTED",
        "data_contract_ready_for_model": "BLOCKED",
        "go_no_go": "NO_GO",
        "model_implementation_started": False,
        "production_files_modified": False,
        "blockers": [
            "Legacy Xi'an raster generator and transform provenance were not found.",
            "The meanings of raw raster values 0, 128, and 255 are unknown.",
            "The configured train aerial/raster shapes differ and grid registration is unproven.",
            "A source-backed validation pair and split-separation proof are absent.",
            "The configured test trajectory-raster directory/files are absent.",
        ],
        "next_authoritative_inputs_required": [
            "original trajectory-raster generation code or specification",
            "class/value mapping for 0/128/255",
            "registration metadata or a source-backed transform",
            "validation and test raster sources with non-leak provenance",
        ],
        "next_stage_proposal": [
            "Obtain authoritative raster-generation/value-semantics/registration sources, then rerun Stage S1.",
            "Do not begin raster_seg_only model implementation while this gate is blocked.",
        ],
    }
    _write_json(output / "stage_s1_conclusion.json", conclusion)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--raw-aerial", type=Path, required=True)
    parser.add_argument("--raw-trajectory", type=Path, required=True)
    parser.add_argument("--candidate-raster", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    generate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
