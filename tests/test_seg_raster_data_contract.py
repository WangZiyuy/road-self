from __future__ import annotations

import ast
import copy
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools.seg_raster.build_commit_manifest import assert_commit_path_allowed
from tools.seg_raster.build_pair_manifest import build_manifest
from tools.seg_raster.contract import (
    AXIS_ORDER_CONTRACT,
    PAIR_MANIFEST_SCHEMA_VERSION,
    inspect_png,
    validate_pair_manifest,
    verify_reproducible_files,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_PATHS = [
    "model/DSFNet.py",
    "model/model.py",
    "model/model2.py",
    "train.py",
    "infer.py",
    "utils/trajectory_mode.py",
    "utils/OSMDataset.py",
    "utils/model_utils.py",
    "utils/tileloader.py",
    "configs/default_self.yml",
    "data_self/gen_dataset.py",
    "tests/test_trajectory_mode.py",
]


def _write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, format="PNG", optimize=False)


@pytest.fixture
def stage_test_root() -> Path:
    root = REPO_ROOT / "artifacts"
    yield root
    for path in root.glob("*_stage_s1_pytest.png"):
        path.unlink()


def _valid_entry(root: Path, *, region: str = "synthetic", split: str = "train") -> dict:
    aerial_rel = f"{region}_{split}_aerial_stage_s1_pytest.png"
    raster_rel = f"{region}_{split}_raster_stage_s1_pytest.png"
    aerial_path = root / aerial_rel
    raster_path = root / raster_rel
    _write_png(aerial_path, np.zeros((4, 5, 3), dtype=np.uint8))
    raster_array = np.zeros((4, 5), dtype=np.uint8)
    raster_array[1, 1] = 128
    raster_array[2, 2] = 255
    _write_png(raster_path, raster_array)
    aerial = inspect_png(aerial_path)
    raster = inspect_png(raster_path)
    return {
        "region": region,
        "split": split,
        "aerial_path": aerial_rel,
        "raster_path": raster_rel,
        "source_identity": f"{region}-{split}-source",
        "aerial": aerial,
        "raster": raster,
        "aerial_sha256": aerial["sha256"],
        "raster_sha256": raster["sha256"],
        "aerial_shape_hwc": aerial["shape_hwc"],
        "raster_shape_hwc": raster["shape_hwc"],
        "aerial_dtype": "uint8",
        "raster_dtype": "uint8",
        "raster_unique_values": [0, 128, 255],
        "shape": {"aerial": aerial["shape_hwc"], "raster": raster["shape_hwc"]},
        "channels": {"aerial": 3, "raster": 1},
        "dtype": {"aerial": "uint8", "raster": "uint8"},
        "value_range": {"aerial": [0, 0], "raster": [0, 255]},
        "allowed_values": {"aerial": list(range(256)), "raster": [0, 128, 255]},
        "normalization": {"aerial": "float32(value)/255.0", "raster": "float32(value)/255.0"},
        "coordinate_reference": {"aerial": "synthetic-grid", "raster": "synthetic-grid"},
        "crs": "synthetic-grid",
        "geotransform": [0, 1, 0, 0, 0, 1],
        "pixel_size": [1, 1],
        "pixel_origin": "upper_left",
        "axis_order": copy.deepcopy(AXIS_ORDER_CONTRACT),
        "coordinate_axis_contract": "synthetic shared image grid",
        "y_direction": "down",
        "source_lineage": {"fixture": "generated in pytest tmp_path"},
        "checksum": {"aerial": aerial["sha256"], "raster": raster["sha256"]},
        "grid_registration_proven": True,
        "registration_evidence": ["Both arrays were created from the same synthetic grid fixture."],
        "shape_match": True,
        "status": "PASS",
        "value_semantics": {
            "status": "KNOWN_AND_SOURCE_BACKED",
            "semantic_assignments": {"0": "background", "128": "support", "255": "trajectory"},
        },
    }


def _manifest(entry: dict) -> dict:
    return {
        "schema_version": PAIR_MANIFEST_SCHEMA_VERSION,
        "manifest_kind": "trajectory_raster_pair_contract",
        "entries": [entry],
        "split_separation": {"status": "PROVEN_NO_LEAKAGE"},
    }


def _codes(manifest: dict, root: Path | None = None) -> set[str]:
    return {issue.code for issue in validate_pair_manifest(manifest, root)}


def test_schema_and_valid_synthetic_pair_pass(stage_test_root: Path) -> None:
    manifest = _manifest(_valid_entry(stage_test_root))
    assert manifest["schema_version"] == "1.0.0"
    assert validate_pair_manifest(manifest, stage_test_root) == []


def test_shape_mismatch_fails_fast(stage_test_root: Path) -> None:
    manifest = _manifest(_valid_entry(stage_test_root))
    manifest["entries"][0]["shape"]["raster"] = [3, 5, 1]
    assert "GRID_SHAPE_MISMATCH" in _codes(manifest)


def test_missing_pair_member_is_blocking(stage_test_root: Path) -> None:
    manifest = _manifest(_valid_entry(stage_test_root))
    manifest["entries"][0]["raster"] = {"exists": False}
    assert "MISSING_PAIR_MEMBER" in _codes(manifest)


def test_checksum_mismatch_is_blocking(stage_test_root: Path) -> None:
    manifest = _manifest(_valid_entry(stage_test_root))
    manifest["entries"][0]["checksum"]["raster"] = "0" * 64
    assert "CHECKSUM_MISMATCH" in _codes(manifest, stage_test_root)


def test_invalid_observed_raster_value_is_blocking(stage_test_root: Path) -> None:
    manifest = _manifest(_valid_entry(stage_test_root))
    manifest["entries"][0]["allowed_values"]["raster"] = [0, 255]
    assert "RASTER_VALUE_NOT_ALLOWED" in _codes(manifest)


def test_unknown_value_semantics_is_blocking(stage_test_root: Path) -> None:
    manifest = _manifest(_valid_entry(stage_test_root))
    manifest["entries"][0]["value_semantics"]["status"] = "UNKNOWN"
    assert "VALUE_SEMANTICS_UNKNOWN" in _codes(manifest)


def test_unknown_registration_evidence_is_blocking(stage_test_root: Path) -> None:
    manifest = _manifest(_valid_entry(stage_test_root))
    manifest["entries"][0]["grid_registration_proven"] = False
    manifest["entries"][0]["registration_evidence"] = []
    assert "GRID_REGISTRATION_UNPROVEN" in _codes(manifest)


def test_axis_contract_is_checked(stage_test_root: Path) -> None:
    manifest = _manifest(_valid_entry(stage_test_root))
    manifest["entries"][0]["axis_order"] = {"decoded_array": "width_height_channels"}
    assert "AXIS_ORDER_CONTRACT_INVALID" in _codes(manifest)


def test_deterministic_output_comparison_uses_bytes(stage_test_root: Path) -> None:
    first = stage_test_root / "repro_first_stage_s1_pytest.png"
    second = stage_test_root / "repro_second_stage_s1_pytest.png"
    third = stage_test_root / "repro_third_stage_s1_pytest.png"
    _write_png(first, np.full((4, 5), 128, dtype=np.uint8))
    _write_png(second, np.full((4, 5), 128, dtype=np.uint8))
    _write_png(third, np.full((4, 5), 255, dtype=np.uint8))
    assert verify_reproducible_files(first, second)["byte_identical"] is True
    assert verify_reproducible_files(first, third)["byte_identical"] is False


def test_train_test_source_overlap_is_blocking(stage_test_root: Path) -> None:
    train = _valid_entry(stage_test_root, region="a", split="train")
    test = _valid_entry(stage_test_root, region="b", split="test")
    test["source_identity"] = train["source_identity"]
    manifest = _manifest(train)
    manifest["entries"].append(test)
    assert "SPLIT_SOURCE_LEAKAGE" in _codes(manifest)


@pytest.mark.parametrize(
    "path",
    [
        "data_self/input/traj/xian_0_0.png",
        "checkpoints/model.pth",
        "cache/result.json",
        "model/model.py",
    ],
)
def test_real_data_weights_cache_and_production_are_excluded_from_commit(path: str) -> None:
    with pytest.raises(ValueError):
        assert_commit_path_allowed(path)


def test_s1_tools_do_not_import_trajectory_sequence_loader() -> None:
    modules = set()
    for path in (REPO_ROOT / "tools/seg_raster").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)
            elif isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
    assert "utils.OSMDataset" not in modules
    assert not any("trajectory_mode" in module for module in modules)


def test_s1_tools_do_not_import_or_construct_dsf_or_torch() -> None:
    stage_s1_tool_names = {
        "__init__.py",
        "audit_stage_s1.py",
        "build_commit_manifest.py",
        "build_pair_manifest.py",
        "contract.py",
        "inspect_raster_provenance.py",
        "validate_pair_manifest.py",
    }
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (REPO_ROOT / "tools/seg_raster").glob("*.py")
        if path.name in stage_s1_tool_names
    )
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    assert not any("DSF" in module for module in imported)
    assert "torch" not in imported


def test_pair_manifest_builder_records_required_fields(stage_test_root: Path) -> None:
    manifest = build_manifest(stage_test_root)
    required = {
        "region", "split", "aerial_path", "raster_path", "shape", "channels",
        "dtype", "value_range", "allowed_values", "normalization",
        "coordinate_reference", "pixel_origin", "axis_order", "y_direction",
        "source_lineage", "checksum", "grid_registration_proven",
        "aerial_sha256", "raster_sha256", "aerial_shape_hwc",
        "raster_shape_hwc", "aerial_dtype", "raster_dtype",
        "raster_unique_values", "crs", "geotransform", "pixel_size",
        "coordinate_axis_contract", "shape_match", "status",
    }
    assert len(manifest["entries"]) == 4
    assert all(required.issubset(entry) for entry in manifest["entries"])
    json.dumps(manifest, allow_nan=False)


def test_s1_historical_commit_changed_no_production_files() -> None:
    s1_commit = "c870019bf68999b15f489b73ba350c5cf74ebb1c"
    result = subprocess.run(
        ["git", "diff", s1_commit + "^", s1_commit, "--", *PRODUCTION_PATHS],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert result.stdout == ""
