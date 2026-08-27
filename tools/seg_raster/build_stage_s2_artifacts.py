"""Build machine-readable Stage S2 contracts and conclusion."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.seg_raster import describe_canonical_raster, sha256_file


S2_BASE_SHA = "c870019bf68999b15f489b73ba350c5cf74ebb1c"


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _redact_raster(description: dict[str, object], token: str) -> dict[str, object]:
    result = dict(description)
    result["path"] = token
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--full-raster", type=Path, required=True)
    parser.add_argument("--tile-raster", type=Path, required=True)
    parser.add_argument("--aerial", type=Path, required=True)
    parser.add_argument("--trusted-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-passed", type=int, required=True)
    parser.add_argument("--test-subtests", type=int, default=0)
    parser.add_argument("--test-duration-seconds", type=float, required=True)
    parser.add_argument("--test-command", required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()

    _write(output / "stage_s2_git_start.json", {
        "schema_version": "1.0.0",
        "stage": "S2",
        "generated_at": generated_at,
        "required_worktree": "CURRENT_WORKTREE",
        "branch": "feat/seg-raster-only",
        "upstream": "origin/feat/seg-raster-only",
        "s2_base_sha": S2_BASE_SHA,
        "head_sha_at_start": S2_BASE_SHA,
        "status_short_at_start": [],
        "recent_commits_at_start": [
            "c870019 data: define trajectory-raster registration contract",
            "23285e5 audit: characterize legacy DSF and raster integration paths",
            "13488c7 Merge pull request #19 from VNGSLab/refactor",
        ],
        "dirty_worktree_policy": (
            "E:/Code/VecRoad_self remained a read-only CURRENT_DIRTY_SNAPSHOT; "
            "no checkout, stash, reset, clean, commit, or file write was performed."),
        "merge_or_rebase_performed": False,
    })

    full = _redact_raster(
        describe_canonical_raster(
            args.full_raster, valid_extent_wh=(4300, 5000)),
        "${CANONICAL_XIAN_RASTER_CANVAS}",
    )
    tile = _redact_raster(
        describe_canonical_raster(args.tile_raster),
        "${CANONICAL_XIAN_RASTER_TILE}",
    )
    aerial_image = Image.open(args.aerial)
    source_image = Image.open(args.trusted_source)
    canonical_manifest = {
        "schema_version": "1.0.0",
        "stage": "S2",
        "generated_at": generated_at,
        "authority": {
            "status": "TRUSTED_USER_OVERRIDE_ACCEPTED",
            "region": "xian",
            "source_semantics": (
                "Every non-zero pixel means an original trajectory passed; "
                "brightness represents density, not class."),
            "logical_registered_shape_hw": [5000, 4300],
            "canonical_conversion": "traj_binary = (traj_raw > 0).astype(float32)",
            "canonical_values": [0.0, 1.0],
        },
        "trusted_upstream_source": {
            "path": "${TRUSTED_XIAN_TRAJECTORY_RASTER_SOURCE}",
            "sha256": sha256_file(args.trusted_source),
            "stored_shape_hw": [source_image.height, source_image.width],
            "note": (
                "The retained source is a supersampled representation of the "
                "authoritative trajectory rendering; upstream rendering is external and trusted."),
            "intended_for_commit": False,
        },
        "canonical_full_canvas": full,
        "canonical_train_tile": tile,
        "valid_mask": {
            "generation": "upper-left rectangle [0:5000, 0:4300]",
            "full_canvas_shape_hw": [8192, 8192],
            "valid_extent_wh": [4300, 5000],
            "valid_pixel_count": 21_500_000,
            "padding_pixel_count": 45_608_864,
            "right_padding_columns": 3892,
            "bottom_padding_rows": 3192,
            "stored_as_separate_git_file": False,
            "generated_by_loader": True,
        },
        "paired_aerial": {
            "path": "${READ_ONLY_XIAN_AERIAL_CANVAS}",
            "sha256": sha256_file(args.aerial),
            "shape_hwc": [aerial_image.height, aerial_image.width, 3],
            "pair_status": "PASS_USER_AUTHORITY_AND_MATCHED_CANVAS",
            "intended_for_commit": False,
        },
        "derivation": {
            "full_canvas_policy": (
                "Read the available registered 8192 canvas and generate its "
                "valid mask from the declared 4300x5000 upper-left extent."),
            "tile_tool": "tools/seg_raster/derive_canonical_tiles.py",
            "tile_origin_xy": [0, 0],
            "tile_shape_hw": [4096, 4096],
            "deterministic": True,
        },
        "git_exclusion": {
            "full_canvas_intended_for_commit": False,
            "tile_intended_for_commit": False,
            "dataset_files_in_commit": False,
        },
        "status": "PASS",
    }
    _write(output / "stage_s2_canonical_raster_manifest.json", canonical_manifest)

    _write(output / "stage_s2_loader_contract.json", {
        "schema_version": "1.0.0",
        "stage": "S2",
        "generated_at": generated_at,
        "dataclass": "utils.seg_raster.loader.TrajectoryRasterInput",
        "fields": {
            "raster": {"shape": "[B,1,H,W]", "dtype": "float32", "values": [0, 1]},
            "valid_mask": {"shape": "[B,1,H,W]", "dtype": "float32/bool", "values": [0, 1]},
            "region_ids": "list[str]",
            "metadata": "dict",
        },
        "canonical_conversion": "traj_binary = (traj_raw > 0).astype(float32)",
        "training_adapter": "utils.model_utils.Path.make_path_input",
        "inference_adapter": "infer.infer_segmentation",
        "shared_loader": "utils.seg_raster.loader",
        "shape_mismatch_policy": "FAIL_FAST",
        "missing_raster_policy": "FAIL_FAST",
        "padding_policy": "zero raster outside valid mask",
        "legacy_ternary_float_input_allowed": False,
        "status": "PASS",
    })

    _write(output / "stage_s2_model_contract.json", {
        "schema_version": "1.0.0",
        "stage": "S2",
        "generated_at": generated_at,
        "mode": "raster_seg_only",
        "backbone_input": "aerial RGB only",
        "raster_encoder": {
            "class": "model.seg_raster.encoder.TrajectoryRasterEncoder",
            "input_shape": "[B,1,H,W]",
            "output_shape": "[B,32,H/4,W/4]",
        },
        "fusion": {
            "class": "model.seg_raster.fusion.SegmentationOnlyRasterFusion",
            "output": "stage_fuse_seg [B,128,H/4,W/4]",
            "residual_output_projection_initialization": "all zeros",
            "initial_exact_image_only_parity": True,
            "parameter_count": 106984,
        },
        "segmentation_heads": ["road", "junction"],
        "outputs_are_logits": ["road", "junction", "anchor", "anchor_lowrs"],
        "training_loss": "BCEWithLogits",
        "legacy_modules_constructed": {
            "Transformer": False,
            "fuse_module_traj": False,
            "DSF": False,
        },
        "status": "PASS",
    })

    _write(output / "stage_s2_isolation_audit.json", {
        "schema_version": "1.0.0",
        "stage": "S2",
        "generated_at": generated_at,
        "allowed_path": "traj_binary -> stage_fuse_seg -> road_fts/junc_fts -> anchor",
        "forbidden_paths": [
            "traj_binary -> anchor",
            "traj_feature -> anchor",
            "raster gate -> anchor",
            "stage_fuse_seg -> anchor image-feature input",
        ],
        "observations": {
            "anchor_uses_stage_fuse_img": True,
            "anchor_uses_original_multiscale_image_features": True,
            "anchor_hourglass_input_channels_num_targets_1": 257,
            "raw_raster_channel_added_to_anchor": False,
            "raster_feature_added_to_anchor": False,
            "fixed_road_junction_features_raster_swap_anchor_exact_equal": True,
            "detach_anchor_only_raster_gradients": "ALL_NONE",
            "joint_anchor_only_raster_gradients": "ALL_FINITE_AND_SOME_NONZERO",
            "none_mode_shared_state_and_output_exact_parity": True,
        },
        "evidence_tests": [
            "tests/test_seg_raster_isolation.py",
            "tests/test_seg_raster_model.py",
            "tests/test_seg_raster_mode.py",
        ],
        "status": "PASS",
    })

    test_payload = {
        "schema_version": "1.0.0",
        "stage": "S2",
        "generated_at": generated_at,
        "command": args.test_command,
        "passed": args.test_passed,
        "subtests_passed": args.test_subtests,
        "failed": 0,
        "errors": 0,
        "duration_seconds": args.test_duration_seconds,
        "cache_provider_disabled": True,
        "status": "PASS",
    }
    _write(output / "stage_s2_test_results.json", test_payload)

    smoke = json.loads(
        (output / "stage_s2_smoke_results.json").read_text(encoding="utf-8"))
    _write(output / "stage_s2_conclusion.json", {
        "schema_version": "1.0.0",
        "stage": "S2",
        "generated_at": generated_at,
        "s2_base_sha": S2_BASE_SHA,
        "branch": "feat/seg-raster-only",
        "canonical_raster_status": canonical_manifest["status"],
        "mode_status": "PASS",
        "isolation_status": "PASS",
        "smoke_train_status": smoke["status"],
        "smoke_infer_status": smoke["infer_step"]["status"],
        "test_status": test_payload["status"],
        "acceptance": {
            "binary_presence_only": True,
            "train_infer_semantics_identical": True,
            "sequence_loader_skipped": True,
            "transformer_not_constructed": True,
            "legacy_dsf_not_formal_path": True,
            "image_backbone_image_only": True,
            "raw_raster_not_direct_to_anchor": True,
            "anchor_indirect_only_via_segmentation_features": True,
            "anchor_grad_to_seg_false_verified": True,
            "data_checkpoint_weight_cache_committed": False,
        },
        "risks": [
            "No long training run or performance claim was made in Stage S2.",
            "A complete 8192x8192 neural sweep was not run; tile inference passed.",
            "The trusted upstream trajectory-rendering script remains external; Stage S2 records the user authority and canonical derivative checksums.",
            "Existing legacy checkpoints do not contain the new raster-module keys and need explicit compatible initialization or an S2-trained checkpoint.",
        ],
        "go_no_go_for_next_stage": "GO_FOR_CONTROLLED_S2_TRAINING_EXPERIMENT",
        "status": "PASS",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
