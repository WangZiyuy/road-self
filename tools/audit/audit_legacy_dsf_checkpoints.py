#!/usr/bin/env python3
"""Inventory checkpoints and compare the legacy epoch-40/50 model tensors."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any


EXTENSIONS = (".pth", ".pth.tar", ".pt", ".ckpt")


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def finite(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else "UNKNOWN"


def labelled_path(path: Path, roots: list[tuple[str, Path]]) -> str:
    resolved = path.resolve()
    for label, root in roots:
        try:
            relative = resolved.relative_to(root.resolve())
        except ValueError:
            continue
        return label + "/" + relative.as_posix()
    return "${ABSOLUTE_PATH_REDACTED}/" + resolved.name


def group_for(key: str) -> str:
    if key.startswith("DSF."):
        return "DSF"
    if key.startswith(("stage_1.", "stage_2.", "stage_3.", "stage_4.", "stage_5.")):
        return "Res2Net"
    if key.startswith("transformer."):
        return "Transformer"
    if key.startswith("fuse_module_traj."):
        return "fuse_module_traj"
    if key.startswith(("up4_anchor.", "up3_anchor.", "up2_anchor.", "up1_anchor.", "up0_anchor.", "trans4_anchor.", "trans3_anchor.", "trans2_anchor.", "trans1_anchor.")):
        return "DSF_anchor_decoder"
    if key.startswith(("decoders.", "next_step_final.", "conv_final.")):
        return "anchor_common_or_origin_decoder"
    if key.startswith(("road_seg.", "conv_road_final.", "junc_seg.", "conv_junc_final.", "conv_2_side.", "conv_3_side.", "conv_4_side.", "conv_5_side.", "conv_fuse.")):
        return "origin_segmentation"
    if key.startswith(("traj_to_img_fc.", "cross_attention.", "stage_1_traj.", "stage_1_traj_aerial.")) or key == "missing_traj_feature":
        return "trajectory_misc"
    if key.startswith("fuse_module."):
        return "origin_fuse_module"
    return "other"


def find_checkpoints(
        roots: list[tuple[str, Path]]) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    seen: set[Path] = set()
    result = []
    candidates: dict[str, Path] = {}
    for label, root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or not any(path.name.lower().endswith(ext) for ext in EXTENSIONS):
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            stat = path.stat()
            is_candidate = path.name in {"40.2047.pth.tar", "50.2047.pth.tar"}
            result.append({
                "path": labelled_path(resolved, roots), "size_bytes": stat.st_size,
                "mtime": dt.datetime.fromtimestamp(stat.st_mtime, dt.timezone.utc).astimezone().isoformat(),
                "legacy_epoch40_or_50_candidate": is_candidate,
            })
            if is_candidate:
                candidates[path.name] = resolved
    return sorted(result, key=lambda item: item["path"]), candidates


def normalize_data_parallel_prefix(state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    keys = list(state)
    prefixed = sum(key.startswith("module.") for key in keys)
    applied = bool(keys and prefixed == len(keys))
    normalized = (
        {key[len("module."):]: value for key, value in state.items()}
        if applied else dict(state)
    )
    return normalized, {
        "rule": "Strip exactly one leading 'module.' only when every state_dict key has that prefix; mixed-prefix dictionaries are left unchanged.",
        "original_key_count": len(keys),
        "module_prefixed_key_count": prefixed,
        "applied": applied,
        "normalized_key_count": len(normalized),
    }


def key_alignment(checkpoint_keys: set[str], expected_keys: set[str]) -> dict[str, Any]:
    checkpoint_dsf = {key for key in checkpoint_keys if key.startswith("DSF.")}
    expected_dsf = {key for key in expected_keys if key.startswith("DSF.")}
    return {
        "checkpoint_key_count": len(checkpoint_keys),
        "expected_key_count": len(expected_keys),
        "intersection_count": len(checkpoint_keys & expected_keys),
        "missing_expected_keys": sorted(expected_keys - checkpoint_keys),
        "extra_checkpoint_keys": sorted(checkpoint_keys - expected_keys),
        "DSF": {
            "expected_key_count": len(expected_dsf),
            "checkpoint_key_count": len(checkpoint_dsf),
            "intersection_count": len(checkpoint_dsf & expected_dsf),
            "missing_expected_keys": sorted(expected_dsf - checkpoint_dsf),
            "extra_checkpoint_keys": sorted(checkpoint_dsf - expected_dsf),
        },
    }


def compare_pair(
        torch: Any, path_a: Path, path_b: Path, expected_keys: set[str],
        roots: list[tuple[str, Path]]) -> dict[str, Any]:
    a = torch.load(path_a, map_location="cpu", weights_only=False, mmap=True)
    b = torch.load(path_b, map_location="cpu", weights_only=False, mmap=True)
    state_a, prefix_a = normalize_data_parallel_prefix(a.get("state_dict", {}))
    state_b, prefix_b = normalize_data_parallel_prefix(b.get("state_dict", {}))
    keys_a = set(state_a)
    keys_b = set(state_b)
    common = sorted(keys_a & keys_b)
    details = []
    summaries: dict[str, dict[str, Any]] = {}
    for key in common:
        left = state_a[key]
        right = state_b[key]
        group = group_for(key)
        summary = summaries.setdefault(group, {
            "tensor_count": 0, "element_count": 0, "changed_tensor_count": 0,
            "changed_element_count": 0, "max_abs_diff": 0.0,
            "sum_abs_diff": 0.0,
        })
        if not torch.is_tensor(left) or not torch.is_tensor(right) or tuple(left.shape) != tuple(right.shape):
            details.append({
                "key": key, "group": group, "comparable": False,
                "shape_a": list(left.shape) if torch.is_tensor(left) else None,
                "shape_b": list(right.shape) if torch.is_tensor(right) else None,
            })
            continue
        delta = (left.detach().to(torch.float32) - right.detach().to(torch.float32)).abs()
        count = left.numel()
        changed = int(torch.count_nonzero(delta).item())
        max_diff = finite(delta.max().item()) if count else 0.0
        mean_diff = finite(delta.mean().item()) if count else 0.0
        details.append({
            "key": key, "group": group, "comparable": True,
            "shape": list(left.shape), "numel": count,
            "exactly_equal": changed == 0, "changed_count": changed,
            "max_abs_diff": max_diff, "mean_abs_diff": mean_diff,
        })
        summary["tensor_count"] += 1
        summary["element_count"] += count
        summary["changed_tensor_count"] += int(changed > 0)
        summary["changed_element_count"] += changed
        summary["max_abs_diff"] = max(summary["max_abs_diff"], max_diff or 0.0)
        summary["sum_abs_diff"] += (mean_diff or 0.0) * count
    for summary in summaries.values():
        elements = summary["element_count"]
        summary["mean_abs_diff"] = finite(summary["sum_abs_diff"] / elements) if elements else 0.0
        del summary["sum_abs_diff"]
        summary["all_tensors_exactly_equal"] = summary["changed_tensor_count"] == 0
    return {
        "status": "PASS",
        "path_a": labelled_path(path_a, roots),
        "path_b": labelled_path(path_b, roots),
        "checkpoint_sha256_a": sha256_file(path_a),
        "checkpoint_sha256_b": sha256_file(path_b),
        "checkpoint_size_bytes_a": path_a.stat().st_size,
        "checkpoint_size_bytes_b": path_b.stat().st_size,
        "checkpoint_source_commit_provenance": "UNKNOWN",
        "metadata_a": {key: value for key, value in a.items() if key not in {"state_dict", "optimizer"} and isinstance(value, (str, int, float, bool, type(None)))},
        "metadata_b": {key: value for key, value in b.items() if key not in {"state_dict", "optimizer"} and isinstance(value, (str, int, float, bool, type(None)))},
        "top_level_keys_a": sorted(a.keys()), "top_level_keys_b": sorted(b.keys()),
        "state_key_count_a": len(keys_a), "state_key_count_b": len(keys_b),
        "data_parallel_prefix_normalization": {"a": prefix_a, "b": prefix_b},
        "expected_model_key_count": len(expected_keys),
        "key_alignment_a": key_alignment(keys_a, expected_keys),
        "key_alignment_b": key_alignment(keys_b, expected_keys),
        "only_a": sorted(keys_a - keys_b), "only_b": sorted(keys_b - keys_a),
        "group_summaries": summaries, "tensor_comparisons": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--source-repo", type=Path, default=Path(r"E:\Code\VecRoad_self"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    source = args.source_repo.resolve()
    import torch
    sys.path.insert(0, str(repo))
    from model import DSFNet as dsf_module
    from model import model as model_module

    expected_status = "PASS"
    expected_error = ""
    try:
        expected_model = model_module.RPNet(
            4, backbone_pretrained=False, enable_trajectory_modules=True)
        expected_keys = set(expected_model.state_dict())
        del expected_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        expected_status = "BLOCKED"
        expected_error = f"{type(exc).__name__}: {exc}"
        expected_keys = set()

    roots = [("${BASELINE_WORKTREE}", repo), ("${DIRTY_WORKTREE}", source)]
    checkpoints, candidates = find_checkpoints(roots)
    pair_result: dict[str, Any]
    if {"40.2047.pth.tar", "50.2047.pth.tar"}.issubset(candidates) and expected_status == "PASS":
        pair_result = compare_pair(
            torch, candidates["40.2047.pth.tar"], candidates["50.2047.pth.tar"],
            expected_keys, roots)
    else:
        pair_result = {
            "status": "BLOCKED" if expected_status == "BLOCKED" else "NOT_PRESENT",
            "reason": expected_error or "Both epoch40 and epoch50 checkpoints were not found.",
            "found_candidates": sorted(candidates),
        }
    payload = {
        "stage": "seg_raster_stage_s0", "generated_at": now_iso(),
        "source_provenance": {
            "snapshot_id": "BASELINE_13488c7_AUDIT_CODE_WITH_CURRENT_DIRTY_READ_ONLY_CHECKPOINTS",
            "cwd": "${BASELINE_WORKTREE}",
            "branch": git_value(repo, "branch", "--show-current"),
            "HEAD": git_value(repo, "rev-parse", "HEAD"),
            "model.model.__file__": "${BASELINE_WORKTREE}/model/model.py",
            "model.DSFNet.__file__": "${BASELINE_WORKTREE}/model/DSFNet.py",
            "imported_source_sha256": {
                "model/model.py": sha256_file(Path(model_module.__file__).resolve()),
                "model/DSFNet.py": sha256_file(Path(dsf_module.__file__).resolve()),
            },
            "expected_state_dict_status": expected_status,
            "expected_state_dict_error": expected_error,
            "expected_state_dict_key_count": len(expected_keys),
            "checkpoint_training_source_commit": "UNKNOWN",
            "monkeypatch_used": False,
            "stub_used": False,
            "audit_harness_used": True,
            "production_files_modified": False,
        },
        "search_roots": ["${BASELINE_WORKTREE}", "${DIRTY_WORKTREE}"],
        "search_patterns": ["*.pth", "*.pth.tar", "*.pt", "*.ckpt"],
        "checkpoint_count": len(checkpoints), "checkpoints": checkpoints,
        "legacy_pair_comparison": pair_result,
        "git_policy": "No checkpoint is copied, modified, or added to Git.",
        "redaction": "Checkpoint paths and worktree roots use logical labels; remote URLs, usernames, and credentials are not stored.",
    }
    write_json(repo / "artifacts" / "stage_s0_checkpoint_audit.json", payload)
    print(json.dumps({
        "checkpoint_count": len(checkpoints), "legacy_pair_status": pair_result["status"],
        "group_summaries": pair_result.get("group_summaries", {}),
    }, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
