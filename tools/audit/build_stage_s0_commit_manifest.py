#!/usr/bin/env python3
"""Build the explicit Stage S0 audit commit candidate manifest."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


ARTIFACTS = [
    "stage_s0_activation_loss_contract.json",
    "stage_s0_activation_loss_contract_current_dirty.json",
    "stage_s0_anchor_step_audit.json",
    "stage_s0_anchor_step_audit_current_dirty.json",
    "stage_s0_checkpoint_audit.json",
    "stage_s0_conclusion.json",
    "stage_s0_execution_path_matrix.json",
    "stage_s0_execution_path_matrix_current_dirty.json",
    "stage_s0_git_start.json",
    "stage_s0_gradient_inventory.json",
    "stage_s0_gradient_inventory_current_dirty.json",
    "stage_s0_infer_contract_runtime.json",
    "stage_s0_infer_contract_runtime_current_dirty.json",
    "stage_s0_parameter_inventory.json",
    "stage_s0_parameter_inventory_current_dirty.json",
    "stage_s0_raster_alignment_audit.json",
    "stage_s0_redaction_audit.json",
    "stage_s0_repository_inventory.json",
    "stage_s0_runtime_audit.json",
    "stage_s0_runtime_audit_current_dirty.json",
    "stage_s0_source_provenance.json",
    "stage_s0_symbol_inventory.json",
    "stage_s0_test_results.json",
    "stage_s0_train_infer_contract.json",
    "stage_s0_train_infer_contract_current_dirty.json",
]

REPORTS = [
    "docs/audits/stage_s0_global_dsf_audit.md",
    "docs/audits/stage_s0_model_dataflow.md",
]

TESTS = ["tests/test_stage_s0_audit.py"]

TOOLS = [
    "tools/audit/README.md",
    "tools/audit/audit_dirty_snapshot_contracts.py",
    "tools/audit/audit_infer_contract_runtime.py",
    "tools/audit/audit_legacy_dsf_checkpoints.py",
    "tools/audit/audit_legacy_dsf_runtime.py",
    "tools/audit/audit_raster_alignment.py",
    "tools/audit/audit_source_provenance.py",
    "tools/audit/audit_stage_s0_static.py",
    "tools/audit/build_stage_s0_commit_manifest.py",
    "tools/audit/regroup_stage_s0_parameters.py",
    "tools/audit/run_stage_s0_tests.py",
    "tools/audit/sanitize_stage_s0_artifacts.py",
]


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, text=True, encoding="utf-8", errors="replace",
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def reason_for(path: str) -> str:
    if path.startswith("artifacts/"):
        if path.endswith("stage_s0_checkpoint_audit.json"):
            return "Stage S0 checkpoint audit evidence JSON; no checkpoint or model-weight bytes."
        return "Stage S0 machine-readable audit evidence."
    if path.startswith("docs/audits/"):
        return "Stage S0 human-readable audit report."
    if path.startswith("tests/"):
        return "Stage S0 non-production audit regression test."
    return "Stage S0 read-only audit tooling or tooling documentation."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    repo = args.repo.resolve()
    intended_paths = [f"artifacts/{name}" for name in ARTIFACTS] + REPORTS + TESTS + TOOLS
    missing = [path for path in intended_paths if not (repo / path).is_file()]
    if missing:
        raise FileNotFoundError("Missing intended Stage S0 files: " + ", ".join(missing))

    entries: list[dict[str, Any]] = []
    for relative in sorted(intended_paths):
        path = repo / relative
        entries.append({
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "generated_by_stage_s0": True,
            "intended_for_commit": True,
            "reason": reason_for(relative),
        })

    forbidden_suffixes = {".pth", ".pt", ".ckpt", ".tar", ".onnx", ".bin", ".npz"}
    checkpoint_or_weight_paths = [
        path for path in intended_paths
        if Path(path).suffix.lower() in forbidden_suffixes or path.lower().endswith(".pth.tar")
    ]
    dataset_paths = [path for path in intended_paths if path.startswith(("data/", "data_self/", "dataset/", "datasets/"))]
    cache_paths = [path for path in intended_paths if any(part.lower().endswith("cache") or part.lower() == "__pycache__" for part in Path(path).parts)]
    production_paths = [
        path for path in intended_paths
        if path.startswith(("model/", "configs/", "utils/")) or path in {"train.py", "infer.py"}
    ]
    if checkpoint_or_weight_paths or dataset_paths or cache_paths or production_paths:
        raise RuntimeError("Forbidden commit candidates detected")

    manifest = {
        "stage": "seg_raster_stage_s0",
        "generated_at": now_iso(),
        "base_sha": git_value(repo, "rev-parse", "HEAD"),
        "branch": git_value(repo, "branch", "--show-current"),
        "manifest_self_policy": {
            "path": "artifacts/stage_s0_commit_manifest.json",
            "listed_in_entries": False,
            "reason": "A file cannot contain a stable SHA-256 digest of itself; the manifest is explicitly added alongside its listed entries.",
        },
        "entries": entries,
        "summary": {
            "listed_entry_count": len(entries),
            "intended_for_commit_count": sum(1 for item in entries if item["intended_for_commit"]),
            "all_listed_files_generated_by_stage_s0": all(item["generated_by_stage_s0"] for item in entries),
            "checkpoint_or_model_weight_file_count": len(checkpoint_or_weight_paths),
            "dataset_file_count": len(dataset_paths),
            "cache_file_count": len(cache_paths),
            "production_file_count": len(production_paths),
        },
        "explicit_commit_exclusions": {
            "checkpoints": "CONFIRMED_NOT_LISTED",
            "datasets": "CONFIRMED_NOT_LISTED",
            "model_weights": "CONFIRMED_NOT_LISTED",
            "caches": "CONFIRMED_NOT_LISTED",
            "production_model_train_infer": "CONFIRMED_NOT_LISTED",
        },
    }
    output = repo / "artifacts/stage_s0_commit_manifest.json"
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({
        "manifest": "artifacts/stage_s0_commit_manifest.json",
        "listed_entry_count": len(entries),
        "excluded_binary_or_data_or_cache_count": 0,
    }, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
