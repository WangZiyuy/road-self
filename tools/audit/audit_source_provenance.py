#!/usr/bin/env python3
"""Compare the audited baseline with the live dirty worktree, without writes there."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


BASE_SHA = "13488c768d147b37632ffeefe4b62cb3e94b36ec"
CRITICAL_FILES = (
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
)


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def run_text(args: list[str], cwd: Path) -> dict[str, Any]:
    proc = subprocess.run(
        args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", check=False,
    )
    return {
        "command": subprocess.list2cmdline(args),
        "cwd": "${BASELINE_WORKTREE}" if cwd.name.endswith("-seg-raster") else "${DIRTY_WORKTREE}",
        "return_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def git_blob(repo: Path, revision: str, path: str) -> tuple[bool, str | None, bytes | None, str]:
    blob_id = run_text(["git", "rev-parse", f"{revision}:{path}"], repo)
    if blob_id["return_code"] != 0:
        return False, None, None, blob_id["stderr"]
    proc = subprocess.run(
        ["git", "show", f"{revision}:{path}"], cwd=repo,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if proc.returncode != 0:
        return False, blob_id["stdout"].strip(), None, proc.stderr.decode("utf-8", errors="replace")
    return True, blob_id["stdout"].strip(), proc.stdout, ""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_line_endings(value: bytes) -> bytes:
    return value.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def artifact_usage(path: str) -> list[dict[str, str]]:
    common_static = [
        "artifacts/stage_s0_repository_inventory.json",
        "artifacts/stage_s0_symbol_inventory.json",
        "artifacts/stage_s0_execution_path_matrix.json",
        "artifacts/stage_s0_activation_loss_contract.json",
        "artifacts/stage_s0_train_infer_contract.json",
    ]
    runtime = [
        "artifacts/stage_s0_runtime_audit.json",
        "artifacts/stage_s0_anchor_step_audit.json",
        "artifacts/stage_s0_parameter_inventory.json",
        "artifacts/stage_s0_gradient_inventory.json",
        "artifacts/stage_s0_infer_contract_runtime.json",
    ] if path in {"model/model.py", "model/DSFNet.py", "utils/trajectory_mode.py"} else []
    tests = ["artifacts/stage_s0_test_results.json"] if path.startswith("tests/") or path == "utils/trajectory_mode.py" else []
    raster = ["artifacts/stage_s0_raster_alignment_audit.json"] if path in {
        "infer.py", "utils/OSMDataset.py", "utils/model_utils.py", "utils/tileloader.py",
        "configs/default_self.yml", "data_self/gen_dataset.py",
    } else []
    checkpoint = ["artifacts/stage_s0_checkpoint_audit.json"] if path in {
        "model/model.py", "model/DSFNet.py", "train.py", "infer.py",
    } else []
    result = []
    paired = {
        "artifacts/stage_s0_execution_path_matrix.json": "artifacts/stage_s0_execution_path_matrix_current_dirty.json",
        "artifacts/stage_s0_activation_loss_contract.json": "artifacts/stage_s0_activation_loss_contract_current_dirty.json",
        "artifacts/stage_s0_train_infer_contract.json": "artifacts/stage_s0_train_infer_contract_current_dirty.json",
        "artifacts/stage_s0_runtime_audit.json": "artifacts/stage_s0_runtime_audit_current_dirty.json",
        "artifacts/stage_s0_anchor_step_audit.json": "artifacts/stage_s0_anchor_step_audit_current_dirty.json",
        "artifacts/stage_s0_parameter_inventory.json": "artifacts/stage_s0_parameter_inventory_current_dirty.json",
        "artifacts/stage_s0_gradient_inventory.json": "artifacts/stage_s0_gradient_inventory_current_dirty.json",
        "artifacts/stage_s0_infer_contract_runtime.json": "artifacts/stage_s0_infer_contract_runtime_current_dirty.json",
    }
    for artifact in common_static + runtime + tests:
        item = {
            "artifact": artifact,
            "executed_snapshot": "BASELINE_13488c7",
            "current_dirty_applicability": "SET_AFTER_HASH_COMPARISON",
        }
        if artifact in paired:
            item["paired_dirty_artifact"] = paired[artifact]
        result.append(item)
    for artifact in raster + checkpoint:
        result.append({
            "artifact": artifact,
            "executed_snapshot": "BASELINE_13488c7_AUDIT_CODE_WITH_CURRENT_DIRTY_READ_ONLY_ASSETS",
            "current_dirty_applicability": "SOURCE_FILE_BYTES_IDENTICAL; ASSETS_READ_FROM_DIRTY_WORKTREE",
        })
    return result


def scrub_comparison_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: scrub_comparison_metadata(item)
            for key, item in value.items()
            if key not in {"generated_at", "source_provenance", "audit_scope", "runtime_checks"}
        }
    if isinstance(value, list):
        return [scrub_comparison_metadata(item) for item in value]
    return value


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        scrub_comparison_metadata(value), ensure_ascii=False,
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(encoded)


def dual_snapshot_evidence(artifact_dir: Path) -> dict[str, Any]:
    pairs = {
        "execution_path": (
            "stage_s0_execution_path_matrix.json",
            "stage_s0_execution_path_matrix_current_dirty.json"),
        "activation_loss": (
            "stage_s0_activation_loss_contract.json",
            "stage_s0_activation_loss_contract_current_dirty.json"),
        "train_infer_contract": (
            "stage_s0_train_infer_contract.json",
            "stage_s0_train_infer_contract_current_dirty.json"),
    }
    static = {}
    for name, (baseline_name, dirty_name) in pairs.items():
        baseline_path = artifact_dir / baseline_name
        dirty_path = artifact_dir / dirty_name
        present = baseline_path.is_file() and dirty_path.is_file()
        if present:
            baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
            dirty_payload = json.loads(dirty_path.read_text(encoding="utf-8"))
            baseline_hash = canonical_sha256(baseline_payload)
            dirty_hash = canonical_sha256(dirty_payload)
        else:
            baseline_hash = dirty_hash = None
        static[name] = {
            "baseline_artifact": "artifacts/" + baseline_name,
            "dirty_artifact": "artifacts/" + dirty_name,
            "both_present": present,
            "baseline_semantic_sha256": baseline_hash,
            "dirty_semantic_sha256": dirty_hash,
            "semantically_identical": bool(present and baseline_hash == dirty_hash),
        }

    runtime_paths = (
        artifact_dir / "stage_s0_runtime_audit.json",
        artifact_dir / "stage_s0_runtime_audit_current_dirty.json",
    )
    runtime_present = all(path.is_file() for path in runtime_paths)
    runtime_shapes = {"baseline": [], "current_dirty": []}
    if runtime_present:
        for label, path in zip(runtime_shapes, runtime_paths):
            payload = json.loads(path.read_text(encoding="utf-8"))
            for run in payload.get("runs", []):
                runtime_shapes[label].append({
                    "label": run.get("label"),
                    "status": run.get("status"),
                    "outputs": {
                        name: (stats or {}).get("shape") if isinstance(stats, dict) else None
                        for name, stats in run.get("outputs", {}).items()
                    },
                })

    infer_paths = (
        artifact_dir / "stage_s0_infer_contract_runtime.json",
        artifact_dir / "stage_s0_infer_contract_runtime_current_dirty.json",
    )
    infer_present = all(path.is_file() for path in infer_paths)
    infer_contracts = {}
    if infer_present:
        for label, path in zip(("baseline", "current_dirty"), infer_paths):
            payload = json.loads(path.read_text(encoding="utf-8"))
            infer_contracts[label] = {
                key: {
                    "status": value.get("status"),
                    "error_type": value.get("error_type"),
                    "road_shape": value.get("road_shape"),
                    "junction_shape": value.get("junction_shape"),
                }
                for key, value in payload.items()
                if isinstance(value, dict) and "status" in value
            }
    gradient_paths = (
        artifact_dir / "stage_s0_gradient_inventory.json",
        artifact_dir / "stage_s0_gradient_inventory_current_dirty.json",
    )
    gradient_present = all(path.is_file() for path in gradient_paths)
    if gradient_present:
        gradient_payloads = [
            json.loads(path.read_text(encoding="utf-8")) for path in gradient_paths
        ]
        gradient_hashes = [canonical_sha256(payload) for payload in gradient_payloads]
        gradient_structural = [{
            "summaries": payload.get("summaries"),
            "runs": [
                {"label": run.get("label"), "status": run.get("status"), "loss": run.get("loss")}
                for run in payload.get("runs", [])
            ],
        } for payload in gradient_payloads]
        gradient_structural_hashes = [canonical_sha256(payload) for payload in gradient_structural]
    else:
        gradient_hashes = [None, None]
        gradient_structural_hashes = [None, None]
    all_present = (
        all(item["both_present"] for item in static.values())
        and runtime_present and infer_present and gradient_present
    )
    return {
        "status": "PASS" if all_present else "BLOCKED",
        "static_contracts": static,
        "runtime_shapes": {
            "both_present": runtime_present,
            "baseline": runtime_shapes["baseline"],
            "current_dirty": runtime_shapes["current_dirty"],
            "identical": bool(runtime_present and runtime_shapes["baseline"] == runtime_shapes["current_dirty"]),
        },
        "inference_contracts": {
            "both_present": infer_present,
            "baseline": infer_contracts.get("baseline"),
            "current_dirty": infer_contracts.get("current_dirty"),
            "identical": bool(infer_present and infer_contracts.get("baseline") == infer_contracts.get("current_dirty")),
        },
        "gradient_artifacts": {
            "baseline": "artifacts/stage_s0_gradient_inventory.json",
            "current_dirty": "artifacts/stage_s0_gradient_inventory_current_dirty.json",
            "both_present": gradient_present,
            "baseline_semantic_sha256": gradient_hashes[0],
            "dirty_semantic_sha256": gradient_hashes[1],
            "full_per_parameter_numerics_identical": bool(
                gradient_present and gradient_hashes[0] == gradient_hashes[1]),
            "baseline_structural_sha256": gradient_structural_hashes[0],
            "dirty_structural_sha256": gradient_structural_hashes[1],
            "structurally_identical": bool(
                gradient_present
                and gradient_structural_hashes[0] == gradient_structural_hashes[1]),
            "interpretation": "Structural reachability/grad-null/zero counts and losses are compared separately from per-parameter floating-point gradient norms.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--dirty-repo", type=Path, required=True)
    args = parser.parse_args()
    baseline = args.baseline_repo.resolve()
    dirty = args.dirty_repo.resolve()

    records = []
    for rel_path in CRITICAL_FILES:
        baseline_exists, blob_sha, blob_bytes, blob_error = git_blob(baseline, BASE_SHA, rel_path)
        dirty_path = dirty / rel_path
        baseline_worktree_path = baseline / rel_path
        dirty_exists = dirty_path.is_file()
        baseline_worktree_exists = baseline_worktree_path.is_file()
        dirty_status = run_text(["git", "status", "--short", "--", rel_path], dirty)
        baseline_sha256 = sha256_bytes(blob_bytes) if blob_bytes is not None else None
        baseline_worktree_sha256 = (
            sha256_file(baseline_worktree_path) if baseline_worktree_exists else None
        )
        dirty_sha256 = sha256_file(dirty_path) if dirty_exists else None
        dirty_bytes = dirty_path.read_bytes() if dirty_exists else None
        normalized_identical = bool(
            blob_bytes is not None and dirty_bytes is not None
            and normalize_line_endings(blob_bytes) == normalize_line_endings(dirty_bytes)
        )
        blob_identical = bool(
            baseline_exists and dirty_exists and baseline_sha256 == dirty_sha256
        )
        worktree_identical = bool(
            baseline_worktree_exists and dirty_exists
            and baseline_worktree_sha256 == dirty_sha256
        )
        if blob_identical:
            difference_class = "IDENTICAL_TO_BASELINE_GIT_BLOB"
        elif normalized_identical:
            difference_class = "LINE_ENDING_BYTES_ONLY_RELATIVE_TO_BASELINE_GIT_BLOB"
        else:
            difference_class = "CONTENT_DIFFERENCE"
        usage = artifact_usage(rel_path)
        for item in usage:
            if blob_identical:
                item["current_dirty_applicability"] = "VALID_BY_IDENTICAL_BASELINE_GIT_BLOB_BYTES"
            elif "paired_dirty_artifact" in item:
                item["current_dirty_applicability"] = "NOT_REUSED; SEPARATE_DIRTY_ARTIFACT_CREATED"
            else:
                item["current_dirty_applicability"] = "BASELINE_ONLY; NOT_REUSED_FOR_DIRTY_SNAPSHOT"
        records.append({
            "path": rel_path,
            "baseline_exists": baseline_exists,
            "baseline_worktree_exists": baseline_worktree_exists,
            "dirty_snapshot_exists": dirty_exists,
            "baseline_git_blob_sha": blob_sha,
            "baseline_sha256": baseline_sha256,
            "baseline_worktree_sha256": baseline_worktree_sha256,
            "dirty_sha256": dirty_sha256,
            "dirty_git_status": dirty_status["stdout"].strip() or "CLEAN_FOR_PATH",
            "identical": blob_identical,
            "baseline_worktree_identical_to_dirty": worktree_identical,
            "identical_after_line_ending_normalization": normalized_identical,
            "difference_class": difference_class,
            "baseline_error": blob_error,
            "artifact_snapshot_usage": usage,
        })

    baseline_diff = run_text(["git", "diff", "--", *CRITICAL_FILES], baseline)
    dirty_diff = run_text(["git", "diff", "--", *CRITICAL_FILES], dirty)
    baseline_status = run_text(["git", "status", "--short", "--", *CRITICAL_FILES], baseline)
    dirty_status = run_text(["git", "status", "--short", "--", *CRITICAL_FILES], dirty)
    manifest_material = "\n".join(
        f"{item['path']}\0{item['dirty_sha256'] or 'MISSING'}" for item in records
    ).encode("utf-8")
    differences = [item["path"] for item in records if not item["identical"]]
    content_differences = [
        item["path"] for item in records
        if item["difference_class"] == "CONTENT_DIFFERENCE"
    ]
    physical_snapshot_differences = [
        item["path"] for item in records
        if not item["baseline_worktree_identical_to_dirty"]
    ]
    dual_evidence = dual_snapshot_evidence(baseline / "artifacts")
    if not differences:
        gate_status = "PASS"
    elif dual_evidence["status"] == "PASS":
        gate_status = "PASS_WITH_DUAL_SNAPSHOT"
    else:
        gate_status = "SPLIT_REQUIRED"
    payload = {
        "stage": "seg_raster_stage_s0",
        "generated_at": now_iso(),
        "gate": "source_provenance",
        "status": gate_status,
        "snapshots": {
            "BASELINE_13488c7": {
                "worktree": "${BASELINE_WORKTREE}",
                "branch": run_text(["git", "branch", "--show-current"], baseline)["stdout"].strip(),
                "head": run_text(["git", "rev-parse", "HEAD"], baseline)["stdout"].strip(),
                "revision": BASE_SHA,
            },
            "CURRENT_DIRTY_SNAPSHOT": {
                "worktree": "${DIRTY_WORKTREE}",
                "branch": run_text(["git", "branch", "--show-current"], dirty)["stdout"].strip(),
                "head": run_text(["git", "rev-parse", "HEAD"], dirty)["stdout"].strip(),
                "critical_file_manifest_sha256": sha256_bytes(manifest_material),
                "description": "Live worktree bytes read without modification; identity covers the 12 required critical files only.",
            },
        },
        "files": records,
        "summary": {
            "critical_file_count": len(records),
            "identical_count": sum(item["identical"] for item in records),
            "different_count": len(differences),
            "different_files": differences,
            "content_different_after_line_ending_normalization_count": len(content_differences),
            "content_different_after_line_ending_normalization_files": content_differences,
            "baseline_worktree_to_dirty_byte_different_count": len(physical_snapshot_differences),
            "baseline_worktree_to_dirty_byte_different_files": physical_snapshot_differences,
            "runtime_reuse_decision": (
                "VALID_BY_IDENTICAL_BASELINE_BLOB_BYTES"
                if not differences else (
                    "NOT_REUSED; BOTH_SNAPSHOTS_EXECUTED"
                    if dual_evidence["runtime_shapes"]["both_present"]
                    else "SEPARATE_DIRTY_RUNTIME_REQUIRED_BY_GATE"
                )
            ),
            "scope_warning": "This gate does not assert that every non-critical dirty file equals the baseline.",
        },
        "dual_snapshot_evidence": dual_evidence,
        "production_file_change_confirmation": {
            "baseline_worktree": {
                "status_command": baseline_status,
                "diff_command": baseline_diff,
                "unchanged": baseline_status["stdout"] == "" and baseline_diff["stdout"] == "",
            },
            "dirty_worktree": {
                "status_command": dirty_status,
                "diff_command": dirty_diff,
                "unchanged_relative_to_HEAD": dirty_status["stdout"] == "" and dirty_diff["stdout"] == "",
            },
        },
        "redaction": {
            "absolute_paths": "Replaced by ${BASELINE_WORKTREE} and ${DIRTY_WORKTREE} labels.",
            "remotes": "Not stored in this artifact.",
            "usernames_and_credentials": "Not stored.",
        },
    }
    output = baseline / "artifacts" / "stage_s0_source_provenance.json"
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"gate_status": gate_status, **payload["summary"]}, ensure_ascii=False, allow_nan=False))
    return 0 if gate_status.startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
