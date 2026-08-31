"""Build the explicit Stage S3E code/results commit manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FORBIDDEN_PARTS = {
    "data_self", "checkpoints", "datasets", "cache", "__pycache__",
    ".pytest_cache", "tensorboard", "weights"}
FORBIDDEN_SUFFIXES = {
    ".pth", ".pt", ".ckpt", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
ALLOWED_PREFIXES = (
    "configs/stage_s3e_", "model/model.py",
    "model/seg_raster/zero_preserving_road_adapter.py",
    "utils/seg_raster/stage_s3e.py", "tools/seg_raster/", "tests/test_stage_s3e_",
    "artifacts/stage_s3e_", "docs/audits/stage_s3e_")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def allowed(raw: str) -> None:
    value = Path(raw).as_posix()
    path = Path(value)
    if not value.startswith(ALLOWED_PREFIXES):
        raise ValueError("path outside S3E allow-list: " + value)
    if any(part.lower() in FORBIDDEN_PARTS for part in path.parts):
        raise ValueError("runtime path forbidden: " + value)
    if value.lower().endswith(tuple(FORBIDDEN_SUFFIXES)):
        raise ValueError("binary artifact forbidden: " + value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--run-code-sha", required=True)
    parser.add_argument("--path", action="append", required=True)
    args = parser.parse_args()
    entries = []
    for raw in sorted(set(args.path)):
        value = Path(raw).as_posix()
        allowed(value)
        path = args.repo / value
        if not path.is_file():
            raise FileNotFoundError(value)
        entries.append({
            "path": value, "size_bytes": path.stat().st_size,
            "sha256": digest(path), "generated_by_stage_s3e": True,
            "intended_for_commit": True,
            "reason": "Stage S3E frozen code, test, audit, or small evidence.",
        })
    report = {
        "stage": "seg_raster_stage_s3e", "branch": "feat/seg-raster-only",
        "s3e_base_sha": args.base_sha, "s3e_run_code_sha": args.run_code_sha,
        "manifest_self_policy": {
            "path": "artifacts/stage_s3e_commit_manifest.json",
            "listed_in_entries": False,
            "reason": "A manifest cannot contain a stable digest of itself."},
        "entries": entries,
        "exclusion_assertions": {
            "checkpoints_in_commit": False, "datasets_in_commit": False,
            "raster_in_commit": False, "model_weights_in_commit": False,
            "cache_in_commit": False, "large_logs_in_commit": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
