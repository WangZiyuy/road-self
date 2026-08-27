"""Build the explicit small-file commit allowlist for final Stage S3 results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.seg_raster.stage_s3 import sha256_file


FORBIDDEN_SUFFIXES = (
    ".pth", ".pth.tar", ".ckpt", ".tif", ".tiff", ".png", ".jpg", ".jpeg")


def reason_for(path: str) -> str:
    if path.startswith("configs/"):
        return "frozen controlled experiment configuration"
    if path.startswith("tests/"):
        return "Stage S3 contract and regression coverage"
    if path.startswith("tools/"):
        return "preflight, scheduling, training, evaluation, or audit tool"
    if path.startswith("docs/"):
        return "Stage S3 audit report"
    if path.startswith("artifacts/"):
        return "small parseable Stage S3 evidence artifact"
    return "minimal production support for the frozen Stage S3 protocol"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-sha", required=True)
    args = parser.parse_args()
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"], cwd=REPO_ROOT,
        check=True, text=True, capture_output=True).stdout.splitlines()
    changed = subprocess.run(
        ["git", "diff", "--name-only", args.base_sha, "--"], cwd=REPO_ROOT,
        check=True, text=True, capture_output=True).stdout.splitlines()
    paths = sorted(
        {line[3:].replace("\\", "/") for line in status if len(line) >= 4}
        | {line.strip().replace("\\", "/") for line in changed if line.strip()})
    output_relative = args.output.resolve().relative_to(REPO_ROOT).as_posix()
    paths = [path for path in paths if path != output_relative]
    entries = []
    for relative in paths:
        path = REPO_ROOT / relative
        if not path.is_file():
            continue
        lower = relative.lower()
        if lower.endswith(FORBIDDEN_SUFFIXES) or "__pycache__" in lower or ".pytest_cache" in lower:
            raise RuntimeError("forbidden commit candidate: {}".format(relative))
        entries.append({
            "path": relative, "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "generated_by_stage_s3": True,
            "intended_for_commit": True, "reason": reason_for(relative),
        })
    payload = {
        "stage": "seg_raster_stage_s3", "entries": entries,
        "entry_count": len(entries),
        "explicit_exclusions": {
            "checkpoints": True,
            "datasets": True, "trajectory_rasters": True,
            "model_weights": True, "cache": True,
            "tensorboard_events": True, "large_logs": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
