"""Build the explicit Stage S2 commit candidate manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Iterable


S2_BASE_SHA = "c870019bf68999b15f489b73ba350c5cf74ebb1c"
FORBIDDEN_SUFFIXES = {
    ".ckpt", ".pth", ".pt", ".onnx", ".npy", ".npz",
    ".png", ".jpg", ".jpeg", ".tif", ".tiff",
}
FORBIDDEN_PARTS = {
    "__pycache__", ".pytest_cache", "cache", "checkpoints", "weights",
}
ALLOWED_EXACT = {
    "infer.py",
    "model/model.py",
    "train.py",
    "utils/OSMDataset.py",
    "utils/model_utils.py",
    "utils/trajectory_mode.py",
}
ALLOWED_PREFIXES = (
    "artifacts/stage_s2_",
    "configs/seg_raster_",
    "docs/audits/stage_s2_",
    "model/seg_raster/",
    "tests/test_seg_raster_",
    "tools/seg_raster/",
    "utils/seg_raster/",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize(path: str) -> str:
    return Path(path).as_posix()


def assert_commit_path_allowed(path: str) -> None:
    normalized = _normalize(path)
    candidate = Path(normalized)
    if candidate.suffix.lower() in FORBIDDEN_SUFFIXES:
        raise ValueError("dataset/image/model artifact is forbidden: {}".format(normalized))
    if any(part.lower() in FORBIDDEN_PARTS for part in candidate.parts):
        raise ValueError("cache/checkpoint/weight path is forbidden: {}".format(normalized))
    if normalized.startswith("data_self/"):
        raise ValueError("dataset path is forbidden: {}".format(normalized))
    if normalized not in ALLOWED_EXACT and not normalized.startswith(ALLOWED_PREFIXES):
        raise ValueError("path is outside the Stage S2 allow-list: {}".format(normalized))


def _preexisting_at_base(repo: Path, path: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", "{}:{}".format(S2_BASE_SHA, path)],
        cwd=repo,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def build_manifest(repo: Path, paths: Iterable[str]) -> dict[str, object]:
    entries = []
    for path in sorted({_normalize(value) for value in paths}):
        assert_commit_path_allowed(path)
        absolute = repo / path
        if not absolute.is_file():
            raise FileNotFoundError(path)
        preexisting = _preexisting_at_base(repo, path)
        entries.append({
            "path": path,
            "size_bytes": absolute.stat().st_size,
            "sha256": _sha256(absolute),
            "generated_by_stage_s2": not preexisting,
            "preexisting_file_modified_by_stage_s2": preexisting,
            "intended_for_commit": True,
            "reason": (
                "Required Stage S2 production/config integration."
                if path in ALLOWED_EXACT or path.startswith("configs/")
                else "Stage S2 model, loader, tool, test, report, or machine-readable evidence."
            ),
        })
    return {
        "schema_version": "1.0.0",
        "stage": "S2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "s2_base_sha": S2_BASE_SHA,
        "branch": "feat/seg-raster-only",
        "manifest_self_policy": {
            "path": "artifacts/stage_s2_commit_manifest.json",
            "listed_in_entries": False,
            "reason": "A file cannot contain a stable SHA-256 digest of itself; it is explicitly added with the listed paths.",
        },
        "entries": entries,
        "exclusion_assertions": {
            "checkpoints_in_commit": False,
            "datasets_in_commit": False,
            "trajectory_raster_pngs_in_commit": False,
            "model_weights_in_commit": False,
            "cache_in_commit": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--path", action="append", required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.repo.resolve(), args.path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
