"""Build an allow-listed Stage S1 commit manifest."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.seg_raster.contract import sha256_file


FORBIDDEN_SUFFIXES = {".ckpt", ".pth", ".pt", ".onnx", ".npy", ".npz", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
FORBIDDEN_PARTS = {"__pycache__", ".pytest_cache", "cache", "checkpoints", "weights"}
PRODUCTION_PATHS = {
    "model/DSFNet.py",
    "model/model.py",
    "model/model2.py",
    "train.py",
    "infer.py",
}


def _normalize(path: str) -> str:
    return Path(path).as_posix()


def assert_commit_path_allowed(path: str) -> None:
    normalized = _normalize(path)
    candidate = Path(normalized)
    if normalized in PRODUCTION_PATHS:
        raise ValueError(f"Production path is forbidden in Stage S1 commit: {normalized}")
    if candidate.suffix.lower() in FORBIDDEN_SUFFIXES:
        raise ValueError(f"Dataset/image/model artifact is forbidden: {normalized}")
    if any(part.lower() in FORBIDDEN_PARTS for part in candidate.parts):
        raise ValueError(f"Cache/checkpoint/weight path is forbidden: {normalized}")
    if normalized.startswith("data_self/input/"):
        raise ValueError(f"Real input data is forbidden: {normalized}")
    allowed_prefixes = ("tools/seg_raster/", "tests/", "docs/audits/", "artifacts/stage_s1_")
    if not normalized.startswith(allowed_prefixes):
        raise ValueError(f"Path is outside the Stage S1 allow-list: {normalized}")


def build_manifest(repo: Path, paths: Iterable[str]) -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = []
    for value in sorted({_normalize(path) for path in paths}):
        assert_commit_path_allowed(value)
        absolute = repo / Path(value)
        if not absolute.is_file():
            raise FileNotFoundError(value)
        entries.append(
            {
                "path": value,
                "size_bytes": absolute.stat().st_size,
                "sha256": sha256_file(absolute),
                "generated_by_stage_s1": True,
                "intended_for_commit": True,
                "reason": "Stage S1 audit tool, synthetic test, report, or machine-readable evidence.",
            }
        )
    return {
        "stage": "seg_raster_stage_s1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "s1_base_sha": "23285e5bc6515ca88a3121d2547aa9ab0476a7ad",
        "branch": "feat/seg-raster-only",
        "manifest_self_policy": {
            "path": "artifacts/stage_s1_commit_manifest.json",
            "listed_in_entries": False,
            "reason": "A file cannot contain a stable SHA-256 digest of itself; it is explicitly added with the listed paths.",
        },
        "entries": entries,
        "exclusion_assertions": {
            "checkpoints_in_commit": False,
            "datasets_in_commit": False,
            "model_weights_in_commit": False,
            "cache_in_commit": False,
            "production_model_train_infer_in_commit": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--path", action="append", required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.repo, args.path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
