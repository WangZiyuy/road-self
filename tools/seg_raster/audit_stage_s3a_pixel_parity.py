"""Create small byte/decoded-pixel manifests without copying image data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.seg_raster.audit_stage_s3a_metrics import pixel_array_record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", required=True)
    parser.add_argument("--aerial-tile", type=Path, required=True)
    parser.add_argument("--aerial-full", type=Path, required=True)
    parser.add_argument("--raster-tile", type=Path, required=True)
    parser.add_argument("--raster-full", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "aerial_tile": args.aerial_tile,
        "aerial_full_canvas": args.aerial_full,
        "raster_tile": args.raster_tile,
        "raster_full_canvas": args.raster_full,
    }
    payload = {
        "stage": "seg_raster_stage_s3a",
        "status": "PASS",
        "execution_environment": args.environment,
        "files": {name: pixel_array_record(path) for name, path in paths.items()},
        "paths_redacted": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
