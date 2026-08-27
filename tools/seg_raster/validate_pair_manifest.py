"""Validate a Stage S1 pair manifest and emit a machine-readable result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.seg_raster.contract import validate_pair_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-blocked",
        action="store_true",
        help="Return zero while preserving validation failures in the output.",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    issues = validate_pair_manifest(manifest, args.source_root)
    result = {
        "stage": "seg_raster_stage_s1",
        "manifest_path": args.manifest.as_posix(),
        "status": "PASS" if not issues else "BLOCKED",
        "issue_count": len(issues),
        "issues": [issue.to_dict() for issue in issues],
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if not issues or args.allow_blocked else 1


if __name__ == "__main__":
    raise SystemExit(main())
