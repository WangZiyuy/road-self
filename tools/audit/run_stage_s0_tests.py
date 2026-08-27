#!/usr/bin/env python3
"""Run required tests and write an evidence-preserving JSON result."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def execute(args: list[str], repo: Path, timeout: int) -> dict[str, Any]:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        proc = subprocess.run(
            args, cwd=repo, env=env, text=True, encoding="utf-8", errors="replace",
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=timeout,
        )
        return {
            "command": subprocess.list2cmdline(args), "cwd": str(repo),
            "return_code": proc.returncode, "stdout": proc.stdout,
            "stderr": proc.stderr, "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": subprocess.list2cmdline(args), "cwd": str(repo),
            "return_code": None, "stdout": (exc.stdout or "") if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "") if isinstance(exc.stderr, str) else "",
            "timed_out": True, "timeout_seconds": timeout,
        }


def coverage_matrix(repo: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((repo / "tests").glob("test_*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        tests = re.findall(r"^\s*(?:async\s+)?def\s+(test_[A-Za-z0-9_]+)|^\s*def\s+(test_[A-Za-z0-9_]+)", text, re.M)
        names = [first or second for first, second in tests]
        if not names:
            names = ["<module-level unittest cases>"]
        low = text.lower()
        for name in names:
            rows.append({
                "test": f"{path.relative_to(repo).as_posix()}::{name}",
                "covers_origin": "origin" in low or "image_only" in low,
                "covers_DSF": "dsfnet" in low or "unet_multistage" in low,
                "covers_raster_loading": "traj_image" in low or "get_traj" in low,
                "covers_sequence_loading": "valid_trajectories" in low or "trajectory_mode" in low,
                "covers_train": "train" in low,
                "covers_segmentation_inference": "infer_segmentation" in low,
                "covers_anchor_inference": "infer_anchor" in low,
                "covers_checkpoint": "checkpoint" in low or "state_dict" in low,
                "covers_shape": "shape" in low,
                "covers_gradient": "grad" in low or "backward" in low,
                "classification_note": "Static token coverage only; runtime commands determine pass/fail.",
            })
    return rows


def parse_summary(text: str) -> dict[str, int]:
    result = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    for key in result:
        match = re.search(rf"(\d+)\s+{key}", text)
        if match:
            result[key] = int(match.group(1))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    repo = args.repo.resolve()
    commands = [
        [sys.executable, "-m", "pytest", "tests/test_trajectory_mode.py", "-q"],
        [sys.executable, "-m", "pytest", "tests", "-q"],
    ]
    results = []
    for command in commands:
        print(f"running {subprocess.list2cmdline(command)}", flush=True)
        result = execute(command, repo, args.timeout)
        result["summary"] = parse_summary(result["stdout"] + "\n" + result["stderr"])
        result["status"] = "BLOCKED" if result["timed_out"] else "PASS" if result["return_code"] == 0 else "FAIL"
        results.append(result)
    payload = {
        "stage": "seg_raster_stage_s0", "generated_at": now_iso(),
        "python": sys.executable, "commands": results,
        "summary": {
            "required_commands": len(results),
            "passed_commands": sum(item["status"] == "PASS" for item in results),
            "failed_commands": sum(item["status"] == "FAIL" for item in results),
            "blocked_commands": sum(item["status"] == "BLOCKED" for item in results),
            "full_suite_passed_tests": results[-1]["summary"]["passed"],
            "full_suite_failed_tests": results[-1]["summary"]["failed"],
        },
        "coverage_matrix": coverage_matrix(repo),
        "audit_runtime_tests": {
            "script": "tools/audit/audit_legacy_dsf_runtime.py",
            "note": "Synthetic forward/backward, shape, anchor-step, and per-parameter gradient results are stored in dedicated runtime artifacts.",
        },
    }
    write_json(repo / "artifacts" / "stage_s0_test_results.json", payload)
    print(json.dumps({"results": [{"status": item["status"], "summary": item["summary"]} for item in results]}, allow_nan=False))
    return 0 if all(item["status"] == "PASS" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
