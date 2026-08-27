#!/usr/bin/env python3
"""Redact local paths, remotes, usernames, and URL credentials from S0 JSON."""

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


WINDOWS_ABSOLUTE = re.compile(r"(?<![A-Za-z0-9_$])([A-Za-z]:[\\/][^\"'\r\n]+)")
URL = re.compile(r"(?:(?:https?|ssh|git)://|git@)[^\s\"']+")


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def remote_urls(repo: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "remote", "get-url", "--all", "origin"], cwd=repo,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        encoding="utf-8", errors="replace", check=False,
    )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def replace_windows_absolute(match: re.Match[str]) -> str:
    raw = match.group(1).rstrip()
    name = re.split(r"[\\/]", raw)[-1]
    return "${ABSOLUTE_PATH_REDACTED}/" + (name or "PATH")


def sanitize_string(value: str, replacements: list[tuple[str, str]]) -> tuple[str, int]:
    result = value
    count = 0
    for raw, label in replacements:
        for variant in {raw, raw.replace("\\", "/")}:
            hits = result.count(variant)
            if hits:
                result = result.replace(variant, label)
                count += hits
    result, url_count = URL.subn("${REMOTE_URL_REDACTED}", result)
    count += url_count
    result, absolute_count = WINDOWS_ABSOLUTE.subn(replace_windows_absolute, result)
    count += absolute_count
    return result, count


def sanitize(value: Any, replacements: list[tuple[str, str]]) -> tuple[Any, int]:
    if isinstance(value, dict):
        output = {}
        total = 0
        for key, item in value.items():
            clean, count = sanitize(item, replacements)
            output[key] = clean
            total += count
        return output, total
    if isinstance(value, list):
        output = []
        total = 0
        for item in value:
            clean, count = sanitize(item, replacements)
            output.append(clean)
            total += count
        return output, total
    if isinstance(value, str):
        return sanitize_string(value, replacements)
    return value, 0


def remaining_sensitive_strings(value: Any, prefix: str = "$") -> list[dict[str, str]]:
    findings = []
    if isinstance(value, dict):
        for key, item in value.items():
            findings.extend(remaining_sensitive_strings(item, prefix + "." + key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            findings.extend(remaining_sensitive_strings(item, f"{prefix}[{index}]"))
    elif isinstance(value, str) and (WINDOWS_ABSOLUTE.search(value) or URL.search(value)):
        findings.append({"json_path": prefix, "snippet": value[:200]})
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--dirty-repo", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    dirty = args.dirty_repo.resolve()
    artifacts = repo / "artifacts"
    replacements = [
        (str(repo), "${BASELINE_WORKTREE}"),
        (str(dirty), "${DIRTY_WORKTREE}"),
        (str(Path.home().resolve()), "${USER_HOME}"),
        (str(Path(sys.prefix).resolve()), "${PYTHON_PREFIX}"),
    ]
    for remote in remote_urls(repo):
        replacements.append((remote, "${ORIGIN_REMOTE_REDACTED}"))
    replacements.sort(key=lambda item: len(item[0]), reverse=True)

    file_results = []
    all_remaining = []
    for path in sorted(artifacts.glob("stage_s0_*.json")):
        if path.name == "stage_s0_redaction_audit.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        clean, replacement_count = sanitize(payload, replacements)
        remaining = remaining_sensitive_strings(clean)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(clean, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        file_results.append({
            "artifact": "artifacts/" + path.name,
            "replacement_count": replacement_count,
            "remaining_sensitive_string_count": len(remaining),
        })
        all_remaining.extend({"artifact": "artifacts/" + path.name, **item} for item in remaining)

    report = {
        "stage": "seg_raster_stage_s0",
        "generated_at": now_iso(),
        "status": "PASS" if not all_remaining else "FAIL",
        "redaction_target_count": len(file_results),
        "total_stage_s0_json_count_including_redaction_report": len(file_results) + 1,
        "count_semantics": {
            "redaction_target_count": "All stage_s0_*.json files except this redaction report.",
            "total_stage_s0_json_count_including_redaction_report": (
                "redaction_target_count plus stage_s0_redaction_audit.json itself."
            ),
        },
        "policy": {
            "baseline_worktree": "${BASELINE_WORKTREE}",
            "dirty_worktree": "${DIRTY_WORKTREE}",
            "user_home": "${USER_HOME}",
            "python_prefix": "${PYTHON_PREFIX}",
            "origin_remote": "${ORIGIN_REMOTE_REDACTED}",
            "unknown_absolute_path": "${ABSOLUTE_PATH_REDACTED}",
        },
        "files": file_results,
        "remaining_findings": all_remaining,
        "credentials_stored": False,
    }
    output = artifacts / "stage_s0_redaction_audit.json"
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({
        "status": report["status"], "files": len(file_results),
        "replacements": sum(item["replacement_count"] for item in file_results),
        "remaining": len(all_remaining),
    }, allow_nan=False))
    return 0 if not all_remaining else 2


if __name__ == "__main__":
    raise SystemExit(main())
