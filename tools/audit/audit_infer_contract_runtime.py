#!/usr/bin/env python3
"""Exercise production test=True and anchor-inference failure contracts."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else "UNKNOWN"


def redact(value: str | Path, repo: Path, cwd: Path) -> str:
    path = Path(value).resolve()
    for root, label in (
        (repo, "${SOURCE_WORKTREE}"), (cwd, "${PROCESS_CWD}"),
        (Path(sys.prefix).resolve(), "${PYTHON_PREFIX}"),
        (Path.home().resolve(), "${USER_HOME}"),
    ):
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        return label if str(relative) == "." else label + "/" + relative.as_posix()
    return "${ABSOLUTE_PATH_REDACTED}/" + path.name


def redact_payload(value, repo: Path, cwd: Path):
    if isinstance(value, dict):
        return {key: redact_payload(item, repo, cwd) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_payload(item, repo, cwd) for item in value]
    if not isinstance(value, str):
        return value
    replacements = (
        (str(cwd), "${PROCESS_CWD}"),
        (str(repo), "${SOURCE_WORKTREE}"),
        (str(Path(sys.prefix).resolve()), "${PYTHON_PREFIX}"),
        (str(Path.home().resolve()), "${USER_HOME}"),
    )
    result = value
    for raw, label in replacements:
        result = result.replace(raw, label).replace(raw.replace("\\", "/"), label)
    return result


def capture(callable_):
    try:
        value = callable_()
        return {"status": "PASS", "value": value}
    except Exception as exc:
        return {
            "status": "FAIL", "error_type": type(exc).__name__,
            "error": str(exc), "traceback": traceback.format_exc(),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--snapshot-id", default="BASELINE_13488c7")
    parser.add_argument("--artifact-suffix", default="")
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    import numpy as np
    import torch
    from model import DSFNet as dsf_module
    from model import model as model_module
    from utils import utils as utils_module
    RPNet = model_module.RPNet
    MapContainer = utils_module.MapContainer
    from utils.trajectory_mode import validate_trajectory_model_compatibility

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    aerial = torch.randn(1, 3, 256, 256, device=device)
    raster = torch.rand(1, 1, 256, 256, device=device)
    walked = torch.zeros(1, 1, 64, 64, device=device)
    sequence = torch.tensor([[[[0.1, 0.2], [0.2, 0.3]]]], device=device)
    valid = torch.ones(1, 1, 2, dtype=torch.bool, device=device)
    cwd = Path.cwd().resolve()
    results = {
        "stage": "seg_raster_stage_s0", "generated_at": now_iso(), "device": str(device),
        "source_provenance": {
            "snapshot_id": args.snapshot_id,
            "cwd": redact(cwd, repo, cwd),
            "source_worktree": "${SOURCE_WORKTREE}",
            "branch": git_value(repo, "branch", "--show-current"),
            "HEAD": git_value(repo, "rev-parse", "HEAD"),
            "model.model.__file__": redact(model_module.__file__, repo, cwd),
            "model.DSFNet.__file__": redact(dsf_module.__file__, repo, cwd),
            "sys.path": [redact(item or cwd, repo, cwd) for item in sys.path],
            "imported_source_sha256": {
                "model/model.py": sha256_file(Path(model_module.__file__).resolve()),
                "model/DSFNet.py": sha256_file(Path(dsf_module.__file__).resolve()),
                "utils/utils.py": sha256_file(Path(utils_module.__file__).resolve()),
            },
            "monkeypatch_used": False,
            "stub_used": False,
            "audit_harness_used": True,
            "production_model_modified": False,
            "redaction": "Absolute paths are represented by logical labels.",
        },
    }

    origin = RPNet(4, backbone_pretrained=False, enable_trajectory_modules=False).to(device).eval()
    with torch.no_grad():
        origin_out = origin(
            aerial, None, None, None, None, None, NUM_TARGETS=4,
            test=True, model="origin", use_traj=False,
        )
    results["origin_test_true"] = {
        "status": "PASS", "road_shape": list(origin_out["road"].shape),
        "junction_shape": list(origin_out["junc"].shape),
    }
    output_dir = args.output_dir.resolve() if args.output_dir else repo / "artifacts"
    origin_container = MapContainer(str(output_dir), "origin_contract_probe", 256)
    results["origin_stitch"] = capture(lambda: (
        origin_container.add_batch_gpu([(0, 0)], origin_out["road"], 256),
        list(origin_container.map.shape),
    )[1])
    del origin
    torch.cuda.empty_cache() if device.type == "cuda" else None

    dsf = RPNet(4, backbone_pretrained=False, enable_trajectory_modules=True).to(device).eval()
    with torch.no_grad():
        dsf_out = dsf(
            aerial, raster, None, None, None, None, NUM_TARGETS=4,
            test=True, model="DSFNet", use_traj=True,
        )
    results["dsf_test_true"] = {
        "status": "PASS", "road_shape": list(dsf_out["road"].shape),
        "junction_shape": list(dsf_out["junc"].shape),
        "road_min": float(dsf_out["road"].min().item()),
        "road_max": float(dsf_out["road"].max().item()),
    }
    dsf_container = MapContainer(str(output_dir), "dsf_contract_probe", 256)
    results["dsf_stitch"] = capture(lambda: dsf_container.add_batch_gpu([(0, 0)], dsf_out["road"], 256))
    results["dsf_anchor_with_infer_arguments"] = capture(lambda: dsf(
        aerial_image=aerial, traj_image=None, aerial_traj_image=None,
        neighborhood_trajectory_norm=sequence, valid_mask=valid,
        walked_path=walked, NUM_TARGETS=4, test=False,
        model="DSFNet", use_traj=True,
    ))
    results["none_mode_dsf_compatibility"] = capture(
        lambda: validate_trajectory_model_compatibility(
            {"TRAJ": {"MODE": "none"}, "TRAIN": {"MODEL": "DSFNet"}}, "none"
        )
    )
    del dsf
    torch.cuda.empty_cache() if device.type == "cuda" else None
    results = redact_payload(results, repo, cwd)
    suffix = args.artifact_suffix
    write_json(output_dir / f"stage_s0_infer_contract_runtime{suffix}.json", results)

    if not suffix:
        contract_path = output_dir / "stage_s0_train_infer_contract.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        contract["runtime_checks"] = results
        write_json(contract_path, contract)
    print(json.dumps({
        key: value["status"] for key, value in results.items()
        if isinstance(value, dict) and "status" in value
    }, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
