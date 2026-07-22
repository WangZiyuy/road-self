"""Run the real two-batch road_self image-only training smoke workflow."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.checkpoint_utils import (  # noqa: E402
    load_checkpoint_payload,
    resolve_inference_checkpoint_path,
    resolve_training_checkpoint_paths,
)
from utils.training_utils import resolve_path_iterations  # noqa: E402
from utils.trajectory_mode import (  # noqa: E402
    TRAJ_MODE_NONE,
    resolve_trajectory_mode,
    validate_trajectory_model_compatibility,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run two real road_self image-only training batches."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "baseline_image_only_smoke.yml",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--gpu-id",
        default=None,
        help="Optional CUDA device id passed to the derived smoke config.",
    )
    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=None,
        help="Optional training graph directory for environments with a different data layout.",
    )
    parser.add_argument(
        "--region-path",
        type=Path,
        default=None,
        help="Optional all-region file for environments with a different data layout.",
    )
    parser.add_argument(
        "--tile-dir",
        type=Path,
        default=None,
        help="Optional training tile directory for environments with a different data layout.",
    )
    parser.add_argument(
        "--validation-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--validation-input-size", type=int, default=64)
    parser.add_argument(
        "--skip-forward-validation",
        action="store_true",
        help="Only verify training and checkpoint metadata, not reload forward.",
    )
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as config_file:
        cfg = yaml.load(config_file, Loader=yaml.UnsafeLoader)
    if not isinstance(cfg, dict):
        raise ValueError("smoke config must contain a mapping")
    return cfg


def _validate_smoke_config(cfg: dict[str, Any]) -> None:
    mode = resolve_trajectory_mode(cfg)
    validate_trajectory_model_compatibility(cfg, mode)
    if mode != TRAJ_MODE_NONE:
        raise ValueError("the stage-0 smoke config must use TRAJ.MODE=none")
    if int(cfg["TRAIN"]["TOTAL_ITERATION"]) != 1:
        raise ValueError("smoke training requires TRAIN.TOTAL_ITERATION=1")
    if resolve_path_iterations(cfg) != 2:
        raise ValueError("smoke training requires TRAIN.PATH_ITERATIONS=2")
    if int(cfg["TRAIN"]["BATCH_SIZE"]) != 1:
        raise ValueError("smoke training requires TRAIN.BATCH_SIZE=1")


def _checkpoint_metadata(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    return {
        "path": os.fspath(path),
        "state_dict_keys": len(payload["state_dict"]),
        "outer_it": payload.get("outer_it"),
        "path_it": payload.get("path_it"),
        "trajectory_mode": payload.get("trajectory_mode"),
        "model_name": payload.get("model_name"),
        "num_targets": payload.get("num_targets"),
        "step_length": payload.get("step_length"),
        "window_size": payload.get("window_size"),
        "random_seed": payload.get("random_seed"),
    }


def main() -> int:
    args = _parse_args()
    config_path = args.config.resolve()
    cfg = _load_config(config_path)
    _validate_smoke_config(cfg)

    path_iterations = resolve_path_iterations(cfg)
    outer_it = int(cfg["TRAIN"]["TOTAL_ITERATION"])
    expected_paths = resolve_training_checkpoint_paths(
        cfg,
        outer_it=outer_it,
        path_it=path_iterations - 1,
    )
    latest_path = resolve_inference_checkpoint_path(cfg).resolve()
    expected_versioned = expected_paths.versioned.resolve()
    expected_latest = (
        expected_paths.latest.resolve() if expected_paths.latest is not None else None
    )
    if expected_latest != latest_path:
        raise AssertionError(
            "training latest and inference checkpoint differ: {} != {}".format(
                expected_latest, latest_path
            )
        )

    previous_mtime = latest_path.stat().st_mtime_ns if latest_path.is_file() else None
    started_ns = time.time_ns()
    trajectory_probe_path_exists = None
    inference_loader_output = None
    with tempfile.TemporaryDirectory(prefix="road_self_stage0_smoke_") as temp_dir:
        derived_cfg = dict(cfg)
        derived_cfg["DIR"] = dict(cfg["DIR"])
        derived_cfg["TRAIN"] = dict(cfg["TRAIN"])
        derived_cfg["TEST"] = dict(cfg["TEST"])
        if args.gpu_id is not None:
            derived_cfg["TRAIN"]["GPU_ID"] = str(args.gpu_id)
            derived_cfg["TEST"]["GPU_ID"] = str(args.gpu_id)
        for argument, config_key in (
            (args.graph_dir, "GRAPH_DIR"),
            (args.region_path, "ALL_REGION_PATH"),
            (args.tile_dir, "TILE_DIR"),
        ):
            if argument is not None:
                derived_cfg["DIR"][config_key] = os.fspath(argument)
        forbidden_trajectory_dir = Path(temp_dir) / "trajectory_must_not_be_read"
        derived_cfg["DIR"]["TRAJ_DIR"] = os.fspath(forbidden_trajectory_dir)
        derived_config_path = Path(temp_dir) / "smoke_config.yml"
        with derived_config_path.open("w", encoding="utf-8") as config_file:
            yaml.safe_dump(derived_cfg, config_file, sort_keys=False)

        run_env = os.environ.copy()
        if args.gpu_id is not None:
            run_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        subprocess.run(
            [args.python, "train.py", "--config", os.fspath(derived_config_path)],
            cwd=REPO_ROOT,
            check=True,
            env=run_env,
        )
        trajectory_probe_path_exists = forbidden_trajectory_dir.exists()
        if trajectory_probe_path_exists:
            raise AssertionError(
                "the forbidden trajectory probe path was unexpectedly created"
            )

        inference_loader_code = (
            "import sys; "
            "sys.argv=['infer.py', '--config', {!r}]; "
            "import infer; "
            "infer.prepare_net(); "
            "print('inference checkpoint load passed')"
        ).format(os.fspath(derived_config_path))
        inference_loader = subprocess.run(
            [args.python, "-c", inference_loader_code],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            env=run_env,
        )
        inference_loader_output = inference_loader.stdout.strip()

    if not expected_versioned.is_file():
        raise FileNotFoundError(expected_versioned)
    if not latest_path.is_file():
        raise FileNotFoundError(latest_path)
    latest_mtime = latest_path.stat().st_mtime_ns
    if latest_mtime < started_ns or (
        previous_mtime is not None and latest_mtime == previous_mtime
    ):
        raise RuntimeError("latest checkpoint was not refreshed by smoke training")

    payload = load_checkpoint_payload(latest_path, map_location="cpu")
    report: dict[str, Any] = {
        "config": os.fspath(config_path),
        "data_overrides": {
            "graph_dir": os.fspath(args.graph_dir) if args.graph_dir else None,
            "region_path": os.fspath(args.region_path) if args.region_path else None,
            "tile_dir": os.fspath(args.tile_dir) if args.tile_dir else None,
        },
        "trajectory_probe_path_exists": trajectory_probe_path_exists,
        "path_iterations": path_iterations,
        "elapsed_seconds": (time.time_ns() - started_ns) / 1e9,
        "inference_checkpoint_load": {
            "status": "passed",
            "output": inference_loader_output,
        },
        "checkpoint": _checkpoint_metadata(payload, latest_path),
        "versioned_checkpoint": os.fspath(expected_versioned),
    }

    if args.skip_forward_validation:
        report["reload_forward"] = {
            "status": "not_run",
            "reason": "--skip-forward-validation was supplied",
        }
    else:
        validation_command = [
            args.python,
            "scripts/validate_stage0_baseline.py",
            "--device",
            args.validation_device,
            "--input-size",
            str(args.validation_input_size),
            "--batch-size",
            "1",
            "--seed",
            str(cfg["TRAIN"].get("SEED", 20260722)),
            "--tolerance",
            "1e-6",
            "--checkpoint",
            os.fspath(latest_path),
        ]
        completed = subprocess.run(
            validation_command,
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            env=run_env,
        )
        report["reload_forward"] = {
            "status": "passed",
            "command": validation_command,
            "output": completed.stdout,
        }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
