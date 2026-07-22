"""Validate the stage-0 image-only VecRoad baseline.

The forward comparison deliberately uses one model instance in ``eval`` mode.
The legacy call supplies trajectory-shaped tensors while setting
``use_traj=False``; the stage-0 call supplies ``None`` for every trajectory
argument. This isolates the configuration/call-path refactor from model
initialization and checkpoint differences.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib import graph as graph_helper  # noqa: E402
from model.model import RPNet  # noqa: E402
from utils.trajectory_mode import (  # noqa: E402
    TRAJ_MODE_LEGACY,
    TRAJ_MODE_NONE,
    load_region_trajectory_inputs_for_mode,
    prepare_trajectory_sequence_batch,
    resolve_trajectory_mode,
    trajectory_enabled,
    trajectory_fetch_fields,
    validate_trajectory_model_compatibility,
)


OUTPUT_KEYS = ("road", "junc", "anchor", "anchor_lowrs")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the stage-0 image-only VecRoad baseline."
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--input-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional RPNet checkpoint. Synthetic weights are used when omitted.",
    )
    parser.add_argument(
        "--skip-forward",
        action="store_true",
        help="Run configuration and dependency-gating checks only.",
    )
    parser.add_argument(
        "--legacy-graph",
        type=Path,
        help="Optional closed-loop graph produced by the legacy image-only call path.",
    )
    parser.add_argument(
        "--stage0-graph",
        type=Path,
        help="Optional closed-loop graph produced with TRAJ.MODE=none.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optionally save the machine-readable report to this path.",
    )
    return parser.parse_args()


def _select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda was requested, but CUDA is unavailable")
    return torch.device(name)


def validate_config_resolution() -> dict[str, Any]:
    cases = {
        "explicit_none": (
            {"TRAJ": {"MODE": "none"}},
            TRAJ_MODE_NONE,
            False,
        ),
        "explicit_legacy_current": (
            {"TRAJ": {"MODE": "legacy_current"}},
            TRAJ_MODE_LEGACY,
            True,
        ),
        "legacy_use_traj_false": (
            {"TRAIN": {"USE_TRAJ": False}},
            TRAJ_MODE_NONE,
            False,
        ),
        "legacy_use_traj_true": (
            {"TRAIN": {"USE_TRAJ": True}},
            TRAJ_MODE_LEGACY,
            True,
        ),
    }
    results: dict[str, Any] = {}
    for name, (cfg, expected_mode, expected_enabled) in cases.items():
        actual_mode = resolve_trajectory_mode(cfg)
        actual_enabled = trajectory_enabled(cfg)
        passed = actual_mode == expected_mode and actual_enabled == expected_enabled
        results[name] = {
            "mode": actual_mode,
            "trajectory_enabled": actual_enabled,
            "passed": passed,
        }
        if not passed:
            raise AssertionError("trajectory mode case failed: {}".format(name))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        conflict_mode = resolve_trajectory_mode({
            "TRAJ": {"MODE": "none"},
            "TRAIN": {"USE_TRAJ": True},
        })
    if conflict_mode != TRAJ_MODE_NONE or len(caught) != 1:
        raise AssertionError("new/legacy conflict precedence or warning failed")
    results["new_mode_conflict_precedence"] = {
        "mode": conflict_mode,
        "warnings": [str(item.message) for item in caught],
        "passed": True,
    }

    try:
        resolve_trajectory_mode({"TRAJ": {"MODE": "invalid"}})
    except ValueError as exc:
        results["invalid_mode"] = {
            "raised": type(exc).__name__,
            "passed": True,
        }
    else:
        raise AssertionError("invalid TRAJ.MODE did not raise ValueError")
    validate_trajectory_model_compatibility({
        "TRAJ": {"MODE": "none"},
        "TRAIN": {"MODEL": "origin"},
    })
    try:
        validate_trajectory_model_compatibility({
            "TRAJ": {"MODE": "none"},
            "TRAIN": {"MODEL": "DSFNet"},
        })
    except ValueError as exc:
        results["image_only_model_guard"] = {
            "raised": type(exc).__name__,
            "passed": True,
        }
    else:
        raise AssertionError("none mode accepted a trajectory-dependent model")
    results["passed"] = True
    return results


def validate_no_trajectory_dependencies() -> dict[str, Any]:
    calls = {"loader": 0, "pad": 0, "normalize": 0}

    def forbidden(name: str):
        def _call(*_args: Any, **_kwargs: Any) -> Any:
            calls[name] += 1
            raise AssertionError("{} must not run in none mode".format(name))

        return _call

    region_inputs = load_region_trajectory_inputs_for_mode(
        TRAJ_MODE_NONE, "synthetic-region", {}, forbidden("loader")
    )
    sequence_inputs = prepare_trajectory_sequence_batch(
        TRAJ_MODE_NONE, None, forbidden("pad"), forbidden("normalize")
    )
    fetch_fields = trajectory_fetch_fields(TRAJ_MODE_NONE, include_raster=True)
    passed = (
        calls == {"loader": 0, "pad": 0, "normalize": 0}
        and region_inputs == (None, [], None, None)
        and sequence_inputs == (None, None)
        and fetch_fields == ()
    )
    if not passed:
        raise AssertionError("none mode touched a trajectory dependency")
    return {
        "calls": calls,
        "region_inputs": [None, [], None, None],
        "sequence_inputs": [None, None],
        "fetch_fields": list(fetch_fields),
        "passed": True,
    }


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a state_dict-compatible mapping")
    state_dict = payload.get("state_dict", payload)
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint state_dict is not a mapping")
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def validate_forward_equivalence(
    device: torch.device,
    input_size: int,
    batch_size: int,
    seed: int,
    tolerance: float,
    checkpoint: Path | None,
) -> dict[str, Any]:
    if input_size <= 0 or input_size % 32 != 0:
        raise ValueError("--input-size must be a positive multiple of 32")
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = RPNet(num_targets=4, backbone_pretrained=False).to(device)
    checkpoint_label = "synthetic_initialization"
    if checkpoint is not None:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        payload = torch.load(checkpoint, map_location=device)
        model.load_state_dict(_extract_state_dict(payload), strict=True)
        checkpoint_label = os.fspath(checkpoint)
    model.eval()

    aerial = torch.randn(batch_size, 3, input_size, input_size, device=device)
    walked = torch.randn(batch_size, 1, input_size, input_size, device=device)
    dummy_traj_image = torch.randn(
        batch_size, 1, input_size, input_size, device=device
    )
    dummy_aerial_traj = torch.randn(
        batch_size, 4, input_size, input_size, device=device
    )
    dummy_tracks = torch.randn(batch_size, 2, 3, 2, device=device)
    dummy_mask = torch.ones(batch_size, 2, 3, dtype=torch.bool, device=device)

    with torch.no_grad():
        legacy = model(
            aerial,
            dummy_traj_image,
            dummy_aerial_traj,
            dummy_tracks,
            dummy_mask,
            walked,
            NUM_TARGETS=4,
            test=False,
            model="origin",
            use_traj=False,
        )
        stage0 = model(
            aerial,
            None,
            None,
            None,
            None,
            walked,
            NUM_TARGETS=4,
            test=False,
            model="origin",
            use_traj=False,
        )

    outputs: dict[str, Any] = {}
    for key in OUTPUT_KEYS:
        old_tensor = legacy[key]
        new_tensor = stage0[key]
        difference = (old_tensor - new_tensor).abs()
        max_abs_diff = float(difference.max().cpu())
        mean_abs_diff = float(difference.mean().cpu())
        finite = bool(torch.isfinite(old_tensor).all() and torch.isfinite(new_tensor).all())
        shape_equal = old_tensor.shape == new_tensor.shape
        passed = finite and shape_equal and max_abs_diff <= tolerance
        outputs[key] = {
            "shape": list(new_tensor.shape),
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "finite": finite,
            "passed": passed,
        }
        if not passed:
            raise AssertionError(
                "{} failed numerical equivalence: max_abs_diff={}".format(
                    key, max_abs_diff
                )
            )

    del model, legacy, stage0
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "device": str(device),
        "seed": seed,
        "checkpoint": checkpoint_label,
        "tolerance": tolerance,
        "outputs": outputs,
        "passed": True,
    }


def _graph_signature(path: Path) -> dict[str, Any]:
    graph = graph_helper.read_graph(os.fspath(path), merge_duplicates=False)
    vertices = sorted(
        (int(vertex.id), float(vertex.point.x), float(vertex.point.y))
        for vertex in graph.vertices.values()
    )
    edges = sorted(
        (int(edge.id), int(edge.src_id), int(edge.dst_id))
        for edge in graph.edges.values()
    )
    return {"vertices": vertices, "edges": edges}


def validate_closed_loop_graphs(
    legacy_graph: Path | None, stage0_graph: Path | None
) -> dict[str, Any]:
    if legacy_graph is None and stage0_graph is None:
        return {
            "status": "not_run",
            "reason": "no legacy/stage0 graph pair was supplied",
        }
    if legacy_graph is None or stage0_graph is None:
        raise ValueError("--legacy-graph and --stage0-graph must be supplied together")
    if not legacy_graph.is_file() or not stage0_graph.is_file():
        raise FileNotFoundError("one or both closed-loop graph files do not exist")
    legacy = _graph_signature(legacy_graph)
    stage0 = _graph_signature(stage0_graph)
    vertices_equal = legacy["vertices"] == stage0["vertices"]
    edges_equal = legacy["edges"] == stage0["edges"]
    return {
        "status": "passed" if vertices_equal and edges_equal else "failed",
        "legacy_vertices": len(legacy["vertices"]),
        "stage0_vertices": len(stage0["vertices"]),
        "legacy_edges": len(legacy["edges"]),
        "stage0_edges": len(stage0["edges"]),
        "vertices_equal": vertices_equal,
        "edges_equal": edges_equal,
    }


def main() -> int:
    args = _parse_args()
    report: dict[str, Any] = {
        "config_resolution": validate_config_resolution(),
        "no_trajectory_dependencies": validate_no_trajectory_dependencies(),
    }
    if args.skip_forward:
        report["forward_equivalence"] = {
            "status": "not_run",
            "reason": "--skip-forward was supplied",
        }
    else:
        report["forward_equivalence"] = validate_forward_equivalence(
            device=_select_device(args.device),
            input_size=args.input_size,
            batch_size=args.batch_size,
            seed=args.seed,
            tolerance=args.tolerance,
            checkpoint=args.checkpoint,
        )
    report["closed_loop"] = validate_closed_loop_graphs(
        args.legacy_graph, args.stage0_graph
    )
    report["passed"] = all(
        section.get("passed", section.get("status") in {"passed", "not_run"})
        for section in report.values()
        if isinstance(section, dict)
    )

    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
