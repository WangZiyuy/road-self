"""Real-data smoke test for the independent trajectory fragment encoder."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib import graph as graph_helper
from model.trajectory_encoder import TrajectoryFragmentEncoder
from utils.structured_trajectory_store import (
    open_structured_trajectory_store,
)
from utils.trajectory_batch import build_trajectory_batch
from utils.trajectory_fragments import SEGMENT_GRID_INDEX_BASIS


NODE_TYPES = ("ordinary", "t_junction", "multi_branch")


def _process_rss_bytes() -> Optional[int]:
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        return int(psutil.Process(os.getpid()).memory_info().rss)

    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        get_process_memory_info.restype = wintypes.BOOL
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.restype = wintypes.HANDLE
        success = get_process_memory_info(
            get_current_process(),
            ctypes.byref(counters),
            counters.cb,
        )
        return int(counters.WorkingSetSize) if success else None

    statm_path = Path("/proc/self/statm")
    if statm_path.exists():
        fields = statm_path.read_text(encoding="ascii").split()
        if len(fields) >= 2:
            return int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))
    return None


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _batch_tensor_bytes(batch: Dict[str, Any]) -> int:
    return sum(
        _tensor_bytes(value)
        for value in batch.values()
        if torch.is_tensor(value)
    )


def _node_type(degree: int) -> Optional[str]:
    if degree == 2:
        return "ordinary"
    if degree == 3:
        return "t_junction"
    if degree >= 4:
        return "multi_branch"
    return None


def _candidate_nodes(graph) -> Dict[str, List[Dict[str, Any]]]:
    candidates = {node_type: [] for node_type in NODE_TYPES}
    for vertex_id in sorted(graph.vertices):
        vertex = graph.vertices[vertex_id]
        degree = len(set(vertex.neighbors(graph)))
        node_type = _node_type(degree)
        if node_type is None:
            continue
        candidates[node_type].append(
            {
                "node_type": node_type,
                "vertex_id": int(vertex_id),
                "degree": int(degree),
                "center_xy": [
                    float(vertex.point.x),
                    float(vertex.point.y),
                ],
            }
        )
    return candidates


def _center_out_indices(length: int):
    center = length // 2
    yield center
    for offset in range(1, length):
        right = center + offset
        left = center - offset
        if right < length:
            yield right
        if left >= 0:
            yield left


def _select_real_cases(
    store,
    graph,
    window_size: float,
    context_points: int,
    max_time_gap_seconds: Optional[float],
    max_spatial_gap_pixels: Optional[float],
) -> Tuple[List[Dict[str, Any]], List[List]]:
    candidates = _candidate_nodes(graph)
    selected_cases = []
    fragment_lists = []
    for node_type in NODE_TYPES:
        nodes = candidates[node_type]
        if not nodes:
            raise RuntimeError(
                "GT graph has no {} nodes".format(node_type))
        selected = None
        for probe_count, node_index in enumerate(
            _center_out_indices(len(nodes)),
            start=1,
        ):
            node = nodes[node_index]
            query_start = time.perf_counter()
            fragments = store.query_trajectory_fragments(
                center_xy=node["center_xy"],
                window_size=window_size,
                context_points=context_points,
                max_time_gap_seconds=max_time_gap_seconds,
                max_spatial_gap_pixels=max_spatial_gap_pixels,
            )
            query_ms = (
                time.perf_counter() - query_start) * 1000.0
            if fragments:
                selected = dict(node)
                selected.update(
                    {
                        "selection_probe_count": probe_count,
                        "query_ms": query_ms,
                        "fragment_count": len(fragments),
                        "track_count": len(
                            {
                                fragment.track_index
                                for fragment in fragments
                            }
                        ),
                        "point_count": sum(
                            len(fragment) for fragment in fragments),
                    }
                )
                selected_cases.append(selected)
                fragment_lists.append(fragments)
                break
        if selected is None:
            raise RuntimeError(
                "no {} GT node returned trajectory fragments".format(
                    node_type))
    return selected_cases, fragment_lists


def _move_batch(
    trajectory_batch: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    return {
        key: (
            value.to(device=device, non_blocking=False)
            if torch.is_tensor(value)
            else value
        )
        for key, value in trajectory_batch.items()
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_budget(
    encoder: TrajectoryFragmentEncoder,
    fragment_lists,
    centers,
    window_size: float,
    budget: int,
    device: torch.device,
    warmup_iterations: int,
    repeat_iterations: int,
) -> Dict[str, Any]:
    batch_start = time.perf_counter()
    cpu_batch = build_trajectory_batch(
        fragment_lists,
        center_xy=centers,
        window_size=window_size,
        max_fragments=budget,
    )
    batch_build_ms = (
        time.perf_counter() - batch_start) * 1000.0
    device_batch = _move_batch(cpu_batch, device)
    input_tensor_bytes = _batch_tensor_bytes(device_batch)

    with torch.inference_mode():
        for _ in range(warmup_iterations):
            output = encoder(device_batch)
        _synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        rss_before = _process_rss_bytes()
        timings_ms = []
        for _ in range(repeat_iterations):
            _synchronize(device)
            forward_start = time.perf_counter()
            output = encoder(device_batch)
            _synchronize(device)
            timings_ms.append(
                (time.perf_counter() - forward_start) * 1000.0)
        rss_after = _process_rss_bytes()

    point_tokens = output["point_tokens"]
    fragment_tokens = output["fragment_tokens"]
    point_mask = output["point_mask"].to(dtype=torch.bool)
    fragment_mask = output["fragment_mask"].to(dtype=torch.bool)
    output_tensor_bytes = (
        _tensor_bytes(point_tokens) + _tensor_bytes(fragment_tokens))
    padding_point_zero = bool(
        torch.count_nonzero(point_tokens[~point_mask]).item() == 0)
    padding_fragment_zero = bool(
        torch.count_nonzero(
            fragment_tokens[~fragment_mask]).item() == 0)
    report = {
        "max_fragments": int(budget),
        "input_shape_b_n_t": list(
            device_batch["traj_xy_norm"].shape[:3]),
        "point_tokens_shape": list(point_tokens.shape),
        "fragment_tokens_shape": list(fragment_tokens.shape),
        "total_fragment_count_per_sample": (
            cpu_batch["total_fragment_count"].tolist()),
        "kept_fragment_count_per_sample": (
            cpu_batch["kept_fragment_count"].tolist()),
        "truncated_fragment_count_per_sample": (
            cpu_batch["truncated_fragment_count"].tolist()),
        "valid_point_count": int(point_mask.sum().item()),
        "valid_fragment_count": int(fragment_mask.sum().item()),
        "point_tokens_finite": bool(
            torch.isfinite(point_tokens).all().item()),
        "fragment_tokens_finite": bool(
            torch.isfinite(fragment_tokens).all().item()),
        "padding_point_tokens_zero": padding_point_zero,
        "padding_fragment_tokens_zero": padding_fragment_zero,
        "batch_build_ms": batch_build_ms,
        "forward_time_ms": {
            "repeats": int(repeat_iterations),
            "min": float(min(timings_ms)),
            "mean": float(statistics.mean(timings_ms)),
            "median": float(statistics.median(timings_ms)),
            "max": float(max(timings_ms)),
        },
        "input_tensor_bytes": input_tensor_bytes,
        "output_tensor_bytes": output_tensor_bytes,
        "process_rss_before_bytes": rss_before,
        "process_rss_after_bytes": rss_after,
        "process_rss_delta_bytes": (
            rss_after - rss_before
            if rss_before is not None and rss_after is not None
            else None
        ),
        "cuda_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
    }
    if not (
        report["point_tokens_finite"]
        and report["fragment_tokens_finite"]
        and padding_point_zero
        and padding_fragment_zero
    ):
        raise RuntimeError(
            "encoder smoke produced non-finite or non-zero padding output")
    return report


def run_smoke(
    cache_dir: Path,
    graph_path: Path,
    device_name: str,
    budgets: Sequence[int],
    window_size: float,
    context_points: int,
    max_time_gap_seconds: Optional[float],
    max_spatial_gap_pixels: Optional[float],
    hidden_dim: int,
    num_heads: int,
    num_layers: int,
    dropout: float,
    seed: int,
    warmup_iterations: int,
    repeat_iterations: int,
) -> Dict[str, Any]:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    store = open_structured_trajectory_store(os.fspath(cache_dir))
    grid_basis = store.meta.get("grid_index_basis")
    if grid_basis != SEGMENT_GRID_INDEX_BASIS:
        raise RuntimeError(
            "smoke requires segment-aware cache basis; found {!r}".format(
                grid_basis))
    graph = graph_helper.read_graph(
        os.fspath(graph_path), merge_duplicates=True)
    cases, fragment_lists = _select_real_cases(
        store=store,
        graph=graph,
        window_size=window_size,
        context_points=context_points,
        max_time_gap_seconds=max_time_gap_seconds,
        max_spatial_gap_pixels=max_spatial_gap_pixels,
    )
    centers = [case["center_xy"] for case in cases]
    encoder = TrajectoryFragmentEncoder(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device).eval()
    budget_reports = [
        _run_budget(
            encoder=encoder,
            fragment_lists=fragment_lists,
            centers=centers,
            window_size=window_size,
            budget=budget,
            device=device,
            warmup_iterations=warmup_iterations,
            repeat_iterations=repeat_iterations,
        )
        for budget in budgets
    ]
    return {
        "purpose": "smoke_only_not_formal_performance",
        "cache_dir": str(cache_dir.resolve()),
        "graph_path": str(graph_path.resolve()),
        "grid_index_basis": grid_basis,
        "device": str(device),
        "torch_version": torch.__version__,
        "seed": int(seed),
        "window_size": float(window_size),
        "context_points": int(context_points),
        "max_time_gap_seconds": max_time_gap_seconds,
        "max_spatial_gap_pixels": max_spatial_gap_pixels,
        "encoder": {
            "hidden_dim": int(hidden_dim),
            "num_heads": int(num_heads),
            "num_layers": int(num_layers),
            "dropout": float(dropout),
            "parameter_count": sum(
                parameter.numel()
                for parameter in encoder.parameters()),
        },
        "cases": cases,
        "budgets": budget_reports,
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke test the independent trajectory fragment encoder on "
            "real GT road nodes."
        )
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--budgets",
        type=_positive_int,
        nargs="+",
        default=[32, 64, 128],
    )
    parser.add_argument("--window-size", type=float, default=256.0)
    parser.add_argument("--context-points", type=_nonnegative_int, default=2)
    parser.add_argument("--max-time-gap-seconds", type=float, default=None)
    parser.add_argument("--max-spatial-gap-pixels", type=float, default=None)
    parser.add_argument("--hidden-dim", type=_positive_int, default=128)
    parser.add_argument("--num-heads", type=_positive_int, default=4)
    parser.add_argument("--num-layers", type=_positive_int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--warmup-iterations",
        type=_nonnegative_int,
        default=1,
    )
    parser.add_argument(
        "--repeat-iterations",
        type=_positive_int,
        default=3,
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.window_size <= 0.0 or not math.isfinite(args.window_size):
        raise ValueError("window_size must be finite and positive")
    budgets = sorted(set(args.budgets))
    report = run_smoke(
        cache_dir=args.cache_dir,
        graph_path=args.graph,
        device_name=args.device,
        budgets=budgets,
        window_size=args.window_size,
        context_points=args.context_points,
        max_time_gap_seconds=args.max_time_gap_seconds,
        max_spatial_gap_pixels=args.max_spatial_gap_pixels,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        seed=args.seed,
        warmup_iterations=args.warmup_iterations,
        repeat_iterations=args.repeat_iterations,
    )
    report_text = json.dumps(
        report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_text + "\n", encoding="utf-8")
    print(report_text)


if __name__ == "__main__":
    main()
