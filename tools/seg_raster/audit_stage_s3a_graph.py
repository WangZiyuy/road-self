"""Run and score one immutable S3 checkpoint under the S3A graph protocol."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import networkx as nx
import torch
import yaml

from utils.seg_raster.stage_s3 import EXPERIMENT_MATRIX, sha256_file


FORMAL_S3_SHA = "2e68f4e5a1c7cfad041182c2edce3194b8175b8c"
RECT = (2176.0, 0.0, 4096.0, 4096.0)
SNAP = 24.0
DENSIFY = 16.0
MAX_CONTROLS = 64


def verify_checkout(expected_sha: str) -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
        text=True, capture_output=True).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=no"], cwd=REPO_ROOT,
        check=True, text=True, capture_output=True).stdout.strip()
    if head != expected_sha or status:
        raise RuntimeError("graph audit requires the clean audit-code checkout")


def parse_graph(path: Path) -> tuple[list[tuple[float, float]], list[tuple[int, int]]]:
    vertices = []
    edges = []
    edge_section = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if not parts:
            edge_section = True
            continue
        if not edge_section:
            if len(parts) == 2:
                vertices.append((float(parts[0]), float(parts[1])))
            elif len(parts) == 3:
                vertices.append((float(parts[1]), float(parts[2])))
        elif len(parts) == 2:
            edges.append((int(parts[0]), int(parts[1])))
        elif len(parts) == 3:
            edges.append((int(parts[1]), int(parts[2])))
    return vertices, edges


def inside(point: tuple[float, float]) -> bool:
    return RECT[0] <= point[0] <= RECT[2] and RECT[1] <= point[1] <= RECT[3]


def raw_stats(vertices, edges) -> tuple[int, int]:
    directed = {}
    undirected = set()
    for left, right in edges:
        if left >= len(vertices) or right >= len(vertices):
            continue
        a, b = vertices[left], vertices[right]
        if not inside(a) or not inside(b) or a == b:
            continue
        directed[(a, b)] = directed.get((a, b), 0) + 1
        undirected.add(tuple(sorted((a, b))))
    return sum(max(0, value - 1) for value in directed.values()), len(undirected)


def build_graph(vertices, edges, *, densify: bool = False) -> nx.Graph:
    graph = nx.Graph()
    seen = set()
    for left, right in edges:
        if left >= len(vertices) or right >= len(vertices):
            continue
        start, end = vertices[left], vertices[right]
        if not inside(start) or not inside(end) or start == end:
            continue
        key = tuple(sorted((start, end)))
        if key in seen:
            continue
        seen.add(key)
        length = math.dist(start, end)
        if not densify or length <= DENSIFY:
            graph.add_edge(start, end, weight=length)
            continue
        count = max(1, int(math.ceil(length / DENSIFY)))
        previous = start
        for index in range(1, count + 1):
            ratio = index / count
            current = end if index == count else (
                round(start[0] + (end[0] - start[0]) * ratio, 6),
                round(start[1] + (end[1] - start[1]) * ratio, 6),
            )
            graph.add_edge(previous, current, weight=math.dist(previous, current))
            previous = current
    return graph


def nearest(point, nodes):
    if not nodes:
        return None, float("inf")
    best = min(nodes, key=lambda value: (math.dist(point, value), value))
    return best, math.dist(point, best)


def sample_controls(graph: nx.Graph) -> list:
    nodes = sorted(graph.nodes())
    if len(nodes) <= MAX_CONTROLS:
        return nodes
    critical = sorted(node for node in nodes if graph.degree(node) != 2)
    selected = critical[:MAX_CONTROLS]
    if len(selected) < MAX_CONTROLS:
        selected_set = set(selected)
        remaining = [node for node in nodes if node not in selected_set]
        needed = MAX_CONTROLS - len(selected)
        for index in range(needed):
            position = min(
                len(remaining) - 1,
                int((index + 0.5) * len(remaining) / needed))
            if remaining[position] not in selected:
                selected.append(remaining[position])
    return sorted(selected)[:MAX_CONTROLS]


def directional_apls(reference: nx.Graph, candidate: nx.Graph) -> dict:
    controls = sample_controls(reference)
    candidate_nodes = list(candidate.nodes())
    mapped = {}
    for node in controls:
        match, distance = nearest(node, candidate_nodes)
        mapped[node] = match if distance <= SNAP else None
    scores = []
    missing = 0
    for index, source in enumerate(controls):
        reference_lengths = nx.single_source_dijkstra_path_length(
            reference, source, weight="weight")
        mapped_source = mapped[source]
        candidate_lengths = (
            nx.single_source_dijkstra_path_length(
                candidate, mapped_source, weight="weight")
            if mapped_source is not None else {})
        for target in controls[index + 1:]:
            reference_length = reference_lengths.get(target)
            if reference_length is None or reference_length < 1.0:
                continue
            mapped_target = mapped[target]
            candidate_length = (
                candidate_lengths.get(mapped_target)
                if mapped_target is not None else None)
            if candidate_length is None:
                scores.append(0.0)
                missing += 1
            else:
                scores.append(max(
                    0.0,
                    1.0 - abs(candidate_length - reference_length) / reference_length))
    return {
        "score": float(sum(scores) / len(scores)) if scores else 0.0,
        "control_node_count": len(controls),
        "path_pair_count": len(scores),
        "missing_path_pair_count": missing,
    }


def critical_adjacency(graph: nx.Graph) -> tuple[set, set]:
    critical = {node for node in graph.nodes() if graph.degree(node) != 2}
    adjacency = set()
    for start in critical:
        for neighbor in graph.neighbors(start):
            previous, current = start, neighbor
            visited = {start}
            while current not in critical:
                if current in visited:
                    break
                visited.add(current)
                candidates = [node for node in graph.neighbors(current) if node != previous]
                if not candidates:
                    break
                previous, current = current, candidates[0]
            if current in critical and current != start:
                adjacency.add(tuple(sorted((start, current))))
    return critical, adjacency


def greedy_match(reference_nodes, candidate_nodes, tolerance: float) -> dict:
    candidates = []
    for left in reference_nodes:
        for right in candidate_nodes:
            distance = math.dist(left, right)
            if distance <= tolerance:
                candidates.append((distance, left, right))
    candidates.sort(key=lambda row: (row[0], row[1], row[2]))
    used_left, used_right, mapping = set(), set(), {}
    for _, left, right in candidates:
        if left in used_left or right in used_right:
            continue
        used_left.add(left)
        used_right.add(right)
        mapping[left] = right
    return mapping


def prf(true_positive: int, predicted: int, target: int) -> tuple[float, float, float]:
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / target if target else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def topology_metrics(reference: nx.Graph, candidate: nx.Graph) -> dict:
    ref_nodes, ref_edges = critical_adjacency(reference)
    cand_nodes, cand_edges = critical_adjacency(candidate)
    ref_to_cand = greedy_match(sorted(ref_nodes), sorted(cand_nodes), SNAP)
    cand_to_ref = {value: key for key, value in ref_to_cand.items()}
    recovered = sum(
        1 for left, right in ref_edges
        if left in ref_to_cand and right in ref_to_cand
        and tuple(sorted((ref_to_cand[left], ref_to_cand[right]))) in cand_edges)
    correct = sum(
        1 for left, right in cand_edges
        if left in cand_to_ref and right in cand_to_ref
        and tuple(sorted((cand_to_ref[left], cand_to_ref[right]))) in ref_edges)
    precision, recall, f1 = prf(correct, len(cand_edges), len(ref_edges))
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_reference_edges": recovered,
        "correct_candidate_edges": correct,
        "reference_edge_count": len(ref_edges),
        "candidate_edge_count": len(cand_edges),
    }


def junction_metrics(reference: nx.Graph, candidate: nx.Graph) -> dict:
    ref_nodes = sorted(node for node in reference.nodes() if reference.degree(node) >= 3)
    cand_nodes = sorted(node for node in candidate.nodes() if candidate.degree(node) >= 3)
    mapping = greedy_match(ref_nodes, cand_nodes, SNAP)
    precision, recall, f1 = prf(len(mapping), len(cand_nodes), len(ref_nodes))
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_count": len(mapping),
        "reference_count": len(ref_nodes),
        "candidate_count": len(cand_nodes),
        "tolerance_pixels": SNAP,
    }


def connectivity(graph: nx.Graph) -> dict:
    if graph.number_of_nodes() == 0:
        return {"component_count": 0, "largest_component_edge_length_ratio": 0.0}
    lengths = [
        sum(data["weight"] for _, _, data in graph.subgraph(component).edges(data=True))
        for component in nx.connected_components(graph)
    ]
    total = sum(lengths)
    return {
        "component_count": len(lengths),
        "largest_component_edge_length_ratio": max(lengths) / total if total else 0.0,
    }


def prepare_config(args, checkpoint_step: int) -> Path:
    with args.base_config.open("r", encoding="utf-8") as handle:
        base = yaml.load(handle, Loader=yaml.UnsafeLoader)
    spec = next(value for value in EXPERIMENT_MATRIX if value.key == args.run_key)
    config = deepcopy(base)
    checkpoint_dir = args.output_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_link = checkpoint_dir / "selected.pth.tar"
    if not checkpoint_link.exists():
        checkpoint_link.symlink_to(args.checkpoint)
    graph_traj_dir = args.output_dir / "graph_trajectory"
    graph_traj_dir.mkdir(parents=True, exist_ok=True)
    control = spec.raster_control or "aligned"
    source_control = args.source_runtime / "controls" / control
    graph_raster = graph_traj_dir / "xian.png"
    if not graph_raster.exists():
        graph_raster.symlink_to(source_control / "xian_0_0.png")
    config["TASK"] = "stage_s3a_{}_{}_graph".format(
        args.run_key, args.checkpoint_kind)
    config["TRAJ"] = {
        "MODE": spec.trajectory_mode,
        "SEQUENCE": {"ENABLED": False},
        "RASTER": {
            "CONTROL": spec.raster_control,
            "INPUT_SEMANTICS": "binary_presence",
            "USE_VALID_MASK": True,
            "VALID_EXTENT_WH": [4096, 4096],
            "ANCHOR_GRAD_TO_SEG": spec.anchor_grad_to_seg,
            "SHIFT_X": 128,
            "SHIFT_Y": 128,
        },
    }
    config["DIR"]["CHECK_POINT_DIR"] = str(checkpoint_dir)
    config["DIR"]["TEST_TRAJ_DIR"] = str(graph_traj_dir)
    config["DIR"]["TRAJ_DIR"] = str(source_control)
    config["DIR"]["SAVE_SEG_DIR"] = str(args.output_dir / "segmentation")
    config["DIR"]["INFER_STEP_DIR"] = str(args.output_dir / "infer_step")
    config["DIR"]["SAVE_GRAPH_DIR"] = str(args.output_dir / "graphs")
    config["DIR"]["SHORTCUT_DIR"] = str(args.output_dir / "shortcuts")
    config["TRAIN"].update({
        "NUM_TARGETS": 4,
        "IMG_SZ": 4096,
        "WINDOW_SIZE": 256,
        "BACKBONE_PRETRAINED": False,
        "DATA_PARALLEL": False,
        "MODEL": "origin",
    })
    config["TEST"].update({
        "CKPT": "selected",
        "CKPT_FILE": "selected.pth.tar",
        "GPU_ID": "0",
        "DATA_PARALLEL": False,
        "BATCH_SIZE_SEG": 32,
        "BATCH_SIZE_ANCHOR": 15,
        "TEST_IMG_SZ": 4096,
        "NUM_TILES": 1,
        "CROP_SZ": 256,
        "SAMPLE_STEP": 1,
        "SKIP_EMPTY_CROP": False,
        "WINDOW_SIZE": 256,
        "INFER_STEP": "start",
        "SINGLE_REGION": "xian",
        "START_FROM_ROAD_PEAK": False,
        "START_FROM_JUNC_PEAK": True,
        "NUM_TARGETS": 4,
        "STEP_LENGTH": 20,
        "RECT_RADIUS": 10,
        "FOLLOW_MODE": "follow_output",
        "AVG_CONFIDENCE_THRESHOLD": 0.2,
        "SAVE_EXAMPLES": False,
        "PRINT_ITERATION": 100,
        "BINARIZE_MAP": {
            "ROAD_SEG_THRESHOLE": 0.3,
            "JUNC_MAX_REGION_AREA": 200,
            "JUNC_SEG_THRESHOLE": 0.3,
            "ANCHOR_MAX_REGION_AREA": 1000,
            "MIN_BAD_ROAD_AREA": 200,
        },
    })
    path = args.output_dir / "resolved_graph_config.yml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-code-sha", required=True)
    parser.add_argument("--run-key", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-kind", choices=("best", "latest"), required=True)
    parser.add_argument("--source-runtime", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    args = parser.parse_args()
    verify_checkout(args.audit_code_sha)
    if not torch.cuda.is_available():
        raise RuntimeError("formal graph evaluation requires CUDA")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("code_sha") != FORMAL_S3_SHA:
        raise RuntimeError("checkpoint is not from the formal S3 code SHA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = prepare_config(args, int(payload["optimizer_step"]))
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    sys.argv = ["infer.py", "--config", str(config_path)]
    import infer
    iterations = []
    original = infer.infer_anchor
    def tracked(*positional, **keywords):
        result = original(*positional, **keywords)
        iterations.append(int(result[0]))
        return result
    infer.infer_anchor = tracked
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        infer.main()
    candidates = sorted((args.output_dir / "graphs").rglob("xian.graph"))
    if len(candidates) != 1:
        raise RuntimeError("expected exactly one output graph, found {}".format(len(candidates)))
    predicted_path = candidates[0]
    reference_path = REPO_ROOT / "data_self/input/graphs/xian.graph"
    reference_vertices, reference_edges = parse_graph(reference_path)
    predicted_vertices, predicted_edges = parse_graph(predicted_path)
    reference = build_graph(reference_vertices, reference_edges)
    predicted = build_graph(predicted_vertices, predicted_edges)
    reference_dense = build_graph(reference_vertices, reference_edges, densify=True)
    predicted_dense = build_graph(predicted_vertices, predicted_edges, densify=True)
    ref_to_pred = directional_apls(reference_dense, predicted_dense)
    pred_to_ref = directional_apls(predicted_dense, reference_dense)
    topo = topology_metrics(reference, predicted)
    junction = junction_metrics(reference, predicted)
    duplicates, undirected = raw_stats(predicted_vertices, predicted_edges)
    result = {
        "stage": "seg_raster_stage_s3a",
        "status": "PASS",
        "execution_environment": "REMOTE_TRAINING_SERVER",
        "audit_code_sha": args.audit_code_sha,
        "formal_s3_run_code_sha": FORMAL_S3_SHA,
        "run_key": args.run_key,
        "run_id": args.run_id,
        "checkpoint_kind": args.checkpoint_kind,
        "checkpoint_step": int(payload["optimizer_step"]),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "fixed_probability_threshold": 0.3,
        "inference_canvas_wh": [4096, 4096],
        "metric_extent_xyxy": list(RECT),
        "apls": float((ref_to_pred["score"] + pred_to_ref["score"]) / 2.0),
        "apls_directional": {"gt_to_pred": ref_to_pred, "pred_to_gt": pred_to_ref},
        "apls_protocol": {
            "kind": "deterministic_pixel_graph_approximation",
            "densify_step_pixels": DENSIFY,
            "snap_tolerance_pixels": SNAP,
            "max_control_nodes_per_direction": MAX_CONTROLS,
            "official_apls_jar_available": False,
        },
        "topo": topo["f1"],
        "topo_metrics": topo,
        "connectivity": connectivity(predicted),
        "junction_correctness": junction,
        "candidate_edge_count": topo["candidate_edge_count"],
        "undirected_edge_count": undirected,
        "dangling_edge_count": sum(
            1 for node in predicted.nodes() if predicted.degree(node) == 1),
        "duplicate_edge_count": duplicates,
        "vertex_count": predicted.number_of_nodes(),
        "graph_iterations": sum(iterations),
        "inference_time_seconds": float(time.monotonic() - started),
        "peak_gpu_memory_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
        "peak_gpu_memory_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
        "physical_gpu_index": args.physical_gpu,
        "gpu_name": torch.cuda.get_device_name(0),
        "started_at_utc": started_utc,
        "graph_logical_path": "${S3A_REMOTE_OUTPUT}/graph/" + args.run_key + "/" + args.checkpoint_kind + "/xian.graph",
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "status": "PASS", "run_key": args.run_key,
        "checkpoint_kind": args.checkpoint_kind,
        "apls": result["apls"],
    }, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
