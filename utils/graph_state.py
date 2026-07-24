"""Continuous VecRoad graph-exploration state features.

This module only converts an existing :class:`Path` and one queued search
visit into tensors.  It does not alter graph growth or search order.
"""

from typing import Dict, Tuple

import numpy as np
import torch

from utils.model_utils import SearchVertexState


def _unit_direction(dx: float, dy: float) -> Tuple[np.ndarray, bool]:
    vector = np.asarray([dx, dy], dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length <= 0.0:
        return np.zeros(2, dtype=np.float32), False
    return (vector / length).astype(np.float32), True


def _parent_vertex(path, search_state: SearchVertexState):
    parent_id = search_state.parent_vertex_id
    if parent_id is None:
        return None
    return path.graph.vertices.get(int(parent_id))


def _neighbor_records(path, search_state: SearchVertexState):
    graph = path.graph
    vertex = search_state.vertex
    graph_vertex = graph.vertices.get(int(vertex.id))
    if graph_vertex is None:
        return []

    neighbor_ids = set()
    for edge_id in graph_vertex.in_edges_id:
        edge = graph.edges.get(edge_id)
        if edge is not None and edge.src_id != graph_vertex.id:
            neighbor_ids.add(int(edge.src_id))
    for edge_id in graph_vertex.out_edges_id:
        edge = graph.edges.get(edge_id)
        if edge is not None and edge.dst_id != graph_vertex.id:
            neighbor_ids.add(int(edge.dst_id))

    parent_id = search_state.parent_vertex_id
    records = []
    for neighbor_id in neighbor_ids:
        neighbor = graph.vertices.get(neighbor_id)
        if neighbor is None:
            continue
        direction, valid = _unit_direction(
            float(neighbor.point.x) - float(vertex.point.x),
            float(neighbor.point.y) - float(vertex.point.y),
        )
        if not valid:
            continue
        is_incoming = (
            parent_id is not None and neighbor_id == int(parent_id))
        records.append((
            bool(is_incoming),
            float(neighbor.point.x),
            float(neighbor.point.y),
            int(neighbor_id),
            direction,
        ))

    # Always retain the incoming neighbor first.  Remaining neighbors are
    # ordered by geometry and ID, making truncation independent of edge
    # insertion order without quantizing directions into fixed buckets.
    records.sort(key=lambda item: (
        0 if item[0] else 1,
        item[1],
        item[2],
        item[3],
    ))
    return records


def build_graph_state(
        path,
        search_state: SearchVertexState,
        max_explored_edges: int = 8,
) -> Dict[str, torch.Tensor]:
    """Build continuous graph-state tensors for one queued exploration visit.

    ``incoming_dir`` points from the parent toward the current node.
    ``explored_edge_dirs`` point from the current node toward each unique
    neighbor already present in the generated graph.
    """

    if not isinstance(search_state, SearchVertexState):
        raise TypeError("search_state must be a SearchVertexState")
    if max_explored_edges < 0:
        raise ValueError("max_explored_edges must be non-negative")

    incoming_dir = np.zeros(2, dtype=np.float32)
    incoming_valid = False
    parent = _parent_vertex(path, search_state)
    if parent is not None:
        incoming_dir, incoming_valid = _unit_direction(
            float(search_state.vertex.point.x) - float(parent.point.x),
            float(search_state.vertex.point.y) - float(parent.point.y),
        )

    explored_edge_dirs = np.zeros(
        (max_explored_edges, 2), dtype=np.float32)
    explored_edge_mask = np.zeros(max_explored_edges, dtype=np.bool_)
    explored_is_incoming = np.zeros(max_explored_edges, dtype=np.bool_)

    records = _neighbor_records(path, search_state)
    for index, record in enumerate(records[:max_explored_edges]):
        is_incoming, _, _, _, direction = record
        explored_edge_dirs[index] = direction
        explored_edge_mask[index] = True
        explored_is_incoming[index] = is_incoming

    return {
        "incoming_dir": torch.from_numpy(incoming_dir),
        "incoming_valid": torch.tensor(
            incoming_valid, dtype=torch.bool),
        "explored_edge_dirs": torch.from_numpy(explored_edge_dirs),
        "explored_edge_mask": torch.from_numpy(explored_edge_mask),
        "explored_is_incoming": torch.from_numpy(explored_is_incoming),
        "is_key_point": torch.tensor(
            bool(search_state.is_key_point), dtype=torch.bool),
    }
