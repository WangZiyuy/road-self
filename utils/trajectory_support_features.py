"""Feature assembly for non-circular trajectory-support experiments."""

from __future__ import annotations

import torch


def build_pre_trajectory_branch_tokens(
    graph_conditioned_queries: torch.Tensor,
    image_cross_attention_context: torch.Tensor,
) -> torch.Tensor:
    """Concatenate only E4 signals computed before trajectory attention."""

    if graph_conditioned_queries.ndim != 3:
        raise ValueError(
            "graph_conditioned_queries must have shape [B, K, D]")
    if tuple(graph_conditioned_queries.shape) != tuple(
            image_cross_attention_context.shape):
        raise ValueError(
            "graph-conditioned queries and image context must match")
    if (
            graph_conditioned_queries.device
            != image_cross_attention_context.device):
        raise ValueError("pre-trajectory inputs must share one device")
    return torch.cat(
        (graph_conditioned_queries, image_cross_attention_context),
        dim=-1,
    )
