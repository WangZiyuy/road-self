"""Continuous encoder for a queued VecRoad graph-exploration state."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class GraphStateEncoder(nn.Module):
    """Encode incoming direction and explored neighbors into one state token.

    Explored edges are processed by a shared MLP and permutation-invariant
    masked pooling.  No direction quantization or persistent vertex embedding
    is used.
    """

    REQUIRED_KEYS = (
        "incoming_dir",
        "incoming_valid",
        "explored_edge_dirs",
        "explored_edge_mask",
        "explored_is_incoming",
        "is_key_point",
    )

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.hidden_dim = int(hidden_dim)
        self.node_projection = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.edge_projection = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    @classmethod
    def _validate(cls, graph_state: Dict[str, torch.Tensor]) -> None:
        missing = [key for key in cls.REQUIRED_KEYS if key not in graph_state]
        if missing:
            raise KeyError(
                "graph state is missing: {}".format(", ".join(missing)))
        for key in cls.REQUIRED_KEYS:
            if not torch.is_tensor(graph_state[key]):
                raise TypeError(
                    "graph state field {!r} must be a tensor".format(key))

        incoming_dir = graph_state["incoming_dir"]
        if incoming_dir.ndim != 2 or incoming_dir.shape[-1] != 2:
            raise ValueError("incoming_dir must have shape [B, 2]")
        batch_size = incoming_dir.shape[0]
        if tuple(graph_state["incoming_valid"].shape) != (batch_size,):
            raise ValueError("incoming_valid must have shape [B]")
        if tuple(graph_state["is_key_point"].shape) != (batch_size,):
            raise ValueError("is_key_point must have shape [B]")

        edge_dirs = graph_state["explored_edge_dirs"]
        if (
                edge_dirs.ndim != 3
                or edge_dirs.shape[0] != batch_size
                or edge_dirs.shape[-1] != 2):
            raise ValueError(
                "explored_edge_dirs must have shape [B, E, 2]")
        edge_shape = tuple(edge_dirs.shape[:2])
        for key in ("explored_edge_mask", "explored_is_incoming"):
            if tuple(graph_state[key].shape) != edge_shape:
                raise ValueError("{} must have shape [B, E]".format(key))

        devices = {
            graph_state[key].device for key in cls.REQUIRED_KEYS
        }
        if len(devices) != 1:
            raise ValueError("all graph state fields must share one device")

    @staticmethod
    def _masked_edge_pool(
        edge_tokens: torch.Tensor,
        edge_mask: torch.Tensor,
    ):
        batch_size, edge_count, hidden_dim = edge_tokens.shape
        if edge_count == 0:
            zeros = edge_tokens.new_zeros((batch_size, hidden_dim))
            return zeros, zeros

        mask_float = edge_mask.unsqueeze(-1).to(dtype=edge_tokens.dtype)
        edge_sum = (edge_tokens * mask_float).sum(dim=1)
        denominator = mask_float.sum(dim=1).clamp_min(1.0)
        edge_mean = edge_sum / denominator

        minimum = torch.finfo(edge_tokens.dtype).min
        masked_for_max = edge_tokens.masked_fill(
            ~edge_mask.unsqueeze(-1), minimum)
        edge_max = masked_for_max.max(dim=1).values
        has_edge = edge_mask.any(dim=1, keepdim=True)
        edge_max = torch.where(
            has_edge, edge_max, torch.zeros_like(edge_max))
        return edge_mean, edge_max

    def forward(
        self,
        graph_state: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        self._validate(graph_state)
        parameter = next(self.parameters())
        if graph_state["incoming_dir"].device != parameter.device:
            raise ValueError(
                "graph state and encoder must be on the same device")
        dtype = parameter.dtype

        incoming_valid = graph_state["incoming_valid"].to(
            dtype=torch.bool)
        incoming_dir = graph_state["incoming_dir"].to(dtype=dtype)
        incoming_dir = incoming_dir * incoming_valid.unsqueeze(
            -1).to(dtype=dtype)
        node_features = torch.cat(
            (
                incoming_dir,
                incoming_valid.unsqueeze(-1).to(dtype=dtype),
                graph_state["is_key_point"].unsqueeze(-1).to(dtype=dtype),
            ),
            dim=-1,
        )
        node_token = self.node_projection(node_features)

        edge_mask = graph_state["explored_edge_mask"].to(dtype=torch.bool)
        edge_features = torch.cat(
            (
                graph_state["explored_edge_dirs"].to(dtype=dtype),
                graph_state["explored_is_incoming"].unsqueeze(-1).to(
                    dtype=dtype),
            ),
            dim=-1,
        )
        edge_tokens = self.edge_projection(edge_features)
        edge_tokens = edge_tokens * edge_mask.unsqueeze(-1).to(dtype=dtype)
        edge_mean, edge_max = self._masked_edge_pool(
            edge_tokens, edge_mask)
        return self.fusion(torch.cat(
            (node_token, edge_mean, edge_max), dim=-1))
