"""Multimodal unordered branch-query decoder used as a Stage 3B side head."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalBranchQueryDecoder(nn.Module):
    """Predict an unordered immediate-road-branch set from three modalities.

    The decoder is independent of RPNet's anchor recursion.  Image features,
    graph state, and structured trajectory fragments all contribute directly
    to the auxiliary branch predictions.
    """

    def __init__(
        self,
        image_channels: int = 128,
        trajectory_dim: int = 128,
        hidden_dim: int = 128,
        num_queries: int = 6,
        num_heads: int = 4,
        image_pool_size: int = 16,
        dropout: float = 0.1,
        query_self_attention_layers: int = 0,
    ) -> None:
        super().__init__()
        if image_channels <= 0 or trajectory_dim <= 0 or hidden_dim <= 0:
            raise ValueError("feature dimensions must be positive")
        if num_queries <= 0:
            raise ValueError("num_queries must be positive")
        if num_heads <= 0 or hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by positive num_heads")
        if image_pool_size <= 0:
            raise ValueError("image_pool_size must be positive")
        if hidden_dim % 4 != 0:
            raise ValueError(
                "hidden_dim must be divisible by four for 2-D encoding")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if query_self_attention_layers not in (0, 1):
            raise ValueError(
                "query_self_attention_layers must be zero or one")

        self.hidden_dim = int(hidden_dim)
        self.trajectory_dim = int(trajectory_dim)
        self.num_queries = int(num_queries)
        self.image_pool_size = int(image_pool_size)
        self.query_self_attention_layers = int(
            query_self_attention_layers)
        self.image_projection = nn.Conv2d(
            image_channels, hidden_dim, kernel_size=1)
        self.walked_path_projection = nn.Sequential(
            nn.Conv2d(
                1, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(
                hidden_dim, hidden_dim, kernel_size=1, bias=False),
        )
        self.trajectory_projection = (
            nn.Identity()
            if trajectory_dim == hidden_dim
            else nn.Linear(trajectory_dim, hidden_dim)
        )
        self.branch_queries = nn.Parameter(
            torch.empty(1, num_queries, hidden_dim))
        nn.init.normal_(self.branch_queries, mean=0.0, std=0.02)

        self.query_norm = nn.LayerNorm(hidden_dim)
        self.image_norm = nn.LayerNorm(hidden_dim)
        self.trajectory_norm = nn.LayerNorm(hidden_dim)
        self.image_cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.trajectory_cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.context_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.branch_exist_head = nn.Linear(hidden_dim, 1)
        self.branch_offset_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        # Construct optional parameters last so enabling the experiment does
        # not perturb the random initialization of any legacy decoder layer.
        if self.query_self_attention_layers == 1:
            self.query_self_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.query_self_attention_norm = nn.LayerNorm(hidden_dim)
            self.query_self_attention_ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
            )
            self.query_self_attention_ffn_norm = nn.LayerNorm(
                hidden_dim)

    @staticmethod
    def _two_dimensional_position_encoding(
        height: int,
        width: int,
        hidden_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        quarter_dim = hidden_dim // 4
        y = torch.arange(height, device=device, dtype=torch.float32)
        x = torch.arange(width, device=device, dtype=torch.float32)
        frequency = torch.exp(
            torch.arange(
                quarter_dim, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / max(quarter_dim, 1))
        )
        y_angle = y.unsqueeze(1) * frequency.unsqueeze(0)
        x_angle = x.unsqueeze(1) * frequency.unsqueeze(0)
        y_encoding = torch.cat(
            (torch.sin(y_angle), torch.cos(y_angle)), dim=1)
        x_encoding = torch.cat(
            (torch.sin(x_angle), torch.cos(x_angle)), dim=1)
        encoding = torch.cat(
            (
                y_encoding[:, None, :].expand(-1, width, -1),
                x_encoding[None, :, :].expand(height, -1, -1),
            ),
            dim=-1,
        )
        return encoding.reshape(height * width, hidden_dim).to(dtype=dtype)

    def _trajectory_attention(
        self,
        queries: torch.Tensor,
        fragment_tokens: torch.Tensor,
        fragment_mask: torch.Tensor,
    ):
        batch_size, query_count, hidden_dim = queries.shape
        fragment_count = fragment_tokens.shape[1]
        context = queries.new_zeros(
            (batch_size, query_count, hidden_dim))
        weights = queries.new_zeros(
            (batch_size, query_count, fragment_count))
        if fragment_count == 0:
            return context, weights

        valid_samples = fragment_mask.any(dim=1)
        valid_indices = torch.nonzero(
            valid_samples, as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            return context, weights

        selected_queries = queries.index_select(0, valid_indices)
        selected_tokens = fragment_tokens.index_select(0, valid_indices)
        selected_mask = fragment_mask.index_select(0, valid_indices)
        selected_context, selected_weights = (
            self.trajectory_cross_attention(
                selected_queries,
                selected_tokens,
                selected_tokens,
                key_padding_mask=~selected_mask,
                need_weights=True,
                average_attn_weights=True,
            )
        )
        context = context.index_copy(
            0, valid_indices, selected_context)
        weights = weights.index_copy(
            0, valid_indices, selected_weights)
        return context, weights

    def forward(
        self,
        stage_fuse: torch.Tensor,
        state_token: torch.Tensor,
        fragment_tokens: Optional[torch.Tensor] = None,
        fragment_mask: Optional[torch.Tensor] = None,
        walked_path: Optional[torch.Tensor] = None,
        image_available: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        return_debug_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if stage_fuse.ndim != 4:
            raise ValueError("stage_fuse must have shape [B, C, H, W]")
        batch_size = stage_fuse.shape[0]
        if tuple(state_token.shape) != (batch_size, self.hidden_dim):
            raise ValueError("state_token must have shape [B, D]")
        if stage_fuse.device != state_token.device:
            raise ValueError("image features and state token must share device")

        if fragment_tokens is None:
            fragment_tokens = stage_fuse.new_zeros(
                (batch_size, 0, self.trajectory_dim))
        if fragment_mask is None:
            fragment_mask = torch.zeros(
                (batch_size, fragment_tokens.shape[1]),
                device=stage_fuse.device,
                dtype=torch.bool,
            )
        if (
                fragment_tokens.ndim != 3
                or fragment_tokens.shape[0] != batch_size):
            raise ValueError(
                "fragment_tokens must have shape [B, N, D]")
        if fragment_tokens.shape[-1] != self.trajectory_dim:
            raise ValueError(
                "fragment token dimension does not match trajectory_dim")
        if tuple(fragment_mask.shape) != tuple(fragment_tokens.shape[:2]):
            raise ValueError("fragment_mask must have shape [B, N]")
        if (
                fragment_tokens.device != stage_fuse.device
                or fragment_mask.device != stage_fuse.device):
            raise ValueError("all decoder inputs must share one device")
        if walked_path is None:
            walked_path = stage_fuse.new_zeros(
                (batch_size, 1, stage_fuse.shape[2], stage_fuse.shape[3]))
        if (
                walked_path.ndim != 4
                or walked_path.shape[0] != batch_size
                or walked_path.shape[1] != 1):
            raise ValueError(
                "walked_path must have shape [B, 1, Hf, Wf]")
        if walked_path.device != stage_fuse.device:
            raise ValueError(
                "walked_path and image features must share one device")
        if image_available is None:
            image_available = torch.ones(
                batch_size,
                dtype=torch.bool,
                device=stage_fuse.device,
            )
        if tuple(image_available.shape) != (batch_size,):
            raise ValueError("image_available must have shape [B]")
        if image_available.device != stage_fuse.device:
            raise ValueError(
                "image_available and image features must share one device")
        image_available = image_available.to(dtype=torch.bool)

        pooled_image = F.adaptive_avg_pool2d(
            stage_fuse,
            (self.image_pool_size, self.image_pool_size),
        )
        image_tokens = self.image_projection(pooled_image)
        walked_features = self.walked_path_projection(
            walked_path.to(dtype=stage_fuse.dtype))
        walked_features = F.adaptive_avg_pool2d(
            walked_features,
            (self.image_pool_size, self.image_pool_size),
        )
        image_tokens = image_tokens + walked_features
        image_tokens = image_tokens.flatten(2).transpose(1, 2)
        position = self._two_dimensional_position_encoding(
            self.image_pool_size,
            self.image_pool_size,
            self.hidden_dim,
            image_tokens.device,
            image_tokens.dtype,
        )
        image_tokens = self.image_norm(
            image_tokens + position.unsqueeze(0))

        learned_queries = self.branch_queries.expand(
            batch_size, -1, -1)
        queries = learned_queries
        if self.query_self_attention_layers == 1:
            self_attention_output, _ = self.query_self_attention(
                queries,
                queries,
                queries,
                need_weights=False,
            )
            queries = self.query_self_attention_norm(
                queries + self_attention_output)
            queries = self.query_self_attention_ffn_norm(
                queries + self.query_self_attention_ffn(queries))
        pre_graph_queries = queries
        graph_conditioned_queries = self.query_norm(
            queries + state_token.unsqueeze(1))
        queries = graph_conditioned_queries
        image_context, image_attention = self.image_cross_attention(
            queries,
            image_tokens,
            image_tokens,
            need_weights=True,
            average_attn_weights=True,
        )
        image_context = image_context * image_available[:, None, None].to(
            dtype=image_context.dtype)
        image_attention = image_attention * image_available[
            :, None, None].to(dtype=image_attention.dtype)

        fragment_tokens = self.trajectory_projection(fragment_tokens)
        fragment_mask = fragment_mask.to(dtype=torch.bool)
        fragment_tokens = self.trajectory_norm(fragment_tokens)
        fragment_tokens = fragment_tokens * fragment_mask.unsqueeze(
            -1).to(dtype=fragment_tokens.dtype)
        trajectory_context, trajectory_attention = (
            self._trajectory_attention(
                queries, fragment_tokens, fragment_mask)
        )

        expanded_state = state_token.unsqueeze(1).expand(
            -1, self.num_queries, -1)
        branch_tokens = self.context_fusion(torch.cat(
            (
                queries,
                image_context,
                trajectory_context,
                expanded_state,
            ),
            dim=-1,
        ))
        branch_exist_logits = self.branch_exist_head(
            branch_tokens).squeeze(-1)
        branch_offsets_norm = torch.tanh(
            self.branch_offset_head(branch_tokens))
        branch_directions = F.normalize(
            branch_offsets_norm, p=2, dim=-1, eps=1e-6)

        outputs = {
            "branch_exist_logits": branch_exist_logits,
            "branch_offsets_norm": branch_offsets_norm,
            "branch_directions": branch_directions,
            "branch_tokens": branch_tokens,
        }
        if return_attention:
            # Attention is diagnostic allocation, not a calibrated support
            # probability or a trajectory-reliability target.
            outputs["image_attention_weights"] = image_attention
            outputs["trajectory_attention_weights"] = trajectory_attention
        if return_debug_states:
            # These tensors are exposed only for collapse diagnosis.  They
            # remain attached so callers may also inspect gradient flow.
            outputs.update({
                "debug_learned_query_embedding": learned_queries,
                "debug_graph_conditioned_queries":
                    graph_conditioned_queries,
                "debug_pre_graph_queries": pre_graph_queries,
                "debug_pre_cross_attention_queries": queries,
                "debug_image_cross_attention_output": image_context,
                "debug_trajectory_cross_attention_output":
                    trajectory_context,
                "debug_final_fused_queries": branch_tokens,
                "debug_graph_state_contribution":
                    expanded_state,
            })
        return outputs
