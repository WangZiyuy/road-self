"""Non-circular support-guided trajectory fusion for Stage 3D-C1."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from model.trajectory_support_head import TrajectorySupportHead
from utils.trajectory_support_aggregation import (
    permute_valid_fragment_values,
    recompute_branch_predictions,
    support_weighted_trajectory_context,
)
from utils.trajectory_support_features import (
    build_pre_trajectory_branch_tokens,
)


FUSION_ORIGINAL_ATTENTION = "original_attention"
FUSION_SUPPORT_AGGREGATION = "support_aggregation"
VALID_TRAJECTORY_FUSION_MODES = {
    FUSION_ORIGINAL_ATTENTION,
    FUSION_SUPPORT_AGGREGATION,
}


def resolve_trajectory_fusion_mode(cfg) -> str:
    """Resolve C1 fusion mode while preserving E4 as the default."""

    mode = str(getattr(
        cfg, "TRAJ_FUSION_MODE", FUSION_ORIGINAL_ATTENTION
    )).strip().lower()
    if mode not in VALID_TRAJECTORY_FUSION_MODES:
        raise ValueError(
            "unknown TRAJ_FUSION_MODE {!r}; expected one of {}".format(
                mode, sorted(VALID_TRAJECTORY_FUSION_MODES)))
    return mode


class SupportGuidedTrajectoryFusion(nn.Module):
    """Score fragments before trajectory fusion and pool them per query."""

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        branch_input_dim: int = 256,
        projection_dim: int = 128,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.hidden_dim = int(hidden_dim)
        self.support_head = TrajectorySupportHead(
            hidden_dim=hidden_dim,
            projection_dim=projection_dim,
            branch_input_dim=branch_input_dim,
            fragment_input_dim=hidden_dim,
        )
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

    def initialize_aggregation_from_decoder(
        self,
        branch_decoder: nn.Module,
    ) -> None:
        """Copy E4 attention V/out projections without changing E4."""

        attention = branch_decoder.trajectory_cross_attention
        hidden_dim = int(attention.embed_dim)
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                "decoder and fusion hidden dimensions differ")
        if attention.in_proj_weight is None:
            raise ValueError(
                "separate q/k/v projection weights are unsupported")
        with torch.no_grad():
            self.value_projection.weight.copy_(
                attention.in_proj_weight[
                    2 * hidden_dim:3 * hidden_dim])
            if attention.in_proj_bias is None:
                self.value_projection.bias.zero_()
            else:
                self.value_projection.bias.copy_(
                    attention.in_proj_bias[
                        2 * hidden_dim:3 * hidden_dim])
            self.output_projection.load_state_dict(
                attention.out_proj.state_dict(), strict=True)

    def forward(
        self,
        *,
        pre_trajectory_branch_tokens: torch.Tensor,
        fragment_tokens: torch.Tensor,
        normalized_fragment_tokens: torch.Tensor,
        fragment_mask: torch.Tensor,
        top_k: Optional[int] = None,
        randomize_fragment_values: bool = False,
        sample_ids: Optional[torch.Tensor] = None,
        random_seed: int = 0,
        epsilon: float = 1e-6,
    ) -> Dict[str, torch.Tensor]:
        if tuple(fragment_tokens.shape) != tuple(
                normalized_fragment_tokens.shape):
            raise ValueError(
                "raw and normalized fragment tokens must share shape")
        support_logits = self.support_head(
            pre_trajectory_branch_tokens,
            fragment_tokens,
            fragment_mask,
        )
        fragment_values = self.value_projection(
            normalized_fragment_tokens)
        fragment_values = fragment_values * fragment_mask.to(
            dtype=torch.bool).unsqueeze(-1).to(
                dtype=fragment_values.dtype)
        if randomize_fragment_values:
            if sample_ids is None:
                raise ValueError(
                    "sample_ids are required for random aggregation")
            fragment_values = permute_valid_fragment_values(
                fragment_values,
                fragment_mask,
                sample_ids,
                seed=int(random_seed),
            )
        aggregation = support_weighted_trajectory_context(
            support_logits,
            fragment_values,
            fragment_mask,
            epsilon=epsilon,
            output_projection=self.output_projection,
            top_k=top_k,
        )
        aggregation.update({
            "support_logits": support_logits,
            "fragment_values": fragment_values,
        })
        return aggregation


def forward_branch_with_trajectory_fusion(
    *,
    branch_decoder: nn.Module,
    fusion_module: Optional[SupportGuidedTrajectoryFusion],
    fusion_mode: str,
    stage_fuse: torch.Tensor,
    state_token: torch.Tensor,
    fragment_tokens: torch.Tensor,
    fragment_mask: torch.Tensor,
    walked_path: torch.Tensor,
    sample_ids: Optional[torch.Tensor] = None,
    top_k: Optional[int] = None,
    randomize_fragment_values: bool = False,
    random_seed: int = 0,
    epsilon: float = 1e-6,
    return_attention: bool = False,
    return_debug_states: bool = False,
) -> Dict[str, torch.Tensor]:
    """Run either strict E4 attention or the non-circular C1 path."""

    mode = str(fusion_mode).strip().lower()
    if mode == FUSION_ORIGINAL_ATTENTION:
        return branch_decoder(
            stage_fuse=stage_fuse,
            state_token=state_token,
            fragment_tokens=fragment_tokens,
            fragment_mask=fragment_mask,
            walked_path=walked_path,
            return_attention=return_attention,
            return_debug_states=return_debug_states,
        )
    if mode != FUSION_SUPPORT_AGGREGATION:
        raise ValueError(
            "unknown trajectory fusion mode {!r}".format(mode))
    if fusion_module is None:
        raise ValueError(
            "support_aggregation requires a fusion module")

    empty_mask = torch.zeros_like(
        fragment_mask, dtype=torch.bool)
    pre_trajectory = branch_decoder(
        stage_fuse=stage_fuse,
        state_token=state_token,
        # The empty mask guarantees that this call never reads trajectory
        # values. Detaching avoids retaining an unnecessary decoder graph
        # when C1-b trains the trajectory encoder.
        fragment_tokens=fragment_tokens.detach(),
        fragment_mask=empty_mask,
        walked_path=walked_path,
        return_attention=return_attention,
        return_debug_states=True,
    )
    graph_queries = pre_trajectory[
        "debug_graph_conditioned_queries"]
    image_context = pre_trajectory[
        "debug_image_cross_attention_output"]
    pre_branch_tokens = build_pre_trajectory_branch_tokens(
        graph_queries, image_context)
    normalized_fragments = branch_decoder.trajectory_norm(
        branch_decoder.trajectory_projection(fragment_tokens))
    normalized_fragments = normalized_fragments * fragment_mask.to(
        dtype=torch.bool).unsqueeze(-1).to(
            dtype=normalized_fragments.dtype)
    fusion = fusion_module(
        pre_trajectory_branch_tokens=pre_branch_tokens,
        fragment_tokens=fragment_tokens,
        normalized_fragment_tokens=normalized_fragments,
        fragment_mask=fragment_mask,
        top_k=top_k,
        randomize_fragment_values=randomize_fragment_values,
        sample_ids=sample_ids,
        random_seed=random_seed,
        epsilon=epsilon,
    )
    outputs = recompute_branch_predictions(
        branch_decoder,
        graph_conditioned_queries=graph_queries,
        image_context=image_context,
        trajectory_context=fusion["context"],
        graph_state_contribution=pre_trajectory[
            "debug_graph_state_contribution"],
    )
    outputs.update({
        "fragment_support_logits": fusion["support_logits"],
        "fragment_support_probabilities": fusion[
            "support_probabilities"],
        "trajectory_context": fusion["context"],
        "trajectory_selection_mask": fusion["selection_mask"],
        "trajectory_weight_sum": fusion["weight_sum"],
    })
    if return_attention:
        outputs.update({
            "image_attention_weights": pre_trajectory[
                "image_attention_weights"],
            "trajectory_attention_weights": fusion["masked_weights"],
        })
    if return_debug_states:
        outputs.update({
            key: value
            for key, value in pre_trajectory.items()
            if key.startswith("debug_")
        })
        outputs["debug_trajectory_cross_attention_output"] = fusion[
            "context"]
        outputs["debug_final_fused_queries"] = outputs[
            "branch_tokens"]
        outputs["debug_pre_trajectory_branch_tokens"] = (
            pre_branch_tokens)
    return outputs
