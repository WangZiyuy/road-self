"""Frozen trajectory-context aggregation for Stage 3D-C0 diagnostics.

The helpers in this module reuse an existing E4 decoder's projection, fusion,
and output heads.  They do not register parameters or alter the decoder.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F


def decoder_fragment_values(
    branch_decoder: torch.nn.Module,
    fragment_tokens: torch.Tensor,
    fragment_mask: torch.Tensor,
) -> torch.Tensor:
    """Return the value-projected fragments used inside E4 attention.

    Support aggregation replaces only the attention allocation, so it keeps
    E4's frozen value projection and later output projection.
    """

    if fragment_tokens.ndim != 3:
        raise ValueError("fragment_tokens must have shape [B, N, D]")
    if tuple(fragment_mask.shape) != tuple(fragment_tokens.shape[:2]):
        raise ValueError("fragment_mask must have shape [B, N]")
    if fragment_tokens.device != fragment_mask.device:
        raise ValueError("fragment tokens and mask must share one device")
    normalized = branch_decoder.trajectory_projection(fragment_tokens)
    normalized = branch_decoder.trajectory_norm(normalized)
    attention = branch_decoder.trajectory_cross_attention
    hidden_dim = int(attention.embed_dim)
    if attention.in_proj_weight is None:
        raise ValueError(
            "separate q/k/v projection weights are not supported")
    value_weight = attention.in_proj_weight[
        2 * hidden_dim:3 * hidden_dim]
    value_bias = (
        None
        if attention.in_proj_bias is None
        else attention.in_proj_bias[2 * hidden_dim:3 * hidden_dim]
    )
    values = F.linear(normalized, value_weight, value_bias)
    return values * fragment_mask.to(
        dtype=torch.bool).unsqueeze(-1).to(dtype=values.dtype)


def support_weighted_trajectory_context(
    support_logits: torch.Tensor,
    fragment_values: torch.Tensor,
    fragment_mask: torch.Tensor,
    *,
    epsilon: float = 1e-6,
    output_projection: Optional[torch.nn.Module] = None,
    top_k: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Pool fragment values independently for every branch query.

    Supplying E4 attention's frozen ``out_proj`` places the result in the
    exact trajectory-context feature space consumed by branch fusion.
    """

    if support_logits.ndim != 3:
        raise ValueError("support_logits must have shape [B, K, N]")
    if fragment_values.ndim != 3:
        raise ValueError("fragment_values must have shape [B, N, D]")
    if (
            support_logits.shape[0] != fragment_values.shape[0]
            or support_logits.shape[2] != fragment_values.shape[1]):
        raise ValueError("support logits and fragment values are incompatible")
    if tuple(fragment_mask.shape) != tuple(fragment_values.shape[:2]):
        raise ValueError("fragment_mask must have shape [B, N]")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if top_k is not None and int(top_k) <= 0:
        raise ValueError("top_k must be positive when supplied")
    if (
            support_logits.device != fragment_values.device
            or fragment_values.device != fragment_mask.device):
        raise ValueError("aggregation inputs must share one device")

    valid = fragment_mask.to(dtype=torch.bool)
    probabilities = torch.sigmoid(support_logits)
    selection_mask = valid.unsqueeze(1).expand_as(
        support_logits)
    if top_k is not None:
        selection_mask = torch.zeros_like(
            support_logits, dtype=torch.bool)
        limit = min(int(top_k), support_logits.shape[-1])
        if limit:
            ranked_logits = support_logits.masked_fill(
                ~valid.unsqueeze(1), float("-inf"))
            selected_indices = torch.topk(
                ranked_logits,
                k=limit,
                dim=-1,
                largest=True,
                sorted=True,
            ).indices
            selection_mask.scatter_(-1, selected_indices, True)
            selection_mask = (
                selection_mask & valid.unsqueeze(1))
    weights = probabilities * selection_mask.to(
        dtype=probabilities.dtype)
    denominator = weights.sum(dim=-1, keepdim=True)
    context = torch.matmul(weights, fragment_values)
    context = context / denominator.clamp_min(float(epsilon))
    context_valid = denominator > 0
    if output_projection is not None:
        context = output_projection(context)
    context = context * context_valid.to(dtype=context.dtype)
    return {
        "context": context,
        "support_probabilities": probabilities,
        "masked_weights": weights,
        "selection_mask": selection_mask,
        "weight_sum": denominator.squeeze(-1),
    }


def permute_valid_fragment_values(
    fragment_values: torch.Tensor,
    fragment_mask: torch.Tensor,
    sample_ids: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    """Deterministically break support/fragment alignment per sample.

    Only valid fragment values are permuted.  The support scores and fragment
    count stay unchanged, so this is a controlled random-association baseline.
    """

    if fragment_values.ndim != 3:
        raise ValueError("fragment_values must have shape [B, N, D]")
    if tuple(fragment_mask.shape) != tuple(fragment_values.shape[:2]):
        raise ValueError("fragment_mask must have shape [B, N]")
    if tuple(sample_ids.shape) != (fragment_values.shape[0],):
        raise ValueError("sample_ids must have shape [B]")
    if (
            fragment_values.device != fragment_mask.device
            or fragment_values.device != sample_ids.device):
        raise ValueError("permutation inputs must share one device")

    output = torch.zeros_like(fragment_values)
    mask = fragment_mask.to(dtype=torch.bool)
    for batch_index in range(fragment_values.shape[0]):
        valid_indices = torch.nonzero(
            mask[batch_index], as_tuple=False).flatten()
        count = int(valid_indices.numel())
        if count == 0:
            continue
        generator = torch.Generator(device="cpu")
        sample_seed = (
            int(seed)
            + 1_000_003 * int(sample_ids[batch_index].item())
        ) % (2 ** 63 - 1)
        generator.manual_seed(sample_seed)
        order = torch.randperm(count, generator=generator).to(
            device=fragment_values.device)
        source_indices = valid_indices.index_select(0, order)
        output[batch_index].index_copy_(
            0,
            valid_indices,
            fragment_values[batch_index].index_select(
                0, source_indices),
        )
    return output


def recompute_branch_predictions(
    branch_decoder: torch.nn.Module,
    *,
    graph_conditioned_queries: torch.Tensor,
    image_context: torch.Tensor,
    trajectory_context: torch.Tensor,
    graph_state_contribution: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Reuse E4's frozen fusion and output heads with a supplied context."""

    shapes = {
        tuple(graph_conditioned_queries.shape),
        tuple(image_context.shape),
        tuple(trajectory_context.shape),
        tuple(graph_state_contribution.shape),
    }
    if len(shapes) != 1 or graph_conditioned_queries.ndim != 3:
        raise ValueError(
            "all branch fusion inputs must share shape [B, K, D]")
    devices = {
        value.device
        for value in (
            graph_conditioned_queries,
            image_context,
            trajectory_context,
            graph_state_contribution,
        )
    }
    if len(devices) != 1:
        raise ValueError("all branch fusion inputs must share one device")

    branch_tokens = branch_decoder.context_fusion(torch.cat(
        (
            graph_conditioned_queries,
            image_context,
            trajectory_context,
            graph_state_contribution,
        ),
        dim=-1,
    ))
    branch_exist_logits = branch_decoder.branch_exist_head(
        branch_tokens).squeeze(-1)
    branch_offsets_norm = torch.tanh(
        branch_decoder.branch_offset_head(branch_tokens))
    branch_directions = F.normalize(
        branch_offsets_norm, p=2, dim=-1, eps=1e-6)
    return {
        "branch_exist_logits": branch_exist_logits,
        "branch_offsets_norm": branch_offsets_norm,
        "branch_directions": branch_directions,
        "branch_tokens": branch_tokens,
    }


def freeze_modules(modules: Sequence[torch.nn.Module]) -> None:
    """Put diagnostic modules in eval mode and reject trainable parameters."""

    for module in modules:
        module.eval().requires_grad_(False)
    if any(
            parameter.requires_grad
            for module in modules
            for parameter in module.parameters()):
        raise RuntimeError("Stage 3D-C0 modules must all be frozen")
