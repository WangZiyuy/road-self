"""Branch-independent latent encoding of local trajectory evidence."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


TRAJECTORY_MODE_ORIGINAL_FRAGMENT = "original_fragment"
TRAJECTORY_MODE_EVIDENCE = "trajectory_evidence"
EVIDENCE_AGGREGATION_LATENT_ATTENTION = "latent_attention"
EVIDENCE_AGGREGATION_MASKED_MEAN = "masked_mean"
VALID_EVIDENCE_AGGREGATION_MODES = {
    EVIDENCE_AGGREGATION_LATENT_ATTENTION,
    EVIDENCE_AGGREGATION_MASKED_MEAN,
}
VALID_TRAJECTORY_EVIDENCE_MODES = {
    TRAJECTORY_MODE_ORIGINAL_FRAGMENT,
    TRAJECTORY_MODE_EVIDENCE,
}


def resolve_trajectory_evidence_mode(cfg) -> str:
    """Resolve the Stage 3E-0 ablation mode with a legacy-safe default."""

    mode = str(getattr(
        cfg,
        "TRAJECTORY_MODE",
        TRAJECTORY_MODE_ORIGINAL_FRAGMENT,
    )).strip().lower()
    if mode not in VALID_TRAJECTORY_EVIDENCE_MODES:
        raise ValueError(
            "unknown TRAJECTORY_MODE {!r}; expected one of {}".format(
                mode, sorted(VALID_TRAJECTORY_EVIDENCE_MODES)))
    return mode


def resolve_evidence_aggregation_mode(cfg) -> str:
    """Resolve the evidence aggregator without changing Stage 3E-0 defaults."""

    model_cfg = getattr(
        getattr(cfg, "STAGE3E0", None), "MODEL", None)
    mode = str(getattr(
        model_cfg,
        "AGGREGATION_MODE",
        EVIDENCE_AGGREGATION_LATENT_ATTENTION,
    )).strip().lower()
    if mode not in VALID_EVIDENCE_AGGREGATION_MODES:
        raise ValueError(
            "unknown evidence AGGREGATION_MODE {!r}; expected one of "
            "{}".format(
                mode, sorted(VALID_EVIDENCE_AGGREGATION_MODES)))
    return mode


class TrajectoryEvidenceEncoder(nn.Module):
    """Summarize a fragment set with branch-independent latent queries.

    The latent queries only read structured trajectory fragment tokens.
    They have no access to branch queries, imagery, walked path, or graph
    exploration state.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        num_evidence_tokens: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        aggregation_mode: str = EVIDENCE_AGGREGATION_LATENT_ATTENTION,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_evidence_tokens <= 0:
            raise ValueError("num_evidence_tokens must be positive")
        if num_heads <= 0 or hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by positive num_heads")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")
        aggregation_mode = str(aggregation_mode).strip().lower()
        if aggregation_mode not in VALID_EVIDENCE_AGGREGATION_MODES:
            raise ValueError(
                "unknown aggregation_mode {!r}; expected one of {}".format(
                    aggregation_mode,
                    sorted(VALID_EVIDENCE_AGGREGATION_MODES),
                )
            )
        if (
                aggregation_mode == EVIDENCE_AGGREGATION_MASKED_MEAN
                and num_evidence_tokens != 1):
            raise ValueError(
                "masked_mean requires num_evidence_tokens=1")

        self.hidden_dim = int(hidden_dim)
        self.num_evidence_tokens = int(num_evidence_tokens)
        self.aggregation_mode = aggregation_mode
        if aggregation_mode == EVIDENCE_AGGREGATION_MASKED_MEAN:
            # This is deliberately parameter-free: adding a projection or
            # affine adapter would no longer be a strict masked-mean baseline.
            self.fragment_norm = None
            self.cross_attention = None
            self.output_norm = None
            self.trajectory_queries = None
            return

        # Initialize all shared layers before the size-dependent query tensor.
        # With a fixed seed this keeps their initialization identical across
        # M=1/4/8 capacity ablations.
        self.fragment_norm = nn.LayerNorm(self.hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.trajectory_queries = nn.Parameter(torch.empty(
            1, self.num_evidence_tokens, self.hidden_dim))
        nn.init.normal_(self.trajectory_queries, mean=0.0, std=0.02)

    def forward(
        self,
        fragment_tokens: torch.Tensor,
        fragment_mask: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if fragment_tokens.ndim != 3:
            raise ValueError(
                "fragment_tokens must have shape [B, N, D]")
        if fragment_tokens.shape[-1] != self.hidden_dim:
            raise ValueError(
                "fragment token dimension does not match hidden_dim")
        if tuple(fragment_mask.shape) != tuple(
                fragment_tokens.shape[:2]):
            raise ValueError("fragment_mask must have shape [B, N]")
        if fragment_tokens.device != fragment_mask.device:
            raise ValueError(
                "fragment_tokens and fragment_mask must share a device")

        batch_size, fragment_count, _ = fragment_tokens.shape
        mask = fragment_mask.to(dtype=torch.bool)
        evidence_tokens = fragment_tokens.new_zeros((
            batch_size,
            self.num_evidence_tokens,
            self.hidden_dim,
        ))
        evidence_mask = torch.zeros(
            (batch_size, self.num_evidence_tokens),
            dtype=torch.bool,
            device=fragment_tokens.device,
        )
        attention = fragment_tokens.new_zeros((
            batch_size,
            self.num_evidence_tokens,
            fragment_count,
        ))
        if self.aggregation_mode == EVIDENCE_AGGREGATION_MASKED_MEAN:
            counts = mask.sum(dim=1, keepdim=True)
            valid_samples = counts.squeeze(1) > 0
            weights = mask.to(dtype=fragment_tokens.dtype)
            weights = weights / counts.clamp_min(1).to(
                dtype=fragment_tokens.dtype)
            evidence_tokens[:, 0] = torch.sum(
                weights.unsqueeze(-1) * fragment_tokens,
                dim=1,
            )
            evidence_mask[:, 0] = valid_samples
            attention[:, 0] = weights
            outputs = {
                "trajectory_evidence_tokens": evidence_tokens,
                "trajectory_evidence_mask": evidence_mask,
            }
            if return_attention:
                outputs["fragment_attention_weights"] = attention
            return outputs

        if fragment_count > 0:
            valid_samples = mask.any(dim=1)
            valid_indices = torch.nonzero(
                valid_samples, as_tuple=False).flatten()
            if valid_indices.numel() > 0:
                fragments = self.fragment_norm(
                    fragment_tokens.index_select(0, valid_indices))
                selected_mask = mask.index_select(0, valid_indices)
                queries = self.trajectory_queries.expand(
                    valid_indices.numel(), -1, -1)
                attended, selected_attention = self.cross_attention(
                    queries,
                    fragments,
                    fragments,
                    key_padding_mask=~selected_mask,
                    need_weights=True,
                    average_attn_weights=True,
                )
                attended = self.output_norm(queries + attended)
                evidence_tokens = evidence_tokens.index_copy(
                    0, valid_indices, attended)
                evidence_mask[valid_indices] = True
                attention = attention.index_copy(
                    0, valid_indices, selected_attention)

        outputs = {
            "trajectory_evidence_tokens": evidence_tokens,
            "trajectory_evidence_mask": evidence_mask,
        }
        if return_attention:
            outputs["fragment_attention_weights"] = attention
        return outputs


def build_trajectory_decoder_inputs(
    *,
    trajectory_mode: str,
    fragment_tokens: torch.Tensor,
    fragment_mask: torch.Tensor,
    evidence_encoder: Optional[TrajectoryEvidenceEncoder] = None,
    return_attention: bool = False,
) -> Dict[str, Optional[torch.Tensor]]:
    """Select original fragments or branch-independent evidence tokens."""

    mode = str(trajectory_mode).strip().lower()
    if mode == TRAJECTORY_MODE_ORIGINAL_FRAGMENT:
        return {
            "decoder_trajectory_tokens": fragment_tokens,
            "decoder_trajectory_mask": fragment_mask.to(dtype=torch.bool),
            "trajectory_evidence_tokens": None,
            "trajectory_evidence_mask": None,
            "fragment_attention_weights": None,
        }
    if mode != TRAJECTORY_MODE_EVIDENCE:
        raise ValueError(
            "unknown trajectory mode {!r}".format(mode))
    if evidence_encoder is None:
        raise ValueError(
            "trajectory_evidence mode requires an evidence encoder")
    evidence = evidence_encoder(
        fragment_tokens,
        fragment_mask,
        return_attention=return_attention,
    )
    return {
        "decoder_trajectory_tokens": evidence[
            "trajectory_evidence_tokens"],
        "decoder_trajectory_mask": evidence[
            "trajectory_evidence_mask"],
        "trajectory_evidence_tokens": evidence[
            "trajectory_evidence_tokens"],
        "trajectory_evidence_mask": evidence[
            "trajectory_evidence_mask"],
        "fragment_attention_weights": evidence.get(
            "fragment_attention_weights"),
    }


def trajectory_context_from_tokens(
    *,
    branch_decoder: nn.Module,
    branch_queries: torch.Tensor,
    trajectory_tokens: torch.Tensor,
    trajectory_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Use the frozen branch decoder's existing trajectory attention."""

    projected = branch_decoder.trajectory_projection(trajectory_tokens)
    normalized = branch_decoder.trajectory_norm(projected)
    mask = trajectory_mask.to(dtype=torch.bool)
    normalized = normalized * mask.unsqueeze(-1).to(
        dtype=normalized.dtype)
    context, attention = branch_decoder._trajectory_attention(
        branch_queries,
        normalized,
        mask,
    )
    return {
        "trajectory_context": context,
        "branch_to_trajectory_attention": attention,
        "normalized_trajectory_tokens": normalized,
    }


def forward_branch_with_trajectory_mode(
    *,
    branch_decoder: nn.Module,
    trajectory_mode: str,
    stage_fuse: torch.Tensor,
    state_token: torch.Tensor,
    fragment_tokens: torch.Tensor,
    fragment_mask: torch.Tensor,
    walked_path: Optional[torch.Tensor] = None,
    image_available: Optional[torch.Tensor] = None,
    evidence_encoder: Optional[TrajectoryEvidenceEncoder] = None,
    return_attention: bool = False,
    return_debug_states: bool = False,
) -> Dict[str, torch.Tensor]:
    """Run the exact legacy fragment path or the evidence-token ablation."""

    adapted = build_trajectory_decoder_inputs(
        trajectory_mode=trajectory_mode,
        fragment_tokens=fragment_tokens,
        fragment_mask=fragment_mask,
        evidence_encoder=evidence_encoder,
        return_attention=return_attention,
    )
    outputs = branch_decoder(
        stage_fuse=stage_fuse,
        state_token=state_token,
        fragment_tokens=adapted["decoder_trajectory_tokens"],
        fragment_mask=adapted["decoder_trajectory_mask"],
        walked_path=walked_path,
        image_available=image_available,
        return_attention=return_attention,
        return_debug_states=return_debug_states,
    )
    evidence_tokens = adapted["trajectory_evidence_tokens"]
    if evidence_tokens is not None:
        outputs["trajectory_evidence_tokens"] = evidence_tokens
        outputs["trajectory_evidence_mask"] = adapted[
            "trajectory_evidence_mask"]
        if return_attention:
            outputs["fragment_attention_weights"] = adapted[
                "fragment_attention_weights"]
    return outputs
