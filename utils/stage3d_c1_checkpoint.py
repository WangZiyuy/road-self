"""Checkpoint lifecycle for Stage 3D-C1 support-guided fusion."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch


SCHEMA_VERSION = "stage3d-c1-v1"


def build_stage3d_c1_checkpoint_payload(
    *,
    fusion_module: torch.nn.Module,
    trajectory_encoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    training_stage: str,
    fusion_mode: str,
    support_top_k: Optional[int],
    random_fragment_aggregation: bool,
    e4_checkpoint: str,
    e4_checkpoint_sha256: str,
    metrics: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    if training_stage not in ("c1_a", "c1_b"):
        raise ValueError("training_stage must be c1_a or c1_b")
    return {
        "schema_version": SCHEMA_VERSION,
        "fusion_module": fusion_module.state_dict(),
        "trajectory_encoder": trajectory_encoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "training_stage": training_stage,
        "fusion_mode": str(fusion_mode),
        "support_top_k": (
            None if support_top_k is None
            else int(support_top_k)),
        "random_fragment_aggregation": bool(
            random_fragment_aggregation),
        "e4_checkpoint": os.fspath(e4_checkpoint),
        "e4_checkpoint_sha256": str(e4_checkpoint_sha256),
        "metrics": dict(metrics),
        "config": dict(config),
        "support_query_source": (
            "concat(graph_conditioned_query,image_cross_attention_context)"
        ),
        "support_reads_trajectory_context": False,
        "support_reads_final_branch_tokens": False,
        "rpnet_frozen": True,
        "branch_decoder_frozen": True,
        "feeds_anchor": False,
        "feeds_path_push": False,
    }


def save_stage3d_c1_checkpoint(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_suffix(resolved.suffix + ".tmp")
    torch.save(dict(payload), str(temporary))
    os.replace(str(temporary), str(resolved))


def load_stage3d_c1_checkpoint(
    path: Path,
    *,
    fusion_module: torch.nn.Module,
    trajectory_encoder: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Any = "cpu",
    expected_e4_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    resolved = Path(path).resolve(strict=False)
    if not resolved.is_file():
        raise FileNotFoundError(
            "Stage 3D-C1 checkpoint not found: {}".format(resolved))
    payload = torch.load(str(resolved), map_location=map_location)
    if not isinstance(payload, Mapping):
        raise ValueError("Stage 3D-C1 checkpoint must be a mapping")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            "unsupported Stage 3D-C1 schema: {!r}".format(
                payload.get("schema_version")))
    if expected_e4_sha256 is not None and (
            payload.get("e4_checkpoint_sha256")
            != expected_e4_sha256):
        raise ValueError(
            "Stage 3D-C1 checkpoint E4 SHA-256 mismatch")
    fusion_module.load_state_dict(
        payload["fusion_module"], strict=True)
    if trajectory_encoder is not None:
        trajectory_encoder.load_state_dict(
            payload["trajectory_encoder"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return dict(payload)
