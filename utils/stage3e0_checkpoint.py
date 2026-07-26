"""Strict checkpoint lifecycle for Stage 3E-0 evidence experiments."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch


STAGE3E0_CHECKPOINT_SCHEMA = "stage3e0-evidence-v1"


def build_stage3e0_checkpoint_payload(
    *,
    evidence_encoder: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    trajectory_mode: str,
    e4_checkpoint: str,
    e4_checkpoint_sha256: str,
    config: Mapping[str, Any],
    metrics: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "schema_version": STAGE3E0_CHECKPOINT_SCHEMA,
        "stage": "3E-0",
        "epoch": int(epoch),
        "trajectory_mode": str(trajectory_mode),
        "e4_checkpoint": str(e4_checkpoint),
        "e4_checkpoint_sha256": str(e4_checkpoint_sha256),
        "trajectory_evidence_encoder":
            evidence_encoder.state_dict(),
        "optimizer": (
            None if optimizer is None else optimizer.state_dict()),
        "config_snapshot": dict(config),
        "metrics": dict(metrics or {}),
        "metadata": {
            "rpnet_frozen": True,
            "graph_state_encoder_frozen": True,
            "trajectory_fragment_encoder_frozen": True,
            "branch_decoder_frozen": True,
            "evidence_queries_are_branch_independent": True,
            "feeds_anchor": False,
            "feeds_path_push": False,
        },
    }


def save_stage3e0_checkpoint(
    path: Path,
    payload: Mapping[str, Any],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(payload), str(temporary))
    os.replace(str(temporary), str(path))
    return path


def load_stage3e0_checkpoint(
    path: Path,
    *,
    evidence_encoder: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Any = "cpu",
    expected_e4_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    resolved = Path(path).resolve(strict=False)
    if not resolved.is_file():
        raise FileNotFoundError(
            "Stage 3E-0 checkpoint not found: {}".format(resolved))
    payload = torch.load(str(resolved), map_location=map_location)
    if payload.get("schema_version") != STAGE3E0_CHECKPOINT_SCHEMA:
        raise ValueError(
            "unsupported Stage 3E-0 checkpoint schema: {!r}".format(
                payload.get("schema_version")))
    required = (
        "trajectory_evidence_encoder",
        "optimizer",
        "e4_checkpoint_sha256",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(
            "Stage 3E-0 checkpoint is missing: {}".format(
                ", ".join(missing)))
    if (
            expected_e4_sha256 is not None
            and payload["e4_checkpoint_sha256"]
            != expected_e4_sha256):
        raise ValueError("Stage 3E-0 E4 checkpoint SHA-256 mismatch")
    evidence_encoder.load_state_dict(
        payload["trajectory_evidence_encoder"], strict=True)
    if optimizer is not None:
        if payload["optimizer"] is None:
            raise ValueError(
                "Stage 3E-0 checkpoint has no optimizer state")
        optimizer.load_state_dict(payload["optimizer"])
    return payload
