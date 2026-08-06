"""Checkpoint lifecycle for the isolated Stage 3F-A fusion module."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch


SCHEMA = "stage3fa-anchor-fusion-v1"


def build_stage3fa_checkpoint_payload(
    *, fusion: torch.nn.Module, optimizer: torch.optim.Optimizer,
    epoch: int, seed: int, validation_anchor_total_loss: float,
    checkpoint_sha256: Mapping[str, str],
    frozen_module_sha256: Mapping[str, str],
    config_snapshot: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "stage": "3F-A",
        "epoch": int(epoch),
        "seed": int(seed),
        "trajectory_anchor_fusion": fusion.state_dict(),
        "optimizer": optimizer.state_dict(),
        "validation_anchor_total_loss": float(
            validation_anchor_total_loss),
        "checkpoint_sha256": dict(checkpoint_sha256),
        "frozen_module_sha256": dict(frozen_module_sha256),
        "config_snapshot": dict(config_snapshot),
        "feeds_path_push": False,
        "changes_branch_predictions": False,
    }


def save_stage3fa_checkpoint(path: Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(payload), str(temporary))
    os.replace(str(temporary), str(path))
    return path


def load_stage3fa_checkpoint(
    path: Path, *, fusion: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Any = "cpu",
) -> Dict[str, Any]:
    resolved = Path(path).resolve(strict=False)
    if not resolved.is_file():
        raise FileNotFoundError(
            "Stage 3F-A checkpoint not found: {}".format(resolved))
    payload = torch.load(str(resolved), map_location=map_location)
    if payload.get("schema_version") != SCHEMA:
        raise ValueError("unsupported Stage 3F-A checkpoint schema")
    fusion.load_state_dict(payload["trajectory_anchor_fusion"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return payload
