"""Checkpoint lifecycle for the independent Stage 3D-A support head."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch


def build_stage3d_support_checkpoint_payload(
    *,
    support_head: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    e4_checkpoint: str,
    e4_checkpoint_sha256: str,
    config_snapshot: Mapping[str, Any],
    metrics: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "format_version": 1,
        "stage": "3D-A",
        "epoch": int(epoch),
        "e4_checkpoint": str(e4_checkpoint),
        "e4_checkpoint_sha256": str(e4_checkpoint_sha256),
        "support_head": support_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config_snapshot": dict(config_snapshot),
        "metrics": dict(metrics or {}),
        "changes_branch_predictions": False,
        "feeds_path_push": False,
    }


def save_stage3d_support_checkpoint(
    path: Path,
    payload: Mapping[str, Any],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(payload), str(temporary))
    os.replace(str(temporary), str(path))
    return path


def load_stage3d_support_checkpoint(
    path: Path,
    *,
    support_head: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Any = "cpu",
) -> Dict[str, Any]:
    resolved = Path(path).resolve(strict=False)
    if not resolved.is_file():
        raise FileNotFoundError(
            "Stage 3D-A support checkpoint not found: {}".format(
                resolved))
    payload = torch.load(str(resolved), map_location=map_location)
    required = (
        "support_head",
        "optimizer",
        "e4_checkpoint",
        "e4_checkpoint_sha256",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(
            "Stage 3D-A checkpoint is missing: {}".format(
                ", ".join(missing)))
    support_head.load_state_dict(payload["support_head"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return payload
