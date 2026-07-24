"""Checkpoint lifecycle for Stage 3C auxiliary modules only."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch


def build_stage3c_checkpoint_payload(
    *,
    trajectory_encoder: torch.nn.Module,
    graph_state_encoder: torch.nn.Module,
    branch_decoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    image_checkpoint: str,
    config_snapshot: Mapping[str, Any],
    metrics: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "format_version": 1,
        "stage": "3C",
        "epoch": int(epoch),
        "image_checkpoint": str(image_checkpoint),
        "trajectory_encoder": trajectory_encoder.state_dict(),
        "graph_state_encoder": graph_state_encoder.state_dict(),
        "branch_decoder": branch_decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config_snapshot": dict(config_snapshot),
        "metrics": dict(metrics or {}),
    }


def save_stage3c_checkpoint(
    path: Path,
    payload: Mapping[str, Any],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(payload), str(temporary))
    os.replace(str(temporary), str(path))
    return path


def load_stage3c_checkpoint(
    path: Path,
    *,
    trajectory_encoder: torch.nn.Module,
    graph_state_encoder: torch.nn.Module,
    branch_decoder: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Any = "cpu",
) -> Dict[str, Any]:
    path = Path(path).resolve(strict=False)
    if not path.is_file():
        raise FileNotFoundError(
            "Stage 3C checkpoint not found: {}".format(path))
    payload = torch.load(str(path), map_location=map_location)
    required = (
        "trajectory_encoder",
        "graph_state_encoder",
        "branch_decoder",
        "optimizer",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(
            "Stage 3C checkpoint is missing: {}".format(
                ", ".join(missing)))
    trajectory_encoder.load_state_dict(
        payload["trajectory_encoder"], strict=True)
    graph_state_encoder.load_state_dict(
        payload["graph_state_encoder"], strict=True)
    branch_decoder.load_state_dict(
        payload["branch_decoder"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return payload
