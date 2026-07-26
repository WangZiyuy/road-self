"""Train and evaluate Stage 3D-C1 support-guided trajectory fusion."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from easydict import EasyDict
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.support_guided_trajectory_fusion import (  # noqa: E402
    FUSION_ORIGINAL_ATTENTION,
    FUSION_SUPPORT_AGGREGATION,
    SupportGuidedTrajectoryFusion,
    forward_branch_with_trajectory_fusion,
    resolve_trajectory_fusion_mode,
)
from model.trajectory_support_head import (  # noqa: E402
    trajectory_support_bce_loss,
)
from scripts.evaluate_stage3d_c0_support_aggregation import (  # noqa: E402
    BranchVariantAccumulator,
)
from train_branch_aux import (  # noqa: E402
    _build_auxiliary_modules,
    _build_branch_criterion,
    _load_config,
    _load_frozen_rpnet,
    _move_nested,
    _precompute_stage_fuse_cache,
    _resolve_device,
    _set_seed,
    _stage_fuse_for_batch,
)
from train_trajectory_support import (  # noqa: E402
    FrozenSupportDataset,
    _move_flat_batch,
    build_frozen_support_cache,
)
from utils.stage3c_branch_dataset import (  # noqa: E402
    Stage3CBranchDataset,
)
from utils.stage3c_checkpoint import load_stage3c_checkpoint  # noqa: E402
from utils.stage3d_c1_checkpoint import (  # noqa: E402
    build_stage3d_c1_checkpoint_payload,
    load_stage3d_c1_checkpoint,
    save_stage3d_c1_checkpoint,
)
from utils.stage3d_checkpoint import (  # noqa: E402
    load_stage3d_support_checkpoint,
)
from utils.trajectory_support_ranking import (  # noqa: E402
    TrajectorySupportRankingAccumulator,
)
from utils.trajectory_support_aggregation import (  # noqa: E402
    recompute_branch_predictions,
)
from utils.trajectory_support_targets import (  # noqa: E402
    build_trajectory_support_targets,
)


TRAINING_STAGE_A = "c1_a"
TRAINING_STAGE_B = "c1_b"


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(
            _plain(value), output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while True:
            block = input_file.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _target_parameters(cfg: EasyDict) -> Dict[str, float]:
    target = cfg.STAGE3D.SUPPORT_TARGET
    return {
        "window_size": float(cfg.TRAIN.WINDOW_SIZE),
        "step_length": float(cfg.TRAIN.STEP_LENGTH),
        "distance_sigma_pixels": float(
            target.DISTANCE_SIGMA_PIXELS),
        "axis_gamma": float(target.AXIS_GAMMA),
        "positive_threshold": float(target.POSITIVE_THRESHOLD),
        "epsilon": float(target.EPSILON),
    }


def _support_top_k(cfg: EasyDict) -> Optional[int]:
    value = cfg.STAGE3D_C1.FUSION.SUPPORT_TOP_K
    if value is None:
        return None
    value = int(value)
    if value <= 0:
        raise ValueError("SUPPORT_TOP_K must be positive or null")
    return value


def _build_fusion_module(
    *,
    cfg: EasyDict,
    branch_decoder: torch.nn.Module,
    device: torch.device,
    e4_sha256: str,
    training_stage: str,
) -> Optional[SupportGuidedTrajectoryFusion]:
    mode = resolve_trajectory_fusion_mode(cfg)
    if mode == FUSION_ORIGINAL_ATTENTION:
        return None
    model_cfg = cfg.STAGE3D_C1.MODEL
    fusion = SupportGuidedTrajectoryFusion(
        hidden_dim=int(cfg.STAGE3C.MODEL.HIDDEN_DIM),
        branch_input_dim=int(model_cfg.BRANCH_INPUT_DIM),
        projection_dim=int(model_cfg.PROJECTION_DIM),
    ).to(device=device)
    fusion.initialize_aggregation_from_decoder(branch_decoder)
    if training_stage == TRAINING_STAGE_A:
        checkpoint = Path(
            cfg.STAGE3D_C1.INITIAL_SUPPORT_CHECKPOINT
        ).resolve(strict=False)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                "non-circular support checkpoint not found: {}".format(
                    checkpoint))
        payload = load_stage3d_support_checkpoint(
            checkpoint,
            support_head=fusion.support_head,
            optimizer=None,
            map_location=device,
        )
        if payload["e4_checkpoint_sha256"] != e4_sha256:
            raise ValueError(
                "initial support checkpoint E4 SHA-256 mismatch")
        metadata = payload.get("metadata", {})
        if metadata.get("reads_trajectory_context", True):
            raise ValueError(
                "C1 requires a non-circular pre-trajectory support head")
    return fusion


def _freeze_e4_modules(
    modules: Sequence[torch.nn.Module],
) -> None:
    for module in modules:
        module.eval().requires_grad_(False)


def _configure_trainable_modules(
    *,
    training_stage: str,
    trajectory_encoder: torch.nn.Module,
    graph_state_encoder: torch.nn.Module,
    branch_decoder: torch.nn.Module,
    fusion_module: SupportGuidedTrajectoryFusion,
) -> Sequence[torch.nn.Parameter]:
    if training_stage not in (TRAINING_STAGE_A, TRAINING_STAGE_B):
        raise ValueError("training stage must be c1_a or c1_b")
    _freeze_e4_modules((
        trajectory_encoder, graph_state_encoder, branch_decoder))
    fusion_module.train().requires_grad_(True)
    if training_stage == TRAINING_STAGE_B:
        trajectory_encoder.train().requires_grad_(True)
    parameters = [
        parameter
        for module in (trajectory_encoder, fusion_module)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    if not parameters:
        raise RuntimeError("Stage 3D-C1 has no trainable parameters")
    return parameters


def _trajectory_batch_for_training(
    batch: Mapping[str, Any],
    *,
    dropout: float,
) -> Dict[str, torch.Tensor]:
    if not 0.0 <= dropout <= 1.0:
        raise ValueError("trajectory modality dropout must be in [0, 1]")
    trajectory_batch = dict(batch["trajectory_batch"])
    fragment_mask = trajectory_batch["fragment_mask"].to(
        dtype=torch.bool)
    if dropout > 0.0:
        dropped = torch.rand(
            fragment_mask.shape[0],
            device=fragment_mask.device,
        ) < dropout
        fragment_mask = fragment_mask & ~dropped.unsqueeze(1)
    trajectory_batch["fragment_mask"] = fragment_mask
    return trajectory_batch


def _forward_support_model(
    *,
    modules: Sequence[torch.nn.Module],
    fusion_module: Optional[SupportGuidedTrajectoryFusion],
    fusion_mode: str,
    trajectory_batch: Mapping[str, torch.Tensor],
    graph_state: Mapping[str, torch.Tensor],
    stage_fuse: torch.Tensor,
    walked_path: torch.Tensor,
    sample_ids: torch.Tensor,
    top_k: Optional[int],
    randomize: bool,
    random_seed: int,
    epsilon: float,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    trajectory_encoder, graph_state_encoder, branch_decoder = modules
    trajectory_output = trajectory_encoder(trajectory_batch)
    with torch.no_grad():
        state_token = graph_state_encoder(graph_state)
    predictions = forward_branch_with_trajectory_fusion(
        branch_decoder=branch_decoder,
        fusion_module=fusion_module,
        fusion_mode=fusion_mode,
        stage_fuse=stage_fuse,
        state_token=state_token,
        fragment_tokens=trajectory_output["fragment_tokens"],
        fragment_mask=trajectory_output["fragment_mask"],
        walked_path=walked_path,
        sample_ids=sample_ids,
        top_k=top_k,
        randomize_fragment_values=randomize,
        random_seed=random_seed,
        epsilon=epsilon,
        return_attention=False,
        return_debug_states=False,
    )
    return predictions, trajectory_output


def _forward_no_trajectory(
    *,
    modules: Sequence[torch.nn.Module],
    trajectory_output: Mapping[str, torch.Tensor],
    graph_state: Mapping[str, torch.Tensor],
    stage_fuse: torch.Tensor,
    walked_path: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    _, graph_state_encoder, branch_decoder = modules
    with torch.no_grad():
        state_token = graph_state_encoder(graph_state)
        return branch_decoder(
            stage_fuse=stage_fuse,
            state_token=state_token,
            fragment_tokens=trajectory_output["fragment_tokens"],
            fragment_mask=torch.zeros_like(
                trajectory_output["fragment_mask"]),
            walked_path=walked_path,
        )


def _targets_from_frozen_batch(
    batch: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return {
        "branch_offsets_norm": batch["branch_offsets_norm"],
        "branch_directions": batch["branch_directions"],
        "branch_mask": batch["branch_mask"],
        "branch_count": batch["branch_count"],
    }


def _forward_frozen_c1a(
    *,
    fusion_module: SupportGuidedTrajectoryFusion,
    branch_decoder: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    fragment_mask: torch.Tensor,
    top_k: Optional[int],
    randomize: bool,
    random_seed: int,
    epsilon: float,
) -> Dict[str, torch.Tensor]:
    """Run C1-a from cached frozen E4 states without recomputation."""

    fragment_tokens = batch["fragment_tokens"]
    normalized_fragments = branch_decoder.trajectory_norm(
        branch_decoder.trajectory_projection(fragment_tokens))
    normalized_fragments = normalized_fragments * fragment_mask.to(
        dtype=torch.bool).unsqueeze(-1).to(
            dtype=normalized_fragments.dtype)
    fusion = fusion_module(
        pre_trajectory_branch_tokens=batch[
            "pre_trajectory_branch_tokens"],
        fragment_tokens=fragment_tokens,
        normalized_fragment_tokens=normalized_fragments,
        fragment_mask=fragment_mask,
        top_k=top_k,
        randomize_fragment_values=randomize,
        sample_ids=batch["sample_ids"].to(dtype=torch.long),
        random_seed=random_seed,
        epsilon=epsilon,
    )
    predictions = recompute_branch_predictions(
        branch_decoder,
        graph_conditioned_queries=batch[
            "graph_conditioned_queries"],
        image_context=batch[
            "image_cross_attention_context"],
        trajectory_context=fusion["context"],
        graph_state_contribution=batch[
            "graph_state_contribution"],
    )
    predictions.update({
        "fragment_support_logits": fusion["support_logits"],
        "fragment_support_probabilities": fusion[
            "support_probabilities"],
        "trajectory_context": fusion["context"],
        "trajectory_selection_mask": fusion["selection_mask"],
        "trajectory_weight_sum": fusion["weight_sum"],
    })
    return predictions


def _forward_frozen_no_trajectory(
    *,
    branch_decoder: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return recompute_branch_predictions(
        branch_decoder,
        graph_conditioned_queries=batch[
            "graph_conditioned_queries"],
        image_context=batch[
            "image_cross_attention_context"],
        trajectory_context=torch.zeros_like(
            batch["graph_conditioned_queries"]),
        graph_state_contribution=batch[
            "graph_state_contribution"],
    )


def evaluate_frozen_c1a(
    *,
    cache: FrozenSupportDataset,
    fusion_module: SupportGuidedTrajectoryFusion,
    branch_decoder: torch.nn.Module,
    criterion: torch.nn.Module,
    cfg: EasyDict,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate C1-a from one deterministic frozen cache."""

    fusion_module.eval()
    branch_decoder.eval()
    loader = DataLoader(
        cache,
        batch_size=int(
            cfg.STAGE3D_C1.TRAINING.VAL_BATCH_SIZE),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    full_accumulator = BranchVariantAccumulator(cfg)
    no_trajectory_accumulator = BranchVariantAccumulator(cfg)
    support_ranking = TrajectorySupportRankingAccumulator(
        ranking_ks=tuple(
            int(value)
            for value in cfg.STAGE3D_C1.EVALUATION.RANKING_KS),
        jaccard_k=int(
            cfg.STAGE3D_C1.EVALUATION.JACCARD_K),
    )
    support_loss_sum = 0.0
    sample_count = 0
    started_at = time.perf_counter()
    with torch.no_grad():
        for cpu_batch in loader:
            batch = _move_flat_batch(cpu_batch, device)
            fragment_mask = batch[
                "fragment_mask"].to(dtype=torch.bool)
            predictions = _forward_frozen_c1a(
                fusion_module=fusion_module,
                branch_decoder=branch_decoder,
                batch=batch,
                fragment_mask=fragment_mask,
                top_k=_support_top_k(cfg),
                randomize=bool(
                    cfg.STAGE3D_C1.FUSION.
                    RANDOM_FRAGMENT_AGGREGATION),
                random_seed=int(cfg.STAGE3C.SEED),
                epsilon=float(
                    cfg.STAGE3D_C1.FUSION.EPSILON),
            )
            targets = _targets_from_frozen_batch(batch)
            losses = criterion(predictions, targets)
            full_accumulator.update(
                predictions, targets, losses["matches"])
            no_trajectory = _forward_frozen_no_trajectory(
                branch_decoder=branch_decoder, batch=batch)
            no_losses = criterion(no_trajectory, targets)
            no_trajectory_accumulator.update(
                no_trajectory, targets, no_losses["matches"])
            support_losses = trajectory_support_bce_loss(
                predictions["fragment_support_logits"],
                batch["support_targets"],
                batch["support_valid"],
                fragment_mask,
                losses["matches"],
            )
            support_ranking.update(
                scores=predictions[
                    "fragment_support_probabilities"],
                support_targets=batch["support_targets"],
                support_positive_mask=batch[
                    "support_positive_mask"],
                support_valid=batch["support_valid"],
                branch_mask=batch["branch_mask"],
                fragment_mask=fragment_mask,
                matches=losses["matches"],
                branch_count=batch["branch_count"],
                sample_ids=batch["sample_ids"].to(
                    dtype=torch.long),
            )
            batch_size = int(fragment_mask.shape[0])
            support_loss_sum += float(
                support_losses["loss"].detach().cpu()
            ) * batch_size
            sample_count += batch_size
    full = full_accumulator.compute()
    no_trajectory = no_trajectory_accumulator.compute()
    return {
        "full": full,
        "no_trajectory": no_trajectory,
        "full_minus_no_trajectory_branch_ap": float(
            full["branch_ap"] - no_trajectory["branch_ap"]),
        "support_selection": support_ranking.compute(),
        "support_loss": support_loss_sum / max(sample_count, 1),
        "sample_count": sample_count,
        "elapsed_seconds": float(
            time.perf_counter() - started_at),
        "frozen_cache": True,
    }


def evaluate_support_fusion(
    *,
    rpnet: torch.nn.Module,
    modules: Sequence[torch.nn.Module],
    fusion_module: Optional[SupportGuidedTrajectoryFusion],
    criterion: torch.nn.Module,
    loader: DataLoader,
    cfg: EasyDict,
    device: torch.device,
    stage_fuse_cache: Optional[torch.Tensor],
) -> Dict[str, Any]:
    fusion_mode = resolve_trajectory_fusion_mode(cfg)
    for module in modules:
        module.eval()
    if fusion_module is not None:
        fusion_module.eval()
    full_accumulator = BranchVariantAccumulator(cfg)
    no_trajectory_accumulator = BranchVariantAccumulator(cfg)
    support_ranking = TrajectorySupportRankingAccumulator(
        ranking_ks=tuple(
            int(value)
            for value in cfg.STAGE3D_C1.EVALUATION.RANKING_KS),
        jaccard_k=int(
            cfg.STAGE3D_C1.EVALUATION.JACCARD_K),
    )
    top_k = _support_top_k(cfg)
    randomize = bool(
        cfg.STAGE3D_C1.FUSION.RANDOM_FRAGMENT_AGGREGATION)
    epsilon = float(cfg.STAGE3D_C1.FUSION.EPSILON)
    support_loss_sum = 0.0
    sample_count = 0
    started_at = time.perf_counter()
    with torch.no_grad():
        for cpu_batch in loader:
            batch = _move_nested(cpu_batch, device)
            stage_fuse = _stage_fuse_for_batch(
                rpnet=rpnet,
                batch=batch,
                cache=stage_fuse_cache,
                device=device,
            )
            trajectory_batch = dict(batch["trajectory_batch"])
            predictions, trajectory_output = _forward_support_model(
                modules=modules,
                fusion_module=fusion_module,
                fusion_mode=fusion_mode,
                trajectory_batch=trajectory_batch,
                graph_state=batch["graph_state"],
                stage_fuse=stage_fuse,
                walked_path=batch["walked_path"],
                sample_ids=batch["metadata"]["dataset_index"].to(
                    dtype=torch.long),
                top_k=top_k,
                randomize=randomize,
                random_seed=int(cfg.STAGE3C.SEED),
                epsilon=epsilon,
            )
            targets = batch["branch_targets"]
            full_losses = criterion(predictions, targets)
            full_accumulator.update(
                predictions, targets, full_losses["matches"])
            no_trajectory = _forward_no_trajectory(
                modules=modules,
                trajectory_output=trajectory_output,
                graph_state=batch["graph_state"],
                stage_fuse=stage_fuse,
                walked_path=batch["walked_path"],
            )
            no_losses = criterion(no_trajectory, targets)
            no_trajectory_accumulator.update(
                no_trajectory, targets, no_losses["matches"])
            if fusion_mode == FUSION_SUPPORT_AGGREGATION:
                support_targets = build_trajectory_support_targets(
                    trajectory_batch, targets,
                    **_target_parameters(cfg))
                support_loss = trajectory_support_bce_loss(
                    predictions["fragment_support_logits"],
                    support_targets["support_targets"],
                    support_targets["support_valid"],
                    trajectory_output["fragment_mask"],
                    full_losses["matches"],
                )
                support_ranking.update(
                    scores=predictions[
                        "fragment_support_probabilities"],
                    support_targets=support_targets[
                        "support_targets"],
                    support_positive_mask=support_targets[
                        "support_positive_mask"],
                    support_valid=support_targets["support_valid"],
                    branch_mask=targets["branch_mask"],
                    fragment_mask=trajectory_output["fragment_mask"],
                    matches=full_losses["matches"],
                    branch_count=targets["branch_count"],
                    sample_ids=batch["metadata"][
                        "dataset_index"].to(dtype=torch.long),
                )
                support_loss_sum += float(
                    support_loss["loss"].detach().cpu()
                ) * int(stage_fuse.shape[0])
            sample_count += int(stage_fuse.shape[0])
    full = full_accumulator.compute()
    no_trajectory = no_trajectory_accumulator.compute()
    return {
        "full": full,
        "no_trajectory": no_trajectory,
        "full_minus_no_trajectory_branch_ap": float(
            full["branch_ap"] - no_trajectory["branch_ap"]),
        "support_selection": (
            support_ranking.compute()
            if fusion_mode == FUSION_SUPPORT_AGGREGATION
            else None
        ),
        "support_loss": (
            support_loss_sum / max(sample_count, 1)
            if fusion_mode == FUSION_SUPPORT_AGGREGATION
            else None
        ),
        "sample_count": sample_count,
        "elapsed_seconds": float(
            time.perf_counter() - started_at),
    }


def _train_one_epoch_frozen_c1a(
    *,
    cache: FrozenSupportDataset,
    fusion_module: SupportGuidedTrajectoryFusion,
    branch_decoder: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: EasyDict,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Train only C1-a parameters from cached frozen E4 states."""

    fusion_module.train().requires_grad_(True)
    branch_decoder.eval().requires_grad_(False)
    parameters = [
        parameter
        for parameter in fusion_module.parameters()
        if parameter.requires_grad
    ]
    loader = DataLoader(
        cache,
        batch_size=int(cfg.STAGE3D_C1.TRAINING.BATCH_SIZE),
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    totals = {
        "total": 0.0,
        "branch": 0.0,
        "support": 0.0,
        "existence": 0.0,
        "endpoint": 0.0,
        "direction": 0.0,
    }
    sample_count = 0
    training_cfg = cfg.STAGE3D_C1.TRAINING
    dropout = float(
        training_cfg.TRAJECTORY_MODALITY_DROPOUT)
    if not 0.0 <= dropout <= 1.0:
        raise ValueError(
            "trajectory modality dropout must be in [0, 1]")
    for batch_index, cpu_batch in enumerate(loader):
        batch = _move_flat_batch(cpu_batch, device)
        fragment_mask = batch[
            "fragment_mask"].to(dtype=torch.bool)
        if dropout > 0.0:
            dropped = torch.rand(
                fragment_mask.shape[0],
                device=device,
            ) < dropout
            fragment_mask = (
                fragment_mask & ~dropped.unsqueeze(1))
        optimizer.zero_grad(set_to_none=True)
        predictions = _forward_frozen_c1a(
            fusion_module=fusion_module,
            branch_decoder=branch_decoder,
            batch=batch,
            fragment_mask=fragment_mask,
            top_k=_support_top_k(cfg),
            randomize=bool(
                cfg.STAGE3D_C1.FUSION.
                RANDOM_FRAGMENT_AGGREGATION),
            random_seed=(
                int(cfg.STAGE3C.SEED)
                + epoch * 1_000_003 + batch_index),
            epsilon=float(
                cfg.STAGE3D_C1.FUSION.EPSILON),
        )
        targets = _targets_from_frozen_batch(batch)
        branch_losses = criterion(predictions, targets)
        support_losses = trajectory_support_bce_loss(
            predictions["fragment_support_logits"],
            batch["support_targets"],
            batch["support_valid"],
            fragment_mask,
            branch_losses["matches"],
        )
        total_loss = (
            branch_losses["loss"]
            + float(training_cfg.SUPPORT_LOSS_WEIGHT)
            * support_losses["loss"]
        )
        if not bool(torch.isfinite(total_loss)):
            raise RuntimeError(
                "non-finite Stage 3D-C1 frozen-cache loss")
        total_loss.backward()
        clip_grad_norm_(
            parameters,
            float(training_cfg.GRADIENT_CLIP_NORM),
        )
        optimizer.step()
        batch_size = int(fragment_mask.shape[0])
        values = {
            "total": float(total_loss.detach().cpu()),
            "branch": float(
                branch_losses["loss"].detach().cpu()),
            "support": float(
                support_losses["loss"].detach().cpu()),
            "existence": float(
                branch_losses[
                    "existence_loss"].detach().cpu()),
            "endpoint": float(
                branch_losses[
                    "endpoint_loss"].detach().cpu()),
            "direction": float(
                branch_losses[
                    "direction_loss"].detach().cpu()),
        }
        for key, value in values.items():
            totals[key] += value * batch_size
        sample_count += batch_size
    return {
        key: value / max(sample_count, 1)
        for key, value in totals.items()
    }


def _train_one_epoch(
    *,
    rpnet: torch.nn.Module,
    modules: Sequence[torch.nn.Module],
    fusion_module: SupportGuidedTrajectoryFusion,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    cfg: EasyDict,
    device: torch.device,
    stage_fuse_cache: Optional[torch.Tensor],
    epoch: int,
    training_stage: str,
) -> Dict[str, float]:
    trajectory_encoder, graph_state_encoder, branch_decoder = modules
    parameters = _configure_trainable_modules(
        training_stage=training_stage,
        trajectory_encoder=trajectory_encoder,
        graph_state_encoder=graph_state_encoder,
        branch_decoder=branch_decoder,
        fusion_module=fusion_module,
    )
    totals = {
        "total": 0.0,
        "branch": 0.0,
        "support": 0.0,
        "existence": 0.0,
        "endpoint": 0.0,
        "direction": 0.0,
    }
    sample_count = 0
    training_cfg = cfg.STAGE3D_C1.TRAINING
    for batch_index, cpu_batch in enumerate(loader):
        batch = _move_nested(cpu_batch, device)
        stage_fuse = _stage_fuse_for_batch(
            rpnet=rpnet,
            batch=batch,
            cache=stage_fuse_cache,
            device=device,
        )
        trajectory_batch = _trajectory_batch_for_training(
            batch,
            dropout=float(
                training_cfg.TRAJECTORY_MODALITY_DROPOUT),
        )
        optimizer.zero_grad(set_to_none=True)
        predictions, trajectory_output = _forward_support_model(
            modules=modules,
            fusion_module=fusion_module,
            fusion_mode=FUSION_SUPPORT_AGGREGATION,
            trajectory_batch=trajectory_batch,
            graph_state=batch["graph_state"],
            stage_fuse=stage_fuse,
            walked_path=batch["walked_path"],
            sample_ids=batch["metadata"]["dataset_index"].to(
                dtype=torch.long),
            top_k=_support_top_k(cfg),
            randomize=bool(
                cfg.STAGE3D_C1.FUSION.
                RANDOM_FRAGMENT_AGGREGATION),
            random_seed=(
                int(cfg.STAGE3C.SEED)
                + epoch * 1_000_003 + batch_index),
            epsilon=float(cfg.STAGE3D_C1.FUSION.EPSILON),
        )
        targets = batch["branch_targets"]
        branch_losses = criterion(predictions, targets)
        support_targets = build_trajectory_support_targets(
            trajectory_batch, targets,
            **_target_parameters(cfg))
        support_losses = trajectory_support_bce_loss(
            predictions["fragment_support_logits"],
            support_targets["support_targets"],
            support_targets["support_valid"],
            trajectory_output["fragment_mask"],
            branch_losses["matches"],
        )
        total_loss = (
            branch_losses["loss"]
            + float(training_cfg.SUPPORT_LOSS_WEIGHT)
            * support_losses["loss"]
        )
        if not bool(torch.isfinite(total_loss)):
            raise RuntimeError("non-finite Stage 3D-C1 loss")
        total_loss.backward()
        clip_grad_norm_(
            parameters,
            float(training_cfg.GRADIENT_CLIP_NORM),
        )
        optimizer.step()
        batch_size = int(stage_fuse.shape[0])
        values = {
            "total": float(total_loss.detach().cpu()),
            "branch": float(
                branch_losses["loss"].detach().cpu()),
            "support": float(
                support_losses["loss"].detach().cpu()),
            "existence": float(
                branch_losses["existence_loss"].detach().cpu()),
            "endpoint": float(
                branch_losses["endpoint_loss"].detach().cpu()),
            "direction": float(
                branch_losses["direction_loss"].detach().cpu()),
        }
        for key, value in values.items():
            totals[key] += value * batch_size
        sample_count += batch_size
    return {
        key: value / max(sample_count, 1)
        for key, value in totals.items()
    }


def _checkpoint_payload(
    *,
    fusion_module: SupportGuidedTrajectoryFusion,
    trajectory_encoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    training_stage: str,
    cfg: EasyDict,
    e4_checkpoint: Path,
    e4_sha256: str,
    metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    return build_stage3d_c1_checkpoint_payload(
        fusion_module=fusion_module,
        trajectory_encoder=trajectory_encoder,
        optimizer=optimizer,
        epoch=epoch,
        training_stage=training_stage,
        fusion_mode=resolve_trajectory_fusion_mode(cfg),
        support_top_k=_support_top_k(cfg),
        random_fragment_aggregation=bool(
            cfg.STAGE3D_C1.FUSION.RANDOM_FRAGMENT_AGGREGATION),
        e4_checkpoint=str(e4_checkpoint.resolve()),
        e4_checkpoint_sha256=e4_sha256,
        metrics=metrics,
        config=_plain(cfg),
    )


def run_training(
    *,
    rpnet: torch.nn.Module,
    modules: Sequence[torch.nn.Module],
    fusion_module: SupportGuidedTrajectoryFusion,
    train_dataset: Stage3CBranchDataset,
    val_dataset: Stage3CBranchDataset,
    cfg: EasyDict,
    device: torch.device,
    output_dir: Path,
    e4_checkpoint: Path,
    e4_sha256: str,
) -> Dict[str, Any]:
    training_cfg = cfg.STAGE3D_C1.TRAINING
    training_stage = str(training_cfg.STAGE).lower()
    parameters = _configure_trainable_modules(
        training_stage=training_stage,
        trajectory_encoder=modules[0],
        graph_state_encoder=modules[1],
        branch_decoder=modules[2],
        fusion_module=fusion_module,
    )
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training_cfg.LEARNING_RATE),
        weight_decay=float(training_cfg.WEIGHT_DECAY),
    )
    if training_stage == TRAINING_STAGE_B:
        initial_checkpoint = Path(
            cfg.STAGE3D_C1.INITIAL_C1_CHECKPOINT
        ).resolve(strict=False)
        load_stage3d_c1_checkpoint(
            initial_checkpoint,
            fusion_module=fusion_module,
            trajectory_encoder=modules[0],
            optimizer=None,
            map_location=device,
            expected_e4_sha256=e4_sha256,
        )
    criterion = _build_branch_criterion(cfg)
    train_loader = None
    val_loader = None
    train_cache = None
    val_cache = None
    frozen_train_cache = None
    frozen_val_cache = None
    if training_stage == TRAINING_STAGE_A:
        frozen_train_cache, train_report = build_frozen_support_cache(
            dataset=train_dataset,
            rpnet=rpnet,
            modules=modules,
            criterion=criterion,
            cfg=cfg,
            device=device,
        )
        frozen_val_cache, val_report = build_frozen_support_cache(
            dataset=val_dataset,
            rpnet=rpnet,
            modules=modules,
            criterion=criterion,
            cfg=cfg,
            device=device,
        )
        cache_report = {
            "enabled": True,
            "kind": "frozen_e4_states",
            "train": train_report,
            "val": val_report,
        }
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(training_cfg.BATCH_SIZE),
            shuffle=True,
            num_workers=int(training_cfg.NUM_WORKERS),
            pin_memory=device.type == "cuda",
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(training_cfg.VAL_BATCH_SIZE),
            shuffle=False,
            num_workers=int(training_cfg.NUM_WORKERS),
            pin_memory=device.type == "cuda",
        )
        cache_report = {
            "enabled": bool(
                training_cfg.PRECOMPUTE_RPNET_FEATURES),
            "kind": "rpnet_stage_fuse",
        }
        if cache_report["enabled"]:
            train_cache, train_report = _precompute_stage_fuse_cache(
                rpnet=rpnet,
                dataset=train_dataset,
                batch_size=int(
                    training_cfg.FEATURE_CACHE_BATCH_SIZE),
                device=device,
            )
            val_cache, val_report = _precompute_stage_fuse_cache(
                rpnet=rpnet,
                dataset=val_dataset,
                batch_size=int(
                    training_cfg.FEATURE_CACHE_BATCH_SIZE),
                device=device,
            )
            cache_report.update({
                "train": train_report,
                "val": val_report,
            })
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    curve = []
    best_ap = -1.0
    best_epoch = 0
    started_at = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, int(training_cfg.EPOCHS) + 1):
        if training_stage == TRAINING_STAGE_A:
            train_loss = _train_one_epoch_frozen_c1a(
                cache=frozen_train_cache,
                fusion_module=fusion_module,
                branch_decoder=modules[2],
                criterion=criterion,
                optimizer=optimizer,
                cfg=cfg,
                device=device,
                epoch=epoch,
            )
            validation = evaluate_frozen_c1a(
                cache=frozen_val_cache,
                fusion_module=fusion_module,
                branch_decoder=modules[2],
                criterion=criterion,
                cfg=cfg,
                device=device,
            )
        else:
            train_loss = _train_one_epoch(
                rpnet=rpnet,
                modules=modules,
                fusion_module=fusion_module,
                criterion=criterion,
                optimizer=optimizer,
                loader=train_loader,
                cfg=cfg,
                device=device,
                stage_fuse_cache=train_cache,
                epoch=epoch,
                training_stage=training_stage,
            )
            validation = evaluate_support_fusion(
                rpnet=rpnet,
                modules=modules,
                fusion_module=fusion_module,
                criterion=criterion,
                loader=val_loader,
                cfg=cfg,
                device=device,
                stage_fuse_cache=val_cache,
            )
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation": validation,
        }
        curve.append(record)
        with (output_dir / "training_curve.jsonl").open(
                "a", encoding="utf-8") as log_file:
            log_file.write(
                json.dumps(_plain(record), sort_keys=True) + "\n")
        payload = _checkpoint_payload(
            fusion_module=fusion_module,
            trajectory_encoder=modules[0],
            optimizer=optimizer,
            epoch=epoch,
            training_stage=training_stage,
            cfg=cfg,
            e4_checkpoint=e4_checkpoint,
            e4_sha256=e4_sha256,
            metrics=validation,
        )
        save_stage3d_c1_checkpoint(
            checkpoint_dir / "stage3d_c1.latest.pth.tar",
            payload,
        )
        branch_ap = float(validation["full"]["branch_ap"])
        if branch_ap > best_ap:
            best_ap = branch_ap
            best_epoch = epoch
            save_stage3d_c1_checkpoint(
                checkpoint_dir / "stage3d_c1.best.pth.tar",
                payload,
            )
        print(
            "C1 {} epoch {}/{} loss={:.6f} branch_ap={:.6f} "
            "delta_no_traj={:+.6f}".format(
                training_stage,
                epoch,
                int(training_cfg.EPOCHS),
                train_loss["total"],
                branch_ap,
                validation[
                    "full_minus_no_trajectory_branch_ap"],
            ),
            flush=True,
        )
    best_path = checkpoint_dir / "stage3d_c1.best.pth.tar"
    load_stage3d_c1_checkpoint(
        best_path,
        fusion_module=fusion_module,
        trajectory_encoder=modules[0],
        optimizer=None,
        map_location=device,
        expected_e4_sha256=e4_sha256,
    )
    if training_stage == TRAINING_STAGE_A:
        best_validation = evaluate_frozen_c1a(
            cache=frozen_val_cache,
            fusion_module=fusion_module,
            branch_decoder=modules[2],
            criterion=criterion,
            cfg=cfg,
            device=device,
        )
    else:
        best_validation = evaluate_support_fusion(
            rpnet=rpnet,
            modules=modules,
            fusion_module=fusion_module,
            criterion=criterion,
            loader=val_loader,
            cfg=cfg,
            device=device,
            stage_fuse_cache=val_cache,
        )
    report = {
        "schema_version": "stage3d-c1-training-v1",
        "training_stage": training_stage,
        "fusion_mode": resolve_trajectory_fusion_mode(cfg),
        "support_top_k": _support_top_k(cfg),
        "random_fragment_aggregation": bool(
            cfg.STAGE3D_C1.FUSION.RANDOM_FRAGMENT_AGGREGATION),
        "best_epoch": best_epoch,
        "best_checkpoint": str(best_path.resolve()),
        "best_validation": best_validation,
        "curve": curve,
        "feature_cache": cache_report,
        "elapsed_seconds": float(
            time.perf_counter() - started_at),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda" else 0),
        "trainable": {
            "support_head": True,
            "aggregation_projection": True,
            "trajectory_encoder":
                training_stage == TRAINING_STAGE_B,
            "graph_state_encoder": False,
            "branch_decoder": False,
            "rpnet": False,
        },
    }
    _write_json(output_dir / "training_summary.json", report)
    return report


def run_evaluation(
    *,
    rpnet: torch.nn.Module,
    modules: Sequence[torch.nn.Module],
    fusion_module: Optional[SupportGuidedTrajectoryFusion],
    val_dataset: Stage3CBranchDataset,
    cfg: EasyDict,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, Any]:
    loader = DataLoader(
        val_dataset,
        batch_size=int(
            cfg.STAGE3D_C1.TRAINING.VAL_BATCH_SIZE),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    criterion = _build_branch_criterion(cfg)
    report = evaluate_support_fusion(
        rpnet=rpnet,
        modules=modules,
        fusion_module=fusion_module,
        criterion=criterion,
        loader=loader,
        cfg=cfg,
        device=device,
        stage_fuse_cache=None,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "evaluation.json", report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("train", "evaluate"), default="train")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = _load_config(args.config)
    fusion_mode = resolve_trajectory_fusion_mode(cfg)
    training_stage = str(
        cfg.STAGE3D_C1.TRAINING.STAGE).lower()
    if training_stage not in (TRAINING_STAGE_A, TRAINING_STAGE_B):
        raise ValueError("TRAINING.STAGE must be c1_a or c1_b")
    if args.mode == "train" and (
            fusion_mode == FUSION_ORIGINAL_ATTENTION):
        raise ValueError(
            "original_attention is an evaluation-only frozen baseline")
    seed = int(cfg.STAGE3C.SEED)
    _set_seed(seed)
    device = _resolve_device(
        args.device or str(cfg.STAGE3C.DEVICE))
    output_dir = (
        args.output_dir
        or Path(cfg.STAGE3D_C1.OUTPUT_DIR))
    e4_checkpoint = Path(
        cfg.STAGE3D.E4_CHECKPOINT).resolve(strict=False)
    image_checkpoint = Path(
        cfg.STAGE3C.IMAGE_CHECKPOINT).resolve(strict=False)
    dataset_dir = Path(cfg.STAGE3C.DATASET_DIR)
    for name, path in (
            ("E4", e4_checkpoint),
            ("RPNet", image_checkpoint)):
        if not path.is_file():
            raise FileNotFoundError(
                "{} checkpoint not found: {}".format(name, path))
    e4_sha256 = _sha256(e4_checkpoint)
    rpnet, _ = _load_frozen_rpnet(
        cfg, image_checkpoint, device)
    modules = _build_auxiliary_modules(cfg, device)
    load_stage3c_checkpoint(
        e4_checkpoint,
        trajectory_encoder=modules[0],
        graph_state_encoder=modules[1],
        branch_decoder=modules[2],
        optimizer=None,
        map_location=device,
    )
    _freeze_e4_modules(modules)
    fusion_module = _build_fusion_module(
        cfg=cfg,
        branch_decoder=modules[2],
        device=device,
        e4_sha256=e4_sha256,
        training_stage=training_stage,
    )
    if args.checkpoint is not None:
        if fusion_module is None:
            raise ValueError(
                "original_attention has no C1 checkpoint")
        load_stage3d_c1_checkpoint(
            args.checkpoint,
            fusion_module=fusion_module,
            trajectory_encoder=modules[0],
            optimizer=None,
            map_location=device,
            expected_e4_sha256=e4_sha256,
        )
    train_dataset = Stage3CBranchDataset(
        dataset_dir, "train", preload=True)
    val_dataset = Stage3CBranchDataset(
        dataset_dir, "val", preload=True)
    if args.mode == "evaluate":
        report = run_evaluation(
            rpnet=rpnet,
            modules=modules,
            fusion_module=fusion_module,
            val_dataset=val_dataset,
            cfg=cfg,
            device=device,
            output_dir=output_dir,
        )
    else:
        report = run_training(
            rpnet=rpnet,
            modules=modules,
            fusion_module=fusion_module,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            cfg=cfg,
            device=device,
            output_dir=output_dir,
            e4_checkpoint=e4_checkpoint,
            e4_sha256=e4_sha256,
        )
    print(json.dumps({
        "fusion_mode": fusion_mode,
        "output_dir": str(output_dir.resolve()),
        "branch_ap": (
            report["best_validation"]["full"]["branch_ap"]
            if "best_validation" in report
            else report["full"]["branch_ap"]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
