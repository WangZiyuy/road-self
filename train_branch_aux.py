"""Train and evaluate the Stage 3C teacher-forced auxiliary branch head."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from easydict import EasyDict
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.branch_query_decoder import MultiModalBranchQueryDecoder
from model.branch_set_loss import BranchSetCriterion
from model.graph_state_encoder import GraphStateEncoder
from model.model import build_model
from model.trajectory_encoder import TrajectoryFragmentEncoder
from utils.branch_metrics import (
    BranchAveragePrecisionAccumulator,
    BranchMetricAccumulator,
)
from utils.branch_diagnostics import (
    binary_average_precision,
    branch_precision_recall_curve,
    distribution_statistics,
    oracle_k_metrics,
)
from utils.checkpoint_utils import load_checkpoint_into_model
from utils.stage3c_branch_dataset import Stage3CBranchDataset
from utils.stage3c_checkpoint import (
    build_stage3c_checkpoint_payload,
    load_stage3c_checkpoint,
    save_stage3c_checkpoint,
)
from utils.stage3c_config import load_stage3c_config
from utils.trajectory_mode import TRAJ_MODE_NONE, resolve_trajectory_mode


MODALITY_FULL = "full"
MODALITY_NO_TRAJECTORY = "no_trajectory"
MODALITY_TRAJECTORY_GRAPH = "trajectory_graph"
VALID_MODALITIES = (
    MODALITY_FULL,
    MODALITY_NO_TRAJECTORY,
    MODALITY_TRAJECTORY_GRAPH,
)


def _load_config(path: Path) -> EasyDict:
    cfg = load_stage3c_config(path)
    if resolve_trajectory_mode(cfg) != TRAJ_MODE_NONE:
        raise ValueError(
            "Stage 3C requires the image-only RPNet configuration")
    if "STAGE3C" not in cfg:
        raise ValueError("config must define STAGE3C")
    return cfg


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while True:
            block = input_file.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _load_frozen_rpnet(
    cfg: EasyDict,
    image_checkpoint: Path,
    device: torch.device,
):
    image_checkpoint = image_checkpoint.resolve(strict=False)
    if not image_checkpoint.is_file():
        raise FileNotFoundError(
            "Stage 3C requires a real image-only checkpoint; not found: "
            "{}".format(image_checkpoint))
    rpnet = build_model(
        num_targets=int(cfg.TRAIN.NUM_TARGETS),
        backbone_pretrained=False,
        enable_trajectory_modules=False,
    )
    payload = load_checkpoint_into_model(
        rpnet,
        image_checkpoint,
        map_location="cpu",
        strict=True,
    )
    rpnet.to(device=device).eval().requires_grad_(False)
    return rpnet, payload


def _build_auxiliary_modules(
    cfg: EasyDict,
    device: torch.device,
):
    model_cfg = cfg.STAGE3C.MODEL
    hidden_dim = int(model_cfg.HIDDEN_DIM)
    trajectory_encoder = TrajectoryFragmentEncoder(
        hidden_dim=hidden_dim,
        num_heads=int(model_cfg.NUM_HEADS),
        num_layers=int(model_cfg.TRAJECTORY_LAYERS),
        dropout=float(model_cfg.DROPOUT),
    ).to(device=device)
    graph_state_encoder = GraphStateEncoder(
        hidden_dim=hidden_dim).to(device=device)
    branch_decoder = MultiModalBranchQueryDecoder(
        image_channels=128,
        trajectory_dim=hidden_dim,
        hidden_dim=hidden_dim,
        num_queries=int(model_cfg.NUM_QUERIES),
        num_heads=int(model_cfg.NUM_HEADS),
        image_pool_size=int(model_cfg.IMAGE_POOL_SIZE),
        dropout=float(model_cfg.DROPOUT),
        query_self_attention_layers=int(
            model_cfg.get("QUERY_SELF_ATTENTION_LAYERS", 0)),
    ).to(device=device)
    return trajectory_encoder, graph_state_encoder, branch_decoder


def _build_branch_criterion(cfg: EasyDict) -> BranchSetCriterion:
    loss_cfg = cfg.STAGE3C.get("LOSS", EasyDict())
    matching_cfg = cfg.STAGE3C.get("MATCHING", EasyDict())
    return BranchSetCriterion(
        existence_weight=float(
            loss_cfg.get("EXISTENCE_WEIGHT", 1.0)),
        endpoint_weight=float(
            loss_cfg.get("ENDPOINT_WEIGHT", 1.0)),
        direction_weight=float(
            loss_cfg.get("DIRECTION_WEIGHT", 1.0)),
        endpoint_cost_weight=float(
            matching_cfg.get("ENDPOINT_COST_WEIGHT", 1.0)),
        direction_cost_weight=float(
            matching_cfg.get("DIRECTION_COST_WEIGHT", 1.0)),
        match_cost_exist_weight=float(
            matching_cfg.get("EXISTENCE_COST_WEIGHT", 0.0)),
        exist_no_object_coef=float(
            loss_cfg.get("EXIST_NO_OBJECT_COEF", 1.0)),
        debug_cost_statistics=bool(
            matching_cfg.get("DEBUG_COST_STATISTICS", False)),
    )


def _trainable_parameters(modules: Sequence[torch.nn.Module]):
    return [
        parameter
        for module in modules
        for parameter in module.parameters()
        if parameter.requires_grad
    ]


def _build_optimizer(
    cfg: EasyDict,
    modules: Sequence[torch.nn.Module],
) -> torch.optim.Optimizer:
    optimizer_cfg = cfg.STAGE3C.OPTIMIZER
    return torch.optim.Adam(
        _trainable_parameters(modules),
        lr=float(optimizer_cfg.LEARNING_RATE),
        weight_decay=float(optimizer_cfg.WEIGHT_DECAY),
    )


def _move_nested(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {
            key: _move_nested(item, device)
            for key, item in value.items()
        }
    return value


def _extract_stage_fuse(
    rpnet: torch.nn.Module,
    aerial_image: torch.Tensor,
) -> torch.Tensor:
    feature_maps = {}
    with torch.no_grad():
        rpnet._forward_origin_backbone(aerial_image, feature_maps)
    stage_fuse = feature_maps["stage_fuse"]
    if stage_fuse.shape[1] != 128:
        raise RuntimeError(
            "unexpected RPNet stage_fuse shape: {}".format(
                tuple(stage_fuse.shape)))
    return stage_fuse


def _precompute_stage_fuse_cache(
    *,
    rpnet: torch.nn.Module,
    dataset: Stage3CBranchDataset,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Compute frozen RPNet features once without changing their precision."""

    if batch_size <= 0:
        raise ValueError("feature cache batch_size must be positive")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    cache = None
    started_at = time.perf_counter()
    for cpu_batch in loader:
        indices = cpu_batch["metadata"]["dataset_index"].to(
            dtype=torch.long)
        aerial_image = cpu_batch["aerial_image"].to(
            device=device, non_blocking=True)
        stage_fuse = _extract_stage_fuse(
            rpnet, aerial_image).detach().cpu()
        if not bool(torch.isfinite(stage_fuse).all()):
            raise RuntimeError(
                "non-finite frozen RPNet feature during precomputation")
        if cache is None:
            cache = torch.empty(
                (len(dataset),) + tuple(stage_fuse.shape[1:]),
                dtype=stage_fuse.dtype,
                device="cpu",
            )
        cache.index_copy_(0, indices, stage_fuse)
    if cache is None:
        cache = torch.empty(
            (0, 128, 64, 64), dtype=torch.float32)
    return cache, {
        "sample_count": len(dataset),
        "shape": list(cache.shape),
        "dtype": str(cache.dtype),
        "size_bytes": int(cache.numel() * cache.element_size()),
        "elapsed_seconds": float(time.perf_counter() - started_at),
        "storage": "volatile_cpu_memory",
    }


def _stage_fuse_for_batch(
    *,
    rpnet: torch.nn.Module,
    batch: Dict[str, Any],
    cache: Optional[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    if cache is None:
        return _extract_stage_fuse(
            rpnet, batch["aerial_image"])
    indices = batch["metadata"]["dataset_index"].detach().cpu().to(
        dtype=torch.long)
    return cache.index_select(0, indices).to(
        device=device, non_blocking=True)


def _set_module_mode(
    modules: Sequence[torch.nn.Module],
    training: bool,
) -> None:
    for module in modules:
        module.train(training)


def _forward_auxiliary(
    *,
    modules: Sequence[torch.nn.Module],
    batch: Dict[str, Any],
    stage_fuse: torch.Tensor,
    modality: str,
    trajectory_dropout: float = 0.0,
    return_attention: bool = False,
    return_debug_states: bool = False,
) -> Dict[str, torch.Tensor]:
    if modality not in VALID_MODALITIES:
        raise ValueError("unknown modality {!r}".format(modality))
    trajectory_encoder, graph_state_encoder, branch_decoder = modules
    trajectory_batch = dict(batch["trajectory_batch"])
    fragment_mask = trajectory_batch["fragment_mask"].to(
        dtype=torch.bool)
    if modality == MODALITY_NO_TRAJECTORY:
        fragment_mask = torch.zeros_like(fragment_mask)
    elif trajectory_dropout > 0.0:
        if not 0.0 <= trajectory_dropout <= 1.0:
            raise ValueError("trajectory_dropout must be in [0, 1]")
        dropped = (
            torch.rand(
                fragment_mask.shape[0],
                device=fragment_mask.device,
            )
            < trajectory_dropout
        )
        fragment_mask = fragment_mask & ~dropped.unsqueeze(1)
    trajectory_batch["fragment_mask"] = fragment_mask
    trajectory_output = trajectory_encoder(trajectory_batch)
    state_token = graph_state_encoder(batch["graph_state"])
    image_available = None
    if modality == MODALITY_TRAJECTORY_GRAPH:
        image_available = torch.zeros(
            stage_fuse.shape[0],
            dtype=torch.bool,
            device=stage_fuse.device,
        )
    return branch_decoder(
        stage_fuse=stage_fuse,
        state_token=state_token,
        fragment_tokens=trajectory_output["fragment_tokens"],
        fragment_mask=trajectory_output["fragment_mask"],
        walked_path=batch["walked_path"],
        image_available=image_available,
        return_attention=return_attention,
        return_debug_states=return_debug_states,
    )


def _metric_accumulator(
    cfg: EasyDict,
    *,
    existence_threshold: Optional[float] = None,
) -> BranchMetricAccumulator:
    evaluation = cfg.STAGE3C.EVALUATION
    return BranchMetricAccumulator(
        window_size=float(cfg.TRAIN.WINDOW_SIZE),
        existence_threshold=(
            float(evaluation.EXISTENCE_THRESHOLD)
            if existence_threshold is None
            else float(existence_threshold)
        ),
        endpoint_match_threshold_pixels=float(
            evaluation.ENDPOINT_MATCH_THRESHOLD_PIXELS),
        direction_match_threshold_degrees=float(
            evaluation.DIRECTION_MATCH_THRESHOLD_DEGREES),
        duplicate_endpoint_threshold_pixels=float(
            evaluation.DUPLICATE_ENDPOINT_THRESHOLD_PIXELS),
        duplicate_direction_threshold_degrees=float(
            evaluation.DUPLICATE_DIRECTION_THRESHOLD_DEGREES),
    )


def _branch_ap_accumulator(
    cfg: EasyDict,
) -> BranchAveragePrecisionAccumulator:
    evaluation = cfg.STAGE3C.EVALUATION
    return BranchAveragePrecisionAccumulator(
        window_size=float(cfg.TRAIN.WINDOW_SIZE),
        endpoint_match_threshold_pixels=float(
            evaluation.ENDPOINT_MATCH_THRESHOLD_PIXELS),
        direction_match_threshold_degrees=float(
            evaluation.DIRECTION_MATCH_THRESHOLD_DEGREES),
    )


def _loss_values(losses: Mapping[str, Any]) -> Dict[str, float]:
    return {
        "total": float(losses["loss"].detach().cpu()),
        "existence": float(
            losses["existence_loss"].detach().cpu()),
        "endpoint": float(losses["endpoint_loss"].detach().cpu()),
        "direction": float(
            losses["direction_loss"].detach().cpu()),
    }


def _evaluate_loader(
    *,
    rpnet: torch.nn.Module,
    modules: Sequence[torch.nn.Module],
    criterion: BranchSetCriterion,
    loader: DataLoader,
    cfg: EasyDict,
    device: torch.device,
    modalities: Sequence[str],
    stage_fuse_cache: Optional[torch.Tensor] = None,
    existence_thresholds: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    _set_module_mode(modules, False)
    accumulators = {
        modality: _metric_accumulator(cfg)
        for modality in modalities
    }
    ap_accumulators = {
        modality: _branch_ap_accumulator(cfg)
        for modality in modalities
    }
    sweep_thresholds = tuple(
        sorted(set(float(value) for value in (
            existence_thresholds or ()))))
    if any(
            value < 0.0 or value > 1.0
            for value in sweep_thresholds):
        raise ValueError(
            "existence threshold sweep values must be in [0, 1]")
    sweep_accumulators = {
        modality: {
            threshold: _metric_accumulator(
                cfg, existence_threshold=threshold)
            for threshold in sweep_thresholds
        }
        for modality in modalities
    }
    loss_sums = {
        modality: {
            "total": 0.0,
            "existence": 0.0,
            "endpoint": 0.0,
            "direction": 0.0,
        }
        for modality in modalities
    }
    sample_counts = {modality: 0 for modality in modalities}
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
            batch_size = int(stage_fuse.shape[0])
            for modality in modalities:
                predictions = _forward_auxiliary(
                    modules=modules,
                    batch=batch,
                    stage_fuse=stage_fuse,
                    modality=modality,
                )
                losses = criterion(
                    predictions, batch["branch_targets"])
                values = _loss_values(losses)
                for key, value in values.items():
                    loss_sums[modality][key] += value * batch_size
                sample_counts[modality] += batch_size
                accumulators[modality].update(
                    predictions, batch["branch_targets"])
                ap_accumulators[modality].update(
                    predictions, batch["branch_targets"])
                for accumulator in sweep_accumulators[
                        modality].values():
                    accumulator.update(
                        predictions, batch["branch_targets"])
    elapsed_seconds = time.perf_counter() - started_at
    results = {}
    for modality in modalities:
        metrics = accumulators[modality].compute()
        branch_ap = ap_accumulators[modality].compute()
        metrics["branch_ap"] = float(
            branch_ap["average_precision"])
        results[modality] = {
            "loss": {
                key: value / max(sample_counts[modality], 1)
                for key, value in loss_sums[modality].items()
            },
            "metrics": metrics,
            "threshold_sweep": {
                "{:.6f}".format(threshold): accumulator.compute()
                for threshold, accumulator in sweep_accumulators[
                    modality].items()
            },
            "sample_count": sample_counts[modality],
            "elapsed_seconds": float(elapsed_seconds),
        }
    return results


def _evaluate_sanity_batch(
    *,
    modules: Sequence[torch.nn.Module],
    criterion: BranchSetCriterion,
    batch: Dict[str, Any],
    stage_fuse: torch.Tensor,
    cfg: EasyDict,
) -> Dict[str, Any]:
    _set_module_mode(modules, False)
    with torch.no_grad():
        predictions = _forward_auxiliary(
            modules=modules,
            batch=batch,
            stage_fuse=stage_fuse,
            modality=MODALITY_FULL,
        )
        losses = criterion(predictions, batch["branch_targets"])
    oracle = _metric_accumulator(cfg, existence_threshold=0.0)
    oracle.update(predictions, batch["branch_targets"])
    thresholded = _metric_accumulator(cfg)
    thresholded.update(predictions, batch["branch_targets"])
    probabilities = torch.sigmoid(
        predictions["branch_exist_logits"]).detach().cpu().numpy()
    existence_targets = losses[
        "existence_targets"].detach().cpu().numpy().astype(bool)
    evaluation_cfg = cfg.STAGE3C.EVALUATION
    oracle_k = oracle_k_metrics(
        probabilities,
        predictions["branch_offsets_norm"].detach().cpu().numpy(),
        predictions["branch_directions"].detach().cpu().numpy(),
        batch["branch_targets"][
            "branch_offsets_norm"].detach().cpu().numpy(),
        batch["branch_targets"][
            "branch_directions"].detach().cpu().numpy(),
        batch["branch_targets"][
            "branch_mask"].detach().cpu().numpy(),
        window_size=float(cfg.TRAIN.WINDOW_SIZE),
        endpoint_threshold_pixels=float(
            evaluation_cfg.ENDPOINT_MATCH_THRESHOLD_PIXELS),
        direction_threshold_degrees=float(
            evaluation_cfg.DIRECTION_MATCH_THRESHOLD_DEGREES),
        duplicate_endpoint_threshold_pixels=float(
            evaluation_cfg.DUPLICATE_ENDPOINT_THRESHOLD_PIXELS),
        duplicate_direction_threshold_degrees=float(
            evaluation_cfg.DUPLICATE_DIRECTION_THRESHOLD_DEGREES),
    )
    # Selected indices are diagnostic implementation detail, not JSON data.
    oracle_k.pop("selected_query_indices")
    matched_probability = probabilities[existence_targets]
    unmatched_probability = probabilities[~existence_targets]
    branch_pr = branch_precision_recall_curve(
        probabilities,
        predictions["branch_offsets_norm"].detach().cpu().numpy(),
        predictions["branch_directions"].detach().cpu().numpy(),
        batch["branch_targets"][
            "branch_offsets_norm"].detach().cpu().numpy(),
        batch["branch_targets"][
            "branch_directions"].detach().cpu().numpy(),
        batch["branch_targets"][
            "branch_mask"].detach().cpu().numpy(),
        window_size=float(cfg.TRAIN.WINDOW_SIZE),
        endpoint_threshold_pixels=float(
            evaluation_cfg.ENDPOINT_MATCH_THRESHOLD_PIXELS),
        direction_threshold_degrees=float(
            evaluation_cfg.DIRECTION_MATCH_THRESHOLD_DEGREES),
    )
    return {
        "loss": _loss_values(losses),
        "oracle_geometry": oracle.compute(),
        "thresholded_metrics": thresholded.compute(),
        "slot_ap": binary_average_precision(
            probabilities, existence_targets),
        "branch_ap": float(branch_pr["average_precision"]),
        "matched_probability": distribution_statistics(
            matched_probability),
        "unmatched_probability": distribution_statistics(
            unmatched_probability),
        "oracle_k": oracle_k,
    }


def _configured_threshold_sweep(cfg: EasyDict) -> Tuple[float, ...]:
    values = cfg.STAGE3C.EVALUATION.get(
        "THRESHOLD_SWEEP", ())
    return tuple(float(value) for value in values)


def _relative_reduction(
    initial: Optional[float],
    final: Optional[float],
) -> Optional[float]:
    if initial is None or final is None or initial <= 0.0:
        return None
    return float(1.0 - final / initial)


def run_overfit_sanity(
    *,
    rpnet: torch.nn.Module,
    cfg: EasyDict,
    dataset: Stage3CBranchDataset,
    device: torch.device,
    output_dir: Path,
    image_checkpoint: Path,
) -> Dict[str, Any]:
    sanity_cfg = cfg.STAGE3C.SANITY
    sample_count = min(int(sanity_cfg.SAMPLE_COUNT), len(dataset))
    if sample_count != int(sanity_cfg.SAMPLE_COUNT):
        raise RuntimeError(
            "sanity dataset has only {} samples; {} required".format(
                sample_count, int(sanity_cfg.SAMPLE_COUNT)))
    minimum_count = int(sanity_cfg.get(
        "GT_BRANCH_COUNT_MIN", 0))
    maximum_count = int(sanity_cfg.get(
        "GT_BRANCH_COUNT_MAX",
        cfg.STAGE3C.DATASET.MAX_BRANCHES,
    ))
    eligible_indices = []
    for dataset_index in range(len(dataset)):
        count = int(dataset[dataset_index][
            "branch_targets"]["branch_count"])
        if minimum_count <= count <= maximum_count:
            eligible_indices.append(dataset_index)
        if len(eligible_indices) == sample_count:
            break
    if len(eligible_indices) != sample_count:
        raise RuntimeError(
            "sanity dataset has only {} samples with GT count in "
            "[{}, {}]; {} required".format(
                len(eligible_indices),
                minimum_count,
                maximum_count,
                sample_count,
            ))
    loader = DataLoader(
        Subset(dataset, eligible_indices),
        batch_size=sample_count,
        shuffle=False,
        num_workers=0,
    )
    batch = _move_nested(next(iter(loader)), device)
    stage_fuse = _extract_stage_fuse(
        rpnet, batch["aerial_image"]).detach()
    modules = _build_auxiliary_modules(cfg, device)
    optimizer = torch.optim.Adam(
        _trainable_parameters(modules),
        lr=float(sanity_cfg.LEARNING_RATE),
        weight_decay=float(
            cfg.STAGE3C.OPTIMIZER.WEIGHT_DECAY),
    )
    criterion = _build_branch_criterion(cfg)
    initial = _evaluate_sanity_batch(
        modules=modules,
        criterion=criterion,
        batch=batch,
        stage_fuse=stage_fuse,
        cfg=cfg,
    )
    curve = [{"epoch": 0, **initial}]
    max_epochs = int(sanity_cfg.MAX_EPOCHS)
    eval_every = int(sanity_cfg.EVAL_EVERY_EPOCHS)
    disable_model_dropout = bool(
        sanity_cfg.get("DISABLE_MODEL_DROPOUT", True))
    started_at = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        # This is a capacity/debugging check, not regularized training.
        # Disabling stochastic dropout makes failure reproducible and avoids
        # confusing regularization with an inability to fit 32 fixed states.
        _set_module_mode(modules, not disable_model_dropout)
        optimizer.zero_grad(set_to_none=True)
        predictions = _forward_auxiliary(
            modules=modules,
            batch=batch,
            stage_fuse=stage_fuse,
            modality=MODALITY_FULL,
            trajectory_dropout=0.0,
        )
        losses = criterion(predictions, batch["branch_targets"])
        if not bool(torch.isfinite(losses["loss"])):
            raise RuntimeError(
                "non-finite loss during 32-sample sanity check")
        losses["loss"].backward()
        clip_grad_norm_(
            _trainable_parameters(modules),
            float(cfg.STAGE3C.TRAINING.GRADIENT_CLIP_NORM),
        )
        optimizer.step()
        if epoch % eval_every == 0 or epoch == max_epochs:
            evaluation = _evaluate_sanity_batch(
                modules=modules,
                criterion=criterion,
                batch=batch,
                stage_fuse=stage_fuse,
                cfg=cfg,
            )
            curve.append({"epoch": epoch, **evaluation})
            print(
                "sanity epoch {}/{} loss={:.6f} endpoint_px={} "
                "direction_deg={}".format(
                    epoch,
                    max_epochs,
                    evaluation["loss"]["total"],
                    evaluation["oracle_geometry"][
                        "endpoint_error_mean_pixels"],
                    evaluation["oracle_geometry"][
                        "direction_error_mean_degrees"],
                ),
                flush=True,
            )

    final = curve[-1]
    total_reduction = _relative_reduction(
        initial["loss"]["total"],
        final["loss"]["total"],
    )
    endpoint_reduction = _relative_reduction(
        initial["oracle_geometry"]["endpoint_error_mean_pixels"],
        final["oracle_geometry"]["endpoint_error_mean_pixels"],
    )
    direction_reduction = _relative_reduction(
        initial["oracle_geometry"]["direction_error_mean_degrees"],
        final["oracle_geometry"]["direction_error_mean_degrees"],
    )
    passed = (
        total_reduction is not None
        and endpoint_reduction is not None
        and direction_reduction is not None
        and total_reduction
        >= float(sanity_cfg.MIN_TOTAL_LOSS_REDUCTION)
        and endpoint_reduction
        >= float(sanity_cfg.MIN_ENDPOINT_ERROR_REDUCTION)
        and direction_reduction
        >= float(sanity_cfg.MIN_DIRECTION_ERROR_REDUCTION)
    )
    report = {
        "sample_count": sample_count,
        "max_epochs": max_epochs,
        "model_dropout_disabled": disable_model_dropout,
        "selected_dataset_indices": eligible_indices,
        "gt_branch_count_range": [minimum_count, maximum_count],
        "passed": bool(passed),
        "thresholds": {
            "minimum_total_loss_reduction": float(
                sanity_cfg.MIN_TOTAL_LOSS_REDUCTION),
            "minimum_endpoint_error_reduction": float(
                sanity_cfg.MIN_ENDPOINT_ERROR_REDUCTION),
            "minimum_direction_error_reduction": float(
                sanity_cfg.MIN_DIRECTION_ERROR_REDUCTION),
        },
        "reductions": {
            "total_loss": total_reduction,
            "endpoint_error": endpoint_reduction,
            "direction_error": direction_reduction,
        },
        "initial": initial,
        "final": final,
        "curve": curve,
        "elapsed_seconds": float(time.perf_counter() - started_at),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "sanity_overfit.json").open(
            "w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    _save_sanity_curve_plot(
        curve, output_dir / "sanity_overfit_curve.png")
    payload = build_stage3c_checkpoint_payload(
        trajectory_encoder=modules[0],
        graph_state_encoder=modules[1],
        branch_decoder=modules[2],
        optimizer=optimizer,
        epoch=max_epochs,
        image_checkpoint=str(image_checkpoint),
        config_snapshot=_plain(cfg),
        metrics=report,
    )
    save_stage3c_checkpoint(
        output_dir / "sanity_overfit.pth.tar", payload)
    return report


def _train_one_epoch(
    *,
    rpnet: torch.nn.Module,
    modules: Sequence[torch.nn.Module],
    criterion: BranchSetCriterion,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    cfg: EasyDict,
    device: torch.device,
    stage_fuse_cache: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    _set_module_mode(modules, True)
    totals = {
        "total": 0.0,
        "existence": 0.0,
        "endpoint": 0.0,
        "direction": 0.0,
    }
    sample_count = 0
    trajectory_dropout = float(
        cfg.STAGE3C.TRAINING.TRAJECTORY_MODALITY_DROPOUT)
    for cpu_batch in loader:
        batch = _move_nested(cpu_batch, device)
        stage_fuse = _stage_fuse_for_batch(
            rpnet=rpnet,
            batch=batch,
            cache=stage_fuse_cache,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        predictions = _forward_auxiliary(
            modules=modules,
            batch=batch,
            stage_fuse=stage_fuse,
            modality=MODALITY_FULL,
            trajectory_dropout=trajectory_dropout,
        )
        losses = criterion(predictions, batch["branch_targets"])
        if not bool(torch.isfinite(losses["loss"])):
            raise RuntimeError("non-finite Stage 3C training loss")
        losses["loss"].backward()
        clip_grad_norm_(
            _trainable_parameters(modules),
            float(cfg.STAGE3C.TRAINING.GRADIENT_CLIP_NORM),
        )
        optimizer.step()
        batch_size = int(stage_fuse.shape[0])
        values = _loss_values(losses)
        for key, value in values.items():
            totals[key] += value * batch_size
        sample_count += batch_size
    return {
        key: value / max(sample_count, 1)
        for key, value in totals.items()
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def _save_curve_plot(
    curve: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not curve:
        return
    epochs = [int(record["epoch"]) for record in curve]
    train_loss = [
        float(record["train_loss"]["total"]) for record in curve
    ]
    val_loss = [
        float(record["validation"]["loss"]["total"])
        for record in curve
    ]
    no_trajectory_val_loss = [
        float(record["validation_no_trajectory"]["loss"]["total"])
        for record in curve
    ]
    metrics = [
        record["validation"]["metrics"] for record in curve
    ]
    no_trajectory_metrics = [
        record["validation_no_trajectory"]["metrics"]
        for record in curve
    ]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(epochs, train_loss, label="train")
    axes[0, 0].plot(epochs, val_loss, label="full validation")
    axes[0, 0].plot(
        epochs,
        no_trajectory_val_loss,
        label="no-trajectory validation",
    )
    axes[0, 0].set_title("Total loss")
    axes[0, 0].legend()
    for key, label in (
            ("precision", "precision"),
            ("recall", "recall"),
            ("f1", "F1")):
        axes[0, 1].plot(
            epochs,
            [float(metric[key]) for metric in metrics],
            label=label,
        )
    axes[0, 1].plot(
        epochs,
        [float(metric["f1"]) for metric in no_trajectory_metrics],
        label="no-trajectory F1",
        linestyle="--",
    )
    axes[0, 1].set_title("Branch detection")
    axes[0, 1].legend()
    axes[1, 0].plot(
        epochs,
        [
            metric["endpoint_error_mean_pixels"]
            for metric in metrics
        ],
    )
    axes[1, 0].set_title("Endpoint error (pixels)")
    axes[1, 1].plot(
        epochs,
        [
            metric["direction_error_mean_degrees"]
            for metric in metrics
        ],
    )
    axes[1, 1].set_title("Direction error (degrees)")
    for axis in axes.flat:
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(figure)


def _save_sanity_curve_plot(
    curve: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(record["epoch"]) for record in curve]
    total_loss = [
        float(record["loss"]["total"]) for record in curve
    ]
    endpoint = [
        record["oracle_geometry"]["endpoint_error_mean_pixels"]
        for record in curve
    ]
    direction = [
        record["oracle_geometry"]["direction_error_mean_degrees"]
        for record in curve
    ]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, values, title in (
            (axes[0], total_loss, "Total loss"),
            (axes[1], endpoint, "Endpoint error (pixels)"),
            (axes[2], direction, "Direction error (degrees)")):
        axis.plot(epochs, values)
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(figure)


def run_formal_training(
    *,
    rpnet: torch.nn.Module,
    cfg: EasyDict,
    train_dataset: Stage3CBranchDataset,
    val_dataset: Stage3CBranchDataset,
    device: torch.device,
    output_dir: Path,
    image_checkpoint: Path,
    resume: Optional[Path],
) -> Dict[str, Any]:
    from torch.utils.tensorboard import SummaryWriter

    training_cfg = cfg.STAGE3C.TRAINING
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
    modules = _build_auxiliary_modules(cfg, device)
    optimizer = _build_optimizer(cfg, modules)
    start_epoch = 1
    selection_metric = str(cfg.STAGE3C.EVALUATION.get(
        "MODEL_SELECTION_METRIC", "f1")).lower()
    if selection_metric not in ("f1", "branch_ap"):
        raise ValueError(
            "MODEL_SELECTION_METRIC must be 'f1' or 'branch_ap'")
    best_selection_score = -1.0
    best_f1 = -1.0
    best_validation_loss = float("inf")
    if resume is not None:
        payload = load_stage3c_checkpoint(
            resume,
            trajectory_encoder=modules[0],
            graph_state_encoder=modules[1],
            branch_decoder=modules[2],
            optimizer=optimizer,
            map_location=device,
        )
        start_epoch = int(payload.get("epoch", 0)) + 1
        resume_metrics = payload.get("metrics", {})
        resume_full = resume_metrics.get(MODALITY_FULL, resume_metrics)
        if (
                isinstance(resume_full, Mapping)
                and isinstance(resume_full.get("metrics"), Mapping)
                and isinstance(resume_full.get("loss"), Mapping)):
            best_f1 = float(resume_full["metrics"].get("f1", -1.0))
            best_selection_score = float(
                resume_full["metrics"].get(
                    selection_metric, best_f1))
            best_validation_loss = float(
                resume_full["loss"].get("total", float("inf")))

    criterion = _build_branch_criterion(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    log_path = output_dir / "training_curve.jsonl"
    writer = SummaryWriter(str(output_dir / "tensorboard"))
    curve = []
    started_at = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    feature_cache_report = {
        "enabled": bool(
            training_cfg.get("PRECOMPUTE_RPNET_FEATURES", False)),
        "train": None,
        "val": None,
    }
    train_stage_fuse_cache = None
    val_stage_fuse_cache = None
    if feature_cache_report["enabled"]:
        cache_batch_size = int(training_cfg.get(
            "FEATURE_CACHE_BATCH_SIZE",
            training_cfg.VAL_BATCH_SIZE,
        ))
        train_stage_fuse_cache, train_cache_report = (
            _precompute_stage_fuse_cache(
                rpnet=rpnet,
                dataset=train_dataset,
                batch_size=cache_batch_size,
                device=device,
            )
        )
        val_stage_fuse_cache, val_cache_report = (
            _precompute_stage_fuse_cache(
                rpnet=rpnet,
                dataset=val_dataset,
                batch_size=cache_batch_size,
                device=device,
            )
        )
        feature_cache_report.update({
            "train": train_cache_report,
            "val": val_cache_report,
        })
        print(
            "cached frozen RPNet features: train {:.1f}s, val {:.1f}s, "
            "{:.2f} GiB".format(
                train_cache_report["elapsed_seconds"],
                val_cache_report["elapsed_seconds"],
                (
                    train_cache_report["size_bytes"]
                    + val_cache_report["size_bytes"]
                ) / float(1024 ** 3),
            ),
            flush=True,
        )
    try:
        for epoch in range(
                start_epoch, int(training_cfg.EPOCHS) + 1):
            epoch_start = time.perf_counter()
            train_loss = _train_one_epoch(
                rpnet=rpnet,
                modules=modules,
                criterion=criterion,
                optimizer=optimizer,
                loader=train_loader,
                cfg=cfg,
                device=device,
                stage_fuse_cache=train_stage_fuse_cache,
            )
            validation_by_modality = _evaluate_loader(
                rpnet=rpnet,
                modules=modules,
                criterion=criterion,
                loader=val_loader,
                cfg=cfg,
                device=device,
                modalities=(
                    MODALITY_FULL,
                    MODALITY_NO_TRAJECTORY,
                ),
                stage_fuse_cache=val_stage_fuse_cache,
            )
            validation = validation_by_modality[MODALITY_FULL]
            validation_no_trajectory = validation_by_modality[
                MODALITY_NO_TRAJECTORY]
            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation": validation,
                "validation_no_trajectory": validation_no_trajectory,
                "full_minus_no_trajectory_f1": float(
                    validation["metrics"]["f1"]
                    - validation_no_trajectory["metrics"]["f1"]
                ),
                "elapsed_seconds": float(
                    time.perf_counter() - epoch_start),
            }
            curve.append(epoch_record)
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(
                    epoch_record, sort_keys=True) + "\n")
            for key, value in train_loss.items():
                writer.add_scalar(
                    "train/{}".format(key), value, epoch)
            for key, value in validation["loss"].items():
                writer.add_scalar(
                    "val_full/loss_{}".format(key), value, epoch)
            for key, value in validation["metrics"].items():
                if isinstance(value, (int, float)) and value is not None:
                    writer.add_scalar(
                        "val_full/{}".format(key), value, epoch)
            for key, value in validation_no_trajectory["loss"].items():
                writer.add_scalar(
                    "val_no_trajectory/loss_{}".format(key),
                    value,
                    epoch,
                )
            for key, value in validation_no_trajectory[
                    "metrics"].items():
                if isinstance(value, (int, float)) and value is not None:
                    writer.add_scalar(
                        "val_no_trajectory/{}".format(key),
                        value,
                        epoch,
                    )
            writer.add_scalar(
                "ablation/full_minus_no_trajectory_f1",
                epoch_record["full_minus_no_trajectory_f1"],
                epoch,
            )

            payload = build_stage3c_checkpoint_payload(
                trajectory_encoder=modules[0],
                graph_state_encoder=modules[1],
                branch_decoder=modules[2],
                optimizer=optimizer,
                epoch=epoch,
                image_checkpoint=str(image_checkpoint),
                config_snapshot=_plain(cfg),
                metrics=validation_by_modality,
            )
            if (
                    epoch % int(training_cfg.SAVE_EVERY_EPOCHS) == 0
                    or epoch == int(training_cfg.EPOCHS)):
                save_stage3c_checkpoint(
                    checkpoint_dir / (
                        "stage3c_aux.epoch_{:03d}.pth.tar".format(epoch)
                    ),
                    payload,
                )
            save_stage3c_checkpoint(
                checkpoint_dir / "stage3c_aux.latest.pth.tar",
                payload,
            )
            f1 = float(validation["metrics"]["f1"])
            selection_score = float(
                validation["metrics"][selection_metric])
            validation_loss = float(validation["loss"]["total"])
            if (
                    selection_score > best_selection_score
                    or (
                        selection_score == best_selection_score
                        and validation_loss < best_validation_loss
                    )):
                best_selection_score = selection_score
                best_f1 = f1
                best_validation_loss = validation_loss
                save_stage3c_checkpoint(
                    checkpoint_dir / "stage3c_aux.best.pth.tar",
                    payload,
                )
            print(
                "epoch {}/{} train={:.6f} val={:.6f} "
                "P={:.4f} R={:.4f} F1={:.4f} no_traj_F1={:.4f} "
                "delta={:+.4f}".format(
                    epoch,
                    int(training_cfg.EPOCHS),
                    train_loss["total"],
                    validation["loss"]["total"],
                    validation["metrics"]["precision"],
                    validation["metrics"]["recall"],
                    validation["metrics"]["f1"],
                    validation_no_trajectory["metrics"]["f1"],
                    epoch_record["full_minus_no_trajectory_f1"],
                ),
                flush=True,
            )
    finally:
        writer.close()

    best_checkpoint_path = (
        checkpoint_dir / "stage3c_aux.best.pth.tar")
    best_payload = load_stage3c_checkpoint(
        best_checkpoint_path,
        trajectory_encoder=modules[0],
        graph_state_encoder=modules[1],
        branch_decoder=modules[2],
        optimizer=optimizer,
        map_location=device,
    )
    final_evaluation = _evaluate_loader(
        rpnet=rpnet,
        modules=modules,
        criterion=criterion,
        loader=val_loader,
        cfg=cfg,
        device=device,
        modalities=VALID_MODALITIES,
        stage_fuse_cache=val_stage_fuse_cache,
        existence_thresholds=_configured_threshold_sweep(cfg),
    )
    report = {
        "train_sample_count": len(train_dataset),
        "val_sample_count": len(val_dataset),
        "image_checkpoint": str(image_checkpoint),
        "epochs": int(training_cfg.EPOCHS),
        "curve": curve,
        "final_ablation": final_evaluation,
        "best_full_f1": best_f1,
        "best_selection_score": best_selection_score,
        "best_full_validation_loss": best_validation_loss,
        "best_epoch": int(best_payload["epoch"]),
        "model_selection": (
            "validation full-modality {}".format(
                selection_metric)),
        "frozen_rpnet_feature_cache": feature_cache_report,
        "elapsed_seconds": float(time.perf_counter() - started_at),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        "latest_checkpoint": str(
            (checkpoint_dir / "stage3c_aux.latest.pth.tar").resolve()),
        "best_checkpoint": str(best_checkpoint_path.resolve()),
    }
    _save_curve_plot(
        curve, output_dir / "training_curves.png")
    _write_json(output_dir / "training_report.json", report)
    return report


def evaluate_checkpoint(
    *,
    rpnet: torch.nn.Module,
    cfg: EasyDict,
    dataset: Stage3CBranchDataset,
    device: torch.device,
    checkpoint: Path,
) -> Dict[str, Any]:
    modules = _build_auxiliary_modules(cfg, device)
    optimizer = _build_optimizer(cfg, modules)
    payload = load_stage3c_checkpoint(
        checkpoint,
        trajectory_encoder=modules[0],
        graph_state_encoder=modules[1],
        branch_decoder=modules[2],
        optimizer=optimizer,
        map_location=device,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.STAGE3C.TRAINING.VAL_BATCH_SIZE),
        shuffle=False,
        num_workers=int(cfg.STAGE3C.TRAINING.NUM_WORKERS),
        pin_memory=device.type == "cuda",
    )
    result = _evaluate_loader(
        rpnet=rpnet,
        modules=modules,
        criterion=_build_branch_criterion(cfg),
        loader=loader,
        cfg=cfg,
        device=device,
        modalities=VALID_MODALITIES,
        existence_thresholds=_configured_threshold_sweep(cfg),
    )
    return {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "ablation": result,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the Stage 3C teacher-forced auxiliary branch head."))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage3c_branch_aux.yml"),
    )
    parser.add_argument(
        "--mode",
        choices=("sanity", "train", "evaluate"),
        default="train",
    )
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--image-checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = _load_config(args.config)
    seed = int(cfg.STAGE3C.SEED)
    _set_seed(seed)
    device = _resolve_device(
        args.device or str(cfg.STAGE3C.DEVICE))
    dataset_dir = (
        args.dataset_dir
        if args.dataset_dir is not None
        else Path(cfg.STAGE3C.DATASET_DIR)
    )
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else Path(cfg.STAGE3C.OUTPUT_DIR)
    )
    image_checkpoint = (
        args.image_checkpoint
        if args.image_checkpoint is not None
        else Path(cfg.STAGE3C.IMAGE_CHECKPOINT)
    )
    rpnet, image_payload = _load_frozen_rpnet(
        cfg, image_checkpoint, device)
    train_dataset = Stage3CBranchDataset(
        dataset_dir, "train", preload=True)
    val_dataset = Stage3CBranchDataset(
        dataset_dir, "val", preload=True)
    run_metadata = {
        "config": str(args.config),
        "dataset_dir": str(dataset_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "image_checkpoint": str(image_checkpoint.resolve()),
        "image_checkpoint_sha256": _sha256(
            image_checkpoint.resolve()),
        "image_checkpoint_size_bytes": int(
            image_checkpoint.resolve().stat().st_size),
        "image_checkpoint_metadata": {
            key: image_payload.get(key)
            for key in (
                "outer_it",
                "path_it",
                "trajectory_mode",
                "model_name",
                "num_targets",
                "step_length",
                "window_size",
            )
        },
        "device": str(device),
        "seed": seed,
        "train_sample_count": len(train_dataset),
        "val_sample_count": len(val_dataset),
        "rpnet_frozen": all(
            not parameter.requires_grad
            for parameter in rpnet.parameters()),
        "branch_predictions_feed_path_push": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "run_metadata.json", run_metadata)

    if args.mode == "evaluate":
        if args.checkpoint is None:
            raise ValueError(
                "--checkpoint is required for evaluate mode")
        report = evaluate_checkpoint(
            rpnet=rpnet,
            cfg=cfg,
            dataset=val_dataset,
            device=device,
            checkpoint=args.checkpoint,
        )
        _write_json(output_dir / "evaluation_report.json", report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    sanity = run_overfit_sanity(
        rpnet=rpnet,
        cfg=cfg,
        dataset=train_dataset,
        device=device,
        output_dir=output_dir,
        image_checkpoint=image_checkpoint,
    )
    print(json.dumps({
        "sanity_passed": sanity["passed"],
        "reductions": sanity["reductions"],
        "elapsed_seconds": sanity["elapsed_seconds"],
    }, indent=2, sort_keys=True))
    if not sanity["passed"]:
        raise RuntimeError(
            "32-sample overfit sanity check failed; formal training "
            "was not started")
    if args.mode == "sanity":
        return

    # A fresh auxiliary model is used for formal training. The overfit model
    # is a gate only and is never promoted to a formal checkpoint.
    # Reset here so all E1-E4 experiments start their shared auxiliary
    # parameters from the same seed regardless of sanity-loop RNG use.
    _set_seed(seed)
    report = run_formal_training(
        rpnet=rpnet,
        cfg=cfg,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        output_dir=output_dir,
        image_checkpoint=image_checkpoint,
        resume=args.resume,
    )
    print(json.dumps({
        "train_sample_count": report["train_sample_count"],
        "val_sample_count": report["val_sample_count"],
        "best_full_f1": report["best_full_f1"],
        "final_ablation": report["final_ablation"],
        "elapsed_seconds": report["elapsed_seconds"],
        "peak_cuda_memory_bytes": report[
            "peak_cuda_memory_bytes"],
        "latest_checkpoint": report["latest_checkpoint"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
