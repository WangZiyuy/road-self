"""Train and evaluate the Stage 3E-0 trajectory evidence encoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from easydict import EasyDict
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.trajectory_evidence_encoder import (  # noqa: E402
    TRAJECTORY_MODE_EVIDENCE,
    TRAJECTORY_MODE_ORIGINAL_FRAGMENT,
    TrajectoryEvidenceEncoder,
    build_trajectory_decoder_inputs,
    resolve_evidence_aggregation_mode,
    resolve_trajectory_evidence_mode,
    trajectory_context_from_tokens,
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
    _resolve_device,
    _set_seed,
    _stage_fuse_for_batch,
)
from utils.stage3c_branch_dataset import (  # noqa: E402
    Stage3CBranchDataset,
)
from utils.stage3c_checkpoint import load_stage3c_checkpoint  # noqa: E402
from utils.stage3e0_checkpoint import (  # noqa: E402
    build_stage3e0_checkpoint_payload,
    load_stage3e0_checkpoint,
    save_stage3e0_checkpoint,
)
from utils.trajectory_support_aggregation import (  # noqa: E402
    recompute_branch_predictions,
)


VARIANT_NO_TRAJECTORY = "image_graph"
VARIANT_ORIGINAL_FRAGMENT = "original_fragment"
VARIANT_TRAJECTORY_EVIDENCE = "trajectory_evidence"


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


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(memoryview(array)).hexdigest()


def _shared_evidence_state_sha256(
    evidence_encoder: TrajectoryEvidenceEncoder,
) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(
            evidence_encoder.state_dict().items()):
        if name == "trajectory_queries":
            continue
        digest.update(name.encode("utf-8"))
        digest.update(memoryview(
            value.detach().cpu().contiguous().numpy()))
    return digest.hexdigest()


class FrozenEvidenceDataset(Dataset):
    """In-memory tensors produced only by frozen Stage 3C modules."""

    def __init__(self, tensors: Mapping[str, torch.Tensor]) -> None:
        if not tensors:
            raise ValueError("frozen evidence tensors cannot be empty")
        lengths = {int(value.shape[0]) for value in tensors.values()}
        if len(lengths) != 1:
            raise ValueError(
                "all frozen evidence tensors need one sample count")
        self.tensors = dict(tensors)
        self.sample_count = lengths.pop()

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            key: value[index]
            for key, value in self.tensors.items()
        }


def _move_flat_batch(
    batch: Mapping[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device=device, non_blocking=True)
        for key, value in batch.items()
    }


def _freeze_modules(modules: Sequence[torch.nn.Module]) -> None:
    for module in modules:
        module.eval().requires_grad_(False)
    if any(
            parameter.requires_grad
            for module in modules
            for parameter in module.parameters()):
        raise RuntimeError("all Stage 3E-0 backbone modules must be frozen")


def _build_evidence_encoder(
    cfg: EasyDict,
    device: torch.device,
) -> TrajectoryEvidenceEncoder:
    model_cfg = cfg.STAGE3E0.MODEL
    return TrajectoryEvidenceEncoder(
        hidden_dim=int(cfg.STAGE3C.MODEL.HIDDEN_DIM),
        num_evidence_tokens=int(model_cfg.NUM_EVIDENCE_TOKENS),
        num_heads=int(model_cfg.NUM_HEADS),
        dropout=float(model_cfg.DROPOUT),
        aggregation_mode=resolve_evidence_aggregation_mode(cfg),
    ).to(device=device)


def build_frozen_evidence_cache(
    *,
    dataset: Stage3CBranchDataset,
    rpnet: torch.nn.Module,
    modules: Sequence[torch.nn.Module],
    cfg: EasyDict,
    device: torch.device,
) -> Tuple[FrozenEvidenceDataset, Dict[str, Any]]:
    """Cache only outputs of modules that remain frozen in Stage 3E-0."""

    _freeze_modules((rpnet,) + tuple(modules))
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.STAGE3E0.TRAINING.CACHE_BATCH_SIZE),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    trajectory_encoder, graph_state_encoder, branch_decoder = modules
    chunks: Dict[str, List[torch.Tensor]] = {}
    started_at = time.perf_counter()
    with torch.no_grad():
        for cpu_batch in loader:
            batch = _move_nested(cpu_batch, device)
            stage_fuse = _stage_fuse_for_batch(
                rpnet=rpnet,
                batch=batch,
                cache=None,
                device=device,
            )
            trajectory = trajectory_encoder(
                batch["trajectory_batch"])
            state_token = graph_state_encoder(batch["graph_state"])
            branch = branch_decoder(
                stage_fuse=stage_fuse,
                state_token=state_token,
                fragment_tokens=trajectory["fragment_tokens"],
                fragment_mask=trajectory["fragment_mask"],
                walked_path=batch["walked_path"],
                return_debug_states=True,
            )
            values = {
                "graph_conditioned_queries": branch[
                    "debug_graph_conditioned_queries"],
                "image_context": branch[
                    "debug_image_cross_attention_output"],
                "graph_state_contribution": branch[
                    "debug_graph_state_contribution"],
                "fragment_tokens": trajectory["fragment_tokens"],
                "fragment_mask": trajectory["fragment_mask"],
                "branch_offsets_norm": batch[
                    "branch_targets"]["branch_offsets_norm"],
                "branch_directions": batch[
                    "branch_targets"]["branch_directions"],
                "branch_mask": batch[
                    "branch_targets"]["branch_mask"],
                "branch_count": batch[
                    "branch_targets"]["branch_count"],
                "sample_ids": batch["metadata"]["dataset_index"],
                "traj_xy_norm": batch[
                    "trajectory_batch"]["traj_xy_norm"],
                "point_mask": batch[
                    "trajectory_batch"]["point_mask"],
            }
            for key, value in values.items():
                chunks.setdefault(key, []).append(
                    value.detach().cpu())

    tensors = {
        key: torch.cat(values, dim=0)
        for key, values in chunks.items()
    }
    cache = FrozenEvidenceDataset(tensors)
    finite = all(
        bool(torch.isfinite(value).all())
        for value in tensors.values()
        if value.is_floating_point()
    )
    return cache, {
        "sample_count": len(cache),
        "elapsed_seconds": float(time.perf_counter() - started_at),
        "finite": finite,
        "size_bytes": int(sum(
            value.numel() * value.element_size()
            for value in tensors.values()
        )),
        "tensor_shapes": {
            key: list(value.shape)
            for key, value in tensors.items()
        },
        "fragment_tokens_sha256": _tensor_sha256(
            tensors["fragment_tokens"]),
        "fragment_mask_sha256": _tensor_sha256(
            tensors["fragment_mask"]),
        "sample_ids_sha256": _tensor_sha256(
            tensors["sample_ids"]),
        "rpnet_frozen": True,
        "trajectory_fragment_encoder_frozen": True,
        "graph_state_encoder_frozen": True,
        "branch_decoder_frozen": True,
    }


def _targets_from_cache(
    batch: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return {
        "branch_offsets_norm": batch["branch_offsets_norm"],
        "branch_directions": batch["branch_directions"],
        "branch_mask": batch["branch_mask"].to(dtype=torch.bool),
        "branch_count": batch["branch_count"],
    }


def _prediction_from_tokens(
    *,
    branch_decoder: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    trajectory_tokens: torch.Tensor,
    trajectory_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    trajectory = trajectory_context_from_tokens(
        branch_decoder=branch_decoder,
        branch_queries=batch["graph_conditioned_queries"],
        trajectory_tokens=trajectory_tokens,
        trajectory_mask=trajectory_mask,
    )
    predictions = recompute_branch_predictions(
        branch_decoder,
        graph_conditioned_queries=batch[
            "graph_conditioned_queries"],
        image_context=batch["image_context"],
        trajectory_context=trajectory["trajectory_context"],
        graph_state_contribution=batch[
            "graph_state_contribution"],
    )
    predictions["branch_to_trajectory_attention"] = trajectory[
        "branch_to_trajectory_attention"]
    return predictions


def _no_trajectory_prediction(
    *,
    branch_decoder: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return recompute_branch_predictions(
        branch_decoder,
        graph_conditioned_queries=batch[
            "graph_conditioned_queries"],
        image_context=batch["image_context"],
        trajectory_context=torch.zeros_like(
            batch["graph_conditioned_queries"]),
        graph_state_contribution=batch[
            "graph_state_contribution"],
    )


def _evidence_prediction(
    *,
    evidence_encoder: TrajectoryEvidenceEncoder,
    branch_decoder: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    return_attention: bool,
) -> Dict[str, torch.Tensor]:
    adapted = build_trajectory_decoder_inputs(
        trajectory_mode=TRAJECTORY_MODE_EVIDENCE,
        fragment_tokens=batch["fragment_tokens"],
        fragment_mask=batch["fragment_mask"],
        evidence_encoder=evidence_encoder,
        return_attention=return_attention,
    )
    predictions = _prediction_from_tokens(
        branch_decoder=branch_decoder,
        batch=batch,
        trajectory_tokens=adapted["decoder_trajectory_tokens"],
        trajectory_mask=adapted["decoder_trajectory_mask"],
    )
    predictions.update({
        "trajectory_evidence_tokens": adapted[
            "trajectory_evidence_tokens"],
        "trajectory_evidence_mask": adapted[
            "trajectory_evidence_mask"],
    })
    if return_attention:
        predictions["fragment_attention_weights"] = adapted[
            "fragment_attention_weights"]
    return predictions


def _distribution(values: Sequence[float]) -> Dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
        }
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
    }


def _evidence_diagnostics(
    tokens: np.ndarray,
    evidence_mask: np.ndarray,
    attention: np.ndarray,
    fragment_mask: np.ndarray,
) -> Dict[str, Any]:
    cosine_values = []
    token_norms = []
    attention_cosine_values = []
    attention_top8_jaccard = []
    entropy_values = []
    maximum_attention = []
    latent_count = tokens.shape[1]
    for sample_index in range(tokens.shape[0]):
        valid_latents = evidence_mask[sample_index].astype(bool)
        sample_tokens = tokens[sample_index, valid_latents]
        if sample_tokens.size:
            norms = np.linalg.norm(sample_tokens, axis=-1)
            token_norms.extend(norms.tolist())
            normalized = sample_tokens / np.maximum(
                norms[:, None], 1e-12)
            similarities = normalized @ normalized.T
            triangle = np.triu_indices(
                sample_tokens.shape[0], k=1)
            cosine_values.extend(similarities[triangle].tolist())
        valid_fragments = fragment_mask[sample_index].astype(bool)
        fragment_count = int(valid_fragments.sum())
        if fragment_count:
            selected = attention[
                sample_index][:, valid_fragments]
            attention_norm = np.linalg.norm(
                selected, axis=-1, keepdims=True)
            normalized_attention = selected / np.maximum(
                attention_norm, 1e-12)
            attention_similarity = (
                normalized_attention @ normalized_attention.T)
            triangle = np.triu_indices(latent_count, k=1)
            attention_cosine_values.extend(
                attention_similarity[triangle].tolist())
            top_count = min(8, fragment_count)
            top_sets = [
                set(np.argsort(row, kind="stable")[-top_count:].tolist())
                for row in selected
            ]
            for left in range(latent_count):
                for right in range(left + 1, latent_count):
                    union = top_sets[left] | top_sets[right]
                    attention_top8_jaccard.append(
                        len(top_sets[left] & top_sets[right])
                        / max(len(union), 1))
            maximum_attention.extend(
                selected.max(axis=-1).tolist())
            denominator = math.log(fragment_count)
            for row in selected:
                entropy = float(-np.sum(
                    row * np.log(np.maximum(row, 1e-12))))
                entropy_values.append(
                    entropy / denominator
                    if fragment_count > 1 else 0.0)
    return {
        "num_evidence_tokens": int(latent_count),
        "pairwise_cosine_similarity": _distribution(cosine_values),
        "fragment_attention_pairwise_cosine_similarity":
            _distribution(attention_cosine_values),
        "fragment_attention_top8_jaccard":
            _distribution(attention_top8_jaccard),
        "hidden_norm": _distribution(token_norms),
        "normalized_fragment_attention_entropy":
            _distribution(entropy_values),
        "maximum_fragment_attention":
            _distribution(maximum_attention),
        "all_finite": bool(
            np.isfinite(tokens).all()
            and np.isfinite(attention).all()),
    }


def evaluate_evidence(
    *,
    evidence_encoder: Optional[TrajectoryEvidenceEncoder],
    branch_decoder: torch.nn.Module,
    criterion: torch.nn.Module,
    cache: FrozenEvidenceDataset,
    cfg: EasyDict,
    device: torch.device,
    collect_diagnostics: bool,
) -> Tuple[Dict[str, Any], Optional[Dict[str, np.ndarray]]]:
    branch_decoder.eval().requires_grad_(False)
    if evidence_encoder is not None:
        evidence_encoder.eval()
    loader = DataLoader(
        cache,
        batch_size=int(cfg.STAGE3E0.TRAINING.VAL_BATCH_SIZE),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    accumulators = {
        VARIANT_NO_TRAJECTORY: BranchVariantAccumulator(cfg),
        VARIANT_ORIGINAL_FRAGMENT: BranchVariantAccumulator(cfg),
    }
    if evidence_encoder is not None:
        accumulators[VARIANT_TRAJECTORY_EVIDENCE] = (
            BranchVariantAccumulator(cfg))
    loss_totals = {
        name: 0.0 for name in accumulators
    }
    sample_count = 0
    array_chunks: Dict[str, List[np.ndarray]] = {}
    with torch.no_grad():
        for cpu_batch in loader:
            batch = _move_flat_batch(cpu_batch, device)
            targets = _targets_from_cache(batch)
            predictions = {
                VARIANT_NO_TRAJECTORY: _no_trajectory_prediction(
                    branch_decoder=branch_decoder,
                    batch=batch,
                ),
                VARIANT_ORIGINAL_FRAGMENT: _prediction_from_tokens(
                    branch_decoder=branch_decoder,
                    batch=batch,
                    trajectory_tokens=batch["fragment_tokens"],
                    trajectory_mask=batch["fragment_mask"],
                ),
            }
            if evidence_encoder is not None:
                predictions[VARIANT_TRAJECTORY_EVIDENCE] = (
                    _evidence_prediction(
                        evidence_encoder=evidence_encoder,
                        branch_decoder=branch_decoder,
                        batch=batch,
                        return_attention=collect_diagnostics,
                    )
                )
            batch_size = int(batch["fragment_tokens"].shape[0])
            sample_count += batch_size
            for name, output in predictions.items():
                losses = criterion(output, targets)
                accumulators[name].update(
                    output, targets, losses["matches"])
                loss_totals[name] += float(
                    losses["loss"].cpu()) * batch_size
            if (
                    collect_diagnostics
                    and evidence_encoder is not None):
                evidence = predictions[
                    VARIANT_TRAJECTORY_EVIDENCE]
                values = {
                    "sample_ids": batch["sample_ids"],
                    "branch_count": batch["branch_count"],
                    "fragment_mask": batch["fragment_mask"],
                    "trajectory_evidence_tokens": evidence[
                        "trajectory_evidence_tokens"],
                    "trajectory_evidence_mask": evidence[
                        "trajectory_evidence_mask"],
                    "fragment_attention_weights": evidence[
                        "fragment_attention_weights"],
                }
                for key, value in values.items():
                    array_chunks.setdefault(key, []).append(
                        value.detach().cpu().numpy())

    results = {
        name: accumulator.compute()
        for name, accumulator in accumulators.items()
    }
    report = {
        "sample_count": sample_count,
        "variants": results,
        "loss": {
            name: value / max(sample_count, 1)
            for name, value in loss_totals.items()
        },
        "trajectory_evidence_minus_no_trajectory_branch_ap": (
            None
            if VARIANT_TRAJECTORY_EVIDENCE not in results
            else float(
                results[VARIANT_TRAJECTORY_EVIDENCE]["branch_ap"]
                - results[VARIANT_NO_TRAJECTORY]["branch_ap"])
        ),
        "original_fragment_minus_no_trajectory_branch_ap": float(
            results[VARIANT_ORIGINAL_FRAGMENT]["branch_ap"]
            - results[VARIANT_NO_TRAJECTORY]["branch_ap"]),
    }
    arrays = None
    if array_chunks:
        arrays = {
            key: np.concatenate(chunks, axis=0)
            for key, chunks in array_chunks.items()
        }
        report["trajectory_evidence_diagnostics"] = (
            _evidence_diagnostics(
                arrays["trajectory_evidence_tokens"],
                arrays["trajectory_evidence_mask"],
                arrays["fragment_attention_weights"],
                arrays["fragment_mask"],
            )
        )
    return report, arrays


def _train_one_epoch(
    *,
    evidence_encoder: TrajectoryEvidenceEncoder,
    branch_decoder: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    cfg: EasyDict,
    device: torch.device,
) -> Dict[str, float]:
    evidence_encoder.train().requires_grad_(True)
    branch_decoder.eval().requires_grad_(False)
    trainable = [
        parameter
        for parameter in evidence_encoder.parameters()
        if parameter.requires_grad
    ]
    totals = {
        "total": 0.0,
        "existence": 0.0,
        "endpoint": 0.0,
        "direction": 0.0,
    }
    sample_count = 0
    for cpu_batch in loader:
        batch = _move_flat_batch(cpu_batch, device)
        optimizer.zero_grad(set_to_none=True)
        predictions = _evidence_prediction(
            evidence_encoder=evidence_encoder,
            branch_decoder=branch_decoder,
            batch=batch,
            return_attention=False,
        )
        losses = criterion(
            predictions, _targets_from_cache(batch))
        loss = losses["loss"]
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite Stage 3E-0 loss")
        loss.backward()
        clip_grad_norm_(
            trainable,
            float(cfg.STAGE3E0.TRAINING.GRADIENT_CLIP_NORM),
        )
        optimizer.step()
        batch_size = int(batch["fragment_tokens"].shape[0])
        values = {
            "total": losses["loss"],
            "existence": losses["existence_loss"],
            "endpoint": losses["endpoint_loss"],
            "direction": losses["direction_loss"],
        }
        for name, value in values.items():
            totals[name] += float(value.detach().cpu()) * batch_size
        sample_count += batch_size
    return {
        name: value / max(sample_count, 1)
        for name, value in totals.items()
    }


def _checkpoint_payload(
    *,
    evidence_encoder: TrajectoryEvidenceEncoder,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    cfg: EasyDict,
    e4_checkpoint: Path,
    e4_sha256: str,
    metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    return build_stage3e0_checkpoint_payload(
        evidence_encoder=evidence_encoder,
        optimizer=optimizer,
        epoch=epoch,
        trajectory_mode=resolve_trajectory_evidence_mode(cfg),
        e4_checkpoint=str(e4_checkpoint.resolve()),
        e4_checkpoint_sha256=e4_sha256,
        config=_plain(cfg),
        metrics=metrics,
    )


def _save_attention_artifacts(
    *,
    arrays: Mapping[str, np.ndarray],
    dataset: Stage3CBranchDataset,
    cfg: EasyDict,
    output_dir: Path,
) -> Dict[str, Any]:
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    npz_path = diagnostics_dir / "fragment_attention.npz"
    np.savez_compressed(
        str(npz_path),
        **{
            key: arrays[key]
            for key in (
                "sample_ids",
                "branch_count",
                "fragment_mask",
                "trajectory_evidence_tokens",
                "trajectory_evidence_mask",
                "fragment_attention_weights",
            )
        },
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    examples_per_category = int(
        cfg.STAGE3E0.DIAGNOSTICS.
        VISUALIZATION_EXAMPLES_PER_CATEGORY)
    top_fragments = int(
        cfg.STAGE3E0.DIAGNOSTICS.VISUALIZATION_TOP_FRAGMENTS)
    categories = {
        "ordinary": lambda count: count == 1,
        "t_junction": lambda count: count == 2,
        "multi_branch": lambda count: count >= 3,
    }
    selected = []
    for category, predicate in categories.items():
        category_indices = [
            index
            for index, count in enumerate(arrays["branch_count"])
            if predicate(int(count))
        ][:examples_per_category]
        selected.extend(
            (category, index) for index in category_indices)

    visualization_dir = diagnostics_dir / "visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    window_size = float(cfg.TRAIN.WINDOW_SIZE)
    center = window_size / 2.0
    for category, array_index in selected:
        dataset_index = int(arrays["sample_ids"][array_index])
        sample = dataset[dataset_index]
        image = sample["aerial_image"][:3].permute(
            1, 2, 0).numpy()
        xy_norm = sample["trajectory_batch"]["traj_xy_norm"].numpy()
        point_mask = sample[
            "trajectory_batch"]["point_mask"].numpy().astype(bool)
        fragment_mask = arrays[
            "fragment_mask"][array_index].astype(bool)
        attention = arrays[
            "fragment_attention_weights"][array_index]
        offsets = sample[
            "branch_targets"]["branch_offsets_norm"].numpy()
        branch_mask = sample[
            "branch_targets"]["branch_mask"].numpy().astype(bool)
        latent_count = attention.shape[0]
        columns = min(4, latent_count)
        rows = int(math.ceil(latent_count / columns))
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(4.2 * columns, 4.2 * rows),
            squeeze=False,
        )
        for latent_index, axis in enumerate(axes.flat):
            if latent_index >= latent_count:
                axis.axis("off")
                continue
            axis.imshow(image)
            for fragment_index in np.flatnonzero(fragment_mask):
                points = xy_norm[fragment_index, point_mask[
                    fragment_index]]
                if not points.size:
                    continue
                pixels = points * center + center
                axis.plot(
                    pixels[:, 0],
                    pixels[:, 1],
                    color="white",
                    alpha=0.12,
                    linewidth=0.5,
                )
            valid_indices = np.flatnonzero(fragment_mask)
            ranked = valid_indices[np.argsort(
                attention[latent_index, valid_indices],
                kind="stable",
            )[::-1]]
            for rank, fragment_index in enumerate(
                    ranked[:top_fragments]):
                points = xy_norm[fragment_index, point_mask[
                    fragment_index]]
                if not points.size:
                    continue
                pixels = points * center + center
                axis.plot(
                    pixels[:, 0],
                    pixels[:, 1],
                    color=plt.cm.viridis(
                        1.0 - rank / max(top_fragments, 1)),
                    alpha=0.95,
                    linewidth=2.0,
                )
            for endpoint in offsets[branch_mask]:
                endpoint_pixel = endpoint * center + center
                axis.plot(
                    [center, endpoint_pixel[0]],
                    [center, endpoint_pixel[1]],
                    color="cyan",
                    linewidth=2.0,
                )
            axis.scatter(
                [center], [center], c="red", marker="x", s=45)
            axis.set_xlim(0, window_size)
            axis.set_ylim(window_size, 0)
            axis.set_title(
                "latent {} | max={:.3f}".format(
                    latent_index,
                    float(attention[latent_index].max()),
                )
            )
            axis.axis("off")
        figure.suptitle(
            "{} sample {}: fragment attention by latent query".format(
                category, dataset_index))
        figure.tight_layout()
        path = visualization_dir / (
            "{}_sample_{:04d}.png".format(category, dataset_index))
        figure.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(figure)
        paths.append(str(path.resolve()))
    return {
        "fragment_attention_npz": str(npz_path.resolve()),
        "visualizations": paths,
    }


def run_training(
    *,
    evidence_encoder: TrajectoryEvidenceEncoder,
    branch_decoder: torch.nn.Module,
    criterion: torch.nn.Module,
    train_cache: FrozenEvidenceDataset,
    val_cache: FrozenEvidenceDataset,
    val_dataset: Stage3CBranchDataset,
    cfg: EasyDict,
    device: torch.device,
    output_dir: Path,
    e4_checkpoint: Path,
    e4_sha256: str,
    cache_report: Mapping[str, Any],
) -> Dict[str, Any]:
    if resolve_trajectory_evidence_mode(
            cfg) != TRAJECTORY_MODE_EVIDENCE:
        raise ValueError(
            "Stage 3E-0 training requires trajectory_evidence mode")
    evidence_encoder.train().requires_grad_(True)
    branch_decoder.eval().requires_grad_(False)
    trainable_parameters = [
        parameter
        for parameter in evidence_encoder.parameters()
        if parameter.requires_grad
    ]
    trainable_parameter_count = sum(
        parameter.numel() for parameter in trainable_parameters)
    optimizer = (
        None
        if not trainable_parameters
        else torch.optim.AdamW(
            trainable_parameters,
            lr=float(cfg.STAGE3E0.TRAINING.LEARNING_RATE),
            weight_decay=float(
                cfg.STAGE3E0.TRAINING.WEIGHT_DECAY),
        )
    )
    train_loader = (
        None
        if optimizer is None
        else DataLoader(
            train_cache,
            batch_size=int(cfg.STAGE3E0.TRAINING.BATCH_SIZE),
            shuffle=True,
            num_workers=int(cfg.STAGE3E0.TRAINING.NUM_WORKERS),
            pin_memory=device.type == "cuda",
            generator=torch.Generator().manual_seed(
                int(cfg.STAGE3C.SEED)),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    curve_path = output_dir / "training_curve.jsonl"
    with curve_path.open("w", encoding="utf-8"):
        pass
    curve = []
    initial_shared_state_sha256 = (
        _shared_evidence_state_sha256(evidence_encoder))
    best_ap = -1.0
    best_epoch = 0
    started_at = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if optimizer is None:
        validation, _ = evaluate_evidence(
            evidence_encoder=evidence_encoder,
            branch_decoder=branch_decoder,
            criterion=criterion,
            cache=val_cache,
            cfg=cfg,
            device=device,
            collect_diagnostics=False,
        )
        branch_ap = float(validation["variants"][
            VARIANT_TRAJECTORY_EVIDENCE]["branch_ap"])
        record = {
            "epoch": 0,
            "train_loss": None,
            "parameter_free_evidence_module": True,
            "validation": validation,
        }
        curve.append(record)
        with curve_path.open("a", encoding="utf-8") as output_file:
            output_file.write(
                json.dumps(_plain(record), sort_keys=True) + "\n")
        payload = _checkpoint_payload(
            evidence_encoder=evidence_encoder,
            optimizer=optimizer,
            epoch=0,
            cfg=cfg,
            e4_checkpoint=e4_checkpoint,
            e4_sha256=e4_sha256,
            metrics=validation,
        )
        save_stage3e0_checkpoint(
            checkpoint_dir / "stage3e0.latest.pth.tar",
            payload,
        )
        best_ap = branch_ap
        best_epoch = 0
        save_stage3e0_checkpoint(
            checkpoint_dir / "stage3e0.best.pth.tar",
            payload,
        )
        print(
            "Stage3E0 parameter-free evidence branch_ap={:.6f} "
            "delta_no_traj={:+.6f}".format(
                branch_ap,
                validation[
                    "trajectory_evidence_minus_no_trajectory_branch_ap"],
            ),
            flush=True,
        )
    else:
        for epoch in range(
                1, int(cfg.STAGE3E0.TRAINING.EPOCHS) + 1):
            train_loss = _train_one_epoch(
                evidence_encoder=evidence_encoder,
                branch_decoder=branch_decoder,
                criterion=criterion,
                optimizer=optimizer,
                loader=train_loader,
                cfg=cfg,
                device=device,
            )
            validation, _ = evaluate_evidence(
                evidence_encoder=evidence_encoder,
                branch_decoder=branch_decoder,
                criterion=criterion,
                cache=val_cache,
                cfg=cfg,
                device=device,
                collect_diagnostics=False,
            )
            branch_ap = float(validation["variants"][
                VARIANT_TRAJECTORY_EVIDENCE]["branch_ap"])
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation": validation,
            }
            curve.append(record)
            with curve_path.open("a", encoding="utf-8") as output_file:
                output_file.write(
                    json.dumps(_plain(record), sort_keys=True) + "\n")
            payload = _checkpoint_payload(
                evidence_encoder=evidence_encoder,
                optimizer=optimizer,
                epoch=epoch,
                cfg=cfg,
                e4_checkpoint=e4_checkpoint,
                e4_sha256=e4_sha256,
                metrics=validation,
            )
            save_stage3e0_checkpoint(
                checkpoint_dir / "stage3e0.latest.pth.tar",
                payload,
            )
            if branch_ap > best_ap:
                best_ap = branch_ap
                best_epoch = epoch
                save_stage3e0_checkpoint(
                    checkpoint_dir / "stage3e0.best.pth.tar",
                    payload,
                )
            print(
                "Stage3E0 epoch {}/{} loss={:.6f} branch_ap={:.6f} "
                "delta_no_traj={:+.6f}".format(
                    epoch,
                    int(cfg.STAGE3E0.TRAINING.EPOCHS),
                    train_loss["total"],
                    branch_ap,
                    validation[
                        "trajectory_evidence_minus_no_trajectory_branch_ap"],
                ),
                flush=True,
            )

    best_path = checkpoint_dir / "stage3e0.best.pth.tar"
    load_stage3e0_checkpoint(
        best_path,
        evidence_encoder=evidence_encoder,
        optimizer=None,
        map_location=device,
        expected_e4_sha256=e4_sha256,
    )
    best_validation, arrays = evaluate_evidence(
        evidence_encoder=evidence_encoder,
        branch_decoder=branch_decoder,
        criterion=criterion,
        cache=val_cache,
        cfg=cfg,
        device=device,
        collect_diagnostics=True,
    )
    artifacts = _save_attention_artifacts(
        arrays=arrays,
        dataset=val_dataset,
        cfg=cfg,
        output_dir=output_dir,
    )
    report = {
        "schema_version": "stage3e0-training-v1",
        "trajectory_mode": TRAJECTORY_MODE_EVIDENCE,
        "evidence_aggregation_mode":
            resolve_evidence_aggregation_mode(cfg),
        "num_evidence_tokens": int(
            cfg.STAGE3E0.MODEL.NUM_EVIDENCE_TOKENS),
        "trainable_parameter_count": int(
            trainable_parameter_count),
        "parameter_free_evidence_module": optimizer is None,
        "seed": int(cfg.STAGE3C.SEED),
        "e4_checkpoint": str(e4_checkpoint.resolve()),
        "e4_checkpoint_sha256": e4_sha256,
        "best_epoch": best_epoch,
        "best_checkpoint": str(best_path.resolve()),
        "best_validation": best_validation,
        "curve": curve,
        "cache": dict(cache_report),
        "initial_shared_evidence_state_sha256":
            initial_shared_state_sha256,
        "artifacts": artifacts,
        "elapsed_seconds": float(
            time.perf_counter() - started_at),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda" else 0),
        "trainable": {
            "trajectory_evidence_encoder": optimizer is not None,
            "trajectory_evidence_module_active": True,
            "rpnet": False,
            "graph_state_encoder": False,
            "trajectory_fragment_encoder": False,
            "branch_decoder": False,
        },
        "branch_predictions_feed_path_push": False,
    }
    _write_json(output_dir / "training_summary.json", report)
    _write_json(output_dir / "evaluation.json", best_validation)
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
    trajectory_mode = resolve_trajectory_evidence_mode(cfg)
    if (
            args.mode == "train"
            and trajectory_mode
            != TRAJECTORY_MODE_EVIDENCE):
        raise ValueError(
            "original_fragment is an evaluation-only ablation")
    _set_seed(int(cfg.STAGE3C.SEED))
    device = _resolve_device(
        args.device or str(cfg.STAGE3C.DEVICE))
    output_dir = (
        args.output_dir or Path(cfg.STAGE3E0.OUTPUT_DIR))
    e4_checkpoint = Path(
        cfg.STAGE3E0.E4_CHECKPOINT).resolve(strict=False)
    image_checkpoint = Path(
        cfg.STAGE3C.IMAGE_CHECKPOINT).resolve(strict=False)
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
    _freeze_modules((rpnet,) + tuple(modules))
    evidence_encoder = _build_evidence_encoder(cfg, device)
    if args.checkpoint is not None:
        load_stage3e0_checkpoint(
            args.checkpoint,
            evidence_encoder=evidence_encoder,
            optimizer=None,
            map_location=device,
            expected_e4_sha256=e4_sha256,
        )
    elif (
            args.mode == "evaluate"
            and trajectory_mode == TRAJECTORY_MODE_EVIDENCE):
        raise ValueError(
            "trajectory_evidence evaluation requires --checkpoint")

    dataset_dir = Path(cfg.STAGE3C.DATASET_DIR)
    val_dataset = Stage3CBranchDataset(
        dataset_dir, "val", preload=True)
    val_cache, val_cache_report = build_frozen_evidence_cache(
        dataset=val_dataset,
        rpnet=rpnet,
        modules=modules,
        cfg=cfg,
        device=device,
    )
    train_cache = None
    cache_report = {"val": val_cache_report}
    if args.mode == "train":
        train_dataset = Stage3CBranchDataset(
            dataset_dir, "train", preload=True)
        train_cache, train_cache_report = build_frozen_evidence_cache(
            dataset=train_dataset,
            rpnet=rpnet,
            modules=modules,
            cfg=cfg,
            device=device,
        )
        cache_report["train"] = train_cache_report
    criterion = _build_branch_criterion(cfg)
    if args.mode == "train":
        report = run_training(
            evidence_encoder=evidence_encoder,
            branch_decoder=modules[2],
            criterion=criterion,
            train_cache=train_cache,
            val_cache=val_cache,
            val_dataset=val_dataset,
            cfg=cfg,
            device=device,
            output_dir=output_dir,
            e4_checkpoint=e4_checkpoint,
            e4_sha256=e4_sha256,
            cache_report=cache_report,
        )
    else:
        selected_encoder = (
            evidence_encoder
            if trajectory_mode == TRAJECTORY_MODE_EVIDENCE
            else None
        )
        evaluation, arrays = evaluate_evidence(
            evidence_encoder=selected_encoder,
            branch_decoder=modules[2],
            criterion=criterion,
            cache=val_cache,
            cfg=cfg,
            device=device,
            collect_diagnostics=selected_encoder is not None,
        )
        artifacts = None
        if arrays is not None:
            artifacts = _save_attention_artifacts(
                arrays=arrays,
                dataset=val_dataset,
                cfg=cfg,
                output_dir=output_dir,
            )
        report = {
            "schema_version": "stage3e0-evaluation-v1",
            "trajectory_mode": trajectory_mode,
            "evidence_aggregation_mode":
                resolve_evidence_aggregation_mode(cfg),
            "evaluation": evaluation,
            "cache": cache_report,
            "artifacts": artifacts,
            "branch_predictions_feed_path_push": False,
        }
        _write_json(output_dir / "evaluation.json", report)
    print(json.dumps({
        "trajectory_mode": trajectory_mode,
        "evidence_aggregation_mode":
            resolve_evidence_aggregation_mode(cfg),
        "output_dir": str(output_dir.resolve()),
        "completed": True,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
