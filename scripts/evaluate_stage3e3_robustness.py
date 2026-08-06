"""Evaluate Stage 3E-3 M=1 trajectory evidence robustness."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_stage3d_c0_support_aggregation import (  # noqa: E402
    BranchVariantAccumulator,
)
from train_branch_aux import (  # noqa: E402
    _build_auxiliary_modules,
    _build_branch_criterion,
    _load_config,
    _load_frozen_rpnet,
    _resolve_device,
    _set_seed,
)
from train_trajectory_evidence import (  # noqa: E402
    FrozenEvidenceDataset,
    _build_evidence_encoder,
    _evidence_diagnostics,
    _evidence_prediction,
    _freeze_modules,
    _module_state_sha256,
    _move_flat_batch,
    _no_trajectory_prediction,
    _plain,
    _prediction_from_tokens,
    _sha256,
    _targets_from_cache,
    _tensor_sha256,
    _write_json,
    build_frozen_evidence_cache,
)
from utils.stage3c_branch_dataset import Stage3CBranchDataset  # noqa: E402
from utils.stage3c_checkpoint import load_stage3c_checkpoint  # noqa: E402
from utils.stage3e0_checkpoint import load_stage3e0_checkpoint  # noqa: E402
from utils.trajectory_evidence_robustness import (  # noqa: E402
    deterministic_fragment_thinning,
    replace_trajectory_with_global_donors,
)


MODE_IMAGE_GRAPH = "image_graph"
MODE_ORIGINAL = "original_fragment"
MODE_FULL = "full_trajectory"
MODE_NO_TRAJECTORY = "no_trajectory"
MODE_RETAIN_75 = "retain_75"
MODE_RETAIN_50 = "retain_50"
MODE_RETAIN_25 = "retain_25"
MODE_WRONG = "wrong_sample_trajectory"
DIAGNOSTIC_MODES = {
    MODE_FULL,
    MODE_RETAIN_50,
    MODE_RETAIN_25,
    MODE_WRONG,
}


def _max_prediction_difference(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> Dict[str, float]:
    keys = (
        "branch_exist_logits",
        "branch_offsets_norm",
        "branch_directions",
    )
    values = {
        key: float(torch.max(torch.abs(left[key] - right[key])).cpu())
        for key in keys
    }
    values["maximum"] = max(values.values())
    return values


def _cache_with_fragment_mask(
    cache: FrozenEvidenceDataset,
    fragment_mask: torch.Tensor,
) -> FrozenEvidenceDataset:
    tensors = dict(cache.tensors)
    tensors["fragment_mask"] = fragment_mask.to(dtype=torch.bool)
    return FrozenEvidenceDataset(tensors)


def build_stage3e3_variant_caches(
    cache: FrozenEvidenceDataset,
    *,
    wrong_sample_shift: int = 1,
) -> Dict[str, FrozenEvidenceDataset]:
    tensors = cache.tensors
    identities = {
        "track_indices": tensors["track_indices"],
        "start_point_indices": tensors["start_point_indices"],
        "end_point_indices": tensors["end_point_indices"],
    }
    variants = {
        MODE_FULL: cache,
    }
    for name, ratio in (
            (MODE_RETAIN_75, 0.75),
            (MODE_RETAIN_50, 0.50),
            (MODE_RETAIN_25, 0.25)):
        mask = deterministic_fragment_thinning(
            fragment_mask=tensors["fragment_mask"],
            sample_ids=tensors["sample_ids"],
            retain_ratio=ratio,
            **identities,
        )
        variants[name] = _cache_with_fragment_mask(cache, mask)
    donor_tensors = replace_trajectory_with_global_donors(
        tensors,
        trajectory_keys=(
            "fragment_tokens",
            "fragment_mask",
            "track_indices",
            "start_point_indices",
            "end_point_indices",
            "traj_xy_norm",
            "point_mask",
        ),
        cyclic_shift=wrong_sample_shift,
    )
    variants[MODE_WRONG] = FrozenEvidenceDataset(donor_tensors)
    return variants


def _evaluate_variant(
    *,
    name: str,
    cache: FrozenEvidenceDataset,
    evidence_encoder: torch.nn.Module,
    branch_decoder: torch.nn.Module,
    criterion: torch.nn.Module,
    cfg,
    device: torch.device,
    collect_diagnostics: bool,
) -> Tuple[Dict[str, Any], Optional[Dict[str, np.ndarray]], Dict[str, float]]:
    accumulator = BranchVariantAccumulator(cfg)
    loader = DataLoader(
        cache,
        batch_size=int(cfg.STAGE3E0.TRAINING.VAL_BATCH_SIZE),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    loss_total = 0.0
    sample_count = 0
    all_finite = True
    arrays: Dict[str, List[np.ndarray]] = {}
    equivalence = {
        "branch_exist_logits": 0.0,
        "branch_offsets_norm": 0.0,
        "branch_directions": 0.0,
        "maximum": 0.0,
    }
    with torch.no_grad():
        for cpu_batch in loader:
            batch = _move_flat_batch(cpu_batch, device)
            targets = _targets_from_cache(batch)
            if name in (MODE_IMAGE_GRAPH, MODE_NO_TRAJECTORY):
                predictions = _no_trajectory_prediction(
                    branch_decoder=branch_decoder,
                    batch=batch,
                )
                empty_batch = dict(batch)
                empty_batch["fragment_mask"] = torch.zeros_like(
                    batch["fragment_mask"], dtype=torch.bool)
                empty_predictions = _evidence_prediction(
                    evidence_encoder=evidence_encoder,
                    branch_decoder=branch_decoder,
                    batch=empty_batch,
                    return_attention=False,
                )
                differences = _max_prediction_difference(
                    predictions, empty_predictions)
                equivalence = {
                    key: max(equivalence[key], value)
                    for key, value in differences.items()
                }
            elif name == MODE_ORIGINAL:
                predictions = _prediction_from_tokens(
                    branch_decoder=branch_decoder,
                    batch=batch,
                    trajectory_tokens=batch["fragment_tokens"],
                    trajectory_mask=batch["fragment_mask"],
                )
            else:
                predictions = _evidence_prediction(
                    evidence_encoder=evidence_encoder,
                    branch_decoder=branch_decoder,
                    batch=batch,
                    return_attention=collect_diagnostics,
                )
            losses = criterion(predictions, targets)
            all_finite = bool(all_finite and all(
                torch.isfinite(value).all()
                for value in predictions.values()
                if isinstance(value, torch.Tensor)
                and value.is_floating_point()
            ))
            if not all_finite or not bool(torch.isfinite(losses["loss"])):
                raise RuntimeError(
                    "non-finite Stage 3E-3 output in mode {}".format(name))
            accumulator.update(predictions, targets, losses["matches"])
            batch_size = int(batch["fragment_tokens"].shape[0])
            sample_count += batch_size
            loss_total += float(losses["loss"].cpu()) * batch_size
            if collect_diagnostics:
                values = {
                    "sample_ids": batch["sample_ids"],
                    "branch_count": batch["branch_count"],
                    "fragment_mask": batch["fragment_mask"],
                    "trajectory_evidence_tokens": predictions[
                        "trajectory_evidence_tokens"],
                    "trajectory_evidence_mask": predictions[
                        "trajectory_evidence_mask"],
                    "fragment_attention_weights": predictions[
                        "fragment_attention_weights"],
                }
                if "trajectory_source_sample_ids" in batch:
                    values["trajectory_source_sample_ids"] = batch[
                        "trajectory_source_sample_ids"]
                for key, value in values.items():
                    arrays.setdefault(key, []).append(
                        value.detach().cpu().numpy())
    metrics = accumulator.compute()
    result = {
        "sample_count": sample_count,
        "loss": loss_total / max(sample_count, 1),
        "all_finite": all_finite,
        **metrics,
    }
    joined = None
    if arrays:
        joined = {
            key: np.concatenate(chunks, axis=0)
            for key, chunks in arrays.items()
        }
        result["attention_diagnostics"] = _evidence_diagnostics(
            joined["trajectory_evidence_tokens"],
            joined["trajectory_evidence_mask"],
            joined["fragment_attention_weights"],
            joined["fragment_mask"],
        )
    return result, joined, equivalence


def _save_visualizations(
    *,
    name: str,
    arrays: Mapping[str, np.ndarray],
    cache: FrozenEvidenceDataset,
    dataset: Stage3CBranchDataset,
    cfg,
    output_dir: Path,
) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = output_dir / "attention_visualizations" / name
    directory.mkdir(parents=True, exist_ok=True)
    sample_to_row = {
        int(sample_id): row
        for row, sample_id in enumerate(
            cache.tensors["sample_ids"].cpu().numpy())
    }
    categories = {
        "ordinary": lambda count: count == 1,
        "t_junction": lambda count: count == 2,
        "multi_branch": lambda count: count >= 3,
    }
    per_category = int(
        cfg.STAGE3E0.DIAGNOSTICS.VISUALIZATION_EXAMPLES_PER_CATEGORY)
    top_count = int(
        cfg.STAGE3E0.DIAGNOSTICS.VISUALIZATION_TOP_FRAGMENTS)
    window_size = float(cfg.TRAIN.WINDOW_SIZE)
    center = window_size / 2.0
    paths = []
    for category, predicate in categories.items():
        candidates = [
            index for index, count in enumerate(arrays["branch_count"])
            if predicate(int(count))
        ][:per_category]
        for array_index in candidates:
            sample_id = int(arrays["sample_ids"][array_index])
            row = sample_to_row[sample_id]
            sample = dataset[sample_id]
            image = sample["aerial_image"][:3].permute(1, 2, 0).numpy()
            xy = cache.tensors["traj_xy_norm"][row].cpu().numpy()
            point_mask = cache.tensors[
                "point_mask"][row].cpu().numpy().astype(bool)
            fragment_mask = arrays[
                "fragment_mask"][array_index].astype(bool)
            attention = arrays[
                "fragment_attention_weights"][array_index, 0]
            figure, axis = plt.subplots(figsize=(5.2, 5.2))
            axis.imshow(image)
            valid = np.flatnonzero(fragment_mask)
            for fragment_index in valid:
                points = xy[fragment_index, point_mask[fragment_index]]
                pixels = points * center + center
                if pixels.size:
                    axis.plot(
                        pixels[:, 0], pixels[:, 1], color="white",
                        alpha=0.12, linewidth=0.5)
            ranked = valid[np.argsort(
                attention[valid], kind="stable")[::-1]]
            for rank, fragment_index in enumerate(ranked[:top_count]):
                points = xy[fragment_index, point_mask[fragment_index]]
                pixels = points * center + center
                if pixels.size:
                    axis.plot(
                        pixels[:, 0], pixels[:, 1],
                        color=plt.cm.viridis(
                            1.0 - rank / max(top_count, 1)),
                        linewidth=2.0, alpha=0.95)
            branch_offsets = cache.tensors[
                "branch_offsets_norm"][row].cpu().numpy()
            branch_mask = cache.tensors[
                "branch_mask"][row].cpu().numpy().astype(bool)
            for endpoint in branch_offsets[branch_mask]:
                pixel = endpoint * center + center
                axis.plot(
                    [center, pixel[0]], [center, pixel[1]],
                    color="cyan", linewidth=2.0)
            axis.scatter([center], [center], c="red", marker="x", s=50)
            axis.set_xlim(0, window_size)
            axis.set_ylim(window_size, 0)
            axis.axis("off")
            axis.set_title("{} | {} | sample {}".format(
                name, category, sample_id))
            path = directory / "{}_sample_{:04d}.png".format(
                category, sample_id)
            figure.savefig(str(path), dpi=150, bbox_inches="tight")
            plt.close(figure)
            paths.append(str(path.resolve()))
    return paths


def _validate_val_hashes(cfg, e4_sha256: str, cache_report: Mapping[str, Any]):
    expected = cfg.STAGE3E3.PREFLIGHT.EXPECTED_SHA256
    pairs = {
        "e4_checkpoint": (e4_sha256, str(expected.E4_CHECKPOINT)),
        "val_fragment_tokens": (
            cache_report["fragment_tokens_sha256"],
            str(expected.VAL_FRAGMENT_TOKENS)),
        "val_fragment_mask": (
            cache_report["fragment_mask_sha256"],
            str(expected.VAL_FRAGMENT_MASK)),
        "val_sample_ids": (
            cache_report["sample_ids_sha256"],
            str(expected.VAL_SAMPLE_IDS)),
    }
    mismatches = {
        key: {"actual": actual, "expected": wanted}
        for key, (actual, wanted) in pairs.items()
        if actual != wanted
    }
    if mismatches:
        raise RuntimeError(
            "Stage 3E-3 validation SHA mismatch: {}".format(
                json.dumps(mismatches, sort_keys=True)))
    return {key: actual for key, (actual, _) in pairs.items()}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = _load_config(args.config)
    if int(cfg.STAGE3E0.MODEL.NUM_EVIDENCE_TOKENS) != 1:
        raise ValueError("Stage 3E-3 requires NUM_EVIDENCE_TOKENS=1")
    if str(cfg.STAGE3E0.MODEL.AGGREGATION_MODE) != "latent_attention":
        raise ValueError("Stage 3E-3 requires latent_attention")
    configured_ratios = [
        float(value) for value in cfg.STAGE3E3.ROBUSTNESS.RETAIN_RATIOS]
    if configured_ratios != [0.75, 0.50, 0.25]:
        raise ValueError(
            "Stage 3E-3 RETAIN_RATIOS must be [0.75, 0.50, 0.25]")
    _set_seed(int(cfg.STAGE3C.SEED))
    device = _resolve_device(args.device or str(cfg.STAGE3C.DEVICE))
    output_dir = args.output_dir or Path(cfg.STAGE3E0.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    e4_checkpoint = Path(cfg.STAGE3E0.E4_CHECKPOINT).resolve(strict=False)
    image_checkpoint = Path(cfg.STAGE3C.IMAGE_CHECKPOINT).resolve(strict=False)
    if not e4_checkpoint.is_file():
        raise FileNotFoundError("E4 checkpoint not found: {}".format(
            e4_checkpoint))
    if not image_checkpoint.is_file():
        raise FileNotFoundError("RPNet checkpoint not found: {}".format(
            image_checkpoint))
    e4_sha256 = _sha256(e4_checkpoint)
    rpnet, _ = _load_frozen_rpnet(cfg, image_checkpoint, device)
    modules = _build_auxiliary_modules(cfg, device)
    load_stage3c_checkpoint(
        e4_checkpoint,
        trajectory_encoder=modules[0],
        graph_state_encoder=modules[1],
        branch_decoder=modules[2],
        optimizer=None,
        map_location=device,
    )
    evidence_encoder = _build_evidence_encoder(cfg, device)
    load_stage3e0_checkpoint(
        args.checkpoint,
        evidence_encoder=evidence_encoder,
        optimizer=None,
        map_location=device,
        expected_e4_sha256=e4_sha256,
    )
    frozen_modules = {
        "rpnet": rpnet,
        "trajectory_fragment_encoder": modules[0],
        "graph_state_encoder": modules[1],
        "branch_decoder": modules[2],
        "trajectory_evidence_encoder": evidence_encoder,
    }
    _freeze_modules(tuple(frozen_modules.values()))
    before_hashes = {
        name: _module_state_sha256(module)
        for name, module in frozen_modules.items()
    }

    dataset = Stage3CBranchDataset(
        Path(cfg.STAGE3C.DATASET_DIR), "val", preload=True)
    cache, cache_report = build_frozen_evidence_cache(
        dataset=dataset,
        rpnet=rpnet,
        modules=modules,
        cfg=cfg,
        device=device,
    )
    hash_checks = _validate_val_hashes(cfg, e4_sha256, cache_report)
    variants = build_stage3e3_variant_caches(
        cache,
        wrong_sample_shift=int(
            cfg.STAGE3E3.ROBUSTNESS.WRONG_SAMPLE_CYCLIC_SHIFT),
    )
    criterion = _build_branch_criterion(cfg)
    results = {}
    diagnostics = {}
    visualizations = {}
    no_trajectory_equivalence = None
    evaluation_caches = {
        MODE_IMAGE_GRAPH: cache,
        MODE_NO_TRAJECTORY: cache,
        MODE_ORIGINAL: cache,
        **variants,
    }
    for name, variant_cache in evaluation_caches.items():
        collect = name in DIAGNOSTIC_MODES
        metrics, arrays, equivalence = _evaluate_variant(
            name=name,
            cache=variant_cache,
            evidence_encoder=evidence_encoder,
            branch_decoder=modules[2],
            criterion=criterion,
            cfg=cfg,
            device=device,
            collect_diagnostics=collect,
        )
        results[name] = metrics
        if name == MODE_NO_TRAJECTORY:
            no_trajectory_equivalence = equivalence
        if arrays is not None:
            diagnostics[name] = metrics["attention_diagnostics"]
            visualizations[name] = _save_visualizations(
                name=name,
                arrays=arrays,
                cache=variant_cache,
                dataset=dataset,
                cfg=cfg,
                output_dir=output_dir,
            )
            np.savez_compressed(
                str(output_dir / "{}_attention.npz".format(name)),
                **arrays,
            )

    tolerance = float(cfg.STAGE3E3.NO_TRAJECTORY_EQUIVALENCE_TOLERANCE)
    if no_trajectory_equivalence["maximum"] > tolerance:
        raise RuntimeError(
            "no_trajectory is not image_graph-equivalent: {} > {}".format(
                no_trajectory_equivalence["maximum"], tolerance))
    after_hashes = {
        name: _module_state_sha256(module)
        for name, module in frozen_modules.items()
    }
    frozen_unchanged = before_hashes == after_hashes
    if not frozen_unchanged:
        raise RuntimeError("a frozen module changed during robustness evaluation")
    report = {
        "schema_version": "stage3e3-robustness-v1",
        "validation_teacher_forced_auxiliary_metrics": True,
        "seed": int(cfg.STAGE3C.SEED),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "e4_checkpoint": str(e4_checkpoint),
        "hash_checks": hash_checks,
        "cache": cache_report,
        "variants": results,
        "attention_diagnostics": diagnostics,
        "no_trajectory_equivalence": no_trajectory_equivalence,
        "frozen_module_sha256_before": before_hashes,
        "frozen_module_sha256_after": after_hashes,
        "frozen_modules_unchanged": frozen_unchanged,
        "visualizations": visualizations,
        "branch_predictions_feed_path_push": False,
    }
    _write_json(output_dir / "robustness_evaluation.json", report)
    _write_json(output_dir / "attention_diagnostics.json", diagnostics)
    print(json.dumps({
        "output": str((output_dir / "robustness_evaluation.json").resolve()),
        "full_branch_ap": results[MODE_FULL]["branch_ap"],
        "image_graph_branch_ap": results[MODE_IMAGE_GRAPH]["branch_ap"],
        "wrong_sample_branch_ap": results[MODE_WRONG]["branch_ap"],
        "no_trajectory_max_abs_difference":
            no_trajectory_equivalence["maximum"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
