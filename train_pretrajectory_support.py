"""Stage 3D-B0 pre-trajectory branch-conditioned support validation.

The trainable head reads only graph-conditioned queries, image/walked-path
cross-attention context, and frozen fragment tokens.  It never reads E4
trajectory context, final branch tokens, branch geometry outputs, RPNet
anchor features, or Path.push state.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
from easydict import EasyDict
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.trajectory_support_head import (  # noqa: E402
    TrajectorySupportHead,
    trajectory_support_bce_loss,
)
from train_branch_aux import (  # noqa: E402
    _build_auxiliary_modules,
    _build_branch_criterion,
    _load_config,
    _load_frozen_rpnet,
    _resolve_device,
    _set_seed,
)
from train_trajectory_support import (  # noqa: E402
    FrozenSupportDataset,
    _freeze_e4_modules,
    _matches_from_batch,
    _move_flat_batch,
    _plain,
    _sha256,
    _target_parameters,
    _write_json,
    build_frozen_support_cache,
    run_label_diagnostics,
)
from utils.stage3c_branch_dataset import (  # noqa: E402
    Stage3CBranchDataset,
)
from utils.stage3c_checkpoint import (  # noqa: E402
    load_stage3c_checkpoint,
)
from utils.stage3d_checkpoint import (  # noqa: E402
    build_stage3d_support_checkpoint_payload,
    load_stage3d_support_checkpoint,
    save_stage3d_support_checkpoint,
)
from utils.trajectory_support_ranking import (  # noqa: E402
    TrajectorySupportRankingAccumulator,
)
from utils.trajectory_support_features import (  # noqa: E402
    build_pre_trajectory_branch_tokens,
)
from utils.trajectory_support_targets import (  # noqa: E402
    build_trajectory_support_targets,
)


PRE_BRANCH_KEY = "pre_trajectory_branch_tokens"
POST_BRANCH_KEY = "branch_tokens"
RAW_ATTENTION_KEY = "trajectory_attention_weights"


def _ranking_accumulator(cfg: EasyDict):
    return TrajectorySupportRankingAccumulator(
        ranking_ks=tuple(
            int(value)
            for value in cfg.STAGE3D_B0.EVALUATION.RANKING_KS),
        jaccard_k=int(cfg.STAGE3D_B0.EVALUATION.JACCARD_K),
    )


def _accumulate_batch(
    accumulator: TrajectorySupportRankingAccumulator,
    *,
    scores: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
) -> None:
    accumulator.update(
        scores=scores,
        support_targets=batch["support_targets"],
        support_positive_mask=batch["support_positive_mask"],
        support_valid=batch["support_valid"],
        branch_mask=batch["branch_mask"],
        fragment_mask=batch["fragment_mask"],
        matches=_matches_from_batch(batch["matched_target_indices"]),
        branch_count=batch["branch_count"],
        sample_ids=batch["sample_ids"],
    )


def evaluate_score_source(
    *,
    dataset: FrozenSupportDataset,
    cfg: EasyDict,
    device: torch.device,
    batch_size: int,
    support_head: TrajectorySupportHead = None,
    branch_key: str = None,
    score_key: str = None,
) -> Dict[str, Any]:
    """Evaluate either a frozen score tensor or one independent support head."""

    if (support_head is None) == (score_key is None):
        raise ValueError(
            "provide exactly one of support_head or score_key")
    if support_head is not None and not branch_key:
        raise ValueError("branch_key is required with support_head")
    if support_head is not None:
        support_head.eval()
    accumulator = _ranking_accumulator(cfg)
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    loss_sum = 0.0
    pair_count = 0
    started_at = time.perf_counter()
    with torch.no_grad():
        for cpu_batch in loader:
            batch = _move_flat_batch(cpu_batch, device)
            if support_head is None:
                scores = batch[score_key]
            else:
                logits = support_head(
                    batch[branch_key],
                    batch["fragment_tokens"],
                    batch["fragment_mask"],
                )
                scores = torch.sigmoid(logits)
                losses = trajectory_support_bce_loss(
                    logits,
                    batch["support_targets"],
                    batch["support_valid"],
                    batch["fragment_mask"],
                    _matches_from_batch(
                        batch["matched_target_indices"]),
                )
                pairs = int(losses["supervised_pair_count"])
                loss_sum += float(losses["loss"]) * pairs
                pair_count += pairs
            _accumulate_batch(
                accumulator, scores=scores, batch=batch)
    result = accumulator.compute()
    result["loss"] = (
        loss_sum / max(pair_count, 1)
        if support_head is not None else None)
    result["supervised_pair_count"] = (
        pair_count if support_head is not None else None)
    result["elapsed_seconds"] = float(
        time.perf_counter() - started_at)
    return result


def _build_pre_head(cfg: EasyDict, device: torch.device):
    model_cfg = cfg.STAGE3D_B0.MODEL
    return TrajectorySupportHead(
        hidden_dim=int(cfg.STAGE3C.MODEL.HIDDEN_DIM),
        branch_input_dim=int(model_cfg.BRANCH_INPUT_DIM),
        fragment_input_dim=int(model_cfg.FRAGMENT_INPUT_DIM),
        projection_dim=int(model_cfg.PROJECTION_DIM),
    ).to(device=device)


def _checkpoint_payload(
    *,
    head: TrajectorySupportHead,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    e4_checkpoint: Path,
    e4_sha256: str,
    cfg: EasyDict,
    metrics: Mapping[str, Any],
    seed: int,
) -> Dict[str, Any]:
    return build_stage3d_support_checkpoint_payload(
        support_head=head,
        optimizer=optimizer,
        epoch=epoch,
        e4_checkpoint=str(e4_checkpoint.resolve()),
        e4_checkpoint_sha256=e4_sha256,
        config_snapshot=_plain(cfg),
        metrics=metrics,
        stage="3D-B0",
        metadata={
            "seed": int(seed),
            "support_input": (
                "concat(graph_conditioned_query,image_cross_attention_context)"
            ),
            "reads_trajectory_context": False,
            "reads_final_branch_tokens": False,
            "reads_branch_geometry_output": False,
            "changes_e4_trajectory_attention": False,
            "changes_e4_branch_output": False,
            "feeds_anchor": False,
            "feeds_path_push": False,
        },
    )


def train_pre_trajectory_seed(
    *,
    seed: int,
    train_cache: FrozenSupportDataset,
    val_cache: FrozenSupportDataset,
    cfg: EasyDict,
    device: torch.device,
    output_dir: Path,
    e4_checkpoint: Path,
    e4_sha256: str,
) -> Tuple[TrajectorySupportHead, Dict[str, Any]]:
    """Train one fresh pre-trajectory head and select by validation AP."""

    _set_seed(int(seed))
    training_cfg = cfg.STAGE3D_B0.TRAINING
    head = _build_pre_head(cfg, device)
    optimizer = torch.optim.Adam(
        head.parameters(),
        lr=float(training_cfg.LEARNING_RATE),
        weight_decay=float(training_cfg.WEIGHT_DECAY),
    )
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        train_cache,
        batch_size=int(training_cfg.BATCH_SIZE),
        shuffle=True,
        generator=generator,
        num_workers=int(training_cfg.NUM_WORKERS),
        pin_memory=device.type == "cuda",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    best_ap = -1.0
    best_epoch = -1
    history = []
    started_at = time.perf_counter()
    for epoch in range(1, int(training_cfg.EPOCHS) + 1):
        head.train()
        loss_sum = 0.0
        pair_count = 0
        for cpu_batch in loader:
            batch = _move_flat_batch(cpu_batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(
                batch[PRE_BRANCH_KEY],
                batch["fragment_tokens"],
                batch["fragment_mask"],
            )
            losses = trajectory_support_bce_loss(
                logits,
                batch["support_targets"],
                batch["support_valid"],
                batch["fragment_mask"],
                _matches_from_batch(batch["matched_target_indices"]),
            )
            pairs = int(losses["supervised_pair_count"])
            if pairs == 0:
                continue
            losses["loss"].backward()
            optimizer.step()
            loss_sum += float(losses["loss"].detach()) * pairs
            pair_count += pairs
        validation = evaluate_score_source(
            dataset=val_cache,
            cfg=cfg,
            device=device,
            batch_size=int(training_cfg.VAL_BATCH_SIZE),
            support_head=head,
            branch_key=PRE_BRANCH_KEY,
        )
        record = {
            "epoch": epoch,
            "training_loss": loss_sum / max(pair_count, 1),
            "validation": validation,
        }
        history.append(record)
        payload = _checkpoint_payload(
            head=head,
            optimizer=optimizer,
            epoch=epoch,
            e4_checkpoint=e4_checkpoint,
            e4_sha256=e4_sha256,
            cfg=cfg,
            metrics=validation,
            seed=seed,
        )
        save_stage3d_support_checkpoint(
            checkpoint_dir / "pre_trajectory_support.latest.pth.tar",
            payload,
        )
        if float(validation["support_ap"]) > best_ap:
            best_ap = float(validation["support_ap"])
            best_epoch = epoch
            save_stage3d_support_checkpoint(
                checkpoint_dir / "pre_trajectory_support.best.pth.tar",
                payload,
            )
        print(
            "seed {} epoch {:03d}: train={:.6f} val={:.6f} "
            "AP={:.4f} multi_AP={:.4f} P@8={:.4f} Hit@8={:.4f}"
            .format(
                seed,
                epoch,
                record["training_loss"],
                validation["loss"],
                validation["support_ap"],
                validation["by_gt_branch_count"][">=2"]["support_ap"],
                validation["precision_at"]["8"],
                validation["hit_at"]["8"],
            ),
            flush=True,
        )
    best_path = (
        checkpoint_dir / "pre_trajectory_support.best.pth.tar")
    load_stage3d_support_checkpoint(
        best_path,
        support_head=head,
        optimizer=None,
        map_location=device,
    )
    best_metrics = evaluate_score_source(
        dataset=val_cache,
        cfg=cfg,
        device=device,
        batch_size=int(training_cfg.VAL_BATCH_SIZE),
        support_head=head,
        branch_key=PRE_BRANCH_KEY,
    )
    report = {
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_checkpoint": str(best_path.resolve()),
        "best_validation": best_metrics,
        "history": history,
        "elapsed_seconds": float(time.perf_counter() - started_at),
        "fresh_initialization": True,
        "optimizer_resume_used": False,
        "only_pre_trajectory_support_head_trainable": True,
    }
    _write_json(output_dir / "training_report.json", report)
    return head, report


def collect_segment_only_examples(
    *,
    dataset: Stage3CBranchDataset,
    cfg: EasyDict,
    max_examples: int,
) -> Dict[str, Any]:
    """Collect deterministic component scores without changing thresholds."""

    if max_examples < 0:
        raise ValueError("max_examples must be non-negative")
    examples = []
    fragment_count = 0
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.STAGE3D.TRAINING.CACHE_BATCH_SIZE),
        shuffle=False,
        num_workers=0,
    )
    for batch in loader:
        support = build_trajectory_support_targets(
            batch["trajectory_batch"],
            batch["branch_targets"],
            **_target_parameters(cfg),
        )
        segment_only = batch["trajectory_batch"]["segment_only"].bool()
        fragment_mask = batch["trajectory_batch"]["fragment_mask"].bool()
        branch_mask = batch["branch_targets"]["branch_mask"].bool()
        selected_fragments = segment_only & fragment_mask
        fragment_count += int(selected_fragments.sum())
        if len(examples) >= max_examples:
            continue
        for batch_index, fragment_index in torch.nonzero(
                selected_fragments, as_tuple=False).tolist():
            for target_index in torch.nonzero(
                    branch_mask[batch_index],
                    as_tuple=False).flatten().tolist():
                if len(examples) >= max_examples:
                    break
                distance = support["minimum_distance_pixels"][
                    batch_index, target_index, fragment_index]
                examples.append({
                    "sample_id": int(
                        batch["metadata"]["dataset_index"][batch_index]),
                    "gt_branch_index": int(target_index),
                    "fragment_index": int(fragment_index),
                    "track_index": int(
                        batch["trajectory_batch"]["track_indices"][
                            batch_index, fragment_index]),
                    "distance_pixels": (
                        float(distance) if torch.isfinite(distance)
                        else None),
                    "distance_score": float(support["distance_score"][
                        batch_index, target_index, fragment_index]),
                    "axis_score": float(support["axis_score"][
                        batch_index, target_index, fragment_index]),
                    "coverage_score": float(support["coverage_score"][
                        batch_index, target_index, fragment_index]),
                    "support_target": float(support["support_targets"][
                        batch_index, target_index, fragment_index]),
                    "positive": bool(support["support_positive_mask"][
                        batch_index, target_index, fragment_index]),
                    "branch_support_valid": bool(support["support_valid"][
                        batch_index, target_index]),
                })
    return {
        "segment_only_fragment_count": fragment_count,
        "example_count": len(examples),
        "examples": examples,
        "thresholds_unchanged": True,
    }


def _mean_std(values: Sequence[float]) -> Dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "values": array.tolist(),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
    }


def aggregate_pre_seed_metrics(
    reports: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    metrics = [
        report["best_validation"] for report in reports]
    invalid_top1_means = []
    invalid_top1_p90s = []
    for value in metrics:
        top1 = value["support_invalid"]["top1_probability"]
        invalid_top1_means.append(
            float(top1["mean"]) if top1["mean"] is not None else 0.0)
        invalid_top1_p90s.append(
            float(top1["p90"]) if top1["p90"] is not None else 0.0)
    return {
        "support_ap": _mean_std([
            value["support_ap"] for value in metrics]),
        "multibranch_support_ap": _mean_std([
            value["by_gt_branch_count"][">=2"]["support_ap"]
            for value in metrics
        ]),
        "precision_at_8": _mean_std([
            value["precision_at"]["8"] for value in metrics]),
        "hit_at_8": _mean_std([
            value["hit_at"]["8"] for value in metrics]),
        "predicted_top8_jaccard_mean": _mean_std([
            value["predicted_top_k_jaccard"]["mean"]
            for value in metrics
        ]),
        "predicted_top8_jaccard_median": _mean_std([
            value["predicted_top_k_jaccard"]["median"]
            for value in metrics
        ]),
        "support_invalid_max_probability": _mean_std([
            value["support_invalid"]["max_probability"]
            for value in metrics
        ]),
        "support_invalid_top1_mean_probability":
            _mean_std(invalid_top1_means),
        "support_invalid_top1_p90_probability":
            _mean_std(invalid_top1_p90s),
    }


def evaluate_acceptance(
    *,
    aggregate: Mapping[str, Any],
    raw_attention: Mapping[str, Any],
    cfg: EasyDict,
) -> Dict[str, Any]:
    acceptance = cfg.STAGE3D_B0.ACCEPTANCE
    pre_ap = float(aggregate["support_ap"]["mean"])
    raw_ap = float(raw_attention["support_ap"])
    pre_multi = float(
        aggregate["multibranch_support_ap"]["mean"])
    raw_multi = float(
        raw_attention["by_gt_branch_count"][">=2"]["support_ap"])
    checks = {
        "pre_support_ap": (
            pre_ap >= float(acceptance.MIN_PRE_SUPPORT_AP)),
        "raw_attention_ap_gain": (
            pre_ap - raw_ap
            >= float(acceptance.MIN_RAW_ATTENTION_AP_GAIN)),
        "multibranch_raw_attention_ap_gain": (
            pre_multi - raw_multi
            >= float(
                acceptance.MIN_MULTIBRANCH_RAW_ATTENTION_AP_GAIN)),
        "three_seed_stability": (
            float(aggregate["support_ap"]["std"])
            <= float(acceptance.MAX_SUPPORT_AP_STD)),
        "top_k_branch_separation": (
            float(aggregate[
                "predicted_top8_jaccard_median"]["mean"])
            <= float(
                acceptance.MAX_PREDICTED_TOP8_JACCARD_MEDIAN)),
        "support_invalid_not_generally_high": (
            float(aggregate[
                "support_invalid_top1_mean_probability"]["mean"])
            <= float(
                acceptance.MAX_INVALID_TOP1_MEAN_PROBABILITY)
            and float(aggregate[
                "support_invalid_top1_p90_probability"]["mean"])
            <= float(
                acceptance.MAX_INVALID_TOP1_P90_PROBABILITY)),
    }
    return {
        "checks": checks,
        "passed": bool(all(checks.values())),
        "recommend_support_guided_aggregation": bool(all(checks.values())),
        "observed": {
            "pre_support_ap_mean": pre_ap,
            "raw_attention_support_ap": raw_ap,
            "pre_minus_raw_attention_ap": pre_ap - raw_ap,
            "pre_multibranch_support_ap_mean": pre_multi,
            "raw_attention_multibranch_support_ap": raw_multi,
            "multibranch_ap_gain": pre_multi - raw_multi,
        },
        "thresholds": _plain(acceptance),
    }


def _write_readme(
    output_dir: Path,
    comparison: Mapping[str, Any],
) -> None:
    raw = comparison["raw_attention"]
    post = comparison["post_fusion_support"]
    aggregate = comparison["pre_trajectory_stability"]
    decision = comparison["acceptance"]
    lines = [
        "# road_self Stage 3D-B0",
        "",
        "This experiment tests a non-circular trajectory selector before "
        "E4 reads trajectory context. E4 remains frozen and unchanged.",
        "",
        "## Fair comparison",
        "",
        "| source | support AP | multi-branch AP | P@8 | Hit@8 |",
        "| --- | ---: | ---: | ---: | ---: |",
        "| raw_attention | {:.4f} | {:.4f} | {:.4f} | {:.4f} |"
        .format(
            raw["support_ap"],
            raw["by_gt_branch_count"][">=2"]["support_ap"],
            raw["precision_at"]["8"],
            raw["hit_at"]["8"],
        ),
        "| post_fusion_support | {:.4f} | {:.4f} | {:.4f} | {:.4f} |"
        .format(
            post["support_ap"],
            post["by_gt_branch_count"][">=2"]["support_ap"],
            post["precision_at"]["8"],
            post["hit_at"]["8"],
        ),
        "| pre_trajectory_support (3-seed mean) | {:.4f} | {:.4f} | "
        "{:.4f} | {:.4f} |".format(
            aggregate["support_ap"]["mean"],
            aggregate["multibranch_support_ap"]["mean"],
            aggregate["precision_at_8"]["mean"],
            aggregate["hit_at_8"]["mean"],
        ),
        "",
        "Pre-trajectory support AP std: **{:.4f}**.".format(
            aggregate["support_ap"]["std"]),
        "Predicted top-8 Jaccard median mean: **{:.4f}**.".format(
            aggregate["predicted_top8_jaccard_median"]["mean"]),
        "Support-invalid top-1 probability mean: **{:.4f}**.".format(
            aggregate[
                "support_invalid_top1_mean_probability"]["mean"]),
        "",
        "## Decision",
        "",
        "Acceptance: **{}**.".format(
            "passed" if decision["passed"] else "failed"),
        "Support-guided aggregation is **{}**.".format(
            "recommended for the next isolated experiment"
            if decision["recommend_support_guided_aggregation"]
            else "not recommended; stop at Stage 3D-B0"),
        "",
        "The support scores were not fed into E4 attention, branch outputs, "
        "anchor prediction, or Path.push.",
    ]
    (output_dir / "README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/stage3d_b0_pretrajectory_support.yml"),
    )
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--e4-checkpoint", type=Path)
    parser.add_argument("--post-fusion-checkpoint", type=Path)
    parser.add_argument("--image-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument(
        "--resume-completed-seeds",
        action="store_true",
        help=(
            "Reuse a seed only when its complete training report and best "
            "checkpoint exist. The default always starts fresh."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = _load_config(args.config)
    if "STAGE3D_B0" not in cfg:
        raise ValueError("config must define STAGE3D_B0")
    device = _resolve_device(
        args.device or str(cfg.STAGE3C.DEVICE))
    dataset_dir = (
        args.dataset_dir or Path(cfg.STAGE3C.DATASET_DIR))
    output_dir = (
        args.output_dir or Path(cfg.STAGE3D_B0.OUTPUT_DIR))
    e4_checkpoint = (
        args.e4_checkpoint or Path(cfg.STAGE3D.E4_CHECKPOINT))
    post_checkpoint = (
        args.post_fusion_checkpoint
        or Path(cfg.STAGE3D_B0.POST_FUSION_CHECKPOINT))
    image_checkpoint = (
        args.image_checkpoint or Path(cfg.STAGE3C.IMAGE_CHECKPOINT))
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = Stage3CBranchDataset(
        dataset_dir, "train", preload=True)
    val_dataset = Stage3CBranchDataset(
        dataset_dir, "val", preload=True)
    if len(train_dataset) != 2048 or len(val_dataset) != 512:
        raise RuntimeError("Stage 3D-B0 requires the unchanged 2048/512 split")

    label_report = run_label_diagnostics(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        cfg=cfg,
    )
    label_report["bounded_64_branch_support_hit_rate"] = (
        label_report["combined"][
            "bounded_64_branch_support_hit_rate"])
    label_report["segment_only_diagnostics"] = (
        collect_segment_only_examples(
            dataset=val_dataset,
            cfg=cfg,
            max_examples=int(
                cfg.STAGE3D_B0.EVALUATION
                .SEGMENT_ONLY_EXAMPLE_COUNT),
        )
    )
    _write_json(output_dir / "label_diagnostics.json", label_report)
    if not label_report["gate"]["passed"]:
        raise RuntimeError(
            "the unchanged Stage 3D-A support label gate failed")

    e4_checkpoint = e4_checkpoint.resolve(strict=False)
    post_checkpoint = post_checkpoint.resolve(strict=False)
    if not e4_checkpoint.is_file():
        raise FileNotFoundError(
            "E4 checkpoint not found: {}".format(e4_checkpoint))
    if not post_checkpoint.is_file():
        raise FileNotFoundError(
            "post-fusion checkpoint not found: {}".format(
                post_checkpoint))
    e4_sha256 = _sha256(e4_checkpoint)
    rpnet, _ = _load_frozen_rpnet(
        cfg, image_checkpoint, device)
    modules = _build_auxiliary_modules(cfg, device)
    e4_payload = load_stage3c_checkpoint(
        e4_checkpoint,
        trajectory_encoder=modules[0],
        graph_state_encoder=modules[1],
        branch_decoder=modules[2],
        optimizer=None,
        map_location=device,
    )
    _freeze_e4_modules(modules)
    criterion = _build_branch_criterion(cfg)
    train_cache, train_cache_report = build_frozen_support_cache(
        dataset=train_dataset,
        rpnet=rpnet,
        modules=modules,
        criterion=criterion,
        cfg=cfg,
        device=device,
    )
    val_cache, val_cache_report = build_frozen_support_cache(
        dataset=val_dataset,
        rpnet=rpnet,
        modules=modules,
        criterion=criterion,
        cfg=cfg,
        device=device,
    )
    expected_pre_dim = int(
        cfg.STAGE3D_B0.MODEL.BRANCH_INPUT_DIM)
    if train_cache.tensors[PRE_BRANCH_KEY].shape[-1] != expected_pre_dim:
        raise RuntimeError("pre-trajectory branch input dimension mismatch")
    _write_json(output_dir / "frozen_cache_report.json", {
        "train": train_cache_report,
        "validation": val_cache_report,
        "e4_checkpoint": str(e4_checkpoint),
        "e4_checkpoint_sha256": e4_sha256,
        "e4_epoch": int(e4_payload["epoch"]),
        "pre_trajectory_definition": (
            "concat(debug_graph_conditioned_queries,"
            "debug_image_cross_attention_output)"
        ),
        "e4_strict_load": True,
        "e4_frozen": True,
        "rpnet_strict_and_frozen": True,
        "forbidden_inputs_absent_from_pre_head": {
            "trajectory_context": True,
            "final_branch_tokens": True,
            "branch_offsets_or_directions": True,
        },
    })

    evaluation_batch_size = int(
        cfg.STAGE3D_B0.TRAINING.VAL_BATCH_SIZE)
    raw_attention = evaluate_score_source(
        dataset=val_cache,
        cfg=cfg,
        device=device,
        batch_size=evaluation_batch_size,
        score_key=RAW_ATTENTION_KEY,
    )
    post_head = TrajectorySupportHead(
        hidden_dim=int(cfg.STAGE3C.MODEL.HIDDEN_DIM),
        projection_dim=int(cfg.STAGE3D.MODEL.PROJECTION_DIM),
    ).to(device=device)
    post_payload = load_stage3d_support_checkpoint(
        post_checkpoint,
        support_head=post_head,
        optimizer=None,
        map_location=device,
    )
    if post_payload["e4_checkpoint_sha256"] != e4_sha256:
        raise ValueError(
            "post-fusion support checkpoint was trained on another E4")
    post_fusion = evaluate_score_source(
        dataset=val_cache,
        cfg=cfg,
        device=device,
        batch_size=evaluation_batch_size,
        support_head=post_head,
        branch_key=POST_BRANCH_KEY,
    )

    reports = []
    seeds = [
        int(value)
        for value in cfg.STAGE3D_B0.STABILITY.SEEDS
    ]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("Stage 3D-B0 requires exactly three unique seeds")
    for seed in seeds:
        seed_output_dir = output_dir / "seed_{}".format(seed)
        completed_report_path = (
            seed_output_dir / "training_report.json")
        completed_checkpoint_path = (
            seed_output_dir / "checkpoints"
            / "pre_trajectory_support.best.pth.tar")
        if (
                args.resume_completed_seeds
                and completed_report_path.is_file()
                and completed_checkpoint_path.is_file()):
            with completed_report_path.open(
                    "r", encoding="utf-8") as file_obj:
                completed_report = json.load(file_obj)
            if int(completed_report.get("seed", -1)) != seed:
                raise RuntimeError(
                    "completed seed report identity mismatch: {}".format(
                        completed_report_path))
            reports.append(completed_report)
            print(
                "seed {}: reusing completed report {}".format(
                    seed, completed_report_path),
                flush=True,
            )
            continue
        _, report = train_pre_trajectory_seed(
            seed=seed,
            train_cache=train_cache,
            val_cache=val_cache,
            cfg=cfg,
            device=device,
            output_dir=seed_output_dir,
            e4_checkpoint=e4_checkpoint,
            e4_sha256=e4_sha256,
        )
        reports.append(report)
    aggregate = aggregate_pre_seed_metrics(reports)
    decision = evaluate_acceptance(
        aggregate=aggregate,
        raw_attention=raw_attention,
        cfg=cfg,
    )
    comparison = {
        "schema_version": "stage3d-b0-v1",
        "raw_attention": raw_attention,
        "post_fusion_support": post_fusion,
        "pre_trajectory_seed_reports": reports,
        "pre_trajectory_stability": aggregate,
        "acceptance": decision,
        "same_labels_matches_and_split": True,
        "e4_checkpoint": str(e4_checkpoint),
        "post_fusion_checkpoint": str(post_checkpoint),
        "constraints": {
            "e4_strict_and_frozen": True,
            "rpnet_strict_and_frozen": True,
            "original_trajectory_attention_unchanged": True,
            "branch_outputs_unchanged": True,
            "support_scores_used_for_aggregation": False,
            "anchor_unchanged": True,
            "path_push_unchanged": True,
        },
    }
    _write_json(output_dir / "comparison.json", comparison)
    _write_readme(output_dir, comparison)
    print(json.dumps({
        "raw_attention_ap": raw_attention["support_ap"],
        "post_fusion_support_ap": post_fusion["support_ap"],
        "pre_trajectory_support_ap": aggregate["support_ap"],
        "acceptance": decision,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
