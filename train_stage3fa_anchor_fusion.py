"""Train only the Stage 3F-A zero-initialized anchor residual fusion."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


STAGE3FA_LOSS_DEFINITION = {
    "name": "official_vecroad_binary_cross_entropy_with_logits",
    "prediction_target_order": "prediction_then_target",
    "spatial_and_recursive_reduction": "sum_per_sample",
    "batch_reduction": "mean",
    "anchor_lowrs_weight": 1.0,
}

from model.trajectory_anchor_fusion import (  # noqa: E402
    ZeroInitializedTrajectoryAnchorFusion,
    fuse_cached_anchor_logits,
)
from train_branch_aux import _load_config, _load_frozen_rpnet  # noqa: E402
from utils.stage3fa_anchor_cache import (  # noqa: E402
    ShardLocalShuffleSampler, Stage3FAAnchorDataset,
    stage3fa_collate,
)
from utils.stage3fa_checkpoint import (  # noqa: E402
    build_stage3fa_checkpoint_payload,
    load_stage3fa_checkpoint,
    save_stage3fa_checkpoint,
)
from utils.stage3fa_loss import original_vecroad_anchor_losses  # noqa: E402


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(_plain(value), output, indent=2, sort_keys=True)
        output.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for block in iter(lambda: input_file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _module_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(memoryview(value.detach().cpu().contiguous().numpy()))
    return digest.hexdigest()


def _resolve(path: Any) -> Path:
    value = Path(str(path))
    return value if value.is_absolute() else ROOT / value


def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_fusion(cfg, device: torch.device) -> ZeroInitializedTrajectoryAnchorFusion:
    model_cfg = cfg.STAGE3FA.MODEL
    return ZeroInitializedTrajectoryAnchorFusion(
        evidence_dim=int(model_cfg.EVIDENCE_DIM),
        anchor_channels=int(model_cfg.ANCHOR_CHANNELS),
        gate_hidden_dim=int(model_cfg.GATE_HIDDEN_DIM)).to(device)


def _move(batch: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device=device, non_blocking=True)
            for key, value in batch.items()}


def _loss(
    anchor: torch.Tensor, anchor_lowrs: torch.Tensor,
    target: torch.Tensor, end_indices: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    return original_vecroad_anchor_losses(
        anchor, anchor_lowrs, target, end_indices)


def _forward(
    fusion, batch, anchor_weight, lowrs_weight,
    *, availability_override: Optional[torch.Tensor] = None,
    evidence_key: str = "trajectory_evidence",
) -> Dict[str, torch.Tensor]:
    available = (batch["trajectory_available"] if availability_override is None
                 else availability_override)
    return fuse_cached_anchor_logits(
        fusion=fusion,
        anchor_features=batch["anchor_features"].float(),
        anchor_lowrs_features=batch["anchor_lowrs_features"].float(),
        original_anchor_logits=batch["original_anchor_logits"].float(),
        original_anchor_lowrs_logits=batch["original_anchor_lowrs_logits"].float(),
        trajectory_evidence=batch[evidence_key].float(),
        trajectory_available=available,
        anchor_head_weight=anchor_weight,
        anchor_lowrs_head_weight=lowrs_weight)


def _run_epoch(
    *, fusion, loader, anchor_weight, lowrs_weight, device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    gradient_clip: float = 0.0,
) -> Dict[str, float]:
    training = optimizer is not None
    fusion.train(training)
    totals = {"anchor_loss": 0.0, "anchor_lowrs_loss": 0.0,
              "anchor_total_loss": 0.0}
    sample_count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for cpu_batch in loader:
            batch = _move(cpu_batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            outputs = _forward(fusion, batch, anchor_weight, lowrs_weight)
            losses = _loss(
                outputs["anchor"], outputs["anchor_lowrs"],
                batch["anchor_target"].float(),
                batch["supervision_end_index"])
            if not bool(torch.isfinite(losses["anchor_total_loss"])):
                raise RuntimeError("non-finite Stage 3F-A loss")
            if training:
                losses["anchor_total_loss"].backward()
                if not any(parameter.grad is not None for parameter in fusion.parameters()):
                    raise RuntimeError("fusion module received no gradients")
                if gradient_clip > 0:
                    clip_grad_norm_(fusion.parameters(), gradient_clip)
                optimizer.step()
            count = int(batch["sample_id"].shape[0])
            sample_count += count
            for name in totals:
                totals[name] += float(losses[name].detach()) * count
    return {name: value / max(sample_count, 1) for name, value in totals.items()}


def _strict_equivalence(fusion, loader, anchor_weight, lowrs_weight,
                        device, tolerance: float) -> Dict[str, float]:
    fusion.eval()
    maximum_full = maximum_lowrs = maximum_no = 0.0
    with torch.no_grad():
        for cpu_batch in loader:
            batch = _move(cpu_batch, device)
            full = _forward(fusion, batch, anchor_weight, lowrs_weight)
            no = _forward(
                fusion, batch, anchor_weight, lowrs_weight,
                availability_override=torch.zeros_like(
                    batch["trajectory_available"], dtype=torch.float32))
            base = batch["original_anchor_logits"].float()
            base_low = batch["original_anchor_lowrs_logits"].float()
            maximum_full = max(maximum_full,
                float((full["anchor"] - base).abs().max()))
            maximum_lowrs = max(maximum_lowrs,
                float((full["anchor_lowrs"] - base_low).abs().max()))
            maximum_no = max(maximum_no,
                float((no["anchor"] - base).abs().max()),
                float((no["anchor_lowrs"] - base_low).abs().max()))
            break
    passed = max(maximum_full, maximum_lowrs, maximum_no) <= tolerance
    if not passed:
        raise RuntimeError("zero-init/no-trajectory equivalence failed")
    return {"zero_init_anchor_max_abs_diff": maximum_full,
            "zero_init_anchor_lowrs_max_abs_diff": maximum_lowrs,
            "no_trajectory_max_abs_diff": maximum_no,
            "tolerance": tolerance, "passed": True}


def _make_loader(dataset, batch_size: int, shuffle: bool, seed: int,
                 workers: int = 0):
    is_cache_view = (
        isinstance(dataset, Stage3FAAnchorDataset)
        or (isinstance(dataset, Subset)
            and isinstance(dataset.dataset, Stage3FAAnchorDataset)))
    shard_sampler = (
        ShardLocalShuffleSampler(dataset, seed)
        if shuffle and is_cache_view
        else None)
    return DataLoader(
        dataset, batch_size=batch_size,
        shuffle=shuffle and shard_sampler is None, sampler=shard_sampler,
        num_workers=workers, collate_fn=stage3fa_collate,
        pin_memory=torch.cuda.is_available(),
        generator=(torch.Generator().manual_seed(seed)
                   if shuffle and shard_sampler is None else None))


def _train_run(
    *, cfg, fusion, train_dataset, val_dataset, anchor_weight,
    lowrs_weight, device, output_dir: Path, epochs: int,
    learning_rate: float, seed: int, checkpoint_metadata,
    checkpoint_prefix: str, batch_size: Optional[int] = None,
    val_batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    training_cfg = cfg.STAGE3FA.TRAINING
    optimizer = torch.optim.AdamW(
        fusion.parameters(), lr=learning_rate,
        weight_decay=float(training_cfg.WEIGHT_DECAY))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(epochs)))
    train_loader = _make_loader(
        train_dataset, int(batch_size or training_cfg.BATCH_SIZE), True, seed,
        int(training_cfg.NUM_WORKERS))
    val_loader = _make_loader(
        val_dataset, int(val_batch_size or training_cfg.VAL_BATCH_SIZE),
        False, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    curve_path = output_dir / "training_curve.jsonl"
    with curve_path.open("w", encoding="utf-8"):
        pass
    best_loss = math.inf
    best_epoch = 0
    lowest_train_loss = math.inf
    final_train_metrics = None
    final_validation_metrics = None
    started = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        train_metrics = _run_epoch(
            fusion=fusion, loader=train_loader, anchor_weight=anchor_weight,
            lowrs_weight=lowrs_weight, device=device, optimizer=optimizer,
            gradient_clip=float(training_cfg.GRADIENT_CLIP_NORM))
        val_metrics = _run_epoch(
            fusion=fusion, loader=val_loader, anchor_weight=anchor_weight,
            lowrs_weight=lowrs_weight, device=device)
        record = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"],
                  "train": train_metrics, "validation": val_metrics}
        lowest_train_loss = min(
            lowest_train_loss, train_metrics["anchor_total_loss"])
        final_train_metrics = train_metrics
        final_validation_metrics = val_metrics
        with curve_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")
        payload = build_stage3fa_checkpoint_payload(
            fusion=fusion, optimizer=optimizer, epoch=epoch, seed=seed,
            validation_anchor_total_loss=val_metrics["anchor_total_loss"],
            checkpoint_sha256=checkpoint_metadata["checkpoint_sha256"],
            frozen_module_sha256=checkpoint_metadata["frozen_module_sha256"],
            config_snapshot=_plain(cfg))
        checkpoint_dir = output_dir / "checkpoints"
        save_stage3fa_checkpoint(
            checkpoint_dir / (checkpoint_prefix + ".latest.pth.tar"), payload)
        if val_metrics["anchor_total_loss"] < best_loss:
            best_loss = val_metrics["anchor_total_loss"]
            best_epoch = epoch
            save_stage3fa_checkpoint(
                checkpoint_dir / (checkpoint_prefix + ".best.pth.tar"), payload)
        scheduler.step()
        print("Stage3FA epoch {}/{} train={:.6f} val={:.6f}".format(
            epoch, epochs, train_metrics["anchor_total_loss"],
            val_metrics["anchor_total_loss"]), flush=True)
    best_path = output_dir / "checkpoints" / (checkpoint_prefix + ".best.pth.tar")
    load_stage3fa_checkpoint(best_path, fusion=fusion, map_location=device)
    return {"best_epoch": best_epoch, "best_validation_anchor_total_loss": best_loss,
            "final_epoch": int(epochs), "elapsed_seconds": time.perf_counter() - started,
            "lowest_train_anchor_total_loss": lowest_train_loss,
            "final_train": final_train_metrics,
            "final_validation": final_validation_metrics,
            "best_checkpoint": str(best_path.resolve()),
            "best_checkpoint_sha256": _sha256(best_path)}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = _load_config(args.config)
    seed = int(cfg.STAGE3C.SEED)
    _seed(seed)
    device = torch.device(args.device or cfg.STAGE3C.DEVICE)
    cache_dir = _resolve(cfg.STAGE3FA.CACHE_DIR)
    resident_setting = cfg.STAGE3FA.TRAINING.RESIDENT_CACHE_SHARDS
    cache_shards = (
        None if str(resident_setting).lower() == "all"
        else int(resident_setting))
    train_dataset = Stage3FAAnchorDataset(
        cache_dir, "train", cache_shards=cache_shards)
    val_dataset = Stage3FAAnchorDataset(
        cache_dir, "val", cache_shards=cache_shards)
    manifest = train_dataset.manifest
    image_checkpoint = Path(manifest["checkpoint_paths"]["image"])
    rpnet, _ = _load_frozen_rpnet(cfg, image_checkpoint, device)
    rpnet.eval().requires_grad_(False)
    rpnet_sha_before = _module_sha256(rpnet)
    if rpnet_sha_before != manifest["frozen_module_sha256"]["rpnet"]:
        raise RuntimeError("RPNet state differs from anchor-cache RPNet")
    anchor_weight = rpnet.conv_final.weight.detach()
    lowrs_weight = rpnet.next_step_final.weight.detach()
    fusion = build_fusion(cfg, device)
    tolerance = float(cfg.STAGE3FA.ACCEPTANCE.STRICT_TOLERANCE)
    initial_loader = _make_loader(val_dataset, 1, False, seed)
    initial_equivalence = _strict_equivalence(
        fusion, initial_loader, anchor_weight, lowrs_weight,
        device, tolerance)
    output_dir = _resolve(cfg.STAGE3FA.OUTPUT_DIR)
    sanity_dir = output_dir / "sanity"
    sanity_summary_path = sanity_dir / "training_summary.json"
    if args.skip_sanity:
        if not sanity_summary_path.is_file():
            raise FileNotFoundError(
                "cannot skip Stage 3F-A sanity; report not found: {}".format(
                    sanity_summary_path.resolve(strict=False)))
        with sanity_summary_path.open("r", encoding="utf-8") as input_file:
            sanity = json.load(input_file)
        if not bool(sanity.get("passed")):
            raise RuntimeError(
                "cannot skip Stage 3F-A sanity; saved gate did not pass")
        sanity = dict(sanity)
        sanity["reused_passed_report"] = str(sanity_summary_path.resolve())
    else:
        sanity_train = Subset(train_dataset, range(min(
            int(cfg.STAGE3FA.TRAINING.SANITY_TRAIN_SAMPLES),
            len(train_dataset))))
        sanity_val = Subset(val_dataset, range(min(
            int(cfg.STAGE3FA.TRAINING.SANITY_VAL_SAMPLES), len(val_dataset))))
        sanity_fusion = build_fusion(cfg, device)
        initial_sanity_train = _run_epoch(
            fusion=sanity_fusion,
            loader=_make_loader(sanity_train, 4, False, seed),
            anchor_weight=anchor_weight, lowrs_weight=lowrs_weight, device=device)
        initial_sanity_val = _run_epoch(
            fusion=sanity_fusion,
            loader=_make_loader(sanity_val, 4, False, seed),
            anchor_weight=anchor_weight, lowrs_weight=lowrs_weight, device=device)
        sanity = _train_run(
            cfg=cfg, fusion=sanity_fusion, train_dataset=sanity_train,
            val_dataset=sanity_val, anchor_weight=anchor_weight,
            lowrs_weight=lowrs_weight, device=device, output_dir=sanity_dir,
            epochs=int(cfg.STAGE3FA.TRAINING.SANITY_EPOCHS),
            learning_rate=float(cfg.STAGE3FA.TRAINING.LEARNING_RATE),
            seed=seed, checkpoint_metadata=manifest,
            checkpoint_prefix="stage3fa_sanity",
            batch_size=int(cfg.STAGE3FA.TRAINING.SANITY_BATCH_SIZE),
            val_batch_size=int(cfg.STAGE3FA.TRAINING.SANITY_BATCH_SIZE))
        sanity_reduction = (
            (initial_sanity_train["anchor_total_loss"]
             - sanity["lowest_train_anchor_total_loss"])
            / max(abs(initial_sanity_train["anchor_total_loss"]), 1e-12))
        sanity["initial_train"] = initial_sanity_train
        sanity["initial_validation"] = initial_sanity_val
        sanity["loss_reduction"] = sanity_reduction
        sanity["passed"] = sanity_reduction >= float(
            cfg.STAGE3FA.TRAINING.SANITY_MIN_LOSS_REDUCTION)
        _write_json(sanity_summary_path, sanity)
        if not sanity["passed"]:
            raise RuntimeError(
                "Stage 3F-A 32-sample sanity did not reduce loss enough")
    if args.sanity_only:
        return {"sanity": sanity, "initial_equivalence": initial_equivalence,
                "loss_definition": STAGE3FA_LOSS_DEFINITION}

    _seed(seed)
    fusion = build_fusion(cfg, device)
    formal = _train_run(
        cfg=cfg, fusion=fusion, train_dataset=train_dataset,
        val_dataset=val_dataset, anchor_weight=anchor_weight,
        lowrs_weight=lowrs_weight, device=device, output_dir=output_dir,
        epochs=int(cfg.STAGE3FA.TRAINING.EPOCHS),
        learning_rate=float(cfg.STAGE3FA.TRAINING.LEARNING_RATE),
        seed=seed, checkpoint_metadata=manifest,
        checkpoint_prefix="stage3fa_anchor_fusion")
    rpnet_sha_after = _module_sha256(rpnet)
    if rpnet_sha_after != rpnet_sha_before:
        raise RuntimeError("frozen RPNet changed during Stage 3F-A training")
    formal.update({
        "seed": seed, "canonical_evidence_checkpoint": manifest["checkpoint_paths"]["evidence"],
        "canonical_evidence_checkpoint_sha256": manifest["checkpoint_sha256"]["evidence"],
        "frozen_rpnet_sha_before": rpnet_sha_before,
        "frozen_rpnet_sha_after": rpnet_sha_after,
        "frozen_sha_unchanged": True,
        "initial_equivalence": initial_equivalence,
        "sanity": sanity,
        "train_samples": len(train_dataset), "validation_samples": len(val_dataset),
        "teacher_forced_anchor_validation": True,
        "closed_loop_road_graph_evaluation": False,
        "loss_definition": STAGE3FA_LOSS_DEFINITION,
    })
    _write_json(output_dir / "training_summary.json", formal)
    return formal


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--sanity-only", action="store_true")
    mode.add_argument("--skip-sanity", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(_plain(run(_parse_args())), indent=2, sort_keys=True))
