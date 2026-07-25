"""Run the requested three-seed E4 stability check for Stage 3D-A."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_stage3c_branch_aux import run_diagnostics  # noqa: E402
from train_branch_aux import (  # noqa: E402
    _load_config,
    _load_frozen_rpnet,
    _resolve_device,
    _set_seed,
    run_formal_training,
)
from utils.stage3c_branch_dataset import Stage3CBranchDataset  # noqa: E402


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(
            value, output_file, ensure_ascii=False,
            indent=2, sort_keys=True)
        output_file.write("\n")


def _mean_std(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "values": array.tolist(),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
    }


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage3d_a_support.yml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_self/stage3d_a/e4_stability"),
    )
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--image-checkpoint", type=Path)
    parser.add_argument("--device")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="reuse existing seed checkpoints and run diagnostics only",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = _load_config(args.config)
    seeds = [int(value) for value in cfg.STAGE3D.STABILITY.SEEDS]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("Stage 3D-A requires exactly three unique seeds")
    device = _resolve_device(
        args.device or str(cfg.STAGE3C.DEVICE))
    dataset_dir = (
        args.dataset_dir or Path(cfg.STAGE3C.DATASET_DIR))
    image_checkpoint = (
        args.image_checkpoint or Path(cfg.STAGE3C.IMAGE_CHECKPOINT))
    train_dataset = Stage3CBranchDataset(
        dataset_dir, "train", preload=True)
    val_dataset = Stage3CBranchDataset(
        dataset_dir, "val", preload=True)
    if len(train_dataset) != 2048 or len(val_dataset) != 512:
        raise RuntimeError(
            "E4 stability requires the unchanged 2048/512 split")
    rpnet, _ = _load_frozen_rpnet(
        cfg, image_checkpoint, device)
    output_root = args.output_dir.resolve(strict=False)
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    started_at = time.perf_counter()
    for seed in seeds:
        seed_cfg = copy.deepcopy(cfg)
        seed_cfg.STAGE3C.SEED = seed
        seed_output = output_root / "seed_{}".format(seed)
        checkpoint = (
            seed_output / "checkpoints"
            / "stage3c_aux.best.pth.tar")
        if not args.skip_training:
            if seed_output.exists() and any(seed_output.iterdir()):
                raise FileExistsError(
                    "seed output already exists: {}".format(seed_output))
            _set_seed(seed)
            training = run_formal_training(
                rpnet=rpnet,
                cfg=seed_cfg,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                device=device,
                output_dir=seed_output,
                image_checkpoint=image_checkpoint,
                resume=None,
            )
        else:
            report_path = seed_output / "training_report.json"
            if not report_path.is_file():
                raise FileNotFoundError(
                    "training report not found: {}".format(report_path))
            with report_path.open(
                    "r", encoding="utf-8") as input_file:
                training = json.load(input_file)
        diagnostics = run_diagnostics(
            cfg=seed_cfg,
            checkpoint=checkpoint,
            image_checkpoint=image_checkpoint,
            dataset_dir=dataset_dir,
            output_dir=seed_output / "diagnostics",
            device=device,
            batch_size=int(
                seed_cfg.STAGE3C.TRAINING.VAL_BATCH_SIZE),
            split="val",
        )
        full = diagnostics["metrics_by_modality"]["full"]
        no_trajectory = diagnostics[
            "metrics_by_modality"]["no_trajectory"]
        result = {
            "seed": seed,
            "best_epoch": int(training["best_epoch"]),
            "checkpoint": str(checkpoint.resolve()),
            "full_branch_ap": float(full["branch_ap"]),
            "no_trajectory_branch_ap": float(
                no_trajectory["branch_ap"]),
            "full_minus_no_trajectory_branch_ap": float(
                full["branch_ap"] - no_trajectory["branch_ap"]),
            "oracle_k_duplicate_ratio": float(
                full["oracle_k_duplicate_ratio"]),
            "oracle_k_distinct_gt_coverage": float(
                full["oracle_k"]["distinct_gt_coverage"]),
            "graph_only_branch_ap": float(
                diagnostics["metrics_by_modality"][
                    "graph_only"]["branch_ap"]),
            "trajectory_graph_branch_ap": float(
                diagnostics["metrics_by_modality"][
                    "trajectory_graph"]["branch_ap"]),
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    summary = {
        "schema_version": "stage3d-a-e4-stability-v1",
        "seeds": seeds,
        "same_train_validation_split": True,
        "fresh_initialization_per_seed": True,
        "optimizer_resume_used": False,
        "rpnet_strict_and_frozen": True,
        "trajectory_dropout": float(
            cfg.STAGE3C.TRAINING.TRAJECTORY_MODALITY_DROPOUT),
        "max_fragments": int(
            cfg.STAGE3C.DATASET.MAX_FRAGMENTS),
        "results": results,
        "aggregate": {
            "full_branch_ap": _mean_std([
                value["full_branch_ap"] for value in results]),
            "no_trajectory_branch_ap": _mean_std([
                value["no_trajectory_branch_ap"] for value in results]),
            "full_minus_no_trajectory_branch_ap": _mean_std([
                value["full_minus_no_trajectory_branch_ap"]
                for value in results]),
            "oracle_k_duplicate_ratio": _mean_std([
                value["oracle_k_duplicate_ratio"] for value in results]),
            "oracle_k_distinct_gt_coverage": _mean_std([
                value["oracle_k_distinct_gt_coverage"]
                for value in results]),
        },
        "elapsed_seconds": float(time.perf_counter() - started_at),
        "support_head_used": False,
        "branch_predictions_feed_path_push": False,
    }
    _write_json(output_root / "stability_summary.json", summary)
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
