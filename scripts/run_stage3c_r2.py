"""Run the controlled Stage 3C-R2 E0-E4 generalization comparison."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_stage3c_branch_aux import (  # noqa: E402
    run_diagnostics,
)
from train_branch_aux import (  # noqa: E402
    _load_config,
    _load_frozen_rpnet,
    _resolve_device,
    _set_seed,
    run_formal_training,
)
from utils.stage3c_branch_dataset import (  # noqa: E402
    Stage3CBranchDataset,
)


EXPERIMENT_NAMES = ("E0", "E1", "E2", "E3", "E4")
DEFAULT_CONFIGS = {
    "E0": Path("configs/stage3c_e0_fresh_baseline.yml"),
    "E1": Path("configs/stage3c_e1_existence_matching_only.yml"),
    "E2": Path("configs/stage3c_e2_no_object_weight_only.yml"),
    "E3": Path("configs/stage3c_e3_matching_plus_no_object.yml"),
    "E4": Path("configs/stage3c_e4_matching_no_object_self_attn.yml"),
}
EXPECTED_REPAIR_SETTINGS = {
    "E0": (0.0, 1.0, 0),
    "E1": (1.0, 1.0, 0),
    "E2": (0.0, 0.2, 0),
    "E3": (1.0, 0.2, 0),
    "E4": (1.0, 0.2, 1),
}
MODALITY_LABELS = {
    "full": "full",
    "no_trajectory": "no-trajectory",
    "trajectory_graph": "trajectory+graph",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in EXPERIMENT_NAMES:
        parser.add_argument(
            "--{}-config".format(name.lower()),
            type=Path,
            default=DEFAULT_CONFIGS[name],
        )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_self/stage3c_r2"),
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("docs/stage3c_r2_20260725"),
    )
    parser.add_argument("--device")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="delete only E0-E4 directories below output-dir",
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="reuse existing E0-E4 best checkpoints",
    )
    return parser.parse_args()


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _plain(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(
            _plain(value),
            output,
            indent=2,
            sort_keys=True,
        )
        output.write("\n")


def _value_at(mapping: Mapping[str, Any], path: Sequence[str]):
    value = mapping
    for key in path:
        value = value[key]
    return value


def validate_r2_configs(
    configs: Mapping[str, Mapping[str, Any]],
) -> None:
    if set(configs) != set(EXPERIMENT_NAMES):
        raise ValueError("R2 requires exactly E0, E1, E2, E3 and E4")
    reference = configs["E0"]
    common_paths = (
        ("STAGE3C", "SEED"),
        ("STAGE3C", "IMAGE_CHECKPOINT"),
        ("STAGE3C", "DATASET_DIR"),
        ("STAGE3C", "DATASET", "TRAIN_SAMPLES"),
        ("STAGE3C", "DATASET", "VAL_SAMPLES"),
        ("STAGE3C", "DATASET", "MAX_FRAGMENTS"),
        ("STAGE3C", "MODEL", "HIDDEN_DIM"),
        ("STAGE3C", "MODEL", "NUM_HEADS"),
        ("STAGE3C", "MODEL", "TRAJECTORY_LAYERS"),
        ("STAGE3C", "MODEL", "NUM_QUERIES"),
        ("STAGE3C", "MODEL", "IMAGE_POOL_SIZE"),
        ("STAGE3C", "MODEL", "DROPOUT"),
        ("STAGE3C", "OPTIMIZER", "LEARNING_RATE"),
        ("STAGE3C", "OPTIMIZER", "WEIGHT_DECAY"),
        ("STAGE3C", "TRAINING", "EPOCHS"),
        ("STAGE3C", "TRAINING", "BATCH_SIZE"),
        ("STAGE3C", "TRAINING", "VAL_BATCH_SIZE"),
        (
            "STAGE3C",
            "TRAINING",
            "TRAJECTORY_MODALITY_DROPOUT",
        ),
    )
    for name, cfg in configs.items():
        for path in common_paths:
            if _value_at(cfg, path) != _value_at(reference, path):
                raise ValueError(
                    "{} differs at {}".format(
                        name, ".".join(path)))
        if str(
                cfg.STAGE3C.EVALUATION.MODEL_SELECTION_METRIC
                ).lower() != "branch_ap":
            raise ValueError(
                "{} must select checkpoints by branch_ap".format(name))
        actual = (
            float(cfg.STAGE3C.MATCHING.EXISTENCE_COST_WEIGHT),
            float(cfg.STAGE3C.LOSS.EXIST_NO_OBJECT_COEF),
            int(cfg.STAGE3C.MODEL.QUERY_SELF_ATTENTION_LAYERS),
        )
        if actual != EXPECTED_REPAIR_SETTINGS[name]:
            raise ValueError(
                "{} settings are {}, expected {}".format(
                    name, actual, EXPECTED_REPAIR_SETTINGS[name]))
    if int(reference.STAGE3C.DATASET.TRAIN_SAMPLES) != 2048:
        raise ValueError("R2 requires 2048 configured train samples")
    if int(reference.STAGE3C.DATASET.VAL_SAMPLES) != 512:
        raise ValueError("R2 requires 512 configured validation samples")
    if int(reference.STAGE3C.DATASET.MAX_FRAGMENTS) != 64:
        raise ValueError("R2 requires bounded trajectory budget 64")
    if float(
            reference.STAGE3C.TRAINING
            .TRAJECTORY_MODALITY_DROPOUT) != 0.25:
        raise ValueError("R2 requires trajectory dropout 0.25")


def _experiment_summary(
    *,
    training_report: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    checkpoint: Path,
) -> Dict[str, Any]:
    return {
        "checkpoint": str(checkpoint.resolve()),
        "best_epoch": int(training_report["best_epoch"]),
        "best_selection_score": float(
            training_report["best_selection_score"]),
        "best_validation_loss": float(
            training_report["best_full_validation_loss"]),
        "training_elapsed_seconds": float(
            training_report["elapsed_seconds"]),
        "peak_cuda_memory_bytes":
            training_report["peak_cuda_memory_bytes"],
        "metrics_by_modality": diagnostics[
            "metrics_by_modality"],
        "branch_ap": float(diagnostics["branch_ap"]),
        "slot_ap": float(diagnostics["slot_ap"]),
        "exact_count_accuracy": float(
            diagnostics["exact_count_accuracy"]),
        "oracle_k_recall": float(
            diagnostics["oracle_k"]["recall"]),
        "oracle_k_distinct_gt_coverage": float(
            diagnostics["oracle_k"]["distinct_gt_coverage"]),
        "oracle_k_duplicate_ratio": float(
            diagnostics["oracle_k_duplicate_ratio"]),
        "actual_matched_duplicate_ratio": float(
            diagnostics["matched_duplicate_ratio"]),
        "matched_probability": diagnostics[
            "matched_probability"],
        "unmatched_probability": diagnostics[
            "unmatched_probability"],
        "missed_branch_rate": float(
            diagnostics["missed_branch_rate"]),
        "extra_branch_rate": float(
            diagnostics["extra_branch_rate"]),
        "query_similarity": diagnostics[
            "metrics_by_modality"]["full"]["query_similarity"],
        "metrics_by_gt_count": diagnostics[
            "metrics_by_modality"]["full"][
                "metrics_by_gt_count"],
    }


def build_r2_decisions(
    experiments: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    minimum_meaningful_ap_gain = 0.01

    def full_ap(name):
        return float(experiments[name][
            "metrics_by_modality"]["full"]["branch_ap"])

    matching_delta = full_ap("E1") - full_ap("E0")
    no_object_delta = full_ap("E2") - full_ap("E0")
    combination_delta = full_ap("E3") - full_ap("E0")
    self_attention_delta = full_ap("E4") - full_ap("E3")
    full_advantages = {
        name: float(
            experiments[name]["metrics_by_modality"][
                "full"]["branch_ap"]
            - experiments[name]["metrics_by_modality"][
                "no_trajectory"]["branch_ap"]
        )
        for name in EXPERIMENT_NAMES
    }
    best_name = max(
        EXPERIMENT_NAMES,
        key=lambda name: (
            full_ap(name),
            experiments[name][
                "oracle_k_distinct_gt_coverage"],
            -experiments[name]["oracle_k_duplicate_ratio"],
        ),
    )
    best = experiments[best_name]
    combination_stable = bool(
        full_ap("E3") >= max(full_ap("E1"), full_ap("E2"))
        and best_name in ("E3", "E4")
    )
    full_stably_better = bool(
        all(value > 0.0 for value in full_advantages.values())
        and full_advantages[best_name]
        >= minimum_meaningful_ap_gain
    )
    collapse_remains = bool(
        best["oracle_k_duplicate_ratio"] > 0.20
        or best["oracle_k_distinct_gt_coverage"] < 0.80
    )
    best_full_metrics = best["metrics_by_modality"]["full"]
    ready = bool(
        full_ap(best_name) >= 0.80
        and float(best_full_metrics["slot_ap"]) >= 0.80
        and not collapse_remains
    )
    return {
        "existence_matching_alone_effective":
            matching_delta >= minimum_meaningful_ap_gain,
        "existence_matching_branch_ap_delta": matching_delta,
        "no_object_weighting_alone_effective":
            no_object_delta >= minimum_meaningful_ap_gain,
        "no_object_weighting_branch_ap_delta": no_object_delta,
        "matching_plus_no_object_most_stable":
            combination_stable,
        "matching_plus_no_object_branch_ap_delta":
            combination_delta,
        "self_attention_improves_validation":
            self_attention_delta >= minimum_meaningful_ap_gain,
        "self_attention_branch_ap_delta":
            self_attention_delta,
        "full_branch_ap_advantage_by_experiment":
            full_advantages,
        "full_stably_better_than_no_trajectory":
            full_stably_better,
        "structured_trajectory_increment_demonstrated":
            full_stably_better,
        "best_experiment": best_name,
        "multi_branch_query_collapse_remains":
            collapse_remains,
        "ready_for_trajectory_support_supervision": ready,
        "ready_for_anchor_fusion": False,
        "minimum_meaningful_branch_ap_gain":
            minimum_meaningful_ap_gain,
        "decision_notes": (
            "Effectiveness requires a validation branch-AP gain of at "
            "least 0.01. Stable trajectory benefit additionally "
            "requires a positive full-minus-no-trajectory delta in all "
            "five controlled runs and at least 0.01 in the best run. "
            "Collapse requires oracle-K duplicate <=0.20 and distinct "
            "GT coverage >=0.80 on the best validation checkpoint."
            " Trajectory-support supervision readiness requires branch "
            "AP and slot AP >=0.80 with no query collapse; it does not "
            "claim that trajectory increment is already demonstrated. "
            "Anchor fusion remains out of scope."
        ),
    }


def _render_readme(comparison: Mapping[str, Any]) -> str:
    experiments = comparison["experiments"]
    decisions = comparison["decisions"]
    rows = []
    diagnostic_rows = []
    for name in EXPERIMENT_NAMES:
        result = experiments[name]
        full = result["metrics_by_modality"]["full"]
        no_trajectory = result[
            "metrics_by_modality"]["no_trajectory"]
        trajectory_graph = result[
            "metrics_by_modality"]["trajectory_graph"]
        rows.append(
            "| {name} | {epoch} | {full:.4f} | {no_traj:.4f} | "
            "{traj_graph:.4f} | {slot:.4f} | {exact:.4f} | "
            "{coverage:.4f} | {oracle_dup:.4f} | "
            "{matched_dup:.4f} |".format(
                name=name,
                epoch=result["best_epoch"],
                full=full["branch_ap"],
                no_traj=no_trajectory["branch_ap"],
                traj_graph=trajectory_graph["branch_ap"],
                slot=full["slot_ap"],
                exact=full["exact_count_accuracy"],
                coverage=full["oracle_k"][
                    "distinct_gt_coverage"],
                oracle_dup=full["oracle_k_duplicate_ratio"],
                matched_dup=full[
                    "actual_matched_duplicate_ratio"],
            )
        )
        query_similarity = full["query_similarity"]
        diagnostic_rows.append(
            "| {name} | {matched:.4f} | {unmatched:.4f} | "
            "{missed:.4f} | {extra:.4f} | {graph:.4f} | "
            "{final:.4f} |".format(
                name=name,
                matched=full["matched_probability"]["mean"],
                unmatched=full["unmatched_probability"]["mean"],
                missed=full["missed_branch_rate"],
                extra=full["extra_branch_rate"],
                graph=query_similarity[
                    "debug_graph_conditioned_queries"
                ]["pairwise_cosine"]["mean"],
                final=query_similarity[
                    "debug_final_fused_queries"
                ]["pairwise_cosine"]["mean"],
            )
        )
    best_name = decisions["best_experiment"]
    group_rows = []
    for group_name, group in experiments[best_name][
            "metrics_by_modality"]["full"][
                "metrics_by_gt_count"].items():
        group_rows.append(
            "| {group} | {samples} | {ap:.4f} | {exact:.4f} | "
            "{coverage:.4f} | {duplicate:.4f} |".format(
                group=group_name,
                samples=group["sample_count"],
                ap=group.get("branch_ap", 0.0),
                exact=group.get(
                    "thresholded_metrics", {},
                ).get("exact_branch_count_accuracy", 0.0),
                coverage=group.get(
                    "oracle_k", {},
                ).get("distinct_gt_coverage", 0.0),
                duplicate=group.get(
                    "duplicates", {},
                ).get("oracle_k", {}).get(
                    "duplicate_pair_ratio", 0.0),
            )
        )
    answer = lambda value: "Yes" if value else "No"
    return "\n".join((
        "# Stage 3C-R2 formal E0-E4 comparison",
        "",
        "All five experiments use the same 2048/512 spatial split, "
        "seed, frozen strict RPNet checkpoint, 30 epochs, optimizer "
        "settings, trajectory dropout 0.25, and bounded trajectory "
        "budget 64. Every auxiliary run starts fresh.",
        "",
        "Summed training time: {:.1f} seconds; peak CUDA memory was "
        "approximately {:.2f} GiB.".format(
            comparison["training_elapsed_seconds"],
            max(
                float(result["peak_cuda_memory_bytes"] or 0)
                for result in experiments.values()
            ) / float(1024 ** 3),
        ),
        "",
        "| Experiment | Best epoch | Full AP | No-traj AP | "
        "Trajectory+graph AP | Slot AP | Exact count | "
        "Distinct coverage | Oracle duplicate | Matched duplicate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## Full-modality diagnostics",
        "",
        "| Experiment | Matched probability | Unmatched probability | "
        "Missed rate | Extra rate | Graph-query cosine | "
        "Final-query cosine |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *diagnostic_rows,
        "",
        "## Best experiment grouped by GT branch count",
        "",
        "Best experiment: **{}**.".format(best_name),
        "",
        "| GT group | Samples | Branch AP | Exact count | "
        "Distinct coverage | Oracle duplicate |",
        "|---|---:|---:|---:|---:|---:|",
        *group_rows,
        "",
        "## Decisions",
        "",
        "1. Existence matching alone effective? **{}** "
        "(AP delta {:+.4f})".format(
            answer(decisions[
                "existence_matching_alone_effective"]),
            decisions[
                "existence_matching_branch_ap_delta"],
        ),
        "2. No-object weighting alone effective? **{}** "
        "(AP delta {:+.4f})".format(
            answer(decisions[
                "no_object_weighting_alone_effective"]),
            decisions[
                "no_object_weighting_branch_ap_delta"],
        ),
        "3. Matching + no-object most stable? **{}**".format(
            answer(decisions[
                "matching_plus_no_object_most_stable"])),
        "4. Self-attention improves validation? **{}** "
        "(E4-E3 AP {:+.4f})".format(
            answer(decisions[
                "self_attention_improves_validation"]),
            decisions["self_attention_branch_ap_delta"],
        ),
        "5. Full stably better than no-trajectory? **{}**".format(
            answer(decisions[
                "full_stably_better_than_no_trajectory"])),
        "6. Multi-branch query collapse remains? **{}**".format(
            answer(decisions[
                "multi_branch_query_collapse_remains"])),
        "7. Ready for trajectory-support supervision? **{}**".format(
            answer(decisions[
                "ready_for_trajectory_support_supervision"])),
        "8. Ready for anchor fusion? **{}**".format(
            answer(decisions["ready_for_anchor_fusion"])),
        "",
        "Best validation experiment: **{}**.".format(
            decisions["best_experiment"]),
        "",
        "Per-modality, GT-count-grouped, probability, duplicate, and "
        "query-representation statistics are preserved in "
        "`comparison.json` and each experiment directory.",
        "",
        "No branch endpoint was passed to `Path.push`; RPNet, anchor, "
        "trajectory encoding, sampling, branch GT, and decoder structure "
        "were not changed.",
        "",
    ))


def _copy_report_artifacts(
    *,
    experiment_output: Path,
    docs_output: Path,
) -> None:
    if docs_output.exists():
        shutil.rmtree(str(docs_output))
    docs_output.mkdir(parents=True, exist_ok=True)
    for filename in (
            "training_report.json",
            "training_curves.png",
            "run_metadata.json"):
        source = experiment_output / filename
        if source.is_file():
            shutil.copy2(str(source), str(docs_output / filename))
    diagnostics = experiment_output / "diagnostics"
    if diagnostics.is_dir():
        shutil.copytree(
            str(diagnostics),
            str(docs_output / "diagnostics"),
        )


def main() -> None:
    args = _parse_args()
    config_paths = {
        name: getattr(args, "{}_config".format(name.lower()))
        for name in EXPERIMENT_NAMES
    }
    configs = {
        name: _load_config(path)
        for name, path in config_paths.items()
    }
    validate_r2_configs(configs)
    reference = configs["E0"]
    device = _resolve_device(
        args.device or str(reference.STAGE3C.DEVICE))
    dataset_dir = Path(reference.STAGE3C.DATASET_DIR)
    image_checkpoint = Path(reference.STAGE3C.IMAGE_CHECKPOINT)
    train_dataset = Stage3CBranchDataset(
        dataset_dir, "train", preload=True)
    val_dataset = Stage3CBranchDataset(
        dataset_dir, "val", preload=True)
    if len(train_dataset) != 2048 or len(val_dataset) != 512:
        raise RuntimeError(
            "R2 requires actual dataset sizes 2048/512, got {}/{}"
            .format(len(train_dataset), len(val_dataset)))
    rpnet, _ = _load_frozen_rpnet(
        reference, image_checkpoint, device)
    if not all(
            not parameter.requires_grad
            for parameter in rpnet.parameters()):
        raise RuntimeError("RPNet must remain frozen")

    output_root = args.output_dir.resolve(strict=False)
    docs_root = args.docs_dir.resolve(strict=False)
    output_root.mkdir(parents=True, exist_ok=True)
    docs_root.mkdir(parents=True, exist_ok=True)
    experiment_results = {}
    started_at = time.perf_counter()
    for name in EXPERIMENT_NAMES:
        cfg = configs[name]
        experiment_output = output_root / name.lower()
        if not args.skip_training:
            if experiment_output.exists():
                if not args.overwrite:
                    raise FileExistsError(
                        "output exists; pass --overwrite: {}".format(
                            experiment_output))
                shutil.rmtree(str(experiment_output))
            _set_seed(int(cfg.STAGE3C.SEED))
            print("training {} from fresh initialization".format(
                name), flush=True)
            training_report = run_formal_training(
                rpnet=rpnet,
                cfg=cfg,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                device=device,
                output_dir=experiment_output,
                image_checkpoint=image_checkpoint,
                resume=None,
            )
        else:
            report_path = (
                experiment_output / "training_report.json")
            if not report_path.is_file():
                raise FileNotFoundError(
                    "training report not found: {}".format(
                        report_path))
            with report_path.open(
                    "r", encoding="utf-8") as input_file:
                training_report = json.load(input_file)
        checkpoint = (
            experiment_output
            / "checkpoints"
            / "stage3c_aux.best.pth.tar"
        )
        diagnostics_dir = experiment_output / "diagnostics"
        diagnostics = run_diagnostics(
            cfg=cfg,
            checkpoint=checkpoint,
            image_checkpoint=image_checkpoint,
            dataset_dir=dataset_dir,
            output_dir=diagnostics_dir,
            device=device,
            batch_size=int(
                cfg.STAGE3C.TRAINING.VAL_BATCH_SIZE),
            split="val",
        )
        experiment_results[name] = _experiment_summary(
            training_report=training_report,
            diagnostics=diagnostics,
            checkpoint=checkpoint,
        )
        _copy_report_artifacts(
            experiment_output=experiment_output,
            docs_output=docs_root / name.lower(),
        )
        print(
            "{} best_epoch={} full_AP={:.4f} no_traj_AP={:.4f}"
            .format(
                name,
                experiment_results[name]["best_epoch"],
                experiment_results[name][
                    "metrics_by_modality"]["full"]["branch_ap"],
                experiment_results[name][
                    "metrics_by_modality"][
                        "no_trajectory"]["branch_ap"],
            ),
            flush=True,
        )

    comparison = {
        "schema_version": "stage3c-r2-v1",
        "source_commit": (
            "a5f5c96a82279cbc7dea8b846da361175e178199"),
        "sample_counts": {"train": 2048, "validation": 512},
        "seed": int(reference.STAGE3C.SEED),
        "epochs": int(reference.STAGE3C.TRAINING.EPOCHS),
        "image_checkpoint": str(image_checkpoint.resolve()),
        "dataset_dir": str(dataset_dir.resolve()),
        "trajectory_modality_dropout": 0.25,
        "max_fragments": 64,
        "fresh_auxiliary_initialization": True,
        "optimizer_resume_used": False,
        "rpnet_strict_and_frozen": True,
        "branch_predictions_feed_path_push": False,
        "config_paths": {
            name: str(path.resolve())
            for name, path in config_paths.items()
        },
        "experiments": experiment_results,
        "decisions": build_r2_decisions(experiment_results),
        "training_elapsed_seconds": float(sum(
            result["training_elapsed_seconds"]
            for result in experiment_results.values()
        )),
        "report_generation_elapsed_seconds": float(
            time.perf_counter() - started_at),
    }
    for root in (output_root, docs_root):
        _write_json(root / "comparison.json", comparison)
        with (root / "README.md").open(
                "w", encoding="utf-8") as output:
            output.write(_render_readme(comparison))
    print(json.dumps({
        "comparison": str(
            (docs_root / "comparison.json").resolve()),
        "decisions": comparison["decisions"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
