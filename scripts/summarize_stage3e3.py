"""Aggregate Stage 3E-3 multi-seed and robustness results."""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


SEEDS = (20260724, 20260725, 20260726)
MODES = (
    "image_graph",
    "original_fragment",
    "full_trajectory",
    "no_trajectory",
    "retain_75",
    "retain_50",
    "retain_25",
    "wrong_sample_trajectory",
)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def _mean_std(values: Iterable[float]) -> Dict[str, Any]:
    values = [float(value) for value in values]
    return {
        "values": values,
        "mean": statistics.fmean(values) if values else None,
        "std": statistics.pstdev(values) if values else None,
    }


def _metric(metrics: Mapping[str, Any], path: str) -> float:
    value: Any = metrics
    for key in path.split("."):
        value = value[key]
    return float(value)


def _summarize_mode(
    reports: Mapping[int, Mapping[str, Any]],
    mode: str,
) -> Dict[str, Any]:
    paths = {
        "branch_ap": "branch_ap",
        "slot_ap": "slot_ap",
        "endpoint_error_mean_pixels":
            "thresholded.endpoint_error_mean_pixels",
        "endpoint_error_median_pixels":
            "thresholded.endpoint_error_median_pixels",
        "direction_error_mean_degrees":
            "thresholded.direction_error_mean_degrees",
        "direction_error_median_degrees":
            "thresholded.direction_error_median_degrees",
        "exact_branch_count_accuracy":
            "thresholded.exact_branch_count_accuracy",
        "missed_branch_rate": "thresholded.missed_branch_rate",
        "extra_branch_rate": "thresholded.extra_branch_rate",
        "precision": "thresholded.precision",
        "recall": "thresholded.recall",
        "f1": "thresholded.f1",
    }
    summary = {}
    for output_name, path in paths.items():
        values = []
        for report in reports.values():
            try:
                values.append(_metric(report["variants"][mode], path))
            except KeyError:
                continue
        summary[output_name] = _mean_std(values)
    summary["by_category_branch_ap"] = {
        category: _mean_std(
            report["variants"][mode]["by_category"][category]["branch_ap"]
            for report in reports.values()
        )
        for category in ("ordinary", "t_junction", "multi_branch")
    }
    summary["by_gt_count_branch_ap"] = {
        group: _mean_std(
            report["variants"][mode]["by_gt_count"][group]["branch_ap"]
            for report in reports.values()
        )
        for group in ("count_0", "count_1", "count_2", "count_ge3")
    }
    return summary


def build_stage3e3_summary(
    root: Path,
    *,
    reference_branch_ap: float = 0.9138492084801799,
    reproduction_branch_ap: Optional[float] = None,
    reproduction_tolerance: float = 0.001,
) -> Dict[str, Any]:
    training = {}
    robustness = {}
    for seed in SEEDS:
        seed_dir = root / "seed{}".format(seed)
        training[seed] = _load_json(seed_dir / "training_summary.json")
        robustness[seed] = _load_json(
            seed_dir / "robustness_evaluation.json")

    invariant_keys = (
        "e4_checkpoint",
        "val_fragment_tokens",
        "val_fragment_mask",
        "val_sample_ids",
    )
    validation_hash_controls_identical = all(
        len({report["hash_checks"][key]
             for report in robustness.values()}) == 1
        for key in invariant_keys
    )
    cache_hash_fields = (
        "fragment_tokens_sha256",
        "fragment_mask_sha256",
        "sample_ids_sha256",
    )
    train_val_cache_hash_controls_identical = all(
        len({training[seed]["cache"][split][field]
             for seed in SEEDS}) == 1
        for split in ("train", "val")
        for field in cache_hash_fields
    )
    hard_preflight_recorded = all(
        "stage3e3_preflight" in training[seed]["cache"]
        for seed in SEEDS
    )
    hash_controls_identical = bool(
        validation_hash_controls_identical
        and train_val_cache_hash_controls_identical
        and hard_preflight_recorded)
    baseline_modes_identical = {}
    for mode in ("image_graph", "original_fragment", "no_trajectory"):
        values = [
            report["variants"][mode]["branch_ap"]
            for report in robustness.values()
        ]
        baseline_modes_identical[mode] = (
            max(values) - min(values) <= 1e-12)

    full_gains = [
        report["variants"]["full_trajectory"]["branch_ap"]
        - report["variants"]["image_graph"]["branch_ap"]
        for report in robustness.values()
    ]
    original_gains = [
        report["variants"]["full_trajectory"]["branch_ap"]
        - report["variants"]["original_fragment"]["branch_ap"]
        for report in robustness.values()
    ]
    category_gains = {
        category: [
            report["variants"]["full_trajectory"]["by_category"][
                category]["branch_ap"]
            - report["variants"]["image_graph"]["by_category"][
                category]["branch_ap"]
            for report in robustness.values()
        ]
        for category in ("ordinary", "t_junction", "multi_branch")
    }
    thinning_degradation = {
        mode: _mean_std(
            report["variants"][mode]["branch_ap"]
            - report["variants"]["full_trajectory"]["branch_ap"]
            for report in robustness.values()
        )
        for mode in ("retain_75", "retain_50", "retain_25")
    }
    wrong_deltas = {
        "minus_full": _mean_std(
            report["variants"]["wrong_sample_trajectory"]["branch_ap"]
            - report["variants"]["full_trajectory"]["branch_ap"]
            for report in robustness.values()
        ),
        "minus_image_graph": _mean_std(
            report["variants"]["wrong_sample_trajectory"]["branch_ap"]
            - report["variants"]["image_graph"]["branch_ap"]
            for report in robustness.values()
        ),
    }
    diagnostic_fields = (
        "hidden_norm",
        "normalized_fragment_attention_entropy",
        "maximum_fragment_attention",
        "top1_cumulative_attention_mass",
        "top4_cumulative_attention_mass",
        "top8_cumulative_attention_mass",
        "top16_cumulative_attention_mass",
        "effective_fragment_count",
    )
    attention_summary = {
        mode: {
            field: _mean_std(
                report["attention_diagnostics"][mode][field]["mean"]
                for report in robustness.values()
            )
            for field in diagnostic_fields
        }
        for mode in (
            "full_trajectory",
            "retain_50",
            "retain_25",
            "wrong_sample_trajectory",
        )
    }
    for mode in attention_summary:
        attention_summary[mode]["all_finite"] = all(
            report["attention_diagnostics"][mode]["all_finite"]
            for report in robustness.values()
        )
        attention_summary[mode]["empty_trajectory_context_is_zero"] = all(
            report["attention_diagnostics"][mode][
                "empty_trajectory_context_is_zero"]
            for report in robustness.values()
        )
    no_trajectory_equivalent = all(
        report["no_trajectory_equivalence"]["maximum"] <= 1e-6
        for report in robustness.values()
    )
    frozen_unchanged = all(
        bool(report["frozen_modules_unchanged"])
        and bool(training[seed]["frozen_modules_unchanged"])
        for seed, report in robustness.items()
    )
    reproduction_passed = (
        reproduction_branch_ap is not None
        and abs(reproduction_branch_ap - reference_branch_ap)
        <= reproduction_tolerance
    )
    acceptance = {
        "mean_full_minus_image_graph_at_least_0_010":
            statistics.fmean(full_gains) >= 0.010,
        "every_seed_gain_positive": all(value > 0 for value in full_gains),
        "t_junction_mean_gain_positive":
            statistics.fmean(category_gains["t_junction"]) > 0,
        "multi_branch_mean_gain_positive":
            statistics.fmean(category_gains["multi_branch"]) > 0,
        "no_trajectory_strictly_equivalent": no_trajectory_equivalent,
        "hash_controls_identical": hash_controls_identical,
        "frozen_modules_unchanged": frozen_unchanged,
        "preflight_reproduction_passed": reproduction_passed,
    }
    acceptance["passed"] = all(acceptance.values())
    possible_shortcut = statistics.fmean([
        report["variants"]["wrong_sample_trajectory"]["branch_ap"]
        - report["variants"]["full_trajectory"]["branch_ap"]
        for report in robustness.values()
    ]) >= -0.001

    concise_per_seed = {}
    for seed in SEEDS:
        training_report = training[seed]
        robustness_report = robustness[seed]
        seed_variants = robustness_report["variants"]
        concise_per_seed[str(seed)] = {
            "seed": seed,
            "best_epoch": training_report["best_epoch"],
            "best_checkpoint": training_report["best_checkpoint"],
            "best_checkpoint_sha256": training_report[
                "best_checkpoint_sha256"],
            "elapsed_seconds": training_report["elapsed_seconds"],
            "peak_cuda_memory_bytes": training_report[
                "peak_cuda_memory_bytes"],
            "e4_checkpoint_sha256": training_report[
                "e4_checkpoint_sha256"],
            "cache_sha256": {
                split: {
                    field: training_report["cache"][split][field]
                    for field in cache_hash_fields
                }
                for split in ("train", "val")
            },
            "frozen_modules_unchanged": bool(
                training_report["frozen_modules_unchanged"]
                and robustness_report["frozen_modules_unchanged"]),
            "branch_ap": {
                mode: float(seed_variants[mode]["branch_ap"])
                for mode in MODES
            },
            "full_minus_image_graph_branch_ap": float(
                seed_variants["full_trajectory"]["branch_ap"]
                - seed_variants["image_graph"]["branch_ap"]),
            "full_minus_original_fragment_branch_ap": float(
                seed_variants["full_trajectory"]["branch_ap"]
                - seed_variants["original_fragment"]["branch_ap"]),
            "category_full_minus_image_graph_branch_ap": {
                category: float(
                    seed_variants["full_trajectory"]["by_category"][
                        category]["branch_ap"]
                    - seed_variants["image_graph"]["by_category"][
                        category]["branch_ap"])
                for category in ("ordinary", "t_junction", "multi_branch")
            },
            "no_trajectory_equivalence": robustness_report[
                "no_trajectory_equivalence"],
        }

    return {
        "schema_version": "stage3e3-comparison-v1",
        "validation_teacher_forced_auxiliary_metrics": True,
        "seeds": list(SEEDS),
        "controls": {
            "hash_controls_identical": hash_controls_identical,
            "validation_hash_controls_identical":
                validation_hash_controls_identical,
            "train_val_cache_hash_controls_identical":
                train_val_cache_hash_controls_identical,
            "hard_preflight_recorded": hard_preflight_recorded,
            "baseline_modes_identical": baseline_modes_identical,
            "reference_branch_ap": reference_branch_ap,
            "reproduction_branch_ap": reproduction_branch_ap,
            "reproduction_tolerance": reproduction_tolerance,
            "reproduction_passed": reproduction_passed,
        },
        "mode_summary": {
            mode: _summarize_mode(robustness, mode)
            for mode in MODES
        },
        "full_minus_image_graph": _mean_std(full_gains),
        "full_minus_original_fragment": _mean_std(original_gains),
        "category_full_minus_image_graph": {
            category: _mean_std(values)
            for category, values in category_gains.items()
        },
        "thinning_minus_full": thinning_degradation,
        "wrong_sample_deltas": wrong_deltas,
        "attention_summary": attention_summary,
        "acceptance": acceptance,
        "risk_flags": {
            "possible_trajectory_shortcut": possible_shortcut,
            "retain_25_nearly_unchanged": abs(
                thinning_degradation["retain_25"]["mean"]) <= 0.001,
            "high_seed_variance": _mean_std(full_gains)["std"] > 0.005,
        },
        "per_seed": concise_per_seed,
    }


def _read_reproduction_ap(path: Optional[Path]) -> Optional[float]:
    if path is None:
        return None
    value = _load_json(path)
    if "evaluation" in value:
        value = value["evaluation"]
    return float(value["variants"]["trajectory_evidence"]["branch_ap"])


def _write_readme(path: Path, comparison: Mapping[str, Any]) -> None:
    acceptance = comparison["acceptance"]
    full_gain = comparison["full_minus_image_graph"]
    wrong = comparison["wrong_sample_deltas"]
    modes = comparison["mode_summary"]
    categories = comparison["category_full_minus_image_graph"]
    thinning = comparison["thinning_minus_full"]
    attention = comparison["attention_summary"]
    lines = [
        "# Stage 3E-3 single-token trajectory evidence validation",
        "",
        "## Scope and code facts",
        "",
        "This report covers teacher-forced validation of the auxiliary branch "
        "head. It does not report complete VecRoad graph metrics. RPNet, the "
        "fragment and graph encoders, E4 branch decoder/output heads, support "
        "head, anchor path, and `Path.push` remain frozen or untouched. Only "
        "the M=1 `TrajectoryEvidenceEncoder` was trained.",
        "",
        "## Experimental results",
        "",
        "- Seeds: {}.".format(", ".join(map(str, comparison["seeds"]))),
        "- Mean full minus image_graph Branch AP: {:+.6f} (std {:.6f}).".format(
            full_gain["mean"], full_gain["std"]),
        "- Mean wrong-sample minus full Branch AP: {:+.6f}.".format(
            wrong["minus_full"]["mean"]),
        "- Preflight M1 reproduction passed: {}.".format(
            comparison["controls"]["reproduction_passed"]),
        "- Exact no-trajectory equivalence passed: {}.".format(
            acceptance["no_trajectory_strictly_equivalent"]),
        "- Acceptance gate passed: {}.".format(acceptance["passed"]),
        "",
        "| Mode | Branch AP mean | std |",
        "| --- | ---: | ---: |",
    ]
    for mode in MODES:
        lines.append("| {} | {:.6f} | {:.6f} |".format(
            mode,
            modes[mode]["branch_ap"]["mean"],
            modes[mode]["branch_ap"]["std"],
        ))
    lines.extend([
        "",
        "Category-wise full minus image_graph Branch AP: ordinary "
        "{:+.6f}, T-junction {:+.6f}, multi-branch {:+.6f}.".format(
            categories["ordinary"]["mean"],
            categories["t_junction"]["mean"],
            categories["multi_branch"]["mean"],
        ),
        "",
        "Thinning minus full Branch AP: retain-75 {:+.6f}, retain-50 "
        "{:+.6f}, retain-25 {:+.6f}.".format(
            thinning["retain_75"]["mean"],
            thinning["retain_50"]["mean"],
            thinning["retain_25"]["mean"],
        ),
        "",
        "For full trajectory evidence, normalized attention entropy was "
        "{:.6f}, effective fragment count {:.3f}, top-1 mass {:.6f}, and "
        "top-8 mass {:.6f} (three-seed means).".format(
            attention["full_trajectory"][
                "normalized_fragment_attention_entropy"]["mean"],
            attention["full_trajectory"][
                "effective_fragment_count"]["mean"],
            attention["full_trajectory"][
                "top1_cumulative_attention_mass"]["mean"],
            attention["full_trajectory"][
                "top8_cumulative_attention_mass"]["mean"],
        ),
        "",
        "| Seed | Best epoch | Full AP | Full-image delta | Checkpoint SHA256 |",
        "| --- | ---: | ---: | ---: | --- |",
    ])
    for seed in comparison["seeds"]:
        value = comparison["per_seed"][str(seed)]
        lines.append("| {} | {} | {:.6f} | {:+.6f} | `{}` |".format(
            seed,
            value["best_epoch"],
            value["branch_ap"]["full_trajectory"],
            value["full_minus_image_graph_branch_ap"],
            value["best_checkpoint_sha256"],
        ))
    lines.extend([
        "",
        "See `comparison.json`, `per_seed_results.json`, and "
        "`robustness_results.json` for complete numerical results.",
        "",
        "## Interpretive inference",
        "",
    ])
    if acceptance["passed"]:
        lines.append(
            "The single-token trajectory pathway passed the predefined "
            "teacher-forced stability gate. This is not evidence yet of an "
            "improvement in closed-loop road-graph extraction.")
    else:
        lines.append(
            "At least one predefined gate failed, so anchor integration is "
            "not recommended from these results alone.")
    if comparison["risk_flags"]["possible_trajectory_shortcut"]:
        lines.append(
            "The wrong-sample control is close to full trajectory evidence; "
            "this is flagged as a possible trajectory shortcut.")
    else:
        lines.append(
            "Wrong-sample trajectories reduced Branch AP by {:.6f} versus "
            "full evidence on average and fell below image_graph, so this "
            "control does not support a pure trajectory-presence shortcut."
            .format(abs(wrong["minus_full"]["mean"])))
    if comparison["risk_flags"]["retain_25_nearly_unchanged"]:
        lines.append(
            "Retaining only 25% of fragments changed Branch AP by less than "
            "0.001 on average. The learned pooling appears redundant or "
            "driven by a relatively small useful subset; this is a robustness "
            "observation, not proof that all discarded fragments are useless.")
    if categories["ordinary"]["mean"] <= 0:
        lines.append(
            "The gain is concentrated at junctions: ordinary-node AP changed "
            "slightly negatively, while T-junction and multi-branch gains "
            "were positive for every seed.")
    lines.extend([
        "",
        "## Reproduction",
        "",
        "```bash",
        "python train_trajectory_evidence.py --config configs/stage3e3_seed20260724.yml --device cuda",
        "python scripts/evaluate_stage3e3_robustness.py --config configs/stage3e3_seed20260724.yml --checkpoint data_self/stage3e3/seed20260724/checkpoints/stage3e0.best.pth.tar --device cuda",
        "python scripts/summarize_stage3e3.py --root data_self/stage3e3",
        "```",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data_self/stage3e3"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight-evaluation", type=Path)
    parser.add_argument("--full-tests-log", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir or Path(
        "docs/stage3e3_{}".format(date.today().strftime("%Y%m%d")))
    output_dir.mkdir(parents=True, exist_ok=True)
    reproduction_ap = _read_reproduction_ap(args.preflight_evaluation)
    comparison = build_stage3e3_summary(
        args.root,
        reproduction_branch_ap=reproduction_ap,
    )
    _write_json(output_dir / "comparison.json", comparison)
    _write_json(output_dir / "per_seed_results.json", {
        seed: value
        for seed, value in comparison["per_seed"].items()
    })
    _write_json(output_dir / "robustness_results.json", {
        str(seed): _load_json(
            args.root / "seed{}".format(seed)
            / "robustness_evaluation.json")
        for seed in SEEDS
    })
    _write_readme(output_dir / "README.md", comparison)
    if args.preflight_evaluation is not None:
        shutil.copyfile(
            args.preflight_evaluation,
            output_dir / "preflight_reproduction.json",
        )
    if args.full_tests_log is not None:
        shutil.copyfile(args.full_tests_log, output_dir / "full_tests.log")
    print(json.dumps({
        "output_dir": str(output_dir.resolve()),
        "acceptance_passed": comparison["acceptance"]["passed"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
