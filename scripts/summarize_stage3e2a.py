"""Summarize the Stage 3E-2A evidence-capacity diagnosis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping


VARIANTS = ("mean", "attention_m1", "latent_m4")


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            "Stage 3E-2A result not found: {}".format(path))
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _distribution_mean(
    diagnostics: Mapping[str, Any],
    name: str,
) -> Any:
    return diagnostics.get(name, {}).get("mean")


def _group_metrics(group: Mapping[str, Any]) -> Dict[str, Any]:
    thresholded = group.get("thresholded", {})
    return {
        "sample_count": int(group.get("sample_count", 0)),
        "gt_branch_count": int(group.get("gt_branch_count", 0)),
        "branch_ap": float(group.get("branch_ap", 0.0)),
        "slot_ap": float(group.get("slot_ap", 0.0)),
        "endpoint_error_mean_pixels": thresholded.get(
            "endpoint_error_mean_pixels"),
        "direction_error_mean_degrees": thresholded.get(
            "direction_error_mean_degrees"),
        "exact_branch_count_accuracy": thresholded.get(
            "exact_branch_count_accuracy"),
        "missed_branch_rate": thresholded.get(
            "missed_branch_rate"),
        "extra_branch_rate": thresholded.get(
            "extra_branch_rate"),
    }


def _extract(summary: Mapping[str, Any]) -> Dict[str, Any]:
    validation = summary["best_validation"]
    metrics = validation["variants"]["trajectory_evidence"]
    diagnostics = validation["trajectory_evidence_diagnostics"]
    thresholded = metrics["thresholded"]
    return {
        "aggregation_mode": summary["evidence_aggregation_mode"],
        "num_evidence_tokens": int(
            summary["num_evidence_tokens"]),
        "trainable_parameter_count": int(
            summary["trainable_parameter_count"]),
        "parameter_free_evidence_module": bool(
            summary["parameter_free_evidence_module"]),
        "best_epoch": int(summary["best_epoch"]),
        "branch_ap": float(metrics["branch_ap"]),
        "slot_ap": float(metrics["slot_ap"]),
        "endpoint_error_mean_pixels": float(
            thresholded["endpoint_error_mean_pixels"]),
        "direction_error_mean_degrees": float(
            thresholded["direction_error_mean_degrees"]),
        "ordinary_branch_ap": float(
            metrics["by_category"]["ordinary"]["branch_ap"]),
        "t_junction_branch_ap": float(
            metrics["by_category"]["t_junction"]["branch_ap"]),
        "multi_branch_ap": float(
            metrics["by_category"]["multi_branch"]["branch_ap"]),
        "by_category": {
            name: _group_metrics(group)
            for name, group in metrics["by_category"].items()
        },
        "by_gt_count": {
            name: _group_metrics(group)
            for name, group in metrics["by_gt_count"].items()
        },
        "evidence_token_norm_mean": _distribution_mean(
            diagnostics, "hidden_norm"),
        "evidence_token_cosine_mean": _distribution_mean(
            diagnostics, "pairwise_cosine_similarity"),
        "fragment_attention_cosine_mean": _distribution_mean(
            diagnostics,
            "fragment_attention_pairwise_cosine_similarity",
        ),
        "fragment_attention_entropy_mean": _distribution_mean(
            diagnostics,
            "normalized_fragment_attention_entropy",
        ),
        "fragment_attention_top8_jaccard_mean": _distribution_mean(
            diagnostics, "fragment_attention_top8_jaccard"),
        "no_trajectory_branch_ap": float(
            validation["variants"]["image_graph"]["branch_ap"]),
        "evidence_minus_no_trajectory_branch_ap": float(
            validation[
                "trajectory_evidence_minus_no_trajectory_branch_ap"]),
        "elapsed_seconds": float(summary["elapsed_seconds"]),
        "peak_cuda_memory_bytes": int(
            summary["peak_cuda_memory_bytes"]),
        "seed": int(summary["seed"]),
        "e4_checkpoint_sha256": summary["e4_checkpoint_sha256"],
        "initial_shared_evidence_state_sha256": summary[
            "initial_shared_evidence_state_sha256"],
        "train_fragment_tokens_sha256": summary[
            "cache"]["train"]["fragment_tokens_sha256"],
        "val_fragment_tokens_sha256": summary[
            "cache"]["val"]["fragment_tokens_sha256"],
        "train_fragment_mask_sha256": summary[
            "cache"]["train"]["fragment_mask_sha256"],
        "val_fragment_mask_sha256": summary[
            "cache"]["val"]["fragment_mask_sha256"],
        "train_sample_ids_sha256": summary[
            "cache"]["train"]["sample_ids_sha256"],
        "val_sample_ids_sha256": summary[
            "cache"]["val"]["sample_ids_sha256"],
        "best_checkpoint": summary["best_checkpoint"],
        "artifacts": summary["artifacts"],
    }


def _all_same(
    results: Mapping[str, Mapping[str, Any]],
    key: str,
) -> bool:
    return len({result[key] for result in results.values()}) == 1


def build_comparison(
    root: Path,
    *,
    equivalence_tolerance: float = 0.001,
    collapse_cosine_threshold: float = 0.99,
) -> Dict[str, Any]:
    if equivalence_tolerance < 0.0:
        raise ValueError("equivalence_tolerance must be non-negative")
    results = {
        name: _extract(_load_json(
            Path(root) / name / "training_summary.json"))
        for name in VARIANTS
    }
    expected = {
        "mean": ("masked_mean", 1, True),
        "attention_m1": ("latent_attention", 1, False),
        "latent_m4": ("latent_attention", 4, False),
    }
    for name, (
            expected_mode,
            expected_tokens,
            parameter_free) in expected.items():
        result = results[name]
        if (
                result["aggregation_mode"] != expected_mode
                or result["num_evidence_tokens"] != expected_tokens
                or result["parameter_free_evidence_module"]
                != parameter_free):
            raise ValueError(
                "{} does not match its controlled configuration".format(
                    name))

    control_keys = (
        "seed",
        "e4_checkpoint_sha256",
        "train_fragment_tokens_sha256",
        "val_fragment_tokens_sha256",
        "train_fragment_mask_sha256",
        "val_fragment_mask_sha256",
        "train_sample_ids_sha256",
        "val_sample_ids_sha256",
    )
    controls = {
        key: {
            "identical": _all_same(results, key),
            "value": results["mean"][key],
        }
        for key in control_keys
    }
    controls_identical = all(
        item["identical"] for item in controls.values())
    attention_shared_initialization_identical = (
        results["attention_m1"][
            "initial_shared_evidence_state_sha256"]
        == results["latent_m4"][
            "initial_shared_evidence_state_sha256"]
    )

    mean_ap = results["mean"]["branch_ap"]
    m1_ap = results["attention_m1"]["branch_ap"]
    m4_ap = results["latent_m4"]["branch_ap"]
    all_equivalent = (
        max(mean_ap, m1_ap, m4_ap)
        - min(mean_ap, m1_ap, m4_ap)
        <= equivalence_tolerance
    )
    learned_attention_gain = (
        max(m1_ap, m4_ap) - mean_ap > equivalence_tolerance)
    m4_gain = m4_ap - m1_ap > equivalence_tolerance
    m4_cosine = results[
        "latent_m4"]["evidence_token_cosine_mean"]
    m4_collapsed = (
        m4_cosine is not None
        and m4_cosine >= collapse_cosine_threshold
    )
    if all_equivalent:
        conclusion = (
            "Masked mean, learned M=1 attention, and latent M=4 are "
            "Branch-AP equivalent; the current pathway behaves as simple "
            "global trajectory aggregation.")
    elif m4_gain and m4_collapsed:
        conclusion = (
            "M=4 exceeds M=1 despite collapsed latent tokens; a later "
            "structure constraint, not more tokens, is the relevant "
            "hypothesis.")
    elif learned_attention_gain:
        conclusion = (
            "Learned attention exceeds strict masked mean; attention-based "
            "trajectory aggregation adds value.")
    else:
        conclusion = (
            "The learned evidence modules do not improve on strict masked "
            "mean under this controlled diagnosis.")
    return {
        "schema_version": "stage3e2a-comparison-v1",
        "results": results,
        "controls": controls,
        "controls_identical": controls_identical,
        "attention_shared_initialization_identical":
            attention_shared_initialization_identical,
        "decision": {
            "equivalence_tolerance": float(
                equivalence_tolerance),
            "collapse_cosine_threshold": float(
                collapse_cosine_threshold),
            "attention_m1_minus_mean_branch_ap": float(
                m1_ap - mean_ap),
            "latent_m4_minus_mean_branch_ap": float(
                m4_ap - mean_ap),
            "latent_m4_minus_attention_m1_branch_ap": float(
                m4_ap - m1_ap),
            "all_equivalent": bool(all_equivalent),
            "learned_attention_gain": bool(
                learned_attention_gain),
            "m4_gain": bool(m4_gain),
            "m4_collapsed": bool(m4_collapsed),
            "conclusion": conclusion,
        },
    }


def _format_optional(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return ("{:.%df}" % digits).format(float(value))


def _markdown(report: Mapping[str, Any]) -> str:
    main_rows = []
    diagnostic_rows = []
    count_rows = []
    category_rows = []
    labels = {
        "mean": "Mean",
        "attention_m1": "Attention M1",
        "latent_m4": "Latent M4",
    }
    for name in VARIANTS:
        result = report["results"][name]
        label = labels[name]
        main_rows.append(
            "| {label} | {ap:.6f} | {endpoint:.3f} | "
            "{direction:.3f} | {ordinary:.6f} | {t:.6f} | "
            "{multi:.6f} |".format(
                label=label,
                ap=result["branch_ap"],
                endpoint=result["endpoint_error_mean_pixels"],
                direction=result["direction_error_mean_degrees"],
                ordinary=result["ordinary_branch_ap"],
                t=result["t_junction_branch_ap"],
                multi=result["multi_branch_ap"],
            )
        )
        diagnostic_rows.append(
            "| {label} | {parameters} | {norm} | {cosine} | "
            "{entropy} |".format(
                label=label,
                parameters=result["trainable_parameter_count"],
                norm=_format_optional(
                    result["evidence_token_norm_mean"]),
                cosine=_format_optional(
                    result["evidence_token_cosine_mean"]),
                entropy=_format_optional(
                    result["fragment_attention_entropy_mean"]),
            )
        )
        for group_name in (
                "count_0", "count_1", "count_2", "count_ge3"):
            group = result["by_gt_count"][group_name]
            count_rows.append(
                "| {label} | {group_name} | {samples} | {ap:.6f} | "
                "{endpoint} | {direction} | {exact} |".format(
                    label=label,
                    group_name=group_name,
                    samples=group["sample_count"],
                    ap=group["branch_ap"],
                    endpoint=_format_optional(
                        group["endpoint_error_mean_pixels"], 3),
                    direction=_format_optional(
                        group["direction_error_mean_degrees"], 3),
                    exact=_format_optional(
                        group["exact_branch_count_accuracy"], 4),
                )
            )
        for category in (
                "ordinary", "t_junction", "multi_branch"):
            group = result["by_category"][category]
            category_rows.append(
                "| {label} | {category} | {samples} | {ap:.6f} | "
                "{endpoint} | {direction} |".format(
                    label=label,
                    category=category,
                    samples=group["sample_count"],
                    ap=group["branch_ap"],
                    endpoint=_format_optional(
                        group["endpoint_error_mean_pixels"], 3),
                    direction=_format_optional(
                        group["direction_error_mean_degrees"], 3),
                )
            )
    decision = report["decision"]
    return "\n".join([
        "# Stage 3E-2A trajectory evidence capacity diagnosis",
        "",
        "Mean, attention M=1, and latent M=4 use the same frozen E4, "
        "teacher-forced split, seed, fragment tokens, masks, and sample "
        "order. Mean is a strict parameter-free masked mean; no learned "
        "adapter is hidden in that baseline.",
        "",
        "| setting | Branch AP | endpoint px | direction deg | "
        "ordinary AP | T-junction AP | multi-branch AP |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *main_rows,
        "",
        "| setting | trainable params | token norm | token cosine | "
        "attention entropy |",
        "|---|---:|---:|---:|---:|",
        *diagnostic_rows,
        "",
        "## GT branch-count groups",
        "",
        "| setting | group | samples | Branch AP | endpoint px | "
        "direction deg | exact count |",
        "|---|---|---:|---:|---:|---:|---:|",
        *count_rows,
        "",
        "## Road categories",
        "",
        "| setting | category | samples | Branch AP | endpoint px | "
        "direction deg |",
        "|---|---|---:|---:|---:|---:|",
        *category_rows,
        "",
        "## Controlled-comparison checks",
        "",
        "- Frozen inputs and split hashes identical: `{}`.".format(
            report["controls_identical"]),
        "- Attention M1/M4 shared initialization identical: `{}`.".format(
            report[
                "attention_shared_initialization_identical"]),
        "- Attention M1 - Mean Branch AP: `{:+.6f}`.".format(
            decision["attention_m1_minus_mean_branch_ap"]),
        "- Latent M4 - Mean Branch AP: `{:+.6f}`.".format(
            decision["latent_m4_minus_mean_branch_ap"]),
        "- Latent M4 - Attention M1 Branch AP: `{:+.6f}`.".format(
            decision[
                "latent_m4_minus_attention_m1_branch_ap"]),
        "- M4 collapsed: `{}`.".format(
            decision["m4_collapsed"]),
        "",
        "## Conclusion",
        "",
        decision["conclusion"],
        "",
        "No diversity/reliability loss, support replacement, anchor "
        "fusion, or Path.push change was introduced.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "python train_trajectory_evidence.py --config "
        "configs/stage3e2a_mean.yml --mode train",
        "python train_trajectory_evidence.py --config "
        "configs/stage3e2a_attention_m1.yml --mode train",
        "python train_trajectory_evidence.py --config "
        "configs/stage3e2a_latent_m4.yml --mode train",
        "python scripts/summarize_stage3e2a.py "
        "--root data_self/stage3e2a "
        "--output-dir docs/stage3e2a_20260726",
        "```",
        "",
    ])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("data_self/stage3e2a"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--equivalence-tolerance", type=float, default=0.001)
    parser.add_argument(
        "--collapse-cosine-threshold", type=float, default=0.99)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = build_comparison(
        args.root,
        equivalence_tolerance=args.equivalence_tolerance,
        collapse_cosine_threshold=args.collapse_cosine_threshold,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "comparison.json").open(
            "w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    with (args.output_dir / "README.md").open(
            "w", encoding="utf-8") as output_file:
        output_file.write(_markdown(report))
    print(json.dumps(
        report["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
