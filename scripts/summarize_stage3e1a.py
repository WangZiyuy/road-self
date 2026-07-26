"""Summarize the Stage 3E-1A M=1/4/8 evidence-token ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping


VARIANTS = ("m1", "m4", "m8")


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            "Stage 3E-1A result not found: {}".format(path))
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _mean_or_none(
    diagnostics: Mapping[str, Any],
    key: str,
) -> Any:
    return diagnostics.get(key, {}).get("mean")


def _extract(summary: Mapping[str, Any]) -> Dict[str, Any]:
    validation = summary["best_validation"]
    metrics = validation["variants"]["trajectory_evidence"]
    diagnostics = validation["trajectory_evidence_diagnostics"]
    thresholded = metrics["thresholded"]
    return {
        "num_evidence_tokens": int(
            summary["num_evidence_tokens"]),
        "best_epoch": int(summary["best_epoch"]),
        "branch_ap": float(metrics["branch_ap"]),
        "slot_ap": float(metrics["slot_ap"]),
        "endpoint_error_mean_pixels": float(
            thresholded["endpoint_error_mean_pixels"]),
        "direction_error_mean_degrees": float(
            thresholded["direction_error_mean_degrees"]),
        "exact_branch_count_accuracy": float(
            thresholded["exact_branch_count_accuracy"]),
        "ordinary_branch_ap": float(
            metrics["by_category"]["ordinary"]["branch_ap"]),
        "t_junction_branch_ap": float(
            metrics["by_category"]["t_junction"]["branch_ap"]),
        "multi_branch_ap": float(
            metrics["by_category"]["multi_branch"]["branch_ap"]),
        "evidence_token_cosine_mean": _mean_or_none(
            diagnostics, "pairwise_cosine_similarity"),
        "fragment_attention_cosine_mean": _mean_or_none(
            diagnostics,
            "fragment_attention_pairwise_cosine_similarity",
        ),
        "fragment_attention_entropy_mean": _mean_or_none(
            diagnostics,
            "normalized_fragment_attention_entropy",
        ),
        "fragment_attention_top8_jaccard_mean": _mean_or_none(
            diagnostics, "fragment_attention_top8_jaccard"),
        "no_trajectory_branch_ap": float(
            validation["variants"]["image_graph"]["branch_ap"]),
        "evidence_minus_no_trajectory_branch_ap": float(
            validation[
                "trajectory_evidence_minus_no_trajectory_branch_ap"]),
        "elapsed_seconds": float(summary["elapsed_seconds"]),
        "peak_cuda_memory_bytes": int(
            summary["peak_cuda_memory_bytes"]),
        "initial_shared_evidence_state_sha256": summary[
            "initial_shared_evidence_state_sha256"],
        "e4_checkpoint_sha256": summary["e4_checkpoint_sha256"],
        "seed": int(summary["seed"]),
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
    expected_counts = {"m1": 1, "m4": 4, "m8": 8}
    if any(
            results[name]["num_evidence_tokens"] != expected
            for name, expected in expected_counts.items()):
        raise ValueError("result token counts do not match M=1/4/8")
    control_keys = (
        "seed",
        "e4_checkpoint_sha256",
        "initial_shared_evidence_state_sha256",
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
            "value": results["m1"][key],
        }
        for key in control_keys
    }
    controls_identical = all(
        value["identical"] for value in controls.values())
    m1_ap = results["m1"]["branch_ap"]
    m4_ap = results["m4"]["branch_ap"]
    m8_ap = results["m8"]["branch_ap"]
    m4_collapsed = (
        results["m4"]["evidence_token_cosine_mean"]
        >= collapse_cosine_threshold
        and results["m4"]["fragment_attention_cosine_mean"]
        >= collapse_cosine_threshold
    )
    m8_collapsed = (
        results["m8"]["evidence_token_cosine_mean"]
        >= collapse_cosine_threshold
        and results["m8"]["fragment_attention_cosine_mean"]
        >= collapse_cosine_threshold
    )
    one_approximately_four = (
        abs(m4_ap - m1_ap) <= equivalence_tolerance)
    eight_capacity_gain = (
        m8_ap - max(m1_ap, m4_ap) > equivalence_tolerance)
    multi_token_gain_with_collapse = (
        max(m4_ap, m8_ap) - m1_ap > equivalence_tolerance
        and (m4_collapsed or m8_collapsed)
    )
    if one_approximately_four:
        conclusion = (
            "M=1 and M=4 are AP-equivalent within tolerance; the current "
            "pathway behaves like global trajectory evidence.")
    elif multi_token_gain_with_collapse:
        conclusion = (
            "Multiple tokens improve AP despite representation collapse; "
            "the result supports a later lightweight diversity ablation.")
    elif eight_capacity_gain:
        conclusion = (
            "M=8 provides a capacity gain over M=1/4.")
    else:
        conclusion = (
            "The token-count ablation does not support increasing evidence "
            "capacity.")
    return {
        "schema_version": "stage3e1a-comparison-v1",
        "results": results,
        "controls": controls,
        "controls_identical": controls_identical,
        "decision": {
            "equivalence_tolerance": float(equivalence_tolerance),
            "collapse_cosine_threshold": float(
                collapse_cosine_threshold),
            "m4_minus_m1_branch_ap": float(m4_ap - m1_ap),
            "m8_minus_m1_branch_ap": float(m8_ap - m1_ap),
            "m8_minus_m4_branch_ap": float(m8_ap - m4_ap),
            "one_approximately_four": one_approximately_four,
            "m4_collapsed": bool(m4_collapsed),
            "m8_collapsed": bool(m8_collapsed),
            "multi_token_gain_with_collapse": bool(
                multi_token_gain_with_collapse),
            "eight_capacity_gain": bool(eight_capacity_gain),
            "conclusion": conclusion,
        },
    }


def _format_optional(value: Any) -> str:
    return "n/a" if value is None else "{:.6f}".format(float(value))


def _markdown(report: Mapping[str, Any]) -> str:
    rows = []
    diagnostic_rows = []
    for name in VARIANTS:
        result = report["results"][name]
        rows.append(
            "| {name} | {ap:.6f} | {endpoint:.3f} | {direction:.3f} | "
            "{ordinary:.6f} | {t:.6f} | {multi:.6f} |".format(
                name=name.upper(),
                ap=result["branch_ap"],
                endpoint=result["endpoint_error_mean_pixels"],
                direction=result["direction_error_mean_degrees"],
                ordinary=result["ordinary_branch_ap"],
                t=result["t_junction_branch_ap"],
                multi=result["multi_branch_ap"],
            )
        )
        diagnostic_rows.append(
            "| {name} | {token} | {attention} | {entropy:.6f} | "
            "{jaccard} |".format(
                name=name.upper(),
                token=_format_optional(
                    result["evidence_token_cosine_mean"]),
                attention=_format_optional(
                    result["fragment_attention_cosine_mean"]),
                entropy=result["fragment_attention_entropy_mean"],
                jaccard=_format_optional(
                    result[
                        "fragment_attention_top8_jaccard_mean"]),
            )
        )
    decision = report["decision"]
    return "\n".join([
        "# Stage 3E-1A evidence-token necessity and diversity",
        "",
        "M=1/4/8 use the same teacher-forced split, seed, frozen E4, "
        "fragment tokens, masks, sample order, optimizer settings and "
        "shared evidence-encoder initialization.",
        "",
        "| setting | Branch AP | endpoint px | direction deg | ordinary AP "
        "| T-junction AP | multi-branch AP |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "| setting | token cosine | attention cosine | attention entropy | "
        "top-8 Jaccard |",
        "|---|---:|---:|---:|---:|",
        *diagnostic_rows,
        "",
        "## Controlled-comparison checks",
        "",
        "- All controls identical: `{}`.".format(
            report["controls_identical"]),
        "- M4 - M1 Branch AP: `{:+.6f}`.".format(
            decision["m4_minus_m1_branch_ap"]),
        "- M8 - M1 Branch AP: `{:+.6f}`.".format(
            decision["m8_minus_m1_branch_ap"]),
        "- M8 - M4 Branch AP: `{:+.6f}`.".format(
            decision["m8_minus_m4_branch_ap"]),
        "- M4 collapsed: `{}`.".format(decision["m4_collapsed"]),
        "- M8 collapsed: `{}`.".format(decision["m8_collapsed"]),
        "",
        "## Conclusion",
        "",
        decision["conclusion"],
        "",
        "No diversity, reliability, count, support, anchor, or Path.push "
        "loss/path was added in this diagnostic stage.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "python train_trajectory_evidence.py --config "
        "configs/stage3e1a_m1.yml --mode train",
        "python train_trajectory_evidence.py --config "
        "configs/stage3e1a_m4.yml --mode train",
        "python train_trajectory_evidence.py --config "
        "configs/stage3e1a_m8.yml --mode train",
        "python scripts/summarize_stage3e1a.py "
        "--root data_self/stage3e1a "
        "--output-dir docs/stage3e1a_20260726",
        "```",
        "",
    ])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("data_self/stage3e1a"))
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
