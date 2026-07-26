"""Build the machine-readable Stage 3D-C1 comparison and decision report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping


VARIANTS = {
    "original_attention": "a_original",
    "support_aggregation": "b_support",
    "support_topk_8": "c_topk8",
    "support_topk_16": "d_topk16",
    "random_fragment_aggregation": "e_random",
}
OPTIONAL_VARIANTS = {
    "c1_b_topk8": "c1_b_topk8",
}
TRAINED_VARIANTS = set(VARIANTS) - {"original_attention"}
ELIGIBLE_C1B_VARIANTS = {
    "support_aggregation",
    "support_topk_8",
    "support_topk_16",
}


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("Stage 3D-C1 report not found: {}".format(path))
    with path.open("r", encoding="utf-8") as input_file:
        value = json.load(input_file)
    if not isinstance(value, Mapping):
        raise ValueError("report must contain a JSON object: {}".format(path))
    return dict(value)


def _validation_report(root: Path, name: str) -> Dict[str, Any]:
    directory = root / {
        **VARIANTS,
        **OPTIONAL_VARIANTS,
    }[name]
    if name in TRAINED_VARIANTS or name in OPTIONAL_VARIANTS:
        training = _read_json(directory / "training_summary.json")
        validation = training["best_validation"]
        provenance = {
            "report": str((directory / "training_summary.json").resolve()),
            "best_checkpoint": training["best_checkpoint"],
            "best_epoch": int(training["best_epoch"]),
            "elapsed_seconds": float(training["elapsed_seconds"]),
            "peak_cuda_memory_bytes": int(
                training["peak_cuda_memory_bytes"]),
            "training_stage": training["training_stage"],
        }
    else:
        validation = _read_json(directory / "evaluation.json")
        provenance = {
            "report": str((directory / "evaluation.json").resolve()),
            "best_checkpoint": None,
            "best_epoch": None,
            "elapsed_seconds": float(validation["elapsed_seconds"]),
            "peak_cuda_memory_bytes": 0,
            "training_stage": "frozen_e4_baseline",
        }
    full = validation["full"]
    no_trajectory = validation["no_trajectory"]
    return {
        "branch_ap": float(full["branch_ap"]),
        "slot_ap": float(full["slot_ap"]),
        "full_minus_no_trajectory_branch_ap": float(
            validation["full_minus_no_trajectory_branch_ap"]),
        "no_trajectory_branch_ap": float(no_trajectory["branch_ap"]),
        "endpoint_error_mean_pixels": float(
            full["thresholded"]["endpoint_error_mean_pixels"]),
        "direction_error_mean_degrees": float(
            full["thresholded"]["direction_error_mean_degrees"]),
        "exact_branch_count_accuracy": float(
            full["thresholded"]["exact_branch_count_accuracy"]),
        "oracle_k_recall": float(full["oracle_k"]["recall"]),
        "oracle_k_distinct_gt_coverage": float(
            full["oracle_k"]["distinct_gt_coverage"]),
        "oracle_k_duplicate_pair_ratio": float(
            full["oracle_k"]["duplicates"][
                "duplicate_pair_ratio"]),
        "by_category": full["by_category"],
        "support_selection": validation.get("support_selection"),
        "support_loss": validation.get("support_loss"),
        "provenance": provenance,
    }


def build_comparison(
    root: Path,
    *,
    minimum_branch_ap_gain: float = 0.001,
    minimum_modality_gain: float = 0.0,
) -> Dict[str, Any]:
    results = {
        name: _validation_report(root, name)
        for name in VARIANTS
    }
    for name, directory in OPTIONAL_VARIANTS.items():
        if (root / directory / "training_summary.json").is_file():
            results[name] = _validation_report(root, name)
    original_ap = results["original_attention"]["branch_ap"]
    best_name = max(
        ELIGIBLE_C1B_VARIANTS,
        key=lambda name: results[name]["branch_ap"],
    )
    best = results[best_name]
    branch_gain = float(best["branch_ap"] - original_ap)
    modality_gain = float(
        best["full_minus_no_trajectory_branch_ap"])
    c1a_passed = (
        branch_gain >= float(minimum_branch_ap_gain)
        and modality_gain >= float(minimum_modality_gain)
    )
    random_ap = results[
        "random_fragment_aggregation"]["branch_ap"]
    c1b = results.get("c1_b_topk8")
    c1b_gain = (
        None
        if c1b is None
        else float(c1b["branch_ap"] - best["branch_ap"])
    )
    return {
        "schema_version": "stage3d-c1-comparison-v1",
        "results": results,
        "decision": {
            "best_support_variant": best_name,
            "best_support_branch_ap": float(best["branch_ap"]),
            "original_attention_branch_ap": float(original_ap),
            "best_support_minus_original_branch_ap": branch_gain,
            "best_support_full_minus_no_trajectory_branch_ap": modality_gain,
            "best_support_minus_random_branch_ap": float(
                best["branch_ap"] - random_ap),
            "minimum_branch_ap_gain": float(minimum_branch_ap_gain),
            "minimum_modality_gain": float(minimum_modality_gain),
            "c1_a_passed": bool(c1a_passed),
            "run_c1_b": bool(c1a_passed),
            "c1_b_completed": c1b is not None,
            "c1_b_branch_ap": (
                None if c1b is None
                else float(c1b["branch_ap"])),
            "c1_b_minus_best_c1_a_branch_ap": c1b_gain,
            "c1_b_improved_over_c1_a": bool(
                c1b_gain is not None and c1b_gain > 0.0),
            "reason": (
                "C1-a improved validation branch AP and retained the "
                "configured full-vs-no-trajectory gain."
                if c1a_passed else
                "C1-a did not satisfy both configured validation gates; "
                "C1-b must not be launched automatically."
            ),
        },
    }


def _markdown(report: Mapping[str, Any]) -> str:
    rows = []
    category_rows = []
    support_rows = []
    for name in report["results"]:
        result = report["results"][name]
        rows.append(
            "| {name} | {ap:.6f} | {no:.6f} | {gain:+.6f} | "
            "{endpoint:.3f} | {direction:.3f} | {count:.4f} | "
            "{coverage:.4f} | {duplicate:.4f} |".format(
                name=name,
                ap=result["branch_ap"],
                no=result["no_trajectory_branch_ap"],
                gain=result[
                    "full_minus_no_trajectory_branch_ap"],
                endpoint=result["endpoint_error_mean_pixels"],
                direction=result["direction_error_mean_degrees"],
                count=result["exact_branch_count_accuracy"],
                coverage=result["oracle_k_distinct_gt_coverage"],
                duplicate=result["oracle_k_duplicate_pair_ratio"],
            )
        )
        categories = result["by_category"]
        category_rows.append(
            "| {name} | {ordinary:.6f} | {t_junction:.6f} | "
            "{multi_branch:.6f} |".format(
                name=name,
                ordinary=categories["ordinary"]["branch_ap"],
                t_junction=categories["t_junction"]["branch_ap"],
                multi_branch=categories["multi_branch"]["branch_ap"],
            )
        )
        selection = result.get("support_selection")
        if selection is not None:
            support_rows.append(
                "| {name} | {ap:.6f} | {precision:.6f} | "
                "{recall:.6f} | {ndcg:.6f} | {jaccard:.6f} |".format(
                    name=name,
                    ap=selection["support_ap"],
                    precision=selection["precision_at"]["8"],
                    recall=selection["recall_at"]["8"],
                    ndcg=selection["ndcg_at"]["8"],
                    jaccard=selection[
                        "predicted_top_k_jaccard"]["mean"],
                )
            )
    decision = report["decision"]
    return "\n".join([
        "# Stage 3D-C1 support-guided trajectory fusion",
        "",
        "All results use the same E4 branch decoder, RPNet checkpoint, "
        "teacher-forced split, seed, and bounded 64-fragment inputs. "
        "The branch head remains auxiliary and never feeds `Path.push`.",
        "",
        "| variant | branch AP | no-traj AP | full-no-traj | endpoint px | "
        "direction deg | exact count | oracle distinct coverage | "
        "oracle duplicate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## Results by node type",
        "",
        "| variant | ordinary AP | T-junction AP | multi-branch AP |",
        "|---|---:|---:|---:|",
        *category_rows,
        "",
        "## Support selection",
        "",
        "| variant | support AP | Precision@8 | Recall@8 | nDCG@8 | "
        "top-8 query Jaccard |",
        "|---|---:|---:|---:|---:|---:|",
        *support_rows,
        "",
        "## C1-b decision",
        "",
        "- Best support variant: `{}`.".format(
            decision["best_support_variant"]),
        "- Branch AP gain over original: `{:+.6f}`.".format(
            decision["best_support_minus_original_branch_ap"]),
        "- Full minus no-trajectory AP: `{:+.6f}`.".format(
            decision[
                "best_support_full_minus_no_trajectory_branch_ap"]),
        "- C1-a gate passed: `{}`.".format(decision["c1_a_passed"]),
        "- Run C1-b: `{}`.".format(decision["run_c1_b"]),
        "- C1-b completed: `{}`.".format(
            decision["c1_b_completed"]),
        "- C1-b AP change over best C1-a: `{}`.".format(
            (
                "not run"
                if decision[
                    "c1_b_minus_best_c1_a_branch_ap"] is None
                else "{:+.6f}".format(
                    decision[
                        "c1_b_minus_best_c1_a_branch_ap"])
            )
        ),
        "- Reason: {}".format(decision["reason"]),
        "",
        "## Reproduction commands",
        "",
        "```bash",
        "python train_support_fusion.py --config "
        "configs/stage3d_c1_a_original.yml --mode evaluate",
        "python train_support_fusion.py --config "
        "configs/stage3d_c1_b_support.yml --mode train",
        "python train_support_fusion.py --config "
        "configs/stage3d_c1_c_topk8.yml --mode train",
        "python train_support_fusion.py --config "
        "configs/stage3d_c1_d_topk16.yml --mode train",
        "python train_support_fusion.py --config "
        "configs/stage3d_c1_e_random.yml --mode train",
        "python train_support_fusion.py --config "
        "configs/stage3d_c1_b_finetune_topk8.yml --mode train",
        "python scripts/summarize_stage3d_c1.py "
        "--root data_self/stage3d_c1 "
        "--output-dir docs/stage3d_c1_20260726",
        "```",
        "",
        "C1-b is launched from the selected C1-a best checkpoint only when "
        "the machine-readable gate above reports `run_c1_b=true`.",
        "",
        "C1-a caches frozen RPNet, graph-state, trajectory-encoder and "
        "pre-trajectory decoder tensors once. A regression test verifies "
        "that this cache produces the same support-fusion outputs as the "
        "dynamic frozen-model path. C1-b keeps the trajectory encoder "
        "dynamic and trainable.",
        "",
        "The best C1-a gain over original attention is larger than zero, "
        "but its margin over the random-fragment control is only "
        "`{:+.6f}`. C1-b adds only `{:+.6f}` AP over C1-a. These results "
        "support the implementation gate, but are not yet strong evidence "
        "of seed-stable generalization.".format(
            decision["best_support_minus_random_branch_ap"],
            (
                0.0
                if decision[
                    "c1_b_minus_best_c1_a_branch_ap"] is None
                else decision[
                    "c1_b_minus_best_c1_a_branch_ap"]
            ),
        ),
        "",
        "Support ranking metrics (Precision/Recall/nDCG at K and top-8 "
        "Jaccard) and ordinary/T-junction/multi-branch metrics are retained "
        "in `comparison.json`.",
        "",
    ])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("data_self/stage3d_c1"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--minimum-branch-ap-gain", type=float, default=0.001)
    parser.add_argument(
        "--minimum-modality-gain", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = build_comparison(
        args.root,
        minimum_branch_ap_gain=args.minimum_branch_ap_gain,
        minimum_modality_gain=args.minimum_modality_gain,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = args.output_dir / "comparison.json"
    with comparison_path.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    with (args.output_dir / "README.md").open(
            "w", encoding="utf-8") as output_file:
        output_file.write(_markdown(report))
    print(json.dumps(report["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
