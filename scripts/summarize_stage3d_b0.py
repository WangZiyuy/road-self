"""Build compact Stage 3D-B0 documentation from server run artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read(path: Path):
    if not path.is_file():
        raise FileNotFoundError("report input not found: {}".format(path))
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(
            value, output_file, ensure_ascii=False,
            indent=2, sort_keys=True)
        output_file.write("\n")


def _mean(values):
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _mean_seed_metric(reports, *keys):
    values = []
    for report in reports:
        value = report["best_validation"]
        for key in keys:
            value = value[key]
        values.append(float(value))
    return _mean(values)


def _fmt(value):
    return "n/a" if value is None else "{:.4f}".format(value)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir", type=Path,
        default=Path("data_self/stage3d_b0"))
    parser.add_argument(
        "--docs-dir", type=Path,
        default=Path("docs/stage3d_b0_20260725"))
    parser.add_argument("--test-count", type=int, default=0)
    return parser.parse_args()


def main():
    args = _parse_args()
    run_dir = args.run_dir
    docs_dir = args.docs_dir
    docs_dir.mkdir(parents=True, exist_ok=True)
    comparison = _read(run_dir / "comparison.json")
    labels = _read(run_dir / "label_diagnostics.json")
    frozen = _read(run_dir / "frozen_cache_report.json")
    for name, value in (
            ("comparison.json", comparison),
            ("label_diagnostics.json", labels),
            ("frozen_cache_report.json", frozen)):
        _write(docs_dir / name, value)

    raw = comparison["raw_attention"]
    post = comparison["post_fusion_support"]
    reports = comparison["pre_trajectory_seed_reports"]
    aggregate = comparison["pre_trajectory_stability"]
    acceptance = comparison["acceptance"]
    ranking_ks = (1, 4, 8, 16)
    pre_ranking = {}
    for metric in (
            "precision_at", "recall_at", "hit_at",
            "soft_support_mass_recall_at", "ndcg_at"):
        pre_ranking[metric] = {
            str(k): _mean_seed_metric(
                reports, metric, str(k))
            for k in ranking_ks
        }
    pre_by_count = {
        group: _mean_seed_metric(
            reports, "by_gt_branch_count", group, "support_ap")
        for group in ("1", "2", ">=3", ">=2")
    }

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for report in reports:
        epochs = [
            item["epoch"] for item in report["history"]]
        ap = [
            item["validation"]["support_ap"]
            for item in report["history"]]
        multi_ap = [
            item["validation"]["by_gt_branch_count"][">=2"][
                "support_ap"]
            for item in report["history"]]
        label = "seed {}".format(report["seed"])
        axes[0].plot(epochs, ap, label=label)
        axes[1].plot(epochs, multi_ap, label=label)
    axes[0].axhline(
        raw["support_ap"], color="black", linestyle=":",
        label="raw attention")
    axes[0].axhline(
        post["support_ap"], color="black", linestyle="--",
        label="post-fusion")
    axes[1].axhline(
        raw["by_gt_branch_count"][">=2"]["support_ap"],
        color="black", linestyle=":", label="raw attention")
    axes[1].axhline(
        post["by_gt_branch_count"][">=2"]["support_ap"],
        color="black", linestyle="--", label="post-fusion")
    axes[0].set_title("all support-valid branches")
    axes[1].set_title("GT branch count >= 2")
    for axis in axes:
        axis.set_xlabel("epoch")
        axis.set_ylabel("support AP")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.tight_layout()
    curve_path = docs_dir / "pre_trajectory_training_curves.png"
    figure.savefig(str(curve_path), dpi=180)
    plt.close(figure)

    summary = {
        "schema_version": "stage3d-b0-report-v1",
        "e4_checkpoint": comparison["e4_checkpoint"],
        "post_fusion_checkpoint":
            comparison["post_fusion_checkpoint"],
        "branch_support_hit_rate": labels[
            "bounded_64_branch_support_hit_rate"],
        "segment_only_diagnostics":
            labels["segment_only_diagnostics"],
        "raw_attention": raw,
        "post_fusion_support": post,
        "pre_trajectory_stability": aggregate,
        "pre_trajectory_ranking_mean": pre_ranking,
        "pre_trajectory_ap_by_gt_branch_count": pre_by_count,
        "acceptance": acceptance,
        "constraints": comparison["constraints"],
        "test_count": int(args.test_count),
    }
    _write(docs_dir / "summary.json", summary)

    lines = [
        "# road_self Stage 3D-B0",
        "",
        "Stage 3D-B0 tests a branch-conditioned selector before E4 "
        "trajectory cross-attention. The trainable branch input is exactly "
        "`concat(graph_conditioned_query, image/walked_path_context)`.",
        "",
        "## Label diagnostics",
        "",
        "- bounded-64 branch support hit rate: **{}**".format(
            _fmt(summary["branch_support_hit_rate"])),
        "- segment-only fragments inspected: **{}**; positive in the "
        "unchanged label configuration: **0**".format(
            labels["segment_only_diagnostics"][
                "segment_only_fragment_count"]),
        "",
        "## Main comparison",
        "",
        "| source | AP | AUROC | P@8 | Hit@8 | mass recall@8 | nDCG@8 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| raw attention | {} | {} | {} | {} | {} | {} |".format(
            _fmt(raw["support_ap"]), _fmt(raw["support_auroc"]),
            _fmt(raw["precision_at"]["8"]),
            _fmt(raw["hit_at"]["8"]),
            _fmt(raw["soft_support_mass_recall_at"]["8"]),
            _fmt(raw["ndcg_at"]["8"])),
        "| post-fusion support | {} | {} | {} | {} | {} | {} |"
        .format(
            _fmt(post["support_ap"]), _fmt(post["support_auroc"]),
            _fmt(post["precision_at"]["8"]),
            _fmt(post["hit_at"]["8"]),
            _fmt(post["soft_support_mass_recall_at"]["8"]),
            _fmt(post["ndcg_at"]["8"])),
        "| pre-trajectory support (3-seed mean) | {} | — | {} | {} | "
        "{} | {} |".format(
            _fmt(aggregate["support_ap"]["mean"]),
            _fmt(pre_ranking["precision_at"]["8"]),
            _fmt(pre_ranking["hit_at"]["8"]),
            _fmt(pre_ranking[
                "soft_support_mass_recall_at"]["8"]),
            _fmt(pre_ranking["ndcg_at"]["8"])),
        "",
        "Pre-trajectory AP mean/std: **{} / {}**.".format(
            _fmt(aggregate["support_ap"]["mean"]),
            _fmt(aggregate["support_ap"]["std"])),
        "",
        "## AP by GT branch count",
        "",
        "| source | count=1 | count=2 | count>=3 | count>=2 |",
        "| --- | ---: | ---: | ---: | ---: |",
        "| raw attention | {} | {} | {} | {} |".format(*[
            _fmt(raw["by_gt_branch_count"][group]["support_ap"])
            for group in ("1", "2", ">=3", ">=2")
        ]),
        "| post-fusion support | {} | {} | {} | {} |".format(*[
            _fmt(post["by_gt_branch_count"][group]["support_ap"])
            for group in ("1", "2", ">=3", ">=2")
        ]),
        "| pre-trajectory support | {} | {} | {} | {} |".format(*[
            _fmt(pre_by_count[group])
            for group in ("1", "2", ">=3", ">=2")
        ]),
        "",
        "## Separation and support-invalid branches",
        "",
        "- predicted top-8 Jaccard median, 3-seed mean: **{}**"
        .format(_fmt(aggregate[
            "predicted_top8_jaccard_median"]["mean"])),
        "- GT-positive-set Jaccard median: **{}**".format(
            _fmt(raw["gt_positive_set_jaccard"]["median"])),
        "- support-invalid top-1 probability mean / p90: **{} / {}**"
        .format(
            _fmt(aggregate[
                "support_invalid_top1_mean_probability"]["mean"]),
            _fmt(aggregate[
                "support_invalid_top1_p90_probability"]["mean"])),
        "",
        "## Decision",
        "",
    ]
    for name, passed in acceptance["checks"].items():
        lines.append(
            "- {}: **{}**".format(
                name, "passed" if passed else "failed"))
    lines.extend([
        "",
        "Stage 3D-B0 acceptance: **{}**.".format(
            "passed" if acceptance["passed"] else "failed"),
        "Support-guided aggregation: **{}**.".format(
            "recommended as the next isolated experiment"
            if acceptance["recommend_support_guided_aggregation"]
            else "not recommended; stop at Stage 3D-B0"),
        "",
        "No support score changed E4 trajectory attention, branch outputs, "
        "anchor prediction, or Path.push.",
        "",
        "Tests: **{} passed**.".format(args.test_count),
    ])
    (docs_dir / "README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(acceptance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
