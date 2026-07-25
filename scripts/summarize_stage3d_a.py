"""Create the machine-readable and human-readable Stage 3D-A report."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def _fmt(value, digits=4):
    return "n/a" if value is None else ("{:.{}f}".format(value, digits))


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("data_self/stage3d_a"),
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("docs/stage3d_a_20260725"),
    )
    parser.add_argument("--test-count", type=int, default=0)
    return parser.parse_args()


def main():
    args = _parse_args()
    run_dir = args.run_dir
    docs_dir = args.docs_dir
    docs_dir.mkdir(parents=True, exist_ok=True)
    label = _read(run_dir / "label_diagnostics.json")
    sanity = _read(run_dir / "support_sanity_report.json")
    training = _read(run_dir / "support_training_report.json")
    modality = _read(
        run_dir / "e4_modality_diagnostics" / "summary.json")
    stability = _read(
        run_dir / "e4_stability" / "stability_summary.json")
    validation = training["best_validation"]

    artifacts = {
        "label_diagnostics.json": label,
        "support_sanity_report.json": sanity,
        "support_training_report.json": training,
        "e4_modality_summary.json": modality,
        "e4_stability_summary.json": stability,
    }
    for name, value in artifacts.items():
        _write(docs_dir / name, value)
    visualization_dir = docs_dir / "visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)
    visualization_paths = []
    for source in sorted((run_dir / "visualizations").glob("*.png")):
        target = visualization_dir / source.name
        shutil.copy2(str(source), str(target))
        visualization_paths.append(str(target))

    epochs = [record["epoch"] for record in training["history"]]
    train_loss = [
        record["training_loss"] for record in training["history"]]
    val_loss = [
        record["validation"]["loss"] for record in training["history"]]
    support_ap = [
        record["validation"]["support_ap"]
        for record in training["history"]]
    attention_ap = [
        record["validation"]["attention_support_ap"]
        for record in training["history"]]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, train_loss, label="train support BCE")
    axes[0].plot(epochs, val_loss, label="validation support BCE")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    axes[1].plot(epochs, support_ap, label="support head AP")
    axes[1].plot(epochs, attention_ap, label="raw attention AP")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("fragment support AP")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    figure.tight_layout()
    curve_path = docs_dir / "support_training_curves.png"
    figure.savefig(str(curve_path), dpi=180)
    plt.close(figure)

    branch_ap = modality["branch_ap_by_modality"]
    aggregate = stability["aggregate"]
    decisions = {
        "label_gate_passed": bool(label["gate"]["passed"]),
        "sanity_overfit_passed": bool(sanity["passed"]),
        "validation_head_beats_attention": bool(
            validation["support_ap"]
            > validation["attention_support_ap"] + 0.05),
        "branch_top_k_sets_separate": bool(
            validation["top_k_fragment_jaccard"]["median"] is not None
            and validation[
                "top_k_fragment_jaccard"]["median"] < 0.5),
    }
    decisions["stage3d_a_passed"] = bool(all(decisions.values()))
    summary = {
        "schema_version": "stage3d-a-report-v1",
        "e4_checkpoint": (
            "data_self/stage3c_r2/e4/checkpoints/"
            "stage3c_aux.best.pth.tar"),
        "support_checkpoint": training["best_checkpoint"],
        "label": {
            "support_available_rate": label[
                "combined"]["support_available_rate"],
            "multibranch_support_available_rate": label[
                "combined"]["by_gt_branch_count"][">=2"][
                    "support_available_rate"],
            "bounded_64_oracle_support_recall": label[
                "combined"]["bounded_64_oracle_support_recall"],
            "segment_only_positive_ratio": label[
                "combined"]["segment_only_positive_ratio"],
        },
        "sanity": {
            "initial_support_ap": sanity["initial"]["support_ap"],
            "best_support_ap": sanity["best"]["support_ap"],
            "attention_support_ap": sanity[
                "best"]["attention_support_ap"],
            "best_epoch": sanity["best_epoch"],
            "raw_loss_reduction": sanity["loss_reduction"],
            "soft_target_entropy_floor": sanity[
                "soft_target_entropy_floor"],
            "reducible_loss_reduction": sanity[
                "reducible_loss_reduction"],
        },
        "validation": validation,
        "branch_ap_by_modality": branch_ap,
        "trajectory_graph_minus_graph_only": float(
            branch_ap["trajectory_graph"] - branch_ap["graph_only"]),
        "full_minus_no_trajectory": float(
            branch_ap["full"] - branch_ap["no_trajectory"]),
        "e4_stability": aggregate,
        "decisions": decisions,
        "test_count": int(args.test_count),
        "constraints": {
            "rpnet_strict_and_frozen": True,
            "e4_modules_strict_and_frozen": True,
            "only_support_head_trained": True,
            "support_replaces_trajectory_attention": False,
            "support_changes_branch_output": False,
            "support_feeds_anchor": False,
            "support_feeds_path_push": False,
        },
        "visualizations": visualization_paths,
        "training_curve": str(curve_path),
    }
    _write(docs_dir / "summary.json", summary)

    lines = [
        "# road_self Stage 3D-A",
        "",
        "Stage 3D-A learns a branch-conditioned fragment support score as "
        "an evaluation-only side head. It does not replace E4 trajectory "
        "attention and is not connected to anchor prediction or Path.push.",
        "",
        "## Label diagnostics",
        "",
        "- Overall support availability: **{}**".format(
            _fmt(summary["label"]["support_available_rate"])),
        "- Multi-branch support availability: **{}**".format(
            _fmt(summary["label"][
                "multibranch_support_available_rate"])),
        "- Bounded-64 oracle support recall: **{}**".format(
            _fmt(summary["label"][
                "bounded_64_oracle_support_recall"])),
        "- Segment-only positive ratio: **{}**".format(
            _fmt(summary["label"]["segment_only_positive_ratio"])),
        "- The configured 80% multi-branch availability gate: **{}**"
        .format("passed" if decisions["label_gate_passed"] else "failed"),
        "",
        "## 32-sample overfit",
        "",
        "- Support AP: {} -> **{}**".format(
            _fmt(summary["sanity"]["initial_support_ap"]),
            _fmt(summary["sanity"]["best_support_ap"])),
        "- Raw attention support AP: **{}**".format(
            _fmt(summary["sanity"]["attention_support_ap"])),
        "- Reducible-loss reduction: **{}**".format(
            _fmt(summary["sanity"]["reducible_loss_reduction"])),
        "- Result: **{}**".format(
            "passed" if decisions["sanity_overfit_passed"] else "failed"),
        "",
        "## Validation support metrics",
        "",
        "- Support-head AP / AUROC: **{} / {}**".format(
            _fmt(validation["support_ap"]),
            _fmt(validation["support_auroc"])),
        "- Raw attention AP / AUROC: **{} / {}**".format(
            _fmt(validation["attention_support_ap"]),
            _fmt(validation["attention_support_auroc"])),
        "- Recall@1/4/8/16: **{} / {} / {} / {}**".format(
            *[
                _fmt(validation["recall_at"][str(value)])
                for value in (1, 4, 8, 16)
            ]),
        "- Top-8 branch-pair Jaccard mean / median: **{} / {}**".format(
            _fmt(validation["top_k_fragment_jaccard"]["mean"]),
            _fmt(validation["top_k_fragment_jaccard"]["median"])),
        "- Best epoch: **{}**".format(training["best_epoch"]),
        "- Recall@K is the fraction of all positive fragments recovered "
        "within the top K, not an at-least-one-hit rate.",
        "- The real validation cache contains no segment-only positive "
        "pair. Segment-only support is therefore code-path tested, but "
        "not empirically validated by this run.",
        "- Visual inspection shows useful branch conditioning overall, "
        "but individual multi-branch queries can still rank low-target "
        "fragments in their top eight; the support head is not perfect.",
        "",
        "## E4 modality ablation",
        "",
        "| modality | branch AP |",
        "| --- | ---: |",
    ]
    for name in (
            "graph_only", "trajectory_graph",
            "no_trajectory", "full"):
        lines.append("| {} | {:.4f} |".format(name, branch_ap[name]))
    lines.extend([
        "",
        "- trajectory_graph - graph_only: **{:+.4f} AP**".format(
            summary["trajectory_graph_minus_graph_only"]),
        "- full - no_trajectory: **{:+.4f} AP**".format(
            summary["full_minus_no_trajectory"]),
        "",
        "## Three-seed E4 stability",
        "",
        "- Full AP mean/std: **{} / {}**".format(
            _fmt(aggregate["full_branch_ap"]["mean"]),
            _fmt(aggregate["full_branch_ap"]["std"])),
        "- No-trajectory AP mean/std: **{} / {}**".format(
            _fmt(aggregate["no_trajectory_branch_ap"]["mean"]),
            _fmt(aggregate["no_trajectory_branch_ap"]["std"])),
        "- Full-minus-no-trajectory mean/std: **{} / {}**".format(
            _fmt(aggregate[
                "full_minus_no_trajectory_branch_ap"]["mean"]),
            _fmt(aggregate[
                "full_minus_no_trajectory_branch_ap"]["std"])),
        "- Oracle duplicate mean/std: **{} / {}**".format(
            _fmt(aggregate["oracle_k_duplicate_ratio"]["mean"]),
            _fmt(aggregate["oracle_k_duplicate_ratio"]["std"])),
        "- Distinct coverage mean/std: **{} / {}**".format(
            _fmt(aggregate[
                "oracle_k_distinct_gt_coverage"]["mean"]),
            _fmt(aggregate[
                "oracle_k_distinct_gt_coverage"]["std"])),
        "",
        "## Decision",
        "",
        "Stage 3D-A acceptance: **{}**.".format(
            "passed" if decisions["stage3d_a_passed"] else "failed"),
        "This result validates an independent support head only. It does "
        "not authorize replacing trajectory cross-attention, adding a "
        "reliability loss, or connecting branch/support outputs to anchor "
        "or Path.push.",
        "",
        "Best support-head checkpoint: `{}`.".format(
            training["best_checkpoint"]),
        "",
        "Reproduction commands (from the road_self root on the 237 "
        "server):",
        "",
        "```bash",
        "python train_trajectory_support.py \\",
        "  --config configs/stage3d_a_support.yml --mode labels",
        "python train_trajectory_support.py \\",
        "  --config configs/stage3d_a_support.yml --mode train \\",
        "  --device cuda",
        "python scripts/run_stage3d_e4_stability.py \\",
        "  --config configs/stage3d_a_support.yml --device cuda",
        "```",
        "",
        "Tests: **{} passed**.".format(args.test_count),
    ])
    (docs_dir / "README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary["decisions"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
