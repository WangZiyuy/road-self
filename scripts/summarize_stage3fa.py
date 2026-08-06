"""Summarize the three fixed-seed Stage 3F-A teacher-forced experiments."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20260724, 20260725, 20260726)


def _read(path: Path):
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")


def _metric(report, mode, group, name):
    return report["modes"][mode]["heads"]["anchor"][group][name]


def _stats(values):
    return {"mean": statistics.fmean(values),
            "std": statistics.pstdev(values), "values": values}


def summarize(root: Path, output_dir: Path):
    reports = {
        seed: _read(root / str(seed) / "evaluation.json") for seed in SEEDS}
    training_reports = {
        seed: _read(root / str(seed) / "training_summary.json")
        for seed in SEEDS}
    primary = {
        "pixel_ap": ("higher", []),
        "hit_at_5_px": ("higher", []),
        "top1_endpoint_error_mean": ("lower", []),
    }
    per_seed = {}
    for seed, report in reports.items():
        deltas = {}
        for name, (direction, values) in primary.items():
            full = _metric(report, "fused_full_trajectory", "all", name)
            base = _metric(report, "original_anchor", "all", name)
            delta = full - base if direction == "higher" else base - full
            deltas[name] = delta; values.append(delta)
        per_seed[str(seed)] = {
            "primary_improvement_deltas": deltas,
            "checkpoint": report["checkpoint"],
            "checkpoint_sha256": report["checkpoint_sha256"],
            "strict_equivalence": report["strict_equivalence"],
            "branch_regression": report["branch_regression"],
            "best_epoch": int(training_reports[seed]["best_epoch"]),
            "best_validation_anchor_total_loss": float(
                training_reports[seed]["best_validation_anchor_total_loss"]),
            "training_elapsed_seconds": float(
                training_reports[seed]["elapsed_seconds"]),
        }
    means = {name: _stats(values) for name, (_, values) in primary.items()}
    improving = [name for name, value in means.items() if value["mean"] > 0]
    seed_consistency = {
        name: sum(delta > 0 for delta in data["values"])
        for name, data in means.items()}
    group_deltas = {}
    for group in ("ordinary", "t_junction", "multi_branch"):
        group_deltas[group] = {
            name: _stats([
                (_metric(report, "fused_full_trajectory", group, name)
                 - _metric(report, "original_anchor", group, name))
                if direction == "higher" else
                (_metric(report, "original_anchor", group, name)
                 - _metric(report, "fused_full_trajectory", group, name))
                for report in reports.values()])
            for name, (direction, _) in primary.items()
        }
    ordinary_safe = (
        group_deltas["ordinary"]["pixel_ap"]["mean"] >= -0.002
        and group_deltas["ordinary"]["hit_at_5_px"]["mean"] >= -0.005
        and group_deltas["ordinary"]["top1_endpoint_error_mean"]["mean"] >= -0.2)
    junction_gain = all(
        (group_deltas[group]["hit_at_5_px"]["mean"] > 0
         or group_deltas[group]["top1_endpoint_error_mean"]["mean"] > 0)
        for group in ("t_junction", "multi_branch"))
    wrong_sample_checks = {}
    for seed, report in reports.items():
        full_ap = _metric(report, "fused_full_trajectory", "all", "pixel_ap")
        wrong_ap = _metric(
            report, "fused_wrong_sample_trajectory", "all", "pixel_ap")
        full_hit = _metric(
            report, "fused_full_trajectory", "all", "hit_at_5_px")
        wrong_hit = _metric(
            report, "fused_wrong_sample_trajectory", "all", "hit_at_5_px")
        full_error = _metric(
            report, "fused_full_trajectory", "all",
            "top1_endpoint_error_mean")
        wrong_error = _metric(
            report, "fused_wrong_sample_trajectory", "all",
            "top1_endpoint_error_mean")
        deltas = {
            "pixel_ap": full_ap - wrong_ap,
            "hit_at_5_px": full_hit - wrong_hit,
            "top1_endpoint_error_mean": wrong_error - full_error,
        }
        wrong_sample_checks[str(seed)] = {
            "correct_minus_wrong_improvements": deltas,
            "wrong_does_not_outperform_any_primary": all(
                value >= 0 for value in deltas.values()),
            "correct_is_strictly_better_on_at_least_one_primary": any(
                value > 0 for value in deltas.values()),
        }
    wrong_below_full = all(
        value["wrong_does_not_outperform_any_primary"]
        and value["correct_is_strictly_better_on_at_least_one_primary"]
        for value in wrong_sample_checks.values())
    engineering = all(
        all(item["passed"] for item in report["strict_equivalence"].values())
        and report["branch_regression"]["passed"]
        and report["frozen_sha_unchanged_from_cache"]
        for report in reports.values())
    same_two = sum(
        1 for name in improving if seed_consistency[name] >= 2) >= 2
    gate = {
        "engineering_conditions": engineering,
        "at_least_two_primary_mean_improvements": len(improving) >= 2,
        "at_least_two_seeds_on_same_two_metrics": same_two,
        "junction_localization_gain": junction_gain,
        "ordinary_safe": ordinary_safe,
        "wrong_sample_below_correct": wrong_below_full,
    }
    passed = all(gate.values())
    if not passed and means["pixel_ap"]["mean"] > 0 and (
            means["hit_at_5_px"]["mean"] <= 0
            and means["top1_endpoint_error_mean"]["mean"] <= 0):
        conclusion = "ranking/calibration improvement only"
    elif not ordinary_safe:
        conclusion = "junction-specific gain with unsafe ordinary regression"
    else:
        conclusion = "passed; Stage 3F-B may be considered" if passed else "Stage 3F-A acceptance gate not passed"
    comparison = {
        "stage": "3F-A", "primary_metric_improvements": means,
        "seed_consistency": seed_consistency, "group_improvements": group_deltas,
        "acceptance_gate": gate, "passed": passed, "conclusion": conclusion,
        "wrong_sample_primary_checks": wrong_sample_checks,
        "teacher_forced_anchor_validation": True,
        "closed_loop_road_graph_extraction": False,
    }
    _write(output_dir / "comparison.json", comparison)
    _write(output_dir / "per_seed_results.json", per_seed)
    _write(output_dir / "anchor_metrics.json", {
        str(seed): report["modes"] for seed, report in reports.items()})
    _write(output_dir / "robustness_results.json", {
        str(seed): {name: report["modes"][name] for name in (
            "fused_full_trajectory", "fused_no_trajectory",
            "fused_wrong_sample_trajectory", "fused_retain_25")}
        for seed, report in reports.items()})
    seed_rows = "\n".join(
        "| {seed} | {epoch} | {base:.6f} | {full:.6f} | {hit:.6f} | "
        "{error:.3f} | `{sha}` |".format(
            seed=seed,
            epoch=training_reports[seed]["best_epoch"],
            base=_metric(report, "original_anchor", "all", "pixel_ap"),
            full=_metric(report, "fused_full_trajectory", "all", "pixel_ap"),
            hit=_metric(report, "fused_full_trajectory", "all", "hit_at_5_px"),
            error=_metric(
                report, "fused_full_trajectory", "all",
                "top1_endpoint_error_mean"),
            sha=report["checkpoint_sha256"])
        for seed, report in reports.items())
    readme = """# road_self Stage 3F-A

## Code facts

- The original RPNet, road/junction heads, both original anchor output heads,
  graph/fragment/evidence encoders, E4 decoder and support head were frozen.
- One shared trajectory projection and gate feed separate zero-initialized
  residual adapters at the final full- and low-resolution anchor-only
  pre-head features. The residual never enters recursive anchor feedback.
- `Path.pop`, `Path.push`, `map_to_coordinate`, threshold, NMS and
  `NUM_TARGETS` were not changed. Auxiliary branch endpoints are not used for
  graph growth.
- The frozen teacher-forced cache contains 2048 train and 512 validation
  states. Both anchor pre-head feature arrays are float32; post-serialization
  anchor outputs and losses matched the dynamic path with max difference 0.
- The canonical Stage 3E-3 evidence checkpoint is seed 20260724 with SHA256
  `d7f00a12a10c8e80687945f0f68596883bdd4b038893d828bba297601b451ff6`.

## Experiment results

Primary mean improvement (positive is better):

- anchor pixel AP: {ap:+.6f}
- hit@5 px: {hit:+.6f}
- top-1 endpoint mean error: {error:+.6f} px

Acceptance gate: **{passed}**. Conclusion: **{conclusion}**.

| Seed | Best epoch | Original AP | Fused AP | Fused hit@5 | Fused error px | Checkpoint SHA256 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
{seed_rows}

Checkpoints were selected only by the lowest validation anchor total loss,
not by localization metrics. Complete full/no-trajectory/wrong-sample and
grouped metrics are stored in `anchor_metrics.json` and
`robustness_results.json`.

## Interpretation

The result above tests frozen teacher-forced anchor prediction only. It is not
a complete VecRoad closed-loop road-graph extraction experiment; it does not
establish improvements in APLS, TOPO, or full road-network quality.

## Reproduction

```bash
python scripts/build_stage3fa_anchor_cache.py --config configs/stage3fa_seed20260724.yml --device cuda --overwrite
python train_stage3fa_anchor_fusion.py --config configs/stage3fa_seed20260724.yml --device cuda
python scripts/evaluate_stage3fa_anchor_fusion.py --config configs/stage3fa_seed20260724.yml --device cuda
python scripts/summarize_stage3fa.py --root data_self/stage3fa --output-dir docs/stage3fa_20260805
```

Repeat train/evaluate for seeds 20260725 and 20260726. `--skip-sanity` is
accepted only when that seed already has a saved `passed=true` sanity report.
""".format(
        ap=means["pixel_ap"]["mean"], hit=means["hit_at_5_px"]["mean"],
        error=means["top1_endpoint_error_mean"]["mean"],
        passed="passed" if passed else "not passed", conclusion=conclusion,
        seed_rows=seed_rows)
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "data_self/stage3fa")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "docs/stage3fa_20260805")
    args = parser.parse_args()
    print(json.dumps(summarize(args.root, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
