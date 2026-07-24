"""Run the Stage 3C-R1 three-way multi-branch overfit comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch


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
    run_overfit_sanity,
)
from utils.stage3c_branch_dataset import Stage3CBranchDataset  # noqa: E402


VARIANT_NAMES = ("M0", "M3", "M4")
REPRESENTATION_STAGES = {
    "learned_query": "debug_learned_query_embedding",
    "pre_graph_query": "debug_pre_graph_queries",
    "graph_conditioned_query": "debug_graph_conditioned_queries",
    "image_context": "debug_image_cross_attention_output",
    "trajectory_context":
        "debug_trajectory_cross_attention_output",
    "final_fused_query": "debug_final_fused_queries",
}


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--m0-config", type=Path,
        default=Path("configs/stage3c_r1_m0.yml"))
    parser.add_argument(
        "--m3-config", type=Path,
        default=Path("configs/stage3c_r1_m3.yml"))
    parser.add_argument(
        "--m4-config", type=Path,
        default=Path("configs/stage3c_r1_m4.yml"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_self/stage3c_r1"),
    )
    parser.add_argument("--device")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace only the three configured R1 variant directories",
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="recompute decisions/README from an existing comparison.json",
    )
    return parser.parse_args()


def _plain(value: Any):
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
            _plain(value), output, indent=2, sort_keys=True)
        output.write("\n")


def _selected_indices(dataset, cfg) -> Sequence[int]:
    sanity = cfg.STAGE3C.SANITY
    minimum = int(sanity.GT_BRANCH_COUNT_MIN)
    maximum = int(sanity.GT_BRANCH_COUNT_MAX)
    required = int(sanity.SAMPLE_COUNT)
    selected = []
    for index in range(len(dataset)):
        count = int(dataset[index][
            "branch_targets"]["branch_count"])
        if minimum <= count <= maximum:
            selected.append(index)
        if len(selected) == required:
            break
    if len(selected) != required:
        raise RuntimeError(
            "only {} samples satisfy GT count {}-{}; {} required".format(
                len(selected), minimum, maximum, required))
    return selected


def _selected_data_fingerprint(dataset, indices) -> str:
    digest = hashlib.sha256()
    tensor_paths = (
        ("trajectory_batch", "traj_xy_norm"),
        ("trajectory_batch", "traj_time_delta"),
        ("trajectory_batch", "point_mask"),
        ("trajectory_batch", "fragment_mask"),
        ("trajectory_batch", "track_indices"),
        ("branch_targets", "branch_offsets_norm"),
        ("branch_targets", "branch_mask"),
        ("metadata", "center_xy"),
        ("metadata", "vertex_id"),
    )
    for index in indices:
        digest.update(int(index).to_bytes(8, "little", signed=False))
        sample = dataset[index]
        for parent, key in tensor_paths:
            array = sample[parent][key].detach().cpu().numpy()
            digest.update(parent.encode("utf-8"))
            digest.update(key.encode("utf-8"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _validate_configs(configs) -> None:
    reference = configs["M0"]
    common_paths = (
        ("STAGE3C", "SEED"),
        ("STAGE3C", "IMAGE_CHECKPOINT"),
        ("STAGE3C", "DATASET_DIR"),
        ("STAGE3C", "MODEL", "HIDDEN_DIM"),
        ("STAGE3C", "MODEL", "NUM_HEADS"),
        ("STAGE3C", "MODEL", "TRAJECTORY_LAYERS"),
        ("STAGE3C", "MODEL", "NUM_QUERIES"),
        ("STAGE3C", "MODEL", "IMAGE_POOL_SIZE"),
        ("STAGE3C", "MODEL", "DROPOUT"),
        ("STAGE3C", "SANITY", "SAMPLE_COUNT"),
        ("STAGE3C", "SANITY", "GT_BRANCH_COUNT_MIN"),
        ("STAGE3C", "SANITY", "GT_BRANCH_COUNT_MAX"),
        ("STAGE3C", "SANITY", "MAX_EPOCHS"),
        ("STAGE3C", "SANITY", "LEARNING_RATE"),
    )

    def value_at(cfg, path):
        value = cfg
        for key in path:
            value = value[key]
        return value

    for name, cfg in configs.items():
        for path in common_paths:
            if value_at(cfg, path) != value_at(reference, path):
                raise ValueError(
                    "{} differs at {}".format(name, ".".join(path)))
        if float(
                cfg.STAGE3C.TRAINING
                .TRAJECTORY_MODALITY_DROPOUT) != 0.0:
            raise ValueError(
                "{} must disable trajectory modality dropout".format(
                    name))
        if not bool(cfg.STAGE3C.SANITY.DISABLE_MODEL_DROPOUT):
            raise ValueError(
                "{} must disable model dropout".format(name))

    expected = {
        "M0": (0.0, 1.0, 0),
        "M3": (1.0, 0.2, 0),
        "M4": (1.0, 0.2, 1),
    }
    for name, cfg in configs.items():
        actual = (
            float(cfg.STAGE3C.MATCHING.EXISTENCE_COST_WEIGHT),
            float(cfg.STAGE3C.LOSS.EXIST_NO_OBJECT_COEF),
            int(cfg.STAGE3C.MODEL.QUERY_SELF_ATTENTION_LAYERS),
        )
        if actual != expected[name]:
            raise ValueError(
                "{} repair settings are {}, expected {}".format(
                    name, actual, expected[name]))


def _representation_summary(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as input_file:
        raw = json.load(input_file)["full"]
    return {
        output_name: {
            "pairwise_cosine": raw[key]["pairwise_cosine"],
            "hidden_norm": raw[key]["hidden_norm"],
        }
        for output_name, key in REPRESENTATION_STAGES.items()
    }


def _variant_summary(
    *,
    sanity: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    representation: Mapping[str, Any],
    checkpoint: Path,
) -> Dict[str, Any]:
    return {
        "checkpoint": str(checkpoint.resolve()),
        "sanity_gate_passed": bool(sanity["passed"]),
        "sanity_gate_checks": sanity["gate_checks"],
        "branch_ap": float(diagnostics["branch_ap"]),
        "slot_ap": float(diagnostics["slot_ap"]),
        "matched_probability": diagnostics["matched_probability"],
        "unmatched_probability": diagnostics["unmatched_probability"],
        "probability_separation_mean": float(
            diagnostics["matched_prob_mean"]
            - diagnostics["unmatched_prob_mean"]),
        "probability_separation_median": float(
            diagnostics["matched_prob_median"]
            - diagnostics["unmatched_prob_median"]),
        "exact_count_accuracy": float(
            diagnostics["exact_count_accuracy"]),
        "oracle_k_recall": float(
            diagnostics["oracle_k"]["recall"]),
        "oracle_k_distinct_gt_coverage": float(
            diagnostics["oracle_k"]["distinct_gt_coverage"]),
        "oracle_k_duplicate_pair_ratio": float(
            diagnostics["oracle_k_duplicate_ratio"]),
        "actual_matched_duplicate_pair_ratio": float(
            diagnostics["matched_duplicate_ratio"]),
        "per_query_match_frequency":
            diagnostics["per_query_match_frequency"],
        "query_representation": representation,
        "curve_gate_history": _curve_gate_history(sanity),
        "elapsed_seconds": float(sanity["elapsed_seconds"]),
    }


def _curve_gate_history(
    sanity: Mapping[str, Any],
) -> Dict[str, Any]:
    gate_checks = sanity["gate_checks"]
    thresholds = {
        "branch_ap": gate_checks["branch_ap"]["threshold"],
        "exact": gate_checks[
            "exact_count_accuracy"]["threshold"],
        "coverage": gate_checks[
            "oracle_k_distinct_gt_coverage"]["threshold"],
        "duplicate": gate_checks[
            "oracle_k_duplicate_pair_ratio"]["threshold"],
    }
    qualifying = []
    for record in sanity["curve"]:
        if int(record["epoch"]) == 0:
            continue
        if (
                float(record["branch_ap"])
                >= thresholds["branch_ap"]
                and float(record["thresholded_metrics"][
                    "exact_branch_count_accuracy"])
                >= thresholds["exact"]
                and float(record["oracle_k"][
                    "distinct_gt_coverage"])
                >= thresholds["coverage"]
                and float(record["oracle_k"]["duplicates"][
                    "duplicate_pair_ratio"])
                <= thresholds["duplicate"]):
            qualifying.append(int(record["epoch"]))
    return {
        "qualifying_epoch_count": len(qualifying),
        "qualifying_epochs": qualifying,
        "first_qualifying_epoch": (
            qualifying[0] if qualifying else None),
        "last_qualifying_epoch": (
            qualifying[-1] if qualifying else None),
        "final_epoch": int(sanity["curve"][-1]["epoch"]),
        "final_regressed_after_qualifying": bool(
            qualifying
            and not sanity["passed"]
            and qualifying[-1]
            < int(sanity["curve"][-1]["epoch"])
        ),
    }


def build_r1_decision(
    variants: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    m0, m3, m4 = (
        variants["M0"], variants["M3"], variants["M4"])
    m3_probability_improved = bool(
        m3["slot_ap"] > m0["slot_ap"]
        and m3["probability_separation_mean"]
        > m0["probability_separation_mean"]
    )
    m3_collapsed = bool(
        m3["oracle_k_duplicate_pair_ratio"] > 0.20
        or m3["oracle_k_distinct_gt_coverage"] < 0.90
        or m3["query_representation"][
            "final_fused_query"]["pairwise_cosine"]["mean"]
        >= 0.95
    )
    m3_graph_conditioning_homogeneous = bool(
        m3["query_representation"][
            "graph_conditioned_query"]["pairwise_cosine"]["mean"]
        >= 0.95
    )
    m4_identity_restored = bool(
        m4["query_representation"][
            "graph_conditioned_query"]["pairwise_cosine"]["mean"]
        < 0.95
        and m4["query_representation"][
            "final_fused_query"]["pairwise_cosine"]["mean"]
        < 0.95
    )
    m4_distinct_gt_covered = bool(
        m4["oracle_k_distinct_gt_coverage"] >= 0.90
        and m4["oracle_k_duplicate_pair_ratio"] <= 0.20
    )
    recommend_full = bool(
        m4["sanity_gate_passed"]
        and m4_identity_restored
        and m4_distinct_gt_covered
    )
    return {
        "m3_probability_separation_improved_over_m0":
            m3_probability_improved,
        "m3_still_has_query_collapse": m3_collapsed,
        "m3_graph_conditioning_remains_homogeneous":
            m3_graph_conditioning_homogeneous,
        "m4_restores_query_identity": m4_identity_restored,
        "m4_covers_distinct_gt_branches": m4_distinct_gt_covered,
        "m4_ever_met_multibranch_metric_gate": bool(
            m4.get("curve_gate_history", {}).get(
                "qualifying_epoch_count", 0) > 0),
        "recommend_full_e1_e4_training": recommend_full,
        "decision_rule": (
            "recommend full E1-E4 only when M4 passes all optional "
            "sanity gates, graph/final cosine are below 0.95, "
            "distinct coverage is at least 0.90, and oracle-K "
            "duplicate ratio is at most 0.20"
        ),
    }


def _render_readme(comparison: Mapping[str, Any]) -> str:
    variants = comparison["variants"]
    decision = comparison["decision"]
    rows = []
    for name in VARIANT_NAMES:
        result = variants[name]
        representation = result["query_representation"]
        rows.append(
            "| {name} | {ap:.4f} | {slot:.4f} | {sep:.4f} | "
            "{exact:.4f} | {recall:.4f} | {coverage:.4f} | "
            "{duplicate:.4f} | {graph:.4f} | {final:.4f} | "
            "{passed} |".format(
                name=name,
                ap=result["branch_ap"],
                slot=result["slot_ap"],
                sep=result["probability_separation_mean"],
                exact=result["exact_count_accuracy"],
                recall=result["oracle_k_recall"],
                coverage=result["oracle_k_distinct_gt_coverage"],
                duplicate=result[
                    "oracle_k_duplicate_pair_ratio"],
                graph=representation[
                    "graph_conditioned_query"
                ]["pairwise_cosine"]["mean"],
                final=representation[
                    "final_fused_query"
                ]["pairwise_cosine"]["mean"],
                passed=result["sanity_gate_passed"],
            )
        )
    answer = lambda value: "Yes" if value else "No"
    return "\n".join((
        "# Stage 3C-R1 query identity comparison",
        "",
        "All variants use the same 32 teacher-forced training states, "
        "trajectory fragments, seed, and strictly loaded frozen RPNet.",
        "",
        "| Variant | Branch AP | Slot AP | Mean probability gap | "
        "Exact count | Oracle recall | Distinct coverage | "
        "Oracle duplicate | Graph cosine | Final cosine | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        *rows,
        "",
        "## Decisions",
        "",
        "1. Did M3 improve probability separation over M0? **{}**"
        .format(answer(decision[
            "m3_probability_separation_improved_over_m0"])),
        "2. Does M3 still have query collapse? **{}**".format(
            answer(decision["m3_still_has_query_collapse"])),
        "   Early graph-conditioned queries remain homogeneous? **{}**"
        .format(answer(decision[
            "m3_graph_conditioning_remains_homogeneous"])),
        "3. Does M4 restore query identity? **{}**".format(
            answer(decision["m4_restores_query_identity"])),
        "4. Does M4 cover distinct GT branches? **{}**".format(
            answer(decision["m4_covers_distinct_gt_branches"])),
        "5. Run full E1-E4 training? **{}**".format(
            answer(decision["recommend_full_e1_e4_training"])),
        "",
        "Qualifying-epoch counts (before the required final-checkpoint "
        "decision): M0={}, M3={}, M4={}.".format(
            variants["M0"]["curve_gate_history"][
                "qualifying_epoch_count"],
            variants["M3"]["curve_gate_history"][
                "qualifying_epoch_count"],
            variants["M4"]["curve_gate_history"][
                "qualifying_epoch_count"],
        ),
        "Final-checkpoint regression after a qualifying epoch: "
        "M0={}, M3={}, M4={}.".format(
            variants["M0"]["curve_gate_history"][
                "final_regressed_after_qualifying"],
            variants["M3"]["curve_gate_history"][
                "final_regressed_after_qualifying"],
            variants["M4"]["curve_gate_history"][
                "final_regressed_after_qualifying"],
        ),
        "",
        "No branch prediction was passed to `Path.push`; no anchor or "
        "RPNet architecture was changed.",
        "",
    ))


def main() -> None:
    args = _parse_args()
    output_root = args.output_dir.resolve(strict=False)
    if args.report_only:
        comparison_path = output_root / "comparison.json"
        if not comparison_path.is_file():
            raise FileNotFoundError(
                "comparison not found: {}".format(comparison_path))
        with comparison_path.open(
                "r", encoding="utf-8") as input_file:
            comparison = json.load(input_file)
        for name in VARIANT_NAMES:
            sanity_path = (
                output_root / name.lower() / "sanity_overfit.json")
            with sanity_path.open(
                    "r", encoding="utf-8") as input_file:
                sanity = json.load(input_file)
            comparison["variants"][name][
                "curve_gate_history"] = _curve_gate_history(sanity)
        comparison["decision"] = build_r1_decision(
            comparison["variants"])
        _write_json(comparison_path, comparison)
        with (output_root / "README.md").open(
                "w", encoding="utf-8") as output:
            output.write(_render_readme(comparison))
        print(json.dumps({
            "comparison": str(comparison_path),
            "decision": comparison["decision"],
        }, indent=2, sort_keys=True))
        return

    config_paths = {
        "M0": args.m0_config,
        "M3": args.m3_config,
        "M4": args.m4_config,
    }
    configs = {
        name: _load_config(path)
        for name, path in config_paths.items()
    }
    _validate_configs(configs)
    reference = configs["M0"]
    device = _resolve_device(
        args.device or str(reference.STAGE3C.DEVICE))
    dataset_dir = Path(reference.STAGE3C.DATASET_DIR)
    image_checkpoint = Path(reference.STAGE3C.IMAGE_CHECKPOINT)
    train_dataset = Stage3CBranchDataset(
        dataset_dir, "train", preload=True)
    selected_indices = list(_selected_indices(
        train_dataset, reference))
    data_fingerprint = _selected_data_fingerprint(
        train_dataset, selected_indices)
    rpnet, _ = _load_frozen_rpnet(
        reference, image_checkpoint, device)
    if not all(
            not parameter.requires_grad
            for parameter in rpnet.parameters()):
        raise RuntimeError("RPNet must remain frozen")

    output_root.mkdir(parents=True, exist_ok=True)
    variant_results = {}
    started_at = time.perf_counter()
    for name in VARIANT_NAMES:
        cfg = configs[name]
        variant_output = Path(
            cfg.STAGE3C.OUTPUT_DIR).resolve(strict=False)
        try:
            variant_output.relative_to(output_root)
        except ValueError:
            raise ValueError(
                "{} output must be inside {}".format(
                    name, output_root))
        if variant_output.exists():
            if not args.overwrite:
                raise FileExistsError(
                    "{} output exists; pass --overwrite: {}".format(
                        name, variant_output))
            shutil.rmtree(str(variant_output))
        _set_seed(int(cfg.STAGE3C.SEED))
        print("running {} sanity".format(name), flush=True)
        sanity = run_overfit_sanity(
            rpnet=rpnet,
            cfg=cfg,
            dataset=train_dataset,
            device=device,
            output_dir=variant_output,
            image_checkpoint=image_checkpoint,
        )
        if list(sanity["selected_dataset_indices"]) != selected_indices:
            raise RuntimeError(
                "{} selected a different sample set".format(name))
        checkpoint = variant_output / "sanity_overfit.pth.tar"
        diagnostics_dir = variant_output / "diagnostics"
        diagnostics = run_diagnostics(
            cfg=cfg,
            checkpoint=checkpoint,
            image_checkpoint=image_checkpoint,
            dataset_dir=dataset_dir,
            output_dir=diagnostics_dir,
            device=device,
            batch_size=len(selected_indices),
            split="train",
            dataset_indices=selected_indices,
        )
        representation = _representation_summary(
            diagnostics_dir / "query_similarity.json")
        variant_results[name] = _variant_summary(
            sanity=sanity,
            diagnostics=diagnostics,
            representation=representation,
            checkpoint=checkpoint,
        )
        print(
            "{} AP={:.4f} coverage={:.4f} duplicate={:.4f} "
            "gate={}".format(
                name,
                variant_results[name]["branch_ap"],
                variant_results[name][
                    "oracle_k_distinct_gt_coverage"],
                variant_results[name][
                    "oracle_k_duplicate_pair_ratio"],
                variant_results[name]["sanity_gate_passed"],
            ),
            flush=True,
        )

    comparison = {
        "schema_version": "stage3c-r1-v1",
        "source_commit": (
            "0c784c68c9604b68bf0dd437612df8e254fdaec5"),
        "sample_count": len(selected_indices),
        "selected_train_dataset_indices": selected_indices,
        "selected_data_sha256": data_fingerprint,
        "seed": int(reference.STAGE3C.SEED),
        "image_checkpoint": str(image_checkpoint.resolve()),
        "dataset_dir": str(dataset_dir.resolve()),
        "fresh_auxiliary_initialization": True,
        "optimizer_resume_used": False,
        "trajectory_modality_dropout": 0.0,
        "branch_predictions_feed_path_push": False,
        "config_paths": {
            name: str(path.resolve())
            for name, path in config_paths.items()
        },
        "variants": variant_results,
        "decision": build_r1_decision(variant_results),
        "elapsed_seconds": float(
            time.perf_counter() - started_at),
    }
    _write_json(output_root / "comparison.json", comparison)
    with (output_root / "README.md").open(
            "w", encoding="utf-8") as output:
        output.write(_render_readme(comparison))
    print(json.dumps({
        "comparison": str(
            (output_root / "comparison.json").resolve()),
        "decision": comparison["decision"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
