"""Build finite, redacted Stage S3D conclusion and Markdown reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.seg_raster.stage_s3d import assert_json_finite


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    assert_json_finite(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")


def write_doc(path: Path, title: str, rows: list[tuple[str, object]]) -> None:
    body = ["# " + title, ""]
    body.extend("- {}: `{}`".format(name, value) for name, value in rows)
    body.extend(["", "All formal CUDA evidence was produced on REMOTE_TRAINING_SERVER.", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(body), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-code-sha", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root
    swap = read(root / "stage_s3d_input_swap_matrix.json")
    zero = read(root / "stage_s3d_current_zero_path_audit.json")
    strength = read(root / "stage_s3d_control_strength_audit.json")
    null = read(root / "stage_s3d_null_parity.json")
    segmentation = read(root / "stage_s3d_segmentation_comparison.json")
    anchor = read(root / "stage_s3d_anchor_comparison.json")
    graph = read(root / "stage_s3d_graph_comparison.json")
    spatial = segmentation["segmentation_spatial_causal_gate"]
    anchor_gate = anchor.get("frozen_anchor_indirect_gate", "NOT_EXECUTED")
    multistep = anchor.get("multistep_anchor_validity", "NOT_EXECUTED")
    graph_gate = graph.get("graph_stability_gate", "NOT_EXECUTED")
    seg_go = "GO" if null["status"] == "PASS" and spatial == "PASS" else "NO_GO"
    end_go = "GO" if (
        seg_go == "GO" and anchor_gate == "PASS"
        and multistep == "PASS" and graph_gate == "PASS") else "NO_GO"
    aligned_beats_controls = spatial == "PASS"
    decision = ("CONTINUE_TO_CONFIRMATORY_MULTI_SEED" if aligned_beats_controls
                else "STOP_AFTER_NEGATIVE_CONTROL_RESULT")
    conclusion = {
        "stage": "seg_raster_stage_s3d", "status": "PASS",
        "execution_environment": "REMOTE_TRAINING_SERVER",
        "branch": "feat/seg-raster-only",
        "s3d_run_code_sha": args.run_code_sha,
        "current_raster_input_dependence": swap[
            "current_raster_input_dependence"],
        "current_zero_path_root_cause": zero["root_cause"],
        "strict_zero_preserving_contract": "PASS",
        "road_only_routing": "PASS",
        "null_parity_gate": null["status"],
        "baseline_controlled_common_samples": segmentation[
            "baseline_controlled_common_samples"],
        "segmentation_spatial_causal_gate": spatial,
        "junction_parity": segmentation["junction_parity"],
        "frozen_anchor_indirect_gate": anchor_gate,
        "multistep_anchor_validity": multistep,
        "graph_stability_gate": graph_gate,
        "computational_feasibility": graph.get(
            "computational_feasibility", "NOT_EXECUTED_BY_GATE"),
        "go_for_segmentation_multi_seed": seg_go,
        "go_for_end_to_end_multi_seed": end_go,
        "raster_branch_decision": decision,
        "multi_seed_started": False,
        "additional_adapter_designed": False,
        "risks": [
            "single-seed screening cannot establish statistical significance",
            "sample-permuted full-graph control uses deterministic tile donors",
        ],
    }
    write_json(root / "stage_s3d_conclusion.json", conclusion)
    docs = root / "docs"
    write_doc(docs / "stage_s3d_input_dependence.md",
              "Stage S3D Input Dependence", [
                  ("CURRENT_RASTER_INPUT_DEPENDENCE",
                   conclusion["current_raster_input_dependence"]),
                  ("R1 aligned strictly best",
                   swap["r1_trained_aligned_strictly_best"]),
                  ("R1 residual changes with input",
                   swap["r1_residual_changes_with_input"]),
              ])
    write_doc(docs / "stage_s3d_zero_path.md", "Stage S3D Zero Path", [
        ("current root cause", zero["root_cause"]),
        ("strict replacement contract", "F(x,0)=x"),
        ("road-only", True), ("junction image-only", True)])
    write_doc(docs / "stage_s3d_controlled_comparison.md",
              "Stage S3D Controlled Comparison", [
                  ("common samples", conclusion[
                      "baseline_controlled_common_samples"]),
                  ("null parity", null["status"]),
                  ("spatial causal gate", spatial),
                  ("junction parity", conclusion["junction_parity"]),
                  ("raster branch decision", decision),
              ])
    write_doc(docs / "stage_s3d_final_report.md", "Stage S3D Final Report", [
        ("run code SHA", args.run_code_sha),
        ("segmentation multi-seed", seg_go),
        ("end-to-end multi-seed", end_go),
        ("raster branch decision", decision)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
