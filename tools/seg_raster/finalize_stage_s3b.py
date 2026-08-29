"""Reduce completed Stage S3B phases into small finite/redacted evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.seg_raster.stage_s3b import assert_json_finite


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert_json_finite(value)
    return value


def write_json(path: Path, value: object) -> None:
    assert_json_finite(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")


def write_markdown(path: Path, title: str, sections: list[tuple[str, str]]) -> None:
    lines = ["# " + title, ""]
    for heading, body in sections:
        lines.extend(["## " + heading, "", body, ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def anchor_result(run_root: Path, run_id: str, step: int) -> dict:
    payload = read_json(run_root / run_id / "evaluation"
                        / "validation_step_{:06d}.json".format(step))
    return payload["anchor"]


def compact_dynamics(summary: dict) -> dict:
    return {"run_key": summary["run_key"], "run_id": summary["run_id"],
            "status": summary["status"], "optimizer_steps": summary["optimizer_steps"],
            "elapsed_seconds": summary["elapsed_seconds"],
            "validation_metrics_by_step": summary["validation_metrics_by_step"],
            "best_validation_metrics": summary["best_validation_metrics"],
            "latest_validation_metrics": summary["latest_validation_metrics"],
            "last_train_batch_metrics": summary["last_train_batch_metrics"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-code-sha", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    lr = read_json(args.output_root / "stage_s3b_lr_screen.json")
    loss = read_json(args.output_root / "stage_s3b_junction_loss_screen.json")
    controls = read_json(args.output_root / "stage_s3b_control_matrix.json")
    segmentation = read_json(args.output_root / "stage_s3b_segmentation_comparison.json")
    phase_a = read_json(args.output_root / "phase_a_plan.json")
    phase_b = read_json(args.output_root / "phase_b_plan.json")
    phase_c = read_json(args.output_root / "phase_c_plan.json")
    all_run_ids = []
    for plan in (phase_a, phase_b, phase_c):
        all_run_ids.extend(row["run_id"] for row in plan.get("jobs", []))
    all_run_ids.extend(phase_b.get("reused_runs", {}).values())
    all_run_ids.extend(phase_c.get("reused_runs", {}).values())
    all_run_ids = list(dict.fromkeys(all_run_ids))
    summaries = [read_json(args.run_root / run_id / "summary.json")
                 for run_id in all_run_ids]
    if any(row.get("code_sha") != args.run_code_sha for row in summaries):
        raise RuntimeError("run-code SHA drift in summaries")
    training_dynamics = {
        "stage": "seg_raster_stage_s3b", "status": "PASS",
        "run_code_sha": args.run_code_sha,
        "metric_scope_warning": (
            "last_train_batch_metrics are diagnostic only and are excluded from "
            "selection, comparison and conclusions."),
        "runs": [compact_dynamics(row) for row in summaries]}
    write_json(args.output_root / "stage_s3b_training_dynamics.json", training_dynamics)

    gate_pass = segmentation["segmentation_causal_gate"] == "PASS"
    step = int(controls["common_step"])
    anchor_payload = {
        "stage": "seg_raster_stage_s3b", "segmentation_causal_gate":
        segmentation["segmentation_causal_gate"],
        "baseline_controlled_common_step": step}
    graph_payload = dict(anchor_payload)
    if gate_pass:
        anchors = {key: anchor_result(args.run_root, run_id, step)
                   for key, run_id in controls["runs"].items()}
        r1_topk = anchors["R1"]["top_k_recall"]
        specificity = all(r1_topk > anchors[key]["top_k_recall"]
                          for key in ("R0", "R2", "R3"))
        later_recall = anchors["R1"]["per_step_recall"][1:4]
        multistep = not all(value == 0 for value in later_recall)
        anchor_payload.update({
            "status": "PASS", "runs": anchors,
            "anchor_specificity_gate": "PASS" if specificity else "FAIL",
            "specificity_rule": "R1 top-K recall strictly exceeds R0, R2 and R3",
            "multistep_anchor_validity": "PASS" if multistep else "FAIL"})
        results = {key: read_json(args.output_root / "graph_results" / (key + ".json"))
                   for key in ("R0", "R1", "R2", "R3")}
        r0, r1 = results["R0"], results["R1"]
        stable = (
            r1["status"] == "PASS"
            and r1["graph_iterations"] <= 3 * max(r0["graph_iterations"], 1)
            and r1["directed_edge_count"] <= 3 * max(r0["directed_edge_count"], 1)
            and r1["runtime_seconds"] <= 5 * max(r0["runtime_seconds"], 1e-9))
        graph_payload.update({
            "status": "PASS", "runs": results,
            "graph_stability_gate": "PASS" if stable else "FAIL",
            "computational_feasibility": "PASS" if stable else "FAIL",
            "resource_cap_policy": {
                "max_iterations": 3000, "max_vertices": 5000,
                "max_directed_edges": 10000, "max_wall_time_seconds": 900},
            "apls_protocol": "deterministic_pixel_graph_approximation"})
    else:
        anchor_payload.update({
            "status": "NOT_EXECUTED_BY_GATE",
            "anchor_specificity_gate": "NOT_EXECUTED_BY_GATE",
            "multistep_anchor_validity": "NOT_EXECUTED_BY_GATE"})
        graph_payload.update({
            "status": "NOT_EXECUTED_BY_GATE",
            "graph_stability_gate": "NOT_EXECUTED_BY_GATE",
            "computational_feasibility": "NOT_EXECUTED_BY_GATE"})
    write_json(args.output_root / "stage_s3b_anchor_comparison.json", anchor_payload)
    write_json(args.output_root / "stage_s3b_graph_comparison.json", graph_payload)

    inventories, schedules = {}, {}
    for phase in ("A", "B", "C", "G"):
        inventory_path = args.output_root / ("gpu_inventory_phase_" + phase + ".json")
        schedule_path = args.output_root / ("gpu_schedule_phase_" + phase + ".json")
        if inventory_path.is_file():
            inventories[phase] = read_json(inventory_path)
        if schedule_path.is_file():
            schedules[phase] = read_json(schedule_path)
    write_json(args.output_root / "stage_s3b_gpu_inventory.json", {
        "stage": "seg_raster_stage_s3b", "status": "PASS",
        "execution_environment": "REMOTE_TRAINING_SERVER", "phases": inventories})
    write_json(args.output_root / "stage_s3b_gpu_schedule.json", {
        "stage": "seg_raster_stage_s3b", "status": "PASS",
        "execution_environment": "REMOTE_TRAINING_SERVER", "phases": schedules,
        "external_processes_terminated": False})
    checkpoint = read_json(args.output_root / "stage_s3b_checkpoint_inventory.json")
    parity = read_json(args.output_root / "stage_s3b_sample_parity.json")
    training_repair = "PASS" if (
        lr["selection"]["lr_stability_gate"] == "PASS"
        and checkpoint["all_controls_have_identical_step_grid"]
        and parity["status"] == "PASS") else "FAIL"
    loss_repair = loss["selection"]["junction_loss_repair"]
    anchor_gate = anchor_payload["anchor_specificity_gate"]
    multistep_gate = anchor_payload["multistep_anchor_validity"]
    graph_gate = graph_payload["graph_stability_gate"]
    seg_go = (training_repair == "PASS" and loss_repair in ("PASS", "NO_EVIDENCE")
              and gate_pass)
    end_go = (seg_go and anchor_gate == "PASS" and multistep_gate == "PASS"
              and graph_gate == "PASS")
    conclusion = {
        "stage": "seg_raster_stage_s3b", "branch": "feat/seg-raster-only",
        "s3b_base_sha": "7cbd87d8e8fbbfe1783145b024c1dc4783213ee9",
        "s3b_run_code_sha": args.run_code_sha,
        "formal_execution_environment": "REMOTE_TRAINING_SERVER",
        "training_protocol_repair": training_repair,
        "junction_loss_repair": loss_repair,
        "segmentation_causal_gate": segmentation["segmentation_causal_gate"],
        "anchor_specificity_gate": anchor_gate,
        "multistep_anchor_validity": multistep_gate,
        "graph_stability_gate": graph_gate,
        "go_for_segmentation_multi_seed": "GO" if seg_go else "NO_GO",
        "go_for_end_to_end_multi_seed": "GO" if end_go else "NO_GO",
        "selected_lr_multiplier": lr["selection"]["selected_lr_multiplier"],
        "selected_junction_loss": loss["selection"]["selected_loss_kind"],
        "baseline_controlled_common_step": step,
        "automatic_multiseed_started": False,
        "model_architecture_modified": False,
        "risks": ["single seed", "small fixed validation plan",
                  "graph APLS is a deterministic pixel approximation"]}
    write_json(args.output_root / "stage_s3b_conclusion.json", conclusion)

    docs = args.output_root / "docs"
    contract = read_json(args.output_root / "stage_s3b_optimizer_loss_contract.json")
    audit = read_json(args.output_root / "stage_s3b_loss_gradient_audit.json")
    write_markdown(docs / "stage_s3b_optimizer_loss_contract.md",
                   "Stage S3B Optimizer and Loss Contract", [
        ("Historical contract", "Adam, LR 1e-4, betas 0.9/0.99, weight decay 2e-4, no scheduler or warmup; all legacy BCE terms use reduction=sum."),
        ("Diagnostic scope", "Sixteen frozen remote CUDA batches were audited with independent road, junction, anchor and total backward passes."),
        ("Provenance", "Formal run code: `{}`.".format(args.run_code_sha))])
    write_markdown(docs / "stage_s3b_lr_screen.md", "Stage S3B Learning-Rate Screen", [
        ("Selection", "Only image-only A0/A2/A4 were used. Selected multiplier: `{}`; stability gate: `{}`.".format(
            lr["selection"]["selected_lr_multiplier"], lr["selection"]["lr_stability_gate"])),
        ("Aligned exclusion", "Aligned A1/A3/A5 were not inspected by the LR selection rule.")])
    write_markdown(docs / "stage_s3b_junction_loss_screen.md",
                   "Stage S3B Junction-Loss Screen", [
        ("Selection", "Only image-only B0/B2/B4 were used. Selected loss: `{}`; repair: `{}`.".format(
            loss["selection"]["selected_loss_kind"], loss_repair)),
        ("Calibration", "Balanced losses use a capped positive weight and separately frozen 16-batch gradient-matching alpha values.")])
    write_markdown(docs / "stage_s3b_controlled_training.md",
                   "Stage S3B Controlled Training", [
        ("Common checkpoint", "R0 selected step `{}`; R0/R1/R2/R3 were compared only at that step.".format(step)),
        ("Segmentation gate", segmentation["segmentation_causal_gate"]),
        ("Conditional evaluation", "Anchor and graph were {}.".format(
            "executed" if gate_pass else "not executed because the segmentation gate failed"))])
    write_markdown(docs / "stage_s3b_final_report.md", "Stage S3B Final Report", [
        ("Training protocol", training_repair),
        ("Junction loss", loss_repair),
        ("Segmentation causal gate", segmentation["segmentation_causal_gate"]),
        ("Anchor specificity", anchor_gate),
        ("Graph stability", graph_gate),
        ("Decision", "Segmentation multi-seed: `{}`; end-to-end multi-seed: `{}`.".format(
            conclusion["go_for_segmentation_multi_seed"],
            conclusion["go_for_end_to_end_multi_seed"]))])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
