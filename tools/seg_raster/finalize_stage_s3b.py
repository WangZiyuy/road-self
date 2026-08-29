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


LR_GATE_NOT_EXECUTED = "NOT_EXECUTED_BY_LR_STABILITY_GATE"


def phase_a_summaries(args, phase_a: dict) -> list[dict]:
    run_ids = [row["run_id"] for row in phase_a.get("jobs", [])]
    summaries = [read_json(args.run_root / run_id / "summary.json")
                 for run_id in run_ids]
    if any(row.get("code_sha") != args.run_code_sha for row in summaries):
        raise RuntimeError("run-code SHA drift in Phase A summaries")
    if any(row.get("status") != "PASS" for row in summaries):
        raise RuntimeError("Phase A contains a non-PASS run")
    expected_steps = list(range(0, 20481, 2560))
    expected_keys = {str(step) for step in expected_steps}
    for row in summaries:
        if set(row.get("validation_metrics_by_step", {})) != expected_keys:
            raise RuntimeError("incomplete Phase A checkpoint grid: "
                               + row.get("run_id", "UNKNOWN"))
        if len(row.get("checkpoint_inventory", [])) != len(expected_steps):
            raise RuntimeError("incomplete Phase A checkpoint inventory: "
                               + row.get("run_id", "UNKNOWN"))
    return summaries


def finalize_lr_stability_failure(args, lr: dict, phase_a: dict) -> int:
    summaries = phase_a_summaries(args, phase_a)
    expected_steps = list(range(0, 20481, 2560))
    training_dynamics = {
        "stage": "seg_raster_stage_s3b", "status": "PASS",
        "run_code_sha": args.run_code_sha,
        "reducer_code_sha": args.reducer_code_sha,
        "metric_scope_warning": (
            "last_train_batch_metrics are diagnostic only and are excluded from "
            "selection, comparison and conclusions."),
        "runs": [compact_dynamics(row) for row in summaries]}
    write_json(args.output_root / "stage_s3b_training_dynamics.json",
               training_dynamics)

    checkpoint_runs = {
        row["run_key"]: row["checkpoint_inventory"] for row in summaries}
    write_json(args.output_root / "stage_s3b_checkpoint_inventory.json", {
        "stage": "seg_raster_stage_s3b", "status": "PASS",
        "scope": "PHASE_A_ONLY_LR_STABILITY_GATE_FAILED",
        "expected_steps": expected_steps,
        "versioned_model_checkpoint_count": sum(
            len(value) for value in checkpoint_runs.values()),
        "all_phase_a_runs_have_identical_step_grid": True,
        "phase_b_c_controls_not_created": True,
        "runs": checkpoint_runs})

    parity = {
        row["run_key"]: {
            "sample_identity_sha256": row["first_100_batch_identity_sha256"],
            "common_tensor_sha256": row["first_100_common_tensor_sha256"],
            "valid_mask_sha256": row.get("first_100_valid_mask_sha256")}
        for row in summaries}
    sample_pass = len({row["sample_identity_sha256"]
                       for row in parity.values()}) == 1
    common_pass = len({row["common_tensor_sha256"]
                       for row in parity.values()}) == 1
    raster_rows = [row for row in parity.values()
                   if row["valid_mask_sha256"] is not None]
    mask_pass = bool(raster_rows) and len({row["valid_mask_sha256"]
                                          for row in raster_rows}) == 1
    write_json(args.output_root / "stage_s3b_sample_parity.json", {
        "stage": "seg_raster_stage_s3b",
        "status": "PASS" if sample_pass and common_pass and mask_pass else "FAIL",
        "scope": "FIRST_100_BATCHES_ACROSS_PHASE_A",
        "first_100_batches": parity,
        "sample_identity_parity": sample_pass,
        "common_tensor_parity": common_pass,
        "raster_valid_mask_parity": mask_pass,
        "image_only_valid_mask": "NOT_APPLICABLE_NOT_LOADED"})

    placeholder = {
        "stage": "seg_raster_stage_s3b",
        "status": LR_GATE_NOT_EXECUTED,
        "blocking_gate": "LR_STABILITY_GATE=FAIL",
        "run_code_sha": args.run_code_sha,
        "reducer_code_sha": args.reducer_code_sha}
    loss_payload = dict(placeholder)
    loss_payload["selection"] = {
        "selected_loss_kind": "NOT_SELECTED",
        "junction_loss_repair": LR_GATE_NOT_EXECUTED}
    write_json(args.output_root / "stage_s3b_junction_loss_screen.json",
               loss_payload)
    threshold_payload = dict(placeholder)
    threshold_payload.update({
        "selected_threshold": "NOT_SELECTED",
        "fixed_0_3_reported_in_phase_a": True,
        "per_run_threshold_tuning_allowed": False})
    write_json(args.output_root / "stage_s3b_shared_threshold.json",
               threshold_payload)
    for name in ("stage_s3b_control_matrix.json",
                 "stage_s3b_segmentation_comparison.json",
                 "stage_s3b_anchor_comparison.json",
                 "stage_s3b_graph_comparison.json"):
        write_json(args.output_root / name, dict(placeholder))

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
        "execution_environment": "REMOTE_TRAINING_SERVER",
        "phases": inventories})
    write_json(args.output_root / "stage_s3b_gpu_schedule.json", {
        "stage": "seg_raster_stage_s3b", "status": "PASS",
        "execution_environment": "REMOTE_TRAINING_SERVER",
        "phases": schedules,
        "external_processes_terminated": False})

    conclusion = {
        "stage": "seg_raster_stage_s3b", "branch": "feat/seg-raster-only",
        "s3b_base_sha": "7cbd87d8e8fbbfe1783145b024c1dc4783213ee9",
        "s3b_run_code_sha": args.run_code_sha,
        "reducer_code_sha": args.reducer_code_sha,
        "formal_execution_environment": "REMOTE_TRAINING_SERVER",
        "phase_a_status": "PASS",
        "phase_b_status": LR_GATE_NOT_EXECUTED,
        "phase_c_status": LR_GATE_NOT_EXECUTED,
        "phase_g_status": LR_GATE_NOT_EXECUTED,
        "training_protocol_repair": "FAIL",
        "lr_stability_gate": "FAIL",
        "junction_loss_repair": LR_GATE_NOT_EXECUTED,
        "segmentation_causal_gate": LR_GATE_NOT_EXECUTED,
        "anchor_specificity_gate": LR_GATE_NOT_EXECUTED,
        "multistep_anchor_validity": LR_GATE_NOT_EXECUTED,
        "graph_stability_gate": LR_GATE_NOT_EXECUTED,
        "computational_feasibility": LR_GATE_NOT_EXECUTED,
        "go_for_segmentation_multi_seed": "NO_GO",
        "go_for_end_to_end_multi_seed": "NO_GO",
        "selected_lr_multiplier_for_diagnosis_only":
            lr["selection"]["selected_lr_multiplier"],
        "selected_junction_loss": "NOT_SELECTED",
        "baseline_controlled_common_step": "NOT_SELECTED",
        "formal_phase_a_runs": {row["run_key"]: row["status"]
                                for row in summaries},
        "invalidated_pre_optimizer_attempts": [
            {"code_sha_prefix": "b494a0d",
             "status": "INVALIDATED_BY_CODE_CHANGE"},
            {"code_sha_prefix": "8c0f50b",
             "status": "INVALIDATED_BY_CODE_CHANGE"}],
        "automatic_multiseed_started": False,
        "model_architecture_modified": False,
        "risks": [
            "All image-only LR candidates had retention below 0.70.",
            "Phase B junction-loss repair was not evaluated.",
            "Controlled aligned/zero/shifted causal comparison was not run.",
            "Single seed and a small fixed validation plan remain limitations."]}
    write_json(args.output_root / "stage_s3b_conclusion.json", conclusion)

    docs = args.output_root / "docs"
    write_markdown(docs / "stage_s3b_optimizer_loss_contract.md",
                   "Stage S3B Optimizer and Loss Contract", [
        ("Historical contract", "Adam, LR 1e-4, betas 0.9/0.99, weight decay 2e-4, no scheduler or warmup; all legacy BCE terms use reduction=sum."),
        ("Diagnostic scope", "Sixteen frozen remote CUDA batches were audited with independent road, junction, anchor and total backward passes."),
        ("Provenance", "Formal training code: `{}`; result reducer: `{}`.".format(
            args.run_code_sha, args.reducer_code_sha))])
    candidates = lr["image_only_candidates"]
    rows = ["- {}: LR×{}, best repair composite {:.6f}, retention {:.6f}".format(
        row["run_key"], row["lr_multiplier"],
        row["best_repair_composite"], row["retention"])
        for row in candidates]
    write_markdown(docs / "stage_s3b_lr_screen.md",
                   "Stage S3B Learning-Rate Screen", [
        ("Image-only selection", "\n".join(rows)),
        ("Gate", "No candidate reached retention >= 0.70. The LR stability gate is `FAIL`; Phase B and all later phases were not executed."),
        ("Selection scope", "Aligned A1/A3/A5 were excluded from LR selection exactly as preregistered.")])
    write_markdown(docs / "stage_s3b_junction_loss_screen.md",
                   "Stage S3B Junction-Loss Screen", [
        ("Status", LR_GATE_NOT_EXECUTED),
        ("Reason", "The preregistered LR stability gate failed, so no Phase B loss run was launched and no junction-loss claim is made.")])
    write_markdown(docs / "stage_s3b_controlled_training.md",
                   "Stage S3B Controlled Training", [
        ("Status", LR_GATE_NOT_EXECUTED),
        ("Boundary", "Phase C aligned/zero/shifted controls, conditional anchor evaluation, and graph evaluation were not executed.")])
    write_markdown(docs / "stage_s3b_final_report.md",
                   "Stage S3B Final Report", [
        ("Phase A", "Six of six formal LR-screen runs completed 20,480 optimizer steps and passed integrity checks."),
        ("Training protocol repair", "`FAIL`: retention stayed below 0.70 for every image-only LR candidate."),
        ("Downstream phases", LR_GATE_NOT_EXECUTED),
        ("Decision", "Segmentation multi-seed: `NO_GO`; end-to-end multi-seed: `NO_GO`. No additional training was started.")])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-code-sha", required=True)
    parser.add_argument("--reducer-code-sha", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    lr = read_json(args.output_root / "stage_s3b_lr_screen.json")
    phase_a = read_json(args.output_root / "phase_a_plan.json")
    if lr["selection"]["lr_stability_gate"] != "PASS":
        return finalize_lr_stability_failure(args, lr, phase_a)
    loss = read_json(args.output_root / "stage_s3b_junction_loss_screen.json")
    controls = read_json(args.output_root / "stage_s3b_control_matrix.json")
    segmentation = read_json(args.output_root / "stage_s3b_segmentation_comparison.json")
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
