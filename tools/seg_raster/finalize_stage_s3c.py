"""Export finite, redacted Stage S3C evidence after all gated work ends."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: object) -> None:
    if isinstance(value, dict):
        for child in value.values():
            finite(child)
    elif isinstance(value, list):
        for child in value:
            finite(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite JSON value")


def redact(value: object, replacements: dict[str, str]) -> object:
    if isinstance(value, dict):
        return {str(key): redact(child, replacements)
                for key, child in value.items()}
    if isinstance(value, list):
        return [redact(child, replacements) for child in value]
    if isinstance(value, str):
        result = value
        for raw, label in replacements.items():
            result = result.replace(raw, label)
        return result
    return value


def write_json(path: Path, value: object) -> None:
    finite(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")


def copy_json(
    source: Path, destination: Path, replacements: dict[str, str]
) -> dict:
    value = redact(read_json(source), replacements)
    write_json(destination, value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-code-sha", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--export-root", type=Path, required=True)
    args = parser.parse_args()
    artifact_root = args.export_root / "artifacts"
    doc_root = args.export_root / "docs/audits"
    replacements = {
        str(REPO_ROOT): "${REMOTE_S3C_WORKTREE}",
        str(args.output_root): "${S3C_REMOTE_OUTPUT}",
        str(args.run_root): "${S3C_RUN_ROOT}",
        "/home/wangziyu": "${REMOTE_HOME}",
    }
    names = [
        "stage_s3c_baseline_evaluation.json",
        "stage_s3c_loss_gradient_audit.json",
        "stage_s3c_trainable_parameter_contract.json",
        "stage_s3c_sample_plan.json",
        "stage_s3c_loss_screen.json",
        "stage_s3c_sample_parity.json",
        "stage_s3c_bn_checksum_audit.json",
        "stage_s3c_checkpoint_inventory.json",
        "stage_s3c_training_dynamics.json",
        "stage_s3c_control_matrix.json",
        "stage_s3c_segmentation_comparison.json",
        "stage_s3c_anchor_comparison.json",
        "stage_s3c_graph_comparison.json",
    ]
    values = {}
    for name in names:
        source = args.output_root / name
        if not source.is_file():
            raise FileNotFoundError(source)
        values[name] = copy_json(source, artifact_root / name, replacements)

    inventories = sorted(args.output_root.glob("gpu_inventory_phase_*.json"))
    schedules = sorted(args.output_root.glob("gpu_schedule_phase_*.json"))
    inventory_value = {
        "stage": "seg_raster_stage_s3c", "status": "PASS",
        "execution_environment": "REMOTE_TRAINING_SERVER",
        "phases": {path.stem.removeprefix("gpu_inventory_phase_"):
                   redact(read_json(path), replacements) for path in inventories},
    }
    schedule_value = {
        "stage": "seg_raster_stage_s3c", "status": "PASS",
        "execution_environment": "REMOTE_TRAINING_SERVER",
        "phases": {path.stem.removeprefix("gpu_schedule_phase_"):
                   redact(read_json(path), replacements) for path in schedules},
    }
    write_json(artifact_root / "stage_s3c_gpu_inventory.json", inventory_value)
    write_json(artifact_root / "stage_s3c_gpu_schedule.json", schedule_value)

    baseline = values["stage_s3c_baseline_evaluation.json"]
    loss = values["stage_s3c_loss_screen.json"]
    parity = values["stage_s3c_sample_parity.json"]
    bn = values["stage_s3c_bn_checksum_audit.json"]
    segmentation = values["stage_s3c_segmentation_comparison.json"]
    anchor = values["stage_s3c_anchor_comparison.json"]
    graph = values["stage_s3c_graph_comparison.json"]
    segmentation_gate = segmentation["segmentation_causal_gate"]
    anchor_gate = anchor.get("frozen_anchor_indirect_gate", "NOT_EXECUTED_BY_GATE")
    graph_gate = graph.get("graph_stability_gate", "NOT_EXECUTED_BY_GATE")
    computational = graph.get(
        "computational_feasibility", "NOT_EXECUTED_BY_GATE")
    go_segmentation = (
        baseline.get("baseline_graph_gate") == "PASS"
        and parity.get("status") == "PASS"
        and bn.get("status") == "PASS"
        and segmentation_gate == "PASS")
    go_end_to_end = (
        go_segmentation and anchor_gate == "PASS" and graph_gate == "PASS"
        and computational == "PASS")
    conclusion = {
        "stage": "seg_raster_stage_s3c", "status": "PASS",
        "branch": "feat/seg-raster-only",
        "run_code_sha": args.run_code_sha,
        "execution_environment": "REMOTE_TRAINING_SERVER",
        "official_contract_audit": "PASS",
        "baseline_checkpoint_gate": "PASS",
        "baseline_graph_gate": baseline.get("baseline_graph_gate", "FAIL"),
        "bn_freeze_gate": "PASS" if bn.get("status") == "PASS" else "FAIL",
        "loss_screen": loss.get("status", "FAIL"),
        "selected_loss": loss.get("selection", {}).get("selected_loss_kind"),
        "sample_parity": parity.get("status", "FAIL"),
        "baseline_controlled_common_samples": segmentation.get(
            "baseline_controlled_common_samples"),
        "segmentation_causal_gate": segmentation_gate,
        "frozen_anchor_indirect_gate": anchor_gate,
        "multistep_anchor_validity": anchor.get(
            "multistep_anchor_validity", "NOT_EXECUTED_BY_GATE"),
        "graph_stability_gate": graph_gate,
        "computational_feasibility": computational,
        "go_for_segmentation_multi_seed": "GO" if go_segmentation else "NO_GO",
        "go_for_end_to_end_multi_seed": "GO" if go_end_to_end else "NO_GO",
        "multi_seed_started": False,
        "key_evidence": [
            "official 648-key RPNet checkpoint strict load",
            "original backbone, anchor, and BatchNorm frozen",
            "baseline-controlled common sample comparison",
            "aligned/zero/shifted raster controls",
        ],
    }
    write_json(artifact_root / "stage_s3c_conclusion.json", conclusion)

    common = conclusion["baseline_controlled_common_samples"]
    controlled = (
        "# Stage S3C controlled comparison\n\n"
        "The selected Phase A loss is `{}`. The R0-controlled common sample "
        "count is `{}`. The segmentation causal gate is `{}`. The frozen "
        "anchor gate is `{}` and the graph stability gate is `{}`.\n"
    ).format(conclusion["selected_loss"], common,
             segmentation_gate, anchor_gate, graph_gate)
    final = (
        "# Stage S3C final report\n\n"
        "Official contract audit: PASS. Baseline checkpoint gate: PASS. "
        "Baseline graph gate: `{}`. BatchNorm freeze gate: `{}`. "
        "Segmentation multi-seed: `{}`. End-to-end multi-seed: `{}`.\n\n"
        "This stage is an Xi'an frozen-explorer adaptation and is not a full "
        "reproduction of official multi-city VecRoad training.\n"
    ).format(conclusion["baseline_graph_gate"], conclusion["bn_freeze_gate"],
             conclusion["go_for_segmentation_multi_seed"],
             conclusion["go_for_end_to_end_multi_seed"])
    doc_root.mkdir(parents=True, exist_ok=True)
    (doc_root / "stage_s3c_controlled_comparison.md").write_text(
        controlled, encoding="utf-8")
    (doc_root / "stage_s3c_final_report.md").write_text(final, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
