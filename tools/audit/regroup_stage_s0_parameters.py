#!/usr/bin/env python3
"""Regroup existing runtime parameter evidence without re-running the model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_legacy_dsf_runtime import aggregate_parameters, module_group, now_iso, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    repo = args.repo.resolve()
    runtime_path = repo / "artifacts" / "stage_s0_runtime_audit.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    gradient_runs = [item for item in runtime["runs"] if item.get("backward_requested")]
    for run in gradient_runs:
        for parameter in run.get("parameters", []):
            parameter["module"] = module_group(parameter["parameter_name"])
    parameter_payload = {
        "stage": "seg_raster_stage_s0", "generated_at": now_iso(),
        "source": "existing stage_s0_runtime_audit.json; no model re-execution",
        "summaries": [aggregate_parameters(item) for item in gradient_runs],
        "runs": [{"label": item["label"], "status": item["status"], "parameters": item.get("parameters", [])} for item in gradient_runs],
    }
    gradient_payload = {
        "stage": "seg_raster_stage_s0", "generated_at": now_iso(),
        "source": "existing stage_s0_runtime_audit.json; no model re-execution",
        "loss_contract": "Production output set and BCEWithLogits semantics; audit uses mean rather than production sum reduction.",
        "summaries": [aggregate_parameters(item) for item in gradient_runs],
        "runs": [{
            "label": item["label"], "status": item["status"], "loss": item.get("loss"),
            "parameters": item.get("parameters", []), "error": item.get("error", ""),
        } for item in gradient_runs],
    }
    write_json(repo / "artifacts" / "stage_s0_parameter_inventory.json", parameter_payload)
    write_json(repo / "artifacts" / "stage_s0_gradient_inventory.json", gradient_payload)
    print(json.dumps({"status": "PASS", "runs": [item["label"] for item in gradient_runs]}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
