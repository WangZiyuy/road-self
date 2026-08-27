#!/usr/bin/env python3
"""Generate static contract artifacts against a read-only dirty source tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from audit_stage_s0_static import (
    activation_contract,
    execution_matrix,
    static_source_provenance,
    train_infer_contract,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--snapshot-id", default="CURRENT_DIRTY_SNAPSHOT")
    args = parser.parse_args()
    source = args.source_repo.resolve()
    output = args.output_dir.resolve()
    provenance = static_source_provenance(source, args.snapshot_id)
    payloads = {
        "stage_s0_execution_path_matrix_current_dirty.json": execution_matrix(source),
        "stage_s0_activation_loss_contract_current_dirty.json": activation_contract(source),
        "stage_s0_train_infer_contract_current_dirty.json": train_infer_contract(source),
    }
    for name, payload in payloads.items():
        payload["source_provenance"] = provenance
        payload["audit_scope"] = "CURRENT_DIRTY_SNAPSHOT_READ_ONLY"
        write_json(output / name, payload)
    print(json.dumps({"status": "PASS", "artifacts": sorted(payloads)}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
