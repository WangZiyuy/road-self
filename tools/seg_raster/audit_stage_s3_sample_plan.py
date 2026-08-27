"""Generate and compare the frozen teacher-forced sample prefix for six runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from easydict import EasyDict

from utils.OSMDataset import OSMDataset
from utils.seg_raster.stage_s3 import (
    EXPERIMENT_MATRIX,
    STAGE_S3_SEED,
    audit_batch_parity,
    identity_sha256,
    load_stage_s3_config,
    sample_identity,
)


def array_group_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")


def config_for(spec, split: dict) -> EasyDict:
    matches = sorted((REPO_ROOT / "configs").glob(
        "stage_s3_{}_*yml".format(spec.key)))
    if len(matches) != 1:
        raise RuntimeError("missing unique config for {}".format(spec.key))
    config = load_stage_s3_config(matches[0])
    extent = split["train_extent"]
    config["TRAIN"]["SPATIAL_EXTENT_XYXY"] = [
        extent["x0"], extent["y0"], extent["x1"], extent["y1"]]
    control = config["TRAJ"]["RASTER"].get("CONTROL")
    if control is not None:
        config["DIR"]["TRAJ_DIR"] = (
            "data_self/stage_s3_seg_raster/runtime/controls/{}".format(control))
    return EasyDict(config)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-count", type=int, default=100)
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "artifacts/stage_s3_sample_plan.json")
    args = parser.parse_args()
    split = json.loads((
        REPO_ROOT / "artifacts/stage_s3_split_manifest.json"
    ).read_text(encoding="utf-8"))
    plan_rows = {}
    common_sha = {}
    raster_sha = {}
    reference_records = []
    for spec in EXPERIMENT_MATRIX:
        random.seed(STAGE_S3_SEED)
        np.random.seed(STAGE_S3_SEED)
        cfg = config_for(spec, split)
        dataset = OSMDataset(cfg, net=None, training=True)
        run_rows, run_common, run_raster = [], [], []
        for batch_index in range(args.batch_count):
            batch = dataset.get_batch()
            run_rows.extend(batch.batch_sample_metadata)
            run_common.append(array_group_sha256(
                batch.batch_inputs,
                batch.batch_walked_path_small,
                batch.batch_road_segmentation,
                batch.batch_junction_segmentation,
                batch.batch_target_maps,
                batch.batch_is_key_point,
                batch.batch_end_index))
            if spec.trajectory_mode == "raster_seg_only":
                run_raster.append({
                    "raster_sha256": array_group_sha256(batch.batch_traj_inputs),
                    "valid_mask_sha256": array_group_sha256(batch.batch_traj_valid_masks),
                    "shape": list(batch.batch_traj_inputs.shape),
                })
            if spec.key == "C0":
                reference_records.append({
                    "batch_index": batch_index,
                    "samples": batch.batch_sample_metadata,
                    "common_tensor_sha256": run_common[-1],
                })
            dataset.push_and_vis_batch(batch, 0, batch_index)
        plan_rows[spec.key] = run_rows
        common_sha[spec.key] = run_common
        raster_sha[spec.key] = run_raster

    parity = audit_batch_parity(plan_rows, count=args.batch_count * int(
        load_stage_s3_config(REPO_ROOT / "configs/stage_s3_common.yml")["TRAIN"]["BATCH_SIZE"]))
    common_reference = common_sha["C0"]
    common_mismatches = {
        key: [idx for idx, pair in enumerate(zip(common_reference, values))
              if pair[0] != pair[1]]
        for key, values in common_sha.items()
    }
    valid_reference = [row["valid_mask_sha256"] for row in raster_sha["C1"]]
    valid_mismatches = {
        key: [idx for idx, row in enumerate(values)
              if row["valid_mask_sha256"] != valid_reference[idx]]
        for key, values in raster_sha.items() if values
    }
    passed = (
        parity["status"] == "PASS"
        and not any(common_mismatches.values())
        and not any(valid_mismatches.values()))
    payload = {
        "stage": "seg_raster_stage_s3",
        "status": "PASS" if passed else "FAIL",
        "seed": STAGE_S3_SEED,
        "region": "xian",
        "follow_mode": "follow_target",
        "model_output_affects_sample_generation": False,
        "batch_count": args.batch_count,
        "sample_count": len(reference_records) * int(
            load_stage_s3_config(REPO_ROOT / "configs/stage_s3_common.yml")["TRAIN"]["BATCH_SIZE"]),
        "sample_order": reference_records,
        "augmentation": {
            "kind": "identity", "aerial_raster_mask_labels_synchronized": True},
        "run_sample_plan_sha256": {
            key: identity_sha256([sample_identity(row) for row in rows])
            for key, rows in plan_rows.items()},
        "first_100_batch_identity_sha256": {
            key: identity_sha256([
                identity_sha256([
                    sample_identity(row) for row in rows[offset:offset + 2]])
                for offset in range(0, min(len(rows), args.batch_count * 2), 2)])
            for key, rows in plan_rows.items()},
        "sample_identity_parity": parity,
        "common_tensor_mismatch_indices": common_mismatches,
        "valid_mask_mismatch_indices": valid_mismatches,
        "raster_control_batch_sha256": raster_sha,
        "split_manifest_sha256": split["manifest_sha256"],
    }
    payload["plan_sha256"] = identity_sha256(payload)
    write_json(args.output, payload)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
