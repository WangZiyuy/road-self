import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.stage3c_branch_dataset import (
    Stage3CBranchDataset,
    array_schema_from_mapping,
    validate_stage3c_dataset,
    write_stage3c_manifest,
    write_stage3c_shard,
)


def _synthetic_arrays(sample_count=2):
    zeros = np.zeros
    arrays = {
        "aerial_image": zeros(
            (sample_count, 3, 8, 8), dtype=np.uint8),
        "walked_path": zeros(
            (sample_count, 1, 2, 2), dtype=np.uint8),
        "incoming_dir": zeros(
            (sample_count, 2), dtype=np.float32),
        "incoming_valid": zeros(
            sample_count, dtype=np.bool_),
        "explored_edge_dirs": zeros(
            (sample_count, 2, 2), dtype=np.float32),
        "explored_edge_mask": zeros(
            (sample_count, 2), dtype=np.bool_),
        "explored_is_incoming": zeros(
            (sample_count, 2), dtype=np.bool_),
        "is_key_point": zeros(sample_count, dtype=np.bool_),
        "branch_offsets_norm": zeros(
            (sample_count, 3, 2), dtype=np.float32),
        "branch_directions": zeros(
            (sample_count, 3, 2), dtype=np.float32),
        "branch_mask": zeros(
            (sample_count, 3), dtype=np.bool_),
        "branch_count": zeros(sample_count, dtype=np.int64),
        "traj_xy_norm": zeros(
            (sample_count, 4, 5, 2), dtype=np.float32),
        "traj_time_delta": zeros(
            (sample_count, 4, 5), dtype=np.float32),
        "point_mask": zeros(
            (sample_count, 4, 5), dtype=np.bool_),
        "fragment_mask": zeros(
            (sample_count, 4), dtype=np.bool_),
        "point_inside_mask": zeros(
            (sample_count, 4, 5), dtype=np.bool_),
        "segment_only": zeros(
            (sample_count, 4), dtype=np.bool_),
        "track_indices": np.full(
            (sample_count, 4), -1, dtype=np.int64),
        "start_point_indices": np.full(
            (sample_count, 4), -1, dtype=np.int64),
        "end_point_indices": np.full(
            (sample_count, 4), -1, dtype=np.int64),
        "total_fragment_count": zeros(
            sample_count, dtype=np.int64),
        "kept_fragment_count": zeros(
            sample_count, dtype=np.int64),
        "truncated_fragment_count": zeros(
            sample_count, dtype=np.int64),
        "trajectory_point_truncated_count": zeros(
            sample_count, dtype=np.int64),
        "center_xy": zeros(
            (sample_count, 2), dtype=np.float32),
        "subtile_index": zeros(
            sample_count, dtype=np.int64),
        "vertex_id": zeros(sample_count, dtype=np.int64),
    }
    arrays["aerial_image"][0, 0, 0, 0] = 255
    arrays["walked_path"][0, 0, 1, 1] = 1
    arrays["center_xy"][0] = [0.0, 0.0]
    arrays["traj_xy_norm"][0, 0, 0] = [0.0, 0.0]
    arrays["point_mask"][0, 0, 0] = True
    arrays["fragment_mask"][0, 0] = True
    arrays["track_indices"][0, 0] = 42
    arrays["branch_offsets_norm"][0, 0] = [0.5, 0.0]
    arrays["branch_directions"][0, 0] = [1.0, 0.0]
    arrays["branch_mask"][0, 0] = True
    arrays["branch_count"][0] = 1
    return arrays


class Stage3CBranchDatasetTest(unittest.TestCase):
    def test_pickle_free_shard_round_trip(self):
        arrays = _synthetic_arrays()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            shard = write_stage3c_shard(
                root / "train" / "train_0000.npz", arrays)
            write_stage3c_manifest(root, {
                "array_schema": array_schema_from_mapping(arrays),
                "splits": {
                    "train": {
                        "sample_count": 2,
                        "shards": [shard],
                        "path_indices": [0],
                    }
                },
            })
            report = validate_stage3c_dataset(root)
            self.assertEqual(
                report["splits"]["train"]["sample_count"], 2)
            with np.load(
                    root / "train" / "train_0000.npz",
                    allow_pickle=False) as archive:
                self.assertFalse(any(
                    archive[key].dtype.hasobject
                    for key in archive.files
                ))

            dataset = Stage3CBranchDataset(
                root, "train", preload=True)
            self.assertEqual(len(dataset), 2)
            sample = dataset[0]
            self.assertEqual(
                tuple(sample["aerial_image"].shape), (3, 8, 8))
            self.assertEqual(
                float(sample["aerial_image"][0, 0, 0]), 1.0)
            self.assertTrue(
                bool(sample["trajectory_batch"]["point_mask"][0, 0]))
            self.assertEqual(
                int(sample["trajectory_batch"]["track_indices"][0]),
                42,
            )
            self.assertEqual(
                sample["metadata"]["center_xy"].tolist(),
                [0.0, 0.0],
            )

    def test_object_dtype_is_rejected(self):
        arrays = _synthetic_arrays()
        arrays["vertex_id"] = np.asarray(
            [object(), object()], dtype=object)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            with self.assertRaises(TypeError):
                write_stage3c_shard(
                    Path(temporary) / "bad.npz", arrays)


if __name__ == "__main__":
    unittest.main()
