import hashlib
import json
import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np

from scripts.prepare_structured_trajectory_cache import (
    build_structured_trajectory_cache,
)
from utils.gis_to_graph import GisToGraphConverter
from utils.structured_trajectory_store import (
    REQUIRED_CACHE_FILES,
    open_structured_trajectory_store,
)
from utils.trajectory_fragments import SEGMENT_GRID_INDEX_BASIS


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file)


def _write_track(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        file.write("data_time,lat,lon\n")
        for timestamp, latitude, longitude in rows:
            file.write("{},{},{}\n".format(
                timestamp, latitude, longitude))


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class StructuredTrajectoryStoreTest(unittest.TestCase):
    def setUp(self):
        # Keep test artifacts inside the repository workspace. This also lets
        # the suite run in restricted environments where the OS temp directory
        # is readable but not writable.
        self.root = Path.cwd() / ".stage1a-test-{}".format(uuid.uuid4().hex)
        self.root.mkdir()
        self.data_root = self.root / "data_self"
        self.piece_dir = self.data_root / "input" / "traj_piece" / "synthetic"
        self.metadata_path = (
            self.data_root
            / "input"
            / "regions"
            / "synthetic_metadata.json"
        )
        self.output_dir = (
            self.data_root
            / "input"
            / "traj_structured"
            / "synthetic"
            / "v1"
        )
        self.metadata = {
            "region": "synthetic",
            "original_size": [1000, 500],
            "bbox_gcj02": {
                "lat_min": 0.0,
                "lon_min": 100.0,
                "lat_max": 10.0,
                "lon_max": 110.0,
            },
        }
        _write_json(self.metadata_path, self.metadata)

        # The index intentionally orders track_b before track_a. The builder
        # must preserve this identity/order instead of sorting file names.
        self.track_b = [
            ("2020-01-01T00:00:02", 0.0, 100.0),
            ("2020-01-01T00:00:04", 5.0, 105.0),
            ("2020-01-01T00:00:03", 10.0, 110.0),
        ]
        self.track_a = [
            ("2020-02-03T04:05:06.000000007", 5.0, 105.0),
            ("2020-02-03T04:05:07.000000008", 6.0, 106.0),
        ]
        _write_track(self.piece_dir / "track_a.csv", self.track_a)
        _write_track(self.piece_dir / "track_b.csv", self.track_b)
        records = [
            {
                "file": "track_b.csv",
                "source_traj_id": "vehicle-B/session-2",
                "point_count": 3,
                "time_start": self.track_b[0][0],
                "time_end": self.track_b[-1][0],
            },
            {
                "file": "track_a.csv",
                "source_traj_id": "vehicle-A/session-1",
                "point_count": 2,
                "time_start": self.track_a[0][0],
                "time_end": self.track_a[-1][0],
            },
        ]
        with (self.piece_dir / "trajectory_index.jsonl").open(
            "w", encoding="utf-8"
        ) as file:
            for record in records:
                file.write(json.dumps(record))
                file.write("\n")

    def tearDown(self):
        shutil.rmtree(str(self.root))

    def _build(self, output_dir=None, overwrite=False):
        return build_structured_trajectory_cache(
            region="synthetic",
            piece_dir=self.piece_dir,
            metadata_path=self.metadata_path,
            output_dir=output_dir or self.output_dir,
            cell_size=100,
            overwrite=overwrite,
        )

    def test_build_offsets_identity_order_time_coordinates_and_grid(self):
        meta = self._build()
        store = open_structured_trajectory_store(str(self.output_dir))
        validation = store.validate()

        self.assertTrue(validation["passed"])
        self.assertEqual(
            validation["grid_index_basis"], SEGMENT_GRID_INDEX_BASIS)
        self.assertEqual(meta["trajectory_count"], 2)
        self.assertEqual(meta["point_count"], 5)
        np.testing.assert_array_equal(
            np.asarray(store.track_offsets),
            np.asarray([0, 3, 5], dtype=np.int64),
        )
        self.assertIsInstance(store.points_xy, np.memmap)
        self.assertIsInstance(store.timestamps_ns, np.memmap)
        self.assertIsInstance(store.track_offsets, np.memmap)

        first = store.get_track(0)
        second = store.get_track(1)
        self.assertEqual(first.source_traj_id, "vehicle-B/session-2")
        self.assertEqual(first.source_file, "track_b.csv")
        self.assertEqual(second.source_traj_id, "vehicle-A/session-1")
        self.assertEqual(first.points_xy.shape, (3, 2))
        self.assertEqual(second.points_xy.shape, (2, 2))

        expected_time_b = np.asarray(
            [
                np.datetime64(row[0], "ns").astype(np.int64)
                for row in self.track_b
            ],
            dtype=np.int64,
        )
        expected_time_a = np.asarray(
            [
                np.datetime64(row[0], "ns").astype(np.int64)
                for row in self.track_a
            ],
            dtype=np.int64,
        )
        # The deliberately non-monotonic B timestamps prove that CSV row
        # order is preserved rather than re-sorted by time.
        np.testing.assert_array_equal(first.timestamps_ns, expected_time_b)
        np.testing.assert_array_equal(second.timestamps_ns, expected_time_a)

        converter = GisToGraphConverter(
            "synthetic",
            [[row[1], row[2]] for row in self.track_b],
            data_root=str(self.data_root),
        )
        expected_points = np.asarray(
            converter.convert_trajectories_to_pixels(),
            dtype=np.float32,
        )
        np.testing.assert_array_equal(first.points_xy, expected_points)

        candidate_ids = store.candidate_track_ids_for_rect(
            495.0, 245.0, 505.0, 255.0)
        np.testing.assert_array_equal(
            candidate_ids, np.asarray([0, 1], dtype=np.int32))
        empty_ids = store.candidate_track_ids_for_rect(
            2000.0, 2000.0, 2010.0, 2010.0)
        self.assertEqual(empty_ids.dtype, np.dtype(np.int32))
        self.assertEqual(empty_ids.shape, (0,))

        for name in REQUIRED_CACHE_FILES:
            self.assertTrue((self.output_dir / name).is_file())
        with np.load(
            self.output_dir / "grid_index.npz",
            allow_pickle=False,
        ) as grid:
            self.assertEqual(grid["cells"].dtype, np.dtype(np.int32))
            self.assertEqual(
                grid["cell_offsets"].dtype, np.dtype(np.int64))
            self.assertEqual(grid["track_ids"].dtype, np.dtype(np.int32))

    def test_reopen_and_repeated_build_are_content_stable(self):
        self._build()
        second_output = self.output_dir.parent / "v1_repeat"
        self._build(output_dir=second_output)

        first_store = open_structured_trajectory_store(str(self.output_dir))
        second_store = open_structured_trajectory_store(str(second_output))
        np.testing.assert_array_equal(
            first_store.points_xy, second_store.points_xy)
        np.testing.assert_array_equal(
            first_store.timestamps_ns, second_store.timestamps_ns)
        np.testing.assert_array_equal(
            first_store.track_offsets, second_store.track_offsets)
        self.assertEqual(first_store.track_records, second_store.track_records)
        self.assertEqual(first_store.meta, second_store.meta)

        # JSON, NPY, and the deterministic NPZ writer are all byte stable.
        for name in REQUIRED_CACHE_FILES:
            self.assertEqual(
                _sha256(self.output_dir / name),
                _sha256(second_output / name),
                name,
            )

    def test_nonempty_output_requires_explicit_overwrite(self):
        self._build()
        marker = self.output_dir / "user_marker.txt"
        marker.write_text("do not silently replace", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "--overwrite"):
            self._build()
        self.assertTrue(marker.is_file())

        self._build(overwrite=True)
        self.assertFalse(marker.exists())
        reopened = open_structured_trajectory_store(str(self.output_dir))
        self.assertTrue(reopened.validate()["passed"])


if __name__ == "__main__":
    unittest.main()
