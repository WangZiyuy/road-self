import json
import shutil
import unittest
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np

from utils.structured_trajectory_store import (
    SCHEMA_VERSION,
    open_structured_trajectory_store,
)
from utils.trajectory_fragments import (
    SEGMENT_GRID_INDEX_BASIS,
    trajectory_grid_cells,
)


BASE_TIME_NS = 1_600_000_000_000_000_000


def _array_meta(array):
    return {
        "shape": list(array.shape),
        "dtype": np.dtype(array.dtype).name,
    }


def _time_text(timestamp_ns):
    return np.datetime_as_string(
        np.datetime64(int(timestamp_ns), "ns"), unit="ns")


def _write_fragment_test_cache(cache_dir):
    tracks = [
        # One visit with one available context point at both ends.
        np.asarray(
            [[20, 50], [35, 50], [45, 50], [55, 50], [65, 50], [80, 50]],
            dtype=np.float32,
        ),
        # Both sampled points are outside, but the segment crosses the window.
        np.asarray([[20, 52], [80, 52]], dtype=np.float32),
        # The same track enters, exits, travels outside, then enters again.
        np.asarray(
            [
                [20, 48],
                [50, 48],
                [80, 48],
                [80, 80],
                [20, 80],
                [20, 48],
                [50, 48],
                [80, 48],
            ],
            dtype=np.float32,
        ),
        # This apparent crossing is invalid when the 100-second gap is capped.
        np.asarray([[20, 46], [80, 46]], dtype=np.float32),
        # This apparent crossing is invalid when the 60-pixel jump is capped.
        np.asarray([[20, 54], [80, 54]], dtype=np.float32),
    ]
    timestamps = [
        # Deliberately non-monotonic at indices 1->2; query must not reorder.
        np.asarray(
            [
                BASE_TIME_NS,
                BASE_TIME_NS + 3_000_000_000,
                BASE_TIME_NS + 2_000_000_000,
                BASE_TIME_NS + 4_000_000_000,
                BASE_TIME_NS + 5_000_000_000,
                BASE_TIME_NS + 6_000_000_000,
            ],
            dtype=np.int64,
        ),
        np.asarray(
            [BASE_TIME_NS, BASE_TIME_NS + 1_000_000_000],
            dtype=np.int64,
        ),
        np.asarray(
            [BASE_TIME_NS + index * 1_000_000_000 for index in range(8)],
            dtype=np.int64,
        ),
        np.asarray(
            [BASE_TIME_NS, BASE_TIME_NS + 100_000_000_000],
            dtype=np.int64,
        ),
        np.asarray(
            [BASE_TIME_NS, BASE_TIME_NS + 1_000_000_000],
            dtype=np.int64,
        ),
    ]

    offsets = np.zeros((len(tracks) + 1,), dtype=np.int64)
    for index, points in enumerate(tracks):
        offsets[index + 1] = offsets[index] + len(points)
    points_xy = np.concatenate(tracks, axis=0)
    timestamps_ns = np.concatenate(timestamps, axis=0)
    np.save(cache_dir / "points_xy.npy", points_xy, allow_pickle=False)
    np.save(
        cache_dir / "timestamps_ns.npy",
        timestamps_ns,
        allow_pickle=False,
    )
    np.save(cache_dir / "track_offsets.npy", offsets, allow_pickle=False)

    cell_size = 16
    cell_to_tracks = defaultdict(list)
    for track_index, points in enumerate(tracks):
        cells = trajectory_grid_cells(
            points, cell_size, include_segments=True)
        for cell_x, cell_y in cells:
            cell_to_tracks[(int(cell_x), int(cell_y))].append(track_index)
    sorted_cells = sorted(cell_to_tracks)
    cells = np.asarray(sorted_cells, dtype=np.int32).reshape((-1, 2))
    cell_offsets = np.zeros((len(cells) + 1,), dtype=np.int64)
    flattened_track_ids = []
    for cell_index, cell in enumerate(sorted_cells):
        flattened_track_ids.extend(sorted(set(cell_to_tracks[cell])))
        cell_offsets[cell_index + 1] = len(flattened_track_ids)
    track_ids = np.asarray(flattened_track_ids, dtype=np.int32)
    np.savez(
        cache_dir / "grid_index.npz",
        cells=cells,
        cell_offsets=cell_offsets,
        track_ids=track_ids,
    )

    records = []
    for track_index, track_timestamps in enumerate(timestamps):
        records.append(
            {
                "track_index": track_index,
                "source_traj_id": "source-{}".format(track_index),
                "source_file": "track_{}.csv".format(track_index),
                "point_count": len(tracks[track_index]),
                "time_start": _time_text(track_timestamps[0]),
                "time_end": _time_text(track_timestamps[-1]),
            }
        )
    with (cache_dir / "track_index.jsonl").open(
        "w", encoding="utf-8"
    ) as file:
        for record in records:
            file.write(json.dumps(record))
            file.write("\n")

    meta = {
        "schema_version": SCHEMA_VERSION,
        "region": "fragment-test",
        "trajectory_count": len(tracks),
        "point_count": len(points_xy),
        "image_size": [100, 100],
        "geographic_bbox": {
            "lat_min": 0,
            "lon_min": 0,
            "lat_max": 1,
            "lon_max": 1,
        },
        "cell_size": cell_size,
        "grid_cell_count": len(cells),
        "grid_membership_count": len(track_ids),
        "grid_index_basis": SEGMENT_GRID_INDEX_BASIS,
        "arrays": {
            "points_xy": _array_meta(points_xy),
            "timestamps_ns": _array_meta(timestamps_ns),
            "track_offsets": _array_meta(offsets),
            "grid_cells": _array_meta(cells),
            "grid_cell_offsets": _array_meta(cell_offsets),
            "grid_track_ids": _array_meta(track_ids),
        },
    }
    with (cache_dir / "meta.json").open("w", encoding="utf-8") as file:
        json.dump(meta, file)
    return tracks, timestamps


class TrajectoryFragmentQueryTest(unittest.TestCase):
    def setUp(self):
        self.root = Path.cwd() / ".stage1b-test-{}".format(uuid.uuid4().hex)
        self.root.mkdir()
        self.tracks, self.timestamps = _write_fragment_test_cache(self.root)
        self.store = open_structured_trajectory_store(str(self.root))
        self.assertTrue(self.store.validate()["passed"])

    def tearDown(self):
        del self.store
        shutil.rmtree(str(self.root))

    def _query(self, **kwargs):
        options = {
            "center_xy": (50, 50),
            "window_size": 20,
            "context_points": 0,
        }
        options.update(kwargs)
        return self.store.query_trajectory_fragments(**options)

    def _track_fragments(self, track_index, **kwargs):
        return [
            fragment
            for fragment in self._query(**kwargs)
            if fragment.track_index == track_index
        ]

    def test_point_visit_preserves_order_timestamps_and_relative_coordinates(self):
        fragments = self._track_fragments(0)
        self.assertEqual(len(fragments), 1)
        fragment = fragments[0]
        self.assertEqual(fragment.source_traj_id, "source-0")
        self.assertEqual(fragment.start_point_index, 1)
        self.assertEqual(fragment.end_point_index, 5)
        np.testing.assert_array_equal(
            fragment.points_global_xy, self.tracks[0][1:5])
        np.testing.assert_array_equal(
            fragment.timestamps_ns, self.timestamps[0][1:5])
        np.testing.assert_array_equal(
            fragment.points_relative_xy,
            self.tracks[0][1:5] - np.asarray([50, 50], dtype=np.float32),
        )

    def test_segment_crossing_window_is_recalled_with_both_endpoints_outside(self):
        direct_candidates = self.store.candidate_track_ids_for_rect(
            40, 40, 60, 60)
        self.assertIn(1, direct_candidates.tolist())
        fragments = self._track_fragments(1)
        self.assertEqual(len(fragments), 1)
        np.testing.assert_array_equal(
            fragments[0].points_global_xy, self.tracks[1])

    def test_same_track_entering_twice_returns_two_ordered_fragments(self):
        fragments = self._track_fragments(2)
        self.assertEqual(len(fragments), 2)
        self.assertEqual(
            [
                (fragment.start_point_index, fragment.end_point_index)
                for fragment in fragments
            ],
            [(0, 3), (5, 8)],
        )

    def test_context_points_are_added_without_changing_track_identity(self):
        fragment = self._track_fragments(0, context_points=1)[0]
        self.assertEqual(fragment.start_point_index, 0)
        self.assertEqual(fragment.end_point_index, 6)
        np.testing.assert_array_equal(
            fragment.points_global_xy, self.tracks[0])

    def test_time_gap_splits_an_apparent_crossing(self):
        self.assertEqual(len(self._track_fragments(3)), 1)
        self.assertEqual(
            self._track_fragments(3, max_time_gap_seconds=10), [])

    def test_spatial_gap_splits_an_apparent_crossing(self):
        self.assertEqual(len(self._track_fragments(4)), 1)
        self.assertEqual(
            self._track_fragments(4, max_spatial_gap_pixels=20), [])

    def test_empty_window_returns_an_empty_list(self):
        fragments = self.store.query_trajectory_fragments(
            center_xy=(500, 500),
            window_size=20,
        )
        self.assertEqual(fragments, [])

    def test_result_order_is_deterministic(self):
        first = self._query()
        second = self._query()
        first_signature = [
            (
                fragment.track_index,
                fragment.start_point_index,
                fragment.end_point_index,
            )
            for fragment in first
        ]
        second_signature = [
            (
                fragment.track_index,
                fragment.start_point_index,
                fragment.end_point_index,
            )
            for fragment in second
        ]
        self.assertEqual(first_signature, second_signature)
        self.assertEqual(first_signature, sorted(first_signature))


if __name__ == "__main__":
    unittest.main()
