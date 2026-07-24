import unittest

import numpy as np
import torch

from utils.trajectory_batch import (
    build_trajectory_batch,
    fragment_minimum_distance,
)
from utils.trajectory_fragments import TrajectoryFragment


BASE_TIME_NS = 1_600_000_000_000_000_000


def _fragment(track_index, start_index, points, time_offsets_ns=None):
    points = np.asarray(points, dtype=np.float32)
    if time_offsets_ns is None:
        time_offsets_ns = np.arange(points.shape[0], dtype=np.int64)
        time_offsets_ns *= 1_000_000_000
    timestamps = BASE_TIME_NS + np.asarray(
        time_offsets_ns, dtype=np.int64)
    return TrajectoryFragment(
        track_index=int(track_index),
        source_traj_id="track-{}".format(track_index),
        source_file="track-{}.csv".format(track_index),
        start_point_index=int(start_index),
        end_point_index=int(start_index + points.shape[0]),
        points_global_xy=points,
        points_relative_xy=points.copy(),
        timestamps_ns=timestamps,
    )


class TrajectoryBatchTests(unittest.TestCase):
    def test_variable_fragments_points_masks_coordinates_and_time(self):
        fragment_with_context = _fragment(
            7,
            10,
            [[100, 100], [105, 100], [120, 100]],
            [0, 1_500_000_000, 3_000_000_000],
        )
        segment_only = _fragment(
            8,
            20,
            [[80, 105], [120, 105]],
            [0, 250_000_000],
        )
        real_zero_coordinate = _fragment(
            9,
            0,
            [[0, 0]],
            [0],
        )

        batch = build_trajectory_batch(
            [
                [fragment_with_context, segment_only],
                [real_zero_coordinate],
            ],
            center_xy=[[100, 100], [0, 0]],
            window_size=20,
        )

        self.assertEqual(tuple(batch["traj_xy_rel"].shape), (2, 2, 3, 2))
        self.assertEqual(batch["traj_xy_rel"].dtype, torch.float32)
        self.assertEqual(batch["traj_time_delta"].dtype, torch.float32)
        self.assertEqual(batch["point_mask"].dtype, torch.bool)
        self.assertEqual(batch["fragment_mask"].dtype, torch.bool)
        self.assertEqual(batch["track_indices"].dtype, torch.int64)
        self.assertEqual(
            batch["fragment_mask"].tolist(),
            [[True, True], [True, False]],
        )
        self.assertEqual(
            batch["point_mask"].tolist(),
            [
                [[True, True, True], [True, True, False]],
                [[True, False, False], [False, False, False]],
            ],
        )
        self.assertEqual(
            batch["point_inside_mask"].tolist(),
            [
                [[True, True, False], [False, False, False]],
                [[True, False, False], [False, False, False]],
            ],
        )
        self.assertEqual(
            batch["segment_only"].tolist(),
            [[False, True], [False, False]],
        )

        torch.testing.assert_close(
            batch["traj_xy_rel"][0, 0, :3],
            torch.tensor([[0.0, 0.0], [5.0, 0.0], [20.0, 0.0]]),
        )
        torch.testing.assert_close(
            batch["traj_xy_norm"][0, 0, :3],
            torch.tensor([[0.0, 0.0], [0.5, 0.0], [2.0, 0.0]]),
        )
        torch.testing.assert_close(
            batch["traj_time_delta"][0, 0, :3],
            torch.tensor([0.0, 1.5, 3.0]),
        )
        torch.testing.assert_close(
            batch["traj_time_delta"][0, 1, :2],
            torch.tensor([0.0, 0.25]),
        )

        # A genuine local coordinate (0, 0) is data, not padding.
        self.assertTrue(bool(batch["point_mask"][0, 0, 0]))
        self.assertTrue(bool(batch["point_mask"][1, 0, 0]))
        self.assertEqual(batch["traj_xy_rel"][1, 0, 0].tolist(), [0.0, 0.0])
        self.assertEqual(batch["track_indices"].tolist(), [[7, 8], [9, -1]])
        self.assertEqual(
            batch["start_point_indices"].tolist(),
            [[10, 20], [0, -1]],
        )
        self.assertEqual(
            batch["end_point_indices"].tolist(),
            [[13, 22], [1, -1]],
        )

    def test_max_fragments_uses_geometry_and_deterministic_ties(self):
        fragments = [
            _fragment(9, 10, [[5, 0]]),
            _fragment(9, 2, [[-5, 0]]),
            _fragment(8, 7, [[1, 0]]),
            _fragment(10, 5, [[-20, 0], [20, 0]]),
        ]
        batch = build_trajectory_batch(
            [fragments],
            center_xy=[0, 0],
            window_size=10,
            max_fragments=3,
        )

        self.assertEqual(batch["total_fragment_count"].tolist(), [4])
        self.assertEqual(batch["kept_fragment_count"].tolist(), [3])
        self.assertEqual(batch["truncated_fragment_count"].tolist(), [1])
        self.assertEqual(batch["track_indices"].tolist(), [[10, 8, 9]])
        self.assertEqual(
            batch["start_point_indices"].tolist(),
            [[5, 7, 2]],
        )
        self.assertEqual(
            batch["source_fragment_indices"].tolist(),
            [[3, 2, 1]],
        )
        torch.testing.assert_close(
            batch["fragment_min_distance"][0],
            torch.tensor([0.0, 1.0, 5.0]),
        )

        repeated = build_trajectory_batch(
            [fragments],
            center_xy=[0, 0],
            window_size=10,
            max_fragments=3,
        )
        self.assertTrue(
            torch.equal(
                batch["source_fragment_indices"],
                repeated["source_fragment_indices"],
            )
        )

    def test_unbounded_mode_preserves_input_order_and_identity(self):
        fragments = [
            _fragment(22, 30, [[50, 0]]),
            _fragment(11, 4, [[1, 0], [2, 0]]),
        ]
        batch = build_trajectory_batch(
            [fragments],
            center_xy=[0, 0],
            window_size=[20, 40],
            max_fragments=None,
        )
        self.assertEqual(batch["track_indices"].tolist(), [[22, 11]])
        self.assertEqual(
            batch["fragment_support_count"].tolist(),
            [[1, 1]],
        )
        self.assertEqual(
            batch["source_fragment_indices"].tolist(),
            [[0, 1]],
        )
        self.assertEqual(batch["total_fragment_count"].tolist(), [2])
        self.assertEqual(batch["truncated_fragment_count"].tolist(), [0])
        self.assertEqual(
            batch["compression_total_count"].tolist(), [2])
        self.assertEqual(
            batch["compression_kept_count"].tolist(), [2])
        self.assertEqual(
            batch["compression_truncated_count"].tolist(), [0])
        torch.testing.assert_close(
            batch["traj_xy_norm"][0, 1, :2],
            torch.tensor([[0.1, 0.0], [0.2, 0.0]]),
        )

    def test_polyline_distance_includes_segment_interior(self):
        fragment = _fragment(3, 0, [[-20, 4], [20, 4]])
        self.assertAlmostEqual(
            fragment_minimum_distance(fragment, [0, 0]),
            4.0,
            places=6,
        )

    def test_empty_and_zero_budget_batches_are_explicit(self):
        unbounded = build_trajectory_batch(
            [[]],
            center_xy=[0, 0],
            window_size=20,
        )
        self.assertEqual(tuple(unbounded["point_mask"].shape), (1, 0, 0))
        self.assertEqual(unbounded["total_fragment_count"].tolist(), [0])

        fragment = _fragment(1, 0, [[0, 0]])
        zero_budget = build_trajectory_batch(
            [[fragment]],
            center_xy=[0, 0],
            window_size=20,
            max_fragments=0,
        )
        self.assertEqual(tuple(zero_budget["fragment_mask"].shape), (1, 0))
        self.assertEqual(zero_budget["total_fragment_count"].tolist(), [1])
        self.assertEqual(zero_budget["kept_fragment_count"].tolist(), [0])
        self.assertEqual(zero_budget["truncated_fragment_count"].tolist(), [1])


if __name__ == "__main__":
    unittest.main()
