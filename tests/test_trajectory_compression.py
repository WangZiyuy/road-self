import math
import unittest

import numpy as np
import torch

from utils.trajectory_batch import build_trajectory_batch
from utils.trajectory_compression import (
    GEOMETRY_DESCRIPTOR_NAMES,
    compress_trajectory_fragments,
    trajectory_fragment_geometry_descriptor,
)
from utils.trajectory_fragments import TrajectoryFragment


BASE_TIME_NS = 1_600_000_000_000_000_000


def _fragment(track_index, start_index, points):
    points = np.asarray(points, dtype=np.float32)
    timestamps = (
        BASE_TIME_NS
        + np.arange(points.shape[0], dtype=np.int64) * 1_000_000_000
    )
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


def _identities(result):
    return {
        (
            fragment.track_index,
            fragment.start_point_index,
            fragment.end_point_index,
        )
        for fragment in result.selected_fragments
    }


class TrajectoryCompressionTests(unittest.TestCase):
    def test_nearest_is_exactly_the_batch_builder_baseline(self):
        fragments = [
            _fragment(9, 10, [[5, 0]]),
            _fragment(9, 2, [[-5, 0]]),
            _fragment(8, 7, [[1, 0]]),
            _fragment(10, 5, [[-20, 0], [20, 0]]),
        ]
        result = compress_trajectory_fragments(
            fragments,
            center_xy=[0, 0],
            window_size=10,
            max_fragments=3,
            strategy="nearest",
        )
        legacy_batch = build_trajectory_batch(
            [fragments],
            center_xy=[0, 0],
            window_size=10,
            max_fragments=3,
        )
        result_batch = build_trajectory_batch(
            [result],
            center_xy=[0, 0],
            window_size=10,
        )

        np.testing.assert_array_equal(
            result.source_fragment_indices,
            legacy_batch["source_fragment_indices"][0].numpy(),
        )
        self.assertTrue(torch.equal(
            legacy_batch["track_indices"],
            result_batch["track_indices"],
        ))
        self.assertTrue(torch.equal(
            legacy_batch["start_point_indices"],
            result_batch["start_point_indices"],
        ))
        torch.testing.assert_close(
            legacy_batch["fragment_min_distance"],
            result_batch["fragment_min_distance"],
        )

    def test_near_diverse_is_bounded_deterministic_and_order_invariant(self):
        fragments = [
            _fragment(
                track_index,
                track_index * 10,
                [[-40, y], [40, y]],
            )
            for track_index, y in enumerate([-8, -4, 0, 4, 8])
        ]
        fragments += [
            _fragment(100, 0, [[20, -40], [20, 40]]),
            _fragment(101, 0, [[-30, -30], [30, 30]]),
        ]
        kwargs = dict(
            center_xy=[0, 0],
            window_size=100,
            max_fragments=4,
            strategy="near_diverse",
            near_fraction=0.25,
        )
        first = compress_trajectory_fragments(fragments, **kwargs)
        repeated = compress_trajectory_fragments(fragments, **kwargs)
        shuffled = [fragments[index] for index in [6, 2, 5, 0, 4, 1, 3]]
        shuffled_result = compress_trajectory_fragments(shuffled, **kwargs)

        self.assertEqual(first.kept_fragment_count, 4)
        self.assertLessEqual(first.kept_fragment_count, 4)
        np.testing.assert_array_equal(
            first.source_fragment_indices,
            repeated.source_fragment_indices,
        )
        self.assertEqual(_identities(first), _identities(shuffled_result))

    def test_near_diverse_ties_use_identity_not_input_position(self):
        fragments = [
            _fragment(11, 0, [[-40, 0], [40, 0]]),
            _fragment(10, 0, [[-40, 0], [40, 0]]),
            _fragment(12, 0, [[-40, 0], [40, 0]]),
        ]
        kwargs = dict(
            center_xy=[0, 0],
            window_size=100,
            max_fragments=1,
            strategy="near_diverse",
            near_fraction=1.0,
        )
        original = compress_trajectory_fragments(fragments, **kwargs)
        reversed_result = compress_trajectory_fragments(
            list(reversed(fragments)), **kwargs)
        self.assertEqual(_identities(original), _identities(reversed_result))
        self.assertEqual(
            original.selected_fragments[0].track_index, 10)

    def test_support_count_covers_every_source_candidate(self):
        fragments = [
            _fragment(index, 0, [[-30, float(index)], [30, float(index)]])
            for index in range(13)
        ]
        result = compress_trajectory_fragments(
            fragments,
            center_xy=[0, 0],
            window_size=100,
            max_fragments=4,
            strategy="near_diverse",
        )
        self.assertEqual(int(result.support_count.sum()), len(fragments))
        self.assertTrue(np.all(result.support_count >= 1))
        self.assertEqual(result.total_fragment_count, 13)
        self.assertEqual(result.truncated_fragment_count, 9)

    def test_duplicate_anomaly_geometry_is_not_selected_repeatedly(self):
        main_road = [
            _fragment(
                index,
                0,
                [[-45, float(offset)], [45, float(offset)]],
            )
            for index, offset in enumerate(range(-10, 11, 2))
        ]
        duplicate_anomalies = [
            _fragment(100 + index, 0, [[25, -40], [25, 40]])
            for index in range(10)
        ]
        result = compress_trajectory_fragments(
            main_road + duplicate_anomalies,
            center_xy=[0, 0],
            window_size=100,
            max_fragments=8,
            strategy="near_diverse",
            near_fraction=0.25,
        )
        selected_anomalies = sum(
            fragment.track_index >= 100
            for fragment in result.selected_fragments
        )
        self.assertLessEqual(selected_anomalies, 1)

    def test_rare_branch_survives_when_nearest_is_main_road_dominated(self):
        main_road = [
            _fragment(
                index,
                0,
                [[-45, float(offset)], [45, float(offset)]],
            )
            for index, offset in enumerate(np.linspace(0.0, 3.0, 20))
        ]
        rare_branch = _fragment(1000, 0, [[12, -45], [12, 45]])
        fragments = main_road + [rare_branch]
        nearest = compress_trajectory_fragments(
            fragments,
            center_xy=[0, 0],
            window_size=100,
            max_fragments=4,
            strategy="nearest",
        )
        diverse = compress_trajectory_fragments(
            fragments,
            center_xy=[0, 0],
            window_size=100,
            max_fragments=4,
            strategy="near_diverse",
            near_fraction=0.25,
        )
        self.assertNotIn(1000, {
            fragment.track_index
            for fragment in nearest.selected_fragments
        })
        self.assertIn(1000, {
            fragment.track_index
            for fragment in diverse.selected_fragments
        })

    def test_reverse_directions_have_the_same_road_axis_descriptor(self):
        forward = _fragment(1, 0, [[-40, -20], [40, 20]])
        reverse = _fragment(2, 0, [[40, 20], [-40, -20]])
        forward_descriptor = trajectory_fragment_geometry_descriptor(
            forward, [0, 0], 100)
        reverse_descriptor = trajectory_fragment_geometry_descriptor(
            reverse, [0, 0], 100)
        cos_index = GEOMETRY_DESCRIPTOR_NAMES.index("axis_cos_2theta")
        sin_index = GEOMETRY_DESCRIPTOR_NAMES.index("axis_sin_2theta")
        np.testing.assert_allclose(
            forward_descriptor[[cos_index, sin_index]],
            reverse_descriptor[[cos_index, sin_index]],
            atol=1e-7,
        )

    def test_angle_description_is_continuous_not_bucketed(self):
        zero_degrees = _fragment(1, 0, [[-40, 0], [40, 0]])
        one_degree = _fragment(
            2,
            0,
            [
                [-40, -40 * math.tan(math.radians(1))],
                [40, 40 * math.tan(math.radians(1))],
            ],
        )
        descriptor_zero = trajectory_fragment_geometry_descriptor(
            zero_degrees, [0, 0], 100)
        descriptor_one = trajectory_fragment_geometry_descriptor(
            one_degree, [0, 0], 100)
        cos_index = GEOMETRY_DESCRIPTOR_NAMES.index("axis_cos_2theta")
        sin_index = GEOMETRY_DESCRIPTOR_NAMES.index("axis_sin_2theta")
        self.assertNotEqual(
            float(descriptor_zero[sin_index]),
            float(descriptor_one[sin_index]),
        )
        self.assertAlmostEqual(
            float(descriptor_one[cos_index]),
            math.cos(math.radians(2)),
            places=6,
        )
        self.assertAlmostEqual(
            float(descriptor_one[sin_index]),
            math.sin(math.radians(2)),
            places=6,
        )

    def test_compression_result_is_not_selected_a_second_time_in_batch(self):
        fragments = [
            _fragment(index, 0, [[-40, float(index)], [40, float(index)]])
            for index in range(8)
        ]
        result = compress_trajectory_fragments(
            fragments,
            center_xy=[0, 0],
            window_size=100,
            max_fragments=4,
            strategy="near_diverse",
        )
        batch = build_trajectory_batch(
            [result],
            center_xy=[0, 0],
            window_size=100,
            max_fragments=1,
        )

        self.assertEqual(batch["fragment_mask"].sum().item(), 4)
        self.assertEqual(batch["compression_total_count"].tolist(), [8])
        self.assertEqual(batch["compression_kept_count"].tolist(), [4])
        self.assertEqual(batch["compression_truncated_count"].tolist(), [4])
        np.testing.assert_array_equal(
            batch["fragment_support_count"][0].numpy(),
            result.support_count,
        )
        np.testing.assert_array_equal(
            batch["source_fragment_indices"][0].numpy(),
            result.source_fragment_indices,
        )


if __name__ == "__main__":
    unittest.main()
