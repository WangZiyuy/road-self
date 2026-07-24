import math
import unittest
from unittest import mock

import numpy as np
import torch

import utils.trajectory_compression as compression_module
from utils.trajectory_batch import build_trajectory_batch
from utils.trajectory_compression import (
    FAST_GEOMETRY_DESCRIPTOR_NAMES,
    GEOMETRY_DESCRIPTOR_NAMES,
    compress_trajectory_fragments,
    trajectory_fragment_fast_descriptor,
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

    def test_bounded_result_count_does_not_exceed_budget(self):
        fragments = [
            _fragment(index, 0, [[-40, index], [40, index]])
            for index in range(20)
        ]
        result = compress_trajectory_fragments(
            fragments,
            center_xy=[0, 0],
            window_size=100,
            max_fragments=7,
            strategy="bounded_near_diverse",
        )
        self.assertEqual(result.kept_fragment_count, 7)
        self.assertLessEqual(len(result.selected_fragments), 7)
        self.assertEqual(result.near_fraction, 0.5)

    def test_bounded_is_repeatable_and_input_order_invariant(self):
        fragments = [
            _fragment(
                index,
                index * 3,
                [[-40, float(index - 5)], [40, float(index - 5)]],
            )
            for index in range(10)
        ]
        fragments.extend([
            _fragment(100, 0, [[12, -40], [12, 40]]),
            _fragment(101, 0, [[-30, -30], [30, 30]]),
        ])
        kwargs = dict(
            center_xy=[0, 0],
            window_size=100,
            max_fragments=6,
            strategy="bounded_near_diverse",
            prepool_multiplier=2,
            near_fraction=0.5,
        )
        first = compress_trajectory_fragments(fragments, **kwargs)
        repeated = compress_trajectory_fragments(fragments, **kwargs)
        order = [11, 5, 2, 8, 0, 10, 1, 7, 3, 9, 4, 6]
        shuffled = compress_trajectory_fragments(
            [fragments[index] for index in order], **kwargs)
        np.testing.assert_array_equal(
            first.source_fragment_indices,
            repeated.source_fragment_indices,
        )
        self.assertEqual(_identities(first), _identities(shuffled))

    def test_legacy_default_near_fraction_remains_unchanged(self):
        fragments = [
            _fragment(index, 0, [[-30, index], [30, index]])
            for index in range(8)
        ]
        result = compress_trajectory_fragments(
            fragments,
            center_xy=[0, 0],
            window_size=100,
            max_fragments=4,
            strategy="near_diverse",
        )
        self.assertEqual(result.near_fraction, 0.25)

    def test_bounded_always_keeps_nearest_fraction(self):
        fragments = [
            _fragment(index, 0, [[float(index), 0]])
            for index in range(1, 13)
        ]
        result = compress_trajectory_fragments(
            fragments,
            center_xy=[0, 0],
            window_size=100,
            max_fragments=6,
            strategy="bounded_near_diverse",
            prepool_multiplier=2,
            near_fraction=0.5,
        )
        selected_tracks = {
            fragment.track_index
            for fragment in result.selected_fragments
        }
        self.assertTrue({1, 2, 3}.issubset(selected_tracks))

    def test_bounded_descriptors_are_only_built_for_prepool(self):
        fragments = [
            _fragment(index, 0, [[-30, index], [30, index]])
            for index in range(100)
        ]
        evaluation_counts = []
        original_builder = (
            compression_module
            .build_fast_trajectory_geometry_descriptors
        )

        def counted_builder(shortlist, center_xy, window_size):
            evaluation_counts.append(len(shortlist))
            return original_builder(shortlist, center_xy, window_size)

        with mock.patch.object(
            compression_module,
            "build_fast_trajectory_geometry_descriptors",
            side_effect=counted_builder,
        ):
            result = compress_trajectory_fragments(
                fragments,
                center_xy=[0, 0],
                window_size=100,
                max_fragments=5,
                strategy="bounded_near_diverse",
                prepool_multiplier=3,
            )
        self.assertEqual(evaluation_counts, [15])
        self.assertEqual(result.prepool_count, 15)
        self.assertEqual(result.descriptor_evaluation_count, 15)
        self.assertLessEqual(
            result.descriptor_evaluation_count,
            min(len(fragments), 3 * 5),
        )

    def test_bounded_does_not_call_full_support_assignment(self):
        fragments = [
            _fragment(index, 0, [[-30, index], [30, index]])
            for index in range(20)
        ]
        with mock.patch.object(
            compression_module,
            "_support_counts",
            side_effect=AssertionError(
                "bounded strategy must not assign full support"),
        ):
            result = compress_trajectory_fragments(
                fragments,
                center_xy=[0, 0],
                window_size=100,
                max_fragments=6,
                strategy="bounded_near_diverse",
            )
        self.assertIsNone(result.support_count)
        self.assertFalse(result.support_count_valid)
        self.assertEqual(
            result.compression_timing_ms["support_assignment"], 0.0)

    def test_invalid_axis_only_enters_near_or_distance_fallback(self):
        fragments = [
            _fragment(0, 0, [[0, 0]]),
            _fragment(1, 0, [[2, -40], [2, 40]]),
            _fragment(2, 0, [[-40, 3], [40, 3]]),
            _fragment(3, 0, [[-40, -4], [40, 4]]),
            _fragment(100, 0, [[20, 20]]),
            _fragment(101, 0, [[-25, 25], [-25, 25]]),
            _fragment(102, 0, [[30, -30]]),
        ]
        result = compress_trajectory_fragments(
            fragments,
            center_xy=[0, 0],
            window_size=100,
            max_fragments=4,
            strategy="bounded_near_diverse",
            prepool_multiplier=2,
            near_fraction=0.25,
        )
        selected_tracks = {
            fragment.track_index
            for fragment in result.selected_fragments
        }
        self.assertEqual(selected_tracks, {0, 1, 2, 3})

    def test_bounded_retains_sparse_branch_from_dense_trunk(self):
        trunk = [
            _fragment(
                index,
                0,
                [[-45, float(offset)], [45, float(offset)]],
            )
            for index, offset in enumerate(np.linspace(0.0, 3.0, 20))
        ]
        branch = _fragment(1000, 0, [[12, -45], [12, 45]])
        result = compress_trajectory_fragments(
            trunk + [branch],
            center_xy=[0, 0],
            window_size=100,
            max_fragments=4,
            strategy="bounded_near_diverse",
            prepool_multiplier=8,
            near_fraction=0.5,
        )
        self.assertIn(1000, {
            fragment.track_index
            for fragment in result.selected_fragments
        })

    def test_fast_descriptor_maps_reverse_travel_to_same_axis(self):
        forward = _fragment(1, 0, [[-40, -20], [40, 20]])
        reverse = _fragment(2, 0, [[40, 20], [-40, -20]])
        forward_descriptor = trajectory_fragment_fast_descriptor(
            forward, [0, 0], 100)
        reverse_descriptor = trajectory_fragment_fast_descriptor(
            reverse, [0, 0], 100)
        self.assertEqual(
            forward_descriptor.shape,
            (len(FAST_GEOMETRY_DESCRIPTOR_NAMES),),
        )
        np.testing.assert_allclose(
            forward_descriptor[2:4],
            reverse_descriptor[2:4],
            atol=1e-7,
        )

    def test_bounded_result_enters_batch_with_invalid_support_mask(self):
        fragments = [
            _fragment(index, 0, [[-40, index], [40, index]])
            for index in range(10)
        ]
        result = compress_trajectory_fragments(
            fragments,
            center_xy=[0, 0],
            window_size=100,
            max_fragments=4,
            strategy="bounded_near_diverse",
        )
        batch = build_trajectory_batch(
            [result],
            center_xy=[0, 0],
            window_size=100,
        )
        self.assertEqual(
            batch["fragment_support_count"].tolist(),
            [[1, 1, 1, 1]],
        )
        self.assertEqual(
            batch["fragment_support_count_valid"].tolist(),
            [[False, False, False, False]],
        )
        self.assertEqual(
            batch["compression_total_count"].tolist(), [10])
        self.assertEqual(
            batch["compression_kept_count"].tolist(), [4])


if __name__ == "__main__":
    unittest.main()
