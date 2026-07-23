import unittest

import numpy as np
import torch

from model.trajectory_encoder import TrajectoryFragmentEncoder
from utils.trajectory_batch import build_trajectory_batch
from utils.trajectory_fragments import TrajectoryFragment


BASE_TIME_NS = 1_600_000_000_000_000_000
SEED = 20260724


def _fragment(
    track_index,
    start_index,
    points,
    time_offsets_seconds=None,
):
    points = np.asarray(points, dtype=np.float32)
    if time_offsets_seconds is None:
        time_offsets_seconds = np.arange(
            points.shape[0], dtype=np.float64)
    timestamps = BASE_TIME_NS + np.rint(
        np.asarray(time_offsets_seconds, dtype=np.float64)
        * 1_000_000_000.0
    ).astype(np.int64)
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


class TrajectoryFragmentEncoderTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(SEED)
        self.hidden_dim = 16
        self.encoder = TrajectoryFragmentEncoder(
            hidden_dim=self.hidden_dim,
            num_heads=4,
            num_layers=2,
            dropout=0.1,
        ).eval()
        self.fragment_a = _fragment(
            7,
            10,
            [[0, 0], [2, 1], [5, 3]],
            [0.0, 1.0, 2.5],
        )
        self.fragment_b = _fragment(
            8,
            20,
            [[-4, 1], [-1, 2], [2, 4], [4, 7], [8, 9]],
            [0.0, 0.5, 1.5, 3.0, 5.0],
        )

    def _encode(self, fragment_lists, centers):
        trajectory_batch = build_trajectory_batch(
            fragment_lists,
            center_xy=centers,
            window_size=20,
        )
        with torch.no_grad():
            output = self.encoder(trajectory_batch)
        return trajectory_batch, output

    def test_shapes_masks_padding_zero_and_finite(self):
        trajectory_batch, output = self._encode(
            [
                [self.fragment_a, self.fragment_b],
                [self.fragment_a],
            ],
            [[0, 0], [0, 0]],
        )
        self.assertEqual(
            tuple(output["point_tokens"].shape),
            (2, 2, 5, self.hidden_dim),
        )
        self.assertEqual(
            tuple(output["fragment_tokens"].shape),
            (2, 2, self.hidden_dim),
        )
        self.assertIs(
            output["point_mask"], trajectory_batch["point_mask"])
        self.assertIs(
            output["fragment_mask"],
            trajectory_batch["fragment_mask"],
        )
        padding_points = ~trajectory_batch["point_mask"]
        self.assertTrue(
            torch.equal(
                output["point_tokens"][padding_points],
                torch.zeros_like(
                    output["point_tokens"][padding_points]),
            )
        )
        padding_fragments = ~trajectory_batch["fragment_mask"]
        self.assertTrue(
            torch.equal(
                output["fragment_tokens"][padding_fragments],
                torch.zeros_like(
                    output["fragment_tokens"][padding_fragments]),
            )
        )
        self.assertTrue(torch.isfinite(output["point_tokens"]).all())
        self.assertTrue(torch.isfinite(output["fragment_tokens"]).all())

    def test_extra_padding_length_does_not_change_valid_encoding(self):
        short_batch, short_output = self._encode(
            [[self.fragment_a]],
            [[0, 0]],
        )
        long_fragment = _fragment(
            9,
            0,
            [[index, index % 3] for index in range(9)],
        )
        padded_batch, padded_output = self._encode(
            [[self.fragment_a, long_fragment]],
            [[0, 0]],
        )
        self.assertEqual(short_batch["traj_xy_norm"].shape[2], 3)
        self.assertEqual(padded_batch["traj_xy_norm"].shape[2], 9)
        torch.testing.assert_close(
            short_output["point_tokens"][0, 0, :3],
            padded_output["point_tokens"][0, 0, :3],
            atol=1e-6,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            short_output["fragment_tokens"][0, 0],
            padded_output["fragment_tokens"][0, 0],
            atol=1e-6,
            rtol=1e-5,
        )

    def test_fragment_permutation_only_permutes_outputs(self):
        batch_ab, output_ab = self._encode(
            [[self.fragment_a, self.fragment_b]],
            [[0, 0]],
        )
        batch_ba, output_ba = self._encode(
            [[self.fragment_b, self.fragment_a]],
            [[0, 0]],
        )
        self.assertEqual(
            batch_ab["track_indices"].tolist(), [[7, 8]])
        self.assertEqual(
            batch_ba["track_indices"].tolist(), [[8, 7]])
        torch.testing.assert_close(
            output_ab["fragment_tokens"][0, 0],
            output_ba["fragment_tokens"][0, 1],
        )
        torch.testing.assert_close(
            output_ab["fragment_tokens"][0, 1],
            output_ba["fragment_tokens"][0, 0],
        )
        torch.testing.assert_close(
            output_ab["point_tokens"][0, 0, :3],
            output_ba["point_tokens"][0, 1, :3],
        )
        torch.testing.assert_close(
            output_ab["point_tokens"][0, 1, :5],
            output_ba["point_tokens"][0, 0, :5],
        )

    def test_point_order_and_direction_change_encoding(self):
        points = [[0, 0], [1, 3], [4, 4], [9, 6]]
        original = _fragment(
            11, 0, points, [0.0, 1.0, 3.0, 6.0])
        reordered = _fragment(
            11,
            0,
            [points[0], points[2], points[1], points[3]],
            [0.0, 1.0, 3.0, 6.0],
        )
        reversed_fragment = _fragment(
            11,
            0,
            list(reversed(points)),
            [0.0, 1.0, 3.0, 6.0],
        )
        _, original_output = self._encode([[original]], [[0, 0]])
        _, reordered_output = self._encode([[reordered]], [[0, 0]])
        _, reversed_output = self._encode(
            [[reversed_fragment]], [[0, 0]])

        self.assertFalse(
            torch.allclose(
                original_output["fragment_tokens"],
                reordered_output["fragment_tokens"],
                atol=1e-5,
                rtol=1e-5,
            )
        )
        self.assertFalse(
            torch.allclose(
                original_output["fragment_tokens"],
                reversed_output["fragment_tokens"],
                atol=1e-5,
                rtol=1e-5,
            )
        )

    def test_real_zero_coordinate_is_a_valid_point(self):
        trajectory_batch, output = self._encode(
            [[self.fragment_a]],
            [[0, 0]],
        )
        self.assertEqual(
            trajectory_batch["traj_xy_norm"][0, 0, 0].tolist(),
            [0.0, 0.0],
        )
        self.assertTrue(bool(trajectory_batch["point_mask"][0, 0, 0]))
        self.assertGreater(
            float(output["point_tokens"][0, 0, 0].norm()),
            0.0,
        )

    def test_empty_sample_is_zero_finite_and_batch_independent(self):
        empty_batch, empty_output = self._encode(
            [[]],
            [[0, 0]],
        )
        self.assertEqual(
            tuple(empty_output["point_tokens"].shape),
            (1, 0, 0, self.hidden_dim),
        )
        self.assertEqual(
            tuple(empty_output["fragment_tokens"].shape),
            (1, 0, self.hidden_dim),
        )
        self.assertTrue(torch.isfinite(empty_output["point_tokens"]).all())
        self.assertTrue(
            torch.isfinite(empty_output["fragment_tokens"]).all())

        _, mixed_output = self._encode(
            [[], [self.fragment_a]],
            [[0, 0], [0, 0]],
        )
        _, occupied_output = self._encode(
            [[self.fragment_b], [self.fragment_a]],
            [[0, 0], [0, 0]],
        )
        self.assertEqual(
            float(mixed_output["point_tokens"][0].abs().sum()),
            0.0,
        )
        self.assertEqual(
            float(mixed_output["fragment_tokens"][0].abs().sum()),
            0.0,
        )
        torch.testing.assert_close(
            mixed_output["fragment_tokens"][1, 0],
            occupied_output["fragment_tokens"][1, 0],
            atol=1e-6,
            rtol=1e-5,
        )

    def test_track_index_has_no_learned_semantic_effect(self):
        trajectory_batch = build_trajectory_batch(
            [[self.fragment_a]],
            center_xy=[[0, 0]],
            window_size=20,
        )
        changed_identity_batch = dict(trajectory_batch)
        changed_identity_batch["track_indices"] = torch.full_like(
            trajectory_batch["track_indices"], 123456)
        with torch.no_grad():
            original = self.encoder(trajectory_batch)
            changed = self.encoder(changed_identity_batch)
        torch.testing.assert_close(
            original["point_tokens"], changed["point_tokens"])
        torch.testing.assert_close(
            original["fragment_tokens"], changed["fragment_tokens"])


if __name__ == "__main__":
    unittest.main()
