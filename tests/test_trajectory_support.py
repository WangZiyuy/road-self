import math
import unittest
import uuid
from pathlib import Path

import numpy as np
import torch

from model.trajectory_support_head import (
    TrajectorySupportHead,
    trajectory_support_bce_loss,
)
from train_branch_aux import (
    MODALITY_GRAPH_ONLY,
    _forward_auxiliary,
)
from train_trajectory_support import _reducible_loss_reduction
from utils.stage3d_checkpoint import (
    build_stage3d_support_checkpoint_payload,
    load_stage3d_support_checkpoint,
    save_stage3d_support_checkpoint,
)
from utils.trajectory_support_metrics import (
    TrajectorySupportMetricAccumulator,
    support_label_diagnostics,
)
from utils.trajectory_support_targets import (
    build_trajectory_support_targets,
)


def _trajectory_batch():
    # 100 px window: one normalized unit is 50 px.
    xy = torch.zeros(1, 5, 3, 2)
    xy[0, 0] = torch.tensor([
        [0.0, 0.0], [0.2, 0.0], [0.4, 0.0]])
    xy[0, 1] = torch.tensor([
        [0.4, 0.0], [0.2, 0.0], [0.0, 0.0]])
    xy[0, 2] = torch.tensor([
        [0.0, 0.4], [0.2, 0.4], [0.4, 0.4]])
    xy[0, 3] = torch.tensor([
        [0.0, 0.0], [0.0, 0.2], [0.0, 0.4]])
    xy[0, 4, 0] = torch.tensor([0.0, 0.0])
    point_mask = torch.tensor([[
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, False, False],
    ]])
    return {
        "traj_xy_norm": xy,
        "point_mask": point_mask,
        "fragment_mask": torch.ones(1, 5, dtype=torch.bool),
        "segment_only": torch.tensor(
            [[False, True, False, False, False]]),
    }


def _branch_targets():
    return {
        "branch_offsets_norm": torch.tensor([[
            [0.4, 0.0],
            [0.0, 0.4],
        ]]),
        "branch_mask": torch.tensor([[True, True]]),
    }


class TrajectorySupportTargetTests(unittest.TestCase):
    def test_distance_axis_coverage_and_reverse_direction(self):
        result = build_trajectory_support_targets(
            _trajectory_batch(),
            _branch_targets(),
            window_size=100,
            step_length=20,
            distance_sigma_pixels=5,
            axis_gamma=1,
            positive_threshold=0.5,
        )
        support = result["support_targets"][0]
        self.assertAlmostEqual(float(support[0, 0]), 1.0, places=6)
        self.assertAlmostEqual(float(support[0, 1]), 1.0, places=6)
        self.assertLess(float(support[0, 2]), 0.001)
        self.assertAlmostEqual(float(support[0, 3]), 0.0, places=6)
        self.assertAlmostEqual(float(support[0, 4]), 0.0, places=6)
        self.assertTrue(result["support_valid"].tolist() == [[True, True]])
        self.assertTrue(
            bool(result["segment_only_positive_mask"][0, 0, 1]))

    def test_crossing_segment_has_zero_distance_but_wrong_axis(self):
        result = build_trajectory_support_targets(
            _trajectory_batch(),
            _branch_targets(),
            window_size=100,
            step_length=20,
            distance_sigma_pixels=5,
            axis_gamma=1,
            positive_threshold=0.1,
        )
        self.assertAlmostEqual(
            float(result["minimum_distance_pixels"][0, 0, 3]),
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(result["axis_score"][0, 0, 3]), 0.0, places=6)

    def test_no_effective_fragment_marks_branch_invalid(self):
        trajectory = _trajectory_batch()
        trajectory["fragment_mask"].zero_()
        result = build_trajectory_support_targets(
            trajectory,
            _branch_targets(),
            window_size=100,
            step_length=20,
            distance_sigma_pixels=5,
            axis_gamma=2,
            positive_threshold=0.1,
        )
        self.assertFalse(bool(result["support_valid"].any()))
        self.assertEqual(float(result["support_targets"].sum()), 0.0)

    def test_zero_threshold_does_not_turn_padding_into_support(self):
        trajectory = _trajectory_batch()
        trajectory["fragment_mask"].zero_()
        result = build_trajectory_support_targets(
            trajectory,
            _branch_targets(),
            window_size=100,
            step_length=20,
            distance_sigma_pixels=5,
            axis_gamma=1,
            positive_threshold=0.0,
        )
        self.assertFalse(bool(result["support_positive_mask"].any()))
        self.assertFalse(bool(result["support_valid"].any()))


class TrajectorySupportHeadTests(unittest.TestCase):
    def test_shape_padding_mask_and_finite_gradients(self):
        torch.manual_seed(7)
        head = TrajectorySupportHead(
            hidden_dim=16, projection_dim=8)
        branch = torch.randn(2, 3, 16, requires_grad=True)
        fragment = torch.randn(2, 4, 16, requires_grad=True)
        mask = torch.tensor([
            [True, True, False, False],
            [True, True, True, False],
        ])
        logits = head(branch, fragment, mask)
        self.assertEqual(tuple(logits.shape), (2, 3, 4))
        self.assertEqual(float(logits[:, :, 3].abs().sum()), 0.0)
        logits.square().mean().backward()
        self.assertTrue(torch.isfinite(branch.grad).all())
        self.assertTrue(torch.isfinite(fragment.grad).all())

    def test_loss_uses_only_matched_valid_branches_and_fragments(self):
        logits = torch.tensor([[
            [0.0, 0.0, 100.0],
            [2.0, -2.0, -100.0],
            [100.0, 100.0, 100.0],
        ]], requires_grad=True)
        targets = torch.tensor([[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]])
        valid = torch.tensor([[True, False]])
        fragment_mask = torch.tensor([[True, True, False]])
        matches = [(
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
        )]
        result = trajectory_support_bce_loss(
            logits, targets, valid, fragment_mask, matches)
        expected = math.log(2.0)
        self.assertAlmostEqual(
            float(result["loss"].detach()), expected, places=6)
        self.assertEqual(
            int(result["supervised_branch_count"]), 1)
        self.assertEqual(int(result["supervised_pair_count"]), 2)
        result["loss"].backward()
        self.assertEqual(float(logits.grad[0, 2].abs().sum()), 0.0)
        self.assertEqual(float(logits.grad[0, 1].abs().sum()), 0.0)

    def test_empty_supervision_is_differentiable(self):
        logits = torch.randn(1, 2, 3, requires_grad=True)
        result = trajectory_support_bce_loss(
            logits,
            torch.zeros(1, 1, 3),
            torch.zeros(1, 1, dtype=torch.bool),
            torch.ones(1, 3, dtype=torch.bool),
            [(torch.tensor([0]), torch.tensor([0]))],
        )
        self.assertEqual(float(result["loss"].detach()), 0.0)
        result["loss"].backward()
        self.assertTrue(torch.equal(
            logits.grad, torch.zeros_like(logits)))

    def test_checkpoint_round_trip_is_strict(self):
        torch.manual_seed(11)
        first = TrajectorySupportHead(hidden_dim=8, projection_dim=4)
        optimizer = torch.optim.Adam(first.parameters(), lr=1e-3)
        branch = torch.randn(1, 2, 8)
        fragment = torch.randn(1, 3, 8)
        mask = torch.tensor([[True, True, False]])
        first_output = first(branch, fragment, mask)
        first_output.sum().backward()
        optimizer.step()
        payload = build_stage3d_support_checkpoint_payload(
            support_head=first,
            optimizer=optimizer,
            epoch=7,
            e4_checkpoint="/tmp/e4.pth.tar",
            e4_checkpoint_sha256="abc",
            config_snapshot={"stage": "3D-A"},
            metrics={"support_ap": 0.9},
        )
        path = Path("tests") / (
            "_stage3d_support_{}.pth.tar".format(uuid.uuid4().hex))
        try:
            save_stage3d_support_checkpoint(path, payload)
            second = TrajectorySupportHead(
                hidden_dim=8, projection_dim=4)
            second_optimizer = torch.optim.Adam(
                second.parameters(), lr=1e-3)
            restored = load_stage3d_support_checkpoint(
                path,
                support_head=second,
                optimizer=second_optimizer,
            )
        finally:
            path.unlink(missing_ok=True)
        self.assertEqual(restored["epoch"], 7)
        self.assertEqual(restored["stage"], "3D-A")
        torch.testing.assert_close(
            first(branch, fragment, mask),
            second(branch, fragment, mask),
            rtol=0.0,
            atol=0.0,
        )


class TrajectorySupportMetricTests(unittest.TestCase):
    def test_soft_label_sanity_uses_reducible_loss_not_zero(self):
        # A soft-label BCE floor of 0.4 makes a raw 50% reduction from 0.7
        # impossible. Reaching 0.43 nevertheless removes 90% of reducible
        # loss and is a valid overfit signal.
        self.assertAlmostEqual(
            _reducible_loss_reduction(0.7, 0.43, 0.4),
            0.9,
            places=6,
        )

    def test_support_metrics_compare_head_and_attention(self):
        accumulator = TrajectorySupportMetricAccumulator(
            recall_ks=(1, 2), jaccard_k=2)
        logits = torch.tensor([[
            [5.0, 3.0, -3.0, -5.0],
            [-5.0, -3.0, 3.0, 5.0],
        ]])
        attention = torch.full((1, 2, 4), 0.25)
        targets = torch.tensor([[
            [1.0, 0.8, 0.0, 0.0],
            [0.0, 0.0, 0.8, 1.0],
        ]])
        positive = targets >= 0.5
        accumulator.update(
            support_logits=logits,
            attention_weights=attention,
            support_targets=targets,
            support_positive_mask=positive,
            support_valid=torch.tensor([[True, True]]),
            fragment_mask=torch.ones(1, 4, dtype=torch.bool),
            segment_only=torch.tensor(
                [[False, True, False, False]]),
            matches=[(
                torch.tensor([0, 1]),
                torch.tensor([0, 1]),
            )],
            sample_ids=torch.tensor([4]),
        )
        result = accumulator.compute()
        self.assertAlmostEqual(result["support_ap"], 1.0, places=6)
        self.assertGreater(
            result["support_ap"], result["attention_support_ap"])
        self.assertAlmostEqual(result["recall_at"]["2"], 1.0, places=6)
        self.assertAlmostEqual(
            result["top_k_fragment_jaccard"]["mean"], 0.0, places=6)
        self.assertAlmostEqual(
            result["segment_only_positive_ratio"], 0.25, places=6)

    def test_label_diagnostics_reports_multibranch_availability(self):
        report = support_label_diagnostics(
            support_positive_mask=[np.array([[
                [True, False], [False, False], [False, False]]])],
            support_valid=[np.array([[True, False, False]])],
            branch_mask=[np.array([[True, True, False]])],
            segment_only_positive_mask=[np.array([[
                [False, False], [False, False], [False, False]]])],
        )
        self.assertAlmostEqual(report["support_available_rate"], 0.5)
        self.assertAlmostEqual(
            report["by_gt_branch_count"]["2"][
                "support_available_rate"],
            0.5,
        )


class GraphOnlyAblationTests(unittest.TestCase):
    class _Trajectory(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen_mask = None

        def forward(self, batch):
            self.seen_mask = batch["fragment_mask"].clone()
            shape = tuple(batch["fragment_mask"].shape) + (4,)
            return {
                "fragment_tokens": torch.ones(shape),
                "fragment_mask": batch["fragment_mask"],
            }

    class _Graph(torch.nn.Module):
        def forward(self, graph_state):
            return graph_state["state_token"]

    class _Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.arguments = None

        def forward(self, **kwargs):
            self.arguments = kwargs
            batch_size = kwargs["stage_fuse"].shape[0]
            return {
                "branch_exist_logits": torch.zeros(batch_size, 2),
                "branch_offsets_norm": torch.zeros(batch_size, 2, 2),
                "branch_directions": torch.zeros(batch_size, 2, 2),
            }

    def test_graph_only_masks_image_walked_path_and_trajectory(self):
        trajectory = self._Trajectory()
        graph = self._Graph()
        decoder = self._Decoder()
        batch = {
            "trajectory_batch": {
                "fragment_mask": torch.tensor([[True, True]]),
            },
            "graph_state": {
                "state_token": torch.tensor([[2.0, 3.0, 4.0, 5.0]]),
            },
            "walked_path": torch.ones(1, 1, 4, 4),
        }
        _forward_auxiliary(
            modules=(trajectory, graph, decoder),
            batch=batch,
            stage_fuse=torch.ones(1, 4, 4, 4),
            modality=MODALITY_GRAPH_ONLY,
        )
        self.assertFalse(bool(trajectory.seen_mask.any()))
        self.assertFalse(bool(decoder.arguments["fragment_mask"].any()))
        self.assertFalse(bool(decoder.arguments["image_available"].any()))
        self.assertEqual(
            float(decoder.arguments["walked_path"].sum()), 0.0)
        torch.testing.assert_close(
            decoder.arguments["state_token"],
            batch["graph_state"]["state_token"],
        )


if __name__ == "__main__":
    unittest.main()
