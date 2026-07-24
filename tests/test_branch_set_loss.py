import unittest

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from model.branch_set_loss import (
    BranchSetCriterion,
    hungarian_match_branches,
)


def _predictions(batch_size=1, query_count=4):
    offsets = torch.tensor([
        [[0.8, 0.1], [-0.5, 0.4], [0.1, -0.9], [0.2, 0.2]]
    ], dtype=torch.float32).repeat(batch_size, 1, 1)
    offsets = offsets[:, :query_count].clone().requires_grad_(True)
    logits = torch.zeros(
        batch_size, query_count, requires_grad=True)
    return {
        "branch_exist_logits": logits,
        "branch_offsets_norm": offsets,
        "branch_directions": F.normalize(offsets, dim=-1),
    }


def _targets(offsets, mask):
    offsets = torch.as_tensor(offsets, dtype=torch.float32)
    directions = F.normalize(offsets, dim=-1)
    return {
        "branch_offsets_norm": offsets,
        "branch_directions": directions,
        "branch_mask": torch.as_tensor(mask, dtype=torch.bool),
    }


class BranchSetCriterionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260724)
        self.criterion = BranchSetCriterion()

    def test_gt_order_does_not_change_loss(self):
        predictions = _predictions()
        targets = _targets(
            [[[0.75, 0.1], [-0.45, 0.35], [0.0, 0.0]]],
            [[True, True, False]],
        )
        permutation = torch.tensor([1, 0, 2])
        permuted_targets = {
            key: value.index_select(1, permutation)
            for key, value in targets.items()
        }
        original = self.criterion(predictions, targets)
        permuted = self.criterion(predictions, permuted_targets)
        for key in (
                "loss",
                "existence_loss",
                "endpoint_loss",
                "direction_loss"):
            torch.testing.assert_close(
                original[key], permuted[key], rtol=1e-6, atol=1e-6)

    def test_batched_cost_transfer_preserves_per_sample_matching(self):
        torch.manual_seed(17)
        offsets = torch.randn(4, 6, 2)
        predictions = {
            "branch_exist_logits": torch.randn(4, 6),
            "branch_offsets_norm": offsets,
            "branch_directions": F.normalize(offsets, dim=-1),
        }
        target_offsets = torch.randn(4, 5, 2)
        targets = {
            "branch_offsets_norm": target_offsets,
            "branch_directions": F.normalize(
                target_offsets, dim=-1),
            "branch_mask": torch.tensor([
                [True, True, False, False, False],
                [True, False, True, True, False],
                [False, False, False, False, False],
                [True, True, True, True, True],
            ]),
        }
        actual = hungarian_match_branches(predictions, targets)
        for batch_index, (actual_rows, actual_targets) in enumerate(
                actual):
            valid_targets = torch.nonzero(
                targets["branch_mask"][batch_index],
                as_tuple=False,
            ).flatten()
            if valid_targets.numel() == 0:
                self.assertEqual(actual_rows.numel(), 0)
                self.assertEqual(actual_targets.numel(), 0)
                continue
            endpoint_cost = torch.cdist(
                predictions["branch_offsets_norm"][batch_index],
                targets["branch_offsets_norm"][
                    batch_index].index_select(0, valid_targets),
                p=1,
            )
            direction_cost = 1.0 - torch.matmul(
                predictions["branch_directions"][batch_index],
                targets["branch_directions"][
                    batch_index].index_select(
                        0, valid_targets).transpose(0, 1),
            )
            expected_rows, expected_columns = linear_sum_assignment(
                (endpoint_cost + direction_cost).numpy())
            expected_targets = valid_targets.index_select(
                0, torch.as_tensor(expected_columns))
            torch.testing.assert_close(
                actual_rows.cpu(),
                torch.as_tensor(expected_rows),
                rtol=0.0,
                atol=0.0,
            )
            torch.testing.assert_close(
                actual_targets.cpu(),
                expected_targets,
                rtol=0.0,
                atol=0.0,
            )

    def test_zero_one_and_multiple_branches_backward(self):
        predictions = _predictions(batch_size=3)
        targets = _targets(
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.7, 0.1], [0.0, 0.0]],
                [[0.7, 0.1], [-0.4, 0.3]],
            ],
            [
                [False, False],
                [True, False],
                [True, True],
            ],
        )
        losses = self.criterion(predictions, targets)
        self.assertEqual(int(losses["matched_count"]), 3)
        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()
        self.assertTrue(torch.isfinite(
            predictions["branch_exist_logits"].grad).all())
        self.assertTrue(torch.isfinite(
            predictions["branch_offsets_norm"].grad).all())

    def test_unmatched_queries_are_no_branch(self):
        predictions = _predictions(query_count=4)
        targets = _targets([[[0.8, 0.1]]], [[True]])
        losses = self.criterion(predictions, targets)
        existence_targets = losses["existence_targets"]
        self.assertEqual(float(existence_targets.sum()), 1.0)
        self.assertEqual(
            int((existence_targets == 0.0).sum()), 3)

    def test_no_gt_has_only_existence_supervision(self):
        predictions = _predictions()
        targets = {
            "branch_offsets_norm": torch.zeros(1, 0, 2),
            "branch_directions": torch.zeros(1, 0, 2),
            "branch_mask": torch.zeros(1, 0, dtype=torch.bool),
        }
        losses = self.criterion(predictions, targets)
        self.assertEqual(
            float(losses["endpoint_loss"].detach()), 0.0)
        self.assertEqual(
            float(losses["direction_loss"].detach()), 0.0)
        self.assertEqual(
            float(losses["existence_targets"].sum().detach()), 0.0)
        losses["loss"].backward()
        self.assertTrue(torch.isfinite(
            predictions["branch_exist_logits"].grad).all())
        self.assertTrue(torch.isfinite(
            predictions["branch_offsets_norm"].grad).all())

    def test_existence_cost_breaks_equal_geometry_tie(self):
        offsets = torch.tensor(
            [[[0.5, 0.0], [0.5, 0.0]]], dtype=torch.float32)
        predictions = {
            "branch_exist_logits": torch.tensor([[-4.0, 4.0]]),
            "branch_offsets_norm": offsets,
            "branch_directions": F.normalize(offsets, dim=-1),
        }
        targets = _targets([[[0.5, 0.0]]], [[True]])
        matches = hungarian_match_branches(
            predictions,
            targets,
            existence_cost_weight=1.0,
        )
        self.assertEqual(matches[0][0].tolist(), [1])

    def test_zero_existence_cost_exactly_reproduces_default_matcher(self):
        predictions = _predictions()
        targets = _targets(
            [[[0.7, 0.1], [-0.4, 0.3]]],
            [[True, True]],
        )
        default = hungarian_match_branches(predictions, targets)
        explicit = hungarian_match_branches(
            predictions, targets, existence_cost_weight=0.0)
        for default_pair, explicit_pair in zip(default, explicit):
            for default_indices, explicit_indices in zip(
                    default_pair, explicit_pair):
                torch.testing.assert_close(
                    default_indices,
                    explicit_indices,
                    rtol=0.0,
                    atol=0.0,
                )

    def test_no_object_coef_one_reproduces_unweighted_bce(self):
        predictions = _predictions()
        targets = _targets([[[0.8, 0.1]]], [[True]])
        losses = BranchSetCriterion(
            exist_no_object_coef=1.0)(predictions, targets)
        expected = F.binary_cross_entropy_with_logits(
            predictions["branch_exist_logits"],
            losses["existence_targets"],
        )
        torch.testing.assert_close(
            losses["existence_loss"], expected, rtol=0.0, atol=0.0)

    def test_no_object_coef_only_downweights_negative_slots(self):
        predictions = _predictions()
        predictions["branch_exist_logits"].data.copy_(
            torch.tensor([[0.2, -0.4, 0.7, -0.1]]))
        targets = _targets([[[0.8, 0.1]]], [[True]])
        losses = BranchSetCriterion(
            exist_no_object_coef=0.2)(predictions, targets)
        positive = losses["existence_targets"] > 0.5
        self.assertTrue(torch.equal(
            losses["existence_weights"][positive],
            torch.ones_like(
                losses["existence_weights"][positive]),
        ))
        torch.testing.assert_close(
            losses["existence_weights"][~positive],
            torch.full_like(
                losses["existence_weights"][~positive], 0.2),
            rtol=0.0,
            atol=0.0,
        )
        losses["existence_loss"].backward()
        gradient = predictions["branch_exist_logits"].grad
        self.assertTrue(bool((gradient[positive] < 0.0).all()))
        self.assertTrue(bool((gradient[~positive] > 0.0).all()))


if __name__ == "__main__":
    unittest.main()
