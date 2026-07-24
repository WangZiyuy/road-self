import unittest

import torch
import torch.nn.functional as F

from model.branch_set_loss import BranchSetCriterion


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


if __name__ == "__main__":
    unittest.main()
