import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict
from torch.utils.data import Dataset

from model.branch_query_decoder import MultiModalBranchQueryDecoder
from model.trajectory_support_head import TrajectorySupportHead
from train_pretrajectory_support import (
    aggregate_pre_seed_metrics,
    collect_segment_only_examples,
    evaluate_acceptance,
)
from utils.stage3d_checkpoint import (
    build_stage3d_support_checkpoint_payload,
    load_stage3d_support_checkpoint,
    save_stage3d_support_checkpoint,
)
from utils.trajectory_support_metrics import support_label_diagnostics
from utils.trajectory_support_ranking import (
    TrajectorySupportRankingAccumulator,
)
from utils.trajectory_support_features import (
    build_pre_trajectory_branch_tokens,
)


class _SingleSegmentOnlyDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        del index
        return {
            "trajectory_batch": {
                "traj_xy_norm": torch.tensor(
                    [[[-1.0, 0.0], [1.0, 0.0]]]),
                "point_mask": torch.tensor([[True, True]]),
                "fragment_mask": torch.tensor([True]),
                "segment_only": torch.tensor([True]),
                "track_indices": torch.tensor([17]),
            },
            "branch_targets": {
                "branch_offsets_norm": torch.tensor([[1.0, 0.0]]),
                "branch_mask": torch.tensor([True]),
            },
            "metadata": {
                "dataset_index": torch.tensor(9),
            },
        }


def _config():
    return EasyDict({
        "TRAIN": {
            "WINDOW_SIZE": 256,
            "STEP_LENGTH": 20,
        },
        "STAGE3D": {
            "SUPPORT_TARGET": {
                "DISTANCE_SIGMA_PIXELS": 12.0,
                "AXIS_GAMMA": 2.0,
                "POSITIVE_THRESHOLD": 0.1,
                "EPSILON": 1e-6,
            },
            "TRAINING": {
                "CACHE_BATCH_SIZE": 1,
            },
        },
        "STAGE3D_B0": {
            "ACCEPTANCE": {
                "MIN_PRE_SUPPORT_AP": 0.72,
                "MIN_RAW_ATTENTION_AP_GAIN": 0.20,
                "MIN_MULTIBRANCH_RAW_ATTENTION_AP_GAIN": 0.15,
                "MAX_SUPPORT_AP_STD": 0.02,
                "MAX_PREDICTED_TOP8_JACCARD_MEDIAN": 0.5,
                "MAX_INVALID_TOP1_MEAN_PROBABILITY": 0.5,
                "MAX_INVALID_TOP1_P90_PROBABILITY": 0.75,
            },
        },
    })


class PreTrajectorySupportInputTests(unittest.TestCase):
    def test_pre_branch_is_exact_graph_image_concatenation(self):
        graph = torch.randn(2, 6, 128)
        image = torch.randn(2, 6, 128)
        result = build_pre_trajectory_branch_tokens(graph, image)
        self.assertEqual(tuple(result.shape), (2, 6, 256))
        torch.testing.assert_close(result[..., :128], graph)
        torch.testing.assert_close(result[..., 128:], image)

    def test_pre_states_do_not_depend_on_trajectory_fragments(self):
        torch.manual_seed(4)
        decoder = MultiModalBranchQueryDecoder(
            hidden_dim=32,
            trajectory_dim=32,
            image_channels=32,
            num_queries=3,
            num_heads=4,
            image_pool_size=4,
            dropout=0.0,
            query_self_attention_layers=1,
        ).eval()
        image = torch.randn(1, 32, 8, 8)
        state = torch.randn(1, 32)
        walked = torch.randn(1, 1, 8, 8)
        mask = torch.ones(1, 4, dtype=torch.bool)
        first = decoder(
            image,
            state,
            torch.randn(1, 4, 32),
            mask,
            walked,
            return_debug_states=True,
        )
        second = decoder(
            image,
            state,
            torch.randn(1, 4, 32) * 10.0,
            mask,
            walked,
            return_debug_states=True,
        )
        for key in (
                "debug_graph_conditioned_queries",
                "debug_image_cross_attention_output"):
            torch.testing.assert_close(first[key], second[key])

    def test_support_head_accepts_256_dim_branch_and_old_strict_state(self):
        old = TrajectorySupportHead(hidden_dim=16, projection_dim=8)
        compatible = TrajectorySupportHead(hidden_dim=16, projection_dim=8)
        compatible.load_state_dict(old.state_dict(), strict=True)
        pre = TrajectorySupportHead(
            hidden_dim=16,
            branch_input_dim=32,
            fragment_input_dim=16,
            projection_dim=8,
        )
        logits = pre(
            torch.randn(2, 3, 32),
            torch.randn(2, 5, 16),
            torch.ones(2, 5, dtype=torch.bool),
        )
        self.assertEqual(tuple(logits.shape), (2, 3, 5))


class SupportRankingMetricTests(unittest.TestCase):
    def _result(self):
        accumulator = TrajectorySupportRankingAccumulator(
            ranking_ks=(1, 4), jaccard_k=2)
        scores = torch.tensor([[
            [0.9, 0.8, 0.2, 0.1],
            [0.1, 0.2, 0.9, 0.8],
            [0.6, 0.4, 0.3, 0.2],
        ]])
        targets = torch.tensor([[
            [1.0, 0.7, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.6],
            [0.05, 0.0, 0.0, 0.0],
        ]])
        positive = targets >= 0.1
        accumulator.update(
            scores=scores,
            support_targets=targets,
            support_positive_mask=positive,
            support_valid=torch.tensor([[True, True, False]]),
            branch_mask=torch.tensor([[True, True, True]]),
            fragment_mask=torch.tensor([[True, True, True, True]]),
            matches=[(
                torch.tensor([0, 1, 2]),
                torch.tensor([0, 1, 2]),
            )],
            branch_count=torch.tensor([3]),
            sample_ids=torch.tensor([12]),
        )
        return accumulator.compute()

    def test_extended_ranking_metrics_and_invalid_diagnostics(self):
        result = self._result()
        self.assertAlmostEqual(result["support_ap"], 1.0)
        self.assertAlmostEqual(result["precision_at"]["1"], 1.0)
        self.assertAlmostEqual(result["hit_at"]["1"], 1.0)
        self.assertAlmostEqual(
            result["soft_support_mass_recall_at"]["4"], 1.0)
        self.assertAlmostEqual(result["ndcg_at"]["4"], 1.0)
        self.assertEqual(
            result["by_gt_branch_count"][">=3"]["branch_count"], 2)
        self.assertEqual(result["support_invalid"]["branch_count"], 1)
        self.assertAlmostEqual(
            result["support_invalid"]["max_probability"], 0.6)
        self.assertEqual(
            result["predicted_top_k_jaccard"]["pair_count"], 1)
        self.assertEqual(
            result["gt_positive_set_jaccard"]["pair_count"], 1)

    def test_branch_hit_rate_has_correct_non_recall_name(self):
        report = support_label_diagnostics(
            support_positive_mask=[
                np.asarray([[[True], [False]]])],
            support_valid=[
                np.asarray([[True, False]])],
            branch_mask=[
                np.asarray([[True, True]])],
            segment_only_positive_mask=[
                np.asarray([[[False], [False]]])],
        )
        self.assertEqual(
            report["bounded_64_branch_support_hit_rate"], 0.5)
        self.assertEqual(
            report["bounded_64_oracle_support_recall"], 0.5)

    def test_segment_only_component_examples_are_reported(self):
        report = collect_segment_only_examples(
            dataset=_SingleSegmentOnlyDataset(),
            cfg=_config(),
            max_examples=2,
        )
        self.assertEqual(report["segment_only_fragment_count"], 1)
        self.assertEqual(report["example_count"], 1)
        example = report["examples"][0]
        self.assertIn("distance_score", example)
        self.assertIn("axis_score", example)
        self.assertIn("coverage_score", example)
        self.assertIn("support_target", example)
        self.assertTrue(report["thresholds_unchanged"])


class Stage3DB0LifecycleTests(unittest.TestCase):
    def test_checkpoint_marks_pretrajectory_non_circular_contract(self):
        head = TrajectorySupportHead(
            hidden_dim=8,
            branch_input_dim=16,
            fragment_input_dim=8,
            projection_dim=4,
        )
        optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
        payload = build_stage3d_support_checkpoint_payload(
            support_head=head,
            optimizer=optimizer,
            epoch=3,
            e4_checkpoint="e4.pth.tar",
            e4_checkpoint_sha256="abc",
            config_snapshot={},
            stage="3D-B0",
            metadata={
                "reads_trajectory_context": False,
                "feeds_path_push": False,
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pre.pth.tar"
            save_stage3d_support_checkpoint(path, payload)
            restored = TrajectorySupportHead(
                hidden_dim=8,
                branch_input_dim=16,
                fragment_input_dim=8,
                projection_dim=4,
            )
            loaded = load_stage3d_support_checkpoint(
                path, support_head=restored)
        self.assertEqual(loaded["stage"], "3D-B0")
        self.assertFalse(
            loaded["metadata"]["reads_trajectory_context"])
        self.assertFalse(loaded["metadata"]["feeds_path_push"])

    def test_three_seed_aggregation_and_acceptance_gate(self):
        def metrics(ap, multi, jaccard, invalid_mean, invalid_p90):
            return {
                "support_ap": ap,
                "by_gt_branch_count": {
                    ">=2": {"support_ap": multi}},
                "precision_at": {"8": 0.7},
                "hit_at": {"8": 0.95},
                "predicted_top_k_jaccard": {
                    "mean": jaccard, "median": jaccard},
                "support_invalid": {
                    "max_probability": 0.6,
                    "top1_probability": {
                        "mean": invalid_mean,
                        "p90": invalid_p90,
                    },
                },
            }
        reports = [
            {"best_validation": metrics(
                value, 0.76, 0.2, 0.3, 0.6)}
            for value in (0.75, 0.76, 0.74)
        ]
        aggregate = aggregate_pre_seed_metrics(reports)
        decision = evaluate_acceptance(
            aggregate=aggregate,
            raw_attention={
                "support_ap": 0.4,
                "by_gt_branch_count": {
                    ">=2": {"support_ap": 0.5}},
            },
            cfg=_config(),
        )
        self.assertTrue(decision["passed"])
        self.assertTrue(
            decision["recommend_support_guided_aggregation"])


if __name__ == "__main__":
    unittest.main()
