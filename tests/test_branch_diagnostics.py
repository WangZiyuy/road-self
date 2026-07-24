import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.branch_diagnostics import (
    binary_average_precision,
    binary_auroc,
    branch_precision_recall_curve,
    calibration_curve,
    duplicate_statistics,
    oracle_k_metrics,
    query_pairwise_statistics,
)
from utils.stage3c_config import load_stage3c_config


class BranchDiagnosticsTest(unittest.TestCase):
    def test_binary_ap_auroc_and_calibration(self):
        scores = np.asarray([0.9, 0.8, 0.2, 0.1])
        labels = np.asarray([1, 0, 1, 0], dtype=bool)
        self.assertAlmostEqual(
            binary_average_precision(scores, labels),
            (1.0 + 2.0 / 3.0) / 2.0,
        )
        self.assertAlmostEqual(binary_auroc(scores, labels), 0.75)
        calibration = calibration_curve(
            scores, labels, bin_count=5)
        self.assertEqual(int(calibration["count"].sum()), 4)
        self.assertTrue(np.isfinite(calibration["ece"]))

    def test_branch_ap_reuses_geometry_true_positive_thresholds(self):
        scores = np.asarray([[0.9, 0.7], [0.8, 0.1]])
        pred_offsets = np.asarray([
            [[0.5, 0.0], [-0.8, 0.0]],
            [[0.5, 0.0], [-0.5, 0.0]],
        ])
        pred_directions = pred_offsets.copy()
        target_offsets = np.asarray([
            [[0.5, 0.0]],
            [[0.0, 0.0]],
        ])
        target_directions = np.asarray([
            [[1.0, 0.0]],
            [[0.0, 0.0]],
        ])
        target_mask = np.asarray([[True], [False]])
        result = branch_precision_recall_curve(
            scores,
            pred_offsets,
            pred_directions,
            target_offsets,
            target_directions,
            target_mask,
            window_size=256,
            endpoint_threshold_pixels=20,
            direction_threshold_degrees=45,
        )
        # The only GT is the highest-scoring, geometrically valid branch.
        self.assertEqual(result["average_precision"], 1.0)
        self.assertEqual(int(result["true_positive"].sum()), 1)

    def test_oracle_k_exposes_real_top_k_duplicates(self):
        scores = np.asarray([[0.9, 0.8, 0.1]])
        pred_offsets = np.asarray(
            [[[0.5, 0.0], [0.51, 0.0], [0.0, 0.5]]])
        target_offsets = np.asarray(
            [[[0.5, 0.0], [0.0, 0.5]]])
        result = oracle_k_metrics(
            scores,
            pred_offsets,
            pred_offsets,
            target_offsets,
            target_offsets,
            np.asarray([[True, True]]),
            window_size=256,
            endpoint_threshold_pixels=20,
            direction_threshold_degrees=45,
            duplicate_endpoint_threshold_pixels=12,
            duplicate_direction_threshold_degrees=25,
        )
        self.assertEqual(
            result["duplicates"]["duplicate_pair_ratio"], 1.0)
        self.assertLess(result["recall"], 1.0)

    def test_duplicate_and_query_similarity_statistics(self):
        offsets = np.asarray([[
            [0.5, 0.0], [0.51, 0.0], [0.0, 0.5],
        ]])
        duplicate = duplicate_statistics(
            offsets,
            offsets,
            [np.asarray([0, 1])],
            window_size=256,
            endpoint_threshold_pixels=12,
            direction_threshold_degrees=25,
        )
        self.assertEqual(duplicate["duplicate_pair_ratio"], 1.0)
        hidden = np.asarray([[
            [1.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        ]])
        similarity = query_pairwise_statistics(hidden)
        self.assertEqual(
            similarity["pairwise_cosine"]["count"], 3)

    def test_stage3c_config_inheritance_is_deep_and_deterministic(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            (root / "base.yml").write_text(
                "TRAJ:\n  MODE: none\n"
                "STAGE3C:\n  MODEL:\n    HIDDEN_DIM: 128\n"
                "    QUERY_SELF_ATTENTION_LAYERS: 0\n",
                encoding="utf-8",
            )
            (root / "child.yml").write_text(
                "BASE_CONFIG: base.yml\n"
                "STAGE3C:\n  MODEL:\n"
                "    QUERY_SELF_ATTENTION_LAYERS: 1\n",
                encoding="utf-8",
            )
            cfg = load_stage3c_config(root / "child.yml")
            self.assertEqual(cfg.STAGE3C.MODEL.HIDDEN_DIM, 128)
            self.assertEqual(
                cfg.STAGE3C.MODEL.QUERY_SELF_ATTENTION_LAYERS, 1)

    def test_base_config_preserves_legacy_training_behavior(self):
        cfg = load_stage3c_config(
            Path("configs/stage3c_branch_aux.yml"))
        self.assertEqual(
            cfg.STAGE3C.MATCHING.EXISTENCE_COST_WEIGHT, 0.0)
        self.assertEqual(
            cfg.STAGE3C.LOSS.EXIST_NO_OBJECT_COEF, 1.0)
        self.assertEqual(
            cfg.STAGE3C.MODEL.QUERY_SELF_ATTENTION_LAYERS, 0)
        self.assertEqual(
            cfg.STAGE3C.EVALUATION.MODEL_SELECTION_METRIC, "f1")


if __name__ == "__main__":
    unittest.main()
