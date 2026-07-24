import unittest
from pathlib import Path

import numpy as np

from scripts.diagnose_stage3c_branch_aux import (
    _modality_diagnostic_summary,
)
from scripts.run_stage3c_r2 import (
    EXPERIMENT_NAMES,
    build_r2_decisions,
    validate_r2_configs,
)
from train_branch_aux import _load_config


def _experiment(branch_ap, no_trajectory_ap, coverage, duplicate):
    full = {
        "branch_ap": branch_ap,
        "slot_ap": branch_ap,
        "oracle_k": {"distinct_gt_coverage": coverage},
        "oracle_k_duplicate_ratio": duplicate,
    }
    return {
        "metrics_by_modality": {
            "full": full,
            "no_trajectory": {"branch_ap": no_trajectory_ap},
        },
        "oracle_k_distinct_gt_coverage": coverage,
        "oracle_k_duplicate_ratio": duplicate,
    }


class Stage3CR2Test(unittest.TestCase):
    def test_e0_e4_configs_are_a_controlled_matrix(self):
        paths = {
            "E0": "configs/stage3c_e0_fresh_baseline.yml",
            "E1": "configs/stage3c_e1_existence_matching_only.yml",
            "E2": "configs/stage3c_e2_no_object_weight_only.yml",
            "E3": "configs/stage3c_e3_matching_plus_no_object.yml",
            "E4": "configs/stage3c_e4_matching_no_object_self_attn.yml",
        }
        configs = {
            name: _load_config(Path(path))
            for name, path in paths.items()
        }
        validate_r2_configs(configs)
        self.assertEqual({
            int(configs[name].STAGE3C.DATASET.TRAIN_SAMPLES)
            for name in EXPERIMENT_NAMES
        }, {2048})
        self.assertEqual({
            int(configs[name].STAGE3C.DATASET.VAL_SAMPLES)
            for name in EXPERIMENT_NAMES
        }, {512})
        self.assertEqual({
            str(configs[name].STAGE3C.EVALUATION
                .MODEL_SELECTION_METRIC)
            for name in EXPERIMENT_NAMES
        }, {"branch_ap"})

    def test_decisions_use_validation_deltas_and_collapse_metrics(self):
        experiments = {
            "E0": _experiment(0.30, 0.28, 0.60, 0.30),
            "E1": _experiment(0.40, 0.35, 0.70, 0.20),
            "E2": _experiment(0.45, 0.38, 0.75, 0.18),
            "E3": _experiment(0.60, 0.50, 0.90, 0.10),
            "E4": _experiment(0.90, 0.72, 0.92, 0.08),
        }
        decisions = build_r2_decisions(experiments)
        self.assertTrue(
            decisions["existence_matching_alone_effective"])
        self.assertTrue(
            decisions["no_object_weighting_alone_effective"])
        self.assertTrue(
            decisions["matching_plus_no_object_most_stable"])
        self.assertTrue(
            decisions["self_attention_improves_validation"])
        self.assertTrue(
            decisions["full_stably_better_than_no_trajectory"])
        self.assertFalse(
            decisions["multi_branch_query_collapse_remains"])
        self.assertTrue(
            decisions["ready_for_trajectory_support_supervision"])

    def test_modality_summary_reports_required_metrics_by_gt_count(self):
        probabilities = np.asarray([[0.99, 0.01]], dtype=np.float32)
        logits = np.log(
            probabilities / (1.0 - probabilities))
        offsets = np.asarray(
            [[[0.5, 0.0], [-0.5, 0.0]]], dtype=np.float32)
        directions = np.asarray(
            [[[1.0, 0.0], [-1.0, 0.0]]], dtype=np.float32)
        target_offsets = offsets[:, :1]
        target_directions = directions[:, :1]
        target_mask = np.asarray([[True]])
        actual_labels = np.asarray([[True, False]])
        cfg = _load_config(Path(
            "configs/stage3c_e0_fresh_baseline.yml"))
        summary = _modality_diagnostic_summary(
            values={
                "probability": probabilities,
                "logit": logits,
                "offsets": offsets,
                "directions": directions,
            },
            actual_labels=actual_labels,
            actual_selected=[np.asarray([0], dtype=np.int64)],
            target_offsets=target_offsets,
            target_directions=target_directions,
            target_mask=target_mask,
            gt_counts=np.asarray([1], dtype=np.int64),
            cfg=cfg,
            threshold=0.5,
        )
        self.assertAlmostEqual(summary["branch_ap"], 1.0)
        self.assertAlmostEqual(summary["slot_ap"], 1.0)
        self.assertAlmostEqual(
            summary["exact_count_accuracy"], 1.0)
        self.assertAlmostEqual(
            summary["oracle_k"]["distinct_gt_coverage"], 1.0)
        self.assertEqual(
            summary["metrics_by_gt_count"]["count_0"][
                "sample_count"],
            0,
        )
        self.assertEqual(
            summary["metrics_by_gt_count"]["count_1"][
                "sample_count"],
            1,
        )

    def test_support_supervision_readiness_does_not_claim_traj_gain(self):
        experiments = {
            "E0": _experiment(0.11, 0.10, 0.53, 0.47),
            "E1": _experiment(0.12, 0.11, 0.70, 1.0),
            "E2": _experiment(0.09, 0.08, 0.66, 1.0),
            "E3": _experiment(0.10, 0.12, 0.68, 1.0),
            "E4": _experiment(0.90, 0.895, 0.85, 0.0),
        }
        decisions = build_r2_decisions(experiments)
        self.assertFalse(decisions[
            "structured_trajectory_increment_demonstrated"])
        self.assertTrue(decisions[
            "ready_for_trajectory_support_supervision"])
        self.assertFalse(decisions["ready_for_anchor_fusion"])


if __name__ == "__main__":
    unittest.main()
