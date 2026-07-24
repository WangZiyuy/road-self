import tempfile
import unittest
from pathlib import Path

import torch
from easydict import EasyDict

from scripts.run_stage3c_r1 import (
    _validate_configs,
    build_r1_decision,
)
from train_branch_aux import (
    _build_sanity_checkpoint_payload,
    _evaluate_optional_sanity_gate,
    _evaluate_sanity_gate,
    _load_config,
    _sanity_candidate_is_better,
)
from utils.stage3c_checkpoint import save_stage3c_checkpoint


def _final_metrics(
    *,
    branch_ap=0.9,
    exact=0.9,
    coverage=0.95,
    duplicate=0.1,
    loss=0.2,
):
    return {
        "branch_ap": branch_ap,
        "loss": {"total": loss},
        "thresholded_metrics": {
            "exact_branch_count_accuracy": exact,
        },
        "oracle_k": {
            "distinct_gt_coverage": coverage,
            "duplicates": {
                "duplicate_pair_ratio": duplicate,
            },
        },
    }


def _decision_variant(
    *,
    slot_ap,
    gap,
    graph_cosine,
    final_cosine,
    coverage,
    duplicate,
    gate,
):
    return {
        "slot_ap": slot_ap,
        "probability_separation_mean": gap,
        "query_representation": {
            "graph_conditioned_query": {
                "pairwise_cosine": {"mean": graph_cosine},
            },
            "final_fused_query": {
                "pairwise_cosine": {"mean": final_cosine},
            },
        },
        "oracle_k_distinct_gt_coverage": coverage,
        "oracle_k_duplicate_pair_ratio": duplicate,
        "sanity_gate_passed": gate,
    }


class Stage3CR1Test(unittest.TestCase):
    def test_base_sanity_without_optional_gate_remains_eligible(self):
        configured, passed, checks = (
            _evaluate_optional_sanity_gate(
                EasyDict({}),
                _final_metrics(branch_ap=0.1),
            )
        )
        self.assertFalse(configured)
        self.assertTrue(passed)
        self.assertEqual(checks, {})

    def test_better_intermediate_sanity_epoch_beats_final_epoch(self):
        intermediate = _final_metrics(
            branch_ap=0.95,
            coverage=0.96,
            duplicate=0.05,
            loss=0.1,
        )
        final = _final_metrics(
            branch_ap=0.85,
            coverage=0.92,
            duplicate=0.08,
            loss=0.2,
        )
        self.assertTrue(_sanity_candidate_is_better(
            candidate=intermediate,
            candidate_gate_passed=True,
            incumbent=final,
            incumbent_gate_passed=True,
            optional_gate_configured=True,
        ))
        self.assertFalse(_sanity_candidate_is_better(
            candidate=final,
            candidate_gate_passed=True,
            incumbent=intermediate,
            incumbent_gate_passed=True,
            optional_gate_configured=True,
        ))

    def test_sanity_best_and_final_checkpoints_are_distinct_snapshots(self):
        modules = tuple(
            torch.nn.Linear(1, 1, bias=False) for _ in range(3))
        optimizer = torch.optim.Adam(
            [parameter for module in modules
             for parameter in module.parameters()],
            lr=1e-3,
        )
        cfg = EasyDict({"STAGE3C": {"name": "unit"}})
        with tempfile.TemporaryDirectory(
                dir=Path.cwd()) as temporary:
            root = Path(temporary)
            best_path = root / "sanity_overfit.best.pth.tar"
            final_path = root / "sanity_overfit.final.pth.tar"
            for module in modules:
                module.weight.data.fill_(1.0)
            best_payload = _build_sanity_checkpoint_payload(
                modules=modules,
                optimizer=optimizer,
                epoch=10,
                image_checkpoint=Path("image.pth.tar"),
                cfg=cfg,
                evaluation=_final_metrics(branch_ap=0.95),
                optional_gate_configured=True,
                optional_gate_passed=True,
            )
            save_stage3c_checkpoint(best_path, best_payload)
            for module in modules:
                module.weight.data.fill_(2.0)
            final_payload = _build_sanity_checkpoint_payload(
                modules=modules,
                optimizer=optimizer,
                epoch=20,
                image_checkpoint=Path("image.pth.tar"),
                cfg=cfg,
                evaluation=_final_metrics(branch_ap=0.80),
                optional_gate_configured=True,
                optional_gate_passed=True,
            )
            save_stage3c_checkpoint(final_path, final_payload)

            best = torch.load(best_path, map_location="cpu")
            final = torch.load(final_path, map_location="cpu")
            self.assertNotEqual(best_path, final_path)
            self.assertEqual(best["epoch"], 10)
            self.assertEqual(final["epoch"], 20)
            torch.testing.assert_close(
                best["trajectory_encoder"]["weight"],
                torch.ones(1, 1),
            )
            torch.testing.assert_close(
                final["trajectory_encoder"]["weight"],
                torch.full((1, 1), 2.0),
            )

    def test_base_sanity_gate_preserves_legacy_reduction_behavior(self):
        sanity = EasyDict({
            "MIN_TOTAL_LOSS_REDUCTION": 0.5,
            "MIN_ENDPOINT_ERROR_REDUCTION": 0.4,
            "MIN_DIRECTION_ERROR_REDUCTION": 0.25,
        })
        passed, checks = _evaluate_sanity_gate(
            sanity_cfg=sanity,
            total_reduction=0.6,
            endpoint_reduction=0.5,
            direction_reduction=0.3,
            final=_final_metrics(
                branch_ap=0.0,
                exact=0.0,
                coverage=0.0,
                duplicate=1.0,
            ),
        )
        self.assertTrue(passed)
        self.assertEqual(len(checks), 3)

    def test_optional_multibranch_gate_checks_all_four_metrics(self):
        sanity = EasyDict({
            "MIN_TOTAL_LOSS_REDUCTION": 0.5,
            "MIN_ENDPOINT_ERROR_REDUCTION": 0.4,
            "MIN_DIRECTION_ERROR_REDUCTION": 0.25,
            "MIN_BRANCH_AP": 0.8,
            "MIN_EXACT_COUNT_ACCURACY": 0.8,
            "MIN_ORACLE_K_DISTINCT_GT_COVERAGE": 0.9,
            "MAX_ORACLE_K_DUPLICATE_PAIR_RATIO": 0.2,
        })
        passed, checks = _evaluate_sanity_gate(
            sanity_cfg=sanity,
            total_reduction=0.6,
            endpoint_reduction=0.5,
            direction_reduction=0.3,
            final=_final_metrics(),
        )
        self.assertTrue(passed)
        self.assertEqual(len(checks), 7)
        failed, failed_checks = _evaluate_sanity_gate(
            sanity_cfg=sanity,
            total_reduction=0.6,
            endpoint_reduction=0.5,
            direction_reduction=0.3,
            final=_final_metrics(duplicate=0.21),
        )
        self.assertFalse(failed)
        self.assertFalse(
            failed_checks[
                "oracle_k_duplicate_pair_ratio"]["passed"])

    def test_m0_m3_m4_configs_have_exact_controlled_differences(self):
        configs = {
            "M0": _load_config(Path("configs/stage3c_r1_m0.yml")),
            "M3": _load_config(Path("configs/stage3c_r1_m3.yml")),
            "M4": _load_config(Path("configs/stage3c_r1_m4.yml")),
        }
        _validate_configs(configs)
        self.assertEqual({
            int(configs[name].STAGE3C.SANITY.SAMPLE_COUNT)
            for name in configs
        }, {32})
        self.assertEqual({
            int(configs[name].STAGE3C.SEED)
            for name in configs
        }, {20260724})

    def test_decision_requires_m4_identity_coverage_and_gate(self):
        variants = {
            "M0": _decision_variant(
                slot_ap=0.3, gap=0.1,
                graph_cosine=0.99, final_cosine=0.99,
                coverage=0.4, duplicate=1.0, gate=False),
            "M3": _decision_variant(
                slot_ap=0.6, gap=0.3,
                graph_cosine=0.99, final_cosine=0.99,
                coverage=0.5, duplicate=0.9, gate=False),
            "M4": _decision_variant(
                slot_ap=0.9, gap=0.7,
                graph_cosine=0.2, final_cosine=0.3,
                coverage=0.95, duplicate=0.1, gate=True),
        }
        decision = build_r1_decision(variants)
        self.assertTrue(decision[
            "m3_probability_separation_improved_over_m0"])
        self.assertTrue(decision["m3_still_has_query_collapse"])
        self.assertTrue(decision["m4_restores_query_identity"])
        self.assertTrue(decision["m4_covers_distinct_gt_branches"])
        self.assertTrue(decision["recommend_full_e1_e4_training"])
        variants["M4"]["sanity_gate_passed"] = False
        self.assertFalse(build_r1_decision(variants)[
            "recommend_full_e1_e4_training"])

    def test_m3_early_homogeneity_is_not_functional_collapse_alone(self):
        variants = {
            "M0": _decision_variant(
                slot_ap=0.3, gap=0.1,
                graph_cosine=0.99, final_cosine=0.4,
                coverage=0.4, duplicate=0.8, gate=False),
            "M3": _decision_variant(
                slot_ap=1.0, gap=0.99,
                graph_cosine=0.99, final_cosine=0.1,
                coverage=1.0, duplicate=0.02, gate=True),
            "M4": _decision_variant(
                slot_ap=0.9, gap=0.8,
                graph_cosine=0.3, final_cosine=0.2,
                coverage=0.6, duplicate=0.01, gate=False),
        }
        decision = build_r1_decision(variants)
        self.assertFalse(decision["m3_still_has_query_collapse"])
        self.assertTrue(decision[
            "m3_graph_conditioning_remains_homogeneous"])


if __name__ == "__main__":
    unittest.main()
