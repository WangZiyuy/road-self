import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import torch

from model.branch_query_decoder import MultiModalBranchQueryDecoder
from model.trajectory_evidence_encoder import TrajectoryEvidenceEncoder
from scripts.evaluate_stage3e3_robustness import (
    build_stage3e3_variant_caches,
)
from scripts.summarize_stage3e3 import SEEDS, build_stage3e3_summary
from train_branch_aux import _load_config
from train_trajectory_evidence import (
    FrozenEvidenceDataset,
    _evidence_diagnostics,
    _evidence_prediction,
    _module_state_sha256,
    _no_trajectory_prediction,
)
from utils.trajectory_evidence_robustness import (
    deterministic_fragment_thinning,
    global_wrong_sample_donor_indices,
    replace_trajectory_with_global_donors,
)


class Stage3E3RobustnessTransformTests(unittest.TestCase):

    def setUp(self):
        self.mask = torch.tensor([
            [True, True, True, True, False],
            [True, True, False, False, False],
            [False, False, False, False, False],
        ])
        self.sample_ids = torch.tensor([30, 10, 20])
        self.track = torch.tensor([
            [3, 4, 5, 6, -1],
            [8, 9, -1, -1, -1],
            [-1, -1, -1, -1, -1],
        ])
        self.start = self.track * 10
        self.end = self.start + 3

    def _thin(self, order=None, ratio=0.5):
        order = torch.arange(3) if order is None else order
        return deterministic_fragment_thinning(
            fragment_mask=self.mask[order],
            sample_ids=self.sample_ids[order],
            track_indices=self.track[order],
            start_point_indices=self.start[order],
            end_point_indices=self.end[order],
            retain_ratio=ratio,
        )

    def test_thinning_is_deterministic_batch_order_invariant_and_safe(self):
        first = self._thin()
        second = self._thin()
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(int(first[0].sum()), 2)
        self.assertEqual(int(first[1].sum()), 1)
        self.assertEqual(int(first[2].sum()), 0)
        self.assertFalse(bool((first & ~self.mask).any()))
        order = torch.tensor([2, 0, 1])
        reordered = self._thin(order)
        inverse = torch.argsort(order)
        self.assertTrue(torch.equal(first, reordered[inverse]))

    def test_thinning_keeps_one_real_fragment_and_never_padding(self):
        selected = self._thin(ratio=0.01)
        self.assertEqual(int(selected[0].sum()), 1)
        self.assertEqual(int(selected[1].sum()), 1)
        self.assertFalse(bool(selected[:, 4].any()))

    def test_wrong_mapping_is_global_and_batch_order_invariant(self):
        donors = global_wrong_sample_donor_indices(self.sample_ids)
        donor_ids = self.sample_ids[donors]
        self.assertEqual(donor_ids.tolist(), [10, 20, 30])
        order = torch.tensor([2, 0, 1])
        reordered_ids = self.sample_ids[order]
        reordered_donors = global_wrong_sample_donor_indices(reordered_ids)
        mapping = {
            int(sample): int(donor)
            for sample, donor in zip(
                reordered_ids, reordered_ids[reordered_donors])
        }
        self.assertEqual(mapping, {30: 10, 10: 20, 20: 30})

    def test_wrong_sample_replaces_only_named_trajectory_fields(self):
        tokens = torch.arange(3 * 5 * 2).reshape(3, 5, 2)
        tensors = {
            "sample_ids": self.sample_ids,
            "fragment_tokens": tokens,
            "fragment_mask": self.mask,
            "targets": torch.tensor([100, 200, 300]),
        }
        replaced = replace_trajectory_with_global_donors(
            tensors,
            trajectory_keys=("fragment_tokens", "fragment_mask"),
        )
        self.assertTrue(torch.equal(replaced["targets"], tensors["targets"]))
        self.assertEqual(
            replaced["trajectory_source_sample_ids"].tolist(),
            [10, 20, 30],
        )
        self.assertTrue(torch.equal(
            replaced["fragment_tokens"][0], tokens[1]))

    def test_variant_cache_does_not_change_fragment_token_values(self):
        tensors = {
            "sample_ids": self.sample_ids,
            "fragment_tokens": torch.randn(3, 5, 4),
            "fragment_mask": self.mask,
            "track_indices": self.track,
            "start_point_indices": self.start,
            "end_point_indices": self.end,
            "traj_xy_norm": torch.randn(3, 5, 2, 2),
            "point_mask": torch.ones(3, 5, 2, dtype=torch.bool),
        }
        cache = FrozenEvidenceDataset(tensors)
        variants = build_stage3e3_variant_caches(cache)
        for name in ("retain_75", "retain_50", "retain_25"):
            self.assertTrue(torch.equal(
                variants[name].tensors["fragment_tokens"],
                tensors["fragment_tokens"],
            ))


class Stage3E3RegressionTests(unittest.TestCase):

    def test_no_trajectory_is_exact_image_graph_prediction(self):
        torch.manual_seed(17)
        decoder = MultiModalBranchQueryDecoder(
            image_channels=8,
            trajectory_dim=16,
            hidden_dim=16,
            num_queries=3,
            num_heads=4,
            image_pool_size=2,
            dropout=0.0,
        ).eval().requires_grad_(False)
        encoder = TrajectoryEvidenceEncoder(
            hidden_dim=16,
            num_evidence_tokens=1,
            num_heads=4,
            dropout=0.0,
        ).eval()
        batch = {
            "graph_conditioned_queries": torch.randn(2, 3, 16),
            "image_context": torch.randn(2, 3, 16),
            "graph_state_contribution": torch.randn(2, 3, 16),
            "fragment_tokens": torch.randn(2, 5, 16),
            "fragment_mask": torch.zeros(2, 5, dtype=torch.bool),
        }
        image_graph = _no_trajectory_prediction(
            branch_decoder=decoder, batch=batch)
        no_trajectory = _evidence_prediction(
            evidence_encoder=encoder,
            branch_decoder=decoder,
            batch=batch,
            return_attention=True,
        )
        for key in (
                "branch_exist_logits",
                "branch_offsets_norm",
                "branch_directions"):
            self.assertTrue(torch.equal(image_graph[key], no_trajectory[key]))
        self.assertTrue(torch.equal(
            no_trajectory["trajectory_evidence_tokens"],
            torch.zeros_like(no_trajectory["trajectory_evidence_tokens"]),
        ))

    def test_only_evidence_encoder_updates_and_frozen_sha_is_stable(self):
        torch.manual_seed(18)
        decoder = MultiModalBranchQueryDecoder(
            image_channels=8,
            trajectory_dim=16,
            hidden_dim=16,
            num_queries=3,
            num_heads=4,
            image_pool_size=2,
            dropout=0.0,
        ).eval().requires_grad_(False)
        encoder = TrajectoryEvidenceEncoder(
            hidden_dim=16,
            num_evidence_tokens=1,
            num_heads=4,
            dropout=0.0,
        ).train()
        decoder_sha = _module_state_sha256(decoder)
        encoder_sha = _module_state_sha256(encoder)
        optimizer = torch.optim.SGD(encoder.parameters(), lr=0.01)
        batch = {
            "graph_conditioned_queries": torch.randn(2, 3, 16),
            "image_context": torch.randn(2, 3, 16),
            "graph_state_contribution": torch.randn(2, 3, 16),
            "fragment_tokens": torch.randn(2, 5, 16),
            "fragment_mask": torch.ones(2, 5, dtype=torch.bool),
        }
        prediction = _evidence_prediction(
            evidence_encoder=encoder,
            branch_decoder=decoder,
            batch=batch,
            return_attention=False,
        )
        prediction["branch_exist_logits"].sum().backward()
        optimizer.step()
        self.assertEqual(_module_state_sha256(decoder), decoder_sha)
        self.assertNotEqual(_module_state_sha256(encoder), encoder_sha)
        self.assertTrue(all(parameter.grad is None
                            for parameter in decoder.parameters()))

    def test_m1_diagnostics_include_mass_entropy_and_empty_zero(self):
        tokens = torch.tensor([[[1.0, 2.0]], [[0.0, 0.0]]]).numpy()
        evidence_mask = torch.tensor([[True], [False]]).numpy()
        attention = torch.tensor([
            [[0.5, 0.3, 0.2]],
            [[0.0, 0.0, 0.0]],
        ]).numpy()
        fragment_mask = torch.tensor([
            [True, True, True],
            [False, False, False],
        ]).numpy()
        diagnostics = _evidence_diagnostics(
            tokens, evidence_mask, attention, fragment_mask)
        self.assertAlmostEqual(
            diagnostics["top1_cumulative_attention_mass"]["mean"], 0.5)
        self.assertAlmostEqual(
            diagnostics["top4_cumulative_attention_mass"]["mean"], 1.0)
        self.assertGreater(
            diagnostics["effective_fragment_count"]["mean"], 1.0)
        self.assertTrue(diagnostics["empty_trajectory_context_is_zero"])

    def test_seed_configs_change_only_seed_and_output(self):
        configs = {
            seed: _load_config(Path("configs") / (
                "stage3e3_seed{}.yml".format(seed)))
            for seed in SEEDS
        }
        normalized = {}
        for seed, cfg in configs.items():
            self.assertEqual(int(cfg.STAGE3C.SEED), seed)
            self.assertEqual(int(cfg.STAGE3E0.MODEL.NUM_EVIDENCE_TOKENS), 1)
            self.assertEqual(
                cfg.STAGE3E0.MODEL.AGGREGATION_MODE, "latent_attention")
            value = deepcopy(dict(cfg))
            value["STAGE3C"]["SEED"] = 0
            value["STAGE3E0"]["OUTPUT_DIR"] = ""
            normalized[seed] = value
        self.assertEqual(normalized[SEEDS[0]], normalized[SEEDS[1]])
        self.assertEqual(normalized[SEEDS[0]], normalized[SEEDS[2]])


class Stage3E3SummaryTests(unittest.TestCase):

    @staticmethod
    def _variant(ap):
        thresholded = {
            "endpoint_error_mean_pixels": 2.0,
            "endpoint_error_median_pixels": 1.0,
            "direction_error_mean_degrees": 3.0,
            "direction_error_median_degrees": 2.0,
            "exact_branch_count_accuracy": 0.8,
            "missed_branch_rate": 0.1,
            "extra_branch_rate": 0.1,
            "precision": 0.8,
            "recall": 0.8,
            "f1": 0.8,
        }
        group = lambda value: {"branch_ap": value}
        return {
            "branch_ap": ap,
            "slot_ap": ap - 0.01,
            "thresholded": thresholded,
            "by_category": {
                "ordinary": group(ap),
                "t_junction": group(ap),
                "multi_branch": group(ap),
            },
            "by_gt_count": {
                "count_0": group(0.0),
                "count_1": group(ap),
                "count_2": group(ap),
                "count_ge3": group(ap),
            },
        }

    def test_summary_applies_acceptance_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in SEEDS:
                seed_dir = root / "seed{}".format(seed)
                seed_dir.mkdir()
                (seed_dir / "training_summary.json").write_text(json.dumps({
                    "frozen_modules_unchanged": True,
                    "best_epoch": 3,
                    "best_checkpoint": "best.pth.tar",
                    "best_checkpoint_sha256": "checkpoint",
                    "elapsed_seconds": 1.0,
                    "peak_cuda_memory_bytes": 10,
                    "e4_checkpoint_sha256": "a",
                    "cache": {
                        "train": {
                            "fragment_tokens_sha256": "train-tokens",
                            "fragment_mask_sha256": "train-mask",
                            "sample_ids_sha256": "train-ids",
                        },
                        "val": {
                            "fragment_tokens_sha256": "val-tokens",
                            "fragment_mask_sha256": "val-mask",
                            "sample_ids_sha256": "val-ids",
                        },
                        "stage3e3_preflight": {"all": "matched"},
                    },
                }), encoding="utf-8")
                variants = {
                    "image_graph": self._variant(0.80),
                    "no_trajectory": self._variant(0.80),
                    "original_fragment": self._variant(0.81),
                    "full_trajectory": self._variant(0.82),
                    "retain_75": self._variant(0.819),
                    "retain_50": self._variant(0.817),
                    "retain_25": self._variant(0.810),
                    "wrong_sample_trajectory": self._variant(0.79),
                }
                report = {
                    "hash_checks": {
                        "e4_checkpoint": "a",
                        "val_fragment_tokens": "b",
                        "val_fragment_mask": "c",
                        "val_sample_ids": "d",
                    },
                    "variants": variants,
                    "attention_diagnostics": {
                        mode: {
                            "hidden_norm": {"mean": 1.0},
                            "normalized_fragment_attention_entropy": {
                                "mean": 0.9},
                            "maximum_fragment_attention": {"mean": 0.1},
                            "top1_cumulative_attention_mass": {"mean": 0.1},
                            "top4_cumulative_attention_mass": {"mean": 0.3},
                            "top8_cumulative_attention_mass": {"mean": 0.5},
                            "top16_cumulative_attention_mass": {"mean": 0.8},
                            "effective_fragment_count": {"mean": 20.0},
                            "all_finite": True,
                            "empty_trajectory_context_is_zero": True,
                        }
                        for mode in (
                            "full_trajectory",
                            "retain_50",
                            "retain_25",
                            "wrong_sample_trajectory",
                        )
                    },
                    "no_trajectory_equivalence": {"maximum": 0.0},
                    "frozen_modules_unchanged": True,
                }
                (seed_dir / "robustness_evaluation.json").write_text(
                    json.dumps(report), encoding="utf-8")
            comparison = build_stage3e3_summary(
                root,
                reproduction_branch_ap=0.9138492084801799,
            )
            self.assertTrue(comparison["acceptance"]["passed"])
            self.assertAlmostEqual(
                comparison["full_minus_image_graph"]["mean"], 0.02)


if __name__ == "__main__":
    unittest.main()
