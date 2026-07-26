import unittest
import uuid
from copy import deepcopy
from pathlib import Path
from unittest import mock

import torch
from easydict import EasyDict

from model.branch_query_decoder import MultiModalBranchQueryDecoder
from model.trajectory_evidence_encoder import (
    EVIDENCE_AGGREGATION_LATENT_ATTENTION,
    EVIDENCE_AGGREGATION_MASKED_MEAN,
    TRAJECTORY_MODE_EVIDENCE,
    TRAJECTORY_MODE_ORIGINAL_FRAGMENT,
    TrajectoryEvidenceEncoder,
    build_trajectory_decoder_inputs,
    resolve_evidence_aggregation_mode,
    resolve_trajectory_evidence_mode,
)
from train_trajectory_evidence import _evidence_diagnostics
from train_branch_aux import _load_config
from scripts.summarize_stage3e1a import (
    build_comparison as build_stage3e1a_comparison,
)
from scripts.summarize_stage3e2a import (
    build_comparison as build_stage3e2a_comparison,
)
from utils.stage3e0_checkpoint import (
    build_stage3e0_checkpoint_payload,
    load_stage3e0_checkpoint,
    save_stage3e0_checkpoint,
)


class TrajectoryEvidenceEncoderTests(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(20260726)
        self.encoder = TrajectoryEvidenceEncoder(
            hidden_dim=32,
            num_evidence_tokens=4,
            num_heads=4,
            dropout=0.0,
        ).eval()
        self.fragments = torch.randn(2, 6, 32)
        self.mask = torch.tensor([
            [True, True, True, False, False, False],
            [True, True, True, True, True, False],
        ])

    def test_output_shape_mask_and_attention(self):
        output = self.encoder(
            self.fragments, self.mask, return_attention=True)
        self.assertEqual(
            tuple(output["trajectory_evidence_tokens"].shape),
            (2, 4, 32),
        )
        self.assertTrue(bool(
            output["trajectory_evidence_mask"].all()))
        attention = output["fragment_attention_weights"]
        self.assertEqual(tuple(attention.shape), (2, 4, 6))
        self.assertTrue(torch.allclose(
            attention.masked_select(~self.mask.unsqueeze(1)),
            torch.zeros_like(
                attention.masked_select(~self.mask.unsqueeze(1))),
        ))
        self.assertTrue(torch.allclose(
            attention.sum(dim=-1),
            torch.ones(2, 4),
            atol=1e-6,
        ))

    def test_empty_sample_is_zero_and_finite(self):
        mask = self.mask.clone()
        mask[0] = False
        output = self.encoder(
            self.fragments, mask, return_attention=True)
        self.assertTrue(torch.equal(
            output["trajectory_evidence_tokens"][0],
            torch.zeros_like(
                output["trajectory_evidence_tokens"][0]),
        ))
        self.assertFalse(bool(
            output["trajectory_evidence_mask"][0].any()))
        self.assertTrue(torch.equal(
            output["fragment_attention_weights"][0],
            torch.zeros_like(
                output["fragment_attention_weights"][0]),
        ))
        self.assertTrue(torch.isfinite(
            output["trajectory_evidence_tokens"]).all())

    def test_zero_fragment_dimension_is_supported(self):
        output = self.encoder(
            torch.zeros(2, 0, 32),
            torch.zeros(2, 0, dtype=torch.bool),
            return_attention=True,
        )
        self.assertEqual(
            tuple(output["fragment_attention_weights"].shape),
            (2, 4, 0),
        )
        self.assertFalse(bool(
            output["trajectory_evidence_mask"].any()))

    def test_fragment_reordering_only_reorders_attention(self):
        permutation = torch.tensor([4, 0, 5, 1, 3, 2])
        original = self.encoder(
            self.fragments, self.mask, return_attention=True)
        reordered = self.encoder(
            self.fragments[:, permutation],
            self.mask[:, permutation],
            return_attention=True,
        )
        self.assertTrue(torch.allclose(
            original["trajectory_evidence_tokens"],
            reordered["trajectory_evidence_tokens"],
            atol=1e-6,
        ))
        self.assertTrue(torch.allclose(
            original["fragment_attention_weights"][:, :, permutation],
            reordered["fragment_attention_weights"],
            atol=1e-6,
        ))

    def test_original_fragment_mode_is_an_exact_passthrough(self):
        output = build_trajectory_decoder_inputs(
            trajectory_mode=TRAJECTORY_MODE_ORIGINAL_FRAGMENT,
            fragment_tokens=self.fragments,
            fragment_mask=self.mask,
        )
        self.assertIs(
            output["decoder_trajectory_tokens"], self.fragments)
        self.assertTrue(torch.equal(
            output["decoder_trajectory_mask"], self.mask))
        self.assertIsNone(output["trajectory_evidence_tokens"])

    def test_evidence_mode_requires_encoder(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            build_trajectory_decoder_inputs(
                trajectory_mode=TRAJECTORY_MODE_EVIDENCE,
                fragment_tokens=self.fragments,
                fragment_mask=self.mask,
            )

    def test_resolver_defaults_to_original_and_rejects_unknown(self):
        self.assertEqual(
            resolve_trajectory_evidence_mode(EasyDict()),
            TRAJECTORY_MODE_ORIGINAL_FRAGMENT,
        )
        self.assertEqual(
            resolve_trajectory_evidence_mode(EasyDict(
                TRAJECTORY_MODE="trajectory_evidence")),
            TRAJECTORY_MODE_EVIDENCE,
        )
        with self.assertRaisesRegex(ValueError, "unknown"):
            resolve_trajectory_evidence_mode(EasyDict(
                TRAJECTORY_MODE="branch_query"))

    def test_evidence_aggregation_resolver_is_legacy_safe(self):
        self.assertEqual(
            resolve_evidence_aggregation_mode(EasyDict()),
            EVIDENCE_AGGREGATION_LATENT_ATTENTION,
        )
        cfg = EasyDict({
            "STAGE3E0": {
                "MODEL": {
                    "AGGREGATION_MODE": "masked_mean",
                },
            },
        })
        self.assertEqual(
            resolve_evidence_aggregation_mode(cfg),
            EVIDENCE_AGGREGATION_MASKED_MEAN,
        )
        cfg.STAGE3E0.MODEL.AGGREGATION_MODE = "unknown"
        with self.assertRaisesRegex(ValueError, "unknown"):
            resolve_evidence_aggregation_mode(cfg)

    def test_masked_mean_is_exact_parameter_free_and_padding_safe(self):
        encoder = TrajectoryEvidenceEncoder(
            hidden_dim=2,
            num_evidence_tokens=1,
            num_heads=1,
            dropout=0.0,
            aggregation_mode=EVIDENCE_AGGREGATION_MASKED_MEAN,
        )
        self.assertEqual(sum(
            parameter.numel()
            for parameter in encoder.parameters()), 0)
        fragments = torch.tensor([
            [[2.0, 4.0], [6.0, 8.0], [100.0, 200.0]],
            [[7.0, 9.0], [11.0, 13.0], [17.0, 19.0]],
        ])
        mask = torch.tensor([
            [True, True, False],
            [False, False, False],
        ])
        output = encoder(
            fragments, mask, return_attention=True)
        torch.testing.assert_close(
            output["trajectory_evidence_tokens"][0, 0],
            torch.tensor([4.0, 6.0]),
        )
        torch.testing.assert_close(
            output["fragment_attention_weights"][0, 0],
            torch.tensor([0.5, 0.5, 0.0]),
        )
        self.assertTrue(output[
            "trajectory_evidence_mask"][0, 0])
        self.assertFalse(output[
            "trajectory_evidence_mask"][1, 0])
        self.assertTrue(torch.equal(
            output["trajectory_evidence_tokens"][1],
            torch.zeros_like(
                output["trajectory_evidence_tokens"][1]),
        ))
        self.assertTrue(torch.isfinite(
            output["trajectory_evidence_tokens"]).all())
        diagnostics = _evidence_diagnostics(
            output["trajectory_evidence_tokens"].detach().numpy(),
            output["trajectory_evidence_mask"].detach().numpy(),
            output["fragment_attention_weights"].detach().numpy(),
            mask.numpy(),
        )
        self.assertAlmostEqual(
            diagnostics[
                "normalized_fragment_attention_entropy"]["mean"],
            1.0,
            places=6,
        )

    def test_masked_mean_rejects_multiple_output_tokens(self):
        with self.assertRaisesRegex(
                ValueError, "num_evidence_tokens=1"):
            TrajectoryEvidenceEncoder(
                hidden_dim=8,
                num_evidence_tokens=4,
                num_heads=1,
                aggregation_mode=(
                    EVIDENCE_AGGREGATION_MASKED_MEAN),
            )

    def test_original_ablation_reproduces_decoder_forward(self):
        decoder = MultiModalBranchQueryDecoder(
            image_channels=16,
            trajectory_dim=32,
            hidden_dim=32,
            num_queries=3,
            num_heads=4,
            image_pool_size=4,
            dropout=0.0,
        ).eval()
        stage_fuse = torch.randn(2, 16, 8, 8)
        state = torch.randn(2, 32)
        walked = torch.randn(2, 1, 8, 8)
        direct = decoder(
            stage_fuse,
            state,
            self.fragments,
            self.mask,
            walked,
        )
        adapted = build_trajectory_decoder_inputs(
            trajectory_mode=TRAJECTORY_MODE_ORIGINAL_FRAGMENT,
            fragment_tokens=self.fragments,
            fragment_mask=self.mask,
        )
        wrapped = decoder(
            stage_fuse,
            state,
            adapted["decoder_trajectory_tokens"],
            adapted["decoder_trajectory_mask"],
            walked,
        )
        for key in (
                "branch_exist_logits",
                "branch_offsets_norm",
                "branch_directions"):
            self.assertTrue(torch.equal(direct[key], wrapped[key]))

    def test_gradient_reaches_only_evidence_encoder(self):
        decoder = MultiModalBranchQueryDecoder(
            image_channels=16,
            trajectory_dim=32,
            hidden_dim=32,
            num_queries=3,
            num_heads=4,
            image_pool_size=4,
            dropout=0.0,
        ).eval().requires_grad_(False)
        encoder = TrajectoryEvidenceEncoder(
            hidden_dim=32,
            num_evidence_tokens=4,
            num_heads=4,
            dropout=0.0,
        ).train()
        adapted = build_trajectory_decoder_inputs(
            trajectory_mode=TRAJECTORY_MODE_EVIDENCE,
            fragment_tokens=self.fragments,
            fragment_mask=self.mask,
            evidence_encoder=encoder,
        )
        output = decoder(
            torch.randn(2, 16, 8, 8),
            torch.randn(2, 32),
            adapted["decoder_trajectory_tokens"],
            adapted["decoder_trajectory_mask"],
            torch.randn(2, 1, 8, 8),
        )
        output["branch_exist_logits"].sum().backward()
        self.assertTrue(any(
            parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all())
            and float(parameter.grad.abs().sum()) > 0.0
            for parameter in encoder.parameters()
        ))
        self.assertTrue(all(
            parameter.grad is None
            for parameter in decoder.parameters()
        ))

    def test_checkpoint_round_trip_and_sha_guard(self):
        encoder = TrajectoryEvidenceEncoder(
            hidden_dim=32,
            num_evidence_tokens=4,
            num_heads=4,
            dropout=0.0,
        )
        optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
        payload = build_stage3e0_checkpoint_payload(
            evidence_encoder=encoder,
            optimizer=optimizer,
            epoch=3,
            trajectory_mode=TRAJECTORY_MODE_EVIDENCE,
            e4_checkpoint="e4.pth.tar",
            e4_checkpoint_sha256="abc123",
            config={"seed": 7},
            metrics={"branch_ap": 0.5},
        )
        path = Path("tests") / (
            "_stage3e0_{}.pth.tar".format(uuid.uuid4().hex))
        try:
            save_stage3e0_checkpoint(path, payload)
            restored = TrajectoryEvidenceEncoder(
                hidden_dim=32,
                num_evidence_tokens=4,
                num_heads=4,
                dropout=0.0,
            )
            loaded = load_stage3e0_checkpoint(
                path,
                evidence_encoder=restored,
                expected_e4_sha256="abc123",
            )
            self.assertEqual(loaded["epoch"], 3)
            for left, right in zip(
                    encoder.parameters(), restored.parameters()):
                self.assertTrue(torch.equal(left, right))
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_stage3e0_checkpoint(
                    path,
                    evidence_encoder=restored,
                    expected_e4_sha256="different",
                )
        finally:
            path.unlink(missing_ok=True)

    def test_parameter_free_mean_checkpoint_round_trip(self):
        encoder = TrajectoryEvidenceEncoder(
            hidden_dim=8,
            num_evidence_tokens=1,
            num_heads=1,
            aggregation_mode=EVIDENCE_AGGREGATION_MASKED_MEAN,
        )
        payload = build_stage3e0_checkpoint_payload(
            evidence_encoder=encoder,
            optimizer=None,
            epoch=0,
            trajectory_mode=TRAJECTORY_MODE_EVIDENCE,
            e4_checkpoint="e4.pth.tar",
            e4_checkpoint_sha256="mean-sha",
            config={"aggregation_mode": "masked_mean"},
        )
        path = Path("tests") / (
            "_stage3e0_mean_{}.pth.tar".format(uuid.uuid4().hex))
        try:
            save_stage3e0_checkpoint(path, payload)
            restored = TrajectoryEvidenceEncoder(
                hidden_dim=8,
                num_evidence_tokens=1,
                num_heads=1,
                aggregation_mode=EVIDENCE_AGGREGATION_MASKED_MEAN,
            )
            loaded = load_stage3e0_checkpoint(
                path,
                evidence_encoder=restored,
                expected_e4_sha256="mean-sha",
            )
            self.assertIsNone(loaded["optimizer"])
            self.assertEqual(
                loaded["trajectory_evidence_encoder"], {})
        finally:
            path.unlink(missing_ok=True)

    def test_diagnostics_expose_latent_and_attention_collapse(self):
        tokens = torch.ones(1, 4, 8).numpy()
        attention = torch.tensor([[
            [0.7, 0.2, 0.1],
            [0.7, 0.2, 0.1],
            [0.7, 0.2, 0.1],
            [0.7, 0.2, 0.1],
        ]]).numpy()
        diagnostics = _evidence_diagnostics(
            tokens,
            torch.ones(1, 4, dtype=torch.bool).numpy(),
            attention,
            torch.ones(1, 3, dtype=torch.bool).numpy(),
        )
        self.assertAlmostEqual(
            diagnostics[
                "pairwise_cosine_similarity"]["mean"],
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            diagnostics[
                "fragment_attention_pairwise_cosine_similarity"][
                    "mean"],
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            diagnostics[
                "fragment_attention_top8_jaccard"]["mean"],
            1.0,
            places=6,
        )

    def test_shared_initialization_is_identical_across_token_counts(self):
        shared_states = []
        initialized_queries = []
        for token_count in (1, 4, 8):
            torch.manual_seed(12345)
            encoder = TrajectoryEvidenceEncoder(
                hidden_dim=32,
                num_evidence_tokens=token_count,
                num_heads=4,
                dropout=0.0,
            )
            shared_states.append({
                key: value
                for key, value in encoder.state_dict().items()
                if key != "trajectory_queries"
            })
            initialized_queries.append(
                encoder.trajectory_queries.detach().clone())
            self.assertEqual(
                tuple(encoder.trajectory_queries.shape),
                (1, token_count, 32),
            )
        for key in shared_states[0]:
            self.assertTrue(torch.equal(
                shared_states[0][key], shared_states[1][key]))
            self.assertTrue(torch.equal(
                shared_states[0][key], shared_states[2][key]))
        self.assertTrue(torch.equal(
            initialized_queries[0], initialized_queries[1][:, :1]))
        self.assertTrue(torch.equal(
            initialized_queries[1], initialized_queries[2][:, :4]))


class Stage3E1AAblationTests(unittest.TestCase):

    def test_m1_m4_m8_configs_only_change_count_and_output(self):
        configs = {
            name: _load_config(Path("configs") / (
                "stage3e1a_{}.yml".format(name)))
            for name in ("m1", "m4", "m8")
        }
        expected_counts = {"m1": 1, "m4": 4, "m8": 8}
        normalized = {}
        for name, cfg in configs.items():
            self.assertEqual(
                int(cfg.STAGE3E0.MODEL.NUM_EVIDENCE_TOKENS),
                expected_counts[name],
            )
            value = deepcopy(dict(cfg))
            value["STAGE3E0"]["MODEL"][
                "NUM_EVIDENCE_TOKENS"] = 0
            value["STAGE3E0"]["OUTPUT_DIR"] = ""
            normalized[name] = value
        self.assertEqual(normalized["m1"], normalized["m4"])
        self.assertEqual(normalized["m1"], normalized["m8"])

    @staticmethod
    def _summary(token_count, branch_ap):
        category = lambda value: {
            "branch_ap": value,
        }
        metrics = {
            "branch_ap": branch_ap,
            "slot_ap": branch_ap - 0.01,
            "thresholded": {
                "endpoint_error_mean_pixels": 4.0,
                "direction_error_mean_degrees": 5.0,
                "exact_branch_count_accuracy": 0.7,
            },
            "by_category": {
                "ordinary": category(branch_ap),
                "t_junction": category(branch_ap - 0.1),
                "multi_branch": category(branch_ap - 0.05),
            },
        }
        diagnostics = {
            "pairwise_cosine_similarity": {
                "mean": None if token_count == 1 else 0.999},
            "fragment_attention_pairwise_cosine_similarity": {
                "mean": None if token_count == 1 else 0.999},
            "normalized_fragment_attention_entropy": {
                "mean": 0.9},
            "fragment_attention_top8_jaccard": {
                "mean": None if token_count == 1 else 0.99},
        }
        cache_split = {
            "fragment_tokens_sha256": "fragment",
            "fragment_mask_sha256": "mask",
            "sample_ids_sha256": "samples",
        }
        return {
            "num_evidence_tokens": token_count,
            "best_epoch": 2,
            "best_checkpoint": "m{}.pth".format(token_count),
            "seed": 7,
            "e4_checkpoint_sha256": "e4",
            "initial_shared_evidence_state_sha256": "shared",
            "elapsed_seconds": 1.0,
            "peak_cuda_memory_bytes": 2,
            "cache": {
                "train": dict(cache_split),
                "val": dict(cache_split),
            },
            "artifacts": {},
            "best_validation": {
                "variants": {
                    "trajectory_evidence": metrics,
                    "image_graph": {"branch_ap": 0.8},
                },
                "trajectory_evidence_minus_no_trajectory_branch_ap":
                    branch_ap - 0.8,
                "trajectory_evidence_diagnostics": diagnostics,
            },
        }

    def test_comparison_verifies_controls_and_global_pooling_case(self):
        summaries = {
            "m1": self._summary(1, 0.9000),
            "m4": self._summary(4, 0.9005),
            "m8": self._summary(8, 0.9004),
        }

        def load(path):
            return summaries[path.parent.name]

        with mock.patch(
                "scripts.summarize_stage3e1a._load_json",
                side_effect=load):
            report = build_stage3e1a_comparison(
                Path("unused"), equivalence_tolerance=0.001)
        self.assertTrue(report["controls_identical"])
        self.assertTrue(
            report["decision"]["one_approximately_four"])
        self.assertTrue(report["decision"]["m4_collapsed"])
        self.assertIn(
            "global trajectory evidence",
            report["decision"]["conclusion"],
        )


class Stage3E2ACapacityTests(unittest.TestCase):

    def test_capacity_configs_only_change_aggregator_count_and_output(self):
        paths = {
            "mean": "stage3e2a_mean.yml",
            "attention_m1": "stage3e2a_attention_m1.yml",
            "latent_m4": "stage3e2a_latent_m4.yml",
        }
        configs = {
            name: _load_config(Path("configs") / filename)
            for name, filename in paths.items()
        }
        expected = {
            "mean": ("masked_mean", 1),
            "attention_m1": ("latent_attention", 1),
            "latent_m4": ("latent_attention", 4),
        }
        normalized = {}
        for name, cfg in configs.items():
            mode, count = expected[name]
            self.assertEqual(
                cfg.STAGE3E0.MODEL.AGGREGATION_MODE, mode)
            self.assertEqual(
                int(cfg.STAGE3E0.MODEL.NUM_EVIDENCE_TOKENS),
                count,
            )
            value = deepcopy(dict(cfg))
            value["STAGE3E0"]["MODEL"]["AGGREGATION_MODE"] = ""
            value["STAGE3E0"]["MODEL"]["NUM_EVIDENCE_TOKENS"] = 0
            value["STAGE3E0"]["OUTPUT_DIR"] = ""
            normalized[name] = value
        self.assertEqual(
            normalized["mean"], normalized["attention_m1"])
        self.assertEqual(
            normalized["mean"], normalized["latent_m4"])

    @staticmethod
    def _summary(
        *,
        mode,
        token_count,
        branch_ap,
        parameter_count,
    ):
        thresholded = {
            "endpoint_error_mean_pixels": 4.0,
            "direction_error_mean_degrees": 5.0,
            "exact_branch_count_accuracy": 0.7,
            "missed_branch_rate": 0.1,
            "extra_branch_rate": 0.2,
        }

        def group(value, sample_count=1):
            return {
                "sample_count": sample_count,
                "gt_branch_count": sample_count,
                "branch_ap": value,
                "slot_ap": value,
                "thresholded": dict(thresholded),
            }

        metrics = {
            "branch_ap": branch_ap,
            "slot_ap": branch_ap - 0.01,
            "thresholded": dict(thresholded),
            "by_category": {
                "ordinary": group(branch_ap),
                "t_junction": group(branch_ap - 0.1),
                "multi_branch": group(branch_ap - 0.05),
            },
            "by_gt_count": {
                "count_0": group(0.0),
                "count_1": group(branch_ap),
                "count_2": group(branch_ap - 0.1),
                "count_ge3": group(branch_ap - 0.05),
            },
        }
        diagnostics = {
            "pairwise_cosine_similarity": {
                "mean": 0.999 if token_count > 1 else None},
            "fragment_attention_pairwise_cosine_similarity": {
                "mean": 0.999 if token_count > 1 else None},
            "normalized_fragment_attention_entropy": {
                "mean": 1.0 if mode == "masked_mean" else 0.9},
            "fragment_attention_top8_jaccard": {
                "mean": 0.99 if token_count > 1 else None},
            "hidden_norm": {"mean": 8.0},
        }
        cache_split = {
            "fragment_tokens_sha256": "fragment",
            "fragment_mask_sha256": "mask",
            "sample_ids_sha256": "samples",
        }
        parameter_free = mode == "masked_mean"
        return {
            "evidence_aggregation_mode": mode,
            "num_evidence_tokens": token_count,
            "trainable_parameter_count": parameter_count,
            "parameter_free_evidence_module": parameter_free,
            "best_epoch": 0 if parameter_free else 2,
            "best_checkpoint": "best.pth",
            "seed": 7,
            "e4_checkpoint_sha256": "e4",
            "initial_shared_evidence_state_sha256": (
                "empty" if parameter_free else "shared"),
            "elapsed_seconds": 1.0,
            "peak_cuda_memory_bytes": 2,
            "cache": {
                "train": dict(cache_split),
                "val": dict(cache_split),
            },
            "artifacts": {},
            "best_validation": {
                "variants": {
                    "trajectory_evidence": metrics,
                    "image_graph": {"branch_ap": 0.8},
                },
                "trajectory_evidence_minus_no_trajectory_branch_ap":
                    branch_ap - 0.8,
                "trajectory_evidence_diagnostics": diagnostics,
            },
        }

    def test_capacity_comparison_detects_global_mean_equivalence(self):
        summaries = {
            "mean": self._summary(
                mode="masked_mean",
                token_count=1,
                branch_ap=0.9000,
                parameter_count=0,
            ),
            "attention_m1": self._summary(
                mode="latent_attention",
                token_count=1,
                branch_ap=0.9005,
                parameter_count=100,
            ),
            "latent_m4": self._summary(
                mode="latent_attention",
                token_count=4,
                branch_ap=0.9004,
                parameter_count=196,
            ),
        }

        def load(path):
            return summaries[path.parent.name]

        with mock.patch(
                "scripts.summarize_stage3e2a._load_json",
                side_effect=load):
            report = build_stage3e2a_comparison(
                Path("unused"), equivalence_tolerance=0.001)
        self.assertTrue(report["controls_identical"])
        self.assertTrue(
            report["attention_shared_initialization_identical"])
        self.assertTrue(report["decision"]["all_equivalent"])
        self.assertTrue(report["decision"]["m4_collapsed"])
        self.assertIn(
            "global trajectory aggregation",
            report["decision"]["conclusion"],
        )


if __name__ == "__main__":
    unittest.main()
