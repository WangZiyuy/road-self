import copy
import json
import shutil
import unittest
import uuid
from pathlib import Path

import torch
from easydict import EasyDict

from model.branch_query_decoder import MultiModalBranchQueryDecoder
from model.support_guided_trajectory_fusion import (
    FUSION_ORIGINAL_ATTENTION,
    FUSION_SUPPORT_AGGREGATION,
    SupportGuidedTrajectoryFusion,
    forward_branch_with_trajectory_fusion,
    resolve_trajectory_fusion_mode,
)
from scripts.summarize_stage3d_c1 import build_comparison
from train_support_fusion import (
    TRAINING_STAGE_A,
    TRAINING_STAGE_B,
    _configure_trainable_modules,
    _forward_frozen_c1a,
)
from utils.stage3d_c1_checkpoint import (
    build_stage3d_c1_checkpoint_payload,
    load_stage3d_c1_checkpoint,
    save_stage3d_c1_checkpoint,
)


class Stage3DC1FusionTests(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(20260726)
        self.decoder = MultiModalBranchQueryDecoder(
            image_channels=16,
            trajectory_dim=32,
            hidden_dim=32,
            num_queries=4,
            num_heads=4,
            image_pool_size=4,
            dropout=0.0,
            query_self_attention_layers=1,
        ).eval()
        self.fusion = SupportGuidedTrajectoryFusion(
            hidden_dim=32,
            branch_input_dim=64,
            projection_dim=32,
        ).eval()
        self.fusion.initialize_aggregation_from_decoder(self.decoder)
        self.inputs = {
            "stage_fuse": torch.randn(2, 16, 8, 8),
            "state_token": torch.randn(2, 32),
            "fragment_tokens": torch.randn(2, 5, 32),
            "fragment_mask": torch.tensor([
                [True, True, True, False, False],
                [False, False, False, False, False],
            ]),
            "walked_path": torch.randn(2, 1, 8, 8),
            "sample_ids": torch.tensor([7, 11]),
        }

    def _forward(self, mode, **overrides):
        values = dict(self.inputs)
        values.update(overrides)
        return forward_branch_with_trajectory_fusion(
            branch_decoder=self.decoder,
            fusion_module=(
                self.fusion
                if mode == FUSION_SUPPORT_AGGREGATION
                else None
            ),
            fusion_mode=mode,
            return_attention=True,
            return_debug_states=True,
            **values,
        )

    def test_default_mode_is_strict_original_attention(self):
        self.assertEqual(
            resolve_trajectory_fusion_mode(EasyDict()),
            FUSION_ORIGINAL_ATTENTION,
        )
        with self.assertRaisesRegex(ValueError, "unknown"):
            resolve_trajectory_fusion_mode(
                EasyDict({"TRAJ_FUSION_MODE": "guess"}))

    def test_original_mode_exactly_reuses_e4_forward(self):
        before = copy.deepcopy(self.decoder.state_dict())
        direct = self.decoder(
            self.inputs["stage_fuse"],
            self.inputs["state_token"],
            self.inputs["fragment_tokens"],
            self.inputs["fragment_mask"],
            walked_path=self.inputs["walked_path"],
            return_attention=True,
            return_debug_states=True,
        )
        routed = self._forward(FUSION_ORIGINAL_ATTENTION)
        for key in (
                "branch_exist_logits",
                "branch_offsets_norm",
                "branch_directions",
                "branch_tokens",
                "trajectory_attention_weights"):
            torch.testing.assert_close(
                routed[key], direct[key], atol=0.0, rtol=0.0)
        for key, value in before.items():
            torch.testing.assert_close(
                self.decoder.state_dict()[key],
                value,
                atol=0.0,
                rtol=0.0,
            )

    def test_aggregation_projection_starts_from_e4_values(self):
        attention = self.decoder.trajectory_cross_attention
        hidden_dim = attention.embed_dim
        torch.testing.assert_close(
            self.fusion.value_projection.weight,
            attention.in_proj_weight[2 * hidden_dim:3 * hidden_dim],
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            self.fusion.output_projection.weight,
            attention.out_proj.weight,
            atol=0.0,
            rtol=0.0,
        )

    def test_support_forward_is_finite_with_empty_trajectory_sample(self):
        output = self._forward(FUSION_SUPPORT_AGGREGATION)
        self.assertEqual(
            tuple(output["fragment_support_logits"].shape), (2, 4, 5))
        self.assertEqual(
            tuple(output["trajectory_context"].shape), (2, 4, 32))
        for key in (
                "branch_exist_logits",
                "branch_offsets_norm",
                "branch_directions",
                "fragment_support_logits",
                "trajectory_context"):
            self.assertTrue(torch.isfinite(output[key]).all())
        self.assertTrue(torch.equal(
            output["trajectory_context"][1],
            torch.zeros_like(output["trajectory_context"][1]),
        ))

    def test_pre_trajectory_branch_identity_is_non_circular(self):
        first = self._forward(FUSION_SUPPORT_AGGREGATION)
        second = self._forward(
            FUSION_SUPPORT_AGGREGATION,
            fragment_tokens=self.inputs["fragment_tokens"] * -13.0,
        )
        torch.testing.assert_close(
            first["debug_pre_trajectory_branch_tokens"],
            second["debug_pre_trajectory_branch_tokens"],
            atol=0.0,
            rtol=0.0,
        )
        self.assertFalse(torch.allclose(
            first["fragment_support_logits"],
            second["fragment_support_logits"],
        ))

    def test_topk_and_random_aggregation_are_explicit_and_deterministic(self):
        topk = self._forward(
            FUSION_SUPPORT_AGGREGATION, top_k=2)
        self.assertEqual(
            topk["trajectory_selection_mask"][0].sum(dim=-1).tolist(),
            [2, 2, 2, 2],
        )
        first = self._forward(
            FUSION_SUPPORT_AGGREGATION,
            randomize_fragment_values=True,
            random_seed=23,
        )
        second = self._forward(
            FUSION_SUPPORT_AGGREGATION,
            randomize_fragment_values=True,
            random_seed=23,
        )
        torch.testing.assert_close(
            first["trajectory_context"],
            second["trajectory_context"],
        )

    def test_branch_gradient_reaches_new_fusion_only(self):
        self.decoder.requires_grad_(False)
        self.fusion.train().requires_grad_(True)
        output = self._forward(FUSION_SUPPORT_AGGREGATION)
        loss = (
            output["branch_exist_logits"].sum()
            + output["branch_offsets_norm"].sum()
        )
        loss.backward()
        self.assertFalse(any(
            parameter.grad is not None
            for parameter in self.decoder.parameters()))
        for module in (
                self.fusion.support_head,
                self.fusion.value_projection,
                self.fusion.output_projection):
            self.assertTrue(any(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all())
                for parameter in module.parameters()))

    def test_frozen_c1a_cache_path_matches_dynamic_frozen_path(self):
        dynamic = self._forward(FUSION_SUPPORT_AGGREGATION)
        cached_batch = {
            "pre_trajectory_branch_tokens": dynamic[
                "debug_pre_trajectory_branch_tokens"],
            "fragment_tokens": self.inputs["fragment_tokens"],
            "sample_ids": self.inputs["sample_ids"],
            "graph_conditioned_queries": dynamic[
                "debug_graph_conditioned_queries"],
            "image_cross_attention_context": dynamic[
                "debug_image_cross_attention_output"],
            "graph_state_contribution": dynamic[
                "debug_graph_state_contribution"],
        }
        cached = _forward_frozen_c1a(
            fusion_module=self.fusion,
            branch_decoder=self.decoder,
            batch=cached_batch,
            fragment_mask=self.inputs["fragment_mask"],
            top_k=None,
            randomize=False,
            random_seed=0,
            epsilon=1e-6,
        )
        for key in (
                "branch_exist_logits",
                "branch_offsets_norm",
                "branch_directions",
                "branch_tokens",
                "fragment_support_logits",
                "trajectory_context"):
            torch.testing.assert_close(
                cached[key], dynamic[key], atol=0.0, rtol=0.0)


class Stage3DC1TrainabilityAndCheckpointTests(unittest.TestCase):

    def _modules(self):
        trajectory = torch.nn.Sequential(
            torch.nn.Linear(3, 5), torch.nn.GELU())
        graph = torch.nn.Linear(2, 5)
        decoder = torch.nn.Linear(5, 2)
        fusion = SupportGuidedTrajectoryFusion(
            hidden_dim=8,
            branch_input_dim=16,
            projection_dim=8,
        )
        return trajectory, graph, decoder, fusion

    def test_c1a_and_c1b_trainability_boundaries(self):
        trajectory, graph, decoder, fusion = self._modules()
        parameters_a = _configure_trainable_modules(
            training_stage=TRAINING_STAGE_A,
            trajectory_encoder=trajectory,
            graph_state_encoder=graph,
            branch_decoder=decoder,
            fusion_module=fusion,
        )
        self.assertFalse(any(
            value.requires_grad for value in trajectory.parameters()))
        self.assertFalse(any(
            value.requires_grad for value in graph.parameters()))
        self.assertFalse(any(
            value.requires_grad for value in decoder.parameters()))
        self.assertEqual(
            {id(value) for value in parameters_a},
            {
                id(value)
                for value in fusion.parameters()
                if value.requires_grad
            },
        )
        parameters_b = _configure_trainable_modules(
            training_stage=TRAINING_STAGE_B,
            trajectory_encoder=trajectory,
            graph_state_encoder=graph,
            branch_decoder=decoder,
            fusion_module=fusion,
        )
        self.assertTrue(all(
            value.requires_grad for value in trajectory.parameters()))
        self.assertFalse(any(
            value.requires_grad for value in graph.parameters()))
        self.assertFalse(any(
            value.requires_grad for value in decoder.parameters()))
        self.assertEqual(
            len(parameters_b),
            sum(1 for value in trajectory.parameters())
            + sum(1 for value in fusion.parameters()),
        )

    def test_checkpoint_round_trip_is_strict_and_sha_guarded(self):
        trajectory, _, _, fusion = self._modules()
        optimizer = torch.optim.AdamW(
            list(trajectory.parameters()) + list(fusion.parameters()),
            lr=1e-3,
        )
        payload = build_stage3d_c1_checkpoint_payload(
            fusion_module=fusion,
            trajectory_encoder=trajectory,
            optimizer=optimizer,
            epoch=3,
            training_stage="c1_a",
            fusion_mode="support_aggregation",
            support_top_k=8,
            random_fragment_aggregation=False,
            e4_checkpoint="e4.pth.tar",
            e4_checkpoint_sha256="abc123",
            metrics={"branch_ap": 0.9},
            config={"name": "unit"},
        )
        path = Path("tests") / (
            "_stage3d_c1_{}.pth.tar".format(uuid.uuid4().hex))
        try:
            save_stage3d_c1_checkpoint(path, payload)
            original = copy.deepcopy(fusion.state_dict())
            with torch.no_grad():
                for parameter in fusion.parameters():
                    parameter.zero_()
            loaded = load_stage3d_c1_checkpoint(
                path,
                fusion_module=fusion,
                trajectory_encoder=trajectory,
                optimizer=optimizer,
                expected_e4_sha256="abc123",
            )
            self.assertEqual(loaded["epoch"], 3)
            for key, value in original.items():
                torch.testing.assert_close(
                    fusion.state_dict()[key], value)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_stage3d_c1_checkpoint(
                    path,
                    fusion_module=fusion,
                    expected_e4_sha256="different",
                )
        finally:
            path.unlink(missing_ok=True)


class Stage3DC1ComparisonTests(unittest.TestCase):

    @staticmethod
    def _variant(branch_ap, no_trajectory_ap):
        category = {
            "branch_ap": branch_ap,
            "slot_ap": branch_ap,
            "thresholded": {},
            "oracle_k": {},
        }
        return {
            "full": {
                "branch_ap": branch_ap,
                "slot_ap": branch_ap - 0.1,
                "thresholded": {
                    "endpoint_error_mean_pixels": 10.0,
                    "direction_error_mean_degrees": 20.0,
                    "exact_branch_count_accuracy": 0.5,
                },
                "oracle_k": {
                    "recall": 0.8,
                    "distinct_gt_coverage": 0.75,
                    "duplicates": {
                        "duplicate_pair_ratio": 0.1,
                    },
                },
                "by_category": {
                    "ordinary": category,
                    "t_junction": category,
                    "multi_branch": category,
                },
            },
            "no_trajectory": {"branch_ap": no_trajectory_ap},
            "full_minus_no_trajectory_branch_ap":
                branch_ap - no_trajectory_ap,
            "support_selection": {"support_ap": 0.8},
            "support_loss": 0.2,
            "elapsed_seconds": 1.0,
        }

    def test_comparison_gate_selects_best_nonrandom_c1a(self):
        root = Path("tests") / (
            "_stage3d_c1_comparison_{}".format(uuid.uuid4().hex))
        root.mkdir()
        try:
            values = {
                "a_original": (0.80, 0.75),
                "b_support": (0.81, 0.75),
                "c_topk8": (0.83, 0.75),
                "d_topk16": (0.82, 0.75),
                "e_random": (0.79, 0.75),
            }
            for folder, (branch_ap, no_ap) in values.items():
                output = root / folder
                output.mkdir()
                report = self._variant(branch_ap, no_ap)
                if folder == "a_original":
                    path = output / "evaluation.json"
                    value = report
                else:
                    path = output / "training_summary.json"
                    value = {
                        "best_validation": report,
                        "best_checkpoint": str(output / "best.pth.tar"),
                        "best_epoch": 4,
                        "elapsed_seconds": 2.0,
                        "peak_cuda_memory_bytes": 123,
                        "training_stage": "c1_a",
                    }
                with path.open("w", encoding="utf-8") as output_file:
                    json.dump(value, output_file)
            comparison = build_comparison(
                root,
                minimum_branch_ap_gain=0.001,
                minimum_modality_gain=0.0,
            )
            self.assertEqual(
                comparison["decision"]["best_support_variant"],
                "support_topk_8",
            )
            self.assertTrue(comparison["decision"]["run_c1_b"])
        finally:
            shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
