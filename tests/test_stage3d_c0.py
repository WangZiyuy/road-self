import unittest

import torch
import torch.nn.functional as F
from easydict import EasyDict

from model.branch_query_decoder import MultiModalBranchQueryDecoder
from scripts.evaluate_stage3d_c0_support_aggregation import (
    assess_stage3d_c0,
)
from utils.trajectory_support_aggregation import (
    decoder_fragment_values,
    freeze_modules,
    permute_valid_fragment_values,
    recompute_branch_predictions,
    support_weighted_trajectory_context,
)


class FrozenSupportAggregationTests(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(20260726)
        self.decoder = MultiModalBranchQueryDecoder(
            image_channels=128,
            trajectory_dim=128,
            hidden_dim=128,
            num_queries=3,
            num_heads=4,
            image_pool_size=4,
            dropout=0.0,
            query_self_attention_layers=1,
        ).eval()

    def _forward(self, fragment_mask):
        batch_size, fragment_count = fragment_mask.shape
        stage_fuse = torch.randn(batch_size, 128, 8, 8)
        state_token = torch.randn(batch_size, 128)
        fragment_tokens = torch.randn(
            batch_size, fragment_count, 128)
        walked_path = torch.randn(batch_size, 1, 8, 8)
        output = self.decoder(
            stage_fuse=stage_fuse,
            state_token=state_token,
            fragment_tokens=fragment_tokens,
            fragment_mask=fragment_mask,
            walked_path=walked_path,
            return_attention=True,
            return_debug_states=True,
        )
        return (
            output,
            stage_fuse,
            state_token,
            fragment_tokens,
            walked_path,
        )

    def test_original_context_recompute_is_exact(self):
        mask = torch.tensor([
            [True, True, False, True],
            [False, False, False, False],
        ])
        output, _, _, _, _ = self._forward(mask)
        recomputed = recompute_branch_predictions(
            self.decoder,
            graph_conditioned_queries=output[
                "debug_graph_conditioned_queries"],
            image_context=output[
                "debug_image_cross_attention_output"],
            trajectory_context=output[
                "debug_trajectory_cross_attention_output"],
            graph_state_contribution=output[
                "debug_graph_state_contribution"],
        )
        for key in (
                "branch_exist_logits",
                "branch_offsets_norm",
                "branch_directions",
                "branch_tokens"):
            torch.testing.assert_close(
                recomputed[key], output[key], atol=0.0, rtol=0.0)

    def test_zero_context_recompute_matches_no_trajectory_decoder(self):
        mask = torch.tensor([[True, True, False, True]])
        (
            output,
            stage_fuse,
            state_token,
            fragment_tokens,
            walked_path,
        ) = self._forward(mask)
        no_trajectory = self.decoder(
            stage_fuse=stage_fuse,
            state_token=state_token,
            fragment_tokens=fragment_tokens,
            fragment_mask=torch.zeros_like(mask),
            walked_path=walked_path,
        )
        recomputed = recompute_branch_predictions(
            self.decoder,
            graph_conditioned_queries=output[
                "debug_graph_conditioned_queries"],
            image_context=output[
                "debug_image_cross_attention_output"],
            trajectory_context=torch.zeros_like(
                output["debug_trajectory_cross_attention_output"]),
            graph_state_contribution=output[
                "debug_graph_state_contribution"],
        )
        for key in (
                "branch_exist_logits",
                "branch_offsets_norm",
                "branch_directions"):
            torch.testing.assert_close(
                recomputed[key],
                no_trajectory[key],
                atol=0.0,
                rtol=0.0,
            )

    def test_decoder_fragment_values_match_attention_representation(self):
        tokens = torch.randn(2, 4, 128)
        mask = torch.tensor([
            [True, False, True, False],
            [False, False, False, False],
        ])
        stage_fuse = torch.randn(2, 128, 8, 8)
        state_token = torch.randn(2, 128)
        outputs = self.decoder(
            stage_fuse,
            state_token,
            tokens,
            mask,
            return_attention=True,
            return_debug_states=True,
        )
        values = decoder_fragment_values(
            self.decoder, tokens, mask)
        attention = self.decoder.trajectory_cross_attention
        hidden_dim = attention.embed_dim
        head_count = attention.num_heads
        head_dim = hidden_dim // head_count
        normalized = self.decoder.trajectory_norm(
            self.decoder.trajectory_projection(tokens))
        weight = attention.in_proj_weight
        bias = attention.in_proj_bias
        queries = F.linear(
            outputs["debug_graph_conditioned_queries"],
            weight[:hidden_dim],
            bias[:hidden_dim],
        )
        keys = F.linear(
            normalized,
            weight[hidden_dim:2 * hidden_dim],
            bias[hidden_dim:2 * hidden_dim],
        )
        query_heads = queries.reshape(
            2, 3, head_count, head_dim).transpose(1, 2)
        key_heads = keys.reshape(
            2, 4, head_count, head_dim).transpose(1, 2)
        value_heads = values.reshape(
            2, 4, head_count, head_dim).transpose(1, 2)
        scores = torch.matmul(
            query_heads,
            key_heads.transpose(-2, -1),
        ) / (head_dim ** 0.5)
        scores = scores.masked_fill(
            ~mask[:, None, None, :],
            float("-inf"),
        )
        head_context = torch.matmul(
            torch.softmax(scores[0:1], dim=-1),
            value_heads[0:1],
        )
        merged_context = head_context.transpose(1, 2).reshape(
            1, 3, hidden_dim)
        reconstructed = attention.out_proj(merged_context)
        torch.testing.assert_close(
            reconstructed,
            outputs["debug_trajectory_cross_attention_output"][0:1],
            atol=1e-6,
            rtol=1e-6,
        )
        self.assertTrue(torch.equal(
            values[~mask], torch.zeros_like(values[~mask])))

    def test_support_pooling_masks_padding_and_handles_empty_samples(self):
        logits = torch.zeros(2, 2, 3)
        values = torch.tensor([
            [[1.0, 0.0], [3.0, 2.0], [100.0, 100.0]],
            [[5.0, 5.0], [7.0, 7.0], [9.0, 9.0]],
        ])
        mask = torch.tensor([
            [True, True, False],
            [False, False, False],
        ])
        result = support_weighted_trajectory_context(
            logits, values, mask)
        expected = torch.tensor([2.0, 1.0])
        torch.testing.assert_close(result["context"][0, 0], expected)
        torch.testing.assert_close(result["context"][0, 1], expected)
        self.assertTrue(torch.equal(
            result["context"][1],
            torch.zeros_like(result["context"][1]),
        ))
        self.assertTrue(torch.isfinite(result["context"]).all())

    def test_random_permutation_is_deterministic_and_batch_invariant(self):
        values = torch.arange(
            2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
        mask = torch.tensor([
            [True, True, True, False, False],
            [True, True, True, True, False],
        ])
        sample_ids = torch.tensor([11, 42])
        first = permute_valid_fragment_values(
            values, mask, sample_ids, seed=7)
        second = permute_valid_fragment_values(
            values, mask, sample_ids, seed=7)
        torch.testing.assert_close(first, second)
        for batch_index in range(2):
            isolated = permute_valid_fragment_values(
                values[batch_index:batch_index + 1],
                mask[batch_index:batch_index + 1],
                sample_ids[batch_index:batch_index + 1],
                seed=7,
            )
            torch.testing.assert_close(
                first[batch_index:batch_index + 1], isolated)
            valid = mask[batch_index]
            original_rows = {
                tuple(row.tolist())
                for row in values[batch_index, valid]
            }
            permuted_rows = {
                tuple(row.tolist())
                for row in first[batch_index, valid]
            }
            self.assertEqual(original_rows, permuted_rows)
            self.assertTrue(torch.equal(
                first[batch_index, ~valid],
                torch.zeros_like(first[batch_index, ~valid]),
            ))

    def test_freeze_modules_sets_eval_and_no_grad(self):
        self.decoder.train()
        freeze_modules([self.decoder])
        self.assertFalse(self.decoder.training)
        self.assertFalse(any(
            parameter.requires_grad
            for parameter in self.decoder.parameters()))


class Stage3DC0DecisionTests(unittest.TestCase):

    @staticmethod
    def _cfg():
        return EasyDict({
            "STAGE3D_C0": {
                "DECISION": {
                    "MIN_SUPPORT_OVER_ORIGINAL_BRANCH_AP": 0.0,
                    "MIN_FULL_OVER_NO_TRAJECTORY_BRANCH_AP": 0.01,
                    "MIN_HIGH_SUPPORT_SELECTION_AP": 0.72,
                },
            },
        })

    def test_positive_offline_result_can_enter_training_experiment(self):
        result = assess_stage3d_c0(
            original_branch_ap=0.50,
            support_branch_ap=0.53,
            no_trajectory_branch_ap=0.48,
            support_selection_ap=0.80,
            cfg=self._cfg(),
        )
        self.assertTrue(result["can_enter_next_stage_training"])
        self.assertTrue(
            result["post_fusion_support_is_circular_for_online_use"])

    def test_high_support_ap_without_branch_gain_reports_fusion_problem(self):
        result = assess_stage3d_c0(
            original_branch_ap=0.50,
            support_branch_ap=0.49,
            no_trajectory_branch_ap=0.47,
            support_selection_ap=0.80,
            cfg=self._cfg(),
        )
        self.assertFalse(result["can_enter_next_stage_training"])
        self.assertIn("fusion", result["diagnosis"])
        self.assertIn("circular", result["diagnosis"])


if __name__ == "__main__":
    unittest.main()
