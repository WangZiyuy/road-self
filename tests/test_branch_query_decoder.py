import unittest

import torch

from model.branch_query_decoder import MultiModalBranchQueryDecoder
from model.graph_state_encoder import GraphStateEncoder


def _batched_graph_state(batch_size=2, edge_count=3):
    state = {
        "incoming_dir": torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32
        )[:batch_size],
        "incoming_valid": torch.tensor(
            [True, False], dtype=torch.bool)[:batch_size],
        "explored_edge_dirs": torch.zeros(
            (batch_size, edge_count, 2), dtype=torch.float32),
        "explored_edge_mask": torch.zeros(
            (batch_size, edge_count), dtype=torch.bool),
        "explored_is_incoming": torch.zeros(
            (batch_size, edge_count), dtype=torch.bool),
        "is_key_point": torch.tensor(
            [False, True], dtype=torch.bool)[:batch_size],
    }
    if edge_count:
        state["explored_edge_dirs"][:, 0] = torch.tensor([1.0, 0.0])
        state["explored_edge_mask"][:, 0] = True
    return state


class MultiModalBranchQueryDecoderTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260724)
        self.decoder = MultiModalBranchQueryDecoder(
            image_channels=16,
            trajectory_dim=32,
            hidden_dim=32,
            num_queries=6,
            num_heads=4,
            image_pool_size=4,
            dropout=0.0,
        ).eval()

    def _inputs(self):
        stage_fuse = torch.randn(2, 16, 9, 11)
        state_token = torch.randn(2, 32)
        fragment_tokens = torch.randn(2, 5, 32)
        fragment_mask = torch.tensor([
            [True, True, True, False, False],
            [True, False, False, False, False],
        ])
        return (
            stage_fuse,
            state_token,
            fragment_tokens,
            fragment_mask,
        )

    def test_output_shapes(self):
        with torch.no_grad():
            outputs = self.decoder(*self._inputs(), return_attention=True)
        self.assertEqual(
            tuple(outputs["branch_exist_logits"].shape), (2, 6))
        self.assertEqual(
            tuple(outputs["branch_offsets_norm"].shape), (2, 6, 2))
        self.assertEqual(
            tuple(outputs["branch_directions"].shape), (2, 6, 2))
        self.assertEqual(
            tuple(outputs["image_attention_weights"].shape), (2, 6, 16))
        self.assertEqual(
            tuple(outputs["trajectory_attention_weights"].shape),
            (2, 6, 5),
        )
        for value in outputs.values():
            self.assertTrue(torch.isfinite(value).all())
        direction_norm = torch.linalg.vector_norm(
            outputs["branch_directions"], dim=-1)
        nonzero_offset = torch.linalg.vector_norm(
            outputs["branch_offsets_norm"], dim=-1) > 1e-6
        torch.testing.assert_close(
            direction_norm[nonzero_offset],
            torch.ones_like(direction_norm[nonzero_offset]),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_no_trajectory_sample_is_finite(self):
        stage_fuse, state_token, _, _ = self._inputs()
        empty_tokens = torch.zeros(2, 0, 32)
        empty_mask = torch.zeros(2, 0, dtype=torch.bool)
        with torch.no_grad():
            outputs = self.decoder(
                stage_fuse,
                state_token,
                empty_tokens,
                empty_mask,
                return_attention=True,
            )
        for value in outputs.values():
            self.assertTrue(torch.isfinite(value).all())
        self.assertEqual(
            tuple(outputs["trajectory_attention_weights"].shape),
            (2, 6, 0),
        )

    def test_implicit_empty_trajectory_supports_projection(self):
        decoder = MultiModalBranchQueryDecoder(
            image_channels=16,
            trajectory_dim=24,
            hidden_dim=32,
            num_queries=3,
            num_heads=4,
            image_pool_size=4,
            dropout=0.0,
        ).eval()
        stage_fuse, state_token, _, _ = self._inputs()
        with torch.no_grad():
            outputs = decoder(
                stage_fuse, state_token,
                fragment_tokens=None, fragment_mask=None)
        self.assertEqual(
            tuple(outputs["branch_exist_logits"].shape), (2, 3))
        self.assertTrue(all(
            torch.isfinite(value).all()
            for value in outputs.values()
        ))

    def test_fragment_permutation_does_not_change_prediction(self):
        inputs = self._inputs()
        permutation = torch.tensor([2, 4, 0, 3, 1])
        with torch.no_grad():
            original = self.decoder(*inputs)
            permuted = self.decoder(
                inputs[0],
                inputs[1],
                inputs[2].index_select(1, permutation),
                inputs[3].index_select(1, permutation),
            )
        for key in (
                "branch_exist_logits",
                "branch_offsets_norm",
                "branch_directions"):
            torch.testing.assert_close(
                original[key], permuted[key], rtol=1e-5, atol=1e-6)

    def test_padding_fragment_does_not_change_prediction(self):
        inputs = self._inputs()
        padding = torch.randn(2, 3, 32) * 1000.0
        padded_tokens = torch.cat((inputs[2], padding), dim=1)
        padded_mask = torch.cat(
            (inputs[3], torch.zeros(2, 3, dtype=torch.bool)), dim=1)
        with torch.no_grad():
            original = self.decoder(*inputs)
            padded = self.decoder(
                inputs[0], inputs[1], padded_tokens, padded_mask)
        for key in (
                "branch_exist_logits",
                "branch_offsets_norm",
                "branch_directions"):
            torch.testing.assert_close(
                original[key], padded[key], rtol=1e-5, atol=1e-6)

    def test_none_and_zero_walked_path_are_equivalent(self):
        inputs = self._inputs()
        zero_walked_path = torch.zeros(2, 1, 9, 11)
        with torch.no_grad():
            none_output = self.decoder(
                *inputs, walked_path=None)
            zero_output = self.decoder(
                *inputs, walked_path=zero_walked_path)
        for key in (
                "branch_exist_logits",
                "branch_offsets_norm",
                "branch_directions"):
            torch.testing.assert_close(
                none_output[key],
                zero_output[key],
                rtol=0.0,
                atol=0.0,
            )

    def test_nonzero_walked_path_changes_prediction(self):
        inputs = self._inputs()
        zero_walked_path = torch.zeros(2, 1, 9, 11)
        nonzero_walked_path = zero_walked_path.clone()
        nonzero_walked_path[:, :, 2:7, 5] = 1.0
        with torch.no_grad():
            zero_output = self.decoder(
                *inputs, walked_path=zero_walked_path)
            nonzero_output = self.decoder(
                *inputs, walked_path=nonzero_walked_path)
        difference = (
            zero_output["branch_exist_logits"]
            - nonzero_output["branch_exist_logits"]
        ).abs().max()
        self.assertGreater(float(difference), 1e-7)

    def test_modality_ablation_outputs_are_finite(self):
        inputs = self._inputs()
        walked_path = torch.randn(2, 1, 9, 11)
        with torch.no_grad():
            full = self.decoder(
                *inputs, walked_path=walked_path)
            no_trajectory = self.decoder(
                inputs[0],
                inputs[1],
                torch.zeros(2, 0, 32),
                torch.zeros(2, 0, dtype=torch.bool),
                walked_path=walked_path,
            )
            trajectory_graph = self.decoder(
                *inputs,
                walked_path=walked_path,
                image_available=torch.zeros(2, dtype=torch.bool),
            )
        for output in (full, no_trajectory, trajectory_graph):
            self.assertTrue(all(
                torch.isfinite(value).all()
                for value in output.values()
            ))

    def test_image_graph_and_trajectory_receive_finite_gradients(self):
        graph_encoder = GraphStateEncoder(hidden_dim=32).eval()
        graph_state = _batched_graph_state(batch_size=2, edge_count=2)
        graph_state["incoming_dir"].requires_grad_(True)
        graph_state["explored_edge_dirs"].requires_grad_(True)
        state_token = graph_encoder(graph_state)

        stage_fuse = torch.randn(
            2, 16, 8, 8, requires_grad=True)
        fragment_tokens = torch.randn(
            2, 4, 32, requires_grad=True)
        fragment_mask = torch.tensor([
            [True, True, False, False],
            [True, True, True, False],
        ])
        outputs = self.decoder(
            stage_fuse,
            state_token,
            fragment_tokens,
            fragment_mask,
        )
        loss = (
            outputs["branch_exist_logits"].square().mean()
            + outputs["branch_offsets_norm"].square().mean()
        )
        loss.backward()
        for tensor in (
                stage_fuse,
                fragment_tokens,
                graph_state["incoming_dir"],
                graph_state["explored_edge_dirs"]):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(float(tensor.grad.abs().sum()), 0.0)

    def test_empty_graph_state_can_run(self):
        graph_encoder = GraphStateEncoder(hidden_dim=32).eval()
        state_token = graph_encoder(
            _batched_graph_state(batch_size=2, edge_count=0))
        stage_fuse, _, fragment_tokens, fragment_mask = self._inputs()
        with torch.no_grad():
            outputs = self.decoder(
                stage_fuse,
                state_token,
                fragment_tokens,
                fragment_mask,
            )
        self.assertTrue(all(
            torch.isfinite(value).all()
            for value in outputs.values()
        ))

    def test_debug_query_stages_have_expected_shape(self):
        with torch.no_grad():
            outputs = self.decoder(
                *self._inputs(), return_debug_states=True)
        for key in (
                "debug_learned_query_embedding",
                "debug_pre_graph_queries",
                "debug_graph_conditioned_queries",
                "debug_pre_cross_attention_queries",
                "debug_image_cross_attention_output",
                "debug_trajectory_cross_attention_output",
                "debug_final_fused_queries",
                "debug_graph_state_contribution"):
            self.assertEqual(tuple(outputs[key].shape), (2, 6, 32))
            self.assertTrue(torch.isfinite(outputs[key]).all())

    def test_zero_self_attention_checkpoint_is_strict_compatible(self):
        state_dict = self.decoder.state_dict()
        restored = MultiModalBranchQueryDecoder(
            image_channels=16,
            trajectory_dim=32,
            hidden_dim=32,
            num_queries=6,
            num_heads=4,
            image_pool_size=4,
            dropout=0.0,
            query_self_attention_layers=0,
        )
        restored.load_state_dict(state_dict, strict=True)
        self.assertFalse(any(
            key.startswith("query_self_attention")
            for key in restored.state_dict()
        ))

    def test_one_self_attention_layer_is_optional_and_finite(self):
        torch.manual_seed(20260724)
        decoder = MultiModalBranchQueryDecoder(
            image_channels=16,
            trajectory_dim=32,
            hidden_dim=32,
            num_queries=6,
            num_heads=4,
            image_pool_size=4,
            dropout=0.0,
            query_self_attention_layers=1,
        ).eval()
        with torch.no_grad():
            outputs = decoder(
                *self._inputs(), return_debug_states=True)
        self.assertTrue(torch.isfinite(
            outputs["branch_exist_logits"]).all())
        self.assertTrue(any(
            key.startswith("query_self_attention.")
            for key in decoder.state_dict()
        ))

    def test_enabling_self_attention_preserves_shared_initialization(self):
        arguments = {
            "image_channels": 16,
            "trajectory_dim": 32,
            "hidden_dim": 32,
            "num_queries": 6,
            "num_heads": 4,
            "image_pool_size": 4,
            "dropout": 0.0,
        }
        torch.manual_seed(20260724)
        legacy = MultiModalBranchQueryDecoder(
            **arguments, query_self_attention_layers=0)
        torch.manual_seed(20260724)
        self_attention = MultiModalBranchQueryDecoder(
            **arguments, query_self_attention_layers=1)
        legacy_state = legacy.state_dict()
        self_attention_state = self_attention.state_dict()
        self.assertTrue(all(
            torch.equal(value, self_attention_state[key])
            for key, value in legacy_state.items()
        ))


if __name__ == "__main__":
    unittest.main()
