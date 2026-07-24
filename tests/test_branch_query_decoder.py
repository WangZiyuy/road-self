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


if __name__ == "__main__":
    unittest.main()
