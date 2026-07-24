import unittest

import torch

from model.graph_state_encoder import GraphStateEncoder


def _graph_state(batch_size=2, edge_count=3):
    incoming_dir = torch.tensor(
        [[1.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    incoming_valid = torch.tensor([True, False])
    is_key_point = torch.tensor([False, True])
    explored_edge_dirs = torch.zeros(
        (batch_size, edge_count, 2), dtype=torch.float32)
    explored_edge_mask = torch.zeros(
        (batch_size, edge_count), dtype=torch.bool)
    explored_is_incoming = torch.zeros(
        (batch_size, edge_count), dtype=torch.bool)
    if edge_count:
        explored_edge_dirs[0, 0] = torch.tensor([1.0, 0.0])
        explored_edge_mask[0, 0] = True
        explored_is_incoming[0, 0] = True
    return {
        "incoming_dir": incoming_dir[:batch_size],
        "incoming_valid": incoming_valid[:batch_size],
        "explored_edge_dirs": explored_edge_dirs,
        "explored_edge_mask": explored_edge_mask,
        "explored_is_incoming": explored_is_incoming,
        "is_key_point": is_key_point[:batch_size],
    }


class GraphStateEncoderTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260724)
        self.encoder = GraphStateEncoder(hidden_dim=32).eval()

    def test_shape_and_empty_edges_are_finite(self):
        state = _graph_state(edge_count=0)
        with torch.no_grad():
            token = self.encoder(state)
        self.assertEqual(tuple(token.shape), (2, 32))
        self.assertTrue(torch.isfinite(token).all())

    def test_explored_edge_order_does_not_change_token(self):
        state = _graph_state(batch_size=1, edge_count=3)
        state["explored_edge_dirs"][0] = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [-0.7071068, 0.7071068],
        ])
        state["explored_edge_mask"][0] = True
        state["explored_is_incoming"][0] = torch.tensor(
            [True, False, False])
        permutation = torch.tensor([2, 0, 1])
        permuted = {
            key: value.clone() for key, value in state.items()
        }
        for key in (
                "explored_edge_dirs",
                "explored_edge_mask",
                "explored_is_incoming"):
            permuted[key] = permuted[key].index_select(1, permutation)
        with torch.no_grad():
            original_token = self.encoder(state)
            permuted_token = self.encoder(permuted)
        torch.testing.assert_close(
            original_token, permuted_token, rtol=1e-6, atol=1e-6)

    def test_continuous_inputs_receive_finite_gradients(self):
        state = _graph_state(batch_size=1, edge_count=2)
        state["explored_edge_dirs"][0, 1] = torch.tensor([0.6, 0.8])
        state["explored_edge_mask"][0, 1] = True
        state["incoming_dir"].requires_grad_(True)
        state["explored_edge_dirs"].requires_grad_(True)
        self.encoder(state).square().mean().backward()
        for tensor in (
                state["incoming_dir"], state["explored_edge_dirs"]):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())


if __name__ == "__main__":
    unittest.main()
