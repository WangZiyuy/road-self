import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from model.branch_query_decoder import MultiModalBranchQueryDecoder
from model.branch_set_loss import BranchSetCriterion
from model.graph_state_encoder import GraphStateEncoder
from model.trajectory_encoder import TrajectoryFragmentEncoder
from utils.branch_metrics import BranchMetricAccumulator
from utils.stage3c_checkpoint import (
    build_stage3c_checkpoint_payload,
    load_stage3c_checkpoint,
    save_stage3c_checkpoint,
)
from train_branch_aux import _precompute_stage_fuse_cache


def _modules():
    return (
        TrajectoryFragmentEncoder(
            hidden_dim=32,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
        ),
        GraphStateEncoder(hidden_dim=32),
        MultiModalBranchQueryDecoder(
            image_channels=16,
            trajectory_dim=32,
            hidden_dim=32,
            num_queries=3,
            num_heads=4,
            image_pool_size=4,
            dropout=0.0,
        ),
    )


class Stage3CMetricsTest(unittest.TestCase):
    def test_perfect_prediction_metrics(self):
        offsets = torch.tensor([[
            [0.5, 0.0],
            [0.0, 0.5],
            [-0.5, -0.5],
        ]])
        predictions = {
            "branch_exist_logits": torch.tensor([[8.0, 8.0, -8.0]]),
            "branch_offsets_norm": offsets,
            "branch_directions": F.normalize(offsets, dim=-1),
        }
        target_offsets = offsets[:, :2].clone()
        targets = {
            "branch_offsets_norm": target_offsets,
            "branch_directions": F.normalize(
                target_offsets, dim=-1),
            "branch_mask": torch.tensor([[True, True]]),
        }
        metrics = BranchMetricAccumulator(
            window_size=256,
            existence_threshold=0.5,
        )
        metrics.update(predictions, targets)
        result = metrics.compute()
        self.assertEqual(result["precision"], 1.0)
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["f1"], 1.0)
        self.assertEqual(result["exact_branch_count_accuracy"], 1.0)
        self.assertAlmostEqual(
            result["endpoint_error_mean_pixels"], 0.0)
        self.assertAlmostEqual(
            result["direction_error_mean_degrees"], 0.0)

    def test_duplicate_queries_are_reported(self):
        offsets = torch.tensor([[
            [0.5, 0.0],
            [0.5, 0.0],
            [-0.5, 0.0],
        ]])
        predictions = {
            "branch_exist_logits": torch.tensor([[8.0, 8.0, -8.0]]),
            "branch_offsets_norm": offsets,
            "branch_directions": F.normalize(offsets, dim=-1),
        }
        targets = {
            "branch_offsets_norm": torch.tensor([[[0.5, 0.0]]]),
            "branch_directions": torch.tensor([[[1.0, 0.0]]]),
            "branch_mask": torch.tensor([[True]]),
        }
        metrics = BranchMetricAccumulator(window_size=256)
        metrics.update(predictions, targets)
        result = metrics.compute()
        self.assertEqual(result["duplicate_query_pair_ratio"], 1.0)
        self.assertEqual(result["false_positive"], 1)


class Stage3CTrainingComponentTest(unittest.TestCase):
    def test_frozen_rpnet_feature_cache_is_float32_and_index_stable(self):
        class TinyDataset(Dataset):
            def __len__(self):
                return 3

            def __getitem__(self, index):
                return {
                    "aerial_image": torch.full(
                        (3, 4, 4), float(index)),
                    "metadata": {
                        "dataset_index": torch.tensor(
                            index, dtype=torch.int64),
                    },
                }

        class TinyRPNet(torch.nn.Module):
            def _forward_origin_backbone(
                    self, aerial_image, feature_maps):
                value = aerial_image.mean(dim=1, keepdim=True)
                feature_maps["stage_fuse"] = value.repeat(
                    1, 128, 1, 1)

        cache, report = _precompute_stage_fuse_cache(
            rpnet=TinyRPNet(),
            dataset=TinyDataset(),
            batch_size=2,
            device=torch.device("cpu"),
        )
        self.assertEqual(tuple(cache.shape), (3, 128, 4, 4))
        self.assertEqual(cache.dtype, torch.float32)
        self.assertEqual(report["storage"], "volatile_cpu_memory")
        for index in range(3):
            torch.testing.assert_close(
                cache[index],
                torch.full_like(cache[index], float(index)),
                rtol=0.0,
                atol=0.0,
            )

    def test_empty_trajectory_sample_is_trainable(self):
        torch.manual_seed(20260724)
        trajectory_encoder, graph_encoder, decoder = _modules()
        trajectory_batch = {
            "traj_xy_norm": torch.zeros(1, 0, 0, 2),
            "traj_time_delta": torch.zeros(1, 0, 0),
            "point_mask": torch.zeros(
                1, 0, 0, dtype=torch.bool),
            "fragment_mask": torch.zeros(
                1, 0, dtype=torch.bool),
            "point_inside_mask": torch.zeros(
                1, 0, 0, dtype=torch.bool),
            "segment_only": torch.zeros(
                1, 0, dtype=torch.bool),
        }
        trajectory_output = trajectory_encoder(trajectory_batch)
        graph_state = {
            "incoming_dir": torch.zeros(1, 2),
            "incoming_valid": torch.zeros(1, dtype=torch.bool),
            "explored_edge_dirs": torch.zeros(1, 0, 2),
            "explored_edge_mask": torch.zeros(
                1, 0, dtype=torch.bool),
            "explored_is_incoming": torch.zeros(
                1, 0, dtype=torch.bool),
            "is_key_point": torch.zeros(1, dtype=torch.bool),
        }
        state_token = graph_encoder(graph_state)
        predictions = decoder(
            stage_fuse=torch.randn(1, 16, 8, 8),
            state_token=state_token,
            fragment_tokens=trajectory_output["fragment_tokens"],
            fragment_mask=trajectory_output["fragment_mask"],
            walked_path=torch.zeros(1, 1, 8, 8),
        )
        targets = {
            "branch_offsets_norm": torch.tensor([[[0.2, 0.0]]]),
            "branch_directions": torch.tensor([[[1.0, 0.0]]]),
            "branch_mask": torch.tensor([[True]]),
        }
        losses = BranchSetCriterion()(predictions, targets)
        losses["loss"].backward()
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertTrue(all(
            parameter.grad is None
            or torch.isfinite(parameter.grad).all()
            for module in (trajectory_encoder, graph_encoder, decoder)
            for parameter in module.parameters()
        ))

    def test_auxiliary_checkpoint_round_trip(self):
        torch.manual_seed(20260724)
        modules = _modules()
        parameters = [
            parameter
            for module in modules
            for parameter in module.parameters()
        ]
        optimizer = torch.optim.Adam(parameters, lr=1e-3)
        optimizer.zero_grad()
        sum(
            parameter.square().mean()
            for parameter in parameters
        ).backward()
        optimizer.step()

        stage_fuse = torch.randn(1, 16, 8, 8)
        state = torch.randn(1, 32)
        fragments = torch.randn(1, 2, 32)
        mask = torch.tensor([[True, True]])
        walked = torch.randn(1, 1, 8, 8)
        modules[2].eval()
        with torch.no_grad():
            expected = modules[2](
                stage_fuse,
                state,
                fragments,
                mask,
                walked_path=walked,
            )

        with tempfile.TemporaryDirectory(
                dir=Path.cwd()) as temporary:
            path = Path(temporary) / "stage3c.pth.tar"
            payload = build_stage3c_checkpoint_payload(
                trajectory_encoder=modules[0],
                graph_state_encoder=modules[1],
                branch_decoder=modules[2],
                optimizer=optimizer,
                epoch=7,
                image_checkpoint="/tmp/image_only.pth.tar",
                config_snapshot={"name": "unit"},
                metrics={"f1": 0.5},
            )
            save_stage3c_checkpoint(path, payload)

            restored = _modules()
            restored_optimizer = torch.optim.Adam([
                parameter
                for module in restored
                for parameter in module.parameters()
            ], lr=1e-3)
            loaded = load_stage3c_checkpoint(
                path,
                trajectory_encoder=restored[0],
                graph_state_encoder=restored[1],
                branch_decoder=restored[2],
                optimizer=restored_optimizer,
            )
            self.assertEqual(loaded["epoch"], 7)
            self.assertTrue(restored_optimizer.state)
            restored[2].eval()
            with torch.no_grad():
                actual = restored[2](
                    stage_fuse,
                    state,
                    fragments,
                    mask,
                    walked_path=walked,
                )
            for key in (
                    "branch_exist_logits",
                    "branch_offsets_norm",
                    "branch_directions"):
                torch.testing.assert_close(
                    expected[key], actual[key],
                    rtol=0.0, atol=0.0)

    def test_e4_small_batch_forward_backward_is_finite(self):
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
        )
        stage_fuse = torch.randn(2, 16, 8, 8)
        state_token = torch.randn(2, 32)
        predictions = decoder(
            stage_fuse,
            state_token,
            fragment_tokens=torch.zeros(2, 0, 32),
            fragment_mask=torch.zeros(2, 0, dtype=torch.bool),
            walked_path=torch.zeros(2, 1, 8, 8),
        )
        targets = {
            "branch_offsets_norm": torch.tensor([
                [[0.5, 0.0], [0.0, 0.5]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]),
            "branch_directions": torch.tensor([
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]),
            "branch_mask": torch.tensor([
                [True, True],
                [False, False],
            ]),
        }
        losses = BranchSetCriterion(
            match_cost_exist_weight=1.0,
            exist_no_object_coef=0.2,
        )(predictions, targets)
        losses["loss"].backward()
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertTrue(all(
            parameter.grad is None
            or torch.isfinite(parameter.grad).all()
            for parameter in decoder.parameters()
        ))


if __name__ == "__main__":
    unittest.main()
