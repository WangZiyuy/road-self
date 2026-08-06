import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model.trajectory_anchor_fusion import (
    ZeroInitializedTrajectoryAnchorFusion,
    fuse_cached_anchor_logits,
)
from model.trajectory_evidence_encoder import TrajectoryEvidenceEncoder
from utils.model_utils import Path as VecRoadPath
from utils.stage3fa_checkpoint import (
    build_stage3fa_checkpoint_payload,
    load_stage3fa_checkpoint,
    save_stage3fa_checkpoint,
)
from utils.stage3fa_anchor_cache import (
    REQUIRED_ARRAYS,
    ShardLocalShuffleSampler,
    Stage3FAAnchorDataset,
    validate_stage3fa_arrays,
)
from utils.stage3fa_metrics import PixelHistogramMetrics, localization_record
from utils.stage3fa_loss import original_vecroad_anchor_losses
from utils.trajectory_evidence_robustness import (
    global_wrong_sample_donor_indices,
)


class Stage3FAFusionTests(unittest.TestCase):
    def test_anchor_loss_matches_official_vecroad_bce_sum(self):
        anchor = torch.tensor([[[[0.2, -0.4], [1.0, 0.0]],
                                [[0.1, 0.2], [0.3, 0.4]]]])
        lowrs = anchor + 0.25
        target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]],
                                [[0.0, 1.0], [1.0, 0.0]]]])
        losses = original_vecroad_anchor_losses(
            anchor, lowrs, target, torch.tensor([1]))
        expected_full = F.binary_cross_entropy_with_logits(
            anchor[:, :1], target[:, :1], reduction="sum")
        expected_lowrs = F.binary_cross_entropy_with_logits(
            lowrs[:, :1], target[:, :1], reduction="sum")
        self.assertTrue(torch.equal(losses["anchor_loss"], expected_full))
        self.assertTrue(torch.equal(losses["anchor_lowrs_loss"], expected_lowrs))
        self.assertGreaterEqual(float(losses["anchor_total_loss"]), 0.0)

    def test_anchor_loss_rejects_reversed_unbounded_targets(self):
        logits = torch.zeros(1, 1, 2, 2)
        target = torch.tensor([[[[-2.0, 0.0], [1.0, 3.0]]]])
        with self.assertRaisesRegex(ValueError, "order may be reversed"):
            original_vecroad_anchor_losses(
                logits, logits, target, torch.tensor([1]))

    def setUp(self):
        torch.manual_seed(17)
        self.fusion = ZeroInitializedTrajectoryAnchorFusion(
            evidence_dim=6, anchor_channels=4, gate_hidden_dim=5)
        self.full = torch.randn(2, 3, 4, 8, 8)
        self.low = torch.randn(2, 3, 4, 2, 2)
        self.evidence = torch.randn(2, 1, 6)
        self.available = torch.tensor([True, True])
        self.full_weight = torch.randn(1, 4, 3, 3)
        self.low_weight = torch.randn(1, 4, 1, 1)
        self.base = torch.randn(2, 3, 8, 8)
        self.base_low = torch.randn(2, 3, 8, 8)

    def _forward(self, fusion=None, available=None, evidence=None):
        return fuse_cached_anchor_logits(
            fusion=fusion or self.fusion,
            anchor_features=self.full,
            anchor_lowrs_features=self.low,
            original_anchor_logits=self.base,
            original_anchor_lowrs_logits=self.base_low,
            trajectory_evidence=evidence if evidence is not None else self.evidence,
            trajectory_available=available if available is not None else self.available,
            anchor_head_weight=self.full_weight,
            anchor_lowrs_head_weight=self.low_weight)

    def test_zero_initialization_is_strictly_equivalent(self):
        output = self._forward()
        self.assertEqual(float((output["anchor"] - self.base).abs().max()), 0.0)
        self.assertEqual(float((output["anchor_lowrs"] - self.base_low).abs().max()), 0.0)

    def test_no_trajectory_is_strictly_equivalent_after_training_change(self):
        with torch.no_grad():
            self.fusion.anchor_adapter.output.weight.fill_(0.1)
            self.fusion.anchor_lowrs_adapter.output.weight.fill_(0.1)
        output = self._forward(available=torch.zeros(2))
        self.assertEqual(float((output["anchor"] - self.base).abs().max()), 0.0)
        self.assertEqual(float((output["anchor_lowrs"] - self.base_low).abs().max()), 0.0)

    def test_anchor_shapes_are_preserved(self):
        output = self._forward()
        self.assertEqual(output["anchor"].shape, self.base.shape)
        self.assertEqual(output["anchor_lowrs"].shape, self.base_low.shape)

    def test_empty_trajectory_is_finite(self):
        output = self._forward(
            available=torch.zeros(2), evidence=torch.zeros_like(self.evidence))
        self.assertTrue(torch.isfinite(output["anchor"]).all())
        self.assertTrue(torch.isfinite(output["anchor_lowrs"]).all())

    def test_only_fusion_parameters_receive_gradients(self):
        full_weight = self.full_weight.clone().requires_grad_(False)
        output = fuse_cached_anchor_logits(
            fusion=self.fusion, anchor_features=self.full,
            anchor_lowrs_features=self.low,
            original_anchor_logits=self.base,
            original_anchor_lowrs_logits=self.base_low,
            trajectory_evidence=self.evidence,
            trajectory_available=self.available,
            anchor_head_weight=full_weight,
            anchor_lowrs_head_weight=self.low_weight)
        output["anchor"].sum().backward()
        self.assertTrue(any(parameter.grad is not None
                            for parameter in self.fusion.parameters()))
        self.assertIsNone(full_weight.grad)

    def test_direct_prehead_and_cached_residual_are_equal(self):
        with torch.no_grad():
            self.fusion.anchor_adapter.output.weight.normal_(0, 0.01)
            self.fusion.anchor_lowrs_adapter.output.weight.normal_(0, 0.01)
        batch_size, steps = self.full.shape[:2]
        evidence = self.evidence[:, None].expand(
            -1, steps, -1, -1).reshape(batch_size * steps, 1, -1)
        available = self.available[:, None].expand(
            -1, steps).reshape(batch_size * steps)
        fused = self.fusion(
            anchor_feature=self.full.reshape(
                batch_size * steps, *self.full.shape[2:]),
            anchor_lowrs_feature=self.low.reshape(
                batch_size * steps, *self.low.shape[2:]),
            trajectory_evidence=evidence,
            trajectory_available=available)
        direct_delta = F.conv2d(
            fused["anchor_feature"] - self.full.reshape(
                batch_size * steps, *self.full.shape[2:]),
            self.full_weight, padding=1).reshape_as(self.base)
        cached = self._forward()["anchor"]
        self.assertTrue(torch.allclose(
            self.base + direct_delta, cached, atol=1e-6, rtol=0))

    def test_fragment_padding_does_not_change_evidence_or_fusion(self):
        encoder = TrajectoryEvidenceEncoder(
            hidden_dim=8, num_evidence_tokens=1, num_heads=2, dropout=0.0)
        encoder.eval()
        fragments = torch.randn(1, 2, 8)
        first = encoder(fragments, torch.tensor([[True, True]]))[
            "trajectory_evidence_tokens"]
        padded = torch.cat([fragments, torch.randn(1, 3, 8)], dim=1)
        second = encoder(padded, torch.tensor([[True, True, False, False, False]]))[
            "trajectory_evidence_tokens"]
        self.assertTrue(torch.allclose(first, second, atol=1e-6, rtol=0))

    def test_checkpoint_strict_round_trip(self):
        optimizer = torch.optim.Adam(self.fusion.parameters(), lr=1e-3)
        payload = build_stage3fa_checkpoint_payload(
            fusion=self.fusion, optimizer=optimizer, epoch=2, seed=7,
            validation_anchor_total_loss=1.2,
            checkpoint_sha256={"image": "a"},
            frozen_module_sha256={"rpnet": "b"},
            config_snapshot={"x": 1})
        path = Path.cwd() / "test_stage3fa_checkpoint.pth.tar"
        try:
            save_stage3fa_checkpoint(path, payload)
            restored = ZeroInitializedTrajectoryAnchorFusion(
                evidence_dim=6, anchor_channels=4, gate_hidden_dim=5)
            load_stage3fa_checkpoint(path, fusion=restored)
            for left, right in zip(self.fusion.parameters(), restored.parameters()):
                self.assertTrue(torch.equal(left, right))
        finally:
            if path.exists():
                path.unlink()

    def test_wrong_sample_mapping_is_global_and_deterministic(self):
        ids = torch.tensor([30, 10, 20])
        first = global_wrong_sample_donor_indices(ids)
        second = global_wrong_sample_donor_indices(ids)
        self.assertTrue(torch.equal(first, second))
        permutation = torch.tensor([2, 0, 1])
        permuted = global_wrong_sample_donor_indices(ids[permutation])
        donors = ids[first]
        permuted_donors = ids[permutation][permuted]
        self.assertTrue(torch.equal(permuted_donors, donors[permutation]))

    def test_pixel_metrics_and_original_coordinate_decoder(self):
        metrics = PixelHistogramMetrics(bins=64)
        metrics.update(
            torch.tensor([0.99, 0.01]).numpy(),
            torch.tensor([1.0, 0.0]).numpy())
        result = metrics.compute()
        self.assertGreater(result["pixel_ap"], 0.99)
        self.assertGreater(result["pixel_auroc"], 0.99)
        heatmap = torch.zeros(1, 256, 256).numpy()
        heatmap[0, 148, 128] = 0.9
        record = localization_record(
            probabilities=heatmap,
            center_xy=torch.tensor([100.0, 100.0]).numpy(),
            gt_xy=torch.tensor([[120.0, 100.0]]).numpy(),
            gt_mask=torch.tensor([True]).numpy(), threshold=0.3,
            step_length=20.0, junction_max_region_area=200,
            match_threshold=5.0)
        self.assertEqual(record["predicted_count"], 1)
        self.assertEqual(record["top1_endpoint_error"], 0.0)

    def test_branch_like_outputs_are_unchanged(self):
        branch = {
            "branch_exist_logits": torch.randn(2, 6),
            "branch_offsets_norm": torch.randn(2, 6, 2),
            "branch_directions": torch.randn(2, 6, 2),
        }
        before = {key: value.clone() for key, value in branch.items()}
        self._forward()
        for key in branch:
            self.assertTrue(torch.equal(before[key], branch[key]))

    def test_path_push_does_not_read_fusion_or_branch_output(self):
        source = inspect.getsource(VecRoadPath.push)
        self.assertNotIn("trajectory_anchor_fusion", source)
        self.assertNotIn("branch_exist_logits", source)
        self.assertNotIn("branch_offsets_norm", source)

    def test_cache_rejects_nonfinite_float_features(self):
        arrays = {
            name: np.zeros((1, 1), dtype=np.float32)
            for name in REQUIRED_ARRAYS
        }
        arrays["anchor_lowrs_features"][0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            validate_stage3fa_arrays(arrays)

    def test_shard_local_shuffle_is_deterministic_and_complete(self):
        dataset = object.__new__(Stage3FAAnchorDataset)
        dataset.index = [
            (0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 0)
        ]
        dataset.__class__.__len__ = lambda self: len(self.index)
        first = list(ShardLocalShuffleSampler(dataset, seed=19))
        repeated = list(ShardLocalShuffleSampler(dataset, seed=19))
        self.assertEqual(first, repeated)
        self.assertEqual(sorted(first), list(range(len(dataset.index))))
        shard_sequence = [dataset.index[index][0] for index in first]
        for shard_id in set(shard_sequence):
            positions = [index for index, value in enumerate(shard_sequence)
                         if value == shard_id]
            self.assertEqual(positions, list(range(min(positions), max(positions) + 1)))


if __name__ == "__main__":
    unittest.main()
