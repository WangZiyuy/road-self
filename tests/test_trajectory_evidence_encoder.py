import unittest
import uuid
from pathlib import Path

import torch
from easydict import EasyDict

from model.branch_query_decoder import MultiModalBranchQueryDecoder
from model.trajectory_evidence_encoder import (
    TRAJECTORY_MODE_EVIDENCE,
    TRAJECTORY_MODE_ORIGINAL_FRAGMENT,
    TrajectoryEvidenceEncoder,
    build_trajectory_decoder_inputs,
    resolve_trajectory_evidence_mode,
)
from train_trajectory_evidence import _evidence_diagnostics
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


if __name__ == "__main__":
    unittest.main()
