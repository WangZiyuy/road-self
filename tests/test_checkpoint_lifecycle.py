import unittest
import warnings
from pathlib import Path

import torch
import torch.nn as nn

from utils.checkpoint_utils import (
    build_checkpoint_payload,
    load_checkpoint_into_model,
    resolve_inference_checkpoint_path,
    resolve_training_checkpoint_paths,
    save_training_checkpoint,
)


OUTPUT_KEYS = ("road", "junc", "anchor", "anchor_lowrs")


class _TinyRoadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(4, 4, kernel_size=1)

    def forward(self, aerial, walked):
        features = self.projection(torch.cat([aerial, walked], dim=1))
        road = features[:, 0:1]
        junc = features[:, 1:2]
        anchor = features
        return {
            "road": road,
            "junc": junc,
            "anchor": anchor,
            "anchor_lowrs": anchor * 0.5,
        }


def _cfg(checkpoint_dir):
    return {
        "TRAJ": {"MODE": "none"},
        "DIR": {"CHECK_POINT_DIR": str(checkpoint_dir)},
        "TRAIN": {
            "MODEL": "origin",
            "NUM_TARGETS": 4,
            "STEP_LENGTH": 20,
            "WINDOW_SIZE": 32,
            "CHECKPOINT": {
                "PREFIX": "image_only_test",
                "SAVE_LATEST": True,
                "SAVE_EVERY_OUTER": 1,
            },
        },
        "TEST": {
            "CKPT": "image_only_test.latest",
            "CKPT_FILE": "image_only_test.latest.pth.tar",
        },
    }


class CheckpointPathTest(unittest.TestCase):
    def test_training_paths_are_versioned_and_have_latest(self):
        cfg = _cfg("relative-checkpoints")
        paths = resolve_training_checkpoint_paths(cfg, outer_it=3, path_it=6)
        self.assertEqual(
            paths.versioned.name,
            "image_only_test.outer_003.path_0007.pth.tar",
        )
        self.assertEqual(paths.latest.name, "image_only_test.latest.pth.tar")
        self.assertEqual(
            paths.latest.resolve(strict=False),
            resolve_inference_checkpoint_path(cfg),
        )

    def test_exact_relative_inference_file_takes_priority(self):
        cfg = _cfg("relative-checkpoints")
        path = resolve_inference_checkpoint_path(cfg)
        self.assertEqual(path.name, "image_only_test.latest.pth.tar")
        self.assertEqual(path.parent.name, "relative-checkpoints")

    def test_exact_absolute_inference_file_is_not_rebased(self):
        absolute = (Path.cwd() / "absolute-model.pth.tar").resolve()
        cfg = _cfg("ignored")
        cfg["TEST"]["CKPT_FILE"] = str(absolute)
        cfg["TEST"]["CKPT"] = str(absolute)
        path = resolve_inference_checkpoint_path(cfg)
        self.assertEqual(path, absolute)

    def test_legacy_test_ckpt_fallback(self):
        cfg = _cfg("legacy-checkpoints")
        del cfg["TEST"]["CKPT_FILE"]
        cfg["TEST"]["CKPT"] = "vecroad2"
        path = resolve_inference_checkpoint_path(cfg)
        self.assertEqual(path.name, "vecroad2.pth.tar")
        self.assertEqual(path.parent.name, "legacy-checkpoints")

    def test_new_old_conflict_warns_and_new_path_wins(self):
        cfg = _cfg("checkpoints")
        cfg["TEST"]["CKPT"] = "old-name"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            path = resolve_inference_checkpoint_path(cfg)
        self.assertEqual(path.name, "image_only_test.latest.pth.tar")
        self.assertEqual(len(caught), 1)
        self.assertIn("takes precedence", str(caught[0].message))

    def test_missing_checkpoint_error_contains_resolved_path(self):
        cfg = _cfg("definitely-missing-checkpoints")
        expected = resolve_inference_checkpoint_path(cfg)
        with self.assertRaisesRegex(
            FileNotFoundError, str(expected).replace("\\", "\\\\")
        ):
            resolve_inference_checkpoint_path(cfg, require_exists=True)


class CheckpointRoundTripTest(unittest.TestCase):
    def test_model_optimizer_metadata_and_forward_round_trip(self):
        torch.manual_seed(20260722)
        model = _TinyRoadModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        aerial = torch.randn(1, 3, 8, 8)
        walked = torch.randn(1, 1, 8, 8)

        train_output = model(aerial, walked)
        sum(value.sum() for value in train_output.values()).backward()
        optimizer.step()
        optimizer.zero_grad()
        model.eval()
        with torch.no_grad():
            reference = model(aerial, walked)

        output_dir = Path(__file__).resolve().parent / "_checkpoint_roundtrip_output"
        output_dir.mkdir(exist_ok=True)
        cfg = _cfg(output_dir)
        paths = resolve_training_checkpoint_paths(
            cfg, outer_it=1, path_it=1
        )
        try:
            payload = build_checkpoint_payload(
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                outer_it=1,
                path_it=1,
                config_path="configs/test.yml",
                random_seed=20260722,
            )
            save_training_checkpoint(payload, paths)
            self.assertTrue(paths.versioned.is_file())
            self.assertTrue(paths.latest.is_file())

            restored_model = _TinyRoadModel().eval()
            restored_optimizer = torch.optim.Adam(
                restored_model.parameters(), lr=9e-2
            )
            restored = load_checkpoint_into_model(
                restored_model,
                paths.latest,
                optimizer=restored_optimizer,
                strict=True,
            )
            with torch.no_grad():
                actual = restored_model(aerial, walked)
        finally:
            for path in (paths.versioned, paths.latest):
                if path is not None:
                    path.unlink(missing_ok=True)
                    path.with_name(path.name + ".tmp").unlink(missing_ok=True)
            output_dir.rmdir()

        self.assertEqual(restored["trajectory_mode"], "none")
        self.assertEqual(restored["outer_it"], 1)
        self.assertEqual(restored["path_it"], 1)
        self.assertEqual(restored["random_seed"], 20260722)
        self.assertEqual(restored["model_name"], "origin")
        self.assertEqual(restored["num_targets"], 4)
        self.assertEqual(restored["step_length"], 20)
        self.assertEqual(restored["window_size"], 32)
        self.assertTrue(restored_optimizer.state_dict()["state"])
        for key in OUTPUT_KEYS:
            max_abs_diff = float((reference[key] - actual[key]).abs().max())
            self.assertLessEqual(max_abs_diff, 1e-6, key)


if __name__ == "__main__":
    unittest.main()
