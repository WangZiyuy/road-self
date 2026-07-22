import unittest
import warnings
from pathlib import Path

import yaml

from utils.trajectory_mode import (
    TRAJ_MODE_LEGACY,
    TRAJ_MODE_NONE,
    load_region_trajectory_inputs_for_mode,
    prepare_trajectory_sequence_batch,
    resolve_trajectory_mode,
    trajectory_enabled,
    trajectory_fetch_fields,
    validate_trajectory_model_compatibility,
)


class TrajectoryModeTest(unittest.TestCase):
    def test_baseline_config_is_image_only_and_preserves_vecroad_constants(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "baseline_image_only.yml"
        )
        with config_path.open("r", encoding="utf-8") as config_file:
            cfg = yaml.load(config_file, Loader=yaml.UnsafeLoader)

        self.assertEqual(resolve_trajectory_mode(cfg), TRAJ_MODE_NONE)
        self.assertEqual(cfg["TRAIN"]["MODEL"], "origin")
        self.assertEqual(cfg["TRAIN"]["NUM_TARGETS"], 4)
        self.assertEqual(cfg["TRAIN"]["STEP_LENGTH"], 20)
        self.assertEqual(cfg["TRAIN"]["WINDOW_SIZE"], 256)
        self.assertEqual(
            cfg["TEST"]["BINARIZE_MAP"]["ROAD_SEG_THRESHOLE"],
            0.3,
        )
        self.assertNotIn("TRAJ_DIR", cfg["DIR"])
        self.assertNotIn("TEST_TRAJ_DIR", cfg["DIR"])

    def test_none_mode_rejects_a_trajectory_dependent_model(self):
        validate_trajectory_model_compatibility({
            "TRAJ": {"MODE": "none"},
            "TRAIN": {"MODEL": "origin"},
        })
        with self.assertRaisesRegex(ValueError, "requires TRAIN.MODEL='origin'"):
            validate_trajectory_model_compatibility({
                "TRAJ": {"MODE": "none"},
                "TRAIN": {"MODEL": "DSFNet"},
            })
        validate_trajectory_model_compatibility({
            "TRAJ": {"MODE": "legacy_current"},
            "TRAIN": {"MODEL": "DSFNet"},
        })

    def test_explicit_modes(self):
        self.assertEqual(
            resolve_trajectory_mode({"TRAJ": {"MODE": "none"}}),
            TRAJ_MODE_NONE,
        )
        self.assertEqual(
            resolve_trajectory_mode({"TRAJ": {"MODE": "legacy_current"}}),
            TRAJ_MODE_LEGACY,
        )
        self.assertFalse(trajectory_enabled({"TRAJ": {"MODE": "none"}}))
        self.assertTrue(
            trajectory_enabled({"TRAJ": {"MODE": "legacy_current"}})
        )

    def test_legacy_use_traj_mapping(self):
        self.assertEqual(
            resolve_trajectory_mode({"TRAIN": {"USE_TRAJ": False}}),
            TRAJ_MODE_NONE,
        )
        self.assertEqual(
            resolve_trajectory_mode({"TRAIN": {"USE_TRAJ": True}}),
            TRAJ_MODE_LEGACY,
        )
        self.assertEqual(resolve_trajectory_mode({"TRAIN": {}}), TRAJ_MODE_NONE)

    def test_new_mode_wins_and_warns_on_conflict(self):
        cfg = {
            "TRAJ": {"MODE": "none"},
            "TRAIN": {"USE_TRAJ": True},
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(resolve_trajectory_mode(cfg), TRAJ_MODE_NONE)
            self.assertEqual(resolve_trajectory_mode(cfg), TRAJ_MODE_NONE)
        self.assertEqual(len(caught), 1)
        self.assertIn("takes precedence", str(caught[0].message))

    def test_unknown_and_reserved_modes_fail(self):
        for mode in ("bad", "structured_all", "branch_slot", None):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                resolve_trajectory_mode({"TRAJ": {"MODE": mode}})

    def test_none_mode_has_no_fetch_fields(self):
        self.assertEqual(
            trajectory_fetch_fields(TRAJ_MODE_NONE, include_raster=True), ()
        )
        self.assertEqual(
            trajectory_fetch_fields(TRAJ_MODE_NONE, include_raster=False), ()
        )
        self.assertEqual(
            trajectory_fetch_fields(TRAJ_MODE_LEGACY, include_raster=False),
            ("valid_trajectories",),
        )

    def test_none_mode_does_not_call_region_loader(self):
        def forbidden_loader(*_args, **_kwargs):
            raise AssertionError("trajectory loader must not be called")

        result = load_region_trajectory_inputs_for_mode(
            TRAJ_MODE_NONE, "region", {}, forbidden_loader
        )
        self.assertEqual(result, (None, [], None, None))

    def test_none_mode_does_not_pad_or_normalize(self):
        def forbidden(*_args, **_kwargs):
            raise AssertionError("trajectory preprocessing must not be called")

        result = prepare_trajectory_sequence_batch(
            TRAJ_MODE_NONE, None, forbidden, forbidden
        )
        self.assertEqual(result, (None, None))

    def test_legacy_mode_preserves_loader_and_preprocessing(self):
        loader_result = ("raw", "pixel", "grid", 256)
        self.assertEqual(
            load_region_trajectory_inputs_for_mode(
                TRAJ_MODE_LEGACY,
                "region",
                {"key": "value"},
                lambda region, cfg: loader_result,
            ),
            loader_result,
        )
        normalized, mask = prepare_trajectory_sequence_batch(
            TRAJ_MODE_LEGACY,
            "tracks",
            lambda tracks: "padded-" + tracks,
            lambda padded: ("normalized-" + padded, "mask"),
        )
        self.assertEqual(normalized, "normalized-padded-tracks")
        self.assertEqual(mask, "mask")


if __name__ == "__main__":
    unittest.main()
