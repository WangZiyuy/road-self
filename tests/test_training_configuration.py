import unittest
from pathlib import Path

import yaml

from utils.training_utils import (
    DEFAULT_PATH_ITERATIONS,
    resolve_path_iterations,
    training_global_step,
)


class TrainingConfigurationTest(unittest.TestCase):
    def test_missing_path_iterations_preserves_legacy_default(self):
        self.assertEqual(
            resolve_path_iterations({"TRAIN": {}}),
            DEFAULT_PATH_ITERATIONS,
        )
        self.assertEqual(DEFAULT_PATH_ITERATIONS, 2048)

    def test_smoke_config_uses_two_path_iterations(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "baseline_image_only_smoke.yml"
        )
        with config_path.open("r", encoding="utf-8") as config_file:
            cfg = yaml.load(config_file, Loader=yaml.UnsafeLoader)
        self.assertEqual(resolve_path_iterations(cfg), 2)
        self.assertEqual(cfg["TRAIN"]["TOTAL_ITERATION"], 1)
        self.assertEqual(cfg["TRAIN"]["BATCH_SIZE"], 1)
        self.assertFalse(cfg["TRAIN"]["SAVE_EXAMPLES"])
        self.assertEqual(
            cfg["TEST"]["CKPT_FILE"],
            "image_only_original_smoke.latest.pth.tar",
        )

    def test_global_step_uses_configured_path_iterations(self):
        self.assertEqual(training_global_step(1, 0, 2), 2)
        self.assertEqual(training_global_step(1, 1, 2), 3)

    def test_train_loop_has_no_functional_hardcoded_2048(self):
        train_source = (
            Path(__file__).resolve().parents[1] / "train.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("range(2048)", train_source)
        self.assertNotIn("/2048]", train_source)
        self.assertNotIn("outer_it * 2048", train_source)


if __name__ == "__main__":
    unittest.main()
