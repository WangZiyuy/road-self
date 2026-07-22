import unittest
from pathlib import Path

import numpy as np
import yaml

from lib import geom
from utils.model_utils import map_to_coordinate


class _Extension:
    def __init__(self, x, y):
        self.point = geom.Point(x, y)


class InferenceThresholdRegressionTest(unittest.TestCase):
    def test_default_self_uses_anchor_safe_threshold(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "default_self.yml"
        )
        with config_path.open("r", encoding="utf-8") as config_file:
            cfg = yaml.load(config_file, Loader=yaml.UnsafeLoader)
        self.assertEqual(
            cfg["TEST"]["BINARIZE_MAP"]["ROAD_SEG_THRESHOLE"],
            0.3,
        )

    def test_low_threshold_can_merge_and_discard_anchor_component(self):
        heatmap = np.zeros((1, 4, 64, 64), dtype=np.float32)
        heatmap[0, 0, 21:44, 21:44] = 0.02
        heatmap[0, 0, 30:35, 30:35] = 0.8
        extension = [_Extension(100, 100)]

        low = map_to_coordinate(
            heatmap.copy(),
            np.ones(1, dtype=np.float32),
            extension,
            ROAD_SEG_THRESHOLE=0.01,
            STEP_LENGTH=20,
            JUNC_MAX_REGION_AREA=200,
        )
        safe = map_to_coordinate(
            heatmap.copy(),
            np.ones(1, dtype=np.float32),
            extension,
            ROAD_SEG_THRESHOLE=0.3,
            STEP_LENGTH=20,
            JUNC_MAX_REGION_AREA=200,
        )

        self.assertEqual(low, [[]])
        self.assertEqual(len(safe[0]), 1)


if __name__ == "__main__":
    unittest.main()
