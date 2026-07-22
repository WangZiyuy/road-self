import sys
import types
import unittest
from unittest import mock

import numpy as np


try:
    from easydict import EasyDict  # noqa: F401
except ImportError:
    class EasyDict(dict):
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError as exc:
                raise AttributeError(key) from exc

        __setattr__ = dict.__setitem__

    easydict_module = types.ModuleType("easydict")
    easydict_module.EasyDict = EasyDict
    sys.modules["easydict"] = easydict_module


from lib import geom, graph as graph_helper
import utils.model_utils as model_utils
import utils.OSMDataset as dataset_module


class _EmptyEdgeIndex:
    @staticmethod
    def search(_rect):
        return []


class _VisualizationGC:
    def __init__(self):
        self.graph = graph_helper.Graph()
        self.edge_index = _EmptyEdgeIndex()


class _ConfigDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    __setattr__ = dict.__setitem__


def _make_path():
    rect = geom.Rectangle(geom.Point(0, 0), geom.Point(64, 64))
    tile_data = {
        "search_rect": rect,
        "starting_locations": {"junction": [], "middle": []},
    }
    path = model_utils.Path(
        idx=0,
        training=False,
        gc=None,
        tile_data=tile_data,
        all_trajectories=[],
        all_pixel_trajectories=[],
    )
    path.gc = _VisualizationGC()
    return path


def _prediction_pairs(window_size=32):
    zeros = np.zeros((window_size, window_size), dtype=np.float32)
    return [
        ("anchor", np.zeros((4, window_size, window_size), dtype=np.float32), None),
        ("road", zeros.copy(), zeros.copy()),
        ("junc", zeros.copy(), zeros.copy()),
    ]


def _extension_vertex():
    vertex = graph_helper.Vertex(-1, geom.Point(16, 16))
    vertex.edge_pos = None
    return vertex


class ImageOnlyVisualizationTest(unittest.TestCase):
    def _run_visualize(self, path):
        radii = []
        original_circle = model_utils.cv.circle

        def recording_circle(image, center, radius, color, thickness=None, **kwargs):
            radii.append(radius)
            if thickness is None:
                return original_circle(image, center, radius, color, **kwargs)
            return original_circle(
                image, center, radius, color, thickness=thickness, **kwargs
            )

        with mock.patch.object(
            model_utils.cv, "circle", side_effect=recording_circle
        ), mock.patch.object(model_utils.Image.Image, "save") as save_image:
            path.visualize_output(
                fname_prefix="unused_image_only_",
                extension_vertex=_extension_vertex(),
                aerial_image=np.zeros((32, 32, 3), dtype=np.float32),
                target_poses=None,
                pred_gt_pair_list=_prediction_pairs(),
                WINDOW_SIZE=32,
            )
        self.assertGreaterEqual(save_image.call_count, 1)
        return radii

    def test_path_initializes_empty_legacy_visualization_state(self):
        path = _make_path()
        self.assertEqual(path.valid_trajectories, [])
        self.assertEqual(path.circles, [])

    def test_visualize_is_safe_without_historical_circles_attribute(self):
        path = _make_path()
        del path.circles
        radii = self._run_visualize(path)
        self.assertNotIn(50, radii)

    def test_visualize_empty_trajectory_state_draws_no_filter_circle(self):
        path = _make_path()
        path.valid_trajectories = []
        path.circles = []
        radii = self._run_visualize(path)
        self.assertNotIn(50, radii)

    def test_legacy_visualization_fields_are_still_consumed(self):
        path = _make_path()
        path.valid_trajectories = [np.asarray([[15, 15], [16, 16]])]
        path.circles = [{"center": geom.Point(18, 18), "radius": 7}]
        radii = self._run_visualize(path)
        self.assertIn(50, radii)
        self.assertIn(7, radii)

    def test_push_and_vis_batch_with_save_examples_uses_real_visualizer(self):
        path = _make_path()
        cfg = _ConfigDict({
            "DIR": _ConfigDict({}),
            "TRAIN": _ConfigDict({
                "SAVE_EXAMPLES": True,
                "FOLLOW_MODE": "follow_target",
            }),
        })
        dataset = dataset_module.OSMDataset.__new__(dataset_module.OSMDataset)
        dataset.cfg = cfg
        dataset.save_idx = 0
        dataset.paths = [path]

        cfg.DIR.SHORTCUT_DIR = "unused"
        result = _ConfigDict({
            "path_indices": [0],
            "batch_extension_vertices": [_extension_vertex()],
            "batch_aerial_images_hwc": [
                np.zeros((32, 32, 3), dtype=np.float32)
            ],
            "batch_target_poses": [None],
            "batch_output_anchor_maps": np.zeros(
                (1, 4, 32, 32), dtype=np.float32
            ),
            "batch_target_maps": np.zeros((1, 4, 32, 32), dtype=np.float32),
            "batch_output_road": np.zeros((1, 1, 8, 8), dtype=np.float32),
            "batch_road_segmentation": np.zeros(
                (1, 1, 8, 8), dtype=np.float32
            ),
            "batch_road_segmentation_thick3": np.zeros(
                (1, 1, 32, 32), dtype=np.float32
            ),
            "batch_output_junc": np.zeros((1, 1, 8, 8), dtype=np.float32),
            "batch_junction_segmentation": np.zeros(
                (1, 1, 8, 8), dtype=np.float32
            ),
            "batch_is_key_point": np.zeros(1, dtype=np.float32),
        })

        with mock.patch.object(model_utils.Image.Image, "save") as save_image:
            dataset.push_and_vis_batch(result, outer_it=1, path_it=0)
        self.assertGreaterEqual(save_image.call_count, 1)


if __name__ == "__main__":
    unittest.main()
