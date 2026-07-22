import sys
import types
import unittest
from unittest import mock

import numpy as np
import torch


try:
    from easydict import EasyDict
except ImportError:  # Keep the stage-0 tests runnable in the minimal local env.
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


import utils.OSMDataset as dataset_module
from model.model import RPNet
from scripts.validate_original_vecroad_alignment import (
    official_reference_forward,
)


class _FakePoint:
    def __repr__(self):
        return "Point(0, 0)"


class _FakeRect:
    start = _FakePoint()
    end = _FakePoint()


class _FakeGC:
    def clone(self):
        return self


class _InitPath:
    instances = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        _InitPath.instances.append(self)

    def visualize_and_save_path(self, *_args, **_kwargs):
        return None


class _FakeTiles:
    last_kwargs = None

    def __init__(self, **kwargs):
        _FakeTiles.last_kwargs = kwargs
        self.train_tiles = [object()]

    def prepare_training(self):
        return [{
            "region": "unit-region",
            "search_rect": _FakeRect(),
            "gc": _FakeGC(),
        }]


class _TargetPoses:
    def get_supervision_end_index(self):
        return 1


class _Graph:
    vertices = {}


class _BatchPath:
    def __init__(self, window_size, num_targets):
        self.window_size = window_size
        self.num_targets = num_targets
        self.graph = _Graph()
        self.fetch_list = None

    def pop(self, **_kwargs):
        return object(), False

    def make_path_input(self, extension_vertex, fetch_list, **_kwargs):
        self.fetch_list = tuple(fetch_list)
        forbidden = {"traj_image_chw", "traj_image_hwc", "valid_trajectories"}
        if forbidden.intersection(fetch_list):
            raise AssertionError("image-only get_batch requested trajectory data")
        w = self.window_size
        return {
            "aerial_image_chw": np.zeros((3, w, w), dtype=np.float32),
            "aerial_image_hwc": np.zeros((w, w, 3), dtype=np.float32),
            "walked_path_small": np.zeros((1, w // 4, w // 4), dtype=np.float32),
            "walked_path": np.zeros((1, w, w), dtype=np.float32),
            "road_seg_small": np.zeros((1, w // 4, w // 4), dtype=np.float32),
            "road_seg_thick3": np.zeros((1, w, w), dtype=np.float32),
            "junc_seg_small": np.zeros((1, w // 4, w // 4), dtype=np.float32),
            "junc_seg_thick3": np.zeros((1, w, w), dtype=np.float32),
        }

    def get_target_poses(self, **_kwargs):
        return _TargetPoses()

    def generate_target_maps(self, *_args, **_kwargs):
        return np.zeros(
            (self.num_targets, self.window_size, self.window_size),
            dtype=np.float32,
        )


def _image_only_cfg():
    return EasyDict({
        "TRAJ": EasyDict({"MODE": "none"}),
        "DIR": EasyDict({
            "ALL_REGION_PATH": "unused-regions",
            "GRAPH_DIR": "unused-graphs",
            "TILE_DIR": "unused-tiles",
        }),
        "TRAIN": EasyDict({
            "BATCH_SIZE": 1,
            "WINDOW_SIZE": 32,
            "NUM_INPUT_CHANNELS": 3,
            "NUM_TARGETS": 4,
            "PARALLEL_TILES": 1,
            "TRAINING_REGIONS": ["unit-region"],
            "MAX_PATH_LENGTH": 10,
            "TRAJ_FILTER": True,
            "STEP_LENGTH": 20,
            "RECT_RADIUS": 8,
        }),
    })


class ImageOnlyDataPathTest(unittest.TestCase):
    def test_dataset_init_does_not_load_trajectories(self):
        _InitPath.instances.clear()

        def forbidden_loader(*_args, **_kwargs):
            raise AssertionError("trajectory loader must not be called")

        with mock.patch.object(dataset_module, "Tiles", _FakeTiles), \
                mock.patch.object(dataset_module.model_utils, "Path", _InitPath), \
                mock.patch.object(
                    dataset_module, "load_region_trajectory_inputs", forbidden_loader
                ):
            dataset_module.OSMDataset(_image_only_cfg(), net=None)

        self.assertIsNone(_FakeTiles.last_kwargs["traj_dir"])
        self.assertEqual(len(_InitPath.instances), 1)
        self.assertIsNone(_InitPath.instances[0].args[4])
        self.assertEqual(_InitPath.instances[0].args[5], [])

    def test_get_batch_omits_all_trajectory_fields(self):
        cfg = _image_only_cfg()
        path = _BatchPath(window_size=32, num_targets=4)
        dataset = dataset_module.OSMDataset.__new__(dataset_module.OSMDataset)
        dataset.cfg = cfg
        dataset.trajectory_mode = "none"
        dataset.use_trajectory = False
        dataset.batch_size = 1
        dataset.window_size = 32
        dataset.input_channels = 3
        dataset.input_traj_channels = 0
        dataset.num_targets = 4
        dataset.paths = [path]
        dataset.subtiles = []
        dataset.all_trajectories = None
        dataset.all_pixel_trajectories = []
        dataset.traj_grid_index = None
        dataset.traj_grid_cell_size = None
        dataset.net = None

        batch = dataset.get_batch()

        self.assertNotIn("batch_traj_inputs", batch)
        self.assertNotIn("batch_aerial_traj", batch)
        self.assertNotIn("batch_traj_images_hwc", batch)
        self.assertNotIn("batch_valid_trajectory_inputs", batch)
        self.assertEqual(batch.batch_inputs.shape, (1, 3, 32, 32))
        self.assertEqual(batch.batch_target_maps.shape, (1, 4, 32, 32))


class ImageOnlyModelPathTest(unittest.TestCase):
    def test_output_keys_shapes_and_numerical_equivalence(self):
        torch.manual_seed(20260722)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = RPNet(num_targets=4, backbone_pretrained=False).to(device).eval()
        aerial = torch.randn(1, 3, 64, 64, device=device)
        walked = torch.randn(1, 1, 16, 16, device=device)
        dummy_traj_image = torch.randn(1, 1, 64, 64, device=device)
        dummy_aerial_traj = torch.randn(1, 4, 64, 64, device=device)
        dummy_tracks = torch.randn(1, 2, 3, 2, device=device)
        dummy_mask = torch.ones(1, 2, 3, dtype=torch.bool, device=device)

        with torch.no_grad():
            legacy = model(
                aerial,
                dummy_traj_image,
                dummy_aerial_traj,
                dummy_tracks,
                dummy_mask,
                walked,
                NUM_TARGETS=4,
                test=False,
                model="origin",
                use_traj=False,
            )
            stage0 = model(
                aerial,
                None,
                None,
                None,
                None,
                walked,
                NUM_TARGETS=4,
                test=False,
                model="origin",
                use_traj=False,
            )
            official = official_reference_forward(
                model, aerial, walked, num_targets=4
            )

        expected_shapes = {
            "road": (1, 1, 16, 16),
            "junc": (1, 1, 16, 16),
            "anchor": (1, 4, 64, 64),
            "anchor_lowrs": (1, 4, 64, 64),
        }
        for key, expected_shape in expected_shapes.items():
            self.assertIn(key, stage0)
            self.assertEqual(tuple(stage0[key].shape), expected_shape)
            self.assertTrue(torch.isfinite(stage0[key]).all())
            self.assertLessEqual(
                float((legacy[key] - stage0[key]).abs().max().cpu()), 1e-6
            )
            self.assertLessEqual(
                float((official[key] - stage0[key]).abs().max().cpu()), 1e-6
            )

        del model, legacy, stage0, official
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_image_only_parameter_surface_excludes_legacy_trajectory_modules(self):
        model = RPNet(num_targets=4, backbone_pretrained=False)
        keys = tuple(model.state_dict())
        self.assertEqual(len(keys), 648)
        forbidden_prefixes = (
            "transformer.",
            "fuse_module_traj.",
            "upsample1.",
            "DSF.",
            "stage_1_traj.",
            "missing_traj_feature",
        )
        self.assertFalse(any(
            key.startswith(forbidden_prefixes) for key in keys
        ))

        trajectory_model = RPNet(
            num_targets=4,
            backbone_pretrained=False,
            enable_trajectory_modules=True,
        )
        trajectory_keys = tuple(trajectory_model.state_dict())
        self.assertTrue(any(
            key.startswith("transformer.") for key in trajectory_keys
        ))
        self.assertTrue(any(
            key.startswith("fuse_module_traj.") for key in trajectory_keys
        ))
        trajectory_model.eval()
        with torch.no_grad():
            output = trajectory_model(
                torch.randn(1, 3, 64, 64),
                None,
                None,
                torch.randn(1, 2, 3, 2),
                torch.ones(1, 2, 3, dtype=torch.bool),
                torch.zeros(1, 1, 16, 16),
                model="origin",
                use_traj=True,
            )
        self.assertEqual(tuple(output["anchor"].shape), (1, 4, 64, 64))
        self.assertEqual(tuple(output["road"].shape), (1, 1, 16, 16))


if __name__ == "__main__":
    unittest.main()
