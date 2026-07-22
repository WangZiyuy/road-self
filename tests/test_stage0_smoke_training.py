import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from easydict import EasyDict
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


import utils.OSMDataset as dataset_module
from utils.checkpoint_utils import (
    build_checkpoint_payload,
    load_checkpoint_into_model,
    resolve_training_checkpoint_paths,
    save_training_checkpoint,
)


class _Point:
    def __repr__(self):
        return "Point(0, 0)"


class _Rect:
    start = _Point()
    end = _Point()


class _GC:
    def clone(self):
        return self


class _Tiles:
    def __init__(self, **_kwargs):
        self.train_tiles = [object()]

    def prepare_training(self):
        return [{
            "region": "smoke-region",
            "search_rect": _Rect(),
            "gc": _GC(),
        }]


class _Extension:
    edge_pos = object()


class _TargetPoses:
    target_poses = [[]]

    def __len__(self):
        return 1

    def get_supervision_end_index(self):
        return 1


class _Graph:
    vertices = {}


class _SmokePath:
    instances = []

    def __init__(self, *args, **kwargs):
        self.graph = _Graph()
        self.window_size = 32
        self.num_targets = 4
        self.fetch_list = None
        self.push_calls = 0
        self.all_trajectories = args[4]
        self.all_pixel_trajectories = args[5]
        self.traj_grid_index = kwargs.get("traj_grid_index")
        self.traj_grid_cell_size = kwargs.get("traj_grid_cell_size")
        _SmokePath.instances.append(self)

    def visualize_and_save_path(self, *_args, **_kwargs):
        return None

    def pop(self, **_kwargs):
        return _Extension(), False

    def make_path_input(self, extension_vertex, fetch_list, **_kwargs):
        del extension_vertex
        self.fetch_list = tuple(fetch_list)
        forbidden = {"traj_image_chw", "traj_image_hwc", "valid_trajectories"}
        if forbidden.intersection(fetch_list):
            raise AssertionError("smoke image-only path requested trajectory data")
        w = self.window_size
        return {
            "aerial_image_chw": np.random.RandomState(1).rand(3, w, w).astype(np.float32),
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
        maps = np.zeros(
            (self.num_targets, self.window_size, self.window_size),
            dtype=np.float32,
        )
        maps[0, self.window_size // 2, self.window_size // 2] = 1.0
        return maps

    def push(self, **_kwargs):
        self.push_calls += 1


class _TinyRoadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Conv2d(4, 6, kernel_size=1)

    def forward(self, aerial, walked):
        walked_full = F.interpolate(
            walked, size=aerial.shape[-2:], mode="nearest")
        features = self.head(torch.cat([aerial, walked_full], dim=1))
        return {
            "road": F.avg_pool2d(features[:, 0:1], 4),
            "junc": F.avg_pool2d(features[:, 1:2], 4),
            "anchor": features[:, 2:6],
            "anchor_lowrs": features[:, 2:6] * 0.5,
        }


def _cfg(checkpoint_dir):
    return EasyDict({
        "TRAJ": EasyDict({"MODE": "none"}),
        "DIR": EasyDict({
            "ALL_REGION_PATH": "unused",
            "GRAPH_DIR": "unused",
            "TILE_DIR": "unused",
            "SHORTCUT_DIR": "unused",
            "CHECK_POINT_DIR": str(checkpoint_dir),
        }),
        "TRAIN": EasyDict({
            "MODEL": "origin",
            "BATCH_SIZE": 1,
            "WINDOW_SIZE": 32,
            "NUM_INPUT_CHANNELS": 3,
            "NUM_TARGETS": 4,
            "PARALLEL_TILES": 1,
            "TRAINING_REGIONS": ["smoke-region"],
            "MAX_PATH_LENGTH": 10,
            "STEP_LENGTH": 20,
            "RECT_RADIUS": 8,
            "AVG_CONFIDENCE_THRESHOLD": 0.2,
            "FOLLOW_MODE": "follow_target",
            "SAVE_EXAMPLES": False,
            "BINARIZE_MAP": EasyDict({
                "SEGMENTATION_THRESHOLD": 0.2,
                "MAX_REGION_AREA": 200,
            }),
            "CHECKPOINT": EasyDict({
                "PREFIX": "smoke_unit",
                "SAVE_LATEST": True,
                "SAVE_EVERY_OUTER": 1,
            }),
        }),
        "TEST": EasyDict({"CKPT_FILE": "smoke_unit.latest.pth.tar"}),
    })


class Stage0SmokeTrainingTest(unittest.TestCase):
    def test_image_only_batch_train_push_save_reload_forward(self):
        output_dir = Path(__file__).resolve().parent / "_smoke_training_output"
        output_dir.mkdir(exist_ok=True)
        cfg = _cfg(output_dir)
        _SmokePath.instances.clear()

        def forbidden_loader(*_args, **_kwargs):
            raise AssertionError("trajectory loader must not run in smoke training")

        try:
            with mock.patch.object(dataset_module, "Tiles", _Tiles), \
                    mock.patch.object(dataset_module.model_utils, "Path", _SmokePath), \
                    mock.patch.object(
                        dataset_module,
                        "load_region_trajectory_inputs",
                        forbidden_loader,
                    ):
                dataset = dataset_module.OSMDataset(cfg, net=None)
                batch = dataset.get_batch()

            self.assertNotIn("batch_valid_trajectory_inputs", batch)
            aerial = torch.from_numpy(batch.batch_inputs).float()
            walked = torch.from_numpy(batch.batch_walked_path_small).float()
            target_maps = torch.from_numpy(batch.batch_target_maps).float()
            road_target = torch.from_numpy(
                batch.batch_road_segmentation
            ).float()
            junc_target = torch.from_numpy(
                batch.batch_junction_segmentation
            ).float()

            torch.manual_seed(20260722)
            model = _TinyRoadModel()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            output = model(aerial, walked)
            anchor_loss = F.binary_cross_entropy_with_logits(
                output["anchor"][:, :1], target_maps[:, :1], reduction="sum"
            )
            anchor_mid_loss = F.binary_cross_entropy_with_logits(
                output["anchor_lowrs"][:, :1],
                target_maps[:, :1],
                reduction="sum",
            )
            anchor_loss = anchor_loss + anchor_mid_loss
            road_loss = F.binary_cross_entropy_with_logits(
                output["road"], road_target, reduction="sum"
            )
            junc_loss = F.binary_cross_entropy_with_logits(
                output["junc"], junc_target, reduction="sum"
            )
            loss = anchor_loss + road_loss + junc_loss
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            batch.batch_output_road = torch.sigmoid(output["road"]).detach().numpy()
            batch.batch_output_junc = torch.sigmoid(output["junc"]).detach().numpy()
            batch.batch_output_anchor_maps = torch.sigmoid(
                output["anchor"]
            ).detach().numpy()
            dataset.push_and_vis_batch(batch, outer_it=1, path_it=0)
            self.assertEqual(_SmokePath.instances[0].push_calls, 1)

            model.eval()
            with torch.no_grad():
                reference = model(aerial, walked)
            paths = resolve_training_checkpoint_paths(
                cfg, outer_it=1, path_it=0
            )
            payload = build_checkpoint_payload(
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                outer_it=1,
                path_it=0,
                config_path="configs/baseline_image_only_smoke.yml",
                random_seed=20260722,
            )
            save_training_checkpoint(payload, paths)

            restored_model = _TinyRoadModel().eval()
            restored_optimizer = torch.optim.Adam(
                restored_model.parameters(), lr=1e-2
            )
            restored_payload = load_checkpoint_into_model(
                restored_model,
                paths.latest,
                optimizer=restored_optimizer,
                strict=True,
            )
            with torch.no_grad():
                restored_output = restored_model(aerial, walked)

            self.assertEqual(restored_payload["trajectory_mode"], "none")
            self.assertTrue(restored_optimizer.state_dict()["state"])
            for key in ("road", "junc", "anchor", "anchor_lowrs"):
                self.assertTrue(torch.isfinite(restored_output[key]).all())
                self.assertLessEqual(
                    float((reference[key] - restored_output[key]).abs().max()),
                    1e-6,
                )
        finally:
            for candidate in output_dir.glob("smoke_unit*.pth.tar*"):
                candidate.unlink(missing_ok=True)
            output_dir.rmdir()


if __name__ == "__main__":
    unittest.main()
