import unittest

import numpy as np
import torch

from lib import geom
from lib import graph as graph_helper
from utils.branch_targets import (
    build_branch_target_batch,
    build_immediate_branch_targets,
)
from utils.model_utils import TargetPosesContainer


def _endpoint_position(edge, graph):
    return graph_helper.EdgePos(
        edge.id, edge.segment(graph).length())


class ImmediateBranchTargetTest(unittest.TestCase):
    def test_only_first_target_slot_is_an_immediate_branch(self):
        graph = graph_helper.Graph()
        current = graph.add_vertex(geom.Point(100, 100))
        immediate = graph.add_vertex(geom.Point(120, 100))
        future = graph.add_vertex(geom.Point(140, 100))
        immediate_edge = graph.add_edge(current.id, immediate.id)
        future_edge = graph.add_edge(immediate.id, future.id)
        target_poses = TargetPosesContainer(NUM_TARGETS=4)
        target_poses.target_poses[0].append(
            _endpoint_position(immediate_edge, graph))
        target_poses.target_poses[1].append(
            _endpoint_position(future_edge, graph))

        targets = build_immediate_branch_targets(
            target_poses, current, graph, window_size=256)

        self.assertEqual(targets.branch_count, 1)
        torch.testing.assert_close(
            targets.branch_offsets_rel,
            torch.tensor([[20.0, 0.0]]),
        )
        torch.testing.assert_close(
            targets.branch_offsets_norm,
            torch.tensor([[20.0 / 128.0, 0.0]]),
        )
        torch.testing.assert_close(
            targets.branch_directions,
            torch.tensor([[1.0, 0.0]]),
        )

    def test_t_junction_and_multibranch_targets_are_preserved(self):
        graph = graph_helper.Graph()
        current = graph.add_vertex(geom.Point(0, 0))
        endpoints = [
            graph.add_vertex(geom.Point(20, 0)),
            graph.add_vertex(geom.Point(0, 20)),
            graph.add_vertex(geom.Point(-20, 0)),
            graph.add_vertex(geom.Point(10, -10)),
        ]
        target_poses = TargetPosesContainer(NUM_TARGETS=4)
        for endpoint in endpoints:
            edge = graph.add_edge(current.id, endpoint.id)
            target_poses.target_poses[0].append(
                _endpoint_position(edge, graph))

        targets = build_immediate_branch_targets(
            target_poses, current, graph)

        self.assertEqual(targets.branch_count, 4)
        direction_set = {
            tuple(round(float(value), 6) for value in direction)
            for direction in targets.branch_directions.numpy()
        }
        self.assertEqual(direction_set, {
            (-1.0, 0.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (
                float(round(1.0 / np.sqrt(2.0), 6)),
                float(round(-1.0 / np.sqrt(2.0), 6)),
            ),
        })
        self.assertTrue(targets.branch_mask.all().item())

    def test_close_duplicate_endpoints_merge_by_distance_not_angle(self):
        graph = graph_helper.Graph()
        current = graph.add_vertex(geom.Point(0, 0))
        target_poses = TargetPosesContainer(NUM_TARGETS=4)
        target_poses.target_poses[0] = [
            geom.Point(20.0, 7.0),
            geom.Point(20.0004, 7.0003),
            geom.Point(20.0, 8.0),
        ]

        targets = build_immediate_branch_targets(
            target_poses,
            current,
            graph,
            merge_distance=1e-3,
        )

        self.assertEqual(targets.branch_count, 2)
        # The two remaining branches are close in angle but remain distinct
        # because only endpoint distance controls merging.
        self.assertFalse(torch.equal(
            targets.branch_directions[0],
            targets.branch_directions[1],
        ))

    def test_diagonal_direction_is_not_quantized(self):
        graph = graph_helper.Graph()
        current = graph.add_vertex(geom.Point(0, 0))
        target_poses = TargetPosesContainer(NUM_TARGETS=4)
        target_poses.target_poses[0] = [geom.Point(3, 4)]

        targets = build_immediate_branch_targets(
            target_poses, current, graph)

        torch.testing.assert_close(
            targets.branch_directions,
            torch.tensor([[0.6, 0.8]], dtype=torch.float32),
        )


class BranchTargetBatchTest(unittest.TestCase):
    def test_variable_and_empty_branch_sets_are_padded_explicitly(self):
        graph = graph_helper.Graph()
        current = graph.add_vertex(geom.Point(0, 0))
        empty = build_immediate_branch_targets(
            TargetPosesContainer(), current, graph)
        multi_container = TargetPosesContainer()
        multi_container.target_poses[0] = [
            geom.Point(10, 0),
            geom.Point(0, 10),
            geom.Point(-10, 0),
        ]
        multi = build_immediate_branch_targets(
            multi_container, current, graph)

        batch = build_branch_target_batch([empty, multi])

        self.assertEqual(
            tuple(batch["branch_offsets_norm"].shape), (2, 3, 2))
        self.assertEqual(
            tuple(batch["branch_directions"].shape), (2, 3, 2))
        self.assertEqual(
            batch["branch_mask"].tolist(),
            [[False, False, False], [True, True, True]],
        )
        self.assertEqual(batch["branch_count"].tolist(), [0, 3])
        self.assertTrue(torch.isfinite(
            batch["branch_offsets_norm"]).all().item())
        self.assertTrue(torch.isfinite(
            batch["branch_directions"]).all().item())
        torch.testing.assert_close(
            batch["branch_directions"][0], torch.zeros((3, 2)))

    def test_all_empty_batch_has_false_mask_and_no_nan(self):
        graph = graph_helper.Graph()
        current = graph.add_vertex(geom.Point(0, 0))
        empty_a = build_immediate_branch_targets(None, current, graph)
        empty_b = build_immediate_branch_targets(None, current, graph)

        batch = build_branch_target_batch([empty_a, empty_b])

        self.assertEqual(
            tuple(batch["branch_offsets_norm"].shape), (2, 0, 2))
        self.assertEqual(tuple(batch["branch_mask"].shape), (2, 0))
        self.assertEqual(batch["branch_count"].tolist(), [0, 0])
        self.assertEqual(batch["branch_mask"].numel(), 0)
        self.assertFalse(torch.isnan(
            batch["branch_offsets_norm"]).any().item())


if __name__ == "__main__":
    unittest.main()
