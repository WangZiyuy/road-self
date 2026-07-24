import types
import unittest

import numpy as np
import torch

from lib import geom
from lib import graph as graph_helper
from utils.graph_state import build_graph_state
from utils.model_utils import Path, SearchVertexState


def _make_path():
    tile_data = {
        "search_rect": geom.Rectangle(
            geom.Point(-100, -100), geom.Point(100, 100)),
        "starting_locations": {
            "junction": [],
            "middle": [],
        },
    }
    return Path(
        idx=0,
        training=False,
        gc=types.SimpleNamespace(graph=graph_helper.Graph()),
        tile_data=tile_data,
        all_trajectories=[],
        all_pixel_trajectories=[],
        road_seg=None,
    )


class SearchVertexStateTest(unittest.TestCase):
    def test_start_state_has_no_incoming_direction(self):
        path = _make_path()
        vertex = path.graph.add_vertex(geom.Point(10, 20))
        state = SearchVertexState(vertex, is_key_point=True)

        features = build_graph_state(path, state)

        self.assertFalse(features["incoming_valid"].item())
        torch.testing.assert_close(
            features["incoming_dir"], torch.zeros(2))
        self.assertTrue(features["is_key_point"].item())

    def test_push_records_parent_and_incoming_edge(self):
        path = _make_path()
        current = path.graph.add_vertex(geom.Point(0, 0))

        def follow_one_step(**_kwargs):
            return 0, False, False, None, None

        path._follow_graph_one_step = follow_one_step
        path.push(
            extension_vertex=current,
            is_key_point=False,
            follow_mode="follow_output",
            target_poses=None,
            output_points=[geom.Point(3, 4)],
        )

        self.assertEqual(len(path.search_vertices), 1)
        state = path.search_vertices[0]
        self.assertIsInstance(state, SearchVertexState)
        self.assertEqual(state.parent_vertex_id, current.id)
        self.assertEqual(
            path.graph.edges[state.incoming_edge_id].src_id, current.id)
        self.assertEqual(
            path.graph.edges[state.incoming_edge_id].dst_id,
            state.vertex.id)

        features = build_graph_state(path, state)
        self.assertTrue(features["incoming_valid"].item())
        torch.testing.assert_close(
            features["incoming_dir"],
            torch.tensor([0.6, 0.8], dtype=torch.float32),
        )

    def test_same_vertex_can_have_distinct_parent_states(self):
        path = _make_path()
        parent_a = path.graph.add_vertex(geom.Point(-10, 0))
        parent_b = path.graph.add_vertex(geom.Point(0, -10))
        vertex = path.graph.add_vertex(geom.Point(0, 0))

        path.prepend_search_vertex(
            vertex, False, parent_vertex_id=parent_a.id)
        path.prepend_search_vertex(
            vertex, False, parent_vertex_id=parent_b.id)

        state_b = path.pop_state()
        state_a = path.pop_state()
        self.assertIs(state_a.vertex, vertex)
        self.assertIs(state_b.vertex, vertex)
        self.assertEqual(state_a.parent_vertex_id, parent_a.id)
        self.assertEqual(state_b.parent_vertex_id, parent_b.id)

    def test_legacy_pop_return_is_unchanged(self):
        path = _make_path()
        vertex = path.graph.add_vertex(geom.Point(1, 2))
        # Historical in-memory tuple entries remain readable as well.
        path.search_vertices.append((vertex, True))

        popped_vertex, is_key_point = path.pop()

        self.assertIs(popped_vertex, vertex)
        self.assertIs(is_key_point, True)
        self.assertEqual(path.pop(), (None, None))


class GraphStateTensorTest(unittest.TestCase):
    def test_incoming_and_explored_edge_directions_are_continuous(self):
        graph = graph_helper.Graph()
        parent = graph.add_vertex(geom.Point(0, 0))
        current = graph.add_vertex(geom.Point(3, 4))
        diagonal = graph.add_vertex(geom.Point(4, 5))
        incoming_edges = graph.add_bidirectional_edge(
            parent.id, current.id)
        graph.add_bidirectional_edge(current.id, diagonal.id)
        path = types.SimpleNamespace(graph=graph)
        state = SearchVertexState(
            current,
            False,
            parent_vertex_id=parent.id,
            incoming_edge_id=incoming_edges[0].id,
        )

        features = build_graph_state(
            path, state, max_explored_edges=4)

        torch.testing.assert_close(
            features["incoming_dir"], torch.tensor([0.6, 0.8]))
        self.assertEqual(
            features["explored_edge_mask"].tolist(),
            [True, True, False, False],
        )
        self.assertEqual(
            features["explored_is_incoming"].tolist(),
            [True, False, False, False],
        )
        torch.testing.assert_close(
            features["explored_edge_dirs"][0],
            torch.tensor([-0.6, -0.8]),
        )
        expected_diagonal = torch.tensor(
            [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
            dtype=torch.float32,
        )
        torch.testing.assert_close(
            features["explored_edge_dirs"][1], expected_diagonal)

    def test_neighbor_order_does_not_depend_on_edge_insertion(self):
        def make_graph(neighbor_points):
            graph = graph_helper.Graph()
            current = graph.add_vertex(geom.Point(0, 0))
            for point in neighbor_points:
                neighbor = graph.add_vertex(geom.Point(*point))
                graph.add_bidirectional_edge(current.id, neighbor.id)
            return graph, current

        graph_a, current_a = make_graph(
            [(4, 3), (-2, 5), (1, -7), (9, 1)])
        graph_b, current_b = make_graph(
            [(9, 1), (1, -7), (4, 3), (-2, 5)])

        features_a = build_graph_state(
            types.SimpleNamespace(graph=graph_a),
            SearchVertexState(current_a, False),
            max_explored_edges=2,
        )
        features_b = build_graph_state(
            types.SimpleNamespace(graph=graph_b),
            SearchVertexState(current_b, False),
            max_explored_edges=2,
        )

        torch.testing.assert_close(
            features_a["explored_edge_dirs"],
            features_b["explored_edge_dirs"],
        )
        self.assertTrue(
            features_a["explored_edge_mask"].all().item())


if __name__ == "__main__":
    unittest.main()
