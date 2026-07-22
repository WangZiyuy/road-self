import unittest

from lib import geom, graph as graph_helper
from scripts.validate_stage0_baseline import canonical_graph_signature


def _first_graph():
    graph = graph_helper.Graph()
    graph.add_vertex(geom.Point(0, 0), vertex_id=0)
    graph.add_vertex(geom.Point(10, 0), vertex_id=1)
    graph.add_vertex(geom.Point(10, 10), vertex_id=2)
    graph.add_edge(0, 1, edge_id=0)
    graph.add_edge(1, 2, edge_id=1)
    return graph


def _same_graph_with_different_ids_and_order():
    graph = graph_helper.Graph()
    graph.add_vertex(geom.Point(10, 10), vertex_id=30)
    graph.add_vertex(geom.Point(0, 0), vertex_id=10)
    graph.add_vertex(geom.Point(10, 0), vertex_id=20)
    graph.add_edge(20, 30, edge_id=91)
    graph.add_edge(10, 20, edge_id=77)
    return graph


class CanonicalGraphSignatureTest(unittest.TestCase):
    def test_signature_ignores_vertex_edge_ids_and_insertion_order(self):
        first = canonical_graph_signature(_first_graph())
        reordered = canonical_graph_signature(
            _same_graph_with_different_ids_and_order()
        )
        self.assertEqual(first, reordered)
        self.assertEqual(first["vertex_count"], 3)
        self.assertEqual(first["directed_edge_count"], 2)
        self.assertEqual(first["undirected_edge_count"], 2)

    def test_signature_detects_a_different_edge(self):
        first = _first_graph()
        changed = _same_graph_with_different_ids_and_order()
        changed.add_edge(10, 30, edge_id=105)
        self.assertNotEqual(
            canonical_graph_signature(first),
            canonical_graph_signature(changed),
        )

    def test_signature_supports_coordinate_tolerance(self):
        first = _first_graph()
        shifted = graph_helper.Graph()
        shifted.add_vertex(geom.FPoint(0.0000001, 0), vertex_id=10)
        shifted.add_vertex(geom.FPoint(10.0000001, 0), vertex_id=20)
        shifted.add_vertex(
            geom.FPoint(10.0000001, 10.0000001), vertex_id=30
        )
        shifted.add_edge(10, 20, edge_id=70)
        shifted.add_edge(20, 30, edge_id=80)
        self.assertEqual(
            canonical_graph_signature(first, coordinate_tolerance=1e-6),
            canonical_graph_signature(shifted, coordinate_tolerance=1e-6),
        )
        self.assertNotEqual(
            canonical_graph_signature(first, coordinate_tolerance=1e-8),
            canonical_graph_signature(shifted, coordinate_tolerance=1e-8),
        )


if __name__ == "__main__":
    unittest.main()
