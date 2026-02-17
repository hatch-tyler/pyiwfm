"""Property-based tests for mesh operations using Hypothesis."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pyiwfm.core.mesh import AppGrid, Element, Node, Subregion

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

coords = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)


@st.composite
def node_strategy(draw: st.DrawFn, node_id: int = 1) -> Node:
    """Generate a random Node with valid coordinates."""
    x = draw(coords)
    y = draw(coords)
    return Node(id=node_id, x=x, y=y)


@st.composite
def quad_grid_strategy(draw: st.DrawFn) -> AppGrid:
    """Generate a random NxM quad grid (2-5 cells per side)."""
    nx = draw(st.integers(min_value=2, max_value=5))
    ny = draw(st.integers(min_value=2, max_value=5))
    spacing_x = draw(st.floats(min_value=1.0, max_value=1000.0))
    spacing_y = draw(st.floats(min_value=1.0, max_value=1000.0))

    nodes: dict[int, Node] = {}
    nid = 1
    for j in range(ny):
        for i in range(nx):
            nodes[nid] = Node(id=nid, x=i * spacing_x, y=j * spacing_y)
            nid += 1

    elements: dict[int, Element] = {}
    eid = 1
    for j in range(ny - 1):
        for i in range(nx - 1):
            n1 = j * nx + i + 1
            n2 = n1 + 1
            n3 = n2 + nx
            n4 = n1 + nx
            elements[eid] = Element(id=eid, vertices=(n1, n2, n3, n4), subregion=1)
            eid += 1

    subregions = {1: Subregion(id=1, name="Region1")}
    return AppGrid(nodes=nodes, elements=elements, subregions=subregions)


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


@pytest.mark.property
class TestNodeProperties:
    """Property-based tests for Node."""

    @given(node_strategy())
    def test_coordinates_tuple_matches_fields(self, node: Node) -> None:
        assert node.coordinates == (node.x, node.y)

    @given(node_strategy(node_id=1), node_strategy(node_id=2))
    def test_distance_is_non_negative(self, a: Node, b: Node) -> None:
        assert a.distance_to(b) >= 0.0

    @given(node_strategy(node_id=1), node_strategy(node_id=2))
    def test_distance_is_symmetric(self, a: Node, b: Node) -> None:
        np.testing.assert_allclose(a.distance_to(b), b.distance_to(a))

    @given(node_strategy())
    def test_distance_to_self_is_zero(self, node: Node) -> None:
        np.testing.assert_allclose(node.distance_to(node), 0.0)


@pytest.mark.property
class TestGridProperties:
    """Property-based tests for AppGrid."""

    @given(quad_grid_strategy())
    @settings(max_examples=20)
    def test_node_ids_are_unique(self, grid: AppGrid) -> None:
        ids = [n.id for n in grid.iter_nodes()]
        assert len(ids) == len(set(ids))

    @given(quad_grid_strategy())
    @settings(max_examples=20)
    def test_element_ids_are_unique(self, grid: AppGrid) -> None:
        ids = [e.id for e in grid.iter_elements()]
        assert len(ids) == len(set(ids))

    @given(quad_grid_strategy())
    @settings(max_examples=20)
    def test_element_vertices_reference_valid_nodes(self, grid: AppGrid) -> None:
        node_ids = set(grid.nodes.keys())
        for elem in grid.iter_elements():
            for vid in elem.vertices:
                assert vid in node_ids, f"Element {elem.id} references missing node {vid}"

    @given(quad_grid_strategy())
    @settings(max_examples=20)
    def test_bounding_box_contains_all_nodes(self, grid: AppGrid) -> None:
        xmin, ymin, xmax, ymax = grid.bounding_box
        for node in grid.iter_nodes():
            assert xmin <= node.x <= xmax
            assert ymin <= node.y <= ymax

    @given(quad_grid_strategy())
    @settings(max_examples=20)
    def test_n_nodes_matches_dict_length(self, grid: AppGrid) -> None:
        assert grid.n_nodes == len(grid.nodes)

    @given(quad_grid_strategy())
    @settings(max_examples=20)
    def test_n_elements_matches_dict_length(self, grid: AppGrid) -> None:
        assert grid.n_elements == len(grid.elements)

    @given(quad_grid_strategy())
    @settings(max_examples=20)
    def test_element_areas_are_positive(self, grid: AppGrid) -> None:
        for elem in grid.iter_elements():
            verts = elem.vertices
            xs = [grid.nodes[v].x for v in verts]
            ys = [grid.nodes[v].y for v in verts]
            # Shoelace formula
            n = len(verts)
            area = 0.0
            for i in range(n):
                j = (i + 1) % n
                area += xs[i] * ys[j]
                area -= xs[j] * ys[i]
            area = abs(area) / 2.0
            assert area > 0.0, f"Element {elem.id} has zero or negative area"
