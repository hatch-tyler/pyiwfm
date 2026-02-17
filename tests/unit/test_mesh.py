"""Unit tests for mesh classes (Node, Element, Face, AppGrid)."""

from __future__ import annotations

import numpy as np
import pytest

from pyiwfm.core.exceptions import MeshError
from pyiwfm.core.mesh import AppGrid, Element, Face, Node, Subregion


class TestNode:
    """Tests for the Node class."""

    def test_node_creation(self, single_node_data: dict) -> None:
        """Test basic node creation with required attributes."""
        node = Node(**single_node_data)

        assert node.id == 1
        assert node.x == 100.0
        assert node.y == 200.0

    def test_node_default_attributes(self) -> None:
        """Test node has correct default attribute values."""
        node = Node(id=1, x=0.0, y=0.0)

        assert node.area == 0.0
        assert node.is_boundary is False
        assert node.connected_nodes == []
        assert node.surrounding_elements == []

    def test_node_with_optional_attributes(self) -> None:
        """Test node creation with optional attributes."""
        node = Node(
            id=1,
            x=100.0,
            y=200.0,
            area=1000.0,
            is_boundary=True,
            connected_nodes=[2, 3, 4],
            surrounding_elements=[1, 2],
        )

        assert node.area == 1000.0
        assert node.is_boundary is True
        assert node.connected_nodes == [2, 3, 4]
        assert node.surrounding_elements == [1, 2]

    def test_node_equality(self) -> None:
        """Test node equality comparison."""
        node1 = Node(id=1, x=100.0, y=200.0)
        node2 = Node(id=1, x=100.0, y=200.0)
        node3 = Node(id=2, x=100.0, y=200.0)

        assert node1 == node2
        assert node1 != node3

    def test_node_repr(self) -> None:
        """Test node string representation."""
        node = Node(id=1, x=100.0, y=200.0)
        repr_str = repr(node)

        assert "Node" in repr_str
        assert "id=1" in repr_str

    def test_node_coordinates_property(self) -> None:
        """Test coordinates tuple property."""
        node = Node(id=1, x=100.0, y=200.0)

        assert node.coordinates == (100.0, 200.0)

    def test_node_distance_to(self) -> None:
        """Test distance calculation between nodes."""
        node1 = Node(id=1, x=0.0, y=0.0)
        node2 = Node(id=2, x=3.0, y=4.0)

        assert node1.distance_to(node2) == 5.0


class TestElement:
    """Tests for the Element class."""

    def test_element_creation_quad(self, single_element_data: dict) -> None:
        """Test quadrilateral element creation."""
        elem = Element(**single_element_data)

        assert elem.id == 1
        assert elem.vertices == (1, 2, 3, 4)
        assert elem.subregion == 1

    def test_element_creation_triangle(self, triangle_element_data: dict) -> None:
        """Test triangular element creation."""
        elem = Element(**triangle_element_data)

        assert elem.id == 1
        assert elem.vertices == (1, 2, 3)
        assert elem.subregion == 1

    def test_element_is_triangle(self) -> None:
        """Test is_triangle property."""
        tri = Element(id=1, vertices=(1, 2, 3))
        quad = Element(id=2, vertices=(1, 2, 3, 4))

        assert tri.is_triangle is True
        assert quad.is_triangle is False

    def test_element_is_quad(self) -> None:
        """Test is_quad property."""
        tri = Element(id=1, vertices=(1, 2, 3))
        quad = Element(id=2, vertices=(1, 2, 3, 4))

        assert tri.is_quad is False
        assert quad.is_quad is True

    def test_element_n_vertices(self) -> None:
        """Test n_vertices property."""
        tri = Element(id=1, vertices=(1, 2, 3))
        quad = Element(id=2, vertices=(1, 2, 3, 4))

        assert tri.n_vertices == 3
        assert quad.n_vertices == 4

    def test_element_default_subregion(self) -> None:
        """Test element has default subregion of 0."""
        elem = Element(id=1, vertices=(1, 2, 3))

        assert elem.subregion == 0

    def test_element_default_area(self) -> None:
        """Test element has default area of 0."""
        elem = Element(id=1, vertices=(1, 2, 3))

        assert elem.area == 0.0

    def test_element_equality(self) -> None:
        """Test element equality comparison."""
        elem1 = Element(id=1, vertices=(1, 2, 3, 4))
        elem2 = Element(id=1, vertices=(1, 2, 3, 4))
        elem3 = Element(id=2, vertices=(1, 2, 3, 4))

        assert elem1 == elem2
        assert elem1 != elem3

    def test_element_invalid_vertices_raises(self) -> None:
        """Test that invalid vertex count raises error."""
        with pytest.raises(MeshError):
            Element(id=1, vertices=(1, 2))  # Too few vertices

        with pytest.raises(MeshError):
            Element(id=1, vertices=(1, 2, 3, 4, 5))  # Too many vertices

    def test_element_edges(self) -> None:
        """Test edges property returns correct edge pairs."""
        tri = Element(id=1, vertices=(1, 2, 3))
        quad = Element(id=2, vertices=(1, 2, 3, 4))

        assert tri.edges == [(1, 2), (2, 3), (3, 1)]
        assert quad.edges == [(1, 2), (2, 3), (3, 4), (4, 1)]


class TestFace:
    """Tests for the Face class."""

    def test_face_creation(self) -> None:
        """Test face creation."""
        face = Face(id=1, nodes=(1, 2), elements=(1, 2))

        assert face.id == 1
        assert face.nodes == (1, 2)
        assert face.elements == (1, 2)

    def test_face_is_boundary(self) -> None:
        """Test is_boundary property."""
        interior_face = Face(id=1, nodes=(1, 2), elements=(1, 2))
        boundary_face = Face(id=2, nodes=(1, 2), elements=(1, None))

        assert interior_face.is_boundary is False
        assert boundary_face.is_boundary is True

    def test_face_equality(self) -> None:
        """Test face equality considers node order."""
        face1 = Face(id=1, nodes=(1, 2), elements=(1, 2))
        face2 = Face(id=1, nodes=(1, 2), elements=(1, 2))
        face3 = Face(id=1, nodes=(2, 1), elements=(1, 2))  # Reversed nodes

        assert face1 == face2
        # Faces with reversed nodes are considered different
        assert face1 != face3


class TestSubregion:
    """Tests for the Subregion class."""

    def test_subregion_creation(self) -> None:
        """Test subregion creation."""
        sr = Subregion(id=1, name="Test Region")

        assert sr.id == 1
        assert sr.name == "Test Region"

    def test_subregion_default_name(self) -> None:
        """Test subregion has empty default name."""
        sr = Subregion(id=1)

        assert sr.name == ""


class TestAppGrid:
    """Tests for the AppGrid class."""

    def test_appgrid_creation(
        self, small_grid_nodes: list[dict], small_grid_elements: list[dict]
    ) -> None:
        """Test AppGrid creation from nodes and elements."""
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        elements = {d["id"]: Element(**d) for d in small_grid_elements}

        grid = AppGrid(nodes=nodes, elements=elements)

        assert grid.n_nodes == 9
        assert grid.n_elements == 4

    def test_appgrid_n_nodes(self, small_grid_nodes: list[dict]) -> None:
        """Test n_nodes property."""
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        grid = AppGrid(nodes=nodes, elements={})

        assert grid.n_nodes == 9

    def test_appgrid_n_elements(
        self, small_grid_nodes: list[dict], small_grid_elements: list[dict]
    ) -> None:
        """Test n_elements property."""
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        elements = {d["id"]: Element(**d) for d in small_grid_elements}
        grid = AppGrid(nodes=nodes, elements=elements)

        assert grid.n_elements == 4

    def test_appgrid_get_node(self, small_grid_nodes: list[dict]) -> None:
        """Test getting a node by ID."""
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        grid = AppGrid(nodes=nodes, elements={})

        node = grid.get_node(5)
        assert node.id == 5
        assert node.x == 100.0
        assert node.y == 100.0

    def test_appgrid_get_element(
        self, small_grid_nodes: list[dict], small_grid_elements: list[dict]
    ) -> None:
        """Test getting an element by ID."""
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        elements = {d["id"]: Element(**d) for d in small_grid_elements}
        grid = AppGrid(nodes=nodes, elements=elements)

        elem = grid.get_element(2)
        assert elem.id == 2
        assert elem.vertices == (2, 3, 6, 5)

    def test_appgrid_get_node_not_found(self, small_grid_nodes: list[dict]) -> None:
        """Test getting non-existent node raises error."""
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        grid = AppGrid(nodes=nodes, elements={})

        with pytest.raises(KeyError):
            grid.get_node(999)

    def test_appgrid_x_array(self, small_grid_nodes: list[dict]) -> None:
        """Test x coordinate array property."""
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        grid = AppGrid(nodes=nodes, elements={})

        x = grid.x
        assert isinstance(x, np.ndarray)
        assert x.shape == (9,)
        assert x[0] == 0.0  # Node 1
        assert x[4] == 100.0  # Node 5

    def test_appgrid_y_array(self, small_grid_nodes: list[dict]) -> None:
        """Test y coordinate array property."""
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        grid = AppGrid(nodes=nodes, elements={})

        y = grid.y
        assert isinstance(y, np.ndarray)
        assert y.shape == (9,)
        assert y[0] == 0.0  # Node 1
        assert y[4] == 100.0  # Node 5

    def test_appgrid_vertex_array(
        self, small_grid_nodes: list[dict], small_grid_elements: list[dict]
    ) -> None:
        """Test vertex connectivity array."""
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        elements = {d["id"]: Element(**d) for d in small_grid_elements}
        grid = AppGrid(nodes=nodes, elements=elements)

        v = grid.vertex
        assert isinstance(v, np.ndarray)
        assert v.shape[0] == 4  # 4 elements
        assert v.shape[1] == 4  # max 4 vertices per element

    def test_appgrid_element_area_calculation(
        self, small_grid_nodes: list[dict], small_grid_elements: list[dict]
    ) -> None:
        """Test element area calculation."""
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        elements = {d["id"]: Element(**d) for d in small_grid_elements}
        grid = AppGrid(nodes=nodes, elements=elements)

        grid.compute_areas()

        # Each element should have area 100*100 = 10000
        for elem_id in grid.elements:
            assert grid.elements[elem_id].area == pytest.approx(10000.0)

    def test_appgrid_node_area_calculation(
        self, small_grid_nodes: list[dict], small_grid_elements: list[dict]
    ) -> None:
        """Test node area calculation (Voronoi-like)."""
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        elements = {d["id"]: Element(**d) for d in small_grid_elements}
        grid = AppGrid(nodes=nodes, elements=elements)

        grid.compute_areas()

        # Total node area should equal total element area
        total_elem_area = sum(e.area for e in grid.elements.values())
        total_node_area = sum(n.area for n in grid.nodes.values())
        assert total_node_area == pytest.approx(total_elem_area)

    def test_appgrid_boundary_nodes(
        self, small_grid_nodes: list[dict], small_grid_elements: list[dict]
    ) -> None:
        """Test boundary node identification."""
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        elements = {d["id"]: Element(**d) for d in small_grid_elements}
        grid = AppGrid(nodes=nodes, elements=elements)

        grid.compute_connectivity()
        boundary_ids = grid.get_boundary_node_ids()

        # All edge nodes should be boundary (1,2,3,4,6,7,8,9)
        # Only center node (5) should be interior
        expected_boundary = {1, 2, 3, 4, 6, 7, 8, 9}
        assert set(boundary_ids) == expected_boundary

    def test_appgrid_build_faces(
        self, small_grid_nodes: list[dict], small_grid_elements: list[dict]
    ) -> None:
        """Test face construction from elements."""
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        elements = {d["id"]: Element(**d) for d in small_grid_elements}
        grid = AppGrid(nodes=nodes, elements=elements)

        grid.build_faces()

        # 2x2 grid should have 12 faces (4 exterior + 8 half of interior)
        # Actually: 4 elements * 4 edges = 16 edge references
        # Interior edges shared: 4 shared edges
        # Total unique faces: 16 - 4 = 12
        assert grid.n_faces == 12

    def test_appgrid_subregions(
        self, small_grid_nodes: list[dict], small_grid_elements: list[dict]
    ) -> None:
        """Test subregion tracking."""
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        elements = {d["id"]: Element(**d) for d in small_grid_elements}
        subregions = {
            1: Subregion(id=1, name="Region 1"),
            2: Subregion(id=2, name="Region 2"),
        }
        grid = AppGrid(nodes=nodes, elements=elements, subregions=subregions)

        assert grid.n_subregions == 2
        assert grid.get_elements_in_subregion(1) == [1, 2]
        assert grid.get_elements_in_subregion(2) == [3, 4]

    def test_appgrid_triangular_mesh(
        self, triangular_grid_nodes: list[dict], triangular_grid_elements: list[dict]
    ) -> None:
        """Test grid with triangular elements."""
        nodes = {d["id"]: Node(**d) for d in triangular_grid_nodes}
        elements = {d["id"]: Element(**d) for d in triangular_grid_elements}
        grid = AppGrid(nodes=nodes, elements=elements)

        assert grid.n_nodes == 4
        assert grid.n_elements == 2
        assert all(e.is_triangle for e in grid.elements.values())

    def test_appgrid_mixed_mesh(self) -> None:
        """Test grid with mixed triangular and quadrilateral elements."""
        nodes = {
            1: Node(id=1, x=0.0, y=0.0),
            2: Node(id=2, x=100.0, y=0.0),
            3: Node(id=3, x=100.0, y=100.0),
            4: Node(id=4, x=0.0, y=100.0),
            5: Node(id=5, x=50.0, y=150.0),
        }
        elements = {
            1: Element(id=1, vertices=(1, 2, 3, 4)),  # Quad
            2: Element(id=2, vertices=(4, 3, 5)),  # Triangle
        }
        grid = AppGrid(nodes=nodes, elements=elements)

        assert grid.n_elements == 2
        assert grid.elements[1].is_quad
        assert grid.elements[2].is_triangle

    def test_appgrid_centroid(
        self, small_grid_nodes: list[dict], small_grid_elements: list[dict]
    ) -> None:
        """Test element centroid calculation."""
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        elements = {d["id"]: Element(**d) for d in small_grid_elements}
        grid = AppGrid(nodes=nodes, elements=elements)

        # Element 1 has vertices (1,2,5,4) at (0,0), (100,0), (100,100), (0,100)
        cx, cy = grid.get_element_centroid(1)
        assert cx == pytest.approx(50.0)
        assert cy == pytest.approx(50.0)

    def test_appgrid_bounding_box(
        self, small_grid_nodes: list[dict], small_grid_elements: list[dict]
    ) -> None:
        """Test grid bounding box."""
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        elements = {d["id"]: Element(**d) for d in small_grid_elements}
        grid = AppGrid(nodes=nodes, elements=elements)

        bbox = grid.bounding_box
        assert bbox == (0.0, 0.0, 200.0, 200.0)  # (xmin, ymin, xmax, ymax)

    def test_appgrid_node_to_element_connectivity(
        self, small_grid_nodes: list[dict], small_grid_elements: list[dict]
    ) -> None:
        """Test node-to-element connectivity computation."""
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        elements = {d["id"]: Element(**d) for d in small_grid_elements}
        grid = AppGrid(nodes=nodes, elements=elements)

        grid.compute_connectivity()

        # Center node (5) should be surrounded by all 4 elements
        assert set(grid.nodes[5].surrounding_elements) == {1, 2, 3, 4}

        # Corner node (1) should only be part of element 1
        assert grid.nodes[1].surrounding_elements == [1]

    def test_appgrid_node_to_node_connectivity(
        self, small_grid_nodes: list[dict], small_grid_elements: list[dict]
    ) -> None:
        """Test node-to-node connectivity computation."""
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        elements = {d["id"]: Element(**d) for d in small_grid_elements}
        grid = AppGrid(nodes=nodes, elements=elements)

        grid.compute_connectivity()

        # Center node (5) should connect to all 8 surrounding nodes
        assert set(grid.nodes[5].connected_nodes) == {1, 2, 3, 4, 6, 7, 8, 9}

        # Corner node (1) should connect to nodes 2, 4, 5 (direct neighbors)
        assert set(grid.nodes[1].connected_nodes) == {2, 4, 5}


class TestAppGridValidation:
    """Tests for AppGrid validation."""

    def test_validate_empty_grid(self) -> None:
        """Test validation fails for empty grid."""
        grid = AppGrid(nodes={}, elements={})

        with pytest.raises(MeshError, match="no nodes"):
            grid.validate()

    def test_validate_no_elements(self, small_grid_nodes: list[dict]) -> None:
        """Test validation fails for grid without elements."""
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        grid = AppGrid(nodes=nodes, elements={})

        with pytest.raises(MeshError, match="no elements"):
            grid.validate()

    def test_validate_invalid_vertex_reference(self, small_grid_nodes: list[dict]) -> None:
        """Test validation fails for invalid vertex references."""
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        elements = {1: Element(id=1, vertices=(1, 2, 999, 4))}  # 999 doesn't exist
        grid = AppGrid(nodes=nodes, elements=elements)

        with pytest.raises(MeshError, match="invalid.*vertex"):
            grid.validate()

    def test_validate_duplicate_vertices(self, small_grid_nodes: list[dict]) -> None:
        """Test validation fails for duplicate vertices in element."""
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        elements = {1: Element(id=1, vertices=(1, 2, 2, 4))}  # Duplicate vertex 2
        grid = AppGrid(nodes=nodes, elements=elements)

        with pytest.raises(MeshError, match="duplicate.*vertices"):
            grid.validate()

    def test_validate_success(
        self, small_grid_nodes: list[dict], small_grid_elements: list[dict]
    ) -> None:
        """Test validation passes for valid grid."""
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        elements = {d["id"]: Element(**d) for d in small_grid_elements}
        grid = AppGrid(nodes=nodes, elements=elements)

        # Should not raise
        grid.validate()


class TestNodeEdgeCases:
    """Additional edge-case tests for Node."""

    def test_node_equality_with_non_node(self) -> None:
        """Test Node.__eq__ returns NotImplemented for non-Node objects."""
        node = Node(id=1, x=0.0, y=0.0)
        result = node.__eq__("not a node")
        assert result is NotImplemented

    def test_node_equality_with_different_coords(self) -> None:
        """Test Node inequality when id matches but coords differ."""
        n1 = Node(id=1, x=0.0, y=0.0)
        n2 = Node(id=1, x=1.0, y=0.0)
        assert n1 != n2

    def test_node_hash_consistency(self) -> None:
        """Test that equal nodes produce the same hash."""
        n1 = Node(id=1, x=100.0, y=200.0)
        n2 = Node(id=1, x=100.0, y=200.0)
        assert hash(n1) == hash(n2)

    def test_node_hash_in_set(self) -> None:
        """Test nodes can be used in sets based on hash."""
        n1 = Node(id=1, x=0.0, y=0.0)
        n2 = Node(id=1, x=0.0, y=0.0)
        n3 = Node(id=2, x=1.0, y=1.0)
        s = {n1, n2, n3}
        assert len(s) == 2

    def test_node_distance_to_self(self) -> None:
        """Test distance from a node to itself is 0."""
        node = Node(id=1, x=50.0, y=50.0)
        assert node.distance_to(node) == 0.0

    def test_node_repr_contains_coords(self) -> None:
        """Test repr includes x and y values."""
        node = Node(id=5, x=3.5, y=7.2)
        r = repr(node)
        assert "x=3.5" in r
        assert "y=7.2" in r


class TestElementEdgeCases:
    """Additional edge-case tests for Element."""

    def test_element_equality_with_non_element(self) -> None:
        """Test Element.__eq__ returns NotImplemented for non-Element."""
        elem = Element(id=1, vertices=(1, 2, 3))
        result = elem.__eq__("not an element")
        assert result is NotImplemented

    def test_element_hash_consistency(self) -> None:
        """Test that equal elements produce the same hash."""
        e1 = Element(id=1, vertices=(1, 2, 3))
        e2 = Element(id=1, vertices=(1, 2, 3))
        assert hash(e1) == hash(e2)

    def test_element_hash_in_set(self) -> None:
        """Test elements can be used in sets."""
        e1 = Element(id=1, vertices=(1, 2, 3))
        e2 = Element(id=1, vertices=(1, 2, 3))
        e3 = Element(id=2, vertices=(4, 5, 6))
        s = {e1, e2, e3}
        assert len(s) == 2

    def test_element_repr(self) -> None:
        """Test element repr contains id and vertices."""
        elem = Element(id=3, vertices=(10, 20, 30, 40))
        r = repr(elem)
        assert "Element" in r
        assert "id=3" in r
        assert "(10, 20, 30, 40)" in r

    def test_element_too_few_vertices_message(self) -> None:
        """Test error message content for too few vertices."""
        with pytest.raises(MeshError, match="too few vertices"):
            Element(id=1, vertices=(1, 2))

    def test_element_too_many_vertices_message(self) -> None:
        """Test error message content for too many vertices."""
        with pytest.raises(MeshError, match="too many vertices"):
            Element(id=1, vertices=(1, 2, 3, 4, 5))


class TestFaceEdgeCases:
    """Additional edge-case tests for Face."""

    def test_face_equality_with_non_face(self) -> None:
        """Test Face.__eq__ returns NotImplemented for non-Face."""
        face = Face(id=1, nodes=(1, 2), elements=(1, 2))
        result = face.__eq__("not a face")
        assert result is NotImplemented

    def test_face_hash_consistency(self) -> None:
        """Test that equal faces produce the same hash."""
        f1 = Face(id=1, nodes=(3, 4), elements=(1, 2))
        f2 = Face(id=1, nodes=(3, 4), elements=(1, 2))
        assert hash(f1) == hash(f2)

    def test_face_hash_in_set(self) -> None:
        """Test faces can be used in sets."""
        f1 = Face(id=1, nodes=(1, 2), elements=(1, None))
        f2 = Face(id=1, nodes=(1, 2), elements=(1, None))
        f3 = Face(id=2, nodes=(2, 3), elements=(1, 2))
        s = {f1, f2, f3}
        assert len(s) == 2

    def test_face_repr(self) -> None:
        """Test face repr format."""
        face = Face(id=5, nodes=(10, 20), elements=(1, 2))
        r = repr(face)
        assert "Face" in r
        assert "id=5" in r
        assert "nodes=(10, 20)" in r


class TestSubregionEdgeCases:
    """Additional edge-case tests for Subregion."""

    def test_subregion_repr(self) -> None:
        """Test subregion repr format."""
        sr = Subregion(id=3, name="My Region")
        r = repr(sr)
        assert "Subregion" in r
        assert "id=3" in r
        assert "My Region" in r

    def test_subregion_repr_empty_name(self) -> None:
        """Test subregion repr with empty name."""
        sr = Subregion(id=1)
        r = repr(sr)
        assert "Subregion" in r
        assert "name=''" in r


class TestAppGridAdditional:
    """Additional tests for AppGrid to cover missed lines."""

    def test_appgrid_repr(
        self, small_grid_nodes: list[dict], small_grid_elements: list[dict]
    ) -> None:
        """Test AppGrid repr format."""
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        elements = {d["id"]: Element(**d) for d in small_grid_elements}
        subregions = {1: Subregion(id=1, name="R1"), 2: Subregion(id=2, name="R2")}
        grid = AppGrid(nodes=nodes, elements=elements, subregions=subregions)
        r = repr(grid)
        assert "AppGrid" in r
        assert "n_nodes=9" in r
        assert "n_elements=4" in r
        assert "n_subregions=2" in r

    def test_appgrid_n_faces_before_build(
        self, small_grid_nodes: list[dict], small_grid_elements: list[dict]
    ) -> None:
        """Test n_faces is 0 before calling build_faces."""
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        elements = {d["id"]: Element(**d) for d in small_grid_elements}
        grid = AppGrid(nodes=nodes, elements=elements)
        assert grid.n_faces == 0

    def test_appgrid_get_face(
        self, small_grid_nodes: list[dict], small_grid_elements: list[dict]
    ) -> None:
        """Test getting a face by ID after building faces."""
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        elements = {d["id"]: Element(**d) for d in small_grid_elements}
        grid = AppGrid(nodes=nodes, elements=elements)
        grid.build_faces()
        face = grid.get_face(1)
        assert isinstance(face, Face)
        assert face.id == 1

    def test_appgrid_get_face_not_found(
        self, small_grid_nodes: list[dict], small_grid_elements: list[dict]
    ) -> None:
        """Test getting non-existent face raises KeyError."""
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        elements = {d["id"]: Element(**d) for d in small_grid_elements}
        grid = AppGrid(nodes=nodes, elements=elements)
        grid.build_faces()
        with pytest.raises(KeyError):
            grid.get_face(999)

    def test_appgrid_get_element_not_found(
        self, small_grid_nodes: list[dict], small_grid_elements: list[dict]
    ) -> None:
        """Test getting non-existent element raises KeyError."""
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        elements = {d["id"]: Element(**d) for d in small_grid_elements}
        grid = AppGrid(nodes=nodes, elements=elements)
        with pytest.raises(KeyError):
            grid.get_element(999)

    def test_appgrid_get_elements_by_subregion(
        self, small_grid_nodes: list[dict], small_grid_elements: list[dict]
    ) -> None:
        """Test grouping elements by subregion."""
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        elements = {d["id"]: Element(**d) for d in small_grid_elements}
        grid = AppGrid(nodes=nodes, elements=elements)
        by_sr = grid.get_elements_by_subregion()
        assert set(by_sr.keys()) == {1, 2}
        assert sorted(by_sr[1]) == [1, 2]
        assert sorted(by_sr[2]) == [3, 4]

    def test_appgrid_get_subregion_areas(
        self, small_grid_nodes: list[dict], small_grid_elements: list[dict]
    ) -> None:
        """Test getting subregion area totals."""
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        elements = {d["id"]: Element(**d) for d in small_grid_elements}
        grid = AppGrid(nodes=nodes, elements=elements)
        grid.compute_areas()
        sr_areas = grid.get_subregion_areas()
        assert sr_areas[1] == pytest.approx(20000.0)  # 2 elements * 10000
        assert sr_areas[2] == pytest.approx(20000.0)

    def test_appgrid_get_element_areas_array(
        self, small_grid_nodes: list[dict], small_grid_elements: list[dict]
    ) -> None:
        """Test getting element areas as array."""
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        elements = {d["id"]: Element(**d) for d in small_grid_elements}
        grid = AppGrid(nodes=nodes, elements=elements)
        grid.compute_areas()
        arr = grid.get_element_areas_array()
        assert arr.shape == (4,)
        for i in range(4):
            assert arr[i] == pytest.approx(10000.0)

    def test_appgrid_get_element_areas_array_empty(self) -> None:
        """Test get_element_areas_array with no elements."""
        nodes = {1: Node(id=1, x=0.0, y=0.0)}
        grid = AppGrid(nodes=nodes, elements={})
        arr = grid.get_element_areas_array()
        assert arr.shape == (0,)

    def test_appgrid_get_boundary_node_ids_no_boundary(self) -> None:
        """Test boundary node IDs when none are boundary."""
        nodes = {
            1: Node(id=1, x=0.0, y=0.0),
            2: Node(id=2, x=1.0, y=0.0),
        }
        grid = AppGrid(nodes=nodes, elements={})
        # Without computing connectivity, no nodes are boundary
        assert grid.get_boundary_node_ids() == []

    def test_appgrid_iter_nodes(self, small_grid_nodes: list[dict]) -> None:
        """Test iterating over nodes in ID order."""
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        grid = AppGrid(nodes=nodes, elements={})
        node_list = list(grid.iter_nodes())
        assert len(node_list) == 9
        assert [n.id for n in node_list] == list(range(1, 10))

    def test_appgrid_iter_elements(
        self, small_grid_nodes: list[dict], small_grid_elements: list[dict]
    ) -> None:
        """Test iterating over elements in ID order."""
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        elements = {d["id"]: Element(**d) for d in small_grid_elements}
        grid = AppGrid(nodes=nodes, elements=elements)
        elem_list = list(grid.iter_elements())
        assert len(elem_list) == 4
        assert [e.id for e in elem_list] == [1, 2, 3, 4]

    def test_appgrid_polygon_area_triangle(self) -> None:
        """Test static polygon area method with a triangle."""
        coords = [(0.0, 0.0), (4.0, 0.0), (0.0, 3.0)]
        area = AppGrid._polygon_area(coords)
        assert area == pytest.approx(6.0)

    def test_appgrid_polygon_area_square(self) -> None:
        """Test static polygon area method with a square."""
        coords = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        area = AppGrid._polygon_area(coords)
        assert area == pytest.approx(100.0)

    def test_appgrid_compute_areas_triangle_mesh(
        self, triangular_grid_nodes: list[dict], triangular_grid_elements: list[dict]
    ) -> None:
        """Test area computation for triangular mesh."""
        nodes = {d["id"]: Node(**d) for d in triangular_grid_nodes}
        elements = {d["id"]: Element(**d) for d in triangular_grid_elements}
        grid = AppGrid(nodes=nodes, elements=elements)
        grid.compute_areas()
        # Each triangle should have positive area
        for elem in grid.elements.values():
            assert elem.area > 0

    def test_appgrid_compute_connectivity_triangle_mesh(
        self, triangular_grid_nodes: list[dict], triangular_grid_elements: list[dict]
    ) -> None:
        """Test connectivity for triangular mesh."""
        nodes = {d["id"]: Node(**d) for d in triangular_grid_nodes}
        elements = {d["id"]: Element(**d) for d in triangular_grid_elements}
        grid = AppGrid(nodes=nodes, elements=elements)
        grid.compute_connectivity()
        # Node 2 is shared by both triangles
        assert set(grid.nodes[2].surrounding_elements) == {1, 2}
        # Node 1 is only in triangle 1
        assert grid.nodes[1].surrounding_elements == [1]
        # Boundary detection: nodes 1 and 4 are corners
        assert grid.nodes[1].is_boundary is True
        assert grid.nodes[4].is_boundary is True

    def test_appgrid_build_faces_triangle_mesh(
        self, triangular_grid_nodes: list[dict], triangular_grid_elements: list[dict]
    ) -> None:
        """Test face building for triangular mesh."""
        nodes = {d["id"]: Node(**d) for d in triangular_grid_nodes}
        elements = {d["id"]: Element(**d) for d in triangular_grid_elements}
        grid = AppGrid(nodes=nodes, elements=elements)
        grid.build_faces()
        # 2 triangles: 3+3=6 edge refs, 1 shared edge => 5 unique faces
        assert grid.n_faces == 5
        # Check boundary vs interior faces
        boundary_faces = [f for f in grid.faces.values() if f.is_boundary]
        interior_faces = [f for f in grid.faces.values() if not f.is_boundary]
        assert len(boundary_faces) == 4
        assert len(interior_faces) == 1

    def test_appgrid_build_faces_clears_previous(
        self, small_grid_nodes: list[dict], small_grid_elements: list[dict]
    ) -> None:
        """Test that build_faces clears previously built faces."""
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        elements = {d["id"]: Element(**d) for d in small_grid_elements}
        grid = AppGrid(nodes=nodes, elements=elements)
        grid.build_faces()
        first_count = grid.n_faces
        grid.build_faces()
        assert grid.n_faces == first_count  # Same count after rebuild

    def test_appgrid_vertex_array_triangle(
        self, triangular_grid_nodes: list[dict], triangular_grid_elements: list[dict]
    ) -> None:
        """Test vertex array for triangles pads 4th vertex with 0."""
        nodes = {d["id"]: Node(**d) for d in triangular_grid_nodes}
        elements = {d["id"]: Element(**d) for d in triangular_grid_elements}
        grid = AppGrid(nodes=nodes, elements=elements)
        v = grid.vertex
        assert v.shape == (2, 4)
        # For triangles, 4th column should be 0
        assert v[0, 3] == 0
        assert v[1, 3] == 0

    def test_appgrid_cache_invalidation(self) -> None:
        """Test that modifying nodes invalidates caches."""
        nodes = {
            1: Node(id=1, x=0.0, y=0.0),
            2: Node(id=2, x=1.0, y=0.0),
        }
        grid = AppGrid(nodes=nodes, elements={})
        # Access x to populate cache
        _ = grid.x
        assert grid._x_cache is not None
        # Invalidate cache
        grid._invalidate_cache()
        assert grid._x_cache is None
        assert grid._y_cache is None
        assert grid._vertex_cache is None
        assert grid._node_id_to_idx is None

    def test_appgrid_centroid_triangle(self) -> None:
        """Test centroid for a triangle element."""
        nodes = {
            1: Node(id=1, x=0.0, y=0.0),
            2: Node(id=2, x=6.0, y=0.0),
            3: Node(id=3, x=3.0, y=6.0),
        }
        elements = {1: Element(id=1, vertices=(1, 2, 3))}
        grid = AppGrid(nodes=nodes, elements=elements)
        cx, cy = grid.get_element_centroid(1)
        assert cx == pytest.approx(3.0)
        assert cy == pytest.approx(2.0)

    def test_appgrid_elements_in_nonexistent_subregion(
        self, small_grid_nodes: list[dict], small_grid_elements: list[dict]
    ) -> None:
        """Test getting elements for a subregion that has none."""
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        elements = {d["id"]: Element(**d) for d in small_grid_elements}
        grid = AppGrid(nodes=nodes, elements=elements)
        assert grid.get_elements_in_subregion(999) == []

    def test_appgrid_get_elements_by_subregion_unassigned(self) -> None:
        """Test get_elements_by_subregion with unassigned (subregion=0) elements."""
        nodes = {
            1: Node(id=1, x=0.0, y=0.0),
            2: Node(id=2, x=1.0, y=0.0),
            3: Node(id=3, x=0.5, y=1.0),
        }
        elements = {1: Element(id=1, vertices=(1, 2, 3))}  # subregion defaults to 0
        grid = AppGrid(nodes=nodes, elements=elements)
        by_sr = grid.get_elements_by_subregion()
        assert 0 in by_sr
        assert by_sr[0] == [1]

    def test_appgrid_n_subregions(
        self, small_grid_nodes: list[dict], small_grid_elements: list[dict]
    ) -> None:
        """Test n_subregions property."""
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        elements = {d["id"]: Element(**d) for d in small_grid_elements}
        subregions = {1: Subregion(id=1, name="A"), 2: Subregion(id=2, name="B")}
        grid = AppGrid(nodes=nodes, elements=elements, subregions=subregions)
        assert grid.n_subregions == 2

    def test_appgrid_x_y_caching(self, small_grid_nodes: list[dict]) -> None:
        """Test that x and y properties use caching."""
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        grid = AppGrid(nodes=nodes, elements={})
        x1 = grid.x
        x2 = grid.x  # Should return cached version
        assert x1 is x2
        y1 = grid.y
        y2 = grid.y
        assert y1 is y2

    def test_appgrid_single_element_mesh(self) -> None:
        """Test a minimal mesh with one triangular element."""
        nodes = {
            1: Node(id=1, x=0.0, y=0.0),
            2: Node(id=2, x=10.0, y=0.0),
            3: Node(id=3, x=5.0, y=10.0),
        }
        elements = {1: Element(id=1, vertices=(1, 2, 3))}
        grid = AppGrid(nodes=nodes, elements=elements)
        grid.compute_connectivity()
        grid.compute_areas()
        grid.build_faces()
        # All 3 nodes should be boundary
        assert all(n.is_boundary for n in grid.nodes.values())
        # 3 faces, all boundary
        assert grid.n_faces == 3
        assert all(f.is_boundary for f in grid.faces.values())
        # Area of triangle: 0.5 * 10 * 10 = 50
        assert grid.elements[1].area == pytest.approx(50.0)
