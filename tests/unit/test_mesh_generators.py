"""
Comprehensive tests for pyiwfm.mesh_generation.generators module.

Tests cover:
- MeshResult dataclass properties and methods
- MeshGenerator abstract base class
- Element area and centroid calculations
- Conversion to AppGrid
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from pyiwfm.mesh_generation.generators import MeshResult, MeshGenerator


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def triangle_mesh():
    """Create a simple triangular mesh."""
    nodes = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 1.0],
    ])
    elements = np.array([
        [0, 1, 2],
    ], dtype=np.int32)

    return MeshResult(nodes=nodes, elements=elements)


@pytest.fixture
def quad_mesh():
    """Create a simple quad mesh."""
    nodes = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ])
    elements = np.array([
        [0, 1, 2, 3],
    ], dtype=np.int32)

    return MeshResult(nodes=nodes, elements=elements)


@pytest.fixture
def mixed_mesh():
    """Create a mesh with both triangles and quads."""
    nodes = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [2.0, 1.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [0.5, 2.0],
    ])
    # Use -1 to indicate triangle (only 3 vertices)
    elements = np.array([
        [0, 1, 4, 5],      # Quad
        [1, 2, 3, 4],      # Quad
        [4, 5, 6, -1],     # Triangle (4th vertex -1)
    ], dtype=np.int32)

    return MeshResult(nodes=nodes, elements=elements)


@pytest.fixture
def mesh_with_markers():
    """Create a mesh with node and element markers."""
    nodes = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ])
    elements = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ], dtype=np.int32)
    node_markers = np.array([1, 1, 0, 1], dtype=np.int32)  # Boundary markers
    element_markers = np.array([1, 2], dtype=np.int32)  # Region markers

    return MeshResult(
        nodes=nodes,
        elements=elements,
        node_markers=node_markers,
        element_markers=element_markers,
    )


@pytest.fixture
def multi_triangle_mesh():
    """Create a mesh with multiple triangles."""
    nodes = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [1.0, 1.0],
        [0.5, 0.5],
    ])
    elements = np.array([
        [0, 1, 4],
        [1, 2, 3],
        [1, 3, 4],
    ], dtype=np.int32)

    return MeshResult(nodes=nodes, elements=elements)


# =============================================================================
# MeshResult Properties Tests
# =============================================================================


class TestMeshResultProperties:
    """Tests for MeshResult property accessors."""

    def test_n_nodes(self, triangle_mesh):
        """Test n_nodes property."""
        assert triangle_mesh.n_nodes == 3

    def test_n_elements(self, triangle_mesh):
        """Test n_elements property."""
        assert triangle_mesh.n_elements == 1

    def test_n_triangles_with_3col_elements(self, triangle_mesh):
        """Test n_triangles with 3-column elements array."""
        assert triangle_mesh.n_triangles == 1

    def test_n_triangles_with_4col_elements(self, mixed_mesh):
        """Test n_triangles with mixed mesh."""
        assert mixed_mesh.n_triangles == 1  # One triangle (4th vertex = -1)

    def test_n_quads_with_3col_elements(self, triangle_mesh):
        """Test n_quads with triangle-only mesh."""
        assert triangle_mesh.n_quads == 0

    def test_n_quads_with_4col_elements(self, mixed_mesh):
        """Test n_quads with mixed mesh."""
        assert mixed_mesh.n_quads == 2

    def test_n_quads_only_quads(self, quad_mesh):
        """Test n_quads with quad-only mesh."""
        assert quad_mesh.n_quads == 1

    def test_n_triangles_quad_only_mesh(self, quad_mesh):
        """Test n_triangles with quad-only mesh."""
        assert quad_mesh.n_triangles == 0


# =============================================================================
# MeshResult Area and Centroid Tests
# =============================================================================


class TestMeshResultAreaCentroid:
    """Tests for area and centroid calculations."""

    def test_get_element_areas_triangle(self, triangle_mesh):
        """Test area calculation for triangle."""
        areas = triangle_mesh.get_element_areas()

        assert len(areas) == 1
        # Triangle with vertices (0,0), (1,0), (0.5,1)
        # Area = 0.5 * |1*1 - 0.5*0| = 0.5
        assert areas[0] == pytest.approx(0.5, rel=1e-5)

    def test_get_element_areas_quad(self, quad_mesh):
        """Test area calculation for quad."""
        areas = quad_mesh.get_element_areas()

        assert len(areas) == 1
        # Unit square area = 1.0
        assert areas[0] == pytest.approx(1.0, rel=1e-5)

    def test_get_element_areas_mixed(self, mixed_mesh):
        """Test area calculation for mixed mesh."""
        areas = mixed_mesh.get_element_areas()

        assert len(areas) == 3
        # First two are 1x1 quads
        assert areas[0] == pytest.approx(1.0, rel=1e-5)
        assert areas[1] == pytest.approx(1.0, rel=1e-5)
        # Third is a triangle

    def test_get_element_centroids_triangle(self, triangle_mesh):
        """Test centroid calculation for triangle."""
        centroids = triangle_mesh.get_element_centroids()

        assert centroids.shape == (1, 2)
        # Centroid of (0,0), (1,0), (0.5,1) is (0.5, 1/3)
        assert centroids[0, 0] == pytest.approx(0.5, rel=1e-5)
        assert centroids[0, 1] == pytest.approx(1.0/3.0, rel=1e-5)

    def test_get_element_centroids_quad(self, quad_mesh):
        """Test centroid calculation for quad."""
        centroids = quad_mesh.get_element_centroids()

        assert centroids.shape == (1, 2)
        # Unit square centroid is (0.5, 0.5)
        assert centroids[0, 0] == pytest.approx(0.5, rel=1e-5)
        assert centroids[0, 1] == pytest.approx(0.5, rel=1e-5)

    def test_get_element_centroids_multiple(self, multi_triangle_mesh):
        """Test centroid calculation for multiple elements."""
        centroids = multi_triangle_mesh.get_element_centroids()

        assert centroids.shape == (3, 2)


# =============================================================================
# MeshResult Conversion Tests
# =============================================================================


class TestMeshResultConversion:
    """Tests for MeshResult conversion methods."""

    def test_to_appgrid_basic(self, triangle_mesh):
        """Test conversion to AppGrid."""
        grid = triangle_mesh.to_appgrid()

        assert grid.n_nodes == 3
        assert grid.n_elements == 1

    def test_to_appgrid_with_markers(self, mesh_with_markers):
        """Test conversion with boundary markers."""
        grid = mesh_with_markers.to_appgrid()

        # Check grid was created with correct number of nodes
        assert grid.n_nodes == 4
        assert grid.n_elements == 2

    def test_to_appgrid_with_element_markers(self, mesh_with_markers):
        """Test conversion with element markers."""
        grid = mesh_with_markers.to_appgrid()

        assert grid.elements[1].subregion == 1
        assert grid.elements[2].subregion == 2

    def test_to_appgrid_quad(self, quad_mesh):
        """Test conversion with quad elements."""
        grid = quad_mesh.to_appgrid()

        assert grid.n_nodes == 4
        assert grid.n_elements == 1
        # Check vertices are 1-indexed
        assert grid.elements[1].vertices == (1, 2, 3, 4)

    def test_to_appgrid_sets_coordinate_arrays(self, triangle_mesh):
        """Test that coordinate arrays are set."""
        grid = triangle_mesh.to_appgrid()

        assert hasattr(grid, '_x')
        assert hasattr(grid, '_y')
        assert len(grid._x) == 3

    def test_repr(self, triangle_mesh):
        """Test string representation."""
        result = repr(triangle_mesh)

        assert "MeshResult" in result
        assert "n_nodes=3" in result
        assert "n_elements=1" in result


# =============================================================================
# MeshResult Edge Cases
# =============================================================================


class TestMeshResultEdgeCases:
    """Tests for edge cases."""

    def test_empty_markers(self):
        """Test mesh without markers."""
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        elements = np.array([[0, 1, 2]], dtype=np.int32)

        mesh = MeshResult(nodes=nodes, elements=elements)

        assert mesh.node_markers is None
        assert mesh.element_markers is None

    def test_to_appgrid_no_markers(self):
        """Test to_appgrid without markers."""
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        elements = np.array([[0, 1, 2]], dtype=np.int32)

        mesh = MeshResult(nodes=nodes, elements=elements)
        grid = mesh.to_appgrid()

        # Should work and create correct grid
        assert grid.n_nodes == 3
        assert grid.n_elements == 1

    def test_single_node(self):
        """Test mesh with single node (degenerate)."""
        nodes = np.array([[0.0, 0.0]])
        elements = np.array([], dtype=np.int32).reshape(0, 3)

        mesh = MeshResult(nodes=nodes, elements=elements)

        assert mesh.n_nodes == 1
        assert mesh.n_elements == 0

    def test_large_mesh(self):
        """Test with larger mesh."""
        n_nodes = 100
        n_elements = 150

        nodes = np.random.rand(n_nodes, 2)
        elements = np.random.randint(0, n_nodes, size=(n_elements, 3), dtype=np.int32)

        mesh = MeshResult(nodes=nodes, elements=elements)

        assert mesh.n_nodes == n_nodes
        assert mesh.n_elements == n_elements
        assert mesh.n_triangles == n_elements


# =============================================================================
# MeshGenerator Tests
# =============================================================================


class ConcreteMeshGenerator(MeshGenerator):
    """Concrete implementation for testing."""

    def generate(
        self,
        boundary,
        max_area=None,
        min_angle=None,
        streams=None,
        refinement_zones=None,
        points=None,
    ):
        """Generate a simple mesh."""
        # Return a simple triangle mesh
        nodes = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
        ])
        elements = np.array([[0, 1, 2]], dtype=np.int32)
        return MeshResult(nodes=nodes, elements=elements)


class TestMeshGenerator:
    """Tests for MeshGenerator abstract class."""

    def test_cannot_instantiate_abstract(self):
        """Test that abstract class cannot be instantiated."""
        with pytest.raises(TypeError):
            MeshGenerator()

    def test_concrete_implementation(self):
        """Test concrete implementation works."""
        generator = ConcreteMeshGenerator()
        boundary = MagicMock()

        result = generator.generate(boundary)

        assert isinstance(result, MeshResult)
        assert result.n_nodes == 3

    def test_generate_with_all_params(self):
        """Test generate with all parameters."""
        generator = ConcreteMeshGenerator()
        boundary = MagicMock()
        streams = [MagicMock()]
        refinement_zones = [MagicMock()]
        points = [MagicMock()]

        result = generator.generate(
            boundary,
            max_area=1000.0,
            min_angle=20.0,
            streams=streams,
            refinement_zones=refinement_zones,
            points=points,
        )

        assert isinstance(result, MeshResult)

    def test_generate_from_shapely(self):
        """Test generate_from_shapely method."""
        generator = ConcreteMeshGenerator()

        # Create a mock Shapely polygon
        mock_polygon = MagicMock()
        mock_polygon.exterior.coords = [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
            (0.0, 0.0),  # Closing point
        ]
        mock_polygon.interiors = []

        result = generator.generate_from_shapely(mock_polygon, max_area=1000.0)

        assert isinstance(result, MeshResult)

    def test_generate_from_shapely_with_holes(self):
        """Test generate_from_shapely with holes."""
        generator = ConcreteMeshGenerator()

        # Create a mock Shapely polygon with a hole
        mock_polygon = MagicMock()
        mock_polygon.exterior.coords = [
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 2.0),
            (0.0, 2.0),
            (0.0, 0.0),
        ]

        mock_hole = MagicMock()
        mock_hole.coords = [
            (0.5, 0.5),
            (1.5, 0.5),
            (1.5, 1.5),
            (0.5, 1.5),
            (0.5, 0.5),
        ]
        mock_polygon.interiors = [mock_hole]

        result = generator.generate_from_shapely(mock_polygon)

        assert isinstance(result, MeshResult)


# =============================================================================
# Integration-style Tests
# =============================================================================


class TestMeshResultIntegration:
    """Integration-style tests for MeshResult."""

    def test_generate_and_convert(self):
        """Test generating mesh and converting to AppGrid."""
        generator = ConcreteMeshGenerator()
        boundary = MagicMock()

        mesh_result = generator.generate(boundary)
        grid = mesh_result.to_appgrid()

        assert grid.n_nodes == mesh_result.n_nodes
        assert grid.n_elements == mesh_result.n_elements

    def test_area_centroid_consistency(self, multi_triangle_mesh):
        """Test areas and centroids are consistent."""
        areas = multi_triangle_mesh.get_element_areas()
        centroids = multi_triangle_mesh.get_element_centroids()

        # All areas should be positive
        assert all(a > 0 for a in areas)

        # Centroids should be within node bounds
        x_min = multi_triangle_mesh.nodes[:, 0].min()
        x_max = multi_triangle_mesh.nodes[:, 0].max()
        y_min = multi_triangle_mesh.nodes[:, 1].min()
        y_max = multi_triangle_mesh.nodes[:, 1].max()

        for cx, cy in centroids:
            assert x_min <= cx <= x_max
            assert y_min <= cy <= y_max
