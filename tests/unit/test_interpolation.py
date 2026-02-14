"""
Comprehensive tests for pyiwfm.core.interpolation module.

Tests cover:
- Point-in-element location for triangles and quads
- Interpolation coefficient computation
- FEInterpolator class methods
- ParametricGrid class methods
- Edge cases and error handling
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from pyiwfm.core.mesh import AppGrid, Node, Element
from pyiwfm.core.interpolation import (
    point_in_element,
    find_containing_element,
    interpolation_coefficients,
    fe_interpolate_at_element,
    fe_interpolate,
    FEInterpolator,
    ParametricGrid,
    InterpolationResult,
    _xpoint,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def triangle_grid():
    """Create a simple triangular mesh."""
    nodes = {
        1: Node(id=1, x=0.0, y=0.0),
        2: Node(id=2, x=100.0, y=0.0),
        3: Node(id=3, x=50.0, y=100.0),
    }
    elements = {
        1: Element(id=1, vertices=(1, 2, 3)),
    }
    return AppGrid(nodes=nodes, elements=elements)


@pytest.fixture
def quad_grid():
    """Create a simple quadrilateral mesh (unit square)."""
    nodes = {
        1: Node(id=1, x=0.0, y=0.0),
        2: Node(id=2, x=100.0, y=0.0),
        3: Node(id=3, x=100.0, y=100.0),
        4: Node(id=4, x=0.0, y=100.0),
    }
    elements = {
        1: Element(id=1, vertices=(1, 2, 3, 4)),
    }
    return AppGrid(nodes=nodes, elements=elements)


@pytest.fixture
def multi_element_grid():
    """Create a mesh with multiple triangular elements."""
    nodes = {
        1: Node(id=1, x=0.0, y=0.0),
        2: Node(id=2, x=100.0, y=0.0),
        3: Node(id=3, x=200.0, y=0.0),
        4: Node(id=4, x=100.0, y=100.0),
        5: Node(id=5, x=50.0, y=50.0),
    }
    elements = {
        1: Element(id=1, vertices=(1, 2, 5)),
        2: Element(id=2, vertices=(2, 3, 4)),
        3: Element(id=3, vertices=(2, 4, 5)),
    }
    return AppGrid(nodes=nodes, elements=elements)


@pytest.fixture
def mixed_mesh():
    """Create a mesh with both triangles and quads."""
    nodes = {
        1: Node(id=1, x=0.0, y=0.0),
        2: Node(id=2, x=100.0, y=0.0),
        3: Node(id=3, x=200.0, y=0.0),
        4: Node(id=4, x=200.0, y=100.0),
        5: Node(id=5, x=100.0, y=100.0),
        6: Node(id=6, x=0.0, y=100.0),
        7: Node(id=7, x=50.0, y=150.0),
    }
    elements = {
        1: Element(id=1, vertices=(1, 2, 5, 6)),  # Quad
        2: Element(id=2, vertices=(2, 3, 4, 5)),  # Quad
        3: Element(id=3, vertices=(6, 5, 7)),     # Triangle (CCW order)
    }
    return AppGrid(nodes=nodes, elements=elements)


# =============================================================================
# XPoint Helper Tests
# =============================================================================


class TestXPoint:
    """Tests for the _xpoint helper function."""

    def test_vertical_line(self):
        """Test projection onto vertical line."""
        xx, yx = _xpoint(5.0, 0.0, 5.0, 10.0, 8.0, 5.0)
        assert xx == 5.0
        assert yx == 5.0

    def test_horizontal_line(self):
        """Test projection onto horizontal line."""
        xx, yx = _xpoint(0.0, 5.0, 10.0, 5.0, 5.0, 8.0)
        assert xx == 5.0
        assert yx == 5.0

    def test_diagonal_line(self):
        """Test projection onto diagonal line."""
        xx, yx = _xpoint(0.0, 0.0, 10.0, 10.0, 0.0, 10.0)
        # Point (0,10) projected onto y=x line should be (5,5)
        assert_allclose([xx, yx], [5.0, 5.0], rtol=1e-10)

    def test_point_on_line(self):
        """Test point already on line."""
        xx, yx = _xpoint(0.0, 0.0, 10.0, 10.0, 5.0, 5.0)
        assert_allclose([xx, yx], [5.0, 5.0], rtol=1e-10)


# =============================================================================
# Point In Element Tests
# =============================================================================


class TestPointInElement:
    """Tests for point_in_element function."""

    def test_point_inside_triangle(self):
        """Test point clearly inside triangle."""
        x = np.array([0.0, 100.0, 50.0])
        y = np.array([0.0, 0.0, 100.0])

        # Centroid should be inside
        assert point_in_element(x, y, 50.0, 33.0)

    def test_point_outside_triangle(self):
        """Test point clearly outside triangle."""
        x = np.array([0.0, 100.0, 50.0])
        y = np.array([0.0, 0.0, 100.0])

        assert not point_in_element(x, y, 200.0, 200.0)
        assert not point_in_element(x, y, -10.0, 50.0)

    def test_point_at_vertex(self):
        """Test point exactly at a vertex."""
        x = np.array([0.0, 100.0, 50.0])
        y = np.array([0.0, 0.0, 100.0])

        assert point_in_element(x, y, 0.0, 0.0)
        assert point_in_element(x, y, 100.0, 0.0)
        assert point_in_element(x, y, 50.0, 100.0)

    def test_point_on_edge(self):
        """Test point on edge of triangle."""
        x = np.array([0.0, 100.0, 50.0])
        y = np.array([0.0, 0.0, 100.0])

        # On bottom edge
        assert point_in_element(x, y, 50.0, 0.0)

    def test_point_inside_quad(self):
        """Test point inside quadrilateral."""
        x = np.array([0.0, 100.0, 100.0, 0.0])
        y = np.array([0.0, 0.0, 100.0, 100.0])

        assert point_in_element(x, y, 50.0, 50.0)

    def test_point_outside_quad(self):
        """Test point outside quadrilateral."""
        x = np.array([0.0, 100.0, 100.0, 0.0])
        y = np.array([0.0, 0.0, 100.0, 100.0])

        assert not point_in_element(x, y, 150.0, 50.0)

    def test_point_at_quad_vertex(self):
        """Test point at quad vertex."""
        x = np.array([0.0, 100.0, 100.0, 0.0])
        y = np.array([0.0, 0.0, 100.0, 100.0])

        assert point_in_element(x, y, 0.0, 0.0)
        assert point_in_element(x, y, 100.0, 100.0)


# =============================================================================
# Find Containing Element Tests
# =============================================================================


class TestFindContainingElement:
    """Tests for find_containing_element function."""

    def test_find_in_single_triangle(self, triangle_grid):
        """Test finding element in single triangle mesh."""
        elem_id = find_containing_element(triangle_grid, 50.0, 33.0)
        assert elem_id == 1

    def test_not_found(self, triangle_grid):
        """Test point outside mesh returns 0."""
        elem_id = find_containing_element(triangle_grid, 200.0, 200.0)
        assert elem_id == 0

    def test_find_in_multi_element(self, multi_element_grid):
        """Test finding correct element in multi-element mesh."""
        # Point in element 1
        elem_id = find_containing_element(multi_element_grid, 30.0, 20.0)
        assert elem_id == 1

        # Point in element 2
        elem_id = find_containing_element(multi_element_grid, 150.0, 40.0)
        assert elem_id == 2

    def test_find_in_quad(self, quad_grid):
        """Test finding element in quad mesh."""
        elem_id = find_containing_element(quad_grid, 50.0, 50.0)
        assert elem_id == 1

    def test_find_in_mixed_mesh(self, mixed_mesh):
        """Test finding elements in mixed mesh."""
        # Point in quad element 1
        elem_id = find_containing_element(mixed_mesh, 50.0, 50.0)
        assert elem_id == 1

        # Point in triangle element 3
        elem_id = find_containing_element(mixed_mesh, 50.0, 120.0)
        assert elem_id == 3


# =============================================================================
# Interpolation Coefficients Tests
# =============================================================================


class TestInterpolationCoefficients:
    """Tests for interpolation_coefficients function."""

    def test_triangle_centroid(self):
        """Test coefficients at triangle centroid are equal."""
        x = np.array([0.0, 100.0, 50.0])
        y = np.array([0.0, 0.0, 100.0])

        # Centroid
        xp, yp = 50.0, 100.0 / 3.0
        coeffs = interpolation_coefficients(3, xp, yp, x, y)

        assert len(coeffs) == 3
        assert_allclose(coeffs, [1/3, 1/3, 1/3], rtol=0.01)
        assert_allclose(sum(coeffs), 1.0, rtol=1e-10)

    def test_triangle_at_vertex(self):
        """Test coefficients at triangle vertex."""
        x = np.array([0.0, 100.0, 50.0])
        y = np.array([0.0, 0.0, 100.0])

        # At vertex 1
        coeffs = interpolation_coefficients(3, 0.0, 0.0, x, y)
        assert_allclose(coeffs, [1.0, 0.0, 0.0], rtol=1e-5)

        # At vertex 2
        coeffs = interpolation_coefficients(3, 100.0, 0.0, x, y)
        assert_allclose(coeffs, [0.0, 1.0, 0.0], rtol=1e-5)

        # At vertex 3
        coeffs = interpolation_coefficients(3, 50.0, 100.0, x, y)
        assert_allclose(coeffs, [0.0, 0.0, 1.0], rtol=1e-5)

    def test_triangle_midpoint(self):
        """Test coefficients at triangle edge midpoint."""
        x = np.array([0.0, 100.0, 50.0])
        y = np.array([0.0, 0.0, 100.0])

        # Midpoint of edge 1-2
        coeffs = interpolation_coefficients(3, 50.0, 0.0, x, y)
        assert_allclose(coeffs[0], 0.5, rtol=1e-5)
        assert_allclose(coeffs[1], 0.5, rtol=1e-5)
        assert_allclose(coeffs[2], 0.0, atol=1e-10)

    def test_quad_center(self):
        """Test coefficients at quad center."""
        x = np.array([0.0, 100.0, 100.0, 0.0])
        y = np.array([0.0, 0.0, 100.0, 100.0])

        coeffs = interpolation_coefficients(4, 50.0, 50.0, x, y)

        assert len(coeffs) == 4
        assert_allclose(coeffs, [0.25, 0.25, 0.25, 0.25], rtol=0.01)
        assert_allclose(sum(coeffs), 1.0, rtol=1e-10)

    def test_quad_at_vertex(self):
        """Test coefficients at quad vertex."""
        x = np.array([0.0, 100.0, 100.0, 0.0])
        y = np.array([0.0, 0.0, 100.0, 100.0])

        # At vertex 1 (0,0)
        coeffs = interpolation_coefficients(4, 0.0, 0.0, x, y)
        assert_allclose(coeffs, [1.0, 0.0, 0.0, 0.0], rtol=1e-5)

        # At vertex 3 (100,100)
        coeffs = interpolation_coefficients(4, 100.0, 100.0, x, y)
        assert_allclose(coeffs, [0.0, 0.0, 1.0, 0.0], rtol=1e-5)

    def test_quad_on_edge(self):
        """Test coefficients at quad edge midpoint."""
        x = np.array([0.0, 100.0, 100.0, 0.0])
        y = np.array([0.0, 0.0, 100.0, 100.0])

        # Midpoint of bottom edge
        coeffs = interpolation_coefficients(4, 50.0, 0.0, x, y)
        assert_allclose(coeffs[0], 0.5, rtol=1e-5)
        assert_allclose(coeffs[1], 0.5, rtol=1e-5)
        assert_allclose(coeffs[2], 0.0, atol=1e-10)
        assert_allclose(coeffs[3], 0.0, atol=1e-10)

    def test_coefficients_sum_to_one(self):
        """Test that coefficients sum to 1 for various points."""
        x = np.array([0.0, 100.0, 50.0])
        y = np.array([0.0, 0.0, 100.0])

        # Test multiple points
        test_points = [
            (25.0, 25.0),
            (50.0, 10.0),
            (75.0, 30.0),
        ]

        for xp, yp in test_points:
            coeffs = interpolation_coefficients(3, xp, yp, x, y)
            assert_allclose(sum(coeffs), 1.0, rtol=1e-10)


# =============================================================================
# FE Interpolate Tests
# =============================================================================


class TestFEInterpolateAtElement:
    """Tests for fe_interpolate_at_element function."""

    def test_triangle_interpolation(self, triangle_grid):
        """Test interpolation at triangle element."""
        coeffs = fe_interpolate_at_element(triangle_grid, 1, 50.0, 33.0)

        assert len(coeffs) == 3
        assert_allclose(sum(coeffs), 1.0, rtol=1e-10)

    def test_quad_interpolation(self, quad_grid):
        """Test interpolation at quad element."""
        coeffs = fe_interpolate_at_element(quad_grid, 1, 50.0, 50.0)

        assert len(coeffs) == 4
        assert_allclose(sum(coeffs), 1.0, rtol=1e-10)

    def test_invalid_element_raises(self, triangle_grid):
        """Test that invalid element ID raises KeyError."""
        with pytest.raises(KeyError):
            fe_interpolate_at_element(triangle_grid, 999, 50.0, 33.0)


class TestFEInterpolate:
    """Tests for fe_interpolate function."""

    def test_point_found(self, triangle_grid):
        """Test interpolation when point is found."""
        result = fe_interpolate(triangle_grid, 50.0, 33.0)

        assert result.found
        assert result.element_id == 1
        assert len(result.node_ids) == 3
        assert len(result.coefficients) == 3
        assert_allclose(sum(result.coefficients), 1.0, rtol=1e-10)

    def test_point_not_found(self, triangle_grid):
        """Test interpolation when point is not found."""
        result = fe_interpolate(triangle_grid, 200.0, 200.0)

        assert not result.found
        assert result.element_id == 0
        assert result.node_ids == ()
        assert len(result.coefficients) == 0


class TestInterpolationResult:
    """Tests for InterpolationResult class."""

    def test_found_property(self):
        """Test found property."""
        result = InterpolationResult(
            element_id=1,
            node_ids=(1, 2, 3),
            coefficients=np.array([0.33, 0.33, 0.34]),
        )
        assert result.found

        result = InterpolationResult(
            element_id=0,
            node_ids=(),
            coefficients=np.array([]),
        )
        assert not result.found

    def test_interpolate_value(self):
        """Test value interpolation."""
        result = InterpolationResult(
            element_id=1,
            node_ids=(1, 2, 3),
            coefficients=np.array([0.5, 0.3, 0.2]),
        )
        values = {1: 100.0, 2: 200.0, 3: 300.0}

        interpolated = result.interpolate(values)
        expected = 0.5 * 100.0 + 0.3 * 200.0 + 0.2 * 300.0
        assert interpolated == expected

    def test_interpolate_not_found_raises(self):
        """Test that interpolating when not found raises error."""
        result = InterpolationResult(
            element_id=0,
            node_ids=(),
            coefficients=np.array([]),
        )

        with pytest.raises(ValueError, match="not found"):
            result.interpolate({1: 100.0})


# =============================================================================
# FEInterpolator Class Tests
# =============================================================================


class TestFEInterpolator:
    """Tests for FEInterpolator class."""

    def test_init(self, triangle_grid):
        """Test interpolator initialization."""
        interp = FEInterpolator(triangle_grid)

        assert interp.grid is triangle_grid
        assert len(interp._elem_ids) == 1

    def test_point_in_element(self, triangle_grid):
        """Test point_in_element method."""
        interp = FEInterpolator(triangle_grid)

        assert interp.point_in_element(50.0, 33.0, 1)
        assert not interp.point_in_element(200.0, 200.0, 1)

    def test_find_element(self, multi_element_grid):
        """Test find_element method."""
        interp = FEInterpolator(multi_element_grid)

        assert interp.find_element(30.0, 20.0) == 1
        assert interp.find_element(150.0, 40.0) == 2
        assert interp.find_element(999.0, 999.0) == 0

    def test_interpolate(self, triangle_grid):
        """Test interpolate method."""
        interp = FEInterpolator(triangle_grid)

        elem_id, node_ids, coeffs = interp.interpolate(50.0, 33.0)

        assert elem_id == 1
        assert len(node_ids) == 3
        assert_allclose(sum(coeffs), 1.0, rtol=1e-10)

    def test_interpolate_not_found(self, triangle_grid):
        """Test interpolate when point not found."""
        interp = FEInterpolator(triangle_grid)

        elem_id, node_ids, coeffs = interp.interpolate(200.0, 200.0)

        assert elem_id == 0
        assert node_ids == ()
        assert len(coeffs) == 0

    def test_interpolate_at_element(self, quad_grid):
        """Test interpolate_at_element method."""
        interp = FEInterpolator(quad_grid)

        coeffs = interp.interpolate_at_element(50.0, 50.0, 1)

        assert len(coeffs) == 4
        assert_allclose(coeffs, [0.25, 0.25, 0.25, 0.25], rtol=0.01)

    def test_interpolate_value(self, triangle_grid):
        """Test interpolate_value method."""
        interp = FEInterpolator(triangle_grid)
        values = {1: 0.0, 2: 100.0, 3: 50.0}

        # At vertex 2, should get 100.0
        result = interp.interpolate_value(100.0, 0.0, values)
        assert_allclose(result, 100.0, rtol=1e-5)

        # At centroid, should get weighted average
        xc, yc = 50.0, 100.0 / 3.0
        result = interp.interpolate_value(xc, yc, values)
        assert result is not None
        assert 0.0 <= result <= 100.0

    def test_interpolate_value_not_found(self, triangle_grid):
        """Test interpolate_value returns None when not found."""
        interp = FEInterpolator(triangle_grid)
        values = {1: 0.0, 2: 100.0, 3: 50.0}

        result = interp.interpolate_value(200.0, 200.0, values)
        assert result is None

    def test_interpolate_array(self, triangle_grid):
        """Test interpolate_array method."""
        interp = FEInterpolator(triangle_grid)
        values = np.array([0.0, 100.0, 50.0])  # Node 1, 2, 3 values

        # At vertex 2 (node 2 = index 1)
        result = interp.interpolate_array(100.0, 0.0, values)
        assert_allclose(result, 100.0, rtol=1e-5)

    def test_interpolate_points(self, triangle_grid):
        """Test interpolate_points method."""
        interp = FEInterpolator(triangle_grid)
        values = {1: 0.0, 2: 100.0, 3: 50.0}

        points = np.array([
            [50.0, 33.0],   # Inside
            [0.0, 0.0],     # At vertex 1
            [200.0, 200.0], # Outside
        ])

        results = interp.interpolate_points(points, values)

        assert len(results) == 3
        assert not np.isnan(results[0])  # Inside
        assert_allclose(results[1], 0.0, rtol=1e-5)  # At vertex 1
        assert np.isnan(results[2])  # Outside


# =============================================================================
# ParametricGrid Tests
# =============================================================================


class TestParametricGrid:
    """Tests for ParametricGrid class."""

    def test_init(self, triangle_grid):
        """Test parametric grid initialization."""
        pgrid = ParametricGrid(triangle_grid, n_layers=3, n_params=2)

        assert pgrid.n_layers == 3
        assert pgrid.n_params == 2
        assert pgrid._values.shape == (3, 3, 2)  # 3 nodes, 3 layers, 2 params

    def test_set_get_value(self, triangle_grid):
        """Test setting and getting values."""
        pgrid = ParametricGrid(triangle_grid, n_layers=2, n_params=2)

        pgrid.set_value(node_id=1, layer=0, param=0, value=100.0)
        pgrid.set_value(node_id=1, layer=1, param=1, value=200.0)

        assert pgrid.get_value(1, 0, 0) == 100.0
        assert pgrid.get_value(1, 1, 1) == 200.0

    def test_set_layer_values(self, triangle_grid):
        """Test setting values for a layer."""
        pgrid = ParametricGrid(triangle_grid, n_layers=2, n_params=2)

        values = {1: 10.0, 2: 20.0, 3: 30.0}
        pgrid.set_layer_values(layer=0, param=0, values=values)

        assert pgrid.get_value(1, 0, 0) == 10.0
        assert pgrid.get_value(2, 0, 0) == 20.0
        assert pgrid.get_value(3, 0, 0) == 30.0

    def test_get_layer_values(self, triangle_grid):
        """Test getting values for a layer."""
        pgrid = ParametricGrid(triangle_grid, n_layers=2, n_params=2)

        pgrid.set_value(1, 0, 0, 10.0)
        pgrid.set_value(2, 0, 0, 20.0)
        pgrid.set_value(3, 0, 0, 30.0)

        values = pgrid.get_layer_values(layer=0, param=0)

        assert values == {1: 10.0, 2: 20.0, 3: 30.0}

    def test_interpolate(self, triangle_grid):
        """Test interpolation."""
        pgrid = ParametricGrid(triangle_grid, n_layers=2, n_params=1)

        # Set nodal values
        pgrid.set_value(1, 0, 0, 0.0)
        pgrid.set_value(2, 0, 0, 100.0)
        pgrid.set_value(3, 0, 0, 50.0)

        # Interpolate at vertex 2
        result = pgrid.interpolate(100.0, 0.0, layer=0, param=0)
        assert_allclose(result, 100.0, rtol=1e-5)

        # Interpolate inside
        result = pgrid.interpolate(50.0, 33.0, layer=0, param=0)
        assert result is not None
        assert 0.0 <= result <= 100.0

    def test_interpolate_not_found(self, triangle_grid):
        """Test interpolate returns None when point not found."""
        pgrid = ParametricGrid(triangle_grid, n_layers=2, n_params=1)

        result = pgrid.interpolate(200.0, 200.0, layer=0, param=0)
        assert result is None

    def test_interpolate_all_params(self, triangle_grid):
        """Test interpolating all parameters."""
        pgrid = ParametricGrid(triangle_grid, n_layers=1, n_params=3)

        # Set values for all parameters
        for param in range(3):
            for node_id in [1, 2, 3]:
                pgrid.set_value(node_id, 0, param, float(param * 10 + node_id))

        result = pgrid.interpolate_all_params(50.0, 33.0, layer=0)

        assert result is not None
        assert len(result) == 3

    def test_interpolate_all_layers(self, triangle_grid):
        """Test interpolating all layers."""
        pgrid = ParametricGrid(triangle_grid, n_layers=3, n_params=1)

        # Set values for all layers
        for layer in range(3):
            for node_id in [1, 2, 3]:
                pgrid.set_value(node_id, layer, 0, float(layer * 100))

        result = pgrid.interpolate_all_layers(50.0, 33.0, param=0)

        assert result is not None
        assert len(result) == 3
        assert_allclose(result, [0.0, 100.0, 200.0], rtol=1e-5)

    def test_interpolate_all_layers_not_found(self, triangle_grid):
        """Test interpolate_all_layers returns None when not found."""
        pgrid = ParametricGrid(triangle_grid, n_layers=2, n_params=1)

        result = pgrid.interpolate_all_layers(200.0, 200.0, param=0)
        assert result is None


# =============================================================================
# Edge Cases and Numerical Precision Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and numerical precision."""

    def test_very_small_triangle(self):
        """Test with very small triangle."""
        nodes = {
            1: Node(id=1, x=0.0, y=0.0),
            2: Node(id=2, x=0.001, y=0.0),
            3: Node(id=3, x=0.0005, y=0.001),
        }
        elements = {1: Element(id=1, vertices=(1, 2, 3))}
        grid = AppGrid(nodes=nodes, elements=elements)

        interp = FEInterpolator(grid)
        elem_id = interp.find_element(0.0005, 0.0003)

        assert elem_id == 1

    def test_very_large_triangle(self):
        """Test with very large triangle."""
        nodes = {
            1: Node(id=1, x=0.0, y=0.0),
            2: Node(id=2, x=1e6, y=0.0),
            3: Node(id=3, x=5e5, y=1e6),
        }
        elements = {1: Element(id=1, vertices=(1, 2, 3))}
        grid = AppGrid(nodes=nodes, elements=elements)

        interp = FEInterpolator(grid)
        elem_id = interp.find_element(5e5, 3e5)

        assert elem_id == 1

    def test_skewed_quad(self):
        """Test with non-rectangular quad."""
        nodes = {
            1: Node(id=1, x=0.0, y=0.0),
            2: Node(id=2, x=100.0, y=10.0),
            3: Node(id=3, x=90.0, y=110.0),
            4: Node(id=4, x=-10.0, y=100.0),
        }
        elements = {1: Element(id=1, vertices=(1, 2, 3, 4))}
        grid = AppGrid(nodes=nodes, elements=elements)

        interp = FEInterpolator(grid)

        # Approximate center
        elem_id = interp.find_element(45.0, 55.0)
        assert elem_id == 1

        coeffs = interp.interpolate_at_element(45.0, 55.0, 1)
        assert_allclose(sum(coeffs), 1.0, rtol=1e-5)

    def test_coincident_points(self, triangle_grid):
        """Test interpolation at exact node locations."""
        interp = FEInterpolator(triangle_grid)

        # Test at each vertex
        for node in triangle_grid.nodes.values():
            elem_id, node_ids, coeffs = interp.interpolate(node.x, node.y)
            assert elem_id == 1
            assert_allclose(sum(coeffs), 1.0, rtol=1e-10)

    def test_point_very_close_to_edge(self):
        """Test point very close to but inside edge."""
        nodes = {
            1: Node(id=1, x=0.0, y=0.0),
            2: Node(id=2, x=100.0, y=0.0),
            3: Node(id=3, x=50.0, y=100.0),
        }
        elements = {1: Element(id=1, vertices=(1, 2, 3))}
        grid = AppGrid(nodes=nodes, elements=elements)

        interp = FEInterpolator(grid)

        # Point very close to bottom edge, slightly inside
        elem_id = interp.find_element(50.0, 0.001)
        assert elem_id == 1

    def test_empty_grid(self):
        """Test with grid having no elements."""
        nodes = {1: Node(id=1, x=0.0, y=0.0)}
        grid = AppGrid(nodes=nodes, elements={})

        interp = FEInterpolator(grid)
        elem_id = interp.find_element(0.0, 0.0)

        assert elem_id == 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for interpolation workflow."""

    def test_hydrograph_interpolation(self, mixed_mesh):
        """Test interpolating groundwater head values like IWFM."""
        interp = FEInterpolator(mixed_mesh)

        # Set up nodal head values
        heads = {
            1: 100.0,
            2: 95.0,
            3: 90.0,
            4: 88.0,
            5: 92.0,
            6: 98.0,
            7: 96.0,
        }

        # Observation well locations
        wells = [
            (50.0, 50.0),    # In quad 1
            (150.0, 50.0),   # In quad 2
            (50.0, 120.0),   # In triangle 3
        ]

        for xp, yp in wells:
            value = interp.interpolate_value(xp, yp, heads)
            assert value is not None
            assert 88.0 <= value <= 100.0

    def test_parametric_aquifer_properties(self, multi_element_grid):
        """Test parametric grid for aquifer properties like IWFM."""
        # Create parametric grid with 2 layers and 2 parameters
        # (e.g., hydraulic conductivity and storage coefficient)
        pgrid = ParametricGrid(multi_element_grid, n_layers=2, n_params=2)

        # Layer 0: shallow aquifer
        # Param 0: Kh, Param 1: Ss
        pgrid.set_layer_values(0, 0, {1: 10.0, 2: 15.0, 3: 20.0, 4: 18.0, 5: 12.0})
        pgrid.set_layer_values(0, 1, {1: 1e-4, 2: 1e-4, 3: 1e-4, 4: 1e-4, 5: 1e-4})

        # Layer 1: deep aquifer
        pgrid.set_layer_values(1, 0, {1: 5.0, 2: 8.0, 3: 10.0, 4: 9.0, 5: 6.0})
        pgrid.set_layer_values(1, 1, {1: 1e-5, 2: 1e-5, 3: 1e-5, 4: 1e-5, 5: 1e-5})

        # Interpolate at a point
        xp, yp = 75.0, 30.0

        # Get all properties at this location
        shallow_props = pgrid.interpolate_all_params(xp, yp, layer=0)
        deep_props = pgrid.interpolate_all_params(xp, yp, layer=1)

        assert shallow_props is not None
        assert deep_props is not None
        assert shallow_props[0] > deep_props[0]  # Shallow Kh > Deep Kh
        assert shallow_props[1] > deep_props[1]  # Shallow Ss > Deep Ss

    def test_coefficient_conservation(self, mixed_mesh):
        """Test that interpolation conserves values at nodes."""
        interp = FEInterpolator(mixed_mesh)

        # Define a linear function: z = x + 2*y
        def linear_func(x, y):
            return x + 2 * y

        # Set nodal values
        values = {
            nid: linear_func(node.x, node.y)
            for nid, node in mixed_mesh.nodes.items()
        }

        # Test at various points - should exactly reproduce linear function
        test_points = [
            (25.0, 25.0),
            (75.0, 75.0),
            (150.0, 50.0),
        ]

        for xp, yp in test_points:
            interpolated = interp.interpolate_value(xp, yp, values)
            if interpolated is not None:
                expected = linear_func(xp, yp)
                assert_allclose(interpolated, expected, rtol=0.01)
