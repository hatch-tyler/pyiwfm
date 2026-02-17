"""Unit tests for Gmsh mesh generator wrapper."""

from __future__ import annotations

import numpy as np
import pytest

# Skip tests if gmsh is not installed
gmsh = pytest.importorskip("gmsh")

from pyiwfm.mesh_generation.constraints import (  # noqa: E402
    Boundary,
    PointConstraint,
    StreamConstraint,
)
from pyiwfm.mesh_generation.gmsh_wrapper import GmshMeshGenerator  # noqa: E402


class TestGmshMeshGenerator:
    """Tests for Gmsh mesh generator."""

    def test_generator_creation(self) -> None:
        """Test generator creation."""
        gen = GmshMeshGenerator()
        assert gen is not None

    def test_generator_element_type(self) -> None:
        """Test generator with different element types."""
        gen_tri = GmshMeshGenerator(element_type="triangle")
        assert gen_tri.element_type == "triangle"

        gen_quad = GmshMeshGenerator(element_type="quad")
        assert gen_quad.element_type == "quad"

        gen_mixed = GmshMeshGenerator(element_type="mixed")
        assert gen_mixed.element_type == "mixed"

    def test_generate_simple_square_triangles(self) -> None:
        """Test generating triangular mesh for simple square boundary."""
        gen = GmshMeshGenerator(element_type="triangle")

        boundary = Boundary(
            vertices=np.array(
                [
                    [0.0, 0.0],
                    [100.0, 0.0],
                    [100.0, 100.0],
                    [0.0, 100.0],
                ]
            )
        )

        result = gen.generate(boundary, max_area=500.0)

        assert result.n_nodes > 4
        assert result.n_elements > 0
        assert result.n_triangles > 0

    def test_generate_simple_square_quads(self) -> None:
        """Test generating quad mesh for simple square boundary."""
        gen = GmshMeshGenerator(element_type="quad")

        boundary = Boundary(
            vertices=np.array(
                [
                    [0.0, 0.0],
                    [100.0, 0.0],
                    [100.0, 100.0],
                    [0.0, 100.0],
                ]
            )
        )

        result = gen.generate(boundary, max_area=500.0)

        assert result.n_nodes > 4
        assert result.n_elements > 0
        # Should have some quads
        assert result.n_quads > 0

    def test_generate_with_hole(self) -> None:
        """Test generating mesh with interior hole."""
        gen = GmshMeshGenerator(element_type="triangle")

        outer = np.array(
            [
                [0.0, 0.0],
                [100.0, 0.0],
                [100.0, 100.0],
                [0.0, 100.0],
            ]
        )
        hole = np.array(
            [
                [40.0, 40.0],
                [60.0, 40.0],
                [60.0, 60.0],
                [40.0, 60.0],
            ]
        )

        boundary = Boundary(vertices=outer, holes=[hole])

        result = gen.generate(boundary, max_area=200.0)

        assert result.n_elements > 0
        # Verify no elements in hole (by checking centroids)
        centroids = result.get_element_centroids()
        for cx, cy in centroids:
            inside_hole = 40 < cx < 60 and 40 < cy < 60
            assert not inside_hole, "Element centroid inside hole"

    def test_generate_with_stream_constraint(self) -> None:
        """Test generating mesh with stream constraint."""
        gen = GmshMeshGenerator(element_type="triangle")

        boundary = Boundary(
            vertices=np.array(
                [
                    [0.0, 0.0],
                    [100.0, 0.0],
                    [100.0, 100.0],
                    [0.0, 100.0],
                ]
            )
        )

        stream = StreamConstraint(
            vertices=np.array(
                [
                    [10.0, 50.0],
                    [50.0, 50.0],
                    [90.0, 50.0],
                ]
            ),
            stream_id=1,
        )

        result = gen.generate(boundary, max_area=200.0, streams=[stream])

        assert result.n_elements > 0

    def test_generate_with_point_constraint(self) -> None:
        """Test generating mesh with fixed point constraint."""
        gen = GmshMeshGenerator(element_type="triangle")

        boundary = Boundary(
            vertices=np.array(
                [
                    [0.0, 0.0],
                    [100.0, 0.0],
                    [100.0, 100.0],
                    [0.0, 100.0],
                ]
            )
        )

        point = PointConstraint(x=50.0, y=50.0, marker=1)

        result = gen.generate(boundary, max_area=200.0, points=[point])

        assert result.n_elements > 0
        # Check that fixed point is in result
        found = False
        for x, y in result.nodes:
            if abs(x - 50.0) < 0.1 and abs(y - 50.0) < 0.1:
                found = True
                break
        assert found, "Fixed point not found in mesh"

    def test_generate_irregular_boundary(self) -> None:
        """Test generating mesh for irregular boundary."""
        gen = GmshMeshGenerator(element_type="triangle")

        # L-shaped boundary
        boundary = Boundary(
            vertices=np.array(
                [
                    [0.0, 0.0],
                    [100.0, 0.0],
                    [100.0, 50.0],
                    [50.0, 50.0],
                    [50.0, 100.0],
                    [0.0, 100.0],
                ]
            )
        )

        result = gen.generate(boundary, max_area=200.0)

        assert result.n_elements > 0

    def test_to_appgrid(self) -> None:
        """Test converting result to AppGrid."""
        gen = GmshMeshGenerator(element_type="triangle")

        boundary = Boundary(
            vertices=np.array(
                [
                    [0.0, 0.0],
                    [100.0, 0.0],
                    [100.0, 100.0],
                    [0.0, 100.0],
                ]
            )
        )

        result = gen.generate(boundary, max_area=500.0)
        grid = result.to_appgrid()

        assert grid.n_nodes == result.n_nodes
        assert grid.n_elements == result.n_elements

    def test_element_quality(self) -> None:
        """Test that generated elements have reasonable quality."""
        gen = GmshMeshGenerator(element_type="triangle")

        boundary = Boundary(
            vertices=np.array(
                [
                    [0.0, 0.0],
                    [100.0, 0.0],
                    [100.0, 100.0],
                    [0.0, 100.0],
                ]
            )
        )

        result = gen.generate(boundary, max_area=200.0, min_angle=20.0)

        areas = result.get_element_areas()

        # All areas should be positive
        assert np.all(areas > 0)
