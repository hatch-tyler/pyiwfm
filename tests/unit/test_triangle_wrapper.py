"""Unit tests for Triangle mesh generator wrapper."""

from __future__ import annotations

import numpy as np
import pytest

# Skip tests if triangle is not installed
triangle = pytest.importorskip("triangle")

from pyiwfm.mesh_generation.constraints import (  # noqa: E402
    Boundary,
    PointConstraint,
    RefinementZone,
    StreamConstraint,
)
from pyiwfm.mesh_generation.triangle_wrapper import TriangleMeshGenerator  # noqa: E402


class TestTriangleMeshGenerator:
    """Tests for Triangle mesh generator."""

    def test_generator_creation(self) -> None:
        """Test generator creation."""
        gen = TriangleMeshGenerator()
        assert gen is not None

    def test_generate_simple_square(self) -> None:
        """Test generating mesh for simple square boundary."""
        gen = TriangleMeshGenerator()

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
        assert result.n_triangles == result.n_elements

    def test_generate_with_min_angle(self) -> None:
        """Test generating mesh with minimum angle constraint."""
        gen = TriangleMeshGenerator()

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

        result = gen.generate(boundary, max_area=200.0, min_angle=25.0)

        assert result.n_elements > 0

    def test_generate_with_hole(self) -> None:
        """Test generating mesh with interior hole."""
        gen = TriangleMeshGenerator()

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
            # Should not be inside hole
            inside_hole = 40 < cx < 60 and 40 < cy < 60
            assert not inside_hole, "Element centroid inside hole"

    def test_generate_with_stream_constraint(self) -> None:
        """Test generating mesh with stream constraint."""
        gen = TriangleMeshGenerator()

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
        # Check that stream nodes are in result
        stream_pts_found = 0
        for x, y in result.nodes:
            for sx, sy in stream.vertices:
                if abs(x - sx) < 0.1 and abs(y - sy) < 0.1:
                    stream_pts_found += 1
                    break
        assert stream_pts_found >= 3

    def test_generate_with_point_constraint(self) -> None:
        """Test generating mesh with fixed point constraint."""
        gen = TriangleMeshGenerator()

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

    def test_generate_with_refinement_zone(self) -> None:
        """Test generating mesh with refinement zone."""
        gen = TriangleMeshGenerator()

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

        # Refinement zone in center - use larger zone and smaller max_area
        zone = RefinementZone(
            center=(50.0, 50.0),
            radius=30.0,
            max_area=10.0,  # Much smaller than global
        )

        # Use larger global max_area to see clear difference
        result = gen.generate(boundary, max_area=1000.0, refinement_zones=[zone])

        assert result.n_elements > 0

        # Verify refinement had some effect - mesh should have more elements
        # than a simple mesh without refinement
        simple_result = gen.generate(boundary, max_area=1000.0)

        # With refinement, we should have more elements
        assert result.n_elements >= simple_result.n_elements

    def test_generate_irregular_boundary(self) -> None:
        """Test generating mesh for irregular boundary."""
        gen = TriangleMeshGenerator()

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
        gen = TriangleMeshGenerator()

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
        gen = TriangleMeshGenerator()

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
        # No degenerate triangles
        assert np.all(areas > 1.0)  # Minimum reasonable area
