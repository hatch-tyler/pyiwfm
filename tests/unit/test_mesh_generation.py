"""Unit tests for mesh generation classes."""

from __future__ import annotations

import numpy as np
import pytest

from pyiwfm.mesh_generation.constraints import (
    Boundary,
    BoundarySegment,
    PointConstraint,
    RefinementZone,
    StreamConstraint,
)
from pyiwfm.mesh_generation.generators import MeshGenerator, MeshResult


class TestBoundary:
    """Tests for boundary class."""

    def test_boundary_creation(self) -> None:
        """Test basic boundary creation."""
        vertices = np.array(
            [
                [0.0, 0.0],
                [100.0, 0.0],
                [100.0, 100.0],
                [0.0, 100.0],
            ]
        )
        boundary = Boundary(vertices=vertices)

        assert len(boundary.vertices) == 4
        assert boundary.n_vertices == 4

    def test_boundary_is_closed(self) -> None:
        """Test boundary closure check."""
        vertices = np.array(
            [
                [0.0, 0.0],
                [100.0, 0.0],
                [100.0, 100.0],
                [0.0, 100.0],
            ]
        )
        boundary = Boundary(vertices=vertices)

        assert boundary.is_closed

    def test_boundary_area(self) -> None:
        """Test boundary area calculation."""
        # Simple square 100x100
        vertices = np.array(
            [
                [0.0, 0.0],
                [100.0, 0.0],
                [100.0, 100.0],
                [0.0, 100.0],
            ]
        )
        boundary = Boundary(vertices=vertices)

        assert boundary.area == pytest.approx(10000.0)

    def test_boundary_centroid(self) -> None:
        """Test boundary centroid calculation."""
        vertices = np.array(
            [
                [0.0, 0.0],
                [100.0, 0.0],
                [100.0, 100.0],
                [0.0, 100.0],
            ]
        )
        boundary = Boundary(vertices=vertices)

        cx, cy = boundary.centroid
        assert cx == pytest.approx(50.0)
        assert cy == pytest.approx(50.0)

    def test_boundary_with_hole(self) -> None:
        """Test boundary with interior hole."""
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

        assert len(boundary.holes) == 1
        # Area should be outer - hole
        assert boundary.area == pytest.approx(10000.0 - 400.0)

    def test_boundary_segments(self) -> None:
        """Test getting boundary segments."""
        vertices = np.array(
            [
                [0.0, 0.0],
                [100.0, 0.0],
                [100.0, 100.0],
                [0.0, 100.0],
            ]
        )
        boundary = Boundary(vertices=vertices)

        segments = boundary.get_segments()
        assert len(segments) == 4


class TestBoundarySegment:
    """Tests for boundary segment class."""

    def test_segment_creation(self) -> None:
        """Test segment creation."""
        segment = BoundarySegment(
            start=np.array([0.0, 0.0]),
            end=np.array([100.0, 0.0]),
            marker=1,
        )

        assert segment.marker == 1
        np.testing.assert_array_equal(segment.start, [0.0, 0.0])
        np.testing.assert_array_equal(segment.end, [100.0, 0.0])

    def test_segment_length(self) -> None:
        """Test segment length calculation."""
        segment = BoundarySegment(
            start=np.array([0.0, 0.0]),
            end=np.array([100.0, 0.0]),
        )

        assert segment.length == pytest.approx(100.0)

    def test_segment_midpoint(self) -> None:
        """Test segment midpoint."""
        segment = BoundarySegment(
            start=np.array([0.0, 0.0]),
            end=np.array([100.0, 100.0]),
        )

        mx, my = segment.midpoint
        assert mx == pytest.approx(50.0)
        assert my == pytest.approx(50.0)


class TestStreamConstraint:
    """Tests for stream constraint class."""

    def test_stream_constraint_creation(self) -> None:
        """Test stream constraint creation."""
        vertices = np.array(
            [
                [10.0, 50.0],
                [30.0, 50.0],
                [50.0, 60.0],
                [70.0, 50.0],
                [90.0, 50.0],
            ]
        )
        stream = StreamConstraint(vertices=vertices, stream_id=1)

        assert stream.stream_id == 1
        assert stream.n_vertices == 5

    def test_stream_constraint_length(self) -> None:
        """Test stream constraint length."""
        vertices = np.array(
            [
                [0.0, 0.0],
                [50.0, 0.0],
                [100.0, 0.0],
            ]
        )
        stream = StreamConstraint(vertices=vertices)

        assert stream.length == pytest.approx(100.0)

    def test_stream_constraint_segments(self) -> None:
        """Test getting stream segments."""
        vertices = np.array(
            [
                [0.0, 0.0],
                [50.0, 0.0],
                [100.0, 0.0],
            ]
        )
        stream = StreamConstraint(vertices=vertices)

        segments = stream.get_segments()
        assert len(segments) == 2


class TestRefinementZone:
    """Tests for refinement zone class."""

    def test_refinement_zone_creation(self) -> None:
        """Test refinement zone creation."""
        zone = RefinementZone(
            center=(50.0, 50.0),
            radius=25.0,
            max_area=10.0,
        )

        assert zone.center == (50.0, 50.0)
        assert zone.radius == 25.0
        assert zone.max_area == 10.0

    def test_refinement_zone_contains(self) -> None:
        """Test point containment check."""
        zone = RefinementZone(
            center=(50.0, 50.0),
            radius=25.0,
            max_area=10.0,
        )

        assert zone.contains(50.0, 50.0)  # Center
        assert zone.contains(60.0, 50.0)  # Within radius
        assert not zone.contains(100.0, 100.0)  # Outside

    def test_refinement_zone_polygon(self) -> None:
        """Test rectangular refinement zone."""
        zone = RefinementZone(
            polygon=np.array(
                [
                    [40.0, 40.0],
                    [60.0, 40.0],
                    [60.0, 60.0],
                    [40.0, 60.0],
                ]
            ),
            max_area=5.0,
        )

        assert zone.max_area == 5.0
        assert zone.contains(50.0, 50.0)
        assert not zone.contains(30.0, 30.0)


class TestPointConstraint:
    """Tests for point constraint class."""

    def test_point_constraint_creation(self) -> None:
        """Test point constraint creation."""
        point = PointConstraint(
            x=50.0,
            y=50.0,
            marker=1,
        )

        assert point.x == 50.0
        assert point.y == 50.0
        assert point.marker == 1

    def test_point_constraint_as_array(self) -> None:
        """Test converting to array."""
        point = PointConstraint(x=50.0, y=75.0)

        arr = point.as_array()
        np.testing.assert_array_equal(arr, [50.0, 75.0])


class TestMeshResult:
    """Tests for mesh result class."""

    def test_mesh_result_creation(self) -> None:
        """Test mesh result creation."""
        nodes = np.array(
            [
                [0.0, 0.0],
                [100.0, 0.0],
                [100.0, 100.0],
                [0.0, 100.0],
                [50.0, 50.0],
            ]
        )
        elements = np.array(
            [
                [0, 1, 4],
                [1, 2, 4],
                [2, 3, 4],
                [3, 0, 4],
            ]
        )

        result = MeshResult(nodes=nodes, elements=elements)

        assert result.n_nodes == 5
        assert result.n_elements == 4

    def test_mesh_result_element_types(self) -> None:
        """Test mesh with mixed element types."""
        nodes = np.array(
            [
                [0.0, 0.0],
                [100.0, 0.0],
                [100.0, 100.0],
                [0.0, 100.0],
                [50.0, 0.0],
                [50.0, 100.0],
            ]
        )
        # Mix of triangles (with -1 padding) and quads
        elements = np.array(
            [
                [0, 4, 5, 3],  # quad
                [4, 1, 2, 5],  # quad
            ]
        )

        result = MeshResult(nodes=nodes, elements=elements)

        assert result.n_quads == 2
        assert result.n_triangles == 0

    def test_mesh_result_triangles(self) -> None:
        """Test mesh with triangles."""
        nodes = np.array(
            [
                [0.0, 0.0],
                [100.0, 0.0],
                [50.0, 100.0],
            ]
        )
        elements = np.array(
            [
                [0, 1, 2, -1],  # triangle with padding
            ]
        )

        result = MeshResult(nodes=nodes, elements=elements)

        assert result.n_triangles == 1
        assert result.n_quads == 0

    def test_mesh_result_to_appgrid(self) -> None:
        """Test converting mesh result to AppGrid."""
        nodes = np.array(
            [
                [0.0, 0.0],
                [100.0, 0.0],
                [100.0, 100.0],
                [0.0, 100.0],
            ]
        )
        elements = np.array(
            [
                [0, 1, 2, 3],  # quad
            ]
        )

        result = MeshResult(nodes=nodes, elements=elements)
        grid = result.to_appgrid()

        assert grid.n_nodes == 4
        assert grid.n_elements == 1


class TestMeshGenerator:
    """Tests for mesh generator base class."""

    def test_generator_abstract(self) -> None:
        """Test that MeshGenerator is abstract."""
        with pytest.raises(TypeError):
            MeshGenerator()  # type: ignore

    def test_generator_interface(self) -> None:
        """Test generator interface with concrete implementation."""

        class SimpleMeshGenerator(MeshGenerator):
            """Simple test implementation."""

            def generate(
                self,
                boundary: Boundary,
                max_area: float | None = None,
                min_angle: float | None = None,
                streams: list[StreamConstraint] | None = None,
                refinement_zones: list[RefinementZone] | None = None,
                points: list[PointConstraint] | None = None,
            ) -> MeshResult:
                # Return a simple mesh
                nodes = boundary.vertices.copy()
                # Create simple triangulation from centroid
                cx, cy = boundary.centroid
                nodes = np.vstack([nodes, [cx, cy]])
                n = len(boundary.vertices)
                elements = []
                for i in range(n):
                    elements.append([i, (i + 1) % n, n, -1])
                return MeshResult(
                    nodes=nodes,
                    elements=np.array(elements),
                )

        gen = SimpleMeshGenerator()

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

        result = gen.generate(boundary)

        assert result.n_nodes == 5
        assert result.n_elements == 4
