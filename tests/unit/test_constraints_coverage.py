"""Tests for mesh_generation/constraints.py geometry calculations.

Covers:
- BoundarySegment.length, midpoint
- Boundary.area (with/without holes), centroid, get_segments, get_hole_points
- StreamConstraint.length, resample (short and normal)
- RefinementZone creation and contains()
- PointConstraint.as_array()
"""

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


class TestSegmentLength:
    """Test BoundarySegment.length."""

    def test_segment_length(self) -> None:
        """Compute segment length."""
        seg = BoundarySegment(
            start=np.array([0.0, 0.0]),
            end=np.array([3.0, 4.0]),
        )
        assert seg.length == pytest.approx(5.0)

    def test_segment_identical_points(self) -> None:
        """Zero-length segment."""
        seg = BoundarySegment(
            start=np.array([5.0, 5.0]),
            end=np.array([5.0, 5.0]),
        )
        assert seg.length == pytest.approx(0.0)


class TestSegmentMidpoint:
    """Test BoundarySegment.midpoint."""

    def test_segment_midpoint(self) -> None:
        """Compute segment midpoint."""
        seg = BoundarySegment(
            start=np.array([0.0, 0.0]),
            end=np.array([10.0, 20.0]),
        )
        mx, my = seg.midpoint
        assert mx == pytest.approx(5.0)
        assert my == pytest.approx(10.0)


class TestBoundaryArea:
    """Test Boundary.area."""

    def test_boundary_area_simple(self) -> None:
        """Triangle area = 0.5 * base * height."""
        verts = np.array([
            [0.0, 0.0],
            [4.0, 0.0],
            [0.0, 3.0],
        ])
        b = Boundary(vertices=verts)
        assert b.area == pytest.approx(6.0)

    def test_boundary_area_square(self) -> None:
        """Square area = side^2."""
        verts = np.array([
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
        ])
        b = Boundary(vertices=verts)
        assert b.area == pytest.approx(100.0)

    def test_boundary_area_with_holes(self) -> None:
        """Area subtracting holes."""
        outer = np.array([
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
        ])
        hole = np.array([
            [2.0, 2.0],
            [4.0, 2.0],
            [4.0, 4.0],
            [2.0, 4.0],
        ])
        b = Boundary(vertices=outer, holes=[hole])
        # 100 - 4 = 96
        assert b.area == pytest.approx(96.0)


class TestBoundaryCentroid:
    """Test Boundary.centroid."""

    def test_boundary_centroid(self) -> None:
        """Centroid of rectangle."""
        verts = np.array([
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 6.0],
            [0.0, 6.0],
        ])
        b = Boundary(vertices=verts)
        cx, cy = b.centroid
        assert cx == pytest.approx(5.0)
        assert cy == pytest.approx(3.0)


class TestBoundaryGetSegments:
    """Test Boundary.get_segments."""

    def test_boundary_get_segments(self) -> None:
        """Segment extraction."""
        verts = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ])
        b = Boundary(vertices=verts)
        segments = b.get_segments()
        assert len(segments) == 3
        assert isinstance(segments[0], BoundarySegment)

    def test_boundary_get_segments_with_markers(self) -> None:
        """Segment markers from boundary markers."""
        verts = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ])
        markers = np.array([10, 20, 30], dtype=np.int32)
        b = Boundary(vertices=verts, markers=markers)
        segments = b.get_segments()
        assert segments[0].marker == 10
        assert segments[1].marker == 20
        assert segments[2].marker == 30


class TestBoundaryGetHolePoints:
    """Test Boundary.get_hole_points."""

    def test_boundary_get_hole_points(self) -> None:
        """Hole point extraction returns centroids."""
        outer = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        hole = np.array([[2, 2], [4, 2], [4, 4], [2, 4]], dtype=float)
        b = Boundary(vertices=outer, holes=[hole])
        points = b.get_hole_points()
        assert len(points) == 1
        hx, hy = points[0]
        assert hx == pytest.approx(3.0)
        assert hy == pytest.approx(3.0)


class TestStreamConstraintLength:
    """Test StreamConstraint.length."""

    def test_stream_length(self) -> None:
        """Stream length from vertices."""
        verts = np.array([
            [0.0, 0.0],
            [3.0, 4.0],
            [6.0, 4.0],
        ])
        sc = StreamConstraint(vertices=verts)
        # 5.0 + 3.0 = 8.0
        assert sc.length == pytest.approx(8.0)

    def test_stream_length_single_vertex(self) -> None:
        """Single vertex -> length 0."""
        sc = StreamConstraint(vertices=np.array([[0.0, 0.0]]))
        assert sc.length == pytest.approx(0.0)


class TestStreamResample:
    """Test StreamConstraint.resample."""

    def test_stream_resample_short(self) -> None:
        """Length <= spacing -> original vertices returned."""
        verts = np.array([[0.0, 0.0], [1.0, 0.0]])
        sc = StreamConstraint(vertices=verts, stream_id=5)
        resampled = sc.resample(spacing=10.0)
        assert resampled.n_vertices == 2
        assert resampled.stream_id == 5

    def test_stream_resample_normal(self) -> None:
        """Normal resampling produces more points."""
        verts = np.array([[0.0, 0.0], [10.0, 0.0]])
        sc = StreamConstraint(vertices=verts, stream_id=1)
        resampled = sc.resample(spacing=3.0)
        # Should have ~4 points (0, 3, 6, 9 + endpoint 10)
        assert resampled.n_vertices >= 4
        # First and last should match
        np.testing.assert_array_almost_equal(resampled.vertices[0], [0, 0])
        np.testing.assert_array_almost_equal(resampled.vertices[-1], [10, 0])

    def test_stream_resample_single_vertex(self) -> None:
        """Single vertex -> returned as-is."""
        sc = StreamConstraint(vertices=np.array([[5.0, 5.0]]))
        resampled = sc.resample(spacing=1.0)
        assert resampled.n_vertices == 1


class TestRefinementZone:
    """Test RefinementZone creation and contains()."""

    def test_refinement_zone_circle(self) -> None:
        """Circular zone creation and contains check."""
        zone = RefinementZone(center=(5.0, 5.0), radius=3.0, max_area=10.0)
        assert zone.contains(5.0, 5.0)  # center
        assert zone.contains(7.0, 5.0)  # on edge
        assert not zone.contains(10.0, 10.0)  # outside

    def test_refinement_zone_polygon(self) -> None:
        """Polygon zone creation and contains check."""
        poly = np.array([
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
        ])
        zone = RefinementZone(polygon=poly, max_area=5.0)
        assert zone.contains(5.0, 5.0)
        assert not zone.contains(15.0, 15.0)

    def test_refinement_zone_missing_both(self) -> None:
        """Neither center+radius nor polygon -> ValueError."""
        with pytest.raises(ValueError, match="Must specify"):
            RefinementZone(max_area=10.0)

    def test_refinement_zone_repr_circle(self) -> None:
        """Repr for circular zone."""
        zone = RefinementZone(center=(1.0, 2.0), radius=3.0)
        assert "center" in repr(zone)
        assert "radius" in repr(zone)

    def test_refinement_zone_repr_polygon(self) -> None:
        """Repr for polygon zone."""
        zone = RefinementZone(polygon=np.array([[0, 0], [1, 0], [1, 1]]))
        assert "polygon" in repr(zone)


class TestPointConstraint:
    """Test PointConstraint.as_array."""

    def test_point_constraint_as_array(self) -> None:
        """PointConstraint.as_array() returns numpy array."""
        pc = PointConstraint(x=3.0, y=7.0, marker=5)
        arr = pc.as_array()
        np.testing.assert_array_equal(arr, [3.0, 7.0])

    def test_point_constraint_repr(self) -> None:
        """PointConstraint repr."""
        pc = PointConstraint(x=1.0, y=2.0)
        assert "1.0" in repr(pc)
        assert "2.0" in repr(pc)
