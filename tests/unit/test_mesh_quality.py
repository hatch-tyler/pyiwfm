"""Tests for the mesh quality metrics module."""

from __future__ import annotations

import math

import pytest


class TestElementQuality:
    """Tests for element quality computation."""

    def test_square_element(self):
        """A perfect square should have aspect_ratio=1 and low skewness."""
        from pyiwfm.core.mesh_quality import compute_element_quality

        # Unit square
        vertices = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        q = compute_element_quality(vertices, element_id=1)

        assert q.element_id == 1
        assert q.n_vertices == 4
        assert q.area == pytest.approx(1.0, abs=1e-10)
        assert q.aspect_ratio == pytest.approx(1.0, abs=0.01)
        assert q.min_angle == pytest.approx(90.0, abs=0.1)
        assert q.max_angle == pytest.approx(90.0, abs=0.1)
        assert q.skewness < 0.1

    def test_equilateral_triangle(self):
        """An equilateral triangle should have low skewness."""
        from pyiwfm.core.mesh_quality import compute_element_quality

        h = math.sqrt(3) / 2
        vertices = [(0.0, 0.0), (1.0, 0.0), (0.5, h)]
        q = compute_element_quality(vertices, element_id=2)

        assert q.n_vertices == 3
        assert q.area == pytest.approx(h / 2, abs=1e-10)
        assert q.aspect_ratio == pytest.approx(1.0, abs=0.01)
        assert q.min_angle == pytest.approx(60.0, abs=0.1)
        assert q.max_angle == pytest.approx(60.0, abs=0.1)
        assert q.skewness < 0.1

    def test_elongated_rectangle(self):
        """An elongated rectangle should have high aspect ratio."""
        from pyiwfm.core.mesh_quality import compute_element_quality

        vertices = [(0.0, 0.0), (10.0, 0.0), (10.0, 1.0), (0.0, 1.0)]
        q = compute_element_quality(vertices)

        assert q.area == pytest.approx(10.0, abs=1e-10)
        assert q.aspect_ratio == pytest.approx(10.0, abs=0.1)

    def test_degenerate_triangle(self):
        """A very flat triangle should have high skewness."""
        from pyiwfm.core.mesh_quality import compute_element_quality

        vertices = [(0.0, 0.0), (10.0, 0.0), (5.0, 0.01)]
        q = compute_element_quality(vertices)

        assert q.skewness > 0.5
        assert q.min_angle < 5.0  # Very small angle


class TestMeshQuality:
    """Tests for full mesh quality computation."""

    def test_simple_grid(self):
        """Test mesh quality on the 2x2 test grid."""
        from pyiwfm.core.mesh_quality import compute_mesh_quality
        from tests.conftest import make_simple_grid

        grid = make_simple_grid()
        report = compute_mesh_quality(grid)

        assert report.n_elements == 4
        assert report.area_min > 0
        assert report.area_max > 0
        assert report.aspect_ratio_min >= 1.0
        assert report.min_angle_global > 0
        assert report.max_angle_global <= 180

    def test_report_to_dict(self):
        """to_dict should serialize without per-element details."""
        from pyiwfm.core.mesh_quality import compute_mesh_quality
        from tests.conftest import make_simple_grid

        grid = make_simple_grid()
        report = compute_mesh_quality(grid)
        d = report.to_dict()

        assert "n_elements" in d
        assert "area_min" in d
        assert "element_qualities" not in d
