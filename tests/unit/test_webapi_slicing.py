"""Unit tests for SlicingController (slicing.py)."""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

pv = pytest.importorskip("pyvista", reason="PyVista not available")

from pyiwfm.visualization.webapi.slicing import SlicingController


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_mesh(
    bounds: tuple = (0.0, 1000.0, 0.0, 2000.0, -100.0, 50.0),
    center: tuple = (500.0, 1000.0, -25.0),
):
    """Create a mock PyVista UnstructuredGrid."""
    mesh = MagicMock()
    mesh.bounds = bounds
    mesh.center = center

    slice_result = MagicMock()
    slice_result.n_cells = 10
    slice_result.n_points = 20
    slice_result.area = 5000.0
    slice_result.bounds = bounds
    slice_result.cell_data = MagicMock()
    slice_result.cell_data.keys.return_value = ["layer"]
    slice_result.point_data = MagicMock()
    slice_result.point_data.keys.return_value = ["head"]
    slice_result.merge.return_value = slice_result

    mesh.slice.return_value = slice_result

    clip_result = MagicMock()
    mesh.clip_box.return_value = clip_result

    return mesh


# ---------------------------------------------------------------------------
# Constructor and properties
# ---------------------------------------------------------------------------


class TestSlicingControllerInit:
    """Tests for constructor and basic properties."""

    def test_constructor(self) -> None:
        mesh = _make_mock_mesh()
        slicer = SlicingController(mesh)
        assert slicer.mesh is mesh
        assert slicer._cache == {}

    def test_bounds(self) -> None:
        mesh = _make_mock_mesh(bounds=(10.0, 20.0, 30.0, 40.0, -5.0, 5.0))
        slicer = SlicingController(mesh)
        assert slicer.bounds == (10.0, 20.0, 30.0, 40.0, -5.0, 5.0)

    def test_x_range(self) -> None:
        mesh = _make_mock_mesh(bounds=(10.0, 20.0, 30.0, 40.0, -5.0, 5.0))
        slicer = SlicingController(mesh)
        assert slicer.x_range == (10.0, 20.0)

    def test_y_range(self) -> None:
        mesh = _make_mock_mesh(bounds=(10.0, 20.0, 30.0, 40.0, -5.0, 5.0))
        slicer = SlicingController(mesh)
        assert slicer.y_range == (30.0, 40.0)

    def test_z_range(self) -> None:
        mesh = _make_mock_mesh(bounds=(10.0, 20.0, 30.0, 40.0, -5.0, 5.0))
        slicer = SlicingController(mesh)
        assert slicer.z_range == (-5.0, 5.0)

    def test_center(self) -> None:
        mesh = _make_mock_mesh(center=(15.0, 35.0, 0.0))
        slicer = SlicingController(mesh)
        assert slicer.center == (15.0, 35.0, 0.0)


# ---------------------------------------------------------------------------
# Axis-aligned slices
# ---------------------------------------------------------------------------


class TestAxisSlices:
    """Tests for slice_x, slice_y, slice_z."""

    def test_slice_x(self) -> None:
        mesh = _make_mock_mesh()
        slicer = SlicingController(mesh)
        result = slicer.slice_x(500.0)
        mesh.slice.assert_called_once()
        call_kwargs = mesh.slice.call_args[1]
        assert call_kwargs["normal"] == (1, 0, 0)

    def test_slice_y(self) -> None:
        mesh = _make_mock_mesh()
        slicer = SlicingController(mesh)
        result = slicer.slice_y(1000.0)
        mesh.slice.assert_called_once()
        call_kwargs = mesh.slice.call_args[1]
        assert call_kwargs["normal"] == (0, 1, 0)

    def test_slice_z(self) -> None:
        mesh = _make_mock_mesh()
        slicer = SlicingController(mesh)
        result = slicer.slice_z(-50.0)
        mesh.slice.assert_called_once()
        call_kwargs = mesh.slice.call_args[1]
        assert call_kwargs["normal"] == (0, 0, 1)

    def test_slice_x_clamps_to_bounds(self) -> None:
        mesh = _make_mock_mesh(bounds=(0.0, 1000.0, 0.0, 2000.0, -100.0, 50.0))
        slicer = SlicingController(mesh)
        slicer.slice_x(9999.0)  # above max
        call_kwargs = mesh.slice.call_args[1]
        origin = call_kwargs["origin"]
        assert origin[0] == 1000.0  # clamped

    def test_slice_z_clamps_below(self) -> None:
        mesh = _make_mock_mesh(bounds=(0.0, 1000.0, 0.0, 2000.0, -100.0, 50.0))
        slicer = SlicingController(mesh)
        slicer.slice_z(-9999.0)  # below min
        call_kwargs = mesh.slice.call_args[1]
        origin = call_kwargs["origin"]
        assert origin[2] == -100.0  # clamped

    def test_slice_x_caches(self) -> None:
        mesh = _make_mock_mesh()
        slicer = SlicingController(mesh)
        r1 = slicer.slice_x(500.0)
        r2 = slicer.slice_x(500.0)
        assert r1 is r2
        assert mesh.slice.call_count == 1  # only called once

    def test_different_positions_different_cache(self) -> None:
        mesh = _make_mock_mesh()
        slicer = SlicingController(mesh)
        slicer.slice_x(100.0)
        slicer.slice_x(200.0)
        assert mesh.slice.call_count == 2


# ---------------------------------------------------------------------------
# Arbitrary slice
# ---------------------------------------------------------------------------


class TestSliceArbitrary:
    """Tests for slice_arbitrary()."""

    def test_normalizes_normal_vector(self) -> None:
        mesh = _make_mock_mesh()
        slicer = SlicingController(mesh)
        slicer.slice_arbitrary(normal=(2, 0, 0))
        call_kwargs = mesh.slice.call_args[1]
        normal = call_kwargs["normal"]
        assert abs(normal[0] - 1.0) < 1e-6
        assert abs(normal[1]) < 1e-6
        assert abs(normal[2]) < 1e-6

    def test_default_origin_is_center(self) -> None:
        mesh = _make_mock_mesh(center=(500.0, 1000.0, -25.0))
        slicer = SlicingController(mesh)
        slicer.slice_arbitrary(normal=(1, 0, 0))
        call_kwargs = mesh.slice.call_args[1]
        assert call_kwargs["origin"] == (500.0, 1000.0, -25.0)

    def test_custom_origin(self) -> None:
        mesh = _make_mock_mesh()
        slicer = SlicingController(mesh)
        slicer.slice_arbitrary(normal=(0, 0, 1), origin=(100.0, 200.0, 0.0))
        call_kwargs = mesh.slice.call_args[1]
        assert call_kwargs["origin"] == (100.0, 200.0, 0.0)


# ---------------------------------------------------------------------------
# Cross-section
# ---------------------------------------------------------------------------


class TestCreateCrossSection:
    """Tests for create_cross_section()."""

    def test_computes_perpendicular_normal(self) -> None:
        mesh = _make_mock_mesh()
        slicer = SlicingController(mesh)
        slicer.create_cross_section(start=(0, 0), end=(1000, 0))
        call_kwargs = mesh.slice.call_args[1]
        normal = call_kwargs["normal"]
        # For a line along X, the perpendicular in XY is (0, 1, 0)
        assert abs(normal[0]) < 1e-6
        assert abs(abs(normal[1]) - 1.0) < 1e-6
        assert abs(normal[2]) < 1e-6

    def test_start_equals_end_raises(self) -> None:
        mesh = _make_mock_mesh()
        slicer = SlicingController(mesh)
        with pytest.raises(ValueError, match="different"):
            slicer.create_cross_section(start=(100, 200), end=(100, 200))


# ---------------------------------------------------------------------------
# Polyline slice
# ---------------------------------------------------------------------------


class TestSliceAlongPolyline:
    """Tests for slice_along_polyline()."""

    def test_too_few_points_raises(self) -> None:
        mesh = _make_mock_mesh()
        slicer = SlicingController(mesh)
        with pytest.raises(ValueError, match="at least 2"):
            slicer.slice_along_polyline([(0, 0)])

    def test_combines_segments(self) -> None:
        mesh = _make_mock_mesh()
        slicer = SlicingController(mesh)
        result = slicer.slice_along_polyline([(0, 0), (500, 500), (1000, 0)])
        assert result is not None

    def test_empty_slices_return_empty(self) -> None:
        mesh = _make_mock_mesh()
        empty_slice = MagicMock()
        empty_slice.n_cells = 0
        mesh.slice.return_value = empty_slice

        slicer = SlicingController(mesh)
        result = slicer.slice_along_polyline([(0, 0), (100, 100)])
        assert result.n_cells == 0  # pv.PolyData()


# ---------------------------------------------------------------------------
# Box slice
# ---------------------------------------------------------------------------


class TestSliceBox:
    """Tests for slice_box()."""

    def test_default_center_third(self) -> None:
        mesh = _make_mock_mesh(bounds=(0.0, 300.0, 0.0, 300.0, -30.0, 30.0))
        slicer = SlicingController(mesh)
        slicer.slice_box()
        mesh.clip_box.assert_called_once()
        call_kwargs = mesh.clip_box.call_args[1]
        bounds = call_kwargs["bounds"]
        assert bounds[0] == pytest.approx(100.0)
        assert bounds[1] == pytest.approx(200.0)

    def test_custom_bounds(self) -> None:
        mesh = _make_mock_mesh()
        slicer = SlicingController(mesh)
        custom = (10.0, 20.0, 30.0, 40.0, -5.0, 5.0)
        slicer.slice_box(bounds=custom)
        call_kwargs = mesh.clip_box.call_args[1]
        assert call_kwargs["bounds"] == custom

    def test_invert_flag(self) -> None:
        mesh = _make_mock_mesh()
        slicer = SlicingController(mesh)
        slicer.slice_box(invert=True)
        call_kwargs = mesh.clip_box.call_args[1]
        assert call_kwargs["invert"] is True


# ---------------------------------------------------------------------------
# Multiple Z slices
# ---------------------------------------------------------------------------


class TestSliceMultipleZ:
    """Tests for slice_multiple_z()."""

    def test_custom_positions(self) -> None:
        mesh = _make_mock_mesh()
        slicer = SlicingController(mesh)
        result = slicer.slice_multiple_z(positions=[-50.0, 0.0, 25.0])
        assert len(result) == 3

    def test_auto_generated_positions(self) -> None:
        mesh = _make_mock_mesh(bounds=(0, 1000, 0, 2000, -100, 50))
        slicer = SlicingController(mesh)
        result = slicer.slice_multiple_z(n_slices=3)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# get_slice_properties
# ---------------------------------------------------------------------------


class TestGetSliceProperties:
    """Tests for get_slice_properties()."""

    def test_returns_correct_structure(self) -> None:
        mesh = _make_mock_mesh()
        slicer = SlicingController(mesh)
        slice_mesh = mesh.slice.return_value
        props = slicer.get_slice_properties(slice_mesh)
        assert "n_cells" in props
        assert "n_points" in props
        assert "area" in props
        assert "bounds" in props
        assert "cell_arrays" in props
        assert "point_arrays" in props

    def test_empty_slice(self) -> None:
        mesh = _make_mock_mesh()
        slicer = SlicingController(mesh)
        empty_slice = MagicMock()
        empty_slice.n_cells = 0
        empty_slice.n_points = 0
        empty_slice.cell_data.keys.return_value = []
        empty_slice.point_data.keys.return_value = []
        props = slicer.get_slice_properties(empty_slice)
        assert props["area"] == 0.0
        assert props["bounds"] is None


# ---------------------------------------------------------------------------
# Normalized position conversions
# ---------------------------------------------------------------------------


class TestNormalizedConversions:
    """Tests for position_to_normalized and normalized_to_position."""

    def test_position_to_normalized_x(self) -> None:
        mesh = _make_mock_mesh(bounds=(0.0, 1000.0, 0.0, 2000.0, -100.0, 50.0))
        slicer = SlicingController(mesh)
        assert slicer.position_to_normalized("x", 500.0) == pytest.approx(0.5)
        assert slicer.position_to_normalized("x", 0.0) == pytest.approx(0.0)
        assert slicer.position_to_normalized("x", 1000.0) == pytest.approx(1.0)

    def test_position_to_normalized_y(self) -> None:
        mesh = _make_mock_mesh(bounds=(0.0, 1000.0, 0.0, 2000.0, -100.0, 50.0))
        slicer = SlicingController(mesh)
        assert slicer.position_to_normalized("y", 1000.0) == pytest.approx(0.5)

    def test_position_to_normalized_z(self) -> None:
        mesh = _make_mock_mesh(bounds=(0.0, 1000.0, 0.0, 2000.0, -100.0, 50.0))
        slicer = SlicingController(mesh)
        assert slicer.position_to_normalized("z", -25.0) == pytest.approx(0.5)

    def test_unknown_axis_raises(self) -> None:
        mesh = _make_mock_mesh()
        slicer = SlicingController(mesh)
        with pytest.raises(ValueError, match="Unknown axis"):
            slicer.position_to_normalized("w", 0.0)

    def test_normalized_to_position_roundtrip(self) -> None:
        mesh = _make_mock_mesh(bounds=(0.0, 1000.0, 0.0, 2000.0, -100.0, 50.0))
        slicer = SlicingController(mesh)
        for axis in ("x", "y", "z"):
            pos = 0.3
            norm = slicer.position_to_normalized(axis, slicer.normalized_to_position(axis, pos))
            assert norm == pytest.approx(pos)

    def test_degenerate_range_returns_half(self) -> None:
        mesh = _make_mock_mesh(bounds=(5.0, 5.0, 0.0, 10.0, 0.0, 10.0))
        slicer = SlicingController(mesh)
        assert slicer.position_to_normalized("x", 5.0) == 0.5

    def test_normalized_to_position_unknown_axis(self) -> None:
        mesh = _make_mock_mesh()
        slicer = SlicingController(mesh)
        with pytest.raises(ValueError, match="Unknown axis"):
            slicer.normalized_to_position("w", 0.5)


# ---------------------------------------------------------------------------
# normalized_to_position_along
# ---------------------------------------------------------------------------


class TestNormalizedToPositionAlong:
    """Tests for normalized_to_position_along()."""

    def test_zero_norm_returns_center(self) -> None:
        mesh = _make_mock_mesh(
            bounds=(0, 100, 0, 200, -10, 10),
            center=(50.0, 100.0, 0.0),
        )
        slicer = SlicingController(mesh)
        result = slicer.normalized_to_position_along((0, 0, 0), 0.5)
        assert result == pytest.approx((50.0, 100.0, 0.0))

    def test_axis_aligned_normal(self) -> None:
        mesh = _make_mock_mesh(
            bounds=(0, 100, 0, 200, -10, 10),
            center=(50.0, 100.0, 0.0),
        )
        slicer = SlicingController(mesh)
        # Along X: normalized 0.0 = xmin, 1.0 = xmax
        result_min = slicer.normalized_to_position_along((1, 0, 0), 0.0)
        result_max = slicer.normalized_to_position_along((1, 0, 0), 1.0)
        assert result_min[0] == pytest.approx(0.0)
        assert result_max[0] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class TestCache:
    """Tests for cache management."""

    def test_cache_eviction_at_max_size(self) -> None:
        mesh = _make_mock_mesh()
        slicer = SlicingController(mesh)
        slicer._max_cache_size = 3

        slicer._add_to_cache("a", MagicMock())
        slicer._add_to_cache("b", MagicMock())
        slicer._add_to_cache("c", MagicMock())
        # At max, adding another should evict "a"
        slicer._add_to_cache("d", MagicMock())
        assert "a" not in slicer._cache
        assert "d" in slicer._cache

    def test_clear_cache(self) -> None:
        mesh = _make_mock_mesh()
        slicer = SlicingController(mesh)
        slicer._add_to_cache("test", MagicMock())
        assert len(slicer._cache) == 1
        slicer.clear_cache()
        assert len(slicer._cache) == 0
