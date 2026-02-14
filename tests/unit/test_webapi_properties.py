"""Unit tests for PropertyVisualizer (properties.py)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

pv = pytest.importorskip("pyvista", reason="PyVista not available")

from pyiwfm.visualization.webapi.properties import (
    PROPERTY_INFO,
    PropertyVisualizer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_mesh(n_cells: int = 100, has_layer: bool = True, has_head: bool = False):
    """Create a mock PyVista mesh."""
    mesh = MagicMock()
    mesh.n_cells = n_cells

    cell_data = {}
    if has_layer:
        # 2 layers: first 50 cells are layer 1, second 50 are layer 2
        cell_data["layer"] = np.array([1] * 50 + [2] * 50)
    if has_head:
        cell_data["head"] = np.random.rand(n_cells) * 100

    mesh.cell_data = cell_data
    mesh.point_data = {}
    return mesh


def _make_mock_aquifer_params(kh=True, kv=True, ss=True, sy=True):
    """Create a mock aquifer parameters object."""
    params = MagicMock()
    params.kh = np.random.rand(10, 2) if kh else None
    params.kv = np.random.rand(10, 2) if kv else None
    params.specific_storage = np.random.rand(10, 2) if ss else None
    params.specific_yield = np.random.rand(10, 2) if sy else None
    # Don't fall back to short names since we provide specific_ names
    params.ss = None
    params.sy = None
    return params


def _make_mock_stratigraphy(n_nodes: int = 10, n_layers: int = 2):
    """Create a mock stratigraphy."""
    strat = MagicMock()
    strat.n_layers = n_layers
    strat.top_elev = np.random.rand(n_nodes, n_layers) * 100
    strat.bottom_elev = np.random.rand(n_nodes, n_layers) * -50
    return strat


# ---------------------------------------------------------------------------
# PROPERTY_INFO
# ---------------------------------------------------------------------------


class TestPropertyInfo:
    """Tests for the PROPERTY_INFO dict."""

    def test_all_expected_keys(self) -> None:
        expected = {"layer", "kh", "kv", "ss", "sy", "head", "thickness", "top_elev", "bottom_elev"}
        assert expected == set(PROPERTY_INFO.keys())

    def test_all_have_required_fields(self) -> None:
        for name, info in PROPERTY_INFO.items():
            assert "name" in info, f"{name} missing 'name'"
            assert "units" in info, f"{name} missing 'units'"
            assert "description" in info, f"{name} missing 'description'"
            assert "cmap" in info, f"{name} missing 'cmap'"
            assert "log_scale" in info, f"{name} missing 'log_scale'"


# ---------------------------------------------------------------------------
# PropertyVisualizer.__init__
# ---------------------------------------------------------------------------


class TestPropertyVisualizerInit:
    """Tests for constructor and property detection."""

    def test_detects_layer(self) -> None:
        mesh = _make_mock_mesh()
        viz = PropertyVisualizer(mesh)
        assert "layer" in viz.available_properties

    def test_detects_kh_kv(self) -> None:
        mesh = _make_mock_mesh()
        params = _make_mock_aquifer_params()
        viz = PropertyVisualizer(mesh, aquifer_params=params)
        assert "kh" in viz.available_properties
        assert "kv" in viz.available_properties

    def test_detects_ss_sy(self) -> None:
        mesh = _make_mock_mesh()
        params = _make_mock_aquifer_params()
        viz = PropertyVisualizer(mesh, aquifer_params=params)
        assert "ss" in viz.available_properties
        assert "sy" in viz.available_properties

    def test_detects_head_from_cell_data(self) -> None:
        mesh = _make_mock_mesh(has_head=True)
        viz = PropertyVisualizer(mesh)
        assert "head" in viz.available_properties

    def test_no_aquifer_params(self) -> None:
        mesh = _make_mock_mesh()
        viz = PropertyVisualizer(mesh, aquifer_params=None)
        assert "kh" not in viz.available_properties
        assert "kv" not in viz.available_properties

    def test_ss_attribute_fallback(self) -> None:
        """When specific_storage attr is absent, fall back to 'ss' attribute."""
        mesh = _make_mock_mesh()
        params = MagicMock(spec=[])  # empty spec so hasattr only finds what we set
        params.kh = None
        params.kv = None
        # Don't set specific_storage at all — getattr fallback should find ss
        params.ss = np.array([0.001])  # short name fallback
        params.sy = None
        viz = PropertyVisualizer(mesh, aquifer_params=params)
        assert "ss" in viz.available_properties


# ---------------------------------------------------------------------------
# set_active_property
# ---------------------------------------------------------------------------


class TestSetActiveProperty:
    """Tests for set_active_property()."""

    def test_valid_property(self) -> None:
        mesh = _make_mock_mesh()
        viz = PropertyVisualizer(mesh)
        viz.set_active_property("layer")
        assert viz.active_property == "layer"

    def test_invalid_raises(self) -> None:
        mesh = _make_mock_mesh()
        viz = PropertyVisualizer(mesh)
        with pytest.raises(ValueError, match="not available"):
            viz.set_active_property("nonexistent")

    def test_sets_default_colormap(self) -> None:
        mesh = _make_mock_mesh(has_head=True)
        viz = PropertyVisualizer(mesh)
        viz.set_active_property("head")
        assert viz.colormap == "coolwarm"


# ---------------------------------------------------------------------------
# set_layer
# ---------------------------------------------------------------------------


class TestSetLayer:
    """Tests for set_layer()."""

    def test_valid_layer(self) -> None:
        mesh = _make_mock_mesh()
        strat = _make_mock_stratigraphy()
        viz = PropertyVisualizer(mesh, stratigraphy=strat)
        viz.set_layer(1)
        assert viz.active_layer == 1

    def test_all_layers(self) -> None:
        mesh = _make_mock_mesh()
        strat = _make_mock_stratigraphy()
        viz = PropertyVisualizer(mesh, stratigraphy=strat)
        viz.set_layer(0)
        assert viz.active_layer == 0

    def test_out_of_range_raises(self) -> None:
        mesh = _make_mock_mesh()
        strat = _make_mock_stratigraphy(n_layers=2)
        viz = PropertyVisualizer(mesh, stratigraphy=strat)
        with pytest.raises(ValueError):
            viz.set_layer(5)


# ---------------------------------------------------------------------------
# set_colormap / set_range / set_auto_range
# ---------------------------------------------------------------------------


class TestColorAndRange:
    """Tests for colormap and range settings."""

    def test_set_colormap(self) -> None:
        mesh = _make_mock_mesh()
        viz = PropertyVisualizer(mesh)
        viz.set_colormap("jet")
        assert viz.colormap == "jet"

    def test_set_range(self) -> None:
        mesh = _make_mock_mesh()
        viz = PropertyVisualizer(mesh)
        viz.set_range(0.0, 100.0)
        assert viz._auto_range is False
        assert viz._vmin == 0.0
        assert viz._vmax == 100.0

    def test_set_auto_range(self) -> None:
        mesh = _make_mock_mesh()
        viz = PropertyVisualizer(mesh)
        viz.set_range(0, 10)
        viz.set_auto_range(True)
        assert viz._auto_range is True


# ---------------------------------------------------------------------------
# get_property_array
# ---------------------------------------------------------------------------


class TestGetPropertyArray:
    """Tests for get_property_array()."""

    def test_layer_property(self) -> None:
        mesh = _make_mock_mesh()
        viz = PropertyVisualizer(mesh)
        arr = viz.get_property_array("layer")
        assert arr is not None
        assert len(arr) == 100

    def test_layer_with_filter(self) -> None:
        mesh = _make_mock_mesh()
        viz = PropertyVisualizer(mesh)
        arr = viz.get_property_array("layer", layer=1)
        assert arr is not None
        # Cells not in layer 1 should be NaN
        assert np.isnan(arr[50:]).all()

    def test_caching_works(self) -> None:
        mesh = _make_mock_mesh()
        viz = PropertyVisualizer(mesh)
        a1 = viz.get_property_array("layer")
        a2 = viz.get_property_array("layer")
        assert a1 is a2  # Same object from cache

    def test_layer_no_cell_data(self) -> None:
        mesh = _make_mock_mesh(has_layer=False)
        viz = PropertyVisualizer(mesh)
        arr = viz.get_property_array("layer")
        assert arr is not None
        assert np.all(arr == 1.0)


# ---------------------------------------------------------------------------
# _compute_thickness_array
# ---------------------------------------------------------------------------


class TestComputeThickness:
    """Tests for thickness computation."""

    def test_no_stratigraphy_returns_none(self) -> None:
        mesh = _make_mock_mesh()
        viz = PropertyVisualizer(mesh, stratigraphy=None)
        result = viz._compute_thickness_array()
        assert result is None

    def test_with_stratigraphy(self) -> None:
        mesh = _make_mock_mesh()
        strat = _make_mock_stratigraphy()
        viz = PropertyVisualizer(mesh, stratigraphy=strat)
        result = viz._compute_thickness_array()
        assert result is not None
        assert len(result) == 100

    def test_layer_filtering(self) -> None:
        mesh = _make_mock_mesh()
        strat = _make_mock_stratigraphy()
        viz = PropertyVisualizer(mesh, stratigraphy=strat)
        result = viz._compute_thickness_array(layer=1)
        assert result is not None
        # Layer 2 cells should be NaN
        assert np.isnan(result[50:]).all()


# ---------------------------------------------------------------------------
# _compute_elevation_array
# ---------------------------------------------------------------------------


class TestComputeElevation:
    """Tests for elevation computation."""

    def test_top_elevation(self) -> None:
        mesh = _make_mock_mesh()
        strat = _make_mock_stratigraphy()
        viz = PropertyVisualizer(mesh, stratigraphy=strat)
        result = viz._compute_elevation_array("top")
        assert result is not None
        assert len(result) == 100

    def test_bottom_elevation(self) -> None:
        mesh = _make_mock_mesh()
        strat = _make_mock_stratigraphy()
        viz = PropertyVisualizer(mesh, stratigraphy=strat)
        result = viz._compute_elevation_array("bottom")
        assert result is not None


# ---------------------------------------------------------------------------
# _compute_aquifer_param_array
# ---------------------------------------------------------------------------


class TestComputeAquiferParam:
    """Tests for aquifer parameter array computation."""

    def test_2d_param_data(self) -> None:
        mesh = _make_mock_mesh()
        params = _make_mock_aquifer_params()
        viz = PropertyVisualizer(mesh, aquifer_params=params)
        arr = viz._compute_aquifer_param_array("kh")
        assert arr is not None
        assert len(arr) == 100

    def test_1d_param_data(self) -> None:
        mesh = _make_mock_mesh()
        params = MagicMock()
        params.kh = np.array([1.0, 2.0, 3.0])  # 1D
        params.kv = None
        params.specific_storage = None
        params.ss = None
        params.specific_yield = None
        params.sy = None
        viz = PropertyVisualizer(mesh, aquifer_params=params)
        arr = viz._compute_aquifer_param_array("kh")
        assert arr is not None
        # All cells should have mean of param_data
        assert np.all(arr[:50] == pytest.approx(2.0))

    def test_no_params_returns_none(self) -> None:
        mesh = _make_mock_mesh()
        viz = PropertyVisualizer(mesh, aquifer_params=None)
        result = viz._compute_aquifer_param_array("kh")
        assert result is None

    def test_ss_attribute_name_mapping(self) -> None:
        """Test that 'ss' maps to specific_storage attribute."""
        mesh = _make_mock_mesh()
        params = _make_mock_aquifer_params()
        viz = PropertyVisualizer(mesh, aquifer_params=params)
        arr = viz._compute_aquifer_param_array("ss")
        assert arr is not None


# ---------------------------------------------------------------------------
# add_head_data
# ---------------------------------------------------------------------------


class TestAddHeadData:
    """Tests for add_head_data()."""

    def test_1d_matches_n_cells(self) -> None:
        mesh = _make_mock_mesh()
        viz = PropertyVisualizer(mesh)
        head = np.random.rand(100)
        viz.add_head_data(head)
        assert "head" in viz.available_properties
        assert np.array_equal(mesh.cell_data["head"], head)

    def test_1d_doesnt_match(self) -> None:
        mesh = _make_mock_mesh()
        viz = PropertyVisualizer(mesh)
        head = np.random.rand(10)  # doesn't match n_cells=100
        viz.add_head_data(head)
        assert mesh.cell_data["head"].shape == (100,)

    def test_2d_multi_layer(self) -> None:
        mesh = _make_mock_mesh()
        viz = PropertyVisualizer(mesh)
        head = np.random.rand(10, 2)
        viz.add_head_data(head)
        assert "head" in viz.available_properties

    def test_clears_cache(self) -> None:
        mesh = _make_mock_mesh(has_head=True)
        viz = PropertyVisualizer(mesh)
        viz.get_property_array("head")
        assert len(viz._property_cache) > 0
        viz.add_head_data(np.random.rand(100))
        assert not any(k.startswith("head") for k in viz._property_cache)


# ---------------------------------------------------------------------------
# get_property_info
# ---------------------------------------------------------------------------


class TestGetPropertyInfo:
    """Tests for get_property_info()."""

    def test_known_property(self) -> None:
        mesh = _make_mock_mesh()
        viz = PropertyVisualizer(mesh)
        info = viz.get_property_info("layer")
        assert info["name"] == "Layer"
        assert "min" in info
        assert "max" in info
        assert "mean" in info

    def test_unknown_property(self) -> None:
        mesh = _make_mock_mesh()
        viz = PropertyVisualizer(mesh)
        info = viz.get_property_info("custom_field")
        assert "Custom property" in info["description"]

    def test_default_active_property(self) -> None:
        mesh = _make_mock_mesh()
        viz = PropertyVisualizer(mesh)
        info = viz.get_property_info()
        assert info["name"] == "Layer"


# ---------------------------------------------------------------------------
# get_colorbar_settings
# ---------------------------------------------------------------------------


class TestGetColorbarSettings:
    """Tests for get_colorbar_settings()."""

    def test_returns_complete_dict(self) -> None:
        mesh = _make_mock_mesh()
        viz = PropertyVisualizer(mesh)
        settings = viz.get_colorbar_settings()
        assert "title" in settings
        assert "units" in settings
        assert "cmap" in settings
        assert "vmin" in settings
        assert "vmax" in settings
        assert "log_scale" in settings


# ---------------------------------------------------------------------------
# value_range
# ---------------------------------------------------------------------------


class TestValueRange:
    """Tests for value_range property."""

    def test_auto_mode(self) -> None:
        mesh = _make_mock_mesh()
        viz = PropertyVisualizer(mesh)
        vmin, vmax = viz.value_range
        assert vmin <= vmax

    def test_manual_mode(self) -> None:
        mesh = _make_mock_mesh()
        viz = PropertyVisualizer(mesh)
        viz.set_range(10.0, 90.0)
        vmin, vmax = viz.value_range
        assert vmin == 10.0
        assert vmax == 90.0

    def test_empty_scalars(self) -> None:
        mesh = _make_mock_mesh(has_layer=False, n_cells=0)
        mesh.n_cells = 0
        viz = PropertyVisualizer(mesh)
        # Force empty scalars
        viz._property_cache["layer_0"] = np.array([])
        vmin, vmax = viz.value_range
        assert vmin == 0.0
        assert vmax == 1.0


# ---------------------------------------------------------------------------
# clear_cache
# ---------------------------------------------------------------------------


class TestClearCache:
    """Tests for cache clearing."""

    def test_clear_cache(self) -> None:
        mesh = _make_mock_mesh()
        viz = PropertyVisualizer(mesh)
        viz.get_property_array("layer")
        assert len(viz._property_cache) > 0
        viz.clear_cache()
        assert len(viz._property_cache) == 0
