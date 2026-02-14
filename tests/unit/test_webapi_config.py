"""Unit tests for the FastAPI web viewer ModelState configuration.

Tests set_model, budget discovery, coordinate reprojection, and caching.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("pydantic", reason="Pydantic not available")

from pyiwfm.core.mesh import AppGrid, Node, Element
from pyiwfm.visualization.webapi.config import ModelState, ViewerSettings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_grid():
    nodes = {
        1: Node(id=1, x=0.0, y=0.0),
        2: Node(id=2, x=100.0, y=0.0),
        3: Node(id=3, x=100.0, y=100.0),
        4: Node(id=4, x=0.0, y=100.0),
    }
    elements = {
        1: Element(id=1, vertices=(1, 2, 3, 4), subregion=1),
    }
    grid = AppGrid(nodes=nodes, elements=elements)
    grid.compute_connectivity()
    grid.compute_areas()
    return grid


def _make_mock_model():
    grid = _make_grid()
    model = MagicMock()
    model.grid = grid
    model.n_nodes = 4
    model.n_elements = 1
    model.has_streams = False
    model.has_lakes = False
    model.streams = None
    model.lakes = None
    model.groundwater = None
    model.stratigraphy = None
    model.metadata = {}
    return model


# ---------------------------------------------------------------------------
# ViewerSettings
# ---------------------------------------------------------------------------


class TestViewerSettings:
    """Tests for the ViewerSettings Pydantic model."""

    def test_defaults(self):
        s = ViewerSettings()
        assert s.host == "127.0.0.1"
        assert s.port == 8080
        assert s.debug is False
        assert s.open_browser is True

    def test_custom_values(self):
        s = ViewerSettings(host="0.0.0.0", port=9000, debug=True)
        assert s.host == "0.0.0.0"
        assert s.port == 9000
        assert s.debug is True

    def test_invalid_port(self):
        with pytest.raises(Exception):
            ViewerSettings(port=99999)

    def test_extra_fields_forbidden(self):
        with pytest.raises(Exception):
            ViewerSettings(unknown_field="test")


# ---------------------------------------------------------------------------
# ModelState
# ---------------------------------------------------------------------------


class TestModelState:
    """Tests for the ModelState singleton."""

    def test_initial_state(self):
        state = ModelState()
        assert state.is_loaded is False
        assert state.model is None

    def test_set_model(self):
        state = ModelState()
        model = _make_mock_model()
        state.set_model(model)

        assert state.is_loaded is True
        assert state.model is model

    def test_set_model_resets_caches(self):
        state = ModelState()
        model = _make_mock_model()
        state.set_model(model)

        # Add some cache entries
        state._layer_surface_cache[1] = {"test": True}
        state._geojson_cache[1] = {"test": True}
        state._budget_readers["gw"] = MagicMock()

        # Re-set model should clear caches
        state.set_model(model)
        assert state._layer_surface_cache == {}
        assert state._geojson_cache == {}
        assert state._budget_readers == {}

    def test_set_model_with_crs(self):
        state = ModelState()
        model = _make_mock_model()
        state.set_model(model, crs="EPSG:4326")
        assert state._crs == "EPSG:4326"

    def test_default_crs(self):
        state = ModelState()
        model = _make_mock_model()
        state.set_model(model)
        assert state._crs == "+proj=utm +zone=10 +datum=NAD83 +units=us-ft +no_defs"

    def test_results_dir_from_metadata(self):
        state = ModelState()
        model = _make_mock_model()
        model.metadata = {"simulation_file": "/some/path/Simulation.in"}
        state.set_model(model)
        assert state._results_dir == Path("/some/path")

    def test_results_dir_none_without_metadata(self):
        state = ModelState()
        model = _make_mock_model()
        state.set_model(model)
        assert state._results_dir is None

    def test_get_available_budgets_no_model(self):
        state = ModelState()
        assert state.get_available_budgets() == []

    def test_get_available_budgets_no_files(self):
        state = ModelState()
        model = _make_mock_model()
        state.set_model(model)
        assert state.get_available_budgets() == []

    def test_get_budget_reader_unknown_type(self):
        state = ModelState()
        model = _make_mock_model()
        state.set_model(model)
        assert state.get_budget_reader("unknown") is None

    def test_compute_bounds(self):
        state = ModelState()
        model = _make_mock_model()
        state.set_model(model)
        bounds = state.get_bounds()
        assert len(bounds) == 6
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        assert xmin == 0.0
        assert xmax == 100.0
        assert ymin == 0.0
        assert ymax == 100.0

    def test_observations_lifecycle(self):
        state = ModelState()
        assert state.list_observations() == []

        state.add_observation("obs1", {
            "filename": "test.csv",
            "location_id": 1,
            "type": "gw",
            "n_records": 10,
            "times": [],
            "values": [],
        })

        obs_list = state.list_observations()
        assert len(obs_list) == 1
        assert obs_list[0]["id"] == "obs1"
        assert obs_list[0]["filename"] == "test.csv"

        obs = state.get_observation("obs1")
        assert obs is not None
        assert obs["n_records"] == 10

        assert state.get_observation("nonexistent") is None

        assert state.delete_observation("obs1") is True
        assert state.delete_observation("obs1") is False
        assert state.list_observations() == []

    def test_get_results_info_no_model(self):
        state = ModelState()
        info = state.get_results_info()
        assert info["has_results"] is False

    def test_get_results_info_empty_model(self):
        state = ModelState()
        model = _make_mock_model()
        state.set_model(model)
        info = state.get_results_info()
        assert info["has_results"] is False
        assert info["available_budgets"] == []
        assert info["n_head_timesteps"] == 0


class TestReprojectCoords:
    """Tests for coordinate reprojection in ModelState."""

    def test_reproject_without_pyproj(self):
        """When pyproj is not available, coords pass through unchanged."""
        state = ModelState()
        state._crs = "EPSG:4326"

        # Mock pyproj not being available
        with patch.object(state, "_get_transformer", return_value=None):
            lng, lat = state.reproject_coords(100.0, 200.0)
            assert lng == 100.0
            assert lat == 200.0

    def test_reproject_with_transformer(self):
        """When a transformer is available, coords are transformed."""
        state = ModelState()

        mock_transformer = MagicMock()
        mock_transformer.transform.return_value = (-121.5, 38.5)
        state._transformer = mock_transformer

        lng, lat = state.reproject_coords(500000.0, 4200000.0)
        assert lng == -121.5
        assert lat == 38.5
        mock_transformer.transform.assert_called_once_with(500000.0, 4200000.0)
