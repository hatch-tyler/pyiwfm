"""Tests for the FastAPI zbudget routes.

Covers endpoints: types, elements, presets, zones CRUD,
data, columns, summary, glossary.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

fastapi = pytest.importorskip("fastapi", reason="FastAPI not available")
pydantic = pytest.importorskip("pydantic", reason="Pydantic not available")

from fastapi.testclient import TestClient  # noqa: E402

from pyiwfm.visualization.webapi.config import model_state  # noqa: E402
from pyiwfm.visualization.webapi.routes.zbudgets import (  # noqa: E402
    ZBUDGET_GLOSSARY,
    _safe_float,
    _sanitize_values,
)
from pyiwfm.visualization.webapi.server import create_app  # noqa: E402


def _reset_model_state():
    """Reset the global model_state to a clean state."""
    model_state._model = None
    model_state._mesh_3d = None
    model_state._mesh_surface = None
    model_state._surface_json_data = None
    model_state._bounds = None
    model_state._pv_mesh_3d = None
    model_state._layer_surface_cache = {}
    model_state._crs = "+proj=utm +zone=10 +datum=NAD83 +units=us-ft +no_defs"
    model_state._transformer = None
    model_state._geojson_cache = {}
    model_state._head_loader = None
    model_state._gw_hydrograph_reader = None
    model_state._stream_hydrograph_reader = None
    model_state._budget_readers = {}
    model_state._zbudget_readers = {}
    model_state._active_zone_def = None
    model_state._observations = {}
    model_state._results_dir = None
    model_state._area_manager = None
    # Restore any monkey-patched methods
    for attr in (
        "get_zbudget_reader",
        "get_available_zbudgets",
        "get_budget_reader",
        "get_available_budgets",
        "get_head_loader",
        "get_gw_hydrograph_reader",
        "get_stream_hydrograph_reader",
        "get_subsidence_reader",
    ):
        if attr in model_state.__dict__:
            del model_state.__dict__[attr]


def _make_mock_model():
    """Create a minimal mock IWFMModel."""
    mock = MagicMock()
    mock.metadata = {"length_unit": "FT"}

    # Grid with 4 elements, 2 subregions
    mock.grid.elements = {}
    mock.grid.subregions = {}
    mock.grid.nodes = {}

    for i in range(1, 5):
        elem = MagicMock()
        elem.vertices = (i,)
        elem.subregion = 1 if i <= 2 else 2
        elem.area = 100.0
        mock.grid.elements[i] = elem

    for i in range(1, 5):
        node = MagicMock()
        node.x = float(i * 1000)
        node.y = float(i * 1000)
        mock.grid.nodes[i] = node

    sr1 = MagicMock()
    sr1.name = "North"
    sr2 = MagicMock()
    sr2.name = "South"
    mock.grid.subregions = {1: sr1, 2: sr2}

    return mock


def _make_mock_zbudget_reader(zones=None, data_names=None, n_timesteps=12):
    """Create a mock ZBudgetReader."""
    mock = MagicMock()
    mock.zones = zones or ["Zone 1", "Zone 2"]
    mock.n_zones = len(mock.zones)
    mock.data_names = data_names or ["Deep Percolation", "Pumping", "Storage"]
    mock.n_layers = 1
    mock.n_timesteps = n_timesteps
    mock.descriptor = "GROUNDWATER ZONE BUDGET"
    mock.header.time_unit = "1MON"

    # get_dataframe returns a DataFrame
    dates = pd.date_range("2000-01-01", periods=n_timesteps, freq="MS")
    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_timesteps, len(mock.data_names)))
    df = pd.DataFrame(data, index=dates, columns=mock.data_names)
    mock.get_dataframe.return_value = df

    return mock


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset model_state before each test."""
    _reset_model_state()
    yield
    _reset_model_state()


@pytest.fixture()
def client():
    """Create a test client with no model loaded."""
    app = create_app()
    return TestClient(app)


@pytest.fixture()
def client_with_model():
    """Create a test client with a mock model loaded."""
    mock_model = _make_mock_model()
    model_state._model = mock_model
    model_state._results_dir = Path("/fake/results")
    app = create_app()
    return TestClient(app)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestSafeFloat:
    def test_normal_float(self):
        assert _safe_float(3.14) == 3.14

    def test_nan_returns_none(self):
        assert _safe_float(float("nan")) is None

    def test_inf_returns_none(self):
        assert _safe_float(float("inf")) is None

    def test_neg_inf_returns_none(self):
        assert _safe_float(float("-inf")) is None

    def test_zero(self):
        assert _safe_float(0.0) == 0.0


class TestSanitizeValues:
    def test_normal_list(self):
        assert _sanitize_values([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]

    def test_with_nan(self):
        result = _sanitize_values([1.0, float("nan"), 3.0])
        assert result == [1.0, None, 3.0]

    def test_with_inf(self):
        result = _sanitize_values([1.0, float("inf"), float("-inf")])
        assert result == [1.0, None, None]

    def test_empty(self):
        assert _sanitize_values([]) == []


# ---------------------------------------------------------------------------
# Endpoint tests: no model loaded
# ---------------------------------------------------------------------------


class TestNoModel:
    def test_types_returns_404(self, client):
        r = client.get("/api/zbudgets/types")
        assert r.status_code == 404

    def test_elements_returns_404(self, client):
        r = client.get("/api/zbudgets/elements")
        assert r.status_code == 404

    def test_presets_returns_404(self, client):
        r = client.get("/api/zbudgets/presets")
        assert r.status_code == 404

    def test_post_zones_returns_404(self, client):
        r = client.post("/api/zbudgets/zones", json={"zones": []})
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Endpoint tests: model loaded
# ---------------------------------------------------------------------------


class TestWithModel:
    def test_types_returns_list(self, client_with_model):
        # No HDF files in /fake/results, so empty list
        r = client_with_model.get("/api/zbudgets/types")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_elements_returns_list(self, client_with_model):
        r = client_with_model.get("/api/zbudgets/elements")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert len(data) == 4  # 4 elements
        for elem in data:
            assert "id" in elem
            assert "centroid" in elem
            assert len(elem["centroid"]) == 2

    def test_presets_returns_subregions(self, client_with_model):
        r = client_with_model.get("/api/zbudgets/presets")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 1
        assert data[0]["name"] == "Subregions"
        assert len(data[0]["zones"]) == 2

    def test_post_zones_creates_definition(self, client_with_model):
        payload = {
            "zones": [
                {"id": 1, "name": "North", "elements": [1, 2]},
                {"id": 2, "name": "South", "elements": [3, 4]},
            ],
            "extent": "horizontal",
        }
        r = client_with_model.post("/api/zbudgets/zones", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["n_zones"] == 2
        assert data["n_elements"] == 4

    def test_post_zones_empty_returns_400(self, client_with_model):
        payload = {"zones": [], "extent": "horizontal"}
        r = client_with_model.post("/api/zbudgets/zones", json=payload)
        assert r.status_code == 400

    def test_get_zones_after_post(self, client_with_model):
        payload = {
            "zones": [
                {"id": 1, "name": "Zone A", "elements": [1, 2]},
            ],
            "extent": "horizontal",
        }
        client_with_model.post("/api/zbudgets/zones", json=payload)
        r = client_with_model.get("/api/zbudgets/zones")
        assert r.status_code == 200
        data = r.json()
        assert data is not None
        assert data["n_zones"] == 1
        assert data["zones"][0]["name"] == "Zone A"

    def test_get_zones_empty(self, client_with_model):
        r = client_with_model.get("/api/zbudgets/zones")
        assert r.status_code == 200
        # No zones defined yet
        assert r.json() is None

    def test_glossary(self, client_with_model):
        r = client_with_model.get("/api/zbudgets/glossary")
        assert r.status_code == 200
        data = r.json()
        assert "gw" in data
        assert "rootzone" in data
        assert "lwu" in data
        assert len(data["gw"]) > 0


class TestZBudgetData:
    def test_columns_endpoint(self, client_with_model):
        reader = _make_mock_zbudget_reader()
        model_state._zbudget_readers["gw"] = reader
        r = client_with_model.get("/api/zbudgets/gw/columns")
        assert r.status_code == 200
        data = r.json()
        assert len(data["columns"]) == 3
        assert data["columns"][0]["name"] == "Deep Percolation"

    def test_data_endpoint(self, client_with_model):
        reader = _make_mock_zbudget_reader()
        model_state._zbudget_readers["gw"] = reader
        r = client_with_model.get("/api/zbudgets/gw/data?zone=Zone+1")
        assert r.status_code == 200
        data = r.json()
        assert data["location"] == "Zone 1"
        assert len(data["times"]) == 12
        assert len(data["columns"]) == 3
        assert "units_metadata" in data
        assert data["units_metadata"]["source_volume_unit"] == "FT3"

    def test_data_default_zone(self, client_with_model):
        reader = _make_mock_zbudget_reader()
        model_state._zbudget_readers["gw"] = reader
        r = client_with_model.get("/api/zbudgets/gw/data")
        assert r.status_code == 200
        # Should use first zone
        reader.get_dataframe.assert_called()

    def test_data_not_found(self, client_with_model):
        r = client_with_model.get("/api/zbudgets/gw/data?zone=Zone+1")
        assert r.status_code == 404

    def test_summary_endpoint(self, client_with_model):
        reader = _make_mock_zbudget_reader()
        model_state._zbudget_readers["gw"] = reader
        r = client_with_model.get("/api/zbudgets/gw/summary?zone=Zone+1")
        assert r.status_code == 200
        data = r.json()
        assert data["location"] == "Zone 1"
        assert data["n_timesteps"] == 12
        assert "totals" in data
        assert "averages" in data

    def test_columns_not_found(self, client_with_model):
        r = client_with_model.get("/api/zbudgets/gw/columns")
        assert r.status_code == 404


class TestGlossary:
    def test_glossary_has_expected_keys(self):
        assert "gw" in ZBUDGET_GLOSSARY
        assert "rootzone" in ZBUDGET_GLOSSARY
        assert "lwu" in ZBUDGET_GLOSSARY
        assert "Storage (+/-)" in ZBUDGET_GLOSSARY["gw"]
        assert "AG_Precip (+)" in ZBUDGET_GLOSSARY["rootzone"]
        assert "AG_Supply Requirement" in ZBUDGET_GLOSSARY["lwu"]
