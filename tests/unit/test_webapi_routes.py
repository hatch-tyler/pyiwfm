"""Unit tests for FastAPI web viewer endpoints.

Tests all route groups using TestClient with a mock model.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("fastapi", reason="FastAPI not available")
pytest.importorskip("pydantic", reason="Pydantic not available")

from fastapi.testclient import TestClient

from pyiwfm.core.mesh import AppGrid, Element, Node
from pyiwfm.visualization.webapi.config import model_state
from pyiwfm.visualization.webapi.server import create_app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_grid():
    """Create a simple 4-node quad grid for testing."""
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


def _make_mock_model(
    with_stratigraphy: bool = False,
    with_streams: bool = False,
    with_groundwater: bool = False,
):
    """Create a minimal mock IWFMModel."""
    grid = _make_grid()

    model = MagicMock()
    model.grid = grid
    model.name = "TestModel"
    model.n_nodes = 4
    model.n_elements = 1
    model.n_layers = 2
    model.n_lakes = 0
    model.has_streams = with_streams
    model.has_lakes = False
    model.streams = None
    model.lakes = None
    model.metadata = {}

    if with_stratigraphy:
        strat = MagicMock()
        strat.n_layers = 2
        strat.n_nodes = 4
        strat.gs_elev = np.array([100.0, 100.0, 100.0, 100.0])
        strat.top_elev = np.full((4, 1), 100.0)
        strat.bottom_elev = np.zeros((4, 2))
        strat.bottom_elev[:, 0] = 50.0
        strat.bottom_elev[:, 1] = 0.0
        model.stratigraphy = strat
    else:
        model.stratigraphy = None

    if with_streams:
        streams = MagicMock()
        streams.n_nodes = 2

        # Create mock reaches with stream nodes
        sn1 = MagicMock()
        sn1.id = 1
        sn1.groundwater_node = 1

        sn2 = MagicMock()
        sn2.id = 2
        sn2.groundwater_node = 2

        reach = MagicMock()
        reach.id = 1
        reach.stream_nodes = [sn1, sn2]

        streams.reaches = [reach]
        model.streams = streams
        model.has_streams = True
        model.n_stream_nodes = 2

    if with_groundwater:
        gw = MagicMock()
        gw.n_hydrograph_locations = 0
        gw.hydrograph_locations = []
        model.groundwater = gw
    else:
        model.groundwater = None

    return model


@pytest.fixture()
def client_no_model():
    """TestClient with no model loaded."""
    # Reset the global model state
    model_state._model = None
    model_state._layer_surface_cache = {}
    model_state._budget_readers = {}
    model_state._observations = {}

    app = create_app()
    return TestClient(app)


@pytest.fixture()
def client_with_model():
    """TestClient with a basic mock model loaded."""
    model = _make_mock_model(with_streams=True, with_stratigraphy=True)

    model_state._model = model
    model_state._layer_surface_cache = {}
    model_state._budget_readers = {}
    model_state._observations = {}
    model_state._crs = "+proj=utm +zone=10 +datum=NAD83 +units=us-ft +no_defs"
    model_state._transformer = None
    model_state._geojson_cache = {}
    model_state._head_loader = None
    model_state._gw_hydrograph_reader = None
    model_state._stream_hydrograph_reader = None
    model_state._results_dir = None

    app = create_app()
    yield TestClient(app)

    # Cleanup
    model_state._model = None


# ---------------------------------------------------------------------------
# Model routes
# ---------------------------------------------------------------------------


class TestModelRoutes:
    """Tests for /api/model/ endpoints."""

    def test_model_info_no_model(self, client_no_model):
        resp = client_no_model.get("/api/model/info")
        assert resp.status_code == 404

    def test_model_info_with_model(self, client_with_model):
        resp = client_with_model.get("/api/model/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_nodes"] == 4
        assert data["n_elements"] == 1
        assert data["has_streams"] is True

    def test_model_bounds_no_model(self, client_no_model):
        resp = client_no_model.get("/api/model/bounds")
        assert resp.status_code == 404

    def test_model_bounds_with_model(self, client_with_model):
        resp = client_with_model.get("/api/model/bounds")
        assert resp.status_code == 200
        data = resp.json()
        assert "xmin" in data
        assert "xmax" in data
        assert data["xmax"] >= data["xmin"]


# ---------------------------------------------------------------------------
# Stream routes
# ---------------------------------------------------------------------------


class TestStreamRoutes:
    """Tests for /api/streams endpoints."""

    def test_streams_no_model(self, client_no_model):
        resp = client_no_model.get("/api/streams")
        assert resp.status_code == 404

    def test_streams_with_model(self, client_with_model):
        resp = client_with_model.get("/api/streams")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_nodes"] == 2
        assert data["n_reaches"] == 1
        # Verify z is present (A3 fix)
        assert "z" in data["nodes"][0]
        assert data["nodes"][0]["z"] == 100.0  # from gs_elev

    def test_streams_no_streams(self, client_no_model):
        """Model loaded but no streams."""
        model = _make_mock_model(with_streams=False)
        model_state._model = model
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/streams")
            assert resp.status_code == 404
        finally:
            model_state._model = None


# ---------------------------------------------------------------------------
# Budget routes
# ---------------------------------------------------------------------------


class TestBudgetRoutes:
    """Tests for /api/budgets/ endpoints."""

    def test_budget_types_no_model(self, client_no_model):
        resp = client_no_model.get("/api/budgets/types")
        assert resp.status_code == 404

    def test_budget_types_with_model(self, client_with_model):
        resp = client_with_model.get("/api/budgets/types")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_budget_locations_not_found(self, client_with_model):
        resp = client_with_model.get("/api/budgets/nonexistent/locations")
        assert resp.status_code == 404

    def test_budget_columns_not_found(self, client_with_model):
        resp = client_with_model.get("/api/budgets/nonexistent/columns")
        assert resp.status_code == 404

    def test_budget_data_not_found(self, client_with_model):
        resp = client_with_model.get("/api/budgets/nonexistent/data")
        assert resp.status_code == 404

    def test_budget_summary_not_found(self, client_with_model):
        resp = client_with_model.get("/api/budgets/nonexistent/summary")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Results routes
# ---------------------------------------------------------------------------


class TestResultsRoutes:
    """Tests for /api/results/ endpoints."""

    def test_results_info_no_model(self, client_no_model):
        resp = client_no_model.get("/api/results/info")
        assert resp.status_code == 404

    def test_results_info_with_model(self, client_with_model):
        resp = client_with_model.get("/api/results/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "has_results" in data
        assert "available_budgets" in data

    def test_hydrograph_locations_no_model(self, client_no_model):
        resp = client_no_model.get("/api/results/hydrograph-locations")
        assert resp.status_code == 404

    def test_hydrograph_locations_with_model(self, client_with_model):
        resp = client_with_model.get("/api/results/hydrograph-locations")
        assert resp.status_code == 200
        data = resp.json()
        assert "gw" in data
        assert "stream" in data


# ---------------------------------------------------------------------------
# Observations routes
# ---------------------------------------------------------------------------


class TestObservationRoutes:
    """Tests for /api/observations/ endpoints."""

    def test_list_observations_empty(self, client_with_model):
        resp = client_with_model.get("/api/observations")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_observation_not_found(self, client_with_model):
        resp = client_with_model.get("/api/observations/nonexistent/data")
        assert resp.status_code == 404

    def test_delete_observation_not_found(self, client_with_model):
        resp = client_with_model.delete("/api/observations/nonexistent")
        assert resp.status_code == 404

    def test_upload_and_list_observation(self, client_with_model):
        """Test uploading a CSV observation file and listing it."""
        csv_content = "date,value\n2020-01-01,10.5\n2020-02-01,11.0\n2020-03-01,10.8\n"
        files = {"file": ("test_obs.csv", io.BytesIO(csv_content.encode()), "text/csv")}
        resp = client_with_model.post("/api/observations/upload", files=files)
        assert resp.status_code == 200
        data = resp.json()
        obs_id = data["observation_id"]
        assert data["n_records"] > 0

        # List should now show the observation
        resp2 = client_with_model.get("/api/observations")
        assert resp2.status_code == 200
        obs_list = resp2.json()
        assert len(obs_list) == 1
        assert obs_list[0]["id"] == obs_id

        # Get observation data
        resp3 = client_with_model.get(f"/api/observations/{obs_id}/data")
        assert resp3.status_code == 200

        # Delete
        resp4 = client_with_model.delete(f"/api/observations/{obs_id}")
        assert resp4.status_code == 200

        # Should be empty again
        resp5 = client_with_model.get("/api/observations")
        assert resp5.json() == []


# ---------------------------------------------------------------------------
# Properties routes
# ---------------------------------------------------------------------------


class TestPropertyRoutes:
    """Tests for /api/properties endpoints."""

    def test_properties_list_no_model(self, client_no_model):
        resp = client_no_model.get("/api/properties")
        assert resp.status_code == 404

    def test_properties_list_with_model(self, client_with_model):
        resp = client_with_model.get("/api/properties")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
