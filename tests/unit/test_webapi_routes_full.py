"""Comprehensive tests for all FastAPI web viewer routes.

Covers results, budgets, observations, streams, slices, properties,
and remaining model/mesh routes not covered by test_webapi_routes.py.
"""

from __future__ import annotations

import io
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi", reason="FastAPI not available")
pydantic = pytest.importorskip("pydantic", reason="Pydantic not available")

from fastapi.testclient import TestClient  # noqa: E402

from pyiwfm.core.mesh import AppGrid, Element, Node  # noqa: E402
from pyiwfm.visualization.webapi.config import model_state  # noqa: E402
from pyiwfm.visualization.webapi.server import create_app  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
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


def _make_mock_model(**kwargs):
    """Create a minimal mock IWFMModel."""
    model = MagicMock()
    model.name = "TestModel"
    model.grid = _make_grid()
    model.metadata = kwargs.get("metadata", {})
    model.has_streams = kwargs.get("with_streams", False)
    model.has_lakes = False
    model.n_nodes = 4
    model.n_elements = 1
    model.n_layers = 2
    model.n_lakes = 0
    model.n_stream_nodes = 0

    # Stratigraphy
    if kwargs.get("with_stratigraphy", True):
        strat = MagicMock()
        strat.n_layers = 2
        strat.n_nodes = 4
        strat.gs_elev = np.array([100.0, 100.0, 100.0, 100.0])
        strat.top_elev = np.full((4, 2), 100.0)
        strat.top_elev[:, 1] = 50.0
        strat.bottom_elev = np.zeros((4, 2))
        strat.bottom_elev[:, 0] = 50.0
        strat.bottom_elev[:, 1] = 0.0
        model.stratigraphy = strat
    else:
        model.stratigraphy = None

    # Streams
    if kwargs.get("with_streams", False):
        streams = MagicMock()
        streams.n_nodes = 2

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
    else:
        model.streams = None

    # Groundwater
    if kwargs.get("with_groundwater", False):
        gw = MagicMock()
        gw.n_hydrograph_locations = 2
        loc1 = MagicMock()
        loc1.x = 50.0
        loc1.y = 50.0
        loc1.name = "Well-1"
        loc1.layer = 1
        loc2 = MagicMock()
        loc2.x = 75.0
        loc2.y = 75.0
        loc2.name = "Well-2"
        loc2.layer = 2
        gw.hydrograph_locations = [loc1, loc2]
        gw.aquifer_params = None
        model.groundwater = gw
    else:
        model.groundwater = None

    return model


def _make_mock_head_loader(n_frames=5, n_nodes=4, n_layers=2):
    """Create a mock head data loader."""
    loader = MagicMock()
    loader.n_frames = n_frames
    base_time = datetime(2020, 1, 1)
    loader.times = [base_time + timedelta(days=30 * i) for i in range(n_frames)]

    def get_frame(timestep):
        return np.random.default_rng(42).random((n_nodes, n_layers)) * 100.0

    loader.get_frame = get_frame
    return loader


def _make_mock_gw_hydrograph_reader(n_columns=2, n_timesteps=10):
    """Create a mock GW hydrograph reader."""
    reader = MagicMock()
    reader.n_columns = n_columns
    reader.n_timesteps = n_timesteps
    base_time = datetime(2020, 1, 1)

    def get_time_series(col_idx):
        times_list = [(base_time + timedelta(days=30 * i)).isoformat() for i in range(n_timesteps)]
        values_list = (np.arange(n_timesteps, dtype=float) + col_idx).tolist()
        return times_list, values_list

    reader.get_time_series = get_time_series
    return reader


def _make_mock_stream_hydrograph_reader(n_columns=3, n_timesteps=10):
    """Create a mock stream hydrograph reader."""
    reader = MagicMock()
    reader.n_columns = n_columns
    reader.n_timesteps = n_timesteps
    reader.hydrograph_ids = [1, 2, 3]
    base_time = datetime(2020, 1, 1)

    def find_column_by_node_id(node_id):
        if node_id in reader.hydrograph_ids:
            return reader.hydrograph_ids.index(node_id)
        return None

    reader.find_column_by_node_id = find_column_by_node_id

    def get_time_series(col_idx):
        times_list = [(base_time + timedelta(days=30 * i)).isoformat() for i in range(n_timesteps)]
        values_list = (np.arange(n_timesteps, dtype=float) * 10 + col_idx).tolist()
        return times_list, values_list

    reader.get_time_series = get_time_series
    return reader


def _make_mock_budget_reader(locations=None, columns=None, n_timesteps=5):
    """Create a mock budget reader."""
    reader = MagicMock()

    if locations is None:
        locations = ["Region 1", "Region 2"]
    if columns is None:
        columns = ["Deep Percolation", "Gain from Stream", "Net Change"]

    reader.locations = locations
    reader.descriptor = "GW Budget"

    def get_location_index(loc):
        if isinstance(loc, int):
            return loc
        for i, name in enumerate(locations):
            if name == loc:
                return i
        raise KeyError(f"Location '{loc}' not found")

    reader.get_location_index = get_location_index

    def get_column_headers(loc):
        return columns

    reader.get_column_headers = get_column_headers

    def get_values(loc, col_indices=None):
        n_cols = len(columns)
        if col_indices is not None:
            n_cols = len(col_indices)
        times_arr = np.arange(n_timesteps)
        values_arr = np.random.default_rng(42).random((n_timesteps, n_cols)) * 1000
        return times_arr, values_arr

    reader.get_values = get_values

    # Mock header for timestamp and column types
    header = MagicMock()
    ts = MagicMock()
    ts.start_datetime = datetime(2020, 1, 1)
    ts.delta_t_minutes = 43200  # 30 days in minutes
    ts.unit = "1MON"
    header.timestep = ts

    loc_data = MagicMock()
    loc_data.column_types = [1, 1, 1]
    loc_data.n_columns = len(columns)
    header.location_data = [loc_data]

    reader.header = header

    return reader


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
    model_state._observations = {}
    model_state._results_dir = None
    # Restore any monkey-patched methods back to the class originals
    for attr in (
        "get_budget_reader",
        "get_available_budgets",
        "reproject_coords",
        "get_stream_reach_boundaries",
        "get_head_loader",
        "get_gw_hydrograph_reader",
        "get_stream_hydrograph_reader",
        "get_area_manager",
        "get_subsidence_reader",
    ):
        if attr in model_state.__dict__:
            del model_state.__dict__[attr]


@pytest.fixture()
def client_no_model():
    """TestClient with no model loaded."""
    _reset_model_state()
    app = create_app()
    yield TestClient(app)
    _reset_model_state()


@pytest.fixture()
def client_with_model():
    """TestClient with a basic mock model loaded."""
    _reset_model_state()
    model = _make_mock_model(with_streams=True, with_stratigraphy=True)
    model_state._model = model
    app = create_app()
    yield TestClient(app), model
    _reset_model_state()


@pytest.fixture()
def client_with_gw():
    """TestClient with mock model that has groundwater hydrograph data."""
    _reset_model_state()
    model = _make_mock_model(
        with_streams=True,
        with_stratigraphy=True,
        with_groundwater=True,
    )
    model_state._model = model
    app = create_app()
    yield TestClient(app), model
    _reset_model_state()


# ===========================================================================
# 4A. routes/results.py tests (~15 tests)
# ===========================================================================


class TestResultsInfo:
    """Tests for GET /api/results/info."""

    def test_results_info_no_model(self, client_no_model):
        resp = client_no_model.get("/api/results/info")
        assert resp.status_code == 404

    def test_results_info_success(self, client_with_model):
        client, model = client_with_model
        resp = client.get("/api/results/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "has_results" in data
        assert "available_budgets" in data
        assert "n_head_timesteps" in data
        assert "n_gw_hydrographs" in data
        assert "n_stream_hydrographs" in data

    def test_results_info_no_results_available(self, client_with_model):
        """When model has no head data or budgets, has_results should be False."""
        client, model = client_with_model
        resp = client.get("/api/results/info")
        data = resp.json()
        assert data["has_results"] is False
        assert data["n_head_timesteps"] == 0

    def test_results_info_with_gw_hydrographs(self, client_with_gw):
        """When model has GW hydrograph locations, they are reported."""
        client, model = client_with_gw
        resp = client.get("/api/results/info")
        data = resp.json()
        assert data["n_gw_hydrographs"] == 2


class TestResultsHeads:
    """Tests for GET /api/results/heads."""

    def test_heads_no_loader(self, client_with_model):
        """404 when no head data loader is available."""
        client, model = client_with_model
        resp = client.get("/api/results/heads")
        assert resp.status_code == 404
        assert "No head data" in resp.json()["detail"]

    def test_heads_success(self, client_with_model):
        """Success returns timestep, datetime, layer, values."""
        client, model = client_with_model
        loader = _make_mock_head_loader(n_frames=5, n_nodes=4, n_layers=2)
        model_state._head_loader = loader

        resp = client.get("/api/results/heads?timestep=0&layer=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["timestep_index"] == 0
        assert data["layer"] == 1
        assert data["datetime"] is not None
        assert len(data["values"]) == 4

    def test_heads_timestep_out_of_range(self, client_with_model):
        """400 when timestep exceeds available frames."""
        client, model = client_with_model
        loader = _make_mock_head_loader(n_frames=5)
        model_state._head_loader = loader

        resp = client.get("/api/results/heads?timestep=10&layer=1")
        assert resp.status_code == 400
        assert "out of range" in resp.json()["detail"]

    def test_heads_layer_out_of_range(self, client_with_model):
        """400 when layer exceeds available layers."""
        client, model = client_with_model
        loader = _make_mock_head_loader(n_frames=5, n_layers=2)
        model_state._head_loader = loader

        resp = client.get("/api/results/heads?timestep=0&layer=5")
        assert resp.status_code == 400
        assert "Layer" in resp.json()["detail"]

    def test_heads_default_params(self, client_with_model):
        """Default params: timestep=0, layer=1."""
        client, model = client_with_model
        loader = _make_mock_head_loader(n_frames=3, n_nodes=4, n_layers=2)
        model_state._head_loader = loader

        resp = client.get("/api/results/heads")
        assert resp.status_code == 200
        data = resp.json()
        assert data["timestep_index"] == 0
        assert data["layer"] == 1


class TestResultsHeadTimes:
    """Tests for GET /api/results/head-times."""

    def test_head_times_no_loader(self, client_with_model):
        client, model = client_with_model
        resp = client.get("/api/results/head-times")
        assert resp.status_code == 404

    def test_head_times_success(self, client_with_model):
        client, model = client_with_model
        loader = _make_mock_head_loader(n_frames=5)
        model_state._head_loader = loader

        resp = client.get("/api/results/head-times")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_timesteps"] == 5
        assert len(data["times"]) == 5
        # Times should be ISO format strings
        assert "2020" in data["times"][0]


class TestResultsHydrographLocations:
    """Tests for GET /api/results/hydrograph-locations."""

    def test_hydrograph_locations_no_model(self, client_no_model):
        resp = client_no_model.get("/api/results/hydrograph-locations")
        assert resp.status_code == 404

    def test_hydrograph_locations_success(self, client_with_model):
        client, model = client_with_model
        resp = client.get("/api/results/hydrograph-locations")
        assert resp.status_code == 200
        data = resp.json()
        assert "gw" in data
        assert "stream" in data

    def test_hydrograph_locations_with_gw(self, client_with_gw):
        """With GW hydrograph locations, they appear in the response."""
        client, model = client_with_gw
        resp = client.get("/api/results/hydrograph-locations")
        data = resp.json()
        assert len(data["gw"]) == 2
        assert data["gw"][0]["name"] == "Well-1"
        assert data["gw"][0]["id"] == 1
        assert data["gw"][1]["name"] == "Well-2"


class TestResultsHydrograph:
    """Tests for GET /api/results/hydrograph."""

    def test_hydrograph_gw_success(self, client_with_gw):
        """GW hydrograph returns time series."""
        client, model = client_with_gw
        reader = _make_mock_gw_hydrograph_reader(n_columns=2, n_timesteps=10)
        model_state._gw_hydrograph_reader = reader

        resp = client.get("/api/results/hydrograph?type=gw&location_id=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "gw"
        assert data["location_id"] == 1
        assert len(data["times"]) == 10
        assert len(data["values"]) == 10
        assert data["units"] == "ft"

    def test_hydrograph_gw_no_reader(self, client_with_model):
        """404 when no GW hydrograph reader is available."""
        client, model = client_with_model
        resp = client.get("/api/results/hydrograph?type=gw&location_id=1")
        assert resp.status_code == 404

    def test_hydrograph_gw_out_of_range(self, client_with_gw):
        """404 when location_id exceeds available columns."""
        client, model = client_with_gw
        reader = _make_mock_gw_hydrograph_reader(n_columns=2)
        model_state._gw_hydrograph_reader = reader

        resp = client.get("/api/results/hydrograph?type=gw&location_id=99")
        assert resp.status_code == 404

    def test_hydrograph_stream_success(self, client_with_model):
        """Stream hydrograph returns time series."""
        client, model = client_with_model
        reader = _make_mock_stream_hydrograph_reader(n_columns=3, n_timesteps=10)
        model_state._stream_hydrograph_reader = reader

        resp = client.get("/api/results/hydrograph?type=stream&location_id=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "stream"
        assert data["location_id"] == 1
        assert data["units"] == "cfs"
        assert len(data["times"]) == 10

    def test_hydrograph_stream_not_found(self, client_with_model):
        """404 when stream node not in hydrograph data."""
        client, model = client_with_model
        reader = _make_mock_stream_hydrograph_reader()
        reader.find_column_by_node_id = MagicMock(return_value=None)
        reader.hydrograph_ids = [10, 20, 30]
        model_state._stream_hydrograph_reader = reader

        resp = client.get("/api/results/hydrograph?type=stream&location_id=999")
        assert resp.status_code == 404

    def test_hydrograph_stream_no_reader(self, client_with_model):
        """404 when no stream hydrograph reader is available."""
        client, model = client_with_model
        resp = client.get("/api/results/hydrograph?type=stream&location_id=1")
        assert resp.status_code == 404

    def test_hydrograph_unknown_type(self, client_with_model):
        """400 for unknown hydrograph type."""
        client, model = client_with_model
        resp = client.get("/api/results/hydrograph?type=unknown&location_id=1")
        assert resp.status_code == 400
        assert "Unknown hydrograph type" in resp.json()["detail"]

    def test_hydrograph_subsidence_type(self, client_with_model):
        """404 for subsidence type (not implemented)."""
        client, model = client_with_model
        resp = client.get("/api/results/hydrograph?type=subsidence&location_id=1")
        assert resp.status_code == 404
        assert "subsidence" in resp.json()["detail"].lower()


# ===========================================================================
# 4B. routes/budgets.py tests (~15 tests)
# ===========================================================================


class TestBudgetTypes:
    """Tests for GET /api/budgets/types."""

    def test_budget_types_no_model(self, client_no_model):
        resp = client_no_model.get("/api/budgets/types")
        assert resp.status_code == 404

    def test_budget_types_empty(self, client_with_model):
        """No budget files available returns empty list."""
        client, model = client_with_model
        resp = client.get("/api/budgets/types")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_budget_types_with_cached_readers(self, client_with_model):
        """When budget readers are cached, types are reported."""
        client, model = client_with_model
        reader = _make_mock_budget_reader()
        model_state._budget_readers["gw"] = reader

        # get_available_budgets() checks model.metadata for file paths.
        # We need to also check that /types returns them. The actual route
        # calls model_state.get_available_budgets() which checks files on disk.
        # With mocked model that has no metadata, it returns [].
        resp = client.get("/api/budgets/types")
        assert resp.status_code == 200
        # Returns empty since metadata has no file paths pointing to existing files.
        assert isinstance(resp.json(), list)


class TestBudgetLocations:
    """Tests for GET /api/budgets/{type}/locations."""

    def test_budget_locations_not_found(self, client_with_model):
        """404 for unknown budget type."""
        client, model = client_with_model
        resp = client.get("/api/budgets/nonexistent/locations")
        assert resp.status_code == 404

    def test_budget_locations_success(self, client_with_model):
        """Success returns locations list."""
        client, model = client_with_model
        reader = _make_mock_budget_reader(locations=["Region 1", "Region 2"])
        model_state._budget_readers["gw"] = reader

        resp = client.get("/api/budgets/gw/locations")
        assert resp.status_code == 200
        data = resp.json()
        assert "locations" in data
        assert len(data["locations"]) == 2
        assert data["locations"][0]["id"] == 0
        assert data["locations"][0]["name"] == "Region 1"
        assert data["locations"][1]["name"] == "Region 2"

    def test_budget_locations_single(self, client_with_model):
        """Budget with a single location."""
        client, model = client_with_model
        reader = _make_mock_budget_reader(locations=["Whole Model"])
        model_state._budget_readers["stream"] = reader

        resp = client.get("/api/budgets/stream/locations")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["locations"]) == 1


class TestBudgetColumns:
    """Tests for GET /api/budgets/{type}/columns."""

    def test_budget_columns_not_found(self, client_with_model):
        """404 for unknown budget type."""
        client, model = client_with_model
        resp = client.get("/api/budgets/nonexistent/columns")
        assert resp.status_code == 404

    def test_budget_columns_success(self, client_with_model):
        """Success returns columns with units."""
        client, model = client_with_model
        columns = ["Deep Percolation", "Gain from Stream", "Pumping"]
        reader = _make_mock_budget_reader(columns=columns)
        model_state._budget_readers["gw"] = reader

        resp = client.get("/api/budgets/gw/columns")
        assert resp.status_code == 200
        data = resp.json()
        assert "columns" in data
        assert len(data["columns"]) == 3
        assert data["columns"][0]["name"] == "Deep Percolation"
        assert "units" in data["columns"][0]

    def test_budget_columns_with_location_param(self, client_with_model):
        """Columns route accepts a location query parameter."""
        client, model = client_with_model
        reader = _make_mock_budget_reader()
        model_state._budget_readers["gw"] = reader

        resp = client.get("/api/budgets/gw/columns?location=Region 1")
        assert resp.status_code == 200
        data = resp.json()
        assert "columns" in data


class TestBudgetData:
    """Tests for GET /api/budgets/{type}/data."""

    def test_budget_data_not_found(self, client_with_model):
        """404 for unknown budget type."""
        client, model = client_with_model
        resp = client.get("/api/budgets/nonexistent/data")
        assert resp.status_code == 404

    def test_budget_data_success(self, client_with_model):
        """Success returns times and column values."""
        client, model = client_with_model
        reader = _make_mock_budget_reader(n_timesteps=5)
        model_state._budget_readers["gw"] = reader

        resp = client.get("/api/budgets/gw/data")
        assert resp.status_code == 200
        data = resp.json()
        assert "times" in data
        assert "columns" in data
        assert "location" in data
        assert len(data["times"]) == 5
        # Each column should have values
        for col in data["columns"]:
            assert "name" in col
            assert "values" in col
            assert len(col["values"]) == 5

    def test_budget_data_with_column_filter(self, client_with_model):
        """Data route with specific column indices."""
        client, model = client_with_model
        reader = _make_mock_budget_reader(n_timesteps=5)
        model_state._budget_readers["gw"] = reader

        resp = client.get("/api/budgets/gw/data?columns=0,1")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["columns"]) == 2

    def test_budget_data_with_location(self, client_with_model):
        """Data route with location parameter."""
        client, model = client_with_model
        reader = _make_mock_budget_reader()
        model_state._budget_readers["gw"] = reader

        resp = client.get("/api/budgets/gw/data?location=Region 1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["location"] == "Region 1"

    def test_budget_data_all_columns(self, client_with_model):
        """Default columns=all returns all columns."""
        client, model = client_with_model
        columns = ["Col A", "Col B", "Col C"]
        reader = _make_mock_budget_reader(columns=columns, n_timesteps=3)
        model_state._budget_readers["gw"] = reader

        resp = client.get("/api/budgets/gw/data")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["columns"]) == 3


class TestBudgetSummary:
    """Tests for GET /api/budgets/{type}/summary."""

    def test_budget_summary_not_found(self, client_with_model):
        """404 for unknown budget type."""
        client, model = client_with_model
        resp = client.get("/api/budgets/nonexistent/summary")
        assert resp.status_code == 404

    def test_budget_summary_success(self, client_with_model):
        """Success returns totals and averages."""
        client, model = client_with_model
        columns = ["Deep Percolation", "Gain from Stream"]
        reader = _make_mock_budget_reader(columns=columns, n_timesteps=5)
        model_state._budget_readers["gw"] = reader

        resp = client.get("/api/budgets/gw/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert "totals" in data
        assert "averages" in data
        assert "n_timesteps" in data
        assert data["n_timesteps"] == 5
        # Check keys match column names
        for col_name in columns:
            assert col_name in data["totals"]
            assert col_name in data["averages"]

    def test_budget_summary_with_location(self, client_with_model):
        """Summary route with location parameter."""
        client, model = client_with_model
        reader = _make_mock_budget_reader()
        model_state._budget_readers["gw"] = reader

        resp = client.get("/api/budgets/gw/summary?location=Region 1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["location"] == "Region 1"


# ===========================================================================
# 4C. routes/observations.py tests (~12 tests)
# ===========================================================================


class TestObservationsUpload:
    """Tests for POST /api/observations/upload."""

    def test_upload_no_model(self, client_no_model):
        """404 when no model loaded."""
        csv_content = "datetime,value\n2020-01-01,100.0\n"
        files = {"file": ("test.csv", io.BytesIO(csv_content.encode()), "text/csv")}
        resp = client_no_model.post("/api/observations/upload", files=files)
        assert resp.status_code == 404

    def test_upload_success(self, client_with_model):
        """Successful upload of a CSV observation file."""
        client, model = client_with_model
        csv_content = "datetime,value\n2020-01-01,100.0\n2020-02-01,200.0\n2020-03-01,150.0\n"
        files = {"file": ("test_obs.csv", io.BytesIO(csv_content.encode()), "text/csv")}
        resp = client.post("/api/observations/upload", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_records"] == 3
        assert data["filename"] == "test_obs.csv"
        assert "observation_id" in data
        assert data["start_time"] is not None
        assert data["end_time"] is not None

    def test_upload_iso_datetime_format(self, client_with_model):
        """Upload with ISO datetime format (YYYY-MM-DDTHH:MM:SS)."""
        client, model = client_with_model
        csv_content = "datetime,value\n2020-01-01T12:00:00,100.0\n2020-02-01T12:00:00,200.0\n"
        files = {"file": ("obs_iso.csv", io.BytesIO(csv_content.encode()), "text/csv")}
        resp = client.post("/api/observations/upload", files=files)
        assert resp.status_code == 200
        assert resp.json()["n_records"] == 2

    def test_upload_us_datetime_format(self, client_with_model):
        """Upload with US date format (MM/DD/YYYY)."""
        client, model = client_with_model
        csv_content = "datetime,value\n01/15/2020,100.0\n02/15/2020,200.0\n"
        files = {"file": ("obs_us.csv", io.BytesIO(csv_content.encode()), "text/csv")}
        resp = client.post("/api/observations/upload", files=files)
        assert resp.status_code == 200
        assert resp.json()["n_records"] == 2

    def test_upload_no_valid_data(self, client_with_model):
        """400 when CSV has no parseable data rows."""
        client, model = client_with_model
        csv_content = "col1,col2\nnotadate,notanumber\nbaddate,badval\n"
        files = {"file": ("bad.csv", io.BytesIO(csv_content.encode()), "text/csv")}
        resp = client.post("/api/observations/upload", files=files)
        assert resp.status_code == 400
        assert "No valid data" in resp.json()["detail"]

    def test_upload_empty_csv(self, client_with_model):
        """400 for an empty CSV file."""
        client, model = client_with_model
        csv_content = "datetime,value\n"
        files = {"file": ("empty.csv", io.BytesIO(csv_content.encode()), "text/csv")}
        resp = client.post("/api/observations/upload", files=files)
        assert resp.status_code == 400

    def test_upload_datetime_with_time(self, client_with_model):
        """Upload with datetime that includes time (YYYY-MM-DD HH:MM:SS)."""
        client, model = client_with_model
        csv_content = "datetime,value\n2020-01-01 14:30:00,100.0\n2020-02-01 08:15:00,200.0\n"
        files = {"file": ("obs_dt.csv", io.BytesIO(csv_content.encode()), "text/csv")}
        resp = client.post("/api/observations/upload", files=files)
        assert resp.status_code == 200
        assert resp.json()["n_records"] == 2


class TestObservationsList:
    """Tests for GET /api/observations."""

    def test_list_observations_empty(self, client_with_model):
        client, model = client_with_model
        resp = client.get("/api/observations")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_observations_after_upload(self, client_with_model):
        """After uploading, observations appear in list."""
        client, model = client_with_model
        csv_content = "datetime,value\n2020-01-01,100.0\n2020-02-01,200.0\n"
        files = {"file": ("test.csv", io.BytesIO(csv_content.encode()), "text/csv")}
        client.post("/api/observations/upload", files=files)

        resp = client.get("/api/observations")
        assert resp.status_code == 200
        obs_list = resp.json()
        assert len(obs_list) == 1
        assert "id" in obs_list[0]
        assert obs_list[0]["filename"] == "test.csv"
        assert obs_list[0]["n_records"] == 2


class TestObservationsData:
    """Tests for GET /api/observations/{id}/data."""

    def test_get_observation_data_not_found(self, client_with_model):
        client, model = client_with_model
        resp = client.get("/api/observations/nonexistent/data")
        assert resp.status_code == 404

    def test_get_observation_data_success(self, client_with_model):
        """Upload, then retrieve the data."""
        client, model = client_with_model
        csv_content = "datetime,value\n2020-01-01,100.0\n2020-02-01,200.0\n"
        files = {"file": ("test.csv", io.BytesIO(csv_content.encode()), "text/csv")}
        upload_resp = client.post("/api/observations/upload", files=files)
        obs_id = upload_resp.json()["observation_id"]

        resp = client.get(f"/api/observations/{obs_id}/data")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["times"]) == 2
        assert len(data["values"]) == 2
        assert data["values"][0] == 100.0
        assert data["values"][1] == 200.0


class TestObservationsLocation:
    """Tests for PUT /api/observations/{id}/location."""

    def test_set_location_not_found(self, client_with_model):
        client, model = client_with_model
        resp = client.put(
            "/api/observations/nonexistent/location",
            params={"location_id": 1, "location_type": "gw"},
        )
        assert resp.status_code == 404

    def test_set_location_success(self, client_with_model):
        """Associate an observation with a hydrograph location."""
        client, model = client_with_model
        csv_content = "datetime,value\n2020-01-01,100.0\n"
        files = {"file": ("test.csv", io.BytesIO(csv_content.encode()), "text/csv")}
        upload_resp = client.post("/api/observations/upload", files=files)
        obs_id = upload_resp.json()["observation_id"]

        resp = client.put(
            f"/api/observations/{obs_id}/location",
            params={"location_id": 5, "location_type": "gw"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["location_id"] == 5


class TestObservationsDelete:
    """Tests for DELETE /api/observations/{id}."""

    def test_delete_not_found(self, client_with_model):
        client, model = client_with_model
        resp = client.delete("/api/observations/nonexistent")
        assert resp.status_code == 404

    def test_delete_success(self, client_with_model):
        """Upload and delete an observation."""
        client, model = client_with_model
        csv_content = "datetime,value\n2020-01-01,100.0\n"
        files = {"file": ("test.csv", io.BytesIO(csv_content.encode()), "text/csv")}
        upload_resp = client.post("/api/observations/upload", files=files)
        obs_id = upload_resp.json()["observation_id"]

        resp = client.delete(f"/api/observations/{obs_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

        # Verify deleted
        resp2 = client.get(f"/api/observations/{obs_id}/data")
        assert resp2.status_code == 404

    def test_delete_then_list_empty(self, client_with_model):
        """After deletion, the list should not include the deleted observation."""
        client, model = client_with_model
        csv_content = "datetime,value\n2020-01-01,100.0\n"
        files = {"file": ("test.csv", io.BytesIO(csv_content.encode()), "text/csv")}
        upload_resp = client.post("/api/observations/upload", files=files)
        obs_id = upload_resp.json()["observation_id"]

        client.delete(f"/api/observations/{obs_id}")
        resp = client.get("/api/observations")
        assert resp.json() == []


# ===========================================================================
# 4D. routes/mesh.py tests (~10 tests)
# ===========================================================================


class TestMeshJson:
    """Tests for GET /api/mesh/json."""

    def test_mesh_json_no_model(self, client_no_model):
        resp = client_no_model.get("/api/mesh/json")
        assert resp.status_code == 404

    def test_mesh_json_success(self, client_with_model):
        """Success returns points, polys, layer arrays."""
        client, model = client_with_model
        mock_data = {
            "n_points": 8,
            "n_cells": 6,
            "n_layers": 2,
            "points": [0.0] * 24,
            "polys": [3, 0, 1, 2] * 6,
            "layer": [1, 1, 1, 2, 2, 2],
        }
        model_state._layer_surface_cache = {0: mock_data}

        resp = client.get("/api/mesh/json")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_points"] == 8
        assert data["n_cells"] == 6
        assert "points" in data
        assert "polys" in data
        assert "layer" in data

    def test_mesh_json_with_layer_param(self, client_with_model):
        """Layer parameter filters to a specific layer."""
        client, model = client_with_model
        mock_data = {
            "n_points": 4,
            "n_cells": 3,
            "n_layers": 1,
            "points": [0.0] * 12,
            "polys": [3, 0, 1, 2] * 3,
            "layer": [1, 1, 1],
        }
        model_state._layer_surface_cache = {1: mock_data}

        resp = client.get("/api/mesh/json?layer=1")
        assert resp.status_code == 200
        data = resp.json()
        assert all(v == 1 for v in data["layer"])

    def test_mesh_json_invalid_layer(self, client_with_model):
        """Layer exceeding model layers returns 400."""
        client, model = client_with_model
        resp = client.get("/api/mesh/json?layer=99")
        assert resp.status_code == 400
        assert "exceeds" in resp.json()["detail"]


class TestMeshGeoJson:
    """Tests for GET /api/mesh/geojson."""

    def test_mesh_geojson_no_model(self, client_no_model):
        resp = client_no_model.get("/api/mesh/geojson")
        assert resp.status_code == 404

    def test_mesh_geojson_success(self, client_with_model):
        """Success returns GeoJSON FeatureCollection."""
        client, model = client_with_model
        resp = client.get("/api/mesh/geojson")
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "FeatureCollection"
        assert "features" in data
        assert len(data["features"]) == 1  # 1 element in grid
        feat = data["features"][0]
        assert feat["type"] == "Feature"
        assert feat["geometry"]["type"] == "Polygon"
        assert feat["properties"]["element_id"] == 1

    def test_mesh_geojson_with_layer(self, client_with_model):
        """GeoJSON with different layer parameter."""
        client, model = client_with_model
        resp = client.get("/api/mesh/geojson?layer=2")
        assert resp.status_code == 200
        data = resp.json()
        feat = data["features"][0]
        assert feat["properties"]["layer"] == 2


class TestMeshHeadMap:
    """Tests for GET /api/mesh/head-map."""

    def test_head_map_no_model(self, client_no_model):
        resp = client_no_model.get("/api/mesh/head-map")
        assert resp.status_code == 404

    def test_head_map_no_head_data(self, client_with_model):
        """404 when no head data loader is available."""
        client, model = client_with_model
        resp = client.get("/api/mesh/head-map")
        assert resp.status_code == 404
        assert "No head data" in resp.json()["detail"]

    def test_head_map_success(self, client_with_model):
        """Success returns GeoJSON with head property on features."""
        client, model = client_with_model
        loader = _make_mock_head_loader(n_frames=3, n_nodes=4, n_layers=2)
        model_state._head_loader = loader

        resp = client.get("/api/mesh/head-map?timestep=0&layer=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "FeatureCollection"
        assert "metadata" in data
        assert data["metadata"]["timestep_index"] == 0
        assert data["metadata"]["layer"] == 1
        # Features should have head property
        for feat in data["features"]:
            assert "head" in feat["properties"]

    def test_head_map_timestep_out_of_range(self, client_with_model):
        """400 when timestep exceeds available frames."""
        client, model = client_with_model
        loader = _make_mock_head_loader(n_frames=3)
        model_state._head_loader = loader

        resp = client.get("/api/mesh/head-map?timestep=10&layer=1")
        assert resp.status_code == 400

    def test_head_map_layer_out_of_range(self, client_with_model):
        """400 when layer exceeds available layers."""
        client, model = client_with_model
        loader = _make_mock_head_loader(n_frames=3, n_layers=2)
        model_state._head_loader = loader

        resp = client.get("/api/mesh/head-map?timestep=0&layer=5")
        assert resp.status_code == 400


# ===========================================================================
# 4E. routes/streams.py tests (~8 tests)
# ===========================================================================


class TestStreamRoutesFull:
    """Additional tests for /api/streams endpoints."""

    def test_streams_no_model(self, client_no_model):
        resp = client_no_model.get("/api/streams")
        assert resp.status_code == 404

    def test_streams_success(self, client_with_model):
        """Success returns nodes with x, y, z and reaches."""
        client, model = client_with_model
        resp = client.get("/api/streams")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_nodes"] == 2
        assert data["n_reaches"] == 1
        assert len(data["nodes"]) == 2
        assert len(data["reaches"]) == 1

    def test_streams_node_coordinates(self, client_with_model):
        """Stream nodes include x, y, z from GW node coordinates."""
        client, model = client_with_model
        resp = client.get("/api/streams")
        data = resp.json()
        node1 = data["nodes"][0]
        assert "x" in node1
        assert "y" in node1
        assert "z" in node1
        # z should come from gs_elev
        assert node1["z"] == 100.0

    def test_streams_reach_structure(self, client_with_model):
        """Reaches should contain lists of stream node IDs."""
        client, model = client_with_model
        resp = client.get("/api/streams")
        data = resp.json()
        reach = data["reaches"][0]
        assert isinstance(reach, list)
        assert 1 in reach
        assert 2 in reach

    def test_streams_no_streams_in_model(self):
        """Model loaded but streams not present."""
        _reset_model_state()
        model = _make_mock_model(with_streams=False)
        model_state._model = model
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/streams")
            assert resp.status_code == 404
            assert "No stream data" in resp.json()["detail"]
        finally:
            _reset_model_state()

    def test_streams_node_reach_id(self, client_with_model):
        """Each stream node should include its reach_id."""
        client, model = client_with_model
        resp = client.get("/api/streams")
        data = resp.json()
        for node in data["nodes"]:
            assert "reach_id" in node
            assert node["reach_id"] == 1

    def test_streams_no_stratigraphy_z_zero(self):
        """Without stratigraphy, z should default to 0.0."""
        _reset_model_state()
        model = _make_mock_model(with_streams=True, with_stratigraphy=False)
        model_state._model = model
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/streams")
            assert resp.status_code == 200
            data = resp.json()
            for node in data["nodes"]:
                assert node["z"] == 0.0
        finally:
            _reset_model_state()

    def test_streams_multiple_reaches(self):
        """Multiple reaches are correctly represented."""
        _reset_model_state()
        model = _make_mock_model(with_streams=False, with_stratigraphy=True)

        # Build custom stream structure with 2 reaches
        streams = MagicMock()
        streams.n_nodes = 4

        sn1, sn2, sn3, sn4 = MagicMock(), MagicMock(), MagicMock(), MagicMock()
        sn1.id, sn1.groundwater_node = 1, 1
        sn2.id, sn2.groundwater_node = 2, 2
        sn3.id, sn3.groundwater_node = 3, 3
        sn4.id, sn4.groundwater_node = 4, 4

        reach1 = MagicMock()
        reach1.id = 1
        reach1.stream_nodes = [sn1, sn2]

        reach2 = MagicMock()
        reach2.id = 2
        reach2.stream_nodes = [sn3, sn4]

        streams.reaches = [reach1, reach2]
        model.streams = streams
        model.has_streams = True
        model.n_stream_nodes = 4

        model_state._model = model
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/streams")
            assert resp.status_code == 200
            data = resp.json()
            assert data["n_reaches"] == 2
            assert data["n_nodes"] == 4
        finally:
            _reset_model_state()

    def test_streams_preprocessor_binary_fallback(self):
        """When reaches are empty and connectivity not populated, use preprocessor binary."""
        _reset_model_state()
        model = _make_mock_model(with_streams=False, with_stratigraphy=True)

        # Build stream with nodes but empty reaches and no connectivity
        streams = MagicMock()
        streams.n_nodes = 6

        # Create 6 stream nodes in 2 reaches (1-3 and 4-6)
        # No upstream/downstream attributes set, reach_id all 0
        sn_dict = {}
        for i in range(1, 7):
            sn = MagicMock()
            sn.id = i
            sn.gw_node = (
                i if i <= 4 else None
            )  # nodes 5,6 have no valid GW node (only 4 grid nodes)
            sn.reach_id = 0
            sn.downstream_node = None
            sn.upstream_node = None
            sn_dict[i] = sn

        # Nodes 1-4 are valid (have grid nodes), 5-6 don't
        streams.nodes = sn_dict
        streams.reaches = {}  # Empty — forces fallback
        model.streams = streams
        model.has_streams = True
        model.n_stream_nodes = 6

        model_state._model = model

        # Mock get_stream_reach_boundaries to return 2 reaches
        boundaries = [
            (1, 1, 2),  # Reach 1: stream nodes 1-2
            (2, 3, 4),  # Reach 2: stream nodes 3-4
        ]
        with patch.object(model_state, "get_stream_reach_boundaries", return_value=boundaries):
            app = create_app()
            client = TestClient(app)
            resp = client.get("/api/streams")
            assert resp.status_code == 200
            data = resp.json()
            # Should have 2 separate reaches, not 1 giant line
            assert data["n_reaches"] == 2
            assert data["n_nodes"] == 4  # nodes 1-4 valid
            assert len(data["reaches"]) == 2
            # Reach 1 should contain nodes [1, 2], reach 2 should contain [3, 4]
            assert data["reaches"][0] == [1, 2]
            assert data["reaches"][1] == [3, 4]
            # Nodes should have correct reach_ids assigned
            for node in data["nodes"]:
                if node["id"] in [1, 2]:
                    assert node["reach_id"] == 1
                elif node["id"] in [3, 4]:
                    assert node["reach_id"] == 2
        _reset_model_state()

    def test_streams_preprocessor_geojson_fallback(self):
        """GeoJSON endpoint also uses preprocessor binary fallback."""
        _reset_model_state()
        model = _make_mock_model(with_streams=False, with_stratigraphy=True)

        streams = MagicMock()
        streams.n_nodes = 4
        sn_dict = {}
        for i in range(1, 5):
            sn = MagicMock()
            sn.id = i
            sn.gw_node = i
            sn.reach_id = 0
            sn.downstream_node = None
            sn.upstream_node = None
            sn_dict[i] = sn

        streams.nodes = sn_dict
        streams.reaches = {}
        model.streams = streams
        model.has_streams = True
        model.n_stream_nodes = 4

        model_state._model = model

        boundaries = [
            (1, 1, 2),
            (2, 3, 4),
        ]
        with patch.object(model_state, "get_stream_reach_boundaries", return_value=boundaries):
            app = create_app()
            client = TestClient(app)
            resp = client.get("/api/streams/geojson")
            assert resp.status_code == 200
            data = resp.json()
            assert data["type"] == "FeatureCollection"
            assert len(data["features"]) == 2
            for feat in data["features"]:
                assert feat["geometry"]["type"] == "LineString"
                assert len(feat["geometry"]["coordinates"]) == 2
        _reset_model_state()


class TestStreamReachesWithIntNodes:
    """Tests for stream endpoints when StrmReach.nodes contains int IDs."""

    def _make_model_with_int_node_reaches(self):
        """Build a model whose stream.reaches use int node IDs (real StrmReach)."""
        from pyiwfm.components.stream import AppStream, StrmNode, StrmReach

        model = _make_mock_model(with_streams=False, with_stratigraphy=True)

        stream = AppStream()
        # 4 stream nodes, 2 reaches of 2 nodes each
        stream.add_node(StrmNode(id=1, x=0.0, y=0.0, reach_id=1, gw_node=1))
        stream.add_node(StrmNode(id=2, x=0.0, y=0.0, reach_id=1, gw_node=2))
        stream.add_node(StrmNode(id=3, x=0.0, y=0.0, reach_id=2, gw_node=3))
        stream.add_node(StrmNode(id=4, x=0.0, y=0.0, reach_id=2, gw_node=4))

        stream.add_reach(
            StrmReach(
                id=1,
                upstream_node=1,
                downstream_node=2,
                nodes=[1, 2],
            )
        )
        stream.add_reach(
            StrmReach(
                id=2,
                upstream_node=3,
                downstream_node=4,
                nodes=[3, 4],
            )
        )

        model.streams = stream
        model.has_streams = True
        model.n_stream_nodes = 4
        return model

    def test_streams_from_populated_reaches_with_int_nodes(self):
        """Populated reaches with int node IDs should produce separate polylines."""
        _reset_model_state()
        model = self._make_model_with_int_node_reaches()
        model_state._model = model
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/streams")
            assert resp.status_code == 200
            data = resp.json()
            assert data["n_reaches"] == 2
            assert data["n_nodes"] == 4
            assert data["reaches"][0] == [1, 2]
            assert data["reaches"][1] == [3, 4]
        finally:
            _reset_model_state()

    def test_streams_geojson_from_populated_reaches(self):
        """GeoJSON should have separate LineString features per reach."""
        _reset_model_state()
        model = self._make_model_with_int_node_reaches()
        model_state._model = model
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/streams/geojson")
            assert resp.status_code == 200
            data = resp.json()
            assert data["type"] == "FeatureCollection"
            assert len(data["features"]) == 2
            for feat in data["features"]:
                assert feat["geometry"]["type"] == "LineString"
                assert len(feat["geometry"]["coordinates"]) == 2
            # Verify reach_id properties
            reach_ids = {f["properties"]["reach_id"] for f in data["features"]}
            assert reach_ids == {1, 2}
        finally:
            _reset_model_state()

    def test_reach_profile_with_int_node_ids(self):
        """Reach profile works when StrmReach.nodes are int IDs."""
        _reset_model_state()
        model = self._make_model_with_int_node_reaches()
        model_state._model = model
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/streams/reach-profile?reach_id=1")
            assert resp.status_code == 200
            data = resp.json()
            assert data["reach_id"] == 1
            assert data["n_nodes"] == 2
            assert len(data["nodes"]) == 2
            assert data["nodes"][0]["stream_node_id"] == 1
            assert data["nodes"][1]["stream_node_id"] == 2
        finally:
            _reset_model_state()


# ===========================================================================
# 4F. routes/slices.py tests (~5 tests)
# ===========================================================================


class TestSliceJson:
    """Tests for GET /api/slice/json."""

    def test_slice_json_no_model(self, client_no_model):
        resp = client_no_model.get("/api/slice/json")
        assert resp.status_code == 404

    def test_slice_json_success(self, client_with_model):
        """Success returns SurfaceMeshData format."""
        client, model = client_with_model
        mock_data = {
            "n_points": 6,
            "n_cells": 4,
            "n_layers": 2,
            "points": [0.0] * 18,
            "polys": [3, 0, 1, 2] * 4,
            "layer": [1, 1, 2, 2],
        }
        with patch.object(model_state, "get_slice_json", return_value=mock_data):
            resp = client.get("/api/slice/json?angle=0&position=0.5")
            assert resp.status_code == 200
            data = resp.json()
            assert data["n_points"] == 6
            assert data["n_cells"] == 4
            assert "points" in data
            assert "polys" in data
            assert "layer" in data

    def test_slice_json_empty(self, client_with_model):
        """404 when slice produces no cells."""
        client, model = client_with_model
        mock_data = {
            "n_points": 0,
            "n_cells": 0,
            "n_layers": 0,
            "points": [],
            "polys": [],
            "layer": [],
        }
        with patch.object(model_state, "get_slice_json", return_value=mock_data):
            resp = client.get("/api/slice/json?angle=90&position=0.0")
            assert resp.status_code == 404
            assert "Empty slice" in resp.json()["detail"]

    def test_slice_json_angle_range(self, client_with_model):
        """Angle parameter is validated between 0 and 180."""
        client, model = client_with_model
        resp = client.get("/api/slice/json?angle=200&position=0.5")
        assert resp.status_code == 422  # Validation error

    def test_slice_json_position_range(self, client_with_model):
        """Position parameter is validated between 0 and 1."""
        client, model = client_with_model
        resp = client.get("/api/slice/json?angle=0&position=2.0")
        assert resp.status_code == 422  # Validation error


# ===========================================================================
# 4G. routes/properties.py tests (~5 tests)
# ===========================================================================


class TestPropertiesList:
    """Tests for GET /api/properties."""

    def test_properties_list_no_model(self, client_no_model):
        resp = client_no_model.get("/api/properties")
        assert resp.status_code == 404

    def test_properties_list_success(self, client_with_model):
        """Returns property list with metadata."""
        client, model = client_with_model
        resp = client.get("/api/properties")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        # "layer" is always available
        prop_ids = [p["id"] for p in data]
        assert "layer" in prop_ids

    def test_properties_list_includes_stratigraphy_props(self, client_with_model):
        """With stratigraphy, thickness/top_elev/bottom_elev are available."""
        client, model = client_with_model
        resp = client.get("/api/properties")
        data = resp.json()
        prop_ids = [p["id"] for p in data]
        assert "thickness" in prop_ids
        assert "top_elev" in prop_ids
        assert "bottom_elev" in prop_ids

    def test_properties_list_item_schema(self, client_with_model):
        """Each property item has required fields."""
        client, model = client_with_model
        resp = client.get("/api/properties")
        data = resp.json()
        for prop in data:
            assert "id" in prop
            assert "name" in prop
            assert "units" in prop
            assert "description" in prop
            assert "cmap" in prop
            assert "log_scale" in prop


class TestPropertyValues:
    """Tests for GET /api/properties/{property_id}."""

    def test_property_layer_success(self, client_with_model):
        """Get layer property values."""
        client, model = client_with_model
        resp = client.get("/api/properties/layer")
        assert resp.status_code == 200
        data = resp.json()
        assert data["property_id"] == "layer"
        assert "values" in data
        assert "min" in data
        assert "max" in data
        assert "mean" in data
        # n_cells = n_elements * n_layers = 1 * 2 = 2
        assert len(data["values"]) == 2

    def test_property_layer_with_filter(self, client_with_model):
        """Layer property with layer filter.

        When layer filter is used, cells outside the selected layer get NaN
        values. The route serializes all values including NaN, which causes
        a JSON serialization error (NaN is not valid JSON). This results
        in a 500 server error -- a known limitation of the current API.
        """
        client, model = client_with_model
        # NaN values from layer masking cause JSON serialization failure
        with pytest.raises(ValueError, match="Out of range float values"):
            client.get("/api/properties/layer?layer=1")

    def test_property_unknown(self, client_with_model):
        """404 for unknown property."""
        client, model = client_with_model
        resp = client.get("/api/properties/nonexistent")
        assert resp.status_code == 404
        assert "not available" in resp.json()["detail"]

    def test_property_thickness(self, client_with_model):
        """Get thickness property values."""
        client, model = client_with_model
        resp = client.get("/api/properties/thickness")
        assert resp.status_code == 200
        data = resp.json()
        assert data["property_id"] == "thickness"
        assert data["name"] == "Layer Thickness"
        assert len(data["values"]) == 2  # n_elements * n_layers

    def test_property_top_elev(self, client_with_model):
        """Get top elevation property values."""
        client, model = client_with_model
        resp = client.get("/api/properties/top_elev")
        assert resp.status_code == 200
        data = resp.json()
        assert data["property_id"] == "top_elev"
        assert data["name"] == "Top Elevation"

    def test_property_bottom_elev(self, client_with_model):
        """Get bottom elevation property values."""
        client, model = client_with_model
        resp = client.get("/api/properties/bottom_elev")
        assert resp.status_code == 200
        data = resp.json()
        assert data["property_id"] == "bottom_elev"

    def test_property_no_model(self, client_no_model):
        """404 when no model is loaded."""
        resp = client_no_model.get("/api/properties/layer")
        assert resp.status_code == 404


# ===========================================================================
# 4H. routes/model.py tests (~10 tests)
# ===========================================================================


class TestModelInfo:
    """Tests for GET /api/model/info."""

    def test_model_info_no_model(self, client_no_model):
        resp = client_no_model.get("/api/model/info")
        assert resp.status_code == 404

    def test_model_info_success(self, client_with_model):
        client, model = client_with_model
        resp = client.get("/api/model/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "TestModel"
        assert data["n_nodes"] == 4
        assert data["n_elements"] == 1
        assert data["n_layers"] == 2
        assert data["has_streams"] is True
        assert data["has_lakes"] is False

    def test_model_info_n_stream_nodes(self, client_with_model):
        """Stream node count is included when streams are present."""
        client, model = client_with_model
        resp = client.get("/api/model/info")
        data = resp.json()
        assert data["n_stream_nodes"] == 2

    def test_model_info_no_streams(self):
        """Model without streams has n_stream_nodes=None."""
        _reset_model_state()
        model = _make_mock_model(with_streams=False)
        model_state._model = model
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/model/info")
            data = resp.json()
            assert data["has_streams"] is False
            assert data["n_stream_nodes"] is None
        finally:
            _reset_model_state()

    def test_model_info_no_lakes(self, client_with_model):
        """n_lakes is None when no lakes."""
        client, model = client_with_model
        resp = client.get("/api/model/info")
        data = resp.json()
        assert data["n_lakes"] is None


class TestModelBounds:
    """Tests for GET /api/model/bounds."""

    def test_model_bounds_no_model(self, client_no_model):
        resp = client_no_model.get("/api/model/bounds")
        assert resp.status_code == 404

    def test_model_bounds_success(self, client_with_model):
        client, model = client_with_model
        resp = client.get("/api/model/bounds")
        assert resp.status_code == 200
        data = resp.json()
        assert "xmin" in data
        assert "xmax" in data
        assert "ymin" in data
        assert "ymax" in data
        assert "zmin" in data
        assert "zmax" in data

    def test_model_bounds_values(self, client_with_model):
        """Bounds match the grid node coordinates."""
        client, model = client_with_model
        resp = client.get("/api/model/bounds")
        data = resp.json()
        # Grid has nodes at (0,0), (100,0), (100,100), (0,100)
        assert data["xmin"] == 0.0
        assert data["xmax"] == 100.0
        assert data["ymin"] == 0.0
        assert data["ymax"] == 100.0

    def test_model_bounds_z_range(self, client_with_model):
        """Z bounds come from stratigraphy elevations."""
        client, model = client_with_model
        resp = client.get("/api/model/bounds")
        data = resp.json()
        # bottom_elev min is 0.0, top_elev max is 100.0
        assert data["zmin"] == 0.0
        assert data["zmax"] == 100.0

    def test_model_bounds_no_stratigraphy(self):
        """Without stratigraphy, z bounds default to 0.0."""
        _reset_model_state()
        model = _make_mock_model(with_stratigraphy=False)
        model_state._model = model
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/model/bounds")
            data = resp.json()
            assert data["zmin"] == 0.0
            assert data["zmax"] == 0.0
        finally:
            _reset_model_state()


# ===========================================================================
# Additional integration-style tests
# ===========================================================================


class TestObservationWorkflow:
    """End-to-end observation upload/query/update/delete workflow."""

    def test_full_observation_workflow(self, client_with_model):
        """Upload -> list -> get data -> set location -> delete."""
        client, model = client_with_model

        # Step 1: Upload
        csv_content = "datetime,value\n2020-01-01,10.5\n2020-06-01,11.0\n2020-12-01,10.8\n"
        files = {"file": ("wells.csv", io.BytesIO(csv_content.encode()), "text/csv")}
        resp = client.post("/api/observations/upload", files=files)
        assert resp.status_code == 200
        obs_id = resp.json()["observation_id"]

        # Step 2: List
        resp = client.get("/api/observations")
        assert len(resp.json()) == 1
        assert resp.json()[0]["id"] == obs_id

        # Step 3: Get data
        resp = client.get(f"/api/observations/{obs_id}/data")
        assert resp.status_code == 200
        assert len(resp.json()["values"]) == 3

        # Step 4: Set location
        resp = client.put(
            f"/api/observations/{obs_id}/location",
            params={"location_id": 7, "location_type": "gw"},
        )
        assert resp.status_code == 200

        # Step 5: Delete
        resp = client.delete(f"/api/observations/{obs_id}")
        assert resp.status_code == 200

        # Step 6: Verify deleted
        resp = client.get("/api/observations")
        assert resp.json() == []

    def test_upload_multiple_observations(self, client_with_model):
        """Can upload multiple observation files."""
        client, model = client_with_model

        for i in range(3):
            csv_content = f"datetime,value\n2020-01-0{i + 1},{i * 10.0}\n"
            files = {"file": (f"obs_{i}.csv", io.BytesIO(csv_content.encode()), "text/csv")}
            resp = client.post("/api/observations/upload", files=files)
            assert resp.status_code == 200

        resp = client.get("/api/observations")
        assert len(resp.json()) == 3


class TestBudgetWorkflow:
    """Budget route integration tests."""

    def test_budget_workflow_locations_to_data(self, client_with_model):
        """Get locations then get data for a location."""
        client, model = client_with_model
        reader = _make_mock_budget_reader(
            locations=["Region 1", "Region 2"],
            columns=["Inflow", "Outflow"],
            n_timesteps=4,
        )
        model_state._budget_readers["gw"] = reader

        # Get locations
        resp = client.get("/api/budgets/gw/locations")
        assert resp.status_code == 200
        locations = resp.json()["locations"]
        assert len(locations) == 2

        # Get columns
        resp = client.get("/api/budgets/gw/columns")
        assert resp.status_code == 200
        columns = resp.json()["columns"]
        assert len(columns) == 2

        # Get data
        resp = client.get("/api/budgets/gw/data")
        assert resp.status_code == 200

        # Get summary
        resp = client.get("/api/budgets/gw/summary")
        assert resp.status_code == 200
        summary = resp.json()
        assert summary["n_timesteps"] == 4


class TestHeadMapWorkflow:
    """Head map route integration tests."""

    def test_head_map_with_datetime(self, client_with_model):
        """Head map response includes datetime in metadata."""
        client, model = client_with_model
        loader = _make_mock_head_loader(n_frames=3, n_nodes=4, n_layers=2)
        model_state._head_loader = loader

        resp = client.get("/api/mesh/head-map?timestep=1&layer=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["metadata"]["datetime"] is not None
        assert "2020" in data["metadata"]["datetime"]


class TestGwHydrographNames:
    """Verify GW hydrograph returns name and layer from model data."""

    def test_gw_hydrograph_name_from_model(self, client_with_gw):
        """GW hydrograph uses name from groundwater.hydrograph_locations."""
        client, model = client_with_gw
        reader = _make_mock_gw_hydrograph_reader(n_columns=2, n_timesteps=5)
        model_state._gw_hydrograph_reader = reader

        resp = client.get("/api/results/hydrograph?type=gw&location_id=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Well-1"
        assert data["layer"] == 1

    def test_gw_hydrograph_second_location(self, client_with_gw):
        """Second GW hydrograph location uses correct name and layer."""
        client, model = client_with_gw
        reader = _make_mock_gw_hydrograph_reader(n_columns=2, n_timesteps=5)
        model_state._gw_hydrograph_reader = reader

        resp = client.get("/api/results/hydrograph?type=gw&location_id=2")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Well-2"
        assert data["layer"] == 2


# ===========================================================================
# 5A. routes/slices.py extended tests (VTU endpoints + cross-section + info)
# ===========================================================================


def _make_mock_slice_mesh(n_cells=4, n_points=6):
    """Create a mock PyVista PolyData for slice results."""
    mock_mesh = MagicMock()
    mock_mesh.n_cells = n_cells
    mock_mesh.n_points = n_points
    mock_mesh.bounds = (0.0, 100.0, 0.0, 100.0, 0.0, 50.0)
    mock_mesh.area = 5000.0
    mock_mesh.cell_data = {"layer": np.array([1, 1, 2, 2][:n_cells])}
    mock_mesh.point_data = {}
    return mock_mesh


def _make_mock_slicer(slice_mesh=None):
    """Create a mock SlicingController."""
    if slice_mesh is None:
        slice_mesh = _make_mock_slice_mesh()

    slicer = MagicMock()
    slicer.normalized_to_position.return_value = 50.0
    slicer.slice_x.return_value = slice_mesh
    slicer.slice_y.return_value = slice_mesh
    slicer.slice_z.return_value = slice_mesh
    slicer.create_cross_section.return_value = slice_mesh
    slicer.get_slice_properties.return_value = {
        "n_cells": slice_mesh.n_cells,
        "n_points": slice_mesh.n_points,
        "area": 5000.0,
        "bounds": list(slice_mesh.bounds) if slice_mesh.n_cells > 0 else None,
        "cell_arrays": list(slice_mesh.cell_data.keys()),
        "point_arrays": list(slice_mesh.point_data.keys()),
    }
    return slicer


class TestSliceVTU:
    """Tests for GET /api/slice (VTU response)."""

    def test_slice_no_model(self, client_no_model):
        """404 when no model is loaded."""
        resp = client_no_model.get("/api/slice?axis=x&position=0.5")
        assert resp.status_code == 404
        assert "No model loaded" in resp.json()["detail"]

    def test_slice_no_stratigraphy(self):
        """400 when model has no stratigraphy."""
        _reset_model_state()
        model = _make_mock_model(with_stratigraphy=False)
        model_state._model = model
        app = create_app()
        client = TestClient(app)
        try:
            with patch(
                "pyiwfm.visualization.webapi.routes.slices.pv",
                create=True,
            ):
                resp = client.get("/api/slice?axis=x&position=0.5")
            assert resp.status_code == 400
            assert "Stratigraphy required" in resp.json()["detail"]
        finally:
            _reset_model_state()

    def test_slice_x_axis_success(self, client_with_model):
        """GET /api/slice with axis=x returns VTU bytes."""
        client, model = client_with_model
        mock_mesh = _make_mock_slice_mesh(n_cells=4, n_points=6)
        mock_slicer = _make_mock_slicer(mock_mesh)
        fake_vtu = b"<VTKFile>mock vtu data</VTKFile>"

        with (
            patch("pyiwfm.visualization.vtk_export.VTKExporter") as MockExporter,
            patch(
                "pyiwfm.visualization.webapi.slicing.SlicingController",
                return_value=mock_slicer,
            ),
            patch(
                "pyiwfm.visualization.webapi.routes.slices._pyvista_to_vtu",
                return_value=fake_vtu,
            ),
        ):
            mock_exporter_inst = MockExporter.return_value
            mock_exporter_inst.to_pyvista_3d.return_value = MagicMock()

            resp = client.get("/api/slice?axis=x&position=0.5")

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/xml"
        assert b"mock vtu data" in resp.content
        mock_slicer.slice_x.assert_called_once()

    def test_slice_y_axis_success(self, client_with_model):
        """GET /api/slice with axis=y routes to slice_y."""
        client, model = client_with_model
        mock_mesh = _make_mock_slice_mesh(n_cells=4, n_points=6)
        mock_slicer = _make_mock_slicer(mock_mesh)
        fake_vtu = b"<VTKFile>vtu</VTKFile>"

        with (
            patch("pyiwfm.visualization.vtk_export.VTKExporter") as MockExporter,
            patch(
                "pyiwfm.visualization.webapi.slicing.SlicingController",
                return_value=mock_slicer,
            ),
            patch(
                "pyiwfm.visualization.webapi.routes.slices._pyvista_to_vtu",
                return_value=fake_vtu,
            ),
        ):
            MockExporter.return_value.to_pyvista_3d.return_value = MagicMock()
            resp = client.get("/api/slice?axis=y&position=0.3")

        assert resp.status_code == 200
        mock_slicer.slice_y.assert_called_once()

    def test_slice_z_axis_success(self, client_with_model):
        """GET /api/slice with axis=z routes to slice_z."""
        client, model = client_with_model
        mock_mesh = _make_mock_slice_mesh(n_cells=4, n_points=6)
        mock_slicer = _make_mock_slicer(mock_mesh)
        fake_vtu = b"<VTKFile>vtu</VTKFile>"

        with (
            patch("pyiwfm.visualization.vtk_export.VTKExporter") as MockExporter,
            patch(
                "pyiwfm.visualization.webapi.slicing.SlicingController",
                return_value=mock_slicer,
            ),
            patch(
                "pyiwfm.visualization.webapi.routes.slices._pyvista_to_vtu",
                return_value=fake_vtu,
            ),
        ):
            MockExporter.return_value.to_pyvista_3d.return_value = MagicMock()
            resp = client.get("/api/slice?axis=z&position=0.8")

        assert resp.status_code == 200
        mock_slicer.slice_z.assert_called_once()

    def test_slice_empty_result(self, client_with_model):
        """404 when slice produces an empty mesh (n_cells=0)."""
        client, model = client_with_model
        empty_mesh = _make_mock_slice_mesh(n_cells=0, n_points=0)
        mock_slicer = _make_mock_slicer(empty_mesh)

        with (
            patch("pyiwfm.visualization.vtk_export.VTKExporter") as MockExporter,
            patch(
                "pyiwfm.visualization.webapi.slicing.SlicingController",
                return_value=mock_slicer,
            ),
        ):
            MockExporter.return_value.to_pyvista_3d.return_value = MagicMock()
            resp = client.get("/api/slice?axis=x&position=0.0")

        assert resp.status_code == 404
        assert "Empty slice" in resp.json()["detail"]

    def test_slice_content_disposition_header(self, client_with_model):
        """Response includes Content-Disposition header for file download."""
        client, model = client_with_model
        mock_mesh = _make_mock_slice_mesh()
        mock_slicer = _make_mock_slicer(mock_mesh)
        fake_vtu = b"<VTKFile/>"

        with (
            patch("pyiwfm.visualization.vtk_export.VTKExporter") as MockExporter,
            patch(
                "pyiwfm.visualization.webapi.slicing.SlicingController",
                return_value=mock_slicer,
            ),
            patch(
                "pyiwfm.visualization.webapi.routes.slices._pyvista_to_vtu",
                return_value=fake_vtu,
            ),
        ):
            MockExporter.return_value.to_pyvista_3d.return_value = MagicMock()
            resp = client.get("/api/slice?axis=x&position=0.5")

        assert resp.status_code == 200
        assert "slice.vtu" in resp.headers.get("content-disposition", "")


class TestSliceJsonExtended:
    """Extended tests for GET /api/slice/json (ValueError handling)."""

    def test_slice_json_value_error(self, client_with_model):
        """400 when model_state.get_slice_json raises ValueError."""
        client, model = client_with_model
        with patch.object(
            model_state,
            "get_slice_json",
            side_effect=ValueError("Invalid slice parameters"),
        ):
            resp = client.get("/api/slice/json?angle=45&position=0.5")
            assert resp.status_code == 400
            assert "Invalid slice parameters" in resp.json()["detail"]


class TestSliceCrossSection:
    """Tests for GET /api/slice/cross-section."""

    def test_cross_section_no_model(self, client_no_model):
        """404 when no model is loaded."""
        resp = client_no_model.get(
            "/api/slice/cross-section?start_x=0&start_y=0&end_x=100&end_y=100"
        )
        assert resp.status_code == 404

    def test_cross_section_no_stratigraphy(self):
        """400 when model has no stratigraphy."""
        _reset_model_state()
        model = _make_mock_model(with_stratigraphy=False)
        model_state._model = model
        app = create_app()
        client = TestClient(app)
        try:
            with patch(
                "pyiwfm.visualization.webapi.routes.slices.pv",
                create=True,
            ):
                resp = client.get(
                    "/api/slice/cross-section?start_x=0&start_y=0&end_x=100&end_y=100"
                )
            assert resp.status_code == 400
            assert "Stratigraphy required" in resp.json()["detail"]
        finally:
            _reset_model_state()

    def test_cross_section_success(self, client_with_model):
        """Success returns VTU bytes with cross_section.vtu filename."""
        client, model = client_with_model
        mock_mesh = _make_mock_slice_mesh(n_cells=4, n_points=6)
        mock_slicer = _make_mock_slicer(mock_mesh)
        fake_vtu = b"<VTKFile>cross section</VTKFile>"

        with (
            patch("pyiwfm.visualization.vtk_export.VTKExporter") as MockExporter,
            patch(
                "pyiwfm.visualization.webapi.slicing.SlicingController",
                return_value=mock_slicer,
            ),
            patch(
                "pyiwfm.visualization.webapi.routes.slices._pyvista_to_vtu",
                return_value=fake_vtu,
            ),
        ):
            MockExporter.return_value.to_pyvista_3d.return_value = MagicMock()
            resp = client.get("/api/slice/cross-section?start_x=10&start_y=20&end_x=80&end_y=90")

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/xml"
        assert "cross_section.vtu" in resp.headers.get("content-disposition", "")
        mock_slicer.create_cross_section.assert_called_once_with(
            start=(10.0, 20.0), end=(80.0, 90.0)
        )

    def test_cross_section_empty_result(self, client_with_model):
        """404 when cross-section produces no cells."""
        client, model = client_with_model
        empty_mesh = _make_mock_slice_mesh(n_cells=0, n_points=0)
        mock_slicer = _make_mock_slicer(empty_mesh)

        with (
            patch("pyiwfm.visualization.vtk_export.VTKExporter") as MockExporter,
            patch(
                "pyiwfm.visualization.webapi.slicing.SlicingController",
                return_value=mock_slicer,
            ),
        ):
            MockExporter.return_value.to_pyvista_3d.return_value = MagicMock()
            resp = client.get("/api/slice/cross-section?start_x=0&start_y=0&end_x=100&end_y=100")

        assert resp.status_code == 404
        assert "Empty cross-section" in resp.json()["detail"]


class TestSliceInfo:
    """Tests for GET /api/slice/info."""

    def test_slice_info_no_model(self, client_no_model):
        """404 when no model is loaded."""
        resp = client_no_model.get("/api/slice/info?axis=x&position=0.5")
        assert resp.status_code == 404

    def test_slice_info_no_stratigraphy(self):
        """400 when model has no stratigraphy."""
        _reset_model_state()
        model = _make_mock_model(with_stratigraphy=False)
        model_state._model = model
        app = create_app()
        client = TestClient(app)
        try:
            with patch(
                "pyiwfm.visualization.webapi.routes.slices.pv",
                create=True,
            ):
                resp = client.get("/api/slice/info?axis=x&position=0.5")
            assert resp.status_code == 400
            assert "Stratigraphy required" in resp.json()["detail"]
        finally:
            _reset_model_state()

    def test_slice_info_x_axis(self, client_with_model):
        """Info for an X-axis slice returns n_cells, n_points, bounds."""
        client, model = client_with_model
        mock_mesh = _make_mock_slice_mesh(n_cells=4, n_points=6)
        mock_slicer = _make_mock_slicer(mock_mesh)

        with (
            patch("pyiwfm.visualization.vtk_export.VTKExporter") as MockExporter,
            patch(
                "pyiwfm.visualization.webapi.slicing.SlicingController",
                return_value=mock_slicer,
            ),
        ):
            MockExporter.return_value.to_pyvista_3d.return_value = MagicMock()
            resp = client.get("/api/slice/info?axis=x&position=0.5")

        assert resp.status_code == 200
        data = resp.json()
        assert data["n_cells"] == 4
        assert data["n_points"] == 6
        assert data["bounds"] is not None
        assert len(data["bounds"]) == 6
        mock_slicer.slice_x.assert_called_once()

    def test_slice_info_y_axis(self, client_with_model):
        """Info for a Y-axis slice."""
        client, model = client_with_model
        mock_mesh = _make_mock_slice_mesh(n_cells=3, n_points=5)
        mock_slicer = _make_mock_slicer(mock_mesh)
        mock_slicer.get_slice_properties.return_value = {
            "n_cells": 3,
            "n_points": 5,
            "bounds": [0.0, 100.0, 50.0, 50.0, 0.0, 50.0],
        }

        with (
            patch("pyiwfm.visualization.vtk_export.VTKExporter") as MockExporter,
            patch(
                "pyiwfm.visualization.webapi.slicing.SlicingController",
                return_value=mock_slicer,
            ),
        ):
            MockExporter.return_value.to_pyvista_3d.return_value = MagicMock()
            resp = client.get("/api/slice/info?axis=y&position=0.7")

        assert resp.status_code == 200
        data = resp.json()
        assert data["n_cells"] == 3
        assert data["n_points"] == 5
        mock_slicer.slice_y.assert_called_once()

    def test_slice_info_z_axis(self, client_with_model):
        """Info for a Z-axis slice."""
        client, model = client_with_model
        mock_mesh = _make_mock_slice_mesh(n_cells=2, n_points=4)
        mock_slicer = _make_mock_slicer(mock_mesh)
        mock_slicer.get_slice_properties.return_value = {
            "n_cells": 2,
            "n_points": 4,
            "bounds": [0.0, 100.0, 0.0, 100.0, 25.0, 25.0],
        }

        with (
            patch("pyiwfm.visualization.vtk_export.VTKExporter") as MockExporter,
            patch(
                "pyiwfm.visualization.webapi.slicing.SlicingController",
                return_value=mock_slicer,
            ),
        ):
            MockExporter.return_value.to_pyvista_3d.return_value = MagicMock()
            resp = client.get("/api/slice/info?axis=z&position=0.5")

        assert resp.status_code == 200
        data = resp.json()
        assert data["n_cells"] == 2
        mock_slicer.slice_z.assert_called_once()

    def test_slice_info_empty_bounds_null(self, client_with_model):
        """Info for an empty slice returns bounds=null."""
        client, model = client_with_model
        empty_mesh = _make_mock_slice_mesh(n_cells=0, n_points=0)
        mock_slicer = _make_mock_slicer(empty_mesh)
        mock_slicer.get_slice_properties.return_value = {
            "n_cells": 0,
            "n_points": 0,
            "bounds": None,
        }

        with (
            patch("pyiwfm.visualization.vtk_export.VTKExporter") as MockExporter,
            patch(
                "pyiwfm.visualization.webapi.slicing.SlicingController",
                return_value=mock_slicer,
            ),
        ):
            MockExporter.return_value.to_pyvista_3d.return_value = MagicMock()
            resp = client.get("/api/slice/info?axis=x&position=0.0")

        assert resp.status_code == 200
        data = resp.json()
        assert data["n_cells"] == 0
        assert data["bounds"] is None


# ===========================================================================
# 5B. routes/streams.py extended tests (VTP endpoint)
# ===========================================================================


class TestStreamVTP:
    """Tests for GET /api/streams/vtp (VTK PolyData format)."""

    def test_streams_vtp_no_model(self, client_no_model):
        """404 when no model is loaded."""
        resp = client_no_model.get("/api/streams/vtp")
        assert resp.status_code == 404

    def test_streams_vtp_no_streams(self):
        """404 when model has no stream data."""
        _reset_model_state()
        model = _make_mock_model(with_streams=False)
        model_state._model = model
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/streams/vtp")
            assert resp.status_code == 404
            assert "No stream data" in resp.json()["detail"]
        finally:
            _reset_model_state()

    def test_streams_vtp_success(self, client_with_model):
        """Success returns VTP XML with correct content-type and filename."""
        client, model = client_with_model

        # Mock the vtk module at the import point in the route
        mock_vtk = MagicMock()

        # Mock vtkPoints
        mock_points = MagicMock()
        mock_vtk.vtkPoints.return_value = mock_points

        # Mock vtkCellArray
        mock_lines = MagicMock()
        mock_vtk.vtkCellArray.return_value = mock_lines

        # Mock vtkIntArray
        mock_reach_ids = MagicMock()
        mock_vtk.vtkIntArray.return_value = mock_reach_ids

        # Mock vtkPolyLine
        mock_polyline = MagicMock()
        mock_point_ids = MagicMock()
        mock_polyline.GetPointIds.return_value = mock_point_ids
        mock_vtk.vtkPolyLine.return_value = mock_polyline

        # Mock vtkPolyData
        mock_polydata = MagicMock()
        mock_cell_data = MagicMock()
        mock_polydata.GetCellData.return_value = mock_cell_data
        mock_vtk.vtkPolyData.return_value = mock_polydata

        # Mock writer
        mock_writer = MagicMock()
        mock_writer.GetOutputString.return_value = "<VTKFile>streams vtp</VTKFile>"
        mock_vtk.vtkXMLPolyDataWriter.return_value = mock_writer

        with patch.dict("sys.modules", {"vtk": mock_vtk}):
            resp = client.get("/api/streams/vtp")

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/xml"
        assert "streams.vtp" in resp.headers.get("content-disposition", "")

    def test_streams_vtp_content(self, client_with_model):
        """VTP response body contains the VTK XML output."""
        client, model = client_with_model

        mock_vtk = MagicMock()
        mock_writer = MagicMock()
        mock_writer.GetOutputString.return_value = "<VTKFile>test_content</VTKFile>"
        mock_vtk.vtkXMLPolyDataWriter.return_value = mock_writer

        # All other VTK mock objects
        mock_vtk.vtkPoints.return_value = MagicMock()
        mock_vtk.vtkCellArray.return_value = MagicMock()
        mock_vtk.vtkIntArray.return_value = MagicMock()
        mock_polyline = MagicMock()
        mock_polyline.GetPointIds.return_value = MagicMock()
        mock_vtk.vtkPolyLine.return_value = mock_polyline
        mock_polydata = MagicMock()
        mock_polydata.GetCellData.return_value = MagicMock()
        mock_vtk.vtkPolyData.return_value = mock_polydata

        with patch.dict("sys.modules", {"vtk": mock_vtk}):
            resp = client.get("/api/streams/vtp")

        assert resp.status_code == 200
        assert b"test_content" in resp.content


# ===========================================================================
# 5C. routes/properties.py extended tests (GW params + layer filter)
# ===========================================================================


def _make_mock_model_with_gw_params(**kwargs):
    """Create a mock model with groundwater aquifer parameters."""
    model = _make_mock_model(with_stratigraphy=True, **kwargs)

    gw = MagicMock()
    params = MagicMock()

    # 2D arrays: (n_nodes, n_layers) = (4, 2)
    params.kh = np.array([[10.0, 5.0], [12.0, 6.0], [11.0, 5.5], [10.5, 5.2]])
    params.kv = np.array([[1.0, 0.5], [1.2, 0.6], [1.1, 0.55], [1.05, 0.52]])
    params.specific_storage = np.array(
        [[1e-4, 2e-4], [1.1e-4, 2.1e-4], [0.9e-4, 1.9e-4], [1e-4, 2e-4]]
    )
    params.specific_yield = np.array([[0.1, 0.15], [0.12, 0.14], [0.11, 0.16], [0.1, 0.15]])
    # Also set short aliases for the getattr fallback
    params.ss = params.specific_storage
    params.sy = params.specific_yield

    gw.aquifer_params = params
    gw.n_hydrograph_locations = 0
    gw.hydrograph_locations = []
    model.groundwater = gw

    return model


@pytest.fixture()
def client_with_gw_params():
    """TestClient with mock model that has GW aquifer parameters."""
    _reset_model_state()
    model = _make_mock_model_with_gw_params()
    model_state._model = model
    app = create_app()
    yield TestClient(app), model
    _reset_model_state()


class TestPropertiesGWParams:
    """Tests for GET /api/properties/{id} with GW aquifer parameters."""

    def test_properties_list_includes_gw_params(self, client_with_gw_params):
        """When GW params are present, kh/kv/ss/sy appear in the list."""
        client, model = client_with_gw_params
        resp = client.get("/api/properties")
        assert resp.status_code == 200
        data = resp.json()
        prop_ids = [p["id"] for p in data]
        assert "kh" in prop_ids
        assert "kv" in prop_ids
        assert "ss" in prop_ids
        assert "sy" in prop_ids

    def test_property_kh_values(self, client_with_gw_params):
        """GET /api/properties/kh returns valid array with stats."""
        client, model = client_with_gw_params
        resp = client.get("/api/properties/kh")
        assert resp.status_code == 200
        data = resp.json()
        assert data["property_id"] == "kh"
        assert data["name"] == "Horizontal Hydraulic Conductivity"
        assert data["units"] == "ft/d"
        # n_cells = n_elements(1) * n_layers(2) = 2
        assert len(data["values"]) == 2
        assert data["min"] > 0
        assert data["max"] > 0
        assert data["mean"] > 0

    def test_property_kv_values(self, client_with_gw_params):
        """GET /api/properties/kv returns vertical conductivity."""
        client, model = client_with_gw_params
        resp = client.get("/api/properties/kv")
        assert resp.status_code == 200
        data = resp.json()
        assert data["property_id"] == "kv"
        assert data["name"] == "Vertical Hydraulic Conductivity"
        assert len(data["values"]) == 2

    def test_property_ss_values(self, client_with_gw_params):
        """GET /api/properties/ss returns specific storage."""
        client, model = client_with_gw_params
        resp = client.get("/api/properties/ss")
        assert resp.status_code == 200
        data = resp.json()
        assert data["property_id"] == "ss"
        assert data["name"] == "Specific Storage"
        assert data["units"] == "1/ft"
        assert len(data["values"]) == 2
        # Values should be small (order of 1e-4)
        assert data["min"] < 0.01
        assert data["max"] < 0.01

    def test_property_sy_values(self, client_with_gw_params):
        """GET /api/properties/sy returns specific yield."""
        client, model = client_with_gw_params
        resp = client.get("/api/properties/sy")
        assert resp.status_code == 200
        data = resp.json()
        assert data["property_id"] == "sy"
        assert data["name"] == "Specific Yield"
        assert len(data["values"]) == 2
        # Sy is dimensionless, typically 0-0.3
        assert 0.0 < data["min"] < 1.0
        assert 0.0 < data["max"] < 1.0

    def test_property_unknown_returns_404(self, client_with_gw_params):
        """GET /api/properties/unknown returns 404."""
        client, model = client_with_gw_params
        resp = client.get("/api/properties/unknown")
        assert resp.status_code == 404
        assert "not available" in resp.json()["detail"]

    def test_property_kh_no_groundwater(self, client_with_model):
        """GET /api/properties/kh without GW component returns 404."""
        client, model = client_with_model
        # Default client_with_model has no groundwater
        resp = client.get("/api/properties/kh")
        assert resp.status_code == 404
        assert "not available" in resp.json()["detail"]


class TestPropertiesLayerFilter:
    """Tests for GET /api/properties/{id}?layer=N with layer filtering."""

    def test_property_layer_all_layers(self, client_with_model):
        """layer=0 (default) returns values for all layers without NaN."""
        client, model = client_with_model
        resp = client.get("/api/properties/layer")
        assert resp.status_code == 200
        data = resp.json()
        values = data["values"]
        # No NaN when layer=0
        assert all(v == v for v in values)  # NaN != NaN

    def test_property_thickness_values(self, client_with_model):
        """Thickness values reflect stratigraphy top_elev - bottom_elev."""
        client, model = client_with_model
        resp = client.get("/api/properties/thickness")
        assert resp.status_code == 200
        data = resp.json()
        # With the mock:
        #   layer 1: top_elev=100, bottom_elev=50 => thickness=50
        #   layer 2: top_elev=50, bottom_elev=0 => thickness=50
        assert data["min"] == pytest.approx(50.0)
        assert data["max"] == pytest.approx(50.0)

    def test_property_top_elev_values(self, client_with_model):
        """Top elevation values are computed correctly per layer."""
        client, model = client_with_model
        resp = client.get("/api/properties/top_elev")
        assert resp.status_code == 200
        data = resp.json()
        # Layer 1 top_elev=100, layer 2 top_elev=50
        assert 50.0 in data["values"]
        assert 100.0 in data["values"]

    def test_property_bottom_elev_values(self, client_with_model):
        """Bottom elevation values are computed correctly per layer."""
        client, model = client_with_model
        resp = client.get("/api/properties/bottom_elev")
        assert resp.status_code == 200
        data = resp.json()
        # Layer 1 bottom_elev=50, layer 2 bottom_elev=0
        assert 0.0 in data["values"]
        assert 50.0 in data["values"]

    def test_property_kh_with_1d_params(self):
        """GW params with 1D arrays (single-layer) handled correctly."""
        _reset_model_state()
        model = _make_mock_model(with_stratigraphy=True)

        gw = MagicMock()
        params = MagicMock()
        # 1D array (no layer dimension)
        params.kh = np.array([10.0, 12.0, 11.0, 10.5])
        params.kv = None
        params.specific_storage = None
        params.ss = None
        params.specific_yield = None
        params.sy = None
        gw.aquifer_params = params
        gw.n_hydrograph_locations = 0
        gw.hydrograph_locations = []
        model.groundwater = gw

        model_state._model = model
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/properties/kh")
            assert resp.status_code == 200
            data = resp.json()
            # 1D array: mean(kh) applied to all cells
            assert len(data["values"]) == 2  # n_elements * n_layers
            # All values should be the mean of [10, 12, 11, 10.5] = 10.875
            assert data["values"][0] == pytest.approx(10.875)
            assert data["values"][1] == pytest.approx(10.875)
        finally:
            _reset_model_state()

    def test_property_values_all_nan_causes_serialization_error(self):
        """When all property values are NaN, JSON serialization fails.

        NaN values in the `values` list cause a ValueError because NaN is
        not valid JSON. This is the same known limitation as layer filtering
        (see test_property_layer_with_filter).
        """
        _reset_model_state()
        model = _make_mock_model(with_stratigraphy=True)
        model_state._model = model
        app = create_app()
        client = TestClient(app)
        try:
            with patch(
                "pyiwfm.visualization.webapi.routes.properties._compute_property_values",
                return_value=np.array([np.nan, np.nan]),
            ):
                with pytest.raises(ValueError, match="Out of range float values"):
                    client.get("/api/properties/layer")
        finally:
            _reset_model_state()

    def test_property_no_model_in_compute(self):
        """_compute_property_values returns None when model is None."""
        _reset_model_state()
        # model_state._model is None
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/properties/layer")
            assert resp.status_code == 404
        finally:
            _reset_model_state()

    def test_property_gw_params_none(self):
        """GW params set to None returns 404 for kh property."""
        _reset_model_state()
        model = _make_mock_model(with_stratigraphy=True)
        gw = MagicMock()
        gw.aquifer_params = None
        gw.n_hydrograph_locations = 0
        gw.hydrograph_locations = []
        model.groundwater = gw
        model_state._model = model
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/properties/kh")
            assert resp.status_code == 404
            assert "not available" in resp.json()["detail"]
        finally:
            _reset_model_state()


# ===========================================================================
# Model Summary tests (GET /api/model/summary)
# ===========================================================================


class TestModelSummary:
    """Tests for GET /api/model/summary."""

    def test_summary_no_model(self, client_no_model):
        """Returns 404 when no model is loaded."""
        resp = client_no_model.get("/api/model/summary")
        assert resp.status_code == 404

    def test_summary_basic_model(self, client_with_model):
        """Basic model returns correct mesh and stream info."""
        client, _ = client_with_model
        resp = client.get("/api/model/summary")
        assert resp.status_code == 200
        data = resp.json()

        assert data["name"] == "TestModel"
        assert data["mesh"]["n_nodes"] == 4
        assert data["mesh"]["n_elements"] == 1
        assert data["mesh"]["n_layers"] == 2

        # Streams loaded in this fixture
        assert data["streams"]["loaded"] is True
        assert data["streams"]["n_nodes"] == 2

        # Groundwater not loaded in basic fixture
        assert data["groundwater"]["loaded"] is False

        # No results loaded
        assert data["available_results"]["has_head_data"] is False

    def test_summary_with_groundwater(self, client_with_gw):
        """GW details are populated when groundwater component is present."""
        client, _ = client_with_gw
        resp = client.get("/api/model/summary")
        assert resp.status_code == 200
        data = resp.json()

        assert data["groundwater"]["loaded"] is True
        assert data["groundwater"]["n_hydrograph_locations"] == 2

    def test_summary_available_results(self):
        """Results section is populated when lazy getters return data."""
        _reset_model_state()
        model = _make_mock_model(with_streams=True, with_stratigraphy=True)
        model_state._model = model

        head_loader = MagicMock()
        head_loader.n_frames = 10

        app = create_app()
        client = TestClient(app)
        try:
            with (
                patch.object(model_state, "get_head_loader", return_value=head_loader),
                patch.object(model_state, "get_gw_hydrograph_reader", return_value=MagicMock()),
                patch.object(
                    model_state,
                    "get_stream_hydrograph_reader",
                    return_value=MagicMock(),
                ),
                patch.object(
                    model_state,
                    "get_available_budgets",
                    return_value=["gw_budget"],
                ),
            ):
                resp = client.get("/api/model/summary")
                assert resp.status_code == 200
                data = resp.json()

                assert data["available_results"]["has_head_data"] is True
                assert data["available_results"]["n_head_timesteps"] == 10
                assert data["available_results"]["has_gw_hydrographs"] is True
                assert data["available_results"]["has_stream_hydrographs"] is True
                assert data["available_results"]["n_budget_types"] == 1
                assert "gw_budget" in data["available_results"]["budget_types"]
        finally:
            _reset_model_state()

    def test_summary_source_metadata(self):
        """Source field comes from model.metadata."""
        _reset_model_state()
        model = _make_mock_model(
            with_stratigraphy=True,
            metadata={"source": "/path/to/model.dat"},
        )
        model_state._model = model
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/model/summary")
            assert resp.status_code == 200
            data = resp.json()
            assert data["source"] == "/path/to/model.dat"
        finally:
            _reset_model_state()

    def test_summary_subregions_from_elements(self):
        """Subregion count derived from Element.subregion when grid.subregions empty."""
        _reset_model_state()
        # Build a grid with 3 elements having subregion IDs 1, 2, 1 → expect 2
        nodes = {
            1: Node(id=1, x=0.0, y=0.0),
            2: Node(id=2, x=100.0, y=0.0),
            3: Node(id=3, x=100.0, y=100.0),
            4: Node(id=4, x=0.0, y=100.0),
            5: Node(id=5, x=200.0, y=0.0),
            6: Node(id=6, x=200.0, y=100.0),
        }
        elements = {
            1: Element(id=1, vertices=(1, 2, 3, 4), subregion=1),
            2: Element(id=2, vertices=(2, 5, 6, 3), subregion=2),
            3: Element(id=3, vertices=(1, 2, 5, 4), subregion=1),
        }
        grid = AppGrid(nodes=nodes, elements=elements)
        # subregions dict is empty, so grid.n_subregions == 0
        assert grid.n_subregions == 0

        model = MagicMock()
        model.name = "SubregionTest"
        model.grid = grid
        model.metadata = {}
        model.n_nodes = 6
        model.n_elements = 3
        model.n_layers = 1
        model.has_streams = False
        model.streams = None
        model.groundwater = None
        model.lakes = None
        model.rootzone = None
        model.small_watersheds = None
        model.unsaturated_zone = None
        model.stratigraphy = None
        model_state._model = model

        app = create_app()
        client = TestClient(app)
        try:
            with (
                patch.object(model_state, "get_head_loader", return_value=None),
                patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
                patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
                patch.object(model_state, "get_available_budgets", return_value=[]),
            ):
                resp = client.get("/api/model/summary")
                assert resp.status_code == 200
                data = resp.json()
                assert data["mesh"]["n_subregions"] == 2
        finally:
            _reset_model_state()

    def test_summary_stream_reaches_from_nodes(self):
        """Reach count derived from StrmNode.reach_id when reaches dict empty."""
        _reset_model_state()
        model = _make_mock_model(with_stratigraphy=True)

        # Build a mock stream component with nodes but empty reaches
        stm = MagicMock()
        stm.n_nodes = 4
        stm.n_reaches = 0  # reaches dict empty
        stm.n_diversions = 0
        stm.n_bypasses = 0

        node1 = MagicMock()
        node1.reach_id = 1
        node2 = MagicMock()
        node2.reach_id = 1
        node3 = MagicMock()
        node3.reach_id = 2
        node4 = MagicMock()
        node4.reach_id = 3
        stm.nodes = {1: node1, 2: node2, 3: node3, 4: node4}

        model.streams = stm
        model.has_streams = True
        model_state._model = model

        app = create_app()
        client = TestClient(app)
        try:
            with (
                patch.object(model_state, "get_head_loader", return_value=None),
                patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
                patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
                patch.object(model_state, "get_available_budgets", return_value=[]),
            ):
                resp = client.get("/api/model/summary")
                assert resp.status_code == 200
                data = resp.json()
                assert data["streams"]["n_reaches"] == 3
        finally:
            _reset_model_state()

    def test_summary_bc_from_metadata(self):
        """BC count falls back to metadata when component returns 0."""
        _reset_model_state()
        model = _make_mock_model(with_stratigraphy=True)

        gw = MagicMock()
        gw.n_boundary_conditions = 0
        gw.n_tile_drains = 0
        gw.n_wells = 5
        gw.n_hydrograph_locations = 2
        gw.aquifer_params = None
        model.groundwater = gw
        model.metadata = {
            "gw_n_specified_flow_bc": 10,
            "gw_n_specified_head_bc": 3,
            "gw_n_general_head_bc": 7,
            "gw_n_tile_drains": 4,
        }
        model_state._model = model

        app = create_app()
        client = TestClient(app)
        try:
            with (
                patch.object(model_state, "get_head_loader", return_value=None),
                patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
                patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
                patch.object(model_state, "get_available_budgets", return_value=[]),
            ):
                resp = client.get("/api/model/summary")
                assert resp.status_code == 200
                data = resp.json()
                assert data["groundwater"]["n_boundary_conditions"] == 20
                assert data["groundwater"]["n_tile_drains"] == 4
        finally:
            _reset_model_state()

    def test_summary_rootzone_crops_from_subconfigs(self):
        """Crop count derived from sub-configs when crop_types dict empty."""
        _reset_model_state()
        model = _make_mock_model(with_stratigraphy=True)

        rz = MagicMock()
        rz.n_crop_types = 0  # crop_types dict empty

        # Sub-configs: 12 nonponded + 5 ponded + 1 urban + 2 native = 20
        nonponded = MagicMock()
        nonponded.n_crops = 12
        rz.nonponded_config = nonponded
        rz.ponded_config = MagicMock()
        rz.urban_config = MagicMock()
        rz.native_riparian_config = MagicMock()

        rz.element_landuse = {}  # empty
        rz.soil_params = {i: MagicMock() for i in range(100)}  # 100 elements

        model.rootzone = rz
        model.metadata = {}
        model_state._model = model

        app = create_app()
        client = TestClient(app)
        try:
            with (
                patch.object(model_state, "get_head_loader", return_value=None),
                patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
                patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
                patch.object(model_state, "get_available_budgets", return_value=[]),
            ):
                resp = client.get("/api/model/summary")
                assert resp.status_code == 200
                data = resp.json()
                assert data["rootzone"]["n_crop_types"] == 20
                # land use is None when element_landuse is empty (no fallback)
                assert data["rootzone"]["n_land_use_elements"] is None
                assert data["rootzone"]["n_soil_parameter_sets"] == 100
        finally:
            _reset_model_state()


# ---------------------------------------------------------------------------
# Stream strategy priority and logging tests
# ---------------------------------------------------------------------------


class TestStreamStrategyPriorityAndLogging:
    """Tests for stream strategy ordering and log output."""

    def test_preprocessor_binary_tried_before_connectivity(self):
        """Preprocessor binary (strategy 2) should be tried before connectivity (strategy 3)."""
        _reset_model_state()
        model = _make_mock_model(with_streams=False, with_stratigraphy=True)

        # Build stream with nodes that have BOTH connectivity AND preprocessor data.
        # Both would succeed, but preprocessor binary should win because it's tried first.
        streams = MagicMock()
        streams.n_nodes = 4

        sn_dict = {}
        for i in range(1, 5):
            sn = MagicMock()
            sn.id = i
            sn.gw_node = i
            sn.reach_id = 0
            # Set up connectivity: 1→2, 3→4 (two chains)
            if i == 1:
                sn.downstream_node = 2
            elif i == 3:
                sn.downstream_node = 4
            else:
                sn.downstream_node = None
            sn_dict[i] = sn

        streams.nodes = sn_dict
        streams.reaches = {}  # Empty — forces fallback to strategies 2+
        model.streams = streams
        model.has_streams = True

        model_state._model = model

        # Preprocessor binary returns 2 reaches: [1,2] and [3,4]
        boundaries = [(1, 1, 2), (2, 3, 4)]

        with patch.object(model_state, "get_stream_reach_boundaries", return_value=boundaries):
            app = create_app()
            client = TestClient(app)
            with patch("pyiwfm.visualization.webapi.routes.streams.logger") as mock_logger:
                resp = client.get("/api/streams")
                assert resp.status_code == 200
                data = resp.json()
                assert data["n_reaches"] == 2
                # Should log strategy 2 (preprocessor binary), NOT strategy 3
                info_calls = [str(c) for c in mock_logger.info.call_args_list]
                assert any("strategy 2" in c for c in info_calls), (
                    f"Expected 'strategy 2' in log calls, got: {info_calls}"
                )
                assert not any("strategy 3" in c for c in info_calls), (
                    "Strategy 3 should not have been reached"
                )
        _reset_model_state()

    def test_single_reach_fallback_emits_warning(self):
        """Strategy 5 (single-reach fallback) should emit a WARNING log."""
        _reset_model_state()
        model = _make_mock_model(with_streams=False, with_stratigraphy=True)

        streams = MagicMock()
        streams.n_nodes = 3

        sn_dict = {}
        for i in range(1, 4):
            sn = MagicMock()
            sn.id = i
            sn.gw_node = i
            sn.reach_id = 0
            sn.downstream_node = None
            sn_dict[i] = sn

        streams.nodes = sn_dict
        streams.reaches = {}  # Empty
        model.streams = streams
        model.has_streams = True

        model_state._model = model

        # No preprocessor binary boundaries either
        with patch.object(model_state, "get_stream_reach_boundaries", return_value=[]):
            app = create_app()
            client = TestClient(app)
            with patch("pyiwfm.visualization.webapi.routes.streams.logger") as mock_logger:
                resp = client.get("/api/streams")
                assert resp.status_code == 200
                data = resp.json()
                # Single-reach fallback
                assert data["n_reaches"] == 1
                # Should have a WARNING log
                assert mock_logger.warning.called, "Expected WARNING log for single-reach fallback"
                warn_msg = str(mock_logger.warning.call_args)
                assert "strategy 5" in warn_msg or "single-reach" in warn_msg
        _reset_model_state()

    def test_populated_reaches_logs_strategy_1(self):
        """Strategy 1 (populated reaches) should emit INFO log."""
        _reset_model_state()
        model = _make_mock_model(with_streams=True, with_stratigraphy=True)

        model_state._model = model
        app = create_app()
        client = TestClient(app)
        try:
            with patch("pyiwfm.visualization.webapi.routes.streams.logger") as mock_logger:
                resp = client.get("/api/streams")
                assert resp.status_code == 200
                assert mock_logger.info.called
                info_msg = str(mock_logger.info.call_args)
                assert "strategy 1" in info_msg
        finally:
            _reset_model_state()


# ---------------------------------------------------------------------------
# Small Watershed Tests
# ---------------------------------------------------------------------------


class TestSmallWatershedsRoute:
    """Tests for /api/small-watersheds endpoint."""

    def _make_watershed_model(self):
        """Create a model with small watersheds for testing."""
        from pyiwfm.components.small_watershed import (
            AppSmallWatershed,
            WatershedGWNode,
            WatershedUnit,
        )

        _reset_model_state()
        model = _make_mock_model(with_stratigraphy=True)

        # Build real watershed objects
        gw_node_bf = WatershedGWNode(
            gw_node_id=1,
            max_perc_rate=0.0,
            is_baseflow=True,
            layer=2,
        )
        gw_node_perc = WatershedGWNode(
            gw_node_id=2,
            max_perc_rate=0.5,
            is_baseflow=False,
            layer=0,
        )

        ws1 = WatershedUnit(
            id=1,
            area=5000.0,
            dest_stream_node=10,
            gw_nodes=[gw_node_bf, gw_node_perc],
            curve_number=75.0,
            wilting_point=0.12,
            field_capacity=0.30,
            total_porosity=0.45,
            lambda_param=0.5,
            root_depth=3.0,
            hydraulic_cond=0.01,
            kunsat_method=1,
            gw_threshold=1.0,
            max_gw_storage=100.0,
            surface_flow_coeff=0.05,
            baseflow_coeff=0.02,
        )

        sw = AppSmallWatershed()
        sw.watersheds[1] = ws1
        model.small_watersheds = sw
        model.streams = None  # No stream lookup for dest coords

        model_state._model = model
        app = create_app()
        client = TestClient(app)
        return model, client

    def test_empty_watersheds(self):
        """No small watersheds returns empty list."""
        _reset_model_state()
        model = _make_mock_model(with_stratigraphy=True)
        model.small_watersheds = None
        model_state._model = model
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/small-watersheds")
            assert resp.status_code == 200
            data = resp.json()
            assert data["n_watersheds"] == 0
            assert data["watersheds"] == []
        finally:
            _reset_model_state()

    def test_marker_position_is_first_gw_node(self):
        """marker_position should be the first GW node's coordinates."""
        _, client = self._make_watershed_model()
        try:
            resp = client.get("/api/small-watersheds")
            assert resp.status_code == 200
            data = resp.json()
            assert data["n_watersheds"] == 1

            ws = data["watersheds"][0]
            assert "marker_position" in ws
            assert "centroid" not in ws

            # marker_position should match first GW node (node_id=1)
            first_gw = ws["gw_nodes"][0]
            assert ws["marker_position"][0] == first_gw["lng"]
            assert ws["marker_position"][1] == first_gw["lat"]
        finally:
            _reset_model_state()

    def test_gw_nodes_enriched_fields(self):
        """GW nodes include max_perc_rate and raw_qmaxwb."""
        _, client = self._make_watershed_model()
        try:
            resp = client.get("/api/small-watersheds")
            data = resp.json()
            ws = data["watersheds"][0]
            gw_nodes = ws["gw_nodes"]

            # First node is baseflow: raw_qmaxwb = -layer = -2
            bf_node = gw_nodes[0]
            assert bf_node["node_id"] == 1
            assert bf_node["is_baseflow"] is True
            assert bf_node["layer"] == 2
            assert bf_node["max_perc_rate"] == 0.0
            assert bf_node["raw_qmaxwb"] == -2.0

            # Second node is percolation: raw_qmaxwb = max_perc_rate = 0.5
            perc_node = gw_nodes[1]
            assert perc_node["node_id"] == 2
            assert perc_node["is_baseflow"] is False
            assert perc_node["max_perc_rate"] == 0.5
            assert perc_node["raw_qmaxwb"] == 0.5
        finally:
            _reset_model_state()

    def test_root_zone_params_present(self):
        """Watershed response includes root zone parameters."""
        _, client = self._make_watershed_model()
        try:
            resp = client.get("/api/small-watersheds")
            ws = resp.json()["watersheds"][0]

            assert ws["wilting_point"] == 0.12
            assert ws["field_capacity"] == 0.30
            assert ws["total_porosity"] == 0.45
            assert ws["lambda_param"] == 0.5
            assert ws["root_depth"] == 3.0
            assert ws["hydraulic_cond"] == 0.01
            assert ws["kunsat_method"] == 1
            assert ws["curve_number"] == 75.0
        finally:
            _reset_model_state()

    def test_aquifer_params_present(self):
        """Watershed response includes aquifer parameters."""
        _, client = self._make_watershed_model()
        try:
            resp = client.get("/api/small-watersheds")
            ws = resp.json()["watersheds"][0]

            assert ws["gw_threshold"] == 1.0
            assert ws["max_gw_storage"] == 100.0
            assert ws["surface_flow_coeff"] == 0.05
            assert ws["baseflow_coeff"] == 0.02
        finally:
            _reset_model_state()

    def test_dest_coords_null_without_streams(self):
        """dest_coords is null when no streams loaded."""
        _, client = self._make_watershed_model()
        try:
            resp = client.get("/api/small-watersheds")
            ws = resp.json()["watersheds"][0]
            assert ws["dest_stream_node"] == 10
            assert ws["dest_coords"] is None
        finally:
            _reset_model_state()

    def test_basic_watershed_fields(self):
        """Verify basic watershed fields (id, area, n_gw_nodes)."""
        _, client = self._make_watershed_model()
        try:
            resp = client.get("/api/small-watersheds")
            ws = resp.json()["watersheds"][0]

            assert ws["id"] == 1
            assert ws["area"] == 5000.0
            assert ws["n_gw_nodes"] == 2
        finally:
            _reset_model_state()
