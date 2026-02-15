"""Comprehensive tests for FastAPI results routes (routes/results.py).

Targets 95%+ coverage of all 11 endpoints:
  /api/results/info, /heads, /head-diff, /head-times, /head-range,
  /hydrograph-locations, /hydrograph, /gw-hydrograph-all-layers,
  /hydrographs-multi, /drawdown, /heads-by-element
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("fastapi", reason="FastAPI not available")
pytest.importorskip("pydantic", reason="Pydantic not available")

from fastapi.testclient import TestClient

from pyiwfm.core.mesh import AppGrid, Element, Node
from pyiwfm.visualization.webapi.config import model_state
from pyiwfm.visualization.webapi.server import create_app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UTM_CRS = "+proj=utm +zone=10 +datum=NAD83 +units=us-ft +no_defs"


def _reset_model_state() -> None:
    """Reset the global model_state to a clean state."""
    model_state._model = None
    model_state._mesh_3d = None
    model_state._mesh_surface = None
    model_state._surface_json_data = None
    model_state._bounds = None
    model_state._pv_mesh_3d = None
    model_state._layer_surface_cache = {}
    model_state._crs = _UTM_CRS
    model_state._transformer = None
    model_state._geojson_cache = {}
    model_state._head_loader = None
    model_state._gw_hydrograph_reader = None
    model_state._stream_hydrograph_reader = None
    model_state._subsidence_reader = None
    model_state._budget_readers = {}
    model_state._observations = {}
    model_state._results_dir = None
    model_state._node_id_to_idx = None
    model_state._sorted_elem_ids = None
    # Restore any monkey-patched methods back to the class originals
    for attr in ("get_budget_reader", "get_available_budgets", "reproject_coords",
                 "get_stream_reach_boundaries", "get_head_loader", "get_gw_hydrograph_reader",
                 "get_stream_hydrograph_reader", "get_area_manager", "get_subsidence_reader"):
        if attr in model_state.__dict__:
            del model_state.__dict__[attr]


def _make_grid() -> AppGrid:
    """Create a minimal 4-node, 1-element quad grid."""
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


def _make_mock_model(**kwargs) -> MagicMock:
    """Create a minimal mock IWFMModel."""
    model = MagicMock()
    model.name = "TestModel"
    model.grid = kwargs.get("grid", _make_grid())
    model.metadata = kwargs.get("metadata", {})
    model.has_streams = kwargs.get("with_streams", False)
    model.has_lakes = False
    model.n_nodes = 4
    model.n_elements = 1
    model.n_layers = 2
    model.n_lakes = 0
    model.n_stream_nodes = 0
    model.stratigraphy = None

    # Streams
    if kwargs.get("with_streams", False):
        streams = MagicMock()
        streams.n_nodes = 2
        model.streams = streams
        model.has_streams = True
        model.n_stream_nodes = 2
    else:
        model.streams = None

    # Groundwater
    if kwargs.get("with_groundwater", False):
        gw = MagicMock()
        gw.n_hydrograph_locations = kwargs.get("n_gw_locs", 2)
        locs = []
        for i in range(gw.n_hydrograph_locations):
            loc = MagicMock()
            loc.x = 50.0 + i * 10
            loc.y = 50.0 + i * 10
            loc.name = f"Well-{i + 1}"
            loc.layer = i + 1
            loc.node_id = i + 1
            loc.gw_node = i + 1
            locs.append(loc)
        gw.hydrograph_locations = locs
        gw.subsidence_config = None
        model.groundwater = gw
    else:
        model.groundwater = None

    return model


def _make_head_loader(
    n_frames: int = 5,
    n_nodes: int = 4,
    n_layers: int = 2,
    *,
    with_dry_cells: bool = False,
    short_times: bool = False,
) -> MagicMock:
    """Create a mock LazyHeadDataLoader."""
    loader = MagicMock()
    loader.n_frames = n_frames
    base = datetime(2020, 1, 1)

    # If short_times, provide fewer times than frames to test the
    # ``timestep >= len(loader.times)`` branch.
    n_times = max(1, n_frames - 1) if short_times else n_frames
    loader.times = [base + timedelta(days=30 * i) for i in range(n_times)]

    rng = np.random.default_rng(42)

    def _get_frame(ts: int) -> np.ndarray:
        arr = rng.random((n_nodes, n_layers)) * 100.0
        if with_dry_cells:
            # Mark first node of every layer as dry
            arr[0, :] = -10000.0
        return arr

    loader.get_frame = MagicMock(side_effect=_get_frame)

    def _get_layer_range(layer: int, max_frames: int = 50):
        return (-5.0, 120.0, min(n_frames, max_frames))

    loader.get_layer_range = MagicMock(side_effect=_get_layer_range)
    return loader


def _make_gw_hydro_reader(
    n_columns: int = 3, n_timesteps: int = 10
) -> MagicMock:
    """Create a mock GW hydrograph reader."""
    reader = MagicMock()
    reader.n_columns = n_columns
    reader.n_timesteps = n_timesteps
    base = datetime(2020, 1, 1)

    def _get_ts(col_idx: int):
        times = [
            (base + timedelta(days=30 * i)).isoformat()
            for i in range(n_timesteps)
        ]
        values = (np.arange(n_timesteps, dtype=float) + col_idx).tolist()
        return times, values

    reader.get_time_series = MagicMock(side_effect=_get_ts)
    return reader


def _make_stream_hydro_reader(
    n_columns: int = 6,
    n_timesteps: int = 10,
    hydrograph_ids: list[int] | None = None,
    *,
    find_returns_none: bool = False,
) -> MagicMock:
    """Create a mock stream hydrograph reader."""
    reader = MagicMock()
    reader.n_columns = n_columns
    reader.n_timesteps = n_timesteps
    reader.hydrograph_ids = hydrograph_ids if hydrograph_ids is not None else [10, 20, 30]
    base = datetime(2020, 1, 1)

    def _find(node_id: int):
        if find_returns_none:
            return None
        try:
            return reader.hydrograph_ids.index(node_id)
        except ValueError:
            return None

    reader.find_column_by_node_id = MagicMock(side_effect=_find)

    def _get_ts(col_idx: int):
        times = [
            (base + timedelta(days=30 * i)).isoformat()
            for i in range(n_timesteps)
        ]
        values = (np.arange(n_timesteps, dtype=float) * 10 + col_idx).tolist()
        return times, values

    reader.get_time_series = MagicMock(side_effect=_get_ts)
    return reader


def _make_subsidence_reader(
    n_columns: int = 3, n_timesteps: int = 10
) -> MagicMock:
    """Create a mock subsidence hydrograph reader."""
    reader = MagicMock()
    reader.n_columns = n_columns
    reader.n_timesteps = n_timesteps
    base = datetime(2020, 1, 1)

    def _get_ts(col_idx: int):
        times = [
            (base + timedelta(days=30 * i)).isoformat()
            for i in range(n_timesteps)
        ]
        values = (np.arange(n_timesteps, dtype=float) * 0.01 + col_idx * 0.001).tolist()
        return times, values

    reader.get_time_series = MagicMock(side_effect=_get_ts)
    return reader


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset global model_state before and after every test."""
    _reset_model_state()
    yield
    _reset_model_state()


@pytest.fixture()
def app():
    """A bare FastAPI app (model may be set per-test)."""
    return create_app()


@pytest.fixture()
def client(app) -> TestClient:
    """TestClient wrapping the app."""
    return TestClient(app)


# ===========================================================================
# 1. GET /api/results/info
# ===========================================================================


class TestResultsInfo:
    """Tests for GET /api/results/info."""

    def test_no_model_returns_404(self, client: TestClient) -> None:
        resp = client.get("/api/results/info")
        assert resp.status_code == 404

    def test_success_with_model(self, client: TestClient) -> None:
        model = _make_mock_model()
        model_state._model = model
        resp = client.get("/api/results/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "has_results" in data
        assert "available_budgets" in data
        assert "n_head_timesteps" in data


# ===========================================================================
# 2. GET /api/results/heads
# ===========================================================================


class TestResultsHeads:
    """Tests for GET /api/results/heads."""

    def test_no_loader_returns_404(self, client: TestClient) -> None:
        model_state._model = _make_mock_model()
        resp = client.get("/api/results/heads")
        assert resp.status_code == 404
        assert "No head data" in resp.json()["detail"]

    def test_timestep_out_of_range(self, client: TestClient) -> None:
        model_state._model = _make_mock_model()
        model_state._head_loader = _make_head_loader(n_frames=3)
        resp = client.get("/api/results/heads?timestep=5&layer=1")
        assert resp.status_code == 400
        assert "out of range" in resp.json()["detail"]

    def test_layer_out_of_range(self, client: TestClient) -> None:
        model_state._model = _make_mock_model()
        model_state._head_loader = _make_head_loader(n_frames=3, n_layers=1)
        resp = client.get("/api/results/heads?timestep=0&layer=5")
        assert resp.status_code == 400
        assert "Layer" in resp.json()["detail"]

    def test_normal_case(self, client: TestClient) -> None:
        model_state._model = _make_mock_model()
        model_state._head_loader = _make_head_loader(n_frames=5, n_nodes=4, n_layers=2)
        resp = client.get("/api/results/heads?timestep=0&layer=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["timestep_index"] == 0
        assert data["layer"] == 1
        assert data["datetime"] is not None
        assert len(data["values"]) == 4

    def test_no_datetime_when_timestep_exceeds_times(self, client: TestClient) -> None:
        """When timestep >= len(loader.times), datetime should be None."""
        model_state._model = _make_mock_model()
        loader = _make_head_loader(n_frames=5, n_nodes=4, n_layers=2, short_times=True)
        model_state._head_loader = loader
        # loader.times has n_frames-1 = 4 entries; timestep 4 is valid but
        # index 4 >= len(times).
        resp = client.get("/api/results/heads?timestep=4&layer=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["datetime"] is None


# ===========================================================================
# 3. GET /api/results/head-diff
# ===========================================================================


class TestResultsHeadDiff:
    """Tests for GET /api/results/head-diff."""

    def test_no_loader(self, client: TestClient) -> None:
        model_state._model = _make_mock_model()
        resp = client.get("/api/results/head-diff?timestep_a=0&timestep_b=1&layer=1")
        assert resp.status_code == 404

    def test_timestep_a_out_of_range(self, client: TestClient) -> None:
        model_state._model = _make_mock_model()
        model_state._head_loader = _make_head_loader(n_frames=3)
        resp = client.get("/api/results/head-diff?timestep_a=5&timestep_b=0&layer=1")
        assert resp.status_code == 400
        assert "timestep_a" in resp.json()["detail"]

    def test_timestep_b_out_of_range(self, client: TestClient) -> None:
        model_state._model = _make_mock_model()
        model_state._head_loader = _make_head_loader(n_frames=3)
        resp = client.get("/api/results/head-diff?timestep_a=0&timestep_b=10&layer=1")
        assert resp.status_code == 400
        assert "timestep_b" in resp.json()["detail"]

    def test_layer_out_of_range(self, client: TestClient) -> None:
        model_state._model = _make_mock_model()
        model_state._head_loader = _make_head_loader(n_frames=3, n_layers=1)
        resp = client.get("/api/results/head-diff?timestep_a=0&timestep_b=1&layer=5")
        assert resp.status_code == 400
        assert "Layer" in resp.json()["detail"]

    def test_normal_diff(self, client: TestClient) -> None:
        model_state._model = _make_mock_model()
        model_state._head_loader = _make_head_loader(n_frames=5, n_nodes=4, n_layers=2)
        resp = client.get("/api/results/head-diff?timestep_a=0&timestep_b=2&layer=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["timestep_a"] == 0
        assert data["timestep_b"] == 2
        assert data["layer"] == 1
        assert "values" in data
        assert "min" in data
        assert "max" in data
        assert len(data["values"]) == 4

    def test_dry_cells_produce_none(self, client: TestClient) -> None:
        """Dry cells (< -9000) should appear as None in the diff list."""
        model_state._model = _make_mock_model()
        model_state._head_loader = _make_head_loader(
            n_frames=5, n_nodes=4, n_layers=2, with_dry_cells=True
        )
        resp = client.get("/api/results/head-diff?timestep_a=0&timestep_b=1&layer=1")
        assert resp.status_code == 200
        data = resp.json()
        # First node should be None (dry)
        assert data["values"][0] is None
        # At least some values should be non-None
        non_none = [v for v in data["values"] if v is not None]
        assert len(non_none) > 0

    def test_all_dry_cells_min_max_zero(self, client: TestClient) -> None:
        """When all cells are dry, min/max should be 0."""
        model_state._model = _make_mock_model()
        loader = MagicMock()
        loader.n_frames = 2
        loader.times = [datetime(2020, 1, 1), datetime(2020, 2, 1)]
        # All values < -9000
        all_dry = np.full((4, 2), -10000.0)
        loader.get_frame = MagicMock(return_value=all_dry)
        model_state._head_loader = loader
        resp = client.get("/api/results/head-diff?timestep_a=0&timestep_b=1&layer=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["min"] == 0.0
        assert data["max"] == 0.0
        assert all(v is None for v in data["values"])

    def test_datetime_none_when_short_times(self, client: TestClient) -> None:
        """datetime_a or datetime_b is None when timestep >= len(times)."""
        model_state._model = _make_mock_model()
        loader = _make_head_loader(n_frames=5, n_nodes=4, n_layers=2, short_times=True)
        model_state._head_loader = loader
        # times has 4 entries (indices 0-3), timestep 4 has no datetime
        resp = client.get("/api/results/head-diff?timestep_a=0&timestep_b=4&layer=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["datetime_a"] is not None
        assert data["datetime_b"] is None


# ===========================================================================
# 4. GET /api/results/head-times
# ===========================================================================


class TestResultsHeadTimes:
    """Tests for GET /api/results/head-times."""

    def test_no_loader(self, client: TestClient) -> None:
        model_state._model = _make_mock_model()
        resp = client.get("/api/results/head-times")
        assert resp.status_code == 404

    def test_normal(self, client: TestClient) -> None:
        model_state._model = _make_mock_model()
        model_state._head_loader = _make_head_loader(n_frames=5)
        resp = client.get("/api/results/head-times")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_timesteps"] == 5
        assert len(data["times"]) == 5
        # Each time should be an ISO string
        for t in data["times"]:
            datetime.fromisoformat(t)


# ===========================================================================
# 5. GET /api/results/head-range
# ===========================================================================


class TestResultsHeadRange:
    """Tests for GET /api/results/head-range."""

    def test_no_loader(self, client: TestClient) -> None:
        model_state._model = _make_mock_model()
        resp = client.get("/api/results/head-range?layer=1")
        assert resp.status_code == 404

    def test_normal(self, client: TestClient) -> None:
        model_state._model = _make_mock_model()
        model_state._head_loader = _make_head_loader(n_frames=10, n_layers=2)
        resp = client.get("/api/results/head-range?layer=1&max_frames=5")
        assert resp.status_code == 200
        data = resp.json()
        assert data["layer"] == 1
        assert "min" in data
        assert "max" in data
        assert data["n_timesteps"] == 10
        assert data["n_frames_scanned"] <= 10


# ===========================================================================
# 6. GET /api/results/hydrograph-locations
# ===========================================================================


class TestHydrographLocations:
    """Tests for GET /api/results/hydrograph-locations."""

    def test_no_model(self, client: TestClient) -> None:
        resp = client.get("/api/results/hydrograph-locations")
        assert resp.status_code == 404

    def test_with_model(self, client: TestClient) -> None:
        model_state._model = _make_mock_model(with_groundwater=True)
        # Mock reproject_coords to identity
        with patch.object(model_state, "reproject_coords", side_effect=lambda x, y: (x, y)):
            resp = client.get("/api/results/hydrograph-locations")
        assert resp.status_code == 200


# ===========================================================================
# 7. GET /api/results/hydrograph
# ===========================================================================


class TestHydrograph:
    """Tests for GET /api/results/hydrograph."""

    # ---- GW type ----

    def test_gw_no_model(self, client: TestClient) -> None:
        resp = client.get("/api/results/hydrograph?type=gw&location_id=1")
        assert resp.status_code == 404

    def test_gw_no_reader(self, client: TestClient) -> None:
        model_state._model = _make_mock_model()
        resp = client.get("/api/results/hydrograph?type=gw&location_id=1")
        assert resp.status_code == 404
        assert "No GW hydrograph" in resp.json()["detail"]

    def test_gw_reader_zero_timesteps(self, client: TestClient) -> None:
        model_state._model = _make_mock_model()
        reader = _make_gw_hydro_reader(n_timesteps=0)
        model_state._gw_hydrograph_reader = reader
        resp = client.get("/api/results/hydrograph?type=gw&location_id=1")
        assert resp.status_code == 404

    def test_gw_location_out_of_range_low(self, client: TestClient) -> None:
        model_state._model = _make_mock_model()
        model_state._gw_hydrograph_reader = _make_gw_hydro_reader(n_columns=3)
        resp = client.get("/api/results/hydrograph?type=gw&location_id=0")
        # location_id=0 => column_index=-1 => out of range
        assert resp.status_code == 404
        assert "out of range" in resp.json()["detail"]

    def test_gw_location_out_of_range_high(self, client: TestClient) -> None:
        model_state._model = _make_mock_model()
        model_state._gw_hydrograph_reader = _make_gw_hydro_reader(n_columns=3)
        resp = client.get("/api/results/hydrograph?type=gw&location_id=10")
        assert resp.status_code == 404
        assert "out of range" in resp.json()["detail"]

    def test_gw_success_with_groundwater_locations(self, client: TestClient) -> None:
        model_state._model = _make_mock_model(with_groundwater=True, n_gw_locs=3)
        model_state._gw_hydrograph_reader = _make_gw_hydro_reader(n_columns=3)
        resp = client.get("/api/results/hydrograph?type=gw&location_id=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "gw"
        assert data["location_id"] == 1
        assert data["name"] == "Well-1"
        assert data["layer"] == 1
        assert data["units"] == "ft"
        assert len(data["times"]) == 10
        assert len(data["values"]) == 10

    def test_gw_success_without_groundwater(self, client: TestClient) -> None:
        """When model has no groundwater component, default name/layer used."""
        model_state._model = _make_mock_model(with_groundwater=False)
        model_state._gw_hydrograph_reader = _make_gw_hydro_reader(n_columns=3)
        resp = client.get("/api/results/hydrograph?type=gw&location_id=2")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "GW Hydrograph 2"
        assert data["layer"] == 1

    def test_gw_location_name_is_none(self, client: TestClient) -> None:
        """When loc.name is None, fallback name is used."""
        model = _make_mock_model(with_groundwater=True, n_gw_locs=2)
        model.groundwater.hydrograph_locations[0].name = None
        model_state._model = model
        model_state._gw_hydrograph_reader = _make_gw_hydro_reader(n_columns=2)
        resp = client.get("/api/results/hydrograph?type=gw&location_id=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "GW Hydrograph 1"

    # ---- Stream type ----

    def test_stream_no_reader(self, client: TestClient) -> None:
        model_state._model = _make_mock_model()
        resp = client.get("/api/results/hydrograph?type=stream&location_id=10")
        assert resp.status_code == 404
        assert "No stream" in resp.json()["detail"]

    def test_stream_reader_zero_timesteps(self, client: TestClient) -> None:
        model_state._model = _make_mock_model()
        reader = _make_stream_hydro_reader(n_timesteps=0)
        model_state._stream_hydrograph_reader = reader
        resp = client.get("/api/results/hydrograph?type=stream&location_id=10")
        assert resp.status_code == 404

    def test_stream_found_by_node_id(self, client: TestClient) -> None:
        model_state._model = _make_mock_model(
            metadata={"stream_hydrograph_specs": [{"node_id": 10, "name": "SR-10"}]}
        )
        model_state._stream_hydrograph_reader = _make_stream_hydro_reader(
            hydrograph_ids=[10, 20, 30]
        )
        resp = client.get("/api/results/hydrograph?type=stream&location_id=10")
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "stream"
        assert data["name"] == "SR-10"
        assert data["units"] == "cfs"

    def test_stream_fallback_to_hydrograph_ids(self, client: TestClient) -> None:
        """When find_column_by_node_id returns None, fall back to hydrograph_ids."""
        model_state._model = _make_mock_model()
        reader = _make_stream_hydro_reader(
            hydrograph_ids=[10, 20, 30], find_returns_none=True
        )
        model_state._stream_hydrograph_reader = reader
        resp = client.get("/api/results/hydrograph?type=stream&location_id=20")
        assert resp.status_code == 200
        data = resp.json()
        assert data["location_id"] == 20
        assert data["name"] == "Stream Node 20"

    def test_stream_not_found(self, client: TestClient) -> None:
        """When node_id is not in either lookup, 404."""
        model_state._model = _make_mock_model()
        reader = _make_stream_hydro_reader(
            hydrograph_ids=[10, 20, 30], find_returns_none=True
        )
        model_state._stream_hydrograph_reader = reader
        resp = client.get("/api/results/hydrograph?type=stream&location_id=999")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"]

    def test_stream_with_stage_output_type_2(self, client: TestClient) -> None:
        """When output_type==2, result should include flow_values and stage_values."""
        model_state._model = _make_mock_model(
            metadata={
                "stream_hydrograph_specs": [
                    {"node_id": 10, "name": "SR-10"},
                    {"node_id": 20, "name": "SR-20"},
                    {"node_id": 30, "name": "SR-30"},
                ],
                "stream_hydrograph_output_type": 2,
            }
        )
        # 6 columns: 3 flow + 3 stage
        model_state._stream_hydrograph_reader = _make_stream_hydro_reader(
            n_columns=6, hydrograph_ids=[10, 20, 30]
        )
        resp = client.get("/api/results/hydrograph?type=stream&location_id=10")
        assert resp.status_code == 200
        data = resp.json()
        assert "flow_values" in data
        assert "stage_values" in data
        assert data["flow_units"] == "cfs"
        assert data["stage_units"] == "ft"

    def test_stream_output_type_2_stage_col_out_of_range(self, client: TestClient) -> None:
        """stage_col_idx >= n_columns, so no stage data added."""
        model_state._model = _make_mock_model(
            metadata={
                "stream_hydrograph_specs": [
                    {"node_id": 10, "name": "SR-10"},
                    {"node_id": 20, "name": "SR-20"},
                    {"node_id": 30, "name": "SR-30"},
                ],
                "stream_hydrograph_output_type": 2,
            }
        )
        # Only 3 columns (not enough for stage)
        model_state._stream_hydrograph_reader = _make_stream_hydro_reader(
            n_columns=3, hydrograph_ids=[10, 20, 30]
        )
        resp = client.get("/api/results/hydrograph?type=stream&location_id=10")
        assert resp.status_code == 200
        data = resp.json()
        # stage_col_idx = 0 + 3 = 3, >= n_columns=3 => no stage
        assert "flow_values" not in data
        assert "stage_values" not in data

    def test_stream_no_model_metadata(self, client: TestClient) -> None:
        """When model is truthy but metadata has no specs, default name is used."""
        model_state._model = _make_mock_model(metadata={})
        model_state._stream_hydrograph_reader = _make_stream_hydro_reader(
            hydrograph_ids=[10, 20]
        )
        resp = client.get("/api/results/hydrograph?type=stream&location_id=10")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Stream Node 10"

    # ---- Subsidence type ----

    def test_subsidence_no_reader(self, client: TestClient) -> None:
        model_state._model = _make_mock_model()
        resp = client.get("/api/results/hydrograph?type=subsidence&location_id=1")
        assert resp.status_code == 404
        assert "No subsidence" in resp.json()["detail"]

    def test_subsidence_reader_zero_timesteps(self, client: TestClient) -> None:
        model_state._model = _make_mock_model()
        reader = _make_subsidence_reader(n_timesteps=0)
        model_state._subsidence_reader = reader
        resp = client.get("/api/results/hydrograph?type=subsidence&location_id=1")
        assert resp.status_code == 404

    def test_subsidence_match_by_spec_id(self, client: TestClient) -> None:
        """When subsidence_config has specs, match by spec.id."""
        model = _make_mock_model(with_groundwater=True)
        subs_config = MagicMock()
        spec = MagicMock()
        spec.id = 5
        spec.name = "Subs-5"
        spec.layer = 2
        subs_config.hydrograph_specs = [spec]
        model.groundwater.subsidence_config = subs_config
        model_state._model = model
        model_state._subsidence_reader = _make_subsidence_reader(n_columns=3)

        resp = client.get("/api/results/hydrograph?type=subsidence&location_id=5")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Subs-5"
        assert data["layer"] == 2

    def test_subsidence_fallback_to_column_index(self, client: TestClient) -> None:
        """When no spec matches, fall back to 1-based column index."""
        model = _make_mock_model(with_groundwater=True)
        subs_config = MagicMock()
        subs_config.hydrograph_specs = []  # no specs
        model.groundwater.subsidence_config = subs_config
        model_state._model = model
        model_state._subsidence_reader = _make_subsidence_reader(n_columns=3)

        resp = client.get("/api/results/hydrograph?type=subsidence&location_id=2")
        assert resp.status_code == 200
        data = resp.json()
        # col_idx = 2-1 = 1 is valid for n_columns=3
        assert data["location_id"] == 2
        assert data["name"] == "Subsidence Obs 2"
        assert data["layer"] == 1  # fallback since no specs

    def test_subsidence_not_found(self, client: TestClient) -> None:
        """When ID doesn't match spec and column index is out of range, 404."""
        model = _make_mock_model(with_groundwater=True)
        subs_config = MagicMock()
        subs_config.hydrograph_specs = []
        model.groundwater.subsidence_config = subs_config
        model_state._model = model
        model_state._subsidence_reader = _make_subsidence_reader(n_columns=2)

        # location_id=10 => candidate=9 >= n_columns=2 => not found
        resp = client.get("/api/results/hydrograph?type=subsidence&location_id=10")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"]

    def test_subsidence_no_groundwater_no_specs(self, client: TestClient) -> None:
        """When model has no groundwater, specs is empty, fallback column used."""
        model_state._model = _make_mock_model(with_groundwater=False)
        model_state._subsidence_reader = _make_subsidence_reader(n_columns=3)
        resp = client.get("/api/results/hydrograph?type=subsidence&location_id=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Subsidence Obs 1"

    def test_subsidence_spec_name_is_none(self, client: TestClient) -> None:
        """When spec.name is None, fallback name used."""
        model = _make_mock_model(with_groundwater=True)
        subs_config = MagicMock()
        spec = MagicMock()
        spec.id = 1
        spec.name = None
        spec.layer = 3
        subs_config.hydrograph_specs = [spec]
        model.groundwater.subsidence_config = subs_config
        model_state._model = model
        model_state._subsidence_reader = _make_subsidence_reader(n_columns=3)

        resp = client.get("/api/results/hydrograph?type=subsidence&location_id=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Subsidence Obs 1"
        assert data["layer"] == 3

    # ---- Unknown type ----

    def test_unknown_type_returns_400(self, client: TestClient) -> None:
        model_state._model = _make_mock_model()
        resp = client.get("/api/results/hydrograph?type=bogus&location_id=1")
        assert resp.status_code == 400
        assert "Unknown hydrograph type" in resp.json()["detail"]


# ===========================================================================
# 8. GET /api/results/gw-hydrograph-all-layers
# ===========================================================================


class TestGwHydrographAllLayers:
    """Tests for GET /api/results/gw-hydrograph-all-layers."""

    def test_no_model(self, client: TestClient) -> None:
        resp = client.get("/api/results/gw-hydrograph-all-layers?location_id=1")
        assert resp.status_code == 404
        assert "No model" in resp.json()["detail"]

    def test_no_groundwater(self, client: TestClient) -> None:
        model_state._model = _make_mock_model(with_groundwater=False)
        resp = client.get("/api/results/gw-hydrograph-all-layers?location_id=1")
        assert resp.status_code == 404
        assert "No groundwater" in resp.json()["detail"]

    def test_model_none_groundwater_none(self, client: TestClient) -> None:
        """model exists but model itself is None (edge case of mock)."""
        model = _make_mock_model(with_groundwater=True)
        model.groundwater = None
        model_state._model = model
        resp = client.get("/api/results/gw-hydrograph-all-layers?location_id=1")
        assert resp.status_code == 404

    def test_location_out_of_range_low(self, client: TestClient) -> None:
        model_state._model = _make_mock_model(with_groundwater=True, n_gw_locs=2)
        resp = client.get("/api/results/gw-hydrograph-all-layers?location_id=0")
        assert resp.status_code == 404
        assert "out of range" in resp.json()["detail"]

    def test_location_out_of_range_high(self, client: TestClient) -> None:
        model_state._model = _make_mock_model(with_groundwater=True, n_gw_locs=2)
        resp = client.get("/api/results/gw-hydrograph-all-layers?location_id=10")
        assert resp.status_code == 404
        assert "out of range" in resp.json()["detail"]

    def test_no_node_id(self, client: TestClient) -> None:
        """When the hydrograph location has node_id=0 and gw_node=0, 404."""
        model = _make_mock_model(with_groundwater=True, n_gw_locs=2)
        loc = model.groundwater.hydrograph_locations[0]
        loc.node_id = 0
        loc.gw_node = 0
        model_state._model = model
        resp = client.get("/api/results/gw-hydrograph-all-layers?location_id=1")
        assert resp.status_code == 404
        assert "No node ID" in resp.json()["detail"]

    def test_no_head_loader(self, client: TestClient) -> None:
        model_state._model = _make_mock_model(with_groundwater=True, n_gw_locs=2)
        # No head_loader set => get_head_loader returns None
        resp = client.get("/api/results/gw-hydrograph-all-layers?location_id=1")
        assert resp.status_code == 404
        assert "No head data" in resp.json()["detail"]

    def test_head_loader_zero_frames(self, client: TestClient) -> None:
        model_state._model = _make_mock_model(with_groundwater=True, n_gw_locs=2)
        loader = _make_head_loader(n_frames=0)
        # Override n_frames to 0
        loader.n_frames = 0
        model_state._head_loader = loader
        resp = client.get("/api/results/gw-hydrograph-all-layers?location_id=1")
        assert resp.status_code == 404
        assert "No head data" in resp.json()["detail"]

    def test_node_not_in_grid(self, client: TestClient) -> None:
        """When node_id is not found in node_id_to_idx mapping."""
        model = _make_mock_model(with_groundwater=True, n_gw_locs=2)
        # Set node_id to something not in the grid
        model.groundwater.hydrograph_locations[0].node_id = 999
        model.groundwater.hydrograph_locations[0].gw_node = 999
        model_state._model = model
        model_state._head_loader = _make_head_loader(n_frames=3, n_nodes=4, n_layers=2)
        resp = client.get("/api/results/gw-hydrograph-all-layers?location_id=1")
        assert resp.status_code == 404
        assert "not found in grid" in resp.json()["detail"]

    def test_normal_case(self, client: TestClient) -> None:
        model = _make_mock_model(with_groundwater=True, n_gw_locs=2)
        model_state._model = model
        model_state._head_loader = _make_head_loader(n_frames=3, n_nodes=4, n_layers=2)
        resp = client.get("/api/results/gw-hydrograph-all-layers?location_id=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["location_id"] == 1
        assert data["node_id"] == 1
        assert data["n_layers"] == 2
        assert len(data["times"]) == 3
        assert len(data["layers"]) == 2
        assert data["layers"][0]["layer"] == 1
        assert data["layers"][1]["layer"] == 2
        # Each layer has 3 timesteps of values
        assert len(data["layers"][0]["values"]) == 3

    def test_dry_cells_become_none(self, client: TestClient) -> None:
        model = _make_mock_model(with_groundwater=True, n_gw_locs=2)
        model_state._model = model
        model_state._head_loader = _make_head_loader(
            n_frames=3, n_nodes=4, n_layers=2, with_dry_cells=True
        )
        # loc 1 maps to node_id=1, which is grid node 1, index 0
        # with_dry_cells=True makes index 0 dry (-10000)
        resp = client.get("/api/results/gw-hydrograph-all-layers?location_id=1")
        assert resp.status_code == 200
        data = resp.json()
        # All values for node_idx=0 should be None since it's dry
        for layer_data in data["layers"]:
            for v in layer_data["values"]:
                assert v is None

    def test_loc_name_is_none(self, client: TestClient) -> None:
        """When loc.name is None, fallback name used."""
        model = _make_mock_model(with_groundwater=True, n_gw_locs=2)
        model.groundwater.hydrograph_locations[0].name = None
        model_state._model = model
        model_state._head_loader = _make_head_loader(n_frames=2, n_nodes=4, n_layers=1)
        resp = client.get("/api/results/gw-hydrograph-all-layers?location_id=1")
        assert resp.status_code == 200
        assert resp.json()["name"] == "GW Hydrograph 1"


# ===========================================================================
# 9. GET /api/results/hydrographs-multi
# ===========================================================================


class TestHydrographsMulti:
    """Tests for GET /api/results/hydrographs-multi."""

    def test_no_model(self, client: TestClient) -> None:
        resp = client.get("/api/results/hydrographs-multi?type=gw&ids=1,2")
        assert resp.status_code == 404

    def test_invalid_ids_format(self, client: TestClient) -> None:
        model_state._model = _make_mock_model()
        resp = client.get("/api/results/hydrographs-multi?type=gw&ids=a,b,c")
        assert resp.status_code == 400
        assert "Invalid IDs" in resp.json()["detail"]

    def test_empty_ids(self, client: TestClient) -> None:
        model_state._model = _make_mock_model()
        resp = client.get("/api/results/hydrographs-multi?type=gw&ids=")
        assert resp.status_code == 400
        assert "No IDs" in resp.json()["detail"]

    def test_unknown_type(self, client: TestClient) -> None:
        model_state._model = _make_mock_model()
        resp = client.get("/api/results/hydrographs-multi?type=bogus&ids=1,2")
        assert resp.status_code == 400
        assert "Unknown type" in resp.json()["detail"]

    # ---- GW multi ----

    def test_gw_no_reader(self, client: TestClient) -> None:
        model_state._model = _make_mock_model()
        resp = client.get("/api/results/hydrographs-multi?type=gw&ids=1,2")
        assert resp.status_code == 404
        assert "No GW" in resp.json()["detail"]

    def test_gw_reader_zero_timesteps(self, client: TestClient) -> None:
        model_state._model = _make_mock_model()
        model_state._gw_hydrograph_reader = _make_gw_hydro_reader(n_timesteps=0)
        resp = client.get("/api/results/hydrographs-multi?type=gw&ids=1")
        assert resp.status_code == 404

    def test_gw_multi_success(self, client: TestClient) -> None:
        model_state._model = _make_mock_model(with_groundwater=True, n_gw_locs=3)
        model_state._gw_hydrograph_reader = _make_gw_hydro_reader(n_columns=3)
        resp = client.get("/api/results/hydrographs-multi?type=gw&ids=1,3")
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "gw"
        assert data["n_series"] == 2
        assert len(data["series"]) == 2
        ids = [s["location_id"] for s in data["series"]]
        assert 1 in ids
        assert 3 in ids

    def test_gw_out_of_range_ids_skipped(self, client: TestClient) -> None:
        """IDs that are out of range should be silently skipped."""
        model_state._model = _make_mock_model(with_groundwater=True, n_gw_locs=3)
        model_state._gw_hydrograph_reader = _make_gw_hydro_reader(n_columns=3)
        resp = client.get("/api/results/hydrographs-multi?type=gw&ids=1,0,50")
        assert resp.status_code == 200
        data = resp.json()
        # Only id=1 is valid (0 and 50 out of range)
        assert data["n_series"] == 1
        assert data["series"][0]["location_id"] == 1

    def test_gw_no_groundwater_component(self, client: TestClient) -> None:
        """When model has no groundwater, default name/layer used."""
        model_state._model = _make_mock_model(with_groundwater=False)
        model_state._gw_hydrograph_reader = _make_gw_hydro_reader(n_columns=3)
        resp = client.get("/api/results/hydrographs-multi?type=gw&ids=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["series"][0]["name"] == "GW Hydrograph 1"
        assert data["series"][0]["layer"] == 1

    def test_gw_loc_name_none(self, client: TestClient) -> None:
        """When loc.name is None, fallback name used."""
        model = _make_mock_model(with_groundwater=True, n_gw_locs=2)
        model.groundwater.hydrograph_locations[0].name = None
        model_state._model = model
        model_state._gw_hydrograph_reader = _make_gw_hydro_reader(n_columns=2)
        resp = client.get("/api/results/hydrographs-multi?type=gw&ids=1")
        assert resp.status_code == 200
        assert resp.json()["series"][0]["name"] == "GW Hydrograph 1"

    # ---- Stream multi ----

    def test_stream_no_reader(self, client: TestClient) -> None:
        model_state._model = _make_mock_model()
        resp = client.get("/api/results/hydrographs-multi?type=stream&ids=10")
        assert resp.status_code == 404
        assert "No stream" in resp.json()["detail"]

    def test_stream_reader_zero_timesteps(self, client: TestClient) -> None:
        model_state._model = _make_mock_model()
        model_state._stream_hydrograph_reader = _make_stream_hydro_reader(n_timesteps=0)
        resp = client.get("/api/results/hydrographs-multi?type=stream&ids=10")
        assert resp.status_code == 404

    def test_stream_multi_success(self, client: TestClient) -> None:
        model_state._model = _make_mock_model(
            metadata={"stream_hydrograph_specs": [
                {"node_id": 10, "name": "SR-10"},
                {"node_id": 20, "name": "SR-20"},
            ]}
        )
        model_state._stream_hydrograph_reader = _make_stream_hydro_reader(
            hydrograph_ids=[10, 20, 30]
        )
        resp = client.get("/api/results/hydrographs-multi?type=stream&ids=10,20")
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "stream"
        assert data["n_series"] == 2
        assert data["series"][0]["name"] == "SR-10"
        assert data["series"][1]["name"] == "SR-20"
        assert data["series"][0]["units"] == "cfs"

    def test_stream_multi_fallback_to_hydrograph_ids(self, client: TestClient) -> None:
        """When find_column_by_node_id returns None, try hydrograph_ids."""
        model_state._model = _make_mock_model()
        reader = _make_stream_hydro_reader(
            hydrograph_ids=[10, 20], find_returns_none=True
        )
        model_state._stream_hydrograph_reader = reader
        resp = client.get("/api/results/hydrographs-multi?type=stream&ids=10,20")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_series"] == 2

    def test_stream_multi_not_found_skipped(self, client: TestClient) -> None:
        """IDs not found in either lookup are silently skipped."""
        model_state._model = _make_mock_model()
        reader = _make_stream_hydro_reader(
            hydrograph_ids=[10, 20], find_returns_none=True
        )
        model_state._stream_hydrograph_reader = reader
        resp = client.get("/api/results/hydrographs-multi?type=stream&ids=10,999")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_series"] == 1
        assert data["series"][0]["location_id"] == 10

    def test_stream_multi_no_model_metadata(self, client: TestClient) -> None:
        """When model metadata has no specs, default name used."""
        model_state._model = _make_mock_model(metadata={})
        model_state._stream_hydrograph_reader = _make_stream_hydro_reader(
            hydrograph_ids=[10]
        )
        resp = client.get("/api/results/hydrographs-multi?type=stream&ids=10")
        assert resp.status_code == 200
        assert resp.json()["series"][0]["name"] == "Stream Node 10"


# ===========================================================================
# 10. GET /api/results/drawdown
# ===========================================================================


class TestDrawdown:
    """Tests for GET /api/results/drawdown."""

    def test_no_loader(self, client: TestClient) -> None:
        model_state._model = _make_mock_model()
        resp = client.get("/api/results/drawdown?layer=1")
        assert resp.status_code == 404

    def test_ref_timestep_out_of_range(self, client: TestClient) -> None:
        model_state._model = _make_mock_model()
        model_state._head_loader = _make_head_loader(n_frames=3)
        resp = client.get("/api/results/drawdown?layer=1&reference_timestep=10")
        assert resp.status_code == 400
        assert "out of range" in resp.json()["detail"]

    def test_layer_out_of_range(self, client: TestClient) -> None:
        model_state._model = _make_mock_model()
        model_state._head_loader = _make_head_loader(n_frames=3, n_layers=1)
        resp = client.get("/api/results/drawdown?layer=5&reference_timestep=0")
        assert resp.status_code == 400
        assert "Layer" in resp.json()["detail"]

    def test_normal(self, client: TestClient) -> None:
        model_state._model = _make_mock_model()
        model_state._head_loader = _make_head_loader(n_frames=3, n_nodes=4, n_layers=2)
        resp = client.get("/api/results/drawdown?layer=1&reference_timestep=0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["layer"] == 1
        assert data["reference_timestep"] == 0
        assert data["n_timesteps"] == 3
        assert len(data["timesteps"]) == 3
        # First timestep should have all-zero drawdown (ref vs. itself)
        ts0 = data["timesteps"][0]
        assert ts0["timestep"] == 0
        assert "values" in ts0
        assert "min" in ts0
        assert "max" in ts0

    def test_dry_cells_become_none(self, client: TestClient) -> None:
        model_state._model = _make_mock_model()
        model_state._head_loader = _make_head_loader(
            n_frames=2, n_nodes=4, n_layers=2, with_dry_cells=True
        )
        resp = client.get("/api/results/drawdown?layer=1")
        assert resp.status_code == 200
        data = resp.json()
        # First node is dry, so its value should be None
        for ts_data in data["timesteps"]:
            assert ts_data["values"][0] is None

    def test_all_dry_min_max_zero(self, client: TestClient) -> None:
        model_state._model = _make_mock_model()
        loader = MagicMock()
        loader.n_frames = 2
        loader.times = [datetime(2020, 1, 1), datetime(2020, 2, 1)]
        all_dry = np.full((4, 2), -10000.0)
        loader.get_frame = MagicMock(return_value=all_dry)
        model_state._head_loader = loader
        resp = client.get("/api/results/drawdown?layer=1")
        assert resp.status_code == 200
        data = resp.json()
        for ts_data in data["timesteps"]:
            assert ts_data["min"] == 0.0
            assert ts_data["max"] == 0.0
            assert all(v is None for v in ts_data["values"])

    def test_datetime_none_when_short_times(self, client: TestClient) -> None:
        """When ts >= len(loader.times), datetime should be None."""
        model_state._model = _make_mock_model()
        loader = _make_head_loader(n_frames=3, n_nodes=4, n_layers=2, short_times=True)
        model_state._head_loader = loader
        resp = client.get("/api/results/drawdown?layer=1")
        assert resp.status_code == 200
        data = resp.json()
        # times has 2 entries for 3 frames; timestep 2 has no datetime
        assert data["timesteps"][2]["datetime"] is None


# ===========================================================================
# 11. GET /api/results/heads-by-element
# ===========================================================================


class TestHeadsByElement:
    """Tests for GET /api/results/heads-by-element."""

    def test_no_loader(self, client: TestClient) -> None:
        model_state._model = _make_mock_model()
        resp = client.get("/api/results/heads-by-element")
        assert resp.status_code == 404

    def test_timestep_out_of_range(self, client: TestClient) -> None:
        model_state._model = _make_mock_model()
        model_state._head_loader = _make_head_loader(n_frames=3)
        resp = client.get("/api/results/heads-by-element?timestep=10")
        assert resp.status_code == 400
        assert "out of range" in resp.json()["detail"]

    def test_layer_out_of_range(self, client: TestClient) -> None:
        model_state._model = _make_mock_model()
        model_state._head_loader = _make_head_loader(n_frames=3, n_layers=1)
        resp = client.get("/api/results/heads-by-element?layer=5")
        assert resp.status_code == 400
        assert "Layer" in resp.json()["detail"]

    def test_no_model_grid(self, client: TestClient) -> None:
        """When model.grid is None, 404."""
        model = _make_mock_model()
        model.grid = None
        model_state._model = model
        model_state._head_loader = _make_head_loader(n_frames=3, n_nodes=4, n_layers=2)
        resp = client.get("/api/results/heads-by-element?timestep=0&layer=1")
        assert resp.status_code == 404
        assert "No model grid" in resp.json()["detail"]

    def test_model_is_none_for_grid(self, client: TestClient) -> None:
        """Edge: model is None after head loader check (hypothetical)."""
        # Set model temporarily to get past is_loaded but then remove it
        model = _make_mock_model()
        model_state._model = model
        loader = _make_head_loader(n_frames=3, n_nodes=4, n_layers=2)
        model_state._head_loader = loader
        # Remove model after loader is cached
        model_state._model = None
        resp = client.get("/api/results/heads-by-element?timestep=0&layer=1")
        assert resp.status_code == 404

    def test_normal_vertex_averaging(self, client: TestClient) -> None:
        """Test normal case with vertex averaging for elements."""
        grid = _make_grid()  # 4 nodes, 1 element with vertices (1,2,3,4)
        model = _make_mock_model(grid=grid)
        model_state._model = model

        # Create a deterministic loader
        loader = MagicMock()
        loader.n_frames = 2
        loader.times = [datetime(2020, 1, 1), datetime(2020, 2, 1)]
        # 4 nodes, 2 layers; all values > -9000
        frame = np.array([
            [10.0, 20.0],
            [30.0, 40.0],
            [50.0, 60.0],
            [70.0, 80.0],
        ])
        loader.get_frame = MagicMock(return_value=frame)
        model_state._head_loader = loader

        resp = client.get("/api/results/heads-by-element?timestep=0&layer=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["timestep_index"] == 0
        assert data["layer"] == 1
        assert data["datetime"] is not None
        assert len(data["values"]) == 1  # 1 element
        # Average of 10, 30, 50, 70 = 40.0
        assert data["values"][0] == 40.0
        assert data["min"] == 40.0
        assert data["max"] == 40.0

    def test_all_dry_lo_hi_defaults(self, client: TestClient) -> None:
        """When all elements are dry, lo=0.0, hi=1.0."""
        grid = _make_grid()
        model = _make_mock_model(grid=grid)
        model_state._model = model

        loader = MagicMock()
        loader.n_frames = 1
        loader.times = [datetime(2020, 1, 1)]
        all_dry = np.full((4, 2), -10000.0)
        loader.get_frame = MagicMock(return_value=all_dry)
        model_state._head_loader = loader

        resp = client.get("/api/results/heads-by-element?timestep=0&layer=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["values"] == [None]
        assert data["min"] == 0.0
        assert data["max"] == 1.0

    def test_partial_dry_nodes(self, client: TestClient) -> None:
        """When some but not all nodes in an element are dry."""
        grid = _make_grid()
        model = _make_mock_model(grid=grid)
        model_state._model = model

        loader = MagicMock()
        loader.n_frames = 1
        loader.times = [datetime(2020, 1, 1)]
        frame = np.array([
            [-10000.0, 20.0],  # node 1 dry in layer 1
            [30.0, 40.0],
            [50.0, 60.0],
            [70.0, 80.0],
        ])
        loader.get_frame = MagicMock(return_value=frame)
        model_state._head_loader = loader

        resp = client.get("/api/results/heads-by-element?timestep=0&layer=1")
        assert resp.status_code == 200
        data = resp.json()
        # Average of 30, 50, 70 = 50.0 (node 1 is dry, excluded)
        assert data["values"][0] == 50.0

    def test_datetime_none_when_short_times(self, client: TestClient) -> None:
        grid = _make_grid()
        model = _make_mock_model(grid=grid)
        model_state._model = model

        loader = MagicMock()
        loader.n_frames = 2
        loader.times = [datetime(2020, 1, 1)]  # only 1 time for 2 frames
        frame = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0]])
        loader.get_frame = MagicMock(return_value=frame)
        model_state._head_loader = loader

        resp = client.get("/api/results/heads-by-element?timestep=1&layer=1")
        assert resp.status_code == 200
        assert resp.json()["datetime"] is None

    def test_multiple_elements_sorting(self, client: TestClient) -> None:
        """Test with 2 elements to verify sorted element order."""
        nodes = {
            1: Node(id=1, x=0.0, y=0.0),
            2: Node(id=2, x=100.0, y=0.0),
            3: Node(id=3, x=100.0, y=100.0),
            4: Node(id=4, x=0.0, y=100.0),
            5: Node(id=5, x=200.0, y=0.0),
            6: Node(id=6, x=200.0, y=100.0),
        }
        elements = {
            2: Element(id=2, vertices=(2, 5, 6, 3), subregion=1),
            1: Element(id=1, vertices=(1, 2, 3, 4), subregion=1),
        }
        grid = AppGrid(nodes=nodes, elements=elements)
        grid.compute_connectivity()
        grid.compute_areas()

        model = _make_mock_model(grid=grid)
        model_state._model = model

        loader = MagicMock()
        loader.n_frames = 1
        loader.times = [datetime(2020, 1, 1)]
        # 6 nodes, 1 layer
        frame = np.array([
            [10.0],  # node 1 idx 0
            [20.0],  # node 2 idx 1
            [30.0],  # node 3 idx 2
            [40.0],  # node 4 idx 3
            [50.0],  # node 5 idx 4
            [60.0],  # node 6 idx 5
        ])
        loader.get_frame = MagicMock(return_value=frame)
        model_state._head_loader = loader

        resp = client.get("/api/results/heads-by-element?timestep=0&layer=1")
        assert resp.status_code == 200
        data = resp.json()
        # Sorted element IDs: [1, 2]
        # Element 1: nodes 1,2,3,4 => avg(10,20,30,40)=25.0
        # Element 2: nodes 2,5,6,3 => avg(20,50,60,30)=40.0
        assert len(data["values"]) == 2
        assert data["values"][0] == 25.0
        assert data["values"][1] == 40.0
