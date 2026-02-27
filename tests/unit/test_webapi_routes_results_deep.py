"""Deep coverage tests for webapi results routes — statistics, cache paths,
drawdown pagination, tile_drain hydrograph, and multi-hydrograph fallback.

Targets uncovered paths in src/pyiwfm/visualization/webapi/routes/results.py.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi", reason="FastAPI not available")
pytest.importorskip("pydantic", reason="Pydantic not available")

import numpy as np
from fastapi.testclient import TestClient

from pyiwfm.core.mesh import AppGrid, Element, Node
from pyiwfm.visualization.webapi.config import ModelState
from pyiwfm.visualization.webapi.server import create_app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RESULTS_PATCH = "pyiwfm.visualization.webapi.routes.results.model_state"


def _make_grid() -> AppGrid:
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


def _make_state_with_head_loader(
    n_frames: int = 5,
    n_nodes: int = 4,
    n_layers: int = 2,
) -> ModelState:
    """Build a ModelState with a mocked head_loader attached."""
    state = ModelState()
    model = MagicMock()
    model.name = "TestModel"
    model.grid = _make_grid()
    model.mesh = model.grid
    model.n_nodes = n_nodes
    model.n_elements = 1
    model.n_layers = n_layers
    model.has_streams = False
    model.has_lakes = False
    model.streams = None
    model.lakes = None
    model.groundwater = None
    model.stratigraphy = None
    model.rootzone = None
    model.metadata = {}
    state._model = model

    # Build a mock head loader
    loader = MagicMock()
    loader.n_frames = n_frames
    loader.shape = (n_nodes, n_layers)
    times = [datetime(2020, 1, 1 + i) for i in range(n_frames)]
    loader.times = times

    def _get_frame(ts: int) -> np.ndarray:
        rng = np.random.default_rng(seed=ts)
        return rng.uniform(low=10.0, high=100.0, size=(n_nodes, n_layers))

    loader.get_frame = MagicMock(side_effect=_get_frame)

    def _get_layer_range(layer: int = 1, max_frames: int = 0) -> tuple:
        return (10.0, 100.0, n_frames)

    loader.get_layer_range = MagicMock(side_effect=_get_layer_range)

    state._head_loader = loader
    return state


def _client_with_state(state: ModelState) -> TestClient:
    app = create_app()
    with patch(RESULTS_PATCH, state):
        yield TestClient(app)


@pytest.fixture
def state_with_heads() -> ModelState:
    return _make_state_with_head_loader()


@pytest.fixture
def client_heads(state_with_heads: ModelState):
    app = create_app()
    with patch(RESULTS_PATCH, state_with_heads):
        yield TestClient(app)


# ---------------------------------------------------------------------------
# 1. get_head_statistics() — GET /api/results/statistics
# ---------------------------------------------------------------------------


class TestHeadStatistics:
    """Tests for the /api/results/statistics endpoint."""

    def test_statistics_returns_expected_fields(self, client_heads) -> None:
        resp = client_heads.get("/api/results/statistics?layer=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["layer"] == 1
        assert data["n_nodes"] == 4
        assert data["n_frames_sampled"] == 5
        assert data["n_total_frames"] == 5
        assert "global" in data
        assert "per_node" in data
        g = data["global"]
        assert g["min"] is not None
        assert g["max"] is not None
        assert g["mean"] is not None
        assert g["std"] is not None
        pn = data["per_node"]
        assert len(pn["min"]) == 4
        assert len(pn["max"]) == 4

    def test_statistics_with_max_frames(self, client_heads) -> None:
        resp = client_heads.get("/api/results/statistics?layer=1&max_frames=2")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_frames_sampled"] == 2

    def test_statistics_layer_out_of_range(self, client_heads) -> None:
        resp = client_heads.get("/api/results/statistics?layer=99")
        assert resp.status_code == 400

    def test_statistics_no_head_loader(self) -> None:
        state = ModelState()
        state._model = MagicMock()
        app = create_app()
        with patch(RESULTS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/results/statistics?layer=1")
            assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 2. get_heads_by_element() — cache hit path
# ---------------------------------------------------------------------------


class TestHeadsByElementCache:
    """Tests for the /api/results/heads-by-element cache path."""

    def test_cache_hit_returns_cached_values(self) -> None:
        state = _make_state_with_head_loader()
        cached_values = [50.0, None]
        state.get_cached_head_by_element = MagicMock(
            return_value=(cached_values, 10.0, 90.0)
        )
        app = create_app()
        with patch(RESULTS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/results/heads-by-element?timestep=0&layer=1")
            assert resp.status_code == 200
            data = resp.json()
            assert data["values"] == cached_values
            assert data["min"] == 10.0
            assert data["max"] == 90.0
            state.get_cached_head_by_element.assert_called_once_with(0, 1)

    def test_cache_miss_falls_through(self) -> None:
        state = _make_state_with_head_loader()
        state.get_cached_head_by_element = MagicMock(return_value=None)
        app = create_app()
        with patch(RESULTS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/results/heads-by-element?timestep=0&layer=1")
            assert resp.status_code == 200
            data = resp.json()
            # Should have computed element heads from frames
            assert "values" in data
            assert data["layer"] == 1


# ---------------------------------------------------------------------------
# 3. get_drawdown() — pagination (offset, limit, skip)
# ---------------------------------------------------------------------------


class TestDrawdownPagination:
    """Tests for the /api/results/drawdown endpoint with pagination params."""

    def test_drawdown_basic(self, client_heads) -> None:
        resp = client_heads.get("/api/results/drawdown?layer=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["layer"] == 1
        assert data["reference_timestep"] == 0
        assert data["n_total_timesteps"] == 5
        assert "timesteps" in data
        assert len(data["timesteps"]) == 5

    def test_drawdown_with_offset_and_limit(self, client_heads) -> None:
        resp = client_heads.get(
            "/api/results/drawdown?layer=1&offset=1&limit=2"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_timesteps"] == 2
        ts_indices = [ts["timestep"] for ts in data["timesteps"]]
        assert ts_indices == [1, 2]

    def test_drawdown_with_skip(self, client_heads) -> None:
        resp = client_heads.get("/api/results/drawdown?layer=1&skip=2")
        assert resp.status_code == 200
        data = resp.json()
        ts_indices = [ts["timestep"] for ts in data["timesteps"]]
        assert ts_indices == [0, 2, 4]

    def test_drawdown_layer_out_of_range(self, client_heads) -> None:
        resp = client_heads.get("/api/results/drawdown?layer=99")
        assert resp.status_code == 400

    def test_drawdown_reference_timestep_out_of_range(self, client_heads) -> None:
        resp = client_heads.get(
            "/api/results/drawdown?layer=1&reference_timestep=999"
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# 4. get_hydrograph(type="tile_drain") — tile drain path
# ---------------------------------------------------------------------------


class TestTileDrainHydrograph:
    """Tests for the tile_drain hydrograph path."""

    def test_tile_drain_hydrograph_success(self) -> None:
        state = ModelState()
        model = MagicMock()
        model.name = "TestModel"
        model.grid = _make_grid()
        model.metadata = {}
        gw = MagicMock()
        td_specs = [{"id": 1, "name": "TD #1"}]
        gw.td_hydro_specs = td_specs
        model.groundwater = gw
        state._model = model

        reader = MagicMock()
        reader.n_timesteps = 3
        reader.n_columns = 2
        reader.get_time_series = MagicMock(
            return_value=(["2020-01-01", "2020-02-01", "2020-03-01"], [1.0, 2.0, 3.0])
        )
        state._tile_drain_reader = reader

        app = create_app()
        with patch(RESULTS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/results/hydrograph?type=tile_drain&location_id=1")
            assert resp.status_code == 200
            data = resp.json()
            assert data["type"] == "tile_drain"
            assert data["name"] == "TD #1"
            assert data["values"] == [1.0, 2.0, 3.0]

    def test_tile_drain_no_reader(self) -> None:
        state = ModelState()
        state._model = MagicMock()
        state._model.metadata = {}
        state._model.groundwater = None
        app = create_app()
        with patch(RESULTS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/results/hydrograph?type=tile_drain&location_id=1")
            assert resp.status_code == 404

    def test_unknown_hydrograph_type(self) -> None:
        state = ModelState()
        state._model = MagicMock()
        state._model.metadata = {}
        app = create_app()
        with patch(RESULTS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/results/hydrograph?type=bogus&location_id=1")
            assert resp.status_code == 400
            assert "Unknown hydrograph type" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# 5. get_hydrographs_multi(type="gw") — head loader fallback path
# ---------------------------------------------------------------------------


class TestHydrographsMultiGWFallback:
    """Tests for the hydrographs-multi endpoint using the head loader fallback."""

    def test_multi_gw_via_head_loader_fallback(self) -> None:
        state = _make_state_with_head_loader(n_frames=3, n_nodes=4, n_layers=1)
        # Set up physical locations pointing to real grid nodes
        phys_locs = [
            {
                "name": "Well A",
                "node_id": 1,
                "columns": [(0, 1)],
                "loc": MagicMock(x=0.0, y=0.0),
            },
            {
                "name": "Well B",
                "node_id": 2,
                "columns": [(1, 1)],
                "loc": MagicMock(x=100.0, y=0.0),
            },
        ]
        state.get_gw_physical_locations = MagicMock(return_value=phys_locs)
        # Disable cache and reader so it falls back to head loader
        state.get_cache_loader = MagicMock(return_value=None)
        state.get_gw_hydrograph_reader = MagicMock(return_value=None)

        app = create_app()
        with patch(RESULTS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/results/hydrographs-multi?type=gw&ids=1,2")
            assert resp.status_code == 200
            data = resp.json()
            assert data["type"] == "gw"
            assert data["n_series"] == 2
            assert data["series"][0]["name"] == "Well A"
            assert data["series"][1]["name"] == "Well B"
            # Each series should have values for all 3 frames
            assert len(data["series"][0]["values"]) == 3

    def test_multi_gw_invalid_ids_format(self) -> None:
        state = ModelState()
        state._model = MagicMock()
        state._model.metadata = {}
        app = create_app()
        with patch(RESULTS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/results/hydrographs-multi?type=gw&ids=abc")
            assert resp.status_code == 400
            assert "Invalid IDs" in resp.json()["detail"]

    def test_multi_unknown_type(self) -> None:
        state = ModelState()
        state._model = MagicMock()
        state._model.metadata = {}
        app = create_app()
        with patch(RESULTS_PATCH, state):
            client = TestClient(app)
            resp = client.get(
                "/api/results/hydrographs-multi?type=bogus&ids=1"
            )
            assert resp.status_code == 400

    def test_multi_stream_type(self) -> None:
        state = ModelState()
        model = MagicMock()
        model.name = "TestModel"
        model.metadata = {"stream_hydrograph_specs": [{"node_id": 10, "name": "S10"}]}
        state._model = model

        reader = MagicMock()
        reader.n_timesteps = 2
        reader.n_columns = 5
        reader.find_column_by_node_id = MagicMock(return_value=0)
        reader.hydrograph_ids = [10]
        reader.get_time_series = MagicMock(
            return_value=(["2020-01-01", "2020-02-01"], [5.5, 6.6])
        )
        state._stream_hydrograph_reader = reader

        app = create_app()
        with patch(RESULTS_PATCH, state):
            client = TestClient(app)
            resp = client.get(
                "/api/results/hydrographs-multi?type=stream&ids=10"
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["type"] == "stream"
            assert data["n_series"] == 1
            assert data["series"][0]["name"] == "S10"

    def test_multi_gw_no_reader_raises_404(self) -> None:
        state = ModelState()
        state._model = MagicMock()
        state._model.metadata = {}
        state._model.groundwater = None
        state.get_gw_physical_locations = MagicMock(return_value=[])
        state.get_cache_loader = MagicMock(return_value=None)
        state._gw_hydrograph_reader = None

        app = create_app()
        with patch(RESULTS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/results/hydrographs-multi?type=gw&ids=1")
            assert resp.status_code == 404
