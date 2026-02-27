"""Sweep tests for webapi results routes — targeting remaining uncovered paths.

Covers:
- get_head_range cache hit path (lines 152-154)
- GW hydrograph cache + HDF5 fallback (lines 224-228, 255-279)
- GW hydrograph name extraction fallback from model.groundwater (lines 286-290)
- Tile drain hydrograph full branch with spec matching (lines 435-463)
- gw_hydrograph_all_layers cache path (lines 512-544)
- gw_hydrograph_all_layers reader path (lines 561-568)
- subsidence_all_layers full function (lines 639-721)
- hydrographs_multi cache path (lines 778-793)
- hydrographs_multi HEAD fallback and name extraction (lines 812-844)
- heads_by_element cache hit (lines 1008-1010)
- head_statistics frame sampling (line 1105)
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("fastapi", reason="FastAPI not available")
pytest.importorskip("pydantic", reason="Pydantic not available")

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


def _make_state(
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


# ---------------------------------------------------------------------------
# 1. get_head_range — cache hit path (lines 152-154)
# ---------------------------------------------------------------------------


class TestHeadRangeCacheHit:
    """Tests for /api/results/head-range when the cache returns data."""

    def test_cache_hit_returns_cached_percentiles(self) -> None:
        state = _make_state(n_frames=10)
        # Mock get_cached_head_range to return data
        state.get_cached_head_range = MagicMock(
            return_value={"percentile_02": -15.5, "percentile_98": 250.7}
        )

        app = create_app()
        with patch(RESULTS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/results/head-range?layer=1")
            assert resp.status_code == 200
            data = resp.json()
            assert data["min"] == -15.5
            assert data["max"] == 250.7
            assert data["n_timesteps"] == 10
            assert data["n_frames_scanned"] == 10
            state.get_cached_head_range.assert_called_once_with(1)

    def test_cache_miss_falls_through_to_loader(self) -> None:
        state = _make_state()
        state.get_cached_head_range = MagicMock(return_value=None)

        app = create_app()
        with patch(RESULTS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/results/head-range?layer=1&max_frames=3")
            assert resp.status_code == 200
            data = resp.json()
            assert data["min"] == 10.0
            assert data["max"] == 100.0


# ---------------------------------------------------------------------------
# 2. GW hydrograph — cache hit path (lines 224-228)
# ---------------------------------------------------------------------------


class TestGWHydrographCacheHit:
    """Tests for /api/results/hydrograph?type=gw with SQLite cache."""

    def test_gw_hydrograph_cache_hit(self) -> None:
        state = ModelState()
        model = MagicMock()
        model.name = "TestModel"
        model.grid = _make_grid()
        model.metadata = {}
        model.groundwater = None
        state._model = model

        phys_locs = [
            {
                "name": "Well Alpha",
                "node_id": 1,
                "columns": [(0, 1)],
                "loc": MagicMock(x=0.0, y=0.0),
            },
        ]
        state.get_gw_physical_locations = MagicMock(return_value=phys_locs)

        mock_cache = MagicMock()
        mock_cache.get_hydrograph.return_value = (
            ["2020-01-01", "2020-02-01"],
            np.array([55.0, 56.5]),
        )
        state.get_cache_loader = MagicMock(return_value=mock_cache)

        app = create_app()
        with patch(RESULTS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/results/hydrograph?type=gw&location_id=1")
            assert resp.status_code == 200
            data = resp.json()
            assert data["name"] == "Well Alpha"
            assert data["type"] == "gw"
            assert data["times"] == ["2020-01-01", "2020-02-01"]
            assert data["values"] == [55.0, 56.5]
            mock_cache.get_hydrograph.assert_called_once_with("gw", 0)


# ---------------------------------------------------------------------------
# 3. GW hydrograph — HDF5 fallback (lines 255-279)
# ---------------------------------------------------------------------------


class TestGWHydrographHDF5Fallback:
    """Tests for GW hydrograph using head HDF5 data when reader unavailable."""

    def test_gw_hydrograph_hdf5_fallback(self) -> None:
        state = _make_state(n_frames=3, n_nodes=4, n_layers=2)

        phys_locs = [
            {
                "name": "Well Beta",
                "node_id": 1,
                "columns": [(0, 1)],
                "loc": MagicMock(x=0.0, y=0.0),
            },
        ]
        state.get_gw_physical_locations = MagicMock(return_value=phys_locs)
        state.get_cache_loader = MagicMock(return_value=None)
        state.get_gw_hydrograph_reader = MagicMock(return_value=None)

        app = create_app()
        with patch(RESULTS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/results/hydrograph?type=gw&location_id=1")
            assert resp.status_code == 200
            data = resp.json()
            assert data["name"] == "Well Beta"
            assert data["type"] == "gw"
            assert data["layer"] == 1
            assert len(data["times"]) == 3
            assert len(data["values"]) == 3
            # Values should be numeric (not None since our mock doesn't produce < -9000)
            for v in data["values"]:
                assert v is not None


# ---------------------------------------------------------------------------
# 4. GW hydrograph — name extraction fallback (lines 286-290)
# ---------------------------------------------------------------------------


class TestGWHydrographNameFallback:
    """Tests for GW hydrograph name from model.groundwater when no phys_locs."""

    def test_gw_hydrograph_name_from_groundwater_component(self) -> None:
        state = ModelState()
        model = MagicMock()
        model.name = "TestModel"
        model.grid = _make_grid()
        model.metadata = {}

        # Create a groundwater component with hydrograph_locations
        loc_obj = MagicMock()
        loc_obj.name = "Monitoring Well 42"
        loc_obj.layer = 3
        gw = MagicMock()
        gw.hydrograph_locations = [loc_obj]
        model.groundwater = gw
        state._model = model

        # Return empty phys_locs to trigger fallback path
        state.get_gw_physical_locations = MagicMock(return_value=[])

        reader = MagicMock()
        reader.n_timesteps = 2
        reader.n_columns = 1
        reader.get_time_series = MagicMock(
            return_value=(["2020-01-01", "2020-02-01"], [100.0, 101.0])
        )
        state._gw_hydrograph_reader = reader

        app = create_app()
        with patch(RESULTS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/results/hydrograph?type=gw&location_id=1")
            assert resp.status_code == 200
            data = resp.json()
            assert data["name"] == "Monitoring Well 42"
            assert data["layer"] == 3
            assert data["values"] == [100.0, 101.0]


# ---------------------------------------------------------------------------
# 5. Tile drain hydrograph — spec matching + fallback (lines 435-463)
# ---------------------------------------------------------------------------


class TestTileDrainHydrographSpecMatch:
    """Tests for tile drain hydrograph with spec ID matching."""

    def test_tile_drain_spec_id_match(self) -> None:
        state = ModelState()
        model = MagicMock()
        model.name = "TestModel"
        model.grid = _make_grid()
        model.metadata = {}
        gw = MagicMock()
        gw.td_hydro_specs = [
            {"id": 10, "name": "Drain A"},
            {"id": 20, "name": "Drain B"},
        ]
        model.groundwater = gw
        state._model = model

        reader = MagicMock()
        reader.n_timesteps = 3
        reader.n_columns = 2
        reader.get_time_series = MagicMock(
            return_value=(["2020-01-01", "2020-02-01", "2020-03-01"], [5.0, 6.0, 7.0])
        )
        state._tile_drain_reader = reader

        app = create_app()
        with patch(RESULTS_PATCH, state):
            client = TestClient(app)
            # Match by spec ID = 20 -> col_idx = 1
            resp = client.get("/api/results/hydrograph?type=tile_drain&location_id=20")
            assert resp.status_code == 200
            data = resp.json()
            assert data["name"] == "Drain B"
            assert data["type"] == "tile_drain"
            assert data["values"] == [5.0, 6.0, 7.0]

    def test_tile_drain_fallback_column_index(self) -> None:
        """When spec ID not found, fallback to 1-based column index."""
        state = ModelState()
        model = MagicMock()
        model.name = "TestModel"
        model.grid = _make_grid()
        model.metadata = {}
        gw = MagicMock()
        gw.td_hydro_specs = []  # Empty specs -> no match by ID
        model.groundwater = gw
        state._model = model

        reader = MagicMock()
        reader.n_timesteps = 2
        reader.n_columns = 3
        reader.get_time_series = MagicMock(return_value=(["2020-01-01", "2020-02-01"], [1.0, 2.0]))
        state._tile_drain_reader = reader

        app = create_app()
        with patch(RESULTS_PATCH, state):
            client = TestClient(app)
            # location_id=2 -> candidate column = 1
            resp = client.get("/api/results/hydrograph?type=tile_drain&location_id=2")
            assert resp.status_code == 200
            data = resp.json()
            assert data["name"] == "Tile Drain 2"

    def test_tile_drain_not_found(self) -> None:
        """When spec ID not found and column out of range, return 404."""
        state = ModelState()
        model = MagicMock()
        model.name = "TestModel"
        model.grid = _make_grid()
        model.metadata = {}
        gw = MagicMock()
        gw.td_hydro_specs = []
        model.groundwater = gw
        state._model = model

        reader = MagicMock()
        reader.n_timesteps = 2
        reader.n_columns = 1
        state._tile_drain_reader = reader

        app = create_app()
        with patch(RESULTS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/results/hydrograph?type=tile_drain&location_id=99")
            assert resp.status_code == 404
            assert "not found" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# 6. gw_hydrograph_all_layers — cache path (lines 512-544)
# ---------------------------------------------------------------------------


class TestGWHydrographAllLayersCache:
    """Tests for /api/results/gw-hydrograph-all-layers with cache."""

    def test_all_layers_cache_path(self) -> None:
        state = ModelState()
        model = MagicMock()
        model.name = "TestModel"
        model.grid = _make_grid()
        model.metadata = {}
        model.groundwater = MagicMock()
        model.groundwater.hydrograph_locations = []
        state._model = model

        phys_locs = [
            {
                "name": "Well Cache",
                "node_id": 1,
                "columns": [(0, 1), (1, 2)],
                "loc": MagicMock(x=0.0, y=0.0),
            },
        ]
        state.get_gw_physical_locations = MagicMock(return_value=phys_locs)

        mock_cache = MagicMock()
        # Return data for both layers from cache
        call_count = [0]

        def _get_hydro(htype: str, col_idx: int):
            call_count[0] += 1
            return (
                ["2020-01-01", "2020-02-01"],
                np.array([10.0 + col_idx, 11.0 + col_idx]),
            )

        mock_cache.get_hydrograph = MagicMock(side_effect=_get_hydro)
        state.get_cache_loader = MagicMock(return_value=mock_cache)

        app = create_app()
        with patch(RESULTS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/results/gw-hydrograph-all-layers?location_id=1")
            assert resp.status_code == 200
            data = resp.json()
            assert data["name"] == "Well Cache"
            assert data["n_layers"] == 2
            assert data["times"] == ["2020-01-01", "2020-02-01"]
            assert len(data["layers"]) == 2
            assert data["layers"][0]["layer"] == 1
            assert data["layers"][1]["layer"] == 2


# ---------------------------------------------------------------------------
# 7. gw_hydrograph_all_layers — reader path (lines 561-568)
# ---------------------------------------------------------------------------


class TestGWHydrographAllLayersReader:
    """Tests for /api/results/gw-hydrograph-all-layers from GW reader."""

    def test_all_layers_reader_path(self) -> None:
        state = ModelState()
        model = MagicMock()
        model.name = "TestModel"
        model.grid = _make_grid()
        model.metadata = {}
        model.groundwater = MagicMock()
        model.groundwater.hydrograph_locations = []
        state._model = model

        phys_locs = [
            {
                "name": "Well Reader",
                "node_id": 1,
                "columns": [(0, 1), (1, 2)],
                "loc": MagicMock(x=0.0, y=0.0),
            },
        ]
        state.get_gw_physical_locations = MagicMock(return_value=phys_locs)
        state.get_cache_loader = MagicMock(return_value=None)

        reader = MagicMock()
        reader.n_timesteps = 2
        reader.n_columns = 2

        def _get_ts(col_idx: int):
            return (
                ["2020-01-01", "2020-02-01"],
                [20.0 + col_idx, 21.0 + col_idx],
            )

        reader.get_time_series = MagicMock(side_effect=_get_ts)
        state._gw_hydrograph_reader = reader

        app = create_app()
        with patch(RESULTS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/results/gw-hydrograph-all-layers?location_id=1")
            assert resp.status_code == 200
            data = resp.json()
            assert data["name"] == "Well Reader"
            assert data["n_layers"] == 2
            assert len(data["layers"]) == 2
            assert data["layers"][0]["layer"] == 1
            assert data["layers"][1]["layer"] == 2


# ---------------------------------------------------------------------------
# 8. subsidence_all_layers — full function (lines 639-721)
# ---------------------------------------------------------------------------


class TestSubsidenceAllLayers:
    """Tests for /api/results/subsidence-all-layers."""

    def _make_subsidence_state(self) -> ModelState:
        state = ModelState()
        model = MagicMock()
        model.name = "TestModel"
        model.grid = _make_grid()
        model.metadata = {}

        # Create subsidence specs at the same physical location (node 1), two layers
        spec1 = MagicMock()
        spec1.id = 1
        spec1.name = "Subs Obs 1"
        spec1.layer = 1
        spec1.node_id = 1
        spec1.gw_node = 0
        spec1.x = 0.0
        spec1.y = 0.0

        spec2 = MagicMock()
        spec2.id = 2
        spec2.name = "Subs Obs 2"
        spec2.layer = 2
        spec2.node_id = 1
        spec2.gw_node = 0
        spec2.x = 0.0
        spec2.y = 0.0

        subs_config = MagicMock()
        subs_config.hydrograph_specs = [spec1, spec2]

        gw = MagicMock()
        gw.subsidence_config = subs_config
        model.groundwater = gw
        state._model = model

        reader = MagicMock()
        reader.n_timesteps = 3
        reader.n_columns = 2

        def _get_ts(col_idx: int):
            return (
                ["2020-01-01", "2020-02-01", "2020-03-01"],
                [0.01 * (col_idx + 1), 0.02 * (col_idx + 1), 0.03 * (col_idx + 1)],
            )

        reader.get_time_series = MagicMock(side_effect=_get_ts)
        state._subsidence_reader = reader

        return state

    def test_subsidence_all_layers_by_spec_id(self) -> None:
        state = self._make_subsidence_state()
        app = create_app()
        with patch(RESULTS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/results/subsidence-all-layers?location_id=1")
            assert resp.status_code == 200
            data = resp.json()
            assert data["name"] == "Subs Obs 1"
            assert data["node_id"] == 1
            assert data["n_layers"] == 2
            assert len(data["layers"]) == 2
            assert data["layers"][0]["layer"] == 1
            assert data["layers"][1]["layer"] == 2
            assert len(data["times"]) == 3

    def test_subsidence_all_layers_fallback_index(self) -> None:
        """When spec.id doesn't match location_id, fall back to 1-based index."""
        state = self._make_subsidence_state()
        # Change spec IDs so they don't match location_id=1
        specs = state._model.groundwater.subsidence_config.hydrograph_specs
        specs[0].id = 100
        specs[1].id = 200

        app = create_app()
        with patch(RESULTS_PATCH, state):
            client = TestClient(app)
            # location_id=1 -> candidate=0, specs[0] matches
            resp = client.get("/api/results/subsidence-all-layers?location_id=1")
            assert resp.status_code == 200
            data = resp.json()
            assert data["n_layers"] == 2  # Both specs at same node

    def test_subsidence_all_layers_no_specs(self) -> None:
        state = ModelState()
        model = MagicMock()
        model.grid = _make_grid()
        model.metadata = {}
        gw = MagicMock()
        gw.subsidence_config = MagicMock()
        gw.subsidence_config.hydrograph_specs = []
        model.groundwater = gw
        state._model = model

        app = create_app()
        with patch(RESULTS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/results/subsidence-all-layers?location_id=1")
            assert resp.status_code == 404
            assert "No subsidence hydrograph specs" in resp.json()["detail"]

    def test_subsidence_all_layers_no_reader(self) -> None:
        state = ModelState()
        model = MagicMock()
        model.grid = _make_grid()
        model.metadata = {}
        gw = MagicMock()
        spec = MagicMock()
        spec.id = 1
        spec.name = "S1"
        spec.layer = 1
        gw.subsidence_config = MagicMock()
        gw.subsidence_config.hydrograph_specs = [spec]
        model.groundwater = gw
        state._model = model
        state._subsidence_reader = None

        app = create_app()
        with patch(RESULTS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/results/subsidence-all-layers?location_id=1")
            assert resp.status_code == 404
            assert "No subsidence hydrograph data" in resp.json()["detail"]

    def test_subsidence_all_layers_no_gw_component(self) -> None:
        state = ModelState()
        model = MagicMock()
        model.grid = _make_grid()
        model.metadata = {}
        model.groundwater = None
        state._model = model

        app = create_app()
        with patch(RESULTS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/results/subsidence-all-layers?location_id=1")
            assert resp.status_code == 404
            assert "No groundwater component" in resp.json()["detail"]

    def test_subsidence_all_layers_coord_matching(self) -> None:
        """Test matching specs by coordinate proximity when node_id=0."""
        state = ModelState()
        model = MagicMock()
        model.grid = _make_grid()
        model.metadata = {}

        spec1 = MagicMock()
        spec1.id = 1
        spec1.name = "Coord Spec L1"
        spec1.layer = 1
        spec1.node_id = 0
        spec1.gw_node = 0
        spec1.x = 50.0
        spec1.y = 50.0

        spec2 = MagicMock()
        spec2.id = 2
        spec2.name = "Coord Spec L2"
        spec2.layer = 2
        spec2.node_id = 0
        spec2.gw_node = 0
        spec2.x = 50.3  # Within 1.0 of spec1
        spec2.y = 50.4

        subs_config = MagicMock()
        subs_config.hydrograph_specs = [spec1, spec2]
        gw = MagicMock()
        gw.subsidence_config = subs_config
        model.groundwater = gw
        state._model = model

        reader = MagicMock()
        reader.n_timesteps = 2
        reader.n_columns = 2
        reader.get_time_series = MagicMock(
            return_value=(["2020-01-01", "2020-02-01"], [0.01, 0.02])
        )
        state._subsidence_reader = reader

        app = create_app()
        with patch(RESULTS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/results/subsidence-all-layers?location_id=1")
            assert resp.status_code == 200
            data = resp.json()
            # Both specs should be matched by coordinate proximity
            assert data["n_layers"] == 2


# ---------------------------------------------------------------------------
# 9. hydrographs_multi — cache path (lines 778-793)
# ---------------------------------------------------------------------------


class TestHydrographsMultiCachePath:
    """Tests for /api/results/hydrographs-multi with SQLite cache."""

    def test_multi_gw_cache_hit(self) -> None:
        state = ModelState()
        model = MagicMock()
        model.name = "TestModel"
        model.grid = _make_grid()
        model.metadata = {}
        model.groundwater = None
        state._model = model

        phys_locs = [
            {
                "name": "Well C1",
                "node_id": 1,
                "columns": [(0, 1)],
                "loc": MagicMock(x=0.0, y=0.0),
            },
            {
                "name": "Well C2",
                "node_id": 2,
                "columns": [(1, 1)],
                "loc": MagicMock(x=100.0, y=0.0),
            },
        ]
        state.get_gw_physical_locations = MagicMock(return_value=phys_locs)

        mock_cache = MagicMock()

        def _get_hydro(htype: str, col_idx: int):
            return (
                ["2020-01-01", "2020-02-01"],
                np.array([30.0 + col_idx, 31.0 + col_idx]),
            )

        mock_cache.get_hydrograph = MagicMock(side_effect=_get_hydro)
        state.get_cache_loader = MagicMock(return_value=mock_cache)
        state.get_gw_hydrograph_reader = MagicMock(return_value=None)
        state.get_head_loader = MagicMock(return_value=None)

        app = create_app()
        with patch(RESULTS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/results/hydrographs-multi?type=gw&ids=1,2")
            assert resp.status_code == 200
            data = resp.json()
            assert data["n_series"] == 2
            assert data["series"][0]["name"] == "Well C1"
            assert data["series"][1]["name"] == "Well C2"
            assert data["series"][0]["values"] == [30.0, 31.0]
            assert data["series"][1]["values"] == [31.0, 32.0]


# ---------------------------------------------------------------------------
# 10. hydrographs_multi — HEAD fallback + name extraction (lines 812-844)
# ---------------------------------------------------------------------------


class TestHydrographsMultiHeadFallback:
    """Tests for hydrographs-multi using head loader fallback and name extraction."""

    def test_multi_gw_name_from_groundwater_locs(self) -> None:
        """No phys_locs, name extracted from model.groundwater.hydrograph_locations."""
        state = ModelState()
        model = MagicMock()
        model.name = "TestModel"
        model.grid = _make_grid()
        model.metadata = {}

        loc = MagicMock()
        loc.name = "Custom GW Well"
        loc.layer = 2
        gw = MagicMock()
        gw.hydrograph_locations = [loc]
        model.groundwater = gw
        state._model = model

        # Return empty phys_locs -> triggers fallback code
        state.get_gw_physical_locations = MagicMock(return_value=[])

        reader = MagicMock()
        reader.n_timesteps = 2
        reader.n_columns = 1
        reader.get_time_series = MagicMock(
            return_value=(["2020-01-01", "2020-02-01"], [50.0, 51.0])
        )
        state.get_cache_loader = MagicMock(return_value=None)
        state._gw_hydrograph_reader = reader
        state.get_head_loader = MagicMock(return_value=None)

        app = create_app()
        with patch(RESULTS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/results/hydrographs-multi?type=gw&ids=1")
            assert resp.status_code == 200
            data = resp.json()
            assert data["n_series"] == 1
            assert data["series"][0]["name"] == "Custom GW Well"
            assert data["series"][0]["layer"] == 2


# ---------------------------------------------------------------------------
# 11. heads_by_element — cache hit with datetime (lines 1008-1010)
# ---------------------------------------------------------------------------


class TestHeadsByElementCacheDateTime:
    """Tests for heads-by-element cache hit with valid datetime."""

    def test_cache_hit_with_datetime(self) -> None:
        state = _make_state(n_frames=5)
        cached_values = [42.0, None]
        state.get_cached_head_by_element = MagicMock(return_value=(cached_values, 10.0, 90.0))

        app = create_app()
        with patch(RESULTS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/results/heads-by-element?timestep=2&layer=1")
            assert resp.status_code == 200
            data = resp.json()
            assert data["values"] == cached_values
            assert data["datetime"] is not None
            # The datetime should come from loader.times[2]
            assert "2020-01-03" in data["datetime"]

    def test_cache_hit_without_loader(self) -> None:
        """Cache hit but no head loader (dt should be None)."""
        state = ModelState()
        model = MagicMock()
        model.name = "TestModel"
        model.grid = _make_grid()
        model.metadata = {}
        model.groundwater = None
        state._model = model

        state.get_cached_head_by_element = MagicMock(return_value=([50.0], 50.0, 50.0))
        state._head_loader = None

        app = create_app()
        with patch(RESULTS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/results/heads-by-element?timestep=0&layer=1")
            assert resp.status_code == 200
            data = resp.json()
            assert data["datetime"] is None


# ---------------------------------------------------------------------------
# 12. head_statistics — frame sampling + zero frames (line 1105)
# ---------------------------------------------------------------------------


class TestHeadStatisticsFrameSampling:
    """Tests for frame sampling and edge cases in statistics."""

    def test_statistics_zero_frames(self) -> None:
        """n_frames=0 should return 404."""
        state = ModelState()
        model = MagicMock()
        model.name = "TestModel"
        model.grid = _make_grid()
        model.metadata = {}
        state._model = model

        loader = MagicMock()
        loader.n_frames = 0
        loader.shape = (4, 1)
        loader.times = []
        state._head_loader = loader

        app = create_app()
        with patch(RESULTS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/results/statistics?layer=1")
            assert resp.status_code == 404
            assert "No timesteps available" in resp.json()["detail"]

    def test_statistics_samples_frames_uniformly(self) -> None:
        """With max_frames < n_frames, should sample uniformly."""
        state = _make_state(n_frames=20, n_nodes=4, n_layers=1)
        app = create_app()
        with patch(RESULTS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/results/statistics?layer=1&max_frames=5")
            assert resp.status_code == 200
            data = resp.json()
            assert data["n_frames_sampled"] == 5
            assert data["n_total_frames"] == 20
