"""Comprehensive tests for model API routes.

Covers /api/model/info, /api/model/bounds, and /api/model/summary
with all branches for each component section (mesh, groundwater,
streams, lakes, rootzone, small_watersheds, unsaturated_zone,
available_results).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi", reason="FastAPI not available")
pydantic = pytest.importorskip("pydantic", reason="Pydantic not available")

from fastapi.testclient import TestClient

from pyiwfm.core.mesh import AppGrid, Element, Node
from pyiwfm.visualization.webapi.config import ModelState, model_state
from pyiwfm.visualization.webapi.server import create_app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
        "get_budget_reader", "get_available_budgets", "reproject_coords",
        "get_stream_reach_boundaries", "get_head_loader",
        "get_gw_hydrograph_reader", "get_stream_hydrograph_reader",
        "get_area_manager", "get_subsidence_reader",
    ):
        if attr in model_state.__dict__:
            del model_state.__dict__[attr]


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
    """Create a minimal mock IWFMModel with no components by default."""
    model = MagicMock()
    model.name = kwargs.get("name", "TestModel")
    model.grid = kwargs.get("grid", _make_grid())
    model.metadata = kwargs.get("metadata", {})
    model.n_nodes = kwargs.get("n_nodes", 4)
    model.n_elements = kwargs.get("n_elements", 1)
    model.n_layers = kwargs.get("n_layers", 2)
    model.has_streams = kwargs.get("has_streams", False)
    model.has_lakes = kwargs.get("has_lakes", False)
    model.n_stream_nodes = kwargs.get("n_stream_nodes", 0)
    model.n_lakes = kwargs.get("n_lakes", 0)
    model.stratigraphy = kwargs.get("stratigraphy", None)
    model.groundwater = kwargs.get("groundwater", None)
    model.streams = kwargs.get("streams", None)
    model.lakes = kwargs.get("lakes", None)
    model.rootzone = kwargs.get("rootzone", None)
    model.small_watersheds = kwargs.get("small_watersheds", None)
    model.unsaturated_zone = kwargs.get("unsaturated_zone", None)
    model.source_files = kwargs.get("source_files", {})
    return model


def _patch_results_none():
    """Return a context manager that patches all lazy result getters to None."""
    return (
        patch.object(model_state, "get_head_loader", return_value=None),
        patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
        patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
        patch.object(model_state, "get_available_budgets", return_value=[]),
        patch.object(model_state, "get_area_manager", return_value=None),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _cleanup():
    """Auto-cleanup model state before and after every test."""
    _reset_model_state()
    yield
    _reset_model_state()


@pytest.fixture()
def client_no_model():
    """TestClient with no model loaded."""
    app = create_app()
    return TestClient(app)


@pytest.fixture()
def client_basic():
    """TestClient with a bare-minimum model loaded (no components)."""
    model = _make_mock_model()
    # Remove optional component attributes entirely so hasattr returns False
    del model.groundwater
    del model.streams
    del model.lakes
    del model.rootzone
    del model.small_watersheds
    del model.unsaturated_zone
    model_state._model = model
    app = create_app()
    return TestClient(app)


# ===========================================================================
# GET /api/model/info
# ===========================================================================


class TestModelInfoEndpoint:
    """Tests for GET /api/model/info covering all branches."""

    def test_no_model_returns_404(self, client_no_model):
        resp = client_no_model.get("/api/model/info")
        assert resp.status_code == 404
        assert "No model loaded" in resp.json()["detail"]

    def test_basic_model_info(self, client_basic):
        resp = client_basic.get("/api/model/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "TestModel"
        assert data["n_nodes"] == 4
        assert data["n_elements"] == 1
        assert data["n_layers"] == 2
        assert data["has_streams"] is False
        assert data["has_lakes"] is False
        assert data["n_stream_nodes"] is None
        assert data["n_lakes"] is None

    def test_model_with_streams(self):
        """When has_streams=True, n_stream_nodes is populated."""
        model = _make_mock_model(has_streams=True, n_stream_nodes=10)
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/model/info")
        data = resp.json()
        assert data["has_streams"] is True
        assert data["n_stream_nodes"] == 10

    def test_model_with_lakes(self):
        """When has_lakes=True, n_lakes is populated."""
        model = _make_mock_model(has_lakes=True, n_lakes=3)
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/model/info")
        data = resp.json()
        assert data["has_lakes"] is True
        assert data["n_lakes"] == 3

    def test_model_with_streams_and_lakes(self):
        """Both streams and lakes can be present simultaneously."""
        model = _make_mock_model(
            has_streams=True, n_stream_nodes=5,
            has_lakes=True, n_lakes=2,
        )
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/model/info")
        data = resp.json()
        assert data["has_streams"] is True
        assert data["n_stream_nodes"] == 5
        assert data["has_lakes"] is True
        assert data["n_lakes"] == 2


# ===========================================================================
# GET /api/model/bounds
# ===========================================================================


class TestModelBoundsEndpoint:
    """Tests for GET /api/model/bounds covering all branches."""

    def test_no_model_returns_404(self, client_no_model):
        resp = client_no_model.get("/api/model/bounds")
        assert resp.status_code == 404
        assert "No model loaded" in resp.json()["detail"]

    def test_bounds_returned(self):
        """Bounds tuple is unpacked into the BoundsInfo fields."""
        model = _make_mock_model()
        model_state._model = model
        # Pre-compute bounds to avoid real computation
        model_state._bounds = (10.0, 200.0, 30.0, 400.0, -50.0, 150.0)
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/model/bounds")
        assert resp.status_code == 200
        data = resp.json()
        assert data["xmin"] == 10.0
        assert data["xmax"] == 200.0
        assert data["ymin"] == 30.0
        assert data["ymax"] == 400.0
        assert data["zmin"] == -50.0
        assert data["zmax"] == 150.0


# ===========================================================================
# GET /api/model/summary - Mesh section
# ===========================================================================


class TestSummaryMesh:
    """Tests for the mesh section of /api/model/summary."""

    def test_grid_from_model_grid_attr(self):
        """When model.grid is set, subregion counting uses it."""
        model = _make_mock_model()
        del model.groundwater
        del model.streams
        del model.lakes
        del model.rootzone
        del model.small_watersheds
        del model.unsaturated_zone
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            assert resp.status_code == 200
            data = resp.json()
            assert data["mesh"]["n_nodes"] == 4
            assert data["mesh"]["n_elements"] == 1
            assert data["mesh"]["n_layers"] == 2
            # Grid has 1 element with subregion=1, so n_subregions == 1
            assert data["mesh"]["n_subregions"] == 1

    def test_grid_from_model_mesh_attr(self):
        """When model.grid is None but model.mesh is set, uses mesh attr."""
        model = _make_mock_model()
        grid = _make_grid()
        model.grid = None
        model.mesh = grid
        del model.groundwater
        del model.streams
        del model.lakes
        del model.rootzone
        del model.small_watersheds
        del model.unsaturated_zone
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            assert resp.status_code == 200
            data = resp.json()
            assert data["mesh"]["n_subregions"] == 1

    def test_no_grid_no_mesh(self):
        """When both model.grid and model.mesh are None, n_subregions is None."""
        model = _make_mock_model()
        model.grid = None
        model.mesh = None
        del model.groundwater
        del model.streams
        del model.lakes
        del model.rootzone
        del model.small_watersheds
        del model.unsaturated_zone
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            assert resp.status_code == 200
            data = resp.json()
            assert data["mesh"]["n_subregions"] is None

    def test_grid_n_subregions_nonzero(self):
        """When grid.n_subregions > 0, uses it directly."""
        from pyiwfm.core.mesh import Subregion

        grid = _make_grid()
        grid.subregions = {1: Subregion(id=1, name="SR1")}

        model = _make_mock_model(grid=grid)
        del model.groundwater
        del model.streams
        del model.lakes
        del model.rootzone
        del model.small_watersheds
        del model.unsaturated_zone
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            assert resp.status_code == 200
            data = resp.json()
            assert data["mesh"]["n_subregions"] == 1

    def test_subregion_fallback_iter_elements_exception(self):
        """When iter_elements raises, n_subregions stays at 0 -> None."""
        grid = MagicMock()
        grid.n_subregions = 0
        grid.n_elements = 5

        def raise_error():
            raise RuntimeError("iter failed")

        grid.iter_elements = raise_error

        model = _make_mock_model(grid=grid)
        del model.groundwater
        del model.streams
        del model.lakes
        del model.rootzone
        del model.small_watersheds
        del model.unsaturated_zone
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            assert resp.status_code == 200
            data = resp.json()
            assert data["mesh"]["n_subregions"] is None

    def test_subregion_zero_elements_no_fallback(self):
        """When grid.n_elements == 0, the element iteration is skipped."""
        grid = MagicMock()
        grid.n_subregions = 0
        grid.n_elements = 0

        model = _make_mock_model(grid=grid)
        del model.groundwater
        del model.streams
        del model.lakes
        del model.rootzone
        del model.small_watersheds
        del model.unsaturated_zone
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            assert resp.status_code == 200
            data = resp.json()
            assert data["mesh"]["n_subregions"] is None

    def test_subregion_elements_all_zero(self):
        """Elements with subregion == 0 are excluded from unique count."""
        grid = MagicMock()
        grid.n_subregions = 0
        grid.n_elements = 2

        elem1 = MagicMock()
        elem1.subregion = 0
        elem2 = MagicMock()
        elem2.subregion = 0
        grid.iter_elements = lambda: iter([elem1, elem2])

        model = _make_mock_model(grid=grid)
        del model.groundwater
        del model.streams
        del model.lakes
        del model.rootzone
        del model.small_watersheds
        del model.unsaturated_zone
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            assert resp.status_code == 200
            data = resp.json()
            # unique_sr is empty set -> n_subregions stays 0 -> None
            assert data["mesh"]["n_subregions"] is None


# ===========================================================================
# GET /api/model/summary - Groundwater section
# ===========================================================================


class TestSummaryGroundwater:
    """Tests for the groundwater section of /api/model/summary."""

    def _make_summary_model(self, **kwargs):
        """Helper to create a model and return a client for summary tests."""
        model = _make_mock_model(**kwargs)
        # Remove attrs we don't need to ensure hasattr returns False
        for attr in ("streams", "lakes", "rootzone", "small_watersheds", "unsaturated_zone"):
            if not kwargs.get(attr):
                try:
                    delattr(model, attr)
                except AttributeError:
                    pass
        model_state._model = model
        app = create_app()
        return TestClient(app)

    def test_no_groundwater(self):
        """When model has no groundwater attr, loaded=False."""
        client = self._make_summary_model()
        delattr(model_state._model, "groundwater")

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            data = resp.json()
            assert data["groundwater"]["loaded"] is False

    def test_groundwater_with_nonzero_bc(self):
        """When gw.n_boundary_conditions > 0, uses it directly (else branch)."""
        gw = MagicMock()
        gw.n_boundary_conditions = 15
        gw.n_tile_drains = 3
        gw.n_wells = 10
        gw.n_hydrograph_locations = 5
        gw.aquifer_params = MagicMock()  # non-None -> has_aquifer_params=True

        client = self._make_summary_model(groundwater=gw)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            data = resp.json()
            gw_data = data["groundwater"]
            assert gw_data["loaded"] is True
            assert gw_data["n_wells"] == 10
            assert gw_data["n_hydrograph_locations"] == 5
            assert gw_data["n_boundary_conditions"] == 15
            assert gw_data["n_tile_drains"] == 3
            assert gw_data["has_aquifer_params"] is True

    def test_groundwater_bc_zero_metadata_fallback(self):
        """When gw.n_boundary_conditions == 0, falls back to metadata sums."""
        gw = MagicMock()
        gw.n_boundary_conditions = 0
        gw.n_tile_drains = 0
        gw.n_wells = 2
        gw.n_hydrograph_locations = 1
        gw.aquifer_params = None

        metadata = {
            "gw_n_specified_flow_bc": 5,
            "gw_n_specified_head_bc": 3,
            "gw_n_general_head_bc": 2,
            "gw_n_tile_drains": 7,
        }
        client = self._make_summary_model(groundwater=gw, metadata=metadata)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            data = resp.json()
            # 5 + 3 + 2 = 10
            assert data["groundwater"]["n_boundary_conditions"] == 10
            assert data["groundwater"]["n_tile_drains"] == 7
            assert data["groundwater"]["has_aquifer_params"] is False

    def test_groundwater_bc_zero_metadata_also_zero(self):
        """When gw BCs are 0 and metadata sums also 0, n_bc is None."""
        gw = MagicMock()
        gw.n_boundary_conditions = 0
        gw.n_tile_drains = 0
        gw.n_wells = 0
        gw.n_hydrograph_locations = 0
        gw.aquifer_params = None

        # Empty metadata -> sums to 0
        client = self._make_summary_model(groundwater=gw, metadata={})

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            data = resp.json()
            assert data["groundwater"]["n_boundary_conditions"] is None
            assert data["groundwater"]["n_tile_drains"] is None

    def test_groundwater_no_metadata_attr(self):
        """When model has no metadata attr, the fallback still works."""
        gw = MagicMock()
        gw.n_boundary_conditions = 0
        gw.n_tile_drains = 0
        gw.n_wells = 0
        gw.n_hydrograph_locations = 0
        gw.aquifer_params = None

        model = _make_mock_model(groundwater=gw)
        del model.metadata  # no metadata attr
        for attr in ("streams", "lakes", "rootzone", "small_watersheds", "unsaturated_zone"):
            try:
                delattr(model, attr)
            except AttributeError:
                pass
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            data = resp.json()
            assert data["groundwater"]["loaded"] is True
            assert data["groundwater"]["n_boundary_conditions"] is None


# ===========================================================================
# GET /api/model/summary - Streams section
# ===========================================================================


class TestSummaryStreams:
    """Tests for the streams section of /api/model/summary."""

    def _summary_client(self, model):
        """Set model and return TestClient."""
        model_state._model = model
        app = create_app()
        return TestClient(app)

    def test_no_streams(self):
        """When model has no streams attr, loaded=False."""
        model = _make_mock_model()
        del model.streams
        del model.groundwater
        del model.lakes
        del model.rootzone
        del model.small_watersheds
        del model.unsaturated_zone
        client = self._summary_client(model)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            data = resp.json()
            assert data["streams"]["loaded"] is False

    def test_streams_n_reaches_nonzero(self):
        """When stm.n_reaches > 0, uses it directly."""
        stm = MagicMock()
        stm.n_nodes = 10
        stm.n_reaches = 5
        stm.n_diversions = 2
        stm.n_bypasses = 1

        model = _make_mock_model(streams=stm)
        del model.groundwater
        del model.lakes
        del model.rootzone
        del model.small_watersheds
        del model.unsaturated_zone
        client = self._summary_client(model)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            data = resp.json()
            assert data["streams"]["loaded"] is True
            assert data["streams"]["n_reaches"] == 5
            assert data["streams"]["n_diversions"] == 2
            assert data["streams"]["n_bypasses"] == 1

    def test_streams_reaches_from_reach_id(self):
        """When n_reaches == 0, counts unique reach_id from nodes."""
        stm = MagicMock()
        stm.n_nodes = 3
        stm.n_reaches = 0
        stm.n_diversions = 0
        stm.n_bypasses = 0

        n1 = MagicMock(); n1.reach_id = 1
        n2 = MagicMock(); n2.reach_id = 1
        n3 = MagicMock(); n3.reach_id = 2
        stm.nodes = {1: n1, 2: n2, 3: n3}

        model = _make_mock_model(streams=stm)
        del model.groundwater
        del model.lakes
        del model.rootzone
        del model.small_watersheds
        del model.unsaturated_zone
        client = self._summary_client(model)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            data = resp.json()
            assert data["streams"]["n_reaches"] == 2

    def test_streams_reaches_from_connectivity(self):
        """When reach_id strategy fails, falls back to connectivity heuristic."""
        stm = MagicMock()
        stm.n_nodes = 3
        stm.n_reaches = 0
        stm.n_diversions = 0
        stm.n_bypasses = 0

        # All nodes have reach_id == 0 -> reach_id strategy produces empty set
        n1 = MagicMock(); n1.id = 1; n1.reach_id = 0
        n1.downstream_node = 2; n1.gw_node = 100
        n2 = MagicMock(); n2.id = 2; n2.reach_id = 0
        n2.downstream_node = 3; n2.gw_node = 200
        n3 = MagicMock(); n3.id = 3; n3.reach_id = 0
        n3.downstream_node = None; n3.gw_node = 300
        stm.nodes = {1: n1, 2: n2, 3: n3}

        model = _make_mock_model(streams=stm)
        del model.groundwater
        del model.lakes
        del model.rootzone
        del model.small_watersheds
        del model.unsaturated_zone
        client = self._summary_client(model)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
            patch.object(model_state, "get_stream_reach_boundaries", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            data = resp.json()
            # Node 3 is terminal (not in has_downstream set), has gw_node -> 1 reach
            assert data["streams"]["n_reaches"] == 1

    def test_streams_reaches_from_preprocessor(self):
        """When all heuristics fail, uses preprocessor binary as last resort."""
        stm = MagicMock()
        stm.n_nodes = 3
        stm.n_reaches = 0
        stm.n_diversions = 0
        stm.n_bypasses = 0

        # All nodes have reach_id == 0, no connectivity data either
        n1 = MagicMock(); n1.id = 1; n1.reach_id = 0
        n1.downstream_node = None; n1.gw_node = None
        n2 = MagicMock(); n2.id = 2; n2.reach_id = 0
        n2.downstream_node = None; n2.gw_node = None
        n3 = MagicMock(); n3.id = 3; n3.reach_id = 0
        n3.downstream_node = None; n3.gw_node = None
        stm.nodes = {1: n1, 2: n2, 3: n3}

        model = _make_mock_model(streams=stm)
        del model.groundwater
        del model.lakes
        del model.rootzone
        del model.small_watersheds
        del model.unsaturated_zone
        client = self._summary_client(model)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
            patch.object(
                model_state, "get_stream_reach_boundaries",
                return_value=[(1, 1, 5), (2, 6, 10), (3, 11, 15)],
            ),
        ):
            resp = client.get("/api/model/summary")
            data = resp.json()
            assert data["streams"]["n_reaches"] == 3

    def test_streams_preprocessor_returns_none(self):
        """When preprocessor returns None, n_reaches stays 0."""
        stm = MagicMock()
        stm.n_nodes = 0
        stm.n_reaches = 0
        stm.n_diversions = None
        stm.n_bypasses = None
        stm.nodes = {}

        model = _make_mock_model(streams=stm)
        del model.groundwater
        del model.lakes
        del model.rootzone
        del model.small_watersheds
        del model.unsaturated_zone
        client = self._summary_client(model)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
            patch.object(model_state, "get_stream_reach_boundaries", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            data = resp.json()
            assert data["streams"]["loaded"] is True
            assert data["streams"]["n_reaches"] == 0

    def test_streams_reach_id_exception(self):
        """Exception during reach_id iteration is caught."""
        stm = MagicMock()
        stm.n_nodes = 2
        stm.n_reaches = 0
        stm.n_diversions = 0
        stm.n_bypasses = 0

        # nodes.values() raises
        stm.nodes = MagicMock()
        stm.nodes.values.side_effect = RuntimeError("broken")

        model = _make_mock_model(streams=stm)
        del model.groundwater
        del model.lakes
        del model.rootzone
        del model.small_watersheds
        del model.unsaturated_zone
        client = self._summary_client(model)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
            patch.object(model_state, "get_stream_reach_boundaries", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            data = resp.json()
            # Falls through all strategies, stays 0
            assert data["streams"]["n_reaches"] == 0

    def test_streams_connectivity_no_downstream_populated(self):
        """When no node has downstream_node in nodes_dict, heuristic skips."""
        stm = MagicMock()
        stm.n_nodes = 2
        stm.n_reaches = 0
        stm.n_diversions = 0
        stm.n_bypasses = 0

        # All have reach_id == 0 (strategy 1 fails)
        # All have downstream_node pointing to IDs NOT in nodes_dict
        n1 = MagicMock(); n1.id = 1; n1.reach_id = 0
        n1.downstream_node = 999  # Not in dict
        n1.gw_node = 100
        n2 = MagicMock(); n2.id = 2; n2.reach_id = 0
        n2.downstream_node = 998  # Not in dict
        n2.gw_node = 200
        stm.nodes = {1: n1, 2: n2}

        model = _make_mock_model(streams=stm)
        del model.groundwater
        del model.lakes
        del model.rootzone
        del model.small_watersheds
        del model.unsaturated_zone
        client = self._summary_client(model)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
            patch.object(model_state, "get_stream_reach_boundaries", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            data = resp.json()
            # has_downstream is empty -> skip heuristic -> falls through to preprocessor
            assert data["streams"]["n_reaches"] == 0

    def test_streams_preprocessor_exception(self):
        """Exception in get_stream_reach_boundaries is caught."""
        stm = MagicMock()
        stm.n_nodes = 0
        stm.n_reaches = 0
        stm.n_diversions = 0
        stm.n_bypasses = 0
        stm.nodes = {}

        model = _make_mock_model(streams=stm)
        del model.groundwater
        del model.lakes
        del model.rootzone
        del model.small_watersheds
        del model.unsaturated_zone
        client = self._summary_client(model)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
            patch.object(
                model_state, "get_stream_reach_boundaries",
                side_effect=RuntimeError("binary parse error"),
            ),
        ):
            resp = client.get("/api/model/summary")
            data = resp.json()
            assert data["streams"]["n_reaches"] == 0

    def test_streams_connectivity_terminal_zero(self):
        """When connectivity exists but all terminal nodes lack gw_node, terminal==0."""
        stm = MagicMock()
        stm.n_nodes = 3
        stm.n_reaches = 0
        stm.n_diversions = 0
        stm.n_bypasses = 0

        # All reach_id == 0 -> strategy 1 fails
        n1 = MagicMock(); n1.id = 1; n1.reach_id = 0
        n1.downstream_node = 2; n1.gw_node = 100
        n2 = MagicMock(); n2.id = 2; n2.reach_id = 0
        n2.downstream_node = 3; n2.gw_node = 200
        # Node 3 is terminal (not in has_downstream), but gw_node is None
        n3 = MagicMock(); n3.id = 3; n3.reach_id = 0
        n3.downstream_node = None; n3.gw_node = None
        stm.nodes = {1: n1, 2: n2, 3: n3}

        model = _make_mock_model(streams=stm)
        del model.groundwater
        del model.lakes
        del model.rootzone
        del model.small_watersheds
        del model.unsaturated_zone
        client = self._summary_client(model)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
            patch.object(model_state, "get_stream_reach_boundaries", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            data = resp.json()
            # terminal == 0 because gw_node is None on the only terminal node
            assert data["streams"]["n_reaches"] == 0

    def test_streams_connectivity_exception(self):
        """Exception during connectivity heuristic is caught."""
        stm = MagicMock()
        stm.n_nodes = 2
        stm.n_reaches = 0
        stm.n_diversions = 0
        stm.n_bypasses = 0

        # reach_id strategy succeeds with empty set (all reach_id==0)
        # But connectivity iteration raises
        n1 = MagicMock(); n1.id = 1; n1.reach_id = 0
        # Getting downstream_node will raise
        type(n1).downstream_node = property(lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
        stm.nodes = {1: n1}

        model = _make_mock_model(streams=stm)
        del model.groundwater
        del model.lakes
        del model.rootzone
        del model.small_watersheds
        del model.unsaturated_zone
        client = self._summary_client(model)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
            patch.object(model_state, "get_stream_reach_boundaries", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            data = resp.json()
            assert data["streams"]["n_reaches"] == 0


# ===========================================================================
# GET /api/model/summary - Lakes section
# ===========================================================================


class TestSummaryLakes:
    """Tests for the lakes section of /api/model/summary."""

    def test_no_lakes(self):
        """When model has no lakes attr, loaded=False."""
        model = _make_mock_model()
        del model.groundwater
        del model.streams
        del model.lakes
        del model.rootzone
        del model.small_watersheds
        del model.unsaturated_zone
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            data = resp.json()
            assert data["lakes"]["loaded"] is False

    def test_lakes_loaded(self):
        """When lakes component present, reports n_lakes and n_lake_elements."""
        lk = MagicMock()
        lk.n_lakes = 3
        lk.n_lake_elements = 12

        model = _make_mock_model(lakes=lk)
        del model.groundwater
        del model.streams
        del model.rootzone
        del model.small_watersheds
        del model.unsaturated_zone
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            data = resp.json()
            assert data["lakes"]["loaded"] is True
            assert data["lakes"]["n_lakes"] == 3
            assert data["lakes"]["n_lake_elements"] == 12


# ===========================================================================
# GET /api/model/summary - Root Zone section
# ===========================================================================


class TestSummaryRootZone:
    """Tests for the rootzone section of /api/model/summary."""

    def test_no_rootzone(self):
        """When model has no rootzone attr, loaded=False."""
        model = _make_mock_model()
        del model.groundwater
        del model.streams
        del model.lakes
        del model.rootzone
        del model.small_watersheds
        del model.unsaturated_zone
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            data = resp.json()
            assert data["rootzone"]["loaded"] is False

    def test_rootzone_with_nonzero_crops(self):
        """When n_crop_types > 0, uses it directly."""
        rz = MagicMock()
        rz.n_crop_types = 20
        rz.nonponded_config = MagicMock()
        rz.ponded_config = MagicMock()
        rz.urban_config = None
        rz.native_riparian_config = MagicMock()
        rz.soil_params = {}
        rz.element_landuse = {}
        rz.nonponded_area_file = None
        rz.ponded_area_file = None
        rz.urban_area_file = None
        rz.native_area_file = None

        model = _make_mock_model(rootzone=rz, n_elements=10)
        del model.groundwater
        del model.streams
        del model.lakes
        del model.small_watersheds
        del model.unsaturated_zone
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            data = resp.json()
            assert data["rootzone"]["loaded"] is True
            assert data["rootzone"]["n_crop_types"] == 20
            # 3 land use types: nonponded, ponded, native_riparian
            assert data["rootzone"]["n_land_use_types"] == 3
            names = data["rootzone"]["land_use_type_names"]
            assert "Non-ponded Agricultural" in names
            assert "Ponded Agricultural" in names
            assert "Native/Riparian" in names

    def test_rootzone_crop_count_from_subconfigs(self):
        """When n_crop_types == 0, computes from sub-configs."""
        rz = MagicMock()
        rz.n_crop_types = 0
        nonponded = MagicMock(); nonponded.n_crops = 10
        rz.nonponded_config = nonponded
        rz.ponded_config = MagicMock()  # Fixed 5
        rz.urban_config = MagicMock()  # Fixed 1
        rz.native_riparian_config = MagicMock()  # Fixed 2
        rz.soil_params = {}
        rz.element_landuse = {}
        rz.nonponded_area_file = None
        rz.ponded_area_file = None
        rz.urban_area_file = None
        rz.native_area_file = None

        model = _make_mock_model(rootzone=rz, n_elements=5)
        del model.groundwater
        del model.streams
        del model.lakes
        del model.small_watersheds
        del model.unsaturated_zone
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            data = resp.json()
            # 10 + 5 + 1 + 2 = 18
            assert data["rootzone"]["n_crop_types"] == 18
            assert data["rootzone"]["n_land_use_types"] == 4

    def test_rootzone_crop_count_zero_no_subconfigs(self):
        """When n_crop_types == 0 and no sub-configs set, n_crops stays 0 -> None."""
        rz = MagicMock()
        rz.n_crop_types = 0
        rz.nonponded_config = None
        rz.ponded_config = None
        rz.urban_config = None
        rz.native_riparian_config = None
        rz.soil_params = {}
        rz.element_landuse = {}
        rz.nonponded_area_file = None
        rz.ponded_area_file = None
        rz.urban_area_file = None
        rz.native_area_file = None

        model = _make_mock_model(rootzone=rz, n_elements=5)
        del model.groundwater
        del model.streams
        del model.lakes
        del model.small_watersheds
        del model.unsaturated_zone
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            data = resp.json()
            # count stays 0 -> n_crops stays 0 -> reported as None
            assert data["rootzone"]["n_crop_types"] is None
            assert data["rootzone"]["n_land_use_types"] is None
            assert data["rootzone"]["land_use_type_names"] is None

    def test_rootzone_crop_count_partial_subconfigs(self):
        """When n_crop_types == 0 with only nonponded and urban, partial count."""
        rz = MagicMock()
        rz.n_crop_types = 0
        nonponded = MagicMock(); nonponded.n_crops = 8
        rz.nonponded_config = nonponded
        rz.ponded_config = None  # Not set
        rz.urban_config = MagicMock()  # Set
        rz.native_riparian_config = None  # Not set
        rz.soil_params = {}
        rz.element_landuse = {}
        rz.nonponded_area_file = None
        rz.ponded_area_file = None
        rz.urban_area_file = None
        rz.native_area_file = None

        model = _make_mock_model(rootzone=rz, n_elements=5)
        del model.groundwater
        del model.streams
        del model.lakes
        del model.small_watersheds
        del model.unsaturated_zone
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            data = resp.json()
            # 8 nonponded + 1 urban = 9
            assert data["rootzone"]["n_crop_types"] == 9
            assert data["rootzone"]["n_land_use_types"] == 2
            names = data["rootzone"]["land_use_type_names"]
            assert "Non-ponded Agricultural" in names
            assert "Urban" in names
            assert "Ponded Agricultural" not in names
            assert "Native/Riparian" not in names

    def test_rootzone_soil_params_with_missing(self):
        """Missing soil param element IDs are identified."""
        rz = MagicMock()
        rz.n_crop_types = 5
        rz.nonponded_config = None
        rz.ponded_config = None
        rz.urban_config = None
        rz.native_riparian_config = None
        # Only 3 of 5 elements have soil params
        rz.soil_params = {1: MagicMock(), 2: MagicMock(), 4: MagicMock()}
        rz.element_landuse = {}
        rz.nonponded_area_file = None
        rz.ponded_area_file = None
        rz.urban_area_file = None
        rz.native_area_file = None

        model = _make_mock_model(rootzone=rz, n_elements=5)
        del model.groundwater
        del model.streams
        del model.lakes
        del model.small_watersheds
        del model.unsaturated_zone
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            data = resp.json()
            assert data["rootzone"]["n_soil_parameter_sets"] == 3
            assert data["rootzone"]["missing_soil_param_elements"] == [3, 5]

    def test_rootzone_soil_params_all_present(self):
        """When all elements have soil params, missing_soil_param_elements is None."""
        rz = MagicMock()
        rz.n_crop_types = 2
        rz.nonponded_config = None
        rz.ponded_config = None
        rz.urban_config = None
        rz.native_riparian_config = None
        rz.soil_params = {1: MagicMock()}  # Matches n_elements=1
        rz.element_landuse = {}
        rz.nonponded_area_file = None
        rz.ponded_area_file = None
        rz.urban_area_file = None
        rz.native_area_file = None

        model = _make_mock_model(rootzone=rz, n_elements=1)
        del model.groundwater
        del model.streams
        del model.lakes
        del model.small_watersheds
        del model.unsaturated_zone
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            data = resp.json()
            assert data["rootzone"]["n_soil_parameter_sets"] == 1
            assert data["rootzone"]["missing_soil_param_elements"] is None

    def test_rootzone_element_landuse_populated(self):
        """When element_landuse is populated, coverage stats are computed."""
        rz = MagicMock()
        rz.n_crop_types = 3
        rz.nonponded_config = None
        rz.ponded_config = None
        rz.urban_config = None
        rz.native_riparian_config = None
        rz.soil_params = {}
        rz.nonponded_area_file = None
        rz.ponded_area_file = None
        rz.urban_area_file = None
        rz.native_area_file = None

        # 3 elements with land use data out of 5 total
        elu1 = MagicMock(); elu1.element_id = 1
        elu2 = MagicMock(); elu2.element_id = 2
        elu3 = MagicMock(); elu3.element_id = 3
        rz.element_landuse = [elu1, elu2, elu3]

        model = _make_mock_model(rootzone=rz, n_elements=5)
        del model.groundwater
        del model.streams
        del model.lakes
        del model.small_watersheds
        del model.unsaturated_zone
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            data = resp.json()
            assert data["rootzone"]["n_land_use_elements"] == 3
            assert data["rootzone"]["n_missing_land_use"] == 2
            assert data["rootzone"]["land_use_coverage"] == "3/5"

    def test_rootzone_area_manager_fallback(self):
        """When element_landuse is empty, falls back to HDF5 area manager."""
        rz = MagicMock()
        rz.n_crop_types = 2
        rz.nonponded_config = None
        rz.ponded_config = None
        rz.urban_config = None
        rz.native_riparian_config = None
        rz.soil_params = {}
        rz.element_landuse = {}  # empty -> trigger fallback
        rz.nonponded_area_file = None
        rz.ponded_area_file = None
        rz.urban_area_file = None
        rz.native_area_file = None

        mgr = MagicMock()
        mgr.n_timesteps = 10
        # get_snapshot returns a dict with 3 elements covered
        mgr.get_snapshot.return_value = {1: {}, 2: {}, 3: {}}

        model = _make_mock_model(rootzone=rz, n_elements=5)
        del model.groundwater
        del model.streams
        del model.lakes
        del model.small_watersheds
        del model.unsaturated_zone
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=mgr),
        ):
            resp = client.get("/api/model/summary")
            data = resp.json()
            assert data["rootzone"]["n_land_use_elements"] == 3
            assert data["rootzone"]["n_missing_land_use"] == 2
            assert data["rootzone"]["land_use_coverage"] == "3/5"
            assert data["rootzone"]["n_area_timesteps"] == 10

    def test_rootzone_area_manager_exception(self):
        """Exception in area manager is caught gracefully."""
        rz = MagicMock()
        rz.n_crop_types = 2
        rz.nonponded_config = None
        rz.ponded_config = None
        rz.urban_config = None
        rz.native_riparian_config = None
        rz.soil_params = {}
        rz.element_landuse = {}
        rz.nonponded_area_file = None
        rz.ponded_area_file = None
        rz.urban_area_file = None
        rz.native_area_file = None

        model = _make_mock_model(rootzone=rz, n_elements=5)
        del model.groundwater
        del model.streams
        del model.lakes
        del model.small_watersheds
        del model.unsaturated_zone
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(
                model_state, "get_area_manager",
                side_effect=RuntimeError("HDF5 error"),
            ),
        ):
            resp = client.get("/api/model/summary")
            data = resp.json()
            assert data["rootzone"]["loaded"] is True
            assert data["rootzone"]["n_land_use_elements"] is None
            assert data["rootzone"]["n_area_timesteps"] is None

    def test_rootzone_lazy_load_triggered(self):
        """When element_landuse is empty and area files exist, triggers lazy load."""
        rz = MagicMock()
        rz.n_crop_types = 1
        rz.nonponded_config = None
        rz.ponded_config = None
        rz.urban_config = None
        rz.native_riparian_config = None
        rz.soil_params = {}
        rz.element_landuse = {}  # empty -> will try lazy load
        rz.nonponded_area_file = "/some/path.dat"
        rz.ponded_area_file = None
        rz.urban_area_file = None
        rz.native_area_file = None

        model = _make_mock_model(rootzone=rz, n_elements=3)
        del model.groundwater
        del model.streams
        del model.lakes
        del model.small_watersheds
        del model.unsaturated_zone
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
            patch(
                "pyiwfm.visualization.webapi.routes.model._ensure_land_use_loaded",
                side_effect=RuntimeError("lazy load failed"),
                create=True,
            ) as mock_lazy,
        ):
            # The lazy load is imported inside the function, so we need to
            # patch the rootzone module's function. Since it's imported
            # dynamically, we let the code path naturally hit the except.
            resp = client.get("/api/model/summary")
            data = resp.json()
            # Should still succeed even with lazy load failure
            assert data["rootzone"]["loaded"] is True

    def test_rootzone_area_manager_zero_timesteps(self):
        """Area manager with zero timesteps is skipped for fallback."""
        rz = MagicMock()
        rz.n_crop_types = 2
        rz.nonponded_config = None
        rz.ponded_config = None
        rz.urban_config = None
        rz.native_riparian_config = None
        rz.soil_params = {}
        rz.element_landuse = {}
        rz.nonponded_area_file = None
        rz.ponded_area_file = None
        rz.urban_area_file = None
        rz.native_area_file = None

        mgr = MagicMock()
        mgr.n_timesteps = 0  # No data

        model = _make_mock_model(rootzone=rz, n_elements=5)
        del model.groundwater
        del model.streams
        del model.lakes
        del model.small_watersheds
        del model.unsaturated_zone
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=mgr),
        ):
            resp = client.get("/api/model/summary")
            data = resp.json()
            assert data["rootzone"]["n_land_use_elements"] is None
            assert data["rootzone"]["n_area_timesteps"] is None

    def test_rootzone_no_soil_params_attr(self):
        """When rootzone has no soil_params attr, n_soil is 0."""
        rz = MagicMock(spec=[])
        rz.n_crop_types = 1
        rz.nonponded_config = None
        rz.ponded_config = None
        rz.urban_config = None
        rz.native_riparian_config = None
        rz.element_landuse = {}
        rz.nonponded_area_file = None
        rz.ponded_area_file = None
        rz.urban_area_file = None
        rz.native_area_file = None
        # No soil_params attribute

        model = _make_mock_model(rootzone=rz, n_elements=2)
        del model.groundwater
        del model.streams
        del model.lakes
        del model.small_watersheds
        del model.unsaturated_zone
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            data = resp.json()
            assert data["rootzone"]["n_soil_parameter_sets"] is None

    def test_rootzone_area_manager_empty_snapshot(self):
        """Area manager with empty snapshot returns no coverage."""
        rz = MagicMock()
        rz.n_crop_types = 1
        rz.nonponded_config = None
        rz.ponded_config = None
        rz.urban_config = None
        rz.native_riparian_config = None
        rz.soil_params = {}
        rz.element_landuse = {}
        rz.nonponded_area_file = None
        rz.ponded_area_file = None
        rz.urban_area_file = None
        rz.native_area_file = None

        mgr = MagicMock()
        mgr.n_timesteps = 5
        mgr.get_snapshot.return_value = {}  # Empty dict

        model = _make_mock_model(rootzone=rz, n_elements=3)
        del model.groundwater
        del model.streams
        del model.lakes
        del model.small_watersheds
        del model.unsaturated_zone
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=mgr),
        ):
            resp = client.get("/api/model/summary")
            data = resp.json()
            # Empty snapshot -> n_lu_elements stays None (snapshot falsy)
            assert data["rootzone"]["n_land_use_elements"] is None
            # But n_area_timesteps is still set from the second call
            assert data["rootzone"]["n_area_timesteps"] == 5


# ===========================================================================
# GET /api/model/summary - Small Watersheds section
# ===========================================================================


class TestSummarySmallWatersheds:
    """Tests for the small_watersheds section of /api/model/summary."""

    def test_no_small_watersheds(self):
        """When model has no small_watersheds attr, loaded=False."""
        model = _make_mock_model()
        del model.groundwater
        del model.streams
        del model.lakes
        del model.rootzone
        del model.small_watersheds
        del model.unsaturated_zone
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            data = resp.json()
            assert data["small_watersheds"]["loaded"] is False

    def test_small_watersheds_loaded(self):
        """When small watersheds present, reports n_watersheds."""
        sw = MagicMock()
        sw.n_watersheds = 7

        model = _make_mock_model(small_watersheds=sw)
        del model.groundwater
        del model.streams
        del model.lakes
        del model.rootzone
        del model.unsaturated_zone
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            data = resp.json()
            assert data["small_watersheds"]["loaded"] is True
            assert data["small_watersheds"]["n_watersheds"] == 7


# ===========================================================================
# GET /api/model/summary - Unsaturated Zone section
# ===========================================================================


class TestSummaryUnsaturatedZone:
    """Tests for the unsaturated_zone section of /api/model/summary."""

    def test_no_unsaturated_zone(self):
        """When model has no unsaturated_zone attr, loaded=False."""
        model = _make_mock_model()
        del model.groundwater
        del model.streams
        del model.lakes
        del model.rootzone
        del model.small_watersheds
        del model.unsaturated_zone
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            data = resp.json()
            assert data["unsaturated_zone"]["loaded"] is False

    def test_unsaturated_zone_loaded(self):
        """When unsaturated zone present, reports n_layers and n_elements."""
        uz = MagicMock()
        uz.n_layers = 3
        uz.n_elements = 100

        model = _make_mock_model(unsaturated_zone=uz)
        del model.groundwater
        del model.streams
        del model.lakes
        del model.rootzone
        del model.small_watersheds
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            data = resp.json()
            assert data["unsaturated_zone"]["loaded"] is True
            assert data["unsaturated_zone"]["n_layers"] == 3
            assert data["unsaturated_zone"]["n_elements"] == 100


# ===========================================================================
# GET /api/model/summary - Available Results section
# ===========================================================================


class TestSummaryAvailableResults:
    """Tests for the available_results section of /api/model/summary."""

    def test_no_results(self):
        """When all result getters return None, everything is default."""
        model = _make_mock_model()
        del model.groundwater
        del model.streams
        del model.lakes
        del model.rootzone
        del model.small_watersheds
        del model.unsaturated_zone
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            data = resp.json()
            ar = data["available_results"]
            assert ar["has_head_data"] is False
            assert ar["n_head_timesteps"] == 0
            assert ar["has_gw_hydrographs"] is False
            assert ar["has_stream_hydrographs"] is False
            assert ar["n_budget_types"] == 0
            assert ar["budget_types"] == []

    def test_all_results_available(self):
        """When all result getters return data, all fields are populated."""
        model = _make_mock_model()
        del model.groundwater
        del model.streams
        del model.lakes
        del model.rootzone
        del model.small_watersheds
        del model.unsaturated_zone
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        head_loader = MagicMock()
        head_loader.n_frames = 50

        with (
            patch.object(model_state, "get_head_loader", return_value=head_loader),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=MagicMock()),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=MagicMock()),
            patch.object(model_state, "get_available_budgets", return_value=["gw", "stream"]),
            patch.object(model_state, "get_area_manager", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            data = resp.json()
            ar = data["available_results"]
            assert ar["has_head_data"] is True
            assert ar["n_head_timesteps"] == 50
            assert ar["has_gw_hydrographs"] is True
            assert ar["has_stream_hydrographs"] is True
            assert ar["n_budget_types"] == 2
            assert ar["budget_types"] == ["gw", "stream"]

    def test_head_loader_exception(self):
        """Exception in get_head_loader is caught, head data defaults."""
        model = _make_mock_model()
        del model.groundwater
        del model.streams
        del model.lakes
        del model.rootzone
        del model.small_watersheds
        del model.unsaturated_zone
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        with (
            patch.object(
                model_state, "get_head_loader",
                side_effect=RuntimeError("HDF5 corrupt"),
            ),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            data = resp.json()
            assert data["available_results"]["has_head_data"] is False

    def test_gw_hydrograph_exception(self):
        """Exception in get_gw_hydrograph_reader is caught."""
        model = _make_mock_model()
        del model.groundwater
        del model.streams
        del model.lakes
        del model.rootzone
        del model.small_watersheds
        del model.unsaturated_zone
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(
                model_state, "get_gw_hydrograph_reader",
                side_effect=RuntimeError("parse error"),
            ),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            data = resp.json()
            assert data["available_results"]["has_gw_hydrographs"] is False

    def test_stream_hydrograph_exception(self):
        """Exception in get_stream_hydrograph_reader is caught."""
        model = _make_mock_model()
        del model.groundwater
        del model.streams
        del model.lakes
        del model.rootzone
        del model.small_watersheds
        del model.unsaturated_zone
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(
                model_state, "get_stream_hydrograph_reader",
                side_effect=RuntimeError("file not found"),
            ),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            data = resp.json()
            assert data["available_results"]["has_stream_hydrographs"] is False

    def test_budget_exception(self):
        """Exception in get_available_budgets is caught."""
        model = _make_mock_model()
        del model.groundwater
        del model.streams
        del model.lakes
        del model.rootzone
        del model.small_watersheds
        del model.unsaturated_zone
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(
                model_state, "get_available_budgets",
                side_effect=RuntimeError("budget error"),
            ),
            patch.object(model_state, "get_area_manager", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            data = resp.json()
            assert data["available_results"]["n_budget_types"] == 0
            assert data["available_results"]["budget_types"] == []


# ===========================================================================
# GET /api/model/summary - Source metadata
# ===========================================================================


class TestSummarySource:
    """Tests for the source field in /api/model/summary."""

    def test_source_from_metadata(self):
        """Source field populated from model.metadata['source']."""
        model = _make_mock_model(metadata={"source": "/data/model.dat"})
        del model.groundwater
        del model.streams
        del model.lakes
        del model.rootzone
        del model.small_watersheds
        del model.unsaturated_zone
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            data = resp.json()
            assert data["source"] == "/data/model.dat"

    def test_source_none_when_no_metadata_attr(self):
        """Source is None when model has no metadata attr."""
        model = _make_mock_model()
        del model.metadata
        del model.groundwater
        del model.streams
        del model.lakes
        del model.rootzone
        del model.small_watersheds
        del model.unsaturated_zone
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            data = resp.json()
            assert data["source"] is None

    def test_source_none_when_key_missing(self):
        """Source is None when metadata exists but has no 'source' key."""
        model = _make_mock_model(metadata={})
        del model.groundwater
        del model.streams
        del model.lakes
        del model.rootzone
        del model.small_watersheds
        del model.unsaturated_zone
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        with (
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=None),
            patch.object(model_state, "get_available_budgets", return_value=[]),
            patch.object(model_state, "get_area_manager", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            data = resp.json()
            assert data["source"] is None


# ===========================================================================
# GET /api/model/summary - Full integration-style tests
# ===========================================================================


class TestSummaryIntegration:
    """Integration tests exercising multiple sections of /api/model/summary."""

    def test_full_model_with_all_components(self):
        """A model with every component set should return a complete summary."""
        gw = MagicMock()
        gw.n_boundary_conditions = 5
        gw.n_tile_drains = 2
        gw.n_wells = 10
        gw.n_hydrograph_locations = 3
        gw.aquifer_params = MagicMock()

        stm = MagicMock()
        stm.n_nodes = 20
        stm.n_reaches = 4
        stm.n_diversions = 3
        stm.n_bypasses = 1

        lk = MagicMock()
        lk.n_lakes = 2
        lk.n_lake_elements = 8

        rz = MagicMock()
        rz.n_crop_types = 15
        rz.nonponded_config = MagicMock()
        rz.ponded_config = None
        rz.urban_config = MagicMock()
        rz.native_riparian_config = None
        rz.soil_params = {1: MagicMock()}
        rz.element_landuse = {}
        rz.nonponded_area_file = None
        rz.ponded_area_file = None
        rz.urban_area_file = None
        rz.native_area_file = None

        sw = MagicMock()
        sw.n_watersheds = 5

        uz = MagicMock()
        uz.n_layers = 4
        uz.n_elements = 50

        model = _make_mock_model(
            groundwater=gw,
            streams=stm,
            lakes=lk,
            rootzone=rz,
            small_watersheds=sw,
            unsaturated_zone=uz,
            metadata={"source": "/full/model"},
            n_elements=1,
        )
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        head_loader = MagicMock()
        head_loader.n_frames = 100

        with (
            patch.object(model_state, "get_head_loader", return_value=head_loader),
            patch.object(model_state, "get_gw_hydrograph_reader", return_value=MagicMock()),
            patch.object(model_state, "get_stream_hydrograph_reader", return_value=MagicMock()),
            patch.object(model_state, "get_available_budgets", return_value=["gw", "stream"]),
            patch.object(model_state, "get_area_manager", return_value=None),
        ):
            resp = client.get("/api/model/summary")
            assert resp.status_code == 200
            data = resp.json()

            assert data["name"] == "TestModel"
            assert data["source"] == "/full/model"
            assert data["groundwater"]["loaded"] is True
            assert data["groundwater"]["n_wells"] == 10
            assert data["streams"]["loaded"] is True
            assert data["streams"]["n_reaches"] == 4
            assert data["lakes"]["loaded"] is True
            assert data["lakes"]["n_lakes"] == 2
            assert data["rootzone"]["loaded"] is True
            assert data["rootzone"]["n_crop_types"] == 15
            assert data["small_watersheds"]["loaded"] is True
            assert data["small_watersheds"]["n_watersheds"] == 5
            assert data["unsaturated_zone"]["loaded"] is True
            assert data["unsaturated_zone"]["n_layers"] == 4
            assert data["available_results"]["has_head_data"] is True
            assert data["available_results"]["n_head_timesteps"] == 100

    def test_summary_no_model_returns_404(self, client_no_model):
        """When no model is loaded, returns 404."""
        resp = client_no_model.get("/api/model/summary")
        assert resp.status_code == 404
        assert "No model loaded" in resp.json()["detail"]
