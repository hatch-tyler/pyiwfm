"""Comprehensive tests for the groundwater API routes.

Covers all endpoints in ``pyiwfm.visualization.webapi.routes.groundwater``:

* ``_safe_float`` helper
* ``GET /api/groundwater/wells``
* ``GET /api/groundwater/boundary-conditions``
* ``GET /api/groundwater/subsidence-locations``
* ``_well_function`` (Theis W(u))
* ``GET /api/groundwater/well-impact``

Every branch and edge case documented in the route source is exercised.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi", reason="FastAPI not available")
pytest.importorskip("pydantic", reason="Pydantic not available")

from fastapi.testclient import TestClient

from pyiwfm.core.mesh import AppGrid, Element, Node
from pyiwfm.visualization.webapi.config import model_state
from pyiwfm.visualization.webapi.routes.groundwater import _safe_float, _well_function
from pyiwfm.visualization.webapi.server import create_app

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
    model.n_nodes = 4
    model.n_elements = 1
    model.n_layers = 1
    model.stratigraphy = None
    model.streams = None
    model.lakes = None
    model.groundwater = None
    return model


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


def _make_mock_well(
    well_id=1,
    x=50.0,
    y=50.0,
    name="TestWell",
    element=1,
    pump_rate=100.0,
    max_pump_rate=200.0,
    top_screen=50.0,
    bottom_screen=10.0,
    layers=None,
    screen_length=40.0,
):
    """Create a mock well object."""
    well = MagicMock()
    well.id = well_id
    well.x = x
    well.y = y
    well.name = name
    well.element = element
    well.pump_rate = pump_rate
    well.max_pump_rate = max_pump_rate
    well.top_screen = top_screen
    well.bottom_screen = bottom_screen
    well.layers = layers or [1]
    well.screen_length = screen_length
    return well


def _make_mock_bc(
    bc_id=1,
    nodes=None,
    bc_type="specified_head",
    values=None,
    layer=1,
    conductance=None,
):
    """Create a mock boundary condition object."""
    bc = MagicMock()
    bc.id = bc_id
    bc.nodes = nodes or [1, 2]
    bc.bc_type = bc_type
    bc.values = values or [10.0, 20.0]
    bc.layer = layer
    bc.conductance = conductance
    return bc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client_no_model():
    """TestClient with no model loaded."""
    _reset_model_state()
    app = create_app()
    yield TestClient(app)
    _reset_model_state()


@pytest.fixture()
def client_with_model():
    """TestClient with a model loaded but no groundwater component."""
    _reset_model_state()
    model = _make_mock_model()
    model_state._model = model
    app = create_app()
    yield TestClient(app)
    _reset_model_state()


@pytest.fixture()
def client_with_gw():
    """TestClient with model that has groundwater + wells."""
    _reset_model_state()
    model = _make_mock_model()
    gw = MagicMock()
    well1 = _make_mock_well(well_id=1, name="Pump A", pump_rate=500.0)
    well2 = _make_mock_well(well_id=2, name=None, pump_rate=0.0)
    gw.iter_wells.return_value = [well1, well2]
    gw.boundary_conditions = []
    gw.aquifer_params = None
    model.groundwater = gw
    model_state._model = model
    # Mock reproject_coords to identity (x, y) -> (x, y)
    model_state.reproject_coords = lambda x, y: (x, y)
    app = create_app()
    yield TestClient(app), model, gw
    _reset_model_state()


# ===========================================================================
# 1. _safe_float tests
# ===========================================================================


class TestSafeFloat:
    """Tests for the _safe_float helper function."""

    def test_none_returns_default(self) -> None:
        assert _safe_float(None) == 0.0

    def test_none_returns_custom_default(self) -> None:
        assert _safe_float(None, default=-1.0) == -1.0

    def test_nan_returns_default(self) -> None:
        assert _safe_float(float("nan")) == 0.0

    def test_inf_returns_default(self) -> None:
        assert _safe_float(float("inf")) == 0.0

    def test_negative_inf_returns_default(self) -> None:
        assert _safe_float(float("-inf")) == 0.0

    def test_normal_float_passes_through(self) -> None:
        assert _safe_float(3.14) == 3.14

    def test_zero_passes_through(self) -> None:
        assert _safe_float(0.0) == 0.0

    def test_negative_float_passes_through(self) -> None:
        assert _safe_float(-42.5) == -42.5

    def test_integer_converted_to_float(self) -> None:
        result = _safe_float(7)
        assert result == 7.0
        assert isinstance(result, float)


# ===========================================================================
# 2. GET /api/groundwater/wells
# ===========================================================================


class TestGetWells:
    """Tests for GET /api/groundwater/wells."""

    def test_no_model_returns_404(self, client_no_model) -> None:
        resp = client_no_model.get("/api/groundwater/wells")
        assert resp.status_code == 404
        assert "No model loaded" in resp.json()["detail"]

    def test_no_groundwater_returns_empty(self, client_with_model) -> None:
        resp = client_with_model.get("/api/groundwater/wells")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_wells"] == 0
        assert data["wells"] == []

    def test_wells_with_names(self, client_with_gw) -> None:
        client, model, gw = client_with_gw
        resp = client.get("/api/groundwater/wells")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_wells"] == 2
        # First well has explicit name
        w1 = data["wells"][0]
        assert w1["name"] == "Pump A"
        assert w1["id"] == 1
        assert w1["pump_rate"] == 500.0

    def test_wells_without_names_get_default(self, client_with_gw) -> None:
        client, model, gw = client_with_gw
        resp = client.get("/api/groundwater/wells")
        data = resp.json()
        # Second well has name=None => "Well 2"
        w2 = data["wells"][1]
        assert w2["name"] == "Well 2"

    def test_wells_with_inf_pump_rate(self) -> None:
        """Well with inf pump rate should be clamped to 0.0 by _safe_float."""
        _reset_model_state()
        model = _make_mock_model()
        gw = MagicMock()
        well = _make_mock_well(
            well_id=1,
            pump_rate=float("inf"),
            max_pump_rate=float("nan"),
            top_screen=float("-inf"),
            bottom_screen=None,
        )
        gw.iter_wells.return_value = [well]
        model.groundwater = gw
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/groundwater/wells")
            assert resp.status_code == 200
            data = resp.json()
            w = data["wells"][0]
            assert w["pump_rate"] == 0.0
            assert w["max_pump_rate"] == 0.0
            assert w["top_screen"] == 0.0
            assert w["bottom_screen"] == 0.0
        finally:
            _reset_model_state()

    def test_wells_coordinates_reprojected(self) -> None:
        """Verify coordinates pass through reproject_coords."""
        _reset_model_state()
        model = _make_mock_model()
        gw = MagicMock()
        well = _make_mock_well(well_id=1, x=1000.0, y=2000.0)
        gw.iter_wells.return_value = [well]
        model.groundwater = gw
        model_state._model = model
        # Shift coordinates to verify reprojection is used
        model_state.reproject_coords = lambda x, y: (x + 1.0, y + 2.0)
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/groundwater/wells")
            w = resp.json()["wells"][0]
            assert w["lng"] == 1001.0
            assert w["lat"] == 2002.0
        finally:
            _reset_model_state()


# ===========================================================================
# 3. GET /api/groundwater/boundary-conditions
# ===========================================================================


class TestGetBoundaryConditions:
    """Tests for GET /api/groundwater/boundary-conditions."""

    def test_no_model_returns_404(self, client_no_model) -> None:
        resp = client_no_model.get("/api/groundwater/boundary-conditions")
        assert resp.status_code == 404
        assert "No model loaded" in resp.json()["detail"]

    def test_no_groundwater_returns_empty(self, client_with_model) -> None:
        resp = client_with_model.get("/api/groundwater/boundary-conditions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_conditions"] == 0
        assert data["nodes"] == []

    def test_bc_with_conductance(self) -> None:
        """BC that has a conductance list."""
        _reset_model_state()
        model = _make_mock_model()
        gw = MagicMock()
        bc = _make_mock_bc(
            bc_id=1,
            nodes=[1, 2],
            bc_type="general_head",
            values=[100.0, 200.0],
            layer=1,
            conductance=[5.0, 10.0],
        )
        gw.boundary_conditions = [bc]
        model.groundwater = gw
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/groundwater/boundary-conditions")
            assert resp.status_code == 200
            data = resp.json()
            assert data["n_conditions"] == 2
            assert data["nodes"][0]["conductance"] == 5.0
            assert data["nodes"][1]["conductance"] == 10.0
            assert data["nodes"][0]["bc_type"] == "general_head"
            assert data["nodes"][0]["value"] == 100.0
            assert data["nodes"][1]["value"] == 200.0
        finally:
            _reset_model_state()

    def test_bc_without_conductance(self) -> None:
        """BC where conductance is None — should result in conductance=None in response."""
        _reset_model_state()
        model = _make_mock_model()
        gw = MagicMock()
        bc = _make_mock_bc(
            bc_id=2,
            nodes=[1],
            bc_type="specified_head",
            values=[50.0],
            layer=2,
            conductance=None,
        )
        gw.boundary_conditions = [bc]
        model.groundwater = gw
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/groundwater/boundary-conditions")
            data = resp.json()
            assert data["n_conditions"] == 1
            assert data["nodes"][0]["conductance"] is None
            assert data["nodes"][0]["layer"] == 2
        finally:
            _reset_model_state()

    def test_bc_node_not_in_grid(self) -> None:
        """BC references a node ID not in grid.nodes — node should be skipped."""
        _reset_model_state()
        model = _make_mock_model()
        gw = MagicMock()
        # Node 999 does not exist in the 4-node grid
        bc = _make_mock_bc(
            bc_id=3,
            nodes=[999],
            bc_type="specified_head",
            values=[10.0],
            layer=1,
            conductance=None,
        )
        gw.boundary_conditions = [bc]
        model.groundwater = gw
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/groundwater/boundary-conditions")
            data = resp.json()
            # Node 999 not in grid, so it's skipped
            assert data["n_conditions"] == 0
            assert data["nodes"] == []
        finally:
            _reset_model_state()

    def test_bc_values_index_out_of_range(self) -> None:
        """BC with more nodes than values — should use 0.0 fallback for raw_val."""
        _reset_model_state()
        model = _make_mock_model()
        gw = MagicMock()
        # 2 nodes but only 1 value
        bc = _make_mock_bc(
            bc_id=4,
            nodes=[1, 2],
            bc_type="specified_head",
            values=[42.0],  # only 1 value for 2 nodes
            layer=1,
            conductance=None,
        )
        gw.boundary_conditions = [bc]
        model.groundwater = gw
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/groundwater/boundary-conditions")
            data = resp.json()
            assert data["n_conditions"] == 2
            # First node uses values[0] = 42.0
            assert data["nodes"][0]["value"] == 42.0
            # Second node: i=1 >= len(values)=1, so raw_val=0.0
            assert data["nodes"][1]["value"] == 0.0
        finally:
            _reset_model_state()

    def test_bc_conductance_index_out_of_range(self) -> None:
        """BC with conductance list shorter than nodes — extra nodes get None."""
        _reset_model_state()
        model = _make_mock_model()
        gw = MagicMock()
        bc = _make_mock_bc(
            bc_id=5,
            nodes=[1, 2],
            bc_type="general_head",
            values=[10.0, 20.0],
            layer=1,
            conductance=[5.0],  # only 1 conductance for 2 nodes
        )
        gw.boundary_conditions = [bc]
        model.groundwater = gw
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/groundwater/boundary-conditions")
            data = resp.json()
            assert data["n_conditions"] == 2
            # First node: i=0 < len(conductance)=1 => 5.0
            assert data["nodes"][0]["conductance"] == 5.0
            # Second node: i=1 >= len(conductance)=1 => None
            assert data["nodes"][1]["conductance"] is None
        finally:
            _reset_model_state()

    def test_bc_conductance_with_nan_value(self) -> None:
        """Conductance value is NaN — should be clamped by _safe_float."""
        _reset_model_state()
        model = _make_mock_model()
        gw = MagicMock()
        bc = _make_mock_bc(
            bc_id=6,
            nodes=[1],
            bc_type="general_head",
            values=[10.0],
            layer=1,
            conductance=[float("nan")],
        )
        gw.boundary_conditions = [bc]
        model.groundwater = gw
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/groundwater/boundary-conditions")
            data = resp.json()
            # NaN conductance is clamped to 0.0
            assert data["nodes"][0]["conductance"] == 0.0
        finally:
            _reset_model_state()

    def test_bc_empty_conductance_list(self) -> None:
        """BC with empty conductance list (truthy check fails) -> None."""
        _reset_model_state()
        model = _make_mock_model()
        gw = MagicMock()
        bc = _make_mock_bc(
            bc_id=7,
            nodes=[1],
            bc_type="specified_head",
            values=[10.0],
            layer=1,
            conductance=[],  # empty list is falsy
        )
        gw.boundary_conditions = [bc]
        model.groundwater = gw
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/groundwater/boundary-conditions")
            data = resp.json()
            # Empty list is falsy => raw_cond = None => cond = None
            assert data["nodes"][0]["conductance"] is None
        finally:
            _reset_model_state()


# ===========================================================================
# 4. GET /api/groundwater/subsidence-locations
# ===========================================================================


class TestGetSubsidenceLocations:
    """Tests for GET /api/groundwater/subsidence-locations."""

    def test_no_model_returns_404(self, client_no_model) -> None:
        resp = client_no_model.get("/api/groundwater/subsidence-locations")
        assert resp.status_code == 404

    def test_no_groundwater_returns_empty(self, client_with_model) -> None:
        resp = client_with_model.get("/api/groundwater/subsidence-locations")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_locations"] == 0

    def test_no_subsidence_config_returns_empty(self) -> None:
        """Groundwater exists but no subsidence_config attribute."""
        _reset_model_state()
        model = _make_mock_model()
        gw = MagicMock(spec=[])  # no attributes
        model.groundwater = gw
        model_state._model = model
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/groundwater/subsidence-locations")
            assert resp.status_code == 200
            data = resp.json()
            assert data["n_locations"] == 0
        finally:
            _reset_model_state()

    def test_empty_hydrograph_specs(self) -> None:
        """subsidence_config exists but hydrograph_specs is empty."""
        _reset_model_state()
        model = _make_mock_model()
        gw = MagicMock()
        subs_config = MagicMock()
        subs_config.hydrograph_specs = []
        gw.subsidence_config = subs_config
        model.groundwater = gw
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/groundwater/subsidence-locations")
            data = resp.json()
            assert data["n_locations"] == 0
            assert data["locations"] == []
        finally:
            _reset_model_state()

    def test_normal_subsidence_locations(self) -> None:
        """Normal case with subsidence observation specs."""
        _reset_model_state()
        model = _make_mock_model()
        gw = MagicMock()
        subs_config = MagicMock()
        spec1 = MagicMock()
        spec1.id = 1
        spec1.x = 50.0
        spec1.y = 50.0
        spec1.layer = 1
        spec1.name = "InSAR Point 1"
        spec2 = MagicMock()
        spec2.id = 2
        spec2.x = 75.0
        spec2.y = 75.0
        spec2.layer = 2
        spec2.name = None  # should default to "Subsidence Obs 2"
        subs_config.hydrograph_specs = [spec1, spec2]
        gw.subsidence_config = subs_config
        model.groundwater = gw
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/groundwater/subsidence-locations")
            data = resp.json()
            assert data["n_locations"] == 2
            assert data["locations"][0]["name"] == "InSAR Point 1"
            assert data["locations"][0]["id"] == 1
            assert data["locations"][0]["layer"] == 1
            assert data["locations"][0]["lng"] == 50.0
            assert data["locations"][0]["lat"] == 50.0
            assert data["locations"][1]["name"] == "Subsidence Obs 2"
        finally:
            _reset_model_state()

    def test_spec_with_reproject_error_is_skipped(self) -> None:
        """If reproject_coords raises an exception, that spec is skipped."""
        _reset_model_state()
        model = _make_mock_model()
        gw = MagicMock()
        subs_config = MagicMock()
        good_spec = MagicMock()
        good_spec.id = 1
        good_spec.x = 50.0
        good_spec.y = 50.0
        good_spec.layer = 1
        good_spec.name = "Good"
        bad_spec = MagicMock()
        bad_spec.id = 2
        bad_spec.x = -999999.0  # might cause reproject failure
        bad_spec.y = -999999.0
        bad_spec.layer = 1
        bad_spec.name = "Bad"
        subs_config.hydrograph_specs = [bad_spec, good_spec]
        gw.subsidence_config = subs_config
        model.groundwater = gw
        model_state._model = model

        call_count = 0

        def _reproject(x, y):
            nonlocal call_count
            call_count += 1
            if x == -999999.0:
                raise ValueError("Cannot reproject invalid coords")
            return (x, y)

        model_state.reproject_coords = _reproject
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/groundwater/subsidence-locations")
            data = resp.json()
            # bad_spec skipped, only good_spec returned
            assert data["n_locations"] == 1
            assert data["locations"][0]["name"] == "Good"
        finally:
            _reset_model_state()


# ===========================================================================
# 5. _well_function tests
# ===========================================================================


class TestWellFunction:
    """Tests for the Theis well function W(u)."""

    def test_u_zero_returns_zero(self) -> None:
        assert _well_function(0.0) == 0.0

    def test_u_negative_returns_zero(self) -> None:
        assert _well_function(-1.0) == 0.0

    def test_u_very_negative_returns_zero(self) -> None:
        assert _well_function(-100.0) == 0.0

    def test_u_small_series_branch(self) -> None:
        """For u < 1, uses series expansion."""
        result = _well_function(0.01)
        # W(0.01) ~ 4.0379 (known analytical value)
        assert result > 0.0
        assert abs(result - 4.0379) < 0.01

    def test_u_approaching_1_series(self) -> None:
        """u just below 1 still uses series branch."""
        result = _well_function(0.99)
        assert result >= 0.0

    def test_u_exactly_1_asymptotic_branch(self) -> None:
        """u == 1 uses the asymptotic branch (u >= 1)."""
        result = _well_function(1.0)
        # W(1) ~ 0.2194 (exact). The rational approximation gives ~0.205.
        assert result > 0.0
        assert abs(result - 0.2194) < 0.02

    def test_u_large_asymptotic_branch(self) -> None:
        """For large u, W(u) approaches zero."""
        result = _well_function(10.0)
        assert result >= 0.0
        assert result < 0.01  # Very small for large u

    def test_u_very_large_approaches_zero(self) -> None:
        """Very large u should give nearly zero."""
        result = _well_function(100.0)
        assert result >= 0.0
        assert result < 1e-30

    def test_u_small_positive(self) -> None:
        """Very small positive u gives large W(u)."""
        result = _well_function(1e-6)
        # W(1e-6) is large (approximately 13.2)
        assert result > 10.0

    def test_series_result_always_non_negative(self) -> None:
        """Series branch clamps result to max(w, 0.0)."""
        # For u close to 1 but < 1, the series should still be non-negative
        for u in [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
            assert _well_function(u) >= 0.0


# ===========================================================================
# 6. GET /api/groundwater/well-impact
# ===========================================================================


class TestGetWellImpact:
    """Tests for GET /api/groundwater/well-impact."""

    def test_no_model_returns_404(self, client_no_model) -> None:
        resp = client_no_model.get("/api/groundwater/well-impact?well_id=1")
        assert resp.status_code == 404
        assert "No model loaded" in resp.json()["detail"]

    def test_no_groundwater_returns_404(self, client_with_model) -> None:
        resp = client_with_model.get("/api/groundwater/well-impact?well_id=1")
        assert resp.status_code == 404
        assert "No groundwater data" in resp.json()["detail"]

    def test_well_not_found_returns_404(self) -> None:
        """Request well-impact for a well_id that doesn't exist."""
        _reset_model_state()
        model = _make_mock_model()
        gw = MagicMock()
        gw.iter_wells.return_value = []
        model.groundwater = gw
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/groundwater/well-impact?well_id=999")
            assert resp.status_code == 404
            assert "Well 999 not found" in resp.json()["detail"]
        finally:
            _reset_model_state()

    def test_zero_pump_rate(self) -> None:
        """Well with zero pump rate returns empty contours and message."""
        _reset_model_state()
        model = _make_mock_model()
        gw = MagicMock()
        well = _make_mock_well(well_id=1, pump_rate=0.0, name="Dry Well")
        gw.iter_wells.return_value = [well]
        model.groundwater = gw
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/groundwater/well-impact?well_id=1")
            assert resp.status_code == 200
            data = resp.json()
            assert data["contours"] == []
            assert data["message"] == "Well has zero pumping rate"
            assert data["well_id"] == 1
            assert data["name"] == "Dry Well"
        finally:
            _reset_model_state()

    def test_zero_pump_rate_well_without_name(self) -> None:
        """Well with zero pump rate and name=None gets default name."""
        _reset_model_state()
        model = _make_mock_model()
        gw = MagicMock()
        well = _make_mock_well(well_id=5, pump_rate=0.0, name=None)
        gw.iter_wells.return_value = [well]
        model.groundwater = gw
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/groundwater/well-impact?well_id=5")
            data = resp.json()
            assert data["name"] == "Well 5"
        finally:
            _reset_model_state()

    def test_very_small_pump_rate_treated_as_zero(self) -> None:
        """Pump rate < 1e-10 treated as zero."""
        _reset_model_state()
        model = _make_mock_model()
        gw = MagicMock()
        well = _make_mock_well(well_id=1, pump_rate=1e-15)
        gw.iter_wells.return_value = [well]
        model.groundwater = gw
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/groundwater/well-impact?well_id=1")
            data = resp.json()
            assert data["contours"] == []
            assert "zero pumping rate" in data["message"]
        finally:
            _reset_model_state()

    def test_negative_pump_rate_uses_abs(self) -> None:
        """Negative pump rate is made positive via abs()."""
        _reset_model_state()
        model = _make_mock_model()
        gw = MagicMock()
        well = _make_mock_well(well_id=1, pump_rate=-500.0, screen_length=100.0)
        gw.iter_wells.return_value = [well]
        gw.aquifer_params = None
        model.groundwater = gw
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/groundwater/well-impact?well_id=1")
            assert resp.status_code == 200
            data = resp.json()
            assert data["pump_rate"] == 500.0
            assert data["n_contours"] >= 0
        finally:
            _reset_model_state()

    def test_normal_pump_rate_without_aquifer_params(self) -> None:
        """Normal case using default Kh=10 and Sy=0.1."""
        _reset_model_state()
        model = _make_mock_model()
        gw = MagicMock()
        well = _make_mock_well(well_id=1, pump_rate=500.0, screen_length=100.0, name="Pump A")
        gw.iter_wells.return_value = [well]
        # No aquifer_params attribute
        gw.aquifer_params = None
        model.groundwater = gw
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/groundwater/well-impact?well_id=1")
            assert resp.status_code == 200
            data = resp.json()
            assert data["well_id"] == 1
            assert data["name"] == "Pump A"
            # T = Kh * thickness = 10 * 100 = 1000
            assert data["transmissivity"] == 1000.0
            assert data["storativity"] == 0.1
            assert data["time_days"] == 365.0  # default
            assert data["n_contours"] == len(data["contours"])
            assert data["center"]["lng"] == 50.0
            assert data["center"]["lat"] == 50.0
        finally:
            _reset_model_state()

    def test_with_aquifer_params(self) -> None:
        """Well with aquifer_params providing custom Kh and Sy."""
        _reset_model_state()
        model = _make_mock_model()
        gw = MagicMock()
        well = _make_mock_well(well_id=1, pump_rate=500.0, screen_length=50.0, element=1)
        gw.iter_wells.return_value = [well]
        params = MagicMock()
        params.get_element_params.return_value = {"kh": 20.0, "sy": 0.2}
        gw.aquifer_params = params
        model.groundwater = gw
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/groundwater/well-impact?well_id=1")
            assert resp.status_code == 200
            data = resp.json()
            # T = kh * thickness = 20 * 50 = 1000
            assert data["transmissivity"] == 1000.0
            assert data["storativity"] == 0.2
        finally:
            _reset_model_state()

    def test_aquifer_params_raises_keyerror_uses_defaults(self) -> None:
        """get_element_params raises KeyError — falls back to defaults."""
        _reset_model_state()
        model = _make_mock_model()
        gw = MagicMock()
        well = _make_mock_well(well_id=1, pump_rate=500.0, screen_length=100.0)
        gw.iter_wells.return_value = [well]
        params = MagicMock()
        params.get_element_params.side_effect = KeyError("No such element")
        gw.aquifer_params = params
        model.groundwater = gw
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/groundwater/well-impact?well_id=1")
            assert resp.status_code == 200
            data = resp.json()
            # Falls back to default Kh=10, Sy=0.1
            assert data["transmissivity"] == 1000.0
            assert data["storativity"] == 0.1
        finally:
            _reset_model_state()

    def test_aquifer_params_raises_indexerror_uses_defaults(self) -> None:
        """get_element_params raises IndexError — falls back to defaults."""
        _reset_model_state()
        model = _make_mock_model()
        gw = MagicMock()
        well = _make_mock_well(well_id=1, pump_rate=500.0, screen_length=100.0)
        gw.iter_wells.return_value = [well]
        params = MagicMock()
        params.get_element_params.side_effect = IndexError("Out of range")
        gw.aquifer_params = params
        model.groundwater = gw
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/groundwater/well-impact?well_id=1")
            assert resp.status_code == 200
            data = resp.json()
            assert data["transmissivity"] == 1000.0
            assert data["storativity"] == 0.1
        finally:
            _reset_model_state()

    def test_aquifer_params_without_get_element_params(self) -> None:
        """aquifer_params exists but lacks get_element_params method."""
        _reset_model_state()
        model = _make_mock_model()
        gw = MagicMock()
        well = _make_mock_well(well_id=1, pump_rate=500.0, screen_length=100.0)
        gw.iter_wells.return_value = [well]
        # aquifer_params without get_element_params
        params = MagicMock(spec=[])
        gw.aquifer_params = params
        model.groundwater = gw
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/groundwater/well-impact?well_id=1")
            assert resp.status_code == 200
            data = resp.json()
            # hasattr(params, "get_element_params") is False => uses defaults
            assert data["transmissivity"] == 1000.0
            assert data["storativity"] == 0.1
        finally:
            _reset_model_state()

    def test_screen_length_zero_uses_default_thickness(self) -> None:
        """screen_length <= 0 should default to 100."""
        _reset_model_state()
        model = _make_mock_model()
        gw = MagicMock()
        well = _make_mock_well(well_id=1, pump_rate=500.0, screen_length=0.0)
        gw.iter_wells.return_value = [well]
        gw.aquifer_params = None
        model.groundwater = gw
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/groundwater/well-impact?well_id=1")
            assert resp.status_code == 200
            data = resp.json()
            # T = 10 * 100 (default thickness) = 1000
            assert data["transmissivity"] == 1000.0
        finally:
            _reset_model_state()

    def test_screen_length_negative_uses_default_thickness(self) -> None:
        """screen_length < 0 should also default to 100."""
        _reset_model_state()
        model = _make_mock_model()
        gw = MagicMock()
        well = _make_mock_well(well_id=1, pump_rate=500.0, screen_length=-50.0)
        gw.iter_wells.return_value = [well]
        gw.aquifer_params = None
        model.groundwater = gw
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/groundwater/well-impact?well_id=1")
            assert resp.status_code == 200
            data = resp.json()
            assert data["transmissivity"] == 1000.0
        finally:
            _reset_model_state()

    def test_t_or_s_le_zero_returns_400(self) -> None:
        """If Kh=0 making T=0, should return HTTP 400."""
        _reset_model_state()
        model = _make_mock_model()
        gw = MagicMock()
        well = _make_mock_well(well_id=1, pump_rate=500.0, screen_length=100.0)
        gw.iter_wells.return_value = [well]
        params = MagicMock()
        params.get_element_params.return_value = {"kh": 0.0, "sy": 0.1}
        gw.aquifer_params = params
        model.groundwater = gw
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/groundwater/well-impact?well_id=1")
            assert resp.status_code == 400
            assert "T or S <= 0" in resp.json()["detail"]
        finally:
            _reset_model_state()

    def test_s_zero_returns_400(self) -> None:
        """If Sy=0 making S=0, should return HTTP 400."""
        _reset_model_state()
        model = _make_mock_model()
        gw = MagicMock()
        well = _make_mock_well(well_id=1, pump_rate=500.0, screen_length=100.0)
        gw.iter_wells.return_value = [well]
        params = MagicMock()
        params.get_element_params.return_value = {"kh": 10.0, "sy": 0.0}
        gw.aquifer_params = params
        model.groundwater = gw
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/groundwater/well-impact?well_id=1")
            assert resp.status_code == 400
            assert "T or S <= 0" in resp.json()["detail"]
        finally:
            _reset_model_state()

    def test_custom_max_radius(self) -> None:
        """Providing max_radius > 0 uses that instead of auto-compute."""
        _reset_model_state()
        model = _make_mock_model()
        gw = MagicMock()
        well = _make_mock_well(well_id=1, pump_rate=500.0, screen_length=100.0)
        gw.iter_wells.return_value = [well]
        gw.aquifer_params = None
        model.groundwater = gw
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/groundwater/well-impact?well_id=1&max_radius=5000&n_rings=5")
            assert resp.status_code == 200
            data = resp.json()
            # With a fixed max_radius=5000 and 5 rings, radii are 1000,2000,3000,4000,5000
            for c in data["contours"]:
                assert c["radius_ft"] <= 5000
        finally:
            _reset_model_state()

    def test_custom_time_parameter(self) -> None:
        """Custom time parameter is reflected in response."""
        _reset_model_state()
        model = _make_mock_model()
        gw = MagicMock()
        well = _make_mock_well(well_id=1, pump_rate=500.0, screen_length=100.0)
        gw.iter_wells.return_value = [well]
        gw.aquifer_params = None
        model.groundwater = gw
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/groundwater/well-impact?well_id=1&time=30")
            assert resp.status_code == 200
            data = resp.json()
            assert data["time_days"] == 30.0
        finally:
            _reset_model_state()

    def test_custom_n_rings(self) -> None:
        """Custom n_rings parameter affects number of contour calculations."""
        _reset_model_state()
        model = _make_mock_model()
        gw = MagicMock()
        well = _make_mock_well(well_id=1, pump_rate=500.0, screen_length=100.0)
        gw.iter_wells.return_value = [well]
        gw.aquifer_params = None
        model.groundwater = gw
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/groundwater/well-impact?well_id=1&n_rings=3")
            assert resp.status_code == 200
            data = resp.json()
            # At most 3 contours (some may be filtered if drawdown < 0.001)
            assert data["n_contours"] <= 3
        finally:
            _reset_model_state()

    def test_contour_fields_present(self) -> None:
        """Each contour has the expected fields."""
        _reset_model_state()
        model = _make_mock_model()
        gw = MagicMock()
        well = _make_mock_well(well_id=1, pump_rate=500.0, screen_length=100.0)
        gw.iter_wells.return_value = [well]
        gw.aquifer_params = None
        model.groundwater = gw
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/groundwater/well-impact?well_id=1")
            data = resp.json()
            if data["contours"]:
                c = data["contours"][0]
                assert "radius_ft" in c
                assert "radius_deg" in c
                assert "drawdown_ft" in c
                assert "u" in c
        finally:
            _reset_model_state()

    def test_contours_radius_increases(self) -> None:
        """Contour radii should be in increasing order."""
        _reset_model_state()
        model = _make_mock_model()
        gw = MagicMock()
        well = _make_mock_well(well_id=1, pump_rate=500.0, screen_length=100.0)
        gw.iter_wells.return_value = [well]
        gw.aquifer_params = None
        model.groundwater = gw
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/groundwater/well-impact?well_id=1")
            data = resp.json()
            radii = [c["radius_ft"] for c in data["contours"]]
            assert radii == sorted(radii)
        finally:
            _reset_model_state()

    def test_drawdown_decreases_with_radius(self) -> None:
        """Drawdown should decrease (or remain constant) as radius increases."""
        _reset_model_state()
        model = _make_mock_model()
        gw = MagicMock()
        well = _make_mock_well(well_id=1, pump_rate=1000.0, screen_length=100.0)
        gw.iter_wells.return_value = [well]
        gw.aquifer_params = None
        model.groundwater = gw
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/groundwater/well-impact?well_id=1&n_rings=20")
            data = resp.json()
            drawdowns = [c["drawdown_ft"] for c in data["contours"]]
            # Drawdown should be non-increasing with distance
            for i in range(1, len(drawdowns)):
                assert drawdowns[i] <= drawdowns[i - 1] + 1e-6
        finally:
            _reset_model_state()

    def test_well_impact_response_name_fallback(self) -> None:
        """Well with name=None in the full response (non-zero pump rate) gets default."""
        _reset_model_state()
        model = _make_mock_model()
        gw = MagicMock()
        well = _make_mock_well(well_id=7, pump_rate=500.0, screen_length=100.0, name=None)
        gw.iter_wells.return_value = [well]
        gw.aquifer_params = None
        model.groundwater = gw
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/groundwater/well-impact?well_id=7")
            data = resp.json()
            assert data["name"] == "Well 7"
        finally:
            _reset_model_state()

    def test_aquifer_params_partial_dict(self) -> None:
        """get_element_params returns dict with only kh, not sy => sy uses default."""
        _reset_model_state()
        model = _make_mock_model()
        gw = MagicMock()
        well = _make_mock_well(well_id=1, pump_rate=500.0, screen_length=50.0)
        gw.iter_wells.return_value = [well]
        params = MagicMock()
        # Only kh in the dict, no sy
        params.get_element_params.return_value = {"kh": 30.0}
        gw.aquifer_params = params
        model.groundwater = gw
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/groundwater/well-impact?well_id=1")
            assert resp.status_code == 200
            data = resp.json()
            # T = 30 * 50 = 1500
            assert data["transmissivity"] == 1500.0
            # sy falls back to default 0.1
            assert data["storativity"] == 0.1
        finally:
            _reset_model_state()

    def test_missing_well_id_param_returns_422(self) -> None:
        """Omitting required well_id parameter returns 422 validation error."""
        _reset_model_state()
        model = _make_mock_model()
        gw = MagicMock()
        gw.iter_wells.return_value = []
        model.groundwater = gw
        model_state._model = model
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/groundwater/well-impact")
            assert resp.status_code == 422
        finally:
            _reset_model_state()

    def test_time_zero_handled(self) -> None:
        """time=0 is valid (ge=0 constraint) but may produce edge cases.

        With time=0 the u formula has time_days=0 in the denominator.
        The auto max_radius formula also involves time_days. With time=0:
        max_radius = sqrt(4*T*0/S) * 3 = 0 and no contours are generated
        because r <= 0 for all rings.
        """
        _reset_model_state()
        model = _make_mock_model()
        gw = MagicMock()
        well = _make_mock_well(well_id=1, pump_rate=500.0, screen_length=100.0)
        gw.iter_wells.return_value = [well]
        gw.aquifer_params = None
        model.groundwater = gw
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/groundwater/well-impact?well_id=1&time=0")
            assert resp.status_code == 200
            data = resp.json()
            assert data["time_days"] == 0.0
            # max_radius=0 => all r=0 => all skipped
            assert data["n_contours"] == 0
        finally:
            _reset_model_state()

    def test_multiple_wells_finds_correct_one(self) -> None:
        """With multiple wells, the correct one is matched by ID."""
        _reset_model_state()
        model = _make_mock_model()
        gw = MagicMock()
        well1 = _make_mock_well(well_id=1, pump_rate=100.0, screen_length=100.0, name="Well1")
        well2 = _make_mock_well(well_id=2, pump_rate=200.0, screen_length=100.0, name="Well2")
        well3 = _make_mock_well(well_id=3, pump_rate=300.0, screen_length=100.0, name="Well3")
        gw.iter_wells.return_value = [well1, well2, well3]
        gw.aquifer_params = None
        model.groundwater = gw
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)
        try:
            resp = client.get("/api/groundwater/well-impact?well_id=2")
            assert resp.status_code == 200
            data = resp.json()
            assert data["well_id"] == 2
            assert data["name"] == "Well2"
            assert data["pump_rate"] == 200.0
        finally:
            _reset_model_state()
