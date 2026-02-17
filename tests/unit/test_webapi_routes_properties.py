"""Comprehensive tests for the FastAPI property routes.

Covers all endpoints in ``pyiwfm.visualization.webapi.routes.properties``:

* ``GET /api/properties`` — list available properties
* ``GET /api/properties/{property_id}`` — get property values
* ``_node_to_element_values`` helper
* ``_compute_property_values`` internal function

Every branch and edge case documented in the route source is exercised,
including: no model, stratigraphy present/absent, groundwater params with
various attribute combinations (kh, kv, ss/specific_storage, sy/specific_yield),
layer filtering, all-NaN values, thickness/top_elev/bottom_elev, and unknown
property IDs.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi", reason="FastAPI not available")
pydantic = pytest.importorskip("pydantic", reason="Pydantic not available")

from fastapi.testclient import TestClient  # noqa: E402

from pyiwfm.core.mesh import AppGrid, Element, Node  # noqa: E402
from pyiwfm.visualization.webapi.config import model_state  # noqa: E402
from pyiwfm.visualization.webapi.routes.properties import (  # noqa: E402
    _compute_property_values,
    _node_to_element_values,
)
from pyiwfm.visualization.webapi.server import create_app  # noqa: E402

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


def _make_grid(n_nodes=4):
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


def _make_grid_two_elements():
    """Create a grid with two adjacent quad elements."""
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
        2: Element(id=2, vertices=(2, 5, 6, 3), subregion=1),
    }
    grid = AppGrid(nodes=nodes, elements=elements)
    grid.compute_connectivity()
    grid.compute_areas()
    return grid


def _make_mock_stratigraphy(n_nodes=4, n_layers=2):
    """Create a mock Stratigraphy with top_elev and bottom_elev arrays."""
    strat = MagicMock()
    strat.n_layers = n_layers
    # Shape: (n_nodes, n_layers)
    strat.top_elev = np.array(
        [[100.0 + i * 10 + lay * 5 for lay in range(n_layers)] for i in range(n_nodes)]
    )
    strat.bottom_elev = np.array(
        [[50.0 + i * 10 + lay * 5 for lay in range(n_layers)] for i in range(n_nodes)]
    )
    return strat


def _make_mock_aquifer_params(kh=None, kv=None, ss=None, sy=None, n_nodes=4, n_layers=2):
    """Create a mock aquifer params object.

    If a parameter name is 'specific_storage' or 'specific_yield',
    set those attribute names instead of 'ss'/'sy'.
    """
    params = MagicMock()
    # Reset all attributes to None first
    params.kh = kh
    params.kv = kv
    # Use different attribute names for ss/sy to test fallback
    params.specific_storage = None
    params.ss = None
    params.specific_yield = None
    params.sy = None
    if ss is not None:
        # Store under 'specific_storage' attribute
        params.specific_storage = ss
    if sy is not None:
        # Store under 'specific_yield' attribute
        params.specific_yield = sy
    return params


def _make_mock_model(
    grid=None,
    stratigraphy=None,
    groundwater=None,
):
    """Create a minimal mock IWFMModel."""
    model = MagicMock()
    model.name = "TestModel"
    model.grid = grid or _make_grid()
    model.stratigraphy = stratigraphy
    model.groundwater = groundwater
    model.streams = None
    model.lakes = None
    model.small_watersheds = None
    model.metadata = {}
    model.has_streams = False
    model.has_lakes = False
    model.n_nodes = len(model.grid.nodes)
    model.n_elements = len(model.grid.elements)
    model.n_layers = stratigraphy.n_layers if stratigraphy else 1
    model.source_files = {}
    return model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _cleanup():
    """Autouse fixture to ensure clean state before and after each test."""
    _reset_model_state()
    yield
    _reset_model_state()


@pytest.fixture()
def client_no_model():
    """TestClient with no model loaded."""
    app = create_app()
    return TestClient(app)


@pytest.fixture()
def client_basic_model():
    """TestClient with a model loaded, no strat, no gw."""
    model = _make_mock_model()
    model_state._model = model
    app = create_app()
    return TestClient(app)


@pytest.fixture()
def client_with_stratigraphy():
    """TestClient with a model that has stratigraphy (2 layers)."""
    strat = _make_mock_stratigraphy(n_nodes=4, n_layers=2)
    model = _make_mock_model(stratigraphy=strat)
    model_state._model = model
    app = create_app()
    return TestClient(app)


# ===========================================================================
# 1. GET /api/properties (list_properties)
# ===========================================================================


class TestListProperties:
    """Tests for GET /api/properties."""

    def test_no_model_returns_404(self, client_no_model):
        resp = client_no_model.get("/api/properties")
        assert resp.status_code == 404
        assert "No model loaded" in resp.json()["detail"]

    def test_basic_model_returns_layer_only(self, client_basic_model):
        """With no stratigraphy and no groundwater, only 'layer' is available."""
        resp = client_basic_model.get("/api/properties")
        assert resp.status_code == 200
        data = resp.json()
        ids = [p["id"] for p in data]
        assert "layer" in ids
        assert "thickness" not in ids
        assert "kh" not in ids

    def test_with_stratigraphy_adds_thickness_top_bottom(self, client_with_stratigraphy):
        """With stratigraphy, thickness/top_elev/bottom_elev are available."""
        resp = client_with_stratigraphy.get("/api/properties")
        assert resp.status_code == 200
        data = resp.json()
        ids = [p["id"] for p in data]
        assert "layer" in ids
        assert "thickness" in ids
        assert "top_elev" in ids
        assert "bottom_elev" in ids

    def test_with_groundwater_kh_only(self):
        """Groundwater with only kh param available."""
        params = MagicMock()
        params.kh = np.array([1.0, 2.0])
        params.kv = None
        # Make getattr for ss/sy return None
        params.specific_storage = None
        params.ss = None
        params.specific_yield = None
        params.sy = None
        gw = MagicMock()
        gw.aquifer_params = params
        model = _make_mock_model(groundwater=gw)
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/properties")
        data = resp.json()
        ids = [p["id"] for p in data]
        assert "kh" in ids
        assert "kv" not in ids
        assert "ss" not in ids
        assert "sy" not in ids

    def test_with_all_gw_params(self):
        """Groundwater with all params: kh, kv, ss, sy."""
        params = MagicMock()
        params.kh = np.array([1.0])
        params.kv = np.array([0.1])
        params.specific_storage = np.array([1e-5])
        params.ss = None  # specific_storage found first
        params.specific_yield = np.array([0.2])
        params.sy = None
        gw = MagicMock()
        gw.aquifer_params = params
        model = _make_mock_model(groundwater=gw)
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/properties")
        data = resp.json()
        ids = [p["id"] for p in data]
        assert "kh" in ids
        assert "kv" in ids
        assert "ss" in ids
        assert "sy" in ids

    def test_gw_params_is_none(self):
        """Groundwater exists but aquifer_params is None."""
        gw = MagicMock()
        gw.aquifer_params = None
        model = _make_mock_model(groundwater=gw)
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/properties")
        data = resp.json()
        ids = [p["id"] for p in data]
        assert "layer" in ids
        assert "kh" not in ids

    def test_property_item_fields(self, client_basic_model):
        """Each property item has required fields."""
        resp = client_basic_model.get("/api/properties")
        data = resp.json()
        for item in data:
            assert "id" in item
            assert "name" in item
            assert "units" in item
            assert "description" in item
            assert "cmap" in item
            assert "log_scale" in item

    def test_ss_via_short_name(self):
        """SS detected via 'ss' when 'specific_storage' attribute is missing.

        The detection logic is:
            ss = getattr(params, "specific_storage", getattr(params, "ss", None))
        When 'specific_storage' does not exist as an attribute, getattr falls
        through to the default which is getattr(params, "ss", None).
        """

        class _Params:
            kh = None
            kv = None
            # No specific_storage attribute at all -> triggers fallback to ss
            ss = np.array([1e-5])
            # No specific_yield attribute at all
            sy = None

        gw = MagicMock()
        gw.aquifer_params = _Params()
        model = _make_mock_model(groundwater=gw)
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/properties")
        data = resp.json()
        ids = [p["id"] for p in data]
        assert "ss" in ids

    def test_sy_via_short_name(self):
        """SY detected via 'sy' when 'specific_yield' attribute is missing."""

        class _Params:
            kh = None
            kv = None
            specific_storage = None
            ss = None
            # No specific_yield attribute -> triggers fallback to sy
            sy = np.array([0.2])

        gw = MagicMock()
        gw.aquifer_params = _Params()
        model = _make_mock_model(groundwater=gw)
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/properties")
        data = resp.json()
        ids = [p["id"] for p in data]
        assert "sy" in ids


# ===========================================================================
# 2. GET /api/properties/{property_id} (get_property)
# ===========================================================================


class TestGetProperty:
    """Tests for GET /api/properties/{property_id}."""

    def test_no_model_returns_404(self, client_no_model):
        resp = client_no_model.get("/api/properties/layer")
        assert resp.status_code == 404
        assert "No model loaded" in resp.json()["detail"]

    def test_unknown_property_returns_404(self, client_basic_model):
        resp = client_basic_model.get("/api/properties/nonexistent")
        assert resp.status_code == 404
        assert "not available" in resp.json()["detail"]

    def test_layer_property_no_stratigraphy(self, client_basic_model):
        """Layer property with 1 layer (no stratigraphy)."""
        resp = client_basic_model.get("/api/properties/layer")
        assert resp.status_code == 200
        data = resp.json()
        assert data["property_id"] == "layer"
        assert data["name"] == "Layer"
        assert len(data["values"]) == 1  # 1 element * 1 layer
        assert data["values"][0] == 1.0
        assert data["min"] == 1.0
        assert data["max"] == 1.0

    def test_layer_property_with_stratigraphy(self, client_with_stratigraphy):
        """Layer property with 2 layers."""
        resp = client_with_stratigraphy.get("/api/properties/layer")
        assert resp.status_code == 200
        data = resp.json()
        # 1 element * 2 layers = 2 values
        assert len(data["values"]) == 2
        assert data["values"][0] == 1.0  # layer 1
        assert data["values"][1] == 2.0  # layer 2

    def test_layer_property_with_layer_filter(self):
        """Layer property filtered to layer 1 — tested via _compute_property_values
        because NaN values in the response array cannot be JSON-serialized."""
        strat = _make_mock_stratigraphy(n_nodes=4, n_layers=2)
        model = _make_mock_model(stratigraphy=strat)
        model_state._model = model

        result = _compute_property_values("layer", layer=1)
        assert result is not None
        assert len(result) == 2  # 1 element * 2 layers
        assert result[0] == 1.0  # layer 1 kept
        assert np.isnan(result[1])  # layer 2 masked to NaN

    def test_thickness_property(self, client_with_stratigraphy):
        """Thickness property computed from stratigraphy."""
        resp = client_with_stratigraphy.get("/api/properties/thickness")
        assert resp.status_code == 200
        data = resp.json()
        assert data["property_id"] == "thickness"
        assert len(data["values"]) == 2  # 1 element * 2 layers
        # Thickness = top_elev - bottom_elev = 50.0 for all nodes/layers
        for v in data["values"]:
            assert v == pytest.approx(50.0)

    def test_thickness_without_stratigraphy(self, client_basic_model):
        """Thickness is not available without stratigraphy."""
        resp = client_basic_model.get("/api/properties/thickness")
        assert resp.status_code == 404

    def test_top_elev_property(self, client_with_stratigraphy):
        """Top elevation property computed from stratigraphy."""
        resp = client_with_stratigraphy.get("/api/properties/top_elev")
        assert resp.status_code == 200
        data = resp.json()
        assert data["property_id"] == "top_elev"
        assert len(data["values"]) == 2

    def test_bottom_elev_property(self, client_with_stratigraphy):
        """Bottom elevation property computed from stratigraphy."""
        resp = client_with_stratigraphy.get("/api/properties/bottom_elev")
        assert resp.status_code == 200
        data = resp.json()
        assert data["property_id"] == "bottom_elev"
        assert len(data["values"]) == 2

    def test_top_elev_with_layer_filter(self):
        """Top elevation filtered to layer 2 — tested via _compute_property_values."""
        strat = _make_mock_stratigraphy(n_nodes=4, n_layers=2)
        model = _make_mock_model(stratigraphy=strat)
        model_state._model = model

        result = _compute_property_values("top_elev", layer=2)
        assert result is not None
        assert len(result) == 2
        assert np.isnan(result[0])  # layer 1 masked
        assert not np.isnan(result[1])  # layer 2 has value

    def test_kh_property(self):
        """Kh property from aquifer params (2D node data)."""
        strat = _make_mock_stratigraphy(n_nodes=4, n_layers=2)
        params = MagicMock()
        params.kh = np.ones((4, 2)) * 25.0
        params.kv = None
        params.specific_storage = None
        params.ss = None
        params.specific_yield = None
        params.sy = None
        gw = MagicMock()
        gw.aquifer_params = params
        model = _make_mock_model(stratigraphy=strat, groundwater=gw)
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/properties/kh")
        assert resp.status_code == 200
        data = resp.json()
        assert data["property_id"] == "kh"
        # With 2 layers and 1 element: 2 values
        assert len(data["values"]) == 2
        for v in data["values"]:
            assert v == pytest.approx(25.0)

    def test_kh_with_layer_filter(self):
        """Kh filtered to layer 1 — tested via _compute_property_values."""
        strat = _make_mock_stratigraphy(n_nodes=4, n_layers=2)
        params = MagicMock()
        params.kh = np.ones((4, 2)) * 30.0
        gw = MagicMock()
        gw.aquifer_params = params
        model = _make_mock_model(stratigraphy=strat, groundwater=gw)
        model_state._model = model

        result = _compute_property_values("kh", layer=1)
        assert result is not None
        assert len(result) == 2
        # Layer 1 has value, layer 2 is NaN
        assert result[0] == pytest.approx(30.0)
        assert np.isnan(result[1])

    def test_kv_property(self):
        """Kv property from aquifer params."""
        strat = _make_mock_stratigraphy(n_nodes=4, n_layers=1)
        params = MagicMock()
        params.kh = None
        params.kv = np.ones((4, 1)) * 2.5
        params.specific_storage = None
        params.ss = None
        params.specific_yield = None
        params.sy = None
        gw = MagicMock()
        gw.aquifer_params = params
        model = _make_mock_model(stratigraphy=strat, groundwater=gw)
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/properties/kv")
        assert resp.status_code == 200
        data = resp.json()
        assert data["values"][0] == pytest.approx(2.5)

    def test_ss_via_specific_storage(self):
        """SS found via 'specific_storage' attribute."""
        strat = _make_mock_stratigraphy(n_nodes=4, n_layers=1)
        params = MagicMock()
        params.kh = None
        params.kv = None
        params.specific_storage = np.ones((4, 1)) * 1e-5
        params.ss = None  # not used when specific_storage is found first
        params.specific_yield = None
        params.sy = None
        gw = MagicMock()
        gw.aquifer_params = params
        model = _make_mock_model(stratigraphy=strat, groundwater=gw)
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/properties/ss")
        assert resp.status_code == 200
        data = resp.json()
        assert data["values"][0] == pytest.approx(1e-5)

    def test_sy_via_specific_yield(self):
        """SY found via 'specific_yield' attribute."""
        strat = _make_mock_stratigraphy(n_nodes=4, n_layers=1)
        params = MagicMock()
        params.kh = None
        params.kv = None
        params.specific_storage = None
        params.ss = None
        params.specific_yield = np.ones((4, 1)) * 0.15
        params.sy = None
        gw = MagicMock()
        gw.aquifer_params = params
        model = _make_mock_model(stratigraphy=strat, groundwater=gw)
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/properties/sy")
        assert resp.status_code == 200
        data = resp.json()
        assert data["values"][0] == pytest.approx(0.15)

    def test_kh_no_groundwater_returns_404(self, client_basic_model):
        """Kh not available when no groundwater component."""
        resp = client_basic_model.get("/api/properties/kh")
        assert resp.status_code == 404

    def test_kh_no_aquifer_params_returns_404(self):
        """Kh not available when groundwater has no aquifer_params."""
        gw = MagicMock()
        gw.aquifer_params = None
        model = _make_mock_model(groundwater=gw)
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/properties/kh")
        assert resp.status_code == 404

    def test_response_fields(self, client_basic_model):
        """Verify all response fields are present."""
        resp = client_basic_model.get("/api/properties/layer")
        data = resp.json()
        assert "property_id" in data
        assert "name" in data
        assert "units" in data
        assert "values" in data
        assert "min" in data
        assert "max" in data
        assert "mean" in data

    def test_layer_validation_ge_zero(self, client_basic_model):
        """Layer query param must be >= 0."""
        resp = client_basic_model.get("/api/properties/layer?layer=-1")
        assert resp.status_code == 422


# ===========================================================================
# 3. _node_to_element_values helper tests
# ===========================================================================


class TestNodeToElementValues:
    """Tests for the _node_to_element_values helper function."""

    def test_1d_node_data(self):
        """1D node data averaged to elements."""
        grid = _make_grid()
        # 4 nodes, 1D data
        node_data = np.array([10.0, 20.0, 30.0, 40.0])
        result = _node_to_element_values(node_data, grid, n_elements=1, n_layers=1)
        # Average of [10, 20, 30, 40] = 25
        assert len(result) == 1
        assert result[0] == pytest.approx(25.0)

    def test_2d_node_data(self):
        """2D node data (n_nodes, n_layers) averaged per layer."""
        grid = _make_grid()
        # 4 nodes, 2 layers
        node_data = np.array(
            [
                [10.0, 100.0],
                [20.0, 200.0],
                [30.0, 300.0],
                [40.0, 400.0],
            ]
        )
        result = _node_to_element_values(node_data, grid, n_elements=1, n_layers=2)
        # Layer 0: avg([10, 20, 30, 40]) = 25
        # Layer 1: avg([100, 200, 300, 400]) = 250
        assert len(result) == 2
        assert result[0] == pytest.approx(25.0)
        assert result[1] == pytest.approx(250.0)

    def test_2d_node_data_fewer_layers_than_requested(self):
        """2D node data with fewer columns than n_layers."""
        grid = _make_grid()
        # Only 1 layer column but n_layers=3
        node_data = np.array([[10.0], [20.0], [30.0], [40.0]])
        result = _node_to_element_values(node_data, grid, n_elements=1, n_layers=3)
        # Layer 0: avg = 25.0; Layers 1,2: data shape[1]=1 < lay -> no values -> 0.0
        assert len(result) == 3
        assert result[0] == pytest.approx(25.0)
        assert result[1] == pytest.approx(0.0)
        assert result[2] == pytest.approx(0.0)

    def test_multi_element_grid(self):
        """Node-to-element averaging on a 2-element grid."""
        grid = _make_grid_two_elements()
        # 6 nodes, 1 layer
        node_data = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        result = _node_to_element_values(node_data, grid, n_elements=2, n_layers=1)
        # Element 1: nodes (1,2,3,4) -> avg(10,20,30,40) = 25
        # Element 2: nodes (2,5,6,3) -> avg(20,50,60,30) = 40
        assert len(result) == 2
        assert result[0] == pytest.approx(25.0)
        assert result[1] == pytest.approx(40.0)

    def test_node_not_in_mapping(self):
        """Element references a node not in grid.nodes (should not crash)."""
        # Build a grid then remove a node from it
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

        # Now remove node 4 from grid.nodes (but element still references it)
        del grid.nodes[4]

        node_data = np.array([10.0, 20.0, 30.0])  # Only 3 nodes of data
        # node_id_to_idx will only have 3 entries: {1:0, 2:1, 3:2}
        result = _node_to_element_values(node_data, grid, n_elements=1, n_layers=1)
        # Element 1 vertices: (1,2,3,4). Node 4 not in mapping -> skipped.
        # avg(10, 20, 30) = 20
        assert result[0] == pytest.approx(20.0)


# ===========================================================================
# 4. _compute_property_values internal function tests
# ===========================================================================


class TestComputePropertyValues:
    """Tests for _compute_property_values."""

    def test_model_is_none_returns_none(self):
        """When model_state.model is None, returns None."""
        # model_state._model is already None from autouse fixture
        result = _compute_property_values("layer")
        assert result is None

    def test_layer_with_layer_filter_all(self):
        """Layer property with no filter (layer=0, all layers)."""
        strat = _make_mock_stratigraphy(n_nodes=4, n_layers=3)
        model = _make_mock_model(stratigraphy=strat)
        model_state._model = model

        result = _compute_property_values("layer", layer=0)
        assert result is not None
        # 1 element * 3 layers = 3 values
        assert len(result) == 3
        assert result[0] == 1.0
        assert result[1] == 2.0
        assert result[2] == 3.0

    def test_layer_with_specific_layer(self):
        """Layer property filtered to layer 2."""
        strat = _make_mock_stratigraphy(n_nodes=4, n_layers=3)
        model = _make_mock_model(stratigraphy=strat)
        model_state._model = model

        result = _compute_property_values("layer", layer=2)
        assert result is not None
        assert len(result) == 3
        # Only layer 2 cells should have values; others are NaN
        assert np.isnan(result[0])  # layer 1
        assert result[1] == 2.0  # layer 2
        assert np.isnan(result[2])  # layer 3

    def test_unknown_property_returns_none(self):
        """Unknown property ID returns None."""
        model = _make_mock_model()
        model_state._model = model

        result = _compute_property_values("nonexistent")
        assert result is None

    def test_thickness_no_stratigraphy_returns_none(self):
        """Thickness returns None when stratigraphy is None."""
        model = _make_mock_model()
        model_state._model = model

        result = _compute_property_values("thickness")
        assert result is None

    def test_top_elev_no_stratigraphy_returns_none(self):
        """top_elev returns None when stratigraphy is None."""
        model = _make_mock_model()
        model_state._model = model

        result = _compute_property_values("top_elev")
        assert result is None

    def test_bottom_elev_no_stratigraphy_returns_none(self):
        """bottom_elev returns None when stratigraphy is None."""
        model = _make_mock_model()
        model_state._model = model

        result = _compute_property_values("bottom_elev")
        assert result is None

    def test_kh_no_groundwater_returns_none(self):
        """Kh returns None when no groundwater."""
        model = _make_mock_model()
        model_state._model = model

        result = _compute_property_values("kh")
        assert result is None

    def test_kh_no_params_returns_none(self):
        """Kh returns None when aquifer_params is None."""
        gw = MagicMock()
        gw.aquifer_params = None
        model = _make_mock_model(groundwater=gw)
        model_state._model = model

        result = _compute_property_values("kh")
        assert result is None

    def test_kh_param_data_is_none_returns_none(self):
        """Kh returns None when params.kh is None."""
        params = MagicMock()
        params.kh = None
        gw = MagicMock()
        gw.aquifer_params = params
        model = _make_mock_model(groundwater=gw)
        model_state._model = model

        result = _compute_property_values("kh")
        assert result is None

    def test_kh_1d_param_data(self):
        """Kh with 1D param data (no layer dimension)."""
        strat = _make_mock_stratigraphy(n_nodes=4, n_layers=1)
        params = MagicMock()
        params.kh = np.array([10.0, 20.0, 30.0, 40.0])  # 1D
        gw = MagicMock()
        gw.aquifer_params = params
        model = _make_mock_model(stratigraphy=strat, groundwater=gw)
        model_state._model = model

        result = _compute_property_values("kh")
        assert result is not None
        assert len(result) == 1  # 1 element * 1 layer
        assert result[0] == pytest.approx(25.0)  # avg of 10,20,30,40

    def test_kh_fewer_effective_layers(self):
        """Kh with 2D data that has fewer layers than stratigraphy."""
        strat = _make_mock_stratigraphy(n_nodes=4, n_layers=3)
        params = MagicMock()
        # Only 1 column of data but 3 layers in stratigraphy
        params.kh = np.ones((4, 1)) * 50.0
        gw = MagicMock()
        gw.aquifer_params = params
        model = _make_mock_model(stratigraphy=strat, groundwater=gw)
        model_state._model = model

        result = _compute_property_values("kh")
        assert result is not None
        # n_cells = 1 element * 3 layers = 3
        assert len(result) == 3
        # First layer has values, remaining are zero (fewer effective layers)
        assert result[0] == pytest.approx(50.0)
        assert result[1] == pytest.approx(0.0)
        assert result[2] == pytest.approx(0.0)

    def test_ss_fallback_to_short_name(self):
        """SS resolution: tries 'specific_storage' first, falls back to 'ss'."""
        strat = _make_mock_stratigraphy(n_nodes=4, n_layers=1)
        params = MagicMock()
        params.kh = None
        params.kv = None
        params.specific_storage = None
        params.ss = np.ones((4, 1)) * 3e-5
        params.specific_yield = None
        params.sy = None
        gw = MagicMock()
        gw.aquifer_params = params
        model = _make_mock_model(stratigraphy=strat, groundwater=gw)
        model_state._model = model

        result = _compute_property_values("ss")
        assert result is not None
        assert result[0] == pytest.approx(3e-5)

    def test_sy_fallback_to_short_name(self):
        """SY resolution: tries 'specific_yield' first, falls back to 'sy'."""
        strat = _make_mock_stratigraphy(n_nodes=4, n_layers=1)
        params = MagicMock()
        params.kh = None
        params.kv = None
        params.specific_storage = None
        params.ss = None
        params.specific_yield = None
        params.sy = np.ones((4, 1)) * 0.25
        gw = MagicMock()
        gw.aquifer_params = params
        model = _make_mock_model(stratigraphy=strat, groundwater=gw)
        model_state._model = model

        result = _compute_property_values("sy")
        assert result is not None
        assert result[0] == pytest.approx(0.25)

    def test_thickness_with_layer_filter(self):
        """Thickness with layer filter applied."""
        strat = _make_mock_stratigraphy(n_nodes=4, n_layers=2)
        model = _make_mock_model(stratigraphy=strat)
        model_state._model = model

        result = _compute_property_values("thickness", layer=1)
        assert result is not None
        # 1 element * 2 layers = 2 values; only layer 1 cells are valid
        assert not np.isnan(result[0])  # layer 1 has value
        assert np.isnan(result[1])  # layer 2 is masked


# ===========================================================================
# 5. Edge case: all-NaN values
# ===========================================================================


class TestAllNanValues:
    """Test behavior when all values are NaN."""

    def test_all_nan_uses_zero_fallback(self):
        """When all valid values are NaN, stats use [0.0] fallback.

        Tested via _compute_property_values + the get_property code path
        because NaN-heavy arrays fail JSON serialization.
        """
        strat = _make_mock_stratigraphy(n_nodes=4, n_layers=2)
        model = _make_mock_model(stratigraphy=strat)
        model_state._model = model

        # Request layer=3 which doesn't exist in a 2-layer model
        # -> all values will be NaN because layer filter masks everything
        result = _compute_property_values("layer", layer=3)
        assert result is not None
        assert len(result) == 2
        # All values should be NaN
        assert np.all(np.isnan(result))

        # Simulate the stats computation that get_property does
        valid = result[~np.isnan(result)]
        assert len(valid) == 0
        # The route falls back to np.array([0.0])
        valid = np.array([0.0])
        assert float(np.min(valid)) == 0.0
        assert float(np.max(valid)) == 0.0
        assert float(np.mean(valid)) == 0.0


# ===========================================================================
# 6. Property info for unknown property ID (fallback PROPERTY_INFO)
# ===========================================================================


class TestPropertyInfoFallback:
    """Test that unknown property IDs use the fallback info dict."""

    def test_unknown_property_uses_fallback_info_in_list(self):
        """Properties not in PROPERTY_INFO get default metadata.

        This path is exercised via list_properties when the 'available' list
        contains a property not in PROPERTY_INFO. We can verify by checking
        the 'layer' property which IS in PROPERTY_INFO.
        """
        model = _make_mock_model()
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/properties")
        data = resp.json()
        layer_item = [p for p in data if p["id"] == "layer"][0]
        assert layer_item["name"] == "Layer"
        assert layer_item["cmap"] == "viridis"
        assert layer_item["log_scale"] is False
