"""Comprehensive tests for the FastAPI small watershed routes.

Covers GET /api/small-watersheds endpoint, including all branches:
no model loaded, no small_watersheds component, empty watersheds,
GW node lookup, baseflow vs percolation nodes, destination stream
node coordinates, missing nodes, and coordinate reprojection.
"""

from __future__ import annotations

from unittest.mock import MagicMock

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


def _make_mock_gw_node(
    gw_node_id: int = 1,
    is_baseflow: bool = False,
    layer: int = 0,
    max_perc_rate: float = 0.5,
):
    """Create a mock WatershedGWNode object."""
    gwn = MagicMock()
    gwn.gw_node_id = gw_node_id
    gwn.is_baseflow = is_baseflow
    gwn.layer = layer
    gwn.max_perc_rate = max_perc_rate
    return gwn


def _make_mock_watershed(
    ws_id: int = 1,
    area: float = 1000.0,
    dest_stream_node: int = 0,
    gw_nodes: list | None = None,
    curve_number: float = 75.0,
    wilting_point: float = 0.1,
    field_capacity: float = 0.3,
    total_porosity: float = 0.4,
    lambda_param: float = 0.5,
    root_depth: float = 3.0,
    hydraulic_cond: float = 0.01,
    kunsat_method: int = 1,
    gw_threshold: float = 10.0,
    max_gw_storage: float = 100.0,
    surface_flow_coeff: float = 0.05,
    baseflow_coeff: float = 0.02,
):
    """Create a mock WatershedUnit object."""
    ws = MagicMock()
    ws.id = ws_id
    ws.area = area
    ws.dest_stream_node = dest_stream_node
    ws.gw_nodes = gw_nodes if gw_nodes is not None else []
    ws.curve_number = curve_number
    ws.wilting_point = wilting_point
    ws.field_capacity = field_capacity
    ws.total_porosity = total_porosity
    ws.lambda_param = lambda_param
    ws.root_depth = root_depth
    ws.hydraulic_cond = hydraulic_cond
    ws.kunsat_method = kunsat_method
    ws.gw_threshold = gw_threshold
    ws.max_gw_storage = max_gw_storage
    ws.surface_flow_coeff = surface_flow_coeff
    ws.baseflow_coeff = baseflow_coeff
    return ws


def _make_mock_small_watersheds(watersheds_list: list):
    """Create a mock AppSmallWatershed component."""
    comp = MagicMock()
    comp.iter_watersheds.return_value = watersheds_list
    return comp


def _make_mock_model(grid=None, small_watersheds=None, streams=None):
    """Create a minimal mock IWFMModel."""
    model = MagicMock()
    model.name = "TestModel"
    model.grid = grid or _make_grid()
    model.small_watersheds = small_watersheds
    model.streams = streams
    model.metadata = {}
    model.stratigraphy = None
    model.groundwater = None
    model.lakes = None
    model.has_streams = streams is not None
    model.has_lakes = False
    model.n_nodes = len(model.grid.nodes)
    model.n_elements = len(model.grid.elements)
    model.n_layers = 1
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
def client_no_small_watersheds():
    """TestClient with a model that has no small_watersheds component."""
    model = _make_mock_model()
    model.small_watersheds = None
    model_state._model = model
    app = create_app()
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /api/small-watersheds - No model loaded
# ---------------------------------------------------------------------------


class TestSmallWatershedsNoModel:
    """Tests when no model is loaded."""

    def test_no_model_returns_404(self, client_no_model):
        resp = client_no_model.get("/api/small-watersheds")
        assert resp.status_code == 404
        assert "No model loaded" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# GET /api/small-watersheds - No small watersheds component
# ---------------------------------------------------------------------------


class TestSmallWatershedsNone:
    """Tests when model has no small_watersheds component (None)."""

    def test_no_small_watersheds_returns_zero(self, client_no_small_watersheds):
        resp = client_no_small_watersheds.get("/api/small-watersheds")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_watersheds"] == 0
        assert data["watersheds"] == []


# ---------------------------------------------------------------------------
# GET /api/small-watersheds - Empty watersheds list
# ---------------------------------------------------------------------------


class TestSmallWatershedsEmpty:
    """Tests when small_watersheds component has no watershed units."""

    def test_empty_watersheds_list(self):
        sw_comp = _make_mock_small_watersheds([])
        model = _make_mock_model(small_watersheds=sw_comp)
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/small-watersheds")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_watersheds"] == 0
        assert data["watersheds"] == []


# ---------------------------------------------------------------------------
# GET /api/small-watersheds - GW node not found in grid
# ---------------------------------------------------------------------------


class TestSmallWatershedsGWNodeNotInGrid:
    """Tests when GW node referenced by watershed is not in the grid."""

    def test_gw_node_not_in_grid_skipped(self):
        """If grid.nodes.get(gw_node_id) returns None, the node is skipped."""
        gwn = _make_mock_gw_node(gw_node_id=999)  # 999 not in grid
        ws = _make_mock_watershed(ws_id=1, gw_nodes=[gwn])
        sw_comp = _make_mock_small_watersheds([ws])
        model = _make_mock_model(small_watersheds=sw_comp)
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/small-watersheds")
        assert resp.status_code == 200
        data = resp.json()
        # All GW nodes missing -> gw_coords empty -> watershed skipped
        assert data["n_watersheds"] == 0
        assert data["watersheds"] == []

    def test_mix_valid_and_invalid_gw_nodes(self):
        """Watershed with some valid and some invalid GW node IDs."""
        gwn_valid = _make_mock_gw_node(gw_node_id=1)  # node 1 exists
        gwn_invalid = _make_mock_gw_node(gw_node_id=999)  # node 999 does not exist
        ws = _make_mock_watershed(ws_id=1, gw_nodes=[gwn_valid, gwn_invalid])
        sw_comp = _make_mock_small_watersheds([ws])
        model = _make_mock_model(small_watersheds=sw_comp)
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/small-watersheds")
        assert resp.status_code == 200
        data = resp.json()
        # At least one valid GW node -> watershed included
        assert data["n_watersheds"] == 1
        ws_data = data["watersheds"][0]
        # Only 1 of 2 GW nodes valid
        assert ws_data["n_gw_nodes"] == 1
        assert ws_data["gw_nodes"][0]["node_id"] == 1


# ---------------------------------------------------------------------------
# GET /api/small-watersheds - Baseflow vs percolation nodes
# ---------------------------------------------------------------------------


class TestSmallWatershedsBaseflowPercolation:
    """Tests for baseflow node (raw_qmaxwb = -layer) vs percolation node."""

    def test_baseflow_node_raw_qmaxwb(self):
        """Baseflow node: raw_qmaxwb = -layer."""
        gwn = _make_mock_gw_node(gw_node_id=1, is_baseflow=True, layer=3, max_perc_rate=0.0)
        ws = _make_mock_watershed(ws_id=1, gw_nodes=[gwn])
        sw_comp = _make_mock_small_watersheds([ws])
        model = _make_mock_model(small_watersheds=sw_comp)
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/small-watersheds")
        data = resp.json()
        gw = data["watersheds"][0]["gw_nodes"][0]
        assert gw["is_baseflow"] is True
        assert gw["raw_qmaxwb"] == -3.0  # -layer
        assert gw["layer"] == 3

    def test_percolation_node_raw_qmaxwb(self):
        """Percolation node: raw_qmaxwb = max_perc_rate."""
        gwn = _make_mock_gw_node(gw_node_id=2, is_baseflow=False, layer=0, max_perc_rate=1.5)
        ws = _make_mock_watershed(ws_id=1, gw_nodes=[gwn])
        sw_comp = _make_mock_small_watersheds([ws])
        model = _make_mock_model(small_watersheds=sw_comp)
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/small-watersheds")
        data = resp.json()
        gw = data["watersheds"][0]["gw_nodes"][0]
        assert gw["is_baseflow"] is False
        assert gw["raw_qmaxwb"] == 1.5  # max_perc_rate
        assert gw["max_perc_rate"] == 1.5


# ---------------------------------------------------------------------------
# GET /api/small-watersheds - Destination stream node coordinates
# ---------------------------------------------------------------------------


class TestSmallWatershedsDestCoords:
    """Tests for destination stream node coordinate lookup."""

    def test_no_streams_dest_coords_none(self):
        """When model.streams is None, dest_coords is None."""
        gwn = _make_mock_gw_node(gw_node_id=1)
        ws = _make_mock_watershed(ws_id=1, gw_nodes=[gwn], dest_stream_node=5)
        sw_comp = _make_mock_small_watersheds([ws])
        model = _make_mock_model(small_watersheds=sw_comp, streams=None)
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/small-watersheds")
        data = resp.json()
        assert data["watersheds"][0]["dest_coords"] is None

    def test_dest_stream_node_zero(self):
        """When dest_stream_node <= 0, dest_coords is None."""
        gwn = _make_mock_gw_node(gw_node_id=1)
        ws = _make_mock_watershed(ws_id=1, gw_nodes=[gwn], dest_stream_node=0)
        sw_comp = _make_mock_small_watersheds([ws])
        streams = MagicMock()
        streams.nodes = {}
        model = _make_mock_model(small_watersheds=sw_comp, streams=streams)
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/small-watersheds")
        data = resp.json()
        assert data["watersheds"][0]["dest_coords"] is None

    def test_dest_stream_node_not_found(self):
        """When dest_stream_node exists but stream_nodes dict does not have it."""
        gwn = _make_mock_gw_node(gw_node_id=1)
        ws = _make_mock_watershed(ws_id=1, gw_nodes=[gwn], dest_stream_node=99)
        sw_comp = _make_mock_small_watersheds([ws])
        streams = MagicMock()
        streams.nodes = {}  # empty -> lookup returns None
        model = _make_mock_model(small_watersheds=sw_comp, streams=streams)
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/small-watersheds")
        data = resp.json()
        assert data["watersheds"][0]["dest_coords"] is None

    def test_dest_stream_node_found_but_gw_node_not_in_grid(self):
        """Stream node found but its gw_node is not in the grid."""
        gwn = _make_mock_gw_node(gw_node_id=1)
        ws = _make_mock_watershed(ws_id=1, gw_nodes=[gwn], dest_stream_node=5)
        sw_comp = _make_mock_small_watersheds([ws])

        sn = MagicMock()
        sn.gw_node = 999  # not in grid
        streams = MagicMock()
        streams.nodes = {5: sn}

        model = _make_mock_model(small_watersheds=sw_comp, streams=streams)
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/small-watersheds")
        data = resp.json()
        assert data["watersheds"][0]["dest_coords"] is None

    def test_dest_stream_node_gw_node_none(self):
        """Stream node found but its gw_node attribute is None/0."""
        gwn = _make_mock_gw_node(gw_node_id=1)
        ws = _make_mock_watershed(ws_id=1, gw_nodes=[gwn], dest_stream_node=5)
        sw_comp = _make_mock_small_watersheds([ws])

        sn = MagicMock()
        sn.gw_node = 0  # falsy
        streams = MagicMock()
        streams.nodes = {5: sn}

        model = _make_mock_model(small_watersheds=sw_comp, streams=streams)
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/small-watersheds")
        data = resp.json()
        assert data["watersheds"][0]["dest_coords"] is None

    def test_dest_stream_node_with_valid_gw_node(self):
        """Stream node found and its gw_node is in the grid -> dest_coords populated."""
        gwn = _make_mock_gw_node(gw_node_id=1)
        ws = _make_mock_watershed(ws_id=1, gw_nodes=[gwn], dest_stream_node=5)
        sw_comp = _make_mock_small_watersheds([ws])

        sn = MagicMock()
        sn.gw_node = 3  # node 3 is at (100, 100)
        streams = MagicMock()
        streams.nodes = {5: sn}

        model = _make_mock_model(small_watersheds=sw_comp, streams=streams)
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/small-watersheds")
        data = resp.json()
        dest = data["watersheds"][0]["dest_coords"]
        assert dest is not None
        assert dest["lng"] == 100.0
        assert dest["lat"] == 100.0

    def test_streams_without_nodes_attr(self):
        """model.streams exists but has no 'nodes' attribute (getattr fallback)."""
        gwn = _make_mock_gw_node(gw_node_id=1)
        ws = _make_mock_watershed(ws_id=1, gw_nodes=[gwn], dest_stream_node=5)
        sw_comp = _make_mock_small_watersheds([ws])

        # Use spec=[] to strip all attributes
        streams = MagicMock(spec=[])
        model = _make_mock_model(small_watersheds=sw_comp, streams=streams)
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/small-watersheds")
        data = resp.json()
        # getattr(streams, "nodes", None) returns None -> {} -> lookup None
        assert data["watersheds"][0]["dest_coords"] is None


# ---------------------------------------------------------------------------
# GET /api/small-watersheds - Full valid response
# ---------------------------------------------------------------------------


class TestSmallWatershedsValidResponse:
    """Tests for a fully valid small watershed response."""

    def test_single_watershed_all_fields(self):
        """Verify all fields are returned for a complete watershed."""
        gwn1 = _make_mock_gw_node(gw_node_id=1, is_baseflow=True, layer=2)
        gwn2 = _make_mock_gw_node(gw_node_id=2, is_baseflow=False, max_perc_rate=0.8)
        ws = _make_mock_watershed(
            ws_id=42,
            area=5000.0,
            dest_stream_node=0,
            gw_nodes=[gwn1, gwn2],
            curve_number=80.0,
            wilting_point=0.15,
            field_capacity=0.35,
            total_porosity=0.45,
            lambda_param=0.6,
            root_depth=4.0,
            hydraulic_cond=0.02,
            kunsat_method=2,
            gw_threshold=15.0,
            max_gw_storage=200.0,
            surface_flow_coeff=0.06,
            baseflow_coeff=0.03,
        )
        sw_comp = _make_mock_small_watersheds([ws])
        model = _make_mock_model(small_watersheds=sw_comp)
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/small-watersheds")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_watersheds"] == 1

        ws_data = data["watersheds"][0]
        assert ws_data["id"] == 42
        assert ws_data["area"] == 5000.0
        assert ws_data["dest_stream_node"] == 0
        assert ws_data["dest_coords"] is None
        assert ws_data["n_gw_nodes"] == 2

        # Marker position = first GW node's coordinates
        # Node 1 is at (0.0, 0.0)
        assert ws_data["marker_position"] == [0.0, 0.0]

        # Root zone parameters
        assert ws_data["curve_number"] == 80.0
        assert ws_data["wilting_point"] == 0.15
        assert ws_data["field_capacity"] == 0.35
        assert ws_data["total_porosity"] == 0.45
        assert ws_data["lambda_param"] == 0.6
        assert ws_data["root_depth"] == 4.0
        assert ws_data["hydraulic_cond"] == 0.02
        assert ws_data["kunsat_method"] == 2

        # Aquifer parameters
        assert ws_data["gw_threshold"] == 15.0
        assert ws_data["max_gw_storage"] == 200.0
        assert ws_data["surface_flow_coeff"] == 0.06
        assert ws_data["baseflow_coeff"] == 0.03

        # GW nodes data
        gw1 = ws_data["gw_nodes"][0]
        assert gw1["node_id"] == 1
        assert gw1["is_baseflow"] is True
        assert gw1["raw_qmaxwb"] == -2.0

        gw2 = ws_data["gw_nodes"][1]
        assert gw2["node_id"] == 2
        assert gw2["is_baseflow"] is False
        assert gw2["raw_qmaxwb"] == 0.8

    def test_multiple_watersheds(self):
        """Multiple watersheds are all returned."""
        gwn1 = _make_mock_gw_node(gw_node_id=1)
        gwn2 = _make_mock_gw_node(gw_node_id=2)
        ws1 = _make_mock_watershed(ws_id=1, gw_nodes=[gwn1])
        ws2 = _make_mock_watershed(ws_id=2, gw_nodes=[gwn2])
        sw_comp = _make_mock_small_watersheds([ws1, ws2])
        model = _make_mock_model(small_watersheds=sw_comp)
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/small-watersheds")
        data = resp.json()
        assert data["n_watersheds"] == 2
        ids = [w["id"] for w in data["watersheds"]]
        assert 1 in ids
        assert 2 in ids

    def test_watershed_skipped_if_all_gw_nodes_invalid(self):
        """Watershed is skipped when all GW nodes are not in grid (gw_coords empty)."""
        gwn1 = _make_mock_gw_node(gw_node_id=888)
        gwn2 = _make_mock_gw_node(gw_node_id=999)
        ws1 = _make_mock_watershed(ws_id=1, gw_nodes=[gwn1, gwn2])
        # ws2 has valid node
        gwn3 = _make_mock_gw_node(gw_node_id=1)
        ws2 = _make_mock_watershed(ws_id=2, gw_nodes=[gwn3])
        sw_comp = _make_mock_small_watersheds([ws1, ws2])
        model = _make_mock_model(small_watersheds=sw_comp)
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x, y)
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/small-watersheds")
        data = resp.json()
        # ws1 skipped (no valid GW nodes), ws2 included
        assert data["n_watersheds"] == 1
        assert data["watersheds"][0]["id"] == 2


# ---------------------------------------------------------------------------
# GET /api/small-watersheds - Coordinate reprojection
# ---------------------------------------------------------------------------


class TestSmallWatershedsReprojection:
    """Tests verifying coordinate reprojection is applied."""

    def test_reproject_coords_applied_to_gw_nodes(self):
        """Verify reproject_coords is called for each GW node."""
        calls = []

        def tracking_reproject(x, y):
            calls.append((x, y))
            return (x + 1000.0, y + 2000.0)

        gwn1 = _make_mock_gw_node(gw_node_id=1)
        gwn2 = _make_mock_gw_node(gw_node_id=3)
        ws = _make_mock_watershed(ws_id=1, gw_nodes=[gwn1, gwn2])
        sw_comp = _make_mock_small_watersheds([ws])
        model = _make_mock_model(small_watersheds=sw_comp)
        model_state._model = model
        model_state.reproject_coords = tracking_reproject
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/small-watersheds")
        assert resp.status_code == 200
        data = resp.json()

        # Should have 2 calls for 2 valid GW nodes
        assert len(calls) == 2

        # Verify reprojection offset
        gw1 = data["watersheds"][0]["gw_nodes"][0]
        assert gw1["lng"] == 1000.0  # 0.0 + 1000.0
        assert gw1["lat"] == 2000.0  # 0.0 + 2000.0

    def test_reproject_applied_to_dest_coords(self):
        """Verify reproject_coords is called for dest stream node coords."""
        gwn = _make_mock_gw_node(gw_node_id=1)
        ws = _make_mock_watershed(ws_id=1, gw_nodes=[gwn], dest_stream_node=5)
        sw_comp = _make_mock_small_watersheds([ws])

        sn = MagicMock()
        sn.gw_node = 3  # node 3 is at (100, 100)
        streams = MagicMock()
        streams.nodes = {5: sn}

        model = _make_mock_model(small_watersheds=sw_comp, streams=streams)
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x + 500.0, y + 600.0)
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/small-watersheds")
        data = resp.json()
        dest = data["watersheds"][0]["dest_coords"]
        assert dest is not None
        assert dest["lng"] == 600.0  # 100.0 + 500.0
        assert dest["lat"] == 700.0  # 100.0 + 600.0

    def test_marker_position_uses_reprojected_coords(self):
        """Marker position uses reprojected coordinates of first GW node."""
        gwn = _make_mock_gw_node(gw_node_id=2)  # node 2 at (100, 0)
        ws = _make_mock_watershed(ws_id=1, gw_nodes=[gwn])
        sw_comp = _make_mock_small_watersheds([ws])
        model = _make_mock_model(small_watersheds=sw_comp)
        model_state._model = model
        model_state.reproject_coords = lambda x, y: (x * 2, y * 3)
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/small-watersheds")
        data = resp.json()
        # Node 2 at (100, 0), reprojected to (200, 0)
        assert data["watersheds"][0]["marker_position"] == [200.0, 0.0]
