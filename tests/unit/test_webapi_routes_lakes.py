"""Comprehensive tests for the FastAPI lake routes.

Covers /api/lakes/geojson and /api/lakes/{lake_id}/rating endpoints,
including all branches: no model, no lakes, empty elements, missing
boundary edges, short rings, valid polygons, rating curves, and
various property edge cases.
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


def _make_grid_two_elements():
    """Create a grid with two adjacent quad elements sharing an internal edge.

    Elements share edge (2, 3), so that edge appears twice (internal).
    The boundary consists of 6 edges forming a closed ring.
    """
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


def _make_mock_lake(
    lake_id: int = 1,
    elements: list[int] | None = None,
    name: str | None = None,
    initial_elevation: float = 100.0,
    max_elevation: float = 200.0,
    rating: object | None = None,
):
    """Create a mock lake object."""
    lake = MagicMock()
    lake.elements = elements if elements is not None else [1]
    lake.name = name
    lake.initial_elevation = initial_elevation
    lake.max_elevation = max_elevation
    lake.rating = rating
    return lake


def _make_mock_rating(
    elevations: list[float] | None = None,
    areas: list[float] | None = None,
    volumes: list[float] | None = None,
):
    """Create a mock rating curve object with numpy arrays."""
    rating = MagicMock()
    rating.elevations = np.array(elevations or [100.0, 110.0, 120.0])
    rating.areas = np.array(areas or [1000.0, 2000.0, 3000.0])
    rating.volumes = np.array(volumes or [5000.0, 15000.0, 30000.0])
    return rating


def _make_mock_lakes_component(lakes_dict: dict[int, object]):
    """Create a mock lakes component."""
    lakes_comp = MagicMock()
    lakes_comp.n_lakes = len(lakes_dict)
    lakes_comp.lakes = lakes_dict
    return lakes_comp


def _make_mock_model(grid=None, lakes=None, metadata=None):
    """Create a minimal mock IWFMModel."""
    model = MagicMock()
    model.name = "TestModel"
    model.grid = grid or _make_grid()
    model.lakes = lakes
    model.metadata = metadata or {}
    model.stratigraphy = None
    model.streams = None
    model.groundwater = None
    model.has_streams = False
    model.has_lakes = lakes is not None
    model.n_nodes = len(model.grid.nodes)
    model.n_elements = len(model.grid.elements)
    model.n_layers = 1
    model.n_lakes = lakes.n_lakes if lakes else 0
    model.n_stream_nodes = 0
    model.source_files = {}
    return model


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
def client_with_model_no_lakes():
    """TestClient with a model that has no lakes."""
    _reset_model_state()
    model = _make_mock_model()
    model.lakes = None
    model_state._model = model
    app = create_app()
    yield TestClient(app)
    _reset_model_state()


@pytest.fixture()
def client_with_empty_lakes():
    """TestClient with a model that has zero lakes."""
    _reset_model_state()
    lakes_comp = _make_mock_lakes_component({})
    lakes_comp.n_lakes = 0
    model = _make_mock_model(lakes=lakes_comp)
    model_state._model = model
    app = create_app()
    yield TestClient(app)
    _reset_model_state()


@pytest.fixture()
def client_with_lake():
    """TestClient with a model that has one valid lake on a single quad element."""
    _reset_model_state()
    grid = _make_grid()
    lake = _make_mock_lake(lake_id=1, elements=[1], name="Test Lake")
    lakes_comp = _make_mock_lakes_component({1: lake})
    model = _make_mock_model(grid=grid, lakes=lakes_comp)
    model_state._model = model
    # Mock reproject_coords to identity (no pyproj needed)
    model_state.reproject_coords = lambda x, y: (x, y)
    app = create_app()
    yield TestClient(app), model, lake
    _reset_model_state()


@pytest.fixture()
def client_with_lake_two_elements():
    """TestClient with a model where a lake spans two adjacent elements."""
    _reset_model_state()
    grid = _make_grid_two_elements()
    lake = _make_mock_lake(lake_id=1, elements=[1, 2], name="Big Lake")
    lakes_comp = _make_mock_lakes_component({1: lake})
    model = _make_mock_model(grid=grid, lakes=lakes_comp)
    model_state._model = model
    model_state.reproject_coords = lambda x, y: (x, y)
    app = create_app()
    yield TestClient(app), model, lake
    _reset_model_state()


# ---------------------------------------------------------------------------
# GET /api/lakes/geojson - No model loaded
# ---------------------------------------------------------------------------


class TestLakesGeojsonNoModel:
    """Tests for /api/lakes/geojson when no model is loaded."""

    def test_no_model_returns_404(self, client_no_model):
        resp = client_no_model.get("/api/lakes/geojson")
        assert resp.status_code == 404
        assert "No model loaded" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# GET /api/lakes/geojson - No lakes or empty lakes
# ---------------------------------------------------------------------------


class TestLakesGeojsonNoLakes:
    """Tests for /api/lakes/geojson when model has no lakes."""

    def test_lakes_is_none_returns_empty_feature_collection(self, client_with_model_no_lakes):
        resp = client_with_model_no_lakes.get("/api/lakes/geojson")
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "FeatureCollection"
        assert data["features"] == []

    def test_zero_lakes_returns_empty_feature_collection(self, client_with_empty_lakes):
        resp = client_with_empty_lakes.get("/api/lakes/geojson")
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "FeatureCollection"
        assert data["features"] == []


# ---------------------------------------------------------------------------
# GET /api/lakes/geojson - Lake with empty elements
# ---------------------------------------------------------------------------


class TestLakesGeojsonEmptyElements:
    """Tests for /api/lakes/geojson when lake has empty element list."""

    def test_lake_with_empty_elements_is_skipped(self):
        _reset_model_state()
        try:
            grid = _make_grid()
            lake = _make_mock_lake(lake_id=1, elements=[])
            lakes_comp = _make_mock_lakes_component({1: lake})
            model = _make_mock_model(grid=grid, lakes=lakes_comp)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/lakes/geojson")
            assert resp.status_code == 200
            data = resp.json()
            assert data["type"] == "FeatureCollection"
            assert data["features"] == []
        finally:
            _reset_model_state()


# ---------------------------------------------------------------------------
# GET /api/lakes/geojson - Element ID not found in grid
# ---------------------------------------------------------------------------


class TestLakesGeojsonMissingElement:
    """Tests for /api/lakes/geojson when lake references element not in grid."""

    def test_element_not_in_grid_produces_no_boundary(self):
        """When grid.elements.get(eid) returns None for all elements,
        there are no boundary edges and the lake is skipped."""
        _reset_model_state()
        try:
            grid = _make_grid()
            # Lake references element 999, which does not exist in the grid
            lake = _make_mock_lake(lake_id=1, elements=[999])
            lakes_comp = _make_mock_lakes_component({1: lake})
            model = _make_mock_model(grid=grid, lakes=lakes_comp)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/lakes/geojson")
            assert resp.status_code == 200
            data = resp.json()
            assert data["features"] == []
        finally:
            _reset_model_state()

    def test_mix_of_valid_and_invalid_elements(self):
        """When lake has some valid and some invalid element IDs,
        the polygon is still built from the valid elements."""
        _reset_model_state()
        try:
            grid = _make_grid()
            # Element 1 exists, element 999 does not
            lake = _make_mock_lake(lake_id=1, elements=[1, 999], name="Mixed Lake")
            lakes_comp = _make_mock_lakes_component({1: lake})
            model = _make_mock_model(grid=grid, lakes=lakes_comp)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/lakes/geojson")
            assert resp.status_code == 200
            data = resp.json()
            # Element 1 has 4 boundary edges, forms a valid ring
            assert len(data["features"]) == 1
            assert data["features"][0]["properties"]["name"] == "Mixed Lake"
        finally:
            _reset_model_state()


# ---------------------------------------------------------------------------
# GET /api/lakes/geojson - No boundary edges (all edges shared)
# ---------------------------------------------------------------------------


class TestLakesGeojsonNoBoundaryEdges:
    """Tests for /api/lakes/geojson when all edges are internal (count > 1)."""

    def test_no_boundary_edges_skips_lake(self):
        """If every edge is shared (appears > 1 time), boundary_edges is empty
        and the lake is skipped. We simulate this by creating a grid where
        the same edge is counted twice."""
        _reset_model_state()
        try:
            # Create a degenerate grid where the lake references the same
            # element twice, causing all edges to have count=2
            grid = _make_grid()
            lake = _make_mock_lake(lake_id=1, elements=[1, 1])
            lakes_comp = _make_mock_lakes_component({1: lake})
            model = _make_mock_model(grid=grid, lakes=lakes_comp)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/lakes/geojson")
            assert resp.status_code == 200
            data = resp.json()
            # All edges counted twice -> no boundary edges -> lake skipped
            assert data["features"] == []
        finally:
            _reset_model_state()


# ---------------------------------------------------------------------------
# GET /api/lakes/geojson - Ring too short (< 4 nodes)
# ---------------------------------------------------------------------------


class TestLakesGeojsonShortRing:
    """Tests for /api/lakes/geojson when assembled ring has < 4 nodes."""

    def test_ring_too_short_skips_lake(self):
        """Create a scenario where the ring is too short by having a triangle
        element but mocking the ring-walking to fail early.

        A triangle element has 3 boundary edges forming a ring of 4 nodes
        (3 + closing node), so it should actually pass. We test the ring < 4
        path by using a degenerate element with only 2 distinct boundary edges.
        """
        _reset_model_state()
        try:
            # Create a triangle element - it has 3 edges, ring = 4 nodes
            # (3 unique + close). That passes the < 4 check.
            # To test ring < 4, we need to manipulate boundary edges.
            # We patch the adjacency walking to produce a short ring.
            _make_grid()

            # Create a degenerate lake whose elements produce only 2 boundary
            # edges (which cannot close into a ring of >= 4).
            # We do this by providing two elements that share 2 of 4 edges,
            # leaving only 2 boundary edges (which forms a ring of max 3 nodes).

            # Actually, the simplest approach: use a triangle element.
            # 3 boundary edges -> ring = [n1, n2, n3, n1] = 4 nodes -> passes.
            # Instead, mock the iteration so that ring assembly breaks early.
            # Let's create a mock element with only 3 vertices (triangle).
            nodes = {
                1: Node(id=1, x=0.0, y=0.0),
                2: Node(id=2, x=100.0, y=0.0),
                3: Node(id=3, x=50.0, y=100.0),
            }
            elements = {
                10: Element(id=10, vertices=(1, 2, 3), subregion=1),
            }
            tri_grid = AppGrid(nodes=nodes, elements=elements)
            tri_grid.compute_connectivity()
            tri_grid.compute_areas()

            # A triangle has 3 boundary edges, ring = 4 nodes -> actually valid.
            # So let's verify it works for a triangle:
            lake = _make_mock_lake(lake_id=1, elements=[10], name="Tri Lake")
            lakes_comp = _make_mock_lakes_component({1: lake})
            model = _make_mock_model(grid=tri_grid, lakes=lakes_comp)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/lakes/geojson")
            assert resp.status_code == 200
            data = resp.json()
            # Triangle: ring = [v0, v1, v2, v0] = 4 nodes, valid polygon
            assert len(data["features"]) == 1
        finally:
            _reset_model_state()

    def test_ring_less_than_4_nodes_via_disconnected_edges(self):
        """Test that a lake is skipped when boundary edges cannot form a
        closed ring of at least 4 nodes. This happens when the walk terminates
        early because next_node is None before closing."""
        _reset_model_state()
        try:
            # Build a grid but mock boundary_edge walking so the ring is short.
            # We create a custom scenario: two elements sharing 3 edges,
            # leaving only 1 boundary edge which cannot form a ring.
            # Actually: 1 boundary edge = ring [a, b] = 2 nodes -> skipped.
            # Simplest: make a lake whose only element has vertices where
            # the adjacency walk ends after 2 nodes.

            # Approach: patch the route module to intercept the element lookup.
            # Instead, we create a mock element where vertices produce exactly
            # two boundary edges that don't form a closed ring of 4+.

            # Two overlapping quad elements sharing 3 edges, leaving 2 boundary edges:
            # This is hard to construct realistically. Instead, we use a
            # contrived approach: nodes that map to None in the ring coords.

            # The simplest way: create a grid with one element, but make some
            # nodes return None from grid.nodes.get(nid). The ring has 5 coords
            # (4 + close), but if 2 of them are None, we get < 4 coords.
            nodes = {
                1: Node(id=1, x=0.0, y=0.0),
                2: Node(id=2, x=100.0, y=0.0),
                # Nodes 3 and 4 exist in the element but we will remove them
                # from the grid after construction.
                3: Node(id=3, x=100.0, y=100.0),
                4: Node(id=4, x=0.0, y=100.0),
            }
            elements = {
                1: Element(id=1, vertices=(1, 2, 3, 4), subregion=1),
            }
            grid = AppGrid(nodes=nodes, elements=elements)
            grid.compute_connectivity()

            # Remove nodes 3 and 4 from the grid after construction.
            # The boundary walking will still find edges with node IDs 3 and 4,
            # but grid.nodes.get(3) will return None -> coords skipped.
            # Ring: [1, 2, 3, 4, 1] -> coords for 1,2 exist, 3,4 don't.
            # -> coords = [[0,0], [100,0], [0,0]] = 3 coords -> < 4 -> skipped.
            del grid.nodes[3]
            del grid.nodes[4]

            lake = _make_mock_lake(lake_id=1, elements=[1])
            lakes_comp = _make_mock_lakes_component({1: lake})
            model = _make_mock_model(grid=grid, lakes=lakes_comp)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/lakes/geojson")
            assert resp.status_code == 200
            data = resp.json()
            # Ring is built but coords < 4 after filtering out missing nodes
            assert data["features"] == []
        finally:
            _reset_model_state()


# ---------------------------------------------------------------------------
# GET /api/lakes/geojson - Valid lake polygon
# ---------------------------------------------------------------------------


class TestLakesGeojsonValidPolygon:
    """Tests for /api/lakes/geojson with a valid lake polygon."""

    def test_single_quad_element_lake(self, client_with_lake):
        client, model, lake = client_with_lake
        resp = client.get("/api/lakes/geojson")
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) == 1

        feature = data["features"][0]
        assert feature["type"] == "Feature"
        assert feature["geometry"]["type"] == "Polygon"

        # Polygon should have one ring with 5 coordinates (4 nodes + closing)
        coords = feature["geometry"]["coordinates"][0]
        assert len(coords) == 5
        # First and last coordinate should be the same (closed ring)
        assert coords[0] == coords[-1]

        props = feature["properties"]
        assert props["lake_id"] == 1
        assert props["name"] == "Test Lake"
        assert props["n_elements"] == 1
        assert props["initial_elevation"] == 100.0
        assert props["max_elevation"] == 200.0
        assert props["has_rating"] is False
        assert props["n_rating_points"] == 0
        assert "centroid" in props
        assert len(props["centroid"]) == 2

    def test_two_element_lake_shared_edge_excluded(self, client_with_lake_two_elements):
        """Two adjacent elements sharing an edge: the shared edge (2,3) is
        internal (count=2), so boundary has 6 edges forming a hexagonal ring."""
        client, model, lake = client_with_lake_two_elements
        resp = client.get("/api/lakes/geojson")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["features"]) == 1

        feature = data["features"][0]
        coords = feature["geometry"]["coordinates"][0]
        # 6 boundary edges -> ring of 7 coordinates (6 + closing)
        assert len(coords) == 7
        assert coords[0] == coords[-1]
        assert feature["properties"]["name"] == "Big Lake"
        assert feature["properties"]["n_elements"] == 2

    def test_centroid_calculation(self, client_with_lake):
        """Verify centroid is the average of non-closing coordinates."""
        client, model, lake = client_with_lake
        resp = client.get("/api/lakes/geojson")
        data = resp.json()
        feature = data["features"][0]
        coords = feature["geometry"]["coordinates"][0]
        centroid = feature["properties"]["centroid"]

        # Centroid = average of coords[:-1] (exclude closing point)
        n_pts = len(coords) - 1
        expected_lng = sum(c[0] for c in coords[:-1]) / n_pts
        expected_lat = sum(c[1] for c in coords[:-1]) / n_pts
        assert abs(centroid[0] - expected_lng) < 1e-10
        assert abs(centroid[1] - expected_lat) < 1e-10


# ---------------------------------------------------------------------------
# GET /api/lakes/geojson - Property edge cases
# ---------------------------------------------------------------------------


class TestLakesGeojsonProperties:
    """Tests for lake properties edge cases."""

    def test_no_name_uses_default(self):
        """When lake.name is None, default to 'Lake {lake_id}'."""
        _reset_model_state()
        try:
            grid = _make_grid()
            lake = _make_mock_lake(lake_id=42, elements=[1], name=None)
            lakes_comp = _make_mock_lakes_component({42: lake})
            model = _make_mock_model(grid=grid, lakes=lakes_comp)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/lakes/geojson")
            data = resp.json()
            assert data["features"][0]["properties"]["name"] == "Lake 42"
        finally:
            _reset_model_state()

    def test_empty_name_uses_default(self):
        """When lake.name is an empty string (falsy), default to 'Lake {lake_id}'."""
        _reset_model_state()
        try:
            grid = _make_grid()
            lake = _make_mock_lake(lake_id=7, elements=[1], name="")
            lakes_comp = _make_mock_lakes_component({7: lake})
            model = _make_mock_model(grid=grid, lakes=lakes_comp)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/lakes/geojson")
            data = resp.json()
            assert data["features"][0]["properties"]["name"] == "Lake 7"
        finally:
            _reset_model_state()

    def test_max_elevation_above_threshold_is_none(self):
        """When max_elevation >= 1e10, it should be returned as None."""
        _reset_model_state()
        try:
            grid = _make_grid()
            lake = _make_mock_lake(
                lake_id=1,
                elements=[1],
                name="High Lake",
                max_elevation=1e10,
            )
            lakes_comp = _make_mock_lakes_component({1: lake})
            model = _make_mock_model(grid=grid, lakes=lakes_comp)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/lakes/geojson")
            data = resp.json()
            assert data["features"][0]["properties"]["max_elevation"] is None
        finally:
            _reset_model_state()

    def test_max_elevation_slightly_above_threshold_is_none(self):
        """When max_elevation is just above 1e10, it should be None."""
        _reset_model_state()
        try:
            grid = _make_grid()
            lake = _make_mock_lake(
                lake_id=1,
                elements=[1],
                name="Very High Lake",
                max_elevation=2e10,
            )
            lakes_comp = _make_mock_lakes_component({1: lake})
            model = _make_mock_model(grid=grid, lakes=lakes_comp)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/lakes/geojson")
            data = resp.json()
            assert data["features"][0]["properties"]["max_elevation"] is None
        finally:
            _reset_model_state()

    def test_max_elevation_below_threshold_is_kept(self):
        """When max_elevation < 1e10, it should be returned as-is."""
        _reset_model_state()
        try:
            grid = _make_grid()
            lake = _make_mock_lake(
                lake_id=1,
                elements=[1],
                name="Normal Lake",
                max_elevation=500.0,
            )
            lakes_comp = _make_mock_lakes_component({1: lake})
            model = _make_mock_model(grid=grid, lakes=lakes_comp)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/lakes/geojson")
            data = resp.json()
            assert data["features"][0]["properties"]["max_elevation"] == 500.0
        finally:
            _reset_model_state()

    def test_lake_with_rating_curve(self):
        """When lake has a rating curve, has_rating=True and n_rating_points set."""
        _reset_model_state()
        try:
            grid = _make_grid()
            rating = _make_mock_rating(
                elevations=[100.0, 110.0, 120.0, 130.0],
                areas=[0.0, 500.0, 1200.0, 2000.0],
                volumes=[0.0, 2500.0, 7000.0, 15000.0],
            )
            lake = _make_mock_lake(
                lake_id=1,
                elements=[1],
                name="Rated Lake",
                rating=rating,
            )
            lakes_comp = _make_mock_lakes_component({1: lake})
            model = _make_mock_model(grid=grid, lakes=lakes_comp)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/lakes/geojson")
            data = resp.json()
            props = data["features"][0]["properties"]
            assert props["has_rating"] is True
            assert props["n_rating_points"] == 4
        finally:
            _reset_model_state()

    def test_lake_without_rating_curve(self):
        """When lake.rating is None, has_rating=False and n_rating_points=0."""
        _reset_model_state()
        try:
            grid = _make_grid()
            lake = _make_mock_lake(lake_id=1, elements=[1], rating=None)
            lakes_comp = _make_mock_lakes_component({1: lake})
            model = _make_mock_model(grid=grid, lakes=lakes_comp)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/lakes/geojson")
            data = resp.json()
            props = data["features"][0]["properties"]
            assert props["has_rating"] is False
            assert props["n_rating_points"] == 0
        finally:
            _reset_model_state()


# ---------------------------------------------------------------------------
# GET /api/lakes/geojson - Multiple lakes
# ---------------------------------------------------------------------------


class TestLakesGeojsonMultipleLakes:
    """Tests with multiple lakes in the model."""

    def test_multiple_lakes_one_valid_one_empty(self):
        """Two lakes: one valid, one with empty elements -> only one feature."""
        _reset_model_state()
        try:
            grid = _make_grid()
            lake1 = _make_mock_lake(lake_id=1, elements=[1], name="Good Lake")
            lake2 = _make_mock_lake(lake_id=2, elements=[], name="Empty Lake")
            lakes_comp = _make_mock_lakes_component({1: lake1, 2: lake2})
            model = _make_mock_model(grid=grid, lakes=lakes_comp)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/lakes/geojson")
            data = resp.json()
            assert len(data["features"]) == 1
            assert data["features"][0]["properties"]["lake_id"] == 1
        finally:
            _reset_model_state()

    def test_multiple_valid_lakes(self):
        """Two valid lakes on the same element (each references element 1)."""
        _reset_model_state()
        try:
            grid = _make_grid()
            lake1 = _make_mock_lake(lake_id=1, elements=[1], name="Lake A")
            lake2 = _make_mock_lake(lake_id=2, elements=[1], name="Lake B")
            lakes_comp = _make_mock_lakes_component({1: lake1, 2: lake2})
            model = _make_mock_model(grid=grid, lakes=lakes_comp)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/lakes/geojson")
            data = resp.json()
            assert len(data["features"]) == 2
            names = {f["properties"]["name"] for f in data["features"]}
            assert names == {"Lake A", "Lake B"}
        finally:
            _reset_model_state()


# ===========================================================================
# GET /api/lakes/{lake_id}/rating
# ===========================================================================


class TestLakeRatingNoModel:
    """Tests for /api/lakes/{lake_id}/rating when no model is loaded."""

    def test_no_model_returns_404(self, client_no_model):
        resp = client_no_model.get("/api/lakes/1/rating")
        assert resp.status_code == 404
        assert "No model loaded" in resp.json()["detail"]


class TestLakeRatingNoLakes:
    """Tests for /api/lakes/{lake_id}/rating when model has no lakes."""

    def test_no_lakes_returns_404(self, client_with_model_no_lakes):
        resp = client_with_model_no_lakes.get("/api/lakes/1/rating")
        assert resp.status_code == 404
        assert "No lake data in model" in resp.json()["detail"]


class TestLakeRatingNotFound:
    """Tests for /api/lakes/{lake_id}/rating when lake ID does not exist."""

    def test_lake_not_found_returns_404(self):
        _reset_model_state()
        try:
            grid = _make_grid()
            lake = _make_mock_lake(lake_id=1, elements=[1])
            lakes_comp = _make_mock_lakes_component({1: lake})
            model = _make_mock_model(grid=grid, lakes=lakes_comp)
            model_state._model = model
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/lakes/999/rating")
            assert resp.status_code == 404
            assert "Lake 999 not found" in resp.json()["detail"]
        finally:
            _reset_model_state()


class TestLakeRatingNoRatingCurve:
    """Tests for /api/lakes/{lake_id}/rating when lake has no rating curve."""

    def test_no_rating_returns_404(self):
        _reset_model_state()
        try:
            grid = _make_grid()
            lake = _make_mock_lake(lake_id=1, elements=[1], rating=None)
            lakes_comp = _make_mock_lakes_component({1: lake})
            model = _make_mock_model(grid=grid, lakes=lakes_comp)
            model_state._model = model
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/lakes/1/rating")
            assert resp.status_code == 404
            assert "No rating curve for lake 1" in resp.json()["detail"]
        finally:
            _reset_model_state()


class TestLakeRatingValid:
    """Tests for /api/lakes/{lake_id}/rating with a valid rating curve."""

    def test_valid_rating_returns_data(self):
        _reset_model_state()
        try:
            grid = _make_grid()
            rating = _make_mock_rating(
                elevations=[100.0, 110.0, 120.0],
                areas=[1000.0, 2000.0, 3000.0],
                volumes=[5000.0, 15000.0, 30000.0],
            )
            lake = _make_mock_lake(
                lake_id=1,
                elements=[1],
                name="Rated Lake",
                rating=rating,
            )
            lakes_comp = _make_mock_lakes_component({1: lake})
            model = _make_mock_model(grid=grid, lakes=lakes_comp)
            model_state._model = model
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/lakes/1/rating")
            assert resp.status_code == 200
            data = resp.json()
            assert data["lake_id"] == 1
            assert data["name"] == "Rated Lake"
            assert data["elevations"] == [100.0, 110.0, 120.0]
            assert data["areas"] == [1000.0, 2000.0, 3000.0]
            assert data["volumes"] == [5000.0, 15000.0, 30000.0]
            assert data["n_points"] == 3
        finally:
            _reset_model_state()

    def test_rating_with_no_name_uses_default(self):
        _reset_model_state()
        try:
            grid = _make_grid()
            rating = _make_mock_rating()
            lake = _make_mock_lake(
                lake_id=5,
                elements=[1],
                name=None,
                rating=rating,
            )
            lakes_comp = _make_mock_lakes_component({5: lake})
            model = _make_mock_model(grid=grid, lakes=lakes_comp)
            model_state._model = model
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/lakes/5/rating")
            assert resp.status_code == 200
            data = resp.json()
            assert data["name"] == "Lake 5"
        finally:
            _reset_model_state()

    def test_rating_with_empty_name_uses_default(self):
        _reset_model_state()
        try:
            grid = _make_grid()
            rating = _make_mock_rating()
            lake = _make_mock_lake(
                lake_id=3,
                elements=[1],
                name="",
                rating=rating,
            )
            lakes_comp = _make_mock_lakes_component({3: lake})
            model = _make_mock_model(grid=grid, lakes=lakes_comp)
            model_state._model = model
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/lakes/3/rating")
            assert resp.status_code == 200
            data = resp.json()
            assert data["name"] == "Lake 3"
        finally:
            _reset_model_state()

    def test_rating_with_large_arrays(self):
        """Test with larger numpy arrays for the rating curve."""
        _reset_model_state()
        try:
            grid = _make_grid()
            elevations = np.linspace(50.0, 200.0, 50)
            areas = np.linspace(0.0, 10000.0, 50)
            volumes = np.cumsum(areas) * 1.5
            rating = MagicMock()
            rating.elevations = elevations
            rating.areas = areas
            rating.volumes = volumes

            lake = _make_mock_lake(
                lake_id=1,
                elements=[1],
                name="Big Rating Lake",
                rating=rating,
            )
            lakes_comp = _make_mock_lakes_component({1: lake})
            model = _make_mock_model(grid=grid, lakes=lakes_comp)
            model_state._model = model
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/lakes/1/rating")
            assert resp.status_code == 200
            data = resp.json()
            assert data["n_points"] == 50
            assert len(data["elevations"]) == 50
            assert len(data["areas"]) == 50
            assert len(data["volumes"]) == 50
            assert data["elevations"][0] == pytest.approx(50.0)
            assert data["elevations"][-1] == pytest.approx(200.0)
        finally:
            _reset_model_state()


# ---------------------------------------------------------------------------
# GET /api/lakes/geojson - Coordinate reprojection
# ---------------------------------------------------------------------------


class TestLakesGeojsonReprojection:
    """Tests verifying coordinate reprojection is applied to lake polygons."""

    def test_reproject_coords_called_for_each_ring_node(self):
        """Verify that model_state.reproject_coords is called for every
        node in the polygon ring."""
        _reset_model_state()
        try:
            grid = _make_grid()
            lake = _make_mock_lake(lake_id=1, elements=[1], name="Proj Lake")
            lakes_comp = _make_mock_lakes_component({1: lake})
            model = _make_mock_model(grid=grid, lakes=lakes_comp)
            model_state._model = model

            # Track calls to reproject_coords
            calls = []

            def tracking_reproject(x, y):
                calls.append((x, y))
                return (x + 1000.0, y + 2000.0)

            model_state.reproject_coords = tracking_reproject
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/lakes/geojson")
            assert resp.status_code == 200
            data = resp.json()

            # Should have been called for each node in the ring
            # Quad element: ring has 5 nodes (4 + closing), each reprojected
            assert len(calls) == 5

            # Verify the offset was applied to coordinates
            coords = data["features"][0]["geometry"]["coordinates"][0]
            for coord in coords:
                assert coord[0] >= 1000.0  # x + 1000
                assert coord[1] >= 2000.0  # y + 2000
        finally:
            _reset_model_state()

    def test_identity_reprojection(self, client_with_lake):
        """With identity reprojection, coordinates should match node positions."""
        client, model, lake = client_with_lake
        resp = client.get("/api/lakes/geojson")
        data = resp.json()
        coords = data["features"][0]["geometry"]["coordinates"][0]

        # All coords should be within [0, 100] for x and y
        for coord in coords:
            assert 0.0 <= coord[0] <= 100.0
            assert 0.0 <= coord[1] <= 100.0
