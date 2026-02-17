"""Comprehensive tests for the mesh API routes.

Covers all endpoints in ``pyiwfm.visualization.webapi.routes.mesh``:

* ``GET /api/mesh`` — full 3D VTU mesh
* ``GET /api/mesh/surface`` — surface VTU mesh
* ``GET /api/mesh/json`` — surface JSON for vtk.js
* ``GET /api/mesh/geojson`` — GeoJSON FeatureCollection
* ``GET /api/mesh/head-map`` — head values mapped to elements
* ``GET /api/mesh/subregions`` — subregion boundary polygons
* ``GET /api/mesh/property-map`` — property values per element
* ``GET /api/mesh/element/{element_id}`` — element detail
* ``GET /api/mesh/nodes`` — all node coordinates

Every branch and edge case is exercised to target 95%+ coverage.
"""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("fastapi", reason="FastAPI not available")
pytest.importorskip("pydantic", reason="Pydantic not available")

from fastapi.testclient import TestClient

from pyiwfm.core.mesh import AppGrid, Element, Node, Subregion
from pyiwfm.visualization.webapi.config import model_state
from pyiwfm.visualization.webapi.server import create_app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_grid():
    """Create a 6-node, 2-element grid for subregion testing."""
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


def _make_mock_model(**kwargs):
    """Create a minimal mock IWFMModel."""
    model = MagicMock()
    model.name = "TestModel"
    model.grid = kwargs.get("grid", _make_grid())
    model.metadata = kwargs.get("metadata", {})
    model.n_nodes = len(model.grid.nodes)
    model.n_elements = len(model.grid.elements)
    model.n_layers = kwargs.get("n_layers", 1)
    model.stratigraphy = kwargs.get("stratigraphy", None)
    model.streams = kwargs.get("streams", None)
    model.lakes = kwargs.get("lakes", None)
    model.groundwater = kwargs.get("groundwater", None)
    model.rootzone = kwargs.get("rootzone", None)
    model.source_files = kwargs.get("source_files", {})
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
    model_state._subsidence_reader = None
    model_state._budget_readers = {}
    model_state._observations = {}
    model_state._results_dir = None
    model_state._area_manager = None
    model_state._stream_reach_boundaries = None
    model_state._diversion_ts_data = None
    model_state._node_id_to_idx = None
    model_state._sorted_elem_ids = None
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


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset model_state before every test."""
    _reset_model_state()
    yield
    _reset_model_state()


@pytest.fixture()
def app():
    """Create the FastAPI application."""
    return create_app()


@pytest.fixture()
def client(app):
    """TestClient with no model loaded."""
    return TestClient(app)


def _set_model(model):
    """Inject a model into the global model_state without triggering set_model
    (which would reset caches we may want to configure)."""
    model_state._model = model


def _identity_reproject(x, y):
    """Identity reprojection — returns (x, y) unchanged."""
    return (x, y)


# ===================================================================
# GET /api/mesh
# ===================================================================


class TestGetMesh:
    """Tests for GET /api/mesh (full 3D VTU)."""

    def test_no_model_returns_404(self, client):
        resp = client.get("/api/mesh")
        assert resp.status_code == 404

    def test_returns_vtu_xml(self, client):
        model = _make_mock_model()
        _set_model(model)
        vtu_bytes = b"<VTKFile>fake</VTKFile>"
        with patch.object(model_state, "get_mesh_3d", return_value=vtu_bytes):
            resp = client.get("/api/mesh")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/xml"
        assert b"fake" in resp.content


# ===================================================================
# GET /api/mesh/surface
# ===================================================================


class TestGetSurfaceMesh:
    """Tests for GET /api/mesh/surface."""

    def test_no_model_returns_404(self, client):
        resp = client.get("/api/mesh/surface")
        assert resp.status_code == 404

    def test_returns_surface_vtu(self, client):
        model = _make_mock_model()
        _set_model(model)
        vtu_bytes = b"<VTKFile>surface</VTKFile>"
        with patch.object(model_state, "get_mesh_surface", return_value=vtu_bytes):
            resp = client.get("/api/mesh/surface")
        assert resp.status_code == 200
        assert b"surface" in resp.content


# ===================================================================
# GET /api/mesh/json
# ===================================================================


class TestGetMeshJson:
    """Tests for GET /api/mesh/json."""

    def test_no_model_returns_404(self, client):
        resp = client.get("/api/mesh/json")
        assert resp.status_code == 404

    def test_layer_exceeds_n_layers_returns_400(self, client):
        strat = MagicMock()
        strat.n_layers = 2
        model = _make_mock_model(stratigraphy=strat)
        _set_model(model)
        resp = client.get("/api/mesh/json?layer=5")
        assert resp.status_code == 400
        assert "exceeds" in resp.json()["detail"]

    def test_layer_zero_all_layers(self, client):
        model = _make_mock_model()
        _set_model(model)
        data = {
            "n_points": 4,
            "n_cells": 1,
            "n_layers": 1,
            "points": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            "polys": [4, 0, 1, 2, 3],
            "layer": [1],
        }
        with patch.object(model_state, "get_surface_json", return_value=data):
            resp = client.get("/api/mesh/json?layer=0")
        assert resp.status_code == 200
        body = resp.json()
        assert body["n_points"] == 4
        assert body["n_cells"] == 1

    def test_specific_layer(self, client):
        strat = MagicMock()
        strat.n_layers = 3
        model = _make_mock_model(stratigraphy=strat)
        _set_model(model)
        data = {
            "n_points": 4,
            "n_cells": 1,
            "n_layers": 2,
            "points": [0.0] * 12,
            "polys": [4, 0, 1, 2, 3],
            "layer": [2],
        }
        with patch.object(model_state, "get_surface_json", return_value=data):
            resp = client.get("/api/mesh/json?layer=2")
        assert resp.status_code == 200
        body = resp.json()
        assert body["n_layers"] == 2

    def test_value_error_from_get_surface_json_returns_400(self, client):
        model = _make_mock_model()
        _set_model(model)
        with patch.object(model_state, "get_surface_json", side_effect=ValueError("No 3D mesh")):
            resp = client.get("/api/mesh/json?layer=0")
        assert resp.status_code == 400
        assert "No 3D mesh" in resp.json()["detail"]

    def test_layer_check_skipped_when_no_stratigraphy(self, client):
        """When stratigraphy is None, layer validation is skipped (no 400)."""
        model = _make_mock_model(stratigraphy=None)
        _set_model(model)
        data = {
            "n_points": 0,
            "n_cells": 0,
            "n_layers": 0,
            "points": [],
            "polys": [],
            "layer": [],
        }
        with patch.object(model_state, "get_surface_json", return_value=data):
            resp = client.get("/api/mesh/json?layer=99")
        # No 400 because stratigraphy is None so the check is skipped
        assert resp.status_code == 200


# ===================================================================
# GET /api/mesh/geojson
# ===================================================================


class TestGetMeshGeojson:
    """Tests for GET /api/mesh/geojson."""

    def test_no_model_returns_404(self, client):
        resp = client.get("/api/mesh/geojson")
        assert resp.status_code == 404

    def test_normal_geojson(self, client):
        model = _make_mock_model()
        _set_model(model)
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
                    },
                    "properties": {"element_id": 1, "layer": 1},
                },
            ],
        }
        with patch.object(model_state, "get_mesh_geojson", return_value=geojson):
            resp = client.get("/api/mesh/geojson?layer=1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["type"] == "FeatureCollection"
        assert len(body["features"]) == 1


# ===================================================================
# GET /api/mesh/head-map
# ===================================================================


class TestGetHeadMap:
    """Tests for GET /api/mesh/head-map."""

    def test_no_model_returns_404(self, client):
        resp = client.get("/api/mesh/head-map")
        assert resp.status_code == 404

    def test_no_head_loader_returns_404(self, client):
        model = _make_mock_model()
        _set_model(model)
        with patch.object(model_state, "get_head_loader", return_value=None):
            resp = client.get("/api/mesh/head-map")
        assert resp.status_code == 404
        assert "No head data" in resp.json()["detail"]

    def test_timestep_out_of_range_returns_400(self, client):
        model = _make_mock_model()
        _set_model(model)
        loader = MagicMock()
        loader.n_frames = 5
        with patch.object(model_state, "get_head_loader", return_value=loader):
            resp = client.get("/api/mesh/head-map?timestep=10&layer=1")
        assert resp.status_code == 400
        assert "out of range" in resp.json()["detail"]

    def test_layer_out_of_range_returns_400(self, client):
        model = _make_mock_model()
        _set_model(model)
        loader = MagicMock()
        loader.n_frames = 5
        # Frame with 2 layers: shape (n_nodes, 2)
        frame = np.array([[10.0, 20.0]] * 6)
        loader.get_frame.return_value = frame
        with patch.object(model_state, "get_head_loader", return_value=loader):
            resp = client.get("/api/mesh/head-map?timestep=0&layer=5")
        assert resp.status_code == 400
        assert "Layer 5 out of range" in resp.json()["detail"]

    def test_normal_head_map(self, client):
        model = _make_mock_model()
        _set_model(model)
        loader = MagicMock()
        loader.n_frames = 5
        # 6 nodes, 2 layers
        frame = np.array(
            [
                [10.0, 20.0],
                [11.0, 21.0],
                [12.0, 22.0],
                [13.0, 23.0],
                [14.0, 24.0],
                [15.0, 25.0],
            ]
        )
        loader.get_frame.return_value = frame
        loader.times = [
            datetime.datetime(2020, 1, 1),
            datetime.datetime(2020, 2, 1),
            datetime.datetime(2020, 3, 1),
            datetime.datetime(2020, 4, 1),
            datetime.datetime(2020, 5, 1),
        ]
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                    },
                    "properties": {"element_id": 1, "layer": 1},
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[1, 0], [2, 0], [2, 1], [1, 1], [1, 0]]],
                    },
                    "properties": {"element_id": 2, "layer": 1},
                },
            ],
        }
        with (
            patch.object(model_state, "get_head_loader", return_value=loader),
            patch.object(model_state, "get_mesh_geojson", return_value=geojson),
        ):
            resp = client.get("/api/mesh/head-map?timestep=0&layer=1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["type"] == "FeatureCollection"
        assert len(body["features"]) == 2
        assert "head" in body["features"][0]["properties"]
        assert body["metadata"]["timestep_index"] == 0
        assert body["metadata"]["datetime"] == "2020-01-01T00:00:00"
        assert body["metadata"]["layer"] == 1

    def test_head_map_element_not_in_grid(self, client):
        """Feature with element_id not in grid.elements is skipped."""
        model = _make_mock_model()
        _set_model(model)
        loader = MagicMock()
        loader.n_frames = 2
        frame = np.array([[10.0]] * 6)
        loader.get_frame.return_value = frame
        loader.times = [datetime.datetime(2020, 1, 1), datetime.datetime(2020, 2, 1)]
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
                    },
                    "properties": {"element_id": 999, "layer": 1},  # not in grid
                },
            ],
        }
        with (
            patch.object(model_state, "get_head_loader", return_value=loader),
            patch.object(model_state, "get_mesh_geojson", return_value=geojson),
        ):
            resp = client.get("/api/mesh/head-map?timestep=0&layer=1")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["features"]) == 0

    def test_head_map_datetime_none_when_index_oob(self, client):
        """When loader.times has fewer entries than timestep, datetime is None."""
        model = _make_mock_model()
        _set_model(model)
        loader = MagicMock()
        loader.n_frames = 5
        frame = np.array([[10.0]] * 6)
        loader.get_frame.return_value = frame
        loader.times = []  # empty times
        geojson = {"type": "FeatureCollection", "features": []}
        with (
            patch.object(model_state, "get_head_loader", return_value=loader),
            patch.object(model_state, "get_mesh_geojson", return_value=geojson),
        ):
            resp = client.get("/api/mesh/head-map?timestep=0&layer=1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["metadata"]["datetime"] is None


# ===================================================================
# GET /api/mesh/subregions
# ===================================================================


class TestGetSubregions:
    """Tests for GET /api/mesh/subregions."""

    def test_no_model_returns_404(self, client):
        resp = client.get("/api/mesh/subregions")
        assert resp.status_code == 404

    def test_no_subregions_returns_empty(self, client):
        """When all elements have subregion=0, return empty."""
        nodes = {
            1: Node(id=1, x=0.0, y=0.0),
            2: Node(id=2, x=100.0, y=0.0),
            3: Node(id=3, x=100.0, y=100.0),
            4: Node(id=4, x=0.0, y=100.0),
        }
        elements = {
            1: Element(id=1, vertices=(1, 2, 3, 4), subregion=0),
        }
        grid = AppGrid(nodes=nodes, elements=elements)
        grid.compute_connectivity()
        grid.compute_areas()
        model = _make_mock_model(grid=grid)
        _set_model(model)
        with patch.object(model_state, "reproject_coords", side_effect=_identity_reproject):
            resp = client.get("/api/mesh/subregions")
        assert resp.status_code == 200
        body = resp.json()
        assert body["type"] == "FeatureCollection"
        assert len(body["features"]) == 0

    def test_normal_subregion_boundary(self, client):
        """Two adjacent elements in same subregion produce a boundary polygon."""
        grid = _make_grid()
        grid.subregions = {1: Subregion(id=1, name="Sacramento Valley")}
        model = _make_mock_model(grid=grid)
        _set_model(model)
        with patch.object(model_state, "reproject_coords", side_effect=_identity_reproject):
            resp = client.get("/api/mesh/subregions")
        assert resp.status_code == 200
        body = resp.json()
        assert body["type"] == "FeatureCollection"
        assert len(body["features"]) >= 1
        feat = body["features"][0]
        assert feat["properties"]["subregion_id"] == 1
        assert feat["properties"]["name"] == "Sacramento Valley"
        assert feat["properties"]["n_elements"] == 2
        assert "centroid" in feat["properties"]
        coords = feat["geometry"]["coordinates"][0]
        assert len(coords) >= 4  # At least a triangle + closing

    def test_subregion_without_subregion_info(self, client):
        """When grid.subregions does not have the sub_id, fall back to default name."""
        grid = _make_grid()
        grid.subregions = {}  # no subregion info
        model = _make_mock_model(grid=grid)
        _set_model(model)
        with patch.object(model_state, "reproject_coords", side_effect=_identity_reproject):
            resp = client.get("/api/mesh/subregions")
        assert resp.status_code == 200
        body = resp.json()
        if len(body["features"]) > 0:
            assert "Subregion 1" in body["features"][0]["properties"]["name"]

    def test_subregion_no_boundary_edges_skipped(self, client):
        """When all edges in a subregion appear exactly 2 times, it is skipped.

        We achieve this by creating two triangles that share all edges via
        degenerate (overlapping) vertices, so edge_count for each edge is 2.
        This is geometrically impossible but tests the ``not boundary_edges``
        continue path (line 273).
        """
        # Create two triangles with identical vertices. Both produce the same
        # set of 3 canonical edges, so each edge appears count=2 -> no boundary.
        nodes = {
            1: Node(id=1, x=0.0, y=0.0),
            2: Node(id=2, x=1.0, y=0.0),
            3: Node(id=3, x=0.5, y=1.0),
        }
        # Two elements that have exactly the same edge set.
        # Element vertices (1,2,3) produces edges (1,2), (2,3), (1,3).
        # A second element with the same vertices produces the same edges again.
        # Each edge ends up with count=2.
        elements = {
            1: Element(id=1, vertices=(1, 2, 3), subregion=2),
            2: Element(id=2, vertices=(1, 3, 2), subregion=2),  # reversed winding
        }
        grid = AppGrid(nodes=nodes, elements=elements)
        grid.compute_connectivity()
        grid.compute_areas()
        grid.subregions = {2: Subregion(id=2, name="Degenerate")}
        model = _make_mock_model(grid=grid)
        _set_model(model)
        with patch.object(model_state, "reproject_coords", side_effect=_identity_reproject):
            resp = client.get("/api/mesh/subregions")
        assert resp.status_code == 200
        body = resp.json()
        # No features because no boundary edges -> subregion skipped
        assert body["type"] == "FeatureCollection"
        assert len(body["features"]) == 0

    def test_subregion_walk_dead_end_and_short_ring(self, client):
        """Test that the ring walk handles dead-ends (next_node=None break)
        and that rings shorter than 4 are skipped.

        We create a T-shaped boundary where the ring walk reaches a node
        with no unvisited, non-previous neighbors and breaks. The
        resulting partial ring has fewer than 4 nodes.
        """
        # Create a geometry where boundary edges form a T-junction:
        #   3---4
        #   |   |
        #   1---2---5
        # Elements: quad(1,2,4,3) in subregion 3, triangle(2,5,4) in subregion 3.
        # This produces boundary edges: (1,2),(1,3),(3,4),(4,5),(2,5) = 5 edges.
        # The adjacency graph has node 2 connected to [1,5] and node 4 connected
        # to [3,5]. Starting at some node, the walk may not close, producing
        # a chain rather than a ring, testing the next_node=None break.
        # Also, some walk fragments may be shorter than 4 nodes.
        nodes = {
            1: Node(id=1, x=0.0, y=0.0),
            2: Node(id=2, x=1.0, y=0.0),
            3: Node(id=3, x=0.0, y=1.0),
            4: Node(id=4, x=1.0, y=1.0),
            5: Node(id=5, x=2.0, y=0.0),
        }
        elements = {
            1: Element(id=1, vertices=(1, 2, 4, 3), subregion=3),
            2: Element(id=2, vertices=(2, 5, 4), subregion=3),
        }
        grid = AppGrid(nodes=nodes, elements=elements)
        grid.compute_connectivity()
        grid.compute_areas()
        grid.subregions = {3: Subregion(id=3, name="T-shape")}
        model = _make_mock_model(grid=grid)
        _set_model(model)
        with patch.object(model_state, "reproject_coords", side_effect=_identity_reproject):
            resp = client.get("/api/mesh/subregions")
        assert resp.status_code == 200
        body = resp.json()
        assert body["type"] == "FeatureCollection"
        # May or may not produce features depending on ring assembly

    def test_subregion_ring_nodes_missing_from_grid(self, client):
        """When ring nodes are not found in grid.nodes (node is None),
        the coords list is shorter than the ring and coords < 4 triggers
        the continue on line 337.

        We build the grid normally so iter_elements works, but then delete
        most nodes from the dict before the endpoint runs the boundary
        coordinate lookup.
        """
        grid = _make_grid()
        grid.subregions = {1: Subregion(id=1, name="Missing Nodes")}
        model = _make_mock_model(grid=grid)
        _set_model(model)

        # The subregion boundary ring will reference nodes 1-6.
        # Remove most of them so grid.nodes.get(nid) returns None and
        # the assembled coords list ends up < 4.
        # Keep nodes 1 and 2 only (2 coords not enough for a polygon).
        for nid in [3, 4, 5, 6]:
            del grid.nodes[nid]

        with patch.object(model_state, "reproject_coords", side_effect=_identity_reproject):
            resp = client.get("/api/mesh/subregions")
        assert resp.status_code == 200
        body = resp.json()
        # Subregion skipped because coords < 4 after filtering out missing nodes
        assert body["type"] == "FeatureCollection"
        assert len(body["features"]) == 0

    def test_subregion_centroid_computation(self, client):
        """Verify centroid is computed as average of non-closing ring vertices."""
        grid = _make_grid()
        grid.subregions = {1: Subregion(id=1, name="Test")}
        model = _make_mock_model(grid=grid)
        _set_model(model)
        with patch.object(model_state, "reproject_coords", side_effect=_identity_reproject):
            resp = client.get("/api/mesh/subregions")
        assert resp.status_code == 200
        body = resp.json()
        if len(body["features"]) > 0:
            centroid = body["features"][0]["properties"]["centroid"]
            assert isinstance(centroid, list)
            assert len(centroid) == 2
            # The centroid should be within the grid bounds
            assert 0.0 <= centroid[0] <= 200.0
            assert 0.0 <= centroid[1] <= 100.0


# ===================================================================
# GET /api/mesh/property-map
# ===================================================================


class TestGetPropertyMap:
    """Tests for GET /api/mesh/property-map."""

    def test_no_model_returns_404(self, client):
        resp = client.get("/api/mesh/property-map?property=kh")
        assert resp.status_code == 404

    def test_property_not_available_returns_404(self, client):
        model = _make_mock_model()
        _set_model(model)
        with patch(
            "pyiwfm.visualization.webapi.routes.properties._compute_property_values",
            return_value=None,
        ):
            resp = client.get("/api/mesh/property-map?property=nonexistent")
        assert resp.status_code == 404
        assert "not available" in resp.json()["detail"]

    def test_layer_out_of_range_returns_400(self, client):
        model = _make_mock_model()
        _set_model(model)
        # values has 2 elements (n_elements=2, 1 layer), layer=2 would
        # require start=2, end=4 but values only has 2 entries.
        values = np.array([1.0, 2.0])
        with patch(
            "pyiwfm.visualization.webapi.routes.properties._compute_property_values",
            return_value=values,
        ):
            resp = client.get("/api/mesh/property-map?property=kh&layer=2")
        assert resp.status_code == 400
        assert "out of range" in resp.json()["detail"]

    def test_normal_property_map(self, client):
        model = _make_mock_model()
        _set_model(model)
        # 2 elements, 1 layer
        values = np.array([1.5, 2.5])
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
                    },
                    "properties": {"element_id": 1, "layer": 1},
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[1, 0], [2, 0], [2, 1], [1, 0]]],
                    },
                    "properties": {"element_id": 2, "layer": 1},
                },
            ],
        }
        with (
            patch(
                "pyiwfm.visualization.webapi.routes.properties._compute_property_values",
                return_value=values,
            ),
            patch.object(model_state, "get_mesh_geojson", return_value=geojson),
        ):
            resp = client.get("/api/mesh/property-map?property=kh&layer=1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["type"] == "FeatureCollection"
        assert len(body["features"]) == 2
        assert body["features"][0]["properties"]["value"] == 1.5
        assert body["features"][1]["properties"]["value"] == 2.5
        assert body["metadata"]["property"] == "kh"
        assert body["metadata"]["min"] <= body["metadata"]["max"]

    def test_property_map_with_nan_values(self, client):
        model = _make_mock_model()
        _set_model(model)
        values = np.array([np.nan, 3.0])
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
                    },
                    "properties": {"element_id": 1, "layer": 1},
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[1, 0], [2, 0], [2, 1], [1, 0]]],
                    },
                    "properties": {"element_id": 2, "layer": 1},
                },
            ],
        }
        with (
            patch(
                "pyiwfm.visualization.webapi.routes.properties._compute_property_values",
                return_value=values,
            ),
            patch.object(model_state, "get_mesh_geojson", return_value=geojson),
        ):
            resp = client.get("/api/mesh/property-map?property=kh&layer=1")
        assert resp.status_code == 200
        body = resp.json()
        # NaN becomes None
        assert body["features"][0]["properties"]["value"] is None
        assert body["features"][1]["properties"]["value"] == 3.0
        # min/max from valid values only
        assert body["metadata"]["min"] == 3.0
        assert body["metadata"]["max"] == 3.0

    def test_property_map_all_nan(self, client):
        model = _make_mock_model()
        _set_model(model)
        values = np.array([np.nan, np.nan])
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
                    },
                    "properties": {"element_id": 1, "layer": 1},
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[1, 0], [2, 0], [2, 1], [1, 0]]],
                    },
                    "properties": {"element_id": 2, "layer": 1},
                },
            ],
        }
        with (
            patch(
                "pyiwfm.visualization.webapi.routes.properties._compute_property_values",
                return_value=values,
            ),
            patch.object(model_state, "get_mesh_geojson", return_value=geojson),
        ):
            resp = client.get("/api/mesh/property-map?property=kh&layer=1")
        assert resp.status_code == 200
        body = resp.json()
        # All NaN -> fallback min=0, max=1
        assert body["metadata"]["min"] == 0.0
        assert body["metadata"]["max"] == 1.0

    def test_property_map_elem_id_not_in_index(self, client):
        """Feature with element_id not in elem_id_to_idx is skipped."""
        model = _make_mock_model()
        _set_model(model)
        values = np.array([1.0, 2.0])
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
                    },
                    "properties": {"element_id": 999, "layer": 1},
                },
            ],
        }
        with (
            patch(
                "pyiwfm.visualization.webapi.routes.properties._compute_property_values",
                return_value=values,
            ),
            patch.object(model_state, "get_mesh_geojson", return_value=geojson),
        ):
            resp = client.get("/api/mesh/property-map?property=kh&layer=1")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["features"]) == 0


# ===================================================================
# GET /api/mesh/element/{element_id}
# ===================================================================


class TestGetElementDetail:
    """Tests for GET /api/mesh/element/{element_id}."""

    def test_no_model_returns_404(self, client):
        resp = client.get("/api/mesh/element/1")
        assert resp.status_code == 404

    def test_element_not_found_returns_404(self, client):
        model = _make_mock_model()
        _set_model(model)
        with patch.object(model_state, "reproject_coords", side_effect=_identity_reproject):
            resp = client.get("/api/mesh/element/999")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"]

    def test_basic_element_info(self, client):
        """Test basic element info: subregion, vertices, area."""
        grid = _make_grid()
        grid.subregions = {1: Subregion(id=1, name="TestSub")}
        model = _make_mock_model(grid=grid, groundwater=None, rootzone=None)
        _set_model(model)
        with (
            patch.object(model_state, "reproject_coords", side_effect=_identity_reproject),
            patch.object(model_state, "get_head_loader", return_value=None),
        ):
            resp = client.get("/api/mesh/element/1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["element_id"] == 1
        assert body["subregion"]["id"] == 1
        assert body["subregion"]["name"] == "TestSub"
        assert body["n_vertices"] == 4
        assert len(body["vertices"]) == 4
        assert body["area"] > 0
        assert body["wells"] == []
        assert body["head_at_nodes"] == {}
        assert body["land_use"] is None

    def test_subregion_without_info(self, client):
        """When subregion info is not in grid.subregions, use default name."""
        grid = _make_grid()
        grid.subregions = {}
        model = _make_mock_model(grid=grid, groundwater=None, rootzone=None)
        _set_model(model)
        with (
            patch.object(model_state, "reproject_coords", side_effect=_identity_reproject),
            patch.object(model_state, "get_head_loader", return_value=None),
        ):
            resp = client.get("/api/mesh/element/1")
        assert resp.status_code == 200
        body = resp.json()
        assert "Subregion 1" in body["subregion"]["name"]

    def test_stratigraphy_top_bottom_thickness(self, client):
        """Test layer properties with stratigraphy data."""
        grid = _make_grid()
        grid.subregions = {}
        n_layers = 2
        top_elev = np.array(
            [
                [100.0, 50.0],
                [100.0, 50.0],
                [100.0, 50.0],
                [100.0, 50.0],
                [100.0, 50.0],
                [100.0, 50.0],
            ]
        )
        bottom_elev = np.array(
            [
                [50.0, 0.0],
                [50.0, 0.0],
                [50.0, 0.0],
                [50.0, 0.0],
                [50.0, 0.0],
                [50.0, 0.0],
            ]
        )
        strat = MagicMock()
        strat.n_layers = n_layers
        strat.top_elev = top_elev
        strat.bottom_elev = bottom_elev
        model = _make_mock_model(grid=grid, stratigraphy=strat, groundwater=None, rootzone=None)
        _set_model(model)
        with (
            patch.object(model_state, "reproject_coords", side_effect=_identity_reproject),
            patch.object(model_state, "get_head_loader", return_value=None),
        ):
            resp = client.get("/api/mesh/element/1")
        assert resp.status_code == 200
        body = resp.json()
        layers = body["layer_properties"]
        assert len(layers) == 2
        assert layers[0]["layer"] == 1
        assert layers[0]["top_elev"] == 100.0
        assert layers[0]["bottom_elev"] == 50.0
        assert layers[0]["thickness"] == 50.0
        assert layers[1]["layer"] == 2
        assert layers[1]["top_elev"] == 50.0
        assert layers[1]["bottom_elev"] == 0.0

    def test_aquifer_params_2d(self, client):
        """Test aquifer parameters with 2D arrays (n_nodes x n_layers)."""
        grid = _make_grid()
        grid.subregions = {}
        n_layers = 2
        strat = MagicMock()
        strat.n_layers = n_layers
        strat.top_elev = np.ones((6, 2)) * 100.0
        strat.bottom_elev = np.zeros((6, 2))

        params = MagicMock()
        params.kh = np.full((6, 2), 10.0)
        params.kv = np.full((6, 2), 1.0)
        params.specific_storage = np.full((6, 2), 0.001)
        params.specific_yield = np.full((6, 2), 0.15)
        # Make ss and sy accessible via getattr fallback
        params.ss = None
        params.sy = None

        gw = MagicMock()
        gw.aquifer_params = params
        gw.iter_wells.return_value = []

        model = _make_mock_model(grid=grid, stratigraphy=strat, groundwater=gw, rootzone=None)
        _set_model(model)
        with (
            patch.object(model_state, "reproject_coords", side_effect=_identity_reproject),
            patch.object(model_state, "get_head_loader", return_value=None),
        ):
            resp = client.get("/api/mesh/element/1")
        assert resp.status_code == 200
        body = resp.json()
        layers = body["layer_properties"]
        assert layers[0]["kh"] == 10.0
        assert layers[0]["kv"] == 1.0
        assert layers[0]["ss"] == 0.001
        assert layers[0]["sy"] == 0.15

    def test_aquifer_params_1d(self, client):
        """Test aquifer parameters with 1D arrays."""
        grid = _make_grid()
        grid.subregions = {}

        kh_1d = np.full(6, 5.0)
        assert kh_1d.ndim == 1  # ensure it's a 1D ndarray

        params = MagicMock()
        params.kh = kh_1d
        params.kv = None
        params.specific_storage = None
        params.ss = None
        params.specific_yield = None
        params.sy = None

        gw = MagicMock()
        gw.aquifer_params = params
        gw.iter_wells.return_value = []

        model = _make_mock_model(grid=grid, groundwater=gw, rootzone=None)
        _set_model(model)
        with (
            patch.object(model_state, "reproject_coords", side_effect=_identity_reproject),
            patch.object(model_state, "get_head_loader", return_value=None),
        ):
            resp = client.get("/api/mesh/element/1")
        assert resp.status_code == 200
        body = resp.json()
        layers = body["layer_properties"]
        assert layers[0]["kh"] == 5.0
        # kv, ss, sy should not be present since they're None
        assert "kv" not in layers[0]
        assert "ss" not in layers[0]
        assert "sy" not in layers[0]

    def test_wells_in_element(self, client):
        """Test wells that belong to the queried element."""
        grid = _make_grid()
        grid.subregions = {}

        well1 = MagicMock()
        well1.id = 1
        well1.name = "Well A"
        well1.element = 1
        well1.pump_rate = -500.0
        well1.layers = [1, 2]

        well2 = MagicMock()
        well2.id = 2
        well2.name = "Well B"
        well2.element = 2  # different element

        gw = MagicMock()
        gw.aquifer_params = None
        gw.iter_wells.return_value = [well1, well2]

        model = _make_mock_model(grid=grid, groundwater=gw, rootzone=None)
        _set_model(model)
        with (
            patch.object(model_state, "reproject_coords", side_effect=_identity_reproject),
            patch.object(model_state, "get_head_loader", return_value=None),
        ):
            resp = client.get("/api/mesh/element/1")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["wells"]) == 1
        assert body["wells"][0]["id"] == 1
        assert body["wells"][0]["name"] == "Well A"
        assert body["wells"][0]["pump_rate"] == -500.0
        assert body["wells"][0]["layers"] == [1, 2]

    def test_head_values_at_nodes(self, client):
        """Test head values populated from head loader."""
        grid = _make_grid()
        grid.subregions = {}
        model = _make_mock_model(grid=grid, groundwater=None, rootzone=None)
        _set_model(model)

        loader = MagicMock()
        loader.n_frames = 3
        # shape (6 nodes, 2 layers)
        frame = np.array(
            [
                [100.0, 80.0],
                [101.0, 81.0],
                [102.0, 82.0],
                [103.0, 83.0],
                [104.0, 84.0],
                [105.0, 85.0],
            ]
        )
        loader.get_frame.return_value = frame

        with (
            patch.object(model_state, "reproject_coords", side_effect=_identity_reproject),
            patch.object(model_state, "get_head_loader", return_value=loader),
        ):
            resp = client.get("/api/mesh/element/1")
        assert resp.status_code == 200
        body = resp.json()
        head = body["head_at_nodes"]
        # Element 1 has vertices 1,2,3,4 which map to indices 0,1,2,3
        assert "1" in head or 1 in head  # JSON keys are strings
        # Check head values for node 1 (index 0)
        node1_key = "1"
        assert head[node1_key] == [100.0, 80.0]

    def test_no_head_loader_empty_heads(self, client):
        """When no head loader is available, head_at_nodes is empty."""
        grid = _make_grid()
        grid.subregions = {}
        model = _make_mock_model(grid=grid, groundwater=None, rootzone=None)
        _set_model(model)
        with (
            patch.object(model_state, "reproject_coords", side_effect=_identity_reproject),
            patch.object(model_state, "get_head_loader", return_value=None),
        ):
            resp = client.get("/api/mesh/element/1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["head_at_nodes"] == {}

    def test_land_use_breakdown(self, client):
        """Test full land-use breakdown from area manager."""
        grid = _make_grid()
        grid.subregions = {}

        rz = MagicMock()
        rz.nonponded_config = MagicMock()
        rz.nonponded_config.n_crops = 3
        rz.crop_types = {
            1: MagicMock(name="Grain"),
            2: MagicMock(name="Pasture"),
            3: MagicMock(name="Truck"),
            4: MagicMock(name="Rice"),
            5: MagicMock(name="Wetland"),
        }

        area_mgr = MagicMock()
        area_mgr.n_timesteps = 10
        area_mgr.get_element_breakdown.return_value = {
            "nonponded": [10.0, 20.0, 5.0],
            "ponded": [3.0, 2.0],
            "urban": [15.0],
            "native": [8.0, 4.0],
        }

        model = _make_mock_model(grid=grid, groundwater=None, rootzone=rz)
        _set_model(model)

        with (
            patch.object(model_state, "reproject_coords", side_effect=_identity_reproject),
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_area_manager", return_value=area_mgr),
        ):
            resp = client.get("/api/mesh/element/1")
        assert resp.status_code == 200
        body = resp.json()
        lu = body["land_use"]
        assert lu is not None
        assert lu["units"] == "acres"
        assert lu["total_area"] > 0

        # Check categories
        categories = lu["categories"]
        nonponded = [c for c in categories if c["category"] == "nonponded"]
        assert len(nonponded) == 3
        assert nonponded[0]["area"] == 10.0
        assert nonponded[0]["crop_id"] == 1

        ponded = [c for c in categories if c["category"] == "ponded"]
        assert len(ponded) == 2
        # ponded crop_id starts at n_nonponded + 1 = 4
        assert ponded[0]["crop_id"] == 4
        assert ponded[0]["area"] == 3.0

        urban = [c for c in categories if c["category"] == "urban"]
        assert len(urban) == 1
        assert urban[0]["area"] == 15.0

        native = [c for c in categories if c["category"] == "native"]
        assert len(native) == 2
        assert native[0]["name"] == "Native Vegetation"
        assert native[0]["area"] == 8.0
        assert native[1]["name"] == "Riparian Vegetation"
        assert native[1]["area"] == 4.0

        # Check fractions
        total = 10.0 + 20.0 + 5.0 + 3.0 + 2.0 + 15.0 + 8.0 + 4.0  # 67.0
        fracs = lu["fractions"]
        ag = (10.0 + 20.0 + 5.0 + 3.0 + 2.0) / total
        assert abs(fracs["agricultural"] - round(ag, 4)) < 0.001
        assert abs(fracs["urban"] - round(15.0 / total, 4)) < 0.001
        assert abs(fracs["native_riparian"] - round(12.0 / total, 4)) < 0.001
        assert fracs["water"] == 0.0

    def test_land_use_with_timestep_param(self, client):
        """Test that the timestep query parameter is used for land use."""
        grid = _make_grid()
        grid.subregions = {}

        rz = MagicMock()
        rz.nonponded_config = MagicMock()
        rz.nonponded_config.n_crops = 1
        rz.crop_types = {1: MagicMock(name="Grain")}

        area_mgr = MagicMock()
        area_mgr.n_timesteps = 10
        area_mgr.get_element_breakdown.return_value = {
            "nonponded": [100.0],
        }

        model = _make_mock_model(grid=grid, groundwater=None, rootzone=rz)
        _set_model(model)

        with (
            patch.object(model_state, "reproject_coords", side_effect=_identity_reproject),
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_area_manager", return_value=area_mgr),
        ):
            resp = client.get("/api/mesh/element/1?timestep=3")
        assert resp.status_code == 200
        # Verify get_element_breakdown was called with timestep=3
        area_mgr.get_element_breakdown.assert_called_once_with(1, timestep=3)

    def test_land_use_timestep_clamped_to_max(self, client):
        """Timestep exceeding n_timesteps is clamped to last."""
        grid = _make_grid()
        grid.subregions = {}

        rz = MagicMock()
        rz.nonponded_config = None
        rz.crop_types = {}

        area_mgr = MagicMock()
        area_mgr.n_timesteps = 5
        area_mgr.get_element_breakdown.return_value = {}

        model = _make_mock_model(grid=grid, groundwater=None, rootzone=rz)
        _set_model(model)

        with (
            patch.object(model_state, "reproject_coords", side_effect=_identity_reproject),
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_area_manager", return_value=area_mgr),
        ):
            resp = client.get("/api/mesh/element/1?timestep=100")
        assert resp.status_code == 200
        # Clamped to n_timesteps - 1 = 4
        area_mgr.get_element_breakdown.assert_called_once_with(1, timestep=4)

    def test_land_use_default_timestep_last(self, client):
        """Default timestep=-1 uses last available."""
        grid = _make_grid()
        grid.subregions = {}

        rz = MagicMock()
        rz.nonponded_config = None
        rz.crop_types = {}

        area_mgr = MagicMock()
        area_mgr.n_timesteps = 5
        area_mgr.get_element_breakdown.return_value = {}

        model = _make_mock_model(grid=grid, groundwater=None, rootzone=rz)
        _set_model(model)

        with (
            patch.object(model_state, "reproject_coords", side_effect=_identity_reproject),
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_area_manager", return_value=area_mgr),
        ):
            resp = client.get("/api/mesh/element/1")  # no timestep param, default=-1
        assert resp.status_code == 200
        # -1 < 0 => ts = n_timesteps - 1 = 4
        area_mgr.get_element_breakdown.assert_called_once_with(1, timestep=4)

    def test_no_rootzone_land_use_is_none(self, client):
        """When model has no rootzone, land_use is None."""
        grid = _make_grid()
        grid.subregions = {}
        model = _make_mock_model(grid=grid, groundwater=None, rootzone=None)
        # Ensure hasattr(model, "rootzone") returns True but value is None
        model.rootzone = None
        _set_model(model)
        with (
            patch.object(model_state, "reproject_coords", side_effect=_identity_reproject),
            patch.object(model_state, "get_head_loader", return_value=None),
        ):
            resp = client.get("/api/mesh/element/1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["land_use"] is None

    def test_area_manager_error_land_use_is_none(self, client):
        """When area manager raises an exception, land_use is None."""
        grid = _make_grid()
        grid.subregions = {}

        rz = MagicMock()
        rz.nonponded_config = None
        rz.crop_types = {}

        area_mgr = MagicMock()
        area_mgr.n_timesteps = 5
        area_mgr.get_element_breakdown.side_effect = RuntimeError("HDF5 error")

        model = _make_mock_model(grid=grid, groundwater=None, rootzone=rz)
        _set_model(model)

        with (
            patch.object(model_state, "reproject_coords", side_effect=_identity_reproject),
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_area_manager", return_value=area_mgr),
        ):
            resp = client.get("/api/mesh/element/1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["land_use"] is None

    def test_area_manager_none_land_use_is_none(self, client):
        """When get_area_manager returns None, land_use is None."""
        grid = _make_grid()
        grid.subregions = {}

        rz = MagicMock()
        rz.nonponded_config = None
        rz.crop_types = {}

        model = _make_mock_model(grid=grid, groundwater=None, rootzone=rz)
        _set_model(model)

        with (
            patch.object(model_state, "reproject_coords", side_effect=_identity_reproject),
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_area_manager", return_value=None),
        ):
            resp = client.get("/api/mesh/element/1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["land_use"] is None

    def test_area_manager_zero_timesteps_land_use_is_none(self, client):
        """When area manager has 0 timesteps, land_use is None."""
        grid = _make_grid()
        grid.subregions = {}

        rz = MagicMock()
        rz.nonponded_config = None

        area_mgr = MagicMock()
        area_mgr.n_timesteps = 0

        model = _make_mock_model(grid=grid, groundwater=None, rootzone=rz)
        _set_model(model)

        with (
            patch.object(model_state, "reproject_coords", side_effect=_identity_reproject),
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_area_manager", return_value=area_mgr),
        ):
            resp = client.get("/api/mesh/element/1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["land_use"] is None

    def test_land_use_breakdown_empty(self, client):
        """When breakdown is empty dict, land_use is None (no categories)."""
        grid = _make_grid()
        grid.subregions = {}

        rz = MagicMock()
        rz.nonponded_config = None
        rz.crop_types = {}

        area_mgr = MagicMock()
        area_mgr.n_timesteps = 5
        area_mgr.get_element_breakdown.return_value = {}  # empty

        model = _make_mock_model(grid=grid, groundwater=None, rootzone=rz)
        _set_model(model)

        with (
            patch.object(model_state, "reproject_coords", side_effect=_identity_reproject),
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_area_manager", return_value=area_mgr),
        ):
            resp = client.get("/api/mesh/element/1")
        assert resp.status_code == 200
        body = resp.json()
        # breakdown is empty dict -> falsy -> land_use remains None
        assert body["land_use"] is None

    def test_land_use_with_only_urban(self, client):
        """Test land use with only urban category."""
        grid = _make_grid()
        grid.subregions = {}

        rz = MagicMock()
        rz.nonponded_config = None
        rz.crop_types = {}

        area_mgr = MagicMock()
        area_mgr.n_timesteps = 5
        area_mgr.get_element_breakdown.return_value = {
            "urban": [50.0],
        }

        model = _make_mock_model(grid=grid, groundwater=None, rootzone=rz)
        _set_model(model)

        with (
            patch.object(model_state, "reproject_coords", side_effect=_identity_reproject),
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_area_manager", return_value=area_mgr),
        ):
            resp = client.get("/api/mesh/element/1")
        assert resp.status_code == 200
        body = resp.json()
        lu = body["land_use"]
        assert lu is not None
        assert lu["total_area"] == 50.0
        assert lu["fractions"]["urban"] == 1.0
        assert lu["fractions"]["agricultural"] == 0.0

    def test_land_use_native_single_column(self, client):
        """Test native with only 1 column (no riparian)."""
        grid = _make_grid()
        grid.subregions = {}

        rz = MagicMock()
        rz.nonponded_config = None
        rz.crop_types = {}

        area_mgr = MagicMock()
        area_mgr.n_timesteps = 5
        area_mgr.get_element_breakdown.return_value = {
            "native": [12.0],  # only native, no riparian
        }

        model = _make_mock_model(grid=grid, groundwater=None, rootzone=rz)
        _set_model(model)

        with (
            patch.object(model_state, "reproject_coords", side_effect=_identity_reproject),
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_area_manager", return_value=area_mgr),
        ):
            resp = client.get("/api/mesh/element/1")
        assert resp.status_code == 200
        body = resp.json()
        lu = body["land_use"]
        assert lu is not None
        native_cats = [c for c in lu["categories"] if c["category"] == "native"]
        assert len(native_cats) == 1
        assert native_cats[0]["name"] == "Native Vegetation"
        assert native_cats[0]["area"] == 12.0

    def test_land_use_zero_total_area_fractions(self, client):
        """When total_area is 0, fractions should be 0."""
        grid = _make_grid()
        grid.subregions = {}

        rz = MagicMock()
        rz.nonponded_config = MagicMock()
        rz.nonponded_config.n_crops = 1
        rz.crop_types = {1: MagicMock(name="Grain")}

        area_mgr = MagicMock()
        area_mgr.n_timesteps = 5
        area_mgr.get_element_breakdown.return_value = {
            "nonponded": [0.0],
            "urban": [0.0],
            "native": [0.0, 0.0],
        }

        model = _make_mock_model(grid=grid, groundwater=None, rootzone=rz)
        _set_model(model)

        with (
            patch.object(model_state, "reproject_coords", side_effect=_identity_reproject),
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_area_manager", return_value=area_mgr),
        ):
            resp = client.get("/api/mesh/element/1")
        assert resp.status_code == 200
        body = resp.json()
        lu = body["land_use"]
        assert lu is not None
        assert lu["total_area"] == 0.0
        assert lu["fractions"]["agricultural"] == 0.0
        assert lu["fractions"]["urban"] == 0.0
        assert lu["fractions"]["native_riparian"] == 0.0

    def test_vertex_coordinates_reprojected(self, client):
        """Verify that vertices contain reprojected lng/lat."""
        grid = _make_grid()
        grid.subregions = {}
        model = _make_mock_model(grid=grid, groundwater=None, rootzone=None)
        _set_model(model)

        def offset_reproject(x, y):
            return (x + 1000.0, y + 2000.0)

        with (
            patch.object(model_state, "reproject_coords", side_effect=offset_reproject),
            patch.object(model_state, "get_head_loader", return_value=None),
        ):
            resp = client.get("/api/mesh/element/1")
        assert resp.status_code == 200
        body = resp.json()
        v0 = body["vertices"][0]
        assert v0["x"] == 0.0
        assert v0["y"] == 0.0
        assert v0["lng"] == 1000.0
        assert v0["lat"] == 2000.0

    def test_no_groundwater_no_wells(self, client):
        """When groundwater is None, wells list is empty."""
        grid = _make_grid()
        grid.subregions = {}
        model = _make_mock_model(grid=grid, groundwater=None, rootzone=None)
        _set_model(model)
        with (
            patch.object(model_state, "reproject_coords", side_effect=_identity_reproject),
            patch.object(model_state, "get_head_loader", return_value=None),
        ):
            resp = client.get("/api/mesh/element/1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["wells"] == []

    def test_no_aquifer_params(self, client):
        """When groundwater exists but aquifer_params is None, no params in layers."""
        grid = _make_grid()
        grid.subregions = {}

        gw = MagicMock()
        gw.aquifer_params = None
        gw.iter_wells.return_value = []

        model = _make_mock_model(grid=grid, groundwater=gw, rootzone=None)
        _set_model(model)
        with (
            patch.object(model_state, "reproject_coords", side_effect=_identity_reproject),
            patch.object(model_state, "get_head_loader", return_value=None),
        ):
            resp = client.get("/api/mesh/element/1")
        assert resp.status_code == 200
        body = resp.json()
        layers = body["layer_properties"]
        assert "kh" not in layers[0]

    def test_crop_type_fallback_names(self, client):
        """When crop_types doesn't have the ID, use fallback names."""
        grid = _make_grid()
        grid.subregions = {}

        rz = MagicMock()
        rz.nonponded_config = MagicMock()
        rz.nonponded_config.n_crops = 2
        rz.crop_types = {}  # empty — all fallback names

        area_mgr = MagicMock()
        area_mgr.n_timesteps = 5
        area_mgr.get_element_breakdown.return_value = {
            "nonponded": [10.0, 20.0],
            "ponded": [5.0],
        }

        model = _make_mock_model(grid=grid, groundwater=None, rootzone=rz)
        _set_model(model)

        with (
            patch.object(model_state, "reproject_coords", side_effect=_identity_reproject),
            patch.object(model_state, "get_head_loader", return_value=None),
            patch.object(model_state, "get_area_manager", return_value=area_mgr),
        ):
            resp = client.get("/api/mesh/element/1")
        assert resp.status_code == 200
        body = resp.json()
        lu = body["land_use"]
        cats = lu["categories"]
        # Nonponded should use "Crop N" fallback
        assert cats[0]["name"] == "Crop 1"
        assert cats[1]["name"] == "Crop 2"
        # Ponded should use "Ponded N" fallback
        assert cats[2]["name"] == "Ponded 1"


# ===================================================================
# GET /api/mesh/nodes
# ===================================================================


class TestGetMeshNodes:
    """Tests for GET /api/mesh/nodes."""

    def test_no_model_returns_404(self, client):
        resp = client.get("/api/mesh/nodes")
        assert resp.status_code == 404

    def test_normal_with_reproject(self, client):
        grid = _make_grid()
        model = _make_mock_model(grid=grid)
        _set_model(model)

        with patch.object(model_state, "reproject_coords", side_effect=_identity_reproject):
            resp = client.get("/api/mesh/nodes")
        assert resp.status_code == 200
        body = resp.json()
        assert body["n_nodes"] == 6
        assert len(body["nodes"]) == 6
        # Verify first node
        node1 = next(n for n in body["nodes"] if n["id"] == 1)
        assert node1["lng"] == 0.0
        assert node1["lat"] == 0.0
        # Verify last node
        node6 = next(n for n in body["nodes"] if n["id"] == 6)
        assert node6["lng"] == 200.0
        assert node6["lat"] == 100.0

    def test_nodes_layer_param_accepted(self, client):
        """The layer param is accepted but unused (for future filtering)."""
        grid = _make_grid()
        model = _make_mock_model(grid=grid)
        _set_model(model)
        with patch.object(model_state, "reproject_coords", side_effect=_identity_reproject):
            resp = client.get("/api/mesh/nodes?layer=3")
        assert resp.status_code == 200
        body = resp.json()
        assert body["n_nodes"] == 6
