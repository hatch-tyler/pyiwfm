"""Comprehensive tests for the slice API routes.

Covers all endpoints in ``pyiwfm.visualization.webapi.routes.slices``:

* ``GET /api/slice`` -- VTU slice along axis
* ``GET /api/slice/json`` -- JSON slice for vtk.js
* ``GET /api/slice/cross-section`` -- VTU cross-section between two points
* ``GET /api/slice/cross-section/json`` -- JSON cross-section for Plotly
* ``GET /api/slice/info`` -- Slice metadata
* ``_pyvista_to_vtu`` -- Helper function

Every branch and edge case is exercised to target 95%+ coverage.
"""

from __future__ import annotations

import sys
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


def _make_grid():
    """Create a 6-node, 2-element grid for testing."""
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
        "get_slice_json",
        "get_pyvista_3d",
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
    """Inject a model into the global model_state."""
    model_state._model = model


def _make_mock_slice_mesh(n_cells=4, n_points=6, has_layer_data=True):
    """Create a mock PyVista PolyData for slice results."""
    mock_mesh = MagicMock()
    mock_mesh.n_cells = n_cells
    mock_mesh.n_points = n_points
    mock_mesh.bounds = (0.0, 100.0, 0.0, 100.0, 0.0, 50.0)
    mock_mesh.area = 5000.0
    mock_mesh.points = np.array(
        [[i * 10.0, i * 5.0, i * 2.0] for i in range(max(n_points, 1))],
        dtype=np.float64,
    )
    mock_mesh.faces = np.array([3, 0, 1, 2] * max(n_cells, 1), dtype=np.int32)
    if has_layer_data and n_cells > 0:
        layer_arr = np.array([1, 1, 2, 2][:n_cells])
        mock_mesh.cell_data = {"layer": layer_arr}
    else:
        mock_mesh.cell_data = {}
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


def _make_model_with_strat():
    """Create a mock model with stratigraphy."""
    strat = MagicMock()
    strat.n_layers = 2
    model = _make_mock_model(stratigraphy=strat)
    return model


# ===================================================================
# GET /api/slice (VTU endpoint)
# ===================================================================


class TestGetSlice:
    """Tests for GET /api/slice (VTU response)."""

    def test_no_model_returns_404(self, client):
        resp = client.get("/api/slice?axis=x&position=0.5")
        assert resp.status_code == 404
        assert "No model loaded" in resp.json()["detail"]

    def test_no_pyvista_returns_500(self, client):
        """500 when PyVista is not installed."""
        model = _make_mock_model(stratigraphy=MagicMock())
        _set_model(model)
        # Remove pyvista from sys.modules to force ImportError
        with patch.dict(sys.modules, {"pyvista": None}):
            resp = client.get("/api/slice?axis=x&position=0.5")
        assert resp.status_code == 500
        assert "PyVista required" in resp.json()["detail"]

    def test_no_stratigraphy_returns_400(self, client):
        """400 when model has no stratigraphy."""
        model = _make_mock_model(stratigraphy=None)
        _set_model(model)
        # pyvista is available in this env, so import succeeds
        resp = client.get("/api/slice?axis=x&position=0.5")
        assert resp.status_code == 400
        assert "Stratigraphy required" in resp.json()["detail"]

    def test_x_axis_success(self, client):
        """GET /api/slice with axis=x returns VTU bytes."""
        model = _make_model_with_strat()
        _set_model(model)
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
            MockExporter.return_value.to_pyvista_3d.return_value = MagicMock()
            resp = client.get("/api/slice?axis=x&position=0.5")

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/xml"
        assert b"mock vtu data" in resp.content
        mock_slicer.slice_x.assert_called_once()

    def test_y_axis_success(self, client):
        """GET /api/slice with axis=y routes to slice_y."""
        model = _make_model_with_strat()
        _set_model(model)
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

    def test_z_axis_success(self, client):
        """GET /api/slice with axis=z routes to slice_z."""
        model = _make_model_with_strat()
        _set_model(model)
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

    def test_empty_slice_returns_404(self, client):
        """404 when slice produces an empty mesh (n_cells=0)."""
        model = _make_model_with_strat()
        _set_model(model)
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

    def test_content_disposition_header(self, client):
        """Response includes Content-Disposition header for file download."""
        model = _make_model_with_strat()
        _set_model(model)
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


# ===================================================================
# GET /api/slice/json
# ===================================================================


class TestGetSliceJson:
    """Tests for GET /api/slice/json."""

    def test_no_model_returns_404(self, client):
        resp = client.get("/api/slice/json")
        assert resp.status_code == 404
        assert "No model loaded" in resp.json()["detail"]

    def test_success(self, client):
        """Success returns SurfaceMeshData format."""
        model = _make_mock_model()
        _set_model(model)
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

    def test_empty_slice_returns_404(self, client):
        """404 when slice produces no cells."""
        model = _make_mock_model()
        _set_model(model)
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

    def test_value_error_returns_400(self, client):
        """400 when model_state.get_slice_json raises ValueError."""
        model = _make_mock_model()
        _set_model(model)
        with patch.object(
            model_state,
            "get_slice_json",
            side_effect=ValueError("Invalid slice parameters"),
        ):
            resp = client.get("/api/slice/json?angle=45&position=0.5")
        assert resp.status_code == 400
        assert "Invalid slice parameters" in resp.json()["detail"]

    def test_angle_out_of_range_returns_422(self, client):
        """Angle parameter is validated between 0 and 180."""
        model = _make_mock_model()
        _set_model(model)
        resp = client.get("/api/slice/json?angle=200&position=0.5")
        assert resp.status_code == 422

    def test_position_out_of_range_returns_422(self, client):
        """Position parameter is validated between 0 and 1."""
        model = _make_mock_model()
        _set_model(model)
        resp = client.get("/api/slice/json?angle=0&position=2.0")
        assert resp.status_code == 422

    def test_default_params(self, client):
        """Default parameters: angle=0.0, position=0.5."""
        model = _make_mock_model()
        _set_model(model)
        mock_data = {
            "n_points": 3,
            "n_cells": 1,
            "n_layers": 1,
            "points": [0.0] * 9,
            "polys": [3, 0, 1, 2],
            "layer": [1],
        }
        with patch.object(model_state, "get_slice_json", return_value=mock_data) as mock_get:
            resp = client.get("/api/slice/json")
        assert resp.status_code == 200
        mock_get.assert_called_once_with(0.0, 0.5)


# ===================================================================
# GET /api/slice/cross-section
# ===================================================================


class TestGetCrossSection:
    """Tests for GET /api/slice/cross-section (VTU response)."""

    def test_no_model_returns_404(self, client):
        resp = client.get("/api/slice/cross-section?start_x=0&start_y=0&end_x=100&end_y=100")
        assert resp.status_code == 404
        assert "No model loaded" in resp.json()["detail"]

    def test_no_pyvista_returns_500(self, client):
        """500 when PyVista is not installed."""
        model = _make_mock_model(stratigraphy=MagicMock())
        _set_model(model)
        with patch.dict(sys.modules, {"pyvista": None}):
            resp = client.get("/api/slice/cross-section?start_x=0&start_y=0&end_x=100&end_y=100")
        assert resp.status_code == 500
        assert "PyVista required" in resp.json()["detail"]

    def test_no_stratigraphy_returns_400(self, client):
        """400 when model has no stratigraphy."""
        model = _make_mock_model(stratigraphy=None)
        _set_model(model)
        resp = client.get("/api/slice/cross-section?start_x=0&start_y=0&end_x=100&end_y=100")
        assert resp.status_code == 400
        assert "Stratigraphy required" in resp.json()["detail"]

    def test_success(self, client):
        """Success returns VTU bytes with cross_section.vtu filename."""
        model = _make_model_with_strat()
        _set_model(model)
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

    def test_empty_result_returns_404(self, client):
        """404 when cross-section produces no cells."""
        model = _make_model_with_strat()
        _set_model(model)
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

    def test_missing_params_returns_422(self, client):
        """Missing required query parameters returns 422."""
        model = _make_model_with_strat()
        _set_model(model)
        resp = client.get("/api/slice/cross-section?start_x=0")
        assert resp.status_code == 422


# ===================================================================
# GET /api/slice/cross-section/json
# ===================================================================


class TestGetCrossSectionJson:
    """Tests for GET /api/slice/cross-section/json."""

    def test_no_model_returns_404(self, client):
        resp = client.get(
            "/api/slice/cross-section/json?"
            "start_lng=-121.0&start_lat=37.0&end_lng=-120.0&end_lat=38.0"
        )
        assert resp.status_code == 404
        assert "No model loaded" in resp.json()["detail"]

    def test_no_stratigraphy_returns_400(self, client):
        """400 when model has no stratigraphy."""
        model = _make_mock_model(stratigraphy=None)
        _set_model(model)
        resp = client.get(
            "/api/slice/cross-section/json?"
            "start_lng=-121.0&start_lat=37.0&end_lng=-120.0&end_lat=38.0"
        )
        assert resp.status_code == 400
        assert "Stratigraphy required" in resp.json()["detail"]

    def test_empty_slice_returns_empty_dict(self, client):
        """Empty slice returns a dict with n_points=0, n_cells=0."""
        model = _make_model_with_strat()
        _set_model(model)

        empty_mesh = _make_mock_slice_mesh(n_cells=0, n_points=0)
        mock_slicer = MagicMock()
        mock_slicer.create_cross_section.return_value = empty_mesh

        mock_pv_mesh = MagicMock()

        with (
            patch.object(model_state, "get_pyvista_3d", return_value=mock_pv_mesh),
            patch(
                "pyiwfm.visualization.webapi.slicing.SlicingController",
                return_value=mock_slicer,
            ),
        ):
            resp = client.get(
                "/api/slice/cross-section/json?"
                "start_lng=-121.0&start_lat=37.0&end_lng=-120.0&end_lat=38.0"
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["n_points"] == 0
        assert data["n_cells"] == 0
        assert data["points"] == []
        assert data["layer"] == []
        assert data["distance"] == []

    def test_success_with_pyproj(self, client):
        """Full success path with pyproj coordinate transformation."""
        model = _make_model_with_strat()
        _set_model(model)

        # Create a non-empty slice mesh
        n_cells = 2
        n_points = 4
        mock_mesh = MagicMock()
        mock_mesh.n_cells = n_cells
        mock_mesh.n_points = n_points
        mock_mesh.points = np.array(
            [
                [100.0, 200.0, 50.0],
                [110.0, 210.0, 40.0],
                [120.0, 220.0, 30.0],
                [130.0, 230.0, 20.0],
            ],
            dtype=np.float64,
        )
        mock_mesh.faces = np.array([3, 0, 1, 2, 3, 1, 2, 3], dtype=np.int32)
        mock_mesh.cell_data = {"layer": np.array([1, 2])}

        mock_slicer = MagicMock()
        mock_slicer.create_cross_section.return_value = mock_mesh

        mock_pv_mesh = MagicMock()

        # Mock pyproj Transformer
        mock_tf = MagicMock()
        # Transform: lng,lat -> x,y (simulate UTM conversion)
        mock_tf.transform.side_effect = lambda lng, lat: (
            lng * 1000.0,
            lat * 1000.0,
        )

        mock_transformer_cls = MagicMock()
        mock_transformer_cls.from_crs.return_value = mock_tf
        mock_pyproj = MagicMock()
        mock_pyproj.Transformer = mock_transformer_cls

        with (
            patch.object(model_state, "get_pyvista_3d", return_value=mock_pv_mesh),
            patch(
                "pyiwfm.visualization.webapi.slicing.SlicingController",
                return_value=mock_slicer,
            ),
            patch.dict(sys.modules, {"pyproj": mock_pyproj}),
        ):
            resp = client.get(
                "/api/slice/cross-section/json?"
                "start_lng=-121.0&start_lat=37.0&end_lng=-120.0&end_lat=38.0"
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["n_points"] == 4
        assert data["n_cells"] == 2
        assert len(data["points"]) == 12  # 4 points * 3 coords
        assert len(data["layer"]) == 2
        assert data["layer"] == [1, 2]
        assert len(data["distance"]) == 4
        assert "polys" in data
        assert "start" in data
        assert "end" in data
        assert "total_distance" in data
        # Verify start/end contain both original and converted coords
        assert data["start"]["lng"] == -121.0
        assert data["start"]["lat"] == 37.0
        assert data["end"]["lng"] == -120.0
        assert data["end"]["lat"] == 38.0

    def test_success_without_pyproj(self, client):
        """Success path when pyproj is not available (coordinates passed through)."""
        model = _make_model_with_strat()
        _set_model(model)

        n_cells = 2
        n_points = 3
        mock_mesh = MagicMock()
        mock_mesh.n_cells = n_cells
        mock_mesh.n_points = n_points
        mock_mesh.points = np.array(
            [
                [10.0, 20.0, 5.0],
                [30.0, 40.0, 3.0],
                [50.0, 60.0, 1.0],
            ],
            dtype=np.float64,
        )
        mock_mesh.faces = np.array([3, 0, 1, 2, 3, 0, 1, 2], dtype=np.int32)
        mock_mesh.cell_data = {"layer": np.array([1, 1])}

        mock_slicer = MagicMock()
        mock_slicer.create_cross_section.return_value = mock_mesh

        mock_pv_mesh = MagicMock()

        with (
            patch.object(model_state, "get_pyvista_3d", return_value=mock_pv_mesh),
            patch(
                "pyiwfm.visualization.webapi.slicing.SlicingController",
                return_value=mock_slicer,
            ),
            patch.dict(sys.modules, {"pyproj": None}),
        ):
            resp = client.get(
                "/api/slice/cross-section/json?"
                "start_lng=10.0&start_lat=20.0&end_lng=50.0&end_lat=60.0"
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["n_points"] == 3
        assert data["n_cells"] == 2
        # Without pyproj, coordinates are passed through as-is
        assert data["start"]["x"] == 10.0
        assert data["start"]["y"] == 20.0
        assert data["end"]["x"] == 50.0
        assert data["end"]["y"] == 60.0

    def test_no_layer_data_defaults_to_ones(self, client):
        """When cell_data has no 'layer' key, default to [1]*n_cells."""
        model = _make_model_with_strat()
        _set_model(model)

        n_cells = 3
        n_points = 4
        mock_mesh = MagicMock()
        mock_mesh.n_cells = n_cells
        mock_mesh.n_points = n_points
        mock_mesh.points = np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
                [30.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        mock_mesh.faces = np.array([3, 0, 1, 2, 3, 1, 2, 3, 3, 0, 2, 3], dtype=np.int32)
        mock_mesh.cell_data = {}  # No layer data

        mock_slicer = MagicMock()
        mock_slicer.create_cross_section.return_value = mock_mesh

        mock_pv_mesh = MagicMock()

        with (
            patch.object(model_state, "get_pyvista_3d", return_value=mock_pv_mesh),
            patch(
                "pyiwfm.visualization.webapi.slicing.SlicingController",
                return_value=mock_slicer,
            ),
            patch.dict(sys.modules, {"pyproj": None}),
        ):
            resp = client.get(
                "/api/slice/cross-section/json?start_lng=0.0&start_lat=0.0&end_lng=30.0&end_lat=0.0"
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["layer"] == [1, 1, 1]

    def test_distance_computation(self, client):
        """Verify distance from start is correctly computed for each point."""
        model = _make_model_with_strat()
        _set_model(model)

        # Start at (0, 0), points at known distances
        n_cells = 1
        n_points = 3
        mock_mesh = MagicMock()
        mock_mesh.n_cells = n_cells
        mock_mesh.n_points = n_points
        mock_mesh.points = np.array(
            [
                [0.0, 0.0, 0.0],  # distance=0
                [3.0, 4.0, 0.0],  # distance=5
                [6.0, 8.0, 0.0],  # distance=10
            ],
            dtype=np.float64,
        )
        mock_mesh.faces = np.array([3, 0, 1, 2], dtype=np.int32)
        mock_mesh.cell_data = {"layer": np.array([1])}

        mock_slicer = MagicMock()
        mock_slicer.create_cross_section.return_value = mock_mesh

        mock_pv_mesh = MagicMock()

        with (
            patch.object(model_state, "get_pyvista_3d", return_value=mock_pv_mesh),
            patch(
                "pyiwfm.visualization.webapi.slicing.SlicingController",
                return_value=mock_slicer,
            ),
            patch.dict(sys.modules, {"pyproj": None}),
        ):
            resp = client.get(
                "/api/slice/cross-section/json?start_lng=0.0&start_lat=0.0&end_lng=10.0&end_lat=0.0"
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["distance"] == [0.0, 5.0, 10.0]
        assert data["total_distance"] == 10.0

    def test_total_distance_computation(self, client):
        """Verify total_distance is computed from start to end points."""
        model = _make_model_with_strat()
        _set_model(model)

        mock_mesh = MagicMock()
        mock_mesh.n_cells = 1
        mock_mesh.n_points = 2
        mock_mesh.points = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]], dtype=np.float64)
        mock_mesh.faces = np.array([2, 0, 1], dtype=np.int32)
        mock_mesh.cell_data = {"layer": np.array([1])}

        mock_slicer = MagicMock()
        mock_slicer.create_cross_section.return_value = mock_mesh

        mock_pv_mesh = MagicMock()

        with (
            patch.object(model_state, "get_pyvista_3d", return_value=mock_pv_mesh),
            patch(
                "pyiwfm.visualization.webapi.slicing.SlicingController",
                return_value=mock_slicer,
            ),
            patch.dict(sys.modules, {"pyproj": None}),
        ):
            # start=(3.0, 0.0), end=(7.0, 3.0)
            # total_distance = sqrt(16 + 9) = 5.0
            resp = client.get(
                "/api/slice/cross-section/json?start_lng=3.0&start_lat=0.0&end_lng=7.0&end_lat=3.0"
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["total_distance"] == 5.0

    def test_missing_params_returns_422(self, client):
        """Missing required query parameters returns 422."""
        model = _make_model_with_strat()
        _set_model(model)
        resp = client.get("/api/slice/cross-section/json?start_lng=-121.0")
        assert resp.status_code == 422

    def test_polys_in_response(self, client):
        """Verify polys (faces) are included in the response."""
        model = _make_model_with_strat()
        _set_model(model)

        mock_mesh = MagicMock()
        mock_mesh.n_cells = 1
        mock_mesh.n_points = 3
        mock_mesh.points = np.array(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [5.0, 10.0, 0.0]],
            dtype=np.float64,
        )
        mock_mesh.faces = np.array([3, 0, 1, 2], dtype=np.int32)
        mock_mesh.cell_data = {"layer": np.array([1])}

        mock_slicer = MagicMock()
        mock_slicer.create_cross_section.return_value = mock_mesh
        mock_pv_mesh = MagicMock()

        with (
            patch.object(model_state, "get_pyvista_3d", return_value=mock_pv_mesh),
            patch(
                "pyiwfm.visualization.webapi.slicing.SlicingController",
                return_value=mock_slicer,
            ),
            patch.dict(sys.modules, {"pyproj": None}),
        ):
            resp = client.get(
                "/api/slice/cross-section/json?"
                "start_lng=0.0&start_lat=0.0&end_lng=10.0&end_lat=10.0"
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["polys"] == [3, 0, 1, 2]


# ===================================================================
# GET /api/slice/info
# ===================================================================


class TestGetSliceInfo:
    """Tests for GET /api/slice/info."""

    def test_no_model_returns_404(self, client):
        resp = client.get("/api/slice/info?axis=x&position=0.5")
        assert resp.status_code == 404
        assert "No model loaded" in resp.json()["detail"]

    def test_no_pyvista_returns_500(self, client):
        """500 when PyVista is not installed."""
        model = _make_mock_model(stratigraphy=MagicMock())
        _set_model(model)
        with patch.dict(sys.modules, {"pyvista": None}):
            resp = client.get("/api/slice/info?axis=x&position=0.5")
        assert resp.status_code == 500
        assert "PyVista required" in resp.json()["detail"]

    def test_no_stratigraphy_returns_400(self, client):
        """400 when model has no stratigraphy."""
        model = _make_mock_model(stratigraphy=None)
        _set_model(model)
        resp = client.get("/api/slice/info?axis=x&position=0.5")
        assert resp.status_code == 400
        assert "Stratigraphy required" in resp.json()["detail"]

    def test_x_axis(self, client):
        """Info for an X-axis slice returns n_cells, n_points, bounds."""
        model = _make_model_with_strat()
        _set_model(model)
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

    def test_y_axis(self, client):
        """Info for a Y-axis slice."""
        model = _make_model_with_strat()
        _set_model(model)
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

    def test_z_axis(self, client):
        """Info for a Z-axis slice."""
        model = _make_model_with_strat()
        _set_model(model)
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

    def test_empty_bounds_null(self, client):
        """Info for an empty slice returns bounds=null."""
        model = _make_model_with_strat()
        _set_model(model)
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


# ===================================================================
# _pyvista_to_vtu helper
# ===================================================================


class TestPyvistaToVtu:
    """Tests for the _pyvista_to_vtu helper function."""

    def test_polydata_input(self):
        """PolyData input is cast to UnstructuredGrid before writing."""
        import pyvista as pv

        from pyiwfm.visualization.webapi.routes.slices import _pyvista_to_vtu

        # Create a simple real PolyData triangle
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        faces = np.array([3, 0, 1, 2])
        poly = pv.PolyData(points, faces)

        result = _pyvista_to_vtu(poly)

        assert isinstance(result, bytes)
        assert len(result) > 0
        # VTU format starts with XML declaration or VTKFile tag
        text = result.decode("utf-8")
        assert "VTKFile" in text

    def test_unstructured_grid_input(self):
        """UnstructuredGrid input is used directly (not cast)."""
        import pyvista as pv

        from pyiwfm.visualization.webapi.routes.slices import _pyvista_to_vtu

        # Create a simple UnstructuredGrid via PolyData conversion
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        faces = np.array([3, 0, 1, 2])
        poly = pv.PolyData(points, faces)
        ug = poly.cast_to_unstructured_grid()

        result = _pyvista_to_vtu(ug)

        assert isinstance(result, bytes)
        assert len(result) > 0
        text = result.decode("utf-8")
        assert "VTKFile" in text


# ===================================================================
# SliceInfo model
# ===================================================================


class TestSliceInfoModel:
    """Tests for the SliceInfo Pydantic model."""

    def test_slice_info_model(self):
        """Test SliceInfo model creation."""
        from pyiwfm.visualization.webapi.routes.slices import SliceInfo

        info = SliceInfo(n_cells=10, n_points=20, bounds=[0, 1, 0, 1, 0, 1])
        assert info.n_cells == 10
        assert info.n_points == 20
        assert info.bounds == [0, 1, 0, 1, 0, 1]

    def test_slice_info_model_no_bounds(self):
        """Test SliceInfo model with bounds=None."""
        from pyiwfm.visualization.webapi.routes.slices import SliceInfo

        info = SliceInfo(n_cells=0, n_points=0, bounds=None)
        assert info.bounds is None
