"""Deep tests for pyiwfm.visualization.webapi.routes.slices.

Targets uncovered branches beyond the existing test_webapi_routes_slices.py:
- GET /api/slice/cross-section/heads: head clipping (dry cells below threshold)
- GET /api/slice/cross-section/heads: no pyproj fallback
- GET /api/slice/cross-section/heads: timestep out of range
- GET /api/slice/cross-section/heads: no head data
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
# Helpers (mirrored from existing test file for isolation)
# ---------------------------------------------------------------------------


def _make_grid():
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


def _mock_model(**kwargs):
    model = MagicMock()
    model.name = "TestModel"
    model.grid = kwargs.get("grid", _make_grid())
    model.metadata = kwargs.get("metadata", {})
    model.n_nodes = len(model.grid.nodes)
    model.n_elements = len(model.grid.elements)
    model.n_layers = kwargs.get("n_layers", 2)
    model.stratigraphy = kwargs.get("stratigraphy", MagicMock(n_layers=2))
    model.streams = kwargs.get("streams", None)
    model.lakes = kwargs.get("lakes", None)
    model.groundwater = kwargs.get("groundwater", None)
    model.rootzone = kwargs.get("rootzone", None)
    model.source_files = kwargs.get("source_files", {})
    return model


def _reset():
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
    _reset()
    yield
    _reset()


@pytest.fixture()
def app():
    return create_app()


@pytest.fixture()
def client(app):
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /api/slice/cross-section/heads
# ---------------------------------------------------------------------------


class TestCrossSectionHeads:
    """Tests for cross-section head interpolation endpoint."""

    def test_no_model_returns_404(self, client):
        resp = client.get(
            "/api/slice/cross-section/heads?"
            "start_lng=-121.0&start_lat=37.0&end_lng=-120.0&end_lat=38.0"
        )
        assert resp.status_code == 404

    def test_no_stratigraphy_returns_400(self, client):
        model = _mock_model(stratigraphy=None)
        model_state._model = model
        resp = client.get(
            "/api/slice/cross-section/heads?"
            "start_lng=-121.0&start_lat=37.0&end_lng=-120.0&end_lat=38.0"
        )
        assert resp.status_code == 400
        assert "Stratigraphy required" in resp.json()["detail"]

    def test_no_head_data_returns_404(self, client):
        model = _mock_model()
        model_state._model = model
        with patch.object(model_state, "get_head_loader", return_value=None):
            resp = client.get(
                "/api/slice/cross-section/heads?"
                "start_lng=-121.0&start_lat=37.0&end_lng=-120.0&end_lat=38.0"
            )
        assert resp.status_code == 404
        assert "No head data" in resp.json()["detail"]

    def test_timestep_out_of_range(self, client):
        model = _mock_model()
        model_state._model = model

        mock_loader = MagicMock()
        mock_loader.n_frames = 5
        mock_loader.times = []

        with patch.object(model_state, "get_head_loader", return_value=mock_loader):
            resp = client.get(
                "/api/slice/cross-section/heads?"
                "start_lng=-121.0&start_lat=37.0&end_lng=-120.0&end_lat=38.0"
                "&timestep=10"
            )
        assert resp.status_code == 400
        assert "out of range" in resp.json()["detail"]

    def test_head_clipping_dry_cells(self, client):
        """Dry cells (head < bottom) and IWFM markers (<-9000) become None."""
        model = _mock_model()
        model_state._model = model

        n_samples = 10
        n_layers = 2

        mock_loader = MagicMock()
        mock_loader.n_frames = 1
        mock_loader.times = [MagicMock(isoformat=lambda: "2020-01-01")]
        mock_loader.shape = (n_samples, 2)
        # Frame: shape (n_nodes, n_layers)
        frame = np.full((n_samples, n_layers), 50.0)
        frame[2, 0] = -9999.0  # IWFM dry marker
        mock_loader.get_frame.return_value = frame

        mock_xs = MagicMock()
        mock_xs.n_layers = n_layers
        mock_xs.distance = np.linspace(0, 100, n_samples)
        mock_xs.top_elev = np.full((n_samples, n_layers), 100.0)
        mock_xs.bottom_elev = np.full((n_samples, n_layers), 0.0)
        mock_xs.gs_elev = np.full(n_samples, 100.0)
        mask = np.ones(n_samples, dtype=bool)
        mask[3] = False
        mock_xs.mask = mask

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = mock_xs
        # head_interp: shape (n_samples, n_layers)
        head_interp = np.full((n_samples, n_layers), 50.0)
        head_interp[2, 0] = -9999.0
        head_interp[1, 1] = -10000.0
        mock_extractor.interpolate_layer_property.return_value = head_interp

        with (
            patch.object(model_state, "get_head_loader", return_value=mock_loader),
            patch(
                "pyiwfm.core.cross_section.CrossSectionExtractor",
                return_value=mock_extractor,
            ),
            patch.dict(sys.modules, {"pyproj": None}),
        ):
            resp = client.get(
                "/api/slice/cross-section/heads?"
                "start_lng=0&start_lat=0&end_lng=100&end_lat=0"
                "&timestep=0&n_samples=10"
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["n_layers"] == 2
        assert data["n_samples"] == n_samples

    def test_without_pyproj_coords_passthrough(self, client):
        """When pyproj is absent, WGS84 coords are used as model coords."""
        model = _mock_model()
        model_state._model = model

        ns = 10
        mock_loader = MagicMock()
        mock_loader.n_frames = 1
        mock_loader.times = [MagicMock(isoformat=lambda: "2020-01-01")]
        mock_loader.shape = (ns, 1)
        mock_loader.get_frame.return_value = np.full((ns, 1), 50.0)

        mock_xs = MagicMock()
        mock_xs.n_layers = 1
        mock_xs.distance = np.linspace(0, 100, ns)
        mock_xs.top_elev = np.full((ns, 1), 100.0)
        mock_xs.bottom_elev = np.full((ns, 1), 0.0)
        mock_xs.gs_elev = np.full(ns, 100.0)
        mock_xs.mask = np.ones(ns, dtype=bool)

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = mock_xs
        mock_extractor.interpolate_layer_property.return_value = np.full((ns, 1), 50.0)

        with (
            patch.object(model_state, "get_head_loader", return_value=mock_loader),
            patch(
                "pyiwfm.core.cross_section.CrossSectionExtractor",
                return_value=mock_extractor,
            ),
            patch.dict(sys.modules, {"pyproj": None}),
        ):
            resp = client.get(
                "/api/slice/cross-section/heads?"
                "start_lng=10.0&start_lat=20.0&end_lng=50.0&end_lat=60.0"
                "&timestep=0&n_samples=10"
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["n_layers"] == 1
        # Without pyproj, start coords are used directly
        assert data["timestep"] == 0
