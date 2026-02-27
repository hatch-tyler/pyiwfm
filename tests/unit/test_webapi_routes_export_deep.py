"""Deep coverage tests for webapi export routes — plot types, SVG format,
hydrograph CSV for subsidence/tile_drain, heads CSV export.

Targets uncovered paths in src/pyiwfm/visualization/webapi/routes/export.py.
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

EXPORT_PATCH = "pyiwfm.visualization.webapi.routes.export.model_state"
PLOTTING_MOD = "pyiwfm.visualization.plotting"


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
    n_frames: int = 3,
    n_nodes: int = 4,
    n_layers: int = 2,
) -> ModelState:
    """Build a ModelState with mocked model and head_loader."""
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
    state._head_loader = loader
    return state


def _csv_lines(text: str) -> list[str]:
    """Split CSV response text into lines, stripping \\r for Windows compat."""
    return [line.rstrip("\r") for line in text.strip().split("\n")]


# ---------------------------------------------------------------------------
# 1. export_heads_csv() — success path
# ---------------------------------------------------------------------------


class TestExportHeadsCsv:
    """Tests for GET /api/export/heads-csv."""

    def test_heads_csv_success(self) -> None:
        state = _make_state_with_head_loader()
        app = create_app()
        with patch(EXPORT_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/export/heads-csv?timestep=0&layer=1")
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "text/csv; charset=utf-8"
            assert "attachment" in resp.headers["content-disposition"]
            lines = _csv_lines(resp.text)
            assert lines[0] == "node_id,head_ft"
            assert len(lines) == 5  # header + 4 nodes

    def test_heads_csv_timestep_out_of_range(self) -> None:
        state = _make_state_with_head_loader()
        app = create_app()
        with patch(EXPORT_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/export/heads-csv?timestep=999&layer=1")
            assert resp.status_code == 400

    def test_heads_csv_layer_out_of_range(self) -> None:
        state = _make_state_with_head_loader()
        app = create_app()
        with patch(EXPORT_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/export/heads-csv?timestep=0&layer=99")
            assert resp.status_code == 400

    def test_heads_csv_no_loader(self) -> None:
        state = ModelState()
        state._model = MagicMock()
        state._model.metadata = {}
        app = create_app()
        with patch(EXPORT_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/export/heads-csv?timestep=0&layer=1")
            assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 2. export_hydrograph_csv() — subsidence and tile_drain types
# ---------------------------------------------------------------------------


class TestExportHydrographCsv:
    """Tests for GET /api/export/hydrograph-csv with various types."""

    def _make_state_with_reader(self, reader_attr: str) -> tuple[ModelState, MagicMock]:
        state = ModelState()
        model = MagicMock()
        model.name = "TestModel"
        model.grid = _make_grid()
        model.metadata = {}
        model.groundwater = None
        state._model = model

        reader = MagicMock()
        reader.n_timesteps = 5
        reader.n_columns = 3
        reader.get_time_series = MagicMock(
            return_value=(
                ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01"],
                [1.1, 2.2, 3.3, 4.4, 5.5],
            )
        )
        setattr(state, reader_attr, reader)
        return state, reader

    def test_subsidence_csv_success(self) -> None:
        state, _ = self._make_state_with_reader("_subsidence_reader")
        app = create_app()
        with patch(EXPORT_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/export/hydrograph-csv?type=subsidence&location_id=1")
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "text/csv; charset=utf-8"
            lines = _csv_lines(resp.text)
            assert lines[0] == "datetime,subsidence_ft"
            assert len(lines) == 6  # header + 5 rows

    def test_subsidence_csv_out_of_range(self) -> None:
        state, _ = self._make_state_with_reader("_subsidence_reader")
        app = create_app()
        with patch(EXPORT_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/export/hydrograph-csv?type=subsidence&location_id=99")
            assert resp.status_code == 404

    def test_tile_drain_csv_success(self) -> None:
        state, _ = self._make_state_with_reader("_tile_drain_reader")
        app = create_app()
        with patch(EXPORT_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/export/hydrograph-csv?type=tile_drain&location_id=1")
            assert resp.status_code == 200
            lines = _csv_lines(resp.text)
            assert lines[0] == "datetime,flow_volume"
            assert len(lines) == 6

    def test_tile_drain_csv_out_of_range(self) -> None:
        state, _ = self._make_state_with_reader("_tile_drain_reader")
        app = create_app()
        with patch(EXPORT_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/export/hydrograph-csv?type=tile_drain&location_id=99")
            assert resp.status_code == 404

    def test_unknown_type_returns_400(self) -> None:
        state = ModelState()
        state._model = MagicMock()
        state._model.metadata = {}
        app = create_app()
        with patch(EXPORT_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/export/hydrograph-csv?type=bogus&location_id=1")
            assert resp.status_code == 400
            assert "Unknown type" in resp.json()["detail"]

    def test_gw_csv_with_phys_locs(self) -> None:
        state, reader = self._make_state_with_reader("_gw_hydrograph_reader")
        phys_locs = [{"columns": [(0, 1)], "name": "W1", "node_id": 1}]
        state.get_gw_physical_locations = MagicMock(return_value=phys_locs)

        app = create_app()
        with patch(EXPORT_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/export/hydrograph-csv?type=gw&location_id=1")
            assert resp.status_code == 200
            lines = _csv_lines(resp.text)
            assert lines[0] == "datetime,head_ft"


# ---------------------------------------------------------------------------
# 3. export_plot() — mesh, heads, SVG format, unknown type
# ---------------------------------------------------------------------------


class TestExportPlot:
    """Tests for GET /api/export/plot/{plot_type}."""

    def _make_mock_fig(self, content: bytes = b"\x89PNG fake data") -> MagicMock:
        """Create a mock matplotlib Figure that writes bytes to a buffer."""
        fig = MagicMock()

        def _savefig(buf, format=None, dpi=None, bbox_inches=None):
            buf.write(content)

        fig.savefig = MagicMock(side_effect=_savefig)
        return fig

    def test_plot_mesh_png(self) -> None:
        state = _make_state_with_head_loader()
        mock_fig = self._make_mock_fig()
        mock_ax = MagicMock()

        app = create_app()
        with (
            patch(EXPORT_PATCH, state),
            patch(
                f"{PLOTTING_MOD}.plot_mesh",
                return_value=(mock_fig, mock_ax),
            ),
        ):
            client = TestClient(app)
            resp = client.get("/api/export/plot/mesh")
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "image/png"

    def test_plot_mesh_svg_format(self) -> None:
        state = _make_state_with_head_loader()
        mock_fig = self._make_mock_fig(content=b"<svg>mock</svg>")
        mock_ax = MagicMock()

        app = create_app()
        with (
            patch(EXPORT_PATCH, state),
            patch(
                f"{PLOTTING_MOD}.plot_mesh",
                return_value=(mock_fig, mock_ax),
            ),
        ):
            client = TestClient(app)
            resp = client.get("/api/export/plot/mesh?format=svg")
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "image/svg+xml"

    def test_plot_heads_success(self) -> None:
        state = _make_state_with_head_loader()
        mock_fig = self._make_mock_fig(content=b"\x89PNG head plot")
        mock_ax = MagicMock()

        app = create_app()
        with (
            patch(EXPORT_PATCH, state),
            patch(
                f"{PLOTTING_MOD}.plot_scalar_field",
                return_value=(mock_fig, mock_ax),
            ),
        ):
            client = TestClient(app)
            resp = client.get("/api/export/plot/heads?layer=1&timestep=0")
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "image/png"

    def test_plot_unknown_type_returns_400(self) -> None:
        state = _make_state_with_head_loader()
        app = create_app()
        with patch(EXPORT_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/export/plot/unknown_type")
            assert resp.status_code == 400
            assert "Unknown plot type" in resp.json()["detail"]

    def test_plot_no_model(self) -> None:
        state = ModelState()
        app = create_app()
        with patch(EXPORT_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/export/plot/mesh")
            assert resp.status_code == 404

    def test_plot_heads_no_loader(self) -> None:
        state = ModelState()
        model = MagicMock()
        model.grid = _make_grid()
        model.metadata = {}
        state._model = model

        app = create_app()
        with patch(EXPORT_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/export/plot/heads")
            assert resp.status_code == 404
            assert "No head data" in resp.json()["detail"]
