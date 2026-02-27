"""Deep coverage tests for webapi zbudgets routes — column filtering,
element centroid fallback, summary defaults, NaN/Inf sanitization.

Targets uncovered paths in src/pyiwfm/visualization/webapi/routes/zbudgets.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi", reason="FastAPI not available")
pytest.importorskip("pydantic", reason="Pydantic not available")

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from pyiwfm.core.mesh import AppGrid, Element, Node
from pyiwfm.visualization.webapi.config import ModelState
from pyiwfm.visualization.webapi.server import create_app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ZBUDGETS_PATCH = "pyiwfm.visualization.webapi.routes.zbudgets.model_state"


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


def _make_state_with_zbudget_reader() -> tuple[ModelState, MagicMock]:
    """Build a ModelState with a mocked zbudget reader attached."""
    state = ModelState()
    model = MagicMock()
    model.name = "TestModel"
    model.grid = _make_grid()
    model.n_nodes = 4
    model.n_elements = 1
    model.n_layers = 1
    model.groundwater = None
    model.streams = None
    model.lakes = None
    model.stratigraphy = None
    model.rootzone = None
    model.metadata = {"length_unit": "FT"}
    state._model = model

    # Build a mock zbudget reader
    reader = MagicMock()
    reader.zones = ["Zone A", "Zone B"]
    reader.data_names = ["Pumping (-)", "Deep Percolation (+)", "Storage (+/-)"]

    header = MagicMock()
    header.n_elements = 100
    header.n_data = 3
    header.time_unit = "1MON"
    reader.header = header
    reader.n_timesteps = 12

    # Build a sample DataFrame from get_dataframe
    dates = pd.date_range("2020-01-01", periods=12, freq="MS")
    df = pd.DataFrame(
        {
            "Pumping (-)": np.random.default_rng(0).uniform(-100, 0, 12),
            "Deep Percolation (+)": np.random.default_rng(1).uniform(0, 50, 12),
            "Storage (+/-)": np.random.default_rng(2).uniform(-20, 20, 12),
        },
        index=dates,
    )
    reader.get_dataframe = MagicMock(return_value=df)

    # Zone info dict for _sync_active_zones
    reader._zone_info = {}

    return state, reader


# ---------------------------------------------------------------------------
# 1. _safe_float() and _sanitize_values() — NaN/Inf handling
# ---------------------------------------------------------------------------


class TestSanitizationHelpers:
    """Test the NaN/Inf sanitization helpers in the zbudgets module."""

    def test_safe_float_nan(self) -> None:
        from pyiwfm.visualization.webapi.routes.zbudgets import _safe_float

        assert _safe_float(float("nan")) is None

    def test_safe_float_inf(self) -> None:
        from pyiwfm.visualization.webapi.routes.zbudgets import _safe_float

        assert _safe_float(float("inf")) is None

    def test_safe_float_neg_inf(self) -> None:
        from pyiwfm.visualization.webapi.routes.zbudgets import _safe_float

        assert _safe_float(float("-inf")) is None

    def test_safe_float_normal(self) -> None:
        from pyiwfm.visualization.webapi.routes.zbudgets import _safe_float

        assert _safe_float(3.14) == 3.14

    def test_safe_float_none(self) -> None:
        from pyiwfm.visualization.webapi.routes.zbudgets import _safe_float

        assert _safe_float(None) is None  # type: ignore[arg-type]

    def test_sanitize_values_mixed(self) -> None:
        from pyiwfm.visualization.webapi.routes.zbudgets import _sanitize_values

        result = _sanitize_values([1.0, float("nan"), 3.0, float("inf"), float("-inf")])
        assert result == [1.0, None, 3.0, None, None]

    def test_sanitize_values_integers(self) -> None:
        from pyiwfm.visualization.webapi.routes.zbudgets import _sanitize_values

        result = _sanitize_values([1, 2, 3])
        assert result == [1, 2, 3]


# ---------------------------------------------------------------------------
# 2. get_zbudget_data() — column filtering
# ---------------------------------------------------------------------------


class TestZBudgetDataColumnFiltering:
    """Tests for /api/zbudgets/{type}/data with column selection."""

    def test_data_all_columns(self) -> None:
        state, reader = _make_state_with_zbudget_reader()
        state._zbudget_readers = {"gw": reader}

        app = create_app()
        with patch(ZBUDGETS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/zbudgets/gw/data?zone=Zone%20A")
            assert resp.status_code == 200
            data = resp.json()
            assert data["location"] == "Zone A"
            assert len(data["columns"]) == 3
            assert len(data["times"]) == 12

    def test_data_filtered_columns(self) -> None:
        state, reader = _make_state_with_zbudget_reader()
        state._zbudget_readers = {"gw": reader}

        app = create_app()
        with patch(ZBUDGETS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/zbudgets/gw/data?zone=Zone%20A&columns=0,2")
            assert resp.status_code == 200
            data = resp.json()
            # Only columns at indices 0 and 2 should appear
            col_names = [c["name"] for c in data["columns"]]
            assert "Pumping (-)" in col_names
            assert "Storage (+/-)" in col_names
            assert "Deep Percolation (+)" not in col_names

    def test_data_default_zone_selection(self) -> None:
        state, reader = _make_state_with_zbudget_reader()
        state._zbudget_readers = {"gw": reader}

        app = create_app()
        with patch(ZBUDGETS_PATCH, state):
            client = TestClient(app)
            # No zone param - should use first zone
            resp = client.get("/api/zbudgets/gw/data")
            assert resp.status_code == 200
            data = resp.json()
            assert data["location"] == "Zone A"

    def test_data_reader_not_found(self) -> None:
        state = ModelState()
        state._model = MagicMock()
        state._model.metadata = {}

        app = create_app()
        with patch(ZBUDGETS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/zbudgets/nonexistent/data")
            assert resp.status_code == 404

    def test_data_exception_from_get_dataframe(self) -> None:
        state, reader = _make_state_with_zbudget_reader()
        reader.get_dataframe = MagicMock(side_effect=KeyError("bad zone"))
        state._zbudget_readers = {"gw": reader}

        app = create_app()
        with patch(ZBUDGETS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/zbudgets/gw/data?zone=BadZone")
            assert resp.status_code == 400

    def test_data_units_metadata_meters(self) -> None:
        state, reader = _make_state_with_zbudget_reader()
        state._model.metadata = {"length_unit": "METERS"}
        state._zbudget_readers = {"gw": reader}

        app = create_app()
        with patch(ZBUDGETS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/zbudgets/gw/data?zone=Zone%20A")
            assert resp.status_code == 200
            data = resp.json()
            assert data["units_metadata"]["source_volume_unit"] == "M3"
            assert data["units_metadata"]["source_area_unit"] == "M2"


# ---------------------------------------------------------------------------
# 3. get_zbudget_elements() — without pyproj (fallback path)
# ---------------------------------------------------------------------------


class TestZBudgetElements:
    """Tests for /api/zbudgets/elements endpoint."""

    def test_elements_no_transformer(self) -> None:
        """When pyproj is not available, lng/lat = cx/cy."""
        state = ModelState()
        model = MagicMock()
        model.grid = _make_grid()
        model.metadata = {}
        state._model = model
        state._transformer = None

        app = create_app()
        with (
            patch(ZBUDGETS_PATCH, state),
            patch("pyiwfm.visualization.webapi.routes.zbudgets.model_state", state),
            patch.dict("sys.modules", {"pyproj": None}),
        ):
            client = TestClient(app)
            resp = client.get("/api/zbudgets/elements")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) == 1
            elem = data[0]
            assert elem["id"] == 1
            # Without pyproj, centroid is raw model coords
            assert "centroid" in elem
            assert elem["subregion"] == 1


# ---------------------------------------------------------------------------
# 4. get_zbudget_summary() — default zone, reader not found
# ---------------------------------------------------------------------------


class TestZBudgetSummary:
    """Tests for /api/zbudgets/{type}/summary endpoint."""

    def test_summary_default_zone(self) -> None:
        state, reader = _make_state_with_zbudget_reader()
        state._zbudget_readers = {"gw": reader}

        app = create_app()
        with patch(ZBUDGETS_PATCH, state):
            client = TestClient(app)
            # No zone specified — should use first zone
            resp = client.get("/api/zbudgets/gw/summary")
            assert resp.status_code == 200
            data = resp.json()
            assert data["location"] == "Zone A"
            assert data["n_timesteps"] == 12
            assert "totals" in data
            assert "averages" in data
            assert "Pumping (-)" in data["totals"]

    def test_summary_reader_not_found_404(self) -> None:
        state = ModelState()
        state._model = MagicMock()
        state._model.metadata = {}

        app = create_app()
        with patch(ZBUDGETS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/zbudgets/nonexistent/summary")
            assert resp.status_code == 404
            assert "not available" in resp.json()["detail"]

    def test_summary_bad_zone_raises_400(self) -> None:
        state, reader = _make_state_with_zbudget_reader()
        reader.get_dataframe = MagicMock(side_effect=KeyError("zone not found"))
        state._zbudget_readers = {"gw": reader}

        app = create_app()
        with patch(ZBUDGETS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/zbudgets/gw/summary?zone=BadZone")
            assert resp.status_code == 400
