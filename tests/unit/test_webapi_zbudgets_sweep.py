"""Sweep tests for webapi zbudgets routes — targeting remaining uncovered paths.

Covers:
- _sync_active_zones full function (lines 108-125)
- get_zbudget_elements CRS transformation paths (lines 167-192)
- upload_zone_file mock file upload (lines 318-425)
- get_zbudget_data exception paths, column filtering, unit branches (lines 474-534)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi", reason="FastAPI not available")
pytest.importorskip("pydantic", reason="Pydantic not available")

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

    reader = MagicMock()
    reader.zones = ["Zone A", "Zone B"]
    reader.data_names = ["Pumping (-)", "Deep Percolation (+)", "Storage (+/-)"]

    header = MagicMock()
    header.n_elements = 100
    header.n_data = 3
    header.time_unit = "1MON"
    reader.header = header
    reader.n_timesteps = 12

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
    reader._zone_info = {}

    return state, reader


# ---------------------------------------------------------------------------
# 1. _sync_active_zones — full function (lines 108-125)
# ---------------------------------------------------------------------------


class TestSyncActiveZones:
    """Tests for the _sync_active_zones helper."""

    def test_sync_injects_new_zones(self) -> None:
        from pyiwfm.visualization.webapi.routes.zbudgets import _sync_active_zones

        reader = MagicMock()
        reader._zone_info = {}

        # Build a mock zone definition
        zone1 = MagicMock()
        zone1.id = 1
        zone1.name = "North Basin"
        zone1.n_elements = 5
        zone1.elements = [1, 2, 3, 4, 5]
        zone1.area = 1500.0

        zone_def = MagicMock()
        zone_def.iter_zones.return_value = [zone1]

        state = ModelState()
        state._active_zone_def = zone_def

        with patch(ZBUDGETS_PATCH, state):
            _sync_active_zones(reader)

        assert "North Basin" in reader._zone_info
        info = reader._zone_info["North Basin"]
        assert info.id == 1
        assert info.name == "North Basin"
        assert info.n_elements == 5
        assert info.element_ids == [1, 2, 3, 4, 5]
        assert info.area == 1500.0

    def test_sync_overwrites_empty_existing_zone(self) -> None:
        from pyiwfm.io.zbudget import ZoneInfo as ZBZoneInfo
        from pyiwfm.visualization.webapi.routes.zbudgets import _sync_active_zones

        # Pre-existing zone with no element_ids
        existing = ZBZoneInfo(id=1, name="North Basin", n_elements=0, element_ids=[], area=0.0)
        reader = MagicMock()
        reader._zone_info = {"North Basin": existing}

        zone1 = MagicMock()
        zone1.id = 1
        zone1.name = "North Basin"
        zone1.n_elements = 3
        zone1.elements = [10, 20, 30]
        zone1.area = 900.0

        zone_def = MagicMock()
        zone_def.iter_zones.return_value = [zone1]

        state = ModelState()
        state._active_zone_def = zone_def

        with patch(ZBUDGETS_PATCH, state):
            _sync_active_zones(reader)

        info = reader._zone_info["North Basin"]
        assert info.element_ids == [10, 20, 30]
        assert info.area == 900.0

    def test_sync_does_not_overwrite_populated_zone(self) -> None:
        from pyiwfm.io.zbudget import ZoneInfo as ZBZoneInfo
        from pyiwfm.visualization.webapi.routes.zbudgets import _sync_active_zones

        existing = ZBZoneInfo(id=1, name="Zone X", n_elements=2, element_ids=[100, 200], area=500.0)
        reader = MagicMock()
        reader._zone_info = {"Zone X": existing}

        zone1 = MagicMock()
        zone1.id = 1
        zone1.name = "Zone X"
        zone1.n_elements = 1
        zone1.elements = [999]
        zone1.area = 100.0

        zone_def = MagicMock()
        zone_def.iter_zones.return_value = [zone1]

        state = ModelState()
        state._active_zone_def = zone_def

        with patch(ZBUDGETS_PATCH, state):
            _sync_active_zones(reader)

        # Should NOT overwrite since existing has element_ids
        info = reader._zone_info["Zone X"]
        assert info.element_ids == [100, 200]

    def test_sync_no_zone_definition(self) -> None:
        from pyiwfm.visualization.webapi.routes.zbudgets import _sync_active_zones

        reader = MagicMock()
        reader._zone_info = {}

        state = ModelState()
        state._active_zone_def = None

        with patch(ZBUDGETS_PATCH, state):
            _sync_active_zones(reader)

        # No zones injected
        assert len(reader._zone_info) == 0


# ---------------------------------------------------------------------------
# 2. get_zbudget_elements — CRS transformation paths (lines 167-192)
# ---------------------------------------------------------------------------


class TestZBudgetElementsCRS:
    """Tests for /api/zbudgets/elements with CRS reprojection."""

    def test_elements_with_transformer(self) -> None:
        """When transformer is already set, use it."""
        state = ModelState()
        model = MagicMock()
        model.grid = _make_grid()
        model.metadata = {}
        state._model = model

        # Mock transformer that shifts coords
        transformer = MagicMock()
        transformer.transform = MagicMock(side_effect=lambda x, y: (x + 1000, y + 2000))
        state._transformer = transformer

        app = create_app()
        with patch(ZBUDGETS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/zbudgets/elements")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) == 1
            centroid = data[0]["centroid"]
            # Centroid of (0,0),(100,0),(100,100),(0,100) = (50,50)
            # After transform: (1050, 2050)
            assert centroid[0] == 1050
            assert centroid[1] == 2050

    def test_elements_pyproj_import_creates_transformer(self) -> None:
        """When transformer is None, pyproj creates one."""
        state = ModelState()
        model = MagicMock()
        model.grid = _make_grid()
        model.metadata = {}
        state._model = model
        state._transformer = None
        state._crs = "EPSG:4326"  # Set a CRS

        mock_transformer = MagicMock()
        mock_transformer.transform = MagicMock(side_effect=lambda x, y: (x, y))

        mock_pyproj = MagicMock()
        mock_pyproj.Transformer.from_crs.return_value = mock_transformer

        app = create_app()
        with (
            patch(ZBUDGETS_PATCH, state),
            patch.dict("sys.modules", {"pyproj": mock_pyproj}),
        ):
            client = TestClient(app)
            resp = client.get("/api/zbudgets/elements")
            assert resp.status_code == 200

    def test_elements_no_grid(self) -> None:
        """When model has no grid, return 404."""
        state = ModelState()
        model = MagicMock()
        model.grid = None
        model.metadata = {}
        state._model = model

        app = create_app()
        with patch(ZBUDGETS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/zbudgets/elements")
            assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 3. upload_zone_file (lines 318-425)
# ---------------------------------------------------------------------------


class TestUploadZoneFile:
    """Tests for /api/zbudgets/upload-zones endpoint."""

    def test_upload_unsupported_format(self) -> None:
        state = ModelState()
        model = MagicMock()
        model.grid = _make_grid()
        model.metadata = {}
        state._model = model

        app = create_app()
        with patch(ZBUDGETS_PATCH, state):
            client = TestClient(app)
            resp = client.post(
                "/api/zbudgets/upload-zones",
                files={"file": ("test.csv", b"a,b,c\n1,2,3", "text/csv")},
            )
            assert resp.status_code == 400
            assert "Unsupported file type" in resp.json()["detail"]

    def test_upload_no_model(self) -> None:
        state = ModelState()
        state._model = None

        app = create_app()
        with patch(ZBUDGETS_PATCH, state):
            client = TestClient(app)
            resp = client.post(
                "/api/zbudgets/upload-zones",
                files={"file": ("zones.geojson", b"{}", "application/json")},
            )
            assert resp.status_code == 404

    def test_upload_geojson_success(self) -> None:
        """Upload a valid GeoJSON that spatially joins with grid elements."""
        gpd = pytest.importorskip("geopandas", reason="geopandas not available")
        pytest.importorskip("shapely", reason="shapely not available")

        from shapely.geometry import Polygon

        state = ModelState()
        model = MagicMock()
        model.grid = _make_grid()
        model.metadata = {}
        state._model = model

        # Pre-set a mock transformer so the code path exercises the transform
        mock_xfm = MagicMock()
        mock_xfm.transform = MagicMock(side_effect=lambda x, y: (x, y))
        state._transformer = mock_xfm

        # Build a GeoJSON polygon that covers the grid element centroid at (50,50)
        poly = Polygon([(0, 0), (200, 0), (200, 200), (0, 200)])
        gdf = gpd.GeoDataFrame(
            {"zone_name": ["Big Zone"]},
            geometry=[poly],
            crs="EPSG:4326",
        )
        geojson_bytes = gdf.to_json().encode("utf-8")

        app = create_app()
        with patch(ZBUDGETS_PATCH, state):
            client = TestClient(app)
            resp = client.post(
                "/api/zbudgets/upload-zones",
                files={"file": ("zones.geojson", geojson_bytes, "application/json")},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "fields" in data
            assert "zones" in data
            assert len(data["zones"]) >= 1
            assert "zone_name" in data["fields"]

    def test_upload_empty_geojson(self) -> None:
        """Upload GeoJSON with no features."""
        gpd = pytest.importorskip("geopandas", reason="geopandas not available")

        gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        geojson_bytes = gdf.to_json().encode("utf-8")

        state = ModelState()
        model = MagicMock()
        model.grid = _make_grid()
        model.metadata = {}
        state._model = model

        app = create_app()
        with patch(ZBUDGETS_PATCH, state):
            client = TestClient(app)
            resp = client.post(
                "/api/zbudgets/upload-zones",
                files={"file": ("empty.geojson", geojson_bytes, "application/json")},
            )
            assert resp.status_code == 400
            assert "no features" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 4. get_zbudget_data — exception paths, column filtering, unit branches
# ---------------------------------------------------------------------------


class TestZBudgetDataExceptionPaths:
    """Tests for error handling in /api/zbudgets/{type}/data."""

    def test_data_unexpected_exception_500(self) -> None:
        """Unexpected exception -> HTTP 500."""
        state, reader = _make_state_with_zbudget_reader()
        reader.get_dataframe = MagicMock(side_effect=RuntimeError("unexpected"))
        state._zbudget_readers = {"gw": reader}

        app = create_app()
        with patch(ZBUDGETS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/zbudgets/gw/data?zone=Zone%20A")
            assert resp.status_code == 500
            assert "Internal error" in resp.json()["detail"]

    def test_data_value_error_400(self) -> None:
        """ValueError -> HTTP 400."""
        state, reader = _make_state_with_zbudget_reader()
        reader.get_dataframe = MagicMock(side_effect=ValueError("bad value"))
        state._zbudget_readers = {"gw": reader}

        app = create_app()
        with patch(ZBUDGETS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/zbudgets/gw/data?zone=Zone%20A")
            assert resp.status_code == 400

    def test_data_index_error_400(self) -> None:
        """IndexError -> HTTP 400."""
        state, reader = _make_state_with_zbudget_reader()
        reader.get_dataframe = MagicMock(side_effect=IndexError("out of range"))
        state._zbudget_readers = {"gw": reader}

        app = create_app()
        with patch(ZBUDGETS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/zbudgets/gw/data?zone=Zone%20A")
            assert resp.status_code == 400

    def test_data_invalid_column_indices_400(self) -> None:
        """Non-integer column indices -> HTTP 400."""
        state, reader = _make_state_with_zbudget_reader()
        state._zbudget_readers = {"gw": reader}

        app = create_app()
        with patch(ZBUDGETS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/zbudgets/gw/data?zone=Zone%20A&columns=abc,xyz")
            assert resp.status_code == 400
            assert "Invalid column" in resp.json()["detail"]

    def test_data_no_zones_available_400(self) -> None:
        """No zone specified and reader has no zones -> HTTP 400."""
        state, reader = _make_state_with_zbudget_reader()
        reader.zones = []  # No zones available
        state._zbudget_readers = {"gw": reader}

        app = create_app()
        with patch(ZBUDGETS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/zbudgets/gw/data")
            assert resp.status_code == 400
            assert "No zone" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# 5. get_zbudget_data — unit logic (lines 529-534)
# ---------------------------------------------------------------------------


class TestZBudgetDataUnitLogic:
    """Tests for unit metadata in zbudget data response."""

    def test_meters_unit(self) -> None:
        state, reader = _make_state_with_zbudget_reader()
        state._model.metadata = {"length_unit": "M"}
        state._zbudget_readers = {"gw": reader}

        app = create_app()
        with patch(ZBUDGETS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/zbudgets/gw/data?zone=Zone%20A")
            assert resp.status_code == 200
            data = resp.json()
            assert data["units_metadata"]["source_volume_unit"] == "M3"
            assert data["units_metadata"]["source_area_unit"] == "M2"

    def test_meter_singular_unit(self) -> None:
        state, reader = _make_state_with_zbudget_reader()
        state._model.metadata = {"length_unit": "METER"}
        state._zbudget_readers = {"gw": reader}

        app = create_app()
        with patch(ZBUDGETS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/zbudgets/gw/data?zone=Zone%20A")
            assert resp.status_code == 200
            data = resp.json()
            assert data["units_metadata"]["source_volume_unit"] == "M3"

    def test_feet_unit(self) -> None:
        state, reader = _make_state_with_zbudget_reader()
        state._model.metadata = {"length_unit": "FEET"}
        state._zbudget_readers = {"gw": reader}

        app = create_app()
        with patch(ZBUDGETS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/zbudgets/gw/data?zone=Zone%20A")
            assert resp.status_code == 200
            data = resp.json()
            assert data["units_metadata"]["source_volume_unit"] == "FT3"

    def test_foot_unit(self) -> None:
        state, reader = _make_state_with_zbudget_reader()
        state._model.metadata = {"length_unit": "FOOT"}
        state._zbudget_readers = {"gw": reader}

        app = create_app()
        with patch(ZBUDGETS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/zbudgets/gw/data?zone=Zone%20A")
            assert resp.status_code == 200
            data = resp.json()
            assert data["units_metadata"]["source_volume_unit"] == "FT3"

    def test_unknown_unit_defaults_to_ft(self) -> None:
        state, reader = _make_state_with_zbudget_reader()
        state._model.metadata = {"length_unit": "CUBITS"}
        state._zbudget_readers = {"gw": reader}

        app = create_app()
        with patch(ZBUDGETS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/zbudgets/gw/data?zone=Zone%20A")
            assert resp.status_code == 200
            data = resp.json()
            assert data["units_metadata"]["source_volume_unit"] == "FT3"
            assert data["units_metadata"]["source_area_unit"] == "SQ.FT."

    def test_volume_factor_applied(self) -> None:
        state, reader = _make_state_with_zbudget_reader()
        state._zbudget_readers = {"gw": reader}

        app = create_app()
        with patch(ZBUDGETS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/zbudgets/gw/data?zone=Zone%20A&volume_factor=43560")
            assert resp.status_code == 200
            resp.json()  # ensure response is valid JSON
            # reader.get_dataframe should have been called with volume_factor
            reader.get_dataframe.assert_called_with("Zone A", volume_factor=43560.0)

    def test_data_non_datetime_index(self) -> None:
        """DataFrame with non-datetime index uses str() for time strings."""
        state, reader = _make_state_with_zbudget_reader()
        # Override get_dataframe to return non-datetime index
        df = pd.DataFrame(
            {"Col A": [1.0, 2.0]},
            index=["step_0", "step_1"],
        )
        reader.get_dataframe = MagicMock(return_value=df)
        state._zbudget_readers = {"gw": reader}

        app = create_app()
        with patch(ZBUDGETS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/zbudgets/gw/data?zone=Zone%20A")
            assert resp.status_code == 200
            data = resp.json()
            assert data["times"] == ["step_0", "step_1"]


# ---------------------------------------------------------------------------
# 6. get_zbudget_columns
# ---------------------------------------------------------------------------


class TestZBudgetColumns:
    """Tests for /api/zbudgets/{type}/columns."""

    def test_columns_success(self) -> None:
        state, reader = _make_state_with_zbudget_reader()
        state._zbudget_readers = {"gw": reader}

        app = create_app()
        with patch(ZBUDGETS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/zbudgets/gw/columns")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["columns"]) == 3
            assert data["columns"][0]["name"] == "Pumping (-)"

    def test_columns_not_found(self) -> None:
        state = ModelState()
        state._model = MagicMock()
        state._model.metadata = {}

        app = create_app()
        with patch(ZBUDGETS_PATCH, state):
            client = TestClient(app)
            resp = client.get("/api/zbudgets/unknown/columns")
            assert resp.status_code == 404
