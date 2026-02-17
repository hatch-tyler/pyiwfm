"""Comprehensive tests for the FastAPI budget routes.

Covers all endpoints in /api/budgets/ and all helper functions:
_safe_float, _sanitize_values, _get_column_units, _parse_title_units,
_detect_budget_category, _get_budget_units_metadata, plus all route
handlers with their branches: types, glossary, locations, columns,
data (monthly/non-monthly, column filtering, units errors),
summary, spatial (stat variants, column lookup, per-location errors),
location-geometry (subregion, stream_node, error), and water-balance
(Sankey node/link generation, matched/unmatched flows, tiny values).
"""

from __future__ import annotations

import math
from datetime import datetime
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi", reason="FastAPI not available")
pydantic = pytest.importorskip("pydantic", reason="Pydantic not available")

from fastapi.testclient import TestClient

from pyiwfm.visualization.webapi.config import ModelState, model_state
from pyiwfm.visualization.webapi.routes.budgets import (
    BUDGET_GLOSSARY,
    _DATA_TYPE_UNITS,
    _detect_budget_category,
    _get_budget_units_metadata,
    _get_column_units,
    _parse_title_units,
    _safe_float,
    _sanitize_values,
)
from pyiwfm.visualization.webapi.server import create_app


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
    model_state._area_manager = None
    # Restore any monkey-patched methods back to the class originals
    for attr in ("get_budget_reader", "get_available_budgets", "reproject_coords",
                 "get_stream_reach_boundaries", "get_head_loader", "get_gw_hydrograph_reader",
                 "get_stream_hydrograph_reader", "get_area_manager", "get_subsidence_reader"):
        if attr in model_state.__dict__:
            del model_state.__dict__[attr]


def _make_mock_reader(
    locations=None,
    headers=None,
    values=None,
    column_types=None,
    timestep_unit="1MON",
    start_datetime=None,
    delta_t_minutes=43200,
    ascii_output=None,
    n_location_data=1,
):
    """Create a mock BudgetReader with configurable properties."""
    reader = MagicMock()
    reader.locations = locations or ["Region 1", "Region 2"]
    _headers = headers or ["Deep Percolation", "Pumping", "Recharge"]

    reader.get_column_headers.return_value = _headers

    n_times = 12
    n_cols = len(_headers)
    if values is not None:
        times_arr, values_arr = values
    else:
        times_arr = np.arange(n_times, dtype=float)
        values_arr = np.random.rand(n_times, n_cols)

    reader.get_values.return_value = (times_arr, values_arr)
    reader.get_location_index.return_value = 0

    # Timestep info
    ts = MagicMock()
    ts.unit = timestep_unit
    ts.start_datetime = start_datetime or datetime(1990, 10, 1)
    ts.delta_t_minutes = delta_t_minutes
    reader.header = MagicMock()
    reader.header.timestep = ts

    # Location data with column types
    _col_types = column_types or [1] * n_cols  # Default: VR (volume rate)
    loc_data = MagicMock()
    loc_data.column_types = _col_types

    if n_location_data == 1:
        reader.header.location_data = [loc_data]
    else:
        reader.header.location_data = [
            MagicMock(column_types=_col_types) for _ in range(n_location_data)
        ]

    reader.header.ascii_output = ascii_output

    return reader


def _make_mock_model(metadata=None):
    """Create a minimal mock IWFMModel."""
    model = MagicMock()
    model.name = "TestModel"
    model.metadata = metadata or {}
    model.streams = None
    model.grid = None
    model.groundwater = None
    model.stratigraphy = None
    model.source_files = {}
    return model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_state():
    """Reset model state before and after every test."""
    _reset_model_state()
    yield
    _reset_model_state()


@pytest.fixture()
def client_no_model():
    """TestClient with no model loaded."""
    app = create_app()
    return TestClient(app)


@pytest.fixture()
def client_with_model():
    """TestClient with a model loaded and a budget reader available."""
    model = _make_mock_model()
    model_state._model = model
    reader = _make_mock_reader()
    model_state._budget_readers["gw"] = reader
    app = create_app()
    return TestClient(app), model, reader


# ===========================================================================
# Unit tests for helper functions
# ===========================================================================


class TestSafeFloat:
    """Tests for _safe_float()."""

    def test_none_returns_none(self):
        assert _safe_float(None) is None

    def test_nan_returns_none(self):
        assert _safe_float(float("nan")) is None

    def test_positive_inf_returns_none(self):
        assert _safe_float(float("inf")) is None

    def test_negative_inf_returns_none(self):
        assert _safe_float(float("-inf")) is None

    def test_normal_float_returns_value(self):
        assert _safe_float(3.14) == 3.14

    def test_zero_returns_zero(self):
        assert _safe_float(0.0) == 0.0

    def test_negative_returns_value(self):
        assert _safe_float(-42.5) == -42.5


class TestSanitizeValues:
    """Tests for _sanitize_values()."""

    def test_empty_list(self):
        assert _sanitize_values([]) == []

    def test_all_normal(self):
        result = _sanitize_values([1.0, 2.0, 3.0])
        assert result == [1.0, 2.0, 3.0]

    def test_nan_replaced(self):
        result = _sanitize_values([1.0, float("nan"), 3.0])
        assert result[0] == 1.0
        assert result[1] is None
        assert result[2] == 3.0

    def test_inf_replaced(self):
        result = _sanitize_values([float("inf"), 2.0, float("-inf")])
        assert result[0] is None
        assert result[1] == 2.0
        assert result[2] is None

    def test_mixed_types_preserved(self):
        """Non-float items (int, str) pass through unchanged."""
        result = _sanitize_values([1, "hello", 3.0, float("nan")])
        assert result == [1, "hello", 3.0, None]

    def test_none_in_list_preserved(self):
        """None values already in the list are kept as-is (not float)."""
        result = _sanitize_values([None, 1.0])
        assert result == [None, 1.0]


class TestGetColumnUnits:
    """Tests for _get_column_units()."""

    def test_single_location_data(self):
        """When header has exactly 1 location_data entry, always use it."""
        reader = _make_mock_reader(
            headers=["Col A", "Col B"],
            column_types=[1, 4],
            n_location_data=1,
        )
        units = _get_column_units(reader, 0)
        assert units == ["Volume/Time", "Area"]

    def test_multiple_location_data_uses_index(self):
        """When header has multiple location_data entries, index by loc_idx."""
        reader = MagicMock()
        reader.get_location_index.return_value = 1

        loc0 = MagicMock()
        loc0.column_types = [1, 1]
        loc1 = MagicMock()
        loc1.column_types = [4, 5]  # Area, Length
        reader.header.location_data = [loc0, loc1]

        units = _get_column_units(reader, 1)
        assert units == ["Area", "Length"]

    def test_unknown_type_code(self):
        """Unknown data type codes map to empty string."""
        reader = _make_mock_reader(
            headers=["Unknown"],
            column_types=[999],
        )
        units = _get_column_units(reader, 0)
        assert units == [""]


class TestParseTitleUnits:
    """Tests for _parse_title_units()."""

    def test_volume_area_length_in_titles(self):
        ascii_output = MagicMock()
        ascii_output.titles = [
            "UNIT OF VOLUME = TAF",
            "UNIT OF AREA = ACRES",
            "UNIT OF LENGTH = FEET",
        ]
        reader = MagicMock()
        reader.header.ascii_output = ascii_output

        result = _parse_title_units(reader)
        assert result["volume"] == "TAF"
        assert result["area"] == "ACRES"
        assert result["length"] == "FEET"

    def test_no_titles_attribute(self):
        """When ascii_output exists but has no titles attribute, return empty dict."""
        reader = MagicMock()
        reader.header.ascii_output = MagicMock(spec=[])  # no 'titles' attr
        result = _parse_title_units(reader)
        assert result == {}

    def test_ascii_output_is_none(self):
        """When ascii_output is None, return empty dict."""
        reader = MagicMock()
        reader.header.ascii_output = None
        result = _parse_title_units(reader)
        assert result == {}

    def test_attribute_error_handled(self):
        """When accessing header raises AttributeError, return empty dict."""
        reader = MagicMock()
        type(reader).header = PropertyMock(side_effect=AttributeError)
        result = _parse_title_units(reader)
        assert result == {}

    def test_title_without_equals_sign_skipped(self):
        """Lines mentioning UNIT OF VOLUME but without '=' are skipped."""
        ascii_output = MagicMock()
        ascii_output.titles = [
            "UNIT OF VOLUME TAF",  # No '='
            "UNIT OF AREA = ACRES",
        ]
        reader = MagicMock()
        reader.header.ascii_output = ascii_output
        result = _parse_title_units(reader)
        assert "volume" not in result
        assert result["area"] == "ACRES"

    def test_type_error_handled(self):
        """When iterating titles raises TypeError, return empty dict."""
        ascii_output = MagicMock()
        ascii_output.titles = None  # Iterating None raises TypeError
        # Need hasattr to return True for "titles"
        reader = MagicMock()
        reader.header.ascii_output = ascii_output
        # Force titles to be a non-iterable that passes hasattr check
        ascii_output_obj = MagicMock()
        ascii_output_obj.titles = 12345  # int is not iterable
        reader.header.ascii_output = ascii_output_obj
        result = _parse_title_units(reader)
        assert result == {}

    def test_mixed_case_titles(self):
        """Case-insensitive matching."""
        ascii_output = MagicMock()
        ascii_output.titles = [
            "Unit of Volume = TAF",
        ]
        reader = MagicMock()
        reader.header.ascii_output = ascii_output
        result = _parse_title_units(reader)
        assert result["volume"] == "TAF"


class TestDetectBudgetCategory:
    """Tests for _detect_budget_category()."""

    @pytest.mark.parametrize("btype,expected", [
        ("gw", "gw"),
        ("GW_Budget", "gw"),
        ("groundwater", "gw"),
        ("Groundwater Budget", "gw"),
        ("lwu", "lwu"),
        ("land_and_water_use", "lwu"),
        ("rootzone", "rootzone"),
        ("Root Zone Budget", "rootzone"),
        ("unsaturated", "unsaturated"),
        ("Unsat Zone Budget", "unsaturated"),
        ("stream_node", "stream_node"),
        ("Stream Node Budget", "stream_node"),
        ("stream_node_budget", "stream_node"),
        ("stream", "stream"),
        ("Stream Budget", "stream"),
        ("diversion", "diversion"),
        ("Diversion Budget", "diversion"),
        ("lake", "lake"),
        ("Lake Budget", "lake"),
        ("small_watershed", "small_watershed"),
        ("Small Watershed Budget", "small_watershed"),
        ("something_else", "other"),
        ("unknown_type", "other"),
    ])
    def test_category_detection(self, btype, expected):
        assert _detect_budget_category(btype) == expected

    def test_stream_node_before_stream(self):
        """stream_node must match before stream to avoid false positive."""
        assert _detect_budget_category("stream_node") == "stream_node"
        assert _detect_budget_category("stream") == "stream"


class TestGetBudgetUnitsMetadata:
    """Tests for _get_budget_units_metadata()."""

    def test_gw_category_with_metadata(self):
        """GW budget derives volume unit from length_unit (FT→FT3)."""
        model = _make_mock_model(metadata={
            "gw_volume_output_unit": "TAF",  # ignored — HDF stores sim units
            "gw_length_output_unit": "FT",   # ignored
            "length_unit": "FT",
            "area_unit": "SQ_MI",
        })
        model_state._model = model
        reader = _make_mock_reader(column_types=[1, 4, 5])

        result = _get_budget_units_metadata("gw", reader)
        # source_volume and source_area are derived from length_unit
        assert result["source_volume_unit"] == "FT3"
        assert result["source_length_unit"] == "FT"
        assert result["source_area_unit"] == "SQ.FT."
        assert result["source_area_output_unit"] == "SQ_MI"
        assert result["has_volume_columns"] is True
        assert result["has_area_columns"] is True
        assert result["has_length_columns"] is True

    def test_non_gw_category_uses_generic_metadata(self):
        """Non-GW budgets derive volume unit from length_unit (M→M3)."""
        model = _make_mock_model(metadata={
            "volume_unit": "CCF",  # ignored — HDF stores sim units
            "area_unit": "HA",
            "length_unit": "M",
        })
        model_state._model = model
        reader = _make_mock_reader(column_types=[1, 1, 1])

        result = _get_budget_units_metadata("stream", reader)
        # source_volume and source_area derived from length_unit "M" → "M3", "M2"
        assert result["source_volume_unit"] == "M3"
        assert result["source_area_unit"] == "M2"
        assert result["source_area_output_unit"] == "HA"
        assert result["source_length_unit"] == "M"

    def test_gw_volume_derived_from_length_unit(self):
        """GW budget derives volume from length_unit, ignoring volume_unit."""
        model = _make_mock_model(metadata={
            "volume_unit": "TAF",  # ignored
            "length_unit": "FT",
            "area_unit": "ACRES",
        })
        model_state._model = model
        reader = _make_mock_reader(column_types=[1])

        result = _get_budget_units_metadata("gw", reader)
        assert result["source_volume_unit"] == "FT3"
        assert result["source_length_unit"] == "FT"

    def test_title_lines_ignored_for_source_units(self):
        """Title parsing no longer affects source units (HDF stores sim units)."""
        model = _make_mock_model(metadata={})
        model_state._model = model

        ascii_output = MagicMock()
        ascii_output.titles = [
            "UNIT OF VOLUME = TAF",
            "UNIT OF AREA = HECTARES",
            "UNIT OF LENGTH = METERS",
        ]
        reader = _make_mock_reader(column_types=[1])
        reader.header.ascii_output = ascii_output

        result = _get_budget_units_metadata("gw", reader)
        # Title units are output units, not source units — they are ignored.
        # Default length_unit is "FT" → source_volume is "FT3"
        assert result["source_volume_unit"] == "FT3"
        assert result["source_area_unit"] == "SQ.FT."
        assert result["source_length_unit"] == "FT"

    def test_no_model_metadata(self):
        """When model is None, metadata defaults to empty dict → FT defaults."""
        model_state._model = None
        reader = _make_mock_reader(column_types=[1])
        result = _get_budget_units_metadata("gw", reader)
        # Default length_unit is "FT" → source_volume "FT3"
        assert result["source_volume_unit"] == "FT3"
        assert result["source_area_unit"] == "SQ.FT."
        assert result["source_length_unit"] == "FT"

    def test_column_type_scanning_volume_only(self):
        """Scanning column types: only volume codes present."""
        model = _make_mock_model()
        model_state._model = model
        reader = _make_mock_reader(column_types=[1, 2, 3])

        result = _get_budget_units_metadata("gw", reader)
        assert result["has_volume_columns"] is True
        assert result["has_area_columns"] is False
        assert result["has_length_columns"] is False

    def test_column_type_scanning_area_and_length(self):
        """Scanning column types: area and length codes present, no volume."""
        model = _make_mock_model()
        model_state._model = model
        reader = _make_mock_reader(column_types=[4, 5])

        result = _get_budget_units_metadata("stream", reader)
        assert result["has_volume_columns"] is False
        assert result["has_area_columns"] is True
        assert result["has_length_columns"] is True

    def test_column_type_scanning_unknown_codes(self):
        """Column types not in any known set leave all flags False."""
        model = _make_mock_model()
        model_state._model = model
        reader = _make_mock_reader(column_types=[99, 100])

        result = _get_budget_units_metadata("gw", reader)
        assert result["has_volume_columns"] is False
        assert result["has_area_columns"] is False
        assert result["has_length_columns"] is False

    def test_column_type_scanning_attribute_error(self):
        """When location_data raises AttributeError, default to has_volume=True."""
        model = _make_mock_model()
        model_state._model = model
        reader = _make_mock_reader(column_types=[1])
        # Make location_data access raise
        reader.header.location_data = MagicMock(
            __getitem__=MagicMock(side_effect=AttributeError)
        )

        result = _get_budget_units_metadata("gw", reader)
        assert result["has_volume_columns"] is True

    def test_column_type_scanning_index_error(self):
        """When location_data is empty, default to has_volume=True."""
        model = _make_mock_model()
        model_state._model = model
        reader = _make_mock_reader(column_types=[1])
        reader.header.location_data = []

        result = _get_budget_units_metadata("gw", reader)
        assert result["has_volume_columns"] is True

    def test_timestep_unit_from_reader(self):
        """Timestep unit comes from reader.header.timestep.unit."""
        model = _make_mock_model()
        model_state._model = model
        reader = _make_mock_reader(timestep_unit="1DAY", column_types=[1])

        result = _get_budget_units_metadata("gw", reader)
        assert result["timestep_unit"] == "1DAY"

    def test_timestep_unit_none_defaults_to_1mon(self):
        """When timestep.unit is None/falsy, default to '1MON'."""
        model = _make_mock_model()
        model_state._model = model
        reader = _make_mock_reader(timestep_unit=None, column_types=[1])
        reader.header.timestep.unit = None

        result = _get_budget_units_metadata("gw", reader)
        assert result["timestep_unit"] == "1MON"

    def test_empty_length_unit_defaults_to_ft3(self):
        """When length_unit is missing, defaults to FT → FT3."""
        model = _make_mock_model(metadata={"volume_unit": ""})
        model_state._model = model
        reader = _make_mock_reader(column_types=[1])
        reader.header.ascii_output = None

        result = _get_budget_units_metadata("stream", reader)
        # volume_unit is ignored; length_unit defaults to "FT" → "FT3"
        assert result["source_volume_unit"] == "FT3"

    def test_empty_column_types_infers_from_headers(self):
        """When column_types is empty, infer volume/area/length from headers."""
        model = _make_mock_model()
        model_state._model = model
        reader = _make_mock_reader(
            headers=["Deep Percolation", "Pumping", "AG_AREA", "CUM_SUBSIDENCE"],
        )
        # Force empty column_types (bypasses `or` default in helper)
        loc = reader.header.location_data[0]
        loc.column_types = []
        loc.column_headers = [
            "Deep Percolation", "Pumping", "AG_AREA", "CUM_SUBSIDENCE",
        ]

        result = _get_budget_units_metadata("gw", reader)
        assert result["has_volume_columns"] is True   # Deep Percolation, Pumping, CUM_SUBSIDENCE
        assert result["has_area_columns"] is True      # AG_AREA
        assert result["has_length_columns"] is False   # subsidence is volumetric in GW budgets

    def test_empty_column_types_volume_only_headers(self):
        """When column_types empty + no area/length keywords → volume only."""
        model = _make_mock_model()
        model_state._model = model
        reader = _make_mock_reader(
            headers=["Pumping", "Recharge", "Net Deep Percolation"],
        )
        loc = reader.header.location_data[0]
        loc.column_types = []
        loc.column_headers = [
            "Pumping", "Recharge", "Net Deep Percolation",
        ]

        result = _get_budget_units_metadata("gw", reader)
        assert result["has_volume_columns"] is True
        assert result["has_area_columns"] is False
        assert result["has_length_columns"] is False

    def test_empty_column_types_no_headers_defaults_volume(self):
        """When both column_types and column_headers are empty → assume volume."""
        model = _make_mock_model()
        model_state._model = model
        reader = _make_mock_reader()
        loc = reader.header.location_data[0]
        loc.column_types = []
        loc.column_headers = []

        result = _get_budget_units_metadata("gw", reader)
        assert result["has_volume_columns"] is True
        assert result["has_area_columns"] is False
        assert result["has_length_columns"] is False


# ===========================================================================
# Endpoint tests
# ===========================================================================


class TestGetBudgetTypes:
    """Tests for GET /api/budgets/types."""

    def test_no_model_returns_404(self, client_no_model):
        resp = client_no_model.get("/api/budgets/types")
        assert resp.status_code == 404
        assert "No model loaded" in resp.json()["detail"]

    def test_returns_available_types(self):
        model = _make_mock_model()
        model_state._model = model
        # Patch get_available_budgets to return known types
        model_state.get_available_budgets = MagicMock(
            return_value=["gw", "stream"]
        )
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/types")
        assert resp.status_code == 200
        assert resp.json() == ["gw", "stream"]


class TestGetBudgetGlossary:
    """Tests for GET /api/budgets/glossary."""

    def test_returns_glossary_dict(self, client_no_model):
        resp = client_no_model.get("/api/budgets/glossary")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)
        assert "gw" in data
        assert "stream" in data
        assert "rootzone" in data
        assert "lwu" in data
        assert "diversion" in data
        assert "lake" in data
        assert "unsaturated" in data
        assert "stream_node" in data
        assert "small_watershed" in data

    def test_glossary_has_descriptions(self, client_no_model):
        resp = client_no_model.get("/api/budgets/glossary")
        data = resp.json()
        assert "Deep Percolation" in data["gw"]
        assert isinstance(data["gw"]["Deep Percolation"], str)


class TestGetBudgetLocations:
    """Tests for GET /api/budgets/{budget_type}/locations."""

    def test_no_reader_returns_404(self):
        model = _make_mock_model()
        model_state._model = model
        # No budget reader registered for "gw"
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/gw/locations")
        assert resp.status_code == 404
        assert "not available" in resp.json()["detail"]

    def test_returns_locations(self, client_with_model):
        client, model, reader = client_with_model

        resp = client.get("/api/budgets/gw/locations")
        assert resp.status_code == 200
        data = resp.json()
        assert "locations" in data
        assert len(data["locations"]) == 2
        assert data["locations"][0] == {"id": 0, "name": "Region 1"}
        assert data["locations"][1] == {"id": 1, "name": "Region 2"}


class TestGetBudgetColumns:
    """Tests for GET /api/budgets/{budget_type}/columns."""

    def test_no_reader_returns_404(self):
        model = _make_mock_model()
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/gw/columns")
        assert resp.status_code == 404

    def test_error_in_get_column_headers_returns_404(self):
        model = _make_mock_model()
        model_state._model = model
        reader = _make_mock_reader()
        reader.get_column_headers.side_effect = KeyError("bad location")
        model_state._budget_readers["gw"] = reader
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/gw/columns?location=bad")
        assert resp.status_code == 404
        assert "bad location" in resp.json()["detail"]

    def test_returns_columns_with_units(self, client_with_model):
        client, model, reader = client_with_model

        resp = client.get("/api/budgets/gw/columns")
        assert resp.status_code == 200
        data = resp.json()
        assert "columns" in data
        assert len(data["columns"]) == 3
        assert data["columns"][0]["name"] == "Deep Percolation"
        assert data["columns"][0]["units"] == "Volume/Time"

    def test_with_location_param(self, client_with_model):
        client, model, reader = client_with_model

        resp = client.get("/api/budgets/gw/columns?location=Region+1")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["columns"]) == 3

    def test_columns_units_shorter_than_headers(self):
        """When units list is shorter than headers, use empty string."""
        model = _make_mock_model()
        model_state._model = model
        reader = _make_mock_reader(
            headers=["A", "B", "C", "D"],
            column_types=[1, 4],  # Only 2 types for 4 headers
        )
        model_state._budget_readers["gw"] = reader
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/gw/columns")
        assert resp.status_code == 200
        data = resp.json()
        # Columns 2 and 3 should have empty units since units list is shorter
        assert data["columns"][2]["units"] == ""
        assert data["columns"][3]["units"] == ""


class TestGetBudgetData:
    """Tests for GET /api/budgets/{budget_type}/data."""

    def test_no_reader_returns_404(self):
        model = _make_mock_model()
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/gw/data")
        assert resp.status_code == 404

    def test_error_in_get_values_returns_400(self):
        model = _make_mock_model()
        model_state._model = model
        reader = _make_mock_reader()
        reader.get_values.side_effect = ValueError("bad data")
        model_state._budget_readers["gw"] = reader
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/gw/data")
        assert resp.status_code == 400
        assert "bad data" in resp.json()["detail"]

    def test_columns_all(self, client_with_model):
        """columns=all returns all columns."""
        client, model, reader = client_with_model

        resp = client.get("/api/budgets/gw/data?columns=all")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["columns"]) == 3
        assert data["columns"][0]["name"] == "Deep Percolation"
        assert "times" in data
        assert len(data["times"]) == 12

    def test_columns_specific(self, client_with_model):
        """columns=0,2 returns only those columns."""
        client, model, reader = client_with_model

        resp = client.get("/api/budgets/gw/data?columns=0,2")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["columns"]) == 2
        assert data["columns"][0]["name"] == "Deep Percolation"
        assert data["columns"][1]["name"] == "Recharge"

    def test_monthly_timestamps(self, client_with_model):
        """Monthly timesteps use relativedelta."""
        client, model, reader = client_with_model

        resp = client.get("/api/budgets/gw/data")
        assert resp.status_code == 200
        data = resp.json()
        times = data["times"]
        assert len(times) == 12
        # First time should be the start_datetime
        assert times[0] == "1990-10-01T00:00:00"

    def test_non_monthly_timestamps(self):
        """Non-monthly timesteps use timedelta."""
        model = _make_mock_model()
        model_state._model = model
        reader = _make_mock_reader(
            timestep_unit="1DAY",
            delta_t_minutes=1440,
            start_datetime=datetime(2000, 1, 1),
        )
        model_state._budget_readers["gw"] = reader
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/gw/data")
        assert resp.status_code == 200
        data = resp.json()
        times = data["times"]
        assert times[0] == "2000-01-01T00:00:00"
        assert times[1] == "2000-01-02T00:00:00"

    def test_no_start_datetime(self):
        """When start_datetime is None, times come from raw array."""
        model = _make_mock_model()
        model_state._model = model
        reader = _make_mock_reader(start_datetime=None)
        reader.header.timestep.start_datetime = None
        model_state._budget_readers["gw"] = reader
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/gw/data")
        assert resp.status_code == 200
        data = resp.json()
        times = data["times"]
        # Times should be string representations of the raw array values
        assert len(times) == 12
        assert times[0] == "0.0"

    def test_units_error_in_data_endpoint(self):
        """When _get_column_units raises, col_units defaults to empty list."""
        model = _make_mock_model()
        model_state._model = model
        reader = _make_mock_reader()
        # Make get_location_index raise on second call (within _get_column_units)
        call_count = [0]
        original_get_location_index = reader.get_location_index

        def side_effect_get_loc(loc):
            call_count[0] += 1
            if call_count[0] > 1:
                raise KeyError("units error")
            return original_get_location_index(loc)

        reader.get_location_index.side_effect = side_effect_get_loc
        model_state._budget_readers["gw"] = reader
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/gw/data")
        assert resp.status_code == 200
        data = resp.json()
        # Should still succeed with empty units
        for col in data["columns"]:
            assert col["units"] == ""

    def test_data_has_units_metadata(self, client_with_model):
        """Response includes units_metadata dict."""
        client, model, reader = client_with_model

        resp = client.get("/api/budgets/gw/data")
        assert resp.status_code == 200
        data = resp.json()
        assert "units_metadata" in data
        meta = data["units_metadata"]
        assert "source_volume_unit" in meta
        assert "timestep_unit" in meta

    def test_data_location_default(self, client_with_model):
        """When location is empty, uses first location name in response."""
        client, model, reader = client_with_model

        resp = client.get("/api/budgets/gw/data")
        assert resp.status_code == 200
        data = resp.json()
        assert data["location"] == "Region 1"

    def test_data_with_location_param(self, client_with_model):
        """When location param is provided, it appears in response."""
        client, model, reader = client_with_model

        resp = client.get("/api/budgets/gw/data?location=Region+2")
        assert resp.status_code == 200
        data = resp.json()
        assert data["location"] == "Region 2"

    def test_data_sanitizes_nan_values(self):
        """NaN values in budget data are replaced with None."""
        model = _make_mock_model()
        model_state._model = model
        times = np.arange(3, dtype=float)
        vals = np.array([[1.0, 2.0], [float("nan"), 3.0], [4.0, float("inf")]])
        reader = _make_mock_reader(
            headers=["A", "B"],
            values=(times, vals),
        )
        model_state._budget_readers["gw"] = reader
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/gw/data")
        assert resp.status_code == 200
        data = resp.json()
        col_a = data["columns"][0]["values"]
        col_b = data["columns"][1]["values"]
        assert col_a[1] is None  # NaN replaced
        assert col_b[2] is None  # Inf replaced

    def test_data_columns_filter_units_shorter(self):
        """When col_indices select columns beyond units list length, empty string used."""
        model = _make_mock_model()
        model_state._model = model
        times = np.arange(3, dtype=float)
        vals = np.random.rand(3, 4)
        reader = _make_mock_reader(
            headers=["A", "B", "C", "D"],
            values=(times, vals),
            column_types=[1],  # Only 1 column type for 4 columns
        )
        model_state._budget_readers["gw"] = reader
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/gw/data?columns=0,3")
        assert resp.status_code == 200
        data = resp.json()
        # Column index 3 should have empty units since units list has only 1 entry
        assert data["columns"][1]["units"] == ""

    def test_data_key_error_returns_400(self):
        """KeyError from get_values returns 400."""
        model = _make_mock_model()
        model_state._model = model
        reader = _make_mock_reader()
        reader.get_values.side_effect = KeyError("no such location")
        model_state._budget_readers["gw"] = reader
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/gw/data?location=nonexistent")
        assert resp.status_code == 400

    def test_data_index_error_returns_400(self):
        """IndexError from get_values returns 400."""
        model = _make_mock_model()
        model_state._model = model
        reader = _make_mock_reader()
        reader.get_values.side_effect = IndexError("out of bounds")
        model_state._budget_readers["gw"] = reader
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/gw/data")
        assert resp.status_code == 400


class TestGetBudgetSummary:
    """Tests for GET /api/budgets/{budget_type}/summary."""

    def test_no_reader_returns_404(self):
        model = _make_mock_model()
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/gw/summary")
        assert resp.status_code == 404

    def test_returns_summary(self, client_with_model):
        client, model, reader = client_with_model

        resp = client.get("/api/budgets/gw/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["location"] == "Region 1"
        assert data["n_timesteps"] == 12
        assert "totals" in data
        assert "averages" in data
        assert "Deep Percolation" in data["totals"]
        assert "Pumping" in data["averages"]

    def test_summary_with_location(self, client_with_model):
        client, model, reader = client_with_model

        resp = client.get("/api/budgets/gw/summary?location=Region+2")
        assert resp.status_code == 200
        data = resp.json()
        assert data["location"] == "Region 2"

    def test_summary_error_returns_400(self):
        model = _make_mock_model()
        model_state._model = model
        reader = _make_mock_reader()
        reader.get_values.side_effect = KeyError("bad")
        model_state._budget_readers["gw"] = reader
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/gw/summary")
        assert resp.status_code == 400


class TestGetBudgetSpatial:
    """Tests for GET /api/budgets/{budget_type}/spatial."""

    def test_no_reader_returns_404(self):
        model = _make_mock_model()
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/gw/spatial")
        assert resp.status_code == 404

    def test_column_not_found_returns_400(self, client_with_model):
        client, model, reader = client_with_model

        resp = client.get("/api/budgets/gw/spatial?column=NonExistent")
        assert resp.status_code == 400
        assert "not found" in resp.json()["detail"]

    def test_default_column(self, client_with_model):
        """When no column param, uses first column (index 0)."""
        client, model, reader = client_with_model

        resp = client.get("/api/budgets/gw/spatial")
        assert resp.status_code == 200
        data = resp.json()
        assert data["column"] == "Deep Percolation"

    def test_stat_total(self, client_with_model):
        client, model, reader = client_with_model

        resp = client.get(
            "/api/budgets/gw/spatial?column=Deep+Percolation&stat=total"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["stat"] == "total"
        assert len(data["locations"]) == 2

    def test_stat_average(self, client_with_model):
        client, model, reader = client_with_model

        resp = client.get(
            "/api/budgets/gw/spatial?column=Deep+Percolation&stat=average"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["stat"] == "average"

    def test_stat_last(self):
        """stat=last uses the final timestep value."""
        model = _make_mock_model()
        model_state._model = model
        times = np.arange(5, dtype=float)
        vals = np.array([[1.0], [2.0], [3.0], [4.0], [99.0]])
        reader = _make_mock_reader(
            locations=["Loc1"],
            headers=["Col1"],
            values=(times, vals),
        )
        model_state._budget_readers["gw"] = reader
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/gw/spatial?stat=last")
        assert resp.status_code == 200
        data = resp.json()
        assert data["locations"][0]["value"] == 99.0

    def test_stat_unknown_defaults_to_total(self, client_with_model):
        """Unknown stat value falls through to nansum (same as total)."""
        client, model, reader = client_with_model

        resp = client.get(
            "/api/budgets/gw/spatial?column=Deep+Percolation&stat=xyz"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["stat"] == "xyz"

    def test_error_in_get_values_for_location(self):
        """When get_values fails for a location, value defaults to 0.0."""
        model = _make_mock_model()
        model_state._model = model

        reader = MagicMock()
        reader.locations = ["Good Loc", "Bad Loc"]
        reader.get_column_headers.return_value = ["Col1"]

        call_count = [0]

        def side_effect_get_values(loc_i, col_indices=None):
            call_count[0] += 1
            if loc_i == 1:
                raise IndexError("bad location data")
            times = np.arange(5, dtype=float)
            vals = np.array([[10.0], [20.0], [30.0], [40.0], [50.0]])
            return times, vals

        reader.get_values.side_effect = side_effect_get_values
        model_state._budget_readers["gw"] = reader
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/gw/spatial")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["locations"]) == 2
        assert data["locations"][0]["value"] == 150.0  # sum of 10+20+30+40+50
        assert data["locations"][1]["value"] == 0.0  # error fallback

    def test_spatial_returns_min_max(self, client_with_model):
        """Response includes min and max across all locations."""
        client, model, reader = client_with_model

        resp = client.get("/api/budgets/gw/spatial")
        assert resp.status_code == 200
        data = resp.json()
        assert "min" in data
        assert "max" in data
        assert data["min"] <= data["max"]

    def test_spatial_available_columns(self, client_with_model):
        """Response includes available_columns list."""
        client, model, reader = client_with_model

        resp = client.get("/api/budgets/gw/spatial")
        assert resp.status_code == 200
        data = resp.json()
        assert "available_columns" in data
        assert data["available_columns"] == [
            "Deep Percolation", "Pumping", "Recharge"
        ]

    def test_spatial_column_case_insensitive(self, client_with_model):
        """Column matching is case-insensitive."""
        client, model, reader = client_with_model

        resp = client.get("/api/budgets/gw/spatial?column=deep+percolation")
        assert resp.status_code == 200

    def test_spatial_nan_value_becomes_zero(self):
        """When computed value is NaN, _safe_float returns None, replaced by 0.0."""
        model = _make_mock_model()
        model_state._model = model
        times = np.arange(3, dtype=float)
        vals = np.array([[float("nan")], [float("nan")], [float("nan")]])
        reader = _make_mock_reader(
            locations=["NanLoc"],
            headers=["Col1"],
            values=(times, vals),
        )
        model_state._budget_readers["gw"] = reader
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/gw/spatial")
        assert resp.status_code == 200
        data = resp.json()
        assert data["locations"][0]["value"] == 0.0


class TestGetBudgetLocationGeometry:
    """Tests for GET /api/budgets/{budget_type}/location-geometry."""

    def test_no_reader_returns_404(self):
        model = _make_mock_model()
        model_state._model = model
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/gw/location-geometry")
        assert resp.status_code == 404

    def test_subregion_category(self, client_with_model):
        """GW budget -> subregion spatial type."""
        client, model, reader = client_with_model

        resp = client.get("/api/budgets/gw/location-geometry")
        assert resp.status_code == 200
        data = resp.json()
        assert data["spatial_type"] == "subregion"
        assert data["location_index"] == 0
        assert data["geometry"] is None

    def test_stream_category_reach(self):
        """Stream budget -> reach spatial type."""
        model = _make_mock_model()
        model_state._model = model
        reader = _make_mock_reader()
        model_state._budget_readers["stream"] = reader
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/stream/location-geometry")
        assert resp.status_code == 200
        data = resp.json()
        assert data["spatial_type"] == "reach"

    def test_lake_category(self):
        """Lake budget -> lake spatial type."""
        model = _make_mock_model()
        model_state._model = model
        reader = _make_mock_reader()
        model_state._budget_readers["lake"] = reader
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/lake/location-geometry")
        assert resp.status_code == 200
        data = resp.json()
        assert data["spatial_type"] == "lake"

    def test_diversion_category(self):
        """Diversion budget -> diversion spatial type."""
        model = _make_mock_model()
        model_state._model = model
        reader = _make_mock_reader()
        model_state._budget_readers["diversion"] = reader
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/diversion/location-geometry")
        assert resp.status_code == 200
        data = resp.json()
        assert data["spatial_type"] == "diversion"

    def test_rootzone_subregion(self):
        """Root zone budget -> subregion spatial type."""
        model = _make_mock_model()
        model_state._model = model
        reader = _make_mock_reader()
        model_state._budget_readers["rootzone"] = reader
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/rootzone/location-geometry")
        assert resp.status_code == 200
        data = resp.json()
        assert data["spatial_type"] == "subregion"

    def test_unsaturated_subregion(self):
        """Unsaturated zone budget -> subregion spatial type."""
        model = _make_mock_model()
        model_state._model = model
        reader = _make_mock_reader()
        model_state._budget_readers["unsaturated"] = reader
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/unsaturated/location-geometry")
        assert resp.status_code == 200
        data = resp.json()
        assert data["spatial_type"] == "subregion"

    def test_lwu_subregion(self):
        """LWU budget -> subregion spatial type."""
        model = _make_mock_model()
        model_state._model = model
        reader = _make_mock_reader()
        model_state._budget_readers["lwu"] = reader
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/lwu/location-geometry")
        assert resp.status_code == 200
        data = resp.json()
        assert data["spatial_type"] == "subregion"

    def test_small_watershed_category(self):
        """Small watershed budget -> small_watershed spatial type."""
        model = _make_mock_model()
        model_state._model = model
        reader = _make_mock_reader()
        model_state._budget_readers["small_watershed"] = reader
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/small_watershed/location-geometry")
        assert resp.status_code == 200
        data = resp.json()
        assert data["spatial_type"] == "small_watershed"

    def test_stream_node_with_geometry(self):
        """Stream node budget returns point geometry from node lookup chain."""
        model = _make_mock_model()
        model_state._model = model

        # Set up streams with budget node IDs and GW node lookup
        stream = MagicMock()
        stream.budget_node_ids = [101, 102]
        snode = MagicMock()
        snode.gw_node = 5
        stream.nodes = {101: snode}
        model.streams = stream

        # Set up grid with the GW node
        gw_node = MagicMock()
        gw_node.x = 500000.0
        gw_node.y = 4000000.0
        grid = MagicMock()
        grid.nodes = {5: gw_node}
        model.grid = grid

        # Mock reproject_coords
        model_state.reproject_coords = lambda x, y: (-121.5, 37.5)

        reader = _make_mock_reader()
        model_state._budget_readers["stream_node"] = reader
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/stream_node/location-geometry")
        assert resp.status_code == 200
        data = resp.json()
        assert data["spatial_type"] == "point"
        assert data["geometry"] is not None
        assert data["geometry"]["type"] == "Point"
        assert data["geometry"]["coordinates"] == [-121.5, 37.5]

    def test_stream_node_no_streams(self):
        """Stream node budget but model.streams is None -> no geometry."""
        model = _make_mock_model()
        model.streams = None
        model_state._model = model
        reader = _make_mock_reader()
        model_state._budget_readers["stream_node"] = reader
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/stream_node/location-geometry")
        assert resp.status_code == 200
        data = resp.json()
        assert data["geometry"] is None

    def test_stream_node_no_budget_node_ids(self):
        """Stream node budget but no budget_node_ids attribute -> no geometry."""
        model = _make_mock_model()
        stream = MagicMock()
        stream.budget_node_ids = None
        model.streams = stream
        model_state._model = model
        reader = _make_mock_reader()
        model_state._budget_readers["stream_node"] = reader
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/stream_node/location-geometry")
        assert resp.status_code == 200
        data = resp.json()
        assert data["geometry"] is None

    def test_stream_node_loc_idx_out_of_range(self):
        """When loc_idx >= len(budget_node_ids), no geometry."""
        model = _make_mock_model()
        stream = MagicMock()
        stream.budget_node_ids = [101]  # Only 1 entry
        model.streams = stream
        model_state._model = model
        reader = _make_mock_reader()
        # Make get_location_index return 5 (out of range of budget_node_ids)
        reader.get_location_index.return_value = 5
        model_state._budget_readers["stream_node"] = reader
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/stream_node/location-geometry")
        assert resp.status_code == 200
        data = resp.json()
        assert data["geometry"] is None

    def test_stream_node_snode_not_found(self):
        """When stream.nodes.get(snode_id) returns None, no geometry."""
        model = _make_mock_model()
        stream = MagicMock()
        stream.budget_node_ids = [101]
        stream.nodes = {999: MagicMock()}  # Node 101 not in dict
        model.streams = stream
        model_state._model = model
        reader = _make_mock_reader()
        model_state._budget_readers["stream_node"] = reader
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/stream_node/location-geometry")
        assert resp.status_code == 200
        data = resp.json()
        assert data["geometry"] is None

    def test_stream_node_no_gw_node(self):
        """When snode.gw_node is None, no geometry."""
        model = _make_mock_model()
        stream = MagicMock()
        stream.budget_node_ids = [101]
        snode = MagicMock()
        snode.gw_node = None
        stream.nodes = {101: snode}
        model.streams = stream
        model_state._model = model
        reader = _make_mock_reader()
        model_state._budget_readers["stream_node"] = reader
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/stream_node/location-geometry")
        assert resp.status_code == 200
        data = resp.json()
        assert data["geometry"] is None

    def test_stream_node_no_grid(self):
        """When model.grid is None, no geometry (even with valid snode/gw_node)."""
        model = _make_mock_model()
        stream = MagicMock()
        stream.budget_node_ids = [101]
        snode = MagicMock()
        snode.gw_node = 5
        stream.nodes = {101: snode}
        model.streams = stream
        model.grid = None
        model_state._model = model
        reader = _make_mock_reader()
        model_state._budget_readers["stream_node"] = reader
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/stream_node/location-geometry")
        assert resp.status_code == 200
        data = resp.json()
        assert data["geometry"] is None

    def test_stream_node_gw_node_not_in_grid(self):
        """When gw_node id not in grid.nodes, no geometry."""
        model = _make_mock_model()
        stream = MagicMock()
        stream.budget_node_ids = [101]
        snode = MagicMock()
        snode.gw_node = 5
        stream.nodes = {101: snode}
        model.streams = stream
        grid = MagicMock()
        grid.nodes = {10: MagicMock()}  # Node 5 not present
        model.grid = grid
        model_state._model = model
        reader = _make_mock_reader()
        model_state._budget_readers["stream_node"] = reader
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/stream_node/location-geometry")
        assert resp.status_code == 200
        data = resp.json()
        assert data["geometry"] is None

    def test_error_in_get_location_index(self):
        """When get_location_index raises, loc_idx defaults to 0."""
        model = _make_mock_model()
        model_state._model = model
        reader = _make_mock_reader()
        reader.get_location_index.side_effect = KeyError("bad loc")
        model_state._budget_readers["gw"] = reader
        app = create_app()
        client = TestClient(app)

        resp = client.get(
            "/api/budgets/gw/location-geometry?location=bad_location"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["location_index"] == 0

    def test_unknown_category_spatial_type(self):
        """Unknown budget category maps to 'unknown' spatial type."""
        model = _make_mock_model()
        model_state._model = model
        reader = _make_mock_reader()
        # Register under a key that _detect_budget_category won't recognize
        model_state._budget_readers["xyz_custom"] = reader
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/xyz_custom/location-geometry")
        assert resp.status_code == 200
        data = resp.json()
        assert data["spatial_type"] == "unknown"


class TestGetWaterBalance:
    """Tests for GET /api/budgets/water-balance."""

    def test_no_model_returns_404(self, client_no_model):
        resp = client_no_model.get("/api/budgets/water-balance")
        assert resp.status_code == 404
        assert "No model loaded" in resp.json()["detail"]

    def test_no_budgets_returns_404(self):
        """Model loaded but no budgets available."""
        model = _make_mock_model()
        model_state._model = model
        model_state.get_available_budgets = MagicMock(return_value=[])
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/water-balance")
        assert resp.status_code == 404
        assert "No budget data" in resp.json()["detail"]

    def test_with_gw_budget_matched_flows(self):
        """GW budget with matching flow mappings creates Sankey links."""
        model = _make_mock_model()
        model_state._model = model

        times = np.arange(3, dtype=float)
        vals = np.array([
            [100.0, 50.0, 30.0],
            [110.0, 55.0, 35.0],
            [120.0, 60.0, 40.0],
        ])
        reader = _make_mock_reader(
            headers=["Deep Percolation", "Pumping", "Recharge"],
            values=(times, vals),
        )
        model_state._budget_readers["gw"] = reader
        model_state.get_available_budgets = MagicMock(return_value=["gw"])
        model_state.get_budget_reader = MagicMock(return_value=reader)
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/water-balance")
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data
        assert "links" in data
        assert len(data["links"]) >= 3

        # Check that specific flow mappings were matched
        link_labels = [link["label"] for link in data["links"]]
        assert "Deep Percolation" in link_labels
        assert "Pumping" in link_labels
        assert "Recharge" in link_labels

    def test_with_stream_budget_matched_flows(self):
        """Stream budget matched flows create correct Sankey links."""
        model = _make_mock_model()
        model_state._model = model

        times = np.arange(3, dtype=float)
        vals = np.array([
            [200.0, 100.0],
            [210.0, 110.0],
            [220.0, 120.0],
        ])
        reader = _make_mock_reader(
            headers=["Upstream Inflow", "Downstream Outflow"],
            values=(times, vals),
        )
        model_state._budget_readers["stream"] = reader
        model_state.get_available_budgets = MagicMock(return_value=["stream"])
        model_state.get_budget_reader = MagicMock(return_value=reader)
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/water-balance")
        assert resp.status_code == 200
        data = resp.json()
        link_labels = [link["label"] for link in data["links"]]
        assert "Upstream Inflow" in link_labels
        assert "Downstream Outflow" in link_labels

    def test_with_rootzone_budget(self):
        """Root zone budget matched flows."""
        model = _make_mock_model()
        model_state._model = model

        times = np.arange(3, dtype=float)
        vals = np.array([[50.0, 30.0], [55.0, 35.0], [60.0, 40.0]])
        reader = _make_mock_reader(
            headers=["Precipitation", "ET"],
            values=(times, vals),
        )
        model_state._budget_readers["rootzone"] = reader
        model_state.get_available_budgets = MagicMock(
            return_value=["rootzone"]
        )
        model_state.get_budget_reader = MagicMock(return_value=reader)
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/water-balance")
        assert resp.status_code == 200
        data = resp.json()
        link_labels = [link["label"] for link in data["links"]]
        assert "Precipitation" in link_labels
        assert "ET" in link_labels

    def test_unmatched_flow_inflow(self):
        """Unmatched column with 'inflow' -> source is column name, target is component."""
        model = _make_mock_model()
        model_state._model = model

        times = np.arange(3, dtype=float)
        vals = np.array([[100.0], [110.0], [120.0]])
        reader = _make_mock_reader(
            headers=["Some Random Inflow"],
            values=(times, vals),
        )
        model_state._budget_readers["gw"] = reader
        model_state.get_available_budgets = MagicMock(return_value=["gw"])
        model_state.get_budget_reader = MagicMock(return_value=reader)
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/water-balance")
        assert resp.status_code == 200
        data = resp.json()
        # Should create inflow link: source = "Some Random Inflow", target = "Groundwater"
        inflow_links = [
            l for l in data["links"]
            if l["label"] == "Some Random Inflow"
        ]
        assert len(inflow_links) == 1
        # The target should be the component node
        target_idx = inflow_links[0]["target"]
        assert data["nodes"][target_idx] == "Groundwater"

    def test_unmatched_flow_outflow(self):
        """Unmatched column without 'inflow' or ending 'in' -> source is component."""
        model = _make_mock_model()
        model_state._model = model

        times = np.arange(3, dtype=float)
        vals = np.array([[100.0], [110.0], [120.0]])
        reader = _make_mock_reader(
            headers=["Tile Drain Outflow"],
            values=(times, vals),
        )
        model_state._budget_readers["gw"] = reader
        model_state.get_available_budgets = MagicMock(return_value=["gw"])
        model_state.get_budget_reader = MagicMock(return_value=reader)
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/water-balance")
        assert resp.status_code == 200
        data = resp.json()
        outflow_links = [
            l for l in data["links"]
            if l["label"] == "Tile Drain Outflow"
        ]
        assert len(outflow_links) == 1
        source_idx = outflow_links[0]["source"]
        assert data["nodes"][source_idx] == "Groundwater"

    def test_unmatched_flow_ending_in_in(self):
        """Unmatched column ending with 'in' (last 2 chars) -> treated as inflow."""
        model = _make_mock_model()
        model_state._model = model

        times = np.arange(3, dtype=float)
        vals = np.array([[100.0], [110.0], [120.0]])
        reader = _make_mock_reader(
            headers=["Subsurface Drain"],  # ends in 'in'
            values=(times, vals),
        )
        model_state._budget_readers["gw"] = reader
        model_state.get_available_budgets = MagicMock(return_value=["gw"])
        model_state.get_budget_reader = MagicMock(return_value=reader)
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/water-balance")
        assert resp.status_code == 200
        data = resp.json()
        links = [l for l in data["links"] if l["label"] == "Subsurface Drain"]
        assert len(links) == 1
        # Ends in "in" -> treated as inflow -> target is component
        target_idx = links[0]["target"]
        assert data["nodes"][target_idx] == "Groundwater"

    def test_tiny_values_skipped(self):
        """Values < 1e-6 are skipped in water balance."""
        model = _make_mock_model()
        model_state._model = model

        times = np.arange(3, dtype=float)
        vals = np.array([[1e-7], [1e-8], [1e-9]])  # All tiny
        reader = _make_mock_reader(
            headers=["Deep Percolation"],
            values=(times, vals),
        )
        model_state._budget_readers["gw"] = reader
        model_state.get_available_budgets = MagicMock(return_value=["gw"])
        model_state.get_budget_reader = MagicMock(return_value=reader)
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/water-balance")
        assert resp.status_code == 200
        data = resp.json()
        # Tiny value should be skipped
        assert len(data["links"]) == 0

    def test_reader_error_skips_budget(self):
        """When a reader raises an error, that budget type is skipped."""
        model = _make_mock_model()
        model_state._model = model

        good_reader = _make_mock_reader(
            headers=["Recharge"],
            values=(np.arange(3, dtype=float), np.array([[10.0], [20.0], [30.0]])),
        )
        bad_reader = MagicMock()
        bad_reader.get_column_headers.side_effect = KeyError("corrupted")

        call_map = {"gw": good_reader, "stream": bad_reader}
        model_state.get_available_budgets = MagicMock(
            return_value=["gw", "stream"]
        )
        model_state.get_budget_reader = MagicMock(
            side_effect=lambda btype: call_map.get(btype)
        )
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/water-balance")
        assert resp.status_code == 200
        data = resp.json()
        # Only GW links should be present, stream skipped due to error
        assert len(data["links"]) >= 1
        link_labels = [l["label"] for l in data["links"]]
        assert "Recharge" in link_labels

    def test_reader_returns_none_skipped(self):
        """When get_budget_reader returns None, budget is skipped."""
        model = _make_mock_model()
        model_state._model = model

        model_state.get_available_budgets = MagicMock(return_value=["gw"])
        model_state.get_budget_reader = MagicMock(return_value=None)
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/water-balance")
        assert resp.status_code == 200
        data = resp.json()
        assert data["nodes"] == []
        assert data["links"] == []

    def test_nan_value_in_water_balance(self):
        """NaN values in budget data become 0.0 in water balance."""
        model = _make_mock_model()
        model_state._model = model

        times = np.arange(3, dtype=float)
        vals = np.array([[float("nan")], [float("nan")], [float("nan")]])
        reader = _make_mock_reader(
            headers=["Deep Percolation"],
            values=(times, vals),
        )
        model_state._budget_readers["gw"] = reader
        model_state.get_available_budgets = MagicMock(return_value=["gw"])
        model_state.get_budget_reader = MagicMock(return_value=reader)
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/water-balance")
        assert resp.status_code == 200
        data = resp.json()
        # nansum of NaN is 0.0, which is < 1e-6, so the link is skipped
        assert len(data["links"]) == 0

    def test_multiple_budget_types(self):
        """Water balance with multiple budget types creates correct node mapping."""
        model = _make_mock_model()
        model_state._model = model

        gw_times = np.arange(3, dtype=float)
        gw_vals = np.array([[100.0, 50.0], [110.0, 55.0], [120.0, 60.0]])
        gw_reader = _make_mock_reader(
            headers=["Deep Percolation", "Pumping"],
            values=(gw_times, gw_vals),
        )

        stream_times = np.arange(3, dtype=float)
        stream_vals = np.array([[200.0], [210.0], [220.0]])
        stream_reader = _make_mock_reader(
            headers=["Upstream Inflow"],
            values=(stream_times, stream_vals),
        )

        def get_reader(btype):
            if btype == "gw":
                return gw_reader
            if btype == "stream":
                return stream_reader
            return None

        model_state.get_available_budgets = MagicMock(
            return_value=["gw", "stream"]
        )
        model_state.get_budget_reader = MagicMock(side_effect=get_reader)
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/water-balance")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["links"]) == 3
        # Nodes should be deduplicated
        assert len(data["nodes"]) == len(set(data["nodes"]))

    def test_unmatched_budget_type_component_name(self):
        """Budget types not in the component map use btype.title() as name."""
        model = _make_mock_model()
        model_state._model = model

        times = np.arange(3, dtype=float)
        vals = np.array([[100.0], [110.0], [120.0]])
        reader = _make_mock_reader(
            headers=["Some Column"],
            values=(times, vals),
        )

        model_state.get_available_budgets = MagicMock(
            return_value=["diversion"]
        )
        model_state.get_budget_reader = MagicMock(return_value=reader)
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/water-balance")
        assert resp.status_code == 200
        data = resp.json()
        # "diversion" is not in {gw, stream, rootzone, lake} map
        # so component name = "diversion".title() = "Diversion"
        assert "Diversion" in data["nodes"]

    def test_lake_budget_component_name(self):
        """Lake budget type uses 'Lakes' as component name."""
        model = _make_mock_model()
        model_state._model = model

        times = np.arange(3, dtype=float)
        vals = np.array([[100.0], [110.0], [120.0]])
        reader = _make_mock_reader(
            headers=["Outflow"],
            values=(times, vals),
        )

        model_state.get_available_budgets = MagicMock(
            return_value=["lake"]
        )
        model_state.get_budget_reader = MagicMock(return_value=reader)
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/water-balance")
        assert resp.status_code == 200
        data = resp.json()
        assert "Lakes" in data["nodes"]

    def test_values_are_rounded(self):
        """Link values should be rounded to 1 decimal."""
        model = _make_mock_model()
        model_state._model = model

        times = np.arange(1, dtype=float)
        vals = np.array([[123.456789]])
        reader = _make_mock_reader(
            headers=["Deep Percolation"],
            values=(times, vals),
        )

        model_state.get_available_budgets = MagicMock(return_value=["gw"])
        model_state.get_budget_reader = MagicMock(return_value=reader)
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/water-balance")
        assert resp.status_code == 200
        data = resp.json()
        link = data["links"][0]
        assert link["value"] == 123.5

    def test_absolute_values_used(self):
        """Water balance uses abs() of summed values."""
        model = _make_mock_model()
        model_state._model = model

        times = np.arange(3, dtype=float)
        vals = np.array([[-100.0], [-110.0], [-120.0]])
        reader = _make_mock_reader(
            headers=["Pumping"],
            values=(times, vals),
        )

        model_state.get_available_budgets = MagicMock(return_value=["gw"])
        model_state.get_budget_reader = MagicMock(return_value=reader)
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/budgets/water-balance")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["links"]) == 1
        assert data["links"][0]["value"] == 330.0  # abs(-330)
