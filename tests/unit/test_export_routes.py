"""Tests for the export API route helpers (notably ``_json_default``).

Covers the JSON-encoding fix for routes/export.py: the four ``json.dumps``
sites previously lacked a ``default=`` callable and would raise
``TypeError`` when payloads contained NumPy ``int64``/``float64`` (e.g.
mesh element indices in GeoJSON exports).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")

from pyiwfm.visualization.webapi.routes.export import _json_default


class TestJsonDefault:
    def test_numpy_integer(self):
        for dtype in (np.int8, np.int16, np.int32, np.int64, np.uint32):
            value = dtype(7)
            assert _json_default(value) == 7
            assert isinstance(_json_default(value), int)

    def test_numpy_floating(self):
        for dtype in (np.float32, np.float64):
            assert _json_default(dtype(1.5)) == pytest.approx(1.5)

    def test_numpy_floating_nan_inf(self):
        assert _json_default(np.float64("nan")) is None
        assert _json_default(np.float64("inf")) is None
        assert _json_default(np.float64("-inf")) is None

    def test_numpy_array(self):
        arr = np.array([1, 2, 3], dtype=np.int64)
        result = _json_default(arr)
        assert result == [1, 2, 3]
        # Each element is a plain Python int after tolist()
        assert all(isinstance(v, int) for v in result)

    def test_pandas_timestamp(self):
        ts = pd.Timestamp("2024-01-15T12:34:56")
        assert _json_default(ts) == "2024-01-15T12:34:56"

    def test_numpy_datetime64(self):
        dt = np.datetime64("2024-01-15T12:34:56")
        result = _json_default(dt)
        assert result.startswith("2024-01-15T12:34:56")

    def test_bytes(self):
        assert _json_default(b"hello") == "hello"

    def test_unsupported_raises(self):
        class Custom:
            pass

        with pytest.raises(TypeError, match="Custom"):
            _json_default(Custom())


class TestJsonDumpsIntegration:
    """End-to-end: json.dumps with our default handles realistic GeoJSON."""

    def test_geojson_with_numpy_int64_indices(self):
        # Mimics a GeoJSON feature whose properties carry mesh node IDs as int64
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [np.float64(1.0), np.float64(2.0)],
                    },
                    "properties": {"node_id": np.int64(42), "head": np.float64(123.456)},
                }
            ],
        }
        payload = json.dumps(geojson, default=_json_default)
        decoded = json.loads(payload)
        feat = decoded["features"][0]
        assert feat["properties"]["node_id"] == 42
        assert feat["properties"]["head"] == pytest.approx(123.456)
        assert feat["geometry"]["coordinates"] == [1.0, 2.0]

    def test_records_with_int64_indices(self):
        # Mirrors the records-export endpoints (lines 762, 879, 956): records
        # built from BudgetReader / hydrograph data may carry np.int64 indices
        # in fields like timestep numbers. Without _json_default these crash.
        records = [
            {"datetime": "2024-01-01", "step": np.int64(0), "head": 100.5},
            {"datetime": "2024-01-02", "step": np.int64(1), "head": 101.2},
        ]
        payload = json.dumps(records, default=_json_default)
        decoded = json.loads(payload)
        assert decoded[0]["step"] == 0
        assert decoded[1]["step"] == 1
        assert isinstance(decoded[0]["step"], int)
