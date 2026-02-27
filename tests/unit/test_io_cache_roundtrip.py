"""Tests for pyiwfm.io.cache_builder + pyiwfm.io.cache_loader roundtrip.

Verifies that data written by SqliteCacheBuilder can be accurately
retrieved by SqliteCacheLoader, including head frames, budget data,
hydrograph timeseries, metadata, timesteps, staleness detection,
and diagnostic statistics.
"""

from __future__ import annotations

import sqlite3
import time
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pyiwfm.io.cache_builder import (
    SCHEMA_VERSION,
    SqliteCacheBuilder,
    _compress_array,
    _compress_int_array,
    get_source_mtimes,
    is_cache_stale,
)
from pyiwfm.io.cache_loader import (
    SqliteCacheLoader,
    _decompress_array,
    _decompress_strings,
)


# ======================================================================
# Compression helpers roundtrip
# ======================================================================


class TestCompressionRoundtrip:
    def test_compress_decompress_float_array(self) -> None:
        arr = np.array([1.0, 2.5, -3.7, 0.0], dtype=np.float64)
        blob = _compress_array(arr)
        recovered = _decompress_array(blob)
        np.testing.assert_array_equal(arr, recovered)

    def test_compress_decompress_2d(self) -> None:
        arr = np.arange(12, dtype=np.float64).reshape(3, 4)
        blob = _compress_array(arr)
        recovered = _decompress_array(blob)
        np.testing.assert_array_equal(arr.ravel(), recovered)

    def test_compress_int_array(self) -> None:
        vals = [1, 2, 3, 4]
        blob = _compress_int_array(vals)
        raw = zlib.decompress(blob)
        import struct

        unpacked = list(struct.unpack(f"<{len(vals)}i", raw))
        assert unpacked == vals

    def test_decompress_strings(self) -> None:
        original = ["2020-01-01", "2020-02-01", "2020-03-01"]
        blob = zlib.compress("\n".join(original).encode("utf-8"), level=1)
        recovered = _decompress_strings(blob)
        assert recovered == original


# ======================================================================
# Mock helpers for model and readers
# ======================================================================


def _make_mock_model(
    n_nodes: int = 4,
    n_elements: int = 2,
    metadata: dict[str, Any] | None = None,
) -> MagicMock:
    """Create a duck-typed mock model with grid, nodes, elements."""
    model = MagicMock()
    model.metadata = metadata or {"name": "test_model"}
    model.streams = None

    # Build grid with nodes
    nodes: dict[int, MagicMock] = {}
    for i in range(1, n_nodes + 1):
        node = MagicMock()
        node.x = float(i)
        node.y = float(i)
        nodes[i] = node

    # Build elements
    elements: dict[int, MagicMock] = {}
    for i in range(1, n_elements + 1):
        elem = MagicMock()
        # Each element uses 2 consecutive nodes
        start = (i - 1) * 2 + 1
        elem.vertices = [start, start + 1]
        elements[i] = elem

    grid = MagicMock()
    grid.nodes = nodes
    grid.elements = elements
    model.grid = grid

    return model


def _make_mock_head_loader(
    n_frames: int = 3,
    n_nodes: int = 4,
    n_layers: int = 2,
) -> MagicMock:
    """Create a duck-typed mock head loader."""
    loader = MagicMock()
    loader.n_frames = n_frames
    loader.times = [datetime(2020 + i // 12, i % 12 + 1, 1) for i in range(n_frames)]

    frames = [
        np.random.rand(n_nodes, n_layers).astype(np.float64) * 100
        for _ in range(n_frames)
    ]
    loader.get_frame = MagicMock(side_effect=lambda idx: frames[idx])
    loader._frames = frames  # store for test verification
    return loader


def _make_mock_budget_reader(
    n_locations: int = 2,
    n_timesteps: int = 5,
    n_cols: int = 3,
) -> MagicMock:
    """Create a duck-typed mock budget reader."""
    reader = MagicMock()
    reader.n_locations = n_locations
    reader.n_timesteps = n_timesteps
    reader.descriptor = "Test Budget"
    reader.location_names = [f"Loc_{i}" for i in range(n_locations)]
    reader.location_areas = [100.0 * (i + 1) for i in range(n_locations)]
    reader.column_names = [f"Col_{i}" for i in range(n_cols)]
    reader.column_units = [f"Unit_{i}" for i in range(n_cols)]

    # Generate deterministic data
    data_map: dict[int, np.ndarray] = {}
    for loc in range(n_locations):
        data_map[loc] = np.arange(
            n_timesteps * n_cols, dtype=np.float64
        ).reshape(n_timesteps, n_cols) + loc * 100

    reader.get_values = MagicMock(side_effect=lambda loc: data_map[loc])
    reader._data_map = data_map  # store for verification
    return reader


def _make_mock_hydrograph_reader(
    n_columns: int = 3,
    n_timesteps: int = 10,
) -> MagicMock:
    """Create a duck-typed mock hydrograph reader with fallback path."""
    reader = MagicMock()
    reader.n_columns = n_columns
    reader.n_timesteps = n_timesteps
    reader.layers = [i + 1 for i in range(n_columns)]
    reader.node_ids = [100 + i for i in range(n_columns)]
    reader.times = [datetime(2020, 1, d + 1) for d in range(n_timesteps)]
    # No _file_path => triggers fallback per-column path
    reader._file_path = None

    ts_data: dict[int, np.ndarray] = {}
    for col in range(n_columns):
        vals = np.arange(n_timesteps, dtype=np.float64) + col * 10.0
        ts_data[col] = vals

    def _get_ts(col_idx: int) -> tuple[list[datetime], np.ndarray]:
        return reader.times, ts_data[col_idx]

    reader.get_time_series = MagicMock(side_effect=_get_ts)
    reader._ts_data = ts_data
    return reader


# ======================================================================
# SqliteCacheBuilder.build: schema, metadata, timesteps
# ======================================================================


class TestCacheBuilderSchemaAndMetadata:
    def test_creates_db_with_schema(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.db"
        builder = SqliteCacheBuilder(cache_path)
        model = _make_mock_model()
        builder.build(model)

        assert cache_path.exists()
        conn = sqlite3.connect(str(cache_path))
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [r[0] for r in cur.fetchall()]
        conn.close()

        assert "metadata" in tables
        assert "timesteps" in tables
        assert "head_frames" in tables
        assert "budget_data" in tables
        assert "hydrograph_series" in tables

    def test_metadata_stored(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.db"
        builder = SqliteCacheBuilder(cache_path)
        model = _make_mock_model(metadata={"name": "MyModel"})
        builder.build(model)

        conn = sqlite3.connect(str(cache_path))
        cur = conn.execute("SELECT value FROM metadata WHERE key = 'schema_version'")
        assert cur.fetchone()[0] == SCHEMA_VERSION
        cur = conn.execute("SELECT value FROM metadata WHERE key = 'model_name'")
        assert cur.fetchone()[0] == "MyModel"
        conn.close()

    def test_deletes_existing_cache(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.db"
        cache_path.write_text("old data")
        builder = SqliteCacheBuilder(cache_path)
        model = _make_mock_model()
        builder.build(model)
        # The file should have been overwritten with a valid SQLite db
        conn = sqlite3.connect(str(cache_path))
        cur = conn.execute("SELECT value FROM metadata WHERE key = 'schema_version'")
        assert cur.fetchone()[0] == SCHEMA_VERSION
        conn.close()


# ======================================================================
# Head data: build -> load roundtrip
# ======================================================================


class TestHeadDataRoundtrip:
    def _build_cache_with_heads(self, tmp_path: Path) -> Path:
        cache_path = tmp_path / "cache.db"
        builder = SqliteCacheBuilder(cache_path)
        model = _make_mock_model(n_nodes=4, n_elements=2)
        head_loader = _make_mock_head_loader(n_frames=3, n_nodes=4, n_layers=2)
        builder.build(model, head_loader=head_loader)
        return cache_path

    def test_head_frames_stored(self, tmp_path: Path) -> None:
        cache_path = self._build_cache_with_heads(tmp_path)
        loader = SqliteCacheLoader(cache_path)

        assert loader.get_n_head_frames() == 3
        loader.close()

    def test_head_frame_roundtrip(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.db"
        builder = SqliteCacheBuilder(cache_path)
        model = _make_mock_model(n_nodes=4, n_elements=2)
        head_loader = _make_mock_head_loader(n_frames=2, n_nodes=4, n_layers=2)
        builder.build(model, head_loader=head_loader)

        loader = SqliteCacheLoader(cache_path)
        for idx in range(2):
            frame = loader.get_head_frame(idx)
            assert frame is not None
            expected = head_loader._frames[idx]
            np.testing.assert_allclose(frame, expected, rtol=1e-12)
        loader.close()

    def test_head_frame_shape(self, tmp_path: Path) -> None:
        cache_path = self._build_cache_with_heads(tmp_path)
        loader = SqliteCacheLoader(cache_path)
        frame = loader.get_head_frame(0)
        assert frame is not None
        assert frame.shape == (4, 2)
        loader.close()

    def test_head_frame_missing_idx(self, tmp_path: Path) -> None:
        cache_path = self._build_cache_with_heads(tmp_path)
        loader = SqliteCacheLoader(cache_path)
        assert loader.get_head_frame(999) is None
        loader.close()

    def test_head_by_element_exists(self, tmp_path: Path) -> None:
        cache_path = self._build_cache_with_heads(tmp_path)
        loader = SqliteCacheLoader(cache_path)
        result = loader.get_head_by_element(0, layer=1)
        assert result is not None
        arr, min_val, max_val = result
        assert len(arr) == 2  # 2 elements
        assert min_val <= max_val
        loader.close()

    def test_head_range_exists(self, tmp_path: Path) -> None:
        cache_path = self._build_cache_with_heads(tmp_path)
        loader = SqliteCacheLoader(cache_path)
        hr = loader.get_head_range(layer=1)
        assert hr is not None
        assert "percentile_02" in hr
        assert "percentile_98" in hr
        assert "abs_min" in hr
        assert "abs_max" in hr
        assert hr["abs_min"] <= hr["abs_max"]
        loader.close()

    def test_timesteps_stored(self, tmp_path: Path) -> None:
        cache_path = self._build_cache_with_heads(tmp_path)
        loader = SqliteCacheLoader(cache_path)
        ts = loader.get_timesteps()
        assert len(ts) == 3
        # Each should be an ISO datetime string
        assert "2020-01-01" in ts[0]
        loader.close()


# ======================================================================
# Budget data: build -> load roundtrip
# ======================================================================


class TestBudgetDataRoundtrip:
    def _build_cache_with_budget(
        self, tmp_path: Path, n_locs: int = 2, n_ts: int = 5, n_cols: int = 3
    ) -> tuple[Path, MagicMock]:
        cache_path = tmp_path / "cache.db"
        builder = SqliteCacheBuilder(cache_path)
        model = _make_mock_model()
        reader = _make_mock_budget_reader(n_locs, n_ts, n_cols)
        builder.build(model, budget_readers={"gw": reader})
        return cache_path, reader

    def test_budget_types(self, tmp_path: Path) -> None:
        cache_path, _ = self._build_cache_with_budget(tmp_path)
        loader = SqliteCacheLoader(cache_path)
        assert loader.get_budget_types() == ["gw"]
        loader.close()

    def test_budget_locations(self, tmp_path: Path) -> None:
        cache_path, _ = self._build_cache_with_budget(tmp_path, n_locs=2)
        loader = SqliteCacheLoader(cache_path)
        locs = loader.get_budget_locations("gw")
        assert len(locs) == 2
        assert locs[0][1] == "Loc_0"
        assert locs[1][2] == pytest.approx(200.0)  # area
        loader.close()

    def test_budget_columns(self, tmp_path: Path) -> None:
        cache_path, _ = self._build_cache_with_budget(tmp_path, n_cols=3)
        loader = SqliteCacheLoader(cache_path)
        cols = loader.get_budget_columns("gw")
        assert len(cols) == 3
        assert cols[0][1] == "Col_0"
        assert cols[0][2] == "Unit_0"
        loader.close()

    def test_budget_data_roundtrip(self, tmp_path: Path) -> None:
        n_locs, n_ts, n_cols = 2, 5, 3
        cache_path, reader = self._build_cache_with_budget(
            tmp_path, n_locs=n_locs, n_ts=n_ts, n_cols=n_cols
        )
        loader = SqliteCacheLoader(cache_path)

        for loc in range(n_locs):
            data = loader.get_budget_data("gw", loc)
            assert data is not None
            expected = reader._data_map[loc]
            np.testing.assert_allclose(data, expected, rtol=1e-12)
        loader.close()

    def test_budget_summary(self, tmp_path: Path) -> None:
        n_locs, n_ts, n_cols = 1, 5, 2
        cache_path, reader = self._build_cache_with_budget(
            tmp_path, n_locs=n_locs, n_ts=n_ts, n_cols=n_cols
        )
        loader = SqliteCacheLoader(cache_path)
        summaries = loader.get_budget_summary("gw", 0)
        assert len(summaries) == n_cols
        expected_data = reader._data_map[0]
        for col_idx, total, avg in summaries:
            col_vals = expected_data[:, col_idx]
            assert total == pytest.approx(float(np.sum(col_vals)), rel=1e-6)
            assert avg == pytest.approx(float(np.mean(col_vals)), rel=1e-6)
        loader.close()

    def test_budget_data_missing_type(self, tmp_path: Path) -> None:
        cache_path, _ = self._build_cache_with_budget(tmp_path)
        loader = SqliteCacheLoader(cache_path)
        assert loader.get_budget_data("nonexistent", 0) is None
        loader.close()

    def test_multi_type_budgets(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.db"
        builder = SqliteCacheBuilder(cache_path)
        model = _make_mock_model()
        gw_reader = _make_mock_budget_reader(n_locations=1, n_timesteps=3, n_cols=2)
        rz_reader = _make_mock_budget_reader(n_locations=2, n_timesteps=4, n_cols=1)
        builder.build(model, budget_readers={"gw": gw_reader, "rz": rz_reader})

        loader = SqliteCacheLoader(cache_path)
        types = loader.get_budget_types()
        assert sorted(types) == ["gw", "rz"]
        assert len(loader.get_budget_locations("gw")) == 1
        assert len(loader.get_budget_locations("rz")) == 2
        loader.close()


# ======================================================================
# Hydrograph data: build -> load roundtrip
# ======================================================================


class TestHydrographDataRoundtrip:
    def _build_cache_with_hydrograph(
        self, tmp_path: Path, n_cols: int = 3, n_ts: int = 10
    ) -> tuple[Path, MagicMock]:
        cache_path = tmp_path / "cache.db"
        builder = SqliteCacheBuilder(cache_path)
        model = _make_mock_model()
        reader = _make_mock_hydrograph_reader(n_columns=n_cols, n_timesteps=n_ts)
        builder.build(model, gw_hydrograph_reader=reader)
        return cache_path, reader

    def test_hydrograph_columns_stored(self, tmp_path: Path) -> None:
        cache_path, reader = self._build_cache_with_hydrograph(tmp_path, n_cols=3)
        loader = SqliteCacheLoader(cache_path)
        cols = loader.get_hydrograph_columns("gw")
        assert len(cols) == 3
        # (col_idx, node_id, layer)
        assert cols[0] == (0, 100, 1)
        assert cols[1] == (1, 101, 2)
        loader.close()

    def test_hydrograph_values_roundtrip(self, tmp_path: Path) -> None:
        n_cols, n_ts = 2, 10
        cache_path, reader = self._build_cache_with_hydrograph(
            tmp_path, n_cols=n_cols, n_ts=n_ts
        )
        loader = SqliteCacheLoader(cache_path)

        for col in range(n_cols):
            result = loader.get_hydrograph(hydro_type="gw", column_idx=col)
            assert result is not None
            times, values = result
            assert len(times) == n_ts
            expected_vals = reader._ts_data[col]
            np.testing.assert_allclose(values, expected_vals, rtol=1e-12)
        loader.close()

    def test_hydrograph_missing(self, tmp_path: Path) -> None:
        cache_path, _ = self._build_cache_with_hydrograph(tmp_path)
        loader = SqliteCacheLoader(cache_path)
        assert loader.get_hydrograph("stream", 0) is None
        loader.close()


# ======================================================================
# SqliteCacheLoader: init, errors, metadata, close
# ======================================================================


class TestCacheLoaderInit:
    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Cache not found"):
            SqliteCacheLoader(tmp_path / "nonexistent.db")

    def test_get_metadata(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.db"
        builder = SqliteCacheBuilder(cache_path)
        builder.build(_make_mock_model(metadata={"name": "TestModel"}))

        loader = SqliteCacheLoader(cache_path)
        assert loader.get_metadata("schema_version") == SCHEMA_VERSION
        assert loader.get_metadata("model_name") == "TestModel"
        assert loader.get_metadata("nonexistent_key") is None
        loader.close()

    def test_close_and_reopen(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.db"
        builder = SqliteCacheBuilder(cache_path)
        builder.build(_make_mock_model())

        loader = SqliteCacheLoader(cache_path)
        loader.get_metadata("schema_version")
        loader.close()
        # Should be able to reopen
        loader2 = SqliteCacheLoader(cache_path)
        assert loader2.get_metadata("schema_version") == SCHEMA_VERSION
        loader2.close()

    def test_close_idempotent(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.db"
        builder = SqliteCacheBuilder(cache_path)
        builder.build(_make_mock_model())

        loader = SqliteCacheLoader(cache_path)
        loader.close()
        loader.close()  # should not raise


# ======================================================================
# get_stats
# ======================================================================


class TestCacheLoaderStats:
    def test_stats_empty_cache(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.db"
        builder = SqliteCacheBuilder(cache_path)
        builder.build(_make_mock_model())

        loader = SqliteCacheLoader(cache_path)
        stats = loader.get_stats()
        assert "head_frames" in stats
        assert "budget_data" in stats
        assert "file_size_mb" in stats
        assert stats["head_frames"] == 0
        loader.close()

    def test_stats_with_data(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.db"
        builder = SqliteCacheBuilder(cache_path)
        model = _make_mock_model(n_nodes=4, n_elements=2)
        head_loader = _make_mock_head_loader(n_frames=2, n_nodes=4, n_layers=1)
        builder.build(model, head_loader=head_loader)

        loader = SqliteCacheLoader(cache_path)
        stats = loader.get_stats()
        assert stats["head_frames"] == 2
        assert stats["file_size_mb"] > 0
        loader.close()


# ======================================================================
# is_cache_stale / get_source_mtimes
# ======================================================================


class TestStalenessDetection:
    def test_missing_cache_is_stale(self, tmp_path: Path) -> None:
        model = _make_mock_model(metadata={"name": "x"})
        assert is_cache_stale(tmp_path / "nonexistent.db", model) is True

    def test_fresh_cache_not_stale(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.db"
        builder = SqliteCacheBuilder(cache_path)
        model = _make_mock_model(metadata={"name": "x"})
        builder.build(model)

        # No source files referenced => not stale
        assert is_cache_stale(cache_path, model) is False

    def test_stale_when_source_newer(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.db"
        builder = SqliteCacheBuilder(cache_path)

        # Create a source file
        src_file = tmp_path / "head_output.hdf"
        src_file.write_text("data")

        model = _make_mock_model(metadata={
            "name": "x",
            "head_output_file": str(src_file),
        })
        builder.build(model)

        # Make source file newer by touching it after a small delay
        time.sleep(0.05)
        src_file.write_text("updated data")

        assert is_cache_stale(cache_path, model) is True

    def test_stale_when_schema_mismatch(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.db"
        builder = SqliteCacheBuilder(cache_path)
        model = _make_mock_model(metadata={"name": "x"})
        builder.build(model)

        # Tamper with schema version
        conn = sqlite3.connect(str(cache_path))
        conn.execute("UPDATE metadata SET value = 'old' WHERE key = 'schema_version'")
        conn.commit()
        conn.close()

        assert is_cache_stale(cache_path, model) is True

    def test_get_source_mtimes_no_files(self) -> None:
        model = _make_mock_model(metadata={"name": "x"})
        mtimes = get_source_mtimes(model)
        assert mtimes == {}

    def test_get_source_mtimes_with_files(self, tmp_path: Path) -> None:
        sim_file = tmp_path / "sim.dat"
        sim_file.write_text("sim")

        model = _make_mock_model(metadata={
            "name": "x",
            "simulation_file": str(sim_file),
        })
        mtimes = get_source_mtimes(model)
        assert "simulation_file" in mtimes
        assert mtimes["simulation_file"] > 0

    def test_get_source_mtimes_nonexistent_path(self) -> None:
        model = _make_mock_model(metadata={
            "name": "x",
            "simulation_file": "/nonexistent/path.dat",
        })
        mtimes = get_source_mtimes(model)
        assert mtimes == {}


# ======================================================================
# Edge cases
# ======================================================================


class TestEdgeCases:
    def test_build_with_no_loaders(self, tmp_path: Path) -> None:
        """Building with no loaders should produce a valid (mostly empty) cache."""
        cache_path = tmp_path / "cache.db"
        builder = SqliteCacheBuilder(cache_path)
        model = _make_mock_model()
        builder.build(model)

        loader = SqliteCacheLoader(cache_path)
        assert loader.get_n_head_frames() == 0
        assert loader.get_budget_types() == []
        assert loader.get_timesteps() == []
        loader.close()

    def test_head_loader_with_zero_frames(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.db"
        builder = SqliteCacheBuilder(cache_path)
        model = _make_mock_model()
        head_loader = MagicMock()
        head_loader.n_frames = 0
        builder.build(model, head_loader=head_loader)

        loader = SqliteCacheLoader(cache_path)
        assert loader.get_n_head_frames() == 0
        loader.close()

    def test_budget_reader_with_zero_locations(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.db"
        builder = SqliteCacheBuilder(cache_path)
        model = _make_mock_model()
        reader = MagicMock()
        reader.n_locations = 0
        reader.n_timesteps = 5
        builder.build(model, budget_readers={"empty": reader})

        loader = SqliteCacheLoader(cache_path)
        assert loader.get_budget_types() == []
        loader.close()

    def test_budget_reader_get_values_returns_none(self, tmp_path: Path) -> None:
        """If reader.get_values returns None for a location, skip it gracefully."""
        cache_path = tmp_path / "cache.db"
        builder = SqliteCacheBuilder(cache_path)
        model = _make_mock_model()
        reader = MagicMock()
        reader.n_locations = 1
        reader.n_timesteps = 3
        reader.descriptor = "broken"
        reader.location_names = ["Broken"]
        reader.location_areas = [0.0]
        reader.column_names = ["col1"]
        reader.column_units = ["ft"]
        reader.get_values = MagicMock(return_value=None)
        builder.build(model, budget_readers={"broken": reader})

        loader = SqliteCacheLoader(cache_path)
        assert loader.get_budget_data("broken", 0) is None
        loader.close()

    def test_progress_callback_called(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.db"
        builder = SqliteCacheBuilder(cache_path)
        model = _make_mock_model(n_nodes=4, n_elements=2)
        # Use enough frames to trigger progress (multiple of 20)
        head_loader = _make_mock_head_loader(n_frames=21, n_nodes=4, n_layers=1)
        cb = MagicMock()
        builder.build(model, head_loader=head_loader, progress_callback=cb)
        # Progress callback should have been called at least once
        assert cb.call_count >= 1
