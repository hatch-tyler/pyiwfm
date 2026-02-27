"""Deep coverage tests for pyiwfm.io.cache_builder.

Targets uncovered paths not exercised by the existing test_io_cache_roundtrip.py:
- Lines 436-437: _build_head_data element averaging with multi-node elements
- Lines 516-517, 552, 560, 568: budget caching exception branches
- Lines 603-632: _build_hydrographs with different reader types (gw, stream, subsidence, tile_drain)
- Lines 654-655, 682-701: _cache_hydrograph_with_metadata HDF5 bulk read path
- Lines 716-755: _build_stream_ratings
- Lines 800-801: _build_area_data
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from pyiwfm.io.cache_builder import (
    SqliteCacheBuilder,
)
from pyiwfm.io.cache_loader import SqliteCacheLoader

# ======================================================================
# Mock helpers
# ======================================================================


def _make_mock_model(
    n_nodes: int = 9,
    n_elements: int = 4,
    vertices_per_element: int = 4,
    metadata: dict[str, Any] | None = None,
    with_streams: bool = False,
) -> MagicMock:
    """Create a mock model with configurable grid and optional streams."""
    model = MagicMock()
    model.metadata = metadata or {"name": "test_model"}

    # Build grid with nodes
    nodes: dict[int, MagicMock] = {}
    for i in range(1, n_nodes + 1):
        node = MagicMock()
        node.x = float(i * 10)
        node.y = float(i * 10)
        nodes[i] = node

    # Build elements with multi-node vertices (e.g., 4-node quads)
    elements: dict[int, MagicMock] = {}
    node_idx = 1
    for i in range(1, n_elements + 1):
        elem = MagicMock()
        verts = list(range(node_idx, min(node_idx + vertices_per_element, n_nodes + 1)))
        # Wrap around if we run out of nodes
        while len(verts) < vertices_per_element:
            verts.append((verts[-1] % n_nodes) + 1)
        elem.vertices = verts
        elements[i] = elem
        node_idx = min(node_idx + vertices_per_element, n_nodes)

    grid = MagicMock()
    grid.nodes = nodes
    grid.elements = elements
    model.grid = grid

    if with_streams:
        model.streams = _make_mock_streams()
    else:
        model.streams = None

    return model


def _make_mock_streams(n_nodes: int = 3) -> MagicMock:
    """Create mock streams with rating tables."""
    streams = MagicMock()
    stream_nodes: dict[int, MagicMock] = {}

    for i in range(1, n_nodes + 1):
        sn = MagicMock()
        sn.id = i
        sn.bottom_elev = float(100 - i * 10)
        rating = MagicMock()
        rating.stages = [0.0, 1.0, 2.0, 5.0, 10.0]
        rating.flows = [0.0, 10.0, 50.0, 200.0, 1000.0]
        sn.rating = rating
        stream_nodes[i] = sn

    streams.nodes = stream_nodes
    return streams


def _make_mock_head_loader(
    n_frames: int = 3,
    n_nodes: int = 9,
    n_layers: int = 2,
) -> MagicMock:
    """Create a mock head loader with deterministic data."""
    loader = MagicMock()
    loader.n_frames = n_frames
    loader.times = [datetime(2020 + m // 12, m % 12 + 1, 1) for m in range(n_frames)]

    frames = []
    for f in range(n_frames):
        # Create head values that are above -9000 (the no-data threshold)
        frame = np.arange(n_nodes * n_layers, dtype=np.float64).reshape(n_nodes, n_layers)
        frame += f * 100 + 50  # shift so all values are well above -9000
        frames.append(frame)

    loader.get_frame = MagicMock(side_effect=lambda idx: frames[idx])
    loader._frames = frames
    return loader


def _make_mock_budget_reader(
    n_locations: int = 2,
    n_timesteps: int = 5,
    n_cols: int = 3,
    fail_on_loc: int | None = None,
) -> MagicMock:
    """Create a mock budget reader. Optionally raise on a specific location."""
    reader = MagicMock()
    reader.n_locations = n_locations
    reader.n_timesteps = n_timesteps
    reader.descriptor = "Test Budget"
    reader.location_names = [f"Loc_{i}" for i in range(n_locations)]
    reader.location_areas = [100.0 * (i + 1) for i in range(n_locations)]
    reader.column_names = [f"Col_{i}" for i in range(n_cols)]
    reader.column_units = [f"Unit_{i}" for i in range(n_cols)]

    data_map: dict[int, np.ndarray] = {}
    for loc in range(n_locations):
        data_map[loc] = (
            np.arange(n_timesteps * n_cols, dtype=np.float64).reshape(n_timesteps, n_cols)
            + loc * 100
        )

    def _get_values(loc: int) -> np.ndarray | None:
        if fail_on_loc is not None and loc == fail_on_loc:
            raise ValueError(f"simulated read error for loc {loc}")
        return data_map[loc]

    reader.get_values = MagicMock(side_effect=_get_values)
    reader._data_map = data_map
    return reader


def _make_mock_hydrograph_reader(
    n_columns: int = 3,
    n_timesteps: int = 10,
    with_hdf_path: str | None = None,
) -> MagicMock:
    """Create a mock hydrograph reader.

    Parameters
    ----------
    with_hdf_path : str, optional
        If set, the reader's _file_path points to this path (for HDF5 bulk path).
        If None, triggers the fallback per-column path.
    """
    reader = MagicMock()
    reader.n_columns = n_columns
    reader.n_timesteps = n_timesteps
    reader.layers = [i + 1 for i in range(n_columns)]
    reader.node_ids = [100 + i for i in range(n_columns)]
    reader.times = [datetime(2020, 1, d + 1) for d in range(n_timesteps)]
    reader._file_path = with_hdf_path

    ts_data: dict[int, np.ndarray] = {}
    for col in range(n_columns):
        vals = np.arange(n_timesteps, dtype=np.float64) + col * 10.0
        ts_data[col] = vals

    def _get_ts(col_idx: int) -> tuple[list[datetime], np.ndarray]:
        return reader.times, ts_data[col_idx]

    reader.get_time_series = MagicMock(side_effect=_get_ts)
    reader._ts_data = ts_data
    return reader


def _make_mock_area_manager(n_timesteps: int = 6, n_elements: int = 4) -> MagicMock:
    """Create a mock area data manager."""
    mgr = MagicMock()
    mgr.n_timesteps = n_timesteps
    mgr.dates = [datetime(2020, m + 1, 1) for m in range(n_timesteps)]

    def _get_snapshot(ts_idx: int) -> list[dict]:
        return [
            {
                "element_id": eid,
                "agricultural": 50.0 + eid,
                "urban": 20.0 + eid,
                "native_rip": 30.0 + eid,
                "total_area": 100.0 + eid,
                "dominant": "agricultural" if eid % 2 == 1 else "urban",
            }
            for eid in range(1, n_elements + 1)
        ]

    mgr.get_land_use_snapshot = MagicMock(side_effect=_get_snapshot)
    return mgr


# ======================================================================
# Head data: element averaging with multi-node elements (lines 436-437)
# ======================================================================


class TestHeadDataMultiNodeElements:
    def test_element_averaging_quad_elements(self, tmp_path: Path) -> None:
        """Elements with 4 nodes get averaged correctly."""
        cache_path = tmp_path / "cache.db"
        builder = SqliteCacheBuilder(cache_path)
        # 9 nodes, 4 elements with 4 vertices each
        model = _make_mock_model(n_nodes=9, n_elements=4, vertices_per_element=4)
        head_loader = _make_mock_head_loader(n_frames=2, n_nodes=9, n_layers=2)
        builder.build(model, head_loader=head_loader)

        loader = SqliteCacheLoader(cache_path)
        result = loader.get_head_by_element(0, layer=1)
        assert result is not None
        arr, min_val, max_val = result
        assert len(arr) == 4  # 4 elements
        assert min_val <= max_val
        # All values should be finite (no NaN since all node values are > -9000)
        assert np.all(np.isfinite(arr))
        loader.close()

    def test_head_range_stored_for_each_layer(self, tmp_path: Path) -> None:
        """Head range (percentile + extremes) is stored per layer."""
        cache_path = tmp_path / "cache.db"
        builder = SqliteCacheBuilder(cache_path)
        model = _make_mock_model(n_nodes=9, n_elements=4, vertices_per_element=4)
        head_loader = _make_mock_head_loader(n_frames=5, n_nodes=9, n_layers=3)
        builder.build(model, head_loader=head_loader)

        loader = SqliteCacheLoader(cache_path)
        for layer in [1, 2, 3]:
            hr = loader.get_head_range(layer)
            assert hr is not None
            assert hr["percentile_02"] <= hr["percentile_98"]
            assert hr["abs_min"] <= hr["abs_max"]
        loader.close()

    def test_head_no_data_nodes_excluded(self, tmp_path: Path) -> None:
        """Nodes with values <= -9000 are excluded from element averaging."""
        cache_path = tmp_path / "cache.db"
        builder = SqliteCacheBuilder(cache_path)
        model = _make_mock_model(n_nodes=4, n_elements=1, vertices_per_element=4)

        loader_mock = MagicMock()
        loader_mock.n_frames = 1
        loader_mock.times = [datetime(2020, 1, 1)]
        # First 2 nodes have valid data, last 2 have no-data (-9999)
        frame = np.array([[100.0, 200.0], [150.0, 250.0], [-9999.0, -9999.0], [-9999.0, -9999.0]])
        loader_mock.get_frame = MagicMock(return_value=frame)

        builder.build(model, head_loader=loader_mock)

        loader = SqliteCacheLoader(cache_path)
        result = loader.get_head_by_element(0, layer=1)
        assert result is not None
        arr, _, _ = result
        # Element average should be mean of valid nodes (100, 150) = 125
        np.testing.assert_allclose(arr[0], 125.0)
        loader.close()

    def test_grid_none_skips_head_data(self, tmp_path: Path) -> None:
        """If model.grid is None, head data building is skipped."""
        cache_path = tmp_path / "cache.db"
        builder = SqliteCacheBuilder(cache_path)
        model = MagicMock()
        model.metadata = {"name": "no-grid"}
        model.grid = None
        model.streams = None
        head_loader = _make_mock_head_loader(n_frames=2, n_nodes=4, n_layers=1)
        builder.build(model, head_loader=head_loader)

        loader = SqliteCacheLoader(cache_path)
        # Timesteps are stored but no element data
        assert loader.get_n_head_frames() == 0
        loader.close()


# ======================================================================
# Budget caching: exception branches (lines 516-517)
# ======================================================================


class TestBudgetCachingExceptions:
    def test_budget_get_values_exception_for_one_location(self, tmp_path: Path) -> None:
        """Exception in get_values for one loc should not abort other locs."""
        cache_path = tmp_path / "cache.db"
        builder = SqliteCacheBuilder(cache_path)
        model = _make_mock_model(n_nodes=4, n_elements=1)
        # fail_on_loc=0 means loc 0 raises, but loc 1 should succeed
        reader = _make_mock_budget_reader(n_locations=2, n_timesteps=3, n_cols=2, fail_on_loc=0)
        builder.build(model, budget_readers={"gw": reader})

        loader = SqliteCacheLoader(cache_path)
        # Location 0 data should be None (failed)
        assert loader.get_budget_data("gw", 0) is None
        # Location 1 data should exist
        data = loader.get_budget_data("gw", 1)
        assert data is not None
        assert data.shape == (3, 2)
        loader.close()

    def test_entire_budget_reader_exception(self, tmp_path: Path) -> None:
        """Exception in _cache_single_budget should be caught and logged."""
        cache_path = tmp_path / "cache.db"
        builder = SqliteCacheBuilder(cache_path)
        model = _make_mock_model(n_nodes=4, n_elements=1)

        # Reader that raises immediately when accessing properties
        bad_reader = MagicMock()
        bad_reader.n_locations = MagicMock(side_effect=RuntimeError("corrupt"))
        type(bad_reader).n_locations = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("corrupt"))
        )

        # Another reader that works
        good_reader = _make_mock_budget_reader(n_locations=1, n_timesteps=2, n_cols=1)

        builder.build(model, budget_readers={"bad": bad_reader, "good": good_reader})

        loader = SqliteCacheLoader(cache_path)
        types = loader.get_budget_types()
        assert "good" in types
        loader.close()

    def test_budget_1d_data_reshaped(self, tmp_path: Path) -> None:
        """1D budget data (single column) is reshaped to 2D before storage."""
        cache_path = tmp_path / "cache.db"
        builder = SqliteCacheBuilder(cache_path)
        model = _make_mock_model(n_nodes=4, n_elements=1)

        reader = MagicMock()
        reader.n_locations = 1
        reader.n_timesteps = 3
        reader.descriptor = "1D Budget"
        reader.location_names = ["Loc"]
        reader.location_areas = [100.0]
        reader.column_names = ["Flow"]
        reader.column_units = ["cfs"]
        # Return 1D array (will be reshaped to (-1, 1))
        reader.get_values = MagicMock(return_value=np.array([10.0, 20.0, 30.0]))

        builder.build(model, budget_readers={"flow": reader})

        loader = SqliteCacheLoader(cache_path)
        data = loader.get_budget_data("flow", 0)
        assert data is not None
        assert data.shape == (3, 1)
        np.testing.assert_allclose(data[:, 0], [10.0, 20.0, 30.0])
        loader.close()


# ======================================================================
# Hydrographs: different reader types (lines 603-632)
# ======================================================================


class TestHydrographMultipleTypes:
    def test_all_hydrograph_types_cached(self, tmp_path: Path) -> None:
        """GW, stream, subsidence, and tile_drain readers all get cached."""
        cache_path = tmp_path / "cache.db"
        builder = SqliteCacheBuilder(cache_path)
        model = _make_mock_model(n_nodes=4, n_elements=1)

        gw_reader = _make_mock_hydrograph_reader(n_columns=2, n_timesteps=5)
        stream_reader = _make_mock_hydrograph_reader(n_columns=3, n_timesteps=5)
        subsidence_reader = _make_mock_hydrograph_reader(n_columns=1, n_timesteps=5)
        tile_drain_reader = _make_mock_hydrograph_reader(n_columns=2, n_timesteps=5)

        builder.build(
            model,
            gw_hydrograph_reader=gw_reader,
            stream_hydrograph_reader=stream_reader,
            subsidence_reader=subsidence_reader,
            tile_drain_reader=tile_drain_reader,
        )

        loader = SqliteCacheLoader(cache_path)
        gw_cols = loader.get_hydrograph_columns("gw")
        assert len(gw_cols) == 2

        stream_cols = loader.get_hydrograph_columns("stream")
        assert len(stream_cols) == 3

        sub_cols = loader.get_hydrograph_columns("subsidence")
        assert len(sub_cols) == 1

        td_cols = loader.get_hydrograph_columns("tile_drain")
        assert len(td_cols) == 2

        # Verify data roundtrip for stream reader
        result = loader.get_hydrograph("stream", 0)
        assert result is not None
        times, values = result
        assert len(times) == 5
        expected = stream_reader._ts_data[0]
        np.testing.assert_allclose(values, expected, rtol=1e-12)

        loader.close()

    def test_zero_timestep_reader_skipped(self, tmp_path: Path) -> None:
        """Reader with n_timesteps=0 is skipped."""
        cache_path = tmp_path / "cache.db"
        builder = SqliteCacheBuilder(cache_path)
        model = _make_mock_model(n_nodes=4, n_elements=1)

        empty_reader = MagicMock()
        empty_reader.n_timesteps = 0
        empty_reader.n_columns = 3

        builder.build(model, stream_hydrograph_reader=empty_reader)

        loader = SqliteCacheLoader(cache_path)
        assert loader.get_hydrograph_columns("stream") == []
        loader.close()

    def test_hydrograph_get_time_series_exception(self, tmp_path: Path) -> None:
        """Exception in get_time_series for one column should not abort others."""
        cache_path = tmp_path / "cache.db"
        builder = SqliteCacheBuilder(cache_path)
        model = _make_mock_model(n_nodes=4, n_elements=1)

        reader = MagicMock()
        reader.n_columns = 3
        reader.n_timesteps = 5
        reader.layers = [1, 2, 3]
        reader.node_ids = [100, 101, 102]
        reader.times = [datetime(2020, 1, d + 1) for d in range(5)]
        reader._file_path = None  # Force fallback path

        call_count = [0]

        def _get_ts(col_idx: int) -> tuple[list[datetime], np.ndarray]:
            call_count[0] += 1
            if col_idx == 1:
                raise ValueError("bad column")
            return reader.times, np.arange(5, dtype=np.float64) + col_idx * 10

        reader.get_time_series = MagicMock(side_effect=_get_ts)

        builder.build(model, gw_hydrograph_reader=reader)

        loader = SqliteCacheLoader(cache_path)
        # Column 0 and 2 should exist, column 1 should be missing
        result0 = loader.get_hydrograph("gw", 0)
        assert result0 is not None
        result1 = loader.get_hydrograph("gw", 1)
        assert result1 is None  # Failed column
        result2 = loader.get_hydrograph("gw", 2)
        assert result2 is not None
        loader.close()

    def test_zero_column_reader_skipped(self, tmp_path: Path) -> None:
        """Reader with n_columns=0 is skipped in _cache_hydrograph_with_metadata."""
        cache_path = tmp_path / "cache.db"
        builder = SqliteCacheBuilder(cache_path)
        model = _make_mock_model(n_nodes=4, n_elements=1)

        reader = MagicMock()
        reader.n_columns = 0
        reader.n_timesteps = 5
        reader.layers = []
        reader.node_ids = []
        reader.times = []
        reader._file_path = None

        builder.build(model, gw_hydrograph_reader=reader)

        loader = SqliteCacheLoader(cache_path)
        assert loader.get_hydrograph_columns("gw") == []
        loader.close()


# ======================================================================
# HDF5 bulk read path (lines 654-655, 682-701)
# ======================================================================


class TestHydrographHdf5BulkPath:
    def test_hdf5_bulk_read_path(self, tmp_path: Path) -> None:
        """Exercise the HDF5 bulk read code path with a real HDF5 file."""
        pytest.importorskip("h5py")
        import h5py

        n_timesteps = 8
        n_columns = 5

        # Create a real HDF5 file for the bulk path
        hdf_path = tmp_path / "hydrographs.hdf"
        data = np.arange(n_timesteps * n_columns, dtype=np.float64).reshape(n_timesteps, n_columns)
        with h5py.File(str(hdf_path), "w") as f:
            f.create_dataset("data", data=data)

        cache_path = tmp_path / "cache.db"
        builder = SqliteCacheBuilder(cache_path)
        model = _make_mock_model(n_nodes=4, n_elements=1)

        reader = MagicMock()
        reader.n_columns = n_columns
        reader.n_timesteps = n_timesteps
        reader.layers = [i + 1 for i in range(n_columns)]
        reader.node_ids = [200 + i for i in range(n_columns)]
        reader.times = [datetime(2020, 1, d + 1) for d in range(n_timesteps)]
        reader._file_path = str(hdf_path)

        builder.build(model, gw_hydrograph_reader=reader)

        loader = SqliteCacheLoader(cache_path)
        cols = loader.get_hydrograph_columns("gw")
        assert len(cols) == n_columns

        for col_idx in range(n_columns):
            result = loader.get_hydrograph("gw", col_idx)
            assert result is not None
            times, values = result
            assert len(times) == n_timesteps
            expected = data[:, col_idx]
            np.testing.assert_allclose(values, expected, rtol=1e-12)

        loader.close()

    def test_hdf5_bulk_large_columns_chunked(self, tmp_path: Path) -> None:
        """Exercise chunking logic with > CHUNK (500) columns."""
        pytest.importorskip("h5py")
        import h5py

        n_timesteps = 3
        n_columns = 520  # > 500 to trigger chunk boundary

        hdf_path = tmp_path / "big_hydro.hdf"
        rng = np.random.default_rng(42)
        data = rng.random((n_timesteps, n_columns)).astype(np.float64)
        with h5py.File(str(hdf_path), "w") as f:
            f.create_dataset("data", data=data)

        cache_path = tmp_path / "cache.db"
        builder = SqliteCacheBuilder(cache_path)
        model = _make_mock_model(n_nodes=4, n_elements=1)

        reader = MagicMock()
        reader.n_columns = n_columns
        reader.n_timesteps = n_timesteps
        reader.layers = [1] * n_columns
        reader.node_ids = list(range(n_columns))
        reader.times = [datetime(2020, 1, d + 1) for d in range(n_timesteps)]
        reader._file_path = str(hdf_path)

        builder.build(model, gw_hydrograph_reader=reader)

        loader = SqliteCacheLoader(cache_path)
        cols = loader.get_hydrograph_columns("gw")
        assert len(cols) == n_columns

        # Spot-check first and last columns
        for ci in [0, 499, 500, 519]:
            result = loader.get_hydrograph("gw", ci)
            assert result is not None
            _, values = result
            np.testing.assert_allclose(values, data[:, ci], rtol=1e-12)

        loader.close()


# ======================================================================
# Stream rating tables (lines 716-755)
# ======================================================================


class TestStreamRatings:
    def test_stream_ratings_cached(self, tmp_path: Path) -> None:
        """Stream rating tables are stored and retrievable."""
        cache_path = tmp_path / "cache.db"
        builder = SqliteCacheBuilder(cache_path)
        model = _make_mock_model(n_nodes=4, n_elements=1, with_streams=True)
        builder.build(model)

        loader = SqliteCacheLoader(cache_path)
        for sn_id in [1, 2, 3]:
            result = loader.get_stream_rating(sn_id)
            assert result is not None
            bottom_elev, stages, flows = result
            assert isinstance(bottom_elev, float)
            assert len(stages) == 5
            assert len(flows) == 5
            np.testing.assert_allclose(stages, [0.0, 1.0, 2.0, 5.0, 10.0])
            np.testing.assert_allclose(flows, [0.0, 10.0, 50.0, 200.0, 1000.0])

        # Non-existent node
        assert loader.get_stream_rating(999) is None
        loader.close()

    def test_stream_nodes_without_rating_skipped(self, tmp_path: Path) -> None:
        """Stream nodes that have no rating attribute are skipped."""
        cache_path = tmp_path / "cache.db"
        builder = SqliteCacheBuilder(cache_path)

        model = _make_mock_model(n_nodes=4, n_elements=1)
        streams = MagicMock()
        node1 = MagicMock()
        node1.id = 1
        node1.rating = None  # No rating
        node1.bottom_elev = 50.0

        node2 = MagicMock()
        node2.id = 2
        rating = MagicMock()
        rating.stages = [0.0, 1.0]
        rating.flows = [0.0, 100.0]
        node2.rating = rating
        node2.bottom_elev = 40.0

        streams.nodes = {1: node1, 2: node2}
        model.streams = streams

        builder.build(model)

        loader = SqliteCacheLoader(cache_path)
        assert loader.get_stream_rating(1) is None  # No rating stored
        result = loader.get_stream_rating(2)
        assert result is not None
        loader.close()

    def test_no_streams_on_model(self, tmp_path: Path) -> None:
        """If model.streams is None, no ratings are cached."""
        cache_path = tmp_path / "cache.db"
        builder = SqliteCacheBuilder(cache_path)
        model = _make_mock_model(n_nodes=4, n_elements=1, with_streams=False)
        builder.build(model)

        loader = SqliteCacheLoader(cache_path)
        assert loader.get_stream_rating(1) is None
        loader.close()


# ======================================================================
# Area / land-use data (lines 800-801)
# ======================================================================


class TestAreaData:
    def test_area_data_cached(self, tmp_path: Path) -> None:
        """Area data snapshots and timesteps are cached."""
        cache_path = tmp_path / "cache.db"
        builder = SqliteCacheBuilder(cache_path)
        model = _make_mock_model(n_nodes=4, n_elements=1)
        area_mgr = _make_mock_area_manager(n_timesteps=6, n_elements=4)

        builder.build(model, area_manager=area_mgr)

        loader = SqliteCacheLoader(cache_path)
        timesteps = loader.get_area_timesteps()
        assert len(timesteps) == 6

        # Get a snapshot for the first timestep (index 0)
        snapshot = loader.get_area_snapshot(0)
        assert snapshot is not None
        assert len(snapshot) == 4  # 4 elements
        assert snapshot[0]["element_id"] == 1
        assert snapshot[0]["dominant"] in ("agricultural", "urban")
        loader.close()

    def test_area_manager_none_skipped(self, tmp_path: Path) -> None:
        """If area_manager is None, no area data is cached."""
        cache_path = tmp_path / "cache.db"
        builder = SqliteCacheBuilder(cache_path)
        model = _make_mock_model(n_nodes=4, n_elements=1)
        builder.build(model, area_manager=None)

        loader = SqliteCacheLoader(cache_path)
        assert loader.get_area_timesteps() == []
        assert loader.get_area_snapshot(0) is None
        loader.close()

    def test_area_zero_timesteps_skipped(self, tmp_path: Path) -> None:
        """If area_manager.n_timesteps is 0, no data is cached."""
        cache_path = tmp_path / "cache.db"
        builder = SqliteCacheBuilder(cache_path)
        model = _make_mock_model(n_nodes=4, n_elements=1)

        mgr = MagicMock()
        mgr.n_timesteps = 0
        builder.build(model, area_manager=mgr)

        loader = SqliteCacheLoader(cache_path)
        assert loader.get_area_timesteps() == []
        loader.close()

    def test_area_snapshot_exception_caught(self, tmp_path: Path) -> None:
        """Exception in get_land_use_snapshot for one timestep is caught."""
        cache_path = tmp_path / "cache.db"
        builder = SqliteCacheBuilder(cache_path)
        model = _make_mock_model(n_nodes=4, n_elements=1)

        mgr = MagicMock()
        mgr.n_timesteps = 3
        mgr.dates = [datetime(2020, m + 1, 1) for m in range(3)]

        def _get_snapshot(ts_idx: int) -> list[dict] | None:
            if ts_idx == 0:
                raise RuntimeError("corrupt data")
            return [
                {
                    "element_id": 1,
                    "agricultural": 50.0,
                    "urban": 20.0,
                    "native_rip": 30.0,
                    "total_area": 100.0,
                    "dominant": "agricultural",
                }
            ]

        mgr.get_land_use_snapshot = MagicMock(side_effect=_get_snapshot)
        builder.build(model, area_manager=mgr)

        # Should not raise; corrupt timestep is skipped
        loader = SqliteCacheLoader(cache_path)
        assert loader.get_area_timesteps() == [
            str(datetime(2020, 1, 1)),
            str(datetime(2020, 2, 1)),
            str(datetime(2020, 3, 1)),
        ]
        loader.close()


# ======================================================================
# Progress callback
# ======================================================================


class TestProgressCallback:
    def test_progress_callback_invoked(self, tmp_path: Path) -> None:
        """Progress callback is called during head caching."""
        cache_path = tmp_path / "cache.db"
        builder = SqliteCacheBuilder(cache_path)
        model = _make_mock_model(n_nodes=9, n_elements=4, vertices_per_element=4)
        head_loader = _make_mock_head_loader(n_frames=25, n_nodes=9, n_layers=1)
        cb = MagicMock()

        builder.build(model, head_loader=head_loader, progress_callback=cb)

        # Should be called for frames 0 and 20 at minimum (every 20 frames)
        assert cb.call_count >= 2
        # First call should be ("head_frames", 0)
        cb.assert_any_call("head_frames", 0)
