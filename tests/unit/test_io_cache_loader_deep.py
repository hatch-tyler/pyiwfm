"""Deep coverage tests for pyiwfm.io.cache_loader.

Targets the UNCOVERED lines not exercised by the existing test_io_cache_roundtrip.py:
- Lines 241-258: get_gw_hydrograph_all_layers()
- Lines 270-288: get_gw_hydrograph_by_columns()
- Lines 299-307: get_stream_rating()
- Lines 322-331: get_area_snapshot()
- Lines 345-349, 375-376, 381-382: get_stats(), get_area_timesteps()

Rather than relying on the builder, this file creates a SQLite database
directly from the schema, inserts test data, and tests the loader methods.
"""

from __future__ import annotations

import sqlite3
import zlib
from pathlib import Path

import numpy as np
import pytest

# The schema DDL from cache_builder is needed to create the database
from pyiwfm.io.cache_builder import (
    _SCHEMA_SQL,
    SCHEMA_VERSION,
    _compress_array,
)
from pyiwfm.io.cache_loader import SqliteCacheLoader

# ======================================================================
# Helpers
# ======================================================================


def _create_test_db(db_path: Path) -> sqlite3.Connection:
    """Create a test SQLite database with the viewer cache schema."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA_SQL)
    conn.execute(
        "INSERT INTO metadata VALUES (?, ?)",
        ("schema_version", SCHEMA_VERSION),
    )
    conn.execute(
        "INSERT INTO metadata VALUES (?, ?)",
        ("model_name", "test_model"),
    )
    conn.execute(
        "INSERT INTO metadata VALUES (?, ?)",
        ("build_time", "2025-01-01T00:00:00"),
    )
    conn.commit()
    return conn


def _insert_hydrograph_data(
    conn: sqlite3.Connection,
    hydro_type: str,
    col_idx: int,
    node_id: int,
    layer: int,
    times: list[str],
    values: np.ndarray,
) -> None:
    """Insert a hydrograph column + series into the test database."""
    conn.execute(
        "INSERT INTO hydrograph_columns VALUES (?, ?, ?, ?)",
        (hydro_type, col_idx, node_id, layer),
    )
    times_blob = zlib.compress("\n".join(times).encode("utf-8"), level=1)
    conn.execute(
        "INSERT INTO hydrograph_series VALUES (?, ?, ?, ?)",
        (hydro_type, col_idx, times_blob, _compress_array(values)),
    )


def _insert_stream_rating(
    conn: sqlite3.Connection,
    stream_node_id: int,
    bottom_elev: float,
    stages: np.ndarray,
    flows: np.ndarray,
) -> None:
    """Insert a stream rating table into the test database."""
    conn.execute(
        "INSERT INTO stream_ratings VALUES (?, ?, ?, ?)",
        (
            stream_node_id,
            bottom_elev,
            _compress_array(stages),
            _compress_array(flows),
        ),
    )


def _insert_landuse_snapshot(
    conn: sqlite3.Connection,
    frame_idx: int,
    element_id: int,
    agricultural: float,
    urban: float,
    native_rip: float,
    total_area: float,
    dominant: str,
) -> None:
    """Insert a land-use snapshot row."""
    conn.execute(
        "INSERT INTO landuse_snapshots VALUES (?, ?, ?, ?, ?, ?, ?)",
        (frame_idx, element_id, agricultural, urban, native_rip, total_area, dominant),
    )


def _insert_area_timestep(
    conn: sqlite3.Connection,
    area_type: str,
    idx: int,
    dt: str,
) -> None:
    """Insert an area timestep entry."""
    conn.execute(
        "INSERT INTO area_timesteps VALUES (?, ?, ?)",
        (area_type, idx, dt),
    )


# ======================================================================
# get_gw_hydrograph_all_layers (lines 241-258)
# ======================================================================


class TestGetGwHydrographAllLayers:
    def test_returns_all_layers_for_node(self, tmp_path: Path) -> None:
        """get_gw_hydrograph_all_layers returns data for all layers at a node."""
        db_path = tmp_path / "cache.db"
        conn = _create_test_db(db_path)

        times = ["2020-01-01", "2020-02-01", "2020-03-01"]

        # Node 101, layer 1
        vals_l1 = np.array([10.0, 11.0, 12.0], dtype=np.float64)
        _insert_hydrograph_data(
            conn, "gw", col_idx=0, node_id=101, layer=1, times=times, values=vals_l1
        )

        # Node 101, layer 2
        vals_l2 = np.array([20.0, 21.0, 22.0], dtype=np.float64)
        _insert_hydrograph_data(
            conn, "gw", col_idx=1, node_id=101, layer=2, times=times, values=vals_l2
        )

        # Node 101, layer 3
        vals_l3 = np.array([30.0, 31.0, 32.0], dtype=np.float64)
        _insert_hydrograph_data(
            conn, "gw", col_idx=2, node_id=101, layer=3, times=times, values=vals_l3
        )

        # Different node (102), should NOT be included
        vals_other = np.array([99.0, 99.0, 99.0], dtype=np.float64)
        _insert_hydrograph_data(
            conn, "gw", col_idx=3, node_id=102, layer=1, times=times, values=vals_other
        )

        conn.commit()
        conn.close()

        loader = SqliteCacheLoader(db_path)
        result = loader.get_gw_hydrograph_all_layers(101)
        assert result is not None
        assert len(result) == 3

        # Should be ordered by layer
        assert result[0][0] == 1  # layer
        assert result[1][0] == 2
        assert result[2][0] == 3

        # Verify times and values
        layer1, times1, values1 = result[0]
        assert len(times1) == 3
        np.testing.assert_allclose(values1, vals_l1)

        layer3, times3, values3 = result[2]
        np.testing.assert_allclose(values3, vals_l3)

        loader.close()

    def test_returns_none_for_missing_node(self, tmp_path: Path) -> None:
        """get_gw_hydrograph_all_layers returns None for non-existent node."""
        db_path = tmp_path / "cache.db"
        conn = _create_test_db(db_path)
        conn.commit()
        conn.close()

        loader = SqliteCacheLoader(db_path)
        result = loader.get_gw_hydrograph_all_layers(999)
        assert result is None
        loader.close()

    def test_excludes_non_gw_types(self, tmp_path: Path) -> None:
        """Only 'gw' type hydrographs are returned, not stream/subsidence."""
        db_path = tmp_path / "cache.db"
        conn = _create_test_db(db_path)
        times = ["2020-01-01"]

        # GW hydrograph at node 101
        _insert_hydrograph_data(conn, "gw", 0, 101, 1, times, np.array([10.0], dtype=np.float64))

        # Stream hydrograph at node 101 (should be excluded)
        _insert_hydrograph_data(
            conn, "stream", 0, 101, 1, times, np.array([99.0], dtype=np.float64)
        )

        conn.commit()
        conn.close()

        loader = SqliteCacheLoader(db_path)
        result = loader.get_gw_hydrograph_all_layers(101)
        assert result is not None
        assert len(result) == 1
        assert result[0][0] == 1  # layer
        np.testing.assert_allclose(result[0][2], [10.0])
        loader.close()


# ======================================================================
# get_gw_hydrograph_by_columns (lines 270-288)
# ======================================================================


class TestGetGwHydrographByColumns:
    def test_consecutive_columns_returned(self, tmp_path: Path) -> None:
        """get_gw_hydrograph_by_columns returns n_layers consecutive columns."""
        db_path = tmp_path / "cache.db"
        conn = _create_test_db(db_path)
        times = ["2020-01-01", "2020-02-01"]

        for offset in range(3):
            col_idx = 5 + offset
            vals = np.array([100.0 + offset, 200.0 + offset], dtype=np.float64)
            _insert_hydrograph_data(
                conn, "gw", col_idx, node_id=200, layer=offset + 1, times=times, values=vals
            )

        conn.commit()
        conn.close()

        loader = SqliteCacheLoader(db_path)
        result = loader.get_gw_hydrograph_by_columns(base_col=5, n_layers=3)
        assert result is not None
        assert len(result) == 3

        # Each tuple: (layer, times, values)
        assert result[0][0] == 1
        assert result[1][0] == 2
        assert result[2][0] == 3

        np.testing.assert_allclose(result[0][2], [100.0, 200.0])
        np.testing.assert_allclose(result[2][2], [102.0, 202.0])

        loader.close()

    def test_returns_none_when_no_columns_found(self, tmp_path: Path) -> None:
        """Returns None when base_col range has no data."""
        db_path = tmp_path / "cache.db"
        conn = _create_test_db(db_path)
        conn.commit()
        conn.close()

        loader = SqliteCacheLoader(db_path)
        result = loader.get_gw_hydrograph_by_columns(base_col=100, n_layers=3)
        assert result is None
        loader.close()

    def test_partial_columns_available(self, tmp_path: Path) -> None:
        """If only some columns exist, only those are returned."""
        db_path = tmp_path / "cache.db"
        conn = _create_test_db(db_path)
        times = ["2020-01-01"]

        # Only insert column 10 and 12 (gap at 11)
        _insert_hydrograph_data(conn, "gw", 10, 300, 1, times, np.array([1.0], dtype=np.float64))
        _insert_hydrograph_data(conn, "gw", 12, 300, 3, times, np.array([3.0], dtype=np.float64))

        conn.commit()
        conn.close()

        loader = SqliteCacheLoader(db_path)
        result = loader.get_gw_hydrograph_by_columns(base_col=10, n_layers=3)
        assert result is not None
        assert len(result) == 2  # Only cols 10 and 12 found
        loader.close()

    def test_layer_zero_replaced_by_offset(self, tmp_path: Path) -> None:
        """When layer is 0, it gets replaced by offset + 1."""
        db_path = tmp_path / "cache.db"
        conn = _create_test_db(db_path)
        times = ["2020-01-01"]

        # Insert with layer=0
        _insert_hydrograph_data(conn, "gw", 20, 400, 0, times, np.array([5.0], dtype=np.float64))
        _insert_hydrograph_data(conn, "gw", 21, 400, 0, times, np.array([6.0], dtype=np.float64))

        conn.commit()
        conn.close()

        loader = SqliteCacheLoader(db_path)
        result = loader.get_gw_hydrograph_by_columns(base_col=20, n_layers=2)
        assert result is not None
        assert len(result) == 2
        # layer should be offset + 1 since stored layer is 0
        assert result[0][0] == 1  # offset 0 + 1
        assert result[1][0] == 2  # offset 1 + 1
        loader.close()


# ======================================================================
# get_stream_rating (lines 299-307)
# ======================================================================


class TestGetStreamRating:
    def test_rating_table_roundtrip(self, tmp_path: Path) -> None:
        """Stream rating data is correctly stored and retrieved."""
        db_path = tmp_path / "cache.db"
        conn = _create_test_db(db_path)

        stages = np.array([0.0, 0.5, 1.0, 2.0, 5.0], dtype=np.float64)
        flows = np.array([0.0, 5.0, 25.0, 100.0, 500.0], dtype=np.float64)
        _insert_stream_rating(conn, stream_node_id=42, bottom_elev=85.3, stages=stages, flows=flows)
        conn.commit()
        conn.close()

        loader = SqliteCacheLoader(db_path)
        result = loader.get_stream_rating(42)
        assert result is not None
        bottom_elev, loaded_stages, loaded_flows = result
        assert bottom_elev == pytest.approx(85.3)
        np.testing.assert_allclose(loaded_stages, stages)
        np.testing.assert_allclose(loaded_flows, flows)
        loader.close()

    def test_returns_none_for_missing_node(self, tmp_path: Path) -> None:
        """get_stream_rating returns None for non-existent node."""
        db_path = tmp_path / "cache.db"
        conn = _create_test_db(db_path)
        conn.commit()
        conn.close()

        loader = SqliteCacheLoader(db_path)
        assert loader.get_stream_rating(999) is None
        loader.close()

    def test_multiple_ratings(self, tmp_path: Path) -> None:
        """Multiple stream nodes can have distinct rating tables."""
        db_path = tmp_path / "cache.db"
        conn = _create_test_db(db_path)

        for sn_id in [1, 2, 3]:
            stages = np.array([0.0, float(sn_id)], dtype=np.float64)
            flows = np.array([0.0, float(sn_id * 100)], dtype=np.float64)
            _insert_stream_rating(conn, sn_id, float(sn_id * 10), stages, flows)

        conn.commit()
        conn.close()

        loader = SqliteCacheLoader(db_path)
        for sn_id in [1, 2, 3]:
            result = loader.get_stream_rating(sn_id)
            assert result is not None
            bottom_elev, stages, flows = result
            assert bottom_elev == pytest.approx(sn_id * 10.0)
            assert flows[-1] == pytest.approx(sn_id * 100.0)
        loader.close()


# ======================================================================
# get_area_snapshot (lines 322-331)
# ======================================================================


class TestGetAreaSnapshot:
    def test_snapshot_roundtrip(self, tmp_path: Path) -> None:
        """Area snapshots are correctly stored and retrieved."""
        db_path = tmp_path / "cache.db"
        conn = _create_test_db(db_path)

        _insert_landuse_snapshot(conn, 0, 1, 50.0, 20.0, 30.0, 100.0, "agricultural")
        _insert_landuse_snapshot(conn, 0, 2, 10.0, 60.0, 30.0, 100.0, "urban")
        _insert_landuse_snapshot(conn, 0, 3, 20.0, 20.0, 60.0, 100.0, "native_rip")

        conn.commit()
        conn.close()

        loader = SqliteCacheLoader(db_path)
        result = loader.get_area_snapshot(0)
        assert result is not None
        assert len(result) == 3

        # Results should be ordered by element_id
        assert result[0]["element_id"] == 1
        assert result[0]["agricultural"] == pytest.approx(50.0)
        assert result[0]["dominant"] == "agricultural"

        assert result[1]["element_id"] == 2
        assert result[1]["urban"] == pytest.approx(60.0)
        assert result[1]["dominant"] == "urban"

        assert result[2]["element_id"] == 3
        assert result[2]["native_rip"] == pytest.approx(60.0)
        loader.close()

    def test_returns_none_for_missing_frame(self, tmp_path: Path) -> None:
        """get_area_snapshot returns None when frame_idx has no data."""
        db_path = tmp_path / "cache.db"
        conn = _create_test_db(db_path)
        conn.commit()
        conn.close()

        loader = SqliteCacheLoader(db_path)
        assert loader.get_area_snapshot(99) is None
        loader.close()

    def test_multiple_frames(self, tmp_path: Path) -> None:
        """Different frames have independent snapshots."""
        db_path = tmp_path / "cache.db"
        conn = _create_test_db(db_path)

        _insert_landuse_snapshot(conn, 0, 1, 50.0, 20.0, 30.0, 100.0, "agricultural")
        _insert_landuse_snapshot(conn, 5, 1, 10.0, 70.0, 20.0, 100.0, "urban")

        conn.commit()
        conn.close()

        loader = SqliteCacheLoader(db_path)
        snap0 = loader.get_area_snapshot(0)
        snap5 = loader.get_area_snapshot(5)

        assert snap0 is not None
        assert snap5 is not None
        assert snap0[0]["dominant"] == "agricultural"
        assert snap5[0]["dominant"] == "urban"
        loader.close()


# ======================================================================
# get_stats (lines 345-349, 375-376, 381-382)
# ======================================================================


class TestGetStats:
    def test_stats_with_populated_tables(self, tmp_path: Path) -> None:
        """get_stats returns correct counts for populated tables."""
        db_path = tmp_path / "cache.db"
        conn = _create_test_db(db_path)

        # Insert some head frames
        rng = np.random.default_rng(42)
        for i in range(3):
            frame = rng.random((4, 2)).astype(np.float64)
            conn.execute(
                "INSERT INTO head_frames VALUES (?, ?, ?, ?)",
                (i, 4, 2, _compress_array(frame)),
            )

        # Insert a hydrograph column + series
        _insert_hydrograph_data(
            conn,
            "gw",
            0,
            100,
            1,
            ["2020-01-01"],
            np.array([10.0], dtype=np.float64),
        )

        # Insert a stream rating
        _insert_stream_rating(
            conn,
            1,
            50.0,
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([0.0, 100.0], dtype=np.float64),
        )

        # Insert landuse snapshot
        _insert_landuse_snapshot(conn, 0, 1, 50.0, 20.0, 30.0, 100.0, "ag")
        _insert_landuse_snapshot(conn, 0, 2, 10.0, 60.0, 30.0, 100.0, "urban")

        conn.commit()
        conn.close()

        loader = SqliteCacheLoader(db_path)
        stats = loader.get_stats()

        assert stats["head_frames"] == 3
        assert stats["hydrograph_series"] == 1
        assert stats["hydrograph_columns"] == 1
        assert stats["stream_ratings"] == 1
        assert stats["landuse_snapshots"] == 2
        assert stats["file_size_mb"] > 0
        # Tables with no data should be 0
        assert stats["budget_data"] == 0
        assert stats["budget_summaries"] == 0
        loader.close()

    def test_stats_empty_db(self, tmp_path: Path) -> None:
        """get_stats returns all zeros for an empty database."""
        db_path = tmp_path / "cache.db"
        conn = _create_test_db(db_path)
        conn.commit()
        conn.close()

        loader = SqliteCacheLoader(db_path)
        stats = loader.get_stats()
        assert stats["head_frames"] == 0
        assert stats["budget_data"] == 0
        assert stats["stream_ratings"] == 0
        assert stats["landuse_snapshots"] == 0
        assert isinstance(stats["file_size_mb"], float)
        loader.close()

    def test_stats_file_size(self, tmp_path: Path) -> None:
        """file_size_mb should be a positive float for an existing db."""
        db_path = tmp_path / "cache.db"
        conn = _create_test_db(db_path)
        # Insert enough data to have non-trivial file size
        rng = np.random.default_rng(99)
        for i in range(50):
            frame = rng.random((100, 4)).astype(np.float64)
            conn.execute(
                "INSERT INTO head_frames VALUES (?, ?, ?, ?)",
                (i, 100, 4, _compress_array(frame)),
            )
        conn.commit()
        conn.close()

        loader = SqliteCacheLoader(db_path)
        stats = loader.get_stats()
        assert stats["file_size_mb"] > 0
        loader.close()


# ======================================================================
# get_area_timesteps (lines 345-349)
# ======================================================================


class TestGetAreaTimesteps:
    def test_area_timesteps_returned(self, tmp_path: Path) -> None:
        """get_area_timesteps returns stored dates in order."""
        db_path = tmp_path / "cache.db"
        conn = _create_test_db(db_path)

        _insert_area_timestep(conn, "landuse", 0, "2020-01-01")
        _insert_area_timestep(conn, "landuse", 1, "2020-02-01")
        _insert_area_timestep(conn, "landuse", 2, "2020-03-01")

        conn.commit()
        conn.close()

        loader = SqliteCacheLoader(db_path)
        result = loader.get_area_timesteps()  # default area_type="landuse"
        assert result == ["2020-01-01", "2020-02-01", "2020-03-01"]
        loader.close()

    def test_area_timesteps_empty(self, tmp_path: Path) -> None:
        """get_area_timesteps returns empty list when no data."""
        db_path = tmp_path / "cache.db"
        conn = _create_test_db(db_path)
        conn.commit()
        conn.close()

        loader = SqliteCacheLoader(db_path)
        assert loader.get_area_timesteps() == []
        loader.close()

    def test_area_timesteps_different_type(self, tmp_path: Path) -> None:
        """get_area_timesteps filters by area_type."""
        db_path = tmp_path / "cache.db"
        conn = _create_test_db(db_path)

        _insert_area_timestep(conn, "landuse", 0, "2020-01-01")
        _insert_area_timestep(conn, "other", 0, "2021-06-01")

        conn.commit()
        conn.close()

        loader = SqliteCacheLoader(db_path)
        lu = loader.get_area_timesteps("landuse")
        assert lu == ["2020-01-01"]

        other = loader.get_area_timesteps("other")
        assert other == ["2021-06-01"]
        loader.close()


# ======================================================================
# Budget loader methods
# ======================================================================


class TestBudgetLoaderMethods:
    def _setup_budget_db(self, tmp_path: Path) -> Path:
        """Create a DB with budget data for testing loader methods."""
        db_path = tmp_path / "cache.db"
        conn = _create_test_db(db_path)

        # Budget file metadata
        conn.execute(
            "INSERT INTO budget_files VALUES (?, ?, ?, ?, ?, ?)",
            ("gw", "Groundwater Budget", 3, 2, "2020-01-01", 43200),
        )

        # Locations
        conn.execute(
            "INSERT INTO budget_locations VALUES (?, ?, ?, ?)",
            ("gw", 0, "Region 1", 1000.0),
        )
        conn.execute(
            "INSERT INTO budget_locations VALUES (?, ?, ?, ?)",
            ("gw", 1, "Region 2", 2000.0),
        )

        # Columns
        conn.execute(
            "INSERT INTO budget_columns VALUES (?, ?, ?, ?)",
            ("gw", 0, "Deep Percolation", "AF"),
        )
        conn.execute(
            "INSERT INTO budget_columns VALUES (?, ?, ?, ?)",
            ("gw", 1, "Pumping", "AF"),
        )

        # Data
        for loc in range(2):
            for ts in range(3):
                row = np.array(
                    [float(loc * 100 + ts), float(loc * 100 + ts + 10)], dtype=np.float64
                )
                conn.execute(
                    "INSERT INTO budget_data VALUES (?, ?, ?, ?)",
                    ("gw", loc, ts, _compress_array(row)),
                )

        # Summaries
        conn.execute(
            "INSERT INTO budget_summaries VALUES (?, ?, ?, ?, ?)",
            ("gw", 0, 0, 300.0, 100.0),
        )
        conn.execute(
            "INSERT INTO budget_summaries VALUES (?, ?, ?, ?, ?)",
            ("gw", 0, 1, 330.0, 110.0),
        )

        conn.commit()
        conn.close()
        return db_path

    def test_budget_types(self, tmp_path: Path) -> None:
        db_path = self._setup_budget_db(tmp_path)
        loader = SqliteCacheLoader(db_path)
        assert loader.get_budget_types() == ["gw"]
        loader.close()

    def test_budget_locations(self, tmp_path: Path) -> None:
        db_path = self._setup_budget_db(tmp_path)
        loader = SqliteCacheLoader(db_path)
        locs = loader.get_budget_locations("gw")
        assert len(locs) == 2
        assert locs[0] == (0, "Region 1", 1000.0)
        assert locs[1] == (1, "Region 2", 2000.0)
        loader.close()

    def test_budget_columns(self, tmp_path: Path) -> None:
        db_path = self._setup_budget_db(tmp_path)
        loader = SqliteCacheLoader(db_path)
        cols = loader.get_budget_columns("gw")
        assert len(cols) == 2
        assert cols[0] == (0, "Deep Percolation", "AF")
        loader.close()

    def test_budget_data_shape(self, tmp_path: Path) -> None:
        db_path = self._setup_budget_db(tmp_path)
        loader = SqliteCacheLoader(db_path)
        data = loader.get_budget_data("gw", 0)
        assert data is not None
        assert data.shape == (3, 2)  # 3 timesteps, 2 columns
        loader.close()

    def test_budget_summary(self, tmp_path: Path) -> None:
        db_path = self._setup_budget_db(tmp_path)
        loader = SqliteCacheLoader(db_path)
        summaries = loader.get_budget_summary("gw", 0)
        assert len(summaries) == 2
        assert summaries[0] == (0, 300.0, 100.0)
        assert summaries[1] == (1, 330.0, 110.0)
        loader.close()


# ======================================================================
# Edge cases for loader
# ======================================================================


class TestLoaderEdgeCases:
    def test_connection_is_per_thread(self, tmp_path: Path) -> None:
        """Verify the loader creates a connection on first use."""
        db_path = tmp_path / "cache.db"
        conn = _create_test_db(db_path)
        conn.commit()
        conn.close()

        loader = SqliteCacheLoader(db_path)
        # First call triggers connection creation
        assert loader.get_metadata("schema_version") == SCHEMA_VERSION
        # Second call reuses the existing connection
        assert loader.get_metadata("model_name") == "test_model"
        loader.close()

    def test_head_frame_returns_none_for_missing(self, tmp_path: Path) -> None:
        db_path = tmp_path / "cache.db"
        conn = _create_test_db(db_path)
        conn.commit()
        conn.close()

        loader = SqliteCacheLoader(db_path)
        assert loader.get_head_frame(0) is None
        loader.close()

    def test_head_by_element_returns_none_for_missing(self, tmp_path: Path) -> None:
        db_path = tmp_path / "cache.db"
        conn = _create_test_db(db_path)
        conn.commit()
        conn.close()

        loader = SqliteCacheLoader(db_path)
        assert loader.get_head_by_element(0, 1) is None
        loader.close()

    def test_head_range_returns_none_for_missing(self, tmp_path: Path) -> None:
        db_path = tmp_path / "cache.db"
        conn = _create_test_db(db_path)
        conn.commit()
        conn.close()

        loader = SqliteCacheLoader(db_path)
        assert loader.get_head_range(1) is None
        loader.close()

    def test_hydrograph_returns_none_for_missing(self, tmp_path: Path) -> None:
        db_path = tmp_path / "cache.db"
        conn = _create_test_db(db_path)
        conn.commit()
        conn.close()

        loader = SqliteCacheLoader(db_path)
        assert loader.get_hydrograph("gw", 0) is None
        loader.close()
