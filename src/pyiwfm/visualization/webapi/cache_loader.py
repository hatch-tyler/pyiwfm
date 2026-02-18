"""
SQLite cache loader for the IWFM web viewer.

Provides fast read access to pre-computed data stored by
:class:`~cache_builder.SqliteCacheBuilder`.  Uses WAL mode for
concurrent reads and connection pooling for thread safety.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import zlib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def _decompress_array(blob: bytes, dtype: type = np.float64) -> NDArray:  # type: ignore[type-arg]
    """Decompress a zlib-compressed numpy array."""
    raw = zlib.decompress(blob)
    arr: NDArray = np.frombuffer(raw, dtype=dtype).copy()  # type: ignore[type-arg]
    return arr


def _decompress_strings(blob: bytes) -> list[str]:
    """Decompress a zlib-compressed newline-separated string list."""
    raw = zlib.decompress(blob)
    return raw.decode("utf-8").split("\n")


class SqliteCacheLoader:
    """Read-only access to the viewer SQLite cache.

    Thread-safe via per-thread connections with WAL mode.

    Parameters
    ----------
    cache_path : Path
        Path to the SQLite cache file.
    """

    def __init__(self, cache_path: Path) -> None:
        self.cache_path = cache_path
        self._local = threading.local()
        # Validate file exists
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache not found: {cache_path}")
        logger.info("SQLite cache loader opened: %s", cache_path)

    def _conn(self) -> sqlite3.Connection:
        """Get the per-thread connection, creating if needed."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(
                str(self.cache_path), check_same_thread=False
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA mmap_size=268435456")  # 256 MB
            conn.execute("PRAGMA cache_size=-65536")  # 64 MB
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA query_only=ON")
            self._local.conn = conn
        return conn

    def close(self) -> None:
        """Close the per-thread connection if open."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_metadata(self, key: str) -> str | None:
        """Get a metadata value by key."""
        cur = self._conn().execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        )
        row = cur.fetchone()
        return row[0] if row else None

    def get_timesteps(self) -> list[str]:
        """Get all timestep datetimes as ISO strings."""
        cur = self._conn().execute(
            "SELECT datetime FROM timesteps ORDER BY idx"
        )
        return [r[0] for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # Head data
    # ------------------------------------------------------------------

    def get_head_frame(self, frame_idx: int) -> NDArray | None:
        """Get a raw head frame (n_nodes, n_layers).

        Returns None if frame not cached.
        """
        cur = self._conn().execute(
            "SELECT n_nodes, n_layers, data_blob FROM head_frames "
            "WHERE frame_idx = ?",
            (frame_idx,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        n_nodes, n_layers, blob = row
        arr = _decompress_array(blob)
        return arr.reshape(n_nodes, n_layers)

    def get_head_by_element(
        self, frame_idx: int, layer: int
    ) -> tuple[NDArray, float, float] | None:
        """Get pre-computed element-averaged heads.

        Returns (values_array, min_val, max_val) or None.
        """
        cur = self._conn().execute(
            "SELECT values_blob, min_val, max_val FROM head_by_element "
            "WHERE frame_idx = ? AND layer = ?",
            (frame_idx, layer),
        )
        row = cur.fetchone()
        if row is None:
            return None
        blob, min_val, max_val = row
        return _decompress_array(blob), float(min_val), float(max_val)

    def get_head_range(self, layer: int) -> dict | None:
        """Get the pre-computed head range for a layer.

        Returns dict with percentile_02, percentile_98, abs_min, abs_max.
        """
        cur = self._conn().execute(
            "SELECT percentile_02, percentile_98, abs_min, abs_max "
            "FROM head_range WHERE layer = ?",
            (layer,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return {
            "percentile_02": row[0],
            "percentile_98": row[1],
            "abs_min": row[2],
            "abs_max": row[3],
        }

    def get_n_head_frames(self) -> int:
        """Get the number of cached head frames."""
        cur = self._conn().execute("SELECT COUNT(*) FROM head_frames")
        row = cur.fetchone()
        return row[0] if row else 0

    # ------------------------------------------------------------------
    # Budget data
    # ------------------------------------------------------------------

    def get_budget_types(self) -> list[str]:
        """Get list of cached budget types."""
        cur = self._conn().execute(
            "SELECT budget_type FROM budget_files ORDER BY budget_type"
        )
        return [r[0] for r in cur.fetchall()]

    def get_budget_locations(self, budget_type: str) -> list[tuple[int, str, float]]:
        """Get locations for a budget type as (idx, name, area) tuples."""
        cur = self._conn().execute(
            "SELECT location_idx, location_name, area "
            "FROM budget_locations WHERE budget_type = ? ORDER BY location_idx",
            (budget_type,),
        )
        return [(r[0], r[1], r[2]) for r in cur.fetchall()]

    def get_budget_columns(self, budget_type: str) -> list[tuple[int, str, str]]:
        """Get columns for a budget type as (idx, name, units) tuples."""
        cur = self._conn().execute(
            "SELECT column_idx, column_name, units "
            "FROM budget_columns WHERE budget_type = ? ORDER BY column_idx",
            (budget_type,),
        )
        return [(r[0], r[1], r[2]) for r in cur.fetchall()]

    def get_budget_data(
        self, budget_type: str, location_idx: int
    ) -> NDArray | None:
        """Get budget data for a location as (n_timesteps, n_cols) array."""
        cur = self._conn().execute(
            "SELECT values_blob FROM budget_data "
            "WHERE budget_type = ? AND location_idx = ? ORDER BY timestep_idx",
            (budget_type, location_idx),
        )
        rows = cur.fetchall()
        if not rows:
            return None
        arrays = [_decompress_array(r[0]) for r in rows]
        return np.vstack(arrays)

    def get_budget_summary(
        self, budget_type: str, location_idx: int
    ) -> list[tuple[int, float, float]]:
        """Get budget summaries as (col_idx, total, average) tuples."""
        cur = self._conn().execute(
            "SELECT column_idx, total, average FROM budget_summaries "
            "WHERE budget_type = ? AND location_idx = ? ORDER BY column_idx",
            (budget_type, location_idx),
        )
        return [(r[0], r[1], r[2]) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # Hydrographs
    # ------------------------------------------------------------------

    def get_hydrograph(
        self, hydro_type: str, column_idx: int
    ) -> tuple[list[str], NDArray] | None:
        """Get a hydrograph timeseries.

        Returns (times_list, values_array) or None.
        """
        cur = self._conn().execute(
            "SELECT times_blob, values_blob FROM hydrograph_series "
            "WHERE hydro_type = ? AND column_idx = ?",
            (hydro_type, column_idx),
        )
        row = cur.fetchone()
        if row is None:
            return None
        times = _decompress_strings(row[0])
        values = _decompress_array(row[1])
        return times, values

    def get_hydrograph_columns(
        self, hydro_type: str
    ) -> list[tuple[int, int, int]]:
        """Get hydrograph column metadata as (col_idx, node_id, layer) tuples."""
        cur = self._conn().execute(
            "SELECT column_idx, node_id, layer FROM hydrograph_columns "
            "WHERE hydro_type = ? ORDER BY column_idx",
            (hydro_type,),
        )
        return [(r[0], r[1], r[2]) for r in cur.fetchall()]

    def get_gw_hydrograph_all_layers(
        self, node_id: int
    ) -> list[tuple[int, list[str], NDArray]] | None:
        """Get all GW hydrograph layers for a node.

        Returns list of (layer, times, values) tuples, or None.
        """
        cur = self._conn().execute(
            "SELECT hc.layer, hs.times_blob, hs.values_blob "
            "FROM hydrograph_columns hc "
            "JOIN hydrograph_series hs "
            "  ON hc.hydro_type = hs.hydro_type AND hc.column_idx = hs.column_idx "
            "WHERE hc.hydro_type = 'gw' AND hc.node_id = ? "
            "ORDER BY hc.layer",
            (node_id,),
        )
        rows = cur.fetchall()
        if not rows:
            return None
        result: list[tuple[int, list[str], NDArray]] = []
        for layer, times_blob, values_blob in rows:
            times = _decompress_strings(times_blob)
            values = _decompress_array(values_blob)
            result.append((layer, times, values))
        return result

    def get_gw_hydrograph_by_columns(
        self, base_col: int, n_layers: int
    ) -> list[tuple[int, list[str], NDArray]] | None:
        """Get GW hydrograph layers by consecutive column range.

        IWFM GW hydrograph output files store n_layers consecutive columns
        per location: [loc_lay1, loc_lay2, ..., loc_layN].

        Returns list of (layer, times, values) tuples, or None.
        """
        result: list[tuple[int, list[str], NDArray]] = []
        for offset in range(n_layers):
            col = base_col + offset
            cur = self._conn().execute(
                "SELECT hc.layer, hs.times_blob, hs.values_blob "
                "FROM hydrograph_columns hc "
                "JOIN hydrograph_series hs "
                "  ON hc.hydro_type = hs.hydro_type AND hc.column_idx = hs.column_idx "
                "WHERE hc.hydro_type = 'gw' AND hc.column_idx = ?",
                (col,),
            )
            row = cur.fetchone()
            if row is None:
                continue
            layer, times_blob, values_blob = row
            times = _decompress_strings(times_blob)
            values = _decompress_array(values_blob)
            result.append((layer if layer > 0 else offset + 1, times, values))
        return result if result else None

    # ------------------------------------------------------------------
    # Stream ratings
    # ------------------------------------------------------------------

    def get_stream_rating(
        self, stream_node_id: int
    ) -> tuple[float, NDArray, NDArray] | None:
        """Get a stream node's rating table.

        Returns (bottom_elev, stages, flows) or None.
        """
        cur = self._conn().execute(
            "SELECT bottom_elev, stages_blob, flows_blob "
            "FROM stream_ratings WHERE stream_node_id = ?",
            (stream_node_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return (
            float(row[0]),
            _decompress_array(row[1]),
            _decompress_array(row[2]),
        )

    # ------------------------------------------------------------------
    # Area / land-use
    # ------------------------------------------------------------------

    def get_area_snapshot(
        self, frame_idx: int
    ) -> list[dict] | None:
        """Get a land-use snapshot for a timestep.

        Returns list of element dicts or None.
        """
        cur = self._conn().execute(
            "SELECT element_id, agricultural, urban, native_rip, "
            "total_area, dominant "
            "FROM landuse_snapshots WHERE frame_idx = ? ORDER BY element_id",
            (frame_idx,),
        )
        rows = cur.fetchall()
        if not rows:
            return None
        return [
            {
                "element_id": r[0],
                "agricultural": r[1],
                "urban": r[2],
                "native_rip": r[3],
                "total_area": r[4],
                "dominant": r[5],
            }
            for r in rows
        ]

    def get_area_timesteps(self, area_type: str = "landuse") -> list[str]:
        """Get area timestep dates."""
        cur = self._conn().execute(
            "SELECT datetime FROM area_timesteps "
            "WHERE area_type = ? ORDER BY idx",
            (area_type,),
        )
        return [r[0] for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Get cache statistics for diagnostics."""
        conn = self._conn()
        stats: dict = {}

        for table in (
            "head_frames", "head_by_element", "head_range",
            "budget_files", "budget_data", "budget_summaries",
            "hydrograph_series", "hydrograph_columns",
            "stream_ratings", "landuse_snapshots",
        ):
            try:
                cur = conn.execute(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
                stats[table] = cur.fetchone()[0]
            except Exception:
                stats[table] = 0

        # File size
        try:
            stats["file_size_mb"] = round(
                self.cache_path.stat().st_size / (1024 * 1024), 1
            )
        except Exception:
            stats["file_size_mb"] = 0

        return stats
