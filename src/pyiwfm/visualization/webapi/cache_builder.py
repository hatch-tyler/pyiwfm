"""
SQLite cache builder for the IWFM web viewer.

Reads from existing HDF5/text loaders and pre-computes aggregates into a
single SQLite database for fast serving.  The cache provides:

- Pre-computed element-averaged head values (by-element)
- Pre-computed head ranges per layer (2nd/98th percentile)
- Pre-computed budget summaries (total/average per column)
- Pre-aggregated zone budget totals
- Unified hydrograph timeseries (GW/stream/subsidence from output files)
- Stream rating tables
- Land-use area snapshots with dominant-type classification
"""

from __future__ import annotations

import logging
import sqlite3
import struct
import time
import zlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pyiwfm.core.model import IWFMModel
    from pyiwfm.visualization.webapi.head_loader import LazyHeadDataLoader

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1"

# --------------------------------------------------------------------------
# Schema DDL
# --------------------------------------------------------------------------

_SCHEMA_SQL = """\
-- Metadata
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS timesteps (
    idx INTEGER PRIMARY KEY,
    datetime TEXT NOT NULL
);

-- Head data (compressed node-level frames)
CREATE TABLE IF NOT EXISTS head_frames (
    frame_idx INTEGER PRIMARY KEY,
    n_nodes INTEGER NOT NULL,
    n_layers INTEGER NOT NULL,
    data_blob BLOB NOT NULL
);

-- Head by element (pre-averaged per element)
CREATE TABLE IF NOT EXISTS head_by_element (
    frame_idx INTEGER NOT NULL,
    layer INTEGER NOT NULL,
    values_blob BLOB NOT NULL,
    min_val REAL,
    max_val REAL,
    PRIMARY KEY (frame_idx, layer)
);

-- Head range (robust percentile + absolute extremes per layer)
CREATE TABLE IF NOT EXISTS head_range (
    layer INTEGER PRIMARY KEY,
    percentile_02 REAL,
    percentile_98 REAL,
    abs_min REAL,
    abs_max REAL
);

-- Budget files
CREATE TABLE IF NOT EXISTS budget_files (
    budget_type TEXT PRIMARY KEY,
    descriptor TEXT,
    n_timesteps INTEGER,
    n_locations INTEGER,
    start_datetime TEXT,
    delta_t_minutes INTEGER
);

CREATE TABLE IF NOT EXISTS budget_locations (
    budget_type TEXT,
    location_idx INTEGER,
    location_name TEXT,
    area REAL,
    PRIMARY KEY (budget_type, location_idx)
);

CREATE TABLE IF NOT EXISTS budget_columns (
    budget_type TEXT,
    column_idx INTEGER,
    column_name TEXT,
    units TEXT,
    PRIMARY KEY (budget_type, column_idx)
);

CREATE TABLE IF NOT EXISTS budget_data (
    budget_type TEXT,
    location_idx INTEGER,
    timestep_idx INTEGER,
    values_blob BLOB NOT NULL,
    PRIMARY KEY (budget_type, location_idx, timestep_idx)
);

CREATE TABLE IF NOT EXISTS budget_summaries (
    budget_type TEXT,
    location_idx INTEGER,
    column_idx INTEGER,
    total REAL,
    average REAL,
    PRIMARY KEY (budget_type, location_idx, column_idx)
);

-- Zone budget (pre-aggregated)
CREATE TABLE IF NOT EXISTS zones (
    zone_id INTEGER PRIMARY KEY,
    zone_name TEXT,
    area REAL,
    element_ids BLOB
);

CREATE TABLE IF NOT EXISTS zbudget_zone_totals (
    data_path TEXT,
    layer INTEGER,
    zone_id INTEGER,
    values_blob BLOB NOT NULL,
    PRIMARY KEY (data_path, layer, zone_id)
);

-- Area / land-use
CREATE TABLE IF NOT EXISTS area_frames (
    area_type TEXT,
    frame_idx INTEGER,
    n_elements INTEGER,
    n_cols INTEGER,
    data_blob BLOB NOT NULL,
    PRIMARY KEY (area_type, frame_idx)
);

CREATE TABLE IF NOT EXISTS area_timesteps (
    area_type TEXT,
    idx INTEGER,
    datetime TEXT,
    PRIMARY KEY (area_type, idx)
);

CREATE TABLE IF NOT EXISTS landuse_snapshots (
    frame_idx INTEGER,
    element_id INTEGER,
    agricultural REAL,
    urban REAL,
    native_rip REAL,
    total_area REAL,
    dominant TEXT,
    PRIMARY KEY (frame_idx, element_id)
);

-- Hydrographs (unified)
CREATE TABLE IF NOT EXISTS hydrograph_columns (
    hydro_type TEXT,
    column_idx INTEGER,
    node_id INTEGER,
    layer INTEGER,
    PRIMARY KEY (hydro_type, column_idx)
);

CREATE TABLE IF NOT EXISTS hydrograph_series (
    hydro_type TEXT,
    column_idx INTEGER,
    times_blob BLOB NOT NULL,
    values_blob BLOB NOT NULL,
    PRIMARY KEY (hydro_type, column_idx)
);

-- Stream rating tables
CREATE TABLE IF NOT EXISTS stream_ratings (
    stream_node_id INTEGER PRIMARY KEY,
    bottom_elev REAL NOT NULL,
    stages_blob BLOB NOT NULL,
    flows_blob BLOB NOT NULL
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_budget_data_loc
    ON budget_data(budget_type, location_idx);
CREATE INDEX IF NOT EXISTS idx_landuse_elem
    ON landuse_snapshots(element_id, frame_idx);
CREATE INDEX IF NOT EXISTS idx_hydro_node
    ON hydrograph_columns(node_id);
"""


def _compress_array(arr: np.ndarray) -> bytes:
    """Compress a numpy array to zlib-compressed bytes."""
    return zlib.compress(arr.astype(np.float64).tobytes(), level=1)


def _compress_int_array(arr: list[int] | np.ndarray) -> bytes:
    """Compress a list of ints to zlib bytes."""
    return zlib.compress(struct.pack(f"<{len(arr)}i", *arr), level=1)


class SqliteCacheBuilder:
    """Builds the SQLite viewer cache from model data and loaders.

    Usage::

        builder = SqliteCacheBuilder(cache_path)
        builder.build(model, head_loader, budget_readers, ...)
    """

    def __init__(self, cache_path: Path) -> None:
        self.cache_path = cache_path

    def build(
        self,
        model: IWFMModel,
        head_loader: LazyHeadDataLoader | None = None,
        budget_readers: dict[str, Any] | None = None,
        area_manager: Any | None = None,
        gw_hydrograph_reader: Any | None = None,
        stream_hydrograph_reader: Any | None = None,
        subsidence_reader: Any | None = None,
        progress_callback: Any | None = None,
    ) -> None:
        """Build the complete cache.

        Parameters
        ----------
        model : IWFMModel
            The loaded model.
        head_loader : LazyHeadDataLoader, optional
            Head data loader (HDF5-backed).
        budget_readers : dict, optional
            Mapping of budget_type -> BudgetReader.
        area_manager : AreaDataManager, optional
            Land-use area data manager.
        gw_hydrograph_reader : IWFMHydrographReader, optional
        stream_hydrograph_reader : IWFMHydrographReader, optional
        subsidence_reader : IWFMHydrographReader, optional
        progress_callback : callable, optional
            Called with (step_name, pct) for progress reporting.
        """
        t0 = time.monotonic()
        logger.info("Building SQLite cache at %s", self.cache_path)

        # Delete existing cache file if present
        if self.cache_path.exists():
            self.cache_path.unlink()

        conn = sqlite3.connect(str(self.cache_path))
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-65536")  # 64 MB
            conn.executescript(_SCHEMA_SQL)

            # Metadata
            conn.execute(
                "INSERT INTO metadata VALUES (?, ?)",
                ("schema_version", SCHEMA_VERSION),
            )
            conn.execute(
                "INSERT INTO metadata VALUES (?, ?)",
                ("build_time", time.strftime("%Y-%m-%dT%H:%M:%S")),
            )
            conn.execute(
                "INSERT INTO metadata VALUES (?, ?)",
                ("model_name", model.metadata.get("name", "unknown")),
            )
            conn.commit()

            self._build_head_data(conn, model, head_loader, progress_callback)
            self._build_budget_data(conn, budget_readers, progress_callback)
            self._build_hydrographs(
                conn, model, head_loader,
                gw_hydrograph_reader, stream_hydrograph_reader,
                subsidence_reader, progress_callback,
            )
            self._build_stream_ratings(conn, model, progress_callback)
            self._build_area_data(conn, area_manager, progress_callback)

            conn.commit()
        finally:
            conn.close()

        elapsed = time.monotonic() - t0
        logger.info("SQLite cache built in %.1f s: %s", elapsed, self.cache_path)

    # ------------------------------------------------------------------
    # Head data
    # ------------------------------------------------------------------

    def _build_head_data(
        self,
        conn: sqlite3.Connection,
        model: IWFMModel,
        head_loader: LazyHeadDataLoader | None,
        progress_callback: Any | None,
    ) -> None:
        if head_loader is None or head_loader.n_frames == 0:
            return

        logger.info("Caching %d head frames...", head_loader.n_frames)

        grid = model.grid
        if grid is None:
            return

        n_frames = head_loader.n_frames
        n_layers = head_loader.get_frame(0).shape[1]

        # Store timesteps
        for idx, t in enumerate(head_loader.times):
            conn.execute(
                "INSERT OR REPLACE INTO timesteps VALUES (?, ?)",
                (idx, t.isoformat()),
            )

        # Build element-to-node mapping for element-averaged heads
        elem_node_map: list[tuple[int, list[int]]] = []
        sorted_node_ids = sorted(grid.nodes.keys())
        node_id_to_idx = {nid: i for i, nid in enumerate(sorted_node_ids)}

        for eid in sorted(grid.elements.keys()):
            elem = grid.elements[eid]
            node_indices = [
                node_id_to_idx[vid]
                for vid in elem.vertices
                if vid in node_id_to_idx
            ]
            if node_indices:
                elem_node_map.append((eid, node_indices))

        n_elements = len(elem_node_map)

        # Per-layer accumulators for range computation
        layer_all_mins: list[float] = [float("inf")] * n_layers
        layer_all_maxs: list[float] = [float("-inf")] * n_layers
        # Reservoir-sample values for percentile estimation
        sample_size = min(50, n_frames)
        sample_indices = set(
            np.linspace(0, n_frames - 1, sample_size, dtype=int).tolist()
        )
        layer_samples: list[list[float]] = [[] for _ in range(n_layers)]

        for ts in range(n_frames):
            frame = head_loader.get_frame(ts)  # shape (n_nodes, n_layers)

            # Store compressed raw frame
            conn.execute(
                "INSERT INTO head_frames VALUES (?, ?, ?, ?)",
                (ts, frame.shape[0], frame.shape[1], _compress_array(frame)),
            )

            # Compute per-element averages for each layer
            for layer_idx in range(n_layers):
                layer_vals = frame[:, layer_idx]
                elem_avgs = np.full(n_elements, np.nan, dtype=np.float64)

                for i, (_eid, node_indices) in enumerate(elem_node_map):
                    node_vals = layer_vals[node_indices]
                    valid = node_vals[node_vals > -9000]
                    if len(valid) > 0:
                        elem_avgs[i] = float(np.mean(valid))

                valid_avgs = elem_avgs[~np.isnan(elem_avgs)]
                min_val = float(np.min(valid_avgs)) if len(valid_avgs) > 0 else 0.0
                max_val = float(np.max(valid_avgs)) if len(valid_avgs) > 0 else 0.0

                conn.execute(
                    "INSERT INTO head_by_element VALUES (?, ?, ?, ?, ?)",
                    (ts, layer_idx + 1, _compress_array(elem_avgs), min_val, max_val),
                )

                # Track range
                if len(valid_avgs) > 0:
                    layer_all_mins[layer_idx] = min(
                        layer_all_mins[layer_idx], min_val
                    )
                    layer_all_maxs[layer_idx] = max(
                        layer_all_maxs[layer_idx], max_val
                    )
                    if ts in sample_indices:
                        layer_samples[layer_idx].extend(valid_avgs.tolist())

            if progress_callback and ts % 20 == 0:
                pct = int(ts / n_frames * 40)
                progress_callback("head_frames", pct)

        # Compute and store head ranges
        for layer_idx in range(n_layers):
            samples = np.array(layer_samples[layer_idx])
            p02 = float(np.percentile(samples, 2)) if len(samples) > 0 else 0.0
            p98 = float(np.percentile(samples, 98)) if len(samples) > 0 else 0.0
            conn.execute(
                "INSERT INTO head_range VALUES (?, ?, ?, ?, ?)",
                (
                    layer_idx + 1,
                    round(p02, 3),
                    round(p98, 3),
                    round(layer_all_mins[layer_idx], 3),
                    round(layer_all_maxs[layer_idx], 3),
                ),
            )

        conn.commit()
        logger.info(
            "Head data cached: %d frames, %d layers, %d elements",
            n_frames, n_layers, n_elements,
        )

    # ------------------------------------------------------------------
    # Budget data
    # ------------------------------------------------------------------

    def _build_budget_data(
        self,
        conn: sqlite3.Connection,
        budget_readers: dict[str, Any] | None,
        progress_callback: Any | None,
    ) -> None:
        if not budget_readers:
            return

        for btype, reader in budget_readers.items():
            try:
                self._cache_single_budget(conn, btype, reader)
            except Exception as e:
                logger.warning("Failed to cache budget %s: %s", btype, e)

        conn.commit()

    def _cache_single_budget(
        self,
        conn: sqlite3.Connection,
        btype: str,
        reader: Any,
    ) -> None:
        n_locs = getattr(reader, "n_locations", 0)
        n_ts = getattr(reader, "n_timesteps", 0)
        if n_locs == 0 or n_ts == 0:
            return

        logger.info("Caching budget %s: %d locs, %d timesteps", btype, n_locs, n_ts)

        # Budget file metadata
        conn.execute(
            "INSERT OR REPLACE INTO budget_files VALUES (?, ?, ?, ?, ?, ?)",
            (btype, getattr(reader, "descriptor", ""), n_ts, n_locs, None, None),
        )

        # Locations
        loc_names = getattr(reader, "location_names", [])
        loc_areas = getattr(reader, "location_areas", [])
        for i in range(n_locs):
            name = loc_names[i] if i < len(loc_names) else f"Location {i + 1}"
            area = loc_areas[i] if i < len(loc_areas) else 0.0
            conn.execute(
                "INSERT OR REPLACE INTO budget_locations VALUES (?, ?, ?, ?)",
                (btype, i, name, area),
            )

        # Columns
        col_names = getattr(reader, "column_names", [])
        col_units = getattr(reader, "column_units", [])
        n_cols = len(col_names)
        for i in range(n_cols):
            conn.execute(
                "INSERT OR REPLACE INTO budget_columns VALUES (?, ?, ?, ?)",
                (
                    btype,
                    i,
                    col_names[i] if i < len(col_names) else f"Col {i + 1}",
                    col_units[i] if i < len(col_units) else "",
                ),
            )

        # Data + summaries
        for loc_idx in range(n_locs):
            try:
                data = reader.get_values(loc_idx)  # shape (n_ts, n_cols)
                if data is None:
                    continue

                data = np.asarray(data, dtype=np.float64)
                if data.ndim == 1:
                    data = data.reshape(-1, 1)

                # Store per-timestep blobs
                for ts_idx in range(min(data.shape[0], n_ts)):
                    row = data[ts_idx, :]
                    conn.execute(
                        "INSERT OR REPLACE INTO budget_data VALUES (?, ?, ?, ?)",
                        (btype, loc_idx, ts_idx, _compress_array(row)),
                    )

                # Summaries
                for col_idx in range(data.shape[1]):
                    col_data = data[:, col_idx]
                    valid = col_data[np.isfinite(col_data)]
                    total = float(np.sum(valid)) if len(valid) > 0 else 0.0
                    avg = float(np.mean(valid)) if len(valid) > 0 else 0.0
                    conn.execute(
                        "INSERT OR REPLACE INTO budget_summaries "
                        "VALUES (?, ?, ?, ?, ?)",
                        (btype, loc_idx, col_idx, total, avg),
                    )

            except Exception as e:
                logger.warning(
                    "Failed to cache budget data for %s loc %d: %s",
                    btype, loc_idx, e,
                )

    # ------------------------------------------------------------------
    # Hydrographs (GW/stream/subsidence from output files)
    # ------------------------------------------------------------------

    def _build_hydrographs(
        self,
        conn: sqlite3.Connection,
        model: IWFMModel,
        head_loader: LazyHeadDataLoader | None,
        gw_reader: Any | None,
        stream_reader: Any | None,
        subsidence_reader: Any | None,
        progress_callback: Any | None,
    ) -> None:
        # GW hydrographs: read from IWFM hydrograph output file (preferred).
        # This file contains IWFM-computed heads at actual observation
        # coordinates (FE-interpolated for element-based obs).
        # Column layout: n_locations × n_layers consecutive columns.
        if gw_reader and gw_reader.n_timesteps > 0:
            self._cache_hydrograph_with_metadata(
                conn, "gw", gw_reader,
            )

        # Stream hydrographs
        if stream_reader and stream_reader.n_timesteps > 0:
            self._cache_hydrograph_with_metadata(
                conn, "stream", stream_reader,
            )

        # Subsidence hydrographs
        if subsidence_reader and subsidence_reader.n_timesteps > 0:
            self._cache_hydrograph_with_metadata(
                conn, "subsidence", subsidence_reader,
            )

        conn.commit()

    def _cache_hydrograph_with_metadata(
        self,
        conn: sqlite3.Connection,
        hydro_type: str,
        reader: Any,
    ) -> None:
        """Cache a hydrograph reader, preserving layer and node metadata.

        Optimized for bulk access: reads HDF5 data in chunks and compresses
        the shared times blob only once.
        """
        layers = getattr(reader, "layers", [])
        node_ids = getattr(reader, "node_ids", [])
        n_cols = reader.n_columns

        if n_cols == 0:
            return

        # Compress times once — they're identical for every column.
        times = getattr(reader, "times", None) or []
        times_blob = zlib.compress(
            "\n".join(str(t) for t in times).encode("utf-8"), level=1
        )

        # Try bulk HDF5 read path (LazyHydrographDataLoader has _file_path).
        hdf_path = getattr(reader, "_file_path", None)
        CHUNK = 500  # columns per chunk

        if hdf_path is not None and Path(str(hdf_path)).exists():
            import h5py

            logger.info(
                "Bulk-caching %d %s hydrograph columns from HDF5...",
                n_cols, hydro_type,
            )
            with h5py.File(str(hdf_path), "r") as f:
                ds = f["data"]  # shape (n_timesteps, n_columns)
                for start in range(0, n_cols, CHUNK):
                    end = min(start + CHUNK, n_cols)
                    # Read chunk of columns at once — one HDF5 I/O operation.
                    chunk_data = ds[:, start:end].astype(np.float64)
                    for offset in range(end - start):
                        col_idx = start + offset
                        vals = chunk_data[:, offset].copy()
                        layer = layers[col_idx] if col_idx < len(layers) else 0
                        node_id = (
                            node_ids[col_idx] if col_idx < len(node_ids) else 0
                        )
                        conn.execute(
                            "INSERT OR REPLACE INTO hydrograph_columns "
                            "VALUES (?, ?, ?, ?)",
                            (hydro_type, col_idx, node_id, layer),
                        )
                        conn.execute(
                            "INSERT OR REPLACE INTO hydrograph_series "
                            "VALUES (?, ?, ?, ?)",
                            (hydro_type, col_idx, times_blob,
                             _compress_array(vals)),
                        )
                    # Commit each chunk to avoid unbounded WAL growth.
                    conn.commit()
                    if start % 5000 == 0 and start > 0:
                        logger.info(
                            "  %s hydrograph progress: %d/%d columns",
                            hydro_type, start, n_cols,
                        )
        else:
            # Fallback: per-column via reader.get_time_series().
            for col_idx in range(n_cols):
                try:
                    _times, values = reader.get_time_series(col_idx)
                    vals_arr = np.array(values, dtype=np.float64)
                    layer = layers[col_idx] if col_idx < len(layers) else 0
                    node_id = (
                        node_ids[col_idx] if col_idx < len(node_ids) else 0
                    )
                    conn.execute(
                        "INSERT OR REPLACE INTO hydrograph_columns "
                        "VALUES (?, ?, ?, ?)",
                        (hydro_type, col_idx, node_id, layer),
                    )
                    conn.execute(
                        "INSERT OR REPLACE INTO hydrograph_series "
                        "VALUES (?, ?, ?, ?)",
                        (hydro_type, col_idx, times_blob,
                         _compress_array(vals_arr)),
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to cache %s hydrograph col %d: %s",
                        hydro_type, col_idx, e,
                    )
            conn.commit()

        logger.info(
            "Cached %d %s hydrograph columns", n_cols, hydro_type,
        )

    # ------------------------------------------------------------------
    # Stream rating tables
    # ------------------------------------------------------------------

    def _build_stream_ratings(
        self,
        conn: sqlite3.Connection,
        model: IWFMModel,
        progress_callback: Any | None,
    ) -> None:
        if model.streams is None or not hasattr(model.streams, "nodes"):
            return

        count = 0
        for sn in model.streams.nodes.values():
            rating = getattr(sn, "rating", None)
            if rating is None:
                continue
            bottom_elev = getattr(sn, "bottom_elev", 0.0)
            conn.execute(
                "INSERT OR REPLACE INTO stream_ratings VALUES (?, ?, ?, ?)",
                (
                    sn.id,
                    float(bottom_elev),
                    _compress_array(np.asarray(rating.stages, dtype=np.float64)),
                    _compress_array(np.asarray(rating.flows, dtype=np.float64)),
                ),
            )
            count += 1

        if count > 0:
            conn.commit()
            logger.info("Cached %d stream rating tables", count)

    # ------------------------------------------------------------------
    # Area / land-use data
    # ------------------------------------------------------------------

    def _build_area_data(
        self,
        conn: sqlite3.Connection,
        area_manager: Any | None,
        progress_callback: Any | None,
    ) -> None:
        if area_manager is None:
            return

        n_ts = getattr(area_manager, "n_timesteps", 0)
        if n_ts == 0:
            return

        logger.info("Caching %d area timesteps...", n_ts)

        # Store timestep dates
        dates = getattr(area_manager, "dates", [])
        for i, d in enumerate(dates):
            conn.execute(
                "INSERT OR REPLACE INTO area_timesteps VALUES (?, ?, ?)",
                ("landuse", i, str(d)),
            )

        # Cache land-use snapshots for a subset of timesteps
        # (every 12th for monthly = annual snapshots, or all if < 24)
        step = max(1, n_ts // 24)
        for ts_idx in range(0, n_ts, step):
            try:
                snapshot = area_manager.get_land_use_snapshot(ts_idx)
                if snapshot is not None:
                    for elem_data in snapshot:
                        eid = elem_data.get("element_id", 0)
                        conn.execute(
                            "INSERT OR REPLACE INTO landuse_snapshots "
                            "VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (
                                ts_idx,
                                eid,
                                elem_data.get("agricultural", 0.0),
                                elem_data.get("urban", 0.0),
                                elem_data.get("native_rip", 0.0),
                                elem_data.get("total_area", 0.0),
                                elem_data.get("dominant", "unknown"),
                            ),
                        )
            except Exception as e:
                logger.debug("Area snapshot %d: %s", ts_idx, e)

        conn.commit()
        logger.info("Area data cached")


def get_source_mtimes(model: IWFMModel) -> dict[str, float]:
    """Collect modification times of key source files for staleness check."""
    mtimes: dict[str, float] = {}
    for key in (
        "simulation_file",
        "preprocessor_file",
        "head_output_file",
        "gw_hydrograph_file",
        "stream_hydrograph_file",
        "subsidence_hydrograph_file",
    ):
        path_str = model.metadata.get(key, "")
        if path_str:
            p = Path(path_str)
            if p.exists():
                mtimes[key] = p.stat().st_mtime
    return mtimes


def is_cache_stale(cache_path: Path, model: IWFMModel) -> bool:
    """Check if the cache file is older than any source data file."""
    if not cache_path.exists():
        return True

    cache_mtime = cache_path.stat().st_mtime
    src_mtimes = get_source_mtimes(model)

    for key, mtime in src_mtimes.items():
        if mtime > cache_mtime:
            logger.info("Cache stale: %s newer than cache", key)
            return True

    # Check schema version
    try:
        conn = sqlite3.connect(str(cache_path))
        cur = conn.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        )
        row = cur.fetchone()
        conn.close()
        if row is None or row[0] != SCHEMA_VERSION:
            logger.info("Cache stale: schema version mismatch")
            return True
    except Exception:
        return True

    return False
