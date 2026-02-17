"""
Lazy hydrograph data loader backed by HDF5.

Provides the same interface as ``IWFMHydrographReader`` but reads from
HDF5 cache files produced by ``hydrograph_converter.py``.  This avoids
loading the full text file into memory and enables LRU-cached access.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class LazyHydrographDataLoader:
    """Lazy loader for hydrograph time series backed by HDF5.

    Exposes the same public interface as ``IWFMHydrographReader`` so the
    web viewer route code can use either interchangeably.

    Parameters
    ----------
    file_path : Path or str
        Path to the HDF5 cache file.
    cache_size : int
        Number of timestep rows to keep in the LRU cache.
    """

    def __init__(self, file_path: Path | str, cache_size: int = 100) -> None:
        self._file_path = Path(file_path)
        self._cache_size = cache_size
        self._cache: OrderedDict[int, NDArray[np.float64]] = OrderedDict()

        self._times: list[str] = []
        self._hydrograph_ids: list[int] = []
        self._layers: list[int] = []
        self._node_ids: list[int] = []
        self._n_columns = 0
        self._n_timesteps = 0

        self._load_metadata()

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _load_metadata(self) -> None:
        if not self._file_path.exists():
            logger.warning("Hydrograph HDF5 not found: %s", self._file_path)
            return

        try:
            with h5py.File(self._file_path, "r") as f:
                if "data" not in f:
                    logger.warning("No 'data' dataset in %s", self._file_path)
                    return

                ds = f["data"]
                self._n_timesteps = ds.shape[0]
                self._n_columns = ds.shape[1] if ds.ndim > 1 else 1

                if "times" in f:
                    raw = f["times"][:]
                    self._times = [
                        t.decode() if isinstance(t, bytes) else str(t)
                        for t in raw
                    ]

                if "hydrograph_ids" in f:
                    self._hydrograph_ids = f["hydrograph_ids"][:].tolist()
                if "layers" in f:
                    self._layers = f["layers"][:].tolist()
                if "node_ids" in f:
                    self._node_ids = f["node_ids"][:].tolist()

            logger.info(
                "Hydrograph HDF5 loaded: %d timesteps, %d columns from %s",
                self._n_timesteps, self._n_columns, self._file_path.name,
            )
        except Exception as e:
            logger.error("Failed to load hydrograph HDF5 metadata: %s", e)

    # ------------------------------------------------------------------
    # Properties (IWFMHydrographReader interface)
    # ------------------------------------------------------------------

    @property
    def n_columns(self) -> int:
        return self._n_columns

    @property
    def n_timesteps(self) -> int:
        return self._n_timesteps

    @property
    def times(self) -> list[str]:
        return self._times

    @property
    def hydrograph_ids(self) -> list[int]:
        return self._hydrograph_ids

    @property
    def layers(self) -> list[int]:
        return self._layers

    @property
    def node_ids(self) -> list[int]:
        return self._node_ids

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def _load_row(self, row_idx: int) -> NDArray[np.float64]:
        """Load a single row from the HDF5 dataset."""
        with h5py.File(self._file_path, "r") as f:
            ds = f["data"]
            return ds[row_idx].astype(np.float64)

    def _evict_if_needed(self) -> None:
        while len(self._cache) >= self._cache_size:
            self._cache.popitem(last=False)

    def get_row(self, row_idx: int) -> NDArray[np.float64]:
        """Get a cached row by index."""
        if row_idx in self._cache:
            self._cache.move_to_end(row_idx)
            return self._cache[row_idx]
        self._evict_if_needed()
        row = self._load_row(row_idx)
        self._cache[row_idx] = row
        return row

    def get_time_series(self, column_index: int) -> tuple[list[str], list[float]]:
        """Get time series for a specific column.

        Parameters
        ----------
        column_index : int
            0-based column index.

        Returns
        -------
        tuple[list[str], list[float]]
            (times, values) where times are ISO 8601 strings.
        """
        if column_index < 0 or column_index >= self._n_columns:
            return [], []

        # For full-column extraction, read the whole column at once
        # (more efficient than row-by-row for column access).
        with h5py.File(self._file_path, "r") as f:
            ds = f["data"]
            values = ds[:, column_index].astype(np.float64).tolist()

        return self._times, values

    def find_column_by_node_id(self, node_id: int) -> int | None:
        """Find column index for a given node/element ID."""
        if node_id in self._node_ids:
            return self._node_ids.index(node_id)
        return None
