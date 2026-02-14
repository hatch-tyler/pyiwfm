"""
Lazy head data loader for IWFM web visualization.

This module provides the LazyHeadDataLoader class for loading
time-varying head data from HDF5 or binary files without
reading all timesteps into memory at once.

Supports two HDF5 formats:
- pyiwfm format: dataset ``head`` with shape ``(n_timesteps, n_nodes, n_layers)``
  and a ``times`` dataset with ISO datetime strings.
- IWFM native format: dataset ``/GWHeadAtAllNodes`` with shape
  ``(n_timesteps, n_nodes * n_layers)`` written directly by the Fortran simulation.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Dataset name used by IWFM Fortran HDF5 output
_IWFM_NATIVE_DATASET = "GWHeadAtAllNodes"


class LazyHeadDataLoader:
    """
    Lazy loader for time-varying head data backed by HDF5 or binary files.

    Uses an LRU cache to keep a bounded number of timesteps in memory.
    Supports dict-like access via ``loader[datetime]``.

    Parameters
    ----------
    file_path : Path or str
        Path to the HDF5 file containing head data.
    dataset_name : str, optional
        Name of the HDF5 dataset. Default is "head".
    cache_size : int, optional
        Maximum number of timesteps to cache. Default is 50.

    Examples
    --------
    >>> loader = LazyHeadDataLoader("GW_HeadAll.hdf5")
    >>> print(loader.times[:3])
    >>> head_at_t0 = loader[loader.times[0]]
    >>> print(head_at_t0.shape)  # (n_nodes, n_layers)
    """

    def __init__(
        self,
        file_path: Path | str,
        dataset_name: str = "head",
        cache_size: int = 50,
    ) -> None:
        """Initialize the lazy loader."""
        self._file_path = Path(file_path)
        self._dataset_name = dataset_name
        self._cache_size = cache_size
        self._cache: OrderedDict[int, NDArray[np.float64]] = OrderedDict()

        # Load metadata (times, shape) without loading all data
        self._times: list[datetime] = []
        self._n_nodes = 0
        self._n_layers = 0
        self._n_frames = 0
        self._h5file = None
        self._iwfm_native = False  # True when using IWFM Fortran HDF5 format

        self._load_metadata()

    @property
    def times(self) -> list[datetime]:
        """Get available time steps."""
        return self._times

    @property
    def n_frames(self) -> int:
        """Get number of available frames."""
        return self._n_frames

    @property
    def shape(self) -> tuple[int, int]:
        """Get shape of each head array (n_nodes, n_layers)."""
        return (self._n_nodes, self._n_layers)

    def _load_metadata(self) -> None:
        """Load metadata from the HDF5 file without reading all data."""
        if not self._file_path.exists():
            logger.warning(f"Head data file not found: {self._file_path}")
            return

        try:
            with h5py.File(self._file_path, "r") as f:
                # Auto-detect IWFM native format
                if _IWFM_NATIVE_DATASET in f:
                    self._iwfm_native = True
                    self._load_metadata_iwfm_native(f)
                elif self._dataset_name in f:
                    self._load_metadata_pyiwfm(f)
                else:
                    logger.warning(
                        f"No recognized dataset found in {self._file_path}. "
                        f"Looked for '{self._dataset_name}' and '{_IWFM_NATIVE_DATASET}'."
                    )
                    return

                logger.info(
                    f"Head data loaded: {self._n_frames} timesteps, "
                    f"{self._n_nodes} nodes, {self._n_layers} layers"
                    f"{' (IWFM native)' if self._iwfm_native else ''}"
                )

        except Exception as e:
            logger.error(f"Failed to load head metadata: {e}")

    def _load_metadata_pyiwfm(self, f) -> None:
        """Load metadata from pyiwfm-format HDF5 (dataset 'head')."""
        ds = f[self._dataset_name]
        # Expected shape: (n_timesteps, n_nodes, n_layers) or (n_timesteps, n_nodes)
        self._n_frames = ds.shape[0]
        self._n_nodes = ds.shape[1]
        self._n_layers = ds.shape[2] if ds.ndim == 3 else 1

        self._load_times(f)

    def _load_metadata_iwfm_native(self, f) -> None:
        """Load metadata from IWFM native HDF5 (dataset '/GWHeadAtAllNodes').

        The Fortran code writes data as shape ``(n_timesteps, n_nodes * n_layers)``
        with data ordered layer-by-layer (all nodes for layer 1, then layer 2, etc.).
        We need ``n_layers`` to reshape; it is stored as an attribute or inferred.
        """
        ds = f[_IWFM_NATIVE_DATASET]
        self._n_frames = ds.shape[0]
        total_columns = ds.shape[1]

        # Try to get n_layers from attributes
        n_layers = None
        if "NLayers" in ds.attrs:
            n_layers = int(ds.attrs["NLayers"])
        elif "NLayers" in f.attrs:
            n_layers = int(f.attrs["NLayers"])

        if n_layers is not None and n_layers > 0:
            self._n_layers = n_layers
            self._n_nodes = total_columns // n_layers
        else:
            # Default: assume 1 layer
            self._n_layers = 1
            self._n_nodes = total_columns

        self._load_times(f)

    def _load_times(self, f) -> None:
        """Load time information from HDF5 file."""
        if "times" in f:
            time_strings = f["times"][:]
            self._times = [
                datetime.fromisoformat(t.decode() if isinstance(t, bytes) else t)
                for t in time_strings
            ]
        elif "time" in f.attrs:
            self._times = []
        else:
            # Generate placeholder times
            from datetime import timedelta
            base = datetime(2000, 1, 1)
            self._times = [base + timedelta(days=i) for i in range(self._n_frames)]

    def _load_frame(self, frame_idx: int) -> NDArray[np.float64]:
        """Load a single frame from disk."""
        with h5py.File(self._file_path, "r") as f:
            if self._iwfm_native:
                ds = f[_IWFM_NATIVE_DATASET]
                flat = ds[frame_idx]  # shape: (n_nodes * n_layers,)
                # Data is stored as [all_nodes_layer1, all_nodes_layer2, ...]
                # Reshape to (n_layers, n_nodes) then transpose to (n_nodes, n_layers)
                data = flat.reshape(self._n_layers, self._n_nodes).T
            else:
                ds = f[self._dataset_name]
                data = ds[frame_idx]

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        return data.astype(np.float64)

    def _evict_if_needed(self) -> None:
        """Evict oldest cache entry if cache is full."""
        while len(self._cache) >= self._cache_size:
            self._cache.popitem(last=False)

    def get_frame(self, frame_idx: int) -> NDArray[np.float64]:
        """
        Get head data for a specific frame index.

        Parameters
        ----------
        frame_idx : int
            Frame index (0-based).

        Returns
        -------
        NDArray[np.float64]
            Head values, shape (n_nodes, n_layers).
        """
        if frame_idx < 0 or frame_idx >= self._n_frames:
            raise IndexError(f"Frame {frame_idx} out of range [0, {self._n_frames})")

        if frame_idx in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(frame_idx)
            return self._cache[frame_idx]

        # Load from disk
        self._evict_if_needed()
        data = self._load_frame(frame_idx)
        self._cache[frame_idx] = data
        return data

    def __getitem__(self, key: datetime | int) -> NDArray[np.float64]:
        """
        Get head data by datetime or frame index.

        Parameters
        ----------
        key : datetime or int
            Time step or frame index.

        Returns
        -------
        NDArray[np.float64]
            Head values.
        """
        if isinstance(key, int):
            return self.get_frame(key)
        elif isinstance(key, datetime):
            if key in self._times:
                idx = self._times.index(key)
                return self.get_frame(idx)
            raise KeyError(f"Time {key} not found in available times")
        else:
            raise TypeError(f"Key must be int or datetime, got {type(key)}")

    def __len__(self) -> int:
        return self._n_frames

    def to_dict(self) -> dict[datetime, NDArray[np.float64]]:
        """
        Load all frames into a dict (for use with TimeAnimationController).

        This loads all data into memory. For large datasets, prefer
        using the lazy access via ``__getitem__``.

        Returns
        -------
        dict[datetime, NDArray]
            All head data keyed by datetime.
        """
        result = {}
        for i, t in enumerate(self._times):
            result[t] = self.get_frame(i)
        return result

    def clear_cache(self) -> None:
        """Clear the frame cache."""
        self._cache.clear()
