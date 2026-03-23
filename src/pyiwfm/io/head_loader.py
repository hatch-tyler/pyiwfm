"""
Lazy nodal data loader for IWFM HDF5 output.

This module provides the LazyHeadDataLoader class for loading
time-varying nodal data (heads or subsidence) from HDF5 files without
reading all timesteps into memory at once.

Supports three IWFM native HDF5 dataset formats:

- ``GWHeadAtAllNodes`` — groundwater head at all nodes
- ``SubsidenceAtAllNodes`` — subsidence at all nodes (v4.1/v5.1)
- pyiwfm format (dataset ``head`` with shape
  ``(n_timesteps, n_nodes, n_layers)``)

All native formats store data as ``(n_timesteps, n_nodes * n_layers)``
with layer-major ordering.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Dataset names used by IWFM Fortran HDF5 output
_IWFM_HEAD_DATASET = "GWHeadAtAllNodes"
_IWFM_SUBSIDENCE_DATASET = "SubsidenceAtAllNodes"
_IWFM_NATIVE_DATASETS = (_IWFM_HEAD_DATASET, _IWFM_SUBSIDENCE_DATASET)

# Backward-compat alias
_IWFM_NATIVE_DATASET = _IWFM_HEAD_DATASET


class LazyHeadDataLoader:
    """
    Lazy loader for time-varying nodal data backed by HDF5 files.

    Uses an LRU cache to keep a bounded number of timesteps in memory.
    Supports dict-like access via ``loader[datetime]``.

    Auto-detects the dataset type (head vs subsidence) from the HDF5
    contents. The ``data_type`` property reports what was found.

    Parameters
    ----------
    file_path : Path or str
        Path to the HDF5 file containing nodal data.
    dataset_name : str, optional
        Name of the HDF5 dataset for pyiwfm-format files. Default is "head".
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
        n_layers: int | None = None,
    ) -> None:
        """Initialize the lazy loader.

        Parameters
        ----------
        file_path : Path or str
            Path to the HDF5 file containing nodal data.
        dataset_name : str, optional
            Name of the HDF5 dataset for pyiwfm-format files.
        cache_size : int, optional
            Maximum number of timesteps to cache. Default is 50.
        n_layers : int or None, optional
            Number of model layers. When provided, takes precedence over
            any ``NLayers`` attribute in the HDF5 file. Required for
            IWFM native HDF5 files that lack the attribute (which is
            the common case — the Fortran writer does not store it).
        """
        self._file_path = Path(file_path)
        self._dataset_name = dataset_name
        self._cache_size = cache_size
        self._cache: OrderedDict[int, NDArray[np.float64]] = OrderedDict()
        self._explicit_n_layers = n_layers

        # Load metadata (times, shape) without loading all data
        self._times: list[datetime] = []
        self._n_nodes = 0
        self._n_layers = 0
        self._n_frames = 0
        self._h5file = None
        self._iwfm_native = False  # True when using IWFM Fortran HDF5 format
        self._native_dataset_name = ""  # Actual dataset name found in HDF5
        self._data_type = "head"  # "head" or "subsidence"

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
        """Get shape of each data array (n_nodes, n_layers)."""
        return (self._n_nodes, self._n_layers)

    @property
    def data_type(self) -> str:
        """Get the detected data type: ``"head"`` or ``"subsidence"``."""
        return self._data_type

    @property
    def n_layers(self) -> int:
        """Get number of layers."""
        return self._n_layers

    @property
    def n_nodes(self) -> int:
        """Get number of nodes."""
        return self._n_nodes

    def _load_metadata(self) -> None:
        """Load metadata from the HDF5 file without reading all data."""
        if not self._file_path.exists():
            logger.warning(f"Data file not found: {self._file_path}")
            return

        try:
            with h5py.File(self._file_path, "r") as f:
                # Auto-detect IWFM native format (head or subsidence)
                for ds_name in _IWFM_NATIVE_DATASETS:
                    if ds_name in f:
                        self._iwfm_native = True
                        self._native_dataset_name = ds_name
                        if ds_name == _IWFM_SUBSIDENCE_DATASET:
                            self._data_type = "subsidence"
                        self._load_metadata_iwfm_native(f)
                        break
                else:
                    if self._dataset_name in f:
                        self._load_metadata_pyiwfm(f)
                    else:
                        logger.warning(
                            f"No recognized dataset found in {self._file_path}. "
                            f"Looked for '{self._dataset_name}', "
                            f"'{_IWFM_HEAD_DATASET}', and "
                            f"'{_IWFM_SUBSIDENCE_DATASET}'."
                        )
                        return

                logger.info(
                    f"{self._data_type.title()} data loaded: {self._n_frames} timesteps, "
                    f"{self._n_nodes} nodes, {self._n_layers} layers"
                    f"{' (IWFM native)' if self._iwfm_native else ''}"
                )

        except (OSError, KeyError, ValueError, TypeError) as e:
            logger.error(f"Failed to load metadata from {self._file_path}: {e}")

    def _load_metadata_pyiwfm(self, f: Any) -> None:
        """Load metadata from pyiwfm-format HDF5 (dataset 'head')."""
        ds = f[self._dataset_name]
        # Expected shape: (n_timesteps, n_nodes, n_layers) or (n_timesteps, n_nodes)
        self._n_frames = ds.shape[0]
        self._n_nodes = ds.shape[1]
        self._n_layers = ds.shape[2] if ds.ndim == 3 else 1

        self._load_times(f)

    def _load_metadata_iwfm_native(self, f: Any) -> None:
        """Load metadata from IWFM native HDF5.

        Handles both ``GWHeadAtAllNodes`` and ``SubsidenceAtAllNodes``.
        The Fortran code writes data as shape ``(n_timesteps, n_nodes * n_layers)``
        with data ordered layer-by-layer (all nodes for layer 1, then layer 2, etc.).
        We need ``n_layers`` to reshape correctly.

        Resolution order for ``n_layers``:
        1. Explicit ``n_layers`` parameter passed to constructor (from model geometry)
        2. ``NLayers`` attribute on the dataset or file
        3. Fallback to 1 with a warning
        """
        ds = f[self._native_dataset_name]
        self._n_frames = ds.shape[0]
        total_columns = ds.shape[1]

        # Determine n_layers: explicit param > HDF5 attribute > fallback
        n_layers = self._explicit_n_layers

        if n_layers is None:
            if "NLayers" in ds.attrs:
                n_layers = int(ds.attrs["NLayers"])
            elif "NLayers" in f.attrs:
                n_layers = int(f.attrs["NLayers"])

        if n_layers is not None and n_layers > 0:
            self._n_layers = n_layers
            self._n_nodes = total_columns // n_layers
        else:
            logger.warning(
                "NLayers not provided and not found in HDF5 attributes for %s. "
                "Assuming 1 layer. For multi-layer models, pass n_layers explicitly.",
                self._file_path.name,
            )
            self._n_layers = 1
            self._n_nodes = total_columns

        self._load_times(f)

    def _load_times(self, f: Any) -> None:
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
                ds = f[self._native_dataset_name]
                flat = ds[frame_idx]  # shape: (n_nodes * n_layers,)
                # Data is stored as [all_nodes_layer1, all_nodes_layer2, ...]
                # Reshape to (n_layers, n_nodes) then transpose to (n_nodes, n_layers)
                data = flat.reshape(self._n_layers, self._n_nodes).T
            else:
                ds = f[self._dataset_name]
                data = ds[frame_idx]

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        result: NDArray[np.float64] = data.astype(np.float64)
        return result

    def _evict_if_needed(self) -> None:
        """Evict oldest cache entry if cache is full."""
        while len(self._cache) >= self._cache_size:
            self._cache.popitem(last=False)

    def get_frame(self, frame_idx: int) -> NDArray[np.float64]:
        """
        Get nodal data for a specific frame index.

        Parameters
        ----------
        frame_idx : int
            Frame index (0-based).

        Returns
        -------
        NDArray[np.float64]
            Nodal values, shape (n_nodes, n_layers).
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

    def get_head(
        self,
        frame_idx: int,
        layer: int,
        node_ids: tuple[int, ...] | None = None,
    ) -> NDArray[np.float64] | None:
        """Get nodal values for a single layer at a specific timestep.

        This method is used by ``ResultsExtractor`` to access per-layer
        data for FE interpolation.

        Parameters
        ----------
        frame_idx : int
            Frame index (0-based).
        layer : int
            0-based layer index.
        node_ids : tuple[int, ...] or None
            If provided, return values only at these 1-based node IDs.
            If None, return values at all nodes.

        Returns
        -------
        NDArray[np.float64] or None
            Values at the requested nodes for the given layer, or None
            if the frame is out of range.
        """
        if frame_idx < 0 or frame_idx >= self._n_frames:
            return None
        if layer < 0 or layer >= self._n_layers:
            return None

        frame = self.get_frame(frame_idx)
        col = frame[:, layer]

        if node_ids is not None:
            indices = np.array([nid - 1 for nid in node_ids])
            result: NDArray[np.float64] = col[indices]
            return result
        return col

    def get_composite_subsidence(
        self,
        frame_idx: int,
    ) -> NDArray[np.float64] | None:
        """Get total subsidence at each node by summing across all layers.

        Subsidence is additive across layers (unlike head which uses
        T-weighted averaging). This matches the Fortran behavior in
        ``Class_IWFM2OBS.f90:ApplyMultiLayerSubsidence``.

        Parameters
        ----------
        frame_idx : int
            Frame index (0-based).

        Returns
        -------
        NDArray[np.float64] or None
            Total subsidence per node, shape (n_nodes,), or None if out
            of range.
        """
        if frame_idx < 0 or frame_idx >= self._n_frames:
            return None

        frame = self.get_frame(frame_idx)  # (n_nodes, n_layers)
        total: NDArray[np.float64] = np.nansum(frame, axis=1)
        return total

    def __getitem__(self, key: datetime | int) -> NDArray[np.float64]:
        """
        Get data by datetime or frame index.

        Parameters
        ----------
        key : datetime or int
            Time step or frame index.

        Returns
        -------
        NDArray[np.float64]
            Nodal values.
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
        dict[datetime, NDArray[np.float64]]
            All data keyed by datetime.
        """
        result = {}
        for i, t in enumerate(self._times):
            result[t] = self.get_frame(i)
        return result

    def get_layer_range(
        self,
        layer: int,
        percentile_lo: float = 2.0,
        percentile_hi: float = 98.0,
        max_frames: int = 0,
    ) -> tuple[float, float, int]:
        """Compute robust min/max values across all (or sampled) timesteps.

        Parameters
        ----------
        layer : int
            1-based layer number. Use 0 for composite subsidence (sum
            across all layers).
        percentile_lo, percentile_hi : float
            Percentiles for robust range (default 2nd-98th).
        max_frames : int
            If > 0, sample at most this many evenly-spaced frames
            instead of scanning all timesteps.

        Returns
        -------
        tuple[float, float, int]
            (min_value, max_value, n_frames_scanned)
        """
        total = self._n_frames
        if total == 0:
            return (0.0, 1.0, 0)

        # Determine which frames to sample
        if max_frames > 0 and max_frames < total:
            indices = np.linspace(0, total - 1, max_frames, dtype=int)
            indices = np.unique(indices)
        else:
            indices = np.arange(total)

        # Collect valid values
        all_valid: list[float] = []
        use_composite = layer == 0 and self._data_type == "subsidence"

        for idx in indices:
            if use_composite:
                col = self.get_composite_subsidence(int(idx))
                if col is None:
                    continue
            else:
                layer_idx = layer - 1
                frame = self.get_frame(int(idx))
                if layer_idx >= frame.shape[1]:
                    continue
                col = frame[:, layer_idx]

            valid = col[col > -9000]
            if len(valid) > 0:
                all_valid.extend(valid.tolist())

        n_scanned = len(indices)
        if not all_valid:
            return (0.0, 1.0, n_scanned)

        arr = np.array(all_valid)
        lo = float(np.percentile(arr, percentile_lo))
        hi = float(np.percentile(arr, percentile_hi))
        return (round(lo, 3), round(hi, 3), n_scanned)

    def clear_cache(self) -> None:
        """Clear the frame cache."""
        self._cache.clear()
