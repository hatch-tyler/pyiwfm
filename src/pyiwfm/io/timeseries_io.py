"""
Unified lazy time-series I/O for IWFM nodal and tabular outputs.

**Read-only by design.** This module reads IWFM simulation output that
the model itself produced; there is no writer because the data here are
*outputs*, not editable model inputs. To change them, edit the model
inputs and re-run IWFM. See ``docs/user_guide/inputs_vs_outputs.rst``.

This module replaces the v1.x cluster of head/hydrograph loader and
converter modules (``head_loader``, ``head_all_converter``,
``hydrograph_loader``, ``hydrograph_converter``) with a single set of
classes that share scaffolding for LRU caching, HDF5 source handling,
and on-disk schema. The two loader classes stay distinct because the
data shapes differ (nodal vs tabular) and forcing them into one class
would require ``isinstance`` dispatch in every method — see
``docs/MIGRATION_v1_to_v2.md`` for the full rationale.

Public API
----------
- :class:`LazyNodalLoader` — replaces ``LazyHeadDataLoader``. Reads
  ``(n_timesteps, n_nodes, n_layers)`` data from IWFM-native HDF5
  (``GWHeadAtAllNodes``, ``SubsidenceAtAllNodes``) or pyiwfm-format
  HDF5 (``head`` dataset).
- :class:`LazyTabularLoader` — replaces ``LazyHydrographDataLoader``.
  Reads ``(n_timesteps, n_columns)`` data from a hydrograph HDF5 cache.
- :class:`TimeSeriesCache` — replaces the ``convert_*_to_hdf``
  functions. Static-method factory: ``from_iwfm_headall_text`` and
  ``from_iwfm_hydrograph_text``.

The text reader for IWFM ``.out`` hydrograph files
(:class:`pyiwfm.io.hydrograph_reader.IWFMHydrographReader`) is kept
separate because its eager-load semantics and SMP-conversion bridge
(:meth:`get_columns_as_smp_dict`) don't fit the lazy-frame model used
here.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

import h5py
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# IWFM-native HDF5 dataset names
_IWFM_HEAD_DATASET = "GWHeadAtAllNodes"
_IWFM_SUBSIDENCE_DATASET = "SubsidenceAtAllNodes"
_IWFM_NATIVE_DATASETS = (_IWFM_HEAD_DATASET, _IWFM_SUBSIDENCE_DATASET)

# HEADALL text-format constants (from GWHydrograph.f90 (2X,F10.4) format)
_HEADALL_COL_WIDTH = 12  # 2 spaces + F10.4 = 12 chars per column
_HEADALL_TIME_WIDTH = 21  # Width of the timestamp prefix
_HEADALL_CHUNK_GROW = 256  # How many timesteps to grow HDF5 by when full


# =============================================================================
# Internal: shared LRU frame cache mixin
# =============================================================================


class _LRUFrameCache:
    """Mixin providing an OrderedDict-backed LRU cache for indexed frames.

    Subclasses must implement :meth:`_load_frame` to fetch a frame from
    the underlying source. The mixin handles cache lookup, MRU promotion,
    and bounded eviction.
    """

    _cache: OrderedDict[int, NDArray[np.float64]]
    _cache_size: int

    def _init_cache(self, cache_size: int) -> None:
        """Initialize the LRU cache. Subclasses call this from ``__init__``."""
        self._cache = OrderedDict()
        self._cache_size = cache_size

    def _evict_if_needed(self) -> None:
        """Drop the oldest cache entry if the cache is full."""
        while len(self._cache) >= self._cache_size:
            self._cache.popitem(last=False)

    def _get_cached_or_load(self, idx: int) -> NDArray[np.float64]:
        """Return the cached frame at ``idx``, loading and caching on miss.

        Cache hits are promoted to MRU position. Subclasses provide the
        on-miss load via :meth:`_load_frame`.
        """
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]
        self._evict_if_needed()
        data = self._load_frame(idx)
        self._cache[idx] = data
        return data

    def _load_frame(self, idx: int) -> NDArray[np.float64]:  # pragma: no cover
        """Load a frame from the underlying source. Subclasses override."""
        raise NotImplementedError

    def clear_cache(self) -> None:
        """Drop all cached frames."""
        self._cache.clear()


# =============================================================================
# Internal HDF5 sources
# =============================================================================


class _HDF5HeadSource:
    """Reads nodal data from an HDF5 file.

    Auto-detects three formats:

    - IWFM-native ``GWHeadAtAllNodes`` (head data)
    - IWFM-native ``SubsidenceAtAllNodes`` (subsidence data)
    - pyiwfm format (a 3-D ``head`` dataset of shape
      ``(n_timesteps, n_nodes, n_layers)``)

    Native IWFM datasets store data as ``(n_timesteps, n_nodes * n_layers)``
    with layer-major ordering — all nodes for layer 0 first, then all
    nodes for layer 1, etc. This source reshapes that flat row to
    ``(n_nodes, n_layers)`` on every read so callers always see the
    same layout.
    """

    def __init__(
        self,
        file_path: Path,
        dataset_name: str,
        explicit_n_layers: int | None,
    ) -> None:
        self._file_path = file_path
        self._dataset_name = dataset_name
        self._explicit_n_layers = explicit_n_layers

        self.times: list[datetime] = []
        self.n_frames = 0
        self.n_nodes = 0
        self.n_layers = 0
        self.data_type = "head"  # "head" or "subsidence"

        self._iwfm_native = False
        self._native_dataset_name = ""

        self._load_metadata()

    def _load_metadata(self) -> None:
        if not self._file_path.exists():
            logger.warning("Data file not found: %s", self._file_path)
            return

        try:
            with h5py.File(self._file_path, "r") as f:
                # Auto-detect IWFM native format (head or subsidence)
                for ds_name in _IWFM_NATIVE_DATASETS:
                    if ds_name in f:
                        self._iwfm_native = True
                        self._native_dataset_name = ds_name
                        if ds_name == _IWFM_SUBSIDENCE_DATASET:
                            self.data_type = "subsidence"
                        self._load_metadata_iwfm_native(f)
                        break
                else:
                    if self._dataset_name in f:
                        self._load_metadata_pyiwfm(f)
                    else:
                        logger.warning(
                            "No recognized dataset found in %s. Looked for '%s', '%s', and '%s'.",
                            self._file_path,
                            self._dataset_name,
                            _IWFM_HEAD_DATASET,
                            _IWFM_SUBSIDENCE_DATASET,
                        )
                        return

                logger.info(
                    "%s data loaded: %d timesteps, %d nodes, %d layers%s",
                    self.data_type.title(),
                    self.n_frames,
                    self.n_nodes,
                    self.n_layers,
                    " (IWFM native)" if self._iwfm_native else "",
                )

        except (OSError, KeyError, ValueError, TypeError) as e:
            logger.error("Failed to load metadata from %s: %s", self._file_path, e)

    def _load_metadata_pyiwfm(self, f: Any) -> None:
        """Read shape from a pyiwfm-format HDF5 file (3-D ``head`` dataset)."""
        ds = f[self._dataset_name]
        self.n_frames = ds.shape[0]
        self.n_nodes = ds.shape[1]
        self.n_layers = ds.shape[2] if ds.ndim == 3 else 1
        self._load_times(f)

    def _load_metadata_iwfm_native(self, f: Any) -> None:
        """Read shape and resolve ``n_layers`` from an IWFM-native HDF5 file.

        The Fortran writer stores data as ``(n_timesteps, n_nodes * n_layers)``
        and does NOT record ``n_layers`` as an attribute. Resolution order:

        1. Explicit ``n_layers`` arg passed to the loader (preferred —
           sourced from model geometry)
        2. ``NLayers`` attribute on the dataset or file (rare)
        3. Fallback to 1 with a warning
        """
        ds = f[self._native_dataset_name]
        self.n_frames = ds.shape[0]
        total_columns = ds.shape[1]

        n_layers = self._explicit_n_layers
        if n_layers is None:
            if "NLayers" in ds.attrs:
                n_layers = int(ds.attrs["NLayers"])
            elif "NLayers" in f.attrs:
                n_layers = int(f.attrs["NLayers"])

        if n_layers is not None and n_layers > 0:
            self.n_layers = n_layers
            self.n_nodes = total_columns // n_layers
        else:
            logger.warning(
                "NLayers not provided and not found in HDF5 attributes for %s. "
                "Assuming 1 layer. For multi-layer models, pass n_layers explicitly.",
                self._file_path.name,
            )
            self.n_layers = 1
            self.n_nodes = total_columns

        self._load_times(f)

    def _load_times(self, f: Any) -> None:
        """Populate ``self.times`` from the HDF5 ``times`` dataset, or fallback."""
        if "times" in f:
            time_strings = f["times"][:]
            self.times = [
                datetime.fromisoformat(t.decode() if isinstance(t, bytes) else t)
                for t in time_strings
            ]
        elif "time" in f.attrs:
            self.times = []
        else:
            base = datetime(2000, 1, 1)
            self.times = [base + timedelta(days=i) for i in range(self.n_frames)]

    def read_frame(self, idx: int) -> NDArray[np.float64]:
        """Read frame ``idx`` from disk, reshape to ``(n_nodes, n_layers)``."""
        with h5py.File(self._file_path, "r") as f:
            if self._iwfm_native:
                ds = f[self._native_dataset_name]
                flat = ds[idx]  # shape: (n_nodes * n_layers,)
                # Layer-major → (n_layers, n_nodes), then transpose
                data = flat.reshape(self.n_layers, self.n_nodes).T
            else:
                ds = f[self._dataset_name]
                data = ds[idx]

        if data.ndim == 1:
            data = data.reshape(-1, 1)
        result: NDArray[np.float64] = data.astype(np.float64)
        return result


class _HDF5HydrographSource:
    """Reads tabular hydrograph data from a pyiwfm HDF5 cache file.

    The schema is the one produced by :meth:`TimeSeriesCache.from_iwfm_hydrograph_text`:

    - ``data`` dataset of shape ``(n_timesteps, n_columns)``
    - ``times`` dataset of ISO-8601 strings
    - optional ``hydrograph_ids`` / ``layers`` / ``node_ids`` int32 datasets
    """

    def __init__(self, file_path: Path) -> None:
        self._file_path = file_path
        self.n_timesteps = 0
        self.n_columns = 0
        self.times: list[str] = []
        self.hydrograph_ids: list[int] = []
        self.layers: list[int] = []
        self.node_ids: list[int] = []
        self._load_metadata()

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
                self.n_timesteps = ds.shape[0]
                self.n_columns = ds.shape[1] if ds.ndim > 1 else 1

                if "times" in f:
                    raw = f["times"][:]
                    self.times = [t.decode() if isinstance(t, bytes) else str(t) for t in raw]
                if "hydrograph_ids" in f:
                    self.hydrograph_ids = f["hydrograph_ids"][:].tolist()
                if "layers" in f:
                    self.layers = f["layers"][:].tolist()
                if "node_ids" in f:
                    self.node_ids = f["node_ids"][:].tolist()

            logger.info(
                "Hydrograph HDF5 loaded: %d timesteps, %d columns from %s",
                self.n_timesteps,
                self.n_columns,
                self._file_path.name,
            )
        except (OSError, KeyError, ValueError, TypeError) as e:
            logger.error("Failed to load hydrograph HDF5 metadata: %s", e)

    def read_frame(self, idx: int) -> NDArray[np.float64]:
        """Read row ``idx`` (one timestep across all columns)."""
        with h5py.File(self._file_path, "r") as f:
            ds = f["data"]
            result: NDArray[np.float64] = ds[idx].astype(np.float64)
            return result

    def read_column(self, col_idx: int) -> NDArray[np.float64]:
        """Read column ``col_idx`` (full time series for one location).

        Bypasses the row cache — column-wise extraction is more efficient
        as a single HDF5 slice than as ``n_timesteps`` cached row reads.
        """
        with h5py.File(self._file_path, "r") as f:
            ds = f["data"]
            result: NDArray[np.float64] = ds[:, col_idx].astype(np.float64)
            return result


# =============================================================================
# Public: loader classes
# =============================================================================


class LazyNodalLoader(_LRUFrameCache):
    """
    Lazy LRU-cached loader for nodal time-series data (heads or subsidence).

    Reads from IWFM-native HDF5 (``GWHeadAtAllNodes`` / ``SubsidenceAtAllNodes``)
    or pyiwfm-format HDF5 (``head`` dataset). Auto-detects the dataset type
    from the file contents; the :attr:`data_type` property reports
    ``"head"`` or ``"subsidence"``.

    Parameters
    ----------
    file_path : Path or str
        Path to the HDF5 file.
    dataset_name : str, optional
        Name of the HDF5 dataset for pyiwfm-format files. Default ``"head"``.
    cache_size : int, optional
        Max number of frames in the LRU cache. Default 50.
    n_layers : int or None, optional
        Number of model layers. Required for IWFM-native HDF5 (the
        Fortran writer doesn't store ``NLayers``). When ``None``, falls
        back to the ``NLayers`` HDF5 attribute or 1 with a warning.

    Examples
    --------
    >>> loader = LazyNodalLoader("GW_HeadAll.hdf5", n_layers=4)
    >>> frame = loader[loader.times[0]]      # (n_nodes, n_layers)
    >>> layer1 = loader.get_head(0, layer=0) # (n_nodes,)
    """

    def __init__(
        self,
        file_path: Path | str,
        dataset_name: str = "head",
        cache_size: int = 50,
        n_layers: int | None = None,
    ) -> None:
        self._init_cache(cache_size)
        self._source = _HDF5HeadSource(Path(file_path), dataset_name, n_layers)

    # -- metadata properties (stable v1.x API) ------------------------------

    @property
    def times(self) -> list[datetime]:
        """Time stamps for each frame (datetime objects)."""
        return self._source.times

    @property
    def n_frames(self) -> int:
        """Number of frames available."""
        return self._source.n_frames

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of each frame as ``(n_nodes, n_layers)``."""
        return (self._source.n_nodes, self._source.n_layers)

    @property
    def data_type(self) -> str:
        """``"head"`` or ``"subsidence"`` (auto-detected)."""
        return self._source.data_type

    @property
    def n_layers(self) -> int:
        return self._source.n_layers

    @property
    def n_nodes(self) -> int:
        return self._source.n_nodes

    # -- frame access -------------------------------------------------------

    def _load_frame(self, idx: int) -> NDArray[np.float64]:
        return self._source.read_frame(idx)

    def get_frame(self, frame_idx: int) -> NDArray[np.float64]:
        """Get the full frame at ``frame_idx`` as ``(n_nodes, n_layers)``.

        Raises :class:`IndexError` for out-of-range indices.
        """
        if frame_idx < 0 or frame_idx >= self._source.n_frames:
            raise IndexError(f"Frame {frame_idx} out of range [0, {self._source.n_frames})")
        return self._get_cached_or_load(frame_idx)

    def get_head(
        self,
        frame_idx: int,
        layer: int,
        node_ids: tuple[int, ...] | None = None,
    ) -> NDArray[np.float64] | None:
        """Get nodal values for a single layer at a specific timestep.

        Used by ``ResultsExtractor`` for FE interpolation.

        Parameters
        ----------
        frame_idx : int
            0-based frame index.
        layer : int
            0-based layer index.
        node_ids : tuple[int, ...] or None
            If provided, return values only at these 1-based node IDs.

        Returns
        -------
        NDArray[np.float64] or None
            Layer slice, or ``None`` if the frame or layer is out of range.
        """
        if frame_idx < 0 or frame_idx >= self._source.n_frames:
            return None
        if layer < 0 or layer >= self._source.n_layers:
            return None

        col = self.get_frame(frame_idx)[:, layer]
        if node_ids is not None:
            indices = np.array([nid - 1 for nid in node_ids])
            result: NDArray[np.float64] = col[indices]
            return result
        return col

    def get_composite_subsidence(
        self,
        frame_idx: int,
    ) -> NDArray[np.float64] | None:
        """Sum subsidence across all layers at a single timestep.

        Subsidence is additive across layers (unlike head, which uses
        T-weighted averaging). Mirrors ``Class_IWFM2OBS.f90:ApplyMultiLayerSubsidence``.
        """
        if frame_idx < 0 or frame_idx >= self._source.n_frames:
            return None
        total: NDArray[np.float64] = np.nansum(self.get_frame(frame_idx), axis=1)
        return total

    def __getitem__(self, key: datetime | int) -> NDArray[np.float64]:
        """Get a frame by frame index or by datetime."""
        if isinstance(key, int):
            return self.get_frame(key)
        if isinstance(key, datetime):
            if key in self._source.times:
                idx = self._source.times.index(key)
                return self.get_frame(idx)
            raise KeyError(f"Time {key} not found in available times")
        raise TypeError(f"Key must be int or datetime, got {type(key)}")

    def __len__(self) -> int:
        return self._source.n_frames

    def to_dict(self) -> dict[datetime, NDArray[np.float64]]:
        """Load all frames into a dict (full in-memory materialization).

        Useful for the ``TimeAnimationController``. For large datasets,
        prefer ``__getitem__`` for lazy access.
        """
        return {t: self.get_frame(i) for i, t in enumerate(self._source.times)}

    def get_layer_range(
        self,
        layer: int,
        percentile_lo: float = 2.0,
        percentile_hi: float = 98.0,
        max_frames: int = 0,
    ) -> tuple[float, float, int]:
        """Compute robust min/max across (sampled) timesteps for one layer.

        Parameters
        ----------
        layer : int
            1-based layer number. Use 0 for composite subsidence (sum
            across all layers).
        percentile_lo, percentile_hi : float
            Percentiles for the robust range.
        max_frames : int
            If > 0, sample at most this many evenly-spaced frames instead
            of scanning every timestep.

        Returns
        -------
        tuple[float, float, int]
            ``(min_value, max_value, n_frames_scanned)``
        """
        total = self._source.n_frames
        if total == 0:
            return (0.0, 1.0, 0)

        if max_frames > 0 and max_frames < total:
            indices = np.unique(np.linspace(0, total - 1, max_frames, dtype=int))
        else:
            indices = np.arange(total)

        all_valid: list[float] = []
        use_composite = layer == 0 and self._source.data_type == "subsidence"

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


class LazyTabularLoader(_LRUFrameCache):
    """
    Lazy LRU-cached loader for tabular hydrograph time-series data.

    Reads from a pyiwfm HDF5 cache produced by
    :meth:`TimeSeriesCache.from_iwfm_hydrograph_text`. Exposes the same
    public interface as :class:`pyiwfm.io.hydrograph_reader.IWFMHydrographReader`
    so route handlers can use either interchangeably.

    Parameters
    ----------
    file_path : Path or str
        Path to the HDF5 cache file.
    cache_size : int, optional
        Max number of timestep rows in the LRU cache. Default 100.
    """

    def __init__(self, file_path: Path | str, cache_size: int = 100) -> None:
        self._init_cache(cache_size)
        self._source = _HDF5HydrographSource(Path(file_path))

    # -- metadata properties (mirrors IWFMHydrographReader) ----------------

    @property
    def n_columns(self) -> int:
        return self._source.n_columns

    @property
    def n_timesteps(self) -> int:
        return self._source.n_timesteps

    @property
    def times(self) -> list[str]:
        return self._source.times

    @property
    def hydrograph_ids(self) -> list[int]:
        return self._source.hydrograph_ids

    @property
    def layers(self) -> list[int]:
        return self._source.layers

    @property
    def node_ids(self) -> list[int]:
        return self._source.node_ids

    # -- data access -------------------------------------------------------

    def _load_frame(self, idx: int) -> NDArray[np.float64]:
        return self._source.read_frame(idx)

    def get_row(self, row_idx: int) -> NDArray[np.float64]:
        """Get the row at ``row_idx`` (one timestep across all columns)."""
        return self._get_cached_or_load(row_idx)

    def get_time_series(self, column_index: int) -> tuple[list[str], list[float]]:
        """Get the time series for a single column.

        Reads the entire column from disk in one slice — much more
        efficient than ``n_timesteps`` cached row reads when the caller
        wants a full time series for one location.
        """
        if column_index < 0 or column_index >= self._source.n_columns:
            return [], []
        values = self._source.read_column(column_index).tolist()
        return self._source.times, values

    def find_column_by_node_id(self, node_id: int) -> int | None:
        """Look up the 0-based column index for a given node/element ID."""
        if node_id in self._source.node_ids:
            return self._source.node_ids.index(node_id)
        return None


# =============================================================================
# Public: cache builder
# =============================================================================


class TimeSeriesCache:
    """Static-method factory: build pyiwfm HDF5 caches from IWFM text outputs.

    Replaces the v1.x ``convert_headall_to_hdf`` and
    ``convert_hydrograph_to_hdf`` functions. The two static methods
    produce HDF5 files that :class:`LazyNodalLoader` and
    :class:`LazyTabularLoader` (respectively) can consume.

    The class itself never needs to be instantiated — the methods are
    grouped here so the related namespaces (``LazyNodalLoader`` /
    ``LazyTabularLoader`` / ``TimeSeriesCache``) are easy to discover.
    """

    @staticmethod
    def from_iwfm_headall_text(
        text_file: str | Path,
        hdf_file: str | Path | None = None,
        n_layers: int = 1,
    ) -> Path:
        """Stream an IWFM ``GWALLOUTFL`` text file into a pyiwfm HDF5 cache.

        Memory usage stays at ``O(n_nodes * n_layers)`` regardless of
        timestep count — the file is parsed line-by-line and each
        timestep is written directly to a resizable HDF5 dataset.

        Text format (per ``GWHydrograph.f90``):

        - 4 title lines (decorative box with unit info)
        - 2 header lines (``* NODE`` row + ``* TIME node1 node2 ...`` row)
        - Data rows: 21-char timestamp + ``n_nodes`` values in ``(2X,F10.4)``
        - Multi-layer: ``n_layers`` consecutive rows per timestep
          (continuation rows start with 21 spaces instead of a timestamp)

        Parameters
        ----------
        text_file : str or Path
            Path to the IWFM text output file (e.g. ``C2VSimFG_GW_HeadAll.out``).
        hdf_file : str or Path or None
            Output HDF5 path. If ``None``, uses ``text_file`` with ``.hdf``.
        n_layers : int
            Number of groundwater layers. Default 1.

        Returns
        -------
        Path
            Path to the created HDF5 file.
        """
        text_path = Path(text_file)
        hdf_path = Path(hdf_file) if hdf_file is not None else text_path.with_suffix(".hdf")
        from pyiwfm.io.timeseries_ascii import parse_iwfm_timestamp

        logger.info("Converting %s -> %s (n_layers=%d)", text_path, hdf_path, n_layers)

        with open(text_path) as fh:
            header_lines_read = _skip_headall_titles(fh)
            header2 = _read_headall_node_header(fh)
            header_lines_read += 2

            node_ids = _parse_headall_node_ids(header2)
            n_nodes = len(node_ids)
            if n_nodes == 0:
                raise ValueError("Could not parse any node IDs from header line")
            logger.info("Detected %d nodes from header", n_nodes)

            data_lines = _count_remaining_lines(fh)
            estimated_timesteps = max(data_lines // n_layers, 1)
            logger.info(
                "Estimated %d timesteps from %d data lines",
                estimated_timesteps,
                data_lines,
            )

            fh.seek(0)
            for _ in range(header_lines_read):
                fh.readline()

            timestamps: list[str] = []
            t_idx = 0

            with h5py.File(hdf_path, "w") as hf:
                ds = hf.create_dataset(
                    "head",
                    shape=(estimated_timesteps, n_nodes, n_layers),
                    maxshape=(None, n_nodes, n_layers),
                    dtype=np.float64,
                    compression="gzip",
                    compression_opts=4,
                    chunks=(1, n_nodes, n_layers),
                )

                row_buf = np.empty((n_nodes, n_layers), dtype=np.float64)

                while True:
                    line = fh.readline()
                    if not line:
                        break

                    line = line.rstrip()
                    if not line or line.startswith("*"):
                        continue

                    ts_text = line[:_HEADALL_TIME_WIDTH].strip()
                    if not ts_text:
                        continue

                    timestamp = parse_iwfm_timestamp(ts_text)
                    timestamps.append(timestamp.isoformat())

                    row_buf[:, 0] = _parse_headall_data_row(line, n_nodes)
                    for layer_idx in range(1, n_layers):
                        cont_line = fh.readline()
                        if not cont_line:
                            break
                        row_buf[:, layer_idx] = _parse_headall_data_row(cont_line.rstrip(), n_nodes)

                    if t_idx >= ds.shape[0]:
                        ds.resize(ds.shape[0] + _HEADALL_CHUNK_GROW, axis=0)
                    ds[t_idx, :, :] = row_buf
                    t_idx += 1

                    if t_idx % 100 == 0:
                        logger.info("  %d timesteps written...", t_idx)

                if t_idx < ds.shape[0]:
                    ds.resize(t_idx, axis=0)

                str_dt = h5py.string_dtype(encoding="utf-8")
                hf.create_dataset("times", data=timestamps, dtype=str_dt)
                hf.attrs["n_nodes"] = n_nodes
                hf.attrs["n_layers"] = n_layers
                hf.attrs["source"] = str(text_path.name)

        logger.info(
            "Wrote %s: head shape (%d, %d, %d), %d timesteps",
            hdf_path,
            t_idx,
            n_nodes,
            n_layers,
            t_idx,
        )
        return hdf_path

    @staticmethod
    def from_iwfm_hydrograph_text(
        text_file: str | Path,
        hdf_file: str | Path | None = None,
    ) -> Path:
        """Convert an IWFM hydrograph ``.out`` text file to a pyiwfm HDF5 cache.

        Parses the header for hydrograph metadata (IDs, layers, node IDs)
        and reads the full time series into an in-memory NumPy array
        before writing the HDF5. Eager-load is fine here because IWFM
        hydrograph files are typically << headall output.

        Parameters
        ----------
        text_file : str or Path
            Path to the IWFM hydrograph ``.out`` file.
        hdf_file : str or Path or None
            Output HDF5 path. If ``None``, uses ``text_file`` with
            ``.hydrograph_cache.hdf``.

        Returns
        -------
        Path
            Path to the created HDF5 file.
        """
        text_path = Path(text_file)
        hdf_path = (
            Path(hdf_file)
            if hdf_file is not None
            else text_path.with_suffix(".hydrograph_cache.hdf")
        )
        from pyiwfm.io.timeseries_ascii import parse_iwfm_timestamp

        logger.info("Converting hydrograph %s -> %s", text_path, hdf_path)

        with open(text_path) as fh:
            all_lines = fh.readlines()

        header_lines: list[str] = []
        data_start_line = 0
        for i, line in enumerate(all_lines):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("*"):
                header_lines.append(stripped)
            else:
                data_start_line = i
                break

        hydrograph_ids, layers, node_ids = _parse_hydrograph_header(header_lines)

        timestamps: list[str] = []
        rows: list[list[float]] = []
        n_cols = 0

        for i in range(data_start_line, len(all_lines)):
            line = all_lines[i].strip()
            if not line or line.startswith("*"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            try:
                iso_time = parse_iwfm_timestamp(parts[0]).isoformat()
            except (ValueError, IndexError):
                continue

            timestamps.append(iso_time)
            vals: list[float] = []
            for v in parts[1:]:
                try:
                    vals.append(float(v))
                except ValueError:
                    vals.append(float("nan"))
            rows.append(vals)
            if len(vals) > n_cols:
                n_cols = len(vals)

        if not rows:
            raise ValueError(f"No data found in hydrograph file: {text_path}")

        n_timesteps = len(rows)
        logger.info("Parsed %d timesteps, %d columns", n_timesteps, n_cols)

        for r in rows:
            while len(r) < n_cols:
                r.append(float("nan"))

        data = np.array(rows, dtype=np.float64)

        str_dt = h5py.string_dtype(encoding="utf-8")
        with h5py.File(hdf_path, "w") as hf:
            hf.create_dataset(
                "data",
                data=data,
                dtype=np.float64,
                compression="gzip",
                compression_opts=4,
                chunks=(1, n_cols),
            )
            hf.create_dataset("times", data=timestamps, dtype=str_dt)
            if hydrograph_ids:
                hf.create_dataset("hydrograph_ids", data=np.array(hydrograph_ids, dtype=np.int32))
            if layers:
                hf.create_dataset("layers", data=np.array(layers, dtype=np.int32))
            if node_ids:
                hf.create_dataset("node_ids", data=np.array(node_ids, dtype=np.int32))
            hf.attrs["n_columns"] = n_cols
            hf.attrs["n_timesteps"] = n_timesteps
            hf.attrs["source"] = str(text_path.name)

        logger.info(
            "Wrote %s: data shape (%d, %d)",
            hdf_path,
            n_timesteps,
            n_cols,
        )
        return hdf_path


# =============================================================================
# Internal text-parsing helpers (HEADALL + hydrograph)
# =============================================================================


def _skip_headall_titles(fh: TextIO) -> int:
    """Read past the 4 ``*``-prefixed title lines; return number of lines read."""
    lines_read = 0
    title_count = 0
    while title_count < 4:
        line = fh.readline()
        if not line:
            raise ValueError("Unexpected end of file while reading title lines")
        lines_read += 1
        if line.startswith("*"):
            title_count += 1
    return lines_read


def _read_headall_node_header(fh: TextIO) -> str:
    """Skip the ``* NODE`` row and return the ``* TIME node1 node2 ...`` row."""
    header1 = fh.readline()
    header2 = fh.readline()
    if not header1 or not header2:
        raise ValueError("Unexpected end of file while reading header lines")
    return header2.rstrip()


def _parse_headall_node_ids(header_line: str) -> list[int]:
    """Extract node IDs from a HEADALL ``* TIME ...`` header (12-char fields)."""
    remainder = header_line[_HEADALL_TIME_WIDTH:]
    ids: list[int] = []
    for i in range(0, len(remainder), _HEADALL_COL_WIDTH):
        chunk = remainder[i : i + _HEADALL_COL_WIDTH].strip()
        if chunk:
            try:
                ids.append(int(chunk))
            except ValueError:
                pass
    return ids


def _parse_headall_data_row(line: str, n_nodes: int) -> NDArray[np.float64]:
    """Parse a HEADALL data row's ``n_nodes`` values into a float64 array.

    Tries whitespace split first (faster); falls back to fixed-width
    slicing if the split count doesn't match ``n_nodes`` (which can
    happen when negative numbers abut the previous field).
    """
    data_part = line[_HEADALL_TIME_WIDTH:]
    parts = data_part.split()
    if len(parts) >= n_nodes:
        return np.array(parts[:n_nodes], dtype=np.float64)

    values = np.empty(n_nodes, dtype=np.float64)
    for i in range(n_nodes):
        start = i * _HEADALL_COL_WIDTH
        end = start + _HEADALL_COL_WIDTH
        chunk = data_part[start:end].strip()
        values[i] = float(chunk) if chunk else np.nan
    return values


def _count_remaining_lines(fh: TextIO) -> int:
    """Count newlines from current position to EOF without buffering content."""
    pos = fh.tell()
    total = 0
    for _ in fh:
        total += 1
    fh.seek(pos)
    return total


def _parse_hydrograph_header(
    header_lines: list[str],
) -> tuple[list[int], list[int], list[int]]:
    """Extract ``(hydrograph_ids, layers, node_ids)`` from header lines.

    Each input line should already be the ``content`` after stripping and
    starting with ``*`` — this helper does its own ``lstrip("*").strip()``
    so it can accept raw header lines as captured by the converter.
    """
    hydrograph_ids: list[int] = []
    layers: list[int] = []
    node_ids: list[int] = []

    for hline in header_lines:
        content = hline.lstrip("*").strip()
        upper = content.upper()
        if upper.startswith("HYDROGRAPH ID"):
            parts = content.split()
            hydrograph_ids = [int(x) for x in parts[2:]]
        elif upper.startswith("LAYER"):
            parts = content.split()
            layers = [int(x) for x in parts[1:]]
        elif upper.startswith("NODE") or upper.startswith("ELEMENT"):
            parts = content.split()
            int_vals: list[int] = []
            for p in parts[1:]:
                try:
                    int_vals.append(int(p))
                except ValueError:
                    continue
            node_ids = int_vals

    return hydrograph_ids, layers, node_ids
