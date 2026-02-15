"""
Lazy area data loader for IWFM web visualization.

Provides ``LazyAreaDataLoader`` for loading time-varying land-use area data
from HDF5 files without reading all timesteps into memory, and
``AreaDataManager`` which manages all four land-use area file types.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pyiwfm.components.rootzone import RootZone

logger = logging.getLogger(__name__)


def _iwfm_date_to_iso(date_str: str) -> str:
    """Convert IWFM date ``MM/DD/YYYY_HH:MM`` to ISO ``YYYY-MM-DD``.

    Handles ``_24:00`` (end of day) by advancing to the next day.
    """
    from datetime import datetime, timedelta

    # Split date and time on underscore
    parts = date_str.split("_")
    date_part = parts[0]  # MM/DD/YYYY
    time_part = parts[1] if len(parts) > 1 else "00:00"

    try:
        dt = datetime.strptime(date_part, "%m/%d/%Y")
        if time_part == "24:00":
            dt += timedelta(days=1)
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return date_str  # Return as-is if parsing fails


class LazyAreaDataLoader:
    """Lazy HDF5 reader for land-use area data with LRU cache.

    Parameters
    ----------
    file_path : Path or str
        Path to the HDF5 file containing area data.
    dataset : str
        Name of the HDF5 dataset. Default is "area".
    cache_size : int
        Maximum number of timestep frames to cache. Default is 50.
    """

    def __init__(
        self,
        file_path: Path | str,
        dataset: str = "area",
        cache_size: int = 50,
    ) -> None:
        self._file_path = Path(file_path)
        self._dataset = dataset
        self._cache_size = cache_size
        self._cache: OrderedDict[int, NDArray[np.float64]] = OrderedDict()

        self._n_frames = 0
        self._n_elements = 0
        self._n_cols = 0
        self._times: list[str] = []
        self._element_ids: NDArray[np.int32] = np.array([], dtype=np.int32)

        self._load_metadata()

    @property
    def n_frames(self) -> int:
        return self._n_frames

    @property
    def n_elements(self) -> int:
        return self._n_elements

    @property
    def n_cols(self) -> int:
        return self._n_cols

    @property
    def times(self) -> list[str]:
        return self._times

    @property
    def element_ids(self) -> NDArray[np.int32]:
        return self._element_ids

    def _load_metadata(self) -> None:
        """Load metadata from the HDF5 file without loading data."""
        if not self._file_path.exists():
            logger.warning("Area data file not found: %s", self._file_path)
            return

        try:
            with h5py.File(self._file_path, "r") as f:
                if self._dataset not in f:
                    logger.warning(
                        "Dataset '%s' not found in %s",
                        self._dataset,
                        self._file_path,
                    )
                    return

                ds = f[self._dataset]
                self._n_frames = ds.shape[0]
                self._n_elements = ds.shape[1]
                self._n_cols = ds.shape[2] if ds.ndim == 3 else 1

                # Load times
                if "times" in f:
                    raw = f["times"][:]
                    self._times = [
                        t.decode() if isinstance(t, bytes) else t
                        for t in raw
                    ]

                # Load element IDs
                if "element_ids" in f:
                    self._element_ids = f["element_ids"][:].astype(np.int32)

                logger.info(
                    "Area data loaded: %d timesteps, %d elements, %d cols "
                    "from %s",
                    self._n_frames,
                    self._n_elements,
                    self._n_cols,
                    self._file_path.name,
                )
        except Exception as e:
            logger.error("Failed to load area metadata from %s: %s", self._file_path, e)

    def _evict_if_needed(self) -> None:
        while len(self._cache) >= self._cache_size:
            self._cache.popitem(last=False)

    def get_frame(self, frame_idx: int) -> NDArray[np.float64]:
        """Get area data for one timestep.

        Returns array of shape ``(n_elements, n_cols)``.
        """
        if frame_idx < 0 or frame_idx >= self._n_frames:
            raise IndexError(f"Frame {frame_idx} out of range [0, {self._n_frames})")

        if frame_idx in self._cache:
            self._cache.move_to_end(frame_idx)
            return self._cache[frame_idx]

        self._evict_if_needed()
        with h5py.File(self._file_path, "r") as f:
            data = f[self._dataset][frame_idx].astype(np.float64)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        self._cache[frame_idx] = data
        return data

    def get_element_timeseries(self, element_idx: int) -> NDArray[np.float64]:
        """Get all timesteps for one element using HDF5 hyperslab slicing.

        Returns array of shape ``(n_timesteps, n_cols)``.
        """
        with h5py.File(self._file_path, "r") as f:
            data = f[self._dataset][:, element_idx, :].astype(np.float64)
        return data

    def get_layer_range(
        self,
        col: int = -1,
        percentile_lo: float = 2.0,
        percentile_hi: float = 98.0,
        max_frames: int = 50,
    ) -> tuple[float, float, int]:
        """Get percentile range across sampled timesteps.

        Parameters
        ----------
        col : int
            Column index, or -1 to sum all columns (total area).
        percentile_lo, percentile_hi : float
            Percentiles for robust range.
        max_frames : int
            Max number of frames to sample.

        Returns
        -------
        tuple[float, float, int]
            (min_value, max_value, n_frames_scanned)
        """
        total = self._n_frames
        if total == 0:
            return (0.0, 1.0, 0)

        if max_frames > 0 and max_frames < total:
            indices = np.linspace(0, total - 1, max_frames, dtype=int)
            indices = np.unique(indices)
        else:
            indices = np.arange(total)

        all_valid: list[float] = []
        for idx in indices:
            frame = self.get_frame(int(idx))
            if col == -1:
                values = frame.sum(axis=1)
            else:
                values = frame[:, col]
            valid = values[values > 0]
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
        self._cache.clear()


class AreaDataManager:
    """Manages lazy loaders for all 4 land-use area file types."""

    def __init__(self) -> None:
        self.nonponded: LazyAreaDataLoader | None = None
        self.ponded: LazyAreaDataLoader | None = None
        self.urban: LazyAreaDataLoader | None = None
        self.native: LazyAreaDataLoader | None = None

    def load_from_rootzone(self, rz: RootZone, cache_dir: Path) -> None:
        """Convert area files to HDF5 if needed and create lazy loaders."""
        from pyiwfm.io.area_converter import convert_area_to_hdf

        for attr, lbl in [
            ("nonponded_area_file", "nonponded"),
            ("ponded_area_file", "ponded"),
            ("urban_area_file", "urban"),
            ("native_area_file", "native"),
        ]:
            src = getattr(rz, attr, None)
            if src is None:
                logger.debug("Area file '%s' (%s): not wired", lbl, attr)
                continue
            src = Path(src)
            if not src.exists():
                logger.warning(
                    "Area file '%s' (%s): path %s does not exist",
                    lbl, attr, src,
                )
                continue

            hdf = cache_dir / f"{lbl}_area_cache.hdf"
            try:
                needs_convert = not hdf.exists() or hdf.stat().st_mtime < src.stat().st_mtime
                # Also reconvert if cached file has too few elements
                # (detects caches from the pre-fix parser that only read
                # the first element due to missing continuation-row support)
                if not needs_convert:
                    try:
                        with h5py.File(hdf, "r") as check_f:
                            cached_n = check_f.attrs.get("n_elements", 0)
                            if cached_n <= 1:
                                logger.info(
                                    "Stale cache for '%s': only %d element(s), reconverting",
                                    lbl, cached_n,
                                )
                                needs_convert = True
                    except Exception:
                        needs_convert = True

                if needs_convert:
                    logger.info(
                        "Converting %s -> %s (%d bytes)",
                        src.name, hdf.name, src.stat().st_size,
                    )
                    convert_area_to_hdf(src, hdf, label=lbl)
                else:
                    logger.info(
                        "Using cached HDF5 for '%s': %s", lbl, hdf.name,
                    )
                loader = LazyAreaDataLoader(hdf, dataset=lbl)
                setattr(self, lbl, loader)
                logger.info(
                    "Area loader '%s': %d timesteps, %d elements, %d cols",
                    lbl,
                    loader.n_frames,
                    loader.n_elements,
                    loader.n_cols,
                )
            except Exception as e:
                logger.error(
                    "Failed to load area data for '%s': %s", lbl, e,
                    exc_info=True,
                )

    def _loaders(self) -> list[tuple[str, LazyAreaDataLoader]]:
        """Return all non-None loaders with their labels."""
        result: list[tuple[str, LazyAreaDataLoader]] = []
        for lbl in ("nonponded", "ponded", "urban", "native"):
            loader = getattr(self, lbl)
            if loader is not None and loader.n_frames > 0:
                result.append((lbl, loader))
        return result

    def get_snapshot(self, timestep: int) -> dict[int, dict]:
        """Get aggregated land-use data for all elements at one timestep.

        Returns a dict mapping element_id to a dict with keys:
        ``agricultural``, ``urban``, ``native_riparian``, ``water``,
        ``total_area``, ``dominant``.
        """
        elem_data: dict[int, dict[str, float]] = {}

        for lbl, loader in self._loaders():
            if timestep >= loader.n_frames:
                continue
            frame = loader.get_frame(timestep)
            eids = loader.element_ids

            for i in range(loader.n_elements):
                eid = int(eids[i]) if i < len(eids) else i + 1
                if eid not in elem_data:
                    elem_data[eid] = {
                        "agricultural": 0.0,
                        "urban": 0.0,
                        "native_riparian": 0.0,
                        "water": 0.0,
                    }
                total = float(frame[i].sum())
                if lbl in ("nonponded", "ponded"):
                    elem_data[eid]["agricultural"] += total
                elif lbl == "urban":
                    elem_data[eid]["urban"] += total
                elif lbl == "native":
                    elem_data[eid]["native_riparian"] += total

        # Compute fractions + dominant type
        result: dict[int, dict] = {}
        for eid, areas in elem_data.items():
            total = sum(areas.values())
            fractions = {
                k: round(v / total, 4) if total > 0 else 0.0
                for k, v in areas.items()
            }
            dominant = max(areas, key=lambda k: areas[k]) if total > 0 else "unknown"
            result[eid] = {
                "fractions": fractions,
                "dominant": dominant,
                "total_area": round(total, 2),
            }
        return result

    def get_element_breakdown(
        self, element_id: int, timestep: int = 0
    ) -> dict[str, list[float]]:
        """Get per-column area breakdown for one element at one timestep.

        Returns a dict mapping land-use label (e.g. ``"nonponded"``) to a
        list of per-column areas.  Each entry corresponds to a column in
        the HDF5 dataset for that land-use type.
        """
        result: dict[str, list[float]] = {}

        for lbl, loader in self._loaders():
            if timestep >= loader.n_frames:
                continue
            eids = loader.element_ids
            matches = np.where(eids == element_id)[0]
            if len(matches) == 0:
                continue
            elem_idx = int(matches[0])
            frame = loader.get_frame(timestep)
            row = frame[elem_idx]  # shape: (n_cols,)
            result[lbl] = [round(float(v), 4) for v in row]

        return result

    def get_element_timeseries(self, element_id: int) -> dict:
        """Get all-timestep timeseries for one element across all land-use types.

        Returns a dict with keys per land-use type, each containing
        ``areas`` (list of per-timestep values) and column count info.
        """
        result: dict = {"element_id": element_id, "dates": self.get_dates()}

        for lbl, loader in self._loaders():
            eids = loader.element_ids
            matches = np.where(eids == element_id)[0]
            if len(matches) == 0:
                continue
            elem_idx = int(matches[0])
            ts = loader.get_element_timeseries(elem_idx)
            # ts shape: (n_timesteps, n_cols)
            result[lbl] = {
                "n_cols": loader.n_cols,
                "areas": ts.tolist(),
            }

        return result

    def get_dates(self, iso: bool = True) -> list[str]:
        """Return timestep dates from whichever loader is available.

        Parameters
        ----------
        iso : bool
            If True, convert IWFM dates to ISO ``YYYY-MM-DD`` format
            so JavaScript/Plotly can parse them.
        """
        for _lbl, loader in self._loaders():
            if loader.times:
                if iso:
                    return [_iwfm_date_to_iso(d) for d in loader.times]
                return loader.times
        return []

    @property
    def n_timesteps(self) -> int:
        for _lbl, loader in self._loaders():
            return loader.n_frames
        return 0
