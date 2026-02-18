"""
Comprehensive tests for pyiwfm.visualization.webapi.area_loader.

Covers:
- ``_iwfm_date_to_iso`` date conversion helper (normal, 24:00, malformed)
- ``LazyAreaDataLoader``: metadata loading, frame access, caching, eviction,
  timeseries slicing, layer range computation, edge cases (missing file,
  missing dataset, 2D dataset)
- ``AreaDataManager``: loader management, snapshot aggregation, element
  breakdown, element timeseries, date retrieval (ISO and raw), n_timesteps,
  stale-cache detection, error handling
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")

from pyiwfm.visualization.webapi.area_loader import (  # noqa: E402
    AreaDataManager,
    LazyAreaDataLoader,
    _iwfm_date_to_iso,
)

# =====================================================================
# Helpers: create test HDF5 area files
# =====================================================================


def _make_hdf(
    path: Path,
    label: str = "area",
    n_timesteps: int = 5,
    n_elements: int = 4,
    n_cols: int = 3,
    times: list[str] | None = None,
    element_ids: list[int] | None = None,
    data: np.ndarray | None = None,
    attrs: dict | None = None,
) -> Path:
    """Create a minimal HDF5 area cache for testing."""
    if times is None:
        times = [f"{m:02d}/01/2000_24:00" for m in range(1, 1 + n_timesteps)]
    if element_ids is None:
        element_ids = list(range(1, n_elements + 1))
    if data is None:
        rng = np.random.default_rng(42)
        data = rng.uniform(50, 200, size=(n_timesteps, n_elements, n_cols))

    with h5py.File(path, "w") as f:
        f.create_dataset(
            label,
            data=data,
            dtype=np.float64,
            chunks=(1, n_elements, n_cols),
        )
        str_dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("times", data=times, dtype=str_dt)
        f.create_dataset("element_ids", data=np.array(element_ids, dtype=np.int32))

        f.attrs["n_elements"] = n_elements
        f.attrs["n_cols"] = n_cols
        f.attrs["label"] = label
        if attrs:
            for k, v in attrs.items():
                f.attrs[k] = v

    return path


# =====================================================================
# _iwfm_date_to_iso
# =====================================================================


class TestIwfmDateToIso:
    def test_normal_date(self):
        assert _iwfm_date_to_iso("01/15/2000_12:00") == "2000-01-15"

    def test_end_of_day_24_00(self):
        """``_24:00`` means end of day -> advance to next day."""
        assert _iwfm_date_to_iso("01/31/2000_24:00") == "2000-02-01"

    def test_midnight(self):
        assert _iwfm_date_to_iso("06/15/2005_00:00") == "2005-06-15"

    def test_date_only_no_time(self):
        """Date without underscore/time falls back to 00:00."""
        assert _iwfm_date_to_iso("03/10/1999") == "1999-03-10"

    def test_malformed_returns_as_is(self):
        result = _iwfm_date_to_iso("not-a-date_00:00")
        assert result == "not-a-date_00:00"

    def test_end_of_year(self):
        """24:00 on Dec 31 rolls to Jan 1 of next year."""
        assert _iwfm_date_to_iso("12/31/1999_24:00") == "2000-01-01"

    def test_leap_year(self):
        assert _iwfm_date_to_iso("02/28/2000_24:00") == "2000-02-29"

    def test_non_leap_year(self):
        assert _iwfm_date_to_iso("02/28/2001_24:00") == "2001-03-01"


# =====================================================================
# LazyAreaDataLoader
# =====================================================================


class TestLazyAreaDataLoaderMetadata:
    """Test metadata loading from HDF5."""

    def test_basic_metadata(self, tmp_path):
        hdf = _make_hdf(tmp_path / "area.hdf", n_timesteps=10, n_elements=6, n_cols=4)
        loader = LazyAreaDataLoader(hdf, dataset="area")
        assert loader.n_frames == 10
        assert loader.n_elements == 6
        assert loader.n_cols == 4
        assert len(loader.times) == 10
        assert len(loader.element_ids) == 6

    def test_missing_file(self, tmp_path, caplog):
        """Non-existent file logs a warning and returns empty loader."""
        missing = tmp_path / "missing.hdf"
        with caplog.at_level(logging.WARNING):
            loader = LazyAreaDataLoader(missing)
        assert loader.n_frames == 0
        assert loader.n_elements == 0
        assert "not found" in caplog.text

    def test_missing_dataset(self, tmp_path, caplog):
        """File exists but dataset name is wrong."""
        hdf = _make_hdf(tmp_path / "area.hdf", label="nonponded")
        with caplog.at_level(logging.WARNING):
            loader = LazyAreaDataLoader(hdf, dataset="wrong_name")
        assert loader.n_frames == 0
        assert "not found" in caplog.text

    def test_no_times_dataset(self, tmp_path):
        """HDF5 without 'times' dataset -> empty times list."""
        hdf_path = tmp_path / "no_times.hdf"
        with h5py.File(hdf_path, "w") as f:
            f.create_dataset("area", data=np.zeros((3, 4, 2)))
        loader = LazyAreaDataLoader(hdf_path, dataset="area")
        assert loader.n_frames == 3
        assert loader.times == []

    def test_no_element_ids_dataset(self, tmp_path):
        """HDF5 without 'element_ids' dataset -> empty array."""
        hdf_path = tmp_path / "no_eids.hdf"
        with h5py.File(hdf_path, "w") as f:
            f.create_dataset("area", data=np.zeros((3, 4, 2)))
        loader = LazyAreaDataLoader(hdf_path, dataset="area")
        assert loader.n_frames == 3
        assert len(loader.element_ids) == 0

    def test_2d_dataset_ncols_is_1(self, tmp_path):
        """A 2D dataset (n_timesteps, n_elements) should report n_cols=1."""
        hdf_path = tmp_path / "2d.hdf"
        with h5py.File(hdf_path, "w") as f:
            f.create_dataset("area", data=np.zeros((5, 10)))
        loader = LazyAreaDataLoader(hdf_path, dataset="area")
        assert loader.n_frames == 5
        assert loader.n_elements == 10
        assert loader.n_cols == 1

    def test_times_bytes_vs_str(self, tmp_path):
        """Times stored as bytes are decoded properly."""
        hdf_path = tmp_path / "bytes_times.hdf"
        with h5py.File(hdf_path, "w") as f:
            f.create_dataset("area", data=np.zeros((2, 3, 1)))
            # Store times as bytes (plain ASCII)
            f.create_dataset("times", data=[b"01/01/2000_24:00", b"02/01/2000_24:00"])
        loader = LazyAreaDataLoader(hdf_path, dataset="area")
        assert loader.times == ["01/01/2000_24:00", "02/01/2000_24:00"]

    def test_corrupted_hdf_file(self, tmp_path, caplog):
        """Corrupted file logs error and returns empty loader."""
        bad = tmp_path / "bad.hdf"
        bad.write_bytes(b"not an hdf5 file")
        with caplog.at_level(logging.ERROR):
            loader = LazyAreaDataLoader(bad)
        assert loader.n_frames == 0
        assert "Failed" in caplog.text


class TestLazyAreaDataLoaderGetFrame:
    """Test get_frame access and caching."""

    @pytest.fixture()
    def loader(self, tmp_path):
        data = np.arange(60, dtype=np.float64).reshape(5, 4, 3)
        hdf = _make_hdf(tmp_path / "area.hdf", data=data)
        return LazyAreaDataLoader(hdf, dataset="area", cache_size=3)

    def test_get_frame_shape(self, loader):
        frame = loader.get_frame(0)
        assert frame.shape == (4, 3)
        assert frame.dtype == np.float64

    def test_get_frame_values(self, loader):
        frame = loader.get_frame(0)
        expected = np.arange(12, dtype=np.float64).reshape(4, 3)
        np.testing.assert_array_equal(frame, expected)

    def test_get_frame_caching(self, loader):
        f0 = loader.get_frame(0)
        f0_again = loader.get_frame(0)
        assert f0 is f0_again  # same object from cache

    def test_get_frame_negative_index(self, loader):
        with pytest.raises(IndexError, match="out of range"):
            loader.get_frame(-1)

    def test_get_frame_out_of_range(self, loader):
        with pytest.raises(IndexError, match="out of range"):
            loader.get_frame(5)

    def test_cache_eviction(self, loader):
        """Cache with size 3: adding 4th evicts oldest."""
        loader.get_frame(0)
        loader.get_frame(1)
        loader.get_frame(2)
        assert len(loader._cache) == 3
        loader.get_frame(3)
        assert len(loader._cache) == 3
        assert 0 not in loader._cache  # oldest evicted
        assert 3 in loader._cache

    def test_cache_move_to_end_on_hit(self, loader):
        """Accessing a cached frame moves it to end (LRU)."""
        loader.get_frame(0)
        loader.get_frame(1)
        loader.get_frame(2)
        # Access 0 again so it becomes most-recent
        loader.get_frame(0)
        # Now adding a new frame should evict 1, not 0
        loader.get_frame(3)
        assert 0 in loader._cache
        assert 1 not in loader._cache

    def test_clear_cache(self, loader):
        loader.get_frame(0)
        loader.get_frame(1)
        assert len(loader._cache) == 2
        loader.clear_cache()
        assert len(loader._cache) == 0

    def test_1d_frame_reshaped(self, tmp_path):
        """2D dataset produces 1D slice; get_frame reshapes to (n, 1)."""
        hdf_path = tmp_path / "2d.hdf"
        data = np.arange(15, dtype=np.float64).reshape(5, 3)
        with h5py.File(hdf_path, "w") as f:
            f.create_dataset("area", data=data)
        loader = LazyAreaDataLoader(hdf_path, dataset="area")
        frame = loader.get_frame(0)
        assert frame.shape == (3, 1)
        np.testing.assert_array_equal(frame.ravel(), [0.0, 1.0, 2.0])


class TestLazyAreaDataLoaderTimeseries:
    def test_get_element_timeseries(self, tmp_path):
        data = np.arange(60, dtype=np.float64).reshape(5, 4, 3)
        hdf = _make_hdf(tmp_path / "area.hdf", data=data)
        loader = LazyAreaDataLoader(hdf, dataset="area")
        ts = loader.get_element_timeseries(1)  # element index 1
        assert ts.shape == (5, 3)
        # First frame, element 1: values 3,4,5
        np.testing.assert_array_equal(ts[0], [3.0, 4.0, 5.0])


class TestLazyAreaDataLoaderLayerRange:
    def test_basic_layer_range(self, tmp_path):
        data = np.ones((10, 5, 2), dtype=np.float64) * 100.0
        hdf = _make_hdf(tmp_path / "area.hdf", data=data, n_timesteps=10, n_elements=5, n_cols=2)
        loader = LazyAreaDataLoader(hdf, dataset="area")
        lo, hi, n = loader.get_layer_range(col=-1, max_frames=5)
        assert lo > 0
        assert hi >= lo
        assert n <= 5

    def test_layer_range_specific_column(self, tmp_path):
        data = np.ones((4, 3, 2), dtype=np.float64)
        data[:, :, 0] = 10.0
        data[:, :, 1] = 50.0
        hdf = _make_hdf(tmp_path / "area.hdf", data=data, n_timesteps=4, n_elements=3, n_cols=2)
        loader = LazyAreaDataLoader(hdf, dataset="area")
        lo, hi, _ = loader.get_layer_range(col=0)
        assert lo == pytest.approx(10.0)
        assert hi == pytest.approx(10.0)

    def test_layer_range_empty_loader(self, tmp_path):
        """Zero frames returns (0.0, 1.0, 0)."""
        missing = tmp_path / "missing.hdf"
        loader = LazyAreaDataLoader(missing)
        assert loader.n_frames == 0
        lo, hi, n = loader.get_layer_range()
        assert (lo, hi, n) == (0.0, 1.0, 0)

    def test_layer_range_all_zeros(self, tmp_path):
        """When all values are 0 (none > 0), returns (0.0, 1.0, n)."""
        data = np.zeros((3, 4, 2), dtype=np.float64)
        hdf = _make_hdf(tmp_path / "area.hdf", data=data, n_timesteps=3, n_elements=4, n_cols=2)
        loader = LazyAreaDataLoader(hdf, dataset="area")
        lo, hi, n = loader.get_layer_range()
        assert lo == 0.0
        assert hi == 1.0
        assert n == 3

    def test_layer_range_max_frames_exceeds_total(self, tmp_path):
        """max_frames larger than n_frames -> scans all frames."""
        data = np.ones((3, 2, 1), dtype=np.float64) * 50.0
        hdf = _make_hdf(tmp_path / "area.hdf", data=data, n_timesteps=3, n_elements=2, n_cols=1)
        loader = LazyAreaDataLoader(hdf, dataset="area")
        lo, hi, n = loader.get_layer_range(max_frames=100)
        assert n == 3

    def test_layer_range_max_frames_zero(self, tmp_path):
        """max_frames=0 scans all frames."""
        data = np.ones((4, 2, 1), dtype=np.float64) * 25.0
        hdf = _make_hdf(tmp_path / "area.hdf", data=data, n_timesteps=4, n_elements=2, n_cols=1)
        loader = LazyAreaDataLoader(hdf, dataset="area")
        lo, hi, n = loader.get_layer_range(max_frames=0)
        assert n == 4


# =====================================================================
# AreaDataManager
# =====================================================================


def _make_manager_with_loaders(
    tmp_path: Path,
    *,
    nonponded: bool = False,
    ponded: bool = False,
    urban: bool = False,
    native: bool = False,
    n_timesteps: int = 5,
    n_elements: int = 4,
) -> AreaDataManager:
    """Create an AreaDataManager with directly-assigned LazyAreaDataLoaders."""
    mgr = AreaDataManager()
    eids = list(range(1, n_elements + 1))

    configs = [
        ("nonponded", nonponded, 3),
        ("ponded", ponded, 2),
        ("urban", urban, 1),
        ("native", native, 2),
    ]
    for label, enabled, n_cols in configs:
        if not enabled:
            continue
        hdf_path = tmp_path / f"{label}_area.hdf"
        rng = np.random.default_rng(hash(label) % (2**31))
        data = rng.uniform(10, 100, size=(n_timesteps, n_elements, n_cols))
        _make_hdf(
            hdf_path,
            label=label,
            data=data,
            n_timesteps=n_timesteps,
            n_elements=n_elements,
            n_cols=n_cols,
            element_ids=eids,
        )
        loader = LazyAreaDataLoader(hdf_path, dataset=label)
        setattr(mgr, label, loader)

    return mgr


class TestAreaDataManagerInit:
    def test_default_none(self):
        mgr = AreaDataManager()
        assert mgr.nonponded is None
        assert mgr.ponded is None
        assert mgr.urban is None
        assert mgr.native is None


class TestAreaDataManagerLoaders:
    def test_loaders_empty(self):
        mgr = AreaDataManager()
        assert mgr._loaders() == []

    def test_loaders_with_some_set(self, tmp_path):
        mgr = _make_manager_with_loaders(tmp_path, nonponded=True, urban=True)
        loaders = mgr._loaders()
        labels = [lbl for lbl, _ in loaders]
        assert "nonponded" in labels
        assert "urban" in labels
        assert "ponded" not in labels
        assert "native" not in labels

    def test_loaders_skips_zero_frame_loader(self, tmp_path):
        """Loaders with n_frames=0 are excluded."""
        mgr = AreaDataManager()
        missing = tmp_path / "missing.hdf"
        loader = LazyAreaDataLoader(missing)
        assert loader.n_frames == 0
        mgr.nonponded = loader
        assert mgr._loaders() == []


class TestAreaDataManagerNTimesteps:
    def test_n_timesteps_empty(self):
        mgr = AreaDataManager()
        assert mgr.n_timesteps == 0

    def test_n_timesteps_with_data(self, tmp_path):
        mgr = _make_manager_with_loaders(tmp_path, ponded=True, n_timesteps=7)
        assert mgr.n_timesteps == 7


class TestAreaDataManagerGetDates:
    def test_get_dates_iso(self, tmp_path):
        mgr = _make_manager_with_loaders(tmp_path, nonponded=True, n_timesteps=3)
        dates = mgr.get_dates(iso=True)
        assert len(dates) == 3
        # ISO format: YYYY-MM-DD
        for d in dates:
            assert "-" in d
            parts = d.split("-")
            assert len(parts) == 3
            assert len(parts[0]) == 4  # year

    def test_get_dates_raw(self, tmp_path):
        mgr = _make_manager_with_loaders(tmp_path, nonponded=True, n_timesteps=3)
        dates = mgr.get_dates(iso=False)
        assert len(dates) == 3
        # Raw IWFM format: MM/DD/YYYY_HH:MM
        for d in dates:
            assert "/" in d

    def test_get_dates_empty(self):
        mgr = AreaDataManager()
        assert mgr.get_dates() == []

    def test_get_dates_loader_with_no_times(self, tmp_path):
        """Loader with data but no times dataset -> falls through, returns []."""
        mgr = AreaDataManager()
        hdf_path = tmp_path / "no_times.hdf"
        with h5py.File(hdf_path, "w") as f:
            f.create_dataset("test", data=np.zeros((3, 2, 1)))
        loader = LazyAreaDataLoader(hdf_path, dataset="test")
        assert loader.n_frames == 3
        assert loader.times == []
        mgr.nonponded = loader
        # _loaders returns the loader (n_frames > 0), but times is empty
        assert mgr.get_dates() == []


class TestAreaDataManagerGetSnapshot:
    def test_snapshot_nonponded_only(self, tmp_path):
        data = np.array([[[100.0, 50.0, 30.0], [200.0, 10.0, 5.0]]])
        hdf = _make_hdf(
            tmp_path / "nonponded.hdf",
            label="nonponded",
            data=data,
            n_timesteps=1,
            n_elements=2,
            n_cols=3,
            element_ids=[1, 2],
        )
        mgr = AreaDataManager()
        mgr.nonponded = LazyAreaDataLoader(hdf, dataset="nonponded")
        snap = mgr.get_snapshot(0)
        assert 1 in snap
        assert 2 in snap
        # All goes to agricultural
        assert snap[1]["fractions"]["agricultural"] == pytest.approx(1.0)
        assert snap[1]["total_area"] == pytest.approx(180.0)  # 100+50+30
        assert snap[1]["dominant"] == "agricultural"

    def test_snapshot_all_types(self, tmp_path):
        mgr = _make_manager_with_loaders(
            tmp_path,
            nonponded=True,
            ponded=True,
            urban=True,
            native=True,
            n_timesteps=3,
            n_elements=2,
        )
        snap = mgr.get_snapshot(0)
        assert len(snap) == 2
        for eid in [1, 2]:
            assert "fractions" in snap[eid]
            assert "dominant" in snap[eid]
            assert "total_area" in snap[eid]
            fracs = snap[eid]["fractions"]
            total_frac = sum(fracs.values())
            assert total_frac == pytest.approx(1.0, abs=0.01)

    def test_snapshot_timestep_exceeds_some_loaders(self, tmp_path):
        """If timestep exceeds a loader's n_frames, that loader is skipped."""
        mgr = AreaDataManager()
        # nonponded: 2 timesteps
        hdf_np = _make_hdf(
            tmp_path / "np.hdf",
            label="nonponded",
            n_timesteps=2,
            n_elements=2,
            n_cols=1,
            data=np.ones((2, 2, 1)) * 50.0,
        )
        # urban: 5 timesteps
        hdf_u = _make_hdf(
            tmp_path / "u.hdf",
            label="urban",
            n_timesteps=5,
            n_elements=2,
            n_cols=1,
            data=np.ones((5, 2, 1)) * 30.0,
        )
        mgr.nonponded = LazyAreaDataLoader(hdf_np, dataset="nonponded")
        mgr.urban = LazyAreaDataLoader(hdf_u, dataset="urban")

        # Timestep 3: nonponded skipped, urban used
        snap = mgr.get_snapshot(3)
        assert len(snap) == 2
        # Only urban contributes
        for eid in [1, 2]:
            assert snap[eid]["fractions"]["urban"] == pytest.approx(1.0)
            assert snap[eid]["fractions"]["agricultural"] == pytest.approx(0.0)

    def test_snapshot_empty_manager(self):
        mgr = AreaDataManager()
        snap = mgr.get_snapshot(0)
        assert snap == {}

    def test_snapshot_element_id_from_index_fallback(self, tmp_path):
        """When eids array is shorter than n_elements, uses 1-based index."""
        hdf_path = tmp_path / "short_eids.hdf"
        data = np.ones((1, 3, 1), dtype=np.float64) * 100.0
        with h5py.File(hdf_path, "w") as f:
            f.create_dataset("nonponded", data=data)
            str_dt = h5py.string_dtype(encoding="utf-8")
            f.create_dataset("times", data=["01/01/2000_24:00"], dtype=str_dt)
            # Only 2 element IDs for 3 elements
            f.create_dataset("element_ids", data=np.array([10, 20], dtype=np.int32))
            f.attrs["n_elements"] = 3
            f.attrs["n_cols"] = 1
        loader = LazyAreaDataLoader(hdf_path, dataset="nonponded")
        mgr = AreaDataManager()
        mgr.nonponded = loader

        snap = mgr.get_snapshot(0)
        # Elements 10, 20 come from eids; element at index 2 has no eid -> uses i+1 = 3
        assert 10 in snap
        assert 20 in snap
        assert 3 in snap

    def test_snapshot_zero_total_area(self, tmp_path):
        """When total area is 0, fractions are 0 and dominant is 'unknown'."""
        data = np.zeros((1, 2, 1), dtype=np.float64)
        hdf = _make_hdf(
            tmp_path / "zero.hdf",
            label="nonponded",
            data=data,
            n_timesteps=1,
            n_elements=2,
            n_cols=1,
            element_ids=[1, 2],
        )
        mgr = AreaDataManager()
        mgr.nonponded = LazyAreaDataLoader(hdf, dataset="nonponded")
        snap = mgr.get_snapshot(0)
        assert snap[1]["dominant"] == "unknown"
        assert snap[1]["total_area"] == 0.0
        assert snap[1]["fractions"]["agricultural"] == 0.0


class TestAreaDataManagerGetElementBreakdown:
    def test_basic_breakdown(self, tmp_path):
        data = np.array([[[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]])
        hdf = _make_hdf(
            tmp_path / "np.hdf",
            label="nonponded",
            data=data,
            n_timesteps=1,
            n_elements=2,
            n_cols=3,
            element_ids=[1, 2],
        )
        mgr = AreaDataManager()
        mgr.nonponded = LazyAreaDataLoader(hdf, dataset="nonponded")
        bd = mgr.get_element_breakdown(element_id=2, timestep=0)
        assert "nonponded" in bd
        assert bd["nonponded"] == [40.0, 50.0, 60.0]

    def test_breakdown_element_not_found(self, tmp_path):
        data = np.ones((1, 2, 1), dtype=np.float64) * 10.0
        hdf = _make_hdf(
            tmp_path / "np.hdf",
            label="nonponded",
            data=data,
            n_timesteps=1,
            n_elements=2,
            n_cols=1,
            element_ids=[1, 2],
        )
        mgr = AreaDataManager()
        mgr.nonponded = LazyAreaDataLoader(hdf, dataset="nonponded")
        bd = mgr.get_element_breakdown(element_id=999)
        assert bd == {}

    def test_breakdown_timestep_exceeds(self, tmp_path):
        """Timestep beyond loader's frames -> that loader is skipped."""
        data = np.ones((2, 2, 1), dtype=np.float64) * 10.0
        hdf = _make_hdf(
            tmp_path / "np.hdf",
            label="nonponded",
            data=data,
            n_timesteps=2,
            n_elements=2,
            n_cols=1,
            element_ids=[1, 2],
        )
        mgr = AreaDataManager()
        mgr.nonponded = LazyAreaDataLoader(hdf, dataset="nonponded")
        bd = mgr.get_element_breakdown(element_id=1, timestep=10)
        assert bd == {}

    def test_breakdown_multiple_types(self, tmp_path):
        mgr = _make_manager_with_loaders(
            tmp_path, nonponded=True, urban=True, n_elements=3, n_timesteps=2
        )
        bd = mgr.get_element_breakdown(element_id=2, timestep=0)
        assert "nonponded" in bd
        assert "urban" in bd
        assert len(bd["nonponded"]) == 3  # nonponded has 3 cols
        assert len(bd["urban"]) == 1  # urban has 1 col


class TestAreaDataManagerGetElementTimeseries:
    def test_basic_timeseries(self, tmp_path):
        mgr = _make_manager_with_loaders(tmp_path, nonponded=True, n_timesteps=5, n_elements=3)
        ts = mgr.get_element_timeseries(element_id=2)
        assert ts["element_id"] == 2
        assert "dates" in ts
        assert "nonponded" in ts
        assert ts["nonponded"]["n_cols"] == 3
        assert len(ts["nonponded"]["areas"]) == 5

    def test_timeseries_element_not_found(self, tmp_path):
        mgr = _make_manager_with_loaders(tmp_path, nonponded=True, n_timesteps=3, n_elements=2)
        ts = mgr.get_element_timeseries(element_id=999)
        assert ts["element_id"] == 999
        assert "nonponded" not in ts

    def test_timeseries_multiple_types(self, tmp_path):
        mgr = _make_manager_with_loaders(
            tmp_path, nonponded=True, urban=True, native=True, n_timesteps=4, n_elements=2
        )
        ts = mgr.get_element_timeseries(element_id=1)
        assert "nonponded" in ts
        assert "urban" in ts
        assert "native" in ts


# =====================================================================
# AreaDataManager.load_from_rootzone
# =====================================================================


class TestAreaDataManagerLoadFromRootzone:
    def _make_rootzone_mock(self, **area_files):
        """Create a mock RootZone with specified area file attributes."""
        rz = MagicMock()
        for attr in [
            "nonponded_area_file",
            "ponded_area_file",
            "urban_area_file",
            "native_area_file",
        ]:
            setattr(rz, attr, area_files.get(attr, None))
        return rz

    def test_load_nonponded_and_native(self, tmp_path):
        """Integration: convert text files and create lazy loaders."""
        # Write minimal area text files
        np_file = tmp_path / "np_area.dat"
        native_file = tmp_path / "nv_area.dat"
        _write_minimal_area_file(np_file, n_elements=3, n_crops=2, n_timesteps=2)
        _write_minimal_area_file(native_file, n_elements=3, n_crops=1, n_timesteps=2)

        rz = self._make_rootzone_mock(
            nonponded_area_file=str(np_file),
            native_area_file=str(native_file),
        )
        mgr = AreaDataManager()
        mgr.load_from_rootzone(rz, tmp_path)

        assert mgr.nonponded is not None
        assert mgr.native is not None
        assert mgr.ponded is None
        assert mgr.urban is None
        assert mgr.nonponded.n_frames == 2
        assert mgr.native.n_frames == 2

    def test_skip_missing_attribute(self, tmp_path):
        """Area files set to None are skipped."""
        rz = self._make_rootzone_mock()  # all None
        mgr = AreaDataManager()
        mgr.load_from_rootzone(rz, tmp_path)
        assert mgr.nonponded is None
        assert mgr.ponded is None

    def test_skip_nonexistent_source(self, tmp_path, caplog):
        """Area files pointing to non-existent paths log warning."""
        rz = self._make_rootzone_mock(
            nonponded_area_file=str(tmp_path / "does_not_exist.dat"),
        )
        mgr = AreaDataManager()
        with caplog.at_level(logging.WARNING):
            mgr.load_from_rootzone(rz, tmp_path)
        assert mgr.nonponded is None
        assert "does not exist" in caplog.text

    def test_stale_cache_detection(self, tmp_path):
        """Cache with n_elements <= 1 is reconverted."""
        np_file = tmp_path / "np_area.dat"
        _write_minimal_area_file(np_file, n_elements=3, n_crops=2, n_timesteps=2)

        # Pre-create a stale HDF5 cache with n_elements=1
        stale_hdf = tmp_path / "nonponded_area_cache.hdf"
        with h5py.File(stale_hdf, "w") as f:
            f.create_dataset("nonponded", data=np.zeros((1, 1, 2)))
            f.attrs["n_elements"] = 1

        rz = self._make_rootzone_mock(nonponded_area_file=str(np_file))
        mgr = AreaDataManager()
        mgr.load_from_rootzone(rz, tmp_path)

        # Should have reconverted to get all 3 elements
        assert mgr.nonponded is not None
        assert mgr.nonponded.n_elements == 3

    def test_uses_existing_cache(self, tmp_path, caplog):
        """Existing up-to-date cache is reused without reconversion."""
        np_file = tmp_path / "np_area.dat"
        _write_minimal_area_file(np_file, n_elements=3, n_crops=2, n_timesteps=2)

        # Convert once normally
        from pyiwfm.io.area_converter import convert_area_to_hdf

        cache_hdf = tmp_path / "nonponded_area_cache.hdf"
        convert_area_to_hdf(np_file, cache_hdf, label="nonponded")

        rz = self._make_rootzone_mock(nonponded_area_file=str(np_file))
        mgr = AreaDataManager()

        # Ensure the cache appears newer than the source
        import os
        import time

        time.sleep(0.05)
        os.utime(cache_hdf, None)  # touch to update mtime

        with caplog.at_level(logging.INFO):
            mgr.load_from_rootzone(rz, tmp_path)

        assert mgr.nonponded is not None
        assert mgr.nonponded.n_elements == 3
        # Should use cached version, not reconvert
        assert "Using cached HDF5" in caplog.text

    def test_conversion_error_handled(self, tmp_path, caplog):
        """If conversion raises, the error is logged and the loader stays None."""
        np_file = tmp_path / "np_area.dat"
        _write_minimal_area_file(np_file, n_elements=2, n_crops=1, n_timesteps=1)

        rz = self._make_rootzone_mock(nonponded_area_file=str(np_file))
        mgr = AreaDataManager()

        with patch(
            "pyiwfm.io.area_converter.convert_area_to_hdf",
            side_effect=RuntimeError("converter exploded"),
        ):
            with caplog.at_level(logging.ERROR):
                mgr.load_from_rootzone(rz, tmp_path)

        assert mgr.nonponded is None
        assert "converter exploded" in caplog.text

    def test_stale_cache_corrupt_hdf(self, tmp_path):
        """Corrupt HDF5 cache triggers reconversion."""
        np_file = tmp_path / "np_area.dat"
        _write_minimal_area_file(np_file, n_elements=2, n_crops=1, n_timesteps=1)

        # Write a corrupt cache
        stale = tmp_path / "nonponded_area_cache.hdf"
        stale.write_bytes(b"not an hdf5 file")

        rz = self._make_rootzone_mock(nonponded_area_file=str(np_file))
        mgr = AreaDataManager()
        mgr.load_from_rootzone(rz, tmp_path)

        # Should have reconverted successfully
        assert mgr.nonponded is not None
        assert mgr.nonponded.n_elements == 2


# =====================================================================
# Helper for writing minimal area text files (for load_from_rootzone tests)
# =====================================================================


def _write_minimal_area_file(
    path: Path,
    n_elements: int = 3,
    n_crops: int = 2,
    n_timesteps: int = 2,
) -> None:
    """Write a minimal IWFM area text file (date on every row)."""
    lines: list[str] = ["C  Area data file"]
    cols = "  ".join(str(i) for i in range(1, n_crops + 1))
    lines.append(f"   {cols}   / column pointers")
    lines.append("   1.0   / FACTARL")
    lines.append("      / DSSFL")

    for ts_idx in range(n_timesteps):
        date = f"{ts_idx + 1:02d}/01/2000_24:00"
        for eid in range(1, n_elements + 1):
            vals = "  ".join(f"{100.0 + eid + ts_idx * 10:.1f}" for _ in range(n_crops))
            lines.append(f"   {date}   {eid}   {vals}")
    path.write_text("\n".join(lines) + "\n")
