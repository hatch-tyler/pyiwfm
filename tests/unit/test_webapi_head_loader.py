"""Unit tests for LazyHeadDataLoader (head_loader.py)."""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest

from pyiwfm.io.head_loader import LazyHeadDataLoader

# ---------------------------------------------------------------------------
# Helpers -- real HDF5 fixtures
# ---------------------------------------------------------------------------

_rng = np.random.default_rng(42)


def _create_pyiwfm_hdf5(
    path: Path,
    n_timesteps: int = 10,
    n_nodes: int = 20,
    n_layers: int = 3,
    *,
    include_times: bool = True,
    string_times: bool = False,
) -> Path:
    """Create a pyiwfm-format HDF5 file with a ``head`` dataset.

    Parameters
    ----------
    string_times : bool
        If True, store time strings as native str (not bytes).
    """
    data = _rng.random((n_timesteps, n_nodes, n_layers)).astype(np.float64)
    base = datetime(2020, 1, 1)
    times = [(base + timedelta(days=i)).isoformat() for i in range(n_timesteps)]

    with h5py.File(path, "w") as f:
        f.create_dataset("head", data=data)
        if include_times:
            if string_times:
                # Store as variable-length strings (not bytes)
                dt = h5py.string_dtype()
                f.create_dataset("times", data=times, dtype=dt)
            else:
                f.create_dataset("times", data=[t.encode() for t in times])
    return path


def _create_pyiwfm_hdf5_2d(
    path: Path,
    n_timesteps: int = 5,
    n_nodes: int = 15,
) -> Path:
    """Create a pyiwfm-format HDF5 file with a 2D ``head`` dataset (no layer dim)."""
    data = _rng.random((n_timesteps, n_nodes)).astype(np.float64)
    base = datetime(2020, 6, 1)
    times = [(base + timedelta(days=i)).isoformat().encode() for i in range(n_timesteps)]

    with h5py.File(path, "w") as f:
        f.create_dataset("head", data=data)
        f.create_dataset("times", data=times)
    return path


def _create_iwfm_native_hdf5(
    path: Path,
    n_timesteps: int = 8,
    n_nodes: int = 10,
    n_layers: int = 2,
    *,
    nlayers_in_ds_attrs: bool = False,
    nlayers_in_file_attrs: bool = True,
    include_times: bool = True,
) -> Path:
    """Create an IWFM-native-format HDF5 file with ``GWHeadAtAllNodes``."""
    # IWFM stores data as (n_timesteps, n_nodes * n_layers) with
    # columns ordered: [all_nodes_layer1, all_nodes_layer2, ...]
    flat_data = _rng.random((n_timesteps, n_nodes * n_layers)).astype(np.float64)
    base = datetime(2021, 3, 1)
    times = [(base + timedelta(days=30 * i)).isoformat().encode() for i in range(n_timesteps)]

    with h5py.File(path, "w") as f:
        ds = f.create_dataset("GWHeadAtAllNodes", data=flat_data)
        if nlayers_in_ds_attrs:
            ds.attrs["NLayers"] = n_layers
        if nlayers_in_file_attrs:
            f.attrs["NLayers"] = n_layers
        if include_times:
            f.create_dataset("times", data=times)
    return path


# ---------------------------------------------------------------------------
# Constructor / _load_metadata
# ---------------------------------------------------------------------------


class TestLazyHeadDataLoaderInit:
    """Tests for constructor and _load_metadata."""

    def test_file_not_found_logs_warning(self, tmp_path: Path) -> None:
        loader = LazyHeadDataLoader(tmp_path / "nonexistent.hdf5")
        assert loader.n_frames == 0
        assert loader.times == []
        assert loader.shape == (0, 0)

    def test_pyiwfm_3d_loads_metadata(self, tmp_path: Path) -> None:
        hdf = _create_pyiwfm_hdf5(tmp_path / "head.hdf5", 10, 20, 3)
        loader = LazyHeadDataLoader(hdf)
        assert loader.n_frames == 10
        assert loader.shape == (20, 3)
        assert len(loader.times) == 10
        assert loader.times[0] == datetime(2020, 1, 1)

    def test_pyiwfm_2d_loads_metadata(self, tmp_path: Path) -> None:
        hdf = _create_pyiwfm_hdf5_2d(tmp_path / "head2d.hdf5", 5, 15)
        loader = LazyHeadDataLoader(hdf)
        assert loader.n_frames == 5
        assert loader.shape == (15, 1)
        assert len(loader.times) == 5

    def test_iwfm_native_nlayers_in_file_attrs(self, tmp_path: Path) -> None:
        hdf = _create_iwfm_native_hdf5(
            tmp_path / "native.hdf5",
            n_timesteps=8,
            n_nodes=10,
            n_layers=2,
            nlayers_in_ds_attrs=False,
            nlayers_in_file_attrs=True,
        )
        loader = LazyHeadDataLoader(hdf)
        assert loader.n_frames == 8
        assert loader.shape == (10, 2)
        assert loader._iwfm_native is True

    def test_iwfm_native_nlayers_in_ds_attrs(self, tmp_path: Path) -> None:
        hdf = _create_iwfm_native_hdf5(
            tmp_path / "native_ds.hdf5",
            n_timesteps=4,
            n_nodes=6,
            n_layers=3,
            nlayers_in_ds_attrs=True,
            nlayers_in_file_attrs=False,
        )
        loader = LazyHeadDataLoader(hdf)
        assert loader.n_frames == 4
        assert loader.shape == (6, 3)

    def test_iwfm_native_no_nlayers_attr_defaults_to_1(self, tmp_path: Path) -> None:
        """When NLayers is not stored, default to 1 layer."""
        hdf = _create_iwfm_native_hdf5(
            tmp_path / "native_no_attr.hdf5",
            n_timesteps=3,
            n_nodes=5,
            n_layers=2,
            nlayers_in_ds_attrs=False,
            nlayers_in_file_attrs=False,
        )
        loader = LazyHeadDataLoader(hdf)
        assert loader.n_frames == 3
        # Without NLayers attribute, defaults to 1 layer -> n_nodes = total_columns
        assert loader._n_layers == 1
        assert loader._n_nodes == 10  # 5 * 2 flattened

    def test_missing_dataset_warns(self, tmp_path: Path) -> None:
        hdf = tmp_path / "empty.hdf5"
        with h5py.File(hdf, "w") as f:
            f.create_dataset("something_else", data=[1, 2, 3])
        loader = LazyHeadDataLoader(hdf)
        assert loader.n_frames == 0

    def test_custom_dataset_name(self, tmp_path: Path) -> None:
        hdf = tmp_path / "custom.hdf5"
        data = _rng.random((4, 8, 2)).astype(np.float64)
        times = [(datetime(2022, 1, 1) + timedelta(days=i)).isoformat().encode() for i in range(4)]
        with h5py.File(hdf, "w") as f:
            f.create_dataset("my_head", data=data)
            f.create_dataset("times", data=times)
        loader = LazyHeadDataLoader(hdf, dataset_name="my_head")
        assert loader.n_frames == 4
        assert loader.shape == (8, 2)

    def test_custom_cache_size(self, tmp_path: Path) -> None:
        hdf = _create_pyiwfm_hdf5(tmp_path / "cache.hdf5", 5, 10, 1)
        loader = LazyHeadDataLoader(hdf, cache_size=2)
        assert loader._cache_size == 2

    def test_corrupted_file_logs_error(self, tmp_path: Path) -> None:
        hdf = tmp_path / "corrupt.hdf5"
        hdf.write_bytes(b"not a real hdf5 file content")
        loader = LazyHeadDataLoader(hdf)
        assert loader.n_frames == 0

    def test_h5py_not_installed(self, tmp_path: Path) -> None:
        dummy = tmp_path / "test.hdf5"
        dummy.touch()

        with patch.dict("sys.modules", {"h5py": None}):
            with patch("builtins.__import__", side_effect=ImportError("no h5py")):
                loader = LazyHeadDataLoader.__new__(LazyHeadDataLoader)
                loader._file_path = dummy
                loader._dataset_name = "head"
                loader._cache_size = 50
                loader._cache = OrderedDict()
                loader._times = []
                loader._n_nodes = 0
                loader._n_layers = 0
                loader._n_frames = 0
                loader._iwfm_native = False
                loader._h5file = None
                loader._load_metadata()
                assert loader.n_frames == 0


# ---------------------------------------------------------------------------
# _load_times branches
# ---------------------------------------------------------------------------


class TestLoadTimes:
    """Tests for the _load_times branches."""

    def test_times_as_bytes(self, tmp_path: Path) -> None:
        hdf = _create_pyiwfm_hdf5(tmp_path / "bytes.hdf5", 3, 5, 1, string_times=False)
        loader = LazyHeadDataLoader(hdf)
        assert len(loader.times) == 3
        assert all(isinstance(t, datetime) for t in loader.times)

    def test_times_as_str(self, tmp_path: Path) -> None:
        hdf = _create_pyiwfm_hdf5(tmp_path / "str.hdf5", 3, 5, 1, string_times=True)
        loader = LazyHeadDataLoader(hdf)
        assert len(loader.times) == 3
        assert all(isinstance(t, datetime) for t in loader.times)

    def test_time_in_attrs_branch(self, tmp_path: Path) -> None:
        """When 'times' dataset is absent but 'time' is in file attrs."""
        hdf = tmp_path / "time_attr.hdf5"
        data = _rng.random((4, 6, 2)).astype(np.float64)
        with h5py.File(hdf, "w") as f:
            f.create_dataset("head", data=data)
            f.attrs["time"] = "some_time_info"
        loader = LazyHeadDataLoader(hdf)
        assert loader.n_frames == 4
        # time attr branch sets _times to empty list
        assert loader.times == []

    def test_placeholder_times_generated(self, tmp_path: Path) -> None:
        """When neither 'times' dataset nor 'time' attr exist."""
        hdf = tmp_path / "no_times.hdf5"
        data = _rng.random((3, 10, 2)).astype(np.float64)
        with h5py.File(hdf, "w") as f:
            f.create_dataset("head", data=data)
        loader = LazyHeadDataLoader(hdf)
        assert len(loader.times) == 3
        assert loader.times[0] == datetime(2000, 1, 1)
        assert loader.times[1] == datetime(2000, 1, 2)
        assert loader.times[2] == datetime(2000, 1, 3)


# ---------------------------------------------------------------------------
# _load_frame
# ---------------------------------------------------------------------------


class TestLoadFrame:
    """Tests for _load_frame reading from real HDF5 files."""

    def test_load_frame_pyiwfm_3d(self, tmp_path: Path) -> None:
        hdf = _create_pyiwfm_hdf5(tmp_path / "f3d.hdf5", 5, 10, 3)
        loader = LazyHeadDataLoader(hdf)
        frame = loader._load_frame(0)
        assert frame.shape == (10, 3)
        assert frame.dtype == np.float64

    def test_load_frame_pyiwfm_2d_reshapes_to_2d(self, tmp_path: Path) -> None:
        """A 2D head dataset (n_timesteps, n_nodes) should be reshaped to (n_nodes, 1)."""
        hdf = _create_pyiwfm_hdf5_2d(tmp_path / "f2d.hdf5", 4, 12)
        loader = LazyHeadDataLoader(hdf)
        frame = loader._load_frame(0)
        assert frame.shape == (12, 1)
        assert frame.dtype == np.float64

    def test_load_frame_iwfm_native(self, tmp_path: Path) -> None:
        """IWFM native format should reshape flat data to (n_nodes, n_layers)."""
        n_nodes, n_layers = 10, 2
        hdf = _create_iwfm_native_hdf5(
            tmp_path / "native_f.hdf5",
            n_timesteps=5,
            n_nodes=n_nodes,
            n_layers=n_layers,
            nlayers_in_file_attrs=True,
        )
        loader = LazyHeadDataLoader(hdf)
        frame = loader._load_frame(0)
        assert frame.shape == (n_nodes, n_layers)
        assert frame.dtype == np.float64

    def test_load_frame_iwfm_native_data_order(self, tmp_path: Path) -> None:
        """Verify the layer-by-layer reshape produces correct data ordering."""
        _n_nodes, n_layers = 3, 2
        hdf = tmp_path / "order.hdf5"
        # Construct known data: layer1 nodes=[10,20,30], layer2 nodes=[40,50,60]
        flat = np.array([[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]])  # 1 timestep
        base_time = datetime(2021, 1, 1).isoformat().encode()
        with h5py.File(hdf, "w") as f:
            ds = f.create_dataset("GWHeadAtAllNodes", data=flat)
            ds.attrs["NLayers"] = n_layers
            f.create_dataset("times", data=[base_time])
        loader = LazyHeadDataLoader(hdf)
        frame = loader._load_frame(0)
        # After reshape(n_layers, n_nodes).T -> (n_nodes, n_layers)
        expected = np.array([[10.0, 40.0], [20.0, 50.0], [30.0, 60.0]])
        np.testing.assert_array_equal(frame, expected)


# ---------------------------------------------------------------------------
# get_frame and caching
# ---------------------------------------------------------------------------


class TestGetFrame:
    """Tests for get_frame() and LRU cache behaviour."""

    def test_out_of_range_raises(self, tmp_path: Path) -> None:
        hdf = _create_pyiwfm_hdf5(tmp_path / "oor.hdf5", 5, 10, 2)
        loader = LazyHeadDataLoader(hdf)
        with pytest.raises(IndexError, match="out of range"):
            loader.get_frame(10)

    def test_negative_raises(self, tmp_path: Path) -> None:
        hdf = _create_pyiwfm_hdf5(tmp_path / "neg.hdf5", 5, 10, 2)
        loader = LazyHeadDataLoader(hdf)
        with pytest.raises(IndexError, match="out of range"):
            loader.get_frame(-1)

    def test_cache_hit_returns_same_array(self, tmp_path: Path) -> None:
        hdf = _create_pyiwfm_hdf5(tmp_path / "ch.hdf5", 3, 8, 2)
        loader = LazyHeadDataLoader(hdf)
        first = loader.get_frame(0)
        second = loader.get_frame(0)
        assert first is second  # exact same object from cache

    def test_cache_miss_then_hit(self, tmp_path: Path) -> None:
        hdf = _create_pyiwfm_hdf5(tmp_path / "mh.hdf5", 3, 8, 2)
        loader = LazyHeadDataLoader(hdf)
        assert len(loader._cache) == 0
        _ = loader.get_frame(1)
        assert 1 in loader._cache
        # Second call is a cache hit
        _ = loader.get_frame(1)
        assert len(loader._cache) == 1

    def test_lru_eviction(self, tmp_path: Path) -> None:
        hdf = _create_pyiwfm_hdf5(tmp_path / "evict.hdf5", 10, 5, 1)
        loader = LazyHeadDataLoader(hdf, cache_size=3)

        # Load frames 0, 1, 2 => fills cache
        for i in range(3):
            loader.get_frame(i)
        assert len(loader._cache) == 3

        # Load frame 3 => should evict frame 0 (oldest)
        loader.get_frame(3)
        assert 0 not in loader._cache
        assert 3 in loader._cache
        assert len(loader._cache) == 3

    def test_lru_move_to_end_on_access(self, tmp_path: Path) -> None:
        hdf = _create_pyiwfm_hdf5(tmp_path / "mte.hdf5", 10, 5, 1)
        loader = LazyHeadDataLoader(hdf, cache_size=3)

        loader.get_frame(0)
        loader.get_frame(1)
        loader.get_frame(2)
        # Access frame 0 again => move to end (most recent)
        loader.get_frame(0)
        # Now add frame 3 => frame 1 should be evicted (oldest)
        loader.get_frame(3)
        assert 1 not in loader._cache
        assert 0 in loader._cache  # was refreshed

    def test_evict_if_needed_on_empty_cache(self) -> None:
        loader = LazyHeadDataLoader.__new__(LazyHeadDataLoader)
        loader._cache_size = 5
        loader._cache = OrderedDict()
        loader._evict_if_needed()  # should not raise
        assert len(loader._cache) == 0


# ---------------------------------------------------------------------------
# __getitem__
# ---------------------------------------------------------------------------


class TestGetItem:
    """Tests for __getitem__ (dict-like access)."""

    def test_int_key(self, tmp_path: Path) -> None:
        hdf = _create_pyiwfm_hdf5(tmp_path / "gi.hdf5", 3, 5, 2)
        loader = LazyHeadDataLoader(hdf)
        result = loader[0]
        assert result.shape == (5, 2)

    def test_datetime_key(self, tmp_path: Path) -> None:
        hdf = _create_pyiwfm_hdf5(tmp_path / "gd.hdf5", 3, 5, 2)
        loader = LazyHeadDataLoader(hdf)
        result = loader[datetime(2020, 1, 1)]
        assert result.shape == (5, 2)

    def test_datetime_not_found(self, tmp_path: Path) -> None:
        hdf = _create_pyiwfm_hdf5(tmp_path / "gnf.hdf5", 3, 5, 2)
        loader = LazyHeadDataLoader(hdf)
        with pytest.raises(KeyError, match="not found"):
            loader[datetime(2099, 12, 31)]

    def test_wrong_type_raises(self, tmp_path: Path) -> None:
        hdf = _create_pyiwfm_hdf5(tmp_path / "gwt.hdf5", 3, 5, 2)
        loader = LazyHeadDataLoader(hdf)
        with pytest.raises(TypeError, match="Key must be"):
            loader["bad_key"]  # type: ignore[index]

    def test_float_type_raises(self, tmp_path: Path) -> None:
        hdf = _create_pyiwfm_hdf5(tmp_path / "gft.hdf5", 3, 5, 2)
        loader = LazyHeadDataLoader(hdf)
        with pytest.raises(TypeError):
            loader[3.14]  # type: ignore[index]


# ---------------------------------------------------------------------------
# __len__
# ---------------------------------------------------------------------------


class TestLen:
    """Tests for __len__."""

    def test_len_with_data(self, tmp_path: Path) -> None:
        hdf = _create_pyiwfm_hdf5(tmp_path / "len.hdf5", 7, 10, 2)
        loader = LazyHeadDataLoader(hdf)
        assert len(loader) == 7

    def test_len_no_data(self, tmp_path: Path) -> None:
        loader = LazyHeadDataLoader(tmp_path / "nonexistent.hdf5")
        assert len(loader) == 0


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------


class TestToDict:
    """Tests for to_dict() method."""

    def test_to_dict_loads_all_frames(self, tmp_path: Path) -> None:
        hdf = _create_pyiwfm_hdf5(tmp_path / "td.hdf5", 4, 6, 2)
        loader = LazyHeadDataLoader(hdf)
        result = loader.to_dict()
        assert len(result) == 4
        for t, arr in result.items():
            assert isinstance(t, datetime)
            assert arr.shape == (6, 2)

    def test_to_dict_populates_cache(self, tmp_path: Path) -> None:
        hdf = _create_pyiwfm_hdf5(tmp_path / "tdc.hdf5", 3, 5, 1)
        loader = LazyHeadDataLoader(hdf)
        assert len(loader._cache) == 0
        _ = loader.to_dict()
        assert len(loader._cache) == 3


# ---------------------------------------------------------------------------
# clear_cache
# ---------------------------------------------------------------------------


class TestClearCache:
    """Tests for clear_cache()."""

    def test_clear_cache(self, tmp_path: Path) -> None:
        hdf = _create_pyiwfm_hdf5(tmp_path / "cc.hdf5", 3, 5, 1)
        loader = LazyHeadDataLoader(hdf)
        loader.get_frame(0)
        loader.get_frame(1)
        assert len(loader._cache) == 2
        loader.clear_cache()
        assert len(loader._cache) == 0

    def test_clear_empty_cache(self, tmp_path: Path) -> None:
        hdf = _create_pyiwfm_hdf5(tmp_path / "ce.hdf5", 2, 5, 1)
        loader = LazyHeadDataLoader(hdf)
        loader.clear_cache()  # should not raise
        assert len(loader._cache) == 0


# ---------------------------------------------------------------------------
# get_layer_range
# ---------------------------------------------------------------------------


class TestGetLayerRange:
    """Tests for get_layer_range() robust min/max computation."""

    def test_basic_range(self, tmp_path: Path) -> None:
        """Compute range over known data."""
        hdf = tmp_path / "range.hdf5"
        n_ts, n_nd, n_ly = 5, 20, 2
        # Create data with known range
        data = np.full((n_ts, n_nd, n_ly), 100.0)
        data[:, :, 0] = np.linspace(50.0, 150.0, n_nd)  # layer 1: 50..150
        data[:, :, 1] = np.linspace(10.0, 80.0, n_nd)  # layer 2: 10..80
        times = [
            (datetime(2020, 1, 1) + timedelta(days=i)).isoformat().encode() for i in range(n_ts)
        ]
        with h5py.File(hdf, "w") as f:
            f.create_dataset("head", data=data)
            f.create_dataset("times", data=times)

        loader = LazyHeadDataLoader(hdf)
        lo, hi, n_scanned = loader.get_layer_range(1)  # layer 1 (1-based)
        assert lo < hi
        assert n_scanned == 5

    def test_range_with_nodata(self, tmp_path: Path) -> None:
        """Values <= -9000 should be excluded."""
        hdf = tmp_path / "nodata.hdf5"
        n_ts, n_nd, n_ly = 3, 10, 1
        data = np.full((n_ts, n_nd, n_ly), -9999.0)
        # Only a few valid values
        data[0, 0, 0] = 100.0
        data[1, 1, 0] = 200.0
        times = [
            (datetime(2020, 1, 1) + timedelta(days=i)).isoformat().encode() for i in range(n_ts)
        ]
        with h5py.File(hdf, "w") as f:
            f.create_dataset("head", data=data)
            f.create_dataset("times", data=times)

        loader = LazyHeadDataLoader(hdf)
        lo, hi, n_scanned = loader.get_layer_range(1)
        assert n_scanned == 3
        # Should only see values 100 and 200
        assert lo >= 100.0
        assert hi <= 200.0

    def test_range_all_nodata_returns_defaults(self, tmp_path: Path) -> None:
        """When all values are nodata, return (0.0, 1.0)."""
        hdf = tmp_path / "all_nodata.hdf5"
        n_ts, n_nd, n_ly = 2, 5, 1
        data = np.full((n_ts, n_nd, n_ly), -9999.0)
        times = [
            (datetime(2020, 1, 1) + timedelta(days=i)).isoformat().encode() for i in range(n_ts)
        ]
        with h5py.File(hdf, "w") as f:
            f.create_dataset("head", data=data)
            f.create_dataset("times", data=times)

        loader = LazyHeadDataLoader(hdf)
        lo, hi, n_scanned = loader.get_layer_range(1)
        assert lo == 0.0
        assert hi == 1.0
        assert n_scanned == 2

    def test_range_zero_frames(self, tmp_path: Path) -> None:
        """Loader with 0 frames returns (0.0, 1.0, 0)."""
        loader = LazyHeadDataLoader(tmp_path / "nonexistent.hdf5")
        lo, hi, n_scanned = loader.get_layer_range(1)
        assert lo == 0.0
        assert hi == 1.0
        assert n_scanned == 0

    def test_range_with_max_frames_sampling(self, tmp_path: Path) -> None:
        """When max_frames < total, only a subset is scanned."""
        hdf = tmp_path / "sampled.hdf5"
        n_ts, n_nd, n_ly = 20, 10, 1
        data = _rng.random((n_ts, n_nd, n_ly)).astype(np.float64) * 100
        times = [
            (datetime(2020, 1, 1) + timedelta(days=i)).isoformat().encode() for i in range(n_ts)
        ]
        with h5py.File(hdf, "w") as f:
            f.create_dataset("head", data=data)
            f.create_dataset("times", data=times)

        loader = LazyHeadDataLoader(hdf)
        lo, hi, n_scanned = loader.get_layer_range(1, max_frames=5)
        assert n_scanned <= 5
        assert lo < hi

    def test_range_max_frames_larger_than_total(self, tmp_path: Path) -> None:
        """When max_frames >= total, all frames are scanned."""
        hdf = _create_pyiwfm_hdf5(tmp_path / "all.hdf5", 4, 8, 1)
        loader = LazyHeadDataLoader(hdf)
        lo, hi, n_scanned = loader.get_layer_range(1, max_frames=100)
        assert n_scanned == 4

    def test_range_max_frames_zero_scans_all(self, tmp_path: Path) -> None:
        """max_frames=0 (default) scans all frames."""
        hdf = _create_pyiwfm_hdf5(tmp_path / "zero.hdf5", 6, 8, 1)
        loader = LazyHeadDataLoader(hdf)
        lo, hi, n_scanned = loader.get_layer_range(1, max_frames=0)
        assert n_scanned == 6

    def test_range_layer_out_of_bounds_skipped(self, tmp_path: Path) -> None:
        """If requested layer exceeds data shape, values are skipped."""
        hdf = _create_pyiwfm_hdf5(tmp_path / "oob.hdf5", 3, 10, 2)
        loader = LazyHeadDataLoader(hdf)
        # Layer 5 (1-based) => layer_idx=4, but we only have 2 layers
        lo, hi, n_scanned = loader.get_layer_range(5)
        assert lo == 0.0
        assert hi == 1.0
        assert n_scanned == 3  # frames were scanned but no valid data

    def test_range_custom_percentiles(self, tmp_path: Path) -> None:
        """Test with custom percentile bounds."""
        hdf = tmp_path / "pct.hdf5"
        n_ts, n_nd, n_ly = 3, 100, 1
        # Create data with spread values
        data = np.zeros((n_ts, n_nd, n_ly))
        for t in range(n_ts):
            data[t, :, 0] = np.linspace(0.0, 1000.0, n_nd)
        times = [
            (datetime(2020, 1, 1) + timedelta(days=i)).isoformat().encode() for i in range(n_ts)
        ]
        with h5py.File(hdf, "w") as f:
            f.create_dataset("head", data=data)
            f.create_dataset("times", data=times)

        loader = LazyHeadDataLoader(hdf)
        lo_narrow, hi_narrow, _ = loader.get_layer_range(1, percentile_lo=10, percentile_hi=90)
        lo_wide, hi_wide, _ = loader.get_layer_range(1, percentile_lo=1, percentile_hi=99)
        # Narrow percentiles should give a tighter range
        assert lo_narrow > lo_wide or abs(lo_narrow - lo_wide) < 1
        assert hi_narrow < hi_wide or abs(hi_narrow - hi_wide) < 1


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    """Tests for properties: times, n_frames, shape."""

    def test_times_property(self, tmp_path: Path) -> None:
        hdf = _create_pyiwfm_hdf5(tmp_path / "tp.hdf5", 3, 5, 1)
        loader = LazyHeadDataLoader(hdf)
        times = loader.times
        assert isinstance(times, list)
        assert len(times) == 3
        assert times[0] == datetime(2020, 1, 1)

    def test_n_frames_property(self, tmp_path: Path) -> None:
        hdf = _create_pyiwfm_hdf5(tmp_path / "nfp.hdf5", 7, 5, 1)
        loader = LazyHeadDataLoader(hdf)
        assert loader.n_frames == 7

    def test_shape_property(self, tmp_path: Path) -> None:
        hdf = _create_pyiwfm_hdf5(tmp_path / "sp.hdf5", 3, 12, 4)
        loader = LazyHeadDataLoader(hdf)
        assert loader.shape == (12, 4)


# ---------------------------------------------------------------------------
# Integration tests with IWFM native format
# ---------------------------------------------------------------------------


class TestIWFMNativeIntegration:
    """Integration tests for full IWFM native format loading."""

    def test_native_get_frame(self, tmp_path: Path) -> None:
        hdf = _create_iwfm_native_hdf5(tmp_path / "int_native.hdf5", 5, 10, 2)
        loader = LazyHeadDataLoader(hdf)
        frame = loader.get_frame(0)
        assert frame.shape == (10, 2)
        assert frame.dtype == np.float64

    def test_native_getitem_int(self, tmp_path: Path) -> None:
        hdf = _create_iwfm_native_hdf5(tmp_path / "int_native2.hdf5", 3, 8, 2)
        loader = LazyHeadDataLoader(hdf)
        frame = loader[2]
        assert frame.shape == (8, 2)

    def test_native_to_dict(self, tmp_path: Path) -> None:
        hdf = _create_iwfm_native_hdf5(tmp_path / "int_native3.hdf5", 3, 6, 2)
        loader = LazyHeadDataLoader(hdf)
        d = loader.to_dict()
        assert len(d) == 3

    def test_native_get_layer_range(self, tmp_path: Path) -> None:
        hdf = tmp_path / "int_range.hdf5"
        n_ts, n_nd, n_ly = 4, 10, 2
        # Known data for predictable range
        flat = np.zeros((n_ts, n_nd * n_ly))
        for t in range(n_ts):
            flat[t, :n_nd] = np.linspace(50.0, 150.0, n_nd)  # layer 1
            flat[t, n_nd:] = np.linspace(10.0, 80.0, n_nd)  # layer 2
        times = [
            (datetime(2021, 1, 1) + timedelta(days=30 * i)).isoformat().encode()
            for i in range(n_ts)
        ]
        with h5py.File(hdf, "w") as f:
            ds = f.create_dataset("GWHeadAtAllNodes", data=flat)
            ds.attrs["NLayers"] = n_ly
            f.create_dataset("times", data=times)

        loader = LazyHeadDataLoader(hdf)
        lo, hi, n = loader.get_layer_range(1)
        assert n == 4
        assert lo < hi

    def test_native_no_times_uses_placeholder(self, tmp_path: Path) -> None:
        hdf = _create_iwfm_native_hdf5(
            tmp_path / "int_notime.hdf5",
            n_timesteps=3,
            n_nodes=5,
            n_layers=1,
            include_times=False,
        )
        loader = LazyHeadDataLoader(hdf)
        assert len(loader.times) == 3
        assert loader.times[0] == datetime(2000, 1, 1)
