"""Unit tests for LazyHeadDataLoader (head_loader.py)."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pyiwfm.visualization.webapi.head_loader import LazyHeadDataLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_h5_file(
    shape: tuple = (10, 100, 3),
    dataset_name: str = "head",
    has_times: bool = True,
):
    """Build a mock h5py.File context manager."""
    mock_ds = MagicMock()
    mock_ds.shape = shape
    mock_ds.ndim = len(shape)
    mock_ds.__getitem__ = MagicMock(
        side_effect=lambda idx: np.random.rand(*shape[1:])
    )

    mock_file = MagicMock()
    mock_file.__contains__ = lambda self, key: key == dataset_name or (key == "times" and has_times)
    mock_file.__getitem__ = MagicMock(
        side_effect=lambda key: mock_ds if key == dataset_name else _make_times_ds(shape[0])
    )
    mock_file.attrs = {}

    return mock_file


def _make_times_ds(n: int):
    """Create a mock times dataset."""
    base = datetime(2020, 1, 1)
    times = [(base + timedelta(days=i)).isoformat().encode() for i in range(n)]
    ds = MagicMock()
    ds.__getitem__ = MagicMock(return_value=times)
    return ds


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestLazyHeadDataLoaderInit:
    """Tests for constructor and _load_metadata."""

    def test_file_not_found_logs_warning(self, tmp_path: Path) -> None:
        loader = LazyHeadDataLoader(tmp_path / "nonexistent.hdf5")
        assert loader.n_frames == 0
        assert loader.times == []

    def test_h5py_not_installed(self, tmp_path: Path) -> None:
        dummy = tmp_path / "test.hdf5"
        dummy.touch()

        with patch.dict("sys.modules", {"h5py": None}):
            with patch("builtins.__import__", side_effect=ImportError("no h5py")):
                loader = LazyHeadDataLoader.__new__(LazyHeadDataLoader)
                loader._file_path = dummy
                loader._dataset_name = "head"
                loader._cache_size = 50
                loader._cache = {}
                loader._times = []
                loader._n_nodes = 0
                loader._n_layers = 0
                loader._n_frames = 0
                loader._iwfm_native = False
                loader._h5file = None
                loader._load_metadata()
                assert loader.n_frames == 0

    @patch("h5py.File")
    def test_3d_shape_loading(self, mock_h5_cls, tmp_path: Path) -> None:
        dummy = tmp_path / "test.hdf5"
        dummy.touch()

        mock_file = _make_mock_h5_file(shape=(10, 100, 3))
        mock_h5_cls.return_value.__enter__ = MagicMock(return_value=mock_file)
        mock_h5_cls.return_value.__exit__ = MagicMock(return_value=False)

        loader = LazyHeadDataLoader(dummy)
        assert loader.n_frames == 10
        assert loader.shape == (100, 3)

    @patch("h5py.File")
    def test_2d_shape_loading(self, mock_h5_cls, tmp_path: Path) -> None:
        dummy = tmp_path / "test.hdf5"
        dummy.touch()

        mock_ds = MagicMock()
        mock_ds.shape = (5, 50)
        mock_ds.ndim = 2

        mock_file = MagicMock()
        mock_file.__contains__ = lambda self, key: key == "head"
        mock_file.__getitem__ = MagicMock(return_value=mock_ds)
        mock_file.attrs = {}

        mock_h5_cls.return_value.__enter__ = MagicMock(return_value=mock_file)
        mock_h5_cls.return_value.__exit__ = MagicMock(return_value=False)

        loader = LazyHeadDataLoader(dummy)
        assert loader.n_frames == 5
        assert loader.shape == (50, 1)

    @patch("h5py.File")
    def test_missing_dataset(self, mock_h5_cls, tmp_path: Path) -> None:
        dummy = tmp_path / "test.hdf5"
        dummy.touch()

        mock_file = MagicMock()
        mock_file.__contains__ = lambda self, key: False
        mock_h5_cls.return_value.__enter__ = MagicMock(return_value=mock_file)
        mock_h5_cls.return_value.__exit__ = MagicMock(return_value=False)

        loader = LazyHeadDataLoader(dummy)
        assert loader.n_frames == 0

    @patch("h5py.File")
    def test_placeholder_times(self, mock_h5_cls, tmp_path: Path) -> None:
        dummy = tmp_path / "test.hdf5"
        dummy.touch()

        mock_ds = MagicMock()
        mock_ds.shape = (3, 10, 2)
        mock_ds.ndim = 3

        mock_file = MagicMock()
        mock_file.__contains__ = lambda self, key: key == "head"
        mock_file.__getitem__ = MagicMock(return_value=mock_ds)
        mock_file.attrs = {}

        mock_h5_cls.return_value.__enter__ = MagicMock(return_value=mock_file)
        mock_h5_cls.return_value.__exit__ = MagicMock(return_value=False)

        loader = LazyHeadDataLoader(dummy)
        assert len(loader.times) == 3
        assert loader.times[0] == datetime(2000, 1, 1)


# ---------------------------------------------------------------------------
# get_frame and caching
# ---------------------------------------------------------------------------


class TestGetFrame:
    """Tests for get_frame() and LRU cache behaviour."""

    def _make_loader(self) -> LazyHeadDataLoader:
        """Create a loader with known state (bypass HDF5)."""
        loader = LazyHeadDataLoader.__new__(LazyHeadDataLoader)
        loader._file_path = Path("fake.hdf5")
        loader._dataset_name = "head"
        loader._cache_size = 3
        loader._cache = {}
        loader._times = [datetime(2020, 1, i + 1) for i in range(5)]
        loader._n_nodes = 10
        loader._n_layers = 2
        loader._n_frames = 5
        loader._h5file = None
        loader._iwfm_native = False
        return loader

    def test_out_of_range_raises(self) -> None:
        loader = self._make_loader()
        with pytest.raises(IndexError):
            loader.get_frame(10)

    def test_negative_raises(self) -> None:
        loader = self._make_loader()
        with pytest.raises(IndexError):
            loader.get_frame(-1)

    def test_cache_hit(self) -> None:
        loader = self._make_loader()
        data = np.ones((10, 2))
        from collections import OrderedDict

        loader._cache = OrderedDict({0: data})
        result = loader.get_frame(0)
        assert np.array_equal(result, data)

    @patch("h5py.File")
    def test_cache_miss_loads_from_disk(self, mock_h5_cls) -> None:
        loader = self._make_loader()
        from collections import OrderedDict

        loader._cache = OrderedDict()

        expected = np.random.rand(10, 2)
        mock_ds = MagicMock()
        mock_ds.__getitem__ = MagicMock(return_value=expected)

        mock_file = MagicMock()
        mock_file.__getitem__ = MagicMock(return_value=mock_ds)
        mock_h5_cls.return_value.__enter__ = MagicMock(return_value=mock_file)
        mock_h5_cls.return_value.__exit__ = MagicMock(return_value=False)

        result = loader.get_frame(0)
        assert result.shape == (10, 2)
        assert 0 in loader._cache

    def test_lru_eviction(self) -> None:
        loader = self._make_loader()
        from collections import OrderedDict

        loader._cache = OrderedDict()
        # Fill cache to max (3)
        for i in range(3):
            loader._cache[i] = np.ones((10, 2)) * i

        # Force eviction by adding a new entry
        loader._evict_if_needed()
        loader._cache[99] = np.ones((10, 2)) * 99

        assert 0 not in loader._cache  # oldest evicted
        assert 99 in loader._cache


# ---------------------------------------------------------------------------
# __getitem__
# ---------------------------------------------------------------------------


class TestGetItem:
    """Tests for __getitem__ (dict-like access)."""

    def _make_loader(self) -> LazyHeadDataLoader:
        loader = LazyHeadDataLoader.__new__(LazyHeadDataLoader)
        loader._file_path = Path("fake.hdf5")
        loader._dataset_name = "head"
        loader._cache_size = 50
        loader._n_nodes = 10
        loader._n_layers = 2
        loader._n_frames = 3
        loader._times = [datetime(2020, 1, i + 1) for i in range(3)]
        loader._iwfm_native = False
        from collections import OrderedDict

        loader._cache = OrderedDict(
            {i: np.ones((10, 2)) * i for i in range(3)}
        )
        loader._h5file = None
        return loader

    def test_int_key(self) -> None:
        loader = self._make_loader()
        result = loader[0]
        assert result is not None

    def test_datetime_key(self) -> None:
        loader = self._make_loader()
        result = loader[datetime(2020, 1, 1)]
        assert result is not None

    def test_datetime_not_found(self) -> None:
        loader = self._make_loader()
        with pytest.raises(KeyError):
            loader[datetime(2099, 1, 1)]

    def test_wrong_type(self) -> None:
        loader = self._make_loader()
        with pytest.raises(TypeError):
            loader["string_key"]  # type: ignore[index]


# ---------------------------------------------------------------------------
# Misc methods
# ---------------------------------------------------------------------------


class TestMiscMethods:
    """Tests for __len__, to_dict, clear_cache."""

    def _make_loader(self) -> LazyHeadDataLoader:
        loader = LazyHeadDataLoader.__new__(LazyHeadDataLoader)
        loader._file_path = Path("fake.hdf5")
        loader._dataset_name = "head"
        loader._cache_size = 50
        loader._n_nodes = 5
        loader._n_layers = 1
        loader._n_frames = 3
        loader._times = [datetime(2020, 1, i + 1) for i in range(3)]
        loader._iwfm_native = False
        from collections import OrderedDict

        loader._cache = OrderedDict(
            {i: np.ones((5, 1)) * i for i in range(3)}
        )
        loader._h5file = None
        return loader

    def test_len(self) -> None:
        loader = self._make_loader()
        assert len(loader) == 3

    def test_shape(self) -> None:
        loader = self._make_loader()
        assert loader.shape == (5, 1)

    def test_to_dict(self) -> None:
        loader = self._make_loader()
        result = loader.to_dict()
        assert len(result) == 3
        assert datetime(2020, 1, 1) in result

    def test_clear_cache(self) -> None:
        loader = self._make_loader()
        assert len(loader._cache) == 3
        loader.clear_cache()
        assert len(loader._cache) == 0
