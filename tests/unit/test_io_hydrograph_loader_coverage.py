"""Unit tests for pyiwfm.io.hydrograph_loader.LazyHydrographDataLoader."""

from __future__ import annotations

from pathlib import Path

import pytest

h5py = pytest.importorskip("h5py")

import numpy as np  # noqa: E402

from pyiwfm.io.hydrograph_loader import LazyHydrographDataLoader  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def hdf_file(tmp_path: Path) -> Path:
    """Create a small HDF5 file matching the hydrograph cache schema."""
    path = tmp_path / "test_hydrograph.hdf"
    n_timesteps = 4
    n_cols = 3

    data = np.array(
        [
            [10.0, 20.0, 30.0],
            [11.0, 21.0, 31.0],
            [12.0, 22.0, 32.0],
            [13.0, 23.0, 33.0],
        ],
        dtype=np.float64,
    )
    times = [
        "2020-01-31T12:00:00",
        "2020-02-29T12:00:00",
        "2020-03-31T12:00:00",
        "2020-04-30T12:00:00",
    ]

    str_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=data, dtype=np.float64)
        f.create_dataset("times", data=times, dtype=str_dt)
        f.create_dataset("hydrograph_ids", data=np.array([1, 2, 3], dtype=np.int32))
        f.create_dataset("layers", data=np.array([1, 1, 2], dtype=np.int32))
        f.create_dataset("node_ids", data=np.array([100, 200, 300], dtype=np.int32))
        f.attrs["n_columns"] = n_cols
        f.attrs["n_timesteps"] = n_timesteps
        f.attrs["source"] = "test.out"

    return path


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    """Tests for metadata property access."""

    def test_n_timesteps(self, hdf_file: Path) -> None:
        loader = LazyHydrographDataLoader(hdf_file)
        assert loader.n_timesteps == 4

    def test_n_columns(self, hdf_file: Path) -> None:
        loader = LazyHydrographDataLoader(hdf_file)
        assert loader.n_columns == 3

    def test_hydrograph_ids(self, hdf_file: Path) -> None:
        loader = LazyHydrographDataLoader(hdf_file)
        assert loader.hydrograph_ids == [1, 2, 3]

    def test_layers(self, hdf_file: Path) -> None:
        loader = LazyHydrographDataLoader(hdf_file)
        assert loader.layers == [1, 1, 2]

    def test_node_ids(self, hdf_file: Path) -> None:
        loader = LazyHydrographDataLoader(hdf_file)
        assert loader.node_ids == [100, 200, 300]

    def test_times(self, hdf_file: Path) -> None:
        loader = LazyHydrographDataLoader(hdf_file)
        assert len(loader.times) == 4
        assert loader.times[0] == "2020-01-31T12:00:00"


# ---------------------------------------------------------------------------
# get_time_series
# ---------------------------------------------------------------------------


class TestGetTimeSeries:
    """Tests for column retrieval."""

    def test_valid_column(self, hdf_file: Path) -> None:
        loader = LazyHydrographDataLoader(hdf_file)
        times, values = loader.get_time_series(0)
        assert len(times) == 4
        assert len(values) == 4
        assert values[0] == pytest.approx(10.0)
        assert values[3] == pytest.approx(13.0)

    def test_last_column(self, hdf_file: Path) -> None:
        loader = LazyHydrographDataLoader(hdf_file)
        times, values = loader.get_time_series(2)
        assert values[0] == pytest.approx(30.0)

    def test_out_of_range_column_returns_empty(self, hdf_file: Path) -> None:
        loader = LazyHydrographDataLoader(hdf_file)
        times, values = loader.get_time_series(99)
        assert times == []
        assert values == []

    def test_negative_column_returns_empty(self, hdf_file: Path) -> None:
        loader = LazyHydrographDataLoader(hdf_file)
        times, values = loader.get_time_series(-1)
        assert times == []
        assert values == []


# ---------------------------------------------------------------------------
# find_column_by_node_id
# ---------------------------------------------------------------------------


class TestFindColumnByNodeId:
    """Tests for node ID lookup."""

    def test_found(self, hdf_file: Path) -> None:
        loader = LazyHydrographDataLoader(hdf_file)
        assert loader.find_column_by_node_id(200) == 1

    def test_not_found(self, hdf_file: Path) -> None:
        loader = LazyHydrographDataLoader(hdf_file)
        assert loader.find_column_by_node_id(999) is None


# ---------------------------------------------------------------------------
# Missing file
# ---------------------------------------------------------------------------


class TestMissingFile:
    """Test graceful handling of missing HDF5."""

    def test_missing_file_does_not_crash(self, tmp_path: Path) -> None:
        loader = LazyHydrographDataLoader(tmp_path / "does_not_exist.hdf")
        assert loader.n_timesteps == 0
        assert loader.n_columns == 0
        assert loader.hydrograph_ids == []
        assert loader.times == []
