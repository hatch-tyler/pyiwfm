"""Unit tests for the unified time series I/O module (pyiwfm.io.timeseries)."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from pyiwfm.io.timeseries import (
    TimeSeriesFileType,
    TimeUnit,
    TimeSeriesMetadata,
    UnifiedTimeSeriesConfig,
    AsciiTimeSeriesAdapter,
    BaseTimeSeriesReader,
    DssTimeSeriesAdapter,
    Hdf5TimeSeriesAdapter,
    UnifiedTimeSeriesReader,
    RecyclingTimeSeriesReader,
    detect_timeseries_format,
    read_timeseries_unified,
    get_timeseries_metadata,
)


# ---------------------------------------------------------------------------
# 2A. Dataclasses & Enums
# ---------------------------------------------------------------------------


class TestTimeSeriesFileType:
    """Tests for the TimeSeriesFileType enum."""

    def test_ascii_value(self) -> None:
        assert TimeSeriesFileType.ASCII.value == "ascii"

    def test_dss_value(self) -> None:
        assert TimeSeriesFileType.DSS.value == "dss"

    def test_hdf5_value(self) -> None:
        assert TimeSeriesFileType.HDF5.value == "hdf5"

    def test_binary_value(self) -> None:
        assert TimeSeriesFileType.BINARY.value == "binary"

    def test_all_members_present(self) -> None:
        names = {m.name for m in TimeSeriesFileType}
        assert names == {"ASCII", "DSS", "HDF5", "BINARY"}


class TestTimeUnit:
    """Tests for the TimeUnit enum."""

    def test_all_values(self) -> None:
        expected = {
            "SECOND": "second",
            "MINUTE": "minute",
            "HOUR": "hour",
            "DAY": "day",
            "WEEK": "week",
            "MONTH": "month",
            "YEAR": "year",
        }
        for name, value in expected.items():
            assert TimeUnit[name].value == value


class TestTimeSeriesMetadata:
    """Tests for the TimeSeriesMetadata dataclass defaults."""

    def test_defaults(self) -> None:
        meta = TimeSeriesMetadata()
        assert meta.file_type == TimeSeriesFileType.ASCII
        assert meta.n_columns == 0
        assert meta.column_ids == []
        assert meta.variable_name == ""
        assert meta.data_unit == ""
        assert meta.time_unit == ""
        assert meta.conversion_factor == 1.0
        assert meta.is_rate_data is False
        assert meta.recycling_interval == 0
        assert meta.start_time is None
        assert meta.end_time is None
        assert meta.n_timesteps == 0
        assert meta.source_path is None

    def test_custom_values(self) -> None:
        meta = TimeSeriesMetadata(
            file_type=TimeSeriesFileType.HDF5,
            n_columns=5,
            conversion_factor=2.5,
            is_rate_data=True,
            recycling_interval=12,
        )
        assert meta.file_type == TimeSeriesFileType.HDF5
        assert meta.n_columns == 5
        assert meta.conversion_factor == 2.5
        assert meta.is_rate_data is True
        assert meta.recycling_interval == 12


class TestUnifiedTimeSeriesConfig:
    """Tests for the UnifiedTimeSeriesConfig dataclass."""

    def test_required_filepath(self) -> None:
        cfg = UnifiedTimeSeriesConfig(filepath=Path("test.dat"))
        assert cfg.filepath == Path("test.dat")

    def test_optional_defaults(self) -> None:
        cfg = UnifiedTimeSeriesConfig(filepath=Path("test.dat"))
        assert cfg.file_type is None
        assert cfg.n_columns is None
        assert cfg.column_ids is None
        assert cfg.data_unit == ""
        assert cfg.time_unit == ""
        assert cfg.conversion_factor == 1.0
        assert cfg.is_rate_data is False
        assert cfg.recycling_interval == 0
        assert cfg.start_filter is None
        assert cfg.end_filter is None
        assert cfg.dss_pathname is None
        assert cfg.hdf5_dataset is None

    def test_all_fields_set(self) -> None:
        start = datetime(2020, 1, 1)
        end = datetime(2020, 12, 31)
        cfg = UnifiedTimeSeriesConfig(
            filepath=Path("pumping.dss"),
            file_type=TimeSeriesFileType.DSS,
            n_columns=3,
            column_ids=[1, 2, 3],
            data_unit="CFS",
            time_unit="day",
            conversion_factor=0.5,
            is_rate_data=True,
            recycling_interval=12,
            start_filter=start,
            end_filter=end,
            dss_pathname="/A/B/C//1DAY/F/",
            hdf5_dataset="/ts/data",
        )
        assert cfg.file_type == TimeSeriesFileType.DSS
        assert cfg.column_ids == [1, 2, 3]
        assert cfg.start_filter == start
        assert cfg.dss_pathname == "/A/B/C//1DAY/F/"


# ---------------------------------------------------------------------------
# 2B. AsciiTimeSeriesAdapter
# ---------------------------------------------------------------------------


class TestAsciiTimeSeriesAdapter:
    """Tests for AsciiTimeSeriesAdapter."""

    def test_read_delegates_to_timeseries_reader(self) -> None:
        """read() should delegate to TimeSeriesReader and return correct tuple."""
        mock_config = MagicMock()
        mock_config.n_columns = 3
        mock_config.column_ids = [1, 2, 3]
        mock_config.factor = 1.0

        fake_times = [datetime(1990, 2, 1), datetime(1990, 3, 1)]
        fake_values = np.array([[100.0, 200.0, 300.0], [110.0, 210.0, 310.0]])

        with patch(
            "pyiwfm.io.timeseries_ascii.TimeSeriesReader"
        ) as MockReader:
            mock_instance = MockReader.return_value
            mock_instance.read.return_value = (fake_times, fake_values, mock_config)

            adapter = AsciiTimeSeriesAdapter()
            times, values, metadata = adapter.read("test.dat")

        np.testing.assert_array_equal(values, fake_values)
        assert metadata.file_type == TimeSeriesFileType.ASCII
        assert metadata.n_columns == 3
        assert metadata.n_timesteps == 2
        assert len(times) == 2

    def test_read_returns_numpy_datetime64_times(self) -> None:
        mock_config = MagicMock()
        mock_config.n_columns = 1
        mock_config.column_ids = [1]
        mock_config.factor = 1.0

        fake_times = [datetime(2000, 6, 15)]
        fake_values = np.array([[42.0]])

        with patch(
            "pyiwfm.io.timeseries_ascii.TimeSeriesReader"
        ) as MockReader:
            mock_instance = MockReader.return_value
            mock_instance.read.return_value = (fake_times, fake_values, mock_config)

            adapter = AsciiTimeSeriesAdapter()
            times, _, _ = adapter.read("test.dat")

        assert times.dtype == np.dtype("datetime64[s]")

    def test_read_with_empty_times(self) -> None:
        """When no data lines, start_time and end_time should be None."""
        mock_config = MagicMock()
        mock_config.n_columns = 2
        mock_config.column_ids = [1, 2]
        mock_config.factor = 1.0

        with patch(
            "pyiwfm.io.timeseries_ascii.TimeSeriesReader"
        ) as MockReader:
            mock_instance = MockReader.return_value
            mock_instance.read.return_value = ([], np.array([]), mock_config)

            adapter = AsciiTimeSeriesAdapter()
            times, values, metadata = adapter.read("empty.dat")

        assert metadata.start_time is None
        assert metadata.end_time is None
        assert metadata.n_timesteps == 0

    def test_read_sets_source_path(self) -> None:
        mock_config = MagicMock()
        mock_config.n_columns = 1
        mock_config.column_ids = [1]
        mock_config.factor = 2.0

        with patch(
            "pyiwfm.io.timeseries_ascii.TimeSeriesReader"
        ) as MockReader:
            mock_instance = MockReader.return_value
            mock_instance.read.return_value = (
                [datetime(2020, 1, 1)],
                np.array([[10.0]]),
                mock_config,
            )

            adapter = AsciiTimeSeriesAdapter()
            _, _, metadata = adapter.read("/some/path/pump.dat")

        assert metadata.source_path == Path("/some/path/pump.dat")
        assert metadata.conversion_factor == 2.0

    def test_read_metadata_parses_ndata_and_factor(self, tmp_path: Path) -> None:
        """read_metadata() should parse NDATA and FACTOR from the file header."""
        content = (
            "C        IWFM comment line\n"
            "   3     / NDATA\n"
            "   1.0   / FACTOR\n"
            "01/31/1990_24:00   100.0   200.0   300.0\n"
            "02/28/1990_24:00   110.0   210.0   310.0\n"
        )
        fp = tmp_path / "ts_test.dat"
        fp.write_text(content)

        adapter = AsciiTimeSeriesAdapter()
        meta = adapter.read_metadata(fp)

        assert meta.n_columns == 3
        assert meta.conversion_factor == 1.0
        assert meta.n_timesteps == 2
        assert meta.column_ids == [1, 2, 3]
        assert meta.file_type == TimeSeriesFileType.ASCII
        assert meta.source_path == fp

    def test_read_metadata_tracks_first_last_timestamps(self, tmp_path: Path) -> None:
        content = (
            "C  header\n"
            "   2     / NDATA\n"
            "   0.5   / FACTOR\n"
            "01/31/1990_24:00   10.0   20.0\n"
            "06/30/1990_24:00   30.0   40.0\n"
            "12/31/1990_24:00   50.0   60.0\n"
        )
        fp = tmp_path / "ts_timestamps.dat"
        fp.write_text(content)

        adapter = AsciiTimeSeriesAdapter()
        meta = adapter.read_metadata(fp)

        # 01/31/1990_24:00 -> 02/01/1990 00:00 (IWFM 24:00 convention)
        assert meta.start_time == datetime(1990, 2, 1)
        # 12/31/1990_24:00 -> 01/01/1991 00:00
        assert meta.end_time == datetime(1991, 1, 1)
        assert meta.n_timesteps == 3
        assert meta.conversion_factor == 0.5

    def test_read_metadata_skips_comment_lines(self, tmp_path: Path) -> None:
        content = (
            "C  comment 1\n"
            "C  comment 2\n"
            "C  comment 3\n"
            "   1     / NDATA\n"
            "C  inline comment\n"
            "   2.5   / FACTOR\n"
            "03/15/2000_12:00   99.9\n"
        )
        fp = tmp_path / "ts_comments.dat"
        fp.write_text(content)

        adapter = AsciiTimeSeriesAdapter()
        meta = adapter.read_metadata(fp)

        assert meta.n_columns == 1
        assert meta.conversion_factor == 2.5
        assert meta.n_timesteps == 1


# ---------------------------------------------------------------------------
# 2C. DssTimeSeriesAdapter
# ---------------------------------------------------------------------------


class TestDssTimeSeriesAdapter:
    """Tests for DssTimeSeriesAdapter."""

    def test_read_raises_import_error_when_dss_missing(self) -> None:
        """read() raises ImportError when pyiwfm.io.dss is not importable."""
        adapter = DssTimeSeriesAdapter()
        with patch.dict("sys.modules", {"pyiwfm.io.dss": None}):
            with pytest.raises(ImportError, match="pydsstools"):
                adapter.read("test.dss")

    def test_read_delegates_to_dss_reader(self) -> None:
        mock_dss_mod = MagicMock()
        fake_times = np.array(
            ["2020-01-01", "2020-02-01"], dtype="datetime64[s]"
        )
        fake_values = np.array([[1.0, 2.0], [3.0, 4.0]])
        mock_dss_mod.DSSTimeSeriesReader.return_value.read.return_value = (
            fake_times,
            fake_values,
        )

        adapter = DssTimeSeriesAdapter()
        with patch.dict("sys.modules", {"pyiwfm.io.dss": mock_dss_mod}):
            times, values, metadata = adapter.read("test.dss", pathname="/A/B/C//1DAY/F/")

        np.testing.assert_array_equal(times, fake_times)
        np.testing.assert_array_equal(values, fake_values)
        assert metadata.file_type == TimeSeriesFileType.DSS
        assert metadata.n_columns == 2
        assert metadata.n_timesteps == 2

    def test_read_metadata_raises_import_error_when_missing(self) -> None:
        adapter = DssTimeSeriesAdapter()
        with patch.dict("sys.modules", {"pyiwfm.io.dss": None}):
            with pytest.raises(ImportError, match="pydsstools"):
                adapter.read_metadata("test.dss")

    def test_read_metadata_returns_catalog_info(self) -> None:
        mock_dss_mod = MagicMock()
        mock_dss_file = MagicMock()
        mock_dss_file.__enter__ = MagicMock(return_value=mock_dss_file)
        mock_dss_file.__exit__ = MagicMock(return_value=False)
        mock_dss_file.catalog.return_value = ["path1", "path2", "path3"]
        mock_dss_mod.DSSFile.return_value = mock_dss_file

        adapter = DssTimeSeriesAdapter()
        with patch.dict("sys.modules", {"pyiwfm.io.dss": mock_dss_mod}):
            meta = adapter.read_metadata("test.dss")

        assert meta.file_type == TimeSeriesFileType.DSS
        assert meta.n_columns == 3
        assert meta.source_path == Path("test.dss")

    def test_read_with_empty_times(self) -> None:
        mock_dss_mod = MagicMock()
        fake_times = np.array([], dtype="datetime64[s]")
        fake_values = np.array([]).reshape(0, 1)
        mock_dss_mod.DSSTimeSeriesReader.return_value.read.return_value = (
            fake_times,
            fake_values,
        )

        adapter = DssTimeSeriesAdapter()
        with patch.dict("sys.modules", {"pyiwfm.io.dss": mock_dss_mod}):
            _, _, meta = adapter.read("test.dss")

        assert meta.start_time is None
        assert meta.end_time is None
        assert meta.n_timesteps == 0


# ---------------------------------------------------------------------------
# 2D. Hdf5TimeSeriesAdapter
# ---------------------------------------------------------------------------


class TestHdf5TimeSeriesAdapter:
    """Tests for Hdf5TimeSeriesAdapter."""

    def test_read_raises_import_error_when_h5py_missing(self) -> None:
        adapter = Hdf5TimeSeriesAdapter()
        with patch.dict("sys.modules", {"h5py": None}):
            with pytest.raises(ImportError, match="h5py"):
                adapter.read("test.h5")

    def test_read_with_both_datasets_present(self) -> None:
        mock_h5py = MagicMock()
        fake_time_data = np.array([0, 86400], dtype=np.int64)
        fake_value_data = np.array([[1.0, 2.0], [3.0, 4.0]])

        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.__contains__ = lambda s, key: key in (
            "/timeseries/time",
            "/timeseries/data",
        )

        mock_time_ds = MagicMock()
        mock_time_ds.__getitem__ = lambda s, key: fake_time_data
        mock_data_ds = MagicMock()
        mock_data_ds.__getitem__ = lambda s, key: fake_value_data

        def getitem(s, key):
            if key == "/timeseries/time":
                return mock_time_ds
            elif key == "/timeseries/data":
                return mock_data_ds
            raise KeyError(key)

        mock_file.__getitem__ = getitem
        mock_h5py.File.return_value = mock_file

        adapter = Hdf5TimeSeriesAdapter()
        with patch.dict("sys.modules", {"h5py": mock_h5py}):
            times, values, metadata = adapter.read("test.h5")

        assert metadata.file_type == TimeSeriesFileType.HDF5
        assert metadata.n_columns == 2
        assert metadata.n_timesteps == 2
        assert values.shape == (2, 2)

    def test_read_missing_time_dataset_returns_empty_times(self) -> None:
        mock_h5py = MagicMock()
        fake_value_data = np.array([[10.0], [20.0]])

        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        # Only data dataset present, no time
        mock_file.__contains__ = lambda s, key: key == "/timeseries/data"

        mock_data_ds = MagicMock()
        mock_data_ds.__getitem__ = lambda s, key: fake_value_data

        def getitem(s, key):
            if key == "/timeseries/data":
                return mock_data_ds
            raise KeyError(key)

        mock_file.__getitem__ = getitem
        mock_h5py.File.return_value = mock_file

        adapter = Hdf5TimeSeriesAdapter()
        with patch.dict("sys.modules", {"h5py": mock_h5py}):
            times, values, metadata = adapter.read("test.h5")

        assert len(times) == 0
        assert times.dtype == np.dtype("datetime64[s]")

    def test_read_missing_data_dataset_returns_empty_values(self) -> None:
        mock_h5py = MagicMock()
        fake_time_data = np.array([0, 86400], dtype=np.int64)

        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        # Only time dataset present, no data
        mock_file.__contains__ = lambda s, key: key == "/timeseries/time"

        mock_time_ds = MagicMock()
        mock_time_ds.__getitem__ = lambda s, key: fake_time_data

        def getitem(s, key):
            if key == "/timeseries/time":
                return mock_time_ds
            raise KeyError(key)

        mock_file.__getitem__ = getitem
        mock_h5py.File.return_value = mock_file

        adapter = Hdf5TimeSeriesAdapter()
        with patch.dict("sys.modules", {"h5py": mock_h5py}):
            times, values, metadata = adapter.read("test.h5")

        assert len(values) == 0
        # With empty values, n_columns should be 1 (1D fallback)
        assert metadata.n_columns == 1

    def test_read_metadata_raises_import_error_when_h5py_missing(self) -> None:
        adapter = Hdf5TimeSeriesAdapter()
        with patch.dict("sys.modules", {"h5py": None}):
            with pytest.raises(ImportError, match="h5py"):
                adapter.read_metadata("test.h5")

    def test_read_metadata_returns_shape_info(self) -> None:
        mock_h5py = MagicMock()

        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.__contains__ = lambda s, key: key == "/timeseries/data"

        mock_ds = MagicMock()
        mock_ds.shape = (100, 5)

        def getitem(s, key):
            if key == "/timeseries/data":
                return mock_ds
            raise KeyError(key)

        mock_file.__getitem__ = getitem
        mock_h5py.File.return_value = mock_file

        adapter = Hdf5TimeSeriesAdapter()
        with patch.dict("sys.modules", {"h5py": mock_h5py}):
            meta = adapter.read_metadata("test.h5")

        assert meta.file_type == TimeSeriesFileType.HDF5
        assert meta.n_columns == 5
        assert meta.n_timesteps == 100

    def test_read_metadata_missing_dataset_returns_zeros(self) -> None:
        mock_h5py = MagicMock()

        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.__contains__ = lambda s, key: False

        mock_h5py.File.return_value = mock_file

        adapter = Hdf5TimeSeriesAdapter()
        with patch.dict("sys.modules", {"h5py": mock_h5py}):
            meta = adapter.read_metadata("test.h5")

        assert meta.n_columns == 0
        assert meta.n_timesteps == 0


# ---------------------------------------------------------------------------
# 2E. UnifiedTimeSeriesReader
# ---------------------------------------------------------------------------


class TestUnifiedTimeSeriesReader:
    """Tests for UnifiedTimeSeriesReader."""

    def test_constructor_registers_ascii_adapter(self) -> None:
        """ASCII adapter should always be registered."""
        with patch.dict("sys.modules", {"pyiwfm.io.dss": None, "h5py": None}):
            reader = UnifiedTimeSeriesReader()
        assert TimeSeriesFileType.ASCII in reader._adapters
        assert isinstance(reader._adapters[TimeSeriesFileType.ASCII], AsciiTimeSeriesAdapter)

    def test_detect_format_dss(self) -> None:
        reader = UnifiedTimeSeriesReader()
        assert reader._detect_format(Path("file.dss")) == TimeSeriesFileType.DSS

    def test_detect_format_hdf5_extensions(self) -> None:
        reader = UnifiedTimeSeriesReader()
        assert reader._detect_format(Path("file.h5")) == TimeSeriesFileType.HDF5
        assert reader._detect_format(Path("file.hdf5")) == TimeSeriesFileType.HDF5
        assert reader._detect_format(Path("file.hdf")) == TimeSeriesFileType.HDF5

    def test_detect_format_binary(self) -> None:
        reader = UnifiedTimeSeriesReader()
        assert reader._detect_format(Path("file.bin")) == TimeSeriesFileType.BINARY

    def test_detect_format_ascii_for_dat_txt_in_and_unknown(self) -> None:
        reader = UnifiedTimeSeriesReader()
        assert reader._detect_format(Path("file.dat")) == TimeSeriesFileType.ASCII
        assert reader._detect_format(Path("file.txt")) == TimeSeriesFileType.ASCII
        assert reader._detect_format(Path("file.in")) == TimeSeriesFileType.ASCII
        assert reader._detect_format(Path("file.xyz")) == TimeSeriesFileType.ASCII

    def test_get_adapter_raises_for_unavailable_format(self) -> None:
        reader = UnifiedTimeSeriesReader()
        # BINARY never gets an adapter registered
        with pytest.raises(ValueError, match="No adapter available"):
            reader._get_adapter(TimeSeriesFileType.BINARY)

    def test_read_delegates_to_correct_adapter(self) -> None:
        reader = UnifiedTimeSeriesReader()
        fake_times = np.array(["2020-01-01"], dtype="datetime64[s]")
        fake_values = np.array([[5.0]])
        fake_meta = TimeSeriesMetadata(
            file_type=TimeSeriesFileType.ASCII, n_columns=1, n_timesteps=1
        )

        mock_adapter = MagicMock(spec=BaseTimeSeriesReader)
        mock_adapter.read.return_value = (fake_times, fake_values, fake_meta)
        reader._adapters[TimeSeriesFileType.ASCII] = mock_adapter

        config = UnifiedTimeSeriesConfig(
            filepath=Path("pump.dat"),
            file_type=TimeSeriesFileType.ASCII,
        )
        times, values, metadata = reader.read(config)

        mock_adapter.read.assert_called_once()
        np.testing.assert_array_equal(times, fake_times)

    def test_read_applies_conversion_factor(self) -> None:
        reader = UnifiedTimeSeriesReader()
        fake_times = np.array(["2020-01-01", "2020-02-01"], dtype="datetime64[s]")
        fake_values = np.array([[10.0], [20.0]])
        fake_meta = TimeSeriesMetadata(
            file_type=TimeSeriesFileType.ASCII, n_columns=1, n_timesteps=2
        )

        mock_adapter = MagicMock(spec=BaseTimeSeriesReader)
        mock_adapter.read.return_value = (fake_times, fake_values.copy(), fake_meta)
        reader._adapters[TimeSeriesFileType.ASCII] = mock_adapter

        config = UnifiedTimeSeriesConfig(
            filepath=Path("pump.dat"),
            file_type=TimeSeriesFileType.ASCII,
            conversion_factor=2.5,
        )
        _, values, metadata = reader.read(config)

        np.testing.assert_allclose(values, np.array([[25.0], [50.0]]))
        assert metadata.conversion_factor == 2.5

    def test_read_applies_time_filter_start_only(self) -> None:
        reader = UnifiedTimeSeriesReader()
        times = np.array(
            ["2020-01-01", "2020-06-01", "2020-12-01"], dtype="datetime64[s]"
        )
        values = np.array([[1.0], [2.0], [3.0]])
        meta = TimeSeriesMetadata(n_timesteps=3)

        mock_adapter = MagicMock(spec=BaseTimeSeriesReader)
        mock_adapter.read.return_value = (times, values, meta)
        reader._adapters[TimeSeriesFileType.ASCII] = mock_adapter

        config = UnifiedTimeSeriesConfig(
            filepath=Path("test.dat"),
            file_type=TimeSeriesFileType.ASCII,
            start_filter=datetime(2020, 4, 1),
        )
        result_times, result_values, result_meta = reader.read(config)

        assert len(result_times) == 2
        np.testing.assert_array_equal(result_values, np.array([[2.0], [3.0]]))
        assert result_meta.n_timesteps == 2

    def test_read_applies_time_filter_end_only(self) -> None:
        reader = UnifiedTimeSeriesReader()
        times = np.array(
            ["2020-01-01", "2020-06-01", "2020-12-01"], dtype="datetime64[s]"
        )
        values = np.array([[1.0], [2.0], [3.0]])
        meta = TimeSeriesMetadata(n_timesteps=3)

        mock_adapter = MagicMock(spec=BaseTimeSeriesReader)
        mock_adapter.read.return_value = (times, values, meta)
        reader._adapters[TimeSeriesFileType.ASCII] = mock_adapter

        config = UnifiedTimeSeriesConfig(
            filepath=Path("test.dat"),
            file_type=TimeSeriesFileType.ASCII,
            end_filter=datetime(2020, 8, 1),
        )
        result_times, result_values, result_meta = reader.read(config)

        assert len(result_times) == 2
        np.testing.assert_array_equal(result_values, np.array([[1.0], [2.0]]))

    def test_read_applies_time_filter_both(self) -> None:
        reader = UnifiedTimeSeriesReader()
        times = np.array(
            ["2020-01-01", "2020-06-01", "2020-12-01"], dtype="datetime64[s]"
        )
        values = np.array([10.0, 20.0, 30.0])
        meta = TimeSeriesMetadata(n_timesteps=3)

        mock_adapter = MagicMock(spec=BaseTimeSeriesReader)
        mock_adapter.read.return_value = (times, values, meta)
        reader._adapters[TimeSeriesFileType.ASCII] = mock_adapter

        config = UnifiedTimeSeriesConfig(
            filepath=Path("test.dat"),
            file_type=TimeSeriesFileType.ASCII,
            start_filter=datetime(2020, 3, 1),
            end_filter=datetime(2020, 9, 1),
        )
        result_times, result_values, _ = reader.read(config)

        assert len(result_times) == 1
        np.testing.assert_array_equal(result_values, np.array([20.0]))

    def test_read_file_convenience_method(self) -> None:
        reader = UnifiedTimeSeriesReader()
        fake_times = np.array(["2020-01-01"], dtype="datetime64[s]")
        fake_values = np.array([[7.0]])
        fake_meta = TimeSeriesMetadata()

        mock_adapter = MagicMock(spec=BaseTimeSeriesReader)
        mock_adapter.read.return_value = (fake_times, fake_values, fake_meta)
        reader._adapters[TimeSeriesFileType.ASCII] = mock_adapter

        times, values, meta = reader.read_file("test.dat")

        mock_adapter.read.assert_called_once_with(Path("test.dat"))

    def test_read_to_collection_returns_timeseries_collection(self) -> None:
        from pyiwfm.core.timeseries import TimeSeriesCollection

        reader = UnifiedTimeSeriesReader()
        fake_times = np.array(["2020-01-01", "2020-02-01"], dtype="datetime64[s]")
        fake_values = np.array([[1.0, 2.0], [3.0, 4.0]])
        fake_meta = TimeSeriesMetadata(
            n_columns=2, column_ids=[1, 2], n_timesteps=2
        )

        mock_adapter = MagicMock(spec=BaseTimeSeriesReader)
        mock_adapter.read.return_value = (fake_times, fake_values, fake_meta)
        reader._adapters[TimeSeriesFileType.ASCII] = mock_adapter

        collection = reader.read_to_collection(
            "test.dat", variable="pumping"
        )

        assert isinstance(collection, TimeSeriesCollection)
        assert collection.variable == "pumping"
        assert len(collection) == 2
        # Default column IDs come from metadata
        assert "1" in collection.locations
        assert "2" in collection.locations

    def test_read_to_collection_handles_1d_values(self) -> None:
        from pyiwfm.core.timeseries import TimeSeriesCollection

        reader = UnifiedTimeSeriesReader()
        fake_times = np.array(["2020-01-01", "2020-02-01"], dtype="datetime64[s]")
        fake_values = np.array([10.0, 20.0])  # 1D
        fake_meta = TimeSeriesMetadata(
            n_columns=1, column_ids=[1], n_timesteps=2
        )

        mock_adapter = MagicMock(spec=BaseTimeSeriesReader)
        mock_adapter.read.return_value = (fake_times, fake_values, fake_meta)
        reader._adapters[TimeSeriesFileType.ASCII] = mock_adapter

        collection = reader.read_to_collection("test.dat", column_ids=["well_1"])

        assert isinstance(collection, TimeSeriesCollection)
        assert len(collection) == 1
        assert "well_1" in collection.locations
        np.testing.assert_array_equal(collection["well_1"].values, np.array([10.0, 20.0]))

    def test_read_metadata_delegates_correctly(self) -> None:
        reader = UnifiedTimeSeriesReader()
        expected_meta = TimeSeriesMetadata(
            file_type=TimeSeriesFileType.ASCII, n_columns=4, n_timesteps=50
        )

        mock_adapter = MagicMock(spec=BaseTimeSeriesReader)
        mock_adapter.read_metadata.return_value = expected_meta
        reader._adapters[TimeSeriesFileType.ASCII] = mock_adapter

        result = reader.read_metadata("test.dat")

        mock_adapter.read_metadata.assert_called_once_with(Path("test.dat"))
        assert result == expected_meta

    def test_read_sets_recycling_interval_in_metadata(self) -> None:
        reader = UnifiedTimeSeriesReader()
        fake_times = np.array(["2020-01-01"], dtype="datetime64[s]")
        fake_values = np.array([[5.0]])
        fake_meta = TimeSeriesMetadata(n_timesteps=1)

        mock_adapter = MagicMock(spec=BaseTimeSeriesReader)
        mock_adapter.read.return_value = (fake_times, fake_values, fake_meta)
        reader._adapters[TimeSeriesFileType.ASCII] = mock_adapter

        config = UnifiedTimeSeriesConfig(
            filepath=Path("test.dat"),
            file_type=TimeSeriesFileType.ASCII,
            recycling_interval=12,
        )
        _, _, metadata = reader.read(config)

        assert metadata.recycling_interval == 12

    def test_read_passes_dss_pathname_kwarg(self) -> None:
        reader = UnifiedTimeSeriesReader()
        fake_times = np.array(["2020-01-01"], dtype="datetime64[s]")
        fake_values = np.array([[1.0]])
        fake_meta = TimeSeriesMetadata()

        mock_adapter = MagicMock(spec=BaseTimeSeriesReader)
        mock_adapter.read.return_value = (fake_times, fake_values, fake_meta)
        reader._adapters[TimeSeriesFileType.DSS] = mock_adapter

        config = UnifiedTimeSeriesConfig(
            filepath=Path("test.dss"),
            file_type=TimeSeriesFileType.DSS,
            dss_pathname="/A/B/C//1DAY/F/",
        )
        reader.read(config)

        call_kwargs = mock_adapter.read.call_args[1]
        assert call_kwargs["pathname"] == "/A/B/C//1DAY/F/"

    def test_read_passes_hdf5_dataset_kwarg(self) -> None:
        reader = UnifiedTimeSeriesReader()
        fake_times = np.array(["2020-01-01"], dtype="datetime64[s]")
        fake_values = np.array([[1.0]])
        fake_meta = TimeSeriesMetadata()

        mock_adapter = MagicMock(spec=BaseTimeSeriesReader)
        mock_adapter.read.return_value = (fake_times, fake_values, fake_meta)
        reader._adapters[TimeSeriesFileType.HDF5] = mock_adapter

        config = UnifiedTimeSeriesConfig(
            filepath=Path("test.h5"),
            file_type=TimeSeriesFileType.HDF5,
            hdf5_dataset="/custom/data",
        )
        reader.read(config)

        call_kwargs = mock_adapter.read.call_args[1]
        assert call_kwargs["dataset"] == "/custom/data"


# ---------------------------------------------------------------------------
# 2F. RecyclingTimeSeriesReader
# ---------------------------------------------------------------------------


class TestRecyclingTimeSeriesReader:
    """Tests for RecyclingTimeSeriesReader."""

    def test_constructor_with_default_reader(self) -> None:
        recycler = RecyclingTimeSeriesReader()
        assert isinstance(recycler._reader, UnifiedTimeSeriesReader)

    def test_constructor_with_custom_reader(self) -> None:
        custom = MagicMock(spec=UnifiedTimeSeriesReader)
        recycler = RecyclingTimeSeriesReader(base_reader=custom)
        assert recycler._reader is custom

    def test_read_with_recycling_empty_source_times_returns_zeros(self) -> None:
        mock_reader = MagicMock(spec=UnifiedTimeSeriesReader)
        empty_times = np.array([], dtype="datetime64[s]")
        empty_values = np.array([]).reshape(0, 2)
        mock_reader.read_file.return_value = (
            empty_times,
            empty_values,
            TimeSeriesMetadata(),
        )

        recycler = RecyclingTimeSeriesReader(base_reader=mock_reader)
        target_times = np.array(
            ["2020-01-01", "2020-02-01", "2020-03-01"], dtype="datetime64[s]"
        )
        result = recycler.read_with_recycling("source.dat", target_times)

        assert result.shape == (3, 2)
        np.testing.assert_array_equal(result, np.zeros((3, 2)))

    def test_read_with_recycling_maps_months(self) -> None:
        mock_reader = MagicMock(spec=UnifiedTimeSeriesReader)
        # Source: Jan, Feb, Mar 1990 with 2 columns
        source_times = np.array(
            ["1990-01-15", "1990-02-15", "1990-03-15"], dtype="datetime64[s]"
        )
        source_values = np.array([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]])
        mock_reader.read_file.return_value = (
            source_times,
            source_values,
            TimeSeriesMetadata(),
        )

        recycler = RecyclingTimeSeriesReader(base_reader=mock_reader)

        # Target: Jan, Feb 2020 -- should recycle from Jan, Feb 1990
        target_times = np.array(
            ["2020-01-20", "2020-02-20"], dtype="datetime64[s]"
        )
        result = recycler.read_with_recycling("source.dat", target_times, recycling_interval=12)

        assert result.shape == (2, 2)
        # Jan target -> source Jan
        np.testing.assert_array_equal(result[0], [10.0, 100.0])
        # Feb target -> source Feb
        np.testing.assert_array_equal(result[1], [20.0, 200.0])

    def test_read_with_recycling_handles_1d_values(self) -> None:
        mock_reader = MagicMock(spec=UnifiedTimeSeriesReader)
        source_times = np.array(
            ["1990-06-15"], dtype="datetime64[s]"
        )
        # 1D values
        source_values = np.array([42.0])
        mock_reader.read_file.return_value = (
            source_times,
            source_values,
            TimeSeriesMetadata(),
        )

        recycler = RecyclingTimeSeriesReader(base_reader=mock_reader)
        target_times = np.array(["2020-06-01"], dtype="datetime64[s]")
        result = recycler.read_with_recycling("source.dat", target_times, recycling_interval=12)

        # Should return 2D: (1, 1)
        assert result.shape == (1, 1)
        assert result[0, 0] == 42.0


# ---------------------------------------------------------------------------
# 2G. Convenience functions
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_detect_timeseries_format_ascii(self) -> None:
        assert detect_timeseries_format("pumping.dat") == TimeSeriesFileType.ASCII
        assert detect_timeseries_format("data.txt") == TimeSeriesFileType.ASCII
        assert detect_timeseries_format("input.in") == TimeSeriesFileType.ASCII

    def test_detect_timeseries_format_dss(self) -> None:
        assert detect_timeseries_format("output.dss") == TimeSeriesFileType.DSS

    def test_detect_timeseries_format_hdf5(self) -> None:
        assert detect_timeseries_format("heads.h5") == TimeSeriesFileType.HDF5
        assert detect_timeseries_format("heads.hdf5") == TimeSeriesFileType.HDF5
        assert detect_timeseries_format("heads.hdf") == TimeSeriesFileType.HDF5

    def test_detect_timeseries_format_binary(self) -> None:
        assert detect_timeseries_format("model.bin") == TimeSeriesFileType.BINARY

    def test_read_timeseries_unified_delegates(self) -> None:
        with patch.object(
            UnifiedTimeSeriesReader, "read_file"
        ) as mock_read_file:
            fake_result = (
                np.array(["2020-01-01"], dtype="datetime64[s]"),
                np.array([[1.0]]),
                TimeSeriesMetadata(),
            )
            mock_read_file.return_value = fake_result

            result = read_timeseries_unified("test.dat")

        mock_read_file.assert_called_once_with(
            "test.dat", None
        )

    def test_get_timeseries_metadata_delegates(self) -> None:
        with patch.object(
            UnifiedTimeSeriesReader, "read_metadata"
        ) as mock_read_meta:
            expected = TimeSeriesMetadata(n_columns=7)
            mock_read_meta.return_value = expected

            result = get_timeseries_metadata("test.dat")

        mock_read_meta.assert_called_once_with(
            "test.dat", None
        )
        assert result.n_columns == 7

    def test_read_timeseries_unified_with_explicit_file_type(self) -> None:
        with patch.object(
            UnifiedTimeSeriesReader, "read_file"
        ) as mock_read_file:
            fake_result = (
                np.array([], dtype="datetime64[s]"),
                np.array([]),
                TimeSeriesMetadata(),
            )
            mock_read_file.return_value = fake_result

            read_timeseries_unified(
                "test.dss", file_type=TimeSeriesFileType.DSS
            )

        mock_read_file.assert_called_once_with(
            "test.dss", TimeSeriesFileType.DSS
        )

    def test_get_timeseries_metadata_with_explicit_file_type(self) -> None:
        with patch.object(
            UnifiedTimeSeriesReader, "read_metadata"
        ) as mock_read_meta:
            mock_read_meta.return_value = TimeSeriesMetadata()

            get_timeseries_metadata(
                "test.h5", file_type=TimeSeriesFileType.HDF5
            )

        mock_read_meta.assert_called_once_with(
            "test.h5", TimeSeriesFileType.HDF5
        )
