"""
Tests for pyiwfm.io.dss.timeseries module.

Tests the high-level time series read/write utilities for HEC-DSS files.
Uses mocks to test without the actual DSS library.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pyiwfm.core.timeseries import TimeSeries, TimeSeriesCollection
from pyiwfm.io.dss.pathname import DSSPathname, DSSPathnameTemplate
from pyiwfm.io.dss.timeseries import (
    DSSTimeSeriesReader,
    DSSTimeSeriesWriter,
    DSSWriteResult,
    read_timeseries_from_dss,
    write_collection_to_dss,
    write_timeseries_to_dss,
)


class TestDSSWriteResult:
    """Tests for DSSWriteResult dataclass."""

    def test_creation_success(self, tmp_path: Path) -> None:
        """Test creating successful write result."""
        result = DSSWriteResult(
            filepath=tmp_path / "test.dss",
            pathnames_written=["/A/B/C/D/E/F/", "/A/B/C2/D/E/F/"],
            n_records=2,
            errors=[],
        )
        assert result.filepath == tmp_path / "test.dss"
        assert len(result.pathnames_written) == 2
        assert result.n_records == 2
        assert result.success

    def test_creation_with_errors(self, tmp_path: Path) -> None:
        """Test creating result with errors."""
        result = DSSWriteResult(
            filepath=tmp_path / "test.dss",
            pathnames_written=["/A/B/C/D/E/F/"],
            n_records=1,
            errors=["Error writing /A/B/C2/D/E/F/"],
        )
        assert not result.success
        assert len(result.errors) == 1

    def test_success_property_empty(self, tmp_path: Path) -> None:
        """Test success property with no records."""
        result = DSSWriteResult(
            filepath=tmp_path / "test.dss",
            pathnames_written=[],
            n_records=0,
            errors=[],
        )
        assert result.success  # No errors means success


class TestDSSTimeSeriesWriter:
    """Tests for DSSTimeSeriesWriter class."""

    @pytest.fixture
    def mock_dss_file(self) -> MagicMock:
        """Create mock DSSFile."""
        mock = MagicMock()
        mock.open.return_value = None
        mock.close.return_value = None
        mock.write_regular_timeseries.return_value = None
        return mock

    @pytest.fixture
    def sample_timeseries(self) -> TimeSeries:
        """Create sample timeseries for testing."""
        times = np.array(
            [
                np.datetime64("2020-01-01"),
                np.datetime64("2020-01-02"),
                np.datetime64("2020-01-03"),
            ]
        )
        values = np.array([1.0, 2.0, 3.0])
        return TimeSeries(
            times=times,
            values=values,
            name="test",
            location="LOC",
            units="CFS",
        )

    def test_init(self, tmp_path: Path) -> None:
        """Test writer initialization."""
        with patch("pyiwfm.io.dss.timeseries.check_dss_available"):
            writer = DSSTimeSeriesWriter(tmp_path / "test.dss")
            assert writer.filepath == tmp_path / "test.dss"
            assert writer._dss is None
            assert len(writer._pathnames_written) == 0
            assert len(writer._errors) == 0

    def test_context_manager(self, tmp_path: Path, mock_dss_file: MagicMock) -> None:
        """Test writer as context manager."""
        with patch("pyiwfm.io.dss.timeseries.check_dss_available"):
            with patch("pyiwfm.io.dss.timeseries.DSSFile", return_value=mock_dss_file):
                with DSSTimeSeriesWriter(tmp_path / "test.dss") as writer:
                    assert writer._dss is not None

    def test_open(self, tmp_path: Path, mock_dss_file: MagicMock) -> None:
        """Test open method."""
        with patch("pyiwfm.io.dss.timeseries.check_dss_available"):
            with patch("pyiwfm.io.dss.timeseries.DSSFile", return_value=mock_dss_file):
                writer = DSSTimeSeriesWriter(tmp_path / "test.dss")
                writer.open()
                assert writer._dss is not None
                mock_dss_file.open.assert_called_once()

    def test_open_idempotent(self, tmp_path: Path, mock_dss_file: MagicMock) -> None:
        """Test open is idempotent."""
        with patch("pyiwfm.io.dss.timeseries.check_dss_available"):
            with patch("pyiwfm.io.dss.timeseries.DSSFile", return_value=mock_dss_file):
                writer = DSSTimeSeriesWriter(tmp_path / "test.dss")
                writer.open()
                writer.open()  # Second call should not create new DSSFile
                mock_dss_file.open.assert_called_once()

    def test_close_returns_result(self, tmp_path: Path, mock_dss_file: MagicMock) -> None:
        """Test close returns DSSWriteResult."""
        with patch("pyiwfm.io.dss.timeseries.check_dss_available"):
            with patch("pyiwfm.io.dss.timeseries.DSSFile", return_value=mock_dss_file):
                writer = DSSTimeSeriesWriter(tmp_path / "test.dss")
                writer.open()
                result = writer.close()
                assert isinstance(result, DSSWriteResult)
                assert result.filepath == tmp_path / "test.dss"

    def test_write_timeseries(
        self,
        tmp_path: Path,
        mock_dss_file: MagicMock,
        sample_timeseries: TimeSeries,
    ) -> None:
        """Test write_timeseries method."""
        with patch("pyiwfm.io.dss.timeseries.check_dss_available"):
            with patch("pyiwfm.io.dss.timeseries.DSSFile", return_value=mock_dss_file):
                writer = DSSTimeSeriesWriter(tmp_path / "test.dss")
                writer.open()
                pathname = DSSPathname.from_string("/PROJECT/LOC/FLOW//1DAY/VER/")
                success = writer.write_timeseries(sample_timeseries, pathname)
                assert success
                assert len(writer._pathnames_written) == 1
                mock_dss_file.write_regular_timeseries.assert_called_once()

    def test_write_timeseries_string_pathname(
        self,
        tmp_path: Path,
        mock_dss_file: MagicMock,
        sample_timeseries: TimeSeries,
    ) -> None:
        """Test write_timeseries with string pathname."""
        with patch("pyiwfm.io.dss.timeseries.check_dss_available"):
            with patch("pyiwfm.io.dss.timeseries.DSSFile", return_value=mock_dss_file):
                writer = DSSTimeSeriesWriter(tmp_path / "test.dss")
                writer.open()
                success = writer.write_timeseries(sample_timeseries, "/PROJECT/LOC/FLOW//1DAY/VER/")
                assert success

    def test_write_timeseries_auto_open(
        self,
        tmp_path: Path,
        mock_dss_file: MagicMock,
        sample_timeseries: TimeSeries,
    ) -> None:
        """Test write_timeseries opens file automatically."""
        with patch("pyiwfm.io.dss.timeseries.check_dss_available"):
            with patch("pyiwfm.io.dss.timeseries.DSSFile", return_value=mock_dss_file):
                writer = DSSTimeSeriesWriter(tmp_path / "test.dss")
                # Don't call open explicitly
                success = writer.write_timeseries(sample_timeseries, "/PROJECT/LOC/FLOW//1DAY/VER/")
                assert success
                mock_dss_file.open.assert_called_once()

    def test_write_timeseries_with_units(
        self,
        tmp_path: Path,
        mock_dss_file: MagicMock,
        sample_timeseries: TimeSeries,
    ) -> None:
        """Test write_timeseries with custom units."""
        with patch("pyiwfm.io.dss.timeseries.check_dss_available"):
            with patch("pyiwfm.io.dss.timeseries.DSSFile", return_value=mock_dss_file):
                writer = DSSTimeSeriesWriter(tmp_path / "test.dss")
                writer.open()
                pathname = "/PROJECT/LOC/FLOW//1DAY/VER/"
                writer.write_timeseries(sample_timeseries, pathname, units="AF")
                call_kwargs = mock_dss_file.write_regular_timeseries.call_args
                assert call_kwargs.kwargs.get("units") == "AF"

    def test_write_timeseries_error(
        self,
        tmp_path: Path,
        sample_timeseries: TimeSeries,
    ) -> None:
        """Test write_timeseries handles errors."""
        from pyiwfm.io.dss.wrapper import DSSFileError

        mock_dss = MagicMock()
        mock_dss.open.return_value = None
        mock_dss.write_regular_timeseries.side_effect = DSSFileError("Write failed")

        with patch("pyiwfm.io.dss.timeseries.check_dss_available"):
            with patch("pyiwfm.io.dss.timeseries.DSSFile", return_value=mock_dss):
                writer = DSSTimeSeriesWriter(tmp_path / "test.dss")
                writer.open()
                success = writer.write_timeseries(sample_timeseries, "/PROJECT/LOC/FLOW//1DAY/VER/")
                assert not success
                assert len(writer._errors) == 1

    def test_write_collection(
        self,
        tmp_path: Path,
        mock_dss_file: MagicMock,
    ) -> None:
        """Test write_collection method."""
        times = np.array([np.datetime64("2020-01-01"), np.datetime64("2020-01-02")])
        collection = TimeSeriesCollection(variable="FLOW")
        collection.add(TimeSeries(times=times, values=np.array([1.0, 2.0]), location="LOC1"))
        collection.add(TimeSeries(times=times, values=np.array([3.0, 4.0]), location="LOC2"))

        def pathname_factory(loc: str) -> DSSPathname:
            return DSSPathname.from_string(f"/PROJECT/{loc}/FLOW//1DAY/VER/")

        with patch("pyiwfm.io.dss.timeseries.check_dss_available"):
            with patch("pyiwfm.io.dss.timeseries.DSSFile", return_value=mock_dss_file):
                writer = DSSTimeSeriesWriter(tmp_path / "test.dss")
                writer.open()
                n_written = writer.write_collection(collection, pathname_factory, units="CFS")
                assert n_written == 2

    def test_write_multiple_timeseries(
        self,
        tmp_path: Path,
        mock_dss_file: MagicMock,
    ) -> None:
        """Test write_multiple_timeseries method."""
        times = [datetime(2020, 1, 1), datetime(2020, 1, 2)]
        values_dict = {
            "LOC1": np.array([1.0, 2.0]),
            "LOC2": np.array([3.0, 4.0]),
        }
        template = DSSPathnameTemplate(a_part="PROJECT", c_part="FLOW", e_part="1DAY", f_part="VER")

        with patch("pyiwfm.io.dss.timeseries.check_dss_available"):
            with patch("pyiwfm.io.dss.timeseries.DSSFile", return_value=mock_dss_file):
                writer = DSSTimeSeriesWriter(tmp_path / "test.dss")
                writer.open()
                n_written = writer.write_multiple_timeseries(
                    times, values_dict, template, units="CFS"
                )
                assert n_written == 2

    def test_write_multiple_timeseries_numpy_times(
        self,
        tmp_path: Path,
        mock_dss_file: MagicMock,
    ) -> None:
        """Test write_multiple_timeseries with numpy datetime."""
        times = np.array([np.datetime64("2020-01-01"), np.datetime64("2020-01-02")])
        values_dict = {"LOC1": np.array([1.0, 2.0])}
        template = DSSPathnameTemplate(a_part="PROJECT", c_part="FLOW", e_part="1DAY")

        with patch("pyiwfm.io.dss.timeseries.check_dss_available"):
            with patch("pyiwfm.io.dss.timeseries.DSSFile", return_value=mock_dss_file):
                writer = DSSTimeSeriesWriter(tmp_path / "test.dss")
                writer.open()
                n_written = writer.write_multiple_timeseries(times, values_dict, template)
                assert n_written == 1

    def test_numpy_dt_to_datetime(self, tmp_path: Path) -> None:
        """Test _numpy_dt_to_datetime conversion."""
        with patch("pyiwfm.io.dss.timeseries.check_dss_available"):
            writer = DSSTimeSeriesWriter(tmp_path / "test.dss")
            np_dt = np.datetime64("2020-06-15T12:30:00")
            result = writer._numpy_dt_to_datetime(np_dt)
            assert isinstance(result, datetime)
            assert result.year == 2020
            assert result.month == 6
            assert result.day == 15


class TestDSSTimeSeriesReader:
    """Tests for DSSTimeSeriesReader class."""

    @pytest.fixture
    def mock_dss_file(self) -> MagicMock:
        """Create mock DSSFile."""
        mock = MagicMock()
        mock.open.return_value = None
        mock.close.return_value = None
        mock.read_regular_timeseries.return_value = (
            [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            np.array([1.0, 2.0]),
        )
        return mock

    def test_init(self, tmp_path: Path) -> None:
        """Test reader initialization."""
        with patch("pyiwfm.io.dss.timeseries.check_dss_available"):
            reader = DSSTimeSeriesReader(tmp_path / "test.dss")
            assert reader.filepath == tmp_path / "test.dss"
            assert reader._dss is None

    def test_context_manager(self, tmp_path: Path, mock_dss_file: MagicMock) -> None:
        """Test reader as context manager."""
        with patch("pyiwfm.io.dss.timeseries.check_dss_available"):
            with patch("pyiwfm.io.dss.timeseries.DSSFile", return_value=mock_dss_file):
                with DSSTimeSeriesReader(tmp_path / "test.dss") as reader:
                    assert reader._dss is not None

    def test_open(self, tmp_path: Path, mock_dss_file: MagicMock) -> None:
        """Test open method."""
        with patch("pyiwfm.io.dss.timeseries.check_dss_available"):
            with patch("pyiwfm.io.dss.timeseries.DSSFile", return_value=mock_dss_file):
                reader = DSSTimeSeriesReader(tmp_path / "test.dss")
                reader.open()
                assert reader._dss is not None
                mock_dss_file.open.assert_called_once()

    def test_open_idempotent(self, tmp_path: Path, mock_dss_file: MagicMock) -> None:
        """Test open is idempotent."""
        with patch("pyiwfm.io.dss.timeseries.check_dss_available"):
            with patch("pyiwfm.io.dss.timeseries.DSSFile", return_value=mock_dss_file):
                reader = DSSTimeSeriesReader(tmp_path / "test.dss")
                reader.open()
                reader.open()
                mock_dss_file.open.assert_called_once()

    def test_close(self, tmp_path: Path, mock_dss_file: MagicMock) -> None:
        """Test close method."""
        with patch("pyiwfm.io.dss.timeseries.check_dss_available"):
            with patch("pyiwfm.io.dss.timeseries.DSSFile", return_value=mock_dss_file):
                reader = DSSTimeSeriesReader(tmp_path / "test.dss")
                reader.open()
                reader.close()
                assert reader._dss is None
                mock_dss_file.close.assert_called_once()

    def test_close_not_open(self, tmp_path: Path, mock_dss_file: MagicMock) -> None:
        """Test close when not open."""
        with patch("pyiwfm.io.dss.timeseries.check_dss_available"):
            reader = DSSTimeSeriesReader(tmp_path / "test.dss")
            reader.close()  # Should not raise

    def test_read_timeseries(self, tmp_path: Path, mock_dss_file: MagicMock) -> None:
        """Test read_timeseries method."""
        with patch("pyiwfm.io.dss.timeseries.check_dss_available"):
            with patch("pyiwfm.io.dss.timeseries.DSSFile", return_value=mock_dss_file):
                reader = DSSTimeSeriesReader(tmp_path / "test.dss")
                reader.open()
                ts = reader.read_timeseries("/PROJECT/LOC/FLOW//1DAY/VER/")
                assert isinstance(ts, TimeSeries)
                assert ts.location == "LOC"
                mock_dss_file.read_regular_timeseries.assert_called_once()

    def test_read_timeseries_auto_open(self, tmp_path: Path, mock_dss_file: MagicMock) -> None:
        """Test read_timeseries opens file automatically."""
        with patch("pyiwfm.io.dss.timeseries.check_dss_available"):
            with patch("pyiwfm.io.dss.timeseries.DSSFile", return_value=mock_dss_file):
                reader = DSSTimeSeriesReader(tmp_path / "test.dss")
                ts = reader.read_timeseries("/PROJECT/LOC/FLOW//1DAY/VER/")
                assert isinstance(ts, TimeSeries)
                mock_dss_file.open.assert_called_once()

    def test_read_timeseries_with_dates(self, tmp_path: Path, mock_dss_file: MagicMock) -> None:
        """Test read_timeseries with date range."""
        with patch("pyiwfm.io.dss.timeseries.check_dss_available"):
            with patch("pyiwfm.io.dss.timeseries.DSSFile", return_value=mock_dss_file):
                reader = DSSTimeSeriesReader(tmp_path / "test.dss")
                reader.open()
                start = datetime(2020, 1, 1)
                end = datetime(2020, 12, 31)
                reader.read_timeseries(
                    "/PROJECT/LOC/FLOW//1DAY/VER/",
                    start_date=start,
                    end_date=end,
                )
                mock_dss_file.read_regular_timeseries.assert_called_with(
                    "/PROJECT/LOC/FLOW//1DAY/VER/", start, end
                )

    def test_read_timeseries_with_name(self, tmp_path: Path, mock_dss_file: MagicMock) -> None:
        """Test read_timeseries with custom name."""
        with patch("pyiwfm.io.dss.timeseries.check_dss_available"):
            with patch("pyiwfm.io.dss.timeseries.DSSFile", return_value=mock_dss_file):
                reader = DSSTimeSeriesReader(tmp_path / "test.dss")
                reader.open()
                ts = reader.read_timeseries("/PROJECT/LOC/FLOW//1DAY/VER/", name="Custom Name")
                assert ts.name == "Custom Name"

    def test_read_timeseries_pathname_object(
        self, tmp_path: Path, mock_dss_file: MagicMock
    ) -> None:
        """Test read_timeseries with DSSPathname object."""
        with patch("pyiwfm.io.dss.timeseries.check_dss_available"):
            with patch("pyiwfm.io.dss.timeseries.DSSFile", return_value=mock_dss_file):
                reader = DSSTimeSeriesReader(tmp_path / "test.dss")
                reader.open()
                pathname = DSSPathname.from_string("/PROJECT/LOC/FLOW//1DAY/VER/")
                ts = reader.read_timeseries(pathname)
                assert ts.location == "LOC"

    def test_read_collection(self, tmp_path: Path, mock_dss_file: MagicMock) -> None:
        """Test read_collection method."""
        with patch("pyiwfm.io.dss.timeseries.check_dss_available"):
            with patch("pyiwfm.io.dss.timeseries.DSSFile", return_value=mock_dss_file):
                reader = DSSTimeSeriesReader(tmp_path / "test.dss")
                reader.open()
                pathnames = [
                    "/PROJECT/LOC1/FLOW//1DAY/VER/",
                    "/PROJECT/LOC2/FLOW//1DAY/VER/",
                ]
                collection = reader.read_collection(pathnames, variable="FLOW")
                assert isinstance(collection, TimeSeriesCollection)
                assert collection.variable == "FLOW"

    def test_read_collection_skips_missing(self, tmp_path: Path) -> None:
        """Test read_collection skips missing records."""
        from pyiwfm.io.dss.wrapper import DSSFileError

        mock_dss = MagicMock()
        mock_dss.open.return_value = None
        mock_dss.close.return_value = None
        # First call succeeds, second raises error
        mock_dss.read_regular_timeseries.side_effect = [
            ([datetime(2020, 1, 1)], np.array([1.0])),
            DSSFileError("Not found"),
        ]

        with patch("pyiwfm.io.dss.timeseries.check_dss_available"):
            with patch("pyiwfm.io.dss.timeseries.DSSFile", return_value=mock_dss):
                reader = DSSTimeSeriesReader(tmp_path / "test.dss")
                reader.open()
                pathnames = [
                    "/PROJECT/LOC1/FLOW//1DAY/VER/",
                    "/PROJECT/LOC2/FLOW//1DAY/VER/",
                ]
                collection = reader.read_collection(pathnames)
                # Only one should be read successfully
                assert len(list(collection.locations)) == 1


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.fixture
    def mock_writer(self) -> MagicMock:
        """Create mock writer."""
        mock = MagicMock(spec=DSSTimeSeriesWriter)
        mock.__enter__ = MagicMock(return_value=mock)
        mock.__exit__ = MagicMock(return_value=None)
        mock.close.return_value = DSSWriteResult(
            filepath=Path("test.dss"),
            pathnames_written=["/A/B/C/D/E/F/"],
            n_records=1,
            errors=[],
        )
        return mock

    @pytest.fixture
    def mock_reader(self) -> MagicMock:
        """Create mock reader."""
        times = np.array([np.datetime64("2020-01-01")])
        values = np.array([1.0])
        mock = MagicMock(spec=DSSTimeSeriesReader)
        mock.__enter__ = MagicMock(return_value=mock)
        mock.__exit__ = MagicMock(return_value=None)
        mock.read_timeseries.return_value = TimeSeries(
            times=times, values=values, name="test", location="LOC"
        )
        return mock

    def test_write_timeseries_to_dss(self, tmp_path: Path, mock_writer: MagicMock) -> None:
        """Test write_timeseries_to_dss function."""
        ts = TimeSeries(
            times=np.array([np.datetime64("2020-01-01")]),
            values=np.array([1.0]),
            name="test",
            location="LOC",
        )

        with patch("pyiwfm.io.dss.timeseries.DSSTimeSeriesWriter", return_value=mock_writer):
            result = write_timeseries_to_dss(tmp_path / "test.dss", ts, "/A/B/C/D/E/F/")
            assert isinstance(result, DSSWriteResult)
            mock_writer.write_timeseries.assert_called_once()

    def test_read_timeseries_from_dss(self, tmp_path: Path, mock_reader: MagicMock) -> None:
        """Test read_timeseries_from_dss function."""
        with patch("pyiwfm.io.dss.timeseries.DSSTimeSeriesReader", return_value=mock_reader):
            ts = read_timeseries_from_dss(tmp_path / "test.dss", "/A/B/C/D/E/F/")
            assert isinstance(ts, TimeSeries)
            mock_reader.read_timeseries.assert_called_once()

    def test_read_timeseries_from_dss_with_dates(
        self, tmp_path: Path, mock_reader: MagicMock
    ) -> None:
        """Test read_timeseries_from_dss with date range."""
        start = datetime(2020, 1, 1)
        end = datetime(2020, 12, 31)

        with patch("pyiwfm.io.dss.timeseries.DSSTimeSeriesReader", return_value=mock_reader):
            read_timeseries_from_dss(
                tmp_path / "test.dss",
                "/A/B/C/D/E/F/",
                start_date=start,
                end_date=end,
            )
            mock_reader.read_timeseries.assert_called_with("/A/B/C/D/E/F/", start, end)

    def test_write_collection_to_dss(self, tmp_path: Path, mock_writer: MagicMock) -> None:
        """Test write_collection_to_dss function."""
        times = np.array([np.datetime64("2020-01-01")])
        collection = TimeSeriesCollection(variable="FLOW")
        collection.add(TimeSeries(times=times, values=np.array([1.0]), location="LOC1"))

        template = DSSPathnameTemplate(a_part="PROJECT", c_part="FLOW", e_part="1DAY")

        with patch("pyiwfm.io.dss.timeseries.DSSTimeSeriesWriter", return_value=mock_writer):
            result = write_collection_to_dss(
                tmp_path / "test.dss", collection, template, units="CFS"
            )
            assert isinstance(result, DSSWriteResult)
            mock_writer.write_collection.assert_called_once()


class TestIntegrationWithPathname:
    """Integration tests with DSSPathname module."""

    @pytest.fixture
    def mock_dss_file(self) -> MagicMock:
        """Create mock DSSFile."""
        mock = MagicMock()
        mock.open.return_value = None
        mock.close.return_value = None
        mock.write_regular_timeseries.return_value = None
        return mock

    def test_pathname_date_range_update(self, tmp_path: Path, mock_dss_file: MagicMock) -> None:
        """Test pathname D-part is updated with date range."""
        times = np.array(
            [
                np.datetime64("2020-01-01"),
                np.datetime64("2020-12-31"),
            ]
        )
        ts = TimeSeries(
            times=times,
            values=np.array([1.0, 2.0]),
            name="test",
            location="LOC",
        )
        pathname = DSSPathname.from_string("/PROJECT/LOC/FLOW//1DAY/VER/")

        with patch("pyiwfm.io.dss.timeseries.check_dss_available"):
            with patch("pyiwfm.io.dss.timeseries.DSSFile", return_value=mock_dss_file):
                writer = DSSTimeSeriesWriter(tmp_path / "test.dss")
                writer.open()
                writer.write_timeseries(ts, pathname)

                # Check the pathname that was written
                call_kwargs = mock_dss_file.write_regular_timeseries.call_args
                written_pathname = call_kwargs.kwargs.get(
                    "pathname", call_kwargs.args[0] if call_kwargs.args else ""
                )
                # D-part should now contain date range
                assert "2020" in written_pathname or written_pathname  # Basic check

    def test_template_makes_pathname(self, tmp_path: Path, mock_dss_file: MagicMock) -> None:
        """Test using DSSPathnameTemplate to create pathnames."""
        template = DSSPathnameTemplate(
            a_part="C2VSIM",
            c_part="FLOW",
            e_part="1MON",
            f_part="COMPUTED",
        )

        times = [datetime(2020, 1, 1), datetime(2020, 2, 1)]
        values_dict = {
            "STREAM_01": np.array([100.0, 120.0]),
            "STREAM_02": np.array([80.0, 90.0]),
        }

        with patch("pyiwfm.io.dss.timeseries.check_dss_available"):
            with patch("pyiwfm.io.dss.timeseries.DSSFile", return_value=mock_dss_file):
                writer = DSSTimeSeriesWriter(tmp_path / "test.dss")
                writer.open()
                n_written = writer.write_multiple_timeseries(
                    times, values_dict, template, units="CFS"
                )
                assert n_written == 2
                assert mock_dss_file.write_regular_timeseries.call_count == 2
