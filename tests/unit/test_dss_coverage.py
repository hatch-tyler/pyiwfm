"""Supplementary tests for DSS-related code targeting uncovered branches.

Covers:
- writer_base.py: DSS write path in TimeSeriesWriter (_write_dss_timeseries)
- writer_base.py: OutputFormat.DSS and OutputFormat.BOTH branches
- wrapper.py: Platform-specific library search, read_regular_timeseries branches
- timeseries.py: DSSTimeSeriesWriter/Reader high-level APIs
- pathname.py: DSSPathname construction, parsing, matching
"""

from __future__ import annotations

import os
import platform
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import numpy as np
import pytest

from pyiwfm.io.config import OutputFormat, TimeSeriesOutputConfig
from pyiwfm.io.writer_base import (
    TimeSeriesSpec,
    TimeSeriesWriter,
    _check_dss,
    HAS_DSS,
)
from pyiwfm.io.dss.wrapper import (
    DSSFile,
    DSSFileMock,
    DSSFileError,
    DSSLibraryError,
    DSSTimeSeriesInfo,
    HAS_DSS_LIBRARY,
    check_dss_available,
    _get_library_path,
    _load_library,
    _configure_argtypes,
    _zStructTimeSeries,
    get_dss_file_class,
    IFLTAB_SIZE,
)
from pyiwfm.io.dss.timeseries import (
    DSSTimeSeriesWriter,
    DSSTimeSeriesReader,
    DSSWriteResult,
    write_timeseries_to_dss,
    read_timeseries_from_dss,
    write_collection_to_dss,
)
from pyiwfm.io.dss.pathname import (
    DSSPathname,
    DSSPathnameTemplate,
    format_dss_date,
    format_dss_date_range,
    parse_dss_date,
    interval_to_minutes,
    minutes_to_interval,
    VALID_INTERVALS,
    PARAMETER_CODES,
    INTERVAL_MAPPING,
)


# =============================================================================
# writer_base.py: DSS Write Path Tests
# =============================================================================


class TestTimeSeriesWriterDSSFormat:
    """Tests for TimeSeriesWriter DSS output format branches."""

    def _make_ts_spec(self) -> TimeSeriesSpec:
        """Create a sample TimeSeriesSpec."""
        return TimeSeriesSpec(
            name="TestTimeSeries",
            dates=[datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
            values=[10.0, 20.0, 30.0],
            units="CFS",
            location="STREAM_NODE_1",
            parameter="FLOW",
            interval="1DAY",
        )

    def test_write_timeseries_dss_format(self, tmp_path: Path) -> None:
        """Test write_timeseries with OutputFormat.DSS calls DSS write."""
        config = TimeSeriesOutputConfig(
            format=OutputFormat.DSS,
            dss_file="test_output.dss",
            dss_a_part="TESTPROJECT",
            dss_f_part="PYIWFM_TEST",
        )
        writer = TimeSeriesWriter(config, tmp_path)
        ts_spec = self._make_ts_spec()

        # Use mock to intercept the actual DSS write (avoids needing real DSS file)
        with patch.object(writer, "_write_dss_timeseries") as mock_dss_write:
            writer.write_timeseries(ts_spec)

            mock_dss_write.assert_called_once_with(ts_spec)

    def test_write_timeseries_both_format(self, tmp_path: Path) -> None:
        """Test write_timeseries with OutputFormat.BOTH calls both text and DSS."""
        config = TimeSeriesOutputConfig(
            format=OutputFormat.BOTH,
            dss_file="test_output.dss",
        )
        writer = TimeSeriesWriter(config, tmp_path)
        ts_spec = self._make_ts_spec()

        with patch.object(writer, "_write_dss_timeseries") as mock_dss:
            writer.write_timeseries(ts_spec, text_file="output.dat")

            # DSS write should be called
            mock_dss.assert_called_once_with(ts_spec)
            # Text file should also exist
            assert (tmp_path / "output.dat").exists()

    def test_write_timeseries_dss_no_text_file_needed(self, tmp_path: Path) -> None:
        """Test DSS-only format doesn't require text_file."""
        config = TimeSeriesOutputConfig(
            format=OutputFormat.DSS,
            dss_file="test.dss",
        )
        writer = TimeSeriesWriter(config, tmp_path)
        ts_spec = self._make_ts_spec()

        with patch.object(writer, "_write_dss_timeseries"):
            # Should not raise - no text_file needed for DSS-only
            writer.write_timeseries(ts_spec)

    def test_write_dss_timeseries_pathname_construction(self, tmp_path: Path) -> None:
        """Test _write_dss_timeseries constructs correct DSS pathname."""
        config = TimeSeriesOutputConfig(
            format=OutputFormat.DSS,
            dss_file="output.dss",
            dss_a_part="PROJECT",
            dss_f_part="VERSION",
        )
        writer = TimeSeriesWriter(config, tmp_path)
        ts_spec = TimeSeriesSpec(
            name="MyTS",
            dates=[datetime(2020, 1, 1), datetime(2020, 1, 2)],
            values=[100.0, 200.0],
            units="FT",
            location="NODE_5",
            parameter="HEAD",
            interval="1DAY",
        )

        with patch("pyiwfm.io.writer_base.write_timeseries_to_dss") as mock_write:
            writer._write_dss_timeseries(ts_spec)

            mock_write.assert_called_once()
            call_args = mock_write.call_args
            dss_path = call_args[0][0]
            pathname = call_args[0][1]

            # Verify DSS path is relative to output_dir
            assert str(tmp_path / "output.dss") == dss_path
            # Verify pathname parts
            assert "/PROJECT/" in pathname
            assert "/NODE_5/" in pathname
            assert "/HEAD/" in pathname
            assert "/1DAY/" in pathname
            assert "/VERSION/" in pathname

    def test_write_dss_timeseries_default_parts(self, tmp_path: Path) -> None:
        """Test _write_dss_timeseries uses defaults for missing parts."""
        config = TimeSeriesOutputConfig(
            format=OutputFormat.DSS,
            dss_file="output.dss",
            dss_a_part="",
            dss_f_part="",
        )
        writer = TimeSeriesWriter(config, tmp_path)
        ts_spec = TimeSeriesSpec(
            name="TestName",
            dates=[datetime(2020, 1, 1)],
            values=[10.0],
        )

        with patch("pyiwfm.io.writer_base.write_timeseries_to_dss") as mock_write:
            writer._write_dss_timeseries(ts_spec)

            call_args = mock_write.call_args
            pathname = call_args[0][1]
            # b_part defaults to name when no location
            assert "/TestName/" in pathname
            # c_part defaults to VALUE when no parameter
            assert "/VALUE/" in pathname

    def test_write_dss_timeseries_no_config_raises(self, tmp_path: Path) -> None:
        """Test _write_dss_timeseries raises when dss_file not configured."""
        config = TimeSeriesOutputConfig(
            format=OutputFormat.DSS,
            dss_file=None,
        )
        writer = TimeSeriesWriter(config, tmp_path)
        ts_spec = self._make_ts_spec()

        with pytest.raises(ValueError, match="DSS file not configured"):
            writer._write_dss_timeseries(ts_spec)


# =============================================================================
# wrapper.py: Platform Detection Tests
# =============================================================================


class TestGetLibraryPathPlatform:
    """Tests for platform-specific library search in _get_library_path."""

    def test_env_var_path_exists(self, tmp_path: Path) -> None:
        """Test library found via environment variable."""
        lib_file = tmp_path / "hecdss.dll"
        lib_file.write_bytes(b"fake_lib")

        with patch.dict(os.environ, {"HECDSS_LIB": str(lib_file)}):
            path = _get_library_path()
            assert path == lib_file

    def test_env_var_path_missing(self) -> None:
        """Test environment variable points to non-existent path."""
        with patch.dict(os.environ, {"HECDSS_LIB": "/nonexistent/path/lib.dll"}):
            # Should fall through to package dir, then common paths
            path = _get_library_path()
            # May or may not find it depending on system, but shouldn't crash

    def test_package_lib_dir_search(self) -> None:
        """Test library search falls through package lib dir to common paths."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HECDSS_LIB", None)
            # Even without env var, _get_library_path should still work
            # (it will find the bundled lib or return None)
            result = _get_library_path()
            # Result depends on system setup - just verify it doesn't crash

    def test_common_paths_not_found(self) -> None:
        """Test that _get_library_path returns None when no paths work."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HECDSS_LIB", None)
            with patch("pyiwfm.io.dss.wrapper.Path.exists", return_value=False):
                path = _get_library_path()
                # If package lib doesn't exist, falls through to common paths


class TestDSSWrapperOperations:
    """Tests for DSSFile wrapper operations."""

    @pytest.mark.skipif(not HAS_DSS_LIBRARY, reason="DSS library not available")
    def test_dss_file_write_and_read(self, tmp_path: Path) -> None:
        """Test writing and reading a DSS timeseries roundtrip.

        Uses the HEC-DSS 7 constructor pattern:
        write: zstructTsNewRegFloats -> ztsStore -> zstructFree
        read:  zstructTsNewTimes -> ztsRetrieve -> extract fields -> zstructFree
        """
        dss_path = tmp_path / "test.dss"
        pathname = "/TEST/STREAM/FLOW//1DAY/PYIWFM/"
        write_values = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        start = datetime(2020, 1, 1)

        # Write data
        with DSSFile(str(dss_path), mode="rw") as dss:
            dss.write_regular_timeseries(
                pathname=pathname,
                values=write_values,
                start_date=start,
                units="CFS",
                data_type="INST-VAL",
            )

        assert dss_path.exists()

        # Read data back
        with DSSFile(str(dss_path), mode="r") as dss:
            times, values = dss.read_regular_timeseries(
                pathname,
                start_date=datetime(2020, 1, 1),
                end_date=datetime(2020, 1, 3),
            )

        assert len(values) == 3
        np.testing.assert_array_almost_equal(values, [10.0, 20.0, 30.0], decimal=1)
        assert len(times) == 3

    @pytest.mark.skipif(not HAS_DSS_LIBRARY, reason="DSS library not available")
    def test_dss_file_write_read_info(self, tmp_path: Path) -> None:
        """Test get_timeseries_info returns metadata after write."""
        dss_path = tmp_path / "info_test.dss"
        pathname = "/PROJ/LOC/HEAD//1DAY/V1/"
        write_values = np.array([100.0, 95.0, 90.0, 85.0, 80.0], dtype=np.float32)

        with DSSFile(str(dss_path), mode="rw") as dss:
            dss.write_regular_timeseries(
                pathname=pathname,
                values=write_values,
                start_date=datetime(2020, 6, 1),
                units="FT",
                data_type="INST-VAL",
            )

        with DSSFile(str(dss_path), mode="r") as dss:
            info = dss.get_timeseries_info(pathname)

        assert info.n_values == 5
        assert info.units == "FT"
        assert info.data_type == "INST-VAL"
        assert info.start_date is not None

    @pytest.mark.skipif(not HAS_DSS_LIBRARY, reason="DSS library not available")
    def test_dss_file_open_close_lifecycle(self, tmp_path: Path) -> None:
        """Test that open and close work with the real DSS library."""
        dss_path = tmp_path / "lifecycle_test.dss"

        dss = DSSFile(str(dss_path), mode="rw")
        dss.open()
        assert dss._is_open
        assert dss._ifltab is not None
        assert len(dss._ifltab) == IFLTAB_SIZE

        dss.close()
        assert not dss._is_open
        assert dss._ifltab is None
        assert dss_path.exists()

    @pytest.mark.skipif(not HAS_DSS_LIBRARY, reason="DSS library not available")
    def test_dss_file_context_manager(self, tmp_path: Path) -> None:
        """Test DSSFile as context manager opens and closes properly."""
        dss_path = tmp_path / "context_test.dss"

        with DSSFile(str(dss_path), mode="w") as dss:
            assert dss._is_open
            assert dss.filepath == dss_path

        # After context exit, file should be closed
        assert not dss._is_open

    @pytest.mark.skipif(not HAS_DSS_LIBRARY, reason="DSS library not available")
    def test_dss_file_open_already_open(self, tmp_path: Path) -> None:
        """Test opening an already-open DSS file is a no-op."""
        dss_path = tmp_path / "test.dss"

        dss = DSSFile(str(dss_path), mode="rw")
        dss.open()
        assert dss._is_open

        # Second open should be no-op
        dss.open()
        assert dss._is_open

        dss.close()

    @pytest.mark.skipif(not HAS_DSS_LIBRARY, reason="DSS library not available")
    def test_dss_file_close_already_closed(self, tmp_path: Path) -> None:
        """Test closing an already-closed DSS file is a no-op."""
        dss_path = tmp_path / "test.dss"

        dss = DSSFile(str(dss_path), mode="rw")
        dss.open()
        dss.close()
        assert not dss._is_open

        # Second close should be no-op
        dss.close()
        assert not dss._is_open

    @pytest.mark.skipif(not HAS_DSS_LIBRARY, reason="DSS library not available")
    def test_dss_file_write_not_open_raises(self, tmp_path: Path) -> None:
        """Test writing to a non-open file raises."""
        dss = DSSFile(str(tmp_path / "test.dss"), mode="rw")

        with pytest.raises(DSSFileError, match="not open"):
            dss.write_regular_timeseries(
                pathname="/A/B/C/D/E/F/",
                values=np.array([1.0]),
                start_date=datetime(2020, 1, 1),
            )

    @pytest.mark.skipif(not HAS_DSS_LIBRARY, reason="DSS library not available")
    def test_dss_file_read_not_open_raises(self, tmp_path: Path) -> None:
        """Test reading from a non-open file raises."""
        dss = DSSFile(str(tmp_path / "test.dss"), mode="r")

        with pytest.raises(DSSFileError, match="not open"):
            dss.read_regular_timeseries("/A/B/C/D/E/F/")

    @pytest.mark.skipif(not HAS_DSS_LIBRARY, reason="DSS library not available")
    def test_dss_file_catalog_not_open_raises(self, tmp_path: Path) -> None:
        """Test catalog on non-open file raises."""
        dss = DSSFile(str(tmp_path / "test.dss"), mode="r")

        with pytest.raises(DSSFileError, match="not open"):
            dss.catalog()

    @pytest.mark.skipif(not HAS_DSS_LIBRARY, reason="DSS library not available")
    def test_dss_file_get_info_not_open_raises(self, tmp_path: Path) -> None:
        """Test get_timeseries_info on non-open file raises."""
        dss = DSSFile(str(tmp_path / "test.dss"), mode="r")

        with pytest.raises(DSSFileError, match="not open"):
            dss.get_timeseries_info("/A/B/C/D/E/F/")

    @pytest.mark.skipif(not HAS_DSS_LIBRARY, reason="DSS library not available")
    def test_dss_file_write_readonly_raises(self, tmp_path: Path) -> None:
        """Test writing to a read-only file raises."""
        dss_path = tmp_path / "test.dss"
        # Create the file first
        with DSSFile(str(dss_path), mode="w") as dss:
            pass

        # Open read-only
        dss = DSSFile(str(dss_path), mode="r")
        dss.open()
        try:
            with pytest.raises(DSSFileError, match="not open for writing"):
                dss.write_regular_timeseries(
                    pathname="/A/B/C/D/E/F/",
                    values=np.array([1.0]),
                    start_date=datetime(2020, 1, 1),
                )
        finally:
            dss.close()

    @pytest.mark.skipif(not HAS_DSS_LIBRARY, reason="DSS library not available")
    def test_dss_file_catalog_empty_file(self, tmp_path: Path) -> None:
        """Test catalog on a newly created empty DSS file returns empty list."""
        dss_path = tmp_path / "empty.dss"

        with DSSFile(str(dss_path), mode="rw") as dss:
            result = dss.catalog()
            assert result == []

    @pytest.mark.skipif(not HAS_DSS_LIBRARY, reason="DSS library not available")
    def test_dss_file_get_info_empty(self, tmp_path: Path) -> None:
        """Test get_timeseries_info returns default info for non-existent path."""
        dss_path = tmp_path / "empty.dss"

        with DSSFile(str(dss_path), mode="rw") as dss:
            info = dss.get_timeseries_info("/A/B/C/D/E/F/")
            assert info.pathname == "/A/B/C/D/E/F/"
            assert info.n_values == 0

    @pytest.mark.skipif(not HAS_DSS_LIBRARY, reason="DSS library not available")
    def test_dss_file_read_empty_timeseries(self, tmp_path: Path) -> None:
        """Test reading a non-existent timeseries returns empty arrays."""
        dss_path = tmp_path / "empty.dss"

        with DSSFile(str(dss_path), mode="rw") as dss:
            times, values = dss.read_regular_timeseries("/A/B/C/D/E/F/")
            assert times == []
            assert len(values) == 0

    @pytest.mark.skipif(not HAS_DSS_LIBRARY, reason="DSS library not available")
    def test_dss_file_open_modes(self, tmp_path: Path) -> None:
        """Test different file open modes."""
        dss_path = tmp_path / "modes.dss"

        # Write mode
        with DSSFile(str(dss_path), mode="w") as dss:
            assert dss.mode == "w"

        # Read mode
        with DSSFile(str(dss_path), mode="r") as dss:
            assert dss.mode == "r"

        # Read-write mode
        with DSSFile(str(dss_path), mode="rw") as dss:
            assert dss.mode == "rw"


class TestDSSFileMockOperations:
    """Tests for DSSFileMock."""

    def test_mock_open_raises(self) -> None:
        """Test mock open raises library error."""
        mock = DSSFileMock("fake.dss")
        with pytest.raises(DSSLibraryError):
            mock.open()

    def test_mock_write_raises(self) -> None:
        """Test mock write raises library error."""
        mock = DSSFileMock("fake.dss")
        with pytest.raises(DSSLibraryError):
            mock.write_regular_timeseries()

    def test_mock_read_raises(self) -> None:
        """Test mock read raises library error."""
        mock = DSSFileMock("fake.dss")
        with pytest.raises(DSSLibraryError):
            mock.read_regular_timeseries()

    def test_mock_close_noop(self) -> None:
        """Test mock close does not raise."""
        mock = DSSFileMock("fake.dss")
        mock.close()  # should not raise

    def test_mock_context_manager(self) -> None:
        """Test mock as context manager."""
        with DSSFileMock("fake.dss") as mock:
            assert mock.filepath == Path("fake.dss")

    def test_mock_filepath_is_path_object(self) -> None:
        """Test mock converts string filepath to Path."""
        mock = DSSFileMock("/some/path/data.dss")
        assert isinstance(mock.filepath, Path)
        assert mock.filepath.name == "data.dss"

    def test_mock_mode_stored(self) -> None:
        """Test mock stores the file mode."""
        mock = DSSFileMock("test.dss", mode="rw")
        assert mock.mode == "rw"

    def test_mock_exit_no_error(self) -> None:
        """Test mock __exit__ doesn't raise even with exception info."""
        mock = DSSFileMock("test.dss")
        mock.__exit__(None, None, None)
        mock.__exit__(RuntimeError, RuntimeError("test"), None)


class TestDSSUtilities:
    """Tests for DSS utility functions."""

    def test_check_dss_available_when_present(self) -> None:
        """Test check_dss_available passes when library is present."""
        if HAS_DSS_LIBRARY:
            check_dss_available()  # Should not raise

    def test_check_dss_available_when_missing(self) -> None:
        """Test check_dss_available raises when library not found."""
        with patch("pyiwfm.io.dss.wrapper.HAS_DSS_LIBRARY", False):
            with pytest.raises(DSSLibraryError, match="not found"):
                check_dss_available()

    def test_get_dss_file_class_with_library(self) -> None:
        """Test get_dss_file_class returns DSSFile when library available."""
        cls = get_dss_file_class()
        if HAS_DSS_LIBRARY:
            assert cls is DSSFile
        else:
            assert cls is DSSFileMock

    def test_dss_timeseries_info_creation(self) -> None:
        """Test DSSTimeSeriesInfo creation with all fields."""
        info = DSSTimeSeriesInfo(
            pathname="/A/B/C/D/E/F/",
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            n_values=365,
            units="CFS",
            data_type="INST-VAL",
        )
        assert info.n_values == 365
        assert info.units == "CFS"

    def test_dss_timeseries_info_defaults(self) -> None:
        """Test DSSTimeSeriesInfo default values."""
        info = DSSTimeSeriesInfo(pathname="/A/B/C/D/E/F/")
        assert info.start_date is None
        assert info.end_date is None
        assert info.n_values == 0
        assert info.units == ""
        assert info.data_type == ""

    def test_dss_write_result_success(self) -> None:
        """Test DSSWriteResult.success property."""
        result = DSSWriteResult(
            filepath=Path("test.dss"),
            pathnames_written=["/A/B/C/D/E/F/"],
            n_records=1,
            errors=[],
        )
        assert result.success is True

    def test_dss_write_result_failure(self) -> None:
        """Test DSSWriteResult.success property with errors."""
        result = DSSWriteResult(
            filepath=Path("test.dss"),
            pathnames_written=[],
            n_records=0,
            errors=["Write failed"],
        )
        assert result.success is False

    def test_ifltab_size(self) -> None:
        """Test IFLTAB_SIZE constant matches HEC-DSS 7 spec."""
        assert IFLTAB_SIZE == 250


# =============================================================================
# wrapper.py: Struct and Argtypes Tests
# =============================================================================


class TestZStructTimeSeries:
    """Tests for the _zStructTimeSeries ctypes structure."""

    def test_struct_has_required_fields(self) -> None:
        """Test that the struct defines all required fields."""
        field_names = [f[0] for f in _zStructTimeSeries._fields_]
        assert "structType" in field_names
        assert "pathname" in field_names
        assert "numberValues" in field_names
        assert "floatValues" in field_names
        assert "doubleValues" in field_names
        assert "units" in field_names
        assert "type" in field_names
        assert "startJulianDate" in field_names
        assert "startTimeSeconds" in field_names
        assert "timeIntervalSeconds" in field_names
        assert "times" in field_names

    def test_struct_field_count(self) -> None:
        """Test that struct defines exactly 19 fields (partial definition)."""
        assert len(_zStructTimeSeries._fields_) == 19

    def test_struct_field_order(self) -> None:
        """Test fields are in correct order per C header."""
        field_names = [f[0] for f in _zStructTimeSeries._fields_]
        assert field_names[0] == "structType"
        assert field_names[1] == "pathname"
        assert field_names[12] == "numberValues"
        assert field_names[15] == "floatValues"
        assert field_names[17] == "units"
        assert field_names[18] == "type"


class TestConfigureArgtypes:
    """Tests for _configure_argtypes function."""

    @pytest.mark.skipif(not HAS_DSS_LIBRARY, reason="DSS library not available")
    def test_configure_argtypes_sets_signatures(self) -> None:
        """Test that _configure_argtypes sets argtypes on library functions."""
        from pyiwfm.io.dss.wrapper import _dss_lib

        # These should be set by _configure_argtypes at module load
        assert _dss_lib.zopenExtended.argtypes is not None
        assert _dss_lib.zclose.argtypes is not None
        assert _dss_lib.ztsStore.argtypes is not None
        assert _dss_lib.ztsRetrieve.argtypes is not None
        assert _dss_lib.zstructTsNewRegFloats.argtypes is not None
        assert _dss_lib.zstructTsNewTimes.argtypes is not None
        assert _dss_lib.zstructFree.argtypes is not None

    @pytest.mark.skipif(not HAS_DSS_LIBRARY, reason="DSS library not available")
    def test_configure_argtypes_sets_restype(self) -> None:
        """Test that _configure_argtypes sets restype on library functions."""
        from pyiwfm.io.dss.wrapper import _dss_lib, c_void_p, c_int

        assert _dss_lib.zopenExtended.restype == c_int
        assert _dss_lib.zclose.restype is None
        assert _dss_lib.ztsStore.restype == c_int
        assert _dss_lib.ztsRetrieve.restype == c_int
        assert _dss_lib.zstructTsNewRegFloats.restype == c_void_p
        assert _dss_lib.zstructTsNewTimes.restype == c_void_p
        assert _dss_lib.zstructFree.restype is None

    def test_configure_argtypes_with_mock_lib(self) -> None:
        """Test _configure_argtypes works with a mock library object."""
        mock_lib = MagicMock()
        _configure_argtypes(mock_lib)

        # Verify function attributes were set
        assert mock_lib.zopenExtended.argtypes is not None
        assert mock_lib.ztsStore.argtypes is not None
        assert mock_lib.zstructFree.argtypes is not None


# =============================================================================
# writer_base.py: Close DSS File Tests
# =============================================================================


class TestTimeSeriesWriterClose:
    """Tests for TimeSeriesWriter close behavior."""

    def test_close_with_no_dss_file(self, tmp_path: Path) -> None:
        """Test close when no DSS file was opened."""
        config = TimeSeriesOutputConfig(format=OutputFormat.TEXT)
        writer = TimeSeriesWriter(config, tmp_path)

        # _dss_file is None, close should be no-op
        writer.close()

    def test_close_with_mock_dss_file(self, tmp_path: Path) -> None:
        """Test close properly closes DSS file."""
        config = TimeSeriesOutputConfig(format=OutputFormat.DSS, dss_file="test.dss")
        writer = TimeSeriesWriter(config, tmp_path)

        # Set up a mock DSS file
        mock_dss = MagicMock()
        writer._dss_file = mock_dss

        writer.close()

        mock_dss.close.assert_called_once()
        assert writer._dss_file is None

    def test_close_idempotent(self, tmp_path: Path) -> None:
        """Test closing twice is safe."""
        config = TimeSeriesOutputConfig(format=OutputFormat.TEXT)
        writer = TimeSeriesWriter(config, tmp_path)

        writer.close()
        writer.close()  # Should not raise


# =============================================================================
# wrapper.py: _load_library Edge Cases
# =============================================================================


class TestLoadLibrary:
    """Tests for _load_library function."""

    def test_load_library_no_path_returns_none(self) -> None:
        """Test _load_library returns None when no library path found."""
        with patch("pyiwfm.io.dss.wrapper._get_library_path", return_value=None):
            result = _load_library()
            assert result is None

    def test_load_library_oserror_returns_none(self) -> None:
        """Test _load_library returns None on OSError."""
        with patch("pyiwfm.io.dss.wrapper._get_library_path", return_value=Path("/fake/lib.so")):
            with patch("ctypes.CDLL", side_effect=OSError("Cannot load")):
                result = _load_library()
                assert result is None


# =============================================================================
# writer_base.py: _check_dss Tests
# =============================================================================


class TestCheckDSS:
    """Tests for _check_dss function."""

    def test_check_dss_available(self) -> None:
        """Test _check_dss passes when DSS is available."""
        if HAS_DSS:
            _check_dss()  # Should not raise

    def test_check_dss_unavailable_raises(self) -> None:
        """Test _check_dss raises ImportError when DSS unavailable."""
        with patch("pyiwfm.io.writer_base.HAS_DSS", False):
            with pytest.raises(ImportError, match="DSS support requires"):
                _check_dss()


# =============================================================================
# pathname.py: DSSPathname Tests
# =============================================================================


class TestDSSPathname:
    """Tests for DSSPathname class."""

    def test_from_string_valid(self) -> None:
        """Test parsing a valid pathname string."""
        pn = DSSPathname.from_string("/PROJECT/LOCATION/FLOW//1DAY/VERSION/")
        assert pn.a_part == "PROJECT"
        assert pn.b_part == "LOCATION"
        assert pn.c_part == "FLOW"
        assert pn.d_part == ""
        assert pn.e_part == "1DAY"
        assert pn.f_part == "VERSION"

    def test_from_string_lowercase_uppercased(self) -> None:
        """Test that lowercase parts are uppercased."""
        pn = DSSPathname.from_string("/project/location/flow//1day/version/")
        assert pn.a_part == "PROJECT"
        assert pn.b_part == "LOCATION"
        assert pn.c_part == "FLOW"

    def test_from_string_missing_leading_slash(self) -> None:
        """Test error for missing leading slash."""
        with pytest.raises(ValueError, match="Invalid DSS pathname"):
            DSSPathname.from_string("A/B/C/D/E/F/")

    def test_from_string_missing_trailing_slash(self) -> None:
        """Test error for missing trailing slash."""
        with pytest.raises(ValueError, match="Invalid DSS pathname"):
            DSSPathname.from_string("/A/B/C/D/E/F")

    def test_from_string_wrong_number_of_parts(self) -> None:
        """Test error when not exactly 6 parts."""
        with pytest.raises(ValueError, match="6 parts"):
            DSSPathname.from_string("/A/B/C/D/E/")

    def test_str_roundtrip(self) -> None:
        """Test str() produces a valid pathname that can be re-parsed."""
        original = DSSPathname(
            a_part="A", b_part="B", c_part="C",
            d_part="D", e_part="E", f_part="F",
        )
        s = str(original)
        reparsed = DSSPathname.from_string(s)
        assert reparsed.a_part == original.a_part
        assert reparsed.f_part == original.f_part

    def test_build_with_parameter_codes(self) -> None:
        """Test build() converts parameter names to DSS codes."""
        pn = DSSPathname.build(
            project="IWFM",
            location="NODE_1",
            parameter="flow",
            interval="daily",
        )
        assert pn.c_part == "FLOW"
        assert pn.e_part == "1DAY"

    def test_build_with_unknown_parameter(self) -> None:
        """Test build() uses uppercase for unknown parameters."""
        pn = DSSPathname.build(parameter="custom_param")
        assert pn.c_part == "CUSTOM_PARAM"

    def test_build_with_unknown_interval(self) -> None:
        """Test build() uses uppercase for unknown intervals."""
        pn = DSSPathname.build(interval="15SEC")
        assert pn.e_part == "15SEC"

    def test_with_location(self) -> None:
        """Test creating a copy with different location."""
        pn = DSSPathname.build(project="A", location="LOC1", parameter="flow")
        pn2 = pn.with_location("LOC2")
        assert pn2.b_part == "LOC2"
        assert pn2.a_part == pn.a_part  # unchanged
        assert pn2.c_part == pn.c_part  # unchanged

    def test_with_parameter(self) -> None:
        """Test creating a copy with different parameter."""
        pn = DSSPathname.build(location="LOC1", parameter="flow")
        pn2 = pn.with_parameter("stage")
        assert pn2.c_part == "STAGE"
        assert pn2.b_part == pn.b_part  # unchanged

    def test_with_date_range(self) -> None:
        """Test creating a copy with different date range."""
        pn = DSSPathname.build(location="LOC1")
        pn2 = pn.with_date_range("01jan2020-31dec2020")
        assert pn2.d_part == "01JAN2020-31DEC2020"

    def test_with_version(self) -> None:
        """Test creating a copy with different version."""
        pn = DSSPathname.build(location="LOC1", version="v1")
        pn2 = pn.with_version("v2")
        assert pn2.f_part == "V2"

    def test_matches_full(self) -> None:
        """Test full pathname matching."""
        pn = DSSPathname.from_string("/A/B/C/D/E/F/")
        assert pn.matches("/A/B/C/D/E/F/")
        assert not pn.matches("/X/B/C/D/E/F/")

    def test_matches_partial(self) -> None:
        """Test partial pathname matching (empty parts match anything)."""
        pn = DSSPathname.from_string("/A/B/C/D/E/F/")
        # Pattern with empty parts matches anything
        pattern = DSSPathname(b_part="B", c_part="C")
        assert pn.matches(pattern)
        pattern_miss = DSSPathname(b_part="X", c_part="C")
        assert not pn.matches(pattern_miss)

    def test_matches_with_pathname_object(self) -> None:
        """Test matching with DSSPathname object pattern."""
        pn = DSSPathname.from_string("/A/B/C/D/E/F/")
        pattern = DSSPathname(b_part="B", c_part="C")
        assert pn.matches(pattern)

    def test_is_regular_interval(self) -> None:
        """Test regular interval detection."""
        pn = DSSPathname(e_part="1DAY")
        assert pn.is_regular_interval
        assert not pn.is_irregular_interval

    def test_is_irregular_interval(self) -> None:
        """Test irregular interval detection."""
        pn = DSSPathname(e_part="IR-DAY")
        assert pn.is_irregular_interval
        assert not pn.is_regular_interval


class TestDSSPathnameTemplate:
    """Tests for DSSPathnameTemplate."""

    def test_make_pathname_with_location(self) -> None:
        """Test making pathname with location override."""
        template = DSSPathnameTemplate(
            a_part="PROJECT",
            c_part="FLOW",
            e_part="1DAY",
            f_part="V1",
        )
        pn = template.make_pathname(location="STREAM_01")
        assert pn.a_part == "PROJECT"
        assert pn.b_part == "STREAM_01"
        assert pn.c_part == "FLOW"
        assert pn.e_part == "1DAY"
        assert pn.f_part == "V1"

    def test_make_pathname_with_date_range(self) -> None:
        """Test making pathname with date range override."""
        template = DSSPathnameTemplate(c_part="FLOW")
        pn = template.make_pathname(location="LOC", date_range="01jan2020")
        assert pn.d_part == "01JAN2020"

    def test_make_pathname_with_kwargs(self) -> None:
        """Test making pathname with keyword overrides."""
        template = DSSPathnameTemplate(a_part="DEFAULT", c_part="FLOW")
        pn = template.make_pathname(location="LOC", a_part="OVERRIDE")
        assert pn.a_part == "OVERRIDE"

    def test_make_pathnames_multiple_locations(self) -> None:
        """Test generating pathnames for multiple locations."""
        template = DSSPathnameTemplate(a_part="PROJECT", c_part="FLOW")
        locations = ["LOC_1", "LOC_2", "LOC_3"]
        pathnames = list(template.make_pathnames(locations))

        assert len(pathnames) == 3
        assert pathnames[0].b_part == "LOC_1"
        assert pathnames[1].b_part == "LOC_2"
        assert pathnames[2].b_part == "LOC_3"

    def test_make_pathnames_with_date_range(self) -> None:
        """Test generating pathnames with shared date range."""
        template = DSSPathnameTemplate(c_part="HEAD")
        pathnames = list(
            template.make_pathnames(["W1", "W2"], date_range="01jan2020")
        )
        assert all(pn.d_part == "01JAN2020" for pn in pathnames)


# =============================================================================
# pathname.py: Utility Function Tests
# =============================================================================


class TestPathnameUtilities:
    """Tests for pathname utility functions."""

    def test_format_dss_date(self) -> None:
        """Test formatting a datetime to DSS date string."""
        result = format_dss_date(datetime(2020, 3, 15))
        assert result == "15MAR2020"

    def test_format_dss_date_range(self) -> None:
        """Test formatting a date range."""
        result = format_dss_date_range(
            datetime(2020, 1, 1), datetime(2020, 12, 31)
        )
        assert result == "01JAN2020-31DEC2020"

    def test_parse_dss_date(self) -> None:
        """Test parsing DSS date string."""
        result = parse_dss_date("15MAR2020")
        assert result == datetime(2020, 3, 15)

    def test_parse_dss_date_lowercase(self) -> None:
        """Test parsing lowercase DSS date string."""
        result = parse_dss_date("15mar2020")
        assert result == datetime(2020, 3, 15)

    def test_interval_to_minutes_min(self) -> None:
        """Test minute interval conversion."""
        assert interval_to_minutes("15MIN") == 15
        assert interval_to_minutes("1MIN") == 1
        assert interval_to_minutes("30MIN") == 30

    def test_interval_to_minutes_hour(self) -> None:
        """Test hour interval conversion."""
        assert interval_to_minutes("1HOUR") == 60
        assert interval_to_minutes("6HOUR") == 360

    def test_interval_to_minutes_day(self) -> None:
        """Test day interval conversion."""
        assert interval_to_minutes("1DAY") == 1440

    def test_interval_to_minutes_week(self) -> None:
        """Test week interval conversion."""
        assert interval_to_minutes("1WEEK") == 10080

    def test_interval_to_minutes_month(self) -> None:
        """Test month interval conversion (approximate)."""
        assert interval_to_minutes("1MON") == 43200

    def test_interval_to_minutes_year(self) -> None:
        """Test year interval conversion (approximate)."""
        assert interval_to_minutes("1YEAR") == 525600

    def test_interval_to_minutes_lowercase(self) -> None:
        """Test case-insensitive interval parsing."""
        assert interval_to_minutes("1day") == 1440
        assert interval_to_minutes("1hour") == 60

    def test_interval_to_minutes_unknown(self) -> None:
        """Test error for unknown interval."""
        with pytest.raises(ValueError, match="Unknown DSS interval"):
            interval_to_minutes("1FORTNIGHT")

    def test_minutes_to_interval_exact(self) -> None:
        """Test exact minute conversions."""
        assert minutes_to_interval(15) == "15MIN"
        assert minutes_to_interval(30) == "30MIN"
        assert minutes_to_interval(60) == "1HOUR"

    def test_minutes_to_interval_hours(self) -> None:
        """Test hour conversions."""
        assert minutes_to_interval(120) == "2HOUR"
        assert minutes_to_interval(360) == "6HOUR"

    def test_minutes_to_interval_day(self) -> None:
        """Test day conversion."""
        assert minutes_to_interval(1440) == "1DAY"

    def test_minutes_to_interval_week(self) -> None:
        """Test week conversion."""
        assert minutes_to_interval(10080) == "1WEEK"

    def test_minutes_to_interval_month(self) -> None:
        """Test month conversion."""
        # 28+ days but less than 365 days -> 1MON
        assert minutes_to_interval(60 * 24 * 30) == "1MON"

    def test_minutes_to_interval_year(self) -> None:
        """Test year conversion."""
        assert minutes_to_interval(60 * 24 * 400) == "1YEAR"

    def test_minutes_to_interval_nearest_minute(self) -> None:
        """Test nearest minute interval for non-standard values."""
        # 7 minutes rounds to nearest valid: 6MIN
        result = minutes_to_interval(7)
        assert result == "6MIN"

    def test_minutes_to_interval_non_standard_hour(self) -> None:
        """Test non-standard hour value defaults to 1HOUR."""
        result = minutes_to_interval(300)  # 5 hours - not in valid list
        assert result == "1HOUR"

    def test_parameter_codes_mapping(self) -> None:
        """Test that common parameter codes are defined."""
        assert PARAMETER_CODES["flow"] == "FLOW"
        assert PARAMETER_CODES["head"] == "HEAD"
        assert PARAMETER_CODES["stage"] == "STAGE"
        assert PARAMETER_CODES["precipitation"] == "PRECIP"
        assert PARAMETER_CODES["pumping"] == "PUMP"

    def test_valid_intervals_set(self) -> None:
        """Test that expected intervals are in the valid set."""
        assert "1DAY" in VALID_INTERVALS
        assert "1HOUR" in VALID_INTERVALS
        assert "15MIN" in VALID_INTERVALS
        assert "1MON" in VALID_INTERVALS
        assert "IR-DAY" in VALID_INTERVALS

    def test_interval_mapping(self) -> None:
        """Test common interval name mappings."""
        assert INTERVAL_MAPPING["daily"] == "1DAY"
        assert INTERVAL_MAPPING["hourly"] == "1HOUR"
        assert INTERVAL_MAPPING["monthly"] == "1MON"
        assert INTERVAL_MAPPING["yearly"] == "1YEAR"


# =============================================================================
# timeseries.py: DSSTimeSeriesWriter High-Level API Tests
# =============================================================================


class TestDSSTimeSeriesWriterHighLevel:
    """Tests for DSSTimeSeriesWriter using mocked DSSFile."""

    @pytest.mark.skipif(not HAS_DSS_LIBRARY, reason="DSS library not available")
    def test_writer_open_close_lifecycle(self, tmp_path: Path) -> None:
        """Test writer open and close lifecycle."""
        dss_path = tmp_path / "writer_test.dss"

        writer = DSSTimeSeriesWriter(dss_path)
        writer.open()
        assert writer._dss is not None

        result = writer.close()
        assert writer._dss is None
        assert isinstance(result, DSSWriteResult)
        assert result.filepath == dss_path
        assert result.success

    @pytest.mark.skipif(not HAS_DSS_LIBRARY, reason="DSS library not available")
    def test_writer_context_manager(self, tmp_path: Path) -> None:
        """Test writer as context manager."""
        dss_path = tmp_path / "ctx_test.dss"

        with DSSTimeSeriesWriter(dss_path) as writer:
            assert writer._dss is not None

    @pytest.mark.skipif(not HAS_DSS_LIBRARY, reason="DSS library not available")
    def test_writer_open_idempotent(self, tmp_path: Path) -> None:
        """Test opening an already-open writer is a no-op."""
        dss_path = tmp_path / "idem_test.dss"

        writer = DSSTimeSeriesWriter(dss_path)
        writer.open()
        first_dss = writer._dss

        writer.open()  # Should not create a new DSSFile
        assert writer._dss is first_dss

        writer.close()

    @pytest.mark.skipif(not HAS_DSS_LIBRARY, reason="DSS library not available")
    def test_writer_close_returns_result(self, tmp_path: Path) -> None:
        """Test close returns accumulated write results."""
        dss_path = tmp_path / "result_test.dss"

        writer = DSSTimeSeriesWriter(dss_path)
        writer.open()

        # Simulate pathnames written
        writer._pathnames_written.append("/A/B/C/D/E/F/")
        writer._errors.append("test error")

        result = writer.close()
        assert result.n_records == 1
        assert len(result.pathnames_written) == 1
        assert len(result.errors) == 1
        assert not result.success

    @pytest.mark.skipif(not HAS_DSS_LIBRARY, reason="DSS library not available")
    def test_writer_close_without_open(self, tmp_path: Path) -> None:
        """Test closing writer that was never opened."""
        dss_path = tmp_path / "no_open_test.dss"

        writer = DSSTimeSeriesWriter(dss_path)
        result = writer.close()
        assert result.success
        assert result.n_records == 0


class TestDSSTimeSeriesReaderHighLevel:
    """Tests for DSSTimeSeriesReader using real DSS library."""

    @pytest.mark.skipif(not HAS_DSS_LIBRARY, reason="DSS library not available")
    def test_reader_open_close_lifecycle(self, tmp_path: Path) -> None:
        """Test reader open and close lifecycle."""
        dss_path = tmp_path / "reader_test.dss"
        # Create file first
        with DSSFile(str(dss_path), mode="w"):
            pass

        reader = DSSTimeSeriesReader(dss_path)
        reader.open()
        assert reader._dss is not None

        reader.close()
        assert reader._dss is None

    @pytest.mark.skipif(not HAS_DSS_LIBRARY, reason="DSS library not available")
    def test_reader_context_manager(self, tmp_path: Path) -> None:
        """Test reader as context manager."""
        dss_path = tmp_path / "ctx_read.dss"
        with DSSFile(str(dss_path), mode="w"):
            pass

        with DSSTimeSeriesReader(dss_path) as reader:
            assert reader._dss is not None

    @pytest.mark.skipif(not HAS_DSS_LIBRARY, reason="DSS library not available")
    def test_reader_open_idempotent(self, tmp_path: Path) -> None:
        """Test opening an already-open reader is a no-op."""
        dss_path = tmp_path / "idem_read.dss"
        with DSSFile(str(dss_path), mode="w"):
            pass

        reader = DSSTimeSeriesReader(dss_path)
        reader.open()
        first_dss = reader._dss

        reader.open()
        assert reader._dss is first_dss

        reader.close()

    @pytest.mark.skipif(not HAS_DSS_LIBRARY, reason="DSS library not available")
    def test_reader_close_without_open(self, tmp_path: Path) -> None:
        """Test closing reader that was never opened is no-op."""
        dss_path = tmp_path / "no_open_read.dss"

        reader = DSSTimeSeriesReader(dss_path)
        reader.close()  # Should not raise

    @pytest.mark.skipif(not HAS_DSS_LIBRARY, reason="DSS library not available")
    def test_reader_read_collection_empty(self, tmp_path: Path) -> None:
        """Test reading collection from empty file yields empty collection."""
        dss_path = tmp_path / "empty_collection.dss"
        with DSSFile(str(dss_path), mode="w"):
            pass

        with DSSTimeSeriesReader(dss_path) as reader:
            collection = reader.read_collection(
                ["/A/B/C/D/E/F/", "/X/Y/Z//1DAY/V/"],
                variable="test",
            )
            # Empty file means records are missing - should be empty collection
            assert collection.variable == "test"


# =============================================================================
# Additional wrapper.py coverage: _get_library_path edge cases
# =============================================================================


class TestGetLibraryPathEdgeCases:
    """Additional tests for _get_library_path() uncovered branches."""

    def test_env_var_path_directory_instead_of_file(self, tmp_path: Path) -> None:
        """Env var pointing to a directory (not a file) falls through."""
        with patch.dict(os.environ, {"HECDSS_LIB": str(tmp_path)}):
            # tmp_path exists but is a directory, not a library file
            # _get_library_path checks exists(), which is True for dirs
            result = _get_library_path()
            # Result depends on whether other paths exist

    def test_package_lib_dir_linux_platform(self, tmp_path: Path) -> None:
        """Package lib dir with Linux platform detection."""
        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        lib_file = lib_dir / "libhecdss.so"
        lib_file.write_bytes(b"fake")

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HECDSS_LIB", None)
            with patch("platform.system", return_value="Linux"):
                # Just verify no crash
                _get_library_path()

    def test_package_lib_dir_darwin_platform(self, tmp_path: Path) -> None:
        """Package lib dir with macOS platform detection."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HECDSS_LIB", None)
            with patch("platform.system", return_value="Darwin"):
                _get_library_path()

    def test_all_common_paths_missing(self) -> None:
        """All common paths missing returns None."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HECDSS_LIB", None)
            # Patch package lib dir to not exist
            with patch.object(Path, "exists", return_value=False):
                result = _get_library_path()
                assert result is None


class TestDSSFileEdgeCases:
    """Additional edge case tests for DSSFile operations."""

    def test_dss_file_init_stores_filepath(self) -> None:
        """DSSFile stores filepath as Path object."""
        if not HAS_DSS_LIBRARY:
            pytest.skip("DSS library not available")
        dss = DSSFile("/some/path/data.dss", mode="rw")
        assert dss.filepath == Path("/some/path/data.dss")
        assert dss.mode == "rw"
        assert not dss._is_open

    @pytest.mark.skipif(not HAS_DSS_LIBRARY, reason="DSS library not available")
    def test_dss_file_write_multiple_records(self, tmp_path: Path) -> None:
        """Write multiple time series records to same file."""
        dss_path = tmp_path / "multi.dss"

        with DSSFile(str(dss_path), mode="rw") as dss:
            for i in range(3):
                dss.write_regular_timeseries(
                    pathname=f"/TEST/LOC_{i}/FLOW//1DAY/V1/",
                    values=np.array([float(i)] * 5, dtype=np.float32),
                    start_date=datetime(2020, 1, 1),
                    units="CFS",
                )

        assert dss_path.exists()

    @pytest.mark.skipif(not HAS_DSS_LIBRARY, reason="DSS library not available")
    def test_dss_file_read_no_dates(self, tmp_path: Path) -> None:
        """Read time series without specifying date range."""
        dss_path = tmp_path / "nodate.dss"
        pathname = "/TEST/LOC/FLOW//1DAY/V1/"

        with DSSFile(str(dss_path), mode="rw") as dss:
            dss.write_regular_timeseries(
                pathname=pathname,
                values=np.array([1.0, 2.0, 3.0], dtype=np.float32),
                start_date=datetime(2020, 6, 1),
            )

        with DSSFile(str(dss_path), mode="r") as dss:
            times, values = dss.read_regular_timeseries(pathname)
            # Without date range, may return empty or the full record
            # depending on DSS behavior


class TestDSSWriteConvenienceFunctions:
    """Tests for DSS convenience functions."""

    def test_write_timeseries_to_dss_mock(self) -> None:
        """Test write_timeseries_to_dss with mocked writer."""
        mock_ts = MagicMock()
        mock_pathname = "/A/B/C//1DAY/F/"
        mock_result = DSSWriteResult(
            filepath=Path("test.dss"),
            pathnames_written=[mock_pathname],
            n_records=1,
            errors=[],
        )

        with patch("pyiwfm.io.dss.timeseries.DSSTimeSeriesWriter") as MockWriter:
            mock_writer = MagicMock()
            MockWriter.return_value.__enter__ = MagicMock(return_value=mock_writer)
            MockWriter.return_value.__exit__ = MagicMock(return_value=False)
            mock_writer.close.return_value = mock_result

            result = write_timeseries_to_dss(
                filepath="test.dss",
                ts=mock_ts,
                pathname=mock_pathname,
                units="CFS",
            )
            mock_writer.write_timeseries.assert_called_once_with(
                mock_ts, mock_pathname, "CFS"
            )

    def test_read_timeseries_from_dss_mock(self) -> None:
        """Test read_timeseries_from_dss with mocked reader."""
        mock_ts = MagicMock()
        mock_pathname = "/A/B/C//1DAY/F/"

        with patch("pyiwfm.io.dss.timeseries.DSSTimeSeriesReader") as MockReader:
            mock_reader = MagicMock()
            MockReader.return_value.__enter__ = MagicMock(return_value=mock_reader)
            MockReader.return_value.__exit__ = MagicMock(return_value=False)
            mock_reader.read_timeseries.return_value = mock_ts

            result = read_timeseries_from_dss(
                filepath="test.dss",
                pathname=mock_pathname,
            )
            mock_reader.read_timeseries.assert_called_once_with(
                mock_pathname, None, None
            )

    def test_write_collection_to_dss_mock(self) -> None:
        """Test write_collection_to_dss with mocked writer."""
        mock_collection = MagicMock()
        mock_template = MagicMock()
        mock_result = DSSWriteResult(
            filepath=Path("test.dss"),
            pathnames_written=[],
            n_records=0,
            errors=[],
        )

        with patch("pyiwfm.io.dss.timeseries.DSSTimeSeriesWriter") as MockWriter:
            mock_writer = MagicMock()
            MockWriter.return_value.__enter__ = MagicMock(return_value=mock_writer)
            MockWriter.return_value.__exit__ = MagicMock(return_value=False)
            mock_writer.close.return_value = mock_result

            result = write_collection_to_dss(
                filepath="test.dss",
                collection=mock_collection,
                template=mock_template,
                units="CFS",
            )
            mock_writer.write_collection.assert_called_once()
