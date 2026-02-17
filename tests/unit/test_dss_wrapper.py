"""
Tests for pyiwfm.io.dss.wrapper module.

Tests the ctypes wrapper for HEC-DSS 7 C library. Since the actual library
may not be available, we use mocks for most tests.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pyiwfm.io.dss.wrapper import (
    HECDSS_LIB_ENV,
    IFLTAB_SIZE,
    DSSFile,
    DSSFileError,
    DSSFileMock,
    DSSLibraryError,
    DSSTimeSeriesInfo,
    _get_library_path,
    _load_library,
    check_dss_available,
    get_dss_file_class,
)


class TestDSSTimeSeriesInfo:
    """Tests for DSSTimeSeriesInfo dataclass."""

    def test_creation_minimal(self) -> None:
        """Test creating info with minimal data."""
        info = DSSTimeSeriesInfo(pathname="/A/B/C/D/E/F/")
        assert info.pathname == "/A/B/C/D/E/F/"
        assert info.start_date is None
        assert info.end_date is None
        assert info.n_values == 0
        assert info.units == ""
        assert info.data_type == ""

    def test_creation_full(self) -> None:
        """Test creating info with all fields."""
        start = datetime(2020, 1, 1)
        end = datetime(2020, 12, 31)
        info = DSSTimeSeriesInfo(
            pathname="/PROJECT/LOC/FLOW//1DAY/VER/",
            start_date=start,
            end_date=end,
            n_values=365,
            units="CFS",
            data_type="INST-VAL",
        )
        assert info.pathname == "/PROJECT/LOC/FLOW//1DAY/VER/"
        assert info.start_date == start
        assert info.end_date == end
        assert info.n_values == 365
        assert info.units == "CFS"
        assert info.data_type == "INST-VAL"


class TestGetLibraryPath:
    """Tests for _get_library_path function."""

    def test_environment_variable_set(self, tmp_path: Path) -> None:
        """Test finding library via environment variable."""
        lib_path = tmp_path / "libhecdss.so"
        lib_path.touch()

        with patch.dict(os.environ, {HECDSS_LIB_ENV: str(lib_path)}):
            result = _get_library_path()
            assert result == lib_path

    def test_environment_variable_not_exists(self) -> None:
        """Test with environment variable pointing to non-existent file."""
        with patch.dict(os.environ, {HECDSS_LIB_ENV: "/nonexistent/path.so"}):
            result = _get_library_path()
            # Should still check common paths
            assert result is None or isinstance(result, Path)

    def test_no_environment_variable(self) -> None:
        """Test behavior without environment variable."""
        env = dict(os.environ)
        env.pop(HECDSS_LIB_ENV, None)
        with patch.dict(os.environ, env, clear=True):
            result = _get_library_path()
            # May find a library in common paths or return None
            assert result is None or isinstance(result, Path)


class TestLoadLibrary:
    """Tests for _load_library function."""

    def test_no_library_available(self) -> None:
        """Test when no library is available."""
        with patch("pyiwfm.io.dss.wrapper._get_library_path", return_value=None):
            result = _load_library()
            assert result is None

    def test_library_load_error(self, tmp_path: Path) -> None:
        """Test when library file exists but can't be loaded."""
        fake_lib = tmp_path / "fake.so"
        fake_lib.write_text("not a library")

        with patch("pyiwfm.io.dss.wrapper._get_library_path", return_value=fake_lib):
            result = _load_library()
            assert result is None


class TestCheckDSSAvailable:
    """Tests for check_dss_available function."""

    def test_library_not_available(self) -> None:
        """Test error when library not available."""
        with patch("pyiwfm.io.dss.wrapper.HAS_DSS_LIBRARY", False):
            with pytest.raises(DSSLibraryError) as exc_info:
                check_dss_available()
            assert HECDSS_LIB_ENV in str(exc_info.value)

    def test_library_available(self) -> None:
        """Test no error when library is available."""
        with patch("pyiwfm.io.dss.wrapper.HAS_DSS_LIBRARY", True):
            check_dss_available()  # Should not raise


class TestGetDSSFileClass:
    """Tests for get_dss_file_class function."""

    def test_returns_mock_when_unavailable(self) -> None:
        """Test returns mock class when library unavailable."""
        with patch("pyiwfm.io.dss.wrapper.HAS_DSS_LIBRARY", False):
            cls = get_dss_file_class()
            assert cls is DSSFileMock

    def test_returns_real_when_available(self) -> None:
        """Test returns real class when library available."""
        with patch("pyiwfm.io.dss.wrapper.HAS_DSS_LIBRARY", True):
            cls = get_dss_file_class()
            assert cls is DSSFile


class TestDSSFileMock:
    """Tests for DSSFileMock class."""

    def test_init(self, tmp_path: Path) -> None:
        """Test mock initialization."""
        mock = DSSFileMock(tmp_path / "test.dss", mode="r")
        assert mock.filepath == tmp_path / "test.dss"
        assert mock.mode == "r"

    def test_context_manager(self, tmp_path: Path) -> None:
        """Test mock as context manager."""
        with DSSFileMock(tmp_path / "test.dss") as mock:
            assert mock.filepath == tmp_path / "test.dss"

    def test_open_raises(self, tmp_path: Path) -> None:
        """Test open raises DSSLibraryError."""
        mock = DSSFileMock(tmp_path / "test.dss")
        with pytest.raises(DSSLibraryError):
            mock.open()

    def test_close_no_error(self, tmp_path: Path) -> None:
        """Test close doesn't raise."""
        mock = DSSFileMock(tmp_path / "test.dss")
        mock.close()  # Should not raise

    def test_write_regular_timeseries_raises(self, tmp_path: Path) -> None:
        """Test write_regular_timeseries raises DSSLibraryError."""
        mock = DSSFileMock(tmp_path / "test.dss", mode="w")
        with pytest.raises(DSSLibraryError):
            mock.write_regular_timeseries(
                "/A/B/C/D/E/F/",
                np.array([1.0, 2.0, 3.0]),
                datetime.now(),
            )

    def test_read_regular_timeseries_raises(self, tmp_path: Path) -> None:
        """Test read_regular_timeseries raises DSSLibraryError."""
        mock = DSSFileMock(tmp_path / "test.dss", mode="r")
        with pytest.raises(DSSLibraryError):
            mock.read_regular_timeseries("/A/B/C/D/E/F/")


class TestDSSFileInit:
    """Tests for DSSFile initialization."""

    def test_init_without_library(self, tmp_path: Path) -> None:
        """Test initialization fails when library unavailable."""
        with patch("pyiwfm.io.dss.wrapper.HAS_DSS_LIBRARY", False):
            with pytest.raises(DSSLibraryError):
                DSSFile(tmp_path / "test.dss")

    def test_init_with_library_mock(self, tmp_path: Path) -> None:
        """Test initialization with mocked library."""
        with patch("pyiwfm.io.dss.wrapper.HAS_DSS_LIBRARY", True):
            with patch("pyiwfm.io.dss.wrapper.check_dss_available"):
                dss = DSSFile(tmp_path / "test.dss", mode="r")
                assert dss.filepath == tmp_path / "test.dss"
                assert dss.mode == "r"
                assert not dss._is_open


class TestDSSFileMethods:
    """Tests for DSSFile methods with mocked library."""

    @pytest.fixture
    def mock_lib(self) -> MagicMock:
        """Create mock DSS library."""
        mock = MagicMock()
        mock.zopenExtended.return_value = 0  # DSS 7 uses zopenExtended
        mock.zclose.return_value = 0
        mock.ztsStore.return_value = 0
        mock.ztsRetrieve.return_value = 0
        return mock

    @pytest.fixture
    def dss_file(self, tmp_path: Path, mock_lib: MagicMock) -> DSSFile:
        """Create DSSFile with mocked library."""
        with patch("pyiwfm.io.dss.wrapper.HAS_DSS_LIBRARY", True):
            with patch("pyiwfm.io.dss.wrapper.check_dss_available"):
                with patch("pyiwfm.io.dss.wrapper._dss_lib", mock_lib):
                    dss = DSSFile(tmp_path / "test.dss", mode="rw")
                    return dss

    def test_open_creates_ifltab(self, dss_file: DSSFile, mock_lib: MagicMock) -> None:
        """Test open creates IFLTAB array."""
        with patch("pyiwfm.io.dss.wrapper._dss_lib", mock_lib):
            dss_file.open()
            assert dss_file._ifltab is not None
            assert len(dss_file._ifltab) == IFLTAB_SIZE
            assert dss_file._ifltab.dtype == np.int64
            assert dss_file._is_open

    def test_open_read_mode(self, tmp_path: Path, mock_lib: MagicMock) -> None:
        """Test open with read mode."""
        with patch("pyiwfm.io.dss.wrapper.HAS_DSS_LIBRARY", True):
            with patch("pyiwfm.io.dss.wrapper.check_dss_available"):
                with patch("pyiwfm.io.dss.wrapper._dss_lib", mock_lib):
                    dss = DSSFile(tmp_path / "test.dss", mode="r")
                    dss.open()
                    # Verify zopenExtended was called
                    mock_lib.zopenExtended.assert_called_once()

    def test_open_write_mode(self, tmp_path: Path, mock_lib: MagicMock) -> None:
        """Test open with write mode."""
        with patch("pyiwfm.io.dss.wrapper.HAS_DSS_LIBRARY", True):
            with patch("pyiwfm.io.dss.wrapper.check_dss_available"):
                with patch("pyiwfm.io.dss.wrapper._dss_lib", mock_lib):
                    dss = DSSFile(tmp_path / "test.dss", mode="w")
                    dss.open()
                    # Verify zopenExtended was called
                    mock_lib.zopenExtended.assert_called_once()

    def test_open_failure(self, dss_file: DSSFile, mock_lib: MagicMock) -> None:
        """Test open failure."""
        mock_lib.zopenExtended.return_value = -1
        with patch("pyiwfm.io.dss.wrapper._dss_lib", mock_lib):
            with pytest.raises(DSSFileError) as exc_info:
                dss_file.open()
            assert "Failed to open" in str(exc_info.value)

    def test_close(self, dss_file: DSSFile, mock_lib: MagicMock) -> None:
        """Test close."""
        with patch("pyiwfm.io.dss.wrapper._dss_lib", mock_lib):
            dss_file.open()
            dss_file.close()
            assert not dss_file._is_open
            assert dss_file._ifltab is None
            mock_lib.zclose.assert_called_once()

    def test_close_not_open(self, dss_file: DSSFile, mock_lib: MagicMock) -> None:
        """Test close when not open."""
        with patch("pyiwfm.io.dss.wrapper._dss_lib", mock_lib):
            dss_file.close()  # Should not raise
            mock_lib.zclose.assert_not_called()

    def test_context_manager(self, tmp_path: Path, mock_lib: MagicMock) -> None:
        """Test context manager."""
        with patch("pyiwfm.io.dss.wrapper.HAS_DSS_LIBRARY", True):
            with patch("pyiwfm.io.dss.wrapper.check_dss_available"):
                with patch("pyiwfm.io.dss.wrapper._dss_lib", mock_lib):
                    with DSSFile(tmp_path / "test.dss", mode="rw") as dss:
                        assert dss._is_open
                    assert not dss._is_open

    def test_write_not_open(self, dss_file: DSSFile) -> None:
        """Test write when not open."""
        with pytest.raises(DSSFileError) as exc_info:
            dss_file.write_regular_timeseries(
                "/A/B/C/D/E/F/",
                np.array([1.0, 2.0]),
                datetime(2020, 1, 1),
            )
        assert "not open" in str(exc_info.value)

    def test_write_not_write_mode(self, tmp_path: Path, mock_lib: MagicMock) -> None:
        """Test write when not in write mode."""
        with patch("pyiwfm.io.dss.wrapper.HAS_DSS_LIBRARY", True):
            with patch("pyiwfm.io.dss.wrapper.check_dss_available"):
                with patch("pyiwfm.io.dss.wrapper._dss_lib", mock_lib):
                    with DSSFile(tmp_path / "test.dss", mode="r") as dss:
                        with pytest.raises(DSSFileError) as exc_info:
                            dss.write_regular_timeseries(
                                "/A/B/C/D/E/F/",
                                np.array([1.0, 2.0]),
                                datetime(2020, 1, 1),
                            )
                        assert "not open for writing" in str(exc_info.value)

    def test_write_regular_timeseries(self, tmp_path: Path, mock_lib: MagicMock) -> None:
        """Test write_regular_timeseries."""
        with patch("pyiwfm.io.dss.wrapper.HAS_DSS_LIBRARY", True):
            with patch("pyiwfm.io.dss.wrapper.check_dss_available"):
                with patch("pyiwfm.io.dss.wrapper._dss_lib", mock_lib):
                    with DSSFile(tmp_path / "test.dss", mode="rw") as dss:
                        dss.write_regular_timeseries(
                            "/PROJECT/LOC/FLOW//1DAY/VER/",
                            np.array([1.0, 2.0, 3.0]),
                            datetime(2020, 1, 1, 12, 0),
                            units="CFS",
                            data_type="INST-VAL",
                        )
                        mock_lib.ztsStore.assert_called_once()

    def test_write_failure(self, tmp_path: Path, mock_lib: MagicMock) -> None:
        """Test write failure."""
        mock_lib.ztsStore.return_value = -1
        with patch("pyiwfm.io.dss.wrapper.HAS_DSS_LIBRARY", True):
            with patch("pyiwfm.io.dss.wrapper.check_dss_available"):
                with patch("pyiwfm.io.dss.wrapper._dss_lib", mock_lib):
                    with DSSFile(tmp_path / "test.dss", mode="rw") as dss:
                        with pytest.raises(DSSFileError) as exc_info:
                            dss.write_regular_timeseries(
                                "/A/B/C/D/E/F/",
                                np.array([1.0]),
                                datetime(2020, 1, 1),
                            )
                        assert "Failed to write" in str(exc_info.value)

    def test_read_not_open(self, dss_file: DSSFile) -> None:
        """Test read when not open."""
        with pytest.raises(DSSFileError) as exc_info:
            dss_file.read_regular_timeseries("/A/B/C/D/E/F/")
        assert "not open" in str(exc_info.value)

    def test_get_timeseries_info_not_open(self, dss_file: DSSFile) -> None:
        """Test get_timeseries_info when not open."""
        with pytest.raises(DSSFileError) as exc_info:
            dss_file.get_timeseries_info("/A/B/C/D/E/F/")
        assert "not open" in str(exc_info.value)

    def test_get_timeseries_info(self, tmp_path: Path, mock_lib: MagicMock) -> None:
        """Test get_timeseries_info returns empty info for missing record."""
        # zstructTsNewTimes returns None → early return with default info
        mock_lib.zstructTsNewTimes.return_value = None
        with patch("pyiwfm.io.dss.wrapper.HAS_DSS_LIBRARY", True):
            with patch("pyiwfm.io.dss.wrapper.check_dss_available"):
                with patch("pyiwfm.io.dss.wrapper._dss_lib", mock_lib):
                    with DSSFile(tmp_path / "test.dss", mode="r") as dss:
                        info = dss.get_timeseries_info("/A/B/C/D/E/F/")
                        assert isinstance(info, DSSTimeSeriesInfo)
                        assert info.pathname == "/A/B/C/D/E/F/"
                        assert info.n_values == 0

    def test_catalog_not_open(self, dss_file: DSSFile) -> None:
        """Test catalog when not open."""
        with pytest.raises(DSSFileError) as exc_info:
            dss_file.catalog()
        assert "not open" in str(exc_info.value)

    def test_catalog(self, tmp_path: Path, mock_lib: MagicMock) -> None:
        """Test catalog returns list."""
        with patch("pyiwfm.io.dss.wrapper.HAS_DSS_LIBRARY", True):
            with patch("pyiwfm.io.dss.wrapper.check_dss_available"):
                with patch("pyiwfm.io.dss.wrapper._dss_lib", mock_lib):
                    with DSSFile(tmp_path / "test.dss", mode="r") as dss:
                        result = dss.catalog()
                        assert isinstance(result, list)


class TestDSSFileJulianConversion:
    """Tests for DSSFile julian date conversion."""

    @pytest.fixture
    def mock_lib(self) -> MagicMock:
        """Create mock DSS library."""
        mock = MagicMock()
        mock.zopenExtended.return_value = 0
        mock.zclose.return_value = 0
        return mock

    def test_julian_to_datetime(self, tmp_path: Path, mock_lib: MagicMock) -> None:
        """Test _julian_to_datetime conversion."""
        with patch("pyiwfm.io.dss.wrapper.HAS_DSS_LIBRARY", True):
            with patch("pyiwfm.io.dss.wrapper.check_dss_available"):
                with patch("pyiwfm.io.dss.wrapper._dss_lib", mock_lib):
                    dss = DSSFile(tmp_path / "test.dss")
                    # DSS epoch is 1899-12-31
                    # Jan 1, 2020 = 43831 days from epoch
                    julian = np.array([43831], dtype=np.int32)
                    minutes = np.array([720], dtype=np.int32)  # 12:00
                    times = dss._julian_to_datetime(julian, minutes)
                    assert len(times) == 1

    def test_julian_to_datetime_multiple(self, tmp_path: Path, mock_lib: MagicMock) -> None:
        """Test _julian_to_datetime with multiple values."""
        with patch("pyiwfm.io.dss.wrapper.HAS_DSS_LIBRARY", True):
            with patch("pyiwfm.io.dss.wrapper.check_dss_available"):
                with patch("pyiwfm.io.dss.wrapper._dss_lib", mock_lib):
                    dss = DSSFile(tmp_path / "test.dss")
                    julian = np.array([1, 2, 3], dtype=np.int32)
                    minutes = np.array([0, 0, 0], dtype=np.int32)
                    times = dss._julian_to_datetime(julian, minutes)
                    assert len(times) == 3


class TestDSSConstants:
    """Tests for DSS module constants."""

    def test_ifltab_size(self) -> None:
        """Test IFLTAB size constant."""
        assert IFLTAB_SIZE == 250

    def test_env_variable_name(self) -> None:
        """Test environment variable name."""
        assert HECDSS_LIB_ENV == "HECDSS_LIB"


class TestDSSExceptions:
    """Tests for DSS exception classes."""

    def test_library_error(self) -> None:
        """Test DSSLibraryError."""
        error = DSSLibraryError("Library not found")
        assert str(error) == "Library not found"
        assert isinstance(error, Exception)

    def test_file_error(self) -> None:
        """Test DSSFileError."""
        error = DSSFileError("File operation failed")
        assert str(error) == "File operation failed"
        assert isinstance(error, Exception)
