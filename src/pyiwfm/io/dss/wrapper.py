"""
HEC-DSS 7 C library wrapper using ctypes.

This module provides a Python interface to the HEC-DSS 7 C library for
reading and writing DSS files. The HEC-DSS library must be installed
separately and the path specified via the HECDSS_LIB environment variable.

Note: HEC-DSS 7 uses 64-bit integers for IFLTAB (250 elements), not the
legacy 32-bit version (600 elements).

Example:
    >>> from pyiwfm.io.dss import DSSFile
    >>> with DSSFile("output.dss", mode="w") as dss:
    ...     dss.write_regular_timeseries(pathname, values, start_date)
"""

from __future__ import annotations

import ctypes
import os
from ctypes import (
    CDLL,
    POINTER,
    Structure,
    byref,
    c_char_p,
    c_double,
    c_float,
    c_int,
    c_int32,
    c_int64,
    c_void_p,
)
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray


# Environment variable for DSS library path
HECDSS_LIB_ENV = "HECDSS_LIB"

# HEC-DSS 7 IFLTAB size (64-bit integers)
IFLTAB_SIZE = 250


class DSSLibraryError(Exception):
    """Error loading or using the HEC-DSS library."""

    pass


class DSSFileError(Exception):
    """Error with DSS file operations."""

    pass


def _get_library_path() -> Path | None:
    """Get the path to the HEC-DSS library."""
    # Check environment variable first
    env_path = os.environ.get(HECDSS_LIB_ENV)
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path

    # Check package lib directory (bundled with pyiwfm)
    package_lib_dir = Path(__file__).parent / "lib"
    if package_lib_dir.exists():
        # Platform-specific library name
        import platform
        if platform.system() == "Windows":
            lib_name = "hecdss.dll"
        elif platform.system() == "Darwin":
            lib_name = "libhecdss.dylib"
        else:
            lib_name = "libhecdss.so"

        package_lib = package_lib_dir / lib_name
        if package_lib.exists():
            # Add lib directory to DLL search path on Windows
            if platform.system() == "Windows":
                try:
                    os.add_dll_directory(str(package_lib_dir))
                except (AttributeError, OSError):
                    pass  # Python < 3.8 or directory already added
            return package_lib

    # Check common system locations
    common_paths = [
        Path("/usr/local/lib/libhecdss.so"),
        Path("/usr/lib/libhecdss.so"),
        Path("C:/Program Files/HEC/HEC-DSS/libhecdss.dll"),
        Path("C:/HEC/HEC-DSS/lib/hecdss.dll"),
    ]

    for path in common_paths:
        if path.exists():
            return path

    return None


def _load_library() -> CDLL | None:
    """Load the HEC-DSS library."""
    lib_path = _get_library_path()

    if lib_path is None:
        return None

    try:
        return ctypes.CDLL(str(lib_path))
    except OSError:
        return None


# Try to load the library at module import
_dss_lib = _load_library()
HAS_DSS_LIBRARY = _dss_lib is not None


# ---------------------------------------------------------------------------
# HEC-DSS 7 zStructTimeSeries partial definition (first 19 fields)
# ---------------------------------------------------------------------------

class _zStructTimeSeries(Structure):
    """Partial ctypes definition of zStructTimeSeries for field access.

    Only defines fields through ``type`` (C-part offset ~96 bytes on 64-bit)
    which is sufficient for reading regular time series data.  The full
    struct has ~50 fields but only these are needed after ``ztsRetrieve``.
    """

    _fields_ = [
        ("structType", c_int),
        ("pathname", c_char_p),
        ("julianBaseDate", c_int),
        ("startJulianDate", c_int),
        ("startTimeSeconds", c_int),
        ("endJulianDate", c_int),
        ("endTimeSeconds", c_int),
        ("timeGranularitySeconds", c_int),
        ("timeIntervalSeconds", c_int),
        ("timeOffsetSeconds", c_int),
        ("times", c_void_p),
        ("boolRetrieveAllTimes", c_int),
        ("numberValues", c_int),
        ("sizeEachValueRead", c_int),
        ("precision", c_int),
        ("floatValues", c_void_p),
        ("doubleValues", c_void_p),
        ("units", c_char_p),
        ("type", c_char_p),
    ]


def _configure_argtypes(lib: CDLL) -> None:
    """Configure argtypes and restype for HEC-DSS 7 library functions.

    Setting explicit argtypes ensures ctypes marshals arguments correctly
    (especially pointer widths on 64-bit) and prevents access violations.
    """
    # zopenExtended(long long *ifltab, const char *dssFilename) -> int
    lib.zopenExtended.argtypes = [POINTER(c_int64), c_char_p]
    lib.zopenExtended.restype = c_int

    # zclose(long long *ifltab) -> void
    lib.zclose.argtypes = [POINTER(c_int64)]
    lib.zclose.restype = None

    # zstructTsNewRegFloats(pathname, floatValues, numberValues,
    #     startDate, startTime, units, type) -> zStructTimeSeries*
    lib.zstructTsNewRegFloats.argtypes = [
        c_char_p, POINTER(c_float), c_int,
        c_char_p, c_char_p, c_char_p, c_char_p,
    ]
    lib.zstructTsNewRegFloats.restype = c_void_p

    # zstructTsNewTimes(pathname, startDate, startTime,
    #     endDate, endTime) -> zStructTimeSeries*
    lib.zstructTsNewTimes.argtypes = [
        c_char_p, c_char_p, c_char_p, c_char_p, c_char_p,
    ]
    lib.zstructTsNewTimes.restype = c_void_p

    # ztsStore(long long *ifltab, zStructTimeSeries *tss,
    #     int storageFlag) -> int
    lib.ztsStore.argtypes = [POINTER(c_int64), c_void_p, c_int]
    lib.ztsStore.restype = c_int

    # ztsRetrieve(long long *ifltab, zStructTimeSeries *tss,
    #     int retrieveFlag, int retrieveDoublesFlag,
    #     int boolRetrieveQualityNotes) -> int
    lib.ztsRetrieve.argtypes = [
        POINTER(c_int64), c_void_p, c_int, c_int, c_int,
    ]
    lib.ztsRetrieve.restype = c_int

    # zstructFree(void *zstruct) -> void
    lib.zstructFree.argtypes = [c_void_p]
    lib.zstructFree.restype = None


if _dss_lib is not None:
    _configure_argtypes(_dss_lib)


def check_dss_available() -> None:
    """
    Check if the DSS library is available.

    Raises:
        DSSLibraryError: If the library is not available
    """
    if not HAS_DSS_LIBRARY:
        raise DSSLibraryError(
            f"HEC-DSS library not found. Set {HECDSS_LIB_ENV} environment "
            "variable to the library path."
        )


@dataclass
class DSSTimeSeriesInfo:
    """
    Information about a DSS time series record.

    Attributes:
        pathname: Full DSS pathname
        start_date: Start datetime
        end_date: End datetime
        n_values: Number of values
        units: Units string
        data_type: Data type string
    """

    pathname: str
    start_date: datetime | None = None
    end_date: datetime | None = None
    n_values: int = 0
    units: str = ""
    data_type: str = ""


class DSSFile:
    """
    Context manager for HEC-DSS file operations.

    Provides methods for reading and writing time series data to DSS files.

    Example:
        >>> with DSSFile("data.dss", mode="rw") as dss:
        ...     times, values = dss.read_regular_timeseries("/A/B/C/D/E/F/")
        ...     dss.write_regular_timeseries("/A/B/C2/D/E/F/", values * 2, times[0])
    """

    def __init__(self, filepath: Path | str, mode: str = "r") -> None:
        """
        Initialize DSS file handle.

        Args:
            filepath: Path to DSS file
            mode: File mode ('r' = read, 'w' = write, 'rw' = read/write)
        """
        check_dss_available()

        self.filepath = Path(filepath)
        self.mode = mode
        self._ifltab: NDArray[np.int64] | None = None
        self._is_open = False

    def __enter__(self) -> "DSSFile":
        """Open the DSS file."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close the DSS file."""
        self.close()

    def open(self) -> None:
        """Open the DSS file."""
        if self._is_open:
            return

        # Create IFLTAB array (64-bit integers for DSS 7)
        self._ifltab = np.zeros(IFLTAB_SIZE, dtype=np.int64)

        # Determine access mode
        if self.mode == "r":
            access = 0  # Read only
        elif self.mode == "w":
            access = 1  # Write (create new)
        else:
            access = 2  # Read/write

        # Call zopenExtended (DSS 7 API)
        filepath_bytes = str(self.filepath).encode("utf-8")
        # zopenExtended signature: int zopenExtended(long long *ifltab, const char *dssFilename)
        # Returns status: 0 = success
        status = _dss_lib.zopenExtended(
            self._ifltab.ctypes.data_as(POINTER(c_int64)),
            filepath_bytes,
        )

        if status != 0:
            raise DSSFileError(f"Failed to open DSS file: {self.filepath} (status={status})")

        self._is_open = True

    def close(self) -> None:
        """Close the DSS file."""
        if not self._is_open:
            return

        if self._ifltab is not None:
            _dss_lib.zclose(self._ifltab.ctypes.data_as(POINTER(c_int64)))

        self._ifltab = None
        self._is_open = False

    def write_regular_timeseries(
        self,
        pathname: str,
        values: NDArray[np.float64] | Sequence[float],
        start_date: datetime,
        units: str = "",
        data_type: str = "INST-VAL",
    ) -> None:
        """
        Write a regular-interval time series to the DSS file.

        Uses the HEC-DSS 7 constructor pattern:
        ``zstructTsNewRegFloats()`` -> ``ztsStore()`` -> ``zstructFree()``

        Args:
            pathname: DSS pathname
            values: Array of values
            start_date: Start datetime
            units: Units string
            data_type: Data type (e.g., "INST-VAL", "PER-AVER")
        """
        if not self._is_open:
            raise DSSFileError("DSS file not open")

        if "w" not in self.mode:
            raise DSSFileError("DSS file not open for writing")

        values_f32 = np.asarray(values, dtype=np.float32)
        n_values = len(values_f32)

        # Format start date for DSS
        start_str = start_date.strftime("%d%b%Y").upper().encode("utf-8")
        start_time = start_date.strftime("%H%M").encode("utf-8")

        # Create time series struct via constructor
        tss = _dss_lib.zstructTsNewRegFloats(
            pathname.encode("utf-8"),
            values_f32.ctypes.data_as(POINTER(c_float)),
            c_int(n_values),
            start_str,
            start_time,
            units.encode("utf-8"),
            data_type.encode("utf-8"),
        )

        if not tss:
            raise DSSFileError(
                f"Failed to create time series struct for: {pathname}"
            )

        try:
            status = _dss_lib.ztsStore(
                self._ifltab.ctypes.data_as(POINTER(c_int64)),
                tss,
                c_int(0),  # storageFlag: 0 = store regular
            )

            if status != 0:
                raise DSSFileError(
                    f"Failed to write time series: {pathname} (status={status})"
                )
        finally:
            _dss_lib.zstructFree(tss)

    def read_regular_timeseries(
        self,
        pathname: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> tuple[list[datetime], NDArray[np.float64]]:
        """
        Read a regular-interval time series from the DSS file.

        Uses the HEC-DSS 7 constructor pattern:
        ``zstructTsNewTimes()`` -> ``ztsRetrieve()`` -> extract fields
        -> ``zstructFree()``

        Args:
            pathname: DSS pathname
            start_date: Start datetime (optional)
            end_date: End datetime (optional)

        Returns:
            Tuple of (times, values)
        """
        if not self._is_open:
            raise DSSFileError("DSS file not open")

        # Format date range
        start_str = (
            start_date.strftime("%d%b%Y").upper().encode("utf-8")
            if start_date else b""
        )
        start_time = (
            start_date.strftime("%H%M").encode("utf-8")
            if start_date else b"0000"
        )
        end_str = (
            end_date.strftime("%d%b%Y").upper().encode("utf-8")
            if end_date else b""
        )
        end_time = (
            end_date.strftime("%H%M").encode("utf-8")
            if end_date else b"2400"
        )

        # Create time series struct for retrieval
        tss = _dss_lib.zstructTsNewTimes(
            pathname.encode("utf-8"),
            start_str, start_time,
            end_str, end_time,
        )

        if not tss:
            raise DSSFileError(
                f"Failed to create time series struct for: {pathname}"
            )

        try:
            status = _dss_lib.ztsRetrieve(
                self._ifltab.ctypes.data_as(POINTER(c_int64)),
                tss,
                c_int(-1),  # retrieveFlag: -1 = retrieve all
                c_int(0),   # retrieveDoublesFlag: 0 = use floats
                c_int(0),   # boolRetrieveQualityNotes: 0 = skip
            )

            # Cast to struct for field access
            tss_struct = ctypes.cast(
                tss, POINTER(_zStructTimeSeries)
            ).contents

            n_values = tss_struct.numberValues

            # Return empty for missing records or zero-length data
            if n_values <= 0:
                return [], np.array([], dtype=np.float64)

            if status != 0:
                raise DSSFileError(
                    f"Failed to read time series: {pathname} (status={status})"
                )

            # Extract float values from struct
            float_ptr = ctypes.cast(
                tss_struct.floatValues, POINTER(c_float * n_values)
            )
            values = np.array(float_ptr.contents[:], dtype=np.float64)

            # Build datetimes from struct fields
            base_date = datetime(1899, 12, 31)  # DSS epoch
            start_dt = base_date + timedelta(
                days=tss_struct.startJulianDate,
                seconds=tss_struct.startTimeSeconds,
            )
            interval_secs = tss_struct.timeIntervalSeconds

            if interval_secs > 0:
                times = [
                    start_dt + timedelta(seconds=i * interval_secs)
                    for i in range(n_values)
                ]
            else:
                # Irregular time series - use times array
                if tss_struct.times:
                    granularity = max(tss_struct.timeGranularitySeconds, 1)
                    times_ptr = ctypes.cast(
                        tss_struct.times, POINTER(c_int32 * n_values)
                    )
                    times = [
                        base_date + timedelta(
                            seconds=int(t) * granularity
                        )
                        for t in times_ptr.contents
                    ]
                else:
                    times = [start_dt] * n_values

            return times, values
        finally:
            _dss_lib.zstructFree(tss)

    def get_timeseries_info(self, pathname: str) -> DSSTimeSeriesInfo:
        """
        Get information about a time series record.

        Retrieves the record and extracts metadata (number of values,
        units, data type, date range).

        Args:
            pathname: DSS pathname

        Returns:
            DSSTimeSeriesInfo object
        """
        if not self._is_open:
            raise DSSFileError("DSS file not open")

        # Create struct for retrieval (no date window = retrieve all)
        tss = _dss_lib.zstructTsNewTimes(
            pathname.encode("utf-8"),
            b"", b"0000", b"", b"2400",
        )

        if not tss:
            return DSSTimeSeriesInfo(pathname=pathname, n_values=0)

        try:
            status = _dss_lib.ztsRetrieve(
                self._ifltab.ctypes.data_as(POINTER(c_int64)),
                tss,
                c_int(-1),  # retrieve all
                c_int(0),   # floats
                c_int(0),   # no quality notes
            )

            tss_struct = ctypes.cast(
                tss, POINTER(_zStructTimeSeries)
            ).contents

            n_values = tss_struct.numberValues

            if n_values <= 0 or status != 0:
                return DSSTimeSeriesInfo(pathname=pathname, n_values=0)

            base_date = datetime(1899, 12, 31)
            start_date = None
            end_date = None

            if tss_struct.startJulianDate > 0:
                start_date = base_date + timedelta(
                    days=tss_struct.startJulianDate,
                    seconds=tss_struct.startTimeSeconds,
                )
                interval_secs = tss_struct.timeIntervalSeconds
                if interval_secs > 0:
                    end_date = start_date + timedelta(
                        seconds=(n_values - 1) * interval_secs,
                    )

            units_str = ""
            if tss_struct.units:
                units_str = tss_struct.units.decode("utf-8", errors="replace")

            type_str = ""
            if tss_struct.type:
                type_str = tss_struct.type.decode("utf-8", errors="replace")

            return DSSTimeSeriesInfo(
                pathname=pathname,
                start_date=start_date,
                end_date=end_date,
                n_values=n_values,
                units=units_str,
                data_type=type_str,
            )
        finally:
            _dss_lib.zstructFree(tss)

    def catalog(self, pattern: str = "") -> list[str]:
        """
        Get a list of pathnames in the file.

        Args:
            pattern: Optional pattern to filter results

        Returns:
            List of pathname strings
        """
        if not self._is_open:
            raise DSSFileError("DSS file not open")

        # This would call zcatalog in a full implementation
        return []

    def _julian_to_datetime(
        self, julian_days: NDArray[np.int32], minutes: NDArray[np.int32]
    ) -> list[datetime]:
        """Convert DSS julian dates to datetime objects."""
        base_date = datetime(1899, 12, 31)  # DSS epoch
        return [
            base_date + timedelta(days=int(jd), minutes=int(mins))
            for jd, mins in zip(julian_days, minutes)
        ]


class DSSFileMock:
    """
    Mock DSS file for when the library is not available.

    Allows code to run without the DSS library, but operations will
    raise informative errors.
    """

    def __init__(self, filepath: Path | str, mode: str = "r") -> None:
        """Initialize mock DSS file."""
        self.filepath = Path(filepath)
        self.mode = mode

    def __enter__(self) -> "DSSFileMock":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    def open(self) -> None:
        raise DSSLibraryError("HEC-DSS library not available")

    def close(self) -> None:
        pass

    def write_regular_timeseries(self, *args, **kwargs) -> None:
        raise DSSLibraryError("HEC-DSS library not available")

    def read_regular_timeseries(self, *args, **kwargs) -> tuple:
        raise DSSLibraryError("HEC-DSS library not available")


def get_dss_file_class() -> type:
    """
    Get the appropriate DSS file class.

    Returns DSSFile if library is available, DSSFileMock otherwise.
    """
    if HAS_DSS_LIBRARY:
        return DSSFile
    return DSSFileMock


# Export the appropriate class
DSSFileClass = get_dss_file_class()
