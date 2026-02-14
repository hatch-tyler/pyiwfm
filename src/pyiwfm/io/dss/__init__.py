"""
HEC-DSS support for pyiwfm.

This package provides Python bindings for reading and writing HEC-DSS 7 files,
which are commonly used for time series data storage in water resources modeling.

The HEC-DSS library must be installed separately. Set the HECDSS_LIB environment
variable to point to the library location.

Example:
    >>> from pyiwfm.io.dss import DSSFile, DSSPathname
    >>>
    >>> # Build a pathname
    >>> pathname = DSSPathname.build(
    ...     project="IWFM_MODEL",
    ...     location="STREAM_NODE_1",
    ...     parameter="flow",
    ...     interval="daily",
    ... )
    >>>
    >>> # Write time series
    >>> with DSSFile("output.dss", mode="w") as dss:
    ...     dss.write_regular_timeseries(str(pathname), values, start_date)
"""

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
)

from pyiwfm.io.dss.wrapper import (
    DSSFile,
    DSSFileMock,
    DSSFileClass,
    DSSFileError,
    DSSLibraryError,
    DSSTimeSeriesInfo,
    HAS_DSS_LIBRARY,
    check_dss_available,
    HECDSS_LIB_ENV,
)

from pyiwfm.io.dss.timeseries import (
    DSSTimeSeriesWriter,
    DSSTimeSeriesReader,
    DSSWriteResult,
    write_timeseries_to_dss,
    read_timeseries_from_dss,
    write_collection_to_dss,
)


__all__ = [
    # Pathname utilities
    "DSSPathname",
    "DSSPathnameTemplate",
    "format_dss_date",
    "format_dss_date_range",
    "parse_dss_date",
    "interval_to_minutes",
    "minutes_to_interval",
    "VALID_INTERVALS",
    "PARAMETER_CODES",
    # Wrapper
    "DSSFile",
    "DSSFileMock",
    "DSSFileClass",
    "DSSFileError",
    "DSSLibraryError",
    "DSSTimeSeriesInfo",
    "HAS_DSS_LIBRARY",
    "check_dss_available",
    "HECDSS_LIB_ENV",
    # Time series utilities
    "DSSTimeSeriesWriter",
    "DSSTimeSeriesReader",
    "DSSWriteResult",
    "write_timeseries_to_dss",
    "read_timeseries_from_dss",
    "write_collection_to_dss",
]
