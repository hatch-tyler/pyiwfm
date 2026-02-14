"""
HEC-DSS time series read/write utilities.

This module provides high-level functions for reading and writing time series
data to HEC-DSS files, with integration to pyiwfm's TimeSeries class.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from pyiwfm.core.timeseries import TimeSeries, TimeSeriesCollection
from pyiwfm.io.dss.pathname import (
    DSSPathname,
    DSSPathnameTemplate,
    format_dss_date,
    format_dss_date_range,
    minutes_to_interval,
)
from pyiwfm.io.dss.wrapper import (
    DSSFile,
    DSSFileError,
    DSSLibraryError,
    HAS_DSS_LIBRARY,
    check_dss_available,
)


@dataclass
class DSSWriteResult:
    """
    Result of a DSS write operation.

    Attributes:
        filepath: Path to DSS file
        pathnames_written: List of pathnames written
        n_records: Number of records written
        errors: List of error messages (if any)
    """

    filepath: Path
    pathnames_written: list[str]
    n_records: int
    errors: list[str]

    @property
    def success(self) -> bool:
        """Return True if write was successful (no errors)."""
        return len(self.errors) == 0


class DSSTimeSeriesWriter:
    """
    High-level writer for time series data to HEC-DSS files.

    Example:
        >>> writer = DSSTimeSeriesWriter(Path("output.dss"))
        >>> template = DSSPathnameTemplate(
        ...     a_part="PROJECT",
        ...     c_part="FLOW",
        ...     e_part="1DAY",
        ... )
        >>> writer.write_timeseries(ts, template.make_pathname(location="STREAM_01"))
        >>> writer.close()
    """

    def __init__(self, filepath: Path | str) -> None:
        """
        Initialize the DSS time series writer.

        Args:
            filepath: Path to DSS file (will be created if doesn't exist)
        """
        check_dss_available()
        self.filepath = Path(filepath)
        self._dss: DSSFile | None = None
        self._pathnames_written: list[str] = []
        self._errors: list[str] = []

    def __enter__(self) -> "DSSTimeSeriesWriter":
        """Open the DSS file."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close the DSS file."""
        self.close()

    def open(self) -> None:
        """Open the DSS file for writing."""
        if self._dss is not None:
            return
        self._dss = DSSFile(self.filepath, mode="rw")
        self._dss.open()

    def close(self) -> DSSWriteResult:
        """
        Close the DSS file and return write result.

        Returns:
            DSSWriteResult with summary of written data
        """
        if self._dss is not None:
            self._dss.close()
            self._dss = None

        return DSSWriteResult(
            filepath=self.filepath,
            pathnames_written=self._pathnames_written.copy(),
            n_records=len(self._pathnames_written),
            errors=self._errors.copy(),
        )

    def write_timeseries(
        self,
        ts: TimeSeries,
        pathname: DSSPathname | str,
        units: str | None = None,
        data_type: str = "INST-VAL",
    ) -> bool:
        """
        Write a TimeSeries to the DSS file.

        Args:
            ts: TimeSeries object to write
            pathname: DSS pathname for the record
            units: Units string (defaults to ts.units)
            data_type: Data type (e.g., "INST-VAL", "PER-AVER")

        Returns:
            True if successful, False otherwise
        """
        if self._dss is None:
            self.open()

        if isinstance(pathname, str):
            pathname = DSSPathname.from_string(pathname)

        # Get start date
        start_date = self._numpy_dt_to_datetime(ts.times[0])

        # Update D part with date range
        end_date = self._numpy_dt_to_datetime(ts.times[-1])
        pathname = pathname.with_date_range(format_dss_date_range(start_date, end_date))

        try:
            self._dss.write_regular_timeseries(
                pathname=str(pathname),
                values=ts.values,
                start_date=start_date,
                units=units or ts.units,
                data_type=data_type,
            )
            self._pathnames_written.append(str(pathname))
            return True

        except DSSFileError as e:
            self._errors.append(f"Error writing {pathname}: {e}")
            return False

    def write_collection(
        self,
        collection: TimeSeriesCollection,
        pathname_factory: Callable[[str], DSSPathname],
        units: str | None = None,
        data_type: str = "INST-VAL",
    ) -> int:
        """
        Write a TimeSeriesCollection to the DSS file.

        Args:
            collection: TimeSeriesCollection to write
            pathname_factory: Function that takes location and returns pathname
            units: Units string
            data_type: Data type

        Returns:
            Number of time series successfully written
        """
        n_written = 0

        for location in collection.locations:
            ts = collection[location]
            pathname = pathname_factory(location)

            if self.write_timeseries(ts, pathname, units, data_type):
                n_written += 1

        return n_written

    def write_multiple_timeseries(
        self,
        times: Sequence[datetime] | NDArray[np.datetime64],
        values_dict: dict[str, NDArray[np.float64]],
        template: DSSPathnameTemplate,
        units: str = "",
        data_type: str = "INST-VAL",
    ) -> int:
        """
        Write multiple time series with a common time axis.

        Args:
            times: Common time array
            values_dict: Dictionary mapping location to values array
            template: Pathname template
            units: Units string
            data_type: Data type

        Returns:
            Number of time series successfully written
        """
        n_written = 0

        for location, values in values_dict.items():
            pathname = template.make_pathname(location=location)

            # Get start/end dates
            if isinstance(times[0], np.datetime64):
                start_date = self._numpy_dt_to_datetime(times[0])
                end_date = self._numpy_dt_to_datetime(times[-1])
            else:
                start_date = times[0]
                end_date = times[-1]

            pathname = pathname.with_date_range(format_dss_date_range(start_date, end_date))

            try:
                self._dss.write_regular_timeseries(
                    pathname=str(pathname),
                    values=values,
                    start_date=start_date,
                    units=units,
                    data_type=data_type,
                )
                self._pathnames_written.append(str(pathname))
                n_written += 1

            except DSSFileError as e:
                self._errors.append(f"Error writing {pathname}: {e}")

        return n_written

    def _numpy_dt_to_datetime(self, dt: np.datetime64) -> datetime:
        """Convert numpy datetime64 to Python datetime.

        Uses timedelta arithmetic instead of datetime.utcfromtimestamp()
        because the latter fails on Windows for dates before 1970-01-01.
        """
        seconds = (dt - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
        return datetime(1970, 1, 1) + timedelta(seconds=float(seconds))


class DSSTimeSeriesReader:
    """
    High-level reader for time series data from HEC-DSS files.

    Example:
        >>> reader = DSSTimeSeriesReader(Path("input.dss"))
        >>> ts = reader.read_timeseries("/A/B/C/D/E/F/")
        >>> reader.close()
    """

    def __init__(self, filepath: Path | str) -> None:
        """
        Initialize the DSS time series reader.

        Args:
            filepath: Path to DSS file
        """
        check_dss_available()
        self.filepath = Path(filepath)
        self._dss: DSSFile | None = None

    def __enter__(self) -> "DSSTimeSeriesReader":
        """Open the DSS file."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close the DSS file."""
        self.close()

    def open(self) -> None:
        """Open the DSS file for reading."""
        if self._dss is not None:
            return
        self._dss = DSSFile(self.filepath, mode="r")
        self._dss.open()

    def close(self) -> None:
        """Close the DSS file."""
        if self._dss is not None:
            self._dss.close()
            self._dss = None

    def read_timeseries(
        self,
        pathname: DSSPathname | str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        name: str = "",
    ) -> TimeSeries:
        """
        Read a time series from the DSS file.

        Args:
            pathname: DSS pathname
            start_date: Optional start datetime
            end_date: Optional end datetime
            name: Name for the TimeSeries

        Returns:
            TimeSeries object
        """
        if self._dss is None:
            self.open()

        if isinstance(pathname, DSSPathname):
            pathname_str = str(pathname)
            location = pathname.b_part
        else:
            pathname_str = pathname
            location = DSSPathname.from_string(pathname).b_part

        times, values = self._dss.read_regular_timeseries(
            pathname_str, start_date, end_date
        )

        np_times = np.array(times, dtype="datetime64[s]")

        return TimeSeries(
            times=np_times,
            values=values,
            name=name or location,
            location=location,
        )

    def read_collection(
        self,
        pathnames: list[DSSPathname | str],
        variable: str = "",
    ) -> TimeSeriesCollection:
        """
        Read multiple time series into a collection.

        Args:
            pathnames: List of DSS pathnames to read
            variable: Variable name for the collection

        Returns:
            TimeSeriesCollection object
        """
        collection = TimeSeriesCollection(variable=variable)

        for pathname in pathnames:
            try:
                ts = self.read_timeseries(pathname)
                collection.add(ts)
            except DSSFileError:
                pass  # Skip missing records

        return collection


# Convenience functions


def write_timeseries_to_dss(
    filepath: Path | str,
    ts: TimeSeries,
    pathname: DSSPathname | str,
    units: str | None = None,
) -> DSSWriteResult:
    """
    Write a single TimeSeries to a DSS file.

    Args:
        filepath: Path to DSS file
        ts: TimeSeries to write
        pathname: DSS pathname
        units: Optional units string

    Returns:
        DSSWriteResult
    """
    with DSSTimeSeriesWriter(filepath) as writer:
        writer.write_timeseries(ts, pathname, units)
        return writer.close()


def read_timeseries_from_dss(
    filepath: Path | str,
    pathname: DSSPathname | str,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> TimeSeries:
    """
    Read a single TimeSeries from a DSS file.

    Args:
        filepath: Path to DSS file
        pathname: DSS pathname
        start_date: Optional start datetime
        end_date: Optional end datetime

    Returns:
        TimeSeries object
    """
    with DSSTimeSeriesReader(filepath) as reader:
        return reader.read_timeseries(pathname, start_date, end_date)


def write_collection_to_dss(
    filepath: Path | str,
    collection: TimeSeriesCollection,
    template: DSSPathnameTemplate,
    units: str = "",
) -> DSSWriteResult:
    """
    Write a TimeSeriesCollection to a DSS file.

    Args:
        filepath: Path to DSS file
        collection: TimeSeriesCollection to write
        template: Pathname template
        units: Units string

    Returns:
        DSSWriteResult
    """
    with DSSTimeSeriesWriter(filepath) as writer:
        writer.write_collection(
            collection,
            pathname_factory=lambda loc: template.make_pathname(location=loc),
            units=units,
        )
        return writer.close()
