"""
ASCII time series I/O handlers for IWFM model files.

This module provides functions for reading and writing IWFM ASCII time series
files using the standard 16-character timestamp format (MM/DD/YYYY_HH:MM).
IWFM uses 24:00 to represent midnight (end of day).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence, TextIO

import numpy as np
from numpy.typing import NDArray

from pyiwfm.core.timeseries import TimeSeries, TimeSeriesCollection
from pyiwfm.core.exceptions import FileFormatError


# IWFM timestamp format: MM/DD/YYYY_HH:MM (16 characters)
# IWFM uses 24:00 for midnight (end of day) instead of 00:00 (start of next day).
IWFM_TIMESTAMP_FORMAT = "%m/%d/%Y_%H:%M"
IWFM_TIMESTAMP_LENGTH = 16

# IWFM comment characters — must appear in column 1 (first character of line)
COMMENT_CHARS = ("C", "c", "*")

# Reference epoch for numpy datetime64 → Python datetime conversion.
# Using timedelta arithmetic instead of datetime.utcfromtimestamp() because
# the latter fails on Windows for dates before 1970-01-01 (negative timestamps).
_EPOCH = datetime(1970, 1, 1)


def _np_dt64_to_datetime(dt64: np.datetime64) -> datetime:
    """Convert a numpy datetime64 to a Python datetime (works for any date)."""
    seconds = (dt64 - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
    return _EPOCH + timedelta(seconds=float(seconds))


def format_iwfm_timestamp(dt: datetime | np.datetime64) -> str:
    """
    Format a datetime as an IWFM timestamp string.

    IWFM uses the format MM/DD/YYYY_HH:MM (16 characters).
    Midnight (00:00) is represented as 24:00 of the *previous* day.
    For example, 2050-11-01 00:00 becomes ``10/31/2050_24:00``.

    Args:
        dt: datetime object or numpy datetime64

    Returns:
        Formatted timestamp string (16 chars)
    """
    if isinstance(dt, np.datetime64):
        dt = _np_dt64_to_datetime(dt)

    # IWFM convention: midnight is written as 24:00 of the previous day
    if dt.hour == 0 and dt.minute == 0 and dt.second == 0:
        prev_day = dt - timedelta(days=1)
        return prev_day.strftime("%m/%d/%Y") + "_24:00"

    return dt.strftime(IWFM_TIMESTAMP_FORMAT).ljust(IWFM_TIMESTAMP_LENGTH)


def parse_iwfm_timestamp(ts_str: str) -> datetime:
    """
    Parse an IWFM timestamp string to datetime.

    IWFM timestamps are exactly 16 characters: MM/DD/YYYY_HH:MM
    The 24:00 convention means midnight at the end of the given day,
    which is converted to 00:00 of the next day internally.

    Args:
        ts_str: Timestamp string (16 chars, MM/DD/YYYY_HH:MM)

    Returns:
        datetime object

    Raises:
        ValueError: If timestamp format is invalid
    """
    ts_str = ts_str.strip()

    # Determine separator (underscore or space)
    sep = "_" if "_" in ts_str else " "
    parts = ts_str.split(sep, 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid IWFM timestamp: '{ts_str}'")

    date_str, time_str = parts

    # Handle 24:00 convention (midnight = start of next day)
    if time_str.startswith("24:00"):
        dt = datetime.strptime(date_str, "%m/%d/%Y")
        dt += timedelta(days=1)
        return dt

    dt = datetime.strptime(f"{date_str}_{time_str}", "%m/%d/%Y_%H:%M")
    return dt


def _is_comment_line(line: str) -> bool:
    """Check if a line is a comment line.

    In IWFM Fortran format, a comment line has the comment character
    in column 1 (the very first character of the line), not after
    leading whitespace.
    """
    if not line or not line.strip():
        return True
    if line[0] in COMMENT_CHARS:
        return True
    return False


def _strip_inline_comment(line: str) -> str:
    """Strip inline comment from a data line.

    IWFM uses '/' as the inline comment delimiter.

    Returns the value portion of the line with whitespace stripped.
    """
    if "/" in line:
        return line.split("/")[0].strip()
    return line.strip()


@dataclass
class TimeSeriesFileConfig:
    """
    Configuration for an IWFM time series file.

    Attributes:
        n_columns: Number of value columns
        column_ids: List of location/column identifiers
        units: Units string for values
        factor: Conversion factor to apply to values
        header_lines: Header comment lines
    """

    n_columns: int
    column_ids: list[str | int]
    units: str = ""
    factor: float = 1.0
    header_lines: list[str] | None = None


class TimeSeriesWriter:
    """
    Writer for IWFM ASCII time series files.

    IWFM time series files have the format:
        C  Header comments
        NDATA                         / Number of data columns
        FACTOR                        / Conversion factor
        MM/DD/YYYY_HH:MM:SS   val1   val2   val3   ...

    Example:
        >>> writer = TimeSeriesWriter()
        >>> writer.write(
        ...     filepath="pumping.dat",
        ...     times=times,
        ...     values=values,
        ...     column_ids=[1, 2, 3],
        ...     units="TAF"
        ... )
    """

    def __init__(
        self,
        value_format: str = "%14.6f",
        timestamp_format: str = IWFM_TIMESTAMP_FORMAT,
    ) -> None:
        """
        Initialize the time series writer.

        Args:
            value_format: Printf-style format for values
            timestamp_format: strftime format for timestamps
        """
        self.value_format = value_format
        self.timestamp_format = timestamp_format

    def write(
        self,
        filepath: Path | str,
        times: Sequence[datetime] | NDArray[np.datetime64],
        values: NDArray[np.float64],
        column_ids: list[str | int] | None = None,
        units: str = "",
        factor: float = 1.0,
        header: str | None = None,
    ) -> None:
        """
        Write time series data to an IWFM ASCII file.

        Args:
            filepath: Output file path
            times: Sequence of datetime values
            values: 2D array of values (n_times, n_columns) or 1D for single column
            column_ids: Optional list of column identifiers
            units: Units string for header
            factor: Conversion factor (written to file)
            header: Optional header comment
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Ensure values is 2D
        if values.ndim == 1:
            values = values.reshape(-1, 1)

        n_times, n_columns = values.shape

        if len(times) != n_times:
            raise ValueError(
                f"Times length ({len(times)}) doesn't match values rows ({n_times})"
            )

        if column_ids is None:
            column_ids = list(range(1, n_columns + 1))

        with open(filepath, "w") as f:
            self._write_header(f, header, n_columns, column_ids, units, factor)
            self._write_data(f, times, values)

    def _write_header(
        self,
        f: TextIO,
        header: str | None,
        n_columns: int,
        column_ids: list[str | int],
        units: str,
        factor: float,
    ) -> None:
        """Write the file header."""
        # Write header comments
        if header:
            for line in header.strip().split("\n"):
                f.write(f"C  {line}\n")
        else:
            f.write("C  Time series data file generated by pyiwfm\n")
            f.write("C\n")

        # Write column ID comment
        col_str = "  ".join(str(cid) for cid in column_ids)
        f.write(f"C  Columns: {col_str}\n")

        if units:
            f.write(f"C  Units: {units}\n")

        f.write("C\n")

        # Write NDATA and FACTOR
        f.write(f"{n_columns:<10}                              / NDATA\n")
        f.write(f"{factor:<14.6f}                          / FACTOR\n")

    def _write_data(
        self,
        f: TextIO,
        times: Sequence[datetime] | NDArray[np.datetime64],
        values: NDArray[np.float64],
    ) -> None:
        """Write the time series data."""
        for i, t in enumerate(times):
            # Format timestamp
            ts_str = format_iwfm_timestamp(t)

            # Format values
            val_strs = [self.value_format % v for v in values[i, :]]
            val_line = "  ".join(val_strs)

            f.write(f"{ts_str}  {val_line}\n")

    def write_from_timeseries(
        self,
        filepath: Path | str,
        ts: TimeSeries,
        header: str | None = None,
        factor: float = 1.0,
    ) -> None:
        """
        Write a TimeSeries object to file.

        Args:
            filepath: Output file path
            ts: TimeSeries object
            header: Optional header comment
            factor: Conversion factor
        """
        # Convert times to datetime
        if ts.times.dtype == np.dtype("datetime64[s]"):
            times = [_np_dt64_to_datetime(t) for t in ts.times]
        else:
            times = list(ts.times)

        self.write(
            filepath=filepath,
            times=times,
            values=ts.values,
            column_ids=[ts.location] if ts.location else None,
            units=ts.units,
            factor=factor,
            header=header,
        )

    def write_from_collection(
        self,
        filepath: Path | str,
        collection: TimeSeriesCollection,
        header: str | None = None,
        factor: float = 1.0,
    ) -> None:
        """
        Write a TimeSeriesCollection to a single file.

        All time series must have the same timestamps.

        Args:
            filepath: Output file path
            collection: TimeSeriesCollection object
            header: Optional header comment
            factor: Conversion factor
        """
        if len(collection) == 0:
            raise ValueError("Empty collection")

        # Get reference times from first series
        first_ts = next(iter(collection))
        times = first_ts.times

        # Build values array
        column_ids = collection.locations
        n_times = len(times)
        n_cols = len(column_ids)
        values = np.zeros((n_times, n_cols))

        for i, loc in enumerate(column_ids):
            ts = collection[loc]
            if len(ts.times) != n_times:
                raise ValueError(
                    f"Time series '{loc}' has different length than reference"
                )
            if ts.values.ndim == 1:
                values[:, i] = ts.values
            else:
                values[:, i] = ts.values[:, 0]

        # Convert times to datetime
        if times.dtype == np.dtype("datetime64[s]"):
            dt_times = [_np_dt64_to_datetime(t) for t in times]
        else:
            dt_times = list(times)

        self.write(
            filepath=filepath,
            times=dt_times,
            values=values,
            column_ids=column_ids,
            units=first_ts.units,
            factor=factor,
            header=header,
        )


class TimeSeriesReader:
    """
    Reader for IWFM ASCII time series files.

    Example:
        >>> reader = TimeSeriesReader()
        >>> times, values, config = reader.read("pumping.dat")
    """

    def read(
        self, filepath: Path | str
    ) -> tuple[list[datetime], NDArray[np.float64], TimeSeriesFileConfig]:
        """
        Read time series data from an IWFM ASCII file.

        Args:
            filepath: Input file path

        Returns:
            Tuple of (times, values, config)
        """
        filepath = Path(filepath)

        times: list[datetime] = []
        values_list: list[list[float]] = []
        header_lines: list[str] = []
        n_columns = 0
        factor = 1.0

        with open(filepath, "r") as f:
            line_num = 0

            # Read NDATA
            for line in f:
                line_num += 1
                if _is_comment_line(line):
                    header_lines.append(line.strip())
                    continue

                value_str = _strip_inline_comment(line)
                try:
                    n_columns = int(value_str)
                except ValueError as e:
                    raise FileFormatError(
                        f"Invalid NDATA value: '{value_str}'",
                        line_number=line_num,
                    ) from e
                break

            # Read FACTOR
            for line in f:
                line_num += 1
                if _is_comment_line(line):
                    continue

                value_str = _strip_inline_comment(line)
                try:
                    factor = float(value_str)
                except ValueError as e:
                    raise FileFormatError(
                        f"Invalid FACTOR value: '{value_str}'",
                        line_number=line_num,
                    ) from e
                break

            # Read data lines.  Some IWFM files (e.g. precipitation,
            # evapotranspiration) have extra header fields after FACTOR
            # (NSPRN, NFQRN, DSSFL, etc.).  We skip non-timestamp lines
            # until we encounter the first valid timestamp data line.
            data_started = False
            for line in f:
                line_num += 1
                if _is_comment_line(line):
                    continue

                stripped = line.strip()
                if not stripped:
                    continue

                # Try to parse as a timestamp data line
                try:
                    ts_str = stripped[:IWFM_TIMESTAMP_LENGTH].strip()
                    dt = parse_iwfm_timestamp(ts_str)
                    data_started = True
                    times.append(dt)

                    # Parse values
                    rest = stripped[IWFM_TIMESTAMP_LENGTH:].strip()
                    vals = [float(v) for v in rest.split()]
                    values_list.append(vals)

                except ValueError:
                    if data_started:
                        raise FileFormatError(
                            f"Invalid data line: '{stripped}'",
                            line_number=line_num,
                        )
                    # Before data started: skip extra header lines
                    continue

        # Convert to numpy array
        values = np.array(values_list, dtype=np.float64)

        # Apply factor
        values *= factor

        config = TimeSeriesFileConfig(
            n_columns=n_columns,
            column_ids=list(range(1, n_columns + 1)),
            factor=factor,
            header_lines=header_lines,
        )

        return times, values, config

    def read_to_timeseries(
        self,
        filepath: Path | str,
        name: str = "",
        location: str = "",
    ) -> TimeSeries:
        """
        Read file and return as TimeSeries object.

        Args:
            filepath: Input file path
            name: Name for the time series
            location: Location identifier

        Returns:
            TimeSeries object
        """
        times, values, config = self.read(filepath)

        np_times = np.array(times, dtype="datetime64[s]")

        return TimeSeries(
            times=np_times,
            values=values,
            name=name,
            location=location,
        )

    def read_to_collection(
        self,
        filepath: Path | str,
        column_ids: list[str] | None = None,
        variable: str = "",
    ) -> TimeSeriesCollection:
        """
        Read file and return as TimeSeriesCollection.

        Args:
            filepath: Input file path
            column_ids: Optional column identifiers
            variable: Variable name for the collection

        Returns:
            TimeSeriesCollection object
        """
        times, values, config = self.read(filepath)
        np_times = np.array(times, dtype="datetime64[s]")

        if column_ids is None:
            column_ids = [str(i) for i in config.column_ids]

        collection = TimeSeriesCollection(variable=variable)

        for i, col_id in enumerate(column_ids):
            if i < values.shape[1]:
                ts = TimeSeries(
                    times=np_times,
                    values=values[:, i],
                    location=col_id,
                )
                collection.add(ts)

        return collection


# Convenience functions


def write_timeseries(
    filepath: Path | str,
    times: Sequence[datetime] | NDArray[np.datetime64],
    values: NDArray[np.float64],
    column_ids: list[str | int] | None = None,
    units: str = "",
    factor: float = 1.0,
    header: str | None = None,
) -> None:
    """
    Write time series data to an IWFM ASCII file.

    Args:
        filepath: Output file path
        times: Sequence of datetime values
        values: 2D array of values (n_times, n_columns)
        column_ids: Optional list of column identifiers
        units: Units string for header
        factor: Conversion factor
        header: Optional header comment
    """
    writer = TimeSeriesWriter()
    writer.write(
        filepath=filepath,
        times=times,
        values=values,
        column_ids=column_ids,
        units=units,
        factor=factor,
        header=header,
    )


def read_timeseries(
    filepath: Path | str,
) -> tuple[list[datetime], NDArray[np.float64], TimeSeriesFileConfig]:
    """
    Read time series data from an IWFM ASCII file.

    Args:
        filepath: Input file path

    Returns:
        Tuple of (times, values, config)
    """
    reader = TimeSeriesReader()
    return reader.read(filepath)
