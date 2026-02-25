"""
SMP (Sample/Bore) file reader and writer for IWFM observation data.

The SMP format is IWFM's standard observation file format used by
IWFM2OBS and other calibration utilities. It uses fixed-width columns:

- Columns 1-25:  Bore ID (left-justified)
- Columns 26-37: Date (MM/DD/YYYY)
- Columns 38-49: Time (HH:MM:SS)
- Columns 50-60: Value (numeric)
- Column 61+:    Optional 'X' for excluded records

Sentinel values (-1.1E38, -9.1E37, etc.) are treated as NaN.

Example
-------
>>> from pyiwfm.io.smp import SMPReader, SMPWriter
>>> reader = SMPReader("observations.smp")
>>> data = reader.read()
>>> for bore_id, ts in data.items():
...     print(f"{bore_id}: {len(ts.values)} records")
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pyiwfm.io.base import BaseReader, BaseWriter

# Sentinel values used by IWFM to indicate missing data
_SENTINEL_THRESHOLD = -1.0e37


@dataclass
class SMPRecord:
    """A single SMP observation record.

    Attributes
    ----------
    bore_id : str
        Observation well / bore identifier.
    datetime : datetime
        Date and time of observation.
    value : float
        Observed value (e.g., water level elevation).
    excluded : bool
        If True, record is flagged as excluded (``X`` marker).
    """

    bore_id: str
    datetime: datetime
    value: float
    excluded: bool = False


@dataclass
class SMPTimeSeries:
    """Time series for a single bore from an SMP file.

    Attributes
    ----------
    bore_id : str
        Observation well / bore identifier.
    times : NDArray[np.datetime64]
        Array of observation timestamps.
    values : NDArray[np.float64]
        Array of observed values.
    excluded : NDArray[np.bool_]
        Boolean mask; True where records are excluded.
    """

    bore_id: str
    times: NDArray[np.datetime64]
    values: NDArray[np.float64]
    excluded: NDArray[np.bool_]

    @property
    def n_records(self) -> int:
        """Number of records in the time series."""
        return len(self.times)

    @property
    def valid_mask(self) -> NDArray[np.bool_]:
        """Mask of non-NaN, non-excluded records."""
        return ~self.excluded & ~np.isnan(self.values)


def _is_sentinel(value: float) -> bool:
    """Check if a value is an IWFM sentinel (missing data indicator)."""
    if math.isnan(value) or math.isinf(value):
        return True
    return value < _SENTINEL_THRESHOLD


def _parse_smp_line(line: str) -> SMPRecord | None:
    """Parse a single SMP line into an SMPRecord.

    Tries fixed-width parsing first, then falls back to whitespace split.
    Returns None for blank or comment lines.
    """
    stripped = line.rstrip("\n\r")
    if not stripped or not stripped.strip():
        return None

    # Try fixed-width parsing: bore_id(1:25), date(26:37), time(38:49), value(50:60)
    if len(stripped) >= 50:
        bore_id = stripped[:25].strip()
        date_str = stripped[25:37].strip()
        time_str = stripped[37:49].strip()
        value_str = stripped[49:60].strip() if len(stripped) >= 60 else stripped[49:].strip()
        excluded_str = stripped[60:].strip() if len(stripped) > 60 else ""

        if bore_id and date_str:
            try:
                return _build_record(bore_id, date_str, time_str, value_str, excluded_str)
            except (ValueError, IndexError):
                pass

    # Fallback: whitespace split
    parts = stripped.split()
    if len(parts) < 4:
        return None

    bore_id = parts[0]
    date_str = parts[1]
    time_str = parts[2]
    value_str = parts[3]
    excluded_str = parts[4] if len(parts) > 4 else ""

    try:
        return _build_record(bore_id, date_str, time_str, value_str, excluded_str)
    except (ValueError, IndexError):
        return None


def _build_record(
    bore_id: str,
    date_str: str,
    time_str: str,
    value_str: str,
    excluded_str: str,
) -> SMPRecord:
    """Build an SMPRecord from parsed string components."""
    # Parse date: MM/DD/YYYY
    dt_str = f"{date_str} {time_str}" if time_str else date_str
    dt: datetime
    if time_str and ":" in time_str:
        try:
            dt = datetime.strptime(dt_str, "%m/%d/%Y %H:%M:%S")
        except ValueError:
            dt = datetime.strptime(dt_str, "%m/%d/%Y %H:%M")
    else:
        dt = datetime.strptime(date_str, "%m/%d/%Y")

    # Parse value
    value = float(value_str)
    if _is_sentinel(value):
        value = float("nan")

    # Excluded flag
    excluded = excluded_str.upper().startswith("X") if excluded_str else False

    return SMPRecord(bore_id=bore_id, datetime=dt, value=value, excluded=excluded)


def _records_to_timeseries(records: list[SMPRecord]) -> SMPTimeSeries:
    """Convert a list of SMPRecords for one bore into an SMPTimeSeries."""
    bore_id = records[0].bore_id
    times = np.array([np.datetime64(r.datetime) for r in records], dtype="datetime64[s]")
    values = np.array([r.value for r in records], dtype=np.float64)
    excluded = np.array([r.excluded for r in records], dtype=np.bool_)
    return SMPTimeSeries(bore_id=bore_id, times=times, values=values, excluded=excluded)


class SMPReader(BaseReader):
    """Reader for IWFM SMP (Sample/Bore) observation files.

    Parameters
    ----------
    filepath : Path | str
        Path to the SMP file.

    Example
    -------
    >>> reader = SMPReader("obs_wells.smp")
    >>> data = reader.read()
    >>> ts = data["WELL_01"]
    >>> print(ts.n_records)
    """

    @property
    def format(self) -> str:
        return "smp"

    def read(self) -> dict[str, SMPTimeSeries]:
        """Read all bore time series from the SMP file.

        Returns
        -------
        dict[str, SMPTimeSeries]
            Mapping of bore ID to time series.
        """
        records_by_bore: dict[str, list[SMPRecord]] = {}

        with open(self.filepath, encoding="utf-8", errors="replace") as f:
            for line in f:
                rec = _parse_smp_line(line)
                if rec is not None:
                    records_by_bore.setdefault(rec.bore_id, []).append(rec)

        result: dict[str, SMPTimeSeries] = {}
        for bore_id, records in records_by_bore.items():
            result[bore_id] = _records_to_timeseries(records)

        return result

    @property
    def bore_ids(self) -> list[str]:
        """Return list of bore IDs without reading all data."""
        ids: list[str] = []
        seen: set[str] = set()

        with open(self.filepath, encoding="utf-8", errors="replace") as f:
            for line in f:
                rec = _parse_smp_line(line)
                if rec is not None and rec.bore_id not in seen:
                    ids.append(rec.bore_id)
                    seen.add(rec.bore_id)

        return ids

    def read_bore(self, bore_id: str) -> SMPTimeSeries | None:
        """Read time series for a single bore.

        Parameters
        ----------
        bore_id : str
            The bore identifier to read.

        Returns
        -------
        SMPTimeSeries | None
            Time series for the bore, or None if not found.
        """
        records: list[SMPRecord] = []

        with open(self.filepath, encoding="utf-8", errors="replace") as f:
            for line in f:
                rec = _parse_smp_line(line)
                if rec is not None and rec.bore_id == bore_id:
                    records.append(rec)

        if not records:
            return None
        return _records_to_timeseries(records)


def _format_smp_line(bore_id: str, dt: datetime, value: float, excluded: bool) -> str:
    """Format a single SMP line in fixed-width format."""
    bore_field = f"{bore_id:<25s}"
    date_field = dt.strftime("%m/%d/%Y").rjust(12)
    time_field = dt.strftime("%H:%M:%S").rjust(12)

    if np.isnan(value):
        val_field = f"{-1.1e38:11.4E}"
    else:
        val_field = f"{value:11.4f}"

    line = f"{bore_field}{date_field}{time_field}{val_field}"
    if excluded:
        line += "  X"
    return line


class SMPWriter(BaseWriter):
    """Writer for IWFM SMP (Sample/Bore) observation files.

    Parameters
    ----------
    filepath : Path | str
        Path to the output SMP file.

    Example
    -------
    >>> writer = SMPWriter("output.smp")
    >>> writer.write(data)
    """

    @property
    def format(self) -> str:
        return "smp"

    def write(self, data: dict[str, SMPTimeSeries]) -> None:
        """Write all bore time series to the SMP file.

        Parameters
        ----------
        data : dict[str, SMPTimeSeries]
            Mapping of bore ID to time series.
        """
        self._ensure_parent_exists()

        with open(self.filepath, "w", encoding="utf-8") as f:
            for _bore_id, ts in data.items():
                self._write_timeseries(f, ts)

    def write_bore(self, ts: SMPTimeSeries) -> None:
        """Append a single bore's time series to the file.

        Parameters
        ----------
        ts : SMPTimeSeries
            Time series to write.
        """
        self._ensure_parent_exists()

        with open(self.filepath, "a", encoding="utf-8") as f:
            self._write_timeseries(f, ts)

    @staticmethod
    def _write_timeseries(f: Any, ts: SMPTimeSeries) -> None:
        """Write a single bore's records to an open file."""
        for i in range(ts.n_records):
            dt = ts.times[i].astype("datetime64[s]").astype(datetime)
            value = float(ts.values[i])
            excluded = bool(ts.excluded[i])
            line = _format_smp_line(ts.bore_id, dt, value, excluded)
            f.write(line + "\n")
