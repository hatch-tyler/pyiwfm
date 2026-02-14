"""
IWFM land-use area time-series file reader.

These files contain timestamped per-element rows with area values for
each crop/category.  The format is::

    NFACTARL  NSPRN  NFLL  NWINT  NSPCL  / column pointers (ignored)
    FACTARL                                / unit conversion factor
    DSSFL                                  / DSS file (blank = none)
    MM/DD/YYYY_HH:MM  IE  A1  A2  ...  AN
    MM/DD/YYYY_HH:MM  IE  A1  A2  ...  AN
    ...

Each timestep block has one row per element, with the date repeated on
every row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# IWFM comment characters (column 1)
_COMMENT_CHARS = ("C", "c", "*")


@dataclass
class AreaFileMetadata:
    """Metadata from an IWFM area time-series file header."""

    n_columns: int = 0
    factor: float = 1.0
    dss_file: str = ""
    dates: list[str] = field(default_factory=list)


def _strip_description(line: str) -> str:
    """Strip inline ``/`` or ``#`` description from an IWFM data line."""
    import re

    m = re.search(r"\s+[/#]", line)
    if m:
        return line[: m.start()].strip()
    return line.strip()


def _read_lines(filepath: Path) -> list[str]:
    with open(filepath, "r") as fh:
        return fh.readlines()


def _iter_data_lines(lines: list[str]) -> list[str]:
    """Return non-comment, non-blank data lines with descriptions stripped."""
    result: list[str] = []
    for line in lines:
        if not line or not line.strip():
            continue
        if line[0] in _COMMENT_CHARS:
            continue
        val = _strip_description(line)
        # Keep even empty values (blank DSS file lines)
        result.append(val)
    return result


def read_area_metadata(filepath: Path | str) -> AreaFileMetadata:
    """Read header metadata from an IWFM area time-series file.

    Returns column count, unit factor, and DSS file path without
    reading any timestep data.
    """
    filepath = Path(filepath)
    data_lines = _iter_data_lines(_read_lines(filepath))
    meta = AreaFileMetadata()

    if len(data_lines) < 3:
        return meta

    # First data line: column pointer integers (count = n_columns)
    meta.n_columns = len(data_lines[0].split())

    # Second data line: unit conversion factor
    try:
        meta.factor = float(data_lines[1])
    except ValueError:
        meta.factor = 1.0

    # Third data line: DSS file path (blank = none)
    meta.dss_file = data_lines[2] if data_lines[2] else ""

    return meta


def read_area_timestep(
    filepath: Path | str,
    timestep_index: int = 0,
) -> dict[int, list[float]]:
    """Read area data for a single timestep from an IWFM area file.

    Args:
        filepath: Path to the IWFM area time-series file.
        timestep_index: Zero-based index of the timestep to read.

    Returns:
        Dictionary mapping element ID to list of area values per crop.
    """
    filepath = Path(filepath)
    data_lines = _iter_data_lines(_read_lines(filepath))

    if len(data_lines) < 3:
        return {}

    # Header: column pointers, factor, DSS file
    n_cols = len(data_lines[0].split())

    try:
        factor = float(data_lines[1])
    except ValueError:
        factor = 1.0

    # Data rows start at index 3
    current_ts = -1
    current_date: str | None = None
    result: dict[int, list[float]] = {}

    for line in data_lines[3:]:
        parts = line.split()
        if len(parts) < 2:
            continue

        # First token should be a date (MM/DD/YYYY_HH:MM)
        date_str = parts[0]
        if "/" not in date_str:
            continue

        elem_id = int(parts[1])
        areas = [float(v) * factor for v in parts[2 : 2 + n_cols]]

        if current_date is None or date_str != current_date:
            current_ts += 1
            current_date = date_str
            if current_ts > timestep_index:
                break
            if current_ts == timestep_index:
                result = {}

        if current_ts == timestep_index:
            result[elem_id] = areas

    return result


def read_all_timesteps(
    filepath: Path | str,
) -> list[tuple[str, dict[int, list[float]]]]:
    """Read all timesteps from an IWFM area time-series file.

    Returns:
        List of (date_string, {element_id: [areas]}) tuples.
    """
    filepath = Path(filepath)
    data_lines = _iter_data_lines(_read_lines(filepath))

    if len(data_lines) < 3:
        return []

    n_cols = len(data_lines[0].split())

    try:
        factor = float(data_lines[1])
    except ValueError:
        factor = 1.0

    timesteps: list[tuple[str, dict[int, list[float]]]] = []
    current_date: str | None = None
    current_block: dict[int, list[float]] = {}

    for line in data_lines[3:]:
        parts = line.split()
        if len(parts) < 2:
            continue

        date_str = parts[0]
        if "/" not in date_str:
            continue

        elem_id = int(parts[1])
        areas = [float(v) * factor for v in parts[2 : 2 + n_cols]]

        if current_date is None:
            current_date = date_str
            current_block = {}
        elif date_str != current_date:
            timesteps.append((current_date, current_block))
            current_date = date_str
            current_block = {}

        current_block[elem_id] = areas

    if current_date is not None and current_block:
        timesteps.append((current_date, current_block))

    return timesteps
