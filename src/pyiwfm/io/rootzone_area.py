"""
IWFM land-use area time-series file reader.

These files contain timestamped per-element rows with area values for
each crop/category.  The format is::

    NFACTARL  NSPRN  NFLL  NWINT  NSPCL  / column pointers (ignored)
    FACTARL                                / unit conversion factor
    DSSFL                                  / DSS file (blank = none)
    MM/DD/YYYY_HH:MM  IE  A1  A2  ...  AN
                       IE  A1  A2  ...  AN
    ...

Each timestep block has one row per element.  The date appears only on
the **first** row of each block; subsequent rows in the same block
contain just the element ID and values (continuation rows).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# IWFM comment characters (column 1)
_COMMENT_CHARS = ("C", "c", "*")


def _is_comment(line: str) -> bool:
    """Check if a line is an IWFM comment.

    IWFM comments start with ``C``, ``c``, or ``*`` in column 1.
    However, ``C:`` is a Windows drive letter (e.g. ``C:\\model\\area.dss``),
    not a comment.
    """
    stripped = line.lstrip()
    if not stripped:
        return True
    ch = stripped[0]
    if ch not in _COMMENT_CHARS:
        return False
    # "C:" or "c:" is a Windows drive letter, not a comment
    if len(stripped) > 1 and stripped[1] == ":":
        return False
    return True


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


def _has_date(token: str) -> bool:
    """Return True if *token* looks like an IWFM date (contains '/')."""
    return "/" in token


def _read_lines(filepath: Path) -> list[str]:
    with open(filepath) as fh:
        return fh.readlines()


def _iter_data_lines(lines: list[str]) -> list[str]:
    """Return non-comment, non-blank data lines with descriptions stripped."""
    result: list[str] = []
    for line in lines:
        if not line or not line.strip():
            continue
        if _is_comment(line):
            continue
        val = _strip_description(line)
        # Keep even empty values (blank DSS file lines)
        result.append(val)
    return result


def _parse_data_row(parts: list[str]) -> tuple[str | None, int, list[str]]:
    """Parse a data row, handling both date-bearing and continuation rows.

    Returns ``(date_or_none, element_id, value_tokens)``.

    - Date-bearing row: ``[date, elem_id, v1, v2, ...]``
    - Continuation row:  ``[elem_id, v1, v2, ...]``
    """
    if _has_date(parts[0]):
        return parts[0], int(parts[1]), parts[2:]
    return None, int(parts[0]), parts[1:]


def _detect_n_cols(data_lines: list[str], header_n_cols: int) -> int:
    """Detect the actual number of area columns from data rows.

    Some IWFM versions put a single ``NDATA`` integer on the first header
    line instead of per-column pointers.  In that case ``header_n_cols``
    would be 1, which is wrong.  We validate against the first actual
    data row and use whichever is larger.
    """
    for line in data_lines[3:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        date, _eid, vals = _parse_data_row(parts)
        row_n_cols = len(vals)
        if row_n_cols == 0:
            continue
        if row_n_cols != header_n_cols:
            logger.debug(
                "Header n_cols=%d but data row has %d area values; using %d",
                header_n_cols,
                row_n_cols,
                row_n_cols,
            )
            return row_n_cols
        return header_n_cols
    return header_n_cols


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
    header_n_cols = len(data_lines[0].split())

    # Second data line: unit conversion factor
    try:
        meta.factor = float(data_lines[1])
    except ValueError:
        meta.factor = 1.0

    # Third data line: DSS file path (blank = none)
    meta.dss_file = data_lines[2] if data_lines[2] else ""

    # Validate n_cols against actual data rows
    meta.n_columns = _detect_n_cols(data_lines, header_n_cols)

    return meta


def read_area_timestep(
    filepath: Path | str,
    timestep_index: int = 0,
) -> dict[int, list[float]]:
    """Read area data for a single timestep from an IWFM area file.

    Handles both formats: date on every row, or date only on the
    first row of each timestep block (continuation rows).

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
    header_n_cols = len(data_lines[0].split())

    try:
        factor = float(data_lines[1])
    except ValueError:
        factor = 1.0

    n_cols = _detect_n_cols(data_lines, header_n_cols)

    # Data rows start at index 3
    current_ts = -1
    current_date: str | None = None
    result: dict[int, list[float]] = {}

    for line in data_lines[3:]:
        parts = line.split()
        if len(parts) < 2:
            continue

        date, elem_id, val_tokens = _parse_data_row(parts)

        # A new date means a new timestep
        if date is not None and (current_date is None or date != current_date):
            current_ts += 1
            current_date = date
            if current_ts > timestep_index:
                break
            if current_ts == timestep_index:
                result = {}

        # Skip rows before we've seen our first date
        if current_date is None:
            continue

        if current_ts == timestep_index:
            areas = [float(v) * factor for v in val_tokens[:n_cols]]
            result[elem_id] = areas

    return result


def read_all_timesteps(
    filepath: Path | str,
) -> list[tuple[str, dict[int, list[float]]]]:
    """Read all timesteps from an IWFM area time-series file.

    Handles both formats: date on every row, or date only on the
    first row of each timestep block (continuation rows).

    Returns:
        List of (date_string, {element_id: [areas]}) tuples.
    """
    filepath = Path(filepath)
    data_lines = _iter_data_lines(_read_lines(filepath))

    if len(data_lines) < 3:
        return []

    header_n_cols = len(data_lines[0].split())
    n_cols = _detect_n_cols(data_lines, header_n_cols)

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

        date, elem_id, val_tokens = _parse_data_row(parts)

        if date is not None and (current_date is None or date != current_date):
            # Flush previous block
            if current_date is not None and current_block:
                timesteps.append((current_date, current_block))
            current_date = date
            current_block = {}

        # Skip rows before we've seen our first date
        if current_date is None:
            continue

        areas = [float(v) * factor for v in val_tokens[:n_cols]]
        current_block[elem_id] = areas

    if current_date is not None and current_block:
        timesteps.append((current_date, current_block))

    return timesteps
