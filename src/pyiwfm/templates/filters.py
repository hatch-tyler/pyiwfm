"""
Custom Jinja2 filters for IWFM file formatting.

This module provides all the formatting filters needed to generate
valid IWFM input files from templates.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# Number Formatting
# =============================================================================


def fortran_float(value: float, width: int = 14, decimals: int = 6) -> str:
    """
    Format a float in Fortran style (right-aligned, fixed width).

    Args:
        value: Float value to format
        width: Total field width
        decimals: Number of decimal places

    Returns:
        Formatted string
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return " " * width
    return f"{float(value):{width}.{decimals}f}"


def fortran_int(value: int, width: int = 10) -> str:
    """
    Format an integer in Fortran style (right-aligned, fixed width).

    Args:
        value: Integer value to format
        width: Total field width

    Returns:
        Formatted string
    """
    if value is None:
        return " " * width
    return f"{int(value):{width}d}"


def fortran_scientific(value: float, width: int = 14, decimals: int = 6) -> str:
    """
    Format a float in Fortran scientific notation.

    Args:
        value: Float value to format
        width: Total field width
        decimals: Number of decimal places

    Returns:
        Formatted string (e.g., "  1.234567E+03")
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return " " * width
    return f"{float(value):{width}.{decimals}E}"


# =============================================================================
# IWFM Comment and Value Formatting
# =============================================================================


def iwfm_comment(text: str, prefix: str = "C") -> str:
    """
    Format text as an IWFM comment line.

    Args:
        text: Comment text
        prefix: Comment prefix character (C, *, or c)

    Returns:
        Formatted comment line
    """
    return f"{prefix}  {text}"


def iwfm_value(
    value: Any,
    description: str = "",
    width: int = 20,
    comment_char: str = "/",
) -> str:
    """
    Format a value with optional description for IWFM input.

    IWFM uses the format: VALUE  / DESCRIPTION

    Args:
        value: Value to format
        description: Optional description
        width: Width for value field
        comment_char: Character separating value from description

    Returns:
        Formatted value line
    """
    if isinstance(value, float):
        value_str = f"{value:.6f}"
    else:
        value_str = str(value)

    if description:
        return f"{value_str:<{width}} {comment_char} {description}"
    return value_str


def iwfm_path(path: str | Path, max_length: int = 1000) -> str:
    """
    Format a file path for IWFM input files.

    IWFM uses forward slashes and has a maximum path length.

    Args:
        path: File path
        max_length: Maximum allowed path length

    Returns:
        Formatted path string
    """
    path_str = str(path).replace("\\", "/")
    if len(path_str) > max_length:
        raise ValueError(f"Path exceeds maximum length of {max_length}: {path_str}")
    return path_str


def iwfm_blank_or_path(path: str | Path | None) -> str:
    """
    Format a path or return empty string for optional files.

    Args:
        path: File path or None

    Returns:
        Formatted path or empty string
    """
    if path is None or str(path).strip() == "":
        return ""
    return iwfm_path(path)


# =============================================================================
# Time and Date Formatting
# =============================================================================


def iwfm_timestamp(dt: datetime | np.datetime64 | str | None) -> str:
    """
    Format a datetime as an IWFM timestamp string.

    IWFM uses the format: MM/DD/YYYY_HH:MM (exactly 16 characters).
    Midnight (00:00) is represented as 24:00 of the previous day.

    Args:
        dt: Datetime object, numpy datetime64, or string

    Returns:
        Formatted timestamp string (16 chars)
    """
    if dt is None:
        return ""

    if isinstance(dt, str):
        return dt

    from pyiwfm.io.timeseries_ascii import format_iwfm_timestamp
    return format_iwfm_timestamp(dt)


def iwfm_date(dt: datetime | np.datetime64 | str | None) -> str:
    """
    Format a date for IWFM (date only, no time).

    Args:
        dt: Datetime object or string

    Returns:
        Formatted date string (MM/DD/YYYY)
    """
    if dt is None:
        return ""

    if isinstance(dt, str):
        return dt

    if isinstance(dt, np.datetime64):
        ts = (dt - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
        dt = datetime.utcfromtimestamp(ts)

    return dt.strftime("%m/%d/%Y")


def iwfm_time_unit(unit: str) -> str:
    """
    Convert a time unit string to IWFM format.

    IWFM time units: 1MIN, 1HOUR, 1DAY, 1WEEK, 1MON, 1YEAR

    Args:
        unit: Time unit (minute, hour, day, week, month, year)

    Returns:
        IWFM time unit code
    """
    unit_map = {
        "minute": "1MIN",
        "min": "1MIN",
        "hour": "1HOUR",
        "hr": "1HOUR",
        "day": "1DAY",
        "week": "1WEEK",
        "wk": "1WEEK",
        "month": "1MON",
        "mon": "1MON",
        "year": "1YEAR",
        "yr": "1YEAR",
    }
    return unit_map.get(unit.lower(), unit.upper())


# =============================================================================
# DSS (HEC-DSS) Formatting
# =============================================================================


def dss_pathname(
    a_part: str = "",
    b_part: str = "",
    c_part: str = "",
    d_part: str = "",
    e_part: str = "",
    f_part: str = "",
) -> str:
    """
    Build an HEC-DSS pathname from its parts.

    DSS pathname format: /A/B/C/D/E/F/
    - A: Project/Basin
    - B: Location
    - C: Parameter (FLOW, HEAD, etc.)
    - D: Date window (start date)
    - E: Time interval (1DAY, 1HOUR, etc.)
    - F: Version

    Args:
        a_part: Project/Basin part
        b_part: Location part
        c_part: Parameter part
        d_part: Date part
        e_part: Interval part
        f_part: Version part

    Returns:
        Formatted DSS pathname
    """
    return f"/{a_part}/{b_part}/{c_part}/{d_part}/{e_part}/{f_part}/"


def dss_date_part(start: datetime, end: datetime) -> str:
    """
    Format the D-part (date range) of a DSS pathname.

    Args:
        start: Start date
        end: End date

    Returns:
        Formatted date range (e.g., "01JAN2020 - 31DEC2020")
    """
    start_str = start.strftime("%d%b%Y").upper()
    end_str = end.strftime("%d%b%Y").upper()
    return f"{start_str} - {end_str}"


def dss_interval(interval: str) -> str:
    """
    Convert interval string to DSS E-part format.

    Args:
        interval: Interval (e.g., "1day", "1hour", "15min")

    Returns:
        DSS interval code (e.g., "1DAY", "1HOUR", "15MIN")
    """
    return interval.upper().replace(" ", "")


# =============================================================================
# Array and Data Block Formatting
# =============================================================================


def iwfm_array_row(
    values: Sequence | NDArray,
    fmt: str = "%14.6f",
    sep: str = " ",
    prefix: str = "",
) -> str:
    """
    Format a row of array values for IWFM output.

    Args:
        values: List or array of values
        fmt: Printf-style format string
        sep: Separator between values
        prefix: Optional prefix (e.g., row ID)

    Returns:
        Formatted row string
    """
    if isinstance(values, np.ndarray):
        values = values.tolist()

    formatted = sep.join(fmt % v if not np.isnan(v) else " " * len(fmt % 0) for v in values)
    if prefix:
        return f"{prefix}{sep}{formatted}"
    return formatted


def iwfm_data_row(
    row_id: int,
    values: Sequence | NDArray,
    int_fmt: str = "%5d",
    float_fmt: str = "%14.6f",
) -> str:
    """
    Format a data row with ID and values.

    Args:
        row_id: Row identifier (e.g., node ID, element ID)
        values: Data values
        int_fmt: Format for integer ID
        float_fmt: Format for float values

    Returns:
        Formatted row string
    """
    id_str = int_fmt % row_id
    if isinstance(values, np.ndarray):
        values = values.tolist()
    values_str = " ".join(float_fmt % v for v in values)
    return f"{id_str} {values_str}"


# =============================================================================
# String Formatting
# =============================================================================


def pad_right(text: str, width: int, char: str = " ") -> str:
    """
    Pad text on the right to a fixed width.

    Args:
        text: Text to pad
        width: Target width
        char: Padding character

    Returns:
        Padded string
    """
    return str(text).ljust(width, char)


def pad_left(text: str, width: int, char: str = " ") -> str:
    """
    Pad text on the left to a fixed width.

    Args:
        text: Text to pad
        width: Target width
        char: Padding character

    Returns:
        Padded string
    """
    return str(text).rjust(width, char)


def truncate(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to maximum length with suffix.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated string
    """
    text = str(text)
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


# =============================================================================
# Time Series Reference Formatting
# =============================================================================


def timeseries_ref(
    filepath: str | Path,
    column: int = 1,
    factor: float = 1.0,
    path_width: int = 60,
) -> str:
    """
    Format a time series file reference for IWFM input files.

    IWFM uses file references in the format:
    FILEPATH  COLUMN  FACTOR

    Args:
        filepath: Path to the time series file
        column: Column number to read (1-based)
        factor: Conversion factor
        path_width: Width for file path field

    Returns:
        Formatted file reference string
    """
    path_str = iwfm_path(filepath)
    return f"{path_str:<{path_width}}  {column:>3}  {factor:>12.6f}"


def dss_timeseries_ref(
    dss_file: str | Path,
    pathname: str,
    factor: float = 1.0,
    path_width: int = 60,
) -> str:
    """
    Format a DSS time series reference for IWFM input files.

    Args:
        dss_file: Path to the DSS file
        pathname: DSS pathname
        factor: Conversion factor
        path_width: Width for file path field

    Returns:
        Formatted DSS reference string
    """
    dss_str = iwfm_path(dss_file)
    return f"{dss_str:<{path_width}}  {pathname}  {factor:>12.6f}"


# =============================================================================
# Filter Registration
# =============================================================================


def register_all_filters(env) -> None:
    """
    Register all IWFM filters with a Jinja2 environment.

    Args:
        env: Jinja2 Environment instance
    """
    # Number formatting
    env.filters["fortran_float"] = fortran_float
    env.filters["fortran_int"] = fortran_int
    env.filters["fortran_scientific"] = fortran_scientific

    # IWFM formatting
    env.filters["iwfm_comment"] = iwfm_comment
    env.filters["iwfm_value"] = iwfm_value
    env.filters["iwfm_path"] = iwfm_path
    env.filters["iwfm_blank_or_path"] = iwfm_blank_or_path

    # Time formatting
    env.filters["iwfm_timestamp"] = iwfm_timestamp
    env.filters["iwfm_date"] = iwfm_date
    env.filters["iwfm_time_unit"] = iwfm_time_unit

    # DSS formatting
    env.filters["dss_pathname"] = dss_pathname
    env.filters["dss_date_part"] = dss_date_part
    env.filters["dss_interval"] = dss_interval

    # Array formatting
    env.filters["iwfm_array_row"] = iwfm_array_row
    env.filters["iwfm_data_row"] = iwfm_data_row

    # String formatting
    env.filters["pad_right"] = pad_right
    env.filters["pad_left"] = pad_left
    env.filters["truncate"] = truncate

    # Time series references
    env.filters["timeseries_ref"] = timeseries_ref
    env.filters["dss_timeseries_ref"] = dss_timeseries_ref
