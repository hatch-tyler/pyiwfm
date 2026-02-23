"""
Shared utilities for budget and zone-budget Excel export and unit conversion.

Provides unit conversion factor application, title-line marker substitution,
and time-range filtering used by both ``budget_excel`` and ``zbudget_excel``.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Column type → conversion factor category (from Budget_Parameters.f90)
# 1 (VR), 2 (VLB), 3 (VLE), 6-11 (LWU variants) → volume
# 4 (AR) → area
# 5 (LT) → length
_VOLUME_TYPE_CODES = {1, 2, 3, 6, 7, 8, 9, 10, 11}
_AREA_TYPE_CODES = {4}
_LENGTH_TYPE_CODES = {5}


def apply_unit_conversion(
    values: NDArray[np.float64],
    column_types: list[int],
    length_factor: float = 1.0,
    area_factor: float = 1.0,
    volume_factor: float = 1.0,
) -> NDArray[np.float64]:
    """Apply IWFM unit conversion factors to budget data columns.

    Parameters
    ----------
    values : NDArray[np.float64]
        2-D array of shape ``(n_timesteps, n_columns)``.
    column_types : list[int]
        IWFM data-type code for each column (see ``Budget_Parameters.f90``).
    length_factor : float
        Multiplicative factor for length columns (type 5).
    area_factor : float
        Multiplicative factor for area columns (type 4).
    volume_factor : float
        Multiplicative factor for volume columns (types 1-3, 6-11).

    Returns
    -------
    NDArray[np.float64]
        Converted array (new copy; input is not modified).
    """
    result: NDArray[np.float64] = values.copy()
    for col_idx, ctype in enumerate(column_types):
        if col_idx >= result.shape[1]:
            break
        if ctype in _VOLUME_TYPE_CODES:
            result[:, col_idx] *= volume_factor
        elif ctype in _AREA_TYPE_CODES:
            result[:, col_idx] *= area_factor
        elif ctype in _LENGTH_TYPE_CODES:
            result[:, col_idx] *= length_factor
    return result


def format_title_lines(
    titles: list[str],
    location_name: str,
    area: float | None,
    length_unit: str,
    area_unit: str,
    volume_unit: str,
) -> list[str]:
    r"""Substitute IWFM unit markers in title strings.

    Recognised markers: ``@UNITVL@``, ``@UNITAR@``, ``@UNITLT@``,
    ``@LOCNAME@``, ``@AREA@``.

    Parameters
    ----------
    titles : list[str]
        Raw title lines from ``BudgetHeader.ascii_output.titles``.
    location_name : str
        Name inserted for ``@LOCNAME@``.
    area : float or None
        Area value inserted for ``@AREA@``.  ``None`` becomes ``"N/A"``.
    length_unit, area_unit, volume_unit : str
        Unit strings inserted for the corresponding markers.

    Returns
    -------
    list[str]
        Title lines with all markers replaced.
    """
    out: list[str] = []
    area_str = f"{area:.2f}" if area is not None else "N/A"
    for line in titles:
        line = line.replace("@UNITVL@", volume_unit)
        line = line.replace("@UNITAR@", area_unit)
        line = line.replace("@UNITLT@", length_unit)
        line = line.replace("@LOCNAME@", location_name)
        line = line.replace("@AREA@", area_str)
        out.append(line)
    return out


def filter_time_range(
    df: pd.DataFrame,
    begin_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    """Filter a DataFrame to the requested time window.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose index is a :class:`~pandas.DatetimeIndex`.
    begin_date, end_date : str or None
        IWFM datetime strings (``MM/DD/YYYY_HH:MM``).  ``None`` means no
        bound in that direction.

    Returns
    -------
    pd.DataFrame
        Filtered copy (or the original if neither bound applies).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return df

    from pyiwfm.io.budget import parse_iwfm_datetime

    start_dt: datetime | None = None
    end_dt: datetime | None = None

    if begin_date:
        start_dt = parse_iwfm_datetime(begin_date)
    if end_date:
        end_dt = parse_iwfm_datetime(end_date)

    if start_dt is not None:
        df = df[df.index >= start_dt]  # type: ignore[index]
    if end_dt is not None:
        df = df[df.index <= end_dt]  # type: ignore[index]
    return df
