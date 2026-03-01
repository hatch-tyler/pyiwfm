"""Residual analysis engine for IWFM model calibration.

Provides tools for computing, filtering, and summarizing residuals
(observed minus simulated) from calibration runs.  Integrates with
the SMP I/O layer and the existing interpolation pipeline in
:mod:`pyiwfm.calibration.iwfm2obs`.

Functions
---------
- :func:`compute_residuals` — Join observed/simulated SMP dicts and compute residuals.
- :func:`mean_residuals` — Per-well mean residual.
- :func:`max_residuals` — Per-well maximum absolute residual.
- :func:`filter_residuals` — Filter by layer, subregion, date range, or screen type.
- :func:`residual_summary` — Aggregate statistics over a residual DataFrame.
- :func:`export_residual_table` — Write residuals to CSV.

Enumerations
------------
- :class:`WellScreenType` — Categorisation of well screen knowledge.
"""

from __future__ import annotations

import enum
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pyiwfm.calibration.iwfm2obs import interpolate_to_obs_times
from pyiwfm.io.smp import SMPTimeSeries


class WellScreenType(enum.Enum):
    """Classification of well screen information quality.

    Mirrors the categories used by SSPA calibration workflows.
    """

    KNOWN_SCREENS = "known_screens"
    INTERPOLATED_TOS = "interpolated_tos"
    INTERPOLATED_TOS_BOS = "interpolated_tos_bos"
    UNKNOWN = "unknown"


# -- public API ---------------------------------------------------------------


def compute_residuals(
    observed: dict[str, SMPTimeSeries],
    simulated: dict[str, SMPTimeSeries],
    well_info: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Join observed and simulated time series and compute residuals.

    For each bore ID present in *both* dictionaries the simulated values
    are interpolated to the observation timestamps using
    :func:`~pyiwfm.calibration.iwfm2obs.interpolate_to_obs_times`.

    Parameters
    ----------
    observed : dict[str, SMPTimeSeries]
        Observed time series keyed by bore ID.
    simulated : dict[str, SMPTimeSeries]
        Simulated time series keyed by bore ID.
    well_info : dict[str, dict[str, Any]] | None
        Optional metadata per well.  Recognised keys:
        ``layer``, ``subregion``, ``screen_type``
        (:class:`WellScreenType` or str), ``x``, ``y``.

    Returns
    -------
    pd.DataFrame
        Columns: ``well_id``, ``datetime``, ``observed``, ``simulated``,
        ``residual`` and any metadata columns from *well_info*.
    """
    rows: list[dict[str, Any]] = []
    common_ids = sorted(set(observed) & set(simulated))

    for well_id in common_ids:
        obs_ts = observed[well_id]
        sim_ts = simulated[well_id]
        interp = interpolate_to_obs_times(obs_ts, sim_ts)

        for i in range(len(obs_ts.times)):
            obs_val = float(obs_ts.values[i])
            sim_val = float(interp.values[i])
            if np.isnan(obs_val) or np.isnan(sim_val):
                continue
            row: dict[str, Any] = {
                "well_id": well_id,
                "datetime": obs_ts.times[i],
                "observed": obs_val,
                "simulated": sim_val,
                "residual": sim_val - obs_val,
            }
            if well_info and well_id in well_info:
                info = well_info[well_id]
                for key in ("layer", "subregion", "screen_type", "x", "y"):
                    if key in info:
                        val = info[key]
                        if key == "screen_type" and isinstance(val, WellScreenType):
                            val = val.value
                        row[key] = val
            rows.append(row)

    return pd.DataFrame(rows)


def mean_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """Return mean residual per well.

    Parameters
    ----------
    df : pd.DataFrame
        Residuals DataFrame (output of :func:`compute_residuals`).

    Returns
    -------
    pd.DataFrame
        One row per well with ``well_id`` and ``mean_residual``.
    """
    grouped = df.groupby("well_id")["residual"].mean().reset_index()
    grouped.columns = pd.Index(["well_id", "mean_residual"])
    return grouped


def max_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """Return maximum absolute residual per well.

    Parameters
    ----------
    df : pd.DataFrame
        Residuals DataFrame.

    Returns
    -------
    pd.DataFrame
        One row per well with ``well_id`` and ``max_abs_residual``.
    """

    def _max_abs(s: pd.Series) -> float:  # type: ignore[type-arg]
        return float(np.max(np.abs(s.values)))

    grouped: pd.DataFrame = df.groupby("well_id")["residual"].agg(_max_abs).reset_index()
    grouped.columns = pd.Index(["well_id", "max_abs_residual"])
    return grouped


def filter_residuals(
    df: pd.DataFrame,
    *,
    layers: list[int] | None = None,
    subregions: list[int | str] | None = None,
    date_range: tuple[datetime, datetime] | None = None,
    screen_types: list[WellScreenType | str] | None = None,
) -> pd.DataFrame:
    """Filter a residuals DataFrame by one or more criteria.

    Parameters
    ----------
    df : pd.DataFrame
        Residuals DataFrame.
    layers : list[int] | None
        Keep only rows whose ``layer`` column is in this list.
    subregions : list[int | str] | None
        Keep only rows whose ``subregion`` column is in this list.
    date_range : tuple[datetime, datetime] | None
        ``(start, end)`` inclusive date window.
    screen_types : list[WellScreenType | str] | None
        Keep only rows whose ``screen_type`` matches.

    Returns
    -------
    pd.DataFrame
        Filtered copy.
    """
    mask = pd.Series(True, index=df.index)

    if layers is not None and "layer" in df.columns:
        mask &= df["layer"].isin(layers)

    if subregions is not None and "subregion" in df.columns:
        mask &= df["subregion"].isin(subregions)

    if date_range is not None:
        start, end = date_range
        dt_col = pd.to_datetime(df["datetime"])
        mask &= (dt_col >= pd.Timestamp(start)) & (dt_col <= pd.Timestamp(end))

    if screen_types is not None and "screen_type" in df.columns:
        str_types = [st.value if isinstance(st, WellScreenType) else st for st in screen_types]
        mask &= df["screen_type"].isin(str_types)

    result: pd.DataFrame = df.loc[mask].copy()
    return result


def residual_summary(df: pd.DataFrame) -> dict[str, float]:
    """Compute aggregate statistics over a residuals DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Residuals DataFrame with ``observed``, ``simulated``, and
        ``residual`` columns.

    Returns
    -------
    dict[str, float]
        Keys: ``n``, ``mean``, ``std``, ``rmse``, ``nash_sutcliffe``,
        ``correlation``, ``index_of_agreement``.
    """
    from pyiwfm.comparison.metrics import (
        correlation_coefficient,
        index_of_agreement,
        nash_sutcliffe,
        rmse,
    )

    obs = df["observed"].to_numpy(dtype=np.float64)
    sim = df["simulated"].to_numpy(dtype=np.float64)
    res = df["residual"].to_numpy(dtype=np.float64)

    n = len(res)
    if n == 0:
        return {"n": 0}

    return {
        "n": float(n),
        "mean": float(np.mean(res)),
        "std": float(np.std(res, ddof=1)) if n > 1 else 0.0,
        "rmse": rmse(obs, sim),
        "nash_sutcliffe": nash_sutcliffe(obs, sim),
        "correlation": correlation_coefficient(obs, sim) if n > 1 else 0.0,
        "index_of_agreement": index_of_agreement(obs, sim),
    }


def export_residual_table(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a residuals DataFrame to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Residuals DataFrame.
    path : str or Path
        Output CSV path.

    Returns
    -------
    Path
        Resolved path of the written file.
    """
    out = Path(path)
    df.to_csv(out, index=False)
    return out
