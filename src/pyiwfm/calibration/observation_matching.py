"""
Observation matching and fit statistics for calibration analysis.

Matches simulated vs observed time series by date and computes standard
goodness-of-fit statistics (RMSE, MAE, bias, NSE, R-squared).

Example
-------
>>> matcher = ObservationMatcher()
>>> result = matcher.match_by_date(sim_times, sim_values, obs_times, obs_values)
>>> print(result.stats.rmse, result.stats.nse)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class FitStatistics:
    """Goodness-of-fit statistics for sim vs obs comparison.

    Attributes
    ----------
    rmse : float
        Root mean square error.
    mae : float
        Mean absolute error.
    bias : float
        Mean bias (obs - sim).
    nse : float
        Nash-Sutcliffe efficiency.
    r_squared : float
        Coefficient of determination.
    n_matched : int
        Number of date-matched pairs.
    """

    rmse: float
    mae: float
    bias: float
    nse: float
    r_squared: float
    n_matched: int


@dataclass
class MatchResult:
    """Result of date-matching sim and obs time series.

    Attributes
    ----------
    sim_matched : NDArray[np.float64]
        Simulated values at matched dates.
    obs_matched : NDArray[np.float64]
        Observed values at matched dates.
    stats : FitStatistics
        Computed fit statistics.
    """

    sim_matched: NDArray[np.float64]
    obs_matched: NDArray[np.float64]
    stats: FitStatistics


def _date_key(dt: object) -> tuple[int, int, int]:
    """Extract (year, month, day) tuple for robust date matching."""
    import datetime as _dt_mod

    if isinstance(dt, _dt_mod.date):
        return (dt.year, dt.month, dt.day)
    # numpy datetime64 or similar — convert to Python date
    py_date = np.datetime64(dt, "D").astype("datetime64[D]").astype(object)  # type: ignore[call-overload]
    return (py_date.year, py_date.month, py_date.day)


class ObservationMatcher:
    """Match simulated vs observed time series and compute fit statistics."""

    @staticmethod
    def compute_statistics(
        sim: NDArray[np.float64],
        obs: NDArray[np.float64],
    ) -> FitStatistics:
        """Compute goodness-of-fit statistics between paired sim/obs arrays.

        Parameters
        ----------
        sim : array-like
            Simulated values.
        obs : array-like
            Observed values (same length as sim).

        Returns
        -------
        FitStatistics
        """
        sim = np.asarray(sim, dtype=np.float64)
        obs = np.asarray(obs, dtype=np.float64)

        if len(sim) != len(obs):
            raise ValueError(f"Arrays must have same length: sim={len(sim)}, obs={len(obs)}")

        n = len(sim)
        if n == 0:
            return FitStatistics(
                rmse=float("nan"),
                mae=float("nan"),
                bias=float("nan"),
                nse=float("nan"),
                r_squared=float("nan"),
                n_matched=0,
            )

        residuals = obs - sim
        rmse = float(np.sqrt(np.mean(residuals**2)))
        mae = float(np.mean(np.abs(residuals)))
        bias = float(np.mean(residuals))

        obs_mean = np.mean(obs)
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((obs - obs_mean) ** 2))
        nse = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        # R-squared via correlation
        if n >= 2 and ss_tot > 0:
            sim_mean = np.mean(sim)
            ss_sim = float(np.sum((sim - sim_mean) ** 2))
            if ss_sim > 0:
                r = float(np.sum((sim - sim_mean) * (obs - obs_mean))) / (
                    np.sqrt(ss_sim) * np.sqrt(ss_tot)
                )
                r_squared = r**2
            else:
                r_squared = float("nan")
        else:
            r_squared = float("nan")

        return FitStatistics(
            rmse=rmse,
            mae=mae,
            bias=bias,
            nse=nse,
            r_squared=r_squared,
            n_matched=n,
        )

    def match_by_date(
        self,
        sim_times: NDArray[np.datetime64],
        sim_values: NDArray[np.float64],
        obs_times: NDArray[np.datetime64],
        obs_values: NDArray[np.float64],
    ) -> MatchResult:
        """Date-match simulated to observed values and compute statistics.

        Matches on (year, month, day), ignoring time-of-day. When multiple
        records share the same date, the first is used.

        Parameters
        ----------
        sim_times : array-like
            Simulated timestamps (datetime-like).
        sim_values : array-like
            Simulated values.
        obs_times : array-like
            Observed timestamps (datetime-like).
        obs_values : array-like
            Observed values.

        Returns
        -------
        MatchResult
        """
        sim_values = np.asarray(sim_values, dtype=np.float64)
        obs_values = np.asarray(obs_values, dtype=np.float64)

        # Build sim lookup by date key
        sim_by_date: dict[tuple[int, int, int], float] = {}
        for i, t in enumerate(sim_times):
            key = _date_key(t)
            if key not in sim_by_date:
                sim_by_date[key] = float(sim_values[i])

        # Match obs dates to sim
        matched_sim: list[float] = []
        matched_obs: list[float] = []
        for i, t in enumerate(obs_times):
            key = _date_key(t)
            if key in sim_by_date:
                matched_sim.append(sim_by_date[key])
                matched_obs.append(float(obs_values[i]))

        sim_arr = np.array(matched_sim, dtype=np.float64)
        obs_arr = np.array(matched_obs, dtype=np.float64)
        stats = self.compute_statistics(sim_arr, obs_arr)

        return MatchResult(sim_matched=sim_arr, obs_matched=obs_arr, stats=stats)
