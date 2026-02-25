"""
CalcTypHyd — compute typical hydrographs from observation well data.

Mirrors the Fortran CalcTypHyd utility. The algorithm:

1. Group observations into seasonal periods (default: 4 seasons)
2. Compute seasonal averages per well
3. De-mean each well's seasonal series
4. Compute cluster-weighted average of de-meaned series

Example
-------
>>> from pyiwfm.calibration.calctyphyd import compute_typical_hydrographs
>>> result = compute_typical_hydrographs(water_levels, cluster_weights)
>>> for th in result.hydrographs:
...     print(f"Cluster {th.cluster_id}: {len(th.contributing_wells)} wells")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pyiwfm.io.smp import SMPTimeSeries

# Default seasonal periods matching IWFM CalcTypHyd convention
_DEFAULT_SEASONS = [
    ("Winter", [12, 1, 2], "01/15"),
    ("Spring", [3, 4, 5], "04/15"),
    ("Summer", [6, 7, 8], "07/15"),
    ("Fall", [9, 10, 11], "10/15"),
]


@dataclass
class SeasonalPeriod:
    """Definition of a seasonal time period.

    Attributes
    ----------
    name : str
        Season name (e.g., "Winter").
    months : list[int]
        Month numbers belonging to this season (1-12).
    representative_date : str
        Representative date for this season in ``MM/DD`` format.
    """

    name: str
    months: list[int]
    representative_date: str


@dataclass
class CalcTypHydConfig:
    """Configuration for typical hydrograph computation.

    Attributes
    ----------
    seasonal_periods : list[SeasonalPeriod] | None
        Seasonal period definitions. Uses 4 standard seasons if ``None``.
    min_records_per_season : int
        Minimum observations required per season to include a well.
    """

    seasonal_periods: list[SeasonalPeriod] | None = None
    min_records_per_season: int = 1


@dataclass
class TypicalHydrograph:
    """A typical hydrograph for one cluster.

    Attributes
    ----------
    cluster_id : int
        Cluster identifier (0-based).
    times : NDArray[np.datetime64]
        Representative timestamps for each season.
    values : NDArray[np.float64]
        De-meaned typical water level values.
    contributing_wells : list[str]
        Wells that contributed to this hydrograph.
    """

    cluster_id: int
    times: NDArray[np.datetime64]
    values: NDArray[np.float64]
    contributing_wells: list[str]


@dataclass
class CalcTypHydResult:
    """Result of typical hydrograph computation.

    Attributes
    ----------
    hydrographs : list[TypicalHydrograph]
        One typical hydrograph per cluster.
    well_means : dict[str, float]
        Mean water level per well (used for de-meaning).
    """

    hydrographs: list[TypicalHydrograph]
    well_means: dict[str, float]


def _get_seasons(config: CalcTypHydConfig) -> list[SeasonalPeriod]:
    """Get seasonal period definitions from config or defaults."""
    if config.seasonal_periods is not None:
        return config.seasonal_periods
    return [
        SeasonalPeriod(name=name, months=months, representative_date=rep)
        for name, months, rep in _DEFAULT_SEASONS
    ]


def read_cluster_weights(filepath: Path) -> dict[str, NDArray[np.float64]]:
    """Read cluster membership weights from a text file.

    The file format is whitespace-separated with columns:
    ``well_id  w1  w2  ...  wN``

    Parameters
    ----------
    filepath : Path
        Path to the weights file.

    Returns
    -------
    dict[str, NDArray[np.float64]]
        Mapping of well ID to membership weight array.
    """
    weights: dict[str, NDArray[np.float64]] = {}

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            well_id = parts[0]
            wts = np.array([float(x) for x in parts[1:]], dtype=np.float64)
            weights[well_id] = wts

    return weights


def compute_seasonal_averages(
    water_levels: dict[str, SMPTimeSeries],
    config: CalcTypHydConfig | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Compute seasonal average water levels per well.

    Parameters
    ----------
    water_levels : dict[str, SMPTimeSeries]
        Water level time series by well ID.
    config : CalcTypHydConfig | None
        Configuration.

    Returns
    -------
    dict[str, NDArray[np.float64]]
        Mapping of well ID to seasonal averages array with shape
        ``(n_seasons,)``.  NaN where insufficient data.
    """
    if config is None:
        config = CalcTypHydConfig()

    seasons = _get_seasons(config)
    n_seasons = len(seasons)

    # Build month-to-season lookup
    month_to_season: dict[int, int] = {}
    for si, sp in enumerate(seasons):
        for m in sp.months:
            month_to_season[m] = si

    result: dict[str, NDArray[np.float64]] = {}

    for well_id, ts in water_levels.items():
        avgs = np.full(n_seasons, np.nan)

        # Extract months from datetime64 array
        # Convert to Python datetimes for month extraction
        dts = ts.times.astype("datetime64[M]").astype(int) % 12 + 1

        for si in range(n_seasons):
            season_months = set(seasons[si].months)
            mask = np.array([int(m) in season_months for m in dts])
            valid = mask & ts.valid_mask
            if np.sum(valid) >= config.min_records_per_season:
                avgs[si] = float(np.nanmean(ts.values[valid]))

        result[well_id] = avgs

    return result


def compute_typical_hydrographs(
    water_levels: dict[str, SMPTimeSeries],
    cluster_weights: dict[str, NDArray[np.float64]],
    config: CalcTypHydConfig | None = None,
) -> CalcTypHydResult:
    """Compute typical hydrographs using cluster membership weights.

    Parameters
    ----------
    water_levels : dict[str, SMPTimeSeries]
        Water level time series by well ID.
    cluster_weights : dict[str, NDArray[np.float64]]
        Cluster membership weights by well ID.
    config : CalcTypHydConfig | None
        Configuration.

    Returns
    -------
    CalcTypHydResult
        Typical hydrographs and per-well means.
    """
    if config is None:
        config = CalcTypHydConfig()

    seasons = _get_seasons(config)
    n_seasons = len(seasons)

    # Compute seasonal averages
    seasonal_avgs = compute_seasonal_averages(water_levels, config)

    # Compute per-well means
    well_means: dict[str, float] = {}
    for well_id, avgs in seasonal_avgs.items():
        valid = ~np.isnan(avgs)
        if np.any(valid):
            well_means[well_id] = float(np.nanmean(avgs))
        else:
            well_means[well_id] = 0.0

    # Determine number of clusters
    n_clusters = 0
    for wts in cluster_weights.values():
        n_clusters = max(n_clusters, len(wts))

    if n_clusters == 0:
        return CalcTypHydResult(hydrographs=[], well_means=well_means)

    # Representative times (use year 2000 as reference)
    rep_times = []
    for sp in seasons:
        month, day = sp.representative_date.split("/")
        rep_times.append(np.datetime64(f"2000-{int(month):02d}-{int(day):02d}"))
    rep_times_arr = np.array(rep_times, dtype="datetime64[D]")

    # Compute typical hydrographs
    hydrographs: list[TypicalHydrograph] = []
    for c in range(n_clusters):
        weighted_sum = np.zeros(n_seasons)
        weight_sum = np.zeros(n_seasons)
        contributing: list[str] = []

        for well_id in cluster_weights:
            if well_id not in seasonal_avgs:
                continue
            wts = cluster_weights[well_id]
            if c >= len(wts):
                continue
            w = wts[c]
            if w <= 0.0:
                continue

            avgs = seasonal_avgs[well_id]
            mean = well_means.get(well_id, 0.0)

            # De-mean and weight
            for si in range(n_seasons):
                if not np.isnan(avgs[si]):
                    weighted_sum[si] += w * (avgs[si] - mean)
                    weight_sum[si] += w

            if w > 0.0:
                contributing.append(well_id)

        # Normalize
        values = np.where(weight_sum > 0, weighted_sum / weight_sum, np.nan)

        hydrographs.append(
            TypicalHydrograph(
                cluster_id=c,
                times=rep_times_arr,
                values=values,
                contributing_wells=contributing,
            )
        )

    return CalcTypHydResult(hydrographs=hydrographs, well_means=well_means)
