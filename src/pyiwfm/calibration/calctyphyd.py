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

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TextIO

import numpy as np
from numpy.typing import NDArray

from pyiwfm.io.smp import SMPTimeSeries

logger = logging.getLogger(__name__)

# Default seasonal periods matching IWFM CalcTypHyd convention (quarterly)
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


# Biannual seasonal periods — spring high / fall low (2 periods per year).
# Uses months 1-4 for spring and 8-11 for fall, matching the clustering
# basis in process_cnra_gw_data.py.  Representative dates are March 1 and
# October 1.
BIANNUAL_SEASONS = [
    SeasonalPeriod("Spring", [1, 2, 3, 4], "03/01"),
    SeasonalPeriod("Fall", [8, 9, 10, 11], "10/01"),
]


@dataclass
class CalcTypHydConfig:
    """Configuration for typical hydrograph computation.

    Attributes
    ----------
    seasonal_periods : list[SeasonalPeriod] | None
        Seasonal period definitions. Uses 4 standard seasons if ``None``.
    min_records_per_season : int
        Minimum observations required per season to include a well.
    start_date : datetime | None
        Optional start date for filtering water levels.
    end_date : datetime | None
        Optional end date for filtering water levels.
    """

    seasonal_periods: list[SeasonalPeriod] | None = None
    min_records_per_season: int = 1
    start_date: datetime | None = None
    end_date: datetime | None = None


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


def read_cluster_weights(
    filepath: Path,
    n_clusters: int | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Read cluster membership weights from a text file.

    The file format is whitespace-separated with columns:
    ``well_id  w1  w2  ...  wN  [extra_cols...]``

    Header auto-detection is applied unconditionally: the second token of
    the first non-blank, non-comment line is tested with ``float()``.  If
    the conversion fails the line is treated as a column header and skipped.
    This allows the function to handle files both with and without a header
    row without requiring ``n_clusters`` to be supplied.

    Parameters
    ----------
    filepath : Path
        Path to the weights file.
    n_clusters : int | None
        If provided, only read this many weight columns after the well ID.
        Extra columns (e.g. ``memb``, ``cluster``, ``subregion``) are ignored.
        When ``None`` all numeric columns are read; header auto-detection is
        still applied so files with a text header row work correctly.

    Returns
    -------
    dict[str, NDArray[np.float64]]
        Mapping of well ID to membership weight array.
    """
    weights: dict[str, NDArray[np.float64]] = {}
    header_skipped = False

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue

            # Auto-detect and skip header: peek at first data line and try to
            # parse the second token as a float.  If it fails the line is a
            # header row (column labels) and is skipped.
            if not header_skipped:
                try:
                    float(parts[1])
                except ValueError:
                    header_skipped = True
                    continue
                header_skipped = True

            well_id = parts[0]
            value_parts = parts[1:]
            if n_clusters is not None:
                value_parts = value_parts[:n_clusters]
            wts = np.array([float(x) for x in value_parts], dtype=np.float64)
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

    # Date-range filtering bounds
    start_dt64: np.datetime64 | None = None
    end_dt64: np.datetime64 | None = None
    if config.start_date is not None:
        start_dt64 = np.datetime64(config.start_date, "s")
    if config.end_date is not None:
        end_dt64 = np.datetime64(config.end_date, "s")

    result: dict[str, NDArray[np.float64]] = {}

    for well_id, ts in water_levels.items():
        avgs = np.full(n_seasons, np.nan)

        # Extract months from datetime64 array
        # Convert to Python datetimes for month extraction
        times_s = ts.times.astype("datetime64[s]")
        dts = ts.times.astype("datetime64[M]").astype(int) % 12 + 1

        # Apply date-range filter
        in_range = np.ones(len(ts.times), dtype=np.bool_)
        if start_dt64 is not None:
            in_range &= times_s >= start_dt64
        if end_dt64 is not None:
            in_range &= times_s <= end_dt64

        for si in range(n_seasons):
            season_months = set(seasons[si].months)
            mask = np.array([int(m) in season_months for m in dts])
            valid = mask & ts.valid_mask & in_range
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


# =====================================================================
# Config file parsing (Fortran CalcTypHyd .in format)
# =====================================================================


@dataclass
class CalcTypHydFileConfig:
    """Configuration parsed from a Fortran-format CalcTypHyd ``.in`` file.

    Attributes
    ----------
    water_level_path : Path
        SMP file with water level observations.
    weights_path : Path
        Cluster membership weights file.
    n_clusters : int
        Total number of clusters in the weights file.
    n_wells : int
        Number of wells.
    desired_clusters : list[int]
        1-based cluster IDs to generate output for.
    start_date : str
        Start date as ``"MM/DD/YYYY"``.
    end_date : str
        End date as ``"MM/DD/YYYY"``.
    pest_base_name : str
        Base name for PEST parameter naming.
    periods : list[SeasonalPeriod]
        Averaging period definitions.
    """

    water_level_path: Path = field(default_factory=lambda: Path())
    weights_path: Path = field(default_factory=lambda: Path())
    n_clusters: int = 0
    n_wells: int = 0
    desired_clusters: list[int] = field(default_factory=list)
    start_date: str = ""
    end_date: str = ""
    pest_base_name: str = ""
    periods: list[SeasonalPeriod] = field(default_factory=list)


def _read_config_line(f: TextIO) -> str:
    """Read next non-comment, non-blank line from config file (no stripping).

    Comment lines start with ``#`` or ``C `` (C followed by space) to
    avoid matching paths like ``C:\\...``.
    """
    for raw in f:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # "C " (capital C + space) is a Fortran comment prefix
        if stripped.startswith("C ") or stripped.startswith("C\t") or stripped == "C":
            continue
        return stripped
    raise StopIteration("Unexpected end of config file")


def _read_config_value(f: TextIO) -> str:
    """Read next non-comment, non-blank line from config file, strip inline comment.

    Handles both ``#`` and standalone ``/`` as inline comment delimiters.
    Returns the first whitespace-delimited token (matching Fortran list-directed read).
    """
    raw = _read_config_line(f)
    # Strip #-style inline comments first (common in CalcTypHyd configs)
    if "#" in raw:
        raw = raw[: raw.index("#")].strip()
    # Also strip standalone / delimiter (IWFM style)
    tokens = raw.split()
    if len(tokens) >= 1:
        for i, tok in enumerate(tokens):
            if tok == "/":
                return " ".join(tokens[:i]).strip()
    # Return first token (matches Fortran list-directed read for scalars)
    return tokens[0] if tokens else raw.strip()


def _read_config_values(f: TextIO) -> str:
    """Read next line and return all tokens before inline comment.

    Like :func:`_read_config_value` but returns the full cleaned line
    (for multi-value lines like month lists).
    """
    raw = _read_config_line(f)
    if "#" in raw:
        raw = raw[: raw.index("#")].strip()
    tokens = raw.split()
    for i, tok in enumerate(tokens):
        if tok == "/":
            return " ".join(tokens[:i]).strip()
    return raw.strip()


def read_calctyphyd_config(filepath: Path) -> CalcTypHydFileConfig:
    """Parse a Fortran-format CalcTypHyd ``.in`` config file.

    Parameters
    ----------
    filepath : Path
        Path to the ``.in`` file.

    Returns
    -------
    CalcTypHydFileConfig
        Parsed configuration.
    """
    config = CalcTypHydFileConfig()
    base_dir = filepath.parent

    with open(filepath, encoding="utf-8") as f:
        # Line 1: water level file (use full line for paths with spaces)
        raw = _read_config_values(f)
        wl_path = Path(raw)
        config.water_level_path = wl_path if wl_path.is_absolute() else base_dir / wl_path

        # Line 2: weights file
        raw = _read_config_values(f)
        wt_path = Path(raw)
        config.weights_path = wt_path if wt_path.is_absolute() else base_dir / wt_path

        # Line 3: n_clusters
        config.n_clusters = int(_read_config_value(f))

        # Line 4: n_wells
        config.n_wells = int(_read_config_value(f))

        # Line 5: n_desired_hydrographs
        n_desired = int(_read_config_value(f))

        # Lines 6..5+N: desired cluster IDs (1-based)
        config.desired_clusters = []
        for _ in range(n_desired):
            config.desired_clusters.append(int(_read_config_value(f)))

        # Start date (use first token to avoid stripping date slashes)
        config.start_date = _read_config_line(f).split()[0]

        # End date
        config.end_date = _read_config_line(f).split()[0]

        # PEST base name
        config.pest_base_name = _read_config_value(f)

        # Number of averaging periods
        n_periods = int(_read_config_value(f))

        # Read each period (3 lines each)
        config.periods = []
        for p_idx in range(n_periods):
            n_months = int(_read_config_value(f))
            month_tokens = _read_config_values(f).split()
            months = [int(m) for m in month_tokens[:n_months]]
            rep_tokens = _read_config_values(f).split()
            rep_month = int(rep_tokens[0])
            rep_day = int(rep_tokens[1])
            config.periods.append(
                SeasonalPeriod(
                    name=f"P{p_idx + 1}",
                    months=months,
                    representative_date=f"{rep_month:02d}/{rep_day:02d}",
                )
            )

    return config


# =====================================================================
# Time-series output (one value per period per year)
# =====================================================================


@dataclass
class TypicalHydrographTimeSeries:
    """Per-period-year time series for one cluster.

    Attributes
    ----------
    cluster_id : int
        Cluster identifier (1-based, matching desired_clusters).
    dates : list[datetime]
        Representative dates for each period-year entry.
    values : list[float]
        De-meaned cluster-weighted values.
    pest_names : list[str]
        PEST parameter names (one per entry).
    """

    cluster_id: int
    dates: list[datetime] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    pest_names: list[str] = field(default_factory=list)


@dataclass
class CalcTypHydTimeSeriesResult:
    """Result of time-series typical hydrograph computation.

    Attributes
    ----------
    hydrographs : list[TypicalHydrographTimeSeries]
        One time series per desired cluster.
    well_means : dict[str, float]
        Mean water level per well.
    """

    hydrographs: list[TypicalHydrographTimeSeries]
    well_means: dict[str, float]


def _parse_mmddyyyy(date_str: str) -> datetime:
    """Parse ``MM/DD/YYYY`` string to datetime."""
    parts = date_str.strip().split("/")
    return datetime(int(parts[2]), int(parts[0]), int(parts[1]))


def compute_typical_hydrographs_timeseries(
    water_levels: dict[str, SMPTimeSeries],
    cluster_weights: dict[str, NDArray[np.float64]],
    file_config: CalcTypHydFileConfig,
    config: CalcTypHydConfig | None = None,
) -> CalcTypHydTimeSeriesResult:
    """Compute typical hydrographs as per-period-year time series.

    Unlike :func:`compute_typical_hydrographs` which produces a single
    seasonal pattern, this function produces one value per period per year
    within the date range — matching the Fortran CalcTypHyd output.

    The algorithm matches the Fortran ``Class_CalcTypeHyd``:

    1. Bin observations into period-year slots (like Fortran ``rMonAvg``).
       Only observations in active months contribute.
    2. Compute per-well mean as the mean of these slot averages
       (like Fortran ``ComputeMeans``), NOT the mean of raw observations.
    3. De-mean each slot average and compute cluster-weighted average.

    Parameters
    ----------
    water_levels : dict[str, SMPTimeSeries]
        Water level time series by well ID.
    cluster_weights : dict[str, NDArray[np.float64]]
        Cluster membership weights by well ID.
    file_config : CalcTypHydFileConfig
        Configuration from the ``.in`` file (date range, periods, clusters).
    config : CalcTypHydConfig | None
        Additional configuration (min_records_per_season).

    Returns
    -------
    CalcTypHydTimeSeriesResult
        Per-period-year time series for each desired cluster.
    """
    if config is None:
        config = CalcTypHydConfig()

    start = _parse_mmddyyyy(file_config.start_date)
    end = _parse_mmddyyyy(file_config.end_date)
    periods = file_config.periods
    pest_base = file_config.pest_base_name

    # Slot validity: a (year, period) slot is valid when its representative
    # month/year falls within the start-end range (matching Fortran
    # MonthRange-based filtering on remapped dates).
    start_val = start.year * 12 + start.month
    end_val = end.year * 12 + end.month

    # Wells present in both water levels and weights
    common_wells = set(water_levels.keys()) & set(cluster_weights.keys())

    # Step 1: Pre-compute period-year averages per well (Fortran rMonAvg).
    # period_avgs[well_id][(year, p_idx)] = average WL for that slot.
    period_avgs: dict[str, dict[tuple[int, int], float]] = {}

    for well_id in common_wells:
        ts = water_levels[well_id]
        months = ts.times.astype("datetime64[M]").astype(int) % 12 + 1
        years = ts.times.astype("datetime64[Y]").astype(int) + 1970
        valid = ts.valid_mask & ~np.isnan(ts.values)

        months_arr = months.astype(np.int32)
        years_arr = years.astype(np.int32)

        avgs: dict[tuple[int, int], float] = {}
        for year in range(start.year, end.year + 1):
            year_mask = years_arr == year
            for p_idx, period in enumerate(periods):
                rep_month = int(period.representative_date.split("/")[0])
                rep_val = year * 12 + rep_month
                if rep_val < start_val or rep_val > end_val:
                    continue

                # Mask: observations in this period's months AND this year
                season_mask = np.isin(months_arr, list(period.months))
                mask = season_mask & year_mask & valid

                if np.any(mask):
                    avgs[(year, p_idx)] = float(np.mean(ts.values[mask]))

        period_avgs[well_id] = avgs

    # Step 2: Compute per-well mean as mean of period-year slot averages
    # (matching Fortran ComputeMeans — NOT mean of raw observations).
    well_means: dict[str, float] = {}
    for well_id, avgs in period_avgs.items():
        if avgs:
            well_means[well_id] = sum(avgs.values()) / len(avgs)
        else:
            well_means[well_id] = 0.0

    # Step 3: Generate type hydrographs
    hydrographs: list[TypicalHydrographTimeSeries] = []

    for cluster_id in file_config.desired_clusters:
        c_idx = cluster_id - 1  # 0-based index into weight arrays
        ts_result = TypicalHydrographTimeSeries(cluster_id=cluster_id)

        for year in range(start.year, end.year + 1):
            for p_idx, period in enumerate(periods):
                rep_parts = period.representative_date.split("/")
                rep_month = int(rep_parts[0])
                rep_day = int(rep_parts[1])

                # Check slot validity (Fortran MonthRange check)
                rep_val = year * 12 + rep_month
                if rep_val < start_val or rep_val > end_val:
                    continue

                weighted_sum = 0.0
                weight_sum = 0.0

                for well_id in common_wells:
                    wts = cluster_weights[well_id]
                    if c_idx >= len(wts):
                        continue
                    w = wts[c_idx]
                    if w <= 0.0:
                        continue

                    slot_avg = period_avgs[well_id].get((year, p_idx))
                    if slot_avg is not None:
                        well_mean = well_means.get(well_id, 0.0)
                        weighted_sum += w * (slot_avg - well_mean)
                        weight_sum += w

                if weight_sum > 0:
                    value = weighted_sum / weight_sum
                    rep_date = datetime(year, rep_month, rep_day)
                    # Fortran PEST naming: global month-based sequence
                    global_seq = (year - start.year) * 12 + rep_month
                    pest_name = f"{pest_base}{cluster_id}_{global_seq}"

                    ts_result.dates.append(rep_date)
                    ts_result.values.append(value)
                    ts_result.pest_names.append(pest_name)

        hydrographs.append(ts_result)

    return CalcTypHydTimeSeriesResult(hydrographs=hydrographs, well_means=well_means)


# =====================================================================
# PEST output writer
# =====================================================================


def write_pest_output(
    result: CalcTypHydTimeSeriesResult,
    file_config: CalcTypHydFileConfig,
    output_dir: Path,
) -> list[Path]:
    """Write PEST ``.out`` and ``.ins`` files matching Fortran CalcTypHyd format.

    For each desired cluster, writes:

    - ``sim_{weights_stem}_cls{N}.out`` — data file
    - ``sim_{weights_stem}_cls{N}.ins`` — PEST instruction file

    Parameters
    ----------
    result : CalcTypHydTimeSeriesResult
        Time-series typical hydrograph results.
    file_config : CalcTypHydFileConfig
        Config (for weights file stem and PEST base name).
    output_dir : Path
        Directory for output files.

    Returns
    -------
    list[Path]
        Paths to all written files.
    """
    from pyiwfm.calibration.pest_files import generate_thyd_ins

    output_dir.mkdir(parents=True, exist_ok=True)
    weights_stem = file_config.weights_path.stem.lower()
    written: list[Path] = []

    for th in result.hydrographs:
        cls_id = th.cluster_id

        # .out file — matches Fortran format: name(16) date(10) value at col 34
        out_path = output_dir / f"sim_{weights_stem}_cls{cls_id}.out"
        with open(out_path, "w", encoding="utf-8") as f:
            # Header matching Fortran format
            f.write(
                f"{'PEST_NAME':>14s}        "
                f"{'DATE':4s}"
                f"                          "
                f"sim_{weights_stem}_cls{cls_id}\n"
            )
            for i in range(len(th.dates)):
                name = th.pest_names[i]
                date_str = th.dates[i].strftime("%m/%d/%Y")
                value = th.values[i]
                # Fortran format: name(16) + date(10) + value(F20.6)
                # Value occupies columns 27-46, PEST extracts 34:46
                f.write(f"{name:<16s}{date_str:>10s}{value:20.6f}\n")
        written.append(out_path)

        # .ins file — PEST instruction file generated from .out
        ins_path = output_dir / f"sim_{weights_stem}_cls{cls_id}.ins"
        generate_thyd_ins(out_path, ins_path)
        written.append(ins_path)

    return written


def compute_obs_type_hydrographs(
    water_levels: dict[str, SMPTimeSeries],
    cluster_weights: dict[str, NDArray[np.float64]],
    file_config: CalcTypHydFileConfig,
    output_dir: Path,
    *,
    obs_prefix: str = "obs_",
) -> list[Path]:
    """Compute observed type hydrographs and write PEST output files.

    Convenience wrapper that computes period-year type hydrographs from
    observation data and writes ``obs_*.out`` / ``obs_*.ins`` files.

    This function is intended to be called from the data processing pipeline
    (Step 8) to pre-compute the observed type hydrograph targets.  The sim
    counterparts are produced by CalcTypeHyd reading the continuous sim SMP.

    Both sim and obs type hydrographs use the same seasonal period
    aggregation (window-averaging) so they are directly comparable.

    Parameters
    ----------
    water_levels : dict[str, SMPTimeSeries]
        Observed water level time series by well ID (from GW_Obs.smp).
    cluster_weights : dict[str, NDArray[np.float64]]
        Fuzzy cluster membership weights by well ID.
    file_config : CalcTypHydFileConfig
        CalcTypHyd configuration (date range, periods, desired clusters,
        pest base name, weights path — for naming output files).
    output_dir : Path
        Directory for output files.
    obs_prefix : str
        Prefix for output file names (default ``"obs_"``).

    Returns
    -------
    list[Path]
        Paths to all written files.
    """
    from pyiwfm.calibration.pest_files import generate_thyd_ins

    result = compute_typical_hydrographs_timeseries(
        water_levels,
        cluster_weights,
        file_config,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    weights_stem = file_config.weights_path.stem.lower()
    written: list[Path] = []

    for th in result.hydrographs:
        cls_id = th.cluster_id

        out_path = output_dir / f"{obs_prefix}{weights_stem}_cls{cls_id}.out"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(
                f"{'PEST_NAME':>14s}        "
                f"{'DATE':4s}"
                f"                          "
                f"{obs_prefix}{weights_stem}_cls{cls_id}\n"
            )
            for i in range(len(th.dates)):
                name = th.pest_names[i]
                date_str = th.dates[i].strftime("%m/%d/%Y")
                value = th.values[i]
                f.write(f"{name:<16s}{date_str:>10s}{value:20.6f}\n")
        written.append(out_path)

        ins_path = output_dir / f"{obs_prefix}{weights_stem}_cls{cls_id}.ins"
        generate_thyd_ins(out_path, ins_path)
        written.append(ins_path)

    logger.info(
        "Wrote %d obs type hydrograph files for %s (%d clusters)",
        len(written),
        weights_stem,
        len(result.hydrographs),
    )
    return written
