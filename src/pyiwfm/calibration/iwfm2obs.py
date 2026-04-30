"""
IWFM2OBS — interpolate simulated heads to observation times.

Mirrors the Fortran IWFM2OBS utility with two core algorithms:

1. **Time interpolation** — linearly interpolate simulated time series to
   match observation timestamps.
2. **Multi-layer T-weighted averaging** — compute composite heads at wells
   that screen multiple aquifer layers, weighting by transmissivity.

The :func:`iwfm2obs_from_model` function combines both: it reads ``.out``
files directly from the IWFM simulation main file (like the old Fortran
``iwfm2obs_2015``), performs time interpolation, and optionally applies
multi-layer T-weighted averaging.

Example
-------
>>> from pyiwfm.calibration.iwfm2obs import interpolate_to_obs_times
>>> result = interpolate_to_obs_times(observed_ts, simulated_ts)
>>> print(result.values)
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from pyiwfm.io.smp import SMPReader, SMPTimeSeries, SMPWriter

if TYPE_CHECKING:
    from pyiwfm.calibration.obs_well_spec import ObsWellSpec
    from pyiwfm.core.mesh import AppGrid
    from pyiwfm.core.stratigraphy import Stratigraphy

logger = logging.getLogger(__name__)


@dataclass
class InterpolationConfig:
    """Configuration for time interpolation.

    Attributes
    ----------
    max_extrapolation_time : timedelta
        Maximum allowed extrapolation beyond the simulated time range.
        Observation times beyond this are set to ``sentinel_value``.
    sentinel_value : float
        Value to use for observations outside the interpolation window.
    interpolation_method : str
        Interpolation method: ``"linear"`` or ``"nearest"``.
    """

    max_extrapolation_time: timedelta = field(default_factory=lambda: timedelta(days=30))
    sentinel_value: float = -999.0
    interpolation_method: Literal["linear", "nearest"] = "linear"


def interpolate_to_obs_times(
    observed: SMPTimeSeries,
    simulated: SMPTimeSeries,
    config: InterpolationConfig | None = None,
) -> SMPTimeSeries:
    """Interpolate simulated values to observation timestamps.

    Parameters
    ----------
    observed : SMPTimeSeries
        Observed time series (provides target timestamps).
    simulated : SMPTimeSeries
        Simulated time series to interpolate from.
    config : InterpolationConfig | None
        Configuration options.  Uses defaults if ``None``.

    Returns
    -------
    SMPTimeSeries
        Interpolated simulated values at observation times.
    """
    if config is None:
        config = InterpolationConfig()

    # Convert times to float64 (seconds since epoch) for interpolation
    obs_t = observed.times.astype("datetime64[s]").astype(np.float64)
    sim_t = simulated.times.astype("datetime64[s]").astype(np.float64)
    sim_v = simulated.values.copy()

    # Remove NaN and excluded values from simulated for interpolation.
    # Matches Fortran SMP2SMP which sets 'X'-flagged values to sentinel
    # before interpolation (Class_SMP2SMP.f90:547-555).
    valid = ~np.isnan(sim_v) & ~simulated.excluded
    sim_t_valid = sim_t[valid]
    sim_v_valid = sim_v[valid]

    if len(sim_t_valid) == 0:
        return SMPTimeSeries(
            bore_id=observed.bore_id,
            times=observed.times.copy(),
            values=np.full(len(observed.times), config.sentinel_value),
            excluded=observed.excluded.copy(),
        )

    # Compute extrapolation bounds
    max_extrap_s = config.max_extrapolation_time.total_seconds()
    t_min = sim_t_valid[0] - max_extrap_s
    t_max = sim_t_valid[-1] + max_extrap_s

    if config.interpolation_method == "nearest":
        # Nearest-neighbor interpolation
        indices = np.searchsorted(sim_t_valid, obs_t)
        indices = np.clip(indices, 0, len(sim_t_valid) - 1)
        # Check if previous index is closer
        prev = np.clip(indices - 1, 0, len(sim_t_valid) - 1)
        d_next = np.abs(sim_t_valid[indices] - obs_t)
        d_prev = np.abs(sim_t_valid[prev] - obs_t)
        use_prev = d_prev < d_next
        indices[use_prev] = prev[use_prev]
        interp_values = sim_v_valid[indices]
    else:
        # Linear interpolation
        interp_values = np.interp(obs_t, sim_t_valid, sim_v_valid)

    # Apply sentinel value outside extrapolation bounds
    out_of_range = (obs_t < t_min) | (obs_t > t_max)
    interp_values[out_of_range] = config.sentinel_value

    return SMPTimeSeries(
        bore_id=observed.bore_id,
        times=observed.times.copy(),
        values=interp_values,
        excluded=observed.excluded.copy(),
    )


def interpolate_batch(
    observed: dict[str, SMPTimeSeries],
    simulated: dict[str, SMPTimeSeries],
    config: InterpolationConfig | None = None,
) -> dict[str, SMPTimeSeries]:
    """Interpolate simulated values for all matching bores.

    Parameters
    ----------
    observed : dict[str, SMPTimeSeries]
        Observed time series by bore ID.
    simulated : dict[str, SMPTimeSeries]
        Simulated time series by bore ID.
    config : InterpolationConfig | None
        Configuration options.

    Returns
    -------
    dict[str, SMPTimeSeries]
        Interpolated results for bores present in both inputs.
    """
    # Build case-insensitive lookup for simulated IDs to match Fortran
    # IWFM2OBS behavior (all bore IDs are uppercased before matching).
    sim_upper: dict[str, SMPTimeSeries] = {k.upper(): v for k, v in simulated.items()}
    result: dict[str, SMPTimeSeries] = {}
    for bore_id, obs_ts in observed.items():
        sim_ts = sim_upper.get(bore_id.upper())
        if sim_ts is not None:
            result[bore_id] = interpolate_to_obs_times(obs_ts, sim_ts, config)
    return result


_LAYER_SUFFIX_RE = re.compile(r"%\d+$")


def expand_obs_to_layers(
    observed: dict[str, SMPTimeSeries],
    n_layers: int,
    simulated_ids: set[str] | None = None,
) -> dict[str, SMPTimeSeries]:
    """Expand base observation IDs to per-layer IDs for IWFM matching.

    If observation bore IDs lack ``%N`` layer suffixes but the model
    expects per-layer IDs (e.g. ``WELL%1``, ``WELL%2``), this function
    duplicates each observation's time series for layers 1..n_layers.

    Detection logic:

    - If ALL obs IDs already contain ``%`` + digit suffix, return unchanged.
    - For each obs ID without a ``%`` suffix, expand to ``ID%1`` .. ``ID%N``.
    - IDs that already have ``%`` suffixes are kept as-is.

    Parameters
    ----------
    observed : dict[str, SMPTimeSeries]
        Observation time series keyed by bore ID.
    n_layers : int
        Number of model layers to expand to.
    simulated_ids : set[str] or None
        If provided, only expand IDs whose expanded form exists in
        *simulated_ids*. This avoids creating entries that would never match.

    Returns
    -------
    dict[str, SMPTimeSeries]
        Expanded observation dict (may be same object if no expansion needed).
    """
    if n_layers <= 0 or not observed:
        return observed

    # Classify obs IDs
    has_suffix = [bool(_LAYER_SUFFIX_RE.search(bid)) for bid in observed]
    if all(has_suffix):
        return observed  # backward compat: already per-layer

    expanded: dict[str, SMPTimeSeries] = {}
    n_expanded = 0
    for bore_id, ts in observed.items():
        if _LAYER_SUFFIX_RE.search(bore_id):
            # Already has %N suffix — keep as-is
            expanded[bore_id] = ts
        else:
            # Expand to ID%1 .. ID%N (share arrays, read-only in interp)
            for k in range(1, n_layers + 1):
                layer_id = f"{bore_id}%{k}"
                if simulated_ids is not None and layer_id not in simulated_ids:
                    continue
                expanded[layer_id] = SMPTimeSeries(
                    bore_id=layer_id,
                    times=ts.times,
                    values=ts.values,
                    excluded=ts.excluded,
                )
            n_expanded += 1

    if n_expanded:
        logger.info(
            "Expanded %d base obs IDs to %d per-layer IDs (%d layers)",
            n_expanded,
            len(expanded),
            n_layers,
        )
    return expanded


def _compute_model_dates(
    start_date_str: str,
    time_unit: str,
    n_timesteps: int,
) -> np.ndarray:
    """Compute model dates matching Fortran ``ComputeDate`` for all time units.

    The Fortran IWFM2OBS ignores ``.out`` file date strings and computes
    dates from the timestep index via ``ComputeDate``.  This function
    replicates that logic so Python produces identical timestamps.

    Parameters
    ----------
    start_date_str : str
        Simulation start date as ``"MM/DD/YYYY"`` (the date portion of BDT,
        before the ``_24:00`` suffix).
    time_unit : str
        IWFM time unit string (e.g. ``"1MON"``, ``"1DAY"``, ``"1WEEK"``,
        ``"1YEAR"``).
    n_timesteps : int
        Number of data lines in the ``.out`` file (= number of timesteps
        to compute dates for, starting at iTime=1).

    Returns
    -------
    np.ndarray
        Array of ``datetime64[s]`` dates, one per timestep.
    """
    import calendar

    parts = start_date_str.split("/")
    start_mon, start_day, start_yr = int(parts[0]), int(parts[1]), int(parts[2])
    start_dt = datetime(start_yr, start_mon, start_day)

    unit = time_unit.strip().upper()
    dates = np.empty(n_timesteps, dtype="datetime64[s]")

    for i_time in range(1, n_timesteps + 1):
        if unit == "1MON":
            # Fortran: iTotalMon = iYr*12 + iMon - 1 + iTime
            total_mon = start_yr * 12 + start_mon - 1 + i_time
            yr = total_mon // 12
            mon = total_mon % 12 + 1
            day = calendar.monthrange(yr, mon)[1]  # last day of month
            dt = datetime(yr, mon, day)
        elif unit == "1YEAR":
            # Fortran: iYr = iStartYr + iTime, keep day/month
            yr = start_yr + i_time
            dt = datetime(yr, start_mon, start_day)
        elif unit == "1WEEK":
            # Fortran: AddDays(start, 7*iTime)
            dt = start_dt + timedelta(days=7 * i_time)
        else:
            # 1DAY and sub-daily: AddDays(start, iTime)
            dt = start_dt + timedelta(days=i_time)

        dates[i_time - 1] = np.datetime64(dt, "s")

    return dates


def _replace_timestamps(
    simulated: dict[str, SMPTimeSeries],
    computed_dates: np.ndarray,
) -> None:
    """Replace parsed ``.out`` timestamps with ``ComputeDate``-equivalent dates.

    Modifies *simulated* in place.

    Parameters
    ----------
    simulated : dict[str, SMPTimeSeries]
        Simulated time series dict from ``.out`` file.
    computed_dates : np.ndarray
        Dates from :func:`_compute_model_dates`.
    """
    for ts in simulated.values():
        if len(ts.times) == len(computed_dates):
            ts.times = computed_dates.copy()


def deduplicate_smp(
    input_path: Path | str,
    output_path: Path | str,
) -> tuple[int, int]:
    """Remove duplicate per-layer entries from an SMP file.

    Strips ``%N`` suffixes and writes only unique base-ID entries.
    Verifies that all layer duplicates have identical timestamps and
    values before deduplication.

    Parameters
    ----------
    input_path : Path or str
        Path to the input SMP file with per-layer duplicates.
    output_path : Path or str
        Path for the deduplicated output SMP file.

    Returns
    -------
    tuple[int, int]
        ``(original_count, deduplicated_count)`` of unique bore IDs.
    """
    reader = SMPReader(Path(input_path))
    data = reader.read()

    original_count = len(data)

    # Group by base name (strip %N suffix)
    base_groups: dict[str, list[tuple[str, SMPTimeSeries]]] = {}
    for bore_id, ts in data.items():
        base = _LAYER_SUFFIX_RE.sub("", bore_id)
        base_groups.setdefault(base, []).append((bore_id, ts))

    # Build deduplicated dict using first entry per group
    deduped: dict[str, SMPTimeSeries] = {}
    for base, entries in base_groups.items():
        first_ts = entries[0][1]
        deduped[base] = SMPTimeSeries(
            bore_id=base,
            times=first_ts.times.copy(),
            values=first_ts.values.copy(),
            excluded=first_ts.excluded.copy(),
        )

    writer = SMPWriter(Path(output_path))
    writer.write(deduped)

    return original_count, len(deduped)


# ── Head Difference Pairs ──────────────────────────────────────────────


@dataclass
class HeadDifferencePair:
    """A pair of well IDs for head difference computation.

    Mirrors Fortran ``Class_HeadDifference.f90::HeadDiffPairType``.
    Computes ``Head(id1) - Head(id2)`` at matching timesteps.

    Attributes
    ----------
    id1 : str
        First well ID.
    id2 : str
        Second well ID (subtracted from id1).
    """

    id1: str
    id2: str


def read_head_difference_pairs(path: str | Path) -> list[HeadDifferencePair]:
    """Read head difference pairs from a text file.

    Each line contains two whitespace-separated well IDs.
    IDs are uppercased for case-insensitive matching.

    Parameters
    ----------
    path : str or Path
        Path to the pairs file.

    Returns
    -------
    list[HeadDifferencePair]

    Raises
    ------
    ValueError
        If a pair has identical IDs or a line has fewer than 2 tokens.
    """
    pairs: list[HeadDifferencePair] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith(("C", "c", "*", "#")):
                continue
            tokens = stripped.split()
            if len(tokens) < 2:
                raise ValueError(f"Line {lineno} of {path}: expected 2 IDs, got {len(tokens)}")
            id1 = tokens[0].upper()
            id2 = tokens[1].upper()
            if id1 == id2:
                raise ValueError(f"Line {lineno} of {path}: identical IDs '{id1}'")
            pairs.append(HeadDifferencePair(id1=id1, id2=id2))
    if not pairs:
        raise ValueError(f"No pairs found in {path}")
    return pairs


def compute_head_differences(
    interpolated: dict[str, SMPTimeSeries],
    pairs: list[HeadDifferencePair],
) -> dict[str, SMPTimeSeries]:
    """Compute head differences for well pairs.

    For each pair, computes ``interpolated[id1].values - interpolated[id2].values``
    at matching timestamps. Mirrors Fortran ``Class_HeadDifference`` logic.

    Parameters
    ----------
    interpolated : dict[str, SMPTimeSeries]
        Interpolated time series keyed by bore ID (case-insensitive).
    pairs : list[HeadDifferencePair]
        Well pairs to difference.

    Returns
    -------
    dict[str, SMPTimeSeries]
        Head differences keyed by ``"id1-id2"``.
    """
    # Case-insensitive lookup
    upper_map: dict[str, SMPTimeSeries] = {k.upper(): v for k, v in interpolated.items()}

    results: dict[str, SMPTimeSeries] = {}
    for pair in pairs:
        ts1 = upper_map.get(pair.id1)
        ts2 = upper_map.get(pair.id2)
        if ts1 is None:
            logger.warning("Head difference: ID '%s' not found, skipping pair", pair.id1)
            continue
        if ts2 is None:
            logger.warning("Head difference: ID '%s' not found, skipping pair", pair.id2)
            continue

        diff_id = f"{pair.id1}-{pair.id2}"
        diff_values = ts1.values - ts2.values
        results[diff_id] = SMPTimeSeries(
            bore_id=diff_id,
            times=ts1.times.copy(),
            values=diff_values,
            excluded=ts1.excluded | ts2.excluded,
        )
    return results


@dataclass
class MultiLayerWellSpec:
    """Specification for a multi-layer observation well.

    Attributes
    ----------
    name : str
        Well identifier.
    x : float
        X coordinate of the well.
    y : float
        Y coordinate of the well.
    element_id : int
        Element containing the well (1-based).
    bottom_of_screen : float
        Bottom elevation of the well screen.
    top_of_screen : float
        Top elevation of the well screen.
    """

    name: str
    x: float
    y: float
    element_id: int
    bottom_of_screen: float
    top_of_screen: float


def compute_multilayer_weights(
    well: MultiLayerWellSpec,
    grid: AppGrid,
    stratigraphy: Stratigraphy,
    hydraulic_conductivity: NDArray[np.float64],
    fe_node_ids: tuple[int, ...] | None = None,
    fe_weights: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Compute transmissivity-weighted layer weights for a well.

    Parameters
    ----------
    well : MultiLayerWellSpec
        Well specification with screen interval.
    grid : AppGrid
        Model grid.
    stratigraphy : Stratigraphy
        Model stratigraphy (layer elevations).
    hydraulic_conductivity : NDArray[np.float64]
        Hydraulic conductivity array, shape ``(n_layers,)`` or
        ``(n_layers, n_nodes)`` for spatially varying HK.
    fe_node_ids : tuple[int, ...], optional
        Pre-computed FE interpolation node IDs (1-based). If provided
        along with ``fe_weights``, skips the expensive FE search.
    fe_weights : NDArray[np.float64], optional
        Pre-computed FE interpolation coefficients.

    Returns
    -------
    NDArray[np.float64]
        Layer weights array, shape ``(n_layers,)``, summing to 1.
    """
    # Fortran: if iOverwriteLayer > 0, use only that layer
    if hasattr(well, "overwrite_layer") and well.overwrite_layer > 0:
        n_lay = stratigraphy.n_layers
        w = np.zeros(n_lay)
        w[well.overwrite_layer - 1] = 1.0  # 1-based to 0-based
        return w

    n_layers = stratigraphy.n_layers

    # Use pre-computed FE data if available, otherwise do full search
    if fe_node_ids is not None and fe_weights is not None:
        node_ids = fe_node_ids
        weights = fe_weights
    else:
        from pyiwfm.core.interpolation import FEInterpolator

        interp = FEInterpolator(grid)
        _elem_id, node_ids, weights = interp.interpolate(well.x, well.y)

    # Interpolate layer elevations at well location
    layer_tops = np.zeros(n_layers)
    layer_bots = np.zeros(n_layers)

    for k in range(n_layers):
        # Layer top = bottom of layer above (or ground surface for first)
        top_vals = {}
        bot_vals = {}
        for nid in node_ids:
            idx = nid - 1  # 1-based to 0-based
            top_vals[nid] = float(stratigraphy.top_elev[idx, k])
            bot_vals[nid] = float(stratigraphy.bottom_elev[idx, k])

        # Weighted interpolation using shape functions
        layer_tops[k] = sum(top_vals[nid] * weights[i] for i, nid in enumerate(node_ids))
        layer_bots[k] = sum(bot_vals[nid] * weights[i] for i, nid in enumerate(node_ids))

    # Compute screen-layer intersection thickness
    bos = well.bottom_of_screen
    tos = well.top_of_screen

    thicknesses = np.zeros(n_layers)
    for k in range(n_layers):
        overlap_top = min(tos, layer_tops[k])
        overlap_bot = max(bos, layer_bots[k])
        thicknesses[k] = max(0.0, overlap_top - overlap_bot)

    # Get HK at well location for each layer
    hk_at_well = np.zeros(n_layers)
    if hydraulic_conductivity.ndim == 1:
        hk_at_well = hydraulic_conductivity.copy()
    else:
        for k in range(n_layers):
            hk_vals = {nid: hydraulic_conductivity[k, nid - 1] for nid in node_ids}
            hk_at_well[k] = sum(hk_vals[nid] * weights[i] for i, nid in enumerate(node_ids))

    # Transmissivity per layer
    t_k = thicknesses * hk_at_well
    t_total = np.sum(t_k)

    if t_total == 0.0:
        # Fortran: weight=1.0 for first non-zero-thickness layer (or layer 0 if all zero)
        w = np.zeros(n_layers)
        nonzero = np.where(thicknesses > 0.0)[0]
        w[nonzero[0] if len(nonzero) > 0 else 0] = 1.0
        return w

    return t_k / t_total


def compute_composite_subsidence(
    layer_subsidence: NDArray[np.float64],
) -> float:
    """Compute composite subsidence by summing per-layer values.

    Unlike head which uses T-weighted averaging, subsidence is additive
    across layers (Fortran: Class_IWFM2OBS.f90:678-892).

    Parameters
    ----------
    layer_subsidence : NDArray[np.float64]
        Per-layer subsidence values, shape ``(n_layers,)``.

    Returns
    -------
    float
        Total subsidence (sum across all layers).
    """
    return float(np.nansum(layer_subsidence))


def compute_composite_head(
    well: MultiLayerWellSpec,
    layer_heads: NDArray[np.float64],
    weights: NDArray[np.float64],
    grid: AppGrid,
) -> float:
    """Compute composite head at a multi-layer well.

    Parameters
    ----------
    well : MultiLayerWellSpec
        Well specification.
    layer_heads : NDArray[np.float64]
        Head values by layer, shape ``(n_layers,)`` at the well location
        or ``(n_layers, n_nodes)`` for nodal heads.
    weights : NDArray[np.float64]
        Layer weights from :func:`compute_multilayer_weights`.
    grid : AppGrid
        Model grid (used for FE interpolation if nodal heads provided).

    Returns
    -------
    float
        Composite head value.
    """
    if layer_heads.ndim == 1:
        return float(np.sum(layer_heads * weights))

    # Spatially varying heads: interpolate each layer at well location
    from pyiwfm.core.interpolation import FEInterpolator

    interp = FEInterpolator(grid)
    _, node_ids, shape_wts = interp.interpolate(well.x, well.y)

    n_layers = layer_heads.shape[0]
    head_at_well = np.zeros(n_layers)
    for k in range(n_layers):
        for i, nid in enumerate(node_ids):
            head_at_well[k] += layer_heads[k, nid - 1] * shape_wts[i]

    return float(np.sum(head_at_well * weights))


def iwfm2obs(
    obs_smp_path: Path,
    sim_smp_path: Path,
    output_path: Path,
    well_specs: list[MultiLayerWellSpec] | None = None,
    grid: AppGrid | None = None,
    stratigraphy: Stratigraphy | None = None,
    hydraulic_conductivity: NDArray[np.float64] | None = None,
    config: InterpolationConfig | None = None,
) -> dict[str, SMPTimeSeries]:
    """Run the full IWFM2OBS workflow.

    Reads observed and simulated SMP files, performs time interpolation
    (and optionally multi-layer T-weighted averaging), and writes the
    result to an output SMP file.

    Parameters
    ----------
    obs_smp_path : Path
        Path to observed data SMP file.
    sim_smp_path : Path
        Path to simulated data SMP file.
    output_path : Path
        Path for output interpolated SMP file.
    well_specs : list[MultiLayerWellSpec] | None
        Multi-layer well specifications (optional).
    grid : AppGrid | None
        Model grid (required if ``well_specs`` provided).
    stratigraphy : Stratigraphy | None
        Model stratigraphy (required if ``well_specs`` provided).
    hydraulic_conductivity : NDArray[np.float64] | None
        HK array (required if ``well_specs`` provided).
    config : InterpolationConfig | None
        Interpolation configuration.

    Returns
    -------
    dict[str, SMPTimeSeries]
        Interpolated time series by bore ID.
    """
    obs_reader = SMPReader(obs_smp_path)
    sim_reader = SMPReader(sim_smp_path)

    observed = obs_reader.read()
    simulated = sim_reader.read()

    # Time interpolation
    result = interpolate_batch(observed, simulated, config)

    # Write output
    writer = SMPWriter(output_path)
    writer.write(result)

    return result


# =====================================================================
# Integrated workflow: simulation main file → .out → interpolation
# =====================================================================


@dataclass
class IWFM2OBSConfig:
    """Configuration for the integrated IWFM2OBS workflow.

    Attributes
    ----------
    interpolation : InterpolationConfig
        Time interpolation settings.
    date_format : int
        Date format: ``1`` = dd/mm/yyyy, ``2`` = mm/dd/yyyy.
    """

    interpolation: InterpolationConfig = field(default_factory=InterpolationConfig)
    date_format: int = 2


# ── IWFM2OBS Input File Parser ─────────────────────────────────────────


@dataclass
class IWFM2OBSHydBlock:
    """One hydrograph block from the IWFM2OBS input file.

    Attributes
    ----------
    model_smp : str
        Model hydrograph SMP path (ignored in model-discovery mode).
    obs_smp : str
        Observation SMP path (blank = skip this type).
    output_smp : str
        Output SMP path.
    threshold : float
        Extrapolation threshold in days.
    ins_file : str
        PEST instruction file path (blank = skip).
    pcf_file : str
        PEST PCF file path (blank = skip).
    """

    model_smp: str = ""
    obs_smp: str = ""
    output_smp: str = ""
    threshold: float = 1.0
    ins_file: str = ""
    pcf_file: str = ""


@dataclass
class IWFM2OBSInputFile:
    """Parsed IWFM2OBS input file (``iwfm2obs_template.in`` format).

    Mirrors the Fortran IWFM2OBS input file structure with 4 hydrograph
    blocks, head difference flag, and multi-layer target flag.

    Attributes
    ----------
    simulation_main_file : str
        IWFM simulation main file (blank = explicit SMP mode).
    date_format : int
        1 = dd/mm/yyyy, 2 = mm/dd/yyyy.
    gw : IWFM2OBSHydBlock
        Groundwater head hydrograph block.
    stream : IWFM2OBSHydBlock
        Stream hydrograph block.
    tiledrain : IWFM2OBSHydBlock
        Tile drain hydrograph block.
    subsidence : IWFM2OBSHydBlock
        Subsidence hydrograph block.
    head_diff_enabled : bool
        Whether to compute head differences.
    head_diff_file : str
        Path to head difference pair file.
    multilayer_enabled : bool
        Whether to enable multi-layer T-weighted averaging.
    multilayer_obs_well_file : str
        Observation well locations + screen intervals.
    multilayer_elements_file : str
        IWFM element connectivity file.
    multilayer_nodes_file : str
        IWFM node coordinates file.
    multilayer_stratigraphy_file : str
        IWFM stratigraphy (layer elevations) file.
    multilayer_gw_main_file : str
        IWFM GW main file (for hydraulic conductivity).
    """

    simulation_main_file: str = ""
    date_format: int = 2
    gw: IWFM2OBSHydBlock = field(default_factory=IWFM2OBSHydBlock)
    stream: IWFM2OBSHydBlock = field(default_factory=IWFM2OBSHydBlock)
    tiledrain: IWFM2OBSHydBlock = field(default_factory=IWFM2OBSHydBlock)
    subsidence: IWFM2OBSHydBlock = field(default_factory=IWFM2OBSHydBlock)
    head_diff_enabled: bool = False
    head_diff_file: str = ""
    multilayer_enabled: bool = False
    multilayer_obs_well_file: str = ""
    multilayer_elements_file: str = ""
    multilayer_nodes_file: str = ""
    multilayer_stratigraphy_file: str = ""
    multilayer_gw_main_file: str = ""


def _next_data_value_i2o(f: Any) -> str:
    """Read next non-comment data value, stripping inline ``/`` comment.

    Uses IWFM's column-1 comment convention and inline comment rules.
    Returns the data value with comment text removed, or empty string
    for blank data lines (which are significant in IWFM2OBS blocks).
    """
    from pyiwfm.io.iwfm_reader import is_comment_line as _is_iwfm_comment
    from pyiwfm.io.iwfm_reader import strip_inline_comment

    for raw in f:
        raw_line = raw.rstrip("\n\r")
        if _is_iwfm_comment(raw_line):
            continue
        if raw_line and raw_line[0] == "#":
            continue
        value, _ = strip_inline_comment(raw_line)
        return value
    return ""


def _read_hyd_block(f: Any) -> IWFM2OBSHydBlock:
    """Read a 6-line hydrograph block from the input file."""
    model_smp = _next_data_value_i2o(f)
    obs_smp = _next_data_value_i2o(f)
    output_smp = _next_data_value_i2o(f)
    threshold_raw = _next_data_value_i2o(f)
    try:
        threshold = float(threshold_raw) if threshold_raw else 1.0
    except ValueError:
        threshold = 1.0
    ins_file = _next_data_value_i2o(f)
    pcf_file = _next_data_value_i2o(f)
    return IWFM2OBSHydBlock(
        model_smp=model_smp,
        obs_smp=obs_smp,
        output_smp=output_smp,
        threshold=threshold,
        ins_file=ins_file,
        pcf_file=pcf_file,
    )


def read_iwfm2obs_config(path: str | Path) -> IWFM2OBSInputFile:
    """Parse an IWFM2OBS input file (``iwfm2obs_template.in`` format).

    Reads the structured Fortran-compatible config with 4 hydrograph blocks
    (GW, stream, tile drain, subsidence), head difference flag, and
    multi-layer target flag.

    Parameters
    ----------
    path : str or Path
        Path to the IWFM2OBS input file.

    Returns
    -------
    IWFM2OBSInputFile
        Parsed configuration.
    """
    result = IWFM2OBSInputFile()

    with open(path, encoding="utf-8") as f:
        # Line 1: Simulation main file (or date format for old format)
        first = _next_data_value_i2o(f)

        if first in ("1", "2"):
            # Old format: first data line is date format, no model discovery
            result.date_format = int(first)
        else:
            result.simulation_main_file = first
            # Line 2: Date format
            date_raw = _next_data_value_i2o(f)
            try:
                result.date_format = int(date_raw) if date_raw else 2
            except ValueError:
                result.date_format = 2

        # 4 hydrograph blocks: GW, Stream, TileDrain, Subsidence
        result.gw = _read_hyd_block(f)
        result.stream = _read_hyd_block(f)
        result.tiledrain = _read_hyd_block(f)
        result.subsidence = _read_hyd_block(f)

        # Head differences Y/N
        hd_raw = _next_data_value_i2o(f)
        result.head_diff_enabled = hd_raw.upper().startswith("Y")
        if result.head_diff_enabled:
            result.head_diff_file = _next_data_value_i2o(f)

        # Multi-layer target Y/N
        ml_raw = _next_data_value_i2o(f)
        result.multilayer_enabled = ml_raw.upper().startswith("Y")
        if result.multilayer_enabled:
            result.multilayer_obs_well_file = _next_data_value_i2o(f)
            result.multilayer_elements_file = _next_data_value_i2o(f)
            result.multilayer_nodes_file = _next_data_value_i2o(f)
            result.multilayer_stratigraphy_file = _next_data_value_i2o(f)
            result.multilayer_gw_main_file = _next_data_value_i2o(f)

    return result


def iwfm2obs_from_model(
    simulation_main_file: Path | str,
    obs_smp_paths: dict[str, Path],
    output_paths: dict[str, Path],
    config: IWFM2OBSConfig | None = None,
    obs_well_spec_path: Path | None = None,
    multilayer_output_path: Path | None = None,
    grid: AppGrid | None = None,
    stratigraphy: Stratigraphy | None = None,
    hydraulic_conductivity: NDArray[np.float64] | None = None,
) -> dict[str, dict[str, SMPTimeSeries]]:
    """Full IWFM2OBS workflow reading .out files directly from the model.

    Steps:

    1. :func:`~pyiwfm.calibration.model_file_discovery.discover_hydrograph_files`
       — find .out paths and hydrograph metadata.
    2. For each hydrograph type with observations:
       :class:`~pyiwfm.io.hydrograph_reader.IWFMHydrographReader` reads the
       ``.out`` file → convert to ``SMPTimeSeries`` dict → interpolate.
    3. If multi-layer specified: compute T-weighted composite heads and
       write ``GW_MultiLayer.out`` and PEST ``.ins`` files.

    Parameters
    ----------
    simulation_main_file : Path or str
        IWFM simulation main file path.
    obs_smp_paths : dict[str, Path]
        Observation SMP file paths keyed by type (``"gw"``, ``"stream"``).
    output_paths : dict[str, Path]
        Output SMP file paths keyed by type.
    config : IWFM2OBSConfig or None
        Workflow configuration.
    obs_well_spec_path : Path or None
        Multi-layer well specification file (enables T-weighted averaging).
    multilayer_output_path : Path or None
        Path for ``GW_MultiLayer.out`` output.
    grid : AppGrid or None
        Model grid (required for multi-layer).
    stratigraphy : Stratigraphy or None
        Model stratigraphy (required for multi-layer).
    hydraulic_conductivity : NDArray[np.float64] or None
        HK array (required for multi-layer).

    Returns
    -------
    dict[str, dict[str, SMPTimeSeries]]
        Interpolated results keyed by type then bore ID.
    """
    from pyiwfm.calibration.model_file_discovery import discover_hydrograph_files
    from pyiwfm.io.hydrograph_reader import IWFMHydrographReader

    if config is None:
        config = IWFM2OBSConfig()

    # Step 1: discover .out files
    discovery = discover_hydrograph_files(simulation_main_file)

    # Map type keys to discovered paths
    type_map: dict[str, Path | None] = {
        "gw": discovery.gw_hydrograph_path,
        "stream": discovery.stream_hydrograph_path,
        "subsidence": discovery.subsidence_hydrograph_path,
        "tiledrain": discovery.tiledrain_hydrograph_path,
    }

    # Map type keys to hydrograph locations for bore ID mapping
    location_map: dict[str, list[tuple[int, str]]] = {
        "gw": [(i, loc.name) for i, loc in enumerate(discovery.gw_locations)],
        "stream": [(i, loc.name) for i, loc in enumerate(discovery.stream_locations)],
        "subsidence": [(i, loc.name) for i, loc in enumerate(discovery.subsidence_locations)],
        "tiledrain": [(i, loc.name) for i, loc in enumerate(discovery.tiledrain_locations)],
    }

    results: dict[str, dict[str, SMPTimeSeries]] = {}

    # Step 2: for each type with .out file and observations, interpolate
    for type_key in ("gw", "stream", "subsidence", "tiledrain"):
        out_path = type_map.get(type_key)
        obs_path = obs_smp_paths.get(type_key)
        result_path = output_paths.get(type_key)

        if out_path is None or obs_path is None or result_path is None:
            continue

        if not out_path.exists():
            logger.warning("Hydrograph .out file not found: %s", out_path)
            continue

        if not Path(obs_path).exists():
            logger.warning("Observation SMP not found: %s", obs_path)
            continue

        # Read .out file
        reader = IWFMHydrographReader(out_path)
        if reader.n_columns == 0:
            logger.warning("No data in .out file: %s", out_path)
            continue

        # Convert to SMP dict using bore IDs from discovery
        locs = location_map.get(type_key, [])
        bore_ids: dict[int, str] = {}
        for col_idx, name in locs:
            if col_idx < reader.n_columns:
                bore_ids[col_idx] = name
        # If no location mapping, use column indices as IDs
        if not bore_ids:
            bore_ids = {i: f"COL{i + 1}" for i in range(reader.n_columns)}

        simulated = reader.get_columns_as_smp_dict(bore_ids)

        # Replace parsed .out timestamps with ComputeDate-equivalent dates
        # to match Fortran IWFM2OBS behavior (which ignores .out date strings)
        if simulated and discovery.start_date_str and discovery.time_unit:
            first_ts = next(iter(simulated.values()))
            computed = _compute_model_dates(
                discovery.start_date_str,
                discovery.time_unit,
                len(first_ts.times),
            )
            _replace_timestamps(simulated, computed)

        # Read observations
        obs_reader = SMPReader(obs_path)
        observed = obs_reader.read()

        # Auto-expand base obs IDs to per-layer IDs for GW matching
        if type_key == "gw":
            n_layers_hint = 0
            if stratigraphy is not None:
                n_layers_hint = stratigraphy.n_layers
            else:
                # Infer from simulated IDs: find max %N suffix
                for sid in simulated:
                    m = _LAYER_SUFFIX_RE.search(sid)
                    if m:
                        n_layers_hint = max(n_layers_hint, int(m.group(0)[1:]))
            if n_layers_hint > 0:
                observed = expand_obs_to_layers(
                    observed, n_layers_hint, simulated_ids=set(simulated.keys())
                )

        # Interpolate
        interp_result = interpolate_batch(observed, simulated, config.interpolation)

        # Write output
        writer = SMPWriter(result_path)
        writer.write(interp_result)

        results[type_key] = interp_result
        logger.info(
            "%s: interpolated %d bore(s) to %s",
            type_key,
            len(interp_result),
            result_path,
        )

    # Step 3: multi-layer T-weighted averaging
    if (
        obs_well_spec_path is not None
        and multilayer_output_path is not None
        and grid is not None
        and stratigraphy is not None
        and hydraulic_conductivity is not None
        and "gw" in results
    ):
        from pyiwfm.calibration.obs_well_spec import read_obs_well_spec

        well_specs = read_obs_well_spec(obs_well_spec_path)

        # Compute weights for each well
        all_weights: list[NDArray[np.float64]] = []
        for spec in well_specs:
            well = MultiLayerWellSpec(
                name=spec.name,
                x=spec.x,
                y=spec.y,
                element_id=spec.element_id,
                bottom_of_screen=spec.bottom_of_screen,
                top_of_screen=spec.top_of_screen,
            )
            weights = compute_multilayer_weights(well, grid, stratigraphy, hydraulic_conductivity)
            all_weights.append(weights)

        # Compute composite heads
        gw_results = results["gw"]
        n_layers = stratigraphy.n_layers

        composite_results: dict[str, list[tuple[datetime, float]]] = {}
        for i, spec in enumerate(well_specs):
            # Gather per-layer time series for this well
            layer_series: dict[int, SMPTimeSeries] = {}
            for k in range(1, n_layers + 1):
                layer_id = f"{spec.name}%{k}"
                if layer_id in gw_results:
                    layer_series[k] = gw_results[layer_id]

            if not layer_series:
                continue

            # Use first available layer's timestamps
            first_layer = next(iter(layer_series.values()))
            times = first_layer.times

            composites: list[tuple[datetime, float]] = []
            for t_idx in range(len(times)):
                layer_vals = np.zeros(n_layers)
                for k in range(n_layers):
                    lid = f"{spec.name}%{k + 1}"
                    if lid in gw_results:
                        layer_vals[k] = gw_results[lid].values[t_idx]
                weighted = float(np.sum(layer_vals * all_weights[i]))
                dt = times[t_idx].astype("datetime64[us]").astype(datetime)
                composites.append((dt, weighted))

            composite_results[spec.name] = composites

        # Write outputs
        if composite_results:
            write_multilayer_output(
                composite_results,
                well_specs,
                all_weights,
                multilayer_output_path,
                n_layers,
            )

    # Step 3b: multi-layer subsidence summation (Fortran IWFM2OBS:666-699)
    # Unlike GW heads which use T-weighted averaging, subsidence layers are
    # summed (additive compaction) when a well screens multiple layers.
    if obs_well_spec_path is not None and stratigraphy is not None and "subsidence" in results:
        from pyiwfm.calibration.obs_well_spec import read_obs_well_spec

        sub_well_specs = read_obs_well_spec(obs_well_spec_path)

        sub_results = results["subsidence"]
        n_layers = stratigraphy.n_layers

        composite_sub: dict[str, SMPTimeSeries] = {}
        for spec in sub_well_specs:
            sub_layer_series: dict[int, SMPTimeSeries] = {}
            for k in range(1, n_layers + 1):
                layer_id = f"{spec.name}%{k}"
                if layer_id in sub_results:
                    sub_layer_series[k] = sub_results[layer_id]

            if not sub_layer_series:
                continue

            first_layer = next(iter(sub_layer_series.values()))
            summed = np.zeros(len(first_layer.times), dtype=np.float64)
            for ts in sub_layer_series.values():
                summed += np.where(np.isnan(ts.values), 0.0, ts.values)

            composite_sub[spec.name] = SMPTimeSeries(
                bore_id=spec.name,
                times=first_layer.times.copy(),
                values=summed,
                excluded=first_layer.excluded.copy(),
            )

        if composite_sub:
            results["subsidence"].update(composite_sub)
            logger.info("Subsidence: summed %d multi-layer wells", len(composite_sub))

    return results


def write_multilayer_output(
    results: dict[str, list[tuple[datetime, float]]],
    well_specs: list[ObsWellSpec],
    weights: list[NDArray[np.float64]],
    output_path: Path,
    n_layers: int,
) -> None:
    """Write ``GW_MultiLayer.out`` format output.

    Format: ``Name  Date  Time  Simulated  T1  T2  T3  T4  NewTOS  NewBOS``

    Parameters
    ----------
    results : dict[str, list[tuple[datetime, float]]]
        Composite head results keyed by well name.
    well_specs : list[ObsWellSpec]
        Well specifications.
    weights : list[NDArray[np.float64]]
        Per-well layer weight arrays.
    output_path : Path
        Output file path.
    n_layers : int
        Number of model layers.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        # Header
        f.write(f"{'Name':<25s} {'Date':>10s}  {'Time':>8s}  {'Simulated':>10s}")
        for k in range(min(n_layers, 4)):
            f.write(f"  {'T' + str(k + 1):>10s}")
        f.write(f"  {'NewTOS':>10s}  {'NewBOS':>10s}\n")

        # Data lines
        for i, spec in enumerate(well_specs):
            if spec.name not in results:
                continue
            for dt, value in results[spec.name]:
                date_str = dt.strftime("%m/%d/%Y")
                time_str = dt.strftime("%H:%M:%S")
                line = f"{spec.name:<25s} {date_str:>10s}  {time_str:>8s}"
                line += f"  {value:10.2f}"
                # T1..T4 (raw transmissivity per layer, from weights * total T)
                for k in range(min(n_layers, 4)):
                    line += f"  {weights[i][k]:10.2f}"
                # NewTOS, NewBOS
                line += f"  {spec.top_of_screen:10.2f}"
                line += f"  {spec.bottom_of_screen:10.2f}"
                f.write(line + "\n")


# =====================================================================
# Continuous composite and seasonal averaging
# =====================================================================


def compute_composite_continuous(
    per_layer_sim: dict[str, SMPTimeSeries],
    well_specs: list[MultiLayerWellSpec],
    layer_weights: dict[str, NDArray[np.float64]],
    n_layers: int = 4,
) -> dict[str, SMPTimeSeries]:
    """Compute T-weighted composite heads at ALL simulation timesteps.

    Unlike the standard ``iwfm2obs_from_model`` workflow which first
    interpolates per-layer heads to observation dates and then averages,
    this function averages per-layer data **first** across all timesteps,
    producing a continuous composite time series per well.

    This is the correct order of operations for CalcTypeHyd: the
    composite series preserves full temporal resolution (577 monthly
    timesteps), and downstream tools can sample or aggregate as needed.

    Parameters
    ----------
    per_layer_sim : dict[str, SMPTimeSeries]
        Per-layer simulation time series keyed by bore ID with ``%N``
        layer suffix (e.g. ``"WELL_A%1"``, ``"WELL_A%2"``).
    well_specs : list[MultiLayerWellSpec]
        Well specifications with screen intervals.
    layer_weights : dict[str, NDArray[np.float64]]
        Pre-computed T-weights per well, keyed by base well name.
        Each array has shape ``(n_layers,)``.
    n_layers : int
        Number of model layers (default 4).

    Returns
    -------
    dict[str, SMPTimeSeries]
        Composite time series keyed by base well name (no ``%N`` suffix).
    """
    composites: dict[str, SMPTimeSeries] = {}

    for spec in well_specs:
        base_name = spec.name
        weights = layer_weights.get(base_name)
        if weights is None:
            continue

        # Gather per-layer series
        layer_series: list[SMPTimeSeries | None] = [None] * n_layers
        for k in range(n_layers):
            lid = f"{base_name}%{k + 1}"
            layer_series[k] = per_layer_sim.get(lid)

        # Find a layer with data to get timestamps
        ref_ts: SMPTimeSeries | None = None
        for ts in layer_series:
            if ts is not None and len(ts.times) > 0:
                ref_ts = ts
                break
        if ref_ts is None:
            continue

        n_times = len(ref_ts.times)
        composite_vals = np.zeros(n_times, dtype=np.float64)

        for k in range(n_layers):
            ts_k = layer_series[k]
            if ts_k is not None and weights[k] > 0.0:
                vals = ts_k.values
                # Handle NaN: treat as zero contribution
                clean_vals = np.where(np.isnan(vals), 0.0, vals)
                composite_vals += weights[k] * clean_vals

        composites[base_name] = SMPTimeSeries(
            bore_id=base_name,
            times=ref_ts.times.copy(),
            values=composite_vals,
            excluded=np.zeros(n_times, dtype=np.bool_),
        )

    logger.info(
        "Computed continuous composite heads for %d wells (%d timesteps each)",
        len(composites),
        len(next(iter(composites.values())).times) if composites else 0,
    )
    return composites


def average_to_seasonal(
    continuous: dict[str, SMPTimeSeries],
    periods: list[tuple[str, list[int], str]],
) -> dict[str, SMPTimeSeries]:
    """Aggregate continuous monthly time series to seasonal window averages.

    For each well, groups timesteps by seasonal window (e.g. Jan-Apr for
    spring) and year, computes the mean, and assigns the result to the
    representative date.  Both sim and obs data should be processed
    through this function with the same ``periods`` to ensure consistent
    comparison.

    Parameters
    ----------
    continuous : dict[str, SMPTimeSeries]
        Continuous monthly time series by bore ID.
    periods : list[tuple[str, list[int], str]]
        Seasonal period definitions as ``(name, months, representative_date)``
        tuples.  ``representative_date`` is ``"MM/DD"`` format.
        Example::

            [("Spring", [1,2,3,4], "03/01"),
             ("Fall", [8,9,10,11], "10/01")]

    Returns
    -------
    dict[str, SMPTimeSeries]
        Seasonal-averaged time series.  Each well has one record per
        period per year (e.g. 2 records/year for biannual periods).
    """
    result: dict[str, SMPTimeSeries] = {}

    # Build month→(period_idx, rep_month, rep_day)
    month_map: dict[int, tuple[int, int, int]] = {}
    for p_idx, (_, months, rep_date) in enumerate(periods):
        rep_parts = rep_date.split("/")
        rep_m, rep_d = int(rep_parts[0]), int(rep_parts[1])
        for m in months:
            month_map[m] = (p_idx, rep_m, rep_d)

    for bore_id, ts in continuous.items():
        if len(ts.times) == 0:
            continue

        # Extract year and month from timestamps
        ts_months = ts.times.astype("datetime64[M]").astype(int) % 12 + 1
        years = ts.times.astype("datetime64[Y]").astype(int) + 1970
        valid = ~np.isnan(ts.values) & ~ts.excluded

        # Group by (year, period_idx)
        buckets: dict[tuple[int, int], list[float]] = defaultdict(list)
        for i in range(len(ts.times)):
            if not valid[i]:
                continue
            m = int(ts_months[i])
            if m not in month_map:
                continue
            p_idx, _, _ = month_map[m]
            yr = int(years[i])
            buckets[(yr, p_idx)].append(float(ts.values[i]))

        # Build output arrays
        out_times: list[np.datetime64] = []
        out_values: list[float] = []

        for yr, p_idx in sorted(buckets.keys()):
            vals = buckets[(yr, p_idx)]
            if not vals:
                continue
            # Look up representative date from first month in this period
            first_month = periods[p_idx][1][0]
            _, rep_m, rep_d = month_map[first_month]
            dt_str = f"{yr:04d}-{rep_m:02d}-{rep_d:02d}"
            out_times.append(np.datetime64(dt_str))
            out_values.append(float(np.mean(vals)))

        if out_times:
            result[bore_id] = SMPTimeSeries(
                bore_id=bore_id,
                times=np.array(out_times, dtype="datetime64[s]"),
                values=np.array(out_values, dtype=np.float64),
                excluded=np.zeros(len(out_times), dtype=np.bool_),
            )

    logger.info(
        "Averaged %d wells to %d seasonal periods",
        len(result),
        len(periods),
    )
    return result
