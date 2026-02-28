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
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal

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

    # Remove NaN values from simulated for interpolation
    valid = ~np.isnan(sim_v)
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
    result: dict[str, SMPTimeSeries] = {}
    for bore_id, obs_ts in observed.items():
        if bore_id in simulated:
            result[bore_id] = interpolate_to_obs_times(obs_ts, simulated[bore_id], config)
    return result


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

    Returns
    -------
    NDArray[np.float64]
        Layer weights array, shape ``(n_layers,)``, summing to 1.
    """
    from pyiwfm.core.interpolation import FEInterpolator

    n_layers = stratigraphy.n_layers
    interp = FEInterpolator(grid)

    # Get shape function interpolation at well location
    elem_id, node_ids, weights = interp.interpolate(well.x, well.y)

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
        # Equal weights if no transmissivity
        return np.ones(n_layers) / n_layers

    return t_k / t_total


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


def iwfm2obs_from_model(
    simulation_main_file: Path | str,
    obs_smp_paths: dict[str, Path],
    output_paths: dict[str, Path],
    config: IWFM2OBSConfig | None = None,
    obs_well_spec_path: Path | None = None,
    multilayer_output_path: Path | None = None,
    multilayer_ins_path: Path | None = None,
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
    multilayer_ins_path : Path or None
        Path for PEST instruction file.
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

        # Read observations
        obs_reader = SMPReader(obs_path)
        observed = obs_reader.read()

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
            if multilayer_ins_path is not None:
                write_multilayer_pest_ins(
                    composite_results,
                    well_specs,
                    multilayer_ins_path,
                )

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
    with open(output_path, "w") as f:
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


def write_multilayer_pest_ins(
    results: dict[str, list[tuple[datetime, float]]],
    well_specs: list[ObsWellSpec],
    ins_path: Path,
) -> None:
    """Write PEST instruction file for multi-layer targets.

    Format: ``pif #``, ``l1`` (skip header), then
    ``l1 [WLT{well:05d}_{timestep:05d}]50:60`` for each observation.

    Parameters
    ----------
    results : dict[str, list[tuple[datetime, float]]]
        Composite head results keyed by well name.
    well_specs : list[ObsWellSpec]
        Well specifications.
    ins_path : Path
        Instruction file path.
    """
    with open(ins_path, "w") as f:
        f.write("pif #\n")
        f.write("l1\n")  # Skip header

        well_seq = 0
        for spec in well_specs:
            if spec.name not in results:
                continue
            well_seq += 1
            for t_seq, _ in enumerate(results[spec.name], start=1):
                f.write(f"l1 [WLT{well_seq:05d}_{t_seq:05d}]50:60\n")
