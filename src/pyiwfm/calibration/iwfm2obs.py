"""
IWFM2OBS — interpolate simulated heads to observation times.

Mirrors the Fortran IWFM2OBS utility with two core algorithms:

1. **Time interpolation** — linearly interpolate simulated time series to
   match observation timestamps.
2. **Multi-layer T-weighted averaging** — compute composite heads at wells
   that screen multiple aquifer layers, weighting by transmissivity.

Example
-------
>>> from pyiwfm.calibration.iwfm2obs import interpolate_to_obs_times
>>> result = interpolate_to_obs_times(observed_ts, simulated_ts)
>>> print(result.values)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from pyiwfm.io.smp import SMPReader, SMPTimeSeries, SMPWriter

if TYPE_CHECKING:
    from pyiwfm.core.mesh import AppGrid
    from pyiwfm.core.stratigraphy import Stratigraphy


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
