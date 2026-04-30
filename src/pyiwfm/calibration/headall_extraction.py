"""
HeadAll extraction pipeline for IWFM models.

Extracts head time series at arbitrary well locations from HeadAll HDF5 output
using finite element interpolation and transmissivity-weighted multi-layer
averaging.

This module composes existing pyiwfm components (LazyNodalLoader,
FEInterpolator, compute_multilayer_weights) into an end-to-end pipeline.

Example
-------
>>> extractor = HeadAllExtractor(model, headall_path)
>>> extractor.prepare(wells)
>>> result = extractor.extract()
>>> extractor.write_cache(Path("continuous_sim.hdf5"))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from pyiwfm.calibration.results_extraction import ExtractionResult

if TYPE_CHECKING:
    from pyiwfm.core.model import IWFMModel

logger = logging.getLogger(__name__)


@dataclass
class WellSpec:
    """Specification for a well extraction point.

    Attributes
    ----------
    name : str
        Well identifier.
    x : float
        X coordinate (in model units, typically feet).
    y : float
        Y coordinate (in model units, typically feet).
    layer : int
        Target model layer (1-based). Use -1 for multi-layer T-weighted.
    element : int
        Model element containing the well (0 = auto-detect).
    bos : float
        Bottom of screen elevation (feet).
    tos : float
        Top of screen elevation (feet).
    """

    name: str
    x: float
    y: float
    layer: int = -1
    element: int = 0
    bos: float = float("nan")
    tos: float = float("nan")


class HeadAllExtractor:
    """Extract head time series at arbitrary well locations from HeadAll HDF5.

    Parameters
    ----------
    model : IWFMModel
        Loaded IWFM model (for mesh geometry and stratigraphy).
    headall_path : Path
        Path to the HeadAll HDF5 output file.
    """

    def __init__(self, model: IWFMModel, headall_path: Path) -> None:
        self._model = model
        self._headall_path = Path(headall_path)
        self._wells: list[WellSpec] = []
        self._fe_weights: dict[str, dict[str, Any]] = {}
        self._t_weights: dict[str, NDArray[np.float64]] = {}
        self._head_loader = None

    def prepare(self, wells: list[WellSpec]) -> None:
        """Pre-compute FE interpolation weights and T-weights for all wells.

        Parameters
        ----------
        wells : list[WellSpec]
            Well specifications with coordinates in model units.
        """
        from pyiwfm.core.interpolation import FEInterpolator

        self._wells = wells

        # Build FE interpolator from model mesh
        grid = self._model.grid
        assert grid is not None, "Model must have a grid loaded"
        strat = self._model.stratigraphy
        interp = FEInterpolator(grid)

        n_layers = self._model.n_layers
        n_outside = 0

        for well in wells:
            # Use element hint if provided (fast path avoids brute-force search)
            used_hint = False
            if well.element > 0:
                try:
                    coeffs = interp.interpolate_at_element(well.x, well.y, well.element)
                    if np.all(coeffs >= -0.1):
                        elem_id = well.element
                        node_ids = interp.grid.elements[well.element].vertices
                        used_hint = True
                except (KeyError, AttributeError):
                    pass

            if not used_hint:
                elem_id, node_ids, coeffs = interp.interpolate(well.x, well.y)

            if elem_id == 0:
                n_outside += 1
                continue
            self._fe_weights[well.name] = {
                "element": elem_id,
                "node_ids": node_ids,
                "weights": coeffs,
            }

            # Compute T-weights for multi-layer averaging
            if not np.isnan(well.bos) and not np.isnan(well.tos):
                from pyiwfm.calibration.iwfm2obs import (
                    MultiLayerWellSpec,
                    compute_multilayer_weights,
                )

                ml_well = MultiLayerWellSpec(
                    name=well.name,
                    x=well.x,
                    y=well.y,
                    element_id=elem_id,
                    bottom_of_screen=well.bos,
                    top_of_screen=well.tos,
                )
                # Get Kh from model; fall back to equal weights if unavailable
                kh = None
                if (
                    self._model.groundwater
                    and self._model.groundwater.aquifer_params
                    and self._model.groundwater.aquifer_params.kh is not None
                ):
                    # AquiferParameters.kh is (n_nodes, n_layers); function expects (n_layers, n_nodes)
                    kh = self._model.groundwater.aquifer_params.kh.T

                if kh is not None and strat is not None:
                    t_weights = compute_multilayer_weights(
                        ml_well,
                        grid,
                        strat,
                        kh,
                        fe_node_ids=node_ids,
                        fe_weights=coeffs,
                    )
                    self._t_weights[well.name] = t_weights
                else:
                    self._t_weights[well.name] = np.ones(n_layers) / n_layers
            else:
                # Default: equal weights across all layers
                self._t_weights[well.name] = np.ones(n_layers) / n_layers

        if n_outside > 0:
            logger.warning("%d wells outside model mesh, skipped", n_outside)

        logger.info(
            "Prepared %d wells for extraction (%d with FE weights)",
            len(wells),
            len(self._fe_weights),
        )

    def extract(self, timesteps: list[int] | None = None) -> ExtractionResult:
        """Extract per-layer and T-weighted multi-layer heads at all wells.

        Iterates timesteps in the outer loop so each HDF5 frame is read
        exactly once, regardless of the number of wells.

        Parameters
        ----------
        timesteps : list[int], optional
            Specific timestep indices to extract. If None, extract all.

        Returns
        -------
        ExtractionResult
        """
        from pyiwfm.io.timeseries.lazy import LazyNodalLoader

        loader = LazyNodalLoader(self._headall_path)
        all_times = loader.times
        n_layers = self._model.n_layers

        if timesteps is not None:
            time_indices = timesteps
        else:
            time_indices = list(range(len(all_times)))

        times = np.array([all_times[i] for i in time_indices])
        n_times = len(time_indices)

        result = ExtractionResult(times=times, data_type="HEAD")

        # Build list of active wells with pre-computed index arrays
        active_wells: list[tuple[str, NDArray, NDArray]] = []
        for well in self._wells:
            if well.name not in self._fe_weights:
                continue
            fe_info = self._fe_weights[well.name]
            node_indices = np.array([nid - 1 for nid in fe_info["node_ids"]])
            coeffs = fe_info["weights"]
            active_wells.append((well.name, node_indices, coeffs))

        # Pre-allocate per-layer arrays for all active wells
        per_layer_arrays: dict[str, NDArray[np.float64]] = {}
        for name, _, _ in active_wells:
            per_layer_arrays[name] = np.full((n_times, n_layers), np.nan, dtype=np.float64)

        # Outer loop: timesteps (each frame read once)
        for ti, ts_idx in enumerate(time_indices):
            frame = loader.get_frame(ts_idx)  # (n_nodes, n_layers)
            for name, node_indices, coeffs in active_wells:
                for layer in range(n_layers):
                    vals = frame[node_indices, layer]
                    if not np.any(np.isnan(vals)):
                        per_layer_arrays[name][ti, layer] = float(np.dot(coeffs, vals))

        # Populate result and compute T-weighted multi-layer heads
        for name, _, _ in active_wells:
            per_layer = per_layer_arrays[name]
            result.per_layer[name] = per_layer
            result.names.append(name)

            t_weights = self._t_weights.get(name)
            if t_weights is not None:
                ml_heads = np.full(n_times, np.nan, dtype=np.float64)
                for ti in range(n_times):
                    layer_heads = per_layer[ti, :]
                    valid = ~np.isnan(layer_heads)
                    if valid.any():
                        w = t_weights[valid]
                        w_sum = w.sum()
                        if w_sum > 0:
                            ml_heads[ti] = float(np.dot(w, layer_heads[valid]) / w_sum)
                result.values[name] = ml_heads

        logger.info(
            "Extracted heads for %d wells across %d timesteps",
            len(result.names),
            n_times,
        )
        return result

    def write_cache(self, output_path: Path, result: ExtractionResult) -> None:
        """Write extraction results to HDF5 cache.

        Layout: ``/times``, ``/per_layer``, ``/values``,
        ``/names``.

        Parameters
        ----------
        output_path : Path
            Output HDF5 file path.
        result : ExtractionResult
            Results from :meth:`extract`.
        """
        import h5py

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_path, "w") as hf:
            # Times
            time_strs = [str(t) for t in result.times]
            hf.create_dataset("times", data=np.array(time_strs, dtype="S30"))

            # Well names
            hf.create_dataset(
                "names",
                data=np.array(result.names, dtype="S50"),
            )

            # Per-layer heads: (n_times, n_wells, n_layers)
            if result.per_layer:
                n_times = len(result.times)
                n_wells = len(result.names)
                n_layers = next(iter(result.per_layer.values())).shape[1]
                pl_arr = np.full((n_times, n_wells, n_layers), np.nan)
                for wi, name in enumerate(result.names):
                    if name in result.per_layer:
                        pl_arr[:, wi, :] = result.per_layer[name]
                hf.create_dataset("per_layer", data=pl_arr, compression="gzip")

            # Multi-layer heads: (n_times, n_wells)
            if result.values:
                n_times = len(result.times)
                n_wells = len(result.names)
                ml_arr = np.full((n_times, n_wells), np.nan)
                for wi, name in enumerate(result.names):
                    if name in result.values:
                        ml_arr[:, wi] = result.values[name]
                hf.create_dataset("values", data=ml_arr, compression="gzip")

        logger.info("Wrote cache to %s", output_path)

    @staticmethod
    def load_cache(cache_path: Path) -> ExtractionResult:
        """Load previously cached extraction results.

        Parameters
        ----------
        cache_path : Path
            Path to the HDF5 cache file.

        Returns
        -------
        ExtractionResult
        """
        import h5py

        with h5py.File(cache_path, "r") as hf:
            times = np.array([np.datetime64(s.decode()) for s in hf["times"][:]])

            names = [s.decode().strip() for s in hf["names"][:]]

            result = ExtractionResult(times=times, names=names, data_type="HEAD")

            if "per_layer" in hf:
                pl_arr = hf["per_layer"][:]
                for wi, name in enumerate(names):
                    result.per_layer[name] = pl_arr[:, wi, :]

            if "values" in hf:
                ml_arr = hf["values"][:]
                for wi, name in enumerate(names):
                    result.values[name] = ml_arr[:, wi]

        return result
