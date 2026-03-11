"""
HeadAll extraction pipeline for IWFM models.

Extracts head time series at arbitrary well locations from HeadAll HDF5 output
using finite element interpolation and transmissivity-weighted multi-layer
averaging.

This module composes existing pyiwfm components (LazyHeadDataLoader,
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

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


@dataclass
class ExtractionResult:
    """Results of HeadAll extraction.

    Attributes
    ----------
    times : NDArray
        Array of timestamps (datetime64).
    well_names : list[str]
        Names of wells in order.
    per_layer_heads : dict[str, NDArray[np.float64]]
        Per-well per-layer heads: {well_name: array of shape (n_times, n_layers)}.
    multi_layer_heads : dict[str, NDArray[np.float64]]
        T-weighted multi-layer heads: {well_name: array of shape (n_times,)}.
    """

    times: NDArray
    well_names: list[str] = field(default_factory=list)
    per_layer_heads: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    multi_layer_heads: dict[str, NDArray[np.float64]] = field(default_factory=dict)


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
        self._fe_weights: dict[str, dict] = {}
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
        interp = FEInterpolator(self._model)  # type: ignore[arg-type]

        n_layers = self._model.n_layers
        n_outside = 0

        for well in wells:
            # Find containing element
            elem = well.element
            if elem <= 0:
                elem = interp.find_element(well.x, well.y)
                if elem <= 0:
                    n_outside += 1
                    continue

            # Compute FE shape function weights
            weights = interp.interpolation_weights(well.x, well.y, elem)  # type: ignore[attr-defined]
            self._fe_weights[well.name] = {
                "element": elem,
                "weights": weights,
            }

            # Compute T-weights for multi-layer averaging
            if not np.isnan(well.bos) and not np.isnan(well.tos):
                from pyiwfm.calibration.iwfm2obs import compute_multilayer_weights

                t_weights = compute_multilayer_weights(  # type: ignore[call-arg]
                    self._model,  # type: ignore[arg-type]
                    elem,  # type: ignore[arg-type]
                    well.bos,  # type: ignore[arg-type]
                    well.tos,  # type: ignore[arg-type]
                    n_layers,
                )
                self._t_weights[well.name] = t_weights
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

        Parameters
        ----------
        timesteps : list[int], optional
            Specific timestep indices to extract. If None, extract all.

        Returns
        -------
        ExtractionResult
        """
        from pyiwfm.io.head_loader import LazyHeadDataLoader

        loader = LazyHeadDataLoader(self._headall_path)
        all_times = loader.times
        n_layers = self._model.n_layers

        if timesteps is not None:
            time_indices = timesteps
        else:
            time_indices = list(range(len(all_times)))

        times = np.array([all_times[i] for i in time_indices])
        n_times = len(time_indices)

        result = ExtractionResult(times=times)

        for well in self._wells:
            if well.name not in self._fe_weights:
                continue

            fe_info = self._fe_weights[well.name]
            elem = fe_info["element"]
            weights = fe_info["weights"]

            # Extract heads at this well for all timesteps and layers
            per_layer = np.full((n_times, n_layers), np.nan, dtype=np.float64)

            for ti, ts_idx in enumerate(time_indices):
                for layer in range(n_layers):
                    head_at_nodes = loader.get_head(ts_idx, layer, elem)  # type: ignore[attr-defined]
                    if head_at_nodes is not None:
                        per_layer[ti, layer] = float(np.dot(weights, head_at_nodes))

            result.per_layer_heads[well.name] = per_layer
            result.well_names.append(well.name)

            # Compute T-weighted multi-layer head
            t_weights = self._t_weights.get(well.name)
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
                result.multi_layer_heads[well.name] = ml_heads

        logger.info(
            "Extracted heads for %d wells across %d timesteps",
            len(result.well_names),
            n_times,
        )
        return result

    def write_cache(self, output_path: Path, result: ExtractionResult) -> None:
        """Write extraction results to HDF5 cache.

        Layout: ``/times``, ``/per_layer/heads``, ``/multi_layer/heads``,
        ``/well_names``.

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
                "well_names",
                data=np.array(result.well_names, dtype="S50"),
            )

            # Per-layer heads: (n_times, n_wells, n_layers)
            if result.per_layer_heads:
                n_times = len(result.times)
                n_wells = len(result.well_names)
                n_layers = next(iter(result.per_layer_heads.values())).shape[1]
                pl_arr = np.full((n_times, n_wells, n_layers), np.nan)
                for wi, name in enumerate(result.well_names):
                    if name in result.per_layer_heads:
                        pl_arr[:, wi, :] = result.per_layer_heads[name]
                hf.create_dataset("per_layer/heads", data=pl_arr, compression="gzip")

            # Multi-layer heads: (n_times, n_wells)
            if result.multi_layer_heads:
                n_times = len(result.times)
                n_wells = len(result.well_names)
                ml_arr = np.full((n_times, n_wells), np.nan)
                for wi, name in enumerate(result.well_names):
                    if name in result.multi_layer_heads:
                        ml_arr[:, wi] = result.multi_layer_heads[name]
                hf.create_dataset("multi_layer/heads", data=ml_arr, compression="gzip")

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

            well_names = [s.decode().strip() for s in hf["well_names"][:]]

            result = ExtractionResult(times=times, well_names=well_names)

            if "per_layer/heads" in hf:
                pl_arr = hf["per_layer/heads"][:]
                for wi, name in enumerate(well_names):
                    result.per_layer_heads[name] = pl_arr[:, wi, :]

            if "multi_layer/heads" in hf:
                ml_arr = hf["multi_layer/heads"][:]
                for wi, name in enumerate(well_names):
                    result.multi_layer_heads[name] = ml_arr[:, wi]

        return result
