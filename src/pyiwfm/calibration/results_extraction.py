"""
Unified results extraction pipeline for IWFM models.

Extracts time series at arbitrary locations from any all-node HDF5 output
(HeadAll or SubsidenceAll) using finite element interpolation.

For HEAD data, supports transmissivity-weighted multi-layer averaging.
For SUBSIDENCE data, supports layer summation (additive compaction) and
cumulative-to-incremental conversion.

This generalizes HeadAllExtractor to handle any all-node output type.

Example
-------
>>> extractor = ResultsExtractor(model, results_path, data_type='SUBSIDENCE')
>>> extractor.prepare(specs)
>>> result = extractor.extract()
>>> extractor.write_smp(Path("SUB_OUT.smp"), result)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pyiwfm.core.model import IWFMModel

logger = logging.getLogger(__name__)


@dataclass
class ExtractionSpec:
    """Specification for an extraction point.

    Attributes
    ----------
    name : str
        Location identifier.
    x : float
        X coordinate (in model units, typically feet).
    y : float
        Y coordinate (in model units, typically feet).
    layer : int
        Target model layer (1-based). Use 0 for all-layer aggregation
        (average for HEAD, sum for SUBSIDENCE). Use -1 for T-weighted
        multi-layer average (HEAD only).
    element : int
        Model element containing the point (0 = auto-detect).
    bos : float
        Bottom of screen elevation (feet), for T-weighted averaging.
    tos : float
        Top of screen elevation (feet), for T-weighted averaging.
    """

    name: str
    x: float
    y: float
    layer: int = 0
    element: int = 0
    bos: float = float("nan")
    tos: float = float("nan")


@dataclass
class ExtractionResult:
    """Results of extraction.

    Attributes
    ----------
    times : NDArray
        Array of timestamps (datetime64).
    names : list[str]
        Names of extraction points in order.
    values : dict[str, NDArray[np.float64]]
        Extracted values: {name: array of shape (n_times,)}.
    per_layer : dict[str, NDArray[np.float64]]
        Per-layer values: {name: array of shape (n_times, n_layers)}.
    data_type : str
        'HEAD' or 'SUBSIDENCE'.
    incremental : bool
        If True, values are incremental (diff between timesteps).
    """

    times: NDArray
    names: list[str] = field(default_factory=list)
    values: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    per_layer: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    data_type: str = "HEAD"
    incremental: bool = False


class ResultsExtractor:
    """Extract time series at arbitrary locations from all-node HDF5 output.

    Parameters
    ----------
    model : IWFMModel
        Loaded IWFM model (for mesh geometry and stratigraphy).
    results_path : Path
        Path to the all-node HDF5 output file (HeadAll or SubsidenceAll).
    data_type : str
        'HEAD' or 'SUBSIDENCE'.
    incremental : bool
        If True and data_type is 'SUBSIDENCE', convert cumulative values
        to incremental (current - previous timestep). Ignored for HEAD.
    """

    def __init__(
        self,
        model: IWFMModel,
        results_path: Path,
        data_type: Literal["HEAD", "SUBSIDENCE"] = "HEAD",
        incremental: bool = True,
    ) -> None:
        self._model = model
        self._results_path = Path(results_path)
        self._data_type = data_type.upper()
        self._incremental = incremental if self._data_type == "SUBSIDENCE" else False
        self._specs: list[ExtractionSpec] = []
        self._fe_weights: dict[str, dict] = {}
        self._t_weights: dict[str, NDArray[np.float64]] = {}

    def prepare(self, specs: list[ExtractionSpec]) -> None:
        """Pre-compute FE interpolation weights for all extraction points.

        Parameters
        ----------
        specs : list[ExtractionSpec]
            Extraction specifications with coordinates in model units.
        """
        from pyiwfm.core.interpolation import FEInterpolator

        self._specs = specs

        interp = FEInterpolator(self._model)  # type: ignore[arg-type]
        n_layers = self._model.n_layers
        n_outside = 0

        for spec in specs:
            elem = spec.element
            if elem <= 0:
                elem = interp.find_element(spec.x, spec.y)
                if elem <= 0:
                    n_outside += 1
                    continue

            weights = interp.interpolation_weights(spec.x, spec.y, elem)  # type: ignore[attr-defined]
            self._fe_weights[spec.name] = {
                "element": elem,
                "weights": weights,
            }

            # T-weights for multi-layer averaging (HEAD only, layer=-1)
            if self._data_type == "HEAD" and spec.layer == -1:
                if not np.isnan(spec.bos) and not np.isnan(spec.tos):
                    from pyiwfm.calibration.iwfm2obs import compute_multilayer_weights

                    t_weights = compute_multilayer_weights(  # type: ignore[call-arg]
                        self._model,  # type: ignore[arg-type]
                        elem,  # type: ignore[arg-type]
                        spec.bos,  # type: ignore[arg-type]
                        spec.tos,  # type: ignore[arg-type]
                        n_layers,
                    )
                    self._t_weights[spec.name] = t_weights
                else:
                    self._t_weights[spec.name] = np.ones(n_layers) / n_layers

        if n_outside > 0:
            logger.warning("%d points outside model mesh, skipped", n_outside)

        logger.info(
            "Prepared %d points for %s extraction (%d with FE weights)",
            len(specs),
            self._data_type,
            len(self._fe_weights),
        )

    def extract(self, timesteps: list[int] | None = None) -> ExtractionResult:
        """Extract values at all locations.

        Parameters
        ----------
        timesteps : list[int], optional
            Specific timestep indices to extract. If None, extract all.

        Returns
        -------
        ExtractionResult
        """
        from pyiwfm.io.head_loader import LazyHeadDataLoader

        loader = LazyHeadDataLoader(self._results_path)
        all_times = loader.times
        n_layers = self._model.n_layers

        if timesteps is not None:
            time_indices = timesteps
        else:
            time_indices = list(range(len(all_times)))

        times = np.array([all_times[i] for i in time_indices])
        n_times = len(time_indices)

        result = ExtractionResult(
            times=times,
            data_type=self._data_type,
            incremental=self._incremental,
        )

        for spec in self._specs:
            if spec.name not in self._fe_weights:
                continue

            fe_info = self._fe_weights[spec.name]
            weights = fe_info["weights"]
            elem = fe_info["element"]

            # Extract per-layer values
            per_layer = np.full((n_times, n_layers), np.nan, dtype=np.float64)

            for ti, ts_idx in enumerate(time_indices):
                for layer in range(n_layers):
                    data_at_nodes = loader.get_head(ts_idx, layer, elem)  # type: ignore[attr-defined]
                    if data_at_nodes is not None:
                        per_layer[ti, layer] = float(np.dot(weights, data_at_nodes))

            result.per_layer.setdefault(spec.name, per_layer)
            result.names.append(spec.name)

            # Aggregate across layers
            aggregated = self._aggregate_layers(
                per_layer,
                spec,
                n_layers,
            )

            # Incremental conversion for subsidence
            if self._incremental:
                incr = np.full(n_times, np.nan, dtype=np.float64)
                incr[0] = 0.0  # No prior timestep
                incr[1:] = aggregated[1:] - aggregated[:-1]
                result.values[spec.name] = incr
            else:
                result.values[spec.name] = aggregated

        logger.info(
            "Extracted %s for %d points across %d timesteps",
            self._data_type,
            len(result.names),
            n_times,
        )
        return result

    def _aggregate_layers(
        self,
        per_layer: NDArray[np.float64],
        spec: ExtractionSpec,
        n_layers: int,
    ) -> NDArray[np.float64]:
        """Aggregate per-layer values into a single time series.

        HEAD:
          - layer > 0: single layer
          - layer == 0: average over layers with valid data
          - layer == -1: T-weighted multi-layer average

        SUBSIDENCE:
          - layer > 0: single layer
          - layer == 0: sum over all layers (additive compaction)
        """
        n_times = per_layer.shape[0]
        result = np.full(n_times, np.nan, dtype=np.float64)

        if spec.layer > 0:
            # Single specific layer
            result = per_layer[:, spec.layer - 1].copy()

        elif self._data_type == "SUBSIDENCE":
            # Sum over all layers (additive compaction)
            for ti in range(n_times):
                valid = ~np.isnan(per_layer[ti, :])
                if valid.any():
                    result[ti] = np.nansum(per_layer[ti, :])

        elif spec.layer == -1 and self._data_type == "HEAD":
            # T-weighted multi-layer average
            t_weights = self._t_weights.get(spec.name)
            if t_weights is not None:
                for ti in range(n_times):
                    valid = ~np.isnan(per_layer[ti, :])
                    if valid.any():
                        w = t_weights[valid]
                        w_sum = w.sum()
                        if w_sum > 0:
                            result[ti] = float(np.dot(w, per_layer[ti, valid]) / w_sum)

        else:
            # layer == 0, HEAD: average over active layers
            for ti in range(n_times):
                valid = ~np.isnan(per_layer[ti, :])
                if valid.any():
                    result[ti] = np.nanmean(per_layer[ti, :])

        return result

    def write_smp(self, output_path: Path, result: ExtractionResult) -> None:
        """Write results to SMP format for PEST/IWFM2OBS.

        Parameters
        ----------
        output_path : Path
            Output SMP file path.
        result : ExtractionResult
            Results from :meth:`extract`.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for name in result.names:
                if name not in result.values:
                    continue
                vals = result.values[name]
                for ti, t in enumerate(result.times):
                    if np.isnan(vals[ti]):
                        continue
                    dt = np.datetime64(t, "s").astype("datetime64[s]")
                    ts = str(dt)
                    # Parse to MM/DD/YYYY HH:MM:SS
                    date_part = ts[:10]  # YYYY-MM-DD
                    time_part = ts[11:19] if len(ts) > 10 else "00:00:00"
                    ymd = date_part.split("-")
                    date_str = f"{ymd[1]}/{ymd[2]}/{ymd[0]}"
                    # Fixed-format SMP: (A25,A12,A12,A11)
                    f.write(f"{name:<25s}{date_str:>12s}{time_part:>12s}{vals[ti]:11.3f}\n")

        logger.info("Wrote SMP to %s (%d locations)", output_path, len(result.names))

    def write_cache(self, output_path: Path, result: ExtractionResult) -> None:
        """Write extraction results to HDF5 cache.

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
            hf.attrs["data_type"] = result.data_type
            hf.attrs["incremental"] = result.incremental

            time_strs = [str(t) for t in result.times]
            hf.create_dataset("times", data=np.array(time_strs, dtype="S30"))
            hf.create_dataset(
                "names",
                data=np.array(result.names, dtype="S50"),
            )

            # Aggregated values: (n_times, n_names)
            n_times = len(result.times)
            n_names = len(result.names)
            vals_arr = np.full((n_times, n_names), np.nan)
            for ni, name in enumerate(result.names):
                if name in result.values:
                    vals_arr[:, ni] = result.values[name]
            hf.create_dataset("values", data=vals_arr, compression="gzip")

            # Per-layer: (n_times, n_names, n_layers)
            if result.per_layer:
                n_layers = next(iter(result.per_layer.values())).shape[1]
                pl_arr = np.full((n_times, n_names, n_layers), np.nan)
                for ni, name in enumerate(result.names):
                    if name in result.per_layer:
                        pl_arr[:, ni, :] = result.per_layer[name]
                hf.create_dataset("per_layer", data=pl_arr, compression="gzip")

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
            data_type = hf.attrs.get("data_type", "HEAD")
            incremental = bool(hf.attrs.get("incremental", False))

            result = ExtractionResult(
                times=times,
                names=names,
                data_type=data_type,
                incremental=incremental,
            )

            if "values" in hf:
                vals_arr = hf["values"][:]
                for ni, name in enumerate(names):
                    result.values[name] = vals_arr[:, ni]

            if "per_layer" in hf:
                pl_arr = hf["per_layer"][:]
                for ni, name in enumerate(names):
                    result.per_layer[name] = pl_arr[:, ni, :]

        return result
