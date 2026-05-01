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
from typing import TYPE_CHECKING, Any, Literal

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

    times: NDArray[np.datetime64]
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
        self._fe_weights: dict[str, dict[str, Any]] = {}
        self._t_weights: dict[str, NDArray[np.float64]] = {}
        self._active_layers: dict[str, NDArray[np.bool_]] = {}

    def prepare(self, specs: list[ExtractionSpec]) -> None:
        """Pre-compute FE interpolation weights for all extraction points.

        Parameters
        ----------
        specs : list[ExtractionSpec]
            Extraction specifications with coordinates in model units.
        """
        from pyiwfm.core.interpolation import FEInterpolator

        self._specs = specs

        grid = self._model.grid
        if grid is None:
            raise ValueError("Model must have a grid loaded")
        strat = self._model.stratigraphy
        interp = FEInterpolator(grid)
        n_layers = self._model.n_layers
        n_outside = 0

        for spec in specs:
            # Find containing element and compute FE interpolation weights
            elem_id, node_ids, coeffs = interp.interpolate(spec.x, spec.y)
            if elem_id == 0:
                n_outside += 1
                continue
            self._fe_weights[spec.name] = {
                "element": elem_id,
                "node_ids": node_ids,
                "weights": coeffs,
            }

            # T-weights for multi-layer averaging (HEAD only, layer=-1)
            if self._data_type == "HEAD" and spec.layer == -1:
                if not np.isnan(spec.bos) and not np.isnan(spec.tos):
                    from pyiwfm.calibration.iwfm2obs import (
                        MultiLayerWellSpec,
                        compute_multilayer_weights,
                    )

                    ml_well = MultiLayerWellSpec(
                        name=spec.name,
                        x=spec.x,
                        y=spec.y,
                        element_id=elem_id,
                        bottom_of_screen=spec.bos,
                        top_of_screen=spec.tos,
                    )
                    kh = None
                    if (
                        self._model.groundwater
                        and self._model.groundwater.aquifer_params
                        and self._model.groundwater.aquifer_params.kh is not None
                    ):
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
                        self._t_weights[spec.name] = t_weights
                    else:
                        self._t_weights[spec.name] = np.ones(n_layers) / n_layers
                else:
                    self._t_weights[spec.name] = np.ones(n_layers) / n_layers

        # Build active-layer masks for HEAD Layer=0 (Fortran averages only
        # over layers where at least one FE interpolation node is active).
        if self._data_type == "HEAD" and strat is not None:
            for spec in specs:
                if spec.layer == 0 and spec.name in self._fe_weights:
                    fe_info = self._fe_weights[spec.name]
                    node_indices = np.array([nid - 1 for nid in fe_info["node_ids"]])
                    # active_node is (n_nodes, n_layers); layer active if ANY
                    # FE node is active there
                    active = np.any(strat.active_node[node_indices, :], axis=0)
                    self._active_layers[spec.name] = active

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

        Iterates timesteps in the outer loop so each HDF5 frame is read
        exactly once, regardless of the number of extraction points.

        Parameters
        ----------
        timesteps : list[int], optional
            Specific timestep indices to extract. If None, extract all.

        Returns
        -------
        ExtractionResult
        """
        from pyiwfm.io.timeseries.lazy import LazyNodalLoader

        n_layers = self._model.n_layers
        loader = LazyNodalLoader(self._results_path, n_layers=n_layers)
        all_times = loader.times

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

        # Build list of active specs with pre-computed index arrays
        active_specs: list[tuple[ExtractionSpec, NDArray, NDArray]] = []
        for spec in self._specs:
            if spec.name not in self._fe_weights:
                continue
            fe_info = self._fe_weights[spec.name]
            node_indices = np.array([nid - 1 for nid in fe_info["node_ids"]])
            coeffs = fe_info["weights"]
            active_specs.append((spec, node_indices, coeffs))

        # Pre-allocate per-layer arrays
        per_layer_arrays: dict[str, NDArray[np.float64]] = {}
        for spec, _, _ in active_specs:
            per_layer_arrays[spec.name] = np.full((n_times, n_layers), np.nan, dtype=np.float64)

        # Outer loop: timesteps (each frame read once). The inner work
        # used to loop over layers and call np.dot once per (location,
        # layer) — that's a Python-overhead-bound inner loop. We hoist
        # the per-layer work into one matrix-vector product per
        # (location, timestep): coeffs @ frame[node_indices, :] gives
        # the FE-interpolated value for every layer at once. NaN
        # propagation matches the loop version: any NaN in the input
        # nodes for a given layer leaves that layer NaN in the output.
        for ti, ts_idx in enumerate(time_indices):
            frame = loader.get_frame(ts_idx)  # (n_nodes, n_layers)
            for spec, node_indices, coeffs in active_specs:
                sub = frame[node_indices, :]  # (n_fe_nodes, n_layers)
                # Per-layer NaN if ANY input node is NaN — matches the
                # `if not np.any(np.isnan(vals))` check from the loop.
                nan_per_layer = np.any(np.isnan(sub), axis=0)
                # One matrix-vector product replaces the per-layer dot.
                # nansum(coeffs * sub) would also work but np.dot via
                # `coeffs @ sub` is the canonical numpy idiom.
                interpolated = coeffs @ sub  # (n_layers,)
                # Where any input node was NaN, propagate NaN.
                interpolated = np.where(nan_per_layer, np.nan, interpolated)
                per_layer_arrays[spec.name][ti, :] = interpolated

        # Populate result and aggregate
        for spec, _, _ in active_specs:
            per_layer = per_layer_arrays[spec.name]
            result.per_layer.setdefault(spec.name, per_layer)
            result.names.append(spec.name)

            aggregated = self._aggregate_layers(per_layer, spec, n_layers)

            if self._incremental:
                incr = np.full(n_times, np.nan, dtype=np.float64)
                incr[0] = 0.0
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
            # Sum over all layers (additive compaction). Vectorised over
            # timesteps: any-valid mask lets us return NaN for timesteps
            # that have no valid layers, matching the per-ts loop.
            valid = ~np.isnan(per_layer)  # (n_times, n_layers)
            any_valid = np.any(valid, axis=1)
            sums = np.nansum(per_layer, axis=1)  # (n_times,) — 0 when all-NaN
            result = np.where(any_valid, sums, np.nan)

        elif spec.layer == -1 and self._data_type == "HEAD":
            # T-weighted multi-layer average. Vectorised: zero out NaN
            # entries in both the values and the broadcast weight array,
            # then divide. Matches the per-ts loop's behaviour of
            # skipping NaN layers and returning NaN when no valid layers
            # contribute (denominator falls to 0).
            t_weights = self._t_weights.get(spec.name)
            if t_weights is not None:
                valid = ~np.isnan(per_layer)
                masked_vals = np.where(valid, per_layer, 0.0)
                masked_w = np.where(valid, t_weights[None, :], 0.0)
                numerator = np.sum(masked_vals * masked_w, axis=1)
                denominator = np.sum(masked_w, axis=1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    result = np.where(denominator > 0, numerator / denominator, np.nan)

        else:
            # layer == 0, HEAD: average over active layers only.
            # Matches Fortran ResultsExtract which uses Stratigraphy%ActiveNode
            # to exclude inactive layers from the average.
            active_mask = self._active_layers.get(spec.name)
            valid = ~np.isnan(per_layer)
            if active_mask is not None:
                valid &= active_mask[None, :]
            counts = np.sum(valid, axis=1)
            masked_vals = np.where(valid, per_layer, 0.0)
            sums = np.sum(masked_vals, axis=1)
            with np.errstate(invalid="ignore"):
                means = np.where(counts > 0, sums / counts, np.nan)
            result = means

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

        with open(output_path, "w", encoding="utf-8") as f:
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


class FortranBackend:
    """Run ResultsExtract Fortran executable as a fast extraction backend.

    Generates an input file, invokes ``ResultsExtract_x64.exe``, and parses
    the IWFM columnar ``.out`` output back into an :class:`ExtractionResult`.

    This is ~300x faster than the pure-Python FE extraction for large
    datasets (e.g. 31K subsidence locations × 577 timesteps: 3s vs 17min).

    Parameters
    ----------
    exe_path : Path
        Path to ``ResultsExtract_x64.exe``.
    sim_file : Path
        Path to the IWFM simulation main file.
    data_type : str
        ``'HEAD'`` or ``'SUBSIDENCE'``.
    incremental : bool
        If True and data_type is ``'SUBSIDENCE'``, the Fortran code produces
        incremental output automatically.
    """

    def __init__(
        self,
        exe_path: Path,
        sim_file: Path,
        data_type: str = "SUBSIDENCE",
        incremental: bool = True,
    ) -> None:
        self._exe_path = Path(exe_path)
        self._sim_file = Path(sim_file)
        self._data_type = data_type.upper()
        self._incremental = incremental if self._data_type == "SUBSIDENCE" else False

    @staticmethod
    def find_exe(search_dirs: list[Path] | None = None) -> Path | None:
        """Locate ``ResultsExtract_x64.exe`` in common locations."""
        candidates = [Path("tools") / "ResultsExtract_x64.exe"]
        if search_dirs:
            candidates.extend(d / "ResultsExtract_x64.exe" for d in search_dirs)
        for p in candidates:
            if p.exists():
                return p
        return None

    def generate_input(
        self,
        specs: list[ExtractionSpec],
        output_file: str = "RE_OUT.out",
        factxy: float = 1.0,
    ) -> tuple[str, list[str]]:
        """Generate a ResultsExtract input file from ExtractionSpec objects.

        Parameters
        ----------
        specs : list[ExtractionSpec]
            Extraction specifications (coordinates already in model units).
        output_file : str
            Name for the ResultsExtract output file.
        factxy : float
            Coordinate conversion factor written to the input file.

        Returns
        -------
        tuple[str, list[str]]
            (input file content, ordered list of spec names)
        """
        lines: list[str] = []
        lines.append(f"  {self._sim_file!s:<40s} / SIMFILE")
        lines.append(f"  {self._data_type:<40s} / DATATYPE")
        lines.append(f"  {output_file:<40s} / OUTFILE")
        lines.append(f"  {len(specs):<40d} / NHYD")
        lines.append(f"  {factxy:<40.5f} / FACTXY")

        names: list[str] = []
        for i, spec in enumerate(specs, 1):
            layer = max(spec.layer, 0)
            lines.append(f"  {i:<6d} 0  {layer:<4d} {spec.x:<18.4f} {spec.y:<18.4f} {spec.name}")
            names.append(spec.name)

        return "\n".join(lines) + "\n", names

    def run(
        self,
        specs: list[ExtractionSpec],
        work_dir: Path | None = None,
        factxy: float = 1.0,
    ) -> ExtractionResult:
        """Run ResultsExtract and parse the output.

        Parameters
        ----------
        specs : list[ExtractionSpec]
            Extraction specifications (coordinates in model units).
        work_dir : Path, optional
            Working directory for the subprocess.
        factxy : float
            Coordinate conversion factor.

        Returns
        -------
        ExtractionResult

        Raises
        ------
        FileNotFoundError
            If the executable is not found.
        RuntimeError
            If ResultsExtract reports a FATAL error.
        """
        import subprocess

        if not self._exe_path.exists():
            raise FileNotFoundError(f"ResultsExtract not found: {self._exe_path}")

        cwd = Path(work_dir) if work_dir else Path.cwd()
        out_name = f"RE_{self._data_type}_OUT.out"
        input_content, names = self.generate_input(specs, out_name, factxy)

        input_path = cwd / f"resultsextract_{self._data_type.lower()}_auto.in"
        input_path.write_text(input_content)

        logger.info(
            "Running ResultsExtract (%s, %d locations)...",
            self._data_type,
            len(specs),
        )

        try:
            proc = subprocess.run(
                [str(self._exe_path), str(input_path)],
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=300,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"ResultsExtract exited with code {proc.returncode}:\n"
                    f"{proc.stderr or proc.stdout}"
                )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("ResultsExtract timed out after 300 seconds") from exc
        finally:
            # Clean up input file regardless of success/failure
            if input_path.exists():
                input_path.unlink(missing_ok=True)

        msg_file = cwd / "ResultsExtract_Messages.out"
        if msg_file.exists():
            msg_text = msg_file.read_text()
            if "* FATAL:" in msg_text:
                raise RuntimeError(f"ResultsExtract FATAL error:\n{msg_text}")

        out_path = cwd / out_name
        if not out_path.exists():
            raise RuntimeError(f"ResultsExtract output not found: {out_path}")

        return self._parse_output(out_path, names)

    def _parse_output(
        self,
        out_path: Path,
        names: list[str],
    ) -> ExtractionResult:
        """Parse IWFM columnar .out file into ExtractionResult.

        The .out file has header lines starting with ``*`` followed by
        data lines: ``MM/DD/YYYY_HH:MM   val1   val2   val3 ...``
        """
        import re
        from datetime import datetime

        times_list: list[np.datetime64] = []
        data_rows: list[list[float]] = []
        n_names = len(names)

        with open(out_path, encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("*"):
                    continue
                ts_match = re.match(r"(\d{2}/\d{2}/\d{4})_(\d{2}:\d{2})", stripped)
                if not ts_match:
                    continue

                date_str = ts_match.group(1)
                time_str = ts_match.group(2)
                if time_str.startswith("24:"):
                    time_str = "00:00"
                try:
                    dt = datetime.strptime(f"{date_str} {time_str}", "%m/%d/%Y %H:%M")
                except ValueError:
                    continue

                times_list.append(np.datetime64(dt))

                val_str = stripped[ts_match.end() :]
                vals = []
                for v in val_str.split():
                    try:
                        vals.append(float(v))
                    except ValueError:
                        vals.append(float("nan"))
                while len(vals) < n_names:
                    vals.append(float("nan"))
                data_rows.append(vals[:n_names])

        times = np.array(times_list)

        result = ExtractionResult(
            times=times,
            names=list(names),
            data_type=self._data_type,
            incremental=self._incremental,
        )

        if data_rows:
            data = np.array(data_rows, dtype=np.float64)
            for ni, name in enumerate(names):
                result.values[name] = data[:, ni]

        logger.info(
            "Parsed ResultsExtract output: %d timesteps × %d locations",
            len(times),
            n_names,
        )
        return result
