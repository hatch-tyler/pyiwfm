"""Compare simulation results between two IWFM model runs.

Provides the critical verification step for roundtrip testing:
compares heads, budgets, and hydrographs between a baseline model
run and a roundtripped (written) model run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HeadComparison:
    """Result of comparing groundwater head outputs.

    Attributes
    ----------
    n_timesteps : int
        Number of timesteps compared.
    n_nodes : int
        Number of nodes compared.
    max_abs_diff : float
        Maximum absolute difference across all nodes and timesteps.
    mean_abs_diff : float
        Mean absolute difference.
    n_mismatched_timesteps : int
        Number of timesteps that exceed tolerance.
    within_tolerance : bool
        True if all timesteps are within the specified tolerance.
    """

    n_timesteps: int = 0
    n_nodes: int = 0
    max_abs_diff: float = 0.0
    mean_abs_diff: float = 0.0
    n_mismatched_timesteps: int = 0
    within_tolerance: bool = False


@dataclass
class BudgetComparison:
    """Result of comparing a single budget file.

    Attributes
    ----------
    name : str
        Budget file identifier.
    n_datasets : int
        Number of datasets compared.
    max_rel_diff : float
        Maximum relative difference.
    within_tolerance : bool
        True if all datasets are within tolerance.
    details : list[str]
        Per-dataset comparison notes.
    """

    name: str = ""
    n_datasets: int = 0
    max_rel_diff: float = 0.0
    within_tolerance: bool = False
    details: list[str] = field(default_factory=list)


@dataclass
class HydrographComparison:
    """Result of comparing hydrograph outputs.

    Attributes
    ----------
    name : str
        Hydrograph file identifier.
    n_locations : int
        Number of hydrograph locations compared.
    min_nse : float
        Minimum Nash-Sutcliffe Efficiency across locations.
    mean_nse : float
        Mean NSE across locations.
    n_poor_matches : int
        Number of locations with NSE below threshold.
    within_tolerance : bool
        True if all locations meet the NSE threshold.
    """

    name: str = ""
    n_locations: int = 0
    min_nse: float = 0.0
    mean_nse: float = 0.0
    n_poor_matches: int = 0
    within_tolerance: bool = False


@dataclass
class ResultsComparison:
    """Aggregate comparison of all simulation outputs.

    Attributes
    ----------
    head_comparison : HeadComparison | None
        Head comparison results.
    budget_comparisons : list[BudgetComparison]
        Per-budget file comparisons.
    hydrograph_comparisons : list[HydrographComparison]
        Per-hydrograph file comparisons.
    final_heads_match : bool | None
        Whether final heads files match.
    errors : list[str]
        Any errors encountered during comparison.
    """

    head_comparison: HeadComparison | None = None
    budget_comparisons: list[BudgetComparison] = field(default_factory=list)
    hydrograph_comparisons: list[HydrographComparison] = field(default_factory=list)
    final_heads_match: bool | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """True if all comparisons pass."""
        if self.errors:
            return False
        if self.head_comparison and not self.head_comparison.within_tolerance:
            return False
        if any(not b.within_tolerance for b in self.budget_comparisons):
            return False
        if any(not h.within_tolerance for h in self.hydrograph_comparisons):
            return False
        if self.final_heads_match is False:
            return False
        return True

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = ["Results Comparison Summary", "-" * 40]

        if self.head_comparison:
            hc = self.head_comparison
            status = "PASS" if hc.within_tolerance else "FAIL"
            lines.append(
                f"  [{status}] Heads: max_diff={hc.max_abs_diff:.6f}, "
                f"mean_diff={hc.mean_abs_diff:.6f}, "
                f"{hc.n_mismatched_timesteps}/{hc.n_timesteps} exceed tol"
            )

        for bc in self.budget_comparisons:
            status = "PASS" if bc.within_tolerance else "FAIL"
            lines.append(f"  [{status}] Budget '{bc.name}': max_rel_diff={bc.max_rel_diff:.6f}")

        for hg in self.hydrograph_comparisons:
            status = "PASS" if hg.within_tolerance else "FAIL"
            lines.append(
                f"  [{status}] Hydrograph '{hg.name}': "
                f"min_NSE={hg.min_nse:.6f}, "
                f"{hg.n_poor_matches}/{hg.n_locations} poor"
            )

        if self.final_heads_match is not None:
            status = "PASS" if self.final_heads_match else "FAIL"
            lines.append(f"  [{status}] Final heads")

        for err in self.errors:
            lines.append(f"  [ERROR] {err}")

        return "\n".join(lines)


class ResultsDiffer:
    """Compare simulation outputs between two model runs.

    Parameters
    ----------
    baseline_dir : Path
        Directory containing baseline model outputs.
    written_dir : Path
        Directory containing written (roundtripped) model outputs.
    head_atol : float
        Absolute tolerance for head comparisons (ft).
    budget_rtol : float
        Relative tolerance for budget comparisons.
    nse_threshold : float
        Minimum NSE for hydrograph comparisons.
    """

    def __init__(
        self,
        baseline_dir: Path | str,
        written_dir: Path | str,
        head_atol: float = 0.01,
        budget_rtol: float = 1e-3,
        nse_threshold: float = 0.9999,
    ) -> None:
        self.baseline_dir = Path(baseline_dir)
        self.written_dir = Path(written_dir)
        self.head_atol = head_atol
        self.budget_rtol = budget_rtol
        self.nse_threshold = nse_threshold

    def compare_all(self) -> ResultsComparison:
        """Run all available comparisons.

        Returns
        -------
        ResultsComparison
            Aggregate results.
        """
        result = ResultsComparison()

        # Compare heads (try HDF5 first, then text)
        try:
            hc = self.compare_heads_hdf5()
            if hc is not None:
                result.head_comparison = hc
            else:
                hc = self.compare_heads_text()
                if hc is not None:
                    result.head_comparison = hc
        except Exception as exc:
            result.errors.append(f"Head comparison failed: {exc}")
            logger.exception("Head comparison failed")

        # Compare final heads
        try:
            result.final_heads_match = self.compare_final_heads()
        except Exception as exc:
            result.errors.append(f"Final heads comparison failed: {exc}")

        # Compare budgets
        try:
            result.budget_comparisons = self.compare_budgets()
        except Exception as exc:
            result.errors.append(f"Budget comparison failed: {exc}")
            logger.exception("Budget comparison failed")

        # Compare hydrographs
        try:
            result.hydrograph_comparisons = self.compare_hydrographs()
        except Exception as exc:
            result.errors.append(f"Hydrograph comparison failed: {exc}")
            logger.exception("Hydrograph comparison failed")

        return result

    def compare_heads_hdf5(self) -> HeadComparison | None:
        """Compare GW head outputs from HDF5 files.

        Loads timesteps one-at-a-time for memory efficiency.

        Returns
        -------
        HeadComparison | None
            Comparison result, or None if no HDF5 head files found.
        """
        baseline_hdf = self._find_head_hdf5(self.baseline_dir)
        written_hdf = self._find_head_hdf5(self.written_dir)

        if baseline_hdf is None or written_hdf is None:
            return None

        try:
            import h5py
        except ImportError:
            logger.warning("h5py not available; skipping HDF5 head comparison")
            return None

        result = HeadComparison()
        all_diffs: list[float] = []

        with h5py.File(baseline_hdf, "r") as fb, h5py.File(written_hdf, "r") as fw:
            # Find head dataset
            ds_name = self._find_head_dataset(fb)
            if ds_name is None:
                logger.warning("No head dataset found in %s", baseline_hdf)
                return None

            ds_b = fb[ds_name]
            ds_w = fw[ds_name]

            n_timesteps = min(ds_b.shape[0], ds_w.shape[0])
            result.n_timesteps = n_timesteps

            if len(ds_b.shape) >= 2:
                result.n_nodes = ds_b.shape[1]

            # Compare timestep-by-timestep for memory efficiency
            for t in range(n_timesteps):
                baseline_data = ds_b[t]
                written_data = ds_w[t]

                # Flatten for comparison
                b_flat = np.asarray(baseline_data, dtype=np.float64).ravel()
                w_flat = np.asarray(written_data, dtype=np.float64).ravel()

                # Mask out NaN/inactive nodes
                mask = ~(np.isnan(b_flat) | np.isnan(w_flat))
                if not np.any(mask):
                    continue

                diff = np.abs(b_flat[mask] - w_flat[mask])
                max_diff = float(np.max(diff))
                all_diffs.append(max_diff)

                if not np.allclose(b_flat[mask], w_flat[mask], atol=self.head_atol):
                    result.n_mismatched_timesteps += 1

        if all_diffs:
            result.max_abs_diff = float(np.max(all_diffs))
            result.mean_abs_diff = float(np.mean(all_diffs))

        result.within_tolerance = result.n_mismatched_timesteps == 0
        return result

    def compare_heads_text(self) -> HeadComparison | None:
        """Compare GW head outputs from text .out files.

        Returns
        -------
        HeadComparison | None
            Comparison result, or None if no text head files found.
        """
        baseline_file = self._find_file(
            self.baseline_dir, ["*HeadAll*", "*GW_HeadAll*", "*_GWHead*"], suffix=".out"
        )
        written_file = self._find_file(
            self.written_dir, ["*HeadAll*", "*GW_HeadAll*", "*_GWHead*"], suffix=".out"
        )

        if baseline_file is None or written_file is None:
            return None

        result = HeadComparison()
        all_diffs: list[float] = []

        # Parse fixed-width text files
        b_blocks = _parse_head_text_file(baseline_file)
        w_blocks = _parse_head_text_file(written_file)

        n_timesteps = min(len(b_blocks), len(w_blocks))
        result.n_timesteps = n_timesteps

        for t in range(n_timesteps):
            b_arr = np.array(b_blocks[t], dtype=np.float64)
            w_arr = np.array(w_blocks[t], dtype=np.float64)

            if b_arr.shape != w_arr.shape:
                result.n_mismatched_timesteps += 1
                continue

            mask = ~(np.isnan(b_arr) | np.isnan(w_arr))
            if not np.any(mask):
                continue

            diff = np.abs(b_arr[mask] - w_arr[mask])
            all_diffs.append(float(np.max(diff)))

            if not np.allclose(b_arr[mask], w_arr[mask], atol=self.head_atol):
                result.n_mismatched_timesteps += 1

        if all_diffs:
            result.max_abs_diff = float(np.max(all_diffs))
            result.mean_abs_diff = float(np.mean(all_diffs))

        result.within_tolerance = result.n_mismatched_timesteps == 0
        return result

    def compare_final_heads(self) -> bool | None:
        """Compare FinalGWHeads.dat files with float tolerance.

        Returns
        -------
        bool | None
            True if they match, False if not, None if files not found.
        """
        baseline_file = self._find_file(
            self.baseline_dir, ["*FinalHeads*", "*FinalGWHeads*"], suffix=".dat"
        )
        written_file = self._find_file(
            self.written_dir, ["*FinalHeads*", "*FinalGWHeads*"], suffix=".dat"
        )

        if baseline_file is None or written_file is None:
            return None

        b_lines = baseline_file.read_text(errors="replace").splitlines()
        w_lines = written_file.read_text(errors="replace").splitlines()

        if len(b_lines) != len(w_lines):
            return False

        for bl, wl in zip(b_lines, w_lines, strict=True):
            b_tokens = bl.split()
            w_tokens = wl.split()
            if len(b_tokens) != len(w_tokens):
                return False
            for bt, wt in zip(b_tokens, w_tokens, strict=True):
                if bt == wt:
                    continue
                try:
                    if abs(float(bt) - float(wt)) > self.head_atol:
                        return False
                except ValueError:
                    if bt != wt:
                        return False

        return True

    def compare_budgets(self) -> list[BudgetComparison]:
        """Compare budget HDF5 files between runs.

        Returns
        -------
        list[BudgetComparison]
            Per-file comparison results.
        """
        try:
            import h5py
        except ImportError:
            logger.warning("h5py not available; skipping budget comparison")
            return []

        results: list[BudgetComparison] = []

        # Find budget HDF5 files
        b_budgets = list(self.baseline_dir.rglob("*Budget*.hdf"))
        w_budgets = list(self.written_dir.rglob("*Budget*.hdf"))

        # Match by filename
        b_map = {f.name: f for f in b_budgets}
        w_map = {f.name: f for f in w_budgets}

        for name in sorted(set(b_map) & set(w_map)):
            comp = BudgetComparison(name=name)

            try:
                with h5py.File(b_map[name], "r") as fb, h5py.File(w_map[name], "r") as fw:
                    b_datasets = set(self._list_datasets(fb))
                    w_datasets = set(self._list_datasets(fw))
                    common = sorted(b_datasets & w_datasets)
                    comp.n_datasets = len(common)

                    max_rel = 0.0
                    all_ok = True

                    for ds_name in common:
                        b_data = np.asarray(fb[ds_name], dtype=np.float64)
                        w_data = np.asarray(fw[ds_name], dtype=np.float64)

                        if b_data.shape != w_data.shape:
                            comp.details.append(
                                f"{ds_name}: shape mismatch {b_data.shape} vs {w_data.shape}"
                            )
                            all_ok = False
                            continue

                        # Relative comparison with zero-safe denominator
                        denom = np.maximum(np.abs(b_data), 1e-10)
                        rel_diff = np.abs(b_data - w_data) / denom
                        this_max = float(np.max(rel_diff))
                        max_rel = max(max_rel, this_max)

                        if not np.allclose(b_data, w_data, rtol=self.budget_rtol, atol=0):
                            comp.details.append(f"{ds_name}: max_rel_diff={this_max:.6f}")
                            all_ok = False

                    comp.max_rel_diff = max_rel
                    comp.within_tolerance = all_ok

            except Exception as exc:
                comp.details.append(f"Error: {exc}")
                comp.within_tolerance = False

            results.append(comp)

        return results

    def compare_hydrographs(self) -> list[HydrographComparison]:
        """Compare hydrograph .out files between runs.

        Returns
        -------
        list[HydrographComparison]
            Per-file comparison results.
        """
        from pyiwfm.comparison.metrics import nash_sutcliffe as calc_nse

        results: list[HydrographComparison] = []

        # Find hydrograph files
        b_hyds = list(self.baseline_dir.rglob("*Hydrograph*.out"))
        b_hyds += list(self.baseline_dir.rglob("*Hyd*.out"))
        w_hyds = list(self.written_dir.rglob("*Hydrograph*.out"))
        w_hyds += list(self.written_dir.rglob("*Hyd*.out"))

        b_map = {f.name: f for f in b_hyds}
        w_map = {f.name: f for f in w_hyds}

        for name in sorted(set(b_map) & set(w_map)):
            comp = HydrographComparison(name=name)

            try:
                b_locs = _parse_hydrograph_text(b_map[name])
                w_locs = _parse_hydrograph_text(w_map[name])

                common_locs = sorted(set(b_locs) & set(w_locs))
                comp.n_locations = len(common_locs)

                nse_values: list[float] = []
                for loc in common_locs:
                    b_arr = np.array(b_locs[loc], dtype=np.float64)
                    w_arr = np.array(w_locs[loc], dtype=np.float64)
                    n = min(len(b_arr), len(w_arr))
                    if n < 2:
                        continue
                    nse = calc_nse(b_arr[:n], w_arr[:n])
                    nse_values.append(nse)
                    if nse < self.nse_threshold:
                        comp.n_poor_matches += 1

                if nse_values:
                    comp.min_nse = float(np.min(nse_values))
                    comp.mean_nse = float(np.mean(nse_values))
                else:
                    comp.min_nse = 0.0
                    comp.mean_nse = 0.0

                comp.within_tolerance = comp.n_poor_matches == 0

            except Exception as exc:
                comp.within_tolerance = False
                logger.warning("Failed to compare hydrograph %s: %s", name, exc)

            results.append(comp)

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_head_hdf5(self, model_dir: Path) -> Path | None:
        """Find GW head HDF5 file in a model directory."""
        return self._find_file(model_dir, ["*HeadAll*", "*GWHeadAll*", "*_GWHead*"], suffix=".hdf")

    @staticmethod
    def _find_head_dataset(hdf_file: Any) -> str | None:
        """Find the head dataset name in an HDF5 file."""
        candidates = ["head", "GWHeadAtAllNodes", "Head", "heads"]
        for name in candidates:
            if name in hdf_file:
                return name

        # Search top-level datasets
        for key in hdf_file.keys():
            if "head" in key.lower():
                return key

        return None

    @staticmethod
    def _find_file(
        model_dir: Path,
        patterns: list[str],
        suffix: str = "",
    ) -> Path | None:
        """Find a file matching patterns in a model directory."""
        for pattern in patterns:
            if suffix:
                matches = [f for f in model_dir.rglob(pattern) if f.suffix == suffix]
            else:
                matches = list(model_dir.rglob(pattern))
            if matches:
                return matches[0]
        return None

    @staticmethod
    def _list_datasets(group: Any, prefix: str = "") -> list[str]:
        """Recursively list HDF5 dataset paths."""
        import h5py

        datasets: list[str] = []
        for key in group.keys():
            item = group[key]
            path = f"{prefix}/{key}" if prefix else key
            if isinstance(item, h5py.Dataset):
                datasets.append(path)
            elif isinstance(item, h5py.Group):
                datasets.extend(ResultsDiffer._list_datasets(item, path))
        return datasets


# ------------------------------------------------------------------
# Text file parsers (lightweight, no external deps beyond numpy)
# ------------------------------------------------------------------


def _parse_head_text_file(filepath: Path) -> list[list[float]]:
    """Parse a GW_HeadAll.out text file into timestep blocks.

    Each block is a list of head values for all nodes at one timestep.

    Parameters
    ----------
    filepath : Path
        Path to the head text file.

    Returns
    -------
    list[list[float]]
        List of timestep blocks, each a list of head values.
    """
    blocks: list[list[float]] = []
    current_block: list[float] = []
    in_data = False

    with open(filepath, errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue

            # Detect timestep headers (contain date patterns)
            if "/" in stripped and "_" in stripped:
                if current_block:
                    blocks.append(current_block)
                    current_block = []
                in_data = True
                continue

            if in_data:
                tokens = stripped.split()
                for token in tokens:
                    try:
                        current_block.append(float(token))
                    except ValueError:
                        pass

    if current_block:
        blocks.append(current_block)

    return blocks


def _parse_hydrograph_text(
    filepath: Path,
) -> dict[str, list[float]]:
    """Parse an IWFM hydrograph .out file.

    Returns a dict mapping location identifier to list of values.

    Parameters
    ----------
    filepath : Path
        Path to the hydrograph file.

    Returns
    -------
    dict[str, list[float]]
        Location -> values mapping.
    """
    locations: dict[str, list[float]] = {}

    with open(filepath, errors="replace") as f:
        lines = f.readlines()

    # Skip header lines (lines starting with * or C or blank)
    data_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and stripped[0] not in ("*", "C", "c", "#"):
            # Check if this looks like data (starts with date or number)
            tokens = stripped.split()
            if len(tokens) >= 2:
                data_start = i
                break

    # Parse data lines
    for line in lines[data_start:]:
        stripped = line.strip()
        if not stripped or stripped[0] in ("*", "C", "c", "#"):
            continue

        tokens = stripped.split()
        if len(tokens) < 2:
            continue

        # First token is date, remaining are values at locations
        for col_idx in range(1, len(tokens)):
            loc_key = str(col_idx)
            try:
                val = float(tokens[col_idx])
                if loc_key not in locations:
                    locations[loc_key] = []
                locations[loc_key].append(val)
            except ValueError:
                pass

    return locations
