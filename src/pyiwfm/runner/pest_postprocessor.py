"""Post-processing module for PEST++ results.

This module provides the PestPostProcessor class for loading,
analyzing, and summarizing PEST++ calibration results from
pestpp-glm, pestpp-ies, and pestpp-sen output files.

Features:
- Load PEST++ residual files (.rei, .res)
- Parse parameter sensitivity files (.sen)
- Load iteration records (.iobj, .isen)
- Compute goodness-of-fit statistics (RMSE, NSE, R², bias)
- Parameter identifiability analysis
- Export calibrated parameter values
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass
class ResidualData:
    """Observation residual data.

    Attributes
    ----------
    names : list[str]
        Observation names.
    groups : list[str]
        Observation group names.
    observed : NDArray
        Observed values.
    simulated : NDArray
        Simulated values.
    residuals : NDArray
        Residuals (observed - simulated).
    weights : NDArray
        Observation weights.
    """

    names: list[str]
    groups: list[str]
    observed: NDArray
    simulated: NDArray
    residuals: NDArray
    weights: NDArray

    @property
    def n_observations(self) -> int:
        """Number of observations."""
        return len(self.names)

    @property
    def weighted_residuals(self) -> NDArray:
        """Weighted residuals."""
        wr: NDArray = self.residuals * self.weights
        return wr

    @property
    def phi(self) -> float:
        """Total objective function (sum of squared weighted residuals)."""
        return float(np.sum(self.weighted_residuals**2))

    def group_phi(self) -> dict[str, float]:
        """Objective function contribution by group."""
        result = {}
        for group in set(self.groups):
            mask = np.array([g == group for g in self.groups])
            wr = self.weighted_residuals[mask]
            result[group] = float(np.sum(wr**2))
        return result


@dataclass
class SensitivityData:
    """Parameter sensitivity data.

    Attributes
    ----------
    parameter_names : list[str]
        Parameter names.
    composite_sensitivities : NDArray
        Composite scaled sensitivities.
    """

    parameter_names: list[str]
    composite_sensitivities: NDArray

    @property
    def n_parameters(self) -> int:
        """Number of parameters."""
        return len(self.parameter_names)

    def most_sensitive(self, n: int = 10) -> list[tuple[str, float]]:
        """Get most sensitive parameters.

        Parameters
        ----------
        n : int
            Number of parameters to return.

        Returns
        -------
        list[tuple[str, float]]
            Sorted (name, sensitivity) pairs.
        """
        indices = np.argsort(self.composite_sensitivities)[::-1][:n]
        return [(self.parameter_names[i], float(self.composite_sensitivities[i])) for i in indices]

    def least_sensitive(self, n: int = 10) -> list[tuple[str, float]]:
        """Get least sensitive parameters.

        Parameters
        ----------
        n : int
            Number of parameters to return.

        Returns
        -------
        list[tuple[str, float]]
            Sorted (name, sensitivity) pairs.
        """
        indices = np.argsort(self.composite_sensitivities)[:n]
        return [(self.parameter_names[i], float(self.composite_sensitivities[i])) for i in indices]


@dataclass
class CalibrationResults:
    """Summary of PEST++ calibration results.

    Attributes
    ----------
    case_name : str
        PEST++ case name.
    n_iterations : int
        Number of iterations completed.
    final_phi : float
        Final objective function value.
    iteration_phi : list[float]
        Objective function values per iteration.
    residuals : ResidualData | None
        Final residual data.
    sensitivities : SensitivityData | None
        Parameter sensitivity data.
    calibrated_values : dict[str, float]
        Calibrated parameter values.
    """

    case_name: str
    n_iterations: int = 0
    final_phi: float = 0.0
    iteration_phi: list[float] = field(default_factory=list)
    residuals: ResidualData | None = None
    sensitivities: SensitivityData | None = None
    calibrated_values: dict[str, float] = field(default_factory=dict)

    def fit_statistics(self, group: str | None = None) -> dict[str, float]:
        """Compute goodness-of-fit statistics.

        Parameters
        ----------
        group : str | None
            Observation group. If None, uses all observations.

        Returns
        -------
        dict[str, float]
            Statistics including RMSE, MAE, R², NSE, bias.
        """
        if self.residuals is None:
            return {}

        if group:
            mask = np.array([g == group for g in self.residuals.groups])
            obs = self.residuals.observed[mask]
            self.residuals.simulated[mask]
            res = self.residuals.residuals[mask]
        else:
            obs = self.residuals.observed
            res = self.residuals.residuals

        if len(obs) == 0:
            return {}

        rmse = float(np.sqrt(np.mean(res**2)))
        mae = float(np.mean(np.abs(res)))
        bias = float(np.mean(res))

        # R² (coefficient of determination)
        ss_res = np.sum(res**2)
        ss_tot = np.sum((obs - np.mean(obs)) ** 2)
        r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Nash-Sutcliffe Efficiency
        nse = r_squared  # Same formula when residual = obs - sim

        # Percent bias
        pbias = float(100 * np.sum(res) / np.sum(obs)) if np.sum(obs) != 0 else 0.0

        return {
            "n_observations": len(obs),
            "rmse": rmse,
            "mae": mae,
            "bias": bias,
            "r_squared": max(r_squared, 0.0),
            "nse": nse,
            "pbias": pbias,
        }


class PestPostProcessor:
    """Post-processor for PEST++ calibration results.

    Loads and analyzes results from pestpp-glm, pestpp-ies, and
    pestpp-sen output files.

    Parameters
    ----------
    pest_dir : Path | str
        Directory containing PEST++ output files.
    case_name : str
        PEST++ case name (base name of .pst file).

    Examples
    --------
    >>> pp = PestPostProcessor("pest_output", "c2vsim_cal")
    >>> results = pp.load_results()
    >>> stats = results.fit_statistics()
    """

    def __init__(
        self,
        pest_dir: Path | str,
        case_name: str,
    ):
        """Initialize the post-processor.

        Parameters
        ----------
        pest_dir : Path | str
            PEST++ output directory.
        case_name : str
            Case name.
        """
        self.pest_dir = Path(pest_dir)
        self.case_name = case_name

    def load_results(self) -> CalibrationResults:
        """Load all available PEST++ results.

        Returns
        -------
        CalibrationResults
            Calibration results summary.
        """
        results = CalibrationResults(case_name=self.case_name)

        # Try to load residuals
        residuals = self._load_residuals()
        if residuals is not None:
            results.residuals = residuals
            results.final_phi = residuals.phi

        # Try to load sensitivities
        sensitivities = self._load_sensitivities()
        if sensitivities is not None:
            results.sensitivities = sensitivities

        # Try to load iteration history
        iteration_phi = self._load_iteration_history()
        if iteration_phi:
            results.iteration_phi = iteration_phi
            results.n_iterations = len(iteration_phi)

        # Try to load calibrated parameter values
        cal_values = self._load_calibrated_parameters()
        if cal_values:
            results.calibrated_values = cal_values

        return results

    def _load_residuals(self) -> ResidualData | None:
        """Load residual file (.rei or .res).

        Returns
        -------
        ResidualData | None
            Residual data, or None if file not found.
        """
        # Try .rei first, then .res
        for ext in [".rei", ".res"]:
            filepath = self.pest_dir / f"{self.case_name}{ext}"
            if filepath.exists():
                return self._parse_residual_file(filepath)
        return None

    def _parse_residual_file(self, filepath: Path) -> ResidualData:
        """Parse a PEST++ residual file.

        Parameters
        ----------
        filepath : Path
            Path to residual file.

        Returns
        -------
        ResidualData
            Parsed residual data.
        """
        lines = filepath.read_text().strip().split("\n")

        names = []
        groups = []
        observed = []
        simulated = []
        residuals = []
        weights = []

        # Skip header line(s)
        data_start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("Name") or line.strip().startswith("name"):
                data_start = i + 1
                break

        for line in lines[data_start:]:
            parts = line.split()
            if len(parts) >= 6:
                names.append(parts[0])
                groups.append(parts[1])
                observed.append(float(parts[2]))
                simulated.append(float(parts[3]))
                residuals.append(float(parts[4]))
                weights.append(float(parts[5]))

        return ResidualData(
            names=names,
            groups=groups,
            observed=np.array(observed),
            simulated=np.array(simulated),
            residuals=np.array(residuals),
            weights=np.array(weights),
        )

    def _load_sensitivities(self) -> SensitivityData | None:
        """Load sensitivity file (.sen).

        Returns
        -------
        SensitivityData | None
            Sensitivity data, or None if file not found.
        """
        filepath = self.pest_dir / f"{self.case_name}.sen"
        if not filepath.exists():
            return None

        lines = filepath.read_text().strip().split("\n")

        param_names = []
        sensitivities = []

        # Skip header
        data_start = 0
        for i, line in enumerate(lines):
            stripped = line.strip().lower()
            if stripped.startswith("parameter") or stripped.startswith("name"):
                data_start = i + 1
                break

        for line in lines[data_start:]:
            parts = line.split()
            if len(parts) >= 2:
                param_names.append(parts[0])
                sensitivities.append(float(parts[-1]))

        if param_names:
            return SensitivityData(
                parameter_names=param_names,
                composite_sensitivities=np.array(sensitivities),
            )
        return None

    def _load_iteration_history(self) -> list[float]:
        """Load iteration objective function history.

        Returns
        -------
        list[float]
            Objective function values per iteration.
        """
        filepath = self.pest_dir / f"{self.case_name}.iobj"
        if not filepath.exists():
            return []

        lines = filepath.read_text().strip().split("\n")
        phi_values = []

        for line in lines[1:]:  # Skip header
            parts = line.split()
            if len(parts) >= 2:
                try:
                    phi_values.append(float(parts[1]))
                except ValueError:
                    continue

        return phi_values

    def _load_calibrated_parameters(self) -> dict[str, float]:
        """Load calibrated parameter values (.par file).

        Returns
        -------
        dict[str, float]
            Parameter name -> calibrated value mapping.
        """
        filepath = self.pest_dir / f"{self.case_name}.par"
        if not filepath.exists():
            return {}

        lines = filepath.read_text().strip().split("\n")
        values = {}

        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    name = parts[0]
                    value = float(parts[1])
                    values[name] = value
                except (ValueError, IndexError):
                    continue

        return values

    def export_calibrated_parameters(
        self,
        filepath: Path | str,
        format: str = "csv",
    ) -> Path:
        """Export calibrated parameter values.

        Parameters
        ----------
        filepath : Path | str
            Output file path.
        format : str
            Output format: "csv" or "pest".

        Returns
        -------
        Path
            Path to written file.
        """
        filepath = Path(filepath)
        results = self.load_results()

        if format == "csv":
            lines = ["parameter_name,calibrated_value"]
            for name, value in results.calibrated_values.items():
                lines.append(f"{name},{value:.8e}")
            filepath.write_text("\n".join(lines))
        else:
            lines = ["single point"]
            for name, value in results.calibrated_values.items():
                lines.append(f"{name}  {value:.8e}  1.0  0.0")
            filepath.write_text("\n".join(lines))

        return filepath

    def compute_identifiability(self) -> dict[str, float] | None:
        """Compute parameter identifiability from sensitivity data.

        Returns identifiability index (0-1) for each parameter,
        where 1 means fully identifiable and 0 means unidentifiable.

        Returns
        -------
        dict[str, float] | None
            Parameter identifiability values, or None if data unavailable.
        """
        results = self.load_results()
        if results.sensitivities is None:
            return None

        sens = results.sensitivities.composite_sensitivities
        max_sens = np.max(sens) if np.max(sens) > 0 else 1.0

        return {
            name: float(s / max_sens)
            for name, s in zip(results.sensitivities.parameter_names, sens, strict=False)
        }

    def summary_report(self) -> str:
        """Generate a text summary report.

        Returns
        -------
        str
            Formatted summary report.
        """
        results = self.load_results()
        lines = []
        lines.append(f"PEST++ Calibration Report: {self.case_name}")
        lines.append("=" * 60)
        lines.append("")

        if results.n_iterations > 0:
            lines.append(f"Iterations completed: {results.n_iterations}")
            if results.iteration_phi:
                lines.append(f"Initial phi: {results.iteration_phi[0]:.4e}")
                lines.append(f"Final phi: {results.iteration_phi[-1]:.4e}")
                reduction = (
                    1.0 - results.iteration_phi[-1] / results.iteration_phi[0]
                    if results.iteration_phi[0] > 0
                    else 0.0
                )
                lines.append(f"Phi reduction: {reduction:.1%}")
            lines.append("")

        if results.residuals:
            lines.append("Fit Statistics (all observations):")
            lines.append("-" * 40)
            stats = results.fit_statistics()
            for key, val in stats.items():
                lines.append(
                    f"  {key:20s}: {val:.4f}" if isinstance(val, float) else f"  {key:20s}: {val}"
                )

            # Per-group statistics
            lines.append("")
            lines.append("Fit Statistics by Group:")
            lines.append("-" * 40)
            group_phi = results.residuals.group_phi()
            for group, phi in sorted(group_phi.items()):
                grp_stats = results.fit_statistics(group)
                lines.append(f"  {group}: phi={phi:.4e}, RMSE={grp_stats.get('rmse', 0):.4f}")

        if results.sensitivities:
            lines.append("")
            lines.append("Top 10 Most Sensitive Parameters:")
            lines.append("-" * 40)
            for name, sens in results.sensitivities.most_sensitive(10):
                lines.append(f"  {name:20s}: {sens:.4e}")

        if results.calibrated_values:
            lines.append("")
            lines.append(f"Calibrated Parameters ({len(results.calibrated_values)}):")
            lines.append("-" * 40)
            for name, val in list(results.calibrated_values.items())[:20]:
                lines.append(f"  {name:20s}: {val:.6e}")
            if len(results.calibrated_values) > 20:
                lines.append(f"  ... and {len(results.calibrated_values) - 20} more")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"PestPostProcessor(case='{self.case_name}', dir='{self.pest_dir}')"
