"""Ensemble management for PEST++ IES workflows.

This module provides the IWFMEnsembleManager class for generating,
writing, and analyzing parameter and observation ensembles for
pestpp-ies (Iterative Ensemble Smoother) workflows.

Features:
- Prior parameter ensemble generation (LHS, Gaussian, uniform)
- Spatially correlated ensembles via geostatistics
- Observation noise ensemble generation
- Ensemble file I/O (CSV format for PEST++)
- Posterior ensemble loading and analysis
- Ensemble statistics computation
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pyiwfm.runner.pest_params import Parameter
from pyiwfm.runner.pest_geostat import Variogram, GeostatManager


@dataclass
class EnsembleStatistics:
    """Summary statistics for a parameter ensemble.

    Attributes
    ----------
    mean : NDArray
        Mean values for each parameter.
    std : NDArray
        Standard deviations for each parameter.
    median : NDArray
        Median values for each parameter.
    q05 : NDArray
        5th percentile for each parameter.
    q95 : NDArray
        95th percentile for each parameter.
    n_realizations : int
        Number of realizations in the ensemble.
    n_parameters : int
        Number of parameters.
    parameter_names : list[str]
        Parameter names.
    """

    mean: NDArray
    std: NDArray
    median: NDArray
    q05: NDArray
    q95: NDArray
    n_realizations: int
    n_parameters: int
    parameter_names: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_realizations": self.n_realizations,
            "n_parameters": self.n_parameters,
            "parameters": {
                name: {
                    "mean": float(self.mean[i]),
                    "std": float(self.std[i]),
                    "median": float(self.median[i]),
                    "q05": float(self.q05[i]),
                    "q95": float(self.q95[i]),
                }
                for i, name in enumerate(self.parameter_names)
            },
        }


class IWFMEnsembleManager:
    """Manages ensemble generation and analysis for IWFM PEST++ workflows.

    This class provides methods for creating prior parameter ensembles,
    observation noise ensembles, and analyzing posterior ensembles from
    pestpp-ies runs.

    Parameters
    ----------
    parameters : list[Parameter]
        List of parameters for ensemble generation.
    geostat : GeostatManager | None
        Geostatistical manager for spatially correlated ensembles.

    Examples
    --------
    >>> em = IWFMEnsembleManager(parameters=params)
    >>> prior = em.generate_prior_ensemble(n_realizations=100)
    >>> em.write_parameter_ensemble(prior, "prior_ensemble.csv")
    """

    def __init__(
        self,
        parameters: list[Parameter] | None = None,
        geostat: GeostatManager | None = None,
    ):
        """Initialize the ensemble manager.

        Parameters
        ----------
        parameters : list[Parameter] | None
            List of parameters.
        geostat : GeostatManager | None
            Geostatistical manager.
        """
        self._parameters = parameters or []
        self._geostat = geostat or GeostatManager()
        self._prior_ensemble: NDArray | None = None
        self._posterior_ensemble: NDArray | None = None

    @property
    def parameter_names(self) -> list[str]:
        """Get parameter names."""
        return [p.name for p in self._parameters]

    @property
    def n_parameters(self) -> int:
        """Number of parameters."""
        return len(self._parameters)

    def generate_prior_ensemble(
        self,
        n_realizations: int = 100,
        method: str = "lhs",
        variogram: Variogram | None = None,
        seed: int | None = None,
    ) -> NDArray:
        """Generate prior parameter ensemble.

        For parameters with spatial locations (pilot points), generates
        spatially correlated realizations using the variogram. For
        non-spatial parameters, uses Latin Hypercube Sampling or
        uniform random sampling.

        Parameters
        ----------
        n_realizations : int
            Number of ensemble members.
        method : str
            Sampling method: "lhs" (Latin Hypercube), "uniform", "gaussian".
        variogram : Variogram | None
            Variogram for spatial correlation.
        seed : int | None
            Random seed for reproducibility.

        Returns
        -------
        NDArray
            Ensemble array (n_realizations x n_parameters).
        """
        if not self._parameters:
            raise ValueError("No parameters defined for ensemble generation")

        ensemble = self._geostat.generate_prior_ensemble(
            parameters=self._parameters,
            n_realizations=n_realizations,
            variogram=variogram,
            seed=seed,
            method=method,
        )

        self._prior_ensemble = ensemble
        return ensemble

    def generate_observation_ensemble(
        self,
        observation_values: NDArray,
        observation_weights: NDArray,
        n_realizations: int = 100,
        noise_type: str = "gaussian",
        seed: int | None = None,
    ) -> NDArray:
        """Generate observation noise ensemble.

        Adds noise to observation values based on their weights
        (weight = 1/standard_deviation).

        Parameters
        ----------
        observation_values : NDArray
            Observed values.
        observation_weights : NDArray
            Observation weights (inverse of std deviation).
        n_realizations : int
            Number of ensemble members.
        noise_type : str
            Noise distribution: "gaussian".
        seed : int | None
            Random seed.

        Returns
        -------
        NDArray
            Observation ensemble (n_realizations x n_observations).
        """
        if seed is not None:
            np.random.seed(seed)

        n_obs = len(observation_values)

        # Standard deviations from weights
        # Weight = 1/std, so std = 1/weight
        stds = np.where(observation_weights > 0, 1.0 / observation_weights, 0.0)

        # Generate noise
        if noise_type == "gaussian":
            noise = np.random.randn(n_realizations, n_obs)
        else:
            noise = np.random.randn(n_realizations, n_obs)

        # Scale noise by standard deviations
        ensemble = observation_values[np.newaxis, :] + noise * stds[np.newaxis, :]

        return ensemble

    def write_parameter_ensemble(
        self,
        ensemble: NDArray,
        filepath: Path | str,
    ) -> Path:
        """Write parameter ensemble to CSV file.

        Format compatible with PEST++ pestpp-ies input.

        Parameters
        ----------
        ensemble : NDArray
            Ensemble array (n_realizations x n_parameters).
        filepath : Path | str
            Output file path.

        Returns
        -------
        Path
            Path to written file.
        """
        filepath = Path(filepath)

        # Header: real_name, param1, param2, ...
        header = ["real_name"] + self.parameter_names

        lines = [",".join(header)]
        for i in range(ensemble.shape[0]):
            row = [f"r{i:04d}"] + [f"{v:.8e}" for v in ensemble[i, :]]
            lines.append(",".join(row))

        filepath.write_text("\n".join(lines))
        return filepath

    def write_observation_ensemble(
        self,
        ensemble: NDArray,
        observation_names: list[str],
        filepath: Path | str,
    ) -> Path:
        """Write observation ensemble to CSV file.

        Parameters
        ----------
        ensemble : NDArray
            Observation ensemble (n_realizations x n_observations).
        observation_names : list[str]
            Observation names.
        filepath : Path | str
            Output file path.

        Returns
        -------
        Path
            Path to written file.
        """
        filepath = Path(filepath)

        header = ["real_name"] + observation_names
        lines = [",".join(header)]
        for i in range(ensemble.shape[0]):
            row = [f"r{i:04d}"] + [f"{v:.8e}" for v in ensemble[i, :]]
            lines.append(",".join(row))

        filepath.write_text("\n".join(lines))
        return filepath

    def load_posterior_ensemble(
        self,
        filepath: Path | str,
    ) -> NDArray:
        """Load posterior parameter ensemble from pestpp-ies output.

        Parameters
        ----------
        filepath : Path | str
            Path to posterior ensemble CSV file.

        Returns
        -------
        NDArray
            Posterior ensemble (n_realizations x n_parameters).

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Posterior ensemble file not found: {filepath}")

        lines = filepath.read_text().strip().split("\n")
        header = lines[0].split(",")

        # Parse data rows
        data_rows = []
        for line in lines[1:]:
            parts = line.split(",")
            # Skip the real_name column
            values = [float(v) for v in parts[1:]]
            data_rows.append(values)

        self._posterior_ensemble = np.array(data_rows)
        return self._posterior_ensemble

    def analyze_ensemble(
        self,
        ensemble: NDArray,
    ) -> EnsembleStatistics:
        """Compute ensemble statistics.

        Parameters
        ----------
        ensemble : NDArray
            Ensemble array (n_realizations x n_parameters).

        Returns
        -------
        EnsembleStatistics
            Computed statistics.
        """
        return EnsembleStatistics(
            mean=np.mean(ensemble, axis=0),
            std=np.std(ensemble, axis=0),
            median=np.median(ensemble, axis=0),
            q05=np.percentile(ensemble, 5, axis=0),
            q95=np.percentile(ensemble, 95, axis=0),
            n_realizations=ensemble.shape[0],
            n_parameters=ensemble.shape[1],
            parameter_names=self.parameter_names[:ensemble.shape[1]],
        )

    def compute_reduction_factor(
        self,
        prior: NDArray,
        posterior: NDArray,
    ) -> NDArray:
        """Compute uncertainty reduction factor.

        Measures how much the posterior uncertainty is reduced
        relative to the prior.

        Parameters
        ----------
        prior : NDArray
            Prior ensemble.
        posterior : NDArray
            Posterior ensemble.

        Returns
        -------
        NDArray
            Reduction factor per parameter (0 = no reduction, 1 = complete).
        """
        prior_std = np.std(prior, axis=0)
        posterior_std = np.std(posterior, axis=0)

        # Avoid division by zero
        reduction = np.where(
            prior_std > 0,
            1.0 - posterior_std / prior_std,
            0.0,
        )
        return np.clip(reduction, 0.0, 1.0)

    def get_best_realization(
        self,
        ensemble: NDArray,
        objective_values: NDArray,
    ) -> tuple[int, NDArray]:
        """Get the best realization based on objective function.

        Parameters
        ----------
        ensemble : NDArray
            Parameter ensemble.
        objective_values : NDArray
            Objective function values for each realization.

        Returns
        -------
        tuple[int, NDArray]
            Index and parameter values of best realization.
        """
        best_idx = int(np.argmin(objective_values))
        return best_idx, ensemble[best_idx, :]

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"IWFMEnsembleManager(n_parameters={self.n_parameters}, "
            f"has_prior={self._prior_ensemble is not None}, "
            f"has_posterior={self._posterior_ensemble is not None})"
        )
