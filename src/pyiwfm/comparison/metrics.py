"""
Comparison metrics for IWFM model outputs.

This module provides functions and classes for computing comparison metrics
between observed and simulated data, commonly used for model calibration
and validation.

Metric Functions
----------------
- :func:`rmse`: Root Mean Square Error
- :func:`mae`: Mean Absolute Error
- :func:`mbe`: Mean Bias Error
- :func:`nash_sutcliffe`: Nash-Sutcliffe Efficiency (NSE)
- :func:`percent_bias`: Percent Bias (PBIAS)
- :func:`correlation_coefficient`: Pearson correlation
- :func:`max_error`: Maximum absolute error
- :func:`scaled_rmse`: Scaled RMSE (dimensionless)
- :func:`index_of_agreement`: Willmott Index of Agreement

Classes
-------
- :class:`ComparisonMetrics`: Container for all metrics
- :class:`TimeSeriesComparison`: Compare time series data
- :class:`SpatialComparison`: Compare spatial fields

Example
-------
Compute metrics for head comparison:

>>> import numpy as np
>>> from pyiwfm.comparison.metrics import ComparisonMetrics, rmse
>>>
>>> observed = np.array([50.0, 52.0, 48.0, 55.0, 51.0])
>>> simulated = np.array([51.0, 51.5, 49.0, 54.0, 52.0])
>>>
>>> # Individual metrics
>>> print(f"RMSE: {rmse(observed, simulated):.3f}")
RMSE: 0.894
>>>
>>> # All metrics at once
>>> metrics = ComparisonMetrics.compute(observed, simulated)
>>> print(metrics.summary())
>>> print(f"Model rating: {metrics.rating()}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


def rmse(
    observed: NDArray[np.float64],
    simulated: NDArray[np.float64],
) -> float:
    """
    Calculate Root Mean Square Error.

    Args:
        observed: Observed values
        simulated: Simulated values

    Returns:
        RMSE value
    """
    diff = simulated - observed
    return float(np.sqrt(np.mean(diff**2)))


def mae(
    observed: NDArray[np.float64],
    simulated: NDArray[np.float64],
) -> float:
    """
    Calculate Mean Absolute Error.

    Args:
        observed: Observed values
        simulated: Simulated values

    Returns:
        MAE value
    """
    return float(np.mean(np.abs(simulated - observed)))


def mbe(
    observed: NDArray[np.float64],
    simulated: NDArray[np.float64],
) -> float:
    """
    Calculate Mean Bias Error.

    Positive values indicate over-prediction,
    negative values indicate under-prediction.

    Args:
        observed: Observed values
        simulated: Simulated values

    Returns:
        MBE value
    """
    return float(np.mean(simulated - observed))


def nash_sutcliffe(
    observed: NDArray[np.float64],
    simulated: NDArray[np.float64],
) -> float:
    """
    Calculate Nash-Sutcliffe Efficiency.

    NSE = 1 - [sum((obs - sim)^2) / sum((obs - mean(obs))^2)]

    Values range from -inf to 1.0:
    - NSE = 1: Perfect model
    - NSE = 0: Model is as good as using mean observed value
    - NSE < 0: Model is worse than using mean

    Args:
        observed: Observed values
        simulated: Simulated values

    Returns:
        NSE value
    """
    numerator = np.sum((observed - simulated) ** 2)
    denominator = np.sum((observed - np.mean(observed)) ** 2)

    if denominator == 0:
        return 1.0 if numerator == 0 else -np.inf

    return float(1.0 - numerator / denominator)


def percent_bias(
    observed: NDArray[np.float64],
    simulated: NDArray[np.float64],
) -> float:
    """
    Calculate Percent Bias.

    PBIAS = 100 * [sum(sim - obs) / sum(obs)]

    Positive values indicate over-prediction,
    negative values indicate under-prediction.

    Args:
        observed: Observed values
        simulated: Simulated values

    Returns:
        Percent bias value
    """
    return float(100.0 * np.sum(simulated - observed) / np.sum(observed))


def correlation_coefficient(
    observed: NDArray[np.float64],
    simulated: NDArray[np.float64],
) -> float:
    """
    Calculate Pearson correlation coefficient.

    Args:
        observed: Observed values
        simulated: Simulated values

    Returns:
        Correlation coefficient (-1 to 1)
    """
    return float(np.corrcoef(observed, simulated)[0, 1])


def relative_error(
    observed: NDArray[np.float64],
    simulated: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Calculate relative error at each point.

    Args:
        observed: Observed values
        simulated: Simulated values

    Returns:
        Array of relative errors
    """
    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = (simulated - observed) / observed
        rel_err = np.where(observed == 0, 0.0, rel_err)
    return rel_err


def max_error(
    observed: NDArray[np.float64],
    simulated: NDArray[np.float64],
) -> float:
    """
    Calculate maximum absolute error.

    Args:
        observed: Observed values
        simulated: Simulated values

    Returns:
        Maximum absolute error
    """
    return float(np.max(np.abs(simulated - observed)))


def scaled_rmse(
    observed: NDArray[np.float64],
    simulated: NDArray[np.float64],
) -> float:
    """
    Calculate Scaled Root Mean Square Error.

    SRMSE = RMSE / (max(obs) - min(obs))

    A dimensionless metric that allows comparison across sites with
    different magnitudes. Values closer to 0 indicate better fit.

    Args:
        observed: Observed values
        simulated: Simulated values

    Returns:
        Scaled RMSE value. Returns inf if observed range is zero.
    """
    obs_range = float(np.max(observed) - np.min(observed))
    if obs_range == 0.0:
        return float("inf")
    return rmse(observed, simulated) / obs_range


def index_of_agreement(
    observed: NDArray[np.float64],
    simulated: NDArray[np.float64],
) -> float:
    """
    Calculate Willmott Index of Agreement (d).

    d = 1 - [sum((sim - obs)^2) / sum((|sim - mean(obs)| + |obs - mean(obs)|)^2)]

    Values range from 0 to 1.0:
    - d = 1: Perfect agreement
    - d = 0: No agreement

    Reference: Willmott, C. J. (1981). On the validation of models.
    Physical Geography, 2(2), 184-194.

    Args:
        observed: Observed values
        simulated: Simulated values

    Returns:
        Index of agreement (0 to 1)
    """
    obs_mean = np.mean(observed)
    numerator = np.sum((simulated - observed) ** 2)
    denominator = np.sum((np.abs(simulated - obs_mean) + np.abs(observed - obs_mean)) ** 2)

    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else 0.0

    return float(1.0 - numerator / denominator)


@dataclass
class ComparisonMetrics:
    """
    Container for all comparison metrics.

    Attributes:
        rmse: Root Mean Square Error
        mae: Mean Absolute Error
        mbe: Mean Bias Error
        nash_sutcliffe: Nash-Sutcliffe Efficiency
        percent_bias: Percent Bias
        correlation: Pearson correlation coefficient
        max_error: Maximum absolute error
        scaled_rmse: Scaled RMSE (dimensionless)
        index_of_agreement: Willmott Index of Agreement
        n_points: Number of data points
    """

    rmse: float
    mae: float
    mbe: float
    nash_sutcliffe: float
    percent_bias: float
    correlation: float
    max_error: float
    scaled_rmse: float
    index_of_agreement: float
    n_points: int

    @classmethod
    def compute(
        cls,
        observed: NDArray[np.float64],
        simulated: NDArray[np.float64],
    ) -> ComparisonMetrics:
        """
        Compute all metrics from observed and simulated data.

        Args:
            observed: Observed values
            simulated: Simulated values

        Returns:
            ComparisonMetrics instance with all metrics computed
        """
        # Remove any NaN values
        mask = ~(np.isnan(observed) | np.isnan(simulated))
        obs_clean = observed[mask]
        sim_clean = simulated[mask]

        return cls(
            rmse=rmse(obs_clean, sim_clean),
            mae=mae(obs_clean, sim_clean),
            mbe=mbe(obs_clean, sim_clean),
            nash_sutcliffe=nash_sutcliffe(obs_clean, sim_clean),
            percent_bias=percent_bias(obs_clean, sim_clean),
            correlation=correlation_coefficient(obs_clean, sim_clean),
            max_error=max_error(obs_clean, sim_clean),
            scaled_rmse=scaled_rmse(obs_clean, sim_clean),
            index_of_agreement=index_of_agreement(obs_clean, sim_clean),
            n_points=len(obs_clean),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert metrics to dictionary.

        Returns:
            Dictionary with all metrics
        """
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "mbe": self.mbe,
            "nash_sutcliffe": self.nash_sutcliffe,
            "percent_bias": self.percent_bias,
            "correlation": self.correlation,
            "max_error": self.max_error,
            "scaled_rmse": self.scaled_rmse,
            "index_of_agreement": self.index_of_agreement,
            "n_points": self.n_points,
        }

    def summary(self) -> str:
        """
        Generate a human-readable summary.

        Returns:
            Summary string
        """
        lines = [
            "Comparison Metrics",
            "=" * 30,
            f"RMSE:            {self.rmse:.4f}",
            f"Scaled RMSE:     {self.scaled_rmse:.4f}",
            f"MAE:             {self.mae:.4f}",
            f"MBE:             {self.mbe:.4f}",
            f"Nash-Sutcliffe:  {self.nash_sutcliffe:.4f}",
            f"Percent Bias:    {self.percent_bias:.2f}%",
            f"Correlation:     {self.correlation:.4f}",
            f"Max Error:       {self.max_error:.4f}",
            f"Index of Agr.:   {self.index_of_agreement:.4f}",
            f"N Points:        {self.n_points}",
            f"Rating:          {self.rating()}",
        ]
        return "\n".join(lines)

    def rating(self) -> str:
        """
        Provide a qualitative rating based on NSE.

        Returns:
            Rating string ('excellent', 'good', 'fair', 'poor')
        """
        nse = self.nash_sutcliffe
        if nse >= 0.90:
            return "excellent"
        elif nse >= 0.65:
            return "good"
        elif nse >= 0.50:
            return "fair"
        else:
            return "poor"


@dataclass
class TimeSeriesComparison:
    """
    Compare time series data.

    Attributes:
        times: Time values
        observed: Observed values
        simulated: Simulated values
    """

    times: NDArray[np.float64]
    observed: NDArray[np.float64]
    simulated: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Compute metrics after initialization."""
        self._metrics: ComparisonMetrics | None = None

    @property
    def metrics(self) -> ComparisonMetrics:
        """Get comparison metrics."""
        if self._metrics is None:
            self._metrics = ComparisonMetrics.compute(self.observed, self.simulated)
        return self._metrics

    @property
    def n_points(self) -> int:
        """Total number of time points."""
        return len(self.times)

    @property
    def n_valid_points(self) -> int:
        """Number of valid (non-NaN) time points."""
        mask = ~(np.isnan(self.observed) | np.isnan(self.simulated))
        return int(np.sum(mask))

    @property
    def residuals(self) -> NDArray[np.float64]:
        """Calculate residuals (simulated - observed)."""
        return self.simulated - self.observed

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "n_points": self.n_points,
            "n_valid_points": self.n_valid_points,
            "metrics": self.metrics.to_dict(),
        }


@dataclass
class SpatialComparison:
    """
    Compare spatial field data.

    Attributes:
        x: X coordinates
        y: Y coordinates
        observed: Observed values at each point
        simulated: Simulated values at each point
    """

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    observed: NDArray[np.float64]
    simulated: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Compute metrics after initialization."""
        self._metrics: ComparisonMetrics | None = None

    @property
    def metrics(self) -> ComparisonMetrics:
        """Get comparison metrics."""
        if self._metrics is None:
            self._metrics = ComparisonMetrics.compute(self.observed, self.simulated)
        return self._metrics

    @property
    def n_points(self) -> int:
        """Total number of spatial points."""
        return len(self.x)

    @property
    def error_field(self) -> NDArray[np.float64]:
        """Calculate error at each point (simulated - observed)."""
        return self.simulated - self.observed

    @property
    def relative_error_field(self) -> NDArray[np.float64]:
        """Calculate relative error at each point."""
        return relative_error(self.observed, self.simulated)

    def metrics_by_region(
        self,
        regions: NDArray[np.int32],
    ) -> dict[int, ComparisonMetrics]:
        """
        Calculate metrics for each region.

        Args:
            regions: Region ID for each point

        Returns:
            Dictionary mapping region ID to metrics
        """
        unique_regions = np.unique(regions)
        result = {}

        for region_id in unique_regions:
            mask = regions == region_id
            obs_region = self.observed[mask]
            sim_region = self.simulated[mask]
            result[int(region_id)] = ComparisonMetrics.compute(obs_region, sim_region)

        return result

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "n_points": self.n_points,
            "metrics": self.metrics.to_dict(),
        }
