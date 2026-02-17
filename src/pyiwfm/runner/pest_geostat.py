"""Geostatistics module for PEST++ pilot point parameterization.

This module provides geostatistical tools for:
- Variogram modeling (spherical, exponential, gaussian, matern)
- Kriging interpolation (ordinary, simple)
- Covariance matrix computation
- Geostatistical realization generation
- Prior ensemble generation with spatial correlation

These tools support highly parameterized inversion with pilot points,
where parameter values at scattered pilot points are interpolated to
model nodes using kriging.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    from scipy import linalg
    from scipy.optimize import curve_fit
    from scipy.spatial.distance import cdist

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class VariogramType(Enum):
    """Types of variogram models."""

    SPHERICAL = "spherical"
    EXPONENTIAL = "exponential"
    GAUSSIAN = "gaussian"
    MATERN = "matern"
    LINEAR = "linear"
    POWER = "power"
    NUGGET = "nugget"


@dataclass
class Variogram:
    """Variogram model for geostatistical analysis.

    A variogram describes the spatial correlation structure of a variable.
    It quantifies how dissimilarity between values increases with distance.

    Parameters
    ----------
    variogram_type : str | VariogramType
        Type of variogram model.
    a : float
        Range parameter - distance at which correlation approaches zero.
    sill : float
        Sill - variance at which the variogram levels off.
    nugget : float
        Nugget effect - discontinuity at the origin (measurement error + micro-scale variation).
    anisotropy_ratio : float
        Ratio of major to minor range (1.0 = isotropic).
    anisotropy_angle : float
        Angle of major axis in degrees from east (counterclockwise).
    power : float
        Power parameter for power variogram model.

    Examples
    --------
    >>> # Exponential variogram with range=10000m, sill=1.0
    >>> vario = Variogram("exponential", a=10000, sill=1.0, nugget=0.1)
    >>> # Evaluate at distances
    >>> gamma = vario.evaluate(np.array([0, 1000, 5000, 10000]))
    """

    variogram_type: str | VariogramType
    a: float  # range
    sill: float = 1.0
    nugget: float = 0.0
    anisotropy_ratio: float = 1.0
    anisotropy_angle: float = 0.0
    power: float = 1.0  # for power model

    def __post_init__(self) -> None:
        """Validate and convert variogram type."""
        if isinstance(self.variogram_type, str):
            self.variogram_type = VariogramType(self.variogram_type)

        if self.a <= 0:
            raise ValueError(f"Range must be positive: {self.a}")
        if self.sill < 0:
            raise ValueError(f"Sill must be non-negative: {self.sill}")
        if self.nugget < 0:
            raise ValueError(f"Nugget must be non-negative: {self.nugget}")
        if self.anisotropy_ratio <= 0:
            raise ValueError(f"Anisotropy ratio must be positive: {self.anisotropy_ratio}")

    @property
    def total_sill(self) -> float:
        """Total sill (nugget + partial sill)."""
        return self.nugget + self.sill

    @property
    def effective_range(self) -> float:
        """Effective range (distance at 95% of sill).

        For exponential: ~3*a
        For gaussian: ~sqrt(3)*a
        For spherical: a
        """
        if self.variogram_type == VariogramType.EXPONENTIAL:
            return float(3.0 * self.a)
        elif self.variogram_type == VariogramType.GAUSSIAN:
            return float(np.sqrt(3.0) * self.a)
        return self.a

    def evaluate(self, h: NDArray | float) -> NDArray | float:
        """Evaluate variogram at lag distances h.

        Parameters
        ----------
        h : NDArray | float
            Lag distance(s).

        Returns
        -------
        NDArray | float
            Variogram value(s) gamma(h).
        """
        h = np.asarray(h)
        scalar_input = h.ndim == 0
        h = np.atleast_1d(h)

        gamma = np.zeros_like(h, dtype=float)

        if self.variogram_type == VariogramType.SPHERICAL:
            gamma = self._spherical(h)
        elif self.variogram_type == VariogramType.EXPONENTIAL:
            gamma = self._exponential(h)
        elif self.variogram_type == VariogramType.GAUSSIAN:
            gamma = self._gaussian(h)
        elif self.variogram_type == VariogramType.MATERN:
            gamma = self._matern(h)
        elif self.variogram_type == VariogramType.LINEAR:
            gamma = self._linear(h)
        elif self.variogram_type == VariogramType.POWER:
            gamma = self._power(h)
        elif self.variogram_type == VariogramType.NUGGET:
            gamma = self._nugget_model(h)

        if scalar_input:
            return float(gamma[0])
        return gamma

    def _spherical(self, h: NDArray) -> NDArray:
        """Spherical variogram model."""
        hr = h / self.a
        gamma = np.where(
            h == 0,
            0.0,
            np.where(
                hr < 1,
                self.nugget + self.sill * (1.5 * hr - 0.5 * hr**3),
                self.nugget + self.sill,
            ),
        )
        return gamma

    def _exponential(self, h: NDArray) -> NDArray:
        """Exponential variogram model."""
        gamma = np.where(
            h == 0,
            0.0,
            self.nugget + self.sill * (1.0 - np.exp(-h / self.a)),
        )
        return gamma

    def _gaussian(self, h: NDArray) -> NDArray:
        """Gaussian variogram model."""
        gamma = np.where(
            h == 0,
            0.0,
            self.nugget + self.sill * (1.0 - np.exp(-((h / self.a) ** 2))),
        )
        return gamma

    def _matern(self, h: NDArray, nu: float = 1.5) -> NDArray:
        """Matern variogram model with smoothness parameter nu."""
        # Simplified Matern for nu=1.5 (common case)
        hr = h / self.a
        gamma = np.where(
            h == 0,
            0.0,
            self.nugget + self.sill * (1.0 - (1.0 + np.sqrt(3) * hr) * np.exp(-np.sqrt(3) * hr)),
        )
        return gamma

    def _linear(self, h: NDArray) -> NDArray:
        """Linear variogram model (unbounded)."""
        gamma = np.where(
            h == 0,
            0.0,
            self.nugget + self.sill * h / self.a,
        )
        return gamma

    def _power(self, h: NDArray) -> NDArray:
        """Power variogram model."""
        gamma = np.where(
            h == 0,
            0.0,
            self.nugget + self.sill * (h / self.a) ** self.power,
        )
        return gamma

    def _nugget_model(self, h: NDArray) -> NDArray:
        """Pure nugget model."""
        gamma = np.where(h == 0, 0.0, self.nugget + self.sill)
        return gamma

    def covariance(self, h: NDArray | float) -> NDArray | float:
        """Compute covariance from variogram.

        C(h) = sill - gamma(h) for stationary variograms.

        Parameters
        ----------
        h : NDArray | float
            Lag distance(s).

        Returns
        -------
        NDArray | float
            Covariance value(s).
        """
        return self.total_sill - self.evaluate(h)

    def transform_coordinates(
        self,
        x: NDArray,
        y: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """Transform coordinates for anisotropy.

        Parameters
        ----------
        x : NDArray
            X coordinates.
        y : NDArray
            Y coordinates.

        Returns
        -------
        tuple[NDArray, NDArray]
            Transformed coordinates.
        """
        if self.anisotropy_ratio == 1.0:
            return x, y

        # Rotate to align with major axis
        angle_rad = np.radians(self.anisotropy_angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        x_rot = x * cos_a + y * sin_a
        y_rot = -x * sin_a + y * cos_a

        # Scale minor axis
        y_scaled = y_rot * self.anisotropy_ratio

        return x_rot, y_scaled

    def compute_distance_matrix(
        self,
        x1: NDArray,
        y1: NDArray,
        x2: NDArray | None = None,
        y2: NDArray | None = None,
    ) -> NDArray:
        """Compute distance matrix with anisotropy.

        Parameters
        ----------
        x1, y1 : NDArray
            First set of coordinates.
        x2, y2 : NDArray | None
            Second set of coordinates. If None, compute self-distances.

        Returns
        -------
        NDArray
            Distance matrix.
        """
        # Transform for anisotropy
        x1_t, y1_t = self.transform_coordinates(x1, y1)
        coords1 = np.column_stack([x1_t, y1_t])

        if x2 is None or y2 is None:
            coords2 = coords1
        else:
            x2_t, y2_t = self.transform_coordinates(x2, y2)
            coords2 = np.column_stack([x2_t, y2_t])

        if HAS_SCIPY:
            return np.asarray(cdist(coords1, coords2))
        else:
            # Manual distance computation
            diff = coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :]
            return np.asarray(np.sqrt(np.sum(diff**2, axis=2)))

    @classmethod
    def from_data(
        cls,
        x: NDArray,
        y: NDArray,
        values: NDArray,
        variogram_type: str = "exponential",
        n_lags: int = 15,
        max_lag: float | None = None,
    ) -> Variogram:
        """Fit variogram to data using empirical variogram.

        Parameters
        ----------
        x, y : NDArray
            Coordinates of data points.
        values : NDArray
            Values at data points.
        variogram_type : str
            Type of variogram model to fit.
        n_lags : int
            Number of lag bins for empirical variogram.
        max_lag : float | None
            Maximum lag distance. If None, uses half of maximum distance.

        Returns
        -------
        Variogram
            Fitted variogram model.

        Raises
        ------
        ImportError
            If scipy is not available.
        """
        if not HAS_SCIPY:
            raise ImportError("scipy required for variogram fitting")

        # Compute empirical variogram
        coords = np.column_stack([x, y])
        distances = cdist(coords, coords)

        if max_lag is None:
            max_lag = np.max(distances) / 2.0

        # Compute lag bins
        lag_edges = np.linspace(0, max_lag, n_lags + 1)
        lag_centers = (lag_edges[:-1] + lag_edges[1:]) / 2

        # Compute semivariance for each lag
        n_points = len(values)
        gamma_emp = np.zeros(n_lags)
        counts = np.zeros(n_lags)

        for i in range(n_points):
            for j in range(i + 1, n_points):
                d = distances[i, j]
                bin_idx = np.searchsorted(lag_edges[1:], d)
                if bin_idx < n_lags:
                    gamma_emp[bin_idx] += 0.5 * (values[i] - values[j]) ** 2
                    counts[bin_idx] += 1

        # Average semivariance per lag
        valid = counts > 0
        gamma_emp[valid] /= counts[valid]

        # Fit variogram model
        valid_lags = lag_centers[valid]
        valid_gamma = gamma_emp[valid]

        # Initial guess
        sill_init = np.var(values)
        range_init = max_lag / 3.0
        nugget_init = 0.0

        # Define model function
        def model_func(h: NDArray, nugget: float, sill: float, a: float) -> NDArray | float:
            v = cls(variogram_type, a=a, sill=sill, nugget=nugget)
            return v.evaluate(h)

        try:
            popt, _ = curve_fit(
                model_func,
                valid_lags,
                valid_gamma,
                p0=[nugget_init, sill_init, range_init],
                bounds=([0, 0, 0], [sill_init * 2, sill_init * 2, max_lag * 2]),
                maxfev=5000,
            )
            nugget, sill, a = popt
        except RuntimeError:
            # Fall back to simple estimates
            nugget = 0.0
            sill = sill_init
            a = range_init

        return cls(
            variogram_type=variogram_type,
            a=a,
            sill=sill,
            nugget=nugget,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        vtype = self.variogram_type
        vtype_str = vtype.value if isinstance(vtype, VariogramType) else vtype
        return {
            "variogram_type": vtype_str,
            "a": self.a,
            "sill": self.sill,
            "nugget": self.nugget,
            "anisotropy_ratio": self.anisotropy_ratio,
            "anisotropy_angle": self.anisotropy_angle,
            "power": self.power,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Variogram:
        """Create from dictionary."""
        return cls(
            variogram_type=d["variogram_type"],
            a=d["a"],
            sill=d.get("sill", 1.0),
            nugget=d.get("nugget", 0.0),
            anisotropy_ratio=d.get("anisotropy_ratio", 1.0),
            anisotropy_angle=d.get("anisotropy_angle", 0.0),
            power=d.get("power", 1.0),
        )

    def __repr__(self) -> str:
        """Return string representation."""
        vtype = self.variogram_type
        vtype_str = vtype.value if isinstance(vtype, VariogramType) else vtype
        return (
            f"Variogram(type={vtype_str}, a={self.a:.2f}, "
            f"sill={self.sill:.4f}, nugget={self.nugget:.4f})"
        )


class GeostatManager:
    """Manages geostatistical operations for pilot point parameterization.

    This class provides methods for:
    - Computing covariance matrices between points
    - Kriging interpolation from pilot points to model nodes
    - Generating geostatistically correlated realizations
    - Writing kriging factors for use in PEST++ preprocessing

    Parameters
    ----------
    model : Any
        IWFM model instance (optional, for mesh information).

    Examples
    --------
    >>> gm = GeostatManager()
    >>> # Compute covariance matrix
    >>> cov = gm.compute_covariance_matrix(pp_x, pp_y, variogram)
    >>> # Krige to model nodes
    >>> node_values = gm.krige(pp_x, pp_y, pp_values, node_x, node_y, variogram)
    """

    def __init__(self, model: Any = None):
        """Initialize the geostat manager.

        Parameters
        ----------
        model : Any
            IWFM model instance (optional).
        """
        self.model = model

    def compute_covariance_matrix(
        self,
        x: NDArray,
        y: NDArray,
        variogram: Variogram,
    ) -> NDArray:
        """Compute covariance matrix between points.

        Parameters
        ----------
        x, y : NDArray
            Coordinates of points.
        variogram : Variogram
            Variogram model.

        Returns
        -------
        NDArray
            Covariance matrix (n x n).
        """
        distances = variogram.compute_distance_matrix(x, y)
        return np.asarray(variogram.covariance(distances))

    def compute_variogram_matrix(
        self,
        x: NDArray,
        y: NDArray,
        variogram: Variogram,
    ) -> NDArray:
        """Compute variogram matrix between points.

        Parameters
        ----------
        x, y : NDArray
            Coordinates of points.
        variogram : Variogram
            Variogram model.

        Returns
        -------
        NDArray
            Variogram matrix (n x n).
        """
        distances = variogram.compute_distance_matrix(x, y)
        return np.asarray(variogram.evaluate(distances))

    def krige(
        self,
        pilot_x: NDArray,
        pilot_y: NDArray,
        pilot_values: NDArray,
        target_x: NDArray,
        target_y: NDArray,
        variogram: Variogram,
        kriging_type: str = "ordinary",
        return_variance: bool = False,
    ) -> NDArray | tuple[NDArray, NDArray]:
        """Interpolate values using kriging.

        Parameters
        ----------
        pilot_x, pilot_y : NDArray
            Coordinates of pilot points.
        pilot_values : NDArray
            Values at pilot points.
        target_x, target_y : NDArray
            Coordinates of target points.
        variogram : Variogram
            Variogram model.
        kriging_type : str
            Type of kriging: "ordinary" or "simple".
        return_variance : bool
            If True, also return kriging variance.

        Returns
        -------
        NDArray | tuple[NDArray, NDArray]
            Interpolated values, optionally with variance.

        Raises
        ------
        ImportError
            If scipy is not available.
        """
        if not HAS_SCIPY:
            raise ImportError("scipy required for kriging")

        n_pilot = len(pilot_values)
        n_target = len(target_x)

        # Compute variogram matrices
        # C_pp: pilot-pilot covariance
        C_pp = self.compute_covariance_matrix(pilot_x, pilot_y, variogram)

        # C_pt: pilot-target covariance
        distances_pt = variogram.compute_distance_matrix(pilot_x, pilot_y, target_x, target_y)
        C_pt = np.asarray(variogram.covariance(distances_pt))

        if kriging_type == "ordinary":
            # Ordinary kriging: estimate mean from data
            # Augment system with Lagrange multiplier for unbiasedness

            # Build kriging matrix [C_pp, 1; 1', 0]
            K = np.zeros((n_pilot + 1, n_pilot + 1))
            K[:n_pilot, :n_pilot] = C_pp
            K[:n_pilot, n_pilot] = 1.0
            K[n_pilot, :n_pilot] = 1.0

            # Solve for weights at each target point
            target_values = np.zeros(n_target)
            target_variance = np.zeros(n_target)

            # Compute kriging weights
            try:
                K_inv = linalg.inv(K)
            except linalg.LinAlgError:
                # Add small regularization if singular
                K += np.eye(n_pilot + 1) * 1e-10
                K_inv = linalg.inv(K)

            for i in range(n_target):
                # Right-hand side: [C_pt[:, i]; 1]
                b = np.zeros(n_pilot + 1)
                b[:n_pilot] = C_pt[:, i]
                b[n_pilot] = 1.0

                # Weights
                w = K_inv @ b
                weights = w[:n_pilot]

                # Interpolated value
                target_values[i] = np.sum(weights * pilot_values)

                # Kriging variance
                if return_variance:
                    target_variance[i] = variogram.total_sill - np.sum(w * b)

        else:  # simple kriging
            # Simple kriging: known mean (assume 0 or subtract mean first)
            mean_val = np.mean(pilot_values)
            centered_values = pilot_values - mean_val

            try:
                C_pp_inv = linalg.inv(C_pp)
            except linalg.LinAlgError:
                C_pp += np.eye(n_pilot) * 1e-10
                C_pp_inv = linalg.inv(C_pp)

            target_values = np.zeros(n_target)
            target_variance = np.zeros(n_target)

            for i in range(n_target):
                weights = C_pp_inv @ C_pt[:, i]
                target_values[i] = mean_val + np.sum(weights * centered_values)

                if return_variance:
                    target_variance[i] = variogram.total_sill - np.sum(weights * C_pt[:, i])

        if return_variance:
            return target_values, np.maximum(target_variance, 0)
        return target_values

    def compute_kriging_factors(
        self,
        pilot_x: NDArray,
        pilot_y: NDArray,
        target_x: NDArray,
        target_y: NDArray,
        variogram: Variogram,
        kriging_type: str = "ordinary",
    ) -> NDArray:
        """Compute kriging interpolation factors.

        These factors can be saved and applied multiple times without
        re-solving the kriging system.

        Parameters
        ----------
        pilot_x, pilot_y : NDArray
            Coordinates of pilot points.
        target_x, target_y : NDArray
            Coordinates of target points.
        variogram : Variogram
            Variogram model.
        kriging_type : str
            Type of kriging.

        Returns
        -------
        NDArray
            Kriging factors matrix (n_target x n_pilot).
        """
        if not HAS_SCIPY:
            raise ImportError("scipy required for kriging factors")

        n_pilot = len(pilot_x)
        n_target = len(target_x)

        # Compute covariance matrices
        C_pp = self.compute_covariance_matrix(pilot_x, pilot_y, variogram)
        distances_pt = variogram.compute_distance_matrix(pilot_x, pilot_y, target_x, target_y)
        C_pt = np.asarray(variogram.covariance(distances_pt))

        factors = np.zeros((n_target, n_pilot))

        if kriging_type == "ordinary":
            # Build augmented system
            K = np.zeros((n_pilot + 1, n_pilot + 1))
            K[:n_pilot, :n_pilot] = C_pp
            K[:n_pilot, n_pilot] = 1.0
            K[n_pilot, :n_pilot] = 1.0

            try:
                K_inv = linalg.inv(K)
            except linalg.LinAlgError:
                K += np.eye(n_pilot + 1) * 1e-10
                K_inv = linalg.inv(K)

            for i in range(n_target):
                b = np.zeros(n_pilot + 1)
                b[:n_pilot] = C_pt[:, i]
                b[n_pilot] = 1.0
                w = K_inv @ b
                factors[i, :] = w[:n_pilot]

        else:  # simple kriging
            try:
                C_pp_inv = linalg.inv(C_pp)
            except linalg.LinAlgError:
                C_pp += np.eye(n_pilot) * 1e-10
                C_pp_inv = linalg.inv(C_pp)

            for i in range(n_target):
                factors[i, :] = C_pp_inv @ C_pt[:, i]

        return factors

    def generate_realizations(
        self,
        x: NDArray,
        y: NDArray,
        variogram: Variogram,
        n_realizations: int = 100,
        mean: float = 0.0,
        conditioning_data: tuple[NDArray, NDArray, NDArray] | None = None,
        seed: int | None = None,
    ) -> NDArray:
        """Generate geostatistical realizations.

        Generates spatially correlated random fields using the covariance
        structure defined by the variogram.

        Parameters
        ----------
        x, y : NDArray
            Coordinates where realizations are generated.
        variogram : Variogram
            Variogram model defining spatial correlation.
        n_realizations : int
            Number of realizations to generate.
        mean : float
            Mean value of the field.
        conditioning_data : tuple[NDArray, NDArray, NDArray] | None
            Optional conditioning data as (x, y, values).
        seed : int | None
            Random seed for reproducibility.

        Returns
        -------
        NDArray
            Realizations array (n_realizations x n_points).

        Raises
        ------
        ImportError
            If scipy is not available.
        """
        if not HAS_SCIPY:
            raise ImportError("scipy required for realization generation")

        rng = np.random.default_rng(seed)

        n_points = len(x)

        # Compute covariance matrix
        C = self.compute_covariance_matrix(x, y, variogram)

        # Add small regularization for numerical stability
        C += np.eye(n_points) * 1e-10

        # Cholesky decomposition
        try:
            L = linalg.cholesky(C, lower=True)
        except linalg.LinAlgError:
            # If Cholesky fails, use eigendecomposition
            eigvals, eigvecs = linalg.eigh(C)
            eigvals = np.maximum(eigvals, 0)
            L = eigvecs @ np.diag(np.sqrt(eigvals))

        # Generate unconditional realizations
        z = rng.standard_normal((n_points, n_realizations))
        realizations = mean + (L @ z).T

        # Condition if data provided
        if conditioning_data is not None:
            cond_x, cond_y, cond_values = conditioning_data

            # Krige conditioning data to realization points
            for i in range(n_realizations):
                # Krige the conditioning data
                self.krige(
                    cond_x,
                    cond_y,
                    cond_values,
                    x,
                    y,
                    variogram,
                    kriging_type="simple",
                )

                # Krige the simulated values at conditioning locations
                sim_at_cond = np.asarray(
                    self.krige(
                        x,
                        y,
                        realizations[i],
                        cond_x,
                        cond_y,
                        variogram,
                        kriging_type="simple",
                    )
                )

                # Condition: Y_s(x) = Y(x) + [Z_s(x_c) - Y(x_c)] kriged
                correction = np.asarray(
                    self.krige(
                        cond_x,
                        cond_y,
                        cond_values - sim_at_cond,
                        x,
                        y,
                        variogram,
                        kriging_type="simple",
                    )
                )
                realizations[i] = realizations[i] + correction

        return np.asarray(realizations)

    def generate_prior_ensemble(
        self,
        parameters: list,
        n_realizations: int = 100,
        variogram: Variogram | None = None,
        seed: int | None = None,
        method: str = "lhs",
    ) -> NDArray:
        """Generate prior parameter ensemble.

        For parameters with spatial locations (pilot points), generates
        spatially correlated realizations. For non-spatial parameters,
        uses Latin Hypercube Sampling or uniform sampling.

        Parameters
        ----------
        parameters : list
            List of Parameter objects.
        n_realizations : int
            Number of ensemble members.
        variogram : Variogram | None
            Variogram for spatial correlation. If None, assumes uncorrelated.
        seed : int | None
            Random seed.
        method : str
            Sampling method: "lhs" for Latin Hypercube, "uniform" for uniform.

        Returns
        -------
        NDArray
            Ensemble array (n_realizations x n_parameters).
        """
        rng = np.random.default_rng(seed)

        n_params = len(parameters)

        # Separate spatial and non-spatial parameters
        spatial_params = []
        spatial_indices = []
        non_spatial_params = []
        non_spatial_indices = []

        for i, p in enumerate(parameters):
            if p.location is not None:
                spatial_params.append(p)
                spatial_indices.append(i)
            else:
                non_spatial_params.append(p)
                non_spatial_indices.append(i)

        ensemble = np.zeros((n_realizations, n_params))

        # Generate spatially correlated realizations for pilot points
        if spatial_params and variogram is not None:
            x = np.array([p.location[0] for p in spatial_params])
            y = np.array([p.location[1] for p in spatial_params])

            # Generate in log space if parameters are log-transformed
            log_transform = any(p.transform == "log" for p in spatial_params)

            if log_transform:
                mean_val = np.mean([np.log10(p.initial_value) for p in spatial_params])
                realizations = self.generate_realizations(
                    x, y, variogram, n_realizations, mean=mean_val, seed=seed
                )
                # Transform back
                realizations = 10**realizations
            else:
                mean_val = np.mean([p.initial_value for p in spatial_params])
                realizations = self.generate_realizations(
                    x, y, variogram, n_realizations, mean=mean_val, seed=seed
                )

            # Clip to bounds
            for j, p in enumerate(spatial_params):
                realizations[:, j] = np.clip(realizations[:, j], p.lower_bound, p.upper_bound)
                ensemble[:, spatial_indices[j]] = realizations[:, j]

        # Generate non-spatial parameters
        if non_spatial_params:
            if method == "lhs":
                lhs_samples = self._latin_hypercube(
                    n_realizations, len(non_spatial_params), rng=rng
                )
            else:
                lhs_samples = rng.random((n_realizations, len(non_spatial_params)))

            for j, p in enumerate(non_spatial_params):
                if p.transform == "log":
                    log_lb = np.log10(p.lower_bound)
                    log_ub = np.log10(p.upper_bound)
                    values = 10 ** (log_lb + lhs_samples[:, j] * (log_ub - log_lb))
                else:
                    values = p.lower_bound + lhs_samples[:, j] * (p.upper_bound - p.lower_bound)

                ensemble[:, non_spatial_indices[j]] = values

        return ensemble

    def _latin_hypercube(self, n: int, d: int, rng: np.random.Generator | None = None) -> NDArray:
        """Generate Latin Hypercube samples.

        Parameters
        ----------
        n : int
            Number of samples.
        d : int
            Number of dimensions.
        rng : np.random.Generator | None
            Random number generator. If None, creates a new one.

        Returns
        -------
        NDArray
            LHS samples (n x d), values in [0, 1].
        """
        if rng is None:
            rng = np.random.default_rng()
        samples = np.zeros((n, d))
        for j in range(d):
            # Create random permutation of strata
            perm = rng.permutation(n)
            # Random position within each stratum
            samples[:, j] = (perm + rng.random(n)) / n
        return samples

    def write_kriging_factors(
        self,
        pilot_x: NDArray,
        pilot_y: NDArray,
        pilot_names: list[str],
        target_x: NDArray,
        target_y: NDArray,
        target_ids: list[int | str],
        variogram: Variogram,
        filepath: Path | str,
        kriging_type: str = "ordinary",
        format: str = "pest",
    ) -> Path:
        """Write kriging interpolation factors to file.

        Parameters
        ----------
        pilot_x, pilot_y : NDArray
            Coordinates of pilot points.
        pilot_names : list[str]
            Names of pilot point parameters.
        target_x, target_y : NDArray
            Coordinates of target points.
        target_ids : list[int | str]
            IDs of target points (nodes, elements).
        variogram : Variogram
            Variogram model.
        filepath : Path | str
            Output file path.
        kriging_type : str
            Type of kriging.
        format : str
            Output format: "pest" for PEST pp_factors, "csv" for CSV.

        Returns
        -------
        Path
            Path to written file.
        """
        filepath = Path(filepath)

        # Compute factors
        factors = self.compute_kriging_factors(
            pilot_x, pilot_y, target_x, target_y, variogram, kriging_type
        )

        if format == "pest":
            # PEST pilot point factors format
            lines = []
            lines.append("# Kriging factors file")
            lines.append(f"# Variogram: {variogram}")
            lines.append(f"# {len(target_ids)} targets, {len(pilot_names)} pilot points")
            lines.append("#")

            for i, target_id in enumerate(target_ids):
                # Write target ID and number of contributing pilot points
                # Only include non-zero weights
                nonzero = np.abs(factors[i, :]) > 1e-10
                n_contrib = np.sum(nonzero)

                lines.append(f"{target_id}  {n_contrib}")
                for j in np.where(nonzero)[0]:
                    lines.append(f"  {pilot_names[j]}  {factors[i, j]:.8e}")

        else:  # csv
            lines = []
            header = ["target_id"] + pilot_names
            lines.append(",".join(header))

            for i, target_id in enumerate(target_ids):
                row = [str(target_id)] + [f"{f:.8e}" for f in factors[i, :]]
                lines.append(",".join(row))

        filepath.write_text("\n".join(lines))
        return filepath

    def write_structure_file(
        self,
        variogram: Variogram,
        filepath: Path | str,
        name: str = "structure1",
    ) -> Path:
        """Write PEST++ structure file for variogram.

        Parameters
        ----------
        variogram : Variogram
            Variogram model.
        filepath : Path | str
            Output file path.
        name : str
            Structure name.

        Returns
        -------
        Path
            Path to written file.
        """
        filepath = Path(filepath)

        vtype = variogram.variogram_type
        vtype_str = vtype.value if isinstance(vtype, VariogramType) else vtype

        lines = []
        lines.append(f"# PEST++ structure file for {name}")
        lines.append(f"# Variogram type: {vtype_str}")
        lines.append("")
        lines.append(f"STRUCTURE {name}")
        lines.append(f"  NUGGET {variogram.nugget}")
        lines.append("  TRANSFORM NONE")
        lines.append("")
        lines.append(f"VARIOGRAM {name}_vario")
        lines.append(f"  VARTYPE {vtype_str.upper()}")
        lines.append(f"  A {variogram.a}")
        lines.append(f"  SILL {variogram.sill}")
        if variogram.anisotropy_ratio != 1.0:
            lines.append(f"  ANISOTROPY {variogram.anisotropy_ratio}")
            lines.append(f"  BEARING {variogram.anisotropy_angle}")
        lines.append("END VARIOGRAM")
        lines.append("")
        lines.append("END STRUCTURE")

        filepath.write_text("\n".join(lines))
        return filepath

    def __repr__(self) -> str:
        """Return string representation."""
        return f"GeostatManager(model={self.model is not None})"


def compute_empirical_variogram(
    x: NDArray,
    y: NDArray,
    values: NDArray,
    n_lags: int = 15,
    max_lag: float | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute empirical variogram from data.

    Parameters
    ----------
    x, y : NDArray
        Coordinates of data points.
    values : NDArray
        Values at data points.
    n_lags : int
        Number of lag bins.
    max_lag : float | None
        Maximum lag distance.

    Returns
    -------
    tuple[NDArray, NDArray, NDArray]
        Lag centers, semivariance values, and pair counts.
    """
    if HAS_SCIPY:
        coords = np.column_stack([x, y])
        distances = cdist(coords, coords)
    else:
        diff_x = x[:, np.newaxis] - x[np.newaxis, :]
        diff_y = y[:, np.newaxis] - y[np.newaxis, :]
        distances = np.sqrt(diff_x**2 + diff_y**2)

    if max_lag is None:
        max_lag = np.max(distances) / 2.0

    lag_edges = np.linspace(0, max_lag, n_lags + 1)
    lag_centers = (lag_edges[:-1] + lag_edges[1:]) / 2

    n_points = len(values)
    gamma = np.zeros(n_lags)
    counts = np.zeros(n_lags, dtype=int)

    for i in range(n_points):
        for j in range(i + 1, n_points):
            d = distances[i, j]
            bin_idx = np.searchsorted(lag_edges[1:], d)
            if bin_idx < n_lags:
                gamma[bin_idx] += 0.5 * (values[i] - values[j]) ** 2
                counts[bin_idx] += 1

    valid = counts > 0
    gamma[valid] /= counts[valid]

    return lag_centers, gamma, counts
