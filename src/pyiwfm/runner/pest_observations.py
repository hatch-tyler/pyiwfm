"""PEST++ observation types and classes for IWFM models.

This module provides IWFM-specific observation types and enhanced observation
classes for use with PEST++ calibration, uncertainty analysis, and optimization.

The observation types cover:
- Groundwater: head, drawdown, head differences, vertical gradients
- Streams: flow, stage, gain/loss
- Lakes: level, storage
- Budgets: GW, stream, root zone components
- Land subsidence: subsidence, compaction
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray


class IWFMObservationType(Enum):
    """Types of observations in IWFM models.

    Each observation type has default properties for weight calculation
    and transformation strategies commonly used in calibration.

    Categories
    ----------
    Groundwater observations:
        HEAD, DRAWDOWN, HEAD_DIFFERENCE, VERTICAL_GRADIENT

    Stream observations:
        STREAM_FLOW, STREAM_STAGE, STREAM_GAIN_LOSS

    Lake observations:
        LAKE_LEVEL, LAKE_STORAGE

    Budget observations:
        GW_BUDGET, STREAM_BUDGET, ROOTZONE_BUDGET, LAKE_BUDGET

    Land subsidence observations:
        SUBSIDENCE, COMPACTION
    """

    # Groundwater observations
    HEAD = "head"
    DRAWDOWN = "drawdown"
    HEAD_DIFFERENCE = "hdiff"
    VERTICAL_GRADIENT = "vgrad"

    # Stream observations
    STREAM_FLOW = "flow"
    STREAM_STAGE = "stage"
    STREAM_GAIN_LOSS = "sgl"

    # Lake observations
    LAKE_LEVEL = "lake"
    LAKE_STORAGE = "lsto"

    # Budget observations
    GW_BUDGET = "gwbud"
    STREAM_BUDGET = "strbud"
    ROOTZONE_BUDGET = "rzbud"
    LAKE_BUDGET = "lakbud"

    # Land subsidence
    SUBSIDENCE = "sub"
    COMPACTION = "comp"

    @property
    def default_transform(self) -> str:
        """Get default transformation for this observation type.

        Returns
        -------
        str
            'none', 'log', or 'sqrt' depending on observation type.
        """
        # Log transform recommended for flows (wide range of values)
        log_types = {
            self.STREAM_FLOW,
            self.LAKE_STORAGE,
        }
        # Square root can help with flow data too
        sqrt_types = {
            self.STREAM_GAIN_LOSS,
        }

        if self in log_types:
            return "log"
        elif self in sqrt_types:
            return "sqrt"
        return "none"

    @property
    def typical_error(self) -> float:
        """Get typical measurement error for this observation type.

        Returns
        -------
        float
            Typical standard deviation of measurement error.
            Units depend on observation type.
        """
        # Measurement errors in typical units (ft for head, cfs for flow)
        errors = {
            self.HEAD: 1.0,  # 1 ft
            self.DRAWDOWN: 0.5,  # 0.5 ft
            self.HEAD_DIFFERENCE: 0.5,  # 0.5 ft
            self.VERTICAL_GRADIENT: 0.01,  # dimensionless
            self.STREAM_FLOW: 0.1,  # 10% relative error
            self.STREAM_STAGE: 0.1,  # 0.1 ft
            self.STREAM_GAIN_LOSS: 0.2,  # 20% relative error
            self.LAKE_LEVEL: 0.5,  # 0.5 ft
            self.LAKE_STORAGE: 0.1,  # 10% relative error
            self.GW_BUDGET: 0.1,  # 10% relative error
            self.STREAM_BUDGET: 0.1,  # 10% relative error
            self.ROOTZONE_BUDGET: 0.15,  # 15% relative error
            self.LAKE_BUDGET: 0.1,  # 10% relative error
            self.SUBSIDENCE: 0.01,  # 0.01 ft
            self.COMPACTION: 0.005,  # 0.005 ft
        }
        return errors.get(self, 1.0)

    @property
    def is_relative_error(self) -> bool:
        """Check if typical error is relative (fraction) or absolute.

        Returns
        -------
        bool
            True if typical_error is a relative error (0-1 range).
        """
        relative_types = {
            self.STREAM_FLOW,
            self.STREAM_GAIN_LOSS,
            self.LAKE_STORAGE,
            self.GW_BUDGET,
            self.STREAM_BUDGET,
            self.ROOTZONE_BUDGET,
            self.LAKE_BUDGET,
        }
        return self in relative_types

    @property
    def group_prefix(self) -> str:
        """Get default group name prefix for this observation type.

        Returns
        -------
        str
            Prefix for observation group names.
        """
        return self.value


class WeightStrategy(Enum):
    """Strategies for calculating observation weights.

    Weight calculation is critical for proper calibration. Different
    strategies are appropriate for different situations.
    """

    EQUAL = "equal"
    """All observations have weight = 1."""

    INVERSE_VARIANCE = "inverse_variance"
    """Weight = 1/variance. Requires measurement error estimates."""

    GROUP_CONTRIBUTION = "group_contribution"
    """Weights adjusted so each group contributes equally to objective function."""

    TEMPORAL_DECAY = "temporal_decay"
    """Recent observations weighted higher than older ones."""

    MAGNITUDE_BASED = "magnitude_based"
    """Weight scales with observation magnitude (for relative errors)."""

    CUSTOM = "custom"
    """User-specified weights."""


@dataclass
class ObservationLocation:
    """Location information for an observation point.

    Attributes
    ----------
    x : float
        X coordinate.
    y : float
        Y coordinate.
    z : float | None
        Z coordinate (elevation or depth).
    node_id : int | None
        Associated model node ID.
    element_id : int | None
        Associated model element ID.
    layer : int | None
        Model layer.
    reach_id : int | None
        Stream reach ID (for stream observations).
    lake_id : int | None
        Lake ID (for lake observations).
    """

    x: float
    y: float
    z: float | None = None
    node_id: int | None = None
    element_id: int | None = None
    layer: int | None = None
    reach_id: int | None = None
    lake_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            k: v
            for k, v in {
                "x": self.x,
                "y": self.y,
                "z": self.z,
                "node_id": self.node_id,
                "element_id": self.element_id,
                "layer": self.layer,
                "reach_id": self.reach_id,
                "lake_id": self.lake_id,
            }.items()
            if v is not None
        }


@dataclass
class IWFMObservation:
    """Enhanced observation class with IWFM-specific attributes.

    This extends the basic Observation class with additional metadata
    useful for IWFM calibration and post-processing.

    Attributes
    ----------
    name : str
        Observation name (up to 200 chars for PEST++).
    value : float
        Observed value.
    weight : float
        Observation weight (inverse of standard deviation).
    group : str
        Observation group name.
    obs_type : IWFMObservationType | None
        Type of observation.
    datetime : datetime | None
        Time of observation.
    location : ObservationLocation | None
        Spatial location of observation.
    simulated_name : str | None
        Name used in model output (if different from obs name).
    error_std : float | None
        Estimated measurement error standard deviation.
    transform : str
        Transformation applied: 'none', 'log', 'sqrt'.
    metadata : dict
        Additional metadata.
    """

    name: str
    value: float
    weight: float = 1.0
    group: str = "default"
    obs_type: IWFMObservationType | None = None
    datetime: datetime | None = None
    location: ObservationLocation | None = None
    simulated_name: str | None = None
    error_std: float | None = None
    transform: str = "none"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate observation."""
        if len(self.name) > 200:
            raise ValueError(f"Observation name too long (max 200): {self.name}")
        if self.weight < 0:
            raise ValueError(f"Weight must be non-negative: {self.weight}")
        if self.transform not in ("none", "log", "sqrt"):
            raise ValueError(f"Invalid transform: {self.transform}")
        if self.transform == "log" and self.value <= 0:
            raise ValueError(f"Log transform requires positive value: {self.value}")

    @property
    def transformed_value(self) -> float:
        """Get transformed observation value.

        Returns
        -------
        float
            Value after applying transform.
        """
        if self.transform == "log":
            return float(np.log10(self.value))
        elif self.transform == "sqrt":
            return float(np.sqrt(self.value))
        return self.value

    def calculate_weight(
        self,
        strategy: WeightStrategy = WeightStrategy.INVERSE_VARIANCE,
        **kwargs: Any,
    ) -> float:
        """Calculate observation weight using specified strategy.

        Parameters
        ----------
        strategy : WeightStrategy
            Weight calculation strategy.
        **kwargs : Any
            Strategy-specific parameters.

        Returns
        -------
        float
            Calculated weight.
        """
        if strategy == WeightStrategy.EQUAL:
            return 1.0

        elif strategy == WeightStrategy.INVERSE_VARIANCE:
            if self.error_std is not None and self.error_std > 0:
                return 1.0 / self.error_std
            elif self.obs_type is not None:
                error = self.obs_type.typical_error
                if self.obs_type.is_relative_error:
                    # Convert relative error to absolute
                    error = abs(self.value) * error
                return 1.0 / max(error, 1e-10)
            return 1.0

        elif strategy == WeightStrategy.TEMPORAL_DECAY:
            decay_factor = kwargs.get("decay_factor", 0.95)
            reference_date = kwargs.get("reference_date")
            if self.datetime is not None and reference_date is not None:
                days = (reference_date - self.datetime).days
                return float(decay_factor ** max(0, days / 365.0))
            return 1.0

        elif strategy == WeightStrategy.MAGNITUDE_BASED:
            if self.obs_type is not None and self.obs_type.is_relative_error:
                error = self.obs_type.typical_error
                return 1.0 / max(abs(self.value) * error, 1e-10)
            return 1.0 / max(abs(self.value) * 0.1, 1e-10)

        return self.weight

    def to_pest_line(self) -> str:
        """Format as PEST control file observation line.

        Returns
        -------
        str
            Formatted line for PEST control file.
        """
        return (
            f"{self.name:20s} {self.transformed_value:15.7e} {self.weight:10.4e} {self.group:20s}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns
        -------
        dict
            Dictionary representation.
        """
        result = {
            "name": self.name,
            "value": self.value,
            "weight": self.weight,
            "group": self.group,
            "transform": self.transform,
        }
        if self.obs_type is not None:
            result["obs_type"] = self.obs_type.value
        if self.datetime is not None:
            result["datetime"] = self.datetime.isoformat()
        if self.location is not None:
            result["location"] = self.location.to_dict()
        if self.simulated_name is not None:
            result["simulated_name"] = self.simulated_name
        if self.error_std is not None:
            result["error_std"] = self.error_std
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    def __repr__(self) -> str:
        """Return string representation."""
        type_str = f", type={self.obs_type.value}" if self.obs_type else ""
        return f"IWFMObservation(name='{self.name}', value={self.value:.4g}, weight={self.weight:.4g}{type_str})"


@dataclass
class IWFMObservationGroup:
    """Group of observations with shared properties.

    Observation groups allow setting common properties and calculating
    weights to achieve target contributions to the objective function.

    Attributes
    ----------
    name : str
        Group name (up to 200 chars for PEST++).
    obs_type : IWFMObservationType | None
        Type of observations in this group.
    observations : list[IWFMObservation]
        Observations in this group.
    target_contribution : float | None
        Target contribution to objective function (0-1).
    covariance_matrix : NDArray | None
        Observation error covariance matrix.
    """

    name: str
    obs_type: IWFMObservationType | None = None
    observations: list[IWFMObservation] = field(default_factory=list)
    target_contribution: float | None = None
    covariance_matrix: NDArray | None = None

    def __post_init__(self) -> None:
        """Validate group."""
        if len(self.name) > 200:
            raise ValueError(f"Group name too long (max 200): {self.name}")

    @property
    def n_observations(self) -> int:
        """Number of observations in group."""
        return len(self.observations)

    @property
    def values(self) -> NDArray:
        """Get array of observation values.

        Returns
        -------
        NDArray
            Array of observation values.
        """
        return np.array([obs.value for obs in self.observations])

    @property
    def weights(self) -> NDArray:
        """Get array of observation weights.

        Returns
        -------
        NDArray
            Array of observation weights.
        """
        return np.array([obs.weight for obs in self.observations])

    @property
    def contribution(self) -> float:
        """Calculate current contribution to objective function.

        Assumes residuals of 1 (equal to weights) for estimation.
        Actual contribution depends on simulated values.

        Returns
        -------
        float
            Estimated contribution (sum of squared weighted values).
        """
        return float(np.sum(self.weights**2))

    def add_observation(
        self,
        name: str,
        value: float,
        weight: float = 1.0,
        **kwargs: Any,
    ) -> IWFMObservation:
        """Add an observation to this group.

        Parameters
        ----------
        name : str
            Observation name.
        value : float
            Observed value.
        weight : float
            Observation weight.
        **kwargs : Any
            Additional observation attributes.

        Returns
        -------
        IWFMObservation
            The created observation.
        """
        obs = IWFMObservation(
            name=name,
            value=value,
            weight=weight,
            group=self.name,
            obs_type=self.obs_type,
            **kwargs,
        )
        self.observations.append(obs)
        return obs

    def set_weights(
        self,
        strategy: WeightStrategy = WeightStrategy.EQUAL,
        **kwargs: Any,
    ) -> None:
        """Set weights for all observations in group.

        Parameters
        ----------
        strategy : WeightStrategy
            Weight calculation strategy.
        **kwargs : Any
            Strategy-specific parameters.
        """
        for obs in self.observations:
            obs.weight = obs.calculate_weight(strategy, **kwargs)

    def scale_weights(self, factor: float) -> None:
        """Scale all weights by a factor.

        Parameters
        ----------
        factor : float
            Multiplicative scaling factor.
        """
        for obs in self.observations:
            obs.weight *= factor

    def normalize_weights(self, target_sum: float = 1.0) -> None:
        """Normalize weights to sum to target value.

        Parameters
        ----------
        target_sum : float
            Target sum of weights.
        """
        current_sum = float(np.sum(self.weights))
        if current_sum > 0:
            factor = target_sum / current_sum
            self.scale_weights(factor)

    def get_observations_by_time(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[IWFMObservation]:
        """Get observations within a time range.

        Parameters
        ----------
        start_date : datetime | None
            Start of time range.
        end_date : datetime | None
            End of time range.

        Returns
        -------
        list[IWFMObservation]
            Observations within the specified range.
        """
        result = []
        for obs in self.observations:
            if obs.datetime is None:
                continue
            if start_date is not None and obs.datetime < start_date:
                continue
            if end_date is not None and obs.datetime > end_date:
                continue
            result.append(obs)
        return result

    def summary(self) -> dict[str, Any]:
        """Get summary statistics for this group.

        Returns
        -------
        dict
            Summary statistics.
        """
        values = self.values
        weights = self.weights

        return {
            "name": self.name,
            "obs_type": self.obs_type.value if self.obs_type else None,
            "n_observations": len(self.observations),
            "value_mean": float(np.mean(values)) if len(values) > 0 else None,
            "value_std": float(np.std(values)) if len(values) > 0 else None,
            "value_min": float(np.min(values)) if len(values) > 0 else None,
            "value_max": float(np.max(values)) if len(values) > 0 else None,
            "weight_mean": float(np.mean(weights)) if len(weights) > 0 else None,
            "weight_sum": float(np.sum(weights)) if len(weights) > 0 else None,
            "target_contribution": self.target_contribution,
        }

    def __repr__(self) -> str:
        """Return string representation."""
        type_str = f", type={self.obs_type.value}" if self.obs_type else ""
        return f"IWFMObservationGroup(name='{self.name}', n_obs={self.n_observations}{type_str})"

    def __len__(self) -> int:
        """Return number of observations."""
        return len(self.observations)

    def __iter__(self) -> Iterator[IWFMObservation]:
        """Iterate over observations."""
        return iter(self.observations)


@dataclass
class DerivedObservation:
    """Observation derived from other observations via expression.

    Derived observations allow creating constraints and composite
    targets from model outputs. Common uses include:
    - Mass balance checks
    - Head differences
    - Flow ratios

    Attributes
    ----------
    name : str
        Derived observation name.
    expression : str
        Mathematical expression using observation names.
    source_observations : list[str]
        Names of observations used in the expression.
    target_value : float
        Target value for the derived quantity.
    weight : float
        Observation weight.
    group : str
        Observation group name.
    """

    name: str
    expression: str
    source_observations: list[str]
    target_value: float = 0.0
    weight: float = 1.0
    group: str = "derived"

    def evaluate(self, obs_values: dict[str, float]) -> float:
        """Evaluate the expression with given observation values.

        Parameters
        ----------
        obs_values : dict[str, float]
            Dictionary mapping observation names to values.

        Returns
        -------
        float
            Result of expression evaluation.

        Raises
        ------
        ValueError
            If required observations are missing.
        """
        # Check all required observations are present
        missing = set(self.source_observations) - set(obs_values.keys())
        if missing:
            raise ValueError(f"Missing observations for evaluation: {missing}")

        # Create safe evaluation environment
        safe_dict: dict[str, Any] = {name: obs_values[name] for name in self.source_observations}
        # Add math functions
        safe_dict.update(
            {
                "abs": abs,
                "min": min,
                "max": max,
                "sum": sum,
                "sqrt": np.sqrt,
                "log": np.log,
                "log10": np.log10,
                "exp": np.exp,
            }
        )

        try:
            return float(eval(self.expression, {"__builtins__": {}}, safe_dict))
        except Exception as e:
            raise ValueError(f"Error evaluating expression '{self.expression}': {e}") from e

    def to_prior_equation(self) -> str:
        """Format as PEST prior information equation.

        Returns
        -------
        str
            PEST-format prior information line.
        """
        # Convert expression to PEST format
        # PEST uses @ prefix for observation names
        pest_expr = self.expression
        for obs_name in self.source_observations:
            pest_expr = pest_expr.replace(obs_name, f"@{obs_name}")

        return f"{self.name} 1.0 * {pest_expr} = {self.target_value} {self.weight} {self.group}"

    def __repr__(self) -> str:
        """Return string representation."""
        return f"DerivedObservation(name='{self.name}', expr='{self.expression}')"
