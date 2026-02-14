"""IWFM parameter types and parameterization strategies for PEST++.

This module provides the core parameter management classes for setting up
highly parameterized PEST++ calibration of IWFM models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray


class IWFMParameterType(Enum):
    """Types of parameters in IWFM models.

    Each parameter type corresponds to a specific physical property
    or input that can be adjusted during calibration.
    """

    # Aquifer parameters (by layer, zone, or pilot point)
    HORIZONTAL_K = "hk"           # Horizontal hydraulic conductivity [L/T]
    VERTICAL_K = "vk"             # Vertical hydraulic conductivity [L/T]
    SPECIFIC_STORAGE = "ss"       # Specific storage [1/L]
    SPECIFIC_YIELD = "sy"         # Specific yield [-]
    POROSITY = "por"              # Porosity [-]

    # Stream parameters
    STREAMBED_K = "strk"          # Streambed hydraulic conductivity [L/T]
    STREAMBED_THICKNESS = "strt"  # Streambed thickness [L]
    STREAM_WIDTH = "strw"         # Stream width factor [-]
    MANNING_N = "mann"            # Manning's roughness coefficient [-]

    # Lake parameters
    LAKEBED_K = "lakk"            # Lakebed hydraulic conductivity [L/T]
    LAKEBED_THICKNESS = "lakt"    # Lakebed thickness [L]

    # Root zone parameters
    CROP_COEFFICIENT = "kc"       # Crop coefficient multiplier [-]
    IRRIGATION_EFFICIENCY = "ie"  # Irrigation efficiency [-]
    ROOT_DEPTH = "rd"             # Root depth factor [-]
    FIELD_CAPACITY = "fc"         # Field capacity [-]
    WILTING_POINT = "wp"          # Wilting point [-]
    SOIL_AWC = "awc"              # Available water capacity [-]

    # Flux multipliers (applied to time series inputs)
    PUMPING_MULT = "pump"         # Pumping rate multiplier [-]
    RECHARGE_MULT = "rech"        # Recharge rate multiplier [-]
    DIVERSION_MULT = "div"        # Diversion rate multiplier [-]
    BYPASS_MULT = "byp"           # Bypass flow multiplier [-]
    PRECIP_MULT = "ppt"           # Precipitation multiplier [-]
    ET_MULT = "et"                # ET multiplier [-]
    STREAM_INFLOW_MULT = "infl"   # Stream inflow multiplier [-]
    RETURN_FLOW_MULT = "rtf"      # Return flow multiplier [-]

    # Boundary conditions
    GHB_CONDUCTANCE = "ghbc"      # General head boundary conductance [L2/T]
    GHB_HEAD = "ghbh"             # General head boundary head [L]
    SPECIFIED_HEAD = "chd"        # Specified head boundary [L]
    SPECIFIED_FLOW = "wel"        # Specified flow boundary [L3/T]

    # Subsidence parameters
    ELASTIC_STORAGE = "ske"       # Elastic skeletal storage [1/L]
    INELASTIC_STORAGE = "skv"     # Inelastic skeletal storage [1/L]
    PRECONSOLIDATION = "pcs"      # Preconsolidation stress [L]

    @property
    def default_bounds(self) -> tuple[float, float]:
        """Get default parameter bounds based on type."""
        bounds_map = {
            # Aquifer parameters
            self.HORIZONTAL_K: (0.001, 10000.0),
            self.VERTICAL_K: (0.0001, 1000.0),
            self.SPECIFIC_STORAGE: (1e-7, 1e-3),
            self.SPECIFIC_YIELD: (0.01, 0.4),
            self.POROSITY: (0.1, 0.5),
            # Stream parameters
            self.STREAMBED_K: (0.001, 100.0),
            self.STREAMBED_THICKNESS: (0.1, 10.0),
            self.STREAM_WIDTH: (0.5, 2.0),
            self.MANNING_N: (0.01, 0.1),
            # Lake parameters
            self.LAKEBED_K: (0.001, 100.0),
            self.LAKEBED_THICKNESS: (0.1, 10.0),
            # Root zone parameters
            self.CROP_COEFFICIENT: (0.5, 1.5),
            self.IRRIGATION_EFFICIENCY: (0.5, 1.0),
            self.ROOT_DEPTH: (0.5, 2.0),
            self.FIELD_CAPACITY: (0.1, 0.4),
            self.WILTING_POINT: (0.05, 0.2),
            self.SOIL_AWC: (0.05, 0.25),
            # Multipliers
            self.PUMPING_MULT: (0.5, 2.0),
            self.RECHARGE_MULT: (0.5, 2.0),
            self.DIVERSION_MULT: (0.5, 2.0),
            self.BYPASS_MULT: (0.5, 2.0),
            self.PRECIP_MULT: (0.8, 1.2),
            self.ET_MULT: (0.8, 1.2),
            self.STREAM_INFLOW_MULT: (0.5, 2.0),
            self.RETURN_FLOW_MULT: (0.5, 2.0),
            # Boundary conditions
            self.GHB_CONDUCTANCE: (0.001, 1000.0),
            self.GHB_HEAD: (-1000.0, 1000.0),
            self.SPECIFIED_HEAD: (-1000.0, 1000.0),
            self.SPECIFIED_FLOW: (-1e6, 1e6),
            # Subsidence
            self.ELASTIC_STORAGE: (1e-6, 1e-3),
            self.INELASTIC_STORAGE: (1e-5, 1e-2),
            self.PRECONSOLIDATION: (0.0, 1000.0),
        }
        return bounds_map.get(self, (0.01, 100.0))

    @property
    def default_transform(self) -> str:
        """Get default transform based on parameter type."""
        # Log transform for parameters spanning orders of magnitude
        log_params = {
            self.HORIZONTAL_K, self.VERTICAL_K,
            self.SPECIFIC_STORAGE, self.STREAMBED_K,
            self.LAKEBED_K, self.GHB_CONDUCTANCE,
            self.ELASTIC_STORAGE, self.INELASTIC_STORAGE,
        }
        if self in log_params:
            return "log"
        return "none"

    @property
    def is_multiplier(self) -> bool:
        """Check if this parameter type is a multiplier."""
        return self.value in {
            "pump", "rech", "div", "byp", "ppt", "et", "infl", "rtf"
        }


class ParameterTransform(Enum):
    """Parameter transformation types for PEST++."""

    NONE = "none"       # No transformation
    LOG = "log"         # Log10 transformation
    FIXED = "fixed"     # Fixed (not adjusted)
    TIED = "tied"       # Tied to another parameter


@dataclass
class ParameterGroup:
    """Parameter group definition for PEST++.

    Attributes
    ----------
    name : str
        Group name (max 12 characters for PEST compatibility).
    inctyp : str
        Increment type for derivatives: 'relative' or 'absolute'.
    derinc : float
        Derivative increment value.
    derinclb : float
        Lower bound for derivative increment.
    forcen : str
        Force numerical derivatives: 'switch', 'always_2', 'always_3'.
    derincmul : float
        Multiplier for derivative increment.
    dermthd : str
        Derivative method: 'parabolic', 'outside_pts', 'best_fit'.
    splitthresh : float
        Split threshold for parameter splitting.
    splitreldiff : float
        Relative difference threshold for splitting.
    """

    name: str
    inctyp: str = "relative"
    derinc: float = 0.01
    derinclb: float = 0.0
    forcen: str = "switch"
    derincmul: float = 2.0
    dermthd: str = "parabolic"
    splitthresh: float = 1e-5
    splitreldiff: float = 0.5

    def __post_init__(self) -> None:
        """Validate group name length."""
        if len(self.name) > 12:
            # Truncate for PEST compatibility
            self.name = self.name[:12]

    def to_pest_line(self) -> str:
        """Format as PEST control file parameter group line."""
        return (
            f"{self.name:12s} {self.inctyp:10s} {self.derinc:12.4e} "
            f"{self.derinclb:12.4e} {self.forcen:10s} {self.derincmul:8.3f} "
            f"{self.dermthd:12s}"
        )


@dataclass
class Parameter:
    """Individual parameter definition for PEST++.

    Attributes
    ----------
    name : str
        Parameter name (max 200 characters for PEST++).
    initial_value : float
        Initial parameter value.
    lower_bound : float
        Lower bound for parameter.
    upper_bound : float
        Upper bound for parameter.
    group : str
        Parameter group name.
    transform : ParameterTransform
        Transformation type.
    scale : float
        Scale factor for parameter.
    offset : float
        Offset for parameter.
    param_type : IWFMParameterType | None
        IWFM parameter type (for metadata).
    layer : int | None
        Model layer (if applicable).
    zone : int | None
        Zone ID (if applicable).
    location : tuple[float, float] | None
        (x, y) location for pilot points.
    tied_to : str | None
        Name of parent parameter if tied.
    tied_ratio : float
        Ratio to parent parameter if tied.
    metadata : dict[str, Any]
        Additional metadata.
    """

    name: str
    initial_value: float
    lower_bound: float
    upper_bound: float
    group: str = "default"
    transform: ParameterTransform = ParameterTransform.NONE
    scale: float = 1.0
    offset: float = 0.0
    param_type: IWFMParameterType | None = None
    layer: int | None = None
    zone: int | None = None
    location: tuple[float, float] | None = None
    tied_to: str | None = None
    tied_ratio: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate parameter."""
        if len(self.name) > 200:
            raise ValueError(f"Parameter name too long (max 200): {self.name}")

        if self.lower_bound > self.upper_bound:
            raise ValueError(
                f"Lower bound ({self.lower_bound}) > upper bound ({self.upper_bound}) "
                f"for parameter '{self.name}'"
            )

        # Ensure initial value is within bounds
        if self.transform != ParameterTransform.FIXED:
            if not self.lower_bound <= self.initial_value <= self.upper_bound:
                # Clip to bounds with warning
                self.initial_value = max(
                    self.lower_bound,
                    min(self.upper_bound, self.initial_value)
                )

        # Convert string transform to enum
        if isinstance(self.transform, str):
            self.transform = ParameterTransform(self.transform.lower())

    @property
    def partrans(self) -> str:
        """Get PEST parameter transformation string."""
        if self.transform == ParameterTransform.TIED:
            return "tied"
        elif self.transform == ParameterTransform.FIXED:
            return "fixed"
        elif self.transform == ParameterTransform.LOG:
            return "log"
        return "none"

    @property
    def parval1(self) -> float:
        """Get initial parameter value for PEST."""
        return self.initial_value

    @property
    def parlbnd(self) -> float:
        """Get lower bound for PEST."""
        return self.lower_bound

    @property
    def parubnd(self) -> float:
        """Get upper bound for PEST."""
        return self.upper_bound

    def to_pest_line(self) -> str:
        """Format as PEST control file parameter data line."""
        partied = self.tied_to if self.tied_to else self.name
        return (
            f"{self.name:20s} {self.partrans:10s} {self.tied_ratio:8.4f} "
            f"{self.parval1:15.7e} {self.parlbnd:15.7e} {self.parubnd:15.7e} "
            f"{self.group:12s} {self.scale:10.4e} {self.offset:10.4e}"
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"Parameter(name='{self.name}', value={self.initial_value:.4g}, "
            f"bounds=[{self.lower_bound:.4g}, {self.upper_bound:.4g}], "
            f"group='{self.group}')"
        )


# --- Parameterization Strategy Classes ---


@dataclass
class ParameterizationStrategy:
    """Base class for parameterization strategies.

    A parameterization strategy defines how a specific type of
    parameter is distributed across the model domain.
    """

    param_type: IWFMParameterType
    transform: ParameterTransform = field(default=ParameterTransform.NONE)
    bounds: tuple[float, float] | None = None
    group_name: str | None = None

    def __post_init__(self) -> None:
        """Set defaults from parameter type."""
        if self.bounds is None:
            self.bounds = self.param_type.default_bounds

        if isinstance(self.transform, str):
            self.transform = ParameterTransform(self.transform.lower())
        elif self.transform == ParameterTransform.NONE:
            # Use default transform for parameter type
            default = self.param_type.default_transform
            self.transform = ParameterTransform(default)

        if self.group_name is None:
            self.group_name = self.param_type.value

    def generate_parameters(self, model: Any) -> list[Parameter]:
        """Generate parameters for this strategy.

        Must be implemented by subclasses.

        Parameters
        ----------
        model : IWFMModel
            The IWFM model to parameterize.

        Returns
        -------
        list[Parameter]
            List of generated parameters.
        """
        raise NotImplementedError("Subclasses must implement generate_parameters")


@dataclass
class ZoneParameterization(ParameterizationStrategy):
    """Zone-based parameterization strategy.

    Creates one parameter per zone for a given parameter type.
    Zones can be model subregions, custom zone definitions, or
    any spatial grouping of elements.

    Attributes
    ----------
    zones : list[int] | str
        List of zone IDs or "subregions" to use model subregions.
    layer : int | None
        Model layer (for layered parameters like K).
    initial_values : float | dict[int, float]
        Initial value(s). If float, used for all zones.
        If dict, maps zone ID to value.
    zone_names : dict[int, str] | None
        Optional zone names for parameter naming.
    """

    zones: list[int] | str = "subregions"
    layer: int | None = None
    initial_values: float | dict[int, float] = 1.0
    zone_names: dict[int, str] | None = None

    def _get_zone_ids(self, model: Any) -> list[int]:
        """Get zone IDs from model or specification."""
        if self.zones == "subregions":
            # Get subregion IDs from model
            if hasattr(model, 'grid') and hasattr(model.grid, 'subregions'):
                return sorted(model.grid.subregions.keys())
            elif hasattr(model, 'subregions'):
                return sorted(model.subregions.keys())
            else:
                raise ValueError("Model does not have subregion information")
        elif isinstance(self.zones, str) and self.zones == "all":
            # Same as subregions
            return self._get_zone_ids_subregions(model)
        else:
            return list(self.zones)

    def _get_zone_ids_subregions(self, model: Any) -> list[int]:
        """Get subregion IDs from model."""
        if hasattr(model, 'grid') and hasattr(model.grid, 'subregions'):
            return sorted(model.grid.subregions.keys())
        return []

    def _get_initial_value(self, zone_id: int) -> float:
        """Get initial value for a zone."""
        if isinstance(self.initial_values, dict):
            return self.initial_values.get(zone_id, 1.0)
        return self.initial_values

    def _get_zone_name(self, zone_id: int) -> str:
        """Get zone name for parameter naming."""
        if self.zone_names and zone_id in self.zone_names:
            return self.zone_names[zone_id]
        return f"z{zone_id}"

    def generate_parameters(self, model: Any) -> list[Parameter]:
        """Generate zone-based parameters.

        Parameters
        ----------
        model : IWFMModel
            The IWFM model to parameterize.

        Returns
        -------
        list[Parameter]
            One parameter per zone.
        """
        parameters = []
        zone_ids = self._get_zone_ids(model)

        for zone_id in zone_ids:
            zone_name = self._get_zone_name(zone_id)
            initial = self._get_initial_value(zone_id)

            # Build parameter name
            if self.layer is not None:
                name = f"{self.param_type.value}_{zone_name}_l{self.layer}"
            else:
                name = f"{self.param_type.value}_{zone_name}"

            param = Parameter(
                name=name,
                initial_value=initial,
                lower_bound=self.bounds[0],
                upper_bound=self.bounds[1],
                group=self.group_name,
                transform=self.transform,
                param_type=self.param_type,
                layer=self.layer,
                zone=zone_id,
            )
            parameters.append(param)

        return parameters


@dataclass
class MultiplierParameterization(ParameterizationStrategy):
    """Multiplier parameterization strategy.

    Creates parameters that act as multipliers on existing model values.
    Useful for adjusting time series inputs like pumping, recharge, etc.

    Attributes
    ----------
    spatial_extent : str
        Spatial scope: 'global', 'zone', or 'element'.
    temporal_extent : str
        Temporal scope: 'constant', 'seasonal', 'monthly', 'annual'.
    zones : list[int] | None
        Zone IDs for zone-based multipliers.
    initial_value : float
        Initial multiplier value (typically 1.0).
    target_file : Path | None
        File containing base values to multiply.
    """

    spatial_extent: str = "global"
    temporal_extent: str = "constant"
    zones: list[int] | None = None
    initial_value: float = 1.0
    target_file: Path | None = None

    def __post_init__(self) -> None:
        """Validate and set defaults."""
        super().__post_init__()

        valid_spatial = {"global", "zone", "element"}
        if self.spatial_extent not in valid_spatial:
            raise ValueError(
                f"Invalid spatial_extent '{self.spatial_extent}'. "
                f"Must be one of: {valid_spatial}"
            )

        valid_temporal = {"constant", "seasonal", "monthly", "annual"}
        if self.temporal_extent not in valid_temporal:
            raise ValueError(
                f"Invalid temporal_extent '{self.temporal_extent}'. "
                f"Must be one of: {valid_temporal}"
            )

        # Multipliers typically don't need log transform
        if self.transform == ParameterTransform.NONE:
            self.transform = ParameterTransform.NONE

    def _get_temporal_periods(self) -> list[str]:
        """Get temporal period names."""
        if self.temporal_extent == "constant":
            return [""]
        elif self.temporal_extent == "seasonal":
            return ["_winter", "_spring", "_summer", "_fall"]
        elif self.temporal_extent == "monthly":
            return [f"_m{i:02d}" for i in range(1, 13)]
        elif self.temporal_extent == "annual":
            # Would need model info for actual years
            return ["_annual"]
        return [""]

    def generate_parameters(self, model: Any) -> list[Parameter]:
        """Generate multiplier parameters.

        Parameters
        ----------
        model : IWFMModel
            The IWFM model to parameterize.

        Returns
        -------
        list[Parameter]
            Multiplier parameters.
        """
        parameters = []
        temporal_periods = self._get_temporal_periods()

        if self.spatial_extent == "global":
            # Single global multiplier (possibly with temporal variation)
            for period in temporal_periods:
                name = f"{self.param_type.value}{period}"
                param = Parameter(
                    name=name,
                    initial_value=self.initial_value,
                    lower_bound=self.bounds[0],
                    upper_bound=self.bounds[1],
                    group=self.group_name,
                    transform=self.transform,
                    param_type=self.param_type,
                    metadata={"spatial": "global", "temporal": self.temporal_extent},
                )
                parameters.append(param)

        elif self.spatial_extent == "zone":
            # One multiplier per zone
            zone_ids = self.zones
            if zone_ids is None:
                # Get from model
                if hasattr(model, 'grid') and hasattr(model.grid, 'subregions'):
                    zone_ids = sorted(model.grid.subregions.keys())
                else:
                    zone_ids = [1]  # Default single zone

            for zone_id in zone_ids:
                for period in temporal_periods:
                    name = f"{self.param_type.value}_z{zone_id}{period}"
                    param = Parameter(
                        name=name,
                        initial_value=self.initial_value,
                        lower_bound=self.bounds[0],
                        upper_bound=self.bounds[1],
                        group=self.group_name,
                        transform=self.transform,
                        param_type=self.param_type,
                        zone=zone_id,
                        metadata={"spatial": "zone", "temporal": self.temporal_extent},
                    )
                    parameters.append(param)

        elif self.spatial_extent == "element":
            # One multiplier per element (highly parameterized)
            n_elements = 1
            if hasattr(model, 'grid'):
                n_elements = model.grid.n_elements
            elif hasattr(model, 'n_elements'):
                n_elements = model.n_elements

            for elem_id in range(1, n_elements + 1):
                for period in temporal_periods:
                    name = f"{self.param_type.value}_e{elem_id}{period}"
                    param = Parameter(
                        name=name,
                        initial_value=self.initial_value,
                        lower_bound=self.bounds[0],
                        upper_bound=self.bounds[1],
                        group=self.group_name,
                        transform=self.transform,
                        param_type=self.param_type,
                        metadata={
                            "spatial": "element",
                            "element_id": elem_id,
                            "temporal": self.temporal_extent,
                        },
                    )
                    parameters.append(param)

        return parameters


@dataclass
class PilotPointParameterization(ParameterizationStrategy):
    """Pilot point parameterization strategy.

    Creates spatially distributed parameters at pilot point locations.
    Values at model nodes/elements are interpolated using kriging.

    Attributes
    ----------
    points : list[tuple[float, float]] | None
        Explicit pilot point (x, y) coordinates.
    spacing : float | None
        Regular grid spacing (if points not specified).
    layer : int
        Model layer for these parameters.
    initial_value : float | NDArray
        Initial value(s) at pilot points.
    variogram : dict | None
        Variogram specification for kriging.
    kriging_type : str
        Type of kriging: 'ordinary', 'simple', 'universal'.
    search_radius : float | None
        Search radius for kriging interpolation.
    min_points : int
        Minimum pilot points in search neighborhood.
    max_points : int
        Maximum pilot points in search neighborhood.
    """

    points: list[tuple[float, float]] | None = None
    spacing: float | None = None
    layer: int = 1
    initial_value: float | NDArray = 1.0
    variogram: dict | None = None
    kriging_type: str = "ordinary"
    search_radius: float | None = None
    min_points: int = 1
    max_points: int = 20

    def __post_init__(self) -> None:
        """Validate configuration."""
        super().__post_init__()

        if self.points is None and self.spacing is None:
            raise ValueError(
                "Must specify either 'points' or 'spacing' for pilot points"
            )

    def generate_pilot_point_grid(
        self,
        model: Any,
        buffer: float = 0.0,
    ) -> list[tuple[float, float]]:
        """Generate regular grid of pilot points within model domain.

        Parameters
        ----------
        model : IWFMModel
            Model with grid information.
        buffer : float
            Buffer distance inside domain boundary.

        Returns
        -------
        list[tuple[float, float]]
            List of (x, y) pilot point coordinates.
        """
        if self.spacing is None:
            raise ValueError("Spacing must be set to generate grid")

        # Get model extent
        if hasattr(model, 'grid'):
            grid = model.grid
            x_coords = grid.node_coordinates[:, 0]
            y_coords = grid.node_coordinates[:, 1]
        elif hasattr(model, 'node_coordinates'):
            x_coords = model.node_coordinates[:, 0]
            y_coords = model.node_coordinates[:, 1]
        else:
            raise ValueError("Cannot determine model extent")

        xmin, xmax = x_coords.min() + buffer, x_coords.max() - buffer
        ymin, ymax = y_coords.min() + buffer, y_coords.max() - buffer

        # Generate regular grid
        x_points = np.arange(xmin, xmax + self.spacing, self.spacing)
        y_points = np.arange(ymin, ymax + self.spacing, self.spacing)

        points = []
        for x in x_points:
            for y in y_points:
                points.append((float(x), float(y)))

        return points

    def _get_initial_value(self, index: int) -> float:
        """Get initial value for a pilot point."""
        if isinstance(self.initial_value, np.ndarray):
            if index < len(self.initial_value):
                return float(self.initial_value[index])
        return float(self.initial_value) if not isinstance(
            self.initial_value, np.ndarray
        ) else float(self.initial_value[0])

    def generate_parameters(self, model: Any) -> list[Parameter]:
        """Generate pilot point parameters.

        Parameters
        ----------
        model : IWFMModel
            The IWFM model to parameterize.

        Returns
        -------
        list[Parameter]
            One parameter per pilot point.
        """
        # Get or generate pilot points
        if self.points is not None:
            points = self.points
        else:
            points = self.generate_pilot_point_grid(model)

        parameters = []
        for i, (x, y) in enumerate(points):
            initial = self._get_initial_value(i)

            name = f"{self.param_type.value}_pp{i+1:04d}_l{self.layer}"

            param = Parameter(
                name=name,
                initial_value=initial,
                lower_bound=self.bounds[0],
                upper_bound=self.bounds[1],
                group=self.group_name,
                transform=self.transform,
                param_type=self.param_type,
                layer=self.layer,
                location=(x, y),
                metadata={
                    "pilot_point_index": i,
                    "variogram": self.variogram,
                    "kriging_type": self.kriging_type,
                },
            )
            parameters.append(param)

        return parameters


@dataclass
class DirectParameterization(ParameterizationStrategy):
    """Direct parameterization strategy.

    Creates a single parameter for direct adjustment of a model value.
    Useful for scalar parameters or specific locations.

    Attributes
    ----------
    name : str
        Parameter name.
    initial_value : float
        Initial parameter value.
    layer : int | None
        Model layer (if applicable).
    location_id : int | None
        Location ID (element, node, reach, etc.).
    """

    name: str = ""
    initial_value: float = 1.0
    layer: int | None = None
    location_id: int | None = None

    def __post_init__(self) -> None:
        """Set defaults."""
        super().__post_init__()

        if not self.name:
            self.name = self.param_type.value

    def generate_parameters(self, model: Any) -> list[Parameter]:
        """Generate a single direct parameter.

        Parameters
        ----------
        model : IWFMModel
            The IWFM model (not used for direct params).

        Returns
        -------
        list[Parameter]
            Single parameter.
        """
        param = Parameter(
            name=self.name,
            initial_value=self.initial_value,
            lower_bound=self.bounds[0],
            upper_bound=self.bounds[1],
            group=self.group_name,
            transform=self.transform,
            param_type=self.param_type,
            layer=self.layer,
            metadata={"location_id": self.location_id} if self.location_id else {},
        )
        return [param]


@dataclass
class StreamParameterization(ParameterizationStrategy):
    """Stream-specific parameterization strategy.

    Creates parameters for stream properties by reach or node.

    Attributes
    ----------
    reaches : list[int] | str
        Reach IDs or "all" for all reaches.
    by_node : bool
        If True, create parameters by stream node instead of reach.
    initial_values : float | dict[int, float]
        Initial values by reach/node ID.
    """

    reaches: list[int] | str = "all"
    by_node: bool = False
    initial_values: float | dict[int, float] = 1.0

    def _get_reach_ids(self, model: Any) -> list[int]:
        """Get reach IDs from model or specification."""
        if self.reaches == "all":
            if hasattr(model, 'streams') and hasattr(model.streams, 'reaches'):
                return sorted(model.streams.reaches.keys())
            elif hasattr(model, 'reach_ids'):
                return sorted(model.reach_ids)
            else:
                return [1]  # Default
        return list(self.reaches)

    def _get_initial_value(self, id_: int) -> float:
        """Get initial value for a reach/node."""
        if isinstance(self.initial_values, dict):
            return self.initial_values.get(id_, 1.0)
        return self.initial_values

    def generate_parameters(self, model: Any) -> list[Parameter]:
        """Generate stream parameters.

        Parameters
        ----------
        model : IWFMModel
            The IWFM model to parameterize.

        Returns
        -------
        list[Parameter]
            Stream parameters by reach or node.
        """
        parameters = []
        reach_ids = self._get_reach_ids(model)

        for reach_id in reach_ids:
            initial = self._get_initial_value(reach_id)
            name = f"{self.param_type.value}_r{reach_id}"

            param = Parameter(
                name=name,
                initial_value=initial,
                lower_bound=self.bounds[0],
                upper_bound=self.bounds[1],
                group=self.group_name,
                transform=self.transform,
                param_type=self.param_type,
                metadata={"reach_id": reach_id},
            )
            parameters.append(param)

        return parameters


@dataclass
class RootZoneParameterization(ParameterizationStrategy):
    """Root zone parameterization strategy.

    Creates parameters for root zone properties by land use type.

    Attributes
    ----------
    land_use_types : list[str] | str
        Land use type names or "all".
    crop_ids : list[int] | None
        Crop/land use IDs if not using names.
    initial_values : float | dict[str, float]
        Initial values by land use type.
    """

    land_use_types: list[str] | str = "all"
    crop_ids: list[int] | None = None
    initial_values: float | dict[str, float] = 1.0

    def _get_land_use_types(self, model: Any) -> list[tuple[int, str]]:
        """Get land use types from model."""
        if self.crop_ids is not None:
            return [(cid, f"crop{cid}") for cid in self.crop_ids]

        if self.land_use_types == "all":
            if hasattr(model, 'rootzone') and hasattr(model.rootzone, 'crop_types'):
                return list(model.rootzone.crop_types.items())
            else:
                # Default land use types
                return [
                    (1, "urban"),
                    (2, "agriculture"),
                    (3, "native"),
                    (4, "riparian"),
                ]
        return [(i, name) for i, name in enumerate(self.land_use_types, 1)]

    def _get_initial_value(self, name: str) -> float:
        """Get initial value for a land use type."""
        if isinstance(self.initial_values, dict):
            return self.initial_values.get(name, 1.0)
        return self.initial_values

    def generate_parameters(self, model: Any) -> list[Parameter]:
        """Generate root zone parameters.

        Parameters
        ----------
        model : IWFMModel
            The IWFM model to parameterize.

        Returns
        -------
        list[Parameter]
            Root zone parameters by land use type.
        """
        parameters = []
        land_use_types = self._get_land_use_types(model)

        for crop_id, lu_name in land_use_types:
            initial = self._get_initial_value(lu_name)
            # Clean land use name for parameter naming
            clean_name = lu_name.replace(" ", "_").lower()[:8]
            name = f"{self.param_type.value}_{clean_name}"

            param = Parameter(
                name=name,
                initial_value=initial,
                lower_bound=self.bounds[0],
                upper_bound=self.bounds[1],
                group=self.group_name,
                transform=self.transform,
                param_type=self.param_type,
                metadata={"crop_id": crop_id, "land_use_type": lu_name},
            )
            parameters.append(param)

        return parameters
