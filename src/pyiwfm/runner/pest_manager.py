"""IWFM Parameter Manager for PEST++ setup.

This module provides the IWFMParameterManager class that coordinates
parameter generation and management for PEST++ calibration.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
from numpy.typing import NDArray

from pyiwfm.runner.pest_params import (
    IWFMParameterType,
    MultiplierParameterization,
    Parameter,
    ParameterGroup,
    ParameterizationStrategy,
    ParameterTransform,
    PilotPointParameterization,
    RootZoneParameterization,
    StreamParameterization,
    ZoneParameterization,
)


class IWFMParameterManager:
    """Manages all parameters for an IWFM PEST++ setup.

    This class provides methods to add various types of parameters
    (zone-based, pilot points, multipliers, etc.) and generates
    the appropriate PEST++ parameter definitions.

    Parameters
    ----------
    model : IWFMModel | None
        The IWFM model to parameterize. If None, some features
        will be limited.

    Examples
    --------
    >>> from pyiwfm import IWFMModel
    >>> from pyiwfm.runner.pest_manager import IWFMParameterManager

    >>> model = IWFMModel.from_preprocessor("model/PreProcessor.in")
    >>> pm = IWFMParameterManager(model)

    >>> # Add zone-based hydraulic conductivity
    >>> pm.add_zone_parameters(
    ...     IWFMParameterType.HORIZONTAL_K,
    ...     zones="subregions",
    ...     layer=1,
    ...     bounds=(0.1, 1000.0),
    ... )

    >>> # Add global pumping multiplier
    >>> pm.add_multiplier_parameters(
    ...     IWFMParameterType.PUMPING_MULT,
    ...     spatial="global",
    ...     bounds=(0.8, 1.2),
    ... )

    >>> # Get all parameters
    >>> params = pm.get_all_parameters()
    >>> print(f"Total parameters: {len(params)}")
    """

    def __init__(self, model: Any = None):
        """Initialize the parameter manager.

        Parameters
        ----------
        model : IWFMModel | None
            The IWFM model to parameterize.
        """
        self.model = model
        self._parameters: dict[str, Parameter] = {}
        self._parameter_groups: dict[str, ParameterGroup] = {}
        self._parameterizations: list[ParameterizationStrategy] = []
        self._tied_parameters: dict[str, tuple[str, float]] = {}  # child -> (parent, ratio)

        # Initialize default parameter groups
        self._init_default_groups()

    def _init_default_groups(self) -> None:
        """Initialize default parameter groups."""
        # Create groups for common parameter types
        default_groups = [
            ParameterGroup("hk", inctyp="relative", derinc=0.01),
            ParameterGroup("vk", inctyp="relative", derinc=0.01),
            ParameterGroup("ss", inctyp="relative", derinc=0.01),
            ParameterGroup("sy", inctyp="relative", derinc=0.01),
            ParameterGroup("strk", inctyp="relative", derinc=0.01),
            ParameterGroup("mult", inctyp="relative", derinc=0.01),
            ParameterGroup("rz", inctyp="relative", derinc=0.01),
            ParameterGroup("default", inctyp="relative", derinc=0.01),
        ]
        for group in default_groups:
            self._parameter_groups[group.name] = group

    # -------------------------------------------------------------------------
    # Zone-based parameters
    # -------------------------------------------------------------------------

    def add_zone_parameters(
        self,
        param_type: IWFMParameterType | str,
        zones: list[int] | str = "subregions",
        layer: int | None = None,
        initial_values: float | dict[int, float] = 1.0,
        bounds: tuple[float, float] | None = None,
        transform: str | ParameterTransform = "auto",
        group: str | None = None,
        zone_names: dict[int, str] | None = None,
    ) -> list[Parameter]:
        """Add zone-based parameters.

        Creates one parameter per zone for the specified property.
        Zones can be subregions or custom zone definitions.

        Parameters
        ----------
        param_type : IWFMParameterType | str
            Parameter type to add.
        zones : list[int] | str
            Zone IDs or "subregions" to use model subregions.
        layer : int | None
            Model layer for layered parameters.
        initial_values : float | dict[int, float]
            Initial value(s). Float for uniform, dict for zone-specific.
        bounds : tuple[float, float] | None
            Parameter bounds. None uses type defaults.
        transform : str | ParameterTransform
            Transformation: 'none', 'log', 'auto' (uses type default).
        group : str | None
            Parameter group name. None uses type abbreviation.
        zone_names : dict[int, str] | None
            Optional zone names for parameter naming.

        Returns
        -------
        list[Parameter]
            List of created parameters.

        Examples
        --------
        >>> pm.add_zone_parameters(
        ...     IWFMParameterType.HORIZONTAL_K,
        ...     zones="subregions",
        ...     layer=1,
        ...     bounds=(0.1, 1000.0),
        ... )
        """
        # Convert string to enum if needed
        if isinstance(param_type, str):
            param_type = IWFMParameterType(param_type.lower())

        # Handle transform
        if transform == "auto":
            transform = ParameterTransform(param_type.default_transform)
        elif isinstance(transform, str):
            transform = ParameterTransform(transform.lower())

        # Create parameterization strategy
        strategy = ZoneParameterization(
            param_type=param_type,
            zones=zones,
            layer=layer,
            initial_values=initial_values,
            bounds=bounds,
            transform=transform,
            group_name=group or param_type.value,
            zone_names=zone_names,
        )

        return self._add_parameterization(strategy)

    # -------------------------------------------------------------------------
    # Multiplier parameters
    # -------------------------------------------------------------------------

    def add_multiplier_parameters(
        self,
        param_type: IWFMParameterType | str,
        spatial: str = "global",
        temporal: str = "constant",
        zones: list[int] | None = None,
        initial_value: float = 1.0,
        bounds: tuple[float, float] | None = None,
        transform: str | ParameterTransform = "none",
        group: str | None = None,
        target_file: Path | str | None = None,
    ) -> list[Parameter]:
        """Add multiplier parameters.

        Multipliers adjust existing model values rather than
        replacing them directly.

        Parameters
        ----------
        param_type : IWFMParameterType | str
            Parameter type (typically a _MULT type).
        spatial : str
            Spatial scope: 'global', 'zone', or 'element'.
        temporal : str
            Temporal scope: 'constant', 'seasonal', 'monthly'.
        zones : list[int] | None
            Zone IDs for zone-based multipliers.
        initial_value : float
            Initial multiplier value (typically 1.0).
        bounds : tuple[float, float] | None
            Parameter bounds. None uses type defaults.
        transform : str | ParameterTransform
            Transformation (typically 'none' for multipliers).
        group : str | None
            Parameter group name.
        target_file : Path | str | None
            File containing base values to multiply.

        Returns
        -------
        list[Parameter]
            List of created multiplier parameters.

        Examples
        --------
        >>> # Global pumping multiplier
        >>> pm.add_multiplier_parameters(
        ...     IWFMParameterType.PUMPING_MULT,
        ...     spatial="global",
        ...     bounds=(0.8, 1.2),
        ... )

        >>> # Seasonal ET multipliers
        >>> pm.add_multiplier_parameters(
        ...     IWFMParameterType.ET_MULT,
        ...     temporal="seasonal",
        ...     bounds=(0.9, 1.1),
        ... )
        """
        if isinstance(param_type, str):
            param_type = IWFMParameterType(param_type.lower())

        if isinstance(transform, str):
            transform = ParameterTransform(transform.lower())

        strategy = MultiplierParameterization(
            param_type=param_type,
            spatial_extent=spatial,
            temporal_extent=temporal,
            zones=zones,
            initial_value=initial_value,
            bounds=bounds,
            transform=transform,
            group_name=group or "mult",
            target_file=Path(target_file) if target_file else None,
        )

        return self._add_parameterization(strategy)

    # -------------------------------------------------------------------------
    # Pilot point parameters
    # -------------------------------------------------------------------------

    def add_pilot_points(
        self,
        param_type: IWFMParameterType | str,
        spacing: float | None = None,
        points: list[tuple[float, float]] | None = None,
        layer: int = 1,
        initial_value: float | NDArray = 1.0,
        bounds: tuple[float, float] | None = None,
        transform: str | ParameterTransform = "auto",
        group: str | None = None,
        variogram: dict | None = None,
        kriging_type: str = "ordinary",
        prefix: str | None = None,
    ) -> list[Parameter]:
        """Add pilot point parameters.

        Pilot points enable highly parameterized spatial heterogeneity.
        Values at model nodes/elements are interpolated using kriging.

        Parameters
        ----------
        param_type : IWFMParameterType | str
            Parameter type (e.g., HORIZONTAL_K).
        spacing : float | None
            Regular grid spacing. If None, must provide points.
        points : list[tuple[float, float]] | None
            Explicit pilot point (x, y) coordinates.
        layer : int
            Model layer for these parameters.
        initial_value : float | NDArray
            Initial value(s) at pilot points.
        bounds : tuple[float, float] | None
            Parameter bounds.
        transform : str | ParameterTransform
            Transformation ('auto' uses type default).
        group : str | None
            Parameter group name.
        variogram : dict | None
            Variogram specification: {'type': 'exponential', 'a': 10000, ...}
        kriging_type : str
            Kriging type: 'ordinary', 'simple', 'universal'.
        prefix : str | None
            Custom prefix for parameter names.

        Returns
        -------
        list[Parameter]
            List of pilot point parameters.

        Examples
        --------
        >>> # Regular grid of pilot points
        >>> pm.add_pilot_points(
        ...     IWFMParameterType.HORIZONTAL_K,
        ...     spacing=5000.0,
        ...     layer=1,
        ...     variogram={'type': 'exponential', 'a': 10000, 'sill': 1.0},
        ... )
        """
        if isinstance(param_type, str):
            param_type = IWFMParameterType(param_type.lower())

        if transform == "auto":
            transform = ParameterTransform(param_type.default_transform)
        elif isinstance(transform, str):
            transform = ParameterTransform(transform.lower())

        strategy = PilotPointParameterization(
            param_type=param_type,
            points=points,
            spacing=spacing,
            layer=layer,
            initial_value=initial_value,
            bounds=bounds,
            transform=transform,
            group_name=group or param_type.value,
            variogram=variogram,
            kriging_type=kriging_type,
        )

        return self._add_parameterization(strategy)

    # -------------------------------------------------------------------------
    # Stream parameters
    # -------------------------------------------------------------------------

    def add_stream_parameters(
        self,
        param_type: IWFMParameterType | str,
        reaches: list[int] | str = "all",
        initial_values: float | dict[int, float] = 1.0,
        bounds: tuple[float, float] | None = None,
        transform: str | ParameterTransform = "auto",
        group: str | None = None,
    ) -> list[Parameter]:
        """Add stream-related parameters by reach.

        Parameters
        ----------
        param_type : IWFMParameterType | str
            Stream parameter type (STREAMBED_K, etc.).
        reaches : list[int] | str
            Reach IDs or "all" for all reaches.
        initial_values : float | dict[int, float]
            Initial values by reach ID.
        bounds : tuple[float, float] | None
            Parameter bounds.
        transform : str | ParameterTransform
            Transformation.
        group : str | None
            Parameter group name.

        Returns
        -------
        list[Parameter]
            List of stream parameters.
        """
        if isinstance(param_type, str):
            param_type = IWFMParameterType(param_type.lower())

        if transform == "auto":
            transform = ParameterTransform(param_type.default_transform)
        elif isinstance(transform, str):
            transform = ParameterTransform(transform.lower())

        strategy = StreamParameterization(
            param_type=param_type,
            reaches=reaches,
            initial_values=initial_values,
            bounds=bounds,
            transform=transform,
            group_name=group or "strk",
        )

        return self._add_parameterization(strategy)

    # -------------------------------------------------------------------------
    # Root zone parameters
    # -------------------------------------------------------------------------

    def add_rootzone_parameters(
        self,
        param_type: IWFMParameterType | str,
        land_use_types: list[str] | str = "all",
        initial_values: float | dict[str, float] = 1.0,
        bounds: tuple[float, float] | None = None,
        transform: str | ParameterTransform = "none",
        group: str | None = None,
    ) -> list[Parameter]:
        """Add root zone parameters by land use type.

        Parameters
        ----------
        param_type : IWFMParameterType | str
            Root zone parameter type (CROP_COEFFICIENT, etc.).
        land_use_types : list[str] | str
            Land use type names or "all".
        initial_values : float | dict[str, float]
            Initial values by land use type name.
        bounds : tuple[float, float] | None
            Parameter bounds.
        transform : str | ParameterTransform
            Transformation.
        group : str | None
            Parameter group name.

        Returns
        -------
        list[Parameter]
            List of root zone parameters.
        """
        if isinstance(param_type, str):
            param_type = IWFMParameterType(param_type.lower())

        if isinstance(transform, str):
            transform = ParameterTransform(transform.lower())

        strategy = RootZoneParameterization(
            param_type=param_type,
            land_use_types=land_use_types,
            initial_values=initial_values,
            bounds=bounds,
            transform=transform,
            group_name=group or "rz",
        )

        return self._add_parameterization(strategy)

    # -------------------------------------------------------------------------
    # Direct (single) parameters
    # -------------------------------------------------------------------------

    def add_parameter(
        self,
        name: str,
        param_type: IWFMParameterType | str,
        initial_value: float,
        bounds: tuple[float, float] | None = None,
        transform: str | ParameterTransform = "auto",
        group: str | None = None,
        layer: int | None = None,
        **metadata: Any,
    ) -> Parameter:
        """Add a single direct parameter.

        Parameters
        ----------
        name : str
            Parameter name.
        param_type : IWFMParameterType | str
            Parameter type.
        initial_value : float
            Initial parameter value.
        bounds : tuple[float, float] | None
            Parameter bounds.
        transform : str | ParameterTransform
            Transformation.
        group : str | None
            Parameter group name.
        layer : int | None
            Model layer (if applicable).
        **metadata : Any
            Additional metadata.

        Returns
        -------
        Parameter
            The created parameter.
        """
        if isinstance(param_type, str):
            param_type = IWFMParameterType(param_type.lower())

        if transform == "auto":
            transform = ParameterTransform(param_type.default_transform)
        elif isinstance(transform, str):
            transform = ParameterTransform(transform.lower())

        if bounds is None:
            bounds = param_type.default_bounds

        param = Parameter(
            name=name,
            initial_value=initial_value,
            lower_bound=bounds[0],
            upper_bound=bounds[1],
            group=group or param_type.value,
            transform=transform,
            param_type=param_type,
            layer=layer,
            metadata=metadata,
        )

        self._parameters[name] = param
        return param

    # -------------------------------------------------------------------------
    # Parameter relationships
    # -------------------------------------------------------------------------

    def tie_parameters(
        self,
        parent: str,
        children: list[str],
        ratios: float | list[float] = 1.0,
    ) -> None:
        """Set up tied parameters (children follow parent).

        Tied parameters are adjusted as a ratio of their parent
        parameter, reducing the effective number of adjustable
        parameters while maintaining relationships.

        Parameters
        ----------
        parent : str
            Parent parameter name.
        children : list[str]
            Child parameter names.
        ratios : float | list[float]
            Ratio(s) to parent. If float, used for all children.
        """
        if parent not in self._parameters:
            raise ValueError(f"Parent parameter '{parent}' not found")

        if isinstance(ratios, (int, float)):
            ratios = [float(ratios)] * len(children)

        if len(ratios) != len(children):
            raise ValueError(
                f"Number of ratios ({len(ratios)}) must match number of children ({len(children)})"
            )

        for child, ratio in zip(children, ratios, strict=False):
            if child not in self._parameters:
                raise ValueError(f"Child parameter '{child}' not found")

            # Update parameter transform
            self._parameters[child].transform = ParameterTransform.TIED
            self._parameters[child].tied_to = parent
            self._parameters[child].tied_ratio = ratio
            self._tied_parameters[child] = (parent, ratio)

    def fix_parameter(self, name: str) -> None:
        """Fix a parameter (no adjustment during calibration).

        Parameters
        ----------
        name : str
            Parameter name to fix.
        """
        if name not in self._parameters:
            raise ValueError(f"Parameter '{name}' not found")

        self._parameters[name].transform = ParameterTransform.FIXED

    def unfix_parameter(
        self,
        name: str,
        transform: str | ParameterTransform = "auto",
    ) -> None:
        """Unfix a parameter.

        Parameters
        ----------
        name : str
            Parameter name to unfix.
        transform : str | ParameterTransform
            Transform to apply after unfixing.
        """
        if name not in self._parameters:
            raise ValueError(f"Parameter '{name}' not found")

        param = self._parameters[name]

        if transform == "auto" and param.param_type:
            transform = ParameterTransform(param.param_type.default_transform)
        elif isinstance(transform, str):
            transform = ParameterTransform(transform.lower())

        param.transform = transform

    # -------------------------------------------------------------------------
    # Parameter groups
    # -------------------------------------------------------------------------

    def add_parameter_group(
        self,
        name: str,
        inctyp: str = "relative",
        derinc: float = 0.01,
        **kwargs: Any,
    ) -> ParameterGroup:
        """Add or update a parameter group.

        Parameters
        ----------
        name : str
            Group name (max 12 characters).
        inctyp : str
            Increment type: 'relative' or 'absolute'.
        derinc : float
            Derivative increment.
        **kwargs : Any
            Additional group settings.

        Returns
        -------
        ParameterGroup
            The created or updated group.
        """
        group = ParameterGroup(
            name=name,
            inctyp=inctyp,
            derinc=derinc,
            **kwargs,
        )
        self._parameter_groups[group.name] = group
        return group

    def get_parameter_group(self, name: str) -> ParameterGroup | None:
        """Get a parameter group by name."""
        return self._parameter_groups.get(name)

    # -------------------------------------------------------------------------
    # Parameter access
    # -------------------------------------------------------------------------

    def get_parameter(self, name: str) -> Parameter | None:
        """Get a parameter by name."""
        return self._parameters.get(name)

    def get_parameters_by_type(
        self,
        param_type: IWFMParameterType | str,
    ) -> list[Parameter]:
        """Get all parameters of a specific type."""
        if isinstance(param_type, str):
            param_type = IWFMParameterType(param_type.lower())

        return [p for p in self._parameters.values() if p.param_type == param_type]

    def get_parameters_by_group(self, group: str) -> list[Parameter]:
        """Get all parameters in a specific group."""
        return [p for p in self._parameters.values() if p.group == group]

    def get_parameters_by_layer(self, layer: int) -> list[Parameter]:
        """Get all parameters for a specific layer."""
        return [p for p in self._parameters.values() if p.layer == layer]

    def get_pilot_point_parameters(self) -> list[Parameter]:
        """Get all pilot point parameters."""
        return [p for p in self._parameters.values() if p.location is not None]

    def get_all_parameters(self) -> list[Parameter]:
        """Get all parameters as a list."""
        return list(self._parameters.values())

    def get_adjustable_parameters(self) -> list[Parameter]:
        """Get all adjustable (non-fixed, non-tied) parameters."""
        return [
            p
            for p in self._parameters.values()
            if p.transform not in {ParameterTransform.FIXED, ParameterTransform.TIED}
        ]

    def get_all_groups(self) -> list[ParameterGroup]:
        """Get all parameter groups that have parameters."""
        used_groups = {p.group for p in self._parameters.values()}
        return [g for g in self._parameter_groups.values() if g.name in used_groups]

    # -------------------------------------------------------------------------
    # DataFrame export
    # -------------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Export parameters to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with parameter information.

        Raises
        ------
        ImportError
            If pandas is not available.
        """
        data = []
        for param in self._parameters.values():
            row = {
                "name": param.name,
                "initial_value": param.initial_value,
                "lower_bound": param.lower_bound,
                "upper_bound": param.upper_bound,
                "group": param.group,
                "transform": param.transform.value,
                "param_type": param.param_type.value if param.param_type else None,
                "layer": param.layer,
                "zone": param.zone,
                "x": param.location[0] if param.location else None,
                "y": param.location[1] if param.location else None,
                "tied_to": param.tied_to,
                "tied_ratio": param.tied_ratio,
            }
            data.append(row)

        return pd.DataFrame(data)

    def from_dataframe(self, df: pd.DataFrame) -> None:
        """Load parameter values from a DataFrame.

        Useful for loading calibrated parameter values.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'name' and 'initial_value' columns.
        """
        for _, row in df.iterrows():
            name = row["name"]
            if name in self._parameters:
                self._parameters[name].initial_value = row["initial_value"]

    # -------------------------------------------------------------------------
    # File I/O
    # -------------------------------------------------------------------------

    def write_parameter_file(self, filepath: Path | str) -> None:
        """Write parameter values to a file.

        Parameters
        ----------
        filepath : Path | str
            Output file path.
        """
        filepath = Path(filepath)

        with open(filepath, "w") as f:
            f.write("# IWFM PEST++ Parameter File\n")
            f.write("# name, initial_value, lower_bound, upper_bound, group, transform\n")
            for param in self._parameters.values():
                f.write(
                    f"{param.name}, {param.initial_value:.8e}, "
                    f"{param.lower_bound:.8e}, {param.upper_bound:.8e}, "
                    f"{param.group}, {param.transform.value}\n"
                )

    def read_parameter_file(self, filepath: Path | str) -> None:
        """Read parameter values from a file.

        Updates initial values for existing parameters.

        Parameters
        ----------
        filepath : Path | str
            Input file path.
        """
        filepath = Path(filepath)

        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    name = parts[0]
                    value = float(parts[1])
                    if name in self._parameters:
                        self._parameters[name].initial_value = value

    # -------------------------------------------------------------------------
    # Statistics and summary
    # -------------------------------------------------------------------------

    @property
    def n_parameters(self) -> int:
        """Total number of parameters."""
        return len(self._parameters)

    @property
    def n_adjustable(self) -> int:
        """Number of adjustable parameters."""
        return len(self.get_adjustable_parameters())

    @property
    def n_groups(self) -> int:
        """Number of parameter groups in use."""
        return len(self.get_all_groups())

    def summary(self) -> str:
        """Get a summary of parameters.

        Returns
        -------
        str
            Summary string.
        """
        lines = [
            "IWFM Parameter Manager Summary",
            "=" * 40,
            f"Total parameters: {self.n_parameters}",
            f"Adjustable parameters: {self.n_adjustable}",
            f"Fixed parameters: {len([p for p in self._parameters.values() if p.transform == ParameterTransform.FIXED])}",
            f"Tied parameters: {len([p for p in self._parameters.values() if p.transform == ParameterTransform.TIED])}",
            f"Pilot point parameters: {len(self.get_pilot_point_parameters())}",
            "",
            "Parameters by type:",
        ]

        # Count by type
        type_counts: dict[str, int] = {}
        for param in self._parameters.values():
            if param.param_type:
                type_name = param.param_type.value
            else:
                type_name = "unknown"
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        for type_name, count in sorted(type_counts.items()):
            lines.append(f"  {type_name}: {count}")

        lines.append("")
        lines.append("Parameters by group:")

        # Count by group
        group_counts: dict[str, int] = {}
        for param in self._parameters.values():
            group_counts[param.group] = group_counts.get(param.group, 0) + 1

        for group_name, count in sorted(group_counts.items()):
            lines.append(f"  {group_name}: {count}")

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Internal methods
    # -------------------------------------------------------------------------

    def _add_parameterization(
        self,
        strategy: ParameterizationStrategy,
    ) -> list[Parameter]:
        """Add a parameterization strategy and generate parameters.

        Parameters
        ----------
        strategy : ParameterizationStrategy
            The parameterization strategy to add.

        Returns
        -------
        list[Parameter]
            List of generated parameters.
        """
        self._parameterizations.append(strategy)

        # Ensure parameter group exists
        group_name = strategy.group_name
        if group_name and group_name not in self._parameter_groups:
            self.add_parameter_group(group_name)

        # Generate parameters
        parameters = strategy.generate_parameters(self.model)

        # Add to collection
        for param in parameters:
            if param.name in self._parameters:
                # Parameter already exists - update it
                existing = self._parameters[param.name]
                existing.initial_value = param.initial_value
                existing.lower_bound = param.lower_bound
                existing.upper_bound = param.upper_bound
            else:
                self._parameters[param.name] = param

        return parameters

    def __iter__(self) -> Iterator[Parameter]:
        """Iterate over all parameters."""
        return iter(self._parameters.values())

    def __len__(self) -> int:
        """Return number of parameters."""
        return len(self._parameters)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"IWFMParameterManager(n_parameters={self.n_parameters}, "
            f"n_adjustable={self.n_adjustable}, n_groups={self.n_groups})"
        )
