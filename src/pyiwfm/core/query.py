"""
High-level query API for multi-scale data access.

This module provides a unified interface for querying model data at
different spatial scales:

- :class:`ModelQueryAPI`: Main query interface with export capabilities

Example
-------
Query and export model data:

>>> from pyiwfm.core.query import ModelQueryAPI
>>> api = ModelQueryAPI(model)
>>> # Get zone-aggregated values
>>> zone_heads = api.get_values("head", scale="subregion", layer=1)
>>> # Export to DataFrame
>>> df = api.export_to_dataframe(["head", "kh"], scale="subregion")
>>> df.to_csv("subregion_data.csv")
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from pyiwfm.core.aggregation import DataAggregator, create_aggregator_from_grid
from pyiwfm.core.zones import ZoneDefinition

if TYPE_CHECKING:
    import pandas as pd

    from pyiwfm.core.model import IWFMModel


@dataclass
class TimeSeries:
    """
    Time series data for a single location or zone.

    Parameters
    ----------
    times : list of datetime
        Timestamps for each value.
    values : NDArray
        Array of values at each timestamp.
    variable : str
        Variable name (e.g., "head", "pumping").
    location_id : int
        Zone, element, or node ID.
    location_type : str
        Type of location: "zone", "element", "node".
    units : str, optional
        Units of the values.
    """

    times: list[datetime]
    values: NDArray[np.float64]
    variable: str
    location_id: int
    location_type: str
    units: str = ""

    @property
    def n_timesteps(self) -> int:
        """Return number of timesteps."""
        return len(self.times)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        return {
            "time": self.times,
            "value": list(self.values),
            "variable": self.variable,
            "location_id": self.location_id,
            "location_type": self.location_type,
            "units": self.units,
        }


class ModelQueryAPI:
    """
    High-level API for querying model data at multiple spatial scales.

    Parameters
    ----------
    model : IWFMModel
        The IWFM model instance.

    Attributes
    ----------
    model : IWFMModel
        The underlying model.
    aggregator : DataAggregator
        Aggregator for zone-level calculations.

    Examples
    --------
    Create API and query data:

    >>> from pyiwfm.core.model import IWFMModel
    >>> model = IWFMModel.from_preprocessor("Preprocessor_MAIN.IN")
    >>> api = ModelQueryAPI(model)

    Get element-level values:

    >>> heads = api.get_values("head", scale="element", layer=1)
    >>> print(f"Head at element 1: {heads[1]:.2f} ft")

    Get zone-aggregated values:

    >>> zone_heads = api.get_values("head", scale="subregion", layer=1)
    >>> for zone_id, head in zone_heads.items():
    ...     print(f"Zone {zone_id}: {head:.2f} ft")

    Export to DataFrame:

    >>> df = api.export_to_dataframe(["head", "kh"], scale="subregion")
    >>> df.to_csv("zone_data.csv", index=False)
    """

    # Map of known properties to their metadata
    PROPERTY_INFO: dict[str, dict[str, Any]] = {
        "head": {"name": "Hydraulic Head", "units": "ft", "source": "results"},
        "kh": {"name": "Horizontal K", "units": "ft/d", "source": "params"},
        "kv": {"name": "Vertical K", "units": "ft/d", "source": "params"},
        "ss": {"name": "Specific Storage", "units": "1/ft", "source": "params"},
        "sy": {"name": "Specific Yield", "units": "", "source": "params"},
        "thickness": {"name": "Layer Thickness", "units": "ft", "source": "stratigraphy"},
        "top_elev": {"name": "Top Elevation", "units": "ft", "source": "stratigraphy"},
        "bottom_elev": {"name": "Bottom Elevation", "units": "ft", "source": "stratigraphy"},
        "area": {"name": "Element Area", "units": "sq ft", "source": "mesh"},
        "subregion": {"name": "Subregion ID", "units": "", "source": "mesh"},
    }

    def __init__(self, model: IWFMModel):
        self.model = model
        self._aggregator: DataAggregator | None = None
        self._zone_definitions: dict[str, ZoneDefinition] = {}
        self._cached_subregion_def: ZoneDefinition | None = None

    @property
    def aggregator(self) -> DataAggregator:
        """Get or create the data aggregator."""
        if self._aggregator is None and self.model.mesh is not None:
            self._aggregator = create_aggregator_from_grid(self.model.mesh)
        if self._aggregator is None:
            raise RuntimeError("Cannot create aggregator: model has no mesh")
        return self._aggregator

    @property
    def subregion_zones(self) -> ZoneDefinition:
        """Get zone definition from model subregions."""
        if self._cached_subregion_def is None and self.model.mesh is not None:
            self._cached_subregion_def = ZoneDefinition.from_subregions(self.model.mesh)
        if self._cached_subregion_def is None:
            raise RuntimeError("Cannot get subregions: model has no mesh")
        return self._cached_subregion_def

    def register_zones(self, name: str, zone_def: ZoneDefinition) -> None:
        """
        Register a custom zone definition.

        Parameters
        ----------
        name : str
            Name to identify this zone definition.
        zone_def : ZoneDefinition
            The zone definition to register.
        """
        self._zone_definitions[name] = zone_def

    def get_zone_definition(self, scale: str) -> ZoneDefinition | None:
        """
        Get a zone definition by scale name.

        Parameters
        ----------
        scale : str
            Scale name: "element" (no zones), "subregion", or custom name.

        Returns
        -------
        ZoneDefinition or None
            The zone definition, or None for element scale.
        """
        if scale == "element":
            return None
        if scale == "subregion":
            return self.subregion_zones
        return self._zone_definitions.get(scale)

    def get_values(
        self,
        variable: str,
        scale: str = "element",
        layer: int | None = None,
        time_index: int | None = None,
        aggregation: str = "area_weighted_mean",
    ) -> dict[int, float]:
        """
        Get property values at specified scale.

        Parameters
        ----------
        variable : str
            Property name (e.g., "head", "kh", "area").
        scale : str, optional
            Spatial scale: "element", "subregion", or custom zone name.
            Default is "element".
        layer : int, optional
            Model layer (1-based). None for 2D properties.
        time_index : int, optional
            Time index for time-varying properties.
        aggregation : str, optional
            Aggregation method for zone scales. Default is "area_weighted_mean".

        Returns
        -------
        dict[int, float]
            Dictionary mapping element/zone ID to value.

        Examples
        --------
        Get element-level heads:

        >>> heads = api.get_values("head", scale="element", layer=1)

        Get subregion-average heads:

        >>> zone_heads = api.get_values("head", scale="subregion", layer=1)
        """
        # Get element-level values first
        element_values = self._get_element_values(variable, layer, time_index)

        if scale == "element":
            # Return element-level directly
            return {i + 1: float(v) for i, v in enumerate(element_values)}

        # Get zone definition and aggregate
        zone_def = self.get_zone_definition(scale)
        if zone_def is None:
            raise ValueError(f"Unknown scale: {scale!r}")

        zone_values = self.aggregator.aggregate(element_values, zone_def, aggregation)
        return zone_values

    def get_timeseries(
        self,
        variable: str,
        location_id: int,
        scale: str = "element",
        layer: int | None = None,
        aggregation: str = "area_weighted_mean",
    ) -> TimeSeries | None:
        """
        Get time series for a location at specified scale.

        Parameters
        ----------
        variable : str
            Property name (e.g., "head").
        location_id : int
            Element or zone ID.
        scale : str, optional
            Spatial scale. Default is "element".
        layer : int, optional
            Model layer (1-based).
        aggregation : str, optional
            Aggregation method for zone scales.

        Returns
        -------
        TimeSeries or None
            Time series data, or None if not available.

        Notes
        -----
        Requires model to have time-varying data loaded.
        """
        # Check if model has time series data
        if not hasattr(self.model, "results") or self.model.results is None:
            return None

        results = self.model.results
        if not hasattr(results, "times") or not results.times:
            return None

        times = results.times
        values_list = []

        for t_idx in range(len(times)):
            vals = self.get_values(variable, scale, layer, t_idx, aggregation)
            if location_id in vals:
                values_list.append(vals[location_id])
            else:
                values_list.append(np.nan)

        units = self.PROPERTY_INFO.get(variable, {}).get("units", "")
        location_type = "element" if scale == "element" else "zone"

        return TimeSeries(
            times=times,
            values=np.array(values_list),
            variable=variable,
            location_id=location_id,
            location_type=location_type,
            units=units,
        )

    def export_to_dataframe(
        self,
        variables: list[str],
        scale: str = "element",
        layer: int | None = None,
        time_index: int | None = None,
        aggregation: str = "area_weighted_mean",
    ) -> pd.DataFrame:
        """
        Export values to a pandas DataFrame.

        Parameters
        ----------
        variables : list of str
            Property names to include.
        scale : str, optional
            Spatial scale. Default is "element".
        layer : int, optional
            Model layer (1-based).
        time_index : int, optional
            Time index for time-varying properties.
        aggregation : str, optional
            Aggregation method for zone scales.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns for location ID and each variable.

        Examples
        --------
        >>> df = api.export_to_dataframe(["head", "kh", "area"], scale="subregion")
        >>> print(df.head())
        """
        import pandas as pd

        # Collect all values
        data: dict[str, list] = {"id": [], "name": []}

        # Get location IDs and names
        zone_def = self.get_zone_definition(scale)
        if zone_def is None:
            # Element scale
            if self.model.mesh:
                for elem_id in sorted(self.model.mesh.elements.keys()):
                    data["id"].append(elem_id)
                    data["name"].append(f"Element {elem_id}")
        else:
            # Zone scale
            for zone_id in sorted(zone_def.zones.keys()):
                zone = zone_def.zones[zone_id]
                data["id"].append(zone_id)
                data["name"].append(zone.name)

        # Get values for each variable
        for var in variables:
            try:
                values = self.get_values(var, scale, layer, time_index, aggregation)
                data[var] = [values.get(loc_id, np.nan) for loc_id in data["id"]]
            except Exception:
                # Variable not available
                data[var] = [np.nan] * len(data["id"])

        return pd.DataFrame(data)

    def export_to_csv(
        self,
        variables: list[str],
        filepath: Path | str,
        scale: str = "element",
        layer: int | None = None,
        time_index: int | None = None,
        aggregation: str = "area_weighted_mean",
    ) -> None:
        """
        Export values to a CSV file.

        Parameters
        ----------
        variables : list of str
            Property names to include.
        filepath : Path or str
            Output file path.
        scale : str, optional
            Spatial scale. Default is "element".
        layer : int, optional
            Model layer (1-based).
        time_index : int, optional
            Time index for time-varying properties.
        aggregation : str, optional
            Aggregation method for zone scales.

        Examples
        --------
        >>> api.export_to_csv(["head", "kh"], "zone_data.csv", scale="subregion")
        """
        df = self.export_to_dataframe(variables, scale, layer, time_index, aggregation)
        df.to_csv(filepath, index=False)

    def export_timeseries_to_csv(
        self,
        variable: str,
        location_ids: list[int],
        filepath: Path | str,
        scale: str = "element",
        layer: int | None = None,
        aggregation: str = "area_weighted_mean",
    ) -> None:
        """
        Export time series for multiple locations to CSV.

        Parameters
        ----------
        variable : str
            Property name.
        location_ids : list of int
            List of element or zone IDs.
        filepath : Path or str
            Output file path.
        scale : str, optional
            Spatial scale. Default is "element".
        layer : int, optional
            Model layer (1-based).
        aggregation : str, optional
            Aggregation method for zone scales.
        """
        import pandas as pd

        data: dict[str, list] = {"time": []}
        ts_dict: dict[int, TimeSeries] = {}

        # Get time series for each location
        for loc_id in location_ids:
            ts = self.get_timeseries(variable, loc_id, scale, layer, aggregation)
            if ts:
                ts_dict[loc_id] = ts
                if not data["time"]:
                    data["time"] = [t.isoformat() for t in ts.times]
                data[f"{variable}_{loc_id}"] = list(ts.values)

        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)

    def get_available_variables(self) -> list[str]:
        """
        Get list of available variables for this model.

        Returns
        -------
        list of str
            Variable names that can be queried.
        """
        available = []

        # Always available if mesh exists
        if self.model.mesh:
            available.extend(["area", "subregion"])

        # Stratigraphy
        if self.model.stratigraphy:
            available.extend(["thickness", "top_elev", "bottom_elev"])

        # Groundwater parameters
        if self.model.groundwater:
            available.extend(["kh", "kv", "ss", "sy"])

        # Time-varying results
        if hasattr(self.model, "results") and self.model.results:
            available.append("head")

        return available

    def get_available_scales(self) -> list[str]:
        """
        Get list of available spatial scales.

        Returns
        -------
        list of str
            Scale names that can be used.
        """
        scales = ["element"]

        # Check for subregions
        if self.model.mesh and self.model.mesh.subregions:
            scales.append("subregion")

        # Add registered custom zones
        scales.extend(self._zone_definitions.keys())

        return scales

    def _get_element_values(
        self,
        variable: str,
        layer: int | None = None,
        time_index: int | None = None,
    ) -> NDArray[np.float64]:
        """
        Get element-level values for a variable.

        Internal method to retrieve raw element data.
        """
        if self.model.mesh is None:
            raise RuntimeError("Model has no mesh")

        max_elem_id = max(self.model.mesh.elements.keys()) if self.model.mesh.elements else 0
        values = np.full(max_elem_id, np.nan, dtype=np.float64)

        # Handle different variable sources
        if variable == "area":
            for elem_id, elem in self.model.mesh.elements.items():
                values[elem_id - 1] = elem.area

        elif variable == "subregion":
            for elem_id, elem in self.model.mesh.elements.items():
                values[elem_id - 1] = float(elem.subregion)

        elif variable in ("thickness", "top_elev", "bottom_elev"):
            if self.model.stratigraphy is None:
                return values  # Return NaN array

            strat = self.model.stratigraphy
            layer_idx = (layer or 1) - 1  # Convert to 0-based

            if variable == "thickness":
                # Average thickness per element
                # Compute thicknesses from top_elev - bottom_elev
                layer_thicknesses = strat.top_elev - strat.bottom_elev
                for elem_id, elem in self.model.mesh.elements.items():
                    node_ids = elem.vertices
                    thick_vals = []
                    for nid in node_ids:
                        node_idx = nid - 1
                        if (
                            node_idx < layer_thicknesses.shape[0]
                            and layer_idx < layer_thicknesses.shape[1]
                        ):
                            thick_vals.append(layer_thicknesses[node_idx, layer_idx])
                    if thick_vals:
                        values[elem_id - 1] = np.mean(thick_vals)

            elif variable == "top_elev":
                for elem_id, elem in self.model.mesh.elements.items():
                    node_ids = elem.vertices
                    elevs = []
                    for nid in node_ids:
                        node_idx = nid - 1
                        if (
                            node_idx < strat.top_elev.shape[0]
                            and layer_idx < strat.top_elev.shape[1]
                        ):
                            elevs.append(strat.top_elev[node_idx, layer_idx])
                    if elevs:
                        values[elem_id - 1] = np.mean(elevs)

            elif variable == "bottom_elev":
                for elem_id, elem in self.model.mesh.elements.items():
                    node_ids = elem.vertices
                    elevs = []
                    for nid in node_ids:
                        node_idx = nid - 1
                        if (
                            node_idx < strat.bottom_elev.shape[0]
                            and layer_idx < strat.bottom_elev.shape[1]
                        ):
                            elevs.append(strat.bottom_elev[node_idx, layer_idx])
                    if elevs:
                        values[elem_id - 1] = np.mean(elevs)

        elif variable in ("kh", "kv", "ss", "sy"):
            if self.model.groundwater is None or not hasattr(
                self.model.groundwater, "aquifer_params"
            ):
                return values

            params = self.model.groundwater.aquifer_params
            if variable in params:  # type: ignore[operator]
                param_arr = params[variable]  # type: ignore[index]
                layer_idx = (layer or 1) - 1

                # Params may be node-based; average to elements
                for elem_id, elem in self.model.mesh.elements.items():
                    node_ids = elem.vertices
                    vals = []
                    for nid in node_ids:
                        node_idx = nid - 1
                        if param_arr.ndim == 1:
                            if node_idx < len(param_arr):
                                vals.append(param_arr[node_idx])
                        else:
                            if node_idx < param_arr.shape[0] and layer_idx < param_arr.shape[1]:
                                vals.append(param_arr[node_idx, layer_idx])
                    if vals:
                        values[elem_id - 1] = np.mean(vals)

        elif variable == "head":
            if hasattr(self.model, "results") and self.model.results:
                results = self.model.results
                t_idx = time_index or 0
                layer_idx = (layer or 1) - 1

                if hasattr(results, "head") and results.head is not None:
                    head_data = results.head
                    # Handle different shapes
                    if head_data.ndim == 3:  # (time, nodes, layers)
                        head_t = head_data[t_idx, :, layer_idx]
                    elif head_data.ndim == 2:  # (time, nodes) or (nodes, layers)
                        head_t = head_data[t_idx, :]
                    else:
                        head_t = head_data

                    # Average node values to elements
                    for elem_id, elem in self.model.mesh.elements.items():
                        node_ids = elem.vertices
                        vals = [head_t[nid - 1] for nid in node_ids if nid - 1 < len(head_t)]
                        if vals:
                            values[elem_id - 1] = np.mean(vals)

        return values

    def __repr__(self) -> str:
        n_zones = len(self._zone_definitions)
        scales = self.get_available_scales()
        return f"ModelQueryAPI(model={self.model.name!r}, scales={scales}, custom_zones={n_zones})"
