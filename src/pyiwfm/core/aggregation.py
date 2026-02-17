"""
Spatial data aggregation for multi-scale viewing.

This module provides aggregation utilities for computing zone-level values
from element-level data:

- :class:`DataAggregator`: Main aggregation engine with multiple methods

Supported aggregation methods:
- sum: Total value across zone
- mean: Simple average
- area_weighted_mean: Area-weighted average (default)
- min: Minimum value
- max: Maximum value
- median: Median value

Example
-------
Aggregate element values to zones:

>>> from pyiwfm.core.aggregation import DataAggregator
>>> from pyiwfm.core.zones import ZoneDefinition
>>> import numpy as np
>>> aggregator = DataAggregator()
>>> element_values = np.array([10.0, 12.0, 11.0, 20.0, 22.0])  # 5 elements
>>> zone_def = ...  # ZoneDefinition with 2 zones
>>> zone_values = aggregator.aggregate(element_values, zone_def, method="mean")
>>> print(zone_values)
{1: 11.0, 2: 21.0}
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from pyiwfm.core.zones import ZoneDefinition

if TYPE_CHECKING:
    from pyiwfm.core.mesh import AppGrid


class AggregationMethod(Enum):
    """Available aggregation methods."""

    SUM = "sum"
    MEAN = "mean"
    AREA_WEIGHTED_MEAN = "area_weighted_mean"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"


class DataAggregator:
    """
    Aggregates element-level data to zone-level values.

    Parameters
    ----------
    element_areas : NDArray, optional
        Array of element areas for area-weighted calculations.
        Shape: (n_elements,). Required for area_weighted_mean.

    Examples
    --------
    Create an aggregator with element areas:

    >>> areas = np.array([100.0, 150.0, 200.0, 125.0, 175.0])
    >>> aggregator = DataAggregator(element_areas=areas)

    Aggregate hydraulic head values:

    >>> head_values = np.array([50.0, 52.0, 48.0, 60.0, 62.0])
    >>> zone_def = ...  # Zone 1: elements 1-3, Zone 2: elements 4-5
    >>> zone_heads = aggregator.aggregate(head_values, zone_def, method="area_weighted_mean")
    """

    METHODS: dict[str, Callable] = {}

    def __init__(self, element_areas: NDArray[np.float64] | None = None):
        self.element_areas = element_areas
        self._method_funcs: dict[str, Callable] = {
            "sum": self._agg_sum,
            "mean": self._agg_mean,
            "area_weighted_mean": self._agg_area_weighted_mean,
            "min": self._agg_min,
            "max": self._agg_max,
            "median": self._agg_median,
        }

    def aggregate(
        self,
        values: NDArray[np.float64],
        zone_def: ZoneDefinition,
        method: str = "area_weighted_mean",
    ) -> dict[int, float]:
        """
        Aggregate element values to zone-level values.

        Parameters
        ----------
        values : NDArray
            Element-level values. Shape: (n_elements,).
        zone_def : ZoneDefinition
            Zone definition with element-to-zone mapping.
        method : str, optional
            Aggregation method. One of: "sum", "mean", "area_weighted_mean",
            "min", "max", "median". Default is "area_weighted_mean".

        Returns
        -------
        dict[int, float]
            Dictionary mapping zone ID to aggregated value.

        Raises
        ------
        ValueError
            If method is not recognized or areas not provided for area_weighted_mean.

        Examples
        --------
        >>> aggregator = DataAggregator(element_areas=areas)
        >>> result = aggregator.aggregate(heads, zone_def, method="mean")
        >>> print(f"Zone 1 mean head: {result[1]:.2f}")
        """
        if method not in self._method_funcs:
            raise ValueError(
                f"Unknown aggregation method: {method!r}. "
                f"Available: {list(self._method_funcs.keys())}"
            )

        if method == "area_weighted_mean" and self.element_areas is None:
            raise ValueError("element_areas required for area_weighted_mean aggregation")

        agg_func = self._method_funcs[method]
        result: dict[int, float] = {}

        for zone_id, zone in zone_def.zones.items():
            if not zone.elements:
                result[zone_id] = np.nan
                continue

            # Get element indices (0-based)
            indices = np.array([e - 1 for e in zone.elements if 0 < e <= len(values)])

            if len(indices) == 0:
                result[zone_id] = np.nan
                continue

            zone_values = values[indices]

            if method == "area_weighted_mean":
                assert self.element_areas is not None
                zone_areas = self.element_areas[indices]
                result[zone_id] = agg_func(zone_values, zone_areas)
            else:
                result[zone_id] = agg_func(zone_values)

        return result

    def aggregate_to_array(
        self,
        values: NDArray[np.float64],
        zone_def: ZoneDefinition,
        method: str = "area_weighted_mean",
    ) -> NDArray[np.float64]:
        """
        Aggregate and expand zone values back to element array.

        Each element gets its zone's aggregated value. Useful for visualization
        where you want to color elements by their zone's aggregate value.

        Parameters
        ----------
        values : NDArray
            Element-level values. Shape: (n_elements,).
        zone_def : ZoneDefinition
            Zone definition with element-to-zone mapping.
        method : str, optional
            Aggregation method. Default is "area_weighted_mean".

        Returns
        -------
        NDArray
            Array of same shape as input, with each element's value replaced
            by its zone's aggregated value. Elements with no zone get NaN.

        Examples
        --------
        >>> expanded = aggregator.aggregate_to_array(heads, zone_def)
        >>> # expanded[0:3] all have same value (Zone 1's aggregate)
        """
        zone_values = self.aggregate(values, zone_def, method)
        result = np.full(len(values), np.nan, dtype=np.float64)

        if zone_def.element_zones is not None:
            for i, zone_id in enumerate(zone_def.element_zones):
                if zone_id in zone_values:
                    result[i] = zone_values[zone_id]

        return result

    def aggregate_timeseries(
        self,
        timeseries_values: list[NDArray[np.float64]],
        zone_def: ZoneDefinition,
        method: str = "area_weighted_mean",
    ) -> dict[int, list[float]]:
        """
        Aggregate a time series of element values to zone-level time series.

        Parameters
        ----------
        timeseries_values : list of NDArray
            List of element-level value arrays, one per timestep.
        zone_def : ZoneDefinition
            Zone definition with element-to-zone mapping.
        method : str, optional
            Aggregation method. Default is "area_weighted_mean".

        Returns
        -------
        dict[int, list[float]]
            Dictionary mapping zone ID to list of aggregated values over time.

        Examples
        --------
        >>> head_series = [heads_t0, heads_t1, heads_t2]  # 3 timesteps
        >>> zone_series = aggregator.aggregate_timeseries(head_series, zone_def)
        >>> print(f"Zone 1 heads over time: {zone_series[1]}")
        """
        result: dict[int, list[float]] = {zone_id: [] for zone_id in zone_def.zones}

        for values in timeseries_values:
            zone_values = self.aggregate(values, zone_def, method)
            for zone_id, val in zone_values.items():
                result[zone_id].append(val)

        return result

    def set_element_areas(self, areas: NDArray[np.float64]) -> None:
        """
        Set or update element areas for area-weighted aggregation.

        Parameters
        ----------
        areas : NDArray
            Array of element areas. Shape: (n_elements,).
        """
        self.element_areas = areas

    @staticmethod
    def _agg_sum(values: NDArray[np.float64], areas: NDArray[np.float64] | None = None) -> float:
        """Compute sum of values."""
        return float(np.nansum(values))

    @staticmethod
    def _agg_mean(values: NDArray[np.float64], areas: NDArray[np.float64] | None = None) -> float:
        """Compute arithmetic mean of values."""
        return float(np.nanmean(values))

    @staticmethod
    def _agg_area_weighted_mean(values: NDArray[np.float64], areas: NDArray[np.float64]) -> float:
        """Compute area-weighted mean of values."""
        # Handle NaN values
        mask = ~np.isnan(values)
        if not np.any(mask):
            return np.nan

        valid_values = values[mask]
        valid_areas = areas[mask]
        total_area = np.sum(valid_areas)

        if total_area == 0:
            return float(np.nanmean(valid_values))

        return float(np.sum(valid_values * valid_areas) / total_area)

    @staticmethod
    def _agg_min(values: NDArray[np.float64], areas: NDArray[np.float64] | None = None) -> float:
        """Compute minimum value."""
        return float(np.nanmin(values))

    @staticmethod
    def _agg_max(values: NDArray[np.float64], areas: NDArray[np.float64] | None = None) -> float:
        """Compute maximum value."""
        return float(np.nanmax(values))

    @staticmethod
    def _agg_median(values: NDArray[np.float64], areas: NDArray[np.float64] | None = None) -> float:
        """Compute median value."""
        return float(np.nanmedian(values))

    @property
    def available_methods(self) -> list[str]:
        """Return list of available aggregation method names."""
        return list(self._method_funcs.keys())

    def __repr__(self) -> str:
        has_areas = self.element_areas is not None
        n_elements = len(self.element_areas) if self.element_areas is not None else 0
        return f"DataAggregator(n_elements={n_elements}, has_areas={has_areas})"


def create_aggregator_from_grid(grid: AppGrid) -> DataAggregator:
    """
    Create a DataAggregator with element areas from an AppGrid.

    Parameters
    ----------
    grid : AppGrid
        The model grid with computed element areas.

    Returns
    -------
    DataAggregator
        Configured aggregator ready for area-weighted calculations.

    Examples
    --------
    >>> from pyiwfm.core.mesh import AppGrid
    >>> grid = ...  # Load from model
    >>> aggregator = create_aggregator_from_grid(grid)
    >>> zone_values = aggregator.aggregate(heads, zone_def, "area_weighted_mean")
    """

    # Build element areas array
    max_elem_id = max(grid.elements.keys()) if grid.elements else 0
    areas = np.zeros(max_elem_id, dtype=np.float64)

    for elem_id, elem in grid.elements.items():
        areas[elem_id - 1] = elem.area

    return DataAggregator(element_areas=areas)
