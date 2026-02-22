"""
Lake component classes for IWFM models.

This module provides classes for representing lakes, including
lake elements, outflows, rating curves, and the main lake application
class. It mirrors IWFM's Package_AppLake.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from pyiwfm.core.base_component import BaseComponent
from pyiwfm.core.exceptions import ComponentError


@dataclass
class LakeRating:
    """
    Elevation-area-volume rating curve for a lake.

    Attributes:
        elevations: Array of water surface elevations
        areas: Array of corresponding surface areas
        volumes: Array of corresponding storage volumes
    """

    elevations: NDArray[np.float64]
    areas: NDArray[np.float64]
    volumes: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate rating curve data."""
        if not (len(self.elevations) == len(self.areas) == len(self.volumes)):
            raise ValueError("elevations, areas, and volumes must have same length")
        if len(self.elevations) < 2:
            raise ValueError("Rating curve must have at least 2 points")

    def get_area(self, elevation: float) -> float:
        """
        Interpolate surface area from elevation.

        Args:
            elevation: Water surface elevation

        Returns:
            Interpolated surface area
        """
        if elevation <= self.elevations[0]:
            return 0.0

        if elevation >= self.elevations[-1]:
            # Linear extrapolation above max
            slope = (self.areas[-1] - self.areas[-2]) / (self.elevations[-1] - self.elevations[-2])
            return float(self.areas[-1] + slope * (elevation - self.elevations[-1]))

        return float(np.interp(elevation, self.elevations, self.areas))

    def get_volume(self, elevation: float) -> float:
        """
        Interpolate storage volume from elevation.

        Args:
            elevation: Water surface elevation

        Returns:
            Interpolated storage volume
        """
        if elevation <= self.elevations[0]:
            return 0.0

        if elevation >= self.elevations[-1]:
            # Linear extrapolation above max
            slope = (self.volumes[-1] - self.volumes[-2]) / (
                self.elevations[-1] - self.elevations[-2]
            )
            return float(self.volumes[-1] + slope * (elevation - self.elevations[-1]))

        return float(np.interp(elevation, self.elevations, self.volumes))

    def get_elevation(self, volume: float) -> float:
        """
        Interpolate elevation from storage volume.

        Args:
            volume: Storage volume

        Returns:
            Interpolated water surface elevation
        """
        if volume <= self.volumes[0]:
            return float(self.elevations[0])

        if volume >= self.volumes[-1]:
            # Linear extrapolation above max
            slope = (self.elevations[-1] - self.elevations[-2]) / (
                self.volumes[-1] - self.volumes[-2]
            )
            return float(self.elevations[-1] + slope * (volume - self.volumes[-1]))

        return float(np.interp(volume, self.volumes, self.elevations))


@dataclass
class LakeElement:
    """
    An element that is part of a lake.

    Attributes:
        element_id: Element ID
        lake_id: ID of the lake this element belongs to
        fraction: Fraction of element covered by lake (0-1)
    """

    element_id: int
    lake_id: int
    fraction: float = 1.0

    def __repr__(self) -> str:
        return f"LakeElement(elem={self.element_id}, lake={self.lake_id})"


@dataclass
class LakeOutflow:
    """
    Lake outflow configuration.

    Attributes:
        lake_id: ID of the source lake
        destination_type: Type of destination ('stream', 'lake', 'outside')
        destination_id: ID of destination (stream node or lake)
        max_rate: Maximum outflow rate
    """

    lake_id: int
    destination_type: str
    destination_id: int
    max_rate: float = float("inf")

    def __repr__(self) -> str:
        return (
            f"LakeOutflow(lake={self.lake_id}, dst={self.destination_type}:{self.destination_id})"
        )


@dataclass
class Lake:
    """
    A lake in the model domain.

    Attributes:
        id: Unique lake identifier
        name: Descriptive name
        max_elevation: Maximum water surface elevation
        initial_storage: Initial storage volume
        elements: List of element IDs that make up the lake
        gw_nodes: List of groundwater node IDs for lake-aquifer interaction
        rating: Elevation-area-volume rating curve
        outflow: Outflow configuration
    """

    id: int
    name: str = ""
    max_elevation: float = float("inf")
    initial_storage: float = 0.0
    initial_elevation: float = 280.0
    elements: list[int] = field(default_factory=list)
    gw_nodes: list[int] = field(default_factory=list)
    rating: LakeRating | None = None
    outflow: LakeOutflow | None = None

    # Lake bed parameters
    bed_conductivity: float = 2.0
    bed_thickness: float = 1.0

    # Column references for TS data
    et_column: int = 7
    precip_column: int = 2
    max_elev_column: int = 0  # ICHLMAX (v4.0 only)

    # v5.0 outflow rating table
    outflow_rating_elevations: list[float] = field(default_factory=list)
    outflow_rating_flows: list[float] = field(default_factory=list)

    @property
    def n_elements(self) -> int:
        """Return number of elements in this lake."""
        return len(self.elements)

    def __repr__(self) -> str:
        return f"Lake(id={self.id}, name='{self.name}')"


@dataclass
class AppLake(BaseComponent):
    """
    Lake application component.

    This class manages all lakes in the model domain including
    lake elements, outflows, and rating curves. It mirrors IWFM's
    Package_AppLake.

    Attributes:
        lakes: Dictionary mapping lake ID to Lake
        lake_elements: List of lake elements
        current_elevations: Dictionary of current water surface elevations
    """

    lakes: dict[int, Lake] = field(default_factory=dict)
    lake_elements: list[LakeElement] = field(default_factory=list)
    current_elevations: dict[int, float] = field(default_factory=dict)

    @property
    def n_items(self) -> int:
        """Return number of lakes (primary entities)."""
        return len(self.lakes)

    @property
    def n_lakes(self) -> int:
        """Return number of lakes."""
        return len(self.lakes)

    @property
    def n_lake_elements(self) -> int:
        """Return number of lake elements."""
        return len(self.lake_elements)

    def add_lake(self, lake: Lake) -> None:
        """Add a lake to the component."""
        self.lakes[lake.id] = lake

    def add_lake_element(self, elem: LakeElement) -> None:
        """Add a lake element."""
        self.lake_elements.append(elem)

    def get_lake(self, lake_id: int) -> Lake:
        """Get a lake by ID."""
        return self.lakes[lake_id]

    def get_elements_for_lake(self, lake_id: int) -> list[LakeElement]:
        """Get all elements for a specific lake."""
        return [e for e in self.lake_elements if e.lake_id == lake_id]

    def set_elevation(self, lake_id: int, elevation: float) -> None:
        """Set the current water surface elevation for a lake."""
        self.current_elevations[lake_id] = elevation

    def get_elevation(self, lake_id: int) -> float:
        """Get the current water surface elevation for a lake."""
        return self.current_elevations.get(lake_id, 0.0)

    def get_area(self, lake_id: int) -> float:
        """Get the current surface area for a lake."""
        lake = self.lakes[lake_id]
        if lake.rating is None:
            return 0.0
        elevation = self.current_elevations.get(lake_id, 0.0)
        return lake.rating.get_area(elevation)

    def get_volume(self, lake_id: int) -> float:
        """Get the current storage volume for a lake."""
        lake = self.lakes[lake_id]
        if lake.rating is None:
            return 0.0
        elevation = self.current_elevations.get(lake_id, 0.0)
        return lake.rating.get_volume(elevation)

    def get_total_area(self) -> float:
        """Calculate total surface area of all lakes."""
        return sum(self.get_area(lid) for lid in self.lakes)

    def get_total_volume(self) -> float:
        """Calculate total storage volume of all lakes."""
        return sum(self.get_volume(lid) for lid in self.lakes)

    def iter_lakes(self) -> Iterator[Lake]:
        """Iterate over lakes in ID order."""
        for lid in sorted(self.lakes.keys()):
            yield self.lakes[lid]

    def validate(self) -> None:
        """
        Validate the lake component.

        Raises:
            ComponentError: If component is invalid
        """
        # Check lake element references
        for elem in self.lake_elements:
            if elem.lake_id not in self.lakes:
                raise ComponentError(
                    f"Lake element {elem.element_id} references non-existent lake {elem.lake_id}"
                )

        # Check outflow references
        for lake in self.lakes.values():
            if lake.outflow is not None:
                if lake.outflow.destination_type == "lake":
                    if lake.outflow.destination_id not in self.lakes:
                        raise ComponentError(
                            f"Lake {lake.id} outflow references "
                            f"non-existent lake {lake.outflow.destination_id}"
                        )

    def to_arrays(self) -> dict[str, NDArray]:
        """
        Convert lake data to numpy arrays.

        Returns:
            Dictionary of arrays
        """
        sorted_ids = sorted(self.lakes.keys())

        lake_ids = np.array(sorted_ids, dtype=np.int32)
        elevations = np.array([self.current_elevations.get(lid, 0.0) for lid in sorted_ids])

        return {
            "lake_ids": lake_ids,
            "elevations": elevations,
        }

    @classmethod
    def from_arrays(
        cls,
        lake_ids: NDArray[np.int32],
        names: list[str],
        max_elevations: NDArray[np.float64] | None = None,
    ) -> AppLake:
        """
        Create lake component from arrays.

        Args:
            lake_ids: Array of lake IDs
            names: List of lake names
            max_elevations: Array of maximum elevations (optional)

        Returns:
            AppLake instance
        """
        lakes = cls()

        for i, lid in enumerate(lake_ids):
            max_elev = float(max_elevations[i]) if max_elevations is not None else float("inf")
            lake = Lake(
                id=int(lid),
                name=names[i] if i < len(names) else "",
                max_elevation=max_elev,
            )
            lakes.add_lake(lake)

        return lakes

    def __repr__(self) -> str:
        return f"AppLake(n_lakes={self.n_lakes})"
