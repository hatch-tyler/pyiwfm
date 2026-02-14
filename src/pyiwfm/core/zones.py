"""
Zone definitions for multi-scale data viewing.

This module provides zone data structures for spatial aggregation:

- :class:`Zone`: A named group of elements with computed area
- :class:`ZoneDefinition`: Collection of zones with element-to-zone mapping

Zone definitions support multiple spatial scales:
- Subregions (predefined groupings from model input files)
- User-specified zones (custom groupings for ZBudget analysis)

Example
-------
Create a simple zone definition:

>>> from pyiwfm.core.zones import Zone, ZoneDefinition
>>> import numpy as np
>>> zones = {
...     1: Zone(id=1, name="North Basin", elements=[1, 2, 3], area=1500.0),
...     2: Zone(id=2, name="South Basin", elements=[4, 5, 6], area=1200.0),
... }
>>> element_zones = np.array([1, 1, 1, 2, 2, 2])  # element_id-1 -> zone_id
>>> zone_def = ZoneDefinition(zones=zones, extent="horizontal", element_zones=element_zones)
>>> zone_def.get_zone_for_element(3)
1
>>> zone_def.get_elements_in_zone(2)
[4, 5, 6]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
from numpy.typing import NDArray


@dataclass
class Zone:
    """
    A named zone containing a group of elements.

    Parameters
    ----------
    id : int
        Unique zone identifier (1-based).
    name : str
        Descriptive name for the zone (e.g., "Sacramento Valley").
    elements : list of int
        List of element IDs belonging to this zone.
    area : float, optional
        Total zone area in model units squared. Default is 0.0,
        can be computed from element areas.

    Examples
    --------
    Create a zone:

    >>> zone = Zone(id=1, name="Central Basin", elements=[10, 11, 12, 13], area=5000.0)
    >>> print(f"Zone {zone.id}: {zone.name} ({len(zone.elements)} elements)")
    Zone 1: Central Basin (4 elements)

    Check if element is in zone:

    >>> 11 in zone.elements
    True
    """

    id: int
    name: str
    elements: list[int] = field(default_factory=list)
    area: float = 0.0

    @property
    def n_elements(self) -> int:
        """Return the number of elements in this zone."""
        return len(self.elements)

    def __repr__(self) -> str:
        return f"Zone(id={self.id}, name={self.name!r}, n_elements={self.n_elements}, area={self.area:.1f})"


@dataclass
class ZoneDefinition:
    """
    A complete zone definition with element-to-zone mapping.

    Parameters
    ----------
    zones : dict[int, Zone]
        Dictionary mapping zone ID to Zone object.
    extent : str, optional
        Zone extent type: "horizontal" (default) or "vertical".
        Horizontal zones span all layers; vertical zones are layer-specific.
    element_zones : NDArray, optional
        Array mapping element index (0-based) to zone ID.
        Shape: (n_elements,). Elements with no zone have value 0.
    name : str, optional
        Name for this zone definition set.
    description : str, optional
        Description of the zone definition.

    Examples
    --------
    Create a zone definition from scratch:

    >>> zones = {
    ...     1: Zone(id=1, name="Zone A", elements=[1, 2, 3]),
    ...     2: Zone(id=2, name="Zone B", elements=[4, 5]),
    ... }
    >>> elem_zones = np.array([1, 1, 1, 2, 2])
    >>> zone_def = ZoneDefinition(zones=zones, element_zones=elem_zones)
    >>> zone_def.n_zones
    2

    Create from subregions:

    >>> from pyiwfm.core.mesh import AppGrid, Subregion
    >>> grid = ...  # Load from model
    >>> zone_def = ZoneDefinition.from_subregions(grid)
    """

    zones: dict[int, Zone] = field(default_factory=dict)
    extent: str = "horizontal"
    element_zones: NDArray[np.int32] | None = None
    name: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        """Validate and normalize zone definition."""
        if self.extent not in ("horizontal", "vertical"):
            raise ValueError(f"extent must be 'horizontal' or 'vertical', got {self.extent!r}")

        # Normalize extent to lowercase
        self.extent = self.extent.lower()

    @property
    def n_zones(self) -> int:
        """Return the total number of zones."""
        return len(self.zones)

    @property
    def n_elements(self) -> int:
        """Return the total number of elements with zone assignments."""
        if self.element_zones is None:
            return 0
        return int(np.count_nonzero(self.element_zones))

    @property
    def zone_ids(self) -> list[int]:
        """Return sorted list of zone IDs."""
        return sorted(self.zones.keys())

    def get_zone_for_element(self, element_id: int) -> int:
        """
        Get the zone ID for a given element.

        Parameters
        ----------
        element_id : int
            Element ID (1-based).

        Returns
        -------
        int
            Zone ID, or 0 if element has no zone assignment.
        """
        if self.element_zones is None:
            return 0
        idx = element_id - 1  # Convert to 0-based index
        if idx < 0 or idx >= len(self.element_zones):
            return 0
        return int(self.element_zones[idx])

    def get_elements_in_zone(self, zone_id: int) -> list[int]:
        """
        Get all element IDs in a zone.

        Parameters
        ----------
        zone_id : int
            Zone ID.

        Returns
        -------
        list of int
            Element IDs (1-based) in the zone.
        """
        if zone_id in self.zones:
            return list(self.zones[zone_id].elements)

        # Fallback to element_zones array if zone not in dict
        if self.element_zones is not None:
            indices = np.where(self.element_zones == zone_id)[0]
            return [int(i + 1) for i in indices]  # Convert to 1-based

        return []

    def get_zone(self, zone_id: int) -> Zone | None:
        """
        Get a Zone by ID.

        Parameters
        ----------
        zone_id : int
            Zone ID.

        Returns
        -------
        Zone or None
            The Zone object, or None if not found.
        """
        return self.zones.get(zone_id)

    def iter_zones(self) -> Iterator[Zone]:
        """Iterate over all zones in ID order."""
        for zone_id in sorted(self.zones.keys()):
            yield self.zones[zone_id]

    @classmethod
    def from_subregions(cls, grid: "AppGrid") -> "ZoneDefinition":
        """
        Create a ZoneDefinition from model subregions.

        Parameters
        ----------
        grid : AppGrid
            The model grid with subregion assignments.

        Returns
        -------
        ZoneDefinition
            Zone definition with one zone per subregion.
        """
        from pyiwfm.core.mesh import AppGrid

        zones: dict[int, Zone] = {}

        # Build element-to-zone mapping
        max_elem_id = max(grid.elements.keys()) if grid.elements else 0
        element_zones = np.zeros(max_elem_id, dtype=np.int32)

        # Group elements by subregion
        subregion_elements: dict[int, list[int]] = {}
        subregion_areas: dict[int, float] = {}

        for elem_id, elem in grid.elements.items():
            sr_id = elem.subregion
            if sr_id > 0:
                if sr_id not in subregion_elements:
                    subregion_elements[sr_id] = []
                    subregion_areas[sr_id] = 0.0
                subregion_elements[sr_id].append(elem_id)
                subregion_areas[sr_id] += elem.area
                element_zones[elem_id - 1] = sr_id

        # Create Zone objects
        for sr_id, elem_list in subregion_elements.items():
            sr = grid.subregions.get(sr_id)
            name = sr.name if sr else f"Subregion {sr_id}"
            zones[sr_id] = Zone(
                id=sr_id,
                name=name,
                elements=elem_list,
                area=subregion_areas[sr_id],
            )

        return cls(
            zones=zones,
            extent="horizontal",
            element_zones=element_zones,
            name="Subregions",
            description="Zone definition from model subregions",
        )

    @classmethod
    def from_element_list(
        cls,
        element_zone_pairs: list[tuple[int, int]],
        zone_names: dict[int, str] | None = None,
        element_areas: dict[int, float] | None = None,
        name: str = "",
        description: str = "",
    ) -> "ZoneDefinition":
        """
        Create a ZoneDefinition from a list of (element_id, zone_id) pairs.

        Parameters
        ----------
        element_zone_pairs : list of (int, int)
            List of (element_id, zone_id) pairs.
        zone_names : dict[int, str], optional
            Mapping of zone ID to zone name.
        element_areas : dict[int, float], optional
            Mapping of element ID to element area.
        name : str, optional
            Name for this zone definition.
        description : str, optional
            Description of the zone definition.

        Returns
        -------
        ZoneDefinition
            The constructed zone definition.
        """
        zone_names = zone_names or {}
        element_areas = element_areas or {}

        # Find max element ID
        max_elem_id = max(e for e, z in element_zone_pairs) if element_zone_pairs else 0
        element_zones = np.zeros(max_elem_id, dtype=np.int32)

        # Group elements by zone
        zone_elements: dict[int, list[int]] = {}
        zone_areas: dict[int, float] = {}

        for elem_id, zone_id in element_zone_pairs:
            if zone_id > 0:
                if zone_id not in zone_elements:
                    zone_elements[zone_id] = []
                    zone_areas[zone_id] = 0.0
                zone_elements[zone_id].append(elem_id)
                zone_areas[zone_id] += element_areas.get(elem_id, 0.0)
                element_zones[elem_id - 1] = zone_id

        # Create Zone objects
        zones: dict[int, Zone] = {}
        for zone_id, elem_list in zone_elements.items():
            zones[zone_id] = Zone(
                id=zone_id,
                name=zone_names.get(zone_id, f"Zone {zone_id}"),
                elements=elem_list,
                area=zone_areas[zone_id],
            )

        return cls(
            zones=zones,
            extent="horizontal",
            element_zones=element_zones,
            name=name,
            description=description,
        )

    def add_zone(self, zone: Zone) -> None:
        """
        Add a new zone to the definition.

        Parameters
        ----------
        zone : Zone
            The zone to add.
        """
        self.zones[zone.id] = zone

        # Update element_zones array
        if self.element_zones is not None and zone.elements:
            max_elem = max(zone.elements)
            if max_elem > len(self.element_zones):
                # Extend array
                new_arr = np.zeros(max_elem, dtype=np.int32)
                new_arr[: len(self.element_zones)] = self.element_zones
                self.element_zones = new_arr

            for elem_id in zone.elements:
                self.element_zones[elem_id - 1] = zone.id

    def remove_zone(self, zone_id: int) -> Zone | None:
        """
        Remove a zone from the definition.

        Parameters
        ----------
        zone_id : int
            ID of the zone to remove.

        Returns
        -------
        Zone or None
            The removed zone, or None if not found.
        """
        zone = self.zones.pop(zone_id, None)
        if zone and self.element_zones is not None:
            # Clear zone assignments
            self.element_zones[self.element_zones == zone_id] = 0
        return zone

    def compute_areas(self, grid: "AppGrid") -> None:
        """
        Recompute zone areas from element areas.

        Parameters
        ----------
        grid : AppGrid
            The model grid with computed element areas.
        """
        for zone in self.zones.values():
            zone.area = sum(
                grid.elements[e].area for e in zone.elements if e in grid.elements
            )

    def validate(self, n_elements: int) -> list[str]:
        """
        Validate the zone definition.

        Parameters
        ----------
        n_elements : int
            Total number of elements in the model.

        Returns
        -------
        list of str
            List of validation error messages (empty if valid).
        """
        errors: list[str] = []

        # Check element_zones array size
        if self.element_zones is not None:
            if len(self.element_zones) != n_elements:
                errors.append(
                    f"element_zones array size ({len(self.element_zones)}) does not match "
                    f"number of elements ({n_elements})"
                )

        # Check for overlapping zones (element in multiple zones)
        all_elements: set[int] = set()
        for zone in self.zones.values():
            overlap = all_elements & set(zone.elements)
            if overlap:
                errors.append(f"Zone {zone.id} ({zone.name}) has overlapping elements: {overlap}")
            all_elements.update(zone.elements)

        # Check that zone_id matches dict key
        for zone_id, zone in self.zones.items():
            if zone.id != zone_id:
                errors.append(f"Zone dict key {zone_id} does not match zone.id {zone.id}")

        return errors

    def __repr__(self) -> str:
        return (
            f"ZoneDefinition(n_zones={self.n_zones}, extent={self.extent!r}, "
            f"n_elements={self.n_elements}, name={self.name!r})"
        )
