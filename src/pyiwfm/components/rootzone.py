"""
Root zone component classes for IWFM models.

This module provides classes for representing the root zone including
land use types, crop parameters, soil properties, and water budget
calculations. It mirrors IWFM's Package_RootZone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from numpy.typing import NDArray

from pyiwfm.core.exceptions import ComponentError


class LandUseType(Enum):
    """Land use type enumeration."""

    AGRICULTURAL = "agricultural"
    URBAN = "urban"
    NATIVE_RIPARIAN = "native_riparian"
    WATER = "water"


@dataclass
class CropType:
    """
    Crop type definition with water requirements.

    Attributes:
        id: Unique crop identifier
        name: Crop name
        root_depth: Root zone depth (length units)
        kc: Crop coefficient for ET calculation
        monthly_kc: Optional monthly crop coefficients (12 values)
    """

    id: int
    name: str = ""
    root_depth: float = 0.0
    kc: float = 1.0
    monthly_kc: NDArray[np.float64] | None = None

    def get_kc(self, month: int = 0) -> float:
        """
        Get crop coefficient for a month.

        Args:
            month: Month (1-12), or 0 for annual kc

        Returns:
            Crop coefficient
        """
        if month == 0 or self.monthly_kc is None:
            return self.kc
        return float(self.monthly_kc[month - 1])

    def __repr__(self) -> str:
        return f"CropType(id={self.id}, name='{self.name}')"


@dataclass
class SoilParameters:
    """
    Soil hydraulic parameters.

    Attributes:
        porosity: Total porosity (volume fraction)
        field_capacity: Water content at field capacity
        wilting_point: Water content at wilting point
        saturated_kv: Saturated vertical hydraulic conductivity
        lambda_param: Pore size distribution index
        kunsat_method: Unsaturated K method (1=Campbell, 2=van Genuchten-Mualem)
        k_ponded: Ponded hydraulic conductivity (-1 = same as saturated_kv)
        capillary_rise: Capillary rise depth (v4.1+)
        precip_column: Column pointer to precipitation data
        precip_factor: Precipitation scaling factor
        generic_moisture_column: Column pointer to generic moisture data
    """

    porosity: float
    field_capacity: float
    wilting_point: float
    saturated_kv: float
    lambda_param: float = 0.5
    kunsat_method: int = 2
    k_ponded: float = -1.0
    capillary_rise: float = 0.0
    precip_column: int = 1
    precip_factor: float = 1.0
    generic_moisture_column: int = 0

    @property
    def available_water(self) -> float:
        """Return available water capacity (field_capacity - wilting_point)."""
        return self.field_capacity - self.wilting_point

    @property
    def drainable_porosity(self) -> float:
        """Return drainable porosity (porosity - field_capacity)."""
        return self.porosity - self.field_capacity

    def __repr__(self) -> str:
        return f"SoilParameters(n={self.porosity:.3f}, fc={self.field_capacity:.3f})"


@dataclass
class ElementLandUse:
    """
    Land use assignment for an element.

    Attributes:
        element_id: Element ID
        land_use_type: Type of land use
        area: Area of this land use in the element
        crop_fractions: For agricultural, mapping of crop_id to area fraction
        impervious_fraction: For urban, fraction of impervious surface
    """

    element_id: int
    land_use_type: LandUseType
    area: float
    crop_fractions: dict[int, float] = field(default_factory=dict)
    impervious_fraction: float = 0.0

    def __repr__(self) -> str:
        return f"ElementLandUse(elem={self.element_id}, type={self.land_use_type.value})"


@dataclass
class RootZone:
    """
    Root zone application component.

    This class manages land use, soil parameters, and soil moisture
    for the model domain. It mirrors IWFM's Package_RootZone.

    Attributes:
        n_elements: Number of elements
        n_layers: Number of soil layers
        crop_types: Dictionary mapping crop ID to CropType
        soil_params: Dictionary mapping element ID to SoilParameters
        element_landuse: List of element land use assignments
        soil_moisture: Soil moisture array (n_elements, n_layers)
    """

    n_elements: int
    n_layers: int
    crop_types: dict[int, CropType] = field(default_factory=dict)
    soil_params: dict[int, SoilParameters] = field(default_factory=dict)
    element_landuse: list[ElementLandUse] = field(default_factory=list)
    soil_moisture: NDArray[np.float64] | None = field(default=None, repr=False)
    # Parsed sub-file configs (v4.x format)
    nonponded_config: Any = field(default=None, repr=False)
    ponded_config: Any = field(default=None, repr=False)
    urban_config: Any = field(default=None, repr=False)
    native_riparian_config: Any = field(default=None, repr=False)
    # Surface flow destinations per element
    # v4.0-v4.11: single destination per element -> (dest_type, dest_id)
    surface_flow_destinations: dict[int, tuple[int, int]] = field(
        default_factory=dict, repr=False
    )
    # Area data file paths (for lazy loading)
    nonponded_area_file: Path | None = field(default=None, repr=False)
    ponded_area_file: Path | None = field(default=None, repr=False)
    urban_area_file: Path | None = field(default=None, repr=False)
    native_area_file: Path | None = field(default=None, repr=False)
    # v4.12+: per-landuse destinations
    surface_flow_dest_ag: dict[int, tuple[int, int]] = field(
        default_factory=dict, repr=False
    )
    surface_flow_dest_urban_in: dict[int, tuple[int, int]] = field(
        default_factory=dict, repr=False
    )
    surface_flow_dest_urban_out: dict[int, tuple[int, int]] = field(
        default_factory=dict, repr=False
    )
    surface_flow_dest_nvrv: dict[int, tuple[int, int]] = field(
        default_factory=dict, repr=False
    )

    @property
    def n_crop_types(self) -> int:
        """Return number of crop types."""
        return len(self.crop_types)

    def add_crop_type(self, crop: CropType) -> None:
        """Add a crop type definition."""
        self.crop_types[crop.id] = crop

    def get_crop_type(self, crop_id: int) -> CropType:
        """Get a crop type by ID."""
        return self.crop_types[crop_id]

    def set_soil_parameters(self, element_id: int, params: SoilParameters) -> None:
        """Set soil parameters for an element."""
        self.soil_params[element_id] = params

    def get_soil_parameters(self, element_id: int) -> SoilParameters:
        """Get soil parameters for an element."""
        return self.soil_params[element_id]

    def add_element_landuse(self, elu: ElementLandUse) -> None:
        """Add an element land use assignment."""
        self.element_landuse.append(elu)

    def get_landuse_for_element(self, element_id: int) -> list[ElementLandUse]:
        """Get all land use assignments for an element."""
        return [e for e in self.element_landuse if e.element_id == element_id]

    def get_total_area(self, land_use_type: LandUseType) -> float:
        """Calculate total area for a land use type."""
        return sum(
            e.area for e in self.element_landuse if e.land_use_type == land_use_type
        )

    def set_soil_moisture(self, moisture: NDArray[np.float64]) -> None:
        """Set the soil moisture array."""
        if moisture.shape != (self.n_elements, self.n_layers):
            raise ValueError(
                f"Moisture shape {moisture.shape} doesn't match "
                f"expected ({self.n_elements}, {self.n_layers})"
            )
        self.soil_moisture = moisture.copy()

    def get_soil_moisture(self, element: int, layer: int) -> float:
        """Get soil moisture at a specific element and layer."""
        if self.soil_moisture is None:
            raise ValueError("Soil moisture not set")
        return float(self.soil_moisture[element, layer])

    def iter_elements_with_landuse(self) -> Iterator[int]:
        """Iterate over element IDs that have land use assignments."""
        seen: set[int] = set()
        for elu in self.element_landuse:
            if elu.element_id not in seen:
                seen.add(elu.element_id)
                yield elu.element_id

    def load_land_use_snapshot(self, timestep: int = 0) -> None:
        """Read land use areas for a single timestep from area data files.

        Populates ``element_landuse`` from the IWFM area time-series files
        referenced by ``nonponded_area_file``, ``ponded_area_file``,
        ``urban_area_file``, and ``native_area_file``.

        Args:
            timestep: Zero-based timestep index to read.
        """
        from pyiwfm.io.rootzone_area import read_area_timestep

        self.element_landuse.clear()

        n_nonponded = 0
        if self.nonponded_config is not None:
            n_nonponded = getattr(self.nonponded_config, "n_crops", 0)

        # Non-ponded agricultural areas
        if self.nonponded_area_file and self.nonponded_area_file.exists():
            data = read_area_timestep(self.nonponded_area_file, timestep)
            for elem_id, areas in data.items():
                total = sum(areas)
                fracs: dict[int, float] = {}
                for i, a in enumerate(areas):
                    crop_id = i + 1
                    if total > 0:
                        fracs[crop_id] = a / total
                self.element_landuse.append(
                    ElementLandUse(
                        element_id=elem_id,
                        land_use_type=LandUseType.AGRICULTURAL,
                        area=total,
                        crop_fractions=fracs,
                    )
                )

        # Ponded agricultural areas
        if self.ponded_area_file and self.ponded_area_file.exists():
            data = read_area_timestep(self.ponded_area_file, timestep)
            for elem_id, areas in data.items():
                total = sum(areas)
                fracs = {}
                for i, a in enumerate(areas):
                    crop_id = n_nonponded + i + 1
                    if total > 0:
                        fracs[crop_id] = a / total
                self.element_landuse.append(
                    ElementLandUse(
                        element_id=elem_id,
                        land_use_type=LandUseType.AGRICULTURAL,
                        area=total,
                        crop_fractions=fracs,
                    )
                )

        # Urban areas
        if self.urban_area_file and self.urban_area_file.exists():
            data = read_area_timestep(self.urban_area_file, timestep)
            for elem_id, areas in data.items():
                total = sum(areas)
                imp_frac = 0.0
                if self.urban_config is not None:
                    for row in getattr(self.urban_config, "element_data", []):
                        if row.element_id == elem_id:
                            imp_frac = 1.0 - getattr(
                                row, "pervious_fraction", 1.0
                            )
                            break
                self.element_landuse.append(
                    ElementLandUse(
                        element_id=elem_id,
                        land_use_type=LandUseType.URBAN,
                        area=total,
                        impervious_fraction=imp_frac,
                    )
                )

        # Native/riparian areas
        if self.native_area_file and self.native_area_file.exists():
            data = read_area_timestep(self.native_area_file, timestep)
            for elem_id, areas in data.items():
                total = sum(areas)
                self.element_landuse.append(
                    ElementLandUse(
                        element_id=elem_id,
                        land_use_type=LandUseType.NATIVE_RIPARIAN,
                        area=total,
                    )
                )

    def validate(self) -> None:
        """
        Validate the root zone component.

        Raises:
            ComponentError: If component is invalid
        """
        # Check element references (1-based IDs)
        for elu in self.element_landuse:
            if elu.element_id > self.n_elements or elu.element_id < 1:
                raise ComponentError(
                    f"Land use references invalid element {elu.element_id}"
                )

        # Check crop references in agricultural land use
        for elu in self.element_landuse:
            if elu.land_use_type == LandUseType.AGRICULTURAL:
                for crop_id in elu.crop_fractions:
                    if crop_id not in self.crop_types:
                        raise ComponentError(
                            f"Element {elu.element_id} references "
                            f"undefined crop type {crop_id}"
                        )

    def to_arrays(self) -> dict[str, NDArray]:
        """
        Convert root zone data to numpy arrays.

        Returns:
            Dictionary of arrays
        """
        result: dict[str, NDArray] = {}

        if self.soil_moisture is not None:
            result["soil_moisture"] = self.soil_moisture.copy()

        return result

    @classmethod
    def from_arrays(
        cls,
        n_elements: int,
        n_layers: int,
        soil_moisture: NDArray[np.float64] | None = None,
    ) -> "RootZone":
        """
        Create root zone component from arrays.

        Args:
            n_elements: Number of elements
            n_layers: Number of soil layers
            soil_moisture: Soil moisture array (optional)

        Returns:
            RootZone instance
        """
        rz = cls(n_elements=n_elements, n_layers=n_layers)

        if soil_moisture is not None:
            rz.set_soil_moisture(soil_moisture)

        return rz

    def __repr__(self) -> str:
        return f"RootZone(n_elements={self.n_elements}, n_crops={self.n_crop_types})"
