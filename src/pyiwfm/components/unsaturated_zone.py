"""
Unsaturated Zone component classes for IWFM models.

This module provides classes for representing the unsaturated (vadose) zone,
including per-element layer properties and initial moisture conditions.
It mirrors IWFM's Package_AppUnsatZone.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from pyiwfm.core.exceptions import ComponentError

if TYPE_CHECKING:
    from pyiwfm.io.unsaturated_zone import UnsatZoneMainConfig


@dataclass
class UnsatZoneLayer:
    """Per-layer unsaturated zone properties for a single element.

    Attributes:
        thickness_max: Maximum layer thickness
        total_porosity: Total porosity
        lambda_param: Pore size distribution parameter
        hyd_cond: Saturated hydraulic conductivity
        kunsat_method: Unsaturated K method code
    """

    thickness_max: float = 0.0
    total_porosity: float = 0.0
    lambda_param: float = 0.0
    hyd_cond: float = 0.0
    kunsat_method: int = 0


@dataclass
class UnsatZoneElement:
    """Per-element unsaturated zone data.

    Attributes:
        element_id: 1-based element ID
        layers: List of per-layer properties
        initial_moisture: Initial soil moisture per layer (optional)
    """

    element_id: int = 0
    layers: list[UnsatZoneLayer] = field(default_factory=list)
    initial_moisture: NDArray[np.float64] | None = None

    @property
    def n_layers(self) -> int:
        """Return number of unsaturated zone layers."""
        return len(self.layers)

    def __repr__(self) -> str:
        return f"UnsatZoneElement(id={self.element_id}, n_layers={self.n_layers})"


@dataclass
class AppUnsatZone:
    """Unsaturated Zone application component.

    This class manages the vadose zone modeling parameters for all elements.
    It mirrors IWFM's Package_AppUnsatZone.

    Attributes:
        n_layers: Number of unsaturated zone layers
        elements: Dictionary mapping element ID to UnsatZoneElement
        solver_tolerance: Solver convergence tolerance
        max_iterations: Maximum solver iterations
        coord_factor: Conversion factor for x-y coordinates
        thickness_factor: Conversion factor for layer thickness
        hyd_cond_factor: Conversion factor for hydraulic conductivity
        time_unit: Time unit for hydraulic conductivity
        n_parametric_grids: Number of parametric grids (0=direct input)
        budget_file: Path to HDF5 budget output file
        zbudget_file: Path to HDF5 zone budget output file
        final_results_file: Path to final simulation results file
    """

    n_layers: int = 0
    elements: dict[int, UnsatZoneElement] = field(default_factory=dict)

    # Solver parameters
    solver_tolerance: float = 1e-8
    max_iterations: int = 2000

    # Conversion factors
    coord_factor: float = 1.0
    thickness_factor: float = 1.0
    hyd_cond_factor: float = 1.0
    time_unit: str = ""
    n_parametric_grids: int = 0

    # Output files
    budget_file: str = ""
    zbudget_file: str = ""
    final_results_file: str = ""

    @property
    def n_elements(self) -> int:
        """Return number of elements with unsaturated zone data."""
        return len(self.elements)

    def add_element(self, elem: UnsatZoneElement) -> None:
        """Add an element to the component."""
        self.elements[elem.element_id] = elem

    def get_element(self, element_id: int) -> UnsatZoneElement:
        """Get an element by ID."""
        return self.elements[element_id]

    def iter_elements(self) -> Iterator[UnsatZoneElement]:
        """Iterate over elements in ID order."""
        for eid in sorted(self.elements.keys()):
            yield self.elements[eid]

    def validate(self) -> None:
        """Validate the unsaturated zone component.

        Raises:
            ComponentError: If component is invalid
        """
        if self.n_layers <= 0:
            raise ComponentError(
                f"Unsaturated zone has non-positive layer count: {self.n_layers}"
            )

        for elem in self.elements.values():
            if elem.n_layers != self.n_layers:
                raise ComponentError(
                    f"Element {elem.element_id} has {elem.n_layers} layers "
                    f"but component expects {self.n_layers}"
                )

    @classmethod
    def from_config(cls, config: UnsatZoneMainConfig) -> AppUnsatZone:
        """Create component from a parsed UnsatZoneMainConfig.

        Args:
            config: Parsed configuration from the reader

        Returns:
            AppUnsatZone instance
        """
        comp = cls(
            n_layers=config.n_layers,
            solver_tolerance=config.solver_tolerance,
            max_iterations=config.max_iterations,
            coord_factor=config.coord_factor,
            thickness_factor=config.thickness_factor,
            hyd_cond_factor=config.hyd_cond_factor,
            time_unit=config.time_unit,
            n_parametric_grids=config.n_parametric_grids,
            budget_file=str(config.budget_file) if config.budget_file else "",
            zbudget_file=str(config.zbudget_file) if config.zbudget_file else "",
            final_results_file=(
                str(config.final_results_file)
                if config.final_results_file
                else ""
            ),
        )

        # Convert element data
        for ed in config.element_data:
            layers = []
            for i in range(config.n_layers):
                layers.append(UnsatZoneLayer(
                    thickness_max=float(ed.thickness_max[i]),
                    total_porosity=float(ed.total_porosity[i]),
                    lambda_param=float(ed.lambda_param[i]),
                    hyd_cond=float(ed.hyd_cond[i]),
                    kunsat_method=int(ed.kunsat_method[i]),
                ))

            moisture = config.initial_soil_moisture.get(ed.element_id)
            if moisture is None:
                # Check for uniform value (key 0)
                moisture = config.initial_soil_moisture.get(0)

            elem = UnsatZoneElement(
                element_id=ed.element_id,
                layers=layers,
                initial_moisture=moisture,
            )
            comp.add_element(elem)

        return comp

    def __repr__(self) -> str:
        return (
            f"AppUnsatZone(n_layers={self.n_layers}, "
            f"n_elements={self.n_elements})"
        )
