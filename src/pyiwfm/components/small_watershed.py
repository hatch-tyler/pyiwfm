"""
Small Watershed component classes for IWFM models.

This module provides classes for representing small watersheds, including
watershed units with root zone and aquifer parameters, and the main
application class. It mirrors IWFM's Package_AppSmallWatershed.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pyiwfm.core.exceptions import ComponentError

if TYPE_CHECKING:
    from pyiwfm.io.small_watershed import SmallWatershedMainConfig


@dataclass
class WatershedGWNode:
    """Groundwater node connection for a small watershed.

    Attributes:
        gw_node_id: Groundwater node ID (1-based)
        max_perc_rate: Maximum percolation rate (positive) or baseflow layer (negative)
        is_baseflow: Whether this is a baseflow node
        layer: Baseflow layer (if is_baseflow)
    """

    gw_node_id: int = 0
    max_perc_rate: float = 0.0
    is_baseflow: bool = False
    layer: int = 0


@dataclass
class WatershedUnit:
    """A single small watershed unit.

    Attributes:
        id: Watershed unit ID (1-based)
        area: Watershed area
        dest_stream_node: Destination stream node for outflow (1-based)
        gw_nodes: Connected groundwater nodes

        precip_col: Precipitation time-series column index
        precip_factor: Precipitation conversion factor
        et_col: ET time-series column index
        wilting_point: Soil wilting point
        field_capacity: Soil field capacity
        total_porosity: Total porosity
        lambda_param: Pore size distribution parameter
        root_depth: Root zone depth
        hydraulic_cond: Hydraulic conductivity
        kunsat_method: Unsaturated K method code
        curve_number: SCS curve number

        gw_threshold: Groundwater storage threshold
        max_gw_storage: Maximum groundwater storage
        surface_flow_coeff: Surface flow recession coefficient
        baseflow_coeff: Baseflow recession coefficient
    """

    id: int = 0
    area: float = 0.0
    dest_stream_node: int = 0
    gw_nodes: list[WatershedGWNode] = field(default_factory=list)

    # Root zone parameters
    precip_col: int = 0
    precip_factor: float = 1.0
    et_col: int = 0
    wilting_point: float = 0.0
    field_capacity: float = 0.0
    total_porosity: float = 0.0
    lambda_param: float = 0.0
    root_depth: float = 0.0
    hydraulic_cond: float = 0.0
    kunsat_method: int = 0
    curve_number: float = 0.0

    # Aquifer parameters
    gw_threshold: float = 0.0
    max_gw_storage: float = 0.0
    surface_flow_coeff: float = 0.0
    baseflow_coeff: float = 0.0

    # Initial conditions
    initial_soil_moisture: float = 0.0
    initial_gw_storage: float = 0.0

    @property
    def n_gw_nodes(self) -> int:
        """Return number of connected groundwater nodes."""
        return len(self.gw_nodes)

    def __repr__(self) -> str:
        return f"WatershedUnit(id={self.id}, area={self.area:.1f})"


@dataclass
class AppSmallWatershed:
    """Small Watershed application component.

    This class manages all small watersheds in the model domain including
    watershed units with root zone and aquifer parameters. It mirrors
    IWFM's Package_AppSmallWatershed.

    Attributes:
        watersheds: Dictionary mapping watershed ID to WatershedUnit
        area_factor: Area conversion factor
        flow_factor: Flow rate conversion factor
        flow_time_unit: Time unit for flow rates
        rz_solver_tolerance: Root zone solver tolerance
        rz_max_iterations: Root zone solver max iterations
        rz_length_factor: Root zone length conversion factor
        rz_cn_factor: Curve number conversion factor
        rz_k_factor: Hydraulic conductivity conversion factor
        rz_k_time_unit: Time unit for hydraulic conductivity
        aq_gw_factor: GW conversion factor
        aq_time_factor: Time conversion factor
        aq_time_unit: Time unit for recession coefficients
        budget_output_file: Path to budget output file
        final_results_file: Path to final simulation results file
    """

    watersheds: dict[int, WatershedUnit] = field(default_factory=dict)

    # Geospatial conversion factors
    area_factor: float = 1.0
    flow_factor: float = 1.0
    flow_time_unit: str = ""

    # Root zone solver parameters
    rz_solver_tolerance: float = 1e-8
    rz_max_iterations: int = 2000
    rz_length_factor: float = 1.0
    rz_cn_factor: float = 1.0
    rz_k_factor: float = 1.0
    rz_k_time_unit: str = ""

    # Aquifer conversion factors
    aq_gw_factor: float = 1.0
    aq_time_factor: float = 1.0
    aq_time_unit: str = ""

    # Initial conditions conversion factor
    ic_factor: float = 1.0

    # Output files
    budget_output_file: str = ""
    final_results_file: str = ""

    @property
    def n_watersheds(self) -> int:
        """Return number of watersheds."""
        return len(self.watersheds)

    def add_watershed(self, ws: WatershedUnit) -> None:
        """Add a watershed unit to the component."""
        self.watersheds[ws.id] = ws

    def get_watershed(self, ws_id: int) -> WatershedUnit:
        """Get a watershed unit by ID."""
        return self.watersheds[ws_id]

    def iter_watersheds(self) -> Iterator[WatershedUnit]:
        """Iterate over watersheds in ID order."""
        for wid in sorted(self.watersheds.keys()):
            yield self.watersheds[wid]

    def validate(self) -> None:
        """Validate the small watershed component.

        Raises:
            ComponentError: If component is invalid
        """
        for ws in self.watersheds.values():
            if ws.area <= 0:
                raise ComponentError(
                    f"Watershed {ws.id} has non-positive area: {ws.area}"
                )
            if ws.dest_stream_node <= 0:
                raise ComponentError(
                    f"Watershed {ws.id} has invalid destination "
                    f"stream node: {ws.dest_stream_node}"
                )
            if ws.n_gw_nodes == 0:
                raise ComponentError(
                    f"Watershed {ws.id} has no connected GW nodes"
                )

    @classmethod
    def from_config(cls, config: SmallWatershedMainConfig) -> AppSmallWatershed:
        """Create component from a parsed SmallWatershedMainConfig.

        Args:
            config: Parsed configuration from the reader

        Returns:
            AppSmallWatershed instance
        """
        comp = cls(
            area_factor=config.area_factor,
            flow_factor=config.flow_factor,
            flow_time_unit=config.flow_time_unit,
            rz_solver_tolerance=config.rz_solver_tolerance,
            rz_max_iterations=config.rz_max_iterations,
            rz_length_factor=config.rz_length_factor,
            rz_cn_factor=config.rz_cn_factor,
            rz_k_factor=config.rz_k_factor,
            rz_k_time_unit=config.rz_k_time_unit,
            aq_gw_factor=config.aq_gw_factor,
            aq_time_factor=config.aq_time_factor,
            aq_time_unit=config.aq_time_unit,
            ic_factor=config.ic_factor,
            budget_output_file=(
                str(config.budget_output_file) if config.budget_output_file else ""
            ),
            final_results_file=(
                str(config.final_results_file) if config.final_results_file else ""
            ),
        )

        # Build lookup dicts for rootzone, aquifer, and IC params by ID
        rz_by_id = {rz.id: rz for rz in config.rootzone_params}
        aq_by_id = {aq.id: aq for aq in config.aquifer_params}
        ic_by_id = {ic.id: ic for ic in config.initial_conditions}

        for spec in config.watershed_specs:
            rz = rz_by_id.get(spec.id)
            aq = aq_by_id.get(spec.id)

            ws = WatershedUnit(
                id=spec.id,
                area=spec.area,
                dest_stream_node=spec.dest_stream_node,
                gw_nodes=[
                    WatershedGWNode(
                        gw_node_id=gn.gw_node_id,
                        max_perc_rate=gn.max_perc_rate,
                        is_baseflow=gn.is_baseflow,
                        layer=gn.layer,
                    )
                    for gn in spec.gw_nodes
                ],
            )

            if rz is not None:
                ws.precip_col = rz.precip_col
                ws.precip_factor = rz.precip_factor
                ws.et_col = rz.et_col
                ws.wilting_point = rz.wilting_point
                ws.field_capacity = rz.field_capacity
                ws.total_porosity = rz.total_porosity
                ws.lambda_param = rz.lambda_param
                ws.root_depth = rz.root_depth
                ws.hydraulic_cond = rz.hydraulic_cond
                ws.kunsat_method = rz.kunsat_method
                ws.curve_number = rz.curve_number

            if aq is not None:
                ws.gw_threshold = aq.gw_threshold
                ws.max_gw_storage = aq.max_gw_storage
                ws.surface_flow_coeff = aq.surface_flow_coeff
                ws.baseflow_coeff = aq.baseflow_coeff

            ic = ic_by_id.get(spec.id)
            if ic is not None:
                ws.initial_soil_moisture = ic.soil_moisture
                ws.initial_gw_storage = ic.gw_storage

            comp.add_watershed(ws)

        return comp

    def __repr__(self) -> str:
        return f"AppSmallWatershed(n_watersheds={self.n_watersheds})"
