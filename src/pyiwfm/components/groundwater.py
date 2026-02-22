"""
Groundwater component classes for IWFM models.

This module provides classes for representing groundwater system components
including wells, pumping, boundary conditions, tile drains, and aquifer
parameters. It mirrors IWFM's Package_AppGW.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pyiwfm.core.base_component import BaseComponent
from pyiwfm.core.exceptions import ComponentError


@dataclass
class Well:
    """
    A pumping well in the groundwater system.

    Attributes:
        id: Unique well identifier
        name: Well name/description
        x: X coordinate
        y: Y coordinate
        element: Element ID containing the well
        top_screen: Top of screen elevation
        bottom_screen: Bottom of screen elevation
        max_pump_rate: Maximum allowable pumping rate
        pump_rate: Current pumping rate
        layers: List of layers the well screens across
        radius: Well radius
        pump_column: Column in time series pumping file
        pump_fraction: Fraction of pumping at this column
        dist_method: Distribution method (0-4)
        dest_type: Destination type (-1, 0, 1, 2, 3)
        dest_id: Destination ID (element, subregion, or group)
        irig_frac_column: Column for irrigation fraction
        adjust_column: Column for supply adjustment
        pump_max_column: Column for maximum pumping
        pump_max_fraction: Fraction of maximum pumping
    """

    id: int
    x: float
    y: float
    element: int
    name: str = ""
    top_screen: float = 0.0
    bottom_screen: float = 0.0
    max_pump_rate: float = float("inf")
    pump_rate: float = 0.0
    layers: list[int] = field(default_factory=list)
    radius: float = 0.0
    pump_column: int = 0
    pump_fraction: float = 1.0
    dist_method: int = 0
    dest_type: int = -1
    dest_id: int = 0
    irig_frac_column: int = 0
    adjust_column: int = 0
    pump_max_column: int = 0
    pump_max_fraction: float = 0.0

    @property
    def screen_length(self) -> float:
        """Return the length of the well screen."""
        return abs(self.top_screen - self.bottom_screen)

    @property
    def n_layers(self) -> int:
        """Return number of layers the well screens across."""
        return len(self.layers)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Well):
            return NotImplemented
        return self.id == other.id and self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        return hash((self.id, self.x, self.y))

    def __repr__(self) -> str:
        return f"Well(id={self.id}, element={self.element}, rate={self.pump_rate})"


@dataclass
class ElementPumping:
    """
    Pumping applied to an element (distributed pumping).

    Attributes:
        element_id: Element ID
        layer: Layer number
        pump_rate: Total pumping rate
        layer_fraction: Fraction of pumping from this layer
        pump_column: Column in time series pumping file
        pump_fraction: Fraction of pumping at this column
        dist_method: Distribution method (0-4)
        layer_factors: Layer distribution factors
        dest_type: Destination type (-1, 0, 1, 2, 3)
        dest_id: Destination ID
        irig_frac_column: Column for irrigation fraction
        adjust_column: Column for supply adjustment
        pump_max_column: Column for maximum pumping
        pump_max_fraction: Fraction of maximum pumping
    """

    element_id: int
    layer: int
    pump_rate: float
    layer_fraction: float = 1.0
    pump_column: int = 0
    pump_fraction: float = 1.0
    dist_method: int = 0
    layer_factors: list[float] = field(default_factory=list)
    dest_type: int = -1
    dest_id: int = 0
    irig_frac_column: int = 0
    adjust_column: int = 0
    pump_max_column: int = 0
    pump_max_fraction: float = 0.0

    @property
    def effective_rate(self) -> float:
        """Return effective pumping rate after applying fraction."""
        return self.pump_rate * self.layer_fraction

    def __repr__(self) -> str:
        return f"ElementPumping(elem={self.element_id}, layer={self.layer}, rate={self.pump_rate})"


@dataclass
class BoundaryCondition:
    """
    A groundwater boundary condition.

    Attributes:
        id: Unique BC identifier
        bc_type: Type of boundary ('specified_head', 'specified_flow',
            'general_head', 'constrained_general_head')
        nodes: List of node IDs where BC is applied
        values: BC values at each node
        layer: Layer number
        conductance: Conductance values for general head BC
        constraining_head: Constraining head for constrained GH BC
        max_flow: Maximum flow for constrained GH BC
        ts_column: Time series column (0 = static)
        max_flow_ts_column: Time series column for max flow (0 = static)
    """

    id: int
    bc_type: str
    nodes: list[int]
    values: list[float]
    layer: int
    conductance: list[float] = field(default_factory=list)
    constraining_head: float = 0.0
    max_flow: float = 0.0
    ts_column: int = 0
    max_flow_ts_column: int = 0

    def __post_init__(self) -> None:
        """Validate boundary condition data."""
        valid_types = (
            "specified_head",
            "specified_flow",
            "general_head",
            "constrained_general_head",
        )
        if self.bc_type not in valid_types:
            raise ValueError(f"bc_type must be one of {valid_types}")

        if len(self.nodes) != len(self.values):
            raise ValueError("nodes and values must have same length")

        if self.bc_type in ("general_head", "constrained_general_head"):
            if len(self.conductance) != len(self.nodes):
                raise ValueError(f"{self.bc_type} BC requires conductance for each node")

    def __repr__(self) -> str:
        return f"BoundaryCondition(id={self.id}, type={self.bc_type}, n_nodes={len(self.nodes)})"


@dataclass
class TileDrain:
    """
    A tile drain in the groundwater system.

    Attributes:
        id: Unique drain identifier
        element: Element ID containing the drain
        elevation: Drain elevation
        conductance: Drain conductance
        destination_type: Where drain water goes ('stream', 'outside', etc.)
        destination_id: ID of destination (e.g., stream node)
    """

    id: int
    element: int
    elevation: float
    conductance: float
    destination_type: str = "outside"
    destination_id: int | None = None

    def __repr__(self) -> str:
        return f"TileDrain(id={self.id}, element={self.element})"


@dataclass
class Subsidence:
    """
    Subsidence parameters for a node/layer.

    Attributes:
        element: Node ID (historical name; actually per-node in IWFM)
        layer: Layer number
        elastic_storage: Elastic skeletal storage coefficient
        inelastic_storage: Inelastic skeletal storage coefficient
        preconsolidation_head: Preconsolidation head
        interbed_thick: Interbed thickness
        interbed_thick_min: Minimum interbed thickness
    """

    element: int
    layer: int
    elastic_storage: float
    inelastic_storage: float
    preconsolidation_head: float
    interbed_thick: float = 0.0
    interbed_thick_min: float = 0.0

    @property
    def node(self) -> int:
        """Alias for element (subsidence is actually per-node)."""
        return self.element

    def __repr__(self) -> str:
        return f"Subsidence(node={self.element}, layer={self.layer})"


@dataclass
class NodeSubsidence:
    """
    Subsidence parameters for a node across all layers.

    Mirrors the per-node subsidence data from IWFM's subsidence file.

    Attributes:
        node_id: GW node ID (1-based)
        elastic_sc: Elastic storage coefficient per layer
        inelastic_sc: Inelastic storage coefficient per layer
        interbed_thick: Interbed thickness per layer
        interbed_thick_min: Minimum interbed thickness per layer
        precompact_head: Pre-compaction head per layer
        kv_sub: Vertical conductivity of interbeds per layer (v5.0 only)
        n_eq: Number of equivalent delay interbeds per layer (v5.0 only)
    """

    node_id: int
    elastic_sc: list[float] = field(default_factory=list)
    inelastic_sc: list[float] = field(default_factory=list)
    interbed_thick: list[float] = field(default_factory=list)
    interbed_thick_min: list[float] = field(default_factory=list)
    precompact_head: list[float] = field(default_factory=list)
    kv_sub: list[float] = field(default_factory=list)
    n_eq: list[float] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"NodeSubsidence(node={self.node_id}, n_layers={len(self.elastic_sc)})"


@dataclass
class SubIrrigation:
    """
    A sub-irrigation location in the groundwater system.

    Attributes:
        id: Unique identifier
        gw_node: GW node ID
        elevation: Sub-irrigation elevation
        conductance: Conductance value
    """

    id: int
    gw_node: int
    elevation: float
    conductance: float

    def __repr__(self) -> str:
        return f"SubIrrigation(id={self.id}, node={self.gw_node})"


@dataclass
class AquiferParameters:
    """
    Aquifer parameters for the groundwater system.

    Attributes:
        n_nodes: Number of nodes
        n_layers: Number of layers
        kh: Horizontal hydraulic conductivity (n_nodes, n_layers)
        kv: Vertical hydraulic conductivity (n_nodes, n_layers)
        specific_storage: Specific storage (n_nodes, n_layers)
        specific_yield: Specific yield (n_nodes, n_layers)
        aquitard_kv: Aquitard vertical hydraulic conductivity (n_nodes, n_layers)
    """

    n_nodes: int
    n_layers: int
    kh: NDArray[np.float64] | None = None
    kv: NDArray[np.float64] | None = None
    specific_storage: NDArray[np.float64] | None = None
    specific_yield: NDArray[np.float64] | None = None
    aquitard_kv: NDArray[np.float64] | None = None

    def get_layer_kh(self, layer: int) -> NDArray[np.float64]:
        """Get horizontal K for a specific layer."""
        if self.kh is None:
            raise ValueError("kh not set")
        return self.kh[:, layer]

    def get_layer_kv(self, layer: int) -> NDArray[np.float64]:
        """Get vertical K for a specific layer."""
        if self.kv is None:
            raise ValueError("kv not set")
        return self.kv[:, layer]

    def __repr__(self) -> str:
        return f"AquiferParameters(n_nodes={self.n_nodes}, n_layers={self.n_layers})"


@dataclass
class HydrographLocation:
    """
    A groundwater hydrograph output location (observation point).

    These are locations in the model where groundwater heads are tracked
    over time for comparison with field observations.

    Attributes:
        node_id: Mesh node ID for this observation
        layer: Aquifer layer (1-based)
        x: X coordinate
        y: Y coordinate
        name: Optional name/description
    """

    node_id: int
    layer: int
    x: float
    y: float
    name: str = ""

    def __repr__(self) -> str:
        return f"HydrographLocation(node={self.node_id}, layer={self.layer})"


@dataclass
class AppGW(BaseComponent):
    """
    Groundwater application component.

    This class manages the groundwater system including wells, pumping,
    boundary conditions, and aquifer parameters. It mirrors IWFM's Package_AppGW.

    Attributes:
        n_nodes: Number of nodes
        n_layers: Number of aquifer layers
        n_elements: Number of elements
        wells: Dictionary mapping well ID to Well
        boundary_conditions: List of boundary conditions
        tile_drains: Dictionary mapping drain ID to TileDrain
        aquifer_params: Aquifer parameters
        heads: Current heads array (n_nodes, n_layers)
        hydrograph_locations: List of observation points for head output
    """

    n_nodes: int
    n_layers: int
    n_elements: int
    wells: dict[int, Well] = field(default_factory=dict)
    boundary_conditions: list[BoundaryCondition] = field(default_factory=list)
    tile_drains: dict[int, TileDrain] = field(default_factory=dict)
    element_pumping: list[ElementPumping] = field(default_factory=list)
    subsidence: list[Subsidence] = field(default_factory=list)
    node_subsidence: list[NodeSubsidence] = field(default_factory=list)
    sub_irrigations: list[SubIrrigation] = field(default_factory=list)
    aquifer_params: AquiferParameters | None = None
    heads: NDArray[np.float64] | None = field(default=None, repr=False)
    hydrograph_locations: list[HydrographLocation] = field(default_factory=list)
    face_flow_specs: list[str] = field(default_factory=list)
    kh_anomalies: list[str] = field(default_factory=list)
    return_flow_destinations: dict[int, tuple[int, int]] = field(default_factory=dict)
    parametric_groups: list[Any] = field(default_factory=list)
    # Full config objects for roundtrip fidelity
    subsidence_config: Any = field(default=None, repr=False)
    bc_config: Any = field(default=None, repr=False)
    gw_main_config: Any = field(default=None, repr=False)
    # Time series file paths for lazy loading
    pumping_ts_file: Any = field(default=None, repr=False)
    bc_ts_file: Any = field(default=None, repr=False)
    # Boundary node flow output (NOUTB section in BC_MAIN)
    n_bc_output_nodes: int = 0
    bc_output_file: str = ""
    bc_output_file_raw: str = ""
    bc_output_specs: list[Any] = field(default_factory=list)
    # Tile drain conversion factors (preserved for roundtrip fidelity)
    td_elev_factor: float = 1.0
    td_cond_factor: float = 1.0
    td_time_unit: str = "1DAY"
    # Sub-irrigation conversion factors
    si_elev_factor: float = 1.0
    si_cond_factor: float = 1.0
    si_time_unit: str = "1MON"
    # Tile drain hydrograph output (preserved for roundtrip fidelity)
    td_n_hydro: int = 0
    td_hydro_volume_factor: float = 1.0
    td_hydro_volume_unit: str = ""
    td_output_file_raw: str = ""
    td_hydro_specs: list[dict[str, Any]] = field(default_factory=list)

    @property
    def n_items(self) -> int:
        """Return number of wells (primary entities)."""
        return len(self.wells)

    @property
    def n_wells(self) -> int:
        """Return number of wells."""
        return len(self.wells)

    @property
    def n_hydrograph_locations(self) -> int:
        """Return number of hydrograph output locations."""
        return len(self.hydrograph_locations)

    @property
    def n_boundary_conditions(self) -> int:
        """Return number of boundary conditions."""
        return len(self.boundary_conditions)

    @property
    def n_tile_drains(self) -> int:
        """Return number of tile drains."""
        return len(self.tile_drains)

    def add_well(self, well: Well) -> None:
        """Add a well to the groundwater system."""
        self.wells[well.id] = well

    def add_boundary_condition(self, bc: BoundaryCondition) -> None:
        """Add a boundary condition."""
        self.boundary_conditions.append(bc)

    def add_tile_drain(self, drain: TileDrain) -> None:
        """Add a tile drain."""
        self.tile_drains[drain.id] = drain

    def add_element_pumping(self, pumping: ElementPumping) -> None:
        """Add element-based pumping."""
        self.element_pumping.append(pumping)

    def add_subsidence(self, sub: Subsidence) -> None:
        """Add subsidence parameters."""
        self.subsidence.append(sub)

    def add_node_subsidence(self, ns: NodeSubsidence) -> None:
        """Add per-node subsidence parameters."""
        self.node_subsidence.append(ns)

    def add_sub_irrigation(self, si: SubIrrigation) -> None:
        """Add a sub-irrigation location."""
        self.sub_irrigations.append(si)

    @property
    def n_sub_irrigations(self) -> int:
        """Return number of sub-irrigation locations."""
        return len(self.sub_irrigations)

    @property
    def n_node_subsidence(self) -> int:
        """Return number of nodes with subsidence data."""
        return len(self.node_subsidence)

    def add_hydrograph_location(self, location: HydrographLocation) -> None:
        """Add a groundwater hydrograph output location."""
        self.hydrograph_locations.append(location)

    def get_well(self, well_id: int) -> Well:
        """Get a well by ID."""
        return self.wells[well_id]

    def get_tile_drain(self, drain_id: int) -> TileDrain:
        """Get a tile drain by ID."""
        return self.tile_drains[drain_id]

    def set_aquifer_parameters(self, params: AquiferParameters) -> None:
        """Set aquifer parameters."""
        if params.n_nodes != self.n_nodes:
            raise ValueError(
                f"Parameter n_nodes ({params.n_nodes}) doesn't match "
                f"component n_nodes ({self.n_nodes})"
            )
        if params.n_layers != self.n_layers:
            raise ValueError(
                f"Parameter n_layers ({params.n_layers}) doesn't match "
                f"component n_layers ({self.n_layers})"
            )
        self.aquifer_params = params

    def set_heads(self, heads: NDArray[np.float64]) -> None:
        """Set the current heads array."""
        if heads.shape != (self.n_nodes, self.n_layers):
            raise ValueError(
                f"Heads shape {heads.shape} doesn't match "
                f"expected ({self.n_nodes}, {self.n_layers})"
            )
        self.heads = heads.copy()

    def get_head(self, node: int, layer: int) -> float:
        """Get head at a specific node and layer."""
        if self.heads is None:
            raise ValueError("Heads not set")
        return float(self.heads[node, layer])

    def get_wells_in_element(self, element_id: int) -> list[Well]:
        """Get all wells in an element."""
        return [w for w in self.wells.values() if w.element == element_id]

    def get_total_pumping(self) -> float:
        """Calculate total pumping from all wells."""
        return sum(w.pump_rate for w in self.wells.values())

    def get_total_element_pumping(self) -> float:
        """Calculate total element-based pumping."""
        return sum(p.effective_rate for p in self.element_pumping)

    def iter_wells(self) -> Iterator[Well]:
        """Iterate over wells in ID order."""
        for wid in sorted(self.wells.keys()):
            yield self.wells[wid]

    def validate(self) -> None:
        """
        Validate the groundwater component.

        Raises:
            ComponentError: If component is invalid
        """
        # Check well element references (element=0 means not yet assigned)
        for well in self.wells.values():
            if well.element != 0 and (well.element > self.n_elements or well.element < 1):
                raise ComponentError(f"Well {well.id} references invalid element {well.element}")

        # Check tile drain element references
        for drain in self.tile_drains.values():
            if drain.element > self.n_elements or drain.element < 1:
                raise ComponentError(
                    f"Tile drain {drain.id} references invalid element {drain.element}"
                )

        # Check boundary condition node references
        for bc in self.boundary_conditions:
            for nid in bc.nodes:
                if nid > self.n_nodes or nid < 1:
                    raise ComponentError(
                        f"Boundary condition {bc.id} references invalid node {nid}"
                    )

    def to_arrays(self) -> dict[str, NDArray]:
        """
        Convert groundwater data to numpy arrays.

        Returns:
            Dictionary of arrays
        """
        result: dict[str, NDArray] = {}

        if self.heads is not None:
            result["heads"] = self.heads.copy()

        if self.aquifer_params is not None:
            if self.aquifer_params.kh is not None:
                result["kh"] = self.aquifer_params.kh.copy()
            if self.aquifer_params.kv is not None:
                result["kv"] = self.aquifer_params.kv.copy()
            if self.aquifer_params.specific_storage is not None:
                result["specific_storage"] = self.aquifer_params.specific_storage.copy()
            if self.aquifer_params.specific_yield is not None:
                result["specific_yield"] = self.aquifer_params.specific_yield.copy()
            if self.aquifer_params.aquitard_kv is not None:
                result["aquitard_kv"] = self.aquifer_params.aquitard_kv.copy()

        return result

    @classmethod
    def from_arrays(
        cls,
        n_nodes: int,
        n_layers: int,
        n_elements: int,
        heads: NDArray[np.float64] | None = None,
        kh: NDArray[np.float64] | None = None,
        kv: NDArray[np.float64] | None = None,
        specific_storage: NDArray[np.float64] | None = None,
        specific_yield: NDArray[np.float64] | None = None,
    ) -> AppGW:
        """
        Create groundwater component from arrays.

        Args:
            n_nodes: Number of nodes
            n_layers: Number of layers
            n_elements: Number of elements
            heads: Initial heads array
            kh: Horizontal hydraulic conductivity
            kv: Vertical hydraulic conductivity
            specific_storage: Specific storage
            specific_yield: Specific yield

        Returns:
            AppGW instance
        """
        gw = cls(n_nodes=n_nodes, n_layers=n_layers, n_elements=n_elements)

        if heads is not None:
            gw.set_heads(heads)

        if any(arr is not None for arr in [kh, kv, specific_storage, specific_yield]):
            params = AquiferParameters(
                n_nodes=n_nodes,
                n_layers=n_layers,
                kh=kh,
                kv=kv,
                specific_storage=specific_storage,
                specific_yield=specific_yield,
            )
            gw.set_aquifer_parameters(params)

        return gw

    def __repr__(self) -> str:
        return f"AppGW(n_nodes={self.n_nodes}, n_layers={self.n_layers}, n_wells={self.n_wells})"
