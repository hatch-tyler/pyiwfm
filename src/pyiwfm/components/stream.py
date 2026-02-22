"""
Stream network classes for IWFM models.

This module provides classes for representing stream networks, including
stream nodes, reaches, diversions, and bypasses. It mirrors IWFM's
Package_AppStream.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pyiwfm.core.base_component import BaseComponent
from pyiwfm.core.exceptions import ComponentError


def _ccw(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> bool:
    """
    Check if three points are in counter-clockwise order.

    Uses the cross product of vectors AB and AC.
    """
    return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)


def segments_intersect(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
    tolerance: float = 1e-10,
) -> bool:
    """
    Check if two line segments intersect (excluding shared endpoints).

    Segment 1: p1 to p2
    Segment 2: p3 to p4

    Parameters
    ----------
    p1, p2 : tuple of float
        Endpoints of first segment (x, y).
    p3, p4 : tuple of float
        Endpoints of second segment (x, y).
    tolerance : float
        Distance tolerance for considering points as shared endpoints.

    Returns
    -------
    bool
        True if segments intersect at a point that is not a shared endpoint.

    Examples
    --------
    >>> segments_intersect((0, 0), (2, 2), (0, 2), (2, 0))  # X crossing
    True
    >>> segments_intersect((0, 0), (1, 1), (1, 1), (2, 2))  # Shared endpoint
    False
    >>> segments_intersect((0, 0), (1, 0), (2, 0), (3, 0))  # Parallel, no overlap
    False
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    # Check if segments share an endpoint (this is allowed for connected reaches)
    def points_equal(pa: tuple[float, float], pb: tuple[float, float]) -> bool:
        return abs(pa[0] - pb[0]) < tolerance and abs(pa[1] - pb[1]) < tolerance

    if points_equal(p1, p3) or points_equal(p1, p4) or points_equal(p2, p3) or points_equal(p2, p4):
        return False

    # Use the CCW algorithm to detect intersection
    # Two segments intersect if and only if:
    # - Points of one segment are on opposite sides of the other segment's line
    # - AND vice versa
    ccw1 = _ccw(x1, y1, x3, y3, x4, y4)
    ccw2 = _ccw(x2, y2, x3, y3, x4, y4)
    ccw3 = _ccw(x1, y1, x2, y2, x3, y3)
    ccw4 = _ccw(x1, y1, x2, y2, x4, y4)

    return ccw1 != ccw2 and ccw3 != ccw4


@dataclass
class ReachCrossing:
    """
    Represents a detected crossing between two stream reach segments.

    Attributes
    ----------
    segment1 : tuple
        First segment as ((x1, y1), (x2, y2), reach_id, segment_index).
    segment2 : tuple
        Second segment as ((x1, y1), (x2, y2), reach_id, segment_index).
    intersection_point : tuple of float, optional
        Approximate intersection point (x, y) if computed.
    """

    segment1: tuple[tuple[float, float], tuple[float, float], int, int]
    segment2: tuple[tuple[float, float], tuple[float, float], int, int]
    intersection_point: tuple[float, float] | None = None

    @property
    def reach1_id(self) -> int:
        """ID of first reach involved in crossing."""
        return self.segment1[2]

    @property
    def reach2_id(self) -> int:
        """ID of second reach involved in crossing."""
        return self.segment2[2]

    def __repr__(self) -> str:
        return (
            f"ReachCrossing(reach {self.reach1_id} segment {self.segment1[3]} "
            f"crosses reach {self.reach2_id} segment {self.segment2[3]})"
        )


@dataclass
class CrossSectionData:
    """v5.0 hydraulic cross-section parameters for Manning's equation.

    Attributes:
        bottom_elev: Stream bed bottom elevation
        B0: Bottom width (0 = triangular channel)
        s: Inverse side gradient (0 = rectangular channel)
        n: Manning's roughness coefficient
        max_flow_depth: Maximum flow depth
    """

    bottom_elev: float = 0.0
    B0: float = 0.0
    s: float = 0.0
    n: float = 0.04
    max_flow_depth: float = 10.0


@dataclass
class StrmEvapNodeSpec:
    """Per-node stream evaporation column pointers.

    Attributes:
        node_id: Stream node ID
        et_column: Column index in ET data file
        area_column: Column index in stream surface area file
    """

    node_id: int
    et_column: int = 0
    area_column: int = 0


@dataclass
class StreamRating:
    """
    Stage-discharge rating curve for a stream node.

    Attributes:
        stages: Array of stage (water level) values
        flows: Array of corresponding flow values
    """

    stages: NDArray[np.float64]
    flows: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate rating curve data."""
        if len(self.stages) != len(self.flows):
            raise ValueError("stages and flows arrays must have same length")
        if len(self.stages) < 2:
            raise ValueError("Rating curve must have at least 2 points")

    def get_flow(self, stage: float) -> float:
        """
        Interpolate flow from stage.

        Args:
            stage: Water level

        Returns:
            Interpolated flow rate
        """
        if stage <= self.stages[0]:
            return max(0.0, float(self.flows[0]))

        if stage >= self.stages[-1]:
            # Linear extrapolation above max stage
            slope = (self.flows[-1] - self.flows[-2]) / (self.stages[-1] - self.stages[-2])
            return float(self.flows[-1] + slope * (stage - self.stages[-1]))

        return float(np.interp(stage, self.stages, self.flows))

    def get_stage(self, flow: float) -> float:
        """
        Interpolate stage from flow.

        Args:
            flow: Flow rate

        Returns:
            Interpolated water level
        """
        if flow <= self.flows[0]:
            return float(self.stages[0])

        if flow >= self.flows[-1]:
            # Linear extrapolation above max flow
            slope = (self.stages[-1] - self.stages[-2]) / (self.flows[-1] - self.flows[-2])
            return float(self.stages[-1] + slope * (flow - self.flows[-1]))

        return float(np.interp(flow, self.flows, self.stages))


@dataclass
class StrmNode:
    """
    A stream node representing a point in the stream network.

    Attributes:
        id: Unique node identifier
        x: X coordinate
        y: Y coordinate
        reach_id: ID of the reach this node belongs to
        gw_node: ID of the associated groundwater node (for stream-aquifer interaction)
        bottom_elev: Stream bed elevation
        wetted_perimeter: Wetted perimeter for flow calculations
        upstream_node: ID of upstream node
        downstream_node: ID of downstream node
        rating: Stage-discharge rating curve
    """

    id: int
    x: float
    y: float
    reach_id: int = 0
    gw_node: int | None = None
    bottom_elev: float = 0.0
    wetted_perimeter: float = 0.0
    upstream_node: int | None = None
    downstream_node: int | None = None
    rating: StreamRating | None = None
    conductivity: float = 0.0
    bed_thickness: float = 0.0
    cross_section: CrossSectionData | None = None
    initial_condition: float = 0.0

    def distance_to(self, other: StrmNode) -> float:
        """Calculate distance to another stream node."""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StrmNode):
            return NotImplemented
        return self.id == other.id and self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        return hash((self.id, self.x, self.y))

    def __repr__(self) -> str:
        return f"StrmNode(id={self.id}, reach={self.reach_id})"


@dataclass
class StrmReach:
    """
    A stream reach representing a segment of the stream network.

    Attributes:
        id: Unique reach identifier
        name: Descriptive name
        upstream_node: ID of the most upstream node
        downstream_node: ID of the most downstream node
        nodes: List of node IDs in this reach (upstream to downstream order)
        outflow_destination: Where outflow goes - tuple of (type, id)
            type: 'reach', 'lake', 'boundary', etc.
            id: destination ID
    """

    id: int
    upstream_node: int
    downstream_node: int
    name: str = ""
    nodes: list[int] = field(default_factory=list)
    outflow_destination: tuple[str, int] | None = None

    @property
    def n_nodes(self) -> int:
        """Return number of nodes in this reach."""
        return len(self.nodes)

    def __repr__(self) -> str:
        return f"StrmReach(id={self.id}, name='{self.name}', n_nodes={self.n_nodes})"


@dataclass
class Diversion:
    """
    A stream diversion that removes water from the stream.

    Attributes:
        id: Unique diversion identifier
        name: Descriptive name
        source_node: Stream node ID where water is diverted
        destination_type: Type of destination ('element', 'stream_node', 'outside')
        destination_id: ID of destination
        max_rate: Maximum diversion rate
        priority: Allocation priority (lower = higher priority)
    """

    id: int
    source_node: int
    destination_type: str
    destination_id: int
    name: str = ""
    max_rate: float = float("inf")
    priority: int = 99

    # Extended diversion spec fields (full IWFM template columns)
    reach_id: int = 0
    max_div_column: int = 0
    max_div_fraction: float = 1.0
    recoverable_loss_column: int = 0
    recoverable_loss_fraction: float = 0.0
    non_recoverable_loss_column: int = 0
    non_recoverable_loss_fraction: float = 0.0
    spill_column: int = 0
    spill_fraction: float = 0.0
    delivery_dest_type: int = 0
    delivery_dest_id: int = 0
    delivery_column: int = 0
    delivery_fraction: float = 1.0
    irrigation_fraction_column: int = 0
    adjustment_column: int = 0

    # Element groups, recharge zones, spill locations
    element_groups: list[Any] = field(default_factory=list)
    recharge_zones: list[Any] = field(default_factory=list)
    spill_locations: list[Any] = field(default_factory=list)

    @property
    def n_delivery_locs(self) -> int:
        """Number of delivery locations (element groups)."""
        return len(self.element_groups)

    def __repr__(self) -> str:
        return f"Diversion(id={self.id}, source={self.source_node}, max_rate={self.max_rate})"


@dataclass
class Bypass:
    """
    A stream bypass that routes water around a section.

    Attributes:
        id: Unique bypass identifier
        name: Descriptive name
        source_node: Stream node ID where bypass starts
        destination_node: Stream node ID where bypass ends
        capacity: Maximum bypass flow capacity
    """

    id: int
    source_node: int
    destination_node: int
    dest_type: int = 0  # 0=outside, 1=stream node, 3=lake
    name: str = ""
    capacity: float = float("inf")

    # Extended bypass spec fields
    flow_factor: float = 1.0
    flow_time_unit: str = "1DAY"
    spill_factor: float = 1.0
    spill_time_unit: str = "1DAY"
    diversion_column: int = 0
    recoverable_loss_fraction: float = 0.0
    non_recoverable_loss_fraction: float = 0.0
    rating_table_flows: list[float] = field(default_factory=list)
    rating_table_spills: list[float] = field(default_factory=list)
    seepage_locations: list[Any] = field(default_factory=list)

    @property
    def n_seepage_locs(self) -> int:
        """Number of seepage locations."""
        return len(self.seepage_locations)

    @property
    def has_rating_table(self) -> bool:
        """Whether this bypass uses a rating table (negative IDIVC)."""
        return self.diversion_column < 0

    @property
    def n_rating_points(self) -> int:
        """Number of rating table points."""
        return abs(self.diversion_column) if self.has_rating_table else 0

    def __repr__(self) -> str:
        return f"Bypass(id={self.id}, src={self.source_node}, dst={self.destination_node})"


@dataclass
class AppStream(BaseComponent):
    """
    Stream network application class.

    This class manages the complete stream network including nodes, reaches,
    diversions, and bypasses. It mirrors IWFM's Package_AppStream.

    Attributes:
        nodes: Dictionary mapping node ID to StrmNode
        reaches: Dictionary mapping reach ID to StrmReach
        diversions: Dictionary mapping diversion ID to Diversion
        bypasses: Dictionary mapping bypass ID to Bypass
    """

    nodes: dict[int, StrmNode] = field(default_factory=dict)
    reaches: dict[int, StrmReach] = field(default_factory=dict)
    diversions: dict[int, Diversion] = field(default_factory=dict)
    bypasses: dict[int, Bypass] = field(default_factory=dict)

    # Stream bed parameter metadata
    conductivity_factor: float = 1.0
    conductivity_time_unit: str = ""
    length_factor: float = 1.0
    interaction_type: int = 1

    # Stream evaporation
    evap_area_file: str = ""
    evap_node_specs: list[StrmEvapNodeSpec] = field(default_factory=list)

    # v5.0 initial conditions
    ic_type: int = 0
    ic_time_unit: str = ""
    ic_factor: float = 1.0

    # v5.0 end-of-simulation file
    final_flow_file: str = ""

    # Stream node budget
    budget_node_count: int = 0
    budget_output_file: str = ""
    budget_node_ids: list[int] = field(default_factory=list)

    # v5.0 cross-section conversion factors
    roughness_factor: float = 1.0
    cross_section_length_factor: float = 1.0

    # Diversion element groups and recharge zones
    diversion_element_groups: list[Any] = field(default_factory=list)
    diversion_recharge_zones: list[Any] = field(default_factory=list)
    diversion_spill_zones: list[Any] = field(default_factory=list)
    diversion_has_spills: bool = False

    # Connectivity caches
    _downstream_map: dict[int, int | None] = field(default_factory=dict, repr=False)
    _upstream_map: dict[int, int | None] = field(default_factory=dict, repr=False)

    @property
    def n_items(self) -> int:
        """Return number of stream nodes (primary entities)."""
        return len(self.nodes)

    @property
    def n_nodes(self) -> int:
        """Return number of stream nodes."""
        return len(self.nodes)

    @property
    def n_reaches(self) -> int:
        """Return number of reaches."""
        return len(self.reaches)

    @property
    def n_diversions(self) -> int:
        """Return number of diversions."""
        return len(self.diversions)

    @property
    def n_bypasses(self) -> int:
        """Return number of bypasses."""
        return len(self.bypasses)

    def add_node(self, node: StrmNode) -> None:
        """Add a stream node to the network."""
        self.nodes[node.id] = node

    def add_reach(self, reach: StrmReach) -> None:
        """Add a reach to the network."""
        self.reaches[reach.id] = reach

    def add_diversion(self, diversion: Diversion) -> None:
        """Add a diversion to the network."""
        self.diversions[diversion.id] = diversion

    def add_bypass(self, bypass: Bypass) -> None:
        """Add a bypass to the network."""
        self.bypasses[bypass.id] = bypass

    def get_node(self, node_id: int) -> StrmNode:
        """Get a stream node by ID."""
        return self.nodes[node_id]

    def get_reach(self, reach_id: int) -> StrmReach:
        """Get a reach by ID."""
        return self.reaches[reach_id]

    def get_diversion(self, div_id: int) -> Diversion:
        """Get a diversion by ID."""
        return self.diversions[div_id]

    def get_bypass(self, bypass_id: int) -> Bypass:
        """Get a bypass by ID."""
        return self.bypasses[bypass_id]

    def get_nodes_in_reach(self, reach_id: int) -> list[StrmNode]:
        """Get all nodes in a reach."""
        reach = self.reaches[reach_id]
        return [self.nodes[nid] for nid in reach.nodes if nid in self.nodes]

    def build_connectivity(self) -> None:
        """Build node connectivity maps from node data."""
        self._downstream_map.clear()
        self._upstream_map.clear()

        for node in self.nodes.values():
            if node.downstream_node is not None:
                self._downstream_map[node.id] = node.downstream_node
            if node.upstream_node is not None:
                self._upstream_map[node.id] = node.upstream_node

    def get_downstream_node(self, node_id: int) -> int | None:
        """Get the downstream node ID for a given node."""
        if node_id in self._downstream_map:
            return self._downstream_map[node_id]
        node = self.nodes.get(node_id)
        if node:
            return node.downstream_node
        return None

    def get_upstream_node(self, node_id: int) -> int | None:
        """Get the upstream node ID for a given node."""
        if node_id in self._upstream_map:
            return self._upstream_map[node_id]
        node = self.nodes.get(node_id)
        if node:
            return node.upstream_node
        return None

    def get_reach_length(self, reach_id: int) -> float:
        """
        Calculate the length of a reach.

        Args:
            reach_id: Reach ID

        Returns:
            Total length of the reach
        """
        self.reaches[reach_id]
        nodes = self.get_nodes_in_reach(reach_id)

        if len(nodes) < 2:
            return 0.0

        total_length = 0.0
        for i in range(len(nodes) - 1):
            total_length += nodes[i].distance_to(nodes[i + 1])

        return total_length

    def get_total_length(self) -> float:
        """Calculate total length of all reaches."""
        return sum(self.get_reach_length(rid) for rid in self.reaches)

    def iter_nodes(self) -> Iterator[StrmNode]:
        """Iterate over nodes in ID order."""
        for nid in sorted(self.nodes.keys()):
            yield self.nodes[nid]

    def iter_reaches(self) -> Iterator[StrmReach]:
        """Iterate over reaches in ID order."""
        for rid in sorted(self.reaches.keys()):
            yield self.reaches[rid]

    def validate(self) -> None:
        """
        Validate the stream network.

        Raises:
            ComponentError: If network is invalid
        """
        if not self.nodes:
            raise ComponentError("Stream network has no nodes")

        # Check reach node references
        for reach in self.reaches.values():
            for nid in reach.nodes:
                if nid not in self.nodes:
                    raise ComponentError(f"Reach {reach.id} references non-existent node {nid}")

        # Check diversion source nodes
        for div in self.diversions.values():
            if div.source_node not in self.nodes:
                raise ComponentError(
                    f"Diversion {div.id} references non-existent source node {div.source_node}"
                )

        # Check bypass nodes
        for bypass in self.bypasses.values():
            if bypass.source_node not in self.nodes:
                raise ComponentError(
                    f"Bypass {bypass.id} references non-existent source node {bypass.source_node}"
                )
            if bypass.destination_node not in self.nodes:
                raise ComponentError(
                    f"Bypass {bypass.id} references non-existent destination node {bypass.destination_node}"
                )

    def get_reach_segments(
        self, reach_id: int
    ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        """
        Get all line segments for a reach.

        Parameters
        ----------
        reach_id : int
            The reach ID.

        Returns
        -------
        list of tuple
            List of segments, each as ((x1, y1), (x2, y2)).
        """
        nodes = self.get_nodes_in_reach(reach_id)
        segments = []
        for i in range(len(nodes) - 1):
            p1 = (nodes[i].x, nodes[i].y)
            p2 = (nodes[i + 1].x, nodes[i + 1].y)
            segments.append((p1, p2))
        return segments

    def detect_crossings(self) -> list[ReachCrossing]:
        """
        Detect all crossing (intersecting) reach segments in the network.

        Stream reaches can share endpoints (confluences/junctions) but cannot
        cross each other geometrically. This method identifies all such invalid
        crossings.

        Returns
        -------
        list of ReachCrossing
            List of detected crossings. Empty if no crossings found.

        Examples
        --------
        >>> stream = AppStream()
        >>> # ... add nodes and reaches ...
        >>> crossings = stream.detect_crossings()
        >>> if crossings:
        ...     print(f"Found {len(crossings)} crossing(s)")
        ...     for c in crossings:
        ...         print(f"  {c}")
        """
        crossings: list[ReachCrossing] = []

        # Build list of all segments with their reach IDs
        all_segments: list[tuple[tuple[float, float], tuple[float, float], int, int]] = []

        for reach in self.reaches.values():
            segments = self.get_reach_segments(reach.id)
            for idx, (p1, p2) in enumerate(segments):
                all_segments.append((p1, p2, reach.id, idx))

        # Check all pairs of segments for intersection
        n_segments = len(all_segments)
        for i in range(n_segments):
            p1, p2, reach1, idx1 = all_segments[i]
            for j in range(i + 1, n_segments):
                p3, p4, reach2, idx2 = all_segments[j]

                if segments_intersect(p1, p2, p3, p4):
                    # Compute approximate intersection point
                    intersection = self._compute_intersection(p1, p2, p3, p4)
                    crossing = ReachCrossing(
                        segment1=(p1, p2, reach1, idx1),
                        segment2=(p3, p4, reach2, idx2),
                        intersection_point=intersection,
                    )
                    crossings.append(crossing)

        return crossings

    def _compute_intersection(
        self,
        p1: tuple[float, float],
        p2: tuple[float, float],
        p3: tuple[float, float],
        p4: tuple[float, float],
    ) -> tuple[float, float] | None:
        """
        Compute the intersection point of two line segments.

        Returns None if segments don't intersect or are parallel.
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None  # Parallel or coincident

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)

        return (ix, iy)

    def validate_no_crossings(self, raise_on_error: bool = True) -> list[ReachCrossing]:
        """
        Validate that no stream reaches cross each other.

        Stream reaches can share endpoints (confluences) but cannot cross
        (intersect at non-endpoint locations).

        Parameters
        ----------
        raise_on_error : bool, default True
            If True, raise ComponentError when crossings are found.
            If False, just return the list of crossings.

        Returns
        -------
        list of ReachCrossing
            List of detected crossings.

        Raises
        ------
        ComponentError
            If raise_on_error is True and crossings are found.

        Examples
        --------
        >>> stream.validate_no_crossings()  # Raises if crossings found

        >>> crossings = stream.validate_no_crossings(raise_on_error=False)
        >>> if crossings:
        ...     print(f"Found {len(crossings)} invalid crossing(s)")
        """
        crossings = self.detect_crossings()

        if crossings and raise_on_error:
            msg_parts = [f"Stream network has {len(crossings)} invalid crossing(s):"]
            for _i, c in enumerate(crossings[:5]):  # Show first 5
                point_str = ""
                if c.intersection_point:
                    point_str = (
                        f" at ({c.intersection_point[0]:.1f}, {c.intersection_point[1]:.1f})"
                    )
                msg_parts.append(f"  - Reach {c.reach1_id} crosses Reach {c.reach2_id}{point_str}")
            if len(crossings) > 5:
                msg_parts.append(f"  ... and {len(crossings) - 5} more")
            raise ComponentError("\n".join(msg_parts))

        return crossings

    def to_arrays(self) -> dict[str, NDArray]:
        """
        Convert stream network to numpy arrays.

        Returns:
            Dictionary of arrays
        """
        sorted_ids = sorted(self.nodes.keys())
        len(sorted_ids)

        node_ids = np.array(sorted_ids, dtype=np.int32)
        x = np.array([self.nodes[nid].x for nid in sorted_ids])
        y = np.array([self.nodes[nid].y for nid in sorted_ids])
        reach_ids = np.array([self.nodes[nid].reach_id for nid in sorted_ids], dtype=np.int32)
        gw_nodes = np.array([self.nodes[nid].gw_node or 0 for nid in sorted_ids], dtype=np.int32)

        return {
            "node_ids": node_ids,
            "x": x,
            "y": y,
            "reach_ids": reach_ids,
            "gw_nodes": gw_nodes,
        }

    @classmethod
    def from_arrays(
        cls,
        node_ids: NDArray[np.int32],
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        reach_ids: NDArray[np.int32],
        gw_nodes: NDArray[np.int32] | None = None,
    ) -> AppStream:
        """
        Create stream network from arrays.

        Args:
            node_ids: Array of node IDs
            x: Array of x coordinates
            y: Array of y coordinates
            reach_ids: Array of reach IDs for each node
            gw_nodes: Array of groundwater node IDs (optional)

        Returns:
            AppStream instance
        """
        stream = cls()

        for i, nid in enumerate(node_ids):
            gw = int(gw_nodes[i]) if gw_nodes is not None and gw_nodes[i] != 0 else None
            node = StrmNode(
                id=int(nid),
                x=float(x[i]),
                y=float(y[i]),
                reach_id=int(reach_ids[i]),
                gw_node=gw,
            )
            stream.add_node(node)

        return stream

    def __repr__(self) -> str:
        return (
            f"AppStream(n_nodes={self.n_nodes}, n_reaches={self.n_reaches}, "
            f"n_diversions={self.n_diversions})"
        )
