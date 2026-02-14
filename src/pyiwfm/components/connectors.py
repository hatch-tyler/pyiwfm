"""
Component connectors for IWFM models.

This module provides classes for representing connections between
model components: stream-groundwater, lake-groundwater, and stream-lake
interactions. It mirrors IWFM's Package_ComponentConnectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
from numpy.typing import NDArray

from pyiwfm.core.exceptions import ComponentError


@dataclass
class StreamGWConnection:
    """
    A connection between a stream node and groundwater node.

    Attributes:
        stream_node_id: Stream node ID
        gw_node_id: Groundwater node ID
        layer: Aquifer layer for interaction
        conductance: Streambed conductance
        stream_bed_elev: Stream bed elevation
        stream_bed_thickness: Stream bed thickness
    """

    stream_node_id: int
    gw_node_id: int
    layer: int = 1
    conductance: float = 0.0
    stream_bed_elev: float = 0.0
    stream_bed_thickness: float = 0.0

    def __repr__(self) -> str:
        return f"StreamGWConnection(strm={self.stream_node_id}, gw={self.gw_node_id})"


@dataclass
class StreamGWConnector:
    """
    Connector managing stream-groundwater interactions.

    This class manages all connections between stream nodes and
    groundwater nodes, calculating exchange flows based on head
    differences and conductances.

    Attributes:
        connections: List of stream-groundwater connections
    """

    connections: list[StreamGWConnection] = field(default_factory=list)

    @property
    def n_connections(self) -> int:
        """Return number of connections."""
        return len(self.connections)

    def add_connection(self, conn: StreamGWConnection) -> None:
        """Add a connection."""
        self.connections.append(conn)

    def get_connections_for_stream_node(self, stream_node_id: int) -> list[StreamGWConnection]:
        """Get all connections for a stream node."""
        return [c for c in self.connections if c.stream_node_id == stream_node_id]

    def get_connections_for_gw_node(self, gw_node_id: int) -> list[StreamGWConnection]:
        """Get all connections for a groundwater node."""
        return [c for c in self.connections if c.gw_node_id == gw_node_id]

    def calculate_flow(
        self,
        stream_node_id: int,
        stream_stage: float,
        gw_head: float,
    ) -> float:
        """
        Calculate stream-aquifer exchange flow for a stream node.

        Positive flow = stream to aquifer (losing stream)
        Negative flow = aquifer to stream (gaining stream)

        Args:
            stream_node_id: Stream node ID
            stream_stage: Water surface elevation in stream
            gw_head: Groundwater head at connected node

        Returns:
            Exchange flow rate
        """
        total_flow = 0.0

        for conn in self.get_connections_for_stream_node(stream_node_id):
            head_diff = stream_stage - gw_head
            flow = conn.conductance * head_diff
            total_flow += flow

        return total_flow

    def calculate_total_exchange(
        self,
        stream_stages: dict[int, float],
        gw_heads: dict[int, float],
    ) -> float:
        """
        Calculate total stream-aquifer exchange.

        Args:
            stream_stages: Dictionary of stream node ID to stage
            gw_heads: Dictionary of GW node ID to head

        Returns:
            Total exchange flow (positive = net to aquifer)
        """
        total = 0.0

        for conn in self.connections:
            if conn.stream_node_id in stream_stages and conn.gw_node_id in gw_heads:
                stage = stream_stages[conn.stream_node_id]
                head = gw_heads[conn.gw_node_id]
                head_diff = stage - head
                flow = conn.conductance * head_diff
                total += flow

        return total

    def validate(self) -> None:
        """Validate the connector."""
        pass

    def to_arrays(self) -> dict[str, NDArray]:
        """
        Convert connector data to numpy arrays.

        Returns:
            Dictionary of arrays
        """
        if not self.connections:
            return {}

        stream_ids = np.array([c.stream_node_id for c in self.connections], dtype=np.int32)
        gw_ids = np.array([c.gw_node_id for c in self.connections], dtype=np.int32)
        layers = np.array([c.layer for c in self.connections], dtype=np.int32)
        conductances = np.array([c.conductance for c in self.connections])

        return {
            "stream_node_ids": stream_ids,
            "gw_node_ids": gw_ids,
            "layers": layers,
            "conductances": conductances,
        }

    def __repr__(self) -> str:
        return f"StreamGWConnector(n_connections={self.n_connections})"


@dataclass
class LakeGWConnection:
    """
    A connection between a lake and groundwater node.

    Attributes:
        lake_id: Lake ID
        gw_node_id: Groundwater node ID
        layer: Aquifer layer for interaction
        conductance: Lake bed conductance
        lake_bed_elev: Lake bed elevation
        lake_bed_thickness: Lake bed thickness
    """

    lake_id: int
    gw_node_id: int
    layer: int = 1
    conductance: float = 0.0
    lake_bed_elev: float = 0.0
    lake_bed_thickness: float = 0.0

    def __repr__(self) -> str:
        return f"LakeGWConnection(lake={self.lake_id}, gw={self.gw_node_id})"


@dataclass
class LakeGWConnector:
    """
    Connector managing lake-groundwater interactions.

    This class manages all connections between lakes and groundwater
    nodes, calculating exchange flows based on head differences.

    Attributes:
        connections: List of lake-groundwater connections
    """

    connections: list[LakeGWConnection] = field(default_factory=list)

    @property
    def n_connections(self) -> int:
        """Return number of connections."""
        return len(self.connections)

    def add_connection(self, conn: LakeGWConnection) -> None:
        """Add a connection."""
        self.connections.append(conn)

    def get_connections_for_lake(self, lake_id: int) -> list[LakeGWConnection]:
        """Get all connections for a lake."""
        return [c for c in self.connections if c.lake_id == lake_id]

    def get_connections_for_gw_node(self, gw_node_id: int) -> list[LakeGWConnection]:
        """Get all connections for a groundwater node."""
        return [c for c in self.connections if c.gw_node_id == gw_node_id]

    def calculate_flow(
        self,
        lake_id: int,
        lake_stage: float,
        gw_head: float,
    ) -> float:
        """
        Calculate lake-aquifer exchange flow.

        Positive flow = lake to aquifer (losing lake)
        Negative flow = aquifer to lake (gaining lake)

        Args:
            lake_id: Lake ID
            lake_stage: Water surface elevation in lake
            gw_head: Groundwater head at connected node

        Returns:
            Exchange flow rate
        """
        total_flow = 0.0

        for conn in self.get_connections_for_lake(lake_id):
            head_diff = lake_stage - gw_head
            flow = conn.conductance * head_diff
            total_flow += flow

        return total_flow

    def calculate_total_exchange(
        self,
        lake_stages: dict[int, float],
        gw_heads: dict[int, float],
    ) -> float:
        """
        Calculate total lake-aquifer exchange.

        Args:
            lake_stages: Dictionary of lake ID to stage
            gw_heads: Dictionary of GW node ID to head

        Returns:
            Total exchange flow (positive = net to aquifer)
        """
        total = 0.0

        for conn in self.connections:
            if conn.lake_id in lake_stages and conn.gw_node_id in gw_heads:
                stage = lake_stages[conn.lake_id]
                head = gw_heads[conn.gw_node_id]
                head_diff = stage - head
                flow = conn.conductance * head_diff
                total += flow

        return total

    def validate(self) -> None:
        """Validate the connector."""
        pass

    def to_arrays(self) -> dict[str, NDArray]:
        """
        Convert connector data to numpy arrays.

        Returns:
            Dictionary of arrays
        """
        if not self.connections:
            return {}

        lake_ids = np.array([c.lake_id for c in self.connections], dtype=np.int32)
        gw_ids = np.array([c.gw_node_id for c in self.connections], dtype=np.int32)
        layers = np.array([c.layer for c in self.connections], dtype=np.int32)
        conductances = np.array([c.conductance for c in self.connections])

        return {
            "lake_ids": lake_ids,
            "gw_node_ids": gw_ids,
            "layers": layers,
            "conductances": conductances,
        }

    def __repr__(self) -> str:
        return f"LakeGWConnector(n_connections={self.n_connections})"


@dataclass
class StreamLakeConnection:
    """
    A connection between a stream node and a lake.

    Attributes:
        stream_node_id: Stream node ID
        lake_id: Lake ID
        connection_type: Type of connection ('inflow' or 'outflow')
        max_flow: Maximum flow rate for outflow connections
    """

    stream_node_id: int
    lake_id: int
    connection_type: str  # 'inflow' or 'outflow'
    max_flow: float = float("inf")

    def __post_init__(self) -> None:
        """Validate connection type."""
        if self.connection_type not in ("inflow", "outflow"):
            raise ValueError("connection_type must be 'inflow' or 'outflow'")

    def __repr__(self) -> str:
        return f"StreamLakeConnection(strm={self.stream_node_id}, lake={self.lake_id}, type={self.connection_type})"


@dataclass
class StreamLakeConnector:
    """
    Connector managing stream-lake interactions.

    This class manages connections where streams flow into or out
    of lakes.

    Attributes:
        connections: List of stream-lake connections
    """

    connections: list[StreamLakeConnection] = field(default_factory=list)

    @property
    def n_connections(self) -> int:
        """Return number of connections."""
        return len(self.connections)

    def add_connection(self, conn: StreamLakeConnection) -> None:
        """Add a connection."""
        self.connections.append(conn)

    def get_connections_for_lake(self, lake_id: int) -> list[StreamLakeConnection]:
        """Get all connections for a lake."""
        return [c for c in self.connections if c.lake_id == lake_id]

    def get_connections_for_stream_node(self, stream_node_id: int) -> list[StreamLakeConnection]:
        """Get all connections for a stream node."""
        return [c for c in self.connections if c.stream_node_id == stream_node_id]

    def get_inflows_for_lake(self, lake_id: int) -> list[StreamLakeConnection]:
        """Get inflow connections for a lake."""
        return [
            c for c in self.connections
            if c.lake_id == lake_id and c.connection_type == "inflow"
        ]

    def get_outflows_for_lake(self, lake_id: int) -> list[StreamLakeConnection]:
        """Get outflow connections for a lake."""
        return [
            c for c in self.connections
            if c.lake_id == lake_id and c.connection_type == "outflow"
        ]

    def validate(self) -> None:
        """Validate the connector."""
        pass

    def to_arrays(self) -> dict[str, NDArray]:
        """
        Convert connector data to numpy arrays.

        Returns:
            Dictionary of arrays
        """
        if not self.connections:
            return {}

        stream_ids = np.array([c.stream_node_id for c in self.connections], dtype=np.int32)
        lake_ids = np.array([c.lake_id for c in self.connections], dtype=np.int32)

        return {
            "stream_node_ids": stream_ids,
            "lake_ids": lake_ids,
        }

    def __repr__(self) -> str:
        return f"StreamLakeConnector(n_connections={self.n_connections})"
