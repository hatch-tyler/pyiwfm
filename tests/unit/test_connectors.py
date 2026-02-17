"""Unit tests for component connector classes."""

from __future__ import annotations

import numpy as np
import pytest

from pyiwfm.components.connectors import (
    LakeGWConnection,
    LakeGWConnector,
    StreamGWConnection,
    StreamGWConnector,
    StreamLakeConnection,
    StreamLakeConnector,
)


class TestStreamGWConnection:
    """Tests for stream-groundwater connection class."""

    def test_connection_creation(self) -> None:
        """Test basic connection creation."""
        conn = StreamGWConnection(
            stream_node_id=10,
            gw_node_id=5,
            layer=1,
            conductance=100.0,
        )

        assert conn.stream_node_id == 10
        assert conn.gw_node_id == 5
        assert conn.layer == 1
        assert conn.conductance == 100.0

    def test_connection_defaults(self) -> None:
        """Test connection default values."""
        conn = StreamGWConnection(
            stream_node_id=10,
            gw_node_id=5,
        )

        assert conn.layer == 1
        assert conn.conductance == 0.0
        assert conn.stream_bed_elev == 0.0

    def test_connection_with_stream_bed(self) -> None:
        """Test connection with stream bed elevation."""
        conn = StreamGWConnection(
            stream_node_id=10,
            gw_node_id=5,
            layer=1,
            conductance=100.0,
            stream_bed_elev=50.0,
            stream_bed_thickness=2.0,
        )

        assert conn.stream_bed_elev == 50.0
        assert conn.stream_bed_thickness == 2.0


class TestStreamGWConnector:
    """Tests for stream-groundwater connector class."""

    def test_connector_creation(self) -> None:
        """Test basic connector creation."""
        connector = StreamGWConnector()

        assert connector.n_connections == 0

    def test_connector_add_connection(self) -> None:
        """Test adding connections."""
        connector = StreamGWConnector()

        conn = StreamGWConnection(stream_node_id=10, gw_node_id=5, conductance=100.0)
        connector.add_connection(conn)

        assert connector.n_connections == 1

    def test_connector_get_connections_for_stream_node(self) -> None:
        """Test getting connections for a stream node."""
        connector = StreamGWConnector()

        connector.add_connection(StreamGWConnection(stream_node_id=10, gw_node_id=5))
        connector.add_connection(StreamGWConnection(stream_node_id=10, gw_node_id=6))
        connector.add_connection(StreamGWConnection(stream_node_id=20, gw_node_id=15))

        conns = connector.get_connections_for_stream_node(10)
        assert len(conns) == 2

    def test_connector_get_connections_for_gw_node(self) -> None:
        """Test getting connections for a groundwater node."""
        connector = StreamGWConnector()

        connector.add_connection(StreamGWConnection(stream_node_id=10, gw_node_id=5))
        connector.add_connection(StreamGWConnection(stream_node_id=20, gw_node_id=5))
        connector.add_connection(StreamGWConnection(stream_node_id=30, gw_node_id=15))

        conns = connector.get_connections_for_gw_node(5)
        assert len(conns) == 2

    def test_connector_calculate_flow(self) -> None:
        """Test calculating stream-aquifer flow."""
        connector = StreamGWConnector()

        conn = StreamGWConnection(
            stream_node_id=10,
            gw_node_id=5,
            conductance=100.0,
            stream_bed_elev=50.0,
        )
        connector.add_connection(conn)

        # Stream stage above GW head -> gaining stream (negative flow = into aquifer)
        flow = connector.calculate_flow(
            stream_node_id=10,
            stream_stage=55.0,
            gw_head=45.0,
        )
        assert flow > 0  # Positive = flow from stream to aquifer

        # GW head above stream stage -> losing aquifer (positive flow = into stream)
        flow = connector.calculate_flow(
            stream_node_id=10,
            stream_stage=45.0,
            gw_head=55.0,
        )
        assert flow < 0  # Negative = flow from aquifer to stream

    def test_connector_total_exchange(self) -> None:
        """Test calculating total exchange."""
        connector = StreamGWConnector()

        connector.add_connection(
            StreamGWConnection(
                stream_node_id=10, gw_node_id=5, conductance=100.0, stream_bed_elev=50.0
            )
        )
        connector.add_connection(
            StreamGWConnection(
                stream_node_id=20, gw_node_id=15, conductance=100.0, stream_bed_elev=50.0
            )
        )

        stream_stages = {10: 55.0, 20: 55.0}
        gw_heads = {5: 45.0, 15: 45.0}

        total = connector.calculate_total_exchange(stream_stages, gw_heads)
        assert total > 0  # Net flow to aquifer

    def test_connector_validate(self) -> None:
        """Test connector validation."""
        connector = StreamGWConnector()

        # Empty should pass
        connector.validate()

    def test_connector_to_arrays(self) -> None:
        """Test converting to arrays."""
        connector = StreamGWConnector()

        connector.add_connection(
            StreamGWConnection(stream_node_id=10, gw_node_id=5, conductance=100.0)
        )
        connector.add_connection(
            StreamGWConnection(stream_node_id=20, gw_node_id=15, conductance=200.0)
        )

        arrays = connector.to_arrays()

        assert "stream_node_ids" in arrays
        assert "gw_node_ids" in arrays
        assert "conductances" in arrays
        np.testing.assert_array_equal(arrays["stream_node_ids"], [10, 20])
        np.testing.assert_array_equal(arrays["conductances"], [100.0, 200.0])


class TestLakeGWConnection:
    """Tests for lake-groundwater connection class."""

    def test_connection_creation(self) -> None:
        """Test basic connection creation."""
        conn = LakeGWConnection(
            lake_id=1,
            gw_node_id=5,
            layer=1,
            conductance=500.0,
        )

        assert conn.lake_id == 1
        assert conn.gw_node_id == 5
        assert conn.layer == 1
        assert conn.conductance == 500.0

    def test_connection_with_lake_bed(self) -> None:
        """Test connection with lake bed parameters."""
        conn = LakeGWConnection(
            lake_id=1,
            gw_node_id=5,
            conductance=500.0,
            lake_bed_elev=100.0,
            lake_bed_thickness=1.0,
        )

        assert conn.lake_bed_elev == 100.0
        assert conn.lake_bed_thickness == 1.0


class TestLakeGWConnector:
    """Tests for lake-groundwater connector class."""

    def test_connector_creation(self) -> None:
        """Test basic connector creation."""
        connector = LakeGWConnector()

        assert connector.n_connections == 0

    def test_connector_add_connection(self) -> None:
        """Test adding connections."""
        connector = LakeGWConnector()

        conn = LakeGWConnection(lake_id=1, gw_node_id=5, conductance=500.0)
        connector.add_connection(conn)

        assert connector.n_connections == 1

    def test_connector_get_connections_for_lake(self) -> None:
        """Test getting connections for a lake."""
        connector = LakeGWConnector()

        connector.add_connection(LakeGWConnection(lake_id=1, gw_node_id=5))
        connector.add_connection(LakeGWConnection(lake_id=1, gw_node_id=6))
        connector.add_connection(LakeGWConnection(lake_id=1, gw_node_id=7))
        connector.add_connection(LakeGWConnection(lake_id=2, gw_node_id=20))

        conns = connector.get_connections_for_lake(1)
        assert len(conns) == 3

    def test_connector_calculate_flow(self) -> None:
        """Test calculating lake-aquifer flow."""
        connector = LakeGWConnector()

        conn = LakeGWConnection(
            lake_id=1,
            gw_node_id=5,
            conductance=500.0,
            lake_bed_elev=100.0,
        )
        connector.add_connection(conn)

        # Lake stage above GW head -> flow into aquifer
        flow = connector.calculate_flow(
            lake_id=1,
            lake_stage=110.0,
            gw_head=95.0,
        )
        assert flow > 0  # Positive = flow from lake to aquifer

    def test_connector_validate(self) -> None:
        """Test connector validation."""
        connector = LakeGWConnector()

        connector.validate()


class TestStreamLakeConnection:
    """Tests for stream-lake connection class."""

    def test_connection_creation(self) -> None:
        """Test basic connection creation."""
        conn = StreamLakeConnection(
            stream_node_id=10,
            lake_id=1,
            connection_type="inflow",
        )

        assert conn.stream_node_id == 10
        assert conn.lake_id == 1
        assert conn.connection_type == "inflow"

    def test_connection_outflow(self) -> None:
        """Test outflow connection."""
        conn = StreamLakeConnection(
            stream_node_id=50,
            lake_id=1,
            connection_type="outflow",
            max_flow=1000.0,
        )

        assert conn.connection_type == "outflow"
        assert conn.max_flow == 1000.0


class TestStreamLakeConnector:
    """Tests for stream-lake connector class."""

    def test_connector_creation(self) -> None:
        """Test basic connector creation."""
        connector = StreamLakeConnector()

        assert connector.n_connections == 0

    def test_connector_add_connection(self) -> None:
        """Test adding connections."""
        connector = StreamLakeConnector()

        conn = StreamLakeConnection(stream_node_id=10, lake_id=1, connection_type="inflow")
        connector.add_connection(conn)

        assert connector.n_connections == 1

    def test_connector_get_inflows_for_lake(self) -> None:
        """Test getting inflow connections for a lake."""
        connector = StreamLakeConnector()

        connector.add_connection(
            StreamLakeConnection(stream_node_id=10, lake_id=1, connection_type="inflow")
        )
        connector.add_connection(
            StreamLakeConnection(stream_node_id=20, lake_id=1, connection_type="inflow")
        )
        connector.add_connection(
            StreamLakeConnection(stream_node_id=50, lake_id=1, connection_type="outflow")
        )

        inflows = connector.get_inflows_for_lake(1)
        assert len(inflows) == 2
        assert all(c.connection_type == "inflow" for c in inflows)

    def test_connector_get_outflows_for_lake(self) -> None:
        """Test getting outflow connections for a lake."""
        connector = StreamLakeConnector()

        connector.add_connection(
            StreamLakeConnection(stream_node_id=10, lake_id=1, connection_type="inflow")
        )
        connector.add_connection(
            StreamLakeConnection(stream_node_id=50, lake_id=1, connection_type="outflow")
        )

        outflows = connector.get_outflows_for_lake(1)
        assert len(outflows) == 1
        assert outflows[0].connection_type == "outflow"

    def test_connector_validate(self) -> None:
        """Test connector validation."""
        connector = StreamLakeConnector()

        connector.validate()


# ---------------------------------------------------------------------------
# Additional tests for expanded coverage
# ---------------------------------------------------------------------------


class TestStreamGWConnectionRepr:
    """Tests for StreamGWConnection __repr__."""

    def test_repr(self) -> None:
        conn = StreamGWConnection(stream_node_id=10, gw_node_id=5)
        result = repr(conn)
        assert "StreamGWConnection" in result
        assert "strm=10" in result
        assert "gw=5" in result

    def test_stream_bed_thickness_default(self) -> None:
        """Verify the default stream_bed_thickness is 0.0."""
        conn = StreamGWConnection(stream_node_id=1, gw_node_id=2)
        assert conn.stream_bed_thickness == 0.0


class TestStreamGWConnectorRepr:
    """Tests for StreamGWConnector __repr__."""

    def test_repr_empty(self) -> None:
        connector = StreamGWConnector()
        result = repr(connector)
        assert "StreamGWConnector" in result
        assert "n_connections=0" in result

    def test_repr_with_connections(self) -> None:
        connector = StreamGWConnector()
        connector.add_connection(StreamGWConnection(stream_node_id=1, gw_node_id=2))
        result = repr(connector)
        assert "n_connections=1" in result


class TestStreamGWConnectorEdgeCases:
    """Edge-case tests for StreamGWConnector."""

    def test_calculate_flow_no_matching_connections(self) -> None:
        """calculate_flow returns 0 when no connections match the stream_node_id."""
        connector = StreamGWConnector()
        connector.add_connection(
            StreamGWConnection(stream_node_id=10, gw_node_id=5, conductance=100.0)
        )
        flow = connector.calculate_flow(stream_node_id=999, stream_stage=55.0, gw_head=45.0)
        assert flow == 0.0

    def test_calculate_flow_zero_conductance(self) -> None:
        """Zero conductance produces zero flow regardless of head difference."""
        connector = StreamGWConnector()
        connector.add_connection(
            StreamGWConnection(stream_node_id=1, gw_node_id=2, conductance=0.0)
        )
        flow = connector.calculate_flow(stream_node_id=1, stream_stage=100.0, gw_head=50.0)
        assert flow == 0.0

    def test_calculate_flow_equal_heads(self) -> None:
        """Equal heads produce zero flow."""
        connector = StreamGWConnector()
        connector.add_connection(
            StreamGWConnection(stream_node_id=1, gw_node_id=2, conductance=200.0)
        )
        flow = connector.calculate_flow(stream_node_id=1, stream_stage=50.0, gw_head=50.0)
        assert flow == 0.0

    def test_calculate_flow_multiple_connections_same_stream_node(self) -> None:
        """Flow is summed across multiple connections for the same stream node."""
        connector = StreamGWConnector()
        connector.add_connection(
            StreamGWConnection(stream_node_id=1, gw_node_id=2, conductance=100.0)
        )
        connector.add_connection(
            StreamGWConnection(stream_node_id=1, gw_node_id=3, conductance=150.0)
        )
        # head_diff = 60 - 50 = 10; total = (100 + 150) * 10 = 2500
        flow = connector.calculate_flow(stream_node_id=1, stream_stage=60.0, gw_head=50.0)
        assert flow == pytest.approx(2500.0)

    def test_calculate_total_exchange_partial_keys(self) -> None:
        """Connections whose IDs are missing from the dicts are skipped."""
        connector = StreamGWConnector()
        connector.add_connection(
            StreamGWConnection(stream_node_id=1, gw_node_id=2, conductance=100.0)
        )
        connector.add_connection(
            StreamGWConnection(stream_node_id=3, gw_node_id=4, conductance=200.0)
        )

        # Only supply keys for the first connection
        total = connector.calculate_total_exchange(
            stream_stages={1: 60.0},
            gw_heads={2: 50.0},
        )
        assert total == pytest.approx(1000.0)

    def test_calculate_total_exchange_empty_dicts(self) -> None:
        """Empty stage/head dicts yield zero total."""
        connector = StreamGWConnector()
        connector.add_connection(
            StreamGWConnection(stream_node_id=1, gw_node_id=2, conductance=100.0)
        )
        total = connector.calculate_total_exchange(stream_stages={}, gw_heads={})
        assert total == 0.0

    def test_calculate_total_exchange_no_connections(self) -> None:
        """An empty connector returns zero total exchange."""
        connector = StreamGWConnector()
        total = connector.calculate_total_exchange(stream_stages={1: 60.0}, gw_heads={2: 50.0})
        assert total == 0.0

    def test_to_arrays_empty(self) -> None:
        """to_arrays on empty connector returns empty dict."""
        connector = StreamGWConnector()
        arrays = connector.to_arrays()
        assert arrays == {}

    def test_to_arrays_layers(self) -> None:
        """to_arrays includes 'layers' key with correct values."""
        connector = StreamGWConnector()
        connector.add_connection(
            StreamGWConnection(stream_node_id=1, gw_node_id=2, layer=3, conductance=10.0)
        )
        arrays = connector.to_arrays()
        assert "layers" in arrays
        np.testing.assert_array_equal(arrays["layers"], [3])

    def test_get_connections_for_stream_node_no_match(self) -> None:
        """Returns empty list when no connections match."""
        connector = StreamGWConnector()
        connector.add_connection(StreamGWConnection(stream_node_id=1, gw_node_id=2))
        assert connector.get_connections_for_stream_node(999) == []

    def test_get_connections_for_gw_node_no_match(self) -> None:
        """Returns empty list when no connections match."""
        connector = StreamGWConnector()
        connector.add_connection(StreamGWConnection(stream_node_id=1, gw_node_id=2))
        assert connector.get_connections_for_gw_node(999) == []


class TestLakeGWConnectionExtended:
    """Extended tests for LakeGWConnection."""

    def test_defaults(self) -> None:
        """Verify all default values."""
        conn = LakeGWConnection(lake_id=1, gw_node_id=2)
        assert conn.layer == 1
        assert conn.conductance == 0.0
        assert conn.lake_bed_elev == 0.0
        assert conn.lake_bed_thickness == 0.0

    def test_repr(self) -> None:
        conn = LakeGWConnection(lake_id=3, gw_node_id=7)
        result = repr(conn)
        assert "LakeGWConnection" in result
        assert "lake=3" in result
        assert "gw=7" in result


class TestLakeGWConnectorExtended:
    """Extended tests for LakeGWConnector."""

    def test_repr_empty(self) -> None:
        connector = LakeGWConnector()
        result = repr(connector)
        assert "LakeGWConnector" in result
        assert "n_connections=0" in result

    def test_repr_with_connections(self) -> None:
        connector = LakeGWConnector()
        connector.add_connection(LakeGWConnection(lake_id=1, gw_node_id=2))
        connector.add_connection(LakeGWConnection(lake_id=1, gw_node_id=3))
        result = repr(connector)
        assert "n_connections=2" in result

    def test_get_connections_for_gw_node(self) -> None:
        """Test getting connections for a groundwater node."""
        connector = LakeGWConnector()
        connector.add_connection(LakeGWConnection(lake_id=1, gw_node_id=5))
        connector.add_connection(LakeGWConnection(lake_id=2, gw_node_id=5))
        connector.add_connection(LakeGWConnection(lake_id=3, gw_node_id=10))

        conns = connector.get_connections_for_gw_node(5)
        assert len(conns) == 2
        assert all(c.gw_node_id == 5 for c in conns)

    def test_get_connections_for_gw_node_no_match(self) -> None:
        connector = LakeGWConnector()
        connector.add_connection(LakeGWConnection(lake_id=1, gw_node_id=5))
        assert connector.get_connections_for_gw_node(999) == []

    def test_get_connections_for_lake_no_match(self) -> None:
        connector = LakeGWConnector()
        connector.add_connection(LakeGWConnection(lake_id=1, gw_node_id=5))
        assert connector.get_connections_for_lake(999) == []

    def test_calculate_flow_gaining_lake(self) -> None:
        """GW head above lake stage produces negative (gaining) flow."""
        connector = LakeGWConnector()
        connector.add_connection(LakeGWConnection(lake_id=1, gw_node_id=5, conductance=500.0))
        flow = connector.calculate_flow(lake_id=1, lake_stage=90.0, gw_head=100.0)
        assert flow < 0  # Negative = aquifer to lake

    def test_calculate_flow_no_matching_connections(self) -> None:
        """No matching lake_id gives zero flow."""
        connector = LakeGWConnector()
        connector.add_connection(LakeGWConnection(lake_id=1, gw_node_id=5, conductance=500.0))
        flow = connector.calculate_flow(lake_id=999, lake_stage=110.0, gw_head=95.0)
        assert flow == 0.0

    def test_calculate_flow_zero_conductance(self) -> None:
        connector = LakeGWConnector()
        connector.add_connection(LakeGWConnection(lake_id=1, gw_node_id=5, conductance=0.0))
        flow = connector.calculate_flow(lake_id=1, lake_stage=110.0, gw_head=95.0)
        assert flow == 0.0

    def test_calculate_flow_multiple_connections_same_lake(self) -> None:
        """Flow is summed across multiple connections for the same lake."""
        connector = LakeGWConnector()
        connector.add_connection(LakeGWConnection(lake_id=1, gw_node_id=5, conductance=100.0))
        connector.add_connection(LakeGWConnection(lake_id=1, gw_node_id=6, conductance=200.0))
        # head_diff = 110 - 100 = 10; total = (100 + 200) * 10 = 3000
        flow = connector.calculate_flow(lake_id=1, lake_stage=110.0, gw_head=100.0)
        assert flow == pytest.approx(3000.0)

    def test_calculate_total_exchange(self) -> None:
        """Test total exchange calculation for lake-GW connector."""
        connector = LakeGWConnector()
        connector.add_connection(LakeGWConnection(lake_id=1, gw_node_id=5, conductance=100.0))
        connector.add_connection(LakeGWConnection(lake_id=2, gw_node_id=10, conductance=200.0))

        lake_stages = {1: 110.0, 2: 105.0}
        gw_heads = {5: 100.0, 10: 95.0}

        # conn1: 100 * (110-100) = 1000
        # conn2: 200 * (105-95)  = 2000
        total = connector.calculate_total_exchange(lake_stages, gw_heads)
        assert total == pytest.approx(3000.0)

    def test_calculate_total_exchange_partial_keys(self) -> None:
        """Missing keys skip those connections."""
        connector = LakeGWConnector()
        connector.add_connection(LakeGWConnection(lake_id=1, gw_node_id=5, conductance=100.0))
        connector.add_connection(LakeGWConnection(lake_id=2, gw_node_id=10, conductance=200.0))
        total = connector.calculate_total_exchange(lake_stages={1: 110.0}, gw_heads={5: 100.0})
        assert total == pytest.approx(1000.0)

    def test_calculate_total_exchange_empty(self) -> None:
        connector = LakeGWConnector()
        total = connector.calculate_total_exchange(lake_stages={}, gw_heads={})
        assert total == 0.0

    def test_to_arrays(self) -> None:
        """Test converting lake-GW connector to arrays."""
        connector = LakeGWConnector()
        connector.add_connection(
            LakeGWConnection(lake_id=1, gw_node_id=5, layer=2, conductance=100.0)
        )
        connector.add_connection(
            LakeGWConnection(lake_id=2, gw_node_id=10, layer=1, conductance=200.0)
        )

        arrays = connector.to_arrays()
        assert "lake_ids" in arrays
        assert "gw_node_ids" in arrays
        assert "layers" in arrays
        assert "conductances" in arrays
        np.testing.assert_array_equal(arrays["lake_ids"], [1, 2])
        np.testing.assert_array_equal(arrays["gw_node_ids"], [5, 10])
        np.testing.assert_array_equal(arrays["layers"], [2, 1])
        np.testing.assert_array_equal(arrays["conductances"], [100.0, 200.0])

    def test_to_arrays_empty(self) -> None:
        connector = LakeGWConnector()
        arrays = connector.to_arrays()
        assert arrays == {}


class TestStreamLakeConnectionExtended:
    """Extended tests for StreamLakeConnection."""

    def test_repr_inflow(self) -> None:
        conn = StreamLakeConnection(stream_node_id=10, lake_id=1, connection_type="inflow")
        result = repr(conn)
        assert "StreamLakeConnection" in result
        assert "strm=10" in result
        assert "lake=1" in result
        assert "type=inflow" in result

    def test_repr_outflow(self) -> None:
        conn = StreamLakeConnection(stream_node_id=50, lake_id=2, connection_type="outflow")
        result = repr(conn)
        assert "type=outflow" in result

    def test_invalid_connection_type(self) -> None:
        """__post_init__ raises ValueError for invalid connection_type."""
        with pytest.raises(ValueError, match="connection_type must be 'inflow' or 'outflow'"):
            StreamLakeConnection(stream_node_id=10, lake_id=1, connection_type="invalid")

    def test_invalid_connection_type_empty(self) -> None:
        """Empty string is also invalid."""
        with pytest.raises(ValueError, match="connection_type must be 'inflow' or 'outflow'"):
            StreamLakeConnection(stream_node_id=10, lake_id=1, connection_type="")

    def test_default_max_flow(self) -> None:
        """Default max_flow is infinity."""
        conn = StreamLakeConnection(stream_node_id=10, lake_id=1, connection_type="inflow")
        assert conn.max_flow == float("inf")


class TestStreamLakeConnectorExtended:
    """Extended tests for StreamLakeConnector."""

    def test_repr_empty(self) -> None:
        connector = StreamLakeConnector()
        result = repr(connector)
        assert "StreamLakeConnector" in result
        assert "n_connections=0" in result

    def test_repr_with_connections(self) -> None:
        connector = StreamLakeConnector()
        connector.add_connection(
            StreamLakeConnection(stream_node_id=10, lake_id=1, connection_type="inflow")
        )
        result = repr(connector)
        assert "n_connections=1" in result

    def test_get_connections_for_lake(self) -> None:
        """Test getting all connections (inflow and outflow) for a lake."""
        connector = StreamLakeConnector()
        connector.add_connection(
            StreamLakeConnection(stream_node_id=10, lake_id=1, connection_type="inflow")
        )
        connector.add_connection(
            StreamLakeConnection(stream_node_id=20, lake_id=1, connection_type="outflow")
        )
        connector.add_connection(
            StreamLakeConnection(stream_node_id=30, lake_id=2, connection_type="inflow")
        )

        conns = connector.get_connections_for_lake(1)
        assert len(conns) == 2
        assert all(c.lake_id == 1 for c in conns)

    def test_get_connections_for_lake_no_match(self) -> None:
        connector = StreamLakeConnector()
        connector.add_connection(
            StreamLakeConnection(stream_node_id=10, lake_id=1, connection_type="inflow")
        )
        assert connector.get_connections_for_lake(999) == []

    def test_get_connections_for_stream_node(self) -> None:
        """Test getting connections for a stream node."""
        connector = StreamLakeConnector()
        connector.add_connection(
            StreamLakeConnection(stream_node_id=10, lake_id=1, connection_type="inflow")
        )
        connector.add_connection(
            StreamLakeConnection(stream_node_id=10, lake_id=2, connection_type="outflow")
        )
        connector.add_connection(
            StreamLakeConnection(stream_node_id=20, lake_id=1, connection_type="inflow")
        )

        conns = connector.get_connections_for_stream_node(10)
        assert len(conns) == 2
        assert all(c.stream_node_id == 10 for c in conns)

    def test_get_connections_for_stream_node_no_match(self) -> None:
        connector = StreamLakeConnector()
        connector.add_connection(
            StreamLakeConnection(stream_node_id=10, lake_id=1, connection_type="inflow")
        )
        assert connector.get_connections_for_stream_node(999) == []

    def test_get_inflows_for_lake_no_match(self) -> None:
        """Lake with only outflows returns empty for inflows query."""
        connector = StreamLakeConnector()
        connector.add_connection(
            StreamLakeConnection(stream_node_id=50, lake_id=1, connection_type="outflow")
        )
        assert connector.get_inflows_for_lake(1) == []

    def test_get_outflows_for_lake_no_match(self) -> None:
        """Lake with only inflows returns empty for outflows query."""
        connector = StreamLakeConnector()
        connector.add_connection(
            StreamLakeConnection(stream_node_id=10, lake_id=1, connection_type="inflow")
        )
        assert connector.get_outflows_for_lake(1) == []

    def test_get_inflows_for_nonexistent_lake(self) -> None:
        connector = StreamLakeConnector()
        assert connector.get_inflows_for_lake(999) == []

    def test_get_outflows_for_nonexistent_lake(self) -> None:
        connector = StreamLakeConnector()
        assert connector.get_outflows_for_lake(999) == []

    def test_to_arrays(self) -> None:
        """Test converting stream-lake connector to arrays."""
        connector = StreamLakeConnector()
        connector.add_connection(
            StreamLakeConnection(stream_node_id=10, lake_id=1, connection_type="inflow")
        )
        connector.add_connection(
            StreamLakeConnection(stream_node_id=50, lake_id=2, connection_type="outflow")
        )

        arrays = connector.to_arrays()
        assert "stream_node_ids" in arrays
        assert "lake_ids" in arrays
        np.testing.assert_array_equal(arrays["stream_node_ids"], [10, 50])
        np.testing.assert_array_equal(arrays["lake_ids"], [1, 2])

    def test_to_arrays_empty(self) -> None:
        connector = StreamLakeConnector()
        arrays = connector.to_arrays()
        assert arrays == {}

    def test_to_arrays_single_connection(self) -> None:
        connector = StreamLakeConnector()
        connector.add_connection(
            StreamLakeConnection(stream_node_id=5, lake_id=3, connection_type="inflow")
        )
        arrays = connector.to_arrays()
        np.testing.assert_array_equal(arrays["stream_node_ids"], [5])
        np.testing.assert_array_equal(arrays["lake_ids"], [3])
