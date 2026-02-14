"""Unit tests for stream network classes."""

from __future__ import annotations

import numpy as np
import pytest

from pyiwfm.components.stream import (
    StrmNode,
    StrmReach,
    Diversion,
    Bypass,
    AppStream,
    StreamRating,
)
from pyiwfm.core.exceptions import ComponentError


class TestStrmNode:
    """Tests for stream node class."""

    def test_strmnode_creation(self) -> None:
        """Test basic stream node creation."""
        node = StrmNode(
            id=1,
            x=1000.0,
            y=2000.0,
            reach_id=1,
            gw_node=5,
        )

        assert node.id == 1
        assert node.x == 1000.0
        assert node.y == 2000.0
        assert node.reach_id == 1
        assert node.gw_node == 5

    def test_strmnode_default_values(self) -> None:
        """Test stream node default values."""
        node = StrmNode(id=1, x=0.0, y=0.0)

        assert node.reach_id == 0
        assert node.gw_node is None
        assert node.bottom_elev == 0.0
        assert node.wetted_perimeter == 0.0

    def test_strmnode_with_rating(self) -> None:
        """Test stream node with rating curve."""
        rating = StreamRating(
            stages=np.array([0.0, 1.0, 2.0]),
            flows=np.array([0.0, 10.0, 50.0]),
        )
        node = StrmNode(id=1, x=0.0, y=0.0, rating=rating)

        assert node.rating is not None
        assert len(node.rating.stages) == 3

    def test_strmnode_upstream_downstream(self) -> None:
        """Test upstream/downstream node references."""
        node = StrmNode(
            id=2,
            x=0.0,
            y=0.0,
            upstream_node=1,
            downstream_node=3,
        )

        assert node.upstream_node == 1
        assert node.downstream_node == 3

    def test_strmnode_equality(self) -> None:
        """Test stream node equality."""
        node1 = StrmNode(id=1, x=100.0, y=200.0)
        node2 = StrmNode(id=1, x=100.0, y=200.0)
        node3 = StrmNode(id=2, x=100.0, y=200.0)

        assert node1 == node2
        assert node1 != node3


class TestStreamRating:
    """Tests for stream rating curve."""

    def test_rating_creation(self) -> None:
        """Test rating curve creation."""
        rating = StreamRating(
            stages=np.array([0.0, 1.0, 2.0, 3.0]),
            flows=np.array([0.0, 5.0, 20.0, 50.0]),
        )

        assert len(rating.stages) == 4
        assert len(rating.flows) == 4

    def test_rating_interpolate_flow(self) -> None:
        """Test flow interpolation from stage."""
        rating = StreamRating(
            stages=np.array([0.0, 1.0, 2.0]),
            flows=np.array([0.0, 10.0, 40.0]),
        )

        # Exact match
        assert rating.get_flow(1.0) == pytest.approx(10.0)

        # Interpolation
        assert rating.get_flow(0.5) == pytest.approx(5.0)
        assert rating.get_flow(1.5) == pytest.approx(25.0)

    def test_rating_interpolate_stage(self) -> None:
        """Test stage interpolation from flow."""
        rating = StreamRating(
            stages=np.array([0.0, 1.0, 2.0]),
            flows=np.array([0.0, 10.0, 40.0]),
        )

        assert rating.get_stage(10.0) == pytest.approx(1.0)
        assert rating.get_stage(5.0) == pytest.approx(0.5)

    def test_rating_extrapolation(self) -> None:
        """Test extrapolation beyond rating curve."""
        rating = StreamRating(
            stages=np.array([0.0, 1.0, 2.0]),
            flows=np.array([0.0, 10.0, 40.0]),
        )

        # Below minimum - should return 0
        assert rating.get_flow(-1.0) == pytest.approx(0.0)

        # Above maximum - linear extrapolation
        flow_above = rating.get_flow(3.0)
        assert flow_above > 40.0


class TestStrmReach:
    """Tests for stream reach class."""

    def test_reach_creation(self) -> None:
        """Test basic reach creation."""
        reach = StrmReach(
            id=1,
            name="Main River",
            upstream_node=1,
            downstream_node=10,
        )

        assert reach.id == 1
        assert reach.name == "Main River"
        assert reach.upstream_node == 1
        assert reach.downstream_node == 10

    def test_reach_with_nodes(self) -> None:
        """Test reach with node list."""
        nodes = [1, 2, 3, 4, 5]
        reach = StrmReach(
            id=1,
            name="Test Reach",
            upstream_node=1,
            downstream_node=5,
            nodes=nodes,
        )

        assert reach.n_nodes == 5
        assert reach.nodes == [1, 2, 3, 4, 5]

    def test_reach_default_values(self) -> None:
        """Test reach default values."""
        reach = StrmReach(id=1, upstream_node=1, downstream_node=5)

        assert reach.name == ""
        assert reach.outflow_destination is None
        assert reach.n_nodes == 0

    def test_reach_outflow_types(self) -> None:
        """Test different outflow destination types."""
        # Lake outflow
        reach1 = StrmReach(
            id=1,
            upstream_node=1,
            downstream_node=5,
            outflow_destination=("lake", 1),
        )
        assert reach1.outflow_destination == ("lake", 1)

        # Another reach
        reach2 = StrmReach(
            id=2,
            upstream_node=6,
            downstream_node=10,
            outflow_destination=("reach", 3),
        )
        assert reach2.outflow_destination == ("reach", 3)


class TestDiversion:
    """Tests for stream diversion class."""

    def test_diversion_creation(self) -> None:
        """Test basic diversion creation."""
        div = Diversion(
            id=1,
            name="Irrigation Canal",
            source_node=5,
            destination_type="element",
            destination_id=10,
            max_rate=100.0,
        )

        assert div.id == 1
        assert div.name == "Irrigation Canal"
        assert div.source_node == 5
        assert div.max_rate == 100.0

    def test_diversion_destinations(self) -> None:
        """Test different diversion destinations."""
        # To element (agricultural use)
        div1 = Diversion(
            id=1,
            source_node=5,
            destination_type="element",
            destination_id=10,
        )
        assert div1.destination_type == "element"

        # To another stream node
        div2 = Diversion(
            id=2,
            source_node=5,
            destination_type="stream_node",
            destination_id=20,
        )
        assert div2.destination_type == "stream_node"

    def test_diversion_priority(self) -> None:
        """Test diversion priority for allocation."""
        div = Diversion(
            id=1,
            source_node=5,
            destination_type="element",
            destination_id=10,
            priority=1,  # Higher priority
        )

        assert div.priority == 1


class TestBypass:
    """Tests for stream bypass class."""

    def test_bypass_creation(self) -> None:
        """Test basic bypass creation."""
        bypass = Bypass(
            id=1,
            name="Flood Bypass",
            source_node=10,
            destination_node=50,
            capacity=1000.0,
        )

        assert bypass.id == 1
        assert bypass.name == "Flood Bypass"
        assert bypass.source_node == 10
        assert bypass.destination_node == 50
        assert bypass.capacity == 1000.0

    def test_bypass_default_capacity(self) -> None:
        """Test bypass with unlimited capacity."""
        bypass = Bypass(
            id=1,
            source_node=10,
            destination_node=50,
        )

        assert bypass.capacity == float("inf")


class TestAppStream:
    """Tests for stream network application class."""

    def test_appstream_creation(self) -> None:
        """Test basic stream network creation."""
        stream = AppStream()

        assert stream.n_nodes == 0
        assert stream.n_reaches == 0

    def test_appstream_add_node(self) -> None:
        """Test adding nodes to stream network."""
        stream = AppStream()

        node1 = StrmNode(id=1, x=0.0, y=0.0, reach_id=1)
        node2 = StrmNode(id=2, x=100.0, y=0.0, reach_id=1)

        stream.add_node(node1)
        stream.add_node(node2)

        assert stream.n_nodes == 2
        assert stream.get_node(1) == node1

    def test_appstream_add_reach(self) -> None:
        """Test adding reaches to stream network."""
        stream = AppStream()

        reach = StrmReach(
            id=1,
            name="Main Channel",
            upstream_node=1,
            downstream_node=5,
            nodes=[1, 2, 3, 4, 5],
        )

        stream.add_reach(reach)

        assert stream.n_reaches == 1
        assert stream.get_reach(1) == reach

    def test_appstream_build_from_data(self) -> None:
        """Test building stream network from data arrays."""
        # Node data
        node_ids = [1, 2, 3, 4, 5]
        x = [0.0, 100.0, 200.0, 300.0, 400.0]
        y = [0.0, 0.0, 0.0, 0.0, 0.0]
        reach_ids = [1, 1, 1, 1, 1]
        gw_nodes = [1, 2, 3, 4, 5]

        stream = AppStream.from_arrays(
            node_ids=np.array(node_ids),
            x=np.array(x),
            y=np.array(y),
            reach_ids=np.array(reach_ids),
            gw_nodes=np.array(gw_nodes),
        )

        assert stream.n_nodes == 5

    def test_appstream_connectivity(self) -> None:
        """Test stream network connectivity."""
        stream = AppStream()

        # Add nodes
        for i in range(1, 6):
            stream.add_node(StrmNode(
                id=i,
                x=float(i * 100),
                y=0.0,
                reach_id=1,
                upstream_node=i - 1 if i > 1 else None,
                downstream_node=i + 1 if i < 5 else None,
            ))

        # Add reach
        stream.add_reach(StrmReach(
            id=1,
            upstream_node=1,
            downstream_node=5,
            nodes=[1, 2, 3, 4, 5],
        ))

        stream.build_connectivity()

        # Check connectivity
        assert stream.get_downstream_node(1) == 2
        assert stream.get_downstream_node(4) == 5
        assert stream.get_upstream_node(5) == 4

    def test_appstream_get_reach_length(self) -> None:
        """Test calculating reach length."""
        stream = AppStream()

        # Add nodes along a straight line
        for i in range(1, 6):
            stream.add_node(StrmNode(
                id=i,
                x=float(i * 100),
                y=0.0,
                reach_id=1,
            ))

        stream.add_reach(StrmReach(
            id=1,
            upstream_node=1,
            downstream_node=5,
            nodes=[1, 2, 3, 4, 5],
        ))

        length = stream.get_reach_length(1)
        assert length == pytest.approx(400.0)  # 4 segments * 100 units

    def test_appstream_get_total_length(self) -> None:
        """Test total stream length calculation."""
        stream = AppStream()

        # Add nodes
        for i in range(1, 11):
            stream.add_node(StrmNode(id=i, x=float(i * 100), y=0.0, reach_id=1 if i <= 5 else 2))

        stream.add_reach(StrmReach(id=1, upstream_node=1, downstream_node=5, nodes=[1, 2, 3, 4, 5]))
        stream.add_reach(StrmReach(id=2, upstream_node=6, downstream_node=10, nodes=[6, 7, 8, 9, 10]))

        total = stream.get_total_length()
        assert total == pytest.approx(800.0)  # 2 reaches * 400 units

    def test_appstream_diversions(self) -> None:
        """Test managing diversions."""
        stream = AppStream()

        div = Diversion(
            id=1,
            source_node=5,
            destination_type="element",
            destination_id=10,
            max_rate=100.0,
        )

        stream.add_diversion(div)

        assert stream.n_diversions == 1
        assert stream.get_diversion(1) == div

    def test_appstream_bypasses(self) -> None:
        """Test managing bypasses."""
        stream = AppStream()

        bypass = Bypass(
            id=1,
            source_node=10,
            destination_node=50,
            capacity=1000.0,
        )

        stream.add_bypass(bypass)

        assert stream.n_bypasses == 1
        assert stream.get_bypass(1) == bypass

    def test_appstream_get_nodes_in_reach(self) -> None:
        """Test getting nodes in a reach."""
        stream = AppStream()

        for i in range(1, 6):
            stream.add_node(StrmNode(id=i, x=float(i * 100), y=0.0, reach_id=1))

        stream.add_reach(StrmReach(
            id=1,
            upstream_node=1,
            downstream_node=5,
            nodes=[1, 2, 3, 4, 5],
        ))

        nodes = stream.get_nodes_in_reach(1)
        assert len(nodes) == 5
        assert [n.id for n in nodes] == [1, 2, 3, 4, 5]

    def test_appstream_validate(self) -> None:
        """Test stream network validation."""
        stream = AppStream()

        # Empty stream should fail
        with pytest.raises(ComponentError, match="no nodes"):
            stream.validate()

    def test_appstream_validate_connectivity(self) -> None:
        """Test connectivity validation."""
        stream = AppStream()

        # Add nodes without proper connectivity
        stream.add_node(StrmNode(id=1, x=0.0, y=0.0, reach_id=1))
        stream.add_node(StrmNode(id=2, x=100.0, y=0.0, reach_id=1))

        stream.add_reach(StrmReach(
            id=1,
            upstream_node=1,
            downstream_node=2,
            nodes=[1, 2],
        ))

        # Should pass basic validation
        stream.validate()


class TestAppStreamIO:
    """Tests for stream network I/O."""

    def test_appstream_to_arrays(self) -> None:
        """Test converting stream network to arrays."""
        stream = AppStream()

        for i in range(1, 4):
            stream.add_node(StrmNode(
                id=i,
                x=float(i * 100),
                y=float(i * 50),
                reach_id=1,
                gw_node=i,
            ))

        stream.add_reach(StrmReach(
            id=1,
            upstream_node=1,
            downstream_node=3,
            nodes=[1, 2, 3],
        ))

        arrays = stream.to_arrays()

        assert len(arrays["node_ids"]) == 3
        np.testing.assert_array_equal(arrays["node_ids"], [1, 2, 3])
        np.testing.assert_array_equal(arrays["x"], [100.0, 200.0, 300.0])
        np.testing.assert_array_equal(arrays["gw_nodes"], [1, 2, 3])

    def test_appstream_from_arrays(self) -> None:
        """Test creating stream network from arrays."""
        node_ids = np.array([1, 2, 3, 4])
        x = np.array([0.0, 100.0, 200.0, 300.0])
        y = np.array([0.0, 0.0, 0.0, 0.0])
        reach_ids = np.array([1, 1, 1, 1])
        gw_nodes = np.array([1, 2, 3, 4])

        stream = AppStream.from_arrays(
            node_ids=node_ids,
            x=x,
            y=y,
            reach_ids=reach_ids,
            gw_nodes=gw_nodes,
        )

        assert stream.n_nodes == 4
        assert stream.get_node(1).x == 0.0
        assert stream.get_node(4).x == 300.0


# ---------------------------------------------------------------------------
# Additional tests appended for increased coverage
# ---------------------------------------------------------------------------

from pyiwfm.components.stream import (
    _ccw,
    segments_intersect,
    ReachCrossing,
)


class TestCCW:
    """Tests for the _ccw helper function."""

    def test_ccw_counterclockwise(self) -> None:
        """Points arranged counterclockwise should return True."""
        assert _ccw(0, 0, 1, 0, 0, 1) is True

    def test_ccw_clockwise(self) -> None:
        """Points arranged clockwise should return False."""
        assert _ccw(0, 0, 0, 1, 1, 0) is False

    def test_ccw_collinear(self) -> None:
        """Collinear points should return False (not strictly CCW)."""
        assert _ccw(0, 0, 1, 1, 2, 2) is False


class TestSegmentsIntersect:
    """Tests for the segments_intersect function."""

    def test_crossing_segments(self) -> None:
        """Two segments that form an X should intersect."""
        assert segments_intersect((0, 0), (2, 2), (0, 2), (2, 0)) is True

    def test_shared_endpoint_not_intersection(self) -> None:
        """Segments sharing an endpoint should not be considered intersecting."""
        assert segments_intersect((0, 0), (1, 1), (1, 1), (2, 2)) is False

    def test_parallel_no_overlap(self) -> None:
        """Parallel non-overlapping segments should not intersect."""
        assert segments_intersect((0, 0), (1, 0), (2, 0), (3, 0)) is False

    def test_parallel_offset(self) -> None:
        """Parallel offset segments should not intersect."""
        assert segments_intersect((0, 0), (2, 0), (0, 1), (2, 1)) is False

    def test_non_crossing_disjoint(self) -> None:
        """Two segments that are far apart should not intersect."""
        assert segments_intersect((0, 0), (1, 0), (10, 10), (11, 10)) is False

    def test_t_shape_intersection(self) -> None:
        """A segment that crosses through the middle of another."""
        assert segments_intersect((0, 0), (4, 0), (2, -1), (2, 1)) is True

    def test_nearly_shared_endpoint(self) -> None:
        """Points within tolerance should be treated as shared endpoints."""
        # With a very small offset within default tolerance
        assert segments_intersect(
            (0, 0), (1, 1), (1 + 1e-12, 1 + 1e-12), (2, 2)
        ) is False


class TestReachCrossingDataclass:
    """Tests for the ReachCrossing dataclass."""

    def test_reach_ids(self) -> None:
        """Test reach1_id and reach2_id properties."""
        seg1 = ((0.0, 0.0), (2.0, 2.0), 1, 0)
        seg2 = ((0.0, 2.0), (2.0, 0.0), 2, 0)
        crossing = ReachCrossing(segment1=seg1, segment2=seg2)

        assert crossing.reach1_id == 1
        assert crossing.reach2_id == 2

    def test_repr(self) -> None:
        """Test string representation."""
        seg1 = ((0.0, 0.0), (2.0, 2.0), 10, 3)
        seg2 = ((0.0, 2.0), (2.0, 0.0), 20, 5)
        crossing = ReachCrossing(segment1=seg1, segment2=seg2)

        r = repr(crossing)
        assert "reach 10" in r
        assert "segment 3" in r
        assert "reach 20" in r
        assert "segment 5" in r

    def test_intersection_point_none_by_default(self) -> None:
        """Intersection point should be None when not provided."""
        seg1 = ((0.0, 0.0), (1.0, 1.0), 1, 0)
        seg2 = ((0.0, 1.0), (1.0, 0.0), 2, 0)
        crossing = ReachCrossing(segment1=seg1, segment2=seg2)
        assert crossing.intersection_point is None

    def test_intersection_point_provided(self) -> None:
        """Intersection point can be set explicitly."""
        seg1 = ((0.0, 0.0), (2.0, 2.0), 1, 0)
        seg2 = ((0.0, 2.0), (2.0, 0.0), 2, 0)
        crossing = ReachCrossing(
            segment1=seg1, segment2=seg2, intersection_point=(1.0, 1.0)
        )
        assert crossing.intersection_point == (1.0, 1.0)


class TestStreamRatingValidation:
    """Additional StreamRating tests for validation and edge cases."""

    def test_mismatched_lengths_raises(self) -> None:
        """Stages and flows with different lengths should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            StreamRating(
                stages=np.array([0.0, 1.0, 2.0]),
                flows=np.array([0.0, 10.0]),
            )

    def test_too_few_points_raises(self) -> None:
        """Rating curve with fewer than 2 points should raise ValueError."""
        with pytest.raises(ValueError, match="at least 2 points"):
            StreamRating(
                stages=np.array([1.0]),
                flows=np.array([5.0]),
            )

    def test_get_flow_below_minimum_stage(self) -> None:
        """Flow below minimum stage returns max(0, first flow)."""
        rating = StreamRating(
            stages=np.array([0.0, 1.0, 2.0]),
            flows=np.array([5.0, 10.0, 40.0]),
        )
        # Below minimum: returns max(0, flows[0]) = 5.0
        assert rating.get_flow(-1.0) == pytest.approx(5.0)

    def test_get_flow_above_maximum_stage(self) -> None:
        """Flow above maximum stage uses linear extrapolation."""
        rating = StreamRating(
            stages=np.array([0.0, 1.0, 2.0]),
            flows=np.array([0.0, 10.0, 40.0]),
        )
        # slope = (40 - 10) / (2 - 1) = 30
        # flow = 40 + 30 * (3 - 2) = 70
        assert rating.get_flow(3.0) == pytest.approx(70.0)

    def test_get_stage_below_minimum_flow(self) -> None:
        """Stage below minimum flow returns first stage."""
        rating = StreamRating(
            stages=np.array([1.0, 2.0, 3.0]),
            flows=np.array([0.0, 10.0, 40.0]),
        )
        assert rating.get_stage(-5.0) == pytest.approx(1.0)

    def test_get_stage_above_maximum_flow(self) -> None:
        """Stage above maximum flow uses linear extrapolation."""
        rating = StreamRating(
            stages=np.array([0.0, 1.0, 2.0]),
            flows=np.array([0.0, 10.0, 40.0]),
        )
        # slope = (2 - 1) / (40 - 10) = 1/30
        # stage = 2 + (1/30) * (70 - 40) = 2 + 1 = 3
        assert rating.get_stage(70.0) == pytest.approx(3.0)

    def test_get_flow_exact_minimum(self) -> None:
        """Flow at exact minimum stage returns first flow (clamped to 0)."""
        rating = StreamRating(
            stages=np.array([0.0, 1.0]),
            flows=np.array([-5.0, 10.0]),
        )
        # stage <= stages[0], return max(0, flows[0]) = max(0, -5) = 0
        assert rating.get_flow(0.0) == pytest.approx(0.0)

    def test_get_stage_exact_minimum(self) -> None:
        """Stage at exact minimum flow returns first stage."""
        rating = StreamRating(
            stages=np.array([1.0, 2.0]),
            flows=np.array([0.0, 10.0]),
        )
        assert rating.get_stage(0.0) == pytest.approx(1.0)


class TestStrmNodeExtended:
    """Additional StrmNode tests for coverage."""

    def test_distance_to(self) -> None:
        """Test Euclidean distance between two nodes."""
        node1 = StrmNode(id=1, x=0.0, y=0.0)
        node2 = StrmNode(id=2, x=3.0, y=4.0)

        assert node1.distance_to(node2) == pytest.approx(5.0)

    def test_distance_to_same_point(self) -> None:
        """Distance to self is zero."""
        node = StrmNode(id=1, x=5.0, y=5.0)
        assert node.distance_to(node) == pytest.approx(0.0)

    def test_hash(self) -> None:
        """Equal nodes should have the same hash."""
        node1 = StrmNode(id=1, x=100.0, y=200.0)
        node2 = StrmNode(id=1, x=100.0, y=200.0)
        assert hash(node1) == hash(node2)

        # Usable as dict key / in set
        s = {node1}
        assert node2 in s

    def test_repr(self) -> None:
        """Test string representation."""
        node = StrmNode(id=42, x=0.0, y=0.0, reach_id=7)
        r = repr(node)
        assert "StrmNode" in r
        assert "id=42" in r
        assert "reach=7" in r

    def test_equality_with_non_strmnode(self) -> None:
        """Comparing with non-StrmNode returns NotImplemented."""
        node = StrmNode(id=1, x=0.0, y=0.0)
        result = node.__eq__("not a node")
        assert result is NotImplemented

    def test_equality_different_coordinates(self) -> None:
        """Nodes with same ID but different coordinates are not equal."""
        node1 = StrmNode(id=1, x=0.0, y=0.0)
        node2 = StrmNode(id=1, x=999.0, y=999.0)
        assert node1 != node2


class TestStrmReachExtended:
    """Additional StrmReach tests for coverage."""

    def test_repr(self) -> None:
        """Test string representation."""
        reach = StrmReach(
            id=3,
            name="Test Reach",
            upstream_node=1,
            downstream_node=5,
            nodes=[1, 2, 3, 4, 5],
        )
        r = repr(reach)
        assert "StrmReach" in r
        assert "id=3" in r
        assert "Test Reach" in r
        assert "n_nodes=5" in r

    def test_n_nodes_empty(self) -> None:
        """Reach with no nodes list."""
        reach = StrmReach(id=1, upstream_node=1, downstream_node=1)
        assert reach.n_nodes == 0


class TestDiversionExtended:
    """Additional Diversion tests for coverage."""

    def test_repr(self) -> None:
        """Test string representation."""
        div = Diversion(
            id=5,
            source_node=10,
            destination_type="element",
            destination_id=20,
            max_rate=500.0,
        )
        r = repr(div)
        assert "Diversion" in r
        assert "id=5" in r
        assert "source=10" in r
        assert "max_rate=500.0" in r

    def test_default_max_rate(self) -> None:
        """Default max_rate should be infinite."""
        div = Diversion(id=1, source_node=1, destination_type="outside", destination_id=0)
        assert div.max_rate == float("inf")

    def test_default_priority(self) -> None:
        """Default priority should be 99."""
        div = Diversion(id=1, source_node=1, destination_type="outside", destination_id=0)
        assert div.priority == 99


class TestBypassExtended:
    """Additional Bypass tests for coverage."""

    def test_repr(self) -> None:
        """Test string representation."""
        bypass = Bypass(id=3, source_node=10, destination_node=50, name="Test BP")
        r = repr(bypass)
        assert "Bypass" in r
        assert "id=3" in r
        assert "src=10" in r
        assert "dst=50" in r

    def test_default_name(self) -> None:
        """Default name should be empty."""
        bypass = Bypass(id=1, source_node=1, destination_node=2)
        assert bypass.name == ""


class TestAppStreamRepr:
    """Test AppStream __repr__ method."""

    def test_repr_empty(self) -> None:
        """Test repr for an empty stream network."""
        stream = AppStream()
        r = repr(stream)
        assert "AppStream" in r
        assert "n_nodes=0" in r
        assert "n_reaches=0" in r
        assert "n_diversions=0" in r

    def test_repr_with_data(self) -> None:
        """Test repr for a populated stream network."""
        stream = AppStream()
        stream.add_node(StrmNode(id=1, x=0.0, y=0.0))
        stream.add_node(StrmNode(id=2, x=1.0, y=0.0))
        stream.add_reach(StrmReach(id=1, upstream_node=1, downstream_node=2, nodes=[1, 2]))
        stream.add_diversion(
            Diversion(id=1, source_node=1, destination_type="outside", destination_id=0)
        )
        r = repr(stream)
        assert "n_nodes=2" in r
        assert "n_reaches=1" in r
        assert "n_diversions=1" in r


class TestAppStreamIterators:
    """Tests for iter_nodes and iter_reaches."""

    def test_iter_nodes_sorted_order(self) -> None:
        """Nodes should be yielded in ID order."""
        stream = AppStream()
        stream.add_node(StrmNode(id=5, x=0.0, y=0.0))
        stream.add_node(StrmNode(id=1, x=1.0, y=0.0))
        stream.add_node(StrmNode(id=3, x=2.0, y=0.0))

        ids = [n.id for n in stream.iter_nodes()]
        assert ids == [1, 3, 5]

    def test_iter_nodes_empty(self) -> None:
        """Iterating over empty network yields nothing."""
        stream = AppStream()
        assert list(stream.iter_nodes()) == []

    def test_iter_reaches_sorted_order(self) -> None:
        """Reaches should be yielded in ID order."""
        stream = AppStream()
        stream.add_reach(StrmReach(id=4, upstream_node=1, downstream_node=2))
        stream.add_reach(StrmReach(id=2, upstream_node=3, downstream_node=4))
        stream.add_reach(StrmReach(id=7, upstream_node=5, downstream_node=6))

        ids = [r.id for r in stream.iter_reaches()]
        assert ids == [2, 4, 7]

    def test_iter_reaches_empty(self) -> None:
        """Iterating over empty reaches yields nothing."""
        stream = AppStream()
        assert list(stream.iter_reaches()) == []


class TestAppStreamConnectivity:
    """Extended connectivity tests."""

    def test_get_downstream_node_without_build(self) -> None:
        """Downstream lookup falls back to node attribute when cache is empty."""
        stream = AppStream()
        stream.add_node(StrmNode(id=1, x=0.0, y=0.0, downstream_node=2))
        stream.add_node(StrmNode(id=2, x=1.0, y=0.0))

        # Without calling build_connectivity, should fall back to node attribute
        assert stream.get_downstream_node(1) == 2

    def test_get_upstream_node_without_build(self) -> None:
        """Upstream lookup falls back to node attribute when cache is empty."""
        stream = AppStream()
        stream.add_node(StrmNode(id=1, x=0.0, y=0.0))
        stream.add_node(StrmNode(id=2, x=1.0, y=0.0, upstream_node=1))

        assert stream.get_upstream_node(2) == 1

    def test_get_downstream_node_nonexistent(self) -> None:
        """Getting downstream of nonexistent node returns None."""
        stream = AppStream()
        assert stream.get_downstream_node(999) is None

    def test_get_upstream_node_nonexistent(self) -> None:
        """Getting upstream of nonexistent node returns None."""
        stream = AppStream()
        assert stream.get_upstream_node(999) is None

    def test_get_downstream_node_none_value(self) -> None:
        """Node with no downstream returns None."""
        stream = AppStream()
        stream.add_node(StrmNode(id=1, x=0.0, y=0.0))
        assert stream.get_downstream_node(1) is None

    def test_get_upstream_node_none_value(self) -> None:
        """Node with no upstream returns None."""
        stream = AppStream()
        stream.add_node(StrmNode(id=1, x=0.0, y=0.0))
        assert stream.get_upstream_node(1) is None

    def test_build_connectivity_clears_cache(self) -> None:
        """Building connectivity should refresh the cache."""
        stream = AppStream()
        stream.add_node(
            StrmNode(id=1, x=0.0, y=0.0, downstream_node=2, upstream_node=None)
        )
        stream.add_node(
            StrmNode(id=2, x=1.0, y=0.0, downstream_node=None, upstream_node=1)
        )

        stream.build_connectivity()
        assert stream.get_downstream_node(1) == 2
        assert stream.get_upstream_node(2) == 1

        # Now remove node references and rebuild
        stream.nodes[1].downstream_node = None
        stream.nodes[2].upstream_node = None
        stream.build_connectivity()

        # The cache should reflect the cleared connectivity
        assert stream.get_downstream_node(1) is None
        assert stream.get_upstream_node(2) is None


class TestAppStreamReachLength:
    """Extended reach length tests."""

    def test_single_node_reach(self) -> None:
        """A reach with a single node has zero length."""
        stream = AppStream()
        stream.add_node(StrmNode(id=1, x=0.0, y=0.0, reach_id=1))
        stream.add_reach(StrmReach(id=1, upstream_node=1, downstream_node=1, nodes=[1]))

        assert stream.get_reach_length(1) == pytest.approx(0.0)

    def test_diagonal_reach_length(self) -> None:
        """Reach length along a diagonal should use Euclidean distance."""
        stream = AppStream()
        stream.add_node(StrmNode(id=1, x=0.0, y=0.0, reach_id=1))
        stream.add_node(StrmNode(id=2, x=3.0, y=4.0, reach_id=1))
        stream.add_reach(StrmReach(id=1, upstream_node=1, downstream_node=2, nodes=[1, 2]))

        assert stream.get_reach_length(1) == pytest.approx(5.0)

    def test_total_length_empty_network(self) -> None:
        """Total length of empty network is zero."""
        stream = AppStream()
        assert stream.get_total_length() == pytest.approx(0.0)


class TestAppStreamValidation:
    """Extended validation tests."""

    def test_validate_reach_references_missing_node(self) -> None:
        """Validation should fail if a reach references a nonexistent node."""
        stream = AppStream()
        stream.add_node(StrmNode(id=1, x=0.0, y=0.0))
        stream.add_reach(
            StrmReach(id=1, upstream_node=1, downstream_node=99, nodes=[1, 99])
        )

        with pytest.raises(ComponentError, match="non-existent node 99"):
            stream.validate()

    def test_validate_diversion_references_missing_node(self) -> None:
        """Validation should fail if a diversion references a nonexistent source node."""
        stream = AppStream()
        stream.add_node(StrmNode(id=1, x=0.0, y=0.0))
        stream.add_diversion(
            Diversion(id=1, source_node=999, destination_type="outside", destination_id=0)
        )

        with pytest.raises(ComponentError, match="non-existent source node 999"):
            stream.validate()

    def test_validate_bypass_source_missing(self) -> None:
        """Validation should fail if bypass source node does not exist."""
        stream = AppStream()
        stream.add_node(StrmNode(id=1, x=0.0, y=0.0))
        stream.add_bypass(Bypass(id=1, source_node=999, destination_node=1))

        with pytest.raises(ComponentError, match="non-existent source node 999"):
            stream.validate()

    def test_validate_bypass_destination_missing(self) -> None:
        """Validation should fail if bypass destination node does not exist."""
        stream = AppStream()
        stream.add_node(StrmNode(id=1, x=0.0, y=0.0))
        stream.add_bypass(Bypass(id=1, source_node=1, destination_node=999))

        with pytest.raises(ComponentError, match="non-existent destination node 999"):
            stream.validate()

    def test_validate_valid_network_with_diversions_and_bypasses(self) -> None:
        """A fully valid network with diversions and bypasses should pass."""
        stream = AppStream()
        stream.add_node(StrmNode(id=1, x=0.0, y=0.0))
        stream.add_node(StrmNode(id=2, x=1.0, y=0.0))
        stream.add_reach(
            StrmReach(id=1, upstream_node=1, downstream_node=2, nodes=[1, 2])
        )
        stream.add_diversion(
            Diversion(id=1, source_node=1, destination_type="outside", destination_id=0)
        )
        stream.add_bypass(Bypass(id=1, source_node=1, destination_node=2))

        # Should not raise
        stream.validate()


class TestAppStreamGetReachSegments:
    """Tests for get_reach_segments method."""

    def test_segments_basic(self) -> None:
        """Test segment extraction from a reach."""
        stream = AppStream()
        stream.add_node(StrmNode(id=1, x=0.0, y=0.0, reach_id=1))
        stream.add_node(StrmNode(id=2, x=1.0, y=0.0, reach_id=1))
        stream.add_node(StrmNode(id=3, x=2.0, y=0.0, reach_id=1))
        stream.add_reach(
            StrmReach(id=1, upstream_node=1, downstream_node=3, nodes=[1, 2, 3])
        )

        segments = stream.get_reach_segments(1)
        assert len(segments) == 2
        assert segments[0] == ((0.0, 0.0), (1.0, 0.0))
        assert segments[1] == ((1.0, 0.0), (2.0, 0.0))

    def test_segments_single_node_reach(self) -> None:
        """A reach with one node produces no segments."""
        stream = AppStream()
        stream.add_node(StrmNode(id=1, x=0.0, y=0.0, reach_id=1))
        stream.add_reach(
            StrmReach(id=1, upstream_node=1, downstream_node=1, nodes=[1])
        )

        segments = stream.get_reach_segments(1)
        assert segments == []

    def test_segments_empty_reach(self) -> None:
        """A reach with no nodes produces no segments."""
        stream = AppStream()
        stream.add_reach(
            StrmReach(id=1, upstream_node=1, downstream_node=1, nodes=[])
        )

        segments = stream.get_reach_segments(1)
        assert segments == []


class TestAppStreamDetectCrossings:
    """Tests for detect_crossings and validate_no_crossings methods."""

    def _build_crossing_network(self) -> AppStream:
        """Build a network with two crossing reaches (X shape)."""
        stream = AppStream()
        # Reach 1: from (0,0) to (4,4)
        stream.add_node(StrmNode(id=1, x=0.0, y=0.0, reach_id=1))
        stream.add_node(StrmNode(id=2, x=4.0, y=4.0, reach_id=1))
        stream.add_reach(
            StrmReach(id=1, upstream_node=1, downstream_node=2, nodes=[1, 2])
        )

        # Reach 2: from (0,4) to (4,0) -- crosses reach 1
        stream.add_node(StrmNode(id=3, x=0.0, y=4.0, reach_id=2))
        stream.add_node(StrmNode(id=4, x=4.0, y=0.0, reach_id=2))
        stream.add_reach(
            StrmReach(id=2, upstream_node=3, downstream_node=4, nodes=[3, 4])
        )

        return stream

    def _build_non_crossing_network(self) -> AppStream:
        """Build a network with two parallel reaches (no crossings)."""
        stream = AppStream()
        # Reach 1: horizontal at y=0
        stream.add_node(StrmNode(id=1, x=0.0, y=0.0, reach_id=1))
        stream.add_node(StrmNode(id=2, x=4.0, y=0.0, reach_id=1))
        stream.add_reach(
            StrmReach(id=1, upstream_node=1, downstream_node=2, nodes=[1, 2])
        )

        # Reach 2: horizontal at y=2
        stream.add_node(StrmNode(id=3, x=0.0, y=2.0, reach_id=2))
        stream.add_node(StrmNode(id=4, x=4.0, y=2.0, reach_id=2))
        stream.add_reach(
            StrmReach(id=2, upstream_node=3, downstream_node=4, nodes=[3, 4])
        )

        return stream

    def test_detect_crossings_found(self) -> None:
        """Crossing reaches should be detected."""
        stream = self._build_crossing_network()
        crossings = stream.detect_crossings()
        assert len(crossings) == 1
        assert crossings[0].reach1_id == 1
        assert crossings[0].reach2_id == 2

    def test_detect_crossings_none(self) -> None:
        """Parallel reaches should produce no crossings."""
        stream = self._build_non_crossing_network()
        crossings = stream.detect_crossings()
        assert len(crossings) == 0

    def test_detect_crossings_empty_network(self) -> None:
        """Empty network should produce no crossings."""
        stream = AppStream()
        crossings = stream.detect_crossings()
        assert crossings == []

    def test_detect_crossings_intersection_point(self) -> None:
        """Intersection point should be computed for crossing segments."""
        stream = self._build_crossing_network()
        crossings = stream.detect_crossings()
        assert len(crossings) == 1
        pt = crossings[0].intersection_point
        assert pt is not None
        assert pt[0] == pytest.approx(2.0)
        assert pt[1] == pytest.approx(2.0)

    def test_validate_no_crossings_raises(self) -> None:
        """validate_no_crossings should raise ComponentError when crossings found."""
        stream = self._build_crossing_network()
        with pytest.raises(ComponentError, match="invalid crossing"):
            stream.validate_no_crossings()

    def test_validate_no_crossings_no_raise(self) -> None:
        """validate_no_crossings with raise_on_error=False returns crossing list."""
        stream = self._build_crossing_network()
        crossings = stream.validate_no_crossings(raise_on_error=False)
        assert len(crossings) == 1

    def test_validate_no_crossings_clean(self) -> None:
        """validate_no_crossings on a clean network should return empty list."""
        stream = self._build_non_crossing_network()
        crossings = stream.validate_no_crossings()
        assert crossings == []

    def test_validate_no_crossings_message_format(self) -> None:
        """Error message should contain reach IDs and point information."""
        stream = self._build_crossing_network()
        with pytest.raises(ComponentError) as exc_info:
            stream.validate_no_crossings()

        msg = str(exc_info.value)
        assert "Reach 1" in msg
        assert "Reach 2" in msg

    def test_shared_endpoint_not_reported_as_crossing(self) -> None:
        """Two reaches sharing an endpoint (confluence) should not be a crossing."""
        stream = AppStream()
        # Reach 1: (0,0) -> (1,1)
        stream.add_node(StrmNode(id=1, x=0.0, y=0.0, reach_id=1))
        stream.add_node(StrmNode(id=2, x=1.0, y=1.0, reach_id=1))
        stream.add_reach(
            StrmReach(id=1, upstream_node=1, downstream_node=2, nodes=[1, 2])
        )

        # Reach 2: (1,1) -> (2,0) -- shares endpoint with reach 1
        stream.add_node(StrmNode(id=3, x=1.0, y=1.0, reach_id=2))
        stream.add_node(StrmNode(id=4, x=2.0, y=0.0, reach_id=2))
        stream.add_reach(
            StrmReach(id=2, upstream_node=3, downstream_node=4, nodes=[3, 4])
        )

        crossings = stream.detect_crossings()
        assert len(crossings) == 0


class TestAppStreamComputeIntersection:
    """Tests for _compute_intersection internal method."""

    def test_normal_intersection(self) -> None:
        """Two crossing segments yield their intersection point."""
        stream = AppStream()
        pt = stream._compute_intersection(
            (0.0, 0.0), (2.0, 2.0), (0.0, 2.0), (2.0, 0.0)
        )
        assert pt is not None
        assert pt[0] == pytest.approx(1.0)
        assert pt[1] == pytest.approx(1.0)

    def test_parallel_segments_return_none(self) -> None:
        """Parallel segments return None (no intersection)."""
        stream = AppStream()
        pt = stream._compute_intersection(
            (0.0, 0.0), (2.0, 0.0), (0.0, 1.0), (2.0, 1.0)
        )
        assert pt is None

    def test_coincident_segments_return_none(self) -> None:
        """Coincident (overlapping) segments return None."""
        stream = AppStream()
        pt = stream._compute_intersection(
            (0.0, 0.0), (2.0, 0.0), (1.0, 0.0), (3.0, 0.0)
        )
        assert pt is None


class TestAppStreamGetNodesInReach:
    """Tests for get_nodes_in_reach with edge cases."""

    def test_reach_with_missing_nodes(self) -> None:
        """Nodes listed in reach but not in the network are skipped."""
        stream = AppStream()
        stream.add_node(StrmNode(id=1, x=0.0, y=0.0, reach_id=1))
        stream.add_node(StrmNode(id=3, x=2.0, y=0.0, reach_id=1))
        # Node 2 is missing from the network
        stream.add_reach(
            StrmReach(id=1, upstream_node=1, downstream_node=3, nodes=[1, 2, 3])
        )

        nodes = stream.get_nodes_in_reach(1)
        assert len(nodes) == 2
        assert [n.id for n in nodes] == [1, 3]


class TestAppStreamFromArraysEdgeCases:
    """Additional from_arrays / to_arrays tests."""

    def test_from_arrays_no_gw_nodes(self) -> None:
        """Creating network without gw_nodes should set gw_node to None."""
        node_ids = np.array([1, 2])
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 0.0])
        reach_ids = np.array([1, 1])

        stream = AppStream.from_arrays(
            node_ids=node_ids, x=x, y=y, reach_ids=reach_ids, gw_nodes=None
        )

        assert stream.get_node(1).gw_node is None
        assert stream.get_node(2).gw_node is None

    def test_from_arrays_zero_gw_node_treated_as_none(self) -> None:
        """A gw_node value of 0 should be treated as None."""
        node_ids = np.array([1])
        x = np.array([0.0])
        y = np.array([0.0])
        reach_ids = np.array([1])
        gw_nodes = np.array([0])

        stream = AppStream.from_arrays(
            node_ids=node_ids, x=x, y=y, reach_ids=reach_ids, gw_nodes=gw_nodes
        )

        assert stream.get_node(1).gw_node is None

    def test_to_arrays_with_none_gw_nodes(self) -> None:
        """Nodes with gw_node=None should produce 0 in the gw_nodes array."""
        stream = AppStream()
        stream.add_node(StrmNode(id=1, x=0.0, y=0.0, reach_id=1, gw_node=None))
        stream.add_node(StrmNode(id=2, x=1.0, y=0.0, reach_id=1, gw_node=5))

        arrays = stream.to_arrays()
        np.testing.assert_array_equal(arrays["gw_nodes"], [0, 5])

    def test_roundtrip_to_from_arrays(self) -> None:
        """Converting to arrays and back preserves data."""
        stream = AppStream()
        stream.add_node(StrmNode(id=1, x=10.0, y=20.0, reach_id=1, gw_node=100))
        stream.add_node(StrmNode(id=2, x=30.0, y=40.0, reach_id=2, gw_node=200))

        arrays = stream.to_arrays()
        stream2 = AppStream.from_arrays(
            node_ids=arrays["node_ids"],
            x=arrays["x"],
            y=arrays["y"],
            reach_ids=arrays["reach_ids"],
            gw_nodes=arrays["gw_nodes"],
        )

        assert stream2.n_nodes == 2
        assert stream2.get_node(1).x == pytest.approx(10.0)
        assert stream2.get_node(1).gw_node == 100
        assert stream2.get_node(2).reach_id == 2


class TestAppStreamProperties:
    """Tests for property methods on AppStream."""

    def test_n_nodes(self) -> None:
        """n_nodes returns the count of nodes."""
        stream = AppStream()
        assert stream.n_nodes == 0
        stream.add_node(StrmNode(id=1, x=0.0, y=0.0))
        assert stream.n_nodes == 1
        stream.add_node(StrmNode(id=2, x=1.0, y=0.0))
        assert stream.n_nodes == 2

    def test_n_reaches(self) -> None:
        """n_reaches returns the count of reaches."""
        stream = AppStream()
        assert stream.n_reaches == 0
        stream.add_reach(StrmReach(id=1, upstream_node=1, downstream_node=2))
        assert stream.n_reaches == 1

    def test_n_diversions(self) -> None:
        """n_diversions returns the count of diversions."""
        stream = AppStream()
        assert stream.n_diversions == 0
        stream.add_diversion(
            Diversion(id=1, source_node=1, destination_type="outside", destination_id=0)
        )
        assert stream.n_diversions == 1

    def test_n_bypasses(self) -> None:
        """n_bypasses returns the count of bypasses."""
        stream = AppStream()
        assert stream.n_bypasses == 0
        stream.add_bypass(Bypass(id=1, source_node=1, destination_node=2))
        assert stream.n_bypasses == 1


class TestAppStreamMultipleCrossings:
    """Test detection of multiple crossings and error message truncation."""

    def test_many_crossings_error_message_truncated(self) -> None:
        """When more than 5 crossings exist, the error message shows '... and N more'."""
        stream = AppStream()
        node_id = 1

        # Create 7 horizontal reaches
        horizontal_reaches = []
        for h in range(7):
            nid1 = node_id
            nid2 = node_id + 1
            node_id += 2
            y_val = float(h)
            stream.add_node(StrmNode(id=nid1, x=-1.0, y=y_val, reach_id=100 + h))
            stream.add_node(StrmNode(id=nid2, x=7.0, y=y_val, reach_id=100 + h))
            rid = 100 + h
            stream.add_reach(
                StrmReach(id=rid, upstream_node=nid1, downstream_node=nid2, nodes=[nid1, nid2])
            )
            horizontal_reaches.append(rid)

        # Create a single vertical reach that crosses all 7 horizontal reaches
        nid1 = node_id
        nid2 = node_id + 1
        stream.add_node(StrmNode(id=nid1, x=3.0, y=-1.0, reach_id=999))
        stream.add_node(StrmNode(id=nid2, x=3.0, y=7.0, reach_id=999))
        stream.add_reach(
            StrmReach(id=999, upstream_node=nid1, downstream_node=nid2, nodes=[nid1, nid2])
        )

        crossings = stream.detect_crossings()
        assert len(crossings) == 7

        with pytest.raises(ComponentError) as exc_info:
            stream.validate_no_crossings()

        msg = str(exc_info.value)
        assert "7 invalid crossing" in msg
        assert "... and 2 more" in msg
