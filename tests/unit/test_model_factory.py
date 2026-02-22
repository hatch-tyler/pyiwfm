"""Tests for pyiwfm.core.model_factory helper functions."""

from __future__ import annotations

from pyiwfm.core.model_factory import (
    build_reaches_from_node_reach_ids,
    resolve_stream_node_coordinates,
)
from tests.conftest import make_simple_grid


class TestBuildReachesFromNodeReachIds:
    """Tests for build_reaches_from_node_reach_ids."""

    def test_builds_reaches_from_nodes(self):
        """Groups nodes by reach_id and creates StrmReach objects."""
        from pyiwfm.components.stream import AppStream, StrmNode

        stream = AppStream()
        stream.add_node(StrmNode(id=1, x=0.0, y=0.0, reach_id=1))
        stream.add_node(StrmNode(id=2, x=1.0, y=0.0, reach_id=1))
        stream.add_node(StrmNode(id=3, x=2.0, y=0.0, reach_id=2))

        build_reaches_from_node_reach_ids(stream)

        assert len(stream.reaches) == 2
        assert stream.reaches[1].upstream_node == 1
        assert stream.reaches[1].downstream_node == 2
        assert stream.reaches[2].upstream_node == 3

    def test_does_not_overwrite_existing_reaches(self):
        """If reaches already exist, does not rebuild them."""
        from pyiwfm.components.stream import AppStream, StrmNode, StrmReach

        stream = AppStream()
        stream.add_node(StrmNode(id=1, x=0.0, y=0.0, reach_id=1))
        stream.add_reach(StrmReach(id=99, upstream_node=1, downstream_node=1, nodes=[1]))

        build_reaches_from_node_reach_ids(stream)

        # Should keep the existing reach, not rebuild
        assert len(stream.reaches) == 1
        assert 99 in stream.reaches

    def test_skips_nodes_without_reach_id(self):
        """Nodes with reach_id=0 are ignored."""
        from pyiwfm.components.stream import AppStream, StrmNode

        stream = AppStream()
        stream.add_node(StrmNode(id=1, x=0.0, y=0.0, reach_id=0))
        stream.add_node(StrmNode(id=2, x=0.0, y=0.0, reach_id=1))

        build_reaches_from_node_reach_ids(stream)

        assert len(stream.reaches) == 1
        assert stream.reaches[1].nodes == [2]


class TestResolveStreamNodeCoordinates:
    """Tests for resolve_stream_node_coordinates."""

    def test_resolves_placeholder_coords(self):
        """Updates (0,0) stream nodes from GW node coordinates."""
        from pyiwfm.components.stream import AppStream, StrmNode
        from pyiwfm.core.model import IWFMModel

        grid = make_simple_grid()

        stream = AppStream()
        # Node 1 in the grid has known coordinates
        gw_node_id = 1
        stream.add_node(StrmNode(id=1, x=0.0, y=0.0, gw_node=gw_node_id))

        model = IWFMModel(name="test", mesh=grid, streams=stream)
        resolved = resolve_stream_node_coordinates(model)

        assert resolved == 1
        assert stream.nodes[1].x == grid.nodes[gw_node_id].x
        assert stream.nodes[1].y == grid.nodes[gw_node_id].y

    def test_skips_nodes_with_real_coords(self):
        """Nodes with non-zero coords are not overwritten."""
        from pyiwfm.components.stream import AppStream, StrmNode
        from pyiwfm.core.model import IWFMModel

        grid = make_simple_grid()

        stream = AppStream()
        stream.add_node(StrmNode(id=1, x=99.0, y=99.0, gw_node=1))

        model = IWFMModel(name="test", mesh=grid, streams=stream)
        resolved = resolve_stream_node_coordinates(model)

        assert resolved == 0
        assert stream.nodes[1].x == 99.0

    def test_no_mesh_returns_zero(self):
        """Returns 0 when model has no mesh."""
        from pyiwfm.core.model import IWFMModel

        model = IWFMModel(name="test")
        assert resolve_stream_node_coordinates(model) == 0

    def test_no_streams_returns_zero(self):
        """Returns 0 when model has no streams."""
        from pyiwfm.core.model import IWFMModel

        model = IWFMModel(name="test", mesh=make_simple_grid())
        assert resolve_stream_node_coordinates(model) == 0
