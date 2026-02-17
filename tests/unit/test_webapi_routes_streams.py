"""Comprehensive tests for the FastAPI stream routes.

Covers all endpoints and helper functions in
``pyiwfm.visualization.webapi.routes.streams``:

* Helper functions: ``_get_gs_elev_lookup``, ``_node_z``, ``_make_stream_node``,
  ``_build_reaches_from_connectivity``, ``_build_reaches_from_preprocessor_binary``,
  ``_build_streams_from_nodes``, ``_build_streams_from_reaches``,
  ``_build_stream_data``, ``_get_gw_nodes_for_reaches``
* ``GET /api/streams`` -- stream network as JSON
* ``GET /api/streams/geojson`` -- stream as GeoJSON LineStrings
* ``GET /api/streams/diversions`` -- diversion arcs
* ``GET /api/streams/diversions/{div_id}`` -- diversion detail
* ``GET /api/streams/reach-profile`` -- longitudinal reach profile

The VTP endpoint (``/api/streams/vtp``) is intentionally skipped because it
requires the ``vtk`` package which may not be available in the test environment.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("fastapi", reason="FastAPI not available")
pytest.importorskip("pydantic", reason="Pydantic not available")

from fastapi.testclient import TestClient

from pyiwfm.core.mesh import AppGrid, Element, Node
from pyiwfm.visualization.webapi.config import model_state
from pyiwfm.visualization.webapi.routes.streams import (
    _build_reaches_from_connectivity,
    _build_reaches_from_preprocessor_binary,
    _build_stream_data,
    _build_streams_from_nodes,
    _build_streams_from_reaches,
    _get_gs_elev_lookup,
    _get_gw_nodes_for_reaches,
    _make_stream_node,
    _node_z,
)
from pyiwfm.visualization.webapi.server import create_app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_model_state():
    """Reset the global model_state to a clean state."""
    model_state._model = None
    model_state._mesh_3d = None
    model_state._mesh_surface = None
    model_state._surface_json_data = None
    model_state._bounds = None
    model_state._pv_mesh_3d = None
    model_state._layer_surface_cache = {}
    model_state._crs = "+proj=utm +zone=10 +datum=NAD83 +units=us-ft +no_defs"
    model_state._transformer = None
    model_state._geojson_cache = {}
    model_state._head_loader = None
    model_state._gw_hydrograph_reader = None
    model_state._stream_hydrograph_reader = None
    model_state._budget_readers = {}
    model_state._observations = {}
    model_state._results_dir = None
    model_state._stream_reach_boundaries = None
    model_state._diversion_ts_data = None
    # Restore any monkey-patched methods back to the class originals
    for attr in (
        "get_budget_reader",
        "get_available_budgets",
        "reproject_coords",
        "get_stream_reach_boundaries",
        "get_head_loader",
        "get_gw_hydrograph_reader",
        "get_stream_hydrograph_reader",
        "get_area_manager",
        "get_subsidence_reader",
    ):
        if attr in model_state.__dict__:
            del model_state.__dict__[attr]


def _make_grid():
    """Create a 6-node, 2-element grid for testing."""
    nodes = {
        1: Node(id=1, x=0.0, y=0.0),
        2: Node(id=2, x=100.0, y=0.0),
        3: Node(id=3, x=200.0, y=0.0),
        4: Node(id=4, x=0.0, y=100.0),
        5: Node(id=5, x=100.0, y=100.0),
        6: Node(id=6, x=200.0, y=100.0),
    }
    elements = {
        1: Element(id=1, vertices=(1, 2, 5, 4), subregion=1),
        2: Element(id=2, vertices=(2, 3, 6, 5), subregion=1),
    }
    grid = AppGrid(nodes=nodes, elements=elements)
    grid.compute_connectivity()
    grid.compute_areas()
    return grid


def _make_strm_node(
    id,
    gw_node,
    reach_id=0,
    downstream_node=None,
    x=0.0,
    y=0.0,
):
    """Create a mock StrmNode object."""
    sn = MagicMock()
    sn.id = id
    sn.gw_node = gw_node
    sn.reach_id = reach_id
    sn.downstream_node = downstream_node
    sn.groundwater_node = gw_node
    sn.bottom_elev = 0.0
    sn.cross_section = None
    sn.conductivity = 0.01
    sn.bed_thickness = 1.0
    sn.x = x
    sn.y = y
    return sn


def _make_mock_stream(nodes_dict=None, reaches=None, diversions=None):
    """Create a mock stream component."""
    stream = MagicMock()
    stream.nodes = nodes_dict or {}
    stream.reaches = reaches or {}
    stream.diversions = diversions or {}
    stream.diversion_element_groups = []
    return stream


def _make_mock_diversion(
    source_node=1,
    destination_type="element",
    destination_id=1,
    name=None,
    max_rate=100.0,
    priority=1,
    max_div_column=0,
    delivery_column=0,
    max_div_fraction=1.0,
    delivery_fraction=1.0,
    delivery_dest_id=None,
):
    """Create a mock diversion object."""
    div = MagicMock()
    div.source_node = source_node
    div.destination_type = destination_type
    div.destination_id = destination_id
    div.name = name
    div.max_rate = max_rate
    div.priority = priority
    div.max_div_column = max_div_column
    div.delivery_column = delivery_column
    div.max_div_fraction = max_div_fraction
    div.delivery_fraction = delivery_fraction
    div.delivery_dest_id = delivery_dest_id if delivery_dest_id is not None else destination_id
    return div


def _make_mock_model(grid=None, streams=None, has_streams=False, metadata=None):
    """Create a minimal mock IWFMModel."""
    model = MagicMock()
    model.name = "TestModel"
    model.grid = grid or _make_grid()
    model.streams = streams
    model.has_streams = has_streams
    model.stratigraphy = None
    model.metadata = metadata or {}
    model.lakes = None
    model.groundwater = None
    model.has_lakes = False
    model.n_nodes = len(model.grid.nodes)
    model.n_elements = len(model.grid.elements)
    model.n_layers = 1
    model.n_lakes = 0
    model.n_stream_nodes = 0
    model.source_files = {}
    return model


def _make_strat(gs_elev_values):
    """Create a mock stratigraphy with gs_elev array."""
    strat = MagicMock()
    strat.gs_elev = np.array(gs_elev_values, dtype=float)
    return strat


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client_no_model():
    """TestClient with no model loaded."""
    _reset_model_state()
    app = create_app()
    yield TestClient(app)
    _reset_model_state()


@pytest.fixture()
def client_model_no_streams():
    """TestClient with a model but no stream data."""
    _reset_model_state()
    model = _make_mock_model()
    model.has_streams = False
    model.streams = None
    model_state._model = model
    app = create_app()
    yield TestClient(app)
    _reset_model_state()


# ===========================================================================
# Unit tests for helper functions
# ===========================================================================


class TestGetGsElevLookup:
    """Tests for _get_gs_elev_lookup."""

    def test_no_stratigraphy_returns_empty(self):
        grid = _make_grid()
        lookup = _get_gs_elev_lookup(grid, None)
        assert lookup == {}

    def test_with_stratigraphy_builds_lookup(self):
        grid = _make_grid()
        strat = _make_strat([100.0] * 6)
        lookup = _get_gs_elev_lookup(grid, strat)
        # Each node in the grid should have an index
        assert len(lookup) == 6
        for nid in grid.nodes:
            assert nid in lookup


class TestNodeZ:
    """Tests for _node_z."""

    def test_no_stratigraphy_returns_zero(self):
        z = _node_z(1, None, {})
        assert z == 0.0

    def test_node_not_in_lookup_returns_zero(self):
        strat = _make_strat([100.0])
        z = _node_z(999, strat, {1: 0})
        assert z == 0.0

    def test_valid_lookup_returns_elevation(self):
        strat = _make_strat([42.5, 55.0, 60.0])
        lookup = {1: 0, 2: 1, 3: 2}
        assert _node_z(1, strat, lookup) == 42.5
        assert _node_z(2, strat, lookup) == 55.0
        assert _node_z(3, strat, lookup) == 60.0


class TestMakeStreamNode:
    """Tests for _make_stream_node."""

    def test_no_gw_node_returns_none(self):
        sn = MagicMock()
        sn.gw_node = None
        sn.id = 1
        grid = _make_grid()
        result = _make_stream_node(sn, grid, None, {})
        assert result is None

    def test_gw_node_not_in_grid_returns_none(self):
        sn = _make_strm_node(id=1, gw_node=999)
        grid = _make_grid()
        result = _make_stream_node(sn, grid, None, {})
        assert result is None

    def test_valid_stream_node(self):
        sn = _make_strm_node(id=10, gw_node=1, reach_id=5)
        grid = _make_grid()
        result = _make_stream_node(sn, grid, None, {}, reach_id=5)
        assert result is not None
        assert result.id == 10
        assert result.x == 0.0
        assert result.y == 0.0
        assert result.z == 0.0
        assert result.reach_id == 5

    def test_reach_id_fallback_to_sn_attribute(self):
        sn = _make_strm_node(id=10, gw_node=1, reach_id=3)
        grid = _make_grid()
        result = _make_stream_node(sn, grid, None, {}, reach_id=0)
        assert result.reach_id == 3

    def test_with_stratigraphy_gets_z(self):
        grid = _make_grid()
        strat = _make_strat([42.0, 50.0, 55.0, 60.0, 65.0, 70.0])
        lookup = _get_gs_elev_lookup(grid, strat)
        sn = _make_strm_node(id=1, gw_node=1)
        result = _make_stream_node(sn, grid, strat, lookup)
        assert result.z == 42.0


class TestBuildReachesFromConnectivity:
    """Tests for _build_reaches_from_connectivity."""

    def test_no_connectivity_returns_none(self):
        """When no downstream_node is set, returns None."""
        sn1 = _make_strm_node(id=1, gw_node=1, downstream_node=None)
        sn2 = _make_strm_node(id=2, gw_node=2, downstream_node=None)
        stream = _make_mock_stream({1: sn1, 2: sn2})
        grid = _make_grid()
        result = _build_reaches_from_connectivity(stream, grid, None, {})
        assert result is None

    def test_simple_chain(self):
        """Chain: 1 -> 2 -> 3 forms one reach."""
        sn1 = _make_strm_node(id=1, gw_node=1, downstream_node=2)
        sn2 = _make_strm_node(id=2, gw_node=2, downstream_node=3)
        sn3 = _make_strm_node(id=3, gw_node=3, downstream_node=None)
        stream = _make_mock_stream({1: sn1, 2: sn2, 3: sn3})
        grid = _make_grid()
        result = _build_reaches_from_connectivity(stream, grid, None, {})
        assert result is not None
        nodes_data, reaches_data = result
        assert len(nodes_data) >= 3
        assert len(reaches_data) >= 1
        # All three node ids should be in the first reach
        assert 1 in reaches_data[0]
        assert 3 in reaches_data[0]

    def test_confluence_splits_reach(self):
        """Two branches merging at node 3.
        1 -> 3, 2 -> 3, 3 -> 4
        Should produce reaches that include the junction node.
        """
        sn1 = _make_strm_node(id=1, gw_node=1, downstream_node=3)
        sn2 = _make_strm_node(id=2, gw_node=2, downstream_node=3)
        sn3 = _make_strm_node(id=3, gw_node=3, downstream_node=4)
        sn4 = _make_strm_node(id=4, gw_node=4, downstream_node=None)
        stream = _make_mock_stream({1: sn1, 2: sn2, 3: sn3, 4: sn4})
        grid = _make_grid()
        result = _build_reaches_from_connectivity(stream, grid, None, {})
        assert result is not None
        nodes_data, reaches_data = result
        # Multiple reaches expected due to confluence
        assert len(reaches_data) >= 2

    def test_visited_node_terminates_reach(self):
        """When tracing hits an already-visited node, the reach ends there."""
        sn1 = _make_strm_node(id=1, gw_node=1, downstream_node=2)
        sn2 = _make_strm_node(id=2, gw_node=2, downstream_node=3)
        sn3 = _make_strm_node(id=3, gw_node=3, downstream_node=2)  # cycle back to 2
        stream = _make_mock_stream({1: sn1, 2: sn2, 3: sn3})
        grid = _make_grid()
        result = _build_reaches_from_connectivity(stream, grid, None, {})
        assert result is not None
        nodes_data, reaches_data = result
        # Should still produce valid reaches without infinite loop
        assert len(reaches_data) >= 1

    def test_invalid_gw_node_skipped(self):
        """Stream nodes with invalid gw_node are not in valid_nodes."""
        sn1 = _make_strm_node(id=1, gw_node=1, downstream_node=2)
        sn2 = _make_strm_node(id=2, gw_node=999, downstream_node=None)  # invalid gw
        stream = _make_mock_stream({1: sn1, 2: sn2})
        grid = _make_grid()
        result = _build_reaches_from_connectivity(stream, grid, None, {})
        # Node 2 is invalid, so downstream[1] = 2 but 2 is not in stream.nodes? No, 2 IS in stream.nodes but not in valid_nodes.
        # The connectivity is 1 -> 2, but dn=2 is in stream.nodes but sn2 isn't in valid_nodes. Actually the code checks:
        # if dn and dn in stream.nodes: downstream[sn.id] = dn   -- sn2 IS in stream.nodes
        # but sn2 is not in valid_nodes (gw_node=999 not in grid.nodes)
        # Head node is 1 (not in has_upstream since 2 is the only one in has_upstream but 2's upstream from 1)
        # Wait: sn1's gw_node=1 is valid, sn2's gw_node=999 is not valid
        # valid_nodes = {1: sn1}  (only node 1)
        # downstream = {1: 2}  (because 2 is in stream.nodes even though invalid gw)
        # has_upstream = {2}
        # head_nodes = {1} - {2} = {1}
        # Trace from 1: current_reach=[1], then nxt=2, 2 is not in visited, upstream_count[2]=1 so no confluence
        # current_reach = [1, 2], visited={1, 2}, node=2
        # node=2 in downstream? No (only 1 is in downstream). So final reach [1, 2] has len >= 2.
        # _add_node(1) works, _add_node(2) fails because 2 not in valid_nodes.
        # Result: nodes_data has 1 node, reaches_data has [[1,2]]
        assert result is not None
        nodes_data, reaches_data = result
        assert len(reaches_data) == 1
        # Only node 1 appears in nodes_data (node 2 has invalid gw_node)
        assert len(nodes_data) == 1
        assert nodes_data[0].id == 1


class TestBuildReachesFromPreprocessorBinary:
    """Tests for _build_reaches_from_preprocessor_binary."""

    def test_no_boundaries_returns_none(self):
        """When model_state.get_stream_reach_boundaries returns None."""
        stream = _make_mock_stream({})
        grid = _make_grid()
        with patch.object(model_state, "get_stream_reach_boundaries", return_value=None):
            result = _build_reaches_from_preprocessor_binary(stream, grid, None, {})
        assert result is None

    def test_empty_boundaries_returns_none(self):
        """When model_state.get_stream_reach_boundaries returns []."""
        stream = _make_mock_stream({})
        grid = _make_grid()
        with patch.object(model_state, "get_stream_reach_boundaries", return_value=[]):
            result = _build_reaches_from_preprocessor_binary(stream, grid, None, {})
        assert result is None

    def test_no_valid_nodes_returns_none(self):
        """All stream nodes have invalid gw_nodes."""
        sn1 = _make_strm_node(id=1, gw_node=999)
        stream = _make_mock_stream({1: sn1})
        grid = _make_grid()
        with patch.object(
            model_state,
            "get_stream_reach_boundaries",
            return_value=[(1, 1, 3)],
        ):
            result = _build_reaches_from_preprocessor_binary(stream, grid, None, {})
        assert result is None

    def test_valid_boundaries(self):
        """Boundaries (reach_id=1, up=1, dn=3) with nodes 1, 2, 3."""
        sn1 = _make_strm_node(id=1, gw_node=1, reach_id=1)
        sn2 = _make_strm_node(id=2, gw_node=2, reach_id=1)
        sn3 = _make_strm_node(id=3, gw_node=3, reach_id=1)
        stream = _make_mock_stream({1: sn1, 2: sn2, 3: sn3})
        grid = _make_grid()
        with patch.object(
            model_state,
            "get_stream_reach_boundaries",
            return_value=[(1, 1, 3)],
        ):
            result = _build_reaches_from_preprocessor_binary(stream, grid, None, {})
        assert result is not None
        nodes_data, reaches_data = result
        assert len(reaches_data) == 1
        assert reaches_data[0] == [1, 2, 3]
        assert len(nodes_data) == 3

    def test_boundary_with_less_than_2_nodes_skipped(self):
        """A boundary range that yields fewer than 2 valid nodes is skipped."""
        sn5 = _make_strm_node(id=5, gw_node=5)
        stream = _make_mock_stream({5: sn5})
        grid = _make_grid()
        # Boundary (1, 5, 5) yields only 1 node
        with patch.object(
            model_state,
            "get_stream_reach_boundaries",
            return_value=[(1, 5, 5)],
        ):
            result = _build_reaches_from_preprocessor_binary(stream, grid, None, {})
        assert result is None  # No nodes_data emitted, so returns None

    def test_multiple_boundaries(self):
        """Two reach boundaries."""
        sn1 = _make_strm_node(id=1, gw_node=1)
        sn2 = _make_strm_node(id=2, gw_node=2)
        sn3 = _make_strm_node(id=3, gw_node=3)
        sn4 = _make_strm_node(id=4, gw_node=4)
        sn5 = _make_strm_node(id=5, gw_node=5)
        stream = _make_mock_stream({1: sn1, 2: sn2, 3: sn3, 4: sn4, 5: sn5})
        grid = _make_grid()
        with patch.object(
            model_state,
            "get_stream_reach_boundaries",
            return_value=[(1, 1, 3), (2, 4, 5)],
        ):
            result = _build_reaches_from_preprocessor_binary(stream, grid, None, {})
        assert result is not None
        nodes_data, reaches_data = result
        assert len(reaches_data) == 2
        assert reaches_data[0] == [1, 2, 3]
        assert reaches_data[1] == [4, 5]


class TestBuildStreamsFromNodes:
    """Tests for _build_streams_from_nodes."""

    def test_groups_by_reach_id(self):
        sn1 = _make_strm_node(id=1, gw_node=1, reach_id=1)
        sn2 = _make_strm_node(id=2, gw_node=2, reach_id=1)
        sn3 = _make_strm_node(id=3, gw_node=3, reach_id=2)
        sn4 = _make_strm_node(id=4, gw_node=4, reach_id=2)
        stream = _make_mock_stream({1: sn1, 2: sn2, 3: sn3, 4: sn4})
        grid = _make_grid()
        nodes_data, reaches_data = _build_streams_from_nodes(stream, grid, None, {})
        assert len(reaches_data) == 2
        assert len(nodes_data) == 4

    def test_single_node_reach_excluded(self):
        """A reach with only one valid node has < 2 nodes, so it's excluded."""
        sn1 = _make_strm_node(id=1, gw_node=1, reach_id=1)
        sn2 = _make_strm_node(id=2, gw_node=2, reach_id=2)
        sn3 = _make_strm_node(id=3, gw_node=3, reach_id=2)
        stream = _make_mock_stream({1: sn1, 2: sn2, 3: sn3})
        grid = _make_grid()
        nodes_data, reaches_data = _build_streams_from_nodes(stream, grid, None, {})
        # Reach 1 has only 1 node, reach 2 has 2 nodes
        assert len(reaches_data) == 1
        assert reaches_data[0] == [2, 3]

    def test_invalid_gw_node_excluded(self):
        """Nodes with gw_node not in grid are excluded."""
        sn1 = _make_strm_node(id=1, gw_node=999, reach_id=1)
        sn2 = _make_strm_node(id=2, gw_node=2, reach_id=1)
        stream = _make_mock_stream({1: sn1, 2: sn2})
        grid = _make_grid()
        nodes_data, reaches_data = _build_streams_from_nodes(stream, grid, None, {})
        # Only 1 valid node in reach 1 -> excluded
        assert len(reaches_data) == 0


class TestBuildStreamsFromReaches:
    """Tests for _build_streams_from_reaches."""

    def test_reaches_dict_with_int_node_ids(self):
        """Reaches with integer node IDs resolved via stream.nodes."""
        sn1 = _make_strm_node(id=1, gw_node=1)
        sn2 = _make_strm_node(id=2, gw_node=2)
        reach = MagicMock()
        reach.id = 1
        reach.stream_nodes = None
        reach.nodes = [1, 2]  # int node IDs
        stream = _make_mock_stream({1: sn1, 2: sn2}, reaches={1: reach})
        grid = _make_grid()
        nodes_data, reaches_data = _build_streams_from_reaches(stream, grid, None, {})
        assert len(reaches_data) == 1
        assert len(nodes_data) == 2

    def test_reaches_dict_with_strm_node_objects(self):
        """Reaches with StrmNode objects directly."""
        sn1 = _make_strm_node(id=1, gw_node=1)
        sn2 = _make_strm_node(id=2, gw_node=2)
        reach = MagicMock()
        reach.id = 1
        reach.stream_nodes = [sn1, sn2]
        stream = _make_mock_stream({1: sn1, 2: sn2}, reaches={1: reach})
        grid = _make_grid()
        nodes_data, reaches_data = _build_streams_from_reaches(stream, grid, None, {})
        assert len(reaches_data) == 1
        assert len(nodes_data) == 2

    def test_reaches_list(self):
        """Reaches as a list instead of dict."""
        sn1 = _make_strm_node(id=1, gw_node=1)
        sn2 = _make_strm_node(id=2, gw_node=2)
        reach = MagicMock()
        reach.id = 1
        reach.stream_nodes = [sn1, sn2]
        stream = _make_mock_stream({1: sn1, 2: sn2}, reaches=[reach])
        grid = _make_grid()
        nodes_data, reaches_data = _build_streams_from_reaches(stream, grid, None, {})
        assert len(reaches_data) == 1

    def test_empty_reach_excluded(self):
        """Reach with no valid nodes produces no reach_data entry."""
        sn1 = _make_strm_node(id=1, gw_node=999)  # invalid gw
        reach = MagicMock()
        reach.id = 1
        reach.stream_nodes = [sn1]
        stream = _make_mock_stream({1: sn1}, reaches={1: reach})
        grid = _make_grid()
        nodes_data, reaches_data = _build_streams_from_reaches(stream, grid, None, {})
        assert len(reaches_data) == 0
        assert len(nodes_data) == 0

    def test_int_node_id_not_in_stream_nodes_skipped(self):
        """Int node ID not found in stream.nodes is skipped."""
        reach = MagicMock()
        reach.id = 1
        reach.stream_nodes = None
        reach.nodes = [99, 100]  # IDs not in stream.nodes
        stream = _make_mock_stream({}, reaches={1: reach})
        grid = _make_grid()
        nodes_data, reaches_data = _build_streams_from_reaches(stream, grid, None, {})
        assert len(reaches_data) == 0


class TestBuildStreamData:
    """Tests for _build_stream_data priority logic."""

    def test_strategy_1_populated_reaches(self):
        """Strategy 1: populated reaches are used first."""
        sn1 = _make_strm_node(id=1, gw_node=1)
        sn2 = _make_strm_node(id=2, gw_node=2)
        reach = MagicMock()
        reach.id = 1
        reach.stream_nodes = [sn1, sn2]
        stream = _make_mock_stream({1: sn1, 2: sn2}, reaches={1: reach})
        grid = _make_grid()
        nodes_data, reaches_data = _build_stream_data(stream, grid, None, {})
        assert len(nodes_data) == 2
        assert len(reaches_data) == 1

    def test_strategy_2_preprocessor_binary(self):
        """Strategy 2: preprocessor binary boundaries."""
        sn1 = _make_strm_node(id=1, gw_node=1)
        sn2 = _make_strm_node(id=2, gw_node=2)
        sn3 = _make_strm_node(id=3, gw_node=3)
        stream = _make_mock_stream({1: sn1, 2: sn2, 3: sn3})
        # No reaches populated
        stream.reaches = {}
        grid = _make_grid()
        with patch.object(
            model_state,
            "get_stream_reach_boundaries",
            return_value=[(1, 1, 3)],
        ):
            nodes_data, reaches_data = _build_stream_data(stream, grid, None, {})
        assert len(reaches_data) == 1
        assert len(nodes_data) == 3

    def test_strategy_3_connectivity(self):
        """Strategy 3: connectivity tracing (no boundaries, no reaches)."""
        sn1 = _make_strm_node(id=1, gw_node=1, downstream_node=2)
        sn2 = _make_strm_node(id=2, gw_node=2, downstream_node=3)
        sn3 = _make_strm_node(id=3, gw_node=3, downstream_node=None)
        stream = _make_mock_stream({1: sn1, 2: sn2, 3: sn3})
        stream.reaches = {}
        grid = _make_grid()
        with patch.object(
            model_state,
            "get_stream_reach_boundaries",
            return_value=None,
        ):
            nodes_data, reaches_data = _build_stream_data(stream, grid, None, {})
        assert len(nodes_data) >= 3
        assert len(reaches_data) >= 1

    def test_strategy_4_reach_id_grouping(self):
        """Strategy 4: reach_id grouping (non-zero reach_ids)."""
        sn1 = _make_strm_node(id=1, gw_node=1, reach_id=1, downstream_node=None)
        sn2 = _make_strm_node(id=2, gw_node=2, reach_id=1, downstream_node=None)
        stream = _make_mock_stream({1: sn1, 2: sn2})
        stream.reaches = {}
        grid = _make_grid()
        with patch.object(
            model_state,
            "get_stream_reach_boundaries",
            return_value=None,
        ):
            nodes_data, reaches_data = _build_stream_data(stream, grid, None, {})
        assert len(nodes_data) == 2
        assert len(reaches_data) == 1

    def test_strategy_5_single_reach_fallback(self):
        """Strategy 5: all nodes in one reach when reach_ids are all 0."""
        sn1 = _make_strm_node(id=1, gw_node=1, reach_id=0, downstream_node=None)
        sn2 = _make_strm_node(id=2, gw_node=2, reach_id=0, downstream_node=None)
        sn3 = _make_strm_node(id=3, gw_node=3, reach_id=0, downstream_node=None)
        stream = _make_mock_stream({1: sn1, 2: sn2, 3: sn3})
        stream.reaches = {}
        grid = _make_grid()
        with patch.object(
            model_state,
            "get_stream_reach_boundaries",
            return_value=None,
        ):
            nodes_data, reaches_data = _build_stream_data(stream, grid, None, {})
        assert len(nodes_data) == 3
        assert len(reaches_data) == 1  # single reach with all nodes

    def test_no_stream_nodes_returns_empty(self):
        """When stream has no nodes and no reaches, return empty."""
        stream = MagicMock()
        stream.reaches = {}
        stream.nodes = {}
        grid = _make_grid()
        nodes_data, reaches_data = _build_stream_data(stream, grid, None, {})
        assert nodes_data == []
        assert reaches_data == []

    def test_strategy_5_single_node_no_reach(self):
        """Strategy 5 with only 1 valid node produces no reaches (needs >= 2)."""
        sn1 = _make_strm_node(id=1, gw_node=1, reach_id=0, downstream_node=None)
        stream = _make_mock_stream({1: sn1})
        stream.reaches = {}
        grid = _make_grid()
        with patch.object(
            model_state,
            "get_stream_reach_boundaries",
            return_value=None,
        ):
            nodes_data, reaches_data = _build_stream_data(stream, grid, None, {})
        assert len(nodes_data) == 1
        assert len(reaches_data) == 0


class TestGetGwNodesForReaches:
    """Tests for _get_gw_nodes_for_reaches."""

    def test_basic_reach(self):
        sn1 = _make_strm_node(id=1, gw_node=1, reach_id=1)
        sn2 = _make_strm_node(id=2, gw_node=2, reach_id=1)
        reach = MagicMock()
        reach.id = 1
        reach.stream_nodes = [sn1, sn2]
        stream = _make_mock_stream({1: sn1, 2: sn2}, reaches={1: reach})
        grid = _make_grid()
        result = _get_gw_nodes_for_reaches(stream, grid, None, {})
        assert len(result) == 1
        rid, name, gw_nodes = result[0]
        assert rid == 1
        assert "Reach 1" in name
        assert 1 in gw_nodes
        assert 2 in gw_nodes

    def test_reach_id_zero_name_format(self):
        """When reach_id=0, name uses node range format."""
        sn1 = _make_strm_node(id=10, gw_node=1, reach_id=0)
        sn2 = _make_strm_node(id=20, gw_node=2, reach_id=0)
        stream = _make_mock_stream({10: sn1, 20: sn2})
        stream.reaches = {}
        grid = _make_grid()
        with patch.object(
            model_state,
            "get_stream_reach_boundaries",
            return_value=None,
        ):
            result = _get_gw_nodes_for_reaches(stream, grid, None, {})
        if result:
            rid, name, gw_nodes = result[0]
            assert "nodes" in name


# ===========================================================================
# Endpoint tests: GET /api/streams
# ===========================================================================


class TestGetStreams:
    """Tests for GET /api/streams."""

    def test_no_model_returns_404(self, client_no_model):
        resp = client_no_model.get("/api/streams")
        assert resp.status_code == 404
        assert "No model loaded" in resp.json()["detail"]

    def test_no_streams_returns_404(self, client_model_no_streams):
        resp = client_model_no_streams.get("/api/streams")
        assert resp.status_code == 404
        assert "No stream data" in resp.json()["detail"]

    def test_valid_stream_network(self):
        _reset_model_state()
        try:
            grid = _make_grid()
            sn1 = _make_strm_node(id=1, gw_node=1, reach_id=1)
            sn2 = _make_strm_node(id=2, gw_node=2, reach_id=1)
            reach = MagicMock()
            reach.id = 1
            reach.stream_nodes = [sn1, sn2]
            stream = _make_mock_stream({1: sn1, 2: sn2}, reaches={1: reach})
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/streams")
            assert resp.status_code == 200
            data = resp.json()
            assert data["n_nodes"] == 2
            assert data["n_reaches"] == 1
            assert len(data["nodes"]) == 2
            assert len(data["reaches"]) == 1
        finally:
            _reset_model_state()


# ===========================================================================
# Endpoint tests: GET /api/streams/geojson
# ===========================================================================


class TestGetStreamsGeojson:
    """Tests for GET /api/streams/geojson."""

    def test_no_model_returns_404(self, client_no_model):
        resp = client_no_model.get("/api/streams/geojson")
        assert resp.status_code == 404

    def test_no_streams_returns_empty_feature_collection(self, client_model_no_streams):
        resp = client_model_no_streams.get("/api/streams/geojson")
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "FeatureCollection"
        assert data["features"] == []

    def test_valid_geojson(self):
        _reset_model_state()
        try:
            grid = _make_grid()
            sn1 = _make_strm_node(id=1, gw_node=1, reach_id=1)
            sn2 = _make_strm_node(id=2, gw_node=2, reach_id=1)
            sn3 = _make_strm_node(id=3, gw_node=3, reach_id=1)
            reach = MagicMock()
            reach.id = 1
            reach.stream_nodes = [sn1, sn2, sn3]
            stream = _make_mock_stream(
                {1: sn1, 2: sn2, 3: sn3},
                reaches={1: reach},
            )
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/streams/geojson")
            assert resp.status_code == 200
            data = resp.json()
            assert data["type"] == "FeatureCollection"
            assert len(data["features"]) == 1
            feature = data["features"][0]
            assert feature["geometry"]["type"] == "LineString"
            assert len(feature["geometry"]["coordinates"]) >= 2
            assert feature["properties"]["reach_id"] == 1
            assert "n_nodes" in feature["properties"]
        finally:
            _reset_model_state()


# ===========================================================================
# Endpoint tests: GET /api/streams/diversions
# ===========================================================================


class TestGetDiversions:
    """Tests for GET /api/streams/diversions."""

    def test_no_model_returns_404(self, client_no_model):
        resp = client_no_model.get("/api/streams/diversions")
        assert resp.status_code == 404

    def test_no_streams_returns_empty(self, client_model_no_streams):
        resp = client_model_no_streams.get("/api/streams/diversions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_diversions"] == 0
        assert data["diversions"] == []

    def test_no_diversions_returns_empty(self):
        _reset_model_state()
        try:
            grid = _make_grid()
            stream = _make_mock_stream()
            stream.diversions = {}
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/streams/diversions")
            assert resp.status_code == 200
            data = resp.json()
            assert data["n_diversions"] == 0
        finally:
            _reset_model_state()

    def test_no_diversions_attr(self):
        """Stream has no diversions attribute."""
        _reset_model_state()
        try:
            grid = _make_grid()
            stream = MagicMock(spec=[])  # no attributes
            stream.nodes = {}
            stream.reaches = {}
            # hasattr(stream, "diversions") will be False
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/streams/diversions")
            assert resp.status_code == 200
            data = resp.json()
            assert data["n_diversions"] == 0
        finally:
            _reset_model_state()

    def test_diversion_element_destination(self):
        """Diversion with destination_type='element' computes centroid."""
        _reset_model_state()
        try:
            grid = _make_grid()
            sn1 = _make_strm_node(id=1, gw_node=1)
            div = _make_mock_diversion(
                source_node=1,
                destination_type="element",
                destination_id=1,
                name="Div A",
                max_rate=50.0,
                priority=2,
            )
            stream = _make_mock_stream(
                nodes_dict={1: sn1},
                diversions={1: div},
            )
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/streams/diversions")
            assert resp.status_code == 200
            data = resp.json()
            assert data["n_diversions"] == 1
            d = data["diversions"][0]
            assert d["id"] == 1
            assert d["name"] == "Div A"
            assert d["source"] is not None
            assert d["source"]["lng"] == 0.0  # node 1 at (0, 0)
            assert d["destination"] is not None
            # Centroid of element 1 (vertices 1,2,5,4 -> (0+100+100+0)/4=50, (0+0+100+100)/4=50)
            assert d["destination"]["lng"] == pytest.approx(50.0)
            assert d["destination"]["lat"] == pytest.approx(50.0)
            assert d["max_rate"] == 50.0
            assert d["priority"] == 2
        finally:
            _reset_model_state()

    def test_diversion_stream_node_destination(self):
        """Diversion with destination_type='stream_node'."""
        _reset_model_state()
        try:
            grid = _make_grid()
            sn1 = _make_strm_node(id=1, gw_node=1)
            sn2 = _make_strm_node(id=2, gw_node=3)
            div = _make_mock_diversion(
                source_node=1,
                destination_type="stream_node",
                destination_id=2,
                name="Stream Div",
            )
            stream = _make_mock_stream(
                nodes_dict={1: sn1, 2: sn2},
                diversions={1: div},
            )
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/streams/diversions")
            data = resp.json()
            d = data["diversions"][0]
            assert d["destination_type"] == "stream_node"
            assert d["destination"] is not None
            # sn2 -> gw_node=3 -> node 3 at (200, 0)
            assert d["destination"]["lng"] == pytest.approx(200.0)
            assert d["destination"]["lat"] == pytest.approx(0.0)
        finally:
            _reset_model_state()

    def test_diversion_max_rate_large_becomes_none(self):
        """max_rate >= 1e30 should become None in the response."""
        _reset_model_state()
        try:
            grid = _make_grid()
            sn1 = _make_strm_node(id=1, gw_node=1)
            div = _make_mock_diversion(
                source_node=1,
                destination_type="element",
                destination_id=1,
                max_rate=1e31,
            )
            stream = _make_mock_stream(
                nodes_dict={1: sn1},
                diversions={1: div},
            )
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/streams/diversions")
            data = resp.json()
            assert data["diversions"][0]["max_rate"] is None
        finally:
            _reset_model_state()

    def test_diversion_source_not_in_grid(self):
        """When source node gw_node is not in the grid, source is None."""
        _reset_model_state()
        try:
            grid = _make_grid()
            sn1 = _make_strm_node(id=1, gw_node=999)  # invalid gw_node
            div = _make_mock_diversion(
                source_node=1,
                destination_type="element",
                destination_id=1,
            )
            stream = _make_mock_stream(
                nodes_dict={1: sn1},
                diversions={1: div},
            )
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/streams/diversions")
            data = resp.json()
            assert data["diversions"][0]["source"] is None
        finally:
            _reset_model_state()

    def test_diversion_source_node_zero(self):
        """source_node=0 means outside model area, source coords are None."""
        _reset_model_state()
        try:
            grid = _make_grid()
            div = _make_mock_diversion(
                source_node=0,
                destination_type="element",
                destination_id=1,
            )
            stream = _make_mock_stream(
                nodes_dict={},
                diversions={1: div},
            )
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/streams/diversions")
            data = resp.json()
            assert data["diversions"][0]["source"] is None
        finally:
            _reset_model_state()

    def test_diversion_element_not_in_grid(self):
        """destination_type='element' but destination_id not in grid -> dest None."""
        _reset_model_state()
        try:
            grid = _make_grid()
            sn1 = _make_strm_node(id=1, gw_node=1)
            div = _make_mock_diversion(
                source_node=1,
                destination_type="element",
                destination_id=999,  # not in grid
            )
            stream = _make_mock_stream(
                nodes_dict={1: sn1},
                diversions={1: div},
            )
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/streams/diversions")
            data = resp.json()
            assert data["diversions"][0]["destination"] is None
        finally:
            _reset_model_state()

    def test_diversion_no_name_uses_default(self):
        """When div.name is None, should use 'Diversion {id}'."""
        _reset_model_state()
        try:
            grid = _make_grid()
            sn1 = _make_strm_node(id=1, gw_node=1)
            div = _make_mock_diversion(
                source_node=1,
                destination_type="element",
                destination_id=1,
                name=None,
            )
            stream = _make_mock_stream(
                nodes_dict={1: sn1},
                diversions={42: div},
            )
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/streams/diversions")
            data = resp.json()
            assert data["diversions"][0]["name"] == "Diversion 42"
        finally:
            _reset_model_state()

    def test_diversion_empty_name_uses_default(self):
        """When div.name is empty string, should use 'Diversion {id}'."""
        _reset_model_state()
        try:
            grid = _make_grid()
            sn1 = _make_strm_node(id=1, gw_node=1)
            div = _make_mock_diversion(
                source_node=1,
                destination_type="element",
                destination_id=1,
                name="",
            )
            stream = _make_mock_stream(
                nodes_dict={1: sn1},
                diversions={5: div},
            )
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/streams/diversions")
            data = resp.json()
            assert data["diversions"][0]["name"] == "Diversion 5"
        finally:
            _reset_model_state()

    def test_diversion_unknown_destination_type(self):
        """Unknown destination_type -> destination coords are None."""
        _reset_model_state()
        try:
            grid = _make_grid()
            sn1 = _make_strm_node(id=1, gw_node=1)
            div = _make_mock_diversion(
                source_node=1,
                destination_type="unknown",
                destination_id=1,
            )
            stream = _make_mock_stream(
                nodes_dict={1: sn1},
                diversions={1: div},
            )
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/streams/diversions")
            data = resp.json()
            assert data["diversions"][0]["destination"] is None
        finally:
            _reset_model_state()

    def test_diversion_from_reaches_not_nodes(self):
        """When stream.nodes is empty but reaches have StrmNode objects, build sn_to_gw from reaches."""
        _reset_model_state()
        try:
            grid = _make_grid()
            sn1 = _make_strm_node(id=1, gw_node=1)
            reach = MagicMock()
            reach.id = 1
            reach.stream_nodes = [sn1]
            reach.nodes = None
            div = _make_mock_diversion(
                source_node=1,
                destination_type="element",
                destination_id=1,
            )
            stream = MagicMock()
            stream.nodes = {}  # empty nodes dict
            stream.reaches = {1: reach}
            stream.diversions = {1: div}
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/streams/diversions")
            data = resp.json()
            assert data["n_diversions"] == 1
            # Source should be resolved from reach nodes
            assert data["diversions"][0]["source"] is not None
        finally:
            _reset_model_state()

    def test_multiple_diversions(self):
        """Multiple diversions returned in order."""
        _reset_model_state()
        try:
            grid = _make_grid()
            sn1 = _make_strm_node(id=1, gw_node=1)
            sn2 = _make_strm_node(id=2, gw_node=2)
            div1 = _make_mock_diversion(
                source_node=1,
                destination_type="element",
                destination_id=1,
                name="Div 1",
            )
            div2 = _make_mock_diversion(
                source_node=2,
                destination_type="element",
                destination_id=2,
                name="Div 2",
            )
            stream = _make_mock_stream(
                nodes_dict={1: sn1, 2: sn2},
                diversions={1: div1, 2: div2},
            )
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/streams/diversions")
            data = resp.json()
            assert data["n_diversions"] == 2
        finally:
            _reset_model_state()


# ===========================================================================
# Endpoint tests: GET /api/streams/diversions/{div_id}
# ===========================================================================


class TestGetDiversionDetail:
    """Tests for GET /api/streams/diversions/{div_id}."""

    def test_no_model_returns_404(self, client_no_model):
        resp = client_no_model.get("/api/streams/diversions/1")
        assert resp.status_code == 404
        assert "No model loaded" in resp.json()["detail"]

    def test_no_streams_returns_404(self, client_model_no_streams):
        resp = client_model_no_streams.get("/api/streams/diversions/1")
        assert resp.status_code == 404
        assert "No stream data" in resp.json()["detail"]

    def test_no_diversions_returns_404(self):
        _reset_model_state()
        try:
            grid = _make_grid()
            stream = _make_mock_stream()
            stream.diversions = {}
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/streams/diversions/1")
            assert resp.status_code == 404
            assert "No diversions" in resp.json()["detail"]
        finally:
            _reset_model_state()

    def test_no_diversions_attr_returns_404(self):
        """Stream has no diversions attribute at all."""
        _reset_model_state()
        try:
            grid = _make_grid()
            stream = MagicMock(spec=[])
            stream.nodes = {}
            stream.reaches = {}
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/streams/diversions/1")
            assert resp.status_code == 404
            assert "No diversions" in resp.json()["detail"]
        finally:
            _reset_model_state()

    def test_div_not_found_returns_404(self):
        _reset_model_state()
        try:
            grid = _make_grid()
            div1 = _make_mock_diversion(source_node=1, destination_type="element", destination_id=1)
            stream = _make_mock_stream(diversions={1: div1})
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/streams/diversions/999")
            assert resp.status_code == 404
            assert "999 not found" in resp.json()["detail"]
        finally:
            _reset_model_state()

    def test_element_destination_detail(self):
        """Detail for element destination with delivery area."""
        _reset_model_state()
        try:
            grid = _make_grid()
            sn1 = _make_strm_node(id=1, gw_node=1)
            div = _make_mock_diversion(
                source_node=1,
                destination_type="element",
                destination_id=1,
                name="Detail Div",
                max_rate=75.0,
                priority=3,
            )
            stream = _make_mock_stream(
                nodes_dict={1: sn1},
                diversions={1: div},
            )
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            with patch.object(model_state, "get_diversion_timeseries", return_value=None):
                app = create_app()
                client = TestClient(app)

                resp = client.get("/api/streams/diversions/1")
            assert resp.status_code == 200
            data = resp.json()
            assert data["id"] == 1
            assert data["name"] == "Detail Div"
            assert data["max_rate"] == 75.0
            assert data["priority"] == 3
            assert data["source"] is not None
            assert data["destination"] is not None
            # delivery area
            assert data["delivery"]["dest_type"] == "element"
            assert data["delivery"]["element_ids"] == [1]
            assert data["delivery"]["element_polygons"] is not None
            assert data["delivery"]["element_polygons"]["type"] == "FeatureCollection"
            assert len(data["delivery"]["element_polygons"]["features"]) == 1
            feat = data["delivery"]["element_polygons"]["features"][0]
            assert feat["geometry"]["type"] == "Polygon"
            # Ring should be closed
            ring = feat["geometry"]["coordinates"][0]
            assert ring[0] == ring[-1]
            assert data["timeseries"] is None
        finally:
            _reset_model_state()

    def test_element_set_destination(self):
        """Destination type 'element_set' uses diversion_element_groups."""
        _reset_model_state()
        try:
            grid = _make_grid()
            sn1 = _make_strm_node(id=1, gw_node=1)
            div = _make_mock_diversion(
                source_node=1,
                destination_type="element_set",
                destination_id=10,
                delivery_dest_id=10,
            )
            eg = MagicMock()
            eg.id = 10
            eg.elements = [1, 2]
            stream = _make_mock_stream(
                nodes_dict={1: sn1},
                diversions={1: div},
            )
            stream.diversion_element_groups = [eg]
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            with patch.object(model_state, "get_diversion_timeseries", return_value=None):
                app = create_app()
                client = TestClient(app)
                resp = client.get("/api/streams/diversions/1")
            assert resp.status_code == 200
            data = resp.json()
            assert data["delivery"]["dest_type"] == "element_set"
            assert data["delivery"]["element_ids"] == [1, 2]
            assert data["delivery"]["element_polygons"] is not None
            assert len(data["delivery"]["element_polygons"]["features"]) == 2
        finally:
            _reset_model_state()

    def test_subregion_destination(self):
        """Destination type 'subregion' collects all elements in that subregion."""
        _reset_model_state()
        try:
            grid = _make_grid()
            sn1 = _make_strm_node(id=1, gw_node=1)
            div = _make_mock_diversion(
                source_node=1,
                destination_type="subregion",
                destination_id=1,  # subregion 1
            )
            stream = _make_mock_stream(
                nodes_dict={1: sn1},
                diversions={1: div},
            )
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            with patch.object(model_state, "get_diversion_timeseries", return_value=None):
                app = create_app()
                client = TestClient(app)
                resp = client.get("/api/streams/diversions/1")
            assert resp.status_code == 200
            data = resp.json()
            assert data["delivery"]["dest_type"] == "subregion"
            # Both elements in grid have subregion=1
            assert set(data["delivery"]["element_ids"]) == {1, 2}
            assert data["delivery"]["element_polygons"] is not None
        finally:
            _reset_model_state()

    def test_stream_node_destination_detail(self):
        """Destination type 'stream_node' with coordinates."""
        _reset_model_state()
        try:
            grid = _make_grid()
            sn1 = _make_strm_node(id=1, gw_node=1)
            sn2 = _make_strm_node(id=2, gw_node=3)
            div = _make_mock_diversion(
                source_node=1,
                destination_type="stream_node",
                destination_id=2,
            )
            stream = _make_mock_stream(
                nodes_dict={1: sn1, 2: sn2},
                diversions={1: div},
            )
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            with patch.object(model_state, "get_diversion_timeseries", return_value=None):
                app = create_app()
                client = TestClient(app)
                resp = client.get("/api/streams/diversions/1")
            assert resp.status_code == 200
            data = resp.json()
            assert data["destination_type"] == "stream_node"
            assert data["destination"] is not None
            # No element_ids for stream_node destination
            assert data["delivery"]["element_ids"] == []
            assert data["delivery"]["element_polygons"] is None
        finally:
            _reset_model_state()

    def test_max_rate_large_becomes_none(self):
        """max_rate >= 1e30 returns None in detail."""
        _reset_model_state()
        try:
            grid = _make_grid()
            sn1 = _make_strm_node(id=1, gw_node=1)
            div = _make_mock_diversion(
                source_node=1,
                destination_type="element",
                destination_id=1,
                max_rate=2e30,
            )
            stream = _make_mock_stream(
                nodes_dict={1: sn1},
                diversions={1: div},
            )
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            with patch.object(model_state, "get_diversion_timeseries", return_value=None):
                app = create_app()
                client = TestClient(app)
                resp = client.get("/api/streams/diversions/1")
            assert resp.status_code == 200
            data = resp.json()
            assert data["max_rate"] is None
        finally:
            _reset_model_state()

    def test_source_not_in_grid_detail(self):
        """Source node with invalid gw_node -> source is None."""
        _reset_model_state()
        try:
            grid = _make_grid()
            sn1 = _make_strm_node(id=1, gw_node=999)
            div = _make_mock_diversion(
                source_node=1,
                destination_type="element",
                destination_id=1,
            )
            stream = _make_mock_stream(
                nodes_dict={1: sn1},
                diversions={1: div},
            )
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            with patch.object(model_state, "get_diversion_timeseries", return_value=None):
                app = create_app()
                client = TestClient(app)
                resp = client.get("/api/streams/diversions/1")
            assert resp.status_code == 200
            data = resp.json()
            assert data["source"] is None
        finally:
            _reset_model_state()

    def test_timeseries_with_max_div_and_delivery(self):
        """Timeseries with max_div_column and delivery_column populated."""
        _reset_model_state()
        try:
            grid = _make_grid()
            sn1 = _make_strm_node(id=1, gw_node=1)
            div = _make_mock_diversion(
                source_node=1,
                destination_type="element",
                destination_id=1,
                max_div_column=1,
                delivery_column=2,
                max_div_fraction=0.5,
                delivery_fraction=0.8,
            )
            stream = _make_mock_stream(
                nodes_dict={1: sn1},
                diversions={1: div},
            )
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)

            times = np.array(["2020-01-01", "2020-02-01", "2020-03-01"], dtype="datetime64")
            values = np.array(
                [
                    [100.0, 200.0],
                    [110.0, 220.0],
                    [120.0, 240.0],
                ]
            )
            meta = {}

            with patch.object(
                model_state,
                "get_diversion_timeseries",
                return_value=(times, values, meta),
            ):
                app = create_app()
                client = TestClient(app)
                resp = client.get("/api/streams/diversions/1")

            assert resp.status_code == 200
            data = resp.json()
            ts = data["timeseries"]
            assert ts is not None
            assert len(ts["times"]) == 3
            assert ts["max_diversion"] is not None
            assert len(ts["max_diversion"]) == 3
            # max_div_column=1 -> values[:,0] * 0.5
            assert ts["max_diversion"][0] == pytest.approx(50.0)
            assert ts["max_diversion"][1] == pytest.approx(55.0)
            assert ts["delivery"] is not None
            assert len(ts["delivery"]) == 3
            # delivery_column=2 -> values[:,1] * 0.8
            assert ts["delivery"][0] == pytest.approx(160.0)
            assert ts["delivery"][1] == pytest.approx(176.0)
        finally:
            _reset_model_state()

    def test_timeseries_no_columns(self):
        """When max_div_column=0 and delivery_column=0, timeseries is None."""
        _reset_model_state()
        try:
            grid = _make_grid()
            sn1 = _make_strm_node(id=1, gw_node=1)
            div = _make_mock_diversion(
                source_node=1,
                destination_type="element",
                destination_id=1,
                max_div_column=0,
                delivery_column=0,
            )
            stream = _make_mock_stream(
                nodes_dict={1: sn1},
                diversions={1: div},
            )
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)

            times = np.array(["2020-01-01"], dtype="datetime64")
            values = np.array([[100.0]])
            meta = {}

            with patch.object(
                model_state,
                "get_diversion_timeseries",
                return_value=(times, values, meta),
            ):
                app = create_app()
                client = TestClient(app)
                resp = client.get("/api/streams/diversions/1")

            assert resp.status_code == 200
            data = resp.json()
            assert data["timeseries"] is None
        finally:
            _reset_model_state()

    def test_timeseries_none(self):
        """When get_diversion_timeseries returns None."""
        _reset_model_state()
        try:
            grid = _make_grid()
            sn1 = _make_strm_node(id=1, gw_node=1)
            div = _make_mock_diversion(
                source_node=1,
                destination_type="element",
                destination_id=1,
            )
            stream = _make_mock_stream(
                nodes_dict={1: sn1},
                diversions={1: div},
            )
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            with patch.object(model_state, "get_diversion_timeseries", return_value=None):
                app = create_app()
                client = TestClient(app)
                resp = client.get("/api/streams/diversions/1")

            assert resp.status_code == 200
            data = resp.json()
            assert data["timeseries"] is None
        finally:
            _reset_model_state()

    def test_timeseries_1d_values(self):
        """Timeseries with 1D values array (single column)."""
        _reset_model_state()
        try:
            grid = _make_grid()
            sn1 = _make_strm_node(id=1, gw_node=1)
            div = _make_mock_diversion(
                source_node=1,
                destination_type="element",
                destination_id=1,
                max_div_column=1,
                delivery_column=0,
                max_div_fraction=1.0,
            )
            stream = _make_mock_stream(
                nodes_dict={1: sn1},
                diversions={1: div},
            )
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)

            times = np.array(["2020-01-01", "2020-02-01"], dtype="datetime64")
            values = np.array([50.0, 60.0])  # 1D array
            meta = {}

            with patch.object(
                model_state,
                "get_diversion_timeseries",
                return_value=(times, values, meta),
            ):
                app = create_app()
                client = TestClient(app)
                resp = client.get("/api/streams/diversions/1")

            assert resp.status_code == 200
            data = resp.json()
            ts = data["timeseries"]
            assert ts is not None
            assert ts["max_diversion"] is not None
            # 1D values with max_div_column=1, n_cols=1 -> uses values directly
            assert ts["max_diversion"][0] == pytest.approx(50.0)
        finally:
            _reset_model_state()

    def test_timeseries_only_max_div(self):
        """Only max_div_column set, delivery_column=0."""
        _reset_model_state()
        try:
            grid = _make_grid()
            sn1 = _make_strm_node(id=1, gw_node=1)
            div = _make_mock_diversion(
                source_node=1,
                destination_type="element",
                destination_id=1,
                max_div_column=1,
                delivery_column=0,
            )
            stream = _make_mock_stream(
                nodes_dict={1: sn1},
                diversions={1: div},
            )
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)

            times = np.array(["2020-01-01"], dtype="datetime64")
            values = np.array([[100.0, 200.0]])
            meta = {}

            with patch.object(
                model_state,
                "get_diversion_timeseries",
                return_value=(times, values, meta),
            ):
                app = create_app()
                client = TestClient(app)
                resp = client.get("/api/streams/diversions/1")

            assert resp.status_code == 200
            data = resp.json()
            ts = data["timeseries"]
            assert ts is not None
            assert ts["max_diversion"] is not None
            assert ts["delivery"] is None
        finally:
            _reset_model_state()

    def test_timeseries_only_delivery(self):
        """Only delivery_column set, max_div_column=0."""
        _reset_model_state()
        try:
            grid = _make_grid()
            sn1 = _make_strm_node(id=1, gw_node=1)
            div = _make_mock_diversion(
                source_node=1,
                destination_type="element",
                destination_id=1,
                max_div_column=0,
                delivery_column=1,
            )
            stream = _make_mock_stream(
                nodes_dict={1: sn1},
                diversions={1: div},
            )
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)

            times = np.array(["2020-01-01"], dtype="datetime64")
            values = np.array([[100.0, 200.0]])
            meta = {}

            with patch.object(
                model_state,
                "get_diversion_timeseries",
                return_value=(times, values, meta),
            ):
                app = create_app()
                client = TestClient(app)
                resp = client.get("/api/streams/diversions/1")

            assert resp.status_code == 200
            data = resp.json()
            ts = data["timeseries"]
            assert ts is not None
            assert ts["max_diversion"] is None
            assert ts["delivery"] is not None
        finally:
            _reset_model_state()

    def test_element_destination_not_in_grid_detail(self):
        """Element destination_id not in grid -> empty element_ids, no delivery GeoJSON."""
        _reset_model_state()
        try:
            grid = _make_grid()
            sn1 = _make_strm_node(id=1, gw_node=1)
            div = _make_mock_diversion(
                source_node=1,
                destination_type="element",
                destination_id=999,
            )
            stream = _make_mock_stream(
                nodes_dict={1: sn1},
                diversions={1: div},
            )
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            with patch.object(model_state, "get_diversion_timeseries", return_value=None):
                app = create_app()
                client = TestClient(app)
                resp = client.get("/api/streams/diversions/1")
            assert resp.status_code == 200
            data = resp.json()
            assert data["delivery"]["element_ids"] == []
            assert data["delivery"]["element_polygons"] is None
        finally:
            _reset_model_state()

    def test_element_set_missing_group(self):
        """element_set destination with no matching group -> empty element_ids."""
        _reset_model_state()
        try:
            grid = _make_grid()
            sn1 = _make_strm_node(id=1, gw_node=1)
            div = _make_mock_diversion(
                source_node=1,
                destination_type="element_set",
                destination_id=99,
                delivery_dest_id=99,
            )
            stream = _make_mock_stream(
                nodes_dict={1: sn1},
                diversions={1: div},
            )
            stream.diversion_element_groups = []
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            with patch.object(model_state, "get_diversion_timeseries", return_value=None):
                app = create_app()
                client = TestClient(app)
                resp = client.get("/api/streams/diversions/1")
            assert resp.status_code == 200
            data = resp.json()
            assert data["delivery"]["element_ids"] == []
            assert data["delivery"]["element_polygons"] is None
        finally:
            _reset_model_state()

    def test_no_name_uses_default_detail(self):
        """No name -> 'Diversion {id}'."""
        _reset_model_state()
        try:
            grid = _make_grid()
            sn1 = _make_strm_node(id=1, gw_node=1)
            div = _make_mock_diversion(
                source_node=1,
                destination_type="element",
                destination_id=1,
                name=None,
            )
            stream = _make_mock_stream(
                nodes_dict={1: sn1},
                diversions={7: div},
            )
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            with patch.object(model_state, "get_diversion_timeseries", return_value=None):
                app = create_app()
                client = TestClient(app)
                resp = client.get("/api/streams/diversions/7")
            assert resp.status_code == 200
            data = resp.json()
            assert data["name"] == "Diversion 7"
        finally:
            _reset_model_state()

    def test_subregion_no_matching_elements(self):
        """Subregion destination with no elements in that subregion."""
        _reset_model_state()
        try:
            grid = _make_grid()
            sn1 = _make_strm_node(id=1, gw_node=1)
            div = _make_mock_diversion(
                source_node=1,
                destination_type="subregion",
                destination_id=99,  # no elements have subregion=99
            )
            stream = _make_mock_stream(
                nodes_dict={1: sn1},
                diversions={1: div},
            )
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            with patch.object(model_state, "get_diversion_timeseries", return_value=None):
                app = create_app()
                client = TestClient(app)
                resp = client.get("/api/streams/diversions/1")
            assert resp.status_code == 200
            data = resp.json()
            assert data["delivery"]["element_ids"] == []
            assert data["delivery"]["element_polygons"] is None
        finally:
            _reset_model_state()


# ===========================================================================
# Endpoint tests: GET /api/streams/reach-profile
# ===========================================================================


class TestGetReachProfile:
    """Tests for GET /api/streams/reach-profile."""

    def test_no_model_returns_404(self, client_no_model):
        resp = client_no_model.get("/api/streams/reach-profile?reach_id=1")
        assert resp.status_code == 404
        assert "No model loaded" in resp.json()["detail"]

    def test_no_streams_returns_404(self, client_model_no_streams):
        resp = client_model_no_streams.get("/api/streams/reach-profile?reach_id=1")
        assert resp.status_code == 404
        assert "No stream data" in resp.json()["detail"]

    def test_reach_not_found_returns_404(self):
        """Reach not found anywhere -> 404."""
        _reset_model_state()
        try:
            grid = _make_grid()
            sn1 = _make_strm_node(id=1, gw_node=1, reach_id=1)
            stream = _make_mock_stream(nodes_dict={1: sn1})
            stream.reaches = {}
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/streams/reach-profile?reach_id=999")
            assert resp.status_code == 404
            assert "999 not found" in resp.json()["detail"]
        finally:
            _reset_model_state()

    def test_reach_from_reaches_dict(self):
        """Reach found in reaches dict."""
        _reset_model_state()
        try:
            grid = _make_grid()
            sn1 = _make_strm_node(id=1, gw_node=1, reach_id=1)
            sn2 = _make_strm_node(id=2, gw_node=2, reach_id=1)
            reach = MagicMock()
            reach.id = 1
            reach.name = "Test Reach"
            reach.stream_nodes = [sn1, sn2]
            stream = _make_mock_stream(
                nodes_dict={1: sn1, 2: sn2},
                reaches={1: reach},
            )
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/streams/reach-profile?reach_id=1")
            assert resp.status_code == 200
            data = resp.json()
            assert data["reach_id"] == 1
            assert data["name"] == "Test Reach"
            assert data["n_nodes"] == 2
            assert len(data["nodes"]) == 2
            # First node distance = 0
            assert data["nodes"][0]["distance"] == 0.0
            # Second node distance = Euclidean from (0,0) to (100,0)
            assert data["nodes"][1]["distance"] == pytest.approx(100.0)
            assert data["total_length"] == pytest.approx(100.0)
        finally:
            _reset_model_state()

    def test_reach_from_reaches_list(self):
        """Reach found in reaches list (not dict)."""
        _reset_model_state()
        try:
            grid = _make_grid()
            sn1 = _make_strm_node(id=1, gw_node=1, reach_id=1)
            sn2 = _make_strm_node(id=2, gw_node=2, reach_id=1)
            reach = MagicMock()
            reach.id = 1
            reach.name = "List Reach"
            reach.stream_nodes = [sn1, sn2]
            stream = _make_mock_stream(
                nodes_dict={1: sn1, 2: sn2},
                reaches=[reach],  # list, not dict
            )
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/streams/reach-profile?reach_id=1")
            assert resp.status_code == 200
            data = resp.json()
            assert data["name"] == "List Reach"
            assert data["n_nodes"] == 2
        finally:
            _reset_model_state()

    def test_reach_with_int_node_ids_resolved(self):
        """Reach stores int node IDs that must be resolved to StrmNode objects."""
        _reset_model_state()
        try:
            grid = _make_grid()
            sn1 = _make_strm_node(id=1, gw_node=1, reach_id=1)
            sn2 = _make_strm_node(id=2, gw_node=2, reach_id=1)
            reach = MagicMock()
            reach.id = 1
            reach.name = ""
            reach.stream_nodes = None
            reach.nodes = [1, 2]  # int IDs
            stream = _make_mock_stream(
                nodes_dict={1: sn1, 2: sn2},
                reaches={1: reach},
            )
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/streams/reach-profile?reach_id=1")
            assert resp.status_code == 200
            data = resp.json()
            assert data["n_nodes"] == 2
            assert data["name"] == "Reach 1"  # empty name -> default
        finally:
            _reset_model_state()

    def test_reach_fallback_to_nodes_by_reach_id(self):
        """No matching reach in dict -> fallback to nodes filtered by reach_id."""
        _reset_model_state()
        try:
            grid = _make_grid()
            sn1 = _make_strm_node(id=1, gw_node=1, reach_id=5)
            sn2 = _make_strm_node(id=2, gw_node=2, reach_id=5)
            sn3 = _make_strm_node(id=3, gw_node=3, reach_id=6)
            stream = _make_mock_stream(
                nodes_dict={1: sn1, 2: sn2, 3: sn3},
            )
            stream.reaches = {}
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/streams/reach-profile?reach_id=5")
            assert resp.status_code == 200
            data = resp.json()
            assert data["reach_id"] == 5
            assert data["n_nodes"] == 2
            # Only nodes with reach_id=5
            node_ids = {n["stream_node_id"] for n in data["nodes"]}
            assert node_ids == {1, 2}
        finally:
            _reset_model_state()

    def test_cumulative_distance_computation(self):
        """Verify cumulative Euclidean distance along stream nodes."""
        _reset_model_state()
        try:
            grid = _make_grid()
            # Nodes at (0,0), (100,0), (200,0) -> distances 0, 100, 200
            sn1 = _make_strm_node(id=1, gw_node=1, reach_id=1)
            sn2 = _make_strm_node(id=2, gw_node=2, reach_id=1)
            sn3 = _make_strm_node(id=3, gw_node=3, reach_id=1)
            reach = MagicMock()
            reach.id = 1
            reach.name = "Profile Reach"
            reach.stream_nodes = [sn1, sn2, sn3]
            stream = _make_mock_stream(
                nodes_dict={1: sn1, 2: sn2, 3: sn3},
                reaches={1: reach},
            )
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/streams/reach-profile?reach_id=1")
            assert resp.status_code == 200
            data = resp.json()
            assert data["nodes"][0]["distance"] == pytest.approx(0.0)
            assert data["nodes"][1]["distance"] == pytest.approx(100.0)
            assert data["nodes"][2]["distance"] == pytest.approx(200.0)
            assert data["total_length"] == pytest.approx(200.0)
        finally:
            _reset_model_state()

    def test_cross_section_attributes(self):
        """Stream nodes with cross_section have bed_elev and mannings_n from it."""
        _reset_model_state()
        try:
            grid = _make_grid()
            sn1 = _make_strm_node(id=1, gw_node=1, reach_id=1)
            xs = MagicMock()
            xs.bottom_elev = 42.0
            xs.n = 0.035
            sn1.cross_section = xs
            sn2 = _make_strm_node(id=2, gw_node=2, reach_id=1)
            sn2.cross_section = None
            sn2.bottom_elev = 38.0
            reach = MagicMock()
            reach.id = 1
            reach.name = "XS Reach"
            reach.stream_nodes = [sn1, sn2]
            stream = _make_mock_stream(
                nodes_dict={1: sn1, 2: sn2},
                reaches={1: reach},
            )
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/streams/reach-profile?reach_id=1")
            assert resp.status_code == 200
            data = resp.json()
            # Node 1: cross_section present
            assert data["nodes"][0]["bed_elev"] == pytest.approx(42.0)
            assert data["nodes"][0]["mannings_n"] == pytest.approx(0.035)
            # Node 2: no cross_section, uses bottom_elev attribute
            assert data["nodes"][1]["bed_elev"] == pytest.approx(38.0)
            assert data["nodes"][1]["mannings_n"] == pytest.approx(0.04)  # default
        finally:
            _reset_model_state()

    def test_conductivity_and_bed_thickness(self):
        """Verify conductivity and bed_thickness are returned."""
        _reset_model_state()
        try:
            grid = _make_grid()
            sn1 = _make_strm_node(id=1, gw_node=1, reach_id=1)
            sn1.conductivity = 0.05
            sn1.bed_thickness = 2.5
            reach = MagicMock()
            reach.id = 1
            reach.name = ""
            reach.stream_nodes = [sn1]
            # Need at least 1 node to avoid 404 - but it's only 1 node
            sn2 = _make_strm_node(id=2, gw_node=2, reach_id=1)
            reach.stream_nodes = [sn1, sn2]
            stream = _make_mock_stream(
                nodes_dict={1: sn1, 2: sn2},
                reaches={1: reach},
            )
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/streams/reach-profile?reach_id=1")
            assert resp.status_code == 200
            data = resp.json()
            assert data["nodes"][0]["conductivity"] == pytest.approx(0.05)
            assert data["nodes"][0]["bed_thickness"] == pytest.approx(2.5)
        finally:
            _reset_model_state()

    def test_ground_surface_elev_with_stratigraphy(self):
        """Profile with stratigraphy produces ground_surface_elev values."""
        _reset_model_state()
        try:
            grid = _make_grid()
            strat = _make_strat([100.0, 95.0, 90.0, 85.0, 80.0, 75.0])
            sn1 = _make_strm_node(id=1, gw_node=1, reach_id=1)
            sn2 = _make_strm_node(id=2, gw_node=2, reach_id=1)
            reach = MagicMock()
            reach.id = 1
            reach.name = "Strat Reach"
            reach.stream_nodes = [sn1, sn2]
            stream = _make_mock_stream(
                nodes_dict={1: sn1, 2: sn2},
                reaches={1: reach},
            )
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model.stratigraphy = strat
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/streams/reach-profile?reach_id=1")
            assert resp.status_code == 200
            data = resp.json()
            assert data["nodes"][0]["ground_surface_elev"] == pytest.approx(100.0)
            assert data["nodes"][1]["ground_surface_elev"] == pytest.approx(95.0)
        finally:
            _reset_model_state()

    def test_node_with_no_gw_node(self):
        """Stream node with gw_node=None uses default coordinates."""
        _reset_model_state()
        try:
            grid = _make_grid()
            sn1 = _make_strm_node(id=1, gw_node=1, reach_id=1)
            sn2 = MagicMock()
            sn2.id = 2
            sn2.groundwater_node = None
            sn2.gw_node = None
            sn2.reach_id = 1
            sn2.bottom_elev = 0.0
            sn2.cross_section = None
            sn2.conductivity = 0.0
            sn2.bed_thickness = 0.0
            reach = MagicMock()
            reach.id = 1
            reach.name = ""
            reach.stream_nodes = [sn1, sn2]
            stream = _make_mock_stream(
                nodes_dict={1: sn1, 2: sn2},
                reaches={1: reach},
            )
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/streams/reach-profile?reach_id=1")
            assert resp.status_code == 200
            data = resp.json()
            assert data["n_nodes"] == 2
            # Node 2 has no gw_node -> defaults
            node2 = data["nodes"][1]
            assert node2["gw_node_id"] is None
            assert node2["lng"] == 0.0
            assert node2["lat"] == 0.0
            assert node2["ground_surface_elev"] == 0.0
        finally:
            _reset_model_state()

    def test_diagonal_distance_computation(self):
        """Nodes at (0,0) and (0,100) -> distance = 100."""
        _reset_model_state()
        try:
            grid = _make_grid()
            # gw_node 1 at (0,0), gw_node 4 at (0,100)
            sn1 = _make_strm_node(id=1, gw_node=1, reach_id=1)
            sn2 = _make_strm_node(id=2, gw_node=4, reach_id=1)
            reach = MagicMock()
            reach.id = 1
            reach.name = ""
            reach.stream_nodes = [sn1, sn2]
            stream = _make_mock_stream(
                nodes_dict={1: sn1, 2: sn2},
                reaches={1: reach},
            )
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/streams/reach-profile?reach_id=1")
            assert resp.status_code == 200
            data = resp.json()
            # (0,0) to (0,100) = 100.0
            assert data["nodes"][1]["distance"] == pytest.approx(100.0)
            assert data["total_length"] == pytest.approx(100.0)
        finally:
            _reset_model_state()

    def test_reach_from_list_not_found_falls_back(self):
        """Reaches list with no matching ID -> falls back to nodes by reach_id."""
        _reset_model_state()
        try:
            grid = _make_grid()
            sn1 = _make_strm_node(id=1, gw_node=1, reach_id=5)
            sn2 = _make_strm_node(id=2, gw_node=2, reach_id=5)
            reach = MagicMock()
            reach.id = 3  # different reach_id
            reach.stream_nodes = [sn1]
            stream = _make_mock_stream(
                nodes_dict={1: sn1, 2: sn2},
                reaches=[reach],  # list
            )
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/streams/reach-profile?reach_id=5")
            assert resp.status_code == 200
            data = resp.json()
            assert data["reach_id"] == 5
            assert data["n_nodes"] == 2
        finally:
            _reset_model_state()

    def test_node_with_gw_node_not_in_grid(self):
        """Stream node with gw_node not in grid.nodes -> node = None, defaults used."""
        _reset_model_state()
        try:
            grid = _make_grid()
            sn1 = _make_strm_node(id=1, gw_node=1, reach_id=1)
            sn2 = _make_strm_node(id=2, gw_node=999, reach_id=1)  # not in grid
            reach = MagicMock()
            reach.id = 1
            reach.name = ""
            reach.stream_nodes = [sn1, sn2]
            stream = _make_mock_stream(
                nodes_dict={1: sn1, 2: sn2},
                reaches={1: reach},
            )
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/streams/reach-profile?reach_id=1")
            assert resp.status_code == 200
            data = resp.json()
            # Node 2 has gw_node=999 not in grid -> grid.nodes.get returns None -> x=0, y=0
            node2 = data["nodes"][1]
            assert node2["lng"] == 0.0
            assert node2["lat"] == 0.0
        finally:
            _reset_model_state()

    def test_missing_reach_id_param_returns_422(self):
        """Missing required reach_id query param returns 422."""
        _reset_model_state()
        try:
            grid = _make_grid()
            stream = _make_mock_stream()
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/streams/reach-profile")
            assert resp.status_code == 422
        finally:
            _reset_model_state()

    def test_profile_with_zero_bottom_elev(self):
        """bottom_elev=0.0 or None should not cause issues."""
        _reset_model_state()
        try:
            grid = _make_grid()
            sn1 = _make_strm_node(id=1, gw_node=1, reach_id=1)
            sn1.bottom_elev = None  # triggers the `or 0.0` branch
            sn2 = _make_strm_node(id=2, gw_node=2, reach_id=1)
            sn2.bottom_elev = 0.0
            reach = MagicMock()
            reach.id = 1
            reach.name = ""
            reach.stream_nodes = [sn1, sn2]
            stream = _make_mock_stream(
                nodes_dict={1: sn1, 2: sn2},
                reaches={1: reach},
            )
            model = _make_mock_model(grid=grid, streams=stream, has_streams=True)
            model_state._model = model
            model_state.reproject_coords = lambda x, y: (x, y)
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/streams/reach-profile?reach_id=1")
            assert resp.status_code == 200
            data = resp.json()
            assert data["nodes"][0]["bed_elev"] == 0.0
            assert data["nodes"][1]["bed_elev"] == 0.0
        finally:
            _reset_model_state()
