"""Pytest configuration and fixtures for pyiwfm tests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pyiwfm.core.mesh import AppGrid
    from pyiwfm.core.stratigraphy import Stratigraphy


@pytest.fixture
def fixtures_path() -> Path:
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def small_model_path(fixtures_path: Path) -> Path:
    """Return path to small test model directory."""
    return fixtures_path / "small_model"


@pytest.fixture
def single_node_data() -> dict:
    """Data for a single node."""
    return {
        "id": 1,
        "x": 100.0,
        "y": 200.0,
    }


@pytest.fixture
def single_element_data() -> dict:
    """Data for a single quadrilateral element."""
    return {
        "id": 1,
        "vertices": (1, 2, 3, 4),
        "subregion": 1,
    }


@pytest.fixture
def triangle_element_data() -> dict:
    """Data for a single triangular element."""
    return {
        "id": 1,
        "vertices": (1, 2, 3),
        "subregion": 1,
    }


@pytest.fixture
def small_grid_nodes() -> list[dict]:
    """
    Node data for a 3x3 grid (9 nodes).

    Layout:
        7---8---9
        |   |   |
        4---5---6
        |   |   |
        1---2---3

    Spacing: 100 units in both x and y directions.
    """
    return [
        {"id": 1, "x": 0.0, "y": 0.0},
        {"id": 2, "x": 100.0, "y": 0.0},
        {"id": 3, "x": 200.0, "y": 0.0},
        {"id": 4, "x": 0.0, "y": 100.0},
        {"id": 5, "x": 100.0, "y": 100.0},
        {"id": 6, "x": 200.0, "y": 100.0},
        {"id": 7, "x": 0.0, "y": 200.0},
        {"id": 8, "x": 100.0, "y": 200.0},
        {"id": 9, "x": 200.0, "y": 200.0},
    ]


@pytest.fixture
def small_grid_elements() -> list[dict]:
    """
    Element data for a 2x2 grid (4 quadrilateral elements).

    Layout (counter-clockwise vertex ordering):
        Element 3: (4,5,8,7)    Element 4: (5,6,9,8)
        Element 1: (1,2,5,4)    Element 2: (2,3,6,5)
    """
    return [
        {"id": 1, "vertices": (1, 2, 5, 4), "subregion": 1},
        {"id": 2, "vertices": (2, 3, 6, 5), "subregion": 1},
        {"id": 3, "vertices": (4, 5, 8, 7), "subregion": 2},
        {"id": 4, "vertices": (5, 6, 9, 8), "subregion": 2},
    ]


@pytest.fixture
def triangular_grid_nodes() -> list[dict]:
    """
    Node data for a simple 2-triangle mesh.

    Layout:
        3
       /|\
      / | \
     /  |  \
    1---2---4
    """
    return [
        {"id": 1, "x": 0.0, "y": 0.0},
        {"id": 2, "x": 100.0, "y": 0.0},
        {"id": 3, "x": 50.0, "y": 86.6},  # Equilateral height
        {"id": 4, "x": 200.0, "y": 0.0},
    ]


@pytest.fixture
def triangular_grid_elements() -> list[dict]:
    """
    Element data for a 2-triangle mesh.

    Layout (counter-clockwise vertex ordering):
        Triangle 1: (1,2,3)
        Triangle 2: (2,4,3)
    """
    return [
        {"id": 1, "vertices": (1, 2, 3), "subregion": 1},
        {"id": 2, "vertices": (2, 4, 3), "subregion": 1},
    ]


@pytest.fixture
def sample_stratigraphy_data() -> dict:
    """
    Stratigraphy data for a 2-layer model with 9 nodes.

    Ground surface elevations vary from 100 to 120.
    Layer 1: 0-50 depth
    Layer 2: 50-100 depth
    """
    n_nodes = 9
    n_layers = 2

    # Ground surface elevations
    gs_elev = np.array([100.0, 105.0, 110.0, 105.0, 110.0, 115.0, 110.0, 115.0, 120.0])

    # Layer top elevations (ground surface for layer 1)
    top_elev = np.zeros((n_nodes, n_layers))
    top_elev[:, 0] = gs_elev  # Layer 1 top = ground surface
    top_elev[:, 1] = gs_elev - 50.0  # Layer 2 top = 50 below ground

    # Layer bottom elevations
    bottom_elev = np.zeros((n_nodes, n_layers))
    bottom_elev[:, 0] = gs_elev - 50.0  # Layer 1 bottom
    bottom_elev[:, 1] = gs_elev - 100.0  # Layer 2 bottom

    # All nodes active in all layers
    active_node = np.ones((n_nodes, n_layers), dtype=bool)

    return {
        "n_layers": n_layers,
        "n_nodes": n_nodes,
        "gs_elev": gs_elev,
        "top_elev": top_elev,
        "bottom_elev": bottom_elev,
        "active_node": active_node,
    }


@pytest.fixture
def sample_subregions() -> list[dict]:
    """Subregion data for small grid."""
    return [
        {"id": 1, "name": "North Region", "elements": [1, 2]},
        {"id": 2, "name": "South Region", "elements": [3, 4]},
    ]


# Helper functions for tests


def assert_arrays_equal(a: np.ndarray, b: np.ndarray, rtol: float = 1e-7) -> None:
    """Assert that two numpy arrays are equal within tolerance."""
    np.testing.assert_allclose(a, b, rtol=rtol)


def make_simple_grid() -> AppGrid:
    """
    Create a simple 2x2 quad grid for testing.

    Returns an AppGrid with 9 nodes and 4 quadrilateral elements.
    """
    from pyiwfm.core.mesh import AppGrid, Element, Node, Subregion

    # Create nodes
    nodes = {}
    node_coords = [
        (0.0, 0.0),
        (100.0, 0.0),
        (200.0, 0.0),
        (0.0, 100.0),
        (100.0, 100.0),
        (200.0, 100.0),
        (0.0, 200.0),
        (100.0, 200.0),
        (200.0, 200.0),
    ]
    for i, (x, y) in enumerate(node_coords, start=1):
        nodes[i] = Node(id=i, x=x, y=y)

    # Create elements (counter-clockwise ordering)
    elements = {
        1: Element(id=1, vertices=(1, 2, 5, 4), subregion=1),
        2: Element(id=2, vertices=(2, 3, 6, 5), subregion=1),
        3: Element(id=3, vertices=(4, 5, 8, 7), subregion=2),
        4: Element(id=4, vertices=(5, 6, 9, 8), subregion=2),
    }

    # Create subregions
    subregions = {
        1: Subregion(id=1, name="Region 1"),
        2: Subregion(id=2, name="Region 2"),
    }

    return AppGrid(nodes=nodes, elements=elements, subregions=subregions)


# ---------------------------------------------------------------------------
# Shared webapi-model fixtures
#
# Used by tests that need to spin up the FastAPI app against a mock IWFMModel
# (e.g. tests/unit/test_webapi_smoke.py). Per-file `_make_mock_model` helpers
# in legacy route-test files predate these and are not yet migrated.
# ---------------------------------------------------------------------------


def _make_webapi_grid():
    """Simple 4-node quad grid for webapi tests."""
    from pyiwfm.core.mesh import AppGrid, Element, Node

    nodes = {
        1: Node(id=1, x=0.0, y=0.0),
        2: Node(id=2, x=100.0, y=0.0),
        3: Node(id=3, x=100.0, y=100.0),
        4: Node(id=4, x=0.0, y=100.0),
    }
    elements = {1: Element(id=1, vertices=(1, 2, 3, 4), subregion=1)}
    grid = AppGrid(nodes=nodes, elements=elements)
    grid.compute_connectivity()
    grid.compute_areas()
    return grid


def make_webapi_mock_model(
    *,
    with_streams: bool = True,
    with_groundwater: bool = True,
    with_stratigraphy: bool = True,
):
    """Return a MagicMock IWFMModel with enough shape for webapi routes.

    Kept aligned with `tests/unit/test_webapi_routes_full.py::_make_mock_model`
    so the smoke test exercises the same model contract existing route tests
    assume.
    """
    from unittest.mock import MagicMock

    model = MagicMock()
    model.name = "TestModel"
    model.grid = _make_webapi_grid()
    model.metadata = {}
    model.has_streams = with_streams
    model.has_lakes = False
    model.n_nodes = 4
    model.n_elements = 1
    model.n_layers = 2
    model.n_lakes = 0
    model.n_stream_nodes = 2 if with_streams else 0

    if with_stratigraphy:
        strat = MagicMock()
        strat.n_layers = 2
        strat.n_nodes = 4
        strat.gs_elev = np.array([100.0, 100.0, 100.0, 100.0])
        strat.top_elev = np.full((4, 2), 100.0)
        strat.top_elev[:, 1] = 50.0
        strat.bottom_elev = np.zeros((4, 2))
        strat.bottom_elev[:, 0] = 50.0
        strat.bottom_elev[:, 1] = 0.0
        model.stratigraphy = strat
    else:
        model.stratigraphy = None

    if with_streams:
        streams = MagicMock()
        streams.n_nodes = 2
        sn1 = MagicMock()
        sn1.id = 1
        sn1.groundwater_node = 1
        sn2 = MagicMock()
        sn2.id = 2
        sn2.groundwater_node = 2
        reach = MagicMock()
        reach.id = 1
        reach.stream_nodes = [sn1, sn2]
        streams.reaches = [reach]
        model.streams = streams
    else:
        model.streams = None

    if with_groundwater:
        gw = MagicMock()
        gw.n_hydrograph_locations = 2
        loc1 = MagicMock()
        loc1.x = 50.0
        loc1.y = 50.0
        loc1.name = "Well-1"
        loc1.layer = 1
        loc2 = MagicMock()
        loc2.x = 75.0
        loc2.y = 75.0
        loc2.name = "Well-2"
        loc2.layer = 2
        gw.hydrograph_locations = [loc1, loc2]
        gw.aquifer_params = None
        model.groundwater = gw
    else:
        model.groundwater = None

    return model


def reset_webapi_model_state(model_state) -> None:
    """Clear `ModelState` singleton back to a clean slate."""
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
    model_state._node_id_to_idx = None
    model_state._sorted_elem_ids = None
    model_state._elem_id_to_idx = None
    model_state._hydrograph_locations_cache = None
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


@pytest.fixture
def webapi_mock_model():
    """A fully-loaded mock IWFMModel for webapi tests."""
    return make_webapi_mock_model()


@pytest.fixture
def webapi_client(webapi_mock_model):
    """FastAPI TestClient with `webapi_mock_model` loaded into ModelState."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from pyiwfm.visualization.webapi.config import model_state
    from pyiwfm.visualization.webapi.server import create_app

    reset_webapi_model_state(model_state)
    model_state._model = webapi_mock_model
    app = create_app()
    try:
        yield TestClient(app)
    finally:
        reset_webapi_model_state(model_state)


@pytest.fixture
def mock_model_dir(tmp_path: Path) -> Path:
    """Create a mock model directory with known file structure."""
    sim_dir = tmp_path / "Simulation"
    sim_dir.mkdir()
    pp_dir = tmp_path / "Preprocessor"
    pp_dir.mkdir()
    (sim_dir / "Simulation_MAIN.IN").touch()
    (pp_dir / "PreProcessor_MAIN.IN").touch()
    return tmp_path


def make_simple_stratigraphy(n_nodes: int = 9, n_layers: int = 2) -> Stratigraphy:
    """
    Create a simple stratigraphy for testing.

    Returns a Stratigraphy with uniform layer thicknesses.
    """
    from pyiwfm.core.stratigraphy import Stratigraphy

    gs_elev = np.full(n_nodes, 100.0)
    top_elev = np.zeros((n_nodes, n_layers))
    bottom_elev = np.zeros((n_nodes, n_layers))

    for layer in range(n_layers):
        top_elev[:, layer] = gs_elev - layer * 50.0
        bottom_elev[:, layer] = gs_elev - (layer + 1) * 50.0

    active_node = np.ones((n_nodes, n_layers), dtype=bool)

    return Stratigraphy(
        n_layers=n_layers,
        n_nodes=n_nodes,
        gs_elev=gs_elev,
        top_elev=top_elev,
        bottom_elev=bottom_elev,
        active_node=active_node,
    )
