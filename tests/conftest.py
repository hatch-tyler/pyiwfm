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
