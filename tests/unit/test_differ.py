"""Unit tests for model differ functionality."""

from __future__ import annotations

import numpy as np
import pytest

from pyiwfm.core.mesh import AppGrid, Node, Element
from pyiwfm.core.stratigraphy import Stratigraphy
from pyiwfm.comparison.differ import (
    ModelDiffer,
    DiffItem,
    DiffType,
    MeshDiff,
    StratigraphyDiff,
    ModelDiff,
)


@pytest.fixture
def simple_grid() -> AppGrid:
    """Create a simple 2x2 quad mesh for testing."""
    nodes = {
        1: Node(id=1, x=0.0, y=0.0, is_boundary=True),
        2: Node(id=2, x=100.0, y=0.0, is_boundary=True),
        3: Node(id=3, x=200.0, y=0.0, is_boundary=True),
        4: Node(id=4, x=0.0, y=100.0, is_boundary=True),
        5: Node(id=5, x=100.0, y=100.0, is_boundary=False),
        6: Node(id=6, x=200.0, y=100.0, is_boundary=True),
        7: Node(id=7, x=0.0, y=200.0, is_boundary=True),
        8: Node(id=8, x=100.0, y=200.0, is_boundary=True),
        9: Node(id=9, x=200.0, y=200.0, is_boundary=True),
    }
    elements = {
        1: Element(id=1, vertices=(1, 2, 5, 4), subregion=1),
        2: Element(id=2, vertices=(2, 3, 6, 5), subregion=1),
        3: Element(id=3, vertices=(4, 5, 8, 7), subregion=2),
        4: Element(id=4, vertices=(5, 6, 9, 8), subregion=2),
    }
    grid = AppGrid(nodes=nodes, elements=elements)
    grid.compute_connectivity()
    return grid


@pytest.fixture
def modified_grid() -> AppGrid:
    """Create a modified version of the simple grid."""
    nodes = {
        1: Node(id=1, x=0.0, y=0.0, is_boundary=True),
        2: Node(id=2, x=100.0, y=0.0, is_boundary=True),
        3: Node(id=3, x=200.0, y=0.0, is_boundary=True),
        4: Node(id=4, x=0.0, y=100.0, is_boundary=True),
        5: Node(id=5, x=105.0, y=105.0, is_boundary=False),  # Modified coordinates
        6: Node(id=6, x=200.0, y=100.0, is_boundary=True),
        7: Node(id=7, x=0.0, y=200.0, is_boundary=True),
        8: Node(id=8, x=100.0, y=200.0, is_boundary=True),
        9: Node(id=9, x=200.0, y=200.0, is_boundary=True),
        10: Node(id=10, x=250.0, y=100.0, is_boundary=True),  # Added node
    }
    elements = {
        1: Element(id=1, vertices=(1, 2, 5, 4), subregion=1),
        2: Element(id=2, vertices=(2, 3, 6, 5), subregion=2),  # Modified subregion
        3: Element(id=3, vertices=(4, 5, 8, 7), subregion=2),
        4: Element(id=4, vertices=(5, 6, 9, 8), subregion=2),
        5: Element(id=5, vertices=(3, 10, 6), subregion=1),  # Added element
    }
    grid = AppGrid(nodes=nodes, elements=elements)
    grid.compute_connectivity()
    return grid


@pytest.fixture
def simple_stratigraphy() -> Stratigraphy:
    """Create simple stratigraphy for testing."""
    n_nodes = 9
    n_layers = 2
    gs_elev = np.full(n_nodes, 100.0)
    top_elev = np.column_stack([
        np.full(n_nodes, 100.0),
        np.full(n_nodes, 50.0),
    ])
    bottom_elev = np.column_stack([
        np.full(n_nodes, 50.0),
        np.full(n_nodes, 0.0),
    ])
    active_node = np.ones((n_nodes, n_layers), dtype=bool)
    return Stratigraphy(
        n_layers=n_layers,
        n_nodes=n_nodes,
        gs_elev=gs_elev,
        top_elev=top_elev,
        bottom_elev=bottom_elev,
        active_node=active_node,
    )


@pytest.fixture
def modified_stratigraphy() -> Stratigraphy:
    """Create modified stratigraphy for testing."""
    n_nodes = 9
    n_layers = 2
    gs_elev = np.full(n_nodes, 100.0)
    gs_elev[4] = 105.0  # Modified elevation at node 5
    top_elev = np.column_stack([
        np.full(n_nodes, 100.0),
        np.full(n_nodes, 55.0),  # Modified layer boundary
    ])
    bottom_elev = np.column_stack([
        np.full(n_nodes, 55.0),
        np.full(n_nodes, 0.0),
    ])
    active_node = np.ones((n_nodes, n_layers), dtype=bool)
    active_node[0, 1] = False  # Deactivated node
    return Stratigraphy(
        n_layers=n_layers,
        n_nodes=n_nodes,
        gs_elev=gs_elev,
        top_elev=top_elev,
        bottom_elev=bottom_elev,
        active_node=active_node,
    )


class TestDiffItem:
    """Tests for DiffItem class."""

    def test_diff_item_creation(self) -> None:
        """Test creating a DiffItem."""
        item = DiffItem(
            path="mesh.nodes.5.x",
            diff_type=DiffType.MODIFIED,
            old_value=100.0,
            new_value=105.0,
        )
        assert item.path == "mesh.nodes.5.x"
        assert item.diff_type == DiffType.MODIFIED
        assert item.old_value == 100.0
        assert item.new_value == 105.0

    def test_diff_item_added(self) -> None:
        """Test DiffItem for added element."""
        item = DiffItem(
            path="mesh.nodes.10",
            diff_type=DiffType.ADDED,
            new_value={"x": 250.0, "y": 100.0},
        )
        assert item.diff_type == DiffType.ADDED
        assert item.old_value is None

    def test_diff_item_removed(self) -> None:
        """Test DiffItem for removed element."""
        item = DiffItem(
            path="mesh.elements.5",
            diff_type=DiffType.REMOVED,
            old_value={"vertices": (1, 2, 3)},
        )
        assert item.diff_type == DiffType.REMOVED
        assert item.new_value is None


class TestMeshDiff:
    """Tests for mesh comparison."""

    def test_identical_meshes(self, simple_grid: AppGrid) -> None:
        """Test comparing identical meshes."""
        diff = MeshDiff.compare(simple_grid, simple_grid)
        assert diff.is_identical
        assert len(diff.items) == 0

    def test_different_node_count(
        self, simple_grid: AppGrid, modified_grid: AppGrid
    ) -> None:
        """Test detecting different node counts."""
        diff = MeshDiff.compare(simple_grid, modified_grid)
        assert not diff.is_identical
        assert diff.nodes_added == 1

    def test_modified_node_coordinates(
        self, simple_grid: AppGrid, modified_grid: AppGrid
    ) -> None:
        """Test detecting modified node coordinates."""
        diff = MeshDiff.compare(simple_grid, modified_grid)
        modified_items = [
            item for item in diff.items
            if item.diff_type == DiffType.MODIFIED and "nodes.5" in item.path
        ]
        assert len(modified_items) > 0

    def test_added_elements(
        self, simple_grid: AppGrid, modified_grid: AppGrid
    ) -> None:
        """Test detecting added elements."""
        diff = MeshDiff.compare(simple_grid, modified_grid)
        assert diff.elements_added == 1

    def test_modified_element_subregion(
        self, simple_grid: AppGrid, modified_grid: AppGrid
    ) -> None:
        """Test detecting modified element subregion."""
        diff = MeshDiff.compare(simple_grid, modified_grid)
        subregion_items = [
            item for item in diff.items
            if "subregion" in item.path
        ]
        assert len(subregion_items) > 0


class TestStratigraphyDiff:
    """Tests for stratigraphy comparison."""

    def test_identical_stratigraphy(
        self, simple_stratigraphy: Stratigraphy
    ) -> None:
        """Test comparing identical stratigraphy."""
        diff = StratigraphyDiff.compare(
            simple_stratigraphy, simple_stratigraphy
        )
        assert diff.is_identical

    def test_modified_ground_surface(
        self,
        simple_stratigraphy: Stratigraphy,
        modified_stratigraphy: Stratigraphy,
    ) -> None:
        """Test detecting modified ground surface elevation."""
        diff = StratigraphyDiff.compare(
            simple_stratigraphy, modified_stratigraphy
        )
        assert not diff.is_identical
        gs_items = [
            item for item in diff.items
            if "gs_elev" in item.path
        ]
        assert len(gs_items) > 0

    def test_modified_layer_elevations(
        self,
        simple_stratigraphy: Stratigraphy,
        modified_stratigraphy: Stratigraphy,
    ) -> None:
        """Test detecting modified layer elevations."""
        diff = StratigraphyDiff.compare(
            simple_stratigraphy, modified_stratigraphy
        )
        layer_items = [
            item for item in diff.items
            if "top_elev" in item.path or "bottom_elev" in item.path
        ]
        assert len(layer_items) > 0

    def test_modified_active_nodes(
        self,
        simple_stratigraphy: Stratigraphy,
        modified_stratigraphy: Stratigraphy,
    ) -> None:
        """Test detecting modified active node flags."""
        diff = StratigraphyDiff.compare(
            simple_stratigraphy, modified_stratigraphy
        )
        active_items = [
            item for item in diff.items
            if "active_node" in item.path
        ]
        assert len(active_items) > 0


class TestModelDiffer:
    """Tests for ModelDiffer class."""

    def test_differ_creation(self) -> None:
        """Test creating a ModelDiffer."""
        differ = ModelDiffer()
        assert differ is not None

    def test_diff_meshes(
        self, simple_grid: AppGrid, modified_grid: AppGrid
    ) -> None:
        """Test diffing meshes through ModelDiffer."""
        differ = ModelDiffer()
        diff = differ.diff_meshes(simple_grid, modified_grid)
        assert isinstance(diff, MeshDiff)
        assert not diff.is_identical

    def test_diff_stratigraphy(
        self,
        simple_stratigraphy: Stratigraphy,
        modified_stratigraphy: Stratigraphy,
    ) -> None:
        """Test diffing stratigraphy through ModelDiffer."""
        differ = ModelDiffer()
        diff = differ.diff_stratigraphy(
            simple_stratigraphy, modified_stratigraphy
        )
        assert isinstance(diff, StratigraphyDiff)
        assert not diff.is_identical


class TestModelDiff:
    """Tests for ModelDiff container."""

    def test_model_diff_summary(
        self, simple_grid: AppGrid, modified_grid: AppGrid
    ) -> None:
        """Test generating diff summary."""
        mesh_diff = MeshDiff.compare(simple_grid, modified_grid)
        model_diff = ModelDiff(mesh_diff=mesh_diff)
        summary = model_diff.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_model_diff_filter_by_path(
        self, simple_grid: AppGrid, modified_grid: AppGrid
    ) -> None:
        """Test filtering diff items by path."""
        mesh_diff = MeshDiff.compare(simple_grid, modified_grid)
        model_diff = ModelDiff(mesh_diff=mesh_diff)
        filtered = model_diff.filter_by_path("nodes")
        assert all("nodes" in item.path for item in filtered.items)

    def test_model_diff_filter_by_type(
        self, simple_grid: AppGrid, modified_grid: AppGrid
    ) -> None:
        """Test filtering diff items by type."""
        mesh_diff = MeshDiff.compare(simple_grid, modified_grid)
        model_diff = ModelDiff(mesh_diff=mesh_diff)
        added = model_diff.filter_by_type(DiffType.ADDED)
        assert all(item.diff_type == DiffType.ADDED for item in added.items)

    def test_model_diff_to_dict(
        self, simple_grid: AppGrid, modified_grid: AppGrid
    ) -> None:
        """Test converting diff to dictionary."""
        mesh_diff = MeshDiff.compare(simple_grid, modified_grid)
        model_diff = ModelDiff(mesh_diff=mesh_diff)
        d = model_diff.to_dict()
        assert isinstance(d, dict)
        assert "mesh" in d

    def test_model_diff_statistics(
        self, simple_grid: AppGrid, modified_grid: AppGrid
    ) -> None:
        """Test diff statistics."""
        mesh_diff = MeshDiff.compare(simple_grid, modified_grid)
        model_diff = ModelDiff(mesh_diff=mesh_diff)
        stats = model_diff.statistics()
        assert "total_changes" in stats
        assert "added" in stats
        assert "removed" in stats
        assert "modified" in stats


class TestDiffItemRepr:
    """Tests for DiffItem __repr__ for all diff types."""

    def test_repr_added(self) -> None:
        """Test repr for ADDED diff type."""
        item = DiffItem(
            path="mesh.nodes.10",
            diff_type=DiffType.ADDED,
            new_value={"x": 250.0},
        )
        r = repr(item)
        assert r.startswith("+ ")
        assert "mesh.nodes.10" in r
        assert "250.0" in r

    def test_repr_removed(self) -> None:
        """Test repr for REMOVED diff type."""
        item = DiffItem(
            path="mesh.nodes.5",
            diff_type=DiffType.REMOVED,
            old_value={"x": 100.0},
        )
        r = repr(item)
        assert r.startswith("- ")
        assert "mesh.nodes.5" in r
        assert "100.0" in r

    def test_repr_modified(self) -> None:
        """Test repr for MODIFIED diff type."""
        item = DiffItem(
            path="mesh.nodes.5.x",
            diff_type=DiffType.MODIFIED,
            old_value=100.0,
            new_value=105.0,
        )
        r = repr(item)
        assert r.startswith("~ ")
        assert "100.0" in r
        assert "105.0" in r
        assert "->" in r


class TestMeshDiffRemoved:
    """Tests for mesh diff with removed nodes and elements."""

    def test_removed_nodes(
        self, simple_grid: AppGrid, modified_grid: AppGrid
    ) -> None:
        """Test detecting removed nodes when comparing reversed."""
        # Compare modified -> simple means node 10 is "removed"
        diff = MeshDiff.compare(modified_grid, simple_grid)
        assert diff.nodes_removed == 1
        removed = [i for i in diff.items if i.diff_type == DiffType.REMOVED and "nodes" in i.path]
        assert len(removed) == 1

    def test_removed_elements(
        self, simple_grid: AppGrid, modified_grid: AppGrid
    ) -> None:
        """Test detecting removed elements when comparing reversed."""
        diff = MeshDiff.compare(modified_grid, simple_grid)
        assert diff.elements_removed == 1
        removed = [i for i in diff.items if i.diff_type == DiffType.REMOVED and "elements" in i.path]
        assert len(removed) == 1

    def test_modified_element_vertices(self) -> None:
        """Test detecting modified element vertices."""
        nodes1 = {
            1: Node(id=1, x=0.0, y=0.0),
            2: Node(id=2, x=1.0, y=0.0),
            3: Node(id=3, x=1.0, y=1.0),
            4: Node(id=4, x=0.0, y=1.0),
        }
        nodes2 = dict(nodes1)
        elem1 = {1: Element(id=1, vertices=(1, 2, 3, 4))}
        elem2 = {1: Element(id=1, vertices=(1, 2, 4, 3))}  # Different vertex order
        mesh1 = AppGrid(nodes=nodes1, elements=elem1)
        mesh2 = AppGrid(nodes=nodes2, elements=elem2)
        diff = MeshDiff.compare(mesh1, mesh2)
        assert diff.elements_modified == 1
        vertex_items = [i for i in diff.items if "vertices" in i.path]
        assert len(vertex_items) == 1

    def test_modified_node_boundary_flag(self) -> None:
        """Test detecting modified is_boundary flag on nodes."""
        nodes1 = {1: Node(id=1, x=0.0, y=0.0, is_boundary=True)}
        nodes2 = {1: Node(id=1, x=0.0, y=0.0, is_boundary=False)}
        mesh1 = AppGrid(nodes=nodes1, elements={1: Element(id=1, vertices=(1, 1, 1))})
        mesh2 = AppGrid(nodes=nodes2, elements={1: Element(id=1, vertices=(1, 1, 1))})
        diff = MeshDiff.compare(mesh1, mesh2)
        boundary_items = [i for i in diff.items if "is_boundary" in i.path]
        assert len(boundary_items) == 1


class TestStratigraphyDiffEdgeCases:
    """Tests for StratigraphyDiff edge cases."""

    def test_different_layer_count(self) -> None:
        """Test comparing stratigraphy with different layer counts."""
        n_nodes = 4
        gs = np.full(n_nodes, 100.0)
        strat1 = Stratigraphy(
            n_layers=2,
            n_nodes=n_nodes,
            gs_elev=gs,
            top_elev=np.column_stack([np.full(n_nodes, 100.0), np.full(n_nodes, 50.0)]),
            bottom_elev=np.column_stack([np.full(n_nodes, 50.0), np.full(n_nodes, 0.0)]),
            active_node=np.ones((n_nodes, 2), dtype=bool),
        )
        strat2 = Stratigraphy(
            n_layers=3,
            n_nodes=n_nodes,
            gs_elev=gs,
            top_elev=np.column_stack([np.full(n_nodes, 100.0), np.full(n_nodes, 66.0), np.full(n_nodes, 33.0)]),
            bottom_elev=np.column_stack([np.full(n_nodes, 66.0), np.full(n_nodes, 33.0), np.full(n_nodes, 0.0)]),
            active_node=np.ones((n_nodes, 3), dtype=bool),
        )
        diff = StratigraphyDiff.compare(strat1, strat2)
        assert not diff.is_identical
        layer_items = [i for i in diff.items if "n_layers" in i.path]
        assert len(layer_items) == 1

    def test_different_node_count(self) -> None:
        """Test comparing stratigraphy with different node counts returns early."""
        strat1 = Stratigraphy(
            n_layers=1, n_nodes=3,
            gs_elev=np.full(3, 100.0),
            top_elev=np.full((3, 1), 100.0),
            bottom_elev=np.full((3, 1), 0.0),
            active_node=np.ones((3, 1), dtype=bool),
        )
        strat2 = Stratigraphy(
            n_layers=1, n_nodes=4,
            gs_elev=np.full(4, 100.0),
            top_elev=np.full((4, 1), 100.0),
            bottom_elev=np.full((4, 1), 0.0),
            active_node=np.ones((4, 1), dtype=bool),
        )
        diff = StratigraphyDiff.compare(strat1, strat2)
        assert not diff.is_identical
        n_nodes_items = [i for i in diff.items if "n_nodes" in i.path]
        assert len(n_nodes_items) == 1
        # Should return early: no elevation comparisons
        elev_items = [i for i in diff.items if "elev" in i.path]
        assert len(elev_items) == 0

    def test_tolerance_parameter(self) -> None:
        """Test that tolerance controls floating point comparisons."""
        n = 3
        gs = np.full(n, 100.0)
        strat1 = Stratigraphy(
            n_layers=1, n_nodes=n,
            gs_elev=gs,
            top_elev=np.full((n, 1), 100.0),
            bottom_elev=np.full((n, 1), 0.0),
            active_node=np.ones((n, 1), dtype=bool),
        )
        gs2 = gs.copy()
        gs2[0] = 100.0 + 1e-8  # Very small change
        strat2 = Stratigraphy(
            n_layers=1, n_nodes=n,
            gs_elev=gs2,
            top_elev=np.full((n, 1), 100.0),
            bottom_elev=np.full((n, 1), 0.0),
            active_node=np.ones((n, 1), dtype=bool),
        )
        # With default tolerance 1e-6, this small change is below threshold
        diff_default = StratigraphyDiff.compare(strat1, strat2, tolerance=1e-6)
        assert diff_default.is_identical

        # With very tight tolerance, the change is detected
        diff_tight = StratigraphyDiff.compare(strat1, strat2, tolerance=1e-10)
        assert not diff_tight.is_identical


class TestModelDiffEdgeCases:
    """Tests for ModelDiff edge cases."""

    def test_model_diff_is_identical_empty(self) -> None:
        """Test is_identical when both diffs are None."""
        md = ModelDiff(mesh_diff=None, stratigraphy_diff=None)
        assert md.is_identical is True

    def test_model_diff_is_identical_with_empty_mesh_diff(self) -> None:
        """Test is_identical when mesh diff exists but has no items."""
        md = ModelDiff(mesh_diff=MeshDiff(), stratigraphy_diff=None)
        assert md.is_identical is True

    def test_model_diff_is_identical_with_empty_strat_diff(self) -> None:
        """Test is_identical when strat diff exists but has no items."""
        md = ModelDiff(mesh_diff=None, stratigraphy_diff=StratigraphyDiff())
        assert md.is_identical is True

    def test_model_diff_items_combined(
        self, simple_grid: AppGrid, modified_grid: AppGrid,
        simple_stratigraphy: Stratigraphy, modified_stratigraphy: Stratigraphy,
    ) -> None:
        """Test items property combines mesh and strat items."""
        mesh_diff = MeshDiff.compare(simple_grid, modified_grid)
        strat_diff = StratigraphyDiff.compare(simple_stratigraphy, modified_stratigraphy)
        md = ModelDiff(mesh_diff=mesh_diff, stratigraphy_diff=strat_diff)
        all_items = md.items
        assert len(all_items) == len(mesh_diff.items) + len(strat_diff.items)

    def test_model_diff_items_no_diffs(self) -> None:
        """Test items property when both diffs are None."""
        md = ModelDiff()
        assert md.items == []

    def test_model_diff_summary_identical(self) -> None:
        """Test summary for identical models."""
        md = ModelDiff(mesh_diff=MeshDiff(), stratigraphy_diff=StratigraphyDiff())
        summary = md.summary()
        assert "identical" in summary.lower()

    def test_model_diff_summary_with_strat_changes(
        self, simple_stratigraphy: Stratigraphy, modified_stratigraphy: Stratigraphy,
    ) -> None:
        """Test summary includes stratigraphy changes."""
        strat_diff = StratigraphyDiff.compare(simple_stratigraphy, modified_stratigraphy)
        md = ModelDiff(stratigraphy_diff=strat_diff)
        summary = md.summary()
        assert "Stratigraphy Changes" in summary
        assert "modifications" in summary

    def test_model_diff_summary_with_mesh_changes(
        self, simple_grid: AppGrid, modified_grid: AppGrid,
    ) -> None:
        """Test summary includes mesh changes with node/element counts."""
        mesh_diff = MeshDiff.compare(simple_grid, modified_grid)
        md = ModelDiff(mesh_diff=mesh_diff)
        summary = md.summary()
        assert "Mesh Changes" in summary
        assert "Nodes:" in summary
        assert "Elements:" in summary

    def test_filter_by_path_with_stratigraphy(
        self, simple_stratigraphy: Stratigraphy, modified_stratigraphy: Stratigraphy,
    ) -> None:
        """Test filter_by_path with stratigraphy diff."""
        strat_diff = StratigraphyDiff.compare(simple_stratigraphy, modified_stratigraphy)
        md = ModelDiff(stratigraphy_diff=strat_diff)
        filtered = md.filter_by_path("gs_elev")
        assert all("gs_elev" in i.path for i in filtered.items)

    def test_filter_by_type_with_stratigraphy(
        self, simple_stratigraphy: Stratigraphy, modified_stratigraphy: Stratigraphy,
    ) -> None:
        """Test filter_by_type with stratigraphy diff."""
        strat_diff = StratigraphyDiff.compare(simple_stratigraphy, modified_stratigraphy)
        md = ModelDiff(stratigraphy_diff=strat_diff)
        modified = md.filter_by_type(DiffType.MODIFIED)
        assert all(i.diff_type == DiffType.MODIFIED for i in modified.items)

    def test_filter_by_path_no_match(
        self, simple_grid: AppGrid, modified_grid: AppGrid,
    ) -> None:
        """Test filter_by_path returns empty when no items match."""
        mesh_diff = MeshDiff.compare(simple_grid, modified_grid)
        md = ModelDiff(mesh_diff=mesh_diff)
        filtered = md.filter_by_path("nonexistent_path_xyz")
        assert len(filtered.items) == 0

    def test_to_dict_with_stratigraphy(
        self, simple_stratigraphy: Stratigraphy, modified_stratigraphy: Stratigraphy,
    ) -> None:
        """Test to_dict includes stratigraphy section."""
        strat_diff = StratigraphyDiff.compare(simple_stratigraphy, modified_stratigraphy)
        md = ModelDiff(stratigraphy_diff=strat_diff)
        d = md.to_dict()
        assert "stratigraphy" in d
        assert "is_identical" in d["stratigraphy"]
        assert d["stratigraphy"]["is_identical"] is False
        assert "items" in d["stratigraphy"]

    def test_to_dict_with_both(
        self, simple_grid: AppGrid, modified_grid: AppGrid,
        simple_stratigraphy: Stratigraphy, modified_stratigraphy: Stratigraphy,
    ) -> None:
        """Test to_dict with both mesh and stratigraphy diffs."""
        mesh_diff = MeshDiff.compare(simple_grid, modified_grid)
        strat_diff = StratigraphyDiff.compare(simple_stratigraphy, modified_stratigraphy)
        md = ModelDiff(mesh_diff=mesh_diff, stratigraphy_diff=strat_diff)
        d = md.to_dict()
        assert "mesh" in d
        assert "stratigraphy" in d

    def test_to_dict_empty(self) -> None:
        """Test to_dict with no diffs returns empty dict."""
        md = ModelDiff()
        d = md.to_dict()
        assert d == {}

    def test_to_dict_mesh_structure(
        self, simple_grid: AppGrid, modified_grid: AppGrid,
    ) -> None:
        """Test to_dict mesh section has all expected keys."""
        mesh_diff = MeshDiff.compare(simple_grid, modified_grid)
        md = ModelDiff(mesh_diff=mesh_diff)
        d = md.to_dict()
        mesh = d["mesh"]
        assert "is_identical" in mesh
        assert "nodes_added" in mesh
        assert "nodes_removed" in mesh
        assert "nodes_modified" in mesh
        assert "elements_added" in mesh
        assert "elements_removed" in mesh
        assert "elements_modified" in mesh
        assert "items" in mesh
        # Verify item structure
        for item in mesh["items"]:
            assert "path" in item
            assert "type" in item
            assert "old_value" in item
            assert "new_value" in item


class TestModelDifferEdgeCases:
    """Tests for ModelDiffer class edge cases."""

    def test_differ_with_custom_tolerance(self) -> None:
        """Test creating ModelDiffer with custom tolerance."""
        differ = ModelDiffer(tolerance=1e-3)
        assert differ.tolerance == 1e-3

    def test_diff_method_mesh_only(
        self, simple_grid: AppGrid, modified_grid: AppGrid,
    ) -> None:
        """Test diff() method with mesh only."""
        differ = ModelDiffer()
        result = differ.diff(mesh1=simple_grid, mesh2=modified_grid)
        assert isinstance(result, ModelDiff)
        assert result.mesh_diff is not None
        assert result.stratigraphy_diff is None

    def test_diff_method_strat_only(
        self, simple_stratigraphy: Stratigraphy, modified_stratigraphy: Stratigraphy,
    ) -> None:
        """Test diff() method with stratigraphy only."""
        differ = ModelDiffer()
        result = differ.diff(strat1=simple_stratigraphy, strat2=modified_stratigraphy)
        assert result.mesh_diff is None
        assert result.stratigraphy_diff is not None

    def test_diff_method_both(
        self, simple_grid: AppGrid, modified_grid: AppGrid,
        simple_stratigraphy: Stratigraphy, modified_stratigraphy: Stratigraphy,
    ) -> None:
        """Test diff() method with both mesh and stratigraphy."""
        differ = ModelDiffer()
        result = differ.diff(
            mesh1=simple_grid, mesh2=modified_grid,
            strat1=simple_stratigraphy, strat2=modified_stratigraphy,
        )
        assert result.mesh_diff is not None
        assert result.stratigraphy_diff is not None

    def test_diff_method_nothing(self) -> None:
        """Test diff() method with no arguments."""
        differ = ModelDiffer()
        result = differ.diff()
        assert result.mesh_diff is None
        assert result.stratigraphy_diff is None
        assert result.is_identical

    def test_diff_method_one_mesh_none(
        self, simple_grid: AppGrid,
    ) -> None:
        """Test diff() with one mesh None does not produce mesh diff."""
        differ = ModelDiffer()
        result = differ.diff(mesh1=simple_grid, mesh2=None)
        assert result.mesh_diff is None

    def test_diff_strat_uses_tolerance(self) -> None:
        """Test that ModelDiffer passes tolerance to StratigraphyDiff."""
        n = 3
        gs = np.full(n, 100.0)
        strat1 = Stratigraphy(
            n_layers=1, n_nodes=n,
            gs_elev=gs,
            top_elev=np.full((n, 1), 100.0),
            bottom_elev=np.full((n, 1), 0.0),
            active_node=np.ones((n, 1), dtype=bool),
        )
        gs2 = gs.copy()
        gs2[0] = 100.001  # Small change
        strat2 = Stratigraphy(
            n_layers=1, n_nodes=n,
            gs_elev=gs2,
            top_elev=np.full((n, 1), 100.0),
            bottom_elev=np.full((n, 1), 0.0),
            active_node=np.ones((n, 1), dtype=bool),
        )
        # With large tolerance, should be identical
        differ_lax = ModelDiffer(tolerance=0.01)
        result_lax = differ_lax.diff(strat1=strat1, strat2=strat2)
        assert result_lax.stratigraphy_diff.is_identical

        # With tight tolerance, should detect change
        differ_tight = ModelDiffer(tolerance=1e-6)
        result_tight = differ_tight.diff(strat1=strat1, strat2=strat2)
        assert not result_tight.stratigraphy_diff.is_identical
