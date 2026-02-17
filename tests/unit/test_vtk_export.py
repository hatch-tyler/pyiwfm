"""Unit tests for VTK export functionality."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Skip all tests if vtk is not available
vtk = pytest.importorskip("vtk")

from pyiwfm.core.mesh import AppGrid, Element, Node  # noqa: E402
from pyiwfm.core.stratigraphy import Stratigraphy  # noqa: E402
from pyiwfm.visualization.vtk_export import VTKExporter  # noqa: E402


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
def simple_stratigraphy() -> Stratigraphy:
    """Create simple stratigraphy for testing."""
    n_nodes = 9
    n_layers = 2
    gs_elev = np.full(n_nodes, 100.0)
    top_elev = np.column_stack(
        [
            np.full(n_nodes, 100.0),
            np.full(n_nodes, 50.0),
        ]
    )
    bottom_elev = np.column_stack(
        [
            np.full(n_nodes, 50.0),
            np.full(n_nodes, 0.0),
        ]
    )
    active_node = np.ones((n_nodes, n_layers), dtype=bool)
    return Stratigraphy(
        n_layers=n_layers,
        n_nodes=n_nodes,
        gs_elev=gs_elev,
        top_elev=top_elev,
        bottom_elev=bottom_elev,
        active_node=active_node,
    )


class TestVTKExporter:
    """Tests for VTK exporter."""

    def test_exporter_creation(
        self, simple_grid: AppGrid, simple_stratigraphy: Stratigraphy
    ) -> None:
        """Test exporter creation."""
        exporter = VTKExporter(grid=simple_grid, stratigraphy=simple_stratigraphy)
        assert exporter is not None

    def test_create_2d_mesh(self, simple_grid: AppGrid) -> None:
        """Test creating 2D VTK mesh."""
        exporter = VTKExporter(grid=simple_grid)
        vtk_grid = exporter.create_2d_mesh()

        assert vtk_grid is not None
        assert vtk_grid.GetNumberOfPoints() == 9
        assert vtk_grid.GetNumberOfCells() == 4

    def test_create_3d_mesh(self, simple_grid: AppGrid, simple_stratigraphy: Stratigraphy) -> None:
        """Test creating 3D VTK mesh."""
        exporter = VTKExporter(grid=simple_grid, stratigraphy=simple_stratigraphy)
        vtk_grid = exporter.create_3d_mesh()

        assert vtk_grid is not None
        # 9 nodes * 3 surfaces (top, layer boundary, bottom) = 27 points
        assert vtk_grid.GetNumberOfPoints() == 27
        # 4 elements * 2 layers = 8 cells
        assert vtk_grid.GetNumberOfCells() == 8

    def test_add_node_scalar(self, simple_grid: AppGrid) -> None:
        """Test adding scalar data to nodes."""
        exporter = VTKExporter(grid=simple_grid)
        vtk_grid = exporter.create_2d_mesh()

        # Add head data
        heads = np.arange(9, dtype=float) * 10
        exporter.add_node_scalar(vtk_grid, "head", heads)

        # Check scalar was added
        scalars = vtk_grid.GetPointData().GetArray("head")
        assert scalars is not None
        assert scalars.GetNumberOfTuples() == 9

    def test_add_cell_scalar(self, simple_grid: AppGrid) -> None:
        """Test adding scalar data to cells."""
        exporter = VTKExporter(grid=simple_grid)
        vtk_grid = exporter.create_2d_mesh()

        # Add K data
        kh = np.array([10.0, 20.0, 15.0, 25.0])
        exporter.add_cell_scalar(vtk_grid, "kh", kh)

        # Check scalar was added
        scalars = vtk_grid.GetCellData().GetArray("kh")
        assert scalars is not None
        assert scalars.GetNumberOfTuples() == 4

    def test_export_vtu(self, simple_grid: AppGrid, tmp_path: Path) -> None:
        """Test exporting to VTU format."""
        exporter = VTKExporter(grid=simple_grid)

        output_file = tmp_path / "mesh.vtu"
        exporter.export_vtu(output_file)

        assert output_file.exists()

    def test_export_3d_vtu(
        self,
        simple_grid: AppGrid,
        simple_stratigraphy: Stratigraphy,
        tmp_path: Path,
    ) -> None:
        """Test exporting 3D mesh to VTU format."""
        exporter = VTKExporter(grid=simple_grid, stratigraphy=simple_stratigraphy)

        output_file = tmp_path / "mesh_3d.vtu"
        exporter.export_vtu(output_file, mode="3d")

        assert output_file.exists()

    def test_export_with_scalars(self, simple_grid: AppGrid, tmp_path: Path) -> None:
        """Test exporting with scalar data."""
        exporter = VTKExporter(grid=simple_grid)

        heads = np.arange(9, dtype=float) * 10
        kh = np.array([10.0, 20.0, 15.0, 25.0])

        output_file = tmp_path / "mesh_scalars.vtu"
        exporter.export_vtu(
            output_file,
            node_scalars={"head": heads},
            cell_scalars={"kh": kh},
        )

        assert output_file.exists()

    def test_export_vtk_legacy(self, simple_grid: AppGrid, tmp_path: Path) -> None:
        """Test exporting to legacy VTK format."""
        exporter = VTKExporter(grid=simple_grid)

        output_file = tmp_path / "mesh.vtk"
        exporter.export_vtk(output_file)

        assert output_file.exists()


class TestVTK3DMesh:
    """Tests for 3D mesh generation."""

    def test_hexahedra_for_quads(
        self,
        simple_grid: AppGrid,
        simple_stratigraphy: Stratigraphy,
    ) -> None:
        """Test that quad elements become hexahedra in 3D."""
        exporter = VTKExporter(grid=simple_grid, stratigraphy=simple_stratigraphy)
        vtk_grid = exporter.create_3d_mesh()

        # Check cell types - should all be hexahedra (VTK_HEXAHEDRON = 12)
        for i in range(vtk_grid.GetNumberOfCells()):
            cell = vtk_grid.GetCell(i)
            assert cell.GetCellType() == vtk.VTK_HEXAHEDRON

    def test_layer_data(
        self,
        simple_grid: AppGrid,
        simple_stratigraphy: Stratigraphy,
    ) -> None:
        """Test that layer information is included."""
        exporter = VTKExporter(grid=simple_grid, stratigraphy=simple_stratigraphy)
        vtk_grid = exporter.create_3d_mesh()

        # Check for layer data
        layer_array = vtk_grid.GetCellData().GetArray("layer")
        assert layer_array is not None
        assert layer_array.GetNumberOfTuples() == 8  # 4 elements * 2 layers


class TestVTKWithTriangles:
    """Tests for meshes with triangular elements."""

    @pytest.fixture
    def triangle_grid(self) -> AppGrid:
        """Create a mesh with triangles for testing."""
        nodes = {
            1: Node(id=1, x=0.0, y=0.0),
            2: Node(id=2, x=100.0, y=0.0),
            3: Node(id=3, x=50.0, y=100.0),
            4: Node(id=4, x=150.0, y=100.0),
        }
        elements = {
            1: Element(id=1, vertices=(1, 2, 3), subregion=1),
            2: Element(id=2, vertices=(2, 4, 3), subregion=1),
        }
        grid = AppGrid(nodes=nodes, elements=elements)
        grid.compute_connectivity()
        return grid

    def test_wedge_for_triangles(self, triangle_grid: AppGrid) -> None:
        """Test that triangles become wedges in 3D."""
        n_nodes = 4
        n_layers = 2
        strat = Stratigraphy(
            n_layers=n_layers,
            n_nodes=n_nodes,
            gs_elev=np.full(n_nodes, 100.0),
            top_elev=np.column_stack([np.full(n_nodes, 100.0), np.full(n_nodes, 50.0)]),
            bottom_elev=np.column_stack([np.full(n_nodes, 50.0), np.full(n_nodes, 0.0)]),
            active_node=np.ones((n_nodes, n_layers), dtype=bool),
        )

        exporter = VTKExporter(grid=triangle_grid, stratigraphy=strat)
        vtk_grid = exporter.create_3d_mesh()

        # Check cell types - should all be wedges (VTK_WEDGE = 13)
        for i in range(vtk_grid.GetNumberOfCells()):
            cell = vtk_grid.GetCell(i)
            assert cell.GetCellType() == vtk.VTK_WEDGE
