"""
Comprehensive tests for pyiwfm.visualization.vtk_export module.

Tests cover:
- VTKExporter initialization
- 2D and 3D mesh creation
- Adding scalar data to nodes and cells
- VTU and VTK export
- PyVista mesh conversion
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_vtk():
    """Create a mock VTK module."""
    mock = MagicMock()

    # Mock VTK classes
    mock.vtkUnstructuredGrid.return_value = MagicMock()
    mock.vtkPoints.return_value = MagicMock()
    mock.vtkTriangle.return_value = MagicMock()
    mock.vtkQuad.return_value = MagicMock()
    mock.vtkWedge.return_value = MagicMock()
    mock.vtkHexahedron.return_value = MagicMock()
    mock.vtkDoubleArray.return_value = MagicMock()
    mock.vtkIntArray.return_value = MagicMock()
    mock.vtkXMLUnstructuredGridWriter.return_value = MagicMock()
    mock.vtkUnstructuredGridWriter.return_value = MagicMock()

    return mock


@pytest.fixture
def mock_pyvista():
    """Create a mock PyVista module."""
    mock = MagicMock()
    mock.CellType.TRIANGLE = 5
    mock.CellType.QUAD = 9
    mock.CellType.WEDGE = 13
    mock.CellType.HEXAHEDRON = 12
    mock.UnstructuredGrid.return_value = MagicMock()
    return mock


@pytest.fixture
def mock_node():
    """Create a mock node."""

    def _create_node(node_id, x, y):
        node = MagicMock()
        node.id = node_id
        node.x = x
        node.y = y
        return node

    return _create_node


@pytest.fixture
def mock_element():
    """Create a mock element."""

    def _create_element(elem_id, vertices, is_triangle=False):
        elem = MagicMock()
        elem.id = elem_id
        elem.vertices = vertices
        elem.is_triangle = is_triangle
        return elem

    return _create_element


@pytest.fixture
def mock_grid(mock_node, mock_element):
    """Create a mock grid with nodes and elements."""
    grid = MagicMock()

    # Create nodes
    nodes = {
        1: mock_node(1, 0.0, 0.0),
        2: mock_node(2, 100.0, 0.0),
        3: mock_node(3, 100.0, 100.0),
        4: mock_node(4, 0.0, 100.0),
    }
    grid.nodes = nodes
    grid.n_nodes = 4

    # Create elements (one quad)
    elements = [mock_element(1, [1, 2, 3, 4], is_triangle=False)]
    grid.iter_elements.return_value = iter(elements)
    grid.iter_nodes.return_value = iter(nodes.values())

    return grid


@pytest.fixture
def mock_grid_with_triangles(mock_node, mock_element):
    """Create a mock grid with triangle elements."""
    grid = MagicMock()

    nodes = {
        1: mock_node(1, 0.0, 0.0),
        2: mock_node(2, 100.0, 0.0),
        3: mock_node(3, 50.0, 100.0),
    }
    grid.nodes = nodes
    grid.n_nodes = 3

    elements = [mock_element(1, [1, 2, 3], is_triangle=True)]
    grid.iter_elements.return_value = iter(elements)
    grid.iter_nodes.return_value = iter(nodes.values())

    return grid


@pytest.fixture
def mock_stratigraphy():
    """Create a mock stratigraphy."""
    strat = MagicMock()
    strat.n_layers = 2

    # 4 nodes, 2 layers
    strat.top_elev = np.array(
        [
            [100.0, 80.0],  # Node 1
            [105.0, 85.0],  # Node 2
            [110.0, 90.0],  # Node 3
            [102.0, 82.0],  # Node 4
        ]
    )
    strat.bottom_elev = np.array(
        [
            [80.0, 50.0],  # Node 1
            [85.0, 55.0],  # Node 2
            [90.0, 60.0],  # Node 3
            [82.0, 52.0],  # Node 4
        ]
    )

    return strat


# =============================================================================
# VTKExporter Initialization Tests
# =============================================================================


class TestVTKExporterInit:
    """Tests for VTKExporter initialization."""

    def test_init_requires_vtk(self, mock_grid, mock_vtk):
        """Test initialization checks for VTK import."""
        # We test that the VTKExporter __init__ tries to import vtk
        # by verifying it works with a mock vtk
        with patch.dict("sys.modules", {"vtk": mock_vtk}):
            from pyiwfm.visualization.vtk_export import VTKExporter

            # Should work with mock vtk
            exporter = VTKExporter(mock_grid)
            assert exporter is not None

    def test_init_with_grid_only(self, mock_grid, mock_vtk):
        """Test initialization with grid only."""
        with patch.dict("sys.modules", {"vtk": mock_vtk}):
            from pyiwfm.visualization.vtk_export import VTKExporter

            exporter = VTKExporter(mock_grid)

            assert exporter.grid is mock_grid
            assert exporter.stratigraphy is None

    def test_init_with_stratigraphy(self, mock_grid, mock_stratigraphy, mock_vtk):
        """Test initialization with stratigraphy."""
        with patch.dict("sys.modules", {"vtk": mock_vtk}):
            from pyiwfm.visualization.vtk_export import VTKExporter

            exporter = VTKExporter(mock_grid, mock_stratigraphy)

            assert exporter.grid is mock_grid
            assert exporter.stratigraphy is mock_stratigraphy


# =============================================================================
# 2D Mesh Creation Tests
# =============================================================================


class TestVTKExporterCreate2DMesh:
    """Tests for VTKExporter.create_2d_mesh method."""

    def test_create_2d_mesh_basic(self, mock_grid, mock_vtk):
        """Test basic 2D mesh creation."""
        with patch.dict("sys.modules", {"vtk": mock_vtk}):
            from pyiwfm.visualization.vtk_export import VTKExporter

            exporter = VTKExporter(mock_grid)
            exporter.create_2d_mesh()

            # Should have created points and cells
            mock_vtk.vtkUnstructuredGrid.assert_called()
            mock_vtk.vtkPoints.assert_called()

    def test_create_2d_mesh_with_quad(self, mock_grid, mock_vtk):
        """Test 2D mesh with quad element."""
        with patch.dict("sys.modules", {"vtk": mock_vtk}):
            from pyiwfm.visualization.vtk_export import VTKExporter

            exporter = VTKExporter(mock_grid)
            exporter.create_2d_mesh()

            # Should create vtkQuad
            mock_vtk.vtkQuad.assert_called()

    def test_create_2d_mesh_with_triangle(self, mock_grid_with_triangles, mock_vtk):
        """Test 2D mesh with triangle element."""
        with patch.dict("sys.modules", {"vtk": mock_vtk}):
            from pyiwfm.visualization.vtk_export import VTKExporter

            exporter = VTKExporter(mock_grid_with_triangles)
            exporter.create_2d_mesh()

            # Should create vtkTriangle
            mock_vtk.vtkTriangle.assert_called()


# =============================================================================
# 3D Mesh Creation Tests
# =============================================================================


class TestVTKExporterCreate3DMesh:
    """Tests for VTKExporter.create_3d_mesh method."""

    def test_create_3d_mesh_requires_stratigraphy(self, mock_grid, mock_vtk):
        """Test 3D mesh creation requires stratigraphy."""
        with patch.dict("sys.modules", {"vtk": mock_vtk}):
            from pyiwfm.visualization.vtk_export import VTKExporter

            exporter = VTKExporter(mock_grid)

            with pytest.raises(ValueError) as exc_info:
                exporter.create_3d_mesh()

            assert "Stratigraphy required" in str(exc_info.value)

    def test_create_3d_mesh_with_stratigraphy(self, mock_grid, mock_stratigraphy, mock_vtk):
        """Test 3D mesh creation with stratigraphy."""
        with patch.dict("sys.modules", {"vtk": mock_vtk}):
            from pyiwfm.visualization.vtk_export import VTKExporter

            exporter = VTKExporter(mock_grid, mock_stratigraphy)
            exporter.create_3d_mesh()

            # Should have created grid with points
            mock_vtk.vtkUnstructuredGrid.assert_called()
            mock_vtk.vtkPoints.assert_called()

    def test_create_3d_mesh_adds_layer_data(self, mock_grid, mock_stratigraphy, mock_vtk):
        """Test 3D mesh adds layer cell data."""
        with patch.dict("sys.modules", {"vtk": mock_vtk}):
            from pyiwfm.visualization.vtk_export import VTKExporter

            exporter = VTKExporter(mock_grid, mock_stratigraphy)
            result = exporter.create_3d_mesh()

            # Should add layer data array to cell data
            result.GetCellData().AddArray.assert_called()


# =============================================================================
# Scalar Data Tests
# =============================================================================


class TestVTKExporterScalarData:
    """Tests for adding scalar data to VTK grids."""

    def test_add_node_scalar(self, mock_grid, mock_vtk):
        """Test adding scalar data to nodes."""
        with patch.dict("sys.modules", {"vtk": mock_vtk}):
            from pyiwfm.visualization.vtk_export import VTKExporter

            exporter = VTKExporter(mock_grid)
            vtk_grid = MagicMock()
            values = np.array([1.0, 2.0, 3.0, 4.0])

            exporter.add_node_scalar(vtk_grid, "test_scalar", values)

            # Should create double array
            mock_vtk.vtkDoubleArray.assert_called()

    def test_add_cell_scalar(self, mock_grid, mock_vtk):
        """Test adding scalar data to cells."""
        with patch.dict("sys.modules", {"vtk": mock_vtk}):
            from pyiwfm.visualization.vtk_export import VTKExporter

            exporter = VTKExporter(mock_grid)
            vtk_grid = MagicMock()
            values = np.array([1.0])

            exporter.add_cell_scalar(vtk_grid, "test_scalar", values)

            # Should create double array
            mock_vtk.vtkDoubleArray.assert_called()


# =============================================================================
# Export Tests
# =============================================================================


class TestVTKExporterExport:
    """Tests for VTK file export methods."""

    def test_export_vtu_2d(self, mock_grid, mock_vtk, tmp_path):
        """Test VTU export in 2D mode."""
        with patch.dict("sys.modules", {"vtk": mock_vtk}):
            from pyiwfm.visualization.vtk_export import VTKExporter

            exporter = VTKExporter(mock_grid)
            output_path = tmp_path / "test.vtu"

            exporter.export_vtu(output_path, mode="2d")

            # Should call XML writer
            mock_vtk.vtkXMLUnstructuredGridWriter.assert_called()

    def test_export_vtu_3d(self, mock_grid, mock_stratigraphy, mock_vtk, tmp_path):
        """Test VTU export in 3D mode."""
        with patch.dict("sys.modules", {"vtk": mock_vtk}):
            from pyiwfm.visualization.vtk_export import VTKExporter

            exporter = VTKExporter(mock_grid, mock_stratigraphy)
            output_path = tmp_path / "test.vtu"

            exporter.export_vtu(output_path, mode="3d")

            mock_vtk.vtkXMLUnstructuredGridWriter.assert_called()

    def test_export_vtu_with_scalars(self, mock_grid, mock_vtk, tmp_path):
        """Test VTU export with scalar data."""
        with patch.dict("sys.modules", {"vtk": mock_vtk}):
            from pyiwfm.visualization.vtk_export import VTKExporter

            exporter = VTKExporter(mock_grid)
            output_path = tmp_path / "test.vtu"

            node_scalars = {"elevation": np.array([1.0, 2.0, 3.0, 4.0])}
            cell_scalars = {"property": np.array([5.0])}

            exporter.export_vtu(
                output_path, mode="2d", node_scalars=node_scalars, cell_scalars=cell_scalars
            )

            # Should have added scalars
            mock_vtk.vtkDoubleArray.assert_called()

    def test_export_vtk_2d(self, mock_grid, mock_vtk, tmp_path):
        """Test legacy VTK export in 2D mode."""
        with patch.dict("sys.modules", {"vtk": mock_vtk}):
            from pyiwfm.visualization.vtk_export import VTKExporter

            exporter = VTKExporter(mock_grid)
            output_path = tmp_path / "test.vtk"

            exporter.export_vtk(output_path, mode="2d")

            # Should call legacy writer
            mock_vtk.vtkUnstructuredGridWriter.assert_called()

    def test_export_vtk_3d(self, mock_grid, mock_stratigraphy, mock_vtk, tmp_path):
        """Test legacy VTK export in 3D mode."""
        with patch.dict("sys.modules", {"vtk": mock_vtk}):
            from pyiwfm.visualization.vtk_export import VTKExporter

            exporter = VTKExporter(mock_grid, mock_stratigraphy)
            output_path = tmp_path / "test.vtk"

            exporter.export_vtk(output_path, mode="3d")

            mock_vtk.vtkUnstructuredGridWriter.assert_called()

    def test_export_vtk_with_scalars(self, mock_grid, mock_vtk, tmp_path):
        """Test legacy VTK export with scalar data."""
        with patch.dict("sys.modules", {"vtk": mock_vtk}):
            from pyiwfm.visualization.vtk_export import VTKExporter

            exporter = VTKExporter(mock_grid)
            output_path = tmp_path / "test.vtk"

            node_scalars = {"elevation": np.array([1.0, 2.0, 3.0, 4.0])}

            exporter.export_vtk(output_path, mode="2d", node_scalars=node_scalars)

            mock_vtk.vtkDoubleArray.assert_called()


# =============================================================================
# PyVista Conversion Tests
# =============================================================================


class TestVTKExporterPyVista:
    """Tests for PyVista mesh conversion."""

    def test_to_pyvista_3d_with_pyvista(self, mock_grid, mock_vtk, mock_pyvista):
        """Test PyVista conversion works with pyvista available."""
        with patch.dict("sys.modules", {"vtk": mock_vtk, "pyvista": mock_pyvista}):
            from pyiwfm.visualization.vtk_export import VTKExporter

            exporter = VTKExporter(mock_grid)
            result = exporter.to_pyvista_3d()

            # Should create UnstructuredGrid
            assert result is not None

    def test_to_pyvista_3d_without_stratigraphy_uses_2d(self, mock_grid, mock_vtk, mock_pyvista):
        """Test to_pyvista_3d falls back to 2D without stratigraphy."""
        with patch.dict("sys.modules", {"vtk": mock_vtk, "pyvista": mock_pyvista}):
            from pyiwfm.visualization.vtk_export import VTKExporter

            exporter = VTKExporter(mock_grid)

            exporter.to_pyvista_3d()

            # Should create UnstructuredGrid
            mock_pyvista.UnstructuredGrid.assert_called()

    def test_to_pyvista_3d_with_stratigraphy(
        self, mock_grid, mock_stratigraphy, mock_vtk, mock_pyvista
    ):
        """Test to_pyvista_3d with stratigraphy creates 3D mesh."""
        with patch.dict("sys.modules", {"vtk": mock_vtk, "pyvista": mock_pyvista}):
            from pyiwfm.visualization.vtk_export import VTKExporter

            exporter = VTKExporter(mock_grid, mock_stratigraphy)

            exporter.to_pyvista_3d()

            mock_pyvista.UnstructuredGrid.assert_called()

    def test_to_pyvista_2d_basic(self, mock_grid, mock_vtk, mock_pyvista):
        """Test _to_pyvista_2d creates 2D mesh."""
        with patch.dict("sys.modules", {"vtk": mock_vtk, "pyvista": mock_pyvista}):
            from pyiwfm.visualization.vtk_export import VTKExporter

            exporter = VTKExporter(mock_grid)

            exporter._to_pyvista_2d()

            mock_pyvista.UnstructuredGrid.assert_called()

    def test_to_pyvista_2d_with_scalars(self, mock_grid, mock_vtk, mock_pyvista):
        """Test _to_pyvista_2d with scalar data."""
        with patch.dict("sys.modules", {"vtk": mock_vtk, "pyvista": mock_pyvista}):
            from pyiwfm.visualization.vtk_export import VTKExporter

            exporter = VTKExporter(mock_grid)

            node_scalars = {"elevation": np.array([1.0, 2.0, 3.0, 4.0])}
            cell_scalars = {"property": np.array([5.0])}

            mesh = MagicMock()
            mock_pyvista.UnstructuredGrid.return_value = mesh

            exporter._to_pyvista_2d(node_scalars=node_scalars, cell_scalars=cell_scalars)

            # Should set point_data and cell_data
            assert mesh.point_data.__setitem__.called or mesh.cell_data.__setitem__.called

    def test_to_pyvista_3d_impl_adds_layer_data(
        self, mock_grid, mock_stratigraphy, mock_vtk, mock_pyvista
    ):
        """Test _to_pyvista_3d_impl adds layer cell data."""
        with patch.dict("sys.modules", {"vtk": mock_vtk, "pyvista": mock_pyvista}):
            from pyiwfm.visualization.vtk_export import VTKExporter

            exporter = VTKExporter(mock_grid, mock_stratigraphy)

            mesh = MagicMock()
            mock_pyvista.UnstructuredGrid.return_value = mesh

            exporter._to_pyvista_3d_impl()

            # Should add layer data to cell_data
            assert mesh.cell_data.__setitem__.called

    def test_to_pyvista_3d_impl_with_scalars(
        self, mock_grid, mock_stratigraphy, mock_vtk, mock_pyvista
    ):
        """Test _to_pyvista_3d_impl with scalar data."""
        with patch.dict("sys.modules", {"vtk": mock_vtk, "pyvista": mock_pyvista}):
            from pyiwfm.visualization.vtk_export import VTKExporter

            exporter = VTKExporter(mock_grid, mock_stratigraphy)

            node_scalars = {"elevation": np.zeros(12)}  # 4 nodes * 3 surfaces
            cell_scalars = {"Kh": np.zeros(2)}  # 2 layers * 1 element

            mesh = MagicMock()
            mock_pyvista.UnstructuredGrid.return_value = mesh

            exporter._to_pyvista_3d_impl(node_scalars=node_scalars, cell_scalars=cell_scalars)

            assert mesh.point_data.__setitem__.called or mesh.cell_data.__setitem__.called


# =============================================================================
# Edge Cases
# =============================================================================


class TestVTKExporterEdgeCases:
    """Tests for edge cases and error handling."""

    def test_mixed_elements(self, mock_node, mock_element, mock_vtk):
        """Test grid with mixed triangle and quad elements."""
        with patch.dict("sys.modules", {"vtk": mock_vtk}):
            from pyiwfm.visualization.vtk_export import VTKExporter

            grid = MagicMock()
            nodes = {
                1: mock_node(1, 0.0, 0.0),
                2: mock_node(2, 100.0, 0.0),
                3: mock_node(3, 100.0, 100.0),
                4: mock_node(4, 0.0, 100.0),
                5: mock_node(5, 50.0, 150.0),
            }
            grid.nodes = nodes
            grid.n_nodes = 5

            elements = [
                mock_element(1, [1, 2, 3, 4], is_triangle=False),  # Quad
                mock_element(2, [3, 4, 5], is_triangle=True),  # Triangle
            ]
            grid.iter_elements.return_value = iter(elements)
            grid.iter_nodes.return_value = iter(nodes.values())

            exporter = VTKExporter(grid)
            exporter.create_2d_mesh()

            # Should create both quad and triangle cells
            mock_vtk.vtkQuad.assert_called()
            mock_vtk.vtkTriangle.assert_called()

    def test_string_path_conversion(self, mock_grid, mock_vtk, tmp_path):
        """Test export handles string paths."""
        with patch.dict("sys.modules", {"vtk": mock_vtk}):
            from pyiwfm.visualization.vtk_export import VTKExporter

            exporter = VTKExporter(mock_grid)
            output_path = str(tmp_path / "test.vtu")

            # Should not raise
            exporter.export_vtu(output_path)

    def test_empty_scalars_dict(self, mock_grid, mock_vtk, tmp_path):
        """Test export with empty scalar dictionaries."""
        with patch.dict("sys.modules", {"vtk": mock_vtk}):
            from pyiwfm.visualization.vtk_export import VTKExporter

            exporter = VTKExporter(mock_grid)
            output_path = tmp_path / "test.vtu"

            # Should not raise with empty dicts
            exporter.export_vtu(output_path, node_scalars={}, cell_scalars={})
