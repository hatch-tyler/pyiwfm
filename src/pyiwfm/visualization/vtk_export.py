"""
VTK export functionality for IWFM models.

This module provides classes for exporting IWFM model data to
VTK formats for 3D visualization in tools like ParaView.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import pyvista as pv
    from pyiwfm.core.mesh import AppGrid
    from pyiwfm.core.stratigraphy import Stratigraphy


class VTKExporter:
    """
    Export IWFM model data to VTK formats.

    This class converts model meshes and stratigraphy to VTK
    UnstructuredGrid objects that can be exported to VTU or
    legacy VTK formats for visualization in ParaView.

    Attributes:
        grid: Model mesh
        stratigraphy: Model stratigraphy (optional, required for 3D)
    """

    def __init__(
        self,
        grid: "AppGrid",
        stratigraphy: "Stratigraphy | None" = None,
    ) -> None:
        """
        Initialize the VTK exporter.

        Args:
            grid: Model mesh
            stratigraphy: Model stratigraphy (optional, required for 3D)
        """
        try:
            import vtk  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "VTK is required for VTK export. "
                "Install with: pip install vtk"
            ) from e

        self.grid = grid
        self.stratigraphy = stratigraphy

    def create_2d_mesh(self) -> "vtk.vtkUnstructuredGrid":
        """
        Create a 2D VTK UnstructuredGrid from the mesh.

        Returns:
            VTK UnstructuredGrid with 2D mesh
        """
        import vtk

        vtk_grid = vtk.vtkUnstructuredGrid()

        # Create points
        points = vtk.vtkPoints()
        node_id_to_vtk_id = {}

        for i, node in enumerate(self.grid.iter_nodes()):
            points.InsertNextPoint(node.x, node.y, 0.0)
            node_id_to_vtk_id[node.id] = i

        vtk_grid.SetPoints(points)

        # Create cells
        for elem in self.grid.iter_elements():
            if elem.is_triangle:
                cell = vtk.vtkTriangle()
                for i, vid in enumerate(elem.vertices):
                    cell.GetPointIds().SetId(i, node_id_to_vtk_id[vid])
            else:  # Quad
                cell = vtk.vtkQuad()
                for i, vid in enumerate(elem.vertices):
                    cell.GetPointIds().SetId(i, node_id_to_vtk_id[vid])

            vtk_grid.InsertNextCell(cell.GetCellType(), cell.GetPointIds())

        return vtk_grid

    def create_3d_mesh(self) -> "vtk.vtkUnstructuredGrid":
        """
        Create a 3D VTK UnstructuredGrid from mesh and stratigraphy.

        Quad elements become hexahedra, triangles become wedges.

        Returns:
            VTK UnstructuredGrid with 3D mesh

        Raises:
            ValueError: If stratigraphy is not set
        """
        import vtk

        if self.stratigraphy is None:
            raise ValueError("Stratigraphy required for 3D mesh")

        vtk_grid = vtk.vtkUnstructuredGrid()

        # Build mapping from node ID to index
        sorted_node_ids = sorted(self.grid.nodes.keys())
        node_id_to_idx = {nid: i for i, nid in enumerate(sorted_node_ids)}

        n_nodes = self.grid.n_nodes
        n_layers = self.stratigraphy.n_layers

        # Create points - nodes at each layer surface
        # Surfaces: top of layer 1, bottom of layer 1/top of layer 2, ..., bottom of last layer
        n_surfaces = n_layers + 1
        points = vtk.vtkPoints()

        # Point ID mapping: [surface_idx, node_idx] -> vtk_point_id
        point_id_map = np.zeros((n_surfaces, n_nodes), dtype=np.int32)
        vtk_pt_id = 0

        for surf_idx in range(n_surfaces):
            for node_idx, node_id in enumerate(sorted_node_ids):
                node = self.grid.nodes[node_id]

                if surf_idx == 0:
                    # Top surface (ground surface / top of layer 1)
                    z = float(self.stratigraphy.top_elev[node_idx, 0])
                else:
                    # Bottom of layer surf_idx (which is top of layer surf_idx+1)
                    z = float(self.stratigraphy.bottom_elev[node_idx, surf_idx - 1])

                points.InsertNextPoint(node.x, node.y, z)
                point_id_map[surf_idx, node_idx] = vtk_pt_id
                vtk_pt_id += 1

        vtk_grid.SetPoints(points)

        # Create cells - one cell per element per layer
        layer_data = []

        for layer in range(n_layers):
            top_surf = layer
            bot_surf = layer + 1

            for elem in self.grid.iter_elements():
                # Get node indices for this element
                node_indices = [node_id_to_idx[vid] for vid in elem.vertices]

                if elem.is_triangle:
                    # Wedge (triangular prism)
                    cell = vtk.vtkWedge()
                    # Bottom triangle (counterclockwise looking up)
                    cell.GetPointIds().SetId(0, point_id_map[bot_surf, node_indices[0]])
                    cell.GetPointIds().SetId(1, point_id_map[bot_surf, node_indices[1]])
                    cell.GetPointIds().SetId(2, point_id_map[bot_surf, node_indices[2]])
                    # Top triangle
                    cell.GetPointIds().SetId(3, point_id_map[top_surf, node_indices[0]])
                    cell.GetPointIds().SetId(4, point_id_map[top_surf, node_indices[1]])
                    cell.GetPointIds().SetId(5, point_id_map[top_surf, node_indices[2]])
                else:
                    # Hexahedron
                    cell = vtk.vtkHexahedron()
                    # Bottom quad (counterclockwise looking up)
                    cell.GetPointIds().SetId(0, point_id_map[bot_surf, node_indices[0]])
                    cell.GetPointIds().SetId(1, point_id_map[bot_surf, node_indices[1]])
                    cell.GetPointIds().SetId(2, point_id_map[bot_surf, node_indices[2]])
                    cell.GetPointIds().SetId(3, point_id_map[bot_surf, node_indices[3]])
                    # Top quad
                    cell.GetPointIds().SetId(4, point_id_map[top_surf, node_indices[0]])
                    cell.GetPointIds().SetId(5, point_id_map[top_surf, node_indices[1]])
                    cell.GetPointIds().SetId(6, point_id_map[top_surf, node_indices[2]])
                    cell.GetPointIds().SetId(7, point_id_map[top_surf, node_indices[3]])

                vtk_grid.InsertNextCell(cell.GetCellType(), cell.GetPointIds())
                layer_data.append(layer + 1)

        # Add layer data as cell attribute
        try:
            from vtk.util.numpy_support import numpy_to_vtk

            layer_np = np.array(layer_data, dtype=np.int32)
            layer_array = numpy_to_vtk(layer_np, deep=True)
        except (ImportError, ModuleNotFoundError):
            layer_array = vtk.vtkIntArray()
            layer_array.SetNumberOfValues(len(layer_data))
            for i, val in enumerate(layer_data):
                layer_array.SetValue(i, val)
        layer_array.SetName("layer")
        vtk_grid.GetCellData().AddArray(layer_array)

        return vtk_grid

    def add_node_scalar(
        self,
        vtk_grid: "vtk.vtkUnstructuredGrid",
        name: str,
        values: NDArray[np.float64],
    ) -> None:
        """
        Add scalar data to mesh nodes.

        Args:
            vtk_grid: VTK grid to add data to
            name: Scalar array name
            values: Scalar values (one per node)
        """
        import vtk

        array = vtk.vtkDoubleArray()
        array.SetName(name)
        array.SetNumberOfTuples(len(values))

        for i, val in enumerate(values):
            array.SetValue(i, float(val))

        vtk_grid.GetPointData().AddArray(array)

    def add_cell_scalar(
        self,
        vtk_grid: "vtk.vtkUnstructuredGrid",
        name: str,
        values: NDArray[np.float64],
    ) -> None:
        """
        Add scalar data to mesh cells.

        Args:
            vtk_grid: VTK grid to add data to
            name: Scalar array name
            values: Scalar values (one per cell)
        """
        import vtk

        array = vtk.vtkDoubleArray()
        array.SetName(name)
        array.SetNumberOfTuples(len(values))

        for i, val in enumerate(values):
            array.SetValue(i, float(val))

        vtk_grid.GetCellData().AddArray(array)

    def export_vtu(
        self,
        output_path: Path | str,
        mode: Literal["2d", "3d"] = "2d",
        node_scalars: dict[str, NDArray[np.float64]] | None = None,
        cell_scalars: dict[str, NDArray[np.float64]] | None = None,
    ) -> None:
        """
        Export mesh to VTU format (XML-based VTK).

        Args:
            output_path: Output file path (.vtu)
            mode: '2d' for surface mesh, '3d' for volumetric mesh
            node_scalars: Dict of name -> values for node data
            cell_scalars: Dict of name -> values for cell data
        """
        import vtk

        output_path = Path(output_path)

        # Create mesh
        if mode == "3d":
            vtk_grid = self.create_3d_mesh()
        else:
            vtk_grid = self.create_2d_mesh()

        # Add scalar data
        if node_scalars:
            for name, values in node_scalars.items():
                self.add_node_scalar(vtk_grid, name, values)

        if cell_scalars:
            for name, values in cell_scalars.items():
                self.add_cell_scalar(vtk_grid, name, values)

        # Write file
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(str(output_path))
        writer.SetInputData(vtk_grid)
        writer.Write()

    def export_vtk(
        self,
        output_path: Path | str,
        mode: Literal["2d", "3d"] = "2d",
        node_scalars: dict[str, NDArray[np.float64]] | None = None,
        cell_scalars: dict[str, NDArray[np.float64]] | None = None,
    ) -> None:
        """
        Export mesh to legacy VTK format.

        Args:
            output_path: Output file path (.vtk)
            mode: '2d' for surface mesh, '3d' for volumetric mesh
            node_scalars: Dict of name -> values for node data
            cell_scalars: Dict of name -> values for cell data
        """
        import vtk

        output_path = Path(output_path)

        # Create mesh
        if mode == "3d":
            vtk_grid = self.create_3d_mesh()
        else:
            vtk_grid = self.create_2d_mesh()

        # Add scalar data
        if node_scalars:
            for name, values in node_scalars.items():
                self.add_node_scalar(vtk_grid, name, values)

        if cell_scalars:
            for name, values in cell_scalars.items():
                self.add_cell_scalar(vtk_grid, name, values)

        # Write file
        writer = vtk.vtkUnstructuredGridWriter()
        writer.SetFileName(str(output_path))
        writer.SetInputData(vtk_grid)
        writer.Write()

    def to_pyvista_3d(
        self,
        node_scalars: dict[str, NDArray[np.float64]] | None = None,
        cell_scalars: dict[str, NDArray[np.float64]] | None = None,
    ) -> "pv.UnstructuredGrid":
        """
        Create a PyVista UnstructuredGrid from mesh and stratigraphy.

        This method converts the IWFM mesh and stratigraphy to a PyVista
        UnstructuredGrid for use in interactive 3D visualization. The
        resulting mesh can be used with PyVista plotting functions or
        the Trame web visualization framework.

        Parameters
        ----------
        node_scalars : dict[str, NDArray], optional
            Dictionary of scalar arrays to add to mesh nodes.
            Keys are array names, values are 1D arrays with one value
            per node (for 2D) or per node-surface point (for 3D).
        cell_scalars : dict[str, NDArray], optional
            Dictionary of scalar arrays to add to mesh cells.
            Keys are array names, values are 1D arrays with one value
            per cell.

        Returns
        -------
        pv.UnstructuredGrid
            PyVista UnstructuredGrid with 3D volumetric mesh if
            stratigraphy is available, otherwise 2D surface mesh.

        Raises
        ------
        ImportError
            If PyVista is not installed.

        Examples
        --------
        Create a 3D mesh for visualization:

        >>> exporter = VTKExporter(grid=grid, stratigraphy=strat)
        >>> pv_mesh = exporter.to_pyvista_3d()
        >>> pv_mesh.plot()

        Add scalar data:

        >>> kh_values = np.random.rand(n_cells)
        >>> pv_mesh = exporter.to_pyvista_3d(cell_scalars={"Kh": kh_values})
        >>> pv_mesh.plot(scalars="Kh", cmap="viridis")
        """
        try:
            import pyvista as pv
        except ImportError as e:
            raise ImportError(
                "PyVista is required for this method. "
                "Install with: pip install pyvista"
            ) from e

        if self.stratigraphy is None:
            return self._to_pyvista_2d(node_scalars, cell_scalars)

        return self._to_pyvista_3d_impl(node_scalars, cell_scalars)

    def _to_pyvista_2d(
        self,
        node_scalars: dict[str, NDArray[np.float64]] | None = None,
        cell_scalars: dict[str, NDArray[np.float64]] | None = None,
    ) -> "pv.UnstructuredGrid":
        """Create a 2D PyVista mesh."""
        import pyvista as pv

        # Build node index mapping
        sorted_node_ids = sorted(self.grid.nodes.keys())
        node_id_to_idx = {nid: i for i, nid in enumerate(sorted_node_ids)}

        # Create points array
        points = np.zeros((len(sorted_node_ids), 3))
        for i, nid in enumerate(sorted_node_ids):
            node = self.grid.nodes[nid]
            points[i] = [node.x, node.y, 0.0]

        # Build cells
        cells = []
        cell_types = []

        for elem in self.grid.iter_elements():
            vertex_indices = [node_id_to_idx[vid] for vid in elem.vertices]
            cells.append(len(vertex_indices))
            cells.extend(vertex_indices)

            if elem.is_triangle:
                cell_types.append(pv.CellType.TRIANGLE)
            else:
                cell_types.append(pv.CellType.QUAD)

        cells = np.array(cells)
        cell_types = np.array(cell_types)

        mesh = pv.UnstructuredGrid(cells, cell_types, points)

        # Add element IDs
        elem_ids = [elem.id for elem in self.grid.iter_elements()]
        mesh.cell_data["element_id"] = np.array(elem_ids)

        # Add custom scalars
        if node_scalars:
            for name, values in node_scalars.items():
                mesh.point_data[name] = values

        if cell_scalars:
            for name, values in cell_scalars.items():
                mesh.cell_data[name] = values

        return mesh

    def _to_pyvista_3d_impl(
        self,
        node_scalars: dict[str, NDArray[np.float64]] | None = None,
        cell_scalars: dict[str, NDArray[np.float64]] | None = None,
    ) -> "pv.UnstructuredGrid":
        """Create a 3D PyVista mesh with stratigraphy."""
        import pyvista as pv

        # Build node index mapping
        sorted_node_ids = sorted(self.grid.nodes.keys())
        node_id_to_idx = {nid: i for i, nid in enumerate(sorted_node_ids)}
        n_nodes = len(sorted_node_ids)
        n_layers = self.stratigraphy.n_layers

        # Create points for all layer surfaces
        n_surfaces = n_layers + 1
        n_total_points = n_nodes * n_surfaces

        points = np.zeros((n_total_points, 3))

        for surf_idx in range(n_surfaces):
            for node_idx, node_id in enumerate(sorted_node_ids):
                node = self.grid.nodes[node_id]
                point_idx = surf_idx * n_nodes + node_idx

                points[point_idx, 0] = node.x
                points[point_idx, 1] = node.y

                if surf_idx == 0:
                    # Top surface (ground surface)
                    points[point_idx, 2] = float(
                        self.stratigraphy.top_elev[node_idx, 0]
                    )
                else:
                    # Bottom of layer surf_idx-1
                    points[point_idx, 2] = float(
                        self.stratigraphy.bottom_elev[node_idx, surf_idx - 1]
                    )

        # Build cells for each element in each layer
        cells = []
        cell_types = []
        layer_data = []
        element_ids = []

        for layer in range(n_layers):
            top_surf_offset = layer * n_nodes
            bot_surf_offset = (layer + 1) * n_nodes

            for elem in self.grid.iter_elements():
                node_indices = [node_id_to_idx[vid] for vid in elem.vertices]

                if elem.is_triangle:
                    # Wedge (triangular prism)
                    cell_types.append(pv.CellType.WEDGE)
                    cells.append(6)
                    # Bottom (layer+1 surface)
                    cells.extend([
                        bot_surf_offset + node_indices[0],
                        bot_surf_offset + node_indices[1],
                        bot_surf_offset + node_indices[2],
                    ])
                    # Top (layer surface)
                    cells.extend([
                        top_surf_offset + node_indices[0],
                        top_surf_offset + node_indices[1],
                        top_surf_offset + node_indices[2],
                    ])
                else:
                    # Hexahedron
                    cell_types.append(pv.CellType.HEXAHEDRON)
                    cells.append(8)
                    # Bottom quad
                    cells.extend([
                        bot_surf_offset + node_indices[0],
                        bot_surf_offset + node_indices[1],
                        bot_surf_offset + node_indices[2],
                        bot_surf_offset + node_indices[3],
                    ])
                    # Top quad
                    cells.extend([
                        top_surf_offset + node_indices[0],
                        top_surf_offset + node_indices[1],
                        top_surf_offset + node_indices[2],
                        top_surf_offset + node_indices[3],
                    ])

                layer_data.append(layer + 1)  # 1-indexed
                element_ids.append(elem.id)

        cells = np.array(cells)
        cell_types = np.array(cell_types)

        mesh = pv.UnstructuredGrid(cells, cell_types, points)

        # Add standard cell data
        mesh.cell_data["layer"] = np.array(layer_data)
        mesh.cell_data["element_id"] = np.array(element_ids)

        # Add custom scalars
        if node_scalars:
            for name, values in node_scalars.items():
                mesh.point_data[name] = values

        if cell_scalars:
            for name, values in cell_scalars.items():
                mesh.cell_data[name] = values

        return mesh
