"""Mixin providing 3D/surface mesh and slicing methods for ModelState."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pyiwfm.core.model import IWFMModel

logger = logging.getLogger(__name__)


class MeshStateMixin:
    """Mixin providing mesh computation and caching methods for ModelState."""

    # -- Attributes set by ModelState.__init__ (declared for type checkers) --
    _model: IWFMModel | None
    _mesh_3d: bytes | None
    _mesh_surface: bytes | None
    _surface_json_data: dict | None
    _bounds: tuple[float, float, float, float, float, float] | None
    _pv_mesh_3d: object | None
    _layer_surface_cache: dict[int, dict]

    def get_mesh_3d(self) -> bytes:
        """Get the 3D mesh as VTU bytes, computing if needed."""
        if self._mesh_3d is None:
            self._mesh_3d = self._compute_mesh_3d()
        return self._mesh_3d

    def get_mesh_surface(self) -> bytes:
        """Get the surface mesh as VTU bytes, computing if needed."""
        if self._mesh_surface is None:
            self._mesh_surface = self._compute_mesh_surface()
        return self._mesh_surface

    def get_bounds(self) -> tuple[float, float, float, float, float, float]:
        """Get model bounding box."""
        if self._bounds is None:
            self._bounds = self._compute_bounds()
        return self._bounds

    def _compute_mesh_3d(self) -> bytes:
        """Compute 3D mesh as VTU XML string."""
        import vtk

        if self._model is None:
            raise ValueError("No model loaded")

        from pyiwfm.visualization.vtk_export import VTKExporter

        grid = self._model.grid
        if grid is None:
            raise ValueError("No grid loaded")

        exporter = VTKExporter(
            grid=grid,
            stratigraphy=self._model.stratigraphy,
        )
        vtk_grid = exporter.create_3d_mesh()

        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetWriteToOutputString(True)
        writer.SetInputData(vtk_grid)
        writer.Write()

        return cast(bytes, writer.GetOutputString().encode("utf-8"))

    def _compute_mesh_surface(self) -> bytes:
        """Compute surface mesh as VTU XML string."""
        import vtk

        if self._model is None:
            raise ValueError("No model loaded")

        from pyiwfm.visualization.vtk_export import VTKExporter

        grid = self._model.grid
        if grid is None:
            raise ValueError("No grid loaded")

        exporter = VTKExporter(grid=grid)
        vtk_grid = exporter.create_2d_mesh()

        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetWriteToOutputString(True)
        writer.SetInputData(vtk_grid)
        writer.Write()

        return cast(bytes, writer.GetOutputString().encode("utf-8"))

    def get_pyvista_3d(self) -> object:
        """Get the cached PyVista 3D mesh, computing if needed."""
        if self._pv_mesh_3d is None:
            if self._model is None:
                raise ValueError("No model loaded")
            strat = self._model.stratigraphy
            if strat is None:
                raise ValueError("3D mesh requires stratigraphy")
            from pyiwfm.visualization.vtk_export import VTKExporter

            grid = self._model.grid
            if grid is None:
                raise ValueError("No grid loaded")
            exporter = VTKExporter(grid=grid, stratigraphy=strat)
            self._pv_mesh_3d = exporter.to_pyvista_3d()
        return self._pv_mesh_3d

    def get_surface_json(self, layer: int = 0) -> dict:
        """Get the extracted surface mesh as flat JSON-serializable dict.

        Parameters
        ----------
        layer : int
            0 = all layers (default), 1..N = specific layer only.
        """
        if layer in self._layer_surface_cache:
            return self._layer_surface_cache[layer]

        # For layer=0, also check legacy cache
        if layer == 0 and self._surface_json_data is not None:
            return self._surface_json_data

        data = self._compute_surface_json(layer)
        self._layer_surface_cache[layer] = data
        if layer == 0:
            self._surface_json_data = data
        return data

    def _compute_surface_json(self, layer: int = 0) -> dict:
        """Extract the outer surface of the 3D mesh and return flat arrays.

        Parameters
        ----------
        layer : int
            0 = all layers, 1..N = specific layer only.
        """
        import numpy as np

        pv_mesh = self.get_pyvista_3d()

        if layer > 0:
            # Filter to specific layer using threshold
            filtered = pv_mesh.threshold(value=[layer, layer], scalars="layer")  # type: ignore[attr-defined]
            surface = filtered.extract_surface()  # type: ignore[attr-defined]
        else:
            surface = pv_mesh.extract_surface()  # type: ignore[attr-defined]

        # Flat points array: [x0, y0, z0, x1, y1, z1, ...]
        points_flat = surface.points.astype(np.float32).ravel().tolist()

        # Flat polys array in VTK format: [nV, v0, v1, ..., nV, v0, v1, ...]
        polys_flat = surface.faces.tolist()

        # Layer cell data (mapped from volumetric to surface)
        if "layer" in surface.cell_data:
            layer_data = surface.cell_data["layer"].tolist()
        else:
            layer_data = [layer if layer > 0 else 1] * surface.n_cells

        # For single-layer requests, n_layers is the layer number itself
        # For all layers, it's the max layer value
        if layer > 0:
            n_layers = layer
        else:
            n_layers = int(max(layer_data)) if layer_data else 1

        return {
            "n_points": surface.n_points,
            "n_cells": surface.n_cells,
            "n_layers": n_layers,
            "points": points_flat,
            "polys": polys_flat,
            "layer": layer_data,
        }

    def get_slice_json(self, angle: float, position: float) -> dict:
        """Get a cross-section slice as flat JSON-serializable dict.

        Parameters
        ----------
        angle : float
            Angle in degrees from a north-south face.
            0° = north-south cross-section (normal points east),
            90° = east-west cross-section (normal points north).
        position : float
            Normalized position (0-1) along the slice normal.
        """
        import math

        import numpy as np

        from pyiwfm.visualization.webapi.slicing import SlicingController

        pv_mesh = self.get_pyvista_3d()
        slicer = SlicingController(pv_mesh)  # type: ignore[arg-type]

        # Convert angle to normal vector.
        # 0° = N-S face -> normal (1,0,0) (east)
        # 90° = E-W face -> normal (0,1,0) (north)
        rad = math.radians(angle)
        normal = (math.cos(rad), math.sin(rad), 0.0)

        # Convert normalized position to world-space origin
        origin = slicer.normalized_to_position_along(normal, position)

        slice_mesh = slicer.slice_arbitrary(normal=normal, origin=origin)

        if slice_mesh.n_cells == 0:
            return {
                "n_points": 0,
                "n_cells": 0,
                "n_layers": 0,
                "points": [],
                "polys": [],
                "layer": [],
            }

        # The slice is already a PolyData
        points_flat = slice_mesh.points.astype(np.float32).ravel().tolist()
        polys_flat = slice_mesh.faces.tolist()

        if "layer" in slice_mesh.cell_data:
            layer_data = slice_mesh.cell_data["layer"].tolist()
        else:
            layer_data = [1] * slice_mesh.n_cells

        n_layers = int(max(layer_data)) if layer_data else 1

        return {
            "n_points": slice_mesh.n_points,
            "n_cells": slice_mesh.n_cells,
            "n_layers": n_layers,
            "points": points_flat,
            "polys": polys_flat,
            "layer": layer_data,
        }

    def _compute_bounds(self) -> tuple[float, float, float, float, float, float]:
        """Compute model bounding box."""
        if self._model is None:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        grid = self._model.grid
        strat = self._model.stratigraphy

        if grid is None:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        xs = [n.x for n in grid.iter_nodes()]
        ys = [n.y for n in grid.iter_nodes()]

        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        if strat is not None:
            import numpy as np

            zmin = float(np.min(strat.bottom_elev))
            zmax = float(np.max(strat.top_elev))
        else:
            zmin, zmax = 0.0, 0.0

        return (xmin, xmax, ymin, ymax, zmin, zmax)
