"""
Cross-section slice API routes.
"""

from __future__ import annotations

import io
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel

from pyiwfm.visualization.webapi.config import model_state
from pyiwfm.visualization.webapi.routes.mesh import SurfaceMeshData

router = APIRouter(prefix="/api/slice", tags=["slices"])


class SliceInfo(BaseModel):
    """Slice metadata."""

    n_cells: int
    n_points: int
    bounds: list[float] | None


@router.get("")
def get_slice(
    axis: Literal["x", "y", "z"] = Query(default="x", description="Slice axis"),
    position: float = Query(default=0.5, ge=0, le=1, description="Normalized position (0-1)"),
) -> Response:
    """
    Get a cross-section slice as VTU format.

    Parameters
    ----------
    axis : str
        Slice axis ('x', 'y', or 'z')
    position : float
        Normalized position along the axis (0-1)

    Returns
    -------
    VTU file data
    """
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    try:
        import pyvista as pv
    except ImportError:
        raise HTTPException(
            status_code=500, detail="PyVista required for slicing"
        )

    model = model_state.model
    if model.stratigraphy is None:
        raise HTTPException(
            status_code=400, detail="Stratigraphy required for 3D slicing"
        )

    from pyiwfm.visualization.vtk_export import VTKExporter
    from pyiwfm.visualization.webapi.slicing import SlicingController

    exporter = VTKExporter(
        grid=model.grid, stratigraphy=model.stratigraphy
    )
    mesh = exporter.to_pyvista_3d()
    slicer = SlicingController(mesh)

    abs_position = slicer.normalized_to_position(axis, position)

    if axis == "x":
        slice_mesh = slicer.slice_x(abs_position)
    elif axis == "y":
        slice_mesh = slicer.slice_y(abs_position)
    else:
        slice_mesh = slicer.slice_z(abs_position)

    if slice_mesh.n_cells == 0:
        raise HTTPException(status_code=404, detail="Empty slice")

    vtu_data = _pyvista_to_vtu(slice_mesh)

    return Response(
        content=vtu_data,
        media_type="application/xml",
        headers={"Content-Disposition": "attachment; filename=slice.vtu"},
    )


@router.get("/json", response_model=SurfaceMeshData)
def get_slice_json(
    angle: float = Query(
        default=0.0, ge=0, le=180,
        description="Angle in degrees from N-S face (0=N-S, 90=E-W)",
    ),
    position: float = Query(
        default=0.5, ge=0, le=1,
        description="Normalized position (0-1) along slice normal",
    ),
) -> SurfaceMeshData:
    """
    Get a cross-section slice as JSON for vtk.js rendering.

    The slice plane orientation is defined by an angle from a north-south
    face (0° = N-S cross-section, 90° = E-W cross-section) and a
    normalized position (0-1) along the domain in that direction.

    Returns the same flat-array format as /api/mesh/json so the client
    can render the slice as a PolyData actor.
    """
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    try:
        data = model_state.get_slice_json(angle, position)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if data["n_cells"] == 0:
        raise HTTPException(status_code=404, detail="Empty slice")

    return SurfaceMeshData(**data)


@router.get("/cross-section")
def get_cross_section(
    start_x: float = Query(..., description="Start X coordinate"),
    start_y: float = Query(..., description="Start Y coordinate"),
    end_x: float = Query(..., description="End X coordinate"),
    end_y: float = Query(..., description="End Y coordinate"),
) -> Response:
    """
    Get a vertical cross-section between two map points.

    Parameters
    ----------
    start_x, start_y : float
        Starting point coordinates
    end_x, end_y : float
        Ending point coordinates

    Returns
    -------
    VTU file data
    """
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    try:
        import pyvista as pv
    except ImportError:
        raise HTTPException(
            status_code=500, detail="PyVista required for slicing"
        )

    model = model_state.model
    if model.stratigraphy is None:
        raise HTTPException(
            status_code=400, detail="Stratigraphy required for cross-sections"
        )

    from pyiwfm.visualization.vtk_export import VTKExporter
    from pyiwfm.visualization.webapi.slicing import SlicingController

    exporter = VTKExporter(
        grid=model.grid, stratigraphy=model.stratigraphy
    )
    mesh = exporter.to_pyvista_3d()
    slicer = SlicingController(mesh)

    slice_mesh = slicer.create_cross_section(
        start=(start_x, start_y),
        end=(end_x, end_y),
    )

    if slice_mesh.n_cells == 0:
        raise HTTPException(status_code=404, detail="Empty cross-section")

    vtu_data = _pyvista_to_vtu(slice_mesh)

    return Response(
        content=vtu_data,
        media_type="application/xml",
        headers={"Content-Disposition": "attachment; filename=cross_section.vtu"},
    )


@router.get("/cross-section/json")
def get_cross_section_json(
    start_lng: float = Query(..., description="Start longitude (WGS84)"),
    start_lat: float = Query(..., description="Start latitude (WGS84)"),
    end_lng: float = Query(..., description="End longitude (WGS84)"),
    end_lat: float = Query(..., description="End latitude (WGS84)"),
) -> dict:
    """
    Get a vertical cross-section between two WGS84 map points as JSON.

    Accepts WGS84 coordinates (from the 2D map click), converts them
    to model CRS, performs the slice, and returns flat arrays for
    visualization in a Plotly chart.
    """
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    model = model_state.model
    if model.stratigraphy is None:
        raise HTTPException(
            status_code=400, detail="Stratigraphy required for cross-sections"
        )

    # Convert WGS84 coordinates to model CRS
    try:
        from pyproj import Transformer

        transformer = Transformer.from_crs(
            "EPSG:4326",
            model_state._crs,
            always_xy=True,
        )
        start_x, start_y = transformer.transform(start_lng, start_lat)
        end_x, end_y = transformer.transform(end_lng, end_lat)
    except ImportError:
        # No pyproj — assume coordinates are already in model CRS
        start_x, start_y = start_lng, start_lat
        end_x, end_y = end_lng, end_lat

    from pyiwfm.visualization.webapi.slicing import SlicingController

    pv_mesh = model_state.get_pyvista_3d()
    slicer = SlicingController(pv_mesh)

    slice_mesh = slicer.create_cross_section(
        start=(start_x, start_y),
        end=(end_x, end_y),
    )

    if slice_mesh.n_cells == 0:
        return {
            "n_points": 0,
            "n_cells": 0,
            "points": [],
            "layer": [],
            "distance": [],
        }

    import math

    points = slice_mesh.points
    # Compute horizontal distance from start for each point
    distances = []
    for pt in points:
        dx = pt[0] - start_x
        dy = pt[1] - start_y
        distances.append(round(math.sqrt(dx * dx + dy * dy), 2))

    points_flat = points.astype(np.float32).ravel().tolist()

    layer_data = []
    if "layer" in slice_mesh.cell_data:
        layer_data = slice_mesh.cell_data["layer"].tolist()
    else:
        layer_data = [1] * slice_mesh.n_cells

    polys_flat = slice_mesh.faces.tolist()

    return {
        "n_points": slice_mesh.n_points,
        "n_cells": slice_mesh.n_cells,
        "points": points_flat,
        "polys": polys_flat,
        "layer": layer_data,
        "distance": distances,
        "start": {"lng": start_lng, "lat": start_lat, "x": start_x, "y": start_y},
        "end": {"lng": end_lng, "lat": end_lat, "x": end_x, "y": end_y},
        "total_distance": round(
            math.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2), 2
        ),
    }


@router.get("/cross-section/heads")
def get_cross_section_heads(
    start_lng: float = Query(..., description="Start longitude (WGS84)"),
    start_lat: float = Query(..., description="Start latitude (WGS84)"),
    end_lng: float = Query(..., description="End longitude (WGS84)"),
    end_lat: float = Query(..., description="End latitude (WGS84)"),
    timestep: int = Query(default=0, ge=0, description="Timestep index"),
    n_samples: int = Query(default=100, ge=10, le=500, description="Sample points"),
) -> dict:
    """
    Get interpolated groundwater head levels along a cross-section.

    Returns per-layer head elevations clipped to layer geometry, with
    NaN where layers are dry or outside the mesh.
    """
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    model = model_state.model
    if model is None or model.stratigraphy is None:
        raise HTTPException(
            status_code=400, detail="Stratigraphy required for cross-sections"
        )

    loader = model_state.get_head_loader()
    if loader is None or loader.n_frames == 0:
        raise HTTPException(status_code=404, detail="No head data available")

    if timestep >= loader.n_frames:
        raise HTTPException(
            status_code=400,
            detail=f"Timestep {timestep} out of range [0, {loader.n_frames})",
        )

    # Convert WGS84 → model CRS
    try:
        from pyproj import Transformer

        transformer = Transformer.from_crs(
            "EPSG:4326", model_state._crs, always_xy=True,
        )
        start_x, start_y = transformer.transform(start_lng, start_lat)
        end_x, end_y = transformer.transform(end_lng, end_lat)
    except ImportError:
        start_x, start_y = start_lng, start_lat
        end_x, end_y = end_lng, end_lat

    from pyiwfm.core.cross_section import CrossSectionExtractor

    extractor = CrossSectionExtractor(model.grid, model.stratigraphy)
    xs = extractor.extract(
        start=(start_x, start_y), end=(end_x, end_y), n_samples=n_samples,
    )

    # Get head frame: shape (n_nodes, n_layers)
    frame = loader.get_frame(timestep)

    # Interpolate head onto cross-section sample points
    head_interp = extractor.interpolate_layer_property(xs, frame, "head")
    # head_interp shape: (n_samples, n_layers)

    n_layers = xs.n_layers
    dt = loader.times[timestep] if timestep < len(loader.times) else None

    layers_out: list[dict] = []
    for layer_idx in range(n_layers):
        top_vals = xs.top_elev[:, layer_idx].copy()
        bot_vals = xs.bottom_elev[:, layer_idx].copy()
        head_vals = head_interp[:, layer_idx].copy()

        # Clip head: NaN where dry (head < bottom) or IWFM dry marker
        for j in range(len(head_vals)):
            if not xs.mask[j]:
                head_vals[j] = np.nan
            elif head_vals[j] < -9000:
                head_vals[j] = np.nan
            elif head_vals[j] < bot_vals[j]:
                head_vals[j] = np.nan
            elif head_vals[j] > top_vals[j]:
                # Confine head to layer top for display
                head_vals[j] = min(head_vals[j], top_vals[j])

        layers_out.append({
            "layer": layer_idx + 1,
            "top": [round(float(v), 2) if not np.isnan(v) else None for v in top_vals],
            "bottom": [round(float(v), 2) if not np.isnan(v) else None for v in bot_vals],
            "head": [round(float(v), 2) if not np.isnan(v) else None for v in head_vals],
        })

    return {
        "n_samples": n_samples,
        "n_layers": n_layers,
        "distance": [round(float(d), 2) for d in xs.distance],
        "timestep": timestep,
        "datetime": dt.isoformat() if dt else None,
        "layers": layers_out,
        "gs_elev": [
            round(float(v), 2) if not np.isnan(v) else None for v in xs.gs_elev
        ],
        "mask": xs.mask.tolist(),
    }


@router.get("/info", response_model=SliceInfo)
def get_slice_info(
    axis: Literal["x", "y", "z"] = Query(default="x"),
    position: float = Query(default=0.5, ge=0, le=1),
) -> SliceInfo:
    """Get metadata about a slice without returning the full mesh."""
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    try:
        import pyvista as pv
    except ImportError:
        raise HTTPException(
            status_code=500, detail="PyVista required for slicing"
        )

    model = model_state.model
    if model.stratigraphy is None:
        raise HTTPException(
            status_code=400, detail="Stratigraphy required for 3D slicing"
        )

    from pyiwfm.visualization.vtk_export import VTKExporter
    from pyiwfm.visualization.webapi.slicing import SlicingController

    exporter = VTKExporter(
        grid=model.grid, stratigraphy=model.stratigraphy
    )
    mesh = exporter.to_pyvista_3d()
    slicer = SlicingController(mesh)

    abs_position = slicer.normalized_to_position(axis, position)

    if axis == "x":
        slice_mesh = slicer.slice_x(abs_position)
    elif axis == "y":
        slice_mesh = slicer.slice_y(abs_position)
    else:
        slice_mesh = slicer.slice_z(abs_position)

    props = slicer.get_slice_properties(slice_mesh)

    return SliceInfo(
        n_cells=props["n_cells"],
        n_points=props["n_points"],
        bounds=list(props["bounds"]) if props["bounds"] else None,
    )


def _pyvista_to_vtu(mesh: "pv.PolyData | pv.UnstructuredGrid") -> bytes:
    """Convert a PyVista mesh to VTU bytes."""
    import pyvista as pv
    import vtk

    if isinstance(mesh, pv.PolyData):
        vtk_mesh = mesh.cast_to_unstructured_grid()
    else:
        vtk_mesh = mesh

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetWriteToOutputString(True)
    writer.SetInputData(vtk_mesh)
    writer.Write()

    return writer.GetOutputString().encode("utf-8")
