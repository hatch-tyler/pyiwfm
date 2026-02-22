"""
Data export API routes: CSV, GeoJSON, GeoPackage, and plot downloads.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import tempfile

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from pyiwfm.visualization.webapi.config import model_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/export", tags=["export"])


@router.get("/heads-csv")
def export_heads_csv(
    timestep: int = Query(default=0, ge=0, description="Timestep index"),
    layer: int = Query(default=1, ge=1, description="Layer number (1-based)"),
) -> Response:
    """
    Export head values as a CSV file.

    Returns per-node head values for the specified timestep and layer.
    """
    loader = model_state.get_head_loader()
    if loader is None:
        raise HTTPException(status_code=404, detail="No head data available")

    if timestep >= loader.n_frames:
        raise HTTPException(
            status_code=400,
            detail=f"Timestep {timestep} out of range [0, {loader.n_frames})",
        )

    frame = loader.get_frame(timestep)
    layer_idx = layer - 1
    if layer_idx >= frame.shape[1]:
        raise HTTPException(
            status_code=400,
            detail=f"Layer {layer} out of range [1, {frame.shape[1]}]",
        )

    values = frame[:, layer_idx]
    dt = loader.times[timestep] if timestep < len(loader.times) else None

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["node_id", "head_ft"])
    for i, val in enumerate(values):
        writer.writerow([i + 1, round(float(val), 3)])

    filename = f"heads_ts{timestep}_layer{layer}"
    if dt:
        filename += f"_{dt.strftime('%Y%m%d')}"
    filename += ".csv"

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/mesh-geojson")
def export_mesh_geojson(
    layer: int = Query(default=1, ge=1, description="Layer number (1-based)"),
) -> Response:
    """
    Export the mesh as a GeoJSON file.

    Returns element polygons in WGS84 as a downloadable GeoJSON file.
    """
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    from pyiwfm.visualization.webapi.routes.mesh import get_mesh_geojson

    try:
        geojson = get_mesh_geojson(layer=layer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return Response(
        content=json.dumps(geojson),
        media_type="application/geo+json",
        headers={"Content-Disposition": f"attachment; filename=mesh_layer{layer}.geojson"},
    )


@router.get("/budget-csv")
def export_budget_csv(
    budget_type: str = Query(..., description="Budget type"),
    location: str = Query(default="", description="Location name or index"),
) -> Response:
    """
    Export budget time series data as a CSV file.
    """
    reader = model_state.get_budget_reader(budget_type)
    if reader is None:
        raise HTTPException(
            status_code=404,
            detail=f"Budget type '{budget_type}' not available",
        )

    loc = location if location else 0

    try:
        times_arr, values_arr = reader.get_values(loc)
        headers = reader.get_column_headers(loc)
    except (KeyError, IndexError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    # Build time strings
    ts = reader.header.timestep
    from datetime import timedelta

    use_months = "MON" in ts.unit.upper() if ts.unit else False
    if use_months:
        from dateutil.relativedelta import relativedelta

    time_strings = []
    if ts.start_datetime:
        for i in range(len(times_arr)):
            if use_months:
                dt = ts.start_datetime + relativedelta(months=i)
            else:
                dt = ts.start_datetime + timedelta(minutes=ts.delta_t_minutes * i)
            time_strings.append(dt.isoformat())
    else:
        time_strings = [str(t) for t in times_arr.tolist()]

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["datetime"] + headers)
    for i in range(len(time_strings)):
        row = [time_strings[i]] + [
            round(float(values_arr[i, j]), 4) for j in range(values_arr.shape[1])
        ]
        writer.writerow(row)

    loc_name = location or reader.locations[0]
    safe_name = loc_name.replace(" ", "_").replace("/", "_")
    filename = f"budget_{budget_type}_{safe_name}.csv"

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/hydrograph-csv")
def export_hydrograph_csv(
    type: str = Query(..., description="Type: gw, stream, subsidence, tile_drain"),
    location_id: int = Query(..., description="Location/node ID"),
) -> Response:
    """
    Export hydrograph time series as a CSV file.
    """
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    if type == "gw":
        reader = model_state.get_gw_hydrograph_reader()
        if reader is None or reader.n_timesteps == 0:
            raise HTTPException(status_code=404, detail="No GW hydrograph data available")

        phys_locs = model_state.get_gw_physical_locations()
        location_index = location_id - 1

        if phys_locs:
            if location_index < 0 or location_index >= len(phys_locs):
                raise HTTPException(
                    status_code=404,
                    detail=f"GW hydrograph {location_id} out of range",
                )
            col_idx = phys_locs[location_index]["columns"][0][0]
        else:
            # Fallback: raw column index
            col_idx = location_index

        if col_idx < 0 or col_idx >= reader.n_columns:
            raise HTTPException(
                status_code=404,
                detail=f"GW hydrograph {location_id} out of range",
            )

        times, values = reader.get_time_series(col_idx)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["datetime", "head_ft"])
        for t, v in zip(times, values, strict=False):
            writer.writerow([t, round(v, 3)])

        filename = f"hydrograph_gw_{location_id}.csv"

    elif type == "stream":
        reader = model_state.get_stream_hydrograph_reader()
        if reader is None or reader.n_timesteps == 0:
            raise HTTPException(
                status_code=404,
                detail="No stream hydrograph data available",
            )

        col_idx = reader.find_column_by_node_id(location_id)
        if col_idx is None and location_id in reader.hydrograph_ids:
            col_idx = reader.hydrograph_ids.index(location_id)
        if col_idx is None:
            raise HTTPException(
                status_code=404,
                detail=f"Stream node {location_id} not found",
            )

        times, values = reader.get_time_series(col_idx)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["datetime", "flow_cfs"])
        for t, v in zip(times, values, strict=False):
            writer.writerow([t, round(v, 3)])

        filename = f"hydrograph_stream_{location_id}.csv"

    elif type == "subsidence":
        reader = model_state.get_subsidence_reader()
        if reader is None or reader.n_timesteps == 0:
            raise HTTPException(
                status_code=404,
                detail="No subsidence hydrograph data available",
            )

        col_idx = location_id - 1
        if col_idx < 0 or col_idx >= reader.n_columns:
            raise HTTPException(
                status_code=404,
                detail=f"Subsidence location {location_id} out of range",
            )

        times, values = reader.get_time_series(col_idx)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["datetime", "subsidence_ft"])
        for t, v in zip(times, values, strict=False):
            writer.writerow([t, round(v, 3)])

        filename = f"hydrograph_subsidence_{location_id}.csv"

    elif type == "tile_drain":
        reader = model_state.get_tile_drain_reader()
        if reader is None or reader.n_timesteps == 0:
            raise HTTPException(
                status_code=404,
                detail="No tile drain hydrograph data available",
            )

        col_idx = location_id - 1
        if col_idx < 0 or col_idx >= reader.n_columns:
            raise HTTPException(
                status_code=404,
                detail=f"Tile drain location {location_id} out of range",
            )

        times, values = reader.get_time_series(col_idx)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["datetime", "flow_volume"])
        for t, v in zip(times, values, strict=False):
            writer.writerow([t, round(v, 3)])

        filename = f"hydrograph_tile_drain_{location_id}.csv"

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown type: {type}. Use: gw, stream, subsidence, tile_drain",
        )

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/geopackage")
def export_geopackage(
    include_streams: bool = Query(default=True, description="Include stream reaches"),
    include_subregions: bool = Query(default=True, description="Include subregion polygons"),
    include_boundary: bool = Query(default=True, description="Include model boundary"),
) -> Response:
    """Export the model mesh as a GeoPackage file.

    Creates a multi-layer GeoPackage containing nodes, elements,
    and optionally streams, subregions, and boundary polygon.
    """
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    model = model_state.model
    if model is None or model.grid is None:
        raise HTTPException(status_code=404, detail="No mesh/grid loaded")

    from pyiwfm.visualization.gis_export import GISExporter

    exporter = GISExporter(
        grid=model.grid,
        stratigraphy=model.stratigraphy,
        streams=model.streams,
        crs=model_state._crs,
    )

    with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        exporter.export_geopackage(
            tmp_path,
            include_streams=include_streams,
            include_subregions=include_subregions,
            include_boundary=include_boundary,
        )

        from pathlib import Path

        data = Path(tmp_path).read_bytes()

        model_name = model.name or "model"
        safe_name = model_name.replace(" ", "_").replace("/", "_")
        filename = f"{safe_name}.gpkg"

        return Response(
            content=data,
            media_type="application/geopackage+sqlite3",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    except Exception as e:
        logger.exception("GeoPackage export failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        import os

        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@router.get("/plot/{plot_type}")
def export_plot(
    plot_type: str,
    format: str = Query(default="png", description="Image format: png or svg"),
    layer: int = Query(default=1, ge=1, description="Layer number (1-based)"),
    timestep: int = Query(default=0, ge=0, description="Timestep index"),
    width: float = Query(default=10.0, gt=0, description="Figure width in inches"),
    height: float = Query(default=8.0, gt=0, description="Figure height in inches"),
    dpi: int = Query(default=150, ge=72, le=600, description="DPI for PNG output"),
) -> Response:
    """Generate publication-quality matplotlib figures.

    Supported plot types:
    - mesh: Model mesh with elements and nodes
    - heads: Head contour map for a timestep/layer
    - streams: Stream network colored by reach
    - elements: Elements colored by subregion
    """
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    model = model_state.model
    if model is None or model.grid is None:
        raise HTTPException(status_code=404, detail="No mesh/grid loaded")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pyiwfm.visualization.plotting import (
        plot_elements,
        plot_mesh,
        plot_scalar_field,
        plot_streams,
    )

    try:
        fig = None
        if plot_type == "mesh":
            fig, _ax = plot_mesh(model.grid, figsize=(width, height))
        elif plot_type == "elements":
            fig, _ax = plot_elements(model.grid, figsize=(width, height))
        elif plot_type == "streams":
            if model.streams is None:
                raise HTTPException(status_code=404, detail="No stream network loaded")
            fig, _ax = plot_streams(model.streams, figsize=(width, height))
        elif plot_type == "heads":
            loader = model_state.get_head_loader()
            if loader is None:
                raise HTTPException(status_code=404, detail="No head data available")
            if timestep >= loader.n_frames:
                raise HTTPException(
                    status_code=400,
                    detail=f"Timestep {timestep} out of range [0, {loader.n_frames})",
                )
            import numpy as np

            frame = loader.get_frame(timestep)
            layer_idx = layer - 1
            if layer_idx >= frame.shape[1]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Layer {layer} out of range [1, {frame.shape[1]}]",
                )
            values = frame[:, layer_idx]
            # Mask dry cells
            values = np.where(values < -9000, np.nan, values)
            fig, _ax = plot_scalar_field(
                model.grid,
                values,
                figsize=(width, height),
            )
            _ax.set_title(f"Head - Layer {layer}, Timestep {timestep}")
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown plot type: {plot_type}. Supported: mesh, elements, streams, heads",
            )

        buf = io.BytesIO()
        if format == "svg":
            fig.savefig(buf, format="svg", bbox_inches="tight")
            media_type = "image/svg+xml"
            ext = "svg"
        else:
            fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
            media_type = "image/png"
            ext = "png"
        plt.close(fig)
        buf.seek(0)

        filename = f"{plot_type}_layer{layer}_ts{timestep}.{ext}"
        return Response(
            content=buf.getvalue(),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Plot generation failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
