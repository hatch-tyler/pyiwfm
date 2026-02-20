"""
Data export API routes: CSV, GeoJSON downloads.
"""

from __future__ import annotations

import csv
import io
import json

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from pyiwfm.visualization.webapi.config import model_state

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
    type: str = Query(..., description="Type: gw, stream"),
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
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown type: {type}. Use: gw, stream",
        )

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
