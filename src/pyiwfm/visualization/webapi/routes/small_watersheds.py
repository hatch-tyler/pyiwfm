"""
Small watershed data API routes.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from pyiwfm.visualization.webapi.config import model_state

router = APIRouter(prefix="/api/small-watersheds", tags=["small_watersheds"])


@router.get("")
def get_small_watersheds() -> dict:
    """
    Get all small watersheds with connectivity info.

    Each watershed includes a marker_position at the first GW routing node,
    enriched GW node data (max_perc_rate, raw_qmaxwb), and root zone /
    aquifer parameters for the detail panel.
    """
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    model = model_state.model
    if model.small_watersheds is None:
        return {"n_watersheds": 0, "watersheds": []}

    grid = model.grid
    watersheds: list[dict] = []

    for ws in model.small_watersheds.iter_watersheds():
        # Collect GW node positions for this watershed
        gw_coords: list[dict] = []
        for gwn in ws.gw_nodes:
            node = grid.nodes.get(gwn.gw_node_id)
            if node is None:
                continue
            lng, lat = model_state.reproject_coords(node.x, node.y)

            # Reconstruct the raw IWFM QMAXWB value:
            #   baseflow nodes: -layer (negative),  percolation nodes: +max_perc_rate
            raw_qmaxwb = -float(gwn.layer) if gwn.is_baseflow else gwn.max_perc_rate

            gw_coords.append({
                "node_id": gwn.gw_node_id,
                "lng": lng,
                "lat": lat,
                "is_baseflow": gwn.is_baseflow,
                "layer": gwn.layer,
                "max_perc_rate": gwn.max_perc_rate,
                "raw_qmaxwb": raw_qmaxwb,
            })

        if not gw_coords:
            continue

        # Marker position = first GW routing node (typically a baseflow node)
        marker_position = [gw_coords[0]["lng"], gw_coords[0]["lat"]]

        # Find destination stream node coordinates
        dest_coords = None
        if model.streams and ws.dest_stream_node > 0:
            # Direct O(1) lookup in stream.nodes dict
            stream_nodes = getattr(model.streams, "nodes", None) or {}
            sn = stream_nodes.get(ws.dest_stream_node)
            if sn is not None:
                gw_node = getattr(sn, "gw_node", None)
                if gw_node and gw_node in grid.nodes:
                    node = grid.nodes[gw_node]
                    dlng, dlat = model_state.reproject_coords(node.x, node.y)
                    dest_coords = {"lng": dlng, "lat": dlat}

        watersheds.append({
            "id": ws.id,
            "area": ws.area,
            "dest_stream_node": ws.dest_stream_node,
            "dest_coords": dest_coords,
            "marker_position": marker_position,
            "n_gw_nodes": len(gw_coords),
            "gw_nodes": gw_coords,
            "curve_number": ws.curve_number,
            # Root zone parameters
            "wilting_point": ws.wilting_point,
            "field_capacity": ws.field_capacity,
            "total_porosity": ws.total_porosity,
            "lambda_param": ws.lambda_param,
            "root_depth": ws.root_depth,
            "hydraulic_cond": ws.hydraulic_cond,
            "kunsat_method": ws.kunsat_method,
            # Aquifer parameters
            "gw_threshold": ws.gw_threshold,
            "max_gw_storage": ws.max_gw_storage,
            "surface_flow_coeff": ws.surface_flow_coeff,
            "baseflow_coeff": ws.baseflow_coeff,
        })

    return {"n_watersheds": len(watersheds), "watersheds": watersheds}
