"""
Stream network API routes.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel

from pyiwfm.visualization.webapi.config import model_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/streams", tags=["streams"])


class StreamNode(BaseModel):
    """Stream node data."""

    id: int
    x: float
    y: float
    z: float
    reach_id: int


class StreamNetwork(BaseModel):
    """Stream network response."""

    n_nodes: int
    n_reaches: int
    nodes: list[StreamNode]
    reaches: list[list[int]]


def _get_gs_elev_lookup(grid, strat) -> dict[int, int]:
    """Build node-ID -> index lookup for ground surface elevation."""
    lookup: dict[int, int] = {}
    if strat is not None:
        for idx, n in enumerate(grid.iter_nodes()):
            lookup[n.id] = idx
    return lookup


def _node_z(gw_node: int, strat, lookup: dict[int, int]) -> float:
    """Look up ground surface elevation for a GW node."""
    if strat is not None and gw_node in lookup:
        return float(strat.gs_elev[lookup[gw_node]])
    return 0.0


def _make_stream_node(
    sn, grid, strat, lookup: dict[int, int], reach_id: int = 0,
) -> StreamNode | None:
    """Resolve a stream node's GW node to grid coordinates and build a StreamNode.

    Returns None if the stream node has no valid gw_node in the grid.
    """
    gw_node = getattr(sn, "gw_node", None)
    if gw_node is None or gw_node not in grid.nodes:
        return None
    nd = grid.nodes[gw_node]
    z = _node_z(gw_node, strat, lookup)
    return StreamNode(
        id=sn.id, x=nd.x, y=nd.y, z=z,
        reach_id=reach_id or getattr(sn, "reach_id", 0),
    )


# ===================================================================
# Shared reach-building logic (used by all 3 stream endpoints)
# ===================================================================


def _build_reaches_from_connectivity(stream, grid, strat, lookup):
    """Build reaches by tracing upstream_node/downstream_node links.

    Returns (nodes_data, reaches_data) or None if connectivity not populated.
    """
    # Build downstream adjacency: stream_node_id -> downstream_stream_node_id
    downstream: dict[int, int] = {}
    has_upstream: set[int] = set()
    valid_nodes: dict[int, object] = {}  # stream_node_id -> StrmNode

    for sn in stream.nodes.values():
        gw = getattr(sn, "gw_node", None)
        if gw is None or gw not in grid.nodes:
            continue
        valid_nodes[sn.id] = sn
        dn = getattr(sn, "downstream_node", None)
        if dn and dn in stream.nodes:
            downstream[sn.id] = dn
            has_upstream.add(dn)

    if not downstream:
        return None  # Connectivity not populated, signal fallback

    # Count upstream connections for confluence detection
    upstream_count: dict[int, int] = defaultdict(int)
    for dn in downstream.values():
        upstream_count[dn] += 1

    # Find headwater nodes (no upstream parent)
    head_nodes = sorted(set(valid_nodes.keys()) - has_upstream)

    nodes_data: list[StreamNode] = []
    reaches_data: list[list[int]] = []
    visited: set[int] = set()
    added_nodes: set[int] = set()

    def _add_node(nid: int) -> None:
        if nid in added_nodes or nid not in valid_nodes:
            return
        sn_node = _make_stream_node(valid_nodes[nid], grid, strat, lookup)
        if sn_node is not None:
            nodes_data.append(sn_node)
            added_nodes.add(nid)

    for head in head_nodes:
        if head in visited:
            continue
        current_reach: list[int] = [head]
        visited.add(head)
        node = head
        while node in downstream:
            nxt = downstream[node]
            if nxt in visited:
                # Append to reach so the line connects to the confluence
                current_reach.append(nxt)
                break
            if upstream_count.get(nxt, 0) > 1 and len(current_reach) > 0:
                # Confluence — end this reach including the junction node, start new one
                current_reach.append(nxt)
                if len(current_reach) >= 2:
                    reaches_data.append(current_reach)
                    for nid in current_reach:
                        _add_node(nid)
                current_reach = [nxt]
            else:
                current_reach.append(nxt)
            visited.add(nxt)
            node = nxt
        # Emit final reach segment
        if len(current_reach) >= 2:
            reaches_data.append(current_reach)
            for nid in current_reach:
                _add_node(nid)

    # Handle unvisited nodes (cycles, disconnected segments)
    unvisited = set(valid_nodes.keys()) - visited
    if unvisited:
        logger.debug("Stream connectivity: %d unvisited nodes", len(unvisited))

    return nodes_data, reaches_data


def _build_reaches_from_preprocessor_binary(stream, grid, strat, lookup):
    """Build reaches using reach boundaries from the preprocessor binary.

    The preprocessor binary stores (reach_id, upstream_node, downstream_node)
    boundaries. Stream nodes with IDs in [upstream_node, downstream_node]
    belong to that reach (IWFM numbers stream nodes contiguously within reaches).

    Returns (nodes_data, reaches_data) or None if boundaries not available.
    """
    boundaries = model_state.get_stream_reach_boundaries()
    if not boundaries:
        return None

    # Build fast lookup: stream_node_id -> StrmNode (only those with valid gw_node)
    valid_nodes: dict[int, object] = {}
    for sn in stream.nodes.values():
        gw = getattr(sn, "gw_node", None)
        if gw is not None and gw in grid.nodes:
            valid_nodes[sn.id] = sn

    if not valid_nodes:
        return None

    nodes_data: list[StreamNode] = []
    reaches_data: list[list[int]] = []
    added_nodes: set[int] = set()

    # Sort node IDs once for efficient range lookups
    sorted_ids = sorted(valid_nodes.keys())

    for reach_id, up_node, dn_node in boundaries:
        # Stream node IDs within a reach are contiguous in [min, max]
        min_id = min(up_node, dn_node)
        max_id = max(up_node, dn_node)

        # Collect valid stream nodes in this range, ordered by ID
        reach_sn_ids: list[int] = [
            sn_id for sn_id in sorted_ids if min_id <= sn_id <= max_id
        ]

        if len(reach_sn_ids) < 2:
            continue

        for sn_id in reach_sn_ids:
            if sn_id not in added_nodes:
                sn_node = _make_stream_node(
                    valid_nodes[sn_id], grid, strat, lookup, reach_id=reach_id,
                )
                if sn_node is not None:
                    nodes_data.append(sn_node)
                    added_nodes.add(sn_id)

        reaches_data.append(reach_sn_ids)

    if not nodes_data:
        return None

    logger.info(
        "Built %d stream reaches from preprocessor binary (%d nodes)",
        len(reaches_data), len(nodes_data),
    )
    return nodes_data, reaches_data


def _build_streams_from_nodes(stream, grid, strat, lookup):
    """Build stream nodes/reaches from the nodes dict (fallback when reaches empty).

    Groups stream nodes by reach_id and orders by node ID within each group.
    Only used when reach_id values are actually populated (not all 0).
    """
    nodes_data: list[StreamNode] = []
    reaches_data: list[list[int]] = []

    # Group by reach_id, order by node ID within each group
    by_reach: dict[int, list] = defaultdict(list)
    for sn in sorted(stream.nodes.values(), key=lambda n: n.id):
        rid = getattr(sn, "reach_id", 0)
        gw_node = getattr(sn, "gw_node", None)
        if gw_node is not None and gw_node in grid.nodes:
            by_reach[rid].append(sn)

    for rid in sorted(by_reach.keys()):
        reach_nodes: list[int] = []
        for sn in by_reach[rid]:
            sn_node = _make_stream_node(sn, grid, strat, lookup, reach_id=rid)
            if sn_node is not None:
                nodes_data.append(sn_node)
                reach_nodes.append(sn.id)
        if len(reach_nodes) >= 2:
            reaches_data.append(reach_nodes)

    return nodes_data, reaches_data


def _build_streams_from_reaches(stream, grid, strat, lookup):
    """Build stream nodes/reaches from the reaches dict/list (original path)."""
    nodes_data: list[StreamNode] = []
    reaches_data: list[list[int]] = []

    reaches = stream.reaches
    items = reaches.values() if isinstance(reaches, dict) else reaches

    for reach in items:
        reach_nodes: list[int] = []
        stream_nodes = getattr(reach, "stream_nodes", None) or getattr(reach, "nodes", [])
        for sn_or_id in stream_nodes:
            if isinstance(sn_or_id, int):
                # StrmReach.nodes stores int IDs — resolve via stream.nodes dict
                sn_id = sn_or_id
                sn_obj = stream.nodes.get(sn_id) if hasattr(stream, "nodes") else None
                if sn_obj is None:
                    continue
                gw_node = getattr(sn_obj, "gw_node", None)
            else:
                sn_id = sn_or_id.id
                gw_node = (
                    getattr(sn_or_id, "groundwater_node", None)
                    or getattr(sn_or_id, "gw_node", None)
                )
            if gw_node is not None and gw_node in grid.nodes:
                sn_node = StreamNode(
                    id=sn_id,
                    x=grid.nodes[gw_node].x,
                    y=grid.nodes[gw_node].y,
                    z=_node_z(gw_node, strat, lookup),
                    reach_id=getattr(reach, "id", 0),
                )
                nodes_data.append(sn_node)
                reach_nodes.append(sn_id)
        if reach_nodes:
            reaches_data.append(reach_nodes)

    return nodes_data, reaches_data


def _build_stream_data(stream, grid, strat, lookup):
    """Build stream reach data using the best available method.

    Priority:
    1. Populated reaches (if stream.reaches is non-empty)
    2. Preprocessor binary reach boundaries
    3. Connectivity tracing (upstream/downstream_node attributes)
    4. Reach_id grouping (only if reach_ids are non-zero)
    5. Single reach fallback (all nodes ordered by ID)
    """
    # 1. Try populated reaches first
    if hasattr(stream, "reaches") and stream.reaches:
        result = _build_streams_from_reaches(stream, grid, strat, lookup)
        if result[0]:  # nodes_data non-empty
            logger.info(
                "Stream strategy 1 (populated reaches): %d reaches, %d nodes",
                len(result[1]), len(result[0]),
            )
            return result

    if hasattr(stream, "nodes") and stream.nodes:
        # 2. Try preprocessor binary reach boundaries
        result = _build_reaches_from_preprocessor_binary(stream, grid, strat, lookup)
        if result is not None and result[0]:
            logger.info(
                "Stream strategy 2 (preprocessor binary): %d reaches, %d nodes",
                len(result[1]), len(result[0]),
            )
            return result

        # 3. Try connectivity tracing
        result = _build_reaches_from_connectivity(stream, grid, strat, lookup)
        if result is not None and result[0]:
            logger.info(
                "Stream strategy 3 (connectivity tracing): %d reaches, %d nodes",
                len(result[1]), len(result[0]),
            )
            return result

        # 4. Try reach_id grouping (only if some reach_ids are > 0)
        has_nonzero_reach_ids = any(
            getattr(sn, "reach_id", 0) > 0
            for sn in stream.nodes.values()
        )
        if has_nonzero_reach_ids:
            result = _build_streams_from_nodes(stream, grid, strat, lookup)
            if result[0]:
                logger.info(
                    "Stream strategy 4 (reach_id grouping): %d reaches, %d nodes",
                    len(result[1]), len(result[0]),
                )
                return result

        # 5. Single-reach fallback: all valid nodes in one reach, ordered by ID
        logger.warning(
            "Stream strategy 5 (single-reach fallback): all %d valid stream "
            "nodes placed in one reach. This likely means reach enrichment "
            "failed during model loading.",
            sum(
                1 for sn in stream.nodes.values()
                if getattr(sn, "gw_node", None) is not None
                and getattr(sn, "gw_node", None) in grid.nodes
            ),
        )
        nodes_data: list[StreamNode] = []
        reach_nodes: list[int] = []
        for sn in sorted(stream.nodes.values(), key=lambda n: n.id):
            sn_node = _make_stream_node(sn, grid, strat, lookup, reach_id=0)
            if sn_node is not None:
                nodes_data.append(sn_node)
                reach_nodes.append(sn.id)
        reaches_data = [reach_nodes] if len(reach_nodes) >= 2 else []
        return nodes_data, reaches_data

    return [], []


def _get_gw_nodes_for_reaches(stream, grid, strat, lookup):
    """Build reach data as (reach_id, name, gw_node_list) tuples for GeoJSON/VTP.

    Uses the same priority logic as _build_stream_data.
    """
    nodes_data, reaches_data = _build_stream_data(stream, grid, strat, lookup)

    # Build stream_node_id -> gw_node lookup
    sn_to_gw: dict[int, int] = {}
    if hasattr(stream, "nodes") and stream.nodes:
        for sn in stream.nodes.values():
            gw = getattr(sn, "gw_node", None)
            if gw is not None:
                sn_to_gw[sn.id] = gw
    if hasattr(stream, "reaches") and stream.reaches:
        items = stream.reaches.values() if isinstance(stream.reaches, dict) else stream.reaches
        for reach in items:
            stream_nodes = getattr(reach, "stream_nodes", None) or getattr(reach, "nodes", [])
            for sn in stream_nodes:
                if isinstance(sn, int):
                    # Int node IDs — already covered by stream.nodes lookup above
                    continue
                gw = getattr(sn, "groundwater_node", None) or getattr(sn, "gw_node", None)
                if gw is not None:
                    sn_to_gw[sn.id] = gw

    # Build stream_node_id -> reach_id from nodes_data for naming
    sn_to_reach: dict[int, int] = {}
    for nd in nodes_data:
        sn_to_reach[nd.id] = nd.reach_id

    result: list[tuple[int, str, list[int]]] = []
    for reach_sn_ids in reaches_data:
        if not reach_sn_ids:
            continue
        rid = sn_to_reach.get(reach_sn_ids[0], 0)
        gw_nodes = [sn_to_gw[sn_id] for sn_id in reach_sn_ids if sn_id in sn_to_gw]
        name = f"Reach {rid}" if rid > 0 else f"Reach (nodes {reach_sn_ids[0]}-{reach_sn_ids[-1]})"
        result.append((rid, name, gw_nodes))

    return result


# ===================================================================
# Endpoints
# ===================================================================


@router.get("", response_model=StreamNetwork)
def get_streams() -> StreamNetwork:
    """Get the stream network as JSON."""
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    model = model_state.model
    if not model.has_streams or model.streams is None:
        raise HTTPException(status_code=404, detail="No stream data in model")

    stream = model.streams
    grid = model.grid
    strat = model.stratigraphy
    lookup = _get_gs_elev_lookup(grid, strat)

    nodes_data, reaches_data = _build_stream_data(stream, grid, strat, lookup)

    return StreamNetwork(
        n_nodes=len(nodes_data),
        n_reaches=len(reaches_data),
        nodes=nodes_data,
        reaches=reaches_data,
    )


@router.get("/geojson")
def get_streams_geojson() -> dict:
    """
    Get the stream network as GeoJSON LineStrings in WGS84.

    Returns a FeatureCollection with one LineString per reach,
    suitable for deck.gl rendering on the 2D Results Map.
    """
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    model = model_state.model
    if not model.has_streams or model.streams is None:
        return {"type": "FeatureCollection", "features": []}

    stream = model.streams
    grid = model.grid
    strat = model.stratigraphy
    lookup = _get_gs_elev_lookup(grid, strat)

    reach_info = _get_gw_nodes_for_reaches(stream, grid, strat, lookup)
    features: list[dict] = []

    for rid, name, gw_nodes in reach_info:
        coords: list[list[float]] = []
        for gw_node in gw_nodes:
            if gw_node in grid.nodes:
                node = grid.nodes[gw_node]
                lng, lat = model_state.reproject_coords(node.x, node.y)
                coords.append([lng, lat])
        if len(coords) >= 2:
            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {
                    "reach_id": rid,
                    "name": name,
                    "n_nodes": len(coords),
                },
            })

    return {"type": "FeatureCollection", "features": features}


@router.get("/vtp")
def get_streams_vtp() -> Response:
    """Get the stream network as VTP (VTK PolyData) format."""
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    model = model_state.model
    if not model.has_streams or model.streams is None:
        raise HTTPException(status_code=404, detail="No stream data in model")

    import vtk

    stream = model.streams
    grid = model.grid
    strat = model.stratigraphy
    lookup = _get_gs_elev_lookup(grid, strat)

    reach_info = _get_gw_nodes_for_reaches(stream, grid, strat, lookup)

    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    reach_ids = vtk.vtkIntArray()
    reach_ids.SetName("reach_id")

    point_map: dict[tuple[float, float], int] = {}
    point_idx = 0

    for rid, _name, gw_nodes in reach_info:
        line_pts: list[int] = []
        for gw_node in gw_nodes:
            if gw_node not in grid.nodes:
                continue
            node = grid.nodes[gw_node]
            key = (node.x, node.y)
            if key not in point_map:
                z = _node_z(gw_node, strat, lookup)
                points.InsertNextPoint(node.x, node.y, z)
                point_map[key] = point_idx
                point_idx += 1
            line_pts.append(point_map[key])

        if len(line_pts) >= 2:
            line = vtk.vtkPolyLine()
            line.GetPointIds().SetNumberOfIds(len(line_pts))
            for i, pt_id in enumerate(line_pts):
                line.GetPointIds().SetId(i, pt_id)
            lines.InsertNextCell(line)
            reach_ids.InsertNextValue(rid)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    polydata.GetCellData().AddArray(reach_ids)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetWriteToOutputString(True)
    writer.SetInputData(polydata)
    writer.Write()

    vtp_data = writer.GetOutputString().encode("utf-8")

    return Response(
        content=vtp_data,
        media_type="application/xml",
        headers={"Content-Disposition": "attachment; filename=streams.vtp"},
    )


@router.get("/diversions")
def get_diversions() -> dict:
    """
    Get diversion routing as arcs with WGS84 source/destination coordinates.

    Returns arcs from stream diversion points to their delivery destinations,
    suitable for deck.gl ArcLayer rendering.
    """
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    model = model_state.model
    if not model.has_streams or model.streams is None:
        return {"n_diversions": 0, "diversions": []}

    stream = model.streams
    grid = model.grid

    if not hasattr(stream, "diversions") or not stream.diversions:
        return {"n_diversions": 0, "diversions": []}

    # Build stream node -> GW node lookup for coordinate resolution
    sn_to_gw: dict[int, int] = {}
    if hasattr(stream, "nodes") and stream.nodes:
        for sn in stream.nodes.values():
            gw = getattr(sn, "gw_node", None)
            if gw is not None:
                sn_to_gw[sn.id] = gw
    elif hasattr(stream, "reaches") and stream.reaches:
        reaches = stream.reaches
        items = reaches.values() if isinstance(reaches, dict) else reaches
        for reach in items:
            stream_nodes = (
                getattr(reach, "stream_nodes", None)
                or getattr(reach, "nodes", [])
            )
            for sn in stream_nodes:
                if isinstance(sn, int):
                    continue  # int IDs — no gw_node info here
                gw = getattr(sn, "groundwater_node", None) or getattr(sn, "gw_node", None)
                if gw is not None:
                    sn_to_gw[sn.id] = gw

    diversions: list[dict] = []
    for div_id, div in stream.diversions.items():
        # Source: stream node -> GW node -> coordinates
        # source_node=0 means diversion from outside the model area (IRDV=0)
        src_lng, src_lat = None, None
        src_gw = sn_to_gw.get(div.source_node)
        if src_gw is not None and src_gw in grid.nodes:
            src_node = grid.nodes[src_gw]
            src_lng, src_lat = model_state.reproject_coords(src_node.x, src_node.y)

        # Destination coordinates depend on destination_type
        dst_lng, dst_lat = None, None
        if div.destination_type == "element" and div.destination_id in grid.elements:
            # Centroid of the destination element
            elem = grid.elements[div.destination_id]
            xs, ys = [], []
            for vid in elem.vertices:
                n = grid.nodes.get(vid)
                if n:
                    xs.append(n.x)
                    ys.append(n.y)
            if xs:
                cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
                dst_lng, dst_lat = model_state.reproject_coords(cx, cy)
        elif div.destination_type == "stream_node":
            dst_gw = sn_to_gw.get(div.destination_id)
            if dst_gw and dst_gw in grid.nodes:
                n = grid.nodes[dst_gw]
                dst_lng, dst_lat = model_state.reproject_coords(n.x, n.y)

        diversions.append({
            "id": div_id,
            "name": div.name or f"Diversion {div_id}",
            "source_node": div.source_node,
            "source": (
                {"lng": src_lng, "lat": src_lat}
                if src_lng is not None
                else None
            ),
            "destination_type": div.destination_type,
            "destination_id": div.destination_id,
            "destination": (
                {"lng": dst_lng, "lat": dst_lat}
                if dst_lng is not None
                else None
            ),
            "max_rate": (
                div.max_rate if div.max_rate < 1e30 else None
            ),
            "priority": div.priority,
        })

    return {"n_diversions": len(diversions), "diversions": diversions}


@router.get("/diversions/{div_id}")
def get_diversion_detail(div_id: int) -> dict:
    """
    Get detailed information about a specific diversion.

    Returns metadata, delivery area as GeoJSON, and timeseries data.
    """
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    model = model_state.model
    if not model.has_streams or model.streams is None:
        raise HTTPException(status_code=404, detail="No stream data in model")

    stream = model.streams
    grid = model.grid

    if not hasattr(stream, "diversions") or not stream.diversions:
        raise HTTPException(status_code=404, detail="No diversions in model")

    div = stream.diversions.get(div_id)
    if div is None:
        raise HTTPException(status_code=404, detail=f"Diversion {div_id} not found")

    # Source coordinates
    sn_to_gw: dict[int, int] = {}
    if hasattr(stream, "nodes") and stream.nodes:
        for sn in stream.nodes.values():
            gw = getattr(sn, "gw_node", None)
            if gw is not None:
                sn_to_gw[sn.id] = gw

    src_lng, src_lat = None, None
    src_gw = sn_to_gw.get(div.source_node)
    if src_gw is not None and src_gw in grid.nodes:
        src_node = grid.nodes[src_gw]
        src_lng, src_lat = model_state.reproject_coords(src_node.x, src_node.y)

    # Destination coordinates (same logic as get_diversions)
    dst_lng, dst_lat = None, None
    if div.destination_type == "element" and div.destination_id in grid.elements:
        elem = grid.elements[div.destination_id]
        xs, ys = [], []
        for vid in elem.vertices:
            n = grid.nodes.get(vid)
            if n:
                xs.append(n.x)
                ys.append(n.y)
        if xs:
            cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
            dst_lng, dst_lat = model_state.reproject_coords(cx, cy)
    elif div.destination_type == "stream_node":
        dst_gw = sn_to_gw.get(div.destination_id)
        if dst_gw and dst_gw in grid.nodes:
            n = grid.nodes[dst_gw]
            dst_lng, dst_lat = model_state.reproject_coords(n.x, n.y)

    # Resolve delivery area element IDs
    element_ids: list[int] = []
    dest_type = div.destination_type
    delivery_dest_id = getattr(div, "delivery_dest_id", div.destination_id)

    if dest_type == "element_set":
        # Look up element group by delivery_dest_id
        eg_list = getattr(stream, "diversion_element_groups", []) or []
        eg_map = {eg.id: eg.elements for eg in eg_list}
        element_ids = eg_map.get(delivery_dest_id, [])
    elif dest_type == "element":
        element_ids = [div.destination_id] if div.destination_id in grid.elements else []
    elif dest_type == "subregion":
        # Collect elements in the matching subregion
        for elem in grid.iter_elements():
            if getattr(elem, "subregion", 0) == div.destination_id:
                element_ids.append(elem.id)

    # Build GeoJSON FeatureCollection for delivery area polygons
    delivery_geojson = None
    if element_ids:
        features: list[dict] = []
        for eid in element_ids:
            elem = grid.elements.get(eid)
            if elem is None:
                continue
            ring: list[list[float]] = []
            for vid in elem.vertices:
                n = grid.nodes.get(vid)
                if n:
                    lng, lat = model_state.reproject_coords(n.x, n.y)
                    ring.append([lng, lat])
            if ring:
                ring.append(ring[0])  # Close ring
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                    "properties": {"element_id": eid},
                })
        if features:
            delivery_geojson = {"type": "FeatureCollection", "features": features}

    # Extract timeseries data
    timeseries = None
    ts_data = model_state.get_diversion_timeseries()
    if ts_data is not None:
        import numpy as np

        times, values, _meta = ts_data
        max_col = getattr(div, "max_div_column", 0)
        del_col = getattr(div, "delivery_column", 0)
        max_frac = getattr(div, "max_div_fraction", 1.0)
        del_frac = getattr(div, "delivery_fraction", 1.0)

        n_cols = values.shape[1] if values.ndim > 1 else 1
        ts_times = [
            str(np.datetime_as_string(t, unit="s")).replace("T", " ") if hasattr(t, "item") else str(t)
            for t in times
        ]

        max_div_vals: list[float | None] = []
        del_vals: list[float | None] = []

        if max_col > 0 and max_col <= n_cols:
            col_data = values[:, max_col - 1] if values.ndim > 1 else values
            max_div_vals = [round(float(v) * max_frac, 4) for v in col_data]
        if del_col > 0 and del_col <= n_cols:
            col_data = values[:, del_col - 1] if values.ndim > 1 else values
            del_vals = [round(float(v) * del_frac, 4) for v in col_data]

        if max_div_vals or del_vals:
            timeseries = {
                "times": ts_times,
                "max_diversion": max_div_vals if max_div_vals else None,
                "delivery": del_vals if del_vals else None,
            }

    return {
        "id": div_id,
        "name": div.name or f"Diversion {div_id}",
        "source_node": div.source_node,
        "source": {"lng": src_lng, "lat": src_lat} if src_lng is not None else None,
        "destination_type": div.destination_type,
        "destination_id": div.destination_id,
        "destination": {"lng": dst_lng, "lat": dst_lat} if dst_lng is not None else None,
        "max_rate": div.max_rate if div.max_rate < 1e30 else None,
        "priority": div.priority,
        "delivery": {
            "dest_type": dest_type,
            "dest_id": delivery_dest_id,
            "element_ids": element_ids,
            "element_polygons": delivery_geojson,
        },
        "timeseries": timeseries,
    }


@router.get("/reach-profile")
def get_reach_profile(
    reach_id: int = Query(..., description="Reach ID"),
) -> dict:
    """
    Get longitudinal profile data for a stream reach.

    Returns per-node distance from upstream, bed elevation, Manning's n,
    and cross-section parameters along the reach.
    """
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    model = model_state.model
    if not model.has_streams or model.streams is None:
        raise HTTPException(status_code=404, detail="No stream data in model")

    stream = model.streams
    grid = model.grid
    strat = model.stratigraphy

    lookup = _get_gs_elev_lookup(grid, strat)

    # Find the reach — try populated reaches first, then build from nodes
    reach_stream_nodes: list = []
    reach_name = f"Reach {reach_id}"

    if hasattr(stream, "reaches") and stream.reaches:
        reaches = stream.reaches
        if isinstance(reaches, dict):
            target_reach = reaches.get(reach_id)
        else:
            target_reach = next((r for r in reaches if r.id == reach_id), None)
        if target_reach is not None:
            reach_stream_nodes = (
                getattr(target_reach, "stream_nodes", None)
                or getattr(target_reach, "nodes", [])
            )
            reach_name = getattr(target_reach, "name", "") or reach_name

    # Resolve int node IDs to StrmNode objects
    if reach_stream_nodes and isinstance(reach_stream_nodes[0], int):
        resolved = []
        for nid in reach_stream_nodes:
            sn_obj = stream.nodes.get(nid) if hasattr(stream, "nodes") else None
            if sn_obj is not None:
                resolved.append(sn_obj)
        reach_stream_nodes = resolved

    if not reach_stream_nodes and hasattr(stream, "nodes") and stream.nodes:
        # Fallback: collect stream nodes with matching reach_id
        reach_stream_nodes = sorted(
            (sn for sn in stream.nodes.values() if getattr(sn, "reach_id", 0) == reach_id),
            key=lambda sn: sn.id,
        )

    if not reach_stream_nodes:
        raise HTTPException(
            status_code=404, detail=f"Reach {reach_id} not found"
        )

    nodes: list[dict] = []
    cumulative_dist = 0.0
    prev_x, prev_y = None, None

    for sn in reach_stream_nodes:
        gw_node = getattr(sn, "groundwater_node", None) or getattr(sn, "gw_node", None)
        node = grid.nodes.get(gw_node) if gw_node else None

        x = node.x if node else 0.0
        y = node.y if node else 0.0
        lng, lat = (
            model_state.reproject_coords(x, y) if node else (0.0, 0.0)
        )

        # Ground surface elevation
        gs_elev = _node_z(gw_node, strat, lookup) if gw_node else 0.0

        # Bed elevation from stream node or cross-section
        bed_elev = getattr(sn, "bottom_elev", 0.0) or 0.0
        if hasattr(sn, "cross_section") and sn.cross_section:
            bed_elev = sn.cross_section.bottom_elev

        # Manning's n
        mannings_n = 0.04
        if hasattr(sn, "cross_section") and sn.cross_section:
            mannings_n = sn.cross_section.n

        # Conductivity and bed thickness
        conductivity = getattr(sn, "conductivity", 0.0) or 0.0
        bed_thickness = getattr(sn, "bed_thickness", 0.0) or 0.0

        # Cumulative distance from upstream
        if prev_x is not None:
            dx = x - prev_x
            dy = y - prev_y
            cumulative_dist += math.sqrt(dx * dx + dy * dy)
        prev_x, prev_y = x, y

        nodes.append({
            "stream_node_id": sn.id,
            "gw_node_id": gw_node,
            "lng": lng,
            "lat": lat,
            "distance": round(cumulative_dist, 1),
            "ground_surface_elev": round(gs_elev, 2),
            "bed_elev": round(bed_elev, 2),
            "mannings_n": round(mannings_n, 4),
            "conductivity": round(conductivity, 4),
            "bed_thickness": round(bed_thickness, 2),
        })

    return {
        "reach_id": reach_id,
        "name": reach_name,
        "n_nodes": len(nodes),
        "total_length": round(cumulative_dist, 1),
        "nodes": nodes,
    }
