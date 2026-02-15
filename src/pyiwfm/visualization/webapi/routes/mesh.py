"""
Mesh data API routes.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel

from pyiwfm.visualization.webapi.config import model_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/mesh", tags=["mesh"])


class SurfaceMeshData(BaseModel):
    """Flat surface mesh data for client-side rendering.

    The server extracts the outer surface of the 3D volumetric mesh,
    returning flat 1D arrays that the client can use directly as
    typed arrays for vtk.js PolyData.
    """

    n_points: int  # Number of points
    n_cells: int  # Number of surface polygons
    n_layers: int  # Max layer number
    points: list[float]  # Flat [x0, y0, z0, x1, y1, z1, ...]
    polys: list[int]  # Flat VTK cell array [nV, v0, v1, ..., nV, v0, v1, ...]
    layer: list[int]  # Layer number per surface polygon


@router.get("")
def get_mesh() -> Response:
    """
    Get the full 3D mesh as VTU format.

    Returns the model mesh as a VTK XML UnstructuredGrid (.vtu) file.
    This includes all layers with cell data for layer numbers.
    """
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    vtu_data = model_state.get_mesh_3d()

    return Response(
        content=vtu_data,
        media_type="application/xml",
        headers={"Content-Disposition": "attachment; filename=mesh.vtu"},
    )


@router.get("/surface")
def get_surface_mesh() -> Response:
    """
    Get the surface mesh as VTU format.

    Returns only the 2D surface mesh (top view) without stratigraphy.
    This is faster to load for initial display.
    """
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    vtu_data = model_state.get_mesh_surface()

    return Response(
        content=vtu_data,
        media_type="application/xml",
        headers={"Content-Disposition": "attachment; filename=surface.vtu"},
    )


@router.get("/json", response_model=SurfaceMeshData)
def get_mesh_json(
    layer: int = Query(default=0, ge=0, description="Layer number (0=all)"),
) -> SurfaceMeshData:
    """
    Get the 3D mesh surface as JSON for vtk.js rendering.

    The server extracts the outer surface of the 3D volumetric mesh
    using VTK's extract_surface(), returning flat 1D arrays that the
    client can load directly into vtk.js PolyData with zero
    post-processing.

    Parameters
    ----------
    layer : int
        0 = all layers (default), 1..N = specific layer only.

    Returns flat arrays:
    - points: [x0, y0, z0, x1, y1, z1, ...] (Float32-compatible)
    - polys: [nV, v0, v1, ..., nV, v0, v1, ...] (VTK cell array format)
    - layer: layer number per surface polygon
    """
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    if layer > 0 and model_state.model and model_state.model.stratigraphy:
        n_layers = model_state.model.stratigraphy.n_layers
        if layer > n_layers:
            raise HTTPException(
                status_code=400,
                detail=f"Layer {layer} exceeds model layers ({n_layers})",
            )

    try:
        data = model_state.get_surface_json(layer=layer)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return SurfaceMeshData(**data)


@router.get("/geojson")
def get_mesh_geojson(
    layer: int = Query(default=1, ge=1, description="Layer number (1-based)"),
) -> dict:
    """
    Get the mesh as GeoJSON FeatureCollection with WGS84 coordinates.

    Returns polygon features for each element, suitable for deck.gl
    GeoJsonLayer rendering. Coordinates are reprojected from the model
    CRS to WGS84 (EPSG:4326) server-side.
    """
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    return model_state.get_mesh_geojson(layer=layer)


@router.get("/head-map")
def get_head_map(
    timestep: int = Query(default=0, ge=0, description="Timestep index"),
    layer: int = Query(default=1, ge=1, description="Layer number (1-based)"),
) -> dict:
    """
    Get head values mapped to mesh elements as GeoJSON.

    Returns a GeoJSON FeatureCollection where each element polygon
    includes a 'head' property with the average head value at that
    element's nodes for the specified timestep and layer.
    """
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    from pyiwfm.visualization.webapi.config import model_state as ms

    loader = ms.get_head_loader()
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

    head_values = frame[:, layer_idx]

    model = ms.model
    grid = model.grid

    # Build node_id -> index mapping
    sorted_node_ids = sorted(grid.nodes.keys())
    node_id_to_idx = {nid: i for i, nid in enumerate(sorted_node_ids)}

    # Get the base GeoJSON and add head values
    geojson = ms.get_mesh_geojson(layer=layer)

    features = []
    for feat in geojson["features"]:
        elem_id = feat["properties"]["element_id"]
        elem = grid.elements.get(elem_id)
        if elem is None:
            continue

        # Average head at element vertices
        node_heads = []
        for nid in elem.vertices:
            idx = node_id_to_idx.get(nid)
            if idx is not None and idx < len(head_values):
                node_heads.append(float(head_values[idx]))

        avg_head = sum(node_heads) / len(node_heads) if node_heads else 0.0

        new_feat = dict(feat)
        new_feat["properties"] = {
            **feat["properties"],
            "head": round(avg_head, 2),
        }
        features.append(new_feat)

    dt = loader.times[timestep] if timestep < len(loader.times) else None

    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "timestep_index": timestep,
            "datetime": dt.isoformat() if dt else None,
            "layer": layer,
        },
    }


@router.get("/subregions")
def get_subregions() -> dict:
    """
    Get subregion boundary polygons as GeoJSON in WGS84.

    Computes the exterior boundary of each subregion by finding
    element edges that are only shared by elements within the
    same subregion, then assembling them into closed rings.
    """
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    model = model_state.model
    grid = model.grid

    # Group elements by subregion (skip subregion 0 = unassigned)
    elements_by_sub: dict[int, list[int]] = {}
    for elem in grid.iter_elements():
        sub_id = elem.subregion
        if sub_id <= 0:
            continue
        if sub_id not in elements_by_sub:
            elements_by_sub[sub_id] = []
        elements_by_sub[sub_id].append(elem.id)

    if not elements_by_sub:
        return {"type": "FeatureCollection", "features": []}

    features: list[dict] = []

    for sub_id, elem_ids in elements_by_sub.items():
        sub_info = grid.subregions.get(sub_id)
        sub_name = sub_info.name if sub_info else f"Subregion {sub_id}"

        # Find boundary edges: edges where only one side is in this subregion
        edge_count: dict[tuple[int, int], int] = {}
        for eid in elem_ids:
            elem = grid.elements[eid]
            verts = list(elem.vertices)
            n = len(verts)
            for i in range(n):
                a, b = verts[i], verts[(i + 1) % n]
                edge = (min(a, b), max(a, b))
                edge_count[edge] = edge_count.get(edge, 0) + 1

        # Boundary edges appear exactly once within subregion
        boundary_edges: list[tuple[int, int]] = []
        for edge, count in edge_count.items():
            if count == 1:
                boundary_edges.append(edge)
            elif count == 2:
                # Internal edge — skip unless it borders an element
                # outside the subregion (shared face)
                pass

        if not boundary_edges:
            continue

        # Also check edges shared with elements outside the subregion
        # (edges that appear once in our set are definitely boundary)
        # Build adjacency for ring assembly
        adjacency: dict[int, list[int]] = {}
        for a, b in boundary_edges:
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)

        # Assemble rings by walking the boundary edges
        used_edges: set[tuple[int, int]] = set()
        rings: list[list[int]] = []

        for start_node in adjacency:
            if all(
                (min(start_node, nb), max(start_node, nb)) in used_edges
                for nb in adjacency[start_node]
            ):
                continue

            ring: list[int] = [start_node]
            current = start_node
            prev = -1

            while True:
                neighbors = adjacency.get(current, [])
                next_node = None
                for nb in neighbors:
                    edge_key = (min(current, nb), max(current, nb))
                    if edge_key not in used_edges and nb != prev:
                        next_node = nb
                        break

                if next_node is None:
                    break

                used_edges.add((min(current, next_node), max(current, next_node)))
                if next_node == start_node:
                    ring.append(next_node)
                    break
                ring.append(next_node)
                prev = current
                current = next_node

            if len(ring) >= 4:  # At least a triangle + closing
                rings.append(ring)

        if not rings:
            continue

        # Use the longest ring as the exterior boundary
        rings.sort(key=len, reverse=True)

        # Convert ring nodes to WGS84 coordinates
        for ring in rings[:1]:  # Only exterior ring for now
            coords: list[list[float]] = []
            for nid in ring:
                node = grid.nodes.get(nid)
                if node:
                    lng, lat = model_state.reproject_coords(node.x, node.y)
                    coords.append([lng, lat])

            if len(coords) < 4:
                continue

            # Compute centroid for label placement
            lng_sum = sum(c[0] for c in coords[:-1])
            lat_sum = sum(c[1] for c in coords[:-1])
            n_pts = len(coords) - 1
            centroid = [lng_sum / n_pts, lat_sum / n_pts]

            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": {
                    "subregion_id": sub_id,
                    "name": sub_name,
                    "n_elements": len(elem_ids),
                    "centroid": centroid,
                },
            })

    return {"type": "FeatureCollection", "features": features}


@router.get("/property-map")
def get_property_map(
    property: str = Query(description="Property ID (e.g., kh, sy, thickness)"),
    layer: int = Query(default=1, ge=1, description="Layer number (1-based)"),
) -> dict:
    """
    Get per-element property values mapped to GeoJSON for 2D map coloring.

    Returns a GeoJSON FeatureCollection where each element polygon
    includes a 'value' property with the property value for that element.
    """
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    import numpy as np

    from pyiwfm.visualization.webapi.properties import PROPERTY_INFO
    from pyiwfm.visualization.webapi.routes.properties import _compute_property_values

    values = _compute_property_values(property, layer)
    if values is None:
        raise HTTPException(
            status_code=404, detail=f"Property '{property}' not available"
        )

    model = model_state.model
    grid = model.grid
    n_elements = grid.n_elements

    # Extract per-element values for the requested layer
    layer_idx = layer - 1
    start = layer_idx * n_elements
    end = start + n_elements

    if end > len(values):
        raise HTTPException(
            status_code=400, detail=f"Layer {layer} out of range"
        )

    elem_values = values[start:end]

    # Get the base GeoJSON
    geojson = model_state.get_mesh_geojson(layer=layer)

    info = PROPERTY_INFO.get(property, {"name": property, "units": "", "log_scale": False})
    valid = elem_values[~np.isnan(elem_values)]
    vmin = float(np.min(valid)) if len(valid) > 0 else 0.0
    vmax = float(np.max(valid)) if len(valid) > 0 else 1.0

    features = []
    sorted_elem_ids = sorted(grid.elements.keys())
    elem_id_to_idx = {eid: i for i, eid in enumerate(sorted_elem_ids)}

    for feat in geojson["features"]:
        elem_id = feat["properties"]["element_id"]
        idx = elem_id_to_idx.get(elem_id)
        if idx is None or idx >= len(elem_values):
            continue

        val = float(elem_values[idx])
        new_feat = dict(feat)
        new_feat["properties"] = {
            **feat["properties"],
            "value": round(val, 6) if not np.isnan(val) else None,
        }
        features.append(new_feat)

    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "property": property,
            "name": info.get("name", property),
            "units": info.get("units", ""),
            "log_scale": info.get("log_scale", False),
            "layer": layer,
            "min": round(vmin, 6),
            "max": round(vmax, 6),
        },
    }


@router.get("/element/{element_id}")
def get_element_detail(element_id: int) -> dict:
    """
    Get detailed information for a single mesh element.

    Returns subregion, vertices with coordinates, area, per-layer
    aquifer properties, wells in element, and current head values.
    """
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    model = model_state.model
    grid = model.grid
    strat = model.stratigraphy

    elem = grid.elements.get(element_id)
    if elem is None:
        raise HTTPException(
            status_code=404, detail=f"Element {element_id} not found"
        )

    # Subregion info
    sub_info = grid.subregions.get(elem.subregion)
    subregion = {
        "id": elem.subregion,
        "name": sub_info.name if sub_info else f"Subregion {elem.subregion}",
    }

    # Vertex coordinates (model CRS and WGS84)
    vertices = []
    for nid in elem.vertices:
        node = grid.nodes.get(nid)
        if node:
            lng, lat = model_state.reproject_coords(node.x, node.y)
            vertices.append({
                "node_id": nid, "x": node.x, "y": node.y,
                "lng": lng, "lat": lat,
            })

    # Area
    area = elem.area

    # Build node-index lookup
    sorted_node_ids = sorted(grid.nodes.keys())
    node_id_to_idx = {nid: i for i, nid in enumerate(sorted_node_ids)}

    # Per-layer aquifer properties
    layer_properties: list[dict] = []
    n_layers = strat.n_layers if strat else 1

    for lay in range(1, n_layers + 1):
        lay_idx = lay - 1
        layer_info: dict = {"layer": lay}

        # Stratigraphy
        if strat is not None:
            node_tops = []
            node_bottoms = []
            for nid in elem.vertices:
                idx = node_id_to_idx.get(nid)
                if idx is not None:
                    node_tops.append(float(strat.top_elev[idx, lay_idx]))
                    node_bottoms.append(float(strat.bottom_elev[idx, lay_idx]))
            if node_tops:
                layer_info["top_elev"] = round(sum(node_tops) / len(node_tops), 2)
                layer_info["bottom_elev"] = round(sum(node_bottoms) / len(node_bottoms), 2)
                layer_info["thickness"] = round(
                    layer_info["top_elev"] - layer_info["bottom_elev"], 2
                )

        # Aquifer parameters
        if model.groundwater and model.groundwater.aquifer_params:
            params = model.groundwater.aquifer_params
            for param_name, attr_names in [
                ("kh", ["kh"]),
                ("kv", ["kv"]),
                ("ss", ["specific_storage", "ss"]),
                ("sy", ["specific_yield", "sy"]),
            ]:
                for attr in attr_names:
                    data = getattr(params, attr, None)
                    if data is not None:
                        # Average across element vertices
                        vals = []
                        for nid in elem.vertices:
                            idx = node_id_to_idx.get(nid)
                            if idx is not None and data.ndim == 2 and lay_idx < data.shape[1]:
                                vals.append(float(data[idx, lay_idx]))
                            elif idx is not None and data.ndim == 1:
                                vals.append(float(data[idx]))
                        if vals:
                            layer_info[param_name] = round(sum(vals) / len(vals), 6)
                        break

        layer_properties.append(layer_info)

    # Wells in this element
    wells: list[dict] = []
    if model.groundwater:
        for well in model.groundwater.iter_wells():
            if well.element == element_id:
                wells.append({
                    "id": well.id,
                    "name": well.name,
                    "pump_rate": well.pump_rate,
                    "layers": well.layers,
                })

    # Head values at element nodes (latest timestep)
    head_at_nodes: dict[int, list[float]] = {}
    loader = model_state.get_head_loader()
    if loader and loader.n_frames > 0:
        frame = loader.get_frame(loader.n_frames - 1)
        for nid in elem.vertices:
            idx = node_id_to_idx.get(nid)
            if idx is not None and idx < frame.shape[0]:
                head_at_nodes[nid] = [
                    round(float(frame[idx, lay]), 2)
                    for lay in range(frame.shape[1])
                ]

    # Land use breakdown
    land_use: dict | None = None
    rz = model.rootzone if hasattr(model, "rootzone") else None
    if rz is not None:
        from pyiwfm.visualization.webapi.routes.rootzone import _ensure_land_use_loaded

        _ensure_land_use_loaded()
        land_uses = rz.get_landuse_for_element(element_id)
        if land_uses:
            fracs = {
                "agricultural": 0.0,
                "urban": 0.0,
                "native_riparian": 0.0,
                "water": 0.0,
            }
            for elu in land_uses:
                fracs[elu.land_use_type.value] += elu.area
            total = sum(fracs.values())
            land_use = {
                "fractions": {
                    k: round(v / total, 4) if total > 0 else 0.0
                    for k, v in fracs.items()
                },
                "total_area": round(total, 2),
                "crops": [],
            }
            for elu in land_uses:
                if elu.land_use_type.value == "agricultural":
                    for crop_id, frac in elu.crop_fractions.items():
                        ct = rz.crop_types.get(crop_id)
                        land_use["crops"].append({
                            "crop_id": crop_id,
                            "name": ct.name if ct else f"Crop {crop_id}",
                            "fraction": round(frac, 4),
                            "area": round(elu.area * frac, 2),
                        })

    return {
        "element_id": element_id,
        "subregion": subregion,
        "n_vertices": len(elem.vertices),
        "vertices": vertices,
        "area": round(area, 2),
        "layer_properties": layer_properties,
        "wells": wells,
        "head_at_nodes": head_at_nodes,
        "land_use": land_use,
    }


@router.get("/nodes")
def get_mesh_nodes(
    layer: int = Query(default=1, ge=1, description="Layer number (unused, for future filtering)"),
) -> dict:
    """Get all mesh nodes as point features in WGS84.

    Returns node IDs and WGS84 coordinates for rendering as a dot layer.
    """
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    grid = model_state.model.grid

    nodes: list[dict] = []
    for n in grid.iter_nodes():
        lng, lat = model_state.reproject_coords(n.x, n.y)
        nodes.append({"id": n.id, "lng": lng, "lat": lat})

    return {"n_nodes": len(nodes), "nodes": nodes}
