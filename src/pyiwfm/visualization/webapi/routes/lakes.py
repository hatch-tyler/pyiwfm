"""
Lake data API routes.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Path

from pyiwfm.visualization.webapi.config import model_state, require_model

router = APIRouter(prefix="/api/lakes", tags=["lakes"])


@router.get("/geojson")
def get_lakes_geojson() -> dict:
    """
    Get lake polygons as GeoJSON in WGS84.

    Each lake is represented as a polygon built from the union of its
    element boundaries. Returns a FeatureCollection with lake properties.
    """
    model = require_model()
    if model.lakes is None or model.lakes.n_lakes == 0:
        return {"type": "FeatureCollection", "features": []}

    grid = model.grid
    if grid is None:
        return {"type": "FeatureCollection", "features": []}
    features: list[dict] = []

    for lake_id, lake in model.lakes.lakes.items():
        if not lake.elements:
            continue

        # Build lake polygon from element boundaries.
        # Find exterior edges of the element set.
        edge_count: dict[tuple[int, int], int] = {}
        for eid in lake.elements:
            elem = grid.elements.get(eid)
            if elem is None:
                continue
            verts = list(elem.vertices)
            n = len(verts)
            for i in range(n):
                a, b = verts[i], verts[(i + 1) % n]
                edge = (min(a, b), max(a, b))
                edge_count[edge] = edge_count.get(edge, 0) + 1

        # Boundary edges appear exactly once
        boundary_edges = [e for e, c in edge_count.items() if c == 1]
        if not boundary_edges:
            continue

        # Assemble ring by walking boundary edges
        adjacency: dict[int, list[int]] = {}
        for a, b in boundary_edges:
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)

        used: set[tuple[int, int]] = set()
        ring: list[int] = []

        start_node = next(iter(adjacency))
        ring.append(start_node)
        current = start_node
        prev = -1

        while True:
            neighbors = adjacency.get(current, [])
            next_node = None
            for nb in neighbors:
                ek = (min(current, nb), max(current, nb))
                if ek not in used and nb != prev:
                    next_node = nb
                    break
            if next_node is None:
                break
            used.add((min(current, next_node), max(current, next_node)))
            if next_node == start_node:
                ring.append(next_node)
                break
            ring.append(next_node)
            prev = current
            current = next_node

        if len(ring) < 4:
            continue

        # Convert to WGS84
        coords: list[list[float]] = []
        for nid in ring:
            node = grid.nodes.get(nid)
            if node:
                lng, lat = model_state.reproject_coords(node.x, node.y)
                coords.append([lng, lat])

        if len(coords) < 4:
            continue

        # Centroid for label
        lng_sum = sum(c[0] for c in coords[:-1])
        lat_sum = sum(c[1] for c in coords[:-1])
        n_pts = len(coords) - 1
        centroid = [lng_sum / n_pts, lat_sum / n_pts]

        # Rating curve info
        has_rating = lake.rating is not None
        n_rating_points = 0
        if has_rating and lake.rating:
            n_rating_points = len(lake.rating.elevations)

        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": {
                    "lake_id": lake_id,
                    "name": lake.name or f"Lake {lake_id}",
                    "n_elements": len(lake.elements),
                    "initial_elevation": lake.initial_elevation,
                    "max_elevation": lake.max_elevation if lake.max_elevation < 1e10 else None,
                    "has_rating": has_rating,
                    "n_rating_points": n_rating_points,
                    "centroid": centroid,
                },
            }
        )

    return {"type": "FeatureCollection", "features": features}


@router.get("/{lake_id}/rating")
def get_lake_rating(
    lake_id: int = Path(description="Lake ID"),
) -> dict:
    """
    Get elevation-area-volume rating curve for a lake.

    Returns arrays of elevations, areas, and volumes for charting.
    """
    model = require_model()
    if model.lakes is None:
        raise HTTPException(status_code=404, detail="No lake data in model")

    lake = model.lakes.lakes.get(lake_id)
    if lake is None:
        raise HTTPException(status_code=404, detail=f"Lake {lake_id} not found")

    if lake.rating is None:
        raise HTTPException(
            status_code=404,
            detail=f"No rating curve for lake {lake_id}",
        )

    return {
        "lake_id": lake_id,
        "name": lake.name or f"Lake {lake_id}",
        "elevations": lake.rating.elevations.tolist(),
        "areas": lake.rating.areas.tolist(),
        "volumes": lake.rating.volumes.tolist(),
        "n_points": len(lake.rating.elevations),
    }
