"""
Groundwater data API routes.
"""

from __future__ import annotations

import math

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from pyiwfm.visualization.webapi.config import model_state

router = APIRouter(prefix="/api/groundwater", tags=["groundwater"])


def _safe_float(val: float, default: float = 0.0) -> float:
    """Clamp inf/nan to a JSON-safe value."""
    if val is None or math.isnan(val) or math.isinf(val):
        return default
    return float(val)


class WellInfo(BaseModel):
    """Well information for map display."""

    id: int
    lng: float
    lat: float
    name: str
    element: int
    pump_rate: float
    max_pump_rate: float
    top_screen: float
    bottom_screen: float
    layers: list[int]


class WellsResponse(BaseModel):
    """Response for well locations endpoint."""

    n_wells: int
    wells: list[WellInfo]


@router.get("/wells", response_model=WellsResponse)
def get_wells() -> WellsResponse:
    """
    Get all pumping wells with WGS84 coordinates.

    Returns well locations, pump rates, and screen information
    for display on the 2D Results Map.
    """
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    model = model_state.model
    if model.groundwater is None:
        return WellsResponse(n_wells=0, wells=[])

    wells: list[WellInfo] = []
    for well in model.groundwater.iter_wells():
        lng, lat = model_state.reproject_coords(well.x, well.y)
        wells.append(
            WellInfo(
                id=well.id,
                lng=lng,
                lat=lat,
                name=well.name or f"Well {well.id}",
                element=well.element,
                pump_rate=_safe_float(well.pump_rate),
                max_pump_rate=_safe_float(well.max_pump_rate),
                top_screen=_safe_float(well.top_screen),
                bottom_screen=_safe_float(well.bottom_screen),
                layers=well.layers,
            )
        )

    return WellsResponse(n_wells=len(wells), wells=wells)


class BCNodeInfo(BaseModel):
    """Boundary condition node for map display."""

    bc_id: int
    node_id: int
    lng: float
    lat: float
    bc_type: str
    value: float
    layer: int
    conductance: float | None = None


class BCResponse(BaseModel):
    """Response for boundary conditions endpoint."""

    n_conditions: int
    nodes: list[BCNodeInfo]


@router.get("/boundary-conditions", response_model=BCResponse)
def get_boundary_conditions() -> BCResponse:
    """
    Get all groundwater boundary conditions with WGS84 coordinates.

    Returns BC nodes with type, value, and layer for display on
    the 2D Results Map. Each node in each BC is returned separately.
    """
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    model = model_state.model
    if model.groundwater is None:
        return BCResponse(n_conditions=0, nodes=[])

    grid = model.grid
    nodes: list[BCNodeInfo] = []

    for bc in model.groundwater.boundary_conditions:
        for i, nid in enumerate(bc.nodes):
            node = grid.nodes.get(nid)
            if node is None:
                continue
            lng, lat = model_state.reproject_coords(node.x, node.y)
            raw_cond = bc.conductance[i] if bc.conductance and i < len(bc.conductance) else None
            cond = _safe_float(raw_cond) if raw_cond is not None else None
            raw_val = bc.values[i] if i < len(bc.values) else 0.0
            nodes.append(
                BCNodeInfo(
                    bc_id=bc.id,
                    node_id=nid,
                    lng=lng,
                    lat=lat,
                    bc_type=bc.bc_type,
                    value=_safe_float(raw_val),
                    layer=bc.layer,
                    conductance=cond,
                )
            )

    return BCResponse(n_conditions=len(nodes), nodes=nodes)


class SubsidenceLocationInfo(BaseModel):
    """Subsidence hydrograph observation location."""

    id: int
    lng: float
    lat: float
    layer: int
    name: str


class SubsidenceLocationsResponse(BaseModel):
    """Response for subsidence hydrograph locations."""

    n_locations: int
    locations: list[SubsidenceLocationInfo]


@router.get("/subsidence-locations", response_model=SubsidenceLocationsResponse)
def get_subsidence_locations() -> SubsidenceLocationsResponse:
    """
    Get subsidence hydrograph observation locations with WGS84 coordinates.

    Returns InSAR, extensometer, and other subsidence observation points
    parsed from the subsidence file's NOUTS section.
    """
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    model = model_state.model
    if model.groundwater is None:
        return SubsidenceLocationsResponse(n_locations=0, locations=[])

    subs_config = getattr(model.groundwater, "subsidence_config", None)
    if subs_config is None:
        return SubsidenceLocationsResponse(n_locations=0, locations=[])

    specs = getattr(subs_config, "hydrograph_specs", [])
    locations: list[SubsidenceLocationInfo] = []

    for spec in specs:
        try:
            lng, lat = model_state.reproject_coords(spec.x, spec.y)
        except Exception:
            continue
        locations.append(
            SubsidenceLocationInfo(
                id=spec.id,
                lng=lng,
                lat=lat,
                layer=spec.layer,
                name=spec.name or f"Subsidence Obs {spec.id}",
            )
        )

    return SubsidenceLocationsResponse(
        n_locations=len(locations), locations=locations
    )


def _well_function(u: float) -> float:
    """
    Theis well function W(u) using series approximation.

    For u < 1: series expansion.
    For u >= 1: continued fraction approximation.
    """
    if u <= 0:
        return 0.0
    if u < 1:
        # Series: W(u) = -ln(u) - gamma + u - u^2/(2*2!) + ...
        gamma = 0.5772156649
        w = -math.log(u) - gamma
        term = u
        for n in range(1, 20):
            w += ((-1) ** (n + 1)) * term / n
            term *= u / (n + 1)
        return max(w, 0.0)
    else:
        # Asymptotic: W(u) ≈ e^-u * (1/u - 1/u^2 + 2/u^3 - ...)
        # Use rational approximation
        eu = math.exp(-u)
        return eu * (1.0 / u) * (u + 0.2677737343) / (
            u * u + 1.0354508440 * u + 0.2360599665
        )


@router.get("/well-impact")
def get_well_impact(
    well_id: int = Query(..., description="Well ID"),
    time: float = Query(
        default=365.0, ge=0, description="Time since pumping started (days)"
    ),
    n_rings: int = Query(
        default=10, ge=3, le=50, description="Number of contour rings"
    ),
    max_radius: float = Query(
        default=0, ge=0,
        description="Maximum radius (model units). 0 = auto-compute.",
    ),
) -> dict:
    """
    Approximate cone of depression using the Theis analytical solution.

    Returns concentric contour rings of drawdown around a pumping well.
    Uses aquifer properties (Kh, Sy) from the model to estimate
    transmissivity and storativity.
    """
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    model = model_state.model
    if model.groundwater is None:
        raise HTTPException(status_code=404, detail="No groundwater data")

    # Find the well
    target_well = None
    for well in model.groundwater.iter_wells():
        if well.id == well_id:
            target_well = well
            break

    if target_well is None:
        raise HTTPException(
            status_code=404, detail=f"Well {well_id} not found"
        )

    pump_rate = abs(target_well.pump_rate)
    if pump_rate < 1e-10:
        return {
            "well_id": well_id,
            "name": target_well.name or f"Well {well_id}",
            "center": model_state.reproject_coords(
                target_well.x, target_well.y
            ),
            "contours": [],
            "message": "Well has zero pumping rate",
        }

    # Get aquifer properties at the well's element
    elem_id = target_well.element
    kh = 10.0  # default Kh (ft/day)
    sy = 0.1  # default specific yield

    if hasattr(model.groundwater, "aquifer_params"):
        params = model.groundwater.aquifer_params
        if hasattr(params, "get_element_params"):
            try:
                ep = params.get_element_params(elem_id, layer=1)
                kh = ep.get("kh", kh)
                sy = ep.get("sy", sy)
            except (KeyError, IndexError):
                pass

    # Compute transmissivity: T = Kh * thickness
    thickness = target_well.screen_length
    if thickness <= 0:
        thickness = 100.0  # Default thickness

    T = kh * thickness  # ft^2/day
    S = sy  # storativity

    if T <= 0 or S <= 0:
        raise HTTPException(
            status_code=400,
            detail="Cannot compute: T or S <= 0",
        )

    # Convert pump rate from model units to ft^3/day
    # Pump rate is typically in acre-ft/month or similar
    Q = pump_rate  # Use as-is; units will scale proportionally

    time_days = time

    # Determine maximum radius (where drawdown < 0.01 ft)
    if max_radius <= 0:
        # Solve for r where W(u) = 0.01 * 4*pi*T / Q
        # Use bisection or estimate
        max_radius = math.sqrt(4 * T * time_days / S) * 3

    # Generate contour rings
    contours: list[dict] = []
    well_lng, well_lat = model_state.reproject_coords(
        target_well.x, target_well.y
    )

    for i in range(1, n_rings + 1):
        r = (i / n_rings) * max_radius
        if r <= 0:
            continue

        u = (r * r * S) / (4 * T * time_days)
        Wu = _well_function(u)
        drawdown = (Q / (4 * math.pi * T)) * Wu

        if drawdown < 0.001:
            continue

        # Convert radius to approximate degrees for map display
        # 1 degree ≈ 364,000 ft at mid-latitudes
        r_deg = r / 364000.0

        contours.append({
            "radius_ft": round(r, 0),
            "radius_deg": round(r_deg, 6),
            "drawdown_ft": round(drawdown, 3),
            "u": round(u, 6),
        })

    return {
        "well_id": well_id,
        "name": target_well.name or f"Well {well_id}",
        "center": {"lng": well_lng, "lat": well_lat},
        "pump_rate": pump_rate,
        "transmissivity": round(T, 2),
        "storativity": round(S, 4),
        "time_days": time_days,
        "n_contours": len(contours),
        "contours": contours,
    }
