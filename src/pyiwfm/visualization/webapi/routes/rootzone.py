"""
Root zone / land use data API routes.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Path, Query

from pyiwfm.visualization.webapi.config import model_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rootzone", tags=["rootzone"])

# Module-level flag so we only attempt lazy land-use loading once.
_land_use_loaded = False


def _ensure_land_use_loaded() -> None:
    """Lazy-load land use data on first request if element_landuse is empty."""
    global _land_use_loaded
    if _land_use_loaded:
        return
    _land_use_loaded = True

    model = model_state.model
    if model is None or model.rootzone is None:
        return

    rz = model.rootzone
    if rz.element_landuse:
        return  # already populated

    # Try loading from area files
    try:
        rz.load_land_use_snapshot(timestep=0)
        logger.info(
            "Loaded %d land use entries from area files",
            len(rz.element_landuse),
        )
    except Exception as exc:
        logger.warning("Could not load land use from area files: %s", exc)


@router.get("/land-use")
def get_land_use(
    timestep: int = Query(0, ge=0, description="Zero-based timestep index"),
) -> dict:
    """
    Get per-element land use fractions and dominant type.

    Returns a dict with element_id keys mapping to fractions of each
    land use type (agricultural, urban, native_riparian, water) and the
    dominant type for choropleth coloring.
    """
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    model = model_state.model
    if model.rootzone is None:
        raise HTTPException(status_code=404, detail="No root zone data in model")

    rz = model.rootzone

    # Lazy-load on first request
    _ensure_land_use_loaded()

    # If a non-zero timestep was requested, reload
    if timestep > 0 and (
        rz.nonponded_area_file
        or rz.ponded_area_file
        or rz.urban_area_file
        or rz.native_area_file
    ):
        try:
            rz.load_land_use_snapshot(timestep=timestep)
        except Exception as exc:
            logger.warning("Could not load timestep %d: %s", timestep, exc)

    # Aggregate land use area by element
    elem_data: dict[int, dict[str, float]] = {}

    for elu in rz.element_landuse:
        eid = elu.element_id
        if eid not in elem_data:
            elem_data[eid] = {
                "agricultural": 0.0,
                "urban": 0.0,
                "native_riparian": 0.0,
                "water": 0.0,
            }
        elem_data[eid][elu.land_use_type.value] += elu.area

    # Build response with fractions and dominant type
    elements: list[dict] = []
    for eid, areas in elem_data.items():
        total = sum(areas.values())
        fractions = {
            k: round(v / total, 4) if total > 0 else 0.0
            for k, v in areas.items()
        }
        dominant = max(areas, key=areas.get) if total > 0 else "unknown"
        elements.append({
            "element_id": eid,
            "fractions": fractions,
            "dominant": dominant,
            "total_area": round(total, 2),
        })

    return {
        "n_elements": len(elements),
        "elements": elements,
    }


@router.get("/land-use/{element_id}/crops")
def get_element_crops(
    element_id: int = Path(description="Element ID"),
) -> dict:
    """
    Get crop breakdown for a specific element.

    Returns agricultural crop fractions with crop names, plus urban
    impervious fraction if applicable.
    """
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    model = model_state.model
    if model.rootzone is None:
        raise HTTPException(status_code=404, detail="No root zone data in model")

    rz = model.rootzone
    _ensure_land_use_loaded()

    land_uses = rz.get_landuse_for_element(element_id)
    if not land_uses:
        raise HTTPException(
            status_code=404,
            detail=f"No land use data for element {element_id}",
        )

    crops: list[dict] = []
    urban_impervious = 0.0

    for elu in land_uses:
        if elu.land_use_type.value == "agricultural":
            for crop_id, frac in elu.crop_fractions.items():
                ct = rz.crop_types.get(crop_id)
                crops.append({
                    "crop_id": crop_id,
                    "name": ct.name if ct else f"Crop {crop_id}",
                    "fraction": round(frac, 4),
                    "area": round(elu.area * frac, 2),
                })
        elif elu.land_use_type.value == "urban":
            urban_impervious = elu.impervious_fraction

    # Soil parameters if available
    soil: dict | None = None
    sp = rz.soil_params.get(element_id)
    if sp:
        soil = {
            "porosity": sp.porosity,
            "field_capacity": sp.field_capacity,
            "wilting_point": sp.wilting_point,
            "saturated_kv": sp.saturated_kv,
            "available_water": sp.available_water,
        }

    return {
        "element_id": element_id,
        "crops": crops,
        "urban_impervious_fraction": round(urban_impervious, 4),
        "soil_parameters": soil,
    }


@router.get("/crops")
def get_crops() -> dict:
    """
    List all crop types with names, IDs, and root depths.
    """
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    model = model_state.model
    if model.rootzone is None:
        raise HTTPException(status_code=404, detail="No root zone data in model")

    rz = model.rootzone
    crops = [
        {
            "id": ct.id,
            "name": ct.name,
            "root_depth": ct.root_depth,
            "kc": ct.kc,
        }
        for ct in sorted(rz.crop_types.values(), key=lambda c: c.id)
    ]

    return {
        "n_crops": len(crops),
        "crops": crops,
    }


@router.get("/soil-params/{element_id}")
def get_soil_params(
    element_id: int = Path(description="Element ID"),
) -> dict:
    """
    Get full soil parameters for an element.
    """
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    model = model_state.model
    if model.rootzone is None:
        raise HTTPException(status_code=404, detail="No root zone data in model")

    sp = model.rootzone.soil_params.get(element_id)
    if sp is None:
        raise HTTPException(
            status_code=404,
            detail=f"No soil parameters for element {element_id}",
        )

    return {
        "element_id": element_id,
        "porosity": sp.porosity,
        "field_capacity": sp.field_capacity,
        "wilting_point": sp.wilting_point,
        "saturated_kv": sp.saturated_kv,
        "lambda_param": sp.lambda_param,
        "kunsat_method": sp.kunsat_method,
        "k_ponded": sp.k_ponded,
        "capillary_rise": sp.capillary_rise,
        "available_water": sp.available_water,
        "drainable_porosity": sp.drainable_porosity,
        "precip_column": sp.precip_column,
        "precip_factor": sp.precip_factor,
    }
