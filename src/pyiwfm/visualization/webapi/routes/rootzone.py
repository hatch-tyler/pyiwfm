"""
Root zone / land use data API routes.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Path, Query

from pyiwfm.visualization.webapi.config import model_state, require_model

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rootzone", tags=["rootzone"])

# Module-level flag so we only attempt lazy land-use loading once.
_land_use_loaded = False


def _ensure_land_use_loaded() -> None:
    """Lazy-load land use data on first request if element_landuse is empty.

    Prefers the HDF5-backed AreaDataManager for fast access.
    Falls back to the text-based ``load_land_use_snapshot()`` method.
    """
    global _land_use_loaded
    if _land_use_loaded:
        return

    model = model_state.model
    if model is None or model.rootzone is None:
        _land_use_loaded = True
        return

    rz = model.rootzone
    if rz.element_landuse:
        _land_use_loaded = True
        return  # already populated

    # Log which area files were wired
    has_any_file = False
    for label, af in [
        ("nonponded", getattr(rz, "nonponded_area_file", None)),
        ("ponded", getattr(rz, "ponded_area_file", None)),
        ("urban", getattr(rz, "urban_area_file", None)),
        ("native", getattr(rz, "native_area_file", None)),
    ]:
        if af is not None:
            exists = af.exists()
            logger.info("  %s area file: %s (exists=%s)", label, af, exists)
            if exists:
                has_any_file = True
        else:
            logger.info("  %s area file: NOT WIRED", label)

    if not has_any_file:
        logger.warning(
            "No area files wired or found on disk. "
            "Land use data will not be available. "
            "Check that root zone sub-files are parsed correctly."
        )
        _land_use_loaded = True
        return

    # Try HDF5 area manager first (triggers conversion if needed)
    mgr = model_state.get_area_manager()
    if mgr is not None and mgr.n_timesteps > 0:
        logger.info(
            "Area data available via HDF5 manager (%d timesteps)",
            mgr.n_timesteps,
        )
        # Populate element_landuse from first timestep for backward compat
        try:
            snapshot = mgr.get_snapshot(0)
            rz.load_land_use_from_arrays(snapshot)
            logger.info(
                "Loaded %d land use entries from HDF5 area cache",
                len(rz.element_landuse),
            )
            _land_use_loaded = True
            return
        except Exception as exc:
            logger.warning(
                "HDF5 area snapshot failed, falling back to text: %s",
                exc,
                exc_info=True,
            )

    # Fallback: text-based loading
    try:
        logger.info("Attempting text-based land use loading...")
        rz.load_land_use_snapshot(timestep=0)
        n_loaded = len(rz.element_landuse)
        if n_loaded > 0:
            logger.info("Loaded %d land use entries from area files", n_loaded)
            _land_use_loaded = True
        else:
            logger.warning(
                "No land use data loaded from text files! "
                "Area files exist but produced no data rows."
            )
    except Exception as exc:
        logger.warning(
            "Could not load land use from area files: %s",
            exc,
            exc_info=True,
        )


@router.get("/status")
def get_rootzone_status() -> dict:
    """Diagnostic endpoint showing land-use loading status."""
    model = require_model()
    if model.rootzone is None:
        return {"loaded": False, "reason": "No rootzone component in model"}

    rz = model.rootzone

    area_files: dict[str, dict] = {}
    for label, attr in [
        ("nonponded", "nonponded_area_file"),
        ("ponded", "ponded_area_file"),
        ("urban", "urban_area_file"),
        ("native", "native_area_file"),
    ]:
        af = getattr(rz, attr, None)
        if af is not None:
            area_files[label] = {
                "path": str(af),
                "exists": af.exists(),
                "size_bytes": af.stat().st_size if af.exists() else None,
            }
        else:
            area_files[label] = {"path": None, "exists": False}

    configs: dict[str, bool] = {
        "nonponded_config": rz.nonponded_config is not None,
        "ponded_config": rz.ponded_config is not None,
        "urban_config": rz.urban_config is not None,
        "native_riparian_config": rz.native_riparian_config is not None,
    }

    mgr = model_state.get_area_manager()
    mgr_info = None
    if mgr is not None:
        mgr_info = {
            "n_timesteps": mgr.n_timesteps,
            "loaders": {
                lbl: {
                    "n_frames": loader.n_frames,
                    "n_elements": loader.n_elements,
                    "n_cols": loader.n_cols,
                }
                for lbl, loader in mgr._loaders()
            },
        }

    return {
        "loaded": True,
        "land_use_loaded": _land_use_loaded,
        "n_element_landuse": len(rz.element_landuse),
        "n_crop_types": len(rz.crop_types),
        "area_files": area_files,
        "configs_loaded": configs,
        "area_manager": mgr_info,
        "rootzone_version": model.metadata.get("rootzone_version"),
    }


@router.get("/land-use")
def get_land_use(
    timestep: int = Query(0, ge=0, description="Zero-based timestep index"),
) -> dict:
    """
    Get per-element land use fractions and dominant type.

    Uses the HDF5 area manager for fast O(1) timestep access when available,
    falling back to text reader.
    """
    model = require_model()
    if model.rootzone is None:
        raise HTTPException(status_code=404, detail="No root zone data in model")

    rz = model.rootzone

    # Lazy-load on first request
    _ensure_land_use_loaded()

    # Try HDF5 manager for fast access
    mgr = model_state.get_area_manager()
    if mgr is not None and mgr.n_timesteps > 0:
        if timestep >= mgr.n_timesteps:
            timestep = mgr.n_timesteps - 1
        snapshot = mgr.get_snapshot(timestep)
        elements: list[dict] = []
        for eid, data in snapshot.items():
            elements.append(
                {
                    "element_id": eid,
                    "fractions": data["fractions"],
                    "dominant": data["dominant"],
                    "total_area": data["total_area"],
                }
            )
        return {"n_elements": len(elements), "elements": elements}

    # Fallback: text-based loading
    if timestep > 0 and (
        rz.nonponded_area_file or rz.ponded_area_file or rz.urban_area_file or rz.native_area_file
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
    elements = []
    for eid, areas in elem_data.items():
        total = sum(areas.values())
        fractions = {k: round(v / total, 4) if total > 0 else 0.0 for k, v in areas.items()}
        dominant = max(areas, key=lambda k: areas[k]) if total > 0 else "unknown"
        elements.append(
            {
                "element_id": eid,
                "fractions": fractions,
                "dominant": dominant,
                "total_area": round(total, 2),
            }
        )

    return {
        "n_elements": len(elements),
        "elements": elements,
    }


@router.get("/timesteps")
def get_land_use_timesteps() -> dict:
    """
    Get available timestep dates for land-use area data.
    """
    model = require_model()
    if model.rootzone is None:
        raise HTTPException(status_code=404, detail="No root zone data in model")

    mgr = model_state.get_area_manager()
    if mgr is None or mgr.n_timesteps == 0:
        return {"n_timesteps": 0, "dates": []}

    dates = mgr.get_dates()
    return {"n_timesteps": len(dates), "dates": dates}


@router.get("/land-use/{element_id}/timeseries")
def get_element_land_use_timeseries(
    element_id: int = Path(description="Element ID"),
) -> dict:
    """
    Get full timeseries of land-use areas for a single element.

    Returns per-category area arrays across all timesteps, suitable
    for plotting as a stacked area chart.
    """
    model = require_model()
    if model.rootzone is None:
        raise HTTPException(status_code=404, detail="No root zone data in model")

    mgr = model_state.get_area_manager()
    if mgr is None or mgr.n_timesteps == 0:
        raise HTTPException(
            status_code=404,
            detail="No area data available for timeseries",
        )

    result = mgr.get_element_timeseries(element_id)
    if len(result) <= 2:  # only element_id + dates, no actual data
        raise HTTPException(
            status_code=404,
            detail=f"No area data for element {element_id}",
        )

    return result


@router.get("/land-use/{element_id}/crops")
def get_element_crops(
    element_id: int = Path(description="Element ID"),
) -> dict:
    """
    Get crop breakdown for a specific element.

    Returns agricultural crop fractions with crop names, plus urban
    impervious fraction if applicable.
    """
    model = require_model()
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
                crops.append(
                    {
                        "crop_id": crop_id,
                        "name": ct.name if ct else f"Crop {crop_id}",
                        "fraction": round(frac, 4),
                        "area": round(elu.area * frac, 2),
                    }
                )
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
    model = require_model()
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
    model = require_model()
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
