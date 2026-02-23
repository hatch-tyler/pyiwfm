"""
Zone budget (ZBudget) API routes.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from pyiwfm.visualization.webapi.config import model_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/zbudgets", tags=["zbudgets"])

# ---------------------------------------------------------------------------
# ZBudget glossary — IWFM zone budget column definitions (Ch. 6)
# ---------------------------------------------------------------------------

ZBUDGET_GLOSSARY: dict[str, dict[str, str]] = {
    "gw": {
        "Deep Percolation (+)": "Water percolating from root zone through unsaturated zone into the aquifer zone.",
        "Gain from Stream (+)": "Groundwater gained from stream-aquifer interaction (stream losing reach).",
        "Tile Drain Outflow (-)": "Water removed from the aquifer by tile drain systems.",
        "Subsurface Irrigation (+)": "Irrigation water applied directly to the subsurface.",
        "Subsidence (+/-)": "Change in aquifer storage due to compaction of fine-grained sediments.",
        "Specified Head BC (+/-)": "Flow at specified-head boundary condition nodes.",
        "General Head BC (+/-)": "Flow at general-head boundary condition nodes.",
        "Constrained General Head BC (+/-)": "Flow at constrained general-head BC nodes.",
        "Small Watershed Baseflow (+)": "Baseflow contribution from small watershed modules.",
        "Small Watershed Percolation (+)": "Deep percolation from small watershed to groundwater.",
        "Diversion Recoverable Loss (+)": "Recoverable seepage from diversion channels.",
        "Bypass Recoverable Loss (+)": "Recoverable seepage from bypass channels.",
        "Lake (+/-)": "Seepage between lakes and the aquifer.",
        "Pumping (-)": "Groundwater extraction by wells.",
        "GW Return Flow (+)": "Return flow from applied groundwater reaching the aquifer.",
        "Root Water Uptake (-)": "Water uptake by plant roots from the saturated zone.",
        "Zone Exchange Flows": "Net horizontal groundwater flow between adjacent zones. Positive = inflow, Negative = outflow.",
        "Storage (+/-)": "Change in groundwater storage volume within the zone.",
        "Discrepancy": "Numerical mass balance error. Should be small relative to other terms.",
        "Absolute Storage": "Total volume of water stored in the zone at end of timestep.",
    },
    "rootzone": {
        "AG_Precip (+)": "Precipitation on agricultural areas within the zone.",
        "AG_Runoff (-)": "Runoff from agricultural areas.",
        "AG_Prime Applied Water (+)": "Primary irrigation water applied to agricultural land.",
        "AG_Re-used Water (+)": "Recycled/reused water applied to agricultural land.",
        "AG_Net Return Flow (-)": "Net return flow from agricultural areas.",
        "AG_Deep Percolation (-)": "Water draining below the agricultural root zone.",
        "AG_ET (-)": "Evapotranspiration from agricultural crops.",
        "AG_Storage (+/-)": "Soil moisture change in agricultural root zone.",
        "URB_Precip (+)": "Precipitation on urban areas.",
        "URB_Runoff (-)": "Runoff from urban areas.",
        "URB_Prime Applied Water (+)": "Primary water supply to urban areas.",
        "URB_Re-used Water (+)": "Recycled water applied to urban areas.",
        "URB_Net Return Flow (-)": "Net return flow from urban areas.",
        "URB_Deep Percolation (-)": "Water draining below the urban root zone.",
        "URB_ET (-)": "Evapotranspiration from urban areas.",
        "URB_Storage (+/-)": "Soil moisture change in urban root zone.",
        "NRV_Precip (+)": "Precipitation on native/riparian vegetation.",
        "NRV_Runoff (-)": "Runoff from native/riparian areas.",
        "NRV_Deep Percolation (-)": "Water draining below native/riparian root zone.",
        "NRV_ET (-)": "Evapotranspiration from native/riparian vegetation.",
        "NRV_Storage (+/-)": "Soil moisture change in native/riparian root zone.",
    },
    "lwu": {
        "AG_Supply Requirement": "Total water supply requirement for agriculture in the zone.",
        "AG_Pumping": "Groundwater pumping for agricultural use.",
        "AG_Diversion": "Surface water diverted for agricultural use.",
        "AG_Shortage": "Shortfall of agricultural water supply vs demand.",
        "URB_Supply Requirement": "Total water supply requirement for urban use.",
        "URB_Pumping": "Groundwater pumping for urban use.",
        "URB_Diversion": "Surface water diverted for urban use.",
        "URB_Shortage": "Shortfall of urban water supply vs demand.",
    },
    "unsaturated": {
        "Deep Percolation (+)": "Water percolating from root zone into the unsaturated zone.",
        "Deep Percolation to GW (-)": "Water draining from unsaturated zone to the aquifer.",
        "Storage (+/-)": "Change in water stored within the unsaturated zone.",
        "Discrepancy": "Numerical mass balance error. Should be small relative to other terms.",
    },
}


def _sync_active_zones(reader: Any) -> None:
    """Register zones from the active zone definition into the reader.

    The ZBudgetReader only knows about zones stored in the HDF5 file.
    When the user creates a custom zone definition via the zone editor,
    those zone names (e.g. "Subregion 1") won't exist in the reader.
    This helper injects them so ``reader.get_zone_info(name)`` succeeds.
    """
    zone_def = model_state.get_zone_definition()
    if zone_def is None:
        return

    from pyiwfm.io.zbudget import ZoneInfo as ZBZoneInfo

    for z in zone_def.iter_zones():
        if z.name not in reader._zone_info:
            reader._zone_info[z.name] = ZBZoneInfo(
                id=z.id,
                name=z.name,
                n_elements=z.n_elements,
                element_ids=list(z.elements),
                area=z.area,
            )


def _safe_float(val: float) -> float | None:
    """Convert NaN/Inf to None for JSON-safe serialization."""
    if val is None or math.isnan(val) or math.isinf(val):
        return None
    return val


def _sanitize_values(values: list) -> list:
    """Replace NaN/Inf with None in a list of numeric values."""
    return [
        None if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else v for v in values
    ]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/types")
def get_zbudget_types() -> list[str]:
    """Get list of available zbudget types."""
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")
    return model_state.get_available_zbudgets()


@router.get("/elements")
def get_zbudget_elements() -> list[dict]:
    """Get element list with centroids in WGS84 for zone map coloring."""
    if not model_state.is_loaded or model_state.model is None:
        raise HTTPException(status_code=404, detail="No model loaded")

    grid = model_state.model.grid
    if grid is None:
        raise HTTPException(status_code=404, detail="No grid available")

    transformer = model_state._transformer
    if transformer is None:
        try:
            import pyproj

            transformer = pyproj.Transformer.from_crs(model_state._crs, "EPSG:4326", always_xy=True)
            model_state._transformer = transformer
        except ImportError:
            transformer = None

    elements: list[dict] = []
    for elem_id in sorted(grid.elements.keys()):
        elem = grid.elements[elem_id]
        # Compute centroid from vertex node IDs
        node_ids = elem.vertices
        if not node_ids:
            continue
        cx = sum(grid.nodes[nid].x for nid in node_ids) / len(node_ids)
        cy = sum(grid.nodes[nid].y for nid in node_ids) / len(node_ids)

        if transformer is not None:
            lng, lat = transformer.transform(cx, cy)
        else:
            lng, lat = cx, cy

        entry: dict[str, Any] = {
            "id": elem_id,
            "centroid": [lng, lat],
        }
        if hasattr(elem, "subregion"):
            entry["subregion"] = elem.subregion
        elements.append(entry)

    return elements


@router.get("/presets")
def get_zbudget_presets() -> list[dict]:
    """Get preset zone definitions (subregions)."""
    if not model_state.is_loaded or model_state.model is None:
        raise HTTPException(status_code=404, detail="No model loaded")

    presets: list[dict] = []

    grid = model_state.model.grid
    if grid is not None and grid.subregions:
        # Build subregions preset
        zones = []
        for sr_id in sorted(grid.subregions.keys()):
            sr = grid.subregions[sr_id]
            # Get elements in this subregion
            elem_ids = [eid for eid, e in grid.elements.items() if e.subregion == sr_id]
            zones.append(
                {
                    "id": sr_id,
                    "name": sr.name if hasattr(sr, "name") else f"Subregion {sr_id}",
                    "elements": elem_ids,
                }
            )
        presets.append({"name": "Subregions", "zones": zones})

    return presets


@router.post("/zones")
def create_zone_definition(payload: dict) -> dict:
    """Create or update zone definition from client JSON."""
    if not model_state.is_loaded or model_state.model is None:
        raise HTTPException(status_code=404, detail="No model loaded")

    zones_data = payload.get("zones", [])
    extent = payload.get("extent", "horizontal")

    if not zones_data:
        raise HTTPException(status_code=400, detail="No zones provided")

    from pyiwfm.core.zones import ZoneDefinition

    grid = model_state.model.grid
    element_areas: dict[int, float] = {}
    if grid is not None:
        element_areas = {eid: e.area for eid, e in grid.elements.items()}

    # Build element-zone pairs
    element_zone_pairs: list[tuple[int, int]] = []
    zone_names: dict[int, str] = {}

    for z in zones_data:
        zone_id = z.get("id", 0)
        zone_name = z.get("name", f"Zone {zone_id}")
        zone_elements = z.get("elements", [])
        zone_names[zone_id] = zone_name
        for eid in zone_elements:
            element_zone_pairs.append((eid, zone_id))

    zone_def = ZoneDefinition.from_element_list(
        element_zone_pairs,
        zone_names=zone_names,
        element_areas=element_areas,
        name="User Defined",
        description="Interactive zone definition from web viewer",
    )
    zone_def.extent = extent

    model_state.set_zone_definition(zone_def)

    return {
        "status": "ok",
        "n_zones": zone_def.n_zones,
        "n_elements": zone_def.n_elements,
    }


@router.get("/zones")
def get_zone_definition() -> dict | None:
    """Get the current active zone definition."""
    zone_def = model_state.get_zone_definition()
    if zone_def is None:
        return None

    zones = []
    for z in zone_def.iter_zones():
        zones.append(
            {
                "id": z.id,
                "name": z.name,
                "elements": z.elements,
                "n_elements": z.n_elements,
                "area": z.area,
            }
        )

    return {
        "zones": zones,
        "extent": zone_def.extent,
        "name": zone_def.name,
        "n_zones": zone_def.n_zones,
    }


@router.get("/{zbudget_type}/columns")
def get_zbudget_columns(zbudget_type: str) -> dict:
    """Get column headers for a zbudget type."""
    reader = model_state.get_zbudget_reader(zbudget_type)
    if reader is None:
        raise HTTPException(
            status_code=404,
            detail=f"ZBudget type '{zbudget_type}' not available",
        )

    return {
        "columns": [
            {"id": i, "name": name, "units": "Volume"} for i, name in enumerate(reader.data_names)
        ],
    }


@router.get("/{zbudget_type}/data")
def get_zbudget_data(
    zbudget_type: str,
    zone: str = Query(default="", description="Zone name or index"),
    columns: str = Query(default="all", description="Column indices or 'all'"),
    volume_factor: float = Query(default=1.0, description="Volume conversion factor"),
) -> dict:
    """Get zone-aggregated ZBudget data in BudgetData JSON shape."""
    reader = model_state.get_zbudget_reader(zbudget_type)
    if reader is None:
        raise HTTPException(
            status_code=404,
            detail=f"ZBudget type '{zbudget_type}' not available",
        )

    # Sync user-defined zones into the reader so custom zone names resolve
    _sync_active_zones(reader)

    # Determine which zone to use
    zone_name = zone
    if not zone_name:
        # Use first zone from reader or active zone def
        if reader.zones:
            zone_name = reader.zones[0]
        else:
            raise HTTPException(status_code=400, detail="No zone specified and no zones available")

    try:
        df = reader.get_dataframe(zone_name, volume_factor=volume_factor)
    except (KeyError, IndexError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    # Convert index to ISO strings
    if hasattr(df.index, "strftime"):
        time_strings = [t.isoformat() for t in df.index]
    else:
        time_strings = [str(t) for t in df.index]

    # Filter columns if specified
    if columns != "all":
        try:
            col_indices = [int(c.strip()) for c in columns.split(",")]
            all_col_names = list(df.columns)
            selected = [all_col_names[i] for i in col_indices if i < len(all_col_names)]
            df = df[selected]
        except (ValueError, IndexError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid column indices: {e}") from e

    result_columns = []
    for col_name in df.columns:
        result_columns.append(
            {
                "name": col_name,
                "values": _sanitize_values(df[col_name].tolist()),
                "units": "Volume",
            }
        )

    # Build units metadata from model
    model = model_state.model
    meta = model.metadata if model else {}
    length_unit = meta.get("length_unit", "FT").upper().strip()
    if length_unit in ("FT", "FEET", "FOOT"):
        source_volume = "FT3"
        source_area = "SQ.FT."
    elif length_unit in ("M", "METERS", "METER"):
        source_volume = "M3"
        source_area = "M2"
    else:
        source_volume = "FT3"
        source_area = "SQ.FT."

    return {
        "location": zone_name,
        "times": time_strings,
        "columns": result_columns,
        "units_metadata": {
            "source_volume_unit": source_volume,
            "source_area_unit": source_area,
            "source_area_output_unit": meta.get("area_unit", "ACRES"),
            "source_length_unit": meta.get("length_unit", "FT"),
            "timestep_unit": reader.header.time_unit or "1MON",
            "has_volume_columns": True,
            "has_area_columns": False,
            "has_length_columns": False,
        },
    }


@router.get("/{zbudget_type}/summary")
def get_zbudget_summary(
    zbudget_type: str,
    zone: str = Query(default="", description="Zone name or index"),
) -> dict:
    """Get ZBudget summary statistics for a zone."""
    reader = model_state.get_zbudget_reader(zbudget_type)
    if reader is None:
        raise HTTPException(
            status_code=404,
            detail=f"ZBudget type '{zbudget_type}' not available",
        )

    # Sync user-defined zones into the reader so custom zone names resolve
    _sync_active_zones(reader)

    zone_name = zone
    if not zone_name and reader.zones:
        zone_name = reader.zones[0]

    try:
        df = reader.get_dataframe(zone_name)
    except (KeyError, IndexError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    totals: dict[str, float | None] = {}
    averages: dict[str, float | None] = {}
    for col_name in df.columns:
        col_vals = np.asarray(df[col_name].values)
        totals[col_name] = _safe_float(float(np.nansum(col_vals)))
        averages[col_name] = _safe_float(float(np.nanmean(col_vals)))

    return {
        "location": zone_name,
        "n_timesteps": len(df),
        "totals": totals,
        "averages": averages,
    }


@router.get("/glossary")
def get_zbudget_glossary() -> dict:
    """Get definitions of ZBudget column names."""
    return ZBUDGET_GLOSSARY
