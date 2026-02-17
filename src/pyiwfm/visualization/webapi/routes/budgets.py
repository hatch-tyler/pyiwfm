"""
Budget data API routes.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from pyiwfm.io.budget import BUDGET_DATA_TYPES
from pyiwfm.visualization.webapi.config import model_state

router = APIRouter(prefix="/api/budgets", tags=["budgets"])

# Map budget data type codes to human-readable unit descriptions
_DATA_TYPE_UNITS = {
    1: "Volume/Time",      # VR - Volumetric rate
    2: "Volume",           # VLB - Volume at beginning
    3: "Volume",           # VLE - Volume at end
    4: "Area",             # AR - Area
    5: "Length",            # LT - Length
    6: "Volume/Time",      # VR_PotCUAW
    7: "Volume/Time",      # VR_AgSupplyReq
    8: "Volume/Time",      # VR_AgShort
    9: "Volume/Time",      # VR_AgPump
    10: "Volume/Time",     # VR_AgDiv
    11: "Volume/Time",     # VR_AgOthIn
}

# Volume-related type codes (VR variants + VLB/VLE)
_VOLUME_TYPE_CODES = {1, 2, 3, 6, 7, 8, 9, 10, 11}
# Area type code
_AREA_TYPE_CODES = {4}
# Length type code
_LENGTH_TYPE_CODES = {5}

# ---------------------------------------------------------------------------
# Budget glossary — IWFM budget column definitions
# ---------------------------------------------------------------------------

BUDGET_GLOSSARY: dict[str, dict[str, str]] = {
    "gw": {
        "Deep Percolation": "Water percolating from the root zone through unsaturated zone to the aquifer. Inflow to groundwater.",
        "Begin Storage": "Volume of water stored in the aquifer at the start of the timestep.",
        "End Storage": "Volume of water stored in the aquifer at the end of the timestep.",
        "Net Deep Percolation": "Net deep percolation from root zone / unsaturated zone to groundwater.",
        "Gain from Stream": "Water gained by groundwater from stream-aquifer interaction (stream losing). Inflow.",
        "Recharge": "Direct recharge to groundwater from various sources.",
        "Gain from Lake": "Seepage from lakes into the aquifer. Inflow to groundwater.",
        "Boundary Inflow": "Inflow from specified flow or general head boundary conditions.",
        "Subsidence": "Change in aquifer volume due to compaction of fine-grained sediments.",
        "Subsurface Irrigation": "Irrigation water applied to the subsurface (sub-irrigation). Inflow.",
        "Tile Drain Outflow": "Water removed from the aquifer by tile drain systems. Outflow.",
        "Pumping": "Water extracted from the aquifer by wells. Outflow.",
        "Net Subsurface Inflow": "Net horizontal groundwater flow into/out of the subregion.",
        "Loss to Stream": "Water lost from groundwater to streams (stream gaining). Outflow.",
        "Discrepancy": "Numerical mass balance error. Should be small relative to other terms.",
        "Cumulative Subsidence": "Cumulative subsidence (compaction) since simulation start.",
    },
    "stream": {
        "Upstream Inflow": "Water flowing in from upstream reaches. Inflow.",
        "Downstream Outflow": "Water flowing out to downstream reaches. Outflow.",
        "Tributary Inflow": "Inflow from tributary streams joining this reach.",
        "Tile Drain": "Water entering the stream from subsurface tile drain systems.",
        "Runoff": "Surface runoff from the land surface entering the stream.",
        "Return Flow": "Irrigation return flow entering the stream.",
        "Gain from Groundwater": "Stream gaining from groundwater-stream interaction. Inflow.",
        "Gain from Lake": "Water flowing from a lake into the stream.",
        "Diversion": "Water diverted from the stream for agricultural or urban use. Outflow.",
        "By-pass Flow": "Water bypassed around diversions or other structures.",
        "Discrepancy": "Numerical mass balance error.",
        "Diversion Shortage": "Shortfall when actual diversion is less than requested.",
    },
    "rootzone": {
        "AG_Precip": "Agricultural precipitation directly applied to cropland.",
        "AG_Runoff": "Runoff from agricultural areas leaving the root zone.",
        "AG_Prime Applied Water": "Primary irrigation water applied to agricultural land.",
        "AG_Re-used Water": "Recycled/reused water applied to agricultural land.",
        "AG_Net Return Flow": "Net return flow from agricultural areas to streams.",
        "AG_Begin Storage": "Soil moisture in agricultural root zone at start of timestep.",
        "AG_End Storage": "Soil moisture in agricultural root zone at end of timestep.",
        "AG_Deep Percolation": "Water draining below the agricultural root zone to deeper layers.",
        "AG_ET": "Evapotranspiration from agricultural crops.",
        "AG_Potential CUAW": "Potential consumptive use of applied water for agriculture.",
        "AG_Supply Requirement": "Total water supply requirement for agricultural crops.",
        "AG_Shortage": "Shortfall of water supply vs. requirement for agriculture.",
        "AG_AREA": "Agricultural land area (acres).",
        "URB_Precip": "Precipitation on urban areas.",
        "URB_Runoff": "Runoff from urban areas.",
        "URB_Prime Applied Water": "Primary water supply to urban areas.",
        "URB_Re-used Water": "Recycled water supplied to urban areas.",
        "URB_Net Return Flow": "Net return flow from urban areas.",
        "URB_Begin Storage": "Soil moisture in urban root zone at start of timestep.",
        "URB_End Storage": "Soil moisture in urban root zone at end of timestep.",
        "URB_Deep Percolation": "Water draining below the urban root zone.",
        "URB_ET": "Evapotranspiration from urban vegetation and pervious areas.",
        "URB_AREA": "Urban land area (acres).",
        "NRV_Precip": "Precipitation on native/riparian vegetation.",
        "NRV_Runoff": "Runoff from native/riparian areas.",
        "NRV_Begin Storage": "Soil moisture in native/riparian root zone at start of timestep.",
        "NRV_End Storage": "Soil moisture in native/riparian root zone at end of timestep.",
        "NRV_Deep Percolation": "Water draining below the native/riparian root zone.",
        "NRV_ET": "Evapotranspiration from native and riparian vegetation.",
        "NRV_AREA": "Native/riparian vegetation area (acres).",
    },
    "lwu": {
        "AG_Demand": "Agricultural water demand for the period.",
        "AG_Supply": "Agricultural water supply delivered.",
        "AG_Shortage": "Shortfall of agricultural supply vs. demand.",
        "AG_Pumping": "Groundwater pumping for agricultural use.",
        "AG_Diversion": "Surface water diverted for agricultural use.",
        "AG_Other Inflow": "Other water inflows to agricultural use.",
        "AG_AREA": "Agricultural land area (acres).",
        "URB_Demand": "Urban water demand for the period.",
        "URB_Supply": "Urban water supply delivered.",
        "URB_Shortage": "Shortfall of urban supply vs. demand.",
        "URB_Pumping": "Groundwater pumping for urban use.",
        "URB_Diversion": "Surface water diverted for urban use.",
        "URB_Other Inflow": "Other water inflows to urban use.",
        "URB_AREA": "Urban land area (acres).",
    },
    "diversion": {
        "Actual Diversion": "Amount of water actually diverted from the stream.",
        "Delivery": "Water successfully delivered to the diversion destination.",
        "Recoverable Loss": "Water lost during conveyance that returns to groundwater.",
        "Non-Recoverable Loss": "Water lost during conveyance that is permanently removed from the system.",
        "Diversion Shortage": "Shortfall when actual diversion is less than requested amount.",
    },
    "lake": {
        "Begin Storage": "Volume of water in the lake at the start of the timestep.",
        "End Storage": "Volume of water in the lake at the end of the timestep.",
        "Inflow from Upstream Lakes": "Water flowing in from upstream connected lakes.",
        "Inflow from Streams": "Water flowing in from connected stream reaches.",
        "Precipitation": "Direct precipitation falling on the lake surface.",
        "Gain from Groundwater": "Groundwater seeping into the lake (gaining lake).",
        "Evaporation": "Water lost from the lake surface to the atmosphere.",
        "Outflow": "Water flowing out of the lake to downstream.",
        "Discrepancy": "Numerical mass balance error.",
    },
    "unsaturated": {
        "Begin Storage": "Water stored in the unsaturated zone at start of timestep.",
        "End Storage": "Water stored in the unsaturated zone at end of timestep.",
        "Percolation In": "Water entering the unsaturated zone from the root zone above.",
        "Percolation Out": "Water leaving the unsaturated zone to the aquifer below.",
        "Deep Percolation": "Deep percolation through the unsaturated zone.",
        "Discrepancy": "Numerical mass balance error.",
    },
    "stream_node": {
        "Upstream Inflow": "Water flowing in from the upstream stream node.",
        "Downstream Outflow": "Water flowing out to the downstream stream node.",
        "Tributary Inflow": "Inflow from tributary streams at this node.",
        "Tile Drain": "Water entering the stream from tile drain systems at this node.",
        "Runoff": "Surface runoff entering the stream at this node.",
        "Return Flow": "Irrigation return flow entering the stream at this node.",
        "Gain from Groundwater": "Groundwater-stream interaction gain at this node.",
        "Gain from Lake": "Lake inflow to the stream at this node.",
        "Diversion": "Water diverted from the stream at this node.",
        "By-pass Flow": "Water bypassed around this node.",
        "Discrepancy": "Numerical mass balance error.",
    },
    "small_watershed": {
        "Precipitation": "Total precipitation on the small watershed area.",
        "Runoff": "Surface runoff generated from the watershed.",
        "Base Flow": "Baseflow contribution from shallow groundwater.",
        "Deep Percolation": "Water percolating below the watershed root zone.",
        "ET": "Evapotranspiration from the small watershed.",
        "Begin Storage": "Soil moisture storage at the start of the timestep.",
        "End Storage": "Soil moisture storage at the end of the timestep.",
        "Discrepancy": "Numerical mass balance error.",
    },
}


def _safe_float(val: float) -> float | None:
    """Convert NaN/Inf to None for JSON-safe serialization."""
    if val is None or math.isnan(val) or math.isinf(val):
        return None
    return val


def _sanitize_values(values: list) -> list:
    """Replace NaN/Inf with None in a list of numeric values."""
    return [
        None if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else v
        for v in values
    ]


def _get_column_units(reader: object, location: str | int = 0) -> list[str]:
    """Get units for each column based on budget data type codes."""
    loc_idx = reader.get_location_index(location)
    if len(reader.header.location_data) == 1:
        loc_data = reader.header.location_data[0]
    else:
        loc_data = reader.header.location_data[loc_idx]

    units = []
    for col_type in loc_data.column_types:
        units.append(_DATA_TYPE_UNITS.get(col_type, ""))
    return units


def _parse_title_units(reader: object) -> dict[str, str]:
    """Try to extract unit strings from budget header titles (e.g. 'UNIT OF VOLUME = TAF')."""
    result: dict[str, str] = {}
    try:
        titles = getattr(reader.header, "ascii_output", None)
        if titles and hasattr(titles, "titles"):
            for line in titles.titles:
                upper = line.upper()
                if "UNIT OF VOLUME" in upper and "=" in upper:
                    result["volume"] = line.split("=")[-1].strip()
                elif "UNIT OF AREA" in upper and "=" in upper:
                    result["area"] = line.split("=")[-1].strip()
                elif "UNIT OF LENGTH" in upper and "=" in upper:
                    result["length"] = line.split("=")[-1].strip()
    except (AttributeError, TypeError):
        pass
    return result


def _detect_budget_category(budget_type: str) -> str:
    """Detect budget type category from the type string."""
    t = budget_type.lower()
    if t.startswith("gw") or "groundwater" in t:
        return "gw"
    if t in ("lwu",) or "land" in t:
        return "lwu"
    if "root" in t:
        return "rootzone"
    if "unsat" in t:
        return "unsaturated"
    # stream_node must come before stream to avoid false match
    if t == "stream_node" or "stream_node" in t or "stream node" in t:
        return "stream_node"
    if "stream" in t:
        return "stream"
    if "diver" in t:
        return "diversion"
    if "lake" in t:
        return "lake"
    if t == "small_watershed" or "small_watershed" in t or "small watershed" in t:
        return "small_watershed"
    return "other"


def _get_budget_units_metadata(budget_type: str, reader: object) -> dict:
    """Build units metadata for a budget type based on model metadata and column types."""
    model = model_state.model
    meta = model.metadata if model else {}
    category = _detect_budget_category(budget_type)

    # HDF budget files store values in simulation units (not output units).
    # IWFM has no direct volume unit field — derive from the length unit:
    #   FT → FT3, M → M3.  The metadata fields like gw_volume_output_unit
    # (UNITVLOU) and volume_unit are post-processing output units and must
    # NOT be used as the source unit for raw HDF data.
    length_unit = meta.get("length_unit", "FT").upper().strip()
    if length_unit in ("FT", "FEET", "FOOT"):
        source_volume = "FT3"
        source_area = "SQ.FT."
    elif length_unit in ("M", "METERS", "METER"):
        source_volume = "M3"
        source_area = "M2"
    else:
        source_volume = "FT3"  # safe default for US models
        source_area = "SQ.FT."
    source_length = meta.get("length_unit", "FT")

    # Determine timestep unit
    ts = reader.header.timestep
    timestep_unit = ts.unit if ts.unit else "1MON"

    # Scan column types for presence of volume/area/length columns
    has_volume = False
    has_area = False
    has_length = False
    try:
        loc_data = reader.header.location_data[0]
        if loc_data.column_types:
            # Use explicit type codes when available
            for ct in loc_data.column_types:
                if ct in _VOLUME_TYPE_CODES:
                    has_volume = True
                elif ct in _AREA_TYPE_CODES:
                    has_area = True
                elif ct in _LENGTH_TYPE_CODES:
                    has_length = True
        elif loc_data.column_headers:
            # Infer from column names when type codes are missing (common in
            # HDF5 files that lack the iDataColumnTypes attribute).
            for hdr in loc_data.column_headers:
                upper = hdr.upper()
                if "AREA" in upper and (
                    upper.endswith("AREA") or upper.endswith("_AREA")
                    or "AREA)" in upper
                ):
                    has_area = True
                elif "SUBSID" in upper or "CUM_SUBSID" in upper:
                    has_volume = True  # subsidence in GW budgets is volumetric
                else:
                    has_volume = True
        else:
            # No column info at all — assume volume
            has_volume = True
    except (AttributeError, IndexError):
        # Default: assume volume columns exist
        has_volume = True

    return {
        "source_volume_unit": source_volume or "AF",
        "source_area_unit": source_area or "ACRES",
        "source_area_output_unit": meta.get("area_unit", "ACRES"),
        "source_length_unit": source_length or "FEET",
        "timestep_unit": timestep_unit,
        "has_volume_columns": has_volume,
        "has_area_columns": has_area,
        "has_length_columns": has_length,
    }


@router.get("/types")
def get_budget_types() -> list[str]:
    """Get list of available budget types."""
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")
    return model_state.get_available_budgets()


@router.get("/glossary")
def get_budget_glossary() -> dict:
    """Get definitions of budget column names for all budget types."""
    return BUDGET_GLOSSARY


@router.get("/{budget_type}/locations")
def get_budget_locations(budget_type: str) -> dict:
    """Get locations/subregions for a budget type."""
    reader = model_state.get_budget_reader(budget_type)
    if reader is None:
        raise HTTPException(
            status_code=404,
            detail=f"Budget type '{budget_type}' not available",
        )

    return {
        "locations": [
            {"id": i, "name": name}
            for i, name in enumerate(reader.locations)
        ],
    }


@router.get("/{budget_type}/columns")
def get_budget_columns(
    budget_type: str,
    location: str = Query(default="", description="Location name or index"),
) -> dict:
    """Get column headers for a budget location."""
    reader = model_state.get_budget_reader(budget_type)
    if reader is None:
        raise HTTPException(
            status_code=404,
            detail=f"Budget type '{budget_type}' not available",
        )

    loc = location if location else 0
    try:
        headers = reader.get_column_headers(loc)
        units = _get_column_units(reader, loc)
    except (KeyError, IndexError) as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "columns": [
            {"id": i, "name": name, "units": units[i] if i < len(units) else ""}
            for i, name in enumerate(headers)
        ],
    }


@router.get("/{budget_type}/data")
def get_budget_data(
    budget_type: str,
    location: str = Query(default="", description="Location name or index"),
    columns: str = Query(default="all", description="Column indices (comma-separated) or 'all'"),
) -> dict:
    """Get budget time series data for a location."""
    reader = model_state.get_budget_reader(budget_type)
    if reader is None:
        raise HTTPException(
            status_code=404,
            detail=f"Budget type '{budget_type}' not available",
        )

    loc = location if location else 0

    try:
        col_indices = None
        if columns != "all":
            col_indices = [int(c.strip()) for c in columns.split(",")]

        times_arr, values_arr = reader.get_values(loc, col_indices)
        headers = reader.get_column_headers(loc)
    except (KeyError, IndexError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Convert times to ISO strings
    ts = reader.header.timestep
    if ts.start_datetime:
        from datetime import timedelta

        # Use relativedelta for monthly timesteps to avoid drift
        use_months = "MON" in ts.unit.upper() if ts.unit else False
        if use_months:
            from dateutil.relativedelta import relativedelta

        time_strings = []
        for i in range(len(times_arr)):
            if use_months:
                dt = ts.start_datetime + relativedelta(months=i)
            else:
                dt = ts.start_datetime + timedelta(minutes=ts.delta_t_minutes * i)
            time_strings.append(dt.isoformat())
    else:
        time_strings = [str(t) for t in times_arr.tolist()]

    # Build column data
    try:
        all_units = _get_column_units(reader, loc)
    except (KeyError, IndexError):
        all_units = []

    if col_indices is not None:
        col_names = [headers[i] for i in col_indices]
        col_units = [all_units[i] if i < len(all_units) else "" for i in col_indices]
    else:
        col_names = headers
        col_units = all_units

    result_columns = []
    for i, name in enumerate(col_names):
        result_columns.append({
            "name": name,
            "values": _sanitize_values(values_arr[:, i].tolist()),
            "units": col_units[i] if i < len(col_units) else "",
        })

    # Build units metadata
    units_metadata = _get_budget_units_metadata(budget_type, reader)

    return {
        "location": location or reader.locations[0],
        "times": time_strings,
        "columns": result_columns,
        "units_metadata": units_metadata,
    }


@router.get("/{budget_type}/summary")
def get_budget_summary(
    budget_type: str,
    location: str = Query(default="", description="Location name or index"),
) -> dict:
    """Get budget summary statistics for a location."""
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
        raise HTTPException(status_code=400, detail=str(e))

    totals = {}
    averages = {}
    for i, name in enumerate(headers):
        col_vals = values_arr[:, i]
        totals[name] = _safe_float(float(np.nansum(col_vals)))
        averages[name] = _safe_float(float(np.nanmean(col_vals)))

    return {
        "location": location or reader.locations[0],
        "n_timesteps": len(times_arr),
        "totals": totals,
        "averages": averages,
    }


@router.get("/{budget_type}/spatial")
def get_budget_spatial(
    budget_type: str,
    column: str = Query(default="", description="Column name to map"),
    stat: str = Query(
        default="total", description="Statistic: 'total', 'average', or 'last'"
    ),
) -> dict:
    """
    Get per-location (subregion) budget values for spatial mapping.

    Returns a value per location suitable for coloring subregion polygons
    on the 2D map as a budget heatmap.
    """
    reader = model_state.get_budget_reader(budget_type)
    if reader is None:
        raise HTTPException(
            status_code=404,
            detail=f"Budget type '{budget_type}' not available",
        )

    # Find the target column index
    first_headers = reader.get_column_headers(0)
    col_idx = None
    if column:
        for i, h in enumerate(first_headers):
            if h.lower() == column.lower():
                col_idx = i
                break
        if col_idx is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Column '{column}' not found. "
                    f"Available: {first_headers}"
                ),
            )
    else:
        # Default to first column
        col_idx = 0

    locations: list[dict] = []
    for loc_i, loc_name in enumerate(reader.locations):
        try:
            times_arr, values_arr = reader.get_values(loc_i, [col_idx])
            col_vals = values_arr[:, 0]

            if stat == "total":
                value = float(np.nansum(col_vals))
            elif stat == "average":
                value = float(np.nanmean(col_vals))
            elif stat == "last":
                value = float(col_vals[-1]) if len(col_vals) > 0 else 0.0
            else:
                value = float(np.nansum(col_vals))

            safe = _safe_float(value)
            locations.append({
                "id": loc_i,
                "name": loc_name,
                "value": round(safe, 2) if safe is not None else 0.0,
            })
        except (KeyError, IndexError):
            locations.append({
                "id": loc_i,
                "name": loc_name,
                "value": 0.0,
            })

    all_vals = [loc["value"] for loc in locations]
    vmin = min(all_vals) if all_vals else 0.0
    vmax = max(all_vals) if all_vals else 0.0

    return {
        "budget_type": budget_type,
        "column": column or first_headers[0],
        "stat": stat,
        "n_locations": len(locations),
        "locations": locations,
        "min": round(vmin, 2),
        "max": round(vmax, 2),
        "available_columns": first_headers,
    }


@router.get("/{budget_type}/location-geometry")
def get_budget_location_geometry(
    budget_type: str,
    location: str = Query(default="", description="Location name or index"),
) -> dict:
    """
    Return spatial metadata for a budget location so the frontend can
    highlight the correct feature on a mini-map.

    For most types ``geometry`` is ``null`` — the frontend matches by
    ``location_index`` against existing spatial data it already fetched.
    For ``stream_node``, the response includes a ``[lng, lat]`` point.
    """
    reader = model_state.get_budget_reader(budget_type)
    if reader is None:
        raise HTTPException(
            status_code=404,
            detail=f"Budget type '{budget_type}' not available",
        )

    category = _detect_budget_category(budget_type)

    loc = location if location else 0
    try:
        loc_idx = reader.get_location_index(loc)
    except (KeyError, IndexError):
        loc_idx = 0

    spatial_type_map = {
        "gw": "subregion",
        "rootzone": "subregion",
        "lwu": "subregion",
        "unsaturated": "subregion",
        "stream": "reach",
        "stream_node": "point",
        "diversion": "diversion",
        "lake": "lake",
        "small_watershed": "small_watershed",
    }

    # Detect "ENTIRE MODEL AREA" or similar whole-model location
    loc_name = ""
    if loc_idx < len(reader.locations):
        loc_name = reader.locations[loc_idx]
    is_entire_model = any(
        kw in loc_name.upper() for kw in ("ENTIRE MODEL", "ENTIRE AREA", "TOTAL MODEL")
    )

    result: dict = {
        "spatial_type": "entire_model" if is_entire_model else spatial_type_map.get(category, "unknown"),
        "location_index": loc_idx,
        "location_name": loc_name,
        "geometry": None,
    }

    # For stream_node, compute the WGS-84 point
    if category == "stream_node":
        model = model_state.model
        if model and model.streams:
            stream = model.streams
            node_ids = getattr(stream, "budget_node_ids", None)
            if node_ids and loc_idx < len(node_ids):
                snode_id = node_ids[loc_idx]
                snode = stream.nodes.get(snode_id)
                if snode:
                    gw_id = getattr(snode, "gw_node", None)
                    if gw_id is not None and model.grid:
                        gw_node = model.grid.nodes.get(gw_id)
                        if gw_node:
                            lng, lat = model_state.reproject_coords(
                                gw_node.x, gw_node.y
                            )
                            result["geometry"] = {
                                "type": "Point",
                                "coordinates": [lng, lat],
                            }

    return result


@router.get("/water-balance")
def get_water_balance() -> dict:
    """
    Get aggregated water balance across all budget types for Sankey diagram.

    Returns nodes (components) and links (flows) for a Plotly Sankey chart.
    Each link represents a flow pathway between model components.
    """
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    available = model_state.get_available_budgets()
    if not available:
        raise HTTPException(status_code=404, detail="No budget data available")

    # Collect total flows from each budget type
    budget_totals: dict[str, dict[str, float]] = {}
    for btype in available:
        reader = model_state.get_budget_reader(btype)
        if reader is None:
            continue
        try:
            headers = reader.get_column_headers(0)
            _, values_arr = reader.get_values(0)
            totals = {}
            for i, name in enumerate(headers):
                val = _safe_float(abs(float(np.nansum(values_arr[:, i]))))
                totals[name] = val if val is not None else 0.0
            budget_totals[btype] = totals
        except (KeyError, IndexError):
            continue

    # Build Sankey nodes and links
    # Nodes represent components: GW, Stream, RootZone, Lake, etc.
    node_names: list[str] = []
    node_set: dict[str, int] = {}

    def get_node_idx(name: str) -> int:
        if name not in node_set:
            node_set[name] = len(node_names)
            node_names.append(name)
        return node_set[name]

    links: list[dict] = []

    # Map common budget column names to Sankey flows
    flow_mappings = {
        "gw": {
            "Deep Percolation": ("Root Zone", "Groundwater"),
            "Stream Leakage": ("Streams", "Groundwater"),
            "Pumping": ("Groundwater", "Pumping"),
            "Net Subsurface Inflow": ("Boundary", "Groundwater"),
            "Recharge": ("Surface", "Groundwater"),
            "Lake Seepage": ("Lakes", "Groundwater"),
            "Subsidence": ("Groundwater", "Subsidence"),
        },
        "stream": {
            "Upstream Inflow": ("Upstream", "Streams"),
            "Tributary Inflow": ("Tributaries", "Streams"),
            "Downstream Outflow": ("Streams", "Downstream"),
            "Diversion": ("Streams", "Diversions"),
            "Return Flow": ("Return Flows", "Streams"),
            "Stream-GW Interaction": ("Streams", "Groundwater"),
            "Lake Inflow": ("Streams", "Lakes"),
        },
        "rootzone": {
            "Precipitation": ("Precipitation", "Root Zone"),
            "Applied Water": ("Applied Water", "Root Zone"),
            "ET": ("Root Zone", "ET"),
            "Deep Percolation": ("Root Zone", "Groundwater"),
            "Runoff": ("Root Zone", "Surface Runoff"),
        },
    }

    for btype, totals in budget_totals.items():
        mappings = flow_mappings.get(btype, {})
        for col_name, total_val in totals.items():
            if total_val < 1e-6:
                continue
            # Check for partial match in mappings
            matched = False
            for pattern, (src, dst) in mappings.items():
                if pattern.lower() in col_name.lower():
                    links.append({
                        "source": get_node_idx(src),
                        "target": get_node_idx(dst),
                        "value": round(total_val, 1),
                        "label": col_name,
                    })
                    matched = True
                    break
            if not matched:
                # Generic: attribute to the budget type component
                component = {
                    "gw": "Groundwater",
                    "stream": "Streams",
                    "rootzone": "Root Zone",
                    "lake": "Lakes",
                }.get(btype, btype.title())
                if "inflow" in col_name.lower() or "in" == col_name[-2:].lower():
                    links.append({
                        "source": get_node_idx(f"{col_name}"),
                        "target": get_node_idx(component),
                        "value": round(total_val, 1),
                        "label": col_name,
                    })
                else:
                    links.append({
                        "source": get_node_idx(component),
                        "target": get_node_idx(f"{col_name}"),
                        "value": round(total_val, 1),
                        "label": col_name,
                    })

    return {
        "nodes": node_names,
        "links": links,
    }
