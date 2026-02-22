"""
Model information API routes.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from pyiwfm.visualization.webapi.config import model_state, require_model

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/model", tags=["model"])


class ModelInfo(BaseModel):
    """Model metadata response."""

    name: str
    n_nodes: int
    n_elements: int
    n_layers: int
    has_streams: bool
    has_lakes: bool
    n_stream_nodes: int | None
    n_lakes: int | None


class BoundsInfo(BaseModel):
    """Bounding box response."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float


@router.get("/info", response_model=ModelInfo)
def get_model_info() -> ModelInfo:
    """Get model metadata."""
    model = require_model()

    return ModelInfo(
        name=model.name,
        n_nodes=model.n_nodes,
        n_elements=model.n_elements,
        n_layers=model.n_layers,
        has_streams=model.has_streams,
        has_lakes=model.has_lakes,
        n_stream_nodes=model.n_stream_nodes if model.has_streams else None,
        n_lakes=model.n_lakes if model.has_lakes else None,
    )


@router.get("/bounds", response_model=BoundsInfo)
def get_model_bounds() -> BoundsInfo:
    """Get model bounding box."""
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    bounds = model_state.get_bounds()

    return BoundsInfo(
        xmin=bounds[0],
        xmax=bounds[1],
        ymin=bounds[2],
        ymax=bounds[3],
        zmin=bounds[4],
        zmax=bounds[5],
    )


# ===================================================================
# Model Summary (detailed overview)
# ===================================================================


class MeshSummary(BaseModel):
    """Mesh summary section."""

    n_nodes: int
    n_elements: int
    n_layers: int
    n_subregions: int | None = None


class GroundwaterSummary(BaseModel):
    """Groundwater component summary."""

    loaded: bool
    n_wells: int | None = None
    n_hydrograph_locations: int | None = None
    n_boundary_conditions: int | None = None
    n_tile_drains: int | None = None
    has_aquifer_params: bool = False


class StreamSummary(BaseModel):
    """Stream component summary."""

    loaded: bool
    n_nodes: int | None = None
    n_reaches: int | None = None
    n_diversions: int | None = None
    n_bypasses: int | None = None


class LakeSummary(BaseModel):
    """Lake component summary."""

    loaded: bool
    n_lakes: int | None = None
    n_lake_elements: int | None = None


class RootZoneSummary(BaseModel):
    """Root zone component summary."""

    loaded: bool
    n_crop_types: int | None = None
    n_land_use_types: int | None = None
    land_use_type_names: list[str] | None = None
    n_soil_parameter_sets: int | None = None
    missing_soil_param_elements: list[int] | None = None
    n_land_use_elements: int | None = None
    n_missing_land_use: int | None = None
    land_use_coverage: str | None = None
    n_area_timesteps: int | None = None


class SmallWatershedSummary(BaseModel):
    """Small watershed component summary."""

    loaded: bool
    n_watersheds: int | None = None


class UnsaturatedZoneSummary(BaseModel):
    """Unsaturated zone component summary."""

    loaded: bool
    n_layers: int | None = None
    n_elements: int | None = None


class AvailableResults(BaseModel):
    """Available simulation results info."""

    has_head_data: bool = False
    n_head_timesteps: int = 0
    has_gw_hydrographs: bool = False
    has_stream_hydrographs: bool = False
    n_budget_types: int = 0
    budget_types: list[str] = []


class ModelSummary(BaseModel):
    """Full model summary response."""

    name: str
    source: str | None = None
    mesh: MeshSummary
    groundwater: GroundwaterSummary
    streams: StreamSummary
    lakes: LakeSummary
    rootzone: RootZoneSummary
    small_watersheds: SmallWatershedSummary
    unsaturated_zone: UnsaturatedZoneSummary
    available_results: AvailableResults


@router.get("/summary", response_model=ModelSummary)
def get_model_summary() -> ModelSummary:
    """Get a structured summary of all model components."""
    model = require_model()

    # Mesh
    n_subregions = None
    grid = getattr(model, "grid", None) or getattr(model, "mesh", None)
    if grid is not None:
        n_subregions = getattr(grid, "n_subregions", None) or 0
        # Fix B1: grid.n_subregions == 0 when Subregion objects not built,
        # but Element.subregion is populated from the elements file.
        if n_subregions == 0 and getattr(grid, "n_elements", 0) > 0:
            try:
                unique_sr = {
                    elem.subregion
                    for elem in grid.iter_elements()
                    if getattr(elem, "subregion", 0) > 0
                }
                if unique_sr:
                    n_subregions = len(unique_sr)
            except Exception:
                pass
        n_subregions = n_subregions or None

    mesh = MeshSummary(
        n_nodes=model.n_nodes,
        n_elements=model.n_elements,
        n_layers=model.n_layers,
        n_subregions=n_subregions,
    )

    # Groundwater
    gw = model.groundwater if hasattr(model, "groundwater") else None
    if gw is not None:
        n_bc = getattr(gw, "n_boundary_conditions", None) or 0
        n_td = getattr(gw, "n_tile_drains", None) or 0

        # Fix B3: Fall back to metadata when component counts are 0
        # (object creation may fail silently, but metadata keys are set)
        meta = model.metadata if hasattr(model, "metadata") else {}
        if n_bc == 0:
            n_bc = (
                meta.get("gw_n_specified_flow_bc", 0)
                + meta.get("gw_n_specified_head_bc", 0)
                + meta.get("gw_n_general_head_bc", 0)
            ) or None
        else:
            n_bc = n_bc or None
        if n_td == 0:
            n_td = meta.get("gw_n_tile_drains") or None
        else:
            n_td = n_td or None

        groundwater = GroundwaterSummary(
            loaded=True,
            n_wells=getattr(gw, "n_wells", None),
            n_hydrograph_locations=getattr(gw, "n_hydrograph_locations", None),
            n_boundary_conditions=n_bc,
            n_tile_drains=n_td,
            has_aquifer_params=getattr(gw, "aquifer_params", None) is not None,
        )
    else:
        groundwater = GroundwaterSummary(loaded=False)

    # Streams
    stm = model.streams if hasattr(model, "streams") else None
    if stm is not None:
        n_reaches = getattr(stm, "n_reaches", None) or 0
        # Fix B2: reaches dict may be empty but StrmNode.reach_id is populated
        if n_reaches == 0 and getattr(stm, "n_nodes", 0) > 0:
            try:
                nodes_dict = getattr(stm, "nodes", {})
                unique_reaches = {
                    node.reach_id
                    for node in nodes_dict.values()
                    if getattr(node, "reach_id", 0) > 0
                }
                if unique_reaches:
                    n_reaches = len(unique_reaches)
            except Exception:
                pass

        # Try counting from connectivity (terminal nodes ≈ reach count)
        # Only valid when downstream_node is actually populated on some nodes;
        # otherwise every node appears "terminal" giving n_reaches == n_nodes.
        if n_reaches == 0 and getattr(stm, "n_nodes", 0) > 0:
            try:
                nodes_dict = getattr(stm, "nodes", {})
                has_downstream = set()
                for sn in nodes_dict.values():
                    dn = getattr(sn, "downstream_node", None)
                    if dn and dn in nodes_dict:
                        has_downstream.add(sn.id)
                # Only use this heuristic if connectivity data exists
                if has_downstream:
                    terminal = sum(
                        1
                        for sn in nodes_dict.values()
                        if sn.id not in has_downstream and getattr(sn, "gw_node", None) is not None
                    )
                    if terminal > 0:
                        n_reaches = terminal
            except Exception:
                pass

        # Try preprocessor binary as last resort
        if n_reaches == 0:
            try:
                boundaries = model_state.get_stream_reach_boundaries()
                if boundaries:
                    n_reaches = len(boundaries)
            except Exception:
                pass

        streams = StreamSummary(
            loaded=True,
            n_nodes=getattr(stm, "n_nodes", None),
            n_reaches=n_reaches,
            n_diversions=getattr(stm, "n_diversions", None),
            n_bypasses=getattr(stm, "n_bypasses", None),
        )
    else:
        streams = StreamSummary(loaded=False)

    # Lakes
    lk = model.lakes if hasattr(model, "lakes") else None
    if lk is not None:
        lakes = LakeSummary(
            loaded=True,
            n_lakes=getattr(lk, "n_lakes", None),
            n_lake_elements=getattr(lk, "n_lake_elements", None),
        )
    else:
        lakes = LakeSummary(loaded=False)

    # Root zone
    rz = model.rootzone if hasattr(model, "rootzone") else None
    if rz is not None:
        n_crops = getattr(rz, "n_crop_types", None) or 0
        # Fix B4: crop_types dict may be empty but sub-configs are loaded
        if n_crops == 0:
            count = 0
            if getattr(rz, "nonponded_config", None) is not None:
                count += getattr(rz.nonponded_config, "n_crops", 0)
            if getattr(rz, "ponded_config", None) is not None:
                count += 5  # Fixed: 3 rice + 2 refuge
            if getattr(rz, "urban_config", None) is not None:
                count += 1
            if getattr(rz, "native_riparian_config", None) is not None:
                count += 2  # native + riparian
            if count > 0:
                n_crops = count

        # Determine land use types from configs
        lu_type_names: list[str] = []
        if getattr(rz, "nonponded_config", None) is not None:
            lu_type_names.append("Non-ponded Agricultural")
        if getattr(rz, "ponded_config", None) is not None:
            lu_type_names.append("Ponded Agricultural")
        if getattr(rz, "urban_config", None) is not None:
            lu_type_names.append("Urban")
        if getattr(rz, "native_riparian_config", None) is not None:
            lu_type_names.append("Native/Riparian")
        n_lu_types = len(lu_type_names)

        n_soil = len(rz.soil_params) if hasattr(rz, "soil_params") else 0

        # Identify missing element IDs in soil params
        missing_elems: list[int] | None = None
        if n_soil > 0 and n_soil < model.n_elements:
            all_elem_ids = set(range(1, model.n_elements + 1))
            present = set(rz.soil_params.keys())
            missing = sorted(all_elem_ids - present)
            if missing:
                missing_elems = missing

        # Compute land use coverage across elements
        n_lu_elements: int | None = None
        n_missing_lu: int | None = None
        lu_coverage: str | None = None
        n_area_ts: int | None = None

        # Trigger lazy loading if needed
        if not rz.element_landuse and (
            getattr(rz, "nonponded_area_file", None)
            or getattr(rz, "ponded_area_file", None)
            or getattr(rz, "urban_area_file", None)
            or getattr(rz, "native_area_file", None)
        ):
            try:
                from pyiwfm.visualization.webapi.routes.rootzone import (
                    _ensure_land_use_loaded,
                )

                _ensure_land_use_loaded()
            except Exception as exc:
                logger.warning(
                    "Summary: land use lazy-load failed: %s",
                    exc,
                )

        if hasattr(rz, "element_landuse") and rz.element_landuse:
            covered_ids = {elu.element_id for elu in rz.element_landuse}
            n_lu_elements = len(covered_ids)
            n_missing_lu = max(0, model.n_elements - n_lu_elements)
            lu_coverage = f"{n_lu_elements}/{model.n_elements}"

        # Fallback: compute coverage directly from HDF5 area manager
        if n_lu_elements is None:
            try:
                mgr = model_state.get_area_manager()
                if mgr is not None and mgr.n_timesteps > 0:
                    snapshot = mgr.get_snapshot(0)
                    if snapshot:
                        n_lu_elements = len(snapshot)
                        n_missing_lu = max(0, model.n_elements - n_lu_elements)
                        lu_coverage = f"{n_lu_elements}/{model.n_elements}"
            except Exception as exc:
                logger.warning(
                    "Summary: HDF5 area manager stats failed: %s",
                    exc,
                )

        # Get area timestep count from HDF5 manager
        try:
            mgr = model_state.get_area_manager()
            if mgr is not None and mgr.n_timesteps > 0:
                n_area_ts = mgr.n_timesteps
        except Exception:
            pass

        rootzone = RootZoneSummary(
            loaded=True,
            n_crop_types=n_crops or None,
            n_land_use_types=n_lu_types or None,
            land_use_type_names=lu_type_names or None,
            n_soil_parameter_sets=n_soil or None,
            missing_soil_param_elements=missing_elems,
            n_land_use_elements=n_lu_elements,
            n_missing_land_use=n_missing_lu,
            land_use_coverage=lu_coverage,
            n_area_timesteps=n_area_ts,
        )
    else:
        rootzone = RootZoneSummary(loaded=False)

    # Small watersheds
    sw = model.small_watersheds if hasattr(model, "small_watersheds") else None
    if sw is not None:
        small_watersheds = SmallWatershedSummary(
            loaded=True,
            n_watersheds=getattr(sw, "n_watersheds", None),
        )
    else:
        small_watersheds = SmallWatershedSummary(loaded=False)

    # Unsaturated zone
    uz = model.unsaturated_zone if hasattr(model, "unsaturated_zone") else None
    if uz is not None:
        unsaturated_zone = UnsaturatedZoneSummary(
            loaded=True,
            n_layers=getattr(uz, "n_layers", None),
            n_elements=getattr(uz, "n_elements", None),
        )
    else:
        unsaturated_zone = UnsaturatedZoneSummary(loaded=False)

    # Available results — use public getters that perform lazy initialization
    available = AvailableResults()
    try:
        head_loader = model_state.get_head_loader()
        if head_loader is not None:
            available.has_head_data = True
            available.n_head_timesteps = getattr(head_loader, "n_frames", 0)
    except Exception:
        pass

    try:
        gw_reader = model_state.get_gw_hydrograph_reader()
        if gw_reader is not None:
            available.has_gw_hydrographs = True
    except Exception:
        pass

    try:
        stream_reader = model_state.get_stream_hydrograph_reader()
        if stream_reader is not None:
            available.has_stream_hydrographs = True
    except Exception:
        pass

    try:
        budget_types = model_state.get_available_budgets()
        if budget_types:
            available.n_budget_types = len(budget_types)
            available.budget_types = budget_types
    except Exception:
        pass

    # Source
    source = model.metadata.get("source") if hasattr(model, "metadata") else None

    return ModelSummary(
        name=model.name,
        source=source,
        mesh=mesh,
        groundwater=groundwater,
        streams=streams,
        lakes=lakes,
        rootzone=rootzone,
        small_watersheds=small_watersheds,
        unsaturated_zone=unsaturated_zone,
        available_results=available,
    )


@router.get("/cache-status")
def get_cache_status() -> dict:
    """Get SQLite cache status and diagnostics."""
    loader = model_state.get_cache_loader()
    if loader is None:
        return {
            "available": False,
            "path": None,
            "stats": {},
        }

    return {
        "available": True,
        "path": str(loader.cache_path),
        "stats": loader.get_stats(),
    }


@router.post("/compare")
def compare_models(body: dict) -> dict:
    """Compare the loaded model mesh with another model.

    Accepts a JSON body with ``path`` pointing to the second model's
    preprocessor or simulation main file.  Returns a summary of mesh
    and stratigraphy differences using the comparison module.

    Request body:
        {"path": "/path/to/other/model/Preprocessor.in"}
    """
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    other_path = body.get("path")
    if not other_path:
        raise HTTPException(status_code=400, detail="Missing 'path' in request body")

    from pathlib import Path

    other_path = Path(other_path)
    if not other_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model path not found: {other_path}",
        )

    try:
        from pyiwfm.comparison import ModelDiffer
        from pyiwfm.core.model import IWFMModel

        other = IWFMModel.from_preprocessor(other_path)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load comparison model: {e}",
        ) from e

    model = model_state.model
    if model is None or model.grid is None:
        raise HTTPException(status_code=404, detail="No mesh/grid loaded")

    try:
        differ = ModelDiffer()
        diff = differ.diff(
            model.grid,
            other.grid,
            model.stratigraphy,
            other.stratigraphy,
        )
        return diff.to_dict()
    except Exception as e:
        logger.exception("Model comparison failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
