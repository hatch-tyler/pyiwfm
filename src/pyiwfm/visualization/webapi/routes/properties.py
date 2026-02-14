"""
Property data API routes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from pyiwfm.visualization.webapi.config import model_state
from pyiwfm.visualization.webapi.properties import PROPERTY_INFO

router = APIRouter(prefix="/api/properties", tags=["properties"])


class PropertyListItem(BaseModel):
    """Property list item."""

    id: str
    name: str
    units: str
    description: str
    cmap: str
    log_scale: bool


class PropertyData(BaseModel):
    """Property array response."""

    property_id: str
    name: str
    units: str
    values: list[float]
    min: float
    max: float
    mean: float


@router.get("", response_model=list[PropertyListItem])
def list_properties() -> list[PropertyListItem]:
    """List available properties for visualization."""
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    model = model_state.model
    available = ["layer"]

    if model.stratigraphy is not None:
        available.extend(["thickness", "top_elev", "bottom_elev"])

    if model.groundwater is not None:
        params = model.groundwater.aquifer_params
        if params is not None:
            if hasattr(params, "kh") and params.kh is not None:
                available.append("kh")
            if hasattr(params, "kv") and params.kv is not None:
                available.append("kv")
            ss = getattr(params, "specific_storage", getattr(params, "ss", None))
            if ss is not None:
                available.append("ss")
            sy = getattr(params, "specific_yield", getattr(params, "sy", None))
            if sy is not None:
                available.append("sy")

    result = []
    for prop_id in available:
        info = PROPERTY_INFO.get(
            prop_id,
            {
                "name": prop_id,
                "units": "",
                "description": prop_id,
                "cmap": "viridis",
                "log_scale": False,
            },
        )
        result.append(
            PropertyListItem(
                id=prop_id,
                name=info["name"],
                units=info["units"],
                description=info["description"],
                cmap=info["cmap"],
                log_scale=info["log_scale"],
            )
        )

    return result


@router.get("/{property_id}", response_model=PropertyData)
def get_property(
    property_id: str,
    layer: int = Query(default=0, ge=0, description="Layer filter (0 for all)"),
) -> PropertyData:
    """Get property values as an array."""
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    model = model_state.model

    values = _compute_property_values(property_id, layer)
    if values is None:
        raise HTTPException(
            status_code=404, detail=f"Property '{property_id}' not available"
        )

    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        valid = np.array([0.0])

    info = PROPERTY_INFO.get(
        property_id, {"name": property_id, "units": "", "description": property_id}
    )

    return PropertyData(
        property_id=property_id,
        name=info["name"],
        units=info.get("units", ""),
        values=values.tolist(),
        min=float(np.min(valid)),
        max=float(np.max(valid)),
        mean=float(np.mean(valid)),
    )


def _node_to_element_values(
    node_data: np.ndarray,
    grid: Any,
    n_elements: int,
    n_layers: int,
) -> np.ndarray:
    """Average node-based values to elements using corner-node averaging."""
    sorted_node_ids = sorted(grid.nodes.keys())
    node_id_to_idx = {nid: i for i, nid in enumerate(sorted_node_ids)}
    sorted_elem_ids = sorted(grid.elements.keys())

    values = np.zeros(n_elements * n_layers, dtype=np.float64)
    is_2d = node_data.ndim == 2

    for lay in range(n_layers):
        offset = lay * n_elements
        for i, eid in enumerate(sorted_elem_ids):
            elem = grid.elements[eid]
            node_vals = []
            for nid in elem.vertices:
                idx = node_id_to_idx.get(nid)
                if idx is not None:
                    if is_2d and lay < node_data.shape[1]:
                        node_vals.append(float(node_data[idx, lay]))
                    elif not is_2d:
                        node_vals.append(float(node_data[idx]))
            values[offset + i] = (
                sum(node_vals) / len(node_vals) if node_vals else 0.0
            )
    return values


def _compute_property_values(
    property_id: str, layer: int = 0
) -> np.ndarray | None:
    """Compute property values for the mesh cells."""
    model = model_state.model
    if model is None:
        return None

    grid = model.grid
    strat = model.stratigraphy

    n_elements = grid.n_elements
    n_layers = strat.n_layers if strat is not None else 1
    n_cells = n_elements * n_layers

    if property_id == "layer":
        values = np.zeros(n_cells, dtype=np.float64)
        for lay in range(n_layers):
            start = lay * n_elements
            end = (lay + 1) * n_elements
            values[start:end] = lay + 1
        if layer > 0:
            mask = np.zeros(n_cells, dtype=bool)
            start = (layer - 1) * n_elements
            end = layer * n_elements
            mask[start:end] = True
            values = np.where(mask, values, np.nan)
        return values

    if property_id == "thickness" and strat is not None:
        thickness_2d = strat.top_elev - strat.bottom_elev
        values = _node_to_element_values(thickness_2d, grid, n_elements, n_layers)
        if layer > 0:
            mask = np.zeros(n_cells, dtype=bool)
            start = (layer - 1) * n_elements
            end = layer * n_elements
            mask[start:end] = True
            values = np.where(mask, values, np.nan)
        return values

    if property_id == "top_elev" and strat is not None:
        values = _node_to_element_values(strat.top_elev, grid, n_elements, n_layers)
        if layer > 0:
            mask = np.zeros(n_cells, dtype=bool)
            start = (layer - 1) * n_elements
            end = layer * n_elements
            mask[start:end] = True
            values = np.where(mask, values, np.nan)
        return values

    if property_id == "bottom_elev" and strat is not None:
        values = _node_to_element_values(strat.bottom_elev, grid, n_elements, n_layers)
        if layer > 0:
            mask = np.zeros(n_cells, dtype=bool)
            start = (layer - 1) * n_elements
            end = layer * n_elements
            mask[start:end] = True
            values = np.where(mask, values, np.nan)
        return values

    if property_id in ("kh", "kv", "ss", "sy"):
        if model.groundwater is None:
            return None
        params = model.groundwater.aquifer_params
        if params is None:
            return None

        attr_map = {
            "kh": ["kh"],
            "kv": ["kv"],
            "ss": ["specific_storage", "ss"],
            "sy": ["specific_yield", "sy"],
        }

        param_data = None
        for attr_name in attr_map.get(property_id, [property_id]):
            param_data = getattr(params, attr_name, None)
            if param_data is not None:
                break

        if param_data is None:
            return None

        effective_layers = (
            min(n_layers, param_data.shape[1]) if param_data.ndim == 2 else n_layers
        )
        partial = _node_to_element_values(
            param_data, grid, n_elements, effective_layers
        )
        if effective_layers < n_layers:
            values = np.zeros(n_cells, dtype=np.float64)
            values[: n_elements * effective_layers] = partial
        else:
            values = partial

        if layer > 0:
            mask = np.zeros(n_cells, dtype=bool)
            start = (layer - 1) * n_elements
            end = layer * n_elements
            mask[start:end] = True
            values = np.where(mask, values, np.nan)
        return values

    return None
