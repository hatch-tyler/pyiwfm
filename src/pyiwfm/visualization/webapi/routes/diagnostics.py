"""
Diagnostics API routes for simulation messages, convergence, and mass balance.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Query

from pyiwfm.visualization.webapi.config import model_state

if TYPE_CHECKING:
    from pyiwfm.io.simulation.messages import SimulationMessagesResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/diagnostics", tags=["diagnostics"])

# Try to import convergence/mass-balance dataclasses (may not exist yet)
_HAS_CONVERGENCE = False
try:
    from pyiwfm.io.simulation.messages import ConvergenceRecord  # noqa: F401

    _HAS_CONVERGENCE = True
except ImportError:
    pass

_HAS_MASS_BALANCE = False
try:
    from pyiwfm.io.simulation.messages import MassBalanceRecord  # noqa: F401

    _HAS_MASS_BALANCE = True
except ImportError:
    pass


def _find_messages_file() -> Path | None:
    """Locate the SimulationMessages.out file in the results directory.

    Returns
    -------
    Path | None
        Path to the file, or ``None`` if not found.
    """
    results_dir = model_state._results_dir
    if results_dir is None or not results_dir.is_dir():
        return None

    # Try several glob patterns
    patterns = ["*SimulationMessages*", "*Messages*.out", "*messages*.out"]
    for pattern in patterns:
        matches = list(results_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _get_diagnostics_result() -> SimulationMessagesResult | None:
    """Return cached diagnostics result, parsing on first access.

    Returns
    -------
    SimulationMessagesResult | None
        Parsed result, or ``None`` if the file doesn't exist.
    """
    # Return cached result if available
    cached: SimulationMessagesResult | None = getattr(model_state, "_diagnostics_result", None)
    if cached is not None:
        return cached

    messages_file = _find_messages_file()
    if messages_file is None:
        return None

    try:
        from pyiwfm.io.simulation.messages import SimulationMessagesReader

        reader = SimulationMessagesReader(messages_file)
        result = reader.read()
        model_state._diagnostics_result = result  # type: ignore[attr-defined]
        return result
    except Exception:
        logger.exception("Failed to parse SimulationMessages file: %s", messages_file)
        return None


def _runtime_seconds(result: SimulationMessagesResult | None) -> float | None:
    """Convert total_runtime timedelta to seconds, or None."""
    if result is None or result.total_runtime is None:
        return None
    return result.total_runtime.total_seconds()


def _resolve_hotspot_coords(entity_type: str, entity_id: int) -> tuple[float | None, float | None]:
    """Resolve coordinates for a convergence hotspot entity.

    Returns
    -------
    tuple[float | None, float | None]
        ``(x, y)`` coordinates, or ``(None, None)`` if unavailable.
    """
    model = model_state._model
    if model is None or model.mesh is None:
        return None, None

    grid = model.mesh
    if entity_type == "groundwater":
        try:
            idx = entity_id - 1  # 1-based to 0-based
            if 0 <= idx < len(grid.nodes):
                node = grid.nodes[idx]
                return node.x, node.y
        except Exception:
            pass
    elif entity_type == "stream":
        try:
            if hasattr(model, "streams") and model.streams is not None:
                strm_node = model.streams.get_node(entity_id)
                if strm_node is not None:
                    # Stream nodes may have x=0, y=0; use gw_node fallback
                    if strm_node.x != 0 or strm_node.y != 0:
                        return strm_node.x, strm_node.y
                    if strm_node.gw_node is not None:
                        gw_idx = strm_node.gw_node - 1
                        if 0 <= gw_idx < len(grid.nodes):
                            gw = grid.nodes[gw_idx]
                            return gw.x, gw.y
        except Exception:
            pass
    elif entity_type == "lake":
        try:
            if hasattr(model, "lakes") and model.lakes is not None:
                lake = model.lakes.get_lake(entity_id)
                if lake is not None and hasattr(lake, "gw_nodes") and lake.gw_nodes:
                    xs: list[float] = []
                    ys: list[float] = []
                    for nid in lake.gw_nodes:
                        idx = nid - 1
                        if 0 <= idx < len(grid.nodes):
                            xs.append(grid.nodes[idx].x)
                            ys.append(grid.nodes[idx].y)
                    if xs:
                        return sum(xs) / len(xs), sum(ys) / len(ys)
        except Exception:
            pass
    return None, None


@router.get("/messages")
def get_messages(
    severity: str | None = Query(default=None, description="Filter by severity (WARN/FATAL/INFO)"),
    limit: int = Query(default=100, ge=1, le=10000, description="Max messages to return"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
) -> dict[str, Any]:
    """Parse and return simulation messages with optional severity filtering."""
    result = _get_diagnostics_result()
    if result is None:
        return {
            "messages": [],
            "total": 0,
            "warning_count": 0,
            "error_count": 0,
            "total_runtime_seconds": None,
        }

    messages = result.messages

    # Filter by severity if requested
    if severity is not None:
        from pyiwfm.io.simulation.messages import MessageSeverity

        try:
            sev = MessageSeverity[severity.upper()]
        except KeyError:
            # Unknown severity — return empty
            return {
                "messages": [],
                "total": 0,
                "warning_count": result.warning_count,
                "error_count": result.error_count,
                "total_runtime_seconds": _runtime_seconds(result),
            }
        messages = result.filter_by_severity(sev)

    total = len(messages)
    page = messages[offset : offset + limit]

    serialized = [
        {
            "severity": m.severity.name,
            "text": m.text,
            "procedure": m.procedure,
            "line_number": m.line_number,
            "node_ids": m.node_ids,
            "element_ids": m.element_ids,
            "reach_ids": m.reach_ids,
            "layer_ids": m.layer_ids,
        }
        for m in page
    ]

    return {
        "messages": serialized,
        "total": total,
        "warning_count": result.warning_count,
        "error_count": result.error_count,
        "total_runtime_seconds": _runtime_seconds(result),
    }


@router.get("/convergence")
def get_convergence() -> dict[str, Any]:
    """Return convergence iteration data from simulation messages."""
    result = _get_diagnostics_result()

    # Convergence records require the extended dataclasses
    if result is None or not _HAS_CONVERGENCE:
        return {
            "records": [],
            "max_iterations": 0,
            "avg_iterations": 0.0,
            "total_timesteps": 0,
        }

    # Use get_convergence_summary if available on the result
    convergence_summary: dict[str, Any] = {}
    if hasattr(result, "get_convergence_summary"):
        convergence_summary = result.get_convergence_summary()

    records: list[dict[str, Any]] = []
    if hasattr(result, "convergence_records"):
        for rec in result.convergence_records:
            records.append(
                {
                    "timestep_index": rec.timestep_index,
                    "date": rec.date,
                    "iteration_count": rec.iteration_count,
                    "max_residual": rec.max_residual,
                    "convergence_achieved": rec.convergence_achieved,
                    "bottleneck_variable": getattr(rec, "bottleneck_variable", ""),
                    "supply_adj_count": getattr(rec, "supply_adj_count", 0),
                }
            )

    iteration_counts = [r["iteration_count"] for r in records]
    max_iter = convergence_summary.get(
        "max_iterations", max(iteration_counts) if iteration_counts else 0
    )
    avg_iter = convergence_summary.get(
        "avg_iterations",
        sum(iteration_counts) / len(iteration_counts) if iteration_counts else 0.0,
    )
    total_ts = convergence_summary.get("total_timesteps", len(records))

    return {
        "records": records,
        "max_iterations": max_iter,
        "avg_iterations": avg_iter,
        "total_timesteps": total_ts,
    }


@router.get("/convergence-hotspots")
def get_convergence_hotspots() -> list[dict[str, Any]]:
    """Return convergence hotspot variables with spatial coordinates."""
    result = _get_diagnostics_result()

    if result is None or not hasattr(result, "get_hotspots"):
        return []

    hotspots = result.get_hotspots()
    serialized: list[dict[str, Any]] = []
    for h in hotspots:
        x, y = _resolve_hotspot_coords(h.entity_type, h.entity_id)
        avg_iter = h.total_iterations / h.occurrence_count if h.occurrence_count > 0 else 0.0
        serialized.append(
            {
                "variable": h.variable,
                "entity_type": h.entity_type,
                "entity_id": h.entity_id,
                "layer": h.layer,
                "occurrence_count": h.occurrence_count,
                "total_iterations": h.total_iterations,
                "avg_iterations": round(avg_iter, 1),
                "worst_timestep_index": h.worst_timestep_index,
                "worst_timestep_date": h.worst_timestep_date,
                "x": x,
                "y": y,
            }
        )
    return serialized


def _get_element_geometry(grid: Any, elem_ids: set[int]) -> list[dict[str, Any]]:
    """Build vertex geometry dicts for a set of element IDs."""
    result: list[dict[str, Any]] = []
    for eid in sorted(elem_ids):
        if eid not in grid.elements:
            continue
        elem = grid.elements[eid]
        verts = [
            [grid.nodes[nid].x, grid.nodes[nid].y] for nid in elem.vertices if nid in grid.nodes
        ]
        if verts:
            result.append({"id": eid, "vertices": verts})
    return result


def _get_2nd_ring_elements(grid: Any, first_ring_ids: set[int]) -> set[int]:
    """Get elements in the 2nd ring (share a node with 1st ring elements)."""
    ring2_nodes: set[int] = set()
    for eid in first_ring_ids:
        if eid in grid.elements:
            ring2_nodes.update(grid.elements[eid].vertices)
    ring2: set[int] = set()
    for nid in ring2_nodes:
        if nid in grid.nodes:
            ring2.update(getattr(grid.nodes[nid], "surrounding_elements", []))
    return ring2 - first_ring_ids


def _get_stream_reach_coords(
    model: Any, stream_node_id: int | None = None, gw_node_ids: set[int] | None = None
) -> tuple[list[list[float]], int | None, str | None]:
    """Get stream reach coordinates.

    If *stream_node_id* is given, returns the reach containing that node.
    If *gw_node_ids* is given, finds any reach whose nodes reference those GW nodes.
    Returns ``(coords_list, reach_id, reach_name)``.
    """
    if not hasattr(model, "streams") or model.streams is None:
        return [], None, None

    streams = model.streams
    grid = model.mesh
    target_reach_id: int | None = None

    if stream_node_id is not None:
        try:
            snode = streams.get_node(stream_node_id)
            if snode is not None:
                target_reach_id = getattr(snode, "reach_id", None)
        except Exception:
            pass
    elif gw_node_ids:
        # Find a reach that has a stream node referencing one of these GW nodes
        try:
            for sn in streams.nodes.values():
                gw = getattr(sn, "gw_node", None)
                if gw is not None and gw in gw_node_ids:
                    target_reach_id = getattr(sn, "reach_id", None)
                    break
        except Exception:
            pass

    if target_reach_id is None:
        return [], None, None

    # Get reach name
    reach_name: str | None = None
    try:
        reach = streams.get_reach(target_reach_id)
        if reach is not None:
            reach_name = getattr(reach, "name", None)
    except Exception:
        pass

    # Collect all stream nodes in the reach
    coords: list[list[float]] = []
    try:
        for sn in streams.nodes.values():
            if getattr(sn, "reach_id", None) != target_reach_id:
                continue
            sx, sy = sn.x, sn.y
            if (sx == 0 and sy == 0) and getattr(sn, "gw_node", None):
                gw_id = sn.gw_node
                if grid and gw_id in grid.nodes:
                    sx, sy = grid.nodes[gw_id].x, grid.nodes[gw_id].y
            if sx != 0 or sy != 0:
                coords.append([sx, sy])
    except Exception:
        pass

    return coords, target_reach_id, reach_name


@router.get("/hotspot-context")
def get_hotspot_context(
    variable: str = Query(description="Convergence variable, e.g. GW_25393_(L1) or ST_2620"),
    timestep_index: int | None = Query(default=None, description="Center timestep for context"),
) -> dict[str, Any]:
    """Return spatial context and iteration history for a convergence hotspot."""
    from pyiwfm.io.simulation.messages import _parse_variable_id

    entity_type, entity_id, layer = _parse_variable_id(variable)
    x, y = _resolve_hotspot_coords(entity_type, entity_id)

    surrounding_element_ids: list[int] = []
    surrounding_elements_geometry: list[dict[str, Any]] = []
    context_elements_geometry: list[dict[str, Any]] = []
    stream_reach_coords: list[list[float]] = []
    stream_reach_id: int | None = None
    stream_reach_name: str | None = None

    model = model_state._model
    if model is not None and hasattr(model, "mesh") and model.mesh is not None:
        grid = model.mesh

        if entity_type == "groundwater" and entity_id in grid.nodes:
            node = grid.nodes[entity_id]
            ring1 = set(getattr(node, "surrounding_elements", []))
            surrounding_element_ids = sorted(ring1)
            surrounding_elements_geometry = _get_element_geometry(grid, ring1)

            # 2nd ring for wider context
            ring2 = _get_2nd_ring_elements(grid, ring1)
            context_elements_geometry = _get_element_geometry(grid, ring2)

            # Nearby stream reach for reference
            ring1_nodes: set[int] = set()
            for eid in ring1:
                if eid in grid.elements:
                    ring1_nodes.update(grid.elements[eid].vertices)
            stream_reach_coords, stream_reach_id, stream_reach_name = _get_stream_reach_coords(
                model, gw_node_ids=ring1_nodes
            )

        elif entity_type == "stream":
            # Get stream reach geometry
            stream_reach_coords, stream_reach_id, stream_reach_name = _get_stream_reach_coords(
                model, stream_node_id=entity_id
            )

            # Get surrounding elements via gw_node for context (non-clickable)
            try:
                if hasattr(model, "streams") and model.streams is not None:
                    snode = model.streams.get_node(entity_id)
                    if snode is not None and getattr(snode, "gw_node", None):
                        gw_id = snode.gw_node
                        if gw_id in grid.nodes:
                            gw_node = grid.nodes[gw_id]
                            ring1 = set(getattr(gw_node, "surrounding_elements", []))
                            surrounding_element_ids = sorted(ring1)
                            surrounding_elements_geometry = _get_element_geometry(grid, ring1)
                            ring2 = _get_2nd_ring_elements(grid, ring1)
                            context_elements_geometry = _get_element_geometry(grid, ring2)
            except Exception:
                pass

    # Iteration history: all records where this variable was the bottleneck
    iteration_history: list[dict[str, Any]] = []
    result = _get_diagnostics_result()
    if result is not None:
        for rec in result.convergence_records:
            if getattr(rec, "bottleneck_variable", "") == variable:
                iteration_history.append(
                    {
                        "timestep_index": rec.timestep_index,
                        "date": rec.date,
                        "iteration_count": rec.iteration_count,
                    }
                )

    return {
        "variable": variable,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "layer": layer,
        "x": x,
        "y": y,
        "surrounding_element_ids": surrounding_element_ids,
        "surrounding_elements_geometry": surrounding_elements_geometry,
        "context_elements_geometry": context_elements_geometry,
        "stream_reach_coords": stream_reach_coords,
        "stream_reach_id": stream_reach_id,
        "stream_reach_name": stream_reach_name,
        "iteration_history": iteration_history,
    }


@router.get("/element-budget")
def get_element_budget(
    element_id: int = Query(description="Element ID to query"),
    timestep_index: int | None = Query(default=None, description="Center timestep for ±5 window"),
    layer: int = Query(default=1, ge=1, description="Layer number (1-based)"),
) -> dict[str, Any]:
    """Fetch GW ZBudget data for a single element."""
    reader = model_state.get_zbudget_reader("gw")
    if reader is None:
        return {"element_id": element_id, "times": [], "columns": []}

    try:
        from pyiwfm.io.budget.zone_reader import ZoneInfo as ZBZoneInfo

        # Inject a temporary single-element zone
        zone_name = f"_diag_elem_{element_id}"
        grid = model_state._model.mesh if model_state._model else None
        area = grid.elements[element_id].area if grid and element_id in grid.elements else 0.0
        reader._zone_info[zone_name] = ZBZoneInfo(
            id=99999,
            name=zone_name,
            n_elements=1,
            element_ids=[element_id],
            area=area,
            adjacent_zones=[],
        )

        df = reader.get_dataframe(zone_name, layer=layer)

        # Clean up temp zone
        reader._zone_info.pop(zone_name, None)

        # Extract ±5 timestep window if timestep_index given
        if timestep_index is not None and len(df) > 0:
            # Find the row closest to the timestep_index
            # Convergence records are 1-based; df rows are 0-based
            center = max(0, min(timestep_index - 1, len(df) - 1))
            start = max(0, center - 5)
            end = min(len(df), center + 6)
            df = df.iloc[start:end]

        times = [t.isoformat() if hasattr(t, "isoformat") else str(t) for t in df.index]
        columns = [{"name": col, "values": df[col].tolist()} for col in df.columns]

        return {"element_id": element_id, "times": times, "columns": columns}
    except Exception:
        logger.exception("Failed to get element budget for element %d", element_id)
        # Clean up temp zone on error
        try:
            reader._zone_info.pop(f"_diag_elem_{element_id}", None)
        except Exception:
            pass
        return {"element_id": element_id, "times": [], "columns": []}


@router.get("/stream-node-budget")
def get_stream_node_budget(
    stream_node_id: int = Query(description="Stream node ID"),
    timestep_index: int | None = Query(default=None, description="Center timestep for ±5 window"),
) -> dict[str, Any]:
    """Fetch stream node budget data for a specific stream node."""
    reader = model_state.get_budget_reader("stream_node")
    if reader is None:
        return {"stream_node_id": stream_node_id, "times": [], "columns": []}

    try:
        # Find the location matching this stream node ID
        loc_idx: int | None = None
        for i, loc_name in enumerate(reader.locations):
            # Location names may be like "1", "2620", or "Node 2620"
            # Try to match the numeric ID
            try:
                if int(loc_name.split()[-1]) == stream_node_id:
                    loc_idx = i
                    break
            except (ValueError, IndexError):
                pass
            if loc_name == str(stream_node_id):
                loc_idx = i
                break

        if loc_idx is None:
            return {"stream_node_id": stream_node_id, "times": [], "columns": []}

        headers = reader.get_column_headers(loc_idx)
        times_arr, values_arr = reader.get_values(loc_idx)

        # Ensure 2D
        if values_arr.ndim == 1:
            values_arr = values_arr.reshape(-1, 1)

        # Extract ±5 window
        if timestep_index is not None and len(times_arr) > 0:
            center = max(0, min(timestep_index - 1, len(times_arr) - 1))
            start = max(0, center - 5)
            end = min(len(times_arr), center + 6)
            times_arr = times_arr[start:end]
            values_arr = values_arr[start:end]

        # Convert times to ISO strings
        import datetime

        times: list[str] = []
        for t in times_arr:
            try:
                dt = datetime.datetime.fromtimestamp(float(t))
                times.append(dt.isoformat())
            except (ValueError, OSError):
                times.append(str(t))

        columns = [
            {
                "name": headers[i] if i < len(headers) else f"Col {i}",
                "values": values_arr[:, i].tolist(),
            }
            for i in range(values_arr.shape[1])
        ]

        return {"stream_node_id": stream_node_id, "times": times, "columns": columns}
    except Exception:
        logger.exception("Failed to get stream node budget for node %d", stream_node_id)
        return {"stream_node_id": stream_node_id, "times": [], "columns": []}


@router.get("/mass-balance")
def get_mass_balance(
    component: str | None = Query(default=None, description="Filter by component name"),
) -> dict[str, Any]:
    """Return mass balance error timeseries from simulation messages."""
    result = _get_diagnostics_result()

    if result is None or not _HAS_MASS_BALANCE:
        return {"records": [], "components": []}

    records: list[dict[str, Any]] = []
    components: set[str] = set()

    if hasattr(result, "mass_balance_records"):
        for rec in result.mass_balance_records:
            components.add(rec.component)
            if component is not None and rec.component != component:
                continue
            records.append(
                {
                    "timestep_index": rec.timestep_index,
                    "date": rec.date,
                    "component": rec.component,
                    "error_value": rec.error_value,
                    "error_percent": rec.error_percent,
                }
            )

    return {
        "records": records,
        "components": sorted(components),
    }


@router.get("/summary")
def get_summary() -> dict[str, Any]:
    """Combined diagnostics summary."""
    result = _get_diagnostics_result()
    has_diag = result is not None

    if not has_diag:
        return {
            "has_diagnostics": False,
            "message_count": 0,
            "warning_count": 0,
            "error_count": 0,
            "total_runtime_seconds": None,
            "max_iterations": 0,
            "avg_iterations": 0.0,
            "total_timesteps": 0,
            "spatial_summary": {"nodes": {}, "elements": {}, "reaches": {}},
        }

    assert result is not None  # for type narrowing

    # Convergence stats
    convergence_data = get_convergence()

    # Spatial summary
    spatial = result.get_spatial_summary()

    return {
        "has_diagnostics": True,
        "message_count": len(result.messages),
        "warning_count": result.warning_count,
        "error_count": result.error_count,
        "total_runtime_seconds": _runtime_seconds(result),
        "max_iterations": convergence_data["max_iterations"],
        "avg_iterations": convergence_data["avg_iterations"],
        "total_timesteps": convergence_data["total_timesteps"],
        "spatial_summary": {
            "nodes": {str(k): v for k, v in spatial["nodes"].items()},
            "elements": {str(k): v for k, v in spatial["elements"].items()},
            "reaches": {str(k): v for k, v in spatial["reaches"].items()},
        },
    }


@router.get("/spatial-summary")
def get_spatial_summary() -> dict[str, dict[str, int]]:
    """Spatial summary of message counts for map overlay."""
    result = _get_diagnostics_result()

    if result is None:
        return {"nodes": {}, "elements": {}, "reaches": {}}

    spatial = result.get_spatial_summary()

    # JSON keys must be strings
    return {
        "nodes": {str(k): v for k, v in spatial["nodes"].items()},
        "elements": {str(k): v for k, v in spatial["elements"].items()},
        "reaches": {str(k): v for k, v in spatial["reaches"].items()},
    }
