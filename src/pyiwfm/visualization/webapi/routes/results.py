"""
Results data API routes: heads, hydrographs, locations.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from pyiwfm.visualization.webapi.config import model_state
from pyiwfm.visualization.webapi.utils import sanitize_values as _sanitize_values

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/results", tags=["results"])



@router.get("/info")
def get_results_info() -> dict:
    """Get summary of available results data."""
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")
    return model_state.get_results_info()


@router.get("/heads")
def get_heads(
    timestep: int = Query(default=0, ge=0, description="Timestep index"),
    layer: int = Query(default=1, ge=1, description="Layer number (1-based)"),
) -> dict:
    """Get head values for all nodes at a given timestep and layer."""
    loader = model_state.get_head_loader()
    if loader is None:
        raise HTTPException(status_code=404, detail="No head data available")

    if timestep >= loader.n_frames:
        raise HTTPException(
            status_code=400,
            detail=f"Timestep {timestep} out of range [0, {loader.n_frames})",
        )

    frame = loader.get_frame(timestep)
    # frame shape: (n_nodes, n_layers)
    layer_idx = layer - 1
    if layer_idx >= frame.shape[1]:
        raise HTTPException(
            status_code=400,
            detail=f"Layer {layer} out of range [1, {frame.shape[1]}]",
        )

    values = _sanitize_values(frame[:, layer_idx].tolist())
    dt = loader.times[timestep] if timestep < len(loader.times) else None

    return {
        "timestep_index": timestep,
        "datetime": dt.isoformat() if dt else None,
        "layer": layer,
        "values": values,
    }


@router.get("/head-diff")
def get_head_diff(
    timestep_a: int = Query(default=0, ge=0, description="First timestep index"),
    timestep_b: int = Query(default=0, ge=0, description="Second timestep index"),
    layer: int = Query(default=1, ge=1, description="Layer number (1-based)"),
) -> dict:
    """
    Compute head difference between two timesteps.

    Returns per-node head difference (timestep_b - timestep_a).
    Positive values indicate head rise, negative values indicate drawdown.
    """
    loader = model_state.get_head_loader()
    if loader is None:
        raise HTTPException(status_code=404, detail="No head data available")

    for ts, label in [(timestep_a, "timestep_a"), (timestep_b, "timestep_b")]:
        if ts >= loader.n_frames:
            raise HTTPException(
                status_code=400,
                detail=f"{label}={ts} out of range [0, {loader.n_frames})",
            )

    frame_a = loader.get_frame(timestep_a)
    frame_b = loader.get_frame(timestep_b)

    layer_idx = layer - 1
    if layer_idx >= frame_a.shape[1]:
        raise HTTPException(
            status_code=400,
            detail=f"Layer {layer} out of range [1, {frame_a.shape[1]}]",
        )

    vals_a = frame_a[:, layer_idx]
    vals_b = frame_b[:, layer_idx]

    import numpy as np

    diff = vals_b - vals_a

    # Replace extreme values (dry cells) with NaN
    mask = (vals_a < -9000) | (vals_b < -9000)
    diff_list = [None if mask[i] else round(float(diff[i]), 3) for i in range(len(diff))]

    valid = diff[~mask]
    vmin = float(np.min(valid)) if len(valid) > 0 else 0.0
    vmax = float(np.max(valid)) if len(valid) > 0 else 0.0

    dt_a = loader.times[timestep_a] if timestep_a < len(loader.times) else None
    dt_b = loader.times[timestep_b] if timestep_b < len(loader.times) else None

    return {
        "timestep_a": timestep_a,
        "timestep_b": timestep_b,
        "datetime_a": dt_a.isoformat() if dt_a else None,
        "datetime_b": dt_b.isoformat() if dt_b else None,
        "layer": layer,
        "values": diff_list,
        "min": round(vmin, 3),
        "max": round(vmax, 3),
    }


@router.get("/head-times")
def get_head_times() -> dict:
    """Get list of all available head timestep datetimes."""
    loader = model_state.get_head_loader()
    if loader is None:
        raise HTTPException(status_code=404, detail="No head data available")

    return {
        "times": [t.isoformat() for t in loader.times],
        "n_timesteps": loader.n_frames,
    }


@router.get("/head-range")
def get_head_range(
    layer: int = Query(default=1, ge=1, description="Layer number (1-based)"),
    max_frames: int = Query(default=50, ge=0, description="Max frames to sample (0=all)"),
) -> dict:
    """Get the global head value range across all timesteps for a layer.

    Returns 2nd–98th percentile range for stable color scale rendering.
    Uses SQLite cache when available for instant response.
    """
    # Try cache first
    cached = model_state.get_cached_head_range(layer)
    if cached is not None:
        loader = model_state.get_head_loader()
        n_frames = loader.n_frames if loader else 0
        return {
            "layer": layer,
            "min": cached["percentile_02"],
            "max": cached["percentile_98"],
            "n_timesteps": n_frames,
            "n_frames_scanned": n_frames,
        }

    loader = model_state.get_head_loader()
    if loader is None:
        raise HTTPException(status_code=404, detail="No head data available")

    lo, hi, n_scanned = loader.get_layer_range(layer=layer, max_frames=max_frames)

    return {
        "layer": layer,
        "min": lo,
        "max": hi,
        "n_timesteps": loader.n_frames,
        "n_frames_scanned": n_scanned,
    }


@router.get("/hydrograph-locations")
def get_hydrograph_locations() -> dict:
    """Get all hydrograph locations with WGS84 coordinates."""
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")
    return model_state.get_hydrograph_locations()


@router.get("/hydrograph")
def get_hydrograph(
    type: str = Query(description="Type: gw, stream, or subsidence"),
    location_id: int = Query(description="Location/node ID"),
) -> dict:
    """Get hydrograph time series for a specific location.

    For GW: location_id is the 1-based hydrograph index (column in output file).
    For stream: location_id is the stream node ID.
    """
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    model = model_state.model

    if type == "gw":
        # location_id is a 1-based *physical* location index.  When the
        # model has a groundwater component we use the grouped physical
        # location mapping; otherwise fall back to raw column indexing.
        phys_locs = model_state.get_gw_physical_locations()
        location_index = location_id - 1

        if phys_locs:
            # --- grouped physical location path ---
            if location_index < 0 or location_index >= len(phys_locs):
                raise HTTPException(
                    status_code=404,
                    detail=f"GW hydrograph {location_id} out of range "
                    f"[1, {len(phys_locs)}]",
                )

            group = phys_locs[location_index]
            name = group["name"] or f"GW Hydrograph {location_id}"
            columns = group["columns"]  # list of (col_idx, layer)
            layer = columns[0][1] if columns else 1
            node_id = group["node_id"]

            # Try SQLite cache first (fastest path).
            cache = model_state.get_cache_loader()
            if cache is not None and columns:
                col_idx = columns[0][0]
                cached = cache.get_hydrograph("gw", col_idx)
                if cached is not None:
                    times_list, values_arr = cached
                    return {
                        "location_id": location_id,
                        "name": name,
                        "type": "gw",
                        "layer": layer,
                        "times": times_list,
                        "values": _sanitize_values(values_arr.tolist()),
                        "units": "ft",
                    }

            # Prefer GW hydrograph output file
            reader = model_state.get_gw_hydrograph_reader()
            if reader is not None and reader.n_timesteps > 0 and columns:
                col_idx = columns[0][0]
                if 0 <= col_idx < reader.n_columns:
                    times_raw, values_raw = reader.get_time_series(col_idx)
                    return {
                        "location_id": location_id,
                        "name": name,
                        "type": "gw",
                        "layer": layer,
                        "times": times_raw,
                        "values": _sanitize_values(values_raw),
                        "units": "ft",
                    }

            # Fall back to HEAD HDF5 data (node-based extraction)
            loader = model_state.get_head_loader()
            if loader is not None and loader.n_frames > 0 and node_id > 0:
                node_id_to_idx = model_state.get_node_id_to_idx()
                node_idx = node_id_to_idx.get(node_id)
                if node_idx is not None:
                    layer_idx = layer - 1
                    n_layers = loader.get_frame(0).shape[1]
                    if 0 <= layer_idx < n_layers:
                        times_iso = [t.isoformat() for t in loader.times]
                        values: list[float | None] = []
                        for ts in range(loader.n_frames):
                            frame = loader.get_frame(ts)
                            v = float(frame[node_idx, layer_idx])
                            values.append(None if v < -9000 else round(v, 3))
                        return {
                            "location_id": location_id,
                            "name": name,
                            "type": "gw",
                            "layer": layer,
                            "times": times_iso,
                            "values": _sanitize_values(values),
                            "units": "ft",
                        }

            raise HTTPException(
                status_code=404, detail="No GW hydrograph data available"
            )

        # --- fallback: no physical location mapping (no GW component) ---
        # Treat location_id - 1 as raw column index.
        name = f"GW Hydrograph {location_id}"
        layer = 1
        if model and model.groundwater:
            locs = model.groundwater.hydrograph_locations
            if 0 <= location_index < len(locs):
                loc = locs[location_index]
                name = getattr(loc, "name", None) or name
                layer = getattr(loc, "layer", 1) or 1

        reader = model_state.get_gw_hydrograph_reader()
        if reader is not None and reader.n_timesteps > 0:
            col = location_index
            if col < 0 or col >= reader.n_columns:
                raise HTTPException(
                    status_code=404,
                    detail=f"GW hydrograph {location_id} out of range "
                    f"[1, {reader.n_columns}]",
                )
            times_raw, values_raw = reader.get_time_series(col)
            return {
                "location_id": location_id,
                "name": name,
                "type": "gw",
                "layer": layer,
                "times": times_raw,
                "values": _sanitize_values(values_raw),
                "units": "ft",
            }

        raise HTTPException(status_code=404, detail="No GW hydrograph data available")

    elif type == "stream":
        reader = model_state.get_stream_hydrograph_reader()
        if reader is None or reader.n_timesteps == 0:
            raise HTTPException(status_code=404, detail="No stream hydrograph data available")

        # location_id is the stream node ID — find its column
        col_idx = reader.find_column_by_node_id(location_id)
        if col_idx is None:
            # Also try matching by hydrograph_ids
            if location_id in reader.hydrograph_ids:
                col_idx = reader.hydrograph_ids.index(location_id)

        if col_idx is None:
            raise HTTPException(
                status_code=404,
                detail=f"Stream node {location_id} not found in hydrograph data",
            )

        # Get location name from stream specs
        name = f"Stream Node {location_id}"
        stream_specs = model.metadata.get("stream_hydrograph_specs", []) if model else []
        for spec in stream_specs:
            if spec["node_id"] == location_id:
                name = spec.get("name", name)
                break

        # Stream files may have flow + stage columns (interleaved or sequential)
        # Check hydrograph_output_type from metadata
        output_type = model.metadata.get("stream_hydrograph_output_type", 0) if model else 0
        n_specs = len(stream_specs) if stream_specs else 0

        times, flow_values = reader.get_time_series(col_idx)

        result: dict = {
            "location_id": location_id,
            "name": name,
            "type": "stream",
            "times": times,
            "values": _sanitize_values(flow_values),
            "units": "cfs",
        }

        # If output_type includes stage (1=stage, 2=both) and there are
        # paired columns (flow columns followed by stage columns)
        if output_type == 2 and n_specs > 0:
            stage_col_idx = col_idx + n_specs
            if stage_col_idx < reader.n_columns:
                _, stage_values = reader.get_time_series(stage_col_idx)
                result["flow_values"] = _sanitize_values(flow_values)
                result["stage_values"] = _sanitize_values(stage_values)
                result["flow_units"] = "cfs"
                result["stage_units"] = "ft"

        return result

    elif type == "subsidence":
        reader = model_state.get_subsidence_reader()
        if reader is None or reader.n_timesteps == 0:
            raise HTTPException(
                status_code=404,
                detail="No subsidence hydrograph data available",
            )

        # location_id matches the 1-based hydrograph spec index
        # Try column index first, then node ID lookup
        subs_col_idx: int | None = None

        # Check subsidence hydrograph specs for matching ID
        subs_config = None
        if model and model.groundwater:
            subs_config = getattr(model.groundwater, "subsidence_config", None)
        specs = getattr(subs_config, "hydrograph_specs", []) if subs_config else []

        # Match by spec ID (1-based)
        for i, spec in enumerate(specs):
            if spec.id == location_id:
                subs_col_idx = i
                break

        # Fallback: try as 1-based column index
        if subs_col_idx is None:
            candidate = location_id - 1
            if 0 <= candidate < reader.n_columns:
                subs_col_idx = candidate

        if subs_col_idx is None:
            raise HTTPException(
                status_code=404,
                detail=f"Subsidence location {location_id} not found in hydrograph data",
            )

        times, subs_values = reader.get_time_series(subs_col_idx)

        name = f"Subsidence Obs {location_id}"
        layer = 1
        if subs_col_idx < len(specs):
            spec = specs[subs_col_idx]
            name = spec.name or name
            layer = spec.layer

        return {
            "location_id": location_id,
            "name": name,
            "type": "subsidence",
            "layer": layer,
            "times": times,
            "values": _sanitize_values(subs_values),
            "units": "ft",
        }

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown hydrograph type: {type}. Use: gw, stream, subsidence",
        )


@router.get("/gw-hydrograph-all-layers")
def get_gw_hydrograph_all_layers(
    location_id: int = Query(description="1-based GW hydrograph location ID"),
) -> dict:
    """Get head time series at a GW hydrograph location for ALL layers.

    Prefers the IWFM GW hydrograph output file (which stores IWFM-computed
    values at the actual observation coordinates for all layers).  Falls
    back to head HDF5 node extraction when no hydrograph file is available.
    """
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    model = model_state.model
    phys_locs = model_state.get_gw_physical_locations()
    location_index = location_id - 1

    # Resolve location metadata from physical location mapping or raw model
    name = f"GW Hydrograph {location_id}"
    node_id = 0
    columns: list[tuple[int, int]] = []  # (col_idx, layer)

    if phys_locs:
        if location_index < 0 or location_index >= len(phys_locs):
            raise HTTPException(
                status_code=404,
                detail=f"GW hydrograph {location_id} out of range "
                f"[1, {len(phys_locs)}]",
            )
        group = phys_locs[location_index]
        name = group["name"] or name
        node_id = group["node_id"]
        columns = group["columns"]
    elif model and model.groundwater:
        locs = model.groundwater.hydrograph_locations
        if location_index < 0 or location_index >= len(locs):
            raise HTTPException(
                status_code=404,
                detail=f"GW hydrograph {location_id} out of range "
                f"[1, {len(locs)}]",
            )
        loc = locs[location_index]
        name = getattr(loc, "name", None) or name
        node_id = getattr(loc, "node_id", 0) or getattr(loc, "gw_node", 0)
        try:
            node_id = int(node_id) if node_id else 0
        except (TypeError, ValueError):
            node_id = 0
    else:
        if model is None or model.groundwater is None:
            raise HTTPException(
                status_code=404, detail="No groundwater component"
            )

    # ------------------------------------------------------------------
    # Path 0 (fastest): SQLite cache — pre-extracted from hydrograph file
    # ------------------------------------------------------------------
    cache = model_state.get_cache_loader()
    if cache is not None and columns:
        cached_layers: list[tuple] = []
        times_list: list[str] = []
        for col_idx, layer in columns:
            result = cache.get_hydrograph("gw", col_idx)
            if result is not None:
                t, vals = result
                if not times_list:
                    times_list = t
                cached_layers.append((layer, vals))
        if cached_layers:
            return {
                "location_id": location_id,
                "node_id": node_id,
                "name": name,
                "n_layers": len(cached_layers),
                "times": times_list,
                "layers": [
                    {"layer": layer, "values": _sanitize_values(vals.tolist())}
                    for layer, vals in cached_layers
                ],
            }

    # ------------------------------------------------------------------
    # Path 1 (preferred): GW hydrograph output file
    # ------------------------------------------------------------------
    reader = model_state.get_gw_hydrograph_reader()
    if reader is not None and reader.n_timesteps > 0 and columns:
        all_valid = all(0 <= ci < reader.n_columns for ci, _ in columns)
        if all_valid:
            times_raw: list[str] = []
            layers_data: list[dict] = []
            for col_idx, layer in columns:
                times_raw, values_raw = reader.get_time_series(col_idx)
                layers_data.append(
                    {"layer": layer, "values": _sanitize_values(values_raw)}
                )
            return {
                "location_id": location_id,
                "node_id": node_id,
                "name": name,
                "n_layers": len(layers_data),
                "times": times_raw,
                "layers": layers_data,
            }

    # ------------------------------------------------------------------
    # Path 2 (fallback): head HDF5 data at nearest grid node
    # ------------------------------------------------------------------
    if node_id == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No hydrograph data or node ID for location {location_id}",
        )

    loader = model_state.get_head_loader()
    if loader is None or loader.n_frames == 0:
        raise HTTPException(status_code=404, detail="No head data available")

    node_id_to_idx = model_state.get_node_id_to_idx()
    node_idx = node_id_to_idx.get(node_id)
    if node_idx is None:
        raise HTTPException(
            status_code=404,
            detail=f"Node {node_id} not found in grid",
        )

    n_layers = loader.get_frame(0).shape[1]
    times_iso = [t.isoformat() for t in loader.times]

    layer_values: list[list[float | None]] = [[] for _ in range(n_layers)]
    for ts in range(loader.n_frames):
        frame = loader.get_frame(ts)
        for layer_idx in range(n_layers):
            v = float(frame[node_idx, layer_idx])
            if v < -9000:
                layer_values[layer_idx].append(None)
            else:
                layer_values[layer_idx].append(round(v, 3))

    layers_data_fb: list[dict] = []
    for layer_idx in range(n_layers):
        layers_data_fb.append(
            {
                "layer": layer_idx + 1,
                "values": _sanitize_values(layer_values[layer_idx]),
            }
        )

    return {
        "location_id": location_id,
        "node_id": node_id,
        "name": name,
        "n_layers": n_layers,
        "times": times_iso,
        "layers": layers_data_fb,
    }


@router.get("/subsidence-all-layers")
def get_subsidence_all_layers(
    location_id: int = Query(description="Subsidence hydrograph location ID"),
) -> dict:
    """Get subsidence time series at a location for ALL layers.

    Groups subsidence specs by physical location (matching node_id) and
    returns one timeseries per layer.
    """
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    model = model_state.model
    if model is None or model.groundwater is None:
        raise HTTPException(status_code=404, detail="No groundwater component")

    subs_config = getattr(model.groundwater, "subsidence_config", None)
    specs = getattr(subs_config, "hydrograph_specs", []) if subs_config else []
    if not specs:
        raise HTTPException(status_code=404, detail="No subsidence hydrograph specs")

    reader = model_state.get_subsidence_reader()
    if reader is None or reader.n_timesteps == 0:
        raise HTTPException(status_code=404, detail="No subsidence hydrograph data")

    # Find the target spec
    target_spec = None
    target_col_idx: int | None = None
    for i, spec in enumerate(specs):
        if spec.id == location_id:
            target_spec = spec
            target_col_idx = i
            break

    if target_spec is None:
        # Fallback: treat location_id as 1-based index
        candidate = location_id - 1
        if 0 <= candidate < len(specs):
            target_spec = specs[candidate]
            target_col_idx = candidate

    if target_spec is None or target_col_idx is None:
        raise HTTPException(
            status_code=404,
            detail=f"Subsidence location {location_id} not found",
        )

    # Find all specs at the same physical location (same node_id or matching x,y)
    target_node_id = getattr(target_spec, "node_id", 0) or getattr(
        target_spec, "gw_node", 0
    )
    target_x = getattr(target_spec, "x", 0.0)
    target_y = getattr(target_spec, "y", 0.0)

    name = target_spec.name or f"Subsidence Obs {location_id}"

    matched_specs: list[tuple[int, int, object]] = []  # (col_idx, layer, spec)
    for i, spec in enumerate(specs):
        sn_id = getattr(spec, "node_id", 0) or getattr(spec, "gw_node", 0)
        sx = getattr(spec, "x", 0.0)
        sy = getattr(spec, "y", 0.0)

        same_loc = False
        if target_node_id > 0 and sn_id == target_node_id:
            same_loc = True
        elif target_x != 0.0 and abs(sx - target_x) < 1.0 and abs(sy - target_y) < 1.0:
            same_loc = True

        if same_loc and i < reader.n_columns:
            matched_specs.append((i, spec.layer, spec))

    if not matched_specs:
        # At minimum, include the target spec itself
        if target_col_idx < reader.n_columns:
            matched_specs.append(
                (target_col_idx, target_spec.layer, target_spec)
            )

    # Sort by layer
    matched_specs.sort(key=lambda t: t[1])

    # Extract time series for each matched layer
    layers_data: list[dict] = []
    times_raw: list[str] = []
    for col_idx, layer_num, _spec in matched_specs:
        times, values = reader.get_time_series(col_idx)
        if not times_raw:
            times_raw = times
        layers_data.append(
            {
                "layer": layer_num,
                "values": _sanitize_values(values),
            }
        )

    return {
        "location_id": location_id,
        "node_id": target_node_id,
        "name": name,
        "n_layers": len(layers_data),
        "times": times_raw,
        "layers": layers_data,
    }


@router.get("/hydrographs-multi")
def get_hydrographs_multi(
    type: str = Query(description="Type: gw or stream"),
    ids: str = Query(description="Comma-separated location IDs"),
) -> dict:
    """
    Get multiple hydrograph time series for comparison overlay.

    Returns a list of hydrograph data objects, one per requested ID.
    """
    if not model_state.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")

    try:
        id_list = [int(x.strip()) for x in ids.split(",") if x.strip()]
    except ValueError as e:
        raise HTTPException(
            status_code=400, detail="Invalid IDs format. Use comma-separated integers."
        ) from e

    if not id_list:
        raise HTTPException(status_code=400, detail="No IDs provided")

    model = model_state.model
    results: list[dict] = []

    if type == "gw":
        phys_locs = model_state.get_gw_physical_locations()
        cache = model_state.get_cache_loader()
        reader = model_state.get_gw_hydrograph_reader()
        loader = model_state.get_head_loader()
        node_id_to_idx = model_state.get_node_id_to_idx() if loader else {}

        for loc_id in id_list:
            location_index = loc_id - 1

            if phys_locs:
                # --- grouped path ---
                if location_index < 0 or location_index >= len(phys_locs):
                    continue
                group = phys_locs[location_index]
                name = group["name"] or f"GW Hydrograph {loc_id}"
                columns = group["columns"]
                node_id = group["node_id"]
                layer = columns[0][1] if columns else 1

                if cache is not None and columns:
                    col_idx = columns[0][0]
                    result = cache.get_hydrograph("gw", col_idx)
                    if result is not None:
                        times_list, values_arr = result
                        results.append(
                            {
                                "location_id": loc_id,
                                "name": name,
                                "type": "gw",
                                "layer": layer,
                                "times": times_list,
                                "values": _sanitize_values(values_arr.tolist()),
                                "units": "ft",
                            }
                        )
                        continue

                if reader is not None and reader.n_timesteps > 0 and columns:
                    col_idx = columns[0][0]
                    if 0 <= col_idx < reader.n_columns:
                        times_raw, values_raw = reader.get_time_series(col_idx)
                        results.append(
                            {
                                "location_id": loc_id,
                                "name": name,
                                "type": "gw",
                                "layer": layer,
                                "times": times_raw,
                                "values": _sanitize_values(values_raw),
                                "units": "ft",
                            }
                        )
                        continue

                if loader is not None and loader.n_frames > 0 and node_id > 0:
                    node_idx = node_id_to_idx.get(node_id)
                    if node_idx is not None:
                        layer_idx = layer - 1
                        n_layers = loader.get_frame(0).shape[1]
                        if 0 <= layer_idx < n_layers:
                            times_iso = [t.isoformat() for t in loader.times]
                            values: list[float | None] = []
                            for ts in range(loader.n_frames):
                                frame = loader.get_frame(ts)
                                v = float(frame[node_idx, layer_idx])
                                values.append(None if v < -9000 else round(v, 3))
                            results.append(
                                {
                                    "location_id": loc_id,
                                    "name": name,
                                    "type": "gw",
                                    "layer": layer,
                                    "times": times_iso,
                                    "values": _sanitize_values(values),
                                    "units": "ft",
                                }
                            )
            else:
                # --- fallback: raw column index ---
                name = f"GW Hydrograph {loc_id}"
                layer = 1
                if model and model.groundwater:
                    locs = model.groundwater.hydrograph_locations
                    if 0 <= location_index < len(locs):
                        loc = locs[location_index]
                        name = getattr(loc, "name", None) or name
                        layer = getattr(loc, "layer", 1) or 1

                if reader is not None and reader.n_timesteps > 0:
                    col = location_index
                    if 0 <= col < reader.n_columns:
                        times_raw, values_raw = reader.get_time_series(col)
                        results.append(
                            {
                                "location_id": loc_id,
                                "name": name,
                                "type": "gw",
                                "layer": layer,
                                "times": times_raw,
                                "values": _sanitize_values(values_raw),
                                "units": "ft",
                            }
                        )

        if not results and (reader is None or reader.n_timesteps == 0):
            raise HTTPException(status_code=404, detail="No GW hydrograph data available")

    elif type == "stream":
        reader = model_state.get_stream_hydrograph_reader()
        if reader is None or reader.n_timesteps == 0:
            raise HTTPException(
                status_code=404,
                detail="No stream hydrograph data available",
            )

        for loc_id in id_list:
            col_idx = reader.find_column_by_node_id(loc_id)
            if col_idx is None and loc_id in reader.hydrograph_ids:
                col_idx = reader.hydrograph_ids.index(loc_id)
            if col_idx is None:
                continue

            stream_times, stream_values = reader.get_time_series(col_idx)

            name = f"Stream Node {loc_id}"
            stream_specs = model.metadata.get("stream_hydrograph_specs", []) if model else []
            for spec in stream_specs:
                if spec["node_id"] == loc_id:
                    name = spec.get("name", name)
                    break

            results.append(
                {
                    "location_id": loc_id,
                    "name": name,
                    "type": "stream",
                    "times": stream_times,
                    "values": _sanitize_values(stream_values),
                    "units": "cfs",
                }
            )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown type: {type}. Use: gw, stream",
        )

    return {"type": type, "n_series": len(results), "series": results}


@router.get("/drawdown")
def get_drawdown(
    layer: int = Query(default=1, ge=1, description="Layer number (1-based)"),
    reference_timestep: int = Query(
        default=0, ge=0, description="Reference timestep (default: first)"
    ),
) -> dict:
    """
    Get drawdown (head change relative to reference timestep) for all timesteps.

    Returns per-timestep arrays of drawdown values (negative = decline).
    Used for drawdown animation.
    """
    loader = model_state.get_head_loader()
    if loader is None:
        raise HTTPException(status_code=404, detail="No head data available")

    if reference_timestep >= loader.n_frames:
        raise HTTPException(
            status_code=400,
            detail=f"Reference timestep {reference_timestep} out of range",
        )

    import numpy as np

    layer_idx = layer - 1
    ref_frame = loader.get_frame(reference_timestep)
    if layer_idx >= ref_frame.shape[1]:
        raise HTTPException(
            status_code=400,
            detail=f"Layer {layer} out of range [1, {ref_frame.shape[1]}]",
        )

    ref_values = ref_frame[:, layer_idx]

    # Compute drawdown for all timesteps
    timesteps: list[dict] = []
    for ts in range(loader.n_frames):
        frame = loader.get_frame(ts)
        vals = frame[:, layer_idx]
        diff = vals - ref_values

        # Mask dry cells
        mask = (ref_values < -9000) | (vals < -9000)
        diff_list = [None if mask[i] else round(float(diff[i]), 2) for i in range(len(diff))]

        valid = diff[~mask]
        vmin = float(np.min(valid)) if len(valid) > 0 else 0.0
        vmax = float(np.max(valid)) if len(valid) > 0 else 0.0

        dt = loader.times[ts] if ts < len(loader.times) else None

        timesteps.append(
            {
                "timestep": ts,
                "datetime": dt.isoformat() if dt else None,
                "values": diff_list,
                "min": round(vmin, 2),
                "max": round(vmax, 2),
            }
        )

    return {
        "layer": layer,
        "reference_timestep": reference_timestep,
        "n_timesteps": len(timesteps),
        "timesteps": timesteps,
    }


@router.get("/heads-by-element")
def get_heads_by_element(
    timestep: int = Query(default=0, ge=0, description="Timestep index"),
    layer: int = Query(default=1, ge=1, description="Layer number (1-based)"),
) -> dict:
    """Get per-element head values (vertex-averaged) for a timestep and layer.

    Returns a lightweight array of head values indexed by element position
    (matching the order of features in the mesh GeoJSON).  Each value is the
    average of the head at the element's vertex nodes.

    Uses SQLite cache when available for ~5-10x faster response.
    """
    # Try cache first
    cached = model_state.get_cached_head_by_element(timestep, layer)
    if cached is not None:
        values, min_val, max_val = cached
        loader = model_state.get_head_loader()
        dt = None
        if loader and timestep < len(loader.times):
            dt = loader.times[timestep]
        return {
            "timestep_index": timestep,
            "datetime": dt.isoformat() if dt else None,
            "layer": layer,
            "values": values,
            "min": round(min_val, 3),
            "max": round(max_val, 3),
        }

    loader = model_state.get_head_loader()
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

    model = model_state.model
    if model is None or model.grid is None:
        raise HTTPException(status_code=404, detail="No model grid available")

    grid = model.grid

    # Use cached mappings
    node_id_to_idx = model_state.get_node_id_to_idx()
    sorted_elem_ids = model_state.get_sorted_elem_ids()
    elem_heads: list[float | None] = []
    for eid in sorted_elem_ids:
        elem = grid.elements[eid]
        node_vals: list[float] = []
        for nid in elem.vertices:
            idx = node_id_to_idx.get(nid)
            if idx is not None and idx < len(head_values):
                v = float(head_values[idx])
                if v > -9000:
                    node_vals.append(v)
        if node_vals:
            avg = sum(node_vals) / len(node_vals)
            elem_heads.append(round(avg, 3))
        else:
            elem_heads.append(None)

    # Compute min/max excluding dry cells
    valid = [v for v in elem_heads if v is not None]
    if valid:
        valid.sort()
        lo = valid[max(0, int(len(valid) * 0.02))]
        hi = valid[min(len(valid) - 1, int(len(valid) * 0.98))]
    else:
        lo, hi = 0.0, 1.0

    dt = loader.times[timestep] if timestep < len(loader.times) else None

    return {
        "timestep_index": timestep,
        "datetime": dt.isoformat() if dt else None,
        "layer": layer,
        "values": elem_heads,
        "min": round(lo, 3),
        "max": round(hi, 3),
    }
