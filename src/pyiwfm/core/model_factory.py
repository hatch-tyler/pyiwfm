"""
Factory helper functions for constructing IWFMModel instances.

This module contains the model-building helper functions extracted from
model.py to keep the IWFMModel class focused on data and properties.
The classmethods on IWFMModel delegate to these functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyiwfm.components.groundwater import AppGW, AquiferParameters
    from pyiwfm.components.stream import AppStream
    from pyiwfm.core.mesh import AppGrid
    from pyiwfm.core.model import IWFMModel
    from pyiwfm.io.groundwater import KhAnomalyEntry
    from pyiwfm.io.preprocessor_binary import PreprocessorBinaryData


def build_reaches_from_node_reach_ids(stream: AppStream) -> None:
    """Build StrmReach objects by grouping stream nodes by their reach_id.

    Only populates ``stream.reaches`` when it is currently empty and at
    least some nodes carry a non-zero ``reach_id``.

    Parameters
    ----------
    stream : AppStream
        Stream network whose ``nodes`` are inspected.
    """
    from collections import defaultdict

    from pyiwfm.components.stream import StrmReach

    if stream.reaches:
        return  # already populated

    by_reach: dict[int, list[int]] = defaultdict(list)
    for node in stream.nodes.values():
        if getattr(node, "reach_id", 0) > 0:
            by_reach[node.reach_id].append(node.id)

    for rid, node_ids in sorted(by_reach.items()):
        node_ids.sort()
        stream.add_reach(
            StrmReach(
                id=rid,
                upstream_node=node_ids[0],
                downstream_node=node_ids[-1],
                nodes=node_ids,
            )
        )


def apply_kh_anomalies(
    params: AquiferParameters,
    anomalies: list[KhAnomalyEntry],
    mesh: AppGrid,
) -> int:
    """Apply Kh anomaly overwrites from element-level data to node arrays.

    For each anomaly element, sets Kh at all vertex nodes to the
    anomaly value.  This matches IWFM Fortran behavior in
    ``ReadAquiferParameters`` (``Class_AppGW.f90:4433–4442``).

    Parameters
    ----------
    params : AquiferParameters
        Aquifer parameters whose ``kh`` array will be modified in-place.
    anomalies : list[KhAnomalyEntry]
        Parsed anomaly entries from the GW main file.
    mesh : AppGrid
        Model mesh providing element-to-node connectivity.

    Returns
    -------
    int
        Number of anomalies successfully applied.
    """
    if params.kh is None:
        return 0

    # Build node_id -> 0-based index lookup
    node_id_to_idx = {nid: i for i, nid in enumerate(sorted(mesh.nodes.keys()))}

    applied = 0
    for entry in anomalies:
        elem = mesh.elements.get(entry.element_id)
        if elem is None:
            continue
        for node_id in elem.vertices:
            idx = node_id_to_idx.get(node_id)
            if idx is None:
                continue
            n_layers = min(len(entry.kh_per_layer), params.n_layers)
            for layer in range(n_layers):
                params.kh[idx, layer] = entry.kh_per_layer[layer]
        applied += 1
    return applied


def apply_parametric_grids(
    gw: AppGW,
    parametric_grids: list,
    mesh: AppGrid,
) -> bool:
    """Interpolate aquifer parameters from parametric grids onto model nodes.

    Builds a :class:`~pyiwfm.io.parametric_grid.ParametricGrid` for each
    group, interpolates all 5 parameters (Kh, Ss, Sy, AquitardKv, Kv)
    at every model node, and assigns the result as aquifer parameters.

    Returns ``True`` if interpolation was performed.
    """
    import numpy as np

    from pyiwfm.components.groundwater import AquiferParameters
    from pyiwfm.io.parametric_grid import ParamElement, ParametricGrid, ParamNode

    node_ids = sorted(mesh.nodes.keys())
    n_nodes = len(node_ids)
    if n_nodes == 0:
        return False

    node_coords = [(mesh.nodes[nid].x, mesh.nodes[nid].y) for nid in node_ids]

    # Determine n_layers from first grid
    first_grid = parametric_grids[0]
    n_layers = first_grid.node_values.shape[1]

    kh = np.zeros((n_nodes, n_layers), dtype=np.float64)
    ss = np.zeros((n_nodes, n_layers), dtype=np.float64)
    sy = np.zeros((n_nodes, n_layers), dtype=np.float64)
    aquitard_kv = np.zeros((n_nodes, n_layers), dtype=np.float64)
    kv = np.zeros((n_nodes, n_layers), dtype=np.float64)

    for grid_data in parametric_grids:
        # Special case: single node with no elements means "uniform everywhere"
        if grid_data.n_nodes == 1 and grid_data.n_elements == 0:
            uniform_vals = grid_data.node_values[0]  # shape (n_layers, 5)
            for i in range(n_nodes):
                for layer in range(min(n_layers, uniform_vals.shape[0])):
                    if uniform_vals[layer, 0] >= 0:
                        kh[i, layer] = uniform_vals[layer, 0]
                    if uniform_vals[layer, 1] >= 0:
                        ss[i, layer] = uniform_vals[layer, 1]
                    if uniform_vals[layer, 2] >= 0:
                        sy[i, layer] = uniform_vals[layer, 2]
                    if uniform_vals[layer, 3] >= 0:
                        aquitard_kv[i, layer] = uniform_vals[layer, 3]
                    if uniform_vals[layer, 4] >= 0:
                        kv[i, layer] = uniform_vals[layer, 4]
            continue

        # Build ParametricGrid from raw data
        pnodes = []
        for j in range(grid_data.n_nodes):
            pnodes.append(
                ParamNode(
                    node_id=j,
                    x=grid_data.node_coords[j, 0],
                    y=grid_data.node_coords[j, 1],
                    values=grid_data.node_values[j],
                )
            )
        pelems = []
        for j, verts in enumerate(grid_data.elements):
            pelems.append(ParamElement(elem_id=j, vertices=verts))

        pgrid = ParametricGrid(nodes=pnodes, elements=pelems)

        for i, (x, y) in enumerate(node_coords):
            result = pgrid.interpolate(x, y)
            if result is None:
                continue
            for layer in range(min(n_layers, result.shape[0])):
                if result[layer, 0] >= 0:
                    kh[i, layer] = result[layer, 0]
                if result[layer, 1] >= 0:
                    ss[i, layer] = result[layer, 1]
                if result[layer, 2] >= 0:
                    sy[i, layer] = result[layer, 2]
                if result[layer, 3] >= 0:
                    aquitard_kv[i, layer] = result[layer, 3]
                if result[layer, 4] >= 0:
                    kv[i, layer] = result[layer, 4]

    params = AquiferParameters(
        n_nodes=n_nodes,
        n_layers=n_layers,
        kh=kh,
        kv=kv,
        specific_storage=ss,
        specific_yield=sy,
        aquitard_kv=aquitard_kv,
    )
    try:
        gw.set_aquifer_parameters(params)
    except ValueError:
        gw.aquifer_params = params
    return True


def apply_parametric_subsidence(
    subs_config: Any,
    mesh: AppGrid,
    n_nodes: int,
    n_layers: int,
) -> list:
    """Interpolate subsidence parameters from parametric grids onto model nodes.

    Returns a list of SubsidenceNodeParams objects.
    """
    import numpy as np

    from pyiwfm.io.gw_subsidence import SubsidenceNodeParams
    from pyiwfm.io.parametric_grid import ParamElement, ParametricGrid, ParamNode

    node_ids = sorted(mesh.nodes.keys())
    node_coords = [(mesh.nodes[nid].x, mesh.nodes[nid].y) for nid in node_ids]

    # Initialize arrays: shape (n_nodes, n_layers)
    elastic_sc = np.zeros((n_nodes, n_layers), dtype=np.float64)
    inelastic_sc = np.zeros((n_nodes, n_layers), dtype=np.float64)
    interbed_thick = np.zeros((n_nodes, n_layers), dtype=np.float64)
    interbed_thick_min = np.zeros((n_nodes, n_layers), dtype=np.float64)
    precompact_head = np.zeros((n_nodes, n_layers), dtype=np.float64)

    for grid_data in subs_config.parametric_grids:
        # Special case: single node with no elements means "uniform everywhere"
        if grid_data.n_nodes == 1 and grid_data.n_elements == 0:
            uniform_vals = grid_data.node_values[0]  # shape (n_layers, 5)
            for i in range(n_nodes):
                for layer in range(min(n_layers, uniform_vals.shape[0])):
                    if uniform_vals[layer, 0] >= 0:
                        elastic_sc[i, layer] = uniform_vals[layer, 0]
                    if uniform_vals[layer, 1] >= 0:
                        inelastic_sc[i, layer] = uniform_vals[layer, 1]
                    if uniform_vals[layer, 2] >= 0:
                        interbed_thick[i, layer] = uniform_vals[layer, 2]
                    if uniform_vals[layer, 3] >= 0:
                        interbed_thick_min[i, layer] = uniform_vals[layer, 3]
                    if uniform_vals[layer, 4] >= 0:
                        precompact_head[i, layer] = uniform_vals[layer, 4]
            continue

        pnodes = []
        for j in range(grid_data.n_nodes):
            pnodes.append(
                ParamNode(
                    node_id=j,
                    x=grid_data.node_coords[j, 0],
                    y=grid_data.node_coords[j, 1],
                    values=grid_data.node_values[j],
                )
            )
        pelems = []
        for j, verts in enumerate(grid_data.elements):
            pelems.append(ParamElement(elem_id=j, vertices=verts))

        pgrid = ParametricGrid(nodes=pnodes, elements=pelems)

        for i, (x, y) in enumerate(node_coords):
            result = pgrid.interpolate(x, y)
            if result is None:
                continue
            for layer in range(min(n_layers, result.shape[0])):
                if result[layer, 0] >= 0:
                    elastic_sc[i, layer] = result[layer, 0]
                if result[layer, 1] >= 0:
                    inelastic_sc[i, layer] = result[layer, 1]
                if result[layer, 2] >= 0:
                    interbed_thick[i, layer] = result[layer, 2]
                if result[layer, 3] >= 0:
                    interbed_thick_min[i, layer] = result[layer, 3]
                if result[layer, 4] >= 0:
                    precompact_head[i, layer] = result[layer, 4]

    # Build SubsidenceNodeParams list
    node_params = []
    for i, nid in enumerate(node_ids):
        node_params.append(
            SubsidenceNodeParams(
                node_id=nid,
                elastic_sc=elastic_sc[i].tolist(),
                inelastic_sc=inelastic_sc[i].tolist(),
                interbed_thick=interbed_thick[i].tolist(),
                interbed_thick_min=interbed_thick_min[i].tolist(),
                precompact_head=precompact_head[i].tolist(),
            )
        )
    return node_params


def binary_data_to_model(
    data: PreprocessorBinaryData,
    name: str = "",
) -> IWFMModel:
    """Convert :class:`PreprocessorBinaryData` to an :class:`IWFMModel`.

    Builds Node, Element, Subregion, Stratigraphy, AppStream, and AppLake
    objects from the raw arrays in *data*.
    """
    from pyiwfm.core.mesh import AppGrid, Element, Node, Subregion
    from pyiwfm.core.model import IWFMModel
    from pyiwfm.core.stratigraphy import Stratigraphy

    # -- Mesh: Nodes -------------------------------------------------------
    nodes: dict[int, Node] = {}
    for i in range(data.n_nodes):
        nid = i + 1  # 1-based
        nodes[nid] = Node(id=nid, x=float(data.x[i]), y=float(data.y[i]))

    # -- Mesh: Elements ----------------------------------------------------
    max_nv = int(data.n_vertex.max()) if data.n_elements > 0 else 4
    vertex_2d = data.vertex.reshape((max_nv, data.n_elements), order="F").T

    elements: dict[int, Element] = {}
    for i in range(data.n_elements):
        eid = i + 1
        nv = int(data.n_vertex[i])
        verts = tuple(int(vertex_2d[i, j]) for j in range(nv) if vertex_2d[i, j] != 0)
        sub_id = int(data.app_elements[i].subregion) if i < len(data.app_elements) else 0
        elements[eid] = Element(id=eid, vertices=verts, subregion=sub_id)

    # -- Mesh: Subregions --------------------------------------------------
    subregions: dict[int, Subregion] = {}
    for sd in data.subregions:
        subregions[sd.id] = Subregion(id=sd.id, name=sd.name)

    mesh = AppGrid(nodes=nodes, elements=elements, subregions=subregions)
    mesh.compute_areas()
    mesh.compute_connectivity()

    # -- Stratigraphy ------------------------------------------------------
    strat: Stratigraphy | None = None
    if data.stratigraphy is not None:
        sd2 = data.stratigraphy
        strat = Stratigraphy(
            n_layers=sd2.n_layers,
            n_nodes=data.n_nodes,
            gs_elev=sd2.ground_surface_elev,
            top_elev=sd2.top_elev,
            bottom_elev=sd2.bottom_elev,
            active_node=sd2.active_node,
        )

    model = IWFMModel(
        name=name,
        mesh=mesh,
        stratigraphy=strat,
        metadata={"source": "preprocessor_binary"},
    )

    # -- Streams -----------------------------------------------------------
    if (
        data.streams is not None
        and data.streams.n_reaches > 0
        and data.stream_gw_connector is not None
    ):
        from pyiwfm.components.stream import AppStream, StrmNode, StrmReach

        stream = AppStream()
        gw_conn = data.stream_gw_connector
        for sn_id in range(1, data.streams.n_stream_nodes + 1):
            gw_node: int | None = None
            if sn_id <= gw_conn.n_stream_nodes:
                gw_node = int(gw_conn.gw_nodes[sn_id - 1])
                if gw_node <= 0:
                    gw_node = None
            stream.add_node(StrmNode(id=sn_id, x=0.0, y=0.0, gw_node=gw_node))

        sd3 = data.streams
        for i in range(sd3.n_reaches):
            rid = int(sd3.reach_ids[i])
            rname = sd3.reach_names[i] if i < len(sd3.reach_names) else ""
            up = int(sd3.reach_upstream_nodes[i])
            dn = int(sd3.reach_downstream_nodes[i])
            reach_nodes = list(range(up, dn + 1))
            stream.add_reach(
                StrmReach(
                    id=rid,
                    upstream_node=up,
                    downstream_node=dn,
                    nodes=reach_nodes,
                    name=rname,
                )
            )
            for sn_id in reach_nodes:
                if sn_id in stream.nodes:
                    stream.nodes[sn_id].reach_id = rid

        model.streams = stream

    # -- Lakes -------------------------------------------------------------
    if data.lakes is not None and data.lakes.n_lakes > 0:
        from pyiwfm.components.lake import AppLake, Lake, LakeElement

        lakes = AppLake()
        ld = data.lakes
        for i in range(ld.n_lakes):
            lid = int(ld.lake_ids[i])
            elem_ids = [int(e) for e in ld.lake_elements[i]] if i < len(ld.lake_elements) else []
            lake = Lake(
                id=lid,
                name=ld.lake_names[i] if i < len(ld.lake_names) else f"Lake {lid}",
                max_elevation=float(ld.lake_max_elevations[i]),
                elements=elem_ids,
            )
            lakes.add_lake(lake)
            for eid in elem_ids:
                lakes.add_lake_element(LakeElement(element_id=eid, lake_id=lid))
        model.lakes = lakes

    return model


def resolve_stream_node_coordinates(model: IWFMModel) -> int:
    """Resolve stream node (0,0) coordinates from associated GW nodes.

    Many IWFM loaders create stream nodes with placeholder ``(0, 0)``
    coordinates.  When the stream node has a ``gw_node`` reference we can
    look up the real coordinates from the mesh.

    Returns the number of nodes updated.
    """
    if model.mesh is None or model.streams is None:
        return 0
    resolved = 0
    for node in model.streams.nodes.values():
        if node.x == 0.0 and node.y == 0.0 and node.gw_node is not None:
            gw = model.mesh.nodes.get(node.gw_node)
            if gw is not None:
                node.x = gw.x
                node.y = gw.y
                resolved += 1
    return resolved
