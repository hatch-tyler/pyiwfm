"""
Plotting functionality for IWFM models.

This module provides matplotlib-based visualization tools for
IWFM model meshes, scalar fields, stream networks, time series,
and water budget data.

Mesh Plotting Functions
-----------------------
- :func:`plot_mesh`: Plot finite element mesh with elements
- :func:`plot_nodes`: Plot mesh nodes as scatter points
- :func:`plot_elements`: Plot elements with coloring options
- :func:`plot_scalar_field`: Plot scalar values on mesh
- :func:`plot_streams`: Plot stream network
- :func:`plot_lakes`: Plot lake elements on mesh
- :func:`plot_boundary`: Plot model boundary

Time Series Plotting Functions
------------------------------
- :func:`plot_timeseries`: Single or multiple time series line chart
- :func:`plot_timeseries_comparison`: Observed vs simulated comparison
- :func:`plot_timeseries_collection`: Plot multiple locations

Budget Plotting Functions
-------------------------
- :func:`plot_budget_bar`: Budget components as bar chart
- :func:`plot_budget_stacked`: Stacked area chart of budget over time
- :func:`plot_budget_pie`: Budget components as pie chart
- :func:`plot_water_balance`: Water balance summary chart
- :func:`plot_zbudget`: Zone budget visualization

Additional Plotting Functions
-----------------------------
- :func:`plot_streams_colored`: Color stream reaches by scalar values
- :func:`plot_timeseries_statistics`: Ensemble mean with min/max or std-dev bands
- :func:`plot_dual_axis`: Dual y-axis comparison of two time series
- :func:`plot_streamflow_hydrograph`: Streamflow hydrograph with baseflow separation

Example
-------
Plot a time series of groundwater heads:

>>> from pyiwfm.visualization.plotting import plot_timeseries
>>> from pyiwfm.core.timeseries import TimeSeries
>>> import numpy as np
>>>
>>> times = np.array(['2020-01-01', '2020-02-01', '2020-03-01'], dtype='datetime64')
>>> values = np.array([100.0, 98.5, 99.2])
>>> ts = TimeSeries(times=times, values=values, name='Well_1', units='ft')
>>> fig, ax = plot_timeseries(ts, title='Groundwater Head')
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

matplotlib.use("Agg")  # Non-interactive backend for saving

if TYPE_CHECKING:
    from pyiwfm.components.lake import AppLake
    from pyiwfm.components.stream import AppStream
    from pyiwfm.core.cross_section import CrossSection
    from pyiwfm.core.mesh import AppGrid
    from pyiwfm.core.timeseries import TimeSeries, TimeSeriesCollection


def _apply_tick_formatting(ax: Axes) -> None:
    """Apply thousands-separator formatting to both axes."""
    from matplotlib.ticker import FuncFormatter

    def _thousands_fmt(x: float, pos: int) -> str:
        return f"{x:,.0f}"

    ax.xaxis.set_major_formatter(FuncFormatter(_thousands_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(_thousands_fmt))
    ax.tick_params(axis="both", labelsize=9)


def plot_mesh(
    grid: AppGrid,
    ax: Axes | None = None,
    show_edges: bool = True,
    show_node_ids: bool = False,
    show_element_ids: bool = False,
    edge_color: str = "black",
    edge_width: float = 0.5,
    fill_color: str = "lightblue",
    alpha: float = 0.3,
    figsize: tuple[float, float] = (10, 8),
) -> tuple[Figure, Axes]:
    """
    Plot the mesh with elements and optional annotations.

    Args:
        grid: Model mesh
        ax: Existing axes to plot on (creates new if None)
        show_edges: Show element edges
        show_node_ids: Label nodes with their IDs
        show_element_ids: Label elements with their IDs
        edge_color: Color for element edges
        edge_width: Width of edge lines
        fill_color: Fill color for elements
        alpha: Transparency of element fill
        figsize: Figure size in inches

    Returns:
        Tuple of (Figure, Axes)
    """

    from matplotlib.collections import PolyCollection

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    # Build element polygons
    polygons = []
    for elem in grid.iter_elements():
        coords = []
        for vid in elem.vertices:
            node = grid.nodes[vid]
            coords.append((node.x, node.y))
        polygons.append(coords)

    # Create polygon collection
    collection = PolyCollection(
        polygons,
        edgecolors=edge_color if show_edges else "none",
        facecolors=fill_color,
        linewidths=edge_width,
        alpha=alpha,
    )
    ax.add_collection(collection)

    # Add node labels
    if show_node_ids:
        for node in grid.iter_nodes():
            ax.annotate(
                str(node.id),
                (node.x, node.y),
                fontsize=8,
                ha="center",
                va="center",
            )

    # Add element labels
    if show_element_ids:
        for elem in grid.iter_elements():
            # Calculate centroid
            x_coords = [grid.nodes[vid].x for vid in elem.vertices]
            y_coords = [grid.nodes[vid].y for vid in elem.vertices]
            cx = sum(x_coords) / len(x_coords)
            cy = sum(y_coords) / len(y_coords)
            ax.annotate(
                str(elem.id),
                (cx, cy),
                fontsize=8,
                ha="center",
                va="center",
                color="red",
            )

    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xlabel("X", fontsize=11)
    ax.set_ylabel("Y", fontsize=11)
    _apply_tick_formatting(ax)
    fig.tight_layout()

    return fig, ax


def plot_nodes(
    grid: AppGrid,
    ax: Axes | None = None,
    highlight_boundary: bool = False,
    marker_size: float = 20,
    color: str = "blue",
    boundary_color: str = "red",
    figsize: tuple[float, float] = (10, 8),
) -> tuple[Figure, Axes]:
    """
    Plot mesh nodes as points.

    Args:
        grid: Model mesh
        ax: Existing axes to plot on
        highlight_boundary: Use different color for boundary nodes
        marker_size: Size of node markers
        color: Color for interior nodes
        boundary_color: Color for boundary nodes
        figsize: Figure size in inches

    Returns:
        Tuple of (Figure, Axes)
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    # Collect node coordinates
    interior_x, interior_y = [], []
    boundary_x, boundary_y = [], []

    for node in grid.iter_nodes():
        if highlight_boundary and node.is_boundary:
            boundary_x.append(node.x)
            boundary_y.append(node.y)
        else:
            interior_x.append(node.x)
            interior_y.append(node.y)

    # Plot nodes
    if interior_x:
        ax.scatter(interior_x, interior_y, s=marker_size, c=color, label="Interior")

    if highlight_boundary and boundary_x:
        ax.scatter(boundary_x, boundary_y, s=marker_size, c=boundary_color, label="Boundary")
        ax.legend(framealpha=0.9, edgecolor="lightgray")

    ax.set_aspect("equal")
    ax.set_xlabel("X", fontsize=11)
    ax.set_ylabel("Y", fontsize=11)
    _apply_tick_formatting(ax)
    fig.tight_layout()

    return fig, ax


def plot_elements(
    grid: AppGrid,
    ax: Axes | None = None,
    color_by: Literal["subregion", "area", "none"] = "none",
    cmap: str = "viridis",
    show_colorbar: bool = True,
    edge_color: str = "black",
    edge_width: float = 0.5,
    alpha: float = 0.7,
    figsize: tuple[float, float] = (10, 8),
) -> tuple[Figure, Axes]:
    """
    Plot mesh elements with optional coloring by attribute.

    Args:
        grid: Model mesh
        ax: Existing axes to plot on
        color_by: Attribute to color elements by
        cmap: Colormap name
        show_colorbar: Show colorbar for colored plots
        edge_color: Color for element edges
        edge_width: Width of edge lines
        alpha: Transparency of element fill
        figsize: Figure size in inches

    Returns:
        Tuple of (Figure, Axes)
    """

    import matplotlib.colors as mcolors
    from matplotlib.collections import PolyCollection

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    # Build element polygons and get color values
    polygons = []
    values_list: list[float] = []

    for elem in grid.iter_elements():
        coords = []
        for vid in elem.vertices:
            node = grid.nodes[vid]
            coords.append((node.x, node.y))
        polygons.append(coords)

        if color_by == "subregion":
            values_list.append(float(elem.subregion))
        elif color_by == "area":
            values_list.append(elem.area)
        else:
            values_list.append(0.0)

    # Create polygon collection
    if color_by == "subregion":
        # Discrete coloring with legend for categorical subregion values
        from matplotlib.patches import Patch

        values = np.array(values_list)
        unique_vals = np.unique(values)
        colormap = plt.get_cmap(cmap)
        n_unique = max(len(unique_vals), 1)
        val_to_color = {v: colormap(i / max(n_unique - 1, 1)) for i, v in enumerate(unique_vals)}
        face_colors = [val_to_color[v] for v in values]

        collection = PolyCollection(
            polygons,
            facecolors=face_colors,
            edgecolors=edge_color,
            linewidths=edge_width,
            alpha=alpha,
        )

        if show_colorbar:
            legend_patches = [
                Patch(
                    facecolor=val_to_color[v],
                    edgecolor=edge_color,
                    alpha=alpha,
                    label=f"Subregion {int(v)}",
                )
                for v in unique_vals
            ]
            ax.legend(handles=legend_patches, loc="best", framealpha=0.9, edgecolor="lightgray")

    elif color_by != "none":
        values = np.array(values_list)
        norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())
        colormap = plt.get_cmap(cmap)

        collection = PolyCollection(
            polygons,
            array=values,
            cmap=colormap,
            norm=norm,
            edgecolors=edge_color,
            linewidths=edge_width,
            alpha=alpha,
        )

        if show_colorbar:
            cbar = fig.colorbar(collection, ax=ax)
            cbar.set_label(color_by.capitalize(), fontsize=10)
    else:
        collection = PolyCollection(
            polygons,
            facecolors="lightblue",
            edgecolors=edge_color,
            linewidths=edge_width,
            alpha=alpha,
        )

    ax.add_collection(collection)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xlabel("X", fontsize=11)
    ax.set_ylabel("Y", fontsize=11)
    _apply_tick_formatting(ax)
    fig.tight_layout()

    return fig, ax


def _subdivide_quads(
    elem_conn: list[list[int]],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    values: NDArray[np.float64],
    n: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """
    Subdivide quad elements using bilinear FE shape functions.

    Each quad is subdivided into an n x n grid of points using bilinear
    interpolation, then triangulated. Triangle elements pass through
    unchanged. Fully vectorized — no Python loop over elements.

    Parameters
    ----------
    elem_conn : list of list of int
        Element connectivity (each inner list has 3 or 4 vertex indices).
    x, y : ndarray
        Node coordinates indexed by the vertex indices in elem_conn.
    values : ndarray
        Scalar values at each node.
    n : int
        Subdivision level (n x n points per quad, must be >= 2).

    Returns
    -------
    sub_x, sub_y, sub_values : ndarray
        Coordinates and values at the subdivided points.
    sub_triangles : ndarray of shape (n_tri, 3)
        Triangle connectivity into the sub_x/sub_y arrays.
    """
    # Separate triangles from quads
    tri_conn = [v for v in elem_conn if len(v) == 3]
    quad_conn = [v for v in elem_conn if len(v) == 4]

    all_x: list[NDArray[np.float64]] = []
    all_y: list[NDArray[np.float64]] = []
    all_v: list[NDArray[np.float64]] = []
    all_tri: list[NDArray[np.int64]] = []
    offset = 0

    # --- Process quads ---
    if quad_conn:
        quad_arr = np.array(quad_conn)  # (n_quads, 4)
        n_quads = quad_arr.shape[0]

        # Precompute reference grid and bilinear shape functions
        xi_1d = np.linspace(-1, 1, n)
        xi, eta = np.meshgrid(xi_1d, xi_1d)
        xi_flat = xi.ravel()
        eta_flat = eta.ravel()
        # Shape function matrix: (n*n, 4)
        shape_funcs = 0.25 * np.column_stack(
            [
                (1 - xi_flat) * (1 - eta_flat),
                (1 + xi_flat) * (1 - eta_flat),
                (1 + xi_flat) * (1 + eta_flat),
                (1 - xi_flat) * (1 + eta_flat),
            ]
        )

        # Precompute triangle template for the n x n structured grid
        row, col = np.mgrid[: n - 1, : n - 1]
        i0 = (row * n + col).ravel()
        i1 = i0 + 1
        i2 = i0 + n
        i3 = i2 + 1
        tri_template = np.column_stack([i0, i1, i3, i0, i3, i2]).reshape(-1, 3)

        # Batch map all quads via matrix multiply
        vx_all = x[quad_arr]  # (n_quads, 4)
        vy_all = y[quad_arr]
        vv_all = values[quad_arr]

        sub_qx = (shape_funcs @ vx_all.T).T  # (n_quads, n*n)
        sub_qy = (shape_funcs @ vy_all.T).T
        sub_qv = (shape_funcs @ vv_all.T).T

        # Build triangle indices with vectorized offsets
        n_pts = n * n
        offsets = np.arange(n_quads, dtype=np.int64) * n_pts + offset
        quad_tris = offsets[:, None, None] + tri_template[None, :, :]

        all_x.append(sub_qx.ravel())
        all_y.append(sub_qy.ravel())
        all_v.append(sub_qv.ravel())
        all_tri.append(quad_tris.reshape(-1, 3))
        offset += n_quads * n_pts

    # --- Process triangles (pass through) ---
    if tri_conn:
        tri_arr = np.array(tri_conn)  # (n_tris, 3)
        all_x.append(x[tri_arr].ravel())
        all_y.append(y[tri_arr].ravel())
        all_v.append(values[tri_arr].ravel())

        n_tris = tri_arr.shape[0]
        tri_indices = np.arange(n_tris * 3, dtype=np.int64).reshape(n_tris, 3) + offset
        all_tri.append(tri_indices)

    return (
        np.concatenate(all_x),
        np.concatenate(all_y),
        np.concatenate(all_v),
        np.concatenate(all_tri).astype(np.int64),
    )


def plot_scalar_field(
    grid: AppGrid,
    values: NDArray[np.float64],
    field_type: Literal["node", "cell"] = "node",
    ax: Axes | None = None,
    cmap: str = "viridis",
    show_colorbar: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    show_mesh: bool = True,
    edge_color: str = "gray",
    edge_width: float = 0.3,
    n_subdiv: int = 4,
    figsize: tuple[float, float] = (10, 8),
) -> tuple[Figure, Axes]:
    """
    Plot scalar field values on the mesh.

    Args:
        grid: Model mesh
        values: Scalar values (one per node or cell)
        field_type: 'node' for node values, 'cell' for cell values
        ax: Existing axes to plot on
        cmap: Colormap name
        show_colorbar: Show colorbar
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        show_mesh: Show mesh edges
        edge_color: Color for mesh edges
        edge_width: Width of mesh edges
        n_subdiv: Subdivision level for bilinear quad interpolation (>=2 enables
            FE subdivision; 1 uses legacy diagonal-split triangulation)
        figsize: Figure size in inches

    Returns:
        Tuple of (Figure, Axes)
    """

    import matplotlib.colors as mcolors
    from matplotlib.collections import PolyCollection
    from matplotlib.tri import Triangulation

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    if vmin is None:
        vmin = values.min()
    if vmax is None:
        vmax = values.max()

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    if field_type == "node":
        # Use triangulation for smooth interpolation
        # Build node coordinate arrays
        sorted_node_ids = sorted(grid.nodes.keys())
        node_id_to_idx = {nid: i for i, nid in enumerate(sorted_node_ids)}

        x = np.array([grid.nodes[nid].x for nid in sorted_node_ids])
        y = np.array([grid.nodes[nid].y for nid in sorted_node_ids])

        # Build element connectivity
        elem_conn: list[list[int]] = []
        for elem in grid.iter_elements():
            verts = [node_id_to_idx[vid] for vid in elem.vertices]
            elem_conn.append(verts)

        has_quads = any(len(v) == 4 for v in elem_conn)

        if has_quads and n_subdiv > 1:
            # Bilinear FE subdivision for quads
            sub_x, sub_y, sub_v, sub_tri = _subdivide_quads(
                elem_conn,
                x,
                y,
                values,
                n_subdiv,
            )
            triang = Triangulation(sub_x, sub_y, sub_tri)
            tcf = ax.tripcolor(triang, sub_v, cmap=cmap, norm=norm, shading="gouraud")
        else:
            # Legacy 2-triangle diagonal split
            triangles_list: list[list[int]] = []
            for verts in elem_conn:
                if len(verts) == 3:
                    triangles_list.append(verts)
                else:
                    triangles_list.append([verts[0], verts[1], verts[2]])
                    triangles_list.append([verts[0], verts[2], verts[3]])

            triangles = np.array(triangles_list)
            triang = Triangulation(x, y, triangles)
            tcf = ax.tripcolor(triang, values, cmap=cmap, norm=norm, shading="gouraud")

        if show_mesh:
            node_xy = np.column_stack([x, y])
            mesh_polys = node_xy[np.array(elem_conn)]
            mesh_collection = PolyCollection(
                mesh_polys,
                edgecolors=edge_color,
                facecolors="none",
                linewidths=edge_width,
            )
            ax.add_collection(mesh_collection)

    else:  # cell values
        # Build element polygons
        polygons = []
        for elem in grid.iter_elements():
            coords = []
            for vid in elem.vertices:
                node = grid.nodes[vid]
                coords.append((node.x, node.y))
            polygons.append(coords)

        collection = PolyCollection(
            polygons,
            array=values,
            cmap=cmap,
            norm=norm,
            edgecolors=edge_color if show_mesh else "none",
            linewidths=edge_width,
        )
        ax.add_collection(collection)
        ax.autoscale_view()
        tcf = collection  # type: ignore[assignment]

    if show_colorbar:
        cbar = fig.colorbar(tcf, ax=ax)
        cbar.ax.tick_params(labelsize=9)

    ax.set_aspect("equal")
    ax.set_xlabel("X", fontsize=11)
    ax.set_ylabel("Y", fontsize=11)
    _apply_tick_formatting(ax)
    fig.tight_layout()

    return fig, ax


def plot_streams(
    streams: AppStream,
    ax: Axes | None = None,
    show_nodes: bool = False,
    line_color: str = "blue",
    line_width: float = 2.0,
    node_color: str = "blue",
    node_size: float = 30,
    figsize: tuple[float, float] = (10, 8),
) -> tuple[Figure, Axes]:
    """
    Plot stream network.

    Args:
        streams: Stream network
        ax: Existing axes to plot on
        show_nodes: Show stream node markers
        line_color: Color for stream lines
        line_width: Width of stream lines
        node_color: Color for stream nodes
        node_size: Size of stream node markers
        figsize: Figure size in inches

    Returns:
        Tuple of (Figure, Axes)
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    # Plot reaches as lines
    for reach in streams.iter_reaches():
        x_coords = []
        y_coords = []
        for nid in reach.nodes:
            if nid in streams.nodes:
                node = streams.nodes[nid]
                x_coords.append(node.x)
                y_coords.append(node.y)

        if len(x_coords) >= 2:
            ax.plot(x_coords, y_coords, color=line_color, linewidth=line_width)

    # Plot stream nodes
    if show_nodes:
        x = [node.x for node in streams.nodes.values()]
        y = [node.y for node in streams.nodes.values()]
        ax.scatter(x, y, s=node_size, c=node_color, zorder=5)

    ax.set_aspect("equal")
    ax.set_xlabel("X", fontsize=11)
    ax.set_ylabel("Y", fontsize=11)
    _apply_tick_formatting(ax)
    fig.tight_layout()

    return fig, ax


def plot_lakes(
    lakes: AppLake,
    grid: AppGrid,
    ax: Axes | None = None,
    fill_color: str = "cyan",
    edge_color: str = "blue",
    edge_width: float = 1.5,
    alpha: float = 0.5,
    show_labels: bool = True,
    label_fontsize: float = 9,
    cmap: str | None = None,
    figsize: tuple[float, float] = (10, 8),
) -> tuple[Figure, Axes]:
    """
    Plot lake elements on the mesh.

    Parameters
    ----------
    lakes : AppLake
        Lake component containing lake definitions and element assignments.
    grid : AppGrid
        Model mesh used to look up element vertex coordinates.
    ax : Axes, optional
        Existing axes to plot on. Creates new figure if None.
    fill_color : str, default "cyan"
        Fill color for lake elements (used when *cmap* is None).
    edge_color : str, default "blue"
        Edge color for lake element polygons.
    edge_width : float, default 1.5
        Width of lake element edges.
    alpha : float, default 0.5
        Transparency of lake element fill.
    show_labels : bool, default True
        Show lake name labels at the centroid of each lake.
    label_fontsize : float, default 9
        Font size for lake labels.
    cmap : str, optional
        If provided, color each lake with a different color from this
        colormap instead of using *fill_color*.
    figsize : tuple, default (10, 8)
        Figure size in inches.

    Returns
    -------
    tuple
        (Figure, Axes) matplotlib objects.
    """

    from matplotlib.patches import Polygon as MplPolygon

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    lake_list = list(lakes.iter_lakes())

    if cmap is not None:
        colormap = plt.get_cmap(cmap)
        n_lakes = max(len(lake_list), 1)

    for idx, lake in enumerate(lake_list):
        color = colormap(idx / n_lakes) if cmap is not None else fill_color
        lake_elems = lakes.get_elements_for_lake(lake.id)

        all_x: list[float] = []
        all_y: list[float] = []

        for le in lake_elems:
            if le.element_id not in grid.elements:
                continue
            elem = grid.elements[le.element_id]
            verts = [(grid.nodes[vid].x, grid.nodes[vid].y) for vid in elem.vertices]
            patch = MplPolygon(
                verts,
                facecolor=color,
                edgecolor=edge_color,
                linewidth=edge_width,
                alpha=alpha,
            )
            ax.add_patch(patch)

            for vx, vy in verts:
                all_x.append(vx)
                all_y.append(vy)

        if show_labels and all_x:
            cx = sum(all_x) / len(all_x)
            cy = sum(all_y) / len(all_y)
            label = lake.name or f"Lake {lake.id}"
            ax.text(
                cx,
                cy,
                label,
                ha="center",
                va="center",
                fontsize=label_fontsize,
                fontweight="bold",
                zorder=10,
            )

    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xlabel("X", fontsize=11)
    ax.set_ylabel("Y", fontsize=11)
    _apply_tick_formatting(ax)
    fig.tight_layout()

    return fig, ax


def plot_boundary(
    grid: AppGrid,
    ax: Axes | None = None,
    line_color: str = "black",
    line_width: float = 2.0,
    fill: bool = False,
    fill_color: str = "lightgray",
    alpha: float = 0.3,
    figsize: tuple[float, float] = (10, 8),
) -> tuple[Figure, Axes]:
    """
    Plot model boundary.

    Args:
        grid: Model mesh
        ax: Existing axes to plot on
        line_color: Color for boundary line
        line_width: Width of boundary line
        fill: Fill the boundary polygon
        fill_color: Fill color
        alpha: Fill transparency
        figsize: Figure size in inches

    Returns:
        Tuple of (Figure, Axes)
    """

    from matplotlib.patches import Polygon as MplPolygon

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    # Extract boundary nodes (ordered)
    boundary_nodes = [n for n in grid.iter_nodes() if n.is_boundary]

    if not boundary_nodes:
        # Fallback: use convex hull of all nodes
        from scipy.spatial import ConvexHull

        points = np.array([[n.x, n.y] for n in grid.iter_nodes()])
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]

        if fill:
            patch = MplPolygon(
                hull_points,
                facecolor=fill_color,
                edgecolor=line_color,
                linewidth=line_width,
                alpha=alpha,
            )
            ax.add_patch(patch)
        else:
            # Close the polygon
            x = np.append(hull_points[:, 0], hull_points[0, 0])
            y = np.append(hull_points[:, 1], hull_points[0, 1])
            ax.plot(x, y, color=line_color, linewidth=line_width)
    else:
        # Use boundary nodes - order them by angle from centroid
        cx = sum(n.x for n in boundary_nodes) / len(boundary_nodes)
        cy = sum(n.y for n in boundary_nodes) / len(boundary_nodes)

        def angle(n: Any) -> Any:
            return np.arctan2(n.y - cy, n.x - cx)

        sorted_nodes = sorted(boundary_nodes, key=angle)
        coords = [(n.x, n.y) for n in sorted_nodes]

        if fill:
            patch = MplPolygon(
                coords,
                facecolor=fill_color,
                edgecolor=line_color,
                linewidth=line_width,
                alpha=alpha,
            )
            ax.add_patch(patch)
        else:
            x_plot = [c[0] for c in coords] + [coords[0][0]]
            y_plot = [c[1] for c in coords] + [coords[0][1]]
            ax.plot(x_plot, y_plot, color=line_color, linewidth=line_width)

    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xlabel("X", fontsize=11)
    ax.set_ylabel("Y", fontsize=11)
    _apply_tick_formatting(ax)
    fig.tight_layout()

    return fig, ax


class MeshPlotter:
    """
    High-level class for creating mesh visualizations.

    This class provides a convenient interface for creating
    multi-layer visualizations of IWFM model meshes.

    Attributes:
        grid: Model mesh
        streams: Stream network (optional)
    """

    def __init__(
        self,
        grid: AppGrid,
        streams: AppStream | None = None,
        figsize: tuple[float, float] = (10, 8),
    ) -> None:
        """
        Initialize the mesh plotter.

        Args:
            grid: Model mesh
            streams: Stream network (optional)
            figsize: Default figure size
        """
        self.grid = grid
        self.streams = streams
        self.figsize = figsize
        self._fig: Figure | None = None
        self._ax: Axes | None = None

    def plot_mesh(
        self,
        show_edges: bool = True,
        show_node_ids: bool = False,
        show_element_ids: bool = False,
        show_streams: bool = False,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """
        Plot the mesh with optional overlays.

        Args:
            show_edges: Show element edges
            show_node_ids: Label nodes with their IDs
            show_element_ids: Label elements with their IDs
            show_streams: Overlay stream network
            **kwargs: Additional arguments passed to plot_mesh

        Returns:
            Tuple of (Figure, Axes)
        """
        fig, ax = plot_mesh(
            self.grid,
            show_edges=show_edges,
            show_node_ids=show_node_ids,
            show_element_ids=show_element_ids,
            figsize=self.figsize,
            **kwargs,
        )

        if show_streams and self.streams is not None:
            plot_streams(self.streams, ax=ax)

        self._fig = fig
        self._ax = ax
        return fig, ax

    def plot_composite(
        self,
        show_mesh: bool = True,
        show_streams: bool = False,
        node_values: NDArray[np.float64] | None = None,
        cell_values: NDArray[np.float64] | None = None,
        title: str | None = None,
        cmap: str = "viridis",
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """
        Create a composite plot with multiple layers.

        Args:
            show_mesh: Show mesh edges
            show_streams: Overlay stream network
            node_values: Scalar values at nodes (optional)
            cell_values: Scalar values at cells (optional)
            title: Plot title
            cmap: Colormap for scalar values
            **kwargs: Additional arguments

        Returns:
            Tuple of (Figure, Axes)
        """

        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot scalar field if provided
        if node_values is not None:
            plot_scalar_field(
                self.grid,
                node_values,
                field_type="node",
                ax=ax,
                cmap=cmap,
                show_mesh=show_mesh,
            )
        elif cell_values is not None:
            plot_scalar_field(
                self.grid,
                cell_values,
                field_type="cell",
                ax=ax,
                cmap=cmap,
                show_mesh=show_mesh,
            )
        elif show_mesh:
            plot_mesh(self.grid, ax=ax)

        # Add streams
        if show_streams and self.streams is not None:
            plot_streams(self.streams, ax=ax)

        if title:
            ax.set_title(title)

        self._fig = fig
        self._ax = ax
        return fig, ax

    def save(
        self,
        output_path: Path | str,
        dpi: int = 150,
        **kwargs: Any,
    ) -> None:
        """
        Save the current figure to file.

        Args:
            output_path: Output file path
            dpi: Resolution in dots per inch
            **kwargs: Additional arguments passed to savefig
        """
        if self._fig is None:
            # Create default plot if none exists
            self.plot_mesh()

        if self._fig is not None:
            self._fig.savefig(output_path, dpi=dpi, bbox_inches="tight", **kwargs)


# =============================================================================
# Time Series Plotting Functions
# =============================================================================


def plot_timeseries(
    timeseries: TimeSeries | Sequence[TimeSeries],
    ax: Axes | None = None,
    title: str | None = None,
    xlabel: str = "Date",
    ylabel: str | None = None,
    legend: bool = True,
    colors: Sequence[str] | None = None,
    linestyles: Sequence[str] | None = None,
    markers: Sequence[str | None] | None = None,
    figsize: tuple[float, float] = (12, 6),
    grid: bool = True,
    date_format: str | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot one or more time series as line charts.

    Parameters
    ----------
    timeseries : TimeSeries or sequence of TimeSeries
        Time series data to plot. Can be a single TimeSeries or a list.
    ax : Axes, optional
        Existing axes to plot on. Creates new figure if None.
    title : str, optional
        Plot title.
    xlabel : str, default "Date"
        X-axis label.
    ylabel : str, optional
        Y-axis label. Uses units from first time series if not specified.
    legend : bool, default True
        Show legend.
    colors : sequence of str, optional
        Line colors for each series.
    linestyles : sequence of str, optional
        Line styles for each series (e.g., '-', '--', ':').
    markers : sequence of str, optional
        Markers for each series (e.g., 'o', 's', '^').
    figsize : tuple, default (12, 6)
        Figure size in inches.
    grid : bool, default True
        Show grid lines.
    date_format : str, optional
        Date format for x-axis (e.g., '%Y-%m').

    Returns
    -------
    tuple
        (Figure, Axes) matplotlib objects.

    Examples
    --------
    Plot a single time series:

    >>> ts = TimeSeries(times=times, values=values, name='Head', units='ft')
    >>> fig, ax = plot_timeseries(ts, title='Groundwater Head')

    Plot multiple time series:

    >>> fig, ax = plot_timeseries([ts1, ts2, ts3], legend=True)
    """

    import matplotlib.dates as mdates

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    # Ensure we have a list of time series
    series_list: list[TimeSeries]
    if hasattr(timeseries, "times"):  # Single TimeSeries
        series_list = [timeseries]  # type: ignore[list-item]
    else:
        series_list = list(timeseries)  # type: ignore[arg-type]

    # Default styling
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if linestyles is None:
        linestyles = ["-"] * len(series_list)
    if markers is None:
        markers = [None] * len(series_list)

    # Plot each time series
    for i, ts in enumerate(series_list):
        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        marker = markers[i % len(markers)]

        label = ts.name or f"Series {i + 1}"
        if ts.units and ts.units not in label:
            label = f"{label} ({ts.units})"

        # Convert times for matplotlib
        times_plot = ts.times.astype("datetime64[us]").astype("O")

        ax.plot(
            times_plot,
            ts.values,
            color=color,
            linestyle=linestyle,
            marker=marker,
            label=label,
            linewidth=1.5,
            markersize=4,
        )

    # Formatting
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    elif series_list and series_list[0].units:
        ax.set_ylabel(series_list[0].units, fontsize=11)

    _has_legend = legend and len(series_list) > 1
    if _has_legend:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0,
            framealpha=0.9,
            edgecolor="lightgray",
        )

    if grid:
        ax.grid(True, alpha=0.3)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)

    # Date formatting
    if date_format:
        ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    fig.autofmt_xdate()
    fig.tight_layout()
    if _has_legend:
        fig.subplots_adjust(right=0.80)

    return fig, ax


def plot_timeseries_comparison(
    observed: TimeSeries,
    simulated: TimeSeries,
    ax: Axes | None = None,
    title: str | None = None,
    show_residuals: bool = False,
    show_metrics: bool = True,
    obs_color: str = "blue",
    sim_color: str = "red",
    obs_marker: str = "o",
    figsize: tuple[float, float] = (12, 8),
) -> tuple[Figure, Axes]:
    """
    Plot observed vs simulated time series comparison.

    Parameters
    ----------
    observed : TimeSeries
        Observed data time series.
    simulated : TimeSeries
        Simulated/modeled data time series.
    ax : Axes, optional
        Existing axes. Creates new figure if None.
    title : str, optional
        Plot title.
    show_residuals : bool, default False
        Show residual subplot below main plot.
    show_metrics : bool, default True
        Display comparison metrics (RMSE, NSE, etc.) on plot.
    obs_color : str, default "blue"
        Color for observed data.
    sim_color : str, default "red"
        Color for simulated data.
    obs_marker : str, default "o"
        Marker for observed data points.
    figsize : tuple, default (12, 8)
        Figure size.

    Returns
    -------
    tuple
        (Figure, Axes) matplotlib objects.

    Examples
    --------
    >>> fig, ax = plot_timeseries_comparison(
    ...     observed=obs_ts,
    ...     simulated=sim_ts,
    ...     title='Head Calibration - Well 1',
    ...     show_metrics=True
    ... )
    """

    if show_residuals:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1], sharex=True)
        ax = ax1
    else:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()  # type: ignore[assignment]

    # Plot observed
    obs_times = observed.times.astype("datetime64[us]").astype("O")
    ax.scatter(
        obs_times,
        observed.values,
        c=obs_color,
        marker=obs_marker,
        s=30,
        label="Observed",
        zorder=5,
        alpha=0.7,
    )

    # Plot simulated
    sim_times = simulated.times.astype("datetime64[us]").astype("O")
    ax.plot(
        sim_times,
        simulated.values,
        c=sim_color,
        linewidth=1.5,
        label="Simulated",
    )

    # Calculate and display metrics
    if show_metrics:
        try:
            from pyiwfm.comparison.metrics import ComparisonMetrics

            # Interpolate to common times for metric calculation
            obs_vals = observed.values
            sim_vals = np.interp(
                observed.times.astype(float),
                simulated.times.astype(float),
                simulated.values,
            )
            metrics = ComparisonMetrics.compute(obs_vals, sim_vals)

            metrics_text = (
                f"RMSE: {metrics.rmse:.3f}\n"
                f"NSE: {metrics.nash_sutcliffe:.3f}\n"
                f"PBIAS: {metrics.percent_bias:.1f}%\n"
                f"r: {metrics.correlation:.3f}"
            )
            ax.text(
                0.02,
                0.98,
                metrics_text,
                transform=ax.transAxes,
                verticalalignment="top",
                fontsize=9,
                fontfamily="monospace",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            )
        except ImportError:
            pass

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        framealpha=0.9,
        edgecolor="lightgray",
    )
    ax.grid(True, alpha=0.3)

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    if observed.units:
        ax.set_ylabel(observed.units, fontsize=11)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)

    # Residuals subplot
    if show_residuals:
        # Interpolate simulated to observed times
        sim_interp = np.interp(
            observed.times.astype(float),
            simulated.times.astype(float),
            simulated.values,
        )
        residuals = sim_interp - observed.values

        ax2.bar(obs_times, residuals, color="gray", alpha=0.7, width=2)
        ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax2.set_ylabel("Residual", fontsize=11)
        ax2.set_xlabel("Date", fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.tick_params(axis="both", labelsize=9)
    else:
        ax.set_xlabel("Date", fontsize=11)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.subplots_adjust(right=0.80)

    return fig, ax


def plot_timeseries_collection(
    collection: TimeSeriesCollection,
    locations: Sequence[str] | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    max_series: int = 10,
    figsize: tuple[float, float] = (12, 6),
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """
    Plot multiple time series from a collection.

    Parameters
    ----------
    collection : TimeSeriesCollection
        Collection of time series data.
    locations : sequence of str, optional
        Specific locations to plot. Plots all if None.
    ax : Axes, optional
        Existing axes to plot on.
    title : str, optional
        Plot title. Uses collection name if not specified.
    max_series : int, default 10
        Maximum number of series to plot (for readability).
    figsize : tuple, default (12, 6)
        Figure size.
    **kwargs
        Additional arguments passed to plot_timeseries.

    Returns
    -------
    tuple
        (Figure, Axes) matplotlib objects.
    """
    if locations is None:
        locations = collection.locations[:max_series]
    else:
        locations = list(locations)[:max_series]

    series_list = [collection[loc] for loc in locations if loc in collection.series]

    if not title and collection.name:
        title = collection.name

    return plot_timeseries(series_list, ax=ax, title=title, figsize=figsize, **kwargs)


# =============================================================================
# Budget Plotting Functions
# =============================================================================


def plot_budget_bar(
    components: dict[str, float],
    ax: Axes | None = None,
    title: str = "Water Budget",
    orientation: Literal["vertical", "horizontal"] = "vertical",
    inflow_color: str = "steelblue",
    outflow_color: str = "coral",
    show_values: bool = True,
    units: str = "AF",
    figsize: tuple[float, float] = (10, 6),
) -> tuple[Figure, Axes]:
    """
    Plot water budget components as a bar chart.

    Parameters
    ----------
    components : dict
        Dictionary of component names to values. Positive values are inflows,
        negative values are outflows.
    ax : Axes, optional
        Existing axes to plot on.
    title : str, default "Water Budget"
        Plot title.
    orientation : {'vertical', 'horizontal'}, default 'vertical'
        Bar orientation.
    inflow_color : str, default "steelblue"
        Color for inflow (positive) bars.
    outflow_color : str, default "coral"
        Color for outflow (negative) bars.
    show_values : bool, default True
        Show values on bars.
    units : str, default "AF"
        Units for y-axis label and values.
    figsize : tuple, default (10, 6)
        Figure size.

    Returns
    -------
    tuple
        (Figure, Axes) matplotlib objects.

    Examples
    --------
    >>> budget = {
    ...     'Precipitation': 1500,
    ...     'Stream Inflow': 800,
    ...     'Pumping': -1200,
    ...     'Evapotranspiration': -600,
    ...     'Stream Outflow': -400,
    ... }
    >>> fig, ax = plot_budget_bar(budget, title='Annual Water Budget')
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    names = list(components.keys())
    values = list(components.values())
    colors = [inflow_color if v >= 0 else outflow_color for v in values]

    if orientation == "vertical":
        bars = ax.bar(names, values, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_ylabel(f"Volume ({units})", fontsize=11)
        ax.axhline(y=0, color="black", linewidth=0.8)

        if show_values:
            for bar, val in zip(bars, values, strict=False):
                height = bar.get_height()
                va = "bottom" if height >= 0 else "top"
                offset = 0.01 * max(abs(v) for v in values)
                y = height + offset if height >= 0 else height - offset
                ax.annotate(
                    f"{val:,.0f}",
                    xy=(bar.get_x() + bar.get_width() / 2, y),
                    ha="center",
                    va=va,
                    fontsize=9,
                )

        plt.xticks(rotation=45, ha="right")
    else:
        bars = ax.barh(names, values, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xlabel(f"Volume ({units})", fontsize=11)
        ax.axvline(x=0, color="black", linewidth=0.8)

        if show_values:
            for bar, val in zip(bars, values, strict=False):
                width = bar.get_width()
                ha = "left" if width >= 0 else "right"
                offset = 0.02 * max(abs(v) for v in values)
                x = width + offset if width >= 0 else width - offset
                ax.annotate(
                    f"{val:,.0f}",
                    xy=(x, bar.get_y() + bar.get_height() / 2),
                    ha=ha,
                    va="center",
                    fontsize=9,
                )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.grid(True, alpha=0.3, axis="y" if orientation == "vertical" else "x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)
    fig.tight_layout()

    return fig, ax


def plot_budget_stacked(
    times: NDArray[np.datetime64],
    components: dict[str, NDArray[np.float64]],
    ax: Axes | None = None,
    title: str = "Water Budget Over Time",
    inflows_above: bool = True,
    cmap: str = "tab10",
    alpha: float = 0.8,
    units: str = "AF",
    show_legend: bool = True,
    figsize: tuple[float, float] = (14, 7),
) -> tuple[Figure, Axes]:
    """
    Plot water budget components as stacked area chart over time.

    Parameters
    ----------
    times : array
        Time array (datetime64).
    components : dict
        Dictionary of component names to time series arrays.
        Positive values are inflows, negative values are outflows.
    ax : Axes, optional
        Existing axes to plot on.
    title : str, default "Water Budget Over Time"
        Plot title.
    inflows_above : bool, default True
        Plot inflows above x-axis and outflows below.
    cmap : str, default "tab10"
        Colormap for components.
    alpha : float, default 0.8
        Fill transparency.
    units : str, default "AF"
        Units for y-axis label.
    show_legend : bool, default True
        Show legend.
    figsize : tuple, default (14, 7)
        Figure size.

    Returns
    -------
    tuple
        (Figure, Axes) matplotlib objects.

    Examples
    --------
    >>> times = np.array(['2020-01', '2020-02', '2020-03'], dtype='datetime64')
    >>> components = {
    ...     'Precipitation': np.array([100, 150, 80]),
    ...     'Pumping': np.array([-50, -60, -55]),
    ... }
    >>> fig, ax = plot_budget_stacked(times, components)
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    # Convert times
    times_plot = times.astype("datetime64[us]").astype("O")

    # Separate inflows and outflows
    inflows = {k: v for k, v in components.items() if np.mean(v) >= 0}
    outflows = {k: -v for k, v in components.items() if np.mean(v) < 0}

    colormap = plt.get_cmap(cmap)
    n_components = len(components)

    # Plot inflows (stacked above zero)
    if inflows:
        labels = list(inflows.keys())
        data = np.array([inflows[k] for k in labels])

        colors = [colormap(i / n_components) for i in range(len(labels))]
        ax.stackplot(times_plot, data, labels=labels, colors=colors, alpha=alpha)

    # Plot outflows (stacked below zero)
    if outflows:
        labels = list(outflows.keys())
        data = np.array([outflows[k] for k in labels])

        start_idx = len(inflows)
        colors = [colormap((start_idx + i) / n_components) for i in range(len(labels))]
        ax.stackplot(
            times_plot, -data, labels=[f"{k} (out)" for k in labels], colors=colors, alpha=alpha
        )

    ax.axhline(y=0, color="black", linewidth=1)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel(f"Flow Rate ({units})", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)

    if show_legend:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0,
            framealpha=0.9,
            edgecolor="lightgray",
        )

    fig.autofmt_xdate()
    fig.tight_layout()
    if show_legend:
        fig.subplots_adjust(right=0.80)

    return fig, ax


def plot_budget_pie(
    components: dict[str, float],
    ax: Axes | None = None,
    title: str = "Water Budget Distribution",
    budget_type: Literal["inflow", "outflow", "both"] = "both",
    cmap: str = "tab10",
    show_values: bool = True,
    units: str = "AF",
    figsize: tuple[float, float] = (10, 8),
) -> tuple[Figure, Axes]:
    """
    Plot water budget components as pie chart(s).

    Parameters
    ----------
    components : dict
        Dictionary of component names to values.
    ax : Axes, optional
        Existing axes (ignored if budget_type='both').
    title : str, default "Water Budget Distribution"
        Plot title.
    budget_type : {'inflow', 'outflow', 'both'}, default 'both'
        Which components to show.
    cmap : str, default "tab10"
        Colormap for slices.
    show_values : bool, default True
        Show values in labels.
    units : str, default "AF"
        Units for value labels.
    figsize : tuple, default (10, 8)
        Figure size.

    Returns
    -------
    tuple
        (Figure, Axes) matplotlib objects.
    """

    inflows = {k: v for k, v in components.items() if v > 0}
    outflows = {k: abs(v) for k, v in components.items() if v < 0}

    if budget_type == "both" and inflows and outflows:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        def make_pie(ax: Any, data: dict[str, float], subtitle: str) -> None:
            labels = list(data.keys())
            values = list(data.values())
            colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(labels)))

            if show_values:
                labels = [f"{k}\n({v:,.0f} {units})" for k, v in zip(labels, values, strict=False)]

            wedges, texts, autotexts = ax.pie(
                values,
                labels=labels,
                autopct="%1.1f%%",
                colors=colors,
                startangle=90,
            )
            ax.set_title(subtitle, fontsize=11, fontweight="bold")

        make_pie(ax1, inflows, "Inflows")
        make_pie(ax2, outflows, "Outflows")
        fig.suptitle(title, fontsize=14, fontweight="bold")
        ax = ax1  # Use first axes as the returned Axes
    else:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()  # type: ignore[assignment]

        data = inflows if budget_type == "inflow" else outflows
        labels = list(data.keys())
        values = list(data.values())
        colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(labels)))

        if show_values:
            labels = [f"{k}\n({v:,.0f} {units})" for k, v in zip(labels, values, strict=False)]

        ax.pie(values, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)  # type: ignore[arg-type]
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    fig.tight_layout()
    return fig, ax


def plot_water_balance(
    inflows: dict[str, float],
    outflows: dict[str, float],
    storage_change: float = 0.0,
    ax: Axes | None = None,
    title: str = "Water Balance Summary",
    units: str = "AF",
    figsize: tuple[float, float] = (12, 6),
) -> tuple[Figure, Axes]:
    """
    Plot comprehensive water balance summary with inflows, outflows, and storage.

    Parameters
    ----------
    inflows : dict
        Dictionary of inflow component names to values.
    outflows : dict
        Dictionary of outflow component names to values (positive values).
    storage_change : float, default 0.0
        Change in storage (positive = increase).
    ax : Axes, optional
        Existing axes to plot on.
    title : str, default "Water Balance Summary"
        Plot title.
    units : str, default "AF"
        Volume units.
    figsize : tuple, default (12, 6)
        Figure size.

    Returns
    -------
    tuple
        (Figure, Axes) matplotlib objects.

    Examples
    --------
    >>> inflows = {'Precip': 1000, 'Stream In': 500, 'Recharge': 200}
    >>> outflows = {'ET': 600, 'Pumping': 800, 'Stream Out': 300}
    >>> fig, ax = plot_water_balance(inflows, outflows, storage_change=-100)
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    total_in = sum(inflows.values())
    total_out = sum(outflows.values())
    balance_error = total_in - total_out - storage_change

    # Create waterfall-style chart
    categories = (
        list(inflows.keys())
        + ["Total Inflow"]
        + [f"-{k}" for k in outflows.keys()]
        + ["Total Outflow", "Storage Change", "Balance Error"]
    )
    values = (
        list(inflows.values())
        + [total_in]
        + [-v for v in outflows.values()]
        + [-total_out, storage_change, balance_error]
    )

    # Colors
    colors = []
    for i, v in enumerate(values):
        if categories[i] in ["Total Inflow", "Total Outflow"]:
            colors.append("gray")
        elif categories[i] == "Storage Change":
            colors.append("gold")
        elif categories[i] == "Balance Error":
            colors.append("purple")
        elif v >= 0:
            colors.append("steelblue")
        else:
            colors.append("coral")

    y_pos = np.arange(len(categories))
    bars = ax.barh(y_pos, values, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, values, strict=False):
        width = bar.get_width()
        ha = "left" if width >= 0 else "right"
        offset = max(abs(v) for v in values) * 0.02
        x = width + offset if width >= 0 else width - offset
        ax.annotate(
            f"{val:,.0f}",
            xy=(x, bar.get_y() + bar.get_height() / 2),
            ha=ha,
            va="center",
            fontsize=9,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories)
    ax.axvline(x=0, color="black", linewidth=1)
    ax.set_xlabel(f"Volume ({units})", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.grid(True, alpha=0.3, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)

    # Summary text
    summary = (
        f"Total In: {total_in:,.0f} {units}\n"
        f"Total Out: {total_out:,.0f} {units}\n"
        f"ΔStorage: {storage_change:,.0f} {units}\n"
        f"Error: {balance_error:,.1f} {units} ({100 * balance_error / total_in:.2f}%)"
    )
    ax.text(
        0.98,
        0.02,
        summary,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        fontfamily="monospace",
        bbox={"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.9},
    )

    fig.tight_layout()
    return fig, ax


def plot_zbudget(
    zone_budgets: dict[int | str, dict[str, float]],
    ax: Axes | None = None,
    title: str = "Zone Budget Summary",
    plot_type: Literal["bar", "heatmap"] = "bar",
    units: str = "AF",
    cmap: str = "RdYlBu",
    figsize: tuple[float, float] = (12, 8),
) -> tuple[Figure, Axes]:
    """
    Plot zone budget data for multiple zones.

    Parameters
    ----------
    zone_budgets : dict
        Dictionary mapping zone ID to budget component dictionaries.
        Example: {1: {'Inflow': 100, 'Outflow': -80}, 2: {...}}
    ax : Axes, optional
        Existing axes to plot on.
    title : str, default "Zone Budget Summary"
        Plot title.
    plot_type : {'bar', 'heatmap'}, default 'bar'
        Type of plot to create.
    units : str, default "AF"
        Volume units.
    cmap : str, default "RdYlBu"
        Colormap for heatmap.
    figsize : tuple, default (12, 8)
        Figure size.

    Returns
    -------
    tuple
        (Figure, Axes) matplotlib objects.

    Examples
    --------
    >>> zone_budgets = {
    ...     1: {'Recharge': 500, 'Pumping': -300, 'Flow to Zone 2': -100},
    ...     2: {'Recharge': 300, 'Pumping': -200, 'Flow from Zone 1': 100},
    ... }
    >>> fig, ax = plot_zbudget(zone_budgets, title='Subregion Budgets')
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    zones = list(zone_budgets.keys())
    all_components: set[str] = set()
    for budget in zone_budgets.values():
        all_components.update(budget.keys())
    components = sorted(all_components)

    if plot_type == "heatmap":
        # Create data matrix
        data = np.zeros((len(zones), len(components)))
        for i, zone in enumerate(zones):
            for j, comp in enumerate(components):
                data[i, j] = zone_budgets[zone].get(comp, 0)

        im = ax.imshow(data, cmap=cmap, aspect="auto")

        ax.set_xticks(np.arange(len(components)))
        ax.set_yticks(np.arange(len(zones)))
        ax.set_xticklabels(components, rotation=45, ha="right")
        ax.set_yticklabels([f"Zone {z}" for z in zones])

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(f"Volume ({units})")

        # Add text annotations
        for i in range(len(zones)):
            for j in range(len(components)):
                val = data[i, j]
                color = "white" if abs(val) > np.max(np.abs(data)) * 0.5 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", color=color, fontsize=8)

    else:  # bar chart
        n_zones = len(zones)
        n_components = len(components)
        x = np.arange(n_components)
        width = 0.8 / n_zones

        colors = plt.get_cmap("tab10")(np.linspace(0, 1, n_zones))

        for i, zone in enumerate(zones):
            values = [zone_budgets[zone].get(comp, 0) for comp in components]
            offset = (i - n_zones / 2 + 0.5) * width
            ax.bar(x + offset, values, width, label=f"Zone {zone}", color=colors[i])

        ax.set_xticks(x)
        ax.set_xticklabels(components, rotation=45, ha="right")
        ax.set_ylabel(f"Volume ({units})", fontsize=11)
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0,
            framealpha=0.9,
            edgecolor="lightgray",
        )
        ax.grid(True, alpha=0.3, axis="y")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)
    fig.tight_layout()
    if plot_type == "bar":
        fig.subplots_adjust(right=0.80)

    return fig, ax


def plot_budget_timeseries(
    times: NDArray[np.datetime64],
    budgets: dict[str, NDArray[np.float64]],
    cumulative: bool = False,
    ax: Axes | None = None,
    title: str = "Budget Components Over Time",
    units: str = "AF",
    show_net: bool = True,
    figsize: tuple[float, float] = (14, 6),
) -> tuple[Figure, Axes]:
    """
    Plot budget component time series as line charts.

    Parameters
    ----------
    times : array
        Time array (datetime64).
    budgets : dict
        Dictionary of component names to value arrays.
    cumulative : bool, default False
        Plot cumulative values.
    ax : Axes, optional
        Existing axes to plot on.
    title : str, default "Budget Components Over Time"
        Plot title.
    units : str, default "AF"
        Volume units.
    show_net : bool, default True
        Show net budget line.
    figsize : tuple, default (14, 6)
        Figure size.

    Returns
    -------
    tuple
        (Figure, Axes) matplotlib objects.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    times_plot = times.astype("datetime64[us]").astype("O")
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(budgets)))

    net = np.zeros(len(times))

    for (name, values), color in zip(budgets.items(), colors, strict=False):
        plot_values = np.cumsum(values) if cumulative else values
        net += values

        linestyle = "-" if np.mean(values) >= 0 else "--"
        ax.plot(
            times_plot, plot_values, label=name, color=color, linestyle=linestyle, linewidth=1.5
        )

    if show_net:
        net_plot = np.cumsum(net) if cumulative else net
        ax.plot(
            times_plot,
            net_plot,
            label="Net",
            color="black",
            linewidth=2,
            linestyle="-",
        )

    ylabel = f"{'Cumulative ' if cumulative else ''}Volume ({units})"
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        framealpha=0.9,
        edgecolor="lightgray",
    )
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.subplots_adjust(right=0.80)

    return fig, ax


class BudgetPlotter:
    """
    High-level class for creating budget visualizations.

    This class provides convenience methods for creating various
    budget-related plots from IWFM model output data.

    Parameters
    ----------
    budgets : dict
        Budget data organized by time step or as totals.
    times : array, optional
        Time array for time-series plots.
    units : str, default "AF"
        Volume units.

    Examples
    --------
    >>> plotter = BudgetPlotter(budgets={'Precip': 1000, 'ET': -600})
    >>> fig, ax = plotter.bar_chart()
    >>> plotter.save('budget.png')
    """

    def __init__(
        self,
        budgets: dict[str, float | NDArray[np.float64]],
        times: NDArray[np.datetime64] | None = None,
        units: str = "AF",
    ) -> None:
        self.budgets = budgets
        self.times = times
        self.units = units
        self._fig: Figure | None = None
        self._ax: Axes | None = None

    def bar_chart(self, **kwargs: Any) -> tuple[Figure, Axes]:
        """Create bar chart of budget components."""
        # Convert arrays to totals if needed
        totals = {}
        for k, v in self.budgets.items():
            if isinstance(v, np.ndarray):
                totals[k] = float(np.sum(v))
            else:
                totals[k] = v

        fig, ax = plot_budget_bar(totals, units=self.units, **kwargs)
        self._fig, self._ax = fig, ax
        return fig, ax

    def stacked_area(self, **kwargs: Any) -> tuple[Figure, Axes]:
        """Create stacked area chart over time."""
        if self.times is None:
            raise ValueError("Time array required for stacked area chart")

        # Ensure values are arrays
        arrays = {}
        for k, v in self.budgets.items():
            if isinstance(v, np.ndarray):
                arrays[k] = v
            else:
                arrays[k] = np.full(len(self.times), v)

        fig, ax = plot_budget_stacked(self.times, arrays, units=self.units, **kwargs)
        self._fig, self._ax = fig, ax
        return fig, ax

    def pie_chart(self, **kwargs: Any) -> tuple[Figure, Axes]:
        """Create pie chart of budget distribution."""
        totals = {}
        for k, v in self.budgets.items():
            if isinstance(v, np.ndarray):
                totals[k] = float(np.sum(v))
            else:
                totals[k] = v

        fig, ax = plot_budget_pie(totals, units=self.units, **kwargs)
        self._fig, self._ax = fig, ax
        return fig, ax

    def line_chart(self, **kwargs: Any) -> tuple[Figure, Axes]:
        """Create line chart of budget components over time."""
        if self.times is None:
            raise ValueError("Time array required for line chart")

        arrays = {}
        for k, v in self.budgets.items():
            if isinstance(v, np.ndarray):
                arrays[k] = v
            else:
                arrays[k] = np.full(len(self.times), v)

        fig, ax = plot_budget_timeseries(self.times, arrays, units=self.units, **kwargs)
        self._fig, self._ax = fig, ax
        return fig, ax

    def save(self, output_path: Path | str, dpi: int = 150, **kwargs: Any) -> None:
        """Save current figure to file."""
        if self._fig is None:
            self.bar_chart()
        if self._fig is not None:
            self._fig.savefig(output_path, dpi=dpi, bbox_inches="tight", **kwargs)


# =============================================================================
# Additional Plotting Functions
# =============================================================================


def plot_streams_colored(
    grid: AppGrid,
    streams: AppStream,
    values: NDArray[np.float64],
    ax: Axes | None = None,
    cmap: str = "Blues",
    vmin: float | None = None,
    vmax: float | None = None,
    line_width: float = 2.0,
    show_colorbar: bool = True,
    colorbar_label: str = "",
    show_mesh: bool = True,
    mesh_alpha: float = 0.15,
    figsize: tuple[float, float] = (10, 8),
) -> tuple[Figure, Axes]:
    """
    Color stream reaches by a scalar value (e.g., flow rate, gaining/losing).

    Parameters
    ----------
    grid : AppGrid
        Model mesh (plotted as background when *show_mesh* is True).
    streams : AppStream
        Stream network.
    values : ndarray
        One value per reach, used for coloring.
    ax : Axes, optional
        Existing axes to plot on.
    cmap : str, default "Blues"
        Matplotlib colormap name.
    vmin, vmax : float, optional
        Limits for the color scale.
    line_width : float, default 2.0
        Width of stream lines.
    show_colorbar : bool, default True
        Whether to add a colorbar.
    colorbar_label : str, default ""
        Label for the colorbar.
    show_mesh : bool, default True
        Whether to draw the mesh as background.
    mesh_alpha : float, default 0.15
        Alpha for mesh background.
    figsize : tuple, default (10, 8)
        Figure size in inches.

    Returns
    -------
    tuple
        (Figure, Axes) matplotlib objects.
    """

    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    # Draw mesh background
    if show_mesh:
        plot_mesh(grid, ax=ax, alpha=mesh_alpha, edge_color="lightgray", edge_width=0.3)

    # Build reach segments
    segments: list[list[tuple[float, float]]] = []
    reach_values: list[float] = []
    for idx, reach in enumerate(streams.iter_reaches()):
        coords: list[tuple[float, float]] = []
        for nid in reach.nodes:
            if nid in streams.nodes:
                node = streams.nodes[nid]
                coords.append((node.x, node.y))
        if len(coords) >= 2 and idx < len(values):
            segments.append(coords)
            reach_values.append(float(values[idx]))

    norm = Normalize(
        vmin=vmin if vmin is not None else min(reach_values),
        vmax=vmax if vmax is not None else max(reach_values),
    )
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=line_width, zorder=5)
    lc.set_array(np.array(reach_values))
    ax.add_collection(lc)

    if show_colorbar:
        cb = fig.colorbar(lc, ax=ax)  # type: ignore[arg-type]
        cb.set_label(colorbar_label)

    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xlabel("X", fontsize=11)
    ax.set_ylabel("Y", fontsize=11)
    _apply_tick_formatting(ax)
    fig.tight_layout()

    return fig, ax


def plot_timeseries_statistics(
    collection: TimeSeriesCollection,
    ax: Axes | None = None,
    band: Literal["minmax", "std"] = "minmax",
    mean_color: str = "steelblue",
    band_alpha: float = 0.25,
    show_individual: bool = False,
    individual_alpha: float = 0.15,
    title: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] = (12, 6),
) -> tuple[Figure, Axes]:
    """
    Plot ensemble mean with min/max or standard-deviation bands.

    Parameters
    ----------
    collection : TimeSeriesCollection
        Collection of time series data.
    ax : Axes, optional
        Existing axes to plot on.
    band : {"minmax", "std"}, default "minmax"
        Band type: min/max envelope or +/- 1 standard deviation.
    mean_color : str, default "steelblue"
        Color for the mean line.
    band_alpha : float, default 0.25
        Transparency for the shaded band.
    show_individual : bool, default False
        If True, draw each individual series behind the statistics.
    individual_alpha : float, default 0.15
        Alpha for individual series lines.
    title : str, optional
        Plot title.
    ylabel : str, optional
        Y-axis label.
    figsize : tuple, default (12, 6)
        Figure size in inches.

    Returns
    -------
    tuple
        (Figure, Axes) matplotlib objects.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    # Stack all series values (assume same time axis)
    series_list = list(collection.series.values())
    if not series_list:
        return fig, ax

    times = series_list[0].times
    all_values = np.column_stack([s.values for s in series_list])

    mean_vals = np.nanmean(all_values, axis=1)

    if show_individual:
        for s in series_list:
            ax.plot(s.times, s.values, color="gray", alpha=individual_alpha, linewidth=0.5)

    ax.plot(times, mean_vals, color=mean_color, linewidth=2, label="Mean", zorder=5)

    if band == "minmax":
        lo = np.nanmin(all_values, axis=1)
        hi = np.nanmax(all_values, axis=1)
        ax.fill_between(times, lo, hi, alpha=band_alpha, color=mean_color, label="Min/Max")
    else:
        std_vals = np.nanstd(all_values, axis=1)
        ax.fill_between(
            times,
            mean_vals - std_vals,
            mean_vals + std_vals,
            alpha=band_alpha,
            color=mean_color,
            label="\u00b11 Std Dev",
        )

    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xlabel("Date")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        framealpha=0.9,
        edgecolor="lightgray",
    )
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.subplots_adjust(right=0.80)

    return fig, ax


def plot_dual_axis(
    ts1: TimeSeries,
    ts2: TimeSeries,
    ax: Axes | None = None,
    color1: str = "steelblue",
    color2: str = "coral",
    style1: str = "-",
    style2: str = "-",
    label1: str | None = None,
    label2: str | None = None,
    ylabel1: str | None = None,
    ylabel2: str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (12, 6),
) -> tuple[Figure, tuple[Axes, Axes]]:
    """
    Dual y-axis comparison of two time series.

    Parameters
    ----------
    ts1, ts2 : TimeSeries
        The two time series to plot.
    ax : Axes, optional
        Primary axes. If None, a new figure is created.
    color1, color2 : str
        Colors for the two series.
    style1, style2 : str
        Line styles (e.g., "-", "--", "o-").
    label1, label2 : str, optional
        Legend labels. Falls back to ``ts.name``.
    ylabel1, ylabel2 : str, optional
        Y-axis labels. Falls back to ``ts.units``.
    title : str, optional
        Plot title.
    figsize : tuple, default (12, 6)
        Figure size in inches.

    Returns
    -------
    tuple
        (Figure, (Axes_left, Axes_right)).
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    lbl1 = label1 or getattr(ts1, "name", "Series 1")
    lbl2 = label2 or getattr(ts2, "name", "Series 2")

    ax.plot(ts1.times, ts1.values, style1, color=color1, label=lbl1)
    ax.set_ylabel(ylabel1 or str(getattr(ts1, "units", "")), color=color1)
    ax.tick_params(axis="y", labelcolor=color1)

    ax2 = ax.twinx()
    ax2.plot(ts2.times, ts2.values, style2, color=color2, label=lbl2)
    ax2.set_ylabel(ylabel2 or str(getattr(ts2, "units", "")), color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    if title:
        ax.set_title(title)
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left",
        bbox_to_anchor=(1.12, 1),
        borderaxespad=0,
        framealpha=0.9,
        edgecolor="lightgray",
    )

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.subplots_adjust(right=0.78)

    return fig, (ax, ax2)


def plot_streamflow_hydrograph(
    times: NDArray[np.datetime64],
    flows: NDArray[np.float64],
    baseflow: NDArray[np.float64] | None = None,
    ax: Axes | None = None,
    flow_color: str = "steelblue",
    baseflow_color: str = "darkorange",
    fill_alpha: float = 0.3,
    log_scale: bool = False,
    title: str = "Streamflow Hydrograph",
    ylabel: str = "Flow",
    units: str = "cfs",
    figsize: tuple[float, float] = (14, 6),
) -> tuple[Figure, Axes]:
    """
    Plot streamflow hydrograph with optional baseflow separation.

    Parameters
    ----------
    times : ndarray
        Datetime array for x-axis.
    flows : ndarray
        Total streamflow values.
    baseflow : ndarray, optional
        Baseflow component. If provided, the area between total flow
        and baseflow is shaded to highlight the quickflow component.
    ax : Axes, optional
        Existing axes to plot on.
    flow_color : str, default "steelblue"
        Color for the total flow line.
    baseflow_color : str, default "darkorange"
        Color for the baseflow line.
    fill_alpha : float, default 0.3
        Alpha for the shaded quickflow area.
    log_scale : bool, default False
        If True, use log scale for the y-axis.
    title : str, default "Streamflow Hydrograph"
        Plot title.
    ylabel : str, default "Flow"
        Y-axis label prefix.
    units : str, default "cfs"
        Flow units appended to ylabel.
    figsize : tuple, default (14, 6)
        Figure size in inches.

    Returns
    -------
    tuple
        (Figure, Axes) matplotlib objects.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    ax.plot(times, flows, color=flow_color, linewidth=1.5, label="Total Flow")
    ax.fill_between(times, 0, flows, alpha=fill_alpha * 0.5, color=flow_color)

    if baseflow is not None:
        ax.plot(times, baseflow, color=baseflow_color, linewidth=1.5, label="Baseflow")
        ax.fill_between(
            times, baseflow, flows, alpha=fill_alpha, color=flow_color, label="Quickflow"
        )
        ax.fill_between(times, 0, baseflow, alpha=fill_alpha, color=baseflow_color)

    if log_scale:
        ax.set_yscale("log")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel(f"{ylabel} ({units})", fontsize=11)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        framealpha=0.9,
        edgecolor="lightgray",
    )
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.subplots_adjust(right=0.80)

    return fig, ax


# =============================================================================
# Cross-Section Plotting Functions
# =============================================================================


def plot_cross_section(
    cross_section: CrossSection,
    ax: Axes | None = None,
    layer_colors: Sequence[str] | None = None,
    layer_labels: Sequence[str] | None = None,
    scalar_name: str | None = None,
    layer_property_name: str | None = None,
    layer_property_cmap: str = "viridis",
    show_ground_surface: bool = True,
    alpha: float = 0.7,
    title: str | None = None,
    figsize: tuple[float, float] = (14, 6),
) -> tuple[Figure, Axes]:
    """
    Plot a cross-section through an IWFM model.

    Supports three rendering modes that can be combined:

    1. **Default** (no property): Layers filled with flat colors via
       ``fill_between``.
    2. **Layer property** (``layer_property_name``): Each layer band is
       color-mapped by a per-layer property (e.g. hydraulic conductivity).
    3. **Scalar overlay** (``scalar_name``): A dashed line showing a
       per-sample scalar value (e.g. water table elevation).

    Parameters
    ----------
    cross_section : CrossSection
        Cross-section data from :class:`CrossSectionExtractor`.
    ax : Axes, optional
        Existing axes to plot on. Creates a new figure if None.
    layer_colors : sequence of str, optional
        Fill colors for each layer (used when ``layer_property_name``
        is None). Defaults to brown tones.
    layer_labels : sequence of str, optional
        Legend labels for each layer. Defaults to "Layer 1", "Layer 2", etc.
    scalar_name : str, optional
        Key into ``cross_section.scalar_values`` to overlay as a line.
    layer_property_name : str, optional
        Key into ``cross_section.layer_properties`` to color-map layers.
    layer_property_cmap : str, default "viridis"
        Colormap used for layer property rendering.
    show_ground_surface : bool, default True
        Draw the ground surface as a green line.
    alpha : float, default 0.7
        Fill transparency.
    title : str, optional
        Plot title.
    figsize : tuple, default (14, 6)
        Figure size in inches.

    Returns
    -------
    tuple
        ``(Figure, Axes)`` matplotlib objects.
    """

    import matplotlib.colors as mcolors

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    n_layers = cross_section.n_layers
    dist = cross_section.distance

    default_colors = ["#8B4513", "#D2691E", "#DEB887", "#F5DEB3", "#C4A882", "#A0826D"]
    if layer_colors is None:
        layer_colors = default_colors
    if layer_labels is None:
        layer_labels = [f"Layer {i + 1}" for i in range(n_layers)]

    # Mask NaN regions for clean rendering
    valid = cross_section.mask

    if layer_property_name is not None and layer_property_name in cross_section.layer_properties:
        # Color-mapped layer rendering using pcolormesh per layer
        prop = cross_section.layer_properties[layer_property_name]
        prop_valid = np.where(valid[:, np.newaxis], prop, np.nan)

        vmin = float(np.nanmin(prop_valid))
        vmax = float(np.nanmax(prop_valid))
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap(layer_property_cmap)

        # For each layer, create a vertical color strip
        for layer in range(n_layers - 1, -1, -1):
            top_vals = cross_section.top_elev[:, layer].copy()
            bot_vals = cross_section.bottom_elev[:, layer].copy()
            prop_vals = prop_valid[:, layer]

            # Render each segment as a filled polygon colored by property
            for j in range(len(dist) - 1):
                if not valid[j] or not valid[j + 1]:
                    continue
                x_seg = [dist[j], dist[j + 1], dist[j + 1], dist[j]]
                y_seg = [bot_vals[j], bot_vals[j + 1], top_vals[j + 1], top_vals[j]]
                avg_prop = 0.5 * (prop_vals[j] + prop_vals[j + 1])
                if np.isnan(avg_prop):
                    continue
                color = cmap(norm(avg_prop))
                ax.fill(x_seg, y_seg, color=color, alpha=alpha)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label=layer_property_name)

    else:
        # Flat-color layer rendering
        for layer in range(n_layers - 1, -1, -1):
            top_vals = cross_section.top_elev[:, layer].copy()
            bot_vals = cross_section.bottom_elev[:, layer].copy()
            color = layer_colors[layer % len(layer_colors)]
            label = layer_labels[layer] if layer < len(layer_labels) else f"Layer {layer + 1}"

            ax.fill_between(
                dist,
                bot_vals,
                top_vals,
                where=valid.tolist(),
                alpha=alpha,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                label=label,
            )

    # Ground surface line
    if show_ground_surface:
        gs_plot = cross_section.gs_elev.copy()
        gs_plot[~valid] = np.nan
        ax.plot(dist, gs_plot, "g-", linewidth=2, label="Ground Surface")

    # Scalar overlay (e.g. head)
    if scalar_name is not None and scalar_name in cross_section.scalar_values:
        sv = cross_section.scalar_values[scalar_name].copy()
        sv[~valid] = np.nan
        ax.plot(dist, sv, "b--", linewidth=2, label=scalar_name)

    ax.set_xlabel("Distance", fontsize=11)
    ax.set_ylabel("Elevation", fontsize=11)
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        framealpha=0.9,
        edgecolor="lightgray",
    )
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)
    fig.tight_layout()
    fig.subplots_adjust(right=0.80)

    return fig, ax


def plot_cross_section_location(
    grid: AppGrid,
    cross_section: CrossSection,
    ax: Axes | None = None,
    line_color: str = "red",
    line_width: float = 2.5,
    mesh_alpha: float = 0.3,
    show_labels: bool = True,
    figsize: tuple[float, float] = (10, 8),
) -> tuple[Figure, Axes]:
    """
    Plot a plan-view map showing the cross-section line on the mesh.

    Parameters
    ----------
    grid : AppGrid
        Model mesh.
    cross_section : CrossSection
        Cross-section whose path will be drawn.
    ax : Axes, optional
        Existing axes. Creates a new figure if None.
    line_color : str, default "red"
        Color of the cross-section line.
    line_width : float, default 2.5
        Width of the cross-section line.
    mesh_alpha : float, default 0.3
        Transparency of the mesh underlay.
    show_labels : bool, default True
        Show A / A' labels at line endpoints.
    figsize : tuple, default (10, 8)
        Figure size in inches.

    Returns
    -------
    tuple
        ``(Figure, Axes)`` matplotlib objects.
    """
    # Draw the mesh first
    fig, ax = plot_mesh(grid, ax=ax, alpha=mesh_alpha, figsize=figsize)

    # Draw the cross-section line
    if cross_section.waypoints is not None:
        wx = [p[0] for p in cross_section.waypoints]
        wy = [p[1] for p in cross_section.waypoints]
    else:
        wx = [cross_section.start[0], cross_section.end[0]]
        wy = [cross_section.start[1], cross_section.end[1]]

    ax.plot(wx, wy, color=line_color, linewidth=line_width, zorder=10)

    if show_labels:
        ax.annotate(
            "A",
            xy=(wx[0], wy[0]),
            fontsize=14,
            fontweight="bold",
            color=line_color,
            ha="center",
            va="bottom",
            zorder=11,
        )
        ax.annotate(
            "A'",
            xy=(wx[-1], wy[-1]),
            fontsize=14,
            fontweight="bold",
            color=line_color,
            ha="center",
            va="bottom",
            zorder=11,
        )

    fig.tight_layout()
    return fig, ax
