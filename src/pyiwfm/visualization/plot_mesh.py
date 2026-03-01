"""Mesh and spatial plotting functions for IWFM models."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from pyiwfm.visualization._plot_utils import (  # noqa: E402
    SPATIAL_STYLE,
    _format_thousands,
    _with_style,
)

if TYPE_CHECKING:
    from pyiwfm.components.lake import AppLake
    from pyiwfm.components.stream import AppStream
    from pyiwfm.core.mesh import AppGrid


@_with_style(SPATIAL_STYLE)
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
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    _format_thousands(ax)

    return fig, ax


@_with_style(SPATIAL_STYLE)
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
        fig.legend(loc="outside right upper")

    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    _format_thousands(ax)

    return fig, ax


@_with_style(SPATIAL_STYLE)
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
            fig.legend(handles=legend_patches, loc="outside right upper")

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
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    _format_thousands(ax)

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
    unchanged. Fully vectorized -- no Python loop over elements.

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


@_with_style(SPATIAL_STYLE)
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
        fig.colorbar(tcf, ax=ax)

    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    _format_thousands(ax)

    return fig, ax


@_with_style(SPATIAL_STYLE)
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
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    _format_thousands(ax)

    return fig, ax


@_with_style(SPATIAL_STYLE)
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
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    _format_thousands(ax)

    return fig, ax


@_with_style(SPATIAL_STYLE)
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
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    _format_thousands(ax)

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
