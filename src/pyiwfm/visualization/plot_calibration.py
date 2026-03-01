"""Calibration and validation plotting functions for IWFM models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from pyiwfm.visualization._plot_utils import (  # noqa: E402
    CHART_STYLE,
    SPATIAL_STYLE,
    _format_thousands,
    _with_style,
)
from pyiwfm.visualization.plot_mesh import plot_mesh  # noqa: E402

if TYPE_CHECKING:
    from pyiwfm.components.stream import AppStream
    from pyiwfm.core.cross_section import CrossSection
    from pyiwfm.core.mesh import AppGrid


@_with_style(SPATIAL_STYLE)
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
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    _format_thousands(ax)

    return fig, ax


@_with_style(CHART_STYLE)
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
        cmap_obj = plt.get_cmap(layer_property_cmap)

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
                color = cmap_obj(norm(avg_prop))
                ax.fill(x_seg, y_seg, color=color, alpha=alpha)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
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

    ax.set_xlabel("Distance")
    ax.set_ylabel("Elevation")
    if title:
        ax.set_title(title)
    fig.legend(loc="outside upper right")

    return fig, ax


@_with_style(SPATIAL_STYLE)
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

    return fig, ax


@_with_style(CHART_STYLE)
def plot_one_to_one(
    observed: NDArray[np.float64],
    simulated: NDArray[np.float64],
    ax: Axes | None = None,
    color_by: NDArray[np.float64] | None = None,
    show_metrics: bool = True,
    show_identity: bool = True,
    show_regression: bool = True,
    title: str | None = None,
    units: str = "",
    figsize: tuple[float, float] = (8, 8),
) -> tuple[Figure, Axes]:
    """Plot a 1:1 comparison of observed vs simulated values.

    Parameters
    ----------
    observed : NDArray[np.float64]
        Observed values.
    simulated : NDArray[np.float64]
        Simulated values.
    ax : Axes | None
        Existing axes to plot on.
    color_by : NDArray[np.float64] | None
        Optional array to color scatter points by.
    show_metrics : bool
        Show a text box with RMSE, NSE, etc.
    show_identity : bool
        Show the 1:1 identity line.
    show_regression : bool
        Show a linear regression line.
    title : str | None
        Plot title.
    units : str
        Unit label for axes.
    figsize : tuple[float, float]
        Figure size.

    Returns
    -------
    tuple[Figure, Axes]
        The figure and axes.
    """
    from pyiwfm.comparison.metrics import ComparisonMetrics

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    # Remove NaNs
    mask = ~(np.isnan(observed) | np.isnan(simulated))
    obs = observed[mask]
    sim = simulated[mask]

    # Scatter plot
    if color_by is not None:
        cb = color_by[mask]
        scatter = ax.scatter(obs, sim, c=cb, s=20, alpha=0.7, edgecolors="none")
        fig.colorbar(scatter, ax=ax, shrink=0.8)
    else:
        ax.scatter(obs, sim, s=20, alpha=0.7, edgecolors="none", color="steelblue")

    # Axis limits (equal, with padding)
    all_vals = np.concatenate([obs, sim])
    vmin, vmax = float(np.min(all_vals)), float(np.max(all_vals))
    pad = (vmax - vmin) * 0.05
    lims = (vmin - pad, vmax + pad)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")

    # 1:1 identity line
    if show_identity:
        ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5, label="1:1")

    # Regression line
    if show_regression and len(obs) > 1:
        coeffs = np.polyfit(obs, sim, 1)
        x_fit = np.array(lims)
        y_fit = np.polyval(coeffs, x_fit)
        ax.plot(x_fit, y_fit, "r-", linewidth=1, alpha=0.7, label="Regression")

    # Metrics text box
    if show_metrics and len(obs) > 1:
        metrics = ComparisonMetrics.compute(obs, sim)
        text = (
            f"N = {metrics.n_points}\n"
            f"RMSE = {metrics.rmse:.2f}\n"
            f"SRMSE = {metrics.scaled_rmse:.3f}\n"
            f"ME = {metrics.mbe:.2f}\n"
            f"NSE = {metrics.nash_sutcliffe:.3f}\n"
            f"r = {metrics.correlation:.3f}"
        )
        ax.text(
            0.05,
            0.95,
            text,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "wheat", "alpha": 0.8},
        )

    unit_str = f" ({units})" if units else ""
    ax.set_xlabel(f"Observed{unit_str}")
    ax.set_ylabel(f"Simulated{unit_str}")
    if title:
        ax.set_title(title)

    return fig, ax


@_with_style(CHART_STYLE)
def plot_residual_cdf(
    residuals: NDArray[np.float64],
    ax: Axes | None = None,
    show_percentile_lines: bool = True,
    title: str = "Cumulative Frequency of Residuals",
    figsize: tuple[float, float] = (8, 6),
) -> tuple[Figure, Axes]:
    """Plot an empirical CDF (cumulative frequency) of residuals.

    Parameters
    ----------
    residuals : NDArray[np.float64]
        Array of residual values (simulated - observed).
    ax : Axes | None
        Existing axes to plot on.
    show_percentile_lines : bool, default True
        Draw reference lines at the 5th and 95th percentiles and
        a vertical line at zero.
    title : str, default "Cumulative Frequency of Residuals"
        Plot title.
    figsize : tuple[float, float], default (8, 6)
        Figure size in inches.

    Returns
    -------
    tuple[Figure, Axes]
        The figure and axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    # Remove NaN
    clean = residuals[~np.isnan(residuals)]
    sorted_vals = np.sort(clean)
    n = len(sorted_vals)
    cdf = np.arange(1, n + 1) / n

    ax.plot(sorted_vals, cdf, linewidth=1.5, color="steelblue")

    if show_percentile_lines and n > 0:
        p5 = float(np.percentile(sorted_vals, 5))
        p95 = float(np.percentile(sorted_vals, 95))
        ax.axhline(0.05, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.axhline(0.95, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axvline(
            p5, color="red", linestyle="--", linewidth=0.8, alpha=0.6, label=f"5th: {p5:.2f}"
        )
        ax.axvline(
            p95, color="red", linestyle="--", linewidth=0.8, alpha=0.6, label=f"95th: {p95:.2f}"
        )
        ax.legend(fontsize=8)

    ax.set_xlabel("Residual")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title(title)
    ax.set_ylim(0, 1)

    return fig, ax


@_with_style(SPATIAL_STYLE)
def plot_spatial_bias(
    grid: AppGrid,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    bias: NDArray[np.float64],
    ax: Axes | None = None,
    show_mesh: bool = True,
    cmap: str = "RdBu_r",
    symmetric_colorbar: bool = True,
    title: str = "Spatial Bias",
    units: str = "",
    figsize: tuple[float, float] = (10, 8),
) -> tuple[Figure, Axes]:
    """Plot spatial bias (simulated - observed) at observation locations.

    Parameters
    ----------
    grid : AppGrid
        Model grid for background mesh.
    x : NDArray[np.float64]
        X coordinates of observation points.
    y : NDArray[np.float64]
        Y coordinates of observation points.
    bias : NDArray[np.float64]
        Bias values (simulated - observed).
    ax : Axes | None
        Existing axes to plot on.
    show_mesh : bool
        Show the mesh as background.
    cmap : str
        Colormap name (should be diverging).
    symmetric_colorbar : bool
        Center the colorbar at 0.
    title : str
        Plot title.
    units : str
        Unit label for colorbar.
    figsize : tuple[float, float]
        Figure size.

    Returns
    -------
    tuple[Figure, Axes]
        The figure and axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    # Plot mesh background
    if show_mesh:
        plot_mesh(grid, ax=ax, alpha=0.1, edge_width=0.3, fill_color="whitesmoke")

    # Determine color limits
    if symmetric_colorbar:
        vmax = float(np.max(np.abs(bias)))
        vmin = -vmax
    else:
        vmin = float(np.min(bias))
        vmax = float(np.max(bias))

    scatter = ax.scatter(
        x,
        y,
        c=bias,
        s=60,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors="black",
        linewidths=0.5,
        zorder=5,
    )

    unit_str = f" ({units})" if units else ""
    fig.colorbar(scatter, ax=ax, label=f"Bias{unit_str}", shrink=0.8)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    _format_thousands(ax)

    return fig, ax
