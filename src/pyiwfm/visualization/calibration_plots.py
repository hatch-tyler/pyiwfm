"""
Publication-quality composite figures for IWFM model calibration.

This module builds on the individual plotting functions in
:mod:`pyiwfm.visualization.plotting` to create multi-panel figures
suitable for reports and journal articles.

Functions
---------
- :func:`plot_calibration_summary` — Multi-panel summary (1:1, spatial bias, histograms)
- :func:`plot_hydrograph_panel` — Grid of observed vs simulated hydrographs
- :func:`plot_metrics_table` — Metrics table rendered as a figure
- :func:`plot_residual_histogram` — Residual distribution histogram
- :func:`plot_water_budget_summary` — Stacked budget components over time
- :func:`plot_zbudget_summary` — Zone budget summary
- :func:`plot_cluster_map` — Spatial map of cluster memberships
- :func:`plot_typical_hydrographs` — Typical hydrograph curves by cluster
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

matplotlib.use("Agg")

if TYPE_CHECKING:
    from pyiwfm.calibration.calctyphyd import CalcTypHydResult
    from pyiwfm.calibration.clustering import ClusteringResult
    from pyiwfm.comparison.metrics import ComparisonMetrics

_STYLES_DIR = Path(__file__).parent / "styles"
PUBLICATION_STYLE = str(_STYLES_DIR / "pyiwfm-publication.mplstyle")


def _with_pub_style(func: Any) -> Any:
    """Decorator that wraps a plotting function in publication style context."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with plt.style.context(PUBLICATION_STYLE):
            return func(*args, **kwargs)

    return wrapper


@_with_pub_style
def plot_calibration_summary(
    well_comparisons: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
    grid: Any | None = None,
    output_dir: Path | None = None,
    dpi: int = 300,
) -> list[Figure]:
    """Create multi-panel calibration summary figures.

    Parameters
    ----------
    well_comparisons : dict[str, tuple[NDArray, NDArray]]
        Mapping of well ID to (observed, simulated) value arrays.
    grid : AppGrid | None
        Model grid for spatial plots.
    output_dir : Path | None
        If provided, save figures to this directory.
    dpi : int
        Output resolution.

    Returns
    -------
    list[Figure]
        List of generated figures.
    """

    figures: list[Figure] = []

    # Collect all obs/sim pairs
    all_obs = np.concatenate([obs for obs, _ in well_comparisons.values()])
    all_sim = np.concatenate([sim for _, sim in well_comparisons.values()])

    # --- Figure 1: 1:1 plot + residual histogram ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax1, ax2 = axes

    # 1:1 plot
    from pyiwfm.visualization.plotting import plot_one_to_one

    plot_one_to_one(all_obs, all_sim, ax=ax1, title="Observed vs Simulated")

    # Residual histogram
    residuals = all_sim - all_obs
    plot_residual_histogram(residuals, ax=ax2)

    fig.suptitle("Calibration Summary", fontsize=14, fontweight="bold")
    figures.append(fig)

    if output_dir:
        fig.savefig(output_dir / "calibration_summary.png", dpi=dpi)

    return figures


@_with_pub_style
def plot_hydrograph_panel(
    comparisons: dict[str, tuple[NDArray[np.datetime64], NDArray[np.float64], NDArray[np.float64]]],
    n_cols: int = 3,
    max_panels: int = 12,
    output_path: Path | None = None,
    dpi: int = 300,
) -> Figure:
    """Plot a grid of observed vs simulated hydrographs.

    Parameters
    ----------
    comparisons : dict[str, tuple[NDArray, NDArray, NDArray]]
        Mapping of well ID to (times, observed, simulated).
    n_cols : int
        Number of columns in the grid.
    max_panels : int
        Maximum number of panels to show.
    output_path : Path | None
        If provided, save the figure.
    dpi : int
        Output resolution.

    Returns
    -------
    Figure
        The generated figure.
    """
    well_ids = list(comparisons.keys())[:max_panels]
    n_panels = len(well_ids)
    n_rows = (n_panels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes_flat = [axes]
    else:
        axes_flat = list(np.array(axes).flat)

    for i, well_id in enumerate(well_ids):
        ax = axes_flat[i]
        times, obs, sim = comparisons[well_id]
        ax.plot(times, obs, "ko", markersize=3, label="Observed", alpha=0.7)
        ax.plot(times, sim, "b-", linewidth=1, label="Simulated", alpha=0.8)
        ax.set_title(well_id, fontsize=9)
        if i == 0:
            ax.legend(fontsize=7)

    # Hide unused axes
    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Hydrograph Comparison", fontsize=14, fontweight="bold")

    if output_path:
        fig.savefig(output_path, dpi=dpi)

    return fig


@_with_pub_style
def plot_metrics_table(
    metrics_by_well: dict[str, ComparisonMetrics],
    output_path: Path | None = None,
    dpi: int = 300,
) -> Figure:
    """Render calibration metrics as a table figure.

    Parameters
    ----------
    metrics_by_well : dict[str, ComparisonMetrics]
        Mapping of well ID to computed metrics.
    output_path : Path | None
        If provided, save the figure.
    dpi : int
        Output resolution.

    Returns
    -------
    Figure
        The generated figure.
    """
    well_ids = list(metrics_by_well.keys())
    columns = ["RMSE", "SRMSE", "MAE", "MBE", "NSE", "r", "N"]

    cell_text: list[list[str]] = []
    for wid in well_ids:
        m = metrics_by_well[wid]
        cell_text.append(
            [
                f"{m.rmse:.2f}",
                f"{m.scaled_rmse:.3f}",
                f"{m.mae:.2f}",
                f"{m.mbe:.2f}",
                f"{m.nash_sutcliffe:.3f}",
                f"{m.correlation:.3f}",
                str(m.n_points),
            ]
        )

    fig, ax = plt.subplots(figsize=(10, 0.5 + 0.4 * len(well_ids)))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        rowLabels=well_ids,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)

    ax.set_title("Calibration Metrics by Well", fontsize=12, fontweight="bold", pad=20)

    if output_path:
        fig.savefig(output_path, dpi=dpi)

    return fig


@_with_pub_style
def plot_residual_histogram(
    residuals: NDArray[np.float64],
    ax: Axes | None = None,
    show_normal_fit: bool = True,
    figsize: tuple[float, float] = (8, 6),
) -> tuple[Figure, Axes]:
    """Plot a histogram of residuals (simulated - observed).

    Parameters
    ----------
    residuals : NDArray[np.float64]
        Residual values.
    ax : Axes | None
        Existing axes to plot on.
    show_normal_fit : bool
        Overlay a normal distribution fit.
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

    # Remove NaNs
    valid = residuals[~np.isnan(residuals)]

    ax.hist(valid, bins="auto", density=True, alpha=0.7, color="steelblue", edgecolor="white")

    if show_normal_fit and len(valid) > 2:
        mu = float(np.mean(valid))
        sigma = float(np.std(valid))
        if sigma > 0:
            x = np.linspace(float(np.min(valid)), float(np.max(valid)), 100)
            pdf = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
            ax.plot(
                x,
                pdf,
                "r-",
                linewidth=1.5,
                label=f"Normal fit\n$\\mu$={mu:.2f}, $\\sigma$={sigma:.2f}",
            )
            ax.legend(fontsize=8)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Residual (Simulated - Observed)")
    ax.set_ylabel("Density")
    ax.set_title("Residual Distribution")

    return fig, ax


@_with_pub_style
def plot_water_budget_summary(
    budget_data: dict[str, NDArray[np.float64]],
    times: NDArray[np.datetime64],
    output_path: Path | None = None,
    dpi: int = 300,
) -> Figure:
    """Plot stacked water budget summary over time.

    Parameters
    ----------
    budget_data : dict[str, NDArray[np.float64]]
        Mapping of component name to time series of values.
    times : NDArray[np.datetime64]
        Time stamps.
    output_path : Path | None
        If provided, save the figure.
    dpi : int
        Output resolution.

    Returns
    -------
    Figure
        The generated figure.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Separate inflows and outflows
    inflows = {k: v for k, v in budget_data.items() if np.nanmean(v) >= 0}
    outflows = {k: v for k, v in budget_data.items() if np.nanmean(v) < 0}

    # Plot inflows
    if inflows:
        labels = list(inflows.keys())
        values = np.array([inflows[k] for k in labels])
        axes[0].stackplot(times, values, labels=labels, alpha=0.8)
        axes[0].set_title("Inflows")
        axes[0].legend(fontsize=7, loc="upper right")

    # Plot outflows (as absolute values)
    if outflows:
        labels = list(outflows.keys())
        values = np.abs(np.array([outflows[k] for k in labels]))
        axes[1].stackplot(times, values, labels=labels, alpha=0.8)
        axes[1].set_title("Outflows")
        axes[1].legend(fontsize=7, loc="upper right")

    fig.suptitle("Water Budget Summary", fontsize=14, fontweight="bold")

    if output_path:
        fig.savefig(output_path, dpi=dpi)

    return fig


@_with_pub_style
def plot_zbudget_summary(
    zone_budgets: dict[str, dict[str, NDArray[np.float64]]],
    times: NDArray[np.datetime64],
    output_path: Path | None = None,
    dpi: int = 300,
) -> Figure:
    """Plot zone budget summary.

    Parameters
    ----------
    zone_budgets : dict[str, dict[str, NDArray[np.float64]]]
        Mapping of zone name to component budget data.
    times : NDArray[np.datetime64]
        Time stamps.
    output_path : Path | None
        If provided, save the figure.
    dpi : int
        Output resolution.

    Returns
    -------
    Figure
        The generated figure.
    """
    n_zones = len(zone_budgets)
    n_cols = min(3, n_zones)
    n_rows = (n_zones + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=True)
    if n_rows == 1 and n_cols == 1:
        axes_flat = [axes]
    else:
        axes_flat = list(np.array(axes).flat)

    for i, (zone_name, components) in enumerate(zone_budgets.items()):
        if i >= len(axes_flat):
            break
        ax = axes_flat[i]
        for comp_name, values in components.items():
            ax.plot(times, values, linewidth=1, label=comp_name)
        ax.set_title(zone_name, fontsize=9)
        ax.legend(fontsize=6)

    # Hide unused axes
    for j in range(n_zones, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Zone Budget Summary", fontsize=14, fontweight="bold")

    if output_path:
        fig.savefig(output_path, dpi=dpi)

    return fig


@_with_pub_style
def plot_cluster_map(
    well_locations: dict[str, tuple[float, float]],
    clustering_result: ClusteringResult,
    grid: Any | None = None,
    figsize: tuple[float, float] = (10, 8),
) -> tuple[Figure, Axes]:
    """Plot a spatial map of cluster memberships.

    Parameters
    ----------
    well_locations : dict[str, tuple[float, float]]
        Mapping of well ID to (x, y) coordinates.
    clustering_result : ClusteringResult
        Clustering result with membership matrix.
    grid : AppGrid | None
        Model grid for background mesh.
    figsize : tuple[float, float]
        Figure size.

    Returns
    -------
    tuple[Figure, Axes]
        The figure and axes.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot mesh background
    if grid is not None:
        from pyiwfm.visualization.plotting import plot_mesh

        plot_mesh(grid, ax=ax, alpha=0.1, edge_width=0.3, fill_color="whitesmoke")

    # Color by dominant cluster
    n_clusters = clustering_result.n_clusters
    cmap = plt.get_cmap("tab10", n_clusters)

    for i, well_id in enumerate(clustering_result.well_ids):
        if well_id not in well_locations:
            continue
        x, y = well_locations[well_id]
        dominant = int(np.argmax(clustering_result.membership[i]))
        max_membership = float(np.max(clustering_result.membership[i]))

        ax.scatter(
            x,
            y,
            c=[cmap(dominant)],
            s=60 * max_membership + 20,
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
        )

    # Legend
    for c in range(n_clusters):
        ax.scatter([], [], c=[cmap(c)], s=40, label=f"Cluster {c}")
    ax.legend(fontsize=8, loc="best")

    ax.set_title(f"Well Clusters (n={n_clusters}, FPC={clustering_result.fpc:.3f})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    return fig, ax


@_with_pub_style
def plot_typical_hydrographs(
    result: CalcTypHydResult,
    figsize: tuple[float, float] = (12, 6),
) -> tuple[Figure, Axes]:
    """Plot typical hydrographs by cluster.

    Parameters
    ----------
    result : CalcTypHydResult
        Typical hydrograph computation result.
    figsize : tuple[float, float]
        Figure size.

    Returns
    -------
    tuple[Figure, Axes]
        The figure and axes.
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.get_cmap("tab10")

    for th in result.hydrographs:
        valid = ~np.isnan(th.values)
        if not np.any(valid):
            continue
        label = f"Cluster {th.cluster_id} ({len(th.contributing_wells)} wells)"
        ax.plot(
            th.times[valid],
            th.values[valid],
            "o-",
            color=colors(th.cluster_id),
            linewidth=1.5,
            markersize=6,
            label=label,
        )

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Season")
    ax.set_ylabel("De-meaned Water Level")
    ax.set_title("Typical Hydrographs")
    ax.legend(fontsize=8)

    return fig, ax
