"""Budget plotting functions for IWFM models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from pyiwfm.visualization._plot_utils import (  # noqa: E402
    CHART_STYLE,
    _rotate_date_labels,
    _with_style,
)


@_with_style(CHART_STYLE)
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
        ax.set_ylabel(f"Volume ({units})")
        ax.axhline(y=0, color="black", linewidth=0.8)
        # Style enables grid on both axes; we only want y-axis grid
        ax.xaxis.grid(False)

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
        ax.set_xlabel(f"Volume ({units})")
        ax.axvline(x=0, color="black", linewidth=0.8)
        # Style enables grid on both axes; we only want x-axis grid
        ax.yaxis.grid(False)

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

    ax.set_title(title)

    return fig, ax


@_with_style(CHART_STYLE)
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
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Flow Rate ({units})")
    ax.set_title(title)

    if show_legend:
        fig.legend(loc="outside upper right")

    _rotate_date_labels(ax)

    return fig, ax


@_with_style(CHART_STYLE)
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
        ax.set_title(title)

    return fig, ax


@_with_style(CHART_STYLE)
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
    ax.set_xlabel(f"Volume ({units})")
    ax.set_title(title)
    # Style enables grid on both axes; we only want x-axis grid
    ax.yaxis.grid(False)

    # Summary text
    summary = (
        f"Total In: {total_in:,.0f} {units}\n"
        f"Total Out: {total_out:,.0f} {units}\n"
        f"\u0394Storage: {storage_change:,.0f} {units}\n"
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

    return fig, ax


@_with_style(CHART_STYLE)
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

        # No grid on heatmap
        ax.grid(False)

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
        ax.set_ylabel(f"Volume ({units})")
        ax.axhline(y=0, color="black", linewidth=0.8)
        fig.legend(loc="outside upper right")
        # Style enables grid on both axes; we only want y-axis grid
        ax.xaxis.grid(False)

    ax.set_title(title)

    return fig, ax


@_with_style(CHART_STYLE)
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
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Date")
    ax.set_title(title)
    fig.legend(loc="outside upper right")
    ax.axhline(y=0, color="black", linewidth=0.5)

    _rotate_date_labels(ax)

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
