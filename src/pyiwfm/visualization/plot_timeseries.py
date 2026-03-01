"""Time series plotting functions for IWFM models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

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

if TYPE_CHECKING:
    from pyiwfm.core.timeseries import TimeSeries, TimeSeriesCollection


@_with_style(CHART_STYLE)
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
        ax.set_title(title)

    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    elif series_list and series_list[0].units:
        ax.set_ylabel(series_list[0].units)

    if legend and len(series_list) > 1:
        fig.legend(loc="outside upper right")

    if not grid:
        ax.grid(False)

    # Date formatting
    if date_format:
        ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    _rotate_date_labels(ax)

    return fig, ax


@_with_style(CHART_STYLE)
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

    fig.legend(loc="outside upper right")

    if title:
        ax.set_title(title)

    if observed.units:
        ax.set_ylabel(observed.units)

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
        ax2.set_ylabel("Residual")
        ax2.set_xlabel("Date")
    else:
        ax.set_xlabel("Date")

    _rotate_date_labels(ax)

    return fig, ax


@_with_style(CHART_STYLE)
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


@_with_style(CHART_STYLE)
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
    fig.legend(loc="outside upper right")
    _rotate_date_labels(ax)

    return fig, ax


@_with_style(CHART_STYLE)
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
    # Chart style hides right spine; restore it for the twin axis
    ax2.spines["right"].set_visible(True)
    ax2.plot(ts2.times, ts2.values, style2, color=color2, label=lbl2)
    ax2.set_ylabel(ylabel2 or str(getattr(ts2, "units", "")), color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    if title:
        ax.set_title(title)
    ax.set_xlabel("Date")

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="outside upper right",
    )

    _rotate_date_labels(ax)

    return fig, (ax, ax2)


@_with_style(CHART_STYLE)
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

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{ylabel} ({units})")
    fig.legend(loc="outside upper right")
    _rotate_date_labels(ax)

    return fig, ax
