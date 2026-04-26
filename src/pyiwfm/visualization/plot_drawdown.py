"""Drawdown time-series plots (Phase 2 drawdown).

Operates on a :class:`~pyiwfm.io.drawdown.DrawdownTimeSeriesReport`
(per-location time series). All functions accept an optional ``ax`` and
return the populated :class:`matplotlib.axes.Axes`.

For spatial maps (cone of depression, max-drawdown map) see
:mod:`pyiwfm.visualization.map_drawdown`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

if TYPE_CHECKING:
    from pyiwfm.io.drawdown import DrawdownTimeSeriesReport


def _resolve_locations(
    report: DrawdownTimeSeriesReport,
    location_keys: Sequence[tuple[int, int]] | None,
) -> list:
    """Return the subset of ``report.locations`` matching ``location_keys``.

    When ``location_keys`` is ``None``, return every location. Otherwise
    filter to the (node_id, layer) pairs present in ``location_keys``.
    Raises :class:`ValueError` listing missing pairs if any requested
    pair isn't in the report.
    """
    if location_keys is None:
        return list(report.locations)
    available = {(loc.node_id, loc.layer) for loc in report.locations}
    requested = list(location_keys)
    missing = [k for k in requested if k not in available]
    if missing:
        raise ValueError(f"location_keys {missing} not in report; available: {sorted(available)}")
    selected_set = set(requested)
    return [loc for loc in report.locations if (loc.node_id, loc.layer) in selected_set]


def plot_drawdown_timeseries(
    report: DrawdownTimeSeriesReport,
    location_keys: Sequence[tuple[int, int]] | None = None,
    ax: Axes | None = None,
    *,
    title: str = "Drawdown time series",
) -> Axes:
    """Plot drawdown vs time, one line per location.

    Parameters
    ----------
    report
        A :class:`~pyiwfm.io.drawdown.DrawdownTimeSeriesReport` produced
        by :meth:`DrawdownComputer.build_timeseries_report`.
    location_keys
        Optional iterable of ``(node_id, layer)`` pairs to plot. When
        ``None`` (default), plots every location.
    ax
        Existing matplotlib Axes to draw on. When ``None``, creates one.
    title
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    ValueError
        Any requested ``(node_id, layer)`` pair is not in the report.
    """
    selected = _resolve_locations(report, location_keys)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    times_idx = np.arange(report.n_timesteps)
    for loc in selected:
        ax.plot(
            times_idx,
            loc.drawdown,
            label=f"node {loc.node_id}, layer {loc.layer}",
        )
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("timestep")
    ax.set_ylabel("drawdown (positive = water-level decline)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if selected:
        ax.legend(loc="best", fontsize="small")
    return ax


def plot_drawdown_summary_bar(
    report: DrawdownTimeSeriesReport,
    metric: Literal["max", "final"] = "max",
    top_n: int = 10,
    ax: Axes | None = None,
    *,
    title: str | None = None,
) -> Axes:
    """Horizontal bar chart of the top-N locations by chosen drawdown metric.

    Parameters
    ----------
    report
        The drawdown report.
    metric
        ``"max"`` (largest absolute drawdown across timesteps) or
        ``"final"`` (drawdown at the last timestep).
    top_n
        Show only the top N locations. Use ``len(report.locations)`` to
        include all.
    ax
        Existing axes.
    title
        Optional title; defaults to ``"Top N locations by <metric> drawdown"``.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    ValueError
        Unknown metric, or non-positive ``top_n``.
    """
    if metric not in ("max", "final"):
        raise ValueError(f"metric must be 'max' or 'final', got {metric!r}")
    if top_n <= 0:
        raise ValueError(f"top_n must be positive, got {top_n}")

    key = "max_drawdown" if metric == "max" else "final_drawdown"
    # NaN locations sort last; sort by absolute value descending
    sorted_locs = sorted(
        report.locations,
        key=lambda loc: -abs(getattr(loc, key)) if np.isfinite(getattr(loc, key)) else float("inf"),
    )[:top_n]

    if ax is None:
        height = max(3.0, 0.35 * len(sorted_locs) + 1.5)
        _, ax = plt.subplots(figsize=(10, height))

    labels = [f"node {loc.node_id}, layer {loc.layer}" for loc in sorted_locs]
    values = [getattr(loc, key) if np.isfinite(getattr(loc, key)) else 0.0 for loc in sorted_locs]

    ax.barh(labels, values, color="firebrick")
    ax.invert_yaxis()
    ax.set_xlabel(f"{metric} drawdown")
    ax.set_title(title or f"Top {top_n} locations by {metric} drawdown")
    ax.grid(True, axis="x", alpha=0.3)
    return ax
