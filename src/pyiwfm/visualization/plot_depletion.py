"""Stream depletion plots (Phase 2.2.a-ii).

Time-series plots over a :class:`~pyiwfm.io.streams.depletion.StreamDepletionReport`.
All functions accept an optional ``ax`` and return the populated
:class:`matplotlib.axes.Axes` so callers can compose figures or restyle.

For spatial maps see :mod:`pyiwfm.visualization.map_depletion`.
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
    import pandas as pd

    from pyiwfm.io.streams.depletion import StreamDepletionReport


def _resolve_reaches(
    report: StreamDepletionReport,
    reach_ids: Sequence[int] | None,
) -> list:
    """Return the list of result objects for the requested reach IDs.

    When ``reach_ids`` is ``None``, return all reaches. Otherwise filter
    by 1-based ``reach_id``. Raises :class:`ValueError` for unknown IDs.
    """
    if reach_ids is None:
        return list(report.results)
    available = {r.reach_id for r in report.results}
    missing = [rid for rid in reach_ids if rid not in available]
    if missing:
        raise ValueError(f"reach_id(s) {missing} not in report; available: {sorted(available)}")
    return [r for r in report.results if r.reach_id in set(reach_ids)]


def plot_cumulative_depletion(
    report: StreamDepletionReport,
    reach_ids: Sequence[int] | None = None,
    ax: Axes | None = None,
    *,
    pumping_timeseries: pd.Series | Sequence[float] | None = None,
    title: str = "Cumulative stream depletion",
) -> Axes:
    """Plot cumulative depletion (volume) vs time, one line per reach.

    If ``pumping_timeseries`` is supplied (length must equal the report
    timestep count), it is overlaid on a secondary y-axis to highlight
    the depletion-from-pumping relationship — the canonical view for
    showing that depletion grows as pumping accumulates.

    Parameters
    ----------
    report
        The depletion report to plot.
    reach_ids
        Optional 1-based reach IDs to include. When ``None``, plots all.
    ax
        Existing matplotlib Axes to draw on. When ``None``, creates one.
    pumping_timeseries
        Optional system-wide pumping rate (length = ``n_timesteps``).
        Drawn on a secondary y-axis.
    title
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the depletion lines were drawn on.

    Raises
    ------
    ValueError
        If a requested ``reach_id`` isn't in the report or
        ``pumping_timeseries`` length doesn't match the report.
    """
    selected = _resolve_reaches(report, reach_ids)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    times_idx = np.arange(report.n_timesteps)
    for r in selected:
        ax.plot(times_idx, r.cumulative_depletion, label=f"{r.reach_name} (#{r.reach_id})")

    ax.set_xlabel("timestep")
    ax.set_ylabel("cumulative depletion")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize="small")

    if pumping_timeseries is not None:
        pump_arr = np.asarray(pumping_timeseries, dtype=np.float64)
        if pump_arr.shape != (report.n_timesteps,):
            raise ValueError(
                f"pumping_timeseries length {pump_arr.shape} does not match "
                f"report n_timesteps={report.n_timesteps}"
            )
        ax2 = ax.twinx()
        ax2.plot(
            times_idx,
            pump_arr,
            color="black",
            linestyle="--",
            alpha=0.6,
            label="pumping",
        )
        ax2.set_ylabel("pumping (rate)", color="black")
        ax2.tick_params(axis="y", labelcolor="black")

    return ax


def plot_depletion_timeseries(
    report: StreamDepletionReport,
    reach_ids: Sequence[int] | None = None,
    ax: Axes | None = None,
    *,
    title: str = "Stream depletion time series",
) -> Axes:
    """Plot instantaneous (per-timestep) depletion vs time, one line per reach.

    Useful for revealing seasonal patterns — pumping season vs recovery —
    that the cumulative view smooths out.
    """
    selected = _resolve_reaches(report, reach_ids)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    times_idx = np.arange(report.n_timesteps)
    for r in selected:
        ax.plot(times_idx, r.depletion, label=f"{r.reach_name} (#{r.reach_id})")
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("timestep")
    ax.set_ylabel("depletion (rate)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    return ax


def plot_depletion_summary_bar(
    report: StreamDepletionReport,
    metric: Literal["total", "max"] = "total",
    top_n: int = 10,
    ax: Axes | None = None,
    *,
    title: str | None = None,
) -> Axes:
    """Horizontal bar chart of the top-N reaches by chosen depletion metric.

    Parameters
    ----------
    metric
        ``"total"`` (cumulative depletion at the last timestep) or
        ``"max"`` (largest absolute depletion in any timestep).
    top_n
        Show only the top N reaches. Use ``len(report.results)`` to
        include all.
    ax
        Existing axes to draw on.
    title
        Optional title; defaults to ``"Top N reaches by <metric> depletion"``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if metric not in ("total", "max"):
        raise ValueError(f"metric must be 'total' or 'max', got {metric!r}")
    if top_n <= 0:
        raise ValueError(f"top_n must be positive, got {top_n}")

    key = "total_depletion" if metric == "total" else "max_depletion"
    sorted_results = sorted(
        report.results,
        key=lambda r: abs(getattr(r, key)),
        reverse=True,
    )[:top_n]

    if ax is None:
        height = max(3.0, 0.35 * len(sorted_results) + 1.5)
        _, ax = plt.subplots(figsize=(10, height))

    labels = [f"{r.reach_name} (#{r.reach_id})" for r in sorted_results]
    values = [getattr(r, key) for r in sorted_results]

    ax.barh(labels, values, color="steelblue")
    ax.invert_yaxis()
    ax.set_xlabel(f"{metric} depletion")
    ax.set_title(title or f"Top {top_n} reaches by {metric} depletion")
    ax.grid(True, axis="x", alpha=0.3)
    return ax
