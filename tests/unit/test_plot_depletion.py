"""Tests for ``pyiwfm.visualization.plot_depletion`` (Phase 2.2.a-ii).

We test that each plot function:

- returns the populated :class:`matplotlib.axes.Axes`
- draws the right number of artists for a known fixture
- accepts an externally-supplied ``ax`` (composability)
- validates inputs (raises ``ValueError`` on bad arguments)

We don't test pixel-level visual output here; the appearance is locked
in by separate visual-regression tests against committed reference PNGs
(out of scope for this PR — the plots use matplotlib's default style).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes

from pyiwfm.io.streams.depletion import StreamDepletionReport, StreamDepletionResult
from pyiwfm.visualization.plot_depletion import (
    plot_cumulative_depletion,
    plot_depletion_summary_bar,
    plot_depletion_timeseries,
)


def _make_report(n_reaches: int = 3, n_timesteps: int = 12) -> StreamDepletionReport:
    rng = np.random.default_rng(1)
    times = [f"t{i}" for i in range(n_timesteps)]
    results = []
    for ri in range(n_reaches):
        base = rng.uniform(50.0, 100.0, n_timesteps)
        scen = base - rng.uniform(1.0, 8.0, n_timesteps)
        depletion = base - scen
        cumulative = np.cumsum(depletion)
        results.append(
            StreamDepletionResult(
                reach_id=ri + 1,
                reach_name=f"Reach {ri + 1}",
                times=times,
                baseline_flow=base,
                scenario_flow=scen,
                depletion=depletion,
                cumulative_depletion=cumulative,
                max_depletion=float(np.max(np.abs(depletion))),
                max_depletion_timestep=int(np.argmax(np.abs(depletion))),
                total_depletion=float(cumulative[-1]),
            )
        )
    return StreamDepletionReport(
        results=results,
        n_reaches=n_reaches,
        n_timesteps=n_timesteps,
        total_max_depletion=max(r.max_depletion for r in results),
        total_cumulative_depletion=sum(r.total_depletion for r in results),
    )


@pytest.fixture(autouse=True)
def _close_figs():
    """Close all figures after each test to avoid memory pressure."""
    yield
    plt.close("all")


class TestPlotCumulativeDepletion:
    def test_returns_axes(self):
        report = _make_report()
        ax = plot_cumulative_depletion(report)
        assert isinstance(ax, Axes)

    def test_one_line_per_reach(self):
        report = _make_report(n_reaches=3, n_timesteps=10)
        ax = plot_cumulative_depletion(report)
        assert len(ax.get_lines()) == 3

    def test_filter_by_reach_ids(self):
        report = _make_report(n_reaches=3)
        ax = plot_cumulative_depletion(report, reach_ids=[1, 3])
        assert len(ax.get_lines()) == 2

    def test_unknown_reach_id_raises(self):
        report = _make_report(n_reaches=3)
        with pytest.raises(ValueError, match=r"\[99\]"):
            plot_cumulative_depletion(report, reach_ids=[1, 99])

    def test_uses_supplied_ax(self):
        report = _make_report(n_reaches=2)
        fig, ax = plt.subplots()
        result = plot_cumulative_depletion(report, ax=ax)
        assert result is ax

    def test_pumping_overlay_adds_secondary_axis(self):
        report = _make_report(n_reaches=2, n_timesteps=12)
        pumping = np.linspace(0, 100, 12)
        fig, ax = plt.subplots()
        plot_cumulative_depletion(report, ax=ax, pumping_timeseries=pumping)
        # The twin axis is registered on the Figure
        twin_axes = [a for a in fig.axes if a is not ax]
        assert len(twin_axes) == 1
        # Twin axis has its own line for pumping
        assert len(twin_axes[0].get_lines()) == 1

    def test_pumping_length_mismatch_raises(self):
        report = _make_report(n_reaches=1, n_timesteps=12)
        with pytest.raises(ValueError, match=r"length .* does not match"):
            plot_cumulative_depletion(report, pumping_timeseries=np.array([1.0, 2.0]))


class TestPlotDepletionTimeseries:
    def test_returns_axes_with_lines(self):
        report = _make_report(n_reaches=3)
        ax = plot_depletion_timeseries(report)
        assert isinstance(ax, Axes)
        assert len(ax.get_lines()) >= 3  # one per reach + zero-line

    def test_zero_reference_line_present(self):
        # The axhline at 0 should be drawn so users see the depletion sign
        report = _make_report(n_reaches=1)
        ax = plot_depletion_timeseries(report)
        ydata = [line.get_ydata() for line in ax.get_lines()]
        # One line should be a constant 0 (the zero reference)
        assert any(np.allclose(y, 0) for y in ydata if len(y) >= 2)

    def test_filter_by_reach_ids(self):
        report = _make_report(n_reaches=4)
        ax = plot_depletion_timeseries(report, reach_ids=[2])
        # 1 reach + zero line
        assert len(ax.get_lines()) == 2


class TestPlotDepletionSummaryBar:
    def test_returns_axes(self):
        report = _make_report(n_reaches=5)
        ax = plot_depletion_summary_bar(report)
        assert isinstance(ax, Axes)

    def test_top_n_limits_bars(self):
        report = _make_report(n_reaches=10)
        ax = plot_depletion_summary_bar(report, top_n=3)
        # In matplotlib, barh draws Rectangle patches — count them
        from matplotlib.patches import Rectangle

        rects = [p for p in ax.patches if isinstance(p, Rectangle)]
        assert len(rects) == 3

    def test_metric_max_vs_total(self):
        report = _make_report(n_reaches=4)
        ax_total = plot_depletion_summary_bar(report, metric="total")
        ax_max = plot_depletion_summary_bar(report, metric="max")
        assert "total" in ax_total.get_xlabel()
        assert "max" in ax_max.get_xlabel()

    def test_unknown_metric_raises(self):
        report = _make_report()
        with pytest.raises(ValueError, match="metric must be"):
            plot_depletion_summary_bar(report, metric="median")  # type: ignore[arg-type]

    def test_zero_top_n_raises(self):
        report = _make_report()
        with pytest.raises(ValueError, match="top_n must be positive"):
            plot_depletion_summary_bar(report, top_n=0)

    def test_top_n_larger_than_n_reaches_ok(self):
        # If top_n > available, just show all available
        report = _make_report(n_reaches=2)
        ax = plot_depletion_summary_bar(report, top_n=10)
        from matplotlib.patches import Rectangle

        rects = [p for p in ax.patches if isinstance(p, Rectangle)]
        assert len(rects) == 2
