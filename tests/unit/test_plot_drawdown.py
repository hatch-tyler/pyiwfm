"""Tests for ``pyiwfm.visualization.plot_drawdown`` (Phase 2 drawdown).

Verifies that each plot returns a populated Axes, draws the right
number of artists, validates inputs, and accepts an externally-supplied
``ax``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes

from pyiwfm.io.drawdown import DrawdownAtLocation, DrawdownTimeSeriesReport
from pyiwfm.visualization.plot_drawdown import (
    plot_drawdown_summary_bar,
    plot_drawdown_timeseries,
)


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _make_report(n_locations: int = 3, n_timesteps: int = 5) -> DrawdownTimeSeriesReport:
    rng = np.random.default_rng(0)
    times = [f"t{i}" for i in range(n_timesteps)]
    locations = []
    for i in range(n_locations):
        dd = rng.uniform(0.0, 5.0 * (i + 1), n_timesteps)
        locations.append(
            DrawdownAtLocation(
                node_id=i + 1,
                layer=1,
                times=times,
                drawdown=dd,
                max_drawdown=float(np.max(dd)),
                max_drawdown_timestep=int(np.argmax(dd)),
                final_drawdown=float(dd[-1]),
            )
        )
    return DrawdownTimeSeriesReport(
        locations=locations,
        n_locations=n_locations,
        n_timesteps=n_timesteps,
        reference_timestep=0,
        times=times,
    )


class TestPlotDrawdownTimeseries:
    def test_returns_axes_with_lines(self):
        report = _make_report(n_locations=3)
        ax = plot_drawdown_timeseries(report)
        assert isinstance(ax, Axes)
        # 3 locations + the zero reference line
        assert len(ax.get_lines()) >= 3

    def test_zero_reference_line(self):
        report = _make_report(n_locations=1)
        ax = plot_drawdown_timeseries(report)
        ydata = [line.get_ydata() for line in ax.get_lines()]
        # One line is constant 0
        assert any(np.allclose(y, 0) for y in ydata if len(y) >= 2)

    def test_filter_by_location_keys(self):
        report = _make_report(n_locations=4)
        ax = plot_drawdown_timeseries(report, location_keys=[(2, 1)])
        # 1 location + zero line
        assert len(ax.get_lines()) == 2

    def test_unknown_location_raises(self):
        report = _make_report(n_locations=2)
        with pytest.raises(ValueError, match=r"\(99, 1\)"):
            plot_drawdown_timeseries(report, location_keys=[(99, 1)])

    def test_uses_supplied_ax(self):
        report = _make_report()
        fig, ax = plt.subplots()
        result = plot_drawdown_timeseries(report, ax=ax)
        assert result is ax

    def test_empty_report_returns_axes(self):
        report = DrawdownTimeSeriesReport(
            locations=[],
            n_locations=0,
            n_timesteps=0,
            reference_timestep=0,
            times=[],
        )
        ax = plot_drawdown_timeseries(report)
        assert isinstance(ax, Axes)


class TestPlotDrawdownSummaryBar:
    def test_returns_axes(self):
        report = _make_report(n_locations=5)
        ax = plot_drawdown_summary_bar(report)
        assert isinstance(ax, Axes)

    def test_top_n_limits_bars(self):
        from matplotlib.patches import Rectangle

        report = _make_report(n_locations=10)
        ax = plot_drawdown_summary_bar(report, top_n=4)
        rects = [p for p in ax.patches if isinstance(p, Rectangle)]
        assert len(rects) == 4

    def test_metric_max_vs_final(self):
        report = _make_report(n_locations=4)
        ax_max = plot_drawdown_summary_bar(report, metric="max")
        ax_final = plot_drawdown_summary_bar(report, metric="final")
        assert "max" in ax_max.get_xlabel()
        assert "final" in ax_final.get_xlabel()

    def test_unknown_metric_raises(self):
        report = _make_report()
        with pytest.raises(ValueError, match="metric must be"):
            plot_drawdown_summary_bar(report, metric="median")  # type: ignore[arg-type]

    def test_zero_top_n_raises(self):
        report = _make_report()
        with pytest.raises(ValueError, match="top_n must be positive"):
            plot_drawdown_summary_bar(report, top_n=0)

    def test_handles_nan_in_metric(self):
        # NaN max_drawdown locations should sort last but not crash
        report = _make_report(n_locations=3)
        report.locations[1].max_drawdown = float("nan")
        ax = plot_drawdown_summary_bar(report)
        assert isinstance(ax, Axes)

    def test_top_n_larger_than_n_locations_ok(self):
        from matplotlib.patches import Rectangle

        report = _make_report(n_locations=2)
        ax = plot_drawdown_summary_bar(report, top_n=10)
        rects = [p for p in ax.patches if isinstance(p, Rectangle)]
        assert len(rects) == 2
