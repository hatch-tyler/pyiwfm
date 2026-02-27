"""Sweep tests for pyiwfm.visualization.plotting targeting remaining uncovered lines.

Covers:
- plot_boundary(): ConvexHull fallback (lines 869-888)
- plot_streams_colored(): LineCollection colored reaches (lines 2190-2233)
- plot_timeseries_statistics(): ensemble mean with min/max and std bands (lines 2281-2325)
- plot_dual_axis(): dual y-axis comparison (lines 2372-2406)
- plot_streamflow_hydrograph(): with baseflow AND log_scale (lines 2461-2485)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from tests.conftest import make_simple_grid

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _close_figures() -> None:  # type: ignore[misc]
    """Close all matplotlib figures after each test."""
    yield  # type: ignore[misc]
    plt.close("all")


# ---------------------------------------------------------------------------
# Helpers -- mock stream objects
# ---------------------------------------------------------------------------


@dataclass
class _FakeStrmNode:
    """Minimal stream node with x, y for plotting."""

    id: int
    x: float
    y: float
    gw_node: int | None = None


@dataclass
class _FakeStrmReach:
    """Minimal stream reach with node list."""

    id: int
    nodes: list[int] = field(default_factory=list)


@dataclass
class _FakeAppStream:
    """Minimal stream network for plot_streams_colored tests."""

    nodes: dict[int, _FakeStrmNode] = field(default_factory=dict)
    reaches: dict[int, _FakeStrmReach] = field(default_factory=dict)

    def iter_reaches(self):  # type: ignore[no-untyped-def]
        for rid in sorted(self.reaches.keys()):
            yield self.reaches[rid]


# ---------------------------------------------------------------------------
# plot_boundary -- ConvexHull fallback
# ---------------------------------------------------------------------------


class TestPlotBoundaryConvexHullFallback:
    """Exercise the ConvexHull fallback when no nodes have is_boundary=True."""

    def test_convex_hull_no_fill(self) -> None:
        """When no boundary nodes exist, ConvexHull outlines the mesh."""
        grid = make_simple_grid()
        # Ensure no node is marked as boundary
        for node in grid.nodes.values():
            node.is_boundary = False

        from pyiwfm.visualization.plotting import plot_boundary

        fig, ax = plot_boundary(grid, fill=False)
        assert isinstance(fig, Figure)
        # Should have plotted at least one line
        assert len(ax.lines) >= 1

    def test_convex_hull_with_fill(self) -> None:
        """When no boundary nodes exist, ConvexHull fills a polygon patch."""
        grid = make_simple_grid()
        for node in grid.nodes.values():
            node.is_boundary = False

        from pyiwfm.visualization.plotting import plot_boundary

        fig, ax = plot_boundary(grid, fill=True, fill_color="lightblue", alpha=0.5)
        assert isinstance(fig, Figure)
        # Should have added a polygon patch
        assert len(ax.patches) >= 1

    def test_convex_hull_custom_colors(self) -> None:
        """ConvexHull fallback respects color arguments."""
        grid = make_simple_grid()
        for node in grid.nodes.values():
            node.is_boundary = False

        from pyiwfm.visualization.plotting import plot_boundary

        fig, ax = plot_boundary(grid, fill=False, line_color="red", line_width=3.0)
        assert isinstance(fig, Figure)
        assert ax.lines[0].get_color() == "red"


# ---------------------------------------------------------------------------
# plot_streams_colored
# ---------------------------------------------------------------------------


class TestPlotStreamsColored:
    """Test colored stream reach plotting via LineCollection."""

    def _make_stream_network(self) -> tuple[_FakeAppStream, np.ndarray]:
        """Build a small 2-reach stream with nodes at known coordinates."""
        nodes = {
            1: _FakeStrmNode(id=1, x=50.0, y=0.0),
            2: _FakeStrmNode(id=2, x=50.0, y=50.0),
            3: _FakeStrmNode(id=3, x=50.0, y=100.0),
            4: _FakeStrmNode(id=4, x=100.0, y=150.0),
        }
        reaches = {
            1: _FakeStrmReach(id=1, nodes=[1, 2]),
            2: _FakeStrmReach(id=2, nodes=[3, 4]),
        }
        streams = _FakeAppStream(nodes=nodes, reaches=reaches)
        values = np.array([10.0, 25.0])
        return streams, values

    def test_basic_colored_streams(self) -> None:
        grid = make_simple_grid()
        streams, values = self._make_stream_network()

        from pyiwfm.visualization.plotting import plot_streams_colored

        fig, ax = plot_streams_colored(grid, streams, values, show_mesh=False, show_colorbar=True)
        assert isinstance(fig, Figure)

    def test_colored_streams_with_mesh(self) -> None:
        grid = make_simple_grid()
        streams, values = self._make_stream_network()

        from pyiwfm.visualization.plotting import plot_streams_colored

        fig, ax = plot_streams_colored(
            grid, streams, values, show_mesh=True, colorbar_label="Flow (cfs)"
        )
        assert isinstance(fig, Figure)

    def test_colored_streams_custom_range(self) -> None:
        grid = make_simple_grid()
        streams, values = self._make_stream_network()

        from pyiwfm.visualization.plotting import plot_streams_colored

        fig, ax = plot_streams_colored(grid, streams, values, vmin=0.0, vmax=50.0, cmap="Reds")
        assert isinstance(fig, Figure)

    def test_colored_streams_no_colorbar(self) -> None:
        grid = make_simple_grid()
        streams, values = self._make_stream_network()

        from pyiwfm.visualization.plotting import plot_streams_colored

        fig, ax = plot_streams_colored(grid, streams, values, show_colorbar=False)
        assert isinstance(fig, Figure)

    def test_colored_streams_on_existing_ax(self) -> None:
        grid = make_simple_grid()
        streams, values = self._make_stream_network()

        from pyiwfm.visualization.plotting import plot_streams_colored

        fig_ext, ax_ext = plt.subplots()
        fig, ax = plot_streams_colored(
            grid, streams, values, ax=ax_ext, show_mesh=False, show_colorbar=False
        )
        assert ax is ax_ext


# ---------------------------------------------------------------------------
# plot_timeseries_statistics
# ---------------------------------------------------------------------------


class TestPlotTimeseriesStatistics:
    """Test ensemble mean with min/max or std bands."""

    @staticmethod
    def _make_collection():  # type: ignore[no-untyped-def]
        from pyiwfm.core.timeseries import TimeSeries, TimeSeriesCollection

        times = np.array(["2020-01-01", "2020-02-01", "2020-03-01"], dtype="datetime64[s]")
        coll = TimeSeriesCollection(name="heads")
        for i in range(5):
            vals = np.array([100.0 + i, 98.0 - i, 99.0 + 0.5 * i])
            ts = TimeSeries(
                times=times,
                values=vals,
                name=f"Well_{i}",
                location=f"Well_{i}",
                units="ft",
            )
            coll.add(ts)
        return coll

    def test_minmax_band(self) -> None:
        from pyiwfm.visualization.plotting import plot_timeseries_statistics

        coll = self._make_collection()
        fig, ax = plot_timeseries_statistics(coll, band="minmax")
        assert isinstance(fig, Figure)
        # Should have the mean line
        assert len(ax.lines) >= 1

    def test_std_band(self) -> None:
        from pyiwfm.visualization.plotting import plot_timeseries_statistics

        coll = self._make_collection()
        fig, ax = plot_timeseries_statistics(coll, band="std", title="Test", ylabel="Head (ft)")
        assert isinstance(fig, Figure)

    def test_show_individual(self) -> None:
        from pyiwfm.visualization.plotting import plot_timeseries_statistics

        coll = self._make_collection()
        fig, ax = plot_timeseries_statistics(coll, show_individual=True)
        assert isinstance(fig, Figure)
        # Individual lines + mean = 6 total
        assert len(ax.lines) >= 6

    def test_empty_collection(self) -> None:
        from pyiwfm.core.timeseries import TimeSeriesCollection
        from pyiwfm.visualization.plotting import plot_timeseries_statistics

        coll = TimeSeriesCollection(name="empty")
        fig, ax = plot_timeseries_statistics(coll)
        assert isinstance(fig, Figure)

    def test_on_existing_axes(self) -> None:
        from pyiwfm.visualization.plotting import plot_timeseries_statistics

        coll = self._make_collection()
        fig_ext, ax_ext = plt.subplots()
        fig, ax = plot_timeseries_statistics(coll, ax=ax_ext)
        assert ax is ax_ext


# ---------------------------------------------------------------------------
# plot_dual_axis
# ---------------------------------------------------------------------------


class TestPlotDualAxis:
    """Test dual y-axis comparison of two time series."""

    @staticmethod
    def _make_pair():  # type: ignore[no-untyped-def]
        from pyiwfm.core.timeseries import TimeSeries

        times = np.array(
            ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01", "2020-06-01"],
            dtype="datetime64[s]",
        )
        ts1 = TimeSeries(
            times=times,
            values=np.array([100.0, 98.0, 97.0, 99.0, 101.0, 100.0]),
            name="GW Head",
            units="ft",
        )
        ts2 = TimeSeries(
            times=times,
            values=np.array([500.0, 480.0, 510.0, 530.0, 490.0, 500.0]),
            name="Streamflow",
            units="cfs",
        )
        return ts1, ts2

    def test_basic_dual_axis(self) -> None:
        from pyiwfm.visualization.plotting import plot_dual_axis

        ts1, ts2 = self._make_pair()
        fig, (ax_left, ax_right) = plot_dual_axis(ts1, ts2)
        assert isinstance(fig, Figure)
        assert ax_left is not ax_right

    def test_dual_axis_with_labels(self) -> None:
        from pyiwfm.visualization.plotting import plot_dual_axis

        ts1, ts2 = self._make_pair()
        fig, (ax_left, ax_right) = plot_dual_axis(
            ts1,
            ts2,
            label1="Head",
            label2="Flow",
            ylabel1="Elevation (ft)",
            ylabel2="Discharge (cfs)",
            title="Comparison",
        )
        assert isinstance(fig, Figure)
        assert ax_left.get_ylabel() == "Elevation (ft)"
        assert ax_right.get_ylabel() == "Discharge (cfs)"

    def test_dual_axis_custom_colors(self) -> None:
        from pyiwfm.visualization.plotting import plot_dual_axis

        ts1, ts2 = self._make_pair()
        fig, _ = plot_dual_axis(
            ts1,
            ts2,
            color1="green",
            color2="purple",
            style1="--",
            style2="o-",
        )
        assert isinstance(fig, Figure)

    def test_dual_axis_on_existing_ax(self) -> None:
        from pyiwfm.visualization.plotting import plot_dual_axis

        ts1, ts2 = self._make_pair()
        fig_ext, ax_ext = plt.subplots()
        fig, (ax_left, ax_right) = plot_dual_axis(ts1, ts2, ax=ax_ext)
        assert ax_left is ax_ext

    def test_dual_axis_fallback_names(self) -> None:
        """When no labels given, falls back to ts.name."""
        from pyiwfm.visualization.plotting import plot_dual_axis

        ts1, ts2 = self._make_pair()
        fig, _ = plot_dual_axis(ts1, ts2)
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# plot_streamflow_hydrograph
# ---------------------------------------------------------------------------


class TestPlotStreamflowHydrograph:
    """Test streamflow hydrograph with baseflow and log scale branches."""

    @staticmethod
    def _make_data():  # type: ignore[no-untyped-def]
        times = np.arange(
            np.datetime64("2020-01-01"),
            np.datetime64("2020-01-11"),
            np.timedelta64(1, "D"),
        ).astype("datetime64[s]")
        flows = np.array([100.0, 150.0, 300.0, 250.0, 200.0, 180.0, 160.0, 140.0, 120.0, 110.0])
        baseflow = np.array([80.0, 85.0, 90.0, 95.0, 100.0, 100.0, 95.0, 90.0, 85.0, 80.0])
        return times, flows, baseflow

    def test_basic_hydrograph(self) -> None:
        from pyiwfm.visualization.plotting import plot_streamflow_hydrograph

        times, flows, _ = self._make_data()
        fig, ax = plot_streamflow_hydrograph(times, flows)
        assert isinstance(fig, Figure)
        assert len(ax.lines) >= 1

    def test_with_baseflow(self) -> None:
        from pyiwfm.visualization.plotting import plot_streamflow_hydrograph

        times, flows, baseflow = self._make_data()
        fig, ax = plot_streamflow_hydrograph(times, flows, baseflow=baseflow)
        assert isinstance(fig, Figure)
        # Total flow line + baseflow line
        assert len(ax.lines) >= 2

    def test_log_scale(self) -> None:
        from pyiwfm.visualization.plotting import plot_streamflow_hydrograph

        times, flows, _ = self._make_data()
        fig, ax = plot_streamflow_hydrograph(times, flows, log_scale=True)
        assert isinstance(fig, Figure)
        assert ax.get_yscale() == "log"

    def test_baseflow_and_log_scale(self) -> None:
        from pyiwfm.visualization.plotting import plot_streamflow_hydrograph

        times, flows, baseflow = self._make_data()
        fig, ax = plot_streamflow_hydrograph(times, flows, baseflow=baseflow, log_scale=True)
        assert isinstance(fig, Figure)
        assert ax.get_yscale() == "log"
        assert len(ax.lines) >= 2

    def test_custom_labels(self) -> None:
        from pyiwfm.visualization.plotting import plot_streamflow_hydrograph

        times, flows, baseflow = self._make_data()
        fig, ax = plot_streamflow_hydrograph(
            times,
            flows,
            baseflow=baseflow,
            title="Test Reach",
            ylabel="Discharge",
            units="m3/s",
        )
        assert isinstance(fig, Figure)
        assert "Discharge" in ax.get_ylabel()

    def test_on_existing_axes(self) -> None:
        from pyiwfm.visualization.plotting import plot_streamflow_hydrograph

        times, flows, _ = self._make_data()
        fig_ext, ax_ext = plt.subplots()
        fig, ax = plot_streamflow_hydrograph(times, flows, ax=ax_ext)
        assert ax is ax_ext
