"""Unit tests for plotting functionality."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pyiwfm.components.stream import AppStream, StrmNode, StrmReach
from pyiwfm.core.mesh import AppGrid, Element, Node
from pyiwfm.core.timeseries import TimeSeries, TimeSeriesCollection
from pyiwfm.visualization.plotting import (
    BudgetPlotter,
    MeshPlotter,
    _subdivide_quads,
    plot_boundary,
    plot_budget_bar,
    plot_budget_pie,
    plot_budget_stacked,
    plot_budget_timeseries,
    plot_elements,
    plot_mesh,
    plot_nodes,
    plot_scalar_field,
    plot_streams,
    plot_timeseries,
    plot_timeseries_collection,
    plot_timeseries_comparison,
    plot_water_balance,
    plot_zbudget,
)


@pytest.fixture
def simple_grid() -> AppGrid:
    """Create a simple 2x2 quad mesh for testing."""
    nodes = {
        1: Node(id=1, x=0.0, y=0.0, is_boundary=True),
        2: Node(id=2, x=100.0, y=0.0, is_boundary=True),
        3: Node(id=3, x=200.0, y=0.0, is_boundary=True),
        4: Node(id=4, x=0.0, y=100.0, is_boundary=True),
        5: Node(id=5, x=100.0, y=100.0, is_boundary=False),
        6: Node(id=6, x=200.0, y=100.0, is_boundary=True),
        7: Node(id=7, x=0.0, y=200.0, is_boundary=True),
        8: Node(id=8, x=100.0, y=200.0, is_boundary=True),
        9: Node(id=9, x=200.0, y=200.0, is_boundary=True),
    }
    elements = {
        1: Element(id=1, vertices=(1, 2, 5, 4), subregion=1),
        2: Element(id=2, vertices=(2, 3, 6, 5), subregion=1),
        3: Element(id=3, vertices=(4, 5, 8, 7), subregion=2),
        4: Element(id=4, vertices=(5, 6, 9, 8), subregion=2),
    }
    grid = AppGrid(nodes=nodes, elements=elements)
    grid.compute_connectivity()
    return grid


@pytest.fixture
def simple_stream() -> AppStream:
    """Create simple stream network for testing."""
    stream = AppStream()

    # Add stream nodes
    for i in range(1, 6):
        stream.add_node(
            StrmNode(
                id=i,
                x=float(i * 40),
                y=100.0,
                reach_id=1,
            )
        )

    # Add reach
    stream.add_reach(
        StrmReach(
            id=1,
            name="Test Stream",
            upstream_node=1,
            downstream_node=5,
            nodes=[1, 2, 3, 4, 5],
        )
    )

    return stream


class TestPlotMesh:
    """Tests for mesh plotting functions."""

    def test_plot_mesh_returns_figure(self, simple_grid: AppGrid) -> None:
        """Test that plot_mesh returns a figure."""
        fig, ax = plot_mesh(simple_grid)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_mesh_with_edges(self, simple_grid: AppGrid) -> None:
        """Test plotting mesh with element edges."""
        fig, ax = plot_mesh(simple_grid, show_edges=True)
        assert fig is not None
        plt.close(fig)

    def test_plot_mesh_with_node_ids(self, simple_grid: AppGrid) -> None:
        """Test plotting mesh with node IDs."""
        fig, ax = plot_mesh(simple_grid, show_node_ids=True)
        assert fig is not None
        plt.close(fig)

    def test_plot_mesh_with_element_ids(self, simple_grid: AppGrid) -> None:
        """Test plotting mesh with element IDs."""
        fig, ax = plot_mesh(simple_grid, show_element_ids=True)
        assert fig is not None
        plt.close(fig)


class TestPlotNodes:
    """Tests for node plotting."""

    def test_plot_nodes_returns_figure(self, simple_grid: AppGrid) -> None:
        """Test that plot_nodes returns a figure."""
        fig, ax = plot_nodes(simple_grid)
        assert fig is not None
        plt.close(fig)

    def test_plot_nodes_highlight_boundary(self, simple_grid: AppGrid) -> None:
        """Test highlighting boundary nodes."""
        fig, ax = plot_nodes(simple_grid, highlight_boundary=True)
        assert fig is not None
        plt.close(fig)


class TestPlotElements:
    """Tests for element plotting."""

    def test_plot_elements_returns_figure(self, simple_grid: AppGrid) -> None:
        """Test that plot_elements returns a figure."""
        fig, ax = plot_elements(simple_grid)
        assert fig is not None
        plt.close(fig)

    def test_plot_elements_color_by_subregion(self, simple_grid: AppGrid) -> None:
        """Test coloring elements by subregion."""
        fig, ax = plot_elements(simple_grid, color_by="subregion")
        assert fig is not None
        plt.close(fig)


class TestPlotScalarField:
    """Tests for scalar field visualization."""

    def test_plot_node_scalars(self, simple_grid: AppGrid) -> None:
        """Test plotting scalar values at nodes."""
        values = np.arange(9, dtype=float) * 10
        fig, ax = plot_scalar_field(simple_grid, values, field_type="node")
        assert fig is not None
        plt.close(fig)

    def test_plot_cell_scalars(self, simple_grid: AppGrid) -> None:
        """Test plotting scalar values at cells."""
        values = np.array([10.0, 20.0, 15.0, 25.0])
        fig, ax = plot_scalar_field(simple_grid, values, field_type="cell")
        assert fig is not None
        plt.close(fig)

    def test_plot_with_colorbar(self, simple_grid: AppGrid) -> None:
        """Test adding colorbar to scalar plot."""
        values = np.arange(9, dtype=float) * 10
        fig, ax = plot_scalar_field(simple_grid, values, field_type="node", show_colorbar=True)
        assert fig is not None
        plt.close(fig)

    def test_plot_with_custom_cmap(self, simple_grid: AppGrid) -> None:
        """Test using custom colormap."""
        values = np.arange(9, dtype=float) * 10
        fig, ax = plot_scalar_field(simple_grid, values, field_type="node", cmap="coolwarm")
        assert fig is not None
        plt.close(fig)


class TestPlotStreams:
    """Tests for stream network plotting."""

    def test_plot_streams_returns_figure(
        self, simple_grid: AppGrid, simple_stream: AppStream
    ) -> None:
        """Test that plot_streams returns a figure."""
        fig, ax = plot_streams(simple_stream)
        assert fig is not None
        plt.close(fig)

    def test_plot_streams_with_nodes(self, simple_grid: AppGrid, simple_stream: AppStream) -> None:
        """Test plotting streams with node markers."""
        fig, ax = plot_streams(simple_stream, show_nodes=True)
        assert fig is not None
        plt.close(fig)


class TestPlotBoundary:
    """Tests for boundary plotting."""

    def test_plot_boundary_returns_figure(self, simple_grid: AppGrid) -> None:
        """Test that plot_boundary returns a figure."""
        fig, ax = plot_boundary(simple_grid)
        assert fig is not None
        plt.close(fig)


class TestMeshPlotter:
    """Tests for MeshPlotter class."""

    def test_plotter_creation(self, simple_grid: AppGrid) -> None:
        """Test creating a MeshPlotter."""
        plotter = MeshPlotter(simple_grid)
        assert plotter is not None

    def test_plotter_plot_mesh(self, simple_grid: AppGrid) -> None:
        """Test plotting mesh through MeshPlotter."""
        plotter = MeshPlotter(simple_grid)
        fig, ax = plotter.plot_mesh()
        assert fig is not None
        plt.close(fig)

    def test_plotter_with_streams(self, simple_grid: AppGrid, simple_stream: AppStream) -> None:
        """Test MeshPlotter with stream network."""
        plotter = MeshPlotter(simple_grid, streams=simple_stream)
        fig, ax = plotter.plot_mesh(show_streams=True)
        assert fig is not None
        plt.close(fig)

    def test_plotter_save_figure(self, simple_grid: AppGrid, tmp_path: Path) -> None:
        """Test saving figure to file."""
        plotter = MeshPlotter(simple_grid)
        output_file = tmp_path / "mesh.png"
        plotter.save(output_file)
        assert output_file.exists()

    def test_plotter_composite_plot(self, simple_grid: AppGrid, simple_stream: AppStream) -> None:
        """Test creating composite plot with multiple layers."""
        plotter = MeshPlotter(simple_grid, streams=simple_stream)

        # Add scalar data
        heads = np.arange(9, dtype=float) * 10

        fig, ax = plotter.plot_composite(
            show_mesh=True,
            show_streams=True,
            node_values=heads,
            title="Composite Plot",
        )
        assert fig is not None
        plt.close(fig)


# =============================================================================
# Time Series Plotting Fixtures
# =============================================================================


@pytest.fixture
def sample_timeseries() -> TimeSeries:
    """Create a sample time series for testing."""
    times = np.array(
        [
            "2020-01-01",
            "2020-02-01",
            "2020-03-01",
            "2020-04-01",
            "2020-05-01",
            "2020-06-01",
            "2020-07-01",
            "2020-08-01",
            "2020-09-01",
            "2020-10-01",
            "2020-11-01",
            "2020-12-01",
        ],
        dtype="datetime64[D]",
    )
    values = np.array([100.0, 98.5, 97.2, 96.0, 95.1, 94.8, 95.5, 96.2, 97.0, 98.1, 99.0, 99.8])
    return TimeSeries(times=times, values=values, name="Well_1", units="ft")


@pytest.fixture
def sample_timeseries_list() -> list[TimeSeries]:
    """Create multiple time series for testing."""
    times = np.array(
        [
            "2020-01-01",
            "2020-02-01",
            "2020-03-01",
            "2020-04-01",
            "2020-05-01",
            "2020-06-01",
        ],
        dtype="datetime64[D]",
    )

    ts1 = TimeSeries(
        times=times,
        values=np.array([100.0, 99.0, 98.0, 97.0, 96.0, 95.0]),
        name="Well_1",
        units="ft",
    )
    ts2 = TimeSeries(
        times=times,
        values=np.array([110.0, 109.5, 109.0, 108.5, 108.0, 107.5]),
        name="Well_2",
        units="ft",
    )
    ts3 = TimeSeries(
        times=times,
        values=np.array([90.0, 89.0, 88.0, 87.5, 87.0, 86.5]),
        name="Well_3",
        units="ft",
    )
    return [ts1, ts2, ts3]


@pytest.fixture
def observed_simulated_pair() -> tuple[TimeSeries, TimeSeries]:
    """Create observed and simulated time series for comparison testing."""
    times = np.array(
        [
            "2020-01-01",
            "2020-02-01",
            "2020-03-01",
            "2020-04-01",
            "2020-05-01",
            "2020-06-01",
        ],
        dtype="datetime64[D]",
    )

    observed = TimeSeries(
        times=times,
        values=np.array([100.0, 98.5, 97.0, 96.5, 95.0, 94.5]),
        name="Observed",
        units="ft",
    )
    simulated = TimeSeries(
        times=times,
        values=np.array([99.5, 98.0, 97.5, 96.0, 95.5, 94.0]),
        name="Simulated",
        units="ft",
    )
    return observed, simulated


@pytest.fixture
def sample_collection(sample_timeseries_list: list[TimeSeries]) -> TimeSeriesCollection:
    """Create a time series collection for testing."""
    collection = TimeSeriesCollection(name="Head Observations", variable="head")
    for ts in sample_timeseries_list:
        ts.location = ts.name
        collection.add(ts)
    return collection


# =============================================================================
# Budget Plotting Fixtures
# =============================================================================


@pytest.fixture
def sample_budget_components() -> dict[str, float]:
    """Create sample budget components for testing."""
    return {
        "Precipitation": 1500.0,
        "Stream Inflow": 800.0,
        "Recharge": 400.0,
        "Pumping": -1200.0,
        "Evapotranspiration": -600.0,
        "Stream Outflow": -500.0,
        "Subsurface Outflow": -300.0,
    }


@pytest.fixture
def sample_budget_timeseries() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Create time-varying budget components for testing."""
    times = np.array(
        [
            "2020-01-01",
            "2020-02-01",
            "2020-03-01",
            "2020-04-01",
            "2020-05-01",
            "2020-06-01",
            "2020-07-01",
            "2020-08-01",
            "2020-09-01",
            "2020-10-01",
            "2020-11-01",
            "2020-12-01",
        ],
        dtype="datetime64[D]",
    )

    components = {
        "Precipitation": np.array([200, 180, 150, 100, 50, 20, 10, 15, 40, 80, 150, 180]),
        "Recharge": np.array([50, 45, 40, 30, 15, 5, 3, 4, 10, 20, 40, 45]),
        "Pumping": -np.array([80, 85, 90, 100, 120, 130, 135, 130, 110, 90, 80, 75]),
        "ET": -np.array([30, 35, 50, 70, 100, 120, 125, 110, 80, 50, 35, 30]),
    }
    return times, components


@pytest.fixture
def sample_zone_budgets() -> dict[int, dict[str, float]]:
    """Create zone budget data for testing."""
    return {
        1: {
            "Recharge": 500.0,
            "Pumping": -300.0,
            "Flow to Zone 2": -100.0,
            "Stream Leakage": 50.0,
        },
        2: {
            "Recharge": 300.0,
            "Pumping": -200.0,
            "Flow from Zone 1": 100.0,
            "Stream Leakage": -80.0,
        },
        3: {
            "Recharge": 400.0,
            "Pumping": -350.0,
            "Flow to Zone 2": -50.0,
            "Stream Leakage": 20.0,
        },
    }


# =============================================================================
# Time Series Plotting Tests
# =============================================================================


class TestPlotTimeseries:
    """Tests for time series plotting functions."""

    def test_plot_single_timeseries(self, sample_timeseries: TimeSeries) -> None:
        """Test plotting a single time series."""
        fig, ax = plot_timeseries(sample_timeseries)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_multiple_timeseries(self, sample_timeseries_list: list[TimeSeries]) -> None:
        """Test plotting multiple time series."""
        fig, ax = plot_timeseries(sample_timeseries_list)
        assert fig is not None
        plt.close(fig)

    def test_plot_with_title(self, sample_timeseries: TimeSeries) -> None:
        """Test plotting with custom title."""
        fig, ax = plot_timeseries(sample_timeseries, title="Groundwater Head")
        assert ax.get_title() == "Groundwater Head"
        plt.close(fig)

    def test_plot_with_custom_colors(self, sample_timeseries_list: list[TimeSeries]) -> None:
        """Test plotting with custom colors."""
        fig, ax = plot_timeseries(
            sample_timeseries_list,
            colors=["red", "green", "blue"],
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_with_markers(self, sample_timeseries: TimeSeries) -> None:
        """Test plotting with markers."""
        fig, ax = plot_timeseries(sample_timeseries, markers=["o"])
        assert fig is not None
        plt.close(fig)

    def test_plot_without_legend(self, sample_timeseries_list: list[TimeSeries]) -> None:
        """Test plotting without legend."""
        fig, ax = plot_timeseries(sample_timeseries_list, legend=False)
        assert fig is not None
        plt.close(fig)

    def test_plot_without_grid(self, sample_timeseries: TimeSeries) -> None:
        """Test plotting without grid."""
        fig, ax = plot_timeseries(sample_timeseries, grid=False)
        assert fig is not None
        plt.close(fig)


class TestPlotTimeseriesComparison:
    """Tests for observed vs simulated comparison plots."""

    def test_comparison_plot(self, observed_simulated_pair: tuple[TimeSeries, TimeSeries]) -> None:
        """Test basic comparison plot."""
        observed, simulated = observed_simulated_pair
        fig, ax = plot_timeseries_comparison(observed, simulated)
        assert fig is not None
        plt.close(fig)

    def test_comparison_with_title(
        self, observed_simulated_pair: tuple[TimeSeries, TimeSeries]
    ) -> None:
        """Test comparison plot with title."""
        observed, simulated = observed_simulated_pair
        fig, ax = plot_timeseries_comparison(observed, simulated, title="Head Calibration")
        assert fig is not None
        plt.close(fig)

    def test_comparison_with_residuals(
        self, observed_simulated_pair: tuple[TimeSeries, TimeSeries]
    ) -> None:
        """Test comparison plot with residuals subplot."""
        observed, simulated = observed_simulated_pair
        fig, ax = plot_timeseries_comparison(observed, simulated, show_residuals=True)
        assert fig is not None
        plt.close(fig)

    def test_comparison_with_metrics(
        self, observed_simulated_pair: tuple[TimeSeries, TimeSeries]
    ) -> None:
        """Test comparison plot showing metrics."""
        observed, simulated = observed_simulated_pair
        fig, ax = plot_timeseries_comparison(observed, simulated, show_metrics=True)
        assert fig is not None
        plt.close(fig)


class TestPlotTimeseriesCollection:
    """Tests for time series collection plotting."""

    def test_plot_collection(self, sample_collection: TimeSeriesCollection) -> None:
        """Test plotting a time series collection."""
        fig, ax = plot_timeseries_collection(sample_collection)
        assert fig is not None
        plt.close(fig)

    def test_plot_specific_locations(self, sample_collection: TimeSeriesCollection) -> None:
        """Test plotting specific locations from collection."""
        fig, ax = plot_timeseries_collection(sample_collection, locations=["Well_1", "Well_2"])
        assert fig is not None
        plt.close(fig)


# =============================================================================
# Budget Plotting Tests
# =============================================================================


class TestPlotBudgetBar:
    """Tests for budget bar chart plotting."""

    def test_budget_bar_vertical(self, sample_budget_components: dict[str, float]) -> None:
        """Test vertical bar chart."""
        fig, ax = plot_budget_bar(sample_budget_components, orientation="vertical")
        assert fig is not None
        plt.close(fig)

    def test_budget_bar_horizontal(self, sample_budget_components: dict[str, float]) -> None:
        """Test horizontal bar chart."""
        fig, ax = plot_budget_bar(sample_budget_components, orientation="horizontal")
        assert fig is not None
        plt.close(fig)

    def test_budget_bar_with_title(self, sample_budget_components: dict[str, float]) -> None:
        """Test bar chart with custom title."""
        fig, ax = plot_budget_bar(sample_budget_components, title="Annual Water Budget")
        assert ax.get_title() == "Annual Water Budget"
        plt.close(fig)

    def test_budget_bar_without_values(self, sample_budget_components: dict[str, float]) -> None:
        """Test bar chart without value labels."""
        fig, ax = plot_budget_bar(sample_budget_components, show_values=False)
        assert fig is not None
        plt.close(fig)

    def test_budget_bar_custom_colors(self, sample_budget_components: dict[str, float]) -> None:
        """Test bar chart with custom colors."""
        fig, ax = plot_budget_bar(
            sample_budget_components,
            inflow_color="green",
            outflow_color="red",
        )
        assert fig is not None
        plt.close(fig)


class TestPlotBudgetStacked:
    """Tests for stacked budget chart plotting."""

    def test_budget_stacked(
        self, sample_budget_timeseries: tuple[np.ndarray, dict[str, np.ndarray]]
    ) -> None:
        """Test stacked area chart."""
        times, components = sample_budget_timeseries
        fig, ax = plot_budget_stacked(times, components)
        assert fig is not None
        plt.close(fig)

    def test_budget_stacked_with_title(
        self, sample_budget_timeseries: tuple[np.ndarray, dict[str, np.ndarray]]
    ) -> None:
        """Test stacked chart with title."""
        times, components = sample_budget_timeseries
        fig, ax = plot_budget_stacked(times, components, title="Monthly Budget")
        assert fig is not None
        plt.close(fig)

    def test_budget_stacked_without_legend(
        self, sample_budget_timeseries: tuple[np.ndarray, dict[str, np.ndarray]]
    ) -> None:
        """Test stacked chart without legend."""
        times, components = sample_budget_timeseries
        fig, ax = plot_budget_stacked(times, components, show_legend=False)
        assert fig is not None
        plt.close(fig)


class TestPlotBudgetPie:
    """Tests for budget pie chart plotting."""

    def test_budget_pie_both(self, sample_budget_components: dict[str, float]) -> None:
        """Test pie chart showing both inflows and outflows."""
        fig, ax = plot_budget_pie(sample_budget_components, budget_type="both")
        assert fig is not None
        plt.close(fig)

    def test_budget_pie_inflow(self, sample_budget_components: dict[str, float]) -> None:
        """Test pie chart showing only inflows."""
        fig, ax = plot_budget_pie(sample_budget_components, budget_type="inflow")
        assert fig is not None
        plt.close(fig)

    def test_budget_pie_outflow(self, sample_budget_components: dict[str, float]) -> None:
        """Test pie chart showing only outflows."""
        fig, ax = plot_budget_pie(sample_budget_components, budget_type="outflow")
        assert fig is not None
        plt.close(fig)


class TestPlotWaterBalance:
    """Tests for water balance summary plotting."""

    def test_water_balance(self) -> None:
        """Test water balance summary chart."""
        inflows = {"Precipitation": 1000, "Stream Inflow": 500, "Recharge": 200}
        outflows = {"ET": 600, "Pumping": 800, "Stream Outflow": 300}
        fig, ax = plot_water_balance(inflows, outflows, storage_change=-100)
        assert fig is not None
        plt.close(fig)

    def test_water_balance_with_title(self) -> None:
        """Test water balance with custom title."""
        inflows = {"Recharge": 500}
        outflows = {"Pumping": 400}
        fig, ax = plot_water_balance(inflows, outflows, title="Subregion 1 Balance")
        assert fig is not None
        plt.close(fig)


class TestPlotZBudget:
    """Tests for zone budget plotting."""

    def test_zbudget_bar(self, sample_zone_budgets: dict[int, dict[str, float]]) -> None:
        """Test zone budget bar chart."""
        fig, ax = plot_zbudget(sample_zone_budgets, plot_type="bar")
        assert fig is not None
        plt.close(fig)

    def test_zbudget_heatmap(self, sample_zone_budgets: dict[int, dict[str, float]]) -> None:
        """Test zone budget heatmap."""
        fig, ax = plot_zbudget(sample_zone_budgets, plot_type="heatmap")
        assert fig is not None
        plt.close(fig)

    def test_zbudget_with_title(self, sample_zone_budgets: dict[int, dict[str, float]]) -> None:
        """Test zone budget with custom title."""
        fig, ax = plot_zbudget(sample_zone_budgets, title="Subregion Budgets")
        assert ax.get_title() == "Subregion Budgets"
        plt.close(fig)


class TestPlotBudgetTimeseries:
    """Tests for budget time series line plots."""

    def test_budget_timeseries(
        self, sample_budget_timeseries: tuple[np.ndarray, dict[str, np.ndarray]]
    ) -> None:
        """Test budget time series line chart."""
        times, components = sample_budget_timeseries
        fig, ax = plot_budget_timeseries(times, components)
        assert fig is not None
        plt.close(fig)

    def test_budget_timeseries_cumulative(
        self, sample_budget_timeseries: tuple[np.ndarray, dict[str, np.ndarray]]
    ) -> None:
        """Test cumulative budget time series."""
        times, components = sample_budget_timeseries
        fig, ax = plot_budget_timeseries(times, components, cumulative=True)
        assert fig is not None
        plt.close(fig)

    def test_budget_timeseries_without_net(
        self, sample_budget_timeseries: tuple[np.ndarray, dict[str, np.ndarray]]
    ) -> None:
        """Test budget time series without net line."""
        times, components = sample_budget_timeseries
        fig, ax = plot_budget_timeseries(times, components, show_net=False)
        assert fig is not None
        plt.close(fig)


class TestBudgetPlotter:
    """Tests for BudgetPlotter class."""

    def test_plotter_creation(self, sample_budget_components: dict[str, float]) -> None:
        """Test creating a BudgetPlotter."""
        plotter = BudgetPlotter(sample_budget_components)
        assert plotter is not None

    def test_plotter_bar_chart(self, sample_budget_components: dict[str, float]) -> None:
        """Test creating bar chart through BudgetPlotter."""
        plotter = BudgetPlotter(sample_budget_components)
        fig, ax = plotter.bar_chart()
        assert fig is not None
        plt.close(fig)

    def test_plotter_pie_chart(self, sample_budget_components: dict[str, float]) -> None:
        """Test creating pie chart through BudgetPlotter."""
        plotter = BudgetPlotter(sample_budget_components)
        fig, ax = plotter.pie_chart()
        assert fig is not None
        plt.close(fig)

    def test_plotter_stacked_area(
        self, sample_budget_timeseries: tuple[np.ndarray, dict[str, np.ndarray]]
    ) -> None:
        """Test creating stacked area through BudgetPlotter."""
        times, components = sample_budget_timeseries
        plotter = BudgetPlotter(components, times=times)
        fig, ax = plotter.stacked_area()
        assert fig is not None
        plt.close(fig)

    def test_plotter_line_chart(
        self, sample_budget_timeseries: tuple[np.ndarray, dict[str, np.ndarray]]
    ) -> None:
        """Test creating line chart through BudgetPlotter."""
        times, components = sample_budget_timeseries
        plotter = BudgetPlotter(components, times=times)
        fig, ax = plotter.line_chart()
        assert fig is not None
        plt.close(fig)

    def test_plotter_save_figure(
        self, sample_budget_components: dict[str, float], tmp_path: Path
    ) -> None:
        """Test saving figure to file."""
        plotter = BudgetPlotter(sample_budget_components)
        output_file = tmp_path / "budget.png"
        plotter.save(output_file)
        assert output_file.exists()


# =========================================================================
# Additional tests to increase coverage to 95%+
# =========================================================================


class TestPlotMeshEdgeCases:
    """Tests for mesh plotting edge cases."""

    def test_plot_mesh_with_existing_axes(self, simple_grid: AppGrid) -> None:
        """Test plot_mesh with pre-created axes."""
        fig, ax = plt.subplots()
        fig2, ax2 = plot_mesh(simple_grid, ax=ax)
        assert fig2 is fig
        assert ax2 is ax
        plt.close(fig)

    def test_plot_mesh_no_edges(self, simple_grid: AppGrid) -> None:
        """Test plot_mesh with edges disabled."""
        fig, ax = plot_mesh(simple_grid, show_edges=False)
        assert fig is not None
        plt.close(fig)

    def test_plot_mesh_with_both_ids(self, simple_grid: AppGrid) -> None:
        """Test plot_mesh with both node and element IDs shown."""
        fig, ax = plot_mesh(
            simple_grid,
            show_node_ids=True,
            show_element_ids=True,
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_mesh_custom_figsize(self, simple_grid: AppGrid) -> None:
        """Test plot_mesh with custom figure size."""
        fig, ax = plot_mesh(simple_grid, figsize=(14, 10))
        assert fig is not None
        plt.close(fig)

    def test_plot_mesh_custom_colors(self, simple_grid: AppGrid) -> None:
        """Test plot_mesh with custom edge and fill colors."""
        fig, ax = plot_mesh(
            simple_grid,
            edge_color="red",
            fill_color="yellow",
            edge_width=2.0,
            alpha=0.5,
        )
        assert fig is not None
        plt.close(fig)


class TestPlotNodesEdgeCases:
    """Tests for node plotting edge cases."""

    def test_plot_nodes_with_existing_axes(self, simple_grid: AppGrid) -> None:
        """Test plot_nodes with pre-created axes."""
        fig, ax = plt.subplots()
        fig2, ax2 = plot_nodes(simple_grid, ax=ax)
        assert fig2 is fig
        plt.close(fig)

    def test_plot_nodes_custom_style(self, simple_grid: AppGrid) -> None:
        """Test plot_nodes with custom marker size and color."""
        fig, ax = plot_nodes(
            simple_grid,
            marker_size=50,
            color="green",
            figsize=(8, 6),
        )
        assert fig is not None
        plt.close(fig)


class TestPlotElementsEdgeCases:
    """Tests for element plotting edge cases."""

    def test_plot_elements_color_by_area(self, simple_grid: AppGrid) -> None:
        """Test coloring elements by area."""
        fig, ax = plot_elements(simple_grid, color_by="area")
        assert fig is not None
        plt.close(fig)

    def test_plot_elements_no_colorbar(self, simple_grid: AppGrid) -> None:
        """Test elements colored by subregion without colorbar."""
        fig, ax = plot_elements(
            simple_grid,
            color_by="subregion",
            show_colorbar=False,
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_elements_with_existing_axes(self, simple_grid: AppGrid) -> None:
        """Test plot_elements with pre-created axes."""
        fig, ax = plt.subplots()
        fig2, ax2 = plot_elements(simple_grid, ax=ax)
        assert fig2 is fig
        plt.close(fig)

    def test_plot_elements_custom_cmap(self, simple_grid: AppGrid) -> None:
        """Test elements with custom colormap."""
        fig, ax = plot_elements(simple_grid, color_by="subregion", cmap="coolwarm")
        assert fig is not None
        plt.close(fig)


class TestPlotScalarFieldEdgeCases:
    """Tests for scalar field plotting edge cases."""

    def test_plot_with_custom_vmin_vmax(self, simple_grid: AppGrid) -> None:
        """Test scalar field with custom value range."""
        values = np.arange(9, dtype=float) * 10
        fig, ax = plot_scalar_field(
            simple_grid,
            values,
            field_type="node",
            vmin=-10.0,
            vmax=100.0,
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_without_mesh_edges(self, simple_grid: AppGrid) -> None:
        """Test scalar field without mesh edge overlay."""
        values = np.arange(9, dtype=float) * 10
        fig, ax = plot_scalar_field(
            simple_grid,
            values,
            field_type="node",
            show_mesh=False,
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_cell_values_no_colorbar(self, simple_grid: AppGrid) -> None:
        """Test cell scalar field without colorbar."""
        values = np.array([10.0, 20.0, 15.0, 25.0])
        fig, ax = plot_scalar_field(
            simple_grid,
            values,
            field_type="cell",
            show_colorbar=False,
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_cell_values_no_mesh(self, simple_grid: AppGrid) -> None:
        """Test cell scalar field without mesh edges."""
        values = np.array([10.0, 20.0, 15.0, 25.0])
        fig, ax = plot_scalar_field(
            simple_grid,
            values,
            field_type="cell",
            show_mesh=False,
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_with_existing_axes(self, simple_grid: AppGrid) -> None:
        """Test scalar field with pre-created axes."""
        fig, ax = plt.subplots()
        values = np.arange(9, dtype=float) * 10
        fig2, ax2 = plot_scalar_field(simple_grid, values, ax=ax)
        assert fig2 is fig
        plt.close(fig)

    def test_plot_node_scalars_with_subdiv(self, simple_grid: AppGrid) -> None:
        """Test plotting node scalars with bilinear FE subdivision."""
        values = np.arange(9, dtype=float) * 10
        fig, ax = plot_scalar_field(simple_grid, values, field_type="node", n_subdiv=4)
        assert fig is not None
        plt.close(fig)

    def test_plot_node_scalars_subdiv_1(self, simple_grid: AppGrid) -> None:
        """Test n_subdiv=1 falls back to legacy diagonal-split path."""
        values = np.arange(9, dtype=float) * 10
        fig, ax = plot_scalar_field(simple_grid, values, field_type="node", n_subdiv=1)
        assert fig is not None
        plt.close(fig)

    def test_plot_cell_values_ignores_subdiv(self, simple_grid: AppGrid) -> None:
        """Test that n_subdiv is ignored for cell-valued fields."""
        values = np.array([10.0, 20.0, 15.0, 25.0])
        fig, ax = plot_scalar_field(simple_grid, values, field_type="cell", n_subdiv=4)
        assert fig is not None
        plt.close(fig)


class TestSubdivideQuads:
    """Tests for _subdivide_quads helper function."""

    def test_triangle_passthrough(self) -> None:
        """Triangle elements pass through unchanged."""
        x = np.array([0.0, 1.0, 0.5])
        y = np.array([0.0, 0.0, 1.0])
        values = np.array([1.0, 2.0, 3.0])
        elem_conn = [[0, 1, 2]]

        sx, sy, sv, stri = _subdivide_quads(elem_conn, x, y, values, n=4)
        assert len(sx) == 3
        assert stri.shape == (1, 3)
        np.testing.assert_array_almost_equal(sv, [1.0, 2.0, 3.0])

    def test_quad_subdivision_count(self) -> None:
        """n=4 gives 16 points and 18 triangles per quad."""
        x = np.array([0.0, 1.0, 1.0, 0.0])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        values = np.array([1.0, 2.0, 3.0, 4.0])
        elem_conn = [[0, 1, 2, 3]]

        sx, sy, sv, stri = _subdivide_quads(elem_conn, x, y, values, n=4)
        assert len(sx) == 16  # 4 x 4 = 16 points
        assert stri.shape == (18, 3)  # 2 * (4-1)^2 = 18 triangles

    def test_quad_corner_values_preserved(self) -> None:
        """Corner values of subdivided quad match original node values."""
        x = np.array([0.0, 2.0, 2.0, 0.0])
        y = np.array([0.0, 0.0, 2.0, 2.0])
        values = np.array([10.0, 20.0, 30.0, 40.0])
        elem_conn = [[0, 1, 2, 3]]

        sx, sy, sv, stri = _subdivide_quads(elem_conn, x, y, values, n=4)
        # The first point in the grid is at (-1,-1) → node 0
        # Find corner points
        corners = {(0.0, 0.0): 10.0, (2.0, 0.0): 20.0, (2.0, 2.0): 30.0, (0.0, 2.0): 40.0}
        for i in range(len(sx)):
            key = (round(sx[i], 6), round(sy[i], 6))
            if key in corners:
                np.testing.assert_almost_equal(sv[i], corners[key])


class TestPlotStreamsEdgeCases:
    """Tests for stream plotting edge cases."""

    def test_plot_streams_with_existing_axes(self, simple_stream: AppStream) -> None:
        """Test plot_streams with pre-created axes."""
        fig, ax = plt.subplots()
        fig2, ax2 = plot_streams(simple_stream, ax=ax)
        assert fig2 is fig
        plt.close(fig)

    def test_plot_streams_custom_style(self, simple_stream: AppStream) -> None:
        """Test plot_streams with custom line and node colors."""
        fig, ax = plot_streams(
            simple_stream,
            show_nodes=True,
            line_color="green",
            line_width=3.0,
            node_color="red",
            node_size=50,
        )
        assert fig is not None
        plt.close(fig)


class TestPlotBoundaryEdgeCases:
    """Tests for boundary plotting edge cases."""

    def test_plot_boundary_with_fill(self, simple_grid: AppGrid) -> None:
        """Test boundary with fill enabled using boundary nodes."""
        fig, ax = plot_boundary(simple_grid, fill=True)
        assert fig is not None
        plt.close(fig)

    def test_plot_boundary_with_existing_axes(self, simple_grid: AppGrid) -> None:
        """Test plot_boundary with pre-created axes."""
        fig, ax = plt.subplots()
        fig2, ax2 = plot_boundary(simple_grid, ax=ax)
        assert fig2 is fig
        plt.close(fig)

    def test_plot_boundary_custom_style(self, simple_grid: AppGrid) -> None:
        """Test boundary with custom line color and width."""
        fig, ax = plot_boundary(
            simple_grid,
            line_color="red",
            line_width=3.0,
            fill=True,
            fill_color="lightyellow",
            alpha=0.5,
        )
        assert fig is not None
        plt.close(fig)


class TestMeshPlotterEdgeCases:
    """Tests for MeshPlotter edge cases."""

    def test_plotter_composite_cell_values(self, simple_grid: AppGrid) -> None:
        """Test composite plot with cell values instead of node values."""
        plotter = MeshPlotter(simple_grid)
        cell_values = np.array([10.0, 20.0, 30.0, 40.0])
        fig, ax = plotter.plot_composite(
            show_mesh=True,
            cell_values=cell_values,
            title="Cell Values",
        )
        assert fig is not None
        plt.close(fig)

    def test_plotter_composite_mesh_only(self, simple_grid: AppGrid) -> None:
        """Test composite plot with no scalar values, mesh only."""
        plotter = MeshPlotter(simple_grid)
        fig, ax = plotter.plot_composite(
            show_mesh=True,
            title="Mesh Only",
        )
        assert fig is not None
        plt.close(fig)

    def test_plotter_composite_no_mesh_no_values(self, simple_grid: AppGrid) -> None:
        """Test composite plot with show_mesh=False and no values."""
        plotter = MeshPlotter(simple_grid)
        fig, ax = plotter.plot_composite(show_mesh=False)
        assert fig is not None
        plt.close(fig)

    def test_plotter_composite_with_streams(
        self, simple_grid: AppGrid, simple_stream: AppStream
    ) -> None:
        """Test composite plot with streams but no scalar values."""
        plotter = MeshPlotter(simple_grid, streams=simple_stream)
        fig, ax = plotter.plot_composite(
            show_mesh=True,
            show_streams=True,
        )
        assert fig is not None
        plt.close(fig)

    def test_plotter_save_without_prior_plot(self, simple_grid: AppGrid, tmp_path: Path) -> None:
        """Test save creates a default plot when none exists."""
        plotter = MeshPlotter(simple_grid)
        output_file = tmp_path / "auto_mesh.png"
        plotter.save(output_file)
        assert output_file.exists()

    def test_plotter_save_with_custom_dpi(self, simple_grid: AppGrid, tmp_path: Path) -> None:
        """Test save with custom DPI."""
        plotter = MeshPlotter(simple_grid)
        plotter.plot_mesh()
        output_file = tmp_path / "high_dpi_mesh.png"
        plotter.save(output_file, dpi=300)
        assert output_file.exists()

    def test_plotter_plot_mesh_without_streams(self, simple_grid: AppGrid) -> None:
        """Test MeshPlotter.plot_mesh with show_streams=True but no streams set."""
        plotter = MeshPlotter(simple_grid, streams=None)
        fig, ax = plotter.plot_mesh(show_streams=True)
        assert fig is not None
        plt.close(fig)


class TestPlotTimeseriesEdgeCases:
    """Tests for time series plotting edge cases."""

    def test_plot_with_date_format(self, sample_timeseries: TimeSeries) -> None:
        """Test plotting with custom date format."""
        fig, ax = plot_timeseries(
            sample_timeseries,
            date_format="%Y-%m",
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_with_ylabel(self, sample_timeseries: TimeSeries) -> None:
        """Test plotting with explicit ylabel."""
        fig, ax = plot_timeseries(
            sample_timeseries,
            ylabel="Head (ft)",
        )
        assert ax.get_ylabel() == "Head (ft)"
        plt.close(fig)

    def test_plot_with_existing_axes(self, sample_timeseries: TimeSeries) -> None:
        """Test plotting on pre-created axes."""
        fig, ax = plt.subplots()
        fig2, ax2 = plot_timeseries(sample_timeseries, ax=ax)
        assert fig2 is fig
        plt.close(fig)

    def test_plot_with_linestyles(self, sample_timeseries_list: list[TimeSeries]) -> None:
        """Test plotting with custom linestyles."""
        fig, ax = plot_timeseries(
            sample_timeseries_list,
            linestyles=["--", "-.", ":"],
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_single_series_no_legend(self, sample_timeseries: TimeSeries) -> None:
        """Test that legend is not shown for single series (even with legend=True)."""
        fig, ax = plot_timeseries(sample_timeseries, legend=True)
        assert fig is not None
        plt.close(fig)


class TestPlotTimeseriesComparisonEdgeCases:
    """Tests for comparison plot edge cases."""

    def test_comparison_with_existing_axes(
        self, observed_simulated_pair: tuple[TimeSeries, TimeSeries]
    ) -> None:
        """Test comparison plot with pre-created axes (no residuals)."""
        observed, simulated = observed_simulated_pair
        fig, ax = plt.subplots()
        fig2, ax2 = plot_timeseries_comparison(observed, simulated, ax=ax, show_residuals=False)
        assert fig2 is fig
        plt.close(fig)

    def test_comparison_without_metrics(
        self, observed_simulated_pair: tuple[TimeSeries, TimeSeries]
    ) -> None:
        """Test comparison plot without metrics display."""
        observed, simulated = observed_simulated_pair
        fig, ax = plot_timeseries_comparison(observed, simulated, show_metrics=False)
        assert fig is not None
        plt.close(fig)

    def test_comparison_custom_colors(
        self, observed_simulated_pair: tuple[TimeSeries, TimeSeries]
    ) -> None:
        """Test comparison with custom colors and marker."""
        observed, simulated = observed_simulated_pair
        fig, ax = plot_timeseries_comparison(
            observed,
            simulated,
            obs_color="green",
            sim_color="orange",
            obs_marker="s",
        )
        assert fig is not None
        plt.close(fig)


class TestPlotTimeseriesCollectionEdgeCases:
    """Tests for time series collection plotting edge cases."""

    def test_plot_collection_with_max_series(self, sample_collection: TimeSeriesCollection) -> None:
        """Test plotting collection limited by max_series."""
        fig, ax = plot_timeseries_collection(sample_collection, max_series=2)
        assert fig is not None
        plt.close(fig)

    def test_plot_collection_with_title(self, sample_collection: TimeSeriesCollection) -> None:
        """Test collection plot uses custom title."""
        fig, ax = plot_timeseries_collection(sample_collection, title="Custom Title")
        assert fig is not None
        plt.close(fig)

    def test_plot_collection_auto_title(self, sample_collection: TimeSeriesCollection) -> None:
        """Test collection plot auto-uses collection name as title."""
        fig, ax = plot_timeseries_collection(sample_collection)
        assert fig is not None
        plt.close(fig)


class TestPlotBudgetBarEdgeCases:
    """Tests for budget bar chart edge cases."""

    def test_budget_bar_with_existing_axes(
        self, sample_budget_components: dict[str, float]
    ) -> None:
        """Test budget bar with pre-created axes."""
        fig, ax = plt.subplots()
        fig2, ax2 = plot_budget_bar(sample_budget_components, ax=ax)
        assert fig2 is fig
        plt.close(fig)

    def test_budget_bar_horizontal_with_values(
        self, sample_budget_components: dict[str, float]
    ) -> None:
        """Test horizontal bar chart with value labels."""
        fig, ax = plot_budget_bar(
            sample_budget_components,
            orientation="horizontal",
            show_values=True,
        )
        assert fig is not None
        plt.close(fig)

    def test_budget_bar_horizontal_without_values(
        self, sample_budget_components: dict[str, float]
    ) -> None:
        """Test horizontal bar chart without value labels."""
        fig, ax = plot_budget_bar(
            sample_budget_components,
            orientation="horizontal",
            show_values=False,
        )
        assert fig is not None
        plt.close(fig)


class TestPlotBudgetStackedEdgeCases:
    """Tests for stacked budget chart edge cases."""

    def test_budget_stacked_with_existing_axes(
        self, sample_budget_timeseries: tuple[np.ndarray, dict[str, np.ndarray]]
    ) -> None:
        """Test stacked chart with pre-created axes."""
        times, components = sample_budget_timeseries
        fig, ax = plt.subplots()
        fig2, ax2 = plot_budget_stacked(times, components, ax=ax)
        assert fig2 is fig
        plt.close(fig)

    def test_budget_stacked_custom_alpha(
        self, sample_budget_timeseries: tuple[np.ndarray, dict[str, np.ndarray]]
    ) -> None:
        """Test stacked chart with custom transparency."""
        times, components = sample_budget_timeseries
        fig, ax = plot_budget_stacked(times, components, alpha=0.5)
        assert fig is not None
        plt.close(fig)


class TestPlotBudgetPieEdgeCases:
    """Tests for budget pie chart edge cases."""

    def test_budget_pie_without_show_values(
        self, sample_budget_components: dict[str, float]
    ) -> None:
        """Test pie chart without value labels."""
        fig, ax = plot_budget_pie(
            sample_budget_components,
            budget_type="inflow",
            show_values=False,
        )
        assert fig is not None
        plt.close(fig)

    def test_budget_pie_with_existing_axes_single(
        self, sample_budget_components: dict[str, float]
    ) -> None:
        """Test pie chart with existing axes for single chart mode."""
        fig, ax = plt.subplots()
        fig2, ax2 = plot_budget_pie(
            sample_budget_components,
            budget_type="outflow",
            ax=ax,
        )
        assert fig2 is fig
        plt.close(fig)

    def test_budget_pie_both_custom_cmap(self, sample_budget_components: dict[str, float]) -> None:
        """Test pie chart both with custom colormap."""
        fig, ax = plot_budget_pie(
            sample_budget_components,
            budget_type="both",
            cmap="Set2",
        )
        assert fig is not None
        plt.close(fig)


class TestPlotWaterBalanceEdgeCases:
    """Tests for water balance edge cases."""

    def test_water_balance_with_existing_axes(self) -> None:
        """Test water balance with pre-created axes."""
        inflows = {"Precip": 1000}
        outflows = {"ET": 800}
        fig, ax = plt.subplots()
        fig2, ax2 = plot_water_balance(inflows, outflows, ax=ax)
        assert fig2 is fig
        plt.close(fig)

    def test_water_balance_zero_storage(self) -> None:
        """Test water balance with zero storage change (default)."""
        inflows = {"Recharge": 500, "Precip": 200}
        outflows = {"Pumping": 600, "ET": 100}
        fig, ax = plot_water_balance(inflows, outflows)
        assert fig is not None
        plt.close(fig)

    def test_water_balance_custom_units(self) -> None:
        """Test water balance with custom units."""
        inflows = {"Precip": 1000}
        outflows = {"ET": 800}
        fig, ax = plot_water_balance(inflows, outflows, units="TAF")
        assert fig is not None
        plt.close(fig)


class TestPlotZBudgetEdgeCases:
    """Tests for zone budget plotting edge cases."""

    def test_zbudget_bar_with_existing_axes(
        self, sample_zone_budgets: dict[int, dict[str, float]]
    ) -> None:
        """Test zone budget bar chart with pre-created axes."""
        fig, ax = plt.subplots()
        fig2, ax2 = plot_zbudget(sample_zone_budgets, ax=ax, plot_type="bar")
        assert fig2 is fig
        plt.close(fig)

    def test_zbudget_heatmap_custom_cmap(
        self, sample_zone_budgets: dict[int, dict[str, float]]
    ) -> None:
        """Test zone budget heatmap with custom colormap."""
        fig, ax = plot_zbudget(
            sample_zone_budgets,
            plot_type="heatmap",
            cmap="coolwarm",
        )
        assert fig is not None
        plt.close(fig)


class TestPlotBudgetTimeseriesEdgeCases:
    """Tests for budget time series line plot edge cases."""

    def test_budget_timeseries_with_existing_axes(
        self, sample_budget_timeseries: tuple[np.ndarray, dict[str, np.ndarray]]
    ) -> None:
        """Test budget timeseries with pre-created axes."""
        times, components = sample_budget_timeseries
        fig, ax = plt.subplots()
        fig2, ax2 = plot_budget_timeseries(times, components, ax=ax)
        assert fig2 is fig
        plt.close(fig)

    def test_budget_timeseries_cumulative_with_net(
        self, sample_budget_timeseries: tuple[np.ndarray, dict[str, np.ndarray]]
    ) -> None:
        """Test cumulative budget timeseries with net line shown."""
        times, components = sample_budget_timeseries
        fig, ax = plot_budget_timeseries(times, components, cumulative=True, show_net=True)
        assert fig is not None
        plt.close(fig)

    def test_budget_timeseries_custom_units(
        self, sample_budget_timeseries: tuple[np.ndarray, dict[str, np.ndarray]]
    ) -> None:
        """Test budget timeseries with custom units label."""
        times, components = sample_budget_timeseries
        fig, ax = plot_budget_timeseries(times, components, units="TAF")
        assert fig is not None
        plt.close(fig)


class TestBudgetPlotterEdgeCases:
    """Tests for BudgetPlotter class edge cases."""

    def test_plotter_bar_chart_with_arrays(
        self, sample_budget_timeseries: tuple[np.ndarray, dict[str, np.ndarray]]
    ) -> None:
        """Test bar chart with array values (sums them)."""
        times, components = sample_budget_timeseries
        plotter = BudgetPlotter(components, times=times)
        fig, ax = plotter.bar_chart()
        assert fig is not None
        plt.close(fig)

    def test_plotter_pie_chart_with_arrays(
        self, sample_budget_timeseries: tuple[np.ndarray, dict[str, np.ndarray]]
    ) -> None:
        """Test pie chart with array values (sums them)."""
        times, components = sample_budget_timeseries
        plotter = BudgetPlotter(components, times=times)
        fig, ax = plotter.pie_chart()
        assert fig is not None
        plt.close(fig)

    def test_plotter_stacked_area_no_times_raises(
        self, sample_budget_components: dict[str, float]
    ) -> None:
        """Test stacked area without times raises ValueError."""
        plotter = BudgetPlotter(sample_budget_components)
        with pytest.raises(ValueError, match="Time array required"):
            plotter.stacked_area()

    def test_plotter_line_chart_no_times_raises(
        self, sample_budget_components: dict[str, float]
    ) -> None:
        """Test line chart without times raises ValueError."""
        plotter = BudgetPlotter(sample_budget_components)
        with pytest.raises(ValueError, match="Time array required"):
            plotter.line_chart()

    def test_plotter_stacked_area_with_scalar_values(self) -> None:
        """Test stacked area with scalar (non-array) budget values."""
        times = np.array(["2020-01-01", "2020-02-01", "2020-03-01"], dtype="datetime64[D]")
        plotter = BudgetPlotter({"Precip": 100.0, "ET": -50.0}, times=times)
        fig, ax = plotter.stacked_area()
        assert fig is not None
        plt.close(fig)

    def test_plotter_line_chart_with_scalar_values(self) -> None:
        """Test line chart with scalar (non-array) budget values."""
        times = np.array(["2020-01-01", "2020-02-01", "2020-03-01"], dtype="datetime64[D]")
        plotter = BudgetPlotter({"Precip": 100.0, "ET": -50.0}, times=times)
        fig, ax = plotter.line_chart()
        assert fig is not None
        plt.close(fig)

    def test_plotter_save_creates_default_plot(
        self, sample_budget_components: dict[str, float], tmp_path: Path
    ) -> None:
        """Test save auto-creates bar chart if no figure exists."""
        plotter = BudgetPlotter(sample_budget_components)
        output_file = tmp_path / "auto_budget.png"
        plotter.save(output_file)
        assert output_file.exists()

    def test_plotter_save_after_plot(
        self, sample_budget_components: dict[str, float], tmp_path: Path
    ) -> None:
        """Test save after explicitly creating a plot."""
        plotter = BudgetPlotter(sample_budget_components)
        plotter.bar_chart()
        output_file = tmp_path / "explicit_budget.png"
        plotter.save(output_file, dpi=72)
        assert output_file.exists()
