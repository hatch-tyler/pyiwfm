"""Tests for calibration plotting functions."""

from __future__ import annotations

import matplotlib
import numpy as np
from matplotlib.figure import Figure

matplotlib.use("Agg")


class TestPlotOneToOne:
    """Tests for plot_one_to_one function."""

    def test_returns_figure_and_axes(self) -> None:
        from pyiwfm.visualization.plotting import plot_one_to_one

        obs = np.array([100.0, 110.0, 105.0, 115.0, 108.0])
        sim = np.array([102.0, 108.0, 106.0, 113.0, 109.0])

        fig, ax = plot_one_to_one(obs, sim)
        assert isinstance(fig, Figure)
        assert ax is not None

    def test_with_title_and_units(self) -> None:
        from pyiwfm.visualization.plotting import plot_one_to_one

        obs = np.array([50.0, 60.0, 55.0])
        sim = np.array([52.0, 58.0, 56.0])

        fig, ax = plot_one_to_one(obs, sim, title="Head Comparison", units="ft")
        assert ax.get_title() == "Head Comparison"
        assert "ft" in ax.get_xlabel()

    def test_no_metrics(self) -> None:
        from pyiwfm.visualization.plotting import plot_one_to_one

        obs = np.array([100.0, 200.0])
        sim = np.array([110.0, 190.0])

        fig, ax = plot_one_to_one(obs, sim, show_metrics=False)
        assert isinstance(fig, Figure)

    def test_color_by(self) -> None:
        from pyiwfm.visualization.plotting import plot_one_to_one

        obs = np.array([100.0, 110.0, 120.0])
        sim = np.array([102.0, 108.0, 118.0])
        colors = np.array([1.0, 2.0, 3.0])

        fig, ax = plot_one_to_one(obs, sim, color_by=colors)
        assert isinstance(fig, Figure)


class TestPlotResidualCdf:
    """Tests for plot_residual_cdf function."""

    def test_returns_figure_and_axes(self) -> None:
        from pyiwfm.visualization.plotting import plot_residual_cdf

        rng = np.random.default_rng(42)
        residuals = rng.normal(0, 5, 100)
        fig, ax = plot_residual_cdf(residuals)
        assert isinstance(fig, Figure)
        assert ax is not None

    def test_with_percentile_lines(self) -> None:
        from pyiwfm.visualization.plotting import plot_residual_cdf

        residuals = np.linspace(-10, 10, 50)
        fig, ax = plot_residual_cdf(residuals, show_percentile_lines=True)
        assert isinstance(fig, Figure)

    def test_without_percentile_lines(self) -> None:
        from pyiwfm.visualization.plotting import plot_residual_cdf

        residuals = np.array([1.0, -1.0, 2.0, -2.0, 0.5])
        fig, ax = plot_residual_cdf(residuals, show_percentile_lines=False)
        assert isinstance(fig, Figure)

    def test_custom_title(self) -> None:
        from pyiwfm.visualization.plotting import plot_residual_cdf

        residuals = np.array([1.0, -1.0, 2.0])
        fig, ax = plot_residual_cdf(residuals, title="My CDF")
        assert ax.get_title() == "My CDF"


class TestPlotResidualHistogram:
    """Tests for plot_residual_histogram function."""

    def test_returns_figure_and_axes(self) -> None:
        from pyiwfm.visualization.calibration_plots import plot_residual_histogram

        rng = np.random.default_rng(42)
        residuals = rng.normal(0, 5, 100)
        fig, ax = plot_residual_histogram(residuals)
        assert isinstance(fig, Figure)

    def test_with_normal_fit(self) -> None:
        from pyiwfm.visualization.calibration_plots import plot_residual_histogram

        rng = np.random.default_rng(42)
        residuals = rng.normal(2, 3, 100)
        fig, ax = plot_residual_histogram(residuals, show_normal_fit=True)
        assert isinstance(fig, Figure)

    def test_without_normal_fit(self) -> None:
        from pyiwfm.visualization.calibration_plots import plot_residual_histogram

        residuals = np.array([1.0, -1.0, 2.0, -2.0, 0.5])
        fig, ax = plot_residual_histogram(residuals, show_normal_fit=False)
        assert isinstance(fig, Figure)


class TestPlotMetricsTable:
    """Tests for plot_metrics_table function."""

    def test_returns_figure(self) -> None:
        from pyiwfm.comparison.metrics import ComparisonMetrics
        from pyiwfm.visualization.calibration_plots import plot_metrics_table

        obs = np.array([100.0, 110.0, 105.0])
        sim = np.array([102.0, 108.0, 106.0])
        metrics = ComparisonMetrics.compute(obs, sim)

        fig = plot_metrics_table({"W1": metrics, "W2": metrics})
        assert isinstance(fig, Figure)


class TestPlotHydrographPanel:
    """Tests for plot_hydrograph_panel function."""

    def test_returns_figure(self) -> None:
        from pyiwfm.visualization.calibration_plots import plot_hydrograph_panel

        times = np.array(["2020-01-01", "2020-06-01", "2020-12-01"], dtype="datetime64")
        comparisons = {
            "W1": (times, np.array([100.0, 105.0, 102.0]), np.array([101.0, 104.0, 103.0])),
            "W2": (times, np.array([200.0, 210.0, 205.0]), np.array([202.0, 208.0, 206.0])),
        }

        fig = plot_hydrograph_panel(comparisons, n_cols=2)
        assert isinstance(fig, Figure)


class TestPlotTypicalHydrographs:
    """Tests for plot_typical_hydrographs function."""

    def test_returns_figure(self) -> None:
        from pyiwfm.calibration.calctyphyd import CalcTypHydResult, TypicalHydrograph
        from pyiwfm.visualization.calibration_plots import plot_typical_hydrographs

        th = TypicalHydrograph(
            cluster_id=0,
            times=np.array(
                ["2000-01-15", "2000-04-15", "2000-07-15", "2000-10-15"], dtype="datetime64[D]"
            ),
            values=np.array([1.0, -0.5, -1.0, 0.5]),
            contributing_wells=["W1", "W2"],
        )
        result = CalcTypHydResult(hydrographs=[th], well_means={"W1": 100.0, "W2": 200.0})

        fig, ax = plot_typical_hydrographs(result)
        assert isinstance(fig, Figure)


class TestPlotClusterMap:
    """Tests for plot_cluster_map function."""

    def test_returns_figure(self) -> None:
        from pyiwfm.calibration.clustering import ClusteringResult
        from pyiwfm.visualization.calibration_plots import plot_cluster_map

        result = ClusteringResult(
            membership=np.array([[0.9, 0.1], [0.3, 0.7], [0.5, 0.5]]),
            cluster_centers=np.zeros((2, 5)),
            well_ids=["W0", "W1", "W2"],
            n_clusters=2,
            fpc=0.7,
        )
        locations = {"W0": (100.0, 200.0), "W1": (300.0, 400.0), "W2": (200.0, 300.0)}

        fig, ax = plot_cluster_map(locations, result)
        assert isinstance(fig, Figure)
