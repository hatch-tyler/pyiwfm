"""Deep tests for calibration_plots functions."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

matplotlib.use("Agg")

from pyiwfm.calibration.calctyphyd import CalcTypHydResult, TypicalHydrograph
from pyiwfm.calibration.clustering import ClusteringResult
from pyiwfm.visualization.calibration_plots import (
    plot_calibration_summary,
    plot_cluster_map,
    plot_hydrograph_panel,
    plot_typical_hydrographs,
)


@pytest.fixture(autouse=True)
def _close_figures() -> None:  # type: ignore[misc]
    """Close all matplotlib figures after each test."""
    yield  # type: ignore[misc]
    plt.close("all")


# ---------------------------------------------------------------------------
# plot_calibration_summary
# ---------------------------------------------------------------------------


class TestPlotCalibrationSummary:
    def test_returns_figures(self) -> None:
        well_comparisons = {
            "W1": (np.array([10.0, 20.0, 30.0]), np.array([11.0, 19.0, 31.0])),
            "W2": (np.array([5.0, 15.0]), np.array([6.0, 14.0])),
        }
        figs = plot_calibration_summary(well_comparisons)
        assert len(figs) >= 1
        assert all(isinstance(f, Figure) for f in figs)

    def test_saves_png(self, tmp_path: Path) -> None:
        well_comparisons = {
            "W1": (np.array([10.0, 20.0]), np.array([12.0, 18.0])),
        }
        figs = plot_calibration_summary(well_comparisons, output_dir=tmp_path)
        assert (tmp_path / "calibration_summary.png").exists()
        assert len(figs) >= 1


# ---------------------------------------------------------------------------
# plot_hydrograph_panel
# ---------------------------------------------------------------------------


class TestPlotHydrographPanel:
    def test_returns_figure(self) -> None:
        times = np.array(["2020-01-01", "2020-07-01"], dtype="datetime64[D]")
        comparisons = {
            "W1": (times, np.array([50.0, 55.0]), np.array([51.0, 54.0])),
            "W2": (times, np.array([30.0, 35.0]), np.array([31.0, 34.0])),
        }
        fig = plot_hydrograph_panel(comparisons)
        assert isinstance(fig, Figure)

    def test_single_panel(self) -> None:
        times = np.array(["2020-01-01", "2020-07-01"], dtype="datetime64[D]")
        comparisons = {
            "W1": (times, np.array([50.0, 55.0]), np.array([51.0, 54.0])),
        }
        fig = plot_hydrograph_panel(comparisons, n_cols=1, max_panels=1)
        assert isinstance(fig, Figure)

    def test_saves_png(self, tmp_path: Path) -> None:
        times = np.array(["2020-01-01", "2020-07-01"], dtype="datetime64[D]")
        comparisons = {
            "W1": (times, np.array([50.0, 55.0]), np.array([51.0, 54.0])),
        }
        out = tmp_path / "hydrographs.png"
        fig = plot_hydrograph_panel(comparisons, output_path=out)
        assert out.exists()
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# plot_cluster_map
# ---------------------------------------------------------------------------


class TestPlotClusterMap:
    def test_returns_figure_axes(self) -> None:
        well_locations = {"W1": (100.0, 200.0), "W2": (300.0, 400.0)}
        clustering = ClusteringResult(
            membership=np.array([[0.8, 0.2], [0.3, 0.7]]),
            cluster_centers=np.array([[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]]),
            well_ids=["W1", "W2"],
            n_clusters=2,
            fpc=0.65,
        )
        fig, ax = plot_cluster_map(well_locations, clustering)
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# plot_typical_hydrographs
# ---------------------------------------------------------------------------


class TestPlotTypicalHydrographs:
    def test_returns_figure_axes(self) -> None:
        times = np.array(
            ["2000-01-15", "2000-04-15", "2000-07-15", "2000-10-15"],
            dtype="datetime64[D]",
        )
        hydrographs = [
            TypicalHydrograph(
                cluster_id=0,
                times=times,
                values=np.array([1.0, -0.5, -1.0, 0.5]),
                contributing_wells=["W1", "W2"],
            ),
            TypicalHydrograph(
                cluster_id=1,
                times=times,
                values=np.array([-0.3, 0.8, 0.2, -0.7]),
                contributing_wells=["W3"],
            ),
        ]
        result = CalcTypHydResult(
            hydrographs=hydrographs,
            well_means={"W1": 50.0, "W2": 60.0, "W3": 45.0},
        )
        fig, ax = plot_typical_hydrographs(result)
        assert isinstance(fig, Figure)
