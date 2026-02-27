"""Sweep tests for pyiwfm.visualization.calibration_plots targeting remaining uncovered lines.

Covers:
- plot_metrics_table() (lines 313-340)
- plot_water_budget_summary() (lines 368-396)
- plot_zbudget_summary() (lines 428-430, 438)
- plot_typical_hydrographs() edge cases (line 491: all-NaN skip)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure


@pytest.fixture(autouse=True)
def _close_figures() -> None:  # type: ignore[misc]
    """Close all matplotlib figures after each test."""
    yield  # type: ignore[misc]
    plt.close("all")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _monthly_times_3() -> np.ndarray:
    """3 monthly timestamps as datetime64[s]."""
    return np.array(["2020-01-01", "2020-02-01", "2020-03-01"], dtype="datetime64[s]")


def _monthly_times_4() -> np.ndarray:
    """4 monthly timestamps as datetime64[s]."""
    return np.array(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"], dtype="datetime64[s]")


def _monthly_times_6() -> np.ndarray:
    """6 monthly timestamps as datetime64[s]."""
    return np.array(
        ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01", "2020-06-01"],
        dtype="datetime64[s]",
    )


# ---------------------------------------------------------------------------
# Minimal mock for ComparisonMetrics
# ---------------------------------------------------------------------------


@dataclass
class _FakeMetrics:
    """Minimal stand-in for ComparisonMetrics with fields used by plot_metrics_table."""

    rmse: float
    scaled_rmse: float
    mae: float
    mbe: float
    nash_sutcliffe: float
    correlation: float
    n_points: int


# ---------------------------------------------------------------------------
# plot_metrics_table
# ---------------------------------------------------------------------------


class TestPlotMetricsTable:
    def test_basic_table(self) -> None:
        from pyiwfm.visualization.calibration_plots import plot_metrics_table

        metrics = {
            "W1": _FakeMetrics(
                rmse=2.5,
                scaled_rmse=0.05,
                mae=2.0,
                mbe=-0.3,
                nash_sutcliffe=0.85,
                correlation=0.95,
                n_points=100,
            ),
            "W2": _FakeMetrics(
                rmse=3.1,
                scaled_rmse=0.07,
                mae=2.8,
                mbe=0.5,
                nash_sutcliffe=0.78,
                correlation=0.91,
                n_points=80,
            ),
        }
        fig = plot_metrics_table(metrics)  # type: ignore[arg-type]
        assert isinstance(fig, Figure)

    def test_single_well(self) -> None:
        from pyiwfm.visualization.calibration_plots import plot_metrics_table

        metrics = {
            "Well_A": _FakeMetrics(
                rmse=1.0,
                scaled_rmse=0.02,
                mae=0.8,
                mbe=-0.1,
                nash_sutcliffe=0.95,
                correlation=0.99,
                n_points=200,
            ),
        }
        fig = plot_metrics_table(metrics)  # type: ignore[arg-type]
        assert isinstance(fig, Figure)

    def test_save_to_file(self, tmp_path: Path) -> None:
        from pyiwfm.visualization.calibration_plots import plot_metrics_table

        metrics = {
            "W1": _FakeMetrics(
                rmse=2.0,
                scaled_rmse=0.04,
                mae=1.5,
                mbe=0.0,
                nash_sutcliffe=0.90,
                correlation=0.96,
                n_points=50,
            ),
        }
        out = tmp_path / "metrics.png"
        fig = plot_metrics_table(metrics, output_path=out)  # type: ignore[arg-type]
        assert isinstance(fig, Figure)
        assert out.exists()


# ---------------------------------------------------------------------------
# plot_water_budget_summary
# ---------------------------------------------------------------------------


class TestPlotWaterBudgetSummary:
    def test_inflows_and_outflows(self) -> None:
        from pyiwfm.visualization.calibration_plots import plot_water_budget_summary

        times = _monthly_times_6()
        budget_data = {
            "Recharge": np.array([100.0, 120.0, 110.0, 130.0, 105.0, 115.0]),
            "Stream Leakage": np.array([50.0, 55.0, 60.0, 45.0, 50.0, 52.0]),
            "Pumping": np.array([-80.0, -90.0, -85.0, -95.0, -88.0, -82.0]),
            "ET": np.array([-30.0, -35.0, -40.0, -38.0, -32.0, -28.0]),
        }
        fig = plot_water_budget_summary(budget_data, times)
        assert isinstance(fig, Figure)

    def test_inflows_only(self) -> None:
        from pyiwfm.visualization.calibration_plots import plot_water_budget_summary

        times = _monthly_times_3()
        budget_data = {
            "Recharge": np.array([100.0, 120.0, 110.0]),
        }
        fig = plot_water_budget_summary(budget_data, times)
        assert isinstance(fig, Figure)

    def test_outflows_only(self) -> None:
        from pyiwfm.visualization.calibration_plots import plot_water_budget_summary

        times = _monthly_times_3()
        budget_data = {
            "Pumping": np.array([-80.0, -90.0, -85.0]),
        }
        fig = plot_water_budget_summary(budget_data, times)
        assert isinstance(fig, Figure)

    def test_save_to_file(self, tmp_path: Path) -> None:
        from pyiwfm.visualization.calibration_plots import plot_water_budget_summary

        times = _monthly_times_3()
        budget_data = {
            "Recharge": np.array([100.0, 120.0, 110.0]),
            "Pumping": np.array([-80.0, -90.0, -85.0]),
        }
        out = tmp_path / "budget.png"
        fig = plot_water_budget_summary(budget_data, times, output_path=out)
        assert isinstance(fig, Figure)
        assert out.exists()


# ---------------------------------------------------------------------------
# plot_zbudget_summary
# ---------------------------------------------------------------------------


class TestPlotZbudgetSummary:
    def test_single_zone(self) -> None:
        from pyiwfm.visualization.calibration_plots import plot_zbudget_summary

        times = _monthly_times_3()
        zone_budgets = {
            "Zone 1": {
                "Recharge": np.array([50.0, 60.0, 55.0]),
                "Pumping": np.array([-30.0, -35.0, -32.0]),
            },
        }
        fig = plot_zbudget_summary(zone_budgets, times)
        assert isinstance(fig, Figure)

    def test_multiple_zones(self) -> None:
        from pyiwfm.visualization.calibration_plots import plot_zbudget_summary

        times = _monthly_times_3()
        zone_budgets = {
            "Zone A": {
                "Recharge": np.array([50.0, 60.0, 55.0]),
                "Pumping": np.array([-30.0, -35.0, -32.0]),
            },
            "Zone B": {
                "Inflow": np.array([20.0, 25.0, 22.0]),
            },
            "Zone C": {
                "Outflow": np.array([-10.0, -15.0, -12.0]),
                "ET": np.array([-5.0, -8.0, -6.0]),
            },
            "Zone D": {
                "Recharge": np.array([40.0, 45.0, 42.0]),
            },
        }
        fig = plot_zbudget_summary(zone_budgets, times)
        assert isinstance(fig, Figure)

    def test_save_to_file(self, tmp_path: Path) -> None:
        from pyiwfm.visualization.calibration_plots import plot_zbudget_summary

        times = _monthly_times_3()
        zone_budgets = {
            "Zone 1": {
                "Recharge": np.array([50.0, 60.0, 55.0]),
            },
        }
        out = tmp_path / "zbudget.png"
        fig = plot_zbudget_summary(zone_budgets, times, output_path=out)
        assert isinstance(fig, Figure)
        assert out.exists()


# ---------------------------------------------------------------------------
# plot_typical_hydrographs -- edge case: all-NaN series should be skipped
# ---------------------------------------------------------------------------


class TestPlotTypicalHydrographsEdgeCases:
    def test_all_nan_hydrograph_skipped(self) -> None:
        """A hydrograph where all values are NaN should not raise."""
        from pyiwfm.calibration.calctyphyd import CalcTypHydResult, TypicalHydrograph
        from pyiwfm.visualization.calibration_plots import plot_typical_hydrographs

        times = _monthly_times_3()
        # One valid, one all-NaN
        hydrographs = [
            TypicalHydrograph(
                cluster_id=0,
                times=times,
                values=np.array([1.0, 2.0, 3.0]),
                contributing_wells=["W1", "W2"],
            ),
            TypicalHydrograph(
                cluster_id=1,
                times=times,
                values=np.array([np.nan, np.nan, np.nan]),
                contributing_wells=["W3"],
            ),
        ]
        result = CalcTypHydResult(hydrographs=hydrographs, well_means={"W1": 50.0, "W2": 55.0})
        fig, ax = plot_typical_hydrographs(result)
        assert isinstance(fig, Figure)
        # Only one valid hydrograph line should be drawn
        assert len(ax.lines) >= 1

    def test_mixed_nan_values(self) -> None:
        """Hydrograph with some NaN values should still plot valid points."""
        from pyiwfm.calibration.calctyphyd import CalcTypHydResult, TypicalHydrograph
        from pyiwfm.visualization.calibration_plots import plot_typical_hydrographs

        times = _monthly_times_4()
        hydrographs = [
            TypicalHydrograph(
                cluster_id=0,
                times=times,
                values=np.array([1.0, np.nan, 3.0, 2.0]),
                contributing_wells=["W1"],
            ),
        ]
        result = CalcTypHydResult(hydrographs=hydrographs, well_means={"W1": 50.0})
        fig, ax = plot_typical_hydrographs(result)
        assert isinstance(fig, Figure)
