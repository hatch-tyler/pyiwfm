"""Unit tests for comparison metrics."""

from __future__ import annotations

import numpy as np

from pyiwfm.comparison.metrics import (
    ComparisonMetrics,
    SpatialComparison,
    TimeSeriesComparison,
    correlation_coefficient,
    mae,
    max_error,
    mbe,
    nash_sutcliffe,
    percent_bias,
    relative_error,
    rmse,
)


class TestBasicMetrics:
    """Tests for basic metric functions."""

    def test_rmse_identical(self) -> None:
        """Test RMSE of identical arrays."""
        observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        simulated = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.isclose(rmse(observed, simulated), 0.0)

    def test_rmse_different(self) -> None:
        """Test RMSE of different arrays."""
        observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        simulated = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        assert np.isclose(rmse(observed, simulated), 0.1)

    def test_mae_identical(self) -> None:
        """Test MAE of identical arrays."""
        observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        simulated = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.isclose(mae(observed, simulated), 0.0)

    def test_mae_different(self) -> None:
        """Test MAE of different arrays."""
        observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        simulated = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        assert np.isclose(mae(observed, simulated), 0.5)

    def test_mbe_positive_bias(self) -> None:
        """Test MBE with positive bias (over-prediction)."""
        observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        simulated = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        assert np.isclose(mbe(observed, simulated), 0.5)

    def test_mbe_negative_bias(self) -> None:
        """Test MBE with negative bias (under-prediction)."""
        observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        simulated = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        assert np.isclose(mbe(observed, simulated), -0.5)

    def test_nash_sutcliffe_perfect(self) -> None:
        """Test Nash-Sutcliffe efficiency for perfect model."""
        observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        simulated = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.isclose(nash_sutcliffe(observed, simulated), 1.0)

    def test_nash_sutcliffe_mean_model(self) -> None:
        """Test Nash-Sutcliffe efficiency for mean model."""
        observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        simulated = np.full(5, observed.mean())
        assert np.isclose(nash_sutcliffe(observed, simulated), 0.0)

    def test_nash_sutcliffe_poor(self) -> None:
        """Test Nash-Sutcliffe efficiency for poor model."""
        observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        simulated = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        nse = nash_sutcliffe(observed, simulated)
        assert nse < 0.0  # Worse than mean model

    def test_percent_bias_positive(self) -> None:
        """Test percent bias with positive bias."""
        observed = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        simulated = np.array([110.0, 220.0, 330.0, 440.0, 550.0])
        assert np.isclose(percent_bias(observed, simulated), 10.0)

    def test_percent_bias_negative(self) -> None:
        """Test percent bias with negative bias."""
        observed = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        simulated = np.array([90.0, 180.0, 270.0, 360.0, 450.0])
        assert np.isclose(percent_bias(observed, simulated), -10.0)

    def test_correlation_perfect(self) -> None:
        """Test correlation coefficient for perfect correlation."""
        observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        simulated = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.isclose(correlation_coefficient(observed, simulated), 1.0)

    def test_correlation_inverse(self) -> None:
        """Test correlation coefficient for inverse correlation."""
        observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        simulated = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        assert np.isclose(correlation_coefficient(observed, simulated), -1.0)

    def test_relative_error(self) -> None:
        """Test relative error calculation."""
        observed = np.array([100.0, 200.0, 300.0])
        simulated = np.array([110.0, 180.0, 330.0])
        rel_err = relative_error(observed, simulated)
        expected = np.array([0.10, -0.10, 0.10])
        np.testing.assert_array_almost_equal(rel_err, expected)

    def test_max_error(self) -> None:
        """Test maximum absolute error."""
        observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        simulated = np.array([1.1, 2.5, 3.0, 4.2, 5.0])
        assert np.isclose(max_error(observed, simulated), 0.5)


class TestComparisonMetrics:
    """Tests for ComparisonMetrics class."""

    def test_compute_all_metrics(self) -> None:
        """Test computing all metrics at once."""
        observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        simulated = np.array([1.1, 2.2, 3.1, 4.2, 5.1])
        metrics = ComparisonMetrics.compute(observed, simulated)

        assert hasattr(metrics, "rmse")
        assert hasattr(metrics, "mae")
        assert hasattr(metrics, "mbe")
        assert hasattr(metrics, "nash_sutcliffe")
        assert hasattr(metrics, "percent_bias")
        assert hasattr(metrics, "correlation")

    def test_metrics_to_dict(self) -> None:
        """Test converting metrics to dictionary."""
        observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        simulated = np.array([1.1, 2.2, 3.1, 4.2, 5.1])
        metrics = ComparisonMetrics.compute(observed, simulated)
        d = metrics.to_dict()

        assert isinstance(d, dict)
        assert "rmse" in d
        assert "mae" in d
        assert "mbe" in d

    def test_metrics_summary(self) -> None:
        """Test metrics summary string."""
        observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        simulated = np.array([1.1, 2.2, 3.1, 4.2, 5.1])
        metrics = ComparisonMetrics.compute(observed, simulated)
        summary = metrics.summary()

        assert isinstance(summary, str)
        assert "RMSE" in summary
        assert "MAE" in summary

    def test_metrics_rating(self) -> None:
        """Test metrics rating (good/fair/poor)."""
        observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        simulated = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        metrics = ComparisonMetrics.compute(observed, simulated)

        assert metrics.rating() in ["excellent", "good", "fair", "poor"]


class TestTimeSeriesComparison:
    """Tests for time series comparison."""

    def test_timeseries_comparison(self) -> None:
        """Test comparing time series data."""
        times = np.arange(10)
        observed = np.sin(times * 0.5) + 10
        simulated = np.sin(times * 0.5 + 0.1) + 10

        comparison = TimeSeriesComparison(
            times=times,
            observed=observed,
            simulated=simulated,
        )

        assert comparison.metrics is not None
        assert comparison.n_points == 10

    def test_timeseries_residuals(self) -> None:
        """Test getting residuals from time series comparison."""
        times = np.arange(10)
        observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        simulated = np.array([1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1])

        comparison = TimeSeriesComparison(
            times=times,
            observed=observed,
            simulated=simulated,
        )

        residuals = comparison.residuals
        assert len(residuals) == 10
        np.testing.assert_array_almost_equal(residuals, np.full(10, 0.1))

    def test_timeseries_with_missing_data(self) -> None:
        """Test time series comparison with missing data."""
        times = np.arange(10)
        observed = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, np.nan, 10.0])
        simulated = np.array([1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1])

        comparison = TimeSeriesComparison(
            times=times,
            observed=observed,
            simulated=simulated,
        )

        assert comparison.n_valid_points == 8  # 2 NaN values excluded


class TestSpatialComparison:
    """Tests for spatial field comparison."""

    def test_spatial_comparison(self) -> None:
        """Test comparing spatial fields."""
        x = np.array([0.0, 100.0, 200.0, 0.0, 100.0, 200.0])
        y = np.array([0.0, 0.0, 0.0, 100.0, 100.0, 100.0])
        observed = np.array([10.0, 20.0, 30.0, 15.0, 25.0, 35.0])
        simulated = np.array([11.0, 21.0, 31.0, 16.0, 26.0, 36.0])

        comparison = SpatialComparison(
            x=x,
            y=y,
            observed=observed,
            simulated=simulated,
        )

        assert comparison.metrics is not None
        assert comparison.n_points == 6

    def test_spatial_error_field(self) -> None:
        """Test getting spatial error field."""
        x = np.array([0.0, 100.0, 200.0])
        y = np.array([0.0, 0.0, 0.0])
        observed = np.array([10.0, 20.0, 30.0])
        simulated = np.array([11.0, 22.0, 33.0])

        comparison = SpatialComparison(
            x=x,
            y=y,
            observed=observed,
            simulated=simulated,
        )

        errors = comparison.error_field
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(errors, expected)

    def test_spatial_relative_error_field(self) -> None:
        """Test getting spatial relative error field."""
        x = np.array([0.0, 100.0, 200.0])
        y = np.array([0.0, 0.0, 0.0])
        observed = np.array([100.0, 200.0, 300.0])
        simulated = np.array([110.0, 220.0, 330.0])

        comparison = SpatialComparison(
            x=x,
            y=y,
            observed=observed,
            simulated=simulated,
        )

        rel_errors = comparison.relative_error_field
        expected = np.array([0.1, 0.1, 0.1])
        np.testing.assert_array_almost_equal(rel_errors, expected)

    def test_spatial_statistics_by_region(self) -> None:
        """Test computing statistics by region."""
        x = np.array([0.0, 100.0, 200.0, 0.0, 100.0, 200.0])
        y = np.array([0.0, 0.0, 0.0, 100.0, 100.0, 100.0])
        observed = np.array([10.0, 20.0, 30.0, 15.0, 25.0, 35.0])
        simulated = np.array([11.0, 21.0, 31.0, 16.0, 26.0, 36.0])
        regions = np.array([1, 1, 1, 2, 2, 2])

        comparison = SpatialComparison(
            x=x,
            y=y,
            observed=observed,
            simulated=simulated,
        )

        regional_metrics = comparison.metrics_by_region(regions)
        assert 1 in regional_metrics
        assert 2 in regional_metrics


# ── Additional tests for increased coverage ──────────────────────────


class TestNashSutcliffeEdgeCases:
    """Tests for Nash-Sutcliffe edge cases (zero variance)."""

    def test_nash_sutcliffe_constant_observed_perfect(self) -> None:
        """Test NSE when observed is constant and simulated matches exactly."""
        observed = np.array([5.0, 5.0, 5.0, 5.0])
        simulated = np.array([5.0, 5.0, 5.0, 5.0])
        # denominator=0, numerator=0 => returns 1.0
        assert nash_sutcliffe(observed, simulated) == 1.0

    def test_nash_sutcliffe_constant_observed_imperfect(self) -> None:
        """Test NSE when observed is constant but simulated differs."""
        observed = np.array([5.0, 5.0, 5.0, 5.0])
        simulated = np.array([5.0, 5.1, 5.0, 5.0])
        # denominator=0, numerator>0 => returns -inf
        result = nash_sutcliffe(observed, simulated)
        assert result == -np.inf


class TestRelativeErrorEdgeCases:
    """Tests for relative_error edge cases."""

    def test_relative_error_zero_observed(self) -> None:
        """Test relative error when observed contains zeros."""
        observed = np.array([0.0, 100.0, 0.0])
        simulated = np.array([10.0, 110.0, 0.0])
        rel_err = relative_error(observed, simulated)
        # Where observed=0, result should be 0.0
        assert rel_err[0] == 0.0
        assert rel_err[2] == 0.0
        assert np.isclose(rel_err[1], 0.1)

    def test_relative_error_all_zeros(self) -> None:
        """Test relative error when all observed values are zero."""
        observed = np.zeros(5)
        simulated = np.ones(5)
        rel_err = relative_error(observed, simulated)
        np.testing.assert_array_equal(rel_err, np.zeros(5))


class TestComparisonMetricsRatings:
    """Tests for ComparisonMetrics rating method across all categories."""

    def test_rating_excellent(self) -> None:
        """Test 'excellent' rating (NSE >= 0.90)."""
        observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        simulated = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        metrics = ComparisonMetrics.compute(observed, simulated)
        assert metrics.rating() == "excellent"

    def test_rating_good(self) -> None:
        """Test 'good' rating (0.65 <= NSE < 0.90)."""
        # obs mean=3, denom=10; diffs=[0.7,-0.7,0.5,-0.5,0.3]
        # sum_sq=0.49+0.49+0.25+0.25+0.09=1.57, NSE=1-1.57/10=0.843
        observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        simulated = np.array([1.7, 1.3, 3.5, 3.5, 5.3])
        metrics = ComparisonMetrics.compute(observed, simulated)
        assert metrics.rating() == "good"

    def test_rating_fair(self) -> None:
        """Test 'fair' rating (0.50 <= NSE < 0.65)."""
        # diffs=[1.0,-1.0,0.8,-0.8,0.5], sum_sq=1+1+0.64+0.64+0.25=3.53
        # NSE=1-3.53/10=0.647; but need <0.65. Use slightly bigger errors.
        # diffs=[1.1,-1.0,0.8,-0.8,0.5], sum_sq=1.21+1+0.64+0.64+0.25=3.74
        # NSE=1-3.74/10=0.626 => fair
        observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        simulated = np.array([2.1, 1.0, 3.8, 3.2, 5.5])
        metrics = ComparisonMetrics.compute(observed, simulated)
        assert metrics.rating() == "fair"

    def test_rating_poor(self) -> None:
        """Test 'poor' rating (NSE < 0.50)."""
        observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        simulated = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        metrics = ComparisonMetrics.compute(observed, simulated)
        assert metrics.rating() == "poor"


class TestComparisonMetricsNaN:
    """Tests for ComparisonMetrics.compute with NaN data."""

    def test_compute_with_nan_values(self) -> None:
        """Test compute filters out NaN values."""
        observed = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        simulated = np.array([1.1, 2.2, np.nan, 4.1, 5.1])
        metrics = ComparisonMetrics.compute(observed, simulated)
        # Two NaN positions filtered out, leaving 3 valid points
        assert metrics.n_points == 3


class TestTimeSeriesComparisonExtended:
    """Extended tests for TimeSeriesComparison."""

    def test_to_dict(self) -> None:
        """Test to_dict method returns correct structure."""
        times = np.arange(5, dtype=float)
        observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        simulated = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

        comparison = TimeSeriesComparison(times=times, observed=observed, simulated=simulated)
        d = comparison.to_dict()

        assert "n_points" in d
        assert d["n_points"] == 5
        assert "n_valid_points" in d
        assert d["n_valid_points"] == 5
        assert "metrics" in d
        assert isinstance(d["metrics"], dict)
        assert "rmse" in d["metrics"]

    def test_metrics_cached(self) -> None:
        """Test that metrics property caches result."""
        times = np.arange(5, dtype=float)
        observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        simulated = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

        comparison = TimeSeriesComparison(times=times, observed=observed, simulated=simulated)
        m1 = comparison.metrics
        m2 = comparison.metrics
        assert m1 is m2  # Same object (cached)


class TestSpatialComparisonExtended:
    """Extended tests for SpatialComparison."""

    def test_to_dict(self) -> None:
        """Test to_dict method returns correct structure."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 2.0])
        observed = np.array([10.0, 20.0, 30.0])
        simulated = np.array([11.0, 21.0, 31.0])

        comparison = SpatialComparison(x=x, y=y, observed=observed, simulated=simulated)
        d = comparison.to_dict()

        assert "n_points" in d
        assert d["n_points"] == 3
        assert "metrics" in d
        assert isinstance(d["metrics"], dict)

    def test_metrics_cached(self) -> None:
        """Test that metrics property caches result."""
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        observed = np.array([10.0, 20.0])
        simulated = np.array([11.0, 21.0])

        comparison = SpatialComparison(x=x, y=y, observed=observed, simulated=simulated)
        m1 = comparison.metrics
        m2 = comparison.metrics
        assert m1 is m2  # Same object (cached)

    def test_metrics_summary_contains_rating(self) -> None:
        """Test that summary includes rating."""
        observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        simulated = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        metrics = ComparisonMetrics.compute(observed, simulated)
        summary = metrics.summary()
        assert "Rating:" in summary
