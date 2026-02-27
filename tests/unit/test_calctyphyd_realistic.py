"""Realistic tests for CalcTypHyd: algorithm fidelity and C2VSimFG-like configs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyiwfm.calibration.calctyphyd import (
    CalcTypHydConfig,
    SeasonalPeriod,
    compute_seasonal_averages,
    compute_typical_hydrographs,
    read_cluster_weights,
)
from pyiwfm.io.smp import SMPReader, SMPTimeSeries

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "calibration"


def _make_ts(
    bore_id: str,
    dates: list[str],
    values: list[float],
    excluded: list[bool] | None = None,
) -> SMPTimeSeries:
    n = len(dates)
    return SMPTimeSeries(
        bore_id=bore_id,
        times=np.array(dates, dtype="datetime64[s]"),
        values=np.array(values, dtype=np.float64),
        excluded=np.array(excluded if excluded else [False] * n, dtype=np.bool_),
    )


def _make_annual_ts(
    bore_id: str,
    base_value: float = 100.0,
    amplitude: float = 5.0,
    n_years: int = 1,
    start_year: int = 2020,
) -> SMPTimeSeries:
    """Create monthly time series with seasonal sinusoidal pattern."""
    dates: list[str] = []
    values: list[float] = []
    for y in range(start_year, start_year + n_years):
        for m in range(1, 13):
            dates.append(f"{y}-{m:02d}-15")
            values.append(base_value + amplitude * np.sin(2 * np.pi * m / 12))
    return _make_ts(bore_id, dates, values)


def _make_12_month_seasons() -> list[SeasonalPeriod]:
    """Create 12 individual monthly seasonal periods."""
    names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    return [
        SeasonalPeriod(name=names[i], months=[i + 1], representative_date=f"{i + 1:02d}/15")
        for i in range(12)
    ]


# ---------------------------------------------------------------------------
# TestMonthlyPeriods — 12 monthly seasonal periods (C2VSimFG-like)
# ---------------------------------------------------------------------------


class TestMonthlyPeriods:
    """Tests with 12 monthly seasonal periods matching C2VSimFG config."""

    def test_12_monthly_averages(self) -> None:
        """12 monthly periods produce 12 seasonal averages."""
        config = CalcTypHydConfig(seasonal_periods=_make_12_month_seasons())
        wl = {"W1": _make_annual_ts("W1", 100.0, 5.0, n_years=1)}
        avgs = compute_seasonal_averages(wl, config)
        assert len(avgs["W1"]) == 12
        assert not np.any(np.isnan(avgs["W1"]))

    def test_multi_year_monthly_averaging(self) -> None:
        """Multi-year data averaged per month across years."""
        config = CalcTypHydConfig(seasonal_periods=_make_12_month_seasons())
        wl = {"W1": _make_annual_ts("W1", 100.0, 5.0, n_years=3)}
        avgs = compute_seasonal_averages(wl, config)
        # Each month has 3 values → average should be stable
        assert not np.any(np.isnan(avgs["W1"]))

    def test_missing_months_give_nan(self) -> None:
        """Wells with data in only some months get NaN for missing months."""
        config = CalcTypHydConfig(seasonal_periods=_make_12_month_seasons())
        # Only Jan and Feb data
        ts = _make_ts("W1", ["2020-01-15", "2020-02-15"], [100.0, 110.0])
        avgs = compute_seasonal_averages({"W1": ts}, config)
        assert not np.isnan(avgs["W1"][0])  # Jan
        assert not np.isnan(avgs["W1"][1])  # Feb
        assert np.isnan(avgs["W1"][5])  # Jun
        assert np.isnan(avgs["W1"][11])  # Dec

    def test_single_record_per_month(self) -> None:
        """Single record per month suffices with min_records=1."""
        config = CalcTypHydConfig(
            seasonal_periods=_make_12_month_seasons(),
            min_records_per_season=1,
        )
        wl = {"W1": _make_annual_ts("W1", 50.0, 0.0, n_years=1)}
        avgs = compute_seasonal_averages(wl, config)
        np.testing.assert_allclose(avgs["W1"], 50.0, atol=0.001)


# ---------------------------------------------------------------------------
# TestMultiYearAveraging
# ---------------------------------------------------------------------------


class TestMultiYearAveraging:
    """Tests for multi-year seasonal averaging."""

    def test_5_year_quarterly_average(self) -> None:
        """5 years of data → stable quarterly averages."""
        wl = {"W1": _make_annual_ts("W1", 100.0, 10.0, n_years=5)}
        avgs = compute_seasonal_averages(wl)
        assert len(avgs["W1"]) == 4
        assert not np.any(np.isnan(avgs["W1"]))

    def test_nan_values_ignored(self) -> None:
        """NaN values in time series are excluded from seasonal averages."""
        dates = [f"2020-{m:02d}-15" for m in range(1, 13)]
        values = [100.0] * 12
        values[0] = float("nan")  # Jan → NaN
        ts = _make_ts("W1", dates, values)
        avgs = compute_seasonal_averages({"W1": ts})
        # Winter (Dec, Jan, Feb): only Feb contributes (Dec has no data, Jan=NaN)
        assert not np.isnan(avgs["W1"][0])  # Winter still has Feb

    def test_excluded_records_filtered_via_valid_mask(self) -> None:
        """Excluded records are not included in seasonal averages."""
        dates = [f"2020-{m:02d}-15" for m in range(1, 13)]
        values = [100.0] * 12
        excluded = [False] * 12
        # Mark all winter months as excluded
        excluded[0] = True  # Jan
        excluded[1] = True  # Feb
        ts = _make_ts("W1", dates, values, excluded=excluded)
        config = CalcTypHydConfig(min_records_per_season=2)
        avgs = compute_seasonal_averages({"W1": ts}, config)
        # Winter: Dec not in data, Jan excluded, Feb excluded → only need 2, have 0 valid
        # Actually Dec is month 12 → not in this single year starting Jan
        # So winter has 0 valid records with min=2 → NaN
        assert np.isnan(avgs["W1"][0])

    def test_single_record_threshold(self) -> None:
        """min_records_per_season=5 filters out seasons with fewer records."""
        wl = {"W1": _make_annual_ts("W1", 100.0, 5.0, n_years=1)}
        config = CalcTypHydConfig(min_records_per_season=5)
        avgs = compute_seasonal_averages(wl, config)
        # Each season has only 3 months (1 year) → all NaN with min=5
        assert np.all(np.isnan(avgs["W1"]))


# ---------------------------------------------------------------------------
# TestDeMeaningCorrectness
# ---------------------------------------------------------------------------


class TestDeMeaningCorrectness:
    """Verify the de-meaning formula: SUM(W*(WL-mean))/SUM(W_active)."""

    def test_exact_formula_single_well(self) -> None:
        """Single well: typical hyd = seasonal_avg - well_mean (weight=1)."""
        wl = {"W1": _make_annual_ts("W1", 100.0, 10.0)}
        weights = {"W1": np.array([1.0])}
        result = compute_typical_hydrographs(wl, weights)
        avgs = compute_seasonal_averages(wl)
        well_mean = float(np.nanmean(avgs["W1"]))
        expected = avgs["W1"] - well_mean
        np.testing.assert_allclose(result.hydrographs[0].values, expected, atol=1e-10)

    def test_partial_nan_handling(self) -> None:
        """Wells with partial NaN seasonal averages contribute only where valid."""
        # W1 has all seasons, W2 only summer and fall
        wl = {
            "W1": _make_annual_ts("W1", 100.0, 5.0),
            "W2": _make_ts("W2", ["2020-07-15", "2020-10-15"], [80.0, 90.0]),
        }
        weights = {"W1": np.array([0.5]), "W2": np.array([0.5])}
        result = compute_typical_hydrographs(wl, weights)
        values = result.hydrographs[0].values
        # All 4 seasons should have values (W1 contributes to all)
        assert not np.all(np.isnan(values))

    def test_demeaned_sum_near_zero(self) -> None:
        """For a single well, de-meaned seasonal values sum to ~0."""
        wl = {"W1": _make_annual_ts("W1", 100.0, 10.0)}
        weights = {"W1": np.array([1.0])}
        result = compute_typical_hydrographs(wl, weights)
        values = result.hydrographs[0].values
        valid = ~np.isnan(values)
        assert abs(float(np.sum(values[valid]))) < 1e-8

    def test_mean_from_valid_seasons_only(self) -> None:
        """Well mean computed from valid (non-NaN) seasonal averages only."""
        # Only summer (Jun-Aug) and fall (Sep-Nov) data
        ts = _make_ts("W1", ["2020-07-15", "2020-10-15"], [80.0, 120.0])
        wl = {"W1": ts}
        weights = {"W1": np.array([1.0])}
        result = compute_typical_hydrographs(wl, weights)
        assert result.well_means["W1"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# TestWeightNormalization
# ---------------------------------------------------------------------------


class TestWeightNormalization:
    """Test per-season denominator with missing data."""

    def test_per_season_denom_with_missing_data(self) -> None:
        """Denominator excludes wells with NaN for that season."""
        wl = {
            "W1": _make_annual_ts("W1", 100.0, 5.0),
            "W2": _make_ts("W2", ["2020-07-15"], [200.0]),  # Only summer
        }
        weights = {"W1": np.array([0.6]), "W2": np.array([0.4])}
        result = compute_typical_hydrographs(wl, weights)
        values = result.hydrographs[0].values
        # All 4 seasons should have values (at least W1 always contributes)
        assert not np.any(np.isnan(values))

    def test_zero_weight_exclusion(self) -> None:
        """Wells with zero weight for a cluster don't contribute."""
        wl = {
            "W1": _make_annual_ts("W1", 100.0, 5.0),
            "W2": _make_annual_ts("W2", 200.0, 5.0),
        }
        weights = {"W1": np.array([1.0, 0.0]), "W2": np.array([0.0, 1.0])}
        result = compute_typical_hydrographs(wl, weights)
        # Cluster 0 should only have W1
        assert "W1" in result.hydrographs[0].contributing_wells
        assert "W2" not in result.hydrographs[0].contributing_wells
        # Cluster 1 should only have W2
        assert "W2" in result.hydrographs[1].contributing_wells
        assert "W1" not in result.hydrographs[1].contributing_wells

    def test_unequal_coverage(self) -> None:
        """Wells covering different seasons produce correct per-season denoms."""
        # W1: full year, W2: only winter/spring
        wl = {
            "W1": _make_annual_ts("W1", 100.0, 5.0),
            "W2": _make_ts(
                "W2",
                ["2020-01-15", "2020-02-15", "2020-03-15", "2020-04-15"],
                [90.0, 92.0, 94.0, 96.0],
            ),
        }
        weights = {"W1": np.array([0.5]), "W2": np.array([0.5])}
        result = compute_typical_hydrographs(wl, weights)
        assert not np.any(np.isnan(result.hydrographs[0].values))

    def test_all_nan_season(self) -> None:
        """Season with no valid data from any well gives NaN."""
        # Only summer data from one well, with zero weight
        wl = {"W1": _make_ts("W1", ["2020-07-15"], [100.0])}
        weights = {"W1": np.array([0.0])}  # Zero weight
        result = compute_typical_hydrographs(wl, weights)
        # All seasons should be NaN since the only well has zero weight
        assert np.all(np.isnan(result.hydrographs[0].values))


# ---------------------------------------------------------------------------
# TestClusterWeightsReading
# ---------------------------------------------------------------------------


class TestClusterWeightsReading:
    """Tests for reading cluster weight files."""

    def test_header_line_format(self, tmp_path: Path) -> None:
        """Header comment lines are skipped."""
        f = tmp_path / "w.txt"
        f.write_text("# Header\n# Cols\nW1 0.8 0.2\nW2 0.2 0.8\n")
        weights = read_cluster_weights(f)
        assert len(weights) == 2

    def test_10x50_matrix(self, tmp_path: Path) -> None:
        """10 wells, 50 clusters weights file."""
        lines = ["# 10 wells, 50 clusters\n"]
        for i in range(10):
            w = np.random.default_rng(i).dirichlet(np.ones(50))
            vals = "  ".join(f"{v:.6f}" for v in w)
            lines.append(f"W{i:02d}  {vals}\n")
        f = tmp_path / "w.txt"
        f.write_text("".join(lines))
        weights = read_cluster_weights(f)
        assert len(weights) == 10
        assert len(weights["W00"]) == 50
        np.testing.assert_allclose(sum(weights["W00"]), 1.0, atol=1e-5)

    def test_tab_separated(self, tmp_path: Path) -> None:
        """Tab-separated weights file parses correctly."""
        f = tmp_path / "w.txt"
        f.write_text("W1\t0.9\t0.1\nW2\t0.3\t0.7\n")
        weights = read_cluster_weights(f)
        assert weights["W1"][0] == pytest.approx(0.9)
        assert weights["W2"][1] == pytest.approx(0.7)

    def test_all_zero_cluster(self, tmp_path: Path) -> None:
        """Cluster with all-zero weights produces NaN typical hydrograph."""
        f = tmp_path / "w.txt"
        f.write_text("W1 0.0 1.0\nW2 0.0 1.0\n")
        weights = read_cluster_weights(f)
        wl = {
            "W1": _make_annual_ts("W1", 100.0),
            "W2": _make_annual_ts("W2", 200.0),
        }
        result = compute_typical_hydrographs(wl, weights)
        # Cluster 0 has zero weights → all NaN
        assert np.all(np.isnan(result.hydrographs[0].values))


# ---------------------------------------------------------------------------
# TestRepresentativeDates
# ---------------------------------------------------------------------------


class TestRepresentativeDates:
    """Test representative date assignment."""

    def test_default_dates(self) -> None:
        """Default seasons have representative dates at mid-season."""
        wl = {"W1": _make_annual_ts("W1")}
        weights = {"W1": np.array([1.0])}
        result = compute_typical_hydrographs(wl, weights)
        hyd = result.hydrographs[0]
        # Default: Winter=01/15, Spring=04/15, Summer=07/15, Fall=10/15
        assert str(hyd.times[0]) == "2000-01-15"
        assert str(hyd.times[1]) == "2000-04-15"
        assert str(hyd.times[2]) == "2000-07-15"
        assert str(hyd.times[3]) == "2000-10-15"

    def test_custom_monthly_dates(self) -> None:
        """Custom monthly periods have correct representative dates."""
        config = CalcTypHydConfig(seasonal_periods=_make_12_month_seasons())
        wl = {"W1": _make_annual_ts("W1")}
        weights = {"W1": np.array([1.0])}
        result = compute_typical_hydrographs(wl, weights, config)
        hyd = result.hydrographs[0]
        assert len(hyd.times) == 12
        assert str(hyd.times[0]) == "2000-01-15"
        assert str(hyd.times[5]) == "2000-06-15"


# ---------------------------------------------------------------------------
# TestMultipleClusters
# ---------------------------------------------------------------------------


class TestMultipleClusters:
    """Tests with varying cluster configurations."""

    def test_3_clusters_2_wells(self) -> None:
        """3 clusters with 2 wells produce 3 hydrographs."""
        wl = {
            "W1": _make_annual_ts("W1", 100.0, 10.0),
            "W2": _make_annual_ts("W2", 200.0, 10.0),
        }
        weights = {
            "W1": np.array([0.8, 0.1, 0.1]),
            "W2": np.array([0.1, 0.8, 0.1]),
        }
        result = compute_typical_hydrographs(wl, weights)
        assert len(result.hydrographs) == 3

    def test_single_well_cluster(self) -> None:
        """Cluster with only one well has that well's de-meaned pattern."""
        wl = {
            "W1": _make_annual_ts("W1", 100.0, 10.0),
            "W2": _make_annual_ts("W2", 200.0, 20.0),
        }
        weights = {
            "W1": np.array([1.0, 0.0]),
            "W2": np.array([0.0, 1.0]),
        }
        result = compute_typical_hydrographs(wl, weights)
        # Cluster 0 = W1's de-meaned pattern
        avgs = compute_seasonal_averages(wl)
        w1_mean = float(np.nanmean(avgs["W1"]))
        expected = avgs["W1"] - w1_mean
        np.testing.assert_allclose(result.hydrographs[0].values, expected, atol=1e-10)

    def test_10_clusters_20_wells(self) -> None:
        """10 clusters with 20 wells produces valid results."""
        rng = np.random.default_rng(42)
        wl: dict[str, SMPTimeSeries] = {}
        weights: dict[str, np.ndarray] = {}
        for i in range(20):
            bid = f"W{i:02d}"
            wl[bid] = _make_annual_ts(bid, base_value=50.0 + i * 5, n_years=2)
            w = rng.dirichlet(np.ones(10))
            weights[bid] = w
        result = compute_typical_hydrographs(wl, weights)
        assert len(result.hydrographs) == 10
        assert len(result.well_means) == 20


# ---------------------------------------------------------------------------
# TestFixtureDataMatch — regression tests against frozen expected output
# ---------------------------------------------------------------------------


class TestFixtureDataMatch:
    """Load fixture files and verify pyiwfm output matches expected."""

    @pytest.fixture()
    def fixture_dir(self) -> Path:
        d = _FIXTURES
        if not d.exists():
            pytest.skip("calibration fixtures not found")
        return d

    def test_calctyphyd_cluster0_matches_expected(self, fixture_dir: Path) -> None:
        """Cluster 0 typical hydrograph matches frozen expected output."""
        wl_data = SMPReader(fixture_dir / "water_levels.smp").read()
        weights = read_cluster_weights(fixture_dir / "cluster_weights.txt")
        result = compute_typical_hydrographs(wl_data, weights)

        expected_file = fixture_dir / "expected_typhyd_cls0.txt"
        expected_values = _load_expected_typhyd(expected_file)

        np.testing.assert_allclose(
            result.hydrographs[0].values,
            expected_values,
            atol=1e-4,
        )

    def test_calctyphyd_cluster1_matches_expected(self, fixture_dir: Path) -> None:
        """Cluster 1 typical hydrograph matches frozen expected output."""
        wl_data = SMPReader(fixture_dir / "water_levels.smp").read()
        weights = read_cluster_weights(fixture_dir / "cluster_weights.txt")
        result = compute_typical_hydrographs(wl_data, weights)

        expected_file = fixture_dir / "expected_typhyd_cls1.txt"
        expected_values = _load_expected_typhyd(expected_file)

        np.testing.assert_allclose(
            result.hydrographs[1].values,
            expected_values,
            atol=1e-4,
        )


def _load_expected_typhyd(filepath: Path) -> np.ndarray:
    """Load expected CalcTypHyd values from text file."""
    values: list[float] = []
    with open(filepath) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            val_str = parts[-1]
            if val_str == "NaN":
                values.append(float("nan"))
            else:
                values.append(float(val_str))
    return np.array(values, dtype=np.float64)


# ---------------------------------------------------------------------------
# TestCalcTypHydBenchmark — performance benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
class TestCalcTypHydBenchmark:
    """Benchmark CalcTypHyd performance."""

    @staticmethod
    def _make_well_data(n_wells: int, n_years: int) -> dict[str, SMPTimeSeries]:
        rng = np.random.default_rng(42)
        wl: dict[str, SMPTimeSeries] = {}
        for i in range(n_wells):
            bid = f"W{i:03d}"
            dates: list[str] = []
            vals: list[float] = []
            for y in range(2020, 2020 + n_years):
                for m in range(1, 13):
                    dates.append(f"{y}-{m:02d}-15")
                    vals.append(100.0 + rng.standard_normal() * 10)
            wl[bid] = _make_ts(bid, dates, vals)
        return wl

    def test_benchmark_seasonal_averages(self, benchmark: object) -> None:
        """Benchmark seasonal averages: 50 wells, 5 years."""
        wl = self._make_well_data(50, 5)
        benchmark(compute_seasonal_averages, wl)  # type: ignore[operator]

    def test_benchmark_typical_hydrographs(self, benchmark: object) -> None:
        """Benchmark typical hydrographs: 50 wells, 10 clusters."""
        wl = self._make_well_data(50, 5)
        rng = np.random.default_rng(42)
        weights = {bid: rng.dirichlet(np.ones(10)) for bid in wl}
        benchmark(compute_typical_hydrographs, wl, weights)  # type: ignore[operator]

    def test_benchmark_read_cluster_weights(self, benchmark: object, tmp_path: Path) -> None:
        """Benchmark reading 500-well weights file."""
        rng = np.random.default_rng(42)
        lines = []
        for i in range(500):
            w = rng.dirichlet(np.ones(10))
            vals = "  ".join(f"{v:.6f}" for v in w)
            lines.append(f"W{i:03d}  {vals}\n")
        f = tmp_path / "weights.txt"
        f.write_text("".join(lines))
        benchmark(read_cluster_weights, f)  # type: ignore[operator]
