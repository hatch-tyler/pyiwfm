"""Tests for CalcTypHyd typical hydrograph computation."""

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
from pyiwfm.io.smp import SMPTimeSeries


def _make_annual_ts(bore_id: str, base_value: float = 100.0) -> SMPTimeSeries:
    """Create a year of monthly water level data."""
    dates = [f"2020-{m:02d}-15" for m in range(1, 13)]
    # Simple seasonal pattern
    values = [base_value + 5.0 * np.sin(2 * np.pi * m / 12) for m in range(12)]
    return SMPTimeSeries(
        bore_id=bore_id,
        times=np.array(dates, dtype="datetime64[s]"),
        values=np.array(values, dtype=np.float64),
        excluded=np.zeros(12, dtype=np.bool_),
    )


class TestReadClusterWeights:
    """Tests for cluster weights file reading."""

    def test_read_weights(self, tmp_path: Path) -> None:
        weights_file = tmp_path / "weights.txt"
        weights_file.write_text(
            "# Header comment\nW1 0.8 0.1 0.1\nW2 0.2 0.7 0.1\nW3 0.1 0.2 0.7\n"
        )

        weights = read_cluster_weights(weights_file)
        assert len(weights) == 3
        assert weights["W1"][0] == pytest.approx(0.8)
        assert weights["W2"][1] == pytest.approx(0.7)
        np.testing.assert_allclose(weights["W3"], [0.1, 0.2, 0.7])

    def test_skip_comments_and_blanks(self, tmp_path: Path) -> None:
        weights_file = tmp_path / "weights.txt"
        weights_file.write_text("# Comment\n\nW1 0.5 0.5\n# Another comment\n")

        weights = read_cluster_weights(weights_file)
        assert len(weights) == 1
        assert "W1" in weights


class TestSeasonalAverages:
    """Tests for seasonal average computation."""

    def test_default_4_seasons(self) -> None:
        water_levels = {"W1": _make_annual_ts("W1")}
        avgs = compute_seasonal_averages(water_levels)

        assert "W1" in avgs
        assert len(avgs["W1"]) == 4
        # All seasons should have valid averages
        assert not np.any(np.isnan(avgs["W1"]))

    def test_custom_seasons(self) -> None:
        config = CalcTypHydConfig(
            seasonal_periods=[
                SeasonalPeriod("Wet", [11, 12, 1, 2, 3, 4], "01/15"),
                SeasonalPeriod("Dry", [5, 6, 7, 8, 9, 10], "07/15"),
            ]
        )
        water_levels = {"W1": _make_annual_ts("W1")}
        avgs = compute_seasonal_averages(water_levels, config)

        assert len(avgs["W1"]) == 2

    def test_insufficient_data(self) -> None:
        """Well with too few records gets NaN for some seasons."""
        ts = SMPTimeSeries(
            bore_id="W1",
            times=np.array(["2020-01-15"], dtype="datetime64[s]"),
            values=np.array([100.0]),
            excluded=np.zeros(1, dtype=np.bool_),
        )
        config = CalcTypHydConfig(min_records_per_season=2)
        avgs = compute_seasonal_averages({"W1": ts}, config)

        # Only 1 record in winter, min is 2 → all NaN
        assert np.all(np.isnan(avgs["W1"]))


class TestComputeTypicalHydrographs:
    """Tests for typical hydrograph computation."""

    def test_basic_computation(self, tmp_path: Path) -> None:
        """Compute typical hydrographs with simple weights."""
        water_levels = {
            "W1": _make_annual_ts("W1", 100.0),
            "W2": _make_annual_ts("W2", 200.0),
        }
        weights = {
            "W1": np.array([0.8, 0.2]),
            "W2": np.array([0.2, 0.8]),
        }

        result = compute_typical_hydrographs(water_levels, weights)

        assert len(result.hydrographs) == 2
        assert result.hydrographs[0].cluster_id == 0
        assert result.hydrographs[1].cluster_id == 1
        assert len(result.well_means) == 2

    def test_contributing_wells(self) -> None:
        """Wells with non-zero weights are listed as contributing."""
        water_levels = {
            "W1": _make_annual_ts("W1"),
            "W2": _make_annual_ts("W2"),
        }
        weights = {
            "W1": np.array([1.0, 0.0]),
            "W2": np.array([0.0, 1.0]),
        }

        result = compute_typical_hydrographs(water_levels, weights)

        assert "W1" in result.hydrographs[0].contributing_wells
        assert "W2" not in result.hydrographs[0].contributing_wells
        assert "W2" in result.hydrographs[1].contributing_wells

    def test_demeaned_values(self) -> None:
        """Typical hydrograph values should be de-meaned (centered around 0)."""
        water_levels = {"W1": _make_annual_ts("W1", 100.0)}
        weights = {"W1": np.array([1.0])}

        result = compute_typical_hydrographs(water_levels, weights)

        values = result.hydrographs[0].values
        valid = ~np.isnan(values)
        if np.any(valid):
            # De-meaned values should be centered near 0
            assert abs(float(np.nanmean(values))) < 5.0

    def test_empty_weights(self) -> None:
        """Empty weights produce empty result."""
        water_levels = {"W1": _make_annual_ts("W1")}
        result = compute_typical_hydrographs(water_levels, {})
        assert len(result.hydrographs) == 0
