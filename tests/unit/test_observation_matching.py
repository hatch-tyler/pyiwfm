"""Tests for pyiwfm.calibration.observation_matching."""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pytest

from pyiwfm.calibration.observation_matching import (
    ObservationMatcher,
)


class TestFitStatistics:
    """Test FitStatistics dataclass."""

    def test_perfect_fit(self) -> None:
        sim = np.array([1.0, 2.0, 3.0])
        obs = np.array([1.0, 2.0, 3.0])
        stats = ObservationMatcher.compute_statistics(sim, obs)
        assert stats.rmse == pytest.approx(0.0)
        assert stats.mae == pytest.approx(0.0)
        assert stats.bias == pytest.approx(0.0)
        assert stats.nse == pytest.approx(1.0)
        assert stats.r_squared == pytest.approx(1.0)
        assert stats.n_matched == 3

    def test_constant_offset(self) -> None:
        sim = np.array([1.0, 2.0, 3.0])
        obs = np.array([2.0, 3.0, 4.0])
        stats = ObservationMatcher.compute_statistics(sim, obs)
        assert stats.rmse == pytest.approx(1.0)
        assert stats.bias == pytest.approx(1.0)
        assert stats.r_squared == pytest.approx(1.0)
        # NSE = 1 - SS_res/SS_tot = 1 - 3/2 = -0.5
        assert stats.nse == pytest.approx(-0.5)

    def test_empty_arrays(self) -> None:
        stats = ObservationMatcher.compute_statistics(
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )
        assert stats.n_matched == 0
        assert math.isnan(stats.rmse)
        assert math.isnan(stats.nse)

    def test_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            ObservationMatcher.compute_statistics(
                np.array([1.0, 2.0]),
                np.array([1.0]),
            )

    def test_nse_negative(self) -> None:
        """NSE < 0 when model is worse than mean."""
        sim = np.array([10.0, 20.0, 30.0])
        obs = np.array([1.0, 2.0, 3.0])
        stats = ObservationMatcher.compute_statistics(sim, obs)
        assert stats.nse < 0.0

    def test_single_value(self) -> None:
        """Single observation: r_squared is NaN (can't compute)."""
        stats = ObservationMatcher.compute_statistics(
            np.array([5.0]),
            np.array([5.0]),
        )
        assert stats.n_matched == 1
        assert stats.rmse == pytest.approx(0.0)
        assert math.isnan(stats.r_squared)

    def test_constant_obs(self) -> None:
        """All obs same value: ss_tot=0, NSE=NaN."""
        sim = np.array([1.0, 2.0, 3.0])
        obs = np.array([5.0, 5.0, 5.0])
        stats = ObservationMatcher.compute_statistics(sim, obs)
        assert math.isnan(stats.nse)


class TestMatchByDate:
    """Test ObservationMatcher.match_by_date."""

    def test_exact_date_match(self) -> None:
        matcher = ObservationMatcher()
        sim_times = np.array(
            [np.datetime64("2020-01-31"), np.datetime64("2020-02-29")],
            dtype="datetime64[D]",
        )
        sim_values = np.array([100.0, 101.0])
        obs_times = np.array(
            [np.datetime64("2020-01-31"), np.datetime64("2020-02-29")],
            dtype="datetime64[D]",
        )
        obs_values = np.array([99.5, 101.5])

        result = matcher.match_by_date(sim_times, sim_values, obs_times, obs_values)
        assert result.stats.n_matched == 2
        assert result.sim_matched[0] == pytest.approx(100.0)
        assert result.obs_matched[0] == pytest.approx(99.5)

    def test_partial_match(self) -> None:
        """Obs has dates not in sim."""
        matcher = ObservationMatcher()
        sim_times = np.array([np.datetime64("2020-01-31")], dtype="datetime64[D]")
        sim_values = np.array([100.0])
        obs_times = np.array(
            [np.datetime64("2020-01-31"), np.datetime64("2020-03-15")],
            dtype="datetime64[D]",
        )
        obs_values = np.array([99.0, 102.0])

        result = matcher.match_by_date(sim_times, sim_values, obs_times, obs_values)
        assert result.stats.n_matched == 1

    def test_no_match(self) -> None:
        matcher = ObservationMatcher()
        sim_times = np.array([np.datetime64("2020-01-31")], dtype="datetime64[D]")
        sim_values = np.array([100.0])
        obs_times = np.array([np.datetime64("2020-06-15")], dtype="datetime64[D]")
        obs_values = np.array([50.0])

        result = matcher.match_by_date(sim_times, sim_values, obs_times, obs_values)
        assert result.stats.n_matched == 0

    def test_datetime_objects(self) -> None:
        """Test with Python datetime objects."""
        matcher = ObservationMatcher()
        sim_times = np.array([datetime(2020, 1, 31), datetime(2020, 2, 29)])
        sim_values = np.array([100.0, 101.0])
        obs_times = np.array([datetime(2020, 1, 31)])
        obs_values = np.array([99.0])

        result = matcher.match_by_date(sim_times, sim_values, obs_times, obs_values)
        assert result.stats.n_matched == 1

    def test_ignores_time_of_day(self) -> None:
        """Match on date only, not time."""
        matcher = ObservationMatcher()
        sim_times = np.array(
            [np.datetime64("2020-01-31T00:00:00")],
            dtype="datetime64[s]",
        )
        sim_values = np.array([100.0])
        obs_times = np.array(
            [np.datetime64("2020-01-31T12:00:00")],
            dtype="datetime64[s]",
        )
        obs_values = np.array([99.0])

        result = matcher.match_by_date(sim_times, sim_values, obs_times, obs_values)
        assert result.stats.n_matched == 1
