"""Tests for IWFM2OBS calibration module."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from pyiwfm.calibration.iwfm2obs import (
    InterpolationConfig,
    _compute_model_dates,
    _replace_timestamps,
    deduplicate_smp,
    expand_obs_to_layers,
    interpolate_batch,
    interpolate_to_obs_times,
)
from pyiwfm.io.smp import SMPTimeSeries


def _make_ts(bore_id: str, dates: list[str], values: list[float]) -> SMPTimeSeries:
    """Helper to create an SMPTimeSeries."""
    return SMPTimeSeries(
        bore_id=bore_id,
        times=np.array(dates, dtype="datetime64[s]"),
        values=np.array(values, dtype=np.float64),
        excluded=np.zeros(len(dates), dtype=np.bool_),
    )


class TestInterpolateToObsTimes:
    """Tests for time interpolation."""

    def test_linear_interpolation(self) -> None:
        """Linear interpolation at midpoint gives mean."""
        obs = _make_ts("W1", ["2020-01-15"], [0.0])
        sim = _make_ts("W1", ["2020-01-01", "2020-02-01"], [100.0, 200.0])

        result = interpolate_to_obs_times(obs, sim)
        # Jan 15 is ~halfway between Jan 1 and Feb 1
        assert result.values[0] == pytest.approx(145.2, abs=1.0)

    def test_exact_match(self) -> None:
        """When obs time matches sim time, return exact value."""
        obs = _make_ts("W1", ["2020-01-01"], [0.0])
        sim = _make_ts("W1", ["2020-01-01", "2020-02-01"], [100.0, 200.0])

        result = interpolate_to_obs_times(obs, sim)
        assert result.values[0] == pytest.approx(100.0)

    def test_outside_range_gets_sentinel(self) -> None:
        """Observations far outside sim range get sentinel value."""
        obs = _make_ts("W1", ["2019-01-01"], [0.0])
        sim = _make_ts("W1", ["2020-01-01", "2020-02-01"], [100.0, 200.0])

        config = InterpolationConfig(
            max_extrapolation_time=timedelta(days=30),
            sentinel_value=-999.0,
        )
        result = interpolate_to_obs_times(obs, sim, config)
        assert result.values[0] == pytest.approx(-999.0)

    def test_within_extrapolation_range(self) -> None:
        """Observations within extrapolation window get interpolated value."""
        obs = _make_ts("W1", ["2020-02-15"], [0.0])
        sim = _make_ts("W1", ["2020-01-01", "2020-02-01"], [100.0, 200.0])

        config = InterpolationConfig(max_extrapolation_time=timedelta(days=30))
        result = interpolate_to_obs_times(obs, sim, config)
        # Within 30 days of end, so should get extrapolated (clamped) value
        assert result.values[0] != -999.0

    def test_nearest_interpolation(self) -> None:
        """Nearest-neighbor returns closest sim value."""
        obs = _make_ts("W1", ["2020-01-10"], [0.0])
        sim = _make_ts("W1", ["2020-01-01", "2020-02-01"], [100.0, 200.0])

        config = InterpolationConfig(interpolation_method="nearest")
        result = interpolate_to_obs_times(obs, sim, config)
        # Jan 10 is closer to Jan 1 than Feb 1
        assert result.values[0] == pytest.approx(100.0)

    def test_preserves_bore_id(self) -> None:
        """Result has same bore_id as observed."""
        obs = _make_ts("MY_WELL", ["2020-01-15"], [0.0])
        sim = _make_ts("MY_WELL", ["2020-01-01", "2020-02-01"], [100.0, 200.0])

        result = interpolate_to_obs_times(obs, sim)
        assert result.bore_id == "MY_WELL"

    def test_empty_simulated(self) -> None:
        """All NaN simulated values → sentinel output."""
        obs = _make_ts("W1", ["2020-01-15"], [0.0])
        sim = SMPTimeSeries(
            bore_id="W1",
            times=np.array(["2020-01-01", "2020-02-01"], dtype="datetime64[s]"),
            values=np.array([float("nan"), float("nan")]),
            excluded=np.zeros(2, dtype=np.bool_),
        )

        result = interpolate_to_obs_times(obs, sim)
        assert result.values[0] == pytest.approx(-999.0)


class TestInterpolateBatch:
    """Tests for batch interpolation."""

    def test_matching_bores_only(self) -> None:
        """Only bores in both obs and sim are processed."""
        obs = {
            "W1": _make_ts("W1", ["2020-01-15"], [0.0]),
            "W2": _make_ts("W2", ["2020-01-15"], [0.0]),
            "W3": _make_ts("W3", ["2020-01-15"], [0.0]),
        }
        sim = {
            "W1": _make_ts("W1", ["2020-01-01", "2020-02-01"], [100.0, 200.0]),
            "W3": _make_ts("W3", ["2020-01-01", "2020-02-01"], [300.0, 400.0]),
        }

        result = interpolate_batch(obs, sim)
        assert set(result.keys()) == {"W1", "W3"}

    def test_empty_inputs(self) -> None:
        """Empty inputs produce empty output."""
        result = interpolate_batch({}, {})
        assert len(result) == 0


class TestIWFM2OBSWorkflow:
    """Tests for the full iwfm2obs workflow."""

    def test_full_workflow(self, tmp_path: Path) -> None:
        """Write SMP files, run iwfm2obs, verify output."""
        from pyiwfm.calibration.iwfm2obs import iwfm2obs
        from pyiwfm.io.smp import SMPWriter

        obs_file = tmp_path / "obs.smp"
        sim_file = tmp_path / "sim.smp"
        out_file = tmp_path / "output.smp"

        obs_data = {
            "W1": _make_ts("W1", ["2020-01-15", "2020-02-15"], [50.0, 51.0]),
        }
        sim_data = {
            "W1": _make_ts("W1", ["2020-01-01", "2020-02-01", "2020-03-01"], [100.0, 110.0, 105.0]),
        }

        SMPWriter(obs_file).write(obs_data)
        SMPWriter(sim_file).write(sim_data)

        result = iwfm2obs(obs_file, sim_file, out_file)

        assert "W1" in result
        assert len(result["W1"].values) == 2
        assert out_file.exists()


class TestExpandObsToLayers:
    """Tests for expand_obs_to_layers()."""

    def test_no_expansion_when_all_suffixed(self) -> None:
        """IDs like WELL1%1, WELL1%2 are returned unchanged."""
        obs = {
            "WELL1%1": _make_ts("WELL1%1", ["2020-01-15"], [50.0]),
            "WELL1%2": _make_ts("WELL1%2", ["2020-01-15"], [51.0]),
        }
        result = expand_obs_to_layers(obs, 4)
        assert result is obs  # same object

    def test_expansion_base_to_layers(self) -> None:
        """WELL1, WELL2 expand to WELL1%1..%4, WELL2%1..%4 for 4 layers."""
        obs = {
            "WELL1": _make_ts("WELL1", ["2020-01-15"], [50.0]),
            "WELL2": _make_ts("WELL2", ["2020-01-15"], [60.0]),
        }
        result = expand_obs_to_layers(obs, 4)
        assert len(result) == 8
        for base in ("WELL1", "WELL2"):
            for k in range(1, 5):
                assert f"{base}%{k}" in result

    def test_mixed_ids(self) -> None:
        """WELL1 (no suffix) + WELL2%1 (has suffix) -> only WELL1 expanded."""
        obs = {
            "WELL1": _make_ts("WELL1", ["2020-01-15"], [50.0]),
            "WELL2%1": _make_ts("WELL2%1", ["2020-01-15"], [60.0]),
        }
        result = expand_obs_to_layers(obs, 2)
        assert "WELL2%1" in result  # kept as-is
        assert "WELL1%1" in result
        assert "WELL1%2" in result
        assert "WELL1" not in result  # base replaced by expanded

    def test_expansion_filters_by_simulated(self) -> None:
        """Only expand where simulated_ids contains the target."""
        obs = {
            "WELL1": _make_ts("WELL1", ["2020-01-15"], [50.0]),
        }
        sim_ids = {"WELL1%1", "WELL1%3"}  # only layers 1 and 3
        result = expand_obs_to_layers(obs, 4, simulated_ids=sim_ids)
        assert "WELL1%1" in result
        assert "WELL1%3" in result
        assert "WELL1%2" not in result
        assert "WELL1%4" not in result

    def test_shared_arrays(self) -> None:
        """Expanded entries share time/value arrays (no copy)."""
        obs = {
            "WELL1": _make_ts("WELL1", ["2020-01-15"], [50.0]),
        }
        result = expand_obs_to_layers(obs, 2)
        ts1 = result["WELL1%1"]
        ts2 = result["WELL1%2"]
        assert ts1.times is ts2.times
        assert ts1.values is ts2.values

    def test_zero_layers(self) -> None:
        """n_layers=0 -> no expansion."""
        obs = {
            "WELL1": _make_ts("WELL1", ["2020-01-15"], [50.0]),
        }
        result = expand_obs_to_layers(obs, 0)
        assert result is obs

    def test_empty_observed(self) -> None:
        """Empty dict returns unchanged."""
        result = expand_obs_to_layers({}, 4)
        assert result == {}


class TestDeduplicateSmp:
    """Tests for deduplicate_smp()."""

    def test_strip_suffixes(self, tmp_path: Path) -> None:
        """WELL1%1..%4 -> single WELL1 entry."""
        from pyiwfm.io.smp import SMPWriter

        data = {}
        for k in range(1, 5):
            bid = f"WELL1%{k}"
            data[bid] = _make_ts(bid, ["2020-01-15", "2020-02-15"], [50.0, 51.0])

        inp = tmp_path / "input.smp"
        out = tmp_path / "output.smp"
        SMPWriter(inp).write(data)

        orig, deduped = deduplicate_smp(inp, out)
        assert orig == 4
        assert deduped == 1
        assert out.exists()

    def test_already_deduplicated(self, tmp_path: Path) -> None:
        """No suffixes -> unchanged count."""
        from pyiwfm.io.smp import SMPWriter

        data = {
            "WELL1": _make_ts("WELL1", ["2020-01-15"], [50.0]),
            "WELL2": _make_ts("WELL2", ["2020-01-15"], [60.0]),
        }

        inp = tmp_path / "input.smp"
        out = tmp_path / "output.smp"
        SMPWriter(inp).write(data)

        orig, deduped = deduplicate_smp(inp, out)
        assert orig == 2
        assert deduped == 2


class TestComputeModelDates:
    """Tests for _compute_model_dates() — Fortran ComputeDate replication."""

    def test_monthly(self) -> None:
        """1MON from 09/30/1973 → end-of-month sequence."""
        dates = _compute_model_dates("09/30/1973", "1MON", 4)
        expected = [
            np.datetime64("1973-10-31", "s"),
            np.datetime64("1973-11-30", "s"),
            np.datetime64("1973-12-31", "s"),
            np.datetime64("1974-01-31", "s"),
        ]
        np.testing.assert_array_equal(dates, expected)

    def test_monthly_feb_leap(self) -> None:
        """1MON ending in Feb of a leap year → Feb 29."""
        dates = _compute_model_dates("12/31/1999", "1MON", 2)
        expected = [
            np.datetime64("2000-01-31", "s"),
            np.datetime64("2000-02-29", "s"),
        ]
        np.testing.assert_array_equal(dates, expected)

    def test_daily(self) -> None:
        """1DAY from 09/30/1973 → consecutive days."""
        dates = _compute_model_dates("09/30/1973", "1DAY", 3)
        expected = [
            np.datetime64("1973-10-01", "s"),
            np.datetime64("1973-10-02", "s"),
            np.datetime64("1973-10-03", "s"),
        ]
        np.testing.assert_array_equal(dates, expected)

    def test_weekly(self) -> None:
        """1WEEK from 09/30/1973 → 7-day increments."""
        dates = _compute_model_dates("09/30/1973", "1WEEK", 3)
        expected = [
            np.datetime64("1973-10-07", "s"),
            np.datetime64("1973-10-14", "s"),
            np.datetime64("1973-10-21", "s"),
        ]
        np.testing.assert_array_equal(dates, expected)

    def test_yearly(self) -> None:
        """1YEAR from 09/30/1973 → same month/day, next years."""
        dates = _compute_model_dates("09/30/1973", "1YEAR", 3)
        expected = [
            np.datetime64("1974-09-30", "s"),
            np.datetime64("1975-09-30", "s"),
            np.datetime64("1976-09-30", "s"),
        ]
        np.testing.assert_array_equal(dates, expected)

    def test_empty(self) -> None:
        """Zero timesteps → empty array."""
        dates = _compute_model_dates("01/01/2000", "1DAY", 0)
        assert len(dates) == 0


class TestReplaceTimestamps:
    """Tests for _replace_timestamps()."""

    def test_replaces_matching_length(self) -> None:
        """Timestamps are replaced when lengths match."""
        sim = {
            "W1": _make_ts("W1", ["2020-01-01", "2020-02-01"], [100.0, 200.0]),
        }
        computed = np.array(["2020-01-31", "2020-02-29"], dtype="datetime64[s]")
        _replace_timestamps(sim, computed)
        np.testing.assert_array_equal(sim["W1"].times, computed)

    def test_skips_mismatched_length(self) -> None:
        """Timestamps are NOT replaced when lengths differ."""
        original_times = np.array(["2020-01-01", "2020-02-01", "2020-03-01"], dtype="datetime64[s]")
        sim = {
            "W1": _make_ts("W1", ["2020-01-01", "2020-02-01", "2020-03-01"], [100.0, 200.0, 300.0]),
        }
        computed = np.array(["2020-01-31", "2020-02-29"], dtype="datetime64[s]")
        _replace_timestamps(sim, computed)
        np.testing.assert_array_equal(sim["W1"].times, original_times)

    def test_integration_with_interpolation(self) -> None:
        """Full pipeline: computed dates align obs on month-end → exact interp."""
        # Simulated data at month-end dates (as Fortran would compute)
        sim_dates = ["2020-01-31", "2020-02-29", "2020-03-31"]
        sim = {
            "W1": _make_ts("W1", sim_dates, [100.0, 110.0, 105.0]),
        }
        # Replace with ComputeDate-style monthly dates
        computed = _compute_model_dates("12/31/2019", "1MON", 3)
        _replace_timestamps(sim, computed)

        # Obs on the exact month-end → should get exact sim values
        obs = {
            "W1": _make_ts("W1", ["2020-01-31", "2020-02-29"], [0.0, 0.0]),
        }
        result = interpolate_batch(obs, sim)
        assert result["W1"].values[0] == pytest.approx(100.0)
        assert result["W1"].values[1] == pytest.approx(110.0)
