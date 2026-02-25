"""Tests for IWFM2OBS calibration module."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from pyiwfm.calibration.iwfm2obs import (
    InterpolationConfig,
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
