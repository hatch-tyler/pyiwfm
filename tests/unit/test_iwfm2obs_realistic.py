"""Realistic tests for IWFM2OBS: interpolation fidelity, edge cases, SMP I/O."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from pyiwfm.calibration.iwfm2obs import (
    InterpolationConfig,
    interpolate_batch,
    interpolate_to_obs_times,
    iwfm2obs,
)
from pyiwfm.io.smp import SMPReader, SMPTimeSeries, SMPWriter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "calibration"

_C2VSIM_BORE_IDS = [
    "S_380313N1219426W001%1",
    "S_375204N1214521W001%1",
    "S_381045N1220102W001%1",
    "S_374830N1213900W001%1",
    "S_380600N1218000W001%1",
]


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


def _make_monthly_ts(
    bore_id: str,
    start_year: int,
    n_months: int,
    func: object,
) -> SMPTimeSeries:
    """Create monthly time series with values from *func(i)*."""
    dates: list[str] = []
    values: list[float] = []
    for i in range(n_months):
        y = start_year + (i // 12)
        m = (i % 12) + 1
        dates.append(f"{y}-{m:02d}-01")
        values.append(float(func(i)))  # type: ignore[operator]
    return _make_ts(bore_id, dates, values)


def _days_between(d1: str, d2: str) -> float:
    """Seconds between two ISO dates."""
    t1 = np.datetime64(d1, "s").astype(np.float64)
    t2 = np.datetime64(d2, "s").astype(np.float64)
    return t2 - t1


# ---------------------------------------------------------------------------
# TestFortranFidelity — extrapolation boundary precision
# ---------------------------------------------------------------------------


class TestFortranFidelity:
    """Verify interpolation matches Fortran IWFM2OBS behaviour."""

    def test_exact_boundary_match(self) -> None:
        """Obs at exact sim time returns exact sim value."""
        sim = _make_ts("W1", ["2020-01-01", "2020-02-01", "2020-03-01"], [10.0, 20.0, 30.0])
        obs = _make_ts("W1", ["2020-02-01"], [0.0])
        result = interpolate_to_obs_times(obs, sim)
        assert result.values[0] == pytest.approx(20.0)

    def test_one_second_inside_range(self) -> None:
        """Obs 1 second inside sim range is interpolated, not sentinel."""
        sim = _make_ts("W1", ["2020-01-01T00:00:00", "2020-12-31T23:59:59"], [0.0, 100.0])
        obs = _make_ts("W1", ["2020-01-01T00:00:01"], [0.0])
        result = interpolate_to_obs_times(obs, sim)
        assert result.values[0] != -999.0
        assert result.values[0] > 0.0

    def test_one_second_past_extrap_gets_sentinel(self) -> None:
        """Obs 1 second past extrapolation limit gets sentinel."""
        sim = _make_ts("W1", ["2020-06-01", "2020-07-01"], [50.0, 60.0])
        extrap = timedelta(days=30)
        config = InterpolationConfig(max_extrapolation_time=extrap, sentinel_value=-999.0)
        # 30 days + 1 second past end of sim
        obs_date = "2020-07-31T00:00:01"
        obs = _make_ts("W1", [obs_date], [0.0])
        result = interpolate_to_obs_times(obs, sim, config)
        assert result.values[0] == pytest.approx(-999.0)

    def test_forward_clamp_at_boundary(self) -> None:
        """Obs exactly at forward extrapolation boundary gets interpolated value."""
        sim = _make_ts("W1", ["2020-06-01", "2020-07-01"], [50.0, 60.0])
        extrap = timedelta(days=30)
        config = InterpolationConfig(max_extrapolation_time=extrap)
        # Exactly 30 days past end → should still be valid (np.interp clamps)
        obs = _make_ts("W1", ["2020-07-31T00:00:00"], [0.0])
        result = interpolate_to_obs_times(obs, sim, config)
        assert result.values[0] != -999.0

    def test_backward_clamp(self) -> None:
        """Obs before sim range but within extrapolation gets clamped value."""
        sim = _make_ts("W1", ["2020-03-01", "2020-04-01"], [100.0, 200.0])
        config = InterpolationConfig(max_extrapolation_time=timedelta(days=60))
        obs = _make_ts("W1", ["2020-02-01"], [0.0])
        result = interpolate_to_obs_times(obs, sim, config)
        # np.interp clamps to first value when before range
        assert result.values[0] == pytest.approx(100.0)

    def test_single_sim_sample(self) -> None:
        """Single sim sample: obs at same time gets value, far away gets sentinel."""
        sim = _make_ts("W1", ["2020-06-15"], [42.0])
        obs = _make_ts("W1", ["2020-06-15", "2025-01-01"], [0.0, 0.0])
        result = interpolate_to_obs_times(obs, sim)
        assert result.values[0] == pytest.approx(42.0)
        assert result.values[1] == pytest.approx(-999.0)

    def test_np_interp_clamp_behaviour(self) -> None:
        """Document np.interp clamping: values outside range get boundary values."""
        sim = _make_ts("W1", ["2020-01-01", "2020-12-31"], [0.0, 100.0])
        config = InterpolationConfig(max_extrapolation_time=timedelta(days=365))
        # Obs before range → np.interp clamps to first value
        obs = _make_ts("W1", ["2019-01-01"], [0.0])
        result = interpolate_to_obs_times(obs, sim, config)
        assert result.values[0] == pytest.approx(0.0)

    def test_midpoint_precision(self) -> None:
        """Verify interpolation at exact midpoint."""
        sim = _make_ts("W1", ["2020-01-01", "2020-01-31"], [0.0, 300.0])
        # Midpoint: Jan 16 at 00:00 (15 days / 30 days = 0.5)
        obs = _make_ts("W1", ["2020-01-16"], [0.0])
        result = interpolate_to_obs_times(obs, sim)
        # 15 days out of 30 days → fraction = 15/30 = 0.5
        expected = 0.0 + 300.0 * (15.0 * 86400) / (30.0 * 86400)
        assert result.values[0] == pytest.approx(expected, rel=1e-6)

    def test_internal_gap_nan_skipped(self) -> None:
        """NaN values in simulated series are skipped during interpolation."""
        sim = _make_ts(
            "W1",
            ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"],
            [10.0, float("nan"), float("nan"), 40.0],
        )
        obs = _make_ts("W1", ["2020-02-15"], [0.0])
        result = interpolate_to_obs_times(obs, sim)
        # NaN removed, so interp between Jan 1 (10) and Apr 1 (40)
        assert 10.0 < result.values[0] < 40.0


# ---------------------------------------------------------------------------
# TestRealisticMultiBore — batch processing with realistic data
# ---------------------------------------------------------------------------


class TestRealisticMultiBore:
    """Tests with multiple bores and realistic data volumes."""

    def test_200_obs_per_bore(self) -> None:
        """200 observations per bore, daily obs vs monthly sim."""
        rng = np.random.default_rng(42)
        sim = _make_monthly_ts("W1", 2020, 24, lambda i: 100.0 + i)
        dates = ["2020-01-01"] + [
            str(np.datetime64("2020-01-01") + np.timedelta64(int(d), "D"))
            for d in sorted(rng.choice(730, size=199, replace=False))
        ]
        obs = _make_ts("W1", dates, [0.0] * 200)
        result = interpolate_to_obs_times(obs, sim)
        assert len(result.values) == 200
        assert np.all(result.values >= 100.0)
        assert np.all(result.values <= 123.0)

    def test_irregular_spacing(self) -> None:
        """Irregularly-spaced observations are handled correctly."""
        sim = _make_monthly_ts("W1", 2020, 12, lambda i: float(i) * 10)
        # Cluster of obs in first week, then a gap, then one at the end
        obs = _make_ts(
            "W1",
            ["2020-01-02", "2020-01-03", "2020-01-04", "2020-11-15"],
            [0.0] * 4,
        )
        result = interpolate_to_obs_times(obs, sim)
        assert len(result.values) == 4
        # First 3 should be close to 0 (near Jan 1)
        assert all(v < 5.0 for v in result.values[:3])
        # Last should be near 100-110
        assert result.values[3] > 95.0

    def test_c2vsimfg_bore_ids_with_percent(self) -> None:
        """Bore IDs with % character are preserved through interpolation."""
        bore_id = "S_380313N1219426W001%1"
        sim = _make_ts(bore_id, ["2020-01-01", "2020-12-01"], [100.0, 200.0])
        obs = _make_ts(bore_id, ["2020-06-01"], [0.0])
        result = interpolate_to_obs_times(obs, sim)
        assert result.bore_id == bore_id

    def test_tab_delimited_smp_roundtrip(self, tmp_path: Path) -> None:
        """SMP files with tab-separated data survive write→read roundtrip."""
        ts = _make_ts("WELL_01", ["2020-01-01", "2020-02-01"], [123.4567, 234.5678])
        writer = SMPWriter(tmp_path / "test.smp")
        writer.write({"WELL_01": ts})
        reader = SMPReader(tmp_path / "test.smp")
        loaded = reader.read()
        assert "WELL_01" in loaded
        np.testing.assert_allclose(loaded["WELL_01"].values, ts.values, atol=0.001)

    def test_50_bore_batch(self) -> None:
        """Batch interpolation with 50 bores."""
        obs: dict[str, SMPTimeSeries] = {}
        sim: dict[str, SMPTimeSeries] = {}
        for i in range(50):
            bid = f"W{i:03d}"
            obs[bid] = _make_ts(bid, ["2020-06-15"], [0.0])
            sim[bid] = _make_ts(bid, ["2020-01-01", "2020-12-01"], [float(i), float(i + 100)])
        result = interpolate_batch(obs, sim)
        assert len(result) == 50
        for _bid, ts in result.items():
            assert len(ts.values) == 1

    def test_unmatched_bore_filtering(self) -> None:
        """Bores only in obs (not in sim) are excluded from output."""
        obs = {
            "MATCH": _make_ts("MATCH", ["2020-06-15"], [0.0]),
            "NOMATCH": _make_ts("NOMATCH", ["2020-06-15"], [0.0]),
        }
        sim = {
            "MATCH": _make_ts("MATCH", ["2020-01-01", "2020-12-01"], [10.0, 20.0]),
        }
        result = interpolate_batch(obs, sim)
        assert "MATCH" in result
        assert "NOMATCH" not in result

    def test_bore_ids_case_sensitive(self) -> None:
        """Bore ID matching is case-sensitive."""
        obs = {"Well_01": _make_ts("Well_01", ["2020-06-15"], [0.0])}
        sim = {"WELL_01": _make_ts("WELL_01", ["2020-01-01", "2020-12-01"], [10.0, 20.0])}
        result = interpolate_batch(obs, sim)
        assert len(result) == 0

    def test_many_bores_different_date_ranges(self) -> None:
        """Bores with different sim date ranges are handled independently."""
        obs = {
            "W1": _make_ts("W1", ["2020-06-15"], [0.0]),
            "W2": _make_ts("W2", ["2021-06-15"], [0.0]),
        }
        sim = {
            "W1": _make_monthly_ts("W1", 2020, 12, lambda i: 100.0 + i),
            "W2": _make_monthly_ts("W2", 2021, 12, lambda i: 200.0 + i),
        }
        result = interpolate_batch(obs, sim)
        assert 100.0 <= result["W1"].values[0] <= 111.0
        assert 200.0 <= result["W2"].values[0] <= 211.0


# ---------------------------------------------------------------------------
# TestExcludedRecords
# ---------------------------------------------------------------------------


class TestExcludedRecords:
    """Verify excluded flag handling."""

    def test_excluded_flags_preserved(self) -> None:
        """Excluded flags from observed are preserved in output."""
        obs = _make_ts(
            "W1",
            ["2020-01-15", "2020-06-15"],
            [0.0, 0.0],
            excluded=[True, False],
        )
        sim = _make_ts("W1", ["2020-01-01", "2020-12-01"], [100.0, 200.0])
        result = interpolate_to_obs_times(obs, sim)
        assert result.excluded[0] is np.True_
        assert result.excluded[1] is np.False_

    def test_excluded_obs_still_interpolated(self) -> None:
        """Excluded obs times still get interpolated values (not sentinel)."""
        obs = _make_ts("W1", ["2020-06-15"], [0.0], excluded=[True])
        sim = _make_ts("W1", ["2020-01-01", "2020-12-01"], [100.0, 200.0])
        result = interpolate_to_obs_times(obs, sim)
        assert result.values[0] != -999.0
        assert 100.0 < result.values[0] < 200.0


# ---------------------------------------------------------------------------
# TestSMPRealFormats — edge cases in SMP parsing
# ---------------------------------------------------------------------------


class TestSMPRealFormats:
    """Test SMP I/O with realistic format variations."""

    def test_25_char_bore_id(self, tmp_path: Path) -> None:
        """Bore ID exactly 25 characters (max SMP width)."""
        bid = "A" * 25
        ts = _make_ts(bid, ["2020-01-01"], [42.0])
        writer = SMPWriter(tmp_path / "test.smp")
        writer.write({bid: ts})
        reader = SMPReader(tmp_path / "test.smp")
        loaded = reader.read()
        assert bid in loaded

    def test_scientific_notation_values(self, tmp_path: Path) -> None:
        """Very large values survive SMP roundtrip."""
        ts = _make_ts("W1", ["2020-01-01"], [1.2345e6])
        writer = SMPWriter(tmp_path / "test.smp")
        writer.write({"W1": ts})
        reader = SMPReader(tmp_path / "test.smp")
        loaded = reader.read()
        assert loaded["W1"].values[0] == pytest.approx(1.2345e6, rel=0.01)

    def test_negative_values(self, tmp_path: Path) -> None:
        """Negative water levels (below sea level) roundtrip correctly."""
        ts = _make_ts("W1", ["2020-01-01", "2020-02-01"], [-15.1234, -0.0001])
        writer = SMPWriter(tmp_path / "test.smp")
        writer.write({"W1": ts})
        reader = SMPReader(tmp_path / "test.smp")
        loaded = reader.read()
        np.testing.assert_allclose(loaded["W1"].values, ts.values, atol=0.001)

    def test_100k_record_file(self, tmp_path: Path) -> None:
        """100K-record SMP file writes and reads correctly."""
        n = 100_000
        times = np.arange(
            np.datetime64("2000-01-01"),
            np.datetime64("2000-01-01") + np.timedelta64(n, "h"),
            np.timedelta64(1, "h"),
        ).astype("datetime64[s]")
        values = np.random.default_rng(42).standard_normal(n) * 10 + 100
        ts = SMPTimeSeries(
            bore_id="BIGWELL",
            times=times,
            values=values,
            excluded=np.zeros(n, dtype=np.bool_),
        )
        writer = SMPWriter(tmp_path / "big.smp")
        writer.write({"BIGWELL": ts})
        reader = SMPReader(tmp_path / "big.smp")
        loaded = reader.read()
        assert loaded["BIGWELL"].n_records == n
        np.testing.assert_allclose(loaded["BIGWELL"].values, values, atol=0.001)


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

    def test_iwfm2obs_matches_expected(self, fixture_dir: Path, tmp_path: Path) -> None:
        """Run iwfm2obs on fixture data and compare against expected output."""
        obs_path = fixture_dir / "obs_gw.smp"
        sim_path = fixture_dir / "sim_gw.smp"
        expected_path = fixture_dir / "expected_gw_out.smp"
        out_path = tmp_path / "gw_out.smp"

        result = iwfm2obs(obs_path, sim_path, out_path)
        expected = SMPReader(expected_path).read()

        assert set(result.keys()) == set(expected.keys())
        for bid in result:
            np.testing.assert_allclose(
                result[bid].values,
                expected[bid].values,
                atol=0.001,
                err_msg=f"Mismatch for bore {bid}",
            )

    def test_written_file_matches_expected(self, fixture_dir: Path, tmp_path: Path) -> None:
        """Output SMP file on disk matches expected when re-read."""
        obs_path = fixture_dir / "obs_gw.smp"
        sim_path = fixture_dir / "sim_gw.smp"
        expected_path = fixture_dir / "expected_gw_out.smp"
        out_path = tmp_path / "gw_out.smp"

        iwfm2obs(obs_path, sim_path, out_path)

        written = SMPReader(out_path).read()
        expected = SMPReader(expected_path).read()

        assert set(written.keys()) == set(expected.keys())
        for bid in written:
            np.testing.assert_allclose(
                written[bid].values,
                expected[bid].values,
                atol=0.001,
                err_msg=f"File mismatch for bore {bid}",
            )


# ---------------------------------------------------------------------------
# TestIWFM2OBSBenchmark — performance benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
class TestIWFM2OBSBenchmark:
    """Benchmark IWFM2OBS performance with pytest-benchmark."""

    @staticmethod
    def _make_large_ts(n: int) -> tuple[SMPTimeSeries, SMPTimeSeries]:
        """Create obs/sim pair with *n* observation points."""
        rng = np.random.default_rng(42)
        base = np.datetime64("2000-01-01", "s")
        sim_times = base + np.arange(0, n * 2, 2).astype("timedelta64[h]")
        sim_values = rng.standard_normal(len(sim_times)) * 10 + 100
        obs_times = base + np.arange(1, n * 2, 2).astype("timedelta64[h]")
        obs_values = np.zeros(len(obs_times))
        sim = SMPTimeSeries(
            bore_id="W1",
            times=sim_times.astype("datetime64[s]"),
            values=sim_values,
            excluded=np.zeros(len(sim_times), dtype=np.bool_),
        )
        obs = SMPTimeSeries(
            bore_id="W1",
            times=obs_times.astype("datetime64[s]"),
            values=obs_values,
            excluded=np.zeros(len(obs_times), dtype=np.bool_),
        )
        return obs, sim

    def test_benchmark_1k_obs(self, benchmark: object) -> None:
        """Benchmark interpolation with 1K obs points."""
        obs, sim = self._make_large_ts(1_000)
        benchmark(interpolate_to_obs_times, obs, sim)  # type: ignore[operator]

    def test_benchmark_10k_obs(self, benchmark: object) -> None:
        """Benchmark interpolation with 10K obs points."""
        obs, sim = self._make_large_ts(10_000)
        benchmark(interpolate_to_obs_times, obs, sim)  # type: ignore[operator]

    def test_benchmark_100k_obs(self, benchmark: object) -> None:
        """Benchmark interpolation with 100K obs points."""
        obs, sim = self._make_large_ts(100_000)
        benchmark(interpolate_to_obs_times, obs, sim)  # type: ignore[operator]

    def test_benchmark_50_bore_batch(self, benchmark: object) -> None:
        """Benchmark batch interpolation with 50 bores."""
        obs: dict[str, SMPTimeSeries] = {}
        sim: dict[str, SMPTimeSeries] = {}
        for i in range(50):
            bid = f"W{i:03d}"
            o, s = self._make_large_ts(500)
            obs[bid] = SMPTimeSeries(
                bore_id=bid, times=o.times, values=o.values, excluded=o.excluded
            )
            sim[bid] = SMPTimeSeries(
                bore_id=bid, times=s.times, values=s.values, excluded=s.excluded
            )
        benchmark(interpolate_batch, obs, sim)  # type: ignore[operator]

    def test_benchmark_full_file_io(self, benchmark: object, tmp_path: Path) -> None:
        """Benchmark full iwfm2obs including file I/O."""
        rng = np.random.default_rng(42)
        obs_data: dict[str, SMPTimeSeries] = {}
        sim_data: dict[str, SMPTimeSeries] = {}
        base = np.datetime64("2000-01-01", "s")
        for i in range(10):
            bid = f"BORE_{i:02d}"
            sim_t = base + np.arange(0, 1000, 1).astype("timedelta64[D]")
            sim_v = rng.standard_normal(1000) * 5 + 100
            obs_t = base + np.arange(5, 995, 10).astype("timedelta64[D]")
            obs_v = np.zeros(len(obs_t))
            sim_data[bid] = SMPTimeSeries(
                bore_id=bid,
                times=sim_t.astype("datetime64[s]"),
                values=sim_v,
                excluded=np.zeros(1000, dtype=np.bool_),
            )
            obs_data[bid] = SMPTimeSeries(
                bore_id=bid,
                times=obs_t.astype("datetime64[s]"),
                values=obs_v,
                excluded=np.zeros(len(obs_t), dtype=np.bool_),
            )
        obs_file = tmp_path / "obs.smp"
        sim_file = tmp_path / "sim.smp"
        out_file = tmp_path / "out.smp"
        SMPWriter(obs_file).write(obs_data)
        SMPWriter(sim_file).write(sim_data)

        benchmark(iwfm2obs, obs_file, sim_file, out_file)  # type: ignore[operator]
