"""End-to-end calibration pipeline and cross-module integration tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyiwfm.calibration.calctyphyd import (
    CalcTypHydConfig,
    SeasonalPeriod,
    compute_typical_hydrographs,
    read_cluster_weights,
)
from pyiwfm.calibration.clustering import (
    ClusteringConfig,
    fuzzy_cmeans_cluster,
)
from pyiwfm.calibration.iwfm2obs import (
    InterpolationConfig,
    iwfm2obs,
)
from pyiwfm.io.smp import SMPReader, SMPTimeSeries, SMPWriter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    dates: list[str] = []
    values: list[float] = []
    for y in range(start_year, start_year + n_years):
        for m in range(1, 13):
            dates.append(f"{y}-{m:02d}-15")
            values.append(base_value + amplitude * np.sin(2 * np.pi * m / 12))
    return _make_ts(bore_id, dates, values)


def _make_well_data(
    n_wells: int,
    seed: int = 42,
) -> tuple[dict[str, tuple[float, float]], dict[str, SMPTimeSeries]]:
    """Create synthetic well locations and time series for clustering."""
    rng = np.random.default_rng(seed)
    locations: dict[str, tuple[float, float]] = {}
    timeseries: dict[str, SMPTimeSeries] = {}
    dates = [f"2020-{m:02d}-15" for m in range(1, 13)]
    times = np.array(dates, dtype="datetime64[s]")

    for i in range(n_wells):
        wid = f"W{i:02d}"
        if i < n_wells // 2:
            x, y = rng.normal(100.0, 10.0), rng.normal(100.0, 10.0)
            base = 50.0
        else:
            x, y = rng.normal(500.0, 10.0), rng.normal(500.0, 10.0)
            base = 150.0
        locations[wid] = (x, y)
        values = base + 5.0 * np.sin(2 * np.pi * np.arange(12) / 12) + rng.normal(0, 1, 12)
        timeseries[wid] = SMPTimeSeries(
            bore_id=wid,
            times=times.copy(),
            values=values.astype(np.float64),
            excluded=np.zeros(12, dtype=np.bool_),
        )
    return locations, timeseries


# ---------------------------------------------------------------------------
# TestClusterToCalcTypHyd — full clustering → CalcTypHyd pipeline
# ---------------------------------------------------------------------------


class TestClusterToCalcTypHyd:
    """Full pipeline: cluster → to_weights_file → read_cluster_weights → compute_typical_hydrographs."""

    def test_full_pipeline(self, tmp_path: Path) -> None:
        """End-to-end: cluster wells, write weights, read weights, compute typical hydrographs."""
        locations, timeseries = _make_well_data(10)
        config = ClusteringConfig(n_clusters=3, random_seed=42)
        cluster_result = fuzzy_cmeans_cluster(locations, timeseries, config)

        weights_file = tmp_path / "weights.txt"
        cluster_result.to_weights_file(weights_file)
        weights = read_cluster_weights(weights_file)

        assert len(weights) == 10
        result = compute_typical_hydrographs(timeseries, weights)
        assert len(result.hydrographs) == 3

    def test_weights_roundtrip_precision(self, tmp_path: Path) -> None:
        """Cluster weights survive write→read roundtrip to 6 decimal places."""
        locations, timeseries = _make_well_data(10)
        config = ClusteringConfig(n_clusters=3, random_seed=42)
        cluster_result = fuzzy_cmeans_cluster(locations, timeseries, config)

        weights_file = tmp_path / "weights.txt"
        cluster_result.to_weights_file(weights_file)
        weights = read_cluster_weights(weights_file)

        for i, wid in enumerate(cluster_result.well_ids):
            if wid in weights:
                np.testing.assert_allclose(
                    weights[wid],
                    cluster_result.membership[i],
                    atol=1e-5,
                    err_msg=f"Weights mismatch for {wid}",
                )

    def test_spatial_group_match(self) -> None:
        """Wells in same spatial group have high membership in same cluster."""
        locations, timeseries = _make_well_data(20, seed=123)
        config = ClusteringConfig(n_clusters=2, random_seed=42, spatial_weight=0.9)
        result = fuzzy_cmeans_cluster(locations, timeseries, config)

        # First 10 wells are near (100,100), last 10 near (500,500)
        # They should mostly belong to different clusters
        first_group_dominant = [result.get_dominant_cluster(f"W{i:02d}") for i in range(10)]
        second_group_dominant = [result.get_dominant_cluster(f"W{i:02d}") for i in range(10, 20)]

        # At least 7/10 in each group should share a cluster
        from collections import Counter

        c1 = Counter(first_group_dominant).most_common(1)[0][1]
        c2 = Counter(second_group_dominant).most_common(1)[0][1]
        assert c1 >= 7
        assert c2 >= 7

    def test_monthly_periods_pipeline(self, tmp_path: Path) -> None:
        """Pipeline with 12 monthly seasonal periods."""
        locations, timeseries = _make_well_data(10)
        config = ClusteringConfig(n_clusters=2, random_seed=42)
        cluster_result = fuzzy_cmeans_cluster(locations, timeseries, config)

        weights_file = tmp_path / "weights.txt"
        cluster_result.to_weights_file(weights_file)
        weights = read_cluster_weights(weights_file)

        seasons = [SeasonalPeriod(f"M{m:02d}", [m], f"{m:02d}/15") for m in range(1, 13)]
        typhyd_config = CalcTypHydConfig(seasonal_periods=seasons)
        result = compute_typical_hydrographs(timeseries, weights, typhyd_config)
        assert len(result.hydrographs) == 2
        assert len(result.hydrographs[0].values) == 12


# ---------------------------------------------------------------------------
# TestSMPToIWFM2OBSToSMP — SMP file pipeline
# ---------------------------------------------------------------------------


class TestSMPToIWFM2OBSToSMP:
    """Full SMP read → iwfm2obs → SMP write pipeline."""

    def test_multi_bore_smp_pipeline(self, tmp_path: Path) -> None:
        """5 bores through full SMP → interp → SMP pipeline."""
        obs_data: dict[str, SMPTimeSeries] = {}
        sim_data: dict[str, SMPTimeSeries] = {}
        for i in range(5):
            bid = f"BORE_{i:02d}"
            obs_data[bid] = _make_ts(bid, ["2020-06-15"], [0.0])
            sim_data[bid] = _make_ts(
                bid,
                ["2020-01-01", "2020-07-01", "2020-12-01"],
                [100.0 + i * 10, 150.0 + i * 10, 120.0 + i * 10],
            )
        obs_file = tmp_path / "obs.smp"
        sim_file = tmp_path / "sim.smp"
        out_file = tmp_path / "out.smp"
        SMPWriter(obs_file).write(obs_data)
        SMPWriter(sim_file).write(sim_data)
        result = iwfm2obs(obs_file, sim_file, out_file)
        assert len(result) == 5
        assert out_file.exists()
        loaded = SMPReader(out_file).read()
        assert set(loaded.keys()) == set(obs_data.keys())

    def test_excluded_preservation(self, tmp_path: Path) -> None:
        """Excluded flags preserved through full pipeline."""
        obs = _make_ts("W1", ["2020-03-15", "2020-06-15"], [0.0, 0.0], excluded=[True, False])
        sim = _make_ts("W1", ["2020-01-01", "2020-12-01"], [100.0, 200.0])
        obs_file = tmp_path / "obs.smp"
        sim_file = tmp_path / "sim.smp"
        out_file = tmp_path / "out.smp"
        SMPWriter(obs_file).write({"W1": obs})
        SMPWriter(sim_file).write({"W1": sim})
        result = iwfm2obs(obs_file, sim_file, out_file)
        assert result["W1"].excluded[0] is np.True_
        assert result["W1"].excluded[1] is np.False_

    def test_custom_config_nearest(self, tmp_path: Path) -> None:
        """Nearest-neighbor interpolation through full pipeline."""
        from datetime import timedelta

        obs = _make_ts("W1", ["2020-01-05", "2020-11-28"], [0.0, 0.0])
        sim = _make_ts("W1", ["2020-01-01", "2020-12-01"], [100.0, 200.0])
        obs_file = tmp_path / "obs.smp"
        sim_file = tmp_path / "sim.smp"
        out_file = tmp_path / "out.smp"
        SMPWriter(obs_file).write({"W1": obs})
        SMPWriter(sim_file).write({"W1": sim})
        config = InterpolationConfig(
            interpolation_method="nearest",
            max_extrapolation_time=timedelta(days=60),
        )
        result = iwfm2obs(obs_file, sim_file, out_file, config=config)
        # Jan 5 closer to Jan 1 → 100, Nov 28 closer to Dec 1 → 200
        assert result["W1"].values[0] == pytest.approx(100.0)
        assert result["W1"].values[1] == pytest.approx(200.0)

    def test_partial_overlap(self, tmp_path: Path) -> None:
        """Obs partly outside sim range: some sentinel, some interpolated."""
        from datetime import timedelta

        obs = _make_ts("W1", ["2019-06-15", "2020-06-15", "2025-01-01"], [0.0, 0.0, 0.0])
        sim = _make_ts("W1", ["2020-01-01", "2020-12-01"], [100.0, 200.0])
        obs_file = tmp_path / "obs.smp"
        sim_file = tmp_path / "sim.smp"
        out_file = tmp_path / "out.smp"
        SMPWriter(obs_file).write({"W1": obs})
        SMPWriter(sim_file).write({"W1": sim})
        config = InterpolationConfig(
            max_extrapolation_time=timedelta(days=30),
            sentinel_value=-999.0,
        )
        result = iwfm2obs(obs_file, sim_file, out_file, config=config)
        assert result["W1"].values[0] == pytest.approx(-999.0)  # Far before
        assert 100.0 < result["W1"].values[1] < 200.0  # Within range
        assert result["W1"].values[2] == pytest.approx(-999.0)  # Far after


# ---------------------------------------------------------------------------
# TestClusteringEdgeCases
# ---------------------------------------------------------------------------


class TestClusteringEdgeCases:
    """Edge cases for clustering module."""

    def test_n_wells_equals_n_clusters(self) -> None:
        """n_wells == n_clusters should still produce valid results."""
        locations, timeseries = _make_well_data(3, seed=42)
        # Take only 3 wells
        locs = dict(list(locations.items())[:3])
        ts = dict(list(timeseries.items())[:3])
        config = ClusteringConfig(n_clusters=3, random_seed=42)
        result = fuzzy_cmeans_cluster(locs, ts, config)
        assert result.membership.shape == (3, 3)
        np.testing.assert_allclose(result.membership.sum(axis=1), 1.0, atol=1e-6)

    def test_single_timestep(self) -> None:
        """Wells with single timestep can still be clustered."""
        locations = {f"W{i}": (float(i * 100), 0.0) for i in range(5)}
        ts: dict[str, SMPTimeSeries] = {}
        for i in range(5):
            ts[f"W{i}"] = _make_ts(f"W{i}", ["2020-01-15"], [float(i * 10)])
        config = ClusteringConfig(n_clusters=2, random_seed=42)
        result = fuzzy_cmeans_cluster(locations, ts, config)
        assert result.membership.shape == (5, 2)

    def test_constant_values(self) -> None:
        """Wells with identical constant values: clustering still converges."""
        locations = {f"W{i}": (float(i * 100), float(i * 50)) for i in range(6)}
        ts: dict[str, SMPTimeSeries] = {}
        for i in range(6):
            ts[f"W{i}"] = _make_ts(
                f"W{i}",
                [f"2020-{m:02d}-15" for m in range(1, 13)],
                [100.0] * 12,
            )
        config = ClusteringConfig(n_clusters=2, random_seed=42)
        result = fuzzy_cmeans_cluster(locations, ts, config)
        # Should still produce valid membership
        np.testing.assert_allclose(result.membership.sum(axis=1), 1.0, atol=1e-6)

    def test_co_located_wells(self) -> None:
        """Wells at the same location differ only by temporal pattern."""
        locations = {f"W{i}": (100.0, 100.0) for i in range(6)}
        ts: dict[str, SMPTimeSeries] = {}
        for i in range(6):
            base = 50.0 if i < 3 else 150.0
            ts[f"W{i}"] = _make_annual_ts(f"W{i}", base_value=base, amplitude=10.0)
        config = ClusteringConfig(n_clusters=2, random_seed=42, temporal_weight=0.9)
        result = fuzzy_cmeans_cluster(locations, ts, config)
        assert result.membership.shape == (6, 2)


# ---------------------------------------------------------------------------
# TestCombinedScenarios
# ---------------------------------------------------------------------------


class TestCombinedScenarios:
    """Cross-module integration scenarios."""

    def test_empty_inputs(self) -> None:
        """Empty water levels + empty weights → empty result."""
        result = compute_typical_hydrographs({}, {})
        assert len(result.hydrographs) == 0
        assert len(result.well_means) == 0

    def test_iwfm2obs_output_to_calctyphyd(self, tmp_path: Path) -> None:
        """IWFM2OBS interpolated output feeds into CalcTypHyd."""
        # Create monthly sim data for 2 bores, 2 years
        obs_data: dict[str, SMPTimeSeries] = {}
        sim_data: dict[str, SMPTimeSeries] = {}
        for bid in ["W1", "W2"]:
            base = 100.0 if bid == "W1" else 200.0
            sim_data[bid] = _make_annual_ts(bid, base, 10.0, n_years=2)
            # Monthly obs
            dates = [f"{y}-{m:02d}-15" for y in [2020, 2021] for m in range(1, 13)]
            obs_data[bid] = _make_ts(bid, dates, [0.0] * 24)

        obs_file = tmp_path / "obs.smp"
        sim_file = tmp_path / "sim.smp"
        out_file = tmp_path / "interp.smp"
        SMPWriter(obs_file).write(obs_data)
        SMPWriter(sim_file).write(sim_data)
        interp_result = iwfm2obs(obs_file, sim_file, out_file)

        # Feed interpolated output into CalcTypHyd
        weights = {"W1": np.array([0.7, 0.3]), "W2": np.array([0.3, 0.7])}
        typhyd = compute_typical_hydrographs(interp_result, weights)
        assert len(typhyd.hydrographs) == 2
        for hyd in typhyd.hydrographs:
            assert len(hyd.values) == 4
            assert not np.all(np.isnan(hyd.values))

    def test_large_scale_pipeline(self, tmp_path: Path) -> None:
        """30 wells, 5 clusters, 5 years of data through full pipeline."""
        locations, timeseries = _make_well_data(30, seed=99)
        # Extend to 5 years
        for wid in list(timeseries.keys()):
            base = float(np.mean(timeseries[wid].values))
            timeseries[wid] = _make_annual_ts(wid, base, 5.0, n_years=5)

        config = ClusteringConfig(n_clusters=5, random_seed=42)
        cluster_result = fuzzy_cmeans_cluster(locations, timeseries, config)

        weights_file = tmp_path / "weights.txt"
        cluster_result.to_weights_file(weights_file)
        weights = read_cluster_weights(weights_file)

        typhyd = compute_typical_hydrographs(timeseries, weights)
        assert len(typhyd.hydrographs) == 5
        assert len(typhyd.well_means) == 30


# ---------------------------------------------------------------------------
# TestPipelineBenchmark — performance benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
class TestPipelineBenchmark:
    """Benchmark end-to-end pipelines."""

    def test_benchmark_cluster_to_calctyphyd(self, benchmark: object, tmp_path: Path) -> None:
        """Benchmark cluster→CalcTypHyd pipeline: 20 wells, 3 clusters, 3 years."""
        locations, timeseries = _make_well_data(20, seed=42)
        for wid in list(timeseries.keys()):
            base = float(np.mean(timeseries[wid].values))
            timeseries[wid] = _make_annual_ts(wid, base, 5.0, n_years=3)

        def pipeline() -> None:
            config = ClusteringConfig(n_clusters=3, random_seed=42)
            cr = fuzzy_cmeans_cluster(locations, timeseries, config)
            weights_file = tmp_path / "bench_weights.txt"
            cr.to_weights_file(weights_file)
            weights = read_cluster_weights(weights_file)
            compute_typical_hydrographs(timeseries, weights)

        benchmark(pipeline)  # type: ignore[operator]

    def test_benchmark_smp_pipeline(self, benchmark: object, tmp_path: Path) -> None:
        """Benchmark SMP→IWFM2OBS→SMP: 20 bores, 500 obs each."""
        rng = np.random.default_rng(42)
        obs_data: dict[str, SMPTimeSeries] = {}
        sim_data: dict[str, SMPTimeSeries] = {}
        base = np.datetime64("2000-01-01", "s")
        for i in range(20):
            bid = f"BORE_{i:02d}"
            sim_t = base + np.arange(0, 1000, 1).astype("timedelta64[D]")
            sim_v = rng.standard_normal(1000) * 5 + 100
            obs_t = base + np.arange(1, 1000, 2).astype("timedelta64[D]")
            sim_data[bid] = SMPTimeSeries(
                bore_id=bid,
                times=sim_t.astype("datetime64[s]"),
                values=sim_v,
                excluded=np.zeros(1000, dtype=np.bool_),
            )
            obs_data[bid] = SMPTimeSeries(
                bore_id=bid,
                times=obs_t.astype("datetime64[s]"),
                values=np.zeros(len(obs_t)),
                excluded=np.zeros(len(obs_t), dtype=np.bool_),
            )
        obs_file = tmp_path / "bench_obs.smp"
        sim_file = tmp_path / "bench_sim.smp"
        out_file = tmp_path / "bench_out.smp"
        SMPWriter(obs_file).write(obs_data)
        SMPWriter(sim_file).write(sim_data)

        benchmark(iwfm2obs, obs_file, sim_file, out_file)  # type: ignore[operator]
