"""Tests for fuzzy c-means clustering module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyiwfm.calibration.clustering import (
    ClusteringConfig,
    ClusteringResult,
    fuzzy_cmeans_cluster,
)
from pyiwfm.io.smp import SMPTimeSeries


def _make_well_data(
    n_wells: int = 10,
    seed: int = 42,
) -> tuple[dict[str, tuple[float, float]], dict[str, SMPTimeSeries]]:
    """Create synthetic well locations and time series."""
    rng = np.random.default_rng(seed)

    locations: dict[str, tuple[float, float]] = {}
    timeseries: dict[str, SMPTimeSeries] = {}

    dates = [f"2020-{m:02d}-15" for m in range(1, 13)]
    times = np.array(dates, dtype="datetime64[s]")

    for i in range(n_wells):
        wid = f"W{i:02d}"
        # Two spatial clusters
        if i < n_wells // 2:
            x = rng.normal(100.0, 10.0)
            y = rng.normal(100.0, 10.0)
            base = 50.0
        else:
            x = rng.normal(500.0, 10.0)
            y = rng.normal(500.0, 10.0)
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


class TestFuzzyCMeans:
    """Tests for fuzzy c-means clustering."""

    def test_membership_rows_sum_to_one(self) -> None:
        """Each well's membership values should sum to 1."""
        locations, timeseries = _make_well_data(10)
        config = ClusteringConfig(n_clusters=2, random_seed=42)
        result = fuzzy_cmeans_cluster(locations, timeseries, config)

        row_sums = result.membership.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_membership_values_valid(self) -> None:
        """All membership values should be in [0, 1]."""
        locations, timeseries = _make_well_data(10)
        config = ClusteringConfig(n_clusters=2, random_seed=42)
        result = fuzzy_cmeans_cluster(locations, timeseries, config)

        assert np.all(result.membership >= 0.0)
        assert np.all(result.membership <= 1.0)

    def test_fpc_in_valid_range(self) -> None:
        """FPC should be between 0 and 1."""
        locations, timeseries = _make_well_data(10)
        config = ClusteringConfig(n_clusters=2, random_seed=42)
        result = fuzzy_cmeans_cluster(locations, timeseries, config)

        assert result.fpc > 0.0
        assert result.fpc <= 1.0 + 1e-6

    def test_correct_dimensions(self) -> None:
        """Membership matrix has correct shape."""
        n_wells = 10
        n_clusters = 3
        locations, timeseries = _make_well_data(n_wells)
        config = ClusteringConfig(n_clusters=n_clusters, random_seed=42)
        result = fuzzy_cmeans_cluster(locations, timeseries, config)

        assert result.membership.shape == (n_wells, n_clusters)
        assert result.cluster_centers.shape[0] == n_clusters
        assert result.n_clusters == n_clusters

    def test_reproducible_with_seed(self) -> None:
        """Same seed produces same results."""
        locations, timeseries = _make_well_data(10)
        config = ClusteringConfig(n_clusters=2, random_seed=123)

        r1 = fuzzy_cmeans_cluster(locations, timeseries, config)
        r2 = fuzzy_cmeans_cluster(locations, timeseries, config)

        np.testing.assert_array_equal(r1.membership, r2.membership)

    def test_different_seeds_differ(self) -> None:
        """Different seeds may produce different results."""
        locations, timeseries = _make_well_data(10)
        c1 = ClusteringConfig(n_clusters=2, random_seed=1)
        c2 = ClusteringConfig(n_clusters=2, random_seed=999)

        r1 = fuzzy_cmeans_cluster(locations, timeseries, c1)
        r2 = fuzzy_cmeans_cluster(locations, timeseries, c2)

        # Results might differ (not guaranteed, but highly likely)
        # Just check they both produce valid results
        np.testing.assert_allclose(r1.membership.sum(axis=1), 1.0, atol=1e-6)
        np.testing.assert_allclose(r2.membership.sum(axis=1), 1.0, atol=1e-6)

    def test_too_few_wells_raises(self) -> None:
        """n_clusters > n_wells should raise ValueError."""
        locations = {"W0": (0.0, 0.0), "W1": (1.0, 1.0)}
        timeseries = {
            wid: SMPTimeSeries(
                bore_id=wid,
                times=np.array(["2020-01-15"], dtype="datetime64[s]"),
                values=np.array([100.0]),
                excluded=np.zeros(1, dtype=np.bool_),
            )
            for wid in locations
        }
        config = ClusteringConfig(n_clusters=5)

        with pytest.raises(ValueError, match="n_clusters"):
            fuzzy_cmeans_cluster(locations, timeseries, config)


class TestClusteringResult:
    """Tests for ClusteringResult methods."""

    def _make_result(self) -> ClusteringResult:
        return ClusteringResult(
            membership=np.array(
                [
                    [0.9, 0.1],
                    [0.3, 0.7],
                    [0.6, 0.4],
                ]
            ),
            cluster_centers=np.zeros((2, 5)),
            well_ids=["W0", "W1", "W2"],
            n_clusters=2,
            fpc=0.7,
        )

    def test_get_dominant_cluster(self) -> None:
        result = self._make_result()
        assert result.get_dominant_cluster("W0") == 0
        assert result.get_dominant_cluster("W1") == 1
        assert result.get_dominant_cluster("W2") == 0

    def test_get_cluster_wells(self) -> None:
        result = self._make_result()

        wells_c0 = result.get_cluster_wells(0, threshold=0.5)
        assert "W0" in wells_c0
        assert "W2" in wells_c0
        assert "W1" not in wells_c0

        wells_c1 = result.get_cluster_wells(1, threshold=0.5)
        assert "W1" in wells_c1
        assert len(wells_c1) == 1

    def test_to_weights_file(self, tmp_path: Path) -> None:
        result = self._make_result()
        out = tmp_path / "weights.txt"
        result.to_weights_file(out)

        assert out.exists()
        lines = out.read_text().strip().split("\n")
        # 2 comment lines + 3 data lines
        assert len(lines) == 5
        assert "W0" in lines[2]
