"""
Fuzzy c-means clustering for IWFM observation wells.

Pure NumPy implementation — no scikit-fuzzy dependency.  Wells are
clustered using a combination of spatial (x, y) and temporal features
(cross-correlation, amplitude, trend, seasonality).

Example
-------
>>> from pyiwfm.calibration.clustering import fuzzy_cmeans_cluster
>>> result = fuzzy_cmeans_cluster(well_locations, well_timeseries)
>>> print(f"FPC: {result.fpc:.3f}")
>>> print(result.get_cluster_wells(0))
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pyiwfm.io.smp import SMPTimeSeries


@dataclass
class ClusteringConfig:
    """Configuration for fuzzy c-means clustering.

    Attributes
    ----------
    n_clusters : int
        Number of clusters.
    fuzziness : float
        Fuzziness parameter ``m`` (must be > 1).
    spatial_weight : float
        Weight for spatial (x, y) features.
    temporal_weight : float
        Weight for temporal features.
    max_iterations : int
        Maximum number of FCM iterations.
    tolerance : float
        Convergence tolerance on membership change.
    random_seed : int | None
        Random seed for reproducibility.
    """

    n_clusters: int = 5
    fuzziness: float = 2.0
    spatial_weight: float = 0.3
    temporal_weight: float = 0.7
    max_iterations: int = 300
    tolerance: float = 1e-6
    random_seed: int | None = None


@dataclass
class ClusteringResult:
    """Result of fuzzy c-means clustering.

    Attributes
    ----------
    membership : NDArray[np.float64]
        Membership matrix, shape ``(n_wells, n_clusters)``.
        Rows sum to 1.
    cluster_centers : NDArray[np.float64]
        Cluster centers in feature space, shape ``(n_clusters, n_features)``.
    well_ids : list[str]
        Well identifiers in row order.
    n_clusters : int
        Number of clusters.
    fpc : float
        Fuzzy Partition Coefficient (0 to 1, higher = crisper).
    """

    membership: NDArray[np.float64]
    cluster_centers: NDArray[np.float64]
    well_ids: list[str]
    n_clusters: int
    fpc: float

    def get_dominant_cluster(self, well_id: str) -> int:
        """Return the dominant cluster ID for a well.

        Parameters
        ----------
        well_id : str
            Well identifier.

        Returns
        -------
        int
            Cluster index with highest membership.
        """
        idx = self.well_ids.index(well_id)
        return int(np.argmax(self.membership[idx]))

    def get_cluster_wells(self, cluster_id: int, threshold: float = 0.5) -> list[str]:
        """Return wells with membership above threshold for a cluster.

        Parameters
        ----------
        cluster_id : int
            Cluster index.
        threshold : float
            Minimum membership value.

        Returns
        -------
        list[str]
            Well IDs above the threshold.
        """
        mask = self.membership[:, cluster_id] >= threshold
        return [self.well_ids[i] for i in range(len(self.well_ids)) if mask[i]]

    def to_weights_file(self, output_path: Path) -> None:
        """Write cluster weights in CalcTypHyd format.

        Parameters
        ----------
        output_path : Path
            Output file path.
        """
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# Fuzzy c-means cluster weights ({self.n_clusters} clusters)\n")
            f.write("# well_id  " + "  ".join(f"c{i}" for i in range(self.n_clusters)) + "\n")
            for i, well_id in enumerate(self.well_ids):
                wts = "  ".join(f"{w:.6f}" for w in self.membership[i])
                f.write(f"{well_id}  {wts}\n")


def _extract_features(
    well_locations: dict[str, tuple[float, float]],
    well_timeseries: dict[str, SMPTimeSeries],
    well_ids: list[str],
    spatial_weight: float,
    temporal_weight: float,
) -> NDArray[np.float64]:
    """Extract and combine spatial + temporal feature vectors.

    Temporal features: amplitude, linear trend slope, seasonal strength
    (ratio of seasonal to total variance).
    """
    n_wells = len(well_ids)

    # Spatial features (normalized)
    xy = np.array([well_locations[wid] for wid in well_ids], dtype=np.float64)
    if n_wells > 1:
        xy_min = xy.min(axis=0)
        xy_range = xy.max(axis=0) - xy_min
        xy_range[xy_range == 0] = 1.0
        xy_norm = (xy - xy_min) / xy_range
    else:
        xy_norm = np.zeros_like(xy)

    # Temporal features
    temporal = np.zeros((n_wells, 3), dtype=np.float64)  # amplitude, trend, seasonality
    for i, wid in enumerate(well_ids):
        if wid not in well_timeseries:
            continue
        ts = well_timeseries[wid]
        valid = ts.valid_mask
        vals = ts.values[valid]

        if len(vals) < 2:
            continue

        # Amplitude: range of valid values
        amplitude = float(np.max(vals) - np.min(vals))

        # Linear trend slope (simple least squares)
        t = np.arange(len(vals), dtype=np.float64)
        if len(t) > 1:
            slope = float(np.polyfit(t, vals, 1)[0])
        else:
            slope = 0.0

        # Seasonal strength: variance of monthly means / total variance
        times = ts.times[valid]
        months = times.astype("datetime64[M]").astype(int) % 12
        total_var = float(np.var(vals))
        if total_var > 0 and len(vals) >= 12:
            monthly_means = np.array(
                [
                    np.mean(vals[months == m]) if np.any(months == m) else np.mean(vals)
                    for m in range(12)
                ]
            )
            seasonal_var = float(np.var(monthly_means))
            seasonality = seasonal_var / total_var
        else:
            seasonality = 0.0

        temporal[i] = [amplitude, slope, seasonality]

    # Normalize temporal features
    if n_wells > 1:
        t_min = temporal.min(axis=0)
        t_range = temporal.max(axis=0) - t_min
        t_range[t_range == 0] = 1.0
        temporal_norm = (temporal - t_min) / t_range
    else:
        temporal_norm = np.zeros_like(temporal)

    # Combine with weights
    spatial_features = xy_norm * spatial_weight
    temporal_features = temporal_norm * temporal_weight

    return np.hstack([spatial_features, temporal_features])


def _fuzzy_cmeans(
    features: NDArray[np.float64],
    n_clusters: int,
    fuzziness: float,
    max_iterations: int,
    tolerance: float,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run fuzzy c-means algorithm.

    Returns (membership, centers).
    """
    n_samples, n_features = features.shape
    m = fuzziness
    eps = 1e-10

    # Initialize membership matrix randomly (rows sum to 1)
    u = rng.random((n_samples, n_clusters))
    u = u / u.sum(axis=1, keepdims=True)

    for _ in range(max_iterations):
        # Update centers
        um = u**m
        centers = (um.T @ features) / um.sum(axis=0)[:, np.newaxis]

        # Compute distances
        dist = np.zeros((n_samples, n_clusters))
        for k in range(n_clusters):
            diff = features - centers[k]
            dist[:, k] = np.sqrt(np.sum(diff**2, axis=1))

        # Update membership
        dist = np.maximum(dist, eps)
        power = 2.0 / (m - 1.0)

        u_new = np.zeros_like(u)
        for k in range(n_clusters):
            denom = np.zeros(n_samples)
            for j in range(n_clusters):
                denom += (dist[:, k] / dist[:, j]) ** power
            u_new[:, k] = 1.0 / denom

        # Check convergence
        if np.max(np.abs(u_new - u)) < tolerance:
            u = u_new
            break

        u = u_new

    return u, centers


def fuzzy_cmeans_cluster(
    well_locations: dict[str, tuple[float, float]],
    well_timeseries: dict[str, SMPTimeSeries],
    config: ClusteringConfig | None = None,
) -> ClusteringResult:
    """Cluster observation wells using fuzzy c-means.

    Parameters
    ----------
    well_locations : dict[str, tuple[float, float]]
        Mapping of well ID to (x, y) coordinates.
    well_timeseries : dict[str, SMPTimeSeries]
        Mapping of well ID to water level time series.
    config : ClusteringConfig | None
        Clustering configuration. Uses defaults if ``None``.

    Returns
    -------
    ClusteringResult
        Clustering membership and cluster centers.
    """
    if config is None:
        config = ClusteringConfig()

    # Determine common well set
    well_ids = sorted(set(well_locations.keys()) & set(well_timeseries.keys()))
    n_wells = len(well_ids)

    if n_wells < config.n_clusters:
        raise ValueError(f"Number of wells ({n_wells}) must be >= n_clusters ({config.n_clusters})")

    # Extract features
    features = _extract_features(
        well_locations,
        well_timeseries,
        well_ids,
        config.spatial_weight,
        config.temporal_weight,
    )

    # Run FCM
    rng = np.random.default_rng(config.random_seed)
    membership, centers = _fuzzy_cmeans(
        features,
        config.n_clusters,
        config.fuzziness,
        config.max_iterations,
        config.tolerance,
        rng,
    )

    # Compute Fuzzy Partition Coefficient
    fpc = float(np.mean(membership**2))

    return ClusteringResult(
        membership=membership,
        cluster_centers=centers,
        well_ids=well_ids,
        n_clusters=config.n_clusters,
        fpc=fpc,
    )
