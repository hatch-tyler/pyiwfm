"""
Calibration tools for IWFM models.

This package provides utilities that mirror IWFM's Fortran IWFM2OBS and
CalcTypHyd tools, plus new capabilities for observation well clustering.

Modules
-------
- :mod:`iwfm2obs` — Time interpolation and multi-layer T-weighted averaging
- :mod:`calctyphyd` — Typical hydrograph computation
- :mod:`clustering` — Fuzzy c-means clustering of observation wells
"""

from __future__ import annotations

from pyiwfm.calibration.calctyphyd import (
    CalcTypHydConfig,
    CalcTypHydResult,
    SeasonalPeriod,
    TypicalHydrograph,
    compute_seasonal_averages,
    compute_typical_hydrographs,
    read_cluster_weights,
)
from pyiwfm.calibration.clustering import (
    ClusteringConfig,
    ClusteringResult,
    fuzzy_cmeans_cluster,
)
from pyiwfm.calibration.iwfm2obs import (
    InterpolationConfig,
    MultiLayerWellSpec,
    compute_composite_head,
    compute_multilayer_weights,
    interpolate_batch,
    interpolate_to_obs_times,
    iwfm2obs,
)

__all__ = [
    # iwfm2obs
    "InterpolationConfig",
    "MultiLayerWellSpec",
    "interpolate_to_obs_times",
    "interpolate_batch",
    "compute_multilayer_weights",
    "compute_composite_head",
    "iwfm2obs",
    # calctyphyd
    "SeasonalPeriod",
    "CalcTypHydConfig",
    "TypicalHydrograph",
    "CalcTypHydResult",
    "read_cluster_weights",
    "compute_seasonal_averages",
    "compute_typical_hydrographs",
    # clustering
    "ClusteringConfig",
    "ClusteringResult",
    "fuzzy_cmeans_cluster",
]
