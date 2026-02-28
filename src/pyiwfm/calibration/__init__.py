"""
Calibration tools for IWFM models.

This package provides utilities that mirror IWFM's Fortran IWFM2OBS and
CalcTypHyd tools, plus new capabilities for observation well clustering.

Modules
-------
- :mod:`iwfm2obs` — Time interpolation and multi-layer T-weighted averaging
- :mod:`calctyphyd` — Typical hydrograph computation
- :mod:`clustering` — Fuzzy c-means clustering of observation wells
- :mod:`model_file_discovery` — Discover .out files from IWFM simulation main
- :mod:`obs_well_spec` — Observation well specification reader
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
    IWFM2OBSConfig,
    MultiLayerWellSpec,
    compute_composite_head,
    compute_multilayer_weights,
    interpolate_batch,
    interpolate_to_obs_times,
    iwfm2obs,
    iwfm2obs_from_model,
    write_multilayer_output,
    write_multilayer_pest_ins,
)
from pyiwfm.calibration.model_file_discovery import (
    HydrographFileInfo,
    discover_hydrograph_files,
)
from pyiwfm.calibration.obs_well_spec import (
    ObsWellSpec,
    read_obs_well_spec,
)

__all__ = [
    # iwfm2obs
    "InterpolationConfig",
    "IWFM2OBSConfig",
    "MultiLayerWellSpec",
    "interpolate_to_obs_times",
    "interpolate_batch",
    "compute_multilayer_weights",
    "compute_composite_head",
    "iwfm2obs",
    "iwfm2obs_from_model",
    "write_multilayer_output",
    "write_multilayer_pest_ins",
    # model_file_discovery
    "HydrographFileInfo",
    "discover_hydrograph_files",
    # obs_well_spec
    "ObsWellSpec",
    "read_obs_well_spec",
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
