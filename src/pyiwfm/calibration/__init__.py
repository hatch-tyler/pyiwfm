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
- :mod:`observation_matching` — Sim vs obs matching and fit statistics
- :mod:`headall_extraction` — HeadAll HDF5 extraction pipeline
- :mod:`head_differences` — Head difference computation for paired wells
"""

from __future__ import annotations

from pyiwfm.calibration.calctyphyd import (
    BIANNUAL_SEASONS,
    CalcTypHydConfig,
    CalcTypHydResult,
    SeasonalPeriod,
    TypicalHydrograph,
    compute_obs_type_hydrographs,
    compute_seasonal_averages,
    compute_typical_hydrographs,
    read_cluster_weights,
)
from pyiwfm.calibration.clustering import (
    ClusteringConfig,
    ClusteringResult,
    fuzzy_cmeans_cluster,
)
from pyiwfm.calibration.head_differences import (
    HeadDiffPair,
    compute_head_differences,
    read_pairs_file,
)
from pyiwfm.calibration.headall_extraction import (
    HeadAllExtractor,
    WellSpec,
)
from pyiwfm.calibration.iwfm2obs import (
    InterpolationConfig,
    IWFM2OBSConfig,
    MultiLayerWellSpec,
    average_to_seasonal,
    compute_composite_continuous,
    compute_composite_head,
    compute_composite_subsidence,
    compute_multilayer_weights,
    interpolate_batch,
    interpolate_to_obs_times,
    iwfm2obs,
    iwfm2obs_from_model,
    write_multilayer_output,
)
from pyiwfm.calibration.model_file_discovery import (
    HydrographFileInfo,
    discover_hydrograph_files,
)
from pyiwfm.calibration.obs_well_spec import (
    ObsWellSpec,
    read_obs_well_spec,
)
from pyiwfm.calibration.observation_matching import (
    FitStatistics,
    MatchResult,
    ObservationMatcher,
)
from pyiwfm.calibration.pest_files import (
    generate_multilayer_ins,
    generate_smp_ins,
    generate_thyd_ins,
)
from pyiwfm.calibration.report import (
    CalibrationReportConfig,
    generate_calibration_report,
)
from pyiwfm.calibration.residuals import (
    WellScreenType,
    compute_residuals,
    export_residual_table,
    filter_residuals,
    max_residuals,
    mean_residuals,
    residual_summary,
)
from pyiwfm.calibration.results_extraction import (
    ExtractionResult,
    ExtractionSpec,
    ResultsExtractor,
)
from pyiwfm.calibration.texture2par import (
    MixingConfig,
    PilotPointSet,
    Texture2Par,
    WellLog,
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
    "compute_composite_subsidence",
    "iwfm2obs",
    "iwfm2obs_from_model",
    "write_multilayer_output",
    "compute_composite_continuous",
    "average_to_seasonal",
    # model_file_discovery
    "HydrographFileInfo",
    "discover_hydrograph_files",
    # obs_well_spec
    "ObsWellSpec",
    "read_obs_well_spec",
    # calctyphyd
    "BIANNUAL_SEASONS",
    "SeasonalPeriod",
    "CalcTypHydConfig",
    "TypicalHydrograph",
    "CalcTypHydResult",
    "read_cluster_weights",
    "compute_seasonal_averages",
    "compute_typical_hydrographs",
    "compute_obs_type_hydrographs",
    # clustering
    "ClusteringConfig",
    "ClusteringResult",
    "fuzzy_cmeans_cluster",
    # report
    "CalibrationReportConfig",
    "generate_calibration_report",
    # residuals
    "WellScreenType",
    "compute_residuals",
    "mean_residuals",
    "max_residuals",
    "filter_residuals",
    "residual_summary",
    "export_residual_table",
    # observation_matching
    "ObservationMatcher",
    "FitStatistics",
    "MatchResult",
    # headall_extraction
    "HeadAllExtractor",
    "WellSpec",
    # head_differences
    "HeadDiffPair",
    "compute_head_differences",
    "read_pairs_file",
    # pest_files
    "generate_smp_ins",
    "generate_thyd_ins",
    "generate_multilayer_ins",
    # results_extraction
    "ResultsExtractor",
    "ExtractionSpec",
    "ExtractionResult",
    # texture2par
    "Texture2Par",
    "MixingConfig",
    "PilotPointSet",
    "WellLog",
]
