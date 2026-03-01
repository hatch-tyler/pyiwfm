"""Model comparison tools for IWFM models."""

from __future__ import annotations

from pyiwfm.comparison.differ import (
    DiffItem,
    DiffType,
    MeshDiff,
    ModelDiff,
    ModelDiffer,
    StratigraphyDiff,
)
from pyiwfm.comparison.metrics import (
    ComparisonMetrics,
    SpatialComparison,
    TimeSeriesComparison,
    correlation_coefficient,
    index_of_agreement,
    mae,
    max_error,
    mbe,
    nash_sutcliffe,
    percent_bias,
    relative_error,
    rmse,
    scaled_rmse,
)
from pyiwfm.comparison.report import (
    ComparisonReport,
    HtmlReport,
    JsonReport,
    ReportGenerator,
    TextReport,
)
from pyiwfm.comparison.results_differ import (
    ResultsComparison,
    ResultsDiffer,
)

__all__ = [
    # Differ
    "DiffItem",
    "DiffType",
    "MeshDiff",
    "ModelDiff",
    "ModelDiffer",
    "StratigraphyDiff",
    # Metrics
    "ComparisonMetrics",
    "SpatialComparison",
    "TimeSeriesComparison",
    "correlation_coefficient",
    "index_of_agreement",
    "mae",
    "max_error",
    "mbe",
    "nash_sutcliffe",
    "percent_bias",
    "relative_error",
    "rmse",
    "scaled_rmse",
    # Report
    "ComparisonReport",
    "HtmlReport",
    "JsonReport",
    "ReportGenerator",
    "TextReport",
    # Results differ
    "ResultsDiffer",
    "ResultsComparison",
]
