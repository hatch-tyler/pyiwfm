"""
Plotting functionality for IWFM models.

This module re-exports all public plotting functions and classes from the
sub-modules :mod:`~pyiwfm.visualization.plot_mesh`,
:mod:`~pyiwfm.visualization.plot_timeseries`,
:mod:`~pyiwfm.visualization.plot_budget`, and
:mod:`~pyiwfm.visualization.plot_calibration`.

Importing from ``pyiwfm.visualization.plotting`` continues to work exactly
as before -- every name listed in ``__all__`` is available at this level.
"""

from __future__ import annotations

from pyiwfm.visualization.plot_budget import (
    BudgetPlotter,
    plot_budget_bar,
    plot_budget_pie,
    plot_budget_stacked,
    plot_budget_timeseries,
    plot_water_balance,
    plot_zbudget,
)
from pyiwfm.visualization.plot_calibration import (
    plot_cross_section,
    plot_cross_section_location,
    plot_one_to_one,
    plot_residual_cdf,
    plot_spatial_bias,
    plot_streams_colored,
)
from pyiwfm.visualization.plot_mesh import (
    MeshPlotter,
    _subdivide_quads,
    plot_boundary,
    plot_elements,
    plot_lakes,
    plot_mesh,
    plot_nodes,
    plot_scalar_field,
    plot_streams,
)
from pyiwfm.visualization.plot_timeseries import (
    plot_dual_axis,
    plot_streamflow_hydrograph,
    plot_timeseries,
    plot_timeseries_collection,
    plot_timeseries_comparison,
    plot_timeseries_statistics,
)

__all__ = [
    # Mesh / spatial (internal helpers re-exported for backward compat)
    "_subdivide_quads",
    "MeshPlotter",
    "plot_mesh",
    "plot_nodes",
    "plot_elements",
    "plot_scalar_field",
    "plot_streams",
    "plot_lakes",
    "plot_boundary",
    # Time series
    "plot_timeseries",
    "plot_timeseries_comparison",
    "plot_timeseries_collection",
    "plot_timeseries_statistics",
    "plot_dual_axis",
    "plot_streamflow_hydrograph",
    # Budget
    "BudgetPlotter",
    "plot_budget_bar",
    "plot_budget_stacked",
    "plot_budget_pie",
    "plot_water_balance",
    "plot_zbudget",
    "plot_budget_timeseries",
    # Calibration / validation
    "plot_streams_colored",
    "plot_cross_section",
    "plot_cross_section_location",
    "plot_one_to_one",
    "plot_residual_cdf",
    "plot_spatial_bias",
]
