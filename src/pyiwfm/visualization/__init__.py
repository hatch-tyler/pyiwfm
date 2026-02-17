"""
Visualization tools for IWFM models.

This package provides visualization tools for IWFM model data including:

- **Mesh plotting**: Finite element mesh visualization with matplotlib
- **GIS export**: Export to GeoPackage, Shapefile, GeoJSON formats
- **VTK export**: 3D visualization for ParaView
- **Web viewer**: Interactive web viewer with FastAPI backend and vtk.js/deck.gl frontend
- **Time series charts**: Line charts for temporal data
- **Budget charts**: Bar, stacked area, and pie charts for water budgets
"""

from __future__ import annotations

# Optional imports - these may fail if libraries not installed
try:
    from pyiwfm.visualization.gis_export import GISExporter
except ImportError:
    GISExporter = None  # type: ignore

try:
    from pyiwfm.visualization.vtk_export import VTKExporter
except ImportError:
    VTKExporter = None  # type: ignore

try:
    from pyiwfm.visualization.plotting import (
        # Budget plotting
        BudgetPlotter,
        # Mesh plotting
        MeshPlotter,
        plot_boundary,
        plot_budget_bar,
        plot_budget_pie,
        plot_budget_stacked,
        plot_budget_timeseries,
        plot_elements,
        plot_mesh,
        plot_nodes,
        plot_scalar_field,
        plot_streams,
        # Time series plotting
        plot_timeseries,
        plot_timeseries_collection,
        plot_timeseries_comparison,
        plot_water_balance,
        plot_zbudget,
    )
except ImportError:
    # Mesh plotting
    MeshPlotter = None  # type: ignore
    plot_mesh = None  # type: ignore
    plot_nodes = None  # type: ignore
    plot_elements = None  # type: ignore
    plot_scalar_field = None  # type: ignore
    plot_streams = None  # type: ignore
    plot_boundary = None  # type: ignore
    # Time series plotting
    plot_timeseries = None  # type: ignore
    plot_timeseries_comparison = None  # type: ignore
    plot_timeseries_collection = None  # type: ignore
    # Budget plotting
    BudgetPlotter = None  # type: ignore
    plot_budget_bar = None  # type: ignore
    plot_budget_stacked = None  # type: ignore
    plot_budget_pie = None  # type: ignore
    plot_water_balance = None  # type: ignore
    plot_zbudget = None  # type: ignore
    plot_budget_timeseries = None  # type: ignore

__all__ = [
    # Export classes
    "GISExporter",
    "VTKExporter",
    # Mesh plotting
    "MeshPlotter",
    "plot_mesh",
    "plot_nodes",
    "plot_elements",
    "plot_scalar_field",
    "plot_streams",
    "plot_boundary",
    # Time series plotting
    "plot_timeseries",
    "plot_timeseries_comparison",
    "plot_timeseries_collection",
    # Budget plotting
    "BudgetPlotter",
    "plot_budget_bar",
    "plot_budget_stacked",
    "plot_budget_pie",
    "plot_water_balance",
    "plot_zbudget",
    "plot_budget_timeseries",
]
