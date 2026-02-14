"""Tests for visualization/__init__.py import fallback branches.

Covers the except ImportError branches (lines 19-20, 24-25, 70-90)
by mocking sys.modules to force ImportError on each optional dependency group.
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import pytest


class TestGISExporterImportFallback:
    """Test GIS exporter import fallback."""

    def test_gis_exporter_import_fallback(self) -> None:
        """Force ImportError for gis_export -> GISExporter is None."""
        # Block the gis_export submodule
        blocked = {"pyiwfm.visualization.gis_export": None}
        with patch.dict(sys.modules, blocked):
            # Remove cached module
            sys.modules.pop("pyiwfm.visualization", None)
            import pyiwfm.visualization as viz
            importlib.reload(viz)
            assert viz.GISExporter is None

        # Restore
        sys.modules.pop("pyiwfm.visualization", None)


class TestVTKExporterImportFallback:
    """Test VTK exporter import fallback."""

    def test_vtk_exporter_import_fallback(self) -> None:
        """Force ImportError for vtk_export -> VTKExporter is None."""
        blocked = {"pyiwfm.visualization.vtk_export": None}
        with patch.dict(sys.modules, blocked):
            sys.modules.pop("pyiwfm.visualization", None)
            import pyiwfm.visualization as viz
            importlib.reload(viz)
            assert viz.VTKExporter is None

        sys.modules.pop("pyiwfm.visualization", None)


class TestPlottingImportFallback:
    """Test plotting import fallback."""

    def test_plotting_import_fallback(self) -> None:
        """Force ImportError for plotting -> all plot functions are None."""
        blocked = {"pyiwfm.visualization.plotting": None}
        with patch.dict(sys.modules, blocked):
            sys.modules.pop("pyiwfm.visualization", None)
            import pyiwfm.visualization as viz
            importlib.reload(viz)
            # Mesh plotting
            assert viz.MeshPlotter is None
            assert viz.plot_mesh is None
            assert viz.plot_nodes is None
            assert viz.plot_elements is None
            assert viz.plot_scalar_field is None
            assert viz.plot_streams is None
            assert viz.plot_boundary is None
            # Time series
            assert viz.plot_timeseries is None
            assert viz.plot_timeseries_comparison is None
            assert viz.plot_timeseries_collection is None
            # Budget
            assert viz.BudgetPlotter is None
            assert viz.plot_budget_bar is None
            assert viz.plot_budget_stacked is None
            assert viz.plot_budget_pie is None
            assert viz.plot_water_balance is None
            assert viz.plot_zbudget is None
            assert viz.plot_budget_timeseries is None

        sys.modules.pop("pyiwfm.visualization", None)


class TestVisualizationPublicAPI:
    """Tests for the public API exports in visualization/__init__."""

    def test_all_exports_defined(self) -> None:
        import pyiwfm.visualization as viz

        assert hasattr(viz, "__all__")
        expected_names = [
            "GISExporter",
            "VTKExporter",
            "MeshPlotter",
            "plot_mesh",
            "plot_nodes",
            "plot_elements",
            "plot_scalar_field",
            "plot_streams",
            "plot_boundary",
            "plot_timeseries",
            "plot_timeseries_comparison",
            "plot_timeseries_collection",
            "BudgetPlotter",
            "plot_budget_bar",
            "plot_budget_stacked",
            "plot_budget_pie",
            "plot_water_balance",
            "plot_zbudget",
            "plot_budget_timeseries",
        ]
        for name in expected_names:
            assert name in viz.__all__

    def test_gis_exporter_attribute_exists(self) -> None:
        import pyiwfm.visualization as viz

        assert hasattr(viz, "GISExporter")

    def test_vtk_exporter_attribute_exists(self) -> None:
        import pyiwfm.visualization as viz

        assert hasattr(viz, "VTKExporter")


class TestWebAPIPublicAPI:
    """Tests for visualization/webapi/__init__.py exports."""

    def test_exports_create_app(self) -> None:
        pytest.importorskip("fastapi")
        from pyiwfm.visualization.webapi import create_app

        assert callable(create_app)

    def test_exports_launch_viewer(self) -> None:
        pytest.importorskip("fastapi")
        from pyiwfm.visualization.webapi import launch_viewer

        assert callable(launch_viewer)

    def test_all_exports(self) -> None:
        pytest.importorskip("fastapi")
        from pyiwfm.visualization import webapi

        assert "create_app" in webapi.__all__
        assert "launch_viewer" in webapi.__all__
