"""Tests for package __init__ modules to ensure imports work correctly.

Tests:
- utils/__init__.py
- visualization/__init__.py
- mesh_generation/__init__.py
"""

from __future__ import annotations


class TestUtilsInit:
    """Tests for utils package init."""

    def test_import_utils(self) -> None:
        """Test utils package can be imported."""
        import pyiwfm.utils
        assert hasattr(pyiwfm.utils, "__all__")

    def test_all_is_empty_list(self) -> None:
        """Test __all__ is an empty list."""
        from pyiwfm.utils import __all__
        assert __all__ == []


class TestVisualizationInit:
    """Tests for visualization package init."""

    def test_import_visualization(self) -> None:
        """Test visualization package can be imported."""
        import pyiwfm.visualization
        assert hasattr(pyiwfm.visualization, "__all__")

    def test_all_contains_expected_names(self) -> None:
        """Test __all__ contains expected exports."""
        from pyiwfm.visualization import __all__
        assert "GISExporter" in __all__
        assert "VTKExporter" in __all__
        assert "MeshPlotter" in __all__
        assert "plot_mesh" in __all__
        assert "plot_timeseries" in __all__
        assert "BudgetPlotter" in __all__

    def test_gis_exporter_availability(self) -> None:
        """Test GISExporter is either a class or None."""
        from pyiwfm.visualization import GISExporter
        # Should be the class or None if geopandas not available
        assert GISExporter is None or callable(GISExporter)

    def test_vtk_exporter_availability(self) -> None:
        """Test VTKExporter is either a class or None."""
        from pyiwfm.visualization import VTKExporter
        assert VTKExporter is None or callable(VTKExporter)

    def test_plotting_availability(self) -> None:
        """Test plotting exports are either valid or None."""
        from pyiwfm.visualization import MeshPlotter, plot_mesh
        # Should be the class/function or None if matplotlib not available
        assert MeshPlotter is None or callable(MeshPlotter)
        assert plot_mesh is None or callable(plot_mesh)

    def test_budget_plotter_availability(self) -> None:
        """Test budget plotter availability."""
        from pyiwfm.visualization import BudgetPlotter, plot_budget_bar
        assert BudgetPlotter is None or callable(BudgetPlotter)
        assert plot_budget_bar is None or callable(plot_budget_bar)

    def test_all_exports_accessible(self) -> None:
        """Test all names in __all__ can be accessed."""
        import pyiwfm.visualization as viz
        for name in viz.__all__:
            attr = getattr(viz, name, "MISSING")
            assert attr != "MISSING", f"{name} not accessible"


class TestMeshGenerationInit:
    """Tests for mesh_generation package init."""

    def test_import_mesh_generation(self) -> None:
        """Test mesh_generation package can be imported."""
        import pyiwfm.mesh_generation
        assert hasattr(pyiwfm.mesh_generation, "__all__")

    def test_constraint_classes_available(self) -> None:
        """Test constraint classes are always importable."""
        from pyiwfm.mesh_generation import (
            Boundary,
            BoundarySegment,
            PointConstraint,
            RefinementZone,
            StreamConstraint,
        )
        assert Boundary is not None
        assert BoundarySegment is not None
        assert StreamConstraint is not None
        assert RefinementZone is not None
        assert PointConstraint is not None

    def test_generator_classes_available(self) -> None:
        """Test generator base classes are always importable."""
        from pyiwfm.mesh_generation import MeshGenerator, MeshResult
        assert MeshGenerator is not None
        assert MeshResult is not None

    def test_optional_generators(self) -> None:
        """Test optional generator wrappers are either valid or None."""
        from pyiwfm.mesh_generation import GmshMeshGenerator, TriangleMeshGenerator
        assert TriangleMeshGenerator is None or callable(TriangleMeshGenerator)
        assert GmshMeshGenerator is None or callable(GmshMeshGenerator)

    def test_all_contains_expected(self) -> None:
        """Test __all__ contains all expected names."""
        from pyiwfm.mesh_generation import __all__
        expected = [
            "Boundary", "BoundarySegment", "StreamConstraint",
            "RefinementZone", "PointConstraint",
            "MeshGenerator", "MeshResult",
            "TriangleMeshGenerator", "GmshMeshGenerator",
        ]
        for name in expected:
            assert name in __all__, f"{name} missing from __all__"
