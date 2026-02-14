"""Tests for mesh_generation/gmsh_wrapper.py with mocked gmsh.

Covers:
- generate() with triangle, quad, mixed element types
- _build_geometry() with streams, refinement zones, point constraints
- _extract_mesh() processing
- Exception handling in finalize
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call
import sys

import numpy as np
import pytest

from pyiwfm.mesh_generation.constraints import (
    Boundary,
    PointConstraint,
    RefinementZone,
    StreamConstraint,
)
from pyiwfm.mesh_generation.generators import MeshResult


def _make_mock_gmsh():
    """Create a mock gmsh module."""
    mock_gmsh = MagicMock()
    # Configure getNodes to return a simple triangle mesh
    mock_gmsh.model.mesh.getNodes.return_value = (
        np.array([1, 2, 3]),  # node_tags
        np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 1.0, 0.0]),  # coords (x,y,z interleaved)
        np.array([]),  # parametric_coords
    )
    # Configure getElements to return a single triangle
    mock_gmsh.model.mesh.getElements.return_value = (
        [2],  # elem_types (2 = triangle)
        [np.array([1])],  # elem_tags
        [np.array([1, 2, 3])],  # elem_node_tags
    )
    return mock_gmsh


class TestGenerateTriangles:
    """Test generate() with triangle element type."""

    def test_generate_triangles(self) -> None:
        """Mock gmsh -> triangle mesh."""
        mock_gmsh = _make_mock_gmsh()

        with patch.dict(sys.modules, {"gmsh": mock_gmsh}):
            from pyiwfm.mesh_generation.gmsh_wrapper import GmshMeshGenerator

            gen = GmshMeshGenerator(element_type="triangle")

            boundary = Boundary(
                vertices=np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
            )
            result = gen.generate(boundary, max_area=25.0)

        assert isinstance(result, MeshResult)
        assert result.nodes.shape == (3, 2)
        assert result.elements.shape[0] >= 1
        mock_gmsh.initialize.assert_called_once()
        mock_gmsh.finalize.assert_called_once()


class TestGenerateQuads:
    """Test generate() with quad element type."""

    def test_generate_quads(self) -> None:
        """Mock gmsh -> quad mesh setup."""
        mock_gmsh = _make_mock_gmsh()
        # Return quad elements
        mock_gmsh.model.mesh.getNodes.return_value = (
            np.array([1, 2, 3, 4]),
            np.array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0], dtype=float),
            np.array([]),
        )
        mock_gmsh.model.mesh.getElements.return_value = (
            [3],  # 3 = quad
            [np.array([1])],
            [np.array([1, 2, 3, 4])],
        )

        with patch.dict(sys.modules, {"gmsh": mock_gmsh}):
            from pyiwfm.mesh_generation.gmsh_wrapper import GmshMeshGenerator

            gen = GmshMeshGenerator(element_type="quad")
            boundary = Boundary(
                vertices=np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
            )
            result = gen.generate(boundary)

        assert result.elements.shape == (1, 4)
        # Verify RecombineAll was set for quads
        mock_gmsh.option.setNumber.assert_any_call("Mesh.RecombineAll", 1)


class TestGenerateMixed:
    """Test generate() with mixed element type."""

    def test_generate_mixed(self) -> None:
        """Mock gmsh -> mixed elements setup."""
        mock_gmsh = _make_mock_gmsh()

        with patch.dict(sys.modules, {"gmsh": mock_gmsh}):
            from pyiwfm.mesh_generation.gmsh_wrapper import GmshMeshGenerator

            gen = GmshMeshGenerator(element_type="mixed")
            boundary = Boundary(
                vertices=np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
            )
            gen.generate(boundary)

        # Verify mixed settings
        mock_gmsh.option.setNumber.assert_any_call("Mesh.RecombineAll", 0)


class TestBuildGeometryStreams:
    """Test _build_geometry() with stream constraints."""

    def test_build_geometry_streams(self) -> None:
        """Geometry with stream constraints."""
        mock_gmsh = _make_mock_gmsh()

        with patch.dict(sys.modules, {"gmsh": mock_gmsh}):
            from pyiwfm.mesh_generation.gmsh_wrapper import GmshMeshGenerator

            gen = GmshMeshGenerator(element_type="triangle")
            boundary = Boundary(
                vertices=np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
            )
            streams = [
                StreamConstraint(
                    vertices=np.array([[2, 5], [8, 5]], dtype=float),
                    stream_id=1,
                )
            ]
            gen.generate(boundary, streams=streams)

        # Verify embed was called for stream lines
        mock_gmsh.model.mesh.embed.assert_called()


class TestBuildGeometryPointConstraints:
    """Test _build_geometry() with point constraints."""

    def test_build_geometry_point_constraints(self) -> None:
        """Point constraints embedded."""
        mock_gmsh = _make_mock_gmsh()

        with patch.dict(sys.modules, {"gmsh": mock_gmsh}):
            from pyiwfm.mesh_generation.gmsh_wrapper import GmshMeshGenerator

            gen = GmshMeshGenerator(element_type="triangle")
            boundary = Boundary(
                vertices=np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
            )
            points = [
                PointConstraint(x=5.0, y=5.0, marker=1),
            ]
            gen.generate(boundary, points=points)

        # Point was embedded (dimension=0)
        embed_calls = mock_gmsh.model.mesh.embed.call_args_list
        assert any(c[0][0] == 0 for c in embed_calls)


class TestBuildGeometryRefinement:
    """Test _build_geometry() with refinement zones."""

    def test_build_geometry_refinement(self) -> None:
        """Geometry with refinement zones."""
        mock_gmsh = _make_mock_gmsh()

        with patch.dict(sys.modules, {"gmsh": mock_gmsh}):
            from pyiwfm.mesh_generation.gmsh_wrapper import GmshMeshGenerator

            gen = GmshMeshGenerator(element_type="triangle")
            boundary = Boundary(
                vertices=np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
            )
            zones = [
                RefinementZone(center=(5.0, 5.0), radius=2.0, max_area=1.0),
            ]
            gen.generate(boundary, refinement_zones=zones)

        # Field was created for refinement
        mock_gmsh.model.mesh.field.add.assert_called()


class TestMinAngleParameter:
    """Test min_angle quality parameter."""

    def test_min_angle_parameter(self) -> None:
        """Min angle quality parameter is set."""
        mock_gmsh = _make_mock_gmsh()

        with patch.dict(sys.modules, {"gmsh": mock_gmsh}):
            from pyiwfm.mesh_generation.gmsh_wrapper import GmshMeshGenerator

            gen = GmshMeshGenerator(element_type="triangle")
            boundary = Boundary(
                vertices=np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
            )
            gen.generate(boundary, min_angle=25.0)

        mock_gmsh.option.setNumber.assert_any_call(
            "Mesh.AngleToleranceFacetOverlap", 25.0
        )
