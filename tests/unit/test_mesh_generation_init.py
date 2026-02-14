"""Tests for mesh_generation/__init__.py import fallback branches.

Covers the except ImportError branches for:
- TriangleMeshGenerator (lines 20-21)
- GmshMeshGenerator (lines 25-26)
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import pytest


class TestTriangleImportFallback:
    """Test Triangle wrapper import fallback."""

    def test_triangle_import_fallback(self) -> None:
        """Force ImportError for triangle_wrapper -> TriangleMeshGenerator is None."""
        blocked = {"pyiwfm.mesh_generation.triangle_wrapper": None}
        with patch.dict(sys.modules, blocked):
            sys.modules.pop("pyiwfm.mesh_generation", None)
            import pyiwfm.mesh_generation as mg
            importlib.reload(mg)
            assert mg.TriangleMeshGenerator is None

        sys.modules.pop("pyiwfm.mesh_generation", None)


class TestGmshImportFallback:
    """Test Gmsh wrapper import fallback."""

    def test_gmsh_import_fallback(self) -> None:
        """Force ImportError for gmsh_wrapper -> GmshMeshGenerator is None."""
        blocked = {"pyiwfm.mesh_generation.gmsh_wrapper": None}
        with patch.dict(sys.modules, blocked):
            sys.modules.pop("pyiwfm.mesh_generation", None)
            import pyiwfm.mesh_generation as mg
            importlib.reload(mg)
            assert mg.GmshMeshGenerator is None

        sys.modules.pop("pyiwfm.mesh_generation", None)
