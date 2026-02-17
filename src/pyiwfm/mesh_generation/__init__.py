"""Mesh generation tools for IWFM models."""

from __future__ import annotations

from pyiwfm.mesh_generation.constraints import (
    Boundary,
    BoundarySegment,
    PointConstraint,
    RefinementZone,
    StreamConstraint,
)
from pyiwfm.mesh_generation.generators import (
    MeshGenerator,
    MeshResult,
)

# Optional imports - these may fail if libraries not installed
try:
    from pyiwfm.mesh_generation.triangle_wrapper import TriangleMeshGenerator
except ImportError:
    TriangleMeshGenerator = None  # type: ignore

try:
    from pyiwfm.mesh_generation.gmsh_wrapper import GmshMeshGenerator
except ImportError:
    GmshMeshGenerator = None  # type: ignore

__all__ = [
    # Constraints
    "Boundary",
    "BoundarySegment",
    "StreamConstraint",
    "RefinementZone",
    "PointConstraint",
    # Generators
    "MeshGenerator",
    "MeshResult",
    # Wrapper implementations
    "TriangleMeshGenerator",
    "GmshMeshGenerator",
]
