"""
Triangle mesh generator wrapper.

This module wraps the Triangle library to provide mesh generation
capabilities for IWFM models using triangular elements.

Triangle is a high-quality mesh generator and Delaunay triangulator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pyiwfm.mesh_generation.generators import MeshGenerator, MeshResult

if TYPE_CHECKING:
    from pyiwfm.mesh_generation.constraints import (
        Boundary,
        PointConstraint,
        RefinementZone,
        StreamConstraint,
    )


class TriangleMeshGenerator(MeshGenerator):
    """
    Mesh generator using the Triangle library.

    This generator creates triangular meshes using Jonathan Shewchuk's
    Triangle library via the triangle Python package.

    Features:
        - Quality mesh generation with angle constraints
        - Area constraints (global and regional)
        - Conforming Delaunay triangulation
        - Support for holes and internal boundaries
    """

    def __init__(self) -> None:
        """Initialize the Triangle mesh generator."""
        # Import triangle here to fail early if not installed
        try:
            import triangle  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "Triangle library is required for TriangleMeshGenerator. "
                "Install with: pip install triangle"
            ) from e

    def generate(
        self,
        boundary: Boundary,
        max_area: float | None = None,
        min_angle: float | None = None,
        streams: list[StreamConstraint] | None = None,
        refinement_zones: list[RefinementZone] | None = None,
        points: list[PointConstraint] | None = None,
    ) -> MeshResult:
        """
        Generate a triangular mesh within the boundary.

        Args:
            boundary: Boundary polygon (with optional holes)
            max_area: Maximum element area
            min_angle: Minimum angle constraint (degrees)
            streams: Stream line constraints
            refinement_zones: Zones requiring smaller elements
            points: Fixed point constraints

        Returns:
            MeshResult with generated triangular mesh
        """
        import triangle

        # Build input for Triangle
        tri_input = self._build_triangle_input(boundary, streams, points)

        # Build Triangle options string
        opts = self._build_options(max_area, min_angle)

        # Generate mesh
        tri_output = triangle.triangulate(tri_input, opts)

        # Apply refinement zones if specified
        if refinement_zones:
            tri_output = self._apply_refinement(tri_output, refinement_zones, min_angle)

        # Convert to MeshResult
        return self._convert_output(tri_output)

    def _build_triangle_input(
        self,
        boundary: Boundary,
        streams: list[StreamConstraint] | None = None,
        points: list[PointConstraint] | None = None,
    ) -> dict:
        """Build input dictionary for Triangle."""
        # Start with boundary vertices
        vertices = list(boundary.vertices)
        n_boundary = len(vertices)

        # Build segments for boundary
        segments = []
        for i in range(n_boundary):
            segments.append([i, (i + 1) % n_boundary])

        # Add hole vertices and segments
        hole_points = []
        vertex_offset = n_boundary

        for hole in boundary.holes:
            n_hole = len(hole)
            vertices.extend(hole)

            for i in range(n_hole):
                segments.append([vertex_offset + i, vertex_offset + (i + 1) % n_hole])

            # Add interior point for hole
            hole_points.append(boundary.get_hole_points()[-1])
            vertex_offset += n_hole

        # Add stream vertices and segments
        if streams:
            for stream in streams:
                n_stream = len(stream.vertices)
                vertices.extend(stream.vertices)

                for i in range(n_stream - 1):
                    segments.append([vertex_offset + i, vertex_offset + i + 1])

                vertex_offset += n_stream

        # Add fixed point constraints
        if points:
            for point in points:
                vertices.append([point.x, point.y])

        # Build input dictionary
        tri_input = {
            "vertices": np.array(vertices, dtype=np.float64),
            "segments": np.array(segments, dtype=np.int32) if segments else None,
        }

        if hole_points:
            tri_input["holes"] = np.array(hole_points, dtype=np.float64)

        return tri_input

    def _build_options(
        self,
        max_area: float | None = None,
        min_angle: float | None = None,
    ) -> str:
        """Build Triangle options string."""
        opts = "p"  # Planar Straight Line Graph

        if min_angle is not None:
            opts += f"q{min_angle}"  # Quality with angle
        else:
            opts += "q"  # Default quality

        if max_area is not None:
            opts += f"a{max_area}"  # Maximum area

        # Additional options
        opts += "D"  # Conforming Delaunay

        return opts

    def _apply_refinement(
        self,
        tri_output: dict,
        refinement_zones: list[RefinementZone],
        min_angle: float | None = None,
    ) -> dict:
        """Apply local refinement to mesh."""
        import triangle

        # Calculate element areas and centroids
        vertices = tri_output["vertices"]
        triangles = tri_output["triangles"]

        # Find elements that need refinement
        needs_refinement = np.zeros(len(triangles), dtype=bool)

        for i, tri in enumerate(triangles):
            # Calculate centroid
            cx = np.mean(vertices[tri, 0])
            cy = np.mean(vertices[tri, 1])

            # Calculate area
            v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
            area = 0.5 * abs((v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1]))

            # Check each refinement zone
            for zone in refinement_zones:
                if zone.contains(cx, cy) and area > zone.max_area:
                    needs_refinement[i] = True
                    break

        # If refinement needed, re-triangulate with area constraints
        if np.any(needs_refinement):
            # Create area constraints array
            triangle_areas = np.full(len(triangles), float("inf"))

            for i, tri in enumerate(triangles):
                cx = np.mean(vertices[tri, 0])
                cy = np.mean(vertices[tri, 1])

                for zone in refinement_zones:
                    if zone.contains(cx, cy):
                        triangle_areas[i] = min(triangle_areas[i], zone.max_area)
                        break

            # Build refine input
            refine_input = {
                "vertices": tri_output["vertices"],
                "triangles": tri_output["triangles"],
                "triangle_max_area": triangle_areas,
            }

            if "segments" in tri_output:
                refine_input["segments"] = tri_output["segments"]

            # Refine options
            opts = "rpq"
            if min_angle is not None:
                opts = f"rpq{min_angle}"

            tri_output = triangle.triangulate(refine_input, opts)

        return tri_output

    def _convert_output(self, tri_output: dict) -> MeshResult:
        """Convert Triangle output to MeshResult."""
        nodes = tri_output["vertices"]
        triangles = tri_output["triangles"]

        # Pad triangles to 4 columns with -1 for 4th vertex
        n_elements = len(triangles)
        elements = np.full((n_elements, 4), -1, dtype=np.int32)
        elements[:, :3] = triangles

        # Get markers if available
        node_markers = None
        if "vertex_markers" in tri_output:
            node_markers = tri_output["vertex_markers"].flatten().astype(np.int32)

        element_markers = None
        if "triangle_attributes" in tri_output:
            element_markers = tri_output["triangle_attributes"][:, 0].astype(np.int32)

        return MeshResult(
            nodes=nodes,
            elements=elements,
            node_markers=node_markers,
            element_markers=element_markers,
        )
