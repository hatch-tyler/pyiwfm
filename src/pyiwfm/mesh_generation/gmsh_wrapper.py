"""
Gmsh mesh generator wrapper.

This module wraps the Gmsh library to provide mesh generation
capabilities for IWFM models supporting triangular, quadrilateral,
and mixed element meshes.

Gmsh is a powerful open-source mesh generator with CAD capabilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal
import math

import numpy as np
from numpy.typing import NDArray

from pyiwfm.mesh_generation.generators import MeshGenerator, MeshResult

if TYPE_CHECKING:
    from pyiwfm.mesh_generation.constraints import (
        Boundary,
        StreamConstraint,
        RefinementZone,
        PointConstraint,
    )


class GmshMeshGenerator(MeshGenerator):
    """
    Mesh generator using the Gmsh library.

    This generator creates meshes using Gmsh, supporting
    triangular, quadrilateral, and mixed element types.

    Attributes:
        element_type: Type of elements to generate
            - 'triangle': Only triangular elements
            - 'quad': Quadrilateral elements (recombined)
            - 'mixed': Mix of triangles and quads
    """

    def __init__(
        self,
        element_type: Literal["triangle", "quad", "mixed"] = "triangle",
    ) -> None:
        """
        Initialize the Gmsh mesh generator.

        Args:
            element_type: Type of elements to generate
        """
        # Import gmsh here to fail early if not installed
        try:
            import gmsh  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "Gmsh library is required for GmshMeshGenerator. "
                "Install with: pip install gmsh"
            ) from e

        self.element_type = element_type

    def generate(
        self,
        boundary: "Boundary",
        max_area: float | None = None,
        min_angle: float | None = None,
        streams: list["StreamConstraint"] | None = None,
        refinement_zones: list["RefinementZone"] | None = None,
        points: list["PointConstraint"] | None = None,
    ) -> MeshResult:
        """
        Generate a mesh within the boundary.

        Args:
            boundary: Boundary polygon (with optional holes)
            max_area: Maximum element area
            min_angle: Minimum angle constraint (degrees)
            streams: Stream line constraints
            refinement_zones: Zones requiring smaller elements
            points: Fixed point constraints

        Returns:
            MeshResult with generated mesh
        """
        import gmsh

        # Initialize Gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)  # Suppress output

        try:
            # Create new model
            gmsh.model.add("mesh")

            # Build geometry
            self._build_geometry(
                boundary, streams, points, refinement_zones
            )

            # Set mesh size
            if max_area is not None:
                # Convert area to characteristic length
                char_length = math.sqrt(max_area)
                gmsh.option.setNumber("Mesh.CharacteristicLengthMax", char_length)

            # Set element type options
            if self.element_type == "quad":
                gmsh.option.setNumber("Mesh.RecombineAll", 1)
                gmsh.option.setNumber("Mesh.Algorithm", 8)  # Frontal-Delaunay for quads
            elif self.element_type == "mixed":
                gmsh.option.setNumber("Mesh.RecombineAll", 0)
                gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)  # Blossom

            # Set minimum angle if specified
            if min_angle is not None:
                # Gmsh uses different angle measure
                gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", min_angle)

            # Generate 2D mesh
            gmsh.model.geo.synchronize()
            gmsh.model.mesh.generate(2)

            # Extract mesh data
            result = self._extract_mesh()

            return result

        finally:
            gmsh.finalize()

    def _build_geometry(
        self,
        boundary: "Boundary",
        streams: list["StreamConstraint"] | None = None,
        points: list["PointConstraint"] | None = None,
        refinement_zones: list["RefinementZone"] | None = None,
    ) -> None:
        """Build Gmsh geometry from constraints."""
        import gmsh

        # Create boundary points
        boundary_point_ids = []
        for i, (x, y) in enumerate(boundary.vertices):
            pt_id = gmsh.model.geo.addPoint(x, y, 0)
            boundary_point_ids.append(pt_id)

        # Create boundary lines
        boundary_line_ids = []
        n = len(boundary_point_ids)
        for i in range(n):
            line_id = gmsh.model.geo.addLine(
                boundary_point_ids[i],
                boundary_point_ids[(i + 1) % n]
            )
            boundary_line_ids.append(line_id)

        # Create boundary curve loop
        boundary_loop = gmsh.model.geo.addCurveLoop(boundary_line_ids)

        # Handle holes
        hole_loops = []
        for hole in boundary.holes:
            hole_point_ids = []
            for x, y in hole:
                pt_id = gmsh.model.geo.addPoint(x, y, 0)
                hole_point_ids.append(pt_id)

            hole_line_ids = []
            n_hole = len(hole_point_ids)
            for i in range(n_hole):
                line_id = gmsh.model.geo.addLine(
                    hole_point_ids[i],
                    hole_point_ids[(i + 1) % n_hole]
                )
                hole_line_ids.append(line_id)

            hole_loop = gmsh.model.geo.addCurveLoop(hole_line_ids)
            hole_loops.append(hole_loop)

        # Create surface (with holes)
        if hole_loops:
            gmsh.model.geo.addPlaneSurface([boundary_loop] + hole_loops)
        else:
            gmsh.model.geo.addPlaneSurface([boundary_loop])

        # Add stream constraints as embedded curves
        if streams:
            for stream in streams:
                stream_point_ids = []
                for x, y in stream.vertices:
                    pt_id = gmsh.model.geo.addPoint(x, y, 0)
                    stream_point_ids.append(pt_id)

                stream_line_ids = []
                for i in range(len(stream_point_ids) - 1):
                    line_id = gmsh.model.geo.addLine(
                        stream_point_ids[i],
                        stream_point_ids[i + 1]
                    )
                    stream_line_ids.append(line_id)

                # Embed in surface
                gmsh.model.geo.synchronize()
                for line_id in stream_line_ids:
                    gmsh.model.mesh.embed(1, [line_id], 2, 1)

        # Add fixed point constraints
        if points:
            for point in points:
                pt_id = gmsh.model.geo.addPoint(point.x, point.y, 0)
                gmsh.model.geo.synchronize()
                gmsh.model.mesh.embed(0, [pt_id], 2, 1)

        # Apply refinement zones using mesh size fields
        if refinement_zones:
            self._apply_refinement_fields(refinement_zones)

    def _apply_refinement_fields(
        self,
        refinement_zones: list["RefinementZone"],
    ) -> None:
        """Apply refinement zones using Gmsh mesh size fields."""
        import gmsh

        field_ids = []

        for i, zone in enumerate(refinement_zones):
            char_length = math.sqrt(zone.max_area)

            if zone.center is not None and zone.radius is not None:
                # Circular refinement zone using Ball field
                field_id = gmsh.model.mesh.field.add("Ball")
                gmsh.model.mesh.field.setNumber(field_id, "XCenter", zone.center[0])
                gmsh.model.mesh.field.setNumber(field_id, "YCenter", zone.center[1])
                gmsh.model.mesh.field.setNumber(field_id, "ZCenter", 0)
                gmsh.model.mesh.field.setNumber(field_id, "Radius", zone.radius)
                gmsh.model.mesh.field.setNumber(field_id, "VIn", char_length)
                gmsh.model.mesh.field.setNumber(field_id, "VOut", char_length * 3)  # Coarser outside
                field_ids.append(field_id)

        if field_ids:
            # Combine fields using Min
            min_field = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", field_ids)
            gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

    def _extract_mesh(self) -> MeshResult:
        """Extract mesh data from Gmsh."""
        import gmsh

        # Get nodes
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()

        # Build node array (x, y only, ignore z)
        n_nodes = len(node_tags)
        nodes = np.zeros((n_nodes, 2))

        # Create mapping from Gmsh node tag to array index
        tag_to_idx = {}
        for i, tag in enumerate(node_tags):
            tag_to_idx[int(tag)] = i
            # Coordinates are interleaved: x, y, z, x, y, z, ...
            nodes[i, 0] = node_coords[i * 3]
            nodes[i, 1] = node_coords[i * 3 + 1]

        # Get elements
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2)

        # Process elements
        elements_list = []

        for etype, etags, enodes in zip(elem_types, elem_tags, elem_node_tags):
            if etype == 2:  # Triangle
                # 3 nodes per triangle
                n_tri = len(etags)
                for i in range(n_tri):
                    tri_nodes = [
                        tag_to_idx[int(enodes[i * 3])],
                        tag_to_idx[int(enodes[i * 3 + 1])],
                        tag_to_idx[int(enodes[i * 3 + 2])],
                        -1,  # Padding for 4th vertex
                    ]
                    elements_list.append(tri_nodes)

            elif etype == 3:  # Quad
                # 4 nodes per quad
                n_quad = len(etags)
                for i in range(n_quad):
                    quad_nodes = [
                        tag_to_idx[int(enodes[i * 4])],
                        tag_to_idx[int(enodes[i * 4 + 1])],
                        tag_to_idx[int(enodes[i * 4 + 2])],
                        tag_to_idx[int(enodes[i * 4 + 3])],
                    ]
                    elements_list.append(quad_nodes)

        elements = np.array(elements_list, dtype=np.int32)

        return MeshResult(nodes=nodes, elements=elements)
