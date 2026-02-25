"""
Gmsh mesh generator wrapper.

This module wraps the Gmsh library to provide mesh generation
capabilities for IWFM models supporting triangular, quadrilateral,
and mixed element meshes.

Gmsh is a powerful open-source mesh generator with CAD capabilities.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import numpy as np

from pyiwfm.mesh_generation.generators import MeshGenerator, MeshResult

if TYPE_CHECKING:
    from pyiwfm.mesh_generation.constraints import (
        Boundary,
        PointConstraint,
        RefinementZone,
        StreamConstraint,
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
                "Gmsh library is required for GmshMeshGenerator. Install with: pip install gmsh"
            ) from e

        self.element_type = element_type

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

        # Convert area to characteristic length for mesh sizing
        char_length: float | None = None
        if max_area is not None:
            char_length = math.sqrt(max_area)

        # Initialize Gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)  # Suppress output

        try:
            # Create new model
            gmsh.model.add("mesh")

            # Build geometry (pass char_length so boundary points get mesh size)
            self._build_geometry(boundary, streams, points, refinement_zones, char_length)

            # Set global mesh size cap
            if char_length is not None:
                gmsh.option.setNumber("Mesh.CharacteristicLengthMax", char_length)

            # Set element type options
            if self.element_type == "quad":
                gmsh.option.setNumber("Mesh.RecombineAll", 1)
                gmsh.option.setNumber("Mesh.Algorithm", 8)  # Frontal-Delaunay for quads
            elif self.element_type == "mixed":
                gmsh.option.setNumber("Mesh.RecombineAll", 1)
                gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)  # Blossom
                # Blossom naturally leaves triangles where quads aren't suitable
                gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 0)  # Don't subdivide

            # Set minimum angle if specified
            if min_angle is not None:
                # Gmsh uses different angle measure
                gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", min_angle)

            # Generate 2D mesh
            gmsh.model.geo.synchronize()
            gmsh.model.mesh.generate(2)

            # Merge duplicate nodes (stream endpoints on boundary edges
            # can create duplicate Gmsh points at the same location)
            gmsh.model.mesh.removeDuplicateNodes()

            # Extract mesh data (filters degenerate elements)
            result = self._extract_mesh()

            return result

        finally:
            gmsh.finalize()

    @staticmethod
    def _insert_stream_points_on_boundary(
        boundary_verts: list[tuple[float, float]],
        streams: list[StreamConstraint] | None,
        tol: float = 1e-6,
    ) -> list[tuple[float, float]]:
        """Insert stream endpoints that lie on boundary edges into the vertex list.

        When a stream endpoint falls on a boundary edge (but not at a
        vertex), the boundary edge must be split so the stream connects
        to a proper boundary vertex.  This prevents Gmsh from creating
        large fan-shaped elements around embedded points on edges.
        """
        if not streams:
            return boundary_verts

        # Collect stream endpoints to test
        edge_points: list[tuple[float, float, int]] = []  # (x, y, edge_index)
        n = len(boundary_verts)

        for stream in streams:
            for sx, sy in [
                (float(stream.vertices[0, 0]), float(stream.vertices[0, 1])),
                (float(stream.vertices[-1, 0]), float(stream.vertices[-1, 1])),
            ]:
                # Skip if already a boundary vertex
                if any(abs(sx - bx) < tol and abs(sy - by) < tol for bx, by in boundary_verts):
                    continue

                # Check if on a boundary edge
                for i in range(n):
                    x1, y1 = boundary_verts[i]
                    x2, y2 = boundary_verts[(i + 1) % n]
                    # Parametric position along edge
                    dx, dy = x2 - x1, y2 - y1
                    edge_len_sq = dx * dx + dy * dy
                    if edge_len_sq < tol * tol:
                        continue
                    t = ((sx - x1) * dx + (sy - y1) * dy) / edge_len_sq
                    if t < tol or t > 1.0 - tol:
                        continue
                    # Distance from point to edge
                    px = x1 + t * dx
                    py = y1 + t * dy
                    dist = math.sqrt((sx - px) ** 2 + (sy - py) ** 2)
                    if dist < tol * math.sqrt(edge_len_sq):
                        edge_points.append((sx, sy, i))
                        break

        if not edge_points:
            return boundary_verts

        # Group insertions by edge, sorted by parameter t (ascending)
        edge_inserts: dict[int, list[tuple[float, float, float]]] = {}
        for sx, sy, edge_idx in edge_points:
            x1, y1 = boundary_verts[edge_idx]
            x2, y2 = boundary_verts[(edge_idx + 1) % n]
            dx, dy = x2 - x1, y2 - y1
            t = ((sx - x1) * dx + (sy - y1) * dy) / (dx * dx + dy * dy)
            edge_inserts.setdefault(edge_idx, []).append((t, sx, sy))

        for edge_idx in edge_inserts:
            edge_inserts[edge_idx].sort()

        # Rebuild vertex list with inserted points
        new_verts: list[tuple[float, float]] = []
        for i in range(n):
            new_verts.append(boundary_verts[i])
            if i in edge_inserts:
                for _t, sx, sy in edge_inserts[i]:
                    new_verts.append((sx, sy))

        return new_verts

    def _build_geometry(
        self,
        boundary: Boundary,
        streams: list[StreamConstraint] | None = None,
        points: list[PointConstraint] | None = None,
        refinement_zones: list[RefinementZone] | None = None,
        char_length: float | None = None,
    ) -> None:
        """Build Gmsh geometry from constraints."""
        import gmsh

        # Insert stream endpoints that lie on boundary edges so the
        # boundary is properly split (prevents large fan-shaped elements).
        raw_verts = [(float(x), float(y)) for x, y in boundary.vertices]
        verts = self._insert_stream_points_on_boundary(raw_verts, streams)

        # Create boundary points (with mesh size if specified)
        boundary_point_ids = []
        for x, y in verts:
            if char_length is not None:
                pt_id = gmsh.model.geo.addPoint(x, y, 0, char_length)
            else:
                pt_id = gmsh.model.geo.addPoint(x, y, 0)
            boundary_point_ids.append(pt_id)

        # Create boundary lines
        boundary_line_ids = []
        n = len(boundary_point_ids)
        for i in range(n):
            line_id = gmsh.model.geo.addLine(boundary_point_ids[i], boundary_point_ids[(i + 1) % n])
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
                    hole_point_ids[i], hole_point_ids[(i + 1) % n_hole]
                )
                hole_line_ids.append(line_id)

            hole_loop = gmsh.model.geo.addCurveLoop(hole_line_ids)
            hole_loops.append(hole_loop)

        # Create surface (with holes)
        if hole_loops:
            gmsh.model.geo.addPlaneSurface([boundary_loop] + hole_loops)
        else:
            gmsh.model.geo.addPlaneSurface([boundary_loop])

        # Build map from boundary coordinates to point IDs for reuse
        coord_to_point: dict[tuple[float, float], int] = {}
        for idx, (bx, by) in enumerate(verts):
            coord_to_point[(bx, by)] = boundary_point_ids[idx]

        # Add stream constraints as embedded curves
        if streams:
            for stream in streams:
                stream_point_ids = []
                for x, y in stream.vertices:
                    fx, fy = float(x), float(y)
                    # Reuse existing point if stream vertex coincides with boundary
                    existing_id = coord_to_point.get((fx, fy))
                    if existing_id is not None:
                        stream_point_ids.append(existing_id)
                    else:
                        pt_id = gmsh.model.geo.addPoint(fx, fy, 0)
                        coord_to_point[(fx, fy)] = pt_id
                        stream_point_ids.append(pt_id)

                stream_line_ids = []
                for i in range(len(stream_point_ids) - 1):
                    line_id = gmsh.model.geo.addLine(stream_point_ids[i], stream_point_ids[i + 1])
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
            self._apply_refinement_fields(refinement_zones, char_length)

    def _apply_refinement_fields(
        self,
        refinement_zones: list[RefinementZone],
        max_char_length: float | None = None,
    ) -> None:
        """Apply refinement zones using Gmsh mesh size fields."""
        import gmsh

        field_ids = []

        for _i, zone in enumerate(refinement_zones):
            zone_char = math.sqrt(zone.max_area)
            # VOut should not exceed the global char_length
            v_out = max_char_length if max_char_length is not None else zone_char * 3

            if zone.center is not None and zone.radius is not None:
                # Circular refinement zone using Ball field
                field_id = gmsh.model.mesh.field.add("Ball")
                gmsh.model.mesh.field.setNumber(field_id, "XCenter", zone.center[0])
                gmsh.model.mesh.field.setNumber(field_id, "YCenter", zone.center[1])
                gmsh.model.mesh.field.setNumber(field_id, "ZCenter", 0)
                gmsh.model.mesh.field.setNumber(field_id, "Radius", zone.radius)
                gmsh.model.mesh.field.setNumber(field_id, "VIn", zone_char)
                gmsh.model.mesh.field.setNumber(field_id, "VOut", v_out)
                field_ids.append(field_id)

            elif zone.polygon is not None:
                # Polygon refinement zone using Box field on bounding box
                xmin, ymin = zone.polygon.min(axis=0)
                xmax, ymax = zone.polygon.max(axis=0)
                field_id = gmsh.model.mesh.field.add("Box")
                gmsh.model.mesh.field.setNumber(field_id, "VIn", zone_char)
                gmsh.model.mesh.field.setNumber(field_id, "VOut", v_out)
                gmsh.model.mesh.field.setNumber(field_id, "XMin", float(xmin))
                gmsh.model.mesh.field.setNumber(field_id, "XMax", float(xmax))
                gmsh.model.mesh.field.setNumber(field_id, "YMin", float(ymin))
                gmsh.model.mesh.field.setNumber(field_id, "YMax", float(ymax))
                gmsh.model.mesh.field.setNumber(field_id, "ZMin", -1)
                gmsh.model.mesh.field.setNumber(field_id, "ZMax", 1)
                gmsh.model.mesh.field.setNumber(field_id, "Thickness", zone_char)
                field_ids.append(field_id)

        # Add a constant field at the global char_length so the background
        # mesh never exceeds the max_area target (setAsBackgroundMesh
        # overrides Mesh.CharacteristicLengthMax).
        if max_char_length is not None:
            const_field = gmsh.model.mesh.field.add("MathEval")
            gmsh.model.mesh.field.setString(const_field, "F", str(max_char_length))
            field_ids.append(const_field)

        if field_ids:
            # Combine fields using Min
            min_field = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", field_ids)
            gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

    def _extract_mesh(self) -> MeshResult:
        """Extract mesh data from Gmsh, filtering degenerate elements."""
        import gmsh

        # Get nodes
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()

        # Build node array (x, y only, ignore z)
        n_nodes = len(node_tags)
        raw_nodes = np.zeros((n_nodes, 2))

        # Create mapping from Gmsh node tag to array index
        tag_to_idx = {}
        for i, tag in enumerate(node_tags):
            tag_to_idx[int(tag)] = i
            # Coordinates are interleaved: x, y, z, x, y, z, ...
            raw_nodes[i, 0] = node_coords[i * 3]
            raw_nodes[i, 1] = node_coords[i * 3 + 1]

        # Get elements
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2)

        # Process elements, skipping degenerate ones (duplicate nodes)
        elements_list: list[list[int]] = []

        for etype, etags, enodes in zip(elem_types, elem_tags, elem_node_tags, strict=False):
            if etype == 2:  # Triangle
                n_tri = len(etags)
                for i in range(n_tri):
                    n0 = tag_to_idx[int(enodes[i * 3])]
                    n1 = tag_to_idx[int(enodes[i * 3 + 1])]
                    n2 = tag_to_idx[int(enodes[i * 3 + 2])]
                    if n0 == n1 or n1 == n2 or n0 == n2:
                        continue  # Skip degenerate triangle
                    elements_list.append([n0, n1, n2, -1])

            elif etype == 3:  # Quad
                n_quad = len(etags)
                for i in range(n_quad):
                    n0 = tag_to_idx[int(enodes[i * 4])]
                    n1 = tag_to_idx[int(enodes[i * 4 + 1])]
                    n2 = tag_to_idx[int(enodes[i * 4 + 2])]
                    n3 = tag_to_idx[int(enodes[i * 4 + 3])]
                    if len({n0, n1, n2, n3}) < 4:
                        continue  # Skip degenerate quad
                    elements_list.append([n0, n1, n2, n3])

        elements = np.array(elements_list, dtype=np.int32)

        # Compact: keep only nodes referenced by elements
        used_nodes = np.unique(elements[elements >= 0])
        old_to_new = np.full(n_nodes, -1, dtype=np.int32)
        for new_idx, old_idx in enumerate(used_nodes):
            old_to_new[old_idx] = new_idx

        nodes = raw_nodes[used_nodes]
        elements = np.where(elements >= 0, old_to_new[elements], -1).astype(np.int32)

        return MeshResult(nodes=nodes, elements=elements)
