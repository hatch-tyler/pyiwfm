"""
Mesh generator base classes for IWFM models.

This module provides the abstract base class for mesh generators
and the MeshResult class for holding generated meshes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from shapely.geometry import Polygon

    from pyiwfm.core.mesh import AppGrid
    from pyiwfm.mesh_generation.constraints import (
        Boundary,
        PointConstraint,
        RefinementZone,
        StreamConstraint,
    )


@dataclass
class MeshResult:
    """
    Result from mesh generation.

    Attributes:
        nodes: Array of node coordinates (n_nodes, 2)
        elements: Array of element vertex indices (n_elements, 3 or 4)
            Triangles use -1 for 4th vertex
        node_markers: Optional node boundary markers
        element_markers: Optional element region markers
    """

    nodes: NDArray[np.float64]
    elements: NDArray[np.int32]
    node_markers: NDArray[np.int32] | None = None
    element_markers: NDArray[np.int32] | None = None

    @property
    def n_nodes(self) -> int:
        """Return number of nodes."""
        return len(self.nodes)

    @property
    def n_elements(self) -> int:
        """Return number of elements."""
        return len(self.elements)

    @property
    def n_triangles(self) -> int:
        """Return number of triangular elements."""
        if self.elements.shape[1] == 3:
            return self.n_elements
        # Elements with -1 in 4th position are triangles
        return int(np.sum(self.elements[:, 3] < 0))

    @property
    def n_quads(self) -> int:
        """Return number of quadrilateral elements."""
        if self.elements.shape[1] == 3:
            return 0
        # Elements with valid 4th vertex are quads
        return int(np.sum(self.elements[:, 3] >= 0))

    def get_element_areas(self) -> NDArray[np.float64]:
        """
        Calculate area of each element.

        Returns:
            Array of element areas
        """
        areas = np.zeros(self.n_elements)

        for i, elem in enumerate(self.elements):
            if self.elements.shape[1] == 3 or elem[3] < 0:
                # Triangle
                v0 = self.nodes[elem[0]]
                v1 = self.nodes[elem[1]]
                v2 = self.nodes[elem[2]]
                areas[i] = 0.5 * abs(
                    (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1])
                )
            else:
                # Quadrilateral - split into two triangles
                v0 = self.nodes[elem[0]]
                v1 = self.nodes[elem[1]]
                v2 = self.nodes[elem[2]]
                v3 = self.nodes[elem[3]]
                # Triangle 0-1-2
                a1 = 0.5 * abs(
                    (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1])
                )
                # Triangle 0-2-3
                a2 = 0.5 * abs(
                    (v2[0] - v0[0]) * (v3[1] - v0[1]) - (v3[0] - v0[0]) * (v2[1] - v0[1])
                )
                areas[i] = a1 + a2

        return areas

    def get_element_centroids(self) -> NDArray[np.float64]:
        """
        Calculate centroid of each element.

        Returns:
            Array of element centroids (n_elements, 2)
        """
        centroids = np.zeros((self.n_elements, 2))

        for i, elem in enumerate(self.elements):
            if self.elements.shape[1] == 3 or elem[3] < 0:
                # Triangle
                vertices = self.nodes[elem[:3]]
            else:
                # Quadrilateral
                vertices = self.nodes[elem]

            centroids[i, 0] = np.mean(vertices[:, 0])
            centroids[i, 1] = np.mean(vertices[:, 1])

        return centroids

    def to_appgrid(self) -> AppGrid:
        """
        Convert mesh result to AppGrid.

        Returns:
            AppGrid instance
        """
        from pyiwfm.core.mesh import AppGrid, Element, Node

        # Create nodes
        nodes = {}
        for i, (x, y) in enumerate(self.nodes):
            node_id = i + 1  # 1-indexed
            is_boundary = False
            if self.node_markers is not None:
                is_boundary = self.node_markers[i] != 0
            nodes[node_id] = Node(
                id=node_id,
                x=float(x),
                y=float(y),
                is_boundary=is_boundary,
            )

        # Create elements
        elements = {}
        for i, elem in enumerate(self.elements):
            elem_id = i + 1  # 1-indexed
            if self.elements.shape[1] == 3 or elem[3] < 0:
                # Triangle - convert to 1-indexed
                vertices = tuple(int(v + 1) for v in elem[:3])
            else:
                # Quad - convert to 1-indexed
                vertices = tuple(int(v + 1) for v in elem)

            subregion = 0
            if self.element_markers is not None:
                subregion = int(self.element_markers[i])

            elements[elem_id] = Element(
                id=elem_id,
                vertices=vertices,
                subregion=subregion,
            )

        # Create AppGrid
        grid = AppGrid(nodes=nodes, elements=elements)

        # Set coordinate arrays (pre-populate cache)
        grid._x_cache = self.nodes[:, 0].copy()
        grid._y_cache = self.nodes[:, 1].copy()

        # Build connectivity
        grid.compute_connectivity()

        return grid

    def __repr__(self) -> str:
        return f"MeshResult(n_nodes={self.n_nodes}, n_elements={self.n_elements})"


class MeshGenerator(ABC):
    """
    Abstract base class for mesh generators.

    Subclasses must implement the generate() method to create
    a mesh from boundary and constraint definitions.
    """

    @abstractmethod
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
        pass

    def generate_from_shapely(
        self,
        polygon: Polygon,  # type: ignore
        max_area: float | None = None,
        min_angle: float | None = None,
    ) -> MeshResult:
        """
        Generate mesh from Shapely polygon.

        Args:
            polygon: Shapely Polygon object
            max_area: Maximum element area
            min_angle: Minimum angle constraint

        Returns:
            MeshResult with generated mesh
        """
        from pyiwfm.mesh_generation.constraints import Boundary

        # Extract exterior coordinates
        exterior = np.array(polygon.exterior.coords)[:-1]  # Remove closing point

        # Extract hole coordinates
        holes = []
        for interior in polygon.interiors:
            hole = np.array(interior.coords)[:-1]
            holes.append(hole)

        boundary = Boundary(vertices=exterior, holes=holes)

        return self.generate(boundary, max_area=max_area, min_angle=min_angle)
