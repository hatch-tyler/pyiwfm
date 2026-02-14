"""
Mesh generation constraints for IWFM models.

This module provides classes for defining constraints used in mesh
generation including boundaries, streams, refinement zones, and
fixed points.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
from numpy.typing import NDArray


@dataclass
class BoundarySegment:
    """
    A segment of a boundary polygon.

    Attributes:
        start: Start point coordinates (x, y)
        end: End point coordinates (x, y)
        marker: Boundary marker for identifying segment type
    """

    start: NDArray[np.float64]
    end: NDArray[np.float64]
    marker: int = 0

    @property
    def length(self) -> float:
        """Return segment length."""
        return float(np.sqrt(np.sum((self.end - self.start) ** 2)))

    @property
    def midpoint(self) -> tuple[float, float]:
        """Return segment midpoint."""
        mid = (self.start + self.end) / 2
        return (float(mid[0]), float(mid[1]))

    def __repr__(self) -> str:
        return f"BoundarySegment(length={self.length:.2f}, marker={self.marker})"


@dataclass
class Boundary:
    """
    A boundary polygon for mesh generation.

    Attributes:
        vertices: Array of boundary vertices (n, 2) in counter-clockwise order
        holes: List of hole polygons (each is array of vertices)
        markers: Optional array of segment markers
    """

    vertices: NDArray[np.float64]
    holes: list[NDArray[np.float64]] = field(default_factory=list)
    markers: NDArray[np.int32] | None = None

    @property
    def n_vertices(self) -> int:
        """Return number of vertices."""
        return len(self.vertices)

    @property
    def is_closed(self) -> bool:
        """Check if boundary is closed (first and last vertices connected)."""
        return self.n_vertices >= 3

    @property
    def area(self) -> float:
        """
        Calculate boundary area using shoelace formula.

        Returns:
            Area (positive for counter-clockwise vertices)
        """
        outer_area = self._polygon_area(self.vertices)

        # Subtract hole areas
        hole_area = sum(self._polygon_area(hole) for hole in self.holes)

        return abs(outer_area) - abs(hole_area)

    @staticmethod
    def _polygon_area(vertices: NDArray[np.float64]) -> float:
        """Calculate polygon area using shoelace formula."""
        n = len(vertices)
        if n < 3:
            return 0.0

        x = vertices[:, 0]
        y = vertices[:, 1]

        # Shoelace formula
        area = 0.5 * abs(
            np.sum(x[:-1] * y[1:]) + x[-1] * y[0]
            - np.sum(x[1:] * y[:-1]) - x[0] * y[-1]
        )
        return float(area)

    @property
    def centroid(self) -> tuple[float, float]:
        """Calculate boundary centroid."""
        cx = float(np.mean(self.vertices[:, 0]))
        cy = float(np.mean(self.vertices[:, 1]))
        return (cx, cy)

    def get_segments(self) -> list[BoundarySegment]:
        """Get list of boundary segments."""
        segments = []
        n = self.n_vertices

        for i in range(n):
            start = self.vertices[i]
            end = self.vertices[(i + 1) % n]
            marker = int(self.markers[i]) if self.markers is not None else 0
            segments.append(BoundarySegment(start=start, end=end, marker=marker))

        return segments

    def get_hole_points(self) -> list[tuple[float, float]]:
        """
        Get interior points for each hole (for mesh generators).

        Returns:
            List of (x, y) points inside each hole
        """
        hole_points = []
        for hole in self.holes:
            cx = float(np.mean(hole[:, 0]))
            cy = float(np.mean(hole[:, 1]))
            hole_points.append((cx, cy))
        return hole_points

    def __repr__(self) -> str:
        return f"Boundary(n_vertices={self.n_vertices}, n_holes={len(self.holes)})"


@dataclass
class StreamConstraint:
    """
    A stream line constraint for mesh generation.

    Stream constraints ensure the mesh conforms to stream locations,
    with nodes placed along the stream path.

    Attributes:
        vertices: Array of stream vertices (n, 2)
        stream_id: Identifier for this stream
        marker: Segment marker for stream edges
    """

    vertices: NDArray[np.float64]
    stream_id: int = 0
    marker: int = 0

    @property
    def n_vertices(self) -> int:
        """Return number of vertices."""
        return len(self.vertices)

    @property
    def length(self) -> float:
        """Calculate total stream length."""
        if self.n_vertices < 2:
            return 0.0

        total = 0.0
        for i in range(self.n_vertices - 1):
            dx = self.vertices[i + 1, 0] - self.vertices[i, 0]
            dy = self.vertices[i + 1, 1] - self.vertices[i, 1]
            total += np.sqrt(dx * dx + dy * dy)

        return float(total)

    def get_segments(self) -> list[BoundarySegment]:
        """Get list of stream segments."""
        segments = []

        for i in range(self.n_vertices - 1):
            start = self.vertices[i]
            end = self.vertices[i + 1]
            segments.append(BoundarySegment(start=start, end=end, marker=self.marker))

        return segments

    def resample(self, spacing: float) -> "StreamConstraint":
        """
        Resample stream at regular spacing.

        Args:
            spacing: Distance between points

        Returns:
            New StreamConstraint with resampled vertices
        """
        if self.n_vertices < 2:
            return StreamConstraint(
                vertices=self.vertices.copy(),
                stream_id=self.stream_id,
                marker=self.marker,
            )

        # Calculate cumulative distances
        distances = [0.0]
        for i in range(self.n_vertices - 1):
            dx = self.vertices[i + 1, 0] - self.vertices[i, 0]
            dy = self.vertices[i + 1, 1] - self.vertices[i, 1]
            distances.append(distances[-1] + np.sqrt(dx * dx + dy * dy))

        total_length = distances[-1]
        if total_length <= spacing:
            return StreamConstraint(
                vertices=self.vertices.copy(),
                stream_id=self.stream_id,
                marker=self.marker,
            )

        # Generate new points at regular spacing
        new_points = [self.vertices[0].copy()]
        current_dist = spacing

        while current_dist < total_length:
            # Find segment containing this distance
            for i in range(len(distances) - 1):
                if distances[i] <= current_dist <= distances[i + 1]:
                    # Interpolate within segment
                    t = (current_dist - distances[i]) / (distances[i + 1] - distances[i])
                    x = self.vertices[i, 0] + t * (self.vertices[i + 1, 0] - self.vertices[i, 0])
                    y = self.vertices[i, 1] + t * (self.vertices[i + 1, 1] - self.vertices[i, 1])
                    new_points.append(np.array([x, y]))
                    break
            current_dist += spacing

        new_points.append(self.vertices[-1].copy())

        return StreamConstraint(
            vertices=np.array(new_points),
            stream_id=self.stream_id,
            marker=self.marker,
        )

    def __repr__(self) -> str:
        return f"StreamConstraint(id={self.stream_id}, n_vertices={self.n_vertices})"


@dataclass
class RefinementZone:
    """
    A zone requiring mesh refinement.

    Can be defined as either a circular zone (center + radius) or
    a polygon.

    Attributes:
        center: Center point for circular zone (x, y)
        radius: Radius for circular zone
        polygon: Polygon vertices for polygonal zone
        max_area: Maximum element area in this zone
    """

    center: tuple[float, float] | None = None
    radius: float | None = None
    polygon: NDArray[np.float64] | None = None
    max_area: float = 100.0

    def __post_init__(self) -> None:
        """Validate zone definition."""
        has_circle = self.center is not None and self.radius is not None
        has_polygon = self.polygon is not None

        if not has_circle and not has_polygon:
            raise ValueError("Must specify either center+radius or polygon")

    def contains(self, x: float, y: float) -> bool:
        """
        Check if point is inside refinement zone.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            True if point is inside zone
        """
        if self.center is not None and self.radius is not None:
            # Circular zone
            dx = x - self.center[0]
            dy = y - self.center[1]
            return dx * dx + dy * dy <= self.radius * self.radius

        if self.polygon is not None:
            # Polygon zone - use ray casting
            return self._point_in_polygon(x, y, self.polygon)

        return False

    @staticmethod
    def _point_in_polygon(x: float, y: float, polygon: NDArray[np.float64]) -> bool:
        """Check if point is inside polygon using ray casting."""
        n = len(polygon)
        inside = False

        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]

            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside

            j = i

        return inside

    def __repr__(self) -> str:
        if self.center is not None:
            return f"RefinementZone(center={self.center}, radius={self.radius}, max_area={self.max_area})"
        return f"RefinementZone(polygon, max_area={self.max_area})"


@dataclass
class PointConstraint:
    """
    A fixed point constraint for mesh generation.

    Ensures a node is placed at this exact location.

    Attributes:
        x: X coordinate
        y: Y coordinate
        marker: Point marker
    """

    x: float
    y: float
    marker: int = 0

    def as_array(self) -> NDArray[np.float64]:
        """Return point as numpy array."""
        return np.array([self.x, self.y])

    def __repr__(self) -> str:
        return f"PointConstraint(x={self.x}, y={self.y})"
