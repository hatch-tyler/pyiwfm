"""
Mesh quality metrics for IWFM finite element meshes.

This module computes per-element quality metrics (area, aspect ratio, skewness,
interior angles) and aggregate statistics for an entire mesh. Useful for
identifying poorly shaped elements that may cause numerical issues.

Example
-------
Compute quality for a single triangle:

>>> verts = [(0.0, 0.0), (1.0, 0.0), (0.5, 0.866)]
>>> q = compute_element_quality(verts, element_id=1)
>>> q.n_vertices
3
>>> q.min_angle > 59.0
True

Compute mesh-wide quality report:

>>> from pyiwfm.core.mesh import AppGrid, Node, Element
>>> nodes = {1: Node(1, 0, 0), 2: Node(2, 1, 0), 3: Node(3, 1, 1), 4: Node(4, 0, 1)}
>>> elements = {1: Element(1, (1, 2, 3, 4))}
>>> grid = AppGrid(nodes=nodes, elements=elements)
>>> report = compute_mesh_quality(grid)
>>> report.n_elements
1
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pyiwfm.core.mesh import AppGrid


@dataclass
class ElementQuality:
    """
    Quality metrics for a single mesh element.

    Parameters
    ----------
    element_id : int
        Element identifier (1-based).
    area : float
        Element area computed via the shoelace formula.
    aspect_ratio : float
        Ratio of the longest edge to the shortest edge. A value of 1.0
        indicates all edges are equal length.
    skewness : float
        Deviation from an ideal shape, ranging from 0.0 (perfect) to 1.0
        (degenerate). For triangles, compares to equilateral; for quads,
        compares diagonal lengths.
    min_angle : float
        Smallest interior angle in degrees.
    max_angle : float
        Largest interior angle in degrees.
    n_vertices : int
        Number of vertices (3 for triangle, 4 for quad).
    """

    element_id: int
    area: float
    aspect_ratio: float
    skewness: float
    min_angle: float
    max_angle: float
    n_vertices: int


@dataclass
class MeshQualityReport:
    """
    Aggregate mesh quality statistics.

    Parameters
    ----------
    n_elements : int
        Total number of elements.
    n_triangles : int
        Number of triangular elements.
    n_quads : int
        Number of quadrilateral elements.
    area_min : float
        Minimum element area.
    area_max : float
        Maximum element area.
    area_mean : float
        Mean element area.
    area_std : float
        Standard deviation of element areas.
    aspect_ratio_min : float
        Minimum aspect ratio across all elements.
    aspect_ratio_max : float
        Maximum aspect ratio across all elements.
    aspect_ratio_mean : float
        Mean aspect ratio.
    skewness_mean : float
        Mean skewness across all elements.
    min_angle_global : float
        Smallest interior angle in the entire mesh (degrees).
    max_angle_global : float
        Largest interior angle in the entire mesh (degrees).
    poor_quality_count : int
        Number of elements with aspect_ratio > 10 or skewness > 0.8.
    element_qualities : list[ElementQuality]
        Per-element quality metrics.
    """

    n_elements: int
    n_triangles: int
    n_quads: int
    area_min: float
    area_max: float
    area_mean: float
    area_std: float
    aspect_ratio_min: float
    aspect_ratio_max: float
    aspect_ratio_mean: float
    skewness_mean: float
    min_angle_global: float
    max_angle_global: float
    poor_quality_count: int
    element_qualities: list[ElementQuality] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize for API response (without per-element details).

        Returns
        -------
        dict[str, Any]
            Dictionary with aggregate statistics only.
        """
        return {
            "n_elements": self.n_elements,
            "n_triangles": self.n_triangles,
            "n_quads": self.n_quads,
            "area_min": self.area_min,
            "area_max": self.area_max,
            "area_mean": self.area_mean,
            "area_std": self.area_std,
            "aspect_ratio_min": self.aspect_ratio_min,
            "aspect_ratio_max": self.aspect_ratio_max,
            "aspect_ratio_mean": self.aspect_ratio_mean,
            "skewness_mean": self.skewness_mean,
            "min_angle_global": self.min_angle_global,
            "max_angle_global": self.max_angle_global,
            "poor_quality_count": self.poor_quality_count,
        }


def _edge_length(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Compute Euclidean distance between two 2D points."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx * dx + dy * dy)


def _shoelace_area(vertices: list[tuple[float, float]]) -> float:
    """
    Compute polygon area using the shoelace formula.

    Parameters
    ----------
    vertices : list[tuple[float, float]]
        Ordered vertex coordinates.

    Returns
    -------
    float
        Absolute area of the polygon.
    """
    n = len(vertices)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    return abs(area) / 2.0


def _compute_interior_angles(vertices: list[tuple[float, float]]) -> list[float]:
    """
    Compute interior angles at each vertex using dot product of edge vectors.

    Parameters
    ----------
    vertices : list[tuple[float, float]]
        Ordered vertex coordinates.

    Returns
    -------
    list[float]
        Interior angles in degrees, one per vertex.
    """
    n = len(vertices)
    angles: list[float] = []
    for i in range(n):
        # Vectors from vertex i to its two neighbors
        prev_idx = (i - 1) % n
        next_idx = (i + 1) % n

        v1x = vertices[prev_idx][0] - vertices[i][0]
        v1y = vertices[prev_idx][1] - vertices[i][1]
        v2x = vertices[next_idx][0] - vertices[i][0]
        v2y = vertices[next_idx][1] - vertices[i][1]

        dot = v1x * v2x + v1y * v2y
        mag1 = math.sqrt(v1x * v1x + v1y * v1y)
        mag2 = math.sqrt(v2x * v2x + v2y * v2y)

        if mag1 < 1e-15 or mag2 < 1e-15:
            angles.append(0.0)
            continue

        cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        angles.append(math.degrees(math.acos(cos_angle)))

    return angles


def _compute_skewness_tri(vertices: list[tuple[float, float]], area: float) -> float:
    """
    Compute skewness for a triangle by comparing to an equilateral triangle.

    Skewness is defined as 1 - (area / area_equilateral), where
    area_equilateral is the area of an equilateral triangle with the
    same longest edge. Returns 0 for equilateral, approaches 1 for
    degenerate triangles.

    Parameters
    ----------
    vertices : list[tuple[float, float]]
        Three vertex coordinates.
    area : float
        Pre-computed area of the triangle.

    Returns
    -------
    float
        Skewness in [0, 1].
    """
    # Find the longest edge
    edges = []
    n = len(vertices)
    for i in range(n):
        edges.append(_edge_length(vertices[i], vertices[(i + 1) % n]))

    max_edge = max(edges)
    if max_edge < 1e-15:
        return 1.0

    # Area of equilateral triangle with side = max_edge
    area_equilateral = (math.sqrt(3.0) / 4.0) * max_edge * max_edge

    if area_equilateral < 1e-15:
        return 1.0

    skewness = 1.0 - (area / area_equilateral)
    return max(0.0, min(1.0, skewness))


def _compute_skewness_quad(vertices: list[tuple[float, float]]) -> float:
    """
    Compute skewness for a quadrilateral by comparing diagonal lengths.

    A perfect quad (square/rhombus) has equal diagonals. Skewness is
    defined as 1 - (min_diag / max_diag).

    Parameters
    ----------
    vertices : list[tuple[float, float]]
        Four vertex coordinates.

    Returns
    -------
    float
        Skewness in [0, 1].
    """
    d1 = _edge_length(vertices[0], vertices[2])
    d2 = _edge_length(vertices[1], vertices[3])

    max_diag = max(d1, d2)
    min_diag = min(d1, d2)

    if max_diag < 1e-15:
        return 1.0

    return 1.0 - (min_diag / max_diag)


def compute_element_quality(
    vertices: list[tuple[float, float]],
    element_id: int = 0,
) -> ElementQuality:
    """
    Compute quality metrics for a single element given its vertex coordinates.

    Parameters
    ----------
    vertices : list[tuple[float, float]]
        Ordered vertex coordinates (3 for triangle, 4 for quad).
    element_id : int, optional
        Element identifier for the returned dataclass. Default is 0.

    Returns
    -------
    ElementQuality
        Quality metrics for the element.

    Raises
    ------
    ValueError
        If the number of vertices is not 3 or 4.
    """
    n = len(vertices)
    if n < 3 or n > 4:
        raise ValueError(f"Expected 3 or 4 vertices, got {n}")

    # Area via shoelace
    area = _shoelace_area(vertices)

    # Edge lengths
    edge_lengths = [_edge_length(vertices[i], vertices[(i + 1) % n]) for i in range(n)]
    min_edge = min(edge_lengths)
    max_edge = max(edge_lengths)

    # Aspect ratio
    aspect_ratio = max_edge / min_edge if min_edge > 1e-15 else float("inf")

    # Interior angles
    angles = _compute_interior_angles(vertices)
    min_angle = min(angles)
    max_angle = max(angles)

    # Skewness
    if n == 3:
        skewness = _compute_skewness_tri(vertices, area)
    else:
        skewness = _compute_skewness_quad(vertices)

    return ElementQuality(
        element_id=element_id,
        area=area,
        aspect_ratio=aspect_ratio,
        skewness=skewness,
        min_angle=min_angle,
        max_angle=max_angle,
        n_vertices=n,
    )


def compute_mesh_quality(grid: AppGrid) -> MeshQualityReport:
    """
    Compute quality metrics for the entire mesh.

    Iterates over all elements, computes per-element quality metrics,
    and aggregates them into summary statistics.

    Parameters
    ----------
    grid : AppGrid
        The mesh to analyze.

    Returns
    -------
    MeshQualityReport
        Aggregate and per-element quality metrics.
    """
    qualities: list[ElementQuality] = []

    for elem in grid.iter_elements():
        verts = [(grid.nodes[vid].x, grid.nodes[vid].y) for vid in elem.vertices]
        q = compute_element_quality(verts, element_id=elem.id)
        qualities.append(q)

    if not qualities:
        return MeshQualityReport(
            n_elements=0,
            n_triangles=0,
            n_quads=0,
            area_min=0.0,
            area_max=0.0,
            area_mean=0.0,
            area_std=0.0,
            aspect_ratio_min=0.0,
            aspect_ratio_max=0.0,
            aspect_ratio_mean=0.0,
            skewness_mean=0.0,
            min_angle_global=0.0,
            max_angle_global=0.0,
            poor_quality_count=0,
            element_qualities=[],
        )

    areas = np.array([q.area for q in qualities])
    aspect_ratios = np.array([q.aspect_ratio for q in qualities])
    skewnesses = np.array([q.skewness for q in qualities])
    min_angles = np.array([q.min_angle for q in qualities])
    max_angles = np.array([q.max_angle for q in qualities])

    n_triangles = sum(1 for q in qualities if q.n_vertices == 3)
    n_quads = sum(1 for q in qualities if q.n_vertices == 4)

    poor_quality_count = int(np.sum((aspect_ratios > 10.0) | (skewnesses > 0.8)))

    return MeshQualityReport(
        n_elements=len(qualities),
        n_triangles=n_triangles,
        n_quads=n_quads,
        area_min=float(np.min(areas)),
        area_max=float(np.max(areas)),
        area_mean=float(np.mean(areas)),
        area_std=float(np.std(areas)),
        aspect_ratio_min=float(np.min(aspect_ratios)),
        aspect_ratio_max=float(np.max(aspect_ratios)),
        aspect_ratio_mean=float(np.mean(aspect_ratios)),
        skewness_mean=float(np.mean(skewnesses)),
        min_angle_global=float(np.min(min_angles)),
        max_angle_global=float(np.max(max_angles)),
        poor_quality_count=poor_quality_count,
        element_qualities=qualities,
    )
