"""
Finite element interpolation for IWFM meshes.

This module provides point-in-element location and finite element
interpolation capabilities similar to IWFM's Class_Grid.

The interpolation methods support both triangular (3 vertices) and
quadrilateral (4 vertices) elements using:
- Barycentric coordinates for triangles (area coordinates)
- Bilinear isoparametric interpolation for quads

Example
-------
Interpolate a value at a point:

>>> from pyiwfm.core.mesh import AppGrid, Node, Element
>>> from pyiwfm.core.interpolation import FEInterpolator
>>> # Create a simple mesh
>>> nodes = {
...     1: Node(id=1, x=0.0, y=0.0),
...     2: Node(id=2, x=100.0, y=0.0),
...     3: Node(id=3, x=50.0, y=100.0),
... }
>>> elements = {1: Element(id=1, vertices=(1, 2, 3))}
>>> grid = AppGrid(nodes=nodes, elements=elements)
>>> interp = FEInterpolator(grid)
>>> # Find element containing point and get interpolation weights
>>> elem_id, nodes, coeffs = interp.interpolate(50.0, 33.0)
>>> elem_id
1
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pyiwfm.core.mesh import AppGrid


@dataclass
class InterpolationResult:
    """
    Result of finite element interpolation.

    Attributes
    ----------
    element_id : int
        ID of the element containing the point (0 if not found).
    node_ids : tuple[int, ...]
        Node IDs of the containing element's vertices.
    coefficients : NDArray[np.float64]
        Interpolation coefficients (shape functions) for each node.
        These sum to 1.0 for valid interpolations.
    """

    element_id: int
    node_ids: tuple[int, ...]
    coefficients: NDArray[np.float64]

    @property
    def found(self) -> bool:
        """Return True if the point was found within an element."""
        return self.element_id > 0

    def interpolate(self, node_values: dict[int, float]) -> float:
        """
        Interpolate a value using the stored coefficients.

        Parameters
        ----------
        node_values : dict[int, float]
            Dictionary mapping node ID to value at that node.

        Returns
        -------
        float
            Interpolated value at the query point.

        Raises
        ------
        ValueError
            If point was not found in any element.
        KeyError
            If a required node is missing from node_values.
        """
        if not self.found:
            raise ValueError("Point was not found in any element")

        result = 0.0
        for node_id, coeff in zip(self.node_ids, self.coefficients, strict=False):
            result += coeff * node_values[node_id]
        return result


def _xpoint(
    x1: float, y1: float, x2: float, y2: float, xp: float, yp: float
) -> tuple[float, float]:
    """
    Find intersection between a line and its perpendicular through a point.

    Given a line segment from (x1, y1) to (x2, y2) and a point (xp, yp),
    find the point (xx, yx) on the line that is closest to (xp, yp).

    Parameters
    ----------
    x1, y1 : float
        First endpoint of line segment.
    x2, y2 : float
        Second endpoint of line segment.
    xp, yp : float
        Query point.

    Returns
    -------
    tuple[float, float]
        Coordinates (xx, yx) of the closest point on the line.
    """
    # Handle vertical line
    if x1 == x2:
        return x1, yp

    # Handle horizontal line
    if y1 == y2:
        return xp, y1

    # General case: solve for intersection
    slope_line = (y2 - y1) / (x2 - x1)
    slope_perp = -1.0 / slope_line
    xx = (slope_line * x1 - slope_perp * xp + yp - y1) / (slope_line - slope_perp)
    yx = slope_perp * xx - slope_perp * xp + yp

    return xx, yx


def point_in_element(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    xp: float,
    yp: float,
) -> bool:
    """
    Check if a point lies inside an element.

    Uses the cross-product test: for a point inside a convex polygon
    (traversed counter-clockwise), the cross product of each edge vector
    with the vector to the query point should be non-negative.

    Parameters
    ----------
    x : NDArray[np.float64]
        X coordinates of element vertices (3 or 4 values).
    y : NDArray[np.float64]
        Y coordinates of element vertices (3 or 4 values).
    xp : float
        X coordinate of query point.
    yp : float
        Y coordinate of query point.

    Returns
    -------
    bool
        True if point is inside or on the boundary of the element.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0.0, 1.0, 0.5])
    >>> y = np.array([0.0, 0.0, 1.0])
    >>> point_in_element(x, y, 0.5, 0.3)  # Inside triangle
    True
    >>> point_in_element(x, y, 2.0, 0.0)  # Outside
    False
    """
    n_vertex = len(x)

    for i in range(n_vertex):
        # Check if point is exactly at a vertex
        if xp == x[i] and yp == y[i]:
            return True

        # Get edge endpoints
        x1, y1 = x[i], y[i]
        if i < n_vertex - 1:
            x2, y2 = x[i + 1], y[i + 1]
            # Check if point is at next vertex
            if xp == x2 and yp == y2:
                return True
        else:
            x2, y2 = x[0], y[0]

        # Find closest point on edge to query point
        xx, yx = _xpoint(x1, y1, x2, y2, xp, yp)

        # Compute dot product to check which side of edge point is on
        # Edge normal (pointing inward for CCW polygon): (y1-y2, x2-x1)
        dot_product = (xp - xx) * (y1 - y2) + (yp - yx) * (x2 - x1)

        if dot_product < 0.0:
            return False

    return True


def find_containing_element(
    grid: AppGrid,
    xp: float,
    yp: float,
) -> int:
    """
    Find the first element containing a given point.

    Parameters
    ----------
    grid : AppGrid
        The mesh to search.
    xp : float
        X coordinate of query point.
    yp : float
        Y coordinate of query point.

    Returns
    -------
    int
        Element ID containing the point, or 0 if not found.

    Examples
    --------
    >>> from pyiwfm.core.mesh import AppGrid, Node, Element
    >>> nodes = {
    ...     1: Node(id=1, x=0.0, y=0.0),
    ...     2: Node(id=2, x=100.0, y=0.0),
    ...     3: Node(id=3, x=50.0, y=100.0),
    ... }
    >>> elements = {1: Element(id=1, vertices=(1, 2, 3))}
    >>> grid = AppGrid(nodes=nodes, elements=elements)
    >>> find_containing_element(grid, 50.0, 33.0)
    1
    >>> find_containing_element(grid, 200.0, 200.0)
    0
    """
    for elem_id, elem in grid.elements.items():
        len(elem.vertices)
        x = np.array([grid.nodes[vid].x for vid in elem.vertices])
        y = np.array([grid.nodes[vid].y for vid in elem.vertices])

        if point_in_element(x, y, xp, yp):
            return elem_id

    return 0


def interpolation_coefficients(
    n_vertex: int,
    xp: float,
    yp: float,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute finite element interpolation coefficients (shape functions).

    For triangular elements, uses barycentric (area) coordinates.
    For quadrilateral elements, uses bilinear isoparametric interpolation.

    Parameters
    ----------
    n_vertex : int
        Number of vertices (3 for triangle, 4 for quad).
    xp : float
        X coordinate of interpolation point.
    yp : float
        Y coordinate of interpolation point.
    x : NDArray[np.float64]
        X coordinates of element vertices.
    y : NDArray[np.float64]
        Y coordinates of element vertices.

    Returns
    -------
    NDArray[np.float64]
        Interpolation coefficients for each vertex. These sum to 1.0
        for points inside the element. Values of -1.0 indicate failure.

    Notes
    -----
    Triangular interpolation uses area coordinates:
        N1 = A_234 / A_total, N2 = A_134 / A_total, etc.

    Quadrilateral interpolation maps to the reference element [-1,1]x[-1,1]:
        N1 = 0.25*(1-xi)*(1-eta)
        N2 = 0.25*(1+xi)*(1-eta)
        N3 = 0.25*(1+xi)*(1+eta)
        N4 = 0.25*(1-xi)*(1+eta)
    """
    coeff = np.full(n_vertex, -1.0)

    if n_vertex == 3:
        # Triangular element - barycentric coordinates
        xij = x[0] - x[1]
        xjk = x[1] - x[2]
        xki = x[2] - x[0]
        yij = y[0] - y[1]
        yjk = y[1] - y[2]
        yki = y[2] - y[0]

        denom = -xki * yjk + xjk * yki

        if abs(denom) < 1e-15:
            # Degenerate triangle
            return coeff

        xt = (-x[0] * y[2] + x[2] * y[0] + yki * xp - xki * yp) / denom
        xt = max(0.0, xt)

        yt = (x[0] * y[1] - x[1] * y[0] + yij * xp - xij * yp) / denom
        yt = max(0.0, yt)

        coeff[0] = 1.0 - min(1.0, xt + yt)
        coeff[1] = xt
        coeff[2] = yt

    else:
        # Quadrilateral element - bilinear isoparametric
        # Find local coordinates (xi, eta) in [-1, 1] x [-1, 1]

        # Solve for xi
        a = (y[0] - y[1]) * (x[2] - x[3]) - (y[2] - y[3]) * (x[0] - x[1])
        bo = y[0] * x[3] - y[1] * x[2] + y[2] * x[1] - y[3] * x[0]
        bx = -y[0] + y[1] - y[2] + y[3]
        by = x[0] - x[1] + x[2] - x[3]
        b = bo + bx * xp + by * yp
        co = -(y[0] + y[1]) * (x[2] + x[3]) + (y[2] + y[3]) * (x[0] + x[1])
        cx = y[0] + y[1] - y[2] - y[3]
        cy = -x[0] - x[1] + x[2] + x[3]
        c = co + 2.0 * (cx * xp + cy * yp)

        if a == 0.0:
            if b == 0.0:
                return coeff
            xt = -c / (2.0 * b)
        else:
            discriminant = b * b - a * c
            if discriminant < 0.0:
                return coeff
            xt = (-b + math.sqrt(discriminant)) / a

        xt = max(-1.0, min(xt, 1.0))

        # Solve for eta
        a = (y[1] - y[2]) * (x[0] - x[3]) - (y[0] - y[3]) * (x[1] - x[2])
        bo = y[0] * x[1] - y[1] * x[0] + y[2] * x[3] - y[3] * x[2]
        bx = -y[0] + y[1] - y[2] + y[3]
        by = x[0] - x[1] + x[2] - x[3]
        b = bo + bx * xp + by * yp
        co = -(y[0] + y[3]) * (x[1] + x[2]) + (y[1] + y[2]) * (x[0] + x[3])
        cx = y[0] - y[1] - y[2] + y[3]
        cy = -x[0] + x[1] + x[2] - x[3]
        c = co + 2.0 * (cx * xp + cy * yp)

        if a == 0.0:
            if b == 0.0:
                return coeff
            yt = -c / (2.0 * b)
        else:
            discriminant = b * b - a * c
            if discriminant < 0.0:
                return coeff
            yt = (-b - math.sqrt(discriminant)) / a

        yt = max(-1.0, min(yt, 1.0))

        # Compute shape functions
        coeff[0] = 0.25 * (1.0 - xt) * (1.0 - yt)
        coeff[1] = 0.25 * (1.0 + xt) * (1.0 - yt)
        coeff[2] = 0.25 * (1.0 + xt) * (1.0 + yt)
        coeff[3] = 0.25 * (1.0 - xt) * (1.0 + yt)

    return coeff


def fe_interpolate_at_element(
    grid: AppGrid,
    element_id: int,
    xp: float,
    yp: float,
) -> NDArray[np.float64]:
    """
    Compute interpolation coefficients at a known element.

    Parameters
    ----------
    grid : AppGrid
        The mesh containing the element.
    element_id : int
        ID of the element containing the point.
    xp : float
        X coordinate of interpolation point.
    yp : float
        Y coordinate of interpolation point.

    Returns
    -------
    NDArray[np.float64]
        Interpolation coefficients for each vertex of the element.

    Raises
    ------
    KeyError
        If element_id is not found in the grid.
    """
    elem = grid.elements[element_id]
    n_vertex = len(elem.vertices)

    x = np.array([grid.nodes[vid].x for vid in elem.vertices])
    y = np.array([grid.nodes[vid].y for vid in elem.vertices])

    return interpolation_coefficients(n_vertex, xp, yp, x, y)


def fe_interpolate(
    grid: AppGrid,
    xp: float,
    yp: float,
) -> InterpolationResult:
    """
    Find containing element and compute interpolation coefficients.

    This is the main interpolation function that combines element
    location and coefficient computation.

    Parameters
    ----------
    grid : AppGrid
        The mesh to search and interpolate in.
    xp : float
        X coordinate of interpolation point.
    yp : float
        Y coordinate of interpolation point.

    Returns
    -------
    InterpolationResult
        Result containing element ID, node IDs, and coefficients.
        If point is not found, element_id will be 0.

    Examples
    --------
    >>> from pyiwfm.core.mesh import AppGrid, Node, Element
    >>> nodes = {
    ...     1: Node(id=1, x=0.0, y=0.0),
    ...     2: Node(id=2, x=100.0, y=0.0),
    ...     3: Node(id=3, x=50.0, y=100.0),
    ... }
    >>> elements = {1: Element(id=1, vertices=(1, 2, 3))}
    >>> grid = AppGrid(nodes=nodes, elements=elements)
    >>> result = fe_interpolate(grid, 50.0, 33.0)
    >>> result.found
    True
    >>> result.element_id
    1
    """
    # Find containing element
    elem_id = find_containing_element(grid, xp, yp)

    if elem_id == 0:
        # Point not found
        return InterpolationResult(
            element_id=0,
            node_ids=(),
            coefficients=np.array([]),
        )

    elem = grid.elements[elem_id]
    coeffs = fe_interpolate_at_element(grid, elem_id, xp, yp)

    return InterpolationResult(
        element_id=elem_id,
        node_ids=elem.vertices,
        coefficients=coeffs,
    )


class FEInterpolator:
    """
    Finite element interpolator for an IWFM mesh.

    This class provides a convenient interface for repeated interpolation
    operations on the same mesh. It caches mesh data for efficiency.

    Parameters
    ----------
    grid : AppGrid
        The mesh to use for interpolation.

    Examples
    --------
    >>> from pyiwfm.core.mesh import AppGrid, Node, Element
    >>> nodes = {
    ...     1: Node(id=1, x=0.0, y=0.0),
    ...     2: Node(id=2, x=100.0, y=0.0),
    ...     3: Node(id=3, x=50.0, y=100.0),
    ... }
    >>> elements = {1: Element(id=1, vertices=(1, 2, 3))}
    >>> grid = AppGrid(nodes=nodes, elements=elements)
    >>> interp = FEInterpolator(grid)
    >>> elem_id, node_ids, coeffs = interp.interpolate(50.0, 33.0)
    >>> # Interpolate nodal values
    >>> values = {1: 100.0, 2: 200.0, 3: 150.0}
    >>> interp.interpolate_value(50.0, 33.0, values)
    150.0  # approximately
    """

    def __init__(self, grid: AppGrid) -> None:
        self._grid = grid
        self._build_cache()

    def _build_cache(self) -> None:
        """Build cached arrays for efficient computation."""
        # Sort element IDs for consistent iteration
        self._elem_ids = sorted(self._grid.elements.keys())

        # Build vertex coordinate arrays for each element
        self._elem_x: dict[int, NDArray[np.float64]] = {}
        self._elem_y: dict[int, NDArray[np.float64]] = {}
        self._elem_vertices: dict[int, tuple[int, ...]] = {}

        for eid in self._elem_ids:
            elem = self._grid.elements[eid]
            self._elem_vertices[eid] = elem.vertices
            self._elem_x[eid] = np.array([self._grid.nodes[vid].x for vid in elem.vertices])
            self._elem_y[eid] = np.array([self._grid.nodes[vid].y for vid in elem.vertices])

    @property
    def grid(self) -> AppGrid:
        """Return the underlying mesh."""
        return self._grid

    def point_in_element(self, xp: float, yp: float, element_id: int) -> bool:
        """
        Check if a point is inside a specific element.

        Parameters
        ----------
        xp : float
            X coordinate of query point.
        yp : float
            Y coordinate of query point.
        element_id : int
            ID of the element to check.

        Returns
        -------
        bool
            True if point is inside or on the boundary.
        """
        x = self._elem_x[element_id]
        y = self._elem_y[element_id]
        return point_in_element(x, y, xp, yp)

    def find_element(self, xp: float, yp: float) -> int:
        """
        Find the element containing a point.

        Parameters
        ----------
        xp : float
            X coordinate of query point.
        yp : float
            Y coordinate of query point.

        Returns
        -------
        int
            Element ID, or 0 if not found.
        """
        for eid in self._elem_ids:
            x = self._elem_x[eid]
            y = self._elem_y[eid]
            if point_in_element(x, y, xp, yp):
                return eid
        return 0

    def interpolate(self, xp: float, yp: float) -> tuple[int, tuple[int, ...], NDArray[np.float64]]:
        """
        Find containing element and compute interpolation coefficients.

        Parameters
        ----------
        xp : float
            X coordinate of interpolation point.
        yp : float
            Y coordinate of interpolation point.

        Returns
        -------
        tuple
            (element_id, node_ids, coefficients)
            element_id is 0 if point not found.
        """
        elem_id = self.find_element(xp, yp)

        if elem_id == 0:
            return 0, (), np.array([])

        x = self._elem_x[elem_id]
        y = self._elem_y[elem_id]
        n_vertex = len(x)
        coeffs = interpolation_coefficients(n_vertex, xp, yp, x, y)

        return elem_id, self._elem_vertices[elem_id], coeffs

    def interpolate_at_element(self, xp: float, yp: float, element_id: int) -> NDArray[np.float64]:
        """
        Compute interpolation coefficients at a known element.

        Parameters
        ----------
        xp : float
            X coordinate of interpolation point.
        yp : float
            Y coordinate of interpolation point.
        element_id : int
            ID of the containing element.

        Returns
        -------
        NDArray[np.float64]
            Interpolation coefficients.
        """
        x = self._elem_x[element_id]
        y = self._elem_y[element_id]
        n_vertex = len(x)
        return interpolation_coefficients(n_vertex, xp, yp, x, y)

    def interpolate_value(
        self,
        xp: float,
        yp: float,
        node_values: dict[int, float],
    ) -> float | None:
        """
        Interpolate a value at a point.

        Parameters
        ----------
        xp : float
            X coordinate of interpolation point.
        yp : float
            Y coordinate of interpolation point.
        node_values : dict[int, float]
            Dictionary mapping node ID to value.

        Returns
        -------
        float or None
            Interpolated value, or None if point not found.
        """
        elem_id, node_ids, coeffs = self.interpolate(xp, yp)

        if elem_id == 0:
            return None

        result = 0.0
        for nid, coeff in zip(node_ids, coeffs, strict=False):
            result += coeff * node_values[nid]
        return result

    def interpolate_array(
        self,
        xp: float,
        yp: float,
        node_values: NDArray[np.float64],
    ) -> float | None:
        """
        Interpolate from a node-ordered array.

        Parameters
        ----------
        xp : float
            X coordinate of interpolation point.
        yp : float
            Y coordinate of interpolation point.
        node_values : NDArray[np.float64]
            Array of values indexed by node (assumes nodes are 1-indexed
            and array index 0 corresponds to node 1).

        Returns
        -------
        float or None
            Interpolated value, or None if point not found.
        """
        elem_id, node_ids, coeffs = self.interpolate(xp, yp)

        if elem_id == 0:
            return None

        result = 0.0
        for nid, coeff in zip(node_ids, coeffs, strict=False):
            # Assume 1-indexed nodes, array is 0-indexed
            result += coeff * node_values[nid - 1]
        return result

    def interpolate_points(
        self,
        points: NDArray[np.float64],
        node_values: dict[int, float],
    ) -> NDArray[np.float64]:
        """
        Interpolate values at multiple points.

        Parameters
        ----------
        points : NDArray[np.float64]
            Array of shape (n_points, 2) with (x, y) coordinates.
        node_values : dict[int, float]
            Dictionary mapping node ID to value.

        Returns
        -------
        NDArray[np.float64]
            Array of interpolated values. NaN for points not found.
        """
        n_points = len(points)
        results = np.full(n_points, np.nan)

        for i in range(n_points):
            xp, yp = points[i, 0], points[i, 1]
            value = self.interpolate_value(xp, yp, node_values)
            if value is not None:
                results[i] = value

        return results


@dataclass
class ParametricGrid:
    """
    Multi-layer, multi-parameter grid for spatial interpolation.

    This class stores parameter values at nodes of a parametric grid
    and provides interpolation to arbitrary points. It mirrors IWFM's
    ParametricGrid module used for aquifer parameters.

    Parameters
    ----------
    grid : AppGrid
        The mesh defining the parametric grid geometry.
    n_layers : int
        Number of vertical layers.
    n_params : int
        Number of parameters at each node/layer.

    Examples
    --------
    >>> from pyiwfm.core.mesh import AppGrid, Node, Element
    >>> from pyiwfm.core.interpolation import ParametricGrid
    >>> # Create mesh
    >>> nodes = {
    ...     1: Node(id=1, x=0.0, y=0.0),
    ...     2: Node(id=2, x=100.0, y=0.0),
    ...     3: Node(id=3, x=50.0, y=100.0),
    ... }
    >>> elements = {1: Element(id=1, vertices=(1, 2, 3))}
    >>> grid = AppGrid(nodes=nodes, elements=elements)
    >>> # Create parametric grid with 2 layers and 3 parameters
    >>> pgrid = ParametricGrid(grid, n_layers=2, n_params=3)
    >>> # Set parameter values
    >>> pgrid.set_value(node_id=1, layer=0, param=0, value=100.0)
    """

    grid: AppGrid
    n_layers: int
    n_params: int
    _values: NDArray[np.float64] = field(init=False, repr=False)
    _interpolator: FEInterpolator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize storage arrays."""
        n_nodes = len(self.grid.nodes)
        self._values = np.zeros((n_nodes, self.n_layers, self.n_params))
        self._interpolator = FEInterpolator(self.grid)
        # Build node ID to index mapping
        self._node_to_idx = {nid: idx for idx, nid in enumerate(sorted(self.grid.nodes.keys()))}
        self._idx_to_node = {idx: nid for nid, idx in self._node_to_idx.items()}

    def set_value(
        self,
        node_id: int,
        layer: int,
        param: int,
        value: float,
    ) -> None:
        """
        Set a parameter value at a specific node/layer.

        Parameters
        ----------
        node_id : int
            Node ID (1-based in IWFM convention).
        layer : int
            Layer index (0-based).
        param : int
            Parameter index (0-based).
        value : float
            Value to set.
        """
        idx = self._node_to_idx[node_id]
        self._values[idx, layer, param] = value

    def get_value(
        self,
        node_id: int,
        layer: int,
        param: int,
    ) -> float:
        """
        Get a parameter value at a specific node/layer.

        Parameters
        ----------
        node_id : int
            Node ID.
        layer : int
            Layer index (0-based).
        param : int
            Parameter index (0-based).

        Returns
        -------
        float
            Parameter value.
        """
        idx = self._node_to_idx[node_id]
        return float(self._values[idx, layer, param])

    def set_layer_values(
        self,
        layer: int,
        param: int,
        values: dict[int, float],
    ) -> None:
        """
        Set values for a parameter across all nodes in a layer.

        Parameters
        ----------
        layer : int
            Layer index (0-based).
        param : int
            Parameter index (0-based).
        values : dict[int, float]
            Dictionary mapping node ID to value.
        """
        for node_id, value in values.items():
            self.set_value(node_id, layer, param, value)

    def get_layer_values(
        self,
        layer: int,
        param: int,
    ) -> dict[int, float]:
        """
        Get values for a parameter across all nodes in a layer.

        Parameters
        ----------
        layer : int
            Layer index (0-based).
        param : int
            Parameter index (0-based).

        Returns
        -------
        dict[int, float]
            Dictionary mapping node ID to value.
        """
        result = {}
        for node_id in self.grid.nodes:
            result[node_id] = self.get_value(node_id, layer, param)
        return result

    def interpolate(
        self,
        xp: float,
        yp: float,
        layer: int,
        param: int,
    ) -> float | None:
        """
        Interpolate a parameter value at an arbitrary point.

        Parameters
        ----------
        xp : float
            X coordinate.
        yp : float
            Y coordinate.
        layer : int
            Layer index (0-based).
        param : int
            Parameter index (0-based).

        Returns
        -------
        float or None
            Interpolated value, or None if point not in grid.
        """
        elem_id, node_ids, coeffs = self._interpolator.interpolate(xp, yp)

        if elem_id == 0:
            return None

        result = 0.0
        for nid, coeff in zip(node_ids, coeffs, strict=False):
            idx = self._node_to_idx[nid]
            result += coeff * self._values[idx, layer, param]

        return result

    def interpolate_all_params(
        self,
        xp: float,
        yp: float,
        layer: int,
    ) -> NDArray[np.float64] | None:
        """
        Interpolate all parameters at a point for a specific layer.

        Parameters
        ----------
        xp : float
            X coordinate.
        yp : float
            Y coordinate.
        layer : int
            Layer index (0-based).

        Returns
        -------
        NDArray[np.float64] or None
            Array of interpolated parameter values, or None if not found.
        """
        elem_id, node_ids, coeffs = self._interpolator.interpolate(xp, yp)

        if elem_id == 0:
            return None

        result = np.zeros(self.n_params)
        for nid, coeff in zip(node_ids, coeffs, strict=False):
            idx = self._node_to_idx[nid]
            result += coeff * self._values[idx, layer, :]

        return result

    def interpolate_all_layers(
        self,
        xp: float,
        yp: float,
        param: int,
    ) -> NDArray[np.float64] | None:
        """
        Interpolate a parameter at a point for all layers.

        Parameters
        ----------
        xp : float
            X coordinate.
        yp : float
            Y coordinate.
        param : int
            Parameter index (0-based).

        Returns
        -------
        NDArray[np.float64] or None
            Array of interpolated values for each layer, or None if not found.
        """
        elem_id, node_ids, coeffs = self._interpolator.interpolate(xp, yp)

        if elem_id == 0:
            return None

        result = np.zeros(self.n_layers)
        for nid, coeff in zip(node_ids, coeffs, strict=False):
            idx = self._node_to_idx[nid]
            result += coeff * self._values[idx, :, param]

        return result
