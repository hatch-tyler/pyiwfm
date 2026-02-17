"""Parametric grid interpolation for IWFM aquifer parameters.

When NGROUP > 0 in the GW main file, aquifer parameters are defined on
a coarser parametric finite-element mesh and interpolated onto the model
nodes.  This module implements the FE interpolation algorithm used in
IWFM's ``ParametricGrid.f90``.

The parametric grid uses standard linear shape functions:
  - Triangles: barycentric (area) coordinates
  - Quadrilaterals: bilinear isoparametric mapping
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class ParamNode:
    """A parametric grid node with coordinates and parameter values.

    Attributes:
        node_id: Node identifier (1-based from file).
        x: X coordinate.
        y: Y coordinate.
        values: Parameter values, shape ``(n_layers, n_params)``.
    """

    node_id: int
    x: float
    y: float
    values: NDArray[np.float64]


@dataclass
class ParamElement:
    """A parametric grid element (triangle or quad).

    Attributes:
        elem_id: Element identifier (1-based from file).
        vertices: Indices into the ``ParametricGrid.nodes`` list (0-based).
    """

    elem_id: int
    vertices: tuple[int, ...]


class ParametricGrid:
    """Lightweight FE grid for parametric interpolation.

    Mirrors IWFM's ``ParametricGrid`` module.  Supports point-in-element
    testing, shape function evaluation, and parameter interpolation at
    arbitrary (x, y) points.

    Parameters
    ----------
    nodes : list[ParamNode]
        Parametric grid nodes with coordinates and parameter values.
    elements : list[ParamElement]
        Parametric grid elements referencing node indices.
    """

    def __init__(
        self,
        nodes: list[ParamNode],
        elements: list[ParamElement],
    ) -> None:
        self.nodes = nodes
        self.elements = elements

    def interpolate(self, x: float, y: float) -> NDArray[np.float64] | None:
        """Interpolate parameter values at point ``(x, y)``.

        Searches all elements to find the one containing the point,
        then computes shape-function-weighted parameter values.

        Returns
        -------
        NDArray or None
            Array of shape ``(n_layers, n_params)`` with interpolated
            values, or ``None`` if the point is outside the grid.
        """
        for elem in self.elements:
            verts = elem.vertices
            if len(verts) == 3:
                inside, coeffs = self._point_in_triangle(
                    x,
                    y,
                    self.nodes[verts[0]],
                    self.nodes[verts[1]],
                    self.nodes[verts[2]],
                )
            elif len(verts) == 4:
                inside, coeffs = self._point_in_quad(
                    x,
                    y,
                    self.nodes[verts[0]],
                    self.nodes[verts[1]],
                    self.nodes[verts[2]],
                    self.nodes[verts[3]],
                )
            else:
                continue

            if not inside:
                continue

            # Interpolate: value = sum(coeff[i] * node[i].values)
            result = np.zeros_like(self.nodes[verts[0]].values)
            for i, vi in enumerate(verts):
                result += coeffs[i] * self.nodes[vi].values
            return result

        return None

    @staticmethod
    def _point_in_triangle(
        x: float,
        y: float,
        v0: ParamNode,
        v1: ParamNode,
        v2: ParamNode,
    ) -> tuple[bool, tuple[float, ...]]:
        """Test if point is inside a triangle using barycentric coordinates.

        Returns ``(is_inside, (lambda0, lambda1, lambda2))``.
        """
        x0, y0 = v0.x, v0.y
        x1, y1 = v1.x, v1.y
        x2, y2 = v2.x, v2.y

        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-30:
            return False, (0.0, 0.0, 0.0)

        lam0 = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denom
        lam1 = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denom
        lam2 = 1.0 - lam0 - lam1

        tol = -1e-10
        inside = lam0 >= tol and lam1 >= tol and lam2 >= tol
        return inside, (lam0, lam1, lam2)

    @staticmethod
    def _point_in_quad(
        x: float,
        y: float,
        v0: ParamNode,
        v1: ParamNode,
        v2: ParamNode,
        v3: ParamNode,
    ) -> tuple[bool, tuple[float, ...]]:
        """Test if point is inside a quad by splitting into two triangles.

        The quad (v0, v1, v2, v3) is split along the diagonal v0-v2.
        Returns ``(is_inside, (w0, w1, w2, w3))`` where the weights
        are the bilinear shape function values.
        """
        # Triangle 1: v0, v1, v2
        x0, y0 = v0.x, v0.y
        x1, y1 = v1.x, v1.y
        x2, y2 = v2.x, v2.y
        x3, y3 = v3.x, v3.y

        denom1 = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom1) > 1e-30:
            lam0 = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denom1
            lam1 = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denom1
            lam2 = 1.0 - lam0 - lam1

            tol = -1e-10
            if lam0 >= tol and lam1 >= tol and lam2 >= tol:
                # Point is in triangle (v0, v1, v2)
                # Map barycentric to quad weights: w0=lam0, w1=lam1, w2=lam2, w3=0
                return True, (lam0, lam1, lam2, 0.0)

        # Triangle 2: v0, v2, v3
        denom2 = (y2 - y3) * (x0 - x3) + (x3 - x2) * (y0 - y3)
        if abs(denom2) > 1e-30:
            mu0 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denom2
            mu1 = ((y3 - y0) * (x - x3) + (x0 - x3) * (y - y3)) / denom2
            mu2 = 1.0 - mu0 - mu1

            tol = -1e-10
            if mu0 >= tol and mu1 >= tol and mu2 >= tol:
                # Point is in triangle (v0, v2, v3)
                # Map barycentric to quad weights: w0=mu0, w1=0, w2=mu1, w3=mu2
                return True, (mu0, 0.0, mu1, mu2)

        return False, (0.0, 0.0, 0.0, 0.0)
