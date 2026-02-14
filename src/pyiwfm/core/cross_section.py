"""
Cross-section extraction along arbitrary lines through IWFM meshes.

This module provides FE-interpolated cross-sections along any line or polyline
through the model domain. Stratigraphy and hydraulic properties are interpolated
at evenly-spaced sample points using proper finite element shape functions.

Classes
-------
- :class:`CrossSection`: Dataclass storing cross-section extraction results
- :class:`CrossSectionExtractor`: Extracts cross-sections from mesh + stratigraphy

Example
-------
Extract a cross-section along a diagonal line:

>>> from pyiwfm.core.mesh import AppGrid, Node, Element
>>> from pyiwfm.core.stratigraphy import Stratigraphy
>>> from pyiwfm.core.cross_section import CrossSectionExtractor
>>> # ... create mesh and stratigraphy ...
>>> extractor = CrossSectionExtractor(grid, strat)
>>> xs = extractor.extract(start=(0, 500), end=(10000, 500), n_samples=100)
>>> xs.total_length
10000.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from pyiwfm.core.interpolation import (
    FEInterpolator,
    interpolation_coefficients,
)

if TYPE_CHECKING:
    from pyiwfm.core.mesh import AppGrid
    from pyiwfm.core.stratigraphy import Stratigraphy


@dataclass
class CrossSection:
    """
    Result of extracting a cross-section through an IWFM mesh.

    Attributes
    ----------
    distance : NDArray[np.float64]
        Cumulative distance along the profile from start, shape ``(n_samples,)``.
    x : NDArray[np.float64]
        Map X coordinates of each sample point, shape ``(n_samples,)``.
    y : NDArray[np.float64]
        Map Y coordinates of each sample point, shape ``(n_samples,)``.
    gs_elev : NDArray[np.float64]
        Interpolated ground surface elevation, shape ``(n_samples,)``.
        NaN where sample is outside the mesh.
    top_elev : NDArray[np.float64]
        Interpolated layer top elevations, shape ``(n_samples, n_layers)``.
    bottom_elev : NDArray[np.float64]
        Interpolated layer bottom elevations, shape ``(n_samples, n_layers)``.
    mask : NDArray[np.bool_]
        True where the sample point is inside the mesh domain, shape ``(n_samples,)``.
    n_layers : int
        Number of stratigraphic layers.
    start : tuple[float, float]
        Start point ``(x, y)`` of the cross-section line.
    end : tuple[float, float]
        End point ``(x, y)`` of the cross-section line.
    waypoints : list[tuple[float, float]] | None
        Polyline waypoints, or None for a simple two-point line.
    scalar_values : dict[str, NDArray[np.float64]]
        Named scalar fields interpolated onto the profile.
        Each value has shape ``(n_samples,)``.
    layer_properties : dict[str, NDArray[np.float64]]
        Named per-layer properties interpolated onto the profile.
        Each value has shape ``(n_samples, n_layers)``.
    """

    distance: NDArray[np.float64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    gs_elev: NDArray[np.float64]
    top_elev: NDArray[np.float64]
    bottom_elev: NDArray[np.float64]
    mask: NDArray[np.bool_]
    n_layers: int
    start: tuple[float, float]
    end: tuple[float, float]
    waypoints: list[tuple[float, float]] | None = None
    scalar_values: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    layer_properties: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    # Internal: cached interpolation data for re-interpolation
    _interp_cache: list[tuple[tuple[int, ...], NDArray[np.float64]] | None] = field(
        default_factory=list, repr=False
    )

    @property
    def total_length(self) -> float:
        """Total length of the cross-section profile."""
        return float(self.distance[-1])

    @property
    def n_samples(self) -> int:
        """Number of sample points along the profile."""
        return len(self.distance)

    @property
    def fraction_inside(self) -> float:
        """Fraction of sample points inside the mesh domain."""
        return float(np.sum(self.mask)) / len(self.mask)

    def get_layer_top(self, layer: int) -> NDArray[np.float64]:
        """
        Get interpolated top elevation for a specific layer.

        Parameters
        ----------
        layer : int
            Layer index (0-based).

        Returns
        -------
        NDArray[np.float64]
            Top elevation at each sample point, shape ``(n_samples,)``.
        """
        return self.top_elev[:, layer]

    def get_layer_bottom(self, layer: int) -> NDArray[np.float64]:
        """
        Get interpolated bottom elevation for a specific layer.

        Parameters
        ----------
        layer : int
            Layer index (0-based).

        Returns
        -------
        NDArray[np.float64]
            Bottom elevation at each sample point, shape ``(n_samples,)``.
        """
        return self.bottom_elev[:, layer]

    def __repr__(self) -> str:
        inside_pct = self.fraction_inside * 100
        return (
            f"CrossSection(n_samples={self.n_samples}, n_layers={self.n_layers}, "
            f"length={self.total_length:.1f}, inside={inside_pct:.0f}%)"
        )


class CrossSectionExtractor:
    """
    Extract FE-interpolated cross-sections from an IWFM mesh.

    Uses finite element shape functions (barycentric for triangles,
    bilinear isoparametric for quads) to interpolate stratigraphy
    and properties at evenly-spaced sample points along arbitrary lines.

    Parameters
    ----------
    grid : AppGrid
        The mesh to extract cross-sections from.
    stratigraphy : Stratigraphy
        Vertical layering data for the mesh.

    Examples
    --------
    >>> extractor = CrossSectionExtractor(grid, strat)
    >>> xs = extractor.extract(start=(0, 500), end=(10000, 500))
    >>> xs.n_samples
    100
    """

    def __init__(self, grid: AppGrid, stratigraphy: Stratigraphy) -> None:
        self._grid = grid
        self._strat = stratigraphy
        self._interp = FEInterpolator(grid)

        # Build node_id → 0-based index mapping matching stratigraphy array layout
        sorted_ids = sorted(grid.nodes.keys())
        self._node_id_to_idx: dict[int, int] = {
            nid: idx for idx, nid in enumerate(sorted_ids)
        }

        # Ensure connectivity is computed for neighbor-walk
        grid.compute_connectivity()

    def extract(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        n_samples: int = 100,
    ) -> CrossSection:
        """
        Extract a cross-section along a straight line.

        Parameters
        ----------
        start : tuple[float, float]
            Start point ``(x, y)``.
        end : tuple[float, float]
            End point ``(x, y)``.
        n_samples : int, default 100
            Number of evenly-spaced sample points along the line.

        Returns
        -------
        CrossSection
            Interpolated cross-section data.
        """
        x0, y0 = start
        x1, y1 = end

        # Generate evenly-spaced sample points
        t = np.linspace(0.0, 1.0, n_samples)
        xs = x0 + t * (x1 - x0)
        ys = y0 + t * (y1 - y0)

        total_len = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        dist = t * total_len

        return self._extract_at_points(xs, ys, dist, start, end)

    def extract_polyline(
        self,
        waypoints: list[tuple[float, float]],
        n_samples_per_segment: int = 50,
    ) -> CrossSection:
        """
        Extract a cross-section along a multi-segment polyline.

        Samples are distributed proportionally to segment length.

        Parameters
        ----------
        waypoints : list[tuple[float, float]]
            Ordered waypoints defining the polyline (at least 2 points).
        n_samples_per_segment : int, default 50
            Number of samples per segment (scaled by relative length).

        Returns
        -------
        CrossSection
            Interpolated cross-section data with continuous distance.

        Raises
        ------
        ValueError
            If fewer than 2 waypoints are provided.
        """
        if len(waypoints) < 2:
            raise ValueError("At least 2 waypoints are required")

        # Compute segment lengths
        seg_lengths = []
        for i in range(len(waypoints) - 1):
            dx = waypoints[i + 1][0] - waypoints[i][0]
            dy = waypoints[i + 1][1] - waypoints[i][1]
            seg_lengths.append(math.sqrt(dx * dx + dy * dy))

        total_len = sum(seg_lengths)
        total_samples = n_samples_per_segment * (len(waypoints) - 1)

        # Distribute samples proportionally to segment length
        all_x: list[float] = []
        all_y: list[float] = []
        all_dist: list[float] = []
        cum_dist = 0.0

        for i, seg_len in enumerate(seg_lengths):
            n_seg = max(2, round(total_samples * seg_len / total_len))
            x0, y0 = waypoints[i]
            x1, y1 = waypoints[i + 1]

            # Avoid duplicate points at segment boundaries
            start_idx = 0 if i == 0 else 1
            t = np.linspace(0.0, 1.0, n_seg)

            for j in range(start_idx, len(t)):
                all_x.append(x0 + t[j] * (x1 - x0))
                all_y.append(y0 + t[j] * (y1 - y0))
                all_dist.append(cum_dist + t[j] * seg_len)

            cum_dist += seg_len

        xs = np.array(all_x)
        ys = np.array(all_y)
        dist = np.array(all_dist)

        return self._extract_at_points(
            xs, ys, dist, waypoints[0], waypoints[-1], waypoints=list(waypoints)
        )

    def interpolate_scalar(
        self,
        cross_section: CrossSection,
        node_values: NDArray[np.float64],
        name: str,
    ) -> NDArray[np.float64]:
        """
        Interpolate a per-node scalar field onto an existing cross-section.

        Parameters
        ----------
        cross_section : CrossSection
            Previously extracted cross-section (must have cached interpolation data).
        node_values : NDArray[np.float64]
            Scalar values at each node, shape ``(n_nodes,)``.
        name : str
            Name for the interpolated field (stored in ``cross_section.scalar_values``).

        Returns
        -------
        NDArray[np.float64]
            Interpolated values at each sample point, shape ``(n_samples,)``.
        """
        result = np.full(cross_section.n_samples, np.nan)

        for i, cache_entry in enumerate(cross_section._interp_cache):
            if cache_entry is None:
                continue
            node_ids, coeffs = cache_entry
            indices = np.array([self._node_id_to_idx[nid] for nid in node_ids])
            result[i] = float(np.dot(coeffs, node_values[indices]))

        cross_section.scalar_values[name] = result
        return result

    def interpolate_layer_property(
        self,
        cross_section: CrossSection,
        node_layer_values: NDArray[np.float64],
        name: str,
    ) -> NDArray[np.float64]:
        """
        Interpolate a per-node per-layer property onto an existing cross-section.

        Parameters
        ----------
        cross_section : CrossSection
            Previously extracted cross-section (must have cached interpolation data).
        node_layer_values : NDArray[np.float64]
            Property values at each node and layer, shape ``(n_nodes, n_layers)``.
        name : str
            Name for the interpolated property (stored in
            ``cross_section.layer_properties``).

        Returns
        -------
        NDArray[np.float64]
            Interpolated values, shape ``(n_samples, n_layers)``.
        """
        n_layers = node_layer_values.shape[1]
        result = np.full((cross_section.n_samples, n_layers), np.nan)

        for i, cache_entry in enumerate(cross_section._interp_cache):
            if cache_entry is None:
                continue
            node_ids, coeffs = cache_entry
            indices = np.array([self._node_id_to_idx[nid] for nid in node_ids])
            # coeffs @ node_layer_values[indices, :] → (n_layers,)
            result[i, :] = coeffs @ node_layer_values[indices, :]

        cross_section.layer_properties[name] = result
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_at_points(
        self,
        xs: NDArray[np.float64],
        ys: NDArray[np.float64],
        dist: NDArray[np.float64],
        start: tuple[float, float],
        end: tuple[float, float],
        waypoints: list[tuple[float, float]] | None = None,
    ) -> CrossSection:
        """Core extraction logic for a set of (x, y) sample points."""
        n = len(xs)
        n_layers = self._strat.n_layers

        gs = np.full(n, np.nan)
        top = np.full((n, n_layers), np.nan)
        bot = np.full((n, n_layers), np.nan)
        mask = np.zeros(n, dtype=bool)
        interp_cache: list[tuple[tuple[int, ...], NDArray[np.float64]] | None] = [
            None
        ] * n

        # Neighbor-walk hint: start with no previous element
        prev_elem_id = 0

        for i in range(n):
            xp, yp = float(xs[i]), float(ys[i])

            # Try neighbor-walk: check previous element, then its neighbors
            elem_id = self._find_element_with_hint(xp, yp, prev_elem_id)

            if elem_id == 0:
                continue

            prev_elem_id = elem_id

            # Get interpolation coefficients
            elem = self._grid.elements[elem_id]
            node_ids = elem.vertices
            n_vert = len(node_ids)

            ex = self._interp._elem_x[elem_id]
            ey = self._interp._elem_y[elem_id]
            coeffs = interpolation_coefficients(n_vert, xp, yp, ex, ey)

            # Map node IDs to stratigraphy array indices
            indices = np.array([self._node_id_to_idx[nid] for nid in node_ids])

            # Interpolate stratigraphy
            gs[i] = float(np.dot(coeffs, self._strat.gs_elev[indices]))
            top[i, :] = coeffs @ self._strat.top_elev[indices, :]
            bot[i, :] = coeffs @ self._strat.bottom_elev[indices, :]

            mask[i] = True
            interp_cache[i] = (node_ids, coeffs)

        return CrossSection(
            distance=dist,
            x=xs,
            y=ys,
            gs_elev=gs,
            top_elev=top,
            bottom_elev=bot,
            mask=mask,
            n_layers=n_layers,
            start=start,
            end=end,
            waypoints=waypoints,
            _interp_cache=interp_cache,
        )

    def _find_element_with_hint(
        self, xp: float, yp: float, hint_elem_id: int
    ) -> int:
        """
        Find element containing (xp, yp), trying hint element and neighbors first.

        Parameters
        ----------
        xp, yp : float
            Query point.
        hint_elem_id : int
            Element ID to try first (0 means no hint).

        Returns
        -------
        int
            Element ID, or 0 if not found.
        """
        if hint_elem_id > 0:
            # Check the hint element itself
            if self._interp.point_in_element(xp, yp, hint_elem_id):
                return hint_elem_id

            # Check neighboring elements via shared nodes
            hint_elem = self._grid.elements[hint_elem_id]
            checked = {hint_elem_id}
            for vid in hint_elem.vertices:
                node = self._grid.nodes[vid]
                for neighbor_eid in node.surrounding_elements:
                    if neighbor_eid not in checked:
                        checked.add(neighbor_eid)
                        if self._interp.point_in_element(xp, yp, neighbor_eid):
                            return neighbor_eid

        # Fall back to full search
        return self._interp.find_element(xp, yp)
