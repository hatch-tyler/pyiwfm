"""Drawdown computation from IWFM head data.

This module provides the :class:`DrawdownComputer` class for computing
drawdown (head change) relative to a reference timestep.  Drawdown is
defined as ``head(t_ref) - head(t)`` so that positive values indicate
water level decline.

Sentinel values (< -9000) used by IWFM to represent dry cells are
treated as NaN.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pyiwfm.core.mesh import AppGrid
    from pyiwfm.io.head_loader import LazyHeadDataLoader

logger = logging.getLogger(__name__)

_SENTINEL = -9000.0


class DrawdownComputer:
    """Compute drawdown (head change) relative to a reference timestep.

    Drawdown = head(t_ref) - head(t)  (positive = water level decline)

    Parameters
    ----------
    head_loader : LazyHeadDataLoader
        Loader providing lazy access to per-timestep head arrays.
    """

    def __init__(self, head_loader: LazyHeadDataLoader) -> None:
        self._loader = head_loader

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_layer_values(self, timestep: int, layer: int) -> NDArray[np.float64]:
        """Return head values for a single layer, with sentinels as NaN.

        Parameters
        ----------
        timestep : int
            0-based timestep index.
        layer : int
            1-based layer number.

        Returns
        -------
        NDArray[np.float64]
            Shape ``(n_nodes,)``.  Dry-cell sentinels replaced with NaN.
        """
        frame = self._loader.get_frame(timestep)
        layer_idx = layer - 1
        if layer_idx >= frame.shape[1]:
            raise IndexError(f"Layer {layer} out of range [1, {frame.shape[1]}]")
        col: NDArray[np.float64] = frame[:, layer_idx].copy()
        col[col < _SENTINEL] = np.nan
        return col

    def _validate_timestep(self, timestep: int) -> None:
        if timestep < 0 or timestep >= self._loader.n_frames:
            raise IndexError(f"Timestep {timestep} out of range [0, {self._loader.n_frames})")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_drawdown(
        self,
        timestep: int,
        reference_timestep: int = 0,
        layer: int = 1,
    ) -> NDArray[np.float64]:
        """Compute per-node drawdown for a single timestep vs reference.

        Parameters
        ----------
        timestep : int
            0-based target timestep index.
        reference_timestep : int
            0-based reference timestep index.  Default is the first
            timestep (0).
        layer : int
            1-based layer number.

        Returns
        -------
        NDArray[np.float64]
            Array of shape ``(n_nodes,)``.  Positive values mean
            water level decline.  NaN where either timestep has a
            dry-cell sentinel.
        """
        self._validate_timestep(timestep)
        self._validate_timestep(reference_timestep)

        ref = self._get_layer_values(reference_timestep, layer)
        cur = self._get_layer_values(timestep, layer)

        drawdown: NDArray[np.float64] = ref - cur
        return drawdown

    def compute_drawdown_by_element(
        self,
        timestep: int,
        reference_timestep: int = 0,
        layer: int = 1,
        grid: AppGrid | None = None,
    ) -> NDArray[np.float64]:
        """Compute per-element drawdown (average of element node values).

        Each element's drawdown is the arithmetic mean of its vertex
        node drawdown values.

        Parameters
        ----------
        timestep : int
            0-based target timestep index.
        reference_timestep : int
            0-based reference timestep index.
        layer : int
            1-based layer number.
        grid : AppGrid or None
            The model grid.  Required to look up element vertices.

        Returns
        -------
        NDArray[np.float64]
            Array of shape ``(n_elements,)`` ordered by element ID.
            NaN where any contributing node is NaN.

        Raises
        ------
        ValueError
            If *grid* is ``None``.
        """
        if grid is None:
            raise ValueError("grid is required for element-level drawdown")

        node_dd = self.compute_drawdown(timestep, reference_timestep, layer)

        # Sort elements by ID for deterministic output order
        sorted_elems = sorted(grid.elements.values(), key=lambda e: e.id)
        n_elems = len(sorted_elems)
        result = np.empty(n_elems, dtype=np.float64)

        for i, elem in enumerate(sorted_elems):
            # vertices are 1-based node IDs
            indices = np.array([nid - 1 for nid in elem.vertices])
            vals = node_dd[indices]
            result[i] = float(np.nanmean(vals))

        return result

    def compute_drawdown_range(
        self,
        reference_timestep: int = 0,
        layer: int = 1,
        max_frames: int = 0,
    ) -> tuple[float, float]:
        """Compute robust min/max drawdown across all timesteps.

        Uses the 2nd and 98th percentile to exclude outliers.

        Parameters
        ----------
        reference_timestep : int
            0-based reference timestep index.
        layer : int
            1-based layer number.
        max_frames : int
            If > 0, sample at most this many evenly-spaced frames
            instead of scanning every timestep.

        Returns
        -------
        tuple[float, float]
            ``(min_drawdown, max_drawdown)`` across all (sampled)
            timesteps.  Falls back to ``(0.0, 1.0)`` if no valid
            data exists.
        """
        self._validate_timestep(reference_timestep)

        total = self._loader.n_frames
        if total == 0:
            return (0.0, 1.0)

        # Determine which frames to sample
        if max_frames > 0 and max_frames < total:
            indices = np.unique(np.linspace(0, total - 1, max_frames, dtype=int))
        else:
            indices = np.arange(total)

        ref = self._get_layer_values(reference_timestep, layer)
        all_valid: list[float] = []

        for idx in indices:
            cur = self._get_layer_values(int(idx), layer)
            dd = ref - cur
            valid = dd[np.isfinite(dd)]
            if len(valid) > 0:
                all_valid.extend(valid.tolist())

        if not all_valid:
            return (0.0, 1.0)

        arr = np.array(all_valid)
        lo = float(np.percentile(arr, 2.0))
        hi = float(np.percentile(arr, 98.0))
        return (round(lo, 3), round(hi, 3))

    def compute_max_drawdown_map(
        self,
        reference_timestep: int = 0,
        layer: int = 1,
    ) -> NDArray[np.float64]:
        """Compute maximum drawdown at each node across all timesteps.

        Parameters
        ----------
        reference_timestep : int
            0-based reference timestep index.
        layer : int
            1-based layer number.

        Returns
        -------
        NDArray[np.float64]
            Array of shape ``(n_nodes,)``.  Each entry is the maximum
            drawdown observed at that node over all timesteps.  NaN
            where the node is always dry.
        """
        self._validate_timestep(reference_timestep)

        ref = self._get_layer_values(reference_timestep, layer)
        n_nodes = len(ref)
        max_dd = np.full(n_nodes, np.nan, dtype=np.float64)

        for ts in range(self._loader.n_frames):
            cur = self._get_layer_values(ts, layer)
            dd = ref - cur
            max_dd = np.fmax(max_dd, dd)

        return max_dd
