"""
Stratigraphy class for IWFM model representation.

This module provides the Stratigraphy class which represents the vertical
layering structure of an IWFM groundwater model. It mirrors IWFM's
Class_Stratigraphy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pyiwfm.core.exceptions import StratigraphyError

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class Stratigraphy:
    """
    Vertical layering structure for an IWFM groundwater model.

    This class stores the elevation data for each node at each layer,
    including ground surface elevation and active/inactive flags.

    Attributes:
        n_layers: Number of aquifer layers
        n_nodes: Number of nodes in the mesh
        gs_elev: Ground surface elevation at each node (n_nodes,)
        top_elev: Top elevation of each layer at each node (n_nodes, n_layers)
        bottom_elev: Bottom elevation of each layer at each node (n_nodes, n_layers)
        active_node: Whether each node is active in each layer (n_nodes, n_layers)
    """

    n_layers: int
    n_nodes: int
    gs_elev: NDArray[np.float64]
    top_elev: NDArray[np.float64]
    bottom_elev: NDArray[np.float64]
    active_node: NDArray[np.bool_]

    @classmethod
    def from_thicknesses(
        cls,
        gs_elev: NDArray[np.float64],
        aquitard_thicknesses: NDArray[np.float64],
        aquifer_thicknesses: NDArray[np.float64],
        active_node: NDArray[np.bool_] | None = None,
    ) -> Stratigraphy:
        """
        Build a Stratigraphy from per-layer aquitard and aquifer thicknesses.

        Matches the IWFM stratigraphy-file convention where aquitard ``k``
        sits above aquifer layer ``k`` (k = 0 is the top aquitard between
        the ground surface and aquifer layer 0). Zero thicknesses are valid
        and indicate that no aquitard (or no aquifer) exists at that layer
        for that node.

        Args:
            gs_elev: Ground surface elevation at each node, shape
                ``(n_nodes,)``.
            aquitard_thicknesses: Thicknesses of each aquitard at each
                node, shape ``(n_nodes, n_layers)``.
            aquifer_thicknesses: Thicknesses of each aquifer layer at each
                node, shape ``(n_nodes, n_layers)``.
            active_node: Optional active-node flags, shape
                ``(n_nodes, n_layers)``. When ``None`` (default), layers
                with zero aquifer thickness are marked inactive.

        Returns:
            Stratigraphy object with top/bottom elevations computed from
            the input thicknesses.

        Raises:
            StratigraphyError: If input shapes are inconsistent or any
                thickness is negative.
        """
        gs_elev = np.asarray(gs_elev, dtype=np.float64)
        aquitard_thicknesses = np.asarray(aquitard_thicknesses, dtype=np.float64)
        aquifer_thicknesses = np.asarray(aquifer_thicknesses, dtype=np.float64)

        if aquitard_thicknesses.shape != aquifer_thicknesses.shape:
            raise StratigraphyError(
                f"aquitard_thicknesses shape {aquitard_thicknesses.shape} "
                f"does not match aquifer_thicknesses shape "
                f"{aquifer_thicknesses.shape}"
            )
        if aquifer_thicknesses.ndim != 2:
            raise StratigraphyError(
                f"aquifer_thicknesses must be 2D (n_nodes, n_layers); "
                f"got shape {aquifer_thicknesses.shape}"
            )
        if np.any(aquitard_thicknesses < 0) or np.any(aquifer_thicknesses < 0):
            raise StratigraphyError(
                "Negative thicknesses are not allowed in aquitard_thicknesses "
                "or aquifer_thicknesses"
            )

        n_nodes, n_layers = aquifer_thicknesses.shape
        if gs_elev.shape != (n_nodes,):
            raise StratigraphyError(
                f"gs_elev shape {gs_elev.shape} does not match expected "
                f"({n_nodes},) inferred from aquifer_thicknesses"
            )

        # Cumulative depth below ground surface at the bottom of each
        # aquifer layer: sum of all aquitards and aquifers down through
        # layer k. Top-of-layer-k depth is that sum minus aquifer k itself.
        cumulative = np.cumsum(aquitard_thicknesses + aquifer_thicknesses, axis=1)
        depth_to_layer_bottom = cumulative
        depth_to_layer_top = cumulative - aquifer_thicknesses

        top_elev = gs_elev[:, None] - depth_to_layer_top
        bottom_elev = gs_elev[:, None] - depth_to_layer_bottom

        if active_node is None:
            # IWFM convention: a layer with zero aquifer thickness at a
            # node is inactive there.
            active_node = aquifer_thicknesses > 0
        else:
            active_node = np.asarray(active_node, dtype=np.bool_)

        return cls(
            n_layers=n_layers,
            n_nodes=n_nodes,
            gs_elev=gs_elev,
            top_elev=top_elev,
            bottom_elev=bottom_elev,
            active_node=active_node,
        )

    def __post_init__(self) -> None:
        """Validate array dimensions after initialization."""
        # Validate gs_elev
        if self.gs_elev.shape != (self.n_nodes,):
            raise StratigraphyError(
                f"gs_elev shape {self.gs_elev.shape} does not match "
                f"expected ({self.n_nodes},) for n_nodes={self.n_nodes}"
            )

        # Validate 2D arrays
        expected_shape = (self.n_nodes, self.n_layers)

        if self.top_elev.shape != expected_shape:
            raise StratigraphyError(
                f"top_elev shape {self.top_elev.shape} does not match "
                f"expected {expected_shape} for n_nodes={self.n_nodes}, n_layers={self.n_layers}"
            )

        if self.bottom_elev.shape != expected_shape:
            raise StratigraphyError(
                f"bottom_elev shape {self.bottom_elev.shape} does not match "
                f"expected {expected_shape} for n_nodes={self.n_nodes}, n_layers={self.n_layers}"
            )

        if self.active_node.shape != expected_shape:
            raise StratigraphyError(
                f"active_node shape {self.active_node.shape} does not match "
                f"expected {expected_shape} for n_nodes={self.n_nodes}, n_layers={self.n_layers}"
            )

    def _check_layer_index(self, layer: int) -> None:
        """Check if layer index is valid."""
        if layer < 0 or layer >= self.n_layers:
            raise IndexError(f"Layer index {layer} out of range [0, {self.n_layers - 1}]")

    def _check_node_index(self, node_idx: int) -> None:
        """Check if node index is valid."""
        if node_idx < 0 or node_idx >= self.n_nodes:
            raise IndexError(f"Node index {node_idx} out of range [0, {self.n_nodes - 1}]")

    @property
    def n_aquitards(self) -> int:
        """
        Number of aquitards in the stratigraphy.

        The IWFM stratigraphy convention places one aquitard above each aquifer
        layer. Aquitard ``k`` sits above aquifer layer ``k``; the top aquitard
        (``k=0``) sits between the ground surface and the first aquifer top.
        Interior or top aquitards of zero thickness are valid.
        """
        return self.n_layers

    def _check_aquitard_index(self, aquitard: int) -> None:
        if aquitard < 0 or aquitard >= self.n_aquitards:
            raise IndexError(f"Aquitard index {aquitard} out of range [0, {self.n_aquitards - 1}]")

    def get_layer_thickness(self, layer: int) -> NDArray[np.float64]:
        """
        Get thickness of a layer at all nodes.

        Args:
            layer: Layer index (0-based)

        Returns:
            Array of thickness values at each node
        """
        self._check_layer_index(layer)
        return self.top_elev[:, layer] - self.bottom_elev[:, layer]

    def get_aquitard_thickness(self, aquitard: int) -> NDArray[np.float64]:
        """
        Get thickness of an aquitard at all nodes.

        Aquitard ``0`` lies between ground surface and the top of aquifer
        layer 0. Aquitard ``k`` (for ``k >= 1``) lies between the bottom of
        aquifer layer ``k-1`` and the top of aquifer layer ``k``. This matches
        the IWFM stratigraphy file ordering where ``W`` values alternate
        ``aquitard_k, aquifer_k`` for each layer.

        Args:
            aquitard: Aquitard index (0-based)

        Returns:
            Array of aquitard thickness values at each node
        """
        self._check_aquitard_index(aquitard)
        if aquitard == 0:
            return self.gs_elev - self.top_elev[:, 0]
        return self.bottom_elev[:, aquitard - 1] - self.top_elev[:, aquitard]

    def get_all_aquitard_thicknesses(self) -> NDArray[np.float64]:
        """
        Get all aquitard thicknesses at every node.

        Returns:
            Array of shape ``(n_nodes, n_aquitards)``. Column ``k`` is the
            thickness of aquitard ``k`` at each node.
        """
        result = np.empty((self.n_nodes, self.n_aquitards), dtype=np.float64)
        result[:, 0] = self.gs_elev - self.top_elev[:, 0]
        if self.n_aquitards > 1:
            result[:, 1:] = self.bottom_elev[:, :-1] - self.top_elev[:, 1:]
        return result

    def get_node_aquitards(self, node_idx: int) -> list[float]:
        """
        Get all aquitard thicknesses for a specific node.

        Args:
            node_idx: Node index (0-based)

        Returns:
            List of aquitard thicknesses (length ``n_aquitards``).
        """
        self._check_node_index(node_idx)
        thicknesses = self.get_all_aquitard_thicknesses()[node_idx, :]
        return [float(v) for v in thicknesses]

    def get_total_thickness(self) -> NDArray[np.float64]:
        """
        Get total thickness of all layers at each node.

        Returns:
            Array of total thickness values at each node
        """
        return self.top_elev[:, 0] - self.bottom_elev[:, -1]

    def get_layer_top(self, layer: int) -> NDArray[np.float64]:
        """
        Get top elevation of a layer at all nodes.

        Args:
            layer: Layer index (0-based)

        Returns:
            Array of top elevation values at each node
        """
        self._check_layer_index(layer)
        return self.top_elev[:, layer]

    def get_layer_bottom(self, layer: int) -> NDArray[np.float64]:
        """
        Get bottom elevation of a layer at all nodes.

        Args:
            layer: Layer index (0-based)

        Returns:
            Array of bottom elevation values at each node
        """
        self._check_layer_index(layer)
        return self.bottom_elev[:, layer]

    def get_node_elevations(self, node_idx: int) -> tuple[float, list[float], list[float]]:
        """
        Get all elevations for a specific node.

        Args:
            node_idx: Node index (0-based)

        Returns:
            Tuple of (ground_surface_elev, layer_tops, layer_bottoms)
        """
        self._check_node_index(node_idx)

        gs = float(self.gs_elev[node_idx])
        tops = self.top_elev[node_idx, :].tolist()
        bottoms = self.bottom_elev[node_idx, :].tolist()

        return (gs, tops, bottoms)

    def is_node_active(self, node_idx: int, layer: int) -> bool:
        """
        Check if a node is active in a specific layer.

        Args:
            node_idx: Node index (0-based)
            layer: Layer index (0-based)

        Returns:
            True if node is active in the layer
        """
        self._check_node_index(node_idx)
        self._check_layer_index(layer)
        return bool(self.active_node[node_idx, layer])

    def get_n_active_nodes(self, layer: int) -> int:
        """
        Count number of active nodes in a layer.

        Args:
            layer: Layer index (0-based)

        Returns:
            Number of active nodes
        """
        self._check_layer_index(layer)
        return int(np.sum(self.active_node[:, layer]))

    def get_elevation_at_depth(self, node_idx: int, depth: float) -> float:
        """
        Get elevation at a given depth below ground surface.

        Args:
            node_idx: Node index (0-based)
            depth: Depth below ground surface (positive value)

        Returns:
            Elevation at the given depth
        """
        self._check_node_index(node_idx)
        return float(self.gs_elev[node_idx] - depth)

    def get_layer_at_elevation(self, node_idx: int, elevation: float) -> int:
        """
        Find which layer contains a given elevation.

        Args:
            node_idx: Node index (0-based)
            elevation: Elevation to locate

        Returns:
            Layer index, or:
            - -1 if elevation is above ground surface
            - n_layers if elevation is below all layers
        """
        self._check_node_index(node_idx)

        gs = self.gs_elev[node_idx]

        # Above ground surface
        if elevation > gs:
            return -1

        # Check each layer (top to bottom)
        for layer in range(self.n_layers):
            top = self.top_elev[node_idx, layer]
            bottom = self.bottom_elev[node_idx, layer]

            # Include both top and bottom boundaries in layer
            # Convention: elevation at layer boundary belongs to upper layer
            if elevation <= top and elevation >= bottom:
                return layer

        # Below all layers
        return self.n_layers

    def validate(self) -> list[str]:
        """
        Validate stratigraphy data.

        Returns:
            List of warning messages (empty if valid)

        Raises:
            StratigraphyError: If critical validation fails
        """
        warnings: list[str] = []

        # Check for negative thickness
        for layer in range(self.n_layers):
            thickness = self.get_layer_thickness(layer)
            if np.any(thickness < 0):
                negative_nodes = np.where(thickness < 0)[0]
                raise StratigraphyError(
                    f"Layer {layer} has negative thickness at nodes: {negative_nodes.tolist()}"
                )

        # Check for layer discontinuities (gaps between layers)
        for layer in range(self.n_layers - 1):
            bottom_current = self.bottom_elev[:, layer]
            top_next = self.top_elev[:, layer + 1]

            # Allow small tolerance for floating point
            gaps = np.abs(bottom_current - top_next)
            if np.any(gaps > 1e-6):
                gap_nodes = np.where(gaps > 1e-6)[0]
                warnings.append(
                    f"Layer discontinuity between layers {layer} and {layer + 1} "
                    f"at nodes: {gap_nodes.tolist()}"
                )

        return warnings

    def copy(self) -> Stratigraphy:
        """
        Create a deep copy of the stratigraphy.

        Returns:
            New Stratigraphy object with copied data
        """
        return Stratigraphy(
            n_layers=self.n_layers,
            n_nodes=self.n_nodes,
            gs_elev=self.gs_elev.copy(),
            top_elev=self.top_elev.copy(),
            bottom_elev=self.bottom_elev.copy(),
            active_node=self.active_node.copy(),
        )

    def __repr__(self) -> str:
        return f"Stratigraphy(n_layers={self.n_layers}, n_nodes={self.n_nodes})"
