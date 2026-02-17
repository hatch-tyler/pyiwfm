"""
Property visualization component for IWFM web visualization.

This module provides the PropertyVisualizer class for managing
the display of aquifer properties on the 3D mesh.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import pyvista as pv

    from pyiwfm.core.stratigraphy import Stratigraphy


# Property metadata for display
PROPERTY_INFO = {
    "layer": {
        "name": "Layer",
        "units": "",
        "description": "Model layer number",
        "cmap": "viridis",
        "log_scale": False,
    },
    "kh": {
        "name": "Horizontal Hydraulic Conductivity",
        "units": "ft/d",
        "description": "Horizontal hydraulic conductivity (Kh)",
        "cmap": "turbo",
        "log_scale": True,
    },
    "kv": {
        "name": "Vertical Hydraulic Conductivity",
        "units": "ft/d",
        "description": "Vertical hydraulic conductivity (Kv)",
        "cmap": "turbo",
        "log_scale": True,
    },
    "ss": {
        "name": "Specific Storage",
        "units": "1/ft",
        "description": "Specific storage coefficient",
        "cmap": "plasma",
        "log_scale": True,
    },
    "sy": {
        "name": "Specific Yield",
        "units": "",
        "description": "Specific yield (dimensionless)",
        "cmap": "plasma",
        "log_scale": False,
    },
    "head": {
        "name": "Hydraulic Head",
        "units": "ft",
        "description": "Simulated groundwater head",
        "cmap": "coolwarm",
        "log_scale": False,
    },
    "thickness": {
        "name": "Layer Thickness",
        "units": "ft",
        "description": "Aquifer layer thickness",
        "cmap": "viridis",
        "log_scale": False,
    },
    "top_elev": {
        "name": "Top Elevation",
        "units": "ft",
        "description": "Layer top elevation",
        "cmap": "terrain",
        "log_scale": False,
    },
    "bottom_elev": {
        "name": "Bottom Elevation",
        "units": "ft",
        "description": "Layer bottom elevation",
        "cmap": "terrain",
        "log_scale": False,
    },
}


class PropertyVisualizer:
    """
    Manages visualization of aquifer properties on the mesh.

    This class handles the selection and display of different aquifer
    properties (Kh, Kv, Ss, Sy, head) on the 3D mesh visualization.
    It manages colormaps, value ranges, and layer filtering.

    Parameters
    ----------
    mesh : pv.UnstructuredGrid
        The PyVista mesh to add properties to.
    stratigraphy : Stratigraphy, optional
        Model stratigraphy for computing layer-based properties.
    aquifer_params : object, optional
        Aquifer parameters object containing Kh, Kv, Ss, Sy values.

    Attributes
    ----------
    mesh : pv.UnstructuredGrid
        The mesh being visualized.
    active_property : str
        Currently displayed property name.
    active_layer : int
        Currently displayed layer (1-indexed, 0 for all layers).
    colormap : str
        Current colormap name.
    """

    # Default colormaps by property type
    DEFAULT_COLORMAPS = {
        "layer": "viridis",
        "kh": "turbo",
        "kv": "turbo",
        "ss": "plasma",
        "sy": "plasma",
        "head": "coolwarm",
        "thickness": "viridis",
        "top_elev": "terrain",
        "bottom_elev": "terrain",
    }

    def __init__(
        self,
        mesh: pv.UnstructuredGrid,
        stratigraphy: Stratigraphy | None = None,
        aquifer_params: Any | None = None,
    ) -> None:
        """Initialize the property visualizer."""
        self.mesh = mesh
        self.stratigraphy = stratigraphy
        self.aquifer_params = aquifer_params

        # Current state
        self._active_property = "layer"
        self._active_layer = 0  # 0 = all layers
        self._colormap = "viridis"
        self._vmin: float | None = None
        self._vmax: float | None = None
        self._auto_range = True

        # Cached property arrays
        self._property_cache: dict[str, NDArray[np.float64]] = {}

        # Initialize available properties
        self._available_properties = self._detect_available_properties()

    @property
    def available_properties(self) -> list[str]:
        """Get list of available property names."""
        return self._available_properties

    @property
    def active_property(self) -> str:
        """Get the currently active property."""
        return self._active_property

    @property
    def active_layer(self) -> int:
        """Get the currently active layer (0 = all layers)."""
        return self._active_layer

    @property
    def colormap(self) -> str:
        """Get the current colormap."""
        return self._colormap

    @property
    def active_scalars(self) -> NDArray[np.float64] | None:
        """Get the scalar array for the currently active property."""
        return self.get_property_array(self._active_property, self._active_layer)

    @property
    def value_range(self) -> tuple[float, float]:
        """Get the current value range (min, max)."""
        scalars = self.active_scalars
        if scalars is None or len(scalars) == 0:
            return (0.0, 1.0)

        if self._auto_range:
            return (float(np.nanmin(scalars)), float(np.nanmax(scalars)))
        elif self._vmin is not None and self._vmax is not None:
            return (self._vmin, self._vmax)
        else:
            return (float(np.nanmin(scalars)), float(np.nanmax(scalars)))

    def _detect_available_properties(self) -> list[str]:
        """Detect which properties are available."""
        available = ["layer"]

        # Check aquifer parameters
        # AquiferParameters uses specific_storage/specific_yield, but also
        # accept the short names ss/sy for compatibility.
        if self.aquifer_params is not None:
            if hasattr(self.aquifer_params, "kh") and self.aquifer_params.kh is not None:
                available.append("kh")
            if hasattr(self.aquifer_params, "kv") and self.aquifer_params.kv is not None:
                available.append("kv")
            ss_val = getattr(
                self.aquifer_params, "specific_storage", getattr(self.aquifer_params, "ss", None)
            )
            if ss_val is not None:
                available.append("ss")
            sy_val = getattr(
                self.aquifer_params, "specific_yield", getattr(self.aquifer_params, "sy", None)
            )
            if sy_val is not None:
                available.append("sy")

        # Check mesh arrays for head data
        if "head" in self.mesh.cell_data or "head" in self.mesh.point_data:
            available.append("head")

        # Remove duplicates and return
        return list(dict.fromkeys(available))

    def set_active_property(self, name: str) -> None:
        """
        Set the displayed property.

        Parameters
        ----------
        name : str
            Property name: 'layer', 'kh', 'kv', 'ss', 'sy', 'head',
            'thickness', 'top_elev', 'bottom_elev'.

        Raises
        ------
        ValueError
            If the property name is not recognized or not available.
        """
        if name not in self._available_properties:
            raise ValueError(
                f"Property '{name}' not available. Available: {self._available_properties}"
            )

        self._active_property = name

        # Set default colormap for this property
        if name in self.DEFAULT_COLORMAPS:
            self._colormap = self.DEFAULT_COLORMAPS[name]

        # Reset range to auto when changing property
        self._auto_range = True
        self._vmin = None
        self._vmax = None

    def set_layer(self, layer: int) -> None:
        """
        Set the active layer for display.

        Parameters
        ----------
        layer : int
            Layer number (1-indexed). Use 0 to show all layers.
        """
        if self.stratigraphy is not None:
            max_layer = self.stratigraphy.n_layers
            if layer < 0 or layer > max_layer:
                raise ValueError(f"Layer must be 0-{max_layer}, got {layer}")

        self._active_layer = layer

    def set_colormap(self, cmap: str) -> None:
        """
        Set the colormap for property display.

        Parameters
        ----------
        cmap : str
            Matplotlib colormap name (e.g., 'viridis', 'coolwarm', 'jet').
        """
        self._colormap = cmap

    def set_range(self, vmin: float, vmax: float) -> None:
        """
        Set the value range for the colormap.

        Parameters
        ----------
        vmin : float
            Minimum value.
        vmax : float
            Maximum value.
        """
        self._vmin = vmin
        self._vmax = vmax
        self._auto_range = False

    def set_auto_range(self, auto: bool = True) -> None:
        """
        Enable or disable automatic range calculation.

        Parameters
        ----------
        auto : bool
            If True, automatically calculate range from data.
        """
        self._auto_range = auto

    def get_property_array(
        self,
        name: str,
        layer: int = 0,
    ) -> NDArray[np.float64] | None:
        """
        Get the scalar array for a property.

        Parameters
        ----------
        name : str
            Property name.
        layer : int
            Layer number (0 for all layers).

        Returns
        -------
        NDArray or None
            Scalar array for the property, or None if not available.
        """
        cache_key = f"{name}_{layer}"
        if cache_key in self._property_cache:
            return self._property_cache[cache_key]

        array = self._compute_property_array(name, layer)

        if array is not None:
            self._property_cache[cache_key] = array

        return array

    def _compute_property_array(
        self,
        name: str,
        layer: int = 0,
    ) -> NDArray[np.float64] | None:
        """Compute the scalar array for a property."""
        n_cells = self.mesh.n_cells

        if name == "layer":
            if "layer" in self.mesh.cell_data:
                array = self.mesh.cell_data["layer"].astype(np.float64)
                if layer > 0:
                    # Mask cells not in the selected layer
                    mask = self.mesh.cell_data["layer"] != layer
                    array = array.copy()
                    array[mask] = np.nan
                return array
            return np.ones(n_cells)

        elif name == "thickness":
            return self._compute_thickness_array(layer)

        elif name == "top_elev":
            return self._compute_elevation_array("top", layer)

        elif name == "bottom_elev":
            return self._compute_elevation_array("bottom", layer)

        elif name in ("kh", "kv", "ss", "sy"):
            return self._compute_aquifer_param_array(name, layer)

        elif name == "head":
            if "head" in self.mesh.cell_data:
                array = self.mesh.cell_data["head"].astype(np.float64)
                if layer > 0 and "layer" in self.mesh.cell_data:
                    mask = self.mesh.cell_data["layer"] != layer
                    array = array.copy()
                    array[mask] = np.nan
                return array

        return None

    def _compute_thickness_array(self, layer: int = 0) -> NDArray[np.float64] | None:
        """Compute layer thickness array."""
        if self.stratigraphy is None:
            return None

        n_cells = self.mesh.n_cells
        n_layers = self.stratigraphy.n_layers

        # For 3D mesh, compute thickness per cell
        if "layer" in self.mesh.cell_data:
            thickness = np.zeros(n_cells)
            layer_data = self.mesh.cell_data["layer"]

            for cell_layer in range(1, n_layers + 1):
                if layer > 0 and cell_layer != layer:
                    continue

                mask = layer_data == cell_layer
                # Average thickness for this layer (simplified)
                layer_idx = cell_layer - 1
                layer_thickness = (
                    self.stratigraphy.top_elev[:, layer_idx]
                    - self.stratigraphy.bottom_elev[:, layer_idx]
                )
                avg_thickness = float(np.mean(layer_thickness))
                thickness[mask] = avg_thickness

            if layer > 0:
                # Mask cells not in selected layer
                mask = layer_data != layer
                thickness[mask] = np.nan

            return thickness

        return None

    def _compute_elevation_array(
        self,
        surface: str,
        layer: int = 0,
    ) -> NDArray[np.float64] | None:
        """Compute elevation array (top or bottom)."""
        if self.stratigraphy is None:
            return None

        n_cells = self.mesh.n_cells

        if "layer" not in self.mesh.cell_data:
            return None

        elevation = np.zeros(n_cells)
        layer_data = self.mesh.cell_data["layer"]
        n_layers = self.stratigraphy.n_layers

        for cell_layer in range(1, n_layers + 1):
            if layer > 0 and cell_layer != layer:
                continue

            mask = layer_data == cell_layer
            layer_idx = cell_layer - 1

            if surface == "top":
                elev_array = self.stratigraphy.top_elev[:, layer_idx]
            else:
                elev_array = self.stratigraphy.bottom_elev[:, layer_idx]

            avg_elev = float(np.mean(elev_array))
            elevation[mask] = avg_elev

        if layer > 0:
            mask = layer_data != layer
            elevation[mask] = np.nan

        return elevation

    def _compute_aquifer_param_array(
        self,
        param_name: str,
        layer: int = 0,
    ) -> NDArray[np.float64] | None:
        """Compute aquifer parameter array (kh, kv, ss, sy)."""
        if self.aquifer_params is None:
            return None

        # Map UI property names to AquiferParameters attribute names.
        # Try the canonical name first (specific_storage), then the short
        # name (ss) for compatibility with objects using either convention.
        attr_candidates = {
            "kh": ["kh"],
            "kv": ["kv"],
            "ss": ["specific_storage", "ss"],
            "sy": ["specific_yield", "sy"],
        }
        param_data = None
        for attr_name in attr_candidates.get(param_name, [param_name]):
            param_data = getattr(self.aquifer_params, attr_name, None)
            if param_data is not None:
                break
        if param_data is None:
            return None

        n_cells = self.mesh.n_cells

        if "layer" not in self.mesh.cell_data:
            return None

        # Assume param_data is shape (n_nodes, n_layers) or (n_elements, n_layers)
        param_array = np.zeros(n_cells)
        layer_data = self.mesh.cell_data["layer"]

        # If param_data is 2D, use layer indexing
        if param_data.ndim == 2:
            n_layers = param_data.shape[1]
            for cell_layer in range(1, n_layers + 1):
                if layer > 0 and cell_layer != layer:
                    continue

                mask = layer_data == cell_layer
                layer_idx = cell_layer - 1

                # Take mean of values for this layer
                avg_value = float(np.mean(param_data[:, layer_idx]))
                param_array[mask] = avg_value
        else:
            # 1D array - use directly
            param_array[:] = np.mean(param_data)

        if layer > 0:
            mask = layer_data != layer
            param_array[mask] = np.nan

        return param_array

    def add_head_data(
        self,
        head: NDArray[np.float64],
        layer: int | None = None,
    ) -> None:
        """
        Add head data to the mesh.

        Parameters
        ----------
        head : NDArray
            Head values. Shape should be (n_nodes,) for single layer
            or (n_nodes, n_layers) for all layers.
        layer : int, optional
            If provided, the layer for 1D head data.
        """
        if "head" not in self._available_properties:
            self._available_properties.append("head")

        # Convert to cell data if needed
        n_cells = self.mesh.n_cells

        if head.ndim == 1:
            if len(head) == n_cells:
                self.mesh.cell_data["head"] = head
            else:
                # Interpolate from nodes to cells
                # Simplified: assign mean head per layer
                self.mesh.cell_data["head"] = np.full(n_cells, np.mean(head))
        else:
            # Multi-layer head data
            cell_head = np.zeros(n_cells)
            if "layer" in self.mesh.cell_data:
                layer_data = self.mesh.cell_data["layer"]
                for lay in range(head.shape[1]):
                    mask = layer_data == (lay + 1)
                    cell_head[mask] = np.mean(head[:, lay])
            self.mesh.cell_data["head"] = cell_head

        # Clear cache
        for key in list(self._property_cache.keys()):
            if key.startswith("head"):
                del self._property_cache[key]

    def get_property_info(self, name: str | None = None) -> dict:
        """
        Get metadata about a property.

        Parameters
        ----------
        name : str, optional
            Property name. If None, returns info for active property.

        Returns
        -------
        dict
            Property metadata including name, units, description,
            recommended colormap, and log scale flag.
        """
        if name is None:
            name = self._active_property

        if name in PROPERTY_INFO:
            info = PROPERTY_INFO[name].copy()
        else:
            info = {
                "name": name,
                "units": "",
                "description": f"Custom property: {name}",
                "cmap": "viridis",
                "log_scale": False,
            }

        # Add current range
        scalars = self.get_property_array(name, self._active_layer)
        if scalars is not None:
            valid = scalars[~np.isnan(scalars)]
            if len(valid) > 0:
                info["min"] = float(np.min(valid))
                info["max"] = float(np.max(valid))
                info["mean"] = float(np.mean(valid))
            else:
                info["min"] = info["max"] = info["mean"] = 0.0
        else:
            info["min"] = info["max"] = info["mean"] = 0.0

        return info

    def get_colorbar_settings(self) -> dict:
        """
        Get settings for the colorbar display.

        Returns
        -------
        dict
            Colorbar settings including title, units, range, colormap.
        """
        info = self.get_property_info()
        vmin, vmax = self.value_range

        return {
            "title": info["name"],
            "units": info["units"],
            "cmap": self._colormap,
            "vmin": vmin,
            "vmax": vmax,
            "log_scale": info.get("log_scale", False),
        }

    def clear_cache(self) -> None:
        """Clear the property cache."""
        self._property_cache.clear()
