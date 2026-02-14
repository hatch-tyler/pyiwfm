"""
Slicing controller for IWFM web visualization.

This module provides the SlicingController class for interactive
slicing of 3D meshes along various planes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import pyvista as pv


class SlicingController:
    """
    Controls for slicing the 3D model along planes.

    This class provides methods for creating slices through the model
    mesh along axis-aligned planes, arbitrary angles, and custom
    cross-sections defined by two points on the map.

    Parameters
    ----------
    mesh : pv.UnstructuredGrid
        The PyVista mesh to slice.

    Attributes
    ----------
    mesh : pv.UnstructuredGrid
        The mesh being sliced.
    bounds : tuple
        Mesh bounding box (xmin, xmax, ymin, ymax, zmin, zmax).

    Examples
    --------
    >>> slicer = SlicingController(mesh)
    >>> slice_x = slicer.slice_x(position=1000.0)
    >>> slice_z = slicer.slice_z(position=-50.0)
    """

    def __init__(self, mesh: "pv.UnstructuredGrid") -> None:
        """Initialize the slicing controller."""
        self.mesh = mesh
        self._bounds = mesh.bounds
        self._cache: dict[str, "pv.PolyData"] = {}
        self._max_cache_size = 50

    @property
    def bounds(self) -> tuple[float, float, float, float, float, float]:
        """Get mesh bounds (xmin, xmax, ymin, ymax, zmin, zmax)."""
        return self._bounds

    @property
    def x_range(self) -> tuple[float, float]:
        """Get X coordinate range."""
        return (self._bounds[0], self._bounds[1])

    @property
    def y_range(self) -> tuple[float, float]:
        """Get Y coordinate range."""
        return (self._bounds[2], self._bounds[3])

    @property
    def z_range(self) -> tuple[float, float]:
        """Get Z coordinate range."""
        return (self._bounds[4], self._bounds[5])

    @property
    def center(self) -> tuple[float, float, float]:
        """Get mesh center point."""
        return self.mesh.center

    def _get_cache_key(
        self,
        slice_type: str,
        position: float | None = None,
        normal: tuple[float, float, float] | None = None,
        start: tuple[float, float] | None = None,
        end: tuple[float, float] | None = None,
    ) -> str:
        """Generate a cache key for a slice."""
        if slice_type in ("x", "y", "z"):
            return f"{slice_type}_{position:.6f}"
        elif slice_type == "arbitrary":
            return f"arb_{normal}_{position:.6f}"
        elif slice_type == "cross_section":
            return f"cs_{start}_{end}"
        return ""

    def _add_to_cache(self, key: str, slice_mesh: "pv.PolyData") -> None:
        """Add a slice to the cache, evicting old entries if needed."""
        if len(self._cache) >= self._max_cache_size:
            # Remove oldest entry (first key)
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[key] = slice_mesh

    def slice_x(self, position: float) -> "pv.PolyData":
        """
        Create a slice perpendicular to the X axis (YZ plane).

        Parameters
        ----------
        position : float
            X coordinate for the slice plane.

        Returns
        -------
        pv.PolyData
            The slice result as a surface mesh.

        Examples
        --------
        >>> slice_mesh = slicer.slice_x(position=1500.0)
        >>> slice_mesh.plot()
        """
        cache_key = self._get_cache_key("x", position=position)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Clamp position to valid range
        position = max(self._bounds[0], min(self._bounds[1], position))

        slice_mesh = self.mesh.slice(
            normal=(1, 0, 0),
            origin=(position, self.center[1], self.center[2]),
        )

        self._add_to_cache(cache_key, slice_mesh)
        return slice_mesh

    def slice_y(self, position: float) -> "pv.PolyData":
        """
        Create a slice perpendicular to the Y axis (XZ plane).

        Parameters
        ----------
        position : float
            Y coordinate for the slice plane.

        Returns
        -------
        pv.PolyData
            The slice result as a surface mesh.

        Examples
        --------
        >>> slice_mesh = slicer.slice_y(position=2000.0)
        >>> slice_mesh.plot()
        """
        cache_key = self._get_cache_key("y", position=position)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Clamp position to valid range
        position = max(self._bounds[2], min(self._bounds[3], position))

        slice_mesh = self.mesh.slice(
            normal=(0, 1, 0),
            origin=(self.center[0], position, self.center[2]),
        )

        self._add_to_cache(cache_key, slice_mesh)
        return slice_mesh

    def slice_z(self, position: float) -> "pv.PolyData":
        """
        Create a horizontal slice at a specific elevation (XY plane).

        Parameters
        ----------
        position : float
            Z (elevation) coordinate for the slice plane.

        Returns
        -------
        pv.PolyData
            The slice result as a surface mesh.

        Examples
        --------
        >>> # Slice at -100 ft elevation
        >>> slice_mesh = slicer.slice_z(position=-100.0)
        >>> slice_mesh.plot()
        """
        cache_key = self._get_cache_key("z", position=position)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Clamp position to valid range
        position = max(self._bounds[4], min(self._bounds[5], position))

        slice_mesh = self.mesh.slice(
            normal=(0, 0, 1),
            origin=(self.center[0], self.center[1], position),
        )

        self._add_to_cache(cache_key, slice_mesh)
        return slice_mesh

    def slice_arbitrary(
        self,
        normal: tuple[float, float, float],
        origin: tuple[float, float, float] | None = None,
    ) -> "pv.PolyData":
        """
        Create a slice along an arbitrary plane.

        Parameters
        ----------
        normal : tuple[float, float, float]
            Normal vector of the slice plane.
        origin : tuple[float, float, float], optional
            Origin point of the plane. Default is mesh center.

        Returns
        -------
        pv.PolyData
            The slice result as a surface mesh.

        Examples
        --------
        >>> # Slice at 45 degrees in XZ plane
        >>> slice_mesh = slicer.slice_arbitrary(
        ...     normal=(0.707, 0, 0.707),
        ...     origin=(1000, 2000, 0),
        ... )
        """
        if origin is None:
            origin = self.center

        # Normalize the normal vector
        norm = np.array(normal)
        norm = norm / np.linalg.norm(norm)
        normal = tuple(norm.tolist())

        # Round values for cache key (avoid floating point precision issues)
        cache_key = (
            f"arb_{normal[0]:.4f}_{normal[1]:.4f}_{normal[2]:.4f}_"
            f"{origin[0]:.1f}_{origin[1]:.1f}_{origin[2]:.1f}"
        )

        if cache_key in self._cache:
            return self._cache[cache_key]

        slice_mesh = self.mesh.slice(normal=normal, origin=origin)

        self._add_to_cache(cache_key, slice_mesh)
        return slice_mesh

    def create_cross_section(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        n_samples: int = 100,
    ) -> "pv.PolyData":
        """
        Create a vertical cross-section between two map points.

        This method creates a vertical slice through the model along
        a line drawn on the map (2D view). The resulting cross-section
        shows the full depth of the model along that transect.

        Parameters
        ----------
        start : tuple[float, float]
            Starting point (x, y) on the map.
        end : tuple[float, float]
            Ending point (x, y) on the map.
        n_samples : int, optional
            Number of sample points along the transect line.
            Default is 100.

        Returns
        -------
        pv.PolyData
            The cross-section as a surface mesh.

        Examples
        --------
        >>> # Create cross-section from point A to point B
        >>> cross_section = slicer.create_cross_section(
        ...     start=(1000, 2000),
        ...     end=(5000, 3000),
        ... )
        >>> cross_section.plot()

        Notes
        -----
        The cross-section is computed by creating a vertical slice plane
        that passes through both start and end points. The plane extends
        from the top to the bottom of the mesh.
        """
        cache_key = self._get_cache_key("cross_section", start=start, end=end)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Calculate the horizontal direction vector
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = np.sqrt(dx * dx + dy * dy)

        if length < 1e-10:
            raise ValueError("Start and end points must be different")

        # Normal vector is perpendicular to the line (in XY plane) and horizontal
        # The normal points "left" when standing at start looking at end
        nx = -dy / length
        ny = dx / length
        nz = 0.0

        # Origin is at the midpoint of the line at the center elevation
        origin = (
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2,
            self.center[2],
        )

        slice_mesh = self.mesh.slice(
            normal=(nx, ny, nz),
            origin=origin,
        )

        self._add_to_cache(cache_key, slice_mesh)
        return slice_mesh

    def slice_along_polyline(
        self,
        points: list[tuple[float, float]],
        resolution: int = 100,
    ) -> "pv.PolyData":
        """
        Create a vertical cross-section along a polyline path.

        Parameters
        ----------
        points : list[tuple[float, float]]
            List of (x, y) points defining the polyline path.
        resolution : int, optional
            Number of sample points per segment. Default is 100.

        Returns
        -------
        pv.PolyData
            Combined cross-section mesh.

        Examples
        --------
        >>> # Create cross-section along a stream path
        >>> path = [(1000, 2000), (2000, 2500), (3000, 2000)]
        >>> cross_section = slicer.slice_along_polyline(path)
        """
        import pyvista as pv

        if len(points) < 2:
            raise ValueError("Need at least 2 points for a polyline")

        slices = []
        for i in range(len(points) - 1):
            start = points[i]
            end = points[i + 1]
            segment_slice = self.create_cross_section(start, end, resolution)
            if segment_slice.n_cells > 0:
                slices.append(segment_slice)

        if not slices:
            return pv.PolyData()

        # Combine all slices
        combined = slices[0]
        for s in slices[1:]:
            combined = combined.merge(s)

        return combined

    def slice_box(
        self,
        bounds: tuple[float, float, float, float, float, float] | None = None,
        invert: bool = False,
    ) -> "pv.UnstructuredGrid":
        """
        Extract a box region from the mesh.

        Parameters
        ----------
        bounds : tuple[float, float, float, float, float, float], optional
            Box bounds (xmin, xmax, ymin, ymax, zmin, zmax).
            Default is center third of the mesh.
        invert : bool, optional
            If True, extract everything outside the box instead.

        Returns
        -------
        pv.UnstructuredGrid
            Extracted mesh region.
        """
        if bounds is None:
            # Default to center third
            xmin, xmax = self.x_range
            ymin, ymax = self.y_range
            zmin, zmax = self.z_range

            dx = (xmax - xmin) / 3
            dy = (ymax - ymin) / 3
            dz = (zmax - zmin) / 3

            bounds = (
                xmin + dx,
                xmax - dx,
                ymin + dy,
                ymax - dy,
                zmin + dz,
                zmax - dz,
            )

        return self.mesh.clip_box(bounds=bounds, invert=invert)

    def slice_multiple_z(
        self,
        positions: list[float] | None = None,
        n_slices: int = 5,
    ) -> list["pv.PolyData"]:
        """
        Create multiple horizontal slices at specified elevations.

        Parameters
        ----------
        positions : list[float], optional
            Z (elevation) coordinates for slice planes.
            If None, evenly space n_slices through the mesh.
        n_slices : int, optional
            Number of slices if positions not specified. Default is 5.

        Returns
        -------
        list[pv.PolyData]
            List of slice meshes.

        Examples
        --------
        >>> # Get 5 evenly spaced horizontal slices
        >>> slices = slicer.slice_multiple_z(n_slices=5)
        >>> for i, s in enumerate(slices):
        ...     s.save(f"slice_{i}.vtk")
        """
        if positions is None:
            zmin, zmax = self.z_range
            positions = np.linspace(zmin, zmax, n_slices + 2)[1:-1].tolist()

        return [self.slice_z(pos) for pos in positions]

    def get_slice_properties(self, slice_mesh: "pv.PolyData") -> dict:
        """
        Get properties of a slice result.

        Parameters
        ----------
        slice_mesh : pv.PolyData
            A slice mesh.

        Returns
        -------
        dict
            Dictionary with slice properties:
            - n_cells: Number of cells in the slice
            - n_points: Number of points in the slice
            - area: Total area of the slice
            - bounds: Bounding box of the slice
            - arrays: Available data arrays
        """
        return {
            "n_cells": slice_mesh.n_cells,
            "n_points": slice_mesh.n_points,
            "area": slice_mesh.area if slice_mesh.n_cells > 0 else 0.0,
            "bounds": slice_mesh.bounds if slice_mesh.n_cells > 0 else None,
            "cell_arrays": list(slice_mesh.cell_data.keys()),
            "point_arrays": list(slice_mesh.point_data.keys()),
        }

    def normalized_to_position_along(
        self,
        normal: tuple[float, float, float],
        normalized: float,
    ) -> tuple[float, float, float]:
        """
        Convert a 0-1 normalized position to world coordinates along an arbitrary normal.

        Projects the mesh bounding box onto the given normal direction and
        interpolates between the minimum and maximum projections.

        Parameters
        ----------
        normal : tuple[float, float, float]
            The normal direction to project along.
        normalized : float
            Normalized position (0-1) along the normal direction.

        Returns
        -------
        tuple[float, float, float]
            World-space origin point for a slice at the given position.
        """
        bounds = self._bounds
        center = np.array([
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2,
        ])
        # All eight corners of the bounding box
        corners = np.array([
            [bounds[i], bounds[j], bounds[k]]
            for i in (0, 1) for j in (2, 3) for k in (4, 5)
        ])
        normal_arr = np.array(normal, dtype=float)
        norm_len = np.linalg.norm(normal_arr)
        if norm_len < 1e-10:
            return tuple(center.tolist())
        normal_arr = normal_arr / norm_len

        projections = corners @ normal_arr
        proj_min, proj_max = projections.min(), projections.max()
        # Interpolate
        proj_val = proj_min + (proj_max - proj_min) * normalized
        # Offset center along normal to reach proj_val
        center_proj = center @ normal_arr
        offset = proj_val - center_proj
        origin = center + offset * normal_arr
        return tuple(origin.tolist())

    def clear_cache(self) -> None:
        """Clear the slice cache."""
        self._cache.clear()

    def position_to_normalized(
        self,
        axis: str,
        position: float,
    ) -> float:
        """
        Convert absolute position to normalized (0-1) range.

        Parameters
        ----------
        axis : str
            Axis name ('x', 'y', or 'z').
        position : float
            Absolute position value.

        Returns
        -------
        float
            Normalized position (0-1).
        """
        if axis == "x":
            vmin, vmax = self.x_range
        elif axis == "y":
            vmin, vmax = self.y_range
        elif axis == "z":
            vmin, vmax = self.z_range
        else:
            raise ValueError(f"Unknown axis: {axis}")

        if vmax - vmin < 1e-10:
            return 0.5

        return (position - vmin) / (vmax - vmin)

    def normalized_to_position(
        self,
        axis: str,
        normalized: float,
    ) -> float:
        """
        Convert normalized (0-1) position to absolute value.

        Parameters
        ----------
        axis : str
            Axis name ('x', 'y', or 'z').
        normalized : float
            Normalized position (0-1).

        Returns
        -------
        float
            Absolute position value.
        """
        if axis == "x":
            vmin, vmax = self.x_range
        elif axis == "y":
            vmin, vmax = self.y_range
        elif axis == "z":
            vmin, vmax = self.z_range
        else:
            raise ValueError(f"Unknown axis: {axis}")

        return vmin + normalized * (vmax - vmin)
