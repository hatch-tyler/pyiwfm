"""
Sample model generators for pyiwfm documentation and testing.

This module provides functions to create synthetic IWFM models for
demonstration, testing, and documentation purposes. The models are
designed to showcase pyiwfm's capabilities without requiring actual
IWFM input files.

Example
-------
>>> from pyiwfm.sample_models import create_sample_mesh, create_sample_model
>>> mesh = create_sample_mesh()
>>> print(f"Sample mesh: {mesh.n_nodes} nodes, {mesh.n_elements} elements")
Sample mesh: 100 nodes, 162 elements
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from pyiwfm.core.mesh import AppGrid, Element, Node, Subregion
from pyiwfm.core.stratigraphy import Stratigraphy
from pyiwfm.core.model import IWFMModel
from pyiwfm.core.timeseries import TimeSeries, TimeSeriesCollection

if TYPE_CHECKING:
    from pyiwfm.components.groundwater import AppGW
    from pyiwfm.components.stream import AppStream
    from pyiwfm.components.lake import AppLake
    from pyiwfm.components.rootzone import RootZone


def create_sample_mesh(
    nx: int = 10,
    ny: int = 10,
    dx: float = 1000.0,
    dy: float = 1000.0,
    n_subregions: int = 4,
) -> AppGrid:
    """
    Create a sample rectangular mesh with quadrilateral elements.

    Parameters
    ----------
    nx : int, optional
        Number of nodes in x direction. Default is 10.
    ny : int, optional
        Number of nodes in y direction. Default is 10.
    dx : float, optional
        Node spacing in x direction (model units). Default is 1000.0.
    dy : float, optional
        Node spacing in y direction (model units). Default is 1000.0.
    n_subregions : int, optional
        Number of subregions to divide the mesh into. Default is 4.

    Returns
    -------
    AppGrid
        Sample mesh with nodes, elements, subregions, and computed connectivity.

    Example
    -------
    >>> mesh = create_sample_mesh(nx=5, ny=5, dx=500.0, dy=500.0)
    >>> print(f"Nodes: {mesh.n_nodes}, Elements: {mesh.n_elements}")
    Nodes: 25, Elements: 16
    """
    # Create nodes
    nodes: dict[int, Node] = {}
    node_id = 1
    for j in range(ny):
        for i in range(nx):
            x = i * dx
            y = j * dy
            nodes[node_id] = Node(id=node_id, x=x, y=y)
            node_id += 1

    # Create quadrilateral elements
    elements: dict[int, Element] = {}
    elem_id = 1
    n_elem_x = nx - 1
    n_elem_y = ny - 1

    # Divide into subregions (grid pattern)
    n_sr_x = int(math.ceil(math.sqrt(n_subregions)))
    n_sr_y = int(math.ceil(n_subregions / n_sr_x))

    for j in range(n_elem_y):
        for i in range(n_elem_x):
            # Node IDs for this quad (counter-clockwise)
            n1 = j * nx + i + 1
            n2 = n1 + 1
            n3 = n2 + nx
            n4 = n1 + nx

            # Determine subregion
            sr_x = min(i * n_sr_x // n_elem_x, n_sr_x - 1)
            sr_y = min(j * n_sr_y // n_elem_y, n_sr_y - 1)
            subregion = sr_y * n_sr_x + sr_x + 1
            if subregion > n_subregions:
                subregion = n_subregions

            elements[elem_id] = Element(
                id=elem_id,
                vertices=(n1, n2, n3, n4),
                subregion=subregion,
            )
            elem_id += 1

    # Create subregion definitions
    subregions: dict[int, Subregion] = {}
    for sr in range(1, n_subregions + 1):
        subregions[sr] = Subregion(id=sr, name=f"Subregion {sr}")

    # Create mesh and compute connectivity
    mesh = AppGrid(nodes=nodes, elements=elements, subregions=subregions)
    mesh.compute_connectivity()
    mesh.compute_areas()

    return mesh


def create_sample_triangular_mesh(
    n_rings: int = 5,
    n_sectors: int = 12,
    radius: float = 5000.0,
    center_x: float = 5000.0,
    center_y: float = 5000.0,
    n_subregions: int = 3,
) -> AppGrid:
    """
    Create a sample radial triangular mesh.

    Parameters
    ----------
    n_rings : int, optional
        Number of concentric rings. Default is 5.
    n_sectors : int, optional
        Number of angular sectors. Default is 12.
    radius : float, optional
        Outer radius of the mesh. Default is 5000.0.
    center_x : float, optional
        X coordinate of center. Default is 5000.0.
    center_y : float, optional
        Y coordinate of center. Default is 5000.0.
    n_subregions : int, optional
        Number of subregions. Default is 3.

    Returns
    -------
    AppGrid
        Sample triangular mesh.
    """
    nodes: dict[int, Node] = {}
    elements: dict[int, Element] = {}

    # Center node
    nodes[1] = Node(id=1, x=center_x, y=center_y)
    node_id = 2

    # Create nodes in concentric rings
    for ring in range(1, n_rings + 1):
        r = ring * radius / n_rings
        for sector in range(n_sectors):
            angle = 2 * math.pi * sector / n_sectors
            x = center_x + r * math.cos(angle)
            y = center_y + r * math.sin(angle)
            nodes[node_id] = Node(id=node_id, x=x, y=y)
            node_id += 1

    # Create triangular elements
    elem_id = 1

    # Inner triangles (connecting to center)
    for sector in range(n_sectors):
        n1 = 1  # center
        n2 = sector + 2
        n3 = (sector + 1) % n_sectors + 2
        subregion = sector % n_subregions + 1
        elements[elem_id] = Element(
            id=elem_id,
            vertices=(n1, n2, n3),
            subregion=subregion,
        )
        elem_id += 1

    # Outer rings (quads split into triangles)
    for ring in range(1, n_rings):
        ring_start = 2 + (ring - 1) * n_sectors
        next_ring_start = ring_start + n_sectors

        for sector in range(n_sectors):
            next_sector = (sector + 1) % n_sectors

            n1 = ring_start + sector
            n2 = ring_start + next_sector
            n3 = next_ring_start + next_sector
            n4 = next_ring_start + sector

            subregion = (sector + ring) % n_subregions + 1

            # Two triangles per quad
            elements[elem_id] = Element(
                id=elem_id,
                vertices=(n1, n2, n3),
                subregion=subregion,
            )
            elem_id += 1

            elements[elem_id] = Element(
                id=elem_id,
                vertices=(n1, n3, n4),
                subregion=subregion,
            )
            elem_id += 1

    # Create subregions
    subregions: dict[int, Subregion] = {}
    for sr in range(1, n_subregions + 1):
        subregions[sr] = Subregion(id=sr, name=f"Zone {sr}")

    mesh = AppGrid(nodes=nodes, elements=elements, subregions=subregions)
    mesh.compute_connectivity()
    mesh.compute_areas()

    return mesh


def create_sample_stratigraphy(
    mesh: AppGrid,
    n_layers: int = 3,
    surface_base: float = 100.0,
    surface_slope: tuple[float, float] = (-0.001, -0.002),
    layer_thickness: float = 50.0,
) -> Stratigraphy:
    """
    Create sample stratigraphy for a mesh.

    Parameters
    ----------
    mesh : AppGrid
        The mesh to create stratigraphy for.
    n_layers : int, optional
        Number of aquifer layers. Default is 3.
    surface_base : float, optional
        Ground surface elevation at origin. Default is 100.0.
    surface_slope : tuple of float, optional
        Slope in x and y directions. Default is (-0.001, -0.002).
    layer_thickness : float, optional
        Thickness of each layer. Default is 50.0.

    Returns
    -------
    Stratigraphy
        Sample stratigraphy with computed elevations.
    """
    n_nodes = mesh.n_nodes

    # Ground surface elevation (sloping)
    gs_elev = np.zeros(n_nodes, dtype=np.float64)
    for i, node in enumerate(mesh.nodes.values()):
        gs_elev[i] = (
            surface_base
            + surface_slope[0] * node.x
            + surface_slope[1] * node.y
        )

    # Layer elevations
    top_elev = np.zeros((n_nodes, n_layers), dtype=np.float64)
    bottom_elev = np.zeros((n_nodes, n_layers), dtype=np.float64)
    active_node = np.ones((n_nodes, n_layers), dtype=np.bool_)

    for layer in range(n_layers):
        top_elev[:, layer] = gs_elev - layer * layer_thickness
        bottom_elev[:, layer] = gs_elev - (layer + 1) * layer_thickness

    return Stratigraphy(
        n_layers=n_layers,
        n_nodes=n_nodes,
        gs_elev=gs_elev,
        top_elev=top_elev,
        bottom_elev=bottom_elev,
        active_node=active_node,
    )


def create_sample_scalar_field(
    mesh: AppGrid,
    field_type: str = "head",
    noise_level: float = 0.05,
) -> NDArray[np.float64]:
    """
    Create sample scalar field data for visualization.

    Parameters
    ----------
    mesh : AppGrid
        The mesh to create data for.
    field_type : str, optional
        Type of field to create. Options: 'head', 'drawdown', 'recharge',
        'pumping', 'subsidence'. Default is 'head'.
    noise_level : float, optional
        Amount of random noise to add (fraction). Default is 0.05.

    Returns
    -------
    NDArray[np.float64]
        Scalar values at each node.
    """
    n_nodes = mesh.n_nodes
    values = np.zeros(n_nodes, dtype=np.float64)

    # Get coordinate arrays
    x_coords = np.array([n.x for n in mesh.nodes.values()])
    y_coords = np.array([n.y for n in mesh.nodes.values()])

    # Normalize coordinates
    x_norm = (x_coords - x_coords.min()) / max(x_coords.max() - x_coords.min(), 1)
    y_norm = (y_coords - y_coords.min()) / max(y_coords.max() - y_coords.min(), 1)

    if field_type == "head":
        # Hydraulic head: generally decreasing from NE to SW
        values = 100.0 - 20.0 * x_norm - 30.0 * y_norm
        values += 5.0 * np.sin(2 * np.pi * x_norm) * np.cos(2 * np.pi * y_norm)
    elif field_type == "drawdown":
        # Pumping cone centered in domain
        cx, cy = 0.5, 0.5
        dist = np.sqrt((x_norm - cx)**2 + (y_norm - cy)**2)
        values = 15.0 * np.exp(-dist**2 / 0.1)
    elif field_type == "recharge":
        # Higher recharge in the east (foothills)
        values = 0.001 + 0.002 * x_norm + 0.001 * np.sin(4 * np.pi * y_norm)
    elif field_type == "pumping":
        # Pumping clusters
        c1, c2, c3 = (0.3, 0.4), (0.6, 0.7), (0.7, 0.3)
        for cx, cy in [c1, c2, c3]:
            dist = np.sqrt((x_norm - cx)**2 + (y_norm - cy)**2)
            values -= 500.0 * np.exp(-dist**2 / 0.02)
    elif field_type == "subsidence":
        # Subsidence cone
        cx, cy = 0.5, 0.5
        dist = np.sqrt((x_norm - cx)**2 + (y_norm - cy)**2)
        values = -2.0 * np.exp(-dist**2 / 0.15)
    else:
        # Generic field
        values = np.sin(2 * np.pi * x_norm) * np.cos(2 * np.pi * y_norm)

    # Add noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level * np.abs(values).mean(), n_nodes)
        values += noise

    return values


def create_sample_element_field(
    mesh: AppGrid,
    field_type: str = "land_use",
) -> NDArray[np.float64]:
    """
    Create sample element-centered field data.

    Parameters
    ----------
    mesh : AppGrid
        The mesh to create data for.
    field_type : str, optional
        Type of field. Options: 'land_use', 'soil_type', 'crop'.

    Returns
    -------
    NDArray[np.float64]
        Scalar values at each element.
    """
    n_elements = mesh.n_elements

    if field_type == "land_use":
        # Random land use categories (1-5)
        values = np.random.randint(1, 6, n_elements).astype(np.float64)
    elif field_type == "soil_type":
        # Soil types based on element position
        values = np.zeros(n_elements, dtype=np.float64)
        for i, elem in enumerate(mesh.elements.values()):
            # Get element centroid
            cx = sum(mesh.nodes[v].x for v in elem.vertices) / elem.n_vertices
            cy = sum(mesh.nodes[v].y for v in elem.vertices) / elem.n_vertices
            x_max = max(n.x for n in mesh.nodes.values())
            # Soil type varies with position
            values[i] = 1 + int(3 * cx / x_max)
    else:
        values = np.random.rand(n_elements)

    return values


def create_sample_timeseries(
    name: str = "Groundwater Head",
    start_date: datetime | None = None,
    n_years: int = 10,
    timestep_days: int = 1,
    seasonal: bool = True,
    trend: float = -0.5,
    noise_level: float = 0.1,
) -> TimeSeries:
    """
    Create a sample time series with realistic patterns.

    Parameters
    ----------
    name : str, optional
        Name of the time series. Default is "Groundwater Head".
    start_date : datetime, optional
        Start date. Default is Jan 1, 2015.
    n_years : int, optional
        Number of years of data. Default is 10.
    timestep_days : int, optional
        Days between samples. Default is 1.
    seasonal : bool, optional
        Include seasonal pattern. Default is True.
    trend : float, optional
        Linear trend per year. Default is -0.5.
    noise_level : float, optional
        Random noise level. Default is 0.1.

    Returns
    -------
    TimeSeries
        Sample time series with synthetic data.
    """
    if start_date is None:
        start_date = datetime(2015, 1, 1)

    n_points = int(365 * n_years / timestep_days)
    times = [start_date + timedelta(days=i * timestep_days) for i in range(n_points)]

    # Base value
    values = np.ones(n_points) * 50.0

    # Add trend
    t = np.arange(n_points) / (365 / timestep_days)
    values += trend * t

    # Add seasonal pattern
    if seasonal:
        seasonal_amplitude = 5.0
        values += seasonal_amplitude * np.sin(2 * np.pi * t)

    # Add noise
    values += np.random.normal(0, noise_level * np.abs(values).mean(), n_points)

    return TimeSeries.from_datetimes(
        times=times,
        values=values,
        name=name,
        units="ft",
    )


def create_sample_timeseries_collection(
    n_locations: int = 5,
    n_years: int = 10,
) -> TimeSeriesCollection:
    """
    Create a collection of sample time series.

    Parameters
    ----------
    n_locations : int, optional
        Number of locations. Default is 5.
    n_years : int, optional
        Number of years of data. Default is 10.

    Returns
    -------
    TimeSeriesCollection
        Collection of sample time series.
    """
    series_dict: dict[str, TimeSeries] = {}

    for i in range(n_locations):
        name = f"Well_{i+1}"
        ts = create_sample_timeseries(
            name=name,
            n_years=n_years,
            trend=-0.3 - 0.1 * i,
            seasonal=True,
            noise_level=0.05 + 0.02 * i,
        )
        series_dict[name] = ts

    return TimeSeriesCollection(
        name="Sample Wells",
        series=series_dict,
    )


def create_sample_stream_network(
    mesh: AppGrid,
) -> tuple[list[tuple[float, float]], list[tuple[int, int]]]:
    """
    Create a sample stream network for visualization.

    Creates a dendritic (tree-like) stream network that naturally avoids
    crossing reaches. Tributaries join the main channel from the sides
    and flow downstream.

    Parameters
    ----------
    mesh : AppGrid
        The mesh to create streams for.

    Returns
    -------
    tuple
        (node_coords, reach_connectivity) where node_coords is a list of
        (x, y) tuples and reach_connectivity is a list of (from_idx, to_idx)
        tuples.
    """
    # Get domain bounds
    x_coords = [n.x for n in mesh.nodes.values()]
    y_coords = [n.y for n in mesh.nodes.values()]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    dx = x_max - x_min
    dy = y_max - y_min

    # Create main channel with tributaries (dendritic network)
    stream_nodes: list[tuple[float, float]] = []
    reaches: list[tuple[int, int]] = []

    # Main channel: flows from top-center to bottom-center
    # This vertical orientation allows tributaries to join from left and right
    n_main = 12
    main_x_center = x_min + 0.5 * dx
    for i in range(n_main):
        t = i / (n_main - 1)
        # Main channel runs roughly north to south with slight meander
        x = main_x_center + 0.08 * dx * math.sin(2 * math.pi * t)
        y = y_max - 0.1 * dy - t * 0.8 * dy
        stream_nodes.append((x, y))
        if i > 0:
            reaches.append((i - 1, i))

    # Helper to add a tributary that joins the main channel
    def add_tributary(
        join_main_idx: int,
        start_x: float,
        start_y: float,
        n_nodes: int,
    ) -> None:
        """Add tributary flowing toward main channel junction point."""
        trib_start = len(stream_nodes)
        join_x, join_y = stream_nodes[join_main_idx]

        for i in range(n_nodes):
            t = i / (n_nodes - 1)
            # Linear interpolation from start to junction
            x = start_x + t * (join_x - start_x)
            y = start_y + t * (join_y - start_y)
            # Add slight curve to make it look natural
            perp_offset = 0.02 * dx * math.sin(math.pi * t)
            # Perpendicular direction
            length = math.sqrt((join_x - start_x) ** 2 + (join_y - start_y) ** 2)
            if length > 0:
                nx = -(join_y - start_y) / length
                x += perp_offset * nx
            stream_nodes.append((x, y))
            if i > 0:
                reaches.append((trib_start + i - 1, trib_start + i))

        # Connect last tributary node to main channel
        reaches.append((trib_start + n_nodes - 1, join_main_idx))

    # Tributary 1: from upper-left, joins at node 3
    add_tributary(
        join_main_idx=3,
        start_x=x_min + 0.1 * dx,
        start_y=y_max - 0.15 * dy,
        n_nodes=5,
    )

    # Tributary 2: from upper-right, joins at node 4
    add_tributary(
        join_main_idx=4,
        start_x=x_max - 0.1 * dx,
        start_y=y_max - 0.2 * dy,
        n_nodes=6,
    )

    # Tributary 3: from left side, joins at node 7
    add_tributary(
        join_main_idx=7,
        start_x=x_min + 0.05 * dx,
        start_y=y_min + 0.5 * dy,
        n_nodes=5,
    )

    # Tributary 4: from right side, joins at node 8
    add_tributary(
        join_main_idx=8,
        start_x=x_max - 0.08 * dx,
        start_y=y_min + 0.45 * dy,
        n_nodes=4,
    )

    return stream_nodes, reaches


def create_sample_budget_data() -> dict[str, dict[str, float]]:
    """
    Create sample water budget data.

    Returns
    -------
    dict
        Dictionary of budget components with inflows and outflows.
    """
    return {
        "Inflows": {
            "Recharge": 15000.0,
            "Stream Seepage": 8500.0,
            "Subsurface Inflow": 5200.0,
            "Lake Seepage": 2100.0,
            "Return Flow": 3800.0,
        },
        "Outflows": {
            "Pumping": -18500.0,
            "Stream Baseflow": -7200.0,
            "Subsurface Outflow": -4800.0,
            "ET from GW": -3100.0,
            "Lake Recharge": -1500.0,
        },
        "Storage Change": {
            "Storage": 500.0,
        },
    }


def create_sample_model(
    name: str = "Sample Model",
    nx: int = 10,
    ny: int = 10,
    n_layers: int = 3,
) -> IWFMModel:
    """
    Create a complete sample IWFMModel for documentation and testing.

    Parameters
    ----------
    name : str, optional
        Model name. Default is "Sample Model".
    nx : int, optional
        Number of nodes in x direction. Default is 10.
    ny : int, optional
        Number of nodes in y direction. Default is 10.
    n_layers : int, optional
        Number of aquifer layers. Default is 3.

    Returns
    -------
    IWFMModel
        Complete sample model with mesh, stratigraphy, and metadata.

    Example
    -------
    >>> model = create_sample_model()
    >>> print(model.summary())
    """
    mesh = create_sample_mesh(nx=nx, ny=ny)
    stratigraphy = create_sample_stratigraphy(mesh, n_layers=n_layers)

    return IWFMModel(
        name=name,
        mesh=mesh,
        stratigraphy=stratigraphy,
        metadata={
            "description": "Sample model for documentation",
            "version": "1.0",
            "units": "feet",
            "projection": "NAD83 / California Albers",
        },
    )


__all__ = [
    "create_sample_mesh",
    "create_sample_triangular_mesh",
    "create_sample_stratigraphy",
    "create_sample_scalar_field",
    "create_sample_element_field",
    "create_sample_timeseries",
    "create_sample_timeseries_collection",
    "create_sample_stream_network",
    "create_sample_budget_data",
    "create_sample_model",
]
