"""
Zone file I/O for IWFM and GeoJSON formats.

This module handles reading and writing zone definition files:

- IWFM ZBudget format (text-based element-zone assignments)
- GeoJSON format (geospatial zone boundaries)

IWFM Zone File Format
---------------------
The IWFM zone file format is used by ZBudget for water budget analysis:

```
C Comment lines start with C
1                           # ZExtent: 1=horizontal, 0=vertical
1  Sacramento Valley        # Zone ID and name
2  San Joaquin Valley
/                           # Separator
1    1                      # Element 1 -> Zone 1
2    1                      # Element 2 -> Zone 1
3    2                      # Element 3 -> Zone 2
```

Example
-------
Read zone file:

>>> from pyiwfm.io.zones import read_iwfm_zone_file, write_iwfm_zone_file
>>> zone_def = read_iwfm_zone_file("ZBudget_Zones.dat")
>>> print(f"Loaded {zone_def.n_zones} zones")

Write zone file:

>>> write_iwfm_zone_file(zone_def, "output_zones.dat")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pyiwfm.core.zones import Zone, ZoneDefinition


def read_iwfm_zone_file(
    filepath: Path | str,
    element_areas: dict[int, float] | None = None,
) -> ZoneDefinition:
    """
    Read an IWFM-format zone definition file.

    Parameters
    ----------
    filepath : Path or str
        Path to the zone file.
    element_areas : dict[int, float], optional
        Element areas for zone area calculation.

    Returns
    -------
    ZoneDefinition
        The loaded zone definition.

    Notes
    -----
    The IWFM zone file format:
    - Lines starting with 'C' or '*' are comments
    - First data line: ZExtent (1=horizontal, 0=vertical)
    - Zone definitions: ID and optional name
    - Separator: '/'
    - Element assignments: element_id  zone_id

    Examples
    --------
    >>> zone_def = read_iwfm_zone_file("zones.dat")
    >>> print(f"Loaded {zone_def.n_zones} zones with {zone_def.n_elements} elements")
    """
    filepath = Path(filepath)
    element_areas = element_areas or {}

    zones: dict[int, Zone] = {}
    zone_names: dict[int, str] = {}
    element_zone_pairs: list[tuple[int, int]] = []
    extent = "horizontal"

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # Parse states
    in_zone_defs = False
    in_element_assignments = False
    got_extent = False

    for line in lines:
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.upper().startswith("C") or line.startswith("*"):
            continue

        # Handle separator
        if line.startswith("/"):
            if in_zone_defs:
                in_zone_defs = False
                in_element_assignments = True
            continue

        # First data: ZExtent
        if not got_extent:
            try:
                z_extent = int(line.split()[0])
                extent = "horizontal" if z_extent == 1 else "vertical"
                got_extent = True
                in_zone_defs = True
                continue
            except (ValueError, IndexError):
                # Maybe no extent line; treat as zone defs
                got_extent = True
                in_zone_defs = True

        # Parse zone definitions
        if in_zone_defs and not in_element_assignments:
            parts = line.split(None, 1)  # Split on first whitespace
            if parts:
                try:
                    zone_id = int(parts[0])
                    zone_name = parts[1].strip() if len(parts) > 1 else f"Zone {zone_id}"
                    zone_names[zone_id] = zone_name
                except ValueError:
                    continue

        # Parse element assignments
        elif in_element_assignments:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    elem_id = int(parts[0])
                    zone_id = int(parts[1])
                    element_zone_pairs.append((elem_id, zone_id))
                except ValueError:
                    continue

    # Build zone definition
    zone_def = ZoneDefinition.from_element_list(
        element_zone_pairs=element_zone_pairs,
        zone_names=zone_names,
        element_areas=element_areas,
        name=filepath.stem,
        description=f"Loaded from {filepath.name}",
    )
    zone_def.extent = extent

    return zone_def


def write_iwfm_zone_file(
    zone_def: ZoneDefinition,
    filepath: Path | str,
    header_comment: str = "",
) -> None:
    """
    Write a zone definition to IWFM format file.

    Parameters
    ----------
    zone_def : ZoneDefinition
        The zone definition to write.
    filepath : Path or str
        Output file path.
    header_comment : str, optional
        Comment text for file header.

    Examples
    --------
    >>> write_iwfm_zone_file(zone_def, "output_zones.dat", header_comment="Custom zones")
    """
    filepath = Path(filepath)

    with open(filepath, "w", encoding="utf-8") as f:
        # Header comments
        f.write("C " + "=" * 70 + "\n")
        f.write(f"C Zone Definition File\n")
        if header_comment:
            f.write(f"C {header_comment}\n")
        if zone_def.name:
            f.write(f"C Name: {zone_def.name}\n")
        if zone_def.description:
            f.write(f"C Description: {zone_def.description}\n")
        f.write(f"C Zones: {zone_def.n_zones}, Elements: {zone_def.n_elements}\n")
        f.write("C " + "=" * 70 + "\n")

        # ZExtent
        z_extent = 1 if zone_def.extent == "horizontal" else 0
        f.write(f"{z_extent}                           / ZExtent: 1=horizontal, 0=vertical\n")

        # Zone definitions
        f.write("C Zone ID   Zone Name\n")
        for zone_id in sorted(zone_def.zones.keys()):
            zone = zone_def.zones[zone_id]
            f.write(f"{zone_id:<8}    {zone.name}\n")

        # Separator
        f.write("/\n")

        # Element assignments
        f.write("C Element   Zone\n")
        if zone_def.element_zones is not None:
            for elem_idx, zone_id in enumerate(zone_def.element_zones):
                if zone_id > 0:
                    elem_id = elem_idx + 1
                    f.write(f"{elem_id:<8}    {zone_id}\n")


def read_geojson_zones(
    filepath: Path | str,
    zone_id_field: str = "id",
    zone_name_field: str = "name",
    element_id_field: str = "element_id",
) -> ZoneDefinition:
    """
    Read zone definitions from a GeoJSON file.

    Parameters
    ----------
    filepath : Path or str
        Path to the GeoJSON file.
    zone_id_field : str, optional
        Property field for zone ID. Default is "id".
    zone_name_field : str, optional
        Property field for zone name. Default is "name".
    element_id_field : str, optional
        Property field for element IDs (array). Default is "element_id".

    Returns
    -------
    ZoneDefinition
        The loaded zone definition.

    Notes
    -----
    Expected GeoJSON structure:

    ```json
    {
      "type": "FeatureCollection",
      "features": [
        {
          "type": "Feature",
          "properties": {
            "id": 1,
            "name": "Zone A",
            "element_id": [1, 2, 3, 4]
          },
          "geometry": {...}
        }
      ]
    }
    ```
    """
    filepath = Path(filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    zones: dict[int, Zone] = {}
    element_zone_pairs: list[tuple[int, int]] = []

    features = geojson.get("features", [])

    for feature in features:
        props = feature.get("properties", {})

        zone_id = props.get(zone_id_field)
        if zone_id is None:
            continue

        zone_id = int(zone_id)
        zone_name = str(props.get(zone_name_field, f"Zone {zone_id}"))
        element_ids = props.get(element_id_field, [])

        if isinstance(element_ids, (list, tuple)):
            element_list = [int(e) for e in element_ids]
        else:
            element_list = []

        # Calculate area if geometry present
        area = props.get("area", 0.0)

        zones[zone_id] = Zone(
            id=zone_id,
            name=zone_name,
            elements=element_list,
            area=float(area),
        )

        for elem_id in element_list:
            element_zone_pairs.append((elem_id, zone_id))

    # Build element_zones array
    if element_zone_pairs:
        max_elem = max(e for e, z in element_zone_pairs)
        element_zones = np.zeros(max_elem, dtype=np.int32)
        for elem_id, zone_id in element_zone_pairs:
            element_zones[elem_id - 1] = zone_id
    else:
        element_zones = np.array([], dtype=np.int32)

    return ZoneDefinition(
        zones=zones,
        extent="horizontal",
        element_zones=element_zones,
        name=filepath.stem,
        description=f"Loaded from GeoJSON: {filepath.name}",
    )


def write_geojson_zones(
    zone_def: ZoneDefinition,
    filepath: Path | str,
    grid: "AppGrid" | None = None,
    include_geometry: bool = True,
) -> None:
    """
    Write zone definitions to a GeoJSON file.

    Parameters
    ----------
    zone_def : ZoneDefinition
        The zone definition to write.
    filepath : Path or str
        Output file path.
    grid : AppGrid, optional
        Model grid for computing zone geometries.
    include_geometry : bool, optional
        Include zone boundary geometries. Default is True.

    Notes
    -----
    If grid is provided and include_geometry is True, zone boundaries
    are computed as the convex hull of element centroids.
    """
    filepath = Path(filepath)

    features = []

    for zone_id in sorted(zone_def.zones.keys()):
        zone = zone_def.zones[zone_id]

        properties = {
            "id": zone.id,
            "name": zone.name,
            "element_id": zone.elements,
            "n_elements": len(zone.elements),
            "area": zone.area,
        }

        geometry = None

        if include_geometry and grid:
            # Compute zone boundary as convex hull of element centroids
            coords = []
            for elem_id in zone.elements:
                if elem_id in grid.elements:
                    cx, cy = grid.get_element_centroid(elem_id)
                    coords.append([cx, cy])

            if len(coords) >= 3:
                # Simple convex hull approximation (bounding polygon)
                try:
                    from scipy.spatial import ConvexHull
                    points = np.array(coords)
                    hull = ConvexHull(points)
                    hull_coords = [[float(points[i, 0]), float(points[i, 1])] for i in hull.vertices]
                    hull_coords.append(hull_coords[0])  # Close the polygon
                    geometry = {
                        "type": "Polygon",
                        "coordinates": [hull_coords],
                    }
                except (ImportError, Exception):
                    # Without scipy or degenerate input, use bounding box
                    points = np.array(coords)
                    xmin, ymin = points.min(axis=0)
                    xmax, ymax = points.max(axis=0)
                    geometry = {
                        "type": "Polygon",
                        "coordinates": [[
                            [float(xmin), float(ymin)],
                            [float(xmax), float(ymin)],
                            [float(xmax), float(ymax)],
                            [float(xmin), float(ymax)],
                            [float(xmin), float(ymin)],
                        ]],
                    }
            elif len(coords) > 0:
                # Single point
                geometry = {
                    "type": "Point",
                    "coordinates": coords[0],
                }

        feature = {
            "type": "Feature",
            "properties": properties,
            "geometry": geometry,
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "name": zone_def.name or "zones",
        "features": features,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)


def auto_detect_zone_file(filepath: Path | str) -> str:
    """
    Auto-detect the format of a zone file.

    Parameters
    ----------
    filepath : Path or str
        Path to the zone file.

    Returns
    -------
    str
        Detected format: "iwfm", "geojson", or "unknown".
    """
    filepath = Path(filepath)

    suffix = filepath.suffix.lower()

    if suffix in (".json", ".geojson"):
        return "geojson"

    if suffix in (".dat", ".txt", ".in"):
        return "iwfm"

    # Try to detect by content
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            first_line = f.readline().strip()

        if first_line.startswith("{"):
            return "geojson"

        return "iwfm"

    except Exception:
        return "unknown"


def read_zone_file(
    filepath: Path | str,
    element_areas: dict[int, float] | None = None,
) -> ZoneDefinition:
    """
    Read a zone file, auto-detecting the format.

    Parameters
    ----------
    filepath : Path or str
        Path to the zone file.
    element_areas : dict[int, float], optional
        Element areas for zone area calculation (IWFM format only).

    Returns
    -------
    ZoneDefinition
        The loaded zone definition.

    Raises
    ------
    ValueError
        If file format cannot be determined.

    Examples
    --------
    >>> zone_def = read_zone_file("zones.dat")  # IWFM format
    >>> zone_def = read_zone_file("zones.geojson")  # GeoJSON format
    """
    filepath = Path(filepath)
    fmt = auto_detect_zone_file(filepath)

    if fmt == "iwfm":
        return read_iwfm_zone_file(filepath, element_areas)
    elif fmt == "geojson":
        return read_geojson_zones(filepath)
    else:
        raise ValueError(f"Cannot determine zone file format: {filepath}")


def write_zone_file(
    zone_def: ZoneDefinition,
    filepath: Path | str,
    grid: "AppGrid" | None = None,
) -> None:
    """
    Write a zone file, choosing format based on extension.

    Parameters
    ----------
    zone_def : ZoneDefinition
        The zone definition to write.
    filepath : Path or str
        Output file path.
    grid : AppGrid, optional
        Model grid for GeoJSON geometry computation.

    Examples
    --------
    >>> write_zone_file(zone_def, "zones.dat")  # IWFM format
    >>> write_zone_file(zone_def, "zones.geojson", grid)  # GeoJSON format
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix in (".json", ".geojson"):
        write_geojson_zones(zone_def, filepath, grid, include_geometry=True)
    else:
        write_iwfm_zone_file(zone_def, filepath)
