"""
GIS export functionality for IWFM models.

This module provides the :class:`GISExporter` class for exporting IWFM model
data to various GIS formats including GeoPackage, Shapefile, and GeoJSON.

Supported Formats
-----------------
- **GeoPackage** (.gpkg): Recommended format, single file with multiple layers
- **Shapefile** (.shp): Widely compatible but limited to 10-char field names
- **GeoJSON** (.geojson): Text-based, good for web applications

Example
-------
Export a mesh to GeoPackage:

>>> from pyiwfm.core.mesh import AppGrid, Node, Element
>>> from pyiwfm.visualization.gis_export import GISExporter
>>>
>>> # Create simple mesh
>>> nodes = {1: Node(id=1, x=0.0, y=0.0), 2: Node(id=2, x=100.0, y=0.0),
...          3: Node(id=3, x=50.0, y=100.0)}
>>> elements = {1: Element(id=1, vertices=(1, 2, 3))}
>>> grid = AppGrid(nodes=nodes, elements=elements)
>>> grid.compute_connectivity()
>>>
>>> # Export to GeoPackage
>>> exporter = GISExporter(grid=grid, crs="EPSG:26910")
>>> exporter.export_geopackage("model.gpkg")
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon, LineString

    from pyiwfm.core.mesh import AppGrid
    from pyiwfm.core.stratigraphy import Stratigraphy
    from pyiwfm.components.stream import AppStream


class GISExporter:
    """
    Export IWFM model data to GIS formats.

    This class converts model meshes, stratigraphy, and stream networks
    to GeoDataFrames that can be exported to various GIS formats.

    Parameters
    ----------
    grid : AppGrid
        Model mesh to export.
    stratigraphy : Stratigraphy, optional
        Model stratigraphy. If provided, layer elevations are added
        as attributes to the nodes GeoDataFrame.
    streams : AppStream, optional
        Stream network. If provided, enables stream layer export.
    crs : str, optional
        Coordinate reference system (e.g., 'EPSG:26910', 'EPSG:2227').
        If None, output files will have no CRS defined.

    Raises
    ------
    ImportError
        If geopandas or shapely are not installed.

    Examples
    --------
    Basic export to GeoPackage:

    >>> exporter = GISExporter(grid=grid, crs="EPSG:26910")
    >>> exporter.export_geopackage("model.gpkg")

    Export with stratigraphy data:

    >>> exporter = GISExporter(grid=grid, stratigraphy=strat, crs="EPSG:26910")
    >>> gdf = exporter.nodes_to_geodataframe()
    >>> # GeoDataFrame includes gs_elev, layer_1_top, layer_1_bottom, etc.

    Export with custom attributes:

    >>> head_data = {1: 50.0, 2: 52.0, 3: 48.0}  # node_id -> value
    >>> gdf = exporter.nodes_to_geodataframe(attributes={"head_ft": head_data})
    >>> gdf.to_file("nodes_with_heads.gpkg", driver="GPKG")

    Export to multiple formats:

    >>> exporter.export_geopackage("model.gpkg")  # GeoPackage
    >>> exporter.export_shapefiles("shapefiles/")  # Shapefiles
    >>> exporter.export_geojson("elements.geojson", layer="elements")
    """

    def __init__(
        self,
        grid: "AppGrid",
        stratigraphy: "Stratigraphy | None" = None,
        streams: "AppStream | None" = None,
        crs: str | None = None,
    ) -> None:
        """
        Initialize the GIS exporter.

        Args:
            grid: Model mesh
            stratigraphy: Model stratigraphy (optional)
            streams: Stream network (optional)
            crs: Coordinate reference system (e.g., 'EPSG:26910')
        """
        try:
            import geopandas  # noqa: F401
            from shapely.geometry import Point, Polygon, LineString  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "geopandas and shapely are required for GIS export. "
                "Install with: pip install geopandas shapely"
            ) from e

        self.grid = grid
        self.stratigraphy = stratigraphy
        self.streams = streams
        self.crs = crs

    def nodes_to_geodataframe(
        self,
        attributes: dict[str, dict[int, Any]] | None = None,
    ) -> "gpd.GeoDataFrame":
        """
        Convert mesh nodes to a GeoDataFrame.

        Args:
            attributes: Optional dict of attribute_name -> {node_id: value}

        Returns:
            GeoDataFrame with node points
        """
        import geopandas as gpd
        from shapely.geometry import Point

        data = []

        for node in self.grid.iter_nodes():
            row = {
                "node_id": node.id,
                "x": node.x,
                "y": node.y,
                "is_boundary": node.is_boundary,
                "area": node.area,
                "geometry": Point(node.x, node.y),
            }

            # Add stratigraphy data if available
            if self.stratigraphy is not None:
                idx = node.id - 1  # Convert to 0-indexed
                if 0 <= idx < self.stratigraphy.n_nodes:
                    row["gs_elev"] = float(self.stratigraphy.gs_elev[idx])
                    for layer in range(self.stratigraphy.n_layers):
                        row[f"layer_{layer + 1}_top"] = float(
                            self.stratigraphy.top_elev[idx, layer]
                        )
                        row[f"layer_{layer + 1}_bottom"] = float(
                            self.stratigraphy.bottom_elev[idx, layer]
                        )

            # Add custom attributes
            if attributes:
                for attr_name, attr_values in attributes.items():
                    if node.id in attr_values:
                        row[attr_name] = attr_values[node.id]

            data.append(row)

        gdf = gpd.GeoDataFrame(data, crs=self.crs)
        return gdf

    def elements_to_geodataframe(
        self,
        attributes: dict[str, dict[int, Any]] | None = None,
    ) -> "gpd.GeoDataFrame":
        """
        Convert mesh elements to a GeoDataFrame.

        Args:
            attributes: Optional dict of attribute_name -> {element_id: value}

        Returns:
            GeoDataFrame with element polygons
        """
        import geopandas as gpd
        from shapely.geometry import Polygon

        data = []

        for elem in self.grid.iter_elements():
            # Get vertex coordinates
            coords = []
            for vid in elem.vertices:
                node = self.grid.nodes[vid]
                coords.append((node.x, node.y))
            # Close the polygon
            coords.append(coords[0])

            row = {
                "element_id": elem.id,
                "subregion": elem.subregion,
                "n_vertices": elem.n_vertices,
                "area": elem.area,
                "geometry": Polygon(coords),
            }

            # Add custom attributes
            if attributes:
                for attr_name, attr_values in attributes.items():
                    if elem.id in attr_values:
                        row[attr_name] = attr_values[elem.id]

            data.append(row)

        gdf = gpd.GeoDataFrame(data, crs=self.crs)
        return gdf

    def streams_to_geodataframe(self) -> "gpd.GeoDataFrame":
        """
        Convert stream network to a GeoDataFrame.

        Returns:
            GeoDataFrame with stream reach linestrings
        """
        import geopandas as gpd
        from shapely.geometry import LineString

        if self.streams is None:
            return gpd.GeoDataFrame(columns=["reach_id", "name", "geometry"], crs=self.crs)

        data = []

        for reach in self.streams.iter_reaches():
            # Get node coordinates for this reach, resolving via gw_node
            coords = []
            for nid in reach.nodes:
                if nid in self.streams.nodes:
                    sn = self.streams.nodes[nid]
                    gw = getattr(sn, "gw_node", None)
                    if gw is not None and gw in self.grid.nodes:
                        gn = self.grid.nodes[gw]
                        coords.append((gn.x, gn.y))
                    elif sn.x != 0.0 or sn.y != 0.0:
                        coords.append((sn.x, sn.y))

            if len(coords) >= 2:
                row = {
                    "reach_id": reach.id,
                    "name": reach.name,
                    "n_nodes": reach.n_nodes,
                    "geometry": LineString(coords),
                }
                data.append(row)

        gdf = gpd.GeoDataFrame(data, crs=self.crs)
        return gdf

    def subregions_to_geodataframe(self) -> "gpd.GeoDataFrame":
        """
        Convert subregions to a GeoDataFrame (dissolved elements).

        Returns:
            GeoDataFrame with subregion polygons
        """
        import geopandas as gpd

        # Get elements GeoDataFrame
        elements_gdf = self.elements_to_geodataframe()

        # Dissolve by subregion
        subregions_gdf = elements_gdf.dissolve(by="subregion", as_index=False)
        subregions_gdf = subregions_gdf.rename(columns={"subregion": "subregion_id"})
        subregions_gdf = subregions_gdf[["subregion_id", "geometry"]]

        return subregions_gdf

    def boundary_to_geodataframe(self) -> "gpd.GeoDataFrame":
        """
        Extract model boundary as a GeoDataFrame.

        Returns:
            GeoDataFrame with model boundary polygon
        """
        import geopandas as gpd
        from shapely.ops import unary_union

        # Get elements and dissolve all to get boundary
        elements_gdf = self.elements_to_geodataframe()
        boundary_geom = unary_union(elements_gdf.geometry)

        gdf = gpd.GeoDataFrame(
            [{"boundary_id": 1, "geometry": boundary_geom}],
            crs=self.crs,
        )
        return gdf

    def export_geopackage(
        self,
        output_path: Path | str,
        include_streams: bool = True,
        include_subregions: bool = True,
        include_boundary: bool = True,
    ) -> None:
        """
        Export model to GeoPackage format.

        Args:
            output_path: Output file path (.gpkg)
            include_streams: Include stream network layer
            include_subregions: Include subregions layer
            include_boundary: Include boundary layer
        """
        output_path = Path(output_path)

        # Export nodes
        nodes_gdf = self.nodes_to_geodataframe()
        nodes_gdf.to_file(output_path, layer="nodes", driver="GPKG")

        # Export elements
        elements_gdf = self.elements_to_geodataframe()
        elements_gdf.to_file(output_path, layer="elements", driver="GPKG")

        # Export streams if available
        if include_streams and self.streams is not None:
            streams_gdf = self.streams_to_geodataframe()
            if len(streams_gdf) > 0:
                streams_gdf.to_file(output_path, layer="streams", driver="GPKG")

        # Export subregions
        if include_subregions:
            subregions_gdf = self.subregions_to_geodataframe()
            if len(subregions_gdf) > 0:
                subregions_gdf.to_file(output_path, layer="subregions", driver="GPKG")

        # Export boundary
        if include_boundary:
            boundary_gdf = self.boundary_to_geodataframe()
            boundary_gdf.to_file(output_path, layer="boundary", driver="GPKG")

    def _shorten_columns_for_shapefile(self, gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
        """
        Shorten column names to fit Shapefile's 10-character limit.

        Args:
            gdf: GeoDataFrame to modify

        Returns:
            GeoDataFrame with shortened column names
        """
        # Mapping of long column names to short versions (max 10 chars)
        column_map = {
            "is_boundary": "is_bndry",
            "subregion_id": "subreg_id",
            "boundary_id": "bndry_id",
        }
        rename_dict = {k: v for k, v in column_map.items() if k in gdf.columns}
        if rename_dict:
            gdf = gdf.rename(columns=rename_dict)
        return gdf

    def export_shapefiles(
        self,
        output_dir: Path | str,
        include_streams: bool = True,
        include_subregions: bool = True,
        include_boundary: bool = True,
    ) -> None:
        """
        Export model to Shapefile format.

        Creates separate shapefiles for nodes, elements, etc.

        Args:
            output_dir: Output directory
            include_streams: Include stream network shapefile
            include_subregions: Include subregions shapefile
            include_boundary: Include boundary shapefile
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export nodes
        nodes_gdf = self.nodes_to_geodataframe()
        nodes_gdf = self._shorten_columns_for_shapefile(nodes_gdf)
        nodes_gdf.to_file(output_dir / "nodes.shp")

        # Export elements
        elements_gdf = self.elements_to_geodataframe()
        elements_gdf = self._shorten_columns_for_shapefile(elements_gdf)
        elements_gdf.to_file(output_dir / "elements.shp")

        # Export streams if available
        if include_streams and self.streams is not None:
            streams_gdf = self.streams_to_geodataframe()
            if len(streams_gdf) > 0:
                streams_gdf = self._shorten_columns_for_shapefile(streams_gdf)
                streams_gdf.to_file(output_dir / "streams.shp")

        # Export subregions
        if include_subregions:
            subregions_gdf = self.subregions_to_geodataframe()
            if len(subregions_gdf) > 0:
                subregions_gdf = self._shorten_columns_for_shapefile(subregions_gdf)
                subregions_gdf.to_file(output_dir / "subregions.shp")

        # Export boundary
        if include_boundary:
            boundary_gdf = self.boundary_to_geodataframe()
            boundary_gdf = self._shorten_columns_for_shapefile(boundary_gdf)
            boundary_gdf.to_file(output_dir / "boundary.shp")

    def export_geojson(
        self,
        output_path: Path | str,
        layer: str = "elements",
    ) -> None:
        """
        Export a single layer to GeoJSON format.

        Args:
            output_path: Output file path (.geojson)
            layer: Layer to export ('nodes', 'elements', 'streams',
                   'subregions', 'boundary')
        """
        output_path = Path(output_path)

        if layer == "nodes":
            gdf = self.nodes_to_geodataframe()
        elif layer == "elements":
            gdf = self.elements_to_geodataframe()
        elif layer == "streams":
            gdf = self.streams_to_geodataframe()
        elif layer == "subregions":
            gdf = self.subregions_to_geodataframe()
        elif layer == "boundary":
            gdf = self.boundary_to_geodataframe()
        else:
            raise ValueError(f"Unknown layer: {layer}")

        gdf.to_file(output_path, driver="GeoJSON")
