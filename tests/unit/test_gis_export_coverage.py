"""Unit tests for visualization/gis_export.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

gpd = pytest.importorskip("geopandas", reason="geopandas not available")

from pyiwfm.visualization.gis_export import GISExporter
from tests.conftest import make_simple_grid, make_simple_stratigraphy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_exporter(with_strat: bool = False, with_streams: bool = False, crs: str | None = None):
    """Create a GISExporter with a simple grid."""
    grid = make_simple_grid()
    grid.compute_connectivity()
    grid.compute_areas()

    strat = make_simple_stratigraphy(n_nodes=9) if with_strat else None
    streams = MagicMock() if with_streams else None

    if with_streams:
        reach = MagicMock()
        reach.id = 1
        reach.name = "Reach 1"
        reach.n_nodes = 3
        reach.nodes = [1, 2, 3]

        node1 = MagicMock()
        node1.x, node1.y = 0.0, 0.0
        node1.gw_node = 1
        node2 = MagicMock()
        node2.x, node2.y = 100.0, 0.0
        node2.gw_node = 2
        node3 = MagicMock()
        node3.x, node3.y = 200.0, 0.0
        node3.gw_node = 3

        streams.nodes = {1: node1, 2: node2, 3: node3}
        streams.iter_reaches.return_value = [reach]

    return GISExporter(grid=grid, stratigraphy=strat, streams=streams, crs=crs)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestGISExporterInit:
    """Tests for GISExporter construction."""

    def test_basic_init(self) -> None:
        exporter = _make_exporter()
        assert exporter.grid is not None

    def test_with_crs(self) -> None:
        exporter = _make_exporter(crs="EPSG:26910")
        assert exporter.crs == "EPSG:26910"


# ---------------------------------------------------------------------------
# nodes_to_geodataframe
# ---------------------------------------------------------------------------


class TestNodesToGeoDataFrame:
    """Tests for nodes_to_geodataframe()."""

    def test_basic_nodes(self) -> None:
        exporter = _make_exporter()
        gdf = exporter.nodes_to_geodataframe()
        assert len(gdf) == 9
        assert "node_id" in gdf.columns
        assert "x" in gdf.columns
        assert "y" in gdf.columns
        assert "geometry" in gdf.columns

    def test_with_stratigraphy(self) -> None:
        exporter = _make_exporter(with_strat=True)
        gdf = exporter.nodes_to_geodataframe()
        assert "gs_elev" in gdf.columns
        assert "layer_1_top" in gdf.columns
        assert "layer_1_bottom" in gdf.columns
        assert "layer_2_top" in gdf.columns

    def test_with_custom_attributes(self) -> None:
        exporter = _make_exporter()
        attrs = {"head": {1: 50.0, 2: 52.0, 3: 48.0}}
        gdf = exporter.nodes_to_geodataframe(attributes=attrs)
        assert "head" in gdf.columns
        assert gdf.loc[gdf["node_id"] == 1, "head"].values[0] == 50.0

    def test_with_crs(self) -> None:
        exporter = _make_exporter(crs="EPSG:26910")
        gdf = exporter.nodes_to_geodataframe()
        assert gdf.crs is not None


# ---------------------------------------------------------------------------
# elements_to_geodataframe
# ---------------------------------------------------------------------------


class TestElementsToGeoDataFrame:
    """Tests for elements_to_geodataframe()."""

    def test_basic_elements(self) -> None:
        exporter = _make_exporter()
        gdf = exporter.elements_to_geodataframe()
        assert len(gdf) == 4
        assert "element_id" in gdf.columns
        assert "subregion" in gdf.columns
        assert "n_vertices" in gdf.columns
        assert "geometry" in gdf.columns

    def test_polygon_geometries(self) -> None:
        from shapely.geometry import Polygon

        exporter = _make_exporter()
        gdf = exporter.elements_to_geodataframe()
        assert all(isinstance(g, Polygon) for g in gdf.geometry)

    def test_with_custom_attributes(self) -> None:
        exporter = _make_exporter()
        attrs = {"conductivity": {1: 10.0, 2: 20.0, 3: 30.0, 4: 40.0}}
        gdf = exporter.elements_to_geodataframe(attributes=attrs)
        assert "conductivity" in gdf.columns


# ---------------------------------------------------------------------------
# streams_to_geodataframe
# ---------------------------------------------------------------------------


class TestStreamsToGeoDataFrame:
    """Tests for streams_to_geodataframe()."""

    def test_with_streams(self) -> None:
        exporter = _make_exporter(with_streams=True)
        gdf = exporter.streams_to_geodataframe()
        assert len(gdf) == 1
        assert "reach_id" in gdf.columns

    def test_without_streams(self) -> None:
        exporter = _make_exporter(with_streams=False)
        gdf = exporter.streams_to_geodataframe()
        assert len(gdf) == 0


# ---------------------------------------------------------------------------
# subregions_to_geodataframe
# ---------------------------------------------------------------------------


class TestSubregionsToGeoDataFrame:
    """Tests for subregions_to_geodataframe()."""

    def test_basic(self) -> None:
        exporter = _make_exporter()
        gdf = exporter.subregions_to_geodataframe()
        assert len(gdf) == 2  # Two subregions in the simple grid
        assert "subregion_id" in gdf.columns


# ---------------------------------------------------------------------------
# boundary_to_geodataframe
# ---------------------------------------------------------------------------


class TestBoundaryToGeoDataFrame:
    """Tests for boundary_to_geodataframe()."""

    def test_basic(self) -> None:
        exporter = _make_exporter()
        gdf = exporter.boundary_to_geodataframe()
        assert len(gdf) == 1
        assert "boundary_id" in gdf.columns


# ---------------------------------------------------------------------------
# export_geopackage
# ---------------------------------------------------------------------------


class TestExportGeoPackage:
    """Tests for export_geopackage()."""

    def test_creates_file(self, tmp_path: Path) -> None:
        exporter = _make_exporter()
        output = tmp_path / "model.gpkg"
        exporter.export_geopackage(output)
        assert output.exists()

    def test_with_all_layers(self, tmp_path: Path) -> None:
        exporter = _make_exporter(with_streams=True)
        output = tmp_path / "model.gpkg"
        exporter.export_geopackage(
            output,
            include_streams=True,
            include_subregions=True,
            include_boundary=True,
        )
        assert output.exists()


# ---------------------------------------------------------------------------
# export_geojson
# ---------------------------------------------------------------------------


class TestExportGeoJSON:
    """Tests for export_geojson()."""

    def test_nodes_layer(self, tmp_path: Path) -> None:
        exporter = _make_exporter()
        output = tmp_path / "nodes.geojson"
        exporter.export_geojson(output, layer="nodes")
        assert output.exists()

    def test_elements_layer(self, tmp_path: Path) -> None:
        exporter = _make_exporter()
        output = tmp_path / "elements.geojson"
        exporter.export_geojson(output, layer="elements")
        assert output.exists()

    def test_unknown_layer_raises(self, tmp_path: Path) -> None:
        exporter = _make_exporter()
        with pytest.raises(ValueError, match="Unknown layer"):
            exporter.export_geojson(tmp_path / "bad.geojson", layer="nonexistent")


# ---------------------------------------------------------------------------
# Column shortening for Shapefile
# ---------------------------------------------------------------------------


class TestShortenColumns:
    """Tests for _shorten_columns_for_shapefile."""

    def test_shortens_known_columns(self) -> None:
        exporter = _make_exporter()
        import pandas as pd

        gdf = gpd.GeoDataFrame({"is_boundary": [True, False], "geometry": [None, None]})
        shortened = exporter._shorten_columns_for_shapefile(gdf)
        assert "is_bndry" in shortened.columns
        assert "is_boundary" not in shortened.columns

    def test_no_change_for_short_columns(self) -> None:
        exporter = _make_exporter()
        gdf = gpd.GeoDataFrame({"x": [1.0], "y": [2.0], "geometry": [None]})
        shortened = exporter._shorten_columns_for_shapefile(gdf)
        assert list(shortened.columns) == ["x", "y", "geometry"]
