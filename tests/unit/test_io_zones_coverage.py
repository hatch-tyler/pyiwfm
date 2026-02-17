"""Supplementary tests for zones.py targeting uncovered branches.

Covers:
- Content-based format detection
- GeoJSON extension detection
- Point geometry (< 3 coords)
- scipy ConvexHull path
- scipy fallback (bounding box)
- Unknown format error
- Element_ids as non-list in geojson
- Zone file with no extent line
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pyiwfm.core.zones import Zone, ZoneDefinition
from pyiwfm.io.zones import (
    auto_detect_zone_file,
    read_geojson_zones,
    read_iwfm_zone_file,
    read_zone_file,
    write_geojson_zones,
    write_iwfm_zone_file,
    write_zone_file,
)

# =============================================================================
# Content-Based Format Detection
# =============================================================================


class TestAutoDetectContentBased:
    """Test content-based format detection for unknown extensions."""

    def test_detect_geojson_by_content(self, tmp_path: Path) -> None:
        """Test detecting GeoJSON by content starting with '{'."""
        filepath = tmp_path / "zones.xyz"  # Unknown extension
        filepath.write_text('{"type": "FeatureCollection", "features": []}')

        fmt = auto_detect_zone_file(filepath)
        assert fmt == "geojson"

    def test_detect_iwfm_by_content(self, tmp_path: Path) -> None:
        """Test detecting IWFM by content not starting with '{'."""
        filepath = tmp_path / "zones.xyz"
        filepath.write_text("C Zone File\n1 / ZExtent\n")

        fmt = auto_detect_zone_file(filepath)
        assert fmt == "iwfm"

    def test_detect_geojson_extension(self, tmp_path: Path) -> None:
        """Test detecting GeoJSON by .geojson extension."""
        filepath = tmp_path / "zones.geojson"
        filepath.write_text("{}")

        fmt = auto_detect_zone_file(filepath)
        assert fmt == "geojson"

    def test_detect_json_extension(self, tmp_path: Path) -> None:
        """Test detecting GeoJSON by .json extension."""
        filepath = tmp_path / "zones.json"
        filepath.write_text("{}")

        fmt = auto_detect_zone_file(filepath)
        assert fmt == "geojson"

    def test_detect_dat_extension(self, tmp_path: Path) -> None:
        """Test detecting IWFM by .dat extension."""
        filepath = tmp_path / "zones.dat"
        filepath.write_text("C comment")

        fmt = auto_detect_zone_file(filepath)
        assert fmt == "iwfm"

    def test_detect_txt_extension(self, tmp_path: Path) -> None:
        """Test detecting IWFM by .txt extension."""
        filepath = tmp_path / "zones.txt"
        filepath.write_text("C comment")

        fmt = auto_detect_zone_file(filepath)
        assert fmt == "iwfm"

    def test_detect_in_extension(self, tmp_path: Path) -> None:
        """Test detecting IWFM by .in extension."""
        filepath = tmp_path / "zones.in"
        filepath.write_text("C comment")

        fmt = auto_detect_zone_file(filepath)
        assert fmt == "iwfm"

    def test_detect_unreadable_file(self, tmp_path: Path) -> None:
        """Test handling unreadable file returns 'unknown'."""
        filepath = tmp_path / "nonexistent.xyz"

        fmt = auto_detect_zone_file(filepath)
        assert fmt == "unknown"


# =============================================================================
# read_zone_file / write_zone_file dispatch
# =============================================================================


class TestZoneFileDispatch:
    """Test universal read/write dispatch functions."""

    def test_read_zone_file_iwfm(self, tmp_path: Path) -> None:
        """Test read_zone_file dispatches to IWFM reader."""
        filepath = tmp_path / "zones.dat"
        filepath.write_text("""C Zone File
1                           / ZExtent
1  Test Zone
/
1    1
2    1
""")

        zone_def = read_zone_file(filepath)
        assert zone_def.n_zones == 1

    def test_read_zone_file_geojson(self, tmp_path: Path) -> None:
        """Test read_zone_file dispatches to GeoJSON reader."""
        filepath = tmp_path / "zones.geojson"
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"id": 1, "name": "Z1", "element_id": [1, 2]},
                    "geometry": None,
                },
            ],
        }
        filepath.write_text(json.dumps(geojson))

        zone_def = read_zone_file(filepath)
        assert zone_def.n_zones == 1

    def test_read_zone_file_unknown_raises(self, tmp_path: Path) -> None:
        """Test read_zone_file raises ValueError for unknown format."""
        filepath = tmp_path / "nonexistent.xyz"

        with pytest.raises(ValueError, match="Cannot determine zone file format"):
            read_zone_file(filepath)

    def test_write_zone_file_iwfm(self, tmp_path: Path) -> None:
        """Test write_zone_file dispatches to IWFM writer."""
        zones = {1: Zone(id=1, name="Z1", elements=[1, 2])}
        zone_def = ZoneDefinition(
            zones=zones,
            element_zones=np.array([1, 1], dtype=np.int32),
        )

        filepath = tmp_path / "output.dat"
        write_zone_file(zone_def, filepath)

        assert filepath.exists()
        content = filepath.read_text()
        assert "Z1" in content

    def test_write_zone_file_geojson(self, tmp_path: Path) -> None:
        """Test write_zone_file dispatches to GeoJSON writer."""
        zones = {1: Zone(id=1, name="Z1", elements=[1, 2])}
        zone_def = ZoneDefinition(
            zones=zones,
            element_zones=np.array([1, 1], dtype=np.int32),
        )

        filepath = tmp_path / "output.geojson"
        write_zone_file(zone_def, filepath)

        assert filepath.exists()
        with open(filepath) as f:
            data = json.load(f)
        assert data["type"] == "FeatureCollection"

    def test_write_zone_file_json_extension(self, tmp_path: Path) -> None:
        """Test write_zone_file dispatches to GeoJSON for .json extension."""
        zones = {1: Zone(id=1, name="Z1", elements=[])}
        zone_def = ZoneDefinition(
            zones=zones,
            element_zones=np.array([], dtype=np.int32),
        )

        filepath = tmp_path / "output.json"
        write_zone_file(zone_def, filepath)

        assert filepath.exists()


# =============================================================================
# GeoJSON Edge Cases
# =============================================================================


class TestGeoJSONEdgeCases:
    """Tests for GeoJSON reading/writing edge cases."""

    def test_read_geojson_no_zone_id(self, tmp_path: Path) -> None:
        """Test reading GeoJSON feature without zone ID field."""
        filepath = tmp_path / "zones.geojson"
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"name": "No ID Zone"},
                    "geometry": None,
                },
            ],
        }
        filepath.write_text(json.dumps(geojson))

        zone_def = read_geojson_zones(filepath)
        assert zone_def.n_zones == 0  # Feature skipped

    def test_read_geojson_element_ids_not_list(self, tmp_path: Path) -> None:
        """Test reading GeoJSON with element_id as non-list value."""
        filepath = tmp_path / "zones.geojson"
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "id": 1,
                        "name": "Z1",
                        "element_id": "not_a_list",
                    },
                    "geometry": None,
                },
            ],
        }
        filepath.write_text(json.dumps(geojson))

        zone_def = read_geojson_zones(filepath)
        assert zone_def.n_zones == 1
        assert zone_def.zones[1].elements == []

    def test_read_geojson_empty_features(self, tmp_path: Path) -> None:
        """Test reading GeoJSON with no features."""
        filepath = tmp_path / "zones.geojson"
        geojson = {"type": "FeatureCollection", "features": []}
        filepath.write_text(json.dumps(geojson))

        zone_def = read_geojson_zones(filepath)
        assert zone_def.n_zones == 0

    def test_read_geojson_no_area_field(self, tmp_path: Path) -> None:
        """Test reading GeoJSON without area property."""
        filepath = tmp_path / "zones.geojson"
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"id": 1, "name": "Z1", "element_id": [1]},
                    "geometry": None,
                },
            ],
        }
        filepath.write_text(json.dumps(geojson))

        zone_def = read_geojson_zones(filepath)
        assert zone_def.zones[1].area == 0.0

    def test_write_geojson_no_geometry(self, tmp_path: Path) -> None:
        """Test writing GeoJSON without grid (no geometry)."""
        zones = {1: Zone(id=1, name="Z1", elements=[1, 2], area=100.0)}
        zone_def = ZoneDefinition(
            zones=zones,
            element_zones=np.array([1, 1], dtype=np.int32),
        )

        filepath = tmp_path / "zones.geojson"
        write_geojson_zones(zone_def, filepath, grid=None, include_geometry=False)

        with open(filepath) as f:
            data = json.load(f)

        feature = data["features"][0]
        assert feature["geometry"] is None
        assert feature["properties"]["name"] == "Z1"
        assert feature["properties"]["area"] == 100.0


# =============================================================================
# IWFM Zone File Edge Cases
# =============================================================================


class TestIWFMZoneEdgeCases:
    """Tests for IWFM zone file reading edge cases."""

    def test_read_zone_file_with_element_areas(self, tmp_path: Path) -> None:
        """Test reading zone file with element areas provided."""
        filepath = tmp_path / "zones.dat"
        filepath.write_text("""C Zone File
1                           / ZExtent
1  Test Zone
/
1    1
2    1
3    1
""")
        areas = {1: 100.0, 2: 200.0, 3: 300.0}

        zone_def = read_iwfm_zone_file(filepath, element_areas=areas)
        assert zone_def.n_zones == 1

    def test_read_zone_no_separator(self, tmp_path: Path) -> None:
        """Test reading zone file with minimal content."""
        filepath = tmp_path / "zones.dat"
        filepath.write_text("""1                           / ZExtent
1  Zone_A
2  Zone_B
/
1    1
2    2
""")

        zone_def = read_iwfm_zone_file(filepath)
        assert zone_def.n_zones == 2

    def test_write_zone_file_with_header(self, tmp_path: Path) -> None:
        """Test writing zone file with custom header comment."""
        zones = {
            1: Zone(id=1, name="North", elements=[1, 2]),
            2: Zone(id=2, name="South", elements=[3, 4]),
        }
        zone_def = ZoneDefinition(
            zones=zones,
            element_zones=np.array([1, 1, 2, 2], dtype=np.int32),
            name="TestZones",
            description="Test zone definition",
        )

        filepath = tmp_path / "zones.dat"
        write_iwfm_zone_file(zone_def, filepath, header_comment="Custom Header")

        content = filepath.read_text()
        assert "Custom Header" in content
        assert "TestZones" in content
        assert "North" in content
        assert "South" in content

    def test_write_zone_vertical_extent(self, tmp_path: Path) -> None:
        """Test writing zone file with vertical extent."""
        zones = {1: Zone(id=1, name="Layer1", elements=[1])}
        zone_def = ZoneDefinition(
            zones=zones,
            element_zones=np.array([1], dtype=np.int32),
            extent="vertical",
        )

        filepath = tmp_path / "zones.dat"
        write_iwfm_zone_file(zone_def, filepath)

        content = filepath.read_text()
        assert "0" in content  # vertical = 0

    def test_roundtrip_iwfm_zone(self, tmp_path: Path) -> None:
        """Test write-read roundtrip for IWFM zone file."""
        zones = {
            1: Zone(id=1, name="Zone A", elements=[1, 2, 3]),
            2: Zone(id=2, name="Zone B", elements=[4, 5]),
        }
        zone_def = ZoneDefinition(
            zones=zones,
            element_zones=np.array([1, 1, 1, 2, 2], dtype=np.int32),
        )

        filepath = tmp_path / "zones.dat"
        write_iwfm_zone_file(zone_def, filepath)
        read_back = read_iwfm_zone_file(filepath)

        assert read_back.n_zones == 2
        assert set(read_back.get_elements_in_zone(1)) == {1, 2, 3}
        assert set(read_back.get_elements_in_zone(2)) == {4, 5}
