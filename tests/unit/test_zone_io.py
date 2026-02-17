"""Unit tests for zone file I/O."""

import json
import tempfile
from pathlib import Path

import numpy as np

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


class TestReadIWFMZoneFile:
    """Tests for reading IWFM zone files."""

    def test_basic_read(self):
        """Test reading a basic IWFM zone file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            content = (
                "C This is a comment\n"
                "1\n"  # ZExtent: horizontal
                "1  Zone A\n"
                "2  Zone B\n"
                "/\n"
                "1    1\n"
                "2    1\n"
                "3    2\n"
                "4    2\n"
            )
            filepath = Path(tmpdir) / "zones.dat"
            filepath.write_text(content)

            zone_def = read_iwfm_zone_file(filepath)

            assert zone_def.n_zones == 2
            assert zone_def.extent == "horizontal"
            assert zone_def.element_zones is not None
            assert zone_def.element_zones[0] == 1  # Element 1 -> Zone 1
            assert zone_def.element_zones[2] == 2  # Element 3 -> Zone 2

    def test_vertical_extent(self):
        """Test reading a file with vertical extent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            content = (
                "0\n"  # ZExtent: vertical
                "1  Layer 1\n"
                "/\n"
                "1    1\n"
            )
            filepath = Path(tmpdir) / "zones.dat"
            filepath.write_text(content)

            zone_def = read_iwfm_zone_file(filepath)
            assert zone_def.extent == "vertical"

    def test_comment_variants(self):
        """Test various comment line styles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            content = (
                "C Standard comment\n* Star comment\n# Hash comment\n1\n1  Test Zone\n/\n1    1\n"
            )
            filepath = Path(tmpdir) / "zones.dat"
            filepath.write_text(content)

            zone_def = read_iwfm_zone_file(filepath)
            assert zone_def.n_zones == 1

    def test_with_element_areas(self):
        """Test reading with element areas."""
        with tempfile.TemporaryDirectory() as tmpdir:
            content = "1\n1  Zone A\n/\n1    1\n2    1\n"
            filepath = Path(tmpdir) / "zones.dat"
            filepath.write_text(content)

            areas = {1: 100.0, 2: 200.0}
            zone_def = read_iwfm_zone_file(filepath, element_areas=areas)
            assert zone_def.n_zones == 1

    def test_empty_file(self):
        """Test reading an empty file with just structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            content = "C Empty zone file\n1\n/\n"
            filepath = Path(tmpdir) / "zones.dat"
            filepath.write_text(content)

            zone_def = read_iwfm_zone_file(filepath)
            assert zone_def.n_zones == 0

    def test_zone_names_auto_generated(self):
        """Test that zone names are auto-generated when missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            content = (
                "1\n"
                "1\n"  # No name
                "2\n"
                "/\n"
                "1    1\n"
                "2    2\n"
            )
            filepath = Path(tmpdir) / "zones.dat"
            filepath.write_text(content)

            zone_def = read_iwfm_zone_file(filepath)
            assert zone_def.n_zones == 2


class TestWriteIWFMZoneFile:
    """Tests for writing IWFM zone files."""

    def _create_zone_def(self):
        """Create a simple zone definition."""
        zones = {
            1: Zone(id=1, name="Zone A", elements=[1, 2], area=300.0),
            2: Zone(id=2, name="Zone B", elements=[3, 4], area=400.0),
        }
        element_zones = np.array([1, 1, 2, 2], dtype=np.int32)
        return ZoneDefinition(
            zones=zones,
            extent="horizontal",
            element_zones=element_zones,
            name="Test Zones",
        )

    def test_basic_write(self):
        """Test basic zone file writing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zone_def = self._create_zone_def()
            filepath = Path(tmpdir) / "output.dat"
            write_iwfm_zone_file(zone_def, filepath)

            assert filepath.exists()
            content = filepath.read_text()
            assert "Zone A" in content
            assert "Zone B" in content

    def test_roundtrip(self):
        """Test writing and reading back gives same data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zone_def = self._create_zone_def()
            filepath = Path(tmpdir) / "roundtrip.dat"

            write_iwfm_zone_file(zone_def, filepath)
            loaded = read_iwfm_zone_file(filepath)

            assert loaded.n_zones == zone_def.n_zones
            assert loaded.extent == zone_def.extent
            np.testing.assert_array_equal(loaded.element_zones[:4], zone_def.element_zones)

    def test_header_comment(self):
        """Test header comment is included."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zone_def = self._create_zone_def()
            filepath = Path(tmpdir) / "output.dat"
            write_iwfm_zone_file(zone_def, filepath, header_comment="Custom comment")

            content = filepath.read_text()
            assert "Custom comment" in content

    def test_vertical_extent(self):
        """Test writing vertical extent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zone_def = self._create_zone_def()
            zone_def.extent = "vertical"
            filepath = Path(tmpdir) / "output.dat"
            write_iwfm_zone_file(zone_def, filepath)

            content = filepath.read_text()
            # Find the ZExtent line (starts with 0 for vertical)
            lines = content.split("\n")
            data_lines = [
                line for line in lines if line.strip() and not line.strip().startswith("C")
            ]
            assert data_lines[0].strip().startswith("0")


class TestReadGeoJSONZones:
    """Tests for reading GeoJSON zone files."""

    def test_basic_read(self):
        """Test reading a basic GeoJSON zone file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            geojson = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "id": 1,
                            "name": "Zone A",
                            "element_id": [1, 2, 3],
                            "area": 500.0,
                        },
                        "geometry": None,
                    },
                    {
                        "type": "Feature",
                        "properties": {
                            "id": 2,
                            "name": "Zone B",
                            "element_id": [4, 5],
                            "area": 300.0,
                        },
                        "geometry": None,
                    },
                ],
            }
            filepath = Path(tmpdir) / "zones.geojson"
            filepath.write_text(json.dumps(geojson))

            zone_def = read_geojson_zones(filepath)

            assert zone_def.n_zones == 2
            assert 1 in zone_def.zones
            assert zone_def.zones[1].name == "Zone A"
            assert len(zone_def.zones[1].elements) == 3

    def test_custom_field_names(self):
        """Test reading with custom field names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            geojson = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "zone_num": 1,
                            "zone_label": "Custom",
                            "elements": [1, 2],
                        },
                        "geometry": None,
                    },
                ],
            }
            filepath = Path(tmpdir) / "zones.geojson"
            filepath.write_text(json.dumps(geojson))

            zone_def = read_geojson_zones(
                filepath,
                zone_id_field="zone_num",
                zone_name_field="zone_label",
                element_id_field="elements",
            )
            assert zone_def.n_zones == 1
            assert zone_def.zones[1].name == "Custom"

    def test_empty_features(self):
        """Test reading with no features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            geojson = {"type": "FeatureCollection", "features": []}
            filepath = Path(tmpdir) / "zones.geojson"
            filepath.write_text(json.dumps(geojson))

            zone_def = read_geojson_zones(filepath)
            assert zone_def.n_zones == 0


class TestWriteGeoJSONZones:
    """Tests for writing GeoJSON zone files."""

    def test_basic_write(self):
        """Test basic GeoJSON writing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zones = {
                1: Zone(id=1, name="Zone A", elements=[1, 2], area=300.0),
            }
            zone_def = ZoneDefinition(
                zones=zones,
                extent="horizontal",
                element_zones=np.array([1, 1], dtype=np.int32),
                name="Test",
            )

            filepath = Path(tmpdir) / "zones.geojson"
            write_geojson_zones(zone_def, filepath, include_geometry=False)

            assert filepath.exists()
            with open(filepath) as f:
                data = json.load(f)
            assert data["type"] == "FeatureCollection"
            assert len(data["features"]) == 1
            assert data["features"][0]["properties"]["name"] == "Zone A"

    def test_roundtrip(self):
        """Test writing and reading back GeoJSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zones = {
                1: Zone(id=1, name="Zone A", elements=[1, 2, 3], area=500.0),
                2: Zone(id=2, name="Zone B", elements=[4, 5], area=300.0),
            }
            zone_def = ZoneDefinition(
                zones=zones,
                extent="horizontal",
                element_zones=np.array([1, 1, 1, 2, 2], dtype=np.int32),
                name="Test",
            )

            filepath = Path(tmpdir) / "zones.geojson"
            write_geojson_zones(zone_def, filepath, include_geometry=False)
            loaded = read_geojson_zones(filepath)

            assert loaded.n_zones == 2
            assert loaded.zones[1].name == "Zone A"
            assert len(loaded.zones[1].elements) == 3


class TestAutoDetect:
    """Tests for format auto-detection."""

    def test_detect_geojson_by_extension(self):
        """Test detection of GeoJSON by extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.geojson"
            filepath.write_text("{}")
            assert auto_detect_zone_file(filepath) == "geojson"

    def test_detect_json_by_extension(self):
        """Test detection of JSON by extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.json"
            filepath.write_text("{}")
            assert auto_detect_zone_file(filepath) == "geojson"

    def test_detect_iwfm_by_extension(self):
        """Test detection of IWFM by extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.dat"
            filepath.write_text("C comment\n1\n")
            assert auto_detect_zone_file(filepath) == "iwfm"

    def test_detect_txt_as_iwfm(self):
        """Test that .txt is detected as IWFM."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.txt"
            filepath.write_text("C comment\n1\n")
            assert auto_detect_zone_file(filepath) == "iwfm"

    def test_detect_by_content(self):
        """Test detection by file content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.zone"
            filepath.write_text('{"type": "FeatureCollection"}')
            assert auto_detect_zone_file(filepath) == "geojson"


class TestUniversalIO:
    """Tests for the universal read/write functions."""

    def test_read_zone_file_iwfm(self):
        """Test read_zone_file with IWFM format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            content = "1\n1  Test\n/\n1    1\n"
            filepath = Path(tmpdir) / "zones.dat"
            filepath.write_text(content)

            zone_def = read_zone_file(filepath)
            assert zone_def.n_zones == 1

    def test_read_zone_file_geojson(self):
        """Test read_zone_file with GeoJSON format."""
        with tempfile.TemporaryDirectory() as tmpdir:
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
            filepath = Path(tmpdir) / "zones.geojson"
            filepath.write_text(json.dumps(geojson))

            zone_def = read_zone_file(filepath)
            assert zone_def.n_zones == 1

    def test_write_zone_file_iwfm(self):
        """Test write_zone_file with IWFM extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zones = {1: Zone(id=1, name="Z1", elements=[1], area=100.0)}
            zone_def = ZoneDefinition(
                zones=zones,
                extent="horizontal",
                element_zones=np.array([1], dtype=np.int32),
            )
            filepath = Path(tmpdir) / "zones.dat"
            write_zone_file(zone_def, filepath)
            assert filepath.exists()

    def test_write_zone_file_geojson(self):
        """Test write_zone_file with GeoJSON extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zones = {1: Zone(id=1, name="Z1", elements=[1], area=100.0)}
            zone_def = ZoneDefinition(
                zones=zones,
                extent="horizontal",
                element_zones=np.array([1], dtype=np.int32),
            )
            filepath = Path(tmpdir) / "zones.geojson"
            write_zone_file(zone_def, filepath)
            assert filepath.exists()
            with open(filepath) as f:
                data = json.load(f)
            assert data["type"] == "FeatureCollection"
