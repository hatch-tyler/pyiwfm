"""Supplementary tests for rootzone.py targeting uncovered branches.

Covers:
- RootZoneWriter additional methods
- RootZoneReader edge cases
- Roundtrip with various crop types
- Error handling for invalid input
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyiwfm.components.rootzone import (
    CropType,
    RootZone,
    SoilParameters,
)
from pyiwfm.core.exceptions import FileFormatError
from pyiwfm.io.rootzone import (
    RootZoneFileConfig,
    RootZoneReader,
    RootZoneWriter,
    read_crop_types,
    write_rootzone,
)

# =============================================================================
# RootZoneWriter Additional Tests
# =============================================================================


class TestRootZoneWriterAdditional:
    """Additional tests for RootZoneWriter."""

    def test_write_crop_types_basic(self, tmp_path: Path) -> None:
        """Test writing crop types file."""
        crops = [
            CropType(id=1, name="Pasture", kc=0.75, root_depth=2.0),
            CropType(id=2, name="Orchard", kc=0.85, root_depth=4.0),
            CropType(id=3, name="Urban", kc=0.2, root_depth=0.5),
        ]
        rz = RootZone(n_elements=3, n_layers=1)
        for c in crops:
            rz.add_crop_type(c)

        config = RootZoneFileConfig(output_dir=tmp_path)
        writer = RootZoneWriter(config)
        filepath = writer.write_crop_types(rz)

        assert filepath.exists()
        content = filepath.read_text()
        assert "NCROPS" in content
        assert "3" in content
        assert "Pasture" in content
        assert "Orchard" in content

    def test_write_soil_params_basic(self, tmp_path: Path) -> None:
        """Test writing soil parameters file."""
        rz = RootZone(n_elements=3, n_layers=1)
        rz.set_soil_parameters(1, SoilParameters(0.40, 0.30, 0.10, 0.50))
        rz.set_soil_parameters(2, SoilParameters(0.42, 0.32, 0.12, 0.52))
        rz.set_soil_parameters(3, SoilParameters(0.41, 0.31, 0.11, 0.51))

        config = RootZoneFileConfig(output_dir=tmp_path)
        writer = RootZoneWriter(config)
        filepath = writer.write_soil_params(rz)

        assert filepath.exists()
        content = filepath.read_text()
        assert "NELEM" in content or "N_ELEMENTS" in content

    def test_write_empty_rootzone(self, tmp_path: Path) -> None:
        """Test writing root zone with no crop types or soil params."""
        rz = RootZone(n_elements=3, n_layers=2)

        config = RootZoneFileConfig(output_dir=tmp_path)
        writer = RootZoneWriter(config)
        files = writer.write(rz)

        # No crop types or soil params means no files
        assert len(files) == 0


# =============================================================================
# RootZoneReader Additional Tests
# =============================================================================


class TestRootZoneReaderAdditional:
    """Additional tests for RootZoneReader."""

    def test_read_crop_types_with_spaces_in_names(self, tmp_path: Path) -> None:
        """Test reading crop types with spaces in names."""
        filepath = tmp_path / "crops.dat"
        filepath.write_text("""C Crop types file
C
3                               / NCROPS
1   0.75   2.00   Irrigated Pasture
2   0.85   4.00   Deciduous Orchard
3   0.20   0.50   Urban Landscape
""")

        reader = RootZoneReader()
        crops = reader.read_crop_types(filepath)

        assert len(crops) == 3
        assert crops[1].name == "Irrigated Pasture"
        assert crops[2].name == "Deciduous Orchard"
        assert crops[3].name == "Urban Landscape"

    def test_read_crop_types_missing_ncrops(self, tmp_path: Path) -> None:
        """Test error when NCROPS is missing."""
        filepath = tmp_path / "crops.dat"
        filepath.write_text("""C Only comments
C No data
""")

        reader = RootZoneReader()
        with pytest.raises(FileFormatError, match="NCROPS"):
            reader.read_crop_types(filepath)

    def test_read_crop_types_invalid_ncrops(self, tmp_path: Path) -> None:
        """Test error when NCROPS is invalid."""
        filepath = tmp_path / "crops.dat"
        filepath.write_text("""C Crop types
abc                             / NCROPS
""")

        reader = RootZoneReader()
        with pytest.raises(FileFormatError, match="Invalid NCROPS"):
            reader.read_crop_types(filepath)

    def test_read_crop_types_invalid_data(self, tmp_path: Path) -> None:
        """Test error when crop data is invalid."""
        filepath = tmp_path / "crops.dat"
        filepath.write_text("""C Crop types
1                               / NCROPS
abc   def   ghi   Invalid Crop
""")

        reader = RootZoneReader()
        with pytest.raises(FileFormatError, match="Invalid crop"):
            reader.read_crop_types(filepath)


# =============================================================================
# Roundtrip Tests
# =============================================================================


class TestRootZoneRoundtrip:
    """Roundtrip read-write tests for root zone data."""

    def test_crop_types_roundtrip(self, tmp_path: Path) -> None:
        """Test crop types write -> read roundtrip."""
        rz = RootZone(n_elements=1, n_layers=1)
        rz.add_crop_type(CropType(id=1, name="Crop_A", kc=0.75, root_depth=2.0))
        rz.add_crop_type(CropType(id=2, name="Crop_B", kc=0.90, root_depth=3.5))

        config = RootZoneFileConfig(output_dir=tmp_path)
        writer = RootZoneWriter(config)
        filepath = writer.write_crop_types(rz)

        reader = RootZoneReader()
        read_crops = reader.read_crop_types(filepath)

        assert len(read_crops) == 2
        assert read_crops[1].name == "Crop_A"
        assert read_crops[1].kc == pytest.approx(0.75)
        assert read_crops[2].root_depth == pytest.approx(3.5)


# =============================================================================
# Convenience Functions
# =============================================================================


class TestRootZoneConvenience:
    """Convenience function tests."""

    def test_read_crop_types_convenience(self, tmp_path: Path) -> None:
        """Test read_crop_types convenience function."""
        filepath = tmp_path / "crops.dat"
        filepath.write_text("""C Crop types
1                               / NCROPS
1   0.75   2.00   TestCrop
""")

        crops = read_crop_types(filepath)
        assert len(crops) == 1

    def test_write_rootzone_string_path(self, tmp_path: Path) -> None:
        """Test write_rootzone with string output directory."""
        rz = RootZone(n_elements=1, n_layers=1)
        rz.add_crop_type(CropType(id=1, name="C1", kc=0.5, root_depth=1.0))

        files = write_rootzone(rz, str(tmp_path))

        assert "crop_types" in files
