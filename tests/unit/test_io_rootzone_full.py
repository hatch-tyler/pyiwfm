"""Unit tests for root zone I/O module.

Tests:
- _is_comment_line function
- _parse_value_line function
- RootZoneFileConfig dataclass
- RootZoneWriter class
- RootZoneReader class
- Convenience functions
- Roundtrip read/write
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyiwfm.io.rootzone import (
    RootZoneFileConfig,
    RootZoneWriter,
    RootZoneReader,
    write_rootzone,
    read_crop_types,
    read_soil_params,
    _is_comment_line,
    _parse_value_line,
)
from pyiwfm.components.rootzone import (
    RootZone,
    CropType,
    SoilParameters,
    ElementLandUse,
    LandUseType,
)
from pyiwfm.core.exceptions import FileFormatError


# =============================================================================
# Test Helper Functions
# =============================================================================


class TestIsCommentLine:
    """Tests for _is_comment_line function."""

    def test_uppercase_c_comment(self) -> None:
        """Test uppercase C comment."""
        assert _is_comment_line("C This is a comment") is True

    def test_lowercase_c_comment(self) -> None:
        """Test lowercase c comment."""
        assert _is_comment_line("c comment line") is True

    def test_asterisk_comment(self) -> None:
        """Test asterisk comment."""
        assert _is_comment_line("* Comment with asterisk") is True

    def test_hash_not_comment(self) -> None:
        """Hash is not a comment character."""
        assert _is_comment_line("# Comment with hash") is False

    def test_empty_line(self) -> None:
        """Test empty line is treated as comment."""
        assert _is_comment_line("") is True
        assert _is_comment_line("   ") is True

    def test_non_comment_line(self) -> None:
        """Test non-comment lines."""
        assert _is_comment_line("10 / NCROPS") is False
        assert _is_comment_line("1  0.5  0.3  0.1  0.01") is False


class TestParseValueLine:
    """Tests for _parse_value_line function."""

    def test_with_description(self) -> None:
        """Test parsing line with description."""
        value, desc = _parse_value_line("10  / NCROPS")
        assert value == "10"
        assert desc == "NCROPS"

    def test_without_description(self) -> None:
        """Test parsing line without description."""
        value, desc = _parse_value_line("10")
        assert value == "10"
        assert desc == ""

    def test_with_whitespace(self) -> None:
        """Test parsing line with extra whitespace."""
        value, desc = _parse_value_line("  100   /   Description   ")
        assert value == "100"
        assert desc == "Description"


# =============================================================================
# Test RootZoneFileConfig
# =============================================================================


class TestRootZoneFileConfig:
    """Tests for RootZoneFileConfig dataclass."""

    def test_basic_creation(self, tmp_path: Path) -> None:
        """Test basic config creation."""
        config = RootZoneFileConfig(output_dir=tmp_path)

        assert config.output_dir == tmp_path
        assert config.crop_types_file == "crop_types.dat"
        assert config.soil_params_file == "soil_params.dat"
        assert config.landuse_file == "landuse.dat"

    def test_custom_filenames(self, tmp_path: Path) -> None:
        """Test config with custom filenames."""
        config = RootZoneFileConfig(
            output_dir=tmp_path,
            crop_types_file="custom_crops.dat",
            soil_params_file="custom_soil.dat",
        )

        assert config.crop_types_file == "custom_crops.dat"
        assert config.soil_params_file == "custom_soil.dat"

    def test_path_methods(self, tmp_path: Path) -> None:
        """Test path getter methods."""
        config = RootZoneFileConfig(output_dir=tmp_path)

        assert config.get_crop_types_path() == tmp_path / "crop_types.dat"
        assert config.get_soil_params_path() == tmp_path / "soil_params.dat"
        assert config.get_landuse_path() == tmp_path / "landuse.dat"
        assert config.get_ag_landuse_path() == tmp_path / "ag_landuse.dat"
        assert config.get_urban_landuse_path() == tmp_path / "urban_landuse.dat"
        assert config.get_native_landuse_path() == tmp_path / "native_landuse.dat"
        assert config.get_soil_moisture_path() == tmp_path / "initial_soil_moisture.dat"


# =============================================================================
# Test RootZoneWriter
# =============================================================================


class TestRootZoneWriter:
    """Tests for RootZoneWriter class."""

    @pytest.fixture
    def config(self, tmp_path: Path) -> RootZoneFileConfig:
        """Create test config."""
        return RootZoneFileConfig(output_dir=tmp_path)

    @pytest.fixture
    def basic_rootzone(self) -> RootZone:
        """Create basic rootzone for testing."""
        rz = RootZone(n_elements=3, n_layers=2)

        # Add crop types
        rz.add_crop_type(CropType(id=1, name="Alfalfa", root_depth=3.0, kc=1.1))
        rz.add_crop_type(CropType(id=2, name="Corn", root_depth=2.0, kc=0.9))

        # Add soil parameters
        rz.set_soil_parameters(1, SoilParameters(0.4, 0.3, 0.1, 0.01))
        rz.set_soil_parameters(2, SoilParameters(0.35, 0.25, 0.08, 0.02))
        rz.set_soil_parameters(3, SoilParameters(0.38, 0.28, 0.09, 0.015))

        # Add land use
        rz.add_element_landuse(ElementLandUse(
            element_id=1,
            land_use_type=LandUseType.AGRICULTURAL,
            area=100.0,
            crop_fractions={1: 0.6, 2: 0.4}
        ))
        rz.add_element_landuse(ElementLandUse(
            element_id=2,
            land_use_type=LandUseType.URBAN,
            area=50.0,
            impervious_fraction=0.7
        ))
        rz.add_element_landuse(ElementLandUse(
            element_id=3,
            land_use_type=LandUseType.NATIVE_RIPARIAN,
            area=75.0
        ))

        return rz

    def test_initialization(self, config: RootZoneFileConfig) -> None:
        """Test writer initialization."""
        writer = RootZoneWriter(config)
        assert writer.config == config
        assert config.output_dir.exists()

    def test_write_crop_types(self, config: RootZoneFileConfig, basic_rootzone: RootZone) -> None:
        """Test writing crop types file."""
        writer = RootZoneWriter(config)
        filepath = writer.write_crop_types(basic_rootzone)

        assert filepath.exists()
        content = filepath.read_text()

        assert "NCROPS" in content
        assert "Alfalfa" in content
        assert "Corn" in content
        assert "3.0" in content  # root depth
        assert "1.1" in content  # kc

    def test_write_crop_types_with_header(self, config: RootZoneFileConfig, basic_rootzone: RootZone) -> None:
        """Test writing crop types with custom header."""
        writer = RootZoneWriter(config)
        filepath = writer.write_crop_types(basic_rootzone, header="Custom header")

        content = filepath.read_text()
        assert "Custom header" in content

    def test_write_crop_types_with_monthly_kc(self, config: RootZoneFileConfig) -> None:
        """Test writing crop types with monthly Kc values."""
        rz = RootZone(n_elements=1, n_layers=1)
        crop = CropType(
            id=1,
            name="Test",
            root_depth=2.0,
            kc=1.0,
            monthly_kc=np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6])
        )
        rz.add_crop_type(crop)

        writer = RootZoneWriter(config)
        filepath = writer.write_crop_types(rz)

        content = filepath.read_text()
        assert "Monthly Kc" in content

    def test_write_soil_params(self, config: RootZoneFileConfig, basic_rootzone: RootZone) -> None:
        """Test writing soil parameters file."""
        writer = RootZoneWriter(config)
        filepath = writer.write_soil_params(basic_rootzone)

        assert filepath.exists()
        content = filepath.read_text()

        assert "NELEM_SOIL" in content
        assert "0.4" in content  # porosity
        assert "0.3" in content  # field capacity

    def test_write_soil_params_with_header(self, config: RootZoneFileConfig, basic_rootzone: RootZone) -> None:
        """Test writing soil parameters with custom header."""
        writer = RootZoneWriter(config)
        filepath = writer.write_soil_params(basic_rootzone, header="Custom soil header")

        content = filepath.read_text()
        assert "Custom soil header" in content

    def test_write_landuse(self, config: RootZoneFileConfig, basic_rootzone: RootZone) -> None:
        """Test writing land use file."""
        writer = RootZoneWriter(config)
        filepath = writer.write_landuse(basic_rootzone)

        assert filepath.exists()
        content = filepath.read_text()

        assert "NLANDUSE" in content
        assert "AGRICULTURAL" in content
        assert "URBAN" in content
        assert "NATIVE/RIPARIAN" in content
        assert "100.0" in content  # ag area
        assert "50.0" in content  # urban area
        assert "0.7" in content  # impervious fraction

    def test_write_landuse_with_header(self, config: RootZoneFileConfig, basic_rootzone: RootZone) -> None:
        """Test writing land use with custom header."""
        writer = RootZoneWriter(config)
        filepath = writer.write_landuse(basic_rootzone, header="Custom landuse header")

        content = filepath.read_text()
        assert "Custom landuse header" in content

    def test_write_soil_moisture(self, config: RootZoneFileConfig, basic_rootzone: RootZone) -> None:
        """Test writing soil moisture file."""
        # Add soil moisture data
        basic_rootzone.soil_moisture = np.array([
            [0.25, 0.30],
            [0.22, 0.28],
            [0.20, 0.26],
        ])

        writer = RootZoneWriter(config)
        filepath = writer.write_soil_moisture(basic_rootzone)

        assert filepath.exists()
        content = filepath.read_text()

        assert "NELEM" in content
        assert "NLAYERS" in content
        assert "0.25" in content

    def test_write_soil_moisture_no_data_raises_error(self, config: RootZoneFileConfig, basic_rootzone: RootZone) -> None:
        """Test writing soil moisture without data raises error."""
        writer = RootZoneWriter(config)

        with pytest.raises(ValueError, match="No soil moisture"):
            writer.write_soil_moisture(basic_rootzone)

    def test_write_all(self, config: RootZoneFileConfig, basic_rootzone: RootZone) -> None:
        """Test writing all files."""
        # Add soil moisture
        basic_rootzone.soil_moisture = np.array([
            [0.25, 0.30],
            [0.22, 0.28],
            [0.20, 0.26],
        ])

        writer = RootZoneWriter(config)
        files = writer.write(basic_rootzone)

        assert "crop_types" in files
        assert "soil_params" in files
        assert "landuse" in files
        assert "soil_moisture" in files

        for path in files.values():
            assert path.exists()

    def test_write_empty_components(self, config: RootZoneFileConfig) -> None:
        """Test writing with empty components."""
        rz = RootZone(n_elements=3, n_layers=2)

        writer = RootZoneWriter(config)
        files = writer.write(rz)

        # Should have no files when components are empty
        assert len(files) == 0


# =============================================================================
# Test RootZoneReader
# =============================================================================


class TestRootZoneReader:
    """Tests for RootZoneReader class."""

    def test_read_crop_types(self, tmp_path: Path) -> None:
        """Test reading crop types file."""
        filepath = tmp_path / "crops.dat"
        filepath.write_text("""C  Crop types
2                              / NCROPS
1       3.0000   1.1000  Alfalfa
2       2.0000   0.9000  Corn
""")

        reader = RootZoneReader()
        crops = reader.read_crop_types(filepath)

        assert len(crops) == 2
        assert crops[1].name == "Alfalfa"
        assert crops[1].root_depth == 3.0
        assert crops[1].kc == 1.1
        assert crops[2].name == "Corn"

    def test_read_crop_types_with_comments(self, tmp_path: Path) -> None:
        """Test reading crop types with various comments."""
        filepath = tmp_path / "crops.dat"
        filepath.write_text("""C  Comment 1
c  lowercase comment
*  asterisk comment
1                              / NCROPS
C  Data starts here
1       2.5000   1.0000  TestCrop
""")

        reader = RootZoneReader()
        crops = reader.read_crop_types(filepath)

        assert len(crops) == 1
        assert crops[1].name == "TestCrop"

    def test_read_crop_types_invalid_ncrops(self, tmp_path: Path) -> None:
        """Test reading crop types with invalid NCROPS."""
        filepath = tmp_path / "crops.dat"
        filepath.write_text("""C  Crop types
abc                            / NCROPS
""")

        reader = RootZoneReader()
        with pytest.raises(FileFormatError, match="Invalid NCROPS"):
            reader.read_crop_types(filepath)

    def test_read_crop_types_missing_ncrops(self, tmp_path: Path) -> None:
        """Test reading crop types without NCROPS."""
        filepath = tmp_path / "crops.dat"
        filepath.write_text("""C  Only comments
C  No data
""")

        reader = RootZoneReader()
        with pytest.raises(FileFormatError, match="Could not find NCROPS"):
            reader.read_crop_types(filepath)

    def test_read_crop_types_invalid_data(self, tmp_path: Path) -> None:
        """Test reading crop types with invalid data."""
        filepath = tmp_path / "crops.dat"
        filepath.write_text("""C  Crop types
1                              / NCROPS
abc   def   ghi  Invalid
""")

        reader = RootZoneReader()
        with pytest.raises(FileFormatError, match="Invalid crop data"):
            reader.read_crop_types(filepath)

    def test_read_soil_params(self, tmp_path: Path) -> None:
        """Test reading soil parameters file."""
        filepath = tmp_path / "soil.dat"
        filepath.write_text("""C  Soil parameters
3                              / NELEM_SOIL
1       0.400000   0.300000   0.100000     0.010000
2       0.350000   0.250000   0.080000     0.020000
3       0.380000   0.280000   0.090000     0.015000
""")

        reader = RootZoneReader()
        params = reader.read_soil_params(filepath)

        assert len(params) == 3
        assert params[1].porosity == pytest.approx(0.4)
        assert params[1].field_capacity == pytest.approx(0.3)
        assert params[1].wilting_point == pytest.approx(0.1)
        assert params[1].saturated_kv == pytest.approx(0.01)

    def test_read_soil_params_invalid_nelem(self, tmp_path: Path) -> None:
        """Test reading soil params with invalid NELEM."""
        filepath = tmp_path / "soil.dat"
        filepath.write_text("""C  Soil parameters
xyz                            / NELEM_SOIL
""")

        reader = RootZoneReader()
        with pytest.raises(FileFormatError, match="Invalid NELEM_SOIL"):
            reader.read_soil_params(filepath)

    def test_read_soil_params_missing_nelem(self, tmp_path: Path) -> None:
        """Test reading soil params without NELEM."""
        filepath = tmp_path / "soil.dat"
        filepath.write_text("""C  Only comments
""")

        reader = RootZoneReader()
        with pytest.raises(FileFormatError, match="Could not find NELEM_SOIL"):
            reader.read_soil_params(filepath)

    def test_read_soil_params_invalid_data(self, tmp_path: Path) -> None:
        """Test reading soil params with invalid data."""
        filepath = tmp_path / "soil.dat"
        filepath.write_text("""C  Soil parameters
1                              / NELEM_SOIL
1  abc  def  ghi  jkl
""")

        reader = RootZoneReader()
        with pytest.raises(FileFormatError, match="Invalid soil parameter"):
            reader.read_soil_params(filepath)


# =============================================================================
# Test Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_write_rootzone_function(self, tmp_path: Path) -> None:
        """Test write_rootzone function."""
        rz = RootZone(n_elements=2, n_layers=1)
        rz.add_crop_type(CropType(id=1, name="Test", root_depth=2.0, kc=1.0))
        rz.set_soil_parameters(1, SoilParameters(0.4, 0.3, 0.1, 0.01))

        files = write_rootzone(rz, tmp_path)

        assert "crop_types" in files
        assert "soil_params" in files

    def test_write_rootzone_with_config(self, tmp_path: Path) -> None:
        """Test write_rootzone with custom config."""
        config = RootZoneFileConfig(
            output_dir=tmp_path,
            crop_types_file="custom_crops.dat"
        )
        rz = RootZone(n_elements=1, n_layers=1)
        rz.add_crop_type(CropType(id=1, name="Test", root_depth=2.0, kc=1.0))

        files = write_rootzone(rz, tmp_path, config=config)

        assert files["crop_types"].name == "custom_crops.dat"

    def test_read_crop_types_function(self, tmp_path: Path) -> None:
        """Test read_crop_types function."""
        filepath = tmp_path / "crops.dat"
        filepath.write_text("""C  Crops
1                              / NCROPS
1       2.5000   1.0000  TestCrop
""")

        crops = read_crop_types(filepath)

        assert len(crops) == 1
        assert crops[1].name == "TestCrop"

    def test_read_soil_params_function(self, tmp_path: Path) -> None:
        """Test read_soil_params function."""
        filepath = tmp_path / "soil.dat"
        filepath.write_text("""C  Soil
1                              / NELEM_SOIL
1       0.400000   0.300000   0.100000     0.010000
""")

        params = read_soil_params(filepath)

        assert len(params) == 1
        assert params[1].porosity == pytest.approx(0.4)


# =============================================================================
# Test Roundtrip
# =============================================================================


class TestRoundtrip:
    """Tests for read/write roundtrip."""

    def test_roundtrip_crop_types(self, tmp_path: Path) -> None:
        """Test roundtrip for crop types."""
        # Create and write
        rz = RootZone(n_elements=1, n_layers=1)
        rz.add_crop_type(CropType(id=1, name="Alfalfa", root_depth=3.5, kc=1.15))
        rz.add_crop_type(CropType(id=2, name="Corn", root_depth=2.5, kc=0.95))
        rz.add_crop_type(CropType(id=3, name="Wheat", root_depth=1.8, kc=0.85))

        config = RootZoneFileConfig(output_dir=tmp_path)
        writer = RootZoneWriter(config)
        filepath = writer.write_crop_types(rz)

        # Read back
        reader = RootZoneReader()
        crops = reader.read_crop_types(filepath)

        assert len(crops) == 3
        assert crops[1].name == "Alfalfa"
        assert crops[1].root_depth == pytest.approx(3.5)
        assert crops[1].kc == pytest.approx(1.15, rel=1e-3)
        assert crops[2].name == "Corn"
        assert crops[3].name == "Wheat"

    def test_roundtrip_soil_params(self, tmp_path: Path) -> None:
        """Test roundtrip for soil parameters."""
        # Create and write
        rz = RootZone(n_elements=3, n_layers=1)
        rz.set_soil_parameters(1, SoilParameters(0.42, 0.31, 0.12, 0.015))
        rz.set_soil_parameters(2, SoilParameters(0.38, 0.27, 0.09, 0.022))
        rz.set_soil_parameters(3, SoilParameters(0.45, 0.33, 0.14, 0.008))

        config = RootZoneFileConfig(output_dir=tmp_path)
        writer = RootZoneWriter(config)
        filepath = writer.write_soil_params(rz)

        # Read back
        reader = RootZoneReader()
        params = reader.read_soil_params(filepath)

        assert len(params) == 3
        assert params[1].porosity == pytest.approx(0.42)
        assert params[1].field_capacity == pytest.approx(0.31)
        assert params[2].saturated_kv == pytest.approx(0.022)


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_crop(self, tmp_path: Path) -> None:
        """Test with single crop."""
        rz = RootZone(n_elements=1, n_layers=1)
        rz.add_crop_type(CropType(id=1, name="Solo", root_depth=1.0, kc=1.0))

        config = RootZoneFileConfig(output_dir=tmp_path)
        writer = RootZoneWriter(config)
        filepath = writer.write_crop_types(rz)

        reader = RootZoneReader()
        crops = reader.read_crop_types(filepath)

        assert len(crops) == 1

    def test_many_crops(self, tmp_path: Path) -> None:
        """Test with many crops."""
        rz = RootZone(n_elements=1, n_layers=1)
        for i in range(50):
            rz.add_crop_type(CropType(id=i+1, name=f"Crop{i+1}", root_depth=float(i % 5 + 1), kc=0.5 + i * 0.02))

        config = RootZoneFileConfig(output_dir=tmp_path)
        writer = RootZoneWriter(config)
        filepath = writer.write_crop_types(rz)

        reader = RootZoneReader()
        crops = reader.read_crop_types(filepath)

        assert len(crops) == 50

    def test_crop_name_with_spaces(self, tmp_path: Path) -> None:
        """Test crop names with spaces."""
        rz = RootZone(n_elements=1, n_layers=1)
        rz.add_crop_type(CropType(id=1, name="Winter Wheat HRS", root_depth=2.0, kc=1.0))

        config = RootZoneFileConfig(output_dir=tmp_path)
        writer = RootZoneWriter(config)
        filepath = writer.write_crop_types(rz)

        reader = RootZoneReader()
        crops = reader.read_crop_types(filepath)

        assert crops[1].name == "Winter Wheat HRS"

    def test_water_landuse_type(self, tmp_path: Path) -> None:
        """Test water land use type."""
        rz = RootZone(n_elements=2, n_layers=1)
        rz.add_element_landuse(ElementLandUse(
            element_id=1,
            land_use_type=LandUseType.WATER,
            area=25.0
        ))

        config = RootZoneFileConfig(output_dir=tmp_path)
        writer = RootZoneWriter(config)
        filepath = writer.write_landuse(rz)

        content = filepath.read_text()
        assert "WATER BODIES" in content
        assert "25.0" in content

    def test_mixed_landuse_types(self, tmp_path: Path) -> None:
        """Test all land use types together."""
        rz = RootZone(n_elements=4, n_layers=1)
        rz.add_element_landuse(ElementLandUse(1, LandUseType.AGRICULTURAL, 100.0, {1: 1.0}))
        rz.add_element_landuse(ElementLandUse(2, LandUseType.URBAN, 50.0, impervious_fraction=0.8))
        rz.add_element_landuse(ElementLandUse(3, LandUseType.NATIVE_RIPARIAN, 75.0))
        rz.add_element_landuse(ElementLandUse(4, LandUseType.WATER, 25.0))
        rz.add_crop_type(CropType(id=1, name="Test", root_depth=1.0, kc=1.0))

        config = RootZoneFileConfig(output_dir=tmp_path)
        writer = RootZoneWriter(config)
        filepath = writer.write_landuse(rz)

        content = filepath.read_text()
        assert "NAG_LANDUSE" in content
        assert "NURBAN_LANDUSE" in content
        assert "NNATIVE_LANDUSE" in content
        assert "NWATER_LANDUSE" in content
