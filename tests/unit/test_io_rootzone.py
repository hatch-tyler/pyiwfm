"""Unit tests for root zone component I/O.

Tests:
- RootZoneFileConfig
- Helper functions
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyiwfm.io.rootzone import (
    RootZoneFileConfig,
    _is_comment_line,
    _strip_comment,
)


# =============================================================================
# Test Helper Functions
# =============================================================================


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_is_comment_line_c_comment(self) -> None:
        """Test C comment detection."""
        assert _is_comment_line("C This is a comment") is True
        assert _is_comment_line("c lowercase comment") is True

    def test_is_comment_line_asterisk_comment(self) -> None:
        """Test asterisk comment detection."""
        assert _is_comment_line("* This is a comment") is True

    def test_is_comment_line_hash_not_comment(self) -> None:
        """Hash is not a comment character."""
        assert _is_comment_line("# This is a comment") is False

    def test_is_comment_line_empty(self) -> None:
        """Test empty line is treated as comment."""
        assert _is_comment_line("") is True
        assert _is_comment_line("   ") is True

    def test_is_comment_line_data(self) -> None:
        """Test data line is not a comment."""
        assert _is_comment_line("1  2  3  4") is False

    def test_strip_comment_with_description(self) -> None:
        """Test parsing line with description."""
        value, desc = _strip_comment("5                / NCROPS")
        assert value == "5"
        assert desc == "NCROPS"

    def test_strip_comment_no_description(self) -> None:
        """Test parsing line without description."""
        value, desc = _strip_comment("5")
        assert value == "5"
        assert desc == ""


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
        assert config.ag_landuse_file == "ag_landuse.dat"
        assert config.urban_landuse_file == "urban_landuse.dat"
        assert config.native_landuse_file == "native_landuse.dat"
        assert config.soil_moisture_file == "initial_soil_moisture.dat"

    def test_custom_file_names(self, tmp_path: Path) -> None:
        """Test config with custom file names."""
        config = RootZoneFileConfig(
            output_dir=tmp_path,
            crop_types_file="custom_crops.dat",
            soil_params_file="custom_soil.dat",
        )

        assert config.crop_types_file == "custom_crops.dat"
        assert config.soil_params_file == "custom_soil.dat"

    def test_get_crop_types_path(self, tmp_path: Path) -> None:
        """Test crop types path getter."""
        config = RootZoneFileConfig(output_dir=tmp_path)
        path = config.get_crop_types_path()

        assert path == tmp_path / "crop_types.dat"

    def test_get_soil_params_path(self, tmp_path: Path) -> None:
        """Test soil params path getter."""
        config = RootZoneFileConfig(output_dir=tmp_path)
        path = config.get_soil_params_path()

        assert path == tmp_path / "soil_params.dat"

    def test_get_landuse_path(self, tmp_path: Path) -> None:
        """Test land use path getter."""
        config = RootZoneFileConfig(output_dir=tmp_path)
        path = config.get_landuse_path()

        assert path == tmp_path / "landuse.dat"

    def test_get_ag_landuse_path(self, tmp_path: Path) -> None:
        """Test agricultural land use path getter."""
        config = RootZoneFileConfig(output_dir=tmp_path)
        path = config.get_ag_landuse_path()

        assert path == tmp_path / "ag_landuse.dat"

    def test_get_urban_landuse_path(self, tmp_path: Path) -> None:
        """Test urban land use path getter."""
        config = RootZoneFileConfig(output_dir=tmp_path)
        path = config.get_urban_landuse_path()

        assert path == tmp_path / "urban_landuse.dat"

    def test_get_native_landuse_path(self, tmp_path: Path) -> None:
        """Test native land use path getter."""
        config = RootZoneFileConfig(output_dir=tmp_path)
        path = config.get_native_landuse_path()

        assert path == tmp_path / "native_landuse.dat"

    def test_get_soil_moisture_path(self, tmp_path: Path) -> None:
        """Test soil moisture path getter."""
        config = RootZoneFileConfig(output_dir=tmp_path)
        path = config.get_soil_moisture_path()

        assert path == tmp_path / "initial_soil_moisture.dat"
