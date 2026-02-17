"""Unit tests for groundwater component I/O.

Tests:
- GWFileConfig
- Helper functions
- GroundwaterReader (wells only)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyiwfm.core.exceptions import FileFormatError
from pyiwfm.io.groundwater import (
    GroundwaterReader,
    GWFileConfig,
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
        value, desc = _strip_comment("10                / NWELLS")
        assert value == "10"
        assert desc == "NWELLS"

    def test_strip_comment_no_description(self) -> None:
        """Test parsing line without description."""
        value, desc = _strip_comment("10")
        assert value == "10"
        assert desc == ""


# =============================================================================
# Test GWFileConfig
# =============================================================================


class TestGWFileConfig:
    """Tests for GWFileConfig dataclass."""

    def test_basic_creation(self, tmp_path: Path) -> None:
        """Test basic config creation."""
        config = GWFileConfig(output_dir=tmp_path)

        assert config.output_dir == tmp_path
        assert config.wells_file == "wells.dat"
        assert config.pumping_file == "pumping.dat"
        assert config.aquifer_params_file == "aquifer_params.dat"
        assert config.boundary_conditions_file == "boundary_conditions.dat"
        assert config.tile_drains_file == "tile_drains.dat"
        assert config.subsidence_file == "subsidence.dat"
        assert config.initial_heads_file == "initial_heads.dat"

    def test_custom_file_names(self, tmp_path: Path) -> None:
        """Test config with custom file names."""
        config = GWFileConfig(
            output_dir=tmp_path,
            wells_file="custom_wells.dat",
            pumping_file="custom_pumping.dat",
        )

        assert config.wells_file == "custom_wells.dat"
        assert config.pumping_file == "custom_pumping.dat"

    def test_get_wells_path(self, tmp_path: Path) -> None:
        """Test wells path getter."""
        config = GWFileConfig(output_dir=tmp_path)
        path = config.get_wells_path()

        assert path == tmp_path / "wells.dat"

    def test_get_pumping_path(self, tmp_path: Path) -> None:
        """Test pumping path getter."""
        config = GWFileConfig(output_dir=tmp_path)
        path = config.get_pumping_path()

        assert path == tmp_path / "pumping.dat"

    def test_get_aquifer_params_path(self, tmp_path: Path) -> None:
        """Test aquifer params path getter."""
        config = GWFileConfig(output_dir=tmp_path)
        path = config.get_aquifer_params_path()

        assert path == tmp_path / "aquifer_params.dat"

    def test_get_boundary_conditions_path(self, tmp_path: Path) -> None:
        """Test boundary conditions path getter."""
        config = GWFileConfig(output_dir=tmp_path)
        path = config.get_boundary_conditions_path()

        assert path == tmp_path / "boundary_conditions.dat"

    def test_get_tile_drains_path(self, tmp_path: Path) -> None:
        """Test tile drains path getter."""
        config = GWFileConfig(output_dir=tmp_path)
        path = config.get_tile_drains_path()

        assert path == tmp_path / "tile_drains.dat"

    def test_get_subsidence_path(self, tmp_path: Path) -> None:
        """Test subsidence path getter."""
        config = GWFileConfig(output_dir=tmp_path)
        path = config.get_subsidence_path()

        assert path == tmp_path / "subsidence.dat"

    def test_get_initial_heads_path(self, tmp_path: Path) -> None:
        """Test initial heads path getter."""
        config = GWFileConfig(output_dir=tmp_path)
        path = config.get_initial_heads_path()

        assert path == tmp_path / "initial_heads.dat"


# =============================================================================
# Test GroundwaterReader
# =============================================================================


class TestGroundwaterReader:
    """Tests for GroundwaterReader class."""

    def test_read_wells_basic(self, tmp_path: Path) -> None:
        """Test reading basic wells file."""
        wells_file = tmp_path / "wells.dat"
        wells_file.write_text("""C Wells definition file
C
2                               / NWELLS
1      1000.0000     2000.0000     1    50.00   100.00     500.00  Well 1
2      1500.0000     2500.0000     2    60.00   120.00     750.00  Well 2
""")

        reader = GroundwaterReader()
        wells = reader.read_wells(wells_file)

        assert len(wells) == 2
        assert wells[1].id == 1
        assert wells[1].name == "Well 1"
        assert wells[1].x == pytest.approx(1000.0)
        assert wells[1].y == pytest.approx(2000.0)
        assert wells[1].element == 1
        assert wells[1].top_screen == pytest.approx(50.0)
        assert wells[1].bottom_screen == pytest.approx(100.0)
        assert wells[1].max_pump_rate == pytest.approx(500.0)

        assert wells[2].id == 2
        assert wells[2].name == "Well 2"

    def test_read_wells_with_comments(self, tmp_path: Path) -> None:
        """Test reading wells with various comment styles."""
        wells_file = tmp_path / "wells.dat"
        wells_file.write_text("""C This is a C comment
* This is an asterisk comment
1                               / NWELLS
C Data row
1      1000.0     2000.0     1    50.0   100.0     500.0  Test Well
""")

        reader = GroundwaterReader()
        wells = reader.read_wells(wells_file)

        assert len(wells) == 1
        assert wells[1].name == "Test Well"

    def test_read_wells_missing_nwells(self, tmp_path: Path) -> None:
        """Test error when NWELLS is missing."""
        wells_file = tmp_path / "wells.dat"
        wells_file.write_text("""C Wells file
C Only comments
""")

        reader = GroundwaterReader()

        with pytest.raises(FileFormatError, match="NWELLS"):
            reader.read_wells(wells_file)

    def test_read_wells_invalid_nwells(self, tmp_path: Path) -> None:
        """Test error when NWELLS is invalid."""
        wells_file = tmp_path / "wells.dat"
        wells_file.write_text("""C Wells file
invalid                         / NWELLS
""")

        reader = GroundwaterReader()

        with pytest.raises(FileFormatError, match="Invalid NWELLS"):
            reader.read_wells(wells_file)

    def test_read_wells_no_name(self, tmp_path: Path) -> None:
        """Test reading wells without names."""
        wells_file = tmp_path / "wells.dat"
        wells_file.write_text("""C Wells file
1                               / NWELLS
1      1000.0     2000.0     1    50.0   100.0     500.0
""")

        reader = GroundwaterReader()
        wells = reader.read_wells(wells_file)

        assert len(wells) == 1
        assert wells[1].name == ""  # Empty name
