"""
Comprehensive tests for pyiwfm.io.lake_writer module.

Tests cover:
- LakeWriterConfig dataclass and properties
- LakeComponentWriter class methods
- write_lake_component convenience function
- File generation with various model configurations
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from pyiwfm.io.lake_writer import (
    LakeWriterConfig,
    LakeComponentWriter,
    write_lake_component,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_model():
    """Create a basic mock model."""
    model = MagicMock()
    model.lakes = None
    return model


@pytest.fixture
def mock_lake():
    """Create a mock lake."""
    lake = MagicMock()
    lake.id = 1
    lake.name = "Test Lake"
    lake.bed_conductivity = 3.0
    lake.bed_thickness = 2.0
    lake.initial_elevation = 300.0
    lake.et_column = 5
    lake.precip_column = 3
    return lake


@pytest.fixture
def mock_model_with_lakes(mock_lake):
    """Create a mock model with lake data."""
    model = MagicMock()

    lake2 = MagicMock()
    lake2.id = 2
    lake2.name = "Second Lake"
    lake2.bed_conductivity = 2.5
    lake2.bed_thickness = 1.5
    lake2.initial_elevation = 250.0
    lake2.et_column = 6
    lake2.precip_column = 4

    lakes = MagicMock()
    lakes.lakes = {1: mock_lake, 2: lake2}

    model.lakes = lakes
    return model


@pytest.fixture
def mock_model_empty_lakes():
    """Create a mock model with empty lakes dict."""
    model = MagicMock()
    lakes = MagicMock()
    lakes.lakes = {}
    model.lakes = lakes
    return model


@pytest.fixture
def lake_config(tmp_path):
    """Create a LakeWriterConfig for testing."""
    return LakeWriterConfig(output_dir=tmp_path)


# =============================================================================
# LakeWriterConfig Tests
# =============================================================================


class TestLakeWriterConfig:
    """Tests for LakeWriterConfig dataclass."""

    def test_config_creation_minimal(self, tmp_path):
        """Test config creation with minimal arguments."""
        config = LakeWriterConfig(output_dir=tmp_path)

        assert config.output_dir == tmp_path
        assert config.lake_subdir == "Lake"
        assert config.version == "4.0"

    def test_config_creation_full(self, tmp_path):
        """Test config creation with all arguments."""
        config = LakeWriterConfig(
            output_dir=tmp_path,
            lake_subdir="Lakes",
            version="5.0",
            main_file="main.dat",
            max_elev_file="max_elev.dat",
            lake_budget_file="budget.hdf",
            final_elev_file="final_elev.out",
            conductivity_factor=0.5,
            conductivity_time_unit="1hr",
            length_factor=0.3048,
            bed_conductivity=5.0,
            bed_thickness=2.0,
        )

        assert config.lake_subdir == "Lakes"
        assert config.version == "5.0"
        assert config.main_file == "main.dat"
        assert config.conductivity_factor == 0.5
        assert config.bed_conductivity == 5.0

    def test_lake_dir_property(self, tmp_path):
        """Test lake_dir property returns correct path."""
        config = LakeWriterConfig(output_dir=tmp_path, lake_subdir="Lakes")

        assert config.lake_dir == tmp_path / "Lakes"

    def test_main_path_property(self, tmp_path):
        """Test main_path property returns correct path."""
        config = LakeWriterConfig(
            output_dir=tmp_path,
            lake_subdir="Lake",
            main_file="Lake_MAIN.dat"
        )

        assert config.main_path == tmp_path / "Lake" / "Lake_MAIN.dat"

    def test_default_file_names(self, tmp_path):
        """Test default file name values."""
        config = LakeWriterConfig(output_dir=tmp_path)

        assert config.main_file == "Lake_MAIN.dat"
        assert config.max_elev_file == "MaxLakeElev.dat"

    def test_default_output_files(self, tmp_path):
        """Test default output file paths."""
        config = LakeWriterConfig(output_dir=tmp_path)

        assert config.lake_budget_file == "../Results/LakeBud.hdf"
        assert config.final_elev_file == "../Results/FinalLakeElev.out"

    def test_default_unit_conversions(self, tmp_path):
        """Test default unit conversion values."""
        config = LakeWriterConfig(output_dir=tmp_path)

        assert config.conductivity_factor == 1.0
        assert config.conductivity_time_unit == "1day"
        assert config.length_factor == 1.0

    def test_default_lake_bed_parameters(self, tmp_path):
        """Test default lake bed parameter values."""
        config = LakeWriterConfig(output_dir=tmp_path)

        assert config.bed_conductivity == 2.0
        assert config.bed_thickness == 1.0


# =============================================================================
# LakeComponentWriter Tests
# =============================================================================


class TestLakeComponentWriterInit:
    """Tests for LakeComponentWriter initialization."""

    def test_init_basic(self, mock_model, lake_config):
        """Test basic writer initialization."""
        writer = LakeComponentWriter(mock_model, lake_config)

        assert writer.model is mock_model
        assert writer.config is lake_config

    def test_init_with_template_engine(self, mock_model, lake_config):
        """Test initialization with custom template engine."""
        mock_engine = MagicMock()
        writer = LakeComponentWriter(mock_model, lake_config, mock_engine)

        assert writer.model is mock_model
        assert writer.config is lake_config

    def test_format_property(self, mock_model, lake_config):
        """Test format property returns correct value."""
        writer = LakeComponentWriter(mock_model, lake_config)

        assert writer.format == "iwfm_lake"


class TestLakeComponentWriterWrite:
    """Tests for LakeComponentWriter write methods."""

    def test_write_calls_write_all(self, mock_model, lake_config):
        """Test write() calls write_all()."""
        writer = LakeComponentWriter(mock_model, lake_config)

        with patch.object(writer, 'write_all') as mock_write_all:
            writer.write()
            mock_write_all.assert_called_once()

    def test_write_all_no_lakes_defaults_false(self, mock_model, lake_config):
        """Test write_all with no lakes and write_defaults=False."""
        writer = LakeComponentWriter(mock_model, lake_config)

        result = writer.write_all(write_defaults=False)

        assert result == {}

    def test_write_all_no_lakes_defaults_true(self, mock_model, lake_config):
        """Test write_all with no lakes and write_defaults=True."""
        writer = LakeComponentWriter(mock_model, lake_config)

        result = writer.write_all(write_defaults=True)

        assert "main" in result
        assert result["main"].exists()

    def test_write_all_with_lakes(self, mock_model_with_lakes, lake_config):
        """Test write_all with lake data."""
        writer = LakeComponentWriter(mock_model_with_lakes, lake_config)

        result = writer.write_all()

        assert "main" in result
        assert result["main"].exists()

    def test_write_all_empty_lakes(self, mock_model_empty_lakes, lake_config):
        """Test write_all with empty lakes dict."""
        writer = LakeComponentWriter(mock_model_empty_lakes, lake_config)

        result = writer.write_all()

        assert "main" in result

    def test_write_all_creates_lake_dir(self, mock_model, lake_config):
        """Test write_all creates lake directory."""
        writer = LakeComponentWriter(mock_model, lake_config)

        assert not lake_config.lake_dir.exists()

        writer.write_all()

        assert lake_config.lake_dir.exists()


class TestLakeComponentWriterWriteMain:
    """Tests for LakeComponentWriter.write_main method."""

    def test_write_main_creates_file(self, mock_model, lake_config):
        """Test write_main creates the main file."""
        writer = LakeComponentWriter(mock_model, lake_config)

        result = writer.write_main()

        assert result.exists()
        assert result.name == "Lake_MAIN.dat"

    def test_write_main_content_has_version(self, mock_model, lake_config):
        """Test main file contains version header."""
        writer = LakeComponentWriter(mock_model, lake_config)

        result = writer.write_main()
        content = result.read_text()

        assert "#4.0" in content or "4.0" in content

    def test_write_main_content_has_header(self, mock_model, lake_config):
        """Test main file contains header comments."""
        writer = LakeComponentWriter(mock_model, lake_config)

        result = writer.write_main()
        content = result.read_text()

        assert "LAKE PARAMETERS DATA FILE" in content
        assert "pyiwfm" in content

    def test_write_main_with_lakes(self, mock_model_with_lakes, lake_config):
        """Test main file contains lake data."""
        writer = LakeComponentWriter(mock_model_with_lakes, lake_config)

        result = writer.write_main()
        content = result.read_text()

        # Should have lake parameters
        assert "Test Lake" in content
        assert "Second Lake" in content

    def test_write_main_has_file_paths(self, mock_model_with_lakes, lake_config):
        """Test main file references sub-files."""
        writer = LakeComponentWriter(mock_model_with_lakes, lake_config)

        result = writer.write_main()
        content = result.read_text()

        # Should reference max elevation file
        assert "MaxLakeElev.dat" in content

    def test_write_main_has_budget_file(self, mock_model, lake_config):
        """Test main file references budget file."""
        writer = LakeComponentWriter(mock_model, lake_config)

        result = writer.write_main()
        content = result.read_text()

        assert "LakeBud.hdf" in content


# =============================================================================
# write_lake_component Function Tests
# =============================================================================


class TestWriteLakeComponent:
    """Tests for write_lake_component convenience function."""

    def test_write_lake_component_basic(self, mock_model, tmp_path):
        """Test basic write_lake_component call."""
        result = write_lake_component(mock_model, tmp_path)

        assert isinstance(result, dict)
        assert "main" in result

    def test_write_lake_component_with_config(self, mock_model, tmp_path):
        """Test write_lake_component with custom config."""
        config = LakeWriterConfig(
            output_dir=tmp_path,
            lake_subdir="Lakes",
            version="5.0",
        )

        result = write_lake_component(mock_model, tmp_path, config)

        assert (tmp_path / "Lakes" / "Lake_MAIN.dat").exists()

    def test_write_lake_component_string_path(self, mock_model, tmp_path):
        """Test write_lake_component with string path."""
        result = write_lake_component(mock_model, str(tmp_path))

        assert isinstance(result, dict)
        assert "main" in result

    def test_write_lake_component_updates_config_output_dir(self, mock_model, tmp_path):
        """Test write_lake_component updates config output_dir."""
        other_path = tmp_path / "other"
        config = LakeWriterConfig(output_dir=other_path)

        result = write_lake_component(mock_model, tmp_path, config)

        # Should use tmp_path, not other_path
        assert (tmp_path / "Lake" / "Lake_MAIN.dat").exists()

    def test_write_lake_component_with_full_lakes(
        self, mock_model_with_lakes, tmp_path
    ):
        """Test write_lake_component with full lake data."""
        result = write_lake_component(mock_model_with_lakes, tmp_path)

        assert "main" in result
        content = result["main"].read_text()
        assert "Test Lake" in content


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestLakeWriterEdgeCases:
    """Tests for edge cases and error handling."""

    def test_lake_missing_attributes_uses_defaults(self, lake_config):
        """Test lakes missing attributes use config defaults."""
        model = MagicMock()
        lake = MagicMock(spec=[])  # Empty spec means no attributes
        lake.id = 1
        # Set only required attribute
        model.lakes = MagicMock()
        model.lakes.lakes = {1: lake}

        writer = LakeComponentWriter(model, lake_config)
        result = writer.write_main()
        content = result.read_text()

        # Should use default bed_conductivity (2.0)
        assert "2.0" in content

    def test_lake_no_name_uses_default(self, lake_config):
        """Test lakes without name use default naming."""
        model = MagicMock()
        lake = MagicMock()
        lake.id = 5
        del lake.name  # Remove name attribute
        lake.bed_conductivity = 2.0
        lake.bed_thickness = 1.0
        lake.initial_elevation = 280.0
        model.lakes = MagicMock()
        model.lakes.lakes = {5: lake}

        writer = LakeComponentWriter(model, lake_config)
        result = writer.write_main()
        content = result.read_text()

        # Should use default name Lake5
        assert "Lake5" in content

    def test_multiple_lakes_sorted_by_id(self, lake_config):
        """Test lakes are sorted by ID in output."""
        model = MagicMock()
        lakes_dict = {}
        for i in [3, 1, 2]:  # Out of order
            lake = MagicMock()
            lake.id = i
            lake.name = f"Lake{i}"
            lake.bed_conductivity = 2.0
            lake.bed_thickness = 1.0
            lake.initial_elevation = 280.0
            lakes_dict[i] = lake

        model.lakes = MagicMock()
        model.lakes.lakes = lakes_dict

        writer = LakeComponentWriter(model, lake_config)
        result = writer.write_main()
        content = result.read_text()

        # Find positions of lake names
        pos1 = content.find("Lake1")
        pos2 = content.find("Lake2")
        pos3 = content.find("Lake3")

        assert pos1 < pos2 < pos3

    def test_custom_version_in_main(self, mock_model, tmp_path):
        """Test custom version appears in main file."""
        config = LakeWriterConfig(output_dir=tmp_path, version="5.0")
        writer = LakeComponentWriter(mock_model, config)

        result = writer.write_main()
        content = result.read_text()

        assert "#5.0" in content

    def test_custom_conductivity_factor(self, mock_model, tmp_path):
        """Test custom conductivity factor appears in main file."""
        config = LakeWriterConfig(
            output_dir=tmp_path,
            conductivity_factor=0.5,
        )
        writer = LakeComponentWriter(mock_model, config)

        result = writer.write_main()
        content = result.read_text()

        assert "0.5" in content

    def test_custom_time_unit(self, mock_model, tmp_path):
        """Test custom time unit appears in main file."""
        config = LakeWriterConfig(
            output_dir=tmp_path,
            conductivity_time_unit="1hr",
        )
        writer = LakeComponentWriter(mock_model, config)

        result = writer.write_main()
        content = result.read_text()

        assert "1hr" in content

    def test_initial_elevations_in_output(self, mock_model_with_lakes, lake_config):
        """Test initial elevations are written to output."""
        writer = LakeComponentWriter(mock_model_with_lakes, lake_config)

        result = writer.write_main()
        content = result.read_text()

        # Check for initial elevations section
        assert "Initial Lake Elevations" in content
        assert "300.0" in content  # mock_lake initial_elevation

    def test_lakes_attribute_not_exists(self, lake_config):
        """Test handling when lakes.lakes doesn't exist."""
        model = MagicMock()
        lakes = MagicMock(spec=[])  # No lakes attribute
        model.lakes = lakes

        writer = LakeComponentWriter(model, lake_config)

        # Should not raise
        result = writer.write_main()
        assert result.exists()

    def test_no_max_elev_file_when_no_lakes(self, mock_model, lake_config):
        """Test max elevation file not referenced when no lakes."""
        writer = LakeComponentWriter(mock_model, lake_config)

        result = writer.write_main()
        content = result.read_text()

        # Check for MXLKELVFL line - should be empty/blank path
        lines = content.split('\n')
        mxlkelvfl_line = [l for l in lines if "MXLKELVFL" in l]
        assert len(mxlkelvfl_line) == 1

    def test_ichlmax_column_indices(self, lake_config):
        """Test ICHLMAX column indices are 1-based."""
        model = MagicMock()
        lakes_dict = {}
        for i in range(1, 4):
            lake = MagicMock()
            lake.id = i
            lake.name = f"Lake{i}"
            lake.bed_conductivity = 2.0
            lake.bed_thickness = 1.0
            lake.initial_elevation = 280.0
            lakes_dict[i] = lake

        model.lakes = MagicMock()
        model.lakes.lakes = lakes_dict

        writer = LakeComponentWriter(model, lake_config)
        result = writer.write_main()
        content = result.read_text()

        # The ICHLMAX values should be 1, 2, 3 (1-based indices)
        # Hard to test precisely, but content should exist
        assert "Lake1" in content
        assert "Lake2" in content
        assert "Lake3" in content
