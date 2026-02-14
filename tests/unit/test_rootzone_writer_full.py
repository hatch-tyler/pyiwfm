"""
Comprehensive tests for pyiwfm.io.rootzone_writer module.

Tests cover:
- RootZoneWriterConfig dataclass and properties
- RootZoneComponentWriter class methods
- write_rootzone_component convenience function
- File generation with various model configurations
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from pyiwfm.io.rootzone_writer import (
    RootZoneWriterConfig,
    RootZoneComponentWriter,
    write_rootzone_component,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_model():
    """Create a basic mock model."""
    model = MagicMock()
    model.rootzone = None
    model.grid = None
    return model


@pytest.fixture
def mock_element():
    """Create a mock element."""
    elem = MagicMock()
    elem.id = 1
    return elem


@pytest.fixture
def mock_model_with_grid(mock_element):
    """Create a mock model with grid."""
    model = MagicMock()
    model.rootzone = None

    elem2 = MagicMock()
    elem2.id = 2
    elem3 = MagicMock()
    elem3.id = 3

    model.grid = MagicMock()
    model.grid.elements = {1: mock_element, 2: elem2, 3: elem3}

    return model


@pytest.fixture
def mock_soil_params():
    """Create mock soil parameters."""
    sp = MagicMock()
    sp.wilting_point = 0.1
    sp.field_capacity = 0.25
    sp.total_porosity = 0.50
    sp.pore_size_index = 0.7
    sp.hydraulic_conductivity = 3.0
    sp.k_ponded = -1.0
    sp.rhc_method = 2
    sp.capillary_rise = 0.5
    return sp


@pytest.fixture
def mock_model_with_rootzone(mock_model_with_grid, mock_soil_params):
    """Create a mock model with rootzone component."""
    rootzone = MagicMock()
    rootzone.soil_params = {1: mock_soil_params, 2: mock_soil_params}

    mock_model_with_grid.rootzone = rootzone
    return mock_model_with_grid


@pytest.fixture
def rootzone_config(tmp_path):
    """Create a RootZoneWriterConfig for testing."""
    return RootZoneWriterConfig(output_dir=tmp_path)


# =============================================================================
# RootZoneWriterConfig Tests
# =============================================================================


class TestRootZoneWriterConfig:
    """Tests for RootZoneWriterConfig dataclass."""

    def test_config_creation_minimal(self, tmp_path):
        """Test config creation with minimal arguments."""
        config = RootZoneWriterConfig(output_dir=tmp_path)

        assert config.output_dir == tmp_path
        assert config.rootzone_subdir == "RootZone"
        assert config.version == "4.12"

    def test_config_creation_full(self, tmp_path):
        """Test config creation with all arguments."""
        config = RootZoneWriterConfig(
            output_dir=tmp_path,
            rootzone_subdir="RZ",
            version="5.0",
            main_file="main.dat",
            return_flow_file="rf.dat",
            reuse_file="ru.dat",
            irig_period_file="ip.dat",
            surface_flow_dest_file="dest.dat",
            lwu_budget_file="lwu.hdf",
            rz_budget_file="rz.hdf",
            lwu_zbudget_file="lwuz.hdf",
            rz_zbudget_file="rzz.hdf",
            convergence=0.01,
            max_iterations=100,
            inch_to_length_factor=0.1,
            gw_uptake=1,
            wilting_point=0.05,
            field_capacity=0.25,
            total_porosity=0.5,
            pore_size_index=0.7,
            hydraulic_conductivity=3.0,
            k_ponded=-1.0,
            rhc_method=1,
            capillary_rise=0.5,
            k_factor=0.05,
            cprise_factor=1.5,
            k_time_unit="1min",
        )

        assert config.rootzone_subdir == "RZ"
        assert config.version == "5.0"
        assert config.convergence == 0.01
        assert config.wilting_point == 0.05
        assert config.hydraulic_conductivity == 3.0

    def test_rootzone_dir_property(self, tmp_path):
        """Test rootzone_dir property returns correct path."""
        config = RootZoneWriterConfig(output_dir=tmp_path, rootzone_subdir="RZ")

        assert config.rootzone_dir == tmp_path / "RZ"

    def test_main_path_property(self, tmp_path):
        """Test main_path property returns correct path."""
        config = RootZoneWriterConfig(
            output_dir=tmp_path,
            rootzone_subdir="RootZone",
            main_file="RootZone_MAIN.dat"
        )

        assert config.main_path == tmp_path / "RootZone" / "RootZone_MAIN.dat"

    def test_default_file_names(self, tmp_path):
        """Test default file name values."""
        config = RootZoneWriterConfig(output_dir=tmp_path)

        assert config.main_file == "RootZone_MAIN.dat"
        assert config.return_flow_file == "ReturnFlowFrac.dat"
        assert config.reuse_file == "ReuseFrac.dat"
        assert config.irig_period_file == "IrigPeriod.dat"
        assert config.surface_flow_dest_file == "SurfaceFlowDest.dat"

    def test_default_output_files(self, tmp_path):
        """Test default output file paths."""
        config = RootZoneWriterConfig(output_dir=tmp_path)

        assert config.lwu_budget_file == "../Results/LWU.hdf"
        assert config.rz_budget_file == "../Results/RootZone.hdf"
        assert config.lwu_zbudget_file == "../Results/LWU_ZBud.hdf"
        assert config.rz_zbudget_file == "../Results/RootZone_ZBud.hdf"

    def test_default_simulation_parameters(self, tmp_path):
        """Test default simulation parameter values."""
        config = RootZoneWriterConfig(output_dir=tmp_path)

        assert config.convergence == 0.001
        assert config.max_iterations == 150
        assert config.inch_to_length_factor == pytest.approx(0.0833333, rel=1e-5)
        assert config.gw_uptake == 0

    def test_default_soil_parameters(self, tmp_path):
        """Test default soil parameter values."""
        config = RootZoneWriterConfig(output_dir=tmp_path)

        assert config.wilting_point == 0.0
        assert config.field_capacity == 0.20
        assert config.total_porosity == 0.45
        assert config.pore_size_index == 0.62
        assert config.hydraulic_conductivity == 2.60
        assert config.k_ponded == -1.0
        assert config.rhc_method == 2
        assert config.capillary_rise == 0.0

    def test_default_unit_conversions(self, tmp_path):
        """Test default unit conversion values."""
        config = RootZoneWriterConfig(output_dir=tmp_path)

        assert config.k_factor == pytest.approx(0.03281, rel=1e-5)
        assert config.cprise_factor == 1.0
        assert config.k_time_unit == "1hour"


# =============================================================================
# RootZoneComponentWriter Tests
# =============================================================================


class TestRootZoneComponentWriterInit:
    """Tests for RootZoneComponentWriter initialization."""

    def test_init_basic(self, mock_model, rootzone_config):
        """Test basic writer initialization."""
        writer = RootZoneComponentWriter(mock_model, rootzone_config)

        assert writer.model is mock_model
        assert writer.config is rootzone_config

    def test_init_with_template_engine(self, mock_model, rootzone_config):
        """Test initialization with custom template engine."""
        mock_engine = MagicMock()
        writer = RootZoneComponentWriter(mock_model, rootzone_config, mock_engine)

        assert writer.model is mock_model
        assert writer.config is rootzone_config

    def test_format_property(self, mock_model, rootzone_config):
        """Test format property returns correct value."""
        writer = RootZoneComponentWriter(mock_model, rootzone_config)

        assert writer.format == "iwfm_rootzone"


class TestRootZoneComponentWriterWrite:
    """Tests for RootZoneComponentWriter write methods."""

    def test_write_calls_write_all(self, mock_model, rootzone_config):
        """Test write() calls write_all()."""
        writer = RootZoneComponentWriter(mock_model, rootzone_config)

        with patch.object(writer, 'write_all') as mock_write_all:
            writer.write()
            mock_write_all.assert_called_once()

    def test_write_all_no_rootzone_defaults_false(self, mock_model, rootzone_config):
        """Test write_all with no rootzone and write_defaults=False."""
        writer = RootZoneComponentWriter(mock_model, rootzone_config)

        result = writer.write_all(write_defaults=False)

        assert result == {}

    def test_write_all_no_rootzone_defaults_true(self, mock_model, rootzone_config):
        """Test write_all with no rootzone and write_defaults=True."""
        writer = RootZoneComponentWriter(mock_model, rootzone_config)

        result = writer.write_all(write_defaults=True)

        assert "main" in result
        assert result["main"].exists()

    def test_write_all_with_rootzone(self, mock_model_with_rootzone, rootzone_config):
        """Test write_all with rootzone data."""
        writer = RootZoneComponentWriter(mock_model_with_rootzone, rootzone_config)

        result = writer.write_all()

        assert "main" in result
        assert result["main"].exists()

    def test_write_all_creates_rootzone_dir(self, mock_model, rootzone_config):
        """Test write_all creates rootzone directory."""
        writer = RootZoneComponentWriter(mock_model, rootzone_config)

        assert not rootzone_config.rootzone_dir.exists()

        writer.write_all()

        assert rootzone_config.rootzone_dir.exists()


class TestRootZoneComponentWriterWriteMain:
    """Tests for RootZoneComponentWriter.write_main method."""

    def test_write_main_creates_file(self, mock_model, rootzone_config):
        """Test write_main creates the main file."""
        writer = RootZoneComponentWriter(mock_model, rootzone_config)

        result = writer.write_main()

        assert result.exists()
        assert result.name == "RootZone_MAIN.dat"

    def test_write_main_content_has_version(self, mock_model, rootzone_config):
        """Test main file contains version header."""
        writer = RootZoneComponentWriter(mock_model, rootzone_config)

        result = writer.write_main()
        content = result.read_text()

        assert "#4.12" in content or "4.12" in content

    def test_write_main_content_has_header(self, mock_model, rootzone_config):
        """Test main file contains header comments."""
        writer = RootZoneComponentWriter(mock_model, rootzone_config)

        result = writer.write_main()
        content = result.read_text()

        assert "ROOT ZONE PARAMETERS DATA FILE" in content
        assert "pyiwfm" in content

    def test_write_main_with_elements(self, mock_model_with_grid, rootzone_config):
        """Test main file contains element data."""
        writer = RootZoneComponentWriter(mock_model_with_grid, rootzone_config)

        result = writer.write_main()
        content = result.read_text()

        # Should have soil parameters section header
        assert "WP" in content
        assert "FC" in content
        assert "TN" in content

    def test_write_main_has_file_paths(self, mock_model, rootzone_config):
        """Test main file references sub-files."""
        writer = RootZoneComponentWriter(mock_model, rootzone_config)

        result = writer.write_main()
        content = result.read_text()

        # Should reference return flow and reuse files
        assert "ReturnFlowFrac.dat" in content
        assert "ReuseFrac.dat" in content

    def test_write_main_has_budget_files(self, mock_model, rootzone_config):
        """Test main file references budget files."""
        writer = RootZoneComponentWriter(mock_model, rootzone_config)

        result = writer.write_main()
        content = result.read_text()

        assert "LWU.hdf" in content
        assert "RootZone.hdf" in content

    def test_write_main_has_convergence_params(self, mock_model, rootzone_config):
        """Test main file contains convergence parameters."""
        writer = RootZoneComponentWriter(mock_model, rootzone_config)

        result = writer.write_main()
        content = result.read_text()

        assert "RZCONV" in content
        assert "RZITERMX" in content


# =============================================================================
# write_rootzone_component Function Tests
# =============================================================================


class TestWriteRootZoneComponent:
    """Tests for write_rootzone_component convenience function."""

    def test_write_rootzone_component_basic(self, mock_model, tmp_path):
        """Test basic write_rootzone_component call."""
        result = write_rootzone_component(mock_model, tmp_path)

        assert isinstance(result, dict)
        assert "main" in result

    def test_write_rootzone_component_with_config(self, mock_model, tmp_path):
        """Test write_rootzone_component with custom config."""
        config = RootZoneWriterConfig(
            output_dir=tmp_path,
            rootzone_subdir="RZ",
            version="5.0",
        )

        result = write_rootzone_component(mock_model, tmp_path, config)

        assert (tmp_path / "RZ" / "RootZone_MAIN.dat").exists()

    def test_write_rootzone_component_string_path(self, mock_model, tmp_path):
        """Test write_rootzone_component with string path."""
        result = write_rootzone_component(mock_model, str(tmp_path))

        assert isinstance(result, dict)
        assert "main" in result

    def test_write_rootzone_component_updates_config_output_dir(self, mock_model, tmp_path):
        """Test write_rootzone_component updates config output_dir."""
        other_path = tmp_path / "other"
        config = RootZoneWriterConfig(output_dir=other_path)

        result = write_rootzone_component(mock_model, tmp_path, config)

        # Should use tmp_path, not other_path
        assert (tmp_path / "RootZone" / "RootZone_MAIN.dat").exists()

    def test_write_rootzone_component_with_full_rootzone(
        self, mock_model_with_rootzone, tmp_path
    ):
        """Test write_rootzone_component with full rootzone data."""
        result = write_rootzone_component(mock_model_with_rootzone, tmp_path)

        assert "main" in result


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestRootZoneWriterEdgeCases:
    """Tests for edge cases and error handling."""

    def test_elements_use_soil_params_when_available(
        self, mock_model_with_rootzone, rootzone_config
    ):
        """Test elements use soil parameters when available."""
        writer = RootZoneComponentWriter(mock_model_with_rootzone, rootzone_config)

        result = writer.write_main()
        content = result.read_text()

        # Should have custom wilting point 0.1 for elements with soil params
        assert "0.1" in content

    def test_elements_use_defaults_when_no_soil_params(
        self, mock_model_with_grid, rootzone_config
    ):
        """Test elements use defaults when no soil parameters."""
        writer = RootZoneComponentWriter(mock_model_with_grid, rootzone_config)

        result = writer.write_main()
        content = result.read_text()

        # Should use default values
        assert "0.0" in content  # default wilting point
        assert "0.20" in content  # default field capacity

    def test_custom_version_in_main(self, mock_model, tmp_path):
        """Test custom version appears in main file."""
        config = RootZoneWriterConfig(output_dir=tmp_path, version="5.0")
        writer = RootZoneComponentWriter(mock_model, config)

        result = writer.write_main()
        content = result.read_text()

        assert "#5.0" in content

    def test_custom_convergence_params(self, mock_model, tmp_path):
        """Test custom convergence params appear in main file."""
        config = RootZoneWriterConfig(
            output_dir=tmp_path,
            convergence=0.01,
            max_iterations=200,
        )
        writer = RootZoneComponentWriter(mock_model, config)

        result = writer.write_main()
        content = result.read_text()

        assert "0.01" in content
        assert "200" in content

    def test_custom_soil_defaults(self, mock_model_with_grid, tmp_path):
        """Test custom soil defaults appear in main file."""
        config = RootZoneWriterConfig(
            output_dir=tmp_path,
            wilting_point=0.05,
            field_capacity=0.30,
            total_porosity=0.50,
        )
        writer = RootZoneComponentWriter(mock_model_with_grid, config)

        result = writer.write_main()
        content = result.read_text()

        # Check all elements have custom defaults
        assert "0.30" in content

    def test_no_grid_produces_empty_elements(self, mock_model, rootzone_config):
        """Test no grid produces no element rows."""
        writer = RootZoneComponentWriter(mock_model, rootzone_config)

        result = writer.write_main()
        content = result.read_text()

        # Should still be valid file with headers
        assert "ROOT ZONE" in content

    def test_rootzone_without_soil_params_attribute(self, mock_model_with_grid, rootzone_config):
        """Test rootzone without soil_params attribute uses defaults."""
        rootzone = MagicMock(spec=[])  # No soil_params
        mock_model_with_grid.rootzone = rootzone

        writer = RootZoneComponentWriter(mock_model_with_grid, rootzone_config)

        result = writer.write_main()
        content = result.read_text()

        # Should use default values
        assert "0.0" in content  # default wilting point

    def test_soil_params_missing_for_some_elements(self, mock_model_with_grid, mock_soil_params, rootzone_config):
        """Test elements without soil params use defaults."""
        rootzone = MagicMock()
        # Only element 1 has soil params
        rootzone.soil_params = {1: mock_soil_params}
        mock_model_with_grid.rootzone = rootzone

        writer = RootZoneComponentWriter(mock_model_with_grid, rootzone_config)

        result = writer.write_main()
        content = result.read_text()

        # Should have both custom (0.1 wilting point) and default (0.0)
        assert "0.1" in content  # custom from mock_soil_params
        assert "0.0" in content  # default for elements without params

    def test_unit_conversion_factors_in_output(self, mock_model, tmp_path):
        """Test unit conversion factors appear in output."""
        config = RootZoneWriterConfig(
            output_dir=tmp_path,
            k_factor=0.05,
            cprise_factor=2.0,
            k_time_unit="1min",
        )
        writer = RootZoneComponentWriter(mock_model, config)

        result = writer.write_main()
        content = result.read_text()

        assert "0.05" in content
        assert "1min" in content

    def test_gw_uptake_flag(self, mock_model, tmp_path):
        """Test GW uptake flag appears in output."""
        config = RootZoneWriterConfig(
            output_dir=tmp_path,
            gw_uptake=1,
        )
        writer = RootZoneComponentWriter(mock_model, config)

        result = writer.write_main()
        content = result.read_text()

        assert "GWUPTK" in content
