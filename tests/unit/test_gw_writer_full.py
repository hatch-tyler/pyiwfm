"""
Comprehensive tests for pyiwfm.io.gw_writer module.

Tests the groundwater component writer for IWFM models.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pyiwfm.io.gw_writer import (
    GWComponentWriter,
    GWWriterConfig,
    write_gw_component,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_model():
    """Create a mock IWFMModel."""
    model = MagicMock()
    model.name = "TestModel"
    model.n_nodes = 10
    model.n_elements = 8

    # Mock stratigraphy
    model.stratigraphy = MagicMock()
    model.stratigraphy.n_layers = 2
    model.stratigraphy.top_elev = np.ones((10, 2)) * 100.0
    model.stratigraphy.bottom_elev = np.ones((10, 2)) * 50.0

    # Mock groundwater component (None by default)
    model.groundwater = None

    return model


@pytest.fixture
def mock_model_with_gw(mock_model):
    """Create a mock model with groundwater component."""
    gw = MagicMock()
    gw.boundary_conditions = []
    gw.wells = {}
    gw.element_pumping = {}
    gw.tile_drains = {}
    gw.subsidence = []
    gw.aquifer_params = None
    gw.heads = None
    gw.gw_main_config = None  # No roundtrip config
    gw.n_bc_output_nodes = 0
    gw.bc_output_specs = []
    gw.bc_output_file_raw = ""
    gw.bc_ts_file = None
    gw.bc_config = None
    # Tile drain writer attributes
    gw.td_n_hydro = 0
    gw.td_hydro_volume_factor = 1.0
    gw.td_hydro_volume_unit = ""
    gw.td_hydro_specs = []
    gw.td_output_file_raw = ""
    gw.sub_irrigations = []

    mock_model.groundwater = gw
    return mock_model


@pytest.fixture
def mock_model_full_gw(mock_model):
    """Create a mock model with full groundwater data."""
    gw = MagicMock()

    # Tile drain factors - set explicitly to prevent MagicMock division issues
    gw.td_elev_factor = 1.0
    gw.td_cond_factor = 1.0
    gw.td_time_unit = "1MON"
    gw.si_elev_factor = 1.0
    gw.si_cond_factor = 1.0
    gw.si_time_unit = "1MON"

    # Boundary conditions
    bc1 = SimpleNamespace(
        bc_type="specified_head",
        id=1,
        nodes=[1],
        values=[100.0],
        layer=1,
        ts_column=0,
    )
    gw.boundary_conditions = [bc1]
    gw.n_bc_output_nodes = 0
    gw.bc_output_specs = []
    gw.bc_output_file_raw = ""
    gw.bc_ts_file = None
    gw.bc_config = None
    gw.gw_main_config = None

    # Wells
    well1 = SimpleNamespace(
        id=1,
        element=1,
        x=100.0,
        y=200.0,
        name="Well_1",
        bottom_screen=50.0,
        top_screen=80.0,
    )
    gw.wells = {1: well1}
    gw.element_pumping = []

    # Tile drains - use SimpleNamespace for Jinja2 template compatibility
    drain1 = SimpleNamespace(
        id=1,
        element=5,
        gw_node=5,
        layer=1,
        elevation=75.0,
        conductance=0.01,
        destination_type="stream",
        dest_type=1,  # 1 for stream node
        destination_id=10,
        dest_id=10,
    )
    gw.tile_drains = {1: drain1}
    gw.td_n_hydro = 0
    gw.td_hydro_volume_factor = 1.0
    gw.td_hydro_volume_unit = ""
    gw.td_hydro_specs = []
    gw.td_output_file_raw = ""
    gw.sub_irrigations = []

    # Subsidence
    sub1 = MagicMock()
    sub1.element = 3
    sub1.layer = 1
    sub1.elastic_storage = 1e-5
    sub1.inelastic_storage = 1e-4
    sub1.preconsolidation_head = 80.0
    gw.subsidence = [sub1]

    # Aquifer parameters
    params = MagicMock()
    params.kh = np.ones((10, 2)) * 10.0
    params.kv = np.ones((10, 2)) * 1.0
    params.specific_storage = np.ones((10, 2)) * 1e-5
    params.specific_yield = np.ones((10, 2)) * 0.15
    gw.aquifer_params = params

    # Initial heads
    gw.heads = np.ones((10, 2)) * 90.0

    mock_model.groundwater = gw
    return mock_model


@pytest.fixture
def config(tmp_path):
    """Create a basic GWWriterConfig."""
    return GWWriterConfig(output_dir=tmp_path)


# =============================================================================
# GWWriterConfig Tests
# =============================================================================


class TestGWWriterConfig:
    """Test GWWriterConfig dataclass."""

    def test_default_config(self, tmp_path):
        """Test config with default values."""
        config = GWWriterConfig(output_dir=tmp_path)

        assert config.output_dir == tmp_path
        assert config.gw_subdir == "GW"
        assert config.version == "4.0"
        assert config.main_file == "GW_MAIN.dat"
        assert config.bc_main_file == "BC_MAIN.dat"
        assert config.pump_main_file == "Pump_MAIN.dat"

    def test_custom_config(self, tmp_path):
        """Test config with custom values."""
        config = GWWriterConfig(
            output_dir=tmp_path,
            gw_subdir="Groundwater",
            version="5.0",
            main_file="GW_Control.dat",
            length_unit="m",
        )

        assert config.gw_subdir == "Groundwater"
        assert config.version == "5.0"
        assert config.main_file == "GW_Control.dat"
        assert config.length_unit == "m"

    def test_gw_dir_property(self, tmp_path):
        """Test gw_dir property."""
        config = GWWriterConfig(output_dir=tmp_path, gw_subdir="GW")
        assert config.gw_dir == tmp_path / "GW"

    def test_main_path_property(self, tmp_path):
        """Test main_path property."""
        config = GWWriterConfig(output_dir=tmp_path)
        assert config.main_path == tmp_path / "GW" / "GW_MAIN.dat"

    def test_bc_main_path_property(self, tmp_path):
        """Test bc_main_path property."""
        config = GWWriterConfig(output_dir=tmp_path)
        assert config.bc_main_path == tmp_path / "GW" / "BC_MAIN.dat"

    def test_pump_main_path_property(self, tmp_path):
        """Test pump_main_path property."""
        config = GWWriterConfig(output_dir=tmp_path)
        assert config.pump_main_path == tmp_path / "GW" / "Pump_MAIN.dat"

    def test_unit_conversion_factors(self, tmp_path):
        """Test unit conversion factors."""
        config = GWWriterConfig(
            output_dir=tmp_path,
            length_factor=0.3048,
            length_unit="m",
            volume_factor=1233.48,
            volume_unit="m3",
        )

        assert config.length_factor == 0.3048
        assert config.length_unit == "m"
        assert config.volume_factor == 1233.48
        assert config.volume_unit == "m3"


# =============================================================================
# GWComponentWriter Tests
# =============================================================================


class TestGWComponentWriter:
    """Test GWComponentWriter class."""

    def test_init(self, mock_model, config):
        """Test writer initialization."""
        writer = GWComponentWriter(mock_model, config)

        assert writer.model == mock_model
        assert writer.config == config

    def test_format_property(self, mock_model, config):
        """Test format property."""
        writer = GWComponentWriter(mock_model, config)
        assert writer.format == "iwfm_groundwater"

    def test_write_calls_write_all(self, mock_model, config):
        """Test write method calls write_all."""
        writer = GWComponentWriter(mock_model, config)

        with patch.object(writer, "write_all") as mock_write_all:
            writer.write()
            mock_write_all.assert_called_once()


class TestGWComponentWriterWriteAll:
    """Test write_all method."""

    def test_write_all_no_gw_component(self, mock_model, config):
        """Test write_all with no groundwater component."""
        writer = GWComponentWriter(mock_model, config)
        results = writer.write_all(write_defaults=True)

        assert "main" in results
        assert config.main_path.exists()

    def test_write_all_empty_gw_component(self, mock_model_with_gw, config):
        """Test write_all with empty groundwater component."""
        writer = GWComponentWriter(mock_model_with_gw, config)
        results = writer.write_all()

        assert "main" in results
        # No BC, pump, drain, or subsidence files
        assert "bc_main" not in results
        assert "pump_main" not in results
        assert "tile_drains" not in results
        assert "subsidence" not in results

    def test_write_all_full_gw_component(self, mock_model_full_gw, config):
        """Test write_all with full groundwater component."""
        writer = GWComponentWriter(mock_model_full_gw, config)
        results = writer.write_all()

        assert "main" in results
        assert "bc_main" in results
        assert "pump_main" in results
        assert "tile_drains" in results
        assert "subsidence" in results

    def test_write_all_creates_directory(self, mock_model, tmp_path):
        """Test write_all creates output directory."""
        config = GWWriterConfig(output_dir=tmp_path / "nonexistent")
        writer = GWComponentWriter(mock_model, config)
        writer.write_all()

        assert config.gw_dir.exists()

    def test_write_all_no_defaults(self, mock_model, config):
        """Test write_all with write_defaults=False."""
        mock_model.groundwater = None
        writer = GWComponentWriter(mock_model, config)
        results = writer.write_all(write_defaults=False)

        # Should not write main file when no GW component and no defaults
        assert "main" not in results


class TestGWComponentWriterWriteMain:
    """Test write_main method."""

    def test_write_main_basic(self, mock_model_with_gw, config):
        """Test writing main file."""
        writer = GWComponentWriter(mock_model_with_gw, config)
        path = writer.write_main()

        assert path.exists()
        content = path.read_text()

        # Check header
        assert "INTEGRATED WATER FLOW MODEL" in content
        assert "GROUNDWATER COMPONENT MAIN DATA FILE" in content
        assert "pyiwfm" in content

        # Check sections
        assert "FACTLTOU" in content
        assert "Aquifer Parameter Data" in content
        assert "Initial Groundwater Heads" in content

    def test_write_main_with_aquifer_params(self, mock_model_full_gw, config):
        """Test write_main includes aquifer parameters."""
        writer = GWComponentWriter(mock_model_full_gw, config)
        path = writer.write_main()

        content = path.read_text()

        # Should have aquifer parameters
        assert "KH" in content or "10.0000" in content

    def test_write_main_with_initial_heads(self, mock_model_full_gw, config):
        """Test write_main includes initial heads."""
        writer = GWComponentWriter(mock_model_full_gw, config)
        path = writer.write_main()

        content = path.read_text()

        # Should have initial heads (90.0)
        assert "90.0000" in content

    def test_write_main_default_params_no_gw_data(self, mock_model_with_gw, config):
        """Test write_main with default parameters when no GW data."""
        mock_model_with_gw.groundwater.aquifer_params = None
        mock_model_with_gw.groundwater.heads = None

        writer = GWComponentWriter(mock_model_with_gw, config)
        path = writer.write_main()

        content = path.read_text()

        # Should have default parameters (1.0 for Kh, 0.1 for Kv)
        assert "1.0000" in content
        assert "0.1000" in content

    def test_write_main_references_component_files(self, mock_model_full_gw, config):
        """Test write_main references other component files."""
        writer = GWComponentWriter(mock_model_full_gw, config)
        path = writer.write_main()

        content = path.read_text()

        # Should reference BC, pump, tile drain files
        assert "BC_MAIN.dat" in content
        assert "Pump_MAIN.dat" in content
        assert "TileDrain.dat" in content


class TestGWComponentWriterWriteBCMain:
    """Test write_bc_main method."""

    def test_write_bc_main_basic(self, mock_model_full_gw, config):
        """Test writing BC main file."""
        writer = GWComponentWriter(mock_model_full_gw, config)
        path = writer.write_bc_main()

        assert path.exists()
        content = path.read_text()

        assert "GROUNDWATER BOUNDARY CONDITIONS MAIN FILE" in content
        assert "SHBCFL" in content
        assert "NOUTB" in content

    def test_write_bc_main_no_bcs(self, mock_model_with_gw, config):
        """Test writing BC main file with no boundary conditions."""
        writer = GWComponentWriter(mock_model_with_gw, config)
        path = writer.write_bc_main()

        content = path.read_text()

        # Should have NOUTB
        assert "NOUTB" in content

    def test_write_bc_main_multiple_types(self, mock_model_full_gw, config):
        """Test writing BC main file with multiple BC types."""
        # Add different BC types
        bc_flow = SimpleNamespace(
            bc_type="specified_flow", id=2, nodes=[2], values=[0.0], layer=1, ts_column=0
        )
        bc_gen = SimpleNamespace(
            bc_type="general_head",
            id=3,
            nodes=[3],
            values=[50.0],
            layer=1,
            ts_column=0,
            conductance=[0.1],
        )

        mock_model_full_gw.groundwater.boundary_conditions.extend([bc_flow, bc_gen])

        writer = GWComponentWriter(mock_model_full_gw, config)
        path = writer.write_bc_main()

        content = path.read_text()

        assert "SFBCFL" in content
        assert "SHBCFL" in content
        assert "GHBCFL" in content


class TestGWComponentWriterWritePumpMain:
    """Test write_pump_main method."""

    def test_write_pump_main_with_wells(self, mock_model_full_gw, config):
        """Test writing pump main file with wells."""
        writer = GWComponentWriter(mock_model_full_gw, config)
        path = writer.write_pump_main()

        assert path.exists()
        content = path.read_text()

        assert "PUMPING COMPONENT MAIN FILE" in content
        assert "WELLFL" in content
        assert "PUMPFL" in content

    def test_write_pump_main_with_elem_pump(self, mock_model_full_gw, config):
        """Test writing pump main file with element pumping."""
        # Set up element pumping instead of wells
        mock_model_full_gw.groundwater.wells = {}
        mock_model_full_gw.groundwater.element_pumping = {1: MagicMock()}

        writer = GWComponentWriter(mock_model_full_gw, config)
        path = writer.write_pump_main()

        content = path.read_text()

        assert "ElemPump.dat" in content
        assert "ELEMPUMPFL" in content

    def test_write_pump_main_no_pumping(self, mock_model_with_gw, config):
        """Test writing pump main file with no pumping."""
        writer = GWComponentWriter(mock_model_with_gw, config)
        path = writer.write_pump_main()

        content = path.read_text()

        # NPUMP = 0 for no pumping
        assert "0" in content


class TestGWComponentWriterWriteTileDrains:
    """Test write_tile_drains method."""

    def test_write_tile_drains_basic(self, mock_model_full_gw, config):
        """Test writing tile drains file."""
        writer = GWComponentWriter(mock_model_full_gw, config)
        path = writer.write_tile_drains()

        assert path.exists()
        content = path.read_text()

        assert "TILE DRAINS DATA FILE" in content
        assert "NTD" in content  # Number of tile drains
        assert "1" in content  # One drain

        # Check drain data
        assert "75.000" in content  # elevation
        assert "TYPDST" in content  # destination type column header

    def test_write_tile_drains_empty(self, mock_model_with_gw, config):
        """Test writing tile drains file with no drains."""
        writer = GWComponentWriter(mock_model_with_gw, config)
        path = writer.write_tile_drains()

        content = path.read_text()

        assert "0" in content  # Zero drains

    def test_write_tile_drains_multiple(self, mock_model_full_gw, config):
        """Test writing tile drains file with multiple drains."""
        # Add more drains - use SimpleNamespace for Jinja2 template compatibility
        drain2 = SimpleNamespace(
            id=2,
            element=7,
            gw_node=7,
            layer=2,
            elevation=65.0,
            conductance=0.02,
            destination_type="element",
            dest_type=0,  # 0 for outside/element
            destination_id=8,
            dest_id=8,
        )

        mock_model_full_gw.groundwater.tile_drains[2] = drain2

        writer = GWComponentWriter(mock_model_full_gw, config)
        path = writer.write_tile_drains()

        content = path.read_text()

        assert "2" in content  # Two drains
        assert "7" in content  # Second drain element


class TestGWComponentWriterWriteSubsidence:
    """Test write_subsidence method."""

    def test_write_subsidence_basic(self, mock_model_full_gw, config):
        """Test writing subsidence file."""
        writer = GWComponentWriter(mock_model_full_gw, config)
        path = writer.write_subsidence()

        assert path.exists()
        content = path.read_text()

        assert "SUBSIDENCE PARAMETERS FILE" in content
        assert "NSUBSIDENCE" in content
        assert "1" in content  # One subsidence location

        # Check subsidence data
        assert "3" in content  # element
        assert "80.0000" in content  # preconsolidation head

    def test_write_subsidence_empty(self, mock_model_with_gw, config):
        """Test writing subsidence file with no subsidence."""
        writer = GWComponentWriter(mock_model_with_gw, config)
        path = writer.write_subsidence()

        content = path.read_text()

        assert "0" in content  # Zero subsidence locations

    def test_write_subsidence_multiple(self, mock_model_full_gw, config):
        """Test writing subsidence file with multiple locations."""
        # Add more subsidence
        sub2 = MagicMock()
        sub2.element = 6
        sub2.layer = 2
        sub2.elastic_storage = 2e-5
        sub2.inelastic_storage = 2e-4
        sub2.preconsolidation_head = 75.0

        mock_model_full_gw.groundwater.subsidence.append(sub2)

        writer = GWComponentWriter(mock_model_full_gw, config)
        path = writer.write_subsidence()

        content = path.read_text()

        assert "2" in content  # Two subsidence locations


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestWriteGWComponent:
    """Test write_gw_component convenience function."""

    def test_write_gw_component_basic(self, mock_model, tmp_path):
        """Test basic write_gw_component call."""
        files = write_gw_component(mock_model, tmp_path)

        assert "main" in files
        assert files["main"].exists()

    def test_write_gw_component_with_config(self, mock_model, tmp_path):
        """Test write_gw_component with custom config."""
        config = GWWriterConfig(
            output_dir=tmp_path,
            gw_subdir="CustomGW",
            version="5.0",
        )

        files = write_gw_component(mock_model, tmp_path, config=config)

        assert "main" in files
        assert "CustomGW" in str(files["main"])

    def test_write_gw_component_full_model(self, mock_model_full_gw, tmp_path):
        """Test write_gw_component with full model."""
        files = write_gw_component(mock_model_full_gw, tmp_path)

        assert "main" in files
        assert "bc_main" in files
        assert "pump_main" in files
        assert "tile_drains" in files
        assert "subsidence" in files

    def test_write_gw_component_string_path(self, mock_model, tmp_path):
        """Test write_gw_component with string path."""
        files = write_gw_component(mock_model, str(tmp_path))

        assert "main" in files


# =============================================================================
# Integration Tests
# =============================================================================


class TestGWWriterIntegration:
    """Integration tests for GW writer."""

    def test_full_workflow(self, mock_model_full_gw, tmp_path):
        """Test full workflow from config to file output."""
        config = GWWriterConfig(output_dir=tmp_path)
        writer = GWComponentWriter(mock_model_full_gw, config)

        # Write all files
        files = writer.write_all()

        # Verify all files exist
        for name, path in files.items():
            assert path.exists(), f"File {name} does not exist"

        # Verify main file content
        main_content = files["main"].read_text()
        assert "IWFM" in main_content

        # Verify BC file references spec head file
        bc_content = files["bc_main"].read_text()
        assert "SpecHeadBC.dat" in bc_content

    def test_round_trip_file_verification(self, mock_model_full_gw, tmp_path):
        """Test that written files can be read back."""
        config = GWWriterConfig(output_dir=tmp_path)
        writer = GWComponentWriter(mock_model_full_gw, config)
        files = writer.write_all()

        # Verify all files are valid text files
        for name, path in files.items():
            content = path.read_text()
            assert len(content) > 0, f"File {name} is empty"
            # Check for expected IWFM comment structure
            assert "C" in content or "/" in content, f"File {name} has no comments"

    def test_multiple_writes_overwrite(self, mock_model_full_gw, tmp_path):
        """Test that multiple writes overwrite existing files."""
        config = GWWriterConfig(output_dir=tmp_path)
        writer = GWComponentWriter(mock_model_full_gw, config)

        # Write twice
        files1 = writer.write_all()
        files2 = writer.write_all()

        # Should produce same files
        assert set(files1.keys()) == set(files2.keys())

        # Files should still be valid
        for path in files2.values():
            assert path.exists()

    def test_config_modification_after_init(self, mock_model, tmp_path):
        """Test that config changes are reflected in output."""
        config = GWWriterConfig(output_dir=tmp_path)
        writer = GWComponentWriter(mock_model, config)

        # Modify config
        config.gw_subdir = "ModifiedGW"

        # Write files
        files = writer.write_all()

        # Check path includes modified subdir
        assert "ModifiedGW" in str(files["main"])
