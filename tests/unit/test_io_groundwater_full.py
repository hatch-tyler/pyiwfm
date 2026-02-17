"""Unit tests for groundwater I/O module.

Tests:
- _is_comment_line function
- _strip_comment function
- GWFileConfig dataclass
- GroundwaterWriter class
- GroundwaterReader class
- Convenience functions
- Roundtrip read/write
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from pyiwfm.io.groundwater import (
    GWFileConfig,
    GroundwaterWriter,
    GroundwaterReader,
    write_groundwater,
    read_wells,
    read_initial_heads,
    _is_comment_line,
    _strip_comment,
)
from pyiwfm.components.groundwater import (
    AppGW,
    Well,
    BoundaryCondition,
    TileDrain,
    Subsidence,
    AquiferParameters,
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
        assert _is_comment_line("10 / NWELLS") is False
        assert _is_comment_line("1  100.0  200.0  5") is False


class TestParseValueLine:
    """Tests for _strip_comment function."""

    def test_with_description(self) -> None:
        """Test parsing line with description."""
        value, desc = _strip_comment("10  / NWELLS")
        assert value == "10"
        assert desc == "NWELLS"

    def test_without_description(self) -> None:
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

    def test_custom_filenames(self, tmp_path: Path) -> None:
        """Test config with custom filenames."""
        config = GWFileConfig(
            output_dir=tmp_path,
            wells_file="custom_wells.dat",
            pumping_file="custom_pumping.dat",
        )

        assert config.wells_file == "custom_wells.dat"
        assert config.pumping_file == "custom_pumping.dat"

    def test_path_methods(self, tmp_path: Path) -> None:
        """Test path getter methods."""
        config = GWFileConfig(output_dir=tmp_path)

        assert config.get_wells_path() == tmp_path / "wells.dat"
        assert config.get_pumping_path() == tmp_path / "pumping.dat"
        assert config.get_aquifer_params_path() == tmp_path / "aquifer_params.dat"
        assert config.get_boundary_conditions_path() == tmp_path / "boundary_conditions.dat"
        assert config.get_tile_drains_path() == tmp_path / "tile_drains.dat"
        assert config.get_subsidence_path() == tmp_path / "subsidence.dat"
        assert config.get_initial_heads_path() == tmp_path / "initial_heads.dat"


# =============================================================================
# Test GroundwaterWriter
# =============================================================================


class TestGroundwaterWriter:
    """Tests for GroundwaterWriter class."""

    @pytest.fixture
    def config(self, tmp_path: Path) -> GWFileConfig:
        """Create test config."""
        return GWFileConfig(output_dir=tmp_path)

    @pytest.fixture
    def basic_gw(self) -> AppGW:
        """Create basic groundwater component for testing."""
        gw = AppGW(n_nodes=10, n_layers=2, n_elements=5)

        # Add wells
        gw.add_well(Well(id=1, x=100.0, y=200.0, element=1, name="Well 1",
                        top_screen=50.0, bottom_screen=10.0, max_pump_rate=100.0))
        gw.add_well(Well(id=2, x=300.0, y=400.0, element=3, name="Well 2",
                        top_screen=45.0, bottom_screen=15.0, max_pump_rate=150.0))

        # Add boundary conditions
        gw.add_boundary_condition(BoundaryCondition(
            id=1, bc_type="specified_head", nodes=[1, 2], values=[100.0, 95.0], layer=1
        ))
        gw.add_boundary_condition(BoundaryCondition(
            id=2, bc_type="specified_flow", nodes=[9, 10], values=[-50.0, -25.0], layer=2
        ))
        gw.add_boundary_condition(BoundaryCondition(
            id=3, bc_type="general_head", nodes=[5], values=[80.0], layer=1, conductance=[0.01]
        ))

        # Add tile drains
        gw.add_tile_drain(TileDrain(id=1, element=2, elevation=30.0, conductance=0.005,
                                    destination_type="stream", destination_id=5))

        # Add subsidence
        gw.add_subsidence(Subsidence(element=1, layer=1, elastic_storage=1e-5,
                                     inelastic_storage=1e-4, preconsolidation_head=90.0))

        return gw

    def test_initialization(self, config: GWFileConfig) -> None:
        """Test writer initialization."""
        writer = GroundwaterWriter(config)
        assert writer.config == config
        assert config.output_dir.exists()

    def test_write_wells(self, config: GWFileConfig, basic_gw: AppGW) -> None:
        """Test writing wells file."""
        writer = GroundwaterWriter(config)
        filepath = writer.write_wells(basic_gw)

        assert filepath.exists()
        content = filepath.read_text()

        assert "NWELLS" in content
        assert "Well 1" in content
        assert "Well 2" in content
        assert "100.0" in content  # x coordinate
        assert "200.0" in content  # y coordinate

    def test_write_wells_with_header(self, config: GWFileConfig, basic_gw: AppGW) -> None:
        """Test writing wells with custom header."""
        writer = GroundwaterWriter(config)
        filepath = writer.write_wells(basic_gw, header="Custom wells header")

        content = filepath.read_text()
        assert "Custom wells header" in content

    def test_write_aquifer_params(self, config: GWFileConfig, basic_gw: AppGW) -> None:
        """Test writing aquifer parameters file."""
        # Add aquifer parameters
        params = AquiferParameters(
            n_nodes=10,
            n_layers=2,
            kh=np.random.rand(10, 2) * 100,
            kv=np.random.rand(10, 2) * 10,
            specific_storage=np.random.rand(10, 2) * 1e-5,
            specific_yield=np.random.rand(10, 2) * 0.3,
        )
        basic_gw.aquifer_params = params

        writer = GroundwaterWriter(config)
        filepath = writer.write_aquifer_params(basic_gw)

        assert filepath.exists()
        content = filepath.read_text()

        assert "NNODES" in content
        assert "NLAYERS" in content

    def test_write_aquifer_params_no_data_raises_error(self, config: GWFileConfig, basic_gw: AppGW) -> None:
        """Test writing aquifer params without data raises error."""
        basic_gw.aquifer_params = None
        writer = GroundwaterWriter(config)

        with pytest.raises(ValueError, match="No aquifer parameters"):
            writer.write_aquifer_params(basic_gw)

    def test_write_boundary_conditions(self, config: GWFileConfig, basic_gw: AppGW) -> None:
        """Test writing boundary conditions file."""
        writer = GroundwaterWriter(config)
        filepath = writer.write_boundary_conditions(basic_gw)

        assert filepath.exists()
        content = filepath.read_text()

        assert "SPECIFIED HEAD" in content
        assert "SPECIFIED FLOW" in content
        assert "GENERAL HEAD" in content
        assert "N_SPEC_HEAD_BC" in content

    def test_write_tile_drains(self, config: GWFileConfig, basic_gw: AppGW) -> None:
        """Test writing tile drains file."""
        writer = GroundwaterWriter(config)
        filepath = writer.write_tile_drains(basic_gw)

        assert filepath.exists()
        content = filepath.read_text()

        assert "NDRAINS" in content
        assert "30.0" in content  # elevation
        assert "stream" in content  # destination type

    def test_write_subsidence(self, config: GWFileConfig, basic_gw: AppGW) -> None:
        """Test writing subsidence file."""
        writer = GroundwaterWriter(config)
        filepath = writer.write_subsidence(basic_gw)

        assert filepath.exists()
        content = filepath.read_text()

        assert "N_SUBSIDENCE" in content
        assert "90.0" in content  # preconsolidation head

    def test_write_initial_heads(self, config: GWFileConfig, basic_gw: AppGW) -> None:
        """Test writing initial heads file."""
        # Add heads
        basic_gw.heads = np.random.rand(10, 2) * 100 + 50

        writer = GroundwaterWriter(config)
        filepath = writer.write_initial_heads(basic_gw)

        assert filepath.exists()
        content = filepath.read_text()

        assert "NNODES" in content
        assert "NLAYERS" in content

    def test_write_initial_heads_no_data_raises_error(self, config: GWFileConfig, basic_gw: AppGW) -> None:
        """Test writing heads without data raises error."""
        writer = GroundwaterWriter(config)

        with pytest.raises(ValueError, match="No initial heads"):
            writer.write_initial_heads(basic_gw)

    def test_write_all(self, config: GWFileConfig, basic_gw: AppGW) -> None:
        """Test writing all files."""
        # Add aquifer params and heads
        params = AquiferParameters(
            n_nodes=10,
            n_layers=2,
            kh=np.random.rand(10, 2) * 100,
            kv=np.random.rand(10, 2) * 10,
            specific_storage=np.random.rand(10, 2) * 1e-5,
            specific_yield=np.random.rand(10, 2) * 0.3,
        )
        basic_gw.aquifer_params = params
        basic_gw.heads = np.random.rand(10, 2) * 100 + 50

        writer = GroundwaterWriter(config)
        files = writer.write(basic_gw)

        assert "wells" in files
        assert "aquifer_params" in files
        assert "boundary_conditions" in files
        assert "tile_drains" in files
        assert "subsidence" in files
        assert "initial_heads" in files

        for path in files.values():
            assert path.exists()

    def test_write_empty_components(self, config: GWFileConfig) -> None:
        """Test writing with empty components."""
        gw = AppGW(n_nodes=10, n_layers=2, n_elements=5)

        writer = GroundwaterWriter(config)
        files = writer.write(gw)

        assert len(files) == 0

    def test_write_pumping_timeseries(self, config: GWFileConfig) -> None:
        """Test writing pumping time series."""
        times = [datetime(2020, 1, 1), datetime(2020, 2, 1), datetime(2020, 3, 1)]
        pumping_rates = {
            1: np.array([100.0, 120.0, 110.0]),
            2: np.array([50.0, 60.0, 55.0]),
        }

        writer = GroundwaterWriter(config)
        filepath = writer.write_pumping_timeseries(
            config.get_pumping_path(),
            times,
            pumping_rates,
            units="TAF"
        )

        assert filepath.exists()
        content = filepath.read_text()

        assert "NDATA" in content
        assert "TAF" in content


# =============================================================================
# Test GroundwaterReader
# =============================================================================


class TestGroundwaterReader:
    """Tests for GroundwaterReader class."""

    def test_read_wells(self, tmp_path: Path) -> None:
        """Test reading wells file."""
        filepath = tmp_path / "wells.dat"
        filepath.write_text("""C  Wells file
2                              / NWELLS
1       100.0000       200.0000     1    50.00    10.00     100.00  Well 1
2       300.0000       400.0000     3    45.00    15.00     150.00  Well 2
""")

        reader = GroundwaterReader()
        wells = reader.read_wells(filepath)

        assert len(wells) == 2
        assert wells[1].x == pytest.approx(100.0)
        assert wells[1].y == pytest.approx(200.0)
        assert wells[1].element == 1
        assert wells[1].name == "Well 1"
        assert wells[2].top_screen == pytest.approx(45.0)

    def test_read_wells_with_comments(self, tmp_path: Path) -> None:
        """Test reading wells with various comments."""
        filepath = tmp_path / "wells.dat"
        filepath.write_text("""C  Comment
c  lowercase
*  asterisk
1                              / NWELLS
C  Data starts here
1       100.0000       200.0000     1    50.00    10.00     100.00  Test Well
""")

        reader = GroundwaterReader()
        wells = reader.read_wells(filepath)

        assert len(wells) == 1
        assert wells[1].name == "Test Well"

    def test_read_wells_invalid_nwells(self, tmp_path: Path) -> None:
        """Test reading wells with invalid NWELLS."""
        filepath = tmp_path / "wells.dat"
        filepath.write_text("""C  Wells
abc                            / NWELLS
""")

        reader = GroundwaterReader()
        with pytest.raises(FileFormatError, match="Invalid NWELLS"):
            reader.read_wells(filepath)

    def test_read_wells_missing_nwells(self, tmp_path: Path) -> None:
        """Test reading wells without NWELLS."""
        filepath = tmp_path / "wells.dat"
        filepath.write_text("""C  Only comments
C  No data
""")

        reader = GroundwaterReader()
        with pytest.raises(FileFormatError, match="Could not find NWELLS"):
            reader.read_wells(filepath)

    def test_read_wells_invalid_data(self, tmp_path: Path) -> None:
        """Test reading wells with invalid data."""
        filepath = tmp_path / "wells.dat"
        filepath.write_text("""C  Wells
1                              / NWELLS
abc  def  ghi  1  50.00  10.00  100.00
""")

        reader = GroundwaterReader()
        with pytest.raises(FileFormatError, match="Invalid well data"):
            reader.read_wells(filepath)

    def test_read_initial_heads(self, tmp_path: Path) -> None:
        """Test reading initial heads file."""
        filepath = tmp_path / "heads.dat"
        filepath.write_text("""C  Initial heads
3                              / NNODES
2                              / NLAYERS
1       100.0000    95.0000
2        98.0000    93.0000
3        96.0000    91.0000
""")

        reader = GroundwaterReader()
        n_nodes, n_layers, heads = reader.read_initial_heads(filepath)

        assert n_nodes == 3
        assert n_layers == 2
        assert heads.shape == (3, 2)
        assert heads[0, 0] == pytest.approx(100.0)
        assert heads[0, 1] == pytest.approx(95.0)
        assert heads[2, 1] == pytest.approx(91.0)

    def test_read_initial_heads_invalid_nnodes(self, tmp_path: Path) -> None:
        """Test reading heads with invalid NNODES."""
        filepath = tmp_path / "heads.dat"
        filepath.write_text("""C  Heads
xyz                            / NNODES
""")

        reader = GroundwaterReader()
        with pytest.raises(FileFormatError, match="Invalid NNODES"):
            reader.read_initial_heads(filepath)

    def test_read_initial_heads_invalid_nlayers(self, tmp_path: Path) -> None:
        """Test reading heads with invalid NLAYERS."""
        filepath = tmp_path / "heads.dat"
        filepath.write_text("""C  Heads
10                             / NNODES
xyz                            / NLAYERS
""")

        reader = GroundwaterReader()
        with pytest.raises(FileFormatError, match="Invalid NLAYERS"):
            reader.read_initial_heads(filepath)


# =============================================================================
# Test Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_write_groundwater_function(self, tmp_path: Path) -> None:
        """Test write_groundwater function."""
        gw = AppGW(n_nodes=5, n_layers=2, n_elements=3)
        gw.add_well(Well(id=1, x=100.0, y=200.0, element=1))

        files = write_groundwater(gw, tmp_path)

        assert "wells" in files
        assert files["wells"].exists()

    def test_write_groundwater_with_config(self, tmp_path: Path) -> None:
        """Test write_groundwater with custom config."""
        config = GWFileConfig(
            output_dir=tmp_path,
            wells_file="custom_wells.dat"
        )
        gw = AppGW(n_nodes=5, n_layers=2, n_elements=3)
        gw.add_well(Well(id=1, x=100.0, y=200.0, element=1))

        files = write_groundwater(gw, tmp_path, config=config)

        assert files["wells"].name == "custom_wells.dat"

    def test_read_wells_function(self, tmp_path: Path) -> None:
        """Test read_wells function."""
        filepath = tmp_path / "wells.dat"
        filepath.write_text("""C  Wells
1                              / NWELLS
1       100.0000       200.0000     1    50.00    10.00     100.00  Test
""")

        wells = read_wells(filepath)

        assert len(wells) == 1
        assert wells[1].name == "Test"

    def test_read_initial_heads_function(self, tmp_path: Path) -> None:
        """Test read_initial_heads function."""
        filepath = tmp_path / "heads.dat"
        filepath.write_text("""C  Heads
2                              / NNODES
1                              / NLAYERS
1       100.0000
2        95.0000
""")

        n_nodes, n_layers, heads = read_initial_heads(filepath)

        assert n_nodes == 2
        assert n_layers == 1
        assert heads.shape == (2, 1)


# =============================================================================
# Test Roundtrip
# =============================================================================


class TestRoundtrip:
    """Tests for read/write roundtrip."""

    def test_roundtrip_wells(self, tmp_path: Path) -> None:
        """Test roundtrip for wells."""
        # Create and write
        gw = AppGW(n_nodes=5, n_layers=2, n_elements=3)
        gw.add_well(Well(id=1, x=100.5, y=200.7, element=1, name="Alfalfa Well",
                        top_screen=50.0, bottom_screen=10.0, max_pump_rate=100.0))
        gw.add_well(Well(id=2, x=300.2, y=400.9, element=2, name="Corn Well",
                        top_screen=45.0, bottom_screen=15.0, max_pump_rate=150.0))

        config = GWFileConfig(output_dir=tmp_path)
        writer = GroundwaterWriter(config)
        filepath = writer.write_wells(gw)

        # Read back
        reader = GroundwaterReader()
        wells = reader.read_wells(filepath)

        assert len(wells) == 2
        assert wells[1].x == pytest.approx(100.5, rel=1e-3)
        assert wells[1].y == pytest.approx(200.7, rel=1e-3)
        assert wells[1].name == "Alfalfa Well"
        assert wells[2].top_screen == pytest.approx(45.0)

    def test_roundtrip_initial_heads(self, tmp_path: Path) -> None:
        """Test roundtrip for initial heads."""
        # Create and write
        gw = AppGW(n_nodes=5, n_layers=3, n_elements=3)
        gw.heads = np.array([
            [100.0, 95.0, 90.0],
            [98.0, 93.0, 88.0],
            [96.0, 91.0, 86.0],
            [94.0, 89.0, 84.0],
            [92.0, 87.0, 82.0],
        ])

        config = GWFileConfig(output_dir=tmp_path)
        writer = GroundwaterWriter(config)
        filepath = writer.write_initial_heads(gw)

        # Read back
        reader = GroundwaterReader()
        n_nodes, n_layers, heads = reader.read_initial_heads(filepath)

        assert n_nodes == 5
        assert n_layers == 3
        np.testing.assert_array_almost_equal(heads, gw.heads, decimal=2)


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_well(self, tmp_path: Path) -> None:
        """Test with single well."""
        gw = AppGW(n_nodes=5, n_layers=1, n_elements=3)
        gw.add_well(Well(id=1, x=100.0, y=200.0, element=1))

        config = GWFileConfig(output_dir=tmp_path)
        writer = GroundwaterWriter(config)
        filepath = writer.write_wells(gw)

        reader = GroundwaterReader()
        wells = reader.read_wells(filepath)

        assert len(wells) == 1

    def test_many_wells(self, tmp_path: Path) -> None:
        """Test with many wells."""
        gw = AppGW(n_nodes=100, n_layers=2, n_elements=50)
        for i in range(100):
            gw.add_well(Well(id=i+1, x=float(i * 100), y=float(i * 50), element=i % 50 + 1))

        config = GWFileConfig(output_dir=tmp_path)
        writer = GroundwaterWriter(config)
        filepath = writer.write_wells(gw)

        reader = GroundwaterReader()
        wells = reader.read_wells(filepath)

        assert len(wells) == 100

    def test_well_name_with_spaces(self, tmp_path: Path) -> None:
        """Test well names with spaces."""
        gw = AppGW(n_nodes=5, n_layers=1, n_elements=3)
        gw.add_well(Well(id=1, x=100.0, y=200.0, element=1, name="My Test Well XYZ"))

        config = GWFileConfig(output_dir=tmp_path)
        writer = GroundwaterWriter(config)
        filepath = writer.write_wells(gw)

        reader = GroundwaterReader()
        wells = reader.read_wells(filepath)

        assert wells[1].name == "My Test Well XYZ"

    def test_negative_coordinates(self, tmp_path: Path) -> None:
        """Test wells with negative coordinates."""
        gw = AppGW(n_nodes=5, n_layers=1, n_elements=3)
        gw.add_well(Well(id=1, x=-1000.5, y=-500.25, element=1))

        config = GWFileConfig(output_dir=tmp_path)
        writer = GroundwaterWriter(config)
        filepath = writer.write_wells(gw)

        reader = GroundwaterReader()
        wells = reader.read_wells(filepath)

        assert wells[1].x == pytest.approx(-1000.5, rel=1e-3)
        assert wells[1].y == pytest.approx(-500.25, rel=1e-3)

    def test_mixed_bc_types(self, tmp_path: Path) -> None:
        """Test writing all BC types."""
        gw = AppGW(n_nodes=20, n_layers=3, n_elements=10)

        gw.add_boundary_condition(BoundaryCondition(
            id=1, bc_type="specified_head", nodes=[1, 2, 3], values=[100.0, 99.0, 98.0], layer=1
        ))
        gw.add_boundary_condition(BoundaryCondition(
            id=2, bc_type="specified_flow", nodes=[18, 19, 20], values=[-10.0, -12.0, -8.0], layer=2
        ))
        gw.add_boundary_condition(BoundaryCondition(
            id=3, bc_type="general_head", nodes=[10, 11], values=[85.0, 84.0], layer=1,
            conductance=[0.01, 0.02]
        ))

        config = GWFileConfig(output_dir=tmp_path)
        writer = GroundwaterWriter(config)
        filepath = writer.write_boundary_conditions(gw)

        content = filepath.read_text()
        assert "N_SPEC_HEAD_BC" in content
        assert "N_SPEC_FLOW_BC" in content
        assert "N_GEN_HEAD_BC" in content

    def test_tile_drain_without_destination(self, tmp_path: Path) -> None:
        """Test tile drain with no destination ID."""
        gw = AppGW(n_nodes=10, n_layers=2, n_elements=5)
        gw.add_tile_drain(TileDrain(id=1, element=2, elevation=30.0, conductance=0.005))

        config = GWFileConfig(output_dir=tmp_path)
        writer = GroundwaterWriter(config)
        filepath = writer.write_tile_drains(gw)

        content = filepath.read_text()
        assert "outside" in content
        assert "0" in content  # default destination_id
