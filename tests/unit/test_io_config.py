"""Unit tests for IO configuration dataclasses.

Tests all config dataclasses in pyiwfm.io.config:
- OutputFormat
- TimeSeriesOutputConfig
- PreProcessorFileConfig
- GWFileConfig
- StreamFileConfig
- LakeFileConfig
- RootZoneFileConfig
- SmallWatershedFileConfig
- UnsatZoneFileConfig
- SimulationFileConfig
- BudgetFileConfig
- ZBudgetFileConfig
- ModelOutputConfig
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyiwfm.io.config import (
    OutputFormat,
    TimeSeriesOutputConfig,
    PreProcessorFileConfig,
    GWFileConfig,
    StreamFileConfig,
    LakeFileConfig,
    RootZoneFileConfig,
    SmallWatershedFileConfig,
    UnsatZoneFileConfig,
    SimulationFileConfig,
    BudgetFileConfig,
    ZBudgetFileConfig,
    ModelOutputConfig,
)


# =============================================================================
# Test OutputFormat
# =============================================================================


class TestOutputFormat:
    """Tests for OutputFormat enum."""

    def test_text_format(self) -> None:
        """Test TEXT format value."""
        assert OutputFormat.TEXT.value == "text"

    def test_dss_format(self) -> None:
        """Test DSS format value."""
        assert OutputFormat.DSS.value == "dss"

    def test_both_format(self) -> None:
        """Test BOTH format value."""
        assert OutputFormat.BOTH.value == "both"

    def test_enum_comparison(self) -> None:
        """Test enum comparison."""
        assert OutputFormat.TEXT == OutputFormat.TEXT
        assert OutputFormat.TEXT != OutputFormat.DSS


# =============================================================================
# Test TimeSeriesOutputConfig
# =============================================================================


class TestTimeSeriesOutputConfig:
    """Tests for TimeSeriesOutputConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = TimeSeriesOutputConfig()

        assert config.format == OutputFormat.TEXT
        assert config.dss_file is None
        assert config.dss_a_part == ""
        assert config.dss_f_part == "PYIWFM"

    def test_custom_values(self) -> None:
        """Test custom values."""
        config = TimeSeriesOutputConfig(
            format=OutputFormat.DSS,
            dss_file="output.dss",
            dss_a_part="PROJECT",
            dss_f_part="VERSION1",
        )

        assert config.format == OutputFormat.DSS
        assert config.dss_file == "output.dss"
        assert config.dss_a_part == "PROJECT"
        assert config.dss_f_part == "VERSION1"

    def test_get_dss_path_with_dss_format(self, tmp_path: Path) -> None:
        """Test get_dss_path with DSS format."""
        config = TimeSeriesOutputConfig(
            format=OutputFormat.DSS,
            dss_file="data.dss",
        )

        path = config.get_dss_path(tmp_path)
        assert path == tmp_path / "data.dss"

    def test_get_dss_path_with_both_format(self, tmp_path: Path) -> None:
        """Test get_dss_path with BOTH format."""
        config = TimeSeriesOutputConfig(
            format=OutputFormat.BOTH,
            dss_file="data.dss",
        )

        path = config.get_dss_path(tmp_path)
        assert path == tmp_path / "data.dss"

    def test_get_dss_path_with_text_format(self, tmp_path: Path) -> None:
        """Test get_dss_path with TEXT format returns None."""
        config = TimeSeriesOutputConfig(
            format=OutputFormat.TEXT,
            dss_file="data.dss",
        )

        path = config.get_dss_path(tmp_path)
        assert path is None

    def test_get_dss_path_without_file(self, tmp_path: Path) -> None:
        """Test get_dss_path without dss_file returns None."""
        config = TimeSeriesOutputConfig(format=OutputFormat.DSS)

        path = config.get_dss_path(tmp_path)
        assert path is None


# =============================================================================
# Test PreProcessorFileConfig
# =============================================================================


class TestPreProcessorFileConfig:
    """Tests for PreProcessorFileConfig dataclass."""

    def test_default_values(self, tmp_path: Path) -> None:
        """Test default values."""
        config = PreProcessorFileConfig(output_dir=tmp_path)

        assert config.output_dir == tmp_path
        assert config.main_file == "Preprocessor.in"
        assert config.node_file == "Nodes.dat"
        assert config.element_file == "Elements.dat"
        assert config.stratigraphy_file == "Stratigraphy.dat"
        assert config.stream_version == "5.0"
        assert config.lake_version == "5.0"

    def test_path_properties(self, tmp_path: Path) -> None:
        """Test path properties."""
        config = PreProcessorFileConfig(output_dir=tmp_path)

        assert config.main_path == tmp_path / "Preprocessor.in"
        assert config.node_path == tmp_path / "Nodes.dat"
        assert config.element_path == tmp_path / "Elements.dat"
        assert config.stratigraphy_path == tmp_path / "Stratigraphy.dat"
        assert config.stream_config_path == tmp_path / "StreamConfig.dat"
        assert config.lake_config_path == tmp_path / "LakeConfig.dat"

    def test_post_init_converts_path(self) -> None:
        """Test that __post_init__ converts string to Path."""
        config = PreProcessorFileConfig(output_dir="/some/path")  # type: ignore

        assert isinstance(config.output_dir, Path)
        assert config.output_dir == Path("/some/path")


# =============================================================================
# Test GWFileConfig
# =============================================================================


class TestGWFileConfig:
    """Tests for GWFileConfig dataclass."""

    def test_default_values(self, tmp_path: Path) -> None:
        """Test default values."""
        config = GWFileConfig(output_dir=tmp_path)

        assert config.output_dir == tmp_path
        assert config.main_file == "Groundwater.dat"
        assert config.aquifer_params_file == "AquiferParameters.dat"
        assert config.bc_main_file == "BoundaryConditions.dat"
        assert config.pumping_main_file == "Pumping.dat"
        assert config.tile_drain_file == "TileDrains.dat"

    def test_main_path_property(self, tmp_path: Path) -> None:
        """Test main_path property."""
        config = GWFileConfig(output_dir=tmp_path)

        assert config.main_path == tmp_path / "Groundwater.dat"

    def test_get_path_method(self, tmp_path: Path) -> None:
        """Test get_path method."""
        config = GWFileConfig(output_dir=tmp_path)

        assert config.get_path("test.dat") == tmp_path / "test.dat"

    def test_ts_config_default(self, tmp_path: Path) -> None:
        """Test default time series config."""
        config = GWFileConfig(output_dir=tmp_path)

        assert config.ts_config is not None
        assert config.ts_config.dss_file == "Groundwater.dss"


# =============================================================================
# Test StreamFileConfig
# =============================================================================


class TestStreamFileConfig:
    """Tests for StreamFileConfig dataclass."""

    def test_default_values(self, tmp_path: Path) -> None:
        """Test default values."""
        config = StreamFileConfig(output_dir=tmp_path)

        assert config.output_dir == tmp_path
        assert config.version == "5.0"
        assert config.main_file == "Streams.dat"
        assert config.inflow_file == "StreamInflows.dat"
        assert config.diversion_spec_file == "DiversionSpecs.dat"
        assert config.bypass_spec_file == "BypassSpecs.dat"

    def test_main_path_property(self, tmp_path: Path) -> None:
        """Test main_path property."""
        config = StreamFileConfig(output_dir=tmp_path)

        assert config.main_path == tmp_path / "Streams.dat"


# =============================================================================
# Test LakeFileConfig
# =============================================================================


class TestLakeFileConfig:
    """Tests for LakeFileConfig dataclass."""

    def test_default_values(self, tmp_path: Path) -> None:
        """Test default values."""
        config = LakeFileConfig(output_dir=tmp_path)

        assert config.output_dir == tmp_path
        assert config.version == "5.0"
        assert config.main_file == "Lakes.dat"
        assert config.max_elevation_file == "MaxLakeElevations.dat"
        assert config.lake_elements_file == "LakeElements.dat"

    def test_main_path_property(self, tmp_path: Path) -> None:
        """Test main_path property."""
        config = LakeFileConfig(output_dir=tmp_path)

        assert config.main_path == tmp_path / "Lakes.dat"


# =============================================================================
# Test RootZoneFileConfig
# =============================================================================


class TestRootZoneFileConfig:
    """Tests for RootZoneFileConfig dataclass."""

    def test_default_values(self, tmp_path: Path) -> None:
        """Test default values."""
        config = RootZoneFileConfig(output_dir=tmp_path)

        assert config.output_dir == tmp_path
        assert config.version == "5.0"
        assert config.main_file == "RootZone.dat"
        assert config.ag_water_supply_file == "AgWaterSupply.dat"
        assert config.nonponded_main_file == "NonPondedCrops.dat"
        assert config.ponded_main_file == "PondedCrops.dat"
        assert config.native_main_file == "NativeRiparian.dat"
        assert config.urban_main_file == "Urban.dat"

    def test_main_path_property(self, tmp_path: Path) -> None:
        """Test main_path property."""
        config = RootZoneFileConfig(output_dir=tmp_path)

        assert config.main_path == tmp_path / "RootZone.dat"


# =============================================================================
# Test SmallWatershedFileConfig
# =============================================================================


class TestSmallWatershedFileConfig:
    """Tests for SmallWatershedFileConfig dataclass."""

    def test_default_values(self, tmp_path: Path) -> None:
        """Test default values."""
        config = SmallWatershedFileConfig(output_dir=tmp_path)

        assert config.output_dir == tmp_path
        assert config.version == "4.1"
        assert config.main_file == "SmallWatersheds.dat"

    def test_main_path_property(self, tmp_path: Path) -> None:
        """Test main_path property."""
        config = SmallWatershedFileConfig(output_dir=tmp_path)

        assert config.main_path == tmp_path / "SmallWatersheds.dat"


# =============================================================================
# Test UnsatZoneFileConfig
# =============================================================================


class TestUnsatZoneFileConfig:
    """Tests for UnsatZoneFileConfig dataclass."""

    def test_default_values(self, tmp_path: Path) -> None:
        """Test default values."""
        config = UnsatZoneFileConfig(output_dir=tmp_path)

        assert config.output_dir == tmp_path
        assert config.main_file == "UnsatZone.dat"

    def test_main_path_property(self, tmp_path: Path) -> None:
        """Test main_path property."""
        config = UnsatZoneFileConfig(output_dir=tmp_path)

        assert config.main_path == tmp_path / "UnsatZone.dat"


# =============================================================================
# Test SimulationFileConfig
# =============================================================================


class TestSimulationFileConfig:
    """Tests for SimulationFileConfig dataclass."""

    def test_default_values(self, tmp_path: Path) -> None:
        """Test default values."""
        config = SimulationFileConfig(output_dir=tmp_path)

        assert config.output_dir == tmp_path
        assert config.main_file == "Simulation.in"
        assert config.precip_file == "Precipitation.dat"
        assert config.et_file == "Evapotranspiration.dat"
        assert config.ts_format == OutputFormat.TEXT
        assert config.gw_version == "4.0"
        assert config.stream_version == "5.0"

    def test_main_path_property(self, tmp_path: Path) -> None:
        """Test main_path property."""
        config = SimulationFileConfig(output_dir=tmp_path)

        assert config.main_path == tmp_path / "Simulation.in"

    def test_groundwater_property_creates_config(self, tmp_path: Path) -> None:
        """Test groundwater property creates config on demand."""
        config = SimulationFileConfig(output_dir=tmp_path)

        gw = config.groundwater
        assert gw is not None
        assert gw.output_dir == tmp_path / "Groundwater"

        # Second access returns same instance
        gw2 = config.groundwater
        assert gw2 is gw

    def test_streams_property_creates_config(self, tmp_path: Path) -> None:
        """Test streams property creates config on demand."""
        config = SimulationFileConfig(output_dir=tmp_path)

        streams = config.streams
        assert streams is not None
        assert streams.output_dir == tmp_path / "Streams"
        assert streams.version == config.stream_version

    def test_lakes_property_creates_config(self, tmp_path: Path) -> None:
        """Test lakes property creates config on demand."""
        config = SimulationFileConfig(output_dir=tmp_path)

        lakes = config.lakes
        assert lakes is not None
        assert lakes.output_dir == tmp_path / "Lakes"
        assert lakes.version == config.lake_version

    def test_rootzone_property_creates_config(self, tmp_path: Path) -> None:
        """Test rootzone property creates config on demand."""
        config = SimulationFileConfig(output_dir=tmp_path)

        rz = config.rootzone
        assert rz is not None
        assert rz.output_dir == tmp_path / "RootZone"
        assert rz.version == config.rootzone_version

    def test_small_watersheds_property_creates_config(self, tmp_path: Path) -> None:
        """Test small_watersheds property creates config on demand."""
        config = SimulationFileConfig(output_dir=tmp_path)

        sw = config.small_watersheds
        assert sw is not None
        assert sw.output_dir == tmp_path / "SmallWatersheds"

    def test_unsatzone_property_creates_config(self, tmp_path: Path) -> None:
        """Test unsatzone property creates config on demand."""
        config = SimulationFileConfig(output_dir=tmp_path)

        uz = config.unsatzone
        assert uz is not None
        assert uz.output_dir == tmp_path / "UnsatZone"

    def test_ts_format_propagates(self, tmp_path: Path) -> None:
        """Test that ts_format propagates to component configs."""
        config = SimulationFileConfig(
            output_dir=tmp_path,
            ts_format=OutputFormat.DSS,
        )

        # Access components to create their configs
        gw = config.groundwater
        streams = config.streams
        lakes = config.lakes
        rz = config.rootzone

        assert gw.ts_config.format == OutputFormat.DSS
        assert streams.ts_config.format == OutputFormat.DSS
        assert lakes.ts_config.format == OutputFormat.DSS
        assert rz.ts_config.format == OutputFormat.DSS


# =============================================================================
# Test BudgetFileConfig
# =============================================================================


class TestBudgetFileConfig:
    """Tests for BudgetFileConfig dataclass."""

    def test_default_values(self, tmp_path: Path) -> None:
        """Test default values."""
        config = BudgetFileConfig(output_dir=tmp_path)

        assert config.output_dir == tmp_path
        assert config.main_file == "Budget.in"

    def test_main_path_property(self, tmp_path: Path) -> None:
        """Test main_path property."""
        config = BudgetFileConfig(output_dir=tmp_path)

        assert config.main_path == tmp_path / "Budget.in"


# =============================================================================
# Test ZBudgetFileConfig
# =============================================================================


class TestZBudgetFileConfig:
    """Tests for ZBudgetFileConfig dataclass."""

    def test_default_values(self, tmp_path: Path) -> None:
        """Test default values."""
        config = ZBudgetFileConfig(output_dir=tmp_path)

        assert config.output_dir == tmp_path
        assert config.main_file == "ZBudget.in"
        assert config.zone_definition_file == "ZoneDefinitions.dat"

    def test_main_path_property(self, tmp_path: Path) -> None:
        """Test main_path property."""
        config = ZBudgetFileConfig(output_dir=tmp_path)

        assert config.main_path == tmp_path / "ZBudget.in"

    def test_zone_definition_path_property(self, tmp_path: Path) -> None:
        """Test zone_definition_path property."""
        config = ZBudgetFileConfig(output_dir=tmp_path)

        assert config.zone_definition_path == tmp_path / "ZoneDefinitions.dat"


# =============================================================================
# Test ModelOutputConfig
# =============================================================================


class TestModelOutputConfig:
    """Tests for ModelOutputConfig dataclass."""

    def test_default_values(self, tmp_path: Path) -> None:
        """Test default values."""
        config = ModelOutputConfig(output_dir=tmp_path)

        assert config.output_dir == tmp_path
        assert config.model_name == "IWFM_Model"
        assert config.ts_format == OutputFormat.TEXT
        assert config.preprocessor_subdir == "Preprocessor"
        assert config.simulation_subdir == "Simulation"

    def test_preprocessor_property(self, tmp_path: Path) -> None:
        """Test preprocessor property creates config on demand."""
        config = ModelOutputConfig(output_dir=tmp_path)

        pp = config.preprocessor
        assert pp is not None
        assert pp.output_dir == tmp_path / "Preprocessor"

        # Second access returns same instance
        pp2 = config.preprocessor
        assert pp2 is pp

    def test_simulation_property(self, tmp_path: Path) -> None:
        """Test simulation property creates config on demand."""
        config = ModelOutputConfig(output_dir=tmp_path)

        sim = config.simulation
        assert sim is not None
        assert sim.output_dir == tmp_path / "Simulation"

    def test_version_propagation(self, tmp_path: Path) -> None:
        """Test that versions propagate to nested configs."""
        config = ModelOutputConfig(
            output_dir=tmp_path,
            stream_version="4.0",
            lake_version="4.0",
            rootzone_version="4.0",
        )

        pp = config.preprocessor
        sim = config.simulation

        assert pp.stream_version == "4.0"
        assert pp.lake_version == "4.0"
        assert sim.stream_version == "4.0"
        assert sim.lake_version == "4.0"
        assert sim.rootzone_version == "4.0"

    def test_ts_format_propagation(self, tmp_path: Path) -> None:
        """Test that ts_format propagates to simulation config."""
        config = ModelOutputConfig(
            output_dir=tmp_path,
            ts_format=OutputFormat.DSS,
        )

        sim = config.simulation
        assert sim.ts_format == OutputFormat.DSS

    def test_ensure_directories(self, tmp_path: Path) -> None:
        """Test ensure_directories creates directory structure."""
        config = ModelOutputConfig(output_dir=tmp_path / "new_model")

        # Directories should not exist yet
        assert not (tmp_path / "new_model").exists()

        config.ensure_directories()

        # Now they should exist
        assert (tmp_path / "new_model").exists()
        assert (tmp_path / "new_model" / "Preprocessor").exists()
        assert (tmp_path / "new_model" / "Simulation").exists()
