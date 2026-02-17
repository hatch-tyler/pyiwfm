"""
File configuration classes for IWFM model I/O.

These dataclasses define the file structure and naming conventions
for IWFM model input/output, supporting both text and DSS formats.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar


class OutputFormat(Enum):
    """Output format for time series data."""

    TEXT = "text"  # Standard ASCII text files
    DSS = "dss"  # HEC-DSS binary files
    BOTH = "both"  # Both text and DSS


@dataclass
class TimeSeriesOutputConfig:
    """Configuration for time series output format."""

    format: OutputFormat = OutputFormat.TEXT
    dss_file: str | None = None  # DSS file name when format is DSS or BOTH
    dss_a_part: str = ""  # Project/Basin
    dss_f_part: str = "PYIWFM"  # Version

    def get_dss_path(self, output_dir: Path) -> Path | None:
        """Get full path to DSS file."""
        if self.format in (OutputFormat.DSS, OutputFormat.BOTH) and self.dss_file:
            return output_dir / self.dss_file
        return None


@dataclass
class PreProcessorFileConfig:
    """
    Configuration for PreProcessor input/output files.

    Defines file names for all preprocessor components.
    """

    output_dir: Path
    main_file: str = "Preprocessor.in"
    node_file: str = "Nodes.dat"
    element_file: str = "Elements.dat"
    stratigraphy_file: str = "Stratigraphy.dat"
    stream_config_file: str = "StreamConfig.dat"
    lake_config_file: str = "LakeConfig.dat"
    binary_output_file: str = "Preprocessor.bin"

    # Version for stream/lake config
    stream_version: str = "5.0"
    lake_version: str = "5.0"

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)

    @property
    def main_path(self) -> Path:
        return self.output_dir / self.main_file

    @property
    def node_path(self) -> Path:
        return self.output_dir / self.node_file

    @property
    def element_path(self) -> Path:
        return self.output_dir / self.element_file

    @property
    def stratigraphy_path(self) -> Path:
        return self.output_dir / self.stratigraphy_file

    @property
    def stream_config_path(self) -> Path:
        return self.output_dir / self.stream_config_file

    @property
    def lake_config_path(self) -> Path:
        return self.output_dir / self.lake_config_file


@dataclass
class GWFileConfig:
    """
    Configuration for groundwater component files.

    Supports both text and DSS output formats for time series.
    """

    output_dir: Path
    main_file: str = "Groundwater.dat"
    aquifer_params_file: str = "AquiferParameters.dat"

    # Boundary conditions
    bc_main_file: str = "BoundaryConditions.dat"
    specified_head_bc_file: str = "SpecifiedHeadBC.dat"
    specified_flow_bc_file: str = "SpecifiedFlowBC.dat"
    general_head_bc_file: str = "GeneralHeadBC.dat"
    constrained_ghbc_file: str = "ConstrainedGHBC.dat"

    # Pumping
    pumping_main_file: str = "Pumping.dat"
    well_spec_file: str = "WellSpecs.dat"
    element_pumping_file: str = "ElementPumping.dat"
    pumping_rates_file: str = "PumpingRates.dat"

    # Tile drains
    tile_drain_file: str = "TileDrains.dat"

    # Subsidence
    subsidence_file: str = "Subsidence.dat"

    # Time series output format
    ts_config: TimeSeriesOutputConfig = field(
        default_factory=lambda: TimeSeriesOutputConfig(dss_file="Groundwater.dss")
    )

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)

    def get_path(self, filename: str) -> Path:
        return self.output_dir / filename

    @property
    def main_path(self) -> Path:
        return self.output_dir / self.main_file


@dataclass
class StreamFileConfig:
    """Configuration for stream component files."""

    output_dir: Path
    version: str = "5.0"

    main_file: str = "Streams.dat"
    inflow_file: str = "StreamInflows.dat"
    diversion_spec_file: str = "DiversionSpecs.dat"
    diversion_data_file: str = "DiversionData.dat"
    bypass_spec_file: str = "BypassSpecs.dat"
    bypass_data_file: str = "BypassData.dat"
    surface_area_file: str = "StreamSurfaceArea.dat"

    # Time series output format
    ts_config: TimeSeriesOutputConfig = field(
        default_factory=lambda: TimeSeriesOutputConfig(dss_file="Streams.dss")
    )

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)

    @property
    def main_path(self) -> Path:
        return self.output_dir / self.main_file


@dataclass
class LakeFileConfig:
    """Configuration for lake component files."""

    output_dir: Path
    version: str = "5.0"

    main_file: str = "Lakes.dat"
    max_elevation_file: str = "MaxLakeElevations.dat"
    lake_elements_file: str = "LakeElements.dat"
    lake_outflow_file: str = "LakeOutflows.dat"

    # Time series output format
    ts_config: TimeSeriesOutputConfig = field(
        default_factory=lambda: TimeSeriesOutputConfig(dss_file="Lakes.dss")
    )

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)

    @property
    def main_path(self) -> Path:
        return self.output_dir / self.main_file


@dataclass
class RootZoneFileConfig:
    """Configuration for root zone component files."""

    output_dir: Path
    version: str = "5.0"

    main_file: str = "RootZone.dat"
    ag_water_supply_file: str = "AgWaterSupply.dat"
    moisture_source_file: str = "MoistureSource.dat"
    irrigation_period_file: str = "IrrigationPeriod.dat"
    return_flow_file: str = "ReturnFlow.dat"
    reuse_fraction_file: str = "ReuseFraction.dat"

    # Non-ponded crops
    nonponded_main_file: str = "NonPondedCrops.dat"
    nonponded_area_file: str = "NonPondedArea.dat"
    irrigation_target_file: str = "IrrigationTarget.dat"
    min_soil_moisture_file: str = "MinSoilMoisture.dat"
    min_perc_factor_file: str = "MinPercFactor.dat"
    root_depth_file: str = "RootDepth.dat"

    # Ponded crops
    ponded_main_file: str = "PondedCrops.dat"
    ponded_area_file: str = "PondedArea.dat"
    pond_depth_file: str = "PondDepth.dat"
    pond_operation_file: str = "PondOperation.dat"

    # Native/riparian
    native_main_file: str = "NativeRiparian.dat"
    native_area_file: str = "NativeArea.dat"

    # Urban
    urban_main_file: str = "Urban.dat"
    urban_area_file: str = "UrbanArea.dat"
    urban_water_use_file: str = "UrbanWaterUse.dat"
    urban_population_file: str = "UrbanPopulation.dat"

    # Time series output format
    ts_config: TimeSeriesOutputConfig = field(
        default_factory=lambda: TimeSeriesOutputConfig(dss_file="RootZone.dss")
    )

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)

    @property
    def main_path(self) -> Path:
        return self.output_dir / self.main_file


@dataclass
class SmallWatershedFileConfig:
    """Configuration for small watershed component files."""

    output_dir: Path
    version: str = "4.1"
    main_file: str = "SmallWatersheds.dat"

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)

    @property
    def main_path(self) -> Path:
        return self.output_dir / self.main_file


@dataclass
class UnsatZoneFileConfig:
    """Configuration for unsaturated zone component files."""

    output_dir: Path
    main_file: str = "UnsatZone.dat"

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)

    @property
    def main_path(self) -> Path:
        return self.output_dir / self.main_file


@dataclass
class SimulationFileConfig:
    """
    Configuration for complete simulation file set.

    Orchestrates all component configurations.
    """

    output_dir: Path
    main_file: str = "Simulation.in"

    # Climate files
    precip_file: str = "Precipitation.dat"
    et_file: str = "Evapotranspiration.dat"
    crop_coef_file: str = "CropCoefficients.dat"

    # Supply adjustment
    supply_adjust_file: str = "SupplyAdjustment.dat"

    # Component configs (created lazily)
    _groundwater: GWFileConfig | None = None
    _streams: StreamFileConfig | None = None
    _lakes: LakeFileConfig | None = None
    _rootzone: RootZoneFileConfig | None = None
    _small_watersheds: SmallWatershedFileConfig | None = None
    _unsatzone: UnsatZoneFileConfig | None = None

    # Global time series format
    ts_format: OutputFormat = OutputFormat.TEXT
    dss_file: str = "Simulation.dss"

    # Component versions
    gw_version: str = "4.0"
    stream_version: str = "5.0"
    lake_version: str = "5.0"
    rootzone_version: str = "5.0"
    small_watershed_version: str = "4.1"

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)

    @property
    def main_path(self) -> Path:
        return self.output_dir / self.main_file

    @property
    def groundwater(self) -> GWFileConfig:
        if self._groundwater is None:
            gw_dir = self.output_dir / "Groundwater"
            ts_config = TimeSeriesOutputConfig(
                format=self.ts_format,
                dss_file="Groundwater.dss" if self.ts_format != OutputFormat.TEXT else None,
            )
            self._groundwater = GWFileConfig(output_dir=gw_dir, ts_config=ts_config)
        return self._groundwater

    @property
    def streams(self) -> StreamFileConfig:
        if self._streams is None:
            stream_dir = self.output_dir / "Streams"
            ts_config = TimeSeriesOutputConfig(
                format=self.ts_format,
                dss_file="Streams.dss" if self.ts_format != OutputFormat.TEXT else None,
            )
            self._streams = StreamFileConfig(
                output_dir=stream_dir, version=self.stream_version, ts_config=ts_config
            )
        return self._streams

    @property
    def lakes(self) -> LakeFileConfig:
        if self._lakes is None:
            lake_dir = self.output_dir / "Lakes"
            ts_config = TimeSeriesOutputConfig(
                format=self.ts_format,
                dss_file="Lakes.dss" if self.ts_format != OutputFormat.TEXT else None,
            )
            self._lakes = LakeFileConfig(
                output_dir=lake_dir, version=self.lake_version, ts_config=ts_config
            )
        return self._lakes

    @property
    def rootzone(self) -> RootZoneFileConfig:
        if self._rootzone is None:
            rz_dir = self.output_dir / "RootZone"
            ts_config = TimeSeriesOutputConfig(
                format=self.ts_format,
                dss_file="RootZone.dss" if self.ts_format != OutputFormat.TEXT else None,
            )
            self._rootzone = RootZoneFileConfig(
                output_dir=rz_dir, version=self.rootzone_version, ts_config=ts_config
            )
        return self._rootzone

    @property
    def small_watersheds(self) -> SmallWatershedFileConfig:
        if self._small_watersheds is None:
            sw_dir = self.output_dir / "SmallWatersheds"
            self._small_watersheds = SmallWatershedFileConfig(
                output_dir=sw_dir, version=self.small_watershed_version
            )
        return self._small_watersheds

    @property
    def unsatzone(self) -> UnsatZoneFileConfig:
        if self._unsatzone is None:
            uz_dir = self.output_dir / "UnsatZone"
            self._unsatzone = UnsatZoneFileConfig(output_dir=uz_dir)
        return self._unsatzone


@dataclass
class BudgetFileConfig:
    """Configuration for budget post-processor files."""

    output_dir: Path
    main_file: str = "Budget.in"

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)

    @property
    def main_path(self) -> Path:
        return self.output_dir / self.main_file


@dataclass
class ZBudgetFileConfig:
    """Configuration for zone budget post-processor files."""

    output_dir: Path
    main_file: str = "ZBudget.in"
    zone_definition_file: str = "ZoneDefinitions.dat"

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)

    @property
    def main_path(self) -> Path:
        return self.output_dir / self.main_file

    @property
    def zone_definition_path(self) -> Path:
        return self.output_dir / self.zone_definition_file


@dataclass
class ModelOutputConfig:
    """
    Complete model output configuration.

    Coordinates preprocessor and simulation file configs.
    """

    output_dir: Path
    model_name: str = "IWFM_Model"

    # Output format for time series
    ts_format: OutputFormat = OutputFormat.TEXT

    # Component versions
    stream_version: str = "5.0"
    lake_version: str = "5.0"
    rootzone_version: str = "5.0"

    # Subdirectory names
    preprocessor_subdir: str = "Preprocessor"
    simulation_subdir: str = "Simulation"

    _preprocessor: PreProcessorFileConfig | None = None
    _simulation: SimulationFileConfig | None = None

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)

    @property
    def preprocessor(self) -> PreProcessorFileConfig:
        if self._preprocessor is None:
            pp_dir = self.output_dir / self.preprocessor_subdir
            self._preprocessor = PreProcessorFileConfig(
                output_dir=pp_dir,
                stream_version=self.stream_version,
                lake_version=self.lake_version,
            )
        return self._preprocessor

    @property
    def simulation(self) -> SimulationFileConfig:
        if self._simulation is None:
            sim_dir = self.output_dir / self.simulation_subdir
            self._simulation = SimulationFileConfig(
                output_dir=sim_dir,
                ts_format=self.ts_format,
                stream_version=self.stream_version,
                lake_version=self.lake_version,
                rootzone_version=self.rootzone_version,
            )
        return self._simulation

    def ensure_directories(self) -> None:
        """Create all output directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessor.output_dir.mkdir(parents=True, exist_ok=True)
        self.simulation.output_dir.mkdir(parents=True, exist_ok=True)

        # Component directories
        if self.simulation._groundwater:
            self.simulation.groundwater.output_dir.mkdir(parents=True, exist_ok=True)
        if self.simulation._streams:
            self.simulation.streams.output_dir.mkdir(parents=True, exist_ok=True)
        if self.simulation._lakes:
            self.simulation.lakes.output_dir.mkdir(parents=True, exist_ok=True)
        if self.simulation._rootzone:
            self.simulation.rootzone.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelWriteConfig:
    """Configuration for writing a complete IWFM model with per-file path control.

    Every output file has a standardized key and a default relative path
    (producing a nested layout). Users can override any path via the
    ``file_paths`` dict. The writer uses ``get_path()`` for absolute paths
    and ``get_relative_path()`` for IWFM-compatible relative references
    between config files.

    Examples:
        Default nested layout::

            config = ModelWriteConfig(output_dir=Path("C:/models/my_model"))

        Flat layout (all files in one directory)::

            config = ModelWriteConfig.flat(output_dir=Path("C:/models/flat"))

        Custom layout::

            config = ModelWriteConfig(output_dir=Path("C:/models/custom"))
            config.set_file("gw_main", "groundwater/main.dat")
            config.set_file("stream_main", "surface/streams.dat")
    """

    output_dir: Path
    ts_format: OutputFormat = OutputFormat.TEXT
    dss_file: str = "model.dss"
    dss_a_part: str = ""  # DSS A-part (project/basin name); defaults to model_name
    dss_f_part: str = "PYIWFM"  # DSS F-part (version/scenario)
    copy_source_ts: bool = True
    model_name: str = "IWFM_Model"

    # Component versions (populated from model.metadata)
    gw_version: str = "4.0"
    stream_version: str = "4.0"
    lake_version: str = "4.0"
    rootzone_version: str = "4.12"

    # Per-file path overrides (relative to output_dir).
    # Any key not present here uses the default from DEFAULT_PATHS.
    file_paths: dict[str, str] = field(default_factory=dict)

    # ---- Default paths (nested layout) ----
    DEFAULT_PATHS: ClassVar[dict[str, str]] = {
        # Preprocessor
        "preprocessor_main": "Preprocessor/Preprocessor.in",
        "nodes": "Preprocessor/Nodes.dat",
        "elements": "Preprocessor/Elements.dat",
        "stratigraphy": "Preprocessor/Stratigraphy.dat",
        "stream_config": "Preprocessor/StreamConfig.dat",
        "lake_config": "Preprocessor/LakeConfig.dat",
        # Simulation-level
        "simulation_main": "Simulation/Simulation_MAIN.IN",
        "preprocessor_bin": "Simulation/PreProcessor.bin",
        "precipitation": "Simulation/Precip.dat",
        "et": "Simulation/ET.dat",
        "irig_frac": "Simulation/IrigFrac.dat",
        "supply_adjust": "Simulation/SupplyAdjust.dat",
        "dss_ts_file": "Simulation/climate_data.dss",
        # GW component
        "gw_main": "Simulation/GW/GW_MAIN.dat",
        "gw_bc_main": "Simulation/GW/BC_MAIN.dat",
        "gw_pump_main": "Simulation/GW/Pump_MAIN.dat",
        "gw_elem_pump": "Simulation/GW/ElemPump.dat",
        "gw_well_spec": "Simulation/GW/WellSpec.dat",
        "gw_ts_pumping": "Simulation/GW/TSPumping.dat",
        "gw_spec_head_bc": "Simulation/GW/SpecHeadBC.dat",
        "gw_spec_flow_bc": "Simulation/GW/SpecFlowBC.dat",
        "gw_bound_tsd": "Simulation/GW/BoundTSD.dat",
        "gw_tile_drain": "Simulation/GW/TileDrain.dat",
        "gw_subsidence": "Simulation/GW/Subsidence.dat",
        # Stream component
        "stream_main": "Simulation/Stream/Stream_MAIN.dat",
        "stream_inflow": "Simulation/Stream/StreamInflow.dat",
        "stream_diver_specs": "Simulation/Stream/DiverSpecs.dat",
        "stream_bypass_specs": "Simulation/Stream/BypassSpecs.dat",
        "stream_diversions": "Simulation/Stream/Diversions.dat",
        # Lake component
        "lake_main": "Simulation/Lake/Lake_MAIN.dat",
        "lake_max_elev": "Simulation/Lake/MaxLakeElev.dat",
        # Small watershed component (copied, not regenerated)
        "swshed_main": "Simulation/SmallWatersheds/SWatersheds.dat",
        # Unsaturated zone component (copied, not regenerated)
        "unsatzone_main": "Simulation/UnsatZone.dat",
        # Root zone component
        "rootzone_main": "Simulation/RootZone/RootZone_MAIN.dat",
        "rootzone_return_flow": "Simulation/RootZone/ReturnFlowFrac.dat",
        "rootzone_reuse": "Simulation/RootZone/ReuseFrac.dat",
        "rootzone_irig_period": "Simulation/RootZone/IrigPeriod.dat",
        "rootzone_surface_flow_dest": "Simulation/RootZone/SurfaceFlowDest.dat",
        "rootzone_nonponded": "Simulation/RootZone/NonPondedCrops.dat",
        "rootzone_ponded": "Simulation/RootZone/PondedCrops.dat",
        "rootzone_urban": "Simulation/RootZone/Urban.dat",
        "rootzone_native": "Simulation/RootZone/NativeRiparian.dat",
        # Results (output files)
        "results_gw_budget": "Results/GW.hdf",
        "results_gw_zbudget": "Results/GW_ZBud.hdf",
        "results_gw_head": "Results/GWHeadAll.out",
        "results_strm_budget": "Results/StrmBud.hdf",
        "results_lake_budget": "Results/LakeBud.hdf",
        "results_lwu_budget": "Results/LWU.hdf",
        "results_rz_budget": "Results/RootZone.hdf",
    }

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)

    def get_path(self, file_key: str) -> Path:
        """Return absolute path for a file key.

        Args:
            file_key: Standardized key from DEFAULT_PATHS.

        Returns:
            Absolute path to the file.

        Raises:
            KeyError: If file_key is not in file_paths or DEFAULT_PATHS.
        """
        rel = self.file_paths.get(file_key)
        if rel is None:
            rel = self.DEFAULT_PATHS[file_key]
        return self.output_dir / rel

    def get_relative_path(self, from_key: str, to_key: str) -> str:
        """Compute relative path from one file's directory to another file.

        This produces the path string that IWFM writes in its config files.
        For example, Simulation_MAIN.IN references GW_MAIN.dat as
        ``GW\\GW_MAIN.dat``.

        Uses ``os.path.relpath`` to handle arbitrary directory structures
        including paths that require ``..`` traversal.

        Args:
            from_key: File key of the referencing file.
            to_key: File key of the referenced file.

        Returns:
            Relative path string with OS-native separators.
        """
        from_dir = self.get_path(from_key).parent
        to_path = self.get_path(to_key)
        rel = os.path.relpath(to_path, from_dir)
        # IWFM Fortran reader uses OS-native separators
        return str(Path(rel))

    def set_file(self, file_key: str, relative_path: str) -> None:
        """Set a custom path for a file key (relative to output_dir).

        Args:
            file_key: Standardized key (must be in DEFAULT_PATHS).
            relative_path: Path relative to output_dir.
        """
        self.file_paths[file_key] = relative_path

    @classmethod
    def nested(cls, output_dir: Path | str, **kwargs: Any) -> ModelWriteConfig:
        """Create config with standard nested layout (default)."""
        return cls(output_dir=Path(output_dir), **kwargs)

    @classmethod
    def flat(cls, output_dir: Path | str, **kwargs: Any) -> ModelWriteConfig:
        """Create config with all files in one directory."""
        flat_paths = {k: Path(v).name for k, v in cls.DEFAULT_PATHS.items()}
        return cls(output_dir=Path(output_dir), file_paths=flat_paths, **kwargs)
