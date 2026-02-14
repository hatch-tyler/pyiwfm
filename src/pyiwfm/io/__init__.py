"""I/O handlers for IWFM file formats."""

from __future__ import annotations

# Base classes and configuration
from pyiwfm.io.base import (
    FileInfo,
    BaseReader,
    BaseWriter,
    ModelReader,
    ModelWriter,
    BinaryReader,
    BinaryWriter,
)

from pyiwfm.io.config import (
    OutputFormat,
    TimeSeriesOutputConfig,
    PreProcessorFileConfig,
    GWFileConfig as GWFileConfigNew,
    StreamFileConfig as StreamFileConfigNew,
    LakeFileConfig as LakeFileConfigNew,
    RootZoneFileConfig as RootZoneFileConfigNew,
    SmallWatershedFileConfig,
    UnsatZoneFileConfig,
    SimulationFileConfig as SimulationFileConfigNew,
    BudgetFileConfig,
    ZBudgetFileConfig,
    ModelOutputConfig,
    ModelWriteConfig,
)

from pyiwfm.io.writer_base import (
    TemplateWriter,
    TimeSeriesSpec,
    TimeSeriesWriter as TimeSeriesWriterNew,
    ComponentWriter,
    IWFMModelWriter,
)

# PreProcessor Writer
from pyiwfm.io.preprocessor_writer import (
    PreProcessorWriter,
    write_preprocessor_files,
    write_nodes_file,
    write_elements_file,
    write_stratigraphy_file,
)

# ASCII I/O
from pyiwfm.io.ascii import (
    read_nodes,
    read_elements,
    read_stratigraphy,
    write_nodes,
    write_elements,
    write_stratigraphy,
)

# Binary I/O
from pyiwfm.io.binary import (
    FortranBinaryReader,
    FortranBinaryWriter,
    read_binary_mesh,
    write_binary_mesh,
    read_binary_stratigraphy,
    write_binary_stratigraphy,
)

# Preprocessor Binary I/O
from pyiwfm.io.preprocessor_binary import (
    PreprocessorBinaryData,
    PreprocessorBinaryReader,
    AppNodeData,
    AppElementData,
    AppFaceData,
    SubregionData,
    StratigraphyData,
    StreamGWConnectorData,
    LakeGWConnectorData,
    StreamLakeConnectorData,
    StreamData as PreprocessorStreamData,
    LakeData as PreprocessorLakeData,
    read_preprocessor_binary,
)

# PreProcessor I/O
from pyiwfm.io.preprocessor import (
    PreProcessorConfig,
    read_preprocessor_main,
    write_preprocessor_main,
    read_subregions_file,
    load_model_from_preprocessor,
    load_complete_model,
    save_model_to_preprocessor,
    save_complete_model,
)

# Time Series ASCII I/O
from pyiwfm.io.timeseries_ascii import (
    TimeSeriesWriter,
    TimeSeriesReader,
    TimeSeriesFileConfig,
    write_timeseries,
    read_timeseries,
    format_iwfm_timestamp,
    parse_iwfm_timestamp,
)

# Unified Time Series I/O
from pyiwfm.io.timeseries import (
    TimeSeriesFileType,
    TimeUnit,
    TimeSeriesMetadata,
    UnifiedTimeSeriesConfig,
    UnifiedTimeSeriesReader,
    RecyclingTimeSeriesReader,
    detect_timeseries_format,
    read_timeseries_unified,
    get_timeseries_metadata,
)

# Groundwater I/O
from pyiwfm.io.groundwater import (
    GWFileConfig,
    GWMainFileConfig,
    GWMainFileReader,
    GroundwaterWriter,
    GroundwaterReader,
    FaceFlowSpec,
    write_groundwater,
    read_wells,
    read_initial_heads,
    read_subsidence,
    read_gw_main_file,
)

# Groundwater Sub-file Readers
from pyiwfm.io.gw_boundary import (
    SpecifiedFlowBC,
    SpecifiedHeadBC,
    GeneralHeadBC,
    ConstrainedGeneralHeadBC,
    GWBoundaryConfig,
    GWBoundaryReader,
    read_gw_boundary,
)

from pyiwfm.io.gw_pumping import (
    WellSpec,
    WellPumpingSpec,
    ElementPumpingSpec,
    ElementGroup,
    PumpingConfig,
    PumpingReader,
    read_gw_pumping,
)

from pyiwfm.io.gw_tiledrain import (
    TileDrainSpec,
    SubIrrigationSpec,
    TileDrainConfig,
    TileDrainReader,
    read_gw_tiledrain,
)

from pyiwfm.io.gw_subsidence import (
    SubsidenceNodeParams,
    SubsidenceConfig,
    SubsidenceReader,
    read_gw_subsidence,
)

# Groundwater Component Writer (simulation files)
from pyiwfm.io.gw_writer import (
    GWWriterConfig,
    GWComponentWriter,
    write_gw_component,
)

# Stream Component Writer (simulation files)
from pyiwfm.io.stream_writer import (
    StreamWriterConfig,
    StreamComponentWriter,
    write_stream_component,
)

# Streams I/O
from pyiwfm.io.streams import (
    StreamFileConfig,
    StreamMainFileConfig,
    StreamMainFileReader,
    StreamReachSpec,
    StreamSpecReader,
    StreamWriter,
    StreamReader,
    StreamBedParamRow,
    CrossSectionRow,
    StreamInitialConditionRow,
    parse_stream_version,
    stream_version_ge,
    write_stream,
    read_stream_nodes,
    read_diversions,
    read_stream_main_file,
    read_stream_spec,
)

# Stream data model extensions
from pyiwfm.components.stream import (
    CrossSectionData,
    StrmEvapNodeSpec,
)

# Stream Diversion Spec Reader
from pyiwfm.io.stream_diversion import (
    DiversionSpec,
    ElementGroup,
    RechargeZoneDest,
    DiversionSpecConfig,
    DiversionSpecReader,
    read_diversion_spec,
)

# Stream Bypass Spec Reader
from pyiwfm.io.stream_bypass import (
    BypassRatingTable,
    BypassSpec,
    BypassSpecConfig,
    BypassSpecReader,
    read_bypass_spec,
)

# Stream Inflow Reader
from pyiwfm.io.stream_inflow import (
    InflowSpec,
    InflowConfig,
    InflowReader,
    read_stream_inflow,
)

# Lakes I/O
from pyiwfm.io.lakes import (
    LakeFileConfig,
    LakeParamSpec,
    LakeOutflowRating,
    OutflowRatingPoint,
    LakeMainFileConfig,
    LakeMainFileReader,
    LakeWriter,
    LakeReader,
    write_lakes,
    read_lake_definitions,
    read_lake_elements,
    read_lake_main_file,
)

# Lake Component Writer (simulation files)
from pyiwfm.io.lake_writer import (
    LakeWriterConfig,
    LakeComponentWriter,
    write_lake_component,
)

# Root Zone I/O
from pyiwfm.io.rootzone import (
    RootZoneFileConfig,
    RootZoneMainFileConfig,
    RootZoneMainFileReader,
    RootZoneWriter,
    RootZoneReader,
    ElementSoilParamRow,
    write_rootzone,
    read_crop_types,
    read_soil_params,
    read_rootzone_main_file,
    parse_version as parse_rootzone_version,
    version_ge as rootzone_version_ge,
)

# Root Zone Sub-file Readers
from pyiwfm.io.rootzone_nonponded import (
    NonPondedCropConfig,
    NonPondedCropReader,
    CurveNumberRow,
    EtcPointerRow,
    IrrigationPointerRow,
    SoilMoisturePointerRow,
    SupplyReturnReuseRow,
    InitialConditionRow,
    read_nonponded_crop,
)

from pyiwfm.io.rootzone_ponded import (
    PondedCropConfig,
    PondedCropReader,
    read_ponded_crop,
)

from pyiwfm.io.rootzone_urban import (
    UrbanLandUseConfig,
    UrbanLandUseReader,
    UrbanCurveNumberRow,
    UrbanManagementRow,
    SurfaceFlowDestRow,
    UrbanInitialConditionRow,
    read_urban_landuse,
)

from pyiwfm.io.rootzone_native import (
    NativeRiparianConfig,
    NativeRiparianReader,
    NativeRiparianCNRow,
    NativeRiparianEtcRow,
    NativeRiparianInitialRow,
    read_native_riparian,
)

# Root Zone v4.x Sub-file Readers/Writers
from pyiwfm.io.rootzone_v4x import (
    # Data classes
    RootDepthRow,
    ElementCropRow,
    AgInitialConditionRow,
    NonPondedCropConfigV4x,
    PondedCropConfigV4x,
    UrbanElementRowV4x,
    UrbanInitialRowV4x,
    UrbanConfigV4x,
    NativeRiparianElementRowV4x,
    NativeRiparianInitialRowV4x,
    NativeRiparianConfigV4x,
    # Readers
    NonPondedCropReaderV4x,
    PondedCropReaderV4x,
    UrbanReaderV4x,
    NativeRiparianReaderV4x,
    # Writers
    NonPondedCropWriterV4x,
    PondedCropWriterV4x,
    UrbanWriterV4x,
    NativeRiparianWriterV4x,
    # Convenience functions
    read_nonponded_v4x,
    read_ponded_v4x,
    read_urban_v4x,
    read_native_riparian_v4x,
)

# Root Zone Component Writer (simulation files)
from pyiwfm.io.rootzone_writer import (
    RootZoneWriterConfig,
    RootZoneComponentWriter,
    write_rootzone_component,
)

# IWFM Time Series Data Writer (generic, all TS file types)
from pyiwfm.io.timeseries_writer import (
    DSSPathItem,
    TimeSeriesDataConfig,
    IWFMTimeSeriesDataWriter,
    make_pumping_ts_config,
    make_stream_inflow_ts_config,
    make_diversion_ts_config,
    make_precip_ts_config,
    make_et_ts_config,
    make_crop_coeff_ts_config,
    make_return_flow_ts_config,
    make_reuse_ts_config,
    make_irig_period_ts_config,
    make_ag_water_demand_ts_config,
    make_max_lake_elev_ts_config,
    make_stream_surface_area_ts_config,
)

# Simulation I/O
from pyiwfm.io.simulation import (
    SimulationConfig,
    SimulationFileConfig,
    SimulationWriter,
    SimulationReader,
    IWFMSimulationReader,
    write_simulation,
    read_simulation,
    read_iwfm_simulation,
)

# Simulation Main Writer (simulation control file)
from pyiwfm.io.simulation_writer import (
    SimulationMainConfig,
    SimulationMainWriter,
    write_simulation_main,
)

# Small Watershed I/O
from pyiwfm.io.small_watershed import (
    WatershedGWNode,
    WatershedSpec,
    WatershedRootZoneParams,
    WatershedAquiferParams,
    SmallWatershedMainConfig,
    SmallWatershedMainReader,
    read_small_watershed_main,
)

# Small Watershed Component Writer
from pyiwfm.io.small_watershed_writer import (
    SmallWatershedWriterConfig,
    SmallWatershedComponentWriter,
    write_small_watershed_component,
)

# Unsaturated Zone I/O
from pyiwfm.io.unsaturated_zone import (
    UnsatZoneElementData,
    UnsatZoneMainConfig,
    UnsatZoneMainReader,
    read_unsaturated_zone_main,
)

# Unsaturated Zone Component Writer
from pyiwfm.io.unsaturated_zone_writer import (
    UnsatZoneWriterConfig,
    UnsatZoneComponentWriter,
    write_unsaturated_zone_component,
)

# Supply Adjustment I/O
from pyiwfm.io.supply_adjust import (
    SupplyAdjustment,
    read_supply_adjustment,
    write_supply_adjustment,
)

# Model Loader
from pyiwfm.io.model_loader import (
    ModelLoadResult,
    CompleteModelLoader,
    load_complete_model as load_complete_iwfm_model,
    # Comment-aware loading
    ModelLoadResultWithComments,
    CommentAwareModelLoader,
    load_model_with_comments,
)

# Complete Model Writer
from pyiwfm.io.model_writer import (
    ModelWriteResult,
    TimeSeriesCopier,
    CompleteModelWriter,
    write_model,
    # Comment-aware writing
    write_model_with_comments,
    save_model_with_comments,
)

# Comment Preservation
from pyiwfm.io.comment_metadata import (
    PreserveMode,
    SectionComments,
    CommentMetadata,
    FileCommentMetadata,
)

from pyiwfm.io.comment_extractor import (
    LineType,
    ParsedLine,
    CommentExtractor,
    extract_comments,
    extract_and_save_comments,
)

from pyiwfm.io.comment_writer import (
    CommentWriter,
    CommentInjector,
)

# Comment-aware base classes
from pyiwfm.io.base import (
    CommentAwareReader,
    CommentAwareWriter,
)

# Zone I/O
from pyiwfm.io.zones import (
    read_iwfm_zone_file,
    write_iwfm_zone_file,
    read_geojson_zones,
    write_geojson_zones,
    read_zone_file,
    write_zone_file,
    auto_detect_zone_file,
)

# HDF5 I/O (h5py is a required dependency)
from pyiwfm.io.hdf5 import (
    HDF5ModelWriter,
    HDF5ModelReader,
    write_model_hdf5,
    read_model_hdf5,
)

# Note: pyiwfm.io.head_all_converter is intentionally NOT imported here.
# It is a script-capable module (python -m pyiwfm.io.head_all_converter)
# and eagerly importing it from __init__.py causes a runpy RuntimeWarning.
# Import directly: from pyiwfm.io.head_all_converter import convert_headall_to_hdf

# Optional DSS imports
try:
    from pyiwfm.io.dss import (
        DSSPathname,
        DSSPathnameTemplate,
        DSSFile,
        DSSFileClass,
        DSSTimeSeriesWriter,
        DSSTimeSeriesReader,
        HAS_DSS_LIBRARY,
        write_timeseries_to_dss,
        read_timeseries_from_dss,
        write_collection_to_dss,
    )

    _dss_exports = [
        "DSSPathname",
        "DSSPathnameTemplate",
        "DSSFile",
        "DSSFileClass",
        "DSSTimeSeriesWriter",
        "DSSTimeSeriesReader",
        "HAS_DSS_LIBRARY",
        "write_timeseries_to_dss",
        "read_timeseries_from_dss",
        "write_collection_to_dss",
    ]
except ImportError:
    _dss_exports = []

# Budget Reader (h5py is a required dependency)
from pyiwfm.io.budget import (
    BudgetReader,
    BudgetHeader,
    TimeStepInfo,
    ASCIIOutputInfo,
    LocationData,
    BUDGET_DATA_TYPES,
    parse_iwfm_datetime,
    julian_to_datetime,
    excel_julian_to_datetime,
)

# ZBudget Reader (h5py is a required dependency)
from pyiwfm.io.zbudget import (
    ZBudgetReader,
    ZBudgetHeader,
    ZoneInfo,
    ZBUDGET_DATA_TYPES,
)


__all__ = [
    # Base classes
    "FileInfo",
    "BaseReader",
    "BaseWriter",
    "ModelReader",
    "ModelWriter",
    "BinaryReader",
    "BinaryWriter",
    # Configuration classes
    "OutputFormat",
    "TimeSeriesOutputConfig",
    "PreProcessorFileConfig",
    "SmallWatershedFileConfig",
    "UnsatZoneFileConfig",
    "BudgetFileConfig",
    "ZBudgetFileConfig",
    "ModelOutputConfig",
    # Writer base classes
    "TemplateWriter",
    "TimeSeriesSpec",
    "ComponentWriter",
    "IWFMModelWriter",
    # PreProcessor Writer
    "PreProcessorWriter",
    "write_preprocessor_files",
    "write_nodes_file",
    "write_elements_file",
    "write_stratigraphy_file",
    # ASCII I/O
    "read_nodes",
    "read_elements",
    "read_stratigraphy",
    "write_nodes",
    "write_elements",
    "write_stratigraphy",
    # Binary I/O
    "FortranBinaryReader",
    "FortranBinaryWriter",
    "read_binary_mesh",
    "write_binary_mesh",
    "read_binary_stratigraphy",
    "write_binary_stratigraphy",
    # Preprocessor Binary I/O
    "PreprocessorBinaryData",
    "PreprocessorBinaryReader",
    "AppNodeData",
    "AppElementData",
    "AppFaceData",
    "SubregionData",
    "StratigraphyData",
    "StreamGWConnectorData",
    "LakeGWConnectorData",
    "StreamLakeConnectorData",
    "PreprocessorStreamData",
    "PreprocessorLakeData",
    "read_preprocessor_binary",
    # PreProcessor I/O
    "PreProcessorConfig",
    "read_preprocessor_main",
    "write_preprocessor_main",
    "read_subregions_file",
    "load_model_from_preprocessor",
    "load_complete_model",
    "save_model_to_preprocessor",
    "save_complete_model",
    # Time Series ASCII I/O
    "TimeSeriesWriter",
    "TimeSeriesReader",
    # Unified Time Series I/O
    "TimeSeriesFileType",
    "TimeUnit",
    "TimeSeriesMetadata",
    "UnifiedTimeSeriesConfig",
    "UnifiedTimeSeriesReader",
    "RecyclingTimeSeriesReader",
    "detect_timeseries_format",
    "read_timeseries_unified",
    "get_timeseries_metadata",
    "TimeSeriesFileConfig",
    "write_timeseries",
    "read_timeseries",
    "format_iwfm_timestamp",
    "parse_iwfm_timestamp",
    # Groundwater I/O
    "GWFileConfig",
    "GWMainFileConfig",
    "GWMainFileReader",
    "GroundwaterWriter",
    "GroundwaterReader",
    "FaceFlowSpec",
    "write_groundwater",
    "read_wells",
    "read_initial_heads",
    "read_subsidence",
    "read_gw_main_file",
    # Groundwater Sub-file Readers
    "SpecifiedFlowBC",
    "SpecifiedHeadBC",
    "GeneralHeadBC",
    "ConstrainedGeneralHeadBC",
    "GWBoundaryConfig",
    "GWBoundaryReader",
    "read_gw_boundary",
    "WellSpec",
    "WellPumpingSpec",
    "ElementPumpingSpec",
    "ElementGroup",
    "PumpingConfig",
    "PumpingReader",
    "read_gw_pumping",
    "TileDrainSpec",
    "SubIrrigationSpec",
    "TileDrainConfig",
    "TileDrainReader",
    "read_gw_tiledrain",
    "SubsidenceNodeParams",
    "SubsidenceConfig",
    "SubsidenceReader",
    "read_gw_subsidence",
    # Groundwater Component Writer
    "GWWriterConfig",
    "GWComponentWriter",
    "write_gw_component",
    # Stream Component Writer
    "StreamWriterConfig",
    "StreamComponentWriter",
    "write_stream_component",
    # Streams I/O
    "StreamFileConfig",
    "StreamMainFileConfig",
    "StreamMainFileReader",
    "StreamReachSpec",
    "StreamSpecReader",
    "StreamWriter",
    "StreamReader",
    "StreamBedParamRow",
    "CrossSectionRow",
    "StreamInitialConditionRow",
    "parse_stream_version",
    "stream_version_ge",
    "CrossSectionData",
    "StrmEvapNodeSpec",
    "write_stream",
    "read_stream_nodes",
    "read_diversions",
    "read_stream_main_file",
    "read_stream_spec",
    # Stream Diversion Spec Reader
    "DiversionSpec",
    "ElementGroup",
    "RechargeZoneDest",
    "DiversionSpecConfig",
    "DiversionSpecReader",
    "read_diversion_spec",
    # Stream Bypass Spec Reader
    "BypassRatingTable",
    "BypassSpec",
    "BypassSpecConfig",
    "BypassSpecReader",
    "read_bypass_spec",
    # Stream Inflow Reader
    "InflowSpec",
    "InflowConfig",
    "InflowReader",
    "read_stream_inflow",
    # Lakes I/O
    "LakeFileConfig",
    "LakeParamSpec",
    "LakeOutflowRating",
    "OutflowRatingPoint",
    "LakeMainFileConfig",
    "LakeMainFileReader",
    "LakeWriter",
    "LakeReader",
    "write_lakes",
    "read_lake_definitions",
    "read_lake_elements",
    "read_lake_main_file",
    # Lake Component Writer
    "LakeWriterConfig",
    "LakeComponentWriter",
    "write_lake_component",
    # Root Zone I/O
    "RootZoneFileConfig",
    "RootZoneMainFileConfig",
    "RootZoneMainFileReader",
    "RootZoneWriter",
    "RootZoneReader",
    "ElementSoilParamRow",
    "write_rootzone",
    "read_crop_types",
    "read_soil_params",
    "read_rootzone_main_file",
    "parse_rootzone_version",
    "rootzone_version_ge",
    # Root Zone Sub-file Readers
    "NonPondedCropConfig",
    "NonPondedCropReader",
    "CurveNumberRow",
    "EtcPointerRow",
    "IrrigationPointerRow",
    "SoilMoisturePointerRow",
    "SupplyReturnReuseRow",
    "InitialConditionRow",
    "read_nonponded_crop",
    "PondedCropConfig",
    "PondedCropReader",
    "read_ponded_crop",
    "UrbanLandUseConfig",
    "UrbanLandUseReader",
    "UrbanCurveNumberRow",
    "UrbanManagementRow",
    "SurfaceFlowDestRow",
    "UrbanInitialConditionRow",
    "read_urban_landuse",
    "NativeRiparianConfig",
    "NativeRiparianReader",
    "NativeRiparianCNRow",
    "NativeRiparianEtcRow",
    "NativeRiparianInitialRow",
    "read_native_riparian",
    # Root Zone v4.x Sub-file Readers/Writers
    "RootDepthRow",
    "ElementCropRow",
    "AgInitialConditionRow",
    "NonPondedCropConfigV4x",
    "PondedCropConfigV4x",
    "UrbanElementRowV4x",
    "UrbanInitialRowV4x",
    "UrbanConfigV4x",
    "NativeRiparianElementRowV4x",
    "NativeRiparianInitialRowV4x",
    "NativeRiparianConfigV4x",
    "NonPondedCropReaderV4x",
    "PondedCropReaderV4x",
    "UrbanReaderV4x",
    "NativeRiparianReaderV4x",
    "NonPondedCropWriterV4x",
    "PondedCropWriterV4x",
    "UrbanWriterV4x",
    "NativeRiparianWriterV4x",
    "read_nonponded_v4x",
    "read_ponded_v4x",
    "read_urban_v4x",
    "read_native_riparian_v4x",
    # Root Zone Component Writer
    "RootZoneWriterConfig",
    "RootZoneComponentWriter",
    "write_rootzone_component",
    # IWFM Time Series Data Writer
    "DSSPathItem",
    "TimeSeriesDataConfig",
    "IWFMTimeSeriesDataWriter",
    "make_pumping_ts_config",
    "make_stream_inflow_ts_config",
    "make_diversion_ts_config",
    "make_precip_ts_config",
    "make_et_ts_config",
    "make_crop_coeff_ts_config",
    "make_return_flow_ts_config",
    "make_reuse_ts_config",
    "make_irig_period_ts_config",
    "make_ag_water_demand_ts_config",
    "make_max_lake_elev_ts_config",
    "make_stream_surface_area_ts_config",
    # Simulation I/O
    "SimulationConfig",
    "SimulationFileConfig",
    "SimulationWriter",
    "SimulationReader",
    "IWFMSimulationReader",
    "write_simulation",
    "read_simulation",
    "read_iwfm_simulation",
    # Simulation Main Writer
    "SimulationMainConfig",
    "SimulationMainWriter",
    "write_simulation_main",
    # Small Watershed I/O
    "WatershedGWNode",
    "WatershedSpec",
    "WatershedRootZoneParams",
    "WatershedAquiferParams",
    "SmallWatershedMainConfig",
    "SmallWatershedMainReader",
    "read_small_watershed_main",
    # Small Watershed Component Writer
    "SmallWatershedWriterConfig",
    "SmallWatershedComponentWriter",
    "write_small_watershed_component",
    # Unsaturated Zone I/O
    "UnsatZoneElementData",
    "UnsatZoneMainConfig",
    "UnsatZoneMainReader",
    "read_unsaturated_zone_main",
    # Unsaturated Zone Component Writer
    "UnsatZoneWriterConfig",
    "UnsatZoneComponentWriter",
    "write_unsaturated_zone_component",
    # Supply Adjustment I/O
    "SupplyAdjustment",
    "read_supply_adjustment",
    "write_supply_adjustment",
    # Model Loader
    "ModelLoadResult",
    "CompleteModelLoader",
    "load_complete_iwfm_model",
    "ModelLoadResultWithComments",
    "CommentAwareModelLoader",
    "load_model_with_comments",
    # Complete Model Writer
    "ModelWriteConfig",
    "ModelWriteResult",
    "TimeSeriesCopier",
    "CompleteModelWriter",
    "write_model",
    "write_model_with_comments",
    "save_model_with_comments",
    # Comment Preservation
    "PreserveMode",
    "SectionComments",
    "CommentMetadata",
    "FileCommentMetadata",
    "LineType",
    "ParsedLine",
    "CommentExtractor",
    "extract_comments",
    "extract_and_save_comments",
    "CommentWriter",
    "CommentInjector",
    "CommentAwareReader",
    "CommentAwareWriter",
    # Zone I/O
    "read_iwfm_zone_file",
    "write_iwfm_zone_file",
    "read_geojson_zones",
    "write_geojson_zones",
    "read_zone_file",
    "write_zone_file",
    "auto_detect_zone_file",
    # HDF5 I/O
    "HDF5ModelWriter",
    "HDF5ModelReader",
    "write_model_hdf5",
    "read_model_hdf5",
    # DSS I/O (optional)
    *_dss_exports,
    # Budget Reader
    "BudgetReader",
    "BudgetHeader",
    "TimeStepInfo",
    "ASCIIOutputInfo",
    "LocationData",
    "BUDGET_DATA_TYPES",
    "parse_iwfm_datetime",
    "julian_to_datetime",
    "excel_julian_to_datetime",
    # ZBudget Reader
    "ZBudgetReader",
    "ZBudgetHeader",
    "ZoneInfo",
    "ZBUDGET_DATA_TYPES",
]
