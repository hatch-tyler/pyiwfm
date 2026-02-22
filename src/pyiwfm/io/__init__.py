"""I/O handlers for IWFM file formats."""

from __future__ import annotations

# Stream data model extensions
from pyiwfm.components.stream import (
    CrossSectionData,
    StrmEvapNodeSpec,
)

# ASCII I/O
from pyiwfm.io.ascii import (
    read_elements,
    read_nodes,
    read_stratigraphy,
    write_elements,
    write_nodes,
    write_stratigraphy,
)

# Base classes and configuration
# Comment-aware base classes
from pyiwfm.io.base import (
    BaseReader,
    BaseWriter,
    BinaryReader,
    BinaryWriter,
    CommentAwareReader,
    CommentAwareWriter,
    FileInfo,
    ModelReader,
    ModelWriter,
)

# Binary I/O
from pyiwfm.io.binary import (
    FortranBinaryReader,
    FortranBinaryWriter,
    StreamAccessBinaryReader,
    write_binary_mesh,
    write_binary_stratigraphy,
)
from pyiwfm.io.comment_extractor import (
    CommentExtractor,
    LineType,
    ParsedLine,
    extract_and_save_comments,
    extract_comments,
)

# Comment Preservation
from pyiwfm.io.comment_metadata import (
    CommentMetadata,
    FileCommentMetadata,
    PreserveMode,
    SectionComments,
)
from pyiwfm.io.comment_writer import (
    CommentInjector,
    CommentWriter,
)
from pyiwfm.io.config import (
    BudgetFileConfig,
    ModelOutputConfig,
    ModelWriteConfig,
    OutputFormat,
    PreProcessorFileConfig,
    SmallWatershedFileConfig,
    TimeSeriesOutputConfig,
    UnsatZoneFileConfig,
    ZBudgetFileConfig,
)
from pyiwfm.io.config import (
    GWFileConfig as GWFileConfigNew,  # noqa: F401
)
from pyiwfm.io.config import (
    LakeFileConfig as LakeFileConfigNew,  # noqa: F401
)
from pyiwfm.io.config import (
    RootZoneFileConfig as RootZoneFileConfigNew,  # noqa: F401
)
from pyiwfm.io.config import (
    SimulationFileConfig as SimulationFileConfigNew,  # noqa: F401
)
from pyiwfm.io.config import (
    StreamFileConfig as StreamFileConfigNew,  # noqa: F401
)

# Groundwater I/O
from pyiwfm.io.groundwater import (
    FaceFlowSpec,
    GroundwaterReader,
    GroundwaterWriter,
    GWFileConfig,
    GWMainFileConfig,
    GWMainFileReader,
    read_gw_main_file,
    read_initial_heads,
    read_subsidence,
    read_wells,
    write_groundwater,
)

# Groundwater Sub-file Readers
from pyiwfm.io.gw_boundary import (
    ConstrainedGeneralHeadBC,
    GeneralHeadBC,
    GWBoundaryConfig,
    GWBoundaryReader,
    SpecifiedFlowBC,
    SpecifiedHeadBC,
    read_gw_boundary,
)
from pyiwfm.io.gw_pumping import (
    ElementGroup,
    ElementPumpingSpec,
    PumpingConfig,
    PumpingReader,
    WellPumpingSpec,
    WellSpec,
    read_gw_pumping,
)
from pyiwfm.io.gw_subsidence import (
    SubsidenceConfig,
    SubsidenceNodeParams,
    SubsidenceReader,
    read_gw_subsidence,
)
from pyiwfm.io.gw_tiledrain import (
    SubIrrigationSpec,
    TileDrainConfig,
    TileDrainReader,
    TileDrainSpec,
    read_gw_tiledrain,
)

# Groundwater Component Writer (simulation files)
from pyiwfm.io.gw_writer import (
    GWComponentWriter,
    GWWriterConfig,
    write_gw_component,
)

# HDF5 I/O (h5py is a required dependency)
from pyiwfm.io.hdf5 import (
    HDF5ModelReader,
    HDF5ModelWriter,
    read_model_hdf5,
    write_model_hdf5,
)

# Lake Component Writer (simulation files)
from pyiwfm.io.lake_writer import (
    LakeComponentWriter,
    LakeWriterConfig,
    write_lake_component,
)

# Lakes I/O
from pyiwfm.io.lakes import (
    LakeFileConfig,
    LakeMainFileConfig,
    LakeMainFileReader,
    LakeOutflowRating,
    LakeParamSpec,
    LakeReader,
    LakeWriter,
    OutflowRatingPoint,
    read_lake_definitions,
    read_lake_elements,
    read_lake_main_file,
    write_lakes,
)

# Model Loader
# Convenience alias: load_complete_model from model_loader (the strong loader)
from pyiwfm.io.model_loader import (
    CommentAwareModelLoader,
    CompleteModelLoader,
    ModelLoadResult,
    # Comment-aware loading
    ModelLoadResultWithComments,
    load_complete_model,  # noqa: E402
    load_model_with_comments,
)
from pyiwfm.io.model_loader import (
    load_complete_model as load_complete_iwfm_model,
)

# Complete Model Writer
from pyiwfm.io.model_writer import (
    CompleteModelWriter,
    ModelWriteResult,
    TimeSeriesCopier,
    save_model_with_comments,
    write_model,
    # Comment-aware writing
    write_model_with_comments,
)

# PreProcessor I/O
from pyiwfm.io.preprocessor import (
    PreProcessorConfig,
    read_preprocessor_main,
    read_subregions_file,
    save_complete_model,
    save_model_to_preprocessor,
    write_preprocessor_main,
)

# Preprocessor Binary I/O
from pyiwfm.io.preprocessor_binary import (
    AppElementData,
    AppFaceData,
    AppNodeData,
    LakeGWConnectorData,
    PreprocessorBinaryData,
    PreprocessorBinaryReader,
    StratigraphyData,
    StreamGWConnectorData,
    StreamLakeConnectorData,
    SubregionData,
    read_preprocessor_binary,
)
from pyiwfm.io.preprocessor_binary import (
    LakeData as PreprocessorLakeData,
)
from pyiwfm.io.preprocessor_binary import (
    StreamData as PreprocessorStreamData,
)

# PreProcessor Writer
from pyiwfm.io.preprocessor_writer import (
    PreProcessorWriter,
    write_elements_file,
    write_nodes_file,
    write_preprocessor_files,
    write_stratigraphy_file,
)

# Root Zone I/O
from pyiwfm.io.rootzone import (
    ElementSoilParamRow,
    RootZoneFileConfig,
    RootZoneMainFileConfig,
    RootZoneMainFileReader,
    RootZoneReader,
    RootZoneWriter,
    read_crop_types,
    read_rootzone_main_file,
    read_soil_params,
    write_rootzone,
)
from pyiwfm.io.rootzone import (
    parse_version as parse_rootzone_version,
)
from pyiwfm.io.rootzone import (
    version_ge as rootzone_version_ge,
)
from pyiwfm.io.rootzone_native import (
    NativeRiparianCNRow,
    NativeRiparianConfig,
    NativeRiparianEtcRow,
    NativeRiparianInitialRow,
    NativeRiparianReader,
    read_native_riparian,
)

# Root Zone Sub-file Readers
from pyiwfm.io.rootzone_nonponded import (
    CurveNumberRow,
    EtcPointerRow,
    InitialConditionRow,
    IrrigationPointerRow,
    NonPondedCropConfig,
    NonPondedCropReader,
    SoilMoisturePointerRow,
    SupplyReturnReuseRow,
    read_nonponded_crop,
)
from pyiwfm.io.rootzone_ponded import (
    PondedCropConfig,
    PondedCropReader,
    read_ponded_crop,
)
from pyiwfm.io.rootzone_urban import (
    SurfaceFlowDestRow,
    UrbanCurveNumberRow,
    UrbanInitialConditionRow,
    UrbanLandUseConfig,
    UrbanLandUseReader,
    UrbanManagementRow,
    read_urban_landuse,
)

# Root Zone v4.x Sub-file Readers/Writers
from pyiwfm.io.rootzone_v4x import (
    AgInitialConditionRow,
    ElementCropRow,
    NativeRiparianConfigV4x,
    NativeRiparianElementRowV4x,
    NativeRiparianInitialRowV4x,
    NativeRiparianReaderV4x,
    NativeRiparianWriterV4x,
    NonPondedCropConfigV4x,
    # Readers
    NonPondedCropReaderV4x,
    # Writers
    NonPondedCropWriterV4x,
    PondedCropConfigV4x,
    PondedCropReaderV4x,
    PondedCropWriterV4x,
    # Data classes
    RootDepthRow,
    UrbanConfigV4x,
    UrbanElementRowV4x,
    UrbanInitialRowV4x,
    UrbanReaderV4x,
    UrbanWriterV4x,
    read_native_riparian_v4x,
    # Convenience functions
    read_nonponded_v4x,
    read_ponded_v4x,
    read_urban_v4x,
)

# Root Zone Component Writer (simulation files)
from pyiwfm.io.rootzone_writer import (
    RootZoneComponentWriter,
    RootZoneWriterConfig,
    write_rootzone_component,
)

# Simulation I/O
from pyiwfm.io.simulation import (
    IWFMSimulationReader,
    SimulationConfig,
    SimulationFileConfig,
    SimulationReader,
    SimulationWriter,
    read_iwfm_simulation,
    read_simulation,
    write_simulation,
)

# Simulation Main Writer (simulation control file)
from pyiwfm.io.simulation_writer import (
    SimulationMainConfig,
    SimulationMainWriter,
    write_simulation_main,
)

# Small Watershed I/O
from pyiwfm.io.small_watershed import (
    SmallWatershedMainConfig,
    SmallWatershedMainReader,
    WatershedAquiferParams,
    WatershedGWNode,
    WatershedRootZoneParams,
    WatershedSpec,
    read_small_watershed_main,
)

# Small Watershed Component Writer
from pyiwfm.io.small_watershed_writer import (
    SmallWatershedComponentWriter,
    SmallWatershedWriterConfig,
    write_small_watershed_component,
)

# Stream Bypass Spec Reader
from pyiwfm.io.stream_bypass import (
    BypassRatingTable,
    BypassSpec,
    BypassSpecConfig,
    BypassSpecReader,
    read_bypass_spec,
)

# Stream Diversion Spec Reader
from pyiwfm.io.stream_diversion import (  # type: ignore[assignment]
    DiversionSpec,
    DiversionSpecConfig,
    DiversionSpecReader,
    ElementGroup,  # noqa: F811
    RechargeZoneDest,
    read_diversion_spec,
)

# Stream Inflow Reader
from pyiwfm.io.stream_inflow import (
    InflowConfig,
    InflowReader,
    InflowSpec,
    read_stream_inflow,
)

# Stream Component Writer (simulation files)
from pyiwfm.io.stream_writer import (
    StreamComponentWriter,
    StreamWriterConfig,
    write_stream_component,
)

# Streams I/O
from pyiwfm.io.streams import (
    CrossSectionRow,
    StreamBedParamRow,
    StreamFileConfig,
    StreamInitialConditionRow,
    StreamMainFileConfig,
    StreamMainFileReader,
    StreamReachSpec,
    StreamReader,
    StreamSpecReader,
    StreamWriter,
    parse_stream_version,
    read_diversions,
    read_stream_main_file,
    read_stream_nodes,
    read_stream_spec,
    stream_version_ge,
    write_stream,
)

# Supply Adjustment I/O
from pyiwfm.io.supply_adjust import (
    SupplyAdjustment,
    read_supply_adjustment,
    write_supply_adjustment,
)

# Unified Time Series I/O
from pyiwfm.io.timeseries import (
    RecyclingTimeSeriesReader,
    TimeSeriesFileType,
    TimeSeriesMetadata,
    TimeUnit,
    UnifiedTimeSeriesConfig,
    UnifiedTimeSeriesReader,
    detect_timeseries_format,
    get_timeseries_metadata,
    read_timeseries_unified,
)

# Time Series ASCII I/O
from pyiwfm.io.timeseries_ascii import (
    TimeSeriesFileConfig,
    TimeSeriesReader,
    TimeSeriesWriter,
    format_iwfm_timestamp,
    parse_iwfm_timestamp,
    read_timeseries,
    write_timeseries,
)

# IWFM Time Series Data Writer (generic, all TS file types)
from pyiwfm.io.timeseries_writer import (
    DSSPathItem,
    IWFMTimeSeriesDataWriter,
    TimeSeriesDataConfig,
    make_ag_water_demand_ts_config,
    make_crop_coeff_ts_config,
    make_diversion_ts_config,
    make_et_ts_config,
    make_irig_period_ts_config,
    make_max_lake_elev_ts_config,
    make_precip_ts_config,
    make_pumping_ts_config,
    make_return_flow_ts_config,
    make_reuse_ts_config,
    make_stream_inflow_ts_config,
    make_stream_surface_area_ts_config,
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
    UnsatZoneComponentWriter,
    UnsatZoneWriterConfig,
    write_unsaturated_zone_component,
)
from pyiwfm.io.writer_base import (
    ComponentWriter,
    IWFMModelWriter,
    TemplateWriter,
    TimeSeriesSpec,
)
from pyiwfm.io.writer_base import (
    TimeSeriesWriter as TimeSeriesWriterNew,  # noqa: F401
)

# Zone I/O
from pyiwfm.io.zones import (
    auto_detect_zone_file,
    read_geojson_zones,
    read_iwfm_zone_file,
    read_zone_file,
    write_geojson_zones,
    write_iwfm_zone_file,
    write_zone_file,
)

# Note: pyiwfm.io.head_all_converter is intentionally NOT imported here.
# It is a script-capable module (python -m pyiwfm.io.head_all_converter)
# and eagerly importing it from __init__.py causes a runpy RuntimeWarning.
# Import directly: from pyiwfm.io.head_all_converter import convert_headall_to_hdf

# Optional DSS imports
try:
    from pyiwfm.io.dss import (  # noqa: F401
        HAS_DSS_LIBRARY,
        DSSFile,
        DSSFileClass,
        DSSPathname,
        DSSPathnameTemplate,
        DSSTimeSeriesReader,
        DSSTimeSeriesWriter,
        read_timeseries_from_dss,
        write_collection_to_dss,
        write_timeseries_to_dss,
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
    BUDGET_DATA_TYPES,
    ASCIIOutputInfo,
    BudgetHeader,
    BudgetReader,
    LocationData,
    TimeStepInfo,
    excel_julian_to_datetime,
    julian_to_datetime,
    parse_iwfm_datetime,
)

# ZBudget Reader (h5py is a required dependency)
from pyiwfm.io.zbudget import (
    ZBUDGET_DATA_TYPES,
    ZBudgetHeader,
    ZBudgetReader,
    ZoneInfo,
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
    "StreamAccessBinaryReader",
    "write_binary_mesh",
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


def __getattr__(name: str) -> object:
    """Lazy import of io submodules for mock.patch compatibility (PEP 562).

    On Python 3.10, submodule attributes may not be set on the package
    during __init__.py execution.  This fallback ensures that
    ``pyiwfm.io.<submodule>`` is accessible for ``unittest.mock.patch``.
    """
    import importlib

    try:
        module = importlib.import_module(f"pyiwfm.io.{name}")
    except ImportError:
        raise AttributeError(f"module 'pyiwfm.io' has no attribute {name!r}") from None
    globals()[name] = module
    return module
