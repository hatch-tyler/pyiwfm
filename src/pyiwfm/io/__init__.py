"""I/O handlers for IWFM file formats."""

from __future__ import annotations

import importlib as _importlib
import types as _types

# ---------------------------------------------------------------------------
# Eager imports
#
# Most ``pyiwfm.io`` submodules are pure-Python parsers with no heavy
# top-level dependencies, so importing them eagerly costs essentially
# nothing. Submodules that pull h5py / pandas / openpyxl / scipy at import
# time are deferred via ``_LAZY_IMPORTS`` below.
#
# An F401 suppression is required on each block because ``__all__`` is built
# dynamically from ``globals()`` further down (ruff cannot statically detect
# the re-export through that derivation).
# ---------------------------------------------------------------------------
from pyiwfm.components.stream import (  # noqa: F401
    CrossSectionData,
    StrmEvapNodeSpec,
)
from pyiwfm.io.ascii.comment_extractor import (  # noqa: F401
    CommentExtractor,
    LineType,
    ParsedLine,
    extract_and_save_comments,
    extract_comments,
)
from pyiwfm.io.ascii.comment_metadata import (  # noqa: F401
    CommentMetadata,
    FileCommentMetadata,
    PreserveMode,
    SectionComments,
)
from pyiwfm.io.ascii.comment_writer import (  # noqa: F401
    CommentInjector,
    CommentWriter,
)
from pyiwfm.io.base import (  # noqa: F401
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
from pyiwfm.io.binary import (  # noqa: F401  # noqa: F401
    AppElementData,
    AppFaceData,
    AppNodeData,
    FortranBinaryReader,
    FortranBinaryWriter,
    LakeGWConnectorData,
    PreprocessorBinaryData,
    PreprocessorBinaryReader,
    StratigraphyData,
    StreamAccessBinaryReader,
    StreamGWConnectorData,
    StreamLakeConnectorData,
    SubregionData,
    read_preprocessor_binary,
    write_binary_mesh,
    write_binary_stratigraphy,
)
from pyiwfm.io.binary import (  # noqa: F401
    LakeData as PreprocessorLakeData,
)
from pyiwfm.io.binary import (  # noqa: F401
    StreamData as PreprocessorStreamData,
)
from pyiwfm.io.budget.checks import (  # noqa: F401
    BalanceCheckResult,
    BudgetSanityReport,
    check_all_budgets,
    check_budget_balance,
)
from pyiwfm.io.budget.control import (  # noqa: F401
    BudgetControlConfig,
    BudgetOutputSpec,
    read_budget_control,
)
from pyiwfm.io.budget.zone_control import (  # noqa: F401
    ZBudgetControlConfig,
    ZBudgetOutputSpec,
    read_zbudget_control,
)
from pyiwfm.io.cache_builder import (  # noqa: F401
    SqliteCacheBuilder,
    is_cache_stale,
)
from pyiwfm.io.cache_loader import SqliteCacheLoader  # noqa: F401
from pyiwfm.io.config import (  # noqa: F401
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
from pyiwfm.io.config import (  # noqa: F401
    GWFileConfig as GWFileConfigNew,
)
from pyiwfm.io.config import (  # noqa: F401
    LakeFileConfig as LakeFileConfigNew,
)
from pyiwfm.io.config import (  # noqa: F401
    RootZoneFileConfig as RootZoneFileConfigNew,
)
from pyiwfm.io.config import (  # noqa: F401
    SimulationFileConfig as SimulationFileConfigNew,
)
from pyiwfm.io.config import (  # noqa: F401
    StreamFileConfig as StreamFileConfigNew,
)
from pyiwfm.io.drawdown import DrawdownComputer  # noqa: F401
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
from pyiwfm.io.groundwater import (  # noqa: F401
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
from pyiwfm.io.gw_boundary import (  # noqa: F401
    ConstrainedGeneralHeadBC,
    GeneralHeadBC,
    GWBoundaryConfig,
    GWBoundaryReader,
    SpecifiedFlowBC,
    SpecifiedHeadBC,
    read_gw_boundary,
)
from pyiwfm.io.gw_pumping import (  # noqa: F401
    ElementGroup,
    ElementPumpingSpec,
    PumpingConfig,
    PumpingReader,
    WellPumpingSpec,
    WellSpec,
    read_gw_pumping,
)
from pyiwfm.io.gw_subsidence import (  # noqa: F401
    SubsidenceConfig,
    SubsidenceNodeParams,
    SubsidenceReader,
    read_gw_subsidence,
)
from pyiwfm.io.gw_tiledrain import (  # noqa: F401
    SubIrrigationSpec,
    TileDrainConfig,
    TileDrainReader,
    TileDrainSpec,
    read_gw_tiledrain,
)
from pyiwfm.io.gw_writer import (  # noqa: F401
    GWComponentWriter,
    GWWriterConfig,
    write_gw_component,
)
from pyiwfm.io.hydrograph_reader import IWFMHydrographReader  # noqa: F401
from pyiwfm.io.lakes import (  # noqa: F401
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
from pyiwfm.io.lakes.writer import (  # noqa: F401
    LakeComponentWriter,
    LakeWriterConfig,
    write_lake_component,
)
from pyiwfm.io.model_loader import (  # noqa: F401
    CommentAwareModelLoader,
    CompleteModelLoader,
    ModelLoadResult,
    ModelLoadResultWithComments,
    load_complete_model,
    load_model_with_comments,
)
from pyiwfm.io.model_loader import (  # noqa: F401
    load_complete_model as load_complete_iwfm_model,
)
from pyiwfm.io.model_packager import (  # noqa: F401
    ModelPackageResult,
    collect_model_files,
    package_model,
)
from pyiwfm.io.model_writer import (  # noqa: F401
    CompleteModelWriter,
    ModelWriteResult,
    TimeSeriesCopier,
    save_model_with_comments,
    write_model,
    write_model_with_comments,
)
from pyiwfm.io.preprocessor import (  # noqa: F401
    PreProcessorConfig,
    read_preprocessor_main,
    read_subregions_file,
    save_complete_model,
    save_model_to_preprocessor,
    write_preprocessor_main,
)
from pyiwfm.io.preprocessor.mesh import (  # noqa: F401
    read_elements,
    read_nodes,
    read_stratigraphy,
    write_elements,
    write_nodes,
    write_stratigraphy,
)
from pyiwfm.io.preprocessor.writer import (  # noqa: F401
    PreProcessorWriter,
    write_preprocessor_files,
)
from pyiwfm.io.rootzone import (  # noqa: F401
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
from pyiwfm.io.rootzone import (  # noqa: F401
    parse_version as parse_rootzone_version,
)
from pyiwfm.io.rootzone import (  # noqa: F401
    version_ge as rootzone_version_ge,
)
from pyiwfm.io.rootzone.native import (  # noqa: F401
    NativeRiparianCNRow,
    NativeRiparianConfig,
    NativeRiparianEtcRow,
    NativeRiparianInitialRow,
    NativeRiparianReader,
    read_native_riparian,
)
from pyiwfm.io.rootzone.nonponded import (  # noqa: F401
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
from pyiwfm.io.rootzone.ponded import (  # noqa: F401
    PondedCropConfig,
    PondedCropReader,
    read_ponded_crop,
)
from pyiwfm.io.rootzone.urban import (  # noqa: F401
    SurfaceFlowDestRow,
    UrbanCurveNumberRow,
    UrbanInitialConditionRow,
    UrbanLandUseConfig,
    UrbanLandUseReader,
    UrbanManagementRow,
    read_urban_landuse,
)
from pyiwfm.io.rootzone.v4x import (  # noqa: F401
    AgInitialConditionRow,
    ElementCropRow,
    NativeRiparianConfigV4x,
    NativeRiparianElementRowV4x,
    NativeRiparianInitialRowV4x,
    NativeRiparianReaderV4x,
    NativeRiparianWriterV4x,
    NonPondedCropConfigV4x,
    NonPondedCropReaderV4x,
    NonPondedCropWriterV4x,
    PondedCropConfigV4x,
    PondedCropReaderV4x,
    PondedCropWriterV4x,
    RootDepthRow,
    UrbanConfigV4x,
    UrbanElementRowV4x,
    UrbanInitialRowV4x,
    UrbanReaderV4x,
    UrbanWriterV4x,
    read_native_riparian_v4x,
    read_nonponded_v4x,
    read_ponded_v4x,
    read_urban_v4x,
)
from pyiwfm.io.rootzone.writer import (  # noqa: F401
    RootZoneComponentWriter,
    RootZoneWriterConfig,
    write_rootzone_component,
)
from pyiwfm.io.simulation import (  # noqa: F401
    IWFMSimulationReader,
    SimulationConfig,
    SimulationFileConfig,
    SimulationReader,
    SimulationWriter,
    read_iwfm_simulation,
    read_simulation,
    write_simulation,
)
from pyiwfm.io.simulation.messages import (  # noqa: F401
    ConvergenceHotspot,
    ConvergenceRecord,
    MassBalanceRecord,
    MessageSeverity,
    SimulationMessage,
    SimulationMessagesReader,
    SimulationMessagesResult,
    TimestepCutRecord,
)
from pyiwfm.io.simulation.writer import (  # noqa: F401
    SimulationMainConfig,
    SimulationMainWriter,
    write_simulation_main,
)
from pyiwfm.io.small_watershed import (  # noqa: F401
    SmallWatershedMainConfig,
    SmallWatershedMainReader,
    WatershedAquiferParams,
    WatershedGWNode,
    WatershedRootZoneParams,
    WatershedSpec,
    read_small_watershed_main,
)
from pyiwfm.io.small_watershed.writer import (  # noqa: F401
    SmallWatershedComponentWriter,
    SmallWatershedWriterConfig,
    write_small_watershed_component,
)
from pyiwfm.io.smp import (  # noqa: F401
    SMPReader,
    SMPRecord,
    SMPTimeSeries,
    SMPWriter,
)
from pyiwfm.io.stream_bypass import (  # noqa: F401
    BypassRatingTable,
    BypassSpec,
    BypassSpecConfig,
    BypassSpecReader,
    read_bypass_spec,
)
from pyiwfm.io.stream_depletion import (  # noqa: F401
    StreamDepletionReport,
    StreamDepletionResult,
    compute_stream_depletion,
)
from pyiwfm.io.stream_diversion import (  # noqa: F401
    DiversionSpec,
    DiversionSpecConfig,
    DiversionSpecReader,
    RechargeZoneDest,
    read_diversion_spec,
)
from pyiwfm.io.stream_inflow import (  # noqa: F401
    InflowConfig,
    InflowReader,
    InflowSpec,
    read_stream_inflow,
)
from pyiwfm.io.stream_writer import (  # noqa: F401
    StreamComponentWriter,
    StreamWriterConfig,
    write_stream_component,
)
from pyiwfm.io.streams import (  # noqa: F401
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
from pyiwfm.io.supply_adjust import (  # noqa: F401
    SupplyAdjustment,
    read_supply_adjustment,
    write_supply_adjustment,
)
from pyiwfm.io.timeseries.ascii import (  # noqa: F401
    TimeSeriesFileConfig,
    TimeSeriesReader,
    TimeSeriesWriter,
    format_iwfm_timestamp,
    iwfm_date_to_iso,
    parse_iwfm_datetime,
    parse_iwfm_timestamp,
    read_timeseries,
    write_timeseries,
)
from pyiwfm.io.timeseries.writer import (  # noqa: F401
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
from pyiwfm.io.unsaturated_zone import (  # noqa: F401
    UnsatZoneElementData,
    UnsatZoneMainConfig,
    UnsatZoneMainReader,
    read_unsaturated_zone_main,
)
from pyiwfm.io.unsaturated_zone.writer import (  # noqa: F401
    UnsatZoneComponentWriter,
    UnsatZoneWriterConfig,
    write_unsaturated_zone_component,
)
from pyiwfm.io.writer_base import (  # noqa: F401
    ComponentWriter,
    IWFMModelWriter,
    TemplateWriter,
    TimeSeriesSpec,
)
from pyiwfm.io.writer_base import (  # noqa: F401
    TimeSeriesWriter as TimeSeriesWriterNew,
)
from pyiwfm.io.zones import (  # noqa: F401
    auto_detect_zone_file,
    read_geojson_zones,
    read_iwfm_zone_file,
    read_zone_file,
    write_geojson_zones,
    write_iwfm_zone_file,
    write_zone_file,
)

# Snapshot of public names bound by the eager imports above.
# Used to derive ``__all__`` without manual maintenance. Submodule attributes
# (auto-bound by Python on the parent package when ``from pkg.sub import …``
# runs) and the ``__future__`` feature flag are filtered out.
_EAGER_PUBLIC_NAMES = frozenset(
    name
    for name, value in globals().items()
    if not name.startswith("_")
    and name != "annotations"
    and not isinstance(value, _types.ModuleType)
)
del _types

# ---------------------------------------------------------------------------
# Lazy imports for heavy submodules
#
# These submodules pull h5py / pandas / openpyxl / scipy at import time. We
# defer them so that ``from pyiwfm.io import <cheap-symbol>`` doesn't trigger
# multi-second cold-loads of unrelated heavy dependencies.
#
# v2.0: ``head_all_converter`` and ``hydrograph_converter`` were merged into
# :class:`TimeSeriesCache` (see :mod:`pyiwfm.io.timeseries.lazy`). Use
# ``TimeSeriesCache.from_iwfm_headall_text(...)`` and
# ``TimeSeriesCache.from_iwfm_hydrograph_text(...)`` instead. The old loader
# classes (``LazyHeadDataLoader`` / ``LazyHydrographDataLoader``) were renamed
# to ``LazyNodalLoader`` / ``LazyTabularLoader`` to match the data shape (see
# ``docs/MIGRATION_v1_to_v2.md``).
# ---------------------------------------------------------------------------
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # area_loader (h5py)
    "AreaDataManager": ("pyiwfm.io.area_loader", "AreaDataManager"),
    "LazyAreaDataLoader": ("pyiwfm.io.area_loader", "LazyAreaDataLoader"),
    # budget (h5py + pandas)
    "ASCIIOutputInfo": ("pyiwfm.io.budget", "ASCIIOutputInfo"),
    "BUDGET_DATA_TYPES": ("pyiwfm.io.budget", "BUDGET_DATA_TYPES"),
    "BudgetHeader": ("pyiwfm.io.budget", "BudgetHeader"),
    "BudgetReader": ("pyiwfm.io.budget", "BudgetReader"),
    "LocationData": ("pyiwfm.io.budget", "LocationData"),
    "TimeStepInfo": ("pyiwfm.io.budget", "TimeStepInfo"),
    "excel_julian_to_datetime": ("pyiwfm.io.budget", "excel_julian_to_datetime"),
    "julian_to_datetime": ("pyiwfm.io.budget", "julian_to_datetime"),
    # budget_excel (pandas + openpyxl)
    "budget_control_to_excel": ("pyiwfm.io.budget.excel", "budget_control_to_excel"),
    "budget_to_excel": ("pyiwfm.io.budget.excel", "budget_to_excel"),
    # budget_pest (pandas)
    "budget_to_pest_instruction": ("pyiwfm.io.budget.pest", "budget_to_pest_instruction"),
    "budget_to_pest_text": ("pyiwfm.io.budget.pest", "budget_to_pest_text"),
    # budget_utils (pandas)
    "apply_unit_conversion": ("pyiwfm.io.budget._utils", "apply_unit_conversion"),
    "filter_time_range": ("pyiwfm.io.budget._utils", "filter_time_range"),
    "format_title_lines": ("pyiwfm.io.budget._utils", "format_title_lines"),
    # hdf5 (h5py)
    "HDF5ModelReader": ("pyiwfm.io.hdf5", "HDF5ModelReader"),
    "HDF5ModelWriter": ("pyiwfm.io.hdf5", "HDF5ModelWriter"),
    "read_model_hdf5": ("pyiwfm.io.hdf5", "read_model_hdf5"),
    "write_model_hdf5": ("pyiwfm.io.hdf5", "write_model_hdf5"),
    # timeseries (h5py + scipy)
    "RecyclingTimeSeriesReader": ("pyiwfm.io.timeseries", "RecyclingTimeSeriesReader"),
    "TimeSeriesFileType": ("pyiwfm.io.timeseries", "TimeSeriesFileType"),
    "TimeSeriesMetadata": ("pyiwfm.io.timeseries", "TimeSeriesMetadata"),
    "TimeUnit": ("pyiwfm.io.timeseries", "TimeUnit"),
    "UnifiedTimeSeriesConfig": ("pyiwfm.io.timeseries", "UnifiedTimeSeriesConfig"),
    "UnifiedTimeSeriesReader": ("pyiwfm.io.timeseries", "UnifiedTimeSeriesReader"),
    "detect_timeseries_format": ("pyiwfm.io.timeseries", "detect_timeseries_format"),
    "get_timeseries_metadata": ("pyiwfm.io.timeseries", "get_timeseries_metadata"),
    "read_timeseries_unified": ("pyiwfm.io.timeseries", "read_timeseries_unified"),
    # timeseries_io (h5py + pandas)
    "LazyNodalLoader": ("pyiwfm.io.timeseries.lazy", "LazyNodalLoader"),
    "LazyTabularLoader": ("pyiwfm.io.timeseries.lazy", "LazyTabularLoader"),
    "TimeSeriesCache": ("pyiwfm.io.timeseries.lazy", "TimeSeriesCache"),
    # zbudget (h5py + pandas)
    "ZBUDGET_DATA_TYPES": ("pyiwfm.io.budget.zone_reader", "ZBUDGET_DATA_TYPES"),
    "ZBudgetHeader": ("pyiwfm.io.budget.zone_reader", "ZBudgetHeader"),
    "ZBudgetReader": ("pyiwfm.io.budget.zone_reader", "ZBudgetReader"),
    "ZoneInfo": ("pyiwfm.io.budget.zone_reader", "ZoneInfo"),
    # zbudget_excel (pandas + openpyxl)
    "zbudget_control_to_excel": ("pyiwfm.io.budget.zone_excel", "zbudget_control_to_excel"),
    "zbudget_to_excel": ("pyiwfm.io.budget.zone_excel", "zbudget_to_excel"),
}

# Eagerly-imported names that are renamed re-exports kept for backwards
# compatibility. They remain importable but are excluded from ``__all__``
# so they don't show up as the canonical name in documentation.
_INTERNAL_ALIASES = frozenset(
    {
        "GWFileConfigNew",
        "LakeFileConfigNew",
        "RootZoneFileConfigNew",
        "SimulationFileConfigNew",
        "StreamFileConfigNew",
        "TimeSeriesWriterNew",
    }
)

__all__ = sorted((_EAGER_PUBLIC_NAMES | _LAZY_IMPORTS.keys()) - _INTERNAL_ALIASES)


def __getattr__(name: str) -> object:
    """Lazy import of heavy io symbols and submodule fallback (PEP 562).

    1. Names listed in ``_LAZY_IMPORTS`` are resolved by importing their
       declared submodule and returning the named attribute.
    2. Otherwise, fall back to ``importlib.import_module(f"pyiwfm.io.{name}")``
       so that ``mock.patch("pyiwfm.io.<submodule>.X")`` works on Python 3.10
       (the original motivation for adding ``__getattr__`` to this package).
    3. Resolved values are cached in ``globals()`` so subsequent access is a
       plain dict lookup.
    """
    spec = _LAZY_IMPORTS.get(name)
    if spec is not None:
        module_path, attr_name = spec
        module = _importlib.import_module(module_path)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value

    try:
        module = _importlib.import_module(f"pyiwfm.io.{name}")
    except ImportError:
        raise AttributeError(f"module 'pyiwfm.io' has no attribute {name!r}") from None
    globals()[name] = module
    return module
