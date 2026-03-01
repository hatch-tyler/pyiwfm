"""I/O handlers for IWFM file formats."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# ---------------------------------------------------------------------------
# Lazy import mapping: symbol_name -> (module_path, attr_name)
# ---------------------------------------------------------------------------
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Stream data model extensions
    "CrossSectionData": ("pyiwfm.components.stream", "CrossSectionData"),
    "StrmEvapNodeSpec": ("pyiwfm.components.stream", "StrmEvapNodeSpec"),
    # ASCII I/O
    "read_elements": ("pyiwfm.io.ascii", "read_elements"),
    "read_nodes": ("pyiwfm.io.ascii", "read_nodes"),
    "read_stratigraphy": ("pyiwfm.io.ascii", "read_stratigraphy"),
    "write_elements": ("pyiwfm.io.ascii", "write_elements"),
    "write_nodes": ("pyiwfm.io.ascii", "write_nodes"),
    "write_stratigraphy": ("pyiwfm.io.ascii", "write_stratigraphy"),
    # Base classes and configuration
    "BaseReader": ("pyiwfm.io.base", "BaseReader"),
    "BaseWriter": ("pyiwfm.io.base", "BaseWriter"),
    "BinaryReader": ("pyiwfm.io.base", "BinaryReader"),
    "BinaryWriter": ("pyiwfm.io.base", "BinaryWriter"),
    "CommentAwareReader": ("pyiwfm.io.base", "CommentAwareReader"),
    "CommentAwareWriter": ("pyiwfm.io.base", "CommentAwareWriter"),
    "FileInfo": ("pyiwfm.io.base", "FileInfo"),
    "ModelReader": ("pyiwfm.io.base", "ModelReader"),
    "ModelWriter": ("pyiwfm.io.base", "ModelWriter"),
    # Binary I/O
    "FortranBinaryReader": ("pyiwfm.io.binary", "FortranBinaryReader"),
    "FortranBinaryWriter": ("pyiwfm.io.binary", "FortranBinaryWriter"),
    "StreamAccessBinaryReader": ("pyiwfm.io.binary", "StreamAccessBinaryReader"),
    "write_binary_mesh": ("pyiwfm.io.binary", "write_binary_mesh"),
    "write_binary_stratigraphy": ("pyiwfm.io.binary", "write_binary_stratigraphy"),
    # Comment extraction
    "CommentExtractor": ("pyiwfm.io.comment_extractor", "CommentExtractor"),
    "LineType": ("pyiwfm.io.comment_extractor", "LineType"),
    "ParsedLine": ("pyiwfm.io.comment_extractor", "ParsedLine"),
    "extract_and_save_comments": ("pyiwfm.io.comment_extractor", "extract_and_save_comments"),
    "extract_comments": ("pyiwfm.io.comment_extractor", "extract_comments"),
    # Comment metadata
    "CommentMetadata": ("pyiwfm.io.comment_metadata", "CommentMetadata"),
    "FileCommentMetadata": ("pyiwfm.io.comment_metadata", "FileCommentMetadata"),
    "PreserveMode": ("pyiwfm.io.comment_metadata", "PreserveMode"),
    "SectionComments": ("pyiwfm.io.comment_metadata", "SectionComments"),
    # Comment writer
    "CommentInjector": ("pyiwfm.io.comment_writer", "CommentInjector"),
    "CommentWriter": ("pyiwfm.io.comment_writer", "CommentWriter"),
    # Config
    "BudgetFileConfig": ("pyiwfm.io.config", "BudgetFileConfig"),
    "ModelOutputConfig": ("pyiwfm.io.config", "ModelOutputConfig"),
    "ModelWriteConfig": ("pyiwfm.io.config", "ModelWriteConfig"),
    "OutputFormat": ("pyiwfm.io.config", "OutputFormat"),
    "PreProcessorFileConfig": ("pyiwfm.io.config", "PreProcessorFileConfig"),
    "SmallWatershedFileConfig": ("pyiwfm.io.config", "SmallWatershedFileConfig"),
    "TimeSeriesOutputConfig": ("pyiwfm.io.config", "TimeSeriesOutputConfig"),
    "UnsatZoneFileConfig": ("pyiwfm.io.config", "UnsatZoneFileConfig"),
    "ZBudgetFileConfig": ("pyiwfm.io.config", "ZBudgetFileConfig"),
    "GWFileConfigNew": ("pyiwfm.io.config", "GWFileConfig"),
    "LakeFileConfigNew": ("pyiwfm.io.config", "LakeFileConfig"),
    "RootZoneFileConfigNew": ("pyiwfm.io.config", "RootZoneFileConfig"),
    "SimulationFileConfigNew": ("pyiwfm.io.config", "SimulationFileConfig"),
    "StreamFileConfigNew": ("pyiwfm.io.config", "StreamFileConfig"),
    # Groundwater I/O
    "FaceFlowSpec": ("pyiwfm.io.groundwater", "FaceFlowSpec"),
    "GroundwaterReader": ("pyiwfm.io.groundwater", "GroundwaterReader"),
    "GroundwaterWriter": ("pyiwfm.io.groundwater", "GroundwaterWriter"),
    "GWFileConfig": ("pyiwfm.io.groundwater", "GWFileConfig"),
    "GWMainFileConfig": ("pyiwfm.io.groundwater", "GWMainFileConfig"),
    "GWMainFileReader": ("pyiwfm.io.groundwater", "GWMainFileReader"),
    "read_gw_main_file": ("pyiwfm.io.groundwater", "read_gw_main_file"),
    "read_initial_heads": ("pyiwfm.io.groundwater", "read_initial_heads"),
    "read_subsidence": ("pyiwfm.io.groundwater", "read_subsidence"),
    "read_wells": ("pyiwfm.io.groundwater", "read_wells"),
    "write_groundwater": ("pyiwfm.io.groundwater", "write_groundwater"),
    # Groundwater sub-file readers
    "ConstrainedGeneralHeadBC": ("pyiwfm.io.gw_boundary", "ConstrainedGeneralHeadBC"),
    "GeneralHeadBC": ("pyiwfm.io.gw_boundary", "GeneralHeadBC"),
    "GWBoundaryConfig": ("pyiwfm.io.gw_boundary", "GWBoundaryConfig"),
    "GWBoundaryReader": ("pyiwfm.io.gw_boundary", "GWBoundaryReader"),
    "SpecifiedFlowBC": ("pyiwfm.io.gw_boundary", "SpecifiedFlowBC"),
    "SpecifiedHeadBC": ("pyiwfm.io.gw_boundary", "SpecifiedHeadBC"),
    "read_gw_boundary": ("pyiwfm.io.gw_boundary", "read_gw_boundary"),
    "ElementGroup": ("pyiwfm.io.gw_pumping", "ElementGroup"),
    "ElementPumpingSpec": ("pyiwfm.io.gw_pumping", "ElementPumpingSpec"),
    "PumpingConfig": ("pyiwfm.io.gw_pumping", "PumpingConfig"),
    "PumpingReader": ("pyiwfm.io.gw_pumping", "PumpingReader"),
    "WellPumpingSpec": ("pyiwfm.io.gw_pumping", "WellPumpingSpec"),
    "WellSpec": ("pyiwfm.io.gw_pumping", "WellSpec"),
    "read_gw_pumping": ("pyiwfm.io.gw_pumping", "read_gw_pumping"),
    "SubsidenceConfig": ("pyiwfm.io.gw_subsidence", "SubsidenceConfig"),
    "SubsidenceNodeParams": ("pyiwfm.io.gw_subsidence", "SubsidenceNodeParams"),
    "SubsidenceReader": ("pyiwfm.io.gw_subsidence", "SubsidenceReader"),
    "read_gw_subsidence": ("pyiwfm.io.gw_subsidence", "read_gw_subsidence"),
    "SubIrrigationSpec": ("pyiwfm.io.gw_tiledrain", "SubIrrigationSpec"),
    "TileDrainConfig": ("pyiwfm.io.gw_tiledrain", "TileDrainConfig"),
    "TileDrainReader": ("pyiwfm.io.gw_tiledrain", "TileDrainReader"),
    "TileDrainSpec": ("pyiwfm.io.gw_tiledrain", "TileDrainSpec"),
    "read_gw_tiledrain": ("pyiwfm.io.gw_tiledrain", "read_gw_tiledrain"),
    # GW Component Writer
    "GWComponentWriter": ("pyiwfm.io.gw_writer", "GWComponentWriter"),
    "GWWriterConfig": ("pyiwfm.io.gw_writer", "GWWriterConfig"),
    "write_gw_component": ("pyiwfm.io.gw_writer", "write_gw_component"),
    # HDF5 I/O
    "HDF5ModelReader": ("pyiwfm.io.hdf5", "HDF5ModelReader"),
    "HDF5ModelWriter": ("pyiwfm.io.hdf5", "HDF5ModelWriter"),
    "read_model_hdf5": ("pyiwfm.io.hdf5", "read_model_hdf5"),
    "write_model_hdf5": ("pyiwfm.io.hdf5", "write_model_hdf5"),
    # Lake Component Writer
    "LakeComponentWriter": ("pyiwfm.io.lake_writer", "LakeComponentWriter"),
    "LakeWriterConfig": ("pyiwfm.io.lake_writer", "LakeWriterConfig"),
    "write_lake_component": ("pyiwfm.io.lake_writer", "write_lake_component"),
    # Lakes I/O
    "LakeFileConfig": ("pyiwfm.io.lakes", "LakeFileConfig"),
    "LakeMainFileConfig": ("pyiwfm.io.lakes", "LakeMainFileConfig"),
    "LakeMainFileReader": ("pyiwfm.io.lakes", "LakeMainFileReader"),
    "LakeOutflowRating": ("pyiwfm.io.lakes", "LakeOutflowRating"),
    "LakeParamSpec": ("pyiwfm.io.lakes", "LakeParamSpec"),
    "LakeReader": ("pyiwfm.io.lakes", "LakeReader"),
    "LakeWriter": ("pyiwfm.io.lakes", "LakeWriter"),
    "OutflowRatingPoint": ("pyiwfm.io.lakes", "OutflowRatingPoint"),
    "read_lake_definitions": ("pyiwfm.io.lakes", "read_lake_definitions"),
    "read_lake_elements": ("pyiwfm.io.lakes", "read_lake_elements"),
    "read_lake_main_file": ("pyiwfm.io.lakes", "read_lake_main_file"),
    "write_lakes": ("pyiwfm.io.lakes", "write_lakes"),
    # Model Loader
    "CommentAwareModelLoader": ("pyiwfm.io.model_loader", "CommentAwareModelLoader"),
    "CompleteModelLoader": ("pyiwfm.io.model_loader", "CompleteModelLoader"),
    "ModelLoadResult": ("pyiwfm.io.model_loader", "ModelLoadResult"),
    "ModelLoadResultWithComments": ("pyiwfm.io.model_loader", "ModelLoadResultWithComments"),
    "load_complete_model": ("pyiwfm.io.model_loader", "load_complete_model"),
    "load_complete_iwfm_model": ("pyiwfm.io.model_loader", "load_complete_model"),
    "load_model_with_comments": ("pyiwfm.io.model_loader", "load_model_with_comments"),
    # Model Packager
    "ModelPackageResult": ("pyiwfm.io.model_packager", "ModelPackageResult"),
    "collect_model_files": ("pyiwfm.io.model_packager", "collect_model_files"),
    "package_model": ("pyiwfm.io.model_packager", "package_model"),
    # Complete Model Writer
    "CompleteModelWriter": ("pyiwfm.io.model_writer", "CompleteModelWriter"),
    "ModelWriteResult": ("pyiwfm.io.model_writer", "ModelWriteResult"),
    "TimeSeriesCopier": ("pyiwfm.io.model_writer", "TimeSeriesCopier"),
    "save_model_with_comments": ("pyiwfm.io.model_writer", "save_model_with_comments"),
    "write_model": ("pyiwfm.io.model_writer", "write_model"),
    "write_model_with_comments": ("pyiwfm.io.model_writer", "write_model_with_comments"),
    # Preprocessor I/O
    "PreProcessorConfig": ("pyiwfm.io.preprocessor", "PreProcessorConfig"),
    "read_preprocessor_main": ("pyiwfm.io.preprocessor", "read_preprocessor_main"),
    "read_subregions_file": ("pyiwfm.io.preprocessor", "read_subregions_file"),
    "save_complete_model": ("pyiwfm.io.preprocessor", "save_complete_model"),
    "save_model_to_preprocessor": ("pyiwfm.io.preprocessor", "save_model_to_preprocessor"),
    "write_preprocessor_main": ("pyiwfm.io.preprocessor", "write_preprocessor_main"),
    # Preprocessor Binary I/O
    "AppElementData": ("pyiwfm.io.preprocessor_binary", "AppElementData"),
    "AppFaceData": ("pyiwfm.io.preprocessor_binary", "AppFaceData"),
    "AppNodeData": ("pyiwfm.io.preprocessor_binary", "AppNodeData"),
    "LakeGWConnectorData": ("pyiwfm.io.preprocessor_binary", "LakeGWConnectorData"),
    "PreprocessorBinaryData": ("pyiwfm.io.preprocessor_binary", "PreprocessorBinaryData"),
    "PreprocessorBinaryReader": ("pyiwfm.io.preprocessor_binary", "PreprocessorBinaryReader"),
    "StratigraphyData": ("pyiwfm.io.preprocessor_binary", "StratigraphyData"),
    "StreamGWConnectorData": ("pyiwfm.io.preprocessor_binary", "StreamGWConnectorData"),
    "StreamLakeConnectorData": ("pyiwfm.io.preprocessor_binary", "StreamLakeConnectorData"),
    "SubregionData": ("pyiwfm.io.preprocessor_binary", "SubregionData"),
    "read_preprocessor_binary": ("pyiwfm.io.preprocessor_binary", "read_preprocessor_binary"),
    "PreprocessorLakeData": ("pyiwfm.io.preprocessor_binary", "LakeData"),
    "PreprocessorStreamData": ("pyiwfm.io.preprocessor_binary", "StreamData"),
    # Preprocessor Writer
    "PreProcessorWriter": ("pyiwfm.io.preprocessor_writer", "PreProcessorWriter"),
    "write_elements_file": ("pyiwfm.io.preprocessor_writer", "write_elements_file"),
    "write_nodes_file": ("pyiwfm.io.preprocessor_writer", "write_nodes_file"),
    "write_preprocessor_files": ("pyiwfm.io.preprocessor_writer", "write_preprocessor_files"),
    "write_stratigraphy_file": ("pyiwfm.io.preprocessor_writer", "write_stratigraphy_file"),
    # Root Zone I/O
    "ElementSoilParamRow": ("pyiwfm.io.rootzone", "ElementSoilParamRow"),
    "RootZoneFileConfig": ("pyiwfm.io.rootzone", "RootZoneFileConfig"),
    "RootZoneMainFileConfig": ("pyiwfm.io.rootzone", "RootZoneMainFileConfig"),
    "RootZoneMainFileReader": ("pyiwfm.io.rootzone", "RootZoneMainFileReader"),
    "RootZoneReader": ("pyiwfm.io.rootzone", "RootZoneReader"),
    "RootZoneWriter": ("pyiwfm.io.rootzone", "RootZoneWriter"),
    "read_crop_types": ("pyiwfm.io.rootzone", "read_crop_types"),
    "read_rootzone_main_file": ("pyiwfm.io.rootzone", "read_rootzone_main_file"),
    "read_soil_params": ("pyiwfm.io.rootzone", "read_soil_params"),
    "write_rootzone": ("pyiwfm.io.rootzone", "write_rootzone"),
    "parse_rootzone_version": ("pyiwfm.io.rootzone", "parse_version"),
    "rootzone_version_ge": ("pyiwfm.io.rootzone", "version_ge"),
    # Root Zone Native/Riparian
    "NativeRiparianCNRow": ("pyiwfm.io.rootzone_native", "NativeRiparianCNRow"),
    "NativeRiparianConfig": ("pyiwfm.io.rootzone_native", "NativeRiparianConfig"),
    "NativeRiparianEtcRow": ("pyiwfm.io.rootzone_native", "NativeRiparianEtcRow"),
    "NativeRiparianInitialRow": ("pyiwfm.io.rootzone_native", "NativeRiparianInitialRow"),
    "NativeRiparianReader": ("pyiwfm.io.rootzone_native", "NativeRiparianReader"),
    "read_native_riparian": ("pyiwfm.io.rootzone_native", "read_native_riparian"),
    # Root Zone Non-ponded
    "CurveNumberRow": ("pyiwfm.io.rootzone_nonponded", "CurveNumberRow"),
    "EtcPointerRow": ("pyiwfm.io.rootzone_nonponded", "EtcPointerRow"),
    "InitialConditionRow": ("pyiwfm.io.rootzone_nonponded", "InitialConditionRow"),
    "IrrigationPointerRow": ("pyiwfm.io.rootzone_nonponded", "IrrigationPointerRow"),
    "NonPondedCropConfig": ("pyiwfm.io.rootzone_nonponded", "NonPondedCropConfig"),
    "NonPondedCropReader": ("pyiwfm.io.rootzone_nonponded", "NonPondedCropReader"),
    "SoilMoisturePointerRow": ("pyiwfm.io.rootzone_nonponded", "SoilMoisturePointerRow"),
    "SupplyReturnReuseRow": ("pyiwfm.io.rootzone_nonponded", "SupplyReturnReuseRow"),
    "read_nonponded_crop": ("pyiwfm.io.rootzone_nonponded", "read_nonponded_crop"),
    # Root Zone Ponded
    "PondedCropConfig": ("pyiwfm.io.rootzone_ponded", "PondedCropConfig"),
    "PondedCropReader": ("pyiwfm.io.rootzone_ponded", "PondedCropReader"),
    "read_ponded_crop": ("pyiwfm.io.rootzone_ponded", "read_ponded_crop"),
    # Root Zone Urban
    "SurfaceFlowDestRow": ("pyiwfm.io.rootzone_urban", "SurfaceFlowDestRow"),
    "UrbanCurveNumberRow": ("pyiwfm.io.rootzone_urban", "UrbanCurveNumberRow"),
    "UrbanInitialConditionRow": ("pyiwfm.io.rootzone_urban", "UrbanInitialConditionRow"),
    "UrbanLandUseConfig": ("pyiwfm.io.rootzone_urban", "UrbanLandUseConfig"),
    "UrbanLandUseReader": ("pyiwfm.io.rootzone_urban", "UrbanLandUseReader"),
    "UrbanManagementRow": ("pyiwfm.io.rootzone_urban", "UrbanManagementRow"),
    "read_urban_landuse": ("pyiwfm.io.rootzone_urban", "read_urban_landuse"),
    # Root Zone v4.x
    "AgInitialConditionRow": ("pyiwfm.io.rootzone_v4x", "AgInitialConditionRow"),
    "ElementCropRow": ("pyiwfm.io.rootzone_v4x", "ElementCropRow"),
    "NativeRiparianConfigV4x": ("pyiwfm.io.rootzone_v4x", "NativeRiparianConfigV4x"),
    "NativeRiparianElementRowV4x": ("pyiwfm.io.rootzone_v4x", "NativeRiparianElementRowV4x"),
    "NativeRiparianInitialRowV4x": ("pyiwfm.io.rootzone_v4x", "NativeRiparianInitialRowV4x"),
    "NativeRiparianReaderV4x": ("pyiwfm.io.rootzone_v4x", "NativeRiparianReaderV4x"),
    "NativeRiparianWriterV4x": ("pyiwfm.io.rootzone_v4x", "NativeRiparianWriterV4x"),
    "NonPondedCropConfigV4x": ("pyiwfm.io.rootzone_v4x", "NonPondedCropConfigV4x"),
    "NonPondedCropReaderV4x": ("pyiwfm.io.rootzone_v4x", "NonPondedCropReaderV4x"),
    "NonPondedCropWriterV4x": ("pyiwfm.io.rootzone_v4x", "NonPondedCropWriterV4x"),
    "PondedCropConfigV4x": ("pyiwfm.io.rootzone_v4x", "PondedCropConfigV4x"),
    "PondedCropReaderV4x": ("pyiwfm.io.rootzone_v4x", "PondedCropReaderV4x"),
    "PondedCropWriterV4x": ("pyiwfm.io.rootzone_v4x", "PondedCropWriterV4x"),
    "RootDepthRow": ("pyiwfm.io.rootzone_v4x", "RootDepthRow"),
    "UrbanConfigV4x": ("pyiwfm.io.rootzone_v4x", "UrbanConfigV4x"),
    "UrbanElementRowV4x": ("pyiwfm.io.rootzone_v4x", "UrbanElementRowV4x"),
    "UrbanInitialRowV4x": ("pyiwfm.io.rootzone_v4x", "UrbanInitialRowV4x"),
    "UrbanReaderV4x": ("pyiwfm.io.rootzone_v4x", "UrbanReaderV4x"),
    "UrbanWriterV4x": ("pyiwfm.io.rootzone_v4x", "UrbanWriterV4x"),
    "read_native_riparian_v4x": ("pyiwfm.io.rootzone_v4x", "read_native_riparian_v4x"),
    "read_nonponded_v4x": ("pyiwfm.io.rootzone_v4x", "read_nonponded_v4x"),
    "read_ponded_v4x": ("pyiwfm.io.rootzone_v4x", "read_ponded_v4x"),
    "read_urban_v4x": ("pyiwfm.io.rootzone_v4x", "read_urban_v4x"),
    # Root Zone Component Writer
    "RootZoneComponentWriter": ("pyiwfm.io.rootzone_writer", "RootZoneComponentWriter"),
    "RootZoneWriterConfig": ("pyiwfm.io.rootzone_writer", "RootZoneWriterConfig"),
    "write_rootzone_component": ("pyiwfm.io.rootzone_writer", "write_rootzone_component"),
    # Simulation I/O
    "IWFMSimulationReader": ("pyiwfm.io.simulation", "IWFMSimulationReader"),
    "SimulationConfig": ("pyiwfm.io.simulation", "SimulationConfig"),
    "SimulationFileConfig": ("pyiwfm.io.simulation", "SimulationFileConfig"),
    "SimulationReader": ("pyiwfm.io.simulation", "SimulationReader"),
    "SimulationWriter": ("pyiwfm.io.simulation", "SimulationWriter"),
    "read_iwfm_simulation": ("pyiwfm.io.simulation", "read_iwfm_simulation"),
    "read_simulation": ("pyiwfm.io.simulation", "read_simulation"),
    "write_simulation": ("pyiwfm.io.simulation", "write_simulation"),
    # Simulation Main Writer
    "SimulationMainConfig": ("pyiwfm.io.simulation_writer", "SimulationMainConfig"),
    "SimulationMainWriter": ("pyiwfm.io.simulation_writer", "SimulationMainWriter"),
    "write_simulation_main": ("pyiwfm.io.simulation_writer", "write_simulation_main"),
    # Small Watershed I/O
    "SmallWatershedMainConfig": ("pyiwfm.io.small_watershed", "SmallWatershedMainConfig"),
    "SmallWatershedMainReader": ("pyiwfm.io.small_watershed", "SmallWatershedMainReader"),
    "WatershedAquiferParams": ("pyiwfm.io.small_watershed", "WatershedAquiferParams"),
    "WatershedGWNode": ("pyiwfm.io.small_watershed", "WatershedGWNode"),
    "WatershedRootZoneParams": ("pyiwfm.io.small_watershed", "WatershedRootZoneParams"),
    "WatershedSpec": ("pyiwfm.io.small_watershed", "WatershedSpec"),
    "read_small_watershed_main": ("pyiwfm.io.small_watershed", "read_small_watershed_main"),
    # Small Watershed Component Writer
    "SmallWatershedComponentWriter": (
        "pyiwfm.io.small_watershed_writer",
        "SmallWatershedComponentWriter",
    ),
    "SmallWatershedWriterConfig": (
        "pyiwfm.io.small_watershed_writer",
        "SmallWatershedWriterConfig",
    ),
    "write_small_watershed_component": (
        "pyiwfm.io.small_watershed_writer",
        "write_small_watershed_component",
    ),
    # Stream Bypass Spec Reader
    "BypassRatingTable": ("pyiwfm.io.stream_bypass", "BypassRatingTable"),
    "BypassSpec": ("pyiwfm.io.stream_bypass", "BypassSpec"),
    "BypassSpecConfig": ("pyiwfm.io.stream_bypass", "BypassSpecConfig"),
    "BypassSpecReader": ("pyiwfm.io.stream_bypass", "BypassSpecReader"),
    "read_bypass_spec": ("pyiwfm.io.stream_bypass", "read_bypass_spec"),
    # Stream Diversion Spec Reader
    "DiversionSpec": ("pyiwfm.io.stream_diversion", "DiversionSpec"),
    "DiversionSpecConfig": ("pyiwfm.io.stream_diversion", "DiversionSpecConfig"),
    "DiversionSpecReader": ("pyiwfm.io.stream_diversion", "DiversionSpecReader"),
    "RechargeZoneDest": ("pyiwfm.io.stream_diversion", "RechargeZoneDest"),
    "read_diversion_spec": ("pyiwfm.io.stream_diversion", "read_diversion_spec"),
    # Stream Inflow Reader
    "InflowConfig": ("pyiwfm.io.stream_inflow", "InflowConfig"),
    "InflowReader": ("pyiwfm.io.stream_inflow", "InflowReader"),
    "InflowSpec": ("pyiwfm.io.stream_inflow", "InflowSpec"),
    "read_stream_inflow": ("pyiwfm.io.stream_inflow", "read_stream_inflow"),
    # Stream Component Writer
    "StreamComponentWriter": ("pyiwfm.io.stream_writer", "StreamComponentWriter"),
    "StreamWriterConfig": ("pyiwfm.io.stream_writer", "StreamWriterConfig"),
    "write_stream_component": ("pyiwfm.io.stream_writer", "write_stream_component"),
    # Streams I/O
    "CrossSectionRow": ("pyiwfm.io.streams", "CrossSectionRow"),
    "StreamBedParamRow": ("pyiwfm.io.streams", "StreamBedParamRow"),
    "StreamFileConfig": ("pyiwfm.io.streams", "StreamFileConfig"),
    "StreamInitialConditionRow": ("pyiwfm.io.streams", "StreamInitialConditionRow"),
    "StreamMainFileConfig": ("pyiwfm.io.streams", "StreamMainFileConfig"),
    "StreamMainFileReader": ("pyiwfm.io.streams", "StreamMainFileReader"),
    "StreamReachSpec": ("pyiwfm.io.streams", "StreamReachSpec"),
    "StreamReader": ("pyiwfm.io.streams", "StreamReader"),
    "StreamSpecReader": ("pyiwfm.io.streams", "StreamSpecReader"),
    "StreamWriter": ("pyiwfm.io.streams", "StreamWriter"),
    "parse_stream_version": ("pyiwfm.io.streams", "parse_stream_version"),
    "read_diversions": ("pyiwfm.io.streams", "read_diversions"),
    "read_stream_main_file": ("pyiwfm.io.streams", "read_stream_main_file"),
    "read_stream_nodes": ("pyiwfm.io.streams", "read_stream_nodes"),
    "read_stream_spec": ("pyiwfm.io.streams", "read_stream_spec"),
    "stream_version_ge": ("pyiwfm.io.streams", "stream_version_ge"),
    "write_stream": ("pyiwfm.io.streams", "write_stream"),
    # Supply Adjustment I/O
    "SupplyAdjustment": ("pyiwfm.io.supply_adjust", "SupplyAdjustment"),
    "read_supply_adjustment": ("pyiwfm.io.supply_adjust", "read_supply_adjustment"),
    "write_supply_adjustment": ("pyiwfm.io.supply_adjust", "write_supply_adjustment"),
    # Unified Time Series I/O
    "RecyclingTimeSeriesReader": ("pyiwfm.io.timeseries", "RecyclingTimeSeriesReader"),
    "TimeSeriesFileType": ("pyiwfm.io.timeseries", "TimeSeriesFileType"),
    "TimeSeriesMetadata": ("pyiwfm.io.timeseries", "TimeSeriesMetadata"),
    "TimeUnit": ("pyiwfm.io.timeseries", "TimeUnit"),
    "UnifiedTimeSeriesConfig": ("pyiwfm.io.timeseries", "UnifiedTimeSeriesConfig"),
    "UnifiedTimeSeriesReader": ("pyiwfm.io.timeseries", "UnifiedTimeSeriesReader"),
    "detect_timeseries_format": ("pyiwfm.io.timeseries", "detect_timeseries_format"),
    "get_timeseries_metadata": ("pyiwfm.io.timeseries", "get_timeseries_metadata"),
    "read_timeseries_unified": ("pyiwfm.io.timeseries", "read_timeseries_unified"),
    # Time Series ASCII I/O
    "TimeSeriesFileConfig": ("pyiwfm.io.timeseries_ascii", "TimeSeriesFileConfig"),
    "TimeSeriesReader": ("pyiwfm.io.timeseries_ascii", "TimeSeriesReader"),
    "TimeSeriesWriter": ("pyiwfm.io.timeseries_ascii", "TimeSeriesWriter"),
    "format_iwfm_timestamp": ("pyiwfm.io.timeseries_ascii", "format_iwfm_timestamp"),
    "iwfm_date_to_iso": ("pyiwfm.io.timeseries_ascii", "iwfm_date_to_iso"),
    "parse_iwfm_datetime": ("pyiwfm.io.timeseries_ascii", "parse_iwfm_datetime"),
    "parse_iwfm_timestamp": ("pyiwfm.io.timeseries_ascii", "parse_iwfm_timestamp"),
    "read_timeseries": ("pyiwfm.io.timeseries_ascii", "read_timeseries"),
    "write_timeseries": ("pyiwfm.io.timeseries_ascii", "write_timeseries"),
    # IWFM Time Series Data Writer
    "DSSPathItem": ("pyiwfm.io.timeseries_writer", "DSSPathItem"),
    "IWFMTimeSeriesDataWriter": ("pyiwfm.io.timeseries_writer", "IWFMTimeSeriesDataWriter"),
    "TimeSeriesDataConfig": ("pyiwfm.io.timeseries_writer", "TimeSeriesDataConfig"),
    "make_ag_water_demand_ts_config": (
        "pyiwfm.io.timeseries_writer",
        "make_ag_water_demand_ts_config",
    ),
    "make_crop_coeff_ts_config": ("pyiwfm.io.timeseries_writer", "make_crop_coeff_ts_config"),
    "make_diversion_ts_config": ("pyiwfm.io.timeseries_writer", "make_diversion_ts_config"),
    "make_et_ts_config": ("pyiwfm.io.timeseries_writer", "make_et_ts_config"),
    "make_irig_period_ts_config": ("pyiwfm.io.timeseries_writer", "make_irig_period_ts_config"),
    "make_max_lake_elev_ts_config": (
        "pyiwfm.io.timeseries_writer",
        "make_max_lake_elev_ts_config",
    ),
    "make_precip_ts_config": ("pyiwfm.io.timeseries_writer", "make_precip_ts_config"),
    "make_pumping_ts_config": ("pyiwfm.io.timeseries_writer", "make_pumping_ts_config"),
    "make_return_flow_ts_config": ("pyiwfm.io.timeseries_writer", "make_return_flow_ts_config"),
    "make_reuse_ts_config": ("pyiwfm.io.timeseries_writer", "make_reuse_ts_config"),
    "make_stream_inflow_ts_config": (
        "pyiwfm.io.timeseries_writer",
        "make_stream_inflow_ts_config",
    ),
    "make_stream_surface_area_ts_config": (
        "pyiwfm.io.timeseries_writer",
        "make_stream_surface_area_ts_config",
    ),
    # Unsaturated Zone I/O
    "UnsatZoneElementData": ("pyiwfm.io.unsaturated_zone", "UnsatZoneElementData"),
    "UnsatZoneMainConfig": ("pyiwfm.io.unsaturated_zone", "UnsatZoneMainConfig"),
    "UnsatZoneMainReader": ("pyiwfm.io.unsaturated_zone", "UnsatZoneMainReader"),
    "read_unsaturated_zone_main": ("pyiwfm.io.unsaturated_zone", "read_unsaturated_zone_main"),
    # Unsaturated Zone Component Writer
    "UnsatZoneComponentWriter": (
        "pyiwfm.io.unsaturated_zone_writer",
        "UnsatZoneComponentWriter",
    ),
    "UnsatZoneWriterConfig": ("pyiwfm.io.unsaturated_zone_writer", "UnsatZoneWriterConfig"),
    "write_unsaturated_zone_component": (
        "pyiwfm.io.unsaturated_zone_writer",
        "write_unsaturated_zone_component",
    ),
    # Writer base classes
    "ComponentWriter": ("pyiwfm.io.writer_base", "ComponentWriter"),
    "IWFMModelWriter": ("pyiwfm.io.writer_base", "IWFMModelWriter"),
    "TemplateWriter": ("pyiwfm.io.writer_base", "TemplateWriter"),
    "TimeSeriesSpec": ("pyiwfm.io.writer_base", "TimeSeriesSpec"),
    "TimeSeriesWriterNew": ("pyiwfm.io.writer_base", "TimeSeriesWriter"),
    # Zone I/O
    "auto_detect_zone_file": ("pyiwfm.io.zones", "auto_detect_zone_file"),
    "read_geojson_zones": ("pyiwfm.io.zones", "read_geojson_zones"),
    "read_iwfm_zone_file": ("pyiwfm.io.zones", "read_iwfm_zone_file"),
    "read_zone_file": ("pyiwfm.io.zones", "read_zone_file"),
    "write_geojson_zones": ("pyiwfm.io.zones", "write_geojson_zones"),
    "write_iwfm_zone_file": ("pyiwfm.io.zones", "write_iwfm_zone_file"),
    "write_zone_file": ("pyiwfm.io.zones", "write_zone_file"),
    # Data loaders
    "AreaDataManager": ("pyiwfm.io.area_loader", "AreaDataManager"),
    "LazyAreaDataLoader": ("pyiwfm.io.area_loader", "LazyAreaDataLoader"),
    "BUDGET_DATA_TYPES": ("pyiwfm.io.budget", "BUDGET_DATA_TYPES"),
    "ASCIIOutputInfo": ("pyiwfm.io.budget", "ASCIIOutputInfo"),
    "BudgetHeader": ("pyiwfm.io.budget", "BudgetHeader"),
    "BudgetReader": ("pyiwfm.io.budget", "BudgetReader"),
    "LocationData": ("pyiwfm.io.budget", "LocationData"),
    "TimeStepInfo": ("pyiwfm.io.budget", "TimeStepInfo"),
    "excel_julian_to_datetime": ("pyiwfm.io.budget", "excel_julian_to_datetime"),
    "julian_to_datetime": ("pyiwfm.io.budget", "julian_to_datetime"),
    # Budget PEST export
    "budget_to_pest_text": ("pyiwfm.io.budget_pest", "budget_to_pest_text"),
    "budget_to_pest_instruction": ("pyiwfm.io.budget_pest", "budget_to_pest_instruction"),
    # Budget control file parsers and Excel export
    "BudgetControlConfig": ("pyiwfm.io.budget_control", "BudgetControlConfig"),
    "BudgetOutputSpec": ("pyiwfm.io.budget_control", "BudgetOutputSpec"),
    "read_budget_control": ("pyiwfm.io.budget_control", "read_budget_control"),
    "budget_control_to_excel": ("pyiwfm.io.budget_excel", "budget_control_to_excel"),
    "budget_to_excel": ("pyiwfm.io.budget_excel", "budget_to_excel"),
    "apply_unit_conversion": ("pyiwfm.io.budget_utils", "apply_unit_conversion"),
    "filter_time_range": ("pyiwfm.io.budget_utils", "filter_time_range"),
    "format_title_lines": ("pyiwfm.io.budget_utils", "format_title_lines"),
    "SqliteCacheBuilder": ("pyiwfm.io.cache_builder", "SqliteCacheBuilder"),
    "is_cache_stale": ("pyiwfm.io.cache_builder", "is_cache_stale"),
    "SqliteCacheLoader": ("pyiwfm.io.cache_loader", "SqliteCacheLoader"),
    "LazyHeadDataLoader": ("pyiwfm.io.head_loader", "LazyHeadDataLoader"),
    "LazyHydrographDataLoader": ("pyiwfm.io.hydrograph_loader", "LazyHydrographDataLoader"),
    "IWFMHydrographReader": ("pyiwfm.io.hydrograph_reader", "IWFMHydrographReader"),
    # SimulationMessages.out Parser
    "MessageSeverity": ("pyiwfm.io.simulation_messages", "MessageSeverity"),
    "SimulationMessage": ("pyiwfm.io.simulation_messages", "SimulationMessage"),
    "SimulationMessagesReader": ("pyiwfm.io.simulation_messages", "SimulationMessagesReader"),
    "SimulationMessagesResult": ("pyiwfm.io.simulation_messages", "SimulationMessagesResult"),
    # SMP (Sample/Bore) I/O
    "SMPReader": ("pyiwfm.io.smp", "SMPReader"),
    "SMPRecord": ("pyiwfm.io.smp", "SMPRecord"),
    "SMPTimeSeries": ("pyiwfm.io.smp", "SMPTimeSeries"),
    "SMPWriter": ("pyiwfm.io.smp", "SMPWriter"),
    # ZBudget Reader
    "ZBUDGET_DATA_TYPES": ("pyiwfm.io.zbudget", "ZBUDGET_DATA_TYPES"),
    "ZBudgetHeader": ("pyiwfm.io.zbudget", "ZBudgetHeader"),
    "ZBudgetReader": ("pyiwfm.io.zbudget", "ZBudgetReader"),
    "ZoneInfo": ("pyiwfm.io.zbudget", "ZoneInfo"),
    # ZBudget control file parser and Excel export
    "ZBudgetControlConfig": ("pyiwfm.io.zbudget_control", "ZBudgetControlConfig"),
    "ZBudgetOutputSpec": ("pyiwfm.io.zbudget_control", "ZBudgetOutputSpec"),
    "read_zbudget_control": ("pyiwfm.io.zbudget_control", "read_zbudget_control"),
    "zbudget_control_to_excel": ("pyiwfm.io.zbudget_excel", "zbudget_control_to_excel"),
    "zbudget_to_excel": ("pyiwfm.io.zbudget_excel", "zbudget_to_excel"),
    # DSS I/O (optional)
    "HAS_DSS_LIBRARY": ("pyiwfm.io.dss", "HAS_DSS_LIBRARY"),
    "DSSFile": ("pyiwfm.io.dss", "DSSFile"),
    "DSSFileClass": ("pyiwfm.io.dss", "DSSFileClass"),
    "DSSPathname": ("pyiwfm.io.dss", "DSSPathname"),
    "DSSPathnameTemplate": ("pyiwfm.io.dss", "DSSPathnameTemplate"),
    "DSSTimeSeriesReader": ("pyiwfm.io.dss", "DSSTimeSeriesReader"),
    "DSSTimeSeriesWriter": ("pyiwfm.io.dss", "DSSTimeSeriesWriter"),
    "read_timeseries_from_dss": ("pyiwfm.io.dss", "read_timeseries_from_dss"),
    "write_collection_to_dss": ("pyiwfm.io.dss", "write_collection_to_dss"),
    "write_timeseries_to_dss": ("pyiwfm.io.dss", "write_timeseries_to_dss"),
}

# Note: pyiwfm.io.head_all_converter is intentionally NOT imported here.
# It is a script-capable module (python -m pyiwfm.io.head_all_converter)
# and eagerly importing it from __init__.py causes a runpy RuntimeWarning.
# Import directly: from pyiwfm.io.head_all_converter import convert_headall_to_hdf

# Build __all__ from the mapping (excluding internal aliases)
_DSS_NAMES = {
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
}
_INTERNAL_ALIASES = {
    "GWFileConfigNew",
    "LakeFileConfigNew",
    "RootZoneFileConfigNew",
    "SimulationFileConfigNew",
    "StreamFileConfigNew",
    "TimeSeriesWriterNew",
}

__all__ = [name for name in _LAZY_IMPORTS if name not in _INTERNAL_ALIASES | _DSS_NAMES]
# DSS exports are conditional — add them only if the DSS library is available
try:
    importlib.import_module("pyiwfm.io.dss")
    __all__.extend(sorted(_DSS_NAMES))
except ImportError:
    pass


def __getattr__(name: str) -> object:
    """Lazy import of io symbols and submodules (PEP 562).

    Looks up *name* in ``_LAZY_IMPORTS`` first, falling back to
    ``importlib.import_module`` for submodule access (e.g.
    ``pyiwfm.io.budget``).  Resolved values are cached in
    ``globals()`` so subsequent access is a plain dict lookup.
    """
    spec = _LAZY_IMPORTS.get(name)
    if spec is not None:
        module_path, attr_name = spec
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            raise AttributeError(f"module 'pyiwfm.io' has no attribute {name!r}") from None
        value = getattr(module, attr_name)
        globals()[name] = value
        return value

    # Fall back: try to import as a submodule
    try:
        module = importlib.import_module(f"pyiwfm.io.{name}")
    except ImportError:
        raise AttributeError(f"module 'pyiwfm.io' has no attribute {name!r}") from None
    globals()[name] = module
    return module


if TYPE_CHECKING:
    # Provide static type information for type checkers and IDEs.
    # At runtime these are never evaluated (guarded by TYPE_CHECKING).
    from pyiwfm.components.stream import CrossSectionData as CrossSectionData
    from pyiwfm.components.stream import StrmEvapNodeSpec as StrmEvapNodeSpec
    from pyiwfm.io.base import BaseReader as BaseReader
    from pyiwfm.io.base import BaseWriter as BaseWriter
    from pyiwfm.io.base import BinaryReader as BinaryReader
    from pyiwfm.io.base import BinaryWriter as BinaryWriter
    from pyiwfm.io.base import CommentAwareReader as CommentAwareReader
    from pyiwfm.io.base import CommentAwareWriter as CommentAwareWriter
    from pyiwfm.io.base import FileInfo as FileInfo
    from pyiwfm.io.base import ModelReader as ModelReader
    from pyiwfm.io.base import ModelWriter as ModelWriter
    from pyiwfm.io.budget import BudgetReader as BudgetReader
    from pyiwfm.io.groundwater import GroundwaterReader as GroundwaterReader
    from pyiwfm.io.model_loader import load_complete_model as load_complete_model
    from pyiwfm.io.timeseries_ascii import parse_iwfm_datetime as parse_iwfm_datetime
    from pyiwfm.io.timeseries_ascii import parse_iwfm_timestamp as parse_iwfm_timestamp
    from pyiwfm.io.writer_base import ComponentWriter as ComponentWriter
    from pyiwfm.io.writer_base import IWFMModelWriter as IWFMModelWriter
    from pyiwfm.io.writer_base import TemplateWriter as TemplateWriter
