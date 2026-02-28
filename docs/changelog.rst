Changelog
=========

All notable changes to pyiwfm will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[1.0.2] - 2026-02-27
--------------------

IWFM2OBS Model Discovery and Multi-Layer Output

Added
~~~~~

**Model File Discovery** (``pyiwfm.calibration.model_file_discovery``)

- ``discover_hydrograph_files()``: Parse IWFM simulation main file to auto-discover
  GW and stream hydrograph ``.out`` file paths and hydrograph metadata (bore IDs,
  layers, coordinates)
- ``HydrographFileInfo``: Dataclass with discovered paths, hydrograph locations,
  start date, and time unit

**Observation Well Specification** (``pyiwfm.calibration.obs_well_spec``)

- ``read_obs_well_spec()``: Read observation well specification files for multi-layer
  target processing (name, coordinates, element, screen top/bottom)
- ``ObsWellSpec``: Dataclass for multi-layer well screen geometry

**IWFM2OBS Model-Discovery Mode** (``pyiwfm.calibration.iwfm2obs``)

- ``iwfm2obs_from_model()``: Full IWFM2OBS workflow that reads ``.out`` files directly
  via simulation main file discovery — combines the old Fortran IWFM2OBS's direct
  file reading with the new multi-layer T-weighted averaging
- ``IWFM2OBSConfig``: Configuration dataclass for the integrated workflow
- ``write_multilayer_output()``: Write ``GW_MultiLayer.out`` format (Name, Date, Time,
  Simulated, T1-T4, NewTOS, NewBOS)
- ``write_multilayer_pest_ins()``: Write PEST instruction file with ``WLT{well:05d}_{timestep:05d}``
  naming at columns 50:60

**Hydrograph Reader Enhancement** (``pyiwfm.io.hydrograph_reader``)

- ``IWFMHydrographReader.get_columns_as_smp_dict()``: Extract ``.out`` columns as
  ``SMPTimeSeries`` dict, bridging the hydrograph reader to the interpolation pipeline

**CLI Model-Discovery Mode** (``pyiwfm.cli.iwfm2obs``)

- ``--model`` flag for automatic model file discovery from simulation main file
- ``--obs-gw``, ``--output-gw``, ``--obs-stream``, ``--output-stream`` for per-type
  observation/output SMP paths
- ``--well-spec``, ``--multilayer-out``, ``--multilayer-ins`` for multi-layer processing

[1.0.0] - 2026-02-24
--------------------

Calibration Tools, Clustering, and Publication-Quality Plotting

Added
~~~~~

**SMP Observation File I/O** (``pyiwfm.io.smp``)

- ``SMPReader`` / ``SMPWriter``: Read and write IWFM SMP (Sample/Bore) observation files
- ``SMPRecord``, ``SMPTimeSeries``: Typed containers for bore ID, datetime, value, exclusion flag
- Fixed-width parsing with sentinel value (NaN) handling

**SimulationMessages.out Parser** (``pyiwfm.io.simulation_messages``)

- ``SimulationMessagesReader``: Parse IWFM simulation diagnostic output files
- ``SimulationMessage``: Structured message with severity, procedure, spatial IDs
- Regex-based extraction of node, element, reach, and layer IDs
- ``to_geodataframe()``: Map messages to spatial locations for GIS analysis

**IWFM2OBS Time Interpolation** (``pyiwfm.calibration.iwfm2obs``)

- ``interpolate_to_obs_times()``: Linear/nearest interpolation of simulated to observed times
- ``compute_multilayer_weights()``: Transmissivity-weighted averaging for multi-layer wells
- ``compute_composite_head()``: Composite head from layer heads using T-weights
- ``iwfm2obs()``: Complete workflow function matching Fortran IWFM2OBS utility

**Typical Hydrograph Computation** (``pyiwfm.calibration.calctyphyd``)

- ``compute_typical_hydrographs()``: Seasonal averaging + de-meaning + weighted combination
- ``compute_seasonal_averages()``: Per-well seasonal period averaging
- ``read_cluster_weights()``: Parse cluster weight files for CalcTypHyd input

**Fuzzy C-Means Clustering** (``pyiwfm.calibration.clustering``)

- ``fuzzy_cmeans_cluster()``: NumPy-only fuzzy c-means with spatial + temporal features
- ``ClusteringResult``: Membership matrix, cluster centers, fuzzy partition coefficient
- ``to_weights_file()``: Export weights in CalcTypHyd-compatible format
- Feature extraction: cross-correlation, amplitude, trend, seasonal strength

**Calibration Plots** (``pyiwfm.visualization.calibration_plots``)

- ``plot_calibration_summary()``: Multi-panel publication figure (1:1, spatial bias, histogram, metrics)
- ``plot_hydrograph_panel()``: Grid of observed vs simulated hydrographs
- ``plot_metrics_table()``: Matplotlib table of per-well statistics
- ``plot_residual_histogram()``: Residual distribution with optional normal fit
- ``plot_water_budget_summary()`` / ``plot_zbudget_summary()``: Stacked bar budget charts
- ``plot_cluster_map()``: Spatial cluster membership visualization
- ``plot_typical_hydrographs()``: Overlay of typical hydrographs by cluster

**New Plot Functions** (``pyiwfm.visualization.plotting``)

- ``plot_one_to_one()``: Scatter with 1:1 line, regression, and metrics text box
- ``plot_spatial_bias()``: Diverging colormap of observation bias on mesh background

**Publication Matplotlib Style** (``visualization/styles/pyiwfm-publication.mplstyle``)

- Journal-quality defaults: serif fonts, no top/right spines, 300 DPI, constrained layout

**Scaled RMSE Metric** (``pyiwfm.comparison.metrics``)

- ``scaled_rmse()``: Dimensionless RMSE / (max - min) for cross-site comparison
- Added ``scaled_rmse`` field to ``ComparisonMetrics`` dataclass

**CLI Subcommands**

- ``pyiwfm iwfm2obs``: Time interpolation of simulated to observed SMP times
- ``pyiwfm calctyphyd``: Compute typical hydrographs from clustered observation wells

[0.4.0] - 2026-01-15
--------------------

Supplemental Package Support and Web Viewer Enhancements

Fixed
~~~~~

**Root Zone Version-Dependent Parsing Bugs**

- Fixed ARSCLFL (land use area scaling) version guard: only read for v4.12+, not v4.11+
- Fixed FinalMoistureOutFile (FMFL) read: skip for v4.12+ where it was removed
- Fixed root zone soil parameter table parsing when ``n_elements`` is known

Changed
~~~~~~~

**I/O Reader Deduplication** (``pyiwfm.io.iwfm_reader``)

- Centralized ``resolve_path()``, ``next_data_or_empty()``, ``parse_version()``,
  and ``version_ge()`` into ``iwfm_reader.py`` — the canonical module for all
  IWFM file-reading utilities
- Replaced 14 identical ``_next_data_or_empty`` method copies across reader modules
  with thin wrappers delegating to the central function
- Replaced 11 identical ``_resolve_path`` copies (10 methods + 1 module-level function
  in ``preprocessor.py``) with delegations to ``iwfm_reader.resolve_path()``
- Unified version parsing: ``rootzone.parse_version`` and ``streams.parse_stream_version``
  merged into ``iwfm_reader.parse_version()`` (handles both ``.`` and ``-`` separators)
- Net reduction of ~42 lines across 16 files with no behavior changes

Added
~~~~~

**Small Watershed Component** (``pyiwfm.components.small_watershed``)

- ``AppSmallWatershed``: Container for small watershed model units
- ``WatershedUnit``: Individual watershed with root zone and aquifer parameters
- ``WatershedGWNode``: Groundwater node connection with percolation rates
- ``from_config()``: Build component from reader config
- ``validate()``: Check areas, stream node references, GW node connectivity

**Unsaturated Zone Component** (``pyiwfm.components.unsaturated_zone``)

- ``AppUnsatZone``: Container for unsaturated zone elements
- ``UnsatZoneElement``: Per-element layer data with initial soil moisture
- ``UnsatZoneLayer``: Layer-level soil hydraulic properties
- ``from_config()``: Build component from reader config
- ``validate()``: Check layer counts and element consistency

**Small Watershed Writer** (``pyiwfm.io.small_watershed_writer``)

- ``SmallWatershedComponentWriter``: Template-based writer for small watershed files
- ``SmallWatershedWriterConfig``: Writer configuration with output paths
- Jinja2 template for IWFM v4.0 format with geospatial, root zone, and aquifer sections

**Unsaturated Zone Writer** (``pyiwfm.io.unsaturated_zone_writer``)

- ``UnsatZoneComponentWriter``: Template-based writer for unsaturated zone files
- ``UnsatZoneWriterConfig``: Writer configuration with output paths
- Jinja2 template for IWFM v4.0 format with element data and initial moisture sections

**Model Integration**

- ``IWFMModel.small_watersheds``: Optional small watershed component attribute
- ``IWFMModel.unsaturated_zone``: Optional unsaturated zone component attribute
- ``CompleteModelWriter``: Dedicated writers replace passthrough file copy for both packages
- Full roundtrip support for small watershed and unsaturated zone files

**BaseComponent ABC** (``pyiwfm.core.base_component``)

- ``BaseComponent``: Abstract base class with ``validate()`` and ``n_items`` interface
- All 6 components (AppGW, AppStream, AppLake, RootZone, AppSmallWatershed,
  AppUnsatZone) inherit from ``BaseComponent``

**Model Factory Extraction** (``pyiwfm.core.model_factory``)

- Extracted 6 helper functions (~420 lines) from ``IWFMModel`` to reduce
  the God Object pattern
- Public functions: ``build_reaches_from_node_reach_ids``,
  ``apply_kh_anomalies``, ``apply_parametric_grids``,
  ``apply_parametric_subsidence``, ``binary_data_to_model``,
  ``resolve_stream_node_coordinates``
- ``IWFMModel`` classmethods delegate to factory functions (backward compatible)

**Writer Config Consolidation** (``pyiwfm.io.writer_config_base``)

- ``BaseComponentWriterConfig``: Shared base dataclass for all 6 component
  writer configs (common fields: output_dir, version, length/volume
  factors/units, subdir)

**I/O Reader Deduplication**

- Consolidated ``_resolve_path()`` and ``_parse_version()`` duplicates across
  readers to use canonical implementations from ``iwfm_reader.py``
- Consolidated Fortran binary record I/O between ``base.py`` and ``binary.py``

**Web Viewer Performance Improvements**

- Cached ``node_id_to_idx`` and ``elem_id_to_idx`` mappings in ``ModelState``
  instead of rebuilding per request
- Cached hydrograph location data (GW and stream) in ``ModelState``
- Drawdown endpoint now supports pagination: ``offset``, ``limit``, ``skip``
  parameters for frame-by-frame animation playback

**New Web Viewer API Endpoints**

- ``GET /api/export/geopackage``: Download multi-layer GeoPackage (nodes,
  elements, streams, subregions, boundary) via ``GISExporter``
- ``GET /api/export/plot/{plot_type}``: Generate publication-quality matplotlib
  figures (mesh, elements, streams, heads) as PNG or SVG
- ``POST /api/model/compare``: Load a second model and compare meshes/stratigraphy
  via ``ModelDiffer``
- ``GET /api/results/statistics``: Time-aggregated head statistics
  (min, max, mean, std per node across all timesteps)

**FastAPI Web Viewer Enhancements** (2026-02)

- Results Map tab with deck.gl + MapLibre GL for head contour visualization
- Budgets tab with Plotly charts for GW, stream, and other budget types
- Hydrograph overlay with observed vs simulated data
- Server-side coordinate reprojection via pyproj (model CRS to WGS84)
- Stream node z elevation from stratigraphy ground surface
- Monthly budget timestep support using ``relativedelta``
- Budget units populated from column type codes
- CRS default corrected to proj string for C2VSimFG

**Component Exports**

- Top-level ``pyiwfm`` package now exports AppGW, AppStream, AppLake, RootZone,
  AppSmallWatershed, AppUnsatZone for convenient ``from pyiwfm import AppGW``

**Budget & Zone Budget Excel Export** (``pyiwfm.io.budget_excel``, ``pyiwfm.io.zbudget_excel``)

- ``budget_to_excel()``: Export budget HDF5 data to formatted Excel workbooks
  (one sheet per location, bold titles/headers, auto-fit columns)
- ``zbudget_to_excel()``: Same for zone budget data (one sheet per zone)
- ``budget_control_to_excel()`` / ``zbudget_control_to_excel()``: Batch export
  from control file configuration (one ``.xlsx`` per budget spec)
- Unit conversion factors (FACTLTOU/FACTAROU/FACTVLOU) applied per column type
  using codes from ``Budget_Parameters.f90``

**Budget & Zone Budget Control File Parsers** (``pyiwfm.io.budget_control``, ``pyiwfm.io.zbudget_control``)

- ``read_budget_control()``: Parse IWFM budget post-processor control files
  (FACTLTOU, UNITLTOU, FACTAROU, UNITAROU, FACTVLOU, UNITVLOU, dates,
  per-budget HDF5 paths, output paths, location IDs)
- ``read_zbudget_control()``: Parse IWFM zone budget control files
  (same pattern with zone definition file support)
- ``BudgetControlConfig`` / ``ZBudgetControlConfig``: Typed dataclasses

**Budget Utilities** (``pyiwfm.io.budget_utils``)

- ``apply_unit_conversion()``: Apply IWFM conversion factors per column type
- ``format_title_lines()``: Substitute ``@UNITVL@``, ``@UNITAR@``, ``@LOCNAME@``,
  ``@AREA@`` markers in title templates
- ``filter_time_range()``: Filter DataFrames by IWFM date range

**Budget CLI Commands** (``pyiwfm.cli.budget``, ``pyiwfm.cli.zbudget``)

- ``pyiwfm budget <control_file>``: Export budgets to Excel from control file
- ``pyiwfm zbudget <control_file>``: Export zone budgets to Excel from control file
- ``--output-dir`` flag to override output directory

**Budget Unit Conversion in Readers**

- ``BudgetReader.get_dataframe()`` now accepts keyword-only ``length_factor``,
  ``area_factor``, ``volume_factor`` for on-the-fly unit conversion
- ``ZBudgetReader.get_dataframe()`` now accepts keyword-only ``volume_factor``
- Backward compatible: all factors default to 1.0

**Budget Excel Download Endpoints**

- ``GET /api/budgets/{budget_type}/excel``: Download formatted budget workbook
- ``GET /api/export/budget-excel``: Download budget Excel from the export routes

**Documentation**

- Added ~30 missing module entries to API docs (``docs/api/io.rst``)
- Added Small Watershed and Unsaturated Zone to component docs
- Reorganized I/O docs into logical sections (Core, GW, Stream, Lake, RZ, Supplemental, etc.)
- Added BaseComponent and model_factory to core API docs
- Added writer_config_base to I/O API docs
- Added API routes summary to visualization docs

[0.2.0] - 2025-06-01
--------------------

Complete IWFM File Writers Implementation

Added
~~~~~

**IWFMModel Class Methods**

- ``IWFMModel.from_preprocessor(pp_file)``: Load from preprocessor input files (mesh, stratigraphy, geometry)
- ``IWFMModel.from_preprocessor_binary(binary_file)``: Load from native IWFM preprocessor binary (``ACCESS='STREAM'``)
- ``IWFMModel.from_simulation(sim_file)``: Load complete model from simulation input file
- ``IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)``: Load using both files
- ``IWFMModel.from_hdf5(hdf5_file)``: Load from HDF5 file
- ``model.to_preprocessor(output_dir)``: Save to preprocessor input files
- ``model.to_simulation(output_dir)``: Save complete model to simulation files
- ``model.to_hdf5(output_file)``: Save to HDF5 format
- ``model.to_binary(output_file)``: Save mesh/stratigraphy to binary format
- ``model.grid`` property: Alias for ``mesh`` for compatibility

**Complete Model I/O**

- ``pyiwfm.io.load_complete_model``: Load complete IWFM model from simulation main file
- ``pyiwfm.io.save_complete_model``: Save complete IWFM model to all input files
- Full roundtrip support for reading, modifying, and writing IWFM models

**Time Series ASCII I/O**

- ``pyiwfm.io.timeseries_ascii``: ASCII time series reader/writer module
- ``TimeSeriesWriter``: Write IWFM ASCII time series files with 21-char timestamp format
- ``TimeSeriesReader``: Read IWFM ASCII time series files
- ``format_iwfm_timestamp``: Format datetime to IWFM 21-character timestamp
- ``parse_iwfm_timestamp``: Parse IWFM timestamp string to datetime

**Groundwater Component I/O**

- ``pyiwfm.io.groundwater``: Complete groundwater component file I/O
- ``GroundwaterWriter``: Write wells, pumping, boundary conditions, aquifer parameters
- ``GroundwaterReader``: Read groundwater component files
- ``GWFileConfig``: Configuration for groundwater file paths

**Stream Component I/O**

- ``pyiwfm.io.streams``: Complete stream network component file I/O
- ``StreamWriter``: Write stream nodes, reaches, diversions, bypasses, rating curves
- ``StreamReader``: Read stream component files
- ``StreamFileConfig``: Configuration for stream file paths

**Lake Component I/O**

- ``pyiwfm.io.lakes``: Complete lake component file I/O
- ``LakeWriter``: Write lake definitions, elements, rating curves, outflows
- ``LakeReader``: Read lake component files
- ``LakeFileConfig``: Configuration for lake file paths

**Root Zone Component I/O**

- ``pyiwfm.io.rootzone``: Complete root zone component file I/O
- ``RootZoneWriter``: Write crop types, soil parameters, land use
- ``RootZoneReader``: Read root zone component files
- ``RootZoneFileConfig``: Configuration for root zone file paths

**Simulation Control I/O**

- ``pyiwfm.io.simulation``: Simulation control file I/O
- ``SimulationWriter``: Write simulation main control file
- ``SimulationReader``: Read simulation control file
- ``SimulationConfig``: Simulation configuration dataclass

**HEC-DSS 7 Support**

- ``pyiwfm.io.dss``: Complete HEC-DSS 7 time series I/O package
- ``DSSFile``: Context manager for DSS file operations using ctypes
- ``DSSPathname``: DSS pathname representation (/A/B/C/D/E/F/)
- ``DSSPathnameTemplate``: Template for generating pathnames
- ``DSSTimeSeriesWriter``: High-level time series writer
- ``DSSTimeSeriesReader``: High-level time series reader
- ``write_timeseries_to_dss``: Convenience function for single time series
- ``read_timeseries_from_dss``: Convenience function for reading time series
- ``write_collection_to_dss``: Write TimeSeriesCollection to DSS
- ``HAS_DSS_LIBRARY``: Flag indicating DSS library availability

**Template Engine Updates**

- ``iwfm_timestamp`` filter: Format datetime for IWFM files
- ``dss_pathname`` filter: Format DSS pathnames
- ``timeseries_ref`` filter: Reference time series files
- ``iwfm_array_row`` filter: Format array rows for IWFM files

Changed
~~~~~~~

- Updated ``pyiwfm.io.__init__.py`` with all new exports
- Extended ``pyiwfm.io.preprocessor`` with ``load_complete_model()`` and ``save_complete_model()``

[0.3.0] - 2025-10-01
--------------------

PEST++ Calibration Interface, Multi-Scale Viewing, and Subprocess Runner

Added
~~~~~

**Subprocess Runner** (``pyiwfm.runner``)

- ``IWFMRunner``: Run IWFM executables (Preprocessor, Simulation, Budget, ZBudget) via subprocess
- ``IWFMExecutables``: Locate and manage IWFM executable paths
- ``find_iwfm_executables()``: Auto-detect executables in PATH or specified directories
- ``RunResult``, ``PreprocessorResult``, ``SimulationResult``, ``BudgetResult``, ``ZBudgetResult``: Typed result classes

**Scenario Management** (``pyiwfm.runner.scenario``)

- ``Scenario``: Define named model scenarios with parameter overrides
- ``ScenarioManager``: Manage, run, and compare multiple scenarios
- ``ScenarioResult``: Collect and compare scenario outputs

**PEST++ Integration** (``pyiwfm.runner.pest``)

- ``PESTInterface``: Low-level PEST++ control file interface
- ``TemplateFile``, ``InstructionFile``: PEST++ template/instruction file management
- ``ObservationGroup``: Observation group definitions
- ``write_pest_control_file()``: Generate PEST++ control files (.pst)

**PEST++ Parameter Management** (``pyiwfm.runner.pest_params``)

- ``IWFMParameterType``: Enum of all IWFM parameter types (aquifer, stream, lake, rootzone, flux multipliers)
- ``ParameterTransform``: Log/none/tied/fixed transform types
- ``ParameterGroup``, ``Parameter``: Parameter definitions with bounds and transforms
- Parameterization strategies: ``ZoneParameterization``, ``MultiplierParameterization``,
  ``PilotPointParameterization``, ``DirectParameterization``, ``StreamParameterization``,
  ``RootZoneParameterization``
- ``IWFMParameterManager``: Central parameter registry and management

**PEST++ Observation Management** (``pyiwfm.runner.pest_observations``)

- ``IWFMObservationType``: Enum of observation types (head, drawdown, flow, stage, etc.)
- ``IWFMObservation``, ``IWFMObservationGroup``: Observation definitions with weights
- ``ObservationLocation``: Spatial location for observations
- ``WeightStrategy``: Equal, inverse variance, decay, and group contribution weighting
- ``DerivedObservation``: Computed observations (gradients, differences)
- ``IWFMObservationManager``: Central observation registry
- ``WellInfo``, ``GageInfo``: Monitoring point metadata

**PEST++ Template/Instruction Generation** (``pyiwfm.runner.pest_templates``, ``pest_instructions``)

- ``IWFMTemplateManager``: Generate PEST++ template files (.tpl) from IWFM input files
- ``TemplateMarker``: Define parameter marker locations in templates
- ``IWFMFileSection``: Track file sections for template generation
- ``IWFMInstructionManager``: Generate PEST++ instruction files (.ins) from IWFM output
- ``OutputFileFormat``: Define output file parsing rules
- ``IWFM_OUTPUT_FORMATS``: Predefined formats for standard IWFM outputs

**Geostatistics** (``pyiwfm.runner.pest_geostat``)

- ``VariogramType``: Spherical, exponential, Gaussian, power variogram models
- ``Variogram``: Variogram definition with nugget, sill, range parameters
- ``GeostatManager``: Manage spatial correlation structures and pilot point kriging
- ``compute_empirical_variogram()``: Compute empirical variograms from spatial data

**Main PEST++ Helper Interface** (``pyiwfm.runner.pest_helper``)

- ``IWFMPestHelper``: High-level interface coordinating all PEST++ components
- Convenience methods: ``add_zone_parameters()``, ``add_multiplier()``, ``add_pilot_points()``,
  ``add_stream_parameters()``, ``add_rootzone_parameters()``
- ``add_head_observations()``, ``add_streamflow_observations()``
- ``set_svd()``, ``set_regularization()``, ``set_model_command()``, ``set_pestpp_options()``
- ``build()``: Generate complete PEST++ setup (control file, templates, instructions, scripts)
- ``run_pestpp()``: Execute PEST++ from within Python
- ``SVDConfig``, ``RegularizationConfig``, ``RegularizationType``: Configuration classes

**Ensemble Management** (``pyiwfm.runner.pest_ensemble``)

- ``IWFMEnsembleManager``: Prior/posterior ensemble generation for pestpp-ies
- ``EnsembleStatistics``: Statistical analysis of parameter ensembles
- Latin Hypercube Sampling and geostatistical realization generation
- CSV I/O compatible with PEST++ ensemble format
- Uncertainty reduction analysis between prior and posterior ensembles

**Post-Processing** (``pyiwfm.runner.pest_postprocessor``)

- ``PestPostProcessor``: Load and analyze PEST++ output files (.rei, .sen, .iobj, .par)
- ``CalibrationResults``: Container for all calibration output data
- ``ResidualData``: Observation residual analysis with weighted statistics and group phi
- ``SensitivityData``: Parameter sensitivity rankings
- Fit statistics: RMSE, MAE, R-squared, Nash-Sutcliffe efficiency, bias, percent bias
- Parameter identifiability analysis
- Summary report generation
- Export to CSV and PEST format

**Zone Management** (``pyiwfm.core.zones``)

- ``Zone``: Dataclass for zone definition with id, name, elements, area
- ``ZoneDefinition``: Element-to-zone mapping with query and validation
- Factory methods: ``from_subregions()``, ``from_element_list()``
- Zone CRUD operations (add, remove, rename)

**Data Aggregation** (``pyiwfm.core.aggregation``)

- ``DataAggregator``: Spatial aggregation engine with 6 methods
- ``AggregationMethod``: Enum (sum, mean, area_weighted_mean, min, max, median)
- ``aggregate_to_array()``: Expand zone values back to elements for visualization
- ``aggregate_timeseries()``: Multi-timestep aggregation
- ``create_aggregator_from_grid()``: Factory from AppGrid

**Model Query API** (``pyiwfm.core.query``)

- ``ModelQueryAPI``: High-level unified query interface for multi-scale data access
- ``get_values()``: Fetch data at any spatial scale with configurable aggregation
- ``get_timeseries()``: Retrieve temporal data for locations or zones
- ``export_to_dataframe()``, ``export_to_csv()``: Data export to Pandas and CSV
- Dynamic variable and scale discovery

**Zone File I/O** (``pyiwfm.io.zones``)

- ``read_iwfm_zone_file()``, ``write_iwfm_zone_file()``: IWFM ZBudget zone format
- ``read_geojson_zones()``, ``write_geojson_zones()``: GeoJSON with geometry
- ``read_zone_file()``, ``write_zone_file()``: Auto-detecting universal I/O
- ``auto_detect_zone_file()``: Format detection by extension and content

**Interactive Web Viewer** (``pyiwfm.visualization.webapi``)

- FastAPI backend + React SPA frontend with 4 tabs (Overview, 3D Mesh, Results Map, Budgets)
- ``ModelState`` singleton for lazy model and results loading
- vtk.js-based 3D mesh rendering with layer visibility, cross-section slicing, and z-exaggeration
- deck.gl + MapLibre Results Map with head contours and hydrograph markers
- Plotly budget charts with location/column selection
- Stream network overlay on both 3D and 2D views
- Server-side coordinate reprojection via ``pyproj``
- ``pyiwfm viewer`` CLI launcher with ``--model-dir``, ``--crs``, ``--port`` options
- Auto-detection of preprocessor and simulation files
- Graceful degradation for missing components

[0.1.0] - 2024-07-01
--------------------

Initial release of pyiwfm.

Added
~~~~~

**Core Modules**

- ``pyiwfm.core.mesh``: AppGrid, Node, Element, Face classes for mesh representation
- ``pyiwfm.core.stratigraphy``: Stratigraphy class for layer structure
- ``pyiwfm.core.timeseries``: TimeSeries and TimeSeriesCollection classes

**Component Modules**

- ``pyiwfm.components.groundwater``: AppGW, AquiferParameters, Well classes
- ``pyiwfm.components.stream``: AppStream, StrmNode, StrmReach classes
- ``pyiwfm.components.lake``: AppLake, Lake, LakeElement classes
- ``pyiwfm.components.rootzone``: RootZone, LandUse, Crop classes
- ``pyiwfm.components.connectors``: StreamGWConnector, LakeGWConnector

**I/O Modules**

- ``pyiwfm.io.ascii``: ASCII file readers and writers
- ``pyiwfm.io.binary``: Fortran binary file handlers
- ``pyiwfm.io.hdf5``: HDF5 file handlers

**Mesh Generation**

- ``pyiwfm.mesh_generation.generators``: MeshGenerator ABC and MeshResult
- ``pyiwfm.mesh_generation.constraints``: BoundaryConstraint, LineConstraint, PointConstraint, RefinementZone
- ``pyiwfm.mesh_generation.triangle_wrapper``: TriangleMeshGenerator
- ``pyiwfm.mesh_generation.gmsh_wrapper``: GmshMeshGenerator

**Visualization**

- ``pyiwfm.visualization.gis_export``: GISExporter for GeoPackage, Shapefile, GeoJSON
- ``pyiwfm.visualization.vtk_export``: VTKExporter for 2D and 3D VTK files
- ``pyiwfm.visualization.plotting``: matplotlib-based visualization functions

**Comparison**

- ``pyiwfm.comparison.differ``: ModelDiffer, MeshDiff, StratigraphyDiff
- ``pyiwfm.comparison.metrics``: ComparisonMetrics, TimeSeriesComparison, SpatialComparison
- ``pyiwfm.comparison.report``: ReportGenerator with text, JSON, and HTML output

Documentation
~~~~~~~~~~~~~

- Comprehensive Sphinx documentation with PyData theme
- User guide with installation, quickstart, and detailed guides
- Tutorials for mesh generation, visualization, and model comparison
- Full API reference with examples
