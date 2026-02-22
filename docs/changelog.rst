Changelog
=========

All notable changes to pyiwfm will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[0.4.0] - 2026-XX-XX
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

**Documentation**

- Added ~30 missing module entries to API docs (``docs/api/io.rst``)
- Added Small Watershed and Unsaturated Zone to component docs
- Reorganized I/O docs into logical sections (Core, GW, Stream, Lake, RZ, Supplemental, etc.)
- Added BaseComponent and model_factory to core API docs
- Added writer_config_base to I/O API docs
- Added API routes summary to visualization docs

[0.2.0] - 2025-XX-XX
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

[0.3.0] - 2025-XX-XX
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

[0.1.0] - 2024-XX-XX
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
