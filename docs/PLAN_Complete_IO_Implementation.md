# Plan: Complete IWFM Input File Reading for pyiwfm

## Executive Summary

This plan addresses the gap between IWFM's comprehensive Fortran I/O capabilities and pyiwfm's current Python implementation. The goal is to enable pyiwfm to read **all** IWFM input files and instantiate a complete model object with full fidelity to the Fortran implementation.

---

## 1. Gap Analysis Summary

### 1.1 IWFM Fortran I/O Architecture

The IWFM Fortran codebase uses a sophisticated polymorphic I/O system:

| File Format | Extension | Fortran Class | pyiwfm Support |
|-------------|-----------|---------------|----------------|
| ASCII | .txt, .dat, .in, .out | `AsciiInFileType` | **Full** |
| Fortran Binary | .bin | `FortBinFileType` | **Partial** (mesh/strat only) |
| HDF5 | .hdf, .h5 | `HDF5FileType` | **Full** (optional) |
| HEC-DSS 7 | .dss | `DssInFileType` | **Full** (optional) |

### 1.2 Component-Level Gaps

| Component | Fortran Files Read | pyiwfm Coverage | Gap |
|-----------|-------------------|-----------------|-----|
| **Groundwater** | 15+ file types | ~60% | Missing: parametric data, advanced BC, constrained head |
| **Streams** | 12+ file types | ~50% | Missing: rating tables, roughness, evaporation |
| **Lakes** | 8+ file types | ~40% | Missing: precipitation, evaporation, detailed outflows |
| **RootZone** | 20+ file types | ~30% | Missing: land use time series, irrigation, ag demands |
| **Small Watershed** | 8+ file types | ~5% | Nearly complete gap |
| **Unsaturated Zone** | 6+ file types | ~5% | Nearly complete gap |
| **Preprocessor** | 10+ file types | ~70% | Missing: stream spec details, lake elements |

---

## 2. Detailed Implementation Plan

### Phase 1: Core Infrastructure Enhancements (Priority: Critical)

#### 1.1 Enhanced Binary File Support

**Current State:** Basic Fortran binary reading for mesh/stratigraphy only.

**Required Enhancements:**

```python
# File: pyiwfm/io/binary.py

class FortranBinaryReader:
    # ADD: Record-level navigation
    def skip_records(self, n: int) -> None
    def get_position(self) -> int
    def seek_to_position(self, pos: int) -> None

    # ADD: Complex record types
    def read_mixed_record(self, dtype_spec: list) -> tuple
    def read_character_record(self, length: int) -> str

    # ADD: Preprocessor binary support
    def read_preprocessor_binary(self, filepath: Path) -> PreprocessorBinaryData
```

**New File:** `pyiwfm/io/preprocessor_binary.py`
```python
@dataclass
class PreprocessorBinaryData:
    """Complete preprocessor binary output structure."""
    n_nodes: int
    n_elements: int
    n_layers: int
    n_subregions: int
    node_coords: NDArray[np.float64]  # (n_nodes, 2)
    element_vertices: NDArray[np.int32]  # (n_elements, 4)
    element_subregions: NDArray[np.int32]
    stratigraphy: NDArray[np.float64]  # (n_nodes, n_layers, 3)
    stream_nodes: NDArray  # If streams defined
    lake_elements: NDArray  # If lakes defined
    # ... additional preprocessor output data

class PreprocessorBinaryReader:
    def read(self, filepath: Path) -> PreprocessorBinaryData
```

#### 1.2 Time Series File Infrastructure

**Current State:** Basic ASCII time series support.

**Required Enhancements:**

```python
# File: pyiwfm/io/timeseries.py (new unified module)

class TimeSeriesFileType(Enum):
    ASCII = "ascii"
    DSS = "dss"
    HDF5 = "hdf5"

@dataclass
class TimeSeriesFileConfig:
    """Configuration for time series file reading."""
    filepath: Path
    file_type: TimeSeriesFileType
    n_columns: int
    column_ids: list[int]  # Entity IDs for each column
    time_unit: str
    data_unit: str
    conversion_factor: float
    is_rate_data: bool  # Whether to normalize by timestep
    recycling_interval: int  # 0=no recycle, else recycle period

class UnifiedTimeSeriesReader:
    """Unified reader for all time series formats."""

    def __init__(self, config: TimeSeriesFileConfig):
        self._config = config
        self._reader = self._create_reader()

    def _create_reader(self):
        if self._config.file_type == TimeSeriesFileType.ASCII:
            return AsciiTimeSeriesReader(...)
        elif self._config.file_type == TimeSeriesFileType.DSS:
            return DSSTimeSeriesReader(...)
        elif self._config.file_type == TimeSeriesFileType.HDF5:
            return HDF5TimeSeriesReader(...)

    def read_timestep(self, timestamp: datetime) -> NDArray
    def read_range(self, start: datetime, end: datetime) -> TimeSeriesCollection
    def read_all(self) -> TimeSeriesCollection
```

---

### Phase 2: Groundwater Component Complete Implementation

#### 2.1 Main File Reader Enhancement

**File:** `pyiwfm/io/groundwater.py`

```python
@dataclass
class GWMainFileConfig:
    """Complete GW main file configuration."""
    version: str

    # Sub-file paths (all resolved to absolute)
    bc_file: Path | None
    tile_drain_file: Path | None
    pumping_file: Path | None
    subsidence_file: Path | None
    overwrite_file: Path | None  # Parameter overwrite

    # Output configuration
    velocity_output_file: Path | None
    vertical_flow_output_file: Path | None
    head_all_output_file: Path | None
    head_tecplot_file: Path | None
    velocity_tecplot_file: Path | None
    budget_output_file: Path | None
    zbudget_output_file: Path | None
    final_heads_file: Path | None

    # Conversion factors
    head_output_factor: float
    head_output_unit: str
    volume_output_factor: float
    volume_output_unit: str
    velocity_output_factor: float
    velocity_output_unit: str

    # Hydrograph output
    n_hydrographs: int
    coord_factor: float
    hydrograph_output_file: Path | None
    hydrograph_locations: list[HydrographLocation]

    # Debug flag
    debug_print: bool
```

#### 2.2 New Sub-file Readers

**Boundary Conditions Reader:**
```python
# File: pyiwfm/io/gw_boundary.py

@dataclass
class SpecifiedHeadBC:
    node_id: int
    layer: int
    head_column: int  # Column in time series file

@dataclass
class SpecifiedFlowBC:
    node_id: int
    layer: int
    flow_column: int
    flow_type: str  # 'inflow' or 'outflow'

@dataclass
class GeneralHeadBC:
    node_id: int
    layer: int
    conductance: float
    head_column: int

@dataclass
class ConstrainedGeneralHeadBC:
    node_id: int
    layer: int
    conductance: float
    max_head: float
    head_column: int

@dataclass
class GWBoundaryConfig:
    """Complete GW boundary conditions."""
    version: str

    # Specified head BCs
    n_specified_head: int
    specified_head_file: Path | None
    specified_head_bcs: list[SpecifiedHeadBC]

    # Specified flow BCs
    n_specified_flow: int
    specified_flow_file: Path | None
    specified_flow_bcs: list[SpecifiedFlowBC]

    # General head BCs
    n_general_head: int
    general_head_file: Path | None
    general_head_bcs: list[GeneralHeadBC]

    # Constrained general head BCs
    n_constrained_gh: int
    constrained_gh_file: Path | None
    constrained_gh_bcs: list[ConstrainedGeneralHeadBC]

class GWBoundaryReader:
    def read(self, filepath: Path, base_dir: Path) -> GWBoundaryConfig
```

**Pumping Reader (Complete):**
```python
# File: pyiwfm/io/gw_pumping.py

@dataclass
class WellSpec:
    """Well specification from pumping file."""
    id: int
    x: float
    y: float
    element: int
    well_id_col: int  # Column for well ID in pumping rates
    n_layers: int
    layer_factors: list[float]  # Distribution across layers

@dataclass
class ElementPumpingSpec:
    """Element-based pumping specification."""
    element_id: int
    pumping_col: int
    n_layers: int
    layer_factors: list[float]

@dataclass
class PumpingConfig:
    version: str

    # Well pumping
    n_wells: int
    well_specs: list[WellSpec]
    well_pumping_file: Path | None

    # Element pumping
    n_elem_pumping: int
    elem_pumping_specs: list[ElementPumpingSpec]
    elem_pumping_file: Path | None

    # Conversion factors
    pumping_factor: float
    pumping_unit: str

class PumpingReader:
    def read(self, filepath: Path, base_dir: Path) -> PumpingConfig
```

**Subsidence Reader:**
```python
# File: pyiwfm/io/gw_subsidence.py

@dataclass
class SubsidenceParams:
    """Subsidence parameters for an element-layer."""
    element_id: int
    layer: int
    interbed_thickness_elastic: float
    interbed_thickness_inelastic: float
    elastic_specific_storage: float
    inelastic_specific_storage: float
    initial_compaction: float
    preconsolidation_head: float

@dataclass
class SubsidenceConfig:
    version: str
    n_subsidence_nodes: int
    subsidence_params: list[SubsidenceParams]

    # Output
    subsidence_output_file: Path | None
    tecplot_output_file: Path | None

class SubsidenceReader:
    def read(self, filepath: Path, base_dir: Path) -> SubsidenceConfig
```

**Tile Drain Reader (Complete):**
```python
# File: pyiwfm/io/gw_tiledrain.py

@dataclass
class TileDrainSpec:
    id: int
    gw_node: int
    drain_node: int  # Stream node receiving drain flow
    layer: int
    conductance: float
    drain_depth: float  # Depth below ground surface

@dataclass
class SubIrrigationSpec:
    id: int
    gw_node: int
    source_node: int  # Stream node providing water
    layer: int
    conductance: float

@dataclass
class TileDrainConfig:
    version: str

    # Tile drains
    n_tile_drains: int
    tile_drains: list[TileDrainSpec]

    # Subsurface irrigation
    n_sub_irrigation: int
    sub_irrigation: list[SubIrrigationSpec]

class TileDrainReader:
    def read(self, filepath: Path, base_dir: Path) -> TileDrainConfig
```

---

### Phase 3: Stream Component Complete Implementation

#### 3.1 Stream Main File Reader Enhancement

```python
# File: pyiwfm/io/streams.py

@dataclass
class StreamMainFileConfig:
    version: str

    # Sub-file paths
    inflow_file: Path | None
    diversion_spec_file: Path | None
    bypass_spec_file: Path | None
    diversion_ts_file: Path | None
    bypass_ts_file: Path | None

    # Output files
    reach_budget_file: Path | None
    diversion_budget_file: Path | None
    hydrograph_output_file: Path | None

    # Hydrograph specs
    n_hydrographs: int
    hydrograph_output_type: int  # 0=flow, 1=stage, 2=both
    hydrograph_specs: list[tuple[int, str]]  # (node_id, name)

    # Conversion factors
    flow_output_factor: float
    flow_output_unit: str
```

#### 3.2 Stream Spec Reader (Preprocessor)

```python
# File: pyiwfm/io/stream_spec.py

@dataclass
class StreamNodeSpec:
    """Stream node specification from preprocessor."""
    stream_node_id: int
    gw_node_id: int
    bottom_elevation: float

    # Rating table data
    rating_stages: NDArray[np.float64]
    rating_flows: NDArray[np.float64]

    # Wetted perimeter vs flow
    wetted_perimeter_coeffs: tuple[float, float]  # A, B for WP = A * Q^B

@dataclass
class StreamReachSpec:
    id: int
    name: str
    n_nodes: int
    outflow_destination: int  # 0=boundary, -n=lake n, +n=stream node n
    upstream_node: int
    downstream_node: int
    nodes: list[StreamNodeSpec]

@dataclass
class StreamSpecConfig:
    version: str
    n_reaches: int
    n_rating_points: int
    reaches: list[StreamReachSpec]

    # Stream-GW interaction
    n_gw_nodes: int
    stream_to_gw_map: dict[int, int]  # stream_node -> gw_node

class StreamSpecReader:
    def read(self, filepath: Path) -> StreamSpecConfig
```

#### 3.3 Diversion and Bypass Readers

```python
# File: pyiwfm/io/stream_diversion.py

@dataclass
class DiversionSpec:
    id: int
    name: str
    source_reach: int
    source_node: int
    destination_type: str  # 'element', 'subregion', 'outside'
    destination_id: int
    delivery_type: str  # 'ag', 'urban', 'both'

    # Allocation
    priority: int
    max_rate: float

    # Recovery
    recoverable_fraction: float
    recovery_destination: int

@dataclass
class BypassSpec:
    id: int
    name: str
    source_reach: int
    source_node: int
    destination_type: str  # 'stream', 'lake', 'outside'
    destination_id: int

    # Ratings
    export_rating_stages: NDArray
    export_rating_flows: NDArray

class DiversionSpecReader:
    def read(self, filepath: Path, base_dir: Path) -> list[DiversionSpec]

class BypassSpecReader:
    def read(self, filepath: Path, base_dir: Path) -> list[BypassSpec]
```

---

### Phase 4: Lake Component Complete Implementation

```python
# File: pyiwfm/io/lakes.py

@dataclass
class LakeMainFileConfig:
    version: str

    # Lake definitions
    n_lakes: int
    lakes: list[LakeDefinition]

    # Sub-files
    precip_file: Path | None
    evap_file: Path | None
    inflow_file: Path | None

    # Output
    budget_file: Path | None
    hydrograph_file: Path | None

@dataclass
class LakeDefinition:
    id: int
    name: str
    n_elements: int
    element_ids: list[int]
    element_fractions: list[float]

    # Rating curve
    rating_elevations: NDArray
    rating_areas: NDArray
    rating_volumes: NDArray

    # Outflow
    outflow_destination_type: str  # 'stream', 'gw', 'outside'
    outflow_destination_id: int
    outflow_rating_elevations: NDArray
    outflow_rating_flows: NDArray

    # Constraints
    max_elevation: float
    initial_storage: float

@dataclass
class LakeElementSpec:
    """Lake element from preprocessor."""
    element_id: int
    lake_id: int
    fraction: float
    gw_node_id: int  # For lake-GW interaction

class LakeSpecReader:
    """Read lake specification from preprocessor."""
    def read(self, filepath: Path) -> list[LakeElementSpec]
```

---

### Phase 5: RootZone Component Complete Implementation

```python
# File: pyiwfm/io/rootzone.py

@dataclass
class RootZoneMainFileConfig:
    version: str

    # Solver parameters
    convergence_tolerance: float
    max_iterations: int

    # Flags
    gw_uptake_method: int  # 0=off, 1=on

    # Sub-files by land use type
    nonponded_ag_file: Path | None
    ponded_ag_file: Path | None
    urban_file: Path | None
    native_riparian_file: Path | None

    # Time series files
    precip_file: Path | None
    et_file: Path | None

    # Return flow and reuse
    return_flow_file: Path | None
    reuse_file: Path | None

    # Irrigation
    irrigation_period_file: Path | None

# Non-ponded agricultural crops
@dataclass
class NonPondedCropConfig:
    version: str
    n_crops: int
    crops: list[CropType]

    # Per-element data files
    crop_area_file: Path | None  # Time series
    initial_conditions_file: Path | None
    root_depth_file: Path | None

    # Parameters
    min_soil_moisture_file: Path | None
    target_soil_moisture_file: Path | None

    # Water supply
    ag_supply_req_file: Path | None
    irrigation_file: Path | None

class NonPondedCropReader:
    def read(self, filepath: Path, base_dir: Path) -> NonPondedCropConfig

# Ponded crops (rice)
@dataclass
class PondedCropConfig:
    version: str
    n_crops: int
    crops: list[CropType]

    # Ponding parameters
    ponding_depth_file: Path | None
    flooding_period_file: Path | None
    decomposition_file: Path | None

class PondedCropReader:
    def read(self, filepath: Path, base_dir: Path) -> PondedCropConfig

# Urban land use
@dataclass
class UrbanConfig:
    version: str

    # Land use areas
    urban_area_file: Path | None  # Time series
    pervious_fraction_file: Path | None

    # Water use
    indoor_demand_file: Path | None
    outdoor_demand_file: Path | None

    # Return flows
    indoor_return_fraction: float
    outdoor_return_fraction: float

class UrbanReader:
    def read(self, filepath: Path, base_dir: Path) -> UrbanConfig

# Native/Riparian vegetation
@dataclass
class NativeRiparianConfig:
    version: str

    # Areas
    native_area_file: Path | None
    riparian_area_file: Path | None

    # Parameters per vegetation type
    root_depths: dict[str, float]
    et_fractions: dict[str, float]

class NativeRiparianReader:
    def read(self, filepath: Path, base_dir: Path) -> NativeRiparianConfig
```

---

### Phase 6: Small Watershed Component (New)

```python
# File: pyiwfm/io/small_watershed.py (NEW)

@dataclass
class SmallWatershedMainConfig:
    version: str
    n_watersheds: int
    watersheds: list[SmallWatershed]

@dataclass
class SmallWatershed:
    id: int
    name: str

    # Elements in watershed
    n_elements: int
    element_ids: list[int]

    # Outlet
    outlet_type: str  # 'stream', 'lake', 'gw'
    outlet_id: int

    # Baseflow parameters
    baseflow_method: int
    baseflow_gw_nodes: list[int]
    baseflow_layers: list[int]
    baseflow_fractions: list[float]

    # Percolation parameters
    percolation_nodes: list[int]
    percolation_layers: list[int]
    percolation_fractions: list[float]

class SmallWatershedReader:
    def read(self, filepath: Path, base_dir: Path) -> SmallWatershedMainConfig
```

---

### Phase 7: Unsaturated Zone Component (New)

```python
# File: pyiwfm/io/unsatzone.py (NEW)

@dataclass
class UnsatZoneMainConfig:
    version: str

    # Simulation control
    n_unsat_layers: int
    solution_method: int

    # Per-element parameters
    max_unsat_depth_file: Path | None
    soil_properties_file: Path | None
    initial_moisture_file: Path | None

@dataclass
class UnsatZoneSoilParams:
    element_id: int
    layer: int
    saturated_moisture: float
    residual_moisture: float
    van_genuchten_alpha: float
    van_genuchten_n: float
    saturated_conductivity: float

class UnsatZoneReader:
    def read(self, filepath: Path, base_dir: Path) -> UnsatZoneMainConfig

class UnsatZoneSoilReader:
    def read(self, filepath: Path) -> list[UnsatZoneSoilParams]
```

---

### Phase 8: Complete Model Assembly

```python
# File: pyiwfm/io/model_loader.py (NEW)

@dataclass
class CompleteModelConfig:
    """Complete IWFM model configuration from all files."""

    # Preprocessor data
    preprocessor: PreProcessorConfig
    preprocessor_binary: PreprocessorBinaryData | None

    # Simulation control
    simulation: SimulationConfig

    # Component configurations (main files parsed)
    groundwater: GWMainFileConfig | None
    gw_boundary: GWBoundaryConfig | None
    gw_pumping: PumpingConfig | None
    gw_subsidence: SubsidenceConfig | None
    gw_tiledrain: TileDrainConfig | None

    streams: StreamMainFileConfig | None
    stream_spec: StreamSpecConfig | None
    diversions: list[DiversionSpec]
    bypasses: list[BypassSpec]

    lakes: LakeMainFileConfig | None
    lake_spec: list[LakeElementSpec]

    rootzone: RootZoneMainFileConfig | None
    nonponded_crops: NonPondedCropConfig | None
    ponded_crops: PondedCropConfig | None
    urban: UrbanConfig | None
    native_riparian: NativeRiparianConfig | None

    small_watersheds: SmallWatershedMainConfig | None
    unsat_zone: UnsatZoneMainConfig | None

class CompleteModelLoader:
    """Load complete IWFM model from all input files."""

    def __init__(self, simulation_file: Path, preprocessor_file: Path):
        self.sim_file = simulation_file
        self.pp_file = preprocessor_file
        self.base_dir = simulation_file.parent
        self.pp_dir = preprocessor_file.parent

    def load(self) -> CompleteModelConfig:
        """Load all configuration files."""
        config = CompleteModelConfig(...)

        # 1. Load preprocessor (mesh, stratigraphy, connectivity)
        config.preprocessor = self._load_preprocessor()
        config.preprocessor_binary = self._load_preprocessor_binary()

        # 2. Load simulation control
        config.simulation = self._load_simulation()

        # 3. Load components based on simulation config
        if config.simulation.groundwater_file:
            config.groundwater = self._load_gw_main()
            config.gw_boundary = self._load_gw_boundary()
            config.gw_pumping = self._load_gw_pumping()
            config.gw_subsidence = self._load_gw_subsidence()
            config.gw_tiledrain = self._load_gw_tiledrain()

        if config.simulation.streams_file:
            config.streams = self._load_stream_main()
            config.stream_spec = self._load_stream_spec()
            config.diversions = self._load_diversions()
            config.bypasses = self._load_bypasses()

        # ... similar for other components

        return config

    def load_model(self) -> IWFMModel:
        """Load complete model object."""
        config = self.load()
        return self._build_model(config)

    def _build_model(self, config: CompleteModelConfig) -> IWFMModel:
        """Build IWFMModel from complete configuration."""
        # Create mesh from preprocessor
        mesh = self._build_mesh(config.preprocessor, config.preprocessor_binary)

        # Create stratigraphy
        strat = self._build_stratigraphy(config.preprocessor)

        # Create model
        model = IWFMModel(
            name=config.simulation.model_name,
            mesh=mesh,
            stratigraphy=strat,
        )

        # Populate groundwater component
        if config.groundwater:
            model.groundwater = self._build_gw_component(config)

        # Populate streams component
        if config.streams:
            model.streams = self._build_stream_component(config)

        # ... continue for all components

        return model
```

---

## 3. Implementation Priority

### Priority 1 (Critical - Weeks 1-4)
1. Binary preprocessor file reader
2. Complete GW main file reader (all output options)
3. GW boundary conditions reader (all 4 types)
4. Stream spec reader (from preprocessor)
5. Unified time series infrastructure

### Priority 2 (High - Weeks 5-8)
1. GW pumping reader (wells + element pumping)
2. GW subsidence reader
3. GW tile drain reader
4. Stream diversion/bypass spec readers
5. Lake main file reader

### Priority 3 (Medium - Weeks 9-12)
1. RootZone main file reader
2. Non-ponded crop reader
3. Ponded crop reader
4. Urban land use reader
5. Native/riparian reader

### Priority 4 (Lower - Weeks 13-16)
1. Small watershed reader
2. Unsaturated zone reader
3. Complete model loader
4. Validation against Fortran behavior

---

## 4. File Format Reference

### 4.1 IWFM Comment Handling

All ASCII files use Fortran-style comments:
- `C` or `c` in column 1 = comment line
- `*` in column 1 = comment line
- Blank lines = skip
- Inline: `value / description` or `value # description`

```python
def _is_comment_line(line: str) -> bool:
    if not line.strip():
        return True
    return line[0] in ('C', 'c', '*')

def _parse_value_line(line: str) -> tuple[str, str]:
    """Parse 'value / description' or 'value # description'."""
    import re
    m = re.search(r'\s+[#/]', line)
    if m:
        return line[:m.start()].strip(), line[m.end():].strip()
    return line.strip(), ""
```

### 4.2 Version Detection

All component main files start with version:
```
#4.0
C Comments...
```

```python
def _read_version(f: TextIO) -> str:
    for line in f:
        stripped = line.strip()
        if stripped.startswith('#'):
            return stripped[1:].strip()
        if line[0] in ('C', 'c', '*'):
            continue
        break
    return ""
```

### 4.3 Time Series File Format

IWFM time series format:
```
NDAT  NFAC  NPTMAX  IREPEAT
FACTOR  DSSFL  UNITTS  UNITFLT
DSSPATH (if DSSFL != '')
10/01/1974_24:00  value1  value2  ...
10/01/1974_24:00  value1  value2  ...
```

---

## 5. Testing Strategy

### 5.1 Unit Tests

For each new reader:
```python
def test_gw_boundary_reader_specified_head():
    """Test reading specified head boundary conditions."""
    content = """#4.0
C Boundary conditions file
5  / NSPECHEAD
...
"""
    with tempfile.NamedTemporaryFile(...) as f:
        f.write(content)
        reader = GWBoundaryReader()
        config = reader.read(f.name)
        assert config.n_specified_head == 5
```

### 5.2 Integration Tests

Test with real C2VSimFG files:
```python
def test_complete_model_loader_c2vsimfg():
    """Test loading complete C2VSimFG model."""
    loader = CompleteModelLoader(
        simulation_file=C2VSIMFG_SIM_FILE,
        preprocessor_file=C2VSIMFG_PP_FILE,
    )
    model = loader.load_model()

    assert model.mesh.n_nodes == 30179
    assert model.mesh.n_elements == 32537
    assert model.groundwater.n_hydrograph_locations == 54544
    assert model.streams.n_reaches == 110
```

### 5.3 Round-Trip Tests

Write and re-read files:
```python
def test_gw_roundtrip():
    """Test write then read preserves data."""
    original = create_test_gw_config()

    with tempfile.TemporaryDirectory() as tmpdir:
        writer = GWWriter(tmpdir)
        writer.write(original)

        reader = GWReader()
        loaded = reader.read(tmpdir / "groundwater.dat")

        assert loaded == original
```

---

## 6. Deliverables

1. **Enhanced binary reader** with preprocessor support
2. **Unified time series infrastructure** for all formats
3. **Complete GW readers** (main, boundary, pumping, subsidence, tiledrain)
4. **Complete Stream readers** (main, spec, diversions, bypasses)
5. **Complete Lake readers** (main, spec, elements)
6. **Complete RootZone readers** (main, crops, urban, native)
7. **New Small Watershed reader**
8. **New Unsaturated Zone reader**
9. **CompleteModelLoader** class for full model instantiation
10. **Comprehensive test suite** with C2VSimFG validation

---

## 7. Dependencies

### Required (Already Available)
- numpy
- pathlib
- dataclasses

### Optional (For Full Functionality)
- h5py (HDF5 support)
- HEC-DSS library (DSS support)
- pandas (DataFrame export)

---

## 8. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Version incompatibility | Medium | High | Test with multiple IWFM versions |
| Binary format changes | Low | High | Version detection and handling |
| Missing edge cases | Medium | Medium | Comprehensive test suite |
| Performance issues | Low | Medium | Lazy loading, caching |
| DSS library availability | Medium | Low | Graceful fallback |

---

## 9. Success Metrics

1. **100% coverage** of IWFM input file types
2. **Successful loading** of C2VSimFG model
3. **Successful loading** of sample model
4. **Round-trip fidelity** for all file types
5. **Performance** comparable to Fortran (within 10x)
6. **All existing tests** continue to pass
