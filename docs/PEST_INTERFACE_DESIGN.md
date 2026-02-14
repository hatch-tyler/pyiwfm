# Highly Parameterized PEST++ Interface for IWFM

## Overview

This document outlines a design for a pyemu-inspired highly parameterized PEST++ interface specifically tailored for IWFM models. The goal is to provide flexible, powerful parameterization and observation handling for:

- **Calibration** (pestpp-glm, pestpp-ies)
- **Uncertainty analysis** (pestpp-ies, linear methods)
- **Sensitivity analysis** (pestpp-sen)
- **Optimization** (pestpp-opt, pestpp-sqp)

## Design Principles

1. **IWFM-aware**: Understands IWFM file formats, model structure, and common parameterization patterns
2. **Flexible parameterization**: Support pilot points, zones, multipliers, and direct parameters
3. **Comprehensive observations**: Time series, spatial data, derived quantities (budgets, drawdown)
4. **Geostatistical support**: Variograms, covariance matrices, regularization
5. **Ensemble-ready**: Native support for pestpp-ies workflows
6. **Integration with pyiwfm**: Leverage existing I/O and mesh capabilities

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        IWFMPestHelper                               │
│  (Main entry point - coordinates all components)                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │ ParameterManager │  │ ObservationMgr   │  │ PriorInfoManager │  │
│  │                  │  │                  │  │                  │  │
│  │ - Pilot points   │  │ - Head obs       │  │ - Regularization │  │
│  │ - Zone params    │  │ - Flow obs       │  │ - Soft knowledge │  │
│  │ - Multipliers    │  │ - Budget obs     │  │ - Constraints    │  │
│  │ - Direct params  │  │ - Derived obs    │  │                  │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘  │
│                                                                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │ GeostatManager   │  │ TemplateManager  │  │ InstructionMgr   │  │
│  │                  │  │                  │  │                  │  │
│  │ - Variograms     │  │ - Generate .tpl  │  │ - Generate .ins  │  │
│  │ - Covariance     │  │ - IWFM-aware     │  │ - Parse outputs  │  │
│  │ - Kriging        │  │ - Multipliers    │  │ - Hydrographs    │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘  │
│                                                                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │ ControlFile      │  │ EnsembleManager  │  │ PostProcessor    │  │
│  │                  │  │                  │  │                  │  │
│  │ - Write .pst     │  │ - Prior ensemble │  │ - Analyze output │  │
│  │ - PEST++ options │  │ - Posterior      │  │ - Uncertainty    │  │
│  │ - SVD settings   │  │ - Realizations   │  │ - Identifiability│  │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Module 1: Parameter Management

### 1.1 Parameter Types for IWFM

```python
class IWFMParameterType(Enum):
    """Types of parameters in IWFM models."""

    # Aquifer parameters (by layer, zone, or pilot point)
    HORIZONTAL_K = "hk"           # Horizontal hydraulic conductivity
    VERTICAL_K = "vk"             # Vertical hydraulic conductivity
    SPECIFIC_STORAGE = "ss"       # Specific storage
    SPECIFIC_YIELD = "sy"         # Specific yield

    # Stream parameters
    STREAMBED_K = "strk"          # Streambed hydraulic conductivity
    STREAMBED_THICKNESS = "strt"  # Streambed thickness
    STREAM_WIDTH = "strw"         # Stream width factor

    # Lake parameters
    LAKEBED_K = "lakk"            # Lakebed hydraulic conductivity

    # Root zone parameters
    CROP_COEFFICIENT = "kc"       # Crop coefficient multiplier
    IRRIGATION_EFFICIENCY = "ie"  # Irrigation efficiency
    ROOT_DEPTH = "rd"             # Root depth factor
    SOIL_POROSITY = "por"         # Soil porosity
    FIELD_CAPACITY = "fc"         # Field capacity
    WILTING_POINT = "wp"          # Wilting point

    # Flux multipliers
    PUMPING_MULT = "pump"         # Pumping rate multiplier
    RECHARGE_MULT = "rech"        # Recharge rate multiplier
    DIVERSION_MULT = "div"        # Diversion rate multiplier
    PRECIP_MULT = "ppt"           # Precipitation multiplier
    ET_MULT = "et"                # ET multiplier

    # Boundary conditions
    GHB_CONDUCTANCE = "ghbc"      # General head boundary conductance
    GHB_HEAD = "ghbh"             # General head boundary head
```

### 1.2 Parameterization Strategies

```python
@dataclass
class ParameterizationStrategy:
    """Base class for parameterization strategies."""
    param_type: IWFMParameterType
    transform: str = "log"  # none, log, fixed


@dataclass
class ZoneParameterization(ParameterizationStrategy):
    """Zone-based parameters (one value per zone/subregion)."""
    zones: list[int]  # Zone IDs
    initial_values: dict[int, float]  # Zone ID -> value
    bounds: tuple[float, float]


@dataclass
class PilotPointParameterization(ParameterizationStrategy):
    """Spatially distributed parameters via pilot points."""
    points: list[tuple[float, float]]  # (x, y) coordinates
    layer: int = 1
    initial_values: NDArray | float = 1.0
    bounds: tuple[float, float] = (0.01, 100.0)
    variogram: "Variogram | None" = None
    kriging_type: str = "ordinary"  # ordinary, simple, universal
    search_radius: float | None = None


@dataclass
class MultiplierParameterization(ParameterizationStrategy):
    """Multiplier parameters applied to existing values."""
    target_file: Path  # File containing base values
    spatial_extent: str = "global"  # global, zone, element
    temporal_extent: str = "constant"  # constant, seasonal, monthly
    initial_value: float = 1.0
    bounds: tuple[float, float] = (0.5, 2.0)


@dataclass
class GridParameterization(ParameterizationStrategy):
    """Element-level parameters (one per element)."""
    layer: int = 1
    initial_values: NDArray | None = None  # n_elements values
    bounds: tuple[float, float] = (0.01, 100.0)
    regularization: str = "preferred_homogeneity"
```

### 1.3 Parameter Manager

```python
class IWFMParameterManager:
    """Manages all parameters for an IWFM PEST++ setup."""

    def __init__(self, model: IWFMModel):
        self.model = model
        self.parameters: dict[str, Parameter] = {}
        self.par_groups: dict[str, ParameterGroup] = {}
        self.parameterizations: list[ParameterizationStrategy] = []

    # --- Zone-based parameters ---

    def add_zone_parameters(
        self,
        param_type: IWFMParameterType,
        zones: list[int] | str = "all",  # "all" uses subregions
        layer: int | None = None,
        initial_values: float | dict[int, float] = 1.0,
        bounds: tuple[float, float] = (0.01, 100.0),
        transform: str = "log",
    ) -> list[Parameter]:
        """Add zone-based parameters.

        Creates one parameter per zone for the specified property.
        Zones can be subregions or custom zone definitions.

        Examples
        --------
        >>> pm.add_zone_parameters(
        ...     IWFMParameterType.HORIZONTAL_K,
        ...     zones="all",  # Use subregions
        ...     layer=1,
        ...     bounds=(0.1, 1000.0),
        ... )
        """

    # --- Pilot point parameters ---

    def add_pilot_points(
        self,
        param_type: IWFMParameterType,
        spacing: float | None = None,
        points: list[tuple[float, float]] | None = None,
        layer: int = 1,
        bounds: tuple[float, float] = (0.01, 100.0),
        variogram: "Variogram | None" = None,
        prefix: str | None = None,
    ) -> list[Parameter]:
        """Add pilot point parameters.

        Pilot points are interpolated to model nodes/elements
        using kriging. This enables highly parameterized spatial
        heterogeneity while maintaining geological plausibility
        through geostatistical regularization.

        Parameters
        ----------
        spacing : float | None
            Regular grid spacing. If None, must provide points.
        points : list[tuple[float, float]] | None
            Explicit pilot point locations. If None, generates
            regular grid with specified spacing.
        layer : int
            Model layer for these parameters.
        variogram : Variogram | None
            Variogram for kriging. If None, uses default exponential.

        Examples
        --------
        >>> # Regular grid of pilot points
        >>> pm.add_pilot_points(
        ...     IWFMParameterType.HORIZONTAL_K,
        ...     spacing=5000.0,  # 5km spacing
        ...     layer=1,
        ...     variogram=Variogram("exponential", a=10000, sill=1.0),
        ... )

        >>> # Custom pilot point locations
        >>> pm.add_pilot_points(
        ...     IWFMParameterType.SPECIFIC_YIELD,
        ...     points=[(x1, y1), (x2, y2), ...],
        ...     layer=1,
        ... )
        """

    def generate_pilot_point_grid(
        self,
        spacing: float,
        layer: int = 1,
        buffer: float = 0.0,
        exclude_inactive: bool = True,
    ) -> list[tuple[float, float]]:
        """Generate regular grid of pilot points within model domain."""

    # --- Multiplier parameters ---

    def add_multiplier_parameters(
        self,
        param_type: IWFMParameterType,
        spatial: str = "global",  # global, zone, element
        temporal: str = "constant",  # constant, seasonal, monthly, stress_period
        zones: list[int] | None = None,
        bounds: tuple[float, float] = (0.5, 2.0),
    ) -> list[Parameter]:
        """Add multiplier parameters.

        Multipliers adjust existing model values rather than
        replacing them directly. Useful for:
        - Pumping rate adjustment
        - Recharge scaling
        - Crop coefficient calibration

        Examples
        --------
        >>> # Global pumping multiplier
        >>> pm.add_multiplier_parameters(
        ...     IWFMParameterType.PUMPING_MULT,
        ...     spatial="global",
        ...     bounds=(0.8, 1.2),
        ... )

        >>> # Zone-specific recharge multipliers
        >>> pm.add_multiplier_parameters(
        ...     IWFMParameterType.RECHARGE_MULT,
        ...     spatial="zone",
        ...     zones=[1, 2, 3],
        ...     bounds=(0.5, 2.0),
        ... )

        >>> # Seasonal ET multipliers
        >>> pm.add_multiplier_parameters(
        ...     IWFMParameterType.ET_MULT,
        ...     temporal="seasonal",  # Creates 4 parameters (one per season)
        ...     bounds=(0.8, 1.2),
        ... )
        """

    # --- Streambed parameters ---

    def add_stream_parameters(
        self,
        param_type: IWFMParameterType,
        reaches: list[int] | str = "all",
        bounds: tuple[float, float] = (0.001, 10.0),
        transform: str = "log",
    ) -> list[Parameter]:
        """Add stream-related parameters.

        Examples
        --------
        >>> # Streambed K by reach
        >>> pm.add_stream_parameters(
        ...     IWFMParameterType.STREAMBED_K,
        ...     reaches="all",
        ...     bounds=(0.01, 100.0),
        ... )
        """

    # --- Root zone parameters ---

    def add_rootzone_parameters(
        self,
        param_type: IWFMParameterType,
        land_use_types: list[str] | str = "all",
        bounds: tuple[float, float] = (0.5, 1.5),
    ) -> list[Parameter]:
        """Add root zone parameters.

        Examples
        --------
        >>> # Crop coefficients by land use type
        >>> pm.add_rootzone_parameters(
        ...     IWFMParameterType.CROP_COEFFICIENT,
        ...     land_use_types=["corn", "alfalfa", "orchard"],
        ...     bounds=(0.8, 1.2),
        ... )
        """

    # --- Parameter utilities ---

    def setup_tied_parameters(
        self,
        parent: str,
        children: list[str],
        ratio: float | list[float] = 1.0,
    ) -> None:
        """Set up tied parameters (children follow parent)."""

    def fix_parameter(self, name: str) -> None:
        """Fix a parameter (no adjustment during calibration)."""

    def get_parameter_dataframe(self) -> pd.DataFrame:
        """Get all parameters as a DataFrame."""

    def write_parameter_values(self, filepath: Path) -> None:
        """Write current parameter values to file."""
```

---

## Module 2: Observation Management

### 2.1 Observation Types for IWFM

```python
class IWFMObservationType(Enum):
    """Types of observations in IWFM models."""

    # Groundwater observations
    HEAD = "head"                    # Groundwater head
    DRAWDOWN = "drawdown"            # Head change from initial
    HEAD_DIFFERENCE = "hdiff"        # Head difference between wells
    VERTICAL_GRADIENT = "vgrad"      # Vertical head gradient

    # Stream observations
    STREAM_FLOW = "flow"             # Stream discharge
    STREAM_STAGE = "stage"           # Stream water level
    STREAM_GAIN_LOSS = "sgl"         # Stream gain/loss

    # Lake observations
    LAKE_LEVEL = "lake"              # Lake water surface elevation
    LAKE_STORAGE = "lsto"            # Lake storage volume

    # Budget observations
    GW_BUDGET = "gwbud"              # GW budget component
    STREAM_BUDGET = "strbud"         # Stream budget component
    ROOTZONE_BUDGET = "rzbud"        # Root zone budget component

    # Land subsidence
    SUBSIDENCE = "sub"               # Land surface subsidence
    COMPACTION = "comp"              # Layer compaction
```

### 2.2 Observation Manager

```python
class IWFMObservationManager:
    """Manages all observations for an IWFM PEST++ setup."""

    def __init__(self, model: IWFMModel):
        self.model = model
        self.observations: dict[str, Observation] = {}
        self.obs_groups: dict[str, ObservationGroup] = {}

    # --- Head observations ---

    def add_head_observations(
        self,
        wells: pd.DataFrame | Path,
        observed_data: pd.DataFrame | Path,
        layers: int | list[int] | str = "auto",
        weight_strategy: str = "equal",  # equal, inverse_variance, decay
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        frequency: str | None = None,  # resample frequency
        group_by: str = "well",  # well, layer, time
    ) -> list[Observation]:
        """Add groundwater head observations.

        Parameters
        ----------
        wells : pd.DataFrame | Path
            Well information with columns: well_id, x, y, screen_top,
            screen_bottom, (optional) layer
        observed_data : pd.DataFrame | Path
            Observed head data with columns: well_id, datetime, head
        layers : int | list[int] | str
            Layer(s) for observations. "auto" determines from screen depth.
        weight_strategy : str
            How to assign observation weights:
            - "equal": All weights = 1
            - "inverse_variance": Weight = 1/variance
            - "decay": Higher weight for recent observations
            - "group_contribution": Equal contribution per group

        Examples
        --------
        >>> om.add_head_observations(
        ...     wells="observation_wells.csv",
        ...     observed_data="head_timeseries.csv",
        ...     layers="auto",
        ...     weight_strategy="inverse_variance",
        ...     frequency="MS",  # Monthly start
        ... )
        """

    def add_drawdown_observations(
        self,
        wells: pd.DataFrame | Path,
        observed_data: pd.DataFrame | Path,
        reference_date: datetime | None = None,
        **kwargs,
    ) -> list[Observation]:
        """Add drawdown observations (change from reference)."""

    def add_head_difference_observations(
        self,
        well_pairs: list[tuple[str, str]],
        observed_data: pd.DataFrame | Path,
        **kwargs,
    ) -> list[Observation]:
        """Add head difference observations between well pairs."""

    # --- Stream observations ---

    def add_streamflow_observations(
        self,
        gages: pd.DataFrame | Path,
        observed_data: pd.DataFrame | Path,
        weight_strategy: str = "equal",
        transform: str = "none",  # none, log, sqrt
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        frequency: str | None = None,
    ) -> list[Observation]:
        """Add stream discharge observations.

        Parameters
        ----------
        gages : pd.DataFrame | Path
            Gage information with columns: gage_id, reach_id or node_id
        observed_data : pd.DataFrame | Path
            Observed flow data with columns: gage_id, datetime, flow
        transform : str
            Transform for flow observations:
            - "none": Use raw values
            - "log": Log-transform (good for wide range of flows)
            - "sqrt": Square root transform

        Examples
        --------
        >>> om.add_streamflow_observations(
        ...     gages="stream_gages.csv",
        ...     observed_data="streamflow_timeseries.csv",
        ...     transform="log",
        ...     weight_strategy="group_contribution",
        ... )
        """

    def add_stream_stage_observations(
        self,
        gages: pd.DataFrame | Path,
        observed_data: pd.DataFrame | Path,
        **kwargs,
    ) -> list[Observation]:
        """Add stream stage observations."""

    def add_gain_loss_observations(
        self,
        reaches: list[int],
        observed_data: pd.DataFrame | Path,
        **kwargs,
    ) -> list[Observation]:
        """Add stream gain/loss observations."""

    # --- Lake observations ---

    def add_lake_observations(
        self,
        lakes: list[int] | str = "all",
        observed_data: pd.DataFrame | Path = None,
        obs_type: str = "level",  # level, storage
        **kwargs,
    ) -> list[Observation]:
        """Add lake level or storage observations."""

    # --- Budget observations ---

    def add_budget_observations(
        self,
        budget_type: str,  # gw, stream, rootzone, lake
        components: list[str] | None = None,
        locations: list[int] | str = "all",
        aggregate: str = "sum",  # sum, mean, by_location
        observed_data: pd.DataFrame | Path = None,
        **kwargs,
    ) -> list[Observation]:
        """Add water budget component observations.

        Examples
        --------
        >>> # Add stream-aquifer exchange observations by reach
        >>> om.add_budget_observations(
        ...     budget_type="stream",
        ...     components=["GW_EXCHANGE"],
        ...     locations="all",
        ...     aggregate="by_location",
        ... )
        """

    # --- Derived observations ---

    def add_derived_observations(
        self,
        expression: str,
        obs_names: list[str],
        result_name: str,
        weight: float = 1.0,
    ) -> Observation:
        """Add derived observation from expression.

        Examples
        --------
        >>> # Mass balance closure
        >>> om.add_derived_observations(
        ...     expression="inflow - outflow - storage_change",
        ...     obs_names=["total_inflow", "total_outflow", "delta_storage"],
        ...     result_name="mass_balance_error",
        ...     weight=10.0,  # High weight for constraint
        ... )
        """

    # --- Weight utilities ---

    def set_group_weights(
        self,
        group: str,
        weight: float | str = "auto",
        contribution: float | None = None,
    ) -> None:
        """Set weights for an observation group.

        Parameters
        ----------
        weight : float | str
            If "auto", calculates weights to achieve target contribution.
        contribution : float | None
            Target contribution to objective function (0-1).
        """

    def balance_observation_groups(
        self,
        target_contributions: dict[str, float] | None = None,
    ) -> None:
        """Balance weights so groups contribute equally or as specified."""

    def apply_temporal_weights(
        self,
        decay_factor: float = 0.95,
        reference_date: datetime | None = None,
    ) -> None:
        """Apply temporal decay to observation weights."""

    def get_observation_dataframe(self) -> pd.DataFrame:
        """Get all observations as a DataFrame."""
```

---

## Module 3: Geostatistics

### 3.1 Variogram Classes

```python
@dataclass
class Variogram:
    """Variogram model for geostatistical analysis."""

    type: str  # spherical, exponential, gaussian, matern
    a: float  # range
    sill: float = 1.0
    nugget: float = 0.0
    anisotropy_ratio: float = 1.0
    anisotropy_angle: float = 0.0  # degrees from east

    def evaluate(self, h: NDArray) -> NDArray:
        """Evaluate variogram at lag distances h."""

    @classmethod
    def from_data(
        cls,
        x: NDArray,
        y: NDArray,
        values: NDArray,
        n_lags: int = 15,
        max_lag: float | None = None,
    ) -> "Variogram":
        """Fit variogram to data."""


class GeostatManager:
    """Manages geostatistical operations for parameterization."""

    def __init__(self, model: IWFMModel):
        self.model = model

    def compute_covariance_matrix(
        self,
        points: list[tuple[float, float]],
        variogram: Variogram,
    ) -> NDArray:
        """Compute covariance matrix between points."""

    def krige(
        self,
        pilot_points: list[tuple[float, float]],
        pilot_values: NDArray,
        target_points: list[tuple[float, float]],
        variogram: Variogram,
        kriging_type: str = "ordinary",
    ) -> NDArray:
        """Interpolate values using kriging."""

    def generate_realizations(
        self,
        points: list[tuple[float, float]],
        variogram: Variogram,
        n_realizations: int = 100,
        conditioning_data: tuple[NDArray, NDArray, NDArray] | None = None,
    ) -> NDArray:
        """Generate geostatistical realizations."""

    def write_kriging_factors(
        self,
        pilot_points: list[tuple[float, float]],
        target_points: list[tuple[float, float]],
        variogram: Variogram,
        filepath: Path,
    ) -> None:
        """Write kriging interpolation factors to file."""
```

---

## Module 4: Template and Instruction File Generation

### 4.1 IWFM-Aware Template Manager

```python
class IWFMTemplateManager:
    """Generates PEST++ template files for IWFM input files."""

    def __init__(self, model: IWFMModel, parameter_manager: IWFMParameterManager):
        self.model = model
        self.pm = parameter_manager

    def generate_aquifer_template(
        self,
        input_file: Path,
        output_template: Path,
        param_type: IWFMParameterType,
        layer: int | None = None,
    ) -> TemplateFile:
        """Generate template for aquifer parameter file."""

    def generate_stream_template(
        self,
        input_file: Path,
        output_template: Path,
        param_type: IWFMParameterType,
    ) -> TemplateFile:
        """Generate template for stream parameter file."""

    def generate_multiplier_template(
        self,
        input_file: Path,
        output_template: Path,
        param_type: IWFMParameterType,
        spatial: str = "global",
        temporal: str = "constant",
    ) -> TemplateFile:
        """Generate template with multiplier parameters."""

    def generate_pilot_point_template(
        self,
        pilot_points: list[tuple[float, float]],
        param_type: IWFMParameterType,
        layer: int,
        output_template: Path,
    ) -> TemplateFile:
        """Generate template for pilot point file.

        The pilot point file is separate from IWFM inputs.
        A preprocessor interpolates pilot point values to
        model nodes/elements before running IWFM.
        """

    def generate_all_templates(self) -> list[TemplateFile]:
        """Generate all required template files based on parameters."""


class IWFMInstructionManager:
    """Generates PEST++ instruction files for IWFM output files."""

    def __init__(self, model: IWFMModel, observation_manager: IWFMObservationManager):
        self.model = model
        self.om = observation_manager

    def generate_head_instructions(
        self,
        output_file: Path,
        instruction_file: Path,
        wells: list[str],
        times: list[datetime],
    ) -> InstructionFile:
        """Generate instructions for head hydrograph output."""

    def generate_flow_instructions(
        self,
        output_file: Path,
        instruction_file: Path,
        gages: list[str],
        times: list[datetime],
    ) -> InstructionFile:
        """Generate instructions for streamflow output."""

    def generate_budget_instructions(
        self,
        budget_file: Path,
        instruction_file: Path,
        components: list[str],
        locations: list[int],
        times: list[datetime],
    ) -> InstructionFile:
        """Generate instructions for budget output."""

    def generate_all_instructions(self) -> list[InstructionFile]:
        """Generate all required instruction files based on observations."""
```

---

## Module 5: Main Interface Class

### 5.1 IWFMPestHelper

```python
class IWFMPestHelper:
    """Main interface for IWFM PEST++ setup.

    This class provides a high-level interface for setting up
    PEST++ calibration, uncertainty analysis, and optimization
    for IWFM models.

    Examples
    --------
    >>> # Initialize helper with IWFM model
    >>> helper = IWFMPestHelper.from_model_dir("C2VSimFG/")

    >>> # Add parameters
    >>> helper.add_pilot_points("hk", spacing=5000, layer=1)
    >>> helper.add_zone_parameters("sy", zones="subregions", layer=1)
    >>> helper.add_multiplier("pumping", spatial="zone")

    >>> # Add observations
    >>> helper.add_head_observations("wells.csv", "heads.csv")
    >>> helper.add_streamflow_observations("gages.csv", "flows.csv")

    >>> # Configure and build
    >>> helper.set_regularization(type="preferred_homogeneity")
    >>> helper.balance_observation_weights()
    >>> helper.build("calibration.pst")

    >>> # Run PEST++
    >>> helper.run_pestpp_glm()  # or run_pestpp_ies()
    """

    def __init__(
        self,
        model: IWFMModel,
        pest_dir: Path | str | None = None,
        case_name: str = "iwfm_cal",
    ):
        self.model = model
        self.pest_dir = Path(pest_dir) if pest_dir else model.model_dir / "pest"
        self.case_name = case_name

        # Initialize managers
        self.parameters = IWFMParameterManager(model)
        self.observations = IWFMObservationManager(model)
        self.geostat = GeostatManager(model)
        self.templates = IWFMTemplateManager(model, self.parameters)
        self.instructions = IWFMInstructionManager(model, self.observations)

    @classmethod
    def from_model_dir(
        cls,
        model_dir: Path | str,
        **kwargs,
    ) -> "IWFMPestHelper":
        """Create helper from IWFM model directory."""
        from pyiwfm.io import load_complete_model
        model = load_complete_model(model_dir)
        return cls(model, **kwargs)

    # --- Convenient parameter methods ---

    def add_pilot_points(
        self,
        param_type: str | IWFMParameterType,
        spacing: float | None = None,
        points: list[tuple[float, float]] | None = None,
        layer: int = 1,
        bounds: tuple[float, float] = (0.01, 100.0),
        variogram: Variogram | dict | None = None,
    ) -> list[Parameter]:
        """Add pilot point parameters (convenient wrapper)."""

    def add_zone_parameters(
        self,
        param_type: str | IWFMParameterType,
        zones: list[int] | str = "subregions",
        layer: int | None = None,
        bounds: tuple[float, float] = (0.01, 100.0),
    ) -> list[Parameter]:
        """Add zone parameters (convenient wrapper)."""

    def add_multiplier(
        self,
        param_type: str | IWFMParameterType,
        spatial: str = "global",
        temporal: str = "constant",
        bounds: tuple[float, float] = (0.5, 2.0),
    ) -> list[Parameter]:
        """Add multiplier parameters (convenient wrapper)."""

    # --- Convenient observation methods ---

    def add_head_observations(
        self,
        wells: pd.DataFrame | Path | str,
        observed_data: pd.DataFrame | Path | str,
        **kwargs,
    ) -> list[Observation]:
        """Add head observations (convenient wrapper)."""

    def add_streamflow_observations(
        self,
        gages: pd.DataFrame | Path | str,
        observed_data: pd.DataFrame | Path | str,
        **kwargs,
    ) -> list[Observation]:
        """Add streamflow observations (convenient wrapper)."""

    # --- Configuration methods ---

    def set_regularization(
        self,
        type: str = "preferred_homogeneity",
        weight: float = 1.0,
    ) -> None:
        """Configure regularization for pilot points."""

    def balance_observation_weights(
        self,
        contributions: dict[str, float] | None = None,
    ) -> None:
        """Balance weights across observation groups."""

    def set_svd(
        self,
        maxsing: int = 100,
        eigthresh: float = 1e-6,
    ) -> None:
        """Configure SVD truncation."""

    def set_pestpp_options(self, **options) -> None:
        """Set PEST++ specific options."""

    # --- Build methods ---

    def build(
        self,
        pst_file: Path | str | None = None,
        write_mult_files: bool = True,
        write_pp_files: bool = True,
    ) -> Path:
        """Build complete PEST++ setup.

        Creates:
        - Control file (.pst)
        - Template files (.tpl)
        - Instruction files (.ins)
        - Multiplier files (if applicable)
        - Pilot point files (if applicable)
        - Model runner script
        - Pre/post-processing scripts
        """

    def write_forward_run_script(self) -> Path:
        """Write script that runs IWFM for PEST++."""

    def write_pp_interpolation_script(self) -> Path:
        """Write pilot point interpolation script."""

    # --- Execution methods ---

    def run_pestpp_glm(
        self,
        n_workers: int = 1,
        **kwargs,
    ) -> Path:
        """Run pestpp-glm for parameter estimation."""

    def run_pestpp_ies(
        self,
        n_realizations: int = 100,
        n_workers: int = 1,
        **kwargs,
    ) -> Path:
        """Run pestpp-ies for ensemble calibration."""

    def run_pestpp_sen(
        self,
        method: str = "sobol",
        n_samples: int = 1000,
        **kwargs,
    ) -> Path:
        """Run pestpp-sen for sensitivity analysis."""

    # --- Post-processing methods ---

    def load_results(self) -> "PestResults":
        """Load PEST++ results."""

    def plot_1to1(
        self,
        groups: list[str] | None = None,
    ) -> "matplotlib.figure.Figure":
        """Plot observed vs simulated."""

    def plot_residuals(
        self,
        groups: list[str] | None = None,
    ) -> "matplotlib.figure.Figure":
        """Plot residual analysis."""

    def plot_parameter_sensitivity(self) -> "matplotlib.figure.Figure":
        """Plot parameter sensitivities."""

    def export_calibrated_model(
        self,
        output_dir: Path,
    ) -> None:
        """Export model with calibrated parameters."""
```

---

## Module 6: Ensemble Support

```python
class IWFMEnsembleManager:
    """Manages ensemble generation and analysis for IWFM."""

    def __init__(self, helper: IWFMPestHelper):
        self.helper = helper

    def generate_prior_ensemble(
        self,
        n_realizations: int = 100,
        method: str = "lhs",  # lhs, gaussian, uniform
        correlation: bool = True,
    ) -> pd.DataFrame:
        """Generate prior parameter ensemble.

        Uses Latin Hypercube Sampling with optional correlation
        structure from geostatistics.
        """

    def generate_observation_ensemble(
        self,
        n_realizations: int = 100,
        noise_type: str = "gaussian",
    ) -> pd.DataFrame:
        """Generate observation noise realizations."""

    def write_ensemble_files(
        self,
        par_ensemble: pd.DataFrame,
        obs_ensemble: pd.DataFrame | None = None,
    ) -> tuple[Path, Path]:
        """Write ensemble files for pestpp-ies."""

    def load_posterior_ensemble(self) -> pd.DataFrame:
        """Load posterior parameter ensemble from pestpp-ies."""

    def analyze_ensemble(
        self,
        ensemble: pd.DataFrame,
    ) -> dict[str, Any]:
        """Compute ensemble statistics."""

    def plot_ensemble_evolution(self) -> "matplotlib.figure.Figure":
        """Plot parameter ensemble evolution through IES iterations."""

    def plot_prediction_uncertainty(
        self,
        predictions: list[str],
    ) -> "matplotlib.figure.Figure":
        """Plot prediction uncertainty from ensemble."""
```

---

## Implementation Plan

### Phase 1: Core Parameter Management (Week 1-2)
1. Implement `IWFMParameterType` enum
2. Implement parameterization strategy classes
3. Implement `IWFMParameterManager` with zone and multiplier support
4. Unit tests for parameter management

### Phase 2: Observation Management (Week 2-3)
1. Implement `IWFMObservationType` enum
2. Implement `IWFMObservationManager` with head and flow support
3. Implement weight calculation strategies
4. Unit tests for observation management

### Phase 3: Template/Instruction Generation (Week 3-4)
1. Implement IWFM-aware template file generation
2. Implement instruction file generation for hydrographs
3. Handle budget file output parsing
4. Integration tests with sample IWFM files

### Phase 4: Geostatistics (Week 4-5)
1. Implement `Variogram` class with common models
2. Implement `GeostatManager` with kriging
3. Implement pilot point parameterization
4. Implement ensemble generation

### Phase 5: Main Interface (Week 5-6)
1. Implement `IWFMPestHelper` main class
2. Implement convenient wrapper methods
3. Implement build and execution methods
4. Integration tests with full PEST++ workflow

### Phase 6: Post-processing (Week 6-7)
1. Implement results loading and parsing
2. Implement visualization methods
3. Implement model export with calibrated parameters
4. Documentation and examples

---

## Example Workflows

### Basic Calibration

```python
from pyiwfm.runner.pest import IWFMPestHelper

# Load model and create helper
helper = IWFMPestHelper.from_model_dir("C2VSimFG/")

# Add parameters
helper.add_zone_parameters("hk", zones="subregions", layer=1, bounds=(0.1, 1000))
helper.add_zone_parameters("sy", zones="subregions", layer=1, bounds=(0.01, 0.3))
helper.add_multiplier("pumping", spatial="global", bounds=(0.8, 1.2))

# Add observations
helper.add_head_observations(
    wells="observation_wells.csv",
    observed_data="head_data.csv",
    weight_strategy="inverse_variance",
)
helper.add_streamflow_observations(
    gages="stream_gages.csv",
    observed_data="flow_data.csv",
    transform="log",
)

# Configure and build
helper.balance_observation_weights({"head": 0.5, "flow": 0.5})
helper.set_svd(maxsing=50)
helper.build("c2vsim_cal.pst")

# Run calibration
helper.run_pestpp_glm(n_workers=8)

# Analyze results
results = helper.load_results()
helper.plot_1to1()
helper.export_calibrated_model("calibrated_model/")
```

### Highly Parameterized with Pilot Points

```python
from pyiwfm.runner.pest import IWFMPestHelper, Variogram

helper = IWFMPestHelper.from_model_dir("C2VSimFG/")

# Add pilot points for hydraulic conductivity
helper.add_pilot_points(
    "hk",
    spacing=5000,  # 5km spacing
    layer=1,
    bounds=(0.1, 1000),
    variogram=Variogram("exponential", a=15000, sill=1.0, nugget=0.1),
)

# Zone-based specific yield
helper.add_zone_parameters("sy", zones="subregions", layer=1)

# Stream parameters
helper.add_zone_parameters("streambed_k", zones="reaches", bounds=(0.01, 10))

# Add observations with appropriate weights
helper.add_head_observations("wells.csv", "heads.csv")
helper.add_streamflow_observations("gages.csv", "flows.csv")

# Enable regularization for pilot points
helper.set_regularization(type="preferred_homogeneity", weight=1.0)

# Build and run
helper.build("c2vsim_pp.pst")
helper.run_pestpp_glm(n_workers=16)
```

### Uncertainty Analysis with IES

```python
from pyiwfm.runner.pest import IWFMPestHelper

helper = IWFMPestHelper.from_model_dir("C2VSimFG/")

# Same parameter and observation setup...
helper.add_pilot_points("hk", spacing=5000, layer=1)
helper.add_head_observations("wells.csv", "heads.csv")

# Configure for IES
helper.set_pestpp_options(
    ies_num_reals=100,
    ies_lambda_mults=[0.1, 1, 10],
    ies_subset_size=20,
)

# Generate and write prior ensemble
ensemble_mgr = helper.get_ensemble_manager()
prior = ensemble_mgr.generate_prior_ensemble(n_realizations=100)
ensemble_mgr.write_ensemble_files(prior)

# Build and run IES
helper.build("c2vsim_ies.pst")
helper.run_pestpp_ies(n_workers=16)

# Analyze posterior uncertainty
posterior = ensemble_mgr.load_posterior_ensemble()
ensemble_mgr.plot_ensemble_evolution()
ensemble_mgr.plot_prediction_uncertainty(["head_well1", "flow_gage1"])
```

---

## File Outputs

The `build()` method creates the following directory structure:

```
pest/
├── c2vsim_cal.pst              # PEST++ control file
├── c2vsim_cal.par              # Initial parameter values
├── c2vsim_cal.obs              # Observation values
├── templates/
│   ├── aquifer_params_l1.tpl   # Aquifer parameter template
│   ├── pumping_mult.tpl        # Pumping multiplier template
│   └── pilot_points_hk.tpl     # Pilot point template
├── instructions/
│   ├── head_hydrographs.ins    # Head observation instructions
│   └── flow_hydrographs.ins    # Flow observation instructions
├── multipliers/
│   ├── pumping_mult.dat        # Pumping multiplier file
│   └── et_mult.dat             # ET multiplier file
├── pilot_points/
│   ├── pp_hk_l1.dat            # Pilot point locations/values
│   └── pp_factors_hk_l1.dat    # Kriging factors
├── scripts/
│   ├── forward_run.py          # Model run script
│   ├── interpolate_pp.py       # Pilot point interpolation
│   └── apply_multipliers.py    # Apply multipliers to inputs
└── prior_ensemble.csv          # Prior parameter ensemble (if IES)
```

---

## Dependencies

- **Required**: numpy, pandas, scipy
- **Optional**:
  - pyemu (for compatibility and some utilities)
  - flopy (for geostats if not implementing our own)
  - matplotlib (for plotting)
  - h5py (for budget files)
