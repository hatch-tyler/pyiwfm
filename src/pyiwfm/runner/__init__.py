"""IWFM subprocess runner module.

This module provides utilities for running IWFM executables via subprocess,
managing scenarios, and integrating with PEST++ for calibration.
"""

from __future__ import annotations

from pyiwfm.runner.executables import (
    IWFMExecutableManager,
)
from pyiwfm.runner.pest import (
    InstructionFile,
    ObservationGroup,
    PESTInterface,
    TemplateFile,
    write_pest_control_file,
)
from pyiwfm.runner.pest_ensemble import (
    EnsembleStatistics,
    IWFMEnsembleManager,
)
from pyiwfm.runner.pest_geostat import (
    GeostatManager,
    Variogram,
    VariogramType,
    compute_empirical_variogram,
)
from pyiwfm.runner.pest_helper import (
    IWFMPestHelper,
    RegularizationConfig,
    RegularizationType,
    SVDConfig,
)
from pyiwfm.runner.pest_instructions import (
    IWFM_OUTPUT_FORMATS,
    IWFMInstructionManager,
    OutputFileFormat,
)
from pyiwfm.runner.pest_manager import (
    IWFMParameterManager,
)
from pyiwfm.runner.pest_obs_manager import (
    GageInfo,
    IWFMObservationManager,
    WellInfo,
)
from pyiwfm.runner.pest_observations import (
    DerivedObservation,
    IWFMObservation,
    IWFMObservationGroup,
    IWFMObservationType,
    ObservationLocation,
    WeightStrategy,
)
from pyiwfm.runner.pest_params import (
    DirectParameterization,
    IWFMParameterType,
    MultiplierParameterization,
    Parameter,
    ParameterGroup,
    ParameterizationStrategy,
    ParameterTransform,
    PilotPointParameterization,
    RootZoneParameterization,
    StreamParameterization,
    ZoneParameterization,
)
from pyiwfm.runner.pest_postprocessor import (
    CalibrationResults,
    PestPostProcessor,
    ResidualData,
    SensitivityData,
)
from pyiwfm.runner.pest_templates import (
    IWFMFileSection,
    IWFMTemplateManager,
    TemplateMarker,
)
from pyiwfm.runner.results import (
    BudgetResult,
    PreprocessorResult,
    RunResult,
    SimulationResult,
    ZBudgetResult,
)
from pyiwfm.runner.runner import (
    IWFMExecutables,
    IWFMRunner,
    find_iwfm_executables,
)
from pyiwfm.runner.scenario import (
    Scenario,
    ScenarioManager,
    ScenarioResult,
)

__all__ = [
    # Runner
    "IWFMRunner",
    "IWFMExecutables",
    "find_iwfm_executables",
    # Executable manager
    "IWFMExecutableManager",
    # Results
    "RunResult",
    "PreprocessorResult",
    "SimulationResult",
    "BudgetResult",
    "ZBudgetResult",
    # Scenarios
    "Scenario",
    "ScenarioManager",
    "ScenarioResult",
    # PEST++ integration
    "PESTInterface",
    "TemplateFile",
    "InstructionFile",
    "ObservationGroup",
    "write_pest_control_file",
    # Parameter types and strategies
    "IWFMParameterType",
    "ParameterTransform",
    "ParameterGroup",
    "Parameter",
    "ParameterizationStrategy",
    "ZoneParameterization",
    "MultiplierParameterization",
    "PilotPointParameterization",
    "DirectParameterization",
    "StreamParameterization",
    "RootZoneParameterization",
    # Parameter manager
    "IWFMParameterManager",
    # Observation types and classes
    "IWFMObservationType",
    "IWFMObservation",
    "IWFMObservationGroup",
    "ObservationLocation",
    "WeightStrategy",
    "DerivedObservation",
    # Observation manager
    "IWFMObservationManager",
    "WellInfo",
    "GageInfo",
    # Template generation
    "IWFMTemplateManager",
    "TemplateMarker",
    "IWFMFileSection",
    # Instruction generation
    "IWFMInstructionManager",
    "OutputFileFormat",
    "IWFM_OUTPUT_FORMATS",
    # Geostatistics
    "VariogramType",
    "Variogram",
    "GeostatManager",
    "compute_empirical_variogram",
    # Main interface
    "IWFMPestHelper",
    "RegularizationType",
    "SVDConfig",
    "RegularizationConfig",
    # Ensemble management
    "IWFMEnsembleManager",
    "EnsembleStatistics",
    # Post-processing
    "PestPostProcessor",
    "CalibrationResults",
    "ResidualData",
    "SensitivityData",
]
