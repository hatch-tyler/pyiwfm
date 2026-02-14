"""IWFM subprocess runner module.

This module provides utilities for running IWFM executables via subprocess,
managing scenarios, and integrating with PEST++ for calibration.
"""

from __future__ import annotations

from pyiwfm.runner.runner import (
    IWFMRunner,
    IWFMExecutables,
    find_iwfm_executables,
)

from pyiwfm.runner.results import (
    RunResult,
    PreprocessorResult,
    SimulationResult,
    BudgetResult,
    ZBudgetResult,
)

from pyiwfm.runner.scenario import (
    Scenario,
    ScenarioManager,
    ScenarioResult,
)

from pyiwfm.runner.pest import (
    PESTInterface,
    TemplateFile,
    InstructionFile,
    ObservationGroup,
    write_pest_control_file,
)

from pyiwfm.runner.pest_params import (
    IWFMParameterType,
    ParameterTransform,
    ParameterGroup,
    Parameter,
    ParameterizationStrategy,
    ZoneParameterization,
    MultiplierParameterization,
    PilotPointParameterization,
    DirectParameterization,
    StreamParameterization,
    RootZoneParameterization,
)

from pyiwfm.runner.pest_manager import (
    IWFMParameterManager,
)

from pyiwfm.runner.pest_observations import (
    IWFMObservationType,
    IWFMObservation,
    IWFMObservationGroup,
    ObservationLocation,
    WeightStrategy,
    DerivedObservation,
)

from pyiwfm.runner.pest_obs_manager import (
    IWFMObservationManager,
    WellInfo,
    GageInfo,
)

from pyiwfm.runner.pest_templates import (
    IWFMTemplateManager,
    TemplateMarker,
    IWFMFileSection,
)

from pyiwfm.runner.pest_instructions import (
    IWFMInstructionManager,
    OutputFileFormat,
    IWFM_OUTPUT_FORMATS,
)

from pyiwfm.runner.pest_geostat import (
    VariogramType,
    Variogram,
    GeostatManager,
    compute_empirical_variogram,
)

from pyiwfm.runner.pest_helper import (
    IWFMPestHelper,
    RegularizationType,
    SVDConfig,
    RegularizationConfig,
)

from pyiwfm.runner.pest_ensemble import (
    IWFMEnsembleManager,
    EnsembleStatistics,
)

from pyiwfm.runner.pest_postprocessor import (
    PestPostProcessor,
    CalibrationResults,
    ResidualData,
    SensitivityData,
)

__all__ = [
    # Runner
    "IWFMRunner",
    "IWFMExecutables",
    "find_iwfm_executables",
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
