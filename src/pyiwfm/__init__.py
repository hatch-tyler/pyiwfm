"""
pyiwfm - Python package for IWFM (Integrated Water Flow Model) models.

This package provides tools for:
- Reading and writing IWFM model files
- Visualizing model meshes and results
- Comparing different model versions
- Generating finite element meshes
"""

from __future__ import annotations

__version__ = "0.1.0"

from pyiwfm.core.exceptions import (
    IWFMIOError,
    MeshError,
    PyIWFMError,
    StratigraphyError,
    ValidationError,
)
from pyiwfm.core.mesh import AppGrid, Element, Face, Node, Subregion
from pyiwfm.core.model import IWFMModel
from pyiwfm.core.stratigraphy import Stratigraphy

# Runner module for executing IWFM via subprocess
from pyiwfm.runner import (
    BudgetResult,
    InstructionFile,
    IWFMExecutables,
    IWFMRunner,
    ObservationGroup,
    PESTInterface,
    PreprocessorResult,
    RunResult,
    Scenario,
    ScenarioManager,
    ScenarioResult,
    SimulationResult,
    TemplateFile,
    ZBudgetResult,
    find_iwfm_executables,
)
from pyiwfm.sample_models import (
    create_sample_budget_data,
    create_sample_element_field,
    create_sample_mesh,
    create_sample_model,
    create_sample_scalar_field,
    create_sample_stratigraphy,
    create_sample_stream_network,
    create_sample_timeseries,
    create_sample_timeseries_collection,
    create_sample_triangular_mesh,
)

__all__ = [
    "__version__",
    # Core mesh classes
    "Node",
    "Element",
    "Face",
    "Subregion",
    "AppGrid",
    # Stratigraphy
    "Stratigraphy",
    # Model
    "IWFMModel",
    # Exceptions
    "PyIWFMError",
    "MeshError",
    "StratigraphyError",
    "ValidationError",
    "IWFMIOError",
    # Sample models
    "create_sample_mesh",
    "create_sample_triangular_mesh",
    "create_sample_stratigraphy",
    "create_sample_scalar_field",
    "create_sample_element_field",
    "create_sample_timeseries",
    "create_sample_timeseries_collection",
    "create_sample_stream_network",
    "create_sample_budget_data",
    "create_sample_model",
    # Runner module
    "IWFMRunner",
    "IWFMExecutables",
    "find_iwfm_executables",
    "RunResult",
    "PreprocessorResult",
    "SimulationResult",
    "BudgetResult",
    "ZBudgetResult",
    "Scenario",
    "ScenarioManager",
    "ScenarioResult",
    "PESTInterface",
    "TemplateFile",
    "InstructionFile",
    "ObservationGroup",
]
