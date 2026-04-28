"""PEST++ integration for IWFM calibration and uncertainty analysis.

This module provides utilities for:

- Creating PEST++ template (.tpl) and instruction (.ins) files
- Writing PEST++ control files (.pst) in v1 or v2 (keyword/external) format
- Running IWFM as a PEST++ model
- Parsing PEST++ output

PEST++ suite includes:

- pestpp-glm: Gauss-Levenberg-Marquardt parameter estimation
- pestpp-ies: Iterative ensemble smoother (uncertainty analysis)
- pestpp-opt: Optimization under uncertainty
- pestpp-sen: Global sensitivity analysis
- pestpp-sqp: Sequential quadratic programming

Control file formats:

- v1 (traditional): Positional control data, inline parameter/observation data
- v2 (keyword/external): ``* control data keyword`` section with key-value pairs,
  external CSV files for parameter data, observation data, and model I/O.
  Introduced in PEST++ 4.3.0.  Counts (NPAR, NOBS, etc.) are inferred from
  the external files; SVD settings and ``++`` options are folded into the
  keyword section.
"""

from __future__ import annotations

from pyiwfm.runner.pest.instruction import InstructionFile
from pyiwfm.runner.pest.interface import PESTInterface
from pyiwfm.runner.pest.observation import Observation, ObservationGroup
from pyiwfm.runner.pest.parameter import Parameter
from pyiwfm.runner.pest.template import TemplateFile
from pyiwfm.runner.pest.write_control_file import write_pest_control_file

__all__ = [
    "InstructionFile",
    "Observation",
    "ObservationGroup",
    "PESTInterface",
    "Parameter",
    "TemplateFile",
    "write_pest_control_file",
]
