"""write_pest_control_file — extracted from runner/pest.py in v2.0 PR 5.

The class body is verbatim from v1.x ``runner/pest.py``; this module
just gives it a dedicated file so the package no longer has a
1,400-line module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pyiwfm.runner.pest.instruction import InstructionFile
from pyiwfm.runner.pest.interface import PESTInterface
from pyiwfm.runner.pest.observation import Observation, ObservationGroup
from pyiwfm.runner.pest.parameter import Parameter
from pyiwfm.runner.pest.template import TemplateFile


def write_pest_control_file(
    filepath: Path | str,
    parameters: list[Parameter],
    observations: list[Observation],
    template_files: list[TemplateFile],
    instruction_files: list[InstructionFile],
    model_command: str = "python run_model.py",
    version: Literal[1, 2] = 1,
    **pestpp_options: Any,
) -> Path:
    """Convenience function to write a PEST++ control file.

    Parameters
    ----------
    filepath : Path | str
        Output path for the control file.
    parameters : list[Parameter]
        List of parameters.
    observations : list[Observation]
        List of observations.
    template_files : list[TemplateFile]
        List of template files.
    instruction_files : list[InstructionFile]
        List of instruction files.
    model_command : str
        Command to run the model.
    version : {1, 2}
        Control file format version (1 = traditional, 2 = keyword/external).
    **pestpp_options : Any
        PEST++ options.

    Returns
    -------
    Path
        Path to the written control file.
    """
    filepath = Path(filepath)

    pest = PESTInterface(
        model_dir=filepath.parent,
        pest_dir=filepath.parent,
        case_name=filepath.stem,
    )

    pest.parameters = parameters
    pest.observations = observations
    pest.template_files = template_files
    pest.instruction_files = instruction_files
    pest.model_command = model_command
    pest.pestpp_options = pestpp_options

    # Build groups from parameters and observations
    for param in parameters:
        if param.group not in pest.parameter_groups:
            pest.parameter_groups[param.group] = {}
    for obs in observations:
        if obs.group not in pest.observation_groups:
            pest.observation_groups[obs.group] = ObservationGroup(name=obs.group)

    return pest.write_control_file(filepath, version=version)
