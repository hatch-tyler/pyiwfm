"""PEST++ integration for IWFM calibration and uncertainty analysis.

This module provides utilities for:
- Creating PEST++ template (.tpl) and instruction (.ins) files
- Writing PEST++ control files (.pst)
- Running IWFM as a PEST++ model
- Parsing PEST++ output

PEST++ suite includes:
- pestpp-glm: Gauss-Levenberg-Marquardt parameter estimation
- pestpp-ies: Iterative ensemble smoother (uncertainty analysis)
- pestpp-opt: Optimization under uncertainty
- pestpp-sen: Global sensitivity analysis
- pestpp-sqp: Sequential quadratic programming
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO


@dataclass
class Parameter:
    """PEST++ parameter definition.

    Attributes
    ----------
    name : str
        Parameter name (up to 200 chars in PEST++).
    initial_value : float
        Initial parameter value.
    lower_bound : float
        Lower bound for parameter.
    upper_bound : float
        Upper bound for parameter.
    group : str
        Parameter group name.
    transform : str
        Transformation type: 'none', 'log', 'fixed', 'tied'.
    scale : float
        Scale factor for parameter.
    offset : float
        Offset for parameter.
    """

    name: str
    initial_value: float
    lower_bound: float
    upper_bound: float
    group: str = "default"
    transform: str = "none"
    scale: float = 1.0
    offset: float = 0.0

    def __post_init__(self) -> None:
        """Validate parameter."""
        if len(self.name) > 200:
            raise ValueError(f"Parameter name too long: {self.name}")
        if self.lower_bound > self.upper_bound:
            raise ValueError(f"Lower bound ({self.lower_bound}) > upper bound ({self.upper_bound})")
        if not self.lower_bound <= self.initial_value <= self.upper_bound:
            raise ValueError(
                f"Initial value ({self.initial_value}) not within bounds "
                f"[{self.lower_bound}, {self.upper_bound}]"
            )

    def to_pest_line(self) -> str:
        """Format as PEST control file parameter line."""
        return (
            f"{self.name:20s} {self.transform:10s} {1:8d} "
            f"{self.initial_value:15.7e} {self.lower_bound:15.7e} "
            f"{self.upper_bound:15.7e} {self.group:20s} "
            f"{self.scale:10.3e} {self.offset:10.3e}"
        )


@dataclass
class Observation:
    """PEST++ observation definition.

    Attributes
    ----------
    name : str
        Observation name (up to 200 chars in PEST++).
    value : float
        Observed value.
    weight : float
        Observation weight (inverse of standard deviation).
    group : str
        Observation group name.
    """

    name: str
    value: float
    weight: float = 1.0
    group: str = "default"

    def __post_init__(self) -> None:
        """Validate observation."""
        if len(self.name) > 200:
            raise ValueError(f"Observation name too long: {self.name}")
        if self.weight < 0:
            raise ValueError(f"Weight must be non-negative: {self.weight}")

    def to_pest_line(self) -> str:
        """Format as PEST control file observation line."""
        return f"{self.name:20s} {self.value:15.7e} {self.weight:10.4e} {self.group:20s}"


@dataclass
class ObservationGroup:
    """Group of observations with shared properties.

    Attributes
    ----------
    name : str
        Group name.
    observations : list[Observation]
        Observations in this group.
    covariance_matrix : str | None
        Path to covariance matrix file for this group.
    """

    name: str
    observations: list[Observation] = field(default_factory=list)
    covariance_matrix: str | None = None

    def add_observation(
        self,
        name: str,
        value: float,
        weight: float = 1.0,
    ) -> Observation:
        """Add an observation to this group.

        Parameters
        ----------
        name : str
            Observation name.
        value : float
            Observed value.
        weight : float
            Observation weight.

        Returns
        -------
        Observation
            The created observation.
        """
        obs = Observation(name=name, value=value, weight=weight, group=self.name)
        self.observations.append(obs)
        return obs


@dataclass
class TemplateFile:
    """PEST++ template file (.tpl) definition.

    A template file is an input file with parameters marked by
    delimiters. PEST++ replaces these markers with parameter values.

    Attributes
    ----------
    template_path : Path
        Path to the template file.
    input_path : Path
        Path to the model input file to generate.
    delimiter : str
        Delimiter character for parameter markers (default: '#').
    parameters : list[str]
        List of parameter names in this template.
    """

    template_path: Path
    input_path: Path
    delimiter: str = "#"
    parameters: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Convert paths."""
        self.template_path = Path(self.template_path)
        self.input_path = Path(self.input_path)

    @classmethod
    def create_from_file(
        cls,
        input_file: Path | str,
        template_file: Path | str,
        parameters: dict[str, float],
        delimiter: str = "#",
    ) -> TemplateFile:
        """Create a template file from an existing input file.

        Parameters
        ----------
        input_file : Path | str
            Path to the original input file.
        template_file : Path | str
            Path where template will be written.
        parameters : dict[str, float]
            Dictionary mapping parameter names to their current values
            in the input file. These values will be replaced with markers.
        delimiter : str
            Delimiter character for parameter markers.

        Returns
        -------
        TemplateFile
            The created template file object.
        """
        input_file = Path(input_file)
        template_file = Path(template_file)

        content = input_file.read_text()

        # Replace parameter values with markers
        param_names = []
        for param_name, value in parameters.items():
            # Create marker with fixed width
            marker = f"{delimiter}{param_name:^12s}{delimiter}"

            # Replace the value with the marker
            # Handle different numeric formats
            patterns = [
                f"{value:.6e}",
                f"{value:.6f}",
                f"{value:.4e}",
                f"{value:.4f}",
                f"{value:g}",
                str(value),
            ]

            replaced = False
            for pattern in patterns:
                if pattern in content:
                    content = content.replace(pattern, marker, 1)
                    replaced = True
                    break

            if replaced:
                param_names.append(param_name)

        # Write template file with header
        with open(template_file, "w") as f:
            f.write(f"ptf {delimiter}\n")
            f.write(content)

        return cls(
            template_path=template_file,
            input_path=input_file,
            delimiter=delimiter,
            parameters=param_names,
        )

    def to_pest_line(self) -> str:
        """Format as PEST control file template line."""
        return f"{self.template_path.name}  {self.input_path.name}"


@dataclass
class InstructionFile:
    """PEST++ instruction file (.ins) definition.

    An instruction file tells PEST++ how to read model output
    to extract simulated observation values.

    Attributes
    ----------
    instruction_path : Path
        Path to the instruction file.
    output_path : Path
        Path to the model output file to read.
    marker : str
        Marker character for instructions (default: '@').
    observations : list[str]
        List of observation names extracted by this file.
    """

    instruction_path: Path
    output_path: Path
    marker: str = "@"
    observations: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Convert paths."""
        self.instruction_path = Path(self.instruction_path)
        self.output_path = Path(self.output_path)

    @classmethod
    def create_for_timeseries(
        cls,
        output_file: Path | str,
        instruction_file: Path | str,
        observations: list[tuple[str, int, int]],
        header_lines: int = 0,
        marker: str = "@",
    ) -> InstructionFile:
        """Create instruction file for reading time series output.

        Parameters
        ----------
        output_file : Path | str
            Path to the model output file.
        instruction_file : Path | str
            Path where instruction file will be written.
        observations : list[tuple[str, int, int]]
            List of (obs_name, line_number, column_number) tuples.
            Line numbers are 1-based (after header).
        header_lines : int
            Number of header lines to skip.
        marker : str
            Marker character for instructions.

        Returns
        -------
        InstructionFile
            The created instruction file object.
        """
        output_file = Path(output_file)
        instruction_file = Path(instruction_file)

        obs_names = []
        lines = [f"pif {marker}"]

        # Sort observations by line number
        sorted_obs = sorted(observations, key=lambda x: (x[1], x[2]))

        current_line = 0
        for obs_name, line_num, col_num in sorted_obs:
            # Skip to correct line
            lines_to_skip = line_num - current_line - 1 + (header_lines if current_line == 0 else 0)
            if lines_to_skip > 0:
                lines.append(f"l{lines_to_skip}")
            current_line = line_num

            # Read observation from column
            # Use whitespace-delimited reading
            # w = skip whitespace, !name! = read into observation
            instruction = " ".join(["w"] * (col_num - 1) + [f"!{obs_name}!"])
            lines.append(instruction)
            obs_names.append(obs_name)

        with open(instruction_file, "w") as f:
            f.write("\n".join(lines))

        return cls(
            instruction_path=instruction_file,
            output_path=output_file,
            marker=marker,
            observations=obs_names,
        )

    @classmethod
    def create_for_hydrograph(
        cls,
        output_file: Path | str,
        instruction_file: Path | str,
        location_name: str,
        observation_times: list[tuple[datetime, str]],
        header_lines: int = 1,
        time_column: int = 1,
        value_column: int = 2,
        marker: str = "@",
    ) -> InstructionFile:
        """Create instruction file for reading hydrograph output.

        This creates instructions to read specific time values from
        a hydrograph file by searching for timestamps.

        Parameters
        ----------
        output_file : Path | str
            Path to the hydrograph output file.
        instruction_file : Path | str
            Path where instruction file will be written.
        location_name : str
            Name prefix for observations.
        observation_times : list[tuple[datetime, str]]
            List of (datetime, obs_suffix) tuples specifying which
            times to extract and their observation name suffix.
        header_lines : int
            Number of header lines to skip.
        time_column : int
            Column containing timestamp (1-based).
        value_column : int
            Column containing value to read (1-based).
        marker : str
            Marker character for instructions.

        Returns
        -------
        InstructionFile
            The created instruction file object.
        """
        output_file = Path(output_file)
        instruction_file = Path(instruction_file)

        obs_names = []
        lines = [f"pif {marker}"]

        # Skip header
        if header_lines > 0:
            lines.append(f"l{header_lines}")

        for obs_time, suffix in observation_times:
            obs_name = f"{location_name}_{suffix}"
            # Format datetime to match IWFM output format
            time_str = obs_time.strftime("%m/%d/%Y")

            # Search for line containing this timestamp
            lines.append(f"{marker}{time_str}{marker}")

            # Read value from specified column
            instruction = " ".join(["w"] * (value_column - 1) + [f"!{obs_name}!"])
            lines.append(instruction)
            obs_names.append(obs_name)

        with open(instruction_file, "w") as f:
            f.write("\n".join(lines))

        return cls(
            instruction_path=instruction_file,
            output_path=output_file,
            marker=marker,
            observations=obs_names,
        )

    def to_pest_line(self) -> str:
        """Format as PEST control file instruction line."""
        return f"{self.instruction_path.name}  {self.output_path.name}"


@dataclass
class PESTInterface:
    """Interface for setting up and running PEST++ with IWFM.

    This class manages the creation of PEST++ input files and
    coordinates running IWFM as a PEST++ model.

    Parameters
    ----------
    model_dir : Path
        Directory containing the IWFM model.
    pest_dir : Path | None
        Directory for PEST++ files. Defaults to model_dir/pest.
    case_name : str
        Base name for PEST++ files (e.g., "iwfm" -> iwfm.pst).

    Examples
    --------
    >>> pest = PESTInterface("C2VSim/Simulation", case_name="c2vsim_cal")
    >>> pest.add_parameter("hk_zone1", 1.0, 0.01, 100.0, group="hk")
    >>> pest.add_observation_group("heads", obs_data)
    >>> pest.write_control_file()
    >>> pest.run_pestpp_glm()
    """

    model_dir: Path
    case_name: str
    pest_dir: Path | None = None
    parameters: list[Parameter] = field(default_factory=list)
    parameter_groups: dict[str, dict[str, Any]] = field(default_factory=dict)
    observations: list[Observation] = field(default_factory=list)
    observation_groups: dict[str, ObservationGroup] = field(default_factory=dict)
    template_files: list[TemplateFile] = field(default_factory=list)
    instruction_files: list[InstructionFile] = field(default_factory=list)
    model_command: str = "python run_model.py"
    pestpp_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize paths."""
        self.model_dir = Path(self.model_dir).resolve()
        if self.pest_dir is None:
            self.pest_dir = self.model_dir / "pest"
        else:
            self.pest_dir = Path(self.pest_dir).resolve()
        self.pest_dir.mkdir(parents=True, exist_ok=True)

    def add_parameter(
        self,
        name: str,
        initial_value: float,
        lower_bound: float,
        upper_bound: float,
        group: str = "default",
        transform: str = "none",
    ) -> Parameter:
        """Add a parameter to the calibration.

        Parameters
        ----------
        name : str
            Parameter name.
        initial_value : float
            Initial value.
        lower_bound : float
            Lower bound.
        upper_bound : float
            Upper bound.
        group : str
            Parameter group name.
        transform : str
            Transformation: 'none', 'log', 'fixed', 'tied'.

        Returns
        -------
        Parameter
            The created parameter.
        """
        param = Parameter(
            name=name,
            initial_value=initial_value,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            group=group,
            transform=transform,
        )
        self.parameters.append(param)

        # Ensure group exists
        if group not in self.parameter_groups:
            self.parameter_groups[group] = {
                "inctyp": "relative",
                "derinc": 0.01,
                "forcen": "switch",
                "derincmul": 2.0,
                "dermthd": "parabolic",
            }

        return param

    def add_parameter_group(
        self,
        name: str,
        inctyp: str = "relative",
        derinc: float = 0.01,
        **kwargs: Any,
    ) -> None:
        """Add or configure a parameter group.

        Parameters
        ----------
        name : str
            Group name.
        inctyp : str
            Increment type for derivatives: 'relative', 'absolute'.
        derinc : float
            Derivative increment.
        **kwargs : Any
            Additional group options.
        """
        self.parameter_groups[name] = {
            "inctyp": inctyp,
            "derinc": derinc,
            **kwargs,
        }

    def add_observation(
        self,
        name: str,
        value: float,
        weight: float = 1.0,
        group: str = "default",
    ) -> Observation:
        """Add an observation.

        Parameters
        ----------
        name : str
            Observation name.
        value : float
            Observed value.
        weight : float
            Observation weight.
        group : str
            Observation group name.

        Returns
        -------
        Observation
            The created observation.
        """
        obs = Observation(name=name, value=value, weight=weight, group=group)
        self.observations.append(obs)

        # Ensure group exists
        if group not in self.observation_groups:
            self.observation_groups[group] = ObservationGroup(name=group)
        self.observation_groups[group].observations.append(obs)

        return obs

    def add_observation_group(
        self,
        name: str,
        observations: list[tuple[str, float, float]] | None = None,
    ) -> ObservationGroup:
        """Add an observation group.

        Parameters
        ----------
        name : str
            Group name.
        observations : list[tuple[str, float, float]] | None
            Optional list of (name, value, weight) tuples.

        Returns
        -------
        ObservationGroup
            The created observation group.
        """
        group = ObservationGroup(name=name)
        self.observation_groups[name] = group

        if observations:
            for obs_name, value, weight in observations:
                self.add_observation(obs_name, value, weight, name)

        return group

    def add_template_file(self, template: TemplateFile) -> None:
        """Add a template file."""
        self.template_files.append(template)

    def add_instruction_file(self, instruction: InstructionFile) -> None:
        """Add an instruction file."""
        self.instruction_files.append(instruction)

    def set_model_command(self, command: str) -> None:
        """Set the model run command.

        Parameters
        ----------
        command : str
            Command to run the model (e.g., "python run_model.py").
        """
        self.model_command = command

    def set_pestpp_option(self, option: str, value: Any) -> None:
        """Set a PEST++ option.

        Parameters
        ----------
        option : str
            Option name (e.g., "svd_pack", "ies_num_reals").
        value : Any
            Option value.
        """
        self.pestpp_options[option] = value

    def write_control_file(self, filepath: Path | str | None = None) -> Path:
        """Write the PEST++ control file (.pst).

        Parameters
        ----------
        filepath : Path | str | None
            Output path. Defaults to pest_dir/case_name.pst.

        Returns
        -------
        Path
            Path to the written control file.
        """
        if filepath is None:
            assert self.pest_dir is not None
            filepath = self.pest_dir / f"{self.case_name}.pst"
        else:
            filepath = Path(filepath)

        with open(filepath, "w") as f:
            self._write_control_data(f)
            self._write_parameter_groups(f)
            self._write_parameter_data(f)
            self._write_observation_groups(f)
            self._write_observation_data(f)
            self._write_model_command(f)
            self._write_model_io(f)
            self._write_prior_information(f)
            self._write_pestpp_options(f)

        return filepath

    def _write_control_data(self, f: TextIO) -> None:
        """Write control data section."""
        f.write("pcf\n")
        f.write("* control data\n")
        f.write("restart  estimation\n")

        npar = len(self.parameters)
        nobs = len(self.observations)
        npargp = len(self.parameter_groups)
        nobsgp = len(self.observation_groups)
        nprior = 0

        f.write(f"{npar:6d} {nobs:6d} {npargp:6d} {nprior:6d} {nobsgp:6d}\n")

        ntplfle = len(self.template_files)
        ninsfle = len(self.instruction_files)
        f.write(f"{ntplfle:6d} {ninsfle:6d} single point 1 0 0\n")

        # Control options
        f.write(
            "  5.0   2.0   0.3    0.03    10\n"
        )  # RLAMBDA1, RLAMFAC, PHIRATSUF, PHIREDLAM, NUMLAM
        f.write("  0.1   3     0.01   3\n")  # RELPARMAX, FACPARMAX, FACORIG, MAXSINGVAL
        f.write("  0.0   0     1\n")  # PHIREDSWH, NOPTSWITCH, SPLITSWH
        f.write(
            "    50  .005     4     4  .005     4\n"
        )  # NOPTMAX, PHIREDSTP, NPHISTP, NPHINORED, RELPARSTP, NRELPAR
        f.write("    1     1      1\n")  # ICOV, ICOR, IEIG

    def _write_parameter_groups(self, f: TextIO) -> None:
        """Write parameter groups section."""
        f.write("* parameter groups\n")
        for group_name, group_opts in self.parameter_groups.items():
            inctyp = group_opts.get("inctyp", "relative")
            derinc = group_opts.get("derinc", 0.01)
            forcen = group_opts.get("forcen", "switch")
            derincmul = group_opts.get("derincmul", 2.0)
            dermthd = group_opts.get("dermthd", "parabolic")
            f.write(
                f"{group_name:20s} {inctyp:10s} {derinc:10.4e} "
                f"{0.0:10.4e} {forcen:10s} {derincmul:10.4e} {dermthd:10s}\n"
            )

    def _write_parameter_data(self, f: TextIO) -> None:
        """Write parameter data section."""
        f.write("* parameter data\n")
        for param in self.parameters:
            f.write(param.to_pest_line() + "\n")

    def _write_observation_groups(self, f: TextIO) -> None:
        """Write observation groups section."""
        f.write("* observation groups\n")
        for group_name in self.observation_groups:
            f.write(f"{group_name}\n")

    def _write_observation_data(self, f: TextIO) -> None:
        """Write observation data section."""
        f.write("* observation data\n")
        for obs in self.observations:
            f.write(obs.to_pest_line() + "\n")

    def _write_model_command(self, f: TextIO) -> None:
        """Write model command line section."""
        f.write("* model command line\n")
        f.write(f"{self.model_command}\n")

    def _write_model_io(self, f: TextIO) -> None:
        """Write model input/output section."""
        f.write("* model input/output\n")
        for tpl in self.template_files:
            f.write(tpl.to_pest_line() + "\n")
        for ins in self.instruction_files:
            f.write(ins.to_pest_line() + "\n")

    def _write_prior_information(self, f: TextIO) -> None:
        """Write prior information section."""
        f.write("* prior information\n")
        # Add prior equations if needed

    def _write_pestpp_options(self, f: TextIO) -> None:
        """Write PEST++ options section."""
        if self.pestpp_options:
            f.write("++\n")
            for option, value in self.pestpp_options.items():
                if isinstance(value, bool):
                    value = str(value).lower()
                f.write(f"++{option}({value})\n")

    def write_model_runner(self, filepath: Path | str | None = None) -> Path:
        """Write a Python script to run IWFM for PEST++.

        This creates a run_model.py script that PEST++ will call.
        The script reads the template-generated input files and
        runs the IWFM simulation.

        Parameters
        ----------
        filepath : Path | str | None
            Output path. Defaults to pest_dir/run_model.py.

        Returns
        -------
        Path
            Path to the written script.
        """
        if filepath is None:
            assert self.pest_dir is not None
            filepath = self.pest_dir / "run_model.py"
        else:
            filepath = Path(filepath)

        script = f'''#!/usr/bin/env python
"""PEST++ model runner for IWFM.

This script is called by PEST++ to run the IWFM model.
It reads template-generated input files and runs the simulation.
"""

import sys
from pathlib import Path

# Add pyiwfm to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pyiwfm.runner import IWFMRunner

def main():
    """Run IWFM simulation."""
    # Model configuration
    model_dir = Path("{self.model_dir}")
    main_file = model_dir / "Simulation.in"  # Adjust as needed

    # Create runner and execute
    runner = IWFMRunner()

    try:
        result = runner.run_simulation(main_file)

        if not result.success:
            print(f"Simulation failed: {{result.errors}}", file=sys.stderr)
            sys.exit(1)

        print(f"Simulation completed in {{result.elapsed_time}}")
        sys.exit(0)

    except Exception as e:
        print(f"Error running simulation: {{e}}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
'''

        filepath.write_text(script)
        return filepath

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"PESTInterface(case_name='{self.case_name}', "
            f"n_parameters={len(self.parameters)}, "
            f"n_observations={len(self.observations)})"
        )


def write_pest_control_file(
    filepath: Path | str,
    parameters: list[Parameter],
    observations: list[Observation],
    template_files: list[TemplateFile],
    instruction_files: list[InstructionFile],
    model_command: str = "python run_model.py",
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

    return pest.write_control_file(filepath)
