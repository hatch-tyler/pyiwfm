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

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TextIO


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
    change_limit : str
        Parameter change limit type: 'factor', 'relative', 'absolute'.
    scale : float
        Scale factor for parameter.
    offset : float
        Offset for parameter.
    dercom : int
        Derivative command index (1-based).
    tied_to : str
        Name of parent parameter if transform is 'tied'.
    """

    name: str
    initial_value: float
    lower_bound: float
    upper_bound: float
    group: str = "default"
    transform: str = "none"
    change_limit: str = "factor"
    scale: float = 1.0
    offset: float = 0.0
    dercom: int = 1
    tied_to: str = ""

    def __post_init__(self) -> None:
        """Validate parameter."""
        if len(self.name) > 200:
            raise ValueError(f"Parameter name too long: {self.name}")
        if self.lower_bound > self.upper_bound:
            raise ValueError(f"Lower bound ({self.lower_bound}) > upper bound ({self.upper_bound})")
        if self.transform == "tied" and not self.tied_to:
            raise ValueError(f"Tied parameter '{self.name}' must specify tied_to")
        # Skip bounds check for tied/fixed parameters (values may not be meaningful)
        if self.transform not in ("tied", "fixed"):
            if not self.lower_bound <= self.initial_value <= self.upper_bound:
                raise ValueError(
                    f"Initial value ({self.initial_value}) not within bounds "
                    f"[{self.lower_bound}, {self.upper_bound}]"
                )

    def to_pest_line(self) -> str:
        """Format as v1 PEST control file parameter line."""
        return (
            f"{self.name:<20s} {self.transform:>5s}   {self.change_limit:<8s}"
            f"{self.initial_value:>15.7g}  {self.lower_bound:>13.7g}  "
            f"{self.upper_bound:>13.7g}  {self.group:<12s}"
            f"{self.scale:>10.3g}  {self.offset:>6.3g}  {self.dercom}"
        )

    def to_csv_dict(self) -> dict[str, str]:
        """Return dict for CSV export (v2 external format)."""
        return {
            "parnme": self.name,
            "partrans": self.transform,
            "parchglim": self.change_limit,
            "parval1": str(self.initial_value),
            "parlbnd": str(self.lower_bound),
            "parubnd": str(self.upper_bound),
            "pargp": self.group,
            "scale": str(self.scale),
            "offset": str(self.offset),
            "dercom": str(self.dercom),
            "partied": self.tied_to,
        }


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
    extra_columns: dict[str, str] | None = None

    def __post_init__(self) -> None:
        """Validate observation."""
        if len(self.name) > 200:
            raise ValueError(f"Observation name too long: {self.name}")
        if self.weight < 0:
            raise ValueError(f"Weight must be non-negative: {self.weight}")
        if self.extra_columns is None:
            self.extra_columns = {}

    def to_pest_line(self) -> str:
        """Format as PEST control file observation line."""
        return f"{self.name:20s} {self.value:15.7e} {self.weight:10.4e} {self.group:20s}"

    def to_csv_dict(self, extra_fieldnames: list[str] | None = None) -> dict[str, str]:
        """Return dict for CSV export (v2 external format).

        Parameters
        ----------
        extra_fieldnames : list[str] | None
            If provided, include these extra columns from ``extra_columns``.
            Columns not present in ``extra_columns`` are written as empty strings.
        """
        d = {
            "obsnme": self.name,
            "obsval": str(self.value),
            "weight": str(self.weight),
            "obgnme": self.group,
        }
        if extra_fieldnames:
            for col in extra_fieldnames:
                d[col] = self.extra_columns.get(col, "") if self.extra_columns else ""
        return d


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

    # Pre-written observation CSV files for v2 format.  When set, these
    # file paths (relative to the PST directory) are listed in the
    # ``* observation data external`` section instead of auto-generating
    # a single ``obs_data.csv`` from ``self.observations``.
    obs_csv_files: list[str] = field(default_factory=list)

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

    # -- Control data settings (used by both v1 and v2) --
    # These map to the positional lines in v1 and to keyword entries in v2.
    # Set via ``set_control_data()``; any key not set uses PEST++ defaults.
    control_data: dict[str, Any] = field(default_factory=dict)

    # -- SVD settings (folded into keyword section in v2) --
    svd_settings: dict[str, Any] = field(default_factory=dict)

    def set_control_data(self, **kwargs: Any) -> None:
        """Set control data values.

        Common keywords: ``pestmode``, ``noptmax``, ``rlambda1``, ``rlamfac``,
        ``phiratsuf``, ``phiredlam``, ``numlam``, ``relparmax``, ``facparmax``,
        ``facorig``, ``phiredswh``, ``phiredstp``, ``nphistp``, ``nphinored``,
        ``relparstp``, ``nrelpar``, ``icov``, ``icor``, ``ieig``.
        """
        self.control_data.update(kwargs)

    def set_svd(self, maxsing: int = 1500, eigthresh: float = 1e-12) -> None:
        """Configure singular value decomposition.

        Parameters
        ----------
        maxsing : int
            Maximum number of singular values.
        eigthresh : float
            Eigenvalue ratio threshold.
        """
        self.svd_settings = {"maxsing": maxsing, "eigthresh": eigthresh}

    # ------------------------------------------------------------------ #
    #  write_control_file -- unified entry point for v1 and v2            #
    # ------------------------------------------------------------------ #

    def write_control_file(
        self,
        filepath: Path | str | None = None,
        *,
        version: Literal[1, 2] = 1,
        external_dir: str | None = None,
    ) -> Path:
        """Write the PEST++ control file (.pst).

        Parameters
        ----------
        filepath : Path | str | None
            Output path.  Defaults to ``pest_dir / case_name.pst``.
        version : {1, 2}
            Control file format version.

            * **1** -- Traditional PEST format with positional control data
              and inline parameter / observation data.
            * **2** -- PEST++ v2 keyword/external format.  Control data is
              written as keyword-value pairs in a ``* control data keyword``
              section.  Parameter data, observation data, and model I/O are
              written to external CSV files referenced from the PST.  SVD
              settings and ``++`` options are folded into the keyword section.
              Introduced in PEST++ 4.3.0.
        external_dir : str | None
            Subdirectory (relative to the PST file) for external CSV files.
            Only used when *version* is 2.  Defaults to ``"pest"``.

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

        if version == 2:
            return self._write_v2(filepath, external_dir or "pest")
        return self._write_v1(filepath)

    # ------------------------------------------------------------------ #
    #  Version 1 (traditional) writer                                     #
    # ------------------------------------------------------------------ #

    def _write_v1(self, filepath: Path) -> Path:
        """Write a traditional (v1) PEST++ control file."""
        with open(filepath, "w") as f:
            self._write_v1_control_data(f)
            self._write_v1_svd(f)
            self._write_v1_parameter_groups(f)
            self._write_v1_parameter_data(f)
            self._write_v1_observation_groups(f)
            self._write_v1_observation_data(f)
            self._write_v1_model_command(f)
            self._write_v1_model_io(f)
            self._write_v1_prior_information(f)
            self._write_v1_pestpp_options(f)
        return filepath

    def _write_v1_control_data(self, f: TextIO) -> None:
        """Write v1 control data section."""
        cd = self.control_data
        f.write("pcf\n")
        f.write("* control data\n")
        f.write(f"norestart {cd.get('pestmode', 'estimation')}\n")

        npar = len(self.parameters)
        nobs = len(self.observations)
        npargp = len(self.parameter_groups)
        nobsgp = len(self.observation_groups)
        f.write(f"{npar} {nobs}    {npargp}     0     {nobsgp}\n")

        ntplfle = len(self.template_files)
        ninsfle = len(self.instruction_files)
        precis = cd.get("precis", "double")
        f.write(f"{ntplfle}  {ninsfle}  {precis}  point  1  0  0\n")

        rlambda1 = cd.get("rlambda1", 5.0)
        rlamfac = cd.get("rlamfac", 2.0)
        phiratsuf = cd.get("phiratsuf", 0.3)
        phiredlam = cd.get("phiredlam", 0.03)
        numlam = cd.get("numlam", 10)
        f.write(f"{rlambda1}  {rlamfac}  {phiratsuf}  {phiredlam}  {numlam}\n")

        relparmax = cd.get("relparmax", 0.1)
        facparmax = cd.get("facparmax", 3)
        facorig = cd.get("facorig", 0.01)
        f.write(f"{relparmax}  {facparmax}  {facorig}\n")

        phiredswh = cd.get("phiredswh", 0.0)
        f.write(f"{phiredswh}\n")

        noptmax = cd.get("noptmax", 50)
        phiredstp = cd.get("phiredstp", 0.005)
        nphistp = cd.get("nphistp", 4)
        nphinored = cd.get("nphinored", 4)
        relparstp = cd.get("relparstp", 0.005)
        nrelpar = cd.get("nrelpar", 4)
        f.write(f"{noptmax}  {phiredstp}  {nphistp}  {nphinored}  {relparstp}  {nrelpar}\n")

        icov = cd.get("icov", 1)
        icor = cd.get("icor", 1)
        ieig = cd.get("ieig", 1)
        f.write(f"{icov}  {icor}  {ieig}\n")

    def _write_v1_svd(self, f: TextIO) -> None:
        """Write v1 SVD section."""
        if not self.svd_settings:
            return
        f.write("* singular value decomposition\n")
        f.write("1\n")
        maxsing = self.svd_settings.get("maxsing", 1500)
        eigthresh = self.svd_settings.get("eigthresh", 1e-12)
        f.write(f"{maxsing} {eigthresh}\n")
        f.write("1\n")

    def _write_v1_parameter_groups(self, f: TextIO) -> None:
        """Write v1 parameter groups section."""
        f.write("* parameter groups\n")
        for group_name, group_opts in self.parameter_groups.items():
            inctyp = group_opts.get("inctyp", "relative")
            derinc = group_opts.get("derinc", 0.01)
            derinclb = group_opts.get("derinclb", 0.0)
            forcen = group_opts.get("forcen", "switch")
            derincmul = group_opts.get("derincmul", 2.0)
            dermthd = group_opts.get("dermthd", "parabolic")
            f.write(
                f"{group_name:<20s} {inctyp:<10s} {derinc:<14.6e} "
                f"{derinclb:<14.6e} {forcen:<10s} {derincmul:<14.6e} {dermthd}\n"
            )

    def _write_v1_parameter_data(self, f: TextIO) -> None:
        """Write v1 parameter data section with tied parent lines."""
        f.write("* parameter data\n")
        for param in self.parameters:
            f.write(param.to_pest_line() + "\n")
        # Tied parameter parent lines
        for param in self.parameters:
            if param.tied_to:
                f.write(f"{param.name}     {param.tied_to}\n")

    def _write_v1_observation_groups(self, f: TextIO) -> None:
        """Write v1 observation groups section."""
        f.write("* observation groups\n")
        for group_name in self.observation_groups:
            f.write(f"{group_name}\n")

    def _write_v1_observation_data(self, f: TextIO) -> None:
        """Write v1 observation data section."""
        f.write("* observation data\n")
        for obs in self.observations:
            f.write(obs.to_pest_line() + "\n")

    def _write_v1_model_command(self, f: TextIO) -> None:
        """Write model command line section."""
        f.write("* model command line\n")
        f.write(f"{self.model_command}\n")

    def _write_v1_model_io(self, f: TextIO) -> None:
        """Write v1 model input/output section."""
        f.write("* model input/output\n")
        for tpl in self.template_files:
            f.write(tpl.to_pest_line() + "\n")
        for ins in self.instruction_files:
            f.write(ins.to_pest_line() + "\n")

    def _write_v1_prior_information(self, f: TextIO) -> None:
        """Write prior information section (placeholder)."""
        f.write("* prior information\n")

    def _write_v1_pestpp_options(self, f: TextIO) -> None:
        """Write PEST++ options as ``++key(value)`` lines."""
        for option, value in self.pestpp_options.items():
            if isinstance(value, bool):
                value = str(value).lower()
            f.write(f"++{option}({value})\n")

    # ------------------------------------------------------------------ #
    #  Version 2 (keyword / external) writer                              #
    # ------------------------------------------------------------------ #

    def _write_v2(self, filepath: Path, external_dir: str) -> Path:
        """Write a v2 keyword/external PEST++ control file.

        Creates external CSV files for parameter data, observation data,
        parameter groups, and model I/O, then writes a compact PST that
        references them.
        """
        pst_dir = filepath.parent
        ext_abs = pst_dir / external_dir
        ext_abs.mkdir(parents=True, exist_ok=True)

        # Write external CSV files
        par_csv = self._write_csv(
            ext_abs / "par_data.csv",
            [p.to_csv_dict() for p in self.parameters],
            [
                "parnme",
                "partrans",
                "parchglim",
                "parval1",
                "parlbnd",
                "parubnd",
                "pargp",
                "scale",
                "offset",
                "dercom",
                "partied",
            ],
        )
        if not self.obs_csv_files:
            obs_csv = self._write_csv(
                ext_abs / "obs_data.csv",
                [o.to_csv_dict() for o in self.observations],
                ["obsnme", "obsval", "weight", "obgnme"],
            )
        pgrp_csv = self._write_csv(
            ext_abs / "par_groups.csv",
            [
                {
                    "pargpnme": name,
                    "inctyp": opts.get("inctyp", "relative"),
                    "derinc": str(opts.get("derinc", 0.01)),
                    "derinclb": str(opts.get("derinclb", 0.0)),
                    "forcen": opts.get("forcen", "switch"),
                    "derincmul": str(opts.get("derincmul", 2.0)),
                    "dermthd": opts.get("dermthd", "parabolic"),
                }
                for name, opts in self.parameter_groups.items()
            ],
            ["pargpnme", "inctyp", "derinc", "derinclb", "forcen", "derincmul", "dermthd"],
        )
        mi_csv = self._write_csv(
            ext_abs / "model_input.csv",
            [
                {"pest_file": str(t.template_path), "model_file": str(t.input_path)}
                for t in self.template_files
            ],
            ["pest_file", "model_file"],
        )
        mo_csv = self._write_csv(
            ext_abs / "model_output.csv",
            [
                {"pest_file": str(i.instruction_path), "model_file": str(i.output_path)}
                for i in self.instruction_files
            ],
            ["pest_file", "model_file"],
        )

        # Build relative paths (forward slashes work on all platforms)
        def _rel(csv_path: Path) -> str:
            return str(csv_path.relative_to(pst_dir)).replace("\\", "/")

        # Write the PST
        with open(filepath, "w") as f:
            f.write("pcf\n")
            f.write("* control data keyword\n")

            # pestmode
            f.write(f"pestmode {self.control_data.get('pestmode', 'estimation')}\n")
            f.write("\n")

            # SVD settings
            if self.svd_settings:
                f.write("# Singular value decomposition\n")
                for key, val in self.svd_settings.items():
                    f.write(f"{key} {val}\n")
                f.write("\n")

            # Control data options (skip pestmode, already written)
            cd_keys = [k for k in self.control_data if k != "pestmode"]
            if cd_keys:
                f.write("# Algorithm control variables\n")
                for key in cd_keys:
                    f.write(f"{key} {self.control_data[key]}\n")
                f.write("\n")

            # PEST++ options (folded in, no ++ prefix)
            if self.pestpp_options:
                f.write("# PEST++ options\n")
                for option, value in self.pestpp_options.items():
                    if isinstance(value, bool):
                        value = str(value).lower()
                    f.write(f"{option} {value}\n")
                f.write("\n")

            # External sections
            f.write("* parameter groups external\n")
            f.write(f"{_rel(pgrp_csv)}\n")
            f.write("\n")

            f.write("* parameter data external\n")
            f.write(f"{_rel(par_csv)}\n")
            f.write("\n")

            f.write("* observation data external\n")
            if self.obs_csv_files:
                for obs_file in self.obs_csv_files:
                    f.write(f"{obs_file}\n")
            else:
                f.write(f"{_rel(obs_csv)}\n")
            f.write("\n")

            f.write("* model command line\n")
            f.write(f"{self.model_command}\n")
            f.write("\n")

            f.write("* model input external\n")
            f.write(f"{_rel(mi_csv)}\n")
            f.write("\n")

            f.write("* model output external\n")
            f.write(f"{_rel(mo_csv)}\n")

        return filepath

    @staticmethod
    def _write_csv(filepath: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
        """Write a list of dicts to a CSV file and return the path."""
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return filepath

    # ------------------------------------------------------------------ #
    #  Static loader: read a v1 PST into a PESTInterface                  #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_pst(cls, filepath: Path | str, model_dir: Path | str | None = None) -> PESTInterface:
        """Read a v1 PEST control file and populate a :class:`PESTInterface`.

        This enables round-tripping: read a v1 PST, then write it back
        as v2 (or v1 with modifications).

        Parameters
        ----------
        filepath : Path | str
            Path to the ``.pst`` file.
        model_dir : Path | str | None
            Model directory.  Defaults to the directory containing the PST.

        Returns
        -------
        PESTInterface
            Populated interface instance.
        """
        filepath = Path(filepath)
        if model_dir is None:
            model_dir = filepath.parent
        pest_dir = filepath.parent / "pest"

        with open(filepath) as f:
            lines = [line.rstrip("\n").rstrip("\r") for line in f]

        # Find section indices
        sections: dict[str, int] = {}
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("*"):
                sections[stripped] = i

        # Parse control data (basic -- extract noptmax and pestmode)
        cd: dict[str, Any] = {}
        ctrl_idx = sections.get("* control data", -1)
        if ctrl_idx >= 0:
            ctrl_lines = []
            next_sect = min((v for v in sections.values() if v > ctrl_idx), default=len(lines))
            for j in range(ctrl_idx + 1, next_sect):
                stripped = lines[j].strip()
                if stripped:
                    ctrl_lines.append(stripped)
            # Line 0: RSTFLE PESTMODE
            if ctrl_lines:
                parts = ctrl_lines[0].split()
                if len(parts) >= 2:
                    cd["pestmode"] = parts[-1]
            # Line 3: RLAMBDA1 RLAMFAC PHIRATSUF PHIREDLAM NUMLAM ...
            if len(ctrl_lines) > 3:
                parts = ctrl_lines[3].split()
                keys = ["rlambda1", "rlamfac", "phiratsuf", "phiredlam", "numlam", "jacupdate"]
                for k, v in zip(keys, parts, strict=False):
                    cd[k] = v
            # Line 4: RELPARMAX FACPARMAX FACORIG
            if len(ctrl_lines) > 4:
                parts = ctrl_lines[4].split()
                for k, v in zip(["relparmax", "facparmax", "facorig"], parts, strict=False):
                    cd[k] = v
            # Line 5: PHIREDSWH
            if len(ctrl_lines) > 5:
                parts = ctrl_lines[5].split()
                if parts:
                    cd["phiredswh"] = parts[0]
            # Line 6: NOPTMAX PHIREDSTP NPHISTP NPHINORED RELPARSTP NRELPAR
            if len(ctrl_lines) > 6:
                parts = ctrl_lines[6].split()
                keys6 = ["noptmax", "phiredstp", "nphistp", "nphinored", "relparstp", "nrelpar"]
                for k, v in zip(keys6, parts, strict=False):
                    cd[k] = v
            # Line 7: ICOV ICOR IEIG + optional tokens
            if len(ctrl_lines) > 7:
                parts = ctrl_lines[7].split()
                for k, v in zip(["icov", "icor", "ieig"], parts, strict=False):
                    cd[k] = v

        # Parse SVD
        svd: dict[str, Any] = {}
        svd_idx = sections.get("* singular value decomposition", -1)
        if svd_idx >= 0:
            next_sect = min((v for v in sections.values() if v > svd_idx), default=len(lines))
            svd_lines = [
                lines[j].strip() for j in range(svd_idx + 1, next_sect) if lines[j].strip()
            ]
            if len(svd_lines) >= 2:
                parts = svd_lines[1].split()
                svd["maxsing"] = int(parts[0]) if parts else 1500
                svd["eigthresh"] = float(parts[1]) if len(parts) > 1 else 1e-12

        # Parse parameter groups
        pgrp: dict[str, dict[str, Any]] = {}
        pgrp_idx = sections.get("* parameter groups", -1)
        pdata_idx = sections.get("* parameter data", -1)
        if pgrp_idx >= 0 and pdata_idx >= 0:
            for j in range(pgrp_idx + 1, pdata_idx):
                parts = lines[j].split()
                if len(parts) >= 7:
                    pgrp[parts[0]] = {
                        "inctyp": parts[1],
                        "derinc": float(parts[2]),
                        "derinclb": float(parts[3]),
                        "forcen": parts[4],
                        "derincmul": float(parts[5]),
                        "dermthd": parts[6],
                    }

        # Parse parameter data
        ogrp_idx = sections.get("* observation groups", -1)
        params: list[Parameter] = []
        tied_map: dict[str, str] = {}  # child -> parent
        if pdata_idx >= 0 and ogrp_idx >= 0:
            for j in range(pdata_idx + 1, ogrp_idx):
                parts = lines[j].split()
                if not parts:
                    continue
                if len(parts) == 2:
                    # Tied parent line
                    tied_map[parts[0]] = parts[1]
                elif len(parts) >= 7:
                    transform = parts[1]
                    # Use placeholder for tied params (real parent set below)
                    tied_placeholder = "_pending" if transform == "tied" else ""
                    p = Parameter(
                        name=parts[0],
                        transform=transform,
                        change_limit=parts[2] if len(parts) > 2 else "factor",
                        initial_value=float(parts[3]),
                        lower_bound=float(parts[4]),
                        upper_bound=float(parts[5]),
                        group=parts[6],
                        scale=float(parts[7]) if len(parts) > 7 else 1.0,
                        offset=float(parts[8]) if len(parts) > 8 else 0.0,
                        dercom=int(parts[9]) if len(parts) > 9 else 1,
                        tied_to=tied_placeholder,
                    )
                    params.append(p)
            # Resolve tied parent names from the two-column lines
            for p in params:
                if p.name in tied_map:
                    p.tied_to = tied_map[p.name]

        # Parse observation groups
        odata_idx = sections.get("* observation data", -1)
        obs_groups: dict[str, ObservationGroup] = {}
        if ogrp_idx >= 0 and odata_idx >= 0:
            for j in range(ogrp_idx + 1, odata_idx):
                name = lines[j].strip()
                if name:
                    obs_groups[name] = ObservationGroup(name=name)

        # Parse observation data
        mcmd_idx = sections.get("* model command line", -1)
        obs_list: list[Observation] = []
        if odata_idx >= 0 and mcmd_idx >= 0:
            for j in range(odata_idx + 1, mcmd_idx):
                parts = lines[j].split()
                if len(parts) >= 4:
                    o = Observation(
                        name=parts[0],
                        value=float(parts[1]),
                        weight=float(parts[2]),
                        group=parts[3],
                    )
                    obs_list.append(o)
                    if parts[3] not in obs_groups:
                        obs_groups[parts[3]] = ObservationGroup(name=parts[3])
                    obs_groups[parts[3]].observations.append(o)

        # Parse model command
        model_cmd = "python forward_run.py"
        mio_idx = sections.get("* model input/output", -1)
        if mcmd_idx >= 0 and mio_idx >= 0:
            for j in range(mcmd_idx + 1, mio_idx):
                stripped = lines[j].strip()
                if stripped:
                    model_cmd = stripped
                    break

        # Parse model I/O + ++ options
        tpl_list: list[TemplateFile] = []
        ins_list: list[InstructionFile] = []
        pp_opts: dict[str, Any] = {}
        if mio_idx >= 0:
            import re

            for j in range(mio_idx + 1, len(lines)):
                stripped = lines[j].strip()
                if not stripped:
                    continue
                m = re.match(r"\+\+(\w+)\((.+)\)", stripped)
                if m:
                    pp_opts[m.group(1)] = m.group(2)
                else:
                    parts = stripped.split()
                    if len(parts) >= 2:
                        pest_file = parts[0]
                        model_file = parts[1]
                        if ".tpl" in pest_file.lower():
                            tpl_list.append(
                                TemplateFile(
                                    template_path=Path(pest_file),
                                    input_path=Path(model_file),
                                )
                            )
                        elif ".ins" in pest_file.lower():
                            ins_list.append(
                                InstructionFile(
                                    instruction_path=Path(pest_file),
                                    output_path=Path(model_file),
                                )
                            )

        instance = cls(
            model_dir=Path(model_dir),
            case_name=filepath.stem,
            pest_dir=pest_dir,
            parameters=params,
            parameter_groups=pgrp,
            observations=obs_list,
            observation_groups=obs_groups,
            template_files=tpl_list,
            instruction_files=ins_list,
            model_command=model_cmd,
            pestpp_options=pp_opts,
            control_data=cd,
            svd_settings=svd,
        )
        return instance

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
