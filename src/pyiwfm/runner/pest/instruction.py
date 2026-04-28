"""InstructionFile — extracted from runner/pest.py in v2.0 PR 5.

The class body is verbatim from v1.x ``runner/pest.py``; this module
just gives it a dedicated file so the package no longer has a
1,400-line module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


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
