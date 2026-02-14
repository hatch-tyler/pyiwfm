"""PEST++ instruction file generation for IWFM models.

This module provides IWFM-aware instruction file generation for PEST++
calibration and uncertainty analysis. It understands IWFM output file
formats and generates appropriate instruction files for extracting
simulated values.

Instruction files (.ins) tell PEST++ how to read model output files
to extract simulated observation values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

from pyiwfm.runner.pest import InstructionFile
from pyiwfm.runner.pest_observations import (
    IWFMObservationType,
    IWFMObservation,
)

if TYPE_CHECKING:
    from pyiwfm.runner.pest_obs_manager import IWFMObservationManager


@dataclass
class OutputFileFormat:
    """Describes the format of an IWFM output file.

    Attributes
    ----------
    name : str
        Format name (e.g., "hydrograph", "budget").
    header_lines : int
        Number of header lines to skip.
    time_column : int
        Column containing timestamp (1-based).
    time_format : str
        strftime format for parsing timestamps.
    value_columns : dict[str, int]
        Mapping of variable names to column indices.
    delimiter : str
        Column delimiter (whitespace, comma, etc.).
    """

    name: str
    header_lines: int = 1
    time_column: int = 1
    time_format: str = "%m/%d/%Y"
    value_columns: dict[str, int] = field(default_factory=dict)
    delimiter: str = "whitespace"


# Standard IWFM output file formats
IWFM_OUTPUT_FORMATS = {
    "head_hydrograph": OutputFileFormat(
        name="head_hydrograph",
        header_lines=3,
        time_column=1,
        time_format="%m/%d/%Y_%H:%M",
        value_columns={"head": 2},
    ),
    "stream_hydrograph": OutputFileFormat(
        name="stream_hydrograph",
        header_lines=3,
        time_column=1,
        time_format="%m/%d/%Y_%H:%M",
        value_columns={"flow": 2, "stage": 3},
    ),
    "gw_budget": OutputFileFormat(
        name="gw_budget",
        header_lines=4,
        time_column=1,
        time_format="%m/%d/%Y",
        value_columns={},  # Dynamic based on budget components
    ),
    "stream_budget": OutputFileFormat(
        name="stream_budget",
        header_lines=4,
        time_column=1,
        time_format="%m/%d/%Y",
        value_columns={},
    ),
    "lake_budget": OutputFileFormat(
        name="lake_budget",
        header_lines=4,
        time_column=1,
        time_format="%m/%d/%Y",
        value_columns={},
    ),
    "subsidence": OutputFileFormat(
        name="subsidence",
        header_lines=3,
        time_column=1,
        time_format="%m/%d/%Y_%H:%M",
        value_columns={"subsidence": 2},
    ),
}


class IWFMInstructionManager:
    """Generates PEST++ instruction files for IWFM output files.

    This class understands IWFM output file formats and generates
    appropriate instruction files for extracting simulated values.

    Supports:
    - Head hydrograph files
    - Stream flow/stage hydrographs
    - Budget output files (GW, stream, lake, root zone)
    - Subsidence output files

    Parameters
    ----------
    model : Any
        IWFM model instance (optional).
    observation_manager : IWFMObservationManager | None
        Observation manager containing observation definitions.
    output_dir : Path | str
        Directory for output instruction files.
    marker : str
        Marker character for instructions (default: '@').

    Examples
    --------
    >>> im = IWFMInstructionManager(
    ...     observation_manager=om,
    ...     output_dir="pest/instructions",
    ... )
    >>> ins = im.generate_head_instructions(
    ...     output_file="GW_Heads.out",
    ...     wells=["W1", "W2", "W3"],
    ... )
    """

    def __init__(
        self,
        model: Any = None,
        observation_manager: "IWFMObservationManager | None" = None,
        output_dir: Path | str | None = None,
        marker: str = "@",
    ):
        """Initialize the instruction manager.

        Parameters
        ----------
        model : Any
            IWFM model instance (optional).
        observation_manager : IWFMObservationManager | None
            Observation manager with defined observations.
        output_dir : Path | str | None
            Output directory for instructions.
        marker : str
            PEST instruction marker character.
        """
        self.model = model
        self.om = observation_manager
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.marker = marker
        self._instructions: list[InstructionFile] = []

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Head Observations
    # =========================================================================

    def generate_head_instructions(
        self,
        output_file: Path | str,
        wells: list[str] | None = None,
        observations: list[IWFMObservation] | None = None,
        times: list[datetime] | None = None,
        header_lines: int = 3,
        time_column: int = 1,
        value_column: int = 2,
        time_format: str = "%m/%d/%Y_%H:%M",
        instruction_file: Path | str | None = None,
    ) -> InstructionFile:
        """Generate instructions for head hydrograph output.

        Creates an instruction file for reading groundwater head
        values from an IWFM hydrograph output file.

        Parameters
        ----------
        output_file : Path | str
            Path to the IWFM head output file.
        wells : list[str] | None
            Well IDs to extract. If None, uses observations.
        observations : list[IWFMObservation] | None
            Observations to extract. If None, uses observation manager.
        times : list[datetime] | None
            Times to extract. If None, uses observation times.
        header_lines : int
            Number of header lines in output file.
        time_column : int
            Column containing timestamp (1-based).
        value_column : int
            Column containing head values (1-based).
        time_format : str
            strftime format for timestamps in output file.
        instruction_file : Path | str | None
            Output instruction file path.

        Returns
        -------
        InstructionFile
            The created instruction file.
        """
        output_file = Path(output_file)

        # Get observations from manager if not provided
        if observations is None and self.om is not None:
            observations = self.om.get_observations_by_type(IWFMObservationType.HEAD)

        if not observations:
            raise ValueError("No head observations provided")

        # Generate instruction file path
        if instruction_file is None:
            instruction_file = self.output_dir / f"{output_file.stem}.ins"
        else:
            instruction_file = Path(instruction_file)

        # Build instruction content
        lines = [f"pif {self.marker}"]

        # Skip header lines
        if header_lines > 0:
            lines.append(f"l{header_lines}")

        # Group observations by time for efficiency
        obs_by_time = {}
        for obs in observations:
            if obs.datetime is not None:
                time_str = obs.datetime.strftime(time_format)
                if time_str not in obs_by_time:
                    obs_by_time[time_str] = []
                obs_by_time[time_str].append(obs)

        # Generate instructions to search for each time
        obs_names = []
        for time_str in sorted(obs_by_time.keys()):
            time_obs = obs_by_time[time_str]

            # Search for line with this timestamp
            lines.append(f"{self.marker}{time_str}{self.marker}")

            # Read values from specified column
            # For multiple wells, assume they are on consecutive lines
            for i, obs in enumerate(time_obs):
                if i > 0:
                    lines.append("l1")  # Move to next line

                # Build whitespace-delimited read instruction
                instruction = self._build_read_instruction(
                    value_column, obs.name
                )
                lines.append(instruction)
                obs_names.append(obs.name)

        # Write instruction file
        instruction_file.write_text("\n".join(lines))

        ins = InstructionFile(
            instruction_path=instruction_file,
            output_path=output_file,
            marker=self.marker,
            observations=obs_names,
        )
        self._instructions.append(ins)
        return ins

    def generate_head_instructions_by_well(
        self,
        output_files: dict[str, Path | str],
        observations: list[IWFMObservation] | None = None,
        header_lines: int = 3,
        value_column: int = 2,
        time_format: str = "%m/%d/%Y_%H:%M",
    ) -> list[InstructionFile]:
        """Generate instructions for per-well head output files.

        IWFM can output head hydrographs to separate files per well.
        This method handles that case.

        Parameters
        ----------
        output_files : dict[str, Path | str]
            Mapping of well IDs to output file paths.
        observations : list[IWFMObservation] | None
            Observations to extract.
        header_lines : int
            Number of header lines.
        value_column : int
            Column containing values.
        time_format : str
            Timestamp format.

        Returns
        -------
        list[InstructionFile]
            List of created instruction files.
        """
        if observations is None and self.om is not None:
            observations = self.om.get_observations_by_type(IWFMObservationType.HEAD)

        # Group observations by well
        obs_by_well = {}
        for obs in observations:
            well_id = obs.metadata.get("well_id")
            if well_id and well_id in output_files:
                if well_id not in obs_by_well:
                    obs_by_well[well_id] = []
                obs_by_well[well_id].append(obs)

        instructions = []
        for well_id, well_obs in obs_by_well.items():
            output_file = Path(output_files[well_id])
            ins = self.generate_head_instructions(
                output_file=output_file,
                observations=well_obs,
                header_lines=header_lines,
                value_column=value_column,
                time_format=time_format,
                instruction_file=self.output_dir / f"head_{well_id}.ins",
            )
            instructions.append(ins)

        return instructions

    # =========================================================================
    # Stream Observations
    # =========================================================================

    def generate_flow_instructions(
        self,
        output_file: Path | str,
        gages: list[str] | None = None,
        observations: list[IWFMObservation] | None = None,
        variable: str = "flow",
        header_lines: int = 3,
        time_column: int = 1,
        value_column: int = 2,
        time_format: str = "%m/%d/%Y_%H:%M",
        instruction_file: Path | str | None = None,
    ) -> InstructionFile:
        """Generate instructions for streamflow output.

        Creates an instruction file for reading stream flow or stage
        values from an IWFM stream hydrograph output file.

        Parameters
        ----------
        output_file : Path | str
            Path to the IWFM stream output file.
        gages : list[str] | None
            Gage IDs to extract.
        observations : list[IWFMObservation] | None
            Observations to extract.
        variable : str
            Variable to extract: "flow" or "stage".
        header_lines : int
            Number of header lines.
        time_column : int
            Column containing timestamp.
        value_column : int
            Column containing values.
        time_format : str
            Timestamp format.
        instruction_file : Path | str | None
            Output instruction file path.

        Returns
        -------
        InstructionFile
            The created instruction file.
        """
        output_file = Path(output_file)

        # Get observations
        if observations is None and self.om is not None:
            if variable == "flow":
                observations = self.om.get_observations_by_type(
                    IWFMObservationType.STREAM_FLOW
                )
            else:
                observations = self.om.get_observations_by_type(
                    IWFMObservationType.STREAM_STAGE
                )

        if not observations:
            raise ValueError(f"No {variable} observations provided")

        # Generate instruction file path
        if instruction_file is None:
            instruction_file = self.output_dir / f"{output_file.stem}_{variable}.ins"
        else:
            instruction_file = Path(instruction_file)

        # Build instruction content
        lines = [f"pif {self.marker}"]

        if header_lines > 0:
            lines.append(f"l{header_lines}")

        # Group observations by time
        obs_by_time = {}
        for obs in observations:
            if obs.datetime is not None:
                time_str = obs.datetime.strftime(time_format)
                if time_str not in obs_by_time:
                    obs_by_time[time_str] = []
                obs_by_time[time_str].append(obs)

        obs_names = []
        for time_str in sorted(obs_by_time.keys()):
            time_obs = obs_by_time[time_str]

            lines.append(f"{self.marker}{time_str}{self.marker}")

            for i, obs in enumerate(time_obs):
                if i > 0:
                    lines.append("l1")

                instruction = self._build_read_instruction(value_column, obs.name)
                lines.append(instruction)
                obs_names.append(obs.name)

        instruction_file.write_text("\n".join(lines))

        ins = InstructionFile(
            instruction_path=instruction_file,
            output_path=output_file,
            marker=self.marker,
            observations=obs_names,
        )
        self._instructions.append(ins)
        return ins

    def generate_gain_loss_instructions(
        self,
        output_file: Path | str,
        reaches: list[int] | None = None,
        observations: list[IWFMObservation] | None = None,
        header_lines: int = 4,
        time_column: int = 1,
        reach_column: int = 2,
        value_column: int = 3,
        time_format: str = "%m/%d/%Y",
        instruction_file: Path | str | None = None,
    ) -> InstructionFile:
        """Generate instructions for stream gain/loss output.

        Parameters
        ----------
        output_file : Path | str
            Path to gain/loss output file.
        reaches : list[int] | None
            Reach IDs to extract.
        observations : list[IWFMObservation] | None
            Observations to extract.
        header_lines : int
            Number of header lines.
        time_column : int
            Column containing timestamp.
        reach_column : int
            Column containing reach ID.
        value_column : int
            Column containing values.
        time_format : str
            Timestamp format.
        instruction_file : Path | str | None
            Output instruction file path.

        Returns
        -------
        InstructionFile
            The created instruction file.
        """
        output_file = Path(output_file)

        if observations is None and self.om is not None:
            observations = self.om.get_observations_by_type(
                IWFMObservationType.STREAM_GAIN_LOSS
            )

        if not observations:
            raise ValueError("No gain/loss observations provided")

        if instruction_file is None:
            instruction_file = self.output_dir / f"{output_file.stem}_sgl.ins"
        else:
            instruction_file = Path(instruction_file)

        lines = [f"pif {self.marker}"]

        if header_lines > 0:
            lines.append(f"l{header_lines}")

        # For gain/loss, we need to search for both time and reach
        obs_names = []
        for obs in sorted(observations, key=lambda o: (o.datetime or datetime.min, o.metadata.get("reach_id", 0))):
            if obs.datetime is None:
                continue

            time_str = obs.datetime.strftime(time_format)
            reach_id = obs.metadata.get("reach_id")

            if reach_id is not None:
                # Search for line with this time and reach
                lines.append(f"{self.marker}{time_str}{self.marker}")
                # Then search for reach ID on same or subsequent lines
                lines.append(f"{self.marker}{reach_id}{self.marker}")

            instruction = self._build_read_instruction(value_column, obs.name)
            lines.append(instruction)
            obs_names.append(obs.name)

        instruction_file.write_text("\n".join(lines))

        ins = InstructionFile(
            instruction_path=instruction_file,
            output_path=output_file,
            marker=self.marker,
            observations=obs_names,
        )
        self._instructions.append(ins)
        return ins

    # =========================================================================
    # Budget Observations
    # =========================================================================

    def generate_budget_instructions(
        self,
        budget_file: Path | str,
        budget_type: str,
        components: list[str] | None = None,
        locations: list[int] | None = None,
        observations: list[IWFMObservation] | None = None,
        header_lines: int = 4,
        time_column: int = 1,
        time_format: str = "%m/%d/%Y",
        instruction_file: Path | str | None = None,
    ) -> InstructionFile:
        """Generate instructions for budget output.

        Creates instruction file for reading water budget components
        from IWFM budget output files.

        Parameters
        ----------
        budget_file : Path | str
            Path to budget output file.
        budget_type : str
            Budget type: "gw", "stream", "lake", "rootzone".
        components : list[str] | None
            Budget components to extract.
        locations : list[int] | None
            Location IDs (subregions, reaches, etc.).
        observations : list[IWFMObservation] | None
            Observations to extract.
        header_lines : int
            Number of header lines.
        time_column : int
            Column containing timestamp.
        time_format : str
            Timestamp format.
        instruction_file : Path | str | None
            Output instruction file path.

        Returns
        -------
        InstructionFile
            The created instruction file.
        """
        budget_file = Path(budget_file)

        # Map budget type to observation type
        budget_obs_types = {
            "gw": IWFMObservationType.GW_BUDGET,
            "stream": IWFMObservationType.STREAM_BUDGET,
            "lake": IWFMObservationType.LAKE_BUDGET,
            "rootzone": IWFMObservationType.ROOTZONE_BUDGET,
        }

        if budget_type not in budget_obs_types:
            raise ValueError(f"Invalid budget type: {budget_type}")

        if observations is None and self.om is not None:
            observations = self.om.get_observations_by_type(
                budget_obs_types[budget_type]
            )

        if not observations:
            raise ValueError(f"No {budget_type} budget observations provided")

        if instruction_file is None:
            instruction_file = self.output_dir / f"{budget_file.stem}_{budget_type}.ins"
        else:
            instruction_file = Path(instruction_file)

        lines = [f"pif {self.marker}"]

        if header_lines > 0:
            lines.append(f"l{header_lines}")

        # Budget files are complex - need to parse based on structure
        # This is a simplified approach using search markers
        obs_names = []
        for obs in observations:
            if obs.datetime is None:
                continue

            time_str = obs.datetime.strftime(time_format)
            component = obs.metadata.get("component", "")
            location_id = obs.metadata.get("location_id")

            # Search for time
            lines.append(f"{self.marker}{time_str}{self.marker}")

            # If component specified, search for it
            if component:
                lines.append(f"{self.marker}{component}{self.marker}")

            # Read value (budget files vary, so use general approach)
            # Read next numeric value after markers
            lines.append(f"!{obs.name}!")
            obs_names.append(obs.name)

        instruction_file.write_text("\n".join(lines))

        ins = InstructionFile(
            instruction_path=instruction_file,
            output_path=budget_file,
            marker=self.marker,
            observations=obs_names,
        )
        self._instructions.append(ins)
        return ins

    # =========================================================================
    # Lake Observations
    # =========================================================================

    def generate_lake_instructions(
        self,
        output_file: Path | str,
        lakes: list[int] | None = None,
        observations: list[IWFMObservation] | None = None,
        variable: str = "level",
        header_lines: int = 3,
        time_column: int = 1,
        value_column: int = 2,
        time_format: str = "%m/%d/%Y_%H:%M",
        instruction_file: Path | str | None = None,
    ) -> InstructionFile:
        """Generate instructions for lake output.

        Parameters
        ----------
        output_file : Path | str
            Path to lake output file.
        lakes : list[int] | None
            Lake IDs to extract.
        observations : list[IWFMObservation] | None
            Observations to extract.
        variable : str
            Variable: "level" or "storage".
        header_lines : int
            Number of header lines.
        time_column : int
            Column containing timestamp.
        value_column : int
            Column containing values.
        time_format : str
            Timestamp format.
        instruction_file : Path | str | None
            Output instruction file path.

        Returns
        -------
        InstructionFile
            The created instruction file.
        """
        output_file = Path(output_file)

        if observations is None and self.om is not None:
            if variable == "level":
                observations = self.om.get_observations_by_type(
                    IWFMObservationType.LAKE_LEVEL
                )
            else:
                observations = self.om.get_observations_by_type(
                    IWFMObservationType.LAKE_STORAGE
                )

        if not observations:
            raise ValueError(f"No lake {variable} observations provided")

        if instruction_file is None:
            instruction_file = self.output_dir / f"{output_file.stem}_lake_{variable}.ins"
        else:
            instruction_file = Path(instruction_file)

        lines = [f"pif {self.marker}"]

        if header_lines > 0:
            lines.append(f"l{header_lines}")

        obs_by_time = {}
        for obs in observations:
            if obs.datetime is not None:
                time_str = obs.datetime.strftime(time_format)
                if time_str not in obs_by_time:
                    obs_by_time[time_str] = []
                obs_by_time[time_str].append(obs)

        obs_names = []
        for time_str in sorted(obs_by_time.keys()):
            time_obs = obs_by_time[time_str]

            lines.append(f"{self.marker}{time_str}{self.marker}")

            for i, obs in enumerate(time_obs):
                if i > 0:
                    lines.append("l1")

                instruction = self._build_read_instruction(value_column, obs.name)
                lines.append(instruction)
                obs_names.append(obs.name)

        instruction_file.write_text("\n".join(lines))

        ins = InstructionFile(
            instruction_path=instruction_file,
            output_path=output_file,
            marker=self.marker,
            observations=obs_names,
        )
        self._instructions.append(ins)
        return ins

    # =========================================================================
    # Subsidence Observations
    # =========================================================================

    def generate_subsidence_instructions(
        self,
        output_file: Path | str,
        observations: list[IWFMObservation] | None = None,
        header_lines: int = 3,
        time_column: int = 1,
        value_column: int = 2,
        time_format: str = "%m/%d/%Y_%H:%M",
        instruction_file: Path | str | None = None,
    ) -> InstructionFile:
        """Generate instructions for subsidence output.

        Parameters
        ----------
        output_file : Path | str
            Path to subsidence output file.
        observations : list[IWFMObservation] | None
            Observations to extract.
        header_lines : int
            Number of header lines.
        time_column : int
            Column containing timestamp.
        value_column : int
            Column containing values.
        time_format : str
            Timestamp format.
        instruction_file : Path | str | None
            Output instruction file path.

        Returns
        -------
        InstructionFile
            The created instruction file.
        """
        output_file = Path(output_file)

        if observations is None and self.om is not None:
            observations = self.om.get_observations_by_type(
                IWFMObservationType.SUBSIDENCE
            )

        if not observations:
            raise ValueError("No subsidence observations provided")

        if instruction_file is None:
            instruction_file = self.output_dir / f"{output_file.stem}_sub.ins"
        else:
            instruction_file = Path(instruction_file)

        lines = [f"pif {self.marker}"]

        if header_lines > 0:
            lines.append(f"l{header_lines}")

        obs_names = []
        for obs in sorted(observations, key=lambda o: o.datetime or datetime.min):
            if obs.datetime is None:
                continue

            time_str = obs.datetime.strftime(time_format)
            lines.append(f"{self.marker}{time_str}{self.marker}")

            instruction = self._build_read_instruction(value_column, obs.name)
            lines.append(instruction)
            obs_names.append(obs.name)

        instruction_file.write_text("\n".join(lines))

        ins = InstructionFile(
            instruction_path=instruction_file,
            output_path=output_file,
            marker=self.marker,
            observations=obs_names,
        )
        self._instructions.append(ins)
        return ins

    # =========================================================================
    # Generic / Custom Instructions
    # =========================================================================

    def generate_custom_instructions(
        self,
        output_file: Path | str,
        observations: list[tuple[str, str, int]],
        header_lines: int = 0,
        instruction_file: Path | str | None = None,
    ) -> InstructionFile:
        """Generate custom instructions for any output file.

        This method allows creating instructions for non-standard
        output file formats.

        Parameters
        ----------
        output_file : Path | str
            Path to output file.
        observations : list[tuple[str, str, int]]
            List of (obs_name, search_string, value_column) tuples.
            search_string is a marker to search for in the file.
        header_lines : int
            Number of header lines to skip.
        instruction_file : Path | str | None
            Output instruction file path.

        Returns
        -------
        InstructionFile
            The created instruction file.
        """
        output_file = Path(output_file)

        if instruction_file is None:
            instruction_file = self.output_dir / f"{output_file.stem}_custom.ins"
        else:
            instruction_file = Path(instruction_file)

        lines = [f"pif {self.marker}"]

        if header_lines > 0:
            lines.append(f"l{header_lines}")

        obs_names = []
        for obs_name, search_string, value_column in observations:
            lines.append(f"{self.marker}{search_string}{self.marker}")
            instruction = self._build_read_instruction(value_column, obs_name)
            lines.append(instruction)
            obs_names.append(obs_name)

        instruction_file.write_text("\n".join(lines))

        ins = InstructionFile(
            instruction_path=instruction_file,
            output_path=output_file,
            marker=self.marker,
            observations=obs_names,
        )
        self._instructions.append(ins)
        return ins

    def generate_fixed_format_instructions(
        self,
        output_file: Path | str,
        observations: list[tuple[str, int, int, int]],
        header_lines: int = 0,
        instruction_file: Path | str | None = None,
    ) -> InstructionFile:
        """Generate instructions for fixed-format output file.

        For output files with fixed-width columns rather than
        delimiter-separated values.

        Parameters
        ----------
        output_file : Path | str
            Path to output file.
        observations : list[tuple[str, int, int, int]]
            List of (obs_name, line_number, start_column, end_column) tuples.
            Line numbers are 1-based (after header).
            Columns are 1-based character positions.
        header_lines : int
            Number of header lines.
        instruction_file : Path | str | None
            Output instruction file path.

        Returns
        -------
        InstructionFile
            The created instruction file.
        """
        output_file = Path(output_file)

        if instruction_file is None:
            instruction_file = self.output_dir / f"{output_file.stem}_fixed.ins"
        else:
            instruction_file = Path(instruction_file)

        lines = [f"pif {self.marker}"]

        # Sort by line number
        sorted_obs = sorted(observations, key=lambda x: x[1])

        current_line = 0
        obs_names = []

        for obs_name, line_num, start_col, end_col in sorted_obs:
            # Skip to correct line
            lines_to_skip = line_num - current_line - 1
            if current_line == 0:
                lines_to_skip += header_lines

            if lines_to_skip > 0:
                lines.append(f"l{lines_to_skip}")
            current_line = line_num

            # Use fixed-format read: [start:end]
            # PEST format: [col1:col2] for fixed columns
            lines.append(f"[{start_col}:{end_col}] !{obs_name}!")
            obs_names.append(obs_name)

        instruction_file.write_text("\n".join(lines))

        ins = InstructionFile(
            instruction_path=instruction_file,
            output_path=output_file,
            marker=self.marker,
            observations=obs_names,
        )
        self._instructions.append(ins)
        return ins

    # =========================================================================
    # Batch Generation
    # =========================================================================

    def generate_all_instructions(
        self,
        output_files: dict[str, Path | str] | None = None,
    ) -> list[InstructionFile]:
        """Generate all required instruction files based on observations.

        Automatically generates instructions for all observations in the
        observation manager.

        Parameters
        ----------
        output_files : dict[str, Path | str] | None
            Mapping of observation type values to output file paths.
            E.g., {"head": "GW_Heads.out", "flow": "StreamFlow.out"}

        Returns
        -------
        list[InstructionFile]
            List of created instruction files.
        """
        if self.om is None:
            raise ValueError("Observation manager required for batch generation")

        instructions = []
        output_files = output_files or {}

        # Generate instructions for each observation type
        type_methods = {
            IWFMObservationType.HEAD: (
                "head", self.generate_head_instructions
            ),
            IWFMObservationType.STREAM_FLOW: (
                "flow", lambda f, **kw: self.generate_flow_instructions(f, variable="flow", **kw)
            ),
            IWFMObservationType.STREAM_STAGE: (
                "stage", lambda f, **kw: self.generate_flow_instructions(f, variable="stage", **kw)
            ),
            IWFMObservationType.LAKE_LEVEL: (
                "lake_level", lambda f, **kw: self.generate_lake_instructions(f, variable="level", **kw)
            ),
            IWFMObservationType.LAKE_STORAGE: (
                "lake_storage", lambda f, **kw: self.generate_lake_instructions(f, variable="storage", **kw)
            ),
            IWFMObservationType.SUBSIDENCE: (
                "subsidence", self.generate_subsidence_instructions
            ),
        }

        for obs_type, (key, method) in type_methods.items():
            observations = self.om.get_observations_by_type(obs_type)
            if observations and key in output_files:
                try:
                    ins = method(output_files[key], observations=observations)
                    instructions.append(ins)
                except Exception as e:
                    # Log warning but continue
                    pass

        self._instructions = instructions
        return instructions

    # =========================================================================
    # Utilities
    # =========================================================================

    def _build_read_instruction(self, column: int, obs_name: str) -> str:
        """Build a whitespace-delimited read instruction.

        Parameters
        ----------
        column : int
            Column to read from (1-based).
        obs_name : str
            Observation name.

        Returns
        -------
        str
            PEST instruction string.
        """
        # w = skip whitespace-delimited token
        # !name! = read into observation
        parts = ["w"] * (column - 1) + [f"!{obs_name}!"]
        return " ".join(parts)

    def get_all_instructions(self) -> list[InstructionFile]:
        """Get all created instruction files.

        Returns
        -------
        list[InstructionFile]
            All instruction files created by this manager.
        """
        return list(self._instructions)

    def clear_instructions(self) -> None:
        """Clear all created instructions."""
        self._instructions.clear()

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"IWFMInstructionManager(output_dir='{self.output_dir}', "
            f"n_instructions={len(self._instructions)})"
        )
