"""
Simulation control I/O handlers for IWFM model files.

This module provides functions for reading and writing IWFM simulation
control files including the main simulation input file, time stepping,
and output control settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re as _re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TextIO

import numpy as np
from numpy.typing import NDArray

from pyiwfm.core.timeseries import TimeUnit, SimulationPeriod
from pyiwfm.core.exceptions import FileFormatError
from pyiwfm.io.iwfm_reader import (
    COMMENT_CHARS,
    is_comment_line as _is_comment_line,
    next_data_or_empty as _next_data_or_empty,
    resolve_path as _resolve_path_f,
    strip_inline_comment as _strip_comment,
)


def _format_iwfm_datetime(dt: datetime) -> str:
    """Format datetime for IWFM input (MM/DD/YYYY_HH:MM, 16 chars).

    Midnight (00:00) is represented as 24:00 of the previous day.
    """
    from pyiwfm.io.timeseries_ascii import format_iwfm_timestamp
    return format_iwfm_timestamp(dt)


@dataclass
class SimulationConfig:
    """
    Configuration for an IWFM simulation.

    Attributes:
        model_name: Name of the model
        title_lines: Project title lines (up to 3)
        start_date: Simulation start datetime
        end_date: Simulation end datetime
        time_step_length: Length of each time step
        time_step_unit: Unit of time step (DAY, HOUR, etc.)
        restart_flag: Restart option (0=No, 1=Yes)
        output_interval: Output interval (multiple of time step)
        preprocessor_file: Path to preprocessor main file
        binary_preprocessor_file: Path to preprocessor binary output
        groundwater_file: Path to groundwater component file
        streams_file: Path to streams component file
        lakes_file: Path to lakes component file
        rootzone_file: Path to rootzone component file
        unsaturated_zone_file: Path to unsaturated zone component file
        small_watershed_file: Path to small watershed component file
        irrigation_fractions_file: Path to irrigation fractions data file
        supply_adjust_file: Path to supply adjustment specification file
        precipitation_file: Path to precipitation data file
        et_file: Path to evapotranspiration data file
        kc_file: Path to crop/habitat coefficient data file
        output_dir: Directory for output files
        restart_output_flag: Generate restart file (0=No, 1=Yes)
        debug_flag: Debug output level (-1, 0, or 1)
        cache_size: Cache size limit for time series entries
        matrix_solver: Matrix solver option (1=SOR, 2=Conjugate gradient)
        relaxation: Relaxation factor for iterative solver
        max_iterations: Maximum flow convergence iterations
        max_supply_iterations: Maximum supply adjustment iterations
        convergence_tolerance: Flow convergence tolerance (STOPC)
        convergence_volume: Volume convergence tolerance (STOPCVL)
        convergence_supply: Supply adjustment convergence tolerance (STOPCSP)
        supply_adjust_option: Water supply adjustment flag
    """

    model_name: str = "IWFM_Model"
    title_lines: list[str] = field(default_factory=list)
    start_date: datetime = field(default_factory=lambda: datetime(2000, 1, 1))
    end_date: datetime = field(default_factory=lambda: datetime(2000, 12, 31))
    time_step_length: int = 1
    time_step_unit: TimeUnit = TimeUnit.DAY
    restart_flag: int = 0

    # Output control
    output_interval: int = 1
    budget_output_interval: int = 1
    heads_output_interval: int = 1

    # Component files (preprocessor)
    preprocessor_file: Path | None = None
    binary_preprocessor_file: Path | None = None

    # Component files (simulation)
    groundwater_file: Path | None = None
    streams_file: Path | None = None
    lakes_file: Path | None = None
    rootzone_file: Path | None = None
    unsaturated_zone_file: Path | None = None
    small_watershed_file: Path | None = None

    # Additional input data files
    irrigation_fractions_file: Path | None = None
    supply_adjust_file: Path | None = None
    precipitation_file: Path | None = None
    et_file: Path | None = None
    kc_file: Path | None = None

    # Output directory
    output_dir: Path | None = None

    # Processing and debugging options
    restart_output_flag: int = 0
    debug_flag: int = 0
    cache_size: int = 500000

    # Solver settings
    matrix_solver: int = 2
    relaxation: float = 1.0
    max_iterations: int = 50
    max_supply_iterations: int = 50
    convergence_tolerance: float = 1e-6
    convergence_volume: float = 0.0
    convergence_supply: float = 0.001
    supply_adjust_option: int = 0

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_time_steps(self) -> int:
        """Calculate the number of time steps in the simulation."""
        delta = self.time_step_unit.to_timedelta(self.time_step_length)
        duration = self.end_date - self.start_date
        return int(duration.total_seconds() / delta.total_seconds())

    def to_simulation_period(self) -> SimulationPeriod:
        """Convert to SimulationPeriod object."""
        return SimulationPeriod(
            start=self.start_date,
            end=self.end_date,
            time_step_length=self.time_step_length,
            time_step_unit=self.time_step_unit,
        )


@dataclass
class SimulationFileConfig:
    """
    Configuration for simulation file paths.

    Attributes:
        output_dir: Directory for output files
        main_file: Main simulation input file name
        time_series_dir: Directory for time series files
    """

    output_dir: Path
    main_file: str = "simulation.in"
    time_series_dir: str = "timeseries"

    def get_main_file_path(self) -> Path:
        return self.output_dir / self.main_file

    def get_time_series_dir(self) -> Path:
        return self.output_dir / self.time_series_dir


class SimulationWriter:
    """
    Writer for IWFM simulation control files.

    Writes the main simulation input file and related control files.

    Example:
        >>> file_config = SimulationFileConfig(output_dir=Path("./model"))
        >>> writer = SimulationWriter(file_config)
        >>> filepath = writer.write(sim_config)
    """

    def __init__(self, file_config: SimulationFileConfig) -> None:
        """
        Initialize the simulation writer.

        Args:
            file_config: File configuration
        """
        self.file_config = file_config
        file_config.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, config: SimulationConfig, header: str | None = None) -> Path:
        """
        Write the main simulation input file.

        Args:
            config: Simulation configuration
            header: Optional header comment

        Returns:
            Path to written file
        """
        filepath = self.file_config.get_main_file_path()

        with open(filepath, "w") as f:
            self._write_header(f, header, config)
            self._write_time_settings(f, config)
            self._write_component_files(f, config)
            self._write_processing_options(f, config)
            self._write_solver_settings(f, config)
            self._write_supply_adjustment(f, config)
            self._write_output_settings(f, config)

        return filepath

    def _write_header(
        self, f: TextIO, header: str | None, config: SimulationConfig
    ) -> None:
        """Write file header."""
        if header:
            for line in header.strip().split("\n"):
                f.write(f"C  {line}\n")
        else:
            f.write("C  IWFM Simulation Main Input File\n")
            f.write("C  Generated by pyiwfm\n")
            f.write("C\n")
            f.write(f"C  Model: {config.model_name}\n")
            f.write("C\n")

        # Model name
        f.write(f"{config.model_name:<40} / MODEL_NAME\n")
        f.write("C\n")

    def _write_time_settings(self, f: TextIO, config: SimulationConfig) -> None:
        """Write time stepping settings."""
        f.write("C  ==================================================================\n")
        f.write("C  SIMULATION TIME PERIOD\n")
        f.write("C  ==================================================================\n")
        f.write("C\n")

        # Start date
        start_str = _format_iwfm_datetime(config.start_date)
        f.write(f"{start_str:<40} / START_DATE\n")

        # End date
        end_str = _format_iwfm_datetime(config.end_date)
        f.write(f"{end_str:<40} / END_DATE\n")

        # Time step
        f.write(f"{config.time_step_length:<10}                              / TIME_STEP_LENGTH\n")
        f.write(f"{config.time_step_unit.value:<10}                              / TIME_STEP_UNIT\n")

        f.write("C\n")

    def _write_component_files(self, f: TextIO, config: SimulationConfig) -> None:
        """Write component file paths."""
        f.write("C  ==================================================================\n")
        f.write("C  COMPONENT INPUT FILES\n")
        f.write("C  ==================================================================\n")
        f.write("C\n")

        if config.preprocessor_file:
            f.write(f"{str(config.preprocessor_file):<60} / PREPROCESSOR_FILE\n")

        if config.binary_preprocessor_file:
            f.write(f"{str(config.binary_preprocessor_file):<60} / BINARY_PREPROCESSOR_FILE\n")

        if config.groundwater_file:
            f.write(f"{str(config.groundwater_file):<60} / GROUNDWATER_FILE\n")

        if config.streams_file:
            f.write(f"{str(config.streams_file):<60} / STREAMS_FILE\n")

        if config.lakes_file:
            f.write(f"{str(config.lakes_file):<60} / LAKES_FILE\n")

        if config.rootzone_file:
            f.write(f"{str(config.rootzone_file):<60} / ROOTZONE_FILE\n")

        if config.small_watershed_file:
            f.write(f"{str(config.small_watershed_file):<60} / SMALL_WATERSHED_FILE\n")

        if config.unsaturated_zone_file:
            f.write(f"{str(config.unsaturated_zone_file):<60} / UNSATURATED_ZONE_FILE\n")

        if config.irrigation_fractions_file:
            f.write(f"{str(config.irrigation_fractions_file):<60} / IRRIGATION_FRACTIONS_FILE\n")

        if config.supply_adjust_file:
            f.write(f"{str(config.supply_adjust_file):<60} / SUPPLY_ADJUST_FILE\n")

        if config.precipitation_file:
            f.write(f"{str(config.precipitation_file):<60} / PRECIPITATION_FILE\n")

        if config.et_file:
            f.write(f"{str(config.et_file):<60} / ET_FILE\n")

        if config.kc_file:
            f.write(f"{str(config.kc_file):<60} / KC_FILE\n")

        f.write("C\n")

    def _write_solver_settings(self, f: TextIO, config: SimulationConfig) -> None:
        """Write solver settings."""
        f.write("C  ==================================================================\n")
        f.write("C  SOLVER SETTINGS\n")
        f.write("C  ==================================================================\n")
        f.write("C\n")

        f.write(f"{config.matrix_solver:<10}                              / MSOLVE\n")
        f.write(f"{config.relaxation:<14.6f}                          / RELAX\n")
        f.write(f"{config.max_iterations:<10}                              / MXITER\n")
        f.write(f"{config.max_supply_iterations:<10}                              / MXITERSP\n")
        f.write(f"{config.convergence_tolerance:<14.6e}                          / STOPC\n")
        if config.convergence_volume != 0.0:
            f.write(f"{config.convergence_volume:<14.6e}                          / STOPCVL\n")
        f.write(f"{config.convergence_supply:<14.6e}                          / STOPCSP\n")

        f.write("C\n")

    def _write_processing_options(self, f: TextIO, config: SimulationConfig) -> None:
        """Write processing, output, and debugging options."""
        f.write("C  ==================================================================\n")
        f.write("C  PROCESSING AND DEBUG OPTIONS\n")
        f.write("C  ==================================================================\n")
        f.write("C\n")

        f.write(f"{config.restart_output_flag:<10}                              / ISTRT\n")
        f.write(f"{config.debug_flag:<10}                              / KDEB\n")
        f.write(f"{config.cache_size:<10}                              / CACHE\n")

        f.write("C\n")

    def _write_supply_adjustment(self, f: TextIO, config: SimulationConfig) -> None:
        """Write supply adjustment control option."""
        f.write("C  ==================================================================\n")
        f.write("C  SUPPLY ADJUSTMENT\n")
        f.write("C  ==================================================================\n")
        f.write("C\n")

        f.write(f"{config.supply_adjust_option:<10}                              / KOPTDV\n")

    def _write_output_settings(self, f: TextIO, config: SimulationConfig) -> None:
        """Write output control settings."""
        f.write("C  ==================================================================\n")
        f.write("C  OUTPUT CONTROL\n")
        f.write("C  ==================================================================\n")
        f.write("C\n")

        if config.output_dir:
            f.write(f"{str(config.output_dir):<60} / OUTPUT_DIR\n")

        f.write(f"{config.output_interval:<10}                              / OUTPUT_INTERVAL\n")
        f.write(f"{config.budget_output_interval:<10}                              / BUDGET_OUTPUT_INTERVAL\n")
        f.write(f"{config.heads_output_interval:<10}                              / HEADS_OUTPUT_INTERVAL\n")


class SimulationReader:
    """
    Reader for IWFM simulation control files.
    """

    def read(self, filepath: Path | str) -> SimulationConfig:
        """
        Read simulation configuration from main input file.

        Args:
            filepath: Path to simulation input file

        Returns:
            SimulationConfig object
        """
        filepath = Path(filepath)

        config = SimulationConfig()

        with open(filepath, "r") as f:
            line_num = 0

            for line in f:
                line_num += 1
                if _is_comment_line(line):
                    continue

                value, desc = _strip_comment(line)
                desc_upper = desc.upper()

                try:
                    self._parse_config_line(config, value, desc_upper)
                except ValueError as e:
                    raise FileFormatError(
                        f"Error parsing value: '{value}'", line_number=line_num
                    ) from e

        return config

    def _parse_config_line(
        self, config: SimulationConfig, value: str, desc: str
    ) -> None:
        """Parse a single configuration line.

        Handles both the pyiwfm writer format (``START_DATE``,
        ``TIME_STEP_UNIT``, etc.) and the C2VSimFG format (``BDT``,
        ``EDT``, ``UNITT``, ``DELTAT``).
        """
        # --- Identifiers ------------------------------------------------
        if "MODEL_NAME" in desc or desc == "NAME":
            config.model_name = value.strip()

        # --- Dates -------------------------------------------------------
        # C2VSimFG: BDT / EDT;  pyiwfm: START_DATE / END_DATE
        elif desc == "BDT" or ("START" in desc and "DATE" in desc):
            config.start_date = self._parse_datetime(value)
        elif desc == "EDT" or ("END" in desc and "DATE" in desc):
            config.end_date = self._parse_datetime(value)

        # --- Restart flag ------------------------------------------------
        elif desc in ("RESTART", "ISTRT"):
            config.restart_flag = int(value)

        # --- Time step ---------------------------------------------------
        # C2VSimFG combined format: UNITT → value like "1MON", "2HOUR"
        elif desc == "UNITT":
            self._parse_combined_timestep(config, value)
        # Separate length: DELTAT / DT / TIME_STEP_LENGTH
        elif desc in ("DELTAT", "DT") or "TIME_STEP_LENGTH" in desc:
            config.time_step_length = int(value)
        # Separate unit: UNIT / TIME_STEP_UNIT
        elif desc == "UNIT" or "TIME_STEP_UNIT" in desc:
            config.time_step_unit = TimeUnit.from_string(value)

        # --- Processing and debug ----------------------------------------
        elif desc == "KDEB":
            config.debug_flag = int(value)
        elif desc == "CACHE":
            config.cache_size = int(value)

        # --- Solver ------------------------------------------------------
        elif desc == "MSOLVE":
            config.matrix_solver = int(value)
        elif desc == "RELAX":
            config.relaxation = float(value)
        elif desc == "MXITER" or "MAX_ITER" in desc:
            config.max_iterations = int(value)
        elif desc == "MXITERSP":
            config.max_supply_iterations = int(value)
        elif desc == "STOPC" or ("CONV" in desc and "TOL" in desc):
            config.convergence_tolerance = float(value)
        elif desc == "STOPCVL":
            config.convergence_volume = float(value)
        elif desc == "STOPCSP":
            config.convergence_supply = float(value)

        # --- Supply adjustment -------------------------------------------
        elif desc in ("KOPTDV", "SUPPLY_ADJUST_OPTION"):
            config.supply_adjust_option = int(value)

        # --- Component files ---------------------------------------------
        # Binary preprocessor must be checked before plain preprocessor
        elif desc.startswith("1:") or ("BINARY" in desc and "PRE" in desc):
            config.binary_preprocessor_file = Path(value)
        elif "PREPROCESS" in desc and "FILE" in desc:
            config.preprocessor_file = Path(value)
        elif desc.startswith("2:") or ("GROUND" in desc and ("FILE" in desc or "MAIN" in desc)):
            config.groundwater_file = Path(value)
        elif desc.startswith("3:") or ("STREAM" in desc and ("FILE" in desc or "MAIN" in desc)):
            config.streams_file = Path(value)
        elif desc.startswith("4:") or ("LAKE" in desc and ("FILE" in desc or "MAIN" in desc)):
            config.lakes_file = Path(value)
        elif desc.startswith("5:") or ("ROOT" in desc and ("FILE" in desc or "MAIN" in desc)):
            config.rootzone_file = Path(value)
        elif desc.startswith("6:") or ("SMALL" in desc and "WATER" in desc):
            config.small_watershed_file = Path(value)
        elif desc.startswith("7:") or ("UNSAT" in desc and ("FILE" in desc or "MAIN" in desc)):
            config.unsaturated_zone_file = Path(value)
        elif desc.startswith("8:") or ("IRRIG" in desc and "FRAC" in desc):
            config.irrigation_fractions_file = Path(value)
        elif desc.startswith("9:") or ("SUPPLY" in desc and "ADJ" in desc):
            config.supply_adjust_file = Path(value)
        elif desc.startswith("10:") or (desc == "PRECIP" or ("PRECIP" in desc and "DATA" in desc)):
            config.precipitation_file = Path(value)
        elif desc.startswith("11:") or (desc == "ET" or ("ET" in desc and "DATA" in desc)):
            config.et_file = Path(value)
        elif desc.startswith("12:") or ("CROP" in desc and "COEFF" in desc):
            config.kc_file = Path(value)

        # --- Output control ----------------------------------------------
        elif "OUTPUT" in desc and "DIR" in desc:
            config.output_dir = Path(value)
        elif "OUTPUT" in desc and "INTERVAL" in desc:
            config.output_interval = int(value)

    def _parse_combined_timestep(
        self, config: SimulationConfig, value: str
    ) -> None:
        """Parse a combined time-step string like ``1MON`` or ``2HOUR``."""
        m = _re.match(r"(\d+)\s*(\w+)", value.strip())
        if m:
            config.time_step_length = int(m.group(1))
            config.time_step_unit = TimeUnit.from_string(m.group(2))
        else:
            # Fall back: treat as unit-only
            config.time_step_unit = TimeUnit.from_string(value)

    def _parse_datetime(self, value: str) -> datetime:
        """Parse an IWFM datetime string (MM/DD/YYYY_HH:MM, 16 chars).

        Hour ``24`` is treated as midnight of the next day.
        """
        from pyiwfm.io.timeseries_ascii import parse_iwfm_timestamp
        return parse_iwfm_timestamp(value)


class IWFMSimulationReader:
    """Reader for IWFM simulation main files in positional sequential format.

    Reads the actual IWFM Fortran format where values appear in fixed order
    (titles, file paths, time settings, solver parameters) as defined in
    ``SIM_ReadMainControlData`` in ``Package_Model.f90``.

    This reader handles both C2VSimFG-style files with ``/ description``
    comments and bare positional files without descriptions.
    """

    def __init__(self) -> None:
        self._line_num = 0

    def read(
        self, filepath: Path | str, base_dir: Path | None = None
    ) -> SimulationConfig:
        """Read IWFM simulation main file in positional format.

        Args:
            filepath: Path to the simulation main input file
            base_dir: Base directory for resolving relative paths

        Returns:
            SimulationConfig with all configuration data
        """
        filepath = Path(filepath)
        if base_dir is None:
            base_dir = filepath.parent

        config = SimulationConfig()
        self._line_num = 0

        with open(filepath, "r") as f:
            # Section 1: Title lines (3 non-comment lines)
            for _ in range(3):
                title = _next_data_or_empty(f)
                if title:
                    config.title_lines.append(title)

            # Section 2: File names (11 or 12 lines)
            # 1: Binary preprocessor file (required)
            bin_pp = _next_data_or_empty(f)
            if bin_pp:
                config.binary_preprocessor_file = _resolve_path_f(
                    base_dir, bin_pp
                )

            # 2: Groundwater main file (required)
            gw = _next_data_or_empty(f)
            if gw:
                config.groundwater_file = _resolve_path_f(base_dir, gw)

            # 3: Stream main file (optional)
            strm = _next_data_or_empty(f)
            if strm:
                config.streams_file = _resolve_path_f(base_dir, strm)

            # 4: Lake main file (optional)
            lake = _next_data_or_empty(f)
            if lake:
                config.lakes_file = _resolve_path_f(base_dir, lake)

            # 5: Root zone main file (optional)
            rz = _next_data_or_empty(f)
            if rz:
                config.rootzone_file = _resolve_path_f(base_dir, rz)

            # 6: Small watershed main file (optional)
            sw = _next_data_or_empty(f)
            if sw:
                config.small_watershed_file = _resolve_path_f(base_dir, sw)

            # 7: Unsaturated zone main file (optional)
            uz = _next_data_or_empty(f)
            if uz:
                config.unsaturated_zone_file = _resolve_path_f(
                    base_dir, uz
                )

            # 8: Irrigation fractions file (optional)
            irig = _next_data_or_empty(f)
            if irig:
                config.irrigation_fractions_file = _resolve_path_f(
                    base_dir, irig
                )

            # 9: Supply adjustment specification file (optional)
            supp = _next_data_or_empty(f)
            if supp:
                config.supply_adjust_file = _resolve_path_f(
                    base_dir, supp
                )

            # 10: Precipitation data file (optional)
            precip = _next_data_or_empty(f)
            if precip:
                config.precipitation_file = _resolve_path_f(
                    base_dir, precip
                )

            # 11: ET data file (optional)
            et = _next_data_or_empty(f)
            if et:
                config.et_file = _resolve_path_f(base_dir, et)

            # 12: Crop coefficient file (optional, backward compatibility)
            # Peek at the next data value to see if it looks like a file path
            # or a date. If it's a date (MM/DD/YYYY), it's BDT and there's
            # no 12th file entry.
            kc_or_bdt = _next_data_or_empty(f)
            if kc_or_bdt and self._looks_like_datetime(kc_or_bdt):
                # This is actually BDT (no KC file entry)
                config.start_date = _parse_iwfm_datetime(kc_or_bdt)
                bdt_already_read = True
            elif kc_or_bdt:
                config.kc_file = _resolve_path_f(base_dir, kc_or_bdt)
                bdt_already_read = False
            else:
                bdt_already_read = False

            # Section 3: Simulation period
            if not bdt_already_read:
                bdt_str = _next_data_or_empty(f)
                if bdt_str:
                    config.start_date = _parse_iwfm_datetime(bdt_str)

            # Restart flag
            restart_str = _next_data_or_empty(f)
            if restart_str:
                config.restart_flag = int(restart_str)

            # Time unit (combined format like "1MON")
            unitt_str = _next_data_or_empty(f)
            if unitt_str:
                m = _re.match(r"(\d+)\s*(\w+)", unitt_str.strip())
                if m:
                    config.time_step_length = int(m.group(1))
                    config.time_step_unit = TimeUnit.from_string(m.group(2))
                else:
                    config.time_step_unit = TimeUnit.from_string(unitt_str)

            # End date
            edt_str = _next_data_or_empty(f)
            if edt_str:
                config.end_date = _parse_iwfm_datetime(edt_str)

            # Section 4: Processing and output options
            # Restart output flag (ISTRT)
            istrt_str = _next_data_or_empty(f)
            if istrt_str:
                config.restart_output_flag = int(istrt_str)

            # Debug flag (KDEB)
            kdeb_str = _next_data_or_empty(f)
            if kdeb_str:
                config.debug_flag = int(kdeb_str)

            # Cache size
            cache_str = _next_data_or_empty(f)
            if cache_str:
                config.cache_size = int(cache_str)

            # Section 5: Solution scheme control
            # Matrix solver (MSOLVE)
            msolve_str = _next_data_or_empty(f)
            if msolve_str:
                config.matrix_solver = int(msolve_str)

            # Relaxation factor (RELAX)
            relax_str = _next_data_or_empty(f)
            if relax_str:
                config.relaxation = float(relax_str)

            # Max iterations (MXITER)
            mxiter_str = _next_data_or_empty(f)
            if mxiter_str:
                config.max_iterations = int(mxiter_str)

            # Max supply iterations (MXITERSP)
            mxitersp_str = _next_data_or_empty(f)
            if mxitersp_str:
                config.max_supply_iterations = int(mxitersp_str)

            # Flow convergence tolerance (STOPC)
            stopc_str = _next_data_or_empty(f)
            if stopc_str:
                config.convergence_tolerance = float(stopc_str)

            # Volume convergence tolerance (STOPCVL) - optional
            # The format may have 6 lines (no STOPCVL) or 7 lines (with STOPCVL):
            # 6-line: STOPC → STOPCSP → KOPTDV
            # 7-line: STOPC → STOPCVL → STOPCSP → KOPTDV
            #
            # Heuristic: KOPTDV is an integer (no decimal point in string).
            # Tolerances always have a decimal point or scientific notation.
            stopcvl_or_stopcsp = _next_data_or_empty(f)
            if stopcvl_or_stopcsp:
                koptdv_or_stopcsp = _next_data_or_empty(f)
                if koptdv_or_stopcsp:
                    if self._looks_like_integer(koptdv_or_stopcsp):
                        # No STOPCVL: stopcvl_or_stopcsp is STOPCSP,
                        # koptdv_or_stopcsp is KOPTDV
                        config.convergence_supply = float(
                            stopcvl_or_stopcsp
                        )
                        config.supply_adjust_option = int(
                            float(koptdv_or_stopcsp)
                        )
                    else:
                        # Has STOPCVL: stopcvl_or_stopcsp is STOPCVL,
                        # koptdv_or_stopcsp is STOPCSP
                        config.convergence_volume = float(
                            stopcvl_or_stopcsp
                        )
                        config.convergence_supply = float(
                            koptdv_or_stopcsp
                        )
                        # Read KOPTDV
                        kopt_str = _next_data_or_empty(f)
                        if kopt_str:
                            config.supply_adjust_option = int(
                                float(kopt_str)
                            )
                else:
                    # Only one more value: it's STOPCSP, no STOPCVL
                    try:
                        config.convergence_supply = float(stopcvl_or_stopcsp)
                    except ValueError:
                        pass

        # Set model_name from first title if not set
        if config.model_name == "IWFM_Model" and config.title_lines:
            config.model_name = config.title_lines[0].strip()

        return config

    @staticmethod
    def _looks_like_datetime(value: str) -> bool:
        """Check if a string looks like an IWFM datetime (MM/DD/YYYY)."""
        return bool(_re.match(r"\d{1,2}/\d{1,2}/\d{4}", value.strip()))

    @staticmethod
    def _looks_like_integer(value: str) -> bool:
        """Check if a string looks like a plain integer (no decimal point).

        KOPTDV is always an integer; tolerances have decimal points or
        scientific notation (``e``/``E``).
        """
        s = value.strip()
        if not s:
            return False
        # Has decimal point or scientific notation → not a plain integer
        if "." in s or "e" in s.lower():
            return False
        try:
            int(s)
            return True
        except ValueError:
            return False


def _parse_iwfm_datetime(value: str) -> datetime:
    """Parse an IWFM datetime string (MM/DD/YYYY_HH:MM, 16 chars).

    Hour ``24`` is treated as midnight of the next day.
    """
    from pyiwfm.io.timeseries_ascii import parse_iwfm_timestamp
    return parse_iwfm_timestamp(value)


# Convenience functions


def read_iwfm_simulation(
    filepath: Path | str, base_dir: Path | None = None
) -> SimulationConfig:
    """Read IWFM simulation main file in positional format.

    This reads files in the native IWFM Fortran format where values appear
    in a fixed order (titles, file paths, time settings, solver parameters).

    Args:
        filepath: Path to the simulation main input file
        base_dir: Base directory for resolving relative paths

    Returns:
        SimulationConfig with all configuration data
    """
    reader = IWFMSimulationReader()
    return reader.read(filepath, base_dir)


def write_simulation(
    config: SimulationConfig,
    output_dir: Path | str,
    file_config: SimulationFileConfig | None = None,
) -> Path:
    """
    Write simulation control file.

    Args:
        config: Simulation configuration
        output_dir: Output directory
        file_config: Optional file configuration

    Returns:
        Path to written file
    """
    output_dir = Path(output_dir)

    if file_config is None:
        file_config = SimulationFileConfig(output_dir=output_dir)
    else:
        file_config.output_dir = output_dir

    writer = SimulationWriter(file_config)
    return writer.write(config)


def read_simulation(filepath: Path | str) -> SimulationConfig:
    """
    Read simulation configuration from file.

    Args:
        filepath: Path to simulation input file

    Returns:
        SimulationConfig object
    """
    reader = SimulationReader()
    return reader.read(filepath)
