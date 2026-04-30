"""
Simulation control I/O handlers for IWFM model files.

This module provides functions for reading and writing IWFM simulation
control files including the main simulation input file, time stepping,
and output control settings.
"""

from __future__ import annotations

import re as _re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from pyiwfm.core.exceptions import FileFormatError
from pyiwfm.core.timeseries import SimulationPeriod, TimeUnit
from pyiwfm.io.ascii.reader import (
    is_comment_line as _is_comment_line,
)
from pyiwfm.io.ascii.reader import (
    next_data_or_empty as _next_data_or_empty,
)
from pyiwfm.io.ascii.reader import (
    resolve_path as _resolve_path_f,
)
from pyiwfm.io.ascii.reader import (
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

        with open(filepath, "w", encoding="utf-8") as f:
            self._write_header(f, header, config)
            self._write_time_settings(f, config)
            self._write_component_files(f, config)
            self._write_processing_options(f, config)
            self._write_solver_settings(f, config)
            self._write_supply_adjustment(f, config)
            self._write_output_settings(f, config)

        return filepath

    def _write_header(self, f: TextIO, header: str | None, config: SimulationConfig) -> None:
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
        f.write(
            f"{config.time_step_unit.value:<10}                              / TIME_STEP_UNIT\n"
        )

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
        f.write(
            f"{config.budget_output_interval:<10}                              / BUDGET_OUTPUT_INTERVAL\n"
        )
        f.write(
            f"{config.heads_output_interval:<10}                              / HEADS_OUTPUT_INTERVAL\n"
        )


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

        with open(filepath, encoding="utf-8") as f:
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

    def _parse_config_line(self, config: SimulationConfig, value: str, desc: str) -> None:
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

    def _parse_combined_timestep(self, config: SimulationConfig, value: str) -> None:
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

    # Ordered list of (field_name, position_label) for the fixed 11 input-file
    # slots that precede the optional 12th (KC) slot. Paths are resolved
    # against ``base_dir`` and assigned only when the line is non-empty.
    _INPUT_FILE_FIELDS: tuple[tuple[str, str], ...] = (
        ("binary_preprocessor_file", "binary preprocessor"),
        ("groundwater_file", "groundwater main"),
        ("streams_file", "stream main"),
        ("lakes_file", "lake main"),
        ("rootzone_file", "root zone main"),
        ("small_watershed_file", "small watershed main"),
        ("unsaturated_zone_file", "unsaturated zone main"),
        ("irrigation_fractions_file", "irrigation fractions"),
        ("supply_adjust_file", "supply adjustment"),
        ("precipitation_file", "precipitation data"),
        ("et_file", "ET data"),
    )

    def read(self, filepath: Path | str, base_dir: Path | None = None) -> SimulationConfig:
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

        with open(filepath, encoding="utf-8") as f:
            self._read_titles(f, config)
            bdt_already_read = self._read_input_files(f, config, base_dir)
            self._read_time_settings(f, config, bdt_already_read)
            self._read_processing_options(f, config)
            self._read_solver_settings(f, config)

        # Default model_name from first title if caller never set one.
        if config.model_name == "IWFM_Model" and config.title_lines:
            config.model_name = config.title_lines[0].strip()

        return config

    def _read_titles(self, f: TextIO, config: SimulationConfig) -> None:
        """Section 1 — three title lines."""
        for _ in range(3):
            title = _next_data_or_empty(f)
            if title:
                config.title_lines.append(title)

    def _read_input_files(
        self,
        f: TextIO,
        config: SimulationConfig,
        base_dir: Path,
    ) -> bool:
        """Section 2 — 11 required/optional input-file paths + 12th KC slot.

        The 12th slot is optional (backward compatibility): if the line parses
        as an IWFM ``MM/DD/YYYY`` date it's actually the BDT value from
        Section 3 and no KC file is present. Returns ``True`` in that case so
        the caller can skip re-reading BDT.
        """
        for attr, _label in self._INPUT_FILE_FIELDS:
            value = _next_data_or_empty(f)
            if value:
                setattr(config, attr, _resolve_path_f(base_dir, value))

        kc_or_bdt = _next_data_or_empty(f)
        if not kc_or_bdt:
            return False
        if self._looks_like_datetime(kc_or_bdt):
            config.start_date = _parse_iwfm_datetime(kc_or_bdt)
            return True
        config.kc_file = _resolve_path_f(base_dir, kc_or_bdt)
        return False

    def _read_time_settings(
        self,
        f: TextIO,
        config: SimulationConfig,
        bdt_already_read: bool,
    ) -> None:
        """Section 3 — BDT, restart flag, time step unit, EDT."""
        if not bdt_already_read:
            bdt_str = _next_data_or_empty(f)
            if bdt_str:
                config.start_date = _parse_iwfm_datetime(bdt_str)

        restart_str = _next_data_or_empty(f)
        if restart_str:
            config.restart_flag = int(restart_str)

        unitt_str = _next_data_or_empty(f)
        if unitt_str:
            # Combined format like "1MON" -> length 1, unit MON.
            m = _re.match(r"(\d+)\s*(\w+)", unitt_str.strip())
            if m:
                config.time_step_length = int(m.group(1))
                config.time_step_unit = TimeUnit.from_string(m.group(2))
            else:
                config.time_step_unit = TimeUnit.from_string(unitt_str)

        edt_str = _next_data_or_empty(f)
        if edt_str:
            config.end_date = _parse_iwfm_datetime(edt_str)

    def _read_processing_options(self, f: TextIO, config: SimulationConfig) -> None:
        """Section 4 — ISTRT (restart output), KDEB (debug), cache size."""
        istrt_str = _next_data_or_empty(f)
        if istrt_str:
            config.restart_output_flag = int(istrt_str)

        kdeb_str = _next_data_or_empty(f)
        if kdeb_str:
            config.debug_flag = int(kdeb_str)

        cache_str = _next_data_or_empty(f)
        if cache_str:
            config.cache_size = int(cache_str)

    def _read_solver_settings(self, f: TextIO, config: SimulationConfig) -> None:
        """Section 5 — MSOLVE, RELAX, MXITER, MXITERSP, STOPC, and the
        optional STOPCVL / STOPCSP / KOPTDV tail.

        The tail has two valid shapes:
            * 6-line: STOPC, STOPCSP, KOPTDV
            * 7-line: STOPC, STOPCVL, STOPCSP, KOPTDV
        Disambiguation: KOPTDV is always an integer (no decimal point /
        exponent); tolerances are floats. See :meth:`_looks_like_integer`.
        """
        msolve_str = _next_data_or_empty(f)
        if msolve_str:
            config.matrix_solver = int(msolve_str)

        relax_str = _next_data_or_empty(f)
        if relax_str:
            config.relaxation = float(relax_str)

        mxiter_str = _next_data_or_empty(f)
        if mxiter_str:
            config.max_iterations = int(mxiter_str)

        mxitersp_str = _next_data_or_empty(f)
        if mxitersp_str:
            config.max_supply_iterations = int(mxitersp_str)

        stopc_str = _next_data_or_empty(f)
        if stopc_str:
            config.convergence_tolerance = float(stopc_str)

        self._read_convergence_tail(f, config)

    def _read_convergence_tail(self, f: TextIO, config: SimulationConfig) -> None:
        """Handle the ambiguous 6-vs-7-line convergence tail."""
        first = _next_data_or_empty(f)
        if not first:
            return

        second = _next_data_or_empty(f)
        if not second:
            # Only one more value: treat it as STOPCSP.
            try:
                config.convergence_supply = float(first)
            except ValueError:
                pass
            return

        if self._looks_like_integer(second):
            # 6-line: first=STOPCSP, second=KOPTDV.
            config.convergence_supply = float(first)
            config.supply_adjust_option = int(float(second))
            return

        # 7-line: first=STOPCVL, second=STOPCSP, followed by KOPTDV.
        config.convergence_volume = float(first)
        config.convergence_supply = float(second)
        kopt_str = _next_data_or_empty(f)
        if kopt_str:
            config.supply_adjust_option = int(float(kopt_str))

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


def read_iwfm_simulation(filepath: Path | str, base_dir: Path | None = None) -> SimulationConfig:
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
