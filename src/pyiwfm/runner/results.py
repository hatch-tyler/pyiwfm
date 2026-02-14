"""Result dataclasses for IWFM subprocess runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


@dataclass
class RunResult:
    """Base result class for IWFM executable runs.

    Attributes
    ----------
    success : bool
        Whether the run completed successfully.
    return_code : int
        Process return code (0 = success).
    stdout : str
        Standard output from the process.
    stderr : str
        Standard error from the process.
    working_dir : Path
        Working directory where the run executed.
    elapsed_time : timedelta
        Wall-clock time for the run.
    log_file : Path | None
        Path to the IWFM log/message file if created.
    log_content : str
        Content of the log file if available.
    errors : list[str]
        List of error messages extracted from output/logs.
    warnings : list[str]
        List of warning messages extracted from output/logs.
    """

    success: bool
    return_code: int
    stdout: str = ""
    stderr: str = ""
    working_dir: Path = field(default_factory=Path.cwd)
    elapsed_time: timedelta = field(default_factory=lambda: timedelta(0))
    log_file: Path | None = None
    log_content: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Convert paths to Path objects."""
        if isinstance(self.working_dir, str):
            self.working_dir = Path(self.working_dir)
        if isinstance(self.log_file, str):
            self.log_file = Path(self.log_file)

    @property
    def failed(self) -> bool:
        """Check if the run failed."""
        return not self.success

    def raise_on_error(self) -> None:
        """Raise an exception if the run failed.

        Raises
        ------
        RuntimeError
            If the run did not complete successfully.
        """
        if not self.success:
            error_msg = f"IWFM run failed with return code {self.return_code}"
            if self.errors:
                error_msg += f": {'; '.join(self.errors)}"
            elif self.stderr:
                error_msg += f": {self.stderr[:500]}"
            raise RuntimeError(error_msg)


@dataclass
class PreprocessorResult(RunResult):
    """Result from running the IWFM PreProcessor.

    Attributes
    ----------
    main_file : Path | None
        Path to the preprocessor main input file.
    binary_output : Path | None
        Path to the generated binary output file.
    n_nodes : int
        Number of nodes in the model.
    n_elements : int
        Number of elements in the model.
    n_layers : int
        Number of layers in the model.
    n_subregions : int
        Number of subregions.
    """

    main_file: Path | None = None
    binary_output: Path | None = None
    n_nodes: int = 0
    n_elements: int = 0
    n_layers: int = 0
    n_subregions: int = 0

    def __post_init__(self) -> None:
        """Convert paths to Path objects."""
        super().__post_init__()
        if isinstance(self.main_file, str):
            self.main_file = Path(self.main_file)
        if isinstance(self.binary_output, str):
            self.binary_output = Path(self.binary_output)


@dataclass
class SimulationResult(RunResult):
    """Result from running the IWFM Simulation.

    Attributes
    ----------
    main_file : Path | None
        Path to the simulation main input file.
    n_timesteps : int
        Number of timesteps completed.
    start_date : datetime | None
        Simulation start date.
    end_date : datetime | None
        Simulation end date.
    budget_files : list[Path]
        List of budget output files generated.
    hydrograph_files : list[Path]
        List of hydrograph output files generated.
    final_heads_file : Path | None
        Path to final groundwater heads file.
    convergence_failures : int
        Number of timesteps with convergence issues.
    mass_balance_error : float
        Maximum mass balance error (if reported).
    """

    main_file: Path | None = None
    n_timesteps: int = 0
    start_date: datetime | None = None
    end_date: datetime | None = None
    budget_files: list[Path] = field(default_factory=list)
    hydrograph_files: list[Path] = field(default_factory=list)
    final_heads_file: Path | None = None
    convergence_failures: int = 0
    mass_balance_error: float = 0.0

    def __post_init__(self) -> None:
        """Convert paths to Path objects."""
        super().__post_init__()
        if isinstance(self.main_file, str):
            self.main_file = Path(self.main_file)
        if isinstance(self.final_heads_file, str):
            self.final_heads_file = Path(self.final_heads_file)
        self.budget_files = [
            Path(f) if isinstance(f, str) else f
            for f in self.budget_files
        ]
        self.hydrograph_files = [
            Path(f) if isinstance(f, str) else f
            for f in self.hydrograph_files
        ]


@dataclass
class BudgetResult(RunResult):
    """Result from running the IWFM Budget post-processor.

    Attributes
    ----------
    budget_file : Path | None
        Path to the budget binary file processed.
    output_file : Path | None
        Path to the output file generated.
    n_locations : int
        Number of locations in the budget.
    n_timesteps : int
        Number of timesteps in the budget.
    components : list[str]
        List of budget components.
    """

    budget_file: Path | None = None
    output_file: Path | None = None
    n_locations: int = 0
    n_timesteps: int = 0
    components: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Convert paths to Path objects."""
        super().__post_init__()
        if isinstance(self.budget_file, str):
            self.budget_file = Path(self.budget_file)
        if isinstance(self.output_file, str):
            self.output_file = Path(self.output_file)


@dataclass
class ZBudgetResult(RunResult):
    """Result from running the IWFM ZBudget post-processor.

    Attributes
    ----------
    zbudget_file : Path | None
        Path to the zone budget HDF5 file processed.
    zone_file : Path | None
        Path to the zone definition file used.
    output_file : Path | None
        Path to the output file generated.
    n_zones : int
        Number of zones processed.
    n_timesteps : int
        Number of timesteps in the output.
    """

    zbudget_file: Path | None = None
    zone_file: Path | None = None
    output_file: Path | None = None
    n_zones: int = 0
    n_timesteps: int = 0

    def __post_init__(self) -> None:
        """Convert paths to Path objects."""
        super().__post_init__()
        if isinstance(self.zbudget_file, str):
            self.zbudget_file = Path(self.zbudget_file)
        if isinstance(self.zone_file, str):
            self.zone_file = Path(self.zone_file)
        if isinstance(self.output_file, str):
            self.output_file = Path(self.output_file)
