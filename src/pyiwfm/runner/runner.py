"""IWFM subprocess runner for executing IWFM executables."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from pyiwfm.runner.results import (
    BudgetResult,
    PreprocessorResult,
    SimulationResult,
    ZBudgetResult,
)


@dataclass
class IWFMExecutables:
    """Paths to IWFM executables.

    Attributes
    ----------
    simulation : Path | None
        Path to the Simulation executable.
    simulation_parallel : Path | None
        Path to the parallel Simulation executable.
    preprocessor : Path | None
        Path to the PreProcessor executable.
    budget : Path | None
        Path to the Budget post-processor executable.
    zbudget : Path | None
        Path to the ZBudget post-processor executable.
    """

    simulation: Path | None = None
    simulation_parallel: Path | None = None
    preprocessor: Path | None = None
    budget: Path | None = None
    zbudget: Path | None = None

    def __post_init__(self) -> None:
        """Validate and convert paths."""
        for attr in ["simulation", "simulation_parallel", "preprocessor", "budget", "zbudget"]:
            value = getattr(self, attr)
            if value is not None:
                path = Path(value)
                if path.exists():
                    setattr(self, attr, path)
                else:
                    setattr(self, attr, None)

    @property
    def available(self) -> list[str]:
        """List of available executables."""
        result = []
        if self.simulation:
            result.append("simulation")
        if self.simulation_parallel:
            result.append("simulation_parallel")
        if self.preprocessor:
            result.append("preprocessor")
        if self.budget:
            result.append("budget")
        if self.zbudget:
            result.append("zbudget")
        return result

    def __repr__(self) -> str:
        """Return string representation."""
        return f"IWFMExecutables(available={self.available})"


def find_iwfm_executables(
    search_paths: list[Path] | None = None,
    env_var: str = "IWFM_BIN",
) -> IWFMExecutables:
    """Find IWFM executables on the system.

    Searches for executables in the following order:
    1. Paths provided in search_paths
    2. Path from environment variable (default: IWFM_BIN)
    3. Current working directory
    4. System PATH

    Parameters
    ----------
    search_paths : list[Path] | None
        Additional paths to search for executables.
    env_var : str
        Environment variable containing IWFM bin directory.

    Returns
    -------
    IWFMExecutables
        Dataclass containing paths to found executables.
    """
    # Build search path list
    paths_to_search: list[Path] = []

    if search_paths:
        paths_to_search.extend(Path(p) for p in search_paths)

    # Check environment variable
    env_path = os.environ.get(env_var)
    if env_path:
        paths_to_search.append(Path(env_path))

    # Add current directory
    paths_to_search.append(Path.cwd())

    # Add common locations relative to this package
    # (assuming pyiwfm is at repo root alongside src/)
    pkg_dir = Path(__file__).parent.parent.parent.parent
    paths_to_search.append(pkg_dir / "Bin")
    paths_to_search.append(pkg_dir.parent / "Bin")
    paths_to_search.append(pkg_dir.parent / "src" / "Bin")

    # Executable name patterns (Windows and Unix)
    exe_suffix = ".exe" if sys.platform == "win32" else ""
    patterns = {
        "simulation": [
            f"Simulation_x64{exe_suffix}",
            f"Simulation{exe_suffix}",
            f"iwfm_simulation{exe_suffix}",
        ],
        "simulation_parallel": [
            f"Simulation_PLL_x64{exe_suffix}",
            f"Simulation_Parallel{exe_suffix}",
        ],
        "preprocessor": [
            f"PreProcessor_x64{exe_suffix}",
            f"PreProcessor{exe_suffix}",
            f"iwfm_preprocessor{exe_suffix}",
        ],
        "budget": [
            f"Budget_x64{exe_suffix}",
            f"Budget{exe_suffix}",
            f"iwfm_budget{exe_suffix}",
        ],
        "zbudget": [
            f"ZBudget_x64{exe_suffix}",
            f"ZBudget{exe_suffix}",
            f"iwfm_zbudget{exe_suffix}",
        ],
    }

    result = IWFMExecutables()

    # Search for each executable
    for exe_type, names in patterns.items():
        for search_path in paths_to_search:
            if not search_path.exists():
                continue
            for name in names:
                exe_path = search_path / name
                if exe_path.exists() and exe_path.is_file():
                    setattr(result, exe_type, exe_path)
                    break
            if getattr(result, exe_type) is not None:
                break

        # Try finding in system PATH
        if getattr(result, exe_type) is None:
            for name in names:
                found = shutil.which(name)
                if found:
                    setattr(result, exe_type, Path(found))
                    break

    return result


class IWFMRunner:
    """Run IWFM executables via subprocess.

    This class provides methods to run IWFM PreProcessor, Simulation,
    Budget, and ZBudget executables. It handles input/output redirection,
    working directory management, and result parsing.

    Parameters
    ----------
    executables : IWFMExecutables | None
        Paths to IWFM executables. If None, will auto-detect.
    working_dir : Path | None
        Default working directory for runs. If None, uses the
        directory containing the input file.

    Examples
    --------
    >>> runner = IWFMRunner()
    >>> result = runner.run_simulation("Simulation/Simulation.in")
    >>> if result.success:
    ...     print(f"Completed {result.n_timesteps} timesteps")
    """

    def __init__(
        self,
        executables: IWFMExecutables | None = None,
        working_dir: Path | None = None,
    ) -> None:
        """Initialize the IWFM runner."""
        self.executables = executables or find_iwfm_executables()
        self.working_dir = Path(working_dir) if working_dir else None

    def _get_working_dir(self, main_file: Path, override: Path | None) -> Path:
        """Determine working directory for a run."""
        if override:
            return Path(override)
        if self.working_dir:
            return self.working_dir
        return main_file.parent

    def _parse_log_messages(
        self,
        log_content: str,
    ) -> tuple[list[str], list[str]]:
        """Extract errors and warnings from log content.

        Returns
        -------
        tuple[list[str], list[str]]
            (errors, warnings) lists
        """
        errors: list[str] = []
        warnings: list[str] = []

        for line in log_content.splitlines():
            line_lower = line.lower()
            if "error" in line_lower or "fatal" in line_lower:
                errors.append(line.strip())
            elif "warning" in line_lower:
                warnings.append(line.strip())

        return errors, warnings

    def _run_executable(
        self,
        executable: Path,
        input_text: str,
        working_dir: Path,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> tuple[int, str, str, timedelta]:
        """Run an executable with input redirection.

        Parameters
        ----------
        executable : Path
            Path to the executable.
        input_text : str
            Text to send to stdin (typically the main file path).
        working_dir : Path
            Working directory for the process.
        timeout : float | None
            Timeout in seconds. None for no timeout.
        env : dict[str, str] | None
            Additional environment variables.

        Returns
        -------
        tuple[int, str, str, timedelta]
            (return_code, stdout, stderr, elapsed_time)
        """
        # Prepare environment
        run_env = os.environ.copy()
        if env:
            run_env.update(env)

        start_time = datetime.now()

        try:
            result = subprocess.run(
                [str(executable)],
                input=input_text,
                capture_output=True,
                text=True,
                cwd=str(working_dir),
                timeout=timeout,
                env=run_env,
            )
            elapsed = datetime.now() - start_time
            return result.returncode, result.stdout, result.stderr, elapsed

        except subprocess.TimeoutExpired as e:
            elapsed = datetime.now() - start_time
            stdout = e.stdout.decode() if e.stdout else ""
            stderr = e.stderr.decode() if e.stderr else "Process timed out"
            return -1, stdout, stderr, elapsed

        except Exception as e:
            elapsed = datetime.now() - start_time
            return -1, "", str(e), elapsed

    def run_preprocessor(
        self,
        main_file: Path | str,
        working_dir: Path | str | None = None,
        timeout: float | None = None,
    ) -> PreprocessorResult:
        """Run the IWFM PreProcessor.

        Parameters
        ----------
        main_file : Path | str
            Path to the PreProcessor main input file.
        working_dir : Path | str | None
            Working directory. Defaults to main_file's directory.
        timeout : float | None
            Timeout in seconds.

        Returns
        -------
        PreprocessorResult
            Result object containing success status and outputs.

        Raises
        ------
        FileNotFoundError
            If the executable or main file is not found.
        """
        if self.executables.preprocessor is None:
            raise FileNotFoundError("PreProcessor executable not found")

        main_file = Path(main_file).resolve()
        if not main_file.exists():
            raise FileNotFoundError(f"Main file not found: {main_file}")

        work_dir = self._get_working_dir(
            main_file,
            Path(working_dir) if working_dir else None,
        )

        # Run the preprocessor
        return_code, stdout, stderr, elapsed = self._run_executable(
            self.executables.preprocessor,
            str(main_file) + "\n",
            work_dir,
            timeout,
        )

        # Look for log file
        log_file = work_dir / "PreprocessorMessages.out"
        log_content = ""
        if log_file.exists():
            log_content = log_file.read_text(errors="replace")

        errors, warnings = self._parse_log_messages(log_content + stdout + stderr)

        # Parse output for model info
        n_nodes = 0
        n_elements = 0
        n_layers = 0
        n_subregions = 0

        # Look for summary info in output
        combined = stdout + log_content
        node_match = re.search(r"(\d+)\s*nodes?", combined, re.IGNORECASE)
        if node_match:
            n_nodes = int(node_match.group(1))

        elem_match = re.search(r"(\d+)\s*elements?", combined, re.IGNORECASE)
        if elem_match:
            n_elements = int(elem_match.group(1))

        layer_match = re.search(r"(\d+)\s*layers?", combined, re.IGNORECASE)
        if layer_match:
            n_layers = int(layer_match.group(1))

        # Look for binary output file
        binary_output = None
        for pattern in ["*.bin", "*.BIN", "*_Binary.dat"]:
            matches = list(work_dir.glob(pattern))
            if matches:
                binary_output = matches[0]
                break

        return PreprocessorResult(
            success=return_code == 0 and len(errors) == 0,
            return_code=return_code,
            stdout=stdout,
            stderr=stderr,
            working_dir=work_dir,
            elapsed_time=elapsed,
            log_file=log_file if log_file.exists() else None,
            log_content=log_content,
            errors=errors,
            warnings=warnings,
            main_file=main_file,
            binary_output=binary_output,
            n_nodes=n_nodes,
            n_elements=n_elements,
            n_layers=n_layers,
            n_subregions=n_subregions,
        )

    def run_simulation(
        self,
        main_file: Path | str,
        working_dir: Path | str | None = None,
        timeout: float | None = None,
        parallel: bool = False,
    ) -> SimulationResult:
        """Run the IWFM Simulation.

        Parameters
        ----------
        main_file : Path | str
            Path to the Simulation main input file.
        working_dir : Path | str | None
            Working directory. Defaults to main_file's directory.
        timeout : float | None
            Timeout in seconds.
        parallel : bool
            Use parallel executable if available.

        Returns
        -------
        SimulationResult
            Result object containing success status and outputs.

        Raises
        ------
        FileNotFoundError
            If the executable or main file is not found.
        """
        # Select executable
        if parallel and self.executables.simulation_parallel:
            executable = self.executables.simulation_parallel
        elif self.executables.simulation:
            executable = self.executables.simulation
        else:
            raise FileNotFoundError("Simulation executable not found")

        main_file = Path(main_file).resolve()
        if not main_file.exists():
            raise FileNotFoundError(f"Main file not found: {main_file}")

        work_dir = self._get_working_dir(
            main_file,
            Path(working_dir) if working_dir else None,
        )

        # Run the simulation
        return_code, stdout, stderr, elapsed = self._run_executable(
            executable,
            str(main_file) + "\n",
            work_dir,
            timeout,
        )

        # Look for log file
        log_file = work_dir / "SimulationMessages.out"
        log_content = ""
        if log_file.exists():
            log_content = log_file.read_text(errors="replace")

        errors, warnings = self._parse_log_messages(log_content + stdout + stderr)

        # Parse output for simulation info
        n_timesteps = 0
        start_date = None
        end_date = None
        convergence_failures = 0
        mass_balance_error = 0.0

        combined = stdout + log_content

        # Count timesteps
        timestep_matches = re.findall(r"time\s*step\s*[\d]+", combined, re.IGNORECASE)
        n_timesteps = len(timestep_matches)

        # Look for convergence issues
        conv_matches = re.findall(r"convergence", combined, re.IGNORECASE)
        convergence_failures = len([m for m in conv_matches if "fail" in combined.lower()])

        # Find budget files
        budget_files: list[Path] = []
        for pattern in ["*Budget*.hdf", "*Budget*.bin", "*_GW.BUD", "*_STR.BUD"]:
            budget_files.extend(work_dir.glob(pattern))

        # Find hydrograph files
        hydrograph_files: list[Path] = []
        for pattern in ["*Hydrograph*.out", "*_GWHyd.out", "*_STRHyd.out"]:
            hydrograph_files.extend(work_dir.glob(pattern))

        # Find final heads file
        final_heads_file = None
        for pattern in ["*FinalHeads*.dat", "*_FinalGWHeads.dat"]:
            matches = list(work_dir.glob(pattern))
            if matches:
                final_heads_file = matches[0]
                break

        return SimulationResult(
            success=return_code == 0 and len(errors) == 0,
            return_code=return_code,
            stdout=stdout,
            stderr=stderr,
            working_dir=work_dir,
            elapsed_time=elapsed,
            log_file=log_file if log_file.exists() else None,
            log_content=log_content,
            errors=errors,
            warnings=warnings,
            main_file=main_file,
            n_timesteps=n_timesteps,
            start_date=start_date,
            end_date=end_date,
            budget_files=budget_files,
            hydrograph_files=hydrograph_files,
            final_heads_file=final_heads_file,
            convergence_failures=convergence_failures,
            mass_balance_error=mass_balance_error,
        )

    def run_budget(
        self,
        budget_file: Path | str,
        working_dir: Path | str | None = None,
        timeout: float | None = None,
        instructions: str | None = None,
    ) -> BudgetResult:
        """Run the IWFM Budget post-processor.

        Parameters
        ----------
        budget_file : Path | str
            Path to the budget binary file.
        working_dir : Path | str | None
            Working directory. Defaults to budget_file's directory.
        timeout : float | None
            Timeout in seconds.
        instructions : str | None
            Budget processing instructions (interactive responses).

        Returns
        -------
        BudgetResult
            Result object containing success status and outputs.

        Raises
        ------
        FileNotFoundError
            If the executable or budget file is not found.
        """
        if self.executables.budget is None:
            raise FileNotFoundError("Budget executable not found")

        budget_file = Path(budget_file).resolve()
        if not budget_file.exists():
            raise FileNotFoundError(f"Budget file not found: {budget_file}")

        work_dir = self._get_working_dir(
            budget_file,
            Path(working_dir) if working_dir else None,
        )

        # Build input for budget processor
        # Default: just provide the budget file path
        input_text = instructions or (str(budget_file) + "\n")

        # Run the budget processor
        return_code, stdout, stderr, elapsed = self._run_executable(
            self.executables.budget,
            input_text,
            work_dir,
            timeout,
        )

        # Look for log file
        log_file = work_dir / "BudgetMessages.out"
        log_content = ""
        if log_file.exists():
            log_content = log_file.read_text(errors="replace")

        errors, warnings = self._parse_log_messages(log_content + stdout + stderr)

        return BudgetResult(
            success=return_code == 0 and len(errors) == 0,
            return_code=return_code,
            stdout=stdout,
            stderr=stderr,
            working_dir=work_dir,
            elapsed_time=elapsed,
            log_file=log_file if log_file.exists() else None,
            log_content=log_content,
            errors=errors,
            warnings=warnings,
            budget_file=budget_file,
        )

    def run_zbudget(
        self,
        zbudget_file: Path | str,
        zone_file: Path | str | None = None,
        working_dir: Path | str | None = None,
        timeout: float | None = None,
        instructions: str | None = None,
    ) -> ZBudgetResult:
        """Run the IWFM ZBudget post-processor.

        Parameters
        ----------
        zbudget_file : Path | str
            Path to the zone budget HDF5 file.
        zone_file : Path | str | None
            Path to the zone definition file.
        working_dir : Path | str | None
            Working directory. Defaults to zbudget_file's directory.
        timeout : float | None
            Timeout in seconds.
        instructions : str | None
            ZBudget processing instructions (interactive responses).

        Returns
        -------
        ZBudgetResult
            Result object containing success status and outputs.

        Raises
        ------
        FileNotFoundError
            If the executable or zbudget file is not found.
        """
        if self.executables.zbudget is None:
            raise FileNotFoundError("ZBudget executable not found")

        zbudget_file = Path(zbudget_file).resolve()
        if not zbudget_file.exists():
            raise FileNotFoundError(f"ZBudget file not found: {zbudget_file}")

        work_dir = self._get_working_dir(
            zbudget_file,
            Path(working_dir) if working_dir else None,
        )

        # Build input for zbudget processor
        input_text = instructions or (str(zbudget_file) + "\n")

        # Run the zbudget processor
        return_code, stdout, stderr, elapsed = self._run_executable(
            self.executables.zbudget,
            input_text,
            work_dir,
            timeout,
        )

        # Look for log file
        log_file = work_dir / "ZBudgetMessages.out"
        log_content = ""
        if log_file.exists():
            log_content = log_file.read_text(errors="replace")

        errors, warnings = self._parse_log_messages(log_content + stdout + stderr)

        return ZBudgetResult(
            success=return_code == 0 and len(errors) == 0,
            return_code=return_code,
            stdout=stdout,
            stderr=stderr,
            working_dir=work_dir,
            elapsed_time=elapsed,
            log_file=log_file if log_file.exists() else None,
            log_content=log_content,
            errors=errors,
            warnings=warnings,
            zbudget_file=zbudget_file,
            zone_file=Path(zone_file) if zone_file else None,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"IWFMRunner(executables={self.executables.available}, working_dir={self.working_dir})"
        )
