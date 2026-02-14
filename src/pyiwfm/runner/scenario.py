"""Scenario management for IWFM model runs."""

from __future__ import annotations

import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from pyiwfm.runner.results import SimulationResult


@dataclass
class Scenario:
    """Definition of a model scenario.

    A scenario represents a modified version of a baseline model run.
    It can include modifications to input files (pumping, diversions,
    land use, etc.) and is identified by a unique name.

    Attributes
    ----------
    name : str
        Unique name identifying this scenario.
    description : str
        Description of what this scenario represents.
    modifications : dict[str, Any]
        Dictionary of modifications to apply to input files.
        Keys are file types (e.g., "pumping", "diversion"),
        values are modification specifications.
    modifier_func : Callable | None
        Optional function to apply custom modifications to the
        scenario directory. Function signature:
        (scenario_dir: Path, baseline_dir: Path) -> None
    """

    name: str
    description: str = ""
    modifications: dict[str, Any] = field(default_factory=dict)
    modifier_func: Callable[[Path, Path], None] | None = None

    def __post_init__(self) -> None:
        """Validate scenario name."""
        # Ensure name is filesystem-safe
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            if char in self.name:
                raise ValueError(
                    f"Scenario name contains invalid character: {char}"
                )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"Scenario(name='{self.name}', modifications={list(self.modifications.keys())})"


@dataclass
class ScenarioResult:
    """Result of a scenario run with comparison to baseline.

    Attributes
    ----------
    scenario : Scenario
        The scenario that was run.
    result : SimulationResult
        The simulation result for this scenario.
    scenario_dir : Path
        Directory where the scenario was run.
    differences : dict[str, Any]
        Computed differences from baseline (if available).
    """

    scenario: Scenario
    result: SimulationResult
    scenario_dir: Path
    differences: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if the scenario run succeeded."""
        return self.result.success

    def __repr__(self) -> str:
        """Return string representation."""
        status = "success" if self.success else "failed"
        return f"ScenarioResult(scenario='{self.scenario.name}', status={status})"


class ScenarioManager:
    """Manage and run multiple IWFM scenarios.

    This class provides utilities for:
    - Creating scenario directories from a baseline
    - Applying modifications to scenario input files
    - Running scenarios in parallel
    - Comparing results to baseline

    Parameters
    ----------
    baseline_dir : Path | str
        Path to the baseline model directory.
    scenarios_root : Path | str | None
        Root directory for scenario runs. If None, creates a
        'scenarios' subdirectory in baseline_dir.
    main_file_name : str
        Name of the simulation main file (relative to model dir).

    Examples
    --------
    >>> manager = ScenarioManager("C2VSim/Simulation")
    >>> scenarios = [
    ...     Scenario("reduced_pumping", modifications={"pumping": 0.8}),
    ...     Scenario("no_diversions", modifications={"diversion": 0.0}),
    ... ]
    >>> results = manager.run_scenarios(scenarios, parallel=4)
    """

    def __init__(
        self,
        baseline_dir: Path | str,
        scenarios_root: Path | str | None = None,
        main_file_name: str = "Simulation.in",
    ) -> None:
        """Initialize the scenario manager."""
        self.baseline_dir = Path(baseline_dir).resolve()
        if not self.baseline_dir.exists():
            raise FileNotFoundError(f"Baseline directory not found: {self.baseline_dir}")

        self.scenarios_root = (
            Path(scenarios_root).resolve()
            if scenarios_root
            else self.baseline_dir.parent / "scenarios"
        )
        self.main_file_name = main_file_name

        # Validate main file exists
        main_file = self.baseline_dir / main_file_name
        if not main_file.exists():
            # Try to find it
            candidates = list(self.baseline_dir.glob("*.in"))
            if candidates:
                self.main_file_name = candidates[0].name

    def create_scenario_dir(
        self,
        scenario: Scenario,
        copy_outputs: bool = False,
    ) -> Path:
        """Create a directory for a scenario run.

        Copies the baseline model files to a new directory and
        applies the scenario modifications.

        Parameters
        ----------
        scenario : Scenario
            The scenario to create.
        copy_outputs : bool
            If True, also copy output files from baseline.

        Returns
        -------
        Path
            Path to the created scenario directory.
        """
        scenario_dir = self.scenarios_root / scenario.name

        # Remove existing scenario dir if present
        if scenario_dir.exists():
            shutil.rmtree(scenario_dir)

        # Copy baseline to scenario directory
        def ignore_outputs(directory: str, files: list[str]) -> list[str]:
            """Ignore output files when copying."""
            if copy_outputs:
                return []
            ignored = []
            for f in files:
                # Skip large output files
                if any(f.endswith(ext) for ext in [".hdf", ".bin", ".BUD", ".out"]):
                    # But keep input files that might have these extensions
                    path = Path(directory) / f
                    if path.stat().st_size > 10_000_000:  # 10MB threshold
                        ignored.append(f)
            return ignored

        shutil.copytree(
            self.baseline_dir,
            scenario_dir,
            ignore=ignore_outputs,
        )

        # Apply modifications
        self._apply_modifications(scenario_dir, scenario)

        return scenario_dir

    def _apply_modifications(
        self,
        scenario_dir: Path,
        scenario: Scenario,
    ) -> None:
        """Apply scenario modifications to input files.

        Parameters
        ----------
        scenario_dir : Path
            Directory containing scenario files.
        scenario : Scenario
            Scenario with modifications to apply.
        """
        # Apply custom modifier function if provided
        if scenario.modifier_func:
            scenario.modifier_func(scenario_dir, self.baseline_dir)

        # Apply standard modifications
        for mod_type, mod_value in scenario.modifications.items():
            if mod_type == "pumping":
                self._modify_pumping(scenario_dir, mod_value)
            elif mod_type == "diversion":
                self._modify_diversions(scenario_dir, mod_value)
            elif mod_type == "recharge":
                self._modify_recharge(scenario_dir, mod_value)
            elif mod_type == "stream_inflow":
                self._modify_stream_inflow(scenario_dir, mod_value)
            # Add more modification types as needed

    def _modify_pumping(
        self,
        scenario_dir: Path,
        factor: float | dict[int, float],
    ) -> None:
        """Modify pumping rates in scenario.

        Parameters
        ----------
        scenario_dir : Path
            Scenario directory.
        factor : float | dict[int, float]
            Multiplier for pumping rates. If dict, keys are well IDs.
        """
        # Find pumping files
        pumping_files = list(scenario_dir.glob("*Pump*.dat")) + \
                        list(scenario_dir.glob("*Pumping*.dat"))

        for pump_file in pumping_files:
            self._apply_factor_to_timeseries(pump_file, factor)

    def _modify_diversions(
        self,
        scenario_dir: Path,
        factor: float | dict[int, float],
    ) -> None:
        """Modify diversion rates in scenario.

        Parameters
        ----------
        scenario_dir : Path
            Scenario directory.
        factor : float | dict[int, float]
            Multiplier for diversion rates.
        """
        # Find diversion files
        div_files = list(scenario_dir.glob("*Diversion*.dat")) + \
                    list(scenario_dir.glob("*Diversions*.dat"))

        for div_file in div_files:
            self._apply_factor_to_timeseries(div_file, factor)

    def _modify_recharge(
        self,
        scenario_dir: Path,
        factor: float,
    ) -> None:
        """Modify recharge rates in scenario.

        Parameters
        ----------
        scenario_dir : Path
            Scenario directory.
        factor : float
            Multiplier for recharge rates.
        """
        # Find recharge-related files
        recharge_files = list(scenario_dir.glob("*Recharge*.dat")) + \
                         list(scenario_dir.glob("*DeepPerc*.dat"))

        for rech_file in recharge_files:
            self._apply_factor_to_timeseries(rech_file, factor)

    def _modify_stream_inflow(
        self,
        scenario_dir: Path,
        factor: float,
    ) -> None:
        """Modify stream inflow rates in scenario.

        Parameters
        ----------
        scenario_dir : Path
            Scenario directory.
        factor : float
            Multiplier for stream inflow rates.
        """
        # Find stream inflow files
        inflow_files = list(scenario_dir.glob("*StreamInflow*.dat")) + \
                       list(scenario_dir.glob("*Inflows*.dat"))

        for inflow_file in inflow_files:
            self._apply_factor_to_timeseries(inflow_file, factor)

    def _apply_factor_to_timeseries(
        self,
        filepath: Path,
        factor: float | dict[int, float],
    ) -> None:
        """Apply a multiplication factor to time series values in a file.

        This is a simple implementation that multiplies numeric values.
        For more complex modifications, use a custom modifier_func.

        Parameters
        ----------
        filepath : Path
            Path to the time series file.
        factor : float | dict[int, float]
            Multiplication factor(s) to apply.
        """
        if not filepath.exists():
            return

        try:
            content = filepath.read_text()
            lines = content.splitlines()
            modified_lines = []

            for line in lines:
                # Skip comment lines
                if line.strip().startswith(("C", "c", "*", "!")):
                    modified_lines.append(line)
                    continue

                # Try to modify numeric values
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        # Assume first column is date/time, rest are values
                        new_parts = [parts[0]]  # Keep timestamp
                        for i, part in enumerate(parts[1:], 1):
                            try:
                                value = float(part)
                                if isinstance(factor, dict):
                                    # Use column-specific factor if available
                                    mult = factor.get(i, 1.0)
                                else:
                                    mult = factor
                                new_parts.append(f"{value * mult:.6g}")
                            except ValueError:
                                new_parts.append(part)
                        modified_lines.append("  ".join(new_parts))
                    except (ValueError, IndexError):
                        modified_lines.append(line)
                else:
                    modified_lines.append(line)

            filepath.write_text("\n".join(modified_lines))

        except Exception:
            # If modification fails, leave file unchanged
            pass

    def run_scenario(
        self,
        scenario: Scenario,
        runner: "IWFMRunner | None" = None,
        timeout: float | None = None,
        cleanup_on_success: bool = False,
    ) -> ScenarioResult:
        """Run a single scenario.

        Parameters
        ----------
        scenario : Scenario
            The scenario to run.
        runner : IWFMRunner | None
            Runner to use. If None, creates a new one.
        timeout : float | None
            Timeout in seconds for the simulation.
        cleanup_on_success : bool
            If True, remove scenario directory after successful run.

        Returns
        -------
        ScenarioResult
            Result of the scenario run.
        """
        from pyiwfm.runner.runner import IWFMRunner

        if runner is None:
            runner = IWFMRunner()

        # Create scenario directory
        scenario_dir = self.create_scenario_dir(scenario)

        # Run simulation
        main_file = scenario_dir / self.main_file_name
        result = runner.run_simulation(main_file, timeout=timeout)

        # Optionally cleanup
        if cleanup_on_success and result.success:
            # Keep just the essential output files
            pass  # Don't delete for now

        return ScenarioResult(
            scenario=scenario,
            result=result,
            scenario_dir=scenario_dir,
        )

    def run_scenarios(
        self,
        scenarios: list[Scenario],
        runner: "IWFMRunner | None" = None,
        parallel: int = 1,
        timeout: float | None = None,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> dict[str, ScenarioResult]:
        """Run multiple scenarios, optionally in parallel.

        Parameters
        ----------
        scenarios : list[Scenario]
            List of scenarios to run.
        runner : IWFMRunner | None
            Runner to use. If None, creates a new one.
        parallel : int
            Number of parallel workers. 1 = sequential.
        timeout : float | None
            Timeout per scenario in seconds.
        progress_callback : Callable | None
            Optional callback for progress updates.
            Signature: (scenario_name, completed, total) -> None

        Returns
        -------
        dict[str, ScenarioResult]
            Dictionary mapping scenario names to results.
        """
        from pyiwfm.runner.runner import IWFMRunner

        if runner is None:
            runner = IWFMRunner()

        results: dict[str, ScenarioResult] = {}
        total = len(scenarios)

        if parallel <= 1:
            # Sequential execution
            for i, scenario in enumerate(scenarios):
                if progress_callback:
                    progress_callback(scenario.name, i, total)

                result = self.run_scenario(scenario, runner, timeout)
                results[scenario.name] = result

                if progress_callback:
                    progress_callback(scenario.name, i + 1, total)
        else:
            # Parallel execution
            # Note: Each process needs its own runner instance
            with ProcessPoolExecutor(max_workers=parallel) as executor:
                # Submit all scenarios
                future_to_scenario = {
                    executor.submit(
                        self._run_scenario_worker,
                        scenario,
                        timeout,
                    ): scenario
                    for scenario in scenarios
                }

                completed = 0
                for future in as_completed(future_to_scenario):
                    scenario = future_to_scenario[future]
                    try:
                        result = future.result()
                        results[scenario.name] = result
                    except Exception as e:
                        # Create a failed result
                        from pyiwfm.runner.results import SimulationResult
                        from datetime import timedelta

                        results[scenario.name] = ScenarioResult(
                            scenario=scenario,
                            result=SimulationResult(
                                success=False,
                                return_code=-1,
                                stderr=str(e),
                                errors=[str(e)],
                            ),
                            scenario_dir=self.scenarios_root / scenario.name,
                        )

                    completed += 1
                    if progress_callback:
                        progress_callback(scenario.name, completed, total)

        return results

    def _run_scenario_worker(
        self,
        scenario: Scenario,
        timeout: float | None,
    ) -> ScenarioResult:
        """Worker function for parallel scenario execution.

        This runs in a separate process, so it creates its own runner.
        """
        from pyiwfm.runner.runner import IWFMRunner

        runner = IWFMRunner()
        return self.run_scenario(scenario, runner, timeout)

    def compare_to_baseline(
        self,
        baseline_result: SimulationResult,
        scenario_results: dict[str, ScenarioResult],
    ) -> dict[str, dict[str, Any]]:
        """Compare scenario results to baseline.

        This is a placeholder for comparison logic. Actual comparison
        depends on the specific outputs being analyzed (budgets,
        hydrographs, etc.).

        Parameters
        ----------
        baseline_result : SimulationResult
            Result from the baseline simulation.
        scenario_results : dict[str, ScenarioResult]
            Results from scenario runs.

        Returns
        -------
        dict[str, dict[str, Any]]
            Comparison results keyed by scenario name.
        """
        comparisons: dict[str, dict[str, Any]] = {}

        for name, scenario_result in scenario_results.items():
            comparison: dict[str, Any] = {
                "success": scenario_result.success,
                "elapsed_time_ratio": (
                    scenario_result.result.elapsed_time.total_seconds() /
                    baseline_result.elapsed_time.total_seconds()
                    if baseline_result.elapsed_time.total_seconds() > 0
                    else 0
                ),
            }

            # Add more comparison metrics as needed:
            # - Budget differences
            # - Head differences at observation wells
            # - Stream flow differences

            comparisons[name] = comparison
            scenario_result.differences = comparison

        return comparisons

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"ScenarioManager(baseline_dir='{self.baseline_dir}', "
            f"main_file='{self.main_file_name}')"
        )
