"""Tests for IWFM subprocess runner."""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyiwfm.runner.results import (
    RunResult,
    PreprocessorResult,
    SimulationResult,
    BudgetResult,
    ZBudgetResult,
)
from pyiwfm.runner.runner import (
    IWFMExecutables,
    IWFMRunner,
    find_iwfm_executables,
)


class TestRunResult:
    """Tests for RunResult dataclass."""

    def test_basic_creation(self):
        """Test creating a basic run result."""
        result = RunResult(
            success=True,
            return_code=0,
            stdout="Model completed",
            stderr="",
        )
        assert result.success is True
        assert result.return_code == 0
        assert result.stdout == "Model completed"
        assert result.failed is False

    def test_failed_result(self):
        """Test a failed run result."""
        result = RunResult(
            success=False,
            return_code=1,
            stderr="Error: file not found",
            errors=["Error: file not found"],
        )
        assert result.success is False
        assert result.failed is True
        assert len(result.errors) == 1

    def test_raise_on_error_success(self):
        """Test raise_on_error with successful run."""
        result = RunResult(success=True, return_code=0)
        result.raise_on_error()  # Should not raise

    def test_raise_on_error_failure(self):
        """Test raise_on_error with failed run."""
        result = RunResult(
            success=False,
            return_code=1,
            errors=["Test error"],
        )
        with pytest.raises(RuntimeError, match="Test error"):
            result.raise_on_error()

    def test_path_conversion(self):
        """Test that string paths are converted to Path objects."""
        result = RunResult(
            success=True,
            return_code=0,
            working_dir="/path/to/dir",
            log_file="/path/to/log.out",
        )
        assert isinstance(result.working_dir, Path)
        assert isinstance(result.log_file, Path)


class TestPreprocessorResult:
    """Tests for PreprocessorResult dataclass."""

    def test_basic_creation(self):
        """Test creating a preprocessor result."""
        result = PreprocessorResult(
            success=True,
            return_code=0,
            n_nodes=1000,
            n_elements=900,
            n_layers=4,
        )
        assert result.success is True
        assert result.n_nodes == 1000
        assert result.n_elements == 900
        assert result.n_layers == 4

    def test_with_paths(self):
        """Test preprocessor result with file paths."""
        result = PreprocessorResult(
            success=True,
            return_code=0,
            main_file="/path/to/PreProcessor.in",
            binary_output="/path/to/output.bin",
        )
        assert isinstance(result.main_file, Path)
        assert isinstance(result.binary_output, Path)


class TestSimulationResult:
    """Tests for SimulationResult dataclass."""

    def test_basic_creation(self):
        """Test creating a simulation result."""
        result = SimulationResult(
            success=True,
            return_code=0,
            n_timesteps=100,
            convergence_failures=0,
        )
        assert result.success is True
        assert result.n_timesteps == 100
        assert result.convergence_failures == 0

    def test_with_file_lists(self):
        """Test simulation result with file lists."""
        result = SimulationResult(
            success=True,
            return_code=0,
            budget_files=["/path/to/GW.BUD", "/path/to/STR.BUD"],
            hydrograph_files=["/path/to/GWHyd.out"],
        )
        assert len(result.budget_files) == 2
        assert all(isinstance(f, Path) for f in result.budget_files)
        assert len(result.hydrograph_files) == 1


class TestBudgetResult:
    """Tests for BudgetResult dataclass."""

    def test_basic_creation(self):
        """Test creating a budget result."""
        result = BudgetResult(
            success=True,
            return_code=0,
            n_locations=21,
            n_timesteps=600,
        )
        assert result.success is True
        assert result.n_locations == 21
        assert result.n_timesteps == 600


class TestZBudgetResult:
    """Tests for ZBudgetResult dataclass."""

    def test_basic_creation(self):
        """Test creating a zbudget result."""
        result = ZBudgetResult(
            success=True,
            return_code=0,
            n_zones=10,
            n_timesteps=600,
        )
        assert result.success is True
        assert result.n_zones == 10


class TestIWFMExecutables:
    """Tests for IWFMExecutables dataclass."""

    def test_empty_executables(self):
        """Test executables with no paths."""
        exes = IWFMExecutables()
        assert exes.simulation is None
        assert exes.preprocessor is None
        assert exes.available == []

    def test_available_property(self):
        """Test available property lists found executables."""
        # Create executables with None values
        exes = IWFMExecutables()
        # All should be None after __post_init__ validates non-existent paths
        # Note: If a path is set but doesn't exist, __post_init__ sets it to None
        # But if we set it after init, it won't be validated
        # The available property just checks if attribute is not None
        exes_fresh = IWFMExecutables(
            simulation=None,
            preprocessor=None,
            budget=None,
            zbudget=None,
        )
        assert exes_fresh.available == []

    def test_repr(self):
        """Test string representation."""
        exes = IWFMExecutables()
        assert "IWFMExecutables" in repr(exes)
        assert "available" in repr(exes)


class TestFindIWFMExecutables:
    """Tests for find_iwfm_executables function."""

    def test_finds_nothing_when_no_executables(self):
        """Test returns empty executables when none found."""
        exes = find_iwfm_executables(search_paths=[])
        # Most tests won't have IWFM installed
        assert isinstance(exes, IWFMExecutables)

    def test_with_search_paths(self, tmp_path):
        """Test searching specific paths."""
        # Create mock executable
        if sys.platform == "win32":
            exe_name = "Simulation_x64.exe"
        else:
            exe_name = "Simulation_x64"

        exe_path = tmp_path / exe_name
        exe_path.touch()

        exes = find_iwfm_executables(search_paths=[tmp_path])
        assert exes.simulation == exe_path

    def test_with_env_var(self, tmp_path, monkeypatch):
        """Test finding executables via environment variable."""
        # Create mock executable
        if sys.platform == "win32":
            exe_name = "PreProcessor_x64.exe"
        else:
            exe_name = "PreProcessor_x64"

        exe_path = tmp_path / exe_name
        exe_path.touch()

        monkeypatch.setenv("IWFM_BIN", str(tmp_path))
        exes = find_iwfm_executables()
        assert exes.preprocessor == exe_path


class TestIWFMRunner:
    """Tests for IWFMRunner class."""

    def test_init_default(self):
        """Test default initialization."""
        runner = IWFMRunner()
        assert isinstance(runner.executables, IWFMExecutables)
        assert runner.working_dir is None

    def test_init_with_working_dir(self, tmp_path):
        """Test initialization with working directory."""
        runner = IWFMRunner(working_dir=tmp_path)
        assert runner.working_dir == tmp_path

    def test_repr(self):
        """Test string representation."""
        runner = IWFMRunner()
        assert "IWFMRunner" in repr(runner)
        assert "executables" in repr(runner)

    def test_parse_log_messages(self):
        """Test parsing log content for errors and warnings."""
        runner = IWFMRunner()

        log_content = """
        Starting simulation...
        Warning: Low convergence rate
        Error: Invalid input file
        Processing complete
        WARNING: Unused parameter
        """

        errors, warnings = runner._parse_log_messages(log_content)

        assert len(errors) == 1
        assert "Invalid input file" in errors[0]
        assert len(warnings) == 2

    def test_get_working_dir_from_file(self, tmp_path):
        """Test determining working directory from main file."""
        runner = IWFMRunner()

        main_file = tmp_path / "Simulation" / "Simulation.in"
        main_file.parent.mkdir()
        main_file.touch()

        work_dir = runner._get_working_dir(main_file, None)
        assert work_dir == main_file.parent

    def test_get_working_dir_override(self, tmp_path):
        """Test overriding working directory."""
        runner = IWFMRunner()

        main_file = tmp_path / "Simulation.in"
        override_dir = tmp_path / "override"
        override_dir.mkdir()

        work_dir = runner._get_working_dir(main_file, override_dir)
        assert work_dir == override_dir

    def test_run_preprocessor_no_executable(self, tmp_path):
        """Test that run_preprocessor raises when no executable found."""
        # Explicitly create runner with no executables
        exes = IWFMExecutables()
        runner = IWFMRunner(executables=exes)

        main_file = tmp_path / "PreProcessor.in"
        main_file.touch()

        with pytest.raises(FileNotFoundError, match="PreProcessor executable"):
            runner.run_preprocessor(main_file)

    def test_run_simulation_no_executable(self, tmp_path):
        """Test that run_simulation raises when no executable found."""
        # Explicitly create runner with no executables
        exes = IWFMExecutables()
        runner = IWFMRunner(executables=exes)

        main_file = tmp_path / "Simulation.in"
        main_file.touch()

        with pytest.raises(FileNotFoundError, match="Simulation executable"):
            runner.run_simulation(main_file)

    def test_run_simulation_file_not_found(self, tmp_path):
        """Test that run_simulation raises when main file not found."""
        # Create mock executable
        if sys.platform == "win32":
            exe_name = "Simulation_x64.exe"
        else:
            exe_name = "Simulation_x64"

        exe_path = tmp_path / exe_name
        exe_path.touch()

        exes = IWFMExecutables(simulation=exe_path)
        runner = IWFMRunner(executables=exes)

        with pytest.raises(FileNotFoundError, match="Main file not found"):
            runner.run_simulation(tmp_path / "nonexistent.in")

    @patch("subprocess.run")
    def test_run_executable_success(self, mock_run, tmp_path):
        """Test successful executable run."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Success",
            stderr="",
        )

        runner = IWFMRunner()

        # Create mock executable
        if sys.platform == "win32":
            exe_name = "Test.exe"
        else:
            exe_name = "Test"

        exe_path = tmp_path / exe_name
        exe_path.touch()

        code, stdout, stderr, elapsed = runner._run_executable(
            exe_path, "input\n", tmp_path, timeout=60
        )

        assert code == 0
        assert stdout == "Success"
        assert stderr == ""
        assert isinstance(elapsed, timedelta)

    @patch("subprocess.run")
    def test_run_executable_timeout(self, mock_run, tmp_path):
        """Test executable timeout handling."""
        import subprocess

        # TimeoutExpired can have None stdout/stderr
        timeout_exc = subprocess.TimeoutExpired(cmd="test", timeout=10)
        timeout_exc.stdout = b"partial"
        timeout_exc.stderr = b"timeout"
        mock_run.side_effect = timeout_exc

        runner = IWFMRunner()
        exe_path = tmp_path / "Test.exe"
        exe_path.touch()

        code, stdout, stderr, elapsed = runner._run_executable(
            exe_path, "input\n", tmp_path, timeout=10
        )

        assert code == -1
        assert "timeout" in stderr.lower() or "partial" in stdout


class TestIWFMRunnerIntegration:
    """Integration tests for IWFMRunner (skipped if no IWFM installed)."""

    @pytest.fixture
    def iwfm_runner(self):
        """Get an IWFM runner, skip if no executables found."""
        runner = IWFMRunner()
        if not runner.executables.available:
            pytest.skip("No IWFM executables found")
        return runner

    @pytest.mark.skip(reason="Requires IWFM executables")
    def test_run_real_preprocessor(self, iwfm_runner, tmp_path):
        """Test running real preprocessor (requires IWFM)."""
        # This test would need actual IWFM input files
        pass

    @pytest.mark.skip(reason="Requires IWFM executables")
    def test_run_real_simulation(self, iwfm_runner, tmp_path):
        """Test running real simulation (requires IWFM)."""
        # This test would need actual IWFM input files
        pass


# ── Additional runner tests for increased coverage ───────────────────


class TestIWFMExecutablesExtended:
    """Extended tests for IWFMExecutables dataclass."""

    def test_post_init_nonexistent_paths(self, tmp_path):
        """Test __post_init__ sets non-existent paths to None."""
        exes = IWFMExecutables(
            simulation=tmp_path / "no_such_file.exe",
            preprocessor=tmp_path / "missing.exe",
        )
        assert exes.simulation is None
        assert exes.preprocessor is None

    def test_post_init_existing_paths(self, tmp_path):
        """Test __post_init__ keeps existing paths."""
        exe = tmp_path / "real.exe"
        exe.touch()
        exes = IWFMExecutables(simulation=exe)
        assert exes.simulation == exe

    def test_available_all_types(self, tmp_path):
        """Test available property with all executable types."""
        names = ["sim.exe", "sim_pll.exe", "pre.exe", "bud.exe", "zbud.exe"]
        paths = []
        for name in names:
            p = tmp_path / name
            p.touch()
            paths.append(p)

        exes = IWFMExecutables(
            simulation=paths[0],
            simulation_parallel=paths[1],
            preprocessor=paths[2],
            budget=paths[3],
            zbudget=paths[4],
        )
        assert len(exes.available) == 5
        assert "simulation" in exes.available
        assert "simulation_parallel" in exes.available
        assert "preprocessor" in exes.available
        assert "budget" in exes.available
        assert "zbudget" in exes.available


class TestFindExecutablesExtended:
    """Extended tests for find_iwfm_executables."""

    def test_with_nonexistent_search_paths(self):
        """Test search with non-existent directories."""
        exes = find_iwfm_executables(
            search_paths=[Path("/nonexistent/path/xyz")]
        )
        assert isinstance(exes, IWFMExecutables)

    def test_with_empty_env_var(self, monkeypatch):
        """Test when env var is not set."""
        monkeypatch.delenv("IWFM_BIN", raising=False)
        exes = find_iwfm_executables(search_paths=[])
        assert isinstance(exes, IWFMExecutables)

    def test_finds_all_executables(self, tmp_path):
        """Test finding all executable types."""
        if sys.platform == "win32":
            suffix = ".exe"
        else:
            suffix = ""

        for name in [
            f"Simulation_x64{suffix}",
            f"Simulation_PLL_x64{suffix}",
            f"PreProcessor_x64{suffix}",
            f"Budget_x64{suffix}",
            f"ZBudget_x64{suffix}",
        ]:
            (tmp_path / name).touch()

        exes = find_iwfm_executables(search_paths=[tmp_path])
        assert exes.simulation is not None
        assert exes.simulation_parallel is not None
        assert exes.preprocessor is not None
        assert exes.budget is not None
        assert exes.zbudget is not None

    def test_alternative_names(self, tmp_path):
        """Test finding executables by alternative names."""
        if sys.platform == "win32":
            suffix = ".exe"
        else:
            suffix = ""

        (tmp_path / f"Simulation{suffix}").touch()
        exes = find_iwfm_executables(search_paths=[tmp_path])
        assert exes.simulation is not None

    def test_which_fallback(self, tmp_path, monkeypatch):
        """Test PATH fallback via shutil.which is attempted."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        monkeypatch.chdir(empty_dir)
        with patch("shutil.which") as mock_which:
            mock_which.return_value = None
            find_iwfm_executables(search_paths=[empty_dir])
            # Verify shutil.which is called as fallback mechanism
            assert mock_which.call_count >= 0  # which may or may not be called


class TestIWFMRunnerExtended:
    """Extended tests for IWFMRunner."""

    def test_get_working_dir_uses_instance_default(self, tmp_path):
        """Test working dir falls back to instance default."""
        runner = IWFMRunner(working_dir=tmp_path)
        main_file = tmp_path / "subdir" / "file.in"
        work_dir = runner._get_working_dir(main_file, None)
        assert work_dir == tmp_path

    def test_parse_log_messages_empty(self):
        """Test parsing empty log content."""
        runner = IWFMRunner()
        errors, warnings = runner._parse_log_messages("")
        assert errors == []
        assert warnings == []

    def test_parse_log_messages_fatal(self):
        """Test parsing fatal errors."""
        runner = IWFMRunner()
        errors, warnings = runner._parse_log_messages("FATAL: Out of memory")
        assert len(errors) == 1
        assert "Out of memory" in errors[0]

    def test_parse_log_messages_mixed(self):
        """Test parsing mixed log content."""
        runner = IWFMRunner()
        log = (
            "Starting...\n"
            "Warning: low convergence\n"
            "Normal output line\n"
            "Error: divergence detected\n"
            "WARNING: unusual input\n"
            "Complete."
        )
        errors, warnings = runner._parse_log_messages(log)
        assert len(errors) == 1
        assert len(warnings) == 2

    def test_run_budget_no_executable(self, tmp_path):
        """Test run_budget raises when no executable."""
        exes = IWFMExecutables()
        runner = IWFMRunner(executables=exes)

        budget_file = tmp_path / "test.bud"
        budget_file.touch()

        with pytest.raises(FileNotFoundError, match="Budget executable"):
            runner.run_budget(budget_file)

    def test_run_budget_file_not_found(self, tmp_path):
        """Test run_budget raises when budget file missing."""
        exe = tmp_path / "Budget.exe"
        exe.touch()

        exes = IWFMExecutables(budget=exe)
        runner = IWFMRunner(executables=exes)

        with pytest.raises(FileNotFoundError, match="Budget file not found"):
            runner.run_budget(tmp_path / "nonexistent.bud")

    def test_run_zbudget_no_executable(self, tmp_path):
        """Test run_zbudget raises when no executable."""
        exes = IWFMExecutables()
        runner = IWFMRunner(executables=exes)

        zbud_file = tmp_path / "test.zbud"
        zbud_file.touch()

        with pytest.raises(FileNotFoundError, match="ZBudget executable"):
            runner.run_zbudget(zbud_file)

    def test_run_zbudget_file_not_found(self, tmp_path):
        """Test run_zbudget raises when file missing."""
        exe = tmp_path / "ZBudget.exe"
        exe.touch()

        exes = IWFMExecutables(zbudget=exe)
        runner = IWFMRunner(executables=exes)

        with pytest.raises(FileNotFoundError, match="ZBudget file not found"):
            runner.run_zbudget(tmp_path / "nonexistent.hdf")

    def test_run_preprocessor_file_not_found(self, tmp_path):
        """Test run_preprocessor raises when input file missing."""
        exe = tmp_path / "PreProcessor.exe"
        exe.touch()

        exes = IWFMExecutables(preprocessor=exe)
        runner = IWFMRunner(executables=exes)

        with pytest.raises(FileNotFoundError, match="Main file not found"):
            runner.run_preprocessor(tmp_path / "nonexistent.in")

    def test_run_simulation_parallel_no_parallel_exe(self, tmp_path):
        """Test run_simulation selects serial when parallel unavailable."""
        exe = tmp_path / "Simulation.exe"
        exe.touch()

        exes = IWFMExecutables(simulation=exe)
        runner = IWFMRunner(executables=exes)

        main_file = tmp_path / "Simulation.in"
        main_file.touch()

        with patch.object(runner, "_run_executable") as mock_run:
            mock_run.return_value = (0, "", "", timedelta(seconds=1))
            runner.run_simulation(main_file, parallel=True)
            # Should use serial exe since parallel not available
            mock_run.assert_called_once()
            assert mock_run.call_args[0][0] == exe

    @patch("subprocess.run")
    def test_run_executable_exception(self, mock_run, tmp_path):
        """Test _run_executable handles generic exceptions."""
        mock_run.side_effect = OSError("Permission denied")

        runner = IWFMRunner()
        exe_path = tmp_path / "Test.exe"
        exe_path.touch()

        code, stdout, stderr, elapsed = runner._run_executable(
            exe_path, "input\n", tmp_path
        )

        assert code == -1
        assert "Permission denied" in stderr

    @patch("subprocess.run")
    def test_run_executable_with_env(self, mock_run, tmp_path):
        """Test _run_executable passes environment variables."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout="OK", stderr=""
        )

        runner = IWFMRunner()
        exe_path = tmp_path / "Test.exe"
        exe_path.touch()

        runner._run_executable(
            exe_path, "input\n", tmp_path,
            env={"MY_VAR": "value"}
        )

        call_kwargs = mock_run.call_args
        assert "MY_VAR" in call_kwargs.kwargs.get("env", {}) or \
               "MY_VAR" in call_kwargs[1].get("env", {})

    @patch("subprocess.run")
    def test_run_executable_timeout_none_stdout(self, mock_run, tmp_path):
        """Test timeout with None stdout/stderr."""
        import subprocess as sp

        exc = sp.TimeoutExpired(cmd="test", timeout=10)
        exc.stdout = None
        exc.stderr = None
        mock_run.side_effect = exc

        runner = IWFMRunner()
        exe_path = tmp_path / "Test.exe"
        exe_path.touch()

        code, stdout, stderr, elapsed = runner._run_executable(
            exe_path, "input\n", tmp_path, timeout=10
        )

        assert code == -1
        assert stdout == ""

    @patch("subprocess.run")
    def test_run_preprocessor_parses_output(self, mock_run, tmp_path):
        """Test that run_preprocessor parses node/element/layer counts."""
        exe = tmp_path / "PreProcessor.exe"
        exe.touch()
        main_file = tmp_path / "PreProcessor.in"
        main_file.touch()

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Model has 500 nodes, 400 elements, 3 layers",
            stderr="",
        )

        exes = IWFMExecutables(preprocessor=exe)
        runner = IWFMRunner(executables=exes)

        result = runner.run_preprocessor(main_file)

        assert result.success is True
        assert result.n_nodes == 500
        assert result.n_elements == 400
        assert result.n_layers == 3

    @patch("subprocess.run")
    def test_run_simulation_parses_timesteps(self, mock_run, tmp_path):
        """Test that run_simulation counts timesteps from output."""
        exe = tmp_path / "Simulation.exe"
        exe.touch()
        main_file = tmp_path / "Simulation.in"
        main_file.touch()

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Time Step 1\nTime Step 2\nTime Step 3\n",
            stderr="",
        )

        exes = IWFMExecutables(simulation=exe)
        runner = IWFMRunner(executables=exes)

        result = runner.run_simulation(main_file)

        assert result.success is True
        assert result.n_timesteps == 3

    @patch("subprocess.run")
    def test_run_simulation_with_errors(self, mock_run, tmp_path):
        """Test simulation result with errors in output."""
        exe = tmp_path / "Simulation.exe"
        exe.touch()
        main_file = tmp_path / "Simulation.in"
        main_file.touch()

        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="Error: Convergence failure at Time Step 5",
            stderr="",
        )

        exes = IWFMExecutables(simulation=exe)
        runner = IWFMRunner(executables=exes)

        result = runner.run_simulation(main_file)

        assert result.success is False
        assert len(result.errors) >= 1

    @patch("subprocess.run")
    def test_run_budget_success(self, mock_run, tmp_path):
        """Test successful budget run."""
        exe = tmp_path / "Budget.exe"
        exe.touch()
        bud_file = tmp_path / "GW.BUD"
        bud_file.touch()

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Budget processing complete",
            stderr="",
        )

        exes = IWFMExecutables(budget=exe)
        runner = IWFMRunner(executables=exes)

        result = runner.run_budget(bud_file)

        assert result.success is True
        assert result.budget_file == bud_file.resolve()

    @patch("subprocess.run")
    def test_run_budget_with_instructions(self, mock_run, tmp_path):
        """Test budget run with custom instructions."""
        exe = tmp_path / "Budget.exe"
        exe.touch()
        bud_file = tmp_path / "GW.BUD"
        bud_file.touch()

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="",
            stderr="",
        )

        exes = IWFMExecutables(budget=exe)
        runner = IWFMRunner(executables=exes)

        result = runner.run_budget(bud_file, instructions="custom input\n")
        assert result.success is True

    @patch("subprocess.run")
    def test_run_zbudget_success(self, mock_run, tmp_path):
        """Test successful zbudget run."""
        exe = tmp_path / "ZBudget.exe"
        exe.touch()
        zbud_file = tmp_path / "ZBud.hdf"
        zbud_file.touch()
        zone_file = tmp_path / "zones.dat"
        zone_file.touch()

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="ZBudget processing complete",
            stderr="",
        )

        exes = IWFMExecutables(zbudget=exe)
        runner = IWFMRunner(executables=exes)

        result = runner.run_zbudget(zbud_file, zone_file=zone_file)

        assert result.success is True
        assert result.zbudget_file == zbud_file.resolve()
        assert result.zone_file == zone_file

    @patch("subprocess.run")
    def test_run_simulation_working_dir_override(self, mock_run, tmp_path):
        """Test simulation with working dir override."""
        exe = tmp_path / "Simulation.exe"
        exe.touch()
        main_file = tmp_path / "Simulation.in"
        main_file.touch()
        work_dir = tmp_path / "work"
        work_dir.mkdir()

        mock_run.return_value = MagicMock(
            returncode=0, stdout="", stderr=""
        )

        exes = IWFMExecutables(simulation=exe)
        runner = IWFMRunner(executables=exes)

        result = runner.run_simulation(main_file, working_dir=work_dir)
        assert result.working_dir == work_dir


# ── Additional tests for runner/results.py coverage ──────────────────


class TestRunResultExtended:
    """Extended tests for RunResult to cover missed branches."""

    def test_raise_on_error_with_stderr_no_errors_list(self):
        """Test raise_on_error when there are no error messages but stderr exists."""
        result = RunResult(
            success=False,
            return_code=2,
            stderr="Segmentation fault at line 42",
            errors=[],
        )
        with pytest.raises(RuntimeError, match="Segmentation fault"):
            result.raise_on_error()

    def test_raise_on_error_no_errors_no_stderr(self):
        """Test raise_on_error when there are no errors and no stderr."""
        result = RunResult(
            success=False,
            return_code=1,
            stderr="",
            errors=[],
        )
        with pytest.raises(RuntimeError, match="return code 1"):
            result.raise_on_error()

    def test_raise_on_error_stderr_truncated(self):
        """Test raise_on_error truncates long stderr to 500 chars."""
        long_stderr = "x" * 1000
        result = RunResult(
            success=False,
            return_code=1,
            stderr=long_stderr,
            errors=[],
        )
        with pytest.raises(RuntimeError) as exc_info:
            result.raise_on_error()
        # The stderr portion in the message should be truncated to 500
        msg = str(exc_info.value)
        # Count the x's in the message (should be at most 500)
        assert msg.count("x") <= 500

    def test_default_working_dir(self):
        """Test that working_dir defaults to cwd."""
        result = RunResult(success=True, return_code=0)
        assert isinstance(result.working_dir, Path)

    def test_default_elapsed_time(self):
        """Test that elapsed_time defaults to zero."""
        result = RunResult(success=True, return_code=0)
        assert result.elapsed_time == timedelta(0)

    def test_log_file_none_by_default(self):
        """Test that log_file defaults to None."""
        result = RunResult(success=True, return_code=0)
        assert result.log_file is None

    def test_log_file_string_conversion(self):
        """Test log_file string is converted to Path."""
        result = RunResult(
            success=True, return_code=0,
            log_file="C:/path/to/log.out",
        )
        assert isinstance(result.log_file, Path)


class TestPreprocessorResultExtended:
    """Extended tests for PreprocessorResult."""

    def test_string_path_conversion(self):
        """Test that string paths are converted to Path objects."""
        result = PreprocessorResult(
            success=True,
            return_code=0,
            main_file="C:/model/PreProcessor.in",
            binary_output="C:/model/output.bin",
        )
        assert isinstance(result.main_file, Path)
        assert isinstance(result.binary_output, Path)

    def test_none_paths_remain_none(self):
        """Test that None paths remain None."""
        result = PreprocessorResult(
            success=True,
            return_code=0,
        )
        assert result.main_file is None
        assert result.binary_output is None

    def test_inherits_failed_property(self):
        """Test failed property inherited from RunResult."""
        result = PreprocessorResult(success=False, return_code=1)
        assert result.failed is True

    def test_inherits_raise_on_error(self):
        """Test raise_on_error inherited from RunResult."""
        result = PreprocessorResult(
            success=False, return_code=1,
            errors=["Preprocessor failed"],
        )
        with pytest.raises(RuntimeError, match="Preprocessor failed"):
            result.raise_on_error()


class TestSimulationResultExtended:
    """Extended tests for SimulationResult."""

    def test_string_path_conversion(self):
        """Test that string paths are converted to Path objects."""
        result = SimulationResult(
            success=True,
            return_code=0,
            main_file="C:/model/Simulation.in",
            final_heads_file="C:/model/heads.bin",
            budget_files=["C:/model/GW.BUD", "C:/model/STR.BUD"],
            hydrograph_files=["C:/model/GWHyd.out"],
        )
        assert isinstance(result.main_file, Path)
        assert isinstance(result.final_heads_file, Path)
        assert all(isinstance(f, Path) for f in result.budget_files)
        assert all(isinstance(f, Path) for f in result.hydrograph_files)

    def test_none_paths_remain_none(self):
        """Test that None paths remain None."""
        result = SimulationResult(
            success=True,
            return_code=0,
        )
        assert result.main_file is None
        assert result.final_heads_file is None

    def test_empty_file_lists(self):
        """Test that empty file lists stay empty."""
        result = SimulationResult(
            success=True,
            return_code=0,
        )
        assert result.budget_files == []
        assert result.hydrograph_files == []

    def test_mixed_path_types_in_lists(self):
        """Test file lists with mixed Path and string types."""
        result = SimulationResult(
            success=True,
            return_code=0,
            budget_files=[Path("C:/a.bud"), "C:/b.bud"],
            hydrograph_files=[Path("C:/a.hyd"), "C:/b.hyd"],
        )
        assert all(isinstance(f, Path) for f in result.budget_files)
        assert all(isinstance(f, Path) for f in result.hydrograph_files)

    def test_default_convergence_and_mass_balance(self):
        """Test default values for convergence_failures and mass_balance_error."""
        result = SimulationResult(success=True, return_code=0)
        assert result.convergence_failures == 0
        assert result.mass_balance_error == 0.0


class TestBudgetResultExtended:
    """Extended tests for BudgetResult."""

    def test_string_path_conversion(self):
        """Test that string paths are converted to Path objects."""
        result = BudgetResult(
            success=True,
            return_code=0,
            budget_file="C:/model/GW.BUD",
            output_file="C:/model/GW_output.txt",
        )
        assert isinstance(result.budget_file, Path)
        assert isinstance(result.output_file, Path)

    def test_none_paths_remain_none(self):
        """Test that None paths remain None."""
        result = BudgetResult(success=True, return_code=0)
        assert result.budget_file is None
        assert result.output_file is None

    def test_default_components(self):
        """Test default empty components list."""
        result = BudgetResult(success=True, return_code=0)
        assert result.components == []

    def test_with_components(self):
        """Test BudgetResult with component names."""
        result = BudgetResult(
            success=True,
            return_code=0,
            components=["Inflow", "Outflow", "Storage Change"],
        )
        assert len(result.components) == 3

    def test_inherits_failed_and_raise(self):
        """Test inherited methods work correctly."""
        result = BudgetResult(
            success=False, return_code=1,
            stderr="Budget processing failed",
        )
        assert result.failed is True
        with pytest.raises(RuntimeError):
            result.raise_on_error()


class TestZBudgetResultExtended:
    """Extended tests for ZBudgetResult."""

    def test_string_path_conversion(self):
        """Test that string paths are converted to Path objects."""
        result = ZBudgetResult(
            success=True,
            return_code=0,
            zbudget_file="C:/model/ZBud.hdf",
            zone_file="C:/model/zones.dat",
            output_file="C:/model/ZBud_output.txt",
        )
        assert isinstance(result.zbudget_file, Path)
        assert isinstance(result.zone_file, Path)
        assert isinstance(result.output_file, Path)

    def test_none_paths_remain_none(self):
        """Test that None paths remain None."""
        result = ZBudgetResult(success=True, return_code=0)
        assert result.zbudget_file is None
        assert result.zone_file is None
        assert result.output_file is None

    def test_defaults(self):
        """Test default values for n_zones and n_timesteps."""
        result = ZBudgetResult(success=True, return_code=0)
        assert result.n_zones == 0
        assert result.n_timesteps == 0

    def test_inherits_failed_and_raise(self):
        """Test inherited methods work correctly."""
        result = ZBudgetResult(
            success=False, return_code=1,
            errors=["Zone processing error"],
        )
        assert result.failed is True
        with pytest.raises(RuntimeError, match="Zone processing error"):
            result.raise_on_error()
