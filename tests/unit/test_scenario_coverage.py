"""Tests for runner/scenario.py edge cases.

Covers:
- Scenario.__post_init__() validation
- create_scenario_dir() ignore_outputs callable
- _apply_factor_to_timeseries() dict factor / exception paths
- run_scenarios() parallel error handling / progress callback
"""

from __future__ import annotations

import shutil
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from pyiwfm.runner.scenario import (
    Scenario,
    ScenarioManager,
    ScenarioResult,
)
from pyiwfm.runner.results import SimulationResult


class TestScenarioValidation:
    """Test Scenario name validation."""

    def test_invalid_chars_in_name(self) -> None:
        """Invalid characters in scenario name -> ValueError."""
        for char in '<>:"/\\|?*':
            with pytest.raises(ValueError, match="invalid character"):
                Scenario(name=f"test{char}name")

    def test_valid_name(self) -> None:
        """Valid scenario name."""
        s = Scenario(name="reduced_pumping_50pct")
        assert s.name == "reduced_pumping_50pct"

    def test_repr(self) -> None:
        """Test __repr__ output."""
        s = Scenario(name="test", modifications={"pumping": 0.5, "diversion": 0.0})
        r = repr(s)
        assert "test" in r
        assert "pumping" in r


class TestApplyFactorDict:
    """Test _apply_factor_to_timeseries with dict factor."""

    def test_apply_factor_dict(self, tmp_path: Path) -> None:
        """Dict-based factor modification."""
        # Create baseline dir with pumping file
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        main_file = baseline / "Simulation.in"
        main_file.write_text("Simulation main file\n")

        manager = ScenarioManager(baseline_dir=baseline)

        # Create test timeseries file
        ts_file = tmp_path / "test_ts.dat"
        ts_file.write_text(
            "C Test timeseries\n"
            "01/01/2020  100.0  200.0  300.0\n"
            "02/01/2020  110.0  210.0  310.0\n"
        )

        # Apply dict factor: column 1 *= 2, column 2 *= 0.5
        manager._apply_factor_to_timeseries(ts_file, {1: 2.0, 2: 0.5})

        content = ts_file.read_text()
        lines = content.splitlines()
        # Comment line preserved
        assert lines[0].startswith("C")
        # Data lines modified
        parts = lines[1].split()
        assert float(parts[1]) == pytest.approx(200.0)  # 100 * 2
        assert float(parts[2]) == pytest.approx(100.0)  # 200 * 0.5

    def test_apply_factor_single(self, tmp_path: Path) -> None:
        """Single float factor modification."""
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        main_file = baseline / "Simulation.in"
        main_file.write_text("main\n")

        manager = ScenarioManager(baseline_dir=baseline)

        ts_file = tmp_path / "test_ts.dat"
        ts_file.write_text(
            "01/01/2020  100.0  200.0\n"
        )

        manager._apply_factor_to_timeseries(ts_file, 0.5)

        content = ts_file.read_text()
        parts = content.strip().split()
        assert float(parts[1]) == pytest.approx(50.0)
        assert float(parts[2]) == pytest.approx(100.0)

    def test_apply_factor_nonexistent_file(self, tmp_path: Path) -> None:
        """Non-existent file -> silently skipped."""
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        (baseline / "Simulation.in").write_text("main\n")

        manager = ScenarioManager(baseline_dir=baseline)
        # Should not raise
        manager._apply_factor_to_timeseries(tmp_path / "nonexistent.dat", 0.5)


class TestRunScenariosParallel:
    """Test run_scenarios() parallel execution."""

    def test_run_scenarios_progress_callback(self, tmp_path: Path) -> None:
        """Progress callback fires during sequential execution."""
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        (baseline / "Simulation.in").write_text("main\n")

        manager = ScenarioManager(baseline_dir=baseline)
        callbacks = []

        def progress(name: str, completed: int, total: int) -> None:
            callbacks.append((name, completed, total))

        scenarios = [Scenario(name="s1"), Scenario(name="s2")]

        # Mock run_scenario to return a quick result
        mock_result = ScenarioResult(
            scenario=scenarios[0],
            result=SimulationResult(
                success=True,
                return_code=0,
            ),
            scenario_dir=tmp_path / "s1",
        )

        with patch.object(manager, "run_scenario", return_value=mock_result):
            manager.run_scenarios(scenarios, parallel=1, progress_callback=progress)

        assert len(callbacks) == 4  # 2 before + 2 after

    def test_run_scenarios_sequential(self, tmp_path: Path) -> None:
        """Sequential execution path."""
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        (baseline / "Simulation.in").write_text("main\n")

        manager = ScenarioManager(baseline_dir=baseline)
        scenarios = [Scenario(name="seq1")]

        mock_result = ScenarioResult(
            scenario=scenarios[0],
            result=SimulationResult(success=True, return_code=0),
            scenario_dir=tmp_path / "seq1",
        )

        with patch.object(manager, "run_scenario", return_value=mock_result):
            results = manager.run_scenarios(scenarios, parallel=1)

        assert "seq1" in results
        assert results["seq1"].success


class TestScenarioResultRepr:
    """Test ScenarioResult properties."""

    def test_success_property(self) -> None:
        """Test success property delegates to result.success."""
        sr = ScenarioResult(
            scenario=Scenario(name="test"),
            result=SimulationResult(success=True, return_code=0),
            scenario_dir=Path("/tmp/test"),
        )
        assert sr.success is True

    def test_repr_success(self) -> None:
        """Test repr for successful scenario."""
        sr = ScenarioResult(
            scenario=Scenario(name="test"),
            result=SimulationResult(success=True, return_code=0),
            scenario_dir=Path("/tmp/test"),
        )
        assert "success" in repr(sr)

    def test_repr_failed(self) -> None:
        """Test repr for failed scenario."""
        sr = ScenarioResult(
            scenario=Scenario(name="fail"),
            result=SimulationResult(success=False, return_code=1),
            scenario_dir=Path("/tmp/fail"),
        )
        assert "failed" in repr(sr)


class TestRunScenariosParallelExecution:
    """Test run_scenarios() with parallel > 1."""

    def test_run_scenarios_parallel(self, tmp_path: Path) -> None:
        """Parallel execution with 2 workers."""
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        (baseline / "Simulation.in").write_text("main\n")

        manager = ScenarioManager(baseline_dir=baseline)
        scenarios = [Scenario(name="p1"), Scenario(name="p2")]

        mock_result = ScenarioResult(
            scenario=scenarios[0],
            result=SimulationResult(success=True, return_code=0),
            scenario_dir=tmp_path / "p1",
        )

        with patch.object(manager, "_run_scenario_worker", return_value=mock_result):
            results = manager.run_scenarios(scenarios, parallel=2)

        assert len(results) == 2

    def test_run_scenarios_parallel_exception(self, tmp_path: Path) -> None:
        """Parallel worker exception -> failed result created."""
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        (baseline / "Simulation.in").write_text("main\n")

        manager = ScenarioManager(baseline_dir=baseline)
        scenarios = [Scenario(name="fail_p")]

        with patch.object(
            manager, "_run_scenario_worker",
            side_effect=Exception("Worker crashed"),
        ):
            results = manager.run_scenarios(scenarios, parallel=2)

        assert "fail_p" in results
        assert not results["fail_p"].success

    def test_run_scenarios_parallel_progress(self, tmp_path: Path) -> None:
        """Progress callback fires during parallel execution."""
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        (baseline / "Simulation.in").write_text("main\n")

        manager = ScenarioManager(baseline_dir=baseline)
        scenarios = [Scenario(name="pp1")]
        callbacks = []

        mock_result = ScenarioResult(
            scenario=scenarios[0],
            result=SimulationResult(success=True, return_code=0),
            scenario_dir=tmp_path / "pp1",
        )

        with patch.object(manager, "_run_scenario_worker", return_value=mock_result):
            manager.run_scenarios(
                scenarios,
                parallel=2,
                progress_callback=lambda n, c, t: callbacks.append((n, c, t)),
            )

        assert len(callbacks) >= 1


class TestScenarioManagerRepr:
    """Test ScenarioManager repr."""

    def test_repr(self, tmp_path: Path) -> None:
        """Repr includes baseline dir."""
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        (baseline / "Simulation.in").write_text("main\n")

        manager = ScenarioManager(baseline_dir=baseline)
        r = repr(manager)
        assert "ScenarioManager" in r
        assert "Simulation.in" in r


class TestCreateScenarioDir:
    """Test create_scenario_dir()."""

    def test_create_scenario_dir_basic(self, tmp_path: Path) -> None:
        """Create scenario directory from baseline."""
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        (baseline / "Simulation.in").write_text("main\n")
        (baseline / "data.dat").write_text("data\n")

        manager = ScenarioManager(baseline_dir=baseline)
        scenario = Scenario(name="test_dir")

        scenario_dir = manager.create_scenario_dir(scenario)
        assert scenario_dir.exists()
        assert (scenario_dir / "Simulation.in").exists()

    def test_create_scenario_dir_replaces_existing(self, tmp_path: Path) -> None:
        """Existing scenario dir replaced."""
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        (baseline / "Simulation.in").write_text("main\n")

        manager = ScenarioManager(baseline_dir=baseline)
        scenario = Scenario(name="replace_test")

        # Create first time
        dir1 = manager.create_scenario_dir(scenario)
        (dir1 / "extra.txt").write_text("extra\n")

        # Create again -> should replace
        dir2 = manager.create_scenario_dir(scenario)
        assert not (dir2 / "extra.txt").exists()

    def test_create_scenario_with_modifier_func(self, tmp_path: Path) -> None:
        """Custom modifier function applied."""
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        (baseline / "Simulation.in").write_text("main\n")

        modifications_applied = []

        def custom_mod(scenario_dir, baseline_dir):
            modifications_applied.append(True)

        manager = ScenarioManager(baseline_dir=baseline)
        scenario = Scenario(name="custom_mod", modifier_func=custom_mod)

        manager.create_scenario_dir(scenario)
        assert len(modifications_applied) == 1
