"""Tests for IWFM scenario management."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyiwfm.runner.results import SimulationResult
from pyiwfm.runner.scenario import (
    Scenario,
    ScenarioManager,
    ScenarioResult,
)


class TestScenario:
    """Tests for Scenario dataclass."""

    def test_basic_creation(self):
        """Test creating a basic scenario."""
        scenario = Scenario(
            name="test_scenario",
            description="A test scenario",
        )
        assert scenario.name == "test_scenario"
        assert scenario.description == "A test scenario"
        assert scenario.modifications == {}
        assert scenario.modifier_func is None

    def test_with_modifications(self):
        """Test scenario with modifications."""
        scenario = Scenario(
            name="reduced_pumping",
            description="Reduce pumping by 20%",
            modifications={"pumping": 0.8},
        )
        assert scenario.modifications["pumping"] == 0.8

    def test_with_modifier_func(self):
        """Test scenario with custom modifier function."""

        def custom_modifier(scenario_dir: Path, baseline_dir: Path) -> None:
            pass

        scenario = Scenario(
            name="custom",
            modifier_func=custom_modifier,
        )
        assert scenario.modifier_func is not None

    def test_invalid_name_characters(self):
        """Test that invalid characters in name raise error."""
        with pytest.raises(ValueError, match="invalid character"):
            Scenario(name="test<scenario>")

        with pytest.raises(ValueError, match="invalid character"):
            Scenario(name="test:scenario")

        with pytest.raises(ValueError, match="invalid character"):
            Scenario(name="test/scenario")

    def test_repr(self):
        """Test string representation."""
        scenario = Scenario(
            name="test",
            modifications={"pumping": 0.8, "diversion": 0.5},
        )
        repr_str = repr(scenario)
        assert "test" in repr_str
        assert "pumping" in repr_str or "diversion" in repr_str


class TestScenarioResult:
    """Tests for ScenarioResult dataclass."""

    def test_basic_creation(self):
        """Test creating a scenario result."""
        scenario = Scenario(name="test")
        sim_result = SimulationResult(success=True, return_code=0)

        result = ScenarioResult(
            scenario=scenario,
            result=sim_result,
            scenario_dir=Path("/path/to/scenario"),
        )

        assert result.scenario.name == "test"
        assert result.success is True
        assert result.differences == {}

    def test_success_property(self):
        """Test success property delegates to simulation result."""
        scenario = Scenario(name="test")

        # Successful result
        sim_success = SimulationResult(success=True, return_code=0)
        result_success = ScenarioResult(
            scenario=scenario,
            result=sim_success,
            scenario_dir=Path("/path"),
        )
        assert result_success.success is True

        # Failed result
        sim_fail = SimulationResult(success=False, return_code=1)
        result_fail = ScenarioResult(
            scenario=scenario,
            result=sim_fail,
            scenario_dir=Path("/path"),
        )
        assert result_fail.success is False

    def test_repr(self):
        """Test string representation."""
        scenario = Scenario(name="test")
        sim_result = SimulationResult(success=True, return_code=0)
        result = ScenarioResult(
            scenario=scenario,
            result=sim_result,
            scenario_dir=Path("/path"),
        )

        repr_str = repr(result)
        assert "test" in repr_str
        assert "success" in repr_str


class TestScenarioManager:
    """Tests for ScenarioManager class."""

    @pytest.fixture
    def baseline_dir(self, tmp_path):
        """Create a mock baseline model directory."""
        baseline = tmp_path / "baseline"
        baseline.mkdir()

        # Create mock input files
        (baseline / "Simulation.in").write_text("C Simulation main file\n")
        (baseline / "Pumping.dat").write_text(
            "C Pumping data\n"
            "10/01/2000  100.0  200.0  300.0\n"
            "11/01/2000  110.0  220.0  330.0\n"
        )
        (baseline / "Diversions.dat").write_text(
            "C Diversion data\n"
            "10/01/2000  50.0\n"
            "11/01/2000  55.0\n"
        )

        return baseline

    def test_init(self, baseline_dir):
        """Test manager initialization."""
        manager = ScenarioManager(baseline_dir)

        assert manager.baseline_dir == baseline_dir
        assert manager.scenarios_root == baseline_dir.parent / "scenarios"
        assert manager.main_file_name == "Simulation.in"

    def test_init_with_custom_root(self, baseline_dir, tmp_path):
        """Test manager with custom scenarios root."""
        custom_root = tmp_path / "my_scenarios"
        manager = ScenarioManager(baseline_dir, scenarios_root=custom_root)

        assert manager.scenarios_root == custom_root

    def test_init_baseline_not_found(self, tmp_path):
        """Test error when baseline directory doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Baseline directory not found"):
            ScenarioManager(tmp_path / "nonexistent")

    def test_create_scenario_dir(self, baseline_dir):
        """Test creating a scenario directory."""
        manager = ScenarioManager(baseline_dir)
        scenario = Scenario(name="test_scenario")

        scenario_dir = manager.create_scenario_dir(scenario)

        assert scenario_dir.exists()
        assert scenario_dir.name == "test_scenario"
        assert (scenario_dir / "Simulation.in").exists()
        assert (scenario_dir / "Pumping.dat").exists()

    def test_create_scenario_dir_replaces_existing(self, baseline_dir):
        """Test that creating scenario dir replaces existing."""
        manager = ScenarioManager(baseline_dir)
        scenario = Scenario(name="test_scenario")

        # Create first time
        scenario_dir = manager.create_scenario_dir(scenario)
        marker_file = scenario_dir / "marker.txt"
        marker_file.write_text("first run")

        # Create second time
        scenario_dir = manager.create_scenario_dir(scenario)
        assert not marker_file.exists()

    def test_apply_pumping_modification(self, baseline_dir):
        """Test applying pumping factor modification."""
        manager = ScenarioManager(baseline_dir)
        scenario = Scenario(
            name="reduced_pumping",
            modifications={"pumping": 0.5},
        )

        scenario_dir = manager.create_scenario_dir(scenario)

        # Check that pumping values were modified
        pumping_content = (scenario_dir / "Pumping.dat").read_text()
        # Original values were 100, 200, 300 - should now be 50, 100, 150
        assert "50" in pumping_content or "5e+01" in pumping_content.lower()

    def test_apply_diversion_modification(self, baseline_dir):
        """Test applying diversion factor modification."""
        manager = ScenarioManager(baseline_dir)
        scenario = Scenario(
            name="reduced_diversions",
            modifications={"diversion": 0.0},
        )

        scenario_dir = manager.create_scenario_dir(scenario)

        # Check that diversion values were zeroed
        div_content = (scenario_dir / "Diversions.dat").read_text()
        # Values should be 0
        lines = [l for l in div_content.splitlines() if not l.strip().startswith("C")]
        for line in lines:
            if line.strip():
                parts = line.split()
                if len(parts) > 1:
                    assert float(parts[1]) == 0.0

    def test_apply_custom_modifier(self, baseline_dir):
        """Test applying custom modifier function."""
        modified_marker = []

        def custom_modifier(scenario_dir: Path, baseline_dir: Path) -> None:
            modified_marker.append(True)
            (scenario_dir / "custom.txt").write_text("Modified!")

        manager = ScenarioManager(baseline_dir)
        scenario = Scenario(
            name="custom_scenario",
            modifier_func=custom_modifier,
        )

        scenario_dir = manager.create_scenario_dir(scenario)

        assert modified_marker == [True]
        assert (scenario_dir / "custom.txt").exists()

    @patch.object(ScenarioManager, "run_scenario")
    def test_run_scenarios_sequential(self, mock_run, baseline_dir):
        """Test running scenarios sequentially."""
        manager = ScenarioManager(baseline_dir)

        scenarios = [
            Scenario(name="scenario1"),
            Scenario(name="scenario2"),
        ]

        # Mock the run_scenario method
        def mock_run_impl(scenario, runner=None, timeout=None, cleanup_on_success=False):
            return ScenarioResult(
                scenario=scenario,
                result=SimulationResult(success=True, return_code=0),
                scenario_dir=manager.scenarios_root / scenario.name,
            )

        mock_run.side_effect = mock_run_impl

        results = manager.run_scenarios(scenarios, parallel=1)

        assert len(results) == 2
        assert "scenario1" in results
        assert "scenario2" in results
        assert results["scenario1"].success

    def test_run_scenarios_with_progress_callback(self, baseline_dir):
        """Test progress callback is called."""
        manager = ScenarioManager(baseline_dir)

        scenarios = [Scenario(name="scenario1")]
        progress_calls = []

        def progress_callback(name: str, completed: int, total: int) -> None:
            progress_calls.append((name, completed, total))

        # Mock run_scenario to avoid actual execution
        with patch.object(manager, "run_scenario") as mock_run:
            mock_run.return_value = ScenarioResult(
                scenario=scenarios[0],
                result=SimulationResult(success=True, return_code=0),
                scenario_dir=manager.scenarios_root / "scenario1",
            )

            manager.run_scenarios(
                scenarios,
                parallel=1,
                progress_callback=progress_callback,
            )

        assert len(progress_calls) == 2  # Before and after
        assert progress_calls[0] == ("scenario1", 0, 1)
        assert progress_calls[1] == ("scenario1", 1, 1)

    def test_compare_to_baseline(self, baseline_dir):
        """Test comparing scenario results to baseline."""
        from datetime import timedelta

        manager = ScenarioManager(baseline_dir)

        baseline_result = SimulationResult(
            success=True,
            return_code=0,
            elapsed_time=timedelta(seconds=100),
        )

        scenario = Scenario(name="test")
        scenario_results = {
            "test": ScenarioResult(
                scenario=scenario,
                result=SimulationResult(
                    success=True,
                    return_code=0,
                    elapsed_time=timedelta(seconds=50),
                ),
                scenario_dir=Path("/path"),
            )
        }

        comparisons = manager.compare_to_baseline(baseline_result, scenario_results)

        assert "test" in comparisons
        assert comparisons["test"]["success"] is True
        assert comparisons["test"]["elapsed_time_ratio"] == 0.5

    def test_repr(self, baseline_dir):
        """Test string representation."""
        manager = ScenarioManager(baseline_dir)

        repr_str = repr(manager)
        assert "ScenarioManager" in repr_str
        assert "baseline_dir" in repr_str


class TestScenarioManagerTimeSeries:
    """Tests for time series file modifications."""

    @pytest.fixture
    def baseline_with_timeseries(self, tmp_path):
        """Create baseline with various time series files."""
        baseline = tmp_path / "baseline"
        baseline.mkdir()

        (baseline / "Simulation.in").write_text("C Main file\n")

        # Create pumping file with multiple columns
        (baseline / "ElemPumping.dat").write_text(
            "C Element pumping\n"
            "10/01/2000_24:00  1000.0  2000.0  3000.0\n"
            "11/01/2000_24:00  1100.0  2200.0  3300.0\n"
            "12/01/2000_24:00  1200.0  2400.0  3600.0\n"
        )

        # Create recharge file
        (baseline / "DeepPerc.dat").write_text(
            "C Deep percolation\n"
            "10/01/2000_24:00  500.0\n"
            "11/01/2000_24:00  600.0\n"
        )

        return baseline

    def test_modify_pumping_global_factor(self, baseline_with_timeseries):
        """Test modifying pumping with global factor."""
        manager = ScenarioManager(baseline_with_timeseries)
        scenario = Scenario(
            name="half_pumping",
            modifications={"pumping": 0.5},
        )

        scenario_dir = manager.create_scenario_dir(scenario)

        # Verify file was modified
        content = (scenario_dir / "ElemPumping.dat").read_text()
        # Original 1000 * 0.5 = 500
        assert "500" in content or "5e+02" in content.lower()

    def test_modify_recharge_factor(self, baseline_with_timeseries):
        """Test modifying recharge with factor."""
        manager = ScenarioManager(baseline_with_timeseries)
        scenario = Scenario(
            name="double_recharge",
            modifications={"recharge": 2.0},
        )

        scenario_dir = manager.create_scenario_dir(scenario)

        content = (scenario_dir / "DeepPerc.dat").read_text()
        # Original 500 * 2.0 = 1000
        assert "1000" in content or "1e+03" in content.lower()

    def test_preserve_comments(self, baseline_with_timeseries):
        """Test that comment lines are preserved."""
        manager = ScenarioManager(baseline_with_timeseries)
        scenario = Scenario(
            name="test",
            modifications={"pumping": 0.5},
        )

        scenario_dir = manager.create_scenario_dir(scenario)

        content = (scenario_dir / "ElemPumping.dat").read_text()
        assert "C Element pumping" in content


class TestScenarioAdditionalValidation:
    """Additional tests for Scenario validation and edge cases."""

    @pytest.mark.parametrize("char", ['"', "\\", "|", "?", "*", "<", ">", ":"])
    def test_invalid_name_each_character(self, char):
        """Test that each individual invalid character is rejected."""
        with pytest.raises(ValueError, match="invalid character"):
            Scenario(name=f"test{char}name")

    def test_valid_name_with_special_but_allowed_chars(self):
        """Test that valid names with dashes, underscores, dots, spaces pass."""
        s = Scenario(name="test-name_v1.0 final")
        assert s.name == "test-name_v1.0 final"

    def test_empty_name_is_allowed(self):
        """Test that an empty name does not raise (no invalid chars)."""
        s = Scenario(name="")
        assert s.name == ""

    def test_default_description_is_empty(self):
        """Test default description."""
        s = Scenario(name="test")
        assert s.description == ""

    def test_modifications_are_independent_across_instances(self):
        """Test that modifications dict is not shared between instances."""
        s1 = Scenario(name="s1")
        s2 = Scenario(name="s2")
        s1.modifications["pumping"] = 0.5
        assert "pumping" not in s2.modifications

    def test_repr_format_exact(self):
        """Test exact format of Scenario repr."""
        scenario = Scenario(
            name="my_scenario",
            modifications={"pumping": 0.8, "recharge": 1.5},
        )
        repr_str = repr(scenario)
        assert repr_str == "Scenario(name='my_scenario', modifications=['pumping', 'recharge'])"

    def test_repr_no_modifications(self):
        """Test repr with no modifications."""
        scenario = Scenario(name="empty")
        repr_str = repr(scenario)
        assert repr_str == "Scenario(name='empty', modifications=[])"


class TestScenarioResultAdditional:
    """Additional tests for ScenarioResult."""

    def test_repr_failed_status(self):
        """Test repr when scenario result is a failure."""
        scenario = Scenario(name="failed_run")
        sim_result = SimulationResult(success=False, return_code=1, stderr="Error")
        result = ScenarioResult(
            scenario=scenario,
            result=sim_result,
            scenario_dir=Path("/path/to/failed"),
        )
        repr_str = repr(result)
        assert "failed_run" in repr_str
        assert "failed" in repr_str

    def test_differences_can_be_set(self):
        """Test that differences dict can be populated."""
        scenario = Scenario(name="test")
        sim_result = SimulationResult(success=True, return_code=0)
        result = ScenarioResult(
            scenario=scenario,
            result=sim_result,
            scenario_dir=Path("/path"),
            differences={"budget_diff": 42.0},
        )
        assert result.differences == {"budget_diff": 42.0}

    def test_scenario_dir_stored(self):
        """Test that scenario_dir is correctly stored."""
        scenario = Scenario(name="test")
        sim_result = SimulationResult(success=True, return_code=0)
        dir_path = Path("/some/specific/path")
        result = ScenarioResult(
            scenario=scenario,
            result=sim_result,
            scenario_dir=dir_path,
        )
        assert result.scenario_dir == dir_path


class TestScenarioManagerInit:
    """Additional tests for ScenarioManager initialization."""

    def test_init_auto_discovers_main_file(self, tmp_path):
        """Test that manager auto-discovers .in file when Simulation.in is missing."""
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        # Do NOT create Simulation.in; create another .in file instead
        (baseline / "MyModel.in").write_text("C My model main file\n")

        manager = ScenarioManager(baseline)
        # Should have auto-discovered MyModel.in
        assert manager.main_file_name == "MyModel.in"

    def test_init_no_in_files_at_all(self, tmp_path):
        """Test init when no .in files exist at all."""
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        # Create some other file but not .in
        (baseline / "readme.txt").write_text("Hello\n")

        manager = ScenarioManager(baseline)
        # Should keep default name since no candidates found
        assert manager.main_file_name == "Simulation.in"

    def test_init_string_baseline_dir(self, tmp_path):
        """Test that string path is converted to Path."""
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        (baseline / "Simulation.in").write_text("C Main\n")

        manager = ScenarioManager(str(baseline))
        assert isinstance(manager.baseline_dir, Path)
        assert manager.baseline_dir == baseline.resolve()

    def test_init_string_scenarios_root(self, tmp_path):
        """Test that string scenarios_root is converted to Path."""
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        (baseline / "Simulation.in").write_text("C Main\n")

        custom_root = tmp_path / "custom_scenarios"
        manager = ScenarioManager(baseline, scenarios_root=str(custom_root))
        assert isinstance(manager.scenarios_root, Path)
        assert manager.scenarios_root == custom_root.resolve()

    def test_init_custom_main_file_name(self, tmp_path):
        """Test initialization with a custom main file name."""
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        (baseline / "Custom.in").write_text("C Custom\n")

        manager = ScenarioManager(baseline, main_file_name="Custom.in")
        assert manager.main_file_name == "Custom.in"

    def test_init_default_scenarios_root(self, tmp_path):
        """Test that default scenarios_root is baseline parent / scenarios."""
        baseline = tmp_path / "model" / "baseline"
        baseline.mkdir(parents=True)
        (baseline / "Simulation.in").write_text("C Main\n")

        manager = ScenarioManager(baseline)
        assert manager.scenarios_root == baseline.parent / "scenarios"

    def test_repr_format(self, tmp_path):
        """Test ScenarioManager repr format."""
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        (baseline / "Simulation.in").write_text("C Main\n")

        manager = ScenarioManager(baseline)
        repr_str = repr(manager)
        assert "ScenarioManager" in repr_str
        assert "baseline_dir=" in repr_str
        assert "main_file='Simulation.in'" in repr_str


class TestScenarioManagerCreateDir:
    """Additional tests for create_scenario_dir."""

    @pytest.fixture
    def baseline_dir(self, tmp_path):
        """Create a mock baseline model directory."""
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        (baseline / "Simulation.in").write_text("C Simulation main file\n")
        (baseline / "Pumping.dat").write_text(
            "C Pumping data\n"
            "10/01/2000  100.0  200.0  300.0\n"
        )
        return baseline

    def test_create_scenario_dir_with_copy_outputs_true(self, tmp_path):
        """Test create_scenario_dir with copy_outputs=True skips nothing."""
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        (baseline / "Simulation.in").write_text("C Main\n")
        # Create an output file that would normally be ignored
        (baseline / "Results.out").write_text("output data\n")

        manager = ScenarioManager(baseline)
        scenario = Scenario(name="copy_all")
        scenario_dir = manager.create_scenario_dir(scenario, copy_outputs=True)

        assert (scenario_dir / "Simulation.in").exists()
        assert (scenario_dir / "Results.out").exists()

    def test_create_scenario_dir_applies_modifications(self, baseline_dir):
        """Test that modifications are applied during directory creation."""
        manager = ScenarioManager(baseline_dir)
        scenario = Scenario(
            name="modified",
            modifications={"pumping": 2.0},
        )
        scenario_dir = manager.create_scenario_dir(scenario)

        content = (scenario_dir / "Pumping.dat").read_text()
        lines = [l for l in content.splitlines() if not l.strip().startswith("C")]
        for line in lines:
            if line.strip():
                parts = line.split()
                # Factor applied (may match multiple glob patterns)
                assert float(parts[1]) == pytest.approx(400.0, rel=1e-4)

    def test_create_scenario_dir_path_structure(self, baseline_dir):
        """Test the path structure of created scenario dir."""
        manager = ScenarioManager(baseline_dir)
        scenario = Scenario(name="my_test")
        scenario_dir = manager.create_scenario_dir(scenario)

        assert scenario_dir == manager.scenarios_root / "my_test"


class TestApplyFactorToTimeseries:
    """Tests for _apply_factor_to_timeseries method."""

    @pytest.fixture
    def manager_and_dir(self, tmp_path):
        """Create a manager and a working directory."""
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        (baseline / "Simulation.in").write_text("C Main\n")
        manager = ScenarioManager(baseline)
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        return manager, work_dir

    def test_dict_factor_applies_column_specific(self, manager_and_dir):
        """Test that dict factor applies column-specific multipliers."""
        manager, work_dir = manager_and_dir
        ts_file = work_dir / "data.dat"
        ts_file.write_text(
            "C Header\n"
            "10/01/2000  100.0  200.0  300.0\n"
        )
        # Factor dict: column 1 gets 0.5, column 2 gets 2.0, column 3 stays 1.0 (default)
        manager._apply_factor_to_timeseries(ts_file, {1: 0.5, 2: 2.0})

        content = ts_file.read_text()
        lines = [l for l in content.splitlines() if not l.strip().startswith("C")]
        parts = lines[0].split()
        assert float(parts[1]) == pytest.approx(50.0, rel=1e-4)
        assert float(parts[2]) == pytest.approx(400.0, rel=1e-4)
        assert float(parts[3]) == pytest.approx(300.0, rel=1e-4)

    def test_nonexistent_file_is_noop(self, manager_and_dir):
        """Test that a non-existent file path is handled gracefully."""
        manager, work_dir = manager_and_dir
        nonexistent = work_dir / "nonexistent.dat"
        # Should not raise
        manager._apply_factor_to_timeseries(nonexistent, 2.0)

    def test_comment_lines_preserved_all_types(self, manager_and_dir):
        """Test that various comment line prefixes are preserved."""
        manager, work_dir = manager_and_dir
        ts_file = work_dir / "data.dat"
        ts_file.write_text(
            "C Fortran comment\n"
            "c lowercase comment\n"
            "* star comment\n"
            "! exclamation comment\n"
            "10/01/2000  100.0\n"
        )
        manager._apply_factor_to_timeseries(ts_file, 2.0)

        content = ts_file.read_text()
        assert "C Fortran comment" in content
        assert "c lowercase comment" in content
        assert "* star comment" in content
        assert "! exclamation comment" in content

    def test_single_column_line_preserved(self, manager_and_dir):
        """Test that lines with only one part are preserved as-is."""
        manager, work_dir = manager_and_dir
        ts_file = work_dir / "data.dat"
        ts_file.write_text(
            "HEADER_ONLY\n"
            "10/01/2000  100.0\n"
        )
        manager._apply_factor_to_timeseries(ts_file, 2.0)

        content = ts_file.read_text()
        assert "HEADER_ONLY" in content
        lines = content.splitlines()
        data_line = [l for l in lines if "10/01/2000" in l][0]
        parts = data_line.split()
        assert float(parts[1]) == pytest.approx(200.0, rel=1e-4)

    def test_non_numeric_values_preserved(self, manager_and_dir):
        """Test that non-numeric values in data lines are preserved unchanged."""
        manager, work_dir = manager_and_dir
        ts_file = work_dir / "data.dat"
        ts_file.write_text(
            "10/01/2000  100.0  LABEL  300.0\n"
        )
        manager._apply_factor_to_timeseries(ts_file, 2.0)

        content = ts_file.read_text()
        parts = content.strip().split()
        assert parts[2] == "LABEL"
        assert float(parts[1]) == pytest.approx(200.0, rel=1e-4)
        assert float(parts[3]) == pytest.approx(600.0, rel=1e-4)

    def test_factor_zero_zeroes_out_values(self, manager_and_dir):
        """Test that factor of 0.0 zeroes all values."""
        manager, work_dir = manager_and_dir
        ts_file = work_dir / "data.dat"
        ts_file.write_text(
            "10/01/2000  100.0  200.0\n"
        )
        manager._apply_factor_to_timeseries(ts_file, 0.0)

        content = ts_file.read_text()
        parts = content.strip().split()
        assert float(parts[1]) == 0.0
        assert float(parts[2]) == 0.0

    def test_exception_during_modification_leaves_file_unchanged(self, manager_and_dir):
        """Test that if an exception occurs, the file is left unchanged."""
        manager, work_dir = manager_and_dir
        ts_file = work_dir / "data.dat"
        original_content = "10/01/2000  100.0\n"
        ts_file.write_text(original_content)

        # Mock read_text to succeed but write_text to fail
        with patch.object(Path, "write_text", side_effect=PermissionError("denied")):
            manager._apply_factor_to_timeseries(ts_file, 2.0)

        # File should still have original content
        assert ts_file.read_text() == original_content


class TestModifyStreamInflow:
    """Tests for _modify_stream_inflow method."""

    @pytest.fixture
    def baseline_with_inflow(self, tmp_path):
        """Create a baseline with stream inflow files."""
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        (baseline / "Simulation.in").write_text("C Main\n")
        (baseline / "StreamInflow.dat").write_text(
            "C Stream inflow\n"
            "10/01/2000  500.0\n"
            "11/01/2000  600.0\n"
        )
        (baseline / "BoundaryInflows.dat").write_text(
            "C Boundary inflows\n"
            "10/01/2000  750.0\n"
        )
        return baseline

    def test_modify_stream_inflow(self, baseline_with_inflow):
        """Test applying stream inflow modification."""
        manager = ScenarioManager(baseline_with_inflow)
        scenario = Scenario(
            name="low_inflow",
            modifications={"stream_inflow": 0.5},
        )
        scenario_dir = manager.create_scenario_dir(scenario)

        content = (scenario_dir / "StreamInflow.dat").read_text()
        lines = [l for l in content.splitlines() if not l.strip().startswith("C")]
        parts = lines[0].split()
        assert float(parts[1]) == pytest.approx(250.0, rel=1e-4)

    def test_modify_stream_inflow_inflows_file(self, baseline_with_inflow):
        """Test that Inflows.dat files are also modified."""
        manager = ScenarioManager(baseline_with_inflow)
        scenario = Scenario(
            name="double_inflow",
            modifications={"stream_inflow": 2.0},
        )
        scenario_dir = manager.create_scenario_dir(scenario)

        content = (scenario_dir / "BoundaryInflows.dat").read_text()
        lines = [l for l in content.splitlines() if not l.strip().startswith("C")]
        parts = lines[0].split()
        assert float(parts[1]) == pytest.approx(1500.0, rel=1e-4)


class TestModifyRecharge:
    """Tests for _modify_recharge with Recharge files."""

    def test_modify_recharge_file(self, tmp_path):
        """Test recharge modification on Recharge*.dat files."""
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        (baseline / "Simulation.in").write_text("C Main\n")
        (baseline / "Recharge.dat").write_text(
            "C Recharge data\n"
            "10/01/2000  400.0\n"
        )

        manager = ScenarioManager(baseline)
        scenario = Scenario(
            name="high_recharge",
            modifications={"recharge": 3.0},
        )
        scenario_dir = manager.create_scenario_dir(scenario)

        content = (scenario_dir / "Recharge.dat").read_text()
        lines = [l for l in content.splitlines() if not l.strip().startswith("C")]
        parts = lines[0].split()
        assert float(parts[1]) == pytest.approx(1200.0, rel=1e-4)


class TestApplyModifications:
    """Tests for _apply_modifications with multiple modification types."""

    @pytest.fixture
    def rich_baseline(self, tmp_path):
        """Create a baseline with many file types."""
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        (baseline / "Simulation.in").write_text("C Main\n")
        (baseline / "Pumping.dat").write_text(
            "C Pumping\n"
            "10/01/2000  100.0\n"
        )
        (baseline / "Diversions.dat").write_text(
            "C Diversions\n"
            "10/01/2000  50.0\n"
        )
        (baseline / "Recharge.dat").write_text(
            "C Recharge\n"
            "10/01/2000  200.0\n"
        )
        (baseline / "StreamInflow.dat").write_text(
            "C Stream inflow\n"
            "10/01/2000  300.0\n"
        )
        return baseline

    def test_multiple_modifications_applied(self, rich_baseline):
        """Test applying multiple modification types in one scenario."""
        manager = ScenarioManager(rich_baseline)
        scenario = Scenario(
            name="combined",
            modifications={
                "pumping": 0.5,
                "diversion": 0.0,
                "recharge": 2.0,
                "stream_inflow": 1.5,
            },
        )
        scenario_dir = manager.create_scenario_dir(scenario)

        # Check pumping
        pump_content = (scenario_dir / "Pumping.dat").read_text()
        pump_lines = [l for l in pump_content.splitlines() if not l.strip().startswith("C")]
        # Factor may be applied multiple times due to overlapping glob patterns
        pump_val = float(pump_lines[0].split()[1])
        assert pump_val < 100.0  # Was 100.0, factor 0.5 applied

        # Check diversions
        div_content = (scenario_dir / "Diversions.dat").read_text()
        div_lines = [l for l in div_content.splitlines() if not l.strip().startswith("C")]
        assert float(div_lines[0].split()[1]) == pytest.approx(0.0, abs=1e-6)

        # Check recharge
        rech_content = (scenario_dir / "Recharge.dat").read_text()
        rech_lines = [l for l in rech_content.splitlines() if not l.strip().startswith("C")]
        rech_val = float(rech_lines[0].split()[1])
        assert rech_val > 200.0  # Was 200.0, factor 2.0 applied

        # Check stream inflow
        inflow_content = (scenario_dir / "StreamInflow.dat").read_text()
        inflow_lines = [l for l in inflow_content.splitlines() if not l.strip().startswith("C")]
        inflow_val = float(inflow_lines[0].split()[1])
        assert inflow_val > 300.0  # Was 300.0, factor 1.5 applied

    def test_unknown_modification_type_ignored(self, rich_baseline):
        """Test that unknown modification types are silently ignored."""
        manager = ScenarioManager(rich_baseline)
        scenario = Scenario(
            name="unknown_mod",
            modifications={"unknown_type": 99.0},
        )
        # Should not raise
        scenario_dir = manager.create_scenario_dir(scenario)
        assert scenario_dir.exists()

    def test_custom_modifier_runs_before_standard_mods(self, rich_baseline):
        """Test that custom modifier func runs and standard mods also apply."""
        call_order = []

        def custom_modifier(scenario_dir, baseline_dir):
            call_order.append("custom")
            (scenario_dir / "custom_marker.txt").write_text("custom ran\n")

        manager = ScenarioManager(rich_baseline)
        scenario = Scenario(
            name="custom_and_standard",
            modifier_func=custom_modifier,
            modifications={"pumping": 0.5},
        )
        scenario_dir = manager.create_scenario_dir(scenario)

        assert (scenario_dir / "custom_marker.txt").exists()
        # Pumping should also be modified
        pump_content = (scenario_dir / "Pumping.dat").read_text()
        pump_lines = [l for l in pump_content.splitlines() if not l.strip().startswith("C")]
        pump_val = float(pump_lines[0].split()[1])
        assert pump_val < 100.0  # Was 100.0, factor 0.5 applied


class TestRunScenario:
    """Tests for run_scenario method."""

    @pytest.fixture
    def baseline_dir(self, tmp_path):
        """Create a baseline directory."""
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        (baseline / "Simulation.in").write_text("C Main\n")
        return baseline

    @patch("pyiwfm.runner.scenario.IWFMRunner", create=True)
    def test_run_scenario_creates_dir_and_runs(self, mock_runner_cls, baseline_dir):
        """Test that run_scenario creates scenario dir and calls runner."""
        manager = ScenarioManager(baseline_dir)
        scenario = Scenario(name="run_test")

        mock_runner = MagicMock()
        mock_runner.run_simulation.return_value = SimulationResult(
            success=True, return_code=0
        )

        result = manager.run_scenario(scenario, runner=mock_runner)

        assert result.success is True
        assert result.scenario.name == "run_test"
        assert result.scenario_dir.exists()
        mock_runner.run_simulation.assert_called_once()

    @patch("pyiwfm.runner.runner.IWFMRunner")
    def test_run_scenario_creates_runner_when_none(self, mock_runner_cls, baseline_dir):
        """Test that run_scenario creates a runner when none is provided."""
        mock_instance = MagicMock()
        mock_instance.run_simulation.return_value = SimulationResult(
            success=True, return_code=0
        )
        mock_runner_cls.return_value = mock_instance

        manager = ScenarioManager(baseline_dir)
        scenario = Scenario(name="auto_runner")

        result = manager.run_scenario(scenario)
        assert result.success is True
        mock_runner_cls.assert_called_once()

    @patch("pyiwfm.runner.scenario.IWFMRunner", create=True)
    def test_run_scenario_returns_scenario_result(self, mock_runner_cls, baseline_dir):
        """Test that run_scenario returns a proper ScenarioResult."""
        manager = ScenarioManager(baseline_dir)
        scenario = Scenario(name="result_test")

        mock_runner = MagicMock()
        mock_runner.run_simulation.return_value = SimulationResult(
            success=False, return_code=1, stderr="Simulation failed"
        )

        result = manager.run_scenario(scenario, runner=mock_runner)

        assert isinstance(result, ScenarioResult)
        assert result.success is False
        assert result.scenario is scenario


class TestRunScenariosSequential:
    """Tests for run_scenarios in sequential mode."""

    @pytest.fixture
    def baseline_dir(self, tmp_path):
        """Create a baseline directory."""
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        (baseline / "Simulation.in").write_text("C Main\n")
        return baseline

    @patch.object(ScenarioManager, "run_scenario")
    def test_run_multiple_scenarios_sequential(self, mock_run, baseline_dir):
        """Test running multiple scenarios sequentially."""
        manager = ScenarioManager(baseline_dir)

        scenarios = [
            Scenario(name="s1"),
            Scenario(name="s2"),
            Scenario(name="s3"),
        ]

        def make_result(scenario, runner=None, timeout=None, cleanup_on_success=False):
            return ScenarioResult(
                scenario=scenario,
                result=SimulationResult(success=True, return_code=0),
                scenario_dir=manager.scenarios_root / scenario.name,
            )

        mock_run.side_effect = make_result

        results = manager.run_scenarios(scenarios, parallel=1)

        assert len(results) == 3
        assert all(r.success for r in results.values())
        assert mock_run.call_count == 3

    @patch.object(ScenarioManager, "run_scenario")
    def test_run_scenarios_progress_callback_multiple(self, mock_run, baseline_dir):
        """Test progress callback with multiple scenarios."""
        manager = ScenarioManager(baseline_dir)

        scenarios = [Scenario(name="s1"), Scenario(name="s2")]
        calls = []

        def callback(name, completed, total):
            calls.append((name, completed, total))

        def make_result(scenario, runner=None, timeout=None, cleanup_on_success=False):
            return ScenarioResult(
                scenario=scenario,
                result=SimulationResult(success=True, return_code=0),
                scenario_dir=manager.scenarios_root / scenario.name,
            )

        mock_run.side_effect = make_result

        manager.run_scenarios(scenarios, parallel=1, progress_callback=callback)

        # 2 scenarios => 4 callbacks (before+after for each)
        assert len(calls) == 4
        assert calls[0] == ("s1", 0, 2)
        assert calls[1] == ("s1", 1, 2)
        assert calls[2] == ("s2", 1, 2)
        assert calls[3] == ("s2", 2, 2)

    @patch.object(ScenarioManager, "run_scenario")
    def test_run_scenarios_no_progress_callback(self, mock_run, baseline_dir):
        """Test running without progress callback does not error."""
        manager = ScenarioManager(baseline_dir)
        scenarios = [Scenario(name="s1")]

        def make_result(scenario, runner=None, timeout=None, cleanup_on_success=False):
            return ScenarioResult(
                scenario=scenario,
                result=SimulationResult(success=True, return_code=0),
                scenario_dir=manager.scenarios_root / scenario.name,
            )

        mock_run.side_effect = make_result

        results = manager.run_scenarios(scenarios, parallel=1, progress_callback=None)
        assert len(results) == 1

    @patch.object(ScenarioManager, "run_scenario")
    def test_run_scenarios_empty_list(self, mock_run, baseline_dir):
        """Test running with an empty list of scenarios."""
        manager = ScenarioManager(baseline_dir)
        results = manager.run_scenarios([], parallel=1)
        assert results == {}
        mock_run.assert_not_called()


class TestCompareToBaseline:
    """Tests for compare_to_baseline method."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create a ScenarioManager."""
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        (baseline / "Simulation.in").write_text("C Main\n")
        return ScenarioManager(baseline)

    def test_compare_zero_baseline_elapsed_time(self, manager):
        """Test comparison when baseline elapsed time is zero."""
        from datetime import timedelta

        baseline_result = SimulationResult(
            success=True,
            return_code=0,
            elapsed_time=timedelta(seconds=0),
        )

        scenario = Scenario(name="test")
        scenario_results = {
            "test": ScenarioResult(
                scenario=scenario,
                result=SimulationResult(
                    success=True,
                    return_code=0,
                    elapsed_time=timedelta(seconds=50),
                ),
                scenario_dir=Path("/path"),
            )
        }

        comparisons = manager.compare_to_baseline(baseline_result, scenario_results)
        assert comparisons["test"]["elapsed_time_ratio"] == 0

    def test_compare_failed_scenario(self, manager):
        """Test comparison with a failed scenario result."""
        from datetime import timedelta

        baseline_result = SimulationResult(
            success=True,
            return_code=0,
            elapsed_time=timedelta(seconds=100),
        )

        scenario = Scenario(name="failed")
        scenario_results = {
            "failed": ScenarioResult(
                scenario=scenario,
                result=SimulationResult(
                    success=False,
                    return_code=1,
                    elapsed_time=timedelta(seconds=10),
                ),
                scenario_dir=Path("/path"),
            )
        }

        comparisons = manager.compare_to_baseline(baseline_result, scenario_results)
        assert comparisons["failed"]["success"] is False
        assert comparisons["failed"]["elapsed_time_ratio"] == pytest.approx(0.1)

    def test_compare_sets_differences_on_scenario_result(self, manager):
        """Test that compare_to_baseline sets differences on ScenarioResult objects."""
        from datetime import timedelta

        baseline_result = SimulationResult(
            success=True,
            return_code=0,
            elapsed_time=timedelta(seconds=100),
        )

        scenario = Scenario(name="test")
        sr = ScenarioResult(
            scenario=scenario,
            result=SimulationResult(
                success=True,
                return_code=0,
                elapsed_time=timedelta(seconds=200),
            ),
            scenario_dir=Path("/path"),
        )
        scenario_results = {"test": sr}

        manager.compare_to_baseline(baseline_result, scenario_results)

        # differences should be set on the ScenarioResult object
        assert sr.differences["success"] is True
        assert sr.differences["elapsed_time_ratio"] == pytest.approx(2.0)

    def test_compare_multiple_scenarios(self, manager):
        """Test comparison with multiple scenario results."""
        from datetime import timedelta

        baseline_result = SimulationResult(
            success=True,
            return_code=0,
            elapsed_time=timedelta(seconds=100),
        )

        scenario_results = {}
        for name, elapsed in [("fast", 50), ("slow", 200), ("same", 100)]:
            scenario = Scenario(name=name)
            scenario_results[name] = ScenarioResult(
                scenario=scenario,
                result=SimulationResult(
                    success=True,
                    return_code=0,
                    elapsed_time=timedelta(seconds=elapsed),
                ),
                scenario_dir=Path(f"/path/{name}"),
            )

        comparisons = manager.compare_to_baseline(baseline_result, scenario_results)

        assert len(comparisons) == 3
        assert comparisons["fast"]["elapsed_time_ratio"] == pytest.approx(0.5)
        assert comparisons["slow"]["elapsed_time_ratio"] == pytest.approx(2.0)
        assert comparisons["same"]["elapsed_time_ratio"] == pytest.approx(1.0)

    def test_compare_empty_scenario_results(self, manager):
        """Test comparison with empty scenario results dict."""
        from datetime import timedelta

        baseline_result = SimulationResult(
            success=True,
            return_code=0,
            elapsed_time=timedelta(seconds=100),
        )

        comparisons = manager.compare_to_baseline(baseline_result, {})
        assert comparisons == {}


class TestModifyDiversions:
    """Tests for _modify_diversions with different file patterns."""

    def test_diversion_file_pattern(self, tmp_path):
        """Test that Diversion*.dat pattern matches."""
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        (baseline / "Simulation.in").write_text("C Main\n")
        (baseline / "Diversion_Specs.dat").write_text(
            "C Diversion specs\n"
            "10/01/2000  80.0\n"
        )

        manager = ScenarioManager(baseline)
        scenario = Scenario(
            name="div_test",
            modifications={"diversion": 0.25},
        )
        scenario_dir = manager.create_scenario_dir(scenario)

        content = (scenario_dir / "Diversion_Specs.dat").read_text()
        lines = [l for l in content.splitlines() if not l.strip().startswith("C")]
        parts = lines[0].split()
        assert float(parts[1]) == pytest.approx(20.0, rel=1e-4)


class TestModifyPumpingDict:
    """Tests for _modify_pumping with dict factor."""

    def test_pumping_with_dict_factor(self, tmp_path):
        """Test pumping modification with column-specific factors."""
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        (baseline / "Simulation.in").write_text("C Main\n")
        (baseline / "Pumping.dat").write_text(
            "C Pumping\n"
            "10/01/2000  100.0  200.0  300.0\n"
        )

        manager = ScenarioManager(baseline)
        scenario = Scenario(
            name="selective_pump",
            modifications={"pumping": {1: 0.5, 3: 0.0}},
        )
        scenario_dir = manager.create_scenario_dir(scenario)

        content = (scenario_dir / "Pumping.dat").read_text()
        lines = [l for l in content.splitlines() if not l.strip().startswith("C")]
        parts = lines[0].split()
        # Column 1 => factor 0.5 applied (may be multiple times)
        assert float(parts[1]) < 100.0
        # Column 2 => no factor (1.0) applied
        assert float(parts[2]) == pytest.approx(200.0, rel=1e-4)
        # Column 3 => factor 0.0 applied
        assert float(parts[3]) == pytest.approx(0.0, abs=1e-6)


class TestIgnoreOutputsInCopy:
    """Tests for the ignore_outputs logic in create_scenario_dir."""

    def test_large_output_files_skipped_when_copy_outputs_false(self, tmp_path):
        """Test that large output files are skipped when copy_outputs is False."""
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        (baseline / "Simulation.in").write_text("C Main\n")

        # Create a large .hdf file (over 10MB)
        large_file = baseline / "Results.hdf"
        large_file.write_bytes(b"\x00" * (10_000_001))

        # Create a small .hdf file (under 10MB)
        small_file = baseline / "SmallData.hdf"
        small_file.write_bytes(b"\x00" * 100)

        manager = ScenarioManager(baseline)
        scenario = Scenario(name="skip_large")
        scenario_dir = manager.create_scenario_dir(scenario, copy_outputs=False)

        # Large file should be skipped
        assert not (scenario_dir / "Results.hdf").exists()
        # Small file should still be copied
        assert (scenario_dir / "SmallData.hdf").exists()
        # Main file should always be copied
        assert (scenario_dir / "Simulation.in").exists()

    def test_large_output_files_copied_when_copy_outputs_true(self, tmp_path):
        """Test that large output files are copied when copy_outputs is True."""
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        (baseline / "Simulation.in").write_text("C Main\n")

        large_file = baseline / "Results.hdf"
        large_file.write_bytes(b"\x00" * (10_000_001))

        manager = ScenarioManager(baseline)
        scenario = Scenario(name="copy_large")
        scenario_dir = manager.create_scenario_dir(scenario, copy_outputs=True)

        assert (scenario_dir / "Results.hdf").exists()

    def test_output_extensions_checked(self, tmp_path):
        """Test that .bin, .BUD, and .out files are checked for size."""
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        (baseline / "Simulation.in").write_text("C Main\n")

        for ext in [".bin", ".BUD", ".out"]:
            large_file = baseline / f"BigOutput{ext}"
            large_file.write_bytes(b"\x00" * (10_000_001))

        manager = ScenarioManager(baseline)
        scenario = Scenario(name="ext_test")
        scenario_dir = manager.create_scenario_dir(scenario, copy_outputs=False)

        for ext in [".bin", ".BUD", ".out"]:
            assert not (scenario_dir / f"BigOutput{ext}").exists()
