"""Supplementary tests for simulation.py targeting uncovered branches.

Covers:
- Unsaturated zone and small watershed component file paths
- _parse_datetime with space separator
- _write_output_settings without output_dir
- SimulationReader with additional config line variants
- Writer with all optional components
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from pyiwfm.core.exceptions import FileFormatError
from pyiwfm.core.timeseries import TimeUnit
from pyiwfm.io.simulation import (
    SimulationConfig,
    SimulationFileConfig,
    SimulationReader,
    SimulationWriter,
    read_simulation,
    write_simulation,
)

# =============================================================================
# Additional Component File Tests
# =============================================================================


class TestWriterAdditionalComponents:
    """Tests for unsaturated zone and small watershed component file writing."""

    def test_write_unsaturated_zone_file(self, tmp_path: Path) -> None:
        """Test writing unsaturated zone component file path."""
        file_config = SimulationFileConfig(output_dir=tmp_path)
        writer = SimulationWriter(file_config)
        config = SimulationConfig(
            unsaturated_zone_file=Path("unsatzone.in"),
        )

        filepath = writer.write(config)

        content = filepath.read_text()
        assert "unsatzone.in" in content
        assert "UNSATURATED_ZONE_FILE" in content

    def test_write_small_watershed_file(self, tmp_path: Path) -> None:
        """Test writing small watershed component file path."""
        file_config = SimulationFileConfig(output_dir=tmp_path)
        writer = SimulationWriter(file_config)
        config = SimulationConfig(
            small_watershed_file=Path("watershed.in"),
        )

        filepath = writer.write(config)

        content = filepath.read_text()
        assert "watershed.in" in content
        assert "SMALL_WATERSHED_FILE" in content

    def test_write_all_component_files(self, tmp_path: Path) -> None:
        """Test writing all component file paths at once."""
        file_config = SimulationFileConfig(output_dir=tmp_path)
        writer = SimulationWriter(file_config)
        config = SimulationConfig(
            preprocessor_file=Path("pre.in"),
            groundwater_file=Path("gw.in"),
            streams_file=Path("str.in"),
            lakes_file=Path("lak.in"),
            rootzone_file=Path("rz.in"),
            unsaturated_zone_file=Path("uz.in"),
            small_watershed_file=Path("sw.in"),
        )

        filepath = writer.write(config)

        content = filepath.read_text()
        assert "pre.in" in content
        assert "gw.in" in content
        assert "str.in" in content
        assert "lak.in" in content
        assert "rz.in" in content
        assert "uz.in" in content
        assert "sw.in" in content

    def test_write_no_component_files(self, tmp_path: Path) -> None:
        """Test writing with no component files set (all None)."""
        file_config = SimulationFileConfig(output_dir=tmp_path)
        writer = SimulationWriter(file_config)
        config = SimulationConfig()

        filepath = writer.write(config)

        content = filepath.read_text()
        assert "COMPONENT INPUT FILES" in content
        # No file paths should be written
        assert "PREPROCESSOR_FILE" not in content

    def test_write_no_output_dir(self, tmp_path: Path) -> None:
        """Test writing with output_dir=None."""
        file_config = SimulationFileConfig(output_dir=tmp_path)
        writer = SimulationWriter(file_config)
        config = SimulationConfig(output_dir=None)

        filepath = writer.write(config)

        content = filepath.read_text()
        assert "OUTPUT CONTROL" in content
        assert "OUTPUT_INTERVAL" in content
        # OUTPUT_DIR line should not be present
        assert "OUTPUT_DIR" not in content


# =============================================================================
# Reader Additional Branches
# =============================================================================


class TestReaderAdditionalBranches:
    """Tests for SimulationReader config line parsing variants."""

    def test_parse_datetime_with_space_separator(self) -> None:
        """Test _parse_datetime directly with space separator."""
        reader = SimulationReader()

        result = reader._parse_datetime("01/15/2020 12:30")

        assert result == datetime(2020, 1, 15, 12, 30, 0)

    def test_parse_datetime_with_underscore_separator(self) -> None:
        """Test _parse_datetime with underscore separator."""
        reader = SimulationReader()

        result = reader._parse_datetime("12/31/2020_23:59")

        assert result == datetime(2020, 12, 31, 23, 59, 0)

    def test_read_dt_description(self, tmp_path: Path) -> None:
        """Test parsing time step length with 'DT' description."""
        sim_file = tmp_path / "sim.in"
        sim_file.write_text("""C Simulation input file
TestModel                               / MODEL_NAME
2                                       / DT
HOUR                                    / TIME_STEP_UNIT
""")

        reader = SimulationReader()
        config = reader.read(sim_file)

        assert config.time_step_length == 2

    def test_read_unit_description(self, tmp_path: Path) -> None:
        """Test parsing time step unit with 'UNIT' description."""
        sim_file = tmp_path / "sim.in"
        sim_file.write_text("""C Simulation input file
TestModel                               / NAME
1                                       / TIME_STEP_LENGTH
MONTH                                   / UNIT
""")

        reader = SimulationReader()
        config = reader.read(sim_file)

        assert config.time_step_unit == TimeUnit.MONTH

    def test_read_budget_output_interval(self, tmp_path: Path) -> None:
        """Test reading budget output interval."""
        sim_file = tmp_path / "sim.in"
        sim_file.write_text("""C Simulation input file
TestModel                               / MODEL_NAME
1                                       / TIME_STEP_LENGTH
DAY                                     / TIME_STEP_UNIT
10                                      / OUTPUT_INTERVAL
""")

        reader = SimulationReader()
        config = reader.read(sim_file)

        assert config.output_interval == 10

    def test_read_empty_file(self, tmp_path: Path) -> None:
        """Test reading a file with only comments returns default config."""
        sim_file = tmp_path / "sim.in"
        sim_file.write_text("""C This file has no data
C Only comments
""")

        reader = SimulationReader()
        config = reader.read(sim_file)

        # Should return defaults
        assert config.model_name == "IWFM_Model"
        assert config.time_step_length == 1

    def test_read_invalid_integer_raises_error(self, tmp_path: Path) -> None:
        """Test that non-integer for max iterations raises error."""
        sim_file = tmp_path / "sim.in"
        sim_file.write_text("""C Simulation
TestModel                               / MODEL_NAME
1                                       / TIME_STEP_LENGTH
DAY                                     / TIME_STEP_UNIT
abc                                     / MAX_ITERATIONS
""")

        reader = SimulationReader()

        with pytest.raises(FileFormatError, match="Error parsing value"):
            reader.read(sim_file)


# =============================================================================
# Writer Output Content Verification
# =============================================================================


class TestWriterOutputBudgetIntervals:
    """Tests verifying budget and heads output interval writing."""

    def test_write_budget_and_heads_intervals(self, tmp_path: Path) -> None:
        """Test writing budget and heads output intervals."""
        file_config = SimulationFileConfig(output_dir=tmp_path)
        writer = SimulationWriter(file_config)
        config = SimulationConfig(
            output_interval=5,
            budget_output_interval=10,
            heads_output_interval=2,
        )

        filepath = writer.write(config)

        content = filepath.read_text()
        assert "BUDGET_OUTPUT_INTERVAL" in content
        assert "HEADS_OUTPUT_INTERVAL" in content

    def test_write_convergence_scientific_notation(self, tmp_path: Path) -> None:
        """Test writing convergence tolerance in scientific notation."""
        file_config = SimulationFileConfig(output_dir=tmp_path)
        writer = SimulationWriter(file_config)
        config = SimulationConfig(convergence_tolerance=1e-10)

        filepath = writer.write(config)

        content = filepath.read_text()
        assert "1.000000e-10" in content
        assert "STOPC" in content


# =============================================================================
# Convenience Function Edge Cases
# =============================================================================


class TestConvenienceFunctionEdgeCases:
    """Edge case tests for convenience functions."""

    def test_write_simulation_string_output_dir(self, tmp_path: Path) -> None:
        """Test write_simulation with string output directory."""
        config = SimulationConfig(model_name="StringPath")

        filepath = write_simulation(config, str(tmp_path))

        assert filepath.exists()
        content = filepath.read_text()
        assert "StringPath" in content

    def test_write_simulation_updates_file_config_output_dir(self, tmp_path: Path) -> None:
        """Test that write_simulation updates file_config output_dir."""
        config = SimulationConfig(model_name="UpdateTest")
        file_config = SimulationFileConfig(
            output_dir=tmp_path / "wrong_dir",
            main_file="custom.in",
        )

        filepath = write_simulation(config, tmp_path, file_config)

        assert filepath.exists()
        assert filepath.name == "custom.in"
        assert filepath.parent == tmp_path

    def test_read_simulation_string_path(self, tmp_path: Path) -> None:
        """Test read_simulation with string path."""
        sim_file = tmp_path / "sim.in"
        sim_file.write_text("""C Simulation
StringPathTest                          / MODEL_NAME
""")

        config = read_simulation(str(sim_file))

        assert config.model_name == "StringPathTest"


# =============================================================================
# SimulationConfig Properties
# =============================================================================


class TestSimulationConfigProperties:
    """Tests for SimulationConfig computed properties."""

    def test_n_time_steps_monthly(self) -> None:
        """Test n_time_steps with monthly time unit."""
        config = SimulationConfig(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 7, 1),
            time_step_length=1,
            time_step_unit=TimeUnit.MONTH,
        )
        # Monthly approximation - depends on TimeUnit implementation
        assert config.n_time_steps > 0

    def test_n_time_steps_zero_duration(self) -> None:
        """Test n_time_steps when start == end."""
        start = datetime(2020, 1, 1)
        config = SimulationConfig(
            start_date=start,
            end_date=start,
            time_step_length=1,
            time_step_unit=TimeUnit.DAY,
        )
        assert config.n_time_steps == 0

    def test_metadata_independence(self) -> None:
        """Test that metadata dicts are independent between instances."""
        c1 = SimulationConfig()
        c2 = SimulationConfig()

        c1.metadata["test"] = "value"

        assert "test" not in c2.metadata
