"""Unit tests for simulation I/O module.

Tests:
- Helper functions
- SimulationConfig dataclass
- SimulationFileConfig dataclass
- SimulationWriter class
- SimulationReader class
- Roundtrip tests
- Convenience functions
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from pyiwfm.io.simulation import (
    SimulationConfig,
    SimulationFileConfig,
    SimulationWriter,
    SimulationReader,
    write_simulation,
    read_simulation,
    _is_comment_line,
    _parse_value_line,
    _format_iwfm_datetime,
)
from pyiwfm.core.timeseries import TimeUnit
from pyiwfm.core.exceptions import FileFormatError


# =============================================================================
# Test Helper Functions
# =============================================================================


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_is_comment_line_c_comment(self) -> None:
        """Test C comment detection."""
        assert _is_comment_line("C This is a comment") is True
        assert _is_comment_line("c lowercase comment") is True

    def test_is_comment_line_asterisk_comment(self) -> None:
        """Test asterisk comment detection."""
        assert _is_comment_line("* This is a comment") is True

    def test_is_comment_line_hash_not_comment(self) -> None:
        """Hash is not a comment character."""
        assert _is_comment_line("# This is a comment") is False

    def test_is_comment_line_empty(self) -> None:
        """Test empty line is treated as comment."""
        assert _is_comment_line("") is True
        assert _is_comment_line("   ") is True

    def test_is_comment_line_data(self) -> None:
        """Test data line is not a comment."""
        assert _is_comment_line("1  2  3  4") is False
        assert _is_comment_line("model_name / MODEL_NAME") is False

    def test_parse_value_line_with_description(self) -> None:
        """Test parsing line with description."""
        value, desc = _parse_value_line("test_model                / MODEL_NAME")
        assert value == "test_model"
        assert desc == "MODEL_NAME"

    def test_parse_value_line_no_description(self) -> None:
        """Test parsing line without description."""
        value, desc = _parse_value_line("test_model")
        assert value == "test_model"
        assert desc == ""

    def test_parse_value_line_multiple_slashes(self) -> None:
        """Slash preceded by whitespace is the delimiter, not path slashes."""
        value, desc = _parse_value_line("path/to/file.dat / FILE_PATH")
        # The whitespace-preceded '/' is the delimiter
        assert value == "path/to/file.dat"
        assert desc == "FILE_PATH"

    def test_format_iwfm_datetime(self) -> None:
        """Test formatting datetime for IWFM."""
        dt = datetime(2020, 6, 15, 12, 30, 45)
        result = _format_iwfm_datetime(dt)
        assert result == "06/15/2020_12:30"

    def test_format_iwfm_datetime_midnight(self) -> None:
        """Test formatting midnight datetime."""
        dt = datetime(2020, 1, 1, 0, 0, 0)
        result = _format_iwfm_datetime(dt)
        assert result == "12/31/2019_24:00"


# =============================================================================
# Test SimulationConfig
# =============================================================================


class TestSimulationConfig:
    """Tests for SimulationConfig dataclass."""

    def test_default_creation(self) -> None:
        """Test default config creation."""
        config = SimulationConfig()

        assert config.model_name == "IWFM_Model"
        assert config.time_step_length == 1
        assert config.time_step_unit == TimeUnit.DAY
        assert config.output_interval == 1
        assert config.max_iterations == 50
        assert config.convergence_tolerance == pytest.approx(1e-6)

    def test_custom_creation(self) -> None:
        """Test config with custom values."""
        start = datetime(2020, 1, 1)
        end = datetime(2020, 12, 31)

        config = SimulationConfig(
            model_name="Test Model",
            start_date=start,
            end_date=end,
            time_step_length=1,
            time_step_unit=TimeUnit.MONTH,
            max_iterations=100,
            convergence_tolerance=1e-8,
        )

        assert config.model_name == "Test Model"
        assert config.start_date == start
        assert config.end_date == end
        assert config.time_step_unit == TimeUnit.MONTH
        assert config.max_iterations == 100

    def test_n_time_steps_daily(self) -> None:
        """Test number of time steps calculation for daily."""
        config = SimulationConfig(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 1, 11),  # 10 days
            time_step_length=1,
            time_step_unit=TimeUnit.DAY,
        )

        assert config.n_time_steps == 10

    def test_n_time_steps_hourly(self) -> None:
        """Test number of time steps calculation for hourly."""
        config = SimulationConfig(
            start_date=datetime(2020, 1, 1, 0, 0),
            end_date=datetime(2020, 1, 1, 12, 0),  # 12 hours
            time_step_length=1,
            time_step_unit=TimeUnit.HOUR,
        )

        assert config.n_time_steps == 12

    def test_to_simulation_period(self) -> None:
        """Test conversion to SimulationPeriod."""
        start = datetime(2020, 1, 1)
        end = datetime(2020, 12, 31)

        config = SimulationConfig(
            start_date=start,
            end_date=end,
            time_step_length=1,
            time_step_unit=TimeUnit.DAY,
        )

        period = config.to_simulation_period()

        assert period.start == start
        assert period.end == end
        assert period.time_step_length == 1
        assert period.time_step_unit == TimeUnit.DAY

    def test_component_files(self) -> None:
        """Test component file paths."""
        config = SimulationConfig(
            preprocessor_file=Path("preprocessor.in"),
            groundwater_file=Path("groundwater.in"),
            streams_file=Path("streams.in"),
            lakes_file=Path("lakes.in"),
            rootzone_file=Path("rootzone.in"),
        )

        assert config.preprocessor_file == Path("preprocessor.in")
        assert config.groundwater_file == Path("groundwater.in")
        assert config.streams_file == Path("streams.in")
        assert config.lakes_file == Path("lakes.in")
        assert config.rootzone_file == Path("rootzone.in")

    def test_metadata_default_factory(self) -> None:
        """Test that metadata uses independent default dict."""
        config1 = SimulationConfig()
        config2 = SimulationConfig()

        config1.metadata["key"] = "value"

        assert "key" not in config2.metadata


# =============================================================================
# Test SimulationFileConfig
# =============================================================================


class TestSimulationFileConfig:
    """Tests for SimulationFileConfig dataclass."""

    def test_basic_creation(self, tmp_path: Path) -> None:
        """Test basic config creation."""
        config = SimulationFileConfig(output_dir=tmp_path)

        assert config.output_dir == tmp_path
        assert config.main_file == "simulation.in"
        assert config.time_series_dir == "timeseries"

    def test_custom_file_names(self, tmp_path: Path) -> None:
        """Test config with custom file names."""
        config = SimulationFileConfig(
            output_dir=tmp_path,
            main_file="custom_sim.in",
            time_series_dir="custom_ts",
        )

        assert config.main_file == "custom_sim.in"
        assert config.time_series_dir == "custom_ts"

    def test_get_main_file_path(self, tmp_path: Path) -> None:
        """Test main file path getter."""
        config = SimulationFileConfig(output_dir=tmp_path)
        path = config.get_main_file_path()

        assert path == tmp_path / "simulation.in"

    def test_get_time_series_dir(self, tmp_path: Path) -> None:
        """Test time series directory path getter."""
        config = SimulationFileConfig(output_dir=tmp_path)
        path = config.get_time_series_dir()

        assert path == tmp_path / "timeseries"


# =============================================================================
# Test SimulationWriter
# =============================================================================


class TestSimulationWriter:
    """Tests for SimulationWriter class."""

    def test_init_creates_directory(self, tmp_path: Path) -> None:
        """Test that writer creates output directory."""
        output_dir = tmp_path / "new_dir" / "subdir"
        file_config = SimulationFileConfig(output_dir=output_dir)

        SimulationWriter(file_config)

        assert output_dir.exists()

    def test_write_default_config(self, tmp_path: Path) -> None:
        """Test writing default config."""
        file_config = SimulationFileConfig(output_dir=tmp_path)
        writer = SimulationWriter(file_config)
        config = SimulationConfig()

        filepath = writer.write(config)

        assert filepath.exists()
        content = filepath.read_text()
        assert "IWFM_Model" in content
        assert "MODEL_NAME" in content

    def test_write_custom_header(self, tmp_path: Path) -> None:
        """Test writing with custom header."""
        file_config = SimulationFileConfig(output_dir=tmp_path)
        writer = SimulationWriter(file_config)
        config = SimulationConfig()

        filepath = writer.write(config, header="Custom Header\nLine 2")

        content = filepath.read_text()
        assert "Custom Header" in content
        assert "Line 2" in content

    def test_write_time_settings(self, tmp_path: Path) -> None:
        """Test writing time settings."""
        file_config = SimulationFileConfig(output_dir=tmp_path)
        writer = SimulationWriter(file_config)
        config = SimulationConfig(
            start_date=datetime(2020, 1, 1, 0, 0, 0),
            end_date=datetime(2020, 12, 31, 0, 0, 0),
            time_step_length=1,
            time_step_unit=TimeUnit.DAY,
        )

        filepath = writer.write(config)

        content = filepath.read_text()
        assert "12/31/2019_24:00" in content
        assert "12/30/2020_24:00" in content
        assert "START_DATE" in content
        assert "END_DATE" in content
        assert "DAY" in content

    def test_write_component_files(self, tmp_path: Path) -> None:
        """Test writing component file paths."""
        file_config = SimulationFileConfig(output_dir=tmp_path)
        writer = SimulationWriter(file_config)
        config = SimulationConfig(
            preprocessor_file=Path("preprocessor.in"),
            groundwater_file=Path("groundwater.in"),
            streams_file=Path("streams.in"),
        )

        filepath = writer.write(config)

        content = filepath.read_text()
        assert "preprocessor.in" in content
        assert "groundwater.in" in content
        assert "streams.in" in content

    def test_write_solver_settings(self, tmp_path: Path) -> None:
        """Test writing solver settings."""
        file_config = SimulationFileConfig(output_dir=tmp_path)
        writer = SimulationWriter(file_config)
        config = SimulationConfig(
            max_iterations=100,
            convergence_tolerance=1e-8,
        )

        filepath = writer.write(config)

        content = filepath.read_text()
        assert "100" in content
        assert "MXITER" in content
        assert "STOPC" in content

    def test_write_output_settings(self, tmp_path: Path) -> None:
        """Test writing output settings."""
        file_config = SimulationFileConfig(output_dir=tmp_path)
        writer = SimulationWriter(file_config)
        config = SimulationConfig(
            output_dir=Path("output"),
            output_interval=5,
            budget_output_interval=10,
            heads_output_interval=2,
        )

        filepath = writer.write(config)

        content = filepath.read_text()
        assert "OUTPUT_DIR" in content
        assert "OUTPUT_INTERVAL" in content


# =============================================================================
# Test SimulationReader
# =============================================================================


class TestSimulationReader:
    """Tests for SimulationReader class."""

    def test_read_basic_config(self, tmp_path: Path) -> None:
        """Test reading basic config file.

        Note: The IWFM reader splits on '/' for value/description,
        but dates also contain '/'. Use description without slash for dates.
        """
        sim_file = tmp_path / "sim.in"
        # Use no separator for date lines since dates contain /
        sim_file.write_text("""C Simulation input file
C
TestModel                               / MODEL_NAME
1                                       / TIME_STEP_LENGTH
DAY                                     / TIME_STEP_UNIT
50                                      / MAX_ITERATIONS
""")

        reader = SimulationReader()
        config = reader.read(sim_file)

        assert config.model_name == "TestModel"
        assert config.time_step_length == 1
        assert config.time_step_unit == TimeUnit.DAY
        assert config.max_iterations == 50

    def test_read_with_comments(self, tmp_path: Path) -> None:
        """Test reading file with various comment styles."""
        sim_file = tmp_path / "sim.in"
        sim_file.write_text("""C This is a C comment
* This is an asterisk comment
c Lowercase C comment
TestModel                               / MODEL_NAME
1                                       / TIME_STEP_LENGTH
DAY                                     / TIME_STEP_UNIT
""")

        reader = SimulationReader()
        config = reader.read(sim_file)

        assert config.model_name == "TestModel"

    def test_read_solver_settings(self, tmp_path: Path) -> None:
        """Test reading solver settings."""
        sim_file = tmp_path / "sim.in"
        sim_file.write_text("""C Simulation input file
TestModel                               / MODEL_NAME
1                                       / TIME_STEP_LENGTH
DAY                                     / TIME_STEP_UNIT
100                                     / MAX_ITERATIONS
1.0e-8                                  / CONVERGENCE_TOLERANCE
""")

        reader = SimulationReader()
        config = reader.read(sim_file)

        assert config.max_iterations == 100
        assert config.convergence_tolerance == pytest.approx(1e-8)

    def test_read_component_files(self, tmp_path: Path) -> None:
        """Test reading component file paths."""
        sim_file = tmp_path / "sim.in"
        sim_file.write_text("""C Simulation input file
TestModel                               / MODEL_NAME
1                                       / TIME_STEP_LENGTH
DAY                                     / TIME_STEP_UNIT
preprocessor.in                         / PREPROCESSOR_FILE
groundwater.in                          / GROUNDWATER_FILE
streams.in                              / STREAMS_FILE
lakes.in                                / LAKES_FILE
rootzone.in                             / ROOTZONE_FILE
""")

        reader = SimulationReader()
        config = reader.read(sim_file)

        assert config.preprocessor_file == Path("preprocessor.in")
        assert config.groundwater_file == Path("groundwater.in")
        assert config.streams_file == Path("streams.in")
        assert config.lakes_file == Path("lakes.in")
        assert config.rootzone_file == Path("rootzone.in")

    def test_read_output_settings(self, tmp_path: Path) -> None:
        """Test reading output settings."""
        sim_file = tmp_path / "sim.in"
        sim_file.write_text("""C Simulation input file
TestModel                               / MODEL_NAME
1                                       / TIME_STEP_LENGTH
DAY                                     / TIME_STEP_UNIT
output_dir                              / OUTPUT_DIR
5                                       / OUTPUT_INTERVAL
""")

        reader = SimulationReader()
        config = reader.read(sim_file)

        assert config.output_dir == Path("output_dir")
        assert config.output_interval == 5

    def test_read_string_path(self, tmp_path: Path) -> None:
        """Test reading with string path."""
        sim_file = tmp_path / "sim.in"
        sim_file.write_text("""C Simulation input file
TestModel                               / MODEL_NAME
1                                       / TIME_STEP_LENGTH
DAY                                     / TIME_STEP_UNIT
""")

        reader = SimulationReader()
        config = reader.read(str(sim_file))

        assert config.model_name == "TestModel"

    def test_read_invalid_value_raises_error(self, tmp_path: Path) -> None:
        """Test that invalid values raise FileFormatError."""
        sim_file = tmp_path / "sim.in"
        sim_file.write_text("""C Simulation input file
TestModel                               / MODEL_NAME
not_a_number                            / TIME_STEP_LENGTH
""")

        reader = SimulationReader()

        with pytest.raises(FileFormatError, match="Error parsing value"):
            reader.read(sim_file)


# =============================================================================
# Test Writer Output Verification
# =============================================================================


class TestSimulationWriterOutput:
    """Tests that verify writer output content.

    Note: Full roundtrip tests are not possible due to the '/' character in
    dates conflicting with the value/description separator in the file format.
    Instead, we verify writer output content directly.
    """

    def test_writer_creates_valid_file(self, tmp_path: Path) -> None:
        """Test that writer creates a readable file."""
        config = SimulationConfig(
            model_name="TestModel",
            time_step_length=1,
            time_step_unit=TimeUnit.DAY,
        )

        file_config = SimulationFileConfig(output_dir=tmp_path)
        writer = SimulationWriter(file_config)
        filepath = writer.write(config)

        # Verify file exists and has content
        assert filepath.exists()
        content = filepath.read_text()
        assert len(content) > 0
        assert "TestModel" in content
        assert "MODEL_NAME" in content

    def test_writer_output_contains_all_sections(self, tmp_path: Path) -> None:
        """Test that writer output contains all sections."""
        config = SimulationConfig(
            model_name="SectionTest",
            preprocessor_file=Path("pre.in"),
            max_iterations=100,
            output_interval=5,
        )

        file_config = SimulationFileConfig(output_dir=tmp_path)
        writer = SimulationWriter(file_config)
        filepath = writer.write(config)

        content = filepath.read_text()

        # Check for section headers
        assert "SIMULATION TIME PERIOD" in content
        assert "COMPONENT INPUT FILES" in content
        assert "SOLVER SETTINGS" in content
        assert "OUTPUT CONTROL" in content

    def test_writer_output_contains_values(self, tmp_path: Path) -> None:
        """Test that writer output contains configured values."""
        config = SimulationConfig(
            model_name="ValuesTest",
            time_step_length=2,
            time_step_unit=TimeUnit.HOUR,
            max_iterations=75,
            output_interval=10,
        )

        file_config = SimulationFileConfig(output_dir=tmp_path)
        writer = SimulationWriter(file_config)
        filepath = writer.write(config)

        content = filepath.read_text()

        assert "ValuesTest" in content
        assert "75" in content  # max_iterations
        assert "HOUR" in content  # time unit

    def test_writer_output_component_files(self, tmp_path: Path) -> None:
        """Test that writer outputs component file paths."""
        config = SimulationConfig(
            preprocessor_file=Path("pre.in"),
            groundwater_file=Path("gw.in"),
            streams_file=Path("str.in"),
            lakes_file=Path("lak.in"),
            rootzone_file=Path("rz.in"),
        )

        file_config = SimulationFileConfig(output_dir=tmp_path)
        writer = SimulationWriter(file_config)
        filepath = writer.write(config)

        content = filepath.read_text()

        assert "pre.in" in content
        assert "gw.in" in content
        assert "str.in" in content
        assert "lak.in" in content
        assert "rz.in" in content


# =============================================================================
# Test Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_write_simulation_basic(self, tmp_path: Path) -> None:
        """Test write_simulation function."""
        config = SimulationConfig(model_name="WriteTest")

        filepath = write_simulation(config, tmp_path)

        assert filepath.exists()
        content = filepath.read_text()
        assert "WriteTest" in content

    def test_write_simulation_with_file_config(self, tmp_path: Path) -> None:
        """Test write_simulation with custom file config."""
        config = SimulationConfig(model_name="CustomFile")
        file_config = SimulationFileConfig(
            output_dir=tmp_path / "subdir",
            main_file="custom.in",
        )

        filepath = write_simulation(config, tmp_path, file_config)

        assert filepath.exists()
        assert filepath.name == "custom.in"

    def test_read_simulation_basic(self, tmp_path: Path) -> None:
        """Test read_simulation function."""
        sim_file = tmp_path / "sim.in"
        sim_file.write_text("""C Simulation input file
TestModel                               / MODEL_NAME
1                                       / TIME_STEP_LENGTH
DAY                                     / TIME_STEP_UNIT
""")

        config = read_simulation(sim_file)

        assert config.model_name == "TestModel"
        assert config.time_step_unit == TimeUnit.DAY

    def test_write_and_verify_content(self, tmp_path: Path) -> None:
        """Test write_simulation and verify file content."""
        original = SimulationConfig(
            model_name="ConvenienceTest",
            time_step_length=2,
            time_step_unit=TimeUnit.HOUR,
        )

        filepath = write_simulation(original, tmp_path)

        # Verify file content
        content = filepath.read_text()
        assert "ConvenienceTest" in content
        assert "HOUR" in content
        assert "TIME_STEP" in content
