"""Unit tests for writer base classes.

Tests:
- TemplateWriter
- TimeSeriesSpec
- TimeSeriesWriter
- ComponentWriter
"""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pyiwfm.io.writer_base import (
    TemplateWriter,
    TimeSeriesSpec,
    TimeSeriesWriter,
    ComponentWriter,
)
from pyiwfm.io.config import OutputFormat, TimeSeriesOutputConfig
from pyiwfm.templates.engine import TemplateEngine


# =============================================================================
# Test TemplateWriter
# =============================================================================


class ConcreteTemplateWriter(TemplateWriter):
    """Concrete implementation for testing abstract TemplateWriter."""

    def write(self, data: Any = None) -> None:
        pass

    @property
    def format(self) -> str:
        return "test_format"


class TestTemplateWriter:
    """Tests for TemplateWriter base class."""

    def test_initialization(self, tmp_path: Path) -> None:
        """Test basic initialization."""
        writer = ConcreteTemplateWriter(tmp_path)

        assert writer.output_dir == tmp_path
        assert writer._engine is not None

    def test_initialization_with_custom_engine(self, tmp_path: Path) -> None:
        """Test initialization with custom template engine."""
        engine = TemplateEngine()
        writer = ConcreteTemplateWriter(tmp_path, template_engine=engine)

        assert writer._engine is engine

    def test_ensure_dir(self, tmp_path: Path) -> None:
        """Test _ensure_dir creates parent directories."""
        writer = ConcreteTemplateWriter(tmp_path)
        nested_path = tmp_path / "a" / "b" / "c" / "file.txt"

        # Parent should not exist yet
        assert not nested_path.parent.exists()

        writer._ensure_dir(nested_path)

        # Now parent should exist
        assert nested_path.parent.exists()

    def test_render_string(self, tmp_path: Path) -> None:
        """Test render_string method."""
        writer = ConcreteTemplateWriter(tmp_path)

        result = writer.render_string("Hello {{ name }}!", name="World")

        assert result == "Hello World!"

    def test_write_data_block(self, tmp_path: Path) -> None:
        """Test write_data_block method."""
        writer = ConcreteTemplateWriter(tmp_path)

        # Create test data
        data = np.array([[1.0, 2.0], [3.0, 4.0]])

        # Write to StringIO
        buffer = io.StringIO()
        writer.write_data_block(buffer, data, fmt="%8.2f")
        result = buffer.getvalue()

        # Check output
        assert "1.00" in result
        assert "2.00" in result
        assert "3.00" in result
        assert "4.00" in result

    def test_write_data_block_with_header(self, tmp_path: Path) -> None:
        """Test write_data_block with header comment."""
        writer = ConcreteTemplateWriter(tmp_path)

        data = np.array([1.0, 2.0, 3.0])
        buffer = io.StringIO()
        writer.write_data_block(buffer, data, fmt="%8.2f", header_comment="Test data")
        result = buffer.getvalue()

        assert "C  Test data" in result

    def test_write_indexed_data(self, tmp_path: Path) -> None:
        """Test write_indexed_data method."""
        writer = ConcreteTemplateWriter(tmp_path)

        ids = np.array([1, 2, 3], dtype=np.int32)
        data = np.array([10.0, 20.0, 30.0])

        buffer = io.StringIO()
        writer.write_indexed_data(buffer, ids, data, id_fmt="%5d", data_fmt="%10.2f")
        result = buffer.getvalue()

        lines = result.strip().split("\n")
        assert len(lines) == 3

        # Check that IDs and values are present
        assert "1" in lines[0] and "10.00" in lines[0]
        assert "2" in lines[1] and "20.00" in lines[1]
        assert "3" in lines[2] and "30.00" in lines[2]

    def test_write_indexed_data_2d(self, tmp_path: Path) -> None:
        """Test write_indexed_data with 2D data."""
        writer = ConcreteTemplateWriter(tmp_path)

        ids = np.array([1, 2], dtype=np.int32)
        data = np.array([[10.0, 100.0], [20.0, 200.0]])

        buffer = io.StringIO()
        writer.write_indexed_data(buffer, ids, data)
        result = buffer.getvalue()

        lines = result.strip().split("\n")
        assert len(lines) == 2

        # Check multi-column output
        assert "100" in lines[0]
        assert "200" in lines[1]


# =============================================================================
# Test TimeSeriesSpec
# =============================================================================


class TestTimeSeriesSpec:
    """Tests for TimeSeriesSpec dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic TimeSeriesSpec creation."""
        dates = [datetime(2020, 1, 1), datetime(2020, 1, 2)]
        values = [10.0, 20.0]

        spec = TimeSeriesSpec(
            name="Test TS",
            dates=dates,
            values=values,
        )

        assert spec.name == "Test TS"
        assert spec.dates == dates
        assert spec.values == values
        assert spec.units == ""  # Default
        assert spec.interval == "1DAY"  # Default

    def test_full_creation(self) -> None:
        """Test TimeSeriesSpec with all fields."""
        dates = [datetime(2020, 1, 1)]
        values = [100.0]

        spec = TimeSeriesSpec(
            name="Flow",
            dates=dates,
            values=values,
            units="cfs",
            location="Station1",
            parameter="FLOW",
            interval="1HOUR",
        )

        assert spec.units == "cfs"
        assert spec.location == "Station1"
        assert spec.parameter == "FLOW"
        assert spec.interval == "1HOUR"

    def test_numpy_arrays(self) -> None:
        """Test TimeSeriesSpec with numpy arrays."""
        dates = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]")
        values = np.array([10.0, 20.0])

        spec = TimeSeriesSpec(
            name="Test",
            dates=dates,
            values=values,
        )

        assert len(spec.dates) == 2
        assert len(spec.values) == 2


# =============================================================================
# Test TimeSeriesWriter
# =============================================================================


class TestTimeSeriesWriter:
    """Tests for TimeSeriesWriter class."""

    def test_initialization(self, tmp_path: Path) -> None:
        """Test TimeSeriesWriter initialization."""
        config = TimeSeriesOutputConfig()
        writer = TimeSeriesWriter(config, tmp_path)

        assert writer.config is config
        assert writer.output_dir == tmp_path

    def test_write_text_timeseries(self, tmp_path: Path) -> None:
        """Test writing time series to text file."""
        config = TimeSeriesOutputConfig(format=OutputFormat.TEXT)
        writer = TimeSeriesWriter(config, tmp_path)

        dates = [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)]
        values = [10.0, 20.0, 30.0]

        spec = TimeSeriesSpec(
            name="Test Flow",
            dates=dates,
            values=values,
            units="cfs",
            location="Station1",
        )

        writer.write_timeseries(spec, "test_ts.dat")

        # Check file exists and contains expected content
        output_file = tmp_path / "test_ts.dat"
        assert output_file.exists()

        content = output_file.read_text()
        assert "Time series: Test Flow" in content
        assert "Units: cfs" in content
        assert "Location: Station1" in content
        assert "01/01/2020" in content
        assert "10.000000" in content

    def test_write_timeseries_table(self, tmp_path: Path) -> None:
        """Test writing multiple time series to a table."""
        config = TimeSeriesOutputConfig(format=OutputFormat.TEXT)
        writer = TimeSeriesWriter(config, tmp_path)

        dates = [datetime(2020, 1, 1), datetime(2020, 1, 2)]
        columns = {
            "Flow": np.array([10.0, 20.0]),
            "Stage": np.array([5.0, 6.0]),
        }

        writer.write_timeseries_table(
            dates=dates,
            columns=columns,
            text_file="table.dat",
            header_lines=["Test table", "Two columns"],
        )

        output_file = tmp_path / "table.dat"
        assert output_file.exists()

        content = output_file.read_text()
        assert "Test table" in content
        assert "Two columns" in content
        assert "Flow" in content
        assert "Stage" in content
        assert "01/01/2020" in content

    def test_write_timeseries_requires_text_file(self, tmp_path: Path) -> None:
        """Test that write_timeseries requires text_file for TEXT format."""
        config = TimeSeriesOutputConfig(format=OutputFormat.TEXT)
        writer = TimeSeriesWriter(config, tmp_path)

        spec = TimeSeriesSpec(
            name="Test",
            dates=[datetime(2020, 1, 1)],
            values=[10.0],
        )

        with pytest.raises(ValueError, match="text_file required"):
            writer.write_timeseries(spec)

    def test_close(self, tmp_path: Path) -> None:
        """Test close method."""
        config = TimeSeriesOutputConfig()
        writer = TimeSeriesWriter(config, tmp_path)

        # Should not raise even if nothing to close
        writer.close()
        assert writer._dss_file is None


# =============================================================================
# Test ComponentWriter
# =============================================================================


class ConcreteComponentWriter(ComponentWriter):
    """Concrete implementation for testing ComponentWriter."""

    def write(self, data: Any = None) -> None:
        pass


class TestComponentWriter:
    """Tests for ComponentWriter base class."""

    def test_initialization(self, tmp_path: Path) -> None:
        """Test ComponentWriter initialization."""
        writer = ConcreteComponentWriter(tmp_path)

        assert writer.output_dir == tmp_path
        assert writer.ts_config is not None
        assert writer.format == "iwfm_component"

    def test_initialization_with_ts_config(self, tmp_path: Path) -> None:
        """Test initialization with custom time series config."""
        ts_config = TimeSeriesOutputConfig(format=OutputFormat.DSS)
        writer = ConcreteComponentWriter(tmp_path, ts_config=ts_config)

        assert writer.ts_config is ts_config

    def test_ts_writer_property(self, tmp_path: Path) -> None:
        """Test ts_writer property creates writer on demand."""
        writer = ConcreteComponentWriter(tmp_path)

        # First access creates writer
        ts_writer = writer.ts_writer
        assert ts_writer is not None
        assert isinstance(ts_writer, TimeSeriesWriter)

        # Second access returns same instance
        ts_writer2 = writer.ts_writer
        assert ts_writer2 is ts_writer

    def test_write_component_header(self, tmp_path: Path) -> None:
        """Test write_component_header method."""
        writer = ConcreteComponentWriter(tmp_path)

        buffer = io.StringIO()
        writer.write_component_header(
            buffer,
            component_name="Test Component",
            version="1.0",
            description="Test description",
        )
        result = buffer.getvalue()

        assert "Test Component" in result
        assert "Version: 1.0" in result
        assert "Test description" in result
        assert "Generated by pyiwfm" in result

    def test_write_file_reference(self, tmp_path: Path) -> None:
        """Test write_file_reference method."""
        writer = ConcreteComponentWriter(tmp_path)

        buffer = io.StringIO()
        writer.write_file_reference(
            buffer,
            ref_path="path/to/file.dat",
            description="Data file",
        )
        result = buffer.getvalue()

        assert "path/to/file.dat" in result
        assert "Data file" in result

    def test_write_file_reference_none(self, tmp_path: Path) -> None:
        """Test write_file_reference with None path."""
        writer = ConcreteComponentWriter(tmp_path)

        buffer = io.StringIO()
        writer.write_file_reference(buffer, ref_path=None, description="Optional file")
        result = buffer.getvalue()

        # Should have empty path but still have description
        assert "Optional file" in result

    def test_write_value_line(self, tmp_path: Path) -> None:
        """Test write_value_line method."""
        writer = ConcreteComponentWriter(tmp_path)

        buffer = io.StringIO()
        writer.write_value_line(buffer, 42, description="Answer")
        result = buffer.getvalue()

        assert "42" in result
        assert "Answer" in result

    def test_write_value_line_float(self, tmp_path: Path) -> None:
        """Test write_value_line with float value."""
        writer = ConcreteComponentWriter(tmp_path)

        buffer = io.StringIO()
        writer.write_value_line(buffer, 3.14159, description="Pi")
        result = buffer.getvalue()

        assert "3.141590" in result  # 6 decimal places
        assert "Pi" in result

    def test_write_value_line_no_description(self, tmp_path: Path) -> None:
        """Test write_value_line without description."""
        writer = ConcreteComponentWriter(tmp_path)

        buffer = io.StringIO()
        writer.write_value_line(buffer, 100)
        result = buffer.getvalue()

        assert "100" in result
        assert "/" not in result  # No description separator


# =============================================================================
# Additional Tests - _check_dss function
# =============================================================================


class TestCheckDSS:
    """Tests for the _check_dss helper function."""

    def test_check_dss_raises_when_no_dss(self, monkeypatch) -> None:
        """Test _check_dss raises ImportError when DSS is not available."""
        import pyiwfm.io.writer_base as wb

        monkeypatch.setattr(wb, "HAS_DSS", False)
        with pytest.raises(ImportError, match="DSS support requires the bundled HEC-DSS library"):
            wb._check_dss()

    def test_check_dss_passes_when_available(self, monkeypatch) -> None:
        """Test _check_dss does not raise when DSS is available."""
        import pyiwfm.io.writer_base as wb

        monkeypatch.setattr(wb, "HAS_DSS", True)
        # Should not raise
        wb._check_dss()


# =============================================================================
# Additional Tests - TemplateWriter edge cases
# =============================================================================


class TestTemplateWriterAdditional:
    """Additional tests for TemplateWriter base class."""

    def test_initialization_with_string_path(self, tmp_path: Path) -> None:
        """Test initialization with string path instead of Path object."""
        writer = ConcreteTemplateWriter(str(tmp_path))

        assert writer.output_dir == tmp_path
        assert isinstance(writer.output_dir, Path)

    def test_render_header(self, tmp_path: Path) -> None:
        """Test render_header delegates to engine.render_template."""
        engine = TemplateEngine()
        writer = ConcreteTemplateWriter(tmp_path, template_engine=engine)

        # render_header calls engine.render_template which requires an actual
        # template file. We test the delegation by mocking the engine.
        from unittest.mock import MagicMock

        writer._engine = MagicMock()
        writer._engine.render_template.return_value = "rendered header"

        result = writer.render_header("test_template.j2", key="value")

        writer._engine.render_template.assert_called_once_with(
            "test_template.j2", key="value"
        )
        assert result == "rendered header"

    def test_write_data_block_with_multiple_formats(self, tmp_path: Path) -> None:
        """Test write_data_block with a list of format strings."""
        writer = ConcreteTemplateWriter(tmp_path)

        data = np.array([[1.0, 200], [3.0, 400]])
        buffer = io.StringIO()
        writer.write_data_block(buffer, data, fmt=["%8.2f", "%10d"])
        result = buffer.getvalue()

        assert "1.00" in result
        assert "200" in result
        assert "3.00" in result
        assert "400" in result

    def test_write_data_block_no_header_comment(self, tmp_path: Path) -> None:
        """Test write_data_block without header comment (header_comment=None)."""
        writer = ConcreteTemplateWriter(tmp_path)

        data = np.array([5.0, 6.0])
        buffer = io.StringIO()
        writer.write_data_block(buffer, data, fmt="%8.2f", header_comment=None)
        result = buffer.getvalue()

        assert "C " not in result
        assert "5.00" in result

    def test_format_property(self, tmp_path: Path) -> None:
        """Test the format property returns expected value."""
        writer = ConcreteTemplateWriter(tmp_path)
        assert writer.format == "test_format"

    def test_abstract_methods_required(self) -> None:
        """Test that TemplateWriter cannot be instantiated directly."""
        with pytest.raises(TypeError):
            TemplateWriter("/tmp")  # type: ignore[abstract]

    def test_render_string_complex_template(self, tmp_path: Path) -> None:
        """Test render_string with more complex template syntax."""
        writer = ConcreteTemplateWriter(tmp_path)

        result = writer.render_string(
            "{% for i in items %}{{ i }} {% endfor %}", items=[1, 2, 3]
        )
        assert result == "1 2 3 "


# =============================================================================
# Additional Tests - TimeSeriesWriter edge cases
# =============================================================================


class TestTimeSeriesWriterAdditional:
    """Additional tests for TimeSeriesWriter edge cases."""

    def test_write_timeseries_dss_format_raises_no_dss(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Test writing DSS format raises ImportError when DSS unavailable."""
        import pyiwfm.io.writer_base as wb

        monkeypatch.setattr(wb, "HAS_DSS", False)

        config = TimeSeriesOutputConfig(format=OutputFormat.DSS)
        writer = TimeSeriesWriter(config, tmp_path)

        spec = TimeSeriesSpec(
            name="Test",
            dates=[datetime(2020, 1, 1)],
            values=[10.0],
        )

        with pytest.raises(ImportError, match="DSS support requires the bundled HEC-DSS library"):
            writer.write_timeseries(spec)

    def test_write_timeseries_both_format_requires_text_file(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Test BOTH format requires text_file parameter."""
        import pyiwfm.io.writer_base as wb

        monkeypatch.setattr(wb, "HAS_DSS", False)

        config = TimeSeriesOutputConfig(format=OutputFormat.BOTH)
        writer = TimeSeriesWriter(config, tmp_path)

        spec = TimeSeriesSpec(
            name="Test",
            dates=[datetime(2020, 1, 1)],
            values=[10.0],
        )

        with pytest.raises(ValueError, match="text_file required"):
            writer.write_timeseries(spec)

    def test_write_timeseries_both_format_writes_text_then_raises_dss(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Test BOTH format writes text file but raises on DSS when unavailable."""
        import pyiwfm.io.writer_base as wb

        monkeypatch.setattr(wb, "HAS_DSS", False)

        config = TimeSeriesOutputConfig(format=OutputFormat.BOTH)
        writer = TimeSeriesWriter(config, tmp_path)

        spec = TimeSeriesSpec(
            name="Test Flow",
            dates=[datetime(2020, 1, 1)],
            values=[10.0],
            units="cfs",
        )

        with pytest.raises(ImportError, match="DSS support requires the bundled HEC-DSS library"):
            writer.write_timeseries(spec, text_file="output.dat")

        # Text file should have been written before the DSS error
        output_file = tmp_path / "output.dat"
        assert output_file.exists()

    def test_write_timeseries_table_dss_format_skips(self, tmp_path: Path) -> None:
        """Test write_timeseries_table returns early for DSS-only format."""
        config = TimeSeriesOutputConfig(format=OutputFormat.DSS)
        writer = TimeSeriesWriter(config, tmp_path)

        dates = [datetime(2020, 1, 1)]
        columns = {"Flow": np.array([10.0])}

        # Should return early without writing
        writer.write_timeseries_table(
            dates=dates,
            columns=columns,
            text_file="should_not_exist.dat",
        )

        output_file = tmp_path / "should_not_exist.dat"
        assert not output_file.exists()

    def test_write_timeseries_table_no_header(self, tmp_path: Path) -> None:
        """Test write_timeseries_table without header lines."""
        config = TimeSeriesOutputConfig(format=OutputFormat.TEXT)
        writer = TimeSeriesWriter(config, tmp_path)

        dates = [datetime(2020, 3, 15)]
        columns = {"Depth": np.array([42.5])}

        writer.write_timeseries_table(
            dates=dates,
            columns=columns,
            text_file="no_header.dat",
            header_lines=None,
        )

        output_file = tmp_path / "no_header.dat"
        assert output_file.exists()

        content = output_file.read_text()
        # Should not have any comment lines except column header line
        lines = content.strip().split("\n")
        # First line should be column header, second line should be data
        assert "Depth" in lines[0]
        assert "03/14/2020_24:00" in lines[1]
        # No "C  " comment lines for header text
        assert not any(
            line.startswith("C  ") and "Depth" not in line and "DATE/TIME" not in line
            for line in lines
        )

    def test_write_timeseries_table_both_format(self, tmp_path: Path) -> None:
        """Test write_timeseries_table with BOTH format writes text file."""
        config = TimeSeriesOutputConfig(format=OutputFormat.BOTH)
        writer = TimeSeriesWriter(config, tmp_path)

        dates = [datetime(2020, 6, 1)]
        columns = {"Stage": np.array([12.34])}

        writer.write_timeseries_table(
            dates=dates,
            columns=columns,
            text_file="both_table.dat",
        )

        output_file = tmp_path / "both_table.dat"
        assert output_file.exists()
        content = output_file.read_text()
        assert "Stage" in content
        assert "12.340000" in content

    def test_write_text_timeseries_numpy_datetime(self, tmp_path: Path) -> None:
        """Test _write_text_timeseries with numpy datetime64 dates."""
        config = TimeSeriesOutputConfig(format=OutputFormat.TEXT)
        writer = TimeSeriesWriter(config, tmp_path)

        dates = np.array(["2020-07-04", "2020-07-05"], dtype="datetime64[D]")
        values = [100.0, 200.0]

        spec = TimeSeriesSpec(
            name="Numpy Dates",
            dates=dates,
            values=values,
        )

        writer.write_timeseries(spec, "numpy_dates.dat")

        output_file = tmp_path / "numpy_dates.dat"
        assert output_file.exists()

        content = output_file.read_text()
        assert "07/03/2020_24:00" in content
        assert "07/04/2020_24:00" in content
        assert "100.000000" in content
        assert "200.000000" in content

    def test_write_text_timeseries_no_units_no_location(
        self, tmp_path: Path
    ) -> None:
        """Test _write_text_timeseries without units or location."""
        config = TimeSeriesOutputConfig(format=OutputFormat.TEXT)
        writer = TimeSeriesWriter(config, tmp_path)

        spec = TimeSeriesSpec(
            name="Bare",
            dates=[datetime(2020, 1, 1)],
            values=[5.0],
        )

        writer.write_timeseries(spec, "bare.dat")

        output_file = tmp_path / "bare.dat"
        content = output_file.read_text()

        assert "Time series: Bare" in content
        assert "Units:" not in content
        assert "Location:" not in content

    def test_write_timeseries_table_numpy_datetime(self, tmp_path: Path) -> None:
        """Test write_timeseries_table with numpy datetime64 dates."""
        config = TimeSeriesOutputConfig(format=OutputFormat.TEXT)
        writer = TimeSeriesWriter(config, tmp_path)

        dates = np.array(["2021-12-25"], dtype="datetime64[D]")
        columns = {"Precip": np.array([2.5])}

        writer.write_timeseries_table(
            dates=dates,
            columns=columns,
            text_file="np_table.dat",
        )

        output_file = tmp_path / "np_table.dat"
        content = output_file.read_text()
        assert "12/24/2021_24:00" in content
        assert "2.500000" in content

    def test_write_dss_timeseries_no_dss_file_configured(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Test _write_dss_timeseries raises when dss_file not configured."""
        import pyiwfm.io.writer_base as wb

        monkeypatch.setattr(wb, "HAS_DSS", True)

        config = TimeSeriesOutputConfig(format=OutputFormat.DSS, dss_file=None)
        writer = TimeSeriesWriter(config, tmp_path)

        spec = TimeSeriesSpec(
            name="Test",
            dates=[datetime(2020, 1, 1)],
            values=[10.0],
        )

        with pytest.raises(ValueError, match="DSS file not configured"):
            writer._write_dss_timeseries(spec)

    def test_close_with_dss_file(self, tmp_path: Path) -> None:
        """Test close method when _dss_file is set."""
        from unittest.mock import MagicMock

        config = TimeSeriesOutputConfig()
        writer = TimeSeriesWriter(config, tmp_path)

        mock_dss = MagicMock()
        writer._dss_file = mock_dss

        writer.close()

        mock_dss.close.assert_called_once()
        assert writer._dss_file is None

    def test_close_idempotent(self, tmp_path: Path) -> None:
        """Test calling close multiple times is safe."""
        config = TimeSeriesOutputConfig()
        writer = TimeSeriesWriter(config, tmp_path)

        writer.close()
        writer.close()
        assert writer._dss_file is None

    def test_write_timeseries_table_nested_dir(self, tmp_path: Path) -> None:
        """Test write_timeseries_table creates nested directories."""
        config = TimeSeriesOutputConfig(format=OutputFormat.TEXT)
        writer = TimeSeriesWriter(config, tmp_path)

        dates = [datetime(2020, 1, 1)]
        columns = {"Flow": np.array([1.0])}

        writer.write_timeseries_table(
            dates=dates,
            columns=columns,
            text_file="subdir/nested/table.dat",
        )

        output_file = tmp_path / "subdir" / "nested" / "table.dat"
        assert output_file.exists()

    def test_write_text_timeseries_nested_dir(self, tmp_path: Path) -> None:
        """Test _write_text_timeseries creates nested directories."""
        config = TimeSeriesOutputConfig(format=OutputFormat.TEXT)
        writer = TimeSeriesWriter(config, tmp_path)

        spec = TimeSeriesSpec(
            name="Nested",
            dates=[datetime(2020, 1, 1)],
            values=[1.0],
        )

        writer.write_timeseries(spec, "deep/path/ts.dat")

        output_file = tmp_path / "deep" / "path" / "ts.dat"
        assert output_file.exists()


# =============================================================================
# Additional Tests - ComponentWriter edge cases
# =============================================================================


class TestComponentWriterAdditional:
    """Additional tests for ComponentWriter edge cases."""

    def test_write_component_header_minimal(self, tmp_path: Path) -> None:
        """Test write_component_header with only component name (no version/description)."""
        writer = ConcreteComponentWriter(tmp_path)

        buffer = io.StringIO()
        writer.write_component_header(buffer, component_name="Streams")
        result = buffer.getvalue()

        assert "Streams" in result
        assert "Generated by pyiwfm" in result
        assert "Version:" not in result
        # The description line should not appear

    def test_write_component_header_with_version_only(self, tmp_path: Path) -> None:
        """Test write_component_header with version but no description."""
        writer = ConcreteComponentWriter(tmp_path)

        buffer = io.StringIO()
        writer.write_component_header(
            buffer, component_name="Lakes", version="2.0"
        )
        result = buffer.getvalue()

        assert "Lakes" in result
        assert "Version: 2.0" in result

    def test_write_component_header_with_description_only(
        self, tmp_path: Path
    ) -> None:
        """Test write_component_header with description but no version."""
        writer = ConcreteComponentWriter(tmp_path)

        buffer = io.StringIO()
        writer.write_component_header(
            buffer,
            component_name="GW",
            description="Groundwater component",
        )
        result = buffer.getvalue()

        assert "GW" in result
        assert "Groundwater component" in result
        assert "Version:" not in result

    def test_write_file_reference_empty_string_path(self, tmp_path: Path) -> None:
        """Test write_file_reference with empty string path."""
        writer = ConcreteComponentWriter(tmp_path)

        buffer = io.StringIO()
        writer.write_file_reference(
            buffer, ref_path="", description="Empty path"
        )
        result = buffer.getvalue()

        assert "Empty path" in result

    def test_write_file_reference_whitespace_path(self, tmp_path: Path) -> None:
        """Test write_file_reference with whitespace-only path."""
        writer = ConcreteComponentWriter(tmp_path)

        buffer = io.StringIO()
        writer.write_file_reference(
            buffer, ref_path="   ", description="Whitespace"
        )
        result = buffer.getvalue()

        # Whitespace-only path should be treated as empty
        assert "Whitespace" in result

    def test_write_file_reference_backslash_conversion(
        self, tmp_path: Path
    ) -> None:
        """Test write_file_reference converts backslashes to forward slashes."""
        writer = ConcreteComponentWriter(tmp_path)

        buffer = io.StringIO()
        writer.write_file_reference(
            buffer,
            ref_path="path\\to\\file.dat",
            description="Windows path",
        )
        result = buffer.getvalue()

        assert "path/to/file.dat" in result
        assert "\\" not in result

    def test_write_file_reference_no_description(self, tmp_path: Path) -> None:
        """Test write_file_reference without description."""
        writer = ConcreteComponentWriter(tmp_path)

        buffer = io.StringIO()
        writer.write_file_reference(buffer, ref_path="some/file.dat")
        result = buffer.getvalue()

        assert "some/file.dat" in result
        assert "/" not in result.split("some/file.dat")[1] or result.strip().endswith(
            "some/file.dat"
        )

    def test_write_file_reference_none_no_description(
        self, tmp_path: Path
    ) -> None:
        """Test write_file_reference with None path and no description."""
        writer = ConcreteComponentWriter(tmp_path)

        buffer = io.StringIO()
        writer.write_file_reference(buffer, ref_path=None)
        result = buffer.getvalue()

        # Should just write an empty line (no description separator)
        assert result == "\n"

    def test_write_value_line_string(self, tmp_path: Path) -> None:
        """Test write_value_line with a string value."""
        writer = ConcreteComponentWriter(tmp_path)

        buffer = io.StringIO()
        writer.write_value_line(
            buffer, "filename.dat", description="Input file"
        )
        result = buffer.getvalue()

        assert "filename.dat" in result
        assert "Input file" in result

    def test_write_value_line_custom_width(self, tmp_path: Path) -> None:
        """Test write_value_line with custom width."""
        writer = ConcreteComponentWriter(tmp_path)

        buffer = io.StringIO()
        writer.write_value_line(buffer, 42, description="Value", width=30)
        result = buffer.getvalue()

        assert "42" in result
        assert "Value" in result

    def test_format_property(self, tmp_path: Path) -> None:
        """Test the format property returns 'iwfm_component'."""
        writer = ConcreteComponentWriter(tmp_path)
        assert writer.format == "iwfm_component"

    def test_initialization_with_all_params(self, tmp_path: Path) -> None:
        """Test initialization with all optional parameters."""
        engine = TemplateEngine()
        ts_config = TimeSeriesOutputConfig(format=OutputFormat.BOTH)
        writer = ConcreteComponentWriter(
            tmp_path, ts_config=ts_config, template_engine=engine
        )

        assert writer._engine is engine
        assert writer.ts_config is ts_config
        assert writer.output_dir == tmp_path


# =============================================================================
# Additional Tests - IWFMModelWriter
# =============================================================================


class ConcreteModelWriter:
    """
    Concrete implementation for testing IWFMModelWriter.

    We avoid importing IWFMModel by using a mock and directly
    subclassing IWFMModelWriter.
    """
    pass


class TestIWFMModelWriter:
    """Tests for IWFMModelWriter abstract base class."""

    def _make_concrete_class(self):
        """Create a concrete subclass of IWFMModelWriter for testing."""
        from pyiwfm.io.writer_base import IWFMModelWriter

        class ConcreteWriter(IWFMModelWriter):
            def write_preprocessor(self):
                return {"main": self.output_dir / "preprocessor.in"}

            def write_simulation(self):
                return {"main": self.output_dir / "simulation.in"}

        return ConcreteWriter

    def test_initialization(self, tmp_path: Path) -> None:
        """Test IWFMModelWriter initialization."""
        from unittest.mock import MagicMock

        ConcreteWriter = self._make_concrete_class()
        model = MagicMock()
        writer = ConcreteWriter(model, tmp_path)

        assert writer.model is model
        assert writer.output_dir == tmp_path
        assert writer.ts_format == OutputFormat.TEXT
        assert writer._engine is not None

    def test_initialization_with_custom_params(self, tmp_path: Path) -> None:
        """Test initialization with custom format and engine."""
        from unittest.mock import MagicMock

        ConcreteWriter = self._make_concrete_class()
        model = MagicMock()
        engine = TemplateEngine()
        writer = ConcreteWriter(
            model, tmp_path, ts_format=OutputFormat.DSS, template_engine=engine
        )

        assert writer.ts_format == OutputFormat.DSS
        assert writer._engine is engine

    def test_initialization_with_string_path(self, tmp_path: Path) -> None:
        """Test initialization with string path."""
        from unittest.mock import MagicMock

        ConcreteWriter = self._make_concrete_class()
        model = MagicMock()
        writer = ConcreteWriter(model, str(tmp_path))

        assert writer.output_dir == tmp_path
        assert isinstance(writer.output_dir, Path)

    def test_write_all(self, tmp_path: Path) -> None:
        """Test write_all combines preprocessor and simulation results."""
        from unittest.mock import MagicMock

        ConcreteWriter = self._make_concrete_class()
        model = MagicMock()
        writer = ConcreteWriter(model, tmp_path)

        results = writer.write_all()

        assert "preprocessor_main" in results
        assert "simulation_main" in results
        assert results["preprocessor_main"] == tmp_path / "preprocessor.in"
        assert results["simulation_main"] == tmp_path / "simulation.in"

    def test_ensure_directories(self, tmp_path: Path) -> None:
        """Test _ensure_directories creates output directory."""
        from unittest.mock import MagicMock

        ConcreteWriter = self._make_concrete_class()
        model = MagicMock()
        nested_dir = tmp_path / "deep" / "nested" / "output"
        writer = ConcreteWriter(model, nested_dir)

        assert not nested_dir.exists()
        writer._ensure_directories()
        assert nested_dir.exists()

    def test_abstract_methods_required(self) -> None:
        """Test that IWFMModelWriter cannot be instantiated directly."""
        from unittest.mock import MagicMock
        from pyiwfm.io.writer_base import IWFMModelWriter

        model = MagicMock()
        with pytest.raises(TypeError):
            IWFMModelWriter(model, "/tmp")  # type: ignore[abstract]

    def test_abstract_methods_must_be_implemented(self) -> None:
        """Test that subclass must implement both abstract methods."""
        from unittest.mock import MagicMock
        from pyiwfm.io.writer_base import IWFMModelWriter

        class PartialWriter(IWFMModelWriter):
            def write_preprocessor(self):
                return {}

            # Missing write_simulation

        model = MagicMock()
        with pytest.raises(TypeError):
            PartialWriter(model, "/tmp")  # type: ignore[abstract]

    def test_write_all_multiple_files(self, tmp_path: Path) -> None:
        """Test write_all with multiple files from each component."""
        from unittest.mock import MagicMock
        from pyiwfm.io.writer_base import IWFMModelWriter

        class MultiWriter(IWFMModelWriter):
            def write_preprocessor(self):
                return {
                    "nodes": self.output_dir / "nodes.dat",
                    "elements": self.output_dir / "elements.dat",
                }

            def write_simulation(self):
                return {
                    "main": self.output_dir / "simulation.in",
                    "gw": self.output_dir / "gw.dat",
                }

        model = MagicMock()
        writer = MultiWriter(model, tmp_path)
        results = writer.write_all()

        assert len(results) == 4
        assert "preprocessor_nodes" in results
        assert "preprocessor_elements" in results
        assert "simulation_main" in results
        assert "simulation_gw" in results
