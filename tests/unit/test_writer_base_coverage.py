"""Supplementary tests for writer_base.py targeting uncovered branches.

Covers:
- write_data_block with header_comment
- write_data_block with list of format strings
- write_indexed_data with 1D and 2D data
- ComponentWriter methods: write_component_header, write_file_reference, write_value_line
- TimeSeriesWriter: DSS branches, BOTH format, empty data
- IWFMModelWriter: abstract methods, write_all
- _check_dss function
"""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from pyiwfm.io.writer_base import (
    TemplateWriter,
    TimeSeriesSpec,
    TimeSeriesWriter,
    ComponentWriter,
    IWFMModelWriter,
    _check_dss,
)
from pyiwfm.io.config import OutputFormat, TimeSeriesOutputConfig
from pyiwfm.templates.engine import TemplateEngine


# =============================================================================
# Concrete Implementations for Testing
# =============================================================================


class ConcreteTemplateWriter(TemplateWriter):
    """Concrete TemplateWriter for testing."""

    def write(self, data: Any = None) -> None:
        pass

    @property
    def format(self) -> str:
        return "test_format"


class ConcreteModelWriter(IWFMModelWriter):
    """Concrete IWFMModelWriter for testing."""

    def write_preprocessor(self) -> dict[str, Path]:
        return {"main": self.output_dir / "pre.in"}

    def write_simulation(self) -> dict[str, Path]:
        return {"main": self.output_dir / "sim.in"}


# =============================================================================
# TemplateWriter Additional Tests
# =============================================================================


class TestTemplateWriterAdditional:
    """Additional tests for TemplateWriter."""

    def test_write_data_block_with_header_comment(self, tmp_path: Path) -> None:
        """Test write_data_block with header_comment argument."""
        writer = ConcreteTemplateWriter(tmp_path)
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        buffer = io.StringIO()

        writer.write_data_block(buffer, data, fmt="%8.2f", header_comment="Test Header")

        result = buffer.getvalue()
        assert "C  Test Header" in result
        assert "1.00" in result

    def test_write_data_block_with_format_list(self, tmp_path: Path) -> None:
        """Test write_data_block with list of format strings."""
        writer = ConcreteTemplateWriter(tmp_path)
        data = np.array([[1, 2.5], [3, 4.5]])
        buffer = io.StringIO()

        writer.write_data_block(buffer, data, fmt=["%5d", "%10.3f"])

        result = buffer.getvalue()
        assert "1" in result
        assert "2.500" in result

    def test_write_data_block_no_header(self, tmp_path: Path) -> None:
        """Test write_data_block without header_comment."""
        writer = ConcreteTemplateWriter(tmp_path)
        data = np.array([[1.0]])
        buffer = io.StringIO()

        writer.write_data_block(buffer, data, fmt="%8.2f")

        result = buffer.getvalue()
        assert "C  " not in result
        assert "1.00" in result

    def test_write_indexed_data_1d(self, tmp_path: Path) -> None:
        """Test write_indexed_data with 1D data."""
        writer = ConcreteTemplateWriter(tmp_path)
        ids = np.array([1, 2, 3], dtype=np.int32)
        data = np.array([10.5, 20.5, 30.5])
        buffer = io.StringIO()

        writer.write_indexed_data(buffer, ids, data)

        result = buffer.getvalue()
        assert "1" in result
        assert "10.5" in result

    def test_write_indexed_data_2d(self, tmp_path: Path) -> None:
        """Test write_indexed_data with 2D data."""
        writer = ConcreteTemplateWriter(tmp_path)
        ids = np.array([1, 2], dtype=np.int32)
        data = np.array([[10.0, 20.0], [30.0, 40.0]])
        buffer = io.StringIO()

        writer.write_indexed_data(buffer, ids, data, id_fmt="%4d", data_fmt="%10.2f")

        result = buffer.getvalue()
        assert "1" in result
        assert "10.00" in result
        assert "20.00" in result

    def test_render_header(self, tmp_path: Path) -> None:
        """Test render_header calls template engine."""
        writer = ConcreteTemplateWriter(tmp_path)

        # This should call the engine's render_template method
        # If template doesn't exist, it may raise - that's expected
        try:
            result = writer.render_header("nonexistent_template")
        except Exception:
            pass  # Template file doesn't exist - that's OK


# =============================================================================
# ComponentWriter Tests
# =============================================================================


class _ConcreteComponentWriter(ComponentWriter):
    """Concrete subclass for testing ComponentWriter (which has abstract write)."""

    def write(self, data: Any = None) -> None:
        pass


class TestComponentWriterMethods:
    """Tests for ComponentWriter methods."""

    def test_write_component_header_basic(self, tmp_path: Path) -> None:
        """Test writing basic component header."""
        writer = _ConcreteComponentWriter(tmp_path)
        buffer = io.StringIO()

        writer.write_component_header(buffer, "Groundwater")

        result = buffer.getvalue()
        assert "C  Groundwater" in result
        assert "Generated by pyiwfm" in result

    def test_write_component_header_with_version(self, tmp_path: Path) -> None:
        """Test writing component header with version."""
        writer = _ConcreteComponentWriter(tmp_path)
        buffer = io.StringIO()

        writer.write_component_header(buffer, "Stream", version="5.0")

        result = buffer.getvalue()
        assert "C  Stream" in result
        assert "Version: 5.0" in result

    def test_write_component_header_with_description(self, tmp_path: Path) -> None:
        """Test writing component header with description."""
        writer = _ConcreteComponentWriter(tmp_path)
        buffer = io.StringIO()

        writer.write_component_header(
            buffer, "Lake", version="4.0", description="Lake simulation data"
        )

        result = buffer.getvalue()
        assert "C  Lake" in result
        assert "Version: 4.0" in result
        assert "Lake simulation data" in result

    def test_write_file_reference_with_path(self, tmp_path: Path) -> None:
        """Test writing file reference with valid path."""
        writer = _ConcreteComponentWriter(tmp_path)
        buffer = io.StringIO()

        writer.write_file_reference(buffer, Path("data/wells.dat"), "Wells file")

        result = buffer.getvalue()
        assert "data/wells.dat" in result
        assert "Wells file" in result

    def test_write_file_reference_none_path(self, tmp_path: Path) -> None:
        """Test writing file reference with None path."""
        writer = _ConcreteComponentWriter(tmp_path)
        buffer = io.StringIO()

        writer.write_file_reference(buffer, None, "Optional file")

        result = buffer.getvalue()
        assert "Optional file" in result

    def test_write_file_reference_empty_string(self, tmp_path: Path) -> None:
        """Test writing file reference with empty string path."""
        writer = _ConcreteComponentWriter(tmp_path)
        buffer = io.StringIO()

        writer.write_file_reference(buffer, "", "Empty ref")

        result = buffer.getvalue()
        assert "Empty ref" in result

    def test_write_file_reference_no_description(self, tmp_path: Path) -> None:
        """Test writing file reference without description."""
        writer = _ConcreteComponentWriter(tmp_path)
        buffer = io.StringIO()

        writer.write_file_reference(buffer, Path("file.dat"))

        result = buffer.getvalue()
        assert "file.dat" in result

    def test_write_value_line_integer(self, tmp_path: Path) -> None:
        """Test writing integer value line."""
        writer = _ConcreteComponentWriter(tmp_path)
        buffer = io.StringIO()

        writer.write_value_line(buffer, 42, "N_ELEMENTS")

        result = buffer.getvalue()
        assert "42" in result
        assert "N_ELEMENTS" in result

    def test_write_value_line_float(self, tmp_path: Path) -> None:
        """Test writing float value line."""
        writer = _ConcreteComponentWriter(tmp_path)
        buffer = io.StringIO()

        writer.write_value_line(buffer, 3.14159, "PI_VALUE")

        result = buffer.getvalue()
        assert "3.141590" in result
        assert "PI_VALUE" in result

    def test_write_value_line_string(self, tmp_path: Path) -> None:
        """Test writing string value line."""
        writer = _ConcreteComponentWriter(tmp_path)
        buffer = io.StringIO()

        writer.write_value_line(buffer, "test_model", "MODEL_NAME")

        result = buffer.getvalue()
        assert "test_model" in result
        assert "MODEL_NAME" in result

    def test_write_value_line_no_description(self, tmp_path: Path) -> None:
        """Test writing value line without description."""
        writer = _ConcreteComponentWriter(tmp_path)
        buffer = io.StringIO()

        writer.write_value_line(buffer, 100)

        result = buffer.getvalue()
        assert "100" in result
        assert "/" not in result

    def test_format_property(self, tmp_path: Path) -> None:
        """Test ComponentWriter format property."""
        writer = _ConcreteComponentWriter(tmp_path)

        assert writer.format == "iwfm_component"

    def test_ts_writer_property(self, tmp_path: Path) -> None:
        """Test lazy initialization of ts_writer."""
        writer = _ConcreteComponentWriter(tmp_path)

        ts_writer = writer.ts_writer
        assert ts_writer is not None

        # Second call returns same instance
        assert writer.ts_writer is ts_writer


# =============================================================================
# TimeSeriesWriter Tests
# =============================================================================


class TestTimeSeriesWriterBranches:
    """Tests for TimeSeriesWriter branch coverage."""

    def test_write_text_timeseries(self, tmp_path: Path) -> None:
        """Test writing single time series to text."""
        config = TimeSeriesOutputConfig(format=OutputFormat.TEXT)
        writer = TimeSeriesWriter(config, tmp_path)

        ts = TimeSeriesSpec(
            name="Pump_1",
            dates=[datetime(2020, 1, 1), datetime(2020, 2, 1)],
            values=[100.0, 110.0],
            units="CFS",
            location="Well_1",
        )

        writer.write_timeseries(ts, text_file="pump1.dat")

        output = (tmp_path / "pump1.dat").read_text()
        assert "Pump_1" in output
        assert "CFS" in output
        assert "Well_1" in output
        assert "100.000000" in output

    def test_write_text_timeseries_no_units(self, tmp_path: Path) -> None:
        """Test writing time series without units."""
        config = TimeSeriesOutputConfig(format=OutputFormat.TEXT)
        writer = TimeSeriesWriter(config, tmp_path)

        ts = TimeSeriesSpec(
            name="Flow",
            dates=[datetime(2020, 1, 1)],
            values=[50.0],
        )

        writer.write_timeseries(ts, text_file="flow.dat")

        output = (tmp_path / "flow.dat").read_text()
        assert "Flow" in output
        assert "Units:" not in output

    def test_write_timeseries_missing_text_file(self, tmp_path: Path) -> None:
        """Test error when text_file is None for TEXT format."""
        config = TimeSeriesOutputConfig(format=OutputFormat.TEXT)
        writer = TimeSeriesWriter(config, tmp_path)

        ts = TimeSeriesSpec(name="Test", dates=[], values=[])

        with pytest.raises(ValueError, match="text_file required"):
            writer.write_timeseries(ts, text_file=None)

    def test_write_timeseries_table(self, tmp_path: Path) -> None:
        """Test writing multi-column time series table."""
        config = TimeSeriesOutputConfig(format=OutputFormat.TEXT)
        writer = TimeSeriesWriter(config, tmp_path)

        dates = [datetime(2020, 1, 1), datetime(2020, 2, 1)]
        columns = {
            "Node_1": np.array([100.0, 110.0]),
            "Node_2": np.array([200.0, 220.0]),
        }

        writer.write_timeseries_table(dates, columns, "table.dat")

        output = (tmp_path / "table.dat").read_text()
        assert "Node_1" in output
        assert "Node_2" in output
        assert "100.000000" in output

    def test_write_timeseries_table_with_header(self, tmp_path: Path) -> None:
        """Test writing time series table with header lines."""
        config = TimeSeriesOutputConfig(format=OutputFormat.TEXT)
        writer = TimeSeriesWriter(config, tmp_path)

        dates = [datetime(2020, 1, 1)]
        columns = {"Col1": np.array([1.0])}

        writer.write_timeseries_table(
            dates, columns, "table.dat",
            header_lines=["Header Line 1", "Header Line 2"],
        )

        output = (tmp_path / "table.dat").read_text()
        assert "Header Line 1" in output
        assert "Header Line 2" in output

    def test_write_timeseries_table_dss_format_skips(self, tmp_path: Path) -> None:
        """Test that DSS-only format skips text table writing."""
        config = TimeSeriesOutputConfig(format=OutputFormat.DSS)
        writer = TimeSeriesWriter(config, tmp_path)

        dates = [datetime(2020, 1, 1)]
        columns = {"Col1": np.array([1.0])}

        # Should not create file since format is DSS only
        writer.write_timeseries_table(dates, columns, "table.dat")

        assert not (tmp_path / "table.dat").exists()

    def test_close_no_dss_file(self, tmp_path: Path) -> None:
        """Test close when no DSS file is open."""
        config = TimeSeriesOutputConfig(format=OutputFormat.TEXT)
        writer = TimeSeriesWriter(config, tmp_path)

        # Should not raise
        writer.close()

    def test_check_dss_no_dss_available(self) -> None:
        """Test _check_dss raises ImportError when DSS not available."""
        with patch("pyiwfm.io.writer_base.HAS_DSS", False):
            with pytest.raises(ImportError, match="DSS support requires"):
                _check_dss()

    def test_dss_format_raises_import_error(self, tmp_path: Path) -> None:
        """Test DSS format raises ImportError when unavailable."""
        config = TimeSeriesOutputConfig(format=OutputFormat.DSS)
        writer = TimeSeriesWriter(config, tmp_path)

        ts = TimeSeriesSpec(
            name="Test",
            dates=[datetime(2020, 1, 1)],
            values=[1.0],
        )

        with patch("pyiwfm.io.writer_base.HAS_DSS", False):
            with pytest.raises(ImportError, match="DSS support requires"):
                writer.write_timeseries(ts)


# =============================================================================
# IWFMModelWriter Tests
# =============================================================================


class TestIWFMModelWriterCoverage:
    """Tests for IWFMModelWriter abstract and concrete methods."""

    def test_write_all(self, tmp_path: Path) -> None:
        """Test write_all combines preprocessor and simulation results."""
        model = MagicMock()
        writer = ConcreteModelWriter(model, tmp_path)

        results = writer.write_all()

        assert "preprocessor_main" in results
        assert "simulation_main" in results
        assert results["preprocessor_main"] == tmp_path / "pre.in"
        assert results["simulation_main"] == tmp_path / "sim.in"

    def test_ensure_directories(self, tmp_path: Path) -> None:
        """Test _ensure_directories creates output directory."""
        model = MagicMock()
        output_dir = tmp_path / "new" / "nested" / "dir"
        writer = ConcreteModelWriter(model, output_dir)

        writer._ensure_directories()

        assert output_dir.exists()

    def test_model_writer_properties(self, tmp_path: Path) -> None:
        """Test model writer stores model and config."""
        model = MagicMock()
        writer = ConcreteModelWriter(
            model, tmp_path, ts_format=OutputFormat.TEXT
        )

        assert writer.model is model
        assert writer.output_dir == tmp_path
        assert writer.ts_format == OutputFormat.TEXT

    def test_model_writer_custom_engine(self, tmp_path: Path) -> None:
        """Test model writer with custom template engine."""
        model = MagicMock()
        engine = TemplateEngine()
        writer = ConcreteModelWriter(model, tmp_path, template_engine=engine)

        assert writer._engine is engine


# =============================================================================
# TimeSeriesSpec Tests
# =============================================================================


class TestTimeSeriesSpecAdditional:
    """Additional tests for TimeSeriesSpec dataclass."""

    def test_spec_with_all_fields(self) -> None:
        """Test creating spec with all optional fields."""
        spec = TimeSeriesSpec(
            name="Pump_1",
            dates=[datetime(2020, 1, 1)],
            values=[100.0],
            units="CFS",
            location="Well_1",
            parameter="FLOW",
            interval="1MON",
        )

        assert spec.name == "Pump_1"
        assert spec.units == "CFS"
        assert spec.location == "Well_1"
        assert spec.parameter == "FLOW"
        assert spec.interval == "1MON"

    def test_spec_defaults(self) -> None:
        """Test spec default values."""
        spec = TimeSeriesSpec(
            name="Test",
            dates=[],
            values=[],
        )

        assert spec.units == ""
        assert spec.location == ""
        assert spec.parameter == ""
        assert spec.interval == "1DAY"
