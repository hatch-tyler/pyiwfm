"""Unit tests for ASCII time series I/O.

Tests:
- format_iwfm_timestamp function
- parse_iwfm_timestamp function
- _is_comment_line function
- TimeSeriesFileConfig dataclass
- TimeSeriesWriter class
- TimeSeriesReader class
- Convenience functions
- Roundtrip read/write
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from pyiwfm.io.timeseries_ascii import (
    COMMENT_CHARS,
    IWFM_TIMESTAMP_FORMAT,
    IWFM_TIMESTAMP_LENGTH,
    TimeSeriesFileConfig,
    TimeSeriesReader,
    TimeSeriesWriter,
    format_iwfm_timestamp,
    parse_iwfm_timestamp,
    read_timeseries,
    write_timeseries,
    _is_comment_line,
)
from pyiwfm.core.timeseries import TimeSeries, TimeSeriesCollection
from pyiwfm.core.exceptions import FileFormatError


# =============================================================================
# Test format_iwfm_timestamp
# =============================================================================


class TestFormatIwfmTimestamp:
    """Tests for format_iwfm_timestamp function."""

    def test_basic_datetime(self) -> None:
        """Test formatting a basic datetime."""
        dt = datetime(2020, 1, 15, 10, 30, 0)
        result = format_iwfm_timestamp(dt)

        assert result == "01/15/2020_10:30"
        assert len(result) == IWFM_TIMESTAMP_LENGTH

    def test_midnight(self) -> None:
        """Test formatting midnight timestamp."""
        dt = datetime(2020, 12, 31, 0, 0, 0)
        result = format_iwfm_timestamp(dt)

        assert result.strip() == "12/30/2020_24:00"

    def test_numpy_datetime64(self) -> None:
        """Test formatting numpy datetime64."""
        dt = np.datetime64("2020-06-15T14:30:00")
        result = format_iwfm_timestamp(dt)

        assert "06/15/2020" in result
        assert "14:30" in result

    def test_padded_to_16_chars(self) -> None:
        """Test that result is exactly 16 characters."""
        dt = datetime(2020, 1, 1, 0, 0, 0)
        result = format_iwfm_timestamp(dt)

        assert len(result) == 16


# =============================================================================
# Test parse_iwfm_timestamp
# =============================================================================


class TestParseIwfmTimestamp:
    """Tests for parse_iwfm_timestamp function."""

    def test_underscore_separator(self) -> None:
        """Test parsing timestamp with underscore separator."""
        ts_str = "01/15/2020_10:30"
        result = parse_iwfm_timestamp(ts_str)

        assert result.year == 2020
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 10
        assert result.minute == 30
        assert result.second == 0

    def test_space_separator(self) -> None:
        """Test parsing timestamp with space separator."""
        ts_str = "12/31/2020 23:59"
        result = parse_iwfm_timestamp(ts_str)

        assert result.year == 2020
        assert result.month == 12
        assert result.day == 31
        assert result.hour == 23
        assert result.minute == 59
        assert result.second == 0

    def test_with_whitespace(self) -> None:
        """Test parsing timestamp with leading/trailing whitespace."""
        ts_str = "  01/01/2020_00:00  "
        result = parse_iwfm_timestamp(ts_str)

        assert result == datetime(2020, 1, 1, 0, 0, 0)

    def test_invalid_format_raises_error(self) -> None:
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError):
            parse_iwfm_timestamp("2020-01-01 00:00:00")  # ISO format

    def test_invalid_date_raises_error(self) -> None:
        """Test that invalid date raises ValueError."""
        with pytest.raises(ValueError):
            parse_iwfm_timestamp("13/45/2020_00:00")  # Invalid month/day


# =============================================================================
# Test _is_comment_line
# =============================================================================


class TestIsCommentLine:
    """Tests for _is_comment_line function."""

    def test_uppercase_c_comment(self) -> None:
        """Test uppercase C comment."""
        assert _is_comment_line("C This is a comment") is True

    def test_lowercase_c_comment(self) -> None:
        """Test lowercase c comment."""
        assert _is_comment_line("c comment line") is True

    def test_asterisk_comment(self) -> None:
        """Test asterisk comment."""
        assert _is_comment_line("* Comment with asterisk") is True

    def test_hash_not_comment(self) -> None:
        """Hash is not a comment character."""
        assert _is_comment_line("# Comment with hash") is False

    def test_empty_line(self) -> None:
        """Test empty line is treated as comment."""
        assert _is_comment_line("") is True
        assert _is_comment_line("   ") is True

    def test_non_comment_line(self) -> None:
        """Test non-comment line."""
        assert _is_comment_line("10  value") is False
        assert _is_comment_line("01/01/2020_00:00:00  1.0") is False


# =============================================================================
# Test TimeSeriesFileConfig
# =============================================================================


class TestTimeSeriesFileConfig:
    """Tests for TimeSeriesFileConfig dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic config creation."""
        config = TimeSeriesFileConfig(
            n_columns=3,
            column_ids=[1, 2, 3],
        )

        assert config.n_columns == 3
        assert config.column_ids == [1, 2, 3]
        assert config.units == ""
        assert config.factor == 1.0
        assert config.header_lines is None

    def test_full_creation(self) -> None:
        """Test config with all fields."""
        config = TimeSeriesFileConfig(
            n_columns=2,
            column_ids=["well1", "well2"],
            units="TAF",
            factor=2.0,
            header_lines=["C Header 1", "C Header 2"],
        )

        assert config.n_columns == 2
        assert config.column_ids == ["well1", "well2"]
        assert config.units == "TAF"
        assert config.factor == 2.0
        assert len(config.header_lines) == 2


# =============================================================================
# Test TimeSeriesWriter
# =============================================================================


class TestTimeSeriesWriter:
    """Tests for TimeSeriesWriter class."""

    def test_initialization_defaults(self) -> None:
        """Test default initialization."""
        writer = TimeSeriesWriter()

        assert writer.value_format == "%14.6f"
        assert writer.timestamp_format == IWFM_TIMESTAMP_FORMAT

    def test_initialization_custom(self) -> None:
        """Test custom initialization."""
        writer = TimeSeriesWriter(
            value_format="%10.2f",
            timestamp_format="%Y-%m-%d",
        )

        assert writer.value_format == "%10.2f"
        assert writer.timestamp_format == "%Y-%m-%d"

    def test_write_basic(self, tmp_path: Path) -> None:
        """Test basic write operation."""
        output_path = tmp_path / "test.dat"
        times = [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)]
        values = np.array([[1.0], [2.0], [3.0]])

        writer = TimeSeriesWriter()
        writer.write(output_path, times, values)

        assert output_path.exists()
        content = output_path.read_text()
        assert "NDATA" in content
        assert "FACTOR" in content
        assert "01/01/2020" in content

    def test_write_multiple_columns(self, tmp_path: Path) -> None:
        """Test write with multiple columns."""
        output_path = tmp_path / "test.dat"
        times = [datetime(2020, 1, 1), datetime(2020, 1, 2)]
        values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        writer = TimeSeriesWriter()
        writer.write(output_path, times, values, column_ids=["a", "b", "c"])

        content = output_path.read_text()
        assert "3" in content  # n_columns
        assert "a  b  c" in content  # column IDs

    def test_write_1d_values(self, tmp_path: Path) -> None:
        """Test write with 1D values array."""
        output_path = tmp_path / "test.dat"
        times = [datetime(2020, 1, 1), datetime(2020, 1, 2)]
        values = np.array([1.0, 2.0])  # 1D array

        writer = TimeSeriesWriter()
        writer.write(output_path, times, values)

        content = output_path.read_text()
        assert "1" in content.split("\n")[5]  # n_columns = 1

    def test_write_with_header(self, tmp_path: Path) -> None:
        """Test write with custom header."""
        output_path = tmp_path / "test.dat"
        times = [datetime(2020, 1, 1)]
        values = np.array([[1.0]])

        writer = TimeSeriesWriter()
        writer.write(output_path, times, values, header="Custom header\nMulti-line")

        content = output_path.read_text()
        assert "C  Custom header" in content
        assert "C  Multi-line" in content

    def test_write_with_units(self, tmp_path: Path) -> None:
        """Test write with units."""
        output_path = tmp_path / "test.dat"
        times = [datetime(2020, 1, 1)]
        values = np.array([[1.0]])

        writer = TimeSeriesWriter()
        writer.write(output_path, times, values, units="TAF")

        content = output_path.read_text()
        assert "Units: TAF" in content

    def test_write_with_factor(self, tmp_path: Path) -> None:
        """Test write with factor."""
        output_path = tmp_path / "test.dat"
        times = [datetime(2020, 1, 1)]
        values = np.array([[1.0]])

        writer = TimeSeriesWriter()
        writer.write(output_path, times, values, factor=2.5)

        content = output_path.read_text()
        assert "2.5" in content

    def test_write_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Test that write creates parent directories."""
        output_path = tmp_path / "a" / "b" / "c" / "test.dat"
        times = [datetime(2020, 1, 1)]
        values = np.array([[1.0]])

        writer = TimeSeriesWriter()
        writer.write(output_path, times, values)

        assert output_path.exists()

    def test_write_times_values_mismatch_raises_error(self, tmp_path: Path) -> None:
        """Test that mismatched times and values raises error."""
        output_path = tmp_path / "test.dat"
        times = [datetime(2020, 1, 1), datetime(2020, 1, 2)]
        values = np.array([[1.0]])  # Only 1 row

        writer = TimeSeriesWriter()
        with pytest.raises(ValueError, match="doesn't match"):
            writer.write(output_path, times, values)

    def test_write_from_timeseries(self, tmp_path: Path) -> None:
        """Test write_from_timeseries method."""
        output_path = tmp_path / "test.dat"
        times = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[s]")
        values = np.array([1.0, 2.0])
        ts = TimeSeries(times=times, values=values, location="well1", units="TAF")

        writer = TimeSeriesWriter()
        writer.write_from_timeseries(output_path, ts)

        assert output_path.exists()
        content = output_path.read_text()
        assert "well1" in content

    def test_write_from_collection(self, tmp_path: Path) -> None:
        """Test write_from_collection method."""
        output_path = tmp_path / "test.dat"
        times = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[s]")

        ts1 = TimeSeries(times=times, values=np.array([1.0, 2.0]), location="loc1")
        ts2 = TimeSeries(times=times, values=np.array([3.0, 4.0]), location="loc2")

        collection = TimeSeriesCollection(variable="pumping")
        collection.add(ts1)
        collection.add(ts2)

        writer = TimeSeriesWriter()
        writer.write_from_collection(output_path, collection)

        assert output_path.exists()
        content = output_path.read_text()
        assert "2" in content  # 2 columns

    def test_write_from_empty_collection_raises_error(self, tmp_path: Path) -> None:
        """Test that empty collection raises error."""
        output_path = tmp_path / "test.dat"
        collection = TimeSeriesCollection(variable="empty")

        writer = TimeSeriesWriter()
        with pytest.raises(ValueError, match="Empty collection"):
            writer.write_from_collection(output_path, collection)


# =============================================================================
# Test TimeSeriesReader
# =============================================================================


class TestTimeSeriesReader:
    """Tests for TimeSeriesReader class."""

    def test_read_basic(self, tmp_path: Path) -> None:
        """Test basic read operation."""
        input_path = tmp_path / "test.dat"
        input_path.write_text(
            """C Header comment
2                              / NDATA
1.0                            / FACTOR
12/31/2019_24:00  1.0  2.0
01/01/2020_24:00  3.0  4.0
"""
        )

        reader = TimeSeriesReader()
        times, values, config = reader.read(input_path)

        assert len(times) == 2
        assert times[0] == datetime(2020, 1, 1, 0, 0, 0)
        assert times[1] == datetime(2020, 1, 2, 0, 0, 0)
        assert values.shape == (2, 2)
        assert values[0, 0] == 1.0
        assert values[0, 1] == 2.0
        assert config.n_columns == 2

    def test_read_with_factor(self, tmp_path: Path) -> None:
        """Test read applies factor."""
        input_path = tmp_path / "test.dat"
        input_path.write_text(
            """C Header
1                              / NDATA
2.0                            / FACTOR
12/31/2019_24:00  10.0
"""
        )

        reader = TimeSeriesReader()
        times, values, config = reader.read(input_path)

        # Factor should be applied
        assert values[0, 0] == 20.0
        assert config.factor == 2.0

    def test_read_with_comments(self, tmp_path: Path) -> None:
        """Test read with various comment styles."""
        input_path = tmp_path / "test.dat"
        input_path.write_text(
            """C Comment with C
c Comment with lowercase c
* Comment with asterisk
1                              / NDATA
1.0                            / FACTOR
C Mid-file comment
12/31/2019_24:00  1.0
* Another mid-file comment
01/01/2020_24:00  2.0
"""
        )

        reader = TimeSeriesReader()
        times, values, config = reader.read(input_path)

        assert len(times) == 2
        assert len(config.header_lines) >= 3

    def test_read_invalid_ndata_raises_error(self, tmp_path: Path) -> None:
        """Test that invalid NDATA raises error."""
        input_path = tmp_path / "test.dat"
        input_path.write_text(
            """C Header
abc                            / NDATA
"""
        )

        reader = TimeSeriesReader()
        with pytest.raises(FileFormatError, match="Invalid NDATA"):
            reader.read(input_path)

    def test_read_invalid_factor_raises_error(self, tmp_path: Path) -> None:
        """Test that invalid FACTOR raises error."""
        input_path = tmp_path / "test.dat"
        input_path.write_text(
            """C Header
1                              / NDATA
xyz                            / FACTOR
"""
        )

        reader = TimeSeriesReader()
        with pytest.raises(FileFormatError, match="Invalid FACTOR"):
            reader.read(input_path)

    def test_read_invalid_data_line_raises_error(self, tmp_path: Path) -> None:
        """Test that invalid data line after valid data raises error."""
        input_path = tmp_path / "test.dat"
        input_path.write_text(
            """C Header
1                              / NDATA
1.0                            / FACTOR
12/31/2019_24:00  1.0
not-a-valid-timestamp  1.0
"""
        )

        reader = TimeSeriesReader()
        with pytest.raises(FileFormatError, match="Invalid data line"):
            reader.read(input_path)

    def test_read_skips_extra_header_lines(self, tmp_path: Path) -> None:
        """Test that extra header lines between FACTOR and data are skipped."""
        input_path = tmp_path / "test.dat"
        input_path.write_text(
            """C Header
2                              / NDATA
1.0                            / FACTOR
1                              / NSPRN
0                              / NFQRN
                               / DSSFL
12/31/2019_24:00  1.0  2.0
01/01/2020_24:00  3.0  4.0
"""
        )

        reader = TimeSeriesReader()
        times, values, config = reader.read(input_path)

        assert len(times) == 2
        assert config.n_columns == 2
        assert values.shape == (2, 2)

    def test_read_to_timeseries(self, tmp_path: Path) -> None:
        """Test read_to_timeseries method."""
        input_path = tmp_path / "test.dat"
        input_path.write_text(
            """C Header
1                              / NDATA
1.0                            / FACTOR
12/31/2019_24:00  1.0
01/01/2020_24:00  2.0
"""
        )

        reader = TimeSeriesReader()
        ts = reader.read_to_timeseries(input_path, name="test", location="loc1")

        assert ts.name == "test"
        assert ts.location == "loc1"
        assert ts.n_times == 2

    def test_read_to_collection(self, tmp_path: Path) -> None:
        """Test read_to_collection method."""
        input_path = tmp_path / "test.dat"
        input_path.write_text(
            """C Header
2                              / NDATA
1.0                            / FACTOR
12/31/2019_24:00  1.0  2.0
01/01/2020_24:00  3.0  4.0
"""
        )

        reader = TimeSeriesReader()
        collection = reader.read_to_collection(
            input_path, column_ids=["col1", "col2"], variable="test_var"
        )

        assert collection.variable == "test_var"
        assert len(collection) == 2
        assert "col1" in collection.locations
        assert "col2" in collection.locations


# =============================================================================
# Test Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_write_timeseries_function(self, tmp_path: Path) -> None:
        """Test write_timeseries function."""
        output_path = tmp_path / "test.dat"
        times = [datetime(2020, 1, 1), datetime(2020, 1, 2)]
        values = np.array([[1.0], [2.0]])

        write_timeseries(output_path, times, values)

        assert output_path.exists()

    def test_read_timeseries_function(self, tmp_path: Path) -> None:
        """Test read_timeseries function."""
        input_path = tmp_path / "test.dat"
        input_path.write_text(
            """C Header
1                              / NDATA
1.0                            / FACTOR
12/31/2019_24:00  5.0
"""
        )

        times, values, config = read_timeseries(input_path)

        assert len(times) == 1
        assert values[0, 0] == 5.0


# =============================================================================
# Test Roundtrip
# =============================================================================


class TestRoundtrip:
    """Tests for read/write roundtrip."""

    def test_roundtrip_single_column(self, tmp_path: Path) -> None:
        """Test roundtrip with single column."""
        filepath = tmp_path / "test.dat"
        times = [
            datetime(2020, 1, 1, 0, 0, 0),
            datetime(2020, 1, 2, 0, 0, 0),
            datetime(2020, 1, 3, 0, 0, 0),
        ]
        values = np.array([[10.5], [20.3], [30.7]])

        writer = TimeSeriesWriter()
        writer.write(filepath, times, values)

        reader = TimeSeriesReader()
        read_times, read_values, config = reader.read(filepath)

        assert len(read_times) == 3
        assert read_times[0] == times[0]
        assert read_times[1] == times[1]
        assert read_times[2] == times[2]
        np.testing.assert_array_almost_equal(read_values, values)

    def test_roundtrip_multiple_columns(self, tmp_path: Path) -> None:
        """Test roundtrip with multiple columns."""
        filepath = tmp_path / "test.dat"
        times = [datetime(2020, 1, 1), datetime(2020, 1, 2)]
        values = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])

        writer = TimeSeriesWriter()
        writer.write(filepath, times, values, column_ids=[1, 2, 3])

        reader = TimeSeriesReader()
        read_times, read_values, config = reader.read(filepath)

        assert len(read_times) == 2
        np.testing.assert_array_almost_equal(read_values, values)
        assert config.n_columns == 3

    def test_roundtrip_timeseries_object(self, tmp_path: Path) -> None:
        """Test roundtrip with TimeSeries object."""
        filepath = tmp_path / "test.dat"
        times = np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[s]")
        values = np.array([100.0, 200.0, 300.0])
        ts = TimeSeries(times=times, values=values, name="test", location="loc1")

        writer = TimeSeriesWriter()
        writer.write_from_timeseries(filepath, ts)

        reader = TimeSeriesReader()
        read_ts = reader.read_to_timeseries(filepath)

        assert read_ts.n_times == 3
        np.testing.assert_array_almost_equal(read_ts.values.flatten(), values)

    def test_roundtrip_collection(self, tmp_path: Path) -> None:
        """Test roundtrip with TimeSeriesCollection."""
        filepath = tmp_path / "test.dat"
        times = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[s]")

        ts1 = TimeSeries(times=times, values=np.array([1.0, 2.0]), location="A")
        ts2 = TimeSeries(times=times, values=np.array([3.0, 4.0]), location="B")

        collection = TimeSeriesCollection(variable="test")
        collection.add(ts1)
        collection.add(ts2)

        writer = TimeSeriesWriter()
        writer.write_from_collection(filepath, collection)

        reader = TimeSeriesReader()
        read_collection = reader.read_to_collection(filepath, column_ids=["A", "B"])

        assert len(read_collection) == 2
        np.testing.assert_array_almost_equal(
            read_collection["A"].values, np.array([1.0, 2.0])
        )
        np.testing.assert_array_almost_equal(
            read_collection["B"].values, np.array([3.0, 4.0])
        )


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_large_values(self, tmp_path: Path) -> None:
        """Test handling of large values."""
        filepath = tmp_path / "test.dat"
        times = [datetime(2020, 1, 1)]
        values = np.array([[1e15]])

        writer = TimeSeriesWriter()
        writer.write(filepath, times, values)

        reader = TimeSeriesReader()
        _, read_values, _ = reader.read(filepath)

        np.testing.assert_array_almost_equal(read_values, values, decimal=5)

    def test_small_values(self, tmp_path: Path) -> None:
        """Test handling of small values."""
        filepath = tmp_path / "test.dat"
        times = [datetime(2020, 1, 1)]
        values = np.array([[1e-10]])

        writer = TimeSeriesWriter()
        writer.write(filepath, times, values)

        reader = TimeSeriesReader()
        _, read_values, _ = reader.read(filepath)

        # Small values may lose some precision in ASCII format
        assert read_values[0, 0] < 1e-5

    def test_negative_values(self, tmp_path: Path) -> None:
        """Test handling of negative values."""
        filepath = tmp_path / "test.dat"
        times = [datetime(2020, 1, 1)]
        values = np.array([[-123.456]])

        writer = TimeSeriesWriter()
        writer.write(filepath, times, values)

        reader = TimeSeriesReader()
        _, read_values, _ = reader.read(filepath)

        np.testing.assert_array_almost_equal(read_values, values)

    def test_many_columns(self, tmp_path: Path) -> None:
        """Test handling of many columns."""
        filepath = tmp_path / "test.dat"
        times = [datetime(2020, 1, 1)]
        values = np.array([[i * 1.0 for i in range(100)]])  # 100 columns

        writer = TimeSeriesWriter()
        writer.write(filepath, times, values)

        reader = TimeSeriesReader()
        _, read_values, config = reader.read(filepath)

        assert config.n_columns == 100
        np.testing.assert_array_almost_equal(read_values, values)

    def test_many_timesteps(self, tmp_path: Path) -> None:
        """Test handling of many timesteps."""
        filepath = tmp_path / "test.dat"
        times = [datetime(2020, 1, 1) + i * np.timedelta64(1, "D") for i in range(365)]
        times = [datetime(2020, 1, 1, 0, 0, 0)]
        for i in range(1, 365):
            times.append(datetime(2020, 1, 1, 0, 0, 0) + (datetime(2020, 1, 2) - datetime(2020, 1, 1)) * i)
        values = np.arange(365).reshape(-1, 1).astype(float)

        writer = TimeSeriesWriter()
        writer.write(filepath, times, values)

        reader = TimeSeriesReader()
        read_times, read_values, _ = reader.read(filepath)

        assert len(read_times) == 365
        np.testing.assert_array_almost_equal(read_values, values)
