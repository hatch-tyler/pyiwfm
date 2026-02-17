"""Unit tests for template filters.

Tests all Jinja2 filters in pyiwfm.templates.filters:
- Number formatting (fortran_float, fortran_int, fortran_scientific)
- IWFM formatting (iwfm_comment, iwfm_value, iwfm_path, etc.)
- Time/date formatting (iwfm_timestamp, iwfm_date, iwfm_time_unit)
- DSS formatting (dss_pathname, dss_date_part, dss_interval)
- Array formatting (iwfm_array_row, iwfm_data_row)
- String formatting (pad_right, pad_left, truncate)
- Time series references (timeseries_ref, dss_timeseries_ref)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from jinja2 import Environment

from pyiwfm.templates.filters import (
    dss_date_part,
    dss_interval,
    # DSS formatting
    dss_pathname,
    dss_timeseries_ref,
    # Number formatting
    fortran_float,
    fortran_int,
    fortran_scientific,
    # Array formatting
    iwfm_array_row,
    iwfm_blank_or_path,
    # IWFM formatting
    iwfm_comment,
    iwfm_data_row,
    iwfm_date,
    iwfm_path,
    iwfm_time_unit,
    # Time formatting
    iwfm_timestamp,
    iwfm_value,
    pad_left,
    # String formatting
    pad_right,
    # Registration
    register_all_filters,
    # Time series references
    timeseries_ref,
    truncate,
)

# =============================================================================
# Test Number Formatting
# =============================================================================


class TestFortranFloat:
    """Tests for fortran_float filter."""

    def test_basic_formatting(self) -> None:
        """Test basic float formatting."""
        result = fortran_float(123.456)
        assert "123.456" in result

    def test_width_and_decimals(self) -> None:
        """Test custom width and decimals."""
        result = fortran_float(1.5, width=10, decimals=2)
        assert len(result) == 10
        assert "1.50" in result

    def test_negative_number(self) -> None:
        """Test negative number formatting."""
        result = fortran_float(-123.456)
        assert "-123.456" in result

    def test_none_value(self) -> None:
        """Test None value returns spaces."""
        result = fortran_float(None)
        assert result == " " * 14  # Default width

    def test_nan_value(self) -> None:
        """Test NaN value returns spaces."""
        result = fortran_float(float("nan"))
        assert result == " " * 14

    def test_right_alignment(self) -> None:
        """Test that value is right-aligned."""
        result = fortran_float(1.0, width=14, decimals=6)
        assert result.strip() == "1.000000"
        assert result[0] == " "  # Padding on left


class TestFortranInt:
    """Tests for fortran_int filter."""

    def test_basic_formatting(self) -> None:
        """Test basic integer formatting."""
        result = fortran_int(42)
        assert "42" in result

    def test_width(self) -> None:
        """Test custom width."""
        result = fortran_int(5, width=5)
        assert len(result) == 5

    def test_negative_number(self) -> None:
        """Test negative integer formatting."""
        result = fortran_int(-100)
        assert "-100" in result

    def test_none_value(self) -> None:
        """Test None value returns spaces."""
        result = fortran_int(None)
        assert result == " " * 10  # Default width

    def test_right_alignment(self) -> None:
        """Test that value is right-aligned."""
        result = fortran_int(1, width=10)
        assert result.strip() == "1"
        assert result[0] == " "


class TestFortranScientific:
    """Tests for fortran_scientific filter."""

    def test_basic_formatting(self) -> None:
        """Test basic scientific notation."""
        result = fortran_scientific(1234.56)
        assert "E" in result

    def test_large_number(self) -> None:
        """Test large number in scientific notation."""
        result = fortran_scientific(1e10)
        assert "E+10" in result or "E+1" in result

    def test_small_number(self) -> None:
        """Test small number in scientific notation."""
        result = fortran_scientific(1e-10)
        assert "E-10" in result or "E-1" in result

    def test_none_value(self) -> None:
        """Test None value returns spaces."""
        result = fortran_scientific(None)
        assert result == " " * 14

    def test_nan_value(self) -> None:
        """Test NaN value returns spaces."""
        result = fortran_scientific(float("nan"))
        assert result == " " * 14


# =============================================================================
# Test IWFM Formatting
# =============================================================================


class TestIwfmComment:
    """Tests for iwfm_comment filter."""

    def test_default_prefix(self) -> None:
        """Test default C prefix."""
        result = iwfm_comment("Test comment")
        assert result == "C  Test comment"

    def test_custom_prefix(self) -> None:
        """Test custom prefix."""
        result = iwfm_comment("Test comment", prefix="*")
        assert result == "*  Test comment"


class TestIwfmValue:
    """Tests for iwfm_value filter."""

    def test_integer_value(self) -> None:
        """Test integer value formatting."""
        result = iwfm_value(42, "Answer")
        assert "42" in result
        assert "Answer" in result
        assert "/" in result

    def test_float_value(self) -> None:
        """Test float value formatting."""
        result = iwfm_value(3.14159, "Pi")
        assert "3.141590" in result
        assert "Pi" in result

    def test_string_value(self) -> None:
        """Test string value formatting."""
        result = iwfm_value("test.dat", "Filename")
        assert "test.dat" in result
        assert "Filename" in result

    def test_no_description(self) -> None:
        """Test value without description."""
        result = iwfm_value(100)
        assert "100" in result
        assert "/" not in result


class TestIwfmPath:
    """Tests for iwfm_path filter."""

    def test_forward_slashes(self) -> None:
        """Test backslash to forward slash conversion."""
        result = iwfm_path("C:\\Users\\test\\file.dat")
        assert "\\" not in result
        assert "/" in result

    def test_path_object(self) -> None:
        """Test Path object input."""
        result = iwfm_path(Path("dir/subdir/file.dat"))
        assert "dir/subdir/file.dat" in result

    def test_max_length_exceeded(self) -> None:
        """Test error on path too long."""
        long_path = "a" * 1001
        with pytest.raises(ValueError, match="exceeds maximum length"):
            iwfm_path(long_path, max_length=1000)

    def test_max_length_ok(self) -> None:
        """Test path within max length."""
        path = "a" * 100
        result = iwfm_path(path, max_length=100)
        assert result == path


class TestIwfmBlankOrPath:
    """Tests for iwfm_blank_or_path filter."""

    def test_none_returns_empty(self) -> None:
        """Test None returns empty string."""
        result = iwfm_blank_or_path(None)
        assert result == ""

    def test_empty_string_returns_empty(self) -> None:
        """Test empty string returns empty."""
        result = iwfm_blank_or_path("")
        assert result == ""

    def test_whitespace_returns_empty(self) -> None:
        """Test whitespace returns empty."""
        result = iwfm_blank_or_path("   ")
        assert result == ""

    def test_valid_path_formatted(self) -> None:
        """Test valid path is formatted."""
        result = iwfm_blank_or_path("dir/file.dat")
        assert result == "dir/file.dat"


# =============================================================================
# Test Time Formatting
# =============================================================================


class TestIwfmTimestamp:
    """Tests for iwfm_timestamp filter."""

    def test_datetime_formatting(self) -> None:
        """Test datetime formatting."""
        dt = datetime(2020, 5, 15, 12, 30)
        result = iwfm_timestamp(dt)
        assert result == "05/15/2020_12:30"

    def test_string_passthrough(self) -> None:
        """Test string is passed through."""
        result = iwfm_timestamp("09/30/2000_24:00")
        assert result == "09/30/2000_24:00"

    def test_none_returns_empty(self) -> None:
        """Test None returns empty string."""
        result = iwfm_timestamp(None)
        assert result == ""

    def test_numpy_datetime64(self) -> None:
        """Test numpy datetime64 conversion."""
        dt64 = np.datetime64("2020-05-15T12:30:00")
        result = iwfm_timestamp(dt64)
        assert "05/15/2020" in result


class TestIwfmDate:
    """Tests for iwfm_date filter."""

    def test_datetime_formatting(self) -> None:
        """Test datetime date formatting."""
        dt = datetime(2020, 5, 15, 12, 30)
        result = iwfm_date(dt)
        assert result == "05/15/2020"

    def test_string_passthrough(self) -> None:
        """Test string is passed through."""
        result = iwfm_date("09/30/2000")
        assert result == "09/30/2000"

    def test_none_returns_empty(self) -> None:
        """Test None returns empty string."""
        result = iwfm_date(None)
        assert result == ""


class TestIwfmTimeUnit:
    """Tests for iwfm_time_unit filter."""

    def test_minute(self) -> None:
        """Test minute conversion."""
        assert iwfm_time_unit("minute") == "1MIN"
        assert iwfm_time_unit("min") == "1MIN"

    def test_hour(self) -> None:
        """Test hour conversion."""
        assert iwfm_time_unit("hour") == "1HOUR"
        assert iwfm_time_unit("hr") == "1HOUR"

    def test_day(self) -> None:
        """Test day conversion."""
        assert iwfm_time_unit("day") == "1DAY"

    def test_week(self) -> None:
        """Test week conversion."""
        assert iwfm_time_unit("week") == "1WEEK"
        assert iwfm_time_unit("wk") == "1WEEK"

    def test_month(self) -> None:
        """Test month conversion."""
        assert iwfm_time_unit("month") == "1MON"
        assert iwfm_time_unit("mon") == "1MON"

    def test_year(self) -> None:
        """Test year conversion."""
        assert iwfm_time_unit("year") == "1YEAR"
        assert iwfm_time_unit("yr") == "1YEAR"

    def test_case_insensitive(self) -> None:
        """Test case insensitivity."""
        assert iwfm_time_unit("DAY") == "1DAY"
        assert iwfm_time_unit("Day") == "1DAY"

    def test_unknown_unit(self) -> None:
        """Test unknown unit is uppercased."""
        assert iwfm_time_unit("1day") == "1DAY"
        assert iwfm_time_unit("custom") == "CUSTOM"


# =============================================================================
# Test DSS Formatting
# =============================================================================


class TestDssPathname:
    """Tests for dss_pathname filter."""

    def test_all_parts(self) -> None:
        """Test pathname with all parts."""
        result = dss_pathname(
            a_part="PROJECT",
            b_part="LOCATION",
            c_part="FLOW",
            d_part="01JAN2020",
            e_part="1DAY",
            f_part="VERSION",
        )
        assert result == "/PROJECT/LOCATION/FLOW/01JAN2020/1DAY/VERSION/"

    def test_empty_parts(self) -> None:
        """Test pathname with empty parts."""
        result = dss_pathname(a_part="A", c_part="FLOW")
        assert result == "/A//FLOW////"

    def test_default_empty(self) -> None:
        """Test all defaults (empty pathname)."""
        result = dss_pathname()
        assert result == "///////"


class TestDssDatePart:
    """Tests for dss_date_part filter."""

    def test_date_range(self) -> None:
        """Test date range formatting."""
        start = datetime(2020, 1, 1)
        end = datetime(2020, 12, 31)
        result = dss_date_part(start, end)
        assert "01JAN2020" in result
        assert "31DEC2020" in result
        assert "-" in result


class TestDssInterval:
    """Tests for dss_interval filter."""

    def test_uppercase(self) -> None:
        """Test interval is uppercased."""
        assert dss_interval("1day") == "1DAY"
        assert dss_interval("1hour") == "1HOUR"

    def test_removes_spaces(self) -> None:
        """Test spaces are removed."""
        assert dss_interval("1 day") == "1DAY"


# =============================================================================
# Test Array Formatting
# =============================================================================


class TestIwfmArrayRow:
    """Tests for iwfm_array_row filter."""

    def test_list_values(self) -> None:
        """Test formatting list of values."""
        result = iwfm_array_row([1.0, 2.0, 3.0])
        assert "1.000000" in result
        assert "2.000000" in result
        assert "3.000000" in result

    def test_numpy_array(self) -> None:
        """Test formatting numpy array."""
        result = iwfm_array_row(np.array([1.0, 2.0]))
        assert "1.000000" in result
        assert "2.000000" in result

    def test_custom_format(self) -> None:
        """Test custom format string."""
        result = iwfm_array_row([1.0, 2.0], fmt="%8.2f")
        assert "1.00" in result
        assert "2.00" in result

    def test_with_prefix(self) -> None:
        """Test with row prefix."""
        result = iwfm_array_row([1.0, 2.0], prefix="  1")
        assert result.startswith("  1")

    def test_custom_separator(self) -> None:
        """Test custom separator."""
        result = iwfm_array_row([1.0, 2.0], sep=",")
        assert "," in result


class TestIwfmDataRow:
    """Tests for iwfm_data_row filter."""

    def test_basic_row(self) -> None:
        """Test basic data row formatting."""
        result = iwfm_data_row(1, [10.0, 20.0])
        assert "1" in result
        assert "10.000000" in result
        assert "20.000000" in result

    def test_custom_formats(self) -> None:
        """Test custom format strings."""
        result = iwfm_data_row(1, [10.0], int_fmt="%3d", float_fmt="%8.2f")
        assert "10.00" in result

    def test_numpy_array(self) -> None:
        """Test with numpy array."""
        result = iwfm_data_row(1, np.array([10.0, 20.0]))
        assert "10.000000" in result


# =============================================================================
# Test String Formatting
# =============================================================================


class TestPadRight:
    """Tests for pad_right filter."""

    def test_padding(self) -> None:
        """Test right padding."""
        result = pad_right("test", 10)
        assert result == "test      "
        assert len(result) == 10

    def test_custom_char(self) -> None:
        """Test custom padding character."""
        result = pad_right("test", 10, "-")
        assert result == "test------"

    def test_already_long(self) -> None:
        """Test text already at or above width."""
        result = pad_right("testing", 5)
        assert result == "testing"


class TestPadLeft:
    """Tests for pad_left filter."""

    def test_padding(self) -> None:
        """Test left padding."""
        result = pad_left("test", 10)
        assert result == "      test"
        assert len(result) == 10

    def test_custom_char(self) -> None:
        """Test custom padding character."""
        result = pad_left("test", 10, "0")
        assert result == "000000test"


class TestTruncate:
    """Tests for truncate filter."""

    def test_short_text(self) -> None:
        """Test text shorter than max length."""
        result = truncate("short", 10)
        assert result == "short"

    def test_truncation(self) -> None:
        """Test truncation with suffix."""
        result = truncate("this is a long text", 10)
        assert len(result) == 10
        assert result.endswith("...")

    def test_custom_suffix(self) -> None:
        """Test custom truncation suffix."""
        result = truncate("this is a long text", 10, suffix="..")
        assert result.endswith("..")


# =============================================================================
# Test Time Series References
# =============================================================================


class TestTimeseriesRef:
    """Tests for timeseries_ref filter."""

    def test_basic_reference(self) -> None:
        """Test basic time series reference."""
        result = timeseries_ref("path/to/file.dat", column=2, factor=1.0)
        assert "path/to/file.dat" in result
        assert "2" in result
        assert "1.000000" in result

    def test_path_object(self) -> None:
        """Test with Path object."""
        result = timeseries_ref(Path("dir/file.dat"))
        assert "dir/file.dat" in result


class TestDssTimeseriesRef:
    """Tests for dss_timeseries_ref filter."""

    def test_basic_reference(self) -> None:
        """Test basic DSS reference."""
        result = dss_timeseries_ref(
            dss_file="data.dss",
            pathname="/A/B/C/D/E/F/",
            factor=2.0,
        )
        assert "data.dss" in result
        assert "/A/B/C/D/E/F/" in result
        assert "2.000000" in result


# =============================================================================
# Test Filter Registration
# =============================================================================


class TestRegisterAllFilters:
    """Tests for register_all_filters function."""

    def test_registers_all_filters(self) -> None:
        """Test that all filters are registered."""
        env = Environment()
        register_all_filters(env)

        # Number formatting
        assert "fortran_float" in env.filters
        assert "fortran_int" in env.filters
        assert "fortran_scientific" in env.filters

        # IWFM formatting
        assert "iwfm_comment" in env.filters
        assert "iwfm_value" in env.filters
        assert "iwfm_path" in env.filters
        assert "iwfm_blank_or_path" in env.filters

        # Time formatting
        assert "iwfm_timestamp" in env.filters
        assert "iwfm_date" in env.filters
        assert "iwfm_time_unit" in env.filters

        # DSS formatting
        assert "dss_pathname" in env.filters
        assert "dss_date_part" in env.filters
        assert "dss_interval" in env.filters

        # Array formatting
        assert "iwfm_array_row" in env.filters
        assert "iwfm_data_row" in env.filters

        # String formatting
        assert "pad_right" in env.filters
        assert "pad_left" in env.filters
        assert "truncate" in env.filters

        # Time series references
        assert "timeseries_ref" in env.filters
        assert "dss_timeseries_ref" in env.filters

    def test_filters_callable_from_template(self) -> None:
        """Test that filters work in templates."""
        env = Environment()
        register_all_filters(env)

        # Test fortran_float in template
        template = env.from_string("{{ value | fortran_float(10, 2) }}")
        result = template.render(value=3.14)
        assert "3.14" in result

        # Test iwfm_comment in template
        template = env.from_string("{{ text | iwfm_comment }}")
        result = template.render(text="Test")
        assert result == "C  Test"

        # Test pad_right in template
        template = env.from_string("{{ text | pad_right(10) }}")
        result = template.render(text="test")
        assert len(result) == 10
