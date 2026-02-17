"""Unit tests for DSS pathname utilities.

Tests:
- DSSPathname dataclass
- DSSPathnameTemplate class
- format_dss_date function
- format_dss_date_range function
- parse_dss_date function
- interval_to_minutes function
- minutes_to_interval function
"""

from __future__ import annotations

from datetime import datetime

import pytest

from pyiwfm.io.dss.pathname import (
    INTERVAL_MAPPING,
    PARAMETER_CODES,
    VALID_INTERVALS,
    DSSPathname,
    DSSPathnameTemplate,
    format_dss_date,
    format_dss_date_range,
    interval_to_minutes,
    minutes_to_interval,
    parse_dss_date,
)

# =============================================================================
# Test DSSPathname
# =============================================================================


class TestDSSPathname:
    """Tests for DSSPathname dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic pathname creation."""
        path = DSSPathname(
            a_part="project",
            b_part="location",
            c_part="flow",
            d_part="",
            e_part="1day",
            f_part="obs",
        )

        # Parts should be uppercase
        assert path.a_part == "PROJECT"
        assert path.b_part == "LOCATION"
        assert path.c_part == "FLOW"
        assert path.e_part == "1DAY"
        assert path.f_part == "OBS"

    def test_str_representation(self) -> None:
        """Test string representation."""
        path = DSSPathname(
            a_part="PROJECT",
            b_part="LOC",
            c_part="FLOW",
            d_part="",
            e_part="1DAY",
            f_part="V1",
        )

        assert str(path) == "/PROJECT/LOC/FLOW//1DAY/V1/"

    def test_empty_parts(self) -> None:
        """Test pathname with empty parts."""
        path = DSSPathname()

        # 6 parts means 7 slashes: /A/B/C/D/E/F/
        assert str(path) == "///////"

    def test_from_string_basic(self) -> None:
        """Test parsing pathname from string."""
        path = DSSPathname.from_string("/PROJECT/LOC/FLOW//1DAY/V1/")

        assert path.a_part == "PROJECT"
        assert path.b_part == "LOC"
        assert path.c_part == "FLOW"
        assert path.d_part == ""
        assert path.e_part == "1DAY"
        assert path.f_part == "V1"

    def test_from_string_with_date(self) -> None:
        """Test parsing pathname with date range."""
        path = DSSPathname.from_string("/PROJ/LOC/HEAD/01JAN2020/1MON/SIM/")

        assert path.d_part == "01JAN2020"

    def test_from_string_lowercase(self) -> None:
        """Test parsing lowercase pathname (should convert to upper)."""
        path = DSSPathname.from_string("/project/loc/flow//1day/v1/")

        assert path.a_part == "PROJECT"
        assert path.c_part == "FLOW"

    def test_from_string_invalid_no_leading_slash(self) -> None:
        """Test parsing pathname without leading slash."""
        with pytest.raises(ValueError, match="Invalid DSS pathname format"):
            DSSPathname.from_string("PROJECT/LOC/FLOW//1DAY/V1/")

    def test_from_string_invalid_no_trailing_slash(self) -> None:
        """Test parsing pathname without trailing slash."""
        with pytest.raises(ValueError, match="Invalid DSS pathname format"):
            DSSPathname.from_string("/PROJECT/LOC/FLOW//1DAY/V1")

    def test_from_string_invalid_wrong_parts(self) -> None:
        """Test parsing pathname with wrong number of parts."""
        with pytest.raises(ValueError, match="exactly 6 parts"):
            DSSPathname.from_string("/PROJECT/LOC/FLOW/1DAY/V1/")  # Only 5 parts

    def test_build_method(self) -> None:
        """Test building pathname with descriptive names."""
        path = DSSPathname.build(
            project="MyProject",
            location="Stream01",
            parameter="flow",
            interval="daily",
            version="Baseline",
        )

        assert path.a_part == "MYPROJECT"
        assert path.b_part == "STREAM01"
        assert path.c_part == "FLOW"  # Converted from parameter code
        assert path.e_part == "1DAY"  # Converted from interval mapping
        assert path.f_part == "BASELINE"

    def test_build_method_head_parameter(self) -> None:
        """Test building with head parameter."""
        path = DSSPathname.build(parameter="groundwater_head")

        assert path.c_part == "GW-HEAD"

    def test_build_method_unknown_parameter(self) -> None:
        """Test building with unknown parameter (should use as-is)."""
        path = DSSPathname.build(parameter="custom_param")

        assert path.c_part == "CUSTOM_PARAM"

    def test_build_method_hourly_interval(self) -> None:
        """Test building with hourly interval."""
        path = DSSPathname.build(interval="hourly")

        assert path.e_part == "1HOUR"

    def test_with_location(self) -> None:
        """Test creating pathname with different location."""
        original = DSSPathname.from_string("/PROJ/LOC1/FLOW//1DAY/V1/")
        modified = original.with_location("LOC2")

        assert modified.b_part == "LOC2"
        assert modified.a_part == "PROJ"  # Other parts unchanged
        assert original.b_part == "LOC1"  # Original unchanged

    def test_with_parameter(self) -> None:
        """Test creating pathname with different parameter."""
        original = DSSPathname.from_string("/PROJ/LOC/FLOW//1DAY/V1/")
        modified = original.with_parameter("head")

        assert modified.c_part == "HEAD"
        assert original.c_part == "FLOW"

    def test_with_date_range(self) -> None:
        """Test creating pathname with different date range."""
        original = DSSPathname.from_string("/PROJ/LOC/FLOW//1DAY/V1/")
        modified = original.with_date_range("01JAN2020-31DEC2020")

        assert modified.d_part == "01JAN2020-31DEC2020"
        assert original.d_part == ""

    def test_with_version(self) -> None:
        """Test creating pathname with different version."""
        original = DSSPathname.from_string("/PROJ/LOC/FLOW//1DAY/V1/")
        modified = original.with_version("V2")

        assert modified.f_part == "V2"
        assert original.f_part == "V1"

    def test_matches_exact(self) -> None:
        """Test exact pathname matching."""
        path = DSSPathname.from_string("/PROJ/LOC/FLOW//1DAY/V1/")
        pattern = DSSPathname.from_string("/PROJ/LOC/FLOW//1DAY/V1/")

        assert path.matches(pattern)

    def test_matches_partial_pattern(self) -> None:
        """Test partial pattern matching."""
        path = DSSPathname.from_string("/PROJ/LOC/FLOW//1DAY/V1/")
        pattern = DSSPathname(a_part="PROJ", c_part="FLOW")  # Empty parts match anything

        assert path.matches(pattern)

    def test_matches_string_pattern(self) -> None:
        """Test matching with string pattern."""
        path = DSSPathname.from_string("/PROJ/LOC/FLOW//1DAY/V1/")

        # Pattern must have 6 parts: /A/B/C/D/E/F/
        assert path.matches("/////1DAY//")  # Only e_part specified

    def test_matches_no_match(self) -> None:
        """Test non-matching pathname."""
        path = DSSPathname.from_string("/PROJ/LOC/FLOW//1DAY/V1/")
        pattern = DSSPathname(a_part="OTHER")

        assert not path.matches(pattern)

    def test_is_regular_interval(self) -> None:
        """Test regular interval detection."""
        regular = DSSPathname(e_part="1DAY")
        assert regular.is_regular_interval is True
        assert regular.is_irregular_interval is False

    def test_is_irregular_interval(self) -> None:
        """Test irregular interval detection."""
        irregular = DSSPathname(e_part="IR-DAY")
        assert irregular.is_irregular_interval is True
        assert irregular.is_regular_interval is False


# =============================================================================
# Test DSSPathnameTemplate
# =============================================================================


class TestDSSPathnameTemplate:
    """Tests for DSSPathnameTemplate class."""

    def test_basic_creation(self) -> None:
        """Test basic template creation."""
        template = DSSPathnameTemplate(
            a_part="PROJECT",
            c_part="FLOW",
            e_part="1DAY",
            f_part="OBS",
        )

        assert template.a_part == "PROJECT"
        assert template.c_part == "FLOW"

    def test_make_pathname(self) -> None:
        """Test creating pathname from template."""
        template = DSSPathnameTemplate(
            a_part="PROJECT",
            c_part="FLOW",
            e_part="1DAY",
            f_part="OBS",
        )

        path = template.make_pathname(location="STREAM01")

        assert path.a_part == "PROJECT"
        assert path.b_part == "STREAM01"
        assert path.c_part == "FLOW"
        assert path.f_part == "OBS"

    def test_make_pathname_with_date_range(self) -> None:
        """Test creating pathname with date range."""
        template = DSSPathnameTemplate(a_part="PROJ", c_part="HEAD")

        path = template.make_pathname(location="WELL01", date_range="01JAN2020")

        assert path.d_part == "01JAN2020"

    def test_make_pathname_override_kwargs(self) -> None:
        """Test overriding template parts."""
        template = DSSPathnameTemplate(a_part="PROJ", c_part="FLOW", f_part="V1")

        path = template.make_pathname(location="LOC", f_part="V2")

        assert path.f_part == "V2"  # Overridden

    def test_make_pathnames_multiple_locations(self) -> None:
        """Test creating pathnames for multiple locations."""
        template = DSSPathnameTemplate(a_part="PROJ", c_part="FLOW", e_part="1DAY")

        locations = ["LOC1", "LOC2", "LOC3"]
        paths = list(template.make_pathnames(locations))

        assert len(paths) == 3
        assert paths[0].b_part == "LOC1"
        assert paths[1].b_part == "LOC2"
        assert paths[2].b_part == "LOC3"

    def test_make_pathnames_with_date(self) -> None:
        """Test creating pathnames with date range."""
        template = DSSPathnameTemplate(a_part="PROJ", c_part="HEAD")

        paths = list(template.make_pathnames(["LOC1", "LOC2"], date_range="01JAN2020"))

        assert paths[0].d_part == "01JAN2020"
        assert paths[1].d_part == "01JAN2020"


# =============================================================================
# Test Date Functions
# =============================================================================


class TestDateFunctions:
    """Tests for date formatting/parsing functions."""

    def test_format_dss_date(self) -> None:
        """Test formatting datetime to DSS date."""
        dt = datetime(2020, 1, 15)
        result = format_dss_date(dt)

        assert result == "15JAN2020"

    def test_format_dss_date_december(self) -> None:
        """Test formatting December date."""
        dt = datetime(2020, 12, 31)
        result = format_dss_date(dt)

        assert result == "31DEC2020"

    def test_format_dss_date_leading_zero_day(self) -> None:
        """Test formatting date with leading zero day."""
        dt = datetime(2020, 6, 1)
        result = format_dss_date(dt)

        assert result == "01JUN2020"

    def test_format_dss_date_range(self) -> None:
        """Test formatting date range."""
        start = datetime(2020, 1, 1)
        end = datetime(2020, 12, 31)
        result = format_dss_date_range(start, end)

        assert result == "01JAN2020-31DEC2020"

    def test_parse_dss_date(self) -> None:
        """Test parsing DSS date string."""
        result = parse_dss_date("15JAN2020")

        assert result == datetime(2020, 1, 15)

    def test_parse_dss_date_lowercase(self) -> None:
        """Test parsing lowercase DSS date."""
        result = parse_dss_date("15jan2020")

        assert result == datetime(2020, 1, 15)

    def test_parse_dss_date_all_months(self) -> None:
        """Test parsing dates for all months."""
        months = [
            ("01JAN2020", 1),
            ("01FEB2020", 2),
            ("01MAR2020", 3),
            ("01APR2020", 4),
            ("01MAY2020", 5),
            ("01JUN2020", 6),
            ("01JUL2020", 7),
            ("01AUG2020", 8),
            ("01SEP2020", 9),
            ("01OCT2020", 10),
            ("01NOV2020", 11),
            ("01DEC2020", 12),
        ]

        for date_str, expected_month in months:
            result = parse_dss_date(date_str)
            assert result.month == expected_month


# =============================================================================
# Test Interval Functions
# =============================================================================


class TestIntervalFunctions:
    """Tests for interval conversion functions."""

    def test_interval_to_minutes_min(self) -> None:
        """Test converting minute intervals."""
        assert interval_to_minutes("1MIN") == 1
        assert interval_to_minutes("5MIN") == 5
        assert interval_to_minutes("15MIN") == 15
        assert interval_to_minutes("30MIN") == 30

    def test_interval_to_minutes_hour(self) -> None:
        """Test converting hour intervals."""
        assert interval_to_minutes("1HOUR") == 60
        assert interval_to_minutes("2HOUR") == 120
        assert interval_to_minutes("6HOUR") == 360

    def test_interval_to_minutes_day(self) -> None:
        """Test converting day intervals."""
        assert interval_to_minutes("1DAY") == 60 * 24

    def test_interval_to_minutes_week(self) -> None:
        """Test converting week intervals."""
        assert interval_to_minutes("1WEEK") == 60 * 24 * 7

    def test_interval_to_minutes_month(self) -> None:
        """Test converting month intervals."""
        assert interval_to_minutes("1MON") == 60 * 24 * 30

    def test_interval_to_minutes_year(self) -> None:
        """Test converting year intervals."""
        assert interval_to_minutes("1YEAR") == 60 * 24 * 365

    def test_interval_to_minutes_lowercase(self) -> None:
        """Test converting lowercase interval."""
        assert interval_to_minutes("1day") == 60 * 24

    def test_interval_to_minutes_invalid(self) -> None:
        """Test converting invalid interval."""
        with pytest.raises(ValueError, match="Unknown DSS interval"):
            interval_to_minutes("INVALID")

    def test_minutes_to_interval_min(self) -> None:
        """Test converting minutes to minute intervals."""
        assert minutes_to_interval(1) == "1MIN"
        assert minutes_to_interval(5) == "5MIN"
        assert minutes_to_interval(15) == "15MIN"
        assert minutes_to_interval(30) == "30MIN"

    def test_minutes_to_interval_min_approx(self) -> None:
        """Test converting non-standard minutes (finds nearest)."""
        result = minutes_to_interval(7)  # Between 6 and 10
        assert result in ["6MIN", "10MIN"]

    def test_minutes_to_interval_hour(self) -> None:
        """Test converting to hour intervals."""
        assert minutes_to_interval(60) == "1HOUR"
        assert minutes_to_interval(120) == "2HOUR"
        assert minutes_to_interval(360) == "6HOUR"

    def test_minutes_to_interval_day(self) -> None:
        """Test converting to day interval."""
        assert minutes_to_interval(60 * 24) == "1DAY"
        assert minutes_to_interval(60 * 24 * 3) == "1DAY"  # Within week

    def test_minutes_to_interval_week(self) -> None:
        """Test converting to week interval."""
        assert minutes_to_interval(60 * 24 * 7) == "1WEEK"
        assert minutes_to_interval(60 * 24 * 14) == "1WEEK"

    def test_minutes_to_interval_month(self) -> None:
        """Test converting to month interval."""
        assert minutes_to_interval(60 * 24 * 30) == "1MON"
        assert minutes_to_interval(60 * 24 * 60) == "1MON"

    def test_minutes_to_interval_year(self) -> None:
        """Test converting to year interval."""
        assert minutes_to_interval(60 * 24 * 400) == "1YEAR"


# =============================================================================
# Test Constants
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_valid_intervals_exist(self) -> None:
        """Test that common intervals are in VALID_INTERVALS."""
        assert "1MIN" in VALID_INTERVALS
        assert "1HOUR" in VALID_INTERVALS
        assert "1DAY" in VALID_INTERVALS
        assert "1MON" in VALID_INTERVALS
        assert "1YEAR" in VALID_INTERVALS
        assert "IR-DAY" in VALID_INTERVALS

    def test_interval_mapping_keys(self) -> None:
        """Test that common interval names are mapped."""
        assert "daily" in INTERVAL_MAPPING
        assert "hourly" in INTERVAL_MAPPING
        assert "monthly" in INTERVAL_MAPPING

    def test_parameter_codes_keys(self) -> None:
        """Test that common parameters are mapped."""
        assert "flow" in PARAMETER_CODES
        assert "head" in PARAMETER_CODES
        assert "storage" in PARAMETER_CODES
        assert "pumping" in PARAMETER_CODES


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_pathname_roundtrip(self) -> None:
        """Test creating and parsing pathname."""
        original = DSSPathname(
            a_part="PROJECT",
            b_part="LOCATION",
            c_part="FLOW",
            d_part="01JAN2020",
            e_part="1DAY",
            f_part="V1",
        )

        parsed = DSSPathname.from_string(str(original))

        assert parsed.a_part == original.a_part
        assert parsed.b_part == original.b_part
        assert parsed.c_part == original.c_part
        assert parsed.d_part == original.d_part
        assert parsed.e_part == original.e_part
        assert parsed.f_part == original.f_part

    def test_long_location_name(self) -> None:
        """Test pathname with long location name."""
        path = DSSPathname(b_part="THIS_IS_A_VERY_LONG_LOCATION_NAME_123")

        assert path.b_part == "THIS_IS_A_VERY_LONG_LOCATION_NAME_123"

    def test_special_characters_in_parts(self) -> None:
        """Test pathname with special characters."""
        path = DSSPathname(
            a_part="PROJ-1",
            b_part="LOC_01",
            c_part="FLOW-CFS",
        )

        assert path.a_part == "PROJ-1"
        assert path.b_part == "LOC_01"
        assert path.c_part == "FLOW-CFS"

    def test_date_range_format(self) -> None:
        """Test date range in d_part."""
        path = DSSPathname.from_string("/PROJ/LOC/FLOW/01JAN2020-31DEC2020/1DAY/V1/")

        assert path.d_part == "01JAN2020-31DEC2020"

    def test_matches_empty_pattern(self) -> None:
        """Test matching with completely empty pattern (matches all)."""
        path = DSSPathname.from_string("/PROJ/LOC/FLOW//1DAY/V1/")
        empty_pattern = DSSPathname()

        assert path.matches(empty_pattern)
