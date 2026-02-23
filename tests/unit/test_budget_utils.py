"""Tests for pyiwfm.io.budget_utils — unit conversion, title formatting, time filtering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyiwfm.io.budget_utils import apply_unit_conversion, filter_time_range, format_title_lines


class TestApplyUnitConversion:
    """Tests for apply_unit_conversion()."""

    def test_volume_columns(self) -> None:
        """Volume type codes (1, 2, 3, 6-11) multiply by volume_factor."""
        values = np.ones((3, 3))
        column_types = [1, 2, 3]  # VR, VLB, VLE
        result = apply_unit_conversion(values, column_types, volume_factor=2.0)
        np.testing.assert_array_equal(result, np.full((3, 3), 2.0))

    def test_area_column(self) -> None:
        """Area type code (4) multiplies by area_factor."""
        values = np.ones((2, 2))
        column_types = [1, 4]  # VR, AR
        result = apply_unit_conversion(values, column_types, volume_factor=3.0, area_factor=5.0)
        assert result[0, 0] == 3.0
        assert result[0, 1] == 5.0

    def test_length_column(self) -> None:
        """Length type code (5) multiplies by length_factor."""
        values = np.ones((2, 1))
        column_types = [5]  # LT
        result = apply_unit_conversion(values, column_types, length_factor=10.0)
        np.testing.assert_array_equal(result, np.full((2, 1), 10.0))

    def test_mixed_types(self) -> None:
        """Mixed column types get correct per-column factors."""
        values = np.ones((1, 5))
        column_types = [1, 4, 5, 6, 3]
        result = apply_unit_conversion(
            values,
            column_types,
            length_factor=2.0,
            area_factor=3.0,
            volume_factor=4.0,
        )
        expected = np.array([[4.0, 3.0, 2.0, 4.0, 4.0]])
        np.testing.assert_array_equal(result, expected)

    def test_does_not_mutate_input(self) -> None:
        """Input array must not be modified."""
        values = np.ones((2, 2))
        column_types = [1, 4]
        apply_unit_conversion(values, column_types, volume_factor=99.0, area_factor=99.0)
        np.testing.assert_array_equal(values, np.ones((2, 2)))

    def test_extra_column_types_ignored(self) -> None:
        """Column types beyond the data width are silently ignored."""
        values = np.ones((2, 2))
        column_types = [1, 4, 5, 1, 1]  # More types than columns
        result = apply_unit_conversion(values, column_types, volume_factor=2.0, area_factor=3.0)
        assert result[0, 0] == 2.0
        assert result[0, 1] == 3.0

    def test_lwu_variant_types(self) -> None:
        """LWU variant codes (6-11) are treated as volume."""
        values = np.ones((1, 6))
        column_types = [6, 7, 8, 9, 10, 11]
        result = apply_unit_conversion(values, column_types, volume_factor=7.0)
        np.testing.assert_array_equal(result, np.full((1, 6), 7.0))


class TestFormatTitleLines:
    """Tests for format_title_lines()."""

    def test_unit_markers_replaced(self) -> None:
        titles = ["Volume in @UNITVL@", "Area in @UNITAR@", "Length in @UNITLT@"]
        result = format_title_lines(
            titles,
            location_name="Subregion 1",
            area=100.0,
            length_unit="FT",
            area_unit="AC",
            volume_unit="AC-FT",
        )
        assert result[0] == "Volume in AC-FT"
        assert result[1] == "Area in AC"
        assert result[2] == "Length in FT"

    def test_location_and_area_markers(self) -> None:
        titles = ["Budget for @LOCNAME@", "Area = @AREA@ @UNITAR@"]
        result = format_title_lines(
            titles,
            location_name="Zone 3",
            area=42.5,
            length_unit="FT",
            area_unit="AC",
            volume_unit="AC-FT",
        )
        assert result[0] == "Budget for Zone 3"
        assert result[1] == "Area = 42.50 AC"

    def test_none_area(self) -> None:
        titles = ["Area = @AREA@"]
        result = format_title_lines(
            titles,
            location_name="X",
            area=None,
            length_unit="FT",
            area_unit="AC",
            volume_unit="AF",
        )
        assert result[0] == "Area = N/A"

    def test_empty_titles(self) -> None:
        result = format_title_lines(
            [],
            location_name="X",
            area=1.0,
            length_unit="FT",
            area_unit="AC",
            volume_unit="AF",
        )
        assert result == []


class TestFilterTimeRange:
    """Tests for filter_time_range()."""

    @pytest.fixture()
    def sample_df(self) -> pd.DataFrame:
        dates = pd.date_range("2000-01-01", periods=12, freq="MS")
        return pd.DataFrame({"val": range(12)}, index=dates)

    def test_no_filter(self, sample_df: pd.DataFrame) -> None:
        result = filter_time_range(sample_df, None, None)
        assert len(result) == 12

    def test_begin_only(self, sample_df: pd.DataFrame) -> None:
        result = filter_time_range(sample_df, "06/01/2000_00:00", None)
        assert len(result) == 7  # Jun-Dec

    def test_end_only(self, sample_df: pd.DataFrame) -> None:
        result = filter_time_range(sample_df, None, "06/01/2000_00:00")
        assert len(result) == 6  # Jan-Jun

    def test_both_bounds(self, sample_df: pd.DataFrame) -> None:
        result = filter_time_range(sample_df, "03/01/2000_00:00", "09/01/2000_00:00")
        assert len(result) == 7  # Mar-Sep

    def test_non_datetime_index(self) -> None:
        df = pd.DataFrame({"val": [1, 2, 3]}, index=[0, 1, 2])
        result = filter_time_range(df, "01/01/2000_00:00", None)
        assert len(result) == 3  # No filtering applied
