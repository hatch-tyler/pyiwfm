"""Unit tests for IWFMHydrographReader (hydrograph_reader.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyiwfm.visualization.webapi.hydrograph_reader import IWFMHydrographReader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_HYDROGRAPH = """\
* IWFM Groundwater Hydrograph Output
* HYDROGRAPH ID       1       2       3
* LAYER               1       1       1
* NODE              101     102     103
01/31/2000_24:00    10.5    20.3    30.1
02/29/2000_24:00    11.0    21.0    31.0
03/31/2000_24:00    11.5    21.5    31.5
"""

SAMPLE_STREAM_HYDROGRAPH = """\
* IWFM Stream Hydrograph Output
* HYDROGRAPH ID       1       2
* ELEMENT            50      51
01/15/2010_12:00    100.0    200.0
02/15/2010_12:00    150.0    250.0
"""


def _write_hydro(tmp_path: Path, content: str, name: str = "test.out") -> Path:
    filepath = tmp_path / name
    filepath.write_text(content)
    return filepath


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestHydrographReaderInit:
    """Tests for constructor and parsing."""

    def test_file_not_found(self, tmp_path: Path) -> None:
        reader = IWFMHydrographReader(tmp_path / "nonexistent.out")
        assert reader.n_columns == 0
        assert reader.n_timesteps == 0

    def test_read_error(self, tmp_path: Path) -> None:
        # Create a directory with the file name to force read error
        bad = tmp_path / "bad.out"
        bad.mkdir()
        reader = IWFMHydrographReader(bad)
        assert reader.n_columns == 0

    def test_empty_file(self, tmp_path: Path) -> None:
        filepath = _write_hydro(tmp_path, "")
        reader = IWFMHydrographReader(filepath)
        assert reader.n_columns == 0
        assert reader.n_timesteps == 0


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------


class TestHeaderParsing:
    """Tests for header metadata extraction."""

    def test_hydrograph_ids(self, tmp_path: Path) -> None:
        filepath = _write_hydro(tmp_path, SAMPLE_HYDROGRAPH)
        reader = IWFMHydrographReader(filepath)
        assert reader.hydrograph_ids == [1, 2, 3]

    def test_layer_numbers(self, tmp_path: Path) -> None:
        filepath = _write_hydro(tmp_path, SAMPLE_HYDROGRAPH)
        reader = IWFMHydrographReader(filepath)
        assert reader.layers == [1, 1, 1]

    def test_node_ids(self, tmp_path: Path) -> None:
        filepath = _write_hydro(tmp_path, SAMPLE_HYDROGRAPH)
        reader = IWFMHydrographReader(filepath)
        assert reader.node_ids == [101, 102, 103]

    def test_element_header(self, tmp_path: Path) -> None:
        filepath = _write_hydro(tmp_path, SAMPLE_STREAM_HYDROGRAPH)
        reader = IWFMHydrographReader(filepath)
        assert reader.node_ids == [50, 51]

    def test_mixed_headers(self, tmp_path: Path) -> None:
        content = """\
* HYDROGRAPH ID       1       2
* LAYER               1       2
* NODE              100     200
01/01/2020_00:00    5.0     6.0
"""
        filepath = _write_hydro(tmp_path, content)
        reader = IWFMHydrographReader(filepath)
        assert reader.hydrograph_ids == [1, 2]
        assert reader.layers == [1, 2]
        assert reader.node_ids == [100, 200]


# ---------------------------------------------------------------------------
# Data parsing
# ---------------------------------------------------------------------------


class TestDataParsing:
    """Tests for time series data extraction."""

    def test_standard_format(self, tmp_path: Path) -> None:
        filepath = _write_hydro(tmp_path, SAMPLE_HYDROGRAPH)
        reader = IWFMHydrographReader(filepath)
        assert reader.n_columns == 3
        assert reader.n_timesteps == 3

    def test_24_00_convention(self, tmp_path: Path) -> None:
        """24:00 = end of day -> next day 00:00."""
        filepath = _write_hydro(tmp_path, SAMPLE_HYDROGRAPH)
        reader = IWFMHydrographReader(filepath)
        # 01/31/2000_24:00 -> 2000-02-01T00:00:00
        assert reader.times[0] == "2000-02-01T00:00:00"

    def test_standard_time_parsing(self, tmp_path: Path) -> None:
        filepath = _write_hydro(tmp_path, SAMPLE_STREAM_HYDROGRAPH)
        reader = IWFMHydrographReader(filepath)
        assert reader.times[0] == "2010-01-15T12:00:00"

    def test_invalid_datetime_skipped(self, tmp_path: Path) -> None:
        content = """\
* HYDROGRAPH ID    1
INVALID_DATE    5.0
01/01/2020_12:00    6.0
"""
        filepath = _write_hydro(tmp_path, content)
        reader = IWFMHydrographReader(filepath)
        assert reader.n_timesteps == 1

    def test_nan_for_non_numeric(self, tmp_path: Path) -> None:
        content = """\
* HYDROGRAPH ID    1    2
01/01/2020_12:00    5.0    NaN
02/01/2020_12:00    6.0    abc
"""
        filepath = _write_hydro(tmp_path, content)
        reader = IWFMHydrographReader(filepath)
        assert reader.n_timesteps == 2
        assert np.isnan(reader._data[1, 1])

    def test_unequal_row_lengths_padded(self, tmp_path: Path) -> None:
        content = """\
* HYDROGRAPH ID    1    2    3
01/01/2020_12:00    5.0    6.0    7.0
02/01/2020_12:00    8.0    9.0
"""
        filepath = _write_hydro(tmp_path, content)
        reader = IWFMHydrographReader(filepath)
        assert reader._data.shape == (2, 3)
        assert np.isnan(reader._data[1, 2])


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    """Tests for reader properties."""

    def test_n_columns(self, tmp_path: Path) -> None:
        filepath = _write_hydro(tmp_path, SAMPLE_HYDROGRAPH)
        reader = IWFMHydrographReader(filepath)
        assert reader.n_columns == 3

    def test_n_timesteps(self, tmp_path: Path) -> None:
        filepath = _write_hydro(tmp_path, SAMPLE_HYDROGRAPH)
        reader = IWFMHydrographReader(filepath)
        assert reader.n_timesteps == 3

    def test_times(self, tmp_path: Path) -> None:
        filepath = _write_hydro(tmp_path, SAMPLE_HYDROGRAPH)
        reader = IWFMHydrographReader(filepath)
        assert len(reader.times) == 3


# ---------------------------------------------------------------------------
# get_time_series
# ---------------------------------------------------------------------------


class TestGetTimeSeries:
    """Tests for get_time_series()."""

    def test_valid_column(self, tmp_path: Path) -> None:
        filepath = _write_hydro(tmp_path, SAMPLE_HYDROGRAPH)
        reader = IWFMHydrographReader(filepath)
        times, values = reader.get_time_series(0)
        assert len(times) == 3
        assert len(values) == 3
        assert values[0] == pytest.approx(10.5)

    def test_out_of_range(self, tmp_path: Path) -> None:
        filepath = _write_hydro(tmp_path, SAMPLE_HYDROGRAPH)
        reader = IWFMHydrographReader(filepath)
        times, values = reader.get_time_series(99)
        assert times == []
        assert values == []

    def test_negative_index(self, tmp_path: Path) -> None:
        filepath = _write_hydro(tmp_path, SAMPLE_HYDROGRAPH)
        reader = IWFMHydrographReader(filepath)
        times, values = reader.get_time_series(-1)
        assert times == []

    def test_unparsed_reader(self, tmp_path: Path) -> None:
        reader = IWFMHydrographReader(tmp_path / "nonexistent.out")
        times, values = reader.get_time_series(0)
        assert times == []


# ---------------------------------------------------------------------------
# find_column_by_node_id
# ---------------------------------------------------------------------------


class TestFindColumnByNodeId:
    """Tests for find_column_by_node_id()."""

    def test_found(self, tmp_path: Path) -> None:
        filepath = _write_hydro(tmp_path, SAMPLE_HYDROGRAPH)
        reader = IWFMHydrographReader(filepath)
        assert reader.find_column_by_node_id(101) == 0
        assert reader.find_column_by_node_id(102) == 1
        assert reader.find_column_by_node_id(103) == 2

    def test_not_found(self, tmp_path: Path) -> None:
        filepath = _write_hydro(tmp_path, SAMPLE_HYDROGRAPH)
        reader = IWFMHydrographReader(filepath)
        assert reader.find_column_by_node_id(999) is None


# ---------------------------------------------------------------------------
# _parse_iwfm_datetime (static method)
# ---------------------------------------------------------------------------


class TestParseIwfmDatetime:
    """Tests for the static datetime parser."""

    def test_normal_format(self) -> None:
        result = IWFMHydrographReader._parse_iwfm_datetime("03/15/2020_14:30")
        assert result == "2020-03-15T14:30:00"

    def test_24_00_end_of_day(self) -> None:
        result = IWFMHydrographReader._parse_iwfm_datetime("01/31/2020_24:00")
        assert result == "2020-02-01T00:00:00"

    def test_invalid_format_returns_none(self) -> None:
        assert IWFMHydrographReader._parse_iwfm_datetime("not-a-date") is None

    def test_midnight(self) -> None:
        result = IWFMHydrographReader._parse_iwfm_datetime("06/01/2015_00:00")
        assert result == "2015-06-01T00:00:00"

    def test_end_of_year(self) -> None:
        result = IWFMHydrographReader._parse_iwfm_datetime("12/31/2020_24:00")
        assert result == "2021-01-01T00:00:00"
