"""Extra tests for pyiwfm.io.budget targeting uncovered branches.

Covers:
- parse_iwfm_datetime edge cases (empty, _24:00, various formats, unparseable)
- julian_to_datetime and excel_julian_to_datetime helpers
- BudgetReader format detection (hdf5 extensions, binary extensions, fallback)
- BudgetReader.__init__ FileNotFoundError
- _h5_get: child dataset (scalar, array), child group, attrs, missing
- _read_header_hdf5: Attributes group, root fallback, descriptor bytes/str,
  timestep metadata with/without prefix, areas, ASCII output, location data,
  DSS pathname fallback, n_columns from data shape, generic headers,
  location data replication, timestep count from data
- _read_header_binary: full binary header round-trip
- Properties: descriptor, locations, n_locations, n_timesteps
- get_location_index: int, out of range, name exact, name case-insensitive, missing
- get_column_headers: single/multiple location data, UNIT_MARKERS, Time strip
- _read_values_hdf5: Dataset/Group, 1D data, transpose, monthly/daily/no-datetime,
  column selection, missing location, no data
- _read_values_binary: single/multiple location data, monthly/daily/no-datetime,
  column selection
- get_dataframe: column name/index, exact/case-insensitive lookup, missing col
- get_all_dataframes: all locations
- get_monthly_averages: DatetimeIndex vs numeric
- get_annual_totals: DatetimeIndex, <12 rows, >=12 rows
- get_cumulative
- __repr__
"""

from __future__ import annotations

import struct
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pandas as pd
import pytest

from pyiwfm.io.budget import (
    BUDGET_DATA_TYPES,
    DSS_DATA_TYPES,
    UNIT_MARKERS,
    ASCIIOutputInfo,
    BudgetHeader,
    BudgetReader,
    LocationData,
    TimeStepInfo,
    excel_julian_to_datetime,
    julian_to_datetime,
    parse_iwfm_datetime,
)


# =============================================================================
# Helpers to create binary and HDF5 budget files for testing
# =============================================================================


def _write_fortran_string(f, s: str, padded_len: int = 0) -> None:
    """Write a Fortran unformatted string record."""
    data = s.encode("ascii")
    if padded_len > 0:
        data = data[:padded_len].ljust(padded_len, b" ")
    rec_len = struct.pack("i", len(data))
    f.write(rec_len)
    f.write(data)
    f.write(rec_len)


def _write_fortran_int(f, val: int) -> None:
    """Write a Fortran unformatted int record."""
    rec_len = struct.pack("i", 4)
    f.write(rec_len)
    f.write(struct.pack("i", val))
    f.write(rec_len)


def _write_fortran_real8(f, val: float) -> None:
    """Write a Fortran unformatted REAL(8) record."""
    rec_len = struct.pack("i", 8)
    f.write(rec_len)
    f.write(struct.pack("d", val))
    f.write(rec_len)


def _write_fortran_logical(f, val: bool) -> None:
    """Write a Fortran unformatted logical record."""
    _write_fortran_int(f, 1 if val else 0)


def _write_fortran_int_array(f, vals: list[int]) -> None:
    """Write a Fortran unformatted int array record."""
    n = len(vals)
    rec_len = struct.pack("i", 4 * n)
    f.write(rec_len)
    for v in vals:
        f.write(struct.pack("i", v))
    f.write(rec_len)


def _write_fortran_real8_array(f, vals: list[float]) -> None:
    """Write a Fortran unformatted REAL(8) array record."""
    n = len(vals)
    rec_len = struct.pack("i", 8 * n)
    f.write(rec_len)
    for v in vals:
        f.write(struct.pack("d", v))
    f.write(rec_len)


def _create_binary_budget(
    filepath: Path,
    *,
    descriptor: str = "GROUNDWATER BUDGET",
    n_timesteps: int = 3,
    track_time: bool = True,
    delta_t: float = 1.0,
    delta_t_minutes: int = 1440,
    unit: str = "1DAY",
    start_datetime_str: str = "10/01/2000_24:00",
    start_time: float = 0.0,
    n_areas: int = 2,
    areas: list[float] | None = None,
    n_titles: int = 1,
    titles: list[str] | None = None,
    n_locations: int = 2,
    location_names: list[str] | None = None,
    n_columns: int = 3,
    n_column_header_lines: int = 3,
    write_data: bool = True,
) -> None:
    """Create a minimal IWFM binary budget file for testing."""
    if areas is None:
        areas = [1000.0 * (i + 1) for i in range(n_areas)]
    if titles is None:
        titles = ["Test Budget Title"]
    if location_names is None:
        location_names = [f"Location {i + 1}" for i in range(n_locations)]

    with open(filepath, "wb") as f:
        # Descriptor (100 chars)
        _write_fortran_string(f, descriptor, 100)

        # Timestep info
        _write_fortran_int(f, n_timesteps)
        _write_fortran_logical(f, track_time)
        _write_fortran_real8(f, delta_t)
        _write_fortran_int(f, delta_t_minutes)
        _write_fortran_string(f, unit, 10)

        if track_time:
            _write_fortran_string(f, start_datetime_str, 21)
        _write_fortran_real8(f, start_time)

        # Areas
        _write_fortran_int(f, n_areas)
        if n_areas > 0:
            _write_fortran_real8_array(f, areas)

        # ASCII output info
        title_len = 160
        _write_fortran_int(f, title_len)
        _write_fortran_int(f, n_titles)
        for t in titles:
            _write_fortran_string(f, t, 1000)
        for _ in range(n_titles):
            _write_fortran_logical(f, True)
        _write_fortran_string(f, "(1X,A16,100(1X,F12.2))", 500)
        _write_fortran_int(f, n_column_header_lines)

        # Locations
        _write_fortran_int(f, n_locations)
        for name in location_names:
            _write_fortran_string(f, name, 100)

        # Location data (1 entry used for all locations)
        n_loc_data = 1
        _write_fortran_int(f, n_loc_data)

        storage_units = n_columns
        _write_fortran_int(f, n_columns)  # n_columns
        _write_fortran_int(f, storage_units)  # storage_units

        col_headers = [f"Col {j + 1}" for j in range(n_columns)]
        for h in col_headers:
            _write_fortran_string(f, h, 100)

        col_types = [1] * n_columns  # VR type
        _write_fortran_int_array(f, col_types)

        col_widths = [12] * n_columns
        _write_fortran_int_array(f, col_widths)

        # Multi-line column headers (skip content, just write blanks)
        for _ in range(n_column_header_lines):
            for _ in range(n_columns):
                _write_fortran_string(f, "", 100)
        for _ in range(n_column_header_lines):
            _write_fortran_string(f, "", 500)

        # DSS output info
        _write_fortran_int(f, 0)  # n_dss_pathnames
        _write_fortran_int(f, 0)  # n_dss_types
        _write_fortran_int_array(f, [])  # empty array - but we said 0 items

        # Record file position (data starts here)
        data_start = f.tell()

        # Write data: n_timesteps * n_locations * n_columns REAL(8)
        if write_data:
            for t in range(n_timesteps):
                for loc in range(n_locations):
                    for col in range(n_columns):
                        val = (t + 1) * 100.0 + (loc + 1) * 10.0 + (col + 1)
                        f.write(struct.pack("d", val))


def _create_hdf5_budget(
    filepath: Path,
    *,
    descriptor: str = "GROUNDWATER BUDGET",
    n_timesteps: int = 3,
    n_locations: int = 2,
    n_columns: int = 3,
    location_names: list[str] | None = None,
    use_attributes_group: bool = True,
    use_bytes: bool = True,
    use_prefixed_keys: bool = False,
    include_areas: bool = True,
    include_column_headers: bool = True,
    include_n_locations: bool = True,
    include_n_timesteps: bool = True,
    include_begin_datetime: bool = True,
    unit: str = "1DAY",
    start_datetime_str: str = "10/01/2000_24:00",
    data_shape: str = "normal",  # "normal", "transposed", "1d"
    store_as_group: bool = False,
    include_dss_pathnames: bool = False,
    include_n_columns: bool = True,
) -> None:
    """Create a minimal HDF5 budget file for testing."""
    if location_names is None:
        location_names = [f"Location {i + 1}" for i in range(n_locations)]

    with h5py.File(filepath, "w") as f:
        if use_attributes_group:
            attrs_g = f.create_group("Attributes")
        else:
            attrs_g = f

        # Descriptor
        if use_bytes:
            attrs_g.attrs["Descriptor"] = descriptor.encode()
        else:
            attrs_g.attrs["Descriptor"] = descriptor

        # Timestep
        if include_n_timesteps:
            attrs_g.attrs["NTimeSteps"] = n_timesteps

        if use_prefixed_keys:
            attrs_g.attrs["TimeStep%TrackTime"] = 1
            attrs_g.attrs["TimeStep%DeltaT"] = 1.0
            attrs_g.attrs["TimeStep%DeltaT_InMinutes"] = 1440
            attrs_g.attrs["TimeStep%Unit"] = unit.encode() if use_bytes else unit
            attrs_g.attrs["TimeStep%BeginTime"] = 0.0
            if include_begin_datetime:
                attrs_g.attrs["TimeStep%BeginDateAndTime"] = (
                    start_datetime_str.encode() if use_bytes else start_datetime_str
                )
        else:
            attrs_g.attrs["TrackTime"] = 1
            attrs_g.attrs["DeltaT"] = 1.0
            attrs_g.attrs["DeltaT_InMinutes"] = 1440
            attrs_g.attrs["Unit"] = unit.encode() if use_bytes else unit
            attrs_g.attrs["BeginTime"] = 0.0
            if include_begin_datetime:
                attrs_g.attrs["BeginDateAndTime"] = (
                    start_datetime_str.encode() if use_bytes else start_datetime_str
                )

        # Areas
        if include_areas:
            attrs_g.attrs["NAreas"] = n_locations
            attrs_g.attrs["Areas"] = [1000.0 * (i + 1) for i in range(n_locations)]

        # ASCII output
        attrs_g.attrs["TitleLen"] = 160
        attrs_g.attrs["NTitles"] = 1
        attrs_g.attrs["cTitles"] = [b"Test Budget Title"] if use_bytes else ["Test Title"]

        # Locations
        if include_n_locations:
            attrs_g.attrs["NLocations"] = n_locations

        enc_names = (
            [n.encode() for n in location_names]
            if use_bytes
            else location_names
        )
        attrs_g.attrs["cLocationNames"] = enc_names

        # Location data
        attrs_g.attrs["NLocationData"] = 1
        if include_n_columns:
            attrs_g.attrs["NDataColumns"] = n_columns

        if include_column_headers:
            col_headers = [f"Col {j + 1}" for j in range(n_columns)]
            enc_headers = (
                [h.encode() for h in col_headers]
                if use_bytes
                else col_headers
            )
            attrs_g.attrs["cFullColumnHeaders"] = enc_headers

        col_types = [1] * n_columns
        attrs_g.attrs["iDataColumnTypes"] = col_types
        attrs_g.attrs["iColWidth"] = [12] * n_columns

        # DSS pathnames fallback
        if include_dss_pathnames:
            pathnames = []
            for loc_name in location_names:
                for j in range(n_columns):
                    pathnames.append(
                        f"/BUDGET/{loc_name}/DATA/01OCT2000/1DAY/DSSCol{j+1}/".encode()
                    )
            attrs_g.attrs["DSSOutput%cPathNames"] = pathnames

        # Write data for each location
        for i, loc_name in enumerate(location_names):
            data = np.array(
                [
                    [(t + 1) * 100.0 + (i + 1) * 10.0 + (c + 1) for c in range(n_columns)]
                    for t in range(n_timesteps)
                ],
                dtype=np.float64,
            )
            if data_shape == "transposed":
                data = data.T
            elif data_shape == "1d":
                data = data.flatten()

            if store_as_group:
                grp = f.create_group(loc_name)
                grp.create_dataset("Data", data=data)
            else:
                f.create_dataset(loc_name, data=data)


# =============================================================================
# Tests for parse_iwfm_datetime
# =============================================================================


class TestParseIWFMDatetime:
    """Tests for parse_iwfm_datetime."""

    def test_empty_string_returns_none(self) -> None:
        assert parse_iwfm_datetime("") is None

    def test_whitespace_returns_none(self) -> None:
        assert parse_iwfm_datetime("   ") is None

    def test_none_returns_none(self) -> None:
        # The function checks `not date_str` first
        assert parse_iwfm_datetime("") is None

    def test_24_00_mmddyyyy(self) -> None:
        result = parse_iwfm_datetime("10/01/2000_24:00")
        assert result == datetime(2000, 10, 2, 0, 0)

    def test_24_00_yyyy_mm_dd(self) -> None:
        result = parse_iwfm_datetime("2000-10-01_24:00")
        assert result == datetime(2000, 10, 2, 0, 0)

    def test_24_00_invalid_date_falls_through(self) -> None:
        # A date that fails both formats for the _24:00 branch
        result = parse_iwfm_datetime("not-a-date_24:00")
        # Falls through to normal parsing, also fails
        assert result is None

    def test_standard_iwfm_format(self) -> None:
        result = parse_iwfm_datetime("10/01/2000_12:00")
        assert result == datetime(2000, 10, 1, 12, 0)

    def test_mmddyyyy_space_hhmm(self) -> None:
        result = parse_iwfm_datetime("10/01/2000 12:00")
        assert result == datetime(2000, 10, 1, 12, 0)

    def test_mmddyyyy_underscore_hhmmss(self) -> None:
        result = parse_iwfm_datetime("10/01/2000_12:00:30")
        assert result == datetime(2000, 10, 1, 12, 0, 30)

    def test_mmddyyyy_space_hhmmss(self) -> None:
        result = parse_iwfm_datetime("10/01/2000 12:00:30")
        assert result == datetime(2000, 10, 1, 12, 0, 30)

    def test_iso_datetime_with_seconds(self) -> None:
        result = parse_iwfm_datetime("2000-10-01 12:00:30")
        assert result == datetime(2000, 10, 1, 12, 0, 30)

    def test_iso_datetime_no_seconds(self) -> None:
        result = parse_iwfm_datetime("2000-10-01 12:00")
        assert result == datetime(2000, 10, 1, 12, 0)

    def test_date_only_mmddyyyy(self) -> None:
        result = parse_iwfm_datetime("10/01/2000")
        assert result == datetime(2000, 10, 1)

    def test_date_only_iso(self) -> None:
        result = parse_iwfm_datetime("2000-10-01")
        assert result == datetime(2000, 10, 1)

    def test_unparseable_returns_none(self) -> None:
        result = parse_iwfm_datetime("garbage-text")
        assert result is None

    def test_leading_trailing_whitespace(self) -> None:
        result = parse_iwfm_datetime("  10/01/2000_12:00  ")
        assert result == datetime(2000, 10, 1, 12, 0)


# =============================================================================
# Tests for julian_to_datetime and excel_julian_to_datetime
# =============================================================================


class TestJulianConversions:
    """Tests for Julian date conversion functions."""

    def test_julian_to_datetime_unix_epoch(self) -> None:
        # Julian day for Unix epoch
        result = julian_to_datetime(2440587.5)
        assert result == datetime(1970, 1, 1)

    def test_julian_to_datetime_known_date(self) -> None:
        # Jan 1, 2000 = Julian day 2451544.5
        result = julian_to_datetime(2451544.5)
        assert result.year == 2000
        assert result.month == 1
        assert result.day == 1

    def test_excel_julian_to_datetime_day_1(self) -> None:
        # Day 1 = Jan 1, 1900
        result = excel_julian_to_datetime(1.0)
        assert result == datetime(1899, 12, 31)

    def test_excel_julian_to_datetime_known(self) -> None:
        # Day 2 = Jan 1, 1900
        result = excel_julian_to_datetime(2.0)
        assert result == datetime(1900, 1, 1)

    def test_excel_julian_to_datetime_large(self) -> None:
        # Day 44197 = Jan 1, 2021
        result = excel_julian_to_datetime(44197.0)
        assert result.year == 2021
        assert result.month == 1
        assert result.day == 1


# =============================================================================
# Tests for data classes
# =============================================================================


class TestDataClasses:
    """Test budget dataclass defaults and constants."""

    def test_timestep_info_defaults(self) -> None:
        ts = TimeStepInfo()
        assert ts.track_time is True
        assert ts.delta_t == 1.0
        assert ts.delta_t_minutes == 1440
        assert ts.unit == "1DAY"
        assert ts.start_datetime is None
        assert ts.start_time == 0.0
        assert ts.n_timesteps == 0

    def test_ascii_output_info_defaults(self) -> None:
        ao = ASCIIOutputInfo()
        assert ao.title_len == 160
        assert ao.n_titles == 0
        assert ao.titles == []
        assert ao.title_persist == []
        assert ao.format_spec == ""
        assert ao.n_column_header_lines == 3

    def test_location_data_defaults(self) -> None:
        ld = LocationData()
        assert ld.n_columns == 0
        assert ld.storage_units == 0
        assert ld.column_headers == []
        assert ld.column_types == []
        assert ld.column_widths == []
        assert ld.dss_pathnames == []
        assert ld.dss_data_types == []

    def test_budget_header_defaults(self) -> None:
        bh = BudgetHeader()
        assert bh.descriptor == ""
        assert bh.n_areas == 0
        assert len(bh.areas) == 0
        assert bh.n_locations == 0
        assert bh.location_names == []
        assert bh.location_data == []
        assert bh.file_position == 0

    def test_budget_data_types_dict(self) -> None:
        assert BUDGET_DATA_TYPES[1] == "VR"
        assert BUDGET_DATA_TYPES[11] == "VR_AgOthIn"

    def test_dss_data_types_dict(self) -> None:
        assert DSS_DATA_TYPES[1] == "PER-CUM"
        assert DSS_DATA_TYPES[2] == "PER-AVER"

    def test_unit_markers_dict(self) -> None:
        assert "@UNITVL@" in UNIT_MARKERS
        assert UNIT_MARKERS["@LOCNAME@"] == "location"


# =============================================================================
# Tests for BudgetReader.__init__ and _detect_format
# =============================================================================


class TestBudgetReaderInit:
    """Tests for BudgetReader initialization and format detection."""

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Budget file not found"):
            BudgetReader(tmp_path / "nonexistent.bin")

    def test_detect_format_hdf_extension(self, tmp_path: Path) -> None:
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(p)
        reader = BudgetReader(p)
        assert reader.format == "hdf5"

    def test_detect_format_h5_extension(self, tmp_path: Path) -> None:
        p = tmp_path / "budget.h5"
        _create_hdf5_budget(p)
        reader = BudgetReader(p)
        assert reader.format == "hdf5"

    def test_detect_format_hdf5_extension(self, tmp_path: Path) -> None:
        p = tmp_path / "budget.hdf5"
        _create_hdf5_budget(p)
        reader = BudgetReader(p)
        assert reader.format == "hdf5"

    def test_detect_format_bin_extension(self, tmp_path: Path) -> None:
        p = tmp_path / "budget.bin"
        _create_binary_budget(p)
        reader = BudgetReader(p)
        assert reader.format == "binary"

    def test_detect_format_out_extension(self, tmp_path: Path) -> None:
        p = tmp_path / "budget.out"
        _create_binary_budget(p)
        reader = BudgetReader(p)
        assert reader.format == "binary"

    def test_detect_format_unknown_extension_hdf5_content(self, tmp_path: Path) -> None:
        """Unknown extension but valid HDF5 content -> detected as hdf5."""
        p = tmp_path / "budget.dat"
        _create_hdf5_budget(p)
        reader = BudgetReader(p)
        assert reader.format == "hdf5"

    def test_detect_format_unknown_extension_binary_content(self, tmp_path: Path) -> None:
        """Unknown extension but binary content -> fallback to binary."""
        p = tmp_path / "budget.dat"
        _create_binary_budget(p)
        reader = BudgetReader(p)
        assert reader.format == "binary"


# =============================================================================
# Tests for HDF5 header reading
# =============================================================================


class TestReadHeaderHDF5:
    """Tests for _read_header_hdf5."""

    def test_basic_hdf5_read(self, tmp_path: Path) -> None:
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(p)
        reader = BudgetReader(p)
        assert reader.descriptor == "GROUNDWATER BUDGET"
        assert reader.n_locations == 2
        assert reader.n_timesteps == 3
        assert len(reader.locations) == 2

    def test_hdf5_without_attributes_group(self, tmp_path: Path) -> None:
        """Attributes stored at root level."""
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(p, use_attributes_group=False)
        reader = BudgetReader(p)
        assert reader.descriptor == "GROUNDWATER BUDGET"

    def test_hdf5_bytes_descriptor(self, tmp_path: Path) -> None:
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(p, use_bytes=True)
        reader = BudgetReader(p)
        assert reader.descriptor == "GROUNDWATER BUDGET"

    def test_hdf5_string_descriptor(self, tmp_path: Path) -> None:
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(p, use_bytes=False)
        reader = BudgetReader(p)
        assert reader.descriptor == "GROUNDWATER BUDGET"

    def test_hdf5_prefixed_timestep_keys(self, tmp_path: Path) -> None:
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(p, use_prefixed_keys=True)
        reader = BudgetReader(p)
        assert reader.header.timestep.track_time is True
        assert reader.header.timestep.delta_t == 1.0
        assert reader.header.timestep.delta_t_minutes == 1440

    def test_hdf5_no_begin_datetime(self, tmp_path: Path) -> None:
        """No BeginDateAndTime attribute -> start_datetime is None."""
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(p, include_begin_datetime=False)
        reader = BudgetReader(p)
        assert reader.header.timestep.start_datetime is None

    def test_hdf5_infer_n_locations_from_names(self, tmp_path: Path) -> None:
        """n_locations not in attrs but location_names present."""
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(p, include_n_locations=False)
        reader = BudgetReader(p)
        assert reader.n_locations == 2

    def test_hdf5_infer_n_areas_from_array(self, tmp_path: Path) -> None:
        """NAreas=0 in attrs but Areas array present."""
        p = tmp_path / "budget.hdf"
        with h5py.File(p, "w") as f:
            attrs_g = f.create_group("Attributes")
            attrs_g.attrs["Descriptor"] = b"TEST"
            attrs_g.attrs["NTimeSteps"] = 2
            attrs_g.attrs["TrackTime"] = 1
            attrs_g.attrs["DeltaT"] = 1.0
            attrs_g.attrs["DeltaT_InMinutes"] = 1440
            attrs_g.attrs["Unit"] = b"1DAY"
            attrs_g.attrs["BeginTime"] = 0.0
            attrs_g.attrs["BeginDateAndTime"] = b"10/01/2000_24:00"
            # NAreas = 0 but provide Areas array
            attrs_g.attrs["NAreas"] = 0
            attrs_g.attrs["Areas"] = [100.0, 200.0]
            attrs_g.attrs["TitleLen"] = 160
            attrs_g.attrs["NTitles"] = 0
            attrs_g.attrs["NLocations"] = 1
            attrs_g.attrs["cLocationNames"] = [b"Loc1"]
            attrs_g.attrs["NLocationData"] = 1
            attrs_g.attrs["NDataColumns"] = 2
            attrs_g.attrs["cFullColumnHeaders"] = [b"A", b"B"]
            attrs_g.attrs["iDataColumnTypes"] = [1, 1]
            attrs_g.attrs["iColWidth"] = [12, 12]
            f.create_dataset("Loc1", data=np.ones((2, 2)))
        reader = BudgetReader(p)
        assert reader.header.n_areas == 2

    def test_hdf5_location_data_replicated(self, tmp_path: Path) -> None:
        """Single location data entry replicated for multiple locations."""
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(p, n_locations=3, location_names=["A", "B", "C"])
        reader = BudgetReader(p)
        assert len(reader.header.location_data) == 3

    def test_hdf5_transposed_data(self, tmp_path: Path) -> None:
        """Data stored in (n_columns, n_timesteps) orientation."""
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(p, data_shape="transposed", n_timesteps=5, n_columns=3)
        reader = BudgetReader(p)
        # Should still read correctly after transpose detection
        times, values = reader.get_values(0)
        assert values.shape == (5, 3)

    def test_hdf5_data_in_group(self, tmp_path: Path) -> None:
        """Location data stored as a Group with a child Dataset."""
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(p, store_as_group=True)
        reader = BudgetReader(p)
        times, values = reader.get_values(0)
        assert values.shape[0] == 3
        assert values.shape[1] == 3

    def test_hdf5_no_column_headers_dss_fallback(self, tmp_path: Path) -> None:
        """Column headers derived from DSS path names when empty."""
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(
            p,
            include_column_headers=False,
            include_dss_pathnames=True,
            n_columns=3,
            include_n_columns=False,
        )
        reader = BudgetReader(p)
        headers = reader.get_column_headers(0)
        assert len(headers) == 3
        assert all("DSSCol" in h for h in headers)

    def test_hdf5_no_column_headers_no_dss_infer_from_shape(self, tmp_path: Path) -> None:
        """Column headers inferred from data shape, generic names generated."""
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(
            p,
            include_column_headers=False,
            include_dss_pathnames=False,
            include_n_columns=False,
            n_columns=4,
        )
        reader = BudgetReader(p)
        headers = reader.get_column_headers(0)
        assert len(headers) == 4
        assert headers[0] == "Column 1"

    def test_hdf5_timestep_count_from_data(self, tmp_path: Path) -> None:
        """Timestep count inferred from data shape when not in attributes."""
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(p, include_n_timesteps=False, n_timesteps=7)
        reader = BudgetReader(p)
        assert reader.n_timesteps == 7

    def test_hdf5_timestep_count_from_group_data(self, tmp_path: Path) -> None:
        """Timestep count inferred from Group child dataset."""
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(
            p,
            include_n_timesteps=False,
            n_timesteps=5,
            store_as_group=True,
        )
        reader = BudgetReader(p)
        assert reader.n_timesteps == 5

    def test_hdf5_monthly_unit(self, tmp_path: Path) -> None:
        """Monthly timestep unit triggers relativedelta-based time array."""
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(p, unit="1MON", n_timesteps=4)
        reader = BudgetReader(p)
        times, values = reader.get_values(0)
        assert len(times) == 4
        # Verify the time intervals are approximately 1 month apart
        dt0 = datetime.fromtimestamp(times[0])
        dt1 = datetime.fromtimestamp(times[1])
        assert 28 <= (dt1 - dt0).days <= 31

    def test_hdf5_no_start_datetime(self, tmp_path: Path) -> None:
        """No start_datetime -> numeric time array."""
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(p, include_begin_datetime=False)
        reader = BudgetReader(p)
        times, values = reader.get_values(0)
        # Should use arange * delta_t + start_time
        assert times[0] == 0.0
        assert times[1] == 1.0


# =============================================================================
# Tests for _h5_get static method
# =============================================================================


class TestH5Get:
    """Tests for _h5_get static method."""

    def test_get_scalar_dataset(self, tmp_path: Path) -> None:
        p = tmp_path / "test.hdf"
        with h5py.File(p, "w") as f:
            f.create_dataset("scalar", data=42)
            result = BudgetReader._h5_get(f, "scalar")
            assert result == 42

    def test_get_array_dataset(self, tmp_path: Path) -> None:
        p = tmp_path / "test.hdf"
        with h5py.File(p, "w") as f:
            f.create_dataset("arr", data=[1, 2, 3])
            result = BudgetReader._h5_get(f, "arr")
            np.testing.assert_array_equal(result, [1, 2, 3])

    def test_get_group(self, tmp_path: Path) -> None:
        p = tmp_path / "test.hdf"
        with h5py.File(p, "w") as f:
            grp = f.create_group("mygroup")
            result = BudgetReader._h5_get(f, "mygroup")
            assert isinstance(result, h5py.Group)

    def test_get_attr(self, tmp_path: Path) -> None:
        p = tmp_path / "test.hdf"
        with h5py.File(p, "w") as f:
            f.attrs["myattr"] = 99
            result = BudgetReader._h5_get(f, "myattr")
            assert result == 99

    def test_get_missing_returns_none(self, tmp_path: Path) -> None:
        p = tmp_path / "test.hdf"
        with h5py.File(p, "w") as f:
            result = BudgetReader._h5_get(f, "nonexistent")
            assert result is None


# =============================================================================
# Tests for binary header reading
# =============================================================================


class TestReadHeaderBinary:
    """Tests for _read_header_binary."""

    def test_basic_binary_read(self, tmp_path: Path) -> None:
        p = tmp_path / "budget.bin"
        _create_binary_budget(p)
        reader = BudgetReader(p)
        assert reader.format == "binary"
        assert reader.descriptor == "GROUNDWATER BUDGET"
        assert reader.n_locations == 2
        assert reader.n_timesteps == 3

    def test_binary_no_track_time(self, tmp_path: Path) -> None:
        p = tmp_path / "budget.bin"
        _create_binary_budget(p, track_time=False)
        reader = BudgetReader(p)
        assert reader.header.timestep.track_time is False
        assert reader.header.timestep.start_datetime is None

    def test_binary_with_areas(self, tmp_path: Path) -> None:
        p = tmp_path / "budget.bin"
        _create_binary_budget(p, n_areas=3, areas=[100.0, 200.0, 300.0])
        reader = BudgetReader(p)
        assert reader.header.n_areas == 3
        np.testing.assert_array_almost_equal(
            reader.header.areas, [100.0, 200.0, 300.0]
        )

    def test_binary_no_areas(self, tmp_path: Path) -> None:
        p = tmp_path / "budget.bin"
        _create_binary_budget(p, n_areas=0, areas=[])
        reader = BudgetReader(p)
        assert reader.header.n_areas == 0

    def test_binary_file_position_set(self, tmp_path: Path) -> None:
        p = tmp_path / "budget.bin"
        _create_binary_budget(p)
        reader = BudgetReader(p)
        assert reader.header.file_position > 0

    def test_binary_location_data(self, tmp_path: Path) -> None:
        p = tmp_path / "budget.bin"
        _create_binary_budget(p, n_columns=4)
        reader = BudgetReader(p)
        assert len(reader.header.location_data) == 1
        assert reader.header.location_data[0].n_columns == 4

    def test_binary_multiple_titles(self, tmp_path: Path) -> None:
        p = tmp_path / "budget.bin"
        _create_binary_budget(
            p,
            n_titles=2,
            titles=["Title Line 1", "Title Line 2"],
        )
        reader = BudgetReader(p)
        assert reader.header.ascii_output.n_titles == 2
        assert len(reader.header.ascii_output.titles) == 2


# =============================================================================
# Tests for properties
# =============================================================================


class TestBudgetReaderProperties:
    """Tests for BudgetReader properties."""

    def test_descriptor_property(self, tmp_path: Path) -> None:
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(p, descriptor="STREAM BUDGET")
        reader = BudgetReader(p)
        assert reader.descriptor == "STREAM BUDGET"

    def test_locations_property(self, tmp_path: Path) -> None:
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(p, location_names=["Reach 1", "Reach 2"])
        reader = BudgetReader(p)
        assert reader.locations == ["Reach 1", "Reach 2"]

    def test_n_locations_property(self, tmp_path: Path) -> None:
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(p, n_locations=5, location_names=[f"L{i}" for i in range(5)])
        reader = BudgetReader(p)
        assert reader.n_locations == 5

    def test_n_timesteps_property(self, tmp_path: Path) -> None:
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(p, n_timesteps=10)
        reader = BudgetReader(p)
        assert reader.n_timesteps == 10


# =============================================================================
# Tests for get_location_index
# =============================================================================


class TestGetLocationIndex:
    """Tests for get_location_index."""

    def test_index_by_int(self, tmp_path: Path) -> None:
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(p, n_locations=3, location_names=["A", "B", "C"])
        reader = BudgetReader(p)
        assert reader.get_location_index(0) == 0
        assert reader.get_location_index(2) == 2

    def test_index_by_int_out_of_range(self, tmp_path: Path) -> None:
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(p, n_locations=2, location_names=["A", "B"])
        reader = BudgetReader(p)
        with pytest.raises(IndexError, match="out of range"):
            reader.get_location_index(5)

    def test_index_by_int_negative(self, tmp_path: Path) -> None:
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(p, n_locations=2, location_names=["A", "B"])
        reader = BudgetReader(p)
        with pytest.raises(IndexError, match="out of range"):
            reader.get_location_index(-1)

    def test_index_by_name_exact(self, tmp_path: Path) -> None:
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(p, n_locations=2, location_names=["Alpha", "Beta"])
        reader = BudgetReader(p)
        assert reader.get_location_index("Alpha") == 0
        assert reader.get_location_index("Beta") == 1

    def test_index_by_name_case_insensitive(self, tmp_path: Path) -> None:
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(p, n_locations=2, location_names=["Alpha", "Beta"])
        reader = BudgetReader(p)
        assert reader.get_location_index("alpha") == 0
        assert reader.get_location_index("BETA") == 1

    def test_index_by_name_not_found(self, tmp_path: Path) -> None:
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(p, n_locations=2, location_names=["Alpha", "Beta"])
        reader = BudgetReader(p)
        with pytest.raises(KeyError, match="not found"):
            reader.get_location_index("Gamma")


# =============================================================================
# Tests for get_column_headers
# =============================================================================


class TestGetColumnHeaders:
    """Tests for get_column_headers."""

    def test_basic_column_headers(self, tmp_path: Path) -> None:
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(p, n_columns=3)
        reader = BudgetReader(p)
        headers = reader.get_column_headers(0)
        assert len(headers) == 3

    def test_unit_marker_replacement(self, tmp_path: Path) -> None:
        """UNIT_MARKERS in column headers are replaced."""
        p = tmp_path / "budget.hdf"
        with h5py.File(p, "w") as f:
            attrs_g = f.create_group("Attributes")
            attrs_g.attrs["Descriptor"] = b"TEST"
            attrs_g.attrs["NTimeSteps"] = 2
            attrs_g.attrs["TrackTime"] = 1
            attrs_g.attrs["DeltaT"] = 1.0
            attrs_g.attrs["DeltaT_InMinutes"] = 1440
            attrs_g.attrs["Unit"] = b"1DAY"
            attrs_g.attrs["BeginTime"] = 0.0
            attrs_g.attrs["BeginDateAndTime"] = b"10/01/2000_24:00"
            attrs_g.attrs["NAreas"] = 1
            attrs_g.attrs["Areas"] = [100.0]
            attrs_g.attrs["TitleLen"] = 160
            attrs_g.attrs["NTitles"] = 0
            attrs_g.attrs["NLocations"] = 1
            attrs_g.attrs["cLocationNames"] = [b"Loc1"]
            attrs_g.attrs["NLocationData"] = 1
            attrs_g.attrs["NDataColumns"] = 2
            attrs_g.attrs["cFullColumnHeaders"] = [
                b"Deep Perc @UNITVL@",
                b"Area @UNITAR@",
            ]
            attrs_g.attrs["iDataColumnTypes"] = [1, 4]
            attrs_g.attrs["iColWidth"] = [12, 12]
            f.create_dataset("Loc1", data=np.ones((2, 2)))
        reader = BudgetReader(p)
        headers = reader.get_column_headers(0)
        assert "(volume)" in headers[0]
        assert "(area)" in headers[1]

    def test_time_column_stripped(self, tmp_path: Path) -> None:
        """When 'Time' is first header and causes mismatch, it's stripped."""
        p = tmp_path / "budget.hdf"
        with h5py.File(p, "w") as f:
            attrs_g = f.create_group("Attributes")
            attrs_g.attrs["Descriptor"] = b"TEST"
            attrs_g.attrs["NTimeSteps"] = 2
            attrs_g.attrs["TrackTime"] = 1
            attrs_g.attrs["DeltaT"] = 1.0
            attrs_g.attrs["DeltaT_InMinutes"] = 1440
            attrs_g.attrs["Unit"] = b"1DAY"
            attrs_g.attrs["BeginTime"] = 0.0
            attrs_g.attrs["BeginDateAndTime"] = b"10/01/2000_24:00"
            attrs_g.attrs["NAreas"] = 1
            attrs_g.attrs["Areas"] = [100.0]
            attrs_g.attrs["TitleLen"] = 160
            attrs_g.attrs["NTitles"] = 0
            attrs_g.attrs["NLocations"] = 1
            attrs_g.attrs["cLocationNames"] = [b"Loc1"]
            attrs_g.attrs["NLocationData"] = 1
            attrs_g.attrs["NDataColumns"] = 2  # only 2 data columns
            # 3 headers where first is "Time" -> 3 > 2, and first is "time"
            attrs_g.attrs["cFullColumnHeaders"] = [b"Time", b"Col A", b"Col B"]
            attrs_g.attrs["iDataColumnTypes"] = [1, 1]
            attrs_g.attrs["iColWidth"] = [12, 12]
            f.create_dataset("Loc1", data=np.ones((2, 2)))
        reader = BudgetReader(p)
        headers = reader.get_column_headers(0)
        assert len(headers) == 2
        assert headers[0] == "Col A"

    def test_column_headers_multiple_location_data(self, tmp_path: Path) -> None:
        """Test headers when there are multiple location data entries."""
        p = tmp_path / "budget.hdf"
        with h5py.File(p, "w") as f:
            attrs_g = f.create_group("Attributes")
            attrs_g.attrs["Descriptor"] = b"TEST"
            attrs_g.attrs["NTimeSteps"] = 2
            attrs_g.attrs["TrackTime"] = 1
            attrs_g.attrs["DeltaT"] = 1.0
            attrs_g.attrs["DeltaT_InMinutes"] = 1440
            attrs_g.attrs["Unit"] = b"1DAY"
            attrs_g.attrs["BeginTime"] = 0.0
            attrs_g.attrs["BeginDateAndTime"] = b"10/01/2000_24:00"
            attrs_g.attrs["NAreas"] = 2
            attrs_g.attrs["Areas"] = [100.0, 200.0]
            attrs_g.attrs["TitleLen"] = 160
            attrs_g.attrs["NTitles"] = 0
            attrs_g.attrs["NLocations"] = 2
            attrs_g.attrs["cLocationNames"] = [b"Loc1", b"Loc2"]
            attrs_g.attrs["NLocationData"] = 2
            # LocationData1
            attrs_g.attrs["LocationData1%NDataColumns"] = 2
            attrs_g.attrs["LocationData1%cFullColumnHeaders"] = [b"A1", b"B1"]
            attrs_g.attrs["LocationData1%iDataColumnTypes"] = [1, 1]
            attrs_g.attrs["LocationData1%iColWidth"] = [12, 12]
            # LocationData2
            attrs_g.attrs["LocationData2%NDataColumns"] = 3
            attrs_g.attrs["LocationData2%cFullColumnHeaders"] = [b"X2", b"Y2", b"Z2"]
            attrs_g.attrs["LocationData2%iDataColumnTypes"] = [1, 1, 1]
            attrs_g.attrs["LocationData2%iColWidth"] = [12, 12, 12]
            f.create_dataset("Loc1", data=np.ones((2, 2)))
            f.create_dataset("Loc2", data=np.ones((2, 3)))
        reader = BudgetReader(p)
        h1 = reader.get_column_headers(0)
        h2 = reader.get_column_headers(1)
        assert h1 == ["A1", "B1"]
        assert h2 == ["X2", "Y2", "Z2"]


# =============================================================================
# Tests for get_values (HDF5)
# =============================================================================


class TestGetValuesHDF5:
    """Tests for get_values with HDF5 files."""

    def test_get_values_basic(self, tmp_path: Path) -> None:
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(p, n_timesteps=3, n_columns=2, n_locations=1,
                            location_names=["Loc1"])
        reader = BudgetReader(p)
        times, values = reader.get_values(0)
        assert times.shape == (3,)
        assert values.shape == (3, 2)

    def test_get_values_with_column_selection(self, tmp_path: Path) -> None:
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(p, n_columns=4, n_locations=1, location_names=["Loc1"])
        reader = BudgetReader(p)
        times, values = reader.get_values(0, columns=[0, 2])
        assert values.shape[1] == 2

    def test_get_values_missing_location(self, tmp_path: Path) -> None:
        """Location name in header but not in file raises KeyError."""
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(p, n_locations=1, location_names=["Loc1"])
        reader = BudgetReader(p)
        # Manually modify location names to simulate mismatch
        reader.header.location_names = ["NonExistent"]
        with pytest.raises(KeyError, match="not found in HDF5"):
            reader.get_values(0)

    def test_get_values_no_data_in_group(self, tmp_path: Path) -> None:
        """Group exists but has no Dataset children."""
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(p, n_locations=1, location_names=["Loc1"],
                            store_as_group=True)
        reader = BudgetReader(p)
        # Delete the dataset within the group
        with h5py.File(p, "a") as f:
            del f["Loc1/Data"]
        with pytest.raises(ValueError, match="No data found"):
            reader.get_values(0)

    def test_get_values_1d_data(self, tmp_path: Path) -> None:
        """1D data is reshaped to 2D."""
        p = tmp_path / "budget.hdf"
        with h5py.File(p, "w") as f:
            attrs_g = f.create_group("Attributes")
            attrs_g.attrs["Descriptor"] = b"TEST"
            attrs_g.attrs["NTimeSteps"] = 5
            attrs_g.attrs["TrackTime"] = 0
            attrs_g.attrs["DeltaT"] = 1.0
            attrs_g.attrs["DeltaT_InMinutes"] = 1440
            attrs_g.attrs["Unit"] = b"1DAY"
            attrs_g.attrs["BeginTime"] = 0.0
            attrs_g.attrs["NAreas"] = 1
            attrs_g.attrs["Areas"] = [100.0]
            attrs_g.attrs["TitleLen"] = 160
            attrs_g.attrs["NTitles"] = 0
            attrs_g.attrs["NLocations"] = 1
            attrs_g.attrs["cLocationNames"] = [b"Loc1"]
            attrs_g.attrs["NLocationData"] = 1
            attrs_g.attrs["NDataColumns"] = 1
            attrs_g.attrs["cFullColumnHeaders"] = [b"Value"]
            attrs_g.attrs["iDataColumnTypes"] = [1]
            attrs_g.attrs["iColWidth"] = [12]
            # Write 1D data
            f.create_dataset("Loc1", data=np.arange(5, dtype=np.float64))
        reader = BudgetReader(p)
        times, values = reader.get_values(0)
        assert values.ndim == 2
        assert values.shape == (5, 1)

    def test_get_values_by_location_name(self, tmp_path: Path) -> None:
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(
            p,
            n_locations=2,
            location_names=["Alpha", "Beta"],
            n_timesteps=3,
            n_columns=2,
        )
        reader = BudgetReader(p)
        times, values = reader.get_values("Beta")
        assert values.shape == (3, 2)


# =============================================================================
# Tests for get_values (binary)
# =============================================================================


class TestGetValuesBinary:
    """Tests for get_values with binary files."""

    def test_get_values_basic_binary(self, tmp_path: Path) -> None:
        p = tmp_path / "budget.bin"
        _create_binary_budget(p, n_timesteps=3, n_columns=3, n_locations=2)
        reader = BudgetReader(p)
        times, values = reader.get_values(0)
        assert times.shape == (3,)
        assert values.shape == (3, 3)
        # First timestep, first location: (1)*100 + (1)*10 + col
        assert values[0, 0] == pytest.approx(111.0)
        assert values[0, 1] == pytest.approx(112.0)
        assert values[0, 2] == pytest.approx(113.0)

    def test_get_values_second_location_binary(self, tmp_path: Path) -> None:
        p = tmp_path / "budget.bin"
        _create_binary_budget(p, n_timesteps=2, n_columns=2, n_locations=2)
        reader = BudgetReader(p)
        times, values = reader.get_values(1)
        # Second location: (t+1)*100 + (2)*10 + (col+1)
        assert values[0, 0] == pytest.approx(121.0)
        assert values[0, 1] == pytest.approx(122.0)

    def test_get_values_column_selection_binary(self, tmp_path: Path) -> None:
        p = tmp_path / "budget.bin"
        _create_binary_budget(p, n_timesteps=2, n_columns=4, n_locations=1)
        reader = BudgetReader(p)
        times, values = reader.get_values(0, columns=[1, 3])
        assert values.shape == (2, 2)

    def test_get_values_no_start_datetime_binary(self, tmp_path: Path) -> None:
        """No start_datetime -> numeric time array."""
        p = tmp_path / "budget.bin"
        _create_binary_budget(p, track_time=False, n_timesteps=3)
        reader = BudgetReader(p)
        times, values = reader.get_values(0)
        assert times[0] == 0.0
        assert times[1] == 1.0

    def test_get_values_monthly_binary(self, tmp_path: Path) -> None:
        """Monthly unit triggers relativedelta in binary reader."""
        p = tmp_path / "budget.bin"
        _create_binary_budget(
            p,
            unit="1MON",
            n_timesteps=3,
            n_columns=2,
            n_locations=1,
        )
        reader = BudgetReader(p)
        times, values = reader.get_values(0)
        assert len(times) == 3


# =============================================================================
# Tests for get_dataframe
# =============================================================================


class TestGetDataframe:
    """Tests for get_dataframe."""

    def test_basic_dataframe(self, tmp_path: Path) -> None:
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(p, n_timesteps=3, n_columns=2, n_locations=1,
                            location_names=["Loc1"])
        reader = BudgetReader(p)
        df = reader.get_dataframe(0)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 2)
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_dataframe_no_datetime_index(self, tmp_path: Path) -> None:
        """When no start_datetime, index is numeric."""
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(p, include_begin_datetime=False, n_timesteps=3,
                            n_columns=2, n_locations=1, location_names=["Loc1"])
        reader = BudgetReader(p)
        df = reader.get_dataframe(0)
        assert not isinstance(df.index, pd.DatetimeIndex)

    def test_dataframe_column_by_name(self, tmp_path: Path) -> None:
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(p, n_columns=3, n_locations=1, location_names=["Loc1"])
        reader = BudgetReader(p)
        df = reader.get_dataframe(0, columns=["Col 1", "Col 3"])
        assert df.shape[1] == 2
        assert list(df.columns) == ["Col 1", "Col 3"]

    def test_dataframe_column_by_name_case_insensitive(self, tmp_path: Path) -> None:
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(p, n_columns=3, n_locations=1, location_names=["Loc1"])
        reader = BudgetReader(p)
        df = reader.get_dataframe(0, columns=["col 2"])
        assert df.shape[1] == 1

    def test_dataframe_column_by_index(self, tmp_path: Path) -> None:
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(p, n_columns=4, n_locations=1, location_names=["Loc1"])
        reader = BudgetReader(p)
        df = reader.get_dataframe(0, columns=[0, 2])
        assert df.shape[1] == 2

    def test_dataframe_column_not_found(self, tmp_path: Path) -> None:
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(p, n_columns=2, n_locations=1, location_names=["Loc1"])
        reader = BudgetReader(p)
        with pytest.raises(KeyError, match="not found"):
            reader.get_dataframe(0, columns=["NonExistent"])

    def test_dataframe_by_location_name(self, tmp_path: Path) -> None:
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(
            p,
            n_locations=2,
            location_names=["Alpha", "Beta"],
            n_columns=2,
        )
        reader = BudgetReader(p)
        df = reader.get_dataframe("Alpha")
        assert isinstance(df, pd.DataFrame)
        assert df.shape[1] == 2


# =============================================================================
# Tests for get_all_dataframes
# =============================================================================


class TestGetAllDataframes:
    """Tests for get_all_dataframes."""

    def test_all_dataframes(self, tmp_path: Path) -> None:
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(
            p,
            n_locations=3,
            location_names=["A", "B", "C"],
            n_columns=2,
            n_timesteps=4,
        )
        reader = BudgetReader(p)
        dfs = reader.get_all_dataframes()
        assert len(dfs) == 3
        assert "A" in dfs
        assert "B" in dfs
        assert "C" in dfs
        for name, df in dfs.items():
            assert isinstance(df, pd.DataFrame)
            assert df.shape == (4, 2)


# =============================================================================
# Tests for get_monthly_averages
# =============================================================================


class TestGetMonthlyAverages:
    """Tests for get_monthly_averages."""

    def test_monthly_averages_datetime_index(self, tmp_path: Path) -> None:
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(
            p,
            n_timesteps=60,
            n_columns=2,
            n_locations=1,
            location_names=["Loc1"],
            unit="1DAY",
        )
        reader = BudgetReader(p)
        df = reader.get_monthly_averages(0)
        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_monthly_averages_no_datetime_index(self, tmp_path: Path) -> None:
        """Non-datetime index -> return as-is."""
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(
            p,
            n_timesteps=5,
            n_columns=2,
            n_locations=1,
            location_names=["Loc1"],
            include_begin_datetime=False,
        )
        reader = BudgetReader(p)
        df = reader.get_monthly_averages(0)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (5, 2)


# =============================================================================
# Tests for get_annual_totals
# =============================================================================


class TestGetAnnualTotals:
    """Tests for get_annual_totals."""

    def test_annual_totals_datetime_index(self, tmp_path: Path) -> None:
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(
            p,
            n_timesteps=365,
            n_columns=2,
            n_locations=1,
            location_names=["Loc1"],
            unit="1DAY",
        )
        reader = BudgetReader(p)
        df = reader.get_annual_totals(0)
        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_annual_totals_no_datetime_12plus_rows(self, tmp_path: Path) -> None:
        """Non-datetime with >= 12 rows -> group by year (12 per year)."""
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(
            p,
            n_timesteps=24,
            n_columns=2,
            n_locations=1,
            location_names=["Loc1"],
            include_begin_datetime=False,
        )
        reader = BudgetReader(p)
        df = reader.get_annual_totals(0)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2  # 24 / 12 = 2 years

    def test_annual_totals_no_datetime_fewer_than_12(self, tmp_path: Path) -> None:
        """Non-datetime with <12 rows -> single row sum."""
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(
            p,
            n_timesteps=5,
            n_columns=2,
            n_locations=1,
            location_names=["Loc1"],
            include_begin_datetime=False,
        )
        reader = BudgetReader(p)
        df = reader.get_annual_totals(0)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1  # single row sum


# =============================================================================
# Tests for get_cumulative
# =============================================================================


class TestGetCumulative:
    """Tests for get_cumulative."""

    def test_cumulative(self, tmp_path: Path) -> None:
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(
            p,
            n_timesteps=5,
            n_columns=2,
            n_locations=1,
            location_names=["Loc1"],
        )
        reader = BudgetReader(p)
        df = reader.get_cumulative(0)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (5, 2)
        # Cumulative sum should be monotonically non-decreasing for positive data
        for col in df.columns:
            diffs = df[col].diff().dropna()
            assert (diffs >= 0).all()


# =============================================================================
# Tests for __repr__
# =============================================================================


class TestRepr:
    """Tests for BudgetReader __repr__."""

    def test_repr_hdf5(self, tmp_path: Path) -> None:
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(p, descriptor="GW BUDGET", n_locations=3,
                            location_names=["A", "B", "C"], n_timesteps=10)
        reader = BudgetReader(p)
        r = repr(reader)
        assert "budget.hdf" in r
        assert "hdf5" in r
        assert "GW BUDGET" in r
        assert "n_locations=3" in r
        assert "n_timesteps=10" in r

    def test_repr_binary(self, tmp_path: Path) -> None:
        p = tmp_path / "budget.bin"
        _create_binary_budget(p)
        reader = BudgetReader(p)
        r = repr(reader)
        assert "budget.bin" in r
        assert "binary" in r
        assert "GROUNDWATER BUDGET" in r


# =============================================================================
# Tests for multiple location data in binary reader
# =============================================================================


class TestBinaryMultipleLocationData:
    """Tests for binary reader with multiple location data entries."""

    def test_binary_multiple_loc_data(self, tmp_path: Path) -> None:
        """Binary file with n_loc_data > 1 (multiple location data entries)."""
        p = tmp_path / "budget.bin"
        n_locations = 2
        n_timesteps = 2
        # We need to create a custom binary file with n_loc_data=2
        with open(p, "wb") as f:
            _write_fortran_string(f, "MULTI LOC DATA BUDGET", 100)
            _write_fortran_int(f, n_timesteps)
            _write_fortran_logical(f, False)  # no track_time
            _write_fortran_real8(f, 1.0)
            _write_fortran_int(f, 1440)
            _write_fortran_string(f, "1DAY", 10)
            # No datetime string since track_time=False
            _write_fortran_real8(f, 0.0)

            # Areas
            _write_fortran_int(f, 0)

            # ASCII output
            _write_fortran_int(f, 160)  # title_len
            _write_fortran_int(f, 1)  # n_titles
            _write_fortran_string(f, "Title", 1000)
            _write_fortran_logical(f, True)  # persist
            _write_fortran_string(f, "(format)", 500)
            n_col_header_lines = 2
            _write_fortran_int(f, n_col_header_lines)

            # Locations
            _write_fortran_int(f, n_locations)
            _write_fortran_string(f, "Loc A", 100)
            _write_fortran_string(f, "Loc B", 100)

            # n_loc_data = 2
            _write_fortran_int(f, 2)

            # Location data 1: 2 columns
            n_cols_1 = 2
            storage_1 = 2
            _write_fortran_int(f, n_cols_1)
            _write_fortran_int(f, storage_1)
            _write_fortran_string(f, "ColA1", 100)
            _write_fortran_string(f, "ColA2", 100)
            _write_fortran_int_array(f, [1, 1])
            _write_fortran_int_array(f, [12, 12])
            for _ in range(n_col_header_lines):
                for _ in range(n_cols_1):
                    _write_fortran_string(f, "", 100)
            for _ in range(n_col_header_lines):
                _write_fortran_string(f, "", 500)

            # Location data 2: 3 columns
            n_cols_2 = 3
            storage_2 = 3
            _write_fortran_int(f, n_cols_2)
            _write_fortran_int(f, storage_2)
            _write_fortran_string(f, "ColB1", 100)
            _write_fortran_string(f, "ColB2", 100)
            _write_fortran_string(f, "ColB3", 100)
            _write_fortran_int_array(f, [1, 1, 1])
            _write_fortran_int_array(f, [12, 12, 12])
            for _ in range(n_col_header_lines):
                for _ in range(n_cols_2):
                    _write_fortran_string(f, "", 100)
            for _ in range(n_col_header_lines):
                _write_fortran_string(f, "", 500)

            # DSS output
            _write_fortran_int(f, 0)
            _write_fortran_int(f, 0)
            _write_fortran_int_array(f, [])

            # Data: total_storage = storage_1 + storage_2 = 5 per timestep
            for t in range(n_timesteps):
                # Loc A: 2 values
                for c in range(n_cols_1):
                    f.write(struct.pack("d", (t + 1) * 10.0 + c + 1))
                # Loc B: 3 values
                for c in range(n_cols_2):
                    f.write(struct.pack("d", (t + 1) * 100.0 + c + 1))

        reader = BudgetReader(p)
        assert reader.n_locations == 2
        assert len(reader.header.location_data) == 2
        assert reader.header.location_data[0].n_columns == 2
        assert reader.header.location_data[1].n_columns == 3

        # Read location A
        times_a, values_a = reader.get_values(0)
        assert values_a.shape == (2, 2)
        assert values_a[0, 0] == pytest.approx(11.0)

        # Read location B
        times_b, values_b = reader.get_values(1)
        assert values_b.shape == (2, 3)
        assert values_b[0, 0] == pytest.approx(101.0)


# =============================================================================
# Tests for DSS pathnames with empty int array edge case
# =============================================================================


class TestBinaryDSSOutput:
    """Tests for DSS output reading in binary files."""

    def test_binary_with_dss_pathnames(self, tmp_path: Path) -> None:
        """Binary file with DSS pathnames."""
        p = tmp_path / "budget.bin"
        with open(p, "wb") as f:
            _write_fortran_string(f, "DSS TEST", 100)
            _write_fortran_int(f, 2)  # n_timesteps
            _write_fortran_logical(f, False)
            _write_fortran_real8(f, 1.0)
            _write_fortran_int(f, 1440)
            _write_fortran_string(f, "1DAY", 10)
            _write_fortran_real8(f, 0.0)
            _write_fortran_int(f, 0)  # n_areas

            # ASCII output
            _write_fortran_int(f, 160)
            _write_fortran_int(f, 0)  # no titles
            _write_fortran_string(f, "(fmt)", 500)
            _write_fortran_int(f, 1)  # n_column_header_lines

            # Location
            _write_fortran_int(f, 1)
            _write_fortran_string(f, "Loc1", 100)

            # Location data
            _write_fortran_int(f, 1)  # n_loc_data
            n_cols = 2
            _write_fortran_int(f, n_cols)
            _write_fortran_int(f, n_cols)  # storage_units
            _write_fortran_string(f, "C1", 100)
            _write_fortran_string(f, "C2", 100)
            _write_fortran_int_array(f, [1, 1])
            _write_fortran_int_array(f, [12, 12])
            for _ in range(1):
                for _ in range(n_cols):
                    _write_fortran_string(f, "", 100)
            for _ in range(1):
                _write_fortran_string(f, "", 500)

            # DSS output: 2 pathnames
            _write_fortran_int(f, 2)
            _write_fortran_string(f, "/A/B/C/D/E/F/", 80)
            _write_fortran_string(f, "/A/B/C/D/E/G/", 80)
            # DSS types: 2
            _write_fortran_int(f, 2)
            _write_fortran_int_array(f, [1, 2])

            # Data
            for t in range(2):
                for c in range(n_cols):
                    f.write(struct.pack("d", float(t * 10 + c)))

        reader = BudgetReader(p)
        assert reader.descriptor == "DSS TEST"
        times, values = reader.get_values(0)
        assert values.shape == (2, 2)


# =============================================================================
# Tests for HDF5 with data stored as group and timestep inference
# =============================================================================


class TestHDF5EdgeCases:
    """Test edge cases in HDF5 reading."""

    def test_hdf5_timestep_from_transposed_dataset(self, tmp_path: Path) -> None:
        """n_timesteps inferred from transposed dataset shape."""
        p = tmp_path / "budget.hdf"
        with h5py.File(p, "w") as f:
            attrs_g = f.create_group("Attributes")
            attrs_g.attrs["Descriptor"] = b"TEST"
            # No NTimeSteps
            attrs_g.attrs["TrackTime"] = 0
            attrs_g.attrs["DeltaT"] = 1.0
            attrs_g.attrs["DeltaT_InMinutes"] = 1440
            attrs_g.attrs["Unit"] = b"1DAY"
            attrs_g.attrs["BeginTime"] = 0.0
            attrs_g.attrs["NAreas"] = 1
            attrs_g.attrs["Areas"] = [100.0]
            attrs_g.attrs["TitleLen"] = 160
            attrs_g.attrs["NTitles"] = 0
            attrs_g.attrs["NLocations"] = 1
            attrs_g.attrs["cLocationNames"] = [b"Loc1"]
            attrs_g.attrs["NLocationData"] = 1
            attrs_g.attrs["NDataColumns"] = 3
            attrs_g.attrs["cFullColumnHeaders"] = [b"A", b"B", b"C"]
            attrs_g.attrs["iDataColumnTypes"] = [1, 1, 1]
            attrs_g.attrs["iColWidth"] = [12, 12, 12]
            # Data shape is (3, 7) -> n_columns=3 matches shape[0], so n_timesteps=7
            f.create_dataset("Loc1", data=np.ones((3, 7)))
        reader = BudgetReader(p)
        assert reader.n_timesteps == 7

    def test_hdf5_timestep_from_group_transposed(self, tmp_path: Path) -> None:
        """n_timesteps inferred from Group child dataset, transposed orientation."""
        p = tmp_path / "budget.hdf"
        with h5py.File(p, "w") as f:
            attrs_g = f.create_group("Attributes")
            attrs_g.attrs["Descriptor"] = b"TEST"
            attrs_g.attrs["TrackTime"] = 0
            attrs_g.attrs["DeltaT"] = 1.0
            attrs_g.attrs["DeltaT_InMinutes"] = 1440
            attrs_g.attrs["Unit"] = b"1DAY"
            attrs_g.attrs["BeginTime"] = 0.0
            attrs_g.attrs["NAreas"] = 1
            attrs_g.attrs["Areas"] = [100.0]
            attrs_g.attrs["TitleLen"] = 160
            attrs_g.attrs["NTitles"] = 0
            attrs_g.attrs["NLocations"] = 1
            attrs_g.attrs["cLocationNames"] = [b"Loc1"]
            attrs_g.attrs["NLocationData"] = 1
            attrs_g.attrs["NDataColumns"] = 4
            attrs_g.attrs["cFullColumnHeaders"] = [b"A", b"B", b"C", b"D"]
            attrs_g.attrs["iDataColumnTypes"] = [1, 1, 1, 1]
            attrs_g.attrs["iColWidth"] = [12, 12, 12, 12]
            # Group with transposed data: (4, 10) where 4=n_columns
            grp = f.create_group("Loc1")
            grp.create_dataset("Data", data=np.ones((4, 10)))
        reader = BudgetReader(p)
        assert reader.n_timesteps == 10

    def test_hdf5_no_areas(self, tmp_path: Path) -> None:
        """HDF5 without area attributes."""
        p = tmp_path / "budget.hdf"
        _create_hdf5_budget(p, include_areas=False)
        reader = BudgetReader(p)
        # n_areas should be 0 since include_areas=False doesn't write NAreas/Areas
        assert reader.header.n_areas == 0

    def test_hdf5_location_not_in_file_for_infer_ncols(self, tmp_path: Path) -> None:
        """Location names exist but none match datasets in file."""
        p = tmp_path / "budget.hdf"
        with h5py.File(p, "w") as f:
            attrs_g = f.create_group("Attributes")
            attrs_g.attrs["Descriptor"] = b"TEST"
            attrs_g.attrs["NTimeSteps"] = 2
            attrs_g.attrs["TrackTime"] = 0
            attrs_g.attrs["DeltaT"] = 1.0
            attrs_g.attrs["DeltaT_InMinutes"] = 1440
            attrs_g.attrs["Unit"] = b"1DAY"
            attrs_g.attrs["BeginTime"] = 0.0
            attrs_g.attrs["TitleLen"] = 160
            attrs_g.attrs["NTitles"] = 0
            attrs_g.attrs["NLocations"] = 1
            attrs_g.attrs["cLocationNames"] = [b"MissingLoc"]
            attrs_g.attrs["NLocationData"] = 1
            # No NDataColumns, no column headers, no DSS -> will try to infer from data
            # But the dataset name won't match "MissingLoc"
            f.create_dataset("OtherName", data=np.ones((2, 3)))
        reader = BudgetReader(p)
        # n_columns should remain 0 since no matching dataset found
        assert reader.header.location_data[0].n_columns == 0
