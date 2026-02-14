"""
Comprehensive tests for pyiwfm.io.budget module.

Tests cover:
- BudgetReader initialization and format detection
- Header parsing for HDF5 and binary formats
- Data reading and extraction
- DataFrame conversion
- Aggregation methods (monthly, annual, cumulative)
"""

import struct

import pytest
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, mock_open

from pyiwfm.io.budget import (
    BudgetReader,
    BudgetHeader,
    TimeStepInfo,
    ASCIIOutputInfo,
    LocationData,
    parse_iwfm_datetime,
    julian_to_datetime,
    excel_julian_to_datetime,
    BUDGET_DATA_TYPES,
    UNIT_MARKERS,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_hdf5_file(tmp_path):
    """Create a mock HDF5 budget file."""
    pytest.importorskip("h5py")
    import h5py

    filepath = tmp_path / "test_budget.hdf"

    with h5py.File(filepath, "w") as f:
        # Create Attributes group
        attrs = f.create_group("Attributes")
        attrs.create_dataset("Descriptor", data="GROUNDWATER BUDGET")
        attrs.create_dataset("NAreas", data=3)
        attrs.create_dataset("Areas", data=[1000.0, 2000.0, 3000.0])
        attrs.create_dataset("NTimeSteps", data=12)
        attrs.create_dataset("nLocations", data=3)
        attrs.create_dataset(
            "cLocationNames",
            data=[b"Subregion 1", b"Subregion 2", b"Subregion 3"]
        )
        attrs.create_dataset("NLocationData", data=1)
        attrs.create_dataset("NDataColumns", data=5)
        attrs.create_dataset(
            "cFullColumnHeaders",
            data=[
                b"Deep Percolation (@UNITVL@)",
                b"Pumping (@UNITVL@)",
                b"Net Stream-GW (@UNITVL@)",
                b"Subsidence (@UNITLT@)",
                b"Storage Change (@UNITVL@)",
            ]
        )
        attrs.create_dataset("iDataColumnTypes", data=[1, 1, 1, 5, 1])
        attrs.create_dataset("iColWidth", data=[15, 15, 15, 15, 15])

        # Create location data
        np.random.seed(42)
        for name in ["Subregion 1", "Subregion 2", "Subregion 3"]:
            loc_group = f.create_group(name)
            # Data shape: (n_columns, n_timesteps)
            data = np.random.rand(5, 12) * 1000
            loc_group.create_dataset("data", data=data)

    return filepath


@pytest.fixture
def sample_header():
    """Create a sample BudgetHeader."""
    return BudgetHeader(
        descriptor="GROUNDWATER BUDGET",
        timestep=TimeStepInfo(
            track_time=True,
            delta_t=1.0,
            delta_t_minutes=1440,
            unit="1DAY",
            start_datetime=datetime(2020, 1, 1),
            n_timesteps=12,
        ),
        n_areas=3,
        areas=np.array([1000.0, 2000.0, 3000.0]),
        n_locations=3,
        location_names=["Subregion 1", "Subregion 2", "Subregion 3"],
        location_data=[
            LocationData(
                n_columns=5,
                column_headers=[
                    "Deep Percolation",
                    "Pumping",
                    "Net Stream-GW",
                    "Subsidence",
                    "Storage Change",
                ],
                column_types=[1, 1, 1, 5, 1],
                column_widths=[15, 15, 15, 15, 15],
            )
        ],
    )


# =============================================================================
# DateTime Parsing Tests
# =============================================================================


class TestDateTimeParsing:
    """Tests for datetime parsing functions."""

    def test_parse_iwfm_datetime_standard(self):
        """Test parsing standard IWFM datetime."""
        dt = parse_iwfm_datetime("01/15/2020_12:30")
        assert dt == datetime(2020, 1, 15, 12, 30)

    def test_parse_iwfm_datetime_with_seconds(self):
        """Test parsing datetime with seconds."""
        dt = parse_iwfm_datetime("12/31/2020 23:59:59")
        assert dt == datetime(2020, 12, 31, 23, 59, 59)

    def test_parse_iwfm_datetime_date_only(self):
        """Test parsing date without time."""
        dt = parse_iwfm_datetime("06/15/2020")
        assert dt == datetime(2020, 6, 15)

    def test_parse_iwfm_datetime_empty(self):
        """Test parsing empty string returns None."""
        assert parse_iwfm_datetime("") is None
        assert parse_iwfm_datetime("   ") is None

    def test_parse_iwfm_datetime_invalid(self):
        """Test parsing invalid string returns None."""
        assert parse_iwfm_datetime("not a date") is None

    def testjulian_to_datetime(self):
        """Test Julian day conversion."""
        # J2000.0 = Julian day 2451545.0 = 2000-01-01 12:00:00
        dt = julian_to_datetime(2451545.0)
        assert dt.year == 2000
        assert dt.month == 1
        assert dt.day == 1

    def test_exceljulian_to_datetime(self):
        """Test Excel Julian day conversion."""
        # Excel day 1 = 1900-01-01
        dt = excel_julian_to_datetime(1.0)
        assert dt.year == 1899
        assert dt.month == 12
        assert dt.day == 31


# =============================================================================
# BudgetHeader Tests
# =============================================================================


class TestBudgetHeader:
    """Tests for BudgetHeader dataclass."""

    def test_header_creation(self, sample_header):
        """Test header creation with all fields."""
        assert sample_header.descriptor == "GROUNDWATER BUDGET"
        assert sample_header.n_locations == 3
        assert sample_header.timestep.n_timesteps == 12

    def test_header_defaults(self):
        """Test header with default values."""
        header = BudgetHeader()
        assert header.descriptor == ""
        assert header.n_locations == 0
        assert header.timestep.n_timesteps == 0


class TestTimeStepInfo:
    """Tests for TimeStepInfo dataclass."""

    def test_timestep_defaults(self):
        """Test default values."""
        ts = TimeStepInfo()
        assert ts.track_time is True
        assert ts.delta_t == 1.0
        assert ts.delta_t_minutes == 1440
        assert ts.unit == "1DAY"

    def test_timestep_custom(self):
        """Test custom values."""
        ts = TimeStepInfo(
            track_time=False,
            delta_t=0.5,
            delta_t_minutes=720,
            unit="12HR",
            n_timesteps=100,
        )
        assert ts.delta_t_minutes == 720
        assert ts.n_timesteps == 100


class TestLocationData:
    """Tests for LocationData dataclass."""

    def test_location_data_defaults(self):
        """Test default values."""
        loc = LocationData()
        assert loc.n_columns == 0
        assert loc.column_headers == []
        assert loc.column_types == []

    def test_location_data_custom(self):
        """Test custom values."""
        loc = LocationData(
            n_columns=3,
            column_headers=["Col1", "Col2", "Col3"],
            column_types=[1, 2, 3],
            column_widths=[10, 10, 10],
        )
        assert loc.n_columns == 3
        assert len(loc.column_headers) == 3


# =============================================================================
# BudgetReader Initialization Tests
# =============================================================================


class TestBudgetReaderInit:
    """Tests for BudgetReader initialization."""

    def test_init_hdf5_file(self, mock_hdf5_file):
        """Test initialization with HDF5 file."""
        reader = BudgetReader(mock_hdf5_file)

        assert reader.filepath == mock_hdf5_file
        assert reader.format == "hdf5"
        assert reader.descriptor == "GROUNDWATER BUDGET"

    def test_init_file_not_found(self, tmp_path):
        """Test initialization with non-existent file."""
        with pytest.raises(FileNotFoundError):
            BudgetReader(tmp_path / "nonexistent.hdf")

    def test_format_detection_by_extension(self, tmp_path):
        """Test format detection by file extension."""
        # Create dummy files
        hdf_file = tmp_path / "test.hdf"
        hdf_file.touch()

        bin_file = tmp_path / "test.bin"
        bin_file.touch()

        # HDF5 detection will fail without valid content
        # but we can test the extension-based detection
        with pytest.raises(Exception):
            reader = BudgetReader(hdf_file)


# =============================================================================
# BudgetReader Properties Tests
# =============================================================================


class TestBudgetReaderProperties:
    """Tests for BudgetReader properties."""

    def test_descriptor_property(self, mock_hdf5_file):
        """Test descriptor property."""
        reader = BudgetReader(mock_hdf5_file)
        assert reader.descriptor == "GROUNDWATER BUDGET"

    def test_locations_property(self, mock_hdf5_file):
        """Test locations property."""
        reader = BudgetReader(mock_hdf5_file)
        assert reader.locations == ["Subregion 1", "Subregion 2", "Subregion 3"]

    def test_n_locations_property(self, mock_hdf5_file):
        """Test n_locations property."""
        reader = BudgetReader(mock_hdf5_file)
        assert reader.n_locations == 3

    def test_n_timesteps_property(self, mock_hdf5_file):
        """Test n_timesteps property."""
        reader = BudgetReader(mock_hdf5_file)
        assert reader.n_timesteps == 12


# =============================================================================
# Location Index Tests
# =============================================================================


class TestLocationIndex:
    """Tests for location index resolution."""

    def test_get_location_index_by_int(self, mock_hdf5_file):
        """Test getting location by integer index."""
        reader = BudgetReader(mock_hdf5_file)

        assert reader.get_location_index(0) == 0
        assert reader.get_location_index(1) == 1
        assert reader.get_location_index(2) == 2

    def test_get_location_index_by_name(self, mock_hdf5_file):
        """Test getting location by name."""
        reader = BudgetReader(mock_hdf5_file)

        assert reader.get_location_index("Subregion 1") == 0
        assert reader.get_location_index("Subregion 2") == 1
        assert reader.get_location_index("Subregion 3") == 2

    def test_get_location_index_case_insensitive(self, mock_hdf5_file):
        """Test case-insensitive name matching."""
        reader = BudgetReader(mock_hdf5_file)

        assert reader.get_location_index("SUBREGION 1") == 0
        assert reader.get_location_index("subregion 1") == 0

    def test_get_location_index_invalid_int(self, mock_hdf5_file):
        """Test invalid integer index raises error."""
        reader = BudgetReader(mock_hdf5_file)

        with pytest.raises(IndexError):
            reader.get_location_index(10)

        with pytest.raises(IndexError):
            reader.get_location_index(-1)

    def test_get_location_index_invalid_name(self, mock_hdf5_file):
        """Test invalid name raises error."""
        reader = BudgetReader(mock_hdf5_file)

        with pytest.raises(KeyError):
            reader.get_location_index("Invalid Zone")


# =============================================================================
# Column Headers Tests
# =============================================================================


class TestColumnHeaders:
    """Tests for column header retrieval."""

    def test_get_column_headers(self, mock_hdf5_file):
        """Test getting column headers."""
        reader = BudgetReader(mock_hdf5_file)
        headers = reader.get_column_headers("Subregion 1")

        assert len(headers) == 5
        assert "Deep Percolation" in headers[0]
        assert "Pumping" in headers[1]

    def test_get_column_headers_by_index(self, mock_hdf5_file):
        """Test getting column headers by location index."""
        reader = BudgetReader(mock_hdf5_file)
        headers = reader.get_column_headers(0)

        assert len(headers) == 5

    def test_unit_marker_replacement(self, mock_hdf5_file):
        """Test that unit markers are replaced."""
        reader = BudgetReader(mock_hdf5_file)
        headers = reader.get_column_headers(0)

        # Unit markers should be replaced with (unit_type)
        assert "@UNITVL@" not in headers[0]
        assert "(volume)" in headers[0]


# =============================================================================
# Data Reading Tests
# =============================================================================


class TestDataReading:
    """Tests for data reading."""

    def test_get_values(self, mock_hdf5_file):
        """Test reading values."""
        reader = BudgetReader(mock_hdf5_file)
        times, values = reader.get_values("Subregion 1")

        assert len(times) == 12
        assert values.shape == (12, 5)

    def test_get_values_by_index(self, mock_hdf5_file):
        """Test reading values by location index."""
        reader = BudgetReader(mock_hdf5_file)
        times, values = reader.get_values(0)

        assert len(times) == 12
        assert values.shape == (12, 5)

    def test_get_values_specific_columns(self, mock_hdf5_file):
        """Test reading specific columns."""
        reader = BudgetReader(mock_hdf5_file)
        times, values = reader.get_values("Subregion 1", columns=[0, 2])

        assert values.shape == (12, 2)

    def test_values_are_numeric(self, mock_hdf5_file):
        """Test that values are numeric."""
        reader = BudgetReader(mock_hdf5_file)
        times, values = reader.get_values("Subregion 1")

        assert np.issubdtype(times.dtype, np.floating)
        assert np.issubdtype(values.dtype, np.floating)


# =============================================================================
# DataFrame Tests
# =============================================================================


class TestDataFrame:
    """Tests for DataFrame conversion."""

    def test_get_dataframe(self, mock_hdf5_file):
        """Test getting DataFrame."""
        pd = pytest.importorskip("pandas")
        reader = BudgetReader(mock_hdf5_file)
        df = reader.get_dataframe("Subregion 1")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 12
        assert len(df.columns) == 5

    def test_get_dataframe_with_columns(self, mock_hdf5_file):
        """Test getting DataFrame with specific columns."""
        pd = pytest.importorskip("pandas")
        reader = BudgetReader(mock_hdf5_file)

        # Get column names first
        headers = reader.get_column_headers(0)
        df = reader.get_dataframe("Subregion 1", columns=[0, 1])

        assert len(df.columns) == 2

    def test_get_all_dataframes(self, mock_hdf5_file):
        """Test getting DataFrames for all locations."""
        pd = pytest.importorskip("pandas")
        reader = BudgetReader(mock_hdf5_file)
        dfs = reader.get_all_dataframes()

        assert len(dfs) == 3
        assert "Subregion 1" in dfs
        assert "Subregion 2" in dfs
        assert "Subregion 3" in dfs


# =============================================================================
# Aggregation Tests
# =============================================================================


class TestAggregation:
    """Tests for data aggregation methods."""

    def test_get_cumulative(self, mock_hdf5_file):
        """Test cumulative calculation."""
        pd = pytest.importorskip("pandas")
        reader = BudgetReader(mock_hdf5_file)
        df = reader.get_cumulative("Subregion 1")

        # Cumulative values should increase (for positive values)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 12

    def test_get_monthly_averages(self, mock_hdf5_file):
        """Test monthly average calculation."""
        pd = pytest.importorskip("pandas")
        reader = BudgetReader(mock_hdf5_file)
        df = reader.get_monthly_averages("Subregion 1")

        assert isinstance(df, pd.DataFrame)

    def test_get_annual_totals(self, mock_hdf5_file):
        """Test annual total calculation."""
        pd = pytest.importorskip("pandas")
        reader = BudgetReader(mock_hdf5_file)
        df = reader.get_annual_totals("Subregion 1")

        assert isinstance(df, pd.DataFrame)


# =============================================================================
# Representation Tests
# =============================================================================


class TestRepresentation:
    """Tests for string representation."""

    def test_repr(self, mock_hdf5_file):
        """Test __repr__ output."""
        reader = BudgetReader(mock_hdf5_file)
        repr_str = repr(reader)

        assert "BudgetReader" in repr_str
        assert "GROUNDWATER BUDGET" in repr_str
        assert "n_locations=3" in repr_str
        assert "n_timesteps=12" in repr_str


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_budget_data_types(self):
        """Test budget data type constants."""
        assert BUDGET_DATA_TYPES[1] == "VR"
        assert BUDGET_DATA_TYPES[2] == "VLB"
        assert BUDGET_DATA_TYPES[3] == "VLE"

    def test_unit_markers(self):
        """Test unit marker constants."""
        assert "@UNITVL@" in UNIT_MARKERS
        assert "@UNITAR@" in UNIT_MARKERS
        assert "@UNITLT@" in UNIT_MARKERS

    def test_dss_data_types(self):
        """Test DSS data type constants."""
        from pyiwfm.io.budget import DSS_DATA_TYPES

        assert DSS_DATA_TYPES[1] == "PER-CUM"
        assert DSS_DATA_TYPES[2] == "PER-AVER"
        assert len(DSS_DATA_TYPES) == 2

    def test_budget_data_types_completeness(self):
        """Test that all expected budget data types are present."""
        assert BUDGET_DATA_TYPES[4] == "AR"
        assert BUDGET_DATA_TYPES[5] == "LT"
        assert BUDGET_DATA_TYPES[6] == "VR_PotCUAW"
        assert BUDGET_DATA_TYPES[7] == "VR_AgSupplyReq"
        assert BUDGET_DATA_TYPES[8] == "VR_AgShort"
        assert BUDGET_DATA_TYPES[9] == "VR_AgPump"
        assert BUDGET_DATA_TYPES[10] == "VR_AgDiv"
        assert BUDGET_DATA_TYPES[11] == "VR_AgOthIn"
        assert len(BUDGET_DATA_TYPES) == 11


# =============================================================================
# Additional Test Fixtures for New Tests
# =============================================================================


@pytest.fixture
def mock_hdf5_no_time(tmp_path):
    """Create an HDF5 budget file with no start_datetime (time tracking off)."""
    pytest.importorskip("h5py")
    import h5py

    filepath = tmp_path / "no_time_budget.hdf"

    with h5py.File(filepath, "w") as f:
        attrs = f.create_group("Attributes")
        attrs.create_dataset("Descriptor", data="STREAM BUDGET")
        attrs.create_dataset("NAreas", data=2)
        attrs.create_dataset("Areas", data=[500.0, 750.0])
        attrs.create_dataset("NTimeSteps", data=6)
        attrs.create_dataset("nLocations", data=2)
        attrs.create_dataset(
            "cLocationNames",
            data=[b"Reach 1", b"Reach 2"],
        )
        attrs.create_dataset("NLocationData", data=1)
        attrs.create_dataset("NDataColumns", data=3)
        attrs.create_dataset(
            "cFullColumnHeaders",
            data=[b"Inflow", b"Outflow", b"Storage"],
        )
        attrs.create_dataset("iDataColumnTypes", data=[1, 1, 3])
        attrs.create_dataset("iColWidth", data=[15, 15, 15])

        # Create location data
        np.random.seed(99)
        for name in ["Reach 1", "Reach 2"]:
            loc_group = f.create_group(name)
            data = np.random.rand(3, 6) * 500
            loc_group.create_dataset("data", data=data)

    return filepath


@pytest.fixture
def mock_hdf5_multi_locdata(tmp_path):
    """Create an HDF5 budget file with multiple location data structures."""
    pytest.importorskip("h5py")
    import h5py

    filepath = tmp_path / "multi_locdata_budget.hdf"

    with h5py.File(filepath, "w") as f:
        attrs = f.create_group("Attributes")
        attrs.create_dataset("Descriptor", data="ROOT ZONE BUDGET")
        attrs.create_dataset("NAreas", data=2)
        attrs.create_dataset("Areas", data=[1200.0, 1800.0])
        attrs.create_dataset("NTimeSteps", data=4)
        attrs.create_dataset("nLocations", data=2)
        attrs.create_dataset(
            "cLocationNames",
            data=[b"Zone A", b"Zone B"],
        )
        attrs.create_dataset("NLocationData", data=2)

        # Location data 1
        attrs.create_dataset("LocationData1%NDataColumns", data=3)
        attrs.create_dataset(
            "LocationData1%cFullColumnHeaders",
            data=[b"Precip (@UNITVL@)", b"ET (@UNITVL@)", b"Runoff (@UNITVL@)"],
        )
        attrs.create_dataset("LocationData1%iDataColumnTypes", data=[1, 1, 1])
        attrs.create_dataset("LocationData1%iColWidth", data=[12, 12, 12])

        # Location data 2
        attrs.create_dataset("LocationData2%NDataColumns", data=2)
        attrs.create_dataset(
            "LocationData2%cFullColumnHeaders",
            data=[b"Recharge (@UNITVL@)", b"Loss (@UNITVL@)"],
        )
        attrs.create_dataset("LocationData2%iDataColumnTypes", data=[1, 1])
        attrs.create_dataset("LocationData2%iColWidth", data=[12, 12])

        # Create location data
        np.random.seed(77)
        loc_group = f.create_group("Zone A")
        loc_group.create_dataset("data", data=np.random.rand(3, 4) * 100)

        loc_group = f.create_group("Zone B")
        loc_group.create_dataset("data", data=np.random.rand(2, 4) * 100)

    return filepath


@pytest.fixture
def mock_hdf5_alt_keys(tmp_path):
    """Create an HDF5 file using alternative attribute key names."""
    pytest.importorskip("h5py")
    import h5py

    filepath = tmp_path / "alt_keys_budget.hdf"

    with h5py.File(filepath, "w") as f:
        attrs = f.create_group("Attributes")
        attrs.create_dataset("Descriptor", data="LAKE BUDGET")
        attrs.create_dataset("NAreas", data=1)
        attrs.create_dataset("Areas", data=[2500.0])
        attrs.create_dataset("NTimeSteps", data=3)
        # Use "NLocations" instead of "nLocations"
        attrs.create_dataset("NLocations", data=1)
        attrs.create_dataset(
            "cLocationNames", data=[b"Lake 1"]
        )
        # Use ASCIIOutput% prefix keys
        attrs.create_dataset("ASCIIOutput%TitleLen", data=80)
        attrs.create_dataset("ASCIIOutput%NTitles", data=1)
        attrs.create_dataset("ASCIIOutput%cTitles", data=[b"Lake Budget Title"])
        attrs.create_dataset("NLocationData", data=1)
        attrs.create_dataset("NDataColumns", data=2)
        attrs.create_dataset(
            "cFullColumnHeaders",
            data=[b"Inflow (@UNITAR@)", b"Outflow (@UNITLT@)"],
        )
        attrs.create_dataset("iDataColumnTypes", data=[4, 5])
        attrs.create_dataset("iColWidth", data=[15, 15])

        loc_group = f.create_group("Lake 1")
        loc_group.create_dataset("data", data=np.array([[10.0, 20.0, 30.0], [5.0, 10.0, 15.0]]))

    return filepath


@pytest.fixture
def mock_hdf5_1d_data(tmp_path):
    """Create an HDF5 file with 1D data arrays."""
    pytest.importorskip("h5py")
    import h5py

    filepath = tmp_path / "1d_data_budget.hdf"

    with h5py.File(filepath, "w") as f:
        attrs = f.create_group("Attributes")
        attrs.create_dataset("Descriptor", data="SIMPLE BUDGET")
        attrs.create_dataset("NAreas", data=0)
        attrs.create_dataset("NTimeSteps", data=5)
        attrs.create_dataset("nLocations", data=1)
        attrs.create_dataset("cLocationNames", data=[b"Loc1"])
        attrs.create_dataset("NLocationData", data=1)
        attrs.create_dataset("NDataColumns", data=1)
        attrs.create_dataset("cFullColumnHeaders", data=[b"Value"])
        attrs.create_dataset("iDataColumnTypes", data=[1])
        attrs.create_dataset("iColWidth", data=[15])

        loc_group = f.create_group("Loc1")
        # 1D data: single column across timesteps
        loc_group.create_dataset("data", data=np.array([10.0, 20.0, 30.0, 40.0, 50.0]))

    return filepath


@pytest.fixture
def mock_hdf5_with_datetime(tmp_path):
    """Create an HDF5 budget file with datetime-based timestep info."""
    pytest.importorskip("h5py")
    import h5py

    filepath = tmp_path / "datetime_budget.hdf"

    with h5py.File(filepath, "w") as f:
        attrs = f.create_group("Attributes")
        attrs.create_dataset("Descriptor", data="GW BUDGET WITH TIME")
        attrs.create_dataset("NAreas", data=2)
        attrs.create_dataset("Areas", data=[100.0, 200.0])
        attrs.create_dataset("NTimeSteps", data=24)
        attrs.create_dataset("nLocations", data=2)
        attrs.create_dataset(
            "cLocationNames",
            data=[b"Sub 1", b"Sub 2"],
        )
        attrs.create_dataset("NLocationData", data=1)
        attrs.create_dataset("NDataColumns", data=3)
        attrs.create_dataset(
            "cFullColumnHeaders",
            data=[b"Deep Perc", b"Pumping", b"Storage"],
        )
        attrs.create_dataset("iDataColumnTypes", data=[1, 1, 3])
        attrs.create_dataset("iColWidth", data=[15, 15, 15])

        np.random.seed(123)
        for name in ["Sub 1", "Sub 2"]:
            loc_group = f.create_group(name)
            data = np.random.rand(3, 24) * 1000
            loc_group.create_dataset("data", data=data)

    return filepath


# =============================================================================
# TestBudgetReaderEdgeCases
# =============================================================================


class TestBudgetReaderEdgeCases:
    """Test edge cases in BudgetReader."""

    def test_invalid_location_negative_index(self, mock_hdf5_file):
        """Test that a negative integer index raises IndexError."""
        reader = BudgetReader(mock_hdf5_file)
        with pytest.raises(IndexError, match="out of range"):
            reader.get_location_index(-1)

    def test_invalid_location_too_large_index(self, mock_hdf5_file):
        """Test that an index beyond range raises IndexError."""
        reader = BudgetReader(mock_hdf5_file)
        with pytest.raises(IndexError, match="out of range"):
            reader.get_location_index(100)

    def test_invalid_location_name(self, mock_hdf5_file):
        """Test that an unknown location name raises KeyError."""
        reader = BudgetReader(mock_hdf5_file)
        with pytest.raises(KeyError, match="not found"):
            reader.get_location_index("Nonexistent Region")

    def test_get_values_invalid_location(self, mock_hdf5_file):
        """Test get_values with invalid location propagates error."""
        reader = BudgetReader(mock_hdf5_file)
        with pytest.raises(KeyError):
            reader.get_values("No Such Place")

    def test_get_column_headers_invalid_location(self, mock_hdf5_file):
        """Test get_column_headers with invalid location propagates error."""
        reader = BudgetReader(mock_hdf5_file)
        with pytest.raises(KeyError):
            reader.get_column_headers("Fake Zone")

    def test_get_dataframe_with_string_column_names(self, mock_hdf5_file):
        """Test get_dataframe with column names as strings."""
        pd = pytest.importorskip("pandas")
        reader = BudgetReader(mock_hdf5_file)
        headers = reader.get_column_headers(0)
        # Select the first column by name
        df = reader.get_dataframe("Subregion 1", columns=[headers[0]])
        assert len(df.columns) == 1
        assert df.columns[0] == headers[0]

    def test_get_dataframe_with_case_insensitive_column_names(self, mock_hdf5_file):
        """Test get_dataframe with case-insensitive string column matching."""
        pd = pytest.importorskip("pandas")
        reader = BudgetReader(mock_hdf5_file)
        headers = reader.get_column_headers(0)
        # Use all-uppercase version of the column name
        upper_name = headers[0].upper()
        df = reader.get_dataframe("Subregion 1", columns=[upper_name])
        assert len(df.columns) == 1

    def test_get_dataframe_invalid_column_name(self, mock_hdf5_file):
        """Test get_dataframe with invalid column name raises KeyError."""
        pd = pytest.importorskip("pandas")
        reader = BudgetReader(mock_hdf5_file)
        with pytest.raises(KeyError, match="not found"):
            reader.get_dataframe("Subregion 1", columns=["Nonexistent Column"])

    def test_get_values_no_time_tracking(self, mock_hdf5_no_time):
        """Test get_values when time tracking is off (no start_datetime)."""
        reader = BudgetReader(mock_hdf5_no_time)
        times, values = reader.get_values("Reach 1")
        # Times should be simple sequential values
        assert len(times) == 6
        assert values.shape[0] == 6
        # Without start_datetime, times start from start_time (default 0.0)
        assert times[0] == pytest.approx(0.0)

    def test_format_detection_bin_extension(self, tmp_path):
        """Test format detection for .bin files."""
        import struct

        bin_file = tmp_path / "test.bin"
        # Write minimal data so _detect_format works (it checks extension first)
        bin_file.write_bytes(b"\x00" * 100)

        reader_obj = object.__new__(BudgetReader)
        reader_obj.filepath = bin_file
        fmt = reader_obj._detect_format()
        assert fmt == "binary"

    def test_format_detection_out_extension(self, tmp_path):
        """Test format detection for .out files."""
        out_file = tmp_path / "test.out"
        out_file.write_bytes(b"\x00" * 100)

        reader_obj = object.__new__(BudgetReader)
        reader_obj.filepath = out_file
        fmt = reader_obj._detect_format()
        assert fmt == "binary"

    def test_format_detection_unknown_extension(self, tmp_path):
        """Test format detection for unknown extension falls back to binary."""
        unk_file = tmp_path / "test.xyz"
        unk_file.write_bytes(b"\x00" * 100)

        reader_obj = object.__new__(BudgetReader)
        reader_obj.filepath = unk_file
        fmt = reader_obj._detect_format()
        assert fmt == "binary"

    def test_format_detection_h5_extension(self, tmp_path):
        """Test format detection for .h5 extension."""
        h5py = pytest.importorskip("h5py")
        h5_file = tmp_path / "test.h5"
        with h5py.File(h5_file, "w") as f:
            f.create_dataset("dummy", data=[1, 2, 3])

        reader_obj = object.__new__(BudgetReader)
        reader_obj.filepath = h5_file
        fmt = reader_obj._detect_format()
        assert fmt == "hdf5"

    def test_format_detection_hdf5_extension(self, tmp_path):
        """Test format detection for .hdf5 extension."""
        h5py = pytest.importorskip("h5py")
        hdf5_file = tmp_path / "test.hdf5"
        with h5py.File(hdf5_file, "w") as f:
            f.create_dataset("dummy", data=[1, 2, 3])

        reader_obj = object.__new__(BudgetReader)
        reader_obj.filepath = hdf5_file
        fmt = reader_obj._detect_format()
        assert fmt == "hdf5"

    def test_parse_iwfm_datetime_iso_format(self):
        """Test parsing ISO format datetime."""
        dt = parse_iwfm_datetime("2020-06-15 10:30:00")
        assert dt == datetime(2020, 6, 15, 10, 30, 0)

    def test_parse_iwfm_datetime_iso_date_only(self):
        """Test parsing ISO format date only."""
        dt = parse_iwfm_datetime("2020-06-15")
        assert dt == datetime(2020, 6, 15)

    def test_parse_iwfm_datetime_with_hm(self):
        """Test parsing datetime with hours and minutes only."""
        dt = parse_iwfm_datetime("03/25/2020 14:30")
        assert dt == datetime(2020, 3, 25, 14, 30)

    def test_get_values_all_columns_default(self, mock_hdf5_file):
        """Test that columns=None returns all columns."""
        reader = BudgetReader(mock_hdf5_file)
        times, values = reader.get_values("Subregion 1", columns=None)
        assert values.shape[1] == 5

    def test_get_values_single_column(self, mock_hdf5_file):
        """Test reading a single column."""
        reader = BudgetReader(mock_hdf5_file)
        times, values = reader.get_values("Subregion 1", columns=[2])
        assert values.shape == (12, 1)


# =============================================================================
# TestBudgetReaderMultiLocation
# =============================================================================


class TestBudgetReaderMultiLocation:
    """Test multi-location operations."""

    def test_read_all_locations(self, mock_hdf5_file):
        """Test reading values from all locations sequentially."""
        reader = BudgetReader(mock_hdf5_file)
        all_data = {}
        for loc in reader.locations:
            times, values = reader.get_values(loc)
            all_data[loc] = (times, values)

        assert len(all_data) == 3
        for loc, (t, v) in all_data.items():
            assert len(t) == 12
            assert v.shape == (12, 5)

    def test_compare_values_across_locations(self, mock_hdf5_file):
        """Test that different locations return different data."""
        reader = BudgetReader(mock_hdf5_file)
        _, v1 = reader.get_values("Subregion 1")
        _, v2 = reader.get_values("Subregion 2")

        # The mock data uses np.random with seed 42,
        # each location gets its own random data so they should differ
        assert not np.array_equal(v1, v2)

    def test_sum_across_locations(self, mock_hdf5_file):
        """Test summing values across all locations."""
        reader = BudgetReader(mock_hdf5_file)
        total = None
        for loc in reader.locations:
            _, values = reader.get_values(loc)
            if total is None:
                total = values.copy()
            else:
                total += values

        assert total is not None
        assert total.shape == (12, 5)
        # Total should be greater than any individual location's values
        _, v1 = reader.get_values("Subregion 1")
        assert np.all(total >= v1)

    def test_all_locations_same_time_axis(self, mock_hdf5_file):
        """Test that all locations share the same time axis."""
        reader = BudgetReader(mock_hdf5_file)
        all_times = []
        for loc in reader.locations:
            times, _ = reader.get_values(loc)
            all_times.append(times)

        for i in range(1, len(all_times)):
            np.testing.assert_array_equal(all_times[0], all_times[i])

    def test_get_all_dataframes_structure(self, mock_hdf5_file):
        """Test that get_all_dataframes returns correct structure."""
        pd = pytest.importorskip("pandas")
        reader = BudgetReader(mock_hdf5_file)
        dfs = reader.get_all_dataframes()

        assert isinstance(dfs, dict)
        for loc_name, df in dfs.items():
            assert isinstance(loc_name, str)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 12
            assert len(df.columns) == 5

    def test_location_index_matches_name(self, mock_hdf5_file):
        """Test that index-based and name-based access return same data."""
        reader = BudgetReader(mock_hdf5_file)
        _, v_by_name = reader.get_values("Subregion 2")
        _, v_by_idx = reader.get_values(1)
        np.testing.assert_array_equal(v_by_name, v_by_idx)


# =============================================================================
# TestBudgetReaderTimeSeries
# =============================================================================


class TestBudgetReaderTimeSeries:
    """Test time series related operations."""

    def test_time_array_length_matches_data(self, mock_hdf5_file):
        """Test that time and data arrays have consistent length."""
        reader = BudgetReader(mock_hdf5_file)
        times, values = reader.get_values("Subregion 1")
        assert len(times) == values.shape[0]

    def test_time_array_monotonically_increasing(self, mock_hdf5_file):
        """Test that time values are strictly increasing."""
        reader = BudgetReader(mock_hdf5_file)
        times, _ = reader.get_values("Subregion 1")
        diffs = np.diff(times)
        assert np.all(diffs > 0)

    def test_cumulative_values_nondecreasing(self, mock_hdf5_file):
        """Test that cumulative values are non-decreasing for positive data."""
        pd = pytest.importorskip("pandas")
        reader = BudgetReader(mock_hdf5_file)
        df = reader.get_cumulative("Subregion 1")
        # For all-positive data (random), cumulative should be non-decreasing
        for col in df.columns:
            diffs = df[col].diff().dropna()
            assert (diffs >= 0).all()

    def test_monthly_averages_result_shape(self, mock_hdf5_file):
        """Test that monthly averages have correct shape."""
        pd = pytest.importorskip("pandas")
        reader = BudgetReader(mock_hdf5_file)
        df = reader.get_monthly_averages("Subregion 1")
        assert isinstance(df, pd.DataFrame)
        assert len(df.columns) == 5

    def test_annual_totals_result_shape(self, mock_hdf5_file):
        """Test that annual totals have correct shape."""
        pd = pytest.importorskip("pandas")
        reader = BudgetReader(mock_hdf5_file)
        df = reader.get_annual_totals("Subregion 1")
        assert isinstance(df, pd.DataFrame)
        assert len(df.columns) == 5

    def test_no_start_datetime_time_axis(self, mock_hdf5_no_time):
        """Test time axis generation without start_datetime."""
        reader = BudgetReader(mock_hdf5_no_time)
        times, _ = reader.get_values("Reach 1")
        # Without start_datetime, times should be dt-spaced from start_time
        ts = reader.header.timestep
        expected = np.arange(6) * ts.delta_t + ts.start_time
        np.testing.assert_array_almost_equal(times, expected)

    def test_monthly_averages_no_datetime_index(self, mock_hdf5_no_time):
        """Test monthly averages when index is not DatetimeIndex."""
        pd = pytest.importorskip("pandas")
        reader = BudgetReader(mock_hdf5_no_time)
        df = reader.get_monthly_averages("Reach 1")
        # When index is not DatetimeIndex, should return DataFrame as-is
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 6  # no resampling, returned unchanged

    def test_annual_totals_no_datetime_fewer_than_12(self, mock_hdf5_no_time):
        """Test annual totals with non-datetime index and fewer than 12 rows."""
        pd = pytest.importorskip("pandas")
        reader = BudgetReader(mock_hdf5_no_time)
        df = reader.get_annual_totals("Reach 1")
        # With 6 rows (< 12), n_years == 0, so should return sum as single row
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_dataframe_non_datetime_index(self, mock_hdf5_no_time):
        """Test get_dataframe with non-datetime time index."""
        pd = pytest.importorskip("pandas")
        reader = BudgetReader(mock_hdf5_no_time)
        df = reader.get_dataframe("Reach 1")
        assert isinstance(df, pd.DataFrame)
        # Index should not be DatetimeIndex
        assert not isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == "Time"


# =============================================================================
# TestBudgetReaderBinary
# =============================================================================


class TestBudgetReaderBinary:
    """Test binary format reading using struct.pack to create minimal files."""

    @staticmethod
    def _write_fortran_string(f, s: str, length: int):
        """Write a Fortran unformatted string record."""
        encoded = s.encode("ascii")
        padded = encoded.ljust(length, b" ")
        rec_len = len(padded)
        f.write(struct.pack("i", rec_len))
        f.write(padded)
        f.write(struct.pack("i", rec_len))

    @staticmethod
    def _write_fortran_int(f, val: int):
        """Write a Fortran unformatted integer record."""
        f.write(struct.pack("i", 4))
        f.write(struct.pack("i", val))
        f.write(struct.pack("i", 4))

    @staticmethod
    def _write_fortran_real8(f, val: float):
        """Write a Fortran unformatted REAL(8) record."""
        f.write(struct.pack("i", 8))
        f.write(struct.pack("d", val))
        f.write(struct.pack("i", 8))

    @staticmethod
    def _write_fortran_logical(f, val: bool):
        """Write a Fortran unformatted logical record."""
        f.write(struct.pack("i", 4))
        f.write(struct.pack("i", 1 if val else 0))
        f.write(struct.pack("i", 4))

    @staticmethod
    def _write_fortran_int_array(f, arr):
        """Write a Fortran unformatted integer array record."""
        data = struct.pack(f"{len(arr)}i", *arr)
        f.write(struct.pack("i", len(data)))
        f.write(data)
        f.write(struct.pack("i", len(data)))

    @staticmethod
    def _write_fortran_real8_array(f, arr):
        """Write a Fortran unformatted REAL(8) array record."""
        data = struct.pack(f"{len(arr)}d", *arr)
        f.write(struct.pack("i", len(data)))
        f.write(data)
        f.write(struct.pack("i", len(data)))

    def _create_minimal_binary_budget(self, filepath, n_timesteps=4,
                                       track_time=True, n_areas=2,
                                       n_locations=2, n_columns=3):
        """Create a minimal binary budget file for testing."""
        import struct

        with open(filepath, "wb") as f:
            # Descriptor (100 chars)
            self._write_fortran_string(f, "TEST BINARY BUDGET", 100)

            # Timestep info
            self._write_fortran_int(f, n_timesteps)          # n_timesteps
            self._write_fortran_logical(f, track_time)        # track_time
            self._write_fortran_real8(f, 1.0)                 # delta_t
            self._write_fortran_int(f, 43200)                 # delta_t_minutes (30 days)
            self._write_fortran_string(f, "1MON", 10)         # unit

            if track_time:
                self._write_fortran_string(f, "10/01/2020_00:00", 21)  # start date
            self._write_fortran_real8(f, 0.0)                 # start_time

            # Areas
            self._write_fortran_int(f, n_areas)
            if n_areas > 0:
                self._write_fortran_real8_array(f, [1000.0 * (i + 1) for i in range(n_areas)])

            # ASCII output info
            self._write_fortran_int(f, 160)                   # title_len
            n_titles = 1
            self._write_fortran_int(f, n_titles)              # n_titles
            for _ in range(n_titles):
                self._write_fortran_string(f, "Test Budget Title", 1000)
            for _ in range(n_titles):
                self._write_fortran_logical(f, True)          # title_persist
            self._write_fortran_string(f, "(A10,3F15.2)", 500)  # format_spec
            n_col_header_lines = 3
            self._write_fortran_int(f, n_col_header_lines)    # n_column_header_lines

            # Locations
            self._write_fortran_int(f, n_locations)
            for i in range(n_locations):
                self._write_fortran_string(f, f"Location {i+1}", 100)

            # Location data (1 shared structure)
            n_loc_data = 1
            self._write_fortran_int(f, n_loc_data)

            for _ in range(n_loc_data):
                self._write_fortran_int(f, n_columns)             # n_columns
                self._write_fortran_int(f, n_columns)             # storage_units
                for c in range(n_columns):
                    self._write_fortran_string(f, f"Column {c+1}", 100)
                self._write_fortran_int_array(f, [1] * n_columns)  # column_types
                self._write_fortran_int_array(f, [15] * n_columns) # column_widths

                # Multi-line column headers
                for _ in range(n_col_header_lines):
                    for _ in range(n_columns):
                        self._write_fortran_string(f, "", 100)
                # Column header format specs
                for _ in range(n_col_header_lines):
                    self._write_fortran_string(f, "", 500)

            # DSS output info
            self._write_fortran_int(f, 0)                     # n_dss_pathnames
            self._write_fortran_int(f, 0)                     # n_dss_types
            self._write_fortran_int_array(f, [])               # empty array

            # Record file_position after header
            header_end = f.tell()

            # Write data: for each timestep, write all locations' data
            for t in range(n_timesteps):
                for loc in range(n_locations):
                    data = [(t + 1) * 100.0 + (loc + 1) * 10.0 + c
                            for c in range(n_columns)]
                    # Write as raw REAL(8) (no Fortran record wrappers for data)
                    for val in data:
                        f.write(struct.pack("d", val))

        return filepath

    def test_binary_header_parsing(self, tmp_path):
        """Test that binary header is parsed correctly."""
        import struct

        filepath = tmp_path / "test_budget.bin"
        self._create_minimal_binary_budget(filepath)

        reader = BudgetReader(filepath)
        assert reader.format == "binary"
        assert reader.descriptor == "TEST BINARY BUDGET"
        assert reader.n_locations == 2
        assert reader.n_timesteps == 4
        assert reader.locations == ["Location 1", "Location 2"]

    def test_binary_header_timestep_info(self, tmp_path):
        """Test that binary timestep info is parsed correctly."""
        filepath = tmp_path / "test_budget.bin"
        self._create_minimal_binary_budget(filepath)

        reader = BudgetReader(filepath)
        ts = reader.header.timestep
        assert ts.track_time is True
        assert ts.delta_t == pytest.approx(1.0)
        assert ts.delta_t_minutes == 43200
        assert ts.unit == "1MON"
        assert ts.start_datetime is not None
        assert ts.start_datetime.year == 2020
        assert ts.start_datetime.month == 10

    def test_binary_header_areas(self, tmp_path):
        """Test that binary areas are parsed correctly."""
        filepath = tmp_path / "test_budget.bin"
        self._create_minimal_binary_budget(filepath, n_areas=3)

        reader = BudgetReader(filepath)
        assert reader.header.n_areas == 3
        np.testing.assert_array_almost_equal(
            reader.header.areas, [1000.0, 2000.0, 3000.0]
        )

    def test_binary_header_ascii_output(self, tmp_path):
        """Test that binary ASCII output info is parsed correctly."""
        filepath = tmp_path / "test_budget.bin"
        self._create_minimal_binary_budget(filepath)

        reader = BudgetReader(filepath)
        ascii_out = reader.header.ascii_output
        assert ascii_out.title_len == 160
        assert ascii_out.n_titles == 1
        assert len(ascii_out.titles) == 1
        assert "Test Budget Title" in ascii_out.titles[0]
        assert ascii_out.title_persist == [True]
        assert "(A10,3F15.2)" in ascii_out.format_spec
        assert ascii_out.n_column_header_lines == 3

    def test_binary_header_location_data(self, tmp_path):
        """Test that binary location data structures are parsed correctly."""
        filepath = tmp_path / "test_budget.bin"
        self._create_minimal_binary_budget(filepath, n_columns=4)

        reader = BudgetReader(filepath)
        assert len(reader.header.location_data) == 1
        loc_data = reader.header.location_data[0]
        assert loc_data.n_columns == 4
        assert loc_data.storage_units == 4
        assert len(loc_data.column_headers) == 4
        assert loc_data.column_headers[0] == "Column 1"
        assert loc_data.column_types == [1, 1, 1, 1]
        assert loc_data.column_widths == [15, 15, 15, 15]

    def test_binary_column_headers(self, tmp_path):
        """Test get_column_headers for binary file."""
        filepath = tmp_path / "test_budget.bin"
        self._create_minimal_binary_budget(filepath)

        reader = BudgetReader(filepath)
        headers = reader.get_column_headers(0)
        assert len(headers) == 3
        assert headers[0] == "Column 1"
        assert headers[1] == "Column 2"
        assert headers[2] == "Column 3"

    def test_binary_read_values(self, tmp_path):
        """Test reading values from binary file."""
        filepath = tmp_path / "test_budget.bin"
        self._create_minimal_binary_budget(
            filepath, n_timesteps=4, n_locations=2, n_columns=3
        )

        reader = BudgetReader(filepath)
        times, values = reader.get_values(0)
        assert len(times) == 4
        assert values.shape == (4, 3)

    def test_binary_no_time_tracking(self, tmp_path):
        """Test binary file without time tracking."""
        filepath = tmp_path / "test_no_time.bin"
        self._create_minimal_binary_budget(filepath, track_time=False)

        reader = BudgetReader(filepath)
        ts = reader.header.timestep
        assert ts.track_time is False
        assert ts.start_datetime is None

    def test_binary_no_areas(self, tmp_path):
        """Test binary file with zero areas."""
        filepath = tmp_path / "test_no_areas.bin"
        self._create_minimal_binary_budget(filepath, n_areas=0)

        reader = BudgetReader(filepath)
        assert reader.header.n_areas == 0

    def test_binary_repr(self, tmp_path):
        """Test __repr__ for binary reader."""
        filepath = tmp_path / "test_budget.bin"
        self._create_minimal_binary_budget(filepath)

        reader = BudgetReader(filepath)
        repr_str = repr(reader)
        assert "BudgetReader" in repr_str
        assert "binary" in repr_str
        assert "TEST BINARY BUDGET" in repr_str


# =============================================================================
# TestBudgetReaderAreaWeights
# =============================================================================


class TestBudgetReaderAreaWeights:
    """Test area-related properties and methods."""

    def test_areas_property_from_header(self, mock_hdf5_file):
        """Test that areas are correctly stored in the header."""
        reader = BudgetReader(mock_hdf5_file)
        areas = reader.header.areas
        np.testing.assert_array_almost_equal(areas, [1000.0, 2000.0, 3000.0])

    def test_n_areas_property(self, mock_hdf5_file):
        """Test that n_areas is correct."""
        reader = BudgetReader(mock_hdf5_file)
        assert reader.header.n_areas == 3

    def test_per_area_calculation(self, mock_hdf5_file):
        """Test computing per-area values manually using header areas."""
        reader = BudgetReader(mock_hdf5_file)
        _, values = reader.get_values("Subregion 1")
        area = reader.header.areas[0]
        per_area = values / area
        assert per_area.shape == values.shape
        # Per-area values should be smaller (area > 1)
        assert np.all(np.abs(per_area) <= np.abs(values) + 1e-10)

    def test_area_weighted_average(self, mock_hdf5_file):
        """Test computing area-weighted average across locations."""
        reader = BudgetReader(mock_hdf5_file)
        areas = reader.header.areas
        total_area = np.sum(areas)

        weighted_sum = None
        for i, loc in enumerate(reader.locations):
            _, values = reader.get_values(loc)
            contribution = values * areas[i]
            if weighted_sum is None:
                weighted_sum = contribution
            else:
                weighted_sum += contribution

        weighted_avg = weighted_sum / total_area
        assert weighted_avg.shape == (12, 5)
        # Weighted average should be between min and max of individual values
        assert np.all(np.isfinite(weighted_avg))

    def test_areas_empty_when_zero(self, mock_hdf5_1d_data):
        """Test areas when n_areas is 0."""
        reader = BudgetReader(mock_hdf5_1d_data)
        assert reader.header.n_areas == 0

    def test_areas_array_length_matches_n_areas(self, mock_hdf5_file):
        """Test that areas array length matches n_areas."""
        reader = BudgetReader(mock_hdf5_file)
        assert len(reader.header.areas) == reader.header.n_areas


# =============================================================================
# TestASCIIOutputInfo
# =============================================================================


class TestASCIIOutputInfo:
    """Test the ASCIIOutputInfo dataclass."""

    def test_default_values(self):
        """Test ASCIIOutputInfo default values."""
        info = ASCIIOutputInfo()
        assert info.title_len == 160
        assert info.n_titles == 0
        assert info.titles == []
        assert info.title_persist == []
        assert info.format_spec == ""
        assert info.n_column_header_lines == 3

    def test_custom_values(self):
        """Test ASCIIOutputInfo with custom values."""
        info = ASCIIOutputInfo(
            title_len=80,
            n_titles=2,
            titles=["Title 1", "Title 2"],
            title_persist=[True, False],
            format_spec="(A10,2F15.2)",
            n_column_header_lines=5,
        )
        assert info.title_len == 80
        assert info.n_titles == 2
        assert len(info.titles) == 2
        assert info.titles[0] == "Title 1"
        assert info.titles[1] == "Title 2"
        assert info.title_persist == [True, False]
        assert info.format_spec == "(A10,2F15.2)"
        assert info.n_column_header_lines == 5

    def test_empty_titles_list_is_independent(self):
        """Test that default list fields are independent across instances."""
        info1 = ASCIIOutputInfo()
        info2 = ASCIIOutputInfo()
        info1.titles.append("Added Title")
        assert len(info2.titles) == 0

    def test_title_persist_default_is_independent(self):
        """Test that default title_persist lists are independent."""
        info1 = ASCIIOutputInfo()
        info2 = ASCIIOutputInfo()
        info1.title_persist.append(True)
        assert len(info2.title_persist) == 0

    def test_ascii_output_from_hdf5_alt_keys(self, mock_hdf5_alt_keys):
        """Test ASCIIOutputInfo populated from HDF5 with alternative keys."""
        reader = BudgetReader(mock_hdf5_alt_keys)
        ascii_out = reader.header.ascii_output
        assert ascii_out.title_len == 80
        assert ascii_out.n_titles == 1
        assert len(ascii_out.titles) == 1
        assert "Lake Budget Title" in ascii_out.titles[0]


# =============================================================================
# TestHDF5EdgeCases
# =============================================================================


class TestHDF5EdgeCases:
    """Test HDF5 parsing edge cases."""

    def test_alternative_nlocation_key(self, mock_hdf5_alt_keys):
        """Test parsing HDF5 with NLocations instead of nLocations."""
        reader = BudgetReader(mock_hdf5_alt_keys)
        assert reader.n_locations == 1
        assert reader.locations == ["Lake 1"]

    def test_multi_location_data_structures(self, mock_hdf5_multi_locdata):
        """Test parsing HDF5 with multiple location data structures."""
        reader = BudgetReader(mock_hdf5_multi_locdata)
        assert reader.n_locations == 2
        assert len(reader.header.location_data) == 2
        assert reader.header.location_data[0].n_columns == 3
        assert reader.header.location_data[1].n_columns == 2

    def test_multi_locdata_column_headers(self, mock_hdf5_multi_locdata):
        """Test column headers differ per location data structure."""
        reader = BudgetReader(mock_hdf5_multi_locdata)
        h0 = reader.get_column_headers(0)
        h1 = reader.get_column_headers(1)
        assert len(h0) == 3
        assert len(h1) == 2
        assert "(volume)" in h0[0]  # Precip with @UNITVL@ replaced
        assert "(volume)" in h1[0]  # Recharge with @UNITVL@ replaced

    def test_1d_data_reshaping(self, mock_hdf5_1d_data):
        """Test that 1D data is properly reshaped."""
        reader = BudgetReader(mock_hdf5_1d_data)
        times, values = reader.get_values("Loc1")
        assert len(times) == 5
        assert values.shape == (5, 1)

    def test_area_and_length_unit_markers(self, mock_hdf5_alt_keys):
        """Test that area and length unit markers are replaced."""
        reader = BudgetReader(mock_hdf5_alt_keys)
        headers = reader.get_column_headers("Lake 1")
        assert "(area)" in headers[0]
        assert "(length)" in headers[1]

    def test_descriptor_from_bytes(self, mock_hdf5_file):
        """Test that byte descriptor is properly decoded."""
        reader = BudgetReader(mock_hdf5_file)
        assert isinstance(reader.descriptor, str)
        assert reader.descriptor == "GROUNDWATER BUDGET"

    def test_location_names_stripped(self, mock_hdf5_file):
        """Test that location names are properly stripped of whitespace."""
        reader = BudgetReader(mock_hdf5_file)
        for name in reader.locations:
            assert name == name.strip()

    def test_hdf5_location_not_found_in_file(self, tmp_path):
        """Test reading values when location group is missing from HDF5."""
        h5py = pytest.importorskip("h5py")
        filepath = tmp_path / "missing_loc.hdf"

        with h5py.File(filepath, "w") as f:
            attrs = f.create_group("Attributes")
            attrs.create_dataset("Descriptor", data="MISSING LOC BUDGET")
            attrs.create_dataset("NTimeSteps", data=2)
            attrs.create_dataset("nLocations", data=1)
            attrs.create_dataset("cLocationNames", data=[b"Ghost Location"])
            attrs.create_dataset("NLocationData", data=1)
            attrs.create_dataset("NDataColumns", data=1)
            attrs.create_dataset("cFullColumnHeaders", data=[b"Value"])
            attrs.create_dataset("iDataColumnTypes", data=[1])
            attrs.create_dataset("iColWidth", data=[15])
            # Intentionally do NOT create the location group

        reader = BudgetReader(filepath)
        with pytest.raises(KeyError, match="not found in HDF5"):
            reader.get_values("Ghost Location")

    def test_hdf5_location_no_data(self, tmp_path):
        """Test reading from location group with no datasets."""
        h5py = pytest.importorskip("h5py")
        filepath = tmp_path / "no_data_loc.hdf"

        with h5py.File(filepath, "w") as f:
            attrs = f.create_group("Attributes")
            attrs.create_dataset("Descriptor", data="EMPTY LOC BUDGET")
            attrs.create_dataset("NTimeSteps", data=2)
            attrs.create_dataset("nLocations", data=1)
            attrs.create_dataset("cLocationNames", data=[b"Empty Loc"])
            attrs.create_dataset("NLocationData", data=1)
            attrs.create_dataset("NDataColumns", data=1)
            attrs.create_dataset("cFullColumnHeaders", data=[b"Value"])
            attrs.create_dataset("iDataColumnTypes", data=[1])
            attrs.create_dataset("iColWidth", data=[15])
            # Create location group but no datasets
            f.create_group("Empty Loc")

        reader = BudgetReader(filepath)
        with pytest.raises(ValueError, match="No data found"):
            reader.get_values("Empty Loc")

    def test_hdf5_root_attrs_fallback(self, tmp_path):
        """Test reading HDF5 without Attributes group (attrs on root)."""
        h5py = pytest.importorskip("h5py")
        filepath = tmp_path / "root_attrs.hdf"

        with h5py.File(filepath, "w") as f:
            # Put everything on root, not in "Attributes" group
            f.create_dataset("Descriptor", data="ROOT ATTRS BUDGET")
            f.create_dataset("NTimeSteps", data=2)
            f.create_dataset("nLocations", data=1)
            f.create_dataset("cLocationNames", data=[b"Root Loc"])
            f.create_dataset("NLocationData", data=1)
            f.create_dataset("NDataColumns", data=1)
            f.create_dataset("cFullColumnHeaders", data=[b"Flow"])
            f.create_dataset("iDataColumnTypes", data=[1])
            f.create_dataset("iColWidth", data=[15])

            loc_group = f.create_group("Root Loc")
            loc_group.create_dataset("data", data=np.array([[1.0, 2.0]]))

        reader = BudgetReader(filepath)
        assert reader.descriptor == "ROOT ATTRS BUDGET"
        assert reader.n_locations == 1

    def test_hdf5_timestep_from_data_shape(self, tmp_path):
        """Test that n_timesteps is inferred from data shape when NTimeSteps is 0."""
        h5py = pytest.importorskip("h5py")
        filepath = tmp_path / "infer_ts.hdf"

        with h5py.File(filepath, "w") as f:
            attrs = f.create_group("Attributes")
            attrs.create_dataset("Descriptor", data="INFER TS BUDGET")
            attrs.create_dataset("NTimeSteps", data=0)
            attrs.create_dataset("nLocations", data=1)
            attrs.create_dataset("cLocationNames", data=[b"Infer Loc"])
            attrs.create_dataset("NLocationData", data=1)
            attrs.create_dataset("NDataColumns", data=2)
            attrs.create_dataset("cFullColumnHeaders", data=[b"A", b"B"])
            attrs.create_dataset("iDataColumnTypes", data=[1, 1])
            attrs.create_dataset("iColWidth", data=[15, 15])

            loc_group = f.create_group("Infer Loc")
            loc_group.create_dataset("data", data=np.random.rand(2, 7))

        reader = BudgetReader(filepath)
        # n_timesteps should be inferred from the data shape
        assert reader.header.timestep.n_timesteps == 7

    def test_single_locdata_replicated(self, mock_hdf5_file):
        """Test that single location data is replicated for all locations."""
        reader = BudgetReader(mock_hdf5_file)
        # File has 3 locations but 1 location data, should be replicated
        assert len(reader.header.location_data) == 3
        # All should reference the same structure
        for ld in reader.header.location_data:
            assert ld.n_columns == 5
