"""
Comprehensive tests for pyiwfm.io.zbudget module.

Tests cover:
- ZBudgetReader initialization
- Header parsing
- Zone information retrieval
- Data reading and extraction
- DataFrame conversion
- Aggregation and export methods
"""

from datetime import datetime

import numpy as np
import pytest

from pyiwfm.io.zbudget import (
    ZBUDGET_DATA_TYPES,
    ZBudgetHeader,
    ZBudgetReader,
    ZoneInfo,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_zbudget_file(tmp_path):
    """Create a mock ZBudget HDF5 file (old-style data_name/Layer_N layout)."""
    pytest.importorskip("h5py")
    import h5py

    filepath = tmp_path / "test_zbudget.hdf"
    n_elements = 30
    n_timesteps = 12

    with h5py.File(filepath, "w") as f:
        # Create Attributes group
        attrs = f.create_group("Attributes")
        attrs.create_dataset("Software_Version", data="IWFM 2025.0")
        attrs.create_dataset("Descriptor", data="GROUNDWATER ZONE BUDGET")
        attrs.create_dataset("lVertFlows_DefinedAtNode", data=False)
        attrs.create_dataset("lFaceFlows_Defined", data=False)
        attrs.create_dataset("lStorages_Defined", data=True)
        attrs.create_dataset("lComputeError", data=False)
        attrs.create_dataset("NData", data=4)
        attrs.create_dataset("DataTypes", data=[1, 4, 4, 1])
        attrs.create_dataset(
            "FullDataNames",
            data=[
                b"Deep Percolation",
                b"Pumping",
                b"Subsurface Inflow",
                b"Storage Change",
            ],
        )
        attrs.create_dataset(
            "DataHDFPaths",
            data=[
                b"/DeepPercolation",
                b"/Pumping",
                b"/SubsurfaceInflow",
                b"/StorageChange",
            ],
        )
        attrs.create_dataset("NTimeSteps", data=n_timesteps)

        # Create ZoneList group
        zone_list = f.create_group("ZoneList")
        zone_list.attrs["NZones"] = 3
        zone_list.create_dataset("ZoneNames", data=[b"Zone 1", b"Zone 2", b"Zone 3"])

        # Zone details
        for i in range(3):
            zone_group = zone_list.create_group(f"Zone_{i + 1}")
            zone_group.create_dataset("Elements", data=list(range(i * 10 + 1, (i + 1) * 10 + 1)))
            zone_group.create_dataset("Area", data=1000.0 * (i + 1))
            zone_group.create_dataset("AdjacentZones", data=[j + 1 for j in range(3) if j != i])

        # Create data groups — old-style: data_name/Layer_N
        # Shape: (n_timesteps, n_elements) to match real IWFM convention
        rng = np.random.default_rng(42)
        for path in ["/DeepPercolation", "/Pumping", "/SubsurfaceInflow", "/StorageChange"]:
            data_group = f.create_group(path.strip("/"))
            data = rng.random((n_timesteps, n_elements)) * 100
            data_group.create_dataset("Layer_1", data=data)
            data_group.create_dataset("Layer_2", data=data * 0.5)

    return filepath


@pytest.fixture
def sample_zone_info():
    """Create sample ZoneInfo."""
    return ZoneInfo(
        id=1,
        name="Test Zone",
        n_elements=10,
        element_ids=list(range(1, 11)),
        area=5000.0,
        adjacent_zones=[2, 3],
    )


# =============================================================================
# ZBudgetHeader Tests
# =============================================================================


class TestZBudgetHeader:
    """Tests for ZBudgetHeader dataclass."""

    def test_header_defaults(self):
        """Test default values."""
        header = ZBudgetHeader()

        assert header.software_version == ""
        assert header.descriptor == ""
        assert header.n_data == 0
        assert header.n_layers == 0
        assert header.n_timesteps == 0

    def test_header_custom(self):
        """Test custom values."""
        header = ZBudgetHeader(
            software_version="IWFM 2025.0",
            descriptor="GW ZONE BUDGET",
            n_data=5,
            n_layers=3,
            n_timesteps=12,
        )

        assert header.software_version == "IWFM 2025.0"
        assert header.n_data == 5
        assert header.n_layers == 3


# =============================================================================
# ZoneInfo Tests
# =============================================================================


class TestZoneInfo:
    """Tests for ZoneInfo dataclass."""

    def test_zone_info_defaults(self):
        """Test default values."""
        zone = ZoneInfo(id=1, name="Zone 1")

        assert zone.id == 1
        assert zone.name == "Zone 1"
        assert zone.n_elements == 0
        assert zone.element_ids == []
        assert zone.area == 0.0

    def test_zone_info_custom(self, sample_zone_info):
        """Test custom values."""
        assert sample_zone_info.id == 1
        assert sample_zone_info.name == "Test Zone"
        assert sample_zone_info.n_elements == 10
        assert len(sample_zone_info.element_ids) == 10
        assert sample_zone_info.area == 5000.0
        assert sample_zone_info.adjacent_zones == [2, 3]


# =============================================================================
# ZBudgetReader Initialization Tests
# =============================================================================


class TestZBudgetReaderInit:
    """Tests for ZBudgetReader initialization."""

    def test_init(self, mock_zbudget_file):
        """Test basic initialization."""
        reader = ZBudgetReader(mock_zbudget_file)

        assert reader.filepath == mock_zbudget_file
        assert reader.descriptor == "GROUNDWATER ZONE BUDGET"

    def test_init_file_not_found(self, tmp_path):
        """Test initialization with non-existent file."""
        with pytest.raises(FileNotFoundError):
            ZBudgetReader(tmp_path / "nonexistent.hdf")


# =============================================================================
# ZBudgetReader Properties Tests
# =============================================================================


class TestZBudgetReaderProperties:
    """Tests for ZBudgetReader properties."""

    def test_descriptor_property(self, mock_zbudget_file):
        """Test descriptor property."""
        reader = ZBudgetReader(mock_zbudget_file)
        assert reader.descriptor == "GROUNDWATER ZONE BUDGET"

    def test_zones_property(self, mock_zbudget_file):
        """Test zones property."""
        reader = ZBudgetReader(mock_zbudget_file)
        assert reader.zones == ["Zone 1", "Zone 2", "Zone 3"]

    def test_n_zones_property(self, mock_zbudget_file):
        """Test n_zones property."""
        reader = ZBudgetReader(mock_zbudget_file)
        assert reader.n_zones == 3

    def test_data_names_property(self, mock_zbudget_file):
        """Test data_names property."""
        reader = ZBudgetReader(mock_zbudget_file)
        assert len(reader.data_names) == 4
        assert "Deep Percolation" in reader.data_names
        assert "Pumping" in reader.data_names

    def test_n_timesteps_property(self, mock_zbudget_file):
        """Test n_timesteps property."""
        reader = ZBudgetReader(mock_zbudget_file)
        assert reader.n_timesteps == 12


# =============================================================================
# Zone Info Tests
# =============================================================================


class TestZoneInfoRetrieval:
    """Tests for zone information retrieval."""

    def test_get_zone_info_by_name(self, mock_zbudget_file):
        """Test getting zone info by name."""
        reader = ZBudgetReader(mock_zbudget_file)
        zone = reader.get_zone_info("Zone 1")

        assert zone.name == "Zone 1"
        assert zone.id == 1
        assert len(zone.element_ids) == 10
        assert zone.area == 1000.0

    def test_get_zone_info_by_index(self, mock_zbudget_file):
        """Test getting zone info by index."""
        reader = ZBudgetReader(mock_zbudget_file)
        zone = reader.get_zone_info(1)  # 1-based index

        assert zone.name == "Zone 1"

    def test_get_zone_info_case_insensitive(self, mock_zbudget_file):
        """Test case-insensitive zone name matching."""
        reader = ZBudgetReader(mock_zbudget_file)
        zone = reader.get_zone_info("zone 1")

        assert zone.name == "Zone 1"

    def test_get_zone_info_invalid_name(self, mock_zbudget_file):
        """Test invalid zone name raises error."""
        reader = ZBudgetReader(mock_zbudget_file)

        with pytest.raises(KeyError):
            reader.get_zone_info("Invalid Zone")

    def test_get_zone_info_invalid_index(self, mock_zbudget_file):
        """Test invalid zone index raises error."""
        reader = ZBudgetReader(mock_zbudget_file)

        with pytest.raises(IndexError):
            reader.get_zone_info(10)


# =============================================================================
# Element Data Tests
# =============================================================================


class TestElementData:
    """Tests for element-level data reading."""

    def test_get_element_data_by_name(self, mock_zbudget_file):
        """Test reading element data by column name."""
        reader = ZBudgetReader(mock_zbudget_file)
        data = reader.get_element_data("Deep Percolation", layer=1)

        assert data.shape == (12, 30)  # n_timesteps x n_elements

    def test_get_element_data_by_index(self, mock_zbudget_file):
        """Test reading element data by column index."""
        reader = ZBudgetReader(mock_zbudget_file)
        data = reader.get_element_data(0, layer=1)

        assert data.shape == (12, 30)

    def test_get_element_data_layer_2(self, mock_zbudget_file):
        """Test reading element data from layer 2."""
        reader = ZBudgetReader(mock_zbudget_file)
        data = reader.get_element_data("Pumping", layer=2)

        assert data.shape == (12, 30)

    def test_get_element_data_invalid_name(self, mock_zbudget_file):
        """Test invalid data name raises error."""
        reader = ZBudgetReader(mock_zbudget_file)

        with pytest.raises(KeyError):
            reader.get_element_data("Invalid Column", layer=1)


# =============================================================================
# Zone Data Tests
# =============================================================================


class TestZoneData:
    """Tests for zone-level data reading."""

    def test_get_zone_data(self, mock_zbudget_file):
        """Test reading zone data."""
        reader = ZBudgetReader(mock_zbudget_file)
        times, values = reader.get_zone_data("Zone 1", layer=1)

        assert len(times) == 12
        assert values.shape[0] == 12  # n_timesteps

    def test_get_zone_data_specific_column(self, mock_zbudget_file):
        """Test reading zone data for specific column."""
        reader = ZBudgetReader(mock_zbudget_file)
        times, values = reader.get_zone_data("Zone 1", data_name="Pumping", layer=1)

        assert len(times) == 12
        # Single column should be 1D
        assert values.ndim == 1

    def test_get_zone_data_all_columns(self, mock_zbudget_file):
        """Test reading zone data for all columns."""
        reader = ZBudgetReader(mock_zbudget_file)
        times, values = reader.get_zone_data("Zone 1", layer=1)

        assert values.shape == (12, 4)  # n_timesteps x n_data

    def test_get_zone_data_by_index(self, mock_zbudget_file):
        """Test reading zone data by zone index."""
        reader = ZBudgetReader(mock_zbudget_file)
        times, values = reader.get_zone_data(2, layer=1)  # Zone 2

        assert len(times) == 12


# =============================================================================
# DataFrame Tests
# =============================================================================


class TestDataFrame:
    """Tests for DataFrame conversion."""

    def test_get_dataframe(self, mock_zbudget_file):
        """Test getting DataFrame."""
        pd = pytest.importorskip("pandas")
        reader = ZBudgetReader(mock_zbudget_file)
        df = reader.get_dataframe("Zone 1", layer=1)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 12
        assert len(df.columns) == 4

    def test_get_dataframe_with_columns(self, mock_zbudget_file):
        """Test getting DataFrame with specific columns."""
        pytest.importorskip("pandas")
        reader = ZBudgetReader(mock_zbudget_file)
        df = reader.get_dataframe("Zone 1", layer=1, data_columns=["Pumping", "Deep Percolation"])

        assert len(df.columns) == 2
        assert "Pumping" in df.columns

    def test_get_all_zones_dataframe(self, mock_zbudget_file):
        """Test getting DataFrame for all zones."""
        pd = pytest.importorskip("pandas")
        reader = ZBudgetReader(mock_zbudget_file)
        df = reader.get_all_zones_dataframe("Pumping", layer=1)

        assert isinstance(df, pd.DataFrame)
        assert len(df.columns) == 3  # 3 zones
        assert "Zone 1" in df.columns
        assert "Zone 2" in df.columns
        assert "Zone 3" in df.columns


# =============================================================================
# Aggregation Tests
# =============================================================================


class TestAggregation:
    """Tests for data aggregation methods."""

    def test_get_monthly_averages(self, mock_zbudget_file):
        """Test monthly average calculation."""
        pd = pytest.importorskip("pandas")
        reader = ZBudgetReader(mock_zbudget_file)
        df = reader.get_monthly_averages("Zone 1", layer=1)

        assert isinstance(df, pd.DataFrame)

    def test_get_annual_totals(self, mock_zbudget_file):
        """Test annual total calculation."""
        pd = pytest.importorskip("pandas")
        reader = ZBudgetReader(mock_zbudget_file)
        df = reader.get_annual_totals("Zone 1", layer=1)

        assert isinstance(df, pd.DataFrame)


# =============================================================================
# Water Balance Tests
# =============================================================================


class TestWaterBalance:
    """Tests for water balance calculation."""

    def test_get_water_balance(self, mock_zbudget_file):
        """Test water balance calculation."""
        pd = pytest.importorskip("pandas")
        reader = ZBudgetReader(mock_zbudget_file)
        df = reader.get_water_balance("Zone 1", layer=1)

        assert isinstance(df, pd.DataFrame)
        assert "Total Inflow" in df.columns
        assert "Total Outflow" in df.columns
        assert "Net Balance" in df.columns

    def test_get_water_balance_custom_columns(self, mock_zbudget_file):
        """Test water balance with custom inflow/outflow columns."""
        pytest.importorskip("pandas")
        reader = ZBudgetReader(mock_zbudget_file)
        df = reader.get_water_balance(
            "Zone 1",
            layer=1,
            inflow_columns=["Deep Percolation"],
            outflow_columns=["Pumping"],
        )

        assert "Total Inflow" in df.columns
        assert "Total Outflow" in df.columns


# =============================================================================
# Export Tests
# =============================================================================


class TestExport:
    """Tests for data export."""

    def test_to_csv(self, mock_zbudget_file, tmp_path):
        """Test CSV export."""
        pytest.importorskip("pandas")
        reader = ZBudgetReader(mock_zbudget_file)
        output_dir = tmp_path / "output"

        files = reader.to_csv(output_dir, layer=1)

        assert len(files) == 3  # 3 zones
        assert all(f.exists() for f in files)
        assert all(f.suffix == ".csv" for f in files)

    def test_to_csv_specific_zones(self, mock_zbudget_file, tmp_path):
        """Test CSV export for specific zones."""
        pytest.importorskip("pandas")
        reader = ZBudgetReader(mock_zbudget_file)
        output_dir = tmp_path / "output"

        files = reader.to_csv(output_dir, zones=["Zone 1"], layer=1)

        assert len(files) == 1


# =============================================================================
# Representation Tests
# =============================================================================


class TestRepresentation:
    """Tests for string representation."""

    def test_repr(self, mock_zbudget_file):
        """Test __repr__ output."""
        reader = ZBudgetReader(mock_zbudget_file)
        repr_str = repr(reader)

        assert "ZBudgetReader" in repr_str
        assert "GROUNDWATER ZONE BUDGET" in repr_str
        assert "n_zones=3" in repr_str


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_zbudget_data_types(self):
        """Test ZBudget data type constants."""
        assert ZBUDGET_DATA_TYPES[1] == "Storage"
        assert ZBUDGET_DATA_TYPES[2] == "VerticalFlow"
        assert ZBUDGET_DATA_TYPES[3] == "FaceFlow"
        assert ZBUDGET_DATA_TYPES[4] == "ElementData"


# =============================================================================
# Additional coverage tests
# =============================================================================


@pytest.fixture
def mock_zbudget_with_datetime(tmp_path):
    """Create a ZBudget HDF5 file that has start_datetime set via attrs."""
    pytest.importorskip("h5py")
    import h5py

    filepath = tmp_path / "zbudget_dt.hdf"
    rng = np.random.default_rng(99)
    n_elements = 10
    n_timesteps = 24  # 2 years of monthly data

    with h5py.File(filepath, "w") as f:
        attrs = f.create_group("Attributes")
        attrs.create_dataset("Software_Version", data="IWFM 2025.0")
        attrs.create_dataset("Descriptor", data="GW ZONE BUDGET WITH DATETIME")
        attrs.create_dataset("NData", data=3)
        attrs.create_dataset("DataTypes", data=[1, 4, 4])
        attrs.create_dataset(
            "FullDataNames",
            data=[b"Recharge Inflow", b"Pumping Outflow", b"ET Evaporation"],
        )
        attrs.create_dataset(
            "DataHDFPaths",
            data=[b"/Recharge", b"/Pumping", b"/ET"],
        )
        attrs.create_dataset("NTimeSteps", data=n_timesteps)
        # Layer ElemDataColumns keys to test n_layers derivation
        attrs.create_dataset(
            "Layer1_ElemDataColumns", data=np.ones((3, n_elements), dtype=np.int32)
        )
        attrs.create_dataset(
            "Layer2_ElemDataColumns", data=np.ones((3, n_elements), dtype=np.int32)
        )

        zone_list = f.create_group("ZoneList")
        zone_list.attrs["NZones"] = 2
        zone_list.create_dataset("ZoneNames", data=[b"Zone A", b"Zone B"])
        for i in range(2):
            zg = zone_list.create_group(f"Zone_{i + 1}")
            zg.create_dataset("Elements", data=list(range(i * 5 + 1, (i + 1) * 5 + 1)))
            zg.create_dataset("Area", data=500.0 * (i + 1))

        for path in ["/Recharge", "/Pumping", "/ET"]:
            dg = f.create_group(path.strip("/"))
            data = rng.random((n_timesteps, n_elements)) * 50
            dg.create_dataset("Layer_1", data=data)
            dg.create_dataset("Layer_2", data=data * 0.3)

    return filepath


@pytest.fixture
def mock_zbudget_no_zonelist(tmp_path):
    """Create a ZBudget HDF5 file without a ZoneList group."""
    pytest.importorskip("h5py")
    import h5py

    filepath = tmp_path / "zbudget_nozone.hdf"
    with h5py.File(filepath, "w") as f:
        attrs = f.create_group("Attributes")
        attrs.create_dataset("Software_Version", data="IWFM 2025.0")
        attrs.create_dataset("Descriptor", data="BUDGET NO ZONES")
        attrs.create_dataset("NData", data=1)
        attrs.create_dataset("DataTypes", data=[4])
        attrs.create_dataset("FullDataNames", data=[b"Flow"])
        attrs.create_dataset("DataHDFPaths", data=[b"/Flow"])
        attrs.create_dataset("NTimeSteps", data=5)
        dg = f.create_group("Flow")
        dg.create_dataset("Layer_1", data=np.ones((5, 10)))

    return filepath


@pytest.fixture
def mock_zbudget_no_attrs_group(tmp_path):
    """Create a ZBudget HDF5 file without an 'Attributes' group (root attrs)."""
    pytest.importorskip("h5py")
    import h5py

    filepath = tmp_path / "zbudget_rootattrs.hdf"
    with h5py.File(filepath, "w") as f:
        # Put attributes directly at root level (no Attributes group)
        f.create_dataset("Software_Version", data="IWFM 2025.0")
        f.create_dataset("Descriptor", data="ROOT LEVEL BUDGET")
        f.create_dataset("NData", data=1)
        f.create_dataset("DataTypes", data=[4])
        f.create_dataset("FullDataNames", data=[b"Test Data"])
        f.create_dataset("DataHDFPaths", data=[b"/TestData"])
        f.create_dataset("NTimeSteps", data=3)
        dg = f.create_group("TestData")
        dg.create_dataset("Layer_1", data=np.ones((3, 5)))

    return filepath


@pytest.fixture
def mock_zbudget_nzones_dataset(tmp_path):
    """Create a ZBudget HDF5 file where NZones is a dataset, not an attribute."""
    pytest.importorskip("h5py")
    import h5py

    filepath = tmp_path / "zbudget_nzones_ds.hdf"
    with h5py.File(filepath, "w") as f:
        attrs = f.create_group("Attributes")
        attrs.create_dataset("Descriptor", data="NZones Dataset Test")
        attrs.create_dataset("NData", data=1)
        attrs.create_dataset("DataTypes", data=[4])
        attrs.create_dataset("FullDataNames", data=[b"Flow"])
        attrs.create_dataset("DataHDFPaths", data=[b"/Flow"])
        attrs.create_dataset("NTimeSteps", data=4)

        zone_list = f.create_group("ZoneList")
        # NZones as a dataset instead of attribute
        zone_list.create_dataset("NZones", data=1)
        # No ZoneNames - should auto-generate

        dg = f.create_group("Flow")
        dg.create_dataset("Layer_1", data=np.ones((4, 5)))

    return filepath


class TestZBudgetReaderHeaderEdgeCases:
    """Additional tests for header parsing edge cases."""

    def test_header_no_attributes_group(self, mock_zbudget_no_attrs_group):
        """Reader uses root-level attributes when no Attributes group exists."""
        reader = ZBudgetReader(mock_zbudget_no_attrs_group)
        assert reader.descriptor == "ROOT LEVEL BUDGET"

    def test_header_layer_elem_columns_sets_n_layers(self, mock_zbudget_with_datetime):
        """n_layers is determined from Layer*_ElemDataColumns keys."""
        reader = ZBudgetReader(mock_zbudget_with_datetime)
        assert reader.n_layers == 2


class TestZBudgetReaderZoneEdgeCases:
    """Additional tests for zone info edge cases."""

    def test_no_zone_list(self, mock_zbudget_no_zonelist):
        """Reader handles missing ZoneList gracefully."""
        reader = ZBudgetReader(mock_zbudget_no_zonelist)
        assert reader.zones == []
        assert reader.n_zones == 0

    def test_zone_info_index_zero(self, mock_zbudget_file):
        """Zone index 0 is out of range (1-based)."""
        reader = ZBudgetReader(mock_zbudget_file)
        with pytest.raises(IndexError):
            reader.get_zone_info(0)

    def test_nzones_as_dataset(self, mock_zbudget_nzones_dataset):
        """Reader handles NZones as a dataset in ZoneList."""
        reader = ZBudgetReader(mock_zbudget_nzones_dataset)
        # Auto-generated zone names since no ZoneNames dataset
        assert reader.zones == ["Zone 1"]


class TestElementDataEdgeCases:
    """Additional tests for element data retrieval."""

    def test_get_element_data_invalid_index(self, mock_zbudget_file):
        """Data index out of range raises IndexError."""
        reader = ZBudgetReader(mock_zbudget_file)
        with pytest.raises(IndexError, match="out of range"):
            reader.get_element_data(99, layer=1)

    def test_get_element_data_case_insensitive(self, mock_zbudget_file):
        """Case-insensitive data name lookup works."""
        reader = ZBudgetReader(mock_zbudget_file)
        data = reader.get_element_data("deep percolation", layer=1)
        assert data.shape == (12, 30)

    def test_get_element_data_nonexistent_path(self, mock_zbudget_file):
        """Reading data from a non-existent HDF5 path raises KeyError."""
        reader = ZBudgetReader(mock_zbudget_file)
        # Temporarily modify the header to have a wrong path
        reader.header.data_hdf_paths[0] = "/NonExistentPath"
        # Also change the data name so neither strategy finds data
        reader.header.data_names[0] = "NonExistent Name"
        with pytest.raises(KeyError, match="not found in file"):
            reader.get_element_data(0, layer=1)

    def test_get_element_data_fallback_path(self, mock_zbudget_file):
        """When data_idx >= len(data_hdf_paths), uses name-based path fallback."""
        reader = ZBudgetReader(mock_zbudget_file)
        # Truncate paths to force fallback
        reader.header.data_hdf_paths = []
        # The fallback will try name "Deep Percolation" which doesn't match a
        # Layer_N group and data_name/Layer_N won't work either
        with pytest.raises(KeyError):
            reader.get_element_data(0, layer=1)


class TestZoneDataEdgeCases:
    """Additional tests for zone data retrieval."""

    def test_get_zone_data_by_integer_data_name(self, mock_zbudget_file):
        """Pass data_name as integer to get_zone_data."""
        reader = ZBudgetReader(mock_zbudget_file)
        times, values = reader.get_zone_data("Zone 1", data_name=0, layer=1)
        assert len(times) == 12
        assert values.ndim == 1

    def test_get_zone_data_case_insensitive_data_name(self, mock_zbudget_file):
        """Case-insensitive data_name lookup in get_zone_data."""
        reader = ZBudgetReader(mock_zbudget_file)
        times, values = reader.get_zone_data("Zone 1", data_name="pumping", layer=1)
        assert values.ndim == 1
        assert len(times) == 12

    def test_get_zone_data_no_elem_ids_by_name(self, mock_zbudget_file):
        """get_zone_data handles empty element_ids for a zone accessed by name."""
        reader = ZBudgetReader(mock_zbudget_file)
        # Clear element IDs to exercise the fallback path
        reader._zone_info["Zone 1"].element_ids = []
        times, values = reader.get_zone_data("Zone 1", data_name="Pumping", layer=1)
        assert len(times) == 12

    def test_get_zone_data_no_elem_ids_by_index(self, mock_zbudget_file):
        """get_zone_data with no elem_ids and zone passed as integer."""
        reader = ZBudgetReader(mock_zbudget_file)
        reader._zone_info["Zone 2"].element_ids = []
        times, values = reader.get_zone_data(2, data_name="Pumping", layer=1)
        assert len(times) == 12

    def test_get_zone_data_with_start_datetime(self, mock_zbudget_with_datetime):
        """Zone data with a start_datetime produces timestamp-based times."""
        reader = ZBudgetReader(mock_zbudget_with_datetime)
        reader.header.start_datetime = datetime(2000, 1, 1)
        reader.header.delta_t_minutes = 43200  # 30 days in minutes
        times, values = reader.get_zone_data("Zone A", layer=1)
        assert len(times) == 24
        # First time should be the start timestamp
        assert times[0] == pytest.approx(datetime(2000, 1, 1).timestamp())


class TestDataFrameEdgeCases:
    """Additional tests for DataFrame conversion edge cases."""

    def test_get_dataframe_nonexistent_columns_filtered(self, mock_zbudget_file):
        """Requesting non-existent data_columns filters to only available ones."""
        pytest.importorskip("pandas")
        reader = ZBudgetReader(mock_zbudget_file)
        df = reader.get_dataframe(
            "Zone 1",
            layer=1,
            data_columns=["Pumping", "NonExistent Column"],
        )
        assert "Pumping" in df.columns
        assert "NonExistent Column" not in df.columns

    def test_get_dataframe_with_datetime_index(self, mock_zbudget_with_datetime):
        """DataFrame with start_datetime has DatetimeIndex."""
        pd = pytest.importorskip("pandas")
        reader = ZBudgetReader(mock_zbudget_with_datetime)
        reader.header.start_datetime = datetime(2000, 1, 1)
        reader.header.delta_t_minutes = 43200
        df = reader.get_dataframe("Zone A", layer=1)
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_get_dataframe_without_datetime_index(self, mock_zbudget_file):
        """DataFrame without start_datetime has numeric index."""
        pd = pytest.importorskip("pandas")
        reader = ZBudgetReader(mock_zbudget_file)
        df = reader.get_dataframe("Zone 1", layer=1)
        # No start_datetime => numeric index
        assert not isinstance(df.index, pd.DatetimeIndex)


class TestAggregationEdgeCases:
    """Additional tests for aggregation methods."""

    def test_get_monthly_averages_with_datetime(self, mock_zbudget_with_datetime):
        """Monthly averages with DatetimeIndex resamples correctly."""
        pd = pytest.importorskip("pandas")
        reader = ZBudgetReader(mock_zbudget_with_datetime)
        reader.header.start_datetime = datetime(2000, 1, 1)
        reader.header.delta_t_minutes = 43200  # ~30 days
        df = reader.get_monthly_averages("Zone A", layer=1)
        assert isinstance(df, pd.DataFrame)

    def test_get_monthly_averages_without_datetime(self, mock_zbudget_file):
        """Monthly averages without DatetimeIndex returns the data unchanged."""
        pd = pytest.importorskip("pandas")
        reader = ZBudgetReader(mock_zbudget_file)
        df = reader.get_monthly_averages("Zone 1", layer=1)
        # Without DatetimeIndex, just returns df as-is
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 12

    def test_get_annual_totals_with_datetime(self, mock_zbudget_with_datetime):
        """Annual totals with DatetimeIndex resamples correctly."""
        pd = pytest.importorskip("pandas")
        reader = ZBudgetReader(mock_zbudget_with_datetime)
        reader.header.start_datetime = datetime(2000, 1, 1)
        reader.header.delta_t_minutes = 43200
        df = reader.get_annual_totals("Zone A", layer=1)
        assert isinstance(df, pd.DataFrame)

    def test_get_annual_totals_without_datetime_enough_months(self, mock_zbudget_file):
        """Annual totals without DatetimeIndex uses 12-row chunking."""
        pd = pytest.importorskip("pandas")
        reader = ZBudgetReader(mock_zbudget_file)
        df = reader.get_annual_totals("Zone 1", layer=1)
        # 12 rows / 12 = 1 year
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_get_annual_totals_without_datetime_few_rows(self, tmp_path):
        """Annual totals without DatetimeIndex and < 12 rows returns sum."""
        pd = pytest.importorskip("pandas")
        h5py = pytest.importorskip("h5py")

        # Create a small file with only 6 timesteps
        filepath = tmp_path / "zbudget_6ts.hdf"
        n_elements = 5
        n_timesteps = 6

        with h5py.File(filepath, "w") as f:
            attrs = f.create_group("Attributes")
            attrs.create_dataset("Descriptor", data="SHORT BUDGET")
            attrs.create_dataset("NData", data=1)
            attrs.create_dataset("DataTypes", data=[4])
            attrs.create_dataset("FullDataNames", data=[b"Flow"])
            attrs.create_dataset("DataHDFPaths", data=[b"/Flow"])
            attrs.create_dataset("NTimeSteps", data=n_timesteps)

            zone_list = f.create_group("ZoneList")
            zone_list.attrs["NZones"] = 1
            zone_list.create_dataset("ZoneNames", data=[b"Zone X"])
            zg = zone_list.create_group("Zone_1")
            zg.create_dataset("Elements", data=[1, 2, 3, 4, 5])

            dg = f.create_group("Flow")
            dg.create_dataset("Layer_1", data=np.ones((n_timesteps, n_elements)))

        reader = ZBudgetReader(filepath)
        # No start_datetime => non-DatetimeIndex path
        df = reader.get_annual_totals("Zone X", layer=1)
        assert isinstance(df, pd.DataFrame)
        # 6 rows / 12 = 0 years => returns sum as single-row DF
        assert len(df) == 1


class TestWaterBalanceEdgeCases:
    """Additional tests for water balance calculation."""

    def test_water_balance_no_matching_inflows(self, mock_zbudget_file):
        """Auto-detected inflow columns may be empty if no keyword match."""
        pytest.importorskip("pandas")
        reader = ZBudgetReader(mock_zbudget_file)
        # The test fixture has "Deep Percolation" which matches "percolation"
        # so at least 1 inflow is auto-detected; test with explicit empty
        df = reader.get_water_balance(
            "Zone 1",
            layer=1,
            inflow_columns=[],
            outflow_columns=[],
        )
        assert (df["Total Inflow"] == 0.0).all()
        assert (df["Total Outflow"] == 0.0).all()

    def test_water_balance_auto_detect(self, mock_zbudget_with_datetime):
        """Auto-detection of inflow/outflow columns based on keywords."""
        pytest.importorskip("pandas")
        reader = ZBudgetReader(mock_zbudget_with_datetime)
        # Data names are "Recharge Inflow", "Pumping Outflow", "ET Evaporation"
        # "Recharge Inflow" matches "inflow" and "recharge"
        # "Pumping Outflow" matches "outflow" and "pump"
        # "ET Evaporation" matches "evap" and "et"
        df = reader.get_water_balance("Zone A", layer=1)
        assert "Total Inflow" in df.columns
        assert "Total Outflow" in df.columns
        assert "Net Balance" in df.columns


class TestExportEdgeCases:
    """Additional tests for CSV export edge cases."""

    def test_to_csv_creates_directory(self, mock_zbudget_file, tmp_path):
        """to_csv creates the output directory if it doesn't exist."""
        pytest.importorskip("pandas")
        reader = ZBudgetReader(mock_zbudget_file)
        output_dir = tmp_path / "deep" / "nested" / "dir"
        files = reader.to_csv(output_dir, zones=["Zone 1"], layer=1)
        assert len(files) == 1
        assert files[0].exists()

    def test_to_csv_zone_name_sanitization(self, mock_zbudget_file, tmp_path):
        """CSV filenames sanitize zone names (spaces -> underscores)."""
        pytest.importorskip("pandas")
        reader = ZBudgetReader(mock_zbudget_file)
        files = reader.to_csv(tmp_path / "csv_out", zones=["Zone 1"], layer=1)
        assert "Zone_1" in files[0].name


class TestReprAdditional:
    """Additional representation tests."""

    def test_repr_includes_all_fields(self, mock_zbudget_file):
        """repr includes all key fields."""
        reader = ZBudgetReader(mock_zbudget_file)
        repr_str = repr(reader)
        assert "n_layers=" in repr_str
        assert "n_timesteps=" in repr_str

    def test_repr_with_datetime_file(self, mock_zbudget_with_datetime):
        """repr works with datetime-enabled file."""
        reader = ZBudgetReader(mock_zbudget_with_datetime)
        repr_str = repr(reader)
        assert "ZBudgetReader" in repr_str


# =============================================================================
# C2VSimFG-style structure tests (Layer_N/data_name + ElemDataColumns)
# =============================================================================


@pytest.fixture
def mock_zbudget_c2vsim_structure(tmp_path):
    """Create a ZBudget HDF5 file matching real C2VSimFG layout.

    Structure:
    - Scalar metadata in /Attributes/.attrs (not datasets)
    - Layer_N/data_name path hierarchy
    - ElemDataColumns for sparse datasets
    - Some datasets have n_items < n_elements
    """
    pytest.importorskip("h5py")
    import h5py

    filepath = tmp_path / "c2vsim_gw_zbudget.hdf"
    n_elements = 20
    n_timesteps = 6
    n_layers = 2
    n_data = 3

    # Sparse dataset: only 8 out of 20 elements have stream data
    n_stream_items = 8
    stream_elems = [2, 5, 7, 10, 12, 14, 17, 19]  # 1-based elem IDs

    with h5py.File(filepath, "w") as f:
        ag = f.create_group("Attributes")

        # Scalars as HDF5 attrs (NOT datasets)
        ag.attrs["NTimeSteps"] = n_timesteps
        ag.attrs["Descriptor"] = "GROUNDWATER ZONE BUDGET"
        ag.attrs["Software_Version"] = "IWFM 2025.0"
        ag.attrs["NData"] = n_data
        ag.attrs["SystemData%NElements"] = n_elements
        ag.attrs["SystemData%NLayers"] = n_layers
        ag.attrs["lVertFlows_DefinedAtNode"] = False
        ag.attrs["lFaceFlows_Defined"] = True
        ag.attrs["lStorages_Defined"] = True
        ag.attrs["lComputeError"] = True
        ag.attrs["TimeStep%BeginDateAndTime"] = "10/31/1973_24:00"
        ag.attrs["TimeStep%Unit"] = "1MON"
        ag.attrs["TimeStep%DeltaT_InMinutes"] = 43200

        # Array datasets under Attributes group
        ag.create_dataset(
            "FullDataNames",
            data=[b"GW Storage_Inflow (+)", b"Streams_Inflow (+)", b"Pumping by Well_Inflow (+)"],
        )
        ag.create_dataset(
            "DataHDFPaths",
            data=[b"/GW Storage", b"/Streams", b"/Pumping"],
        )
        ag.create_dataset("DataTypes", data=[1, 4, 4])

        # ElemDataColumns — shape (n_data, n_elements)
        rng = np.random.default_rng(42)
        for layer_num in range(1, n_layers + 1):
            edc = np.zeros((n_data, n_elements), dtype=np.int32)
            # Data 0 (GW Storage): all elements → 1:1 mapping
            for j in range(n_elements):
                edc[0, j] = j + 1  # 1-based column index
            # Data 1 (Streams): only stream_elems have data
            for col_idx, eid in enumerate(stream_elems):
                edc[1, eid - 1] = col_idx + 1  # 1-based compressed column
            # Data 2 (Pumping): all elements
            for j in range(n_elements):
                edc[2, j] = j + 1
            ag.create_dataset(f"Layer{layer_num}_ElemDataColumns", data=edc)

        # Layer_N/data_name datasets
        for layer_num in range(1, n_layers + 1):
            lg = f.create_group(f"Layer_{layer_num}")
            # GW Storage: all elements
            lg.create_dataset(
                "GW Storage_Inflow (+)",
                data=rng.random((n_timesteps, n_elements)) * 100,
            )
            # Streams: sparse — only 8 items
            lg.create_dataset(
                "Streams_Inflow (+)",
                data=rng.random((n_timesteps, n_stream_items)) * 50,
            )
            # Pumping: all elements
            lg.create_dataset(
                "Pumping by Well_Inflow (+)",
                data=rng.random((n_timesteps, n_elements)) * 30,
            )

    return filepath


class TestC2VSimFGStructure:
    """Tests for real C2VSimFG HDF5 layout."""

    def test_header_reads_attrs_metadata(self, mock_zbudget_c2vsim_structure):
        """n_timesteps, descriptor, start_datetime read from .attrs."""
        reader = ZBudgetReader(mock_zbudget_c2vsim_structure)
        assert reader.n_timesteps == 6
        assert reader.descriptor == "GROUNDWATER ZONE BUDGET"
        assert reader.header.start_datetime is not None
        # 10/31/1973_24:00 → 11/01/1973 00:00
        assert reader.header.start_datetime == datetime(1973, 11, 1, 0, 0)

    def test_header_reads_system_data(self, mock_zbudget_c2vsim_structure):
        """n_elements, n_layers read from SystemData% attrs."""
        reader = ZBudgetReader(mock_zbudget_c2vsim_structure)
        assert reader.header.n_elements == 20
        assert reader.header.n_layers == 2

    def test_header_reads_elem_data_columns(self, mock_zbudget_c2vsim_structure):
        """ElemDataColumns arrays are loaded into header."""
        reader = ZBudgetReader(mock_zbudget_c2vsim_structure)
        assert 1 in reader.header.elem_data_columns
        assert 2 in reader.header.elem_data_columns
        edc1 = reader.header.elem_data_columns[1]
        assert edc1.shape == (3, 20)  # n_data x n_elements

    def test_get_element_data_layer_first_structure(self, mock_zbudget_c2vsim_structure):
        """Layer_N/data_name path works for element data retrieval."""
        reader = ZBudgetReader(mock_zbudget_c2vsim_structure)
        # Full-elements dataset
        data = reader.get_element_data("GW Storage_Inflow (+)", layer=1)
        assert data.shape == (6, 20)  # n_timesteps x n_elements
        # Sparse dataset
        data = reader.get_element_data("Streams_Inflow (+)", layer=1)
        assert data.shape == (6, 8)  # n_timesteps x n_stream_items

    def test_get_zone_data_with_elem_data_columns(self, mock_zbudget_c2vsim_structure):
        """Sparse aggregation using ElemDataColumns mapping."""
        reader = ZBudgetReader(mock_zbudget_c2vsim_structure)
        # Inject a zone with elements that include some stream-adjacent ones
        reader._zone_info["TestZone"] = ZoneInfo(
            id=1,
            name="TestZone",
            n_elements=5,
            element_ids=[1, 2, 5, 10, 15],  # elems 2, 5, 10 have stream data
        )
        times, values = reader.get_zone_data("TestZone", data_name="Streams_Inflow (+)", layer=1)
        assert len(times) == 6
        assert values.ndim == 1
        # Values should be non-zero (3 elements contribute)
        assert np.any(values != 0)

    def test_elem_data_columns_subset_aggregation(self, mock_zbudget_c2vsim_structure):
        """Elements not in sparse dataset are correctly skipped."""
        reader = ZBudgetReader(mock_zbudget_c2vsim_structure)
        # Zone with elements that have NO stream data
        reader._zone_info["NoStreamZone"] = ZoneInfo(
            id=2,
            name="NoStreamZone",
            n_elements=3,
            element_ids=[1, 3, 4],  # none in stream_elems
        )
        times, values = reader.get_zone_data(
            "NoStreamZone", data_name="Streams_Inflow (+)", layer=1
        )
        assert len(times) == 6
        # All zeros — none of these elements have stream data
        assert np.all(values == 0)

    def test_get_dataframe_monthly_timestamps(self, mock_zbudget_c2vsim_structure):
        """Monthly timesteps use relativedelta for proper dates."""
        pd = pytest.importorskip("pandas")
        reader = ZBudgetReader(mock_zbudget_c2vsim_structure)
        reader._zone_info["TestZone"] = ZoneInfo(
            id=1,
            name="TestZone",
            n_elements=3,
            element_ids=[1, 2, 3],
        )
        df = reader.get_dataframe("TestZone", layer=1)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert len(df) == 6
        # Check monthly increments
        assert df.index[0].year == 1973
        assert df.index[0].month == 11
        assert df.index[1].month == 12
        assert df.index[2].year == 1974
        assert df.index[2].month == 1

    def test_both_path_structures_work(self, mock_zbudget_file, mock_zbudget_c2vsim_structure):
        """Old fixture structure (data_name/Layer_N) and new (Layer_N/data_name) both work."""
        # Old structure
        reader_old = ZBudgetReader(mock_zbudget_file)
        data_old = reader_old.get_element_data("Deep Percolation", layer=1)
        assert data_old.ndim == 2

        # New structure
        reader_new = ZBudgetReader(mock_zbudget_c2vsim_structure)
        data_new = reader_new.get_element_data("GW Storage_Inflow (+)", layer=1)
        assert data_new.ndim == 2

    def test_flags_from_attrs(self, mock_zbudget_c2vsim_structure):
        """Boolean flags read from .attrs correctly."""
        reader = ZBudgetReader(mock_zbudget_c2vsim_structure)
        assert reader.header.face_flows_defined is True
        assert reader.header.storages_defined is True
        assert reader.header.compute_error is True
        assert reader.header.vert_flows_at_node is False

    def test_time_unit_from_attrs(self, mock_zbudget_c2vsim_structure):
        """Time unit and delta read from attrs."""
        reader = ZBudgetReader(mock_zbudget_c2vsim_structure)
        assert reader.header.time_unit == "1MON"
        assert reader.header.delta_t_minutes == 43200
