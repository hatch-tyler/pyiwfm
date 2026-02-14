"""Unit tests for query API.

Tests:
- TimeSeries dataclass
- ModelQueryAPI class
- Data aggregation at different scales
- Export to DataFrame/CSV
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pyiwfm.core.query import TimeSeries, ModelQueryAPI
from pyiwfm.core.zones import Zone, ZoneDefinition
from pyiwfm.core.mesh import AppGrid, Node, Element, Subregion


# =============================================================================
# Test TimeSeries Dataclass
# =============================================================================


class TestTimeSeries:
    """Tests for TimeSeries dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic time series creation."""
        times = [datetime(2020, 1, 1), datetime(2020, 2, 1), datetime(2020, 3, 1)]
        values = np.array([10.0, 12.0, 11.0])

        ts = TimeSeries(
            times=times,
            values=values,
            variable="head",
            location_id=1,
            location_type="element",
        )

        assert ts.times == times
        assert np.array_equal(ts.values, values)
        assert ts.variable == "head"
        assert ts.location_id == 1
        assert ts.location_type == "element"
        assert ts.units == ""

    def test_with_units(self) -> None:
        """Test time series with units."""
        ts = TimeSeries(
            times=[datetime(2020, 1, 1)],
            values=np.array([100.0]),
            variable="pumping",
            location_id=5,
            location_type="zone",
            units="acre-ft/month",
        )

        assert ts.units == "acre-ft/month"

    def test_n_timesteps_property(self) -> None:
        """Test n_timesteps property."""
        times = [datetime(2020, i, 1) for i in range(1, 13)]
        values = np.random.random(12)

        ts = TimeSeries(
            times=times,
            values=values,
            variable="head",
            location_id=1,
            location_type="element",
        )

        assert ts.n_timesteps == 12

    def test_n_timesteps_empty(self) -> None:
        """Test n_timesteps for empty time series."""
        ts = TimeSeries(
            times=[],
            values=np.array([]),
            variable="head",
            location_id=1,
            location_type="element",
        )

        assert ts.n_timesteps == 0

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        times = [datetime(2020, 1, 1), datetime(2020, 2, 1)]
        values = np.array([10.0, 12.0])

        ts = TimeSeries(
            times=times,
            values=values,
            variable="head",
            location_id=1,
            location_type="element",
            units="ft",
        )

        result = ts.to_dict()

        assert result["time"] == times
        assert result["value"] == [10.0, 12.0]
        assert result["variable"] == "head"
        assert result["location_id"] == 1
        assert result["location_type"] == "element"
        assert result["units"] == "ft"


# =============================================================================
# Test ModelQueryAPI
# =============================================================================


class TestModelQueryAPIInit:
    """Tests for ModelQueryAPI initialization."""

    @pytest.fixture
    def mock_model(self) -> MagicMock:
        """Create a mock model for testing."""
        model = MagicMock()
        model.name = "Test Model"
        model.mesh = MagicMock()
        model.mesh.n_elements = 4
        model.mesh.elements = {
            1: MagicMock(area=100.0, subregion=1, vertices=(1, 2, 3, 4)),
            2: MagicMock(area=150.0, subregion=1, vertices=(2, 5, 6, 3)),
            3: MagicMock(area=120.0, subregion=2, vertices=(5, 7, 8, 6)),
            4: MagicMock(area=130.0, subregion=2, vertices=(7, 9, 10, 8)),
        }
        model.mesh.subregions = {
            1: MagicMock(id=1, name="Region A"),
            2: MagicMock(id=2, name="Region B"),
        }
        model.stratigraphy = None
        model.groundwater = None
        model.results = None
        return model

    def test_basic_creation(self, mock_model: MagicMock) -> None:
        """Test basic API creation."""
        api = ModelQueryAPI(mock_model)

        assert api.model == mock_model
        assert api._aggregator is None
        assert api._zone_definitions == {}

    def test_repr(self, mock_model: MagicMock) -> None:
        """Test string representation."""
        api = ModelQueryAPI(mock_model)

        result = repr(api)
        assert "ModelQueryAPI" in result
        assert "Test Model" in result


class TestModelQueryAPIAggregator:
    """Tests for ModelQueryAPI aggregator property."""

    @pytest.fixture
    def simple_grid(self) -> AppGrid:
        """Create a simple grid for testing."""
        nodes = {
            1: Node(id=1, x=0.0, y=0.0),
            2: Node(id=2, x=100.0, y=0.0),
            3: Node(id=3, x=100.0, y=100.0),
            4: Node(id=4, x=0.0, y=100.0),
        }
        elements = {
            1: Element(id=1, vertices=(1, 2, 3, 4), subregion=1, area=10000.0),
        }
        subregions = {1: Subregion(id=1, name="Region A")}
        return AppGrid(nodes=nodes, elements=elements, subregions=subregions)

    def test_aggregator_creation(self, simple_grid: AppGrid) -> None:
        """Test aggregator is created from grid."""
        model = MagicMock()
        model.mesh = simple_grid

        api = ModelQueryAPI(model)
        aggregator = api.aggregator

        assert aggregator is not None
        assert aggregator.element_areas is not None

    def test_aggregator_cached(self, simple_grid: AppGrid) -> None:
        """Test aggregator is cached after first access."""
        model = MagicMock()
        model.mesh = simple_grid

        api = ModelQueryAPI(model)
        agg1 = api.aggregator
        agg2 = api.aggregator

        assert agg1 is agg2

    def test_aggregator_no_mesh_raises(self) -> None:
        """Test aggregator access without mesh raises error."""
        model = MagicMock()
        model.mesh = None

        api = ModelQueryAPI(model)

        with pytest.raises(RuntimeError, match="no mesh"):
            _ = api.aggregator


class TestModelQueryAPIZones:
    """Tests for ModelQueryAPI zone handling."""

    @pytest.fixture
    def simple_grid(self) -> AppGrid:
        """Create a simple grid for testing."""
        nodes = {
            1: Node(id=1, x=0.0, y=0.0),
            2: Node(id=2, x=100.0, y=0.0),
            3: Node(id=3, x=100.0, y=100.0),
            4: Node(id=4, x=0.0, y=100.0),
            5: Node(id=5, x=200.0, y=0.0),
            6: Node(id=6, x=200.0, y=100.0),
        }
        elements = {
            1: Element(id=1, vertices=(1, 2, 3, 4), subregion=1, area=10000.0),
            2: Element(id=2, vertices=(2, 5, 6, 3), subregion=2, area=10000.0),
        }
        subregions = {
            1: Subregion(id=1, name="Region A"),
            2: Subregion(id=2, name="Region B"),
        }
        return AppGrid(nodes=nodes, elements=elements, subregions=subregions)

    def test_subregion_zones(self, simple_grid: AppGrid) -> None:
        """Test getting subregion zones."""
        model = MagicMock()
        model.mesh = simple_grid

        api = ModelQueryAPI(model)
        zone_def = api.subregion_zones

        assert zone_def is not None
        assert zone_def.n_zones == 2
        assert zone_def.get_zone(1).name == "Region A"
        assert zone_def.get_zone(2).name == "Region B"

    def test_subregion_zones_cached(self, simple_grid: AppGrid) -> None:
        """Test subregion zones are cached."""
        model = MagicMock()
        model.mesh = simple_grid

        api = ModelQueryAPI(model)
        zones1 = api.subregion_zones
        zones2 = api.subregion_zones

        assert zones1 is zones2

    def test_subregion_zones_no_mesh_raises(self) -> None:
        """Test subregion zones without mesh raises error."""
        model = MagicMock()
        model.mesh = None

        api = ModelQueryAPI(model)

        with pytest.raises(RuntimeError, match="no mesh"):
            _ = api.subregion_zones

    def test_register_zones(self, simple_grid: AppGrid) -> None:
        """Test registering custom zone definition."""
        model = MagicMock()
        model.mesh = simple_grid

        api = ModelQueryAPI(model)

        custom_zones = ZoneDefinition(
            zones={1: Zone(id=1, name="Custom Zone", elements=[1, 2])},
            element_zones=np.array([1, 1], dtype=np.int32),
        )
        api.register_zones("custom", custom_zones)

        assert "custom" in api._zone_definitions

    def test_get_zone_definition_element(self, simple_grid: AppGrid) -> None:
        """Test get_zone_definition returns None for element scale."""
        model = MagicMock()
        model.mesh = simple_grid

        api = ModelQueryAPI(model)

        assert api.get_zone_definition("element") is None

    def test_get_zone_definition_subregion(self, simple_grid: AppGrid) -> None:
        """Test get_zone_definition returns subregion zones."""
        model = MagicMock()
        model.mesh = simple_grid

        api = ModelQueryAPI(model)
        zone_def = api.get_zone_definition("subregion")

        assert zone_def is not None
        assert zone_def.n_zones == 2

    def test_get_zone_definition_custom(self, simple_grid: AppGrid) -> None:
        """Test get_zone_definition returns custom zones."""
        model = MagicMock()
        model.mesh = simple_grid

        api = ModelQueryAPI(model)
        custom_zones = ZoneDefinition(
            zones={1: Zone(id=1, name="Custom", elements=[1, 2])}
        )
        api.register_zones("my_zones", custom_zones)

        result = api.get_zone_definition("my_zones")
        assert result is custom_zones

    def test_get_zone_definition_unknown(self, simple_grid: AppGrid) -> None:
        """Test get_zone_definition returns None for unknown scale."""
        model = MagicMock()
        model.mesh = simple_grid

        api = ModelQueryAPI(model)

        assert api.get_zone_definition("unknown") is None


class TestModelQueryAPIGetValues:
    """Tests for ModelQueryAPI.get_values method."""

    @pytest.fixture
    def simple_grid(self) -> AppGrid:
        """Create a simple grid for testing."""
        nodes = {
            1: Node(id=1, x=0.0, y=0.0),
            2: Node(id=2, x=100.0, y=0.0),
            3: Node(id=3, x=100.0, y=100.0),
            4: Node(id=4, x=0.0, y=100.0),
            5: Node(id=5, x=200.0, y=0.0),
            6: Node(id=6, x=200.0, y=100.0),
        }
        elements = {
            1: Element(id=1, vertices=(1, 2, 3, 4), subregion=1, area=10000.0),
            2: Element(id=2, vertices=(2, 5, 6, 3), subregion=2, area=15000.0),
        }
        subregions = {
            1: Subregion(id=1, name="Region A"),
            2: Subregion(id=2, name="Region B"),
        }
        return AppGrid(nodes=nodes, elements=elements, subregions=subregions)

    def test_get_values_element_area(self, simple_grid: AppGrid) -> None:
        """Test getting element-level area values."""
        model = MagicMock()
        model.mesh = simple_grid

        api = ModelQueryAPI(model)
        values = api.get_values("area", scale="element")

        assert values[1] == pytest.approx(10000.0)
        assert values[2] == pytest.approx(15000.0)

    def test_get_values_element_subregion(self, simple_grid: AppGrid) -> None:
        """Test getting element-level subregion values."""
        model = MagicMock()
        model.mesh = simple_grid

        api = ModelQueryAPI(model)
        values = api.get_values("subregion", scale="element")

        assert values[1] == 1.0
        assert values[2] == 2.0

    def test_get_values_unknown_scale_raises(self, simple_grid: AppGrid) -> None:
        """Test getting values with unknown scale raises error."""
        model = MagicMock()
        model.mesh = simple_grid

        api = ModelQueryAPI(model)

        with pytest.raises(ValueError, match="Unknown scale"):
            api.get_values("area", scale="nonexistent")


class TestModelQueryAPIGetTimeseries:
    """Tests for ModelQueryAPI.get_timeseries method."""

    @pytest.fixture
    def mock_model_with_results(self) -> MagicMock:
        """Create mock model with time series results."""
        model = MagicMock()
        model.name = "Test Model"

        # Create simple grid
        nodes = {
            1: Node(id=1, x=0.0, y=0.0),
            2: Node(id=2, x=100.0, y=0.0),
            3: Node(id=3, x=100.0, y=100.0),
            4: Node(id=4, x=0.0, y=100.0),
        }
        elements = {
            1: Element(id=1, vertices=(1, 2, 3, 4), subregion=1, area=10000.0),
        }
        subregions = {1: Subregion(id=1, name="Region A")}
        model.mesh = AppGrid(nodes=nodes, elements=elements, subregions=subregions)

        model.stratigraphy = None
        model.groundwater = None

        # Mock results with time series
        model.results = MagicMock()
        model.results.times = [datetime(2020, i, 1) for i in range(1, 4)]
        model.results.head = np.array([
            [[10.0], [11.0], [12.0], [13.0]],  # Time 0
            [[11.0], [12.0], [13.0], [14.0]],  # Time 1
            [[12.0], [13.0], [14.0], [15.0]],  # Time 2
        ])

        return model

    def test_get_timeseries_no_results(self) -> None:
        """Test get_timeseries returns None when no results."""
        model = MagicMock()
        model.mesh = MagicMock()
        model.results = None

        api = ModelQueryAPI(model)
        ts = api.get_timeseries("head", location_id=1)

        assert ts is None

    def test_get_timeseries_no_times(self) -> None:
        """Test get_timeseries returns None when no times."""
        model = MagicMock()
        model.mesh = MagicMock()
        model.results = MagicMock()
        model.results.times = []

        api = ModelQueryAPI(model)
        ts = api.get_timeseries("head", location_id=1)

        assert ts is None


class TestModelQueryAPIExport:
    """Tests for ModelQueryAPI export methods."""

    @pytest.fixture
    def simple_grid(self) -> AppGrid:
        """Create a simple grid for testing."""
        nodes = {
            1: Node(id=1, x=0.0, y=0.0),
            2: Node(id=2, x=100.0, y=0.0),
            3: Node(id=3, x=100.0, y=100.0),
            4: Node(id=4, x=0.0, y=100.0),
        }
        elements = {
            1: Element(id=1, vertices=(1, 2, 3, 4), subregion=1, area=10000.0),
        }
        subregions = {1: Subregion(id=1, name="Region A")}
        return AppGrid(nodes=nodes, elements=elements, subregions=subregions)

    def test_export_to_dataframe_element(self, simple_grid: AppGrid) -> None:
        """Test exporting element-level data to DataFrame."""
        model = MagicMock()
        model.mesh = simple_grid
        model.stratigraphy = None
        model.groundwater = None
        model.results = None

        api = ModelQueryAPI(model)
        df = api.export_to_dataframe(["area"], scale="element")

        assert len(df) == 1
        assert "id" in df.columns
        assert "name" in df.columns
        assert "area" in df.columns
        assert df["id"].iloc[0] == 1
        assert df["area"].iloc[0] == pytest.approx(10000.0)

    def test_export_to_dataframe_zone(self, simple_grid: AppGrid) -> None:
        """Test exporting zone-level data to DataFrame."""
        model = MagicMock()
        model.mesh = simple_grid
        model.stratigraphy = None
        model.groundwater = None
        model.results = None

        api = ModelQueryAPI(model)
        df = api.export_to_dataframe(["area"], scale="subregion")

        assert len(df) == 1
        assert "id" in df.columns
        assert "name" in df.columns
        assert df["name"].iloc[0] == "Region A"

    def test_export_to_dataframe_unavailable_variable(self, simple_grid: AppGrid) -> None:
        """Test exporting unavailable variable returns NaN."""
        model = MagicMock()
        model.mesh = simple_grid
        model.stratigraphy = None
        model.groundwater = None
        model.results = None

        api = ModelQueryAPI(model)
        df = api.export_to_dataframe(["head"], scale="element")

        assert "head" in df.columns
        assert np.isnan(df["head"].iloc[0])

    def test_export_to_csv(self, simple_grid: AppGrid, tmp_path: Path) -> None:
        """Test exporting to CSV file."""
        model = MagicMock()
        model.mesh = simple_grid
        model.stratigraphy = None
        model.groundwater = None
        model.results = None

        api = ModelQueryAPI(model)
        output_file = tmp_path / "output.csv"

        api.export_to_csv(["area"], output_file, scale="element")

        assert output_file.exists()
        content = output_file.read_text()
        assert "id" in content
        assert "area" in content


class TestModelQueryAPIAvailable:
    """Tests for ModelQueryAPI available variables/scales methods."""

    @pytest.fixture
    def mock_model(self) -> MagicMock:
        """Create mock model with various components."""
        model = MagicMock()
        model.name = "Test Model"

        nodes = {
            1: Node(id=1, x=0.0, y=0.0),
            2: Node(id=2, x=100.0, y=0.0),
            3: Node(id=3, x=100.0, y=100.0),
            4: Node(id=4, x=0.0, y=100.0),
        }
        elements = {
            1: Element(id=1, vertices=(1, 2, 3, 4), subregion=1, area=10000.0),
        }
        subregions = {1: Subregion(id=1, name="Region A")}
        model.mesh = AppGrid(nodes=nodes, elements=elements, subregions=subregions)

        return model

    def test_get_available_variables_basic(self, mock_model: MagicMock) -> None:
        """Test available variables with basic mesh only."""
        mock_model.stratigraphy = None
        mock_model.groundwater = None
        mock_model.results = None

        api = ModelQueryAPI(mock_model)
        variables = api.get_available_variables()

        assert "area" in variables
        assert "subregion" in variables
        assert "head" not in variables

    def test_get_available_variables_with_stratigraphy(self, mock_model: MagicMock) -> None:
        """Test available variables with stratigraphy."""
        mock_model.stratigraphy = MagicMock()
        mock_model.groundwater = None
        mock_model.results = None

        api = ModelQueryAPI(mock_model)
        variables = api.get_available_variables()

        assert "thickness" in variables
        assert "top_elev" in variables
        assert "bottom_elev" in variables

    def test_get_available_variables_with_groundwater(self, mock_model: MagicMock) -> None:
        """Test available variables with groundwater."""
        mock_model.stratigraphy = None
        mock_model.groundwater = MagicMock()
        mock_model.results = None

        api = ModelQueryAPI(mock_model)
        variables = api.get_available_variables()

        assert "kh" in variables
        assert "kv" in variables
        assert "ss" in variables
        assert "sy" in variables

    def test_get_available_variables_with_results(self, mock_model: MagicMock) -> None:
        """Test available variables with results."""
        mock_model.stratigraphy = None
        mock_model.groundwater = None
        mock_model.results = MagicMock()

        api = ModelQueryAPI(mock_model)
        variables = api.get_available_variables()

        assert "head" in variables

    def test_get_available_scales_basic(self, mock_model: MagicMock) -> None:
        """Test available scales with basic mesh."""
        mock_model.stratigraphy = None
        mock_model.groundwater = None
        mock_model.results = None

        api = ModelQueryAPI(mock_model)
        scales = api.get_available_scales()

        assert "element" in scales
        assert "subregion" in scales

    def test_get_available_scales_with_custom(self, mock_model: MagicMock) -> None:
        """Test available scales includes registered custom zones."""
        mock_model.stratigraphy = None
        mock_model.groundwater = None
        mock_model.results = None

        api = ModelQueryAPI(mock_model)
        api.register_zones("my_zones", ZoneDefinition())

        scales = api.get_available_scales()

        assert "element" in scales
        assert "subregion" in scales
        assert "my_zones" in scales

    def test_get_available_scales_no_subregions(self) -> None:
        """Test available scales without subregions."""
        model = MagicMock()
        nodes = {
            1: Node(id=1, x=0.0, y=0.0),
            2: Node(id=2, x=100.0, y=0.0),
            3: Node(id=3, x=50.0, y=100.0),
        }
        elements = {1: Element(id=1, vertices=(1, 2, 3), subregion=0, area=5000.0)}
        model.mesh = AppGrid(nodes=nodes, elements=elements, subregions={})
        model.stratigraphy = None
        model.groundwater = None
        model.results = None

        api = ModelQueryAPI(model)
        scales = api.get_available_scales()

        assert "element" in scales
        assert "subregion" not in scales


class TestModelQueryAPIGetElementValues:
    """Tests for ModelQueryAPI._get_element_values internal method."""

    @pytest.fixture
    def mock_model_with_stratigraphy(self) -> MagicMock:
        """Create mock model with stratigraphy."""
        model = MagicMock()
        model.name = "Test Model"

        nodes = {
            1: Node(id=1, x=0.0, y=0.0),
            2: Node(id=2, x=100.0, y=0.0),
            3: Node(id=3, x=100.0, y=100.0),
            4: Node(id=4, x=0.0, y=100.0),
        }
        elements = {
            1: Element(id=1, vertices=(1, 2, 3, 4), subregion=1, area=10000.0),
        }
        subregions = {1: Subregion(id=1, name="Region A")}
        model.mesh = AppGrid(nodes=nodes, elements=elements, subregions=subregions)

        # Create stratigraphy with 2 layers
        model.stratigraphy = MagicMock()
        model.stratigraphy.thicknesses = np.array([
            [10.0, 20.0],  # Node 1
            [12.0, 22.0],  # Node 2
            [11.0, 21.0],  # Node 3
            [13.0, 23.0],  # Node 4
        ])
        model.stratigraphy.top_elev = np.array([
            [100.0, 90.0],
            [100.0, 88.0],
            [100.0, 89.0],
            [100.0, 87.0],
        ])
        model.stratigraphy.bottom_elev = np.array([
            [90.0, 70.0],
            [88.0, 66.0],
            [89.0, 68.0],
            [87.0, 64.0],
        ])

        model.groundwater = None
        model.results = None

        return model

    def test_get_element_values_no_mesh_raises(self) -> None:
        """Test _get_element_values raises error without mesh."""
        model = MagicMock()
        model.mesh = None

        api = ModelQueryAPI(model)

        with pytest.raises(RuntimeError, match="no mesh"):
            api._get_element_values("area")

    def test_get_element_values_area(self) -> None:
        """Test getting area values."""
        model = MagicMock()
        nodes = {
            1: Node(id=1, x=0.0, y=0.0),
            2: Node(id=2, x=100.0, y=0.0),
            3: Node(id=3, x=100.0, y=100.0),
            4: Node(id=4, x=0.0, y=100.0),
        }
        elements = {
            1: Element(id=1, vertices=(1, 2, 3, 4), subregion=1, area=10000.0),
        }
        model.mesh = AppGrid(nodes=nodes, elements=elements)

        api = ModelQueryAPI(model)
        values = api._get_element_values("area")

        assert values[0] == pytest.approx(10000.0)

    def test_get_element_values_subregion(self) -> None:
        """Test getting subregion values."""
        model = MagicMock()
        nodes = {
            1: Node(id=1, x=0.0, y=0.0),
            2: Node(id=2, x=100.0, y=0.0),
            3: Node(id=3, x=50.0, y=100.0),
        }
        elements = {
            1: Element(id=1, vertices=(1, 2, 3), subregion=5, area=5000.0),
        }
        model.mesh = AppGrid(nodes=nodes, elements=elements)

        api = ModelQueryAPI(model)
        values = api._get_element_values("subregion")

        assert values[0] == 5.0

    def test_get_element_values_thickness(self, mock_model_with_stratigraphy: MagicMock) -> None:
        """Test getting thickness values."""
        api = ModelQueryAPI(mock_model_with_stratigraphy)
        values = api._get_element_values("thickness", layer=1)

        # Average of [10, 12, 11, 13] = 11.5
        assert values[0] == pytest.approx(11.5)

    def test_get_element_values_top_elev(self, mock_model_with_stratigraphy: MagicMock) -> None:
        """Test getting top elevation values."""
        api = ModelQueryAPI(mock_model_with_stratigraphy)
        values = api._get_element_values("top_elev", layer=1)

        # All nodes have top_elev = 100.0 for layer 1
        assert values[0] == pytest.approx(100.0)

    def test_get_element_values_bottom_elev(self, mock_model_with_stratigraphy: MagicMock) -> None:
        """Test getting bottom elevation values."""
        api = ModelQueryAPI(mock_model_with_stratigraphy)
        values = api._get_element_values("bottom_elev", layer=1)

        # Average of [90, 88, 89, 87] = 88.5
        assert values[0] == pytest.approx(88.5)

    def test_get_element_values_no_stratigraphy(self) -> None:
        """Test getting stratigraphy values when no stratigraphy exists."""
        model = MagicMock()
        nodes = {
            1: Node(id=1, x=0.0, y=0.0),
            2: Node(id=2, x=100.0, y=0.0),
            3: Node(id=3, x=50.0, y=100.0),
        }
        elements = {1: Element(id=1, vertices=(1, 2, 3), subregion=1, area=5000.0)}
        model.mesh = AppGrid(nodes=nodes, elements=elements)
        model.stratigraphy = None

        api = ModelQueryAPI(model)
        values = api._get_element_values("thickness")

        # Should return NaN when no stratigraphy
        assert np.isnan(values[0])


class TestModelQueryAPIPropertyInfo:
    """Tests for ModelQueryAPI.PROPERTY_INFO."""

    def test_property_info_exists(self) -> None:
        """Test PROPERTY_INFO dictionary exists with expected keys."""
        info = ModelQueryAPI.PROPERTY_INFO

        assert "head" in info
        assert "kh" in info
        assert "kv" in info
        assert "ss" in info
        assert "sy" in info
        assert "thickness" in info
        assert "area" in info

    def test_property_info_structure(self) -> None:
        """Test PROPERTY_INFO entries have expected structure."""
        info = ModelQueryAPI.PROPERTY_INFO

        for prop_name, prop_info in info.items():
            assert "name" in prop_info
            assert "units" in prop_info
            assert "source" in prop_info


# =============================================================================
# Additional coverage tests
# =============================================================================


class TestModelQueryAPIGetValuesZoneAggregation:
    """Tests for zone-aggregated get_values."""

    @pytest.fixture
    def model_with_zones(self) -> MagicMock:
        """Create mock model with zones for aggregation testing."""
        model = MagicMock()
        model.name = "Test Model"

        nodes = {
            1: Node(id=1, x=0.0, y=0.0),
            2: Node(id=2, x=100.0, y=0.0),
            3: Node(id=3, x=100.0, y=100.0),
            4: Node(id=4, x=0.0, y=100.0),
            5: Node(id=5, x=200.0, y=0.0),
            6: Node(id=6, x=200.0, y=100.0),
        }
        elements = {
            1: Element(id=1, vertices=(1, 2, 3, 4), subregion=1, area=10000.0),
            2: Element(id=2, vertices=(2, 5, 6, 3), subregion=2, area=15000.0),
        }
        subregions = {
            1: Subregion(id=1, name="Region A"),
            2: Subregion(id=2, name="Region B"),
        }
        model.mesh = AppGrid(nodes=nodes, elements=elements, subregions=subregions)
        model.stratigraphy = None
        model.groundwater = None
        model.results = None
        return model

    def test_get_values_subregion_scale(self, model_with_zones: MagicMock) -> None:
        """Test getting values at subregion scale."""
        api = ModelQueryAPI(model_with_zones)
        values = api.get_values("area", scale="subregion")

        assert isinstance(values, dict)
        assert len(values) > 0

    def test_get_values_custom_zones(self, model_with_zones: MagicMock) -> None:
        """Test getting values with registered custom zones."""
        api = ModelQueryAPI(model_with_zones)

        custom_zones = ZoneDefinition(
            zones={1: Zone(id=1, name="All", elements=[1, 2])},
            element_zones=np.array([1, 1], dtype=np.int32),
        )
        api.register_zones("all", custom_zones)
        values = api.get_values("area", scale="all")

        assert isinstance(values, dict)


class TestModelQueryAPIGetTimeseriesWithData:
    """Tests for get_timeseries with actual data."""

    @pytest.fixture
    def mock_model_full(self) -> MagicMock:
        """Create mock model with full results."""
        model = MagicMock()
        model.name = "Test Model"

        nodes = {
            1: Node(id=1, x=0.0, y=0.0),
            2: Node(id=2, x=100.0, y=0.0),
            3: Node(id=3, x=100.0, y=100.0),
            4: Node(id=4, x=0.0, y=100.0),
        }
        elements = {
            1: Element(id=1, vertices=(1, 2, 3, 4), subregion=1, area=10000.0),
        }
        subregions = {1: Subregion(id=1, name="Region A")}
        model.mesh = AppGrid(nodes=nodes, elements=elements, subregions=subregions)

        model.stratigraphy = None
        model.groundwater = None

        model.results = MagicMock()
        model.results.times = [datetime(2020, i, 1) for i in range(1, 4)]
        model.results.head = np.array([
            [[10.0], [11.0], [12.0], [13.0]],
            [[11.0], [12.0], [13.0], [14.0]],
            [[12.0], [13.0], [14.0], [15.0]],
        ])

        return model

    def test_get_timeseries_returns_data(self, mock_model_full: MagicMock) -> None:
        """Test get_timeseries returns valid TimeSeries."""
        api = ModelQueryAPI(mock_model_full)
        ts = api.get_timeseries("area", location_id=1)

        assert ts is not None
        assert ts.n_timesteps == 3
        assert ts.variable == "area"
        assert ts.location_id == 1

    def test_get_timeseries_missing_location(self, mock_model_full: MagicMock) -> None:
        """Test get_timeseries with non-existent location returns NaN values."""
        api = ModelQueryAPI(mock_model_full)
        ts = api.get_timeseries("area", location_id=999)

        assert ts is not None
        assert all(np.isnan(v) for v in ts.values)


class TestModelQueryAPIExportTimeseriesCsv:
    """Tests for export_timeseries_to_csv."""

    def test_export_timeseries_csv(self, tmp_path: Path) -> None:
        """Test exporting time series to CSV."""
        model = MagicMock()
        model.name = "Test"

        nodes = {
            1: Node(id=1, x=0.0, y=0.0),
            2: Node(id=2, x=100.0, y=0.0),
            3: Node(id=3, x=50.0, y=100.0),
        }
        elements = {
            1: Element(id=1, vertices=(1, 2, 3), subregion=1, area=5000.0),
        }
        model.mesh = AppGrid(nodes=nodes, elements=elements)
        model.stratigraphy = None
        model.groundwater = None

        model.results = MagicMock()
        model.results.times = [datetime(2020, 1, 1), datetime(2020, 2, 1)]
        model.results.head = None

        api = ModelQueryAPI(model)
        output_file = tmp_path / "timeseries.csv"

        api.export_timeseries_to_csv("area", [1], output_file)

        assert output_file.exists()
        content = output_file.read_text()
        assert "time" in content

    def test_export_timeseries_csv_no_results(self, tmp_path: Path) -> None:
        """Test export when no results available."""
        model = MagicMock()
        model.name = "Test"
        nodes = {1: Node(id=1, x=0.0, y=0.0), 2: Node(id=2, x=100.0, y=0.0), 3: Node(id=3, x=50.0, y=100.0)}
        elements = {1: Element(id=1, vertices=(1, 2, 3), subregion=1, area=5000.0)}
        model.mesh = AppGrid(nodes=nodes, elements=elements)
        model.results = None

        api = ModelQueryAPI(model)
        output_file = tmp_path / "empty.csv"

        api.export_timeseries_to_csv("area", [1], output_file)
        assert output_file.exists()


class TestModelQueryAPIGetElementValuesGW:
    """Tests for _get_element_values with groundwater parameters."""

    def test_get_element_values_kh(self) -> None:
        """Test getting Kh values from groundwater params."""
        model = MagicMock()
        nodes = {
            1: Node(id=1, x=0.0, y=0.0),
            2: Node(id=2, x=100.0, y=0.0),
            3: Node(id=3, x=100.0, y=100.0),
            4: Node(id=4, x=0.0, y=100.0),
        }
        elements = {
            1: Element(id=1, vertices=(1, 2, 3, 4), subregion=1, area=10000.0),
        }
        model.mesh = AppGrid(nodes=nodes, elements=elements)
        model.stratigraphy = None

        # Mock groundwater with aquifer params
        model.groundwater = MagicMock()
        model.groundwater.aquifer_params = {
            "kh": np.array([[10.0, 5.0], [12.0, 6.0], [11.0, 5.5], [13.0, 6.5]]),
        }
        model.results = None

        api = ModelQueryAPI(model)
        values = api._get_element_values("kh", layer=1)

        # Average of [10, 12, 11, 13] = 11.5
        assert values[0] == pytest.approx(11.5)

    def test_get_element_values_kh_1d(self) -> None:
        """Test getting Kh values when params are 1D."""
        model = MagicMock()
        nodes = {
            1: Node(id=1, x=0.0, y=0.0),
            2: Node(id=2, x=100.0, y=0.0),
            3: Node(id=3, x=50.0, y=100.0),
        }
        elements = {
            1: Element(id=1, vertices=(1, 2, 3), subregion=1, area=5000.0),
        }
        model.mesh = AppGrid(nodes=nodes, elements=elements)
        model.stratigraphy = None
        model.groundwater = MagicMock()
        model.groundwater.aquifer_params = {
            "kh": np.array([10.0, 20.0, 30.0]),
        }
        model.results = None

        api = ModelQueryAPI(model)
        values = api._get_element_values("kh")

        assert values[0] == pytest.approx(20.0)  # mean of [10, 20, 30]

    def test_get_element_values_no_gw(self) -> None:
        """Test getting GW values when no groundwater component."""
        model = MagicMock()
        nodes = {1: Node(id=1, x=0.0, y=0.0), 2: Node(id=2, x=100.0, y=0.0), 3: Node(id=3, x=50.0, y=100.0)}
        elements = {1: Element(id=1, vertices=(1, 2, 3), subregion=1, area=5000.0)}
        model.mesh = AppGrid(nodes=nodes, elements=elements)
        model.stratigraphy = None
        model.groundwater = None
        model.results = None

        api = ModelQueryAPI(model)
        values = api._get_element_values("kh")

        assert np.isnan(values[0])

    def test_get_element_values_head_3d(self) -> None:
        """Test getting head values from 3D results array."""
        model = MagicMock()
        nodes = {
            1: Node(id=1, x=0.0, y=0.0),
            2: Node(id=2, x=100.0, y=0.0),
            3: Node(id=3, x=100.0, y=100.0),
            4: Node(id=4, x=0.0, y=100.0),
        }
        elements = {
            1: Element(id=1, vertices=(1, 2, 3, 4), subregion=1, area=10000.0),
        }
        model.mesh = AppGrid(nodes=nodes, elements=elements)
        model.stratigraphy = None
        model.groundwater = None

        model.results = MagicMock()
        model.results.head = np.array([
            [[100.0, 90.0], [102.0, 91.0], [101.0, 89.0], [103.0, 92.0]],
        ])

        api = ModelQueryAPI(model)
        values = api._get_element_values("head", layer=1, time_index=0)

        # Average of [100, 102, 101, 103] = 101.5
        assert values[0] == pytest.approx(101.5)

    def test_get_element_values_head_2d(self) -> None:
        """Test getting head values from 2D results array."""
        model = MagicMock()
        nodes = {
            1: Node(id=1, x=0.0, y=0.0),
            2: Node(id=2, x=100.0, y=0.0),
            3: Node(id=3, x=50.0, y=100.0),
        }
        elements = {
            1: Element(id=1, vertices=(1, 2, 3), subregion=1, area=5000.0),
        }
        model.mesh = AppGrid(nodes=nodes, elements=elements)
        model.stratigraphy = None
        model.groundwater = None

        model.results = MagicMock()
        model.results.head = np.array([
            [100.0, 102.0, 101.0],
        ])

        api = ModelQueryAPI(model)
        values = api._get_element_values("head", time_index=0)

        assert values[0] == pytest.approx(101.0)

    def test_get_element_values_head_no_results(self) -> None:
        """Test getting head values when no head results."""
        model = MagicMock()
        nodes = {1: Node(id=1, x=0.0, y=0.0), 2: Node(id=2, x=100.0, y=0.0), 3: Node(id=3, x=50.0, y=100.0)}
        elements = {1: Element(id=1, vertices=(1, 2, 3), subregion=1, area=5000.0)}
        model.mesh = AppGrid(nodes=nodes, elements=elements)
        model.stratigraphy = None
        model.groundwater = None
        model.results = MagicMock()
        model.results.head = None

        api = ModelQueryAPI(model)
        values = api._get_element_values("head")

        assert np.isnan(values[0])


class TestModelQueryAPIExportMultipleVars:
    """Tests for exporting multiple variables."""

    @pytest.fixture
    def full_model(self) -> MagicMock:
        """Model with stratigraphy for multi-variable export."""
        model = MagicMock()
        model.name = "Test"
        nodes = {
            1: Node(id=1, x=0.0, y=0.0),
            2: Node(id=2, x=100.0, y=0.0),
            3: Node(id=3, x=100.0, y=100.0),
            4: Node(id=4, x=0.0, y=100.0),
        }
        elements = {
            1: Element(id=1, vertices=(1, 2, 3, 4), subregion=1, area=10000.0),
        }
        subregions = {1: Subregion(id=1, name="Region A")}
        model.mesh = AppGrid(nodes=nodes, elements=elements, subregions=subregions)

        model.stratigraphy = MagicMock()
        model.stratigraphy.thicknesses = np.array([
            [50.0, 30.0], [50.0, 30.0], [50.0, 30.0], [50.0, 30.0],
        ])
        model.stratigraphy.top_elev = np.array([
            [100.0, 50.0], [100.0, 50.0], [100.0, 50.0], [100.0, 50.0],
        ])
        model.stratigraphy.bottom_elev = np.array([
            [50.0, 20.0], [50.0, 20.0], [50.0, 20.0], [50.0, 20.0],
        ])

        model.groundwater = None
        model.results = None
        return model

    def test_export_multiple_variables(self, full_model: MagicMock) -> None:
        """Test exporting multiple variables to DataFrame."""
        api = ModelQueryAPI(full_model)
        df = api.export_to_dataframe(["area", "thickness"], scale="element", layer=1)

        assert "area" in df.columns
        assert "thickness" in df.columns
        assert len(df) == 1

    def test_export_to_csv_multiple(self, full_model: MagicMock, tmp_path: Path) -> None:
        """Test exporting multiple variables to CSV."""
        api = ModelQueryAPI(full_model)
        output = tmp_path / "multi.csv"
        api.export_to_csv(["area", "thickness"], output, scale="element", layer=1)

        assert output.exists()
        content = output.read_text()
        assert "area" in content
        assert "thickness" in content
