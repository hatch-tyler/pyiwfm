"""Unit tests for data aggregation functionality.

Tests:
- AggregationMethod enum
- DataAggregator class
- create_aggregator_from_grid function
"""

from __future__ import annotations

import numpy as np
import pytest

from pyiwfm.core.aggregation import (
    AggregationMethod,
    DataAggregator,
    create_aggregator_from_grid,
)
from pyiwfm.core.mesh import AppGrid, Element, Node
from pyiwfm.core.zones import Zone, ZoneDefinition

# =============================================================================
# Test AggregationMethod Enum
# =============================================================================


class TestAggregationMethod:
    """Tests for AggregationMethod enum."""

    def test_sum_value(self) -> None:
        """Test SUM enum value."""
        assert AggregationMethod.SUM.value == "sum"

    def test_mean_value(self) -> None:
        """Test MEAN enum value."""
        assert AggregationMethod.MEAN.value == "mean"

    def test_area_weighted_mean_value(self) -> None:
        """Test AREA_WEIGHTED_MEAN enum value."""
        assert AggregationMethod.AREA_WEIGHTED_MEAN.value == "area_weighted_mean"

    def test_min_value(self) -> None:
        """Test MIN enum value."""
        assert AggregationMethod.MIN.value == "min"

    def test_max_value(self) -> None:
        """Test MAX enum value."""
        assert AggregationMethod.MAX.value == "max"

    def test_median_value(self) -> None:
        """Test MEDIAN enum value."""
        assert AggregationMethod.MEDIAN.value == "median"


# =============================================================================
# Test DataAggregator
# =============================================================================


class TestDataAggregator:
    """Tests for DataAggregator class."""

    @pytest.fixture
    def element_areas(self) -> np.ndarray:
        """Create sample element areas."""
        return np.array([100.0, 150.0, 200.0, 125.0, 175.0, 250.0], dtype=np.float64)

    @pytest.fixture
    def element_values(self) -> np.ndarray:
        """Create sample element values."""
        return np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=np.float64)

    @pytest.fixture
    def zone_def(self) -> ZoneDefinition:
        """Create sample zone definition."""
        zones = {
            1: Zone(id=1, name="Zone A", elements=[1, 2, 3]),
            2: Zone(id=2, name="Zone B", elements=[4, 5, 6]),
        }
        element_zones = np.array([1, 1, 1, 2, 2, 2], dtype=np.int32)
        return ZoneDefinition(zones=zones, element_zones=element_zones)

    def test_initialization_with_areas(self, element_areas: np.ndarray) -> None:
        """Test initialization with element areas."""
        aggregator = DataAggregator(element_areas=element_areas)

        assert aggregator.element_areas is not None
        assert len(aggregator.element_areas) == 6

    def test_initialization_without_areas(self) -> None:
        """Test initialization without element areas."""
        aggregator = DataAggregator()

        assert aggregator.element_areas is None

    def test_available_methods(self) -> None:
        """Test available_methods property."""
        aggregator = DataAggregator()

        methods = aggregator.available_methods
        assert "sum" in methods
        assert "mean" in methods
        assert "area_weighted_mean" in methods
        assert "min" in methods
        assert "max" in methods
        assert "median" in methods

    def test_set_element_areas(self, element_areas: np.ndarray) -> None:
        """Test set_element_areas method."""
        aggregator = DataAggregator()
        assert aggregator.element_areas is None

        aggregator.set_element_areas(element_areas)
        assert aggregator.element_areas is not None
        assert len(aggregator.element_areas) == 6

    def test_repr_without_areas(self) -> None:
        """Test string representation without areas."""
        aggregator = DataAggregator()

        result = repr(aggregator)
        assert "DataAggregator" in result
        assert "n_elements=0" in result
        assert "has_areas=False" in result

    def test_repr_with_areas(self, element_areas: np.ndarray) -> None:
        """Test string representation with areas."""
        aggregator = DataAggregator(element_areas=element_areas)

        result = repr(aggregator)
        assert "n_elements=6" in result
        assert "has_areas=True" in result


# =============================================================================
# Test Aggregation Methods
# =============================================================================


class TestAggregationMethods:
    """Tests for individual aggregation methods."""

    @pytest.fixture
    def aggregator(self) -> DataAggregator:
        """Create aggregator with areas."""
        areas = np.array([100.0, 200.0, 300.0, 150.0, 250.0], dtype=np.float64)
        return DataAggregator(element_areas=areas)

    @pytest.fixture
    def values(self) -> np.ndarray:
        """Create sample values."""
        return np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64)

    @pytest.fixture
    def zone_def(self) -> ZoneDefinition:
        """Create zone definition."""
        zones = {
            1: Zone(id=1, name="Zone A", elements=[1, 2, 3]),
            2: Zone(id=2, name="Zone B", elements=[4, 5]),
        }
        element_zones = np.array([1, 1, 1, 2, 2], dtype=np.int32)
        return ZoneDefinition(zones=zones, element_zones=element_zones)

    def test_aggregate_sum(
        self, aggregator: DataAggregator, values: np.ndarray, zone_def: ZoneDefinition
    ) -> None:
        """Test sum aggregation."""
        result = aggregator.aggregate(values, zone_def, method="sum")

        assert result[1] == pytest.approx(60.0)  # 10 + 20 + 30
        assert result[2] == pytest.approx(90.0)  # 40 + 50

    def test_aggregate_mean(
        self, aggregator: DataAggregator, values: np.ndarray, zone_def: ZoneDefinition
    ) -> None:
        """Test mean aggregation."""
        result = aggregator.aggregate(values, zone_def, method="mean")

        assert result[1] == pytest.approx(20.0)  # (10 + 20 + 30) / 3
        assert result[2] == pytest.approx(45.0)  # (40 + 50) / 2

    def test_aggregate_area_weighted_mean(
        self, aggregator: DataAggregator, values: np.ndarray, zone_def: ZoneDefinition
    ) -> None:
        """Test area-weighted mean aggregation."""
        result = aggregator.aggregate(values, zone_def, method="area_weighted_mean")

        # Zone 1: (10*100 + 20*200 + 30*300) / (100 + 200 + 300) = 14000/600 = 23.33
        assert result[1] == pytest.approx(14000.0 / 600.0)
        # Zone 2: (40*150 + 50*250) / (150 + 250) = 18500/400 = 46.25
        assert result[2] == pytest.approx(18500.0 / 400.0)

    def test_aggregate_min(
        self, aggregator: DataAggregator, values: np.ndarray, zone_def: ZoneDefinition
    ) -> None:
        """Test min aggregation."""
        result = aggregator.aggregate(values, zone_def, method="min")

        assert result[1] == pytest.approx(10.0)
        assert result[2] == pytest.approx(40.0)

    def test_aggregate_max(
        self, aggregator: DataAggregator, values: np.ndarray, zone_def: ZoneDefinition
    ) -> None:
        """Test max aggregation."""
        result = aggregator.aggregate(values, zone_def, method="max")

        assert result[1] == pytest.approx(30.0)
        assert result[2] == pytest.approx(50.0)

    def test_aggregate_median(
        self, aggregator: DataAggregator, values: np.ndarray, zone_def: ZoneDefinition
    ) -> None:
        """Test median aggregation."""
        result = aggregator.aggregate(values, zone_def, method="median")

        assert result[1] == pytest.approx(20.0)  # median of [10, 20, 30]
        assert result[2] == pytest.approx(45.0)  # median of [40, 50]

    def test_aggregate_unknown_method(
        self, aggregator: DataAggregator, values: np.ndarray, zone_def: ZoneDefinition
    ) -> None:
        """Test error on unknown aggregation method."""
        with pytest.raises(ValueError, match="Unknown aggregation method"):
            aggregator.aggregate(values, zone_def, method="invalid")

    def test_aggregate_area_weighted_without_areas(
        self, values: np.ndarray, zone_def: ZoneDefinition
    ) -> None:
        """Test error when using area_weighted_mean without areas."""
        aggregator = DataAggregator()  # No areas

        with pytest.raises(ValueError, match="element_areas required"):
            aggregator.aggregate(values, zone_def, method="area_weighted_mean")


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestAggregationEdgeCases:
    """Tests for edge cases in aggregation."""

    @pytest.fixture
    def aggregator(self) -> DataAggregator:
        """Create aggregator with areas."""
        areas = np.array([100.0, 200.0, 300.0, 150.0, 250.0], dtype=np.float64)
        return DataAggregator(element_areas=areas)

    def test_empty_zone(self, aggregator: DataAggregator) -> None:
        """Test aggregation with empty zone."""
        zones = {
            1: Zone(id=1, name="Empty", elements=[]),
        }
        zone_def = ZoneDefinition(zones=zones)
        values = np.array([10.0, 20.0, 30.0], dtype=np.float64)

        result = aggregator.aggregate(values, zone_def, method="mean")

        assert np.isnan(result[1])

    def test_zone_with_invalid_elements(self, aggregator: DataAggregator) -> None:
        """Test aggregation when zone has elements beyond array bounds."""
        zones = {
            1: Zone(id=1, name="OutOfBounds", elements=[100, 200]),
        }
        zone_def = ZoneDefinition(zones=zones)
        values = np.array([10.0, 20.0, 30.0], dtype=np.float64)

        result = aggregator.aggregate(values, zone_def, method="mean")

        assert np.isnan(result[1])

    def test_values_with_nan(self, aggregator: DataAggregator) -> None:
        """Test aggregation with NaN values."""
        zones = {
            1: Zone(id=1, name="WithNaN", elements=[1, 2, 3]),
        }
        element_zones = np.array([1, 1, 1], dtype=np.int32)
        zone_def = ZoneDefinition(zones=zones, element_zones=element_zones)
        values = np.array([10.0, np.nan, 30.0], dtype=np.float64)

        result = aggregator.aggregate(values, zone_def, method="mean")

        assert result[1] == pytest.approx(20.0)  # (10 + 30) / 2, ignoring NaN

    def test_all_nan_values(self, aggregator: DataAggregator) -> None:
        """Test aggregation when all values are NaN."""
        zones = {
            1: Zone(id=1, name="AllNaN", elements=[1, 2, 3]),
        }
        element_zones = np.array([1, 1, 1], dtype=np.int32)
        zone_def = ZoneDefinition(zones=zones, element_zones=element_zones)
        values = np.array([np.nan, np.nan, np.nan], dtype=np.float64)

        result = aggregator.aggregate(values, zone_def, method="area_weighted_mean")

        assert np.isnan(result[1])

    def test_zero_total_area(self) -> None:
        """Test area-weighted mean when total area is zero."""
        areas = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        aggregator = DataAggregator(element_areas=areas)

        zones = {1: Zone(id=1, name="ZeroArea", elements=[1, 2, 3])}
        element_zones = np.array([1, 1, 1], dtype=np.int32)
        zone_def = ZoneDefinition(zones=zones, element_zones=element_zones)
        values = np.array([10.0, 20.0, 30.0], dtype=np.float64)

        result = aggregator.aggregate(values, zone_def, method="area_weighted_mean")

        # Should fall back to simple mean
        assert result[1] == pytest.approx(20.0)


# =============================================================================
# Test aggregate_to_array
# =============================================================================


class TestAggregateToArray:
    """Tests for aggregate_to_array method."""

    @pytest.fixture
    def aggregator(self) -> DataAggregator:
        """Create aggregator with areas."""
        areas = np.array([100.0, 100.0, 100.0, 100.0, 100.0], dtype=np.float64)
        return DataAggregator(element_areas=areas)

    @pytest.fixture
    def zone_def(self) -> ZoneDefinition:
        """Create zone definition."""
        zones = {
            1: Zone(id=1, name="Zone A", elements=[1, 2]),
            2: Zone(id=2, name="Zone B", elements=[3, 4, 5]),
        }
        element_zones = np.array([1, 1, 2, 2, 2], dtype=np.int32)
        return ZoneDefinition(zones=zones, element_zones=element_zones)

    def test_aggregate_to_array(self, aggregator: DataAggregator, zone_def: ZoneDefinition) -> None:
        """Test expanding zone values back to element array."""
        values = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64)

        result = aggregator.aggregate_to_array(values, zone_def, method="mean")

        # Zone 1 mean = 15, Zone 2 mean = 40
        assert result[0] == pytest.approx(15.0)
        assert result[1] == pytest.approx(15.0)
        assert result[2] == pytest.approx(40.0)
        assert result[3] == pytest.approx(40.0)
        assert result[4] == pytest.approx(40.0)

    def test_aggregate_to_array_unassigned_elements(self, aggregator: DataAggregator) -> None:
        """Test aggregate_to_array with unassigned elements."""
        zones = {1: Zone(id=1, name="Zone A", elements=[1, 2])}
        element_zones = np.array([1, 1, 0, 0, 0], dtype=np.int32)  # Last 3 unassigned
        zone_def = ZoneDefinition(zones=zones, element_zones=element_zones)
        values = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64)

        result = aggregator.aggregate_to_array(values, zone_def, method="mean")

        assert result[0] == pytest.approx(15.0)
        assert result[1] == pytest.approx(15.0)
        assert np.isnan(result[2])
        assert np.isnan(result[3])
        assert np.isnan(result[4])


# =============================================================================
# Test aggregate_timeseries
# =============================================================================


class TestAggregateTimeseries:
    """Tests for aggregate_timeseries method."""

    @pytest.fixture
    def aggregator(self) -> DataAggregator:
        """Create aggregator."""
        areas = np.array([100.0, 100.0, 100.0], dtype=np.float64)
        return DataAggregator(element_areas=areas)

    @pytest.fixture
    def zone_def(self) -> ZoneDefinition:
        """Create zone definition."""
        zones = {
            1: Zone(id=1, name="Zone A", elements=[1, 2]),
            2: Zone(id=2, name="Zone B", elements=[3]),
        }
        element_zones = np.array([1, 1, 2], dtype=np.int32)
        return ZoneDefinition(zones=zones, element_zones=element_zones)

    def test_aggregate_timeseries(
        self, aggregator: DataAggregator, zone_def: ZoneDefinition
    ) -> None:
        """Test time series aggregation."""
        timeseries = [
            np.array([10.0, 20.0, 100.0], dtype=np.float64),  # t=0
            np.array([11.0, 21.0, 110.0], dtype=np.float64),  # t=1
            np.array([12.0, 22.0, 120.0], dtype=np.float64),  # t=2
        ]

        result = aggregator.aggregate_timeseries(timeseries, zone_def, method="mean")

        # Zone 1 means over time
        assert result[1][0] == pytest.approx(15.0)  # (10+20)/2
        assert result[1][1] == pytest.approx(16.0)  # (11+21)/2
        assert result[1][2] == pytest.approx(17.0)  # (12+22)/2

        # Zone 2 means over time
        assert result[2][0] == pytest.approx(100.0)
        assert result[2][1] == pytest.approx(110.0)
        assert result[2][2] == pytest.approx(120.0)

    def test_aggregate_empty_timeseries(
        self, aggregator: DataAggregator, zone_def: ZoneDefinition
    ) -> None:
        """Test with empty time series."""
        result = aggregator.aggregate_timeseries([], zone_def, method="mean")

        assert result[1] == []
        assert result[2] == []


# =============================================================================
# Test create_aggregator_from_grid
# =============================================================================


class TestCreateAggregatorFromGrid:
    """Tests for create_aggregator_from_grid function."""

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
            1: Element(id=1, vertices=(1, 2, 3, 4), area=10000.0),
            2: Element(id=2, vertices=(2, 5, 6, 3), area=10000.0),
        }
        return AppGrid(nodes=nodes, elements=elements)

    def test_create_aggregator(self, simple_grid: AppGrid) -> None:
        """Test creating aggregator from grid."""
        aggregator = create_aggregator_from_grid(simple_grid)

        assert aggregator.element_areas is not None
        assert len(aggregator.element_areas) == 2
        assert aggregator.element_areas[0] == pytest.approx(10000.0)
        assert aggregator.element_areas[1] == pytest.approx(10000.0)

    def test_create_aggregator_empty_grid(self) -> None:
        """Test creating aggregator from empty grid."""
        grid = AppGrid(nodes={}, elements={})

        aggregator = create_aggregator_from_grid(grid)

        assert aggregator.element_areas is not None
        assert len(aggregator.element_areas) == 0
