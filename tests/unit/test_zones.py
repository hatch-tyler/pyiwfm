"""Unit tests for zone functionality.

Tests:
- Zone dataclass
- ZoneDefinition class
- Zone file I/O
"""

from __future__ import annotations

import numpy as np
import pytest

from pyiwfm.core.mesh import AppGrid, Element, Node, Subregion
from pyiwfm.core.zones import Zone, ZoneDefinition

# =============================================================================
# Test Zone Dataclass
# =============================================================================


class TestZone:
    """Tests for Zone dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic zone creation."""
        zone = Zone(id=1, name="Test Zone", elements=[1, 2, 3], area=1000.0)

        assert zone.id == 1
        assert zone.name == "Test Zone"
        assert zone.elements == [1, 2, 3]
        assert zone.area == 1000.0

    def test_default_values(self) -> None:
        """Test default values."""
        zone = Zone(id=1, name="Empty Zone")

        assert zone.elements == []
        assert zone.area == 0.0

    def test_n_elements_property(self) -> None:
        """Test n_elements property."""
        zone = Zone(id=1, name="Test", elements=[1, 2, 3, 4, 5])

        assert zone.n_elements == 5

    def test_empty_zone_n_elements(self) -> None:
        """Test n_elements for empty zone."""
        zone = Zone(id=1, name="Empty")

        assert zone.n_elements == 0

    def test_repr(self) -> None:
        """Test string representation."""
        zone = Zone(id=1, name="Test Zone", elements=[1, 2, 3], area=1500.5)

        result = repr(zone)
        assert "Zone(id=1" in result
        assert "name='Test Zone'" in result
        assert "n_elements=3" in result
        assert "area=1500.5" in result


# =============================================================================
# Test ZoneDefinition
# =============================================================================


class TestZoneDefinition:
    """Tests for ZoneDefinition class."""

    @pytest.fixture
    def sample_zones(self) -> dict[int, Zone]:
        """Create sample zones for testing."""
        return {
            1: Zone(id=1, name="North", elements=[1, 2, 3], area=3000.0),
            2: Zone(id=2, name="South", elements=[4, 5, 6], area=2500.0),
        }

    @pytest.fixture
    def sample_element_zones(self) -> np.ndarray:
        """Create sample element-to-zone mapping."""
        return np.array([1, 1, 1, 2, 2, 2], dtype=np.int32)

    def test_basic_creation(
        self, sample_zones: dict[int, Zone], sample_element_zones: np.ndarray
    ) -> None:
        """Test basic zone definition creation."""
        zone_def = ZoneDefinition(
            zones=sample_zones,
            extent="horizontal",
            element_zones=sample_element_zones,
        )

        assert zone_def.n_zones == 2
        assert zone_def.extent == "horizontal"

    def test_default_values(self) -> None:
        """Test default values."""
        zone_def = ZoneDefinition()

        assert zone_def.zones == {}
        assert zone_def.extent == "horizontal"
        assert zone_def.element_zones is None
        assert zone_def.name == ""
        assert zone_def.description == ""

    def test_invalid_extent(self) -> None:
        """Test invalid extent raises error."""
        with pytest.raises(ValueError, match="extent must be"):
            ZoneDefinition(extent="invalid")

    def test_n_zones_property(self, sample_zones: dict[int, Zone]) -> None:
        """Test n_zones property."""
        zone_def = ZoneDefinition(zones=sample_zones)

        assert zone_def.n_zones == 2

    def test_n_elements_property(
        self, sample_zones: dict[int, Zone], sample_element_zones: np.ndarray
    ) -> None:
        """Test n_elements property."""
        zone_def = ZoneDefinition(zones=sample_zones, element_zones=sample_element_zones)

        assert zone_def.n_elements == 6  # All elements have non-zero zone

    def test_n_elements_with_unassigned(self) -> None:
        """Test n_elements with unassigned elements."""
        element_zones = np.array([1, 0, 1, 0, 2, 2], dtype=np.int32)
        zone_def = ZoneDefinition(element_zones=element_zones)

        assert zone_def.n_elements == 4  # Only non-zero elements

    def test_zone_ids_property(self, sample_zones: dict[int, Zone]) -> None:
        """Test zone_ids property returns sorted list."""
        # Add zones in non-sorted order
        zones = {3: Zone(id=3, name="C"), 1: Zone(id=1, name="A"), 2: Zone(id=2, name="B")}
        zone_def = ZoneDefinition(zones=zones)

        assert zone_def.zone_ids == [1, 2, 3]

    def test_get_zone_for_element(
        self, sample_zones: dict[int, Zone], sample_element_zones: np.ndarray
    ) -> None:
        """Test get_zone_for_element method."""
        zone_def = ZoneDefinition(zones=sample_zones, element_zones=sample_element_zones)

        assert zone_def.get_zone_for_element(1) == 1
        assert zone_def.get_zone_for_element(3) == 1
        assert zone_def.get_zone_for_element(4) == 2
        assert zone_def.get_zone_for_element(6) == 2

    def test_get_zone_for_element_invalid_id(
        self, sample_zones: dict[int, Zone], sample_element_zones: np.ndarray
    ) -> None:
        """Test get_zone_for_element with invalid element ID."""
        zone_def = ZoneDefinition(zones=sample_zones, element_zones=sample_element_zones)

        assert zone_def.get_zone_for_element(0) == 0  # Invalid (0-based would be -1)
        assert zone_def.get_zone_for_element(100) == 0  # Out of range

    def test_get_zone_for_element_no_mapping(self) -> None:
        """Test get_zone_for_element when element_zones is None."""
        zone_def = ZoneDefinition()

        assert zone_def.get_zone_for_element(1) == 0

    def test_get_elements_in_zone(
        self, sample_zones: dict[int, Zone], sample_element_zones: np.ndarray
    ) -> None:
        """Test get_elements_in_zone method."""
        zone_def = ZoneDefinition(zones=sample_zones, element_zones=sample_element_zones)

        assert zone_def.get_elements_in_zone(1) == [1, 2, 3]
        assert zone_def.get_elements_in_zone(2) == [4, 5, 6]

    def test_get_elements_in_zone_not_found(
        self, sample_zones: dict[int, Zone], sample_element_zones: np.ndarray
    ) -> None:
        """Test get_elements_in_zone for non-existent zone."""
        zone_def = ZoneDefinition(zones=sample_zones, element_zones=sample_element_zones)

        assert zone_def.get_elements_in_zone(99) == []

    def test_get_zone(self, sample_zones: dict[int, Zone]) -> None:
        """Test get_zone method."""
        zone_def = ZoneDefinition(zones=sample_zones)

        zone = zone_def.get_zone(1)
        assert zone is not None
        assert zone.name == "North"

        assert zone_def.get_zone(99) is None

    def test_iter_zones(self, sample_zones: dict[int, Zone]) -> None:
        """Test iter_zones method."""
        zone_def = ZoneDefinition(zones=sample_zones)

        zones_list = list(zone_def.iter_zones())
        assert len(zones_list) == 2
        assert zones_list[0].id == 1
        assert zones_list[1].id == 2

    def test_add_zone(self) -> None:
        """Test add_zone method."""
        zone_def = ZoneDefinition(element_zones=np.zeros(10, dtype=np.int32))

        new_zone = Zone(id=1, name="New Zone", elements=[1, 2, 3])
        zone_def.add_zone(new_zone)

        assert zone_def.n_zones == 1
        assert zone_def.get_zone(1) is not None
        assert zone_def.get_zone_for_element(1) == 1
        assert zone_def.get_zone_for_element(2) == 1

    def test_add_zone_extends_array(self) -> None:
        """Test add_zone extends element_zones array if needed."""
        zone_def = ZoneDefinition(element_zones=np.zeros(5, dtype=np.int32))

        # Add zone with element beyond current array
        new_zone = Zone(id=1, name="New Zone", elements=[1, 10])
        zone_def.add_zone(new_zone)

        assert len(zone_def.element_zones) >= 10
        assert zone_def.get_zone_for_element(10) == 1

    def test_remove_zone(self, sample_zones: dict[int, Zone]) -> None:
        """Test remove_zone method."""
        sample_element_zones = np.array([1, 1, 1, 2, 2, 2], dtype=np.int32)
        zone_def = ZoneDefinition(zones=sample_zones, element_zones=sample_element_zones)

        removed = zone_def.remove_zone(1)

        assert removed is not None
        assert removed.name == "North"
        assert zone_def.n_zones == 1
        assert zone_def.get_zone_for_element(1) == 0  # Cleared

    def test_remove_zone_not_found(self) -> None:
        """Test remove_zone for non-existent zone."""
        zone_def = ZoneDefinition()

        result = zone_def.remove_zone(99)
        assert result is None

    def test_validate_valid(
        self, sample_zones: dict[int, Zone], sample_element_zones: np.ndarray
    ) -> None:
        """Test validate on valid zone definition."""
        zone_def = ZoneDefinition(zones=sample_zones, element_zones=sample_element_zones)

        errors = zone_def.validate(n_elements=6)
        assert len(errors) == 0

    def test_validate_array_size_mismatch(
        self, sample_zones: dict[int, Zone], sample_element_zones: np.ndarray
    ) -> None:
        """Test validate detects array size mismatch."""
        zone_def = ZoneDefinition(zones=sample_zones, element_zones=sample_element_zones)

        errors = zone_def.validate(n_elements=10)  # Mismatch
        assert len(errors) == 1
        assert "does not match" in errors[0]

    def test_validate_overlapping_zones(self) -> None:
        """Test validate detects overlapping zones."""
        zones = {
            1: Zone(id=1, name="A", elements=[1, 2, 3]),
            2: Zone(id=2, name="B", elements=[3, 4, 5]),  # Element 3 overlaps
        }
        zone_def = ZoneDefinition(zones=zones)

        errors = zone_def.validate(n_elements=5)
        assert len(errors) == 1
        assert "overlapping" in errors[0]

    def test_validate_id_mismatch(self) -> None:
        """Test validate detects zone ID mismatch."""
        zones = {
            1: Zone(id=99, name="Wrong ID", elements=[1]),  # ID doesn't match key
        }
        zone_def = ZoneDefinition(zones=zones)

        errors = zone_def.validate(n_elements=1)
        assert len(errors) == 1
        assert "does not match zone.id" in errors[0]

    def test_repr(self, sample_zones: dict[int, Zone]) -> None:
        """Test string representation."""
        zone_def = ZoneDefinition(
            zones=sample_zones,
            extent="horizontal",
            element_zones=np.array([1, 1, 1, 2, 2, 2], dtype=np.int32),
            name="Test Definition",
        )

        result = repr(zone_def)
        assert "ZoneDefinition" in result
        assert "n_zones=2" in result
        assert "extent='horizontal'" in result
        assert "name='Test Definition'" in result


# =============================================================================
# Test ZoneDefinition Factory Methods
# =============================================================================


class TestZoneDefinitionFactories:
    """Tests for ZoneDefinition factory methods."""

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

    def test_from_subregions(self, simple_grid: AppGrid) -> None:
        """Test creating zone definition from subregions."""
        zone_def = ZoneDefinition.from_subregions(simple_grid)

        assert zone_def.n_zones == 2
        assert zone_def.name == "Subregions"
        assert zone_def.get_zone(1).name == "Region A"
        assert zone_def.get_zone(2).name == "Region B"
        assert zone_def.get_zone_for_element(1) == 1
        assert zone_def.get_zone_for_element(2) == 2

    def test_from_element_list(self) -> None:
        """Test creating zone definition from element list."""
        pairs = [
            (1, 1),
            (2, 1),
            (3, 2),
            (4, 2),
        ]
        zone_names = {1: "Zone A", 2: "Zone B"}
        element_areas = {1: 100.0, 2: 100.0, 3: 150.0, 4: 150.0}

        zone_def = ZoneDefinition.from_element_list(
            element_zone_pairs=pairs,
            zone_names=zone_names,
            element_areas=element_areas,
            name="Test Definition",
        )

        assert zone_def.n_zones == 2
        assert zone_def.name == "Test Definition"
        assert zone_def.get_zone(1).name == "Zone A"
        assert zone_def.get_zone(1).area == pytest.approx(200.0)
        assert zone_def.get_zone(2).name == "Zone B"
        assert zone_def.get_zone(2).area == pytest.approx(300.0)

    def test_from_element_list_default_names(self) -> None:
        """Test from_element_list with default zone names."""
        pairs = [(1, 1), (2, 2)]

        zone_def = ZoneDefinition.from_element_list(element_zone_pairs=pairs)

        assert zone_def.get_zone(1).name == "Zone 1"
        assert zone_def.get_zone(2).name == "Zone 2"

    def test_from_element_list_skips_zero_zone(self) -> None:
        """Test from_element_list skips elements with zone 0."""
        pairs = [(1, 1), (2, 0), (3, 2)]  # Element 2 has no zone

        zone_def = ZoneDefinition.from_element_list(element_zone_pairs=pairs)

        assert zone_def.n_zones == 2
        assert zone_def.get_zone_for_element(2) == 0


# =============================================================================
# Test ZoneDefinition compute_areas
# =============================================================================


class TestZoneDefinitionComputeAreas:
    """Tests for ZoneDefinition.compute_areas method."""

    def test_compute_areas(self) -> None:
        """Test computing zone areas from grid."""
        # Create grid with known areas
        nodes = {
            1: Node(id=1, x=0.0, y=0.0),
            2: Node(id=2, x=100.0, y=0.0),
            3: Node(id=3, x=100.0, y=100.0),
            4: Node(id=4, x=0.0, y=100.0),
        }
        elements = {
            1: Element(id=1, vertices=(1, 2, 3, 4), subregion=1, area=10000.0),
        }
        grid = AppGrid(nodes=nodes, elements=elements)

        # Create zone definition with zero area
        zone = Zone(id=1, name="Test", elements=[1], area=0.0)
        zone_def = ZoneDefinition(zones={1: zone})

        # Compute areas
        zone_def.compute_areas(grid)

        assert zone_def.get_zone(1).area == pytest.approx(10000.0)
