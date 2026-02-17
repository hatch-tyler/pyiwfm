"""Unit tests for root zone component classes."""

from __future__ import annotations

import numpy as np
import pytest

from pyiwfm.components.rootzone import (
    CropType,
    ElementLandUse,
    LandUseType,
    RootZone,
    SoilParameters,
)
from pyiwfm.core.exceptions import ComponentError


class TestLandUseType:
    """Tests for land use type enumeration."""

    def test_land_use_types(self) -> None:
        """Test land use type values."""
        assert LandUseType.AGRICULTURAL.value == "agricultural"
        assert LandUseType.URBAN.value == "urban"
        assert LandUseType.NATIVE_RIPARIAN.value == "native_riparian"
        assert LandUseType.WATER.value == "water"

    def test_land_use_from_string(self) -> None:
        """Test creating land use from string."""
        assert LandUseType("agricultural") == LandUseType.AGRICULTURAL
        assert LandUseType("urban") == LandUseType.URBAN


class TestCropType:
    """Tests for crop type class."""

    def test_crop_creation(self) -> None:
        """Test basic crop type creation."""
        crop = CropType(
            id=1,
            name="Corn",
            root_depth=1.2,
            kc=1.0,  # Crop coefficient
        )

        assert crop.id == 1
        assert crop.name == "Corn"
        assert crop.root_depth == 1.2
        assert crop.kc == 1.0

    def test_crop_defaults(self) -> None:
        """Test crop type default values."""
        crop = CropType(id=1)

        assert crop.name == ""
        assert crop.root_depth == 0.0
        assert crop.kc == 1.0

    def test_crop_seasonal_kc(self) -> None:
        """Test crop with seasonal crop coefficients."""
        crop = CropType(
            id=1,
            name="Wheat",
            monthly_kc=np.array([0.3, 0.3, 0.5, 0.8, 1.1, 1.2, 1.1, 0.9, 0.6, 0.4, 0.3, 0.3]),
        )

        assert len(crop.monthly_kc) == 12
        assert crop.monthly_kc[5] == 1.2  # June peak


class TestSoilParameters:
    """Tests for soil parameters."""

    def test_soil_params_creation(self) -> None:
        """Test soil parameter creation."""
        soil = SoilParameters(
            porosity=0.4,
            field_capacity=0.25,
            wilting_point=0.1,
            saturated_kv=10.0,
        )

        assert soil.porosity == 0.4
        assert soil.field_capacity == 0.25
        assert soil.wilting_point == 0.1
        assert soil.saturated_kv == 10.0

    def test_soil_available_water(self) -> None:
        """Test available water capacity calculation."""
        soil = SoilParameters(
            porosity=0.4,
            field_capacity=0.25,
            wilting_point=0.1,
            saturated_kv=10.0,
        )

        # Available water = field_capacity - wilting_point
        assert soil.available_water == pytest.approx(0.15)

    def test_soil_drainable_porosity(self) -> None:
        """Test drainable porosity calculation."""
        soil = SoilParameters(
            porosity=0.4,
            field_capacity=0.25,
            wilting_point=0.1,
            saturated_kv=10.0,
        )

        # Drainable = porosity - field_capacity
        assert soil.drainable_porosity == pytest.approx(0.15)


class TestElementLandUse:
    """Tests for element land use class."""

    def test_element_landuse_creation(self) -> None:
        """Test element land use creation."""
        elu = ElementLandUse(
            element_id=10,
            land_use_type=LandUseType.AGRICULTURAL,
            area=5000.0,
        )

        assert elu.element_id == 10
        assert elu.land_use_type == LandUseType.AGRICULTURAL
        assert elu.area == 5000.0

    def test_element_landuse_with_crops(self) -> None:
        """Test element land use with crop fractions."""
        elu = ElementLandUse(
            element_id=10,
            land_use_type=LandUseType.AGRICULTURAL,
            area=10000.0,
            crop_fractions={1: 0.4, 2: 0.3, 3: 0.3},  # crop_id: fraction
        )

        assert sum(elu.crop_fractions.values()) == pytest.approx(1.0)
        assert elu.crop_fractions[1] == 0.4

    def test_element_landuse_urban(self) -> None:
        """Test urban land use element."""
        elu = ElementLandUse(
            element_id=5,
            land_use_type=LandUseType.URBAN,
            area=2000.0,
            impervious_fraction=0.6,
        )

        assert elu.land_use_type == LandUseType.URBAN
        assert elu.impervious_fraction == 0.6

    def test_element_landuse_native(self) -> None:
        """Test native/riparian land use element."""
        elu = ElementLandUse(
            element_id=15,
            land_use_type=LandUseType.NATIVE_RIPARIAN,
            area=8000.0,
        )

        assert elu.land_use_type == LandUseType.NATIVE_RIPARIAN


class TestRootZone:
    """Tests for root zone class."""

    def test_rootzone_creation(self) -> None:
        """Test basic root zone creation."""
        rz = RootZone(n_elements=10, n_layers=1)

        assert rz.n_elements == 10
        assert rz.n_layers == 1
        assert rz.n_crop_types == 0

    def test_rootzone_add_crop_type(self) -> None:
        """Test adding crop types."""
        rz = RootZone(n_elements=10, n_layers=1)

        crop = CropType(id=1, name="Corn", root_depth=1.0)
        rz.add_crop_type(crop)

        assert rz.n_crop_types == 1
        assert rz.get_crop_type(1) == crop

    def test_rootzone_add_soil_params(self) -> None:
        """Test adding soil parameters."""
        rz = RootZone(n_elements=10, n_layers=1)

        soil = SoilParameters(
            porosity=0.4,
            field_capacity=0.25,
            wilting_point=0.1,
            saturated_kv=10.0,
        )
        rz.set_soil_parameters(5, soil)

        assert rz.get_soil_parameters(5) == soil

    def test_rootzone_add_element_landuse(self) -> None:
        """Test adding element land use."""
        rz = RootZone(n_elements=10, n_layers=1)

        elu = ElementLandUse(
            element_id=5,
            land_use_type=LandUseType.AGRICULTURAL,
            area=5000.0,
        )
        rz.add_element_landuse(elu)

        assert len(rz.element_landuse) == 1

    def test_rootzone_get_landuse_by_element(self) -> None:
        """Test getting land use for an element."""
        rz = RootZone(n_elements=10, n_layers=1)

        rz.add_element_landuse(
            ElementLandUse(
                element_id=5,
                land_use_type=LandUseType.AGRICULTURAL,
                area=5000.0,
            )
        )
        rz.add_element_landuse(
            ElementLandUse(
                element_id=5,
                land_use_type=LandUseType.URBAN,
                area=1000.0,
            )
        )

        uses = rz.get_landuse_for_element(5)
        assert len(uses) == 2

    def test_rootzone_total_area_by_type(self) -> None:
        """Test calculating total area by land use type."""
        rz = RootZone(n_elements=10, n_layers=1)

        rz.add_element_landuse(
            ElementLandUse(
                element_id=1,
                land_use_type=LandUseType.AGRICULTURAL,
                area=5000.0,
            )
        )
        rz.add_element_landuse(
            ElementLandUse(
                element_id=2,
                land_use_type=LandUseType.AGRICULTURAL,
                area=3000.0,
            )
        )
        rz.add_element_landuse(
            ElementLandUse(
                element_id=3,
                land_use_type=LandUseType.URBAN,
                area=2000.0,
            )
        )

        ag_area = rz.get_total_area(LandUseType.AGRICULTURAL)
        assert ag_area == pytest.approx(8000.0)

        urban_area = rz.get_total_area(LandUseType.URBAN)
        assert urban_area == pytest.approx(2000.0)

    def test_rootzone_set_soil_moisture(self) -> None:
        """Test setting soil moisture content."""
        rz = RootZone(n_elements=10, n_layers=1)

        moisture = np.ones((10, 1)) * 0.2
        rz.set_soil_moisture(moisture)

        assert rz.soil_moisture.shape == (10, 1)
        assert rz.soil_moisture[0, 0] == pytest.approx(0.2)

    def test_rootzone_get_soil_moisture(self) -> None:
        """Test getting soil moisture for element/layer."""
        rz = RootZone(n_elements=10, n_layers=2)

        moisture = np.arange(20).reshape(10, 2).astype(float) / 100
        rz.set_soil_moisture(moisture)

        assert rz.get_soil_moisture(0, 0) == pytest.approx(0.0)
        assert rz.get_soil_moisture(5, 1) == pytest.approx(0.11)

    def test_rootzone_validate(self) -> None:
        """Test root zone validation."""
        rz = RootZone(n_elements=10, n_layers=1)

        # Should pass basic validation
        rz.validate()

    def test_rootzone_validate_invalid_element(self) -> None:
        """Test validation catches invalid element references."""
        rz = RootZone(n_elements=5, n_layers=1)

        rz.add_element_landuse(
            ElementLandUse(
                element_id=100,  # Invalid element
                land_use_type=LandUseType.AGRICULTURAL,
                area=5000.0,
            )
        )

        with pytest.raises(ComponentError, match="element"):
            rz.validate()

    def test_rootzone_iter_elements(self) -> None:
        """Test iterating over elements with land use."""
        rz = RootZone(n_elements=10, n_layers=1)

        rz.add_element_landuse(
            ElementLandUse(element_id=3, land_use_type=LandUseType.URBAN, area=1000.0)
        )
        rz.add_element_landuse(
            ElementLandUse(element_id=1, land_use_type=LandUseType.AGRICULTURAL, area=2000.0)
        )
        rz.add_element_landuse(
            ElementLandUse(element_id=5, land_use_type=LandUseType.NATIVE_RIPARIAN, area=3000.0)
        )

        elements = list(rz.iter_elements_with_landuse())
        assert len(elements) == 3


class TestRootZoneIO:
    """Tests for root zone I/O operations."""

    def test_rootzone_to_arrays(self) -> None:
        """Test converting root zone to arrays."""
        rz = RootZone(n_elements=4, n_layers=1)

        moisture = np.array([[0.2], [0.25], [0.3], [0.15]])
        rz.set_soil_moisture(moisture)

        arrays = rz.to_arrays()

        assert "soil_moisture" in arrays
        np.testing.assert_array_almost_equal(arrays["soil_moisture"], moisture)

    def test_rootzone_from_arrays(self) -> None:
        """Test creating root zone from arrays."""
        n_elements = 5
        n_layers = 2

        soil_moisture = np.ones((n_elements, n_layers)) * 0.25

        rz = RootZone.from_arrays(
            n_elements=n_elements,
            n_layers=n_layers,
            soil_moisture=soil_moisture,
        )

        assert rz.n_elements == n_elements
        assert rz.n_layers == n_layers
        assert rz.get_soil_moisture(0, 0) == pytest.approx(0.25)


# =============================================================================
# Additional tests for 95%+ coverage
# =============================================================================


class TestCropTypeEdgeCases:
    """Additional tests for CropType edge cases."""

    def test_crop_get_kc_monthly(self) -> None:
        """Test get_kc returns monthly value when monthly_kc is set."""
        monthly = np.array([0.3, 0.3, 0.5, 0.8, 1.1, 1.2, 1.1, 0.9, 0.6, 0.4, 0.3, 0.3])
        crop = CropType(id=1, name="Wheat", kc=0.7, monthly_kc=monthly)

        # month=0 should return annual kc
        assert crop.get_kc(0) == pytest.approx(0.7)
        # month=6 should return June value (index 5)
        assert crop.get_kc(6) == pytest.approx(1.2)
        # month=1 should return January value (index 0)
        assert crop.get_kc(1) == pytest.approx(0.3)
        # month=12 should return December value (index 11)
        assert crop.get_kc(12) == pytest.approx(0.3)

    def test_crop_get_kc_no_monthly(self) -> None:
        """Test get_kc returns annual kc when no monthly_kc is set."""
        crop = CropType(id=1, name="Corn", kc=1.0)
        assert crop.get_kc(6) == pytest.approx(1.0)

    def test_crop_repr(self) -> None:
        """Test CropType __repr__."""
        crop = CropType(id=1, name="Corn")
        result = repr(crop)
        assert "CropType" in result
        assert "id=1" in result
        assert "Corn" in result


class TestSoilParametersRepr:
    """Test SoilParameters repr."""

    def test_soil_params_repr(self) -> None:
        """Test SoilParameters __repr__."""
        soil = SoilParameters(
            porosity=0.4,
            field_capacity=0.25,
            wilting_point=0.1,
            saturated_kv=10.0,
        )
        result = repr(soil)
        assert "SoilParameters" in result
        assert "n=0.400" in result
        assert "fc=0.250" in result


class TestElementLandUseRepr:
    """Test ElementLandUse repr."""

    def test_element_landuse_repr(self) -> None:
        """Test ElementLandUse __repr__."""
        elu = ElementLandUse(
            element_id=10,
            land_use_type=LandUseType.AGRICULTURAL,
            area=5000.0,
        )
        result = repr(elu)
        assert "ElementLandUse" in result
        assert "elem=10" in result
        assert "agricultural" in result


class TestRootZoneEdgeCases:
    """Additional edge case tests for RootZone."""

    def test_set_soil_moisture_shape_mismatch(self) -> None:
        """Test set_soil_moisture raises ValueError for shape mismatch."""
        rz = RootZone(n_elements=10, n_layers=2)
        wrong_shape = np.ones((5, 3))

        with pytest.raises(ValueError, match="shape"):
            rz.set_soil_moisture(wrong_shape)

    def test_get_soil_moisture_not_set(self) -> None:
        """Test get_soil_moisture raises ValueError when not set."""
        rz = RootZone(n_elements=10, n_layers=1)

        with pytest.raises(ValueError, match="not set"):
            rz.get_soil_moisture(0, 0)

    def test_validate_crop_reference_error(self) -> None:
        """Test validation catches undefined crop type references."""
        rz = RootZone(n_elements=10, n_layers=1)
        rz.add_element_landuse(
            ElementLandUse(
                element_id=1,  # valid 1-based ID
                land_use_type=LandUseType.AGRICULTURAL,
                area=5000.0,
                crop_fractions={99: 0.5},  # crop 99 not defined
            )
        )

        with pytest.raises(ComponentError, match="crop type"):
            rz.validate()

    def test_validate_zero_element_id(self) -> None:
        """Test validation catches zero element ID (IDs are 1-based)."""
        rz = RootZone(n_elements=5, n_layers=1)
        rz.add_element_landuse(
            ElementLandUse(
                element_id=0,
                land_use_type=LandUseType.URBAN,
                area=1000.0,
            )
        )

        with pytest.raises(ComponentError, match="element"):
            rz.validate()

    def test_validate_negative_element_id(self) -> None:
        """Test validation catches negative element IDs."""
        rz = RootZone(n_elements=5, n_layers=1)
        rz.add_element_landuse(
            ElementLandUse(
                element_id=-1,
                land_use_type=LandUseType.URBAN,
                area=1000.0,
            )
        )

        with pytest.raises(ComponentError, match="element"):
            rz.validate()

    def test_validate_element_id_exceeds_n_elements(self) -> None:
        """Test validation catches element ID > n_elements (1-based)."""
        rz = RootZone(n_elements=5, n_layers=1)
        rz.add_element_landuse(
            ElementLandUse(
                element_id=6,  # n_elements=5, so max valid ID is 5
                land_use_type=LandUseType.URBAN,
                area=1000.0,
            )
        )

        with pytest.raises(ComponentError, match="element"):
            rz.validate()

    def test_validate_max_valid_element_id(self) -> None:
        """Test validation accepts element_id == n_elements (1-based)."""
        rz = RootZone(n_elements=5, n_layers=1)
        rz.add_element_landuse(
            ElementLandUse(
                element_id=5,  # n_elements=5, so 5 is valid
                land_use_type=LandUseType.URBAN,
                area=1000.0,
            )
        )

        # Should not raise — element 5 is valid for n_elements=5
        rz.validate()

    def test_from_arrays_no_soil_moisture(self) -> None:
        """Test from_arrays without soil moisture."""
        rz = RootZone.from_arrays(n_elements=5, n_layers=2)

        assert rz.n_elements == 5
        assert rz.n_layers == 2
        assert rz.soil_moisture is None

    def test_to_arrays_no_soil_moisture(self) -> None:
        """Test to_arrays returns empty dict when no soil moisture set."""
        rz = RootZone(n_elements=4, n_layers=1)
        arrays = rz.to_arrays()
        assert "soil_moisture" not in arrays

    def test_rootzone_repr(self) -> None:
        """Test RootZone __repr__."""
        rz = RootZone(n_elements=10, n_layers=1)
        rz.add_crop_type(CropType(id=1, name="Corn"))
        result = repr(rz)
        assert "RootZone" in result
        assert "n_elements=10" in result
        assert "n_crops=1" in result

    def test_iter_elements_with_landuse_deduplication(self) -> None:
        """Test iter_elements deduplicates when element has multiple land uses."""
        rz = RootZone(n_elements=10, n_layers=1)
        rz.add_element_landuse(
            ElementLandUse(element_id=5, land_use_type=LandUseType.AGRICULTURAL, area=3000.0)
        )
        rz.add_element_landuse(
            ElementLandUse(element_id=5, land_use_type=LandUseType.URBAN, area=1000.0)
        )
        rz.add_element_landuse(
            ElementLandUse(element_id=7, land_use_type=LandUseType.NATIVE_RIPARIAN, area=2000.0)
        )

        elements = list(rz.iter_elements_with_landuse())
        assert elements == [5, 7]

    def test_set_soil_moisture_copies_array(self) -> None:
        """Test set_soil_moisture makes a copy of the array."""
        rz = RootZone(n_elements=2, n_layers=1)
        moisture = np.array([[0.2], [0.3]])
        rz.set_soil_moisture(moisture)

        # Modify original - should not affect stored moisture
        moisture[0, 0] = 999.0
        assert rz.get_soil_moisture(0, 0) == pytest.approx(0.2)
