"""Unit tests for lake component classes."""

from __future__ import annotations

import numpy as np
import pytest

from pyiwfm.components.lake import (
    AppLake,
    Lake,
    LakeElement,
    LakeOutflow,
    LakeRating,
)
from pyiwfm.core.exceptions import ComponentError


class TestLakeRating:
    """Tests for lake rating curve."""

    def test_rating_creation(self) -> None:
        """Test rating curve creation."""
        rating = LakeRating(
            elevations=np.array([100.0, 105.0, 110.0, 115.0]),
            areas=np.array([0.0, 1000.0, 5000.0, 12000.0]),
            volumes=np.array([0.0, 2500.0, 17500.0, 60000.0]),
        )

        assert len(rating.elevations) == 4
        assert len(rating.areas) == 4
        assert len(rating.volumes) == 4

    def test_rating_interpolate_area(self) -> None:
        """Test area interpolation from elevation."""
        rating = LakeRating(
            elevations=np.array([100.0, 105.0, 110.0]),
            areas=np.array([0.0, 1000.0, 4000.0]),
            volumes=np.array([0.0, 2500.0, 15000.0]),
        )

        # Exact match
        assert rating.get_area(105.0) == pytest.approx(1000.0)

        # Interpolation
        assert rating.get_area(102.5) == pytest.approx(500.0)

    def test_rating_interpolate_volume(self) -> None:
        """Test volume interpolation from elevation."""
        rating = LakeRating(
            elevations=np.array([100.0, 105.0, 110.0]),
            areas=np.array([0.0, 1000.0, 4000.0]),
            volumes=np.array([0.0, 2500.0, 15000.0]),
        )

        # Exact match
        assert rating.get_volume(105.0) == pytest.approx(2500.0)

        # Interpolation
        assert rating.get_volume(107.5) == pytest.approx(8750.0)

    def test_rating_get_elevation_from_volume(self) -> None:
        """Test elevation interpolation from volume."""
        rating = LakeRating(
            elevations=np.array([100.0, 105.0, 110.0]),
            areas=np.array([0.0, 1000.0, 4000.0]),
            volumes=np.array([0.0, 2500.0, 15000.0]),
        )

        assert rating.get_elevation(2500.0) == pytest.approx(105.0)

    def test_rating_below_minimum(self) -> None:
        """Test rating curve below minimum elevation."""
        rating = LakeRating(
            elevations=np.array([100.0, 105.0, 110.0]),
            areas=np.array([0.0, 1000.0, 4000.0]),
            volumes=np.array([0.0, 2500.0, 15000.0]),
        )

        # Below minimum should return 0
        assert rating.get_area(95.0) == 0.0
        assert rating.get_volume(95.0) == 0.0


class TestLakeElement:
    """Tests for lake element class."""

    def test_lake_element_creation(self) -> None:
        """Test basic lake element creation."""
        elem = LakeElement(
            element_id=10,
            lake_id=1,
            fraction=0.75,
        )

        assert elem.element_id == 10
        assert elem.lake_id == 1
        assert elem.fraction == 0.75

    def test_lake_element_defaults(self) -> None:
        """Test lake element default values."""
        elem = LakeElement(element_id=5, lake_id=1)

        assert elem.fraction == 1.0


class TestLakeOutflow:
    """Tests for lake outflow class."""

    def test_outflow_to_stream(self) -> None:
        """Test lake outflow to stream."""
        outflow = LakeOutflow(
            lake_id=1,
            destination_type="stream",
            destination_id=5,
            max_rate=1000.0,
        )

        assert outflow.lake_id == 1
        assert outflow.destination_type == "stream"
        assert outflow.destination_id == 5
        assert outflow.max_rate == 1000.0

    def test_outflow_to_another_lake(self) -> None:
        """Test lake outflow to another lake."""
        outflow = LakeOutflow(
            lake_id=1,
            destination_type="lake",
            destination_id=2,
        )

        assert outflow.destination_type == "lake"
        assert outflow.destination_id == 2

    def test_outflow_outside_model(self) -> None:
        """Test lake outflow leaving model domain."""
        outflow = LakeOutflow(
            lake_id=1,
            destination_type="outside",
            destination_id=0,
        )

        assert outflow.destination_type == "outside"

    def test_outflow_default_capacity(self) -> None:
        """Test outflow with unlimited capacity."""
        outflow = LakeOutflow(
            lake_id=1,
            destination_type="stream",
            destination_id=5,
        )

        assert outflow.max_rate == float("inf")


class TestLake:
    """Tests for lake class."""

    def test_lake_creation(self) -> None:
        """Test basic lake creation."""
        lake = Lake(
            id=1,
            name="Reservoir A",
            max_elevation=120.0,
        )

        assert lake.id == 1
        assert lake.name == "Reservoir A"
        assert lake.max_elevation == 120.0

    def test_lake_default_values(self) -> None:
        """Test lake default values."""
        lake = Lake(id=1)

        assert lake.name == ""
        assert lake.max_elevation == float("inf")
        assert lake.initial_storage == 0.0
        assert lake.elements == []

    def test_lake_with_elements(self) -> None:
        """Test lake with element list."""
        lake = Lake(
            id=1,
            name="Test Lake",
            elements=[10, 11, 12, 13],
        )

        assert lake.elements == [10, 11, 12, 13]
        assert lake.n_elements == 4

    def test_lake_with_rating(self) -> None:
        """Test lake with rating curve."""
        rating = LakeRating(
            elevations=np.array([100.0, 110.0, 120.0]),
            areas=np.array([0.0, 5000.0, 20000.0]),
            volumes=np.array([0.0, 25000.0, 150000.0]),
        )
        lake = Lake(id=1, name="Rated Lake", rating=rating)

        assert lake.rating is not None
        assert lake.rating.get_area(110.0) == pytest.approx(5000.0)

    def test_lake_with_outflow(self) -> None:
        """Test lake with outflow configuration."""
        outflow = LakeOutflow(
            lake_id=1,
            destination_type="stream",
            destination_id=10,
            max_rate=500.0,
        )
        lake = Lake(id=1, name="Outflow Lake", outflow=outflow)

        assert lake.outflow is not None
        assert lake.outflow.max_rate == 500.0

    def test_lake_gw_nodes(self) -> None:
        """Test lake groundwater node association."""
        lake = Lake(
            id=1,
            name="GW Lake",
            gw_nodes=[1, 2, 3, 4],
        )

        assert lake.gw_nodes == [1, 2, 3, 4]


class TestAppLake:
    """Tests for lake application class."""

    def test_applake_creation(self) -> None:
        """Test basic lake application creation."""
        lakes = AppLake()

        assert lakes.n_lakes == 0

    def test_applake_add_lake(self) -> None:
        """Test adding lakes."""
        lakes = AppLake()

        lake = Lake(id=1, name="Lake A")
        lakes.add_lake(lake)

        assert lakes.n_lakes == 1
        assert lakes.get_lake(1) == lake

    def test_applake_add_multiple_lakes(self) -> None:
        """Test adding multiple lakes."""
        lakes = AppLake()

        lakes.add_lake(Lake(id=1, name="Lake A"))
        lakes.add_lake(Lake(id=2, name="Lake B"))
        lakes.add_lake(Lake(id=3, name="Lake C"))

        assert lakes.n_lakes == 3

    def test_applake_add_lake_element(self) -> None:
        """Test adding lake elements."""
        lakes = AppLake()
        lakes.add_lake(Lake(id=1, name="Test"))

        elem = LakeElement(element_id=10, lake_id=1)
        lakes.add_lake_element(elem)

        assert lakes.n_lake_elements == 1

    def test_applake_get_elements_for_lake(self) -> None:
        """Test getting elements for a specific lake."""
        lakes = AppLake()
        lakes.add_lake(Lake(id=1, name="Lake A"))
        lakes.add_lake(Lake(id=2, name="Lake B"))

        lakes.add_lake_element(LakeElement(element_id=10, lake_id=1))
        lakes.add_lake_element(LakeElement(element_id=11, lake_id=1))
        lakes.add_lake_element(LakeElement(element_id=20, lake_id=2))

        elems_1 = lakes.get_elements_for_lake(1)
        assert len(elems_1) == 2
        assert all(e.lake_id == 1 for e in elems_1)

        elems_2 = lakes.get_elements_for_lake(2)
        assert len(elems_2) == 1

    def test_applake_total_area(self) -> None:
        """Test calculating total lake area."""
        lakes = AppLake()

        rating1 = LakeRating(
            elevations=np.array([100.0, 110.0]),
            areas=np.array([0.0, 1000.0]),
            volumes=np.array([0.0, 5000.0]),
        )
        rating2 = LakeRating(
            elevations=np.array([100.0, 110.0]),
            areas=np.array([0.0, 2000.0]),
            volumes=np.array([0.0, 10000.0]),
        )

        lakes.add_lake(Lake(id=1, name="A", rating=rating1))
        lakes.add_lake(Lake(id=2, name="B", rating=rating2))

        # Set current elevations
        lakes.set_elevation(1, 110.0)
        lakes.set_elevation(2, 110.0)

        total = lakes.get_total_area()
        assert total == pytest.approx(3000.0)

    def test_applake_total_volume(self) -> None:
        """Test calculating total lake volume."""
        lakes = AppLake()

        rating1 = LakeRating(
            elevations=np.array([100.0, 110.0]),
            areas=np.array([0.0, 1000.0]),
            volumes=np.array([0.0, 5000.0]),
        )
        rating2 = LakeRating(
            elevations=np.array([100.0, 110.0]),
            areas=np.array([0.0, 2000.0]),
            volumes=np.array([0.0, 10000.0]),
        )

        lakes.add_lake(Lake(id=1, name="A", rating=rating1))
        lakes.add_lake(Lake(id=2, name="B", rating=rating2))

        lakes.set_elevation(1, 110.0)
        lakes.set_elevation(2, 110.0)

        total = lakes.get_total_volume()
        assert total == pytest.approx(15000.0)

    def test_applake_iter_lakes(self) -> None:
        """Test iterating over lakes."""
        lakes = AppLake()

        lakes.add_lake(Lake(id=3, name="C"))
        lakes.add_lake(Lake(id=1, name="A"))
        lakes.add_lake(Lake(id=2, name="B"))

        # Should iterate in ID order
        ids = [lake.id for lake in lakes.iter_lakes()]
        assert ids == [1, 2, 3]

    def test_applake_validate(self) -> None:
        """Test lake validation."""
        lakes = AppLake()

        # Empty should pass validation
        lakes.validate()

    def test_applake_validate_orphan_element(self) -> None:
        """Test validation catches orphan lake elements."""
        lakes = AppLake()
        lakes.add_lake(Lake(id=1, name="A"))

        # Add element for non-existent lake
        elem = LakeElement(element_id=10, lake_id=99)
        lakes.add_lake_element(elem)

        with pytest.raises(ComponentError, match="lake"):
            lakes.validate()


class TestAppLakeIO:
    """Tests for lake I/O operations."""

    def test_applake_to_arrays(self) -> None:
        """Test converting lakes to arrays."""
        lakes = AppLake()

        lakes.add_lake(Lake(id=1, name="Lake A", max_elevation=120.0))
        lakes.add_lake(Lake(id=2, name="Lake B", max_elevation=130.0))

        lakes.set_elevation(1, 115.0)
        lakes.set_elevation(2, 125.0)

        arrays = lakes.to_arrays()

        assert "lake_ids" in arrays
        assert "elevations" in arrays
        np.testing.assert_array_equal(arrays["lake_ids"], [1, 2])
        np.testing.assert_array_equal(arrays["elevations"], [115.0, 125.0])

    def test_applake_from_arrays(self) -> None:
        """Test creating lakes from arrays."""
        lake_ids = np.array([1, 2, 3])
        names = ["Lake A", "Lake B", "Lake C"]
        max_elevations = np.array([120.0, 130.0, 140.0])

        lakes = AppLake.from_arrays(
            lake_ids=lake_ids,
            names=names,
            max_elevations=max_elevations,
        )

        assert lakes.n_lakes == 3
        assert lakes.get_lake(1).name == "Lake A"
        assert lakes.get_lake(2).max_elevation == 130.0


# =============================================================================
# Additional tests for 95%+ coverage
# =============================================================================


class TestLakeRatingValidation:
    """Tests for lake rating curve validation and edge cases."""

    def test_rating_length_mismatch(self) -> None:
        """Test rating curve raises ValueError for mismatched array lengths."""
        with pytest.raises(ValueError, match="same length"):
            LakeRating(
                elevations=np.array([100.0, 105.0]),
                areas=np.array([0.0, 1000.0, 5000.0]),
                volumes=np.array([0.0, 2500.0]),
            )

    def test_rating_too_few_points(self) -> None:
        """Test rating curve raises ValueError for fewer than 2 points."""
        with pytest.raises(ValueError, match="at least 2 points"):
            LakeRating(
                elevations=np.array([100.0]),
                areas=np.array([0.0]),
                volumes=np.array([0.0]),
            )

    def test_rating_area_extrapolation_above_max(self) -> None:
        """Test area linear extrapolation above maximum elevation."""
        rating = LakeRating(
            elevations=np.array([100.0, 110.0]),
            areas=np.array([0.0, 1000.0]),
            volumes=np.array([0.0, 5000.0]),
        )
        # slope = (1000 - 0) / (110 - 100) = 100 per unit
        # area = 1000 + 100 * (120 - 110) = 2000
        assert rating.get_area(120.0) == pytest.approx(2000.0)

    def test_rating_volume_extrapolation_above_max(self) -> None:
        """Test volume linear extrapolation above maximum elevation."""
        rating = LakeRating(
            elevations=np.array([100.0, 110.0]),
            areas=np.array([0.0, 1000.0]),
            volumes=np.array([0.0, 5000.0]),
        )
        # slope = (5000 - 0) / (110 - 100) = 500 per unit
        # volume = 5000 + 500 * (120 - 110) = 10000
        assert rating.get_volume(120.0) == pytest.approx(10000.0)

    def test_rating_elevation_below_minimum_volume(self) -> None:
        """Test elevation returns minimum for volume below min."""
        rating = LakeRating(
            elevations=np.array([100.0, 110.0]),
            areas=np.array([0.0, 1000.0]),
            volumes=np.array([0.0, 5000.0]),
        )
        assert rating.get_elevation(-1.0) == pytest.approx(100.0)

    def test_rating_elevation_extrapolation_above_max_volume(self) -> None:
        """Test elevation linear extrapolation above maximum volume."""
        rating = LakeRating(
            elevations=np.array([100.0, 110.0]),
            areas=np.array([0.0, 1000.0]),
            volumes=np.array([0.0, 5000.0]),
        )
        # slope = (110 - 100) / (5000 - 0) = 0.002 per unit
        # elevation = 110 + 0.002 * (10000 - 5000) = 120
        assert rating.get_elevation(10000.0) == pytest.approx(120.0)

    def test_rating_area_at_exact_minimum(self) -> None:
        """Test area at exact minimum elevation returns 0."""
        rating = LakeRating(
            elevations=np.array([100.0, 110.0]),
            areas=np.array([0.0, 1000.0]),
            volumes=np.array([0.0, 5000.0]),
        )
        assert rating.get_area(100.0) == 0.0

    def test_rating_volume_at_exact_minimum(self) -> None:
        """Test volume at exact minimum elevation returns 0."""
        rating = LakeRating(
            elevations=np.array([100.0, 110.0]),
            areas=np.array([0.0, 1000.0]),
            volumes=np.array([0.0, 5000.0]),
        )
        assert rating.get_volume(100.0) == 0.0

    def test_rating_elevation_at_exact_minimum_volume(self) -> None:
        """Test elevation at exact minimum volume returns minimum elevation."""
        rating = LakeRating(
            elevations=np.array([100.0, 110.0]),
            areas=np.array([0.0, 1000.0]),
            volumes=np.array([0.0, 5000.0]),
        )
        assert rating.get_elevation(0.0) == pytest.approx(100.0)


class TestReprMethods:
    """Tests for __repr__ methods on lake classes."""

    def test_lake_element_repr(self) -> None:
        """Test LakeElement __repr__."""
        elem = LakeElement(element_id=10, lake_id=1, fraction=0.75)
        result = repr(elem)
        assert "LakeElement" in result
        assert "elem=10" in result
        assert "lake=1" in result

    def test_lake_outflow_repr(self) -> None:
        """Test LakeOutflow __repr__."""
        outflow = LakeOutflow(
            lake_id=1, destination_type="stream", destination_id=5, max_rate=1000.0
        )
        result = repr(outflow)
        assert "LakeOutflow" in result
        assert "lake=1" in result
        assert "stream:5" in result

    def test_lake_repr(self) -> None:
        """Test Lake __repr__."""
        lake = Lake(id=1, name="Reservoir A")
        result = repr(lake)
        assert "Lake" in result
        assert "id=1" in result
        assert "Reservoir A" in result

    def test_applake_repr(self) -> None:
        """Test AppLake __repr__."""
        app = AppLake()
        app.add_lake(Lake(id=1, name="A"))
        app.add_lake(Lake(id=2, name="B"))
        result = repr(app)
        assert "AppLake" in result
        assert "n_lakes=2" in result


class TestAppLakeEdgeCases:
    """Additional edge case tests for AppLake."""

    def test_get_area_no_rating(self) -> None:
        """Test get_area returns 0.0 for a lake without a rating curve."""
        app = AppLake()
        app.add_lake(Lake(id=1, name="No Rating"))
        assert app.get_area(1) == 0.0

    def test_get_volume_no_rating(self) -> None:
        """Test get_volume returns 0.0 for a lake without a rating curve."""
        app = AppLake()
        app.add_lake(Lake(id=1, name="No Rating"))
        assert app.get_volume(1) == 0.0

    def test_get_elevation_default(self) -> None:
        """Test get_elevation returns 0.0 for unset elevation."""
        app = AppLake()
        app.add_lake(Lake(id=1, name="Unset"))
        assert app.get_elevation(1) == 0.0

    def test_validate_outflow_to_nonexistent_lake(self) -> None:
        """Test validation catches outflow to non-existent lake."""
        app = AppLake()
        outflow = LakeOutflow(lake_id=1, destination_type="lake", destination_id=99)
        app.add_lake(Lake(id=1, name="Source", outflow=outflow))

        with pytest.raises(ComponentError, match="non-existent lake"):
            app.validate()

    def test_validate_outflow_to_stream_ok(self) -> None:
        """Test validation passes for outflow to stream (not validated)."""
        app = AppLake()
        outflow = LakeOutflow(lake_id=1, destination_type="stream", destination_id=5)
        app.add_lake(Lake(id=1, name="Source", outflow=outflow))
        # Should not raise
        app.validate()

    def test_from_arrays_no_max_elevations(self) -> None:
        """Test from_arrays with no max_elevations uses inf."""
        lake_ids = np.array([1, 2])
        names = ["A", "B"]
        lakes = AppLake.from_arrays(lake_ids=lake_ids, names=names)

        assert lakes.get_lake(1).max_elevation == float("inf")
        assert lakes.get_lake(2).max_elevation == float("inf")

    def test_from_arrays_names_shorter_than_ids(self) -> None:
        """Test from_arrays when names list is shorter than lake_ids."""
        lake_ids = np.array([1, 2, 3])
        names = ["Only One"]
        lakes = AppLake.from_arrays(lake_ids=lake_ids, names=names)

        assert lakes.get_lake(1).name == "Only One"
        assert lakes.get_lake(2).name == ""
        assert lakes.get_lake(3).name == ""

    def test_to_arrays_with_missing_elevation(self) -> None:
        """Test to_arrays defaults to 0.0 for lakes without set elevation."""
        app = AppLake()
        app.add_lake(Lake(id=1, name="A"))
        app.add_lake(Lake(id=2, name="B"))
        app.set_elevation(1, 50.0)
        # Lake 2 elevation not set

        arrays = app.to_arrays()
        np.testing.assert_array_equal(arrays["lake_ids"], [1, 2])
        np.testing.assert_array_equal(arrays["elevations"], [50.0, 0.0])
