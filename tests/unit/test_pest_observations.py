"""Unit tests for PEST++ observation classes."""

from datetime import datetime

import numpy as np
import pytest

from pyiwfm.runner.pest_observations import (
    DerivedObservation,
    IWFMObservation,
    IWFMObservationGroup,
    IWFMObservationType,
    ObservationLocation,
    WeightStrategy,
)


class TestIWFMObservationType:
    """Tests for IWFMObservationType enum."""

    def test_groundwater_types_exist(self):
        """Test groundwater observation types exist."""
        assert IWFMObservationType.HEAD.value == "head"
        assert IWFMObservationType.DRAWDOWN.value == "drawdown"
        assert IWFMObservationType.HEAD_DIFFERENCE.value == "hdiff"
        assert IWFMObservationType.VERTICAL_GRADIENT.value == "vgrad"

    def test_stream_types_exist(self):
        """Test stream observation types exist."""
        assert IWFMObservationType.STREAM_FLOW.value == "flow"
        assert IWFMObservationType.STREAM_STAGE.value == "stage"
        assert IWFMObservationType.STREAM_GAIN_LOSS.value == "sgl"

    def test_lake_types_exist(self):
        """Test lake observation types exist."""
        assert IWFMObservationType.LAKE_LEVEL.value == "lake"
        assert IWFMObservationType.LAKE_STORAGE.value == "lsto"

    def test_budget_types_exist(self):
        """Test budget observation types exist."""
        assert IWFMObservationType.GW_BUDGET.value == "gwbud"
        assert IWFMObservationType.STREAM_BUDGET.value == "strbud"
        assert IWFMObservationType.ROOTZONE_BUDGET.value == "rzbud"

    def test_default_transform(self):
        """Test default transform for different observation types."""
        # Log transform for flow
        assert IWFMObservationType.STREAM_FLOW.default_transform == "log"
        assert IWFMObservationType.LAKE_STORAGE.default_transform == "log"

        # No transform for head
        assert IWFMObservationType.HEAD.default_transform == "none"
        assert IWFMObservationType.DRAWDOWN.default_transform == "none"

    def test_typical_error(self):
        """Test typical error values."""
        assert IWFMObservationType.HEAD.typical_error == 1.0
        assert IWFMObservationType.DRAWDOWN.typical_error == 0.5
        assert IWFMObservationType.STREAM_FLOW.typical_error == 0.1  # 10% relative

    def test_is_relative_error(self):
        """Test relative error flag."""
        assert IWFMObservationType.STREAM_FLOW.is_relative_error is True
        assert IWFMObservationType.HEAD.is_relative_error is False

    def test_group_prefix(self):
        """Test group prefix generation."""
        assert IWFMObservationType.HEAD.group_prefix == "head"
        assert IWFMObservationType.STREAM_FLOW.group_prefix == "flow"


class TestWeightStrategy:
    """Tests for WeightStrategy enum."""

    def test_strategy_values(self):
        """Test weight strategy values."""
        assert WeightStrategy.EQUAL.value == "equal"
        assert WeightStrategy.INVERSE_VARIANCE.value == "inverse_variance"
        assert WeightStrategy.GROUP_CONTRIBUTION.value == "group_contribution"
        assert WeightStrategy.TEMPORAL_DECAY.value == "temporal_decay"


class TestObservationLocation:
    """Tests for ObservationLocation class."""

    def test_basic_creation(self):
        """Test basic location creation."""
        loc = ObservationLocation(x=100.0, y=200.0)
        assert loc.x == 100.0
        assert loc.y == 200.0
        assert loc.z is None

    def test_full_creation(self):
        """Test location with all attributes."""
        loc = ObservationLocation(
            x=100.0,
            y=200.0,
            z=-50.0,
            node_id=123,
            element_id=45,
            layer=2,
            reach_id=10,
        )
        assert loc.x == 100.0
        assert loc.y == 200.0
        assert loc.z == -50.0
        assert loc.node_id == 123
        assert loc.element_id == 45
        assert loc.layer == 2
        assert loc.reach_id == 10

    def test_to_dict(self):
        """Test conversion to dictionary."""
        loc = ObservationLocation(x=100.0, y=200.0, layer=1)
        d = loc.to_dict()
        assert d["x"] == 100.0
        assert d["y"] == 200.0
        assert d["layer"] == 1
        assert "z" not in d  # None values excluded


class TestIWFMObservation:
    """Tests for IWFMObservation class."""

    def test_basic_creation(self):
        """Test basic observation creation."""
        obs = IWFMObservation(name="obs1", value=100.0)
        assert obs.name == "obs1"
        assert obs.value == 100.0
        assert obs.weight == 1.0
        assert obs.group == "default"

    def test_with_obs_type(self):
        """Test observation with type."""
        obs = IWFMObservation(
            name="head_well1",
            value=250.5,
            obs_type=IWFMObservationType.HEAD,
        )
        assert obs.obs_type == IWFMObservationType.HEAD

    def test_with_datetime(self):
        """Test observation with datetime."""
        dt = datetime(2020, 1, 15)
        obs = IWFMObservation(
            name="head_20200115",
            value=250.5,
            datetime=dt,
        )
        assert obs.datetime == dt

    def test_with_location(self):
        """Test observation with location."""
        loc = ObservationLocation(x=100.0, y=200.0, layer=1)
        obs = IWFMObservation(
            name="head_well1",
            value=250.5,
            location=loc,
        )
        assert obs.location.x == 100.0
        assert obs.location.layer == 1

    def test_invalid_name_length(self):
        """Test that long names raise error."""
        with pytest.raises(ValueError, match="too long"):
            IWFMObservation(name="x" * 201, value=100.0)

    def test_invalid_weight(self):
        """Test that negative weights raise error."""
        with pytest.raises(ValueError, match="non-negative"):
            IWFMObservation(name="obs1", value=100.0, weight=-1.0)

    def test_invalid_transform(self):
        """Test that invalid transform raises error."""
        with pytest.raises(ValueError, match="Invalid transform"):
            IWFMObservation(name="obs1", value=100.0, transform="invalid")

    def test_log_transform_positive_value(self):
        """Test log transform requires positive values."""
        with pytest.raises(ValueError, match="positive value"):
            IWFMObservation(name="obs1", value=-1.0, transform="log")

    def test_transformed_value_none(self):
        """Test transformed value with no transform."""
        obs = IWFMObservation(name="obs1", value=100.0, transform="none")
        assert obs.transformed_value == 100.0

    def test_transformed_value_log(self):
        """Test transformed value with log transform."""
        obs = IWFMObservation(name="obs1", value=100.0, transform="log")
        assert obs.transformed_value == pytest.approx(2.0)  # log10(100)

    def test_transformed_value_sqrt(self):
        """Test transformed value with sqrt transform."""
        obs = IWFMObservation(name="obs1", value=100.0, transform="sqrt")
        assert obs.transformed_value == pytest.approx(10.0)

    def test_calculate_weight_equal(self):
        """Test weight calculation with equal strategy."""
        obs = IWFMObservation(name="obs1", value=100.0)
        weight = obs.calculate_weight(WeightStrategy.EQUAL)
        assert weight == 1.0

    def test_calculate_weight_inverse_variance(self):
        """Test weight calculation with inverse variance."""
        obs = IWFMObservation(
            name="obs1",
            value=100.0,
            obs_type=IWFMObservationType.HEAD,
        )
        weight = obs.calculate_weight(WeightStrategy.INVERSE_VARIANCE)
        assert weight == pytest.approx(1.0)  # 1/1.0 (typical error)

    def test_calculate_weight_with_error_std(self):
        """Test weight calculation with specified error."""
        obs = IWFMObservation(
            name="obs1",
            value=100.0,
            error_std=2.0,
        )
        weight = obs.calculate_weight(WeightStrategy.INVERSE_VARIANCE)
        assert weight == pytest.approx(0.5)

    def test_calculate_weight_temporal_decay(self):
        """Test weight calculation with temporal decay."""
        reference = datetime(2020, 1, 1)
        obs = IWFMObservation(
            name="obs1",
            value=100.0,
            datetime=datetime(2019, 1, 1),  # 1 year before reference
        )
        weight = obs.calculate_weight(
            WeightStrategy.TEMPORAL_DECAY,
            decay_factor=0.9,
            reference_date=reference,
        )
        assert weight == pytest.approx(0.9)

    def test_to_pest_line(self):
        """Test PEST line formatting."""
        obs = IWFMObservation(
            name="obs1",
            value=100.0,
            weight=2.0,
            group="head",
        )
        line = obs.to_pest_line()
        assert "obs1" in line
        assert "head" in line

    def test_to_dict(self):
        """Test conversion to dictionary."""
        obs = IWFMObservation(
            name="obs1",
            value=100.0,
            group="head",
            obs_type=IWFMObservationType.HEAD,
        )
        d = obs.to_dict()
        assert d["name"] == "obs1"
        assert d["value"] == 100.0
        assert d["obs_type"] == "head"

    def test_repr(self):
        """Test string representation."""
        obs = IWFMObservation(
            name="obs1",
            value=100.0,
            obs_type=IWFMObservationType.HEAD,
        )
        r = repr(obs)
        assert "obs1" in r
        assert "head" in r


class TestIWFMObservationGroup:
    """Tests for IWFMObservationGroup class."""

    def test_basic_creation(self):
        """Test basic group creation."""
        grp = IWFMObservationGroup(name="head")
        assert grp.name == "head"
        assert grp.n_observations == 0

    def test_with_obs_type(self):
        """Test group with observation type."""
        grp = IWFMObservationGroup(
            name="head",
            obs_type=IWFMObservationType.HEAD,
        )
        assert grp.obs_type == IWFMObservationType.HEAD

    def test_add_observation(self):
        """Test adding observations to group."""
        grp = IWFMObservationGroup(name="head")
        obs = grp.add_observation("obs1", 100.0, weight=2.0)
        assert grp.n_observations == 1
        assert obs.group == "head"

    def test_values_property(self):
        """Test values property."""
        grp = IWFMObservationGroup(name="head")
        grp.add_observation("obs1", 100.0)
        grp.add_observation("obs2", 200.0)
        values = grp.values
        assert len(values) == 2
        assert values[0] == 100.0
        assert values[1] == 200.0

    def test_weights_property(self):
        """Test weights property."""
        grp = IWFMObservationGroup(name="head")
        grp.add_observation("obs1", 100.0, weight=1.0)
        grp.add_observation("obs2", 200.0, weight=2.0)
        weights = grp.weights
        assert len(weights) == 2
        assert weights[0] == 1.0
        assert weights[1] == 2.0

    def test_set_weights(self):
        """Test setting weights for all observations."""
        grp = IWFMObservationGroup(name="head")
        grp.add_observation("obs1", 100.0)
        grp.add_observation("obs2", 200.0)
        grp.set_weights(WeightStrategy.EQUAL)
        assert all(obs.weight == 1.0 for obs in grp.observations)

    def test_scale_weights(self):
        """Test scaling weights."""
        grp = IWFMObservationGroup(name="head")
        grp.add_observation("obs1", 100.0, weight=1.0)
        grp.add_observation("obs2", 200.0, weight=2.0)
        grp.scale_weights(0.5)
        assert grp.observations[0].weight == 0.5
        assert grp.observations[1].weight == 1.0

    def test_normalize_weights(self):
        """Test normalizing weights."""
        grp = IWFMObservationGroup(name="head")
        grp.add_observation("obs1", 100.0, weight=2.0)
        grp.add_observation("obs2", 200.0, weight=2.0)
        grp.normalize_weights(target_sum=1.0)
        assert np.sum(grp.weights) == pytest.approx(1.0)

    def test_get_observations_by_time(self):
        """Test filtering observations by time."""
        grp = IWFMObservationGroup(name="head")
        grp.add_observation("obs1", 100.0, datetime=datetime(2020, 1, 1))
        grp.add_observation("obs2", 200.0, datetime=datetime(2020, 6, 1))
        grp.add_observation("obs3", 300.0, datetime=datetime(2020, 12, 1))

        filtered = grp.get_observations_by_time(
            start_date=datetime(2020, 3, 1),
            end_date=datetime(2020, 9, 1),
        )
        assert len(filtered) == 1
        assert filtered[0].name == "obs2"

    def test_summary(self):
        """Test group summary."""
        grp = IWFMObservationGroup(
            name="head",
            obs_type=IWFMObservationType.HEAD,
        )
        grp.add_observation("obs1", 100.0)
        grp.add_observation("obs2", 200.0)
        summary = grp.summary()
        assert summary["name"] == "head"
        assert summary["n_observations"] == 2
        assert summary["value_mean"] == 150.0

    def test_iteration(self):
        """Test iterating over group."""
        grp = IWFMObservationGroup(name="head")
        grp.add_observation("obs1", 100.0)
        grp.add_observation("obs2", 200.0)
        names = [obs.name for obs in grp]
        assert names == ["obs1", "obs2"]

    def test_len(self):
        """Test len of group."""
        grp = IWFMObservationGroup(name="head")
        grp.add_observation("obs1", 100.0)
        grp.add_observation("obs2", 200.0)
        assert len(grp) == 2

    def test_repr(self):
        """Test string representation."""
        grp = IWFMObservationGroup(
            name="head",
            obs_type=IWFMObservationType.HEAD,
        )
        grp.add_observation("obs1", 100.0)
        r = repr(grp)
        assert "head" in r
        assert "n_obs=1" in r


class TestDerivedObservation:
    """Tests for DerivedObservation class."""

    def test_basic_creation(self):
        """Test basic derived observation creation."""
        derived = DerivedObservation(
            name="mass_balance",
            expression="inflow - outflow",
            source_observations=["inflow", "outflow"],
        )
        assert derived.name == "mass_balance"
        assert derived.target_value == 0.0

    def test_with_target_and_weight(self):
        """Test derived observation with target and weight."""
        derived = DerivedObservation(
            name="mass_balance",
            expression="inflow - outflow",
            source_observations=["inflow", "outflow"],
            target_value=0.0,
            weight=10.0,
        )
        assert derived.target_value == 0.0
        assert derived.weight == 10.0

    def test_evaluate_simple(self):
        """Test simple expression evaluation."""
        derived = DerivedObservation(
            name="diff",
            expression="a - b",
            source_observations=["a", "b"],
        )
        result = derived.evaluate({"a": 100.0, "b": 30.0})
        assert result == 70.0

    def test_evaluate_complex(self):
        """Test complex expression evaluation."""
        derived = DerivedObservation(
            name="result",
            expression="(a + b) / c",
            source_observations=["a", "b", "c"],
        )
        result = derived.evaluate({"a": 10.0, "b": 20.0, "c": 3.0})
        assert result == pytest.approx(10.0)

    def test_evaluate_with_functions(self):
        """Test expression with math functions."""
        derived = DerivedObservation(
            name="result",
            expression="sqrt(a) + abs(b)",
            source_observations=["a", "b"],
        )
        result = derived.evaluate({"a": 100.0, "b": -5.0})
        assert result == pytest.approx(15.0)

    def test_evaluate_missing_observation(self):
        """Test evaluation with missing observation."""
        derived = DerivedObservation(
            name="diff",
            expression="a - b",
            source_observations=["a", "b"],
        )
        with pytest.raises(ValueError, match="Missing observations"):
            derived.evaluate({"a": 100.0})

    def test_to_prior_equation(self):
        """Test PEST prior equation formatting."""
        derived = DerivedObservation(
            name="balance",
            expression="inflow - outflow",
            source_observations=["inflow", "outflow"],
            target_value=0.0,
            weight=1.0,
            group="prior",
        )
        eq = derived.to_prior_equation()
        assert "balance" in eq
        assert "@inflow" in eq
        assert "@outflow" in eq

    def test_repr(self):
        """Test string representation."""
        derived = DerivedObservation(
            name="balance",
            expression="a - b",
            source_observations=["a", "b"],
        )
        r = repr(derived)
        assert "balance" in r
        assert "a - b" in r


# ── Additional tests for increased coverage ──────────────────────────


class TestIWFMObservationTypeExtended:
    """Extended tests for IWFMObservationType enum covering missed branches."""

    def test_default_transform_sqrt(self):
        """Test sqrt transform for STREAM_GAIN_LOSS."""
        assert IWFMObservationType.STREAM_GAIN_LOSS.default_transform == "sqrt"

    def test_default_transform_none_for_all_non_log_non_sqrt(self):
        """Test that non-flow, non-gain types return 'none' transform."""
        none_types = [
            IWFMObservationType.DRAWDOWN,
            IWFMObservationType.HEAD_DIFFERENCE,
            IWFMObservationType.VERTICAL_GRADIENT,
            IWFMObservationType.STREAM_STAGE,
            IWFMObservationType.LAKE_LEVEL,
            IWFMObservationType.GW_BUDGET,
            IWFMObservationType.STREAM_BUDGET,
            IWFMObservationType.ROOTZONE_BUDGET,
            IWFMObservationType.LAKE_BUDGET,
            IWFMObservationType.SUBSIDENCE,
            IWFMObservationType.COMPACTION,
        ]
        for obs_type in none_types:
            assert obs_type.default_transform == "none", f"{obs_type} should be 'none'"

    def test_typical_error_all_types(self):
        """Test typical_error returns correct values for all types."""
        assert IWFMObservationType.HEAD_DIFFERENCE.typical_error == 0.5
        assert IWFMObservationType.VERTICAL_GRADIENT.typical_error == 0.01
        assert IWFMObservationType.STREAM_STAGE.typical_error == 0.1
        assert IWFMObservationType.STREAM_GAIN_LOSS.typical_error == 0.2
        assert IWFMObservationType.LAKE_LEVEL.typical_error == 0.5
        assert IWFMObservationType.LAKE_STORAGE.typical_error == 0.1
        assert IWFMObservationType.GW_BUDGET.typical_error == 0.1
        assert IWFMObservationType.STREAM_BUDGET.typical_error == 0.1
        assert IWFMObservationType.ROOTZONE_BUDGET.typical_error == 0.15
        assert IWFMObservationType.LAKE_BUDGET.typical_error == 0.1
        assert IWFMObservationType.SUBSIDENCE.typical_error == 0.01
        assert IWFMObservationType.COMPACTION.typical_error == 0.005

    def test_is_relative_error_all_relative_types(self):
        """Test is_relative_error for all relative types."""
        relative_types = [
            IWFMObservationType.STREAM_FLOW,
            IWFMObservationType.STREAM_GAIN_LOSS,
            IWFMObservationType.LAKE_STORAGE,
            IWFMObservationType.GW_BUDGET,
            IWFMObservationType.STREAM_BUDGET,
            IWFMObservationType.ROOTZONE_BUDGET,
            IWFMObservationType.LAKE_BUDGET,
        ]
        for obs_type in relative_types:
            assert obs_type.is_relative_error is True, f"{obs_type} should be relative"

    def test_is_relative_error_absolute_types(self):
        """Test is_relative_error is False for absolute error types."""
        absolute_types = [
            IWFMObservationType.HEAD,
            IWFMObservationType.DRAWDOWN,
            IWFMObservationType.HEAD_DIFFERENCE,
            IWFMObservationType.VERTICAL_GRADIENT,
            IWFMObservationType.STREAM_STAGE,
            IWFMObservationType.LAKE_LEVEL,
            IWFMObservationType.SUBSIDENCE,
            IWFMObservationType.COMPACTION,
        ]
        for obs_type in absolute_types:
            assert obs_type.is_relative_error is False, f"{obs_type} should not be relative"

    def test_subsidence_types_exist(self):
        """Test subsidence observation types exist."""
        assert IWFMObservationType.SUBSIDENCE.value == "sub"
        assert IWFMObservationType.COMPACTION.value == "comp"

    def test_lake_budget_type_exists(self):
        """Test lake budget type exists."""
        assert IWFMObservationType.LAKE_BUDGET.value == "lakbud"


class TestObservationLocationExtended:
    """Extended tests for ObservationLocation."""

    def test_to_dict_with_lake_id(self):
        """Test to_dict includes lake_id when present."""
        loc = ObservationLocation(x=10.0, y=20.0, lake_id=5)
        d = loc.to_dict()
        assert d["lake_id"] == 5

    def test_to_dict_excludes_all_none_optional(self):
        """Test to_dict excludes all None optional fields."""
        loc = ObservationLocation(x=1.0, y=2.0)
        d = loc.to_dict()
        assert set(d.keys()) == {"x", "y"}

    def test_to_dict_with_all_fields(self):
        """Test to_dict includes all non-None fields."""
        loc = ObservationLocation(
            x=1.0,
            y=2.0,
            z=-10.0,
            node_id=1,
            element_id=2,
            layer=3,
            reach_id=4,
            lake_id=5,
        )
        d = loc.to_dict()
        assert len(d) == 8


class TestIWFMObservationExtended:
    """Extended tests for IWFMObservation covering missed branches."""

    def test_calculate_weight_magnitude_based_relative(self):
        """Test MAGNITUDE_BASED weight with relative error obs_type."""
        obs = IWFMObservation(
            name="flow_obs",
            value=1000.0,
            obs_type=IWFMObservationType.STREAM_FLOW,
        )
        weight = obs.calculate_weight(WeightStrategy.MAGNITUDE_BASED)
        # relative error = 0.1, so weight = 1 / (1000 * 0.1) = 0.01
        assert weight == pytest.approx(0.01)

    def test_calculate_weight_magnitude_based_non_relative(self):
        """Test MAGNITUDE_BASED weight with non-relative obs_type (falls through)."""
        obs = IWFMObservation(
            name="head_obs",
            value=100.0,
            obs_type=IWFMObservationType.HEAD,
        )
        weight = obs.calculate_weight(WeightStrategy.MAGNITUDE_BASED)
        # HEAD is not relative, so falls to default: 1 / (|value| * 0.1)
        assert weight == pytest.approx(1.0 / (100.0 * 0.1))

    def test_calculate_weight_magnitude_based_no_obs_type(self):
        """Test MAGNITUDE_BASED weight with no obs_type."""
        obs = IWFMObservation(name="obs", value=50.0)
        weight = obs.calculate_weight(WeightStrategy.MAGNITUDE_BASED)
        assert weight == pytest.approx(1.0 / (50.0 * 0.1))

    def test_calculate_weight_custom_returns_existing_weight(self):
        """Test CUSTOM strategy returns current weight."""
        obs = IWFMObservation(name="obs", value=100.0, weight=5.0)
        weight = obs.calculate_weight(WeightStrategy.CUSTOM)
        assert weight == 5.0

    def test_calculate_weight_group_contribution_returns_existing_weight(self):
        """Test GROUP_CONTRIBUTION strategy returns current weight (fallthrough)."""
        obs = IWFMObservation(name="obs", value=100.0, weight=3.0)
        weight = obs.calculate_weight(WeightStrategy.GROUP_CONTRIBUTION)
        assert weight == 3.0

    def test_calculate_weight_inverse_variance_with_relative_obs_type(self):
        """Test inverse variance with relative error obs_type."""
        obs = IWFMObservation(
            name="flow_obs",
            value=500.0,
            obs_type=IWFMObservationType.STREAM_FLOW,
        )
        weight = obs.calculate_weight(WeightStrategy.INVERSE_VARIANCE)
        # relative error = 0.1, abs error = 500 * 0.1 = 50, weight = 1/50
        assert weight == pytest.approx(1.0 / 50.0)

    def test_calculate_weight_inverse_variance_no_obs_type_no_error(self):
        """Test inverse variance with no obs_type and no error_std."""
        obs = IWFMObservation(name="obs", value=100.0)
        weight = obs.calculate_weight(WeightStrategy.INVERSE_VARIANCE)
        assert weight == 1.0

    def test_calculate_weight_temporal_decay_no_datetime(self):
        """Test temporal decay returns 1.0 when datetime is None."""
        obs = IWFMObservation(name="obs", value=100.0)
        weight = obs.calculate_weight(
            WeightStrategy.TEMPORAL_DECAY,
            reference_date=datetime(2020, 1, 1),
        )
        assert weight == 1.0

    def test_calculate_weight_temporal_decay_no_reference_date(self):
        """Test temporal decay returns 1.0 when reference_date missing."""
        obs = IWFMObservation(
            name="obs",
            value=100.0,
            datetime=datetime(2019, 1, 1),
        )
        weight = obs.calculate_weight(WeightStrategy.TEMPORAL_DECAY)
        assert weight == 1.0

    def test_to_dict_with_all_optional_fields(self):
        """Test to_dict includes all optional fields when populated."""
        loc = ObservationLocation(x=1.0, y=2.0)
        obs = IWFMObservation(
            name="full_obs",
            value=99.0,
            weight=2.0,
            group="heads",
            obs_type=IWFMObservationType.HEAD,
            datetime=datetime(2020, 6, 15),
            location=loc,
            simulated_name="sim_head_1",
            error_std=0.5,
            transform="none",
            metadata={"source": "field"},
        )
        d = obs.to_dict()
        assert d["name"] == "full_obs"
        assert d["obs_type"] == "head"
        assert "datetime" in d
        assert "location" in d
        assert d["simulated_name"] == "sim_head_1"
        assert d["error_std"] == 0.5
        assert d["metadata"] == {"source": "field"}

    def test_to_dict_without_optional_fields(self):
        """Test to_dict excludes None optional fields."""
        obs = IWFMObservation(name="minimal", value=10.0)
        d = obs.to_dict()
        assert "obs_type" not in d
        assert "datetime" not in d
        assert "location" not in d
        assert "simulated_name" not in d
        assert "error_std" not in d
        assert "metadata" not in d

    def test_repr_without_obs_type(self):
        """Test repr when obs_type is None."""
        obs = IWFMObservation(name="obs1", value=42.0)
        r = repr(obs)
        assert "obs1" in r
        assert "type=" not in r

    def test_log_transform_zero_value_raises(self):
        """Test that log transform with zero value raises ValueError."""
        with pytest.raises(ValueError, match="positive value"):
            IWFMObservation(name="obs1", value=0.0, transform="log")


class TestIWFMObservationGroupExtended:
    """Extended tests for IWFMObservationGroup."""

    def test_post_init_long_name_raises(self):
        """Test that group name over 200 chars raises ValueError."""
        with pytest.raises(ValueError, match="too long"):
            IWFMObservationGroup(name="x" * 201)

    def test_summary_empty_group(self):
        """Test summary with empty observations."""
        grp = IWFMObservationGroup(name="empty")
        summary = grp.summary()
        assert summary["n_observations"] == 0
        assert summary["value_mean"] is None
        assert summary["value_std"] is None
        assert summary["value_min"] is None
        assert summary["value_max"] is None
        assert summary["weight_mean"] is None
        assert summary["weight_sum"] is None

    def test_summary_without_obs_type(self):
        """Test summary when obs_type is None."""
        grp = IWFMObservationGroup(name="generic")
        grp.add_observation("obs1", 10.0)
        summary = grp.summary()
        assert summary["obs_type"] is None

    def test_repr_without_obs_type(self):
        """Test repr when obs_type is None."""
        grp = IWFMObservationGroup(name="test_grp")
        r = repr(grp)
        assert "test_grp" in r
        assert "type=" not in r

    def test_contribution_property(self):
        """Test contribution calculation (sum of squared weights)."""
        grp = IWFMObservationGroup(name="test")
        grp.add_observation("obs1", 100.0, weight=2.0)
        grp.add_observation("obs2", 200.0, weight=3.0)
        assert grp.contribution == pytest.approx(4.0 + 9.0)

    def test_normalize_weights_zero_sum(self):
        """Test normalize_weights when current sum is zero."""
        grp = IWFMObservationGroup(name="test")
        grp.add_observation("obs1", 100.0, weight=0.0)
        grp.add_observation("obs2", 200.0, weight=0.0)
        grp.normalize_weights(target_sum=1.0)
        # Weights should remain 0 since current_sum is 0
        assert grp.observations[0].weight == 0.0
        assert grp.observations[1].weight == 0.0

    def test_get_observations_by_time_no_datetime(self):
        """Test get_observations_by_time skips obs with no datetime."""
        grp = IWFMObservationGroup(name="test")
        grp.add_observation("obs1", 100.0)  # No datetime
        grp.add_observation("obs2", 200.0, datetime=datetime(2020, 6, 1))

        filtered = grp.get_observations_by_time(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
        )
        assert len(filtered) == 1
        assert filtered[0].name == "obs2"

    def test_get_observations_by_time_no_bounds(self):
        """Test get_observations_by_time with no start/end date."""
        grp = IWFMObservationGroup(name="test")
        grp.add_observation("obs1", 100.0, datetime=datetime(2020, 1, 1))
        grp.add_observation("obs2", 200.0, datetime=datetime(2021, 1, 1))

        filtered = grp.get_observations_by_time()
        assert len(filtered) == 2

    def test_add_observation_inherits_group_type(self):
        """Test that add_observation sets the obs_type from the group."""
        grp = IWFMObservationGroup(
            name="heads",
            obs_type=IWFMObservationType.HEAD,
        )
        obs = grp.add_observation("h1", 50.0)
        assert obs.obs_type == IWFMObservationType.HEAD
        assert obs.group == "heads"


class TestDerivedObservationExtended:
    """Extended tests for DerivedObservation."""

    def test_evaluate_expression_error(self):
        """Test evaluate raises ValueError on bad expression."""
        derived = DerivedObservation(
            name="bad",
            expression="a / 0",
            source_observations=["a"],
        )
        # Division by zero should be caught as an error in eval
        with pytest.raises(ValueError, match="Error evaluating expression"):
            derived.evaluate({"a": 1.0})

    def test_evaluate_with_log_function(self):
        """Test evaluate with log function."""
        derived = DerivedObservation(
            name="log_test",
            expression="log10(a)",
            source_observations=["a"],
        )
        result = derived.evaluate({"a": 100.0})
        assert result == pytest.approx(np.log10(100.0))

    def test_evaluate_with_exp_function(self):
        """Test evaluate with exp function."""
        derived = DerivedObservation(
            name="exp_test",
            expression="exp(a)",
            source_observations=["a"],
        )
        result = derived.evaluate({"a": 0.0})
        assert result == pytest.approx(1.0)

    def test_to_prior_equation_format(self):
        """Test PEST prior equation format completeness."""
        derived = DerivedObservation(
            name="bal",
            expression="inflow - outflow",
            source_observations=["inflow", "outflow"],
            target_value=0.0,
            weight=10.0,
            group="constraints",
        )
        eq = derived.to_prior_equation()
        assert "bal" in eq
        assert "@inflow" in eq
        assert "@outflow" in eq
        assert "0.0" in eq
        assert "10.0" in eq
        assert "constraints" in eq
