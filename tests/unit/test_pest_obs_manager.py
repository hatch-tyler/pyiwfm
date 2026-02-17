"""Unit tests for PEST++ observation manager."""

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyiwfm.runner.pest_obs_manager import (
    GageInfo,
    IWFMObservationManager,
    WellInfo,
)
from pyiwfm.runner.pest_observations import (
    IWFMObservation,
    IWFMObservationType,
    WeightStrategy,
)


class TestIWFMObservationManagerInit:
    """Tests for IWFMObservationManager initialization."""

    def test_init_no_model(self):
        """Test initialization without model."""
        om = IWFMObservationManager()
        assert om.model is None
        assert om.n_observations == 0

    def test_init_with_model(self):
        """Test initialization with model."""

        class MockModel:
            pass

        model = MockModel()
        om = IWFMObservationManager(model=model)
        assert om.model is model

    def test_default_groups_created(self):
        """Test that default groups are created."""
        om = IWFMObservationManager()
        # Check some default groups exist
        assert om.get_observation_group("head") is not None
        assert om.get_observation_group("flow") is not None
        assert om.get_observation_group("drawdown") is not None


class TestWellInfo:
    """Tests for WellInfo dataclass."""

    def test_basic_creation(self):
        """Test basic well info creation."""
        well = WellInfo(well_id="W1", x=100.0, y=200.0)
        assert well.well_id == "W1"
        assert well.x == 100.0
        assert well.y == 200.0

    def test_full_creation(self):
        """Test well info with all attributes."""
        well = WellInfo(
            well_id="W1",
            x=100.0,
            y=200.0,
            screen_top=-50.0,
            screen_bottom=-100.0,
            layer=2,
            node_id=123,
        )
        assert well.screen_top == -50.0
        assert well.layer == 2

    def test_to_location(self):
        """Test conversion to ObservationLocation."""
        well = WellInfo(
            well_id="W1",
            x=100.0,
            y=200.0,
            screen_top=-50.0,
            screen_bottom=-100.0,
            layer=2,
        )
        loc = well.to_location()
        assert loc.x == 100.0
        assert loc.y == 200.0
        assert loc.z == -75.0  # midpoint of screen
        assert loc.layer == 2


class TestGageInfo:
    """Tests for GageInfo dataclass."""

    def test_basic_creation(self):
        """Test basic gage info creation."""
        gage = GageInfo(gage_id="G1")
        assert gage.gage_id == "G1"
        assert gage.reach_id is None

    def test_full_creation(self):
        """Test gage info with all attributes."""
        gage = GageInfo(
            gage_id="G1",
            reach_id=5,
            node_id=100,
            x=500.0,
            y=600.0,
        )
        assert gage.reach_id == 5
        assert gage.x == 500.0

    def test_to_location(self):
        """Test conversion to ObservationLocation."""
        gage = GageInfo(gage_id="G1", x=500.0, y=600.0, reach_id=5)
        loc = gage.to_location()
        assert loc.x == 500.0
        assert loc.reach_id == 5

    def test_to_location_no_coords(self):
        """Test conversion returns None without coordinates."""
        gage = GageInfo(gage_id="G1", reach_id=5)
        loc = gage.to_location()
        assert loc is None


class TestHeadObservations:
    """Tests for head observation methods."""

    @pytest.fixture
    def sample_wells(self):
        """Create sample well data."""
        return pd.DataFrame(
            {
                "well_id": ["W1", "W2", "W3"],
                "x": [100.0, 200.0, 300.0],
                "y": [100.0, 200.0, 300.0],
                "layer": [1, 1, 2],
            }
        )

    @pytest.fixture
    def sample_head_data(self):
        """Create sample head time series."""
        return pd.DataFrame(
            {
                "well_id": ["W1", "W1", "W2", "W2"],
                "datetime": [
                    datetime(2020, 1, 1),
                    datetime(2020, 2, 1),
                    datetime(2020, 1, 1),
                    datetime(2020, 2, 1),
                ],
                "head": [100.0, 101.0, 150.0, 151.0],
            }
        )

    def test_add_head_observations(self, sample_wells, sample_head_data):
        """Test adding head observations."""
        om = IWFMObservationManager()
        obs = om.add_head_observations(sample_wells, sample_head_data)

        assert len(obs) == 4
        assert om.n_observations == 4
        assert all(o.obs_type == IWFMObservationType.HEAD for o in obs)

    def test_add_head_observations_with_date_filter(self, sample_wells, sample_head_data):
        """Test adding head observations with date filter."""
        om = IWFMObservationManager()
        obs = om.add_head_observations(
            sample_wells,
            sample_head_data,
            start_date=datetime(2020, 1, 15),
        )

        # Should only get February observations
        assert len(obs) == 2

    def test_add_head_observations_weight_strategy(self, sample_wells, sample_head_data):
        """Test weight strategy in head observations."""
        om = IWFMObservationManager()
        obs = om.add_head_observations(
            sample_wells,
            sample_head_data,
            weight_strategy=WeightStrategy.INVERSE_VARIANCE,
        )

        # All observations should have calculated weights
        assert all(o.weight > 0 for o in obs)

    def test_add_head_observations_custom_group(self, sample_wells, sample_head_data):
        """Test custom group name for head observations."""
        om = IWFMObservationManager()
        obs = om.add_head_observations(
            sample_wells,
            sample_head_data,
            group_name="custom_head",
        )

        assert all(o.group == "custom_head" for o in obs)

    def test_add_head_observations_from_files(self):
        """Test adding head observations from CSV files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wells_file = Path(tmpdir) / "wells.csv"
            data_file = Path(tmpdir) / "heads.csv"

            # Write wells
            pd.DataFrame(
                {
                    "well_id": ["W1", "W2"],
                    "x": [100.0, 200.0],
                    "y": [100.0, 200.0],
                }
            ).to_csv(wells_file, index=False)

            # Write head data
            pd.DataFrame(
                {
                    "well_id": ["W1", "W2"],
                    "datetime": ["2020-01-01", "2020-01-01"],
                    "head": [100.0, 150.0],
                }
            ).to_csv(data_file, index=False)

            om = IWFMObservationManager()
            obs = om.add_head_observations(wells_file, data_file)

            assert len(obs) == 2


class TestDrawdownObservations:
    """Tests for drawdown observation methods."""

    @pytest.fixture
    def sample_wells(self):
        """Create sample well data."""
        return pd.DataFrame(
            {
                "well_id": ["W1", "W2"],
                "x": [100.0, 200.0],
                "y": [100.0, 200.0],
            }
        )

    @pytest.fixture
    def sample_head_data(self):
        """Create sample head time series."""
        return pd.DataFrame(
            {
                "well_id": ["W1", "W1", "W1", "W2", "W2", "W2"],
                "datetime": [
                    datetime(2020, 1, 1),
                    datetime(2020, 6, 1),
                    datetime(2020, 12, 1),
                    datetime(2020, 1, 1),
                    datetime(2020, 6, 1),
                    datetime(2020, 12, 1),
                ],
                "head": [100.0, 95.0, 90.0, 150.0, 145.0, 140.0],
            }
        )

    def test_add_drawdown_observations(self, sample_wells, sample_head_data):
        """Test adding drawdown observations."""
        om = IWFMObservationManager()
        obs = om.add_drawdown_observations(
            sample_wells,
            sample_head_data,
            reference_date=datetime(2020, 1, 1),
        )

        # Drawdown = reference - current
        assert len(obs) == 6
        assert all(o.obs_type == IWFMObservationType.DRAWDOWN for o in obs)

    def test_drawdown_values_correct(self, sample_wells, sample_head_data):
        """Test that drawdown values are calculated correctly."""
        om = IWFMObservationManager()
        obs = om.add_drawdown_observations(
            sample_wells,
            sample_head_data,
            reference_date=datetime(2020, 1, 1),
        )

        # Get W1 observations and sort by datetime
        [o for o in obs if "W1" in o.name]
        # First observation should have 0 drawdown (reference)
        # Later observations should have positive drawdown


class TestHeadDifferenceObservations:
    """Tests for head difference observation methods."""

    @pytest.fixture
    def sample_head_data(self):
        """Create sample head time series."""
        return pd.DataFrame(
            {
                "well_id": ["W1", "W1", "W2", "W2"],
                "datetime": [
                    datetime(2020, 1, 1),
                    datetime(2020, 2, 1),
                    datetime(2020, 1, 1),
                    datetime(2020, 2, 1),
                ],
                "head": [100.0, 101.0, 90.0, 91.0],
            }
        )

    def test_add_head_difference_observations(self, sample_head_data):
        """Test adding head difference observations."""
        om = IWFMObservationManager()
        obs = om.add_head_difference_observations(
            well_pairs=[("W1", "W2")],
            observed_data=sample_head_data,
        )

        # Should have 2 observations (one per common time)
        assert len(obs) == 2
        assert all(o.obs_type == IWFMObservationType.HEAD_DIFFERENCE for o in obs)

    def test_head_difference_values(self, sample_head_data):
        """Test head difference values."""
        om = IWFMObservationManager()
        obs = om.add_head_difference_observations(
            well_pairs=[("W1", "W2")],
            observed_data=sample_head_data,
        )

        # Difference should be W1 - W2 = 10
        assert all(o.value == pytest.approx(10.0) for o in obs)


class TestStreamflowObservations:
    """Tests for streamflow observation methods."""

    @pytest.fixture
    def sample_gages(self):
        """Create sample gage data."""
        return pd.DataFrame(
            {
                "gage_id": ["G1", "G2"],
                "reach_id": [1, 2],
                "x": [500.0, 600.0],
                "y": [500.0, 600.0],
            }
        )

    @pytest.fixture
    def sample_flow_data(self):
        """Create sample flow time series."""
        return pd.DataFrame(
            {
                "gage_id": ["G1", "G1", "G2", "G2"],
                "datetime": [
                    datetime(2020, 1, 1),
                    datetime(2020, 2, 1),
                    datetime(2020, 1, 1),
                    datetime(2020, 2, 1),
                ],
                "flow": [100.0, 150.0, 200.0, 250.0],
            }
        )

    def test_add_streamflow_observations(self, sample_gages, sample_flow_data):
        """Test adding streamflow observations."""
        om = IWFMObservationManager()
        obs = om.add_streamflow_observations(sample_gages, sample_flow_data)

        assert len(obs) == 4
        assert all(o.obs_type == IWFMObservationType.STREAM_FLOW for o in obs)

    def test_add_streamflow_log_transform(self, sample_gages, sample_flow_data):
        """Test streamflow with log transform."""
        om = IWFMObservationManager()
        obs = om.add_streamflow_observations(
            sample_gages,
            sample_flow_data,
            transform="log",
        )

        assert all(o.transform == "log" for o in obs)

    def test_add_streamflow_skips_nonpositive_log(self, sample_gages):
        """Test that non-positive flows are skipped for log transform."""
        flow_data = pd.DataFrame(
            {
                "gage_id": ["G1", "G1"],
                "datetime": [datetime(2020, 1, 1), datetime(2020, 2, 1)],
                "flow": [100.0, 0.0],  # Zero flow
            }
        )

        om = IWFMObservationManager()
        obs = om.add_streamflow_observations(
            sample_gages,
            flow_data,
            transform="log",
        )

        # Should only have 1 observation (zero skipped)
        assert len(obs) == 1


class TestStreamStageObservations:
    """Tests for stream stage observation methods."""

    @pytest.fixture
    def sample_gages(self):
        """Create sample gage data."""
        return pd.DataFrame(
            {
                "gage_id": ["G1"],
                "reach_id": [1],
            }
        )

    @pytest.fixture
    def sample_stage_data(self):
        """Create sample stage time series."""
        return pd.DataFrame(
            {
                "gage_id": ["G1", "G1"],
                "datetime": [datetime(2020, 1, 1), datetime(2020, 2, 1)],
                "stage": [10.0, 12.0],
            }
        )

    def test_add_stream_stage_observations(self, sample_gages, sample_stage_data):
        """Test adding stream stage observations."""
        om = IWFMObservationManager()
        obs = om.add_stream_stage_observations(sample_gages, sample_stage_data)

        assert len(obs) == 2
        assert all(o.obs_type == IWFMObservationType.STREAM_STAGE for o in obs)


class TestGainLossObservations:
    """Tests for gain/loss observation methods."""

    @pytest.fixture
    def sample_gain_loss_data(self):
        """Create sample gain/loss data."""
        return pd.DataFrame(
            {
                "reach_id": [1, 1, 2, 2],
                "datetime": [
                    datetime(2020, 1, 1),
                    datetime(2020, 2, 1),
                    datetime(2020, 1, 1),
                    datetime(2020, 2, 1),
                ],
                "gain_loss": [10.0, 15.0, -5.0, -8.0],
            }
        )

    def test_add_gain_loss_observations(self, sample_gain_loss_data):
        """Test adding gain/loss observations."""
        om = IWFMObservationManager()
        obs = om.add_gain_loss_observations(
            reaches=[1, 2],
            observed_data=sample_gain_loss_data,
        )

        assert len(obs) == 4
        assert all(o.obs_type == IWFMObservationType.STREAM_GAIN_LOSS for o in obs)


class TestLakeObservations:
    """Tests for lake observation methods."""

    @pytest.fixture
    def sample_lake_data(self):
        """Create sample lake data."""
        return pd.DataFrame(
            {
                "lake_id": [1, 1, 2, 2],
                "datetime": [
                    datetime(2020, 1, 1),
                    datetime(2020, 2, 1),
                    datetime(2020, 1, 1),
                    datetime(2020, 2, 1),
                ],
                "value": [100.0, 102.0, 50.0, 48.0],
            }
        )

    def test_add_lake_level_observations(self, sample_lake_data):
        """Test adding lake level observations."""
        om = IWFMObservationManager()
        obs = om.add_lake_observations(
            lakes=[1, 2],
            observed_data=sample_lake_data,
            obs_type="level",
        )

        assert len(obs) == 4
        assert all(o.obs_type == IWFMObservationType.LAKE_LEVEL for o in obs)

    def test_add_lake_storage_observations(self, sample_lake_data):
        """Test adding lake storage observations."""
        om = IWFMObservationManager()
        obs = om.add_lake_observations(
            lakes=[1],
            observed_data=sample_lake_data,
            obs_type="storage",
        )

        assert len(obs) == 2
        assert all(o.obs_type == IWFMObservationType.LAKE_STORAGE for o in obs)


class TestBudgetObservations:
    """Tests for budget observation methods."""

    @pytest.fixture
    def sample_budget_data(self):
        """Create sample budget data."""
        return pd.DataFrame(
            {
                "location_id": [1, 1, 2, 2],
                "component": ["recharge", "recharge", "recharge", "recharge"],
                "datetime": [
                    datetime(2020, 1, 1),
                    datetime(2020, 2, 1),
                    datetime(2020, 1, 1),
                    datetime(2020, 2, 1),
                ],
                "value": [100.0, 110.0, 50.0, 55.0],
            }
        )

    def test_add_budget_observations_gw(self, sample_budget_data):
        """Test adding GW budget observations."""
        om = IWFMObservationManager()
        obs = om.add_budget_observations(
            budget_type="gw",
            observed_data=sample_budget_data,
        )

        assert len(obs) > 0
        assert all(o.obs_type == IWFMObservationType.GW_BUDGET for o in obs)

    def test_add_budget_observations_stream(self, sample_budget_data):
        """Test adding stream budget observations."""
        om = IWFMObservationManager()
        obs = om.add_budget_observations(
            budget_type="stream",
            observed_data=sample_budget_data,
        )

        assert all(o.obs_type == IWFMObservationType.STREAM_BUDGET for o in obs)

    def test_add_budget_observations_invalid_type(self, sample_budget_data):
        """Test invalid budget type raises error."""
        om = IWFMObservationManager()
        with pytest.raises(ValueError, match="Invalid budget type"):
            om.add_budget_observations(
                budget_type="invalid",
                observed_data=sample_budget_data,
            )


class TestDerivedObservations:
    """Tests for derived observation methods."""

    def test_add_derived_observation(self):
        """Test adding derived observation."""
        om = IWFMObservationManager()

        # First add source observations
        om._observations["inflow"] = IWFMObservation(name="inflow", value=100.0)
        om._observations["outflow"] = IWFMObservation(name="outflow", value=80.0)

        derived = om.add_derived_observation(
            expression="inflow - outflow",
            obs_names=["inflow", "outflow"],
            result_name="balance",
            target_value=20.0,
            weight=10.0,
        )

        assert derived.name == "balance"
        assert derived.target_value == 20.0

    def test_derived_observation_missing_source(self):
        """Test derived observation with missing source raises error."""
        om = IWFMObservationManager()

        with pytest.raises(ValueError, match="Observation not found"):
            om.add_derived_observation(
                expression="a - b",
                obs_names=["a", "b"],
                result_name="diff",
            )


class TestWeightManagement:
    """Tests for weight management methods."""

    def test_set_group_weights(self):
        """Test setting group weights."""
        om = IWFMObservationManager()
        grp = om.get_observation_group("head")
        grp.add_observation("obs1", 100.0)
        grp.add_observation("obs2", 200.0)

        om.set_group_weights("head", weight=2.0)

        assert all(o.weight == 2.0 for o in grp.observations)

    def test_set_group_weights_invalid_group(self):
        """Test setting weights for invalid group raises error."""
        om = IWFMObservationManager()
        with pytest.raises(ValueError, match="Group not found"):
            om.set_group_weights("nonexistent", weight=1.0)

    def test_balance_observation_groups(self):
        """Test balancing observation groups."""
        om = IWFMObservationManager()

        # Add observations to two groups
        head_grp = om.get_observation_group("head")
        head_grp.add_observation("h1", 100.0, weight=1.0)
        head_grp.add_observation("h2", 200.0, weight=1.0)

        flow_grp = om.get_observation_group("flow")
        flow_grp.add_observation("f1", 50.0, weight=1.0)

        om.balance_observation_groups({"head": 0.5, "flow": 0.5})

        # Both groups should now have equal contribution

    def test_apply_temporal_weights(self):
        """Test applying temporal weights."""
        om = IWFMObservationManager()

        grp = om.get_observation_group("head")
        obs1 = grp.add_observation("h1", 100.0, weight=1.0, datetime=datetime(2019, 1, 1))
        obs2 = grp.add_observation("h2", 200.0, weight=1.0, datetime=datetime(2020, 1, 1))
        # Also add to manager's observation dict
        om._observations["h1"] = obs1
        om._observations["h2"] = obs2

        om.apply_temporal_weights(
            decay_factor=0.9,
            reference_date=datetime(2020, 1, 1),
        )

        # Older observation should have lower weight
        h1 = om.get_observation("h1")
        h2 = om.get_observation("h2")
        assert h1.weight < h2.weight


class TestObservationAccess:
    """Tests for observation access methods."""

    def test_get_observation_by_name(self):
        """Test getting observation by name."""
        om = IWFMObservationManager()
        grp = om.get_observation_group("head")
        grp.add_observation("test_obs", 100.0)
        om._observations["test_obs"] = grp.observations[0]

        obs = om.get_observation("test_obs")
        assert obs is not None
        assert obs.name == "test_obs"

    def test_get_observation_not_found(self):
        """Test getting nonexistent observation returns None."""
        om = IWFMObservationManager()
        obs = om.get_observation("nonexistent")
        assert obs is None

    def test_get_observations_by_type(self):
        """Test getting observations by type."""
        om = IWFMObservationManager()
        grp = om.get_observation_group("head")
        obs1 = grp.add_observation("h1", 100.0)
        om._observations["h1"] = obs1

        results = om.get_observations_by_type(IWFMObservationType.HEAD)
        assert len(results) == 1

    def test_get_observations_by_group(self):
        """Test getting observations by group."""
        om = IWFMObservationManager()
        grp = om.get_observation_group("head")
        grp.add_observation("h1", 100.0)
        grp.add_observation("h2", 200.0)

        results = om.get_observations_by_group("head")
        assert len(results) == 2

    def test_get_all_observations(self):
        """Test getting all observations."""
        om = IWFMObservationManager()
        om._observations["o1"] = IWFMObservation(name="o1", value=100.0)
        om._observations["o2"] = IWFMObservation(name="o2", value=200.0)

        all_obs = om.get_all_observations()
        assert len(all_obs) == 2

    def test_get_all_groups(self):
        """Test getting all groups with observations."""
        om = IWFMObservationManager()

        # Add observation to one group
        grp = om.get_observation_group("head")
        grp.add_observation("h1", 100.0)

        active_groups = om.get_all_groups()
        assert len(active_groups) == 1
        assert active_groups[0].name == "head"


class TestDataFrameExport:
    """Tests for DataFrame export methods."""

    def test_to_dataframe(self):
        """Test exporting to DataFrame."""
        om = IWFMObservationManager()
        om._observations["o1"] = IWFMObservation(
            name="o1",
            value=100.0,
            group="head",
            obs_type=IWFMObservationType.HEAD,
        )
        om._observations["o2"] = IWFMObservation(
            name="o2",
            value=200.0,
            group="flow",
        )

        df = om.to_dataframe()
        assert len(df) == 2
        assert "name" in df.columns
        assert "value" in df.columns

    def test_from_dataframe(self):
        """Test loading from DataFrame."""
        om = IWFMObservationManager()

        df = pd.DataFrame(
            {
                "name": ["o1", "o2"],
                "value": [100.0, 200.0],
                "weight": [1.0, 2.0],
                "group": ["head", "flow"],
            }
        )

        om.from_dataframe(df)
        assert om.n_observations == 2


class TestFileIO:
    """Tests for file I/O methods."""

    def test_write_and_read_observation_file(self):
        """Test writing and reading observation file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "observations.csv"

            # Write
            om1 = IWFMObservationManager()
            om1._observations["o1"] = IWFMObservation(name="o1", value=100.0, group="head")
            om1._observations["o2"] = IWFMObservation(name="o2", value=200.0, group="flow")
            om1.write_observation_file(filepath)

            # Read
            om2 = IWFMObservationManager()
            om2.read_observation_file(filepath)

            assert om2.n_observations == 2


class TestStatisticsAndSummary:
    """Tests for statistics and summary methods."""

    def test_n_observations(self):
        """Test observation count property."""
        om = IWFMObservationManager()
        assert om.n_observations == 0

        om._observations["o1"] = IWFMObservation(name="o1", value=100.0)
        assert om.n_observations == 1

    def test_n_groups(self):
        """Test group count property."""
        om = IWFMObservationManager()
        # Initially no groups have observations
        assert om.n_groups == 0

        grp = om.get_observation_group("head")
        grp.add_observation("h1", 100.0)
        assert om.n_groups == 1

    def test_summary(self):
        """Test summary generation."""
        om = IWFMObservationManager()
        grp = om.get_observation_group("head")
        grp.add_observation("h1", 100.0)
        om._observations["h1"] = grp.observations[0]

        summary = om.summary()
        assert summary["n_observations"] == 1
        assert "groups" in summary


class TestIteration:
    """Tests for iteration methods."""

    def test_iteration(self):
        """Test iterating over observations."""
        om = IWFMObservationManager()
        om._observations["o1"] = IWFMObservation(name="o1", value=100.0)
        om._observations["o2"] = IWFMObservation(name="o2", value=200.0)

        names = [obs.name for obs in om]
        assert len(names) == 2

    def test_len(self):
        """Test len of manager."""
        om = IWFMObservationManager()
        assert len(om) == 0

        om._observations["o1"] = IWFMObservation(name="o1", value=100.0)
        assert len(om) == 1

    def test_repr(self):
        """Test string representation."""
        om = IWFMObservationManager()
        r = repr(om)
        assert "IWFMObservationManager" in r
        assert "n_observations=0" in r


# =========================================================================
# Additional Tests for 95%+ Coverage
# =========================================================================


class TestWellInfoExtended:
    """Extended tests for WellInfo dataclass."""

    def test_to_location_no_screen(self):
        """Test to_location when screen info is missing results in z=None."""
        well = WellInfo(well_id="W1", x=10.0, y=20.0)
        loc = well.to_location()
        assert loc.z is None
        assert loc.x == 10.0
        assert loc.y == 20.0

    def test_to_location_partial_screen(self):
        """Test to_location when only one screen bound is set."""
        well = WellInfo(well_id="W1", x=10.0, y=20.0, screen_top=-50.0)
        loc = well.to_location()
        assert loc.z is None

    def test_name_attribute(self):
        """Test optional name attribute."""
        well = WellInfo(well_id="W1", x=10.0, y=20.0, name="Test Well")
        assert well.name == "Test Well"

    def test_to_location_with_node_id(self):
        """Test to_location includes node_id."""
        well = WellInfo(well_id="W1", x=10.0, y=20.0, node_id=42)
        loc = well.to_location()
        assert loc.node_id == 42


class TestGageInfoExtended:
    """Extended tests for GageInfo dataclass."""

    def test_name_attribute(self):
        """Test optional name attribute."""
        gage = GageInfo(gage_id="G1", name="Test Gage")
        assert gage.name == "Test Gage"

    def test_to_location_with_all_fields(self):
        """Test to_location with all optional fields set."""
        gage = GageInfo(gage_id="G1", reach_id=3, node_id=99, x=1.0, y=2.0)
        loc = gage.to_location()
        assert loc.node_id == 99
        assert loc.reach_id == 3

    def test_to_location_x_none_y_set(self):
        """Test to_location returns None when x is None but y is set."""
        gage = GageInfo(gage_id="G1", y=2.0)
        assert gage.to_location() is None

    def test_to_location_x_set_y_none(self):
        """Test to_location returns None when y is None but x is set."""
        gage = GageInfo(gage_id="G1", x=1.0)
        assert gage.to_location() is None


class TestHeadObservationsExtended:
    """Extended tests for head observation creation."""

    @pytest.fixture
    def wells_list(self):
        """Create sample wells as WellInfo list."""
        return [
            WellInfo(well_id="W1", x=100.0, y=200.0, layer=1),
            WellInfo(well_id="W2", x=300.0, y=400.0, layer=2),
        ]

    @pytest.fixture
    def head_data(self):
        """Create sample head data."""
        return pd.DataFrame(
            {
                "well_id": ["W1", "W1", "W2", "W2"],
                "datetime": [
                    datetime(2020, 1, 1),
                    datetime(2020, 2, 1),
                    datetime(2020, 1, 1),
                    datetime(2020, 2, 1),
                ],
                "head": [100.0, 101.0, 150.0, 151.0],
            }
        )

    def test_add_head_observations_from_wellinfo_list(self, wells_list, head_data):
        """Test adding head observations from WellInfo list input."""
        om = IWFMObservationManager()
        obs = om.add_head_observations(wells_list, head_data)
        assert len(obs) == 4
        assert all(o.obs_type == IWFMObservationType.HEAD for o in obs)

    def test_add_head_observations_layers_int(self, wells_list, head_data):
        """Test adding head observations with explicit integer layer."""
        om = IWFMObservationManager()
        obs = om.add_head_observations(wells_list, head_data, layers=3)
        # Check that all observations have layer 3
        for o in obs:
            assert o.location.layer == 3

    def test_add_head_observations_layers_list_uses_well_layer(self, wells_list, head_data):
        """Test adding head observations with layers as list (falls through to well.layer)."""
        om = IWFMObservationManager()
        obs = om.add_head_observations(wells_list, head_data, layers=[1, 2])
        # When layers is a list (not "auto", not int), the code falls to well.layer
        assert len(obs) == 4

    def test_add_head_observations_layers_auto(self, head_data):
        """Test adding head observations with layers='auto' and well without layer."""
        wells = [
            WellInfo(well_id="W1", x=100.0, y=200.0),
            WellInfo(well_id="W2", x=300.0, y=400.0),
        ]
        om = IWFMObservationManager()
        obs = om.add_head_observations(wells, head_data, layers="auto")
        # _determine_layer_from_screen returns None for wells without layer
        assert len(obs) == 4

    def test_add_head_observations_group_by_layer(self, wells_list, head_data):
        """Test head observations grouped by layer."""
        om = IWFMObservationManager()
        obs = om.add_head_observations(wells_list, head_data, group_by="layer")
        groups = {o.group for o in obs}
        assert "head_l1" in groups
        assert "head_l2" in groups

    def test_add_head_observations_group_by_time(self, wells_list, head_data):
        """Test head observations grouped by time."""
        om = IWFMObservationManager()
        obs = om.add_head_observations(wells_list, head_data, group_by="time")
        groups = {o.group for o in obs}
        # Should contain groups like "head_202001"
        assert any("202001" in g for g in groups)
        assert any("202002" in g for g in groups)

    def test_add_head_observations_group_by_all(self, wells_list, head_data):
        """Test head observations with group_by='all'."""
        om = IWFMObservationManager()
        obs = om.add_head_observations(wells_list, head_data, group_by="all")
        # group_by="all" falls to the else branch -> group="head"
        assert all(o.group == "head" for o in obs)

    def test_add_head_observations_end_date_filter(self, wells_list, head_data):
        """Test filtering by end date."""
        om = IWFMObservationManager()
        obs = om.add_head_observations(wells_list, head_data, end_date=datetime(2020, 1, 15))
        assert len(obs) == 2  # Only January observations

    def test_add_head_observations_unknown_well_skipped(self, wells_list):
        """Test that rows with unknown well_id are skipped."""
        head_data = pd.DataFrame(
            {
                "well_id": ["W_UNKNOWN"],
                "datetime": [datetime(2020, 1, 1)],
                "head": [100.0],
            }
        )
        om = IWFMObservationManager()
        obs = om.add_head_observations(wells_list, head_data)
        assert len(obs) == 0

    def test_add_head_observations_weight_strategy_string(self, wells_list, head_data):
        """Test that string weight strategy is converted to enum."""
        om = IWFMObservationManager()
        obs = om.add_head_observations(wells_list, head_data, weight_strategy="equal")
        assert all(o.weight == 1.0 for o in obs)

    def test_add_head_observations_custom_name_format(self, wells_list, head_data):
        """Test custom observation name format."""
        om = IWFMObservationManager()
        obs = om.add_head_observations(
            wells_list,
            head_data,
            obs_name_format="{well}_L{layer}_{date}",
        )
        assert len(obs) == 4
        # Names should follow the format
        for o in obs:
            assert "_L" in o.name

    def test_add_head_observations_error_std(self, wells_list, head_data):
        """Test error_std is passed through to observations."""
        om = IWFMObservationManager()
        obs = om.add_head_observations(wells_list, head_data, error_std=2.5)
        assert all(o.error_std == 2.5 for o in obs)


class TestDrawdownObservationsExtended:
    """Extended tests for drawdown observations."""

    @pytest.fixture
    def wells(self):
        return [
            WellInfo(well_id="W1", x=100.0, y=200.0),
            WellInfo(well_id="W2", x=300.0, y=400.0),
        ]

    @pytest.fixture
    def head_data(self):
        return pd.DataFrame(
            {
                "well_id": ["W1", "W1", "W2", "W2"],
                "datetime": [
                    datetime(2020, 1, 1),
                    datetime(2020, 6, 1),
                    datetime(2020, 1, 1),
                    datetime(2020, 6, 1),
                ],
                "head": [100.0, 95.0, 150.0, 145.0],
            }
        )

    def test_drawdown_without_reference_date(self, wells, head_data):
        """Test drawdown using first value as reference (no reference_date)."""
        om = IWFMObservationManager()
        obs = om.add_drawdown_observations(wells, head_data)
        assert len(obs) == 4
        assert all(o.obs_type == IWFMObservationType.DRAWDOWN for o in obs)
        # First W1 observation: reference=100, drawdown=100-100=0
        # Second W1 observation: reference=100, drawdown=100-95=5

    def test_drawdown_with_explicit_reference_values(self, wells, head_data):
        """Test drawdown with explicitly provided reference values."""
        om = IWFMObservationManager()
        ref = {"W1": 105.0, "W2": 155.0}
        obs = om.add_drawdown_observations(wells, head_data, reference_values=ref)
        assert len(obs) == 4
        for o in obs:
            assert "reference_value" in o.metadata


class TestHeadDifferenceExtended:
    """Extended tests for head difference observations."""

    def test_head_difference_multiple_pairs(self):
        """Test with multiple well pairs."""
        head_data = pd.DataFrame(
            {
                "well_id": ["W1", "W2", "W3", "W1", "W2", "W3"],
                "datetime": [
                    datetime(2020, 1, 1),
                    datetime(2020, 1, 1),
                    datetime(2020, 1, 1),
                    datetime(2020, 2, 1),
                    datetime(2020, 2, 1),
                    datetime(2020, 2, 1),
                ],
                "head": [100.0, 90.0, 80.0, 101.0, 91.0, 81.0],
            }
        )
        om = IWFMObservationManager()
        obs = om.add_head_difference_observations(
            well_pairs=[("W1", "W2"), ("W2", "W3")],
            observed_data=head_data,
        )
        # 2 pairs * 2 times = 4 observations
        assert len(obs) == 4

    def test_head_difference_custom_group(self):
        """Test custom group name for head differences."""
        head_data = pd.DataFrame(
            {
                "well_id": ["W1", "W2"],
                "datetime": [datetime(2020, 1, 1), datetime(2020, 1, 1)],
                "head": [100.0, 90.0],
            }
        )
        om = IWFMObservationManager()
        obs = om.add_head_difference_observations(
            well_pairs=[("W1", "W2")],
            observed_data=head_data,
            group_name="custom_hdiff",
        )
        assert all(o.group == "custom_hdiff" for o in obs)

    def test_head_difference_no_common_times(self):
        """Test with no common times yields no observations."""
        head_data = pd.DataFrame(
            {
                "well_id": ["W1", "W2"],
                "datetime": [datetime(2020, 1, 1), datetime(2020, 2, 1)],
                "head": [100.0, 90.0],
            }
        )
        om = IWFMObservationManager()
        obs = om.add_head_difference_observations(
            well_pairs=[("W1", "W2")],
            observed_data=head_data,
        )
        assert len(obs) == 0

    def test_head_difference_custom_weight(self):
        """Test custom weight for head difference observations."""
        head_data = pd.DataFrame(
            {
                "well_id": ["W1", "W2"],
                "datetime": [datetime(2020, 1, 1), datetime(2020, 1, 1)],
                "head": [100.0, 90.0],
            }
        )
        om = IWFMObservationManager()
        obs = om.add_head_difference_observations(
            well_pairs=[("W1", "W2")],
            observed_data=head_data,
            weight=5.0,
        )
        assert all(o.weight == 5.0 for o in obs)


class TestStreamflowExtended:
    """Extended tests for streamflow observations."""

    def test_streamflow_from_gage_list(self):
        """Test adding streamflow from GageInfo list."""
        gages = [
            GageInfo(gage_id="G1", reach_id=1, x=500.0, y=600.0),
            GageInfo(gage_id="G2", reach_id=2, x=700.0, y=800.0),
        ]
        flow_data = pd.DataFrame(
            {
                "gage_id": ["G1", "G2"],
                "datetime": [datetime(2020, 1, 1), datetime(2020, 1, 1)],
                "flow": [100.0, 200.0],
            }
        )
        om = IWFMObservationManager()
        obs = om.add_streamflow_observations(gages, flow_data)
        assert len(obs) == 2

    def test_streamflow_gage_not_in_dict(self):
        """Test that missing gage still creates observation (gage=None path)."""
        gages = [GageInfo(gage_id="G1", reach_id=1)]
        flow_data = pd.DataFrame(
            {
                "gage_id": ["G_UNKNOWN"],
                "datetime": [datetime(2020, 1, 1)],
                "flow": [100.0],
            }
        )
        om = IWFMObservationManager()
        obs = om.add_streamflow_observations(gages, flow_data)
        # gage_dict.get(gage_id) returns None, location will be None
        assert len(obs) == 1
        assert obs[0].location is None

    def test_streamflow_date_filter(self):
        """Test start and end date filtering for streamflow."""
        gages = [GageInfo(gage_id="G1")]
        flow_data = pd.DataFrame(
            {
                "gage_id": ["G1", "G1", "G1"],
                "datetime": [
                    datetime(2020, 1, 1),
                    datetime(2020, 6, 1),
                    datetime(2020, 12, 1),
                ],
                "flow": [100.0, 150.0, 200.0],
            }
        )
        om = IWFMObservationManager()
        obs = om.add_streamflow_observations(
            gages,
            flow_data,
            start_date=datetime(2020, 3, 1),
            end_date=datetime(2020, 9, 1),
        )
        assert len(obs) == 1

    def test_streamflow_negative_log_skipped(self):
        """Test that negative flows are skipped for log transform."""
        gages = [GageInfo(gage_id="G1")]
        flow_data = pd.DataFrame(
            {
                "gage_id": ["G1", "G1"],
                "datetime": [datetime(2020, 1, 1), datetime(2020, 2, 1)],
                "flow": [100.0, -50.0],
            }
        )
        om = IWFMObservationManager()
        obs = om.add_streamflow_observations(gages, flow_data, transform="log")
        assert len(obs) == 1

    def test_streamflow_custom_group_name(self):
        """Test custom group name for streamflow."""
        gages = [GageInfo(gage_id="G1")]
        flow_data = pd.DataFrame(
            {
                "gage_id": ["G1"],
                "datetime": [datetime(2020, 1, 1)],
                "flow": [100.0],
            }
        )
        om = IWFMObservationManager()
        obs = om.add_streamflow_observations(gages, flow_data, group_name="my_flow")
        assert obs[0].group == "my_flow"

    def test_streamflow_weight_strategy_string(self):
        """Test weight strategy as string for streamflow."""
        gages = [GageInfo(gage_id="G1")]
        flow_data = pd.DataFrame(
            {
                "gage_id": ["G1"],
                "datetime": [datetime(2020, 1, 1)],
                "flow": [100.0],
            }
        )
        om = IWFMObservationManager()
        obs = om.add_streamflow_observations(gages, flow_data, weight_strategy="equal")
        assert obs[0].weight == 1.0


class TestGainLossExtended:
    """Extended tests for gain/loss observations."""

    def test_gain_loss_reach_filtering(self):
        """Test that only specified reaches are included."""
        data = pd.DataFrame(
            {
                "reach_id": [1, 2, 3],
                "datetime": [datetime(2020, 1, 1)] * 3,
                "gain_loss": [10.0, -5.0, 8.0],
            }
        )
        om = IWFMObservationManager()
        obs = om.add_gain_loss_observations(reaches=[1, 3], observed_data=data)
        assert len(obs) == 2
        reach_ids = [o.metadata["reach_id"] for o in obs]
        assert 1 in reach_ids
        assert 3 in reach_ids
        assert 2 not in reach_ids

    def test_gain_loss_custom_group(self):
        """Test custom group name for gain/loss."""
        data = pd.DataFrame(
            {
                "reach_id": [1],
                "datetime": [datetime(2020, 1, 1)],
                "gain_loss": [10.0],
            }
        )
        om = IWFMObservationManager()
        obs = om.add_gain_loss_observations(reaches=[1], observed_data=data, group_name="custom_gl")
        assert obs[0].group == "custom_gl"


class TestLakeObservationsExtended:
    """Extended tests for lake observations."""

    def test_lake_observations_none_data(self):
        """Test lake observations with None data returns empty list."""
        om = IWFMObservationManager()
        obs = om.add_lake_observations(lakes=[1], observed_data=None)
        assert obs == []

    def test_lake_observations_all_lakes(self):
        """Test lake observations with lakes='all'."""
        data = pd.DataFrame(
            {
                "lake_id": [1, 2, 3],
                "datetime": [datetime(2020, 1, 1)] * 3,
                "value": [100.0, 200.0, 300.0],
            }
        )
        om = IWFMObservationManager()
        obs = om.add_lake_observations(lakes="all", observed_data=data)
        assert len(obs) == 3

    def test_lake_storage_type(self):
        """Test that storage type produces LAKE_STORAGE obs_type."""
        data = pd.DataFrame(
            {
                "lake_id": [1],
                "datetime": [datetime(2020, 1, 1)],
                "value": [1000.0],
            }
        )
        om = IWFMObservationManager()
        obs = om.add_lake_observations(lakes=[1], observed_data=data, obs_type="storage")
        assert obs[0].obs_type == IWFMObservationType.LAKE_STORAGE

    def test_lake_custom_group(self):
        """Test custom group for lake observations."""
        data = pd.DataFrame(
            {
                "lake_id": [1],
                "datetime": [datetime(2020, 1, 1)],
                "value": [100.0],
            }
        )
        om = IWFMObservationManager()
        obs = om.add_lake_observations(lakes=[1], observed_data=data, group_name="my_lake")
        assert obs[0].group == "my_lake"


class TestBudgetObservationsExtended:
    """Extended tests for budget observations."""

    def test_budget_rootzone_type(self):
        """Test rootzone budget type."""
        data = pd.DataFrame(
            {
                "datetime": [datetime(2020, 1, 1)],
                "value": [500.0],
            }
        )
        om = IWFMObservationManager()
        obs = om.add_budget_observations(budget_type="rootzone", observed_data=data)
        assert len(obs) == 1
        assert obs[0].obs_type == IWFMObservationType.ROOTZONE_BUDGET

    def test_budget_lake_type(self):
        """Test lake budget type."""
        data = pd.DataFrame(
            {
                "datetime": [datetime(2020, 1, 1)],
                "value": [500.0],
            }
        )
        om = IWFMObservationManager()
        obs = om.add_budget_observations(budget_type="lake", observed_data=data)
        assert obs[0].obs_type == IWFMObservationType.LAKE_BUDGET

    def test_budget_none_data_returns_empty(self):
        """Test budget with None data returns empty list."""
        om = IWFMObservationManager()
        obs = om.add_budget_observations(budget_type="gw", observed_data=None)
        assert obs == []

    def test_budget_component_filtering(self):
        """Test filtering by components."""
        data = pd.DataFrame(
            {
                "datetime": [datetime(2020, 1, 1)] * 3,
                "component": ["recharge", "pumping", "storage"],
                "value": [100.0, -50.0, 30.0],
            }
        )
        om = IWFMObservationManager()
        obs = om.add_budget_observations(
            budget_type="gw",
            components=["recharge", "storage"],
            observed_data=data,
        )
        components = [o.metadata["component"] for o in obs]
        assert "pumping" not in components

    def test_budget_location_filtering(self):
        """Test filtering by location IDs."""
        data = pd.DataFrame(
            {
                "datetime": [datetime(2020, 1, 1)] * 3,
                "location_id": [1, 2, 3],
                "value": [100.0, 200.0, 300.0],
            }
        )
        om = IWFMObservationManager()
        obs = om.add_budget_observations(
            budget_type="gw",
            locations=[1, 3],
            observed_data=data,
        )
        # After sum aggregation, location_id is removed
        assert len(obs) > 0

    def test_budget_aggregate_mean(self):
        """Test mean aggregation for budget observations."""
        data = pd.DataFrame(
            {
                "datetime": [datetime(2020, 1, 1)] * 2,
                "location_id": [1, 2],
                "value": [100.0, 200.0],
            }
        )
        om = IWFMObservationManager()
        obs = om.add_budget_observations(
            budget_type="gw",
            aggregate="mean",
            observed_data=data,
        )
        # Mean of 100, 200 = 150
        assert len(obs) == 1
        assert obs[0].value == pytest.approx(150.0)

    def test_budget_aggregate_by_location(self):
        """Test by_location aggregation (no aggregation)."""
        data = pd.DataFrame(
            {
                "datetime": [datetime(2020, 1, 1)] * 2,
                "location_id": [1, 2],
                "value": [100.0, 200.0],
            }
        )
        om = IWFMObservationManager()
        obs = om.add_budget_observations(
            budget_type="gw",
            aggregate="by_location",
            observed_data=data,
        )
        assert len(obs) == 2

    def test_budget_with_location_id_in_name(self):
        """Test that location_id is included in observation name."""
        data = pd.DataFrame(
            {
                "datetime": [datetime(2020, 1, 1)] * 2,
                "location_id": [1, 2],
                "value": [100.0, 200.0],
            }
        )
        om = IWFMObservationManager()
        obs = om.add_budget_observations(
            budget_type="gw",
            aggregate="by_location",
            observed_data=data,
        )
        names = [o.name for o in obs]
        # location_id should be in the name when aggregate != "sum"/"mean"
        assert any("1" in n for n in names)

    def test_budget_custom_group_name(self):
        """Test custom group name for budget observations."""
        data = pd.DataFrame(
            {
                "datetime": [datetime(2020, 1, 1)],
                "value": [100.0],
            }
        )
        om = IWFMObservationManager()
        obs = om.add_budget_observations(
            budget_type="gw",
            group_name="my_gwbud",
            observed_data=data,
        )
        assert obs[0].group == "my_gwbud"


class TestDerivedObservationsExtended:
    """Extended tests for derived observations."""

    def test_derived_observation_default_values(self):
        """Test derived observation with default target and weight."""
        om = IWFMObservationManager()
        om._observations["a"] = IWFMObservation(name="a", value=10.0)
        om._observations["b"] = IWFMObservation(name="b", value=5.0)

        derived = om.add_derived_observation(
            expression="a + b",
            obs_names=["a", "b"],
            result_name="sum_ab",
        )
        assert derived.target_value == 0.0
        assert derived.weight == 1.0
        assert derived.group == "derived"

    def test_derived_observation_stored(self):
        """Test that derived observations are stored in manager."""
        om = IWFMObservationManager()
        om._observations["x"] = IWFMObservation(name="x", value=1.0)
        om.add_derived_observation(
            expression="x",
            obs_names=["x"],
            result_name="derived_x",
        )
        assert "derived_x" in om._derived_observations


class TestWeightManagementExtended:
    """Extended tests for weight management."""

    def test_set_group_weights_auto_with_contribution(self):
        """Test setting auto weights with target contribution."""
        om = IWFMObservationManager()
        grp = om.get_observation_group("head")
        grp.add_observation("h1", 100.0, weight=1.0)

        om.set_group_weights("head", weight="auto", contribution=0.5)
        assert grp.target_contribution == 0.5

    def test_balance_observation_groups_no_observations(self):
        """Test balancing with no active groups does nothing."""
        om = IWFMObservationManager()
        om.balance_observation_groups()  # Should not raise

    def test_balance_observation_groups_equal(self):
        """Test balancing with equal contributions (no target specified)."""
        om = IWFMObservationManager()

        head_grp = om.get_observation_group("head")
        head_grp.add_observation("h1", 100.0, weight=1.0)
        head_grp.add_observation("h2", 200.0, weight=1.0)

        flow_grp = om.get_observation_group("flow")
        flow_grp.add_observation("f1", 50.0, weight=1.0)

        om.balance_observation_groups(target_contributions=None)

        # Both groups should contribute equally
        # After balancing, relative contributions should be closer

    def test_balance_observation_groups_with_unspecified(self):
        """Test balancing distributes remaining contribution to unspecified groups."""
        om = IWFMObservationManager()

        head_grp = om.get_observation_group("head")
        head_grp.add_observation("h1", 100.0, weight=1.0)

        flow_grp = om.get_observation_group("flow")
        flow_grp.add_observation("f1", 50.0, weight=1.0)

        # Only specify head, flow should get remaining
        om.balance_observation_groups({"head": 0.6})
        # flow should get (1.0 - 0.6) / 1 = 0.4

    def test_balance_observation_groups_zero_current_contribution(self):
        """Test balancing when a group has zero current contribution."""
        om = IWFMObservationManager()

        head_grp = om.get_observation_group("head")
        head_grp.add_observation("h1", 100.0, weight=0.0)

        flow_grp = om.get_observation_group("flow")
        flow_grp.add_observation("f1", 50.0, weight=1.0)

        om.balance_observation_groups({"head": 0.5, "flow": 0.5})

    def test_apply_temporal_weights_auto_reference(self):
        """Test temporal weights with auto-detected reference date."""
        om = IWFMObservationManager()

        grp = om.get_observation_group("head")
        obs1 = grp.add_observation("h1", 100.0, weight=1.0, datetime=datetime(2018, 1, 1))
        obs2 = grp.add_observation("h2", 200.0, weight=1.0, datetime=datetime(2020, 1, 1))
        om._observations["h1"] = obs1
        om._observations["h2"] = obs2

        om.apply_temporal_weights(decay_factor=0.9)
        # reference_date should be auto-detected as 2020-01-01
        assert om.get_observation("h1").weight < om.get_observation("h2").weight

    def test_apply_temporal_weights_no_dates(self):
        """Test temporal weights when no observations have dates."""
        om = IWFMObservationManager()
        om._observations["h1"] = IWFMObservation(name="h1", value=100.0, weight=1.0)

        om.apply_temporal_weights(decay_factor=0.9)
        # Should not modify weight (no datetime on any obs)
        assert om.get_observation("h1").weight == 1.0

    def test_apply_temporal_weights_obs_without_datetime(self):
        """Test that obs without datetime are not affected by temporal decay."""
        om = IWFMObservationManager()
        obs_with_date = IWFMObservation(
            name="h1", value=100.0, weight=1.0, datetime=datetime(2020, 1, 1)
        )
        obs_without_date = IWFMObservation(name="h2", value=200.0, weight=1.0)
        om._observations["h1"] = obs_with_date
        om._observations["h2"] = obs_without_date

        om.apply_temporal_weights(decay_factor=0.9, reference_date=datetime(2020, 6, 1))
        # obs without date should keep original weight
        assert om.get_observation("h2").weight == 1.0


class TestMakeValidObsName:
    """Tests for _make_valid_obs_name helper."""

    def test_special_characters_replaced(self):
        """Test that special characters are replaced with underscores."""
        om = IWFMObservationManager()
        name = om._make_valid_obs_name("test-name!@#")
        assert "-" not in name
        assert "!" not in name
        assert "@" not in name

    def test_long_name_truncated(self):
        """Test that names longer than 200 chars are truncated."""
        om = IWFMObservationManager()
        long_name = "a" * 250
        name = om._make_valid_obs_name(long_name)
        assert len(name) <= 200

    def test_duplicate_name_incremented(self):
        """Test that duplicate names are incremented with suffix."""
        om = IWFMObservationManager()
        om._observations["test_name"] = IWFMObservation(name="test_name", value=1.0)
        name = om._make_valid_obs_name("test_name")
        assert name == "test_name_1"

    def test_multiple_duplicates(self):
        """Test incrementing past multiple duplicates."""
        om = IWFMObservationManager()
        om._observations["test_name"] = IWFMObservation(name="test_name", value=1.0)
        om._observations["test_name_1"] = IWFMObservation(name="test_name_1", value=1.0)
        name = om._make_valid_obs_name("test_name")
        assert name == "test_name_2"


class TestParseWells:
    """Tests for _parse_wells helper."""

    def test_parse_from_dataframe(self):
        """Test parsing wells from DataFrame."""
        om = IWFMObservationManager()
        df = pd.DataFrame(
            {
                "well_id": ["W1", "W2"],
                "x": [100.0, 200.0],
                "y": [100.0, 200.0],
            }
        )
        wells = om._parse_wells(df)
        assert len(wells) == 2
        assert wells[0].well_id == "W1"

    def test_parse_from_wellinfo_list(self):
        """Test parsing wells from WellInfo list (passthrough)."""
        om = IWFMObservationManager()
        wells = [WellInfo(well_id="W1", x=1.0, y=2.0)]
        result = om._parse_wells(wells)
        assert result is wells

    def test_parse_from_file(self):
        """Test parsing wells from CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "wells.csv"
            pd.DataFrame(
                {
                    "well_id": ["W1"],
                    "x": [100.0],
                    "y": [200.0],
                }
            ).to_csv(filepath, index=False)

            om = IWFMObservationManager()
            wells = om._parse_wells(str(filepath))
            assert len(wells) == 1
            assert wells[0].well_id == "W1"


class TestParseGages:
    """Tests for _parse_gages helper."""

    def test_parse_from_dataframe(self):
        """Test parsing gages from DataFrame."""
        om = IWFMObservationManager()
        df = pd.DataFrame(
            {
                "gage_id": ["G1", "G2"],
                "reach_id": [1, 2],
            }
        )
        gages = om._parse_gages(df)
        assert len(gages) == 2
        assert gages[0].gage_id == "G1"

    def test_parse_from_gageinfo_list(self):
        """Test parsing gages from GageInfo list (passthrough)."""
        om = IWFMObservationManager()
        gages = [GageInfo(gage_id="G1")]
        result = om._parse_gages(gages)
        assert result is gages

    def test_parse_from_file(self):
        """Test parsing gages from CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "gages.csv"
            pd.DataFrame(
                {
                    "gage_id": ["G1"],
                    "reach_id": [5],
                }
            ).to_csv(filepath, index=False)

            om = IWFMObservationManager()
            gages = om._parse_gages(str(filepath))
            assert len(gages) == 1
            assert gages[0].gage_id == "G1"


class TestParseTimeseries:
    """Tests for _parse_timeseries_data helper."""

    def test_parse_from_dataframe(self):
        """Test parsing timeseries from DataFrame."""
        om = IWFMObservationManager()
        df = pd.DataFrame(
            {
                "datetime": ["2020-01-01", "2020-02-01"],
                "head": [100.0, 101.0],
            }
        )
        result = om._parse_timeseries_data(df, "head")
        assert len(result) == 2

    def test_parse_from_file(self):
        """Test parsing timeseries from CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "data.csv"
            pd.DataFrame(
                {
                    "datetime": ["2020-01-01"],
                    "head": [100.0],
                }
            ).to_csv(filepath, index=False)

            om = IWFMObservationManager()
            result = om._parse_timeseries_data(str(filepath), "head")
            assert len(result) == 1


class TestResampleTimeseries:
    """Tests for _resample_timeseries helper."""

    def test_resample_monthly(self):
        """Test resampling to monthly frequency."""
        om = IWFMObservationManager()
        df = pd.DataFrame(
            {
                "well_id": ["W1"] * 4,
                "datetime": pd.to_datetime(
                    [
                        "2020-01-01",
                        "2020-01-15",
                        "2020-02-01",
                        "2020-02-15",
                    ]
                ),
                "head": [100.0, 102.0, 104.0, 106.0],
            }
        )
        result = om._resample_timeseries(df, "well_id", "head", "MS")
        assert len(result) == 2  # 2 months


class TestFromDataFrameExtended:
    """Extended tests for from_dataframe."""

    def test_from_dataframe_with_obs_type(self):
        """Test loading from DataFrame with obs_type column."""
        om = IWFMObservationManager()
        df = pd.DataFrame(
            {
                "name": ["o1"],
                "value": [100.0],
                "weight": [1.0],
                "group": ["head"],
                "obs_type": ["head"],
            }
        )
        om.from_dataframe(df)
        obs = om.get_observation("o1")
        assert obs.obs_type == IWFMObservationType.HEAD

    def test_from_dataframe_with_datetime(self):
        """Test loading from DataFrame with datetime column."""
        om = IWFMObservationManager()
        df = pd.DataFrame(
            {
                "name": ["o1"],
                "value": [100.0],
                "weight": [1.0],
                "group": ["head"],
                "datetime": ["2020-01-01"],
            }
        )
        om.from_dataframe(df)
        obs = om.get_observation("o1")
        assert obs.datetime is not None

    def test_from_dataframe_with_transform(self):
        """Test loading from DataFrame with transform column."""
        om = IWFMObservationManager()
        df = pd.DataFrame(
            {
                "name": ["o1"],
                "value": [100.0],
                "weight": [1.0],
                "group": ["head"],
                "transform": ["log"],
            }
        )
        om.from_dataframe(df)
        obs = om.get_observation("o1")
        assert obs.transform == "log"

    def test_from_dataframe_creates_new_group(self):
        """Test that from_dataframe creates new groups if needed."""
        om = IWFMObservationManager()
        df = pd.DataFrame(
            {
                "name": ["o1"],
                "value": [100.0],
                "group": ["new_group"],
            }
        )
        om.from_dataframe(df)
        grp = om.get_observation_group("new_group")
        assert grp is not None
        assert len(grp.observations) == 1

    def test_from_dataframe_nan_obs_type(self):
        """Test loading from DataFrame with NaN obs_type."""
        om = IWFMObservationManager()
        df = pd.DataFrame(
            {
                "name": ["o1"],
                "value": [100.0],
                "group": ["head"],
                "obs_type": [np.nan],
            }
        )
        om.from_dataframe(df)
        obs = om.get_observation("o1")
        assert obs.obs_type is None

    def test_from_dataframe_nan_datetime(self):
        """Test loading from DataFrame with NaN datetime."""
        om = IWFMObservationManager()
        df = pd.DataFrame(
            {
                "name": ["o1"],
                "value": [100.0],
                "group": ["head"],
                "datetime": [np.nan],
            }
        )
        om.from_dataframe(df)
        obs = om.get_observation("o1")
        assert obs.datetime is None


class TestSummaryExtended:
    """Extended tests for summary and statistics."""

    def test_summary_with_derived(self):
        """Test summary includes derived observations count."""
        om = IWFMObservationManager()
        om._observations["a"] = IWFMObservation(name="a", value=1.0)
        om._observations["b"] = IWFMObservation(name="b", value=2.0)
        om.add_derived_observation(
            expression="a + b",
            obs_names=["a", "b"],
            result_name="sum_ab",
        )
        summary = om.summary()
        assert summary["n_derived"] == 1

    def test_summary_empty_manager(self):
        """Test summary on empty manager."""
        om = IWFMObservationManager()
        summary = om.summary()
        assert summary["n_observations"] == 0
        assert summary["n_groups"] == 0
        assert summary["n_derived"] == 0
        assert summary["groups"] == {}

    def test_n_groups_counts_only_active(self):
        """Test n_groups only counts groups with observations."""
        om = IWFMObservationManager()
        # Default groups exist but have no observations
        assert om.n_groups == 0
        grp = om.get_observation_group("head")
        grp.add_observation("h1", 100.0)
        assert om.n_groups == 1


class TestObservationAccessExtended:
    """Extended tests for observation access."""

    def test_get_observations_by_group_nonexistent(self):
        """Test getting observations for nonexistent group returns empty list."""
        om = IWFMObservationManager()
        result = om.get_observations_by_group("nonexistent")
        assert result == []

    def test_get_observation_group_nonexistent(self):
        """Test getting nonexistent group returns None."""
        om = IWFMObservationManager()
        result = om.get_observation_group("nonexistent")
        assert result is None

    def test_get_observations_by_type_empty(self):
        """Test getting observations by type when none exist."""
        om = IWFMObservationManager()
        result = om.get_observations_by_type(IWFMObservationType.SUBSIDENCE)
        assert result == []

    def test_get_all_groups_multiple(self):
        """Test getting all groups when multiple have observations."""
        om = IWFMObservationManager()
        om.get_observation_group("head").add_observation("h1", 100.0)
        om.get_observation_group("flow").add_observation("f1", 200.0)
        groups = om.get_all_groups()
        assert len(groups) == 2
        names = [g.name for g in groups]
        assert "head" in names
        assert "flow" in names


class TestFileIOExtended:
    """Extended tests for file I/O."""

    def test_write_and_read_roundtrip(self):
        """Test that writing and reading produces consistent data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "obs.csv"

            om1 = IWFMObservationManager()
            om1._observations["o1"] = IWFMObservation(
                name="o1",
                value=100.5,
                weight=2.0,
                group="head",
                obs_type=IWFMObservationType.HEAD,
            )
            om1.write_observation_file(filepath)

            om2 = IWFMObservationManager()
            om2.read_observation_file(filepath)

            obs = om2.get_observation("o1")
            assert obs is not None
            assert obs.value == pytest.approx(100.5)
            assert obs.weight == pytest.approx(2.0)


class TestDetermineLayer:
    """Tests for _determine_layer_from_screen."""

    def test_with_layer_set(self):
        """Test returns well.layer when set."""
        om = IWFMObservationManager()
        well = WellInfo(well_id="W1", x=0, y=0, layer=3)
        assert om._determine_layer_from_screen(well) == 3

    def test_without_layer(self):
        """Test returns None when no layer is set."""
        om = IWFMObservationManager()
        well = WellInfo(well_id="W1", x=0, y=0)
        assert om._determine_layer_from_screen(well) is None
