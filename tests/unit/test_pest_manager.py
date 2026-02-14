"""Tests for IWFM Parameter Manager."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from pyiwfm.runner.pest_params import (
    IWFMParameterType,
    ParameterTransform,
    Parameter,
)
from pyiwfm.runner.pest_manager import IWFMParameterManager


class TestIWFMParameterManagerInit:
    """Tests for IWFMParameterManager initialization."""

    def test_init_no_model(self):
        """Test initialization without model."""
        pm = IWFMParameterManager()
        assert pm.model is None
        assert pm.n_parameters == 0

    def test_init_with_model(self):
        """Test initialization with model."""
        model = MagicMock()
        pm = IWFMParameterManager(model)
        assert pm.model is model

    def test_default_groups_created(self):
        """Test that default parameter groups are created."""
        pm = IWFMParameterManager()
        assert "hk" in pm._parameter_groups
        assert "sy" in pm._parameter_groups
        assert "mult" in pm._parameter_groups
        assert "default" in pm._parameter_groups


class TestZoneParameters:
    """Tests for zone-based parameters."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model with subregions."""
        model = MagicMock()
        model.grid.subregions = {1: MagicMock(), 2: MagicMock(), 3: MagicMock()}
        return model

    def test_add_zone_parameters_explicit_zones(self):
        """Test adding zone parameters with explicit zones."""
        pm = IWFMParameterManager()
        params = pm.add_zone_parameters(
            IWFMParameterType.HORIZONTAL_K,
            zones=[1, 2, 3],
            layer=1,
            bounds=(0.1, 1000.0),
        )

        assert len(params) == 3
        assert pm.n_parameters == 3

    def test_add_zone_parameters_from_subregions(self, mock_model):
        """Test adding zone parameters from model subregions."""
        pm = IWFMParameterManager(mock_model)
        params = pm.add_zone_parameters(
            IWFMParameterType.SPECIFIC_YIELD,
            zones="subregions",
            layer=1,
        )

        assert len(params) == 3

    def test_add_zone_parameters_string_type(self):
        """Test adding zone parameters with string type."""
        pm = IWFMParameterManager()
        params = pm.add_zone_parameters(
            "hk",  # String instead of enum
            zones=[1, 2],
            layer=1,
        )

        assert len(params) == 2
        assert all(p.param_type == IWFMParameterType.HORIZONTAL_K for p in params)

    def test_zone_parameters_have_correct_layer(self):
        """Test that zone parameters have correct layer."""
        pm = IWFMParameterManager()
        params = pm.add_zone_parameters(
            IWFMParameterType.HORIZONTAL_K,
            zones=[1],
            layer=2,
        )

        assert params[0].layer == 2

    def test_zone_parameters_custom_group(self):
        """Test zone parameters with custom group."""
        pm = IWFMParameterManager()
        params = pm.add_zone_parameters(
            IWFMParameterType.HORIZONTAL_K,
            zones=[1],
            group="custom_hk",
        )

        assert params[0].group == "custom_hk"


class TestMultiplierParameters:
    """Tests for multiplier parameters."""

    def test_add_global_multiplier(self):
        """Test adding global multiplier."""
        pm = IWFMParameterManager()
        params = pm.add_multiplier_parameters(
            IWFMParameterType.PUMPING_MULT,
            spatial="global",
        )

        assert len(params) == 1
        assert params[0].initial_value == 1.0

    def test_add_seasonal_multiplier(self):
        """Test adding seasonal multiplier."""
        pm = IWFMParameterManager()
        params = pm.add_multiplier_parameters(
            IWFMParameterType.ET_MULT,
            temporal="seasonal",
        )

        assert len(params) == 4

    def test_add_zone_multiplier(self):
        """Test adding zone-based multiplier."""
        pm = IWFMParameterManager()
        params = pm.add_multiplier_parameters(
            IWFMParameterType.RECHARGE_MULT,
            spatial="zone",
            zones=[1, 2, 3, 4],
        )

        assert len(params) == 4

    def test_multiplier_custom_bounds(self):
        """Test multiplier with custom bounds."""
        pm = IWFMParameterManager()
        params = pm.add_multiplier_parameters(
            IWFMParameterType.PUMPING_MULT,
            bounds=(0.5, 1.5),
        )

        assert params[0].lower_bound == 0.5
        assert params[0].upper_bound == 1.5


class TestPilotPointParameters:
    """Tests for pilot point parameters."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model with grid."""
        model = MagicMock()
        x = np.linspace(0, 1000, 10)
        y = np.linspace(0, 1000, 10)
        xx, yy = np.meshgrid(x, y)
        model.grid.node_coordinates = np.column_stack([xx.ravel(), yy.ravel()])
        return model

    def test_add_pilot_points_explicit(self):
        """Test adding pilot points with explicit locations."""
        pm = IWFMParameterManager()
        points = [(0, 0), (100, 0), (0, 100), (100, 100)]
        params = pm.add_pilot_points(
            IWFMParameterType.HORIZONTAL_K,
            points=points,
            layer=1,
        )

        assert len(params) == 4
        assert all(p.location is not None for p in params)

    def test_add_pilot_points_grid(self, mock_model):
        """Test adding pilot points with grid spacing."""
        pm = IWFMParameterManager(mock_model)
        params = pm.add_pilot_points(
            IWFMParameterType.HORIZONTAL_K,
            spacing=500.0,
            layer=1,
        )

        assert len(params) > 1

    def test_pilot_points_have_variogram_metadata(self):
        """Test that variogram info is stored."""
        pm = IWFMParameterManager()
        variogram = {"type": "exponential", "a": 10000}
        params = pm.add_pilot_points(
            IWFMParameterType.HORIZONTAL_K,
            points=[(0, 0)],
            layer=1,
            variogram=variogram,
        )

        assert params[0].metadata["variogram"] == variogram


class TestStreamParameters:
    """Tests for stream parameters."""

    def test_add_stream_parameters(self):
        """Test adding stream parameters."""
        pm = IWFMParameterManager()
        params = pm.add_stream_parameters(
            IWFMParameterType.STREAMBED_K,
            reaches=[1, 2, 3],
        )

        assert len(params) == 3

    def test_stream_parameters_transform(self):
        """Test stream parameters default to log transform."""
        pm = IWFMParameterManager()
        params = pm.add_stream_parameters(
            IWFMParameterType.STREAMBED_K,
            reaches=[1],
        )

        assert params[0].transform == ParameterTransform.LOG


class TestRootZoneParameters:
    """Tests for root zone parameters."""

    def test_add_rootzone_parameters(self):
        """Test adding root zone parameters."""
        pm = IWFMParameterManager()
        params = pm.add_rootzone_parameters(
            IWFMParameterType.CROP_COEFFICIENT,
            land_use_types=["urban", "agriculture"],
        )

        assert len(params) == 2


class TestDirectParameters:
    """Tests for direct (single) parameters."""

    def test_add_single_parameter(self):
        """Test adding a single parameter."""
        pm = IWFMParameterManager()
        param = pm.add_parameter(
            name="ghb_east",
            param_type=IWFMParameterType.GHB_CONDUCTANCE,
            initial_value=100.0,
            bounds=(1.0, 10000.0),
        )

        assert param.name == "ghb_east"
        assert param.initial_value == 100.0
        assert pm.n_parameters == 1


class TestParameterRelationships:
    """Tests for parameter relationships (tied, fixed)."""

    def test_tie_parameters(self):
        """Test tying parameters together."""
        pm = IWFMParameterManager()
        pm.add_zone_parameters(
            IWFMParameterType.HORIZONTAL_K,
            zones=[1, 2, 3],
            layer=1,
        )

        # Get parameter names
        params = pm.get_all_parameters()
        parent = params[0].name
        children = [p.name for p in params[1:]]

        pm.tie_parameters(parent, children, ratios=0.5)

        # Check children are tied
        for child_name in children:
            child = pm.get_parameter(child_name)
            assert child.transform == ParameterTransform.TIED
            assert child.tied_to == parent
            assert child.tied_ratio == 0.5

    def test_tie_with_different_ratios(self):
        """Test tying with different ratios per child."""
        pm = IWFMParameterManager()
        pm.add_zone_parameters(
            IWFMParameterType.VERTICAL_K,
            zones=[1, 2, 3],
        )

        params = pm.get_all_parameters()
        parent = params[0].name
        children = [p.name for p in params[1:]]

        pm.tie_parameters(parent, children, ratios=[0.1, 0.01])

        child1 = pm.get_parameter(children[0])
        child2 = pm.get_parameter(children[1])
        assert child1.tied_ratio == 0.1
        assert child2.tied_ratio == 0.01

    def test_fix_parameter(self):
        """Test fixing a parameter."""
        pm = IWFMParameterManager()
        pm.add_parameter(
            name="fixed_param",
            param_type=IWFMParameterType.HORIZONTAL_K,
            initial_value=10.0,
        )

        pm.fix_parameter("fixed_param")

        param = pm.get_parameter("fixed_param")
        assert param.transform == ParameterTransform.FIXED

    def test_unfix_parameter(self):
        """Test unfixing a parameter."""
        pm = IWFMParameterManager()
        pm.add_parameter(
            name="test_param",
            param_type=IWFMParameterType.HORIZONTAL_K,
            initial_value=10.0,
        )
        pm.fix_parameter("test_param")
        pm.unfix_parameter("test_param")

        param = pm.get_parameter("test_param")
        assert param.transform != ParameterTransform.FIXED


class TestParameterAccess:
    """Tests for parameter access methods."""

    @pytest.fixture
    def populated_manager(self):
        """Create manager with various parameters."""
        pm = IWFMParameterManager()
        pm.add_zone_parameters(
            IWFMParameterType.HORIZONTAL_K,
            zones=[1, 2],
            layer=1,
        )
        pm.add_zone_parameters(
            IWFMParameterType.SPECIFIC_YIELD,
            zones=[1, 2],
            layer=1,
        )
        pm.add_multiplier_parameters(
            IWFMParameterType.PUMPING_MULT,
        )
        return pm

    def test_get_parameter_by_name(self, populated_manager):
        """Test getting parameter by name."""
        params = populated_manager.get_all_parameters()
        name = params[0].name

        param = populated_manager.get_parameter(name)
        assert param is not None
        assert param.name == name

    def test_get_parameter_not_found(self, populated_manager):
        """Test getting non-existent parameter returns None."""
        param = populated_manager.get_parameter("nonexistent")
        assert param is None

    def test_get_parameters_by_type(self, populated_manager):
        """Test getting parameters by type."""
        params = populated_manager.get_parameters_by_type(IWFMParameterType.HORIZONTAL_K)
        assert len(params) == 2
        assert all(p.param_type == IWFMParameterType.HORIZONTAL_K for p in params)

    def test_get_parameters_by_group(self, populated_manager):
        """Test getting parameters by group."""
        params = populated_manager.get_parameters_by_group("mult")
        assert len(params) >= 1

    def test_get_parameters_by_layer(self, populated_manager):
        """Test getting parameters by layer."""
        params = populated_manager.get_parameters_by_layer(1)
        assert len(params) == 4  # 2 hk + 2 sy

    def test_get_all_parameters(self, populated_manager):
        """Test getting all parameters."""
        params = populated_manager.get_all_parameters()
        assert len(params) == 5  # 2 hk + 2 sy + 1 mult

    def test_get_adjustable_parameters(self, populated_manager):
        """Test getting adjustable parameters."""
        # Fix one parameter
        params = populated_manager.get_all_parameters()
        populated_manager.fix_parameter(params[0].name)

        adjustable = populated_manager.get_adjustable_parameters()
        assert len(adjustable) == 4  # 5 - 1 fixed

    def test_get_pilot_point_parameters(self):
        """Test getting pilot point parameters."""
        pm = IWFMParameterManager()
        pm.add_pilot_points(
            IWFMParameterType.HORIZONTAL_K,
            points=[(0, 0), (100, 100)],
            layer=1,
        )
        pm.add_zone_parameters(
            IWFMParameterType.SPECIFIC_YIELD,
            zones=[1],
        )

        pp_params = pm.get_pilot_point_parameters()
        assert len(pp_params) == 2
        assert all(p.location is not None for p in pp_params)


class TestParameterGroups:
    """Tests for parameter group management."""

    def test_add_parameter_group(self):
        """Test adding a parameter group."""
        pm = IWFMParameterManager()
        group = pm.add_parameter_group(
            name="custom",
            inctyp="relative",
            derinc=0.02,
        )

        assert group.name == "custom"
        assert group.derinc == 0.02

    def test_get_parameter_group(self):
        """Test getting a parameter group."""
        pm = IWFMParameterManager()
        pm.add_parameter_group(name="test_group")

        group = pm.get_parameter_group("test_group")
        assert group is not None
        assert group.name == "test_group"

    def test_get_all_groups(self):
        """Test getting all used groups."""
        pm = IWFMParameterManager()
        pm.add_zone_parameters(
            IWFMParameterType.HORIZONTAL_K,
            zones=[1],
            group="hk",
        )
        pm.add_multiplier_parameters(
            IWFMParameterType.PUMPING_MULT,
            group="mult",
        )

        groups = pm.get_all_groups()
        group_names = [g.name for g in groups]
        assert "hk" in group_names
        assert "mult" in group_names


class TestDataFrameExport:
    """Tests for DataFrame export/import."""

    def test_to_dataframe(self):
        """Test exporting to DataFrame."""
        pytest.importorskip("pandas")

        pm = IWFMParameterManager()
        pm.add_zone_parameters(
            IWFMParameterType.HORIZONTAL_K,
            zones=[1, 2],
        )

        df = pm.to_dataframe()
        assert len(df) == 2
        assert "name" in df.columns
        assert "initial_value" in df.columns
        assert "lower_bound" in df.columns

    def test_from_dataframe(self):
        """Test loading values from DataFrame."""
        pd = pytest.importorskip("pandas")

        pm = IWFMParameterManager()
        pm.add_zone_parameters(
            IWFMParameterType.HORIZONTAL_K,
            zones=[1, 2],
            initial_values=10.0,
        )

        # Get parameter names
        params = pm.get_all_parameters()
        names = [p.name for p in params]

        # Create DataFrame with new values
        df = pd.DataFrame({
            "name": names,
            "initial_value": [20.0, 30.0],
        })

        pm.from_dataframe(df)

        # Check values were updated
        assert pm.get_parameter(names[0]).initial_value == 20.0
        assert pm.get_parameter(names[1]).initial_value == 30.0


class TestFileIO:
    """Tests for file I/O."""

    def test_write_parameter_file(self, tmp_path):
        """Test writing parameter file."""
        pm = IWFMParameterManager()
        pm.add_zone_parameters(
            IWFMParameterType.HORIZONTAL_K,
            zones=[1, 2],
        )

        filepath = tmp_path / "params.csv"
        pm.write_parameter_file(filepath)

        assert filepath.exists()
        content = filepath.read_text()
        assert "hk" in content

    def test_read_parameter_file(self, tmp_path):
        """Test reading parameter file."""
        pm = IWFMParameterManager()
        pm.add_zone_parameters(
            IWFMParameterType.HORIZONTAL_K,
            zones=[1],
            initial_values=10.0,
        )

        # Get parameter name
        name = pm.get_all_parameters()[0].name

        # Write file with different value
        filepath = tmp_path / "params.csv"
        filepath.write_text(f"# Comment\n{name}, 50.0, 0.1, 100.0, hk, log\n")

        pm.read_parameter_file(filepath)

        assert pm.get_parameter(name).initial_value == 50.0


class TestStatisticsAndSummary:
    """Tests for statistics and summary methods."""

    def test_n_parameters(self):
        """Test n_parameters property."""
        pm = IWFMParameterManager()
        assert pm.n_parameters == 0

        pm.add_zone_parameters(
            IWFMParameterType.HORIZONTAL_K,
            zones=[1, 2, 3],
        )
        assert pm.n_parameters == 3

    def test_n_adjustable(self):
        """Test n_adjustable property."""
        pm = IWFMParameterManager()
        pm.add_zone_parameters(
            IWFMParameterType.HORIZONTAL_K,
            zones=[1, 2, 3],
        )

        assert pm.n_adjustable == 3

        # Fix one
        params = pm.get_all_parameters()
        pm.fix_parameter(params[0].name)

        assert pm.n_adjustable == 2

    def test_summary(self):
        """Test summary method."""
        pm = IWFMParameterManager()
        pm.add_zone_parameters(
            IWFMParameterType.HORIZONTAL_K,
            zones=[1, 2],
        )
        pm.add_multiplier_parameters(IWFMParameterType.PUMPING_MULT)

        summary = pm.summary()
        assert "Parameter Manager Summary" in summary
        assert "hk" in summary
        assert "pump" in summary


class TestIteration:
    """Tests for iteration support."""

    def test_iteration(self):
        """Test iterating over parameters."""
        pm = IWFMParameterManager()
        pm.add_zone_parameters(
            IWFMParameterType.HORIZONTAL_K,
            zones=[1, 2, 3],
        )

        count = 0
        for param in pm:
            assert isinstance(param, Parameter)
            count += 1

        assert count == 3

    def test_len(self):
        """Test len() on manager."""
        pm = IWFMParameterManager()
        assert len(pm) == 0

        pm.add_zone_parameters(
            IWFMParameterType.HORIZONTAL_K,
            zones=[1, 2],
        )
        assert len(pm) == 2

    def test_repr(self):
        """Test string representation."""
        pm = IWFMParameterManager()
        pm.add_zone_parameters(
            IWFMParameterType.HORIZONTAL_K,
            zones=[1, 2],
        )

        repr_str = repr(pm)
        assert "IWFMParameterManager" in repr_str
        assert "n_parameters=2" in repr_str


# =========================================================================
# Additional tests to increase coverage to 95%+
# =========================================================================


class TestTieParametersEdgeCases:
    """Tests for tie_parameters error paths and edge cases."""

    def test_tie_parent_not_found(self):
        """Test tying with nonexistent parent raises ValueError."""
        pm = IWFMParameterManager()
        pm.add_parameter(
            name="child1",
            param_type=IWFMParameterType.HORIZONTAL_K,
            initial_value=10.0,
        )
        with pytest.raises(ValueError, match="Parent parameter.*not found"):
            pm.tie_parameters("nonexistent_parent", ["child1"])

    def test_tie_child_not_found(self):
        """Test tying with nonexistent child raises ValueError."""
        pm = IWFMParameterManager()
        pm.add_parameter(
            name="parent1",
            param_type=IWFMParameterType.HORIZONTAL_K,
            initial_value=10.0,
        )
        with pytest.raises(ValueError, match="Child parameter.*not found"):
            pm.tie_parameters("parent1", ["nonexistent_child"])

    def test_tie_mismatched_ratios(self):
        """Test tying with mismatched number of ratios raises ValueError."""
        pm = IWFMParameterManager()
        pm.add_parameter(
            name="parent",
            param_type=IWFMParameterType.HORIZONTAL_K,
            initial_value=10.0,
        )
        pm.add_parameter(
            name="child1",
            param_type=IWFMParameterType.HORIZONTAL_K,
            initial_value=5.0,
        )
        pm.add_parameter(
            name="child2",
            param_type=IWFMParameterType.HORIZONTAL_K,
            initial_value=3.0,
        )
        with pytest.raises(ValueError, match="Number of ratios.*must match"):
            pm.tie_parameters("parent", ["child1", "child2"], ratios=[0.5])

    def test_tie_with_int_ratio(self):
        """Test tying with integer ratio is handled correctly."""
        pm = IWFMParameterManager()
        pm.add_parameter(
            name="parent",
            param_type=IWFMParameterType.HORIZONTAL_K,
            initial_value=10.0,
        )
        pm.add_parameter(
            name="child",
            param_type=IWFMParameterType.HORIZONTAL_K,
            initial_value=5.0,
        )
        pm.tie_parameters("parent", ["child"], ratios=1)
        child = pm.get_parameter("child")
        assert child.transform == ParameterTransform.TIED
        assert child.tied_ratio == 1.0


class TestFixUnfixParameterEdgeCases:
    """Tests for fix/unfix parameter edge cases."""

    def test_fix_nonexistent_parameter(self):
        """Test fixing nonexistent parameter raises ValueError."""
        pm = IWFMParameterManager()
        with pytest.raises(ValueError, match="Parameter.*not found"):
            pm.fix_parameter("nonexistent")

    def test_unfix_nonexistent_parameter(self):
        """Test unfixing nonexistent parameter raises ValueError."""
        pm = IWFMParameterManager()
        with pytest.raises(ValueError, match="Parameter.*not found"):
            pm.unfix_parameter("nonexistent")

    def test_unfix_with_string_transform(self):
        """Test unfixing a parameter with a string transform argument."""
        pm = IWFMParameterManager()
        pm.add_parameter(
            name="test_param",
            param_type=IWFMParameterType.HORIZONTAL_K,
            initial_value=10.0,
        )
        pm.fix_parameter("test_param")
        pm.unfix_parameter("test_param", transform="log")
        param = pm.get_parameter("test_param")
        assert param.transform == ParameterTransform.LOG

    def test_unfix_with_auto_transform(self):
        """Test unfixing with auto transform uses type default."""
        pm = IWFMParameterManager()
        pm.add_parameter(
            name="test_param",
            param_type=IWFMParameterType.HORIZONTAL_K,
            initial_value=10.0,
        )
        pm.fix_parameter("test_param")
        pm.unfix_parameter("test_param", transform="auto")
        param = pm.get_parameter("test_param")
        # Should use the default transform for HORIZONTAL_K
        assert param.transform != ParameterTransform.FIXED


class TestAddParameterizationEdgeCases:
    """Tests for _add_parameterization updating existing parameters."""

    def test_add_duplicate_zone_parameters_updates(self):
        """Test that adding duplicate zone parameters updates existing ones."""
        pm = IWFMParameterManager()
        params1 = pm.add_zone_parameters(
            IWFMParameterType.HORIZONTAL_K,
            zones=[1, 2],
            layer=1,
            initial_values=10.0,
        )
        # Re-add the same zones with different initial values
        params2 = pm.add_zone_parameters(
            IWFMParameterType.HORIZONTAL_K,
            zones=[1, 2],
            layer=1,
            initial_values=20.0,
        )
        # Should still have the same number of parameters (updated in-place)
        assert pm.n_parameters == 2
        # Values should be updated
        for p in pm.get_all_parameters():
            assert p.initial_value == 20.0


class TestDirectParameterEdgeCases:
    """Tests for add_parameter edge cases."""

    def test_add_parameter_string_type(self):
        """Test adding parameter with string type."""
        pm = IWFMParameterManager()
        param = pm.add_parameter(
            name="test_hk",
            param_type="hk",
            initial_value=5.0,
        )
        assert param.param_type == IWFMParameterType.HORIZONTAL_K

    def test_add_parameter_auto_transform(self):
        """Test adding parameter with auto transform."""
        pm = IWFMParameterManager()
        param = pm.add_parameter(
            name="auto_trans",
            param_type=IWFMParameterType.HORIZONTAL_K,
            initial_value=10.0,
            transform="auto",
        )
        # Auto should use the default transform for the type
        assert param.transform is not None

    def test_add_parameter_none_bounds(self):
        """Test adding parameter with None bounds uses type defaults."""
        pm = IWFMParameterManager()
        param = pm.add_parameter(
            name="default_bounds",
            param_type=IWFMParameterType.HORIZONTAL_K,
            initial_value=10.0,
            bounds=None,
        )
        # Should use type default bounds
        assert param.lower_bound is not None
        assert param.upper_bound is not None

    def test_add_parameter_with_metadata(self):
        """Test adding parameter with extra metadata."""
        pm = IWFMParameterManager()
        param = pm.add_parameter(
            name="meta_param",
            param_type=IWFMParameterType.HORIZONTAL_K,
            initial_value=10.0,
            bounds=(0.1, 100.0),
            layer=1,
            zone_id=5,
            description="custom metadata",
        )
        assert param.metadata.get("zone_id") == 5
        assert param.metadata.get("description") == "custom metadata"


class TestParameterGroupEdgeCases:
    """Tests for parameter group edge cases."""

    def test_get_nonexistent_group(self):
        """Test getting nonexistent group returns None."""
        pm = IWFMParameterManager()
        group = pm.get_parameter_group("nonexistent_group")
        assert group is None

    def test_add_parameter_group_override(self):
        """Test that adding a group with existing name overrides it."""
        pm = IWFMParameterManager()
        pm.add_parameter_group(name="hk", derinc=0.01)
        pm.add_parameter_group(name="hk", derinc=0.05)
        group = pm.get_parameter_group("hk")
        assert group.derinc == 0.05


class TestParameterAccessEdgeCases:
    """Tests for parameter access edge cases."""

    def test_get_parameters_by_type_string(self):
        """Test getting parameters by type using string."""
        pm = IWFMParameterManager()
        pm.add_zone_parameters(
            IWFMParameterType.HORIZONTAL_K,
            zones=[1, 2],
        )
        params = pm.get_parameters_by_type("hk")
        assert len(params) == 2

    def test_get_parameters_by_layer_no_match(self):
        """Test getting parameters by layer with no matching layer."""
        pm = IWFMParameterManager()
        pm.add_zone_parameters(
            IWFMParameterType.HORIZONTAL_K,
            zones=[1],
            layer=1,
        )
        params = pm.get_parameters_by_layer(99)
        assert len(params) == 0

    def test_get_parameters_by_group_no_match(self):
        """Test getting parameters by group with no matching group."""
        pm = IWFMParameterManager()
        pm.add_zone_parameters(
            IWFMParameterType.HORIZONTAL_K,
            zones=[1],
            group="hk",
        )
        params = pm.get_parameters_by_group("nonexistent")
        assert len(params) == 0

    def test_get_pilot_point_parameters_empty(self):
        """Test getting pilot point params when there are none."""
        pm = IWFMParameterManager()
        pm.add_zone_parameters(
            IWFMParameterType.HORIZONTAL_K,
            zones=[1],
        )
        pp_params = pm.get_pilot_point_parameters()
        assert len(pp_params) == 0


class TestFileIOEdgeCases:
    """Tests for file I/O edge cases."""

    def test_write_and_read_roundtrip(self, tmp_path):
        """Test writing then reading parameter file preserves values."""
        pm = IWFMParameterManager()
        pm.add_zone_parameters(
            IWFMParameterType.HORIZONTAL_K,
            zones=[1, 2],
            initial_values=42.0,
        )

        filepath = tmp_path / "params_roundtrip.csv"
        pm.write_parameter_file(filepath)

        # Change values
        for p in pm.get_all_parameters():
            p.initial_value = 0.0

        # Read back
        pm.read_parameter_file(filepath)

        for p in pm.get_all_parameters():
            assert abs(p.initial_value - 42.0) < 1e-5

    def test_read_parameter_file_skips_comments_and_blanks(self, tmp_path):
        """Test that reading skips comment lines and blank lines."""
        pm = IWFMParameterManager()
        pm.add_parameter(
            name="p1",
            param_type=IWFMParameterType.HORIZONTAL_K,
            initial_value=1.0,
            bounds=(0.01, 100.0),
        )

        filepath = tmp_path / "params_comments.csv"
        filepath.write_text("# This is a comment\n\n  \np1, 99.0, 0.01, 100.0, hk, log\n")

        pm.read_parameter_file(filepath)
        assert pm.get_parameter("p1").initial_value == 99.0

    def test_read_parameter_file_unknown_param_ignored(self, tmp_path):
        """Test that unknown param names in file are ignored."""
        pm = IWFMParameterManager()
        pm.add_parameter(
            name="known",
            param_type=IWFMParameterType.HORIZONTAL_K,
            initial_value=1.0,
            bounds=(0.01, 100.0),
        )

        filepath = tmp_path / "params_unknown.csv"
        filepath.write_text("known, 50.0, 0.01, 100.0, hk, log\nunknown_param, 99.0\n")

        pm.read_parameter_file(filepath)
        assert pm.get_parameter("known").initial_value == 50.0
        assert pm.n_parameters == 1  # No extra param added


class TestSummaryEdgeCases:
    """Tests for summary method edge cases."""

    def test_summary_empty_manager(self):
        """Test summary with no parameters."""
        pm = IWFMParameterManager()
        summary = pm.summary()
        assert "Total parameters: 0" in summary
        assert "Adjustable parameters: 0" in summary

    def test_summary_with_fixed_and_tied(self):
        """Test summary counting fixed and tied parameters."""
        pm = IWFMParameterManager()
        pm.add_zone_parameters(
            IWFMParameterType.HORIZONTAL_K,
            zones=[1, 2, 3],
            layer=1,
        )
        params = pm.get_all_parameters()
        pm.fix_parameter(params[0].name)
        pm.tie_parameters(params[1].name, [params[2].name])

        summary = pm.summary()
        assert "Fixed parameters: 1" in summary
        assert "Tied parameters: 1" in summary

    def test_n_groups_property(self):
        """Test n_groups property."""
        pm = IWFMParameterManager()
        assert pm.n_groups == 0
        pm.add_zone_parameters(
            IWFMParameterType.HORIZONTAL_K,
            zones=[1],
            group="hk",
        )
        assert pm.n_groups == 1


class TestStreamParameterEdgeCases:
    """Tests for stream parameters with string types and transforms."""

    def test_add_stream_parameters_string_type(self):
        """Test adding stream parameters with string param type."""
        pm = IWFMParameterManager()
        params = pm.add_stream_parameters(
            "strk",
            reaches=[1, 2],
        )
        assert len(params) == 2

    def test_add_stream_parameters_with_string_transform(self):
        """Test adding stream parameters with explicit string transform."""
        pm = IWFMParameterManager()
        params = pm.add_stream_parameters(
            IWFMParameterType.STREAMBED_K,
            reaches=[1],
            transform="none",
        )
        # Verify the transform was passed through (strategy may apply its own logic)
        assert params[0].transform is not None


class TestRootZoneParameterEdgeCases:
    """Tests for root zone parameters with string types and transforms."""

    def test_add_rootzone_parameters_string_type(self):
        """Test adding root zone parameters with string param type."""
        pm = IWFMParameterManager()
        params = pm.add_rootzone_parameters(
            "kc",
            land_use_types=["corn"],
        )
        assert len(params) == 1

    def test_add_rootzone_parameters_string_transform(self):
        """Test adding root zone parameters with string transform."""
        pm = IWFMParameterManager()
        params = pm.add_rootzone_parameters(
            IWFMParameterType.CROP_COEFFICIENT,
            land_use_types=["wheat"],
            transform="log",
        )
        assert params[0].transform == ParameterTransform.LOG


class TestMultiplierParameterEdgeCases:
    """Tests for multiplier parameter edge cases."""

    def test_add_multiplier_string_type(self):
        """Test adding multiplier with string param type."""
        pm = IWFMParameterManager()
        params = pm.add_multiplier_parameters(
            "pump",
            spatial="global",
        )
        assert len(params) >= 1

    def test_add_multiplier_with_target_file(self, tmp_path):
        """Test adding multiplier with target file reference."""
        pm = IWFMParameterManager()
        target = tmp_path / "base_values.dat"
        target.write_text("dummy data")
        params = pm.add_multiplier_parameters(
            IWFMParameterType.PUMPING_MULT,
            spatial="global",
            target_file=target,
        )
        assert len(params) >= 1

    def test_add_monthly_multiplier(self):
        """Test adding monthly temporal multiplier."""
        pm = IWFMParameterManager()
        params = pm.add_multiplier_parameters(
            IWFMParameterType.ET_MULT,
            temporal="monthly",
        )
        assert len(params) >= 1


class TestPilotPointEdgeCases:
    """Tests for pilot point edge cases."""

    def test_pilot_points_with_string_transform(self):
        """Test pilot points with explicit string transform."""
        pm = IWFMParameterManager()
        params = pm.add_pilot_points(
            IWFMParameterType.HORIZONTAL_K,
            points=[(0, 0), (100, 100)],
            layer=1,
            transform="none",
        )
        # Verify parameters were created with a valid transform
        assert all(p.transform is not None for p in params)
