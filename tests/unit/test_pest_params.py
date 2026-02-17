"""Tests for PEST++ parameter types and parameterization strategies."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from pyiwfm.runner.pest_params import (
    DirectParameterization,
    IWFMParameterType,
    MultiplierParameterization,
    Parameter,
    ParameterGroup,
    ParameterTransform,
    PilotPointParameterization,
    RootZoneParameterization,
    StreamParameterization,
    ZoneParameterization,
)


class TestIWFMParameterType:
    """Tests for IWFMParameterType enum."""

    def test_aquifer_parameters_exist(self):
        """Test that aquifer parameter types exist."""
        assert IWFMParameterType.HORIZONTAL_K.value == "hk"
        assert IWFMParameterType.VERTICAL_K.value == "vk"
        assert IWFMParameterType.SPECIFIC_STORAGE.value == "ss"
        assert IWFMParameterType.SPECIFIC_YIELD.value == "sy"

    def test_multiplier_parameters_exist(self):
        """Test that multiplier parameter types exist."""
        assert IWFMParameterType.PUMPING_MULT.value == "pump"
        assert IWFMParameterType.RECHARGE_MULT.value == "rech"
        assert IWFMParameterType.ET_MULT.value == "et"

    def test_stream_parameters_exist(self):
        """Test that stream parameter types exist."""
        assert IWFMParameterType.STREAMBED_K.value == "strk"
        assert IWFMParameterType.STREAMBED_THICKNESS.value == "strt"

    def test_default_bounds(self):
        """Test that default bounds are provided."""
        bounds = IWFMParameterType.HORIZONTAL_K.default_bounds
        assert bounds[0] < bounds[1]
        assert bounds[0] > 0

    def test_default_transform(self):
        """Test that default transforms are provided."""
        # K parameters should default to log
        assert IWFMParameterType.HORIZONTAL_K.default_transform == "log"
        # Multipliers should default to none
        assert IWFMParameterType.PUMPING_MULT.default_transform == "none"

    def test_is_multiplier(self):
        """Test is_multiplier property."""
        assert IWFMParameterType.PUMPING_MULT.is_multiplier is True
        assert IWFMParameterType.HORIZONTAL_K.is_multiplier is False


class TestParameterTransform:
    """Tests for ParameterTransform enum."""

    def test_transform_values(self):
        """Test transform enum values."""
        assert ParameterTransform.NONE.value == "none"
        assert ParameterTransform.LOG.value == "log"
        assert ParameterTransform.FIXED.value == "fixed"
        assert ParameterTransform.TIED.value == "tied"


class TestParameterGroup:
    """Tests for ParameterGroup dataclass."""

    def test_basic_creation(self):
        """Test creating a basic parameter group."""
        group = ParameterGroup(name="hk")
        assert group.name == "hk"
        assert group.inctyp == "relative"
        assert group.derinc == 0.01

    def test_name_truncation(self):
        """Test that long names are truncated."""
        group = ParameterGroup(name="very_long_group_name")
        assert len(group.name) <= 12

    def test_to_pest_line(self):
        """Test PEST control file formatting."""
        group = ParameterGroup(name="hk", inctyp="relative", derinc=0.01)
        line = group.to_pest_line()
        assert "hk" in line
        assert "relative" in line


class TestParameter:
    """Tests for Parameter dataclass."""

    def test_basic_creation(self):
        """Test creating a basic parameter."""
        param = Parameter(
            name="hk_zone1",
            initial_value=10.0,
            lower_bound=0.1,
            upper_bound=1000.0,
        )
        assert param.name == "hk_zone1"
        assert param.initial_value == 10.0
        assert param.lower_bound == 0.1
        assert param.upper_bound == 1000.0

    def test_with_param_type(self):
        """Test parameter with IWFM type."""
        param = Parameter(
            name="hk_z1",
            initial_value=10.0,
            lower_bound=0.1,
            upper_bound=1000.0,
            param_type=IWFMParameterType.HORIZONTAL_K,
        )
        assert param.param_type == IWFMParameterType.HORIZONTAL_K

    def test_with_layer_and_zone(self):
        """Test parameter with layer and zone."""
        param = Parameter(
            name="hk_z1_l1",
            initial_value=10.0,
            lower_bound=0.1,
            upper_bound=1000.0,
            layer=1,
            zone=1,
        )
        assert param.layer == 1
        assert param.zone == 1

    def test_with_location(self):
        """Test parameter with pilot point location."""
        param = Parameter(
            name="hk_pp001",
            initial_value=10.0,
            lower_bound=0.1,
            upper_bound=1000.0,
            location=(100.0, 200.0),
        )
        assert param.location == (100.0, 200.0)

    def test_invalid_name_length(self):
        """Test that long names raise error."""
        with pytest.raises(ValueError, match="name too long"):
            Parameter(
                name="x" * 201,
                initial_value=1.0,
                lower_bound=0.0,
                upper_bound=2.0,
            )

    def test_invalid_bounds(self):
        """Test that invalid bounds raise error."""
        with pytest.raises(ValueError, match="Lower bound"):
            Parameter(
                name="test",
                initial_value=1.0,
                lower_bound=10.0,
                upper_bound=5.0,
            )

    def test_initial_clipped_to_bounds(self):
        """Test that initial value is clipped to bounds."""
        param = Parameter(
            name="test",
            initial_value=100.0,
            lower_bound=0.0,
            upper_bound=10.0,
        )
        # Value should be clipped
        assert param.initial_value == 10.0

    def test_transform_string_conversion(self):
        """Test that string transform is converted to enum."""
        param = Parameter(
            name="test",
            initial_value=1.0,
            lower_bound=0.0,
            upper_bound=10.0,
            transform="log",
        )
        assert param.transform == ParameterTransform.LOG

    def test_partrans_property(self):
        """Test PEST transformation string."""
        param = Parameter(
            name="test",
            initial_value=1.0,
            lower_bound=0.1,
            upper_bound=10.0,
            transform=ParameterTransform.LOG,
        )
        assert param.partrans == "log"

    def test_tied_parameter(self):
        """Test tied parameter configuration."""
        param = Parameter(
            name="child",
            initial_value=1.0,
            lower_bound=0.1,
            upper_bound=10.0,
            transform=ParameterTransform.TIED,
            tied_to="parent",
            tied_ratio=0.5,
        )
        assert param.partrans == "tied"
        assert param.tied_to == "parent"
        assert param.tied_ratio == 0.5

    def test_to_pest_line(self):
        """Test PEST control file formatting."""
        param = Parameter(
            name="hk_zone1",
            initial_value=10.0,
            lower_bound=0.1,
            upper_bound=1000.0,
            group="hk",
        )
        line = param.to_pest_line()
        assert "hk_zone1" in line
        assert "hk" in line

    def test_repr(self):
        """Test string representation."""
        param = Parameter(
            name="test",
            initial_value=10.0,
            lower_bound=0.1,
            upper_bound=100.0,
        )
        repr_str = repr(param)
        assert "test" in repr_str
        assert "10" in repr_str


class TestZoneParameterization:
    """Tests for ZoneParameterization strategy."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock IWFM model."""
        model = MagicMock()
        model.grid.subregions = {1: MagicMock(), 2: MagicMock(), 3: MagicMock()}
        return model

    def test_basic_creation(self):
        """Test creating zone parameterization."""
        strategy = ZoneParameterization(
            param_type=IWFMParameterType.HORIZONTAL_K,
            zones=[1, 2, 3],
            layer=1,
        )
        assert strategy.param_type == IWFMParameterType.HORIZONTAL_K
        assert strategy.zones == [1, 2, 3]
        assert strategy.layer == 1

    def test_generate_parameters_explicit_zones(self):
        """Test generating parameters with explicit zones."""
        strategy = ZoneParameterization(
            param_type=IWFMParameterType.HORIZONTAL_K,
            zones=[1, 2, 3],
            layer=1,
            initial_values=10.0,
            bounds=(0.1, 1000.0),
        )
        params = strategy.generate_parameters(None)

        assert len(params) == 3
        assert all(p.param_type == IWFMParameterType.HORIZONTAL_K for p in params)
        assert all(p.initial_value == 10.0 for p in params)
        assert all(p.layer == 1 for p in params)

    def test_generate_parameters_from_subregions(self, mock_model):
        """Test generating parameters from model subregions."""
        strategy = ZoneParameterization(
            param_type=IWFMParameterType.SPECIFIC_YIELD,
            zones="subregions",
            layer=1,
        )
        params = strategy.generate_parameters(mock_model)

        assert len(params) == 3  # 3 subregions in mock

    def test_zone_specific_initial_values(self):
        """Test zone-specific initial values."""
        strategy = ZoneParameterization(
            param_type=IWFMParameterType.HORIZONTAL_K,
            zones=[1, 2, 3],
            initial_values={1: 5.0, 2: 10.0, 3: 15.0},
        )
        params = strategy.generate_parameters(None)

        values = {p.zone: p.initial_value for p in params}
        assert values[1] == 5.0
        assert values[2] == 10.0
        assert values[3] == 15.0

    def test_custom_zone_names(self):
        """Test custom zone names in parameter naming."""
        strategy = ZoneParameterization(
            param_type=IWFMParameterType.HORIZONTAL_K,
            zones=[1, 2],
            zone_names={1: "north", 2: "south"},
        )
        params = strategy.generate_parameters(None)

        names = [p.name for p in params]
        assert any("north" in n for n in names)
        assert any("south" in n for n in names)


class TestMultiplierParameterization:
    """Tests for MultiplierParameterization strategy."""

    def test_global_multiplier(self):
        """Test global multiplier parameter."""
        strategy = MultiplierParameterization(
            param_type=IWFMParameterType.PUMPING_MULT,
            spatial_extent="global",
            temporal_extent="constant",
        )
        params = strategy.generate_parameters(None)

        assert len(params) == 1
        assert params[0].initial_value == 1.0

    def test_seasonal_multipliers(self):
        """Test seasonal multiplier parameters."""
        strategy = MultiplierParameterization(
            param_type=IWFMParameterType.ET_MULT,
            spatial_extent="global",
            temporal_extent="seasonal",
        )
        params = strategy.generate_parameters(None)

        assert len(params) == 4  # One per season
        assert any("winter" in p.name for p in params)
        assert any("summer" in p.name for p in params)

    def test_monthly_multipliers(self):
        """Test monthly multiplier parameters."""
        strategy = MultiplierParameterization(
            param_type=IWFMParameterType.PRECIP_MULT,
            spatial_extent="global",
            temporal_extent="monthly",
        )
        params = strategy.generate_parameters(None)

        assert len(params) == 12  # One per month

    def test_zone_multipliers(self):
        """Test zone-based multiplier parameters."""
        strategy = MultiplierParameterization(
            param_type=IWFMParameterType.RECHARGE_MULT,
            spatial_extent="zone",
            temporal_extent="constant",
            zones=[1, 2, 3],
        )
        params = strategy.generate_parameters(None)

        assert len(params) == 3

    def test_invalid_spatial_extent(self):
        """Test that invalid spatial extent raises error."""
        with pytest.raises(ValueError, match="Invalid spatial_extent"):
            MultiplierParameterization(
                param_type=IWFMParameterType.PUMPING_MULT,
                spatial_extent="invalid",
            )

    def test_invalid_temporal_extent(self):
        """Test that invalid temporal extent raises error."""
        with pytest.raises(ValueError, match="Invalid temporal_extent"):
            MultiplierParameterization(
                param_type=IWFMParameterType.PUMPING_MULT,
                temporal_extent="invalid",
            )


class TestPilotPointParameterization:
    """Tests for PilotPointParameterization strategy."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model with grid."""
        model = MagicMock()
        # Create node coordinates for a 10x10 grid
        x = np.linspace(0, 1000, 10)
        y = np.linspace(0, 1000, 10)
        xx, yy = np.meshgrid(x, y)
        model.grid.node_coordinates = np.column_stack([xx.ravel(), yy.ravel()])
        return model

    def test_explicit_points(self):
        """Test pilot points with explicit coordinates."""
        points = [(0, 0), (100, 0), (0, 100), (100, 100)]
        strategy = PilotPointParameterization(
            param_type=IWFMParameterType.HORIZONTAL_K,
            points=points,
            layer=1,
        )
        params = strategy.generate_parameters(None)

        assert len(params) == 4
        assert all(p.location is not None for p in params)

    def test_grid_generation(self, mock_model):
        """Test pilot point grid generation."""
        strategy = PilotPointParameterization(
            param_type=IWFMParameterType.HORIZONTAL_K,
            spacing=500.0,
            layer=1,
        )
        params = strategy.generate_parameters(mock_model)

        # Should have multiple pilot points
        assert len(params) > 1
        assert all(p.location is not None for p in params)

    def test_must_specify_points_or_spacing(self):
        """Test that either points or spacing must be specified."""
        with pytest.raises(ValueError, match="Must specify either"):
            PilotPointParameterization(
                param_type=IWFMParameterType.HORIZONTAL_K,
                layer=1,
            )

    def test_variogram_in_metadata(self):
        """Test that variogram info is stored in metadata."""
        variogram = {"type": "exponential", "a": 10000, "sill": 1.0}
        strategy = PilotPointParameterization(
            param_type=IWFMParameterType.HORIZONTAL_K,
            points=[(0, 0), (100, 100)],
            layer=1,
            variogram=variogram,
        )
        params = strategy.generate_parameters(None)

        assert params[0].metadata["variogram"] == variogram


class TestDirectParameterization:
    """Tests for DirectParameterization strategy."""

    def test_single_parameter(self):
        """Test creating a single direct parameter."""
        strategy = DirectParameterization(
            param_type=IWFMParameterType.GHB_CONDUCTANCE,
            name="ghb_east",
            initial_value=100.0,
        )
        params = strategy.generate_parameters(None)

        assert len(params) == 1
        assert params[0].name == "ghb_east"
        assert params[0].initial_value == 100.0


class TestStreamParameterization:
    """Tests for StreamParameterization strategy."""

    def test_explicit_reaches(self):
        """Test stream parameters with explicit reaches."""
        strategy = StreamParameterization(
            param_type=IWFMParameterType.STREAMBED_K,
            reaches=[1, 2, 3, 4, 5],
        )
        params = strategy.generate_parameters(None)

        assert len(params) == 5
        assert all("strk" in p.name for p in params)

    def test_reach_specific_values(self):
        """Test reach-specific initial values."""
        strategy = StreamParameterization(
            param_type=IWFMParameterType.STREAMBED_K,
            reaches=[1, 2],
            initial_values={1: 0.1, 2: 1.0},
        )
        params = strategy.generate_parameters(None)

        # Parameter names are like "strk_r1", "strk_r2"
        values = {}
        for p in params:
            # Extract reach ID from metadata or name
            if "reach_id" in p.metadata:
                reach_id = p.metadata["reach_id"]
            else:
                # Parse from name: "strk_r1" -> 1
                reach_id = int(p.name.split("_r")[1])
            values[reach_id] = p.initial_value

        assert values[1] == 0.1
        assert values[2] == 1.0


class TestRootZoneParameterization:
    """Tests for RootZoneParameterization strategy."""

    def test_default_land_use_types(self):
        """Test with default land use types."""
        strategy = RootZoneParameterization(
            param_type=IWFMParameterType.CROP_COEFFICIENT,
            land_use_types="all",
        )
        params = strategy.generate_parameters(None)

        # Should have default land use types
        assert len(params) >= 1

    def test_explicit_land_use_types(self):
        """Test with explicit land use types."""
        strategy = RootZoneParameterization(
            param_type=IWFMParameterType.IRRIGATION_EFFICIENCY,
            land_use_types=["corn", "alfalfa", "orchard"],
        )
        params = strategy.generate_parameters(None)

        assert len(params) == 3

    def test_land_use_specific_values(self):
        """Test land use-specific initial values."""
        strategy = RootZoneParameterization(
            param_type=IWFMParameterType.CROP_COEFFICIENT,
            land_use_types=["corn", "wheat"],
            initial_values={"corn": 1.1, "wheat": 0.9},
        )
        params = strategy.generate_parameters(None)

        # Find corn parameter
        corn_param = next(p for p in params if "corn" in p.name)
        assert corn_param.initial_value == 1.1


# ---------------------------------------------------------------------------
# Additional tests to increase coverage beyond 88%
# ---------------------------------------------------------------------------

from pyiwfm.runner.pest_params import ParameterizationStrategy  # noqa: E402


class TestParameterTiedAndPestLine:
    """Cover tied parameter pest_line output and partrans branches."""

    def test_tied_pest_line_uses_tied_to(self):
        """to_pest_line uses tied_to name when set."""
        param = Parameter(
            name="child",
            initial_value=1.0,
            lower_bound=0.1,
            upper_bound=10.0,
            transform=ParameterTransform.TIED,
            tied_to="parent",
            tied_ratio=0.5,
        )
        line = param.to_pest_line()
        assert "child" in line
        assert "tied" in line

    def test_untied_pest_line_uses_own_name(self):
        """to_pest_line uses own name when tied_to is None."""
        param = Parameter(
            name="standalone",
            initial_value=5.0,
            lower_bound=1.0,
            upper_bound=10.0,
        )
        line = param.to_pest_line()
        assert "standalone" in line

    def test_partrans_none(self):
        """partrans returns 'none' for ParameterTransform.NONE."""
        param = Parameter(
            name="test_none",
            initial_value=1.0,
            lower_bound=0.0,
            upper_bound=10.0,
            transform=ParameterTransform.NONE,
        )
        assert param.partrans == "none"

    def test_partrans_fixed(self):
        """partrans returns 'fixed' for ParameterTransform.FIXED."""
        param = Parameter(
            name="test_fixed",
            initial_value=1.0,
            lower_bound=0.0,
            upper_bound=10.0,
            transform=ParameterTransform.FIXED,
        )
        assert param.partrans == "fixed"

    def test_parval1(self):
        """parval1 returns initial_value."""
        param = Parameter(
            name="pv",
            initial_value=3.14,
            lower_bound=1.0,
            upper_bound=5.0,
        )
        assert param.parval1 == 3.14

    def test_parlbnd(self):
        """parlbnd returns lower_bound."""
        param = Parameter(
            name="lb",
            initial_value=2.0,
            lower_bound=1.0,
            upper_bound=5.0,
        )
        assert param.parlbnd == 1.0

    def test_parubnd(self):
        """parubnd returns upper_bound."""
        param = Parameter(
            name="ub",
            initial_value=2.0,
            lower_bound=1.0,
            upper_bound=5.0,
        )
        assert param.parubnd == 5.0


class TestParameterFixedNotClipped:
    """Fixed transform parameters skip initial value bound clipping."""

    def test_fixed_parameter_allows_out_of_bounds_initial(self):
        """Fixed parameter initial value is not clipped to bounds."""
        param = Parameter(
            name="fixed_oob",
            initial_value=100.0,
            lower_bound=0.0,
            upper_bound=10.0,
            transform=ParameterTransform.FIXED,
        )
        # Fixed params skip the clipping logic
        assert param.initial_value == 100.0

    def test_initial_clipped_below_lower(self):
        """Initial value below lower bound is clipped up."""
        param = Parameter(
            name="clip_low",
            initial_value=-5.0,
            lower_bound=0.0,
            upper_bound=10.0,
        )
        assert param.initial_value == 0.0


class TestParameterizationStrategyBase:
    """Cover base ParameterizationStrategy class."""

    def test_generate_parameters_not_implemented(self):
        """Base generate_parameters raises NotImplementedError."""
        strategy = ParameterizationStrategy(
            param_type=IWFMParameterType.HORIZONTAL_K,
        )
        with pytest.raises(NotImplementedError, match="Subclasses must implement"):
            strategy.generate_parameters(None)

    def test_default_bounds_from_type(self):
        """Bounds default to param_type.default_bounds."""
        strategy = ParameterizationStrategy(
            param_type=IWFMParameterType.HORIZONTAL_K,
        )
        assert strategy.bounds == IWFMParameterType.HORIZONTAL_K.default_bounds

    def test_default_group_name_from_type(self):
        """group_name defaults to param_type.value."""
        strategy = ParameterizationStrategy(
            param_type=IWFMParameterType.SPECIFIC_YIELD,
        )
        assert strategy.group_name == "sy"

    def test_string_transform_converted(self):
        """String transform is converted to ParameterTransform enum."""
        strategy = ParameterizationStrategy(
            param_type=IWFMParameterType.PUMPING_MULT,
            transform="log",
        )
        assert strategy.transform == ParameterTransform.LOG

    def test_default_transform_from_param_type(self):
        """When transform is NONE, defaults to param_type.default_transform."""
        strategy = ParameterizationStrategy(
            param_type=IWFMParameterType.HORIZONTAL_K,
            transform=ParameterTransform.NONE,
        )
        # HK default is log
        assert strategy.transform == ParameterTransform.LOG


class TestZoneParameterizationEdgeCases:
    """Cover zone parameterization uncovered paths."""

    def test_get_zone_ids_all_keyword(self):
        """'all' zones delegates to _get_zone_ids_subregions."""
        model = MagicMock()
        model.grid.subregions = {10: MagicMock(), 20: MagicMock()}
        strategy = ZoneParameterization(
            param_type=IWFMParameterType.HORIZONTAL_K,
            zones="all",
        )
        params = strategy.generate_parameters(model)
        assert len(params) == 2

    def test_get_zone_ids_subregions_no_grid(self):
        """_get_zone_ids_subregions returns empty if no grid."""
        model = MagicMock(spec=[])  # no attributes
        strategy = ZoneParameterization(
            param_type=IWFMParameterType.HORIZONTAL_K,
            zones="all",
        )
        params = strategy.generate_parameters(model)
        assert len(params) == 0

    def test_get_zone_ids_from_model_subregions_attr(self):
        """Zone IDs from model.subregions when model.grid unavailable."""
        model = MagicMock(spec=["subregions"])
        model.subregions = {5: "a", 6: "b"}
        strategy = ZoneParameterization(
            param_type=IWFMParameterType.SPECIFIC_YIELD,
            zones="subregions",
        )
        params = strategy.generate_parameters(model)
        assert len(params) == 2

    def test_subregions_model_no_info_raises(self):
        """ValueError raised when model has no subregion info."""
        model = MagicMock(spec=[])
        strategy = ZoneParameterization(
            param_type=IWFMParameterType.HORIZONTAL_K,
            zones="subregions",
        )
        with pytest.raises(ValueError, match="does not have subregion"):
            strategy.generate_parameters(model)

    def test_zone_without_layer(self):
        """Parameter name omits layer suffix when layer is None."""
        strategy = ZoneParameterization(
            param_type=IWFMParameterType.SPECIFIC_YIELD,
            zones=[1],
            layer=None,
        )
        params = strategy.generate_parameters(None)
        assert "_l" not in params[0].name

    def test_get_initial_value_missing_zone_in_dict(self):
        """Missing zone in dict falls back to default."""
        strategy = ZoneParameterization(
            param_type=IWFMParameterType.HORIZONTAL_K,
            zones=[1, 99],
            initial_values={1: 5.0},
        )
        params = strategy.generate_parameters(None)
        p99 = [p for p in params if p.zone == 99][0]
        assert p99.initial_value == 1.0  # default fallback

    def test_get_zone_name_default(self):
        """Default zone naming when zone_names is None."""
        strategy = ZoneParameterization(
            param_type=IWFMParameterType.HORIZONTAL_K,
            zones=[7],
        )
        params = strategy.generate_parameters(None)
        assert "z7" in params[0].name


class TestMultiplierParameterizationEdgeCases:
    """Cover multiplier parameterization uncovered paths."""

    def test_zone_multiplier_from_model(self):
        """Zone multiplier gets zone IDs from model grid."""
        model = MagicMock()
        model.grid.subregions = {1: "a", 2: "b"}
        strategy = MultiplierParameterization(
            param_type=IWFMParameterType.RECHARGE_MULT,
            spatial_extent="zone",
            zones=None,
        )
        params = strategy.generate_parameters(model)
        assert len(params) == 2

    def test_zone_multiplier_no_model_info(self):
        """Zone multiplier defaults to single zone when model has no grid."""
        model = MagicMock(spec=[])
        strategy = MultiplierParameterization(
            param_type=IWFMParameterType.RECHARGE_MULT,
            spatial_extent="zone",
            zones=None,
        )
        params = strategy.generate_parameters(model)
        assert len(params) == 1

    def test_element_multiplier_from_grid(self):
        """Element multiplier reads n_elements from model.grid."""
        model = MagicMock()
        model.grid.n_elements = 3
        strategy = MultiplierParameterization(
            param_type=IWFMParameterType.PUMPING_MULT,
            spatial_extent="element",
        )
        params = strategy.generate_parameters(model)
        assert len(params) == 3
        assert "_e1" in params[0].name

    def test_element_multiplier_from_n_elements_attr(self):
        """Element multiplier reads n_elements directly from model."""
        model = MagicMock(spec=["n_elements"])
        model.n_elements = 2
        strategy = MultiplierParameterization(
            param_type=IWFMParameterType.PUMPING_MULT,
            spatial_extent="element",
        )
        params = strategy.generate_parameters(model)
        assert len(params) == 2

    def test_element_multiplier_default_one_element(self):
        """Element multiplier defaults to 1 element when model has no info."""
        model = MagicMock(spec=[])
        strategy = MultiplierParameterization(
            param_type=IWFMParameterType.PUMPING_MULT,
            spatial_extent="element",
        )
        params = strategy.generate_parameters(model)
        assert len(params) == 1

    def test_annual_temporal_extent(self):
        """Annual temporal extent produces _annual suffix."""
        strategy = MultiplierParameterization(
            param_type=IWFMParameterType.PRECIP_MULT,
            spatial_extent="global",
            temporal_extent="annual",
        )
        params = strategy.generate_parameters(None)
        assert len(params) == 1
        assert "_annual" in params[0].name

    def test_element_seasonal_multiplier(self):
        """Element x seasonal produces n_elements * 4 params."""
        model = MagicMock()
        model.grid.n_elements = 2
        strategy = MultiplierParameterization(
            param_type=IWFMParameterType.PUMPING_MULT,
            spatial_extent="element",
            temporal_extent="seasonal",
        )
        params = strategy.generate_parameters(model)
        assert len(params) == 8  # 2 elements * 4 seasons


class TestPilotPointParameterizationEdgeCases:
    """Cover pilot point parameterization uncovered paths."""

    def test_initial_value_from_ndarray(self):
        """Initial values from numpy array."""
        import numpy as np

        strategy = PilotPointParameterization(
            param_type=IWFMParameterType.HORIZONTAL_K,
            points=[(0, 0), (100, 0), (200, 0)],
            layer=1,
            initial_value=np.array([1.0, 2.0, 3.0]),
        )
        params = strategy.generate_parameters(None)
        assert params[0].initial_value == 1.0
        assert params[1].initial_value == 2.0
        assert params[2].initial_value == 3.0

    def test_initial_value_ndarray_index_out_of_range(self):
        """When ndarray index exceeds length, falls back to first element."""
        import numpy as np

        strategy = PilotPointParameterization(
            param_type=IWFMParameterType.HORIZONTAL_K,
            points=[(0, 0), (100, 0), (200, 0)],
            layer=1,
            initial_value=np.array([5.0]),  # only 1 element
        )
        params = strategy.generate_parameters(None)
        assert params[0].initial_value == 5.0
        # index 1 >= len(1) => falls to last branch => initial_value[0]
        assert params[1].initial_value == 5.0

    def test_generate_grid_no_spacing_raises(self):
        """generate_pilot_point_grid raises without spacing."""
        strategy = PilotPointParameterization(
            param_type=IWFMParameterType.HORIZONTAL_K,
            points=[(0, 0)],
            layer=1,
        )
        strategy.spacing = None
        with pytest.raises(ValueError, match="Spacing must be set"):
            strategy.generate_pilot_point_grid(None)

    def test_generate_grid_from_node_coordinates(self):
        """generate_pilot_point_grid from model.node_coordinates."""
        model = MagicMock(spec=["node_coordinates"])
        model.node_coordinates = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]])
        strategy = PilotPointParameterization(
            param_type=IWFMParameterType.HORIZONTAL_K,
            spacing=50.0,
            layer=1,
        )
        points = strategy.generate_pilot_point_grid(model)
        assert len(points) > 0

    def test_generate_grid_model_no_extent_raises(self):
        """generate_pilot_point_grid raises when model has no grid info."""
        model = MagicMock(spec=[])
        strategy = PilotPointParameterization(
            param_type=IWFMParameterType.HORIZONTAL_K,
            spacing=50.0,
            layer=1,
        )
        with pytest.raises(ValueError, match="Cannot determine model extent"):
            strategy.generate_pilot_point_grid(model)

    def test_generate_parameters_from_grid(self):
        """generate_parameters uses grid when points is None."""
        model = MagicMock()
        model.grid.node_coordinates = np.array(
            [[0.0, 0.0], [200.0, 0.0], [0.0, 200.0], [200.0, 200.0]]
        )
        strategy = PilotPointParameterization(
            param_type=IWFMParameterType.HORIZONTAL_K,
            spacing=100.0,
            layer=1,
        )
        params = strategy.generate_parameters(model)
        assert len(params) > 0
        assert all(p.location is not None for p in params)


class TestDirectParameterizationEdgeCases:
    """Cover direct parameterization uncovered paths."""

    def test_default_name_from_param_type(self):
        """Default name comes from param_type.value."""
        strategy = DirectParameterization(
            param_type=IWFMParameterType.GHB_CONDUCTANCE,
            initial_value=10.0,
        )
        params = strategy.generate_parameters(None)
        assert params[0].name == "ghbc"

    def test_with_location_id(self):
        """Location ID is stored in metadata."""
        strategy = DirectParameterization(
            param_type=IWFMParameterType.SPECIFIED_HEAD,
            name="chd_north",
            initial_value=100.0,
            location_id=42,
        )
        params = strategy.generate_parameters(None)
        assert params[0].metadata["location_id"] == 42

    def test_without_location_id(self):
        """No location_id produces empty metadata."""
        strategy = DirectParameterization(
            param_type=IWFMParameterType.SPECIFIED_HEAD,
            name="chd_south",
            initial_value=50.0,
        )
        params = strategy.generate_parameters(None)
        assert params[0].metadata == {}


class TestStreamParameterizationEdgeCases:
    """Cover stream parameterization uncovered paths."""

    def test_reach_ids_from_model_streams(self):
        """Reach IDs from model.streams.reaches."""
        model = MagicMock()
        model.streams.reaches = {1: "a", 3: "b", 5: "c"}
        strategy = StreamParameterization(
            param_type=IWFMParameterType.STREAMBED_K,
            reaches="all",
        )
        params = strategy.generate_parameters(model)
        assert len(params) == 3

    def test_reach_ids_from_model_reach_ids(self):
        """Reach IDs from model.reach_ids when streams unavailable."""
        model = MagicMock(spec=["reach_ids"])
        model.reach_ids = [10, 20]
        strategy = StreamParameterization(
            param_type=IWFMParameterType.STREAMBED_K,
            reaches="all",
        )
        params = strategy.generate_parameters(model)
        assert len(params) == 2

    def test_reach_ids_default_single(self):
        """Defaults to [1] when model has no reach info."""
        model = MagicMock(spec=[])
        strategy = StreamParameterization(
            param_type=IWFMParameterType.STREAMBED_K,
            reaches="all",
        )
        params = strategy.generate_parameters(model)
        assert len(params) == 1

    def test_initial_value_missing_reach_in_dict(self):
        """Default fallback when reach not in dict."""
        strategy = StreamParameterization(
            param_type=IWFMParameterType.STREAMBED_K,
            reaches=[1, 99],
            initial_values={1: 0.5},
        )
        params = strategy.generate_parameters(None)
        p99 = [p for p in params if p.metadata.get("reach_id") == 99][0]
        assert p99.initial_value == 1.0


class TestRootZoneParameterizationEdgeCases:
    """Cover root zone parameterization uncovered paths."""

    def test_crop_ids_explicit(self):
        """Explicit crop_ids generate cropN names."""
        strategy = RootZoneParameterization(
            param_type=IWFMParameterType.CROP_COEFFICIENT,
            crop_ids=[10, 20, 30],
        )
        params = strategy.generate_parameters(None)
        assert len(params) == 3
        assert any("crop10" in p.metadata.get("land_use_type", "") for p in params)

    def test_land_use_from_model(self):
        """Land use types from model.rootzone.crop_types."""
        model = MagicMock()
        model.rootzone.crop_types = {1: "alfalfa", 2: "orchard"}
        strategy = RootZoneParameterization(
            param_type=IWFMParameterType.CROP_COEFFICIENT,
            land_use_types="all",
        )
        params = strategy.generate_parameters(model)
        assert len(params) == 2

    def test_land_use_default_when_no_model_info(self):
        """Default 4 land use types when model has no rootzone."""
        model = MagicMock(spec=[])
        strategy = RootZoneParameterization(
            param_type=IWFMParameterType.CROP_COEFFICIENT,
            land_use_types="all",
        )
        params = strategy.generate_parameters(model)
        assert len(params) == 4  # urban, agriculture, native, riparian

    def test_initial_value_missing_landuse_in_dict(self):
        """Missing land use name falls back to 1.0."""
        strategy = RootZoneParameterization(
            param_type=IWFMParameterType.CROP_COEFFICIENT,
            land_use_types=["corn", "unknown"],
            initial_values={"corn": 1.2},
        )
        params = strategy.generate_parameters(None)
        unknown_param = [p for p in params if "unknown" in p.name][0]
        assert unknown_param.initial_value == 1.0


class TestIWFMParameterTypeEdgeCases:
    """Cover additional parameter type property branches."""

    def test_default_bounds_unknown_type(self):
        """Fallback default bounds for types not in bounds_map (coverage)."""
        # All types are in the map, but let's ensure accessing works
        for pt in IWFMParameterType:
            lb, ub = pt.default_bounds
            assert lb < ub

    def test_default_transform_log_types(self):
        """All expected log-transform types return 'log'."""
        log_types = [
            IWFMParameterType.HORIZONTAL_K,
            IWFMParameterType.VERTICAL_K,
            IWFMParameterType.SPECIFIC_STORAGE,
            IWFMParameterType.STREAMBED_K,
            IWFMParameterType.LAKEBED_K,
            IWFMParameterType.GHB_CONDUCTANCE,
            IWFMParameterType.ELASTIC_STORAGE,
            IWFMParameterType.INELASTIC_STORAGE,
        ]
        for pt in log_types:
            assert pt.default_transform == "log", f"{pt} should be log"

    def test_default_transform_none_types(self):
        """Non-log types return 'none'."""
        assert IWFMParameterType.SPECIFIC_YIELD.default_transform == "none"
        assert IWFMParameterType.MANNING_N.default_transform == "none"

    def test_is_multiplier_all(self):
        """Check is_multiplier for all multiplier types."""
        mult_types = [
            IWFMParameterType.PUMPING_MULT,
            IWFMParameterType.RECHARGE_MULT,
            IWFMParameterType.DIVERSION_MULT,
            IWFMParameterType.BYPASS_MULT,
            IWFMParameterType.PRECIP_MULT,
            IWFMParameterType.ET_MULT,
            IWFMParameterType.STREAM_INFLOW_MULT,
            IWFMParameterType.RETURN_FLOW_MULT,
        ]
        for pt in mult_types:
            assert pt.is_multiplier is True, f"{pt} should be multiplier"

    def test_is_not_multiplier(self):
        """Non-multiplier types return False."""
        assert IWFMParameterType.POROSITY.is_multiplier is False
        assert IWFMParameterType.PRECONSOLIDATION.is_multiplier is False


class TestParameterGroupEdgeCases:
    """Cover parameter group edge cases."""

    def test_custom_fields(self):
        """Test all custom fields of ParameterGroup."""
        group = ParameterGroup(
            name="mygroup",
            inctyp="absolute",
            derinc=0.05,
            derinclb=0.001,
            forcen="always_2",
            derincmul=3.0,
            dermthd="best_fit",
            splitthresh=1e-4,
            splitreldiff=0.3,
        )
        assert group.inctyp == "absolute"
        assert group.derinc == 0.05
        assert group.forcen == "always_2"
        assert group.dermthd == "best_fit"

    def test_pest_line_format(self):
        """Pest line includes all core fields."""
        group = ParameterGroup(name="hk", derinc=0.02, forcen="always_3")
        line = group.to_pest_line()
        assert "hk" in line
        assert "always_3" in line
