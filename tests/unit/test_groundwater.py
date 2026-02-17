"""Unit tests for groundwater component classes."""

from __future__ import annotations

import numpy as np
import pytest

from pyiwfm.components.groundwater import (
    AppGW,
    AquiferParameters,
    BoundaryCondition,
    ElementPumping,
    Subsidence,
    TileDrain,
    Well,
)
from pyiwfm.core.exceptions import ComponentError


class TestWell:
    """Tests for well class."""

    def test_well_creation(self) -> None:
        """Test basic well creation."""
        well = Well(
            id=1,
            name="Production Well 1",
            x=1000.0,
            y=2000.0,
            element=5,
            top_screen=50.0,
            bottom_screen=100.0,
        )

        assert well.id == 1
        assert well.name == "Production Well 1"
        assert well.x == 1000.0
        assert well.y == 2000.0
        assert well.element == 5
        assert well.top_screen == 50.0
        assert well.bottom_screen == 100.0

    def test_well_default_values(self) -> None:
        """Test well default values."""
        well = Well(id=1, x=0.0, y=0.0, element=1)

        assert well.name == ""
        assert well.top_screen == 0.0
        assert well.bottom_screen == 0.0
        assert well.max_pump_rate == float("inf")
        assert well.pump_rate == 0.0

    def test_well_screen_depth(self) -> None:
        """Test well screen depth calculation."""
        well = Well(
            id=1,
            x=0.0,
            y=0.0,
            element=1,
            top_screen=50.0,
            bottom_screen=150.0,
        )

        assert well.screen_length == 100.0

    def test_well_layers(self) -> None:
        """Test well layer assignment."""
        well = Well(
            id=1,
            x=0.0,
            y=0.0,
            element=1,
            layers=[1, 2],  # Well screens across layers 1 and 2
        )

        assert well.layers == [1, 2]
        assert well.n_layers == 2

    def test_well_equality(self) -> None:
        """Test well equality."""
        well1 = Well(id=1, x=100.0, y=200.0, element=1)
        well2 = Well(id=1, x=100.0, y=200.0, element=1)
        well3 = Well(id=2, x=100.0, y=200.0, element=1)

        assert well1 == well2
        assert well1 != well3


class TestElementPumping:
    """Tests for element pumping class."""

    def test_element_pumping_creation(self) -> None:
        """Test basic element pumping creation."""
        pumping = ElementPumping(
            element_id=10,
            layer=1,
            pump_rate=500.0,
        )

        assert pumping.element_id == 10
        assert pumping.layer == 1
        assert pumping.pump_rate == 500.0

    def test_element_pumping_fraction(self) -> None:
        """Test element pumping with layer fractions."""
        pumping = ElementPumping(
            element_id=10,
            layer=1,
            pump_rate=1000.0,
            layer_fraction=0.6,
        )

        assert pumping.effective_rate == pytest.approx(600.0)


class TestBoundaryCondition:
    """Tests for boundary condition class."""

    def test_specified_head_bc(self) -> None:
        """Test specified head boundary condition."""
        bc = BoundaryCondition(
            id=1,
            bc_type="specified_head",
            nodes=[1, 2, 3],
            values=[100.0, 100.0, 100.0],
            layer=1,
        )

        assert bc.bc_type == "specified_head"
        assert bc.nodes == [1, 2, 3]
        assert len(bc.values) == 3

    def test_specified_flow_bc(self) -> None:
        """Test specified flow boundary condition."""
        bc = BoundaryCondition(
            id=1,
            bc_type="specified_flow",
            nodes=[10],
            values=[-100.0],  # Negative = extraction
            layer=1,
        )

        assert bc.bc_type == "specified_flow"
        assert bc.values[0] == -100.0

    def test_general_head_bc(self) -> None:
        """Test general head boundary condition."""
        bc = BoundaryCondition(
            id=1,
            bc_type="general_head",
            nodes=[5],
            values=[50.0],  # Reference head
            conductance=[10.0],  # Boundary conductance
            layer=1,
        )

        assert bc.bc_type == "general_head"
        assert bc.conductance == [10.0]


class TestTileDrain:
    """Tests for tile drain class."""

    def test_tile_drain_creation(self) -> None:
        """Test basic tile drain creation."""
        drain = TileDrain(
            id=1,
            element=5,
            elevation=90.0,
            conductance=100.0,
        )

        assert drain.id == 1
        assert drain.element == 5
        assert drain.elevation == 90.0
        assert drain.conductance == 100.0

    def test_tile_drain_destination(self) -> None:
        """Test tile drain with stream destination."""
        drain = TileDrain(
            id=1,
            element=5,
            elevation=90.0,
            conductance=100.0,
            destination_type="stream",
            destination_id=3,  # Stream node
        )

        assert drain.destination_type == "stream"
        assert drain.destination_id == 3


class TestSubsidence:
    """Tests for subsidence parameters."""

    def test_subsidence_creation(self) -> None:
        """Test basic subsidence parameter creation."""
        sub = Subsidence(
            element=1,
            layer=1,
            elastic_storage=0.0001,
            inelastic_storage=0.001,
            preconsolidation_head=50.0,
        )

        assert sub.element == 1
        assert sub.layer == 1
        assert sub.elastic_storage == 0.0001
        assert sub.inelastic_storage == 0.001
        assert sub.preconsolidation_head == 50.0


class TestAquiferParameters:
    """Tests for aquifer parameters."""

    def test_aquifer_params_creation(self) -> None:
        """Test aquifer parameter creation."""
        params = AquiferParameters(
            n_nodes=100,
            n_layers=2,
        )

        assert params.n_nodes == 100
        assert params.n_layers == 2

    def test_aquifer_params_with_arrays(self) -> None:
        """Test aquifer parameters with data arrays."""
        n_nodes = 4
        n_layers = 2

        kh = np.ones((n_nodes, n_layers)) * 10.0  # Horizontal K
        kv = np.ones((n_nodes, n_layers)) * 1.0  # Vertical K
        ss = np.ones((n_nodes, n_layers)) * 0.0001  # Specific storage
        sy = np.ones((n_nodes, n_layers)) * 0.1  # Specific yield

        params = AquiferParameters(
            n_nodes=n_nodes,
            n_layers=n_layers,
            kh=kh,
            kv=kv,
            specific_storage=ss,
            specific_yield=sy,
        )

        assert params.kh.shape == (4, 2)
        assert params.kh[0, 0] == 10.0
        assert params.specific_yield[0, 0] == 0.1

    def test_aquifer_params_get_layer(self) -> None:
        """Test getting parameters for a specific layer."""
        n_nodes = 4
        n_layers = 2

        kh = np.array([[10.0, 5.0]] * n_nodes)
        params = AquiferParameters(
            n_nodes=n_nodes,
            n_layers=n_layers,
            kh=kh,
        )

        layer_kh = params.get_layer_kh(0)
        assert len(layer_kh) == n_nodes
        np.testing.assert_array_equal(layer_kh, 10.0)


class TestAppGW:
    """Tests for groundwater application class."""

    def test_appgw_creation(self) -> None:
        """Test basic groundwater component creation."""
        gw = AppGW(n_nodes=100, n_layers=2, n_elements=50)

        assert gw.n_nodes == 100
        assert gw.n_layers == 2
        assert gw.n_elements == 50

    def test_appgw_add_well(self) -> None:
        """Test adding wells."""
        gw = AppGW(n_nodes=10, n_layers=2, n_elements=5)

        well = Well(id=1, x=100.0, y=200.0, element=1)
        gw.add_well(well)

        assert gw.n_wells == 1
        assert gw.get_well(1) == well

    def test_appgw_add_boundary_condition(self) -> None:
        """Test adding boundary conditions."""
        gw = AppGW(n_nodes=10, n_layers=2, n_elements=5)

        bc = BoundaryCondition(
            id=1,
            bc_type="specified_head",
            nodes=[1, 2],
            values=[100.0, 100.0],
            layer=1,
        )
        gw.add_boundary_condition(bc)

        assert gw.n_boundary_conditions == 1

    def test_appgw_add_tile_drain(self) -> None:
        """Test adding tile drains."""
        gw = AppGW(n_nodes=10, n_layers=2, n_elements=5)

        drain = TileDrain(id=1, element=1, elevation=90.0, conductance=100.0)
        gw.add_tile_drain(drain)

        assert gw.n_tile_drains == 1

    def test_appgw_set_aquifer_params(self) -> None:
        """Test setting aquifer parameters."""
        n_nodes = 10
        n_layers = 2

        gw = AppGW(n_nodes=n_nodes, n_layers=n_layers, n_elements=5)

        params = AquiferParameters(
            n_nodes=n_nodes,
            n_layers=n_layers,
            kh=np.ones((n_nodes, n_layers)) * 10.0,
        )
        gw.set_aquifer_parameters(params)

        assert gw.aquifer_params is not None
        assert gw.aquifer_params.kh[0, 0] == 10.0

    def test_appgw_heads(self) -> None:
        """Test head array management."""
        gw = AppGW(n_nodes=10, n_layers=2, n_elements=5)

        # Set initial heads
        heads = np.ones((10, 2)) * 100.0
        gw.set_heads(heads)

        assert gw.heads.shape == (10, 2)
        assert gw.heads[0, 0] == 100.0

    def test_appgw_get_head_at_node(self) -> None:
        """Test getting head at specific node and layer."""
        gw = AppGW(n_nodes=10, n_layers=2, n_elements=5)

        heads = np.arange(20).reshape(10, 2).astype(float)
        gw.set_heads(heads)

        assert gw.get_head(node=0, layer=0) == 0.0
        assert gw.get_head(node=0, layer=1) == 1.0
        assert gw.get_head(node=5, layer=0) == 10.0

    def test_appgw_get_wells_in_element(self) -> None:
        """Test getting wells in an element."""
        gw = AppGW(n_nodes=10, n_layers=2, n_elements=5)

        gw.add_well(Well(id=1, x=100.0, y=100.0, element=1))
        gw.add_well(Well(id=2, x=150.0, y=100.0, element=1))
        gw.add_well(Well(id=3, x=200.0, y=200.0, element=2))

        wells_in_1 = gw.get_wells_in_element(1)
        assert len(wells_in_1) == 2

        wells_in_2 = gw.get_wells_in_element(2)
        assert len(wells_in_2) == 1

    def test_appgw_total_pumping(self) -> None:
        """Test calculating total pumping."""
        gw = AppGW(n_nodes=10, n_layers=2, n_elements=5)

        gw.add_well(Well(id=1, x=100.0, y=100.0, element=1, pump_rate=100.0))
        gw.add_well(Well(id=2, x=150.0, y=100.0, element=1, pump_rate=200.0))
        gw.add_well(Well(id=3, x=200.0, y=200.0, element=2, pump_rate=150.0))

        total = gw.get_total_pumping()
        assert total == pytest.approx(450.0)

    def test_appgw_validate(self) -> None:
        """Test groundwater component validation."""
        gw = AppGW(n_nodes=10, n_layers=2, n_elements=5)

        # Should pass basic validation
        gw.validate()

    def test_appgw_validate_well_element(self) -> None:
        """Test validation catches invalid well elements."""
        gw = AppGW(n_nodes=10, n_layers=2, n_elements=5)

        # Add well with invalid element
        gw.add_well(Well(id=1, x=100.0, y=100.0, element=100))  # Element doesn't exist

        with pytest.raises(ComponentError, match="element"):
            gw.validate()


class TestAppGWIO:
    """Tests for groundwater I/O operations."""

    def test_appgw_to_arrays(self) -> None:
        """Test converting groundwater data to arrays."""
        gw = AppGW(n_nodes=4, n_layers=2, n_elements=2)

        params = AquiferParameters(
            n_nodes=4,
            n_layers=2,
            kh=np.ones((4, 2)) * 10.0,
            kv=np.ones((4, 2)) * 1.0,
        )
        gw.set_aquifer_parameters(params)

        heads = np.ones((4, 2)) * 100.0
        gw.set_heads(heads)

        arrays = gw.to_arrays()

        assert "heads" in arrays
        assert "kh" in arrays
        np.testing.assert_array_equal(arrays["heads"], heads)

    def test_appgw_from_arrays(self) -> None:
        """Test creating groundwater from arrays."""
        n_nodes = 4
        n_layers = 2
        n_elements = 2

        heads = np.ones((n_nodes, n_layers)) * 100.0
        kh = np.ones((n_nodes, n_layers)) * 10.0

        gw = AppGW.from_arrays(
            n_nodes=n_nodes,
            n_layers=n_layers,
            n_elements=n_elements,
            heads=heads,
            kh=kh,
        )

        assert gw.n_nodes == n_nodes
        np.testing.assert_array_equal(gw.heads, heads)
        np.testing.assert_array_equal(gw.aquifer_params.kh, kh)
