"""Tests for components/groundwater.py validation and conversion paths.

Covers:
- BoundaryCondition.__post_init__() validation (invalid bc_type, nodes/values mismatch, general_head)
- AppGW.validate() well/tile drain/BC validation
- AppGW.to_arrays() conditional branches
- AppGW.from_arrays() with/without params
"""

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


class TestBCInvalidType:
    """Test BoundaryCondition with invalid bc_type."""

    def test_bc_invalid_type(self) -> None:
        """Invalid bc_type -> ValueError."""
        with pytest.raises(ValueError, match="bc_type must be one of"):
            BoundaryCondition(id=1, bc_type="invalid", nodes=[1], values=[10.0], layer=1)


class TestBCNodesValuesMismatch:
    """Test BoundaryCondition nodes/values length mismatch."""

    def test_bc_nodes_values_mismatch(self) -> None:
        """nodes and values different lengths -> ValueError."""
        with pytest.raises(ValueError, match="same length"):
            BoundaryCondition(
                id=1,
                bc_type="specified_head",
                nodes=[1, 2, 3],
                values=[10.0, 20.0],
                layer=1,
            )


class TestBCGeneralHeadValidation:
    """Test general_head BC requires conductance."""

    def test_bc_general_head_missing_conductance(self) -> None:
        """General head BC without conductance -> ValueError."""
        with pytest.raises(ValueError, match="conductance"):
            BoundaryCondition(
                id=1,
                bc_type="general_head",
                nodes=[1, 2],
                values=[10.0, 20.0],
                layer=1,
                conductance=[],  # Empty, should match nodes length
            )

    def test_bc_general_head_valid(self) -> None:
        """General head BC with matching conductance -> OK."""
        bc = BoundaryCondition(
            id=1,
            bc_type="general_head",
            nodes=[1, 2],
            values=[10.0, 20.0],
            layer=1,
            conductance=[0.1, 0.2],
        )
        assert len(bc.conductance) == 2


class TestValidateTileDrainInvalid:
    """Test AppGW.validate() with invalid tile drain."""

    def test_validate_tile_drain_invalid_element(self) -> None:
        """Tile drain with element > n_elements -> ComponentError."""
        gw = AppGW(n_nodes=10, n_layers=2, n_elements=5)
        gw.add_tile_drain(TileDrain(id=1, element=99, elevation=50.0, conductance=0.1))
        with pytest.raises(ComponentError, match="Tile drain.*invalid element"):
            gw.validate()

    def test_validate_tile_drain_zero_element(self) -> None:
        """Tile drain with element < 1 -> ComponentError."""
        gw = AppGW(n_nodes=10, n_layers=2, n_elements=5)
        gw.add_tile_drain(TileDrain(id=1, element=0, elevation=50.0, conductance=0.1))
        with pytest.raises(ComponentError, match="Tile drain.*invalid element"):
            gw.validate()


class TestValidateBCNodeInvalid:
    """Test AppGW.validate() with invalid BC node."""

    def test_validate_bc_node_too_large(self) -> None:
        """BC referencing node > n_nodes -> ComponentError."""
        gw = AppGW(n_nodes=10, n_layers=2, n_elements=5)
        bc = BoundaryCondition(
            id=1,
            bc_type="specified_head",
            nodes=[1, 999],
            values=[10.0, 20.0],
            layer=1,
        )
        gw.add_boundary_condition(bc)
        with pytest.raises(ComponentError, match="invalid node"):
            gw.validate()

    def test_validate_bc_node_zero(self) -> None:
        """BC referencing node 0 -> ComponentError."""
        gw = AppGW(n_nodes=10, n_layers=2, n_elements=5)
        bc = BoundaryCondition(
            id=1,
            bc_type="specified_head",
            nodes=[0],
            values=[10.0],
            layer=1,
        )
        gw.add_boundary_condition(bc)
        with pytest.raises(ComponentError, match="invalid node"):
            gw.validate()


class TestValidateWellInvalid:
    """Test AppGW.validate() with invalid well element."""

    def test_validate_well_invalid_element(self) -> None:
        """Well with element > n_elements -> ComponentError."""
        gw = AppGW(n_nodes=10, n_layers=2, n_elements=5)
        gw.add_well(Well(id=1, x=100.0, y=200.0, element=99))
        with pytest.raises(ComponentError, match="Well.*invalid element"):
            gw.validate()


class TestToArraysWithAquiferParams:
    """Test AppGW.to_arrays() with full aquifer parameters."""

    def test_to_arrays_with_aquifer_params(self) -> None:
        """Full parameter export."""
        gw = AppGW(n_nodes=4, n_layers=2, n_elements=2)
        heads = np.ones((4, 2)) * 100.0
        gw.set_heads(heads)

        kh = np.ones((4, 2)) * 10.0
        kv = np.ones((4, 2)) * 1.0
        ss = np.ones((4, 2)) * 0.001
        sy = np.ones((4, 2)) * 0.1
        params = AquiferParameters(
            n_nodes=4,
            n_layers=2,
            kh=kh,
            kv=kv,
            specific_storage=ss,
            specific_yield=sy,
        )
        gw.set_aquifer_parameters(params)

        arrays = gw.to_arrays()
        assert "heads" in arrays
        assert "kh" in arrays
        assert "kv" in arrays
        assert "specific_storage" in arrays
        assert "specific_yield" in arrays
        np.testing.assert_array_equal(arrays["heads"], heads)

    def test_to_arrays_no_heads(self) -> None:
        """Export without initial heads."""
        gw = AppGW(n_nodes=4, n_layers=2, n_elements=2)
        arrays = gw.to_arrays()
        assert "heads" not in arrays


class TestFromArraysMinimal:
    """Test AppGW.from_arrays() with minimal inputs."""

    def test_from_arrays_minimal(self) -> None:
        """Minimal array construction (no params, no heads)."""
        gw = AppGW.from_arrays(n_nodes=4, n_layers=2, n_elements=2)
        assert gw.n_nodes == 4
        assert gw.n_layers == 2
        assert gw.heads is None
        assert gw.aquifer_params is None

    def test_from_arrays_with_all_params(self) -> None:
        """Full array construction with heads and all params."""
        heads = np.ones((4, 2)) * 100.0
        kh = np.ones((4, 2)) * 10.0

        gw = AppGW.from_arrays(
            n_nodes=4,
            n_layers=2,
            n_elements=2,
            heads=heads,
            kh=kh,
        )
        assert gw.heads is not None
        assert gw.aquifer_params is not None
        assert gw.aquifer_params.kh is not None
        np.testing.assert_array_equal(gw.aquifer_params.kh, kh)


class TestWellDunderMethods:
    """Test Well __eq__, __hash__, __repr__."""

    def test_eq_not_implemented(self) -> None:
        """Comparing Well to non-Well returns NotImplemented."""
        w = Well(id=1, x=0.0, y=0.0, element=1)
        assert w.__eq__("not a well") is NotImplemented

    def test_eq_same(self) -> None:
        """Two wells with same id/x/y are equal."""
        w1 = Well(id=1, x=100.0, y=200.0, element=1)
        w2 = Well(id=1, x=100.0, y=200.0, element=2)
        assert w1 == w2

    def test_hash(self) -> None:
        """Hash is consistent for equal wells."""
        w1 = Well(id=1, x=100.0, y=200.0, element=1)
        w2 = Well(id=1, x=100.0, y=200.0, element=2)
        assert hash(w1) == hash(w2)

    def test_repr(self) -> None:
        """Repr includes id, element, rate."""
        w = Well(id=5, x=0.0, y=0.0, element=3, pump_rate=-100.0)
        r = repr(w)
        assert "Well" in r
        assert "5" in r


class TestElementPumpingRepr:
    """Test ElementPumping.__repr__."""

    def test_repr(self) -> None:
        r = repr(ElementPumping(element_id=10, layer=1, pump_rate=-50.0))
        assert "ElementPumping" in r
        assert "10" in r


class TestBoundaryConditionRepr:
    """Test BoundaryCondition.__repr__."""

    def test_repr(self) -> None:
        bc = BoundaryCondition(
            id=3,
            bc_type="specified_head",
            nodes=[1, 2],
            values=[10.0, 20.0],
            layer=1,
        )
        r = repr(bc)
        assert "BoundaryCondition" in r
        assert "3" in r


class TestTileDrainRepr:
    """Test TileDrain.__repr__."""

    def test_repr(self) -> None:
        td = TileDrain(id=7, element=4, elevation=50.0, conductance=0.1)
        r = repr(td)
        assert "TileDrain" in r
        assert "7" in r


class TestSubsidenceRepr:
    """Test Subsidence.__repr__."""

    def test_repr(self) -> None:
        s = Subsidence(
            element=2,
            layer=1,
            elastic_storage=0.001,
            inelastic_storage=0.01,
            preconsolidation_head=50.0,
        )
        r = repr(s)
        assert "Subsidence" in r
        assert "2" in r


class TestAquiferParametersMethods:
    """Test AquiferParameters get_layer_kh/kv and repr."""

    def test_get_layer_kh_none(self) -> None:
        """kh not set -> ValueError."""
        params = AquiferParameters(n_nodes=4, n_layers=2, kh=None)
        with pytest.raises(ValueError, match="kh not set"):
            params.get_layer_kh(0)

    def test_get_layer_kv_none(self) -> None:
        """kv not set -> ValueError."""
        params = AquiferParameters(n_nodes=4, n_layers=2, kv=None)
        with pytest.raises(ValueError, match="kv not set"):
            params.get_layer_kv(0)

    def test_repr(self) -> None:
        """Repr includes n_nodes and n_layers."""
        params = AquiferParameters(n_nodes=10, n_layers=3)
        r = repr(params)
        assert "AquiferParameters" in r
        assert "10" in r
        assert "3" in r


class TestAppGWAdditionalMethods:
    """Test AppGW additional methods."""

    def test_add_element_pumping(self) -> None:
        """add_element_pumping adds to list."""
        gw = AppGW(n_nodes=4, n_layers=2, n_elements=2)
        ep = ElementPumping(element_id=1, layer=1, pump_rate=-50.0)
        gw.add_element_pumping(ep)
        assert len(gw.element_pumping) == 1

    def test_get_tile_drain(self) -> None:
        """get_tile_drain returns correct drain."""
        gw = AppGW(n_nodes=4, n_layers=2, n_elements=2)
        td = TileDrain(id=3, element=1, elevation=50.0, conductance=0.1)
        gw.add_tile_drain(td)
        result = gw.get_tile_drain(3)
        assert result.id == 3

    def test_set_aquifer_params_node_mismatch(self) -> None:
        """Mismatched n_nodes -> ValueError."""
        gw = AppGW(n_nodes=4, n_layers=2, n_elements=2)
        params = AquiferParameters(n_nodes=10, n_layers=2)
        with pytest.raises(ValueError, match="n_nodes"):
            gw.set_aquifer_parameters(params)

    def test_set_aquifer_params_layer_mismatch(self) -> None:
        """Mismatched n_layers -> ValueError."""
        gw = AppGW(n_nodes=4, n_layers=2, n_elements=2)
        params = AquiferParameters(n_nodes=4, n_layers=5)
        with pytest.raises(ValueError, match="n_layers"):
            gw.set_aquifer_parameters(params)

    def test_set_heads_shape_mismatch(self) -> None:
        """Wrong shape -> ValueError."""
        gw = AppGW(n_nodes=4, n_layers=2, n_elements=2)
        wrong_heads = np.ones((3, 3))
        with pytest.raises(ValueError, match="shape"):
            gw.set_heads(wrong_heads)

    def test_get_head_none(self) -> None:
        """Heads not set -> ValueError."""
        gw = AppGW(n_nodes=4, n_layers=2, n_elements=2)
        with pytest.raises(ValueError, match="Heads not set"):
            gw.get_head(0, 0)

    def test_get_total_element_pumping(self) -> None:
        """Total element pumping calculation."""
        gw = AppGW(n_nodes=4, n_layers=2, n_elements=2)
        gw.add_element_pumping(
            ElementPumping(element_id=1, layer=1, pump_rate=-100.0, layer_fraction=0.5)
        )
        gw.add_element_pumping(
            ElementPumping(element_id=2, layer=1, pump_rate=-200.0, layer_fraction=1.0)
        )
        total = gw.get_total_element_pumping()
        assert total == pytest.approx(-250.0)

    def test_iter_wells(self) -> None:
        """iter_wells yields wells in ID order."""
        gw = AppGW(n_nodes=4, n_layers=2, n_elements=2)
        gw.add_well(Well(id=3, x=0.0, y=0.0, element=1))
        gw.add_well(Well(id=1, x=0.0, y=0.0, element=1))
        gw.add_well(Well(id=2, x=0.0, y=0.0, element=1))
        ids = [w.id for w in gw.iter_wells()]
        assert ids == [1, 2, 3]

    def test_repr(self) -> None:
        """Repr includes n_nodes, n_layers, n_wells."""
        gw = AppGW(n_nodes=10, n_layers=3, n_elements=5)
        r = repr(gw)
        assert "AppGW" in r
        assert "10" in r
        assert "3" in r
