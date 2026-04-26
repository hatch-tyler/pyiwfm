"""Tests for the IWFMModel mutation helpers added in Phase 2.1.

Covers ergonomic setters for aquifer parameters, stratigraphy, and
groundwater observation wells. Each helper:

- validates inputs (raises ``ValueError`` / ``KeyError``)
- mutates the underlying component in place
- adds the component name to ``model._dirty``
"""

from __future__ import annotations

import numpy as np
import pytest

from pyiwfm.components.groundwater import AppGW, AquiferParameters, HydrographLocation
from pyiwfm.core.model import IWFMModel
from pyiwfm.core.stratigraphy import Stratigraphy


def _make_gw(n_nodes: int = 4, n_layers: int = 2) -> AppGW:
    """Build a minimal AppGW with all five aquifer-parameter arrays allocated."""
    gw = AppGW(n_nodes=n_nodes, n_layers=n_layers, n_elements=2)
    gw.aquifer_params = AquiferParameters(
        n_nodes=n_nodes,
        n_layers=n_layers,
        kh=np.ones((n_nodes, n_layers)),
        kv=np.ones((n_nodes, n_layers)) * 0.1,
        specific_storage=np.ones((n_nodes, n_layers)) * 1e-5,
        specific_yield=np.ones((n_nodes, n_layers)) * 0.2,
        aquitard_kv=np.ones((n_nodes, n_layers)) * 1e-7,
    )
    return gw


class TestSetAquiferParameter:
    def test_replaces_layer_values(self):
        gw = _make_gw()
        model = IWFMModel(name="test", groundwater=gw)
        new = np.array([2.0, 3.0, 4.0, 5.0])

        model.set_aquifer_parameter("kh", layer=1, values=new)

        assert np.array_equal(gw.aquifer_params.kh[:, 0], new)
        # Other layer untouched
        assert np.all(gw.aquifer_params.kh[:, 1] == 1.0)
        # Marked dirty
        assert "groundwater" in model._dirty

    def test_accepts_short_param_names(self):
        gw = _make_gw()
        model = IWFMModel(name="test", groundwater=gw)
        new = np.array([0.3, 0.3, 0.3, 0.3])

        model.set_aquifer_parameter("sy", layer=2, values=new)
        assert np.array_equal(gw.aquifer_params.specific_yield[:, 1], new)

    def test_unknown_param_raises_keyerror(self):
        model = IWFMModel(name="test", groundwater=_make_gw())
        with pytest.raises(KeyError, match="Unknown parameter"):
            model.set_aquifer_parameter("not_a_param", layer=1, values=np.zeros(4))

    def test_layer_out_of_range_raises(self):
        model = IWFMModel(name="test", groundwater=_make_gw(n_layers=2))
        with pytest.raises(ValueError, match=r"layer 3 is out of range \[1, 2\]"):
            model.set_aquifer_parameter("kh", layer=3, values=np.zeros(4))
        with pytest.raises(ValueError, match=r"layer 0 is out of range"):
            model.set_aquifer_parameter("kh", layer=0, values=np.zeros(4))

    def test_wrong_length_raises_value_error(self):
        model = IWFMModel(name="test", groundwater=_make_gw(n_nodes=4))
        with pytest.raises(ValueError, match=r"shape \(4,\)"):
            model.set_aquifer_parameter("kh", layer=1, values=np.zeros(3))

    def test_no_groundwater_raises(self):
        model = IWFMModel(name="empty")
        with pytest.raises(ValueError, match="groundwater component is not loaded"):
            model.set_aquifer_parameter("kh", layer=1, values=np.zeros(4))

    def test_unset_param_array_raises(self):
        gw = AppGW(n_nodes=3, n_layers=1, n_elements=1)
        gw.aquifer_params = AquiferParameters(n_nodes=3, n_layers=1)  # all None
        model = IWFMModel(name="test", groundwater=gw)
        with pytest.raises(ValueError, match="not set"):
            model.set_aquifer_parameter("kh", layer=1, values=np.zeros(3))


class TestSetAquiferParameterAt:
    def test_sets_single_cell(self):
        gw = _make_gw()
        model = IWFMModel(name="test", groundwater=gw)

        model.set_aquifer_parameter_at("kh", node_id=2, layer=1, value=99.0)

        assert gw.aquifer_params.kh[1, 0] == 99.0
        # Neighbors untouched
        assert gw.aquifer_params.kh[0, 0] == 1.0
        assert gw.aquifer_params.kh[2, 0] == 1.0
        assert "groundwater" in model._dirty

    def test_node_out_of_range_raises(self):
        model = IWFMModel(name="test", groundwater=_make_gw(n_nodes=4))
        with pytest.raises(ValueError, match="node"):
            model.set_aquifer_parameter_at("kh", node_id=5, layer=1, value=1.0)

    def test_layer_out_of_range_raises(self):
        model = IWFMModel(name="test", groundwater=_make_gw(n_layers=2))
        with pytest.raises(ValueError, match="layer"):
            model.set_aquifer_parameter_at("kh", node_id=1, layer=5, value=1.0)


class TestSetStratigraphyFromThicknesses:
    def test_builds_and_attaches_stratigraphy(self):
        model = IWFMModel(name="test")
        gs = np.array([100.0, 100.0, 100.0])
        aquitards = np.array([[5.0, 3.0], [5.0, 3.0], [5.0, 3.0]])
        aquifers = np.array([[20.0, 30.0], [20.0, 30.0], [20.0, 30.0]])

        model.set_stratigraphy_from_thicknesses(gs, aquitards, aquifers)

        assert isinstance(model.stratigraphy, Stratigraphy)
        assert model.stratigraphy.n_nodes == 3
        assert model.stratigraphy.n_layers == 2
        assert "stratigraphy" in model._dirty

    def test_validates_against_mesh_when_present(self):
        from pyiwfm.core.mesh import AppGrid, Element, Node

        nodes = {i: Node(id=i, x=float(i), y=0.0) for i in (1, 2, 3)}
        elements = {1: Element(id=1, vertices=[1, 2, 3])}
        mesh = AppGrid(nodes=nodes, elements=elements)
        model = IWFMModel(name="test", mesh=mesh)

        # Wrong length: 2 elevations for a 3-node mesh
        with pytest.raises(ValueError, match="does not match mesh"):
            model.set_stratigraphy_from_thicknesses(
                gs_elev=[100.0, 100.0],
                aquitard_thicknesses=[[5.0], [5.0]],
                aquifer_thicknesses=[[20.0], [20.0]],
            )

    def test_negative_thickness_propagates_stratigraphy_error(self):
        from pyiwfm.core.exceptions import StratigraphyError

        model = IWFMModel(name="test")
        with pytest.raises(StratigraphyError, match="Negative thicknesses"):
            model.set_stratigraphy_from_thicknesses(
                gs_elev=[100.0],
                aquitard_thicknesses=[[-1.0]],
                aquifer_thicknesses=[[20.0]],
            )


class TestObservationWellHelpers:
    def test_add_observation_well(self):
        gw = _make_gw(n_nodes=4, n_layers=2)
        model = IWFMModel(name="test", groundwater=gw)

        model.add_observation_well(node_id=2, layer=1, x=100.0, y=200.0, name="OW-1")

        assert len(gw.hydrograph_locations) == 1
        loc: HydrographLocation = gw.hydrograph_locations[0]
        assert loc.node_id == 2
        assert loc.layer == 1
        assert loc.x == 100.0
        assert loc.name == "OW-1"
        assert "groundwater" in model._dirty

    def test_add_observation_well_validates_node(self):
        model = IWFMModel(name="test", groundwater=_make_gw(n_nodes=4))
        with pytest.raises(ValueError, match="node"):
            model.add_observation_well(node_id=99, layer=1, x=0.0, y=0.0)

    def test_add_observation_well_validates_layer(self):
        model = IWFMModel(name="test", groundwater=_make_gw(n_layers=2))
        with pytest.raises(ValueError, match="layer"):
            model.add_observation_well(node_id=1, layer=99, x=0.0, y=0.0)

    def test_add_observation_well_no_groundwater_raises(self):
        model = IWFMModel(name="empty")
        with pytest.raises(ValueError, match="groundwater"):
            model.add_observation_well(node_id=1, layer=1, x=0.0, y=0.0)

    def test_remove_observation_well(self):
        gw = _make_gw()
        model = IWFMModel(name="test", groundwater=gw)
        model.add_observation_well(1, 1, 0.0, 0.0, name="A")
        model.add_observation_well(2, 1, 0.0, 0.0, name="B")
        model.add_observation_well(3, 1, 0.0, 0.0, name="A")  # duplicate name
        model._dirty.clear()  # reset to test that remove also marks dirty

        n = model.remove_observation_well("A")

        assert n == 2
        assert len(gw.hydrograph_locations) == 1
        assert gw.hydrograph_locations[0].name == "B"
        assert "groundwater" in model._dirty

    def test_remove_returns_zero_when_no_match(self):
        gw = _make_gw()
        model = IWFMModel(name="test", groundwater=gw)
        model.add_observation_well(1, 1, 0.0, 0.0, name="X")
        model._dirty.clear()

        n = model.remove_observation_well("nonexistent")

        assert n == 0
        assert len(gw.hydrograph_locations) == 1
        # No mutation -> not dirty
        assert "groundwater" not in model._dirty


class TestMarkDirty:
    def test_mark_dirty_records_component(self):
        model = IWFMModel(name="test")
        assert model._dirty == set()
        model.mark_dirty("streams")
        assert "streams" in model._dirty

    def test_mark_dirty_is_idempotent(self):
        model = IWFMModel(name="test")
        model.mark_dirty("streams")
        model.mark_dirty("streams")
        assert model._dirty == {"streams"}
