"""Unit tests for Stratigraphy class."""

from __future__ import annotations

import numpy as np
import pytest

from pyiwfm.core.exceptions import StratigraphyError
from pyiwfm.core.stratigraphy import Stratigraphy


class TestStratigraphy:
    """Tests for the Stratigraphy class."""

    def test_stratigraphy_creation(self, sample_stratigraphy_data: dict) -> None:
        """Test basic stratigraphy creation."""
        strat = Stratigraphy(**sample_stratigraphy_data)

        assert strat.n_layers == 2
        assert strat.n_nodes == 9

    def test_stratigraphy_arrays_shape(self, sample_stratigraphy_data: dict) -> None:
        """Test array shapes are correct."""
        strat = Stratigraphy(**sample_stratigraphy_data)

        assert strat.gs_elev.shape == (9,)
        assert strat.top_elev.shape == (9, 2)
        assert strat.bottom_elev.shape == (9, 2)
        assert strat.active_node.shape == (9, 2)

    def test_stratigraphy_layer_thickness(self, sample_stratigraphy_data: dict) -> None:
        """Test layer thickness calculation."""
        strat = Stratigraphy(**sample_stratigraphy_data)

        thickness = strat.get_layer_thickness(layer=0)
        assert thickness.shape == (9,)
        np.testing.assert_allclose(thickness, 50.0)  # All 50 units thick

    def test_stratigraphy_total_thickness(self, sample_stratigraphy_data: dict) -> None:
        """Test total thickness calculation."""
        strat = Stratigraphy(**sample_stratigraphy_data)

        total = strat.get_total_thickness()
        assert total.shape == (9,)
        np.testing.assert_allclose(total, 100.0)  # 2 layers * 50 units each

    def test_stratigraphy_get_node_elevations(self, sample_stratigraphy_data: dict) -> None:
        """Test getting elevations for a specific node."""
        strat = Stratigraphy(**sample_stratigraphy_data)

        gs, tops, bottoms = strat.get_node_elevations(node_idx=0)
        assert gs == 100.0
        assert tops == pytest.approx([100.0, 50.0])
        assert bottoms == pytest.approx([50.0, 0.0])

    def test_stratigraphy_layer_top_elev(self, sample_stratigraphy_data: dict) -> None:
        """Test layer top elevation access."""
        strat = Stratigraphy(**sample_stratigraphy_data)

        # Layer 0 (top layer) should have top = ground surface
        top_layer0 = strat.get_layer_top(layer=0)
        np.testing.assert_allclose(top_layer0, strat.gs_elev)

    def test_stratigraphy_layer_bottom_elev(self, sample_stratigraphy_data: dict) -> None:
        """Test layer bottom elevation access."""
        strat = Stratigraphy(**sample_stratigraphy_data)

        # Layer 0 bottom should equal layer 1 top
        bottom_layer0 = strat.get_layer_bottom(layer=0)
        top_layer1 = strat.get_layer_top(layer=1)
        np.testing.assert_allclose(bottom_layer0, top_layer1)

    def test_stratigraphy_active_nodes(self, sample_stratigraphy_data: dict) -> None:
        """Test active node checking."""
        strat = Stratigraphy(**sample_stratigraphy_data)

        # All nodes should be active in test data
        assert strat.is_node_active(node_idx=0, layer=0) is True
        assert strat.is_node_active(node_idx=0, layer=1) is True

    def test_stratigraphy_inactive_nodes(self) -> None:
        """Test stratigraphy with some inactive nodes."""
        n_nodes = 4
        n_layers = 2
        gs_elev = np.array([100.0, 100.0, 100.0, 100.0])
        top_elev = np.array([[100.0, 50.0]] * 4)
        bottom_elev = np.array([[50.0, 0.0]] * 4)
        active_node = np.array([[True, True], [True, False], [False, True], [False, False]])

        strat = Stratigraphy(
            n_layers=n_layers,
            n_nodes=n_nodes,
            gs_elev=gs_elev,
            top_elev=top_elev,
            bottom_elev=bottom_elev,
            active_node=active_node,
        )

        assert strat.is_node_active(0, 0) is True
        assert strat.is_node_active(0, 1) is True
        assert strat.is_node_active(1, 1) is False
        assert strat.is_node_active(3, 0) is False

    def test_stratigraphy_n_active_nodes(self, sample_stratigraphy_data: dict) -> None:
        """Test counting active nodes per layer."""
        strat = Stratigraphy(**sample_stratigraphy_data)

        # All 9 nodes active in both layers
        assert strat.get_n_active_nodes(layer=0) == 9
        assert strat.get_n_active_nodes(layer=1) == 9

    def test_stratigraphy_invalid_layer_index(self, sample_stratigraphy_data: dict) -> None:
        """Test error on invalid layer index."""
        strat = Stratigraphy(**sample_stratigraphy_data)

        with pytest.raises(IndexError):
            strat.get_layer_thickness(layer=5)

    def test_stratigraphy_invalid_node_index(self, sample_stratigraphy_data: dict) -> None:
        """Test error on invalid node index."""
        strat = Stratigraphy(**sample_stratigraphy_data)

        with pytest.raises(IndexError):
            strat.get_node_elevations(node_idx=100)


class TestStratigraphyValidation:
    """Tests for stratigraphy validation."""

    def test_validate_layer_count_mismatch(self) -> None:
        """Test validation fails when array dimensions don't match n_layers."""
        n_nodes = 4
        n_layers = 2
        gs_elev = np.array([100.0, 100.0, 100.0, 100.0])
        top_elev = np.array([[100.0, 50.0, 25.0]] * 4)  # 3 layers, not 2
        bottom_elev = np.array([[50.0, 0.0]] * 4)
        active_node = np.ones((4, 2), dtype=bool)

        with pytest.raises(StratigraphyError, match="layer"):
            Stratigraphy(
                n_layers=n_layers,
                n_nodes=n_nodes,
                gs_elev=gs_elev,
                top_elev=top_elev,
                bottom_elev=bottom_elev,
                active_node=active_node,
            )

    def test_validate_node_count_mismatch(self) -> None:
        """Test validation fails when array dimensions don't match n_nodes."""
        n_nodes = 4
        n_layers = 2
        gs_elev = np.array([100.0, 100.0, 100.0])  # Only 3 nodes
        top_elev = np.array([[100.0, 50.0]] * 4)
        bottom_elev = np.array([[50.0, 0.0]] * 4)
        active_node = np.ones((4, 2), dtype=bool)

        with pytest.raises(StratigraphyError, match="node"):
            Stratigraphy(
                n_layers=n_layers,
                n_nodes=n_nodes,
                gs_elev=gs_elev,
                top_elev=top_elev,
                bottom_elev=bottom_elev,
                active_node=active_node,
            )

    def test_validate_negative_thickness(self) -> None:
        """Test validation fails when layer has negative thickness."""
        n_nodes = 4
        n_layers = 2
        gs_elev = np.array([100.0, 100.0, 100.0, 100.0])
        top_elev = np.array([[100.0, 50.0]] * 4)
        # Bottom higher than top = negative thickness
        bottom_elev = np.array([[150.0, 0.0]] * 4)
        active_node = np.ones((4, 2), dtype=bool)

        strat = Stratigraphy(
            n_layers=n_layers,
            n_nodes=n_nodes,
            gs_elev=gs_elev,
            top_elev=top_elev,
            bottom_elev=bottom_elev,
            active_node=active_node,
        )

        with pytest.raises(StratigraphyError, match="negative thickness"):
            strat.validate()

    def test_validate_layer_discontinuity(self) -> None:
        """Test validation warns when layer bottoms don't match next layer tops."""
        n_nodes = 4
        n_layers = 2
        gs_elev = np.array([100.0, 100.0, 100.0, 100.0])
        top_elev = np.array([[100.0, 40.0]] * 4)  # Layer 1 top at 40
        bottom_elev = np.array([[50.0, 0.0]] * 4)  # Layer 0 bottom at 50
        active_node = np.ones((4, 2), dtype=bool)

        strat = Stratigraphy(
            n_layers=n_layers,
            n_nodes=n_nodes,
            gs_elev=gs_elev,
            top_elev=top_elev,
            bottom_elev=bottom_elev,
            active_node=active_node,
        )

        # Should report discontinuity (gap between layers)
        warnings = strat.validate()
        assert any("discontinuity" in w.lower() for w in warnings)

    def test_validate_success(self, sample_stratigraphy_data: dict) -> None:
        """Test validation passes for valid stratigraphy."""
        strat = Stratigraphy(**sample_stratigraphy_data)

        # Should return empty list (no warnings)
        warnings = strat.validate()
        assert warnings == []


class TestStratigraphyOperations:
    """Tests for stratigraphy operations."""

    def test_get_elevation_at_depth(self, sample_stratigraphy_data: dict) -> None:
        """Test calculating elevation at a given depth below ground."""
        strat = Stratigraphy(**sample_stratigraphy_data)

        # At node 0, GS=100, depth 25 should give elevation 75
        elev = strat.get_elevation_at_depth(node_idx=0, depth=25.0)
        assert elev == pytest.approx(75.0)

    def test_get_layer_at_elevation(self, sample_stratigraphy_data: dict) -> None:
        """Test finding which layer contains a given elevation."""
        strat = Stratigraphy(**sample_stratigraphy_data)

        # At node 0: Layer 0 is 100-50, Layer 1 is 50-0
        assert strat.get_layer_at_elevation(node_idx=0, elevation=75.0) == 0
        assert strat.get_layer_at_elevation(node_idx=0, elevation=25.0) == 1

    def test_get_layer_at_elevation_boundary(self, sample_stratigraphy_data: dict) -> None:
        """Test layer determination at layer boundaries."""
        strat = Stratigraphy(**sample_stratigraphy_data)

        # At elevation 50 (boundary), should return layer 0 (convention: include top)
        assert strat.get_layer_at_elevation(node_idx=0, elevation=50.0) == 0

    def test_get_layer_at_elevation_above_gs(self, sample_stratigraphy_data: dict) -> None:
        """Test layer determination above ground surface."""
        strat = Stratigraphy(**sample_stratigraphy_data)

        # Above ground surface should return -1 or raise
        layer = strat.get_layer_at_elevation(node_idx=0, elevation=150.0)
        assert layer == -1  # Convention: -1 means above surface

    def test_get_layer_at_elevation_below_bottom(self, sample_stratigraphy_data: dict) -> None:
        """Test layer determination below all layers."""
        strat = Stratigraphy(**sample_stratigraphy_data)

        # Below all layers should return n_layers or raise
        layer = strat.get_layer_at_elevation(node_idx=0, elevation=-50.0)
        assert layer == strat.n_layers  # Convention: n_layers means below bottom

    def test_copy(self, sample_stratigraphy_data: dict) -> None:
        """Test creating a deep copy."""
        strat = Stratigraphy(**sample_stratigraphy_data)
        strat_copy = strat.copy()

        # Modify copy and verify original unchanged
        strat_copy.gs_elev[0] = 999.0
        assert strat.gs_elev[0] == 100.0

    def test_repr(self, sample_stratigraphy_data: dict) -> None:
        """Test string representation."""
        strat = Stratigraphy(**sample_stratigraphy_data)
        repr_str = repr(strat)

        assert "Stratigraphy" in repr_str
        assert "n_layers=2" in repr_str
        assert "n_nodes=9" in repr_str


class TestAquitardAccessors:
    """Tests for the aquitard helpers added in Stratigraphy."""

    @staticmethod
    def _build_known_strat() -> Stratigraphy:
        """2-layer, 3-node stratigraphy with non-zero top aquitard.

        Node 0: gs=100, AQT1=5, AQF1=45, AQT2=10, AQF2=40
        Node 1: gs=110, AQT1=0, AQF1=50, AQT2=20, AQF2=40
        Node 2: gs=120, AQT1=2, AQF1=58, AQT2=0,  AQF2=60
        """
        n_nodes = 3
        n_layers = 2
        gs_elev = np.array([100.0, 110.0, 120.0])

        aquitard_1 = np.array([5.0, 0.0, 2.0])
        aquifer_1 = np.array([45.0, 50.0, 58.0])
        aquitard_2 = np.array([10.0, 20.0, 0.0])
        aquifer_2 = np.array([40.0, 40.0, 60.0])

        top_elev = np.zeros((n_nodes, n_layers))
        bottom_elev = np.zeros((n_nodes, n_layers))
        top_elev[:, 0] = gs_elev - aquitard_1
        bottom_elev[:, 0] = top_elev[:, 0] - aquifer_1
        top_elev[:, 1] = bottom_elev[:, 0] - aquitard_2
        bottom_elev[:, 1] = top_elev[:, 1] - aquifer_2

        return Stratigraphy(
            n_layers=n_layers,
            n_nodes=n_nodes,
            gs_elev=gs_elev,
            top_elev=top_elev,
            bottom_elev=bottom_elev,
            active_node=np.ones((n_nodes, n_layers), dtype=bool),
        )

    def test_n_aquitards(self, sample_stratigraphy_data: dict) -> None:
        strat = Stratigraphy(**sample_stratigraphy_data)
        assert strat.n_aquitards == strat.n_layers

    def test_get_aquitard_thickness_top(self) -> None:
        strat = self._build_known_strat()
        np.testing.assert_allclose(strat.get_aquitard_thickness(0), [5.0, 0.0, 2.0])

    def test_get_aquitard_thickness_interior(self) -> None:
        strat = self._build_known_strat()
        np.testing.assert_allclose(strat.get_aquitard_thickness(1), [10.0, 20.0, 0.0])

    def test_get_aquitard_thickness_out_of_range(self) -> None:
        strat = self._build_known_strat()
        with pytest.raises(IndexError):
            strat.get_aquitard_thickness(strat.n_aquitards)
        with pytest.raises(IndexError):
            strat.get_aquitard_thickness(-1)

    def test_get_all_aquitard_thicknesses(self) -> None:
        strat = self._build_known_strat()
        all_aqt = strat.get_all_aquitard_thicknesses()
        assert all_aqt.shape == (3, 2)
        np.testing.assert_allclose(all_aqt[:, 0], [5.0, 0.0, 2.0])
        np.testing.assert_allclose(all_aqt[:, 1], [10.0, 20.0, 0.0])

    def test_get_node_aquitards(self) -> None:
        strat = self._build_known_strat()
        assert strat.get_node_aquitards(0) == pytest.approx([5.0, 10.0])
        assert strat.get_node_aquitards(2) == pytest.approx([2.0, 0.0])

    def test_aquitard_plus_aquifer_equals_gs_minus_bottom(self) -> None:
        """Total column (aquitards + aquifers) must equal gs - bottom_of_last_layer."""
        strat = self._build_known_strat()
        aquitard_sum = strat.get_all_aquitard_thicknesses().sum(axis=1)
        aquifer_sum = (strat.top_elev - strat.bottom_elev).sum(axis=1)
        np.testing.assert_allclose(
            aquitard_sum + aquifer_sum,
            strat.gs_elev - strat.bottom_elev[:, -1],
        )


class TestFromThicknesses:
    """Tests for Stratigraphy.from_thicknesses."""

    def test_from_thicknesses_roundtrip(self) -> None:
        """Round-trip aquitard/aquifer thicknesses through from_thicknesses."""
        gs_elev = np.array([100.0, 110.0, 120.0])
        aquitard = np.array([[5.0, 10.0], [0.0, 20.0], [2.0, 0.0]])
        aquifer = np.array([[45.0, 40.0], [50.0, 40.0], [58.0, 60.0]])

        strat = Stratigraphy.from_thicknesses(gs_elev, aquitard, aquifer)

        assert strat.n_nodes == 3
        assert strat.n_layers == 2
        np.testing.assert_allclose(strat.gs_elev, gs_elev)
        np.testing.assert_allclose(strat.get_all_aquitard_thicknesses(), aquitard)
        # Aquifer thickness is top - bottom
        np.testing.assert_allclose(strat.top_elev - strat.bottom_elev, aquifer)

    def test_from_thicknesses_zero_aquitards(self) -> None:
        """All-zero aquitards => consecutive layers touch (no gaps)."""
        gs_elev = np.array([100.0, 100.0])
        aquitard = np.zeros((2, 3))
        aquifer = np.array([[30.0, 40.0, 20.0], [25.0, 35.0, 15.0]])

        strat = Stratigraphy.from_thicknesses(gs_elev, aquitard, aquifer)

        np.testing.assert_allclose(strat.top_elev[:, 0], gs_elev)
        # Layer k top == layer k-1 bottom when aquitard k is zero
        np.testing.assert_allclose(strat.top_elev[:, 1], strat.bottom_elev[:, 0])
        np.testing.assert_allclose(strat.top_elev[:, 2], strat.bottom_elev[:, 1])
        np.testing.assert_allclose(strat.get_all_aquitard_thicknesses(), 0.0, atol=1e-12)

    def test_from_thicknesses_matches_reader(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """from_thicknesses + write_stratigraphy + read_stratigraphy round-trip."""
        from pyiwfm.io.preprocessor.mesh import read_stratigraphy, write_stratigraphy

        gs_elev = np.array([500.0, 600.0, 700.0])
        aquitard = np.array([[5.0, 10.0], [0.0, 20.0], [2.0, 0.0]])
        aquifer = np.array([[45.0, 40.0], [50.0, 40.0], [58.0, 60.0]])

        strat = Stratigraphy.from_thicknesses(gs_elev, aquitard, aquifer)

        out = tmp_path / "strat.dat"
        write_stratigraphy(out, strat)
        reread = read_stratigraphy(out)

        np.testing.assert_allclose(reread.gs_elev, strat.gs_elev, atol=1e-3)
        np.testing.assert_allclose(reread.top_elev, strat.top_elev, atol=1e-3)
        np.testing.assert_allclose(reread.bottom_elev, strat.bottom_elev, atol=1e-3)

    def test_from_thicknesses_shape_mismatch_raises(self) -> None:
        gs = np.array([100.0, 110.0])
        aquitard = np.zeros((2, 3))
        aquifer_wrong = np.zeros((2, 2))  # different n_layers
        with pytest.raises(StratigraphyError, match="shape"):
            Stratigraphy.from_thicknesses(gs, aquitard, aquifer_wrong)

    def test_from_thicknesses_gs_shape_mismatch_raises(self) -> None:
        gs_wrong = np.array([100.0])  # n_nodes=1, but thicknesses say 2
        aquitard = np.zeros((2, 2))
        aquifer = np.zeros((2, 2))
        with pytest.raises(StratigraphyError, match="gs_elev"):
            Stratigraphy.from_thicknesses(gs_wrong, aquitard, aquifer)

    def test_from_thicknesses_negative_raises(self) -> None:
        gs = np.array([100.0, 110.0])
        aquitard = np.array([[5.0, 10.0], [-1.0, 20.0]])  # negative
        aquifer = np.array([[45.0, 40.0], [50.0, 40.0]])
        with pytest.raises(StratigraphyError, match="[Nn]egative"):
            Stratigraphy.from_thicknesses(gs, aquitard, aquifer)

    def test_from_thicknesses_default_active_node(self) -> None:
        """Default active_node: True where aquifer thickness > 0."""
        gs = np.array([100.0, 100.0])
        aquitard = np.zeros((2, 2))
        # Node 0 active in both layers; node 1 inactive in layer 1
        aquifer = np.array([[30.0, 20.0], [30.0, 0.0]])
        strat = Stratigraphy.from_thicknesses(gs, aquitard, aquifer)
        assert strat.is_node_active(0, 0) is True
        assert strat.is_node_active(0, 1) is True
        assert strat.is_node_active(1, 0) is True
        assert strat.is_node_active(1, 1) is False

    def test_from_thicknesses_explicit_active_node(self) -> None:
        gs = np.array([100.0, 100.0])
        aquitard = np.zeros((2, 2))
        aquifer = np.ones((2, 2)) * 10.0
        custom = np.array([[True, False], [False, True]])
        strat = Stratigraphy.from_thicknesses(gs, aquitard, aquifer, active_node=custom)
        np.testing.assert_array_equal(strat.active_node, custom)
