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
