"""Deep tests for compute_multilayer_weights and compute_composite_head."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from pyiwfm.calibration.iwfm2obs import (
    MultiLayerWellSpec,
    compute_composite_head,
    compute_multilayer_weights,
)
from tests.conftest import make_simple_grid, make_simple_stratigraphy

# The functions under test use a local import:
#   from pyiwfm.core.interpolation import FEInterpolator
# So the correct mock target is the *source* module attribute.
_FE_INTERP_TARGET = "pyiwfm.core.interpolation.FEInterpolator"


def _make_well() -> MultiLayerWellSpec:
    """Well at center of element 1, screen spanning full depth."""
    return MultiLayerWellSpec(
        name="W1",
        x=50.0,
        y=50.0,
        element_id=1,
        bottom_of_screen=0.0,
        top_of_screen=100.0,
    )


def _mock_interpolate(x: float, y: float) -> tuple[int, tuple[int, ...], np.ndarray]:
    """Return deterministic interpolation at element 1, nodes (1,2,5,4)."""
    return 1, (1, 2, 5, 4), np.array([0.25, 0.25, 0.25, 0.25])


# ---------------------------------------------------------------------------
# compute_multilayer_weights
# ---------------------------------------------------------------------------


class TestComputeMultilayerWeights:
    """Tests for compute_multilayer_weights."""

    def test_screen_single_layer(self) -> None:
        """Screen confined to layer 1 only gives weight=1 for layer 0."""
        grid = make_simple_grid()
        strat = make_simple_stratigraphy(n_nodes=9, n_layers=2)
        # Uniform HK per layer
        hk = np.array([10.0, 5.0])  # 1-D: layer-uniform
        # Screen from 60 to 90 => entirely within layer 0 (top=100, bot=50)
        well = MultiLayerWellSpec(
            name="W1",
            x=50.0,
            y=50.0,
            element_id=1,
            bottom_of_screen=60.0,
            top_of_screen=90.0,
        )
        with patch(_FE_INTERP_TARGET) as MockInterp:
            MockInterp.return_value.interpolate.return_value = _mock_interpolate(50.0, 50.0)
            weights = compute_multilayer_weights(well, grid, strat, hk)

        assert weights.shape == (2,)
        np.testing.assert_allclose(weights[0], 1.0, atol=1e-12)
        np.testing.assert_allclose(weights[1], 0.0, atol=1e-12)

    def test_screen_spans_two_layers(self) -> None:
        """Screen spanning both layers yields T-weighted result."""
        grid = make_simple_grid()
        strat = make_simple_stratigraphy(n_nodes=9, n_layers=2)
        hk = np.array([10.0, 5.0])
        # Screen from 20 to 80 => layer 0: overlap 80-50=30, layer 1: overlap 50-20=30
        well = MultiLayerWellSpec(
            name="W1",
            x=50.0,
            y=50.0,
            element_id=1,
            bottom_of_screen=20.0,
            top_of_screen=80.0,
        )
        with patch(_FE_INTERP_TARGET) as MockInterp:
            MockInterp.return_value.interpolate.return_value = _mock_interpolate(50.0, 50.0)
            weights = compute_multilayer_weights(well, grid, strat, hk)

        # T_0 = 30 * 10 = 300, T_1 = 30 * 5 = 150, total = 450
        np.testing.assert_allclose(weights[0], 300.0 / 450.0, atol=1e-12)
        np.testing.assert_allclose(weights[1], 150.0 / 450.0, atol=1e-12)
        np.testing.assert_allclose(np.sum(weights), 1.0, atol=1e-12)

    def test_zero_transmissivity_fallback(self) -> None:
        """Zero HK everywhere gives equal weights."""
        grid = make_simple_grid()
        strat = make_simple_stratigraphy(n_nodes=9, n_layers=2)
        hk = np.array([0.0, 0.0])
        well = _make_well()
        with patch(_FE_INTERP_TARGET) as MockInterp:
            MockInterp.return_value.interpolate.return_value = _mock_interpolate(50.0, 50.0)
            weights = compute_multilayer_weights(well, grid, strat, hk)

        np.testing.assert_allclose(weights, [0.5, 0.5], atol=1e-12)


# ---------------------------------------------------------------------------
# compute_composite_head
# ---------------------------------------------------------------------------


class TestComputeCompositeHead:
    """Tests for compute_composite_head."""

    def test_1d_layer_heads(self) -> None:
        """1-D layer_heads: simple weighted sum, no FE interpolation needed."""
        grid = make_simple_grid()
        well = _make_well()
        layer_heads = np.array([80.0, 40.0])
        weights = np.array([0.6, 0.4])

        result = compute_composite_head(well, layer_heads, weights, grid)
        expected = 80.0 * 0.6 + 40.0 * 0.4
        assert pytest.approx(result, abs=1e-12) == expected

    def test_2d_spatially_varying_heads(self) -> None:
        """2-D layer_heads (n_layers, n_nodes): FE interpolation applied."""
        grid = make_simple_grid()
        well = _make_well()
        n_layers, n_nodes = 2, 9
        layer_heads = np.ones((n_layers, n_nodes)) * 50.0
        # Make layer 0 = 100, layer 1 = 30 at all nodes
        layer_heads[0, :] = 100.0
        layer_heads[1, :] = 30.0
        weights = np.array([0.7, 0.3])

        with patch(_FE_INTERP_TARGET) as MockInterp:
            MockInterp.return_value.interpolate.return_value = _mock_interpolate(50.0, 50.0)
            result = compute_composite_head(well, layer_heads, weights, grid)

        # All nodes identical, so interpolated head = nodal value
        expected = 100.0 * 0.7 + 30.0 * 0.3
        assert pytest.approx(result, abs=1e-12) == expected
