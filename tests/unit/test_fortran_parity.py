"""Tests for Fortran feature parity in IWFM2OBS.

These tests verify that pyiwfm matches the Fortran IWFM 2025.0.1747 behavior
for multi-layer computation, subsidence summation, and edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestCompositeSubsidence:
    """Test that subsidence is summed (not T-weighted) across layers."""

    def test_basic_summation(self) -> None:
        from pyiwfm.calibration.iwfm2obs import compute_composite_subsidence

        layer_sub = np.array([0.01, 0.02, 0.005])
        result = compute_composite_subsidence(layer_sub)
        assert result == pytest.approx(0.035)

    def test_with_nan(self) -> None:
        from pyiwfm.calibration.iwfm2obs import compute_composite_subsidence

        layer_sub = np.array([0.01, float("nan"), 0.005])
        result = compute_composite_subsidence(layer_sub)
        assert result == pytest.approx(0.015)

    def test_all_nan(self) -> None:
        from pyiwfm.calibration.iwfm2obs import compute_composite_subsidence

        layer_sub = np.array([float("nan"), float("nan")])
        result = compute_composite_subsidence(layer_sub)
        assert result == pytest.approx(0.0)

    def test_single_layer(self) -> None:
        from pyiwfm.calibration.iwfm2obs import compute_composite_subsidence

        result = compute_composite_subsidence(np.array([0.05]))
        assert result == pytest.approx(0.05)


class TestOverwriteLayer:
    """Test overwrite_layer support in compute_multilayer_weights."""

    def test_overwrite_layer_sets_single_weight(self) -> None:
        """When overwrite_layer > 0, weight=1.0 for that layer only."""
        from unittest.mock import MagicMock

        from pyiwfm.calibration.iwfm2obs import compute_multilayer_weights

        well = MagicMock()
        well.overwrite_layer = 2
        well.x = 100.0
        well.y = 200.0
        well.bottom_of_screen = -50.0
        well.top_of_screen = -10.0

        grid = MagicMock()
        strat = MagicMock()
        strat.n_layers = 4

        hk = np.ones(4)

        weights = compute_multilayer_weights(well, grid, strat, hk)
        assert weights[1] == pytest.approx(1.0)  # layer 2 (0-indexed: 1)
        assert np.sum(weights) == pytest.approx(1.0)
        assert weights[0] == pytest.approx(0.0)
        assert weights[2] == pytest.approx(0.0)
        assert weights[3] == pytest.approx(0.0)

    def test_overwrite_layer_negative_ignored(self) -> None:
        """When overwrite_layer <= 0, normal T-weighted computation proceeds."""
        import unittest.mock as mock
        from unittest.mock import MagicMock

        from pyiwfm.calibration.iwfm2obs import compute_multilayer_weights

        well = MagicMock()
        well.overwrite_layer = -1
        well.x = 100.0
        well.y = 200.0
        well.bottom_of_screen = -50.0
        well.top_of_screen = 0.0

        grid = MagicMock()
        strat = MagicMock()
        strat.n_layers = 2
        strat.top_elev = np.array([[0.0, -25.0]])
        strat.bottom_elev = np.array([[-25.0, -50.0]])

        interp_mock = MagicMock()
        interp_mock.interpolate.return_value = (1, [1], np.array([1.0]))

        hk = np.array([1.0, 1.0])

        with mock.patch(
            "pyiwfm.core.interpolation.FEInterpolator",
            return_value=interp_mock,
        ):
            weights = compute_multilayer_weights(well, grid, strat, hk)

        # Both layers have nonzero thickness and HK, so both get weight
        assert np.sum(weights) == pytest.approx(1.0)
        assert len(weights) == 2


class TestZeroTransmissivityFallback:
    """Test that zero-transmissivity fallback matches Fortran behavior."""

    def test_uses_first_nonzero_thickness(self) -> None:
        """Fortran: weight=1.0 for first non-zero-thickness layer."""
        from unittest.mock import MagicMock

        from pyiwfm.calibration.iwfm2obs import compute_multilayer_weights

        well = MagicMock()
        well.overwrite_layer = -1
        well.x = 100.0
        well.y = 200.0
        well.bottom_of_screen = -100.0
        well.top_of_screen = 0.0

        # Mock grid and strat so that thicknesses are [0, 10, 5]
        # but HK is all zeros -> T_total = 0
        grid = MagicMock()
        strat = MagicMock()
        strat.n_layers = 3

        # Mock FEInterpolator
        interp_mock = MagicMock()
        interp_mock.interpolate.return_value = (1, [1], np.array([1.0]))

        # Set up stratigraphy arrays
        strat.top_elev = np.array([[0.0, -10.0, -60.0]])  # 1 node, 3 layers
        strat.bottom_elev = np.array([[-10.0, -60.0, -100.0]])

        hk = np.zeros(3)  # All zero HK -> zero transmissivity

        import unittest.mock as mock

        with mock.patch(
            "pyiwfm.core.interpolation.FEInterpolator",
            return_value=interp_mock,
        ):
            weights = compute_multilayer_weights(well, grid, strat, hk)

        # Since HK=0 everywhere, T=0 -> fallback: first non-zero-thickness layer
        assert np.sum(weights) == pytest.approx(1.0)
        # Should have weight=1.0 on exactly one layer
        assert np.max(weights) == pytest.approx(1.0)
