"""Tests for the drawdown computation module."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest


def _make_mock_loader(n_frames: int = 5, n_nodes: int = 4, n_layers: int = 2):
    """Create a mock LazyNodalLoader with deterministic data."""
    loader = MagicMock()
    loader.n_frames = n_frames
    loader.n_nodes = n_nodes
    loader.n_layers = n_layers
    loader.shape = (n_nodes, n_layers)

    # Each frame: heads decrease linearly over time for each node
    frames = []
    for t in range(n_frames):
        frame = np.array(
            [[100.0 - t * 2 - layer * 10 for layer in range(n_layers)] for _ in range(n_nodes)],
            dtype=np.float64,
        )
        frames.append(frame)

    loader.get_frame.side_effect = lambda idx: frames[idx]
    return loader


class TestDrawdownComputer:
    """Tests for DrawdownComputer."""

    def test_compute_drawdown_basic(self):
        """Drawdown = head(ref) - head(t), positive = decline."""
        from pyiwfm.io.drawdown import DrawdownComputer

        loader = _make_mock_loader()
        dc = DrawdownComputer(loader)

        # Reference at t=0, compute at t=2, layer 1
        dd = dc.compute_drawdown(timestep=2, reference_timestep=0, layer=1)

        # head(t=0, layer=0) = 100, head(t=2, layer=0) = 96
        # drawdown = 100 - 96 = 4
        assert dd.shape == (4,)
        np.testing.assert_allclose(dd, 4.0)

    def test_compute_drawdown_zero_at_reference(self):
        """Drawdown at reference timestep should be zero."""
        from pyiwfm.io.drawdown import DrawdownComputer

        loader = _make_mock_loader()
        dc = DrawdownComputer(loader)

        dd = dc.compute_drawdown(timestep=0, reference_timestep=0, layer=1)
        np.testing.assert_allclose(dd, 0.0)

    def test_compute_drawdown_handles_nan(self):
        """NaN/sentinel values should propagate as NaN in drawdown."""
        from pyiwfm.io.drawdown import DrawdownComputer

        loader = _make_mock_loader(n_nodes=2)
        # Inject a sentinel value
        frame0 = np.array([[100.0, 80.0], [-10000.0, 80.0]], dtype=np.float64)
        frame1 = np.array([[98.0, 78.0], [96.0, 78.0]], dtype=np.float64)
        loader.get_frame.side_effect = lambda idx: [frame0, frame1][idx]
        loader.n_frames = 2

        dc = DrawdownComputer(loader)
        dd = dc.compute_drawdown(timestep=1, reference_timestep=0, layer=1)

        assert dd.shape == (2,)
        assert dd[0] == pytest.approx(2.0)
        assert np.isnan(dd[1])  # sentinel at reference should produce NaN

    def test_compute_drawdown_range(self):
        """compute_drawdown_range returns robust min/max."""
        from pyiwfm.io.drawdown import DrawdownComputer

        loader = _make_mock_loader()
        dc = DrawdownComputer(loader)

        lo, hi = dc.compute_drawdown_range(reference_timestep=0, layer=1)
        # Drawdown increases from 0 to 8 across timesteps 0-4
        assert lo >= 0.0
        assert hi <= 8.5  # Allow for percentile smoothing

    def test_compute_max_drawdown_map(self):
        """compute_max_drawdown_map returns max drawdown at each node."""
        from pyiwfm.io.drawdown import DrawdownComputer

        loader = _make_mock_loader()
        dc = DrawdownComputer(loader)

        max_dd = dc.compute_max_drawdown_map(reference_timestep=0, layer=1)
        assert max_dd.shape == (4,)
        # Max drawdown should be at the last timestep: 100 - 92 = 8
        np.testing.assert_allclose(max_dd, 8.0)
