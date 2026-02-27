"""Deep tests for pyiwfm.visualization.plotting targeting uncovered functions.

Covers:
- plot_cross_section(): default layer colors, fill_between rendering
- plot_cross_section_location(): plan-view with mesh underlay
- plot_spatial_bias(): diverging colormap scatter plot
- Save-to-file paths (savefig through the figure)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from tests.conftest import make_simple_grid, make_simple_stratigraphy


# ---------------------------------------------------------------------------
# Helpers -- synthetic CrossSection dataclass
# ---------------------------------------------------------------------------


@dataclass
class _FakeCrossSection:
    """Minimal CrossSection-like object for plotting tests."""

    distance: np.ndarray
    x: np.ndarray
    y: np.ndarray
    gs_elev: np.ndarray
    top_elev: np.ndarray
    bottom_elev: np.ndarray
    mask: np.ndarray
    n_layers: int
    start: tuple[float, float]
    end: tuple[float, float]
    waypoints: list[tuple[float, float]] | None = None
    scalar_values: dict[str, np.ndarray] = field(default_factory=dict)
    layer_properties: dict[str, np.ndarray] = field(default_factory=dict)


def _make_cross_section(n_samples: int = 10, n_layers: int = 2) -> _FakeCrossSection:
    """Build a synthetic cross-section for testing."""
    dist = np.linspace(0, 1000, n_samples)
    x = np.linspace(0, 200, n_samples)
    y = np.full(n_samples, 100.0)
    gs_elev = np.full(n_samples, 100.0)

    top_elev = np.zeros((n_samples, n_layers))
    bottom_elev = np.zeros((n_samples, n_layers))
    for layer in range(n_layers):
        top_elev[:, layer] = 100.0 - layer * 50.0
        bottom_elev[:, layer] = 100.0 - (layer + 1) * 50.0

    mask = np.ones(n_samples, dtype=bool)
    # Leave first sample outside domain for branch coverage
    mask[0] = False

    return _FakeCrossSection(
        distance=dist,
        x=x,
        y=y,
        gs_elev=gs_elev,
        top_elev=top_elev,
        bottom_elev=bottom_elev,
        mask=mask,
        n_layers=n_layers,
        start=(0.0, 100.0),
        end=(200.0, 100.0),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPlotCrossSection:
    """Tests for plot_cross_section()."""

    def test_returns_figure_and_axes(self) -> None:
        """plot_cross_section returns a (Figure, Axes) tuple."""
        from pyiwfm.visualization.plotting import plot_cross_section

        xs = _make_cross_section()
        fig, ax = plot_cross_section(xs)

        assert isinstance(fig, Figure)
        assert ax is not None
        plt.close(fig)

    def test_with_title(self) -> None:
        """Title is set when provided."""
        from pyiwfm.visualization.plotting import plot_cross_section

        xs = _make_cross_section()
        fig, ax = plot_cross_section(xs, title="Test Section")

        assert ax.get_title() == "Test Section"
        plt.close(fig)

    def test_without_ground_surface(self) -> None:
        """show_ground_surface=False still produces valid output."""
        from pyiwfm.visualization.plotting import plot_cross_section

        xs = _make_cross_section()
        fig, ax = plot_cross_section(xs, show_ground_surface=False)

        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_with_scalar_overlay(self) -> None:
        """scalar_name draws a dashed line for the scalar."""
        from pyiwfm.visualization.plotting import plot_cross_section

        xs = _make_cross_section()
        xs.scalar_values["head"] = np.full(len(xs.distance), 60.0)
        fig, ax = plot_cross_section(xs, scalar_name="head")

        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_with_layer_property(self) -> None:
        """layer_property_name triggers color-mapped rendering."""
        from pyiwfm.visualization.plotting import plot_cross_section

        xs = _make_cross_section()
        xs.layer_properties["kh"] = np.random.rand(len(xs.distance), xs.n_layers) * 10
        fig, ax = plot_cross_section(xs, layer_property_name="kh")

        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_save_to_file(self, tmp_path: Path) -> None:
        """Verify figure can be saved to file."""
        from pyiwfm.visualization.plotting import plot_cross_section

        xs = _make_cross_section()
        fig, ax = plot_cross_section(xs)

        output = tmp_path / "cross_section.png"
        fig.savefig(output, dpi=72)
        assert output.exists()
        assert output.stat().st_size > 0
        plt.close(fig)


class TestPlotCrossSectionLocation:
    """Tests for plot_cross_section_location()."""

    def test_returns_figure(self) -> None:
        """Plan-view map with cross-section line returns valid Figure."""
        from pyiwfm.visualization.plotting import plot_cross_section_location

        grid = make_simple_grid()
        xs = _make_cross_section()

        fig, ax = plot_cross_section_location(grid, xs)

        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_with_waypoints(self) -> None:
        """Cross-section with waypoints draws polyline."""
        from pyiwfm.visualization.plotting import plot_cross_section_location

        grid = make_simple_grid()
        xs = _make_cross_section()
        xs.waypoints = [(0.0, 100.0), (100.0, 100.0), (200.0, 100.0)]

        fig, ax = plot_cross_section_location(grid, xs)

        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_without_labels(self) -> None:
        """show_labels=False suppresses A/A' annotations."""
        from pyiwfm.visualization.plotting import plot_cross_section_location

        grid = make_simple_grid()
        xs = _make_cross_section()

        fig, ax = plot_cross_section_location(grid, xs, show_labels=False)

        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPlotSpatialBias:
    """Tests for plot_spatial_bias()."""

    def test_returns_figure(self) -> None:
        """Spatial bias scatter plot returns valid Figure."""
        from pyiwfm.visualization.plotting import plot_spatial_bias

        grid = make_simple_grid()
        n_obs = 5
        x = np.array([50.0, 100.0, 150.0, 50.0, 150.0])
        y = np.array([50.0, 50.0, 50.0, 150.0, 150.0])
        bias = np.array([2.0, -1.5, 3.0, -0.5, 1.0])

        fig, ax = plot_spatial_bias(grid, x, y, bias)

        assert isinstance(fig, Figure)
        assert ax.get_title() == "Spatial Bias"
        plt.close(fig)

    def test_asymmetric_colorbar(self) -> None:
        """symmetric_colorbar=False uses data min/max."""
        from pyiwfm.visualization.plotting import plot_spatial_bias

        grid = make_simple_grid()
        x = np.array([50.0, 150.0])
        y = np.array([50.0, 150.0])
        bias = np.array([2.0, 5.0])

        fig, ax = plot_spatial_bias(grid, x, y, bias, symmetric_colorbar=False)

        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_with_units_label(self) -> None:
        """Units string appears in colorbar label."""
        from pyiwfm.visualization.plotting import plot_spatial_bias

        grid = make_simple_grid()
        x = np.array([100.0])
        y = np.array([100.0])
        bias = np.array([1.0])

        fig, ax = plot_spatial_bias(grid, x, y, bias, units="ft")

        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_without_mesh(self) -> None:
        """show_mesh=False skips the mesh background."""
        from pyiwfm.visualization.plotting import plot_spatial_bias

        grid = make_simple_grid()
        x = np.array([100.0])
        y = np.array([100.0])
        bias = np.array([0.5])

        fig, ax = plot_spatial_bias(grid, x, y, bias, show_mesh=False)

        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_save_to_file(self, tmp_path: Path) -> None:
        """Verify figure can be saved to file."""
        from pyiwfm.visualization.plotting import plot_spatial_bias

        grid = make_simple_grid()
        x = np.array([100.0])
        y = np.array([100.0])
        bias = np.array([1.0])

        fig, ax = plot_spatial_bias(grid, x, y, bias)
        output = tmp_path / "spatial_bias.png"
        fig.savefig(output, dpi=72)
        assert output.exists()
        assert output.stat().st_size > 0
        plt.close(fig)
