"""Additional coverage tests for core/cross_section.py."""

from __future__ import annotations

import numpy as np
import pytest

from pyiwfm.core.cross_section import CrossSection, CrossSectionExtractor
from tests.conftest import make_simple_grid, make_simple_stratigraphy

# ---------------------------------------------------------------------------
# CrossSection dataclass
# ---------------------------------------------------------------------------


class TestCrossSectionDataclass:
    """Tests for CrossSection properties."""

    def _make_xs(self, n_samples: int = 10, n_layers: int = 2) -> CrossSection:
        dist = np.linspace(0.0, 1000.0, n_samples)
        x = np.linspace(0.0, 1000.0, n_samples)
        y = np.full(n_samples, 500.0)
        gs = np.full(n_samples, 100.0)
        top = np.full((n_samples, n_layers), 100.0)
        bot = np.full((n_samples, n_layers), 0.0)
        mask = np.ones(n_samples, dtype=bool)
        mask[0] = False  # first sample outside
        return CrossSection(
            distance=dist,
            x=x,
            y=y,
            gs_elev=gs,
            top_elev=top,
            bottom_elev=bot,
            mask=mask,
            n_layers=n_layers,
            start=(0.0, 500.0),
            end=(1000.0, 500.0),
        )

    def test_total_length(self) -> None:
        xs = self._make_xs()
        assert xs.total_length == pytest.approx(1000.0)

    def test_n_samples(self) -> None:
        xs = self._make_xs(n_samples=20)
        assert xs.n_samples == 20

    def test_fraction_inside(self) -> None:
        xs = self._make_xs(n_samples=10)
        assert xs.fraction_inside == pytest.approx(0.9)

    def test_get_layer_top(self) -> None:
        xs = self._make_xs()
        top0 = xs.get_layer_top(0)
        assert len(top0) == 10
        assert top0[0] == pytest.approx(100.0)

    def test_get_layer_bottom(self) -> None:
        xs = self._make_xs()
        bot1 = xs.get_layer_bottom(1)
        assert len(bot1) == 10
        assert bot1[0] == pytest.approx(0.0)

    def test_repr(self) -> None:
        xs = self._make_xs()
        r = repr(xs)
        assert "CrossSection" in r
        assert "n_samples=10" in r
        assert "n_layers=2" in r


# ---------------------------------------------------------------------------
# CrossSectionExtractor
# ---------------------------------------------------------------------------


class TestCrossSectionExtractor:
    """Tests for the CrossSectionExtractor."""

    def _make_extractor(self):
        grid = make_simple_grid()
        strat = make_simple_stratigraphy(n_nodes=9, n_layers=2)
        return CrossSectionExtractor(grid, strat)

    def test_extract_horizontal(self) -> None:
        """Extract along a horizontal line through the mesh."""
        ext = self._make_extractor()
        xs = ext.extract(start=(0.0, 100.0), end=(200.0, 100.0), n_samples=5)
        assert xs.n_samples == 5
        assert xs.n_layers == 2
        assert xs.start == (0.0, 100.0)
        assert xs.end == (200.0, 100.0)

    def test_extract_diagonal(self) -> None:
        """Extract along a diagonal line."""
        ext = self._make_extractor()
        xs = ext.extract(start=(0.0, 0.0), end=(200.0, 200.0), n_samples=10)
        assert xs.n_samples == 10
        assert xs.total_length > 0

    def test_extract_outside_mesh(self) -> None:
        """Points outside the mesh should have mask=False and NaN elevations."""
        ext = self._make_extractor()
        xs = ext.extract(start=(500.0, 500.0), end=(600.0, 600.0), n_samples=5)
        assert xs.n_samples == 5
        # All points should be outside the 0-200 mesh
        assert np.all(~xs.mask)
        assert np.all(np.isnan(xs.gs_elev))

    def test_single_element_mesh(self) -> None:
        """Test with a mesh that has just one element."""
        from pyiwfm.core.mesh import AppGrid, Element, Node
        from pyiwfm.core.stratigraphy import Stratigraphy

        nodes = {
            1: Node(id=1, x=0.0, y=0.0),
            2: Node(id=2, x=100.0, y=0.0),
            3: Node(id=3, x=50.0, y=100.0),
        }
        elements = {
            1: Element(id=1, vertices=(1, 2, 3), subregion=1),
        }
        grid = AppGrid(nodes=nodes, elements=elements)
        grid.compute_connectivity()

        strat = Stratigraphy(
            n_layers=1,
            n_nodes=3,
            gs_elev=np.array([100.0, 100.0, 100.0]),
            top_elev=np.array([[100.0], [100.0], [100.0]]),
            bottom_elev=np.array([[0.0], [0.0], [0.0]]),
            active_node=np.ones((3, 1), dtype=bool),
        )

        ext = CrossSectionExtractor(grid, strat)
        xs = ext.extract(start=(10.0, 10.0), end=(80.0, 10.0), n_samples=5)
        assert xs.n_samples == 5

    def test_extract_polyline(self) -> None:
        """Extract along a multi-segment polyline."""
        ext = self._make_extractor()
        waypoints = [(0.0, 100.0), (100.0, 100.0), (200.0, 100.0)]
        xs = ext.extract_polyline(waypoints, n_samples_per_segment=10)
        assert xs.n_samples > 0
        assert xs.waypoints is not None
        assert len(xs.waypoints) == 3

    def test_extract_polyline_too_few_points(self) -> None:
        ext = self._make_extractor()
        with pytest.raises(ValueError, match="At least 2"):
            ext.extract_polyline([(0.0, 0.0)])

    def test_interpolate_scalar(self) -> None:
        ext = self._make_extractor()
        xs = ext.extract(start=(0.0, 100.0), end=(200.0, 100.0), n_samples=5)
        node_values = np.arange(9, dtype=np.float64) * 10
        result = ext.interpolate_scalar(xs, node_values, "test_field")
        assert "test_field" in xs.scalar_values
        assert len(result) == 5

    def test_interpolate_layer_property(self) -> None:
        ext = self._make_extractor()
        xs = ext.extract(start=(0.0, 100.0), end=(200.0, 100.0), n_samples=5)
        rng = np.random.default_rng(42)
        node_layer_values = rng.random((9, 2))
        result = ext.interpolate_layer_property(xs, node_layer_values, "kh")
        assert "kh" in xs.layer_properties
        assert result.shape == (5, 2)

    def test_zero_length_line(self) -> None:
        """Degenerate case: start == end."""
        ext = self._make_extractor()
        xs = ext.extract(start=(100.0, 100.0), end=(100.0, 100.0), n_samples=3)
        assert xs.total_length == pytest.approx(0.0)
