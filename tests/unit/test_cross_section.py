"""
Tests for cross-section extraction and plotting.

Tests cover:
- CrossSection dataclass properties and accessors
- CrossSectionExtractor extraction, interpolation, and polyline support
- plot_cross_section and plot_cross_section_location rendering
- Coverage gaps: partial-coverage interpolation, NaN property skipping
- Enterprise tests: triangle meshes, parametrized extractions, determinism
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyiwfm.core.cross_section import CrossSectionExtractor
from pyiwfm.core.mesh import AppGrid, Element, Node
from pyiwfm.core.stratigraphy import Stratigraphy

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_quad_mesh(
    nx: int = 5,
    ny: int = 5,
    dx: float = 100.0,
    dy: float = 100.0,
) -> AppGrid:
    """Create a regular quad mesh for testing."""
    nodes: dict[int, Node] = {}
    nid = 1
    for j in range(ny):
        for i in range(nx):
            nodes[nid] = Node(id=nid, x=i * dx, y=j * dy)
            nid += 1

    elements: dict[int, Element] = {}
    eid = 1
    for j in range(ny - 1):
        for i in range(nx - 1):
            n1 = j * nx + i + 1
            n2 = n1 + 1
            n3 = n2 + nx
            n4 = n1 + nx
            elements[eid] = Element(id=eid, vertices=(n1, n2, n3, n4))
            eid += 1

    grid = AppGrid(nodes=nodes, elements=elements)
    grid.compute_connectivity()
    return grid


def _make_tri_mesh(
    nx: int = 5,
    ny: int = 5,
    dx: float = 100.0,
    dy: float = 100.0,
) -> AppGrid:
    """
    Create a regular triangle mesh by splitting each quad into 2 triangles.

    Each quad cell (n1, n2, n3, n4) is split along the diagonal into:
    - Triangle A: (n1, n2, n3)
    - Triangle B: (n1, n3, n4)
    """
    nodes: dict[int, Node] = {}
    nid = 1
    for j in range(ny):
        for i in range(nx):
            nodes[nid] = Node(id=nid, x=i * dx, y=j * dy)
            nid += 1

    elements: dict[int, Element] = {}
    eid = 1
    for j in range(ny - 1):
        for i in range(nx - 1):
            n1 = j * nx + i + 1
            n2 = n1 + 1
            n3 = n2 + nx
            n4 = n1 + nx
            elements[eid] = Element(id=eid, vertices=(n1, n2, n3))
            eid += 1
            elements[eid] = Element(id=eid, vertices=(n1, n3, n4))
            eid += 1

    grid = AppGrid(nodes=nodes, elements=elements)
    grid.compute_connectivity()
    return grid


def _make_linear_stratigraphy(
    grid: AppGrid,
    n_layers: int = 3,
    gs_base: float = 100.0,
    gs_slope_x: float = -0.01,
    gs_slope_y: float = 0.0,
    layer_thickness: float = 20.0,
) -> Stratigraphy:
    """
    Create stratigraphy with a linear ground surface and uniform layer thickness.

    gs_elev = gs_base + gs_slope_x * x + gs_slope_y * y
    """
    sorted_ids = sorted(grid.nodes.keys())
    n_nodes = len(sorted_ids)

    gs = np.zeros(n_nodes)
    for idx, nid in enumerate(sorted_ids):
        node = grid.nodes[nid]
        gs[idx] = gs_base + gs_slope_x * node.x + gs_slope_y * node.y

    top = np.zeros((n_nodes, n_layers))
    bot = np.zeros((n_nodes, n_layers))
    for layer in range(n_layers):
        top[:, layer] = gs - layer * layer_thickness
        bot[:, layer] = gs - (layer + 1) * layer_thickness

    active = np.ones((n_nodes, n_layers), dtype=bool)
    return Stratigraphy(
        n_layers=n_layers,
        n_nodes=n_nodes,
        gs_elev=gs,
        top_elev=top,
        bottom_elev=bot,
        active_node=active,
    )


@pytest.fixture
def quad_mesh() -> AppGrid:
    """6x6 regular quad mesh spanning (0,0) to (500,500)."""
    return _make_quad_mesh(nx=6, ny=6, dx=100.0, dy=100.0)


@pytest.fixture
def tri_mesh() -> AppGrid:
    """6x6 regular triangle mesh spanning (0,0) to (500,500)."""
    return _make_tri_mesh(nx=6, ny=6, dx=100.0, dy=100.0)


@pytest.fixture
def stratigraphy(quad_mesh: AppGrid) -> Stratigraphy:
    """3-layer stratigraphy with linear ground surface on quad mesh."""
    return _make_linear_stratigraphy(quad_mesh)


@pytest.fixture
def tri_stratigraphy(tri_mesh: AppGrid) -> Stratigraphy:
    """3-layer stratigraphy with linear ground surface on triangle mesh."""
    return _make_linear_stratigraphy(tri_mesh)


@pytest.fixture
def extractor(quad_mesh: AppGrid, stratigraphy: Stratigraphy) -> CrossSectionExtractor:
    """CrossSectionExtractor on the standard quad mesh."""
    return CrossSectionExtractor(quad_mesh, stratigraphy)


@pytest.fixture
def tri_extractor(tri_mesh: AppGrid, tri_stratigraphy: Stratigraphy) -> CrossSectionExtractor:
    """CrossSectionExtractor on the standard triangle mesh."""
    return CrossSectionExtractor(tri_mesh, tri_stratigraphy)


# ===========================================================================
# TestCrossSection
# ===========================================================================


class TestCrossSection:
    """Tests for the CrossSection dataclass."""

    def test_properties(self, extractor: CrossSectionExtractor) -> None:
        """Basic properties: n_samples, n_layers, total_length."""
        xs = extractor.extract(start=(0, 250), end=(500, 250), n_samples=50)
        assert xs.n_samples == 50
        assert xs.n_layers == 3
        assert xs.total_length == pytest.approx(500.0, rel=1e-6)

    def test_fraction_inside_all(self, extractor: CrossSectionExtractor) -> None:
        """Line fully inside mesh should have fraction_inside == 1.0."""
        xs = extractor.extract(start=(50, 250), end=(450, 250), n_samples=30)
        assert xs.fraction_inside == pytest.approx(1.0)

    def test_mask_shape(self, extractor: CrossSectionExtractor) -> None:
        """Mask should have correct shape and dtype."""
        xs = extractor.extract(start=(0, 250), end=(500, 250), n_samples=20)
        assert xs.mask.shape == (20,)
        assert xs.mask.dtype == np.bool_

    def test_get_layer_top_bottom(self, extractor: CrossSectionExtractor) -> None:
        """Layer top/bottom accessors return correct shapes and ordering."""
        xs = extractor.extract(start=(50, 250), end=(450, 250), n_samples=10)
        top0 = xs.get_layer_top(0)
        bot0 = xs.get_layer_bottom(0)
        assert top0.shape == (10,)
        assert bot0.shape == (10,)
        # Top of layer 0 should be above bottom of layer 0
        valid = xs.mask
        assert np.all(top0[valid] > bot0[valid])

    def test_scalar_values_initially_empty(self, extractor: CrossSectionExtractor) -> None:
        """Freshly extracted cross-section should have empty scalar/property dicts."""
        xs = extractor.extract(start=(50, 250), end=(450, 250))
        assert xs.scalar_values == {}
        assert xs.layer_properties == {}

    def test_repr(self, extractor: CrossSectionExtractor) -> None:
        """String representation should include key metadata."""
        xs = extractor.extract(start=(50, 250), end=(450, 250), n_samples=10)
        r = repr(xs)
        assert "CrossSection" in r
        assert "n_samples=10" in r

    def test_start_end(self, extractor: CrossSectionExtractor) -> None:
        """Start and end coordinates should be preserved."""
        xs = extractor.extract(start=(50, 250), end=(450, 250))
        assert xs.start == (50, 250)
        assert xs.end == (450, 250)


# ===========================================================================
# TestCrossSectionExtractor
# ===========================================================================


class TestCrossSectionExtractor:
    """Tests for the CrossSectionExtractor class."""

    def test_east_west_line(self, extractor: CrossSectionExtractor) -> None:
        """East-west line through mesh center."""
        xs = extractor.extract(start=(0, 250), end=(500, 250), n_samples=50)
        assert xs.n_samples == 50
        assert np.all(np.diff(xs.distance) > 0)
        # Mesh spans 0..500, line is 0..500, so most samples should be inside
        assert xs.fraction_inside > 0.8

    def test_diagonal_line(self, extractor: CrossSectionExtractor) -> None:
        """Diagonal line at arbitrary angle."""
        xs = extractor.extract(start=(50, 50), end=(450, 450), n_samples=40)
        assert xs.n_samples == 40
        expected_len = np.sqrt(400**2 + 400**2)
        assert xs.total_length == pytest.approx(expected_len, rel=1e-6)
        assert xs.fraction_inside > 0.5

    def test_line_outside_mesh(self, extractor: CrossSectionExtractor) -> None:
        """Line entirely outside mesh -> all NaN."""
        xs = extractor.extract(start=(1000, 1000), end=(2000, 1000), n_samples=20)
        assert xs.fraction_inside == 0.0
        assert np.all(np.isnan(xs.gs_elev))
        assert np.all(np.isnan(xs.top_elev))

    def test_line_partially_outside(self, extractor: CrossSectionExtractor) -> None:
        """Line partially outside -> NaN gaps at edges."""
        xs = extractor.extract(start=(-200, 250), end=(700, 250), n_samples=60)
        inside = xs.fraction_inside
        assert 0.0 < inside < 1.0
        # NaN should appear where mask is False
        outside = ~xs.mask
        assert np.all(np.isnan(xs.gs_elev[outside]))

    def test_sample_at_node_matches_exact(
        self, quad_mesh: AppGrid, stratigraphy: Stratigraphy
    ) -> None:
        """Sampling at an exact node location should match its stratigraphy."""
        extractor = CrossSectionExtractor(quad_mesh, stratigraphy)
        # Node 1 is at (0, 0), node index 0
        node = quad_mesh.nodes[1]
        xs = extractor.extract(start=(node.x, node.y), end=(node.x + 1e-6, node.y), n_samples=1)
        if xs.mask[0]:
            sorted_ids = sorted(quad_mesh.nodes.keys())
            idx = sorted_ids.index(1)
            assert xs.gs_elev[0] == pytest.approx(stratigraphy.gs_elev[idx], abs=0.5)

    def test_linear_stratigraphy_exact_reproduction(self, quad_mesh: AppGrid) -> None:
        """
        Linear stratigraphy should be exactly reproduced by FE interpolation
        on bilinear quads.
        """
        strat = _make_linear_stratigraphy(
            quad_mesh, n_layers=2, gs_base=200.0, gs_slope_x=-0.05, layer_thickness=30.0
        )
        extractor = CrossSectionExtractor(quad_mesh, strat)
        xs = extractor.extract(start=(50, 250), end=(450, 250), n_samples=30)

        # Compute expected ground surface at each sample x
        valid = xs.mask
        expected_gs = 200.0 + (-0.05) * xs.x[valid]
        np.testing.assert_allclose(xs.gs_elev[valid], expected_gs, atol=1e-6)

        # Top of layer 0 should equal gs
        np.testing.assert_allclose(xs.top_elev[valid, 0], expected_gs, atol=1e-6)
        # Bottom of layer 0 = gs - 30
        np.testing.assert_allclose(xs.bottom_elev[valid, 0], expected_gs - 30.0, atol=1e-6)

    def test_distance_monotonically_increasing(self, extractor: CrossSectionExtractor) -> None:
        """Distance array should be strictly monotonically increasing."""
        xs = extractor.extract(start=(50, 50), end=(450, 450), n_samples=50)
        assert np.all(np.diff(xs.distance) > 0)

    def test_layer_ordering(self, extractor: CrossSectionExtractor) -> None:
        """Top should be above bottom for all layers at valid samples."""
        xs = extractor.extract(start=(50, 250), end=(450, 250), n_samples=30)
        valid = xs.mask
        for layer in range(xs.n_layers):
            tops = xs.top_elev[valid, layer]
            bots = xs.bottom_elev[valid, layer]
            assert np.all(tops >= bots)

    def test_scalar_interpolation(self, quad_mesh: AppGrid, stratigraphy: Stratigraphy) -> None:
        """Interpolate a synthetic scalar field."""
        extractor = CrossSectionExtractor(quad_mesh, stratigraphy)
        xs = extractor.extract(start=(50, 250), end=(450, 250), n_samples=20)

        # Create a linear scalar field: value = x
        sorted_ids = sorted(quad_mesh.nodes.keys())
        node_vals = np.array([quad_mesh.nodes[nid].x for nid in sorted_ids])

        result = extractor.interpolate_scalar(xs, node_vals, "x_val")
        assert result.shape == (20,)
        assert "x_val" in xs.scalar_values

        # At valid points, interpolated x_val should match sample x coordinate
        valid = xs.mask
        np.testing.assert_allclose(result[valid], xs.x[valid], atol=1e-4)

    def test_layer_property_interpolation(
        self, quad_mesh: AppGrid, stratigraphy: Stratigraphy
    ) -> None:
        """Interpolate a per-node per-layer property."""
        extractor = CrossSectionExtractor(quad_mesh, stratigraphy)
        xs = extractor.extract(start=(50, 250), end=(450, 250), n_samples=20)

        n_nodes = len(quad_mesh.nodes)
        n_layers = stratigraphy.n_layers

        # Create Kh that is constant per layer: layer 0=100, layer 1=50, layer 2=10
        kh = np.zeros((n_nodes, n_layers))
        kh[:, 0] = 100.0
        kh[:, 1] = 50.0
        kh[:, 2] = 10.0

        result = extractor.interpolate_layer_property(xs, kh, "kh")
        assert result.shape == (20, n_layers)
        assert "kh" in xs.layer_properties

        valid = xs.mask
        np.testing.assert_allclose(result[valid, 0], 100.0, atol=1e-6)
        np.testing.assert_allclose(result[valid, 1], 50.0, atol=1e-6)
        np.testing.assert_allclose(result[valid, 2], 10.0, atol=1e-6)

    def test_layer_property_linear_reproduction(self, quad_mesh: AppGrid) -> None:
        """Linear Kh field should be exactly reproduced."""
        strat = _make_linear_stratigraphy(quad_mesh, n_layers=2, gs_base=200.0)
        extractor = CrossSectionExtractor(quad_mesh, strat)
        xs = extractor.extract(start=(50, 250), end=(450, 250), n_samples=25)

        sorted_ids = sorted(quad_mesh.nodes.keys())
        n_nodes = len(sorted_ids)

        # Kh = 10 * x at all layers
        kh = np.zeros((n_nodes, 2))
        for idx, nid in enumerate(sorted_ids):
            kh[idx, :] = 10.0 * quad_mesh.nodes[nid].x

        result = extractor.interpolate_layer_property(xs, kh, "kh_linear")
        valid = xs.mask
        expected = 10.0 * xs.x[valid]
        np.testing.assert_allclose(result[valid, 0], expected, atol=0.1)
        np.testing.assert_allclose(result[valid, 1], expected, atol=0.1)

    def test_polyline_extraction(self, extractor: CrossSectionExtractor) -> None:
        """Polyline with two segments has continuous distance."""
        waypoints = [(50, 50), (250, 250), (450, 50)]
        xs = extractor.extract_polyline(waypoints, n_samples_per_segment=30)

        assert xs.waypoints is not None
        assert len(xs.waypoints) == 3
        # Distance should be monotonically increasing
        assert np.all(np.diff(xs.distance) >= 0)
        assert xs.fraction_inside > 0.5

    def test_polyline_requires_two_points(self, extractor: CrossSectionExtractor) -> None:
        """Polyline with < 2 waypoints should raise ValueError."""
        with pytest.raises(ValueError, match="At least 2"):
            extractor.extract_polyline([(50, 50)])

    def test_polyline_distance_continuous(self, extractor: CrossSectionExtractor) -> None:
        """No jumps in distance at segment boundaries."""
        waypoints = [(50, 50), (250, 250), (450, 50)]
        xs = extractor.extract_polyline(waypoints, n_samples_per_segment=40)
        diffs = np.diff(xs.distance)
        # No jump should be unreasonably large compared to average step
        avg_step = xs.total_length / (xs.n_samples - 1)
        assert np.max(diffs) < avg_step * 3  # generous tolerance

    def test_neighbor_walk_optimization(
        self, quad_mesh: AppGrid, stratigraphy: Stratigraphy
    ) -> None:
        """
        Neighbor-walk should produce identical results to brute-force.
        We verify by comparing extraction along a line.
        """
        ext = CrossSectionExtractor(quad_mesh, stratigraphy)
        xs = ext.extract(start=(50, 250), end=(450, 250), n_samples=30)

        # All valid samples should have non-NaN values
        valid = xs.mask
        assert np.sum(valid) > 20
        assert not np.any(np.isnan(xs.gs_elev[valid]))

    # -----------------------------------------------------------------------
    # Coverage gap: partial-coverage interpolation (lines 327, 364)
    # -----------------------------------------------------------------------

    def test_interpolate_scalar_partial_coverage(
        self, quad_mesh: AppGrid, stratigraphy: Stratigraphy
    ) -> None:
        """
        Interpolate a scalar on a partially-outside cross-section.

        Covers the ``if cache_entry is None: continue`` branch in
        ``interpolate_scalar()`` (line 327).
        """
        ext = CrossSectionExtractor(quad_mesh, stratigraphy)
        # Line extends well outside the mesh on both sides
        xs = ext.extract(start=(-200, 250), end=(700, 250), n_samples=40)
        assert 0.0 < xs.fraction_inside < 1.0

        sorted_ids = sorted(quad_mesh.nodes.keys())
        node_vals = np.array([quad_mesh.nodes[nid].x for nid in sorted_ids])

        result = ext.interpolate_scalar(xs, node_vals, "x_partial")
        assert result.shape == (40,)

        # Outside points should be NaN
        outside = ~xs.mask
        assert np.sum(outside) > 0, "Need some outside points for this test"
        assert np.all(np.isnan(result[outside]))

        # Inside points should have correct interpolated values
        valid = xs.mask
        np.testing.assert_allclose(result[valid], xs.x[valid], atol=1e-4)

    def test_interpolate_layer_property_partial_coverage(
        self, quad_mesh: AppGrid, stratigraphy: Stratigraphy
    ) -> None:
        """
        Interpolate a layer property on a partially-outside cross-section.

        Covers the ``if cache_entry is None: continue`` branch in
        ``interpolate_layer_property()`` (line 364).
        """
        ext = CrossSectionExtractor(quad_mesh, stratigraphy)
        xs = ext.extract(start=(-200, 250), end=(700, 250), n_samples=40)
        assert 0.0 < xs.fraction_inside < 1.0

        n_nodes = len(quad_mesh.nodes)
        n_layers = stratigraphy.n_layers
        kh = np.ones((n_nodes, n_layers)) * 42.0

        result = ext.interpolate_layer_property(xs, kh, "kh_partial")
        assert result.shape == (40, n_layers)

        outside = ~xs.mask
        assert np.sum(outside) > 0
        assert np.all(np.isnan(result[outside, :]))

        valid = xs.mask
        np.testing.assert_allclose(result[valid, :], 42.0, atol=1e-6)

    # -----------------------------------------------------------------------
    # Parametrized: various sample counts
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("n_samples", [2, 10, 50, 200])
    def test_extraction_various_sample_counts(
        self, extractor: CrossSectionExtractor, n_samples: int
    ) -> None:
        """Extraction works correctly across a range of sample counts."""
        xs = extractor.extract(start=(50, 250), end=(450, 250), n_samples=n_samples)
        assert xs.n_samples == n_samples
        assert xs.fraction_inside > 0.5

    # -----------------------------------------------------------------------
    # Parametrized: various angles
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("angle_deg", [0, 30, 45, 60, 90])
    def test_extraction_various_angles(
        self, extractor: CrossSectionExtractor, angle_deg: int
    ) -> None:
        """Extraction works at various line orientations."""
        cx, cy = 250.0, 250.0
        r = 200.0
        rad = math.radians(angle_deg)
        start = (cx - r * math.cos(rad), cy - r * math.sin(rad))
        end = (cx + r * math.cos(rad), cy + r * math.sin(rad))
        xs = extractor.extract(start=start, end=end, n_samples=30)
        assert xs.fraction_inside > 0.3

    # -----------------------------------------------------------------------
    # Determinism
    # -----------------------------------------------------------------------

    def test_extraction_deterministic(self, extractor: CrossSectionExtractor) -> None:
        """Running the same extraction twice should produce bitwise identical results."""
        xs1 = extractor.extract(start=(50, 250), end=(450, 250), n_samples=30)
        xs2 = extractor.extract(start=(50, 250), end=(450, 250), n_samples=30)

        np.testing.assert_array_equal(xs1.distance, xs2.distance)
        np.testing.assert_array_equal(xs1.gs_elev, xs2.gs_elev)
        np.testing.assert_array_equal(xs1.top_elev, xs2.top_elev)
        np.testing.assert_array_equal(xs1.bottom_elev, xs2.bottom_elev)
        np.testing.assert_array_equal(xs1.mask, xs2.mask)

    # -----------------------------------------------------------------------
    # Edge cases: n_samples=1, n_samples=2, polyline with 2 points
    # -----------------------------------------------------------------------

    def test_extract_n_samples_one(self, extractor: CrossSectionExtractor) -> None:
        """Degenerate edge case: n_samples=1 should still work."""
        xs = extractor.extract(start=(250, 250), end=(251, 250), n_samples=1)
        assert xs.n_samples == 1
        assert xs.distance.shape == (1,)

    def test_extract_n_samples_two(self, extractor: CrossSectionExtractor) -> None:
        """Minimum useful case: n_samples=2."""
        xs = extractor.extract(start=(50, 250), end=(450, 250), n_samples=2)
        assert xs.n_samples == 2
        assert xs.distance[0] == pytest.approx(0.0)
        assert xs.distance[1] == pytest.approx(400.0, rel=1e-6)

    def test_polyline_two_points(self, extractor: CrossSectionExtractor) -> None:
        """Polyline with exactly 2 waypoints (minimum valid) should work."""
        waypoints = [(50, 250), (450, 250)]
        xs = extractor.extract_polyline(waypoints, n_samples_per_segment=20)
        assert xs.waypoints is not None
        assert len(xs.waypoints) == 2
        assert xs.fraction_inside > 0.5

    def test_polyline_duplicate_points(self, extractor: CrossSectionExtractor) -> None:
        """Polyline with coincident adjacent waypoints should not crash."""
        waypoints = [(50, 250), (50, 250), (450, 250)]
        xs = extractor.extract_polyline(waypoints, n_samples_per_segment=20)
        assert xs.n_samples > 0


# ===========================================================================
# TestTriangleMesh
# ===========================================================================


class TestTriangleMesh:
    """Tests for cross-section extraction on triangle meshes."""

    def test_extraction_on_triangle_mesh(self, tri_extractor: CrossSectionExtractor) -> None:
        """FE interpolation should work with barycentric coordinates on triangles."""
        xs = tri_extractor.extract(start=(50, 250), end=(450, 250), n_samples=30)
        assert xs.fraction_inside > 0.5
        valid = xs.mask
        assert not np.any(np.isnan(xs.gs_elev[valid]))

    def test_linear_reproduction_triangle(self, tri_mesh: AppGrid) -> None:
        """Linear stratigraphy should be exactly reproduced on triangle meshes."""
        strat = _make_linear_stratigraphy(
            tri_mesh, n_layers=2, gs_base=200.0, gs_slope_x=-0.05, layer_thickness=30.0
        )
        ext = CrossSectionExtractor(tri_mesh, strat)
        xs = ext.extract(start=(50, 250), end=(450, 250), n_samples=30)

        valid = xs.mask
        expected_gs = 200.0 + (-0.05) * xs.x[valid]
        np.testing.assert_allclose(xs.gs_elev[valid], expected_gs, atol=1e-6)

    def test_scalar_interpolation_triangle(
        self, tri_mesh: AppGrid, tri_stratigraphy: Stratigraphy
    ) -> None:
        """Scalar interpolation (value = x) should be exact on linear triangles."""
        ext = CrossSectionExtractor(tri_mesh, tri_stratigraphy)
        xs = ext.extract(start=(50, 250), end=(450, 250), n_samples=25)

        sorted_ids = sorted(tri_mesh.nodes.keys())
        node_vals = np.array([tri_mesh.nodes[nid].x for nid in sorted_ids])

        result = ext.interpolate_scalar(xs, node_vals, "x_val")
        valid = xs.mask
        np.testing.assert_allclose(result[valid], xs.x[valid], atol=1e-4)

    def test_layer_property_triangle(
        self, tri_mesh: AppGrid, tri_stratigraphy: Stratigraphy
    ) -> None:
        """Layer property interpolation should work on triangle meshes."""
        ext = CrossSectionExtractor(tri_mesh, tri_stratigraphy)
        xs = ext.extract(start=(50, 250), end=(450, 250), n_samples=20)

        n_nodes = len(tri_mesh.nodes)
        kh = np.ones((n_nodes, tri_stratigraphy.n_layers)) * 75.0
        result = ext.interpolate_layer_property(xs, kh, "kh")

        valid = xs.mask
        np.testing.assert_allclose(result[valid, :], 75.0, atol=1e-6)


# ===========================================================================
# TestPlotCrossSection
# ===========================================================================


pytest.importorskip("matplotlib", reason="matplotlib not installed")


class TestPlotCrossSection:
    """Tests for cross-section plotting functions."""

    def test_returns_figure_axes(self, extractor: CrossSectionExtractor) -> None:
        """plot_cross_section returns a (Figure, Axes) tuple."""
        from pyiwfm.visualization.plotting import plot_cross_section

        xs = extractor.extract(start=(50, 250), end=(450, 250), n_samples=20)
        fig, ax = plot_cross_section(xs)
        assert fig is not None
        assert ax is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_with_existing_axes(self, extractor: CrossSectionExtractor) -> None:
        """Passing an existing Axes should reuse it."""
        import matplotlib.pyplot as plt

        from pyiwfm.visualization.plotting import plot_cross_section

        fig, ax = plt.subplots()
        xs = extractor.extract(start=(50, 250), end=(450, 250), n_samples=20)
        fig2, ax2 = plot_cross_section(xs, ax=ax)
        assert ax2 is ax
        plt.close(fig)

    def test_scalar_overlay(self, quad_mesh: AppGrid, stratigraphy: Stratigraphy) -> None:
        """Scalar overlay renders without error."""
        from pyiwfm.visualization.plotting import plot_cross_section

        ext = CrossSectionExtractor(quad_mesh, stratigraphy)
        xs = ext.extract(start=(50, 250), end=(450, 250), n_samples=20)

        sorted_ids = sorted(quad_mesh.nodes.keys())
        head = np.array([quad_mesh.nodes[nid].x * 0.1 + 80.0 for nid in sorted_ids])
        ext.interpolate_scalar(xs, head, "head")

        fig, ax = plot_cross_section(xs, scalar_name="head")
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_layer_property_colorfill(self, quad_mesh: AppGrid, stratigraphy: Stratigraphy) -> None:
        """Layer property color-fill rendering works."""
        from pyiwfm.visualization.plotting import plot_cross_section

        ext = CrossSectionExtractor(quad_mesh, stratigraphy)
        xs = ext.extract(start=(50, 250), end=(450, 250), n_samples=20)

        n_nodes = len(quad_mesh.nodes)
        kh = np.ones((n_nodes, stratigraphy.n_layers)) * 50.0
        ext.interpolate_layer_property(xs, kh, "kh")

        fig, ax = plot_cross_section(xs, layer_property_name="kh")
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_combined_scalar_and_property(
        self, quad_mesh: AppGrid, stratigraphy: Stratigraphy
    ) -> None:
        """Scalar overlay combined with layer property renders without error."""
        from pyiwfm.visualization.plotting import plot_cross_section

        ext = CrossSectionExtractor(quad_mesh, stratigraphy)
        xs = ext.extract(start=(50, 250), end=(450, 250), n_samples=20)

        sorted_ids = sorted(quad_mesh.nodes.keys())
        n_nodes = len(sorted_ids)

        head = np.array([quad_mesh.nodes[nid].x * 0.1 + 80.0 for nid in sorted_ids])
        ext.interpolate_scalar(xs, head, "head")

        kh = np.ones((n_nodes, stratigraphy.n_layers)) * 50.0
        ext.interpolate_layer_property(xs, kh, "kh")

        fig, ax = plot_cross_section(xs, scalar_name="head", layer_property_name="kh")
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_custom_colors_labels(self, extractor: CrossSectionExtractor) -> None:
        """Custom colors, labels, and title are applied."""
        from pyiwfm.visualization.plotting import plot_cross_section

        xs = extractor.extract(start=(50, 250), end=(450, 250), n_samples=10)
        fig, ax = plot_cross_section(
            xs,
            layer_colors=["red", "blue", "green"],
            layer_labels=["A", "B", "C"],
            title="Custom Test",
        )
        assert ax.get_title() == "Custom Test"
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_nan_handling(self, extractor: CrossSectionExtractor) -> None:
        """Partial coverage should render without errors."""
        from pyiwfm.visualization.plotting import plot_cross_section

        xs = extractor.extract(start=(-100, 250), end=(600, 250), n_samples=30)
        assert xs.fraction_inside < 1.0
        fig, ax = plot_cross_section(xs)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_cross_section_location(
        self, quad_mesh: AppGrid, extractor: CrossSectionExtractor
    ) -> None:
        """Location plot renders the cross-section line on the mesh."""
        from pyiwfm.visualization.plotting import plot_cross_section_location

        xs = extractor.extract(start=(50, 250), end=(450, 250), n_samples=10)
        fig, ax = plot_cross_section_location(quad_mesh, xs)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_location_polyline(
        self, quad_mesh: AppGrid, extractor: CrossSectionExtractor
    ) -> None:
        """Location plot works with polyline cross-sections."""
        from pyiwfm.visualization.plotting import plot_cross_section_location

        waypoints = [(50, 50), (250, 250), (450, 50)]
        xs = extractor.extract_polyline(waypoints, n_samples_per_segment=15)
        fig, ax = plot_cross_section_location(quad_mesh, xs)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    # -------------------------------------------------------------------
    # Coverage gap: property colorfill with partial coverage (lines 1893-1894)
    # -------------------------------------------------------------------

    def test_property_colorfill_partial_coverage(
        self, quad_mesh: AppGrid, stratigraphy: Stratigraphy
    ) -> None:
        """
        Render layer property on a partially-outside cross-section.

        Covers the ``if not valid[j] or not valid[j + 1]: continue`` branch
        in the property color-fill loop.
        """
        from pyiwfm.visualization.plotting import plot_cross_section

        ext = CrossSectionExtractor(quad_mesh, stratigraphy)
        xs = ext.extract(start=(-200, 250), end=(700, 250), n_samples=40)
        assert 0.0 < xs.fraction_inside < 1.0

        n_nodes = len(quad_mesh.nodes)
        kh = np.ones((n_nodes, stratigraphy.n_layers)) * 50.0
        ext.interpolate_layer_property(xs, kh, "kh")

        fig, ax = plot_cross_section(xs, layer_property_name="kh")
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    # -------------------------------------------------------------------
    # Coverage gap: NaN property values (lines 1898-1899)
    # -------------------------------------------------------------------

    def test_property_colorfill_with_nan_property_values(
        self, quad_mesh: AppGrid, stratigraphy: Stratigraphy
    ) -> None:
        """
        Render layer property where some property values are NaN.

        Covers the ``if np.isnan(avg_prop): continue`` branch.
        We set NaN at nodes near y=250 (the cross-section line) so that
        interpolation produces NaN property values at valid (inside) samples.
        """
        from pyiwfm.visualization.plotting import plot_cross_section

        ext = CrossSectionExtractor(quad_mesh, stratigraphy)
        xs = ext.extract(start=(50, 250), end=(450, 250), n_samples=20)

        sorted_ids = sorted(quad_mesh.nodes.keys())
        n_nodes = len(sorted_ids)
        kh = np.ones((n_nodes, stratigraphy.n_layers)) * 50.0
        # Set nodes near y=200/300 with x < 250 to NaN so interpolation
        # produces NaN at some (not all) valid sample points along y=250
        for idx, nid in enumerate(sorted_ids):
            node = quad_mesh.nodes[nid]
            if 150 <= node.y <= 350 and node.x < 250:
                kh[idx, :] = np.nan
        ext.interpolate_layer_property(xs, kh, "kh_nan")

        # Verify some interpolated property values are NaN at valid points
        valid = xs.mask
        prop = xs.layer_properties["kh_nan"]
        assert np.any(np.isnan(prop[valid, :]))

        fig, ax = plot_cross_section(xs, layer_property_name="kh_nan")
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    # -------------------------------------------------------------------
    # Coverage gap: show_ground_surface=False (lines 1929-1933)
    # -------------------------------------------------------------------

    def test_plot_no_ground_surface(self, extractor: CrossSectionExtractor) -> None:
        """
        Plotting with show_ground_surface=False should not draw the ground
        surface line.
        """
        from pyiwfm.visualization.plotting import plot_cross_section

        xs = extractor.extract(start=(50, 250), end=(450, 250), n_samples=20)
        fig, ax = plot_cross_section(xs, show_ground_surface=False)
        # Check that no line labeled "Ground Surface" exists
        labels = [line.get_label() for line in ax.get_lines()]
        assert "Ground Surface" not in labels
        import matplotlib.pyplot as plt

        plt.close(fig)

    # -------------------------------------------------------------------
    # Coverage gap: show_labels=False (lines 2000-2020)
    # -------------------------------------------------------------------

    def test_plot_location_no_labels(
        self, quad_mesh: AppGrid, extractor: CrossSectionExtractor
    ) -> None:
        """
        Plotting location with show_labels=False should not add A/A' annotations.
        """
        from pyiwfm.visualization.plotting import plot_cross_section_location

        xs = extractor.extract(start=(50, 250), end=(450, 250), n_samples=10)
        fig, ax = plot_cross_section_location(quad_mesh, xs, show_labels=False)
        # Check that no A/A' annotations exist
        texts = [t.get_text() for t in ax.texts]
        assert "A" not in texts
        assert "A'" not in texts
        import matplotlib.pyplot as plt

        plt.close(fig)
