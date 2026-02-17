"""Property-based tests for I/O roundtrip consistency using Hypothesis."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pyiwfm.core.mesh import AppGrid, Element, Node, Subregion
from pyiwfm.core.stratigraphy import Stratigraphy


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

reasonable_float = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
positive_float = st.floats(min_value=0.1, max_value=1e6, allow_nan=False, allow_infinity=False)


@st.composite
def stratigraphy_strategy(draw: st.DrawFn) -> Stratigraphy:
    """Generate a random Stratigraphy with consistent layer ordering."""
    n_nodes = draw(st.integers(min_value=4, max_value=20))
    n_layers = draw(st.integers(min_value=1, max_value=4))

    # Ground surface elevations
    gs_elev = np.array([draw(positive_float) for _ in range(n_nodes)])

    # Ensure layers are properly ordered (top > bottom, each layer below previous)
    layer_thickness = draw(st.floats(min_value=5.0, max_value=50.0))

    top_elev = np.zeros((n_nodes, n_layers))
    bottom_elev = np.zeros((n_nodes, n_layers))

    for layer in range(n_layers):
        top_elev[:, layer] = gs_elev - layer * layer_thickness
        bottom_elev[:, layer] = gs_elev - (layer + 1) * layer_thickness

    active_node = np.ones((n_nodes, n_layers), dtype=bool)

    return Stratigraphy(
        n_layers=n_layers,
        n_nodes=n_nodes,
        gs_elev=gs_elev,
        top_elev=top_elev,
        bottom_elev=bottom_elev,
        active_node=active_node,
    )


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


@pytest.mark.property
class TestStratigraphyProperties:
    """Property-based tests for Stratigraphy invariants."""

    @given(stratigraphy_strategy())
    @settings(max_examples=30)
    def test_layer_tops_above_bottoms(self, strat: Stratigraphy) -> None:
        """Every layer top must be above its bottom."""
        for layer in range(strat.n_layers):
            assert np.all(strat.top_elev[:, layer] >= strat.bottom_elev[:, layer])

    @given(stratigraphy_strategy())
    @settings(max_examples=30)
    def test_layers_are_stacked(self, strat: Stratigraphy) -> None:
        """Each layer bottom equals the next layer top."""
        for layer in range(strat.n_layers - 1):
            np.testing.assert_allclose(
                strat.bottom_elev[:, layer],
                strat.top_elev[:, layer + 1],
            )

    @given(stratigraphy_strategy())
    @settings(max_examples=30)
    def test_ground_surface_equals_first_layer_top(self, strat: Stratigraphy) -> None:
        """Ground surface elevation equals the top of layer 1."""
        np.testing.assert_allclose(strat.gs_elev, strat.top_elev[:, 0])

    @given(stratigraphy_strategy())
    @settings(max_examples=30)
    def test_total_thickness_positive(self, strat: Stratigraphy) -> None:
        """Total thickness (GS - bottom of last layer) is positive."""
        total = strat.gs_elev - strat.bottom_elev[:, -1]
        assert np.all(total > 0)

    @given(stratigraphy_strategy())
    @settings(max_examples=30)
    def test_shape_consistency(self, strat: Stratigraphy) -> None:
        """Array shapes are consistent with n_nodes and n_layers."""
        assert strat.gs_elev.shape == (strat.n_nodes,)
        assert strat.top_elev.shape == (strat.n_nodes, strat.n_layers)
        assert strat.bottom_elev.shape == (strat.n_nodes, strat.n_layers)
        assert strat.active_node.shape == (strat.n_nodes, strat.n_layers)


@pytest.mark.property
class TestIWFMReaderProperties:
    """Property-based tests for IWFM reader helper functions."""

    @given(st.text(min_size=0, max_size=200))
    def test_is_comment_line_returns_bool(self, line: str) -> None:
        from pyiwfm.io.iwfm_reader import is_comment_line
        result = is_comment_line(line)
        assert isinstance(result, bool)

    @given(st.text(min_size=1, max_size=200))
    def test_strip_inline_comment_returns_pair(self, line: str) -> None:
        from pyiwfm.io.iwfm_reader import strip_inline_comment
        value, desc = strip_inline_comment(line)
        assert isinstance(value, str)
        assert isinstance(desc, str)

    @given(st.text(min_size=1, max_size=200).filter(lambda s: "/" not in s))
    def test_no_slash_means_no_comment(self, line: str) -> None:
        from pyiwfm.io.iwfm_reader import strip_inline_comment
        value, desc = strip_inline_comment(line)
        assert desc == ""
        assert value == line.strip()

    @given(
        st.from_regex(r"[0-9]+", fullmatch=True),
        st.text(min_size=0, max_size=20),
    )
    def test_parse_int_roundtrip(self, num_str: str, ctx: str) -> None:
        from pyiwfm.io.iwfm_reader import parse_int
        result = parse_int(num_str, ctx)
        assert result == int(num_str)

    @given(
        st.floats(min_value=-1e10, max_value=1e10, allow_nan=False, allow_infinity=False),
    )
    def test_parse_float_roundtrip(self, num: float) -> None:
        from pyiwfm.io.iwfm_reader import parse_float
        result = parse_float(str(num))
        np.testing.assert_allclose(result, num, rtol=1e-10)

    @given(st.text(min_size=1, max_size=50).filter(lambda s: not s.strip().replace(".", "").replace("-", "").replace("+", "").replace("e", "").replace("E", "").isdigit()))
    def test_parse_int_bad_input_raises(self, bad_str: str) -> None:
        from pyiwfm.io.iwfm_reader import parse_int
        from pyiwfm.core.exceptions import FileFormatError
        with pytest.raises(FileFormatError):
            parse_int(bad_str, "test")
