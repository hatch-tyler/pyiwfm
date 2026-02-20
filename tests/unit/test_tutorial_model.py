"""Tests for build_tutorial_model() and plot_lakes()."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from pyiwfm.components.lake import AppLake
from pyiwfm.components.stream import AppStream
from pyiwfm.core.mesh import AppGrid
from pyiwfm.core.stratigraphy import Stratigraphy
from pyiwfm.core.timeseries import TimeSeries
from pyiwfm.sample_models import build_tutorial_model
from pyiwfm.visualization.plotting import plot_lakes

# ---------------------------------------------------------------------------
# build_tutorial_model tests
# ---------------------------------------------------------------------------


class TestBuildTutorialModel:
    """Tests for the build_tutorial_model() helper."""

    @pytest.fixture()
    def model(self):
        return build_tutorial_model()

    def test_returns_namespace(self, model):
        attrs = [
            "grid",
            "gs_elev",
            "stratigraphy",
            "initial_heads",
            "stream",
            "lakes",
            "lake_elem_ids",
            "final_heads",
            "head_timeseries",
            "gw_budget",
            "gw_budget_timeseries",
            "rz_budget",
        ]
        for attr in attrs:
            assert hasattr(model, attr), f"Missing attribute: {attr}"

    def test_grid_dimensions(self, model):
        assert isinstance(model.grid, AppGrid)
        assert model.grid.n_nodes == 441
        assert model.grid.n_elements == 400

    def test_gs_elev_range(self, model):
        assert model.gs_elev.shape == (441,)
        assert model.gs_elev.min() == pytest.approx(200.0, abs=1.0)
        assert model.gs_elev.max() == pytest.approx(400.0, abs=1.0)

    def test_stratigraphy(self, model):
        assert isinstance(model.stratigraphy, Stratigraphy)
        assert model.stratigraphy.n_layers == 2
        assert model.stratigraphy.n_nodes == 441

    def test_initial_heads_shape(self, model):
        assert model.initial_heads.shape == (441, 2)

    def test_stream(self, model):
        assert isinstance(model.stream, AppStream)
        assert model.stream.n_nodes == 21
        assert model.stream.n_reaches == 3

    def test_lakes(self, model):
        assert isinstance(model.lakes, AppLake)
        assert model.lakes.n_lakes == 1
        lake_elems = model.lakes.get_elements_for_lake(1)
        assert len(lake_elems) == 9

    def test_lake_elem_ids(self, model):
        assert len(model.lake_elem_ids) == 9
        for eid in model.lake_elem_ids:
            assert eid in model.grid.elements

    def test_final_heads(self, model):
        assert model.final_heads.shape == (441,)

    def test_head_timeseries(self, model):
        assert isinstance(model.head_timeseries, list)
        assert len(model.head_timeseries) == 3
        for ts in model.head_timeseries:
            assert isinstance(ts, TimeSeries)

    def test_gw_budget(self, model):
        assert isinstance(model.gw_budget, dict)
        assert len(model.gw_budget) > 0

    def test_gw_budget_timeseries(self, model):
        times, components = model.gw_budget_timeseries
        assert len(times) == 10
        assert isinstance(components, dict)
        for arr in components.values():
            assert len(arr) == 10

    def test_rz_budget(self, model):
        assert isinstance(model.rz_budget, dict)
        assert len(model.rz_budget) > 0

    def test_deterministic(self):
        """Calling twice produces identical results."""
        m1 = build_tutorial_model()
        m2 = build_tutorial_model()
        np.testing.assert_array_equal(m1.gs_elev, m2.gs_elev)
        np.testing.assert_array_equal(m1.final_heads, m2.final_heads)


# ---------------------------------------------------------------------------
# plot_lakes tests
# ---------------------------------------------------------------------------


class TestPlotLakes:
    """Tests for the plot_lakes() function."""

    @pytest.fixture()
    def model(self):
        return build_tutorial_model()

    def test_basic_call(self, model):
        fig, ax = plot_lakes(model.lakes, model.grid)
        assert fig is not None
        assert ax is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_existing_axes(self, model):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        fig2, ax2 = plot_lakes(model.lakes, model.grid, ax=ax)
        assert ax2 is ax
        plt.close(fig)

    def test_with_cmap(self, model):
        fig, ax = plot_lakes(model.lakes, model.grid, cmap="tab10")
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_no_labels(self, model):
        fig, ax = plot_lakes(model.lakes, model.grid, show_labels=False)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_empty_lake_component(self, model):
        """plot_lakes handles an AppLake with no lakes."""
        empty = AppLake()
        fig, ax = plot_lakes(empty, model.grid)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)
