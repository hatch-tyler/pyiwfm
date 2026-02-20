"""Tests for build_tutorial_model() and plot_lakes()."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from pyiwfm.components.groundwater import AppGW
from pyiwfm.components.lake import AppLake
from pyiwfm.components.rootzone import RootZone
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
            "groundwater",
            "stream",
            "lakes",
            "lake_elem_ids",
            "rootzone",
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
        assert model.gs_elev.min() == pytest.approx(250.0, abs=1.0)
        assert model.gs_elev.max() == pytest.approx(500.0, abs=1.0)

    def test_stratigraphy(self, model):
        assert isinstance(model.stratigraphy, Stratigraphy)
        assert model.stratigraphy.n_layers == 2
        assert model.stratigraphy.n_nodes == 441

    def test_initial_heads_shape(self, model):
        assert model.initial_heads.shape == (441, 2)

    def test_initial_heads_values(self, model):
        """Initial heads are uniform 280/290, not gs_elev offsets."""
        np.testing.assert_allclose(model.initial_heads[:, 0], 280.0)
        np.testing.assert_allclose(model.initial_heads[:, 1], 290.0)

    # -- Groundwater component --

    def test_groundwater_component(self, model):
        gw = model.groundwater
        assert isinstance(gw, AppGW)
        assert gw.n_nodes == 441
        assert gw.n_layers == 2
        assert gw.aquifer_params is not None
        assert len(gw.boundary_conditions) == 2
        assert len(gw.element_pumping) == 5
        assert len(gw.tile_drains) == 21
        assert len(gw.node_subsidence) == 441
        assert len(gw.hydrograph_locations) == 42

    def test_aquifer_params_shape(self, model):
        ap = model.groundwater.aquifer_params
        assert ap.kh is not None
        assert ap.kh.shape == (441, 2)
        np.testing.assert_allclose(ap.kh, 50.0)

    def test_boundary_condition_nodes(self, model):
        """Each BC has 21 nodes (west and east boundaries)."""
        for bc in model.groundwater.boundary_conditions:
            assert len(bc.nodes) == 21

    def test_element_pumping_ids(self, model):
        pump_ids = sorted(ep.element_id for ep in model.groundwater.element_pumping)
        assert pump_ids == [73, 134, 193, 274, 333]

    def test_tile_drain_count(self, model):
        assert len(model.groundwater.tile_drains) == 21
        for td in model.groundwater.tile_drains.values():
            assert td.elevation == pytest.approx(280.0)
            assert td.conductance == pytest.approx(20_000.0)

    def test_node_subsidence_uniform(self, model):
        for ns in model.groundwater.node_subsidence:
            assert ns.elastic_sc == [5e-6, 5e-6]
            assert ns.inelastic_sc == [5e-5, 5e-5]

    def test_hydrograph_locations_layers(self, model):
        """42 hydrograph locations: 21 nodes x 2 layers."""
        layers = [h.layer for h in model.groundwater.hydrograph_locations]
        assert layers.count(1) == 21
        assert layers.count(2) == 21

    # -- Stream component --

    def test_stream(self, model):
        assert isinstance(model.stream, AppStream)
        assert model.stream.n_nodes == 23
        assert model.stream.n_reaches == 3

    def test_stream_bed_params(self, model):
        for sn in model.stream.nodes.values():
            assert sn.conductivity == pytest.approx(10.0)
            assert sn.bed_thickness == pytest.approx(1.0)
            assert sn.wetted_perimeter == pytest.approx(150.0)

    def test_stream_diversions(self, model):
        assert len(model.stream.diversions) == 5

    def test_stream_bypasses(self, model):
        assert len(model.stream.bypasses) == 2
        # Second bypass has a rating table
        bp2 = model.stream.bypasses[2]
        assert len(bp2.rating_table_flows) == 4
        assert len(bp2.rating_table_spills) == 4

    # -- Lake component --

    def test_lakes(self, model):
        assert isinstance(model.lakes, AppLake)
        assert model.lakes.n_lakes == 1
        lake_elems = model.lakes.get_elements_for_lake(1)
        assert len(lake_elems) == 10

    def test_lake_params(self, model):
        lake = model.lakes.lakes[1]
        assert lake.bed_conductivity == pytest.approx(2.0)
        assert lake.bed_thickness == pytest.approx(1.0)
        assert lake.initial_elevation == pytest.approx(280.0)
        assert lake.et_column == 7
        assert lake.precip_column == 2
        assert lake.max_elev_column == 1
        assert lake.outflow is not None
        assert lake.outflow.destination_type == "stream"
        assert lake.outflow.destination_id == 10

    def test_lake_elem_ids(self, model):
        assert len(model.lake_elem_ids) == 10
        for eid in model.lake_elem_ids:
            assert eid in model.grid.elements

    # -- Root Zone component --

    def test_rootzone_component(self, model):
        rz = model.rootzone
        assert isinstance(rz, RootZone)
        assert len(rz.crop_types) == 7
        assert len(rz.soil_params) == 400

    def test_rootzone_crop_root_depths(self, model):
        rz = model.rootzone
        assert rz.crop_types[1].root_depth == pytest.approx(5.0)  # TO
        assert rz.crop_types[2].root_depth == pytest.approx(6.0)  # AL
        assert rz.crop_types[3].root_depth == pytest.approx(3.0)  # RICE_FL

    def test_rootzone_sandy_clay_split(self, model):
        rz = model.rootzone
        # Sandy soils (elements 1-200)
        assert rz.soil_params[1].porosity == pytest.approx(0.45)
        assert rz.soil_params[100].lambda_param == pytest.approx(0.62)
        # Clay soils (elements 201-400)
        assert rz.soil_params[201].porosity == pytest.approx(0.50)
        assert rz.soil_params[400].lambda_param == pytest.approx(0.36)

    # -- Other existing tests --

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
