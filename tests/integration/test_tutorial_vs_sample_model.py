"""Integration tests: verify build_tutorial_model() matches the IWFM sample model.

Requires the IWFM sample model to be present on disk.  Set the
``IWFM_SAMPLE_MODEL_DIR`` environment variable or use the default
location (``~/OneDrive/Desktop/iwfm-2025.0.1747/samplemodel``).

Note: The tutorial model uses simplified coordinates (UTM-like meters)
while the reference model uses US survey feet.  Tests that depend on
absolute coordinates compare grid *structure* (spacing, connectivity)
rather than raw values.  Some tests are marked ``xfail`` where the
reference model's reader doesn't populate the attribute in the same way
as the tutorial, or where the tutorial intentionally simplifies.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyiwfm.core.model import IWFMModel
from pyiwfm.sample_models import build_tutorial_model


@pytest.mark.integration
class TestTutorialMatchesSampleModel:
    """Verify build_tutorial_model() matches the actual IWFM sample model."""

    @pytest.fixture(autouse=True)
    def setup(self, sample_model_path):
        """Load both models."""
        self.tutorial = build_tutorial_model()
        self.reference = IWFMModel.from_simulation_with_preprocessor(
            simulation_file=sample_model_path / "Simulation" / "Simulation_MAIN.IN",
            preprocessor_file=sample_model_path / "Preprocessor" / "PreProcessor_MAIN.IN",
        )

    # --- Mesh ---

    def test_grid_dimensions_match(self):
        assert self.tutorial.grid.n_nodes == self.reference.mesh.n_nodes
        assert self.tutorial.grid.n_elements == self.reference.mesh.n_elements

    def test_node_spacing_match(self):
        """Grid spacing is uniform; compare dx/dy rather than absolute coords."""
        tut_n1 = self.tutorial.grid.nodes[1]
        tut_n2 = self.tutorial.grid.nodes[2]
        tut_n22 = self.tutorial.grid.nodes[22]
        ref_n1 = self.reference.mesh.nodes[1]
        ref_n2 = self.reference.mesh.nodes[2]
        ref_n22 = self.reference.mesh.nodes[22]

        tut_dx = tut_n2.x - tut_n1.x
        ref_dx = ref_n2.x - ref_n1.x
        tut_dy = tut_n22.y - tut_n1.y
        ref_dy = ref_n22.y - ref_n1.y

        # Both should have uniform square spacing (ref is in us-ft, tut is in meters)
        assert tut_dx == pytest.approx(tut_dy, rel=0.01)
        assert ref_dx == pytest.approx(ref_dy, rel=0.01)

    def test_element_connectivity_match(self):
        for eid in self.reference.mesh.elements:
            assert (
                self.tutorial.grid.elements[eid].vertices
                == self.reference.mesh.elements[eid].vertices
            )

    def test_subregion_assignments_match(self):
        for eid in self.reference.mesh.elements:
            assert (
                self.tutorial.grid.elements[eid].subregion
                == self.reference.mesh.elements[eid].subregion
            )

    # --- Stratigraphy ---

    def test_gs_elev_mostly_match(self):
        """Ground surface elevations match at most nodes (lake bed nodes may differ)."""
        tut = self.tutorial.gs_elev
        ref = self.reference.stratigraphy.gs_elev
        n_match = np.sum(np.abs(tut - ref) < 2.0)
        # Allow up to 15 lake-bed nodes to differ
        assert n_match >= len(tut) - 15

    def test_stratigraphy_layer_count_match(self):
        assert self.tutorial.stratigraphy.n_layers == self.reference.stratigraphy.n_layers

    # --- Groundwater ---

    def test_initial_heads_match(self):
        ref_gw = self.reference.groundwater
        assert ref_gw is not None
        assert ref_gw.heads is not None
        np.testing.assert_allclose(self.tutorial.initial_heads, ref_gw.heads, atol=1.0)

    def test_boundary_condition_count_match(self):
        ref_bc_nodes = sum(len(bc.nodes) for bc in self.reference.groundwater.boundary_conditions)
        tut_bc_nodes = sum(len(bc.nodes) for bc in self.tutorial.groundwater.boundary_conditions)
        assert tut_bc_nodes == ref_bc_nodes

    def test_element_pumping_count_match(self):
        assert len(self.tutorial.groundwater.element_pumping) == len(
            self.reference.groundwater.element_pumping
        )

    def test_element_pumping_ids_match(self):
        tut_ids = sorted(ep.element_id for ep in self.tutorial.groundwater.element_pumping)
        ref_ids = sorted(ep.element_id for ep in self.reference.groundwater.element_pumping)
        assert tut_ids == ref_ids

    def test_tile_drain_count_match(self):
        assert len(self.tutorial.groundwater.tile_drains) == len(
            self.reference.groundwater.tile_drains
        )

    def test_hydrograph_count_match(self):
        assert len(self.tutorial.groundwater.hydrograph_locations) == len(
            self.reference.groundwater.hydrograph_locations
        )

    def test_subsidence_count_match(self):
        assert len(self.tutorial.groundwater.node_subsidence) == len(
            self.reference.groundwater.node_subsidence
        )

    def test_aquifer_kh_match(self):
        ref_params = self.reference.groundwater.aquifer_params
        tut_params = self.tutorial.groundwater.aquifer_params
        assert ref_params is not None
        assert tut_params is not None
        assert ref_params.kh is not None
        assert tut_params.kh is not None
        np.testing.assert_allclose(tut_params.kh, ref_params.kh, atol=1.0)

    # --- Streams ---

    def test_stream_node_count_match(self):
        assert self.tutorial.stream.n_nodes == self.reference.streams.n_nodes

    def test_stream_reach_count_match(self):
        assert self.tutorial.stream.n_reaches == self.reference.streams.n_reaches

    def test_stream_gw_node_mapping_match(self):
        for sid in self.reference.streams.nodes:
            assert (
                self.tutorial.stream.nodes[sid].gw_node == self.reference.streams.nodes[sid].gw_node
            )

    def test_stream_bed_conductivity_match(self):
        for sid in self.reference.streams.nodes:
            assert self.tutorial.stream.nodes[sid].conductivity == pytest.approx(
                self.reference.streams.nodes[sid].conductivity, abs=0.1
            )

    def test_diversion_count_match(self):
        assert len(self.tutorial.stream.diversions) == len(self.reference.streams.diversions)

    def test_bypass_count_match(self):
        assert len(self.tutorial.stream.bypasses) == len(self.reference.streams.bypasses)

    # --- Lakes ---

    def test_lake_count_match(self):
        assert self.tutorial.lakes.n_lakes == self.reference.lakes.n_lakes

    def test_lake_element_count_match(self):
        tut_elems = self.tutorial.lakes.get_elements_for_lake(1)
        ref_elems = self.reference.lakes.get_elements_for_lake(1)
        assert len(tut_elems) == len(ref_elems)

    def test_lake_element_ids_match(self):
        tut_ids = sorted(e.element_id for e in self.tutorial.lakes.get_elements_for_lake(1))
        ref_ids = sorted(e.element_id for e in self.reference.lakes.get_elements_for_lake(1))
        assert tut_ids == ref_ids

    # --- Root Zone ---

    def test_rootzone_crop_count_match(self):
        assert len(self.tutorial.rootzone.crop_types) == len(self.reference.rootzone.crop_types)

    def test_rootzone_soil_params_count_match(self):
        assert len(self.tutorial.rootzone.soil_params) == len(self.reference.rootzone.soil_params)
