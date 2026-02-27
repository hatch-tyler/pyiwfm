"""Tests for remaining model_factory functions: apply_kh_anomalies,
apply_parametric_grids, apply_parametric_subsidence, binary_data_to_model."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pyiwfm.components.groundwater import AquiferParameters
from pyiwfm.core.model_factory import (
    apply_kh_anomalies,
    apply_parametric_grids,
    apply_parametric_subsidence,
    binary_data_to_model,
)
from pyiwfm.io.groundwater import KhAnomalyEntry
from tests.conftest import make_simple_grid, make_simple_stratigraphy


# ---------------------------------------------------------------------------
# apply_kh_anomalies
# ---------------------------------------------------------------------------


class TestApplyKhAnomalies:
    """Tests for apply_kh_anomalies."""

    def test_anomaly_applied_to_vertex_nodes(self) -> None:
        """Anomaly Kh values are written to all vertex nodes of the element."""
        mesh = make_simple_grid()
        n_nodes = len(mesh.nodes)
        n_layers = 2
        kh = np.ones((n_nodes, n_layers), dtype=np.float64)
        params = AquiferParameters(n_nodes=n_nodes, n_layers=n_layers, kh=kh)

        # Element 1 has vertices (1, 2, 5, 4) -> 0-based indices 0, 1, 4, 3
        anomaly = KhAnomalyEntry(element_id=1, kh_per_layer=[10.0, 20.0])
        applied = apply_kh_anomalies(params, [anomaly], mesh)

        assert applied == 1
        # Check the vertex nodes of element 1 received anomaly values
        node_id_to_idx = {nid: i for i, nid in enumerate(sorted(mesh.nodes.keys()))}
        for nid in mesh.elements[1].vertices:
            idx = node_id_to_idx[nid]
            assert params.kh[idx, 0] == 10.0
            assert params.kh[idx, 1] == 20.0

    def test_missing_element_skipped(self) -> None:
        """Anomaly with non-existent element_id is skipped."""
        mesh = make_simple_grid()
        n_nodes = len(mesh.nodes)
        kh = np.ones((n_nodes, 1), dtype=np.float64)
        params = AquiferParameters(n_nodes=n_nodes, n_layers=1, kh=kh)

        anomaly = KhAnomalyEntry(element_id=999, kh_per_layer=[50.0])
        applied = apply_kh_anomalies(params, [anomaly], mesh)

        assert applied == 0
        # All values remain unchanged
        np.testing.assert_array_equal(params.kh, np.ones((n_nodes, 1)))

    def test_multi_layer_partial_anomaly(self) -> None:
        """When anomaly has fewer layers than params, only those layers update."""
        mesh = make_simple_grid()
        n_nodes = len(mesh.nodes)
        n_layers = 3
        kh = np.zeros((n_nodes, n_layers), dtype=np.float64)
        params = AquiferParameters(n_nodes=n_nodes, n_layers=n_layers, kh=kh)

        # Provide only 2 layers of anomaly for a 3-layer model
        anomaly = KhAnomalyEntry(element_id=2, kh_per_layer=[5.0, 7.0])
        applied = apply_kh_anomalies(params, [anomaly], mesh)

        assert applied == 1
        node_id_to_idx = {nid: i for i, nid in enumerate(sorted(mesh.nodes.keys()))}
        for nid in mesh.elements[2].vertices:
            idx = node_id_to_idx[nid]
            assert params.kh[idx, 0] == 5.0
            assert params.kh[idx, 1] == 7.0
            assert params.kh[idx, 2] == 0.0  # untouched

    def test_kh_none_returns_zero(self) -> None:
        """If params.kh is None, returns 0 immediately."""
        params = AquiferParameters(n_nodes=9, n_layers=1, kh=None)
        mesh = make_simple_grid()
        anomaly = KhAnomalyEntry(element_id=1, kh_per_layer=[1.0])
        assert apply_kh_anomalies(params, [anomaly], mesh) == 0


# ---------------------------------------------------------------------------
# apply_parametric_grids
# ---------------------------------------------------------------------------


class TestApplyParametricGrids:
    """Tests for apply_parametric_grids."""

    def test_uniform_single_node_grid(self) -> None:
        """A single-node grid with 0 elements applies uniform values everywhere."""
        mesh = make_simple_grid()
        n_nodes = len(mesh.nodes)
        n_layers = 1

        # Build a mock parametric grid: 1 node, 0 elements -> uniform
        grid_data = MagicMock()
        grid_data.n_nodes = 1
        grid_data.n_elements = 0
        # node_values shape: (1, n_layers, 5) — Kh, Ss, Sy, AquitardKv, Kv
        grid_data.node_values = np.array([[[3.0, 0.01, 0.2, 0.001, 1.5]]])

        gw = MagicMock()
        result = apply_parametric_grids(gw, [grid_data], mesh)

        assert result is True
        # Verify set_aquifer_parameters was called
        assert gw.set_aquifer_parameters.called or hasattr(gw, "aquifer_params")
        call_args = gw.set_aquifer_parameters.call_args
        params = call_args[0][0]
        assert params.n_nodes == n_nodes
        assert params.n_layers == n_layers
        # All nodes should have Kh=3.0
        np.testing.assert_array_equal(params.kh[:, 0], 3.0)
        np.testing.assert_array_equal(params.specific_storage[:, 0], 0.01)
        np.testing.assert_array_equal(params.specific_yield[:, 0], 0.2)

    def test_empty_mesh_returns_false(self) -> None:
        """An empty mesh (0 nodes) causes early return of False."""
        from pyiwfm.core.mesh import AppGrid

        empty_mesh = AppGrid(nodes={}, elements={}, subregions={})
        gw = MagicMock()
        result = apply_parametric_grids(gw, [], empty_mesh)

        assert result is False
        gw.set_aquifer_parameters.assert_not_called()

    def test_set_aquifer_parameters_value_error_fallback(self) -> None:
        """When set_aquifer_parameters raises ValueError, falls back to direct assignment."""
        mesh = make_simple_grid()

        grid_data = MagicMock()
        grid_data.n_nodes = 1
        grid_data.n_elements = 0
        grid_data.node_values = np.array([[[1.0, 1.0, 1.0, 1.0, 1.0]]])

        gw = MagicMock()
        gw.set_aquifer_parameters.side_effect = ValueError("n_nodes mismatch")

        result = apply_parametric_grids(gw, [grid_data], mesh)

        assert result is True
        # Should have fallen back to direct attribute assignment
        assert gw.aquifer_params is not None


# ---------------------------------------------------------------------------
# apply_parametric_subsidence
# ---------------------------------------------------------------------------


class TestApplyParametricSubsidence:
    """Tests for apply_parametric_subsidence."""

    def test_uniform_single_node_subsidence(self) -> None:
        """Single-node uniform grid populates all nodes with same values."""
        mesh = make_simple_grid()
        n_nodes = len(mesh.nodes)
        n_layers = 2

        grid_data = MagicMock()
        grid_data.n_nodes = 1
        grid_data.n_elements = 0
        # shape (1, n_layers, 5): elastic_sc, inelastic_sc, interbed_thick,
        # interbed_thick_min, precompact_head
        grid_data.node_values = np.array([
            [[0.1, 0.5, 10.0, 1.0, 50.0],
             [0.2, 0.6, 12.0, 1.5, 55.0]],
        ])

        subs_config = MagicMock()
        subs_config.parametric_grids = [grid_data]

        node_params = apply_parametric_subsidence(subs_config, mesh, n_nodes, n_layers)

        assert len(node_params) == n_nodes
        # Each node should have the uniform values
        for np_ in node_params:
            assert np_.elastic_sc == [0.1, 0.2]
            assert np_.inelastic_sc == [0.5, 0.6]
            assert np_.interbed_thick == [10.0, 12.0]
            assert np_.interbed_thick_min == [1.0, 1.5]
            assert np_.precompact_head == [50.0, 55.0]

    def test_node_ids_match_mesh(self) -> None:
        """Returned SubsidenceNodeParams have node IDs matching sorted mesh nodes."""
        mesh = make_simple_grid()
        n_nodes = len(mesh.nodes)
        n_layers = 1

        grid_data = MagicMock()
        grid_data.n_nodes = 1
        grid_data.n_elements = 0
        grid_data.node_values = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]])

        subs_config = MagicMock()
        subs_config.parametric_grids = [grid_data]

        node_params = apply_parametric_subsidence(subs_config, mesh, n_nodes, n_layers)

        returned_ids = [p.node_id for p in node_params]
        assert returned_ids == sorted(mesh.nodes.keys())


# ---------------------------------------------------------------------------
# binary_data_to_model
# ---------------------------------------------------------------------------


class TestBinaryDataToModel:
    """Tests for binary_data_to_model."""

    def _make_binary_data(self) -> MagicMock:
        """Create a minimal PreprocessorBinaryData mock with 4 nodes, 1 element."""
        data = MagicMock()
        data.n_nodes = 4
        data.n_elements = 1
        data.x = np.array([0.0, 100.0, 100.0, 0.0])
        data.y = np.array([0.0, 0.0, 100.0, 100.0])
        data.n_vertex = np.array([4], dtype=np.int32)
        # Fortran column-major: vertex array flattened as (max_nv, n_elements)
        data.vertex = np.array([1, 2, 3, 4], dtype=np.int32)

        # AppElementData with subregion
        elem_data = MagicMock()
        elem_data.subregion = 1
        data.app_elements = [elem_data]

        # Subregions
        sub = MagicMock()
        sub.id = 1
        sub.name = "Sub1"
        data.subregions = [sub]

        # No stratigraphy, streams, or lakes
        data.stratigraphy = None
        data.streams = None
        data.stream_gw_connector = None
        data.lakes = None

        return data

    def test_basic_model_creation(self) -> None:
        """binary_data_to_model returns IWFMModel with correct grid dimensions."""
        data = self._make_binary_data()
        model = binary_data_to_model(data, name="test-model")

        assert model.name == "test-model"
        assert model.mesh is not None
        assert len(model.mesh.nodes) == 4
        assert len(model.mesh.elements) == 1
        assert model.mesh.elements[1].vertices == (1, 2, 3, 4)

    def test_no_stratigraphy(self) -> None:
        """When stratigraphy is None, model.stratigraphy stays None."""
        data = self._make_binary_data()
        data.stratigraphy = None
        model = binary_data_to_model(data)
        assert model.stratigraphy is None

    def test_with_stratigraphy(self) -> None:
        """Stratigraphy data is converted into a Stratigraphy object."""
        data = self._make_binary_data()
        strat_data = MagicMock()
        strat_data.n_layers = 2
        strat_data.ground_surface_elev = np.array([100.0, 100.0, 100.0, 100.0])
        strat_data.top_elev = np.zeros((4, 2))
        strat_data.bottom_elev = np.zeros((4, 2))
        strat_data.active_node = np.ones((4, 2), dtype=bool)
        data.stratigraphy = strat_data

        model = binary_data_to_model(data)

        assert model.stratigraphy is not None
        assert model.stratigraphy.n_layers == 2
        assert model.stratigraphy.n_nodes == 4

    def test_metadata_source(self) -> None:
        """Model metadata records the source as preprocessor_binary."""
        data = self._make_binary_data()
        model = binary_data_to_model(data)
        assert model.metadata["source"] == "preprocessor_binary"
