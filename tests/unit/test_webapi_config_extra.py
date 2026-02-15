"""Additional tests for ModelState covering uncovered branches.

Targets: get_stream_reach_boundaries, get_diversion_timeseries,
get_subsidence_reader, get_area_manager, reproject_coords,
_get_transformer success path, get_head_loader text/unknown branches,
get_node_id_to_idx, get_sorted_elem_ids, ViewerSettings validation,
subsidence hydrograph locations, set_model with simulation_file.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

pydantic = pytest.importorskip("pydantic")

from pyiwfm.core.mesh import AppGrid, Node, Element
from pyiwfm.visualization.webapi.config import ModelState, ViewerSettings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_grid():
    nodes = {
        1: Node(id=1, x=0.0, y=0.0),
        2: Node(id=2, x=100.0, y=0.0),
        3: Node(id=3, x=100.0, y=100.0),
        4: Node(id=4, x=0.0, y=100.0),
    }
    elements = {1: Element(id=1, vertices=(1, 2, 3, 4), subregion=1)}
    grid = AppGrid(nodes=nodes, elements=elements)
    grid.compute_connectivity()
    grid.compute_areas()
    return grid


def _make_mock_model(
    with_stratigraphy=True,
    with_streams=False,
    with_groundwater=False,
    simulation_file=None,
):
    model = MagicMock()
    model.name = "TestModel"
    model.grid = _make_grid()
    model.metadata = {}
    model.source_files = {}
    model.has_streams = with_streams
    model.has_lakes = False
    if simulation_file:
        model.metadata["simulation_file"] = simulation_file
    if with_stratigraphy:
        strat = MagicMock()
        strat.n_layers = 2
        strat.gs_elev = np.array([10.0, 20.0, 30.0, 40.0])
        strat.top_elev = np.array(
            [[10.0, 5.0], [20.0, 10.0], [30.0, 15.0], [40.0, 20.0]]
        )
        strat.bottom_elev = np.array(
            [[5.0, 0.0], [10.0, 5.0], [15.0, 10.0], [20.0, 15.0]]
        )
        model.stratigraphy = strat
    else:
        model.stratigraphy = None
    if with_groundwater:
        gw = MagicMock()
        gw.aquifer_params = MagicMock()
        gw.n_hydrograph_locations = 3
        gw.hydrograph_locations = []
        gw.subsidence_config = None
        model.groundwater = gw
    else:
        model.groundwater = None
    if with_streams:
        streams = MagicMock()
        streams.n_nodes = 5
        streams.nodes = {}
        streams.reaches = []
        model.streams = streams
    else:
        model.streams = None
    model.lakes = None
    model.rootzone = None
    model.n_nodes = 4
    model.n_elements = 1
    model.n_layers = 2 if with_stratigraphy else 0
    return model


def _state_with_model(**kwargs):
    """Create a ModelState with a mock model already loaded."""
    state = ModelState()
    model = _make_mock_model(**kwargs)
    state.set_model(model)
    return state, model


# ===========================================================================
# ViewerSettings
# ===========================================================================


class TestViewerSettings:
    """Tests for ViewerSettings Pydantic model."""

    def test_defaults(self):
        s = ViewerSettings()
        assert s.host == "127.0.0.1"
        assert s.port == 8080
        assert s.title == "IWFM Viewer"
        assert s.open_browser is True
        assert s.debug is False
        assert s.reload is False

    def test_custom_values(self):
        s = ViewerSettings(
            host="0.0.0.0", port=9090, title="My Viewer",
            open_browser=False, debug=True, reload=True,
        )
        assert s.host == "0.0.0.0"
        assert s.port == 9090
        assert s.title == "My Viewer"
        assert s.open_browser is False
        assert s.debug is True
        assert s.reload is True

    def test_port_validation_min(self):
        with pytest.raises(Exception):
            ViewerSettings(port=0)

    def test_port_validation_max(self):
        with pytest.raises(Exception):
            ViewerSettings(port=70000)

    def test_extra_fields_forbidden(self):
        with pytest.raises(Exception):
            ViewerSettings(unknown_field="bad")


# ===========================================================================
# set_model with simulation_file
# ===========================================================================


class TestSetModel:
    """Tests for set_model branches."""

    def test_set_model_with_simulation_file(self, tmp_path):
        """When model metadata has simulation_file, _results_dir is set."""
        state = ModelState()
        sim_file = tmp_path / "Simulation" / "sim.in"
        sim_file.parent.mkdir(parents=True, exist_ok=True)
        sim_file.touch()
        model = _make_mock_model(simulation_file=str(sim_file))
        state.set_model(model)
        assert state._results_dir == sim_file.parent

    def test_set_model_without_simulation_file(self):
        """When no simulation_file, _results_dir is None."""
        state = ModelState()
        model = _make_mock_model()
        state.set_model(model)
        assert state._results_dir is None

    def test_set_model_clears_caches(self):
        """set_model resets all internal caches."""
        state = ModelState()
        # Populate some caches
        state._mesh_3d = b"old"
        state._mesh_surface = b"old"
        state._bounds = (0, 1, 0, 1, 0, 1)
        state._pv_mesh_3d = object()
        state._layer_surface_cache = {1: {"data": True}}
        state._geojson_cache = {1: {"data": True}}
        state._head_loader = MagicMock()
        state._gw_hydrograph_reader = MagicMock()
        state._stream_hydrograph_reader = MagicMock()
        state._subsidence_reader = MagicMock()
        state._budget_readers = {"gw": MagicMock()}
        state._area_manager = MagicMock()
        state._observations = {"o1": {"data": True}}
        state._stream_reach_boundaries = [(1, 1, 5)]
        state._diversion_ts_data = ("times", "values", "meta")
        state._node_id_to_idx = {1: 0}
        state._sorted_elem_ids = [1]

        model = _make_mock_model()
        state.set_model(model)

        assert state._mesh_3d is None
        assert state._mesh_surface is None
        assert state._bounds is None
        assert state._pv_mesh_3d is None
        assert state._layer_surface_cache == {}
        assert state._geojson_cache == {}
        assert state._head_loader is None
        assert state._gw_hydrograph_reader is None
        assert state._stream_hydrograph_reader is None
        assert state._subsidence_reader is None
        assert state._budget_readers == {}
        assert state._area_manager is None
        assert state._observations == {}
        assert state._stream_reach_boundaries is None
        assert state._diversion_ts_data is None
        assert state._node_id_to_idx is None
        assert state._sorted_elem_ids is None

    def test_set_model_with_crs(self):
        """Custom CRS is stored."""
        state = ModelState()
        model = _make_mock_model()
        state.set_model(model, crs="EPSG:2227")
        assert state._crs == "EPSG:2227"


# ===========================================================================
# _get_transformer and reproject_coords
# ===========================================================================


class TestTransformerAndReproject:
    """Tests for _get_transformer and reproject_coords."""

    def test_get_transformer_success(self):
        """When pyproj is available, creates and caches a Transformer."""
        state = ModelState()
        mock_transformer = MagicMock()
        mock_Transformer = MagicMock()
        mock_Transformer.from_crs.return_value = mock_transformer

        with patch.dict("sys.modules", {"pyproj": MagicMock(Transformer=mock_Transformer)}):
            result = state._get_transformer()

        assert result is mock_transformer
        # Second call should return cached
        result2 = state._get_transformer()
        assert result2 is mock_transformer

    def test_reproject_coords_with_transformer(self):
        """When transformer works, returns transformed coords."""
        state = ModelState()
        mock_transformer = MagicMock()
        mock_transformer.transform.return_value = (-121.5, 38.5)
        state._transformer = mock_transformer

        lng, lat = state.reproject_coords(1000.0, 2000.0)
        assert lng == -121.5
        assert lat == 38.5
        mock_transformer.transform.assert_called_once_with(1000.0, 2000.0)

    def test_reproject_coords_without_transformer(self):
        """When transformer is None (pyproj missing), returns input unchanged."""
        state = ModelState()
        state._transformer = None
        with patch.object(state, "_get_transformer", return_value=None):
            lng, lat = state.reproject_coords(1000.0, 2000.0)

        assert lng == 1000.0
        assert lat == 2000.0


# ===========================================================================
# get_stream_reach_boundaries
# ===========================================================================


class TestStreamReachBoundaries:
    """Tests for get_stream_reach_boundaries."""

    def test_returns_cached(self):
        """When already cached, returns immediately."""
        state, _ = _state_with_model()
        cached = [(1, 1, 5), (2, 6, 10)]
        state._stream_reach_boundaries = cached
        result = state.get_stream_reach_boundaries()
        assert result is cached

    def test_returns_none_no_model(self):
        state = ModelState()
        result = state.get_stream_reach_boundaries()
        assert result is None

    def test_strategy1_binary_preprocessor(self, tmp_path):
        """Strategy 1: reads from preprocessor binary."""
        state, model = _state_with_model()
        binary_file = tmp_path / "preprocessor.bin"
        binary_file.write_bytes(b"fake")
        model.source_files = {"binary_preprocessor": str(binary_file)}

        mock_data = MagicMock()
        mock_data.streams.n_reaches = 2
        mock_data.streams.reach_ids = [1, 2]
        mock_data.streams.reach_upstream_nodes = [10, 20]
        mock_data.streams.reach_downstream_nodes = [15, 25]

        mock_reader_cls = MagicMock()
        mock_reader_cls.return_value.read.return_value = mock_data

        with patch(
            "pyiwfm.io.preprocessor_binary.PreprocessorBinaryReader",
            mock_reader_cls,
        ):
            result = state.get_stream_reach_boundaries()

        assert result == [(1, 10, 15), (2, 20, 25)]

    def test_strategy1_binary_exception_falls_through(self, tmp_path):
        """When binary read raises, falls through to strategy 2."""
        state, model = _state_with_model()
        binary_file = tmp_path / "preprocessor.bin"
        binary_file.write_bytes(b"fake")
        model.source_files = {"binary_preprocessor": str(binary_file)}

        mock_reader_cls = MagicMock()
        mock_reader_cls.return_value.read.side_effect = RuntimeError("corrupt")

        with patch(
            "pyiwfm.io.preprocessor_binary.PreprocessorBinaryReader",
            mock_reader_cls,
        ):
            result = state.get_stream_reach_boundaries()

        assert result is None

    def test_strategy1_binary_no_streams_data(self, tmp_path):
        """When binary has no streams data, falls through."""
        state, model = _state_with_model()
        binary_file = tmp_path / "preprocessor.bin"
        binary_file.write_bytes(b"fake")
        model.source_files = {"binary_preprocessor": str(binary_file)}

        mock_data = MagicMock()
        mock_data.streams = None

        mock_reader_cls = MagicMock()
        mock_reader_cls.return_value.read.return_value = mock_data

        with patch(
            "pyiwfm.io.preprocessor_binary.PreprocessorBinaryReader",
            mock_reader_cls,
        ):
            result = state.get_stream_reach_boundaries()

        assert result is None

    def test_strategy1_binary_zero_reaches(self, tmp_path):
        """When binary has streams but 0 reaches, falls through."""
        state, model = _state_with_model()
        binary_file = tmp_path / "preprocessor.bin"
        binary_file.write_bytes(b"fake")
        model.source_files = {"binary_preprocessor": str(binary_file)}

        mock_data = MagicMock()
        mock_data.streams.n_reaches = 0

        mock_reader_cls = MagicMock()
        mock_reader_cls.return_value.read.return_value = mock_data

        with patch(
            "pyiwfm.io.preprocessor_binary.PreprocessorBinaryReader",
            mock_reader_cls,
        ):
            result = state.get_stream_reach_boundaries()

        assert result is None

    def test_strategy2_streams_spec(self, tmp_path):
        """Strategy 2: reads from streams spec text file."""
        state, model = _state_with_model()
        spec_file = tmp_path / "streams.dat"
        spec_file.write_text("fake")
        model.source_files = {"streams_spec": str(spec_file)}

        rs1 = MagicMock()
        rs1.id = 1
        rs1.node_ids = [10, 11, 12, 15]
        rs2 = MagicMock()
        rs2.id = 2
        rs2.node_ids = [20, 21, 25]

        mock_reader_cls = MagicMock()
        mock_reader_cls.return_value.read.return_value = (2, 7, [rs1, rs2])

        with patch(
            "pyiwfm.io.streams.StreamSpecReader",
            mock_reader_cls,
        ):
            result = state.get_stream_reach_boundaries()

        assert result == [(1, 10, 15), (2, 20, 25)]

    def test_strategy2_streams_spec_empty_node_ids(self, tmp_path):
        """Strategy 2: reach with empty node_ids is skipped."""
        state, model = _state_with_model()
        spec_file = tmp_path / "streams.dat"
        spec_file.write_text("fake")
        model.source_files = {"streams_spec": str(spec_file)}

        rs1 = MagicMock()
        rs1.id = 1
        rs1.node_ids = []  # Empty
        rs2 = MagicMock()
        rs2.id = 2
        rs2.node_ids = [20, 25]

        mock_reader_cls = MagicMock()
        mock_reader_cls.return_value.read.return_value = (2, 3, [rs1, rs2])

        with patch(
            "pyiwfm.io.streams.StreamSpecReader",
            mock_reader_cls,
        ):
            result = state.get_stream_reach_boundaries()

        assert result == [(2, 20, 25)]

    def test_strategy2_streams_spec_exception(self, tmp_path):
        """Strategy 2: when read raises, falls through to strategy 3."""
        state, model = _state_with_model()
        spec_file = tmp_path / "streams.dat"
        spec_file.write_text("fake")
        model.source_files = {"streams_spec": str(spec_file)}

        mock_reader_cls = MagicMock()
        mock_reader_cls.return_value.read.side_effect = RuntimeError("bad format")

        with patch(
            "pyiwfm.io.streams.StreamSpecReader",
            mock_reader_cls,
        ):
            result = state.get_stream_reach_boundaries()

        assert result is None

    def test_strategy2_streams_spec_no_reaches(self, tmp_path):
        """Strategy 2: empty reach_specs falls through."""
        state, model = _state_with_model()
        spec_file = tmp_path / "streams.dat"
        spec_file.write_text("fake")
        model.source_files = {"streams_spec": str(spec_file)}

        mock_reader_cls = MagicMock()
        mock_reader_cls.return_value.read.return_value = (0, 0, [])

        with patch(
            "pyiwfm.io.streams.StreamSpecReader",
            mock_reader_cls,
        ):
            result = state.get_stream_reach_boundaries()

        assert result is None

    def test_strategy2_all_empty_node_ids_falls_through(self, tmp_path):
        """Strategy 2: all reaches with empty node_ids -> empty boundaries -> falls through."""
        state, model = _state_with_model()
        spec_file = tmp_path / "streams.dat"
        spec_file.write_text("fake")
        model.source_files = {"streams_spec": str(spec_file)}

        rs1 = MagicMock()
        rs1.id = 1
        rs1.node_ids = []

        mock_reader_cls = MagicMock()
        mock_reader_cls.return_value.read.return_value = (1, 0, [rs1])

        with patch(
            "pyiwfm.io.streams.StreamSpecReader",
            mock_reader_cls,
        ):
            result = state.get_stream_reach_boundaries()

        assert result is None

    def test_strategy3_preprocessor_main(self, tmp_path):
        """Strategy 3: reads preprocessor main to find streams spec."""
        state, model = _state_with_model()
        pp_file = tmp_path / "PreProcessor_MAIN.IN"
        pp_file.write_text("fake")
        model.source_files = {"preprocessor_main": str(pp_file)}

        streams_file = tmp_path / "streams.dat"
        streams_file.write_text("fake")

        mock_pp_config = MagicMock()
        mock_pp_config.streams_file = streams_file

        rs1 = MagicMock()
        rs1.id = 1
        rs1.node_ids = [1, 2, 3]

        mock_spec_reader_cls = MagicMock()
        mock_spec_reader_cls.return_value.read.return_value = (1, 3, [rs1])

        with patch(
            "pyiwfm.io.preprocessor.read_preprocessor_main",
            return_value=mock_pp_config,
        ), patch(
            "pyiwfm.io.streams.StreamSpecReader",
            mock_spec_reader_cls,
        ):
            result = state.get_stream_reach_boundaries()

        assert result == [(1, 1, 3)]

    def test_strategy3_preprocessor_main_exception(self, tmp_path):
        """Strategy 3: when preprocessor read raises, returns None."""
        state, model = _state_with_model()
        pp_file = tmp_path / "PreProcessor_MAIN.IN"
        pp_file.write_text("fake")
        model.source_files = {"preprocessor_main": str(pp_file)}

        with patch(
            "pyiwfm.io.preprocessor.read_preprocessor_main",
            side_effect=RuntimeError("bad preprocessor"),
        ):
            result = state.get_stream_reach_boundaries()

        assert result is None

    def test_strategy3_preprocessor_main_no_streams_file(self, tmp_path):
        """Strategy 3: pp_config has no streams_file."""
        state, model = _state_with_model()
        pp_file = tmp_path / "PreProcessor_MAIN.IN"
        pp_file.write_text("fake")
        model.source_files = {"preprocessor_main": str(pp_file)}

        mock_pp_config = MagicMock()
        mock_pp_config.streams_file = None

        with patch(
            "pyiwfm.io.preprocessor.read_preprocessor_main",
            return_value=mock_pp_config,
        ):
            result = state.get_stream_reach_boundaries()

        assert result is None

    def test_strategy3_empty_node_ids(self, tmp_path):
        """Strategy 3: all reach specs have empty node_ids -> returns None."""
        state, model = _state_with_model()
        pp_file = tmp_path / "PreProcessor_MAIN.IN"
        pp_file.write_text("fake")
        model.source_files = {"preprocessor_main": str(pp_file)}

        streams_file = tmp_path / "streams.dat"
        streams_file.write_text("fake")

        mock_pp_config = MagicMock()
        mock_pp_config.streams_file = streams_file

        rs1 = MagicMock()
        rs1.id = 1
        rs1.node_ids = []

        mock_spec_reader_cls = MagicMock()
        mock_spec_reader_cls.return_value.read.return_value = (1, 0, [rs1])

        with patch(
            "pyiwfm.io.preprocessor.read_preprocessor_main",
            return_value=mock_pp_config,
        ), patch(
            "pyiwfm.io.streams.StreamSpecReader",
            mock_spec_reader_cls,
        ):
            result = state.get_stream_reach_boundaries()

        assert result is None

    def test_strategy3_empty_reach_specs(self, tmp_path):
        """Strategy 3: reach_specs is empty list -> falls through."""
        state, model = _state_with_model()
        pp_file = tmp_path / "PreProcessor_MAIN.IN"
        pp_file.write_text("fake")
        model.source_files = {"preprocessor_main": str(pp_file)}

        streams_file = tmp_path / "streams.dat"
        streams_file.write_text("fake")

        mock_pp_config = MagicMock()
        mock_pp_config.streams_file = streams_file

        mock_spec_reader_cls = MagicMock()
        mock_spec_reader_cls.return_value.read.return_value = (0, 0, [])

        with patch(
            "pyiwfm.io.preprocessor.read_preprocessor_main",
            return_value=mock_pp_config,
        ), patch(
            "pyiwfm.io.streams.StreamSpecReader",
            mock_spec_reader_cls,
        ):
            result = state.get_stream_reach_boundaries()

        assert result is None

    def test_no_source_files_at_all(self):
        """When model has no source_files at all, returns None."""
        state, model = _state_with_model()
        model.source_files = {}
        result = state.get_stream_reach_boundaries()
        assert result is None

    def test_source_files_is_none(self):
        """When source_files attr is None, returns None."""
        state, model = _state_with_model()
        model.source_files = None
        result = state.get_stream_reach_boundaries()
        assert result is None

    def test_binary_path_does_not_exist(self, tmp_path):
        """Binary path in source_files but file doesn't exist, falls through."""
        state, model = _state_with_model()
        model.source_files = {
            "binary_preprocessor": str(tmp_path / "nonexistent.bin")
        }
        result = state.get_stream_reach_boundaries()
        assert result is None


# ===========================================================================
# get_diversion_timeseries
# ===========================================================================


class TestDiversionTimeseries:
    """Tests for get_diversion_timeseries."""

    def test_returns_cached(self):
        state, _ = _state_with_model()
        cached = ("times", "values", "meta")
        state._diversion_ts_data = cached
        result = state.get_diversion_timeseries()
        assert result is cached

    def test_returns_none_no_model(self):
        state = ModelState()
        result = state.get_diversion_timeseries()
        assert result is None

    def test_returns_none_no_source_files_key(self):
        state, model = _state_with_model()
        model.source_files = {}
        result = state.get_diversion_timeseries()
        assert result is None

    def test_returns_none_empty_path(self):
        state, model = _state_with_model()
        model.source_files = {"stream_diversion_ts": ""}
        result = state.get_diversion_timeseries()
        assert result is None

    def test_returns_none_file_not_found(self, tmp_path):
        state, model = _state_with_model()
        model.source_files = {
            "stream_diversion_ts": str(tmp_path / "nonexistent.dat")
        }
        result = state.get_diversion_timeseries()
        assert result is None

    def test_reads_file_absolute_path(self, tmp_path):
        state, model = _state_with_model()
        ts_file = tmp_path / "diversions.dat"
        ts_file.write_text("fake")
        model.source_files = {"stream_diversion_ts": str(ts_file)}

        mock_times = [1, 2, 3]
        mock_values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        mock_metadata = {"columns": ["div1", "div2"]}

        mock_reader_cls = MagicMock()
        mock_reader_cls.return_value.read_file.return_value = (
            mock_times, mock_values, mock_metadata,
        )

        with patch(
            "pyiwfm.io.timeseries.UnifiedTimeSeriesReader",
            mock_reader_cls,
        ):
            result = state.get_diversion_timeseries()

        assert result is not None
        assert result[0] is mock_times
        assert result[1] is mock_values
        assert result[2] is mock_metadata

    def test_reads_file_relative_path(self, tmp_path):
        state, model = _state_with_model()
        state._results_dir = tmp_path
        ts_file = tmp_path / "diversions.dat"
        ts_file.write_text("fake")
        model.source_files = {"stream_diversion_ts": "diversions.dat"}

        mock_times = [1]
        mock_values = np.array([[1.0]])
        mock_metadata = {}

        mock_reader_cls = MagicMock()
        mock_reader_cls.return_value.read_file.return_value = (
            mock_times, mock_values, mock_metadata,
        )

        with patch(
            "pyiwfm.io.timeseries.UnifiedTimeSeriesReader",
            mock_reader_cls,
        ):
            result = state.get_diversion_timeseries()

        assert result is not None

    def test_returns_none_on_exception(self, tmp_path):
        state, model = _state_with_model()
        ts_file = tmp_path / "diversions.dat"
        ts_file.write_text("fake")
        model.source_files = {"stream_diversion_ts": str(ts_file)}

        mock_reader_cls = MagicMock()
        mock_reader_cls.return_value.read_file.side_effect = RuntimeError("parse error")

        with patch(
            "pyiwfm.io.timeseries.UnifiedTimeSeriesReader",
            mock_reader_cls,
        ):
            result = state.get_diversion_timeseries()

        assert result is None

    def test_caches_result(self, tmp_path):
        state, model = _state_with_model()
        ts_file = tmp_path / "diversions.dat"
        ts_file.write_text("fake")
        model.source_files = {"stream_diversion_ts": str(ts_file)}

        mock_times = [1]
        mock_values = np.array([[1.0]])
        mock_metadata = {}

        mock_reader_cls = MagicMock()
        mock_reader_cls.return_value.read_file.return_value = (
            mock_times, mock_values, mock_metadata,
        )

        with patch(
            "pyiwfm.io.timeseries.UnifiedTimeSeriesReader",
            mock_reader_cls,
        ):
            first = state.get_diversion_timeseries()
            second = state.get_diversion_timeseries()

        assert first is second

    def test_1d_values(self, tmp_path):
        """When values is 1D, logger logs '1 columns'."""
        state, model = _state_with_model()
        ts_file = tmp_path / "diversions.dat"
        ts_file.write_text("fake")
        model.source_files = {"stream_diversion_ts": str(ts_file)}

        mock_times = [1, 2]
        mock_values = np.array([1.0, 2.0])  # 1D
        mock_metadata = {}

        mock_reader_cls = MagicMock()
        mock_reader_cls.return_value.read_file.return_value = (
            mock_times, mock_values, mock_metadata,
        )

        with patch(
            "pyiwfm.io.timeseries.UnifiedTimeSeriesReader",
            mock_reader_cls,
        ):
            result = state.get_diversion_timeseries()

        assert result is not None

    def test_source_files_attr_none(self):
        """When source_files is None, uses empty dict."""
        state, model = _state_with_model()
        model.source_files = None
        result = state.get_diversion_timeseries()
        assert result is None


# ===========================================================================
# get_subsidence_reader
# ===========================================================================


class TestSubsidenceReader:
    """Tests for get_subsidence_reader."""

    def test_returns_cached(self):
        state, model = _state_with_model(with_groundwater=True)
        cached = MagicMock()
        state._subsidence_reader = cached
        result = state.get_subsidence_reader()
        assert result is cached

    def test_returns_none_no_model(self):
        state = ModelState()
        result = state.get_subsidence_reader()
        assert result is None

    def test_from_subsidence_config(self, tmp_path):
        """Reads from subsidence_config.hydrograph_output_file."""
        state, model = _state_with_model(with_groundwater=True)
        subs_file = tmp_path / "subsidence.out"
        subs_file.write_text("fake data")

        subs_config = MagicMock()
        subs_config.hydrograph_output_file = str(subs_file)
        model.groundwater.subsidence_config = subs_config

        mock_reader = MagicMock()
        mock_reader.n_columns = 5
        mock_reader.n_timesteps = 20

        with patch(
            "pyiwfm.visualization.webapi.hydrograph_reader.IWFMHydrographReader",
            return_value=mock_reader,
        ):
            result = state.get_subsidence_reader()

        assert result is mock_reader

    def test_from_subsidence_config_relative_path(self, tmp_path):
        """Relative path resolved via _results_dir."""
        state, model = _state_with_model(with_groundwater=True)
        state._results_dir = tmp_path
        subs_file = tmp_path / "subsidence.out"
        subs_file.write_text("fake data")

        subs_config = MagicMock()
        subs_config.hydrograph_output_file = "subsidence.out"
        model.groundwater.subsidence_config = subs_config

        mock_reader = MagicMock()
        mock_reader.n_columns = 5
        mock_reader.n_timesteps = 20

        with patch(
            "pyiwfm.visualization.webapi.hydrograph_reader.IWFMHydrographReader",
            return_value=mock_reader,
        ):
            result = state.get_subsidence_reader()

        assert result is mock_reader

    def test_from_subsidence_config_exception(self, tmp_path):
        """Exception falls through to glob scanning."""
        state, model = _state_with_model(with_groundwater=True)
        subs_file = tmp_path / "subsidence.out"
        subs_file.write_text("fake data")

        subs_config = MagicMock()
        subs_config.hydrograph_output_file = str(subs_file)
        model.groundwater.subsidence_config = subs_config

        with patch(
            "pyiwfm.visualization.webapi.hydrograph_reader.IWFMHydrographReader",
            side_effect=RuntimeError("bad file"),
        ):
            result = state.get_subsidence_reader()

        assert result is None

    def test_from_subsidence_config_file_not_found(self, tmp_path):
        """When config output file doesn't exist, falls through to glob."""
        state, model = _state_with_model(with_groundwater=True)
        subs_config = MagicMock()
        subs_config.hydrograph_output_file = str(tmp_path / "nonexistent.out")
        model.groundwater.subsidence_config = subs_config

        result = state.get_subsidence_reader()
        assert result is None

    def test_from_subsidence_config_no_output_file(self):
        """When subsidence_config has no hydrograph_output_file."""
        state, model = _state_with_model(with_groundwater=True)
        subs_config = MagicMock()
        subs_config.hydrograph_output_file = None
        model.groundwater.subsidence_config = subs_config

        result = state.get_subsidence_reader()
        assert result is None

    def test_from_subsidence_config_empty_output_file(self):
        """When subsidence_config has empty string for hydrograph_output_file."""
        state, model = _state_with_model(with_groundwater=True)
        subs_config = MagicMock()
        subs_config.hydrograph_output_file = ""
        model.groundwater.subsidence_config = subs_config

        result = state.get_subsidence_reader()
        assert result is None

    def test_glob_scan_fallback(self, tmp_path):
        """When config fails, scans for *Subsidence*.out in results dir."""
        state, model = _state_with_model(with_groundwater=True)
        state._results_dir = tmp_path
        model.groundwater.subsidence_config = None

        subs_file = tmp_path / "GW_Subsidence.out"
        subs_file.write_text("fake data")

        mock_reader = MagicMock()
        mock_reader.n_columns = 3
        mock_reader.n_timesteps = 10

        with patch(
            "pyiwfm.visualization.webapi.hydrograph_reader.IWFMHydrographReader",
            return_value=mock_reader,
        ):
            result = state.get_subsidence_reader()

        assert result is mock_reader

    def test_glob_scan_exception(self, tmp_path):
        """When glob match file can't be read, continues to next pattern."""
        state, model = _state_with_model(with_groundwater=True)
        state._results_dir = tmp_path
        model.groundwater.subsidence_config = None

        subs_file = tmp_path / "GW_Subsidence.out"
        subs_file.write_text("fake data")

        with patch(
            "pyiwfm.visualization.webapi.hydrograph_reader.IWFMHydrographReader",
            side_effect=RuntimeError("corrupt"),
        ):
            result = state.get_subsidence_reader()

        assert result is None

    def test_no_groundwater(self):
        """When model has no groundwater, skips config check."""
        state, model = _state_with_model(with_groundwater=False)
        result = state.get_subsidence_reader()
        assert result is None

    def test_no_results_dir_and_no_config(self):
        """When no _results_dir and no subsidence_config, returns None."""
        state, model = _state_with_model(with_groundwater=True)
        model.groundwater.subsidence_config = None
        state._results_dir = None
        result = state.get_subsidence_reader()
        assert result is None


# ===========================================================================
# get_area_manager
# ===========================================================================


class TestAreaManager:
    """Tests for get_area_manager."""

    def test_returns_cached(self):
        state, _ = _state_with_model()
        cached = MagicMock()
        state._area_manager = cached
        result = state.get_area_manager()
        assert result is cached

    def test_returns_none_no_model(self):
        state = ModelState()
        result = state.get_area_manager()
        assert result is None

    def test_returns_none_no_rootzone(self):
        state, model = _state_with_model()
        model.rootzone = None
        result = state.get_area_manager()
        assert result is None

    def test_returns_none_no_area_files(self):
        """When rootzone exists but no area file attrs are set."""
        state, model = _state_with_model()
        rz = MagicMock()
        rz.nonponded_area_file = None
        rz.ponded_area_file = None
        rz.urban_area_file = None
        rz.native_area_file = None
        model.rootzone = rz
        result = state.get_area_manager()
        assert result is None

    def test_creates_manager(self, tmp_path):
        """When area files exist, creates AreaDataManager."""
        state, model = _state_with_model()
        state._results_dir = tmp_path
        rz = MagicMock()
        rz.nonponded_area_file = "nonponded.dat"
        rz.ponded_area_file = None
        rz.urban_area_file = None
        rz.native_area_file = None
        model.rootzone = rz

        mock_mgr = MagicMock()
        mock_mgr.n_timesteps = 12
        mock_mgr_cls = MagicMock(return_value=mock_mgr)

        with patch(
            "pyiwfm.visualization.webapi.area_loader.AreaDataManager",
            mock_mgr_cls,
        ):
            result = state.get_area_manager()

        assert result is mock_mgr
        mock_mgr.load_from_rootzone.assert_called_once_with(rz, tmp_path)

    def test_creates_manager_default_cache_dir(self):
        """When _results_dir is None, uses Path('.')."""
        state, model = _state_with_model()
        state._results_dir = None
        rz = MagicMock()
        rz.nonponded_area_file = "nonponded.dat"
        rz.ponded_area_file = None
        rz.urban_area_file = None
        rz.native_area_file = None
        model.rootzone = rz

        mock_mgr = MagicMock()
        mock_mgr.n_timesteps = 5
        mock_mgr_cls = MagicMock(return_value=mock_mgr)

        with patch(
            "pyiwfm.visualization.webapi.area_loader.AreaDataManager",
            mock_mgr_cls,
        ):
            result = state.get_area_manager()

        assert result is mock_mgr
        mock_mgr.load_from_rootzone.assert_called_once_with(rz, Path("."))

    def test_returns_none_on_exception(self):
        """When AreaDataManager raises, returns None."""
        state, model = _state_with_model()
        rz = MagicMock()
        rz.nonponded_area_file = "nonponded.dat"
        rz.ponded_area_file = None
        rz.urban_area_file = None
        rz.native_area_file = None
        model.rootzone = rz

        mock_mgr_cls = MagicMock(side_effect=RuntimeError("area fail"))

        with patch(
            "pyiwfm.visualization.webapi.area_loader.AreaDataManager",
            mock_mgr_cls,
        ):
            result = state.get_area_manager()

        assert result is None


# ===========================================================================
# get_head_loader: text and unknown extension branches
# ===========================================================================


class TestHeadLoaderBranches:
    """Tests for get_head_loader text conversion and unknown extension."""

    def test_text_file_conversion(self, tmp_path):
        """Text .out files are converted to HDF cache then loaded."""
        state, model = _state_with_model()
        head_file = tmp_path / "head.out"
        head_file.write_text("fake data")
        model.metadata["gw_head_all_file"] = str(head_file)
        model.metadata["n_gw_layers"] = 3

        mock_loader = MagicMock()
        mock_loader.n_frames = 10

        mock_convert = MagicMock()

        with patch(
            "pyiwfm.visualization.webapi.head_loader.LazyHeadDataLoader",
            return_value=mock_loader,
        ), patch(
            "pyiwfm.io.head_all_converter.convert_headall_to_hdf",
            mock_convert,
        ):
            result = state.get_head_loader()

        assert result is mock_loader
        mock_convert.assert_called_once()
        # Check n_layers=3 was passed
        call_kwargs = mock_convert.call_args
        assert call_kwargs[1]["n_layers"] == 3 or call_kwargs[0][2] == 3 if len(call_kwargs[0]) > 2 else call_kwargs[1].get("n_layers") == 3

    def test_text_file_with_existing_cache(self, tmp_path):
        """When HDF cache exists and is newer, skips conversion."""
        state, model = _state_with_model()
        head_file = tmp_path / "head.txt"
        head_file.write_text("fake data")

        import time
        time.sleep(0.05)

        hdf_cache = head_file.with_suffix(".head_cache.hdf")
        hdf_cache.write_bytes(b"cached")

        model.metadata["gw_head_all_file"] = str(head_file)

        mock_loader = MagicMock()
        mock_loader.n_frames = 5

        with patch(
            "pyiwfm.visualization.webapi.head_loader.LazyHeadDataLoader",
            return_value=mock_loader,
        ):
            result = state.get_head_loader()

        assert result is mock_loader

    def test_text_file_with_stale_cache(self, tmp_path):
        """When HDF cache is older than source, re-converts."""
        state, model = _state_with_model()

        hdf_cache = tmp_path / "head.head_cache.hdf"
        hdf_cache.write_bytes(b"old cached")

        import time
        time.sleep(0.05)

        head_file = tmp_path / "head.dat"
        head_file.write_text("new data")

        model.metadata["gw_head_all_file"] = str(head_file)

        mock_loader = MagicMock()
        mock_loader.n_frames = 10
        mock_convert = MagicMock()

        with patch(
            "pyiwfm.visualization.webapi.head_loader.LazyHeadDataLoader",
            return_value=mock_loader,
        ), patch(
            "pyiwfm.io.head_all_converter.convert_headall_to_hdf",
            mock_convert,
        ):
            result = state.get_head_loader()

        assert result is mock_loader
        mock_convert.assert_called_once()

    def test_unknown_extension(self, tmp_path):
        """Unknown extension tries direct HDF load."""
        state, model = _state_with_model()
        head_file = tmp_path / "head.xyz"
        head_file.write_bytes(b"unknown format")
        model.metadata["gw_head_all_file"] = str(head_file)

        mock_loader = MagicMock()
        mock_loader.n_frames = 2

        with patch(
            "pyiwfm.visualization.webapi.head_loader.LazyHeadDataLoader",
            return_value=mock_loader,
        ):
            result = state.get_head_loader()

        assert result is mock_loader

    def test_text_file_no_n_gw_layers_defaults_to_1(self, tmp_path):
        """When n_gw_layers not in metadata, defaults to 1."""
        state, model = _state_with_model()
        head_file = tmp_path / "head.out"
        head_file.write_text("fake data")
        model.metadata["gw_head_all_file"] = str(head_file)
        # No n_gw_layers in metadata

        mock_loader = MagicMock()
        mock_loader.n_frames = 10
        mock_convert = MagicMock()

        with patch(
            "pyiwfm.visualization.webapi.head_loader.LazyHeadDataLoader",
            return_value=mock_loader,
        ), patch(
            "pyiwfm.io.head_all_converter.convert_headall_to_hdf",
            mock_convert,
        ):
            result = state.get_head_loader()

        assert result is mock_loader
        # n_layers should default to 1
        call_kwargs = mock_convert.call_args
        if call_kwargs[1]:
            assert call_kwargs[1].get("n_layers") == 1
        else:
            assert call_kwargs[0][2] == 1


# ===========================================================================
# get_node_id_to_idx and get_sorted_elem_ids
# ===========================================================================


class TestGridIndexMappings:
    """Tests for get_node_id_to_idx and get_sorted_elem_ids."""

    def test_node_id_to_idx(self):
        state, _ = _state_with_model()
        mapping = state.get_node_id_to_idx()
        assert mapping == {1: 0, 2: 1, 3: 2, 4: 3}

    def test_node_id_to_idx_caches(self):
        state, _ = _state_with_model()
        first = state.get_node_id_to_idx()
        second = state.get_node_id_to_idx()
        assert first is second

    def test_node_id_to_idx_no_model(self):
        state = ModelState()
        result = state.get_node_id_to_idx()
        assert result == {}

    def test_node_id_to_idx_no_grid(self):
        state = ModelState()
        model = MagicMock()
        model.grid = None
        state._model = model
        result = state.get_node_id_to_idx()
        assert result == {}

    def test_sorted_elem_ids(self):
        state, _ = _state_with_model()
        result = state.get_sorted_elem_ids()
        assert result == [1]

    def test_sorted_elem_ids_caches(self):
        state, _ = _state_with_model()
        first = state.get_sorted_elem_ids()
        second = state.get_sorted_elem_ids()
        assert first is second

    def test_sorted_elem_ids_no_model(self):
        state = ModelState()
        result = state.get_sorted_elem_ids()
        assert result == []

    def test_sorted_elem_ids_no_grid(self):
        state = ModelState()
        model = MagicMock()
        model.grid = None
        state._model = model
        result = state.get_sorted_elem_ids()
        assert result == []


# ===========================================================================
# Subsidence hydrograph locations
# ===========================================================================


class TestSubsidenceHydrographLocations:
    """Tests for subsidence locations in get_hydrograph_locations."""

    def test_subsidence_locations_from_config(self):
        state, model = _state_with_model(with_groundwater=True)

        spec1 = MagicMock()
        spec1.id = 1
        spec1.x = 500.0
        spec1.y = 600.0
        spec1.name = "Subs Obs 1"
        spec1.layer = 1
        spec1.node_id = 0
        spec1.gw_node = 0

        spec2 = MagicMock()
        spec2.id = 2
        spec2.x = 700.0
        spec2.y = 800.0
        spec2.name = None  # Default name
        spec2.layer = 2
        spec2.node_id = 0
        spec2.gw_node = 0

        subs_config = MagicMock()
        subs_config.hydrograph_specs = [spec1, spec2]
        model.groundwater.subsidence_config = subs_config

        with patch.object(state, "reproject_coords", side_effect=lambda x, y: (x, y)):
            result = state.get_hydrograph_locations()

        assert len(result["subsidence"]) == 2
        assert result["subsidence"][0]["id"] == 1
        assert result["subsidence"][0]["name"] == "Subs Obs 1"
        assert result["subsidence"][0]["lng"] == 500.0
        assert result["subsidence"][0]["lat"] == 600.0
        assert result["subsidence"][1]["name"] == "Subsidence Obs 2"

    def test_subsidence_locations_lookup_from_grid(self):
        """When spec has (0,0), looks up coords from grid node."""
        state, model = _state_with_model(with_groundwater=True)

        spec1 = MagicMock()
        spec1.id = 1
        spec1.x = 0.0
        spec1.y = 0.0
        spec1.name = "Subs Obs 1"
        spec1.layer = 1
        spec1.node_id = 2  # References grid node 2 at (100,0)
        spec1.gw_node = 0

        subs_config = MagicMock()
        subs_config.hydrograph_specs = [spec1]
        model.groundwater.subsidence_config = subs_config

        with patch.object(state, "reproject_coords", side_effect=lambda x, y: (x, y)):
            result = state.get_hydrograph_locations()

        assert len(result["subsidence"]) == 1
        assert result["subsidence"][0]["lng"] == 100.0
        assert result["subsidence"][0]["lat"] == 0.0

    def test_subsidence_locations_lookup_from_gw_node(self):
        """When spec has (0,0) and node_id=0, uses gw_node attr."""
        state, model = _state_with_model(with_groundwater=True)

        spec1 = MagicMock()
        spec1.id = 1
        spec1.x = 0.0
        spec1.y = 0.0
        spec1.name = "Subs Obs 1"
        spec1.layer = 1
        spec1.node_id = 0
        spec1.gw_node = 3  # References grid node 3 at (100,100)

        subs_config = MagicMock()
        subs_config.hydrograph_specs = [spec1]
        model.groundwater.subsidence_config = subs_config

        with patch.object(state, "reproject_coords", side_effect=lambda x, y: (x, y)):
            result = state.get_hydrograph_locations()

        assert len(result["subsidence"]) == 1
        assert result["subsidence"][0]["lng"] == 100.0
        assert result["subsidence"][0]["lat"] == 100.0

    def test_subsidence_locations_skip_zero_coords(self):
        """When spec has (0,0) and no valid node lookup, skip it."""
        state, model = _state_with_model(with_groundwater=True)

        spec1 = MagicMock()
        spec1.id = 1
        spec1.x = 0.0
        spec1.y = 0.0
        spec1.name = "Ghost"
        spec1.layer = 1
        spec1.node_id = 0
        spec1.gw_node = 0

        subs_config = MagicMock()
        subs_config.hydrograph_specs = [spec1]
        model.groundwater.subsidence_config = subs_config

        result = state.get_hydrograph_locations()
        assert len(result["subsidence"]) == 0

    def test_subsidence_locations_reproject_exception(self):
        """When reproject_coords raises, skip that spec."""
        state, model = _state_with_model(with_groundwater=True)

        spec1 = MagicMock()
        spec1.id = 1
        spec1.x = 500.0
        spec1.y = 600.0
        spec1.name = "Bad"
        spec1.layer = 1

        subs_config = MagicMock()
        subs_config.hydrograph_specs = [spec1]
        model.groundwater.subsidence_config = subs_config

        with patch.object(
            state, "reproject_coords", side_effect=RuntimeError("proj error")
        ):
            result = state.get_hydrograph_locations()

        assert len(result["subsidence"]) == 0

    def test_no_subsidence_config(self):
        """When no subsidence_config, subsidence list is empty."""
        state, model = _state_with_model(with_groundwater=True)
        model.groundwater.subsidence_config = None

        result = state.get_hydrograph_locations()
        assert result["subsidence"] == []


# ===========================================================================
# ModelState properties
# ===========================================================================


class TestModelStateProperties:
    """Tests for model property and is_loaded."""

    def test_model_property_returns_none(self):
        state = ModelState()
        assert state.model is None

    def test_model_property_returns_model(self):
        state, model = _state_with_model()
        assert state.model is model

    def test_is_loaded_false(self):
        state = ModelState()
        assert state.is_loaded is False

    def test_is_loaded_true(self):
        state, _ = _state_with_model()
        assert state.is_loaded is True


# ===========================================================================
# get_available_budgets: empty model
# ===========================================================================


class TestAvailableBudgetsEdgeCases:
    """Edge cases for get_available_budgets."""

    def test_no_model(self):
        state = ModelState()
        result = state.get_available_budgets()
        assert result == []

    def test_no_matching_metadata(self):
        state, model = _state_with_model()
        # No budget file metadata set at all
        result = state.get_available_budgets()
        assert result == []

    def test_budget_with_absolute_path(self, tmp_path):
        """Absolute budget paths are used directly."""
        state, model = _state_with_model()
        budget_file = tmp_path / "gw.hdf5"
        budget_file.write_bytes(b"fake")
        model.metadata["gw_budget_file"] = str(budget_file)

        result = state.get_available_budgets()
        assert "gw" in result


# ===========================================================================
# Global model_state instance
# ===========================================================================


class TestStreamHydrographReaderCaching:
    """Test the stream hydrograph reader cached return (line 684)."""

    def test_cached_return(self, tmp_path):
        state, model = _state_with_model()
        stream_file = tmp_path / "stream_hydro.out"
        stream_file.write_text("fake data")
        model.metadata["stream_hydrograph_file"] = str(stream_file)

        mock_reader = MagicMock()
        mock_reader.n_columns = 10
        mock_reader.n_timesteps = 50

        with patch(
            "pyiwfm.visualization.webapi.hydrograph_reader.IWFMHydrographReader",
            return_value=mock_reader,
        ) as mock_cls:
            first = state.get_stream_hydrograph_reader()
            second = state.get_stream_hydrograph_reader()

        assert first is second
        mock_cls.assert_called_once()


class TestStreamHydrographLocGwNodeNotFound:
    """Cover partial branch line 810->813: gw_node lookup returns None."""

    def test_gw_node_not_in_grid(self):
        """When stream node has (0,0) and gw_node points to a non-existent grid node."""
        state, model = _state_with_model(with_streams=True)
        model.metadata["stream_hydrograph_specs"] = [
            {"node_id": 5, "name": "S5"},
        ]

        stream_node = MagicMock()
        stream_node.x = 0.0
        stream_node.y = 0.0
        stream_node.gw_node = 999  # Not in grid (grid has nodes 1-4)
        stream_node.reach_id = 1
        model.streams.nodes = {5: stream_node}

        result = state.get_hydrograph_locations()
        # gw_node 999 not found -> x,y still (0,0) -> skipped
        assert len(result["stream"]) == 0


class TestSubsidenceLocGwNodeNotFound:
    """Cover partial branch line 839->841: subsidence gw_node lookup returns None."""

    def test_subsidence_gw_node_not_in_grid(self):
        """When subsidence spec has (0,0) and node_id points to non-existent grid node."""
        state, model = _state_with_model(with_groundwater=True)

        spec1 = MagicMock()
        spec1.id = 1
        spec1.x = 0.0
        spec1.y = 0.0
        spec1.name = "Subs"
        spec1.layer = 1
        spec1.node_id = 999  # Not in grid
        spec1.gw_node = 0

        subs_config = MagicMock()
        subs_config.hydrograph_specs = [spec1]
        model.groundwater.subsidence_config = subs_config

        result = state.get_hydrograph_locations()
        # gw_node not found -> still (0,0) -> skipped
        assert len(result["subsidence"]) == 0


class TestSubsidenceGlobNoMatchPattern:
    """Cover partial branch line 754->752: glob pattern has no matches."""

    def test_glob_no_matches_for_any_pattern(self, tmp_path):
        """When _results_dir exists but no Subsidence files are found."""
        state, model = _state_with_model(with_groundwater=True)
        state._results_dir = tmp_path
        model.groundwater.subsidence_config = None
        # Don't create any *Subsidence*.out files

        result = state.get_subsidence_reader()
        assert result is None


class TestGlobalModelState:
    """Test the global model_state instance."""

    def test_global_instance_exists(self):
        from pyiwfm.visualization.webapi.config import model_state
        assert isinstance(model_state, ModelState)
