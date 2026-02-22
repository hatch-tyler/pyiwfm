"""Extended unit tests for ModelState methods not covered by test_webapi_config.py.

Covers mesh computation (3D, surface, PyVista, surface JSON, bounds),
slice computation, results loaders (head, hydrograph, budget),
GeoJSON generation, observation management, and results info aggregation.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pydantic = pytest.importorskip("pydantic")

from pyiwfm.core.mesh import AppGrid, Element, Node  # noqa: E402
from pyiwfm.visualization.webapi.config import ModelState  # noqa: E402

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


def _make_mock_model(with_stratigraphy=True, with_streams=False, with_groundwater=False):
    model = MagicMock()
    model.name = "TestModel"
    model.grid = _make_grid()
    model.metadata = {}
    model.has_streams = with_streams
    model.has_lakes = False
    if with_stratigraphy:
        strat = MagicMock()
        strat.n_layers = 2
        strat.gs_elev = np.array([10.0, 20.0, 30.0, 40.0])
        strat.top_elev = np.array([[10.0, 5.0], [20.0, 10.0], [30.0, 15.0], [40.0, 20.0]])
        strat.bottom_elev = np.array([[5.0, 0.0], [10.0, 5.0], [15.0, 10.0], [20.0, 15.0]])
        model.stratigraphy = strat
    else:
        model.stratigraphy = None
    if with_groundwater:
        gw = MagicMock()
        gw.aquifer_params = MagicMock()
        gw.n_hydrograph_locations = 3
        gw.hydrograph_locations = []
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
# 3A. Mesh computation methods
# ===========================================================================


class TestMesh3D:
    """Tests for get_mesh_3d and _compute_mesh_3d."""

    def test_compute_mesh_3d_raises_no_model(self):
        state = ModelState()
        with pytest.raises(ValueError, match="No model loaded"):
            state._compute_mesh_3d()

    def test_get_mesh_3d_caches_result(self):
        """Second call returns the same cached bytes without recomputing."""
        state, _ = _state_with_model()
        fake_bytes = b"<VTKFile>mock</VTKFile>"

        with patch.object(state, "_compute_mesh_3d", return_value=fake_bytes) as mock_compute:
            first = state.get_mesh_3d()
            second = state.get_mesh_3d()

        assert first is second
        # _compute_mesh_3d called only once; second call used cache
        mock_compute.assert_called_once()

    def test_compute_mesh_3d_returns_bytes(self):
        """_compute_mesh_3d produces bytes via VTKExporter and vtk writer."""
        state, _ = _state_with_model()

        mock_writer = MagicMock()
        mock_writer.GetOutputString.return_value = "<VTKFile>test</VTKFile>"

        mock_vtk = MagicMock()
        mock_vtk.vtkXMLUnstructuredGridWriter.return_value = mock_writer

        mock_exporter_cls = MagicMock()

        with (
            patch.dict("sys.modules", {"vtk": mock_vtk}),
            patch("pyiwfm.visualization.vtk_export.VTKExporter", mock_exporter_cls),
        ):
            result = state._compute_mesh_3d()

        assert isinstance(result, bytes)
        assert b"<VTKFile>test</VTKFile>" in result


class TestMeshSurface:
    """Tests for get_mesh_surface and _compute_mesh_surface."""

    def test_compute_mesh_surface_raises_no_model(self):
        state = ModelState()
        with pytest.raises(ValueError, match="No model loaded"):
            state._compute_mesh_surface()

    def test_get_mesh_surface_caches_result(self):
        """Second call returns the cached bytes."""
        state, _ = _state_with_model(with_stratigraphy=False)
        fake_bytes = b"<VTKFile>surface</VTKFile>"

        with patch.object(state, "_compute_mesh_surface", return_value=fake_bytes) as mock_compute:
            first = state.get_mesh_surface()
            second = state.get_mesh_surface()

        assert first is second
        mock_compute.assert_called_once()


class TestPyvista3D:
    """Tests for get_pyvista_3d."""

    def test_get_pyvista_3d_raises_no_model(self):
        state = ModelState()
        with pytest.raises(ValueError, match="No model loaded"):
            state.get_pyvista_3d()

    def test_get_pyvista_3d_raises_no_stratigraphy(self):
        state, _ = _state_with_model(with_stratigraphy=False)
        with pytest.raises(ValueError, match="3D mesh requires stratigraphy"):
            state.get_pyvista_3d()

    def test_get_pyvista_3d_caches_result(self):
        state, _ = _state_with_model()
        mock_pv = MagicMock()
        mock_exporter_cls = MagicMock()
        mock_exporter_cls.return_value.to_pyvista_3d.return_value = mock_pv

        with patch("pyiwfm.visualization.vtk_export.VTKExporter", mock_exporter_cls):
            first = state.get_pyvista_3d()
            second = state.get_pyvista_3d()

        assert first is second
        # to_pyvista_3d called only once
        mock_exporter_cls.return_value.to_pyvista_3d.assert_called_once()


class TestSurfaceJson:
    """Tests for get_surface_json."""

    def test_get_surface_json_returns_expected_keys(self):
        """get_surface_json() dict has n_points, n_cells, points, polys, layer."""
        state, _ = _state_with_model()

        # Mock the internal _compute_surface_json to avoid needing real PyVista
        expected = {
            "n_points": 8,
            "n_cells": 6,
            "n_layers": 2,
            "points": [0.0] * 24,
            "polys": [3, 0, 1, 2, 3, 2, 3, 4],
            "layer": [1, 1, 1, 2, 2, 2],
        }
        with patch.object(state, "_compute_surface_json", return_value=expected):
            result = state.get_surface_json(layer=0)

        assert "n_points" in result
        assert "n_cells" in result
        assert "points" in result
        assert "polys" in result
        assert "layer" in result

    def test_get_surface_json_caches_per_layer(self):
        """Different layers are cached separately."""
        state, _ = _state_with_model()

        layer0_data = {
            "n_points": 8,
            "n_cells": 6,
            "n_layers": 2,
            "points": [],
            "polys": [],
            "layer": [],
        }
        layer1_data = {
            "n_points": 4,
            "n_cells": 3,
            "n_layers": 1,
            "points": [],
            "polys": [],
            "layer": [],
        }

        with patch.object(state, "_compute_surface_json", side_effect=[layer0_data, layer1_data]):
            r0 = state.get_surface_json(layer=0)
            r1 = state.get_surface_json(layer=1)
            # Call again - should be cached
            r0_again = state.get_surface_json(layer=0)
            r1_again = state.get_surface_json(layer=1)

        assert r0 is r0_again
        assert r1 is r1_again
        assert r0["n_points"] == 8
        assert r1["n_points"] == 4

    def test_get_surface_json_layer0_legacy_cache(self):
        """Layer 0 result is also stored in _surface_json_data for legacy compat."""
        state, _ = _state_with_model()
        data = {"n_points": 10, "n_cells": 5, "n_layers": 2, "points": [], "polys": [], "layer": []}

        with patch.object(state, "_compute_surface_json", return_value=data):
            state.get_surface_json(layer=0)

        assert state._surface_json_data is data


class TestBounds:
    """Tests for get_bounds and _compute_bounds."""

    def test_get_bounds_returns_six_tuple(self):
        state, _ = _state_with_model()
        bounds = state.get_bounds()
        assert len(bounds) == 6

    def test_get_bounds_no_model(self):
        state = ModelState()
        bounds = state.get_bounds()
        assert bounds == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def test_get_bounds_with_stratigraphy(self):
        state, _ = _state_with_model()
        bounds = state.get_bounds()
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        assert xmin == 0.0
        assert xmax == 100.0
        assert ymin == 0.0
        assert ymax == 100.0
        # zmin = min(bottom_elev) = 0.0, zmax = max(top_elev) = 40.0
        assert zmin == 0.0
        assert zmax == 40.0

    def test_get_bounds_without_stratigraphy(self):
        state, _ = _state_with_model(with_stratigraphy=False)
        bounds = state.get_bounds()
        _, _, _, _, zmin, zmax = bounds
        assert zmin == 0.0
        assert zmax == 0.0

    def test_get_bounds_caches(self):
        state, _ = _state_with_model()
        first = state.get_bounds()
        second = state.get_bounds()
        assert first is second


# ===========================================================================
# 3B. Slice computation
# ===========================================================================


class TestSliceJson:
    """Tests for get_slice_json."""

    def test_get_slice_json_returns_expected_structure(self):
        """Result dict has n_points, n_cells, n_layers, points, polys, layer."""
        state, _ = _state_with_model()

        mock_pv = MagicMock()

        mock_slice = MagicMock()
        mock_slice.n_cells = 3
        mock_slice.n_points = 6
        mock_slice.points = np.zeros((6, 3), dtype=np.float32)
        mock_slice.faces = np.array([3, 0, 1, 2, 3, 3, 4, 5])
        mock_slice.cell_data = {"layer": np.array([1, 1, 2])}

        with (
            patch.object(state, "get_pyvista_3d", return_value=mock_pv),
            patch("pyiwfm.visualization.webapi.slicing.SlicingController") as mock_slicer_cls,
        ):
            mock_slicer = mock_slicer_cls.return_value
            mock_slicer.normalized_to_position_along.return_value = (50.0, 50.0, 0.0)
            mock_slicer.slice_arbitrary.return_value = mock_slice

            result = state.get_slice_json(angle=45.0, position=0.5)

        assert result["n_points"] == 6
        assert result["n_cells"] == 3
        assert result["n_layers"] == 2
        assert "points" in result
        assert "polys" in result
        assert result["layer"] == [1, 1, 2]

    def test_get_slice_json_empty_result(self):
        """When the slice misses the mesh, returns zeros."""
        state, _ = _state_with_model()

        mock_pv = MagicMock()

        mock_slice = MagicMock()
        mock_slice.n_cells = 0

        with (
            patch.object(state, "get_pyvista_3d", return_value=mock_pv),
            patch("pyiwfm.visualization.webapi.slicing.SlicingController") as mock_slicer_cls,
        ):
            mock_slicer = mock_slicer_cls.return_value
            mock_slicer.normalized_to_position_along.return_value = (0.0, 0.0, 0.0)
            mock_slicer.slice_arbitrary.return_value = mock_slice

            result = state.get_slice_json(angle=0.0, position=0.0)

        assert result["n_points"] == 0
        assert result["n_cells"] == 0
        assert result["points"] == []
        assert result["polys"] == []

    def test_get_slice_json_no_layer_cell_data(self):
        """When slice has no 'layer' cell_data, defaults to [1]*n_cells."""
        state, _ = _state_with_model()

        mock_pv = MagicMock()

        mock_slice = MagicMock()
        mock_slice.n_cells = 2
        mock_slice.n_points = 4
        mock_slice.points = np.zeros((4, 3), dtype=np.float32)
        mock_slice.faces = np.array([3, 0, 1, 2, 3, 1, 2, 3])
        mock_slice.cell_data = {}  # No 'layer' key

        with (
            patch.object(state, "get_pyvista_3d", return_value=mock_pv),
            patch("pyiwfm.visualization.webapi.slicing.SlicingController") as mock_slicer_cls,
        ):
            mock_slicer = mock_slicer_cls.return_value
            mock_slicer.normalized_to_position_along.return_value = (50.0, 50.0, 0.0)
            mock_slicer.slice_arbitrary.return_value = mock_slice

            result = state.get_slice_json(angle=90.0, position=0.5)

        assert result["layer"] == [1, 1]


# ===========================================================================
# 3C. Results loaders
# ===========================================================================


class TestHeadLoader:
    """Tests for get_head_loader."""

    def test_returns_none_no_model(self):
        state = ModelState()
        assert state.get_head_loader() is None

    def test_returns_none_no_metadata_key(self):
        state, _ = _state_with_model()
        assert state.get_head_loader() is None

    def test_returns_none_file_not_found(self, tmp_path):
        state, model = _state_with_model()
        model.metadata["gw_head_all_file"] = str(tmp_path / "nonexistent.hdf5")
        assert state.get_head_loader() is None

    def test_creates_loader_when_file_exists(self, tmp_path):
        state, model = _state_with_model()
        head_file = tmp_path / "head.hdf5"
        head_file.write_bytes(b"fake")
        model.metadata["gw_head_all_file"] = str(head_file)

        mock_loader = MagicMock()
        mock_loader.n_frames = 10
        with patch(
            "pyiwfm.io.head_loader.LazyHeadDataLoader",
            return_value=mock_loader,
        ):
            result = state.get_head_loader()

        assert result is mock_loader

    def test_caches_loader(self, tmp_path):
        state, model = _state_with_model()
        head_file = tmp_path / "head.hdf5"
        head_file.write_bytes(b"fake")
        model.metadata["gw_head_all_file"] = str(head_file)

        mock_loader = MagicMock()
        mock_loader.n_frames = 10
        # shape must match model's n_layers to avoid triggering re-conversion
        mock_loader.shape = (100, 2)
        with patch(
            "pyiwfm.io.head_loader.LazyHeadDataLoader",
            return_value=mock_loader,
        ) as mock_cls:
            first = state.get_head_loader()
            second = state.get_head_loader()

        assert first is second
        mock_cls.assert_called_once()

    def test_returns_none_on_exception(self, tmp_path):
        state, model = _state_with_model()
        head_file = tmp_path / "head.hdf5"
        head_file.write_bytes(b"fake")
        model.metadata["gw_head_all_file"] = str(head_file)

        with patch(
            "pyiwfm.io.head_loader.LazyHeadDataLoader",
            side_effect=RuntimeError("corrupt file"),
        ):
            result = state.get_head_loader()

        assert result is None


class TestGWHydrographReader:
    """Tests for get_gw_hydrograph_reader."""

    def test_returns_none_no_model(self):
        state = ModelState()
        assert state.get_gw_hydrograph_reader() is None

    def test_returns_none_no_metadata_key(self):
        state, _ = _state_with_model()
        assert state.get_gw_hydrograph_reader() is None

    def test_returns_none_file_not_found(self, tmp_path):
        state, model = _state_with_model()
        model.metadata["gw_hydrograph_file"] = str(tmp_path / "gw_hydro.out")
        assert state.get_gw_hydrograph_reader() is None

    def test_creates_reader_when_file_exists(self, tmp_path):
        state, model = _state_with_model()
        hydro_file = tmp_path / "gw_hydro.out"
        hydro_file.write_text("fake data")
        model.metadata["gw_hydrograph_file"] = str(hydro_file)

        mock_reader = MagicMock()
        mock_reader.n_columns = 5
        mock_reader.n_timesteps = 100
        with patch(
            "pyiwfm.io.hydrograph_reader.IWFMHydrographReader",
            return_value=mock_reader,
        ):
            result = state.get_gw_hydrograph_reader()

        assert result is mock_reader

    def test_caches_reader(self, tmp_path):
        state, model = _state_with_model()
        hydro_file = tmp_path / "gw_hydro.out"
        hydro_file.write_text("fake data")
        model.metadata["gw_hydrograph_file"] = str(hydro_file)

        mock_reader = MagicMock()
        mock_reader.n_columns = 5
        mock_reader.n_timesteps = 100
        with patch(
            "pyiwfm.io.hydrograph_reader.IWFMHydrographReader",
            return_value=mock_reader,
        ) as mock_cls:
            first = state.get_gw_hydrograph_reader()
            second = state.get_gw_hydrograph_reader()

        assert first is second
        mock_cls.assert_called_once()


class TestStreamHydrographReader:
    """Tests for get_stream_hydrograph_reader."""

    def test_returns_none_no_model(self):
        state = ModelState()
        assert state.get_stream_hydrograph_reader() is None

    def test_returns_none_no_metadata_key(self):
        state, _ = _state_with_model()
        assert state.get_stream_hydrograph_reader() is None

    def test_creates_reader_when_file_exists(self, tmp_path):
        state, model = _state_with_model()
        stream_file = tmp_path / "stream_hydro.out"
        stream_file.write_text("fake data")
        model.metadata["stream_hydrograph_file"] = str(stream_file)

        mock_reader = MagicMock()
        mock_reader.n_columns = 10
        mock_reader.n_timesteps = 50
        with patch(
            "pyiwfm.io.hydrograph_reader.IWFMHydrographReader",
            return_value=mock_reader,
        ):
            result = state.get_stream_hydrograph_reader()

        assert result is mock_reader

    def test_returns_none_file_not_found(self, tmp_path):
        state, model = _state_with_model()
        model.metadata["stream_hydrograph_file"] = str(tmp_path / "missing.out")
        assert state.get_stream_hydrograph_reader() is None


class TestBudgetReader:
    """Tests for get_budget_reader and get_available_budgets."""

    def test_returns_none_unknown_type(self):
        state, _ = _state_with_model()
        assert state.get_budget_reader("unknown_budget") is None

    def test_returns_none_no_model(self):
        state = ModelState()
        assert state.get_budget_reader("gw") is None

    def test_returns_none_no_metadata_path(self):
        state, _ = _state_with_model()
        # Model has no budget file metadata
        assert state.get_budget_reader("gw") is None

    def test_creates_reader_when_file_exists(self, tmp_path):
        state, model = _state_with_model()
        budget_file = tmp_path / "gw_budget.hdf5"
        budget_file.write_bytes(b"fake")
        model.metadata["gw_budget_file"] = str(budget_file)

        mock_reader = MagicMock()
        mock_reader.descriptor = "GROUNDWATER BUDGET"
        with patch(
            "pyiwfm.io.budget.BudgetReader",
            return_value=mock_reader,
        ):
            result = state.get_budget_reader("gw")

        assert result is mock_reader

    def test_caches_reader(self, tmp_path):
        state, model = _state_with_model()
        budget_file = tmp_path / "gw_budget.hdf5"
        budget_file.write_bytes(b"fake")
        model.metadata["gw_budget_file"] = str(budget_file)

        mock_reader = MagicMock()
        mock_reader.descriptor = "GROUNDWATER BUDGET"
        with patch(
            "pyiwfm.io.budget.BudgetReader",
            return_value=mock_reader,
        ) as mock_cls:
            first = state.get_budget_reader("gw")
            second = state.get_budget_reader("gw")

        assert first is second
        mock_cls.assert_called_once()

    def test_returns_none_on_exception(self, tmp_path):
        state, model = _state_with_model()
        budget_file = tmp_path / "bad_budget.hdf5"
        budget_file.write_bytes(b"corrupt")
        model.metadata["gw_budget_file"] = str(budget_file)

        with patch(
            "pyiwfm.io.budget.BudgetReader",
            side_effect=RuntimeError("cannot read"),
        ):
            result = state.get_budget_reader("gw")

        assert result is None

    def test_get_available_budgets_lists_existing(self, tmp_path):
        state, model = _state_with_model()
        gw_file = tmp_path / "gw_budget.hdf5"
        gw_file.write_bytes(b"fake")
        stream_file = tmp_path / "stream_budget.hdf5"
        stream_file.write_bytes(b"fake")
        model.metadata["gw_budget_file"] = str(gw_file)
        model.metadata["stream_budget_file"] = str(stream_file)
        # Also add a path that does NOT exist
        model.metadata["lake_budget_file"] = str(tmp_path / "nonexistent.hdf5")

        available = state.get_available_budgets()
        assert "gw" in available
        assert "stream" in available
        assert "lake" not in available

    def test_get_available_budgets_resolves_relative_paths(self, tmp_path):
        state, model = _state_with_model()
        state._results_dir = tmp_path
        # Create the file at results_dir / relative_name
        (tmp_path / "rootzone_rz.hdf5").write_bytes(b"fake")
        model.metadata["rootzone_rz_budget_file"] = "rootzone_rz.hdf5"

        available = state.get_available_budgets()
        assert "rootzone" in available


# ===========================================================================
# 3D. GeoJSON methods
# ===========================================================================


class TestMeshGeoJSON:
    """Tests for get_mesh_geojson."""

    def test_returns_valid_feature_collection(self):
        state, _ = _state_with_model()
        # Mock transformer to passthrough (no reprojection)
        state._transformer = None
        with patch.object(state, "_get_transformer", return_value=None):
            geojson = state.get_mesh_geojson(layer=1)

        assert geojson["type"] == "FeatureCollection"
        assert len(geojson["features"]) == 1
        feat = geojson["features"][0]
        assert feat["type"] == "Feature"
        assert feat["geometry"]["type"] == "Polygon"
        assert feat["properties"]["element_id"] == 1
        assert feat["properties"]["layer"] == 1

    def test_ring_is_closed(self):
        """The polygon ring's first and last coords should be identical."""
        state, _ = _state_with_model()
        with patch.object(state, "_get_transformer", return_value=None):
            geojson = state.get_mesh_geojson(layer=1)

        ring = geojson["features"][0]["geometry"]["coordinates"][0]
        assert ring[0] == ring[-1]

    def test_caches_per_layer(self):
        state, _ = _state_with_model()
        with patch.object(state, "_get_transformer", return_value=None):
            result_a = state.get_mesh_geojson(layer=1)
            result_b = state.get_mesh_geojson(layer=2)
            result_a2 = state.get_mesh_geojson(layer=1)

        assert result_a is result_a2
        assert result_a is not result_b

    def test_geojson_with_reprojection(self):
        """When transformer is available, coordinates are reprojected."""
        state, _ = _state_with_model()

        mock_transformer = MagicMock()
        # Return constant reprojected coords for any input
        mock_transformer.transform.return_value = (-121.5, 38.5)
        state._transformer = mock_transformer

        geojson = state.get_mesh_geojson(layer=1)
        ring = geojson["features"][0]["geometry"]["coordinates"][0]
        # All coords should be the mocked reprojected values
        for coord in ring:
            assert coord == [-121.5, 38.5]

    def test_geojson_empty_when_no_model(self):
        state = ModelState()
        state._model = MagicMock()
        state._model.grid = None
        geojson = state._compute_mesh_geojson(layer=1)
        assert geojson == {"type": "FeatureCollection", "features": []}


# ===========================================================================
# 3E. Observation management
# ===========================================================================


class TestObservations:
    """Tests for add/get/list/delete observation."""

    def test_add_and_get(self):
        state = ModelState()
        data = {
            "filename": "obs.csv",
            "type": "gw",
            "n_records": 5,
            "location_id": 42,
            "times": [],
            "values": [],
        }
        state.add_observation("obs-1", data)
        retrieved = state.get_observation("obs-1")
        assert retrieved is data
        assert retrieved["filename"] == "obs.csv"

    def test_get_unknown_returns_none(self):
        state = ModelState()
        assert state.get_observation("nonexistent") is None

    def test_list_observations_returns_all(self):
        state = ModelState()
        state.add_observation(
            "a", {"filename": "a.csv", "type": "gw", "location_id": 1, "n_records": 3}
        )
        state.add_observation(
            "b", {"filename": "b.csv", "type": "stream", "location_id": 2, "n_records": 7}
        )
        obs_list = state.list_observations()
        assert len(obs_list) == 2
        ids = {o["id"] for o in obs_list}
        assert ids == {"a", "b"}

    def test_list_observations_includes_summary_fields(self):
        state = ModelState()
        state.add_observation(
            "x",
            {
                "filename": "x.csv",
                "type": "gw",
                "location_id": 10,
                "n_records": 20,
                "extra_field": "should_not_appear",
            },
        )
        obs_list = state.list_observations()
        entry = obs_list[0]
        assert entry["id"] == "x"
        assert entry["filename"] == "x.csv"
        assert entry["type"] == "gw"
        assert entry["location_id"] == 10
        assert entry["n_records"] == 20
        assert "extra_field" not in entry

    def test_delete_existing(self):
        state = ModelState()
        state.add_observation("d1", {"filename": "d1.csv"})
        assert state.delete_observation("d1") is True
        assert state.get_observation("d1") is None
        assert state.list_observations() == []

    def test_delete_unknown_returns_false(self):
        state = ModelState()
        assert state.delete_observation("ghost") is False

    def test_observations_cleared_on_set_model(self):
        state = ModelState()
        state.add_observation("obs1", {"filename": "test.csv"})
        model = _make_mock_model()
        state.set_model(model)
        assert state.list_observations() == []
        assert state.get_observation("obs1") is None


# ===========================================================================
# Results info aggregation
# ===========================================================================


class TestResultsInfo:
    """Tests for get_results_info."""

    def test_no_model(self):
        state = ModelState()
        info = state.get_results_info()
        assert info["has_results"] is False
        assert info["available_budgets"] == []
        assert info["n_head_timesteps"] == 0
        assert info["head_time_range"] is None

    def test_with_head_data(self):
        state, model = _state_with_model()

        mock_loader = MagicMock()
        mock_loader.n_frames = 24
        mock_loader.times = [datetime(2000, 1, 1), datetime(2001, 12, 31)]

        with (
            patch.object(state, "get_head_loader", return_value=mock_loader),
            patch.object(state, "get_available_budgets", return_value=[]),
            patch.object(state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(state, "get_stream_hydrograph_reader", return_value=None),
        ):
            info = state.get_results_info()

        assert info["has_results"] is True
        assert info["n_head_timesteps"] == 24
        assert info["head_time_range"]["start"] == "2000-01-01T00:00:00"
        assert info["head_time_range"]["end"] == "2001-12-31T00:00:00"

    def test_with_budgets(self):
        state, model = _state_with_model()

        with (
            patch.object(state, "get_head_loader", return_value=None),
            patch.object(state, "get_available_budgets", return_value=["gw", "stream"]),
            patch.object(state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(state, "get_stream_hydrograph_reader", return_value=None),
        ):
            info = state.get_results_info()

        assert info["has_results"] is True
        assert info["available_budgets"] == ["gw", "stream"]

    def test_with_gw_hydrographs(self):
        state, model = _state_with_model(with_groundwater=True)

        # Populate hydrograph_locations so physical count is 3
        locs = []
        for i in range(3):
            loc = MagicMock()
            loc.x = float(i * 100)
            loc.y = float(i * 100)
            loc.name = f"Well-{i + 1}"
            loc.layer = 1
            loc.node_id = i + 1
            locs.append(loc)
        model.groundwater.hydrograph_locations = locs

        mock_gw_reader = MagicMock()
        mock_gw_reader.n_timesteps = 50

        with (
            patch.object(state, "get_head_loader", return_value=None),
            patch.object(state, "get_available_budgets", return_value=[]),
            patch.object(state, "get_gw_hydrograph_reader", return_value=mock_gw_reader),
            patch.object(state, "get_stream_hydrograph_reader", return_value=None),
        ):
            info = state.get_results_info()

        assert info["has_results"] is True
        assert info["has_gw_hydrographs"] is True
        assert info["n_gw_hydrographs"] == 3

    def test_with_stream_hydrographs(self):
        state, model = _state_with_model()
        model.metadata["stream_hydrograph_specs"] = [
            {"node_id": 1, "name": "S1"},
            {"node_id": 2, "name": "S2"},
        ]

        mock_stream_reader = MagicMock()
        mock_stream_reader.n_timesteps = 30

        with (
            patch.object(state, "get_head_loader", return_value=None),
            patch.object(state, "get_available_budgets", return_value=[]),
            patch.object(state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(state, "get_stream_hydrograph_reader", return_value=mock_stream_reader),
        ):
            info = state.get_results_info()

        assert info["has_stream_hydrographs"] is True
        assert info["n_stream_hydrographs"] == 2

    def test_has_results_false_when_all_empty(self):
        state, model = _state_with_model()

        with (
            patch.object(state, "get_head_loader", return_value=None),
            patch.object(state, "get_available_budgets", return_value=[]),
            patch.object(state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(state, "get_stream_hydrograph_reader", return_value=None),
        ):
            info = state.get_results_info()

        assert info["has_results"] is False

    def test_head_time_range_none_when_no_times(self):
        state, model = _state_with_model()

        mock_loader = MagicMock()
        mock_loader.n_frames = 0
        mock_loader.times = []

        with (
            patch.object(state, "get_head_loader", return_value=mock_loader),
            patch.object(state, "get_available_budgets", return_value=[]),
            patch.object(state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(state, "get_stream_hydrograph_reader", return_value=None),
        ):
            info = state.get_results_info()

        assert info["head_time_range"] is None
        assert info["n_head_timesteps"] == 0


# ===========================================================================
# Additional coverage tests targeting specific uncovered lines
# ===========================================================================


class TestComputeMeshSurfaceBody:
    """Cover lines 151-161: _compute_mesh_surface VTKExporter + vtk writer body."""

    def test_compute_mesh_surface_returns_bytes(self):
        """_compute_mesh_surface produces bytes via VTKExporter and vtk writer."""
        state, _ = _state_with_model(with_stratigraphy=False)

        mock_writer = MagicMock()
        mock_writer.GetOutputString.return_value = "<VTKFile>surface</VTKFile>"

        mock_vtk = MagicMock()
        mock_vtk.vtkXMLUnstructuredGridWriter.return_value = mock_writer

        mock_exporter_cls = MagicMock()
        mock_2d_mesh = MagicMock()
        mock_exporter_cls.return_value.create_2d_mesh.return_value = mock_2d_mesh

        with (
            patch.dict("sys.modules", {"vtk": mock_vtk}),
            patch("pyiwfm.visualization.vtk_export.VTKExporter", mock_exporter_cls),
        ):
            result = state._compute_mesh_surface()

        assert isinstance(result, bytes)
        assert b"<VTKFile>surface</VTKFile>" in result
        mock_exporter_cls.assert_called_once_with(grid=state._model.grid)
        mock_exporter_cls.return_value.create_2d_mesh.assert_called_once()
        mock_writer.SetWriteToOutputString.assert_called_once_with(True)
        mock_writer.SetInputData.assert_called_once_with(mock_2d_mesh)
        mock_writer.Write.assert_called_once()


class TestSurfaceJsonLegacyCache:
    """Cover line 192: get_surface_json layer=0 returning from _surface_json_data."""

    def test_layer0_returns_from_legacy_cache_when_not_in_layer_cache(self):
        """When layer=0 is NOT in _layer_surface_cache but IS in _surface_json_data."""
        state, _ = _state_with_model()
        legacy_data = {
            "n_points": 42,
            "n_cells": 10,
            "n_layers": 2,
            "points": [],
            "polys": [],
            "layer": [],
        }
        # Set the legacy cache directly, but NOT in _layer_surface_cache
        state._surface_json_data = legacy_data

        result = state.get_surface_json(layer=0)
        assert result is legacy_data


class TestComputeSurfaceJsonBody:
    """Cover lines 208-240: _compute_surface_json with real PyVista-like mocking."""

    def test_compute_surface_json_all_layers(self):
        """layer=0 extracts surface from the full mesh."""
        state, _ = _state_with_model()

        mock_surface = MagicMock()
        mock_surface.n_points = 8
        mock_surface.n_cells = 6
        mock_surface.points = MagicMock()
        mock_surface.points.astype.return_value.ravel.return_value.tolist.return_value = [0.0] * 24
        mock_surface.faces = MagicMock()
        mock_surface.faces.tolist.return_value = [3, 0, 1, 2, 3, 2, 3, 4]
        mock_surface.cell_data = {"layer": MagicMock()}
        mock_surface.cell_data["layer"].tolist.return_value = [1, 1, 1, 2, 2, 2]

        mock_pv_mesh = MagicMock()
        mock_pv_mesh.extract_surface.return_value = mock_surface

        with patch.object(state, "get_pyvista_3d", return_value=mock_pv_mesh):
            result = state._compute_surface_json(layer=0)

        assert result["n_points"] == 8
        assert result["n_cells"] == 6
        assert result["n_layers"] == 2
        assert result["points"] == [0.0] * 24
        assert result["polys"] == [3, 0, 1, 2, 3, 2, 3, 4]
        assert result["layer"] == [1, 1, 1, 2, 2, 2]
        mock_pv_mesh.extract_surface.assert_called_once()

    def test_compute_surface_json_specific_layer(self):
        """layer>0 filters via threshold then extract_surface."""
        state, _ = _state_with_model()

        mock_surface = MagicMock()
        mock_surface.n_points = 4
        mock_surface.n_cells = 2
        mock_surface.points = MagicMock()
        mock_surface.points.astype.return_value.ravel.return_value.tolist.return_value = [1.0] * 12
        mock_surface.faces = MagicMock()
        mock_surface.faces.tolist.return_value = [3, 0, 1, 2]
        mock_surface.cell_data = {"layer": MagicMock()}
        mock_surface.cell_data["layer"].tolist.return_value = [2, 2]

        mock_filtered = MagicMock()
        mock_filtered.extract_surface.return_value = mock_surface

        mock_pv_mesh = MagicMock()
        mock_pv_mesh.threshold.return_value = mock_filtered

        with patch.object(state, "get_pyvista_3d", return_value=mock_pv_mesh):
            result = state._compute_surface_json(layer=2)

        assert result["n_points"] == 4
        assert result["n_cells"] == 2
        assert result["n_layers"] == 2  # For single layer, n_layers = layer number
        assert result["layer"] == [2, 2]
        mock_pv_mesh.threshold.assert_called_once_with(value=[2, 2], scalars="layer")

    def test_compute_surface_json_no_layer_cell_data(self):
        """When extract_surface has no 'layer' in cell_data, uses fallback."""
        state, _ = _state_with_model()

        mock_surface = MagicMock()
        mock_surface.n_points = 4
        mock_surface.n_cells = 3
        mock_surface.points = MagicMock()
        mock_surface.points.astype.return_value.ravel.return_value.tolist.return_value = [0.0] * 12
        mock_surface.faces = MagicMock()
        mock_surface.faces.tolist.return_value = [3, 0, 1, 2]
        mock_surface.cell_data = {}  # No 'layer' key

        mock_pv_mesh = MagicMock()
        mock_pv_mesh.extract_surface.return_value = mock_surface

        with patch.object(state, "get_pyvista_3d", return_value=mock_pv_mesh):
            result = state._compute_surface_json(layer=0)

        # Fallback: layer_data = [1] * n_cells for layer=0
        assert result["layer"] == [1, 1, 1]
        assert result["n_layers"] == 1

    def test_compute_surface_json_specific_layer_no_cell_data(self):
        """layer>0 with no 'layer' cell_data uses [layer]*n_cells."""
        state, _ = _state_with_model()

        mock_surface = MagicMock()
        mock_surface.n_points = 2
        mock_surface.n_cells = 1
        mock_surface.points = MagicMock()
        mock_surface.points.astype.return_value.ravel.return_value.tolist.return_value = [0.0] * 6
        mock_surface.faces = MagicMock()
        mock_surface.faces.tolist.return_value = [3, 0, 1, 2]
        mock_surface.cell_data = {}

        mock_filtered = MagicMock()
        mock_filtered.extract_surface.return_value = mock_surface

        mock_pv_mesh = MagicMock()
        mock_pv_mesh.threshold.return_value = mock_filtered

        with patch.object(state, "get_pyvista_3d", return_value=mock_pv_mesh):
            result = state._compute_surface_json(layer=3)

        # For layer>0 with no cell_data: [layer]*n_cells
        assert result["layer"] == [3]
        assert result["n_layers"] == 3

    def test_compute_surface_json_empty_layer_data(self):
        """When layer_data is empty, n_layers defaults to 1 for layer=0."""
        state, _ = _state_with_model()

        mock_surface = MagicMock()
        mock_surface.n_points = 0
        mock_surface.n_cells = 0
        mock_surface.points = MagicMock()
        mock_surface.points.astype.return_value.ravel.return_value.tolist.return_value = []
        mock_surface.faces = MagicMock()
        mock_surface.faces.tolist.return_value = []
        mock_surface.cell_data = {"layer": MagicMock()}
        mock_surface.cell_data["layer"].tolist.return_value = []

        mock_pv_mesh = MagicMock()
        mock_pv_mesh.extract_surface.return_value = mock_surface

        with patch.object(state, "get_pyvista_3d", return_value=mock_pv_mesh):
            result = state._compute_surface_json(layer=0)

        assert result["n_layers"] == 1  # Empty layer_data defaults to 1
        assert result["layer"] == []


class TestGetTransformerImportError:
    """Cover _get_transformer ImportError branch."""

    def test_returns_none_when_pyproj_not_available(self):
        """When pyproj cannot be imported, returns None and logs warning."""
        state = ModelState()
        state._transformer = None  # Ensure not cached

        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pyproj":
                raise ImportError("No module named 'pyproj'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = state._get_transformer()

        assert result is None


class TestHeadLoaderRelativePath:
    """Cover line 403: head_path relative path resolution via _results_dir."""

    def test_head_file_relative_path_resolved(self, tmp_path):
        state, model = _state_with_model()
        state._results_dir = tmp_path
        head_file = tmp_path / "output" / "head.hdf5"
        head_file.parent.mkdir(parents=True, exist_ok=True)
        head_file.write_bytes(b"fake")
        # Use a relative path (not absolute)
        model.metadata["gw_head_all_file"] = "output/head.hdf5"

        mock_loader = MagicMock()
        mock_loader.n_frames = 5
        with patch(
            "pyiwfm.io.head_loader.LazyHeadDataLoader",
            return_value=mock_loader,
        ):
            result = state.get_head_loader()

        assert result is mock_loader


class TestGWHydrographRelativePathAndException:
    """Cover lines 437 (relative path) and 451-453 (exception branch)."""

    def test_gw_hydrograph_relative_path_resolved(self, tmp_path):
        state, model = _state_with_model()
        state._results_dir = tmp_path
        hydro_file = tmp_path / "gw_hydro.out"
        hydro_file.write_text("fake data")
        model.metadata["gw_hydrograph_file"] = "gw_hydro.out"

        mock_reader = MagicMock()
        mock_reader.n_columns = 3
        mock_reader.n_timesteps = 20
        with patch(
            "pyiwfm.io.hydrograph_reader.IWFMHydrographReader",
            return_value=mock_reader,
        ):
            result = state.get_gw_hydrograph_reader()

        assert result is mock_reader

    def test_gw_hydrograph_returns_none_on_exception(self, tmp_path):
        state, model = _state_with_model()
        hydro_file = tmp_path / "gw_hydro.out"
        hydro_file.write_text("bad data")
        model.metadata["gw_hydrograph_file"] = str(hydro_file)

        with patch(
            "pyiwfm.io.hydrograph_reader.IWFMHydrographReader",
            side_effect=RuntimeError("parse error"),
        ):
            result = state.get_gw_hydrograph_reader()

        assert result is None


class TestStreamHydrographRelativePathAndException:
    """Cover lines 471 (relative path) and 485-487 (exception branch)."""

    def test_stream_hydrograph_relative_path_resolved(self, tmp_path):
        state, model = _state_with_model()
        state._results_dir = tmp_path
        stream_file = tmp_path / "stream_hydro.out"
        stream_file.write_text("fake data")
        model.metadata["stream_hydrograph_file"] = "stream_hydro.out"

        mock_reader = MagicMock()
        mock_reader.n_columns = 8
        mock_reader.n_timesteps = 40
        with patch(
            "pyiwfm.io.hydrograph_reader.IWFMHydrographReader",
            return_value=mock_reader,
        ):
            result = state.get_stream_hydrograph_reader()

        assert result is mock_reader

    def test_stream_hydrograph_returns_none_on_exception(self, tmp_path):
        state, model = _state_with_model()
        stream_file = tmp_path / "stream_hydro.out"
        stream_file.write_text("bad data")
        model.metadata["stream_hydrograph_file"] = str(stream_file)

        with patch(
            "pyiwfm.io.hydrograph_reader.IWFMHydrographReader",
            side_effect=RuntimeError("stream parse error"),
        ):
            result = state.get_stream_hydrograph_reader()

        assert result is None


class TestHydrographLocations:
    """Cover lines 500 and 518-536: get_hydrograph_locations."""

    def test_returns_empty_when_no_model(self):
        """Line 500: early return when _model is None."""
        state = ModelState()
        result = state.get_hydrograph_locations()
        assert result == {"gw": [], "stream": [], "subsidence": [], "tile_drain": []}

    def test_gw_hydrograph_locations(self):
        """GW locations use 1-based index from hydrograph_locations list."""
        state, model = _state_with_model(with_groundwater=True)

        loc1 = MagicMock()
        loc1.x = 100.0
        loc1.y = 200.0
        loc1.name = "Well A"
        loc1.layer = 1

        loc2 = MagicMock()
        loc2.x = 300.0
        loc2.y = 400.0
        loc2.name = None  # Will use default name
        loc2.layer = 2

        model.groundwater.hydrograph_locations = [loc1, loc2]

        with patch.object(state, "reproject_coords", side_effect=lambda x, y: (x, y)):
            result = state.get_hydrograph_locations()

        assert len(result["gw"]) == 2
        assert result["gw"][0]["id"] == 1
        assert result["gw"][0]["name"] == "Well A"
        assert result["gw"][0]["lng"] == 100.0
        assert result["gw"][0]["lat"] == 200.0
        assert result["gw"][0]["layer"] == 1
        assert result["gw"][1]["id"] == 2
        assert result["gw"][1]["name"] == "GW Hydrograph 2"

    def test_stream_hydrograph_locations_with_valid_coords(self):
        """Lines 518-536: stream specs iteration with valid node coords."""
        state, model = _state_with_model(with_streams=True)
        model.metadata["stream_hydrograph_specs"] = [
            {"node_id": 10, "name": "Stream Obs 1"},
        ]

        stream_node = MagicMock()
        stream_node.x = 500.0
        stream_node.y = 600.0
        stream_node.gw_node = None
        stream_node.reach_id = 3
        model.streams.nodes = {10: stream_node}

        with patch.object(state, "reproject_coords", side_effect=lambda x, y: (x, y)):
            result = state.get_hydrograph_locations()

        assert len(result["stream"]) == 1
        assert result["stream"][0]["id"] == 10
        assert result["stream"][0]["lng"] == 500.0
        assert result["stream"][0]["lat"] == 600.0
        assert result["stream"][0]["name"] == "Stream Obs 1"
        assert result["stream"][0]["reach_id"] == 3

    def test_stream_hydrograph_locations_uses_gw_node_when_zero(self):
        """When stream node has x=0,y=0, falls back to GW node coords."""
        state, model = _state_with_model(with_streams=True)
        model.metadata["stream_hydrograph_specs"] = [
            {"node_id": 5, "name": "S5"},
        ]

        stream_node = MagicMock()
        stream_node.x = 0.0
        stream_node.y = 0.0
        stream_node.gw_node = 2  # References GW node ID 2
        stream_node.reach_id = 1
        model.streams.nodes = {5: stream_node}

        # GW node 2 has valid coords in the grid
        # The grid from _make_grid has node 2 at (100.0, 0.0)
        with patch.object(state, "reproject_coords", side_effect=lambda x, y: (x, y)):
            result = state.get_hydrograph_locations()

        assert len(result["stream"]) == 1
        assert result["stream"][0]["lng"] == 100.0
        assert result["stream"][0]["lat"] == 0.0

    def test_stream_hydrograph_locations_skip_zero_coords(self):
        """When stream node and GW fallback both have (0,0), skip the node."""
        state, model = _state_with_model(with_streams=True)
        model.metadata["stream_hydrograph_specs"] = [
            {"node_id": 99, "name": "Ghost"},
        ]

        stream_node = MagicMock()
        stream_node.x = 0.0
        stream_node.y = 0.0
        stream_node.gw_node = None  # No GW node to fall back to
        model.streams.nodes = {99: stream_node}

        result = state.get_hydrograph_locations()
        assert len(result["stream"]) == 0

    def test_stream_hydrograph_locations_skip_missing_node(self):
        """When stream node ID is not found in streams.nodes, skip it."""
        state, model = _state_with_model(with_streams=True)
        model.metadata["stream_hydrograph_specs"] = [
            {"node_id": 999, "name": "Missing"},
        ]
        model.streams.nodes = {}  # Empty dict

        result = state.get_hydrograph_locations()
        assert len(result["stream"]) == 0

    def test_stream_hydrograph_default_name(self):
        """When spec has no 'name' key, uses default f-string."""
        state, model = _state_with_model(with_streams=True)
        model.metadata["stream_hydrograph_specs"] = [
            {"node_id": 7},  # No 'name' key
        ]

        stream_node = MagicMock()
        stream_node.x = 50.0
        stream_node.y = 75.0
        stream_node.gw_node = None
        stream_node.reach_id = 2
        model.streams.nodes = {7: stream_node}

        with patch.object(state, "reproject_coords", side_effect=lambda x, y: (x, y)):
            result = state.get_hydrograph_locations()

        assert len(result["stream"]) == 1
        assert result["stream"][0]["name"] == "Stream Node 7"


class TestBudgetReaderRelativePathAndNotFound:
    """Cover lines 605 and 607-608: budget reader relative path + file not found."""

    def test_budget_reader_relative_path_not_found(self, tmp_path):
        """When budget file relative path resolves but does not exist."""
        state, model = _state_with_model()
        state._results_dir = tmp_path
        # Relative path that does NOT exist on disk
        model.metadata["gw_budget_file"] = "missing_budget.hdf5"

        result = state.get_budget_reader("gw")
        assert result is None

    def test_budget_reader_relative_path_resolved_and_found(self, tmp_path):
        """Line 605: relative path resolved via _results_dir."""
        state, model = _state_with_model()
        state._results_dir = tmp_path
        budget_file = tmp_path / "gw_budget.hdf5"
        budget_file.write_bytes(b"fake")
        model.metadata["gw_budget_file"] = "gw_budget.hdf5"

        mock_reader = MagicMock()
        mock_reader.descriptor = "GROUNDWATER BUDGET"
        with patch(
            "pyiwfm.io.budget.BudgetReader",
            return_value=mock_reader,
        ):
            result = state.get_budget_reader("gw")

        assert result is mock_reader


class TestResultsInfoHeadTimeRange:
    """Cover lines 672->678: results_info head_time_range when times is truthy."""

    def test_results_info_with_head_times_populates_time_range(self):
        """When loader has frames > 0 and times is non-empty, head_time_range is set."""
        state, model = _state_with_model(with_groundwater=True)

        mock_loader = MagicMock()
        mock_loader.n_frames = 12
        mock_loader.times = [datetime(2010, 1, 1), datetime(2010, 12, 31)]

        mock_gw_reader = MagicMock()
        mock_gw_reader.n_timesteps = 100

        with (
            patch.object(state, "get_head_loader", return_value=mock_loader),
            patch.object(state, "get_available_budgets", return_value=["gw"]),
            patch.object(state, "get_gw_hydrograph_reader", return_value=mock_gw_reader),
            patch.object(state, "get_stream_hydrograph_reader", return_value=None),
        ):
            info = state.get_results_info()

        assert info["has_results"] is True
        assert info["n_head_timesteps"] == 12
        assert info["head_time_range"] is not None
        assert info["head_time_range"]["start"] == "2010-01-01T00:00:00"
        assert info["head_time_range"]["end"] == "2010-12-31T00:00:00"

    def test_results_info_with_head_no_times_no_range(self):
        """When loader.n_frames > 0 but times is empty list, head_time_range stays None."""
        state, model = _state_with_model()

        mock_loader = MagicMock()
        mock_loader.n_frames = 5
        mock_loader.times = []  # Empty - truthy n_frames but falsy times

        with (
            patch.object(state, "get_head_loader", return_value=mock_loader),
            patch.object(state, "get_available_budgets", return_value=[]),
            patch.object(state, "get_gw_hydrograph_reader", return_value=None),
            patch.object(state, "get_stream_hydrograph_reader", return_value=None),
        ):
            info = state.get_results_info()

        assert info["has_results"] is True
        assert info["n_head_timesteps"] == 5
        assert info["head_time_range"] is None


class TestGeoJSONNoModel:
    """Cover the edge case where model is set but grid is None."""

    def test_compute_mesh_geojson_model_none(self):
        """_compute_mesh_geojson returns empty collection when _model is None."""
        state = ModelState()
        result = state._compute_mesh_geojson(layer=1)
        assert result == {"type": "FeatureCollection", "features": []}


class TestGetSliceJsonFullCoverage:
    """Additional slice tests for full method coverage."""

    def test_get_slice_json_angle_0_north_south(self):
        """Verify normal calculation at angle=0 (N-S face, normal points east)."""
        state, _ = _state_with_model()

        mock_pv = MagicMock()
        mock_slice = MagicMock()
        mock_slice.n_cells = 1
        mock_slice.n_points = 3
        mock_slice.points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        mock_slice.faces = np.array([3, 0, 1, 2])
        mock_slice.cell_data = {"layer": np.array([1])}

        with (
            patch.object(state, "get_pyvista_3d", return_value=mock_pv),
            patch("pyiwfm.visualization.webapi.slicing.SlicingController") as mock_slicer_cls,
        ):
            mock_slicer = mock_slicer_cls.return_value
            mock_slicer.normalized_to_position_along.return_value = (50.0, 0.0, 0.0)
            mock_slicer.slice_arbitrary.return_value = mock_slice

            result = state.get_slice_json(angle=0.0, position=0.5)

        assert result["n_points"] == 3
        assert result["n_cells"] == 1
        # Verify the normal passed to slice_arbitrary is approximately (1, 0, 0)
        call_args = mock_slicer.slice_arbitrary.call_args
        normal_arg = call_args[1]["normal"] if "normal" in call_args[1] else call_args[0][0]
        assert abs(normal_arg[0] - 1.0) < 1e-10
        assert abs(normal_arg[1] - 0.0) < 1e-10
