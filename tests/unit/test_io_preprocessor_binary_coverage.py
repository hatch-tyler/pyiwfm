"""Tests for pyiwfm.io.preprocessor_binary module.

Covers the 11 dataclasses, PreprocessorBinaryReader with mocked
FortranBinaryReader, and the convenience function read_preprocessor_binary.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pyiwfm.io.preprocessor_binary import (
    AppElementData,
    AppNodeData,
    PreprocessorBinaryData,
    PreprocessorBinaryReader,
    StratigraphyData,
    read_preprocessor_binary,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_binary_reader() -> MagicMock:
    """Create a MagicMock that behaves as a FortranBinaryReader context manager."""
    reader = MagicMock()
    reader.__enter__ = MagicMock(return_value=reader)
    reader.__exit__ = MagicMock(return_value=False)
    return reader


# ---------------------------------------------------------------------------
# Dataclass construction tests
# ---------------------------------------------------------------------------


class TestAppNodeDataCreation:
    """Test AppNodeData dataclass creation with valid fields."""

    def test_create_with_valid_fields(self) -> None:
        node = AppNodeData(
            id=1,
            area=100.0,
            boundary_node=True,
            n_connected_node=3,
            n_face_id=2,
            surrounding_elements=np.array([1, 2, 3], dtype=np.int32),
            connected_nodes=np.array([10, 20, 30], dtype=np.int32),
            face_ids=np.array([5, 6], dtype=np.int32),
            elem_id_on_ccw_side=np.array([1, 2], dtype=np.int32),
            irrotational_coeff=np.array([0.5, 0.7], dtype=np.float64),
        )
        assert node.id == 1
        assert node.area == 100.0
        assert node.boundary_node is True
        assert node.n_connected_node == 3
        assert node.n_face_id == 2
        np.testing.assert_array_equal(node.surrounding_elements, [1, 2, 3])
        np.testing.assert_array_equal(node.connected_nodes, [10, 20, 30])
        np.testing.assert_array_equal(node.face_ids, [5, 6])
        np.testing.assert_array_equal(node.elem_id_on_ccw_side, [1, 2])
        np.testing.assert_allclose(node.irrotational_coeff, [0.5, 0.7])


class TestAppElementDataCreation:
    """Test AppElementData dataclass creation with valid fields."""

    def test_create_with_valid_fields(self) -> None:
        elem = AppElementData(
            id=42,
            subregion=2,
            area=500.0,
            face_ids=np.array([10, 11, 12], dtype=np.int32),
            vertex_areas=np.array([125.0, 125.0, 125.0, 125.0], dtype=np.float64),
            vertex_area_fractions=np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64),
            integral_del_shp_i_del_shp_j=np.ones(9, dtype=np.float64),
            integral_rot_del_shp_i_del_shp_j=np.zeros(9, dtype=np.float64),
        )
        assert elem.id == 42
        assert elem.subregion == 2
        assert elem.area == 500.0
        np.testing.assert_array_equal(elem.face_ids, [10, 11, 12])
        assert elem.vertex_areas.shape == (4,)
        assert elem.integral_del_shp_i_del_shp_j.shape == (9,)


class TestPreprocessorBinaryDataDefaults:
    """Test PreprocessorBinaryData construction with default values."""

    def test_defaults(self) -> None:
        data = PreprocessorBinaryData()
        assert data.n_nodes == 0
        assert data.n_elements == 0
        assert data.n_faces == 0
        assert data.n_subregions == 0
        assert data.n_boundary_faces == 0
        assert data.x.shape == (0,)
        assert data.y.shape == (0,)
        assert data.app_nodes == []
        assert data.app_elements == []
        assert data.app_faces is None
        assert data.stratigraphy is None
        assert data.stream_lake_connector is None
        assert data.stream_gw_connector is None
        assert data.lake_gw_connector is None
        assert data.lakes is None
        assert data.streams is None
        assert data.matrix_n_equations == 0
        assert data.matrix_connectivity.shape == (0,)


class TestStratigraphyDataCreation:
    """Test StratigraphyData dataclass creation with numpy arrays."""

    def test_create_with_numpy_arrays(self) -> None:
        n_nodes, n_layers = 4, 2
        strat = StratigraphyData(
            n_layers=n_layers,
            ground_surface_elev=np.array([100.0, 200.0, 150.0, 120.0]),
            top_elev=np.ones((n_nodes, n_layers), dtype=np.float64) * 90.0,
            bottom_elev=np.ones((n_nodes, n_layers), dtype=np.float64) * 10.0,
            active_node=np.ones((n_nodes, n_layers), dtype=np.bool_),
            active_layer_above=np.full((n_nodes, n_layers), -1, dtype=np.int32),
            active_layer_below=np.full((n_nodes, n_layers), -1, dtype=np.int32),
            top_active_layer=np.ones(n_nodes, dtype=np.int32),
            bottom_active_layer=np.full(n_nodes, n_layers, dtype=np.int32),
        )
        assert strat.n_layers == 2
        assert strat.ground_surface_elev.shape == (4,)
        assert strat.top_elev.shape == (4, 2)
        assert strat.bottom_elev.shape == (4, 2)
        assert strat.active_node.dtype == np.bool_
        assert strat.top_active_layer.shape == (4,)


# ---------------------------------------------------------------------------
# PreprocessorBinaryReader tests
# ---------------------------------------------------------------------------


class TestReadFullFile:
    """Test PreprocessorBinaryReader.read() orchestration with a mocked binary reader."""

    @patch("pyiwfm.io.preprocessor_binary.FortranBinaryReader")
    def test_read_orchestrates_all_sections(self, mock_fbr_cls: MagicMock) -> None:
        """read() should call all _read_*() helpers and return populated data."""
        f = _make_mock_binary_reader()
        mock_fbr_cls.return_value = f

        n_nodes = 2
        n_elements = 1
        n_faces = 0
        n_subregions = 0
        n_boundary_faces = 0
        n_layers = 1

        # Sequences of return values for read_int, read_double, etc.
        # Grid dimensions (5 ints), then per-node reads, per-element reads, ...
        int_values = iter(
            [
                # _read_grid_data: dimensions
                n_nodes,
                n_elements,
                n_faces,
                n_subregions,
                n_boundary_faces,
                # _read_app_node (node 1): id, boundary, n_connected, n_face_id, n_surround, n_connected_arr
                1,
                0,
                2,
                0,
                0,
                0,
                # _read_app_node (node 2):
                2,
                1,
                1,
                0,
                0,
                0,
                # _read_app_element (elem 1): id, subregion, n_faces, n_vert_area, n_del_shp, n_rot_shp
                1,
                1,
                0,
                0,
                0,
                0,
                # _read_stratigraphy: n_layers
                n_layers,
                # _read_stream_lake_connector: n_connections=0
                0,
                # _read_stream_gw_connector: n_stream_nodes=0
                0,
                # _read_lake_gw_connector: n_lakes=0
                0,
                # _read_lake_data: n_lakes=0
                0,
                # _read_stream_data: n_reaches=0, n_stream_nodes=0
                0,
                0,
                # _read_matrix_data: matrix_n_equations=0
                0,
            ]
        )
        f.read_int.side_effect = lambda: next(int_values)

        double_values = iter(
            [
                # _read_app_node (node 1): area
                50.0,
                # _read_app_node (node 2): area
                75.0,
                # _read_app_element (elem 1): area
                200.0,
            ]
        )
        f.read_double.side_effect = lambda: next(double_values)

        # Arrays: x, y, n_vertex, vertex for grid; then stratigraphy arrays
        array_calls = iter(
            [
                # _read_grid_data: x
                np.array([0.0, 1.0]),
                # _read_grid_data: y
                np.array([0.0, 1.0]),
                # _read_grid_data: n_vertex
                np.array([3], dtype=np.int32),
                # _read_grid_data: vertex
                np.array([1, 2, 1], dtype=np.int32),
                # _read_stratigraphy: gs_elev
                np.array([100.0, 110.0]),
                # top_flat (n_nodes * n_layers = 2)
                np.array([90.0, 95.0]),
                # bottom_flat
                np.array([10.0, 15.0]),
                # active_flat
                np.array([1, 1], dtype=np.int32),
                # active_above_flat
                np.array([-1, -1], dtype=np.int32),
                # active_below_flat
                np.array([-1, -1], dtype=np.int32),
                # top_active
                np.array([1, 1], dtype=np.int32),
                # bottom_active
                np.array([1, 1], dtype=np.int32),
            ]
        )

        def _double_or_int_array() -> np.ndarray:
            return next(array_calls)

        f.read_double_array.side_effect = _double_or_int_array
        f.read_int_array.side_effect = _double_or_int_array

        # Create a temporary file so the Path.exists() check passes
        reader = PreprocessorBinaryReader(endian="<")
        tmp = Path("__fake_preproc.bin")
        with patch.object(Path, "exists", return_value=True):
            data = reader.read(tmp)

        assert data.n_nodes == 2
        assert data.n_elements == 1
        assert data.n_faces == 0
        assert len(data.app_nodes) == 2
        assert len(data.app_elements) == 1
        assert data.stratigraphy is not None
        assert data.stratigraphy.n_layers == 1
        assert data.stream_lake_connector is not None
        assert data.stream_lake_connector.n_connections == 0
        assert data.stream_gw_connector is not None
        assert data.stream_gw_connector.n_stream_nodes == 0
        assert data.lake_gw_connector is not None
        assert data.lake_gw_connector.n_lakes == 0
        assert data.lakes is not None
        assert data.lakes.n_lakes == 0
        assert data.streams is not None
        assert data.streams.n_reaches == 0


class TestReadGridData:
    """Test _read_grid_data reads dimensions and coordinates."""

    def test_reads_dimensions_and_coordinates(self) -> None:
        f = _make_mock_binary_reader()

        int_values = iter(
            [
                # dimensions
                3,
                2,
                0,
                0,
                0,
                # node 1: id, boundary, n_connected, n_face_id, n_surround, n_connected_arr
                1,
                0,
                2,
                0,
                0,
                0,
                # node 2
                2,
                0,
                2,
                0,
                0,
                0,
                # node 3
                3,
                1,
                1,
                0,
                0,
                0,
                # elem 1: id, subregion, n_faces, n_vert_area, n_del_shp, n_rot_shp
                1,
                1,
                0,
                0,
                0,
                0,
                # elem 2
                2,
                1,
                0,
                0,
                0,
                0,
            ]
        )
        f.read_int.side_effect = lambda: next(int_values)

        double_values = iter(
            [
                # node areas (3 nodes)
                10.0,
                20.0,
                30.0,
                # element areas (2 elems)
                100.0,
                200.0,
            ]
        )
        f.read_double.side_effect = lambda: next(double_values)

        array_calls = iter(
            [
                np.array([0.0, 1.0, 2.0]),  # x
                np.array([0.0, 1.0, 2.0]),  # y
                np.array([3, 4], dtype=np.int32),  # n_vertex
                np.array([1, 2, 3, 1, 2, 3, 4], dtype=np.int32),  # vertex
            ]
        )
        f.read_double_array.side_effect = lambda: next(array_calls)
        f.read_int_array.side_effect = lambda: next(array_calls)

        reader = PreprocessorBinaryReader()
        data = PreprocessorBinaryData()
        reader._read_grid_data(f, data)

        assert data.n_nodes == 3
        assert data.n_elements == 2
        assert len(data.app_nodes) == 3
        assert len(data.app_elements) == 2
        np.testing.assert_array_equal(data.x, [0.0, 1.0, 2.0])


class TestReadAppNode:
    """Test _read_app_node reads a single node record."""

    def test_reads_node_with_connections(self) -> None:
        f = _make_mock_binary_reader()

        # id=5, boundary=1(True), n_connected=3, n_face_id=2, n_surround=4, n_connected_arr=3
        int_values = iter([5, 1, 3, 2, 4, 3])
        f.read_int.side_effect = lambda: next(int_values)
        f.read_double.return_value = 250.0

        arr_calls = iter(
            [
                np.array([1, 2, 3, 4], dtype=np.int32),  # surrounding
                np.array([10, 20, 30], dtype=np.int32),  # connected
                np.array([100, 200], dtype=np.int32),  # face_ids
                np.array([1, 2], dtype=np.int32),  # elem_ccw
                np.array([0.1, 0.2], dtype=np.float64),  # irrot_coeff
            ]
        )
        f.read_int_array.side_effect = lambda: next(arr_calls)
        f.read_double_array.side_effect = lambda: next(arr_calls)

        reader = PreprocessorBinaryReader()
        node = reader._read_app_node(f)

        assert node.id == 5
        assert node.area == 250.0
        assert node.boundary_node is True
        assert node.n_connected_node == 3
        assert node.n_face_id == 2
        np.testing.assert_array_equal(node.surrounding_elements, [1, 2, 3, 4])
        np.testing.assert_array_equal(node.connected_nodes, [10, 20, 30])
        np.testing.assert_array_equal(node.face_ids, [100, 200])
        np.testing.assert_allclose(node.irrotational_coeff, [0.1, 0.2])

    def test_reads_node_no_connections(self) -> None:
        """Node with n_face_id=0, n_surround=0, n_connected_arr=0 yields empty arrays."""
        f = _make_mock_binary_reader()

        int_values = iter([99, 0, 0, 0, 0, 0])
        f.read_int.side_effect = lambda: next(int_values)
        f.read_double.return_value = 0.0

        reader = PreprocessorBinaryReader()
        node = reader._read_app_node(f)

        assert node.id == 99
        assert node.boundary_node is False
        assert node.surrounding_elements.size == 0
        assert node.connected_nodes.size == 0
        assert node.face_ids.size == 0
        assert node.irrotational_coeff.size == 0


class TestReadAppElement:
    """Test _read_app_element reads a single element record."""

    def test_reads_element_with_data(self) -> None:
        f = _make_mock_binary_reader()

        int_values = iter([7, 2, 3, 4, 9, 9])
        f.read_int.side_effect = lambda: next(int_values)
        f.read_double.return_value = 1500.0

        arr_calls = iter(
            [
                np.array([1, 2, 3], dtype=np.int32),  # face_ids
                np.array([375.0, 375.0, 375.0, 375.0], dtype=np.float64),  # vert_areas
                np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64),  # vert_fracs
                np.ones(9, dtype=np.float64),  # del_shp
                np.zeros(9, dtype=np.float64),  # rot_shp
            ]
        )
        f.read_int_array.side_effect = lambda: next(arr_calls)
        f.read_double_array.side_effect = lambda: next(arr_calls)

        reader = PreprocessorBinaryReader()
        elem = reader._read_app_element(f)

        assert elem.id == 7
        assert elem.subregion == 2
        assert elem.area == 1500.0
        np.testing.assert_array_equal(elem.face_ids, [1, 2, 3])
        assert elem.vertex_areas.shape == (4,)
        assert elem.integral_del_shp_i_del_shp_j.shape == (9,)
        assert elem.integral_rot_del_shp_i_del_shp_j.shape == (9,)


class TestReadAppFaces:
    """Test _read_app_faces with zero and non-zero face counts."""

    def test_zero_faces_returns_empty(self) -> None:
        f = _make_mock_binary_reader()
        reader = PreprocessorBinaryReader()
        face_data = reader._read_app_faces(f, n_faces=0)

        assert face_data.nodes.shape == (0, 2)
        assert face_data.elements.shape == (0, 2)
        assert face_data.boundary.size == 0
        assert face_data.lengths.size == 0
        # No reads should have been performed
        f.read_int_array.assert_not_called()
        f.read_double_array.assert_not_called()

    def test_nonzero_faces_reads_all_arrays(self) -> None:
        f = _make_mock_binary_reader()

        arr_calls = iter(
            [
                np.array([1, 2, 3, 4], dtype=np.int32),  # nodes_flat (2 faces x 2)
                np.array([10, 0, 20, 30], dtype=np.int32),  # elements_flat (2 faces x 2)
                np.array([1, 0], dtype=np.int32),  # boundary_int
                np.array([5.5, 8.3], dtype=np.float64),  # lengths
            ]
        )
        f.read_int_array.side_effect = lambda: next(arr_calls)
        f.read_double_array.side_effect = lambda: next(arr_calls)

        reader = PreprocessorBinaryReader()
        face_data = reader._read_app_faces(f, n_faces=2)

        assert face_data.nodes.shape == (2, 2)
        assert face_data.elements.shape == (2, 2)
        np.testing.assert_array_equal(face_data.nodes[0], [1, 2])
        np.testing.assert_array_equal(face_data.nodes[1], [3, 4])
        assert face_data.boundary[0] is np.bool_(True)
        assert face_data.boundary[1] is np.bool_(False)
        np.testing.assert_allclose(face_data.lengths, [5.5, 8.3])


class TestReadSubregion:
    """Test _read_subregion reads a subregion record."""

    def test_reads_subregion_with_neighbors(self) -> None:
        f = _make_mock_binary_reader()

        int_values = iter(
            [
                1,  # sub_id
                3,  # n_elements
                2,  # n_neighbors
            ]
        )
        f.read_int.side_effect = lambda: next(int_values)
        f.read_string.return_value = "Region_A"
        f.read_double.return_value = 999.0

        arr_calls = iter(
            [
                np.array([10, 20, 30], dtype=np.int32),  # elements
                np.array([2, 3], dtype=np.int32),  # neighbor_ids
                np.array([1, 2], dtype=np.int32),  # neighbor_n_faces
                np.array([55], dtype=np.int32),  # neighbor 1 boundary faces (1 face)
                np.array([66, 77], dtype=np.int32),  # neighbor 2 boundary faces (2 faces)
            ]
        )
        f.read_int_array.side_effect = lambda: next(arr_calls)

        reader = PreprocessorBinaryReader()
        sub = reader._read_subregion(f)

        assert sub.id == 1
        assert sub.name == "Region_A"
        assert sub.n_elements == 3
        assert sub.n_neighbor_regions == 2
        assert sub.area == 999.0
        np.testing.assert_array_equal(sub.region_elements, [10, 20, 30])
        assert len(sub.neighbor_boundary_faces) == 2
        np.testing.assert_array_equal(sub.neighbor_boundary_faces[0], [55])
        np.testing.assert_array_equal(sub.neighbor_boundary_faces[1], [66, 77])


class TestReadStratigraphy:
    """Test _read_stratigraphy reads layer data and reshapes correctly."""

    def test_reads_and_reshapes(self) -> None:
        f = _make_mock_binary_reader()
        n_nodes, n_layers = 3, 2

        f.read_int.return_value = n_layers

        arr_calls = iter(
            [
                # gs_elev
                np.array([100.0, 110.0, 120.0]),
                # top_flat (n_nodes * n_layers = 6)
                np.array([90.0, 80.0, 95.0, 85.0, 100.0, 90.0]),
                # bottom_flat
                np.array([50.0, 40.0, 55.0, 45.0, 60.0, 50.0]),
                # active_flat
                np.array([1, 1, 1, 0, 1, 1], dtype=np.int32),
                # active_above_flat
                np.array([-1, -1, -1, 1, -1, -1], dtype=np.int32),
                # active_below_flat
                np.array([2, -1, 2, -1, 2, -1], dtype=np.int32),
                # top_active
                np.array([1, 1, 1], dtype=np.int32),
                # bottom_active
                np.array([2, 1, 2], dtype=np.int32),
            ]
        )
        f.read_double_array.side_effect = lambda: next(arr_calls)
        f.read_int_array.side_effect = lambda: next(arr_calls)

        reader = PreprocessorBinaryReader()
        data = PreprocessorBinaryData()
        data.n_nodes = n_nodes
        reader._read_stratigraphy(f, data)

        strat = data.stratigraphy
        assert strat is not None
        assert strat.n_layers == 2
        assert strat.ground_surface_elev.shape == (3,)
        assert strat.top_elev.shape == (3, 2)
        assert strat.bottom_elev.shape == (3, 2)
        assert strat.active_node.dtype == np.bool_
        assert strat.active_node.shape == (3, 2)
        # Check a specific reshape: first node, second layer
        assert strat.top_elev[0, 1] == 80.0
        assert strat.active_node[1, 1] is np.bool_(False)


class TestReadStreamGWConnector:
    """Test _read_stream_gw_connector with zero and nonzero stream nodes."""

    def test_zero_stream_nodes(self) -> None:
        f = _make_mock_binary_reader()
        f.read_int.return_value = 0

        reader = PreprocessorBinaryReader()
        data = PreprocessorBinaryData()
        reader._read_stream_gw_connector(f, data)

        conn = data.stream_gw_connector
        assert conn is not None
        assert conn.n_stream_nodes == 0
        assert conn.gw_nodes.size == 0
        assert conn.layers.size == 0

    def test_with_data(self) -> None:
        f = _make_mock_binary_reader()
        f.read_int.return_value = 3

        arr_calls = iter(
            [
                np.array([10, 20, 30], dtype=np.int32),  # gw_nodes
                np.array([1, 1, 2], dtype=np.int32),  # layers
            ]
        )
        f.read_int_array.side_effect = lambda: next(arr_calls)

        reader = PreprocessorBinaryReader()
        data = PreprocessorBinaryData()
        reader._read_stream_gw_connector(f, data)

        conn = data.stream_gw_connector
        assert conn is not None
        assert conn.n_stream_nodes == 3
        np.testing.assert_array_equal(conn.gw_nodes, [10, 20, 30])
        np.testing.assert_array_equal(conn.layers, [1, 1, 2])


class TestReadLakeGWConnector:
    """Test _read_lake_gw_connector with zero and nonzero lakes."""

    def test_zero_lakes(self) -> None:
        f = _make_mock_binary_reader()
        f.read_int.return_value = 0

        reader = PreprocessorBinaryReader()
        data = PreprocessorBinaryData()
        reader._read_lake_gw_connector(f, data)

        conn = data.lake_gw_connector
        assert conn is not None
        assert conn.n_lakes == 0
        assert conn.lake_elements == []
        assert conn.lake_nodes == []

    def test_with_data_reads_per_lake(self) -> None:
        f = _make_mock_binary_reader()

        # n_lakes=2, then for each lake: n_elems, elements, n_nodes, nodes
        int_values = iter(
            [
                2,  # n_lakes
                3,  # lake 1 n_elems
                2,  # lake 1 n_nodes
                0,  # lake 2 n_elems
                1,  # lake 2 n_nodes
            ]
        )
        f.read_int.side_effect = lambda: next(int_values)

        arr_calls = iter(
            [
                np.array([1, 2, 3], dtype=np.int32),  # lake 1 elements
                np.array([10, 20], dtype=np.int32),  # lake 1 nodes
                # lake 2 elements: n_elems=0, so no read_int_array
                np.array([30], dtype=np.int32),  # lake 2 nodes
            ]
        )
        f.read_int_array.side_effect = lambda: next(arr_calls)

        reader = PreprocessorBinaryReader()
        data = PreprocessorBinaryData()
        reader._read_lake_gw_connector(f, data)

        conn = data.lake_gw_connector
        assert conn is not None
        assert conn.n_lakes == 2
        assert len(conn.lake_elements) == 2
        np.testing.assert_array_equal(conn.lake_elements[0], [1, 2, 3])
        assert conn.lake_elements[1].size == 0  # n_elems was 0
        np.testing.assert_array_equal(conn.lake_nodes[0], [10, 20])
        np.testing.assert_array_equal(conn.lake_nodes[1], [30])


class TestReadStreamData:
    """Test _read_stream_data with zero and nonzero reaches."""

    def test_zero_reaches(self) -> None:
        f = _make_mock_binary_reader()
        int_values = iter([0, 0])  # n_reaches=0, n_stream_nodes=0
        f.read_int.side_effect = lambda: next(int_values)

        reader = PreprocessorBinaryReader()
        data = PreprocessorBinaryData()
        reader._read_stream_data(f, data)

        streams = data.streams
        assert streams is not None
        assert streams.n_reaches == 0
        assert streams.reach_ids.size == 0
        assert streams.reach_names == []

    def test_with_reaches_reads_names(self) -> None:
        f = _make_mock_binary_reader()
        int_values = iter([2, 10])  # n_reaches=2, n_stream_nodes=10
        f.read_int.side_effect = lambda: next(int_values)

        arr_calls = iter(
            [
                np.array([1, 2], dtype=np.int32),  # reach_ids
                np.array([1, 5], dtype=np.int32),  # upstream_nodes
                np.array([4, 10], dtype=np.int32),  # downstream_nodes
                np.array([2, 0], dtype=np.int32),  # outflow_dest
            ]
        )
        f.read_int_array.side_effect = lambda: next(arr_calls)

        name_calls = iter(["Sacramento", "San_Joaquin"])
        f.read_string.side_effect = lambda: next(name_calls)

        reader = PreprocessorBinaryReader()
        data = PreprocessorBinaryData()
        reader._read_stream_data(f, data)

        streams = data.streams
        assert streams is not None
        assert streams.n_reaches == 2
        assert streams.n_stream_nodes == 10
        np.testing.assert_array_equal(streams.reach_ids, [1, 2])
        assert streams.reach_names == ["Sacramento", "San_Joaquin"]
        np.testing.assert_array_equal(streams.reach_upstream_nodes, [1, 5])
        np.testing.assert_array_equal(streams.reach_downstream_nodes, [4, 10])
        np.testing.assert_array_equal(streams.reach_outflow_dest, [2, 0])


class TestReadLakeData:
    """Test _read_lake_data with zero and nonzero lakes."""

    def test_zero_lakes(self) -> None:
        f = _make_mock_binary_reader()
        f.read_int.return_value = 0

        reader = PreprocessorBinaryReader()
        data = PreprocessorBinaryData()
        reader._read_lake_data(f, data)

        lakes = data.lakes
        assert lakes is not None
        assert lakes.n_lakes == 0
        assert lakes.lake_ids.size == 0
        assert lakes.lake_names == []
        assert lakes.lake_max_elevations.size == 0

    def test_with_lakes_reads_details(self) -> None:
        f = _make_mock_binary_reader()

        int_values = iter(
            [
                2,  # n_lakes
                3,  # lake 1 n_elems
                2,  # lake 2 n_elems
            ]
        )
        f.read_int.side_effect = lambda: next(int_values)

        arr_calls = iter(
            [
                np.array([1, 2], dtype=np.int32),  # lake_ids
                np.array([10, 11, 12], dtype=np.int32),  # lake 1 elements
                np.array([20, 21], dtype=np.int32),  # lake 2 elements
            ]
        )
        f.read_int_array.side_effect = lambda: next(arr_calls)

        name_calls = iter(["Clear_Lake", "Tulare_Lake"])
        f.read_string.side_effect = lambda: next(name_calls)

        double_calls = iter([150.0, 200.0])
        f.read_double.side_effect = lambda: next(double_calls)

        reader = PreprocessorBinaryReader()
        data = PreprocessorBinaryData()
        reader._read_lake_data(f, data)

        lakes = data.lakes
        assert lakes is not None
        assert lakes.n_lakes == 2
        np.testing.assert_array_equal(lakes.lake_ids, [1, 2])
        assert lakes.lake_names == ["Clear_Lake", "Tulare_Lake"]
        np.testing.assert_allclose(lakes.lake_max_elevations, [150.0, 200.0])
        assert len(lakes.lake_elements) == 2
        np.testing.assert_array_equal(lakes.lake_elements[0], [10, 11, 12])
        np.testing.assert_array_equal(lakes.lake_elements[1], [20, 21])


class TestReadMatrixData:
    """Test _read_matrix_data including EOFError handling."""

    def test_reads_matrix_with_equations(self) -> None:
        f = _make_mock_binary_reader()
        f.read_int.return_value = 5
        f.read_int_array.return_value = np.array([1, 2, 3, 4, 5], dtype=np.int32)

        reader = PreprocessorBinaryReader()
        data = PreprocessorBinaryData()
        reader._read_matrix_data(f, data)

        assert data.matrix_n_equations == 5
        np.testing.assert_array_equal(data.matrix_connectivity, [1, 2, 3, 4, 5])

    def test_handles_eof_gracefully(self) -> None:
        f = _make_mock_binary_reader()
        f.read_int.side_effect = EOFError("End of file")

        reader = PreprocessorBinaryReader()
        data = PreprocessorBinaryData()
        reader._read_matrix_data(f, data)

        # Should swallow the EOFError and leave defaults
        assert data.matrix_n_equations == 0
        assert data.matrix_connectivity.shape == (0,)

    def test_zero_equations_skips_array_read(self) -> None:
        f = _make_mock_binary_reader()
        f.read_int.return_value = 0

        reader = PreprocessorBinaryReader()
        data = PreprocessorBinaryData()
        reader._read_matrix_data(f, data)

        assert data.matrix_n_equations == 0
        f.read_int_array.assert_not_called()


class TestReadStreamLakeConnector:
    """Test _read_stream_lake_connector with zero and nonzero connections."""

    def test_zero_connections(self) -> None:
        f = _make_mock_binary_reader()
        f.read_int.return_value = 0

        reader = PreprocessorBinaryReader()
        data = PreprocessorBinaryData()
        reader._read_stream_lake_connector(f, data)

        conn = data.stream_lake_connector
        assert conn is not None
        assert conn.n_connections == 0
        assert conn.stream_nodes.size == 0
        assert conn.lake_ids.size == 0

    def test_with_connections(self) -> None:
        f = _make_mock_binary_reader()
        f.read_int.return_value = 2

        arr_calls = iter(
            [
                np.array([5, 10], dtype=np.int32),  # stream_nodes
                np.array([1, 2], dtype=np.int32),  # lake_ids
            ]
        )
        f.read_int_array.side_effect = lambda: next(arr_calls)

        reader = PreprocessorBinaryReader()
        data = PreprocessorBinaryData()
        reader._read_stream_lake_connector(f, data)

        conn = data.stream_lake_connector
        assert conn is not None
        assert conn.n_connections == 2
        np.testing.assert_array_equal(conn.stream_nodes, [5, 10])
        np.testing.assert_array_equal(conn.lake_ids, [1, 2])


class TestConvenienceFunction:
    """Test read_preprocessor_binary convenience function."""

    def test_file_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="Binary file not found"):
            read_preprocessor_binary(Path("nonexistent_file.bin"))

    @patch("pyiwfm.io.preprocessor_binary.FortranBinaryReader")
    def test_passes_endian_to_reader(self, mock_fbr_cls: MagicMock) -> None:
        """Verify endian parameter propagates through the convenience function."""
        f = _make_mock_binary_reader()
        mock_fbr_cls.return_value = f

        # Minimal mock: 0 nodes/elements so no loops
        int_values = iter(
            [
                0,
                0,
                0,
                0,
                0,  # grid dimensions all zero
                1,  # stratigraphy n_layers
                0,  # stream_lake n_connections
                0,  # stream_gw n_stream_nodes
                0,  # lake_gw n_lakes
                0,  # lake_data n_lakes
                0,
                0,  # stream_data n_reaches, n_stream_nodes
                0,  # matrix n_equations
            ]
        )
        f.read_int.side_effect = lambda: next(int_values)

        # Stratigraphy needs arrays for n_nodes=0: each should be empty
        empty_double = np.array([], dtype=np.float64)
        empty_int = np.array([], dtype=np.int32)
        arr_calls = iter(
            [
                empty_double,  # x
                empty_double,  # y
                empty_int,  # n_vertex
                empty_int,  # vertex
                empty_double,  # gs_elev
                empty_double,  # top_flat
                empty_double,  # bottom_flat
                empty_int,  # active_flat
                empty_int,  # active_above_flat
                empty_int,  # active_below_flat
                empty_int,  # top_active
                empty_int,  # bottom_active
            ]
        )
        f.read_double_array.side_effect = lambda: next(arr_calls)
        f.read_int_array.side_effect = lambda: next(arr_calls)

        with patch.object(Path, "exists", return_value=True):
            result = read_preprocessor_binary("fake.bin", endian=">")

        mock_fbr_cls.assert_called_once_with(Path("fake.bin"), ">")
        assert isinstance(result, PreprocessorBinaryData)


class TestReaderEndianDefault:
    """Test that PreprocessorBinaryReader has sensible defaults."""

    def test_default_endian_is_little(self) -> None:
        reader = PreprocessorBinaryReader()
        assert reader.endian == "<"

    def test_custom_endian(self) -> None:
        reader = PreprocessorBinaryReader(endian=">")
        assert reader.endian == ">"
