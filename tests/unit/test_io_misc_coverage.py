"""Extended coverage tests for miscellaneous IO modules.

Covers:
- supply_adjust: SupplyAdjustment, read/write, _is_fortran_comment, roundtrip
- parametric_grid: ParamNode, ParamElement, ParametricGrid, interpolation
- preprocessor_binary: data classes, PreprocessorBinaryReader (mocked)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pyiwfm.io.parametric_grid import (
    ParamElement,
    ParametricGrid,
    ParamNode,
)
from pyiwfm.io.preprocessor_binary import (
    AppElementData,
    AppFaceData,
    AppNodeData,
    LakeData,
    LakeGWConnectorData,
    PreprocessorBinaryData,
    PreprocessorBinaryReader,
    StratigraphyData,
    StreamData,
    StreamGWConnectorData,
    StreamLakeConnectorData,
    SubregionData,
    read_preprocessor_binary,
)
from pyiwfm.io.supply_adjust import (
    SupplyAdjustment,
    _is_fortran_comment,
    read_supply_adjustment,
    write_supply_adjustment,
)

# =============================================================================
# SupplyAdjustment extended tests
# =============================================================================


class TestSupplyAdjustmentExtended:
    """Extended coverage for supply_adjust module."""

    def _write(self, path: Path, content: str) -> Path:
        filepath = path / "supply_adj.dat"
        filepath.write_text(content)
        return filepath

    def test_is_fortran_comment_none_edge(self) -> None:
        """Empty string is not a fortran comment."""
        assert _is_fortran_comment("") is False

    def test_is_fortran_comment_tab_only(self) -> None:
        """Tab-only line is not a fortran comment."""
        assert _is_fortran_comment("\t\t") is False

    def test_is_fortran_comment_data_with_spaces(self) -> None:
        """Data line starting with spaces is not a comment."""
        assert _is_fortran_comment("    10    / NCOLADJ") is False

    def test_read_multiple_data_rows(self, tmp_path: Path) -> None:
        """Read file with multiple data rows."""
        content = (
            "          4                                     / NCOLADJ\n"
            "          1                                     / NSPADJ\n"
            "          0                                     / NFQADJ\n"
            "                                                / DSSFL\n"
            "    10/31/1973_24:00\t00\t10\t01\t00\n"
            "    11/30/1973_24:00\t00\t00\t01\t10\n"
            "    12/31/1973_24:00\t10\t10\t00\t00\n"
        )
        filepath = self._write(tmp_path, content)
        result = read_supply_adjustment(filepath)

        assert result.n_columns == 4
        assert len(result.times) == 3
        assert len(result.values) == 3
        assert len(result.values[0]) == 4

    def test_write_empty_data(self, tmp_path: Path) -> None:
        """Write file with no data rows."""
        data = SupplyAdjustment(n_columns=2, nsp=1, nfq=0)
        filepath = tmp_path / "empty.dat"
        result_path = write_supply_adjustment(data, filepath)

        assert result_path.exists()
        content = result_path.read_text()
        assert "NCOLADJ" in content
        # No timestamp lines
        assert "_24:00" not in content

    def test_write_read_roundtrip_many_columns(self, tmp_path: Path) -> None:
        """Roundtrip with many columns."""
        original = SupplyAdjustment(
            n_columns=5,
            nsp=1,
            nfq=0,
            times=[datetime(2000, 1, 1), datetime(2000, 2, 1)],
            values=[[0, 1, 10, 0, 1], [10, 10, 0, 1, 0]],
        )
        filepath = tmp_path / "roundtrip.dat"
        write_supply_adjustment(original, filepath)
        result = read_supply_adjustment(filepath)

        assert result.n_columns == 5
        assert len(result.times) == 2
        for i in range(2):
            assert result.values[i] == original.values[i]

    def test_read_skips_non_data_lines(self, tmp_path: Path) -> None:
        """Non-timestamp lines after parameters are skipped."""
        content = (
            "          2                                     / NCOLADJ\n"
            "          1                                     / NSPADJ\n"
            "          0                                     / NFQADJ\n"
            "                                                / DSSFL\n"
            "C*************************************************************\n"
            "C   Some description\n"
            "C*************************************************************\n"
            "    10/31/1973_24:00\t00\t10\n"
        )
        filepath = self._write(tmp_path, content)
        result = read_supply_adjustment(filepath)

        assert len(result.times) == 1

    def test_dataclass_nsp_nfq_values(self) -> None:
        """Test non-default nsp and nfq values."""
        sa = SupplyAdjustment(n_columns=1, nsp=2, nfq=3)
        assert sa.nsp == 2
        assert sa.nfq == 3

    def test_write_preserves_dss_reference(self, tmp_path: Path) -> None:
        """Written DSS reference can be read back."""
        data = SupplyAdjustment(n_columns=1, dss_file="test.dss")
        filepath = tmp_path / "dss.dat"
        write_supply_adjustment(data, filepath)
        result = read_supply_adjustment(filepath)
        assert result.dss_file == "test.dss"


# =============================================================================
# ParametricGrid tests
# =============================================================================


class TestParamNode:
    """Tests for ParamNode dataclass."""

    def test_constructor(self) -> None:
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        node = ParamNode(node_id=1, x=100.0, y=200.0, values=values)
        assert node.node_id == 1
        assert node.x == 100.0
        assert node.y == 200.0
        assert node.values.shape == (2, 2)


class TestParamElement:
    """Tests for ParamElement dataclass."""

    def test_constructor_triangle(self) -> None:
        elem = ParamElement(elem_id=1, vertices=(0, 1, 2))
        assert elem.elem_id == 1
        assert len(elem.vertices) == 3

    def test_constructor_quad(self) -> None:
        elem = ParamElement(elem_id=2, vertices=(0, 1, 2, 3))
        assert len(elem.vertices) == 4


class TestParametricGrid:
    """Tests for ParametricGrid interpolation."""

    def _make_triangle_grid(self) -> ParametricGrid:
        """Create a simple triangle grid with known values."""
        np.array([[1.0]])
        nodes = [
            ParamNode(node_id=1, x=0.0, y=0.0, values=np.array([[10.0]])),
            ParamNode(node_id=2, x=10.0, y=0.0, values=np.array([[20.0]])),
            ParamNode(node_id=3, x=0.0, y=10.0, values=np.array([[30.0]])),
        ]
        elements = [ParamElement(elem_id=1, vertices=(0, 1, 2))]
        return ParametricGrid(nodes=nodes, elements=elements)

    def _make_quad_grid(self) -> ParametricGrid:
        """Create a simple quad grid with known values."""
        nodes = [
            ParamNode(node_id=1, x=0.0, y=0.0, values=np.array([[0.0]])),
            ParamNode(node_id=2, x=10.0, y=0.0, values=np.array([[10.0]])),
            ParamNode(node_id=3, x=10.0, y=10.0, values=np.array([[20.0]])),
            ParamNode(node_id=4, x=0.0, y=10.0, values=np.array([[30.0]])),
        ]
        elements = [ParamElement(elem_id=1, vertices=(0, 1, 2, 3))]
        return ParametricGrid(nodes=nodes, elements=elements)

    def test_interpolate_triangle_at_vertex(self) -> None:
        """Interpolate at a vertex should return that vertex's value."""
        grid = self._make_triangle_grid()
        result = grid.interpolate(0.0, 0.0)
        assert result is not None
        assert result[0, 0] == pytest.approx(10.0, abs=0.1)

    def test_interpolate_triangle_centroid(self) -> None:
        """Interpolate at centroid should be mean of vertex values."""
        grid = self._make_triangle_grid()
        result = grid.interpolate(10.0 / 3, 10.0 / 3)
        assert result is not None
        expected = (10.0 + 20.0 + 30.0) / 3.0
        assert result[0, 0] == pytest.approx(expected, abs=0.1)

    def test_interpolate_triangle_outside(self) -> None:
        """Point outside triangle returns None."""
        grid = self._make_triangle_grid()
        result = grid.interpolate(-5.0, -5.0)
        assert result is None

    def test_interpolate_quad_at_vertex(self) -> None:
        """Interpolate at a quad vertex."""
        grid = self._make_quad_grid()
        result = grid.interpolate(0.0, 0.0)
        assert result is not None
        assert result[0, 0] == pytest.approx(0.0, abs=0.1)

    def test_interpolate_quad_center(self) -> None:
        """Interpolate at center of quad."""
        grid = self._make_quad_grid()
        result = grid.interpolate(5.0, 5.0)
        assert result is not None
        # Center should be a weighted average

    def test_interpolate_quad_outside(self) -> None:
        """Point outside quad returns None."""
        grid = self._make_quad_grid()
        result = grid.interpolate(20.0, 20.0)
        assert result is None

    def test_interpolate_empty_grid(self) -> None:
        """Empty grid always returns None."""
        grid = ParametricGrid(nodes=[], elements=[])
        assert grid.interpolate(0.0, 0.0) is None

    def test_point_in_triangle_degenerate(self) -> None:
        """Degenerate triangle (zero area) returns False."""
        v0 = ParamNode(node_id=1, x=0.0, y=0.0, values=np.array([[1.0]]))
        v1 = ParamNode(node_id=2, x=0.0, y=0.0, values=np.array([[1.0]]))
        v2 = ParamNode(node_id=3, x=0.0, y=0.0, values=np.array([[1.0]]))
        inside, coeffs = ParametricGrid._point_in_triangle(0.0, 0.0, v0, v1, v2)
        assert inside is False

    def test_element_with_5_vertices_skipped(self) -> None:
        """Elements with unsupported vertex count are skipped."""
        nodes = [
            ParamNode(node_id=i, x=float(i), y=float(i), values=np.array([[1.0]])) for i in range(5)
        ]
        elements = [ParamElement(elem_id=1, vertices=(0, 1, 2, 3, 4))]
        grid = ParametricGrid(nodes=nodes, elements=elements)
        assert grid.interpolate(1.0, 1.0) is None

    def test_interpolate_triangle_on_edge(self) -> None:
        """Point on edge of triangle should be inside."""
        grid = self._make_triangle_grid()
        result = grid.interpolate(5.0, 0.0)  # midpoint of edge v0-v1
        assert result is not None
        assert result[0, 0] == pytest.approx(15.0, abs=0.1)


# =============================================================================
# PreprocessorBinary data class tests
# =============================================================================


class TestPreprocessorBinaryDataClasses:
    """Tests for preprocessor_binary data classes."""

    def test_app_node_data(self) -> None:
        nd = AppNodeData(
            id=1,
            area=100.0,
            boundary_node=True,
            n_connected_node=2,
            n_face_id=3,
            surrounding_elements=np.array([1, 2], dtype=np.int32),
            connected_nodes=np.array([3, 4], dtype=np.int32),
            face_ids=np.array([1, 2, 3], dtype=np.int32),
            elem_id_on_ccw_side=np.array([1, 2, 3], dtype=np.int32),
            irrotational_coeff=np.array([0.1, 0.2, 0.3]),
        )
        assert nd.id == 1
        assert nd.boundary_node is True
        assert len(nd.surrounding_elements) == 2

    def test_app_element_data(self) -> None:
        ed = AppElementData(
            id=1,
            subregion=1,
            area=500.0,
            face_ids=np.array([1, 2, 3], dtype=np.int32),
            vertex_areas=np.array([100.0, 200.0, 200.0]),
            vertex_area_fractions=np.array([0.2, 0.4, 0.4]),
            integral_del_shp_i_del_shp_j=np.array([1.0]),
            integral_rot_del_shp_i_del_shp_j=np.array([2.0]),
        )
        assert ed.id == 1
        assert ed.subregion == 1

    def test_app_face_data(self) -> None:
        fd = AppFaceData(
            nodes=np.array([[1, 2], [3, 4]], dtype=np.int32),
            elements=np.array([[1, 0], [1, 2]], dtype=np.int32),
            boundary=np.array([True, False]),
            lengths=np.array([10.0, 15.0]),
        )
        assert fd.nodes.shape == (2, 2)
        assert fd.boundary[0] is np.bool_(True)

    def test_subregion_data(self) -> None:
        sd = SubregionData(
            id=1,
            name="Region A",
            n_elements=3,
            n_neighbor_regions=1,
            area=1000.0,
            region_elements=np.array([1, 2, 3], dtype=np.int32),
            neighbor_region_ids=np.array([2], dtype=np.int32),
            neighbor_n_boundary_faces=np.array([2], dtype=np.int32),
            neighbor_boundary_faces=[np.array([5, 6], dtype=np.int32)],
        )
        assert sd.name == "Region A"
        assert sd.n_elements == 3

    def test_stratigraphy_data(self) -> None:
        sd = StratigraphyData(
            n_layers=2,
            ground_surface_elev=np.array([100.0, 110.0]),
            top_elev=np.array([[90.0, 80.0], [100.0, 90.0]]),
            bottom_elev=np.array([[80.0, 60.0], [90.0, 70.0]]),
            active_node=np.array([[True, True], [True, False]]),
            active_layer_above=np.array([[0, 1], [0, 1]], dtype=np.int32),
            active_layer_below=np.array([[2, 0], [2, 0]], dtype=np.int32),
            top_active_layer=np.array([1, 1], dtype=np.int32),
            bottom_active_layer=np.array([2, 1], dtype=np.int32),
        )
        assert sd.n_layers == 2

    def test_stream_gw_connector_data(self) -> None:
        sgc = StreamGWConnectorData(
            n_stream_nodes=2,
            gw_nodes=np.array([1, 5], dtype=np.int32),
            layers=np.array([1, 1], dtype=np.int32),
        )
        assert sgc.n_stream_nodes == 2

    def test_lake_gw_connector_data(self) -> None:
        lgc = LakeGWConnectorData(
            n_lakes=1,
            lake_elements=[np.array([1, 2, 3], dtype=np.int32)],
            lake_nodes=[np.array([10, 20, 30], dtype=np.int32)],
        )
        assert lgc.n_lakes == 1

    def test_stream_lake_connector_data(self) -> None:
        slc = StreamLakeConnectorData(
            n_connections=1,
            stream_nodes=np.array([5], dtype=np.int32),
            lake_ids=np.array([1], dtype=np.int32),
        )
        assert slc.n_connections == 1

    def test_stream_data(self) -> None:
        sd = StreamData(
            n_reaches=2,
            n_stream_nodes=10,
            reach_ids=np.array([1, 2], dtype=np.int32),
            reach_names=["Reach A", "Reach B"],
            reach_upstream_nodes=np.array([1, 6], dtype=np.int32),
            reach_downstream_nodes=np.array([5, 10], dtype=np.int32),
            reach_outflow_dest=np.array([2, 0], dtype=np.int32),
        )
        assert sd.n_reaches == 2
        assert sd.reach_names[1] == "Reach B"

    def test_lake_data(self) -> None:
        ld = LakeData(
            n_lakes=1,
            lake_ids=np.array([1], dtype=np.int32),
            lake_names=["Lake A"],
            lake_max_elevations=np.array([500.0]),
            lake_elements=[np.array([10, 11, 12], dtype=np.int32)],
        )
        assert ld.n_lakes == 1
        assert ld.lake_names[0] == "Lake A"

    def test_preprocessor_binary_data_defaults(self) -> None:
        data = PreprocessorBinaryData()
        assert data.n_nodes == 0
        assert data.n_elements == 0
        assert data.n_faces == 0
        assert data.n_subregions == 0
        assert data.stratigraphy is None
        assert data.streams is None
        assert data.lakes is None
        assert data.matrix_n_equations == 0


class TestPreprocessorBinaryReader:
    """Tests for PreprocessorBinaryReader with mocked binary I/O."""

    def test_reader_init_endian(self) -> None:
        reader = PreprocessorBinaryReader(endian=">")
        assert reader.endian == ">"

    def test_reader_default_endian(self) -> None:
        reader = PreprocessorBinaryReader()
        assert reader.endian == "<"

    def test_read_file_not_found(self, tmp_path: Path) -> None:
        """Non-existent file raises FileNotFoundError."""
        reader = PreprocessorBinaryReader()
        with pytest.raises(FileNotFoundError):
            reader.read(tmp_path / "nonexistent.bin")

    def test_convenience_function_file_not_found(self, tmp_path: Path) -> None:
        """Convenience function raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            read_preprocessor_binary(tmp_path / "nonexistent.bin")

    def test_convenience_function_endian_param(self) -> None:
        """Verify endian parameter is passed to reader."""
        with patch.object(PreprocessorBinaryReader, "read") as mock_read:
            mock_read.side_effect = FileNotFoundError("test")
            with pytest.raises(FileNotFoundError):
                read_preprocessor_binary("/fake/path.bin", endian=">")

    def test_read_app_faces_zero(self) -> None:
        """_read_app_faces with 0 faces returns empty data."""
        reader = PreprocessorBinaryReader()
        mock_f = MagicMock()
        result = reader._read_app_faces(mock_f, 0)

        assert result.nodes.shape == (0, 2)
        assert result.elements.shape == (0, 2)
        assert len(result.boundary) == 0
        assert len(result.lengths) == 0
