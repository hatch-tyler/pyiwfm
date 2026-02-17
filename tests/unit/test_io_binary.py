"""Unit tests for binary I/O handlers."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from pyiwfm.core.exceptions import FileFormatError
from pyiwfm.core.mesh import AppGrid, Element, Node
from pyiwfm.core.stratigraphy import Stratigraphy
from pyiwfm.io.binary import (
    FortranBinaryReader,
    FortranBinaryWriter,
    read_binary_mesh,
    read_binary_stratigraphy,
    write_binary_mesh,
    write_binary_stratigraphy,
)


class TestFortranBinaryReader:
    """Tests for FortranBinaryReader."""

    def test_read_int(self, tmp_path: Path) -> None:
        """Test reading a single integer."""
        filepath = tmp_path / "test.bin"

        # Write a Fortran-style integer record
        with open(filepath, "wb") as f:
            value = 42
            data = struct.pack("<i", value)
            marker = struct.pack("<i", len(data))
            f.write(marker + data + marker)

        with FortranBinaryReader(filepath) as reader:
            result = reader.read_int()

        assert result == 42

    def test_read_int_array(self, tmp_path: Path) -> None:
        """Test reading an integer array."""
        filepath = tmp_path / "test.bin"

        values = [1, 2, 3, 4, 5]
        with open(filepath, "wb") as f:
            data = struct.pack("<5i", *values)
            marker = struct.pack("<i", len(data))
            f.write(marker + data + marker)

        with FortranBinaryReader(filepath) as reader:
            result = reader.read_int_array()

        np.testing.assert_array_equal(result, values)

    def test_read_double_array(self, tmp_path: Path) -> None:
        """Test reading a double array."""
        filepath = tmp_path / "test.bin"

        values = [1.1, 2.2, 3.3]
        with open(filepath, "wb") as f:
            data = struct.pack("<3d", *values)
            marker = struct.pack("<i", len(data))
            f.write(marker + data + marker)

        with FortranBinaryReader(filepath) as reader:
            result = reader.read_double_array()

        np.testing.assert_allclose(result, values)

    def test_read_multiple_records(self, tmp_path: Path) -> None:
        """Test reading multiple records."""
        filepath = tmp_path / "test.bin"

        with open(filepath, "wb") as f:
            # Record 1: integer
            data1 = struct.pack("<i", 100)
            marker1 = struct.pack("<i", len(data1))
            f.write(marker1 + data1 + marker1)

            # Record 2: double array
            data2 = struct.pack("<3d", 1.0, 2.0, 3.0)
            marker2 = struct.pack("<i", len(data2))
            f.write(marker2 + data2 + marker2)

        with FortranBinaryReader(filepath) as reader:
            n = reader.read_int()
            arr = reader.read_double_array()

        assert n == 100
        np.testing.assert_allclose(arr, [1.0, 2.0, 3.0])

    def test_read_mismatched_markers(self, tmp_path: Path) -> None:
        """Test error on mismatched record markers."""
        filepath = tmp_path / "test.bin"

        with open(filepath, "wb") as f:
            data = struct.pack("<i", 42)
            leading_marker = struct.pack("<i", len(data))
            trailing_marker = struct.pack("<i", len(data) + 1)  # Wrong!
            f.write(leading_marker + data + trailing_marker)

        with FortranBinaryReader(filepath) as reader:
            with pytest.raises(FileFormatError, match="marker mismatch"):
                reader.read_int()


class TestFortranBinaryWriter:
    """Tests for FortranBinaryWriter."""

    def test_write_int(self, tmp_path: Path) -> None:
        """Test writing a single integer."""
        filepath = tmp_path / "test.bin"

        with FortranBinaryWriter(filepath) as writer:
            writer.write_int(42)

        # Verify by reading back
        with FortranBinaryReader(filepath) as reader:
            result = reader.read_int()

        assert result == 42

    def test_write_double_array(self, tmp_path: Path) -> None:
        """Test writing a double array."""
        filepath = tmp_path / "test.bin"

        values = np.array([1.1, 2.2, 3.3])
        with FortranBinaryWriter(filepath) as writer:
            writer.write_double_array(values)

        with FortranBinaryReader(filepath) as reader:
            result = reader.read_double_array()

        np.testing.assert_allclose(result, values)

    def test_roundtrip_multiple_records(self, tmp_path: Path) -> None:
        """Test writing and reading multiple records."""
        filepath = tmp_path / "test.bin"

        with FortranBinaryWriter(filepath) as writer:
            writer.write_int(10)
            writer.write_int(20)
            writer.write_double_array(np.array([1.0, 2.0, 3.0]))
            writer.write_int_array(np.array([100, 200, 300], dtype=np.int32))

        with FortranBinaryReader(filepath) as reader:
            n1 = reader.read_int()
            n2 = reader.read_int()
            arr_d = reader.read_double_array()
            arr_i = reader.read_int_array()

        assert n1 == 10
        assert n2 == 20
        np.testing.assert_allclose(arr_d, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(arr_i, [100, 200, 300])


class TestBinaryMeshIO:
    """Tests for binary mesh I/O."""

    def test_write_read_mesh_roundtrip(
        self,
        tmp_path: Path,
        small_grid_nodes: list[dict],
        small_grid_elements: list[dict],
    ) -> None:
        """Test mesh write/read roundtrip."""
        # Create original mesh
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        elements = {d["id"]: Element(**d) for d in small_grid_elements}
        grid = AppGrid(nodes=nodes, elements=elements)

        # Write to binary
        filepath = tmp_path / "mesh.bin"
        write_binary_mesh(filepath, grid)

        # Read back
        grid_back = read_binary_mesh(filepath)

        # Verify
        assert grid_back.n_nodes == grid.n_nodes
        assert grid_back.n_elements == grid.n_elements

        # Check coordinates
        for nid in grid.nodes:
            assert grid_back.nodes[nid].x == pytest.approx(grid.nodes[nid].x)
            assert grid_back.nodes[nid].y == pytest.approx(grid.nodes[nid].y)

        # Check elements
        for eid in grid.elements:
            assert grid_back.elements[eid].vertices == grid.elements[eid].vertices
            assert grid_back.elements[eid].subregion == grid.elements[eid].subregion

    def test_write_read_triangular_mesh(self, tmp_path: Path) -> None:
        """Test roundtrip with triangular elements."""
        nodes = {
            1: Node(id=1, x=0.0, y=0.0),
            2: Node(id=2, x=100.0, y=0.0),
            3: Node(id=3, x=50.0, y=86.6),
        }
        elements = {
            1: Element(id=1, vertices=(1, 2, 3), subregion=1),
        }
        grid = AppGrid(nodes=nodes, elements=elements)

        filepath = tmp_path / "tri_mesh.bin"
        write_binary_mesh(filepath, grid)
        grid_back = read_binary_mesh(filepath)

        assert grid_back.elements[1].is_triangle
        assert grid_back.elements[1].vertices == (1, 2, 3)


class TestBinaryStratigraphyIO:
    """Tests for binary stratigraphy I/O."""

    def test_write_read_stratigraphy_roundtrip(
        self, tmp_path: Path, sample_stratigraphy_data: dict
    ) -> None:
        """Test stratigraphy write/read roundtrip."""
        strat = Stratigraphy(**sample_stratigraphy_data)

        filepath = tmp_path / "strat.bin"
        write_binary_stratigraphy(filepath, strat)
        strat_back = read_binary_stratigraphy(filepath)

        assert strat_back.n_nodes == strat.n_nodes
        assert strat_back.n_layers == strat.n_layers
        np.testing.assert_allclose(strat_back.gs_elev, strat.gs_elev)
        np.testing.assert_allclose(strat_back.top_elev, strat.top_elev)
        np.testing.assert_allclose(strat_back.bottom_elev, strat.bottom_elev)
        np.testing.assert_array_equal(strat_back.active_node, strat.active_node)

    def test_binary_stratigraphy_inactive_nodes(self, tmp_path: Path) -> None:
        """Test stratigraphy with inactive nodes."""
        n_nodes = 4
        n_layers = 2
        gs_elev = np.array([100.0, 100.0, 100.0, 100.0])
        top_elev = np.array([[100.0, 50.0]] * 4)
        bottom_elev = np.array([[50.0, 0.0]] * 4)
        active_node = np.array([[True, True], [True, False], [False, True], [False, False]])

        strat = Stratigraphy(
            n_layers=n_layers,
            n_nodes=n_nodes,
            gs_elev=gs_elev,
            top_elev=top_elev,
            bottom_elev=bottom_elev,
            active_node=active_node,
        )

        filepath = tmp_path / "strat_inactive.bin"
        write_binary_stratigraphy(filepath, strat)
        strat_back = read_binary_stratigraphy(filepath)

        np.testing.assert_array_equal(strat_back.active_node, active_node)


class TestBinaryEndianness:
    """Tests for endianness handling."""

    def test_big_endian_roundtrip(self, tmp_path: Path) -> None:
        """Test big-endian write/read."""
        filepath = tmp_path / "big_endian.bin"

        with FortranBinaryWriter(filepath, endian=">") as writer:
            writer.write_int(12345)
            writer.write_double_array(np.array([1.5, 2.5, 3.5]))

        with FortranBinaryReader(filepath, endian=">") as reader:
            n = reader.read_int()
            arr = reader.read_double_array()

        assert n == 12345
        np.testing.assert_allclose(arr, [1.5, 2.5, 3.5])

    def test_mesh_big_endian(
        self,
        tmp_path: Path,
        small_grid_nodes: list[dict],
        small_grid_elements: list[dict],
    ) -> None:
        """Test mesh I/O with big endian."""
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        elements = {d["id"]: Element(**d) for d in small_grid_elements}
        grid = AppGrid(nodes=nodes, elements=elements)

        filepath = tmp_path / "mesh_be.bin"
        write_binary_mesh(filepath, grid, endian=">")
        grid_back = read_binary_mesh(filepath, endian=">")

        assert grid_back.n_nodes == grid.n_nodes
        assert grid_back.n_elements == grid.n_elements


# =============================================================================
# Additional coverage tests
# =============================================================================


class TestFortranBinaryReaderEdgeCases:
    """Additional edge-case tests for FortranBinaryReader."""

    def test_read_record_file_not_open(self, tmp_path: Path) -> None:
        """Reading from a reader that is not opened raises RuntimeError."""
        filepath = tmp_path / "test.bin"
        filepath.write_bytes(b"")
        reader = FortranBinaryReader(filepath)
        # Do NOT use 'with', so _file stays None
        with pytest.raises(RuntimeError, match="File not open"):
            reader.read_record()

    def test_read_record_eof(self, tmp_path: Path) -> None:
        """Reading from an empty file raises EOFError."""
        filepath = tmp_path / "empty.bin"
        filepath.write_bytes(b"")
        with FortranBinaryReader(filepath) as reader:
            with pytest.raises(EOFError):
                reader.read_record()

    def test_read_record_incomplete_data(self, tmp_path: Path) -> None:
        """Incomplete record (marker says 100 bytes, only 4 available) raises FileFormatError."""
        filepath = tmp_path / "incomplete.bin"
        with open(filepath, "wb") as f:
            # marker says 100 bytes
            f.write(struct.pack("<i", 100))
            # only 4 bytes of data
            f.write(struct.pack("<i", 42))
        with FortranBinaryReader(filepath) as reader:
            with pytest.raises(FileFormatError, match="Incomplete record"):
                reader.read_record()

    def test_read_float(self, tmp_path: Path) -> None:
        """Test reading a single float record."""
        filepath = tmp_path / "float.bin"
        with FortranBinaryWriter(filepath) as writer:
            writer.write_float(3.14)
        with FortranBinaryReader(filepath) as reader:
            result = reader.read_float()
        assert result == pytest.approx(3.14, rel=1e-5)

    def test_read_float_array(self, tmp_path: Path) -> None:
        """Test reading a float array record."""
        filepath = tmp_path / "float_arr.bin"
        values = np.array([1.0, 2.5, 3.5], dtype=np.float32)
        with FortranBinaryWriter(filepath) as writer:
            writer.write_float_array(values)
        with FortranBinaryReader(filepath) as reader:
            result = reader.read_float_array()
        np.testing.assert_allclose(result, values, rtol=1e-5)

    def test_read_double(self, tmp_path: Path) -> None:
        """Test reading a single double record."""
        filepath = tmp_path / "double.bin"
        with FortranBinaryWriter(filepath) as writer:
            writer.write_double(2.718281828)
        with FortranBinaryReader(filepath) as reader:
            result = reader.read_double()
        assert result == pytest.approx(2.718281828)

    def test_read_string(self, tmp_path: Path) -> None:
        """Test reading a string record."""
        filepath = tmp_path / "string.bin"
        with FortranBinaryWriter(filepath) as writer:
            writer.write_string("Hello")
        with FortranBinaryReader(filepath) as reader:
            result = reader.read_string()
        assert result == "Hello"

    def test_read_string_with_length(self, tmp_path: Path) -> None:
        """Test reading a string record truncated to specified length."""
        filepath = tmp_path / "string_len.bin"
        with FortranBinaryWriter(filepath) as writer:
            writer.write_string("Hello World", length=20)
        with FortranBinaryReader(filepath) as reader:
            result = reader.read_string(length=5)
        assert result == "Hello"

    def test_read_eof_marker(self, tmp_path: Path) -> None:
        """Reading past the last record raises EOFError."""
        filepath = tmp_path / "single_record.bin"
        with FortranBinaryWriter(filepath) as writer:
            writer.write_int(42)
        with FortranBinaryReader(filepath) as reader:
            reader.read_int()
            with pytest.raises(EOFError):
                reader.read_int()


class TestFortranBinaryWriterEdgeCases:
    """Additional edge-case tests for FortranBinaryWriter."""

    def test_write_record_file_not_open(self, tmp_path: Path) -> None:
        """Writing to a writer that is not opened raises RuntimeError."""
        filepath = tmp_path / "test.bin"
        writer = FortranBinaryWriter(filepath)
        with pytest.raises(RuntimeError, match="File not open"):
            writer.write_record(b"data")

    def test_write_float(self, tmp_path: Path) -> None:
        """Test writing a single float record."""
        filepath = tmp_path / "wfloat.bin"
        with FortranBinaryWriter(filepath) as writer:
            writer.write_float(1.5)
        with FortranBinaryReader(filepath) as reader:
            result = reader.read_float()
        assert result == pytest.approx(1.5, rel=1e-5)

    def test_write_float_array(self, tmp_path: Path) -> None:
        """Test writing a float array record."""
        filepath = tmp_path / "wfloat_arr.bin"
        values = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        with FortranBinaryWriter(filepath) as writer:
            writer.write_float_array(values)
        with FortranBinaryReader(filepath) as reader:
            result = reader.read_float_array()
        np.testing.assert_allclose(result, values, rtol=1e-5)

    def test_write_double(self, tmp_path: Path) -> None:
        """Test writing a single double record."""
        filepath = tmp_path / "wdouble.bin"
        with FortranBinaryWriter(filepath) as writer:
            writer.write_double(9.81)
        with FortranBinaryReader(filepath) as reader:
            result = reader.read_double()
        assert result == pytest.approx(9.81)

    def test_write_string(self, tmp_path: Path) -> None:
        """Test writing a string record without length padding."""
        filepath = tmp_path / "wstring.bin"
        with FortranBinaryWriter(filepath) as writer:
            writer.write_string("IWFM")
        with FortranBinaryReader(filepath) as reader:
            result = reader.read_string()
        assert result == "IWFM"

    def test_write_string_with_length(self, tmp_path: Path) -> None:
        """Test writing a string record padded/truncated to a length."""
        filepath = tmp_path / "wstring_len.bin"
        with FortranBinaryWriter(filepath) as writer:
            writer.write_string("Hi", length=10)
        with FortranBinaryReader(filepath) as reader:
            result = reader.read_string()
        assert result == "Hi"
        # The string should have been padded to 10 chars
        with FortranBinaryReader(filepath) as reader:
            raw = reader.read_record()
        assert len(raw) == 10

    def test_write_string_truncated(self, tmp_path: Path) -> None:
        """Test that writing a long string truncates to specified length."""
        filepath = tmp_path / "wstring_trunc.bin"
        with FortranBinaryWriter(filepath) as writer:
            writer.write_string("LongStringHere", length=4)
        with FortranBinaryReader(filepath) as reader:
            result = reader.read_string()
        assert result == "Long"

    def test_write_creates_parent_directory(self, tmp_path: Path) -> None:
        """Writer creates parent directories if they don't exist."""
        filepath = tmp_path / "subdir" / "nested" / "test.bin"
        with FortranBinaryWriter(filepath) as writer:
            writer.write_int(1)
        assert filepath.exists()


class TestBinaryMeshIOEdgeCases:
    """Additional edge-case tests for binary mesh I/O."""

    def test_read_binary_mesh_coord_mismatch(self, tmp_path: Path) -> None:
        """Test error when coordinate arrays don't match n_nodes."""
        filepath = tmp_path / "badmesh.bin"
        with FortranBinaryWriter(filepath) as writer:
            writer.write_int(3)  # n_nodes = 3
            writer.write_int(1)  # n_elements = 1
            # But only write 2 coordinates
            writer.write_double_array(np.array([0.0, 1.0]))
            writer.write_double_array(np.array([0.0, 1.0]))
        with pytest.raises(FileFormatError, match="Coordinate array size mismatch"):
            read_binary_mesh(filepath)

    def test_binary_mesh_roundtrip_mixed_elements(self, tmp_path: Path) -> None:
        """Test roundtrip with a mix of triangles and quads."""
        nodes = {
            1: Node(id=1, x=0.0, y=0.0),
            2: Node(id=2, x=100.0, y=0.0),
            3: Node(id=3, x=100.0, y=100.0),
            4: Node(id=4, x=0.0, y=100.0),
            5: Node(id=5, x=50.0, y=150.0),
        }
        elements = {
            1: Element(id=1, vertices=(1, 2, 3, 4), subregion=1),
            2: Element(id=2, vertices=(4, 3, 5), subregion=2),
        }
        grid = AppGrid(nodes=nodes, elements=elements)

        filepath = tmp_path / "mixed.bin"
        write_binary_mesh(filepath, grid)
        grid_back = read_binary_mesh(filepath)

        assert grid_back.n_nodes == 5
        assert grid_back.n_elements == 2
        assert grid_back.elements[1].is_quad
        assert grid_back.elements[2].is_triangle
        assert grid_back.elements[1].subregion == 1
        assert grid_back.elements[2].subregion == 2

    def test_write_read_single_element_mesh(self, tmp_path: Path) -> None:
        """Test roundtrip with a single triangular element."""
        nodes = {
            1: Node(id=1, x=0.0, y=0.0),
            2: Node(id=2, x=10.0, y=0.0),
            3: Node(id=3, x=5.0, y=8.66),
        }
        elements = {
            1: Element(id=1, vertices=(1, 2, 3), subregion=1),
        }
        grid = AppGrid(nodes=nodes, elements=elements)

        filepath = tmp_path / "single.bin"
        write_binary_mesh(filepath, grid)
        grid_back = read_binary_mesh(filepath)

        assert grid_back.n_nodes == 3
        assert grid_back.n_elements == 1
        assert grid_back.elements[1].vertices == (1, 2, 3)


class TestBinaryStratigraphyIOEdgeCases:
    """Additional edge-case tests for binary stratigraphy I/O."""

    def test_stratigraphy_big_endian_roundtrip(self, tmp_path: Path) -> None:
        """Test stratigraphy I/O with big-endian byte order."""
        n_nodes = 3
        n_layers = 1
        gs_elev = np.array([100.0, 110.0, 120.0])
        top_elev = np.array([[100.0], [110.0], [120.0]])
        bottom_elev = np.array([[50.0], [60.0], [70.0]])
        active_node = np.array([[True], [True], [False]])

        strat = Stratigraphy(
            n_layers=n_layers,
            n_nodes=n_nodes,
            gs_elev=gs_elev,
            top_elev=top_elev,
            bottom_elev=bottom_elev,
            active_node=active_node,
        )

        filepath = tmp_path / "strat_be.bin"
        write_binary_stratigraphy(filepath, strat, endian=">")
        strat_back = read_binary_stratigraphy(filepath, endian=">")

        assert strat_back.n_nodes == n_nodes
        assert strat_back.n_layers == n_layers
        np.testing.assert_allclose(strat_back.gs_elev, gs_elev)
        np.testing.assert_allclose(strat_back.top_elev, top_elev)
        np.testing.assert_allclose(strat_back.bottom_elev, bottom_elev)
        np.testing.assert_array_equal(strat_back.active_node, active_node)

    def test_stratigraphy_single_node_single_layer(self, tmp_path: Path) -> None:
        """Test the minimal stratigraphy: 1 node, 1 layer."""
        strat = Stratigraphy(
            n_layers=1,
            n_nodes=1,
            gs_elev=np.array([200.0]),
            top_elev=np.array([[200.0]]),
            bottom_elev=np.array([[100.0]]),
            active_node=np.array([[True]]),
        )
        filepath = tmp_path / "strat_min.bin"
        write_binary_stratigraphy(filepath, strat)
        strat_back = read_binary_stratigraphy(filepath)

        assert strat_back.n_nodes == 1
        assert strat_back.n_layers == 1
        assert strat_back.gs_elev[0] == pytest.approx(200.0)

    def test_stratigraphy_many_layers(self, tmp_path: Path) -> None:
        """Test stratigraphy with several layers."""
        n_nodes = 2
        n_layers = 5
        gs = np.array([500.0, 500.0])
        top = np.zeros((n_nodes, n_layers))
        bot = np.zeros((n_nodes, n_layers))
        for layer in range(n_layers):
            top[:, layer] = 500.0 - layer * 100.0
            bot[:, layer] = 400.0 - layer * 100.0
        active = np.ones((n_nodes, n_layers), dtype=bool)

        strat = Stratigraphy(
            n_layers=n_layers,
            n_nodes=n_nodes,
            gs_elev=gs,
            top_elev=top,
            bottom_elev=bot,
            active_node=active,
        )
        filepath = tmp_path / "strat_many.bin"
        write_binary_stratigraphy(filepath, strat)
        strat_back = read_binary_stratigraphy(filepath)

        assert strat_back.n_layers == 5
        np.testing.assert_allclose(strat_back.top_elev, top)
        np.testing.assert_allclose(strat_back.bottom_elev, bot)
