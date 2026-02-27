"""Tests for write_binary_mesh and write_binary_stratigraphy in pyiwfm.io.binary."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyiwfm.io.binary import (
    FortranBinaryReader,
    StreamAccessBinaryReader,
    write_binary_mesh,
    write_binary_stratigraphy,
)

# ======================================================================
# write_binary_mesh
# ======================================================================


class TestWriteBinaryMesh:
    """Tests for the write_binary_mesh function."""

    @pytest.fixture()
    def simple_grid(self) -> object:
        """Create a minimal 2-element, 4-node grid mock."""
        from unittest.mock import MagicMock

        grid = MagicMock()
        grid.n_nodes = 4
        grid.n_elements = 2
        grid.x = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64)
        grid.y = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)

        elem1 = MagicMock()
        elem1.vertices = [1, 2, 3, 4]
        elem1.subregion = 1

        elem2 = MagicMock()
        elem2.vertices = [2, 5, 3]  # triangle (only 3 vertices)
        elem2.subregion = 2

        grid.elements = {1: elem1, 2: elem2}
        return grid

    def test_writes_node_and_element_counts(self, tmp_path: Path, simple_grid: object) -> None:
        outpath = tmp_path / "mesh.bin"
        write_binary_mesh(outpath, simple_grid)

        with FortranBinaryReader(outpath) as r:
            n_nodes = r.read_int()
            n_elements = r.read_int()
            assert n_nodes == 4
            assert n_elements == 2

    def test_writes_coordinates(self, tmp_path: Path, simple_grid: object) -> None:
        outpath = tmp_path / "mesh.bin"
        write_binary_mesh(outpath, simple_grid)

        with FortranBinaryReader(outpath) as r:
            r.read_int()  # n_nodes
            r.read_int()  # n_elements
            x = r.read_double_array()
            y = r.read_double_array()
            np.testing.assert_array_almost_equal(x, simple_grid.x)
            np.testing.assert_array_almost_equal(y, simple_grid.y)

    def test_writes_vertex_array_zero_based(self, tmp_path: Path, simple_grid: object) -> None:
        outpath = tmp_path / "mesh.bin"
        write_binary_mesh(outpath, simple_grid)

        with FortranBinaryReader(outpath) as r:
            r.read_int()  # n_nodes
            r.read_int()  # n_elements
            r.read_double_array()  # x
            r.read_double_array()  # y
            flat_verts = r.read_int_array()
            # 2 elements * 4 = 8 vertex entries
            assert len(flat_verts) == 8
            # First element (1,2,3,4) -> 0-based: (0,1,2,3)
            assert flat_verts[0] == 0
            assert flat_verts[1] == 1
            assert flat_verts[2] == 2
            assert flat_verts[3] == 3

    def test_writes_subregion_array(self, tmp_path: Path, simple_grid: object) -> None:
        outpath = tmp_path / "mesh.bin"
        write_binary_mesh(outpath, simple_grid)

        with FortranBinaryReader(outpath) as r:
            r.read_int()  # n_nodes
            r.read_int()  # n_elements
            r.read_double_array()  # x
            r.read_double_array()  # y
            r.read_int_array()  # vertex
            subregions = r.read_int_array()
            assert len(subregions) == 2
            assert subregions[0] == 1
            assert subregions[1] == 2

    def test_string_path_accepted(self, tmp_path: Path, simple_grid: object) -> None:
        outpath = str(tmp_path / "mesh.bin")
        write_binary_mesh(outpath, simple_grid)
        assert Path(outpath).exists()

    def test_big_endian(self, tmp_path: Path, simple_grid: object) -> None:
        outpath = tmp_path / "mesh_be.bin"
        write_binary_mesh(outpath, simple_grid, endian=">")

        with FortranBinaryReader(outpath, endian=">") as r:
            n_nodes = r.read_int()
            assert n_nodes == 4


# ======================================================================
# write_binary_stratigraphy
# ======================================================================


class TestWriteBinaryStratigraphy:
    """Tests for the write_binary_stratigraphy function."""

    @pytest.fixture()
    def simple_strat(self) -> object:
        """Create a minimal stratigraphy mock (3 nodes, 2 layers)."""
        from unittest.mock import MagicMock

        strat = MagicMock()
        strat.n_nodes = 3
        strat.n_layers = 2
        strat.gs_elev = np.array([100.0, 110.0, 105.0], dtype=np.float64)
        strat.top_elev = np.array([[100.0, 80.0], [110.0, 90.0], [105.0, 85.0]], dtype=np.float64)
        strat.bottom_elev = np.array([[80.0, 50.0], [90.0, 60.0], [85.0, 55.0]], dtype=np.float64)
        strat.active_node = np.array([[1, 1], [1, 0], [1, 1]], dtype=np.float64)
        return strat

    def test_writes_dimensions(self, tmp_path: Path, simple_strat: object) -> None:
        outpath = tmp_path / "strat.bin"
        write_binary_stratigraphy(outpath, simple_strat)

        with FortranBinaryReader(outpath) as r:
            n_nodes = r.read_int()
            n_layers = r.read_int()
            assert n_nodes == 3
            assert n_layers == 2

    def test_writes_elevations(self, tmp_path: Path, simple_strat: object) -> None:
        outpath = tmp_path / "strat.bin"
        write_binary_stratigraphy(outpath, simple_strat)

        with FortranBinaryReader(outpath) as r:
            r.read_int()  # n_nodes
            r.read_int()  # n_layers
            gs_elev = r.read_double_array()
            np.testing.assert_array_almost_equal(gs_elev, simple_strat.gs_elev)

            top_flat = r.read_double_array()
            assert len(top_flat) == 6  # 3 nodes * 2 layers
            np.testing.assert_array_almost_equal(top_flat, simple_strat.top_elev.flatten())

            bot_flat = r.read_double_array()
            assert len(bot_flat) == 6
            np.testing.assert_array_almost_equal(bot_flat, simple_strat.bottom_elev.flatten())

    def test_writes_active_nodes_as_int32(self, tmp_path: Path, simple_strat: object) -> None:
        outpath = tmp_path / "strat.bin"
        write_binary_stratigraphy(outpath, simple_strat)

        with FortranBinaryReader(outpath) as r:
            r.read_int()  # n_nodes
            r.read_int()  # n_layers
            r.read_double_array()  # gs_elev
            r.read_double_array()  # top_elev
            r.read_double_array()  # bottom_elev
            active = r.read_int_array()
            assert len(active) == 6
            assert active[0] == 1
            assert active[3] == 0  # node 2, layer 2

    def test_string_path_accepted(self, tmp_path: Path, simple_strat: object) -> None:
        outpath = str(tmp_path / "strat.bin")
        write_binary_stratigraphy(outpath, simple_strat)
        assert Path(outpath).exists()


# ======================================================================
# StreamAccessBinaryReader edge cases
# ======================================================================


class TestStreamAccessBinaryReaderEdges:
    """Edge case tests for StreamAccessBinaryReader."""

    def test_read_ints_zero_count(self, tmp_path: Path) -> None:
        fpath = tmp_path / "empty.bin"
        fpath.write_bytes(b"\x00" * 4)  # dummy byte
        with StreamAccessBinaryReader(fpath) as r:
            arr = r.read_ints(0)
            assert len(arr) == 0
            assert arr.dtype == np.int32

    def test_read_ints_negative_count(self, tmp_path: Path) -> None:
        fpath = tmp_path / "empty.bin"
        fpath.write_bytes(b"\x00" * 4)
        with StreamAccessBinaryReader(fpath) as r:
            arr = r.read_ints(-5)
            assert len(arr) == 0

    def test_read_doubles_zero_count(self, tmp_path: Path) -> None:
        fpath = tmp_path / "empty.bin"
        fpath.write_bytes(b"\x00" * 8)
        with StreamAccessBinaryReader(fpath) as r:
            arr = r.read_doubles(0)
            assert len(arr) == 0
            assert arr.dtype == np.float64

    def test_read_doubles_negative_count(self, tmp_path: Path) -> None:
        fpath = tmp_path / "empty.bin"
        fpath.write_bytes(b"\x00" * 8)
        with StreamAccessBinaryReader(fpath) as r:
            arr = r.read_doubles(-3)
            assert len(arr) == 0
