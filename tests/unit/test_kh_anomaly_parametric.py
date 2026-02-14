"""Unit tests for Kh anomaly overwrites and parametric grid interpolation.

Tests:
- KhAnomalyEntry parsing from GW main file
- Anomaly application to node arrays via mesh connectivity
- ParametricGrid point-in-triangle and point-in-quad
- Parametric grid interpolation at known locations
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pytest

from pyiwfm.io.groundwater import (
    KhAnomalyEntry,
    ParametricGridData,
    GWMainFileReader,
    GWMainFileConfig,
)
from pyiwfm.io.parametric_grid import (
    ParametricGrid,
    ParamNode,
    ParamElement,
)
from pyiwfm.components.groundwater import AquiferParameters
from pyiwfm.core.mesh import AppGrid, Node, Element
from pyiwfm.core.model import _apply_kh_anomalies


# =============================================================================
# Test KhAnomalyEntry parsing
# =============================================================================


class TestReadKhAnomaly:
    """Tests for _read_kh_anomaly parsing."""

    def _make_reader(self):
        reader = GWMainFileReader()
        reader._line_num = 0
        return reader

    def test_parse_basic_anomaly(self):
        """Parse a basic anomaly section with 2 elements and 2 layers."""
        text = (
            "2                         / NEBK\n"
            "1.0                       / FACT\n"
            "1DAY                      / TUNITH\n"
            "1   100  1.14E+02  6.00E-05\n"
            "2   200  9.00E+01  3.00E-05\n"
        )
        reader = self._make_reader()
        entries = reader._read_kh_anomaly(StringIO(text))

        assert len(entries) == 2
        assert entries[0].element_id == 100
        assert entries[0].kh_per_layer == pytest.approx([114.0, 6e-5])
        assert entries[1].element_id == 200
        assert entries[1].kh_per_layer == pytest.approx([90.0, 3e-5])

    def test_parse_with_factor(self):
        """Factor should multiply all Kh values."""
        text = (
            "1                         / NEBK\n"
            "2.0                       / FACT\n"
            "1DAY                      / TUNITH\n"
            "1   50   10.0   20.0\n"
        )
        reader = self._make_reader()
        entries = reader._read_kh_anomaly(StringIO(text))

        assert len(entries) == 1
        assert entries[0].element_id == 50
        assert entries[0].kh_per_layer == pytest.approx([20.0, 40.0])

    def test_zero_nebk(self):
        """NEBK=0 should return empty list."""
        text = "0                         / NEBK\n"
        reader = self._make_reader()
        entries = reader._read_kh_anomaly(StringIO(text))
        assert entries == []

    def test_empty_section(self):
        """Empty input should return empty list."""
        reader = self._make_reader()
        entries = reader._read_kh_anomaly(StringIO(""))
        assert entries == []

    def test_comment_lines_skipped(self):
        """Comment lines within anomaly data should be skipped."""
        text = (
            "1                         / NEBK\n"
            "1.0                       / FACT\n"
            "1DAY                      / TUNITH\n"
            "C  This is a comment\n"
            "1   300  50.0  50.0  50.0\n"
        )
        reader = self._make_reader()
        entries = reader._read_kh_anomaly(StringIO(text))

        assert len(entries) == 1
        assert entries[0].element_id == 300


# =============================================================================
# Test _apply_kh_anomalies
# =============================================================================


class TestApplyKhAnomalies:
    """Tests for applying Kh anomaly overwrites to node arrays."""

    def _make_mesh(self):
        """Create a simple 4-element, 9-node mesh.

        Layout (2x2 quad grid):
            7---8---9
            |   |   |
            4---5---6
            |   |   |
            1---2---3

        Elements:
            1: (1, 2, 5, 4)
            2: (2, 3, 6, 5)
            3: (4, 5, 8, 7)
            4: (5, 6, 9, 8)
        """
        nodes = {
            i: Node(id=i, x=float((i - 1) % 3), y=float((i - 1) // 3))
            for i in range(1, 10)
        }
        elements = {
            1: Element(id=1, vertices=(1, 2, 5, 4), subregion=1),
            2: Element(id=2, vertices=(2, 3, 6, 5), subregion=1),
            3: Element(id=3, vertices=(4, 5, 8, 7), subregion=1),
            4: Element(id=4, vertices=(5, 6, 9, 8), subregion=1),
        }
        return AppGrid(nodes=nodes, elements=elements)

    def _make_params(self, n_nodes=9, n_layers=2):
        """Create uniform aquifer parameters."""
        kh = np.ones((n_nodes, n_layers), dtype=np.float64) * 100.0
        return AquiferParameters(
            n_nodes=n_nodes,
            n_layers=n_layers,
            kh=kh,
            kv=np.ones((n_nodes, n_layers)),
            specific_storage=np.ones((n_nodes, n_layers)) * 1e-5,
            specific_yield=np.ones((n_nodes, n_layers)) * 0.2,
        )

    def test_anomaly_overwrites_vertex_nodes(self):
        """Anomaly on element 1 should overwrite nodes 1, 2, 5, 4."""
        mesh = self._make_mesh()
        params = self._make_params()

        anomalies = [
            KhAnomalyEntry(element_id=1, kh_per_layer=[50.0, 25.0]),
        ]

        applied = _apply_kh_anomalies(params, anomalies, mesh)

        assert applied == 1

        # node_id_to_idx: node 1->0, 2->1, 3->2, 4->3, 5->4, ...
        # Element 1 vertices: (1, 2, 5, 4) -> indices 0, 1, 4, 3
        assert params.kh[0, 0] == pytest.approx(50.0)  # node 1
        assert params.kh[0, 1] == pytest.approx(25.0)
        assert params.kh[1, 0] == pytest.approx(50.0)  # node 2
        assert params.kh[1, 1] == pytest.approx(25.0)
        assert params.kh[4, 0] == pytest.approx(50.0)  # node 5
        assert params.kh[3, 0] == pytest.approx(50.0)  # node 4

        # Unaffected nodes should remain at 100.0
        assert params.kh[2, 0] == pytest.approx(100.0)  # node 3
        assert params.kh[5, 0] == pytest.approx(100.0)  # node 6
        assert params.kh[6, 0] == pytest.approx(100.0)  # node 7
        assert params.kh[7, 0] == pytest.approx(100.0)  # node 8
        assert params.kh[8, 0] == pytest.approx(100.0)  # node 9

    def test_missing_element_skipped(self):
        """Anomaly for nonexistent element should be skipped."""
        mesh = self._make_mesh()
        params = self._make_params()

        anomalies = [
            KhAnomalyEntry(element_id=999, kh_per_layer=[50.0, 25.0]),
        ]

        applied = _apply_kh_anomalies(params, anomalies, mesh)
        assert applied == 0
        # All values should remain unchanged
        assert np.all(params.kh == 100.0)

    def test_multiple_anomalies(self):
        """Multiple anomalies should all be applied."""
        mesh = self._make_mesh()
        params = self._make_params()

        anomalies = [
            KhAnomalyEntry(element_id=1, kh_per_layer=[50.0, 25.0]),
            KhAnomalyEntry(element_id=4, kh_per_layer=[200.0, 150.0]),
        ]

        applied = _apply_kh_anomalies(params, anomalies, mesh)
        assert applied == 2

        # Node 5 (idx 4) is shared by elements 1 and 4.
        # Element 4 is processed second, so its value should win.
        assert params.kh[4, 0] == pytest.approx(200.0)
        assert params.kh[4, 1] == pytest.approx(150.0)

    def test_no_kh_array(self):
        """Should return 0 if kh is None."""
        mesh = self._make_mesh()
        params = AquiferParameters(n_nodes=9, n_layers=2, kh=None)

        anomalies = [
            KhAnomalyEntry(element_id=1, kh_per_layer=[50.0, 25.0]),
        ]

        applied = _apply_kh_anomalies(params, anomalies, mesh)
        assert applied == 0


# =============================================================================
# Test ParametricGrid
# =============================================================================


class TestParametricGrid:
    """Tests for ParametricGrid FE interpolation."""

    def _triangle_grid(self):
        """Create a single-triangle parametric grid.

        Triangle: (0,0) - (10,0) - (0,10)
        Each node has 1 layer, 2 params.
        """
        nodes = [
            ParamNode(node_id=0, x=0.0, y=0.0,
                      values=np.array([[1.0, 10.0]])),
            ParamNode(node_id=1, x=10.0, y=0.0,
                      values=np.array([[2.0, 20.0]])),
            ParamNode(node_id=2, x=0.0, y=10.0,
                      values=np.array([[3.0, 30.0]])),
        ]
        elements = [ParamElement(elem_id=0, vertices=(0, 1, 2))]
        return ParametricGrid(nodes=nodes, elements=elements)

    def test_triangle_at_vertex(self):
        """Interpolation at vertex should return exact vertex value."""
        grid = self._triangle_grid()

        result = grid.interpolate(0.0, 0.0)
        assert result is not None
        assert result[0, 0] == pytest.approx(1.0)
        assert result[0, 1] == pytest.approx(10.0)

        result = grid.interpolate(10.0, 0.0)
        assert result is not None
        assert result[0, 0] == pytest.approx(2.0)
        assert result[0, 1] == pytest.approx(20.0)

    def test_triangle_at_centroid(self):
        """Interpolation at centroid should return average of vertex values."""
        grid = self._triangle_grid()

        # Centroid of (0,0), (10,0), (0,10) is (10/3, 10/3)
        cx, cy = 10.0 / 3, 10.0 / 3
        result = grid.interpolate(cx, cy)
        assert result is not None
        assert result[0, 0] == pytest.approx((1.0 + 2.0 + 3.0) / 3)
        assert result[0, 1] == pytest.approx((10.0 + 20.0 + 30.0) / 3)

    def test_triangle_outside(self):
        """Point outside triangle should return None."""
        grid = self._triangle_grid()
        result = grid.interpolate(20.0, 20.0)
        assert result is None

    def test_triangle_on_edge(self):
        """Point on triangle edge should be inside."""
        grid = self._triangle_grid()
        # Midpoint of edge (0,0)-(10,0) = (5, 0)
        result = grid.interpolate(5.0, 0.0)
        assert result is not None
        assert result[0, 0] == pytest.approx(1.5)  # average of 1.0 and 2.0

    def _quad_grid(self):
        """Create a single-quad parametric grid.

        Quad: (0,0) - (10,0) - (10,10) - (0,10)
        Each node has 1 layer, 1 param.
        """
        nodes = [
            ParamNode(node_id=0, x=0.0, y=0.0,
                      values=np.array([[1.0]])),
            ParamNode(node_id=1, x=10.0, y=0.0,
                      values=np.array([[2.0]])),
            ParamNode(node_id=2, x=10.0, y=10.0,
                      values=np.array([[3.0]])),
            ParamNode(node_id=3, x=0.0, y=10.0,
                      values=np.array([[4.0]])),
        ]
        elements = [ParamElement(elem_id=0, vertices=(0, 1, 2, 3))]
        return ParametricGrid(nodes=nodes, elements=elements)

    def test_quad_at_vertex(self):
        """Interpolation at quad vertex should return exact value."""
        grid = self._quad_grid()
        result = grid.interpolate(0.0, 0.0)
        assert result is not None
        assert result[0, 0] == pytest.approx(1.0)

        result = grid.interpolate(10.0, 10.0)
        assert result is not None
        assert result[0, 0] == pytest.approx(3.0)

    def test_quad_at_center(self):
        """Point at quad center should be inside and interpolated."""
        grid = self._quad_grid()
        result = grid.interpolate(5.0, 5.0)
        assert result is not None
        # The center of the quad should give a reasonable interpolation.
        # With the triangle-split approach, the center is on the edge
        # between the two sub-triangles.
        assert 1.0 <= result[0, 0] <= 4.0

    def test_quad_outside(self):
        """Point outside quad should return None."""
        grid = self._quad_grid()
        result = grid.interpolate(-5.0, -5.0)
        assert result is None

    def test_multiple_elements(self):
        """Grid with multiple elements should find correct element."""
        # Two triangles sharing an edge
        nodes = [
            ParamNode(node_id=0, x=0.0, y=0.0,
                      values=np.array([[10.0]])),
            ParamNode(node_id=1, x=10.0, y=0.0,
                      values=np.array([[20.0]])),
            ParamNode(node_id=2, x=5.0, y=10.0,
                      values=np.array([[30.0]])),
            ParamNode(node_id=3, x=5.0, y=-10.0,
                      values=np.array([[40.0]])),
        ]
        elements = [
            ParamElement(elem_id=0, vertices=(0, 1, 2)),  # Upper triangle
            ParamElement(elem_id=1, vertices=(0, 3, 1)),  # Lower triangle
        ]
        grid = ParametricGrid(nodes=nodes, elements=elements)

        # Point in upper triangle
        result = grid.interpolate(5.0, 5.0)
        assert result is not None

        # Point in lower triangle
        result = grid.interpolate(5.0, -5.0)
        assert result is not None

        # Point outside both
        result = grid.interpolate(-10.0, 0.0)
        assert result is None


# =============================================================================
# Test ParametricGridData dataclass
# =============================================================================


class TestParametricGridData:
    """Tests for ParametricGridData."""

    def test_creation(self):
        """Basic creation of ParametricGridData."""
        data = ParametricGridData(
            n_nodes=3,
            n_elements=1,
            elements=[(0, 1, 2)],
            node_coords=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            node_values=np.zeros((3, 2, 5)),
        )
        assert data.n_nodes == 3
        assert data.n_elements == 1
        assert data.node_values.shape == (3, 2, 5)


# =============================================================================
# Test GW main file with anomaly section (integration-style)
# =============================================================================


class TestGWMainFileWithAnomaly:
    """Test reading a GW main file that includes Kh anomaly section."""

    def test_anomaly_stored_in_config(self, tmp_path: Path):
        """Anomaly data should be stored in config.kh_anomalies."""
        # Minimal GW main file with version, file paths, output factors,
        # hydrographs, face flow, aquifer params, and anomaly section.
        # NOUTH must be >0 for reader to continue past hydrograph section.
        gw_content = """\
C  GW Main File
#4.0
C  BCFL

C  TDFL

C  PUMPFL

C  SUBSFL

C  OVRWRTFL

C  FACTLTOU
1.0
C  UNITLTOU
FEET
C  FACTVLOU
1.0
C  UNITVLOU
TAF
C  FACTVROU
1.0
C  UNITVROU
FT/DAY
C  VELOUTFL

C  VFLOWOUTFL

C  GWALLOUTFL

C  HTPOUTFL

C  VTPOUTFL

C  GWBUDFL

C  ZBUDFL

C  FNGWFL

C  KDEB
0
C  NOUTH
1
C  FACTXY
1.0
C  GWHYDOUTFL

C  Hydrograph data: ID  HYDTYP  LAYER  NODE  NAME
1  1  1  1  OBS1
C  NOUTF
0
C  FCHYDOUTFL

C  ---- AQUIFER PARAMETERS ----
C  NGROUP
0
C  FX  FKH  FS  FN  FV  FL
1.0  1.0  1.0  1.0  1.0  1.0
C  TUNITKH
1DAY
C  TUNITV
1DAY
C  TUNITL
1DAY
C  Node data: ID  PKH  PS  PN  PV  PL
1  100.0  1.0E-05  0.2  0.01  50.0
2  100.0  1.0E-05  0.2  0.01  50.0
3  100.0  1.0E-05  0.2  0.01  50.0
4  100.0  1.0E-05  0.2  0.01  50.0
C  ---- ANOMALY ----
2
1.0
1DAY
1   1   50.0
2   2   75.0
C  ---- INITIAL HEADS ----
1.0
1  10.0
2  10.0
3  10.0
4  10.0
"""
        gw_file = tmp_path / "gw_main.dat"
        gw_file.write_text(gw_content)

        reader = GWMainFileReader()
        config = reader.read(gw_file)

        # Aquifer params should be loaded
        assert config.aquifer_params is not None
        assert config.aquifer_params.n_nodes == 4
        assert config.aquifer_params.n_layers == 1

        # Anomalies should be parsed
        assert len(config.kh_anomalies) == 2
        assert config.kh_anomalies[0].element_id == 1
        assert config.kh_anomalies[0].kh_per_layer == pytest.approx([50.0])
        assert config.kh_anomalies[1].element_id == 2
        assert config.kh_anomalies[1].kh_per_layer == pytest.approx([75.0])

        # Initial heads should still be loaded (after anomaly)
        assert config.initial_heads is not None
        assert config.initial_heads.shape == (4, 1)
