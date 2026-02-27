"""Deep tests for pyiwfm.io.gw_subsidence targeting uncovered reading paths.

Covers:
- v5.0 format (7 factors, interbed_dz, Kv/NEQ columns)
- v4.0 direct parameter reading
- Parametric grids reading
- IC file reading
- Missing IC file
- Hydrograph output section
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyiwfm.io.gw_subsidence import (
    SubsidenceConfig,
    SubsidenceReader,
    read_gw_subsidence,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_v40_file(path: Path, n_nodes: int, n_layers: int) -> None:
    """Write a minimal v4.0 subsidence parameter file."""
    lines = [
        "#4.0",
        "                                        / ICFL (IC file)",
        "                                        / TECPLOTFL",
        "                                        / FINSUBSFL",
        "  1.0                                   / FACTSUBS",
        "  FEET                                  / UNITSUBS",
        "  0                                     / NOUTS",
        "  0                                     / NGROUP",
        "  1.0  1.0  1.0  1.0  1.0  1.0         / Factors: FX FSE FSI FINT FINTMIN FPC",
    ]
    # Per-node data: ID + 5 params * n_layers
    for nid in range(1, n_nodes + 1):
        for layer in range(n_layers):
            if layer == 0:
                line = f"  {nid}  0.001  0.01  10.0  1.0  50.0"
            else:
                line = "  0.002  0.02  20.0  2.0  60.0"
            lines.append(line)

    path.write_text("\n".join(lines) + "\n")


def _write_v50_file(path: Path, n_nodes: int, n_layers: int) -> None:
    """Write a minimal v5.0 subsidence parameter file."""
    lines = [
        "#5.0",
        "                                        / ICFL (IC file)",
        "                                        / TECPLOTFL",
        "                                        / FINSUBSFL",
        "  1.0                                   / FACTSUBS",
        "  FEET                                  / UNITSUBS",
        "  0                                     / NOUTS",
        "  5.0                                   / INTERBED_DZ (v5.0)",
        "  0                                     / NGROUP",
        "  1.0  1.0  1.0  1.0  1.0  1.0  1.0    / Factors (7 for v5.0)",
    ]
    # Per-node data: ID + 7 params * n_layers (v5.0 adds Kv, NEQ)
    for nid in range(1, n_nodes + 1):
        for layer in range(n_layers):
            if layer == 0:
                line = f"  {nid}  0.001  0.01  10.0  1.0  50.0  0.5  3"
            else:
                line = "  0.002  0.02  20.0  2.0  60.0  0.8  5"
            lines.append(line)

    path.write_text("\n".join(lines) + "\n")


def _write_ic_file(path: Path, n_nodes: int, n_layers: int) -> None:
    """Write a subsidence IC file."""
    lines = ["  1.0"]  # Conversion factor
    for nid in range(1, n_nodes + 1):
        # ID + InterbedThick(NLayers) + PreCompactHead(NLayers)
        vals = [str(nid)]
        for _l in range(n_layers):
            vals.append("5.0")  # interbed thick
        for _l in range(n_layers):
            vals.append("40.0")  # precompact head
        lines.append("  " + "  ".join(vals))
    path.write_text("\n".join(lines) + "\n")


def _write_parametric_file(path: Path) -> None:
    """Write a subsidence file with parametric grid data."""
    lines = [
        "#4.0",
        "                                        / ICFL",
        "                                        / TECPLOTFL",
        "                                        / FINSUBSFL",
        "  1.0                                   / FACTSUBS",
        "  FEET                                  / UNITSUBS",
        "  0                                     / NOUTS",
        "  1                                     / NGROUP",
        "  1.0  1.0  1.0  1.0  1.0  1.0         / Factors",
        "  1-3                                   / NodeRange",
        "  3                                     / NDP",
        "  0                                     / NEP",
        # 3 parametric nodes with 5 params per layer (1 layer)
        "  1  0.0  0.0  0.001  0.01  10.0  1.0  50.0",
        "  2  100.0  0.0  0.002  0.02  20.0  2.0  60.0",
        "  3  50.0  86.6  0.003  0.03  30.0  3.0  70.0",
        "C end of parametric data",
    ]
    path.write_text("\n".join(lines) + "\n")


def _write_hydrograph_file(path: Path) -> None:
    """Write a subsidence file with hydrograph output specs."""
    lines = [
        "#4.0",
        "                                        / ICFL",
        "                                        / TECPLOTFL",
        "                                        / FINSUBSFL",
        "  1.0                                   / FACTSUBS",
        "  FEET                                  / UNITSUBS",
        "  2                                     / NOUTS",
        "  1.0                                   / FACTXY",
        "  ../Results/SubsHyd.out                / SUBHYDOUTFL",
        "  1  0  1  500.0  600.0  Obs1",
        "  2  1  2  42  ObsNode",
        "  0                                     / NGROUP",
        "  1.0  1.0  1.0  1.0  1.0  1.0         / Factors",
    ]
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestV40Format:
    """Tests for v4.0 subsidence file reading."""

    def test_read_v40_direct_params(self, tmp_path: Path) -> None:
        """Read v4.0 with direct (non-parametric) node parameters."""
        fpath = tmp_path / "Subsidence.dat"
        _write_v40_file(fpath, n_nodes=3, n_layers=2)

        config = read_gw_subsidence(fpath, n_nodes=3, n_layers=2)

        assert config.version == "4.0"
        assert len(config.node_params) == 3
        assert len(config.node_params[0].elastic_sc) == 2
        assert len(config.node_params[0].inelastic_sc) == 2
        assert len(config.node_params[0].interbed_thick) == 2
        assert config.node_params[0].node_id == 1
        assert config.interbed_dz == 0.0  # Not present in v4.0

    def test_read_v40_single_layer(self, tmp_path: Path) -> None:
        """Read v4.0 with a single layer."""
        fpath = tmp_path / "Subsidence.dat"
        _write_v40_file(fpath, n_nodes=2, n_layers=1)

        config = read_gw_subsidence(fpath, n_nodes=2, n_layers=1)

        assert len(config.node_params) == 2
        assert len(config.node_params[0].elastic_sc) == 1


class TestV50Format:
    """Tests for v5.0 subsidence file reading."""

    def test_read_v50_with_kv_neq(self, tmp_path: Path) -> None:
        """Read v5.0 format with Kv and NEQ columns."""
        fpath = tmp_path / "Subsidence.dat"
        _write_v50_file(fpath, n_nodes=3, n_layers=2)

        config = read_gw_subsidence(fpath, n_nodes=3, n_layers=2)

        assert config.version == "5.0"
        assert config.interbed_dz == 5.0
        assert len(config.conversion_factors) == 7
        assert len(config.node_params) == 3
        # v5.0 extras
        assert len(config.node_params[0].kv_sub) == 2
        assert len(config.node_params[0].n_eq) == 2
        assert config.node_params[0].kv_sub[0] == pytest.approx(0.5)


class TestICFile:
    """Tests for IC file reading and missing IC file handling."""

    def test_read_with_ic_file(self, tmp_path: Path) -> None:
        """IC file is parsed when present."""
        subs_path = tmp_path / "Subsidence.dat"
        ic_path = tmp_path / "SubsidenceIC.dat"
        _write_ic_file(ic_path, n_nodes=3, n_layers=1)

        lines = [
            "#4.0",
            f"  {ic_path.name}",
            "                                        / TECPLOTFL",
            "                                        / FINSUBSFL",
            "  1.0                                   / FACTSUBS",
            "  FEET                                  / UNITSUBS",
            "  0                                     / NOUTS",
            "  0                                     / NGROUP",
            "  1.0  1.0  1.0  1.0  1.0  1.0         / Factors",
        ]
        for nid in range(1, 4):
            lines.append(f"  {nid}  0.001  0.01  10.0  1.0  50.0")
        subs_path.write_text("\n".join(lines) + "\n")

        config = read_gw_subsidence(subs_path, base_dir=tmp_path, n_nodes=3, n_layers=1)

        assert config.ic_file is not None
        assert config.ic_interbed_thick is not None
        assert config.ic_interbed_thick.shape == (3, 1)
        assert config.ic_precompact_head is not None
        np.testing.assert_allclose(config.ic_interbed_thick[:, 0], 5.0)
        np.testing.assert_allclose(config.ic_precompact_head[:, 0], 40.0)

    def test_missing_ic_file_skipped(self, tmp_path: Path) -> None:
        """When IC file path is specified but doesn't exist, skip gracefully."""
        subs_path = tmp_path / "Subsidence.dat"
        lines = [
            "#4.0",
            "  NonExistent_IC.dat",
            "                                        / TECPLOTFL",
            "                                        / FINSUBSFL",
            "  1.0                                   / FACTSUBS",
            "  FEET                                  / UNITSUBS",
            "  0                                     / NOUTS",
            "  0                                     / NGROUP",
            "  1.0  1.0  1.0  1.0  1.0  1.0         / Factors",
        ]
        for nid in range(1, 3):
            lines.append(f"  {nid}  0.001  0.01  10.0  1.0  50.0")
        subs_path.write_text("\n".join(lines) + "\n")

        config = read_gw_subsidence(subs_path, base_dir=tmp_path, n_nodes=2, n_layers=1)

        # IC data should be None since the file doesn't exist
        assert config.ic_interbed_thick is None
        assert config.ic_precompact_head is None


class TestParametricGrids:
    """Tests for parametric grid subsidence data."""

    def test_read_parametric(self, tmp_path: Path) -> None:
        """Parametric grid data is parsed when NGROUP > 0."""
        fpath = tmp_path / "Subsidence.dat"
        _write_parametric_file(fpath)

        config = read_gw_subsidence(fpath, n_nodes=3, n_layers=1)

        assert config.n_parametric_grids == 1
        assert len(config.parametric_grids) == 1
        grid = config.parametric_grids[0]
        assert grid.n_nodes == 3
        assert grid.n_elements == 0
        assert grid.node_coords.shape == (3, 2)
        # First node at (0, 0), second at (100, 0)
        np.testing.assert_allclose(grid.node_coords[0], [0.0, 0.0])
        np.testing.assert_allclose(grid.node_coords[1], [100.0, 0.0])


class TestHydrographOutput:
    """Tests for hydrograph output section parsing."""

    def test_read_hydrograph_specs(self, tmp_path: Path) -> None:
        """Hydrograph specs with SUBTYP=0 and SUBTYP=1 are parsed."""
        fpath = tmp_path / "Subsidence.dat"
        _write_hydrograph_file(fpath)

        config = read_gw_subsidence(fpath, n_nodes=0, n_layers=0)

        assert config.n_hydrograph_outputs == 2
        assert len(config.hydrograph_specs) == 2

        # First spec: SUBTYP=0 (x-y coords)
        s0 = config.hydrograph_specs[0]
        assert s0.hydtyp == 0
        assert s0.x == pytest.approx(500.0)
        assert s0.y == pytest.approx(600.0)

        # Second spec: SUBTYP=1 (node number)
        s1 = config.hydrograph_specs[1]
        assert s1.hydtyp == 1
        assert s1.layer == 2
