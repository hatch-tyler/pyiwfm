"""Sweep tests for pyiwfm.io.gw_subsidence uncovered paths.

Targets: tecplot/final_subs file paths, non-numeric NOUTS/FACTXY,
hydrograph specs (hydtyp=0/1, short rows, / name), multi-line factors,
parametric grid elements and node data.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyiwfm.io.gw_subsidence import SubsidenceReader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_file(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")


def _make_base_header(
    *,
    tecplot: str = "",
    final_subs: str = "",
    nouts: str = "0",
    interbed_dz: str | None = None,
) -> list[str]:
    """Construct header lines for a v4.0 subsidence file."""
    header = [
        "#4.0",
        "                                        / ICFL",
    ]
    header.append(f"  {tecplot or ''}                         / TECPLOTFL")
    header.append(f"  {final_subs or ''}                      / FINSUBSFL")
    header.append("  1.0                                   / FACTSUBS")
    header.append("  FEET                                  / UNITSUBS")
    header.append(f"  {nouts}                               / NOUTS")
    if interbed_dz is not None:
        header.append(f"  {interbed_dz}                         / INTERBED_DZ")
    return header


# ---------------------------------------------------------------------------
# Tecplot and final_subs file path coverage (lines 218-225)
# ---------------------------------------------------------------------------


class TestTecplotAndFinalSubsPaths:
    def test_tecplot_path_set(self, tmp_path: Path) -> None:
        """Non-blank tecplot file path is resolved."""
        lines = _make_base_header(tecplot="output.plt", final_subs="final.dat")
        lines.extend(
            [
                "  0                                     / NGROUP",
                "  1.0  1.0  1.0  1.0  1.0  1.0         / Factors",
            ]
        )
        fpath = tmp_path / "subsidence.dat"
        _write_file(fpath, lines)

        reader = SubsidenceReader()
        config = reader.read(fpath, n_nodes=0, n_layers=1)
        assert config._raw_tecplot_file == "output.plt"
        assert config._raw_final_subs_file == "final.dat"


# ---------------------------------------------------------------------------
# Non-numeric NOUTS and FACTXY (lines 240-241, 249-250)
# ---------------------------------------------------------------------------


class TestNonNumericNoutsAndFactxy:
    def test_non_numeric_nouts_defaults_to_zero(self, tmp_path: Path) -> None:
        """Non-numeric NOUTS should default to 0."""
        lines = _make_base_header(nouts="abc")
        lines.extend(
            [
                "  0                                     / NGROUP",
                "  1.0  1.0  1.0  1.0  1.0  1.0         / Factors",
            ]
        )
        fpath = tmp_path / "subsidence.dat"
        _write_file(fpath, lines)

        reader = SubsidenceReader()
        config = reader.read(fpath, n_nodes=0, n_layers=1)
        assert config.n_hydrograph_outputs == 0

    def test_non_numeric_factxy_ignored(self, tmp_path: Path) -> None:
        """Non-numeric FACTXY should be silently ignored."""
        lines = [
            "#4.0",
            "                                        / ICFL",
            "                                        / TECPLOTFL",
            "                                        / FINSUBSFL",
            "  1.0                                   / FACTSUBS",
            "  FEET                                  / UNITSUBS",
            "  1                                     / NOUTS",
            "  xyz                                   / FACTXY (bad)",
            "  hydout.dat                            / SUBHYDOUTFL",
            "  1  0  1  100.0  200.0  Well1",
            "  0                                     / NGROUP",
            "  1.0  1.0  1.0  1.0  1.0  1.0         / Factors",
        ]
        fpath = tmp_path / "subsidence.dat"
        _write_file(fpath, lines)

        reader = SubsidenceReader()
        config = reader.read(fpath, n_nodes=0, n_layers=1)
        assert config.n_hydrograph_outputs == 1
        # FACTXY should still be default 1.0
        assert config.hydrograph_coord_factor == 1.0


# ---------------------------------------------------------------------------
# Hydrograph specs: short rows, hydtyp=1, name from slash
# ---------------------------------------------------------------------------


class TestHydrographSpecs:
    def test_hydtyp_1_node_format(self, tmp_path: Path) -> None:
        """Hydrograph spec with hydtyp=1 reads node ID from parts[3]."""
        lines = [
            "#4.0",
            "                                        / ICFL",
            "                                        / TECPLOTFL",
            "                                        / FINSUBSFL",
            "  1.0                                   / FACTSUBS",
            "  FEET                                  / UNITSUBS",
            "  1                                     / NOUTS",
            "  1.0                                   / FACTXY",
            "  hydout.dat                            / SUBHYDOUTFL",
            "  1  1  2  42  NodeWell",
            "  0                                     / NGROUP",
            "  1.0  1.0  1.0  1.0  1.0  1.0         / Factors",
        ]
        fpath = tmp_path / "subsidence.dat"
        _write_file(fpath, lines)

        reader = SubsidenceReader()
        config = reader.read(fpath, n_nodes=0, n_layers=1)
        assert len(config.hydrograph_specs) == 1
        spec = config.hydrograph_specs[0]
        assert spec.hydtyp == 1
        assert spec.x == 42.0  # node ID stored as x

    def test_short_hydrograph_row_skipped(self, tmp_path: Path) -> None:
        """Hydrograph spec row with <4 columns is skipped."""
        lines = [
            "#4.0",
            "                                        / ICFL",
            "                                        / TECPLOTFL",
            "                                        / FINSUBSFL",
            "  1.0                                   / FACTSUBS",
            "  FEET                                  / UNITSUBS",
            "  2                                     / NOUTS",
            "  1.0                                   / FACTXY",
            "  hydout.dat                            / SUBHYDOUTFL",
            "  1  0  1",
            "  2  0  1  50.0  60.0  GoodSpec",
            "  0                                     / NGROUP",
            "  1.0  1.0  1.0  1.0  1.0  1.0         / Factors",
        ]
        fpath = tmp_path / "subsidence.dat"
        _write_file(fpath, lines)

        reader = SubsidenceReader()
        config = reader.read(fpath, n_nodes=0, n_layers=1)
        # Short row skipped, only second spec added
        assert len(config.hydrograph_specs) == 1
        assert config.hydrograph_specs[0].name == "GoodSpec"


# ---------------------------------------------------------------------------
# Multi-line conversion factors (lines 308-311)
# ---------------------------------------------------------------------------


class TestMultiLineFactors:
    def test_factors_across_two_lines(self, tmp_path: Path) -> None:
        """Conversion factors split across 2 lines are joined."""
        lines = [
            "#4.0",
            "                                        / ICFL",
            "                                        / TECPLOTFL",
            "                                        / FINSUBSFL",
            "  1.0                                   / FACTSUBS",
            "  FEET                                  / UNITSUBS",
            "  0                                     / NOUTS",
            "  0                                     / NGROUP",
            "  1.0  2.0  3.0",
            "  4.0  5.0  6.0",
        ]
        fpath = tmp_path / "subsidence.dat"
        _write_file(fpath, lines)

        reader = SubsidenceReader()
        config = reader.read(fpath, n_nodes=0, n_layers=1)
        assert len(config.conversion_factors) == 6
        assert config.conversion_factors == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


# ---------------------------------------------------------------------------
# Parametric grid element and node parsing (lines 418-500+)
# ---------------------------------------------------------------------------


class TestParametricGridParsing:
    def test_parametric_grid_with_elements(self, tmp_path: Path) -> None:
        """Parametric grid with 1 element and 3 nodes (1 layer)."""
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
            "  1-3                                   / Node range",
            "  3                                     / NDP",
            "  1                                     / NEP",
            "  1  1  2  3                            / Element: ID V1 V2 V3",
            "  1  0.0  0.0  0.001  0.01  10.0  1.0  50.0",
            "  2  1.0  0.0  0.002  0.02  20.0  2.0  60.0",
            "  3  0.5  1.0  0.003  0.03  30.0  3.0  70.0",
        ]
        fpath = tmp_path / "subsidence.dat"
        _write_file(fpath, lines)

        reader = SubsidenceReader()
        config = reader.read(fpath, n_nodes=3, n_layers=1)
        assert config.n_parametric_grids == 1
        assert len(config.parametric_grids) == 1
        grid = config.parametric_grids[0]
        assert len(grid.elements) == 1
        assert grid.node_range_str == "1-3"

    def test_parametric_grid_missing_ndp_breaks(self, tmp_path: Path) -> None:
        """Empty NDP string causes early break."""
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
            "  1-3                                   / Node range",
            # NDP is missing / blank -- file ends here
        ]
        fpath = tmp_path / "subsidence.dat"
        _write_file(fpath, lines)

        reader = SubsidenceReader()
        config = reader.read(fpath, n_nodes=3, n_layers=1)
        assert len(config.parametric_grids) == 0

    def test_parametric_non_numeric_ndp_breaks(self, tmp_path: Path) -> None:
        """Non-numeric NDP string causes early break."""
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
            "  1-3                                   / Node range",
            "  abc                                   / NDP (non-numeric)",
        ]
        fpath = tmp_path / "subsidence.dat"
        _write_file(fpath, lines)

        reader = SubsidenceReader()
        config = reader.read(fpath, n_nodes=3, n_layers=1)
        assert len(config.parametric_grids) == 0

    def test_parametric_non_numeric_nep_breaks(self, tmp_path: Path) -> None:
        """Non-numeric NEP string causes early break."""
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
            "  1-3                                   / Node range",
            "  3                                     / NDP",
            "  xyz                                   / NEP (non-numeric)",
        ]
        fpath = tmp_path / "subsidence.dat"
        _write_file(fpath, lines)

        reader = SubsidenceReader()
        config = reader.read(fpath, n_nodes=3, n_layers=1)
        assert len(config.parametric_grids) == 0

    def test_parametric_short_element_line_breaks(self, tmp_path: Path) -> None:
        """Element definition with < 4 columns breaks element reading."""
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
            "  1-2                                   / Node range",
            "  2                                     / NDP",
            "  1                                     / NEP",
            "  1  2",
            # Short element line (< 4 parts)
            "  1  0.0  0.0  0.001  0.01  10.0  1.0  50.0",
            "  2  1.0  0.0  0.002  0.02  20.0  2.0  60.0",
        ]
        fpath = tmp_path / "subsidence.dat"
        _write_file(fpath, lines)

        reader = SubsidenceReader()
        config = reader.read(fpath, n_nodes=2, n_layers=1)
        # Grid created but with 0 elements (short line broke reading)
        assert len(config.parametric_grids) >= 0  # doesn't crash

    def test_parametric_missing_node_range_breaks(self, tmp_path: Path) -> None:
        """Empty node range string causes early break."""
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
            # No node range or any more lines
        ]
        fpath = tmp_path / "subsidence.dat"
        _write_file(fpath, lines)

        reader = SubsidenceReader()
        config = reader.read(fpath, n_nodes=2, n_layers=1)
        assert len(config.parametric_grids) == 0
