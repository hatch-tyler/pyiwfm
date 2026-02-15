"""Tests for pyiwfm.io.gw_boundary_writer module.

Covers all five writer functions and the two private helpers,
exercising both empty and populated BC lists, factor division logic,
the zero-factor fallback branches, and the NOUTB output section.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyiwfm.io.gw_boundary import (
    ConstrainedGeneralHeadBC,
    GeneralHeadBC,
    GWBoundaryConfig,
    SpecifiedFlowBC,
    SpecifiedHeadBC,
)
from pyiwfm.io.gw_boundary_writer import (
    _write_comment,
    _write_value,
    write_bc_main,
    write_constrained_gh_bc,
    write_general_head_bc,
    write_specified_flow_bc,
    write_specified_head_bc,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> GWBoundaryConfig:
    """Build a GWBoundaryConfig with sensible defaults, overridden by kwargs."""
    defaults = dict(
        sp_flow_file=Path("flow.dat"),
        sp_head_file=Path("head.dat"),
        gh_file=Path("gh.dat"),
        cgh_file=Path("cgh.dat"),
        ts_data_file=Path("ts.dat"),
        n_bc_output_nodes=0,
        bc_output_file=None,
        bc_output_specs=[],
        specified_flow_bcs=[],
        sp_flow_factor=1.0,
        sp_flow_time_unit="1DAY",
        specified_head_bcs=[],
        sp_head_factor=1.0,
        general_head_bcs=[],
        gh_head_factor=1.0,
        gh_conductance_factor=1.0,
        gh_time_unit="1DAY",
        constrained_gh_bcs=[],
        cgh_head_factor=1.0,
        cgh_max_flow_factor=1.0,
        cgh_head_time_unit="1DAY",
        cgh_conductance_factor=1.0,
        cgh_conductance_time_unit="1DAY",
    )
    defaults.update(overrides)
    return GWBoundaryConfig(**defaults)


# ---------------------------------------------------------------------------
# _write_comment / _write_value
# ---------------------------------------------------------------------------


class TestWriteComment:
    def test_basic(self, tmp_path: Path) -> None:
        out = tmp_path / "comment.txt"
        with open(out, "w") as f:
            _write_comment(f, "hello world")
        text = out.read_text()
        assert text == "C  hello world\n"


class TestWriteValue:
    def test_with_description(self, tmp_path: Path) -> None:
        out = tmp_path / "val.txt"
        with open(out, "w") as f:
            _write_value(f, 42, "answer")
        text = out.read_text()
        assert "42" in text
        assert "/ answer" in text

    def test_without_description(self, tmp_path: Path) -> None:
        out = tmp_path / "val.txt"
        with open(out, "w") as f:
            _write_value(f, "something")
        text = out.read_text()
        assert "something" in text
        assert "/" not in text


# ---------------------------------------------------------------------------
# write_bc_main
# ---------------------------------------------------------------------------


class TestWriteBcMain:
    def test_minimal_config(self, tmp_path: Path) -> None:
        """Config with no output nodes and all sub-file paths set."""
        cfg = _make_config()
        result = write_bc_main(cfg, tmp_path / "bc_main.dat")
        assert result.exists()
        text = result.read_text()
        # Comment header
        assert "C  IWFM Boundary Conditions Main File" in text
        # Five sub-file path lines
        assert "flow.dat" in text
        assert "head.dat" in text
        assert "gh.dat" in text
        assert "cgh.dat" in text
        assert "ts.dat" in text
        # NOUTB = 0 means no output nodes section body
        assert "NOUTB" in text

    def test_with_none_sub_files(self, tmp_path: Path) -> None:
        """None sub-file paths should be written as empty strings."""
        cfg = _make_config(
            sp_flow_file=None,
            sp_head_file=None,
            gh_file=None,
            cgh_file=None,
            ts_data_file=None,
        )
        result = write_bc_main(cfg, tmp_path / "bc_main.dat")
        text = result.read_text()
        # The "or ''" fallback means empty string values
        lines = text.strip().splitlines()
        # Should still have the comment, 5 path lines, and NOUTB line
        assert len(lines) >= 7

    def test_with_output_nodes(self, tmp_path: Path) -> None:
        """NOUTB > 0 triggers output file and node spec lines."""
        cfg = _make_config(
            n_bc_output_nodes=3,
            bc_output_file=Path("bc_out.hyd"),
            bc_output_specs=[10, 20, 30],
        )
        result = write_bc_main(cfg, tmp_path / "bc_main.dat")
        text = result.read_text()
        assert "3" in text  # NOUTB value
        assert "bc_out.hyd" in text
        assert "10" in text
        assert "20" in text
        assert "30" in text

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        deep = tmp_path / "a" / "b" / "c" / "bc_main.dat"
        cfg = _make_config()
        result = write_bc_main(cfg, deep)
        assert result.exists()

    def test_returns_path_object(self, tmp_path: Path) -> None:
        cfg = _make_config()
        result = write_bc_main(cfg, str(tmp_path / "test.dat"))
        assert isinstance(result, Path)

    def test_output_nodes_with_none_output_file(self, tmp_path: Path) -> None:
        """NOUTB > 0 but bc_output_file is None should write empty string."""
        cfg = _make_config(
            n_bc_output_nodes=1,
            bc_output_file=None,
            bc_output_specs=[99],
        )
        result = write_bc_main(cfg, tmp_path / "bc_main.dat")
        text = result.read_text()
        assert "BHYDOUTFL" in text
        assert "99" in text


# ---------------------------------------------------------------------------
# write_specified_flow_bc
# ---------------------------------------------------------------------------


class TestWriteSpecifiedFlowBC:
    def test_empty_bcs(self, tmp_path: Path) -> None:
        cfg = _make_config(specified_flow_bcs=[])
        result = write_specified_flow_bc(cfg, tmp_path / "flow.dat")
        text = result.read_text()
        assert "NQB" in text
        # Count should be 0
        lines = text.strip().splitlines()
        # Should have comment + NQB line only (no FACT / TUNIT)
        assert len(lines) == 2

    def test_with_bcs_factor_one(self, tmp_path: Path) -> None:
        bcs = [
            SpecifiedFlowBC(node_id=1, layer=1, ts_column=0, base_flow=100.0),
            SpecifiedFlowBC(node_id=5, layer=2, ts_column=3, base_flow=200.0),
        ]
        cfg = _make_config(
            specified_flow_bcs=bcs,
            sp_flow_factor=1.0,
            sp_flow_time_unit="1DAY",
        )
        result = write_specified_flow_bc(cfg, tmp_path / "flow.dat")
        text = result.read_text()
        assert "2" in text  # NQB
        assert "FACT" in text
        assert "TUNIT" in text
        # flow = base_flow / factor = 100 / 1 = 100
        assert "100.0000" in text
        assert "200.0000" in text

    def test_factor_division(self, tmp_path: Path) -> None:
        """base_flow is divided by sp_flow_factor before writing."""
        bcs = [SpecifiedFlowBC(node_id=10, layer=1, ts_column=0, base_flow=500.0)]
        cfg = _make_config(specified_flow_bcs=bcs, sp_flow_factor=2.0)
        result = write_specified_flow_bc(cfg, tmp_path / "flow.dat")
        text = result.read_text()
        # 500 / 2 = 250
        assert "250.0000" in text

    def test_zero_factor_fallback(self, tmp_path: Path) -> None:
        """When factor is 0, the raw base_flow is written (no division)."""
        bcs = [SpecifiedFlowBC(node_id=10, layer=1, ts_column=0, base_flow=123.0)]
        cfg = _make_config(specified_flow_bcs=bcs, sp_flow_factor=0.0)
        result = write_specified_flow_bc(cfg, tmp_path / "flow.dat")
        text = result.read_text()
        assert "123.0000" in text

    def test_node_layer_ts_written(self, tmp_path: Path) -> None:
        bcs = [SpecifiedFlowBC(node_id=42, layer=3, ts_column=7, base_flow=0.0)]
        cfg = _make_config(specified_flow_bcs=bcs)
        result = write_specified_flow_bc(cfg, tmp_path / "flow.dat")
        text = result.read_text()
        assert "42" in text
        assert "3" in text
        assert "7" in text

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        cfg = _make_config()
        result = write_specified_flow_bc(cfg, tmp_path / "sub" / "flow.dat")
        assert result.exists()


# ---------------------------------------------------------------------------
# write_specified_head_bc
# ---------------------------------------------------------------------------


class TestWriteSpecifiedHeadBC:
    def test_empty_bcs(self, tmp_path: Path) -> None:
        cfg = _make_config(specified_head_bcs=[])
        result = write_specified_head_bc(cfg, tmp_path / "head.dat")
        text = result.read_text()
        assert "NHB" in text
        lines = text.strip().splitlines()
        assert len(lines) == 2  # comment + NHB

    def test_with_bcs_factor_one(self, tmp_path: Path) -> None:
        bcs = [
            SpecifiedHeadBC(node_id=1, layer=1, ts_column=0, head_value=50.0),
            SpecifiedHeadBC(node_id=2, layer=1, ts_column=1, head_value=75.0),
        ]
        cfg = _make_config(specified_head_bcs=bcs, sp_head_factor=1.0)
        result = write_specified_head_bc(cfg, tmp_path / "head.dat")
        text = result.read_text()
        assert "NHB" in text
        assert "FACT" in text
        assert "50.0000" in text
        assert "75.0000" in text

    def test_factor_division(self, tmp_path: Path) -> None:
        bcs = [SpecifiedHeadBC(node_id=1, layer=1, ts_column=0, head_value=300.0)]
        cfg = _make_config(specified_head_bcs=bcs, sp_head_factor=3.0)
        result = write_specified_head_bc(cfg, tmp_path / "head.dat")
        text = result.read_text()
        # 300 / 3 = 100
        assert "100.0000" in text

    def test_zero_factor_fallback(self, tmp_path: Path) -> None:
        bcs = [SpecifiedHeadBC(node_id=1, layer=1, ts_column=0, head_value=88.0)]
        cfg = _make_config(specified_head_bcs=bcs, sp_head_factor=0.0)
        result = write_specified_head_bc(cfg, tmp_path / "head.dat")
        text = result.read_text()
        assert "88.0000" in text

    def test_node_layer_ts_written(self, tmp_path: Path) -> None:
        bcs = [SpecifiedHeadBC(node_id=99, layer=4, ts_column=12, head_value=0.0)]
        cfg = _make_config(specified_head_bcs=bcs)
        result = write_specified_head_bc(cfg, tmp_path / "head.dat")
        text = result.read_text()
        assert "99" in text
        assert "4" in text
        assert "12" in text

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        cfg = _make_config()
        result = write_specified_head_bc(cfg, tmp_path / "d1" / "d2" / "head.dat")
        assert result.exists()


# ---------------------------------------------------------------------------
# write_general_head_bc
# ---------------------------------------------------------------------------


class TestWriteGeneralHeadBC:
    def test_empty_bcs(self, tmp_path: Path) -> None:
        cfg = _make_config(general_head_bcs=[])
        result = write_general_head_bc(cfg, tmp_path / "gh.dat")
        text = result.read_text()
        assert "NGB" in text
        lines = text.strip().splitlines()
        assert len(lines) == 2

    def test_with_bcs_factor_one(self, tmp_path: Path) -> None:
        bcs = [
            GeneralHeadBC(
                node_id=1, layer=1, ts_column=0,
                external_head=100.0, conductance=0.5,
            ),
        ]
        cfg = _make_config(
            general_head_bcs=bcs,
            gh_head_factor=1.0,
            gh_conductance_factor=1.0,
            gh_time_unit="1DAY",
        )
        result = write_general_head_bc(cfg, tmp_path / "gh.dat")
        text = result.read_text()
        assert "NGB" in text
        assert "FACTH" in text
        assert "FACTC" in text
        assert "TUNIT" in text
        assert "100.0000" in text
        assert "0.500000" in text

    def test_factor_division(self, tmp_path: Path) -> None:
        bcs = [
            GeneralHeadBC(
                node_id=1, layer=1, ts_column=0,
                external_head=200.0, conductance=6.0,
            ),
        ]
        cfg = _make_config(
            general_head_bcs=bcs,
            gh_head_factor=2.0,
            gh_conductance_factor=3.0,
        )
        result = write_general_head_bc(cfg, tmp_path / "gh.dat")
        text = result.read_text()
        # head: 200 / 2 = 100; conductance: 6 / 3 = 2
        assert "100.0000" in text
        assert "2.000000" in text

    def test_zero_head_factor_fallback(self, tmp_path: Path) -> None:
        bcs = [
            GeneralHeadBC(
                node_id=1, layer=1, ts_column=0,
                external_head=77.0, conductance=3.0,
            ),
        ]
        cfg = _make_config(
            general_head_bcs=bcs,
            gh_head_factor=0.0,
            gh_conductance_factor=1.0,
        )
        result = write_general_head_bc(cfg, tmp_path / "gh.dat")
        text = result.read_text()
        assert "77.0000" in text

    def test_zero_conductance_factor_fallback(self, tmp_path: Path) -> None:
        bcs = [
            GeneralHeadBC(
                node_id=1, layer=1, ts_column=0,
                external_head=10.0, conductance=5.5,
            ),
        ]
        cfg = _make_config(
            general_head_bcs=bcs,
            gh_head_factor=1.0,
            gh_conductance_factor=0.0,
        )
        result = write_general_head_bc(cfg, tmp_path / "gh.dat")
        text = result.read_text()
        assert "5.500000" in text

    def test_multiple_bcs(self, tmp_path: Path) -> None:
        bcs = [
            GeneralHeadBC(node_id=1, layer=1, ts_column=0, external_head=10.0, conductance=1.0),
            GeneralHeadBC(node_id=2, layer=2, ts_column=1, external_head=20.0, conductance=2.0),
            GeneralHeadBC(node_id=3, layer=3, ts_column=2, external_head=30.0, conductance=3.0),
        ]
        cfg = _make_config(general_head_bcs=bcs)
        result = write_general_head_bc(cfg, tmp_path / "gh.dat")
        text = result.read_text()
        # Count line should show 3
        assert "3" in text
        # All nodes present
        for nid in [1, 2, 3]:
            assert str(nid) in text

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        cfg = _make_config()
        result = write_general_head_bc(cfg, tmp_path / "x" / "gh.dat")
        assert result.exists()


# ---------------------------------------------------------------------------
# write_constrained_gh_bc
# ---------------------------------------------------------------------------


class TestWriteConstrainedGhBC:
    def test_empty_bcs(self, tmp_path: Path) -> None:
        cfg = _make_config(constrained_gh_bcs=[])
        result = write_constrained_gh_bc(cfg, tmp_path / "cgh.dat")
        text = result.read_text()
        assert "NCGB" in text
        lines = text.strip().splitlines()
        assert len(lines) == 2

    def test_with_bcs_all_factors_one(self, tmp_path: Path) -> None:
        bcs = [
            ConstrainedGeneralHeadBC(
                node_id=1, layer=1, ts_column=0,
                external_head=100.0, conductance=0.5,
                constraining_head=50.0, max_flow_ts_column=0, max_flow=10.0,
            ),
        ]
        cfg = _make_config(
            constrained_gh_bcs=bcs,
            cgh_head_factor=1.0,
            cgh_max_flow_factor=1.0,
            cgh_head_time_unit="1DAY",
            cgh_conductance_factor=1.0,
            cgh_conductance_time_unit="1DAY",
        )
        result = write_constrained_gh_bc(cfg, tmp_path / "cgh.dat")
        text = result.read_text()
        assert "NCGB" in text
        assert "FACTH" in text
        assert "FACTVL" in text
        assert "TUNIT" in text
        assert "FACTC" in text
        assert "TUNITC" in text
        assert "100.0000" in text
        assert "0.500000" in text
        assert "50.0000" in text
        assert "10.0000" in text

    def test_factor_division(self, tmp_path: Path) -> None:
        bcs = [
            ConstrainedGeneralHeadBC(
                node_id=1, layer=1, ts_column=0,
                external_head=400.0, conductance=9.0,
                constraining_head=200.0, max_flow_ts_column=0, max_flow=50.0,
            ),
        ]
        cfg = _make_config(
            constrained_gh_bcs=bcs,
            cgh_head_factor=2.0,
            cgh_conductance_factor=3.0,
            cgh_max_flow_factor=5.0,
        )
        result = write_constrained_gh_bc(cfg, tmp_path / "cgh.dat")
        text = result.read_text()
        # head: 400/2=200, cond: 9/3=3, constraining_head: 200/2=100, max_flow: 50/5=10
        assert "200.0000" in text
        assert "3.000000" in text
        assert "100.0000" in text
        assert "10.0000" in text

    def test_zero_head_factor_fallback(self, tmp_path: Path) -> None:
        bcs = [
            ConstrainedGeneralHeadBC(
                node_id=1, layer=1, ts_column=0,
                external_head=77.0, conductance=1.0,
                constraining_head=33.0, max_flow_ts_column=0, max_flow=5.0,
            ),
        ]
        cfg = _make_config(
            constrained_gh_bcs=bcs,
            cgh_head_factor=0.0,
            cgh_conductance_factor=1.0,
            cgh_max_flow_factor=1.0,
        )
        result = write_constrained_gh_bc(cfg, tmp_path / "cgh.dat")
        text = result.read_text()
        # Zero factor => raw values used for head and constraining_head
        assert "77.0000" in text
        assert "33.0000" in text

    def test_zero_conductance_factor_fallback(self, tmp_path: Path) -> None:
        bcs = [
            ConstrainedGeneralHeadBC(
                node_id=1, layer=1, ts_column=0,
                external_head=10.0, conductance=4.4,
                constraining_head=5.0, max_flow_ts_column=0, max_flow=1.0,
            ),
        ]
        cfg = _make_config(
            constrained_gh_bcs=bcs,
            cgh_head_factor=1.0,
            cgh_conductance_factor=0.0,
            cgh_max_flow_factor=1.0,
        )
        result = write_constrained_gh_bc(cfg, tmp_path / "cgh.dat")
        text = result.read_text()
        assert "4.400000" in text

    def test_zero_max_flow_factor_fallback(self, tmp_path: Path) -> None:
        bcs = [
            ConstrainedGeneralHeadBC(
                node_id=1, layer=1, ts_column=0,
                external_head=10.0, conductance=1.0,
                constraining_head=5.0, max_flow_ts_column=0, max_flow=99.0,
            ),
        ]
        cfg = _make_config(
            constrained_gh_bcs=bcs,
            cgh_head_factor=1.0,
            cgh_conductance_factor=1.0,
            cgh_max_flow_factor=0.0,
        )
        result = write_constrained_gh_bc(cfg, tmp_path / "cgh.dat")
        text = result.read_text()
        assert "99.0000" in text

    def test_multiple_bcs(self, tmp_path: Path) -> None:
        bcs = [
            ConstrainedGeneralHeadBC(
                node_id=i, layer=1, ts_column=0,
                external_head=float(i * 10), conductance=1.0,
                constraining_head=float(i * 5), max_flow_ts_column=0,
                max_flow=float(i),
            )
            for i in range(1, 4)
        ]
        cfg = _make_config(constrained_gh_bcs=bcs)
        result = write_constrained_gh_bc(cfg, tmp_path / "cgh.dat")
        text = result.read_text()
        assert "3" in text  # NCGB count

    def test_max_flow_ts_column_written(self, tmp_path: Path) -> None:
        bcs = [
            ConstrainedGeneralHeadBC(
                node_id=1, layer=1, ts_column=5,
                external_head=10.0, conductance=1.0,
                constraining_head=5.0, max_flow_ts_column=8, max_flow=1.0,
            ),
        ]
        cfg = _make_config(constrained_gh_bcs=bcs)
        result = write_constrained_gh_bc(cfg, tmp_path / "cgh.dat")
        text = result.read_text()
        assert "8" in text  # max_flow_ts_column

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        cfg = _make_config()
        result = write_constrained_gh_bc(cfg, tmp_path / "deep" / "cgh.dat")
        assert result.exists()


# ---------------------------------------------------------------------------
# Integration-style: write and verify formatting consistency
# ---------------------------------------------------------------------------


class TestOutputFormatting:
    """Verify exact formatting of data lines."""

    def test_specified_flow_bc_line_format(self, tmp_path: Path) -> None:
        bcs = [SpecifiedFlowBC(node_id=123, layer=2, ts_column=5, base_flow=999.5)]
        cfg = _make_config(specified_flow_bcs=bcs, sp_flow_factor=1.0)
        result = write_specified_flow_bc(cfg, tmp_path / "flow.dat")
        lines = result.read_text().splitlines()
        # Last line is the data row
        data_line = lines[-1]
        parts = data_line.split()
        assert parts[0] == "123"
        assert parts[1] == "2"
        assert parts[2] == "5"
        assert parts[3] == "999.5000"

    def test_specified_head_bc_line_format(self, tmp_path: Path) -> None:
        bcs = [SpecifiedHeadBC(node_id=456, layer=3, ts_column=7, head_value=88.25)]
        cfg = _make_config(specified_head_bcs=bcs, sp_head_factor=1.0)
        result = write_specified_head_bc(cfg, tmp_path / "head.dat")
        lines = result.read_text().splitlines()
        data_line = lines[-1]
        parts = data_line.split()
        assert parts[0] == "456"
        assert parts[1] == "3"
        assert parts[2] == "7"
        assert parts[3] == "88.2500"

    def test_general_head_bc_line_format(self, tmp_path: Path) -> None:
        bcs = [
            GeneralHeadBC(
                node_id=789, layer=1, ts_column=2,
                external_head=150.0, conductance=0.001234,
            )
        ]
        cfg = _make_config(general_head_bcs=bcs)
        result = write_general_head_bc(cfg, tmp_path / "gh.dat")
        lines = result.read_text().splitlines()
        data_line = lines[-1]
        parts = data_line.split()
        assert parts[0] == "789"
        assert parts[1] == "1"
        assert parts[2] == "2"
        assert parts[3] == "150.0000"
        assert parts[4] == "0.001234"

    def test_constrained_gh_bc_line_format(self, tmp_path: Path) -> None:
        bcs = [
            ConstrainedGeneralHeadBC(
                node_id=321, layer=2, ts_column=4,
                external_head=500.0, conductance=1.5,
                constraining_head=250.0, max_flow_ts_column=6, max_flow=75.0,
            )
        ]
        cfg = _make_config(constrained_gh_bcs=bcs)
        result = write_constrained_gh_bc(cfg, tmp_path / "cgh.dat")
        lines = result.read_text().splitlines()
        data_line = lines[-1]
        parts = data_line.split()
        assert parts[0] == "321"
        assert parts[1] == "2"
        assert parts[2] == "4"
        assert parts[3] == "500.0000"
        assert parts[4] == "1.500000"
        assert parts[5] == "250.0000"
        assert parts[6] == "6"
        assert parts[7] == "75.0000"


class TestStringPathAcceptance:
    """Verify all writers accept string paths (not just Path objects)."""

    def test_write_bc_main_str(self, tmp_path: Path) -> None:
        cfg = _make_config()
        result = write_bc_main(cfg, str(tmp_path / "main.dat"))
        assert isinstance(result, Path)
        assert result.exists()

    def test_write_specified_flow_bc_str(self, tmp_path: Path) -> None:
        cfg = _make_config()
        result = write_specified_flow_bc(cfg, str(tmp_path / "flow.dat"))
        assert isinstance(result, Path)
        assert result.exists()

    def test_write_specified_head_bc_str(self, tmp_path: Path) -> None:
        cfg = _make_config()
        result = write_specified_head_bc(cfg, str(tmp_path / "head.dat"))
        assert isinstance(result, Path)
        assert result.exists()

    def test_write_general_head_bc_str(self, tmp_path: Path) -> None:
        cfg = _make_config()
        result = write_general_head_bc(cfg, str(tmp_path / "gh.dat"))
        assert isinstance(result, Path)
        assert result.exists()

    def test_write_constrained_gh_bc_str(self, tmp_path: Path) -> None:
        cfg = _make_config()
        result = write_constrained_gh_bc(cfg, str(tmp_path / "cgh.dat"))
        assert isinstance(result, Path)
        assert result.exists()
