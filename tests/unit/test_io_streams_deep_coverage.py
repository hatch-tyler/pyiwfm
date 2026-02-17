"""Deep coverage tests for streams.py StreamMainFileReader, StreamSpecReader,
and related parsing methods.

Targets uncovered lines: 615, 796-830, 838-842, 847-853, 886-904, 920-935,
943-964, 976-992, 996-1000, 1009-1044, 1053-1087, 1105-1113, 1125-1135,
1167-1188, 1194, 1272-1287, 1305-1324, 1342-1385, 1472-1498.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from pyiwfm.core.exceptions import FileFormatError
from pyiwfm.io.streams import (
    StreamMainFileReader,
    StreamReader,
    StreamSpecReader,
    parse_stream_version,
    read_stream_main_file,
    read_stream_spec,
    stream_version_ge,
)

# =============================================================================
# Helpers
# =============================================================================

# IMPORTANT: IWFM comment detection checks line[0] in ("C","c","*").
# Therefore unit strings like "CFS" will be skipped as comments.
# Use unit strings that do NOT start with C, c, or * (e.g., "FT3/S").


def _write(tmp_path: Path, name: str, content: str) -> Path:
    """Write dedented content to a file and return its path."""
    p = tmp_path / name
    p.write_text(dedent(content))
    return p


# =============================================================================
# StreamMainFileReader -- full v4.2 parse with hydrographs, node budget,
# bed params, evaporation
# =============================================================================


class TestStreamMainFileReaderV42:
    """Tests for StreamMainFileReader parsing a complete v4.2 file."""

    def _build_v42_file(self, tmp_path: Path) -> Path:
        """Build a minimal but complete v4.2 main file."""
        return _write(
            tmp_path,
            "stream_main.dat",
            """\
            C  Stream component main file (v4.2)
            #4.2
            C ---- sub-files ----
            inflows.dat                         / INFLOWFL
            divspec.dat                         / DIVSPECFL
            bypspec.dat                         / BYPSPECFL
            div_ts.dat                          / DIVFL
            budget.hdf                          / STRMRCHBUDFL
            divbudget.hdf                       / DIVDTLBUDFL
            C ---- hydrograph section ----
            3                                   / NOUTR
            2                                   / IHSQR (0=flow,1=stage,2=both)
            1.0                                 / FACTSQOU
            FT3/S                               / UNITSQOU
            1.0                                 / FACTLTOU (stage factor)
            FT                                  / UNITLTOU (stage unit)
            hydout.out                          / STRMHYDOUTFL
            C  hydrograph specs (IOUTR  NAME)
            10   Gage-A
            20   Gage-B
            30   Gage-C
            C ---- node budget section ----
            2                                   / NBUDR
            nodebudget.hdf                      / node budget file
            5
            12
            C ---- bed parameters ----
            0.5                                 / FACTK
            1DAY                                / TUNITSK
            1.0                                 / FACTL
            C  IR  WETPR  IRGW  CSTRM  DSTRM   (v4.2 layout, 5 cols)
            1    3.5   10   0.005   1.2
            2    4.0   11   0.006   1.3
            C ---- INTRCTYPE ----
            2
            C ---- evaporation ----
            evap_area.dat                       / STARFL
            C  IR  ICETST  ICARST
            1   3   5
            2   4   6
            """,
        )

    def test_full_v42_parse(self, tmp_path: Path) -> None:
        """Parse a complete v4.2 stream main file and verify all sections."""
        fpath = self._build_v42_file(tmp_path)
        reader = StreamMainFileReader()
        cfg = reader.read(fpath, base_dir=tmp_path)

        # Version
        assert cfg.version == "4.2"

        # Sub-file paths resolved relative to base_dir
        assert cfg.inflow_file == tmp_path / "inflows.dat"
        assert cfg.diversion_spec_file == tmp_path / "divspec.dat"
        assert cfg.bypass_spec_file == tmp_path / "bypspec.dat"
        assert cfg.diversion_file == tmp_path / "div_ts.dat"
        assert cfg.budget_output_file == tmp_path / "budget.hdf"
        assert cfg.diversion_budget_file == tmp_path / "divbudget.hdf"

        # Hydrograph section
        assert cfg.hydrograph_count == 3
        assert cfg.hydrograph_output_type == 2
        assert cfg.hydrograph_flow_factor == pytest.approx(1.0)
        assert cfg.hydrograph_flow_unit == "FT3/S"
        assert cfg.hydrograph_elev_factor == pytest.approx(1.0)
        assert cfg.hydrograph_elev_unit == "FT"
        assert cfg.hydrograph_output_file == tmp_path / "hydout.out"
        assert len(cfg.hydrograph_specs) == 3
        assert cfg.hydrograph_specs[0] == (10, "Gage-A")
        assert cfg.hydrograph_specs[2] == (30, "Gage-C")

        # Node budget section
        assert cfg.node_budget_count == 2
        assert cfg.node_budget_output_file == tmp_path / "nodebudget.hdf"
        assert cfg.node_budget_ids == [5, 12]

        # Bed parameters (v4.2 layout)
        assert cfg.conductivity_factor == pytest.approx(0.5)
        assert cfg.conductivity_time_unit == "1DAY"
        assert cfg.length_factor == pytest.approx(1.0)
        assert len(cfg.bed_params) == 2
        bp1 = cfg.bed_params[0]
        assert bp1.node_id == 1
        assert bp1.wetted_perimeter == pytest.approx(3.5)
        assert bp1.gw_node == 10
        assert bp1.conductivity == pytest.approx(0.005)
        assert bp1.bed_thickness == pytest.approx(1.2)

        # INTRCTYPE
        assert cfg.interaction_type == 2

        # Evaporation
        assert cfg.evap_area_file == tmp_path / "evap_area.dat"
        assert len(cfg.evap_node_specs) == 2
        assert cfg.evap_node_specs[0] == (1, 3, 5)
        assert cfg.evap_node_specs[1] == (2, 4, 6)


class TestStreamMainFileReaderV50:
    """Tests for StreamMainFileReader with v5.0 features."""

    def _build_v50_file(self, tmp_path: Path) -> Path:
        """Build a minimal v5.0 main file with cross-sections and ICs."""
        return _write(
            tmp_path,
            "stream_v50.dat",
            """\
            C  v5.0 stream main file
            #5.0
            inflows.dat
            divspec.dat
            bypspec.dat
            div_ts.dat
            budget.hdf
            divbudget.hdf
            final_flows.dat                     / end-of-sim flows (v5.0)
            C ---- hydrograph section ----
            1                                   / NOUTR
            0                                   / IHSQR (flow only)
            1.0                                 / FACTSQOU
            FT3/S                               / UNITSQOU
            strmhyd.out                         / STRMHYDOUTFL
            C  hydrograph specs
            5   Test-Gage
            C ---- node budget ----
            0                                   / NBUDR (none)
            C ---- bed parameters ----
            1.0                                 / FACTK
            1DAY                                / TUNITSK
            1.0                                 / FACTL
            C  IR  CSTRM  DSTRM  (v5.0 = 3 cols)
            1   0.01   0.5
            2   0.02   0.6
            C ---- INTRCTYPE ----
            1
            C ---- cross-section data (v5.0) ----
            1.0                                 / FACTN
            1.0                                 / FACTLT
            C  IR  BottomElev  B0   s    n    MaxDepth
            1   100.0   5.0   2.0   0.035   10.0
            2    95.0   4.0   1.5   0.040   12.0
            C ---- initial conditions (v5.0) ----
            0                                   / ICType
            1DAY                                / time unit
            1.0                                 / FACTH
            C  IR   value
            1   50.0
            2   45.0
            C ---- evaporation ----
            evap.dat
            1   2   3
            """,
        )

    def test_full_v50_parse(self, tmp_path: Path) -> None:
        """Parse a complete v5.0 file with cross-sections and initial conditions."""
        fpath = self._build_v50_file(tmp_path)
        reader = StreamMainFileReader()
        cfg = reader.read(fpath, base_dir=tmp_path)

        assert cfg.version == "5.0"
        assert cfg.final_flow_file == tmp_path / "final_flows.dat"

        # Hydrograph - flow only (type 0), no stage factor
        assert cfg.hydrograph_count == 1
        assert cfg.hydrograph_output_type == 0
        assert cfg.hydrograph_output_file == tmp_path / "strmhyd.out"
        assert cfg.hydrograph_specs == [(5, "Test-Gage")]

        # Node budget count is zero
        assert cfg.node_budget_count == 0

        # Bed params (v5.0 = 3-col: IR, CSTRM, DSTRM)
        assert len(cfg.bed_params) == 2
        bp = cfg.bed_params[0]
        assert bp.node_id == 1
        assert bp.conductivity == pytest.approx(0.01)
        assert bp.bed_thickness == pytest.approx(0.5)

        # Cross-section data
        assert cfg.roughness_factor == pytest.approx(1.0)
        assert cfg.cross_section_length_factor == pytest.approx(1.0)
        assert len(cfg.cross_section_data) == 2
        cs = cfg.cross_section_data[0]
        assert cs.node_id == 1
        assert cs.bottom_elev == pytest.approx(100.0)
        assert cs.B0 == pytest.approx(5.0)
        assert cs.s == pytest.approx(2.0)
        assert cs.n == pytest.approx(0.035)
        assert cs.max_flow_depth == pytest.approx(10.0)

        # Initial conditions
        assert cfg.ic_type == 0
        assert cfg.ic_time_unit == "1DAY"
        assert cfg.ic_factor == pytest.approx(1.0)
        assert len(cfg.initial_conditions) == 2
        ic = cfg.initial_conditions[0]
        assert ic.node_id == 1
        assert ic.value == pytest.approx(50.0)

        # Evaporation
        assert cfg.evap_area_file == tmp_path / "evap.dat"
        assert cfg.evap_node_specs == [(1, 2, 3)]


class TestStreamMainFileReaderEdgeCases:
    """Edge-case tests for StreamMainFileReader."""

    def test_empty_noutr_returns_early(self, tmp_path: Path) -> None:
        """When NOUTR line is blank/missing, reader returns early (line 804)."""
        fpath = _write(
            tmp_path,
            "stream_empty.dat",
            """\
            C  Minimal file
            #4.2
            inflows.dat
            divspec.dat
            bypspec.dat
            div_ts.dat
            budget.hdf
            divbudget.hdf
            """,
        )
        reader = StreamMainFileReader()
        cfg = reader.read(fpath, base_dir=tmp_path)

        assert cfg.version == "4.2"
        assert cfg.hydrograph_count == 0

    def test_invalid_noutr_returns_early(self, tmp_path: Path) -> None:
        """When NOUTR is non-integer, reader returns early (lines 808-809)."""
        fpath = _write(
            tmp_path,
            "stream_bad_noutr.dat",
            """\
            C  Bad NOUTR
            #4.2
            inflows.dat
            divspec.dat
            bypspec.dat
            div_ts.dat
            budget.hdf
            divbudget.hdf
            abc                                 / NOUTR (invalid)
            """,
        )
        reader = StreamMainFileReader()
        cfg = reader.read(fpath, base_dir=tmp_path)

        assert cfg.hydrograph_count == 0

    def test_zero_hydrograph_count_reads_post_sections(self, tmp_path: Path) -> None:
        """When NOUTR=0, should skip hydrographs but read bed params (lines 812-814).

        Uses v4.1 (3-col bed params) to avoid column count mismatch.
        """
        fpath = _write(
            tmp_path,
            "stream_zero_hyd.dat",
            """\
            C  Stream v4.1 with 0 hydrographs
            #4.1
            inflows.dat
            divspec.dat
            bypspec.dat
            div_ts.dat
            budget.hdf
            divbudget.hdf
            0                                   / NOUTR
            C ---- node budget ----
            0                                   / NBUDR
            C ---- bed parameters ----
            1.0                                 / FACTK
            1DAY                                / TUNITSK
            1.0                                 / FACTL
            1   0.01   0.5
            C ---- INTRCTYPE ----
            1
            C ---- evaporation ----
            evap.dat
            """,
        )
        reader = StreamMainFileReader()
        cfg = reader.read(fpath, base_dir=tmp_path)

        assert cfg.hydrograph_count == 0
        assert len(cfg.hydrograph_specs) == 0
        assert len(cfg.bed_params) == 1
        assert cfg.interaction_type == 1
        assert cfg.evap_area_file == tmp_path / "evap.dat"

    def test_invalid_ihsqr_ignored(self, tmp_path: Path) -> None:
        """Non-numeric IHSQR is silently ignored (lines 821-822)."""
        fpath = _write(
            tmp_path,
            "stream_bad_ihsqr.dat",
            """\
            #4.1
            inflows.dat
            divspec.dat
            bypspec.dat
            div_ts.dat
            budget.hdf
            divbudget.hdf
            1                                   / NOUTR
            xyz                                 / IHSQR (invalid)
            1.0                                 / FACTSQOU
            FT3/S                               / UNITSQOU
            hydout.out
            10   Gage
            C ---- node budget ----
            0
            C ---- bed params ----
            1.0
            1DAY
            1.0
            1
            """,
        )
        reader = StreamMainFileReader()
        cfg = reader.read(fpath, base_dir=tmp_path)

        # IHSQR stays at default 0
        assert cfg.hydrograph_output_type == 0

    def test_invalid_factsqou_ignored(self, tmp_path: Path) -> None:
        """Non-numeric FACTSQOU is silently ignored (lines 829-830)."""
        fpath = _write(
            tmp_path,
            "stream_bad_factsqou.dat",
            """\
            #4.1
            inflows.dat
            divspec.dat
            bypspec.dat
            div_ts.dat
            budget.hdf
            divbudget.hdf
            1                                   / NOUTR
            0                                   / IHSQR
            bad                                 / FACTSQOU (invalid)
            FT3/S                               / UNITSQOU
            hydout.out
            10   Gage
            C ---- node budget ----
            0
            C ---- bed params ----
            1.0
            1DAY
            1.0
            1
            """,
        )
        reader = StreamMainFileReader()
        cfg = reader.read(fpath, base_dir=tmp_path)

        assert cfg.hydrograph_flow_factor == pytest.approx(1.0)

    def test_invalid_stage_factor_ignored(self, tmp_path: Path) -> None:
        """Non-numeric FACTLTOU is silently ignored (lines 841-842)."""
        fpath = _write(
            tmp_path,
            "stream_bad_stage.dat",
            """\
            #4.1
            inflows.dat
            divspec.dat
            bypspec.dat
            div_ts.dat
            budget.hdf
            divbudget.hdf
            1                                   / NOUTR
            1                                   / IHSQR (stage only)
            1.0
            FT3/S
            bad_factor                          / FACTLTOU (invalid)
            FT
            hydout.out
            10   Gage
            C ---- remaining sections ----
            0
            1.0
            1DAY
            1.0
            1
            """,
        )
        reader = StreamMainFileReader()
        cfg = reader.read(fpath, base_dir=tmp_path)

        assert cfg.hydrograph_elev_factor == pytest.approx(1.0)

    def test_resolve_absolute_path(self, tmp_path: Path) -> None:
        """Absolute path is returned as-is."""
        from pyiwfm.io.iwfm_reader import resolve_path

        abs_path = Path("C:/absolute/path/file.dat")
        result = resolve_path(tmp_path, str(abs_path))
        assert result == abs_path

    def test_version_with_empty_file(self, tmp_path: Path) -> None:
        """Empty version line returns empty string (lines 1128-1129)."""
        fpath = _write(
            tmp_path,
            "stream_no_version.dat",
            """\


            C  comment after blank lines
            #4.2
            """,
        )
        reader = StreamMainFileReader()
        # Directly test _read_version
        with open(fpath) as f:
            version = reader._read_version(f)
        # The blank lines are skipped, comment is skipped, then #4.2 found
        assert version == "4.2"

    def test_version_with_no_hash_header(self, tmp_path: Path) -> None:
        """When first non-comment data line has no #, returns empty string (line 1133)."""
        fpath = _write(
            tmp_path,
            "stream_no_hash.dat",
            """\
            C  comment
            c  another comment
            4.2
            """,
        )
        reader = StreamMainFileReader()
        with open(fpath) as f:
            version = reader._read_version(f)
        # "4.2" doesn't start with # and is not a comment -> break -> return ""
        assert version == ""


class TestStreamMainFileReaderNodeBudget:
    """Tests for _read_node_budget_section edge cases."""

    def test_node_budget_invalid_count_returns(self, tmp_path: Path) -> None:
        """Non-integer NBUDR causes early return (lines 889-890)."""
        fpath = _write(
            tmp_path,
            "stream_bad_nbudr.dat",
            """\
            #4.1
            inflows.dat
            divspec.dat
            bypspec.dat
            div_ts.dat
            budget.hdf
            divbudget.hdf
            1                                   / NOUTR
            0
            1.0
            FT3/S
            hydout.out
            10   Gage
            C ---- node budget ----
            abc                                 / NBUDR (invalid)
            C ---- bed params ----
            1.0
            1DAY
            1.0
            1
            """,
        )
        reader = StreamMainFileReader()
        cfg = reader.read(fpath, base_dir=tmp_path)

        # Invalid NBUDR -> returns early, node_budget_count stays 0
        assert cfg.node_budget_count == 0

    def test_node_budget_with_invalid_ids(self, tmp_path: Path) -> None:
        """Non-integer node IDs in budget section cause break (lines 903-904)."""
        fpath = _write(
            tmp_path,
            "stream_bad_nbudr_ids.dat",
            """\
            #4.1
            inflows.dat
            divspec.dat
            bypspec.dat
            div_ts.dat
            budget.hdf
            divbudget.hdf
            1
            0
            1.0
            FT3/S
            hydout.out
            10   Gage
            C ---- node budget ----
            2                                   / NBUDR
            nodebudget.hdf
            5
            not_a_number
            C ---- bed params ----
            1.0
            1DAY
            1.0
            1
            """,
        )
        reader = StreamMainFileReader()
        cfg = reader.read(fpath, base_dir=tmp_path)

        # First ID (5) parsed, second fails -> break
        assert cfg.node_budget_count == 2
        assert cfg.node_budget_ids == [5]


class TestStreamMainFileReaderBedParams:
    """Tests for _read_bed_params_section edge cases."""

    def test_bed_params_v40_layout(self, tmp_path: Path) -> None:
        """v4.0 uses 4-column layout: IR, CSTRM, DSTRM, WETPR (lines 980-984)."""
        fpath = _write(
            tmp_path,
            "stream_v40.dat",
            """\
            #4.0
            inflows.dat
            divspec.dat
            bypspec.dat
            div_ts.dat
            budget.hdf
            divbudget.hdf
            0                                   / NOUTR
            C ---- node budget ----
            0
            C ---- bed params (v4.0: 4 cols) ----
            1.0
            1DAY
            1.0
            C  IR  CSTRM  DSTRM  WETPR
            1   0.01   0.5   3.0
            2   0.02   0.6   4.0
            C ---- INTRCTYPE ----
            1
            C ---- evaporation ----
            evap.dat
            """,
        )
        reader = StreamMainFileReader()
        cfg = reader.read(fpath, base_dir=tmp_path)

        assert len(cfg.bed_params) == 2
        bp = cfg.bed_params[0]
        assert bp.node_id == 1
        assert bp.conductivity == pytest.approx(0.01)
        assert bp.bed_thickness == pytest.approx(0.5)
        assert bp.wetted_perimeter == pytest.approx(3.0)

    def test_bed_params_auto_detect_5_cols(self, tmp_path: Path) -> None:
        """Auto-detect 5-column layout even when version is not 4.2 (lines 959-964)."""
        # The auto-detect kicks in when first row has >= 5 columns
        fpath = _write(
            tmp_path,
            "stream_autodetect.dat",
            """\
            #4.1
            inflows.dat
            divspec.dat
            bypspec.dat
            div_ts.dat
            budget.hdf
            divbudget.hdf
            0                                   / NOUTR
            C ---- node budget ----
            0
            C ---- bed params ----
            1.0
            1DAY
            1.0
            C  version says 4.1 but data has 5 columns -> auto-detect v4.2
            1    3.5   10   0.005   1.2
            2    4.0   11   0.006   1.3
            C ---- INTRCTYPE ----
            1
            """,
        )
        reader = StreamMainFileReader()
        cfg = reader.read(fpath, base_dir=tmp_path)

        assert len(cfg.bed_params) == 2
        bp = cfg.bed_params[0]
        # Auto-detected as v4.2 layout: IR WETPR IRGW CSTRM DSTRM
        assert bp.wetted_perimeter == pytest.approx(3.5)
        assert bp.gw_node == 10
        assert bp.conductivity == pytest.approx(0.005)

    def test_bed_params_invalid_row_causes_pushback(self, tmp_path: Path) -> None:
        """ValueError in bed param row causes pushback (lines 990-992)."""
        fpath = _write(
            tmp_path,
            "stream_bad_bed.dat",
            """\
            #4.1
            inflows.dat
            divspec.dat
            bypspec.dat
            div_ts.dat
            budget.hdf
            divbudget.hdf
            0
            0
            1.0
            1DAY
            1.0
            C  valid row then invalid
            1   0.01   0.5
            bad   xyz   abc
            C ---- evaporation ----
            evap.dat
            """,
        )
        reader = StreamMainFileReader()
        cfg = reader.read(fpath, base_dir=tmp_path)

        # First row parsed, second row caused pushback
        assert len(cfg.bed_params) == 1

    def test_bed_params_factk_invalid_returns(self, tmp_path: Path) -> None:
        """Non-numeric FACTK causes early return (lines 922-924)."""
        fpath = _write(
            tmp_path,
            "stream_bad_factk.dat",
            """\
            #4.1
            inflows.dat
            divspec.dat
            bypspec.dat
            div_ts.dat
            budget.hdf
            divbudget.hdf
            0
            0
            bad_factk
            """,
        )
        reader = StreamMainFileReader()
        cfg = reader.read(fpath, base_dir=tmp_path)

        # FACTK parse failure -> return, bed_params empty
        assert len(cfg.bed_params) == 0

    def test_bed_params_factl_invalid_ignored(self, tmp_path: Path) -> None:
        """Non-numeric FACTL is silently ignored (lines 934-935)."""
        fpath = _write(
            tmp_path,
            "stream_bad_factl.dat",
            """\
            #4.1
            inflows.dat
            divspec.dat
            bypspec.dat
            div_ts.dat
            budget.hdf
            divbudget.hdf
            0
            0
            1.0
            1DAY
            bad_factl
            1   0.01   0.5
            1
            """,
        )
        reader = StreamMainFileReader()
        cfg = reader.read(fpath, base_dir=tmp_path)

        # FACTL stays at default
        assert cfg.length_factor == pytest.approx(1.0)


class TestStreamMainFileReaderCrossSection:
    """Tests for _read_cross_section_data edge cases."""

    def test_cross_section_factn_invalid_returns(self, tmp_path: Path) -> None:
        """Non-numeric FACTN causes early return (lines 1012-1013)."""
        fpath = _write(
            tmp_path,
            "stream_cs_bad.dat",
            """\
            #5.0
            inflows.dat
            divspec.dat
            bypspec.dat
            div_ts.dat
            budget.hdf
            divbudget.hdf
            final.dat
            0
            0
            1.0
            1DAY
            1.0
            1   0.01   0.5
            1
            bad_factn
            """,
        )
        reader = StreamMainFileReader()
        cfg = reader.read(fpath, base_dir=tmp_path)

        assert len(cfg.cross_section_data) == 0

    def test_cross_section_factlt_invalid_ignored(self, tmp_path: Path) -> None:
        """Non-numeric FACTLT is silently ignored (lines 1020-1021)."""
        fpath = _write(
            tmp_path,
            "stream_cs_bad_factlt.dat",
            """\
            #5.0
            inflows.dat
            divspec.dat
            bypspec.dat
            div_ts.dat
            budget.hdf
            divbudget.hdf
            final.dat
            0
            0
            1.0
            1DAY
            1.0
            1   0.01   0.5
            1
            1.0
            bad_factlt
            1   100.0   5.0   2.0   0.035   10.0
            0
            1DAY
            1.0
            1   50.0
            """,
        )
        reader = StreamMainFileReader()
        cfg = reader.read(fpath, base_dir=tmp_path)

        assert cfg.cross_section_length_factor == pytest.approx(1.0)

    def test_cross_section_invalid_row_pushback(self, tmp_path: Path) -> None:
        """ValueError in cross-section row causes pushback (lines 1042-1044)."""
        fpath = _write(
            tmp_path,
            "stream_cs_bad_row.dat",
            """\
            #5.0
            inflows.dat
            divspec.dat
            bypspec.dat
            div_ts.dat
            budget.hdf
            divbudget.hdf
            final.dat
            0
            0
            1.0
            1DAY
            1.0
            1   0.01   0.5
            1
            1.0
            1.0
            1   100.0   5.0   2.0   0.035   10.0
            bad   x   y   z   a   b
            C ---- IC ----
            0
            1DAY
            1.0
            1   50.0
            """,
        )
        reader = StreamMainFileReader()
        cfg = reader.read(fpath, base_dir=tmp_path)

        assert len(cfg.cross_section_data) == 1


class TestStreamMainFileReaderInitialConditions:
    """Tests for _read_initial_conditions edge cases."""

    def test_ic_type_invalid_returns(self, tmp_path: Path) -> None:
        """Non-integer ICType causes early return (lines 1056-1057)."""
        fpath = _write(
            tmp_path,
            "stream_ic_bad.dat",
            """\
            #5.0
            inflows.dat
            divspec.dat
            bypspec.dat
            div_ts.dat
            budget.hdf
            divbudget.hdf
            final.dat
            0
            0
            1.0
            1DAY
            1.0
            1   0.01   0.5
            1
            1.0
            1.0
            1   100.0   5.0   2.0   0.035   10.0
            C ---- IC (invalid type) ----
            bad_ic_type
            """,
        )
        reader = StreamMainFileReader()
        cfg = reader.read(fpath, base_dir=tmp_path)

        assert cfg.ic_type == 0  # default

    def test_ic_facth_invalid_ignored(self, tmp_path: Path) -> None:
        """Non-numeric FACTH is silently ignored (lines 1067-1068)."""
        fpath = _write(
            tmp_path,
            "stream_ic_bad_facth.dat",
            """\
            #5.0
            inflows.dat
            divspec.dat
            bypspec.dat
            div_ts.dat
            budget.hdf
            divbudget.hdf
            final.dat
            0
            0
            1.0
            1DAY
            1.0
            1   0.01   0.5
            1
            1.0
            1.0
            1   100.0   5.0   2.0   0.035   10.0
            0
            1DAY
            bad_facth
            1   50.0
            """,
        )
        reader = StreamMainFileReader()
        cfg = reader.read(fpath, base_dir=tmp_path)

        assert cfg.ic_factor == pytest.approx(1.0)
        assert len(cfg.initial_conditions) == 1

    def test_ic_row_less_than_2_cols_pushback(self, tmp_path: Path) -> None:
        """IC row with <2 columns causes pushback (lines 1077-1078)."""
        fpath = _write(
            tmp_path,
            "stream_ic_short_row.dat",
            """\
            #5.0
            inflows.dat
            divspec.dat
            bypspec.dat
            div_ts.dat
            budget.hdf
            divbudget.hdf
            final.dat
            0
            0
            1.0
            1DAY
            1.0
            1   0.01   0.5
            1
            1.0
            1.0
            1   100.0   5.0   2.0   0.035   10.0
            0
            1DAY
            1.0
            1   50.0
            evap.dat
            """,
        )
        reader = StreamMainFileReader()
        cfg = reader.read(fpath, base_dir=tmp_path)

        # "evap.dat" is a single-column row; it should cause pushback from IC loop
        # and then get read by the evaporation section
        assert len(cfg.initial_conditions) == 1
        assert cfg.evap_area_file == tmp_path / "evap.dat"

    def test_ic_row_invalid_values_pushback(self, tmp_path: Path) -> None:
        """ValueError in IC row causes pushback (lines 1085-1087)."""
        fpath = _write(
            tmp_path,
            "stream_ic_bad_row.dat",
            """\
            #5.0
            inflows.dat
            divspec.dat
            bypspec.dat
            div_ts.dat
            budget.hdf
            divbudget.hdf
            final.dat
            0
            0
            1.0
            1DAY
            1.0
            1   0.01   0.5
            1
            1.0
            1.0
            1   100.0   5.0   2.0   0.035   10.0
            0
            1DAY
            1.0
            1   50.0
            bad   xyz
            """,
        )
        reader = StreamMainFileReader()
        cfg = reader.read(fpath, base_dir=tmp_path)

        assert len(cfg.initial_conditions) == 1


class TestStreamMainFileReaderHydrographSpecs:
    """Tests for _read_hydrograph_specs edge cases."""

    def test_hydrograph_specs_skips_blank_and_invalid(self, tmp_path: Path) -> None:
        """Blank/invalid lines in hydrograph specs are skipped (lines 1174, 1185-1186)."""
        fpath = _write(
            tmp_path,
            "stream_hyd_specs.dat",
            """\
            #4.1
            inflows.dat
            divspec.dat
            bypspec.dat
            div_ts.dat
            budget.hdf
            divbudget.hdf
            2                                   / NOUTR
            0
            1.0
            FT3/S
            hydout.out
            C  comment line inside specs
            bad_line
            10   Gage-A
            20   Gage-B
            C ---- post hydrograph ----
            0
            1.0
            1DAY
            1.0
            1
            """,
        )
        reader = StreamMainFileReader()
        cfg = reader.read(fpath, base_dir=tmp_path)

        # "bad_line" is skipped (ValueError on int("bad_line"))
        assert len(cfg.hydrograph_specs) == 2
        assert cfg.hydrograph_specs[0] == (10, "Gage-A")


class TestStreamMainFileReaderEvaporation:
    """Tests for _read_evaporation_section edge cases."""

    def test_evap_node_row_too_few_cols(self, tmp_path: Path) -> None:
        """Evap row with <3 cols causes break (line 1105)."""
        fpath = _write(
            tmp_path,
            "stream_evap_short.dat",
            """\
            #4.1
            inflows.dat
            divspec.dat
            bypspec.dat
            div_ts.dat
            budget.hdf
            divbudget.hdf
            0
            0
            1.0
            1DAY
            1.0
            1   0.01   0.5
            1
            evap.dat
            1   2   3
            4   5
            """,
        )
        reader = StreamMainFileReader()
        cfg = reader.read(fpath, base_dir=tmp_path)

        # First evap row parsed; second has only 2 cols -> break
        assert len(cfg.evap_node_specs) == 1

    def test_evap_node_row_invalid_values(self, tmp_path: Path) -> None:
        """Non-integer evap spec causes break (lines 1112-1113)."""
        fpath = _write(
            tmp_path,
            "stream_evap_bad.dat",
            """\
            #4.1
            inflows.dat
            divspec.dat
            bypspec.dat
            div_ts.dat
            budget.hdf
            divbudget.hdf
            0
            0
            1.0
            1DAY
            1.0
            1   0.01   0.5
            1
            evap.dat
            bad   xyz   abc
            """,
        )
        reader = StreamMainFileReader()
        cfg = reader.read(fpath, base_dir=tmp_path)

        assert len(cfg.evap_node_specs) == 0


# =============================================================================
# StreamSpecReader
# =============================================================================


class TestStreamSpecReader:
    """Tests for StreamSpecReader parsing."""

    def test_basic_spec_file(self, tmp_path: Path) -> None:
        """Parse a basic StreamsSpec file with 2 reaches and rating tables.

        Rating tables are in a separate section after ALL reach definitions
        (not interleaved with node data). Each node's first rating line has
        4 columns: node_id, bottom_elev, depth, flow. Remaining NRTB-1 lines
        have 2 columns: depth, flow.
        """
        fpath = _write(
            tmp_path,
            "streams_spec.dat",
            """\
            C  StreamsSpec file
            #4.2
            C  NRH (number of reaches)
            2                                   / NRH
            C  NRTB (rating table points)
            3                                   / NRTB
            C  Reach 1: ID NSNRH IOUTRH NAME
            1   2   0   Upper_Reach
            C  Node 1: stream_node_id  gw_node_id
            1   10
            C  Node 2
            2   11
            C  Reach 2
            2   1   1   Lower_Reach
            3   12
            C  Rating tables section
            1.0                                 / FACTLT
            1.0                                 / FACTQ
            1min                                / TUNIT
            C  Rating table for node 1
            1   100.0   0.0   0.0
                        1.0   10.0
                        2.0   50.0
            C  Rating table for node 2
            2   90.0    0.0   0.0
                        1.0   15.0
                        2.0   60.0
            C  Rating table for node 3
            3   80.0    0.0   0.0
                        1.0   20.0
                        2.0   80.0
            """,
        )
        n_reaches, n_rtb, reaches = StreamSpecReader().read(fpath)

        assert n_reaches == 2
        assert n_rtb == 3
        assert len(reaches) == 2

        r1 = reaches[0]
        assert r1.id == 1
        assert r1.n_nodes == 2
        assert r1.outflow_node == 0
        assert r1.name == "Upper_Reach"
        assert r1.node_ids == [1, 2]
        assert r1.node_to_gw_node == {1: 10, 2: 11}
        assert len(r1.node_rating_tables) == 2
        stages, flows = r1.node_rating_tables[1]
        assert stages == pytest.approx([0.0, 1.0, 2.0])
        assert flows == pytest.approx([0.0, 10.0, 50.0])
        assert r1.node_bottom_elevations[1] == pytest.approx(100.0)
        assert r1.node_bottom_elevations[2] == pytest.approx(90.0)

        r2 = reaches[1]
        assert r2.id == 2
        assert r2.n_nodes == 1
        assert r2.outflow_node == 1
        assert r2.node_bottom_elevations[3] == pytest.approx(80.0)

    def test_v50_spec_file_no_rating_tables(self, tmp_path: Path) -> None:
        """v5.0 spec files have no rating tables (line 1280)."""
        fpath = _write(
            tmp_path,
            "streams_spec_v50.dat",
            """\
            #5.0
            1
            1   2   0   Only_Reach
            1   10
            2   11
            """,
        )
        n_reaches, n_rtb, reaches = StreamSpecReader().read(fpath)

        assert n_reaches == 1
        assert n_rtb == 0
        assert len(reaches) == 1
        assert reaches[0].node_ids == [1, 2]
        assert len(reaches[0].node_rating_tables) == 0

    def test_invalid_nrh_raises(self, tmp_path: Path) -> None:
        """Non-integer NRH raises FileFormatError (lines 1272-1273)."""
        fpath = _write(
            tmp_path,
            "streams_spec_bad_nrh.dat",
            """\
            #4.2
            bad_nrh
            """,
        )
        with pytest.raises(FileFormatError, match="Invalid NRH"):
            StreamSpecReader().read(fpath)

    def test_invalid_nrtb_raises(self, tmp_path: Path) -> None:
        """Non-integer NRTB raises FileFormatError (lines 1286-1287)."""
        fpath = _write(
            tmp_path,
            "streams_spec_bad_nrtb.dat",
            """\
            #4.2
            2
            bad_nrtb
            """,
        )
        with pytest.raises(FileFormatError, match="Invalid NRTB"):
            StreamSpecReader().read(fpath)

    def test_reach_header_too_short_raises(self, tmp_path: Path) -> None:
        """Reach header with <3 parts raises FileFormatError (line 1342)."""
        fpath = _write(
            tmp_path,
            "streams_spec_bad_reach.dat",
            """\
            #4.2
            1
            0
            1   2
            """,
        )
        with pytest.raises(FileFormatError, match="Invalid reach header"):
            StreamSpecReader().read(fpath)

    def test_rating_table_invalid_values(self, tmp_path: Path) -> None:
        """Non-numeric rating table continuation values are silently skipped."""
        fpath = _write(
            tmp_path,
            "streams_spec_bad_rt.dat",
            """\
            #4.2
            1
            2
            1   1   0   TestReach
            1   10
            C  Rating tables section
            1.0
            1.0
            1min
            C  Rating table for node 1
            1   100.0   0.0   0.0
            bad   xyz
            """,
        )
        n_reaches, n_rtb, reaches = StreamSpecReader().read(fpath)

        # First rating table point parsed from 4-col line, bad one skipped
        assert 1 in reaches[0].node_rating_tables
        stages, flows = reaches[0].node_rating_tables[1]
        assert len(stages) == 1

    def test_spec_reader_version_without_hash(self, tmp_path: Path) -> None:
        """Spec file with no # version header (lines 1305, 1308-1313).

        When _read_version encounters a non-comment, non-# data line, it
        breaks out returning ''. That consumed line ('1') is lost, so
        subsequent parsing gets misaligned and raises FileFormatError.
        """
        fpath = _write(
            tmp_path,
            "streams_spec_no_ver.dat",
            """\
            C  comment only
            c  another comment
            1
            0
            1   1   0   Reach1
            1   5
            """,
        )
        # _read_version reads "1" (first non-comment line), sees it isn't
        # a # version -> break, returns "". This causes misaligned parsing.
        with pytest.raises(FileFormatError):
            StreamSpecReader().read(fpath)

    def test_node_gw_node_zero(self, tmp_path: Path) -> None:
        """gw_node_id=0 is not added to node_to_gw_node (lines 1369)."""
        fpath = _write(
            tmp_path,
            "streams_spec_gw0.dat",
            """\
            #4.2
            1
            0
            1   2   0   TestReach
            1   0
            2   10
            """,
        )
        n_reaches, n_rtb, reaches = StreamSpecReader().read(fpath)

        assert 1 not in reaches[0].node_to_gw_node
        assert reaches[0].node_to_gw_node[2] == 10

    def test_spec_reader_empty_version_line(self, tmp_path: Path) -> None:
        """Version line that is blank is skipped (line 1305)."""
        fpath = _write(
            tmp_path,
            "streams_spec_blank.dat",
            """\

            #4.2
            1
            0
            1   1   0   TestReach
            1   10
            """,
        )
        n_reaches, n_rtb, reaches = StreamSpecReader().read(fpath)
        assert n_reaches == 1


# =============================================================================
# Convenience functions
# =============================================================================


class TestConvenienceFunctionsDeep:
    """Tests for read_stream_main_file and read_stream_spec convenience functions."""

    def test_read_stream_main_file_convenience(self, tmp_path: Path) -> None:
        """Test read_stream_main_file() wrapper (lines 1472-1473)."""
        fpath = _write(
            tmp_path,
            "stream.dat",
            """\
            #4.1
            inflows.dat
            divspec.dat
            bypspec.dat
            div_ts.dat
            budget.hdf
            divbudget.hdf
            0
            0
            1.0
            1DAY
            1.0
            1
            """,
        )
        cfg = read_stream_main_file(fpath, base_dir=tmp_path)
        assert cfg.version == "4.1"

    def test_read_stream_main_file_no_base_dir(self, tmp_path: Path) -> None:
        """Test read_stream_main_file() uses filepath.parent when no base_dir."""
        fpath = _write(
            tmp_path,
            "stream2.dat",
            """\
            #4.1
            inflows.dat
            divspec.dat
            bypspec.dat
            div_ts.dat
            budget.hdf
            divbudget.hdf
            0
            0
            1.0
            1DAY
            1.0
            1
            """,
        )
        cfg = read_stream_main_file(fpath)
        assert cfg.inflow_file == tmp_path / "inflows.dat"

    def test_read_stream_spec_convenience(self, tmp_path: Path) -> None:
        """Test read_stream_spec() wrapper (lines 1497-1498)."""
        fpath = _write(
            tmp_path,
            "spec.dat",
            """\
            #5.0
            1
            1   1   0   Reach1
            1   10
            """,
        )
        n_reaches, n_rtb, reaches = read_stream_spec(fpath)
        assert n_reaches == 1
        assert n_rtb == 0


# =============================================================================
# StreamReader: diversion comment continuation (line 615)
# =============================================================================


class TestStreamReaderDiversionComments:
    """Test that comments interspersed in diversion data are skipped (line 615)."""

    def test_diversions_with_inline_comments(self, tmp_path: Path) -> None:
        """Comments between diversion data rows should be skipped."""
        fpath = _write(
            tmp_path,
            "diversions.dat",
            """\
            C Diversions file
            2                               / NDIVERSIONS
            C  First diversion
            1        1 element         5      100.0000    1  Div1
            C  Second diversion
            2        2 subregion       3       50.0000    2  Div2
            """,
        )
        reader = StreamReader()
        diversions = reader.read_diversions(fpath)

        assert len(diversions) == 2
        assert diversions[1].name == "Div1"
        assert diversions[2].name == "Div2"


# =============================================================================
# parse_stream_version / stream_version_ge
# =============================================================================


class TestParseStreamVersion:
    """Tests for parse_stream_version and stream_version_ge."""

    def test_hyphenated_version(self) -> None:
        """Version with hyphen separator like '4-21'."""
        assert parse_stream_version("4-21") == (4, 21)

    def test_stream_version_ge_true(self) -> None:
        assert stream_version_ge("5.0", (4, 2)) is True

    def test_stream_version_ge_false(self) -> None:
        assert stream_version_ge("4.0", (4, 2)) is False

    def test_stream_version_ge_equal(self) -> None:
        assert stream_version_ge("4.2", (4, 2)) is True
