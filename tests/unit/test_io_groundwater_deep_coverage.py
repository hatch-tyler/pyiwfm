"""Deep coverage tests for io/groundwater.py module.

Targets uncovered lines in GWMainFileReader parsing methods,
including output file paths, error handlers, edge cases in
hydrograph/face-flow/aquifer-parameter/initial-heads parsing,
and convenience functions.

Missing lines targeted:
  977, 982, 989, 996, 1001, 1008, 1013, 1025-1026, 1046-1047,
  1068-1069, 1089-1091, 1099-1101, 1109-1111, 1120-1124,
  1180, 1191, 1215-1217, 1225, 1230-1237, 1254, 1264-1265,
  1303-1304, 1309, 1312, 1320-1321, 1365, 1381-1382, 1400-1401,
  1406-1409, 1412, 1480, 1483, 1487-1488, 1496, 1500, 1509-1510,
  1526, 1530, 1537-1538, 1544, 1595-1596, 1605-1606, 1621,
  1627-1628, 1651-1652, 1658, 1662, 1667-1668, 1671, 1774-1775
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from pyiwfm.io.groundwater import (
    GWMainFileConfig,
    GWMainFileReader,
    read_gw_main_file,
    read_subsidence,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gw_main_header_lines() -> list[str]:
    """Return the standard header + file-path lines for a GW main file.

    This includes version, sub-file paths, conversion factors, and units
    through the velocity output unit line.  Tests append section-specific
    content after this prefix.
    """
    return [
        "C  Groundwater component main file\n",
        "# 4.0\n",
        # Sub-file paths
        "  / BCFL\n",
        "  / TDFL\n",
        "  / PUMPFL\n",
        "  / SUBSFL\n",
        "  / OVRWRTFL\n",
        # Conversion factors and units
        "1.0  / FACTLTOU\n",
        "FT  / UNITLTOU\n",
        "1.0  / FACTVLOU\n",
        "TAF  / UNITVLOU\n",
        "1.0  / FACTVROU\n",
        "FT/DAY  / UNITVROU\n",
    ]


def _gw_main_output_files_all_populated() -> list[str]:
    """Return output file lines where every optional output file is set."""
    return [
        "velocity.out  / VELOUTFL\n",
        "vflow.out  / VFLOWOUTFL\n",
        "headall.hdf  / GWALLOUTFL\n",
        "head_tec.dat  / HTPOUTFL\n",
        "vel_tec.dat  / VTPOUTFL\n",
        "gw_budget.hdf  / GWBUDFL\n",
        "zbudget.hdf  / ZBUDFL\n",
        "final_heads.dat  / FNGWFL\n",
    ]


def _gw_main_output_files_empty() -> list[str]:
    """Return output file lines where every optional output file is empty."""
    return [
        "  / VELOUTFL\n",
        "  / VFLOWOUTFL\n",
        "  / GWALLOUTFL\n",
        "  / HTPOUTFL\n",
        "  / VTPOUTFL\n",
        "  / GWBUDFL\n",
        "  / ZBUDFL\n",
        "  / FNGWFL\n",
    ]


def _gw_main_tail_minimal() -> list[str]:
    """Minimal tail: debug=0, no hydrographs, no face flows, no aq params."""
    return [
        "0  / KDEB\n",
        "0  / NOUTH\n",
        "1.0  / FACTXY\n",
        "  / GWHYDOUTFL\n",
        "0  / NOUTF\n",
        "  / FCHYDOUTFL\n",
    ]


def _write_file(path: Path, lines: list[str]) -> None:
    path.write_text("".join(lines))


# ---------------------------------------------------------------------------
# 1. Output file path resolution (lines 977, 982, 989, 996, 1001, 1008, 1013)
# ---------------------------------------------------------------------------


class TestOutputFilePaths:
    """Cover the 7 optional output file path assignments."""

    def test_all_output_files_populated(self, tmp_path: Path) -> None:
        """When all output files are specified, config attributes are set."""
        filepath = tmp_path / "gw_main.dat"
        lines = (
            _gw_main_header_lines()
            + _gw_main_output_files_all_populated()
            + _gw_main_tail_minimal()
        )
        _write_file(filepath, lines)

        config = GWMainFileReader().read(filepath)

        assert config.velocity_output_file is not None
        assert config.velocity_output_file.name == "velocity.out"

        assert config.vertical_flow_output_file is not None
        assert config.vertical_flow_output_file.name == "vflow.out"

        assert config.head_all_output_file is not None
        assert config.head_all_output_file.name == "headall.hdf"

        assert config.head_tecplot_file is not None
        assert config.head_tecplot_file.name == "head_tec.dat"

        assert config.velocity_tecplot_file is not None
        assert config.velocity_tecplot_file.name == "vel_tec.dat"

        assert config.budget_output_file is not None
        assert config.budget_output_file.name == "gw_budget.hdf"

        assert config.zbudget_output_file is not None
        assert config.zbudget_output_file.name == "zbudget.hdf"

        assert config.final_heads_file is not None
        assert config.final_heads_file.name == "final_heads.dat"


# ---------------------------------------------------------------------------
# 2. Debug flag ValueError (lines 1025-1026)
# ---------------------------------------------------------------------------


class TestDebugFlagValueError:
    """Cover the ValueError branch when KDEB is not an integer."""

    def test_non_integer_debug_flag(self, tmp_path: Path) -> None:
        filepath = tmp_path / "gw_main.dat"
        lines = (
            _gw_main_header_lines()
            + _gw_main_output_files_empty()
            + [
                "abc  / KDEB\n",  # non-integer debug flag
                "0  / NOUTH\n",
                "1.0  / FACTXY\n",
                "  / GWHYDOUTFL\n",
                "0  / NOUTF\n",
                "  / FCHYDOUTFL\n",
            ]
        )
        _write_file(filepath, lines)

        config = GWMainFileReader().read(filepath)
        # Default debug_flag should remain 0 on ValueError
        assert config.debug_flag == 0


# ---------------------------------------------------------------------------
# 3. Coord factor ValueError (lines 1046-1047)
# ---------------------------------------------------------------------------


class TestCoordFactorValueError:
    """Cover the ValueError branch when FACTXY is not a float."""

    def test_non_float_coord_factor(self, tmp_path: Path) -> None:
        filepath = tmp_path / "gw_main.dat"
        lines = (
            _gw_main_header_lines()
            + _gw_main_output_files_empty()
            + [
                "0  / KDEB\n",
                "0  / NOUTH\n",
                "xyz  / FACTXY\n",  # non-float coord factor
                "  / GWHYDOUTFL\n",
                "0  / NOUTF\n",
                "  / FCHYDOUTFL\n",
            ]
        )
        _write_file(filepath, lines)

        config = GWMainFileReader().read(filepath)
        # Default coord_factor should remain 1.0 on ValueError
        assert config.coord_factor == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 4. Face flow outputs ValueError (lines 1068-1069)
# ---------------------------------------------------------------------------


class TestFaceFlowOutputsValueError:
    """Cover the ValueError branch when NOUTF is not an integer."""

    def test_non_integer_noutf(self, tmp_path: Path) -> None:
        filepath = tmp_path / "gw_main.dat"
        lines = (
            _gw_main_header_lines()
            + _gw_main_output_files_empty()
            + [
                "0  / KDEB\n",
                "0  / NOUTH\n",
                "1.0  / FACTXY\n",
                "  / GWHYDOUTFL\n",
                "abc  / NOUTF\n",  # non-integer face flow count
                "  / FCHYDOUTFL\n",
            ]
        )
        _write_file(filepath, lines)

        config = GWMainFileReader().read(filepath)
        # n_face_flow_outputs should remain 0 on ValueError
        assert config.n_face_flow_outputs == 0


# ---------------------------------------------------------------------------
# 5. Exception handlers for aquifer params, kh anomaly, initial heads
#    (lines 1089-1091, 1099-1101, 1109-1111)
# ---------------------------------------------------------------------------


class TestSectionExceptionHandlers:
    """Cover the try/except blocks around section readers."""

    def test_aquifer_params_exception_logged(self, tmp_path: Path) -> None:
        """When _read_aquifer_parameters raises, it is caught and logged."""
        filepath = tmp_path / "gw_main.dat"
        lines = (
            _gw_main_header_lines()
            + _gw_main_output_files_empty()
            + [
                "0  / KDEB\n",
                "0  / NOUTH\n",
                "1.0  / FACTXY\n",
                "  / GWHYDOUTFL\n",
                "0  / NOUTF\n",
                "  / FCHYDOUTFL\n",
            ]
            # No aquifer params section at all -- will cause exception
        )
        _write_file(filepath, lines)

        reader = GWMainFileReader()
        with patch.object(
            reader, "_read_aquifer_parameters", side_effect=RuntimeError("boom")
        ):
            config = reader.read(filepath)
        # Should not raise; aquifer_params remains None
        assert config.aquifer_params is None

    def test_kh_anomaly_exception_logged(self, tmp_path: Path) -> None:
        """When _read_kh_anomaly raises, it is caught and logged."""
        filepath = tmp_path / "gw_main.dat"
        lines = (
            _gw_main_header_lines()
            + _gw_main_output_files_empty()
            + [
                "0  / KDEB\n",
                "0  / NOUTH\n",
                "1.0  / FACTXY\n",
                "  / GWHYDOUTFL\n",
                "0  / NOUTF\n",
                "  / FCHYDOUTFL\n",
                # Minimal aquifer params
                "0  / NGROUP\n",
                "1.0 1.0 1.0 1.0 1.0 1.0\n",
                "1DAY\n",
                "1DAY\n",
                "1DAY\n",
                "1  10.0  0.001  0.15  0.01  5.0\n",
                "C  End aquifer params\n",
            ]
        )
        _write_file(filepath, lines)

        reader = GWMainFileReader()
        with patch.object(
            reader, "_read_kh_anomaly", side_effect=RuntimeError("boom")
        ):
            config = reader.read(filepath)
        # Should not raise; kh_anomalies remains empty
        assert config.kh_anomalies == []

    def test_initial_heads_exception_logged(self, tmp_path: Path) -> None:
        """When _read_initial_heads raises, it is caught and logged."""
        filepath = tmp_path / "gw_main.dat"
        lines = (
            _gw_main_header_lines()
            + _gw_main_output_files_empty()
            + [
                "0  / KDEB\n",
                "0  / NOUTH\n",
                "1.0  / FACTXY\n",
                "  / GWHYDOUTFL\n",
                "0  / NOUTF\n",
                "  / FCHYDOUTFL\n",
                # Minimal aquifer params
                "0  / NGROUP\n",
                "1.0 1.0 1.0 1.0 1.0 1.0\n",
                "1DAY\n",
                "1DAY\n",
                "1DAY\n",
                "1  10.0  0.001  0.15  0.01  5.0\n",
                "C  End aquifer params\n",
                # Kh anomaly
                "0  / NEBK\n",
            ]
        )
        _write_file(filepath, lines)

        reader = GWMainFileReader()
        with patch.object(
            reader, "_read_initial_heads", side_effect=RuntimeError("boom")
        ):
            config = reader.read(filepath)
        # Should not raise; initial_heads remains None
        assert config.initial_heads is None


# ---------------------------------------------------------------------------
# 6. _read_version edge cases (lines 1120-1124)
# ---------------------------------------------------------------------------


class TestReadVersionBlankLines:
    """Cover the blank-line skip inside _read_version."""

    def test_version_after_blank_lines(self, tmp_path: Path) -> None:
        """Blank lines before the version header are skipped."""
        filepath = tmp_path / "gw_main.dat"
        lines = [
            "\n",  # blank line
            "   \n",  # whitespace-only line
            "# 4.2\n",
            # Minimal rest
            "  / BCFL\n",
            "  / TDFL\n",
            "  / PUMPFL\n",
            "  / SUBSFL\n",
            "  / OVRWRTFL\n",
            "1.0\n",
            "FT\n",
            "1.0\n",
            "TAF\n",
            "1.0\n",
            "FT/DAY\n",
        ] + _gw_main_output_files_empty() + _gw_main_tail_minimal()
        _write_file(filepath, lines)

        config = GWMainFileReader().read(filepath)
        assert config.version == "4.2"


# ---------------------------------------------------------------------------
# 7. Hydrograph parsing edge cases (lines 1180, 1191, 1215-1217)
# ---------------------------------------------------------------------------


class TestHydrographParsingEdgeCases:
    """Cover skipped/malformed hydrograph lines."""

    def _build_with_hydrograph_lines(
        self, tmp_path: Path, hyd_lines: list[str], n_hydro: int = 2
    ) -> Path:
        filepath = tmp_path / "gw_main.dat"
        lines = (
            _gw_main_header_lines()
            + _gw_main_output_files_empty()
            + [
                "0  / KDEB\n",
                f"{n_hydro}  / NOUTH\n",
                "1.0  / FACTXY\n",
                "hydout.dat  / GWHYDOUTFL\n",
            ]
            + hyd_lines
            + [
                "0  / NOUTF\n",
                "  / FCHYDOUTFL\n",
            ]
        )
        _write_file(filepath, lines)
        return filepath

    def test_short_hydrograph_line_skipped(self, tmp_path: Path) -> None:
        """Lines with < 4 parts are skipped (line 1180)."""
        filepath = self._build_with_hydrograph_lines(
            tmp_path,
            [
                "1  1\n",  # only 2 parts -- skipped
                "2  1  1  10  Station_B\n",
            ],
            n_hydro=1,
        )
        config = GWMainFileReader().read(filepath)
        assert len(config.hydrograph_locations) == 1
        assert config.hydrograph_locations[0].name == "Station_B"

    def test_hydtyp0_with_less_than_5_parts(self, tmp_path: Path) -> None:
        """HYDTYP=0 line with < 5 parts is skipped (line 1191)."""
        filepath = self._build_with_hydrograph_lines(
            tmp_path,
            [
                "1  0  1  1000.0\n",  # only 4 parts (missing Y) -- skipped
                "2  1  1  20  Station_B\n",
            ],
            n_hydro=1,
        )
        config = GWMainFileReader().read(filepath)
        assert len(config.hydrograph_locations) == 1
        assert config.hydrograph_locations[0].node_id == 20

    def test_malformed_hydrograph_data_skipped(self, tmp_path: Path) -> None:
        """Non-numeric data triggers ValueError and is skipped (lines 1215-1217)."""
        filepath = self._build_with_hydrograph_lines(
            tmp_path,
            [
                "abc  def  ghi  jkl\n",  # all non-numeric
                "2  1  1  20  Good_Station\n",
            ],
            n_hydro=1,
        )
        config = GWMainFileReader().read(filepath)
        assert len(config.hydrograph_locations) == 1
        assert config.hydrograph_locations[0].name == "Good_Station"


# ---------------------------------------------------------------------------
# 8. _resolve_path with absolute path (line 1225)
# ---------------------------------------------------------------------------


class TestResolvePathAbsolute:
    """Cover _resolve_path when the filepath is already absolute."""

    def test_absolute_path_returned_unchanged(self) -> None:
        """An absolute file path is returned unchanged by _resolve_path."""
        import io

        reader = GWMainFileReader()
        # Directly test the private method to avoid IWFM comment-char
        # issues with Windows paths starting with 'C' (treated as comment).
        abs_path = Path("/some/absolute/path/bc.dat")
        result = reader._resolve_path(Path("/base"), str(abs_path))
        assert result == abs_path


# ---------------------------------------------------------------------------
# 9. _skip_data_lines (lines 1230-1237)
# ---------------------------------------------------------------------------


class TestSkipDataLines:
    """Cover the _skip_data_lines helper method."""

    def test_skip_data_lines_skips_comments(self) -> None:
        """_skip_data_lines skips comment lines and counts only data."""
        import io

        reader = GWMainFileReader()
        reader._line_num = 0
        text = (
            "C  comment\n"
            "data line 1\n"
            "C  another comment\n"
            "data line 2\n"
            "data line 3\n"
        )
        f = io.StringIO(text)
        reader._skip_data_lines(f, 2)
        # After skipping 2 data lines, line_num should be 4
        # (comment at 1, data at 2, comment at 3, data at 4)
        assert reader._line_num == 4


# ---------------------------------------------------------------------------
# 10. Face flow spec edge cases (lines 1254, 1264-1265)
# ---------------------------------------------------------------------------


class TestFaceFlowSpecEdgeCases:
    """Cover short-line skip and ValueError in face flow parsing."""

    def test_short_face_flow_line_skipped(self, tmp_path: Path) -> None:
        """Face flow line with < 4 parts is skipped (line 1254)."""
        filepath = tmp_path / "gw_main.dat"
        lines = (
            _gw_main_header_lines()
            + _gw_main_output_files_empty()
            + [
                "0  / KDEB\n",
                "0  / NOUTH\n",
                "1.0  / FACTXY\n",
                "  / GWHYDOUTFL\n",
                "2  / NOUTF\n",
                "face.dat  / FCHYDOUTFL\n",
                "1  2\n",  # only 2 parts -- skipped
                "2  1  10  20  GoodSpec\n",
            ]
        )
        _write_file(filepath, lines)

        config = GWMainFileReader().read(filepath)
        assert len(config.face_flow_specs) == 1
        assert config.face_flow_specs[0].name == "GoodSpec"

    def test_malformed_face_flow_line_skipped(self, tmp_path: Path) -> None:
        """Non-numeric face flow data triggers ValueError (lines 1264-1265)."""
        filepath = tmp_path / "gw_main.dat"
        lines = (
            _gw_main_header_lines()
            + _gw_main_output_files_empty()
            + [
                "0  / KDEB\n",
                "0  / NOUTH\n",
                "1.0  / FACTXY\n",
                "  / GWHYDOUTFL\n",
                "2  / NOUTF\n",
                "face.dat  / FCHYDOUTFL\n",
                "abc  def  ghi  jkl\n",  # non-numeric
                "1  1  10  20  GoodSpec\n",
            ]
        )
        _write_file(filepath, lines)

        config = GWMainFileReader().read(filepath)
        assert len(config.face_flow_specs) == 1
        assert config.face_flow_specs[0].name == "GoodSpec"


# ---------------------------------------------------------------------------
# 11. Aquifer parameters early returns (lines 1303-1304, 1309, 1312, 1320-1321)
# ---------------------------------------------------------------------------


class TestAquiferParamsEarlyReturns:
    """Cover the early return paths in _read_aquifer_parameters."""

    def test_empty_ngroup_returns_none(self, tmp_path: Path) -> None:
        """Empty NGROUP string returns None (lines 1303-1304)."""
        filepath = tmp_path / "gw_main.dat"
        lines = (
            _gw_main_header_lines()
            + _gw_main_output_files_empty()
            + [
                "0  / KDEB\n",
                "0  / NOUTH\n",
                "1.0  / FACTXY\n",
                "  / GWHYDOUTFL\n",
                "0  / NOUTF\n",
                "  / FCHYDOUTFL\n",
                # NGROUP is empty -> triggers early return None
                "  / NGROUP\n",
            ]
        )
        _write_file(filepath, lines)

        config = GWMainFileReader().read(filepath)
        assert config.aquifer_params is None

    def test_non_integer_ngroup_returns_none(self, tmp_path: Path) -> None:
        """Non-integer NGROUP returns None (ValueError at line 1303-1304)."""
        filepath = tmp_path / "gw_main.dat"
        lines = (
            _gw_main_header_lines()
            + _gw_main_output_files_empty()
            + [
                "0  / KDEB\n",
                "0  / NOUTH\n",
                "1.0  / FACTXY\n",
                "  / GWHYDOUTFL\n",
                "0  / NOUTF\n",
                "  / FCHYDOUTFL\n",
                "abc  / NGROUP\n",  # non-integer
            ]
        )
        _write_file(filepath, lines)

        config = GWMainFileReader().read(filepath)
        assert config.aquifer_params is None

    def test_empty_factors_returns_none(self, tmp_path: Path) -> None:
        """Empty conversion factors string returns None (line 1309)."""
        filepath = tmp_path / "gw_main.dat"
        lines = (
            _gw_main_header_lines()
            + _gw_main_output_files_empty()
            + [
                "0  / KDEB\n",
                "0  / NOUTH\n",
                "1.0  / FACTXY\n",
                "  / GWHYDOUTFL\n",
                "0  / NOUTF\n",
                "  / FCHYDOUTFL\n",
                "0  / NGROUP\n",
                "  / FACTORS\n",  # empty factors string
            ]
        )
        _write_file(filepath, lines)

        config = GWMainFileReader().read(filepath)
        assert config.aquifer_params is None

    def test_too_few_factors_returns_none(self, tmp_path: Path) -> None:
        """Fewer than 6 conversion factor values returns None (line 1312)."""
        filepath = tmp_path / "gw_main.dat"
        lines = (
            _gw_main_header_lines()
            + _gw_main_output_files_empty()
            + [
                "0  / KDEB\n",
                "0  / NOUTH\n",
                "1.0  / FACTXY\n",
                "  / GWHYDOUTFL\n",
                "0  / NOUTF\n",
                "  / FCHYDOUTFL\n",
                "0  / NGROUP\n",
                "1.0 1.0 1.0\n",  # only 3 factors (need 6)
            ]
        )
        _write_file(filepath, lines)

        config = GWMainFileReader().read(filepath)
        assert config.aquifer_params is None

    def test_non_float_factors_returns_none(self, tmp_path: Path) -> None:
        """Non-float values in factors returns None (lines 1320-1321)."""
        filepath = tmp_path / "gw_main.dat"
        lines = (
            _gw_main_header_lines()
            + _gw_main_output_files_empty()
            + [
                "0  / KDEB\n",
                "0  / NOUTH\n",
                "1.0  / FACTXY\n",
                "  / GWHYDOUTFL\n",
                "0  / NOUTF\n",
                "  / FCHYDOUTFL\n",
                "0  / NGROUP\n",
                "1.0 abc 1.0 1.0 1.0 1.0\n",  # abc is not a float
            ]
        )
        _write_file(filepath, lines)

        config = GWMainFileReader().read(filepath)
        assert config.aquifer_params is None


# ---------------------------------------------------------------------------
# 12. Aquifer params: empty parts line, unexpected format, empty node_ids
#     (lines 1365, 1381-1382, 1400-1401, 1406-1409, 1412)
# ---------------------------------------------------------------------------


class TestAquiferParamsParsingEdgeCases:
    """Cover edge cases in per-node aquifer parameter parsing."""

    def _build_aq_params_file(
        self, tmp_path: Path, aq_lines: list[str]
    ) -> Path:
        filepath = tmp_path / "gw_main.dat"
        lines = (
            _gw_main_header_lines()
            + _gw_main_output_files_empty()
            + [
                "0  / KDEB\n",
                "0  / NOUTH\n",
                "1.0  / FACTXY\n",
                "  / GWHYDOUTFL\n",
                "0  / NOUTF\n",
                "  / FCHYDOUTFL\n",
                "0  / NGROUP\n",
                "1.0 1.0 1.0 1.0 1.0 1.0\n",
                "1DAY\n",
                "1DAY\n",
                "1DAY\n",
            ]
            + aq_lines
        )
        _write_file(filepath, lines)
        return filepath

    def test_empty_data_line_skipped(self, tmp_path: Path) -> None:
        """A data line that parses to empty parts is skipped (line 1365)."""
        filepath = self._build_aq_params_file(
            tmp_path,
            [
                "  / empty value line\n",  # parsed value is empty string
                "1  10.0  0.001  0.15  0.01  5.0\n",
                "C  End\n",
                "0  / NEBK\n",
                "1.0\n",
                "1  50.0\n",
            ],
        )
        config = GWMainFileReader().read(filepath)
        assert config.aquifer_params is not None
        assert config.aquifer_params.n_nodes == 1

    def test_value_error_in_6field_line_breaks(self, tmp_path: Path) -> None:
        """ValueError in a 6-field line ends section (lines 1381-1382).

        The first node must be finalized (via a second good node line or
        comment) before the bad line appears.  Here, two good nodes are
        read first; the third triggers ValueError and breaks.
        """
        filepath = self._build_aq_params_file(
            tmp_path,
            [
                "1  10.0  0.001  0.15  0.01  5.0\n",
                "2  12.0  0.002  0.18  0.02  6.0\n",
                "abc  def  ghi  jkl  mno  pqr\n",  # 6 fields, non-numeric
                "0  / NEBK\n",
                "1.0\n",
                "1  50.0\n",
                "2  55.0\n",
            ],
        )
        config = GWMainFileReader().read(filepath)
        assert config.aquifer_params is not None
        # Only the first node was finalized before ValueError
        assert config.aquifer_params.n_nodes == 1

    def test_value_error_in_5field_continuation_breaks(
        self, tmp_path: Path
    ) -> None:
        """ValueError in a 5-field continuation line ends section (lines 1400-1401).

        The first node must be finalized first.  Here, node 1 is complete
        (1 layer), node 2 starts, then its continuation line is bad.
        """
        filepath = self._build_aq_params_file(
            tmp_path,
            [
                "1  10.0  0.001  0.15  0.01  5.0\n",
                "2  12.0  0.002  0.18  0.02  6.0\n",
                "abc  def  ghi  jkl  mno\n",  # 5 fields, non-numeric continuation
                "0  / NEBK\n",
                "1.0\n",
                "1  50.0\n",
                "2  55.0\n",
            ],
        )
        config = GWMainFileReader().read(filepath)
        assert config.aquifer_params is not None
        # Node 1 was finalized when node 2 started.
        # Node 2 was being read when bad continuation hit; it is lost.
        assert config.aquifer_params.n_nodes == 1

    def test_unexpected_field_count_ends_section(self, tmp_path: Path) -> None:
        """A line with neither 5 nor 6 fields ends the section (lines 1406-1409)."""
        filepath = self._build_aq_params_file(
            tmp_path,
            [
                "1  10.0  0.001  0.15  0.01  5.0\n",
                "1  2  3  4  5  6  7\n",  # 7 fields -- unexpected
                "0  / NEBK\n",
                "1.0\n",
                "1  50.0\n",
            ],
        )
        config = GWMainFileReader().read(filepath)
        assert config.aquifer_params is not None
        assert config.aquifer_params.n_nodes == 1

    def test_no_node_data_returns_none(self, tmp_path: Path) -> None:
        """If no node data lines found, returns None (line 1412)."""
        filepath = self._build_aq_params_file(
            tmp_path,
            [
                "C  Comment immediately after time units\n",
                "C  End with no data lines\n",
            ],
        )
        config = GWMainFileReader().read(filepath)
        # No node data means aquifer_params should be None
        assert config.aquifer_params is None


# ---------------------------------------------------------------------------
# 13. Parametric grid edge cases (lines 1480, 1483, 1487-1488,
#     1496, 1500, 1509-1510, 1526, 1530, 1537-1538, 1544)
# ---------------------------------------------------------------------------


class TestParametricGridEdgeCases:
    """Cover edge cases in _read_parametric_aquifer_params."""

    def _build_parametric_file(
        self, tmp_path: Path, param_lines: list[str]
    ) -> Path:
        filepath = tmp_path / "gw_main.dat"
        lines = (
            _gw_main_header_lines()
            + _gw_main_output_files_empty()
            + [
                "0  / KDEB\n",
                "0  / NOUTH\n",
                "1.0  / FACTXY\n",
                "  / GWHYDOUTFL\n",
                "0  / NOUTF\n",
                "  / FCHYDOUTFL\n",
            ]
            + param_lines
        )
        _write_file(filepath, lines)
        return filepath

    def test_empty_ndp_nep_string_breaks(self, tmp_path: Path) -> None:
        """Empty NDP NEP string breaks the loop (line 1480)."""
        filepath = self._build_parametric_file(
            tmp_path,
            [
                "1  / NGROUP\n",
                "1.0 1.0 1.0 1.0 1.0 1.0\n",
                "1DAY\n",
                "1DAY\n",
                "1DAY\n",
                "  / NDP NEP\n",  # empty
            ],
        )
        config = GWMainFileReader().read(filepath)
        assert config.parametric_grids == []

    def test_single_value_ndp_nep_breaks(self, tmp_path: Path) -> None:
        """NDP NEP line with only 1 part breaks (line 1483)."""
        filepath = self._build_parametric_file(
            tmp_path,
            [
                "1  / NGROUP\n",
                "1.0 1.0 1.0 1.0 1.0 1.0\n",
                "1DAY\n",
                "1DAY\n",
                "1DAY\n",
                "3\n",  # only 1 part
            ],
        )
        config = GWMainFileReader().read(filepath)
        assert config.parametric_grids == []

    def test_non_integer_ndp_nep_breaks(self, tmp_path: Path) -> None:
        """Non-integer NDP or NEP breaks (lines 1487-1488)."""
        filepath = self._build_parametric_file(
            tmp_path,
            [
                "1  / NGROUP\n",
                "1.0 1.0 1.0 1.0 1.0 1.0\n",
                "1DAY\n",
                "1DAY\n",
                "1DAY\n",
                "abc  def\n",  # non-integer
            ],
        )
        config = GWMainFileReader().read(filepath)
        assert config.parametric_grids == []

    def test_element_line_too_short_breaks(self, tmp_path: Path) -> None:
        """Element line with < 4 parts breaks (line 1500)."""
        filepath = self._build_parametric_file(
            tmp_path,
            [
                "1  / NGROUP\n",
                "1.0 1.0 1.0 1.0 1.0 1.0\n",
                "1DAY\n",
                "1DAY\n",
                "1DAY\n",
                "3  2\n",  # 3 nodes, 2 elements
                "1  1  2\n",  # only 3 parts (need >= 4)
            ],
        )
        config = GWMainFileReader().read(filepath)
        # Grid should still be attempted but elements list will be short
        # The node reading loop may also fail, but that is okay
        # The important thing is no crash
        assert config.aquifer_params is None

    def test_element_value_error_breaks(self, tmp_path: Path) -> None:
        """Non-integer element data triggers ValueError (lines 1509-1510)."""
        filepath = self._build_parametric_file(
            tmp_path,
            [
                "1  / NGROUP\n",
                "1.0 1.0 1.0 1.0 1.0 1.0\n",
                "1DAY\n",
                "1DAY\n",
                "1DAY\n",
                "3  1\n",  # 3 nodes, 1 element
                "1  abc  def  ghi\n",  # non-integer node indices
            ],
        )
        config = GWMainFileReader().read(filepath)
        assert config.aquifer_params is None

    def test_node_line_too_short_breaks(self, tmp_path: Path) -> None:
        """Node data line with < 4 parts breaks (line 1530)."""
        filepath = self._build_parametric_file(
            tmp_path,
            [
                "1  / NGROUP\n",
                "1.0 1.0 1.0 1.0 1.0 1.0\n",
                "1DAY\n",
                "1DAY\n",
                "1DAY\n",
                "2  1\n",  # 2 nodes, 1 element
                "1  1  2  3\n",  # element def (triangle)
                "1  0.0\n",  # only 2 parts for node (need >= 4)
            ],
        )
        config = GWMainFileReader().read(filepath)
        assert config.aquifer_params is None

    def test_node_value_error_breaks(self, tmp_path: Path) -> None:
        """Non-numeric node data triggers ValueError (lines 1537-1538)."""
        filepath = self._build_parametric_file(
            tmp_path,
            [
                "1  / NGROUP\n",
                "1.0 1.0 1.0 1.0 1.0 1.0\n",
                "1DAY\n",
                "1DAY\n",
                "1DAY\n",
                "2  1\n",
                "1  1  2  3\n",
                "1  abc  def  ghi  jkl\n",  # non-numeric coords
            ],
        )
        config = GWMainFileReader().read(filepath)
        assert config.aquifer_params is None

    def test_empty_raw_values_skips_grid(self, tmp_path: Path) -> None:
        """If no node values parsed, the grid is skipped (line 1544)."""
        filepath = self._build_parametric_file(
            tmp_path,
            [
                "1  / NGROUP\n",
                "1.0 1.0 1.0 1.0 1.0 1.0\n",
                "1DAY\n",
                "1DAY\n",
                "1DAY\n",
                "1  1\n",  # 1 node, 1 element
                "1  1  2  3\n",
                # No node data lines follow => empty raw_values
                "C  End\n",
            ],
        )
        config = GWMainFileReader().read(filepath)
        assert config.parametric_grids == []

    def test_comment_in_element_section_skipped(self, tmp_path: Path) -> None:
        """Comment lines in element definitions are skipped (line 1496)."""
        filepath = self._build_parametric_file(
            tmp_path,
            [
                "1  / NGROUP\n",
                "1.0 1.0 1.0 1.0 1.0 1.0\n",
                "1DAY\n",
                "1DAY\n",
                "1DAY\n",
                "3  1\n",  # 3 nodes, 1 element
                "C  Element data below\n",
                "1  1  2  3\n",  # single triangle element
                "C  Node data below\n",
                "1  0.0  0.0  10.0  0.001  0.15  0.01  5.0\n",
                "2  100.0  0.0  12.0  0.002  0.18  0.02  6.0\n",
                "3  50.0  100.0  11.0  0.0015  0.16  0.015  5.5\n",
                # Kh anomaly
                "0  / NEBK\n",
                "1.0\n",
                "1  50.0\n",
                "2  55.0\n",
                "3  60.0\n",
            ],
        )
        config = GWMainFileReader().read(filepath)
        assert len(config.parametric_grids) == 1
        assert config.parametric_grids[0].n_nodes == 3


# ---------------------------------------------------------------------------
# 14. Kh anomaly parsing edge cases (lines 1595-1596, 1605-1606,
#     1621, 1627-1628)
# ---------------------------------------------------------------------------


class TestKhAnomalyEdgeCases:
    """Cover edge cases in _read_kh_anomaly."""

    def _build_kh_file(self, tmp_path: Path, kh_lines: list[str]) -> Path:
        filepath = tmp_path / "gw_main.dat"
        lines = (
            _gw_main_header_lines()
            + _gw_main_output_files_empty()
            + [
                "0  / KDEB\n",
                "0  / NOUTH\n",
                "1.0  / FACTXY\n",
                "  / GWHYDOUTFL\n",
                "0  / NOUTF\n",
                "  / FCHYDOUTFL\n",
                # Minimal aquifer params
                "0  / NGROUP\n",
                "1.0 1.0 1.0 1.0 1.0 1.0\n",
                "1DAY\n",
                "1DAY\n",
                "1DAY\n",
                "1  10.0  0.001  0.15  0.01  5.0\n",
                "C  End aquifer params\n",
            ]
            + kh_lines
        )
        _write_file(filepath, lines)
        return filepath

    def test_non_integer_nebk_returns_empty(self, tmp_path: Path) -> None:
        """Non-integer NEBK returns empty list (lines 1595-1596)."""
        filepath = self._build_kh_file(
            tmp_path,
            [
                "abc  / NEBK\n",  # non-integer
                "1.0\n",
                "1  50.0\n",
            ],
        )
        config = GWMainFileReader().read(filepath)
        assert config.kh_anomalies == []

    def test_non_float_fact_defaults_to_one(self, tmp_path: Path) -> None:
        """Non-float FACT defaults to 1.0 (lines 1605-1606)."""
        filepath = self._build_kh_file(
            tmp_path,
            [
                "1  / NEBK\n",
                "abc  / FACT\n",  # non-float fact
                "1DAY\n",
                "1  5  0.5\n",
                # Initial heads
                "1.0\n",
                "1  50.0\n",
            ],
        )
        config = GWMainFileReader().read(filepath)
        assert len(config.kh_anomalies) == 1
        # BK=0.5 * FACT=1.0 (default) = 0.5
        assert config.kh_anomalies[0].kh_per_layer == [pytest.approx(0.5)]

    def test_short_anomaly_line_breaks(self, tmp_path: Path) -> None:
        """Anomaly line with < 3 parts breaks the loop (line 1621)."""
        filepath = self._build_kh_file(
            tmp_path,
            [
                "2  / NEBK\n",
                "1.0  / FACT\n",
                "1DAY\n",
                "1  5\n",  # only 2 parts -- breaks
                # Initial heads
                "1.0\n",
                "1  50.0\n",
            ],
        )
        config = GWMainFileReader().read(filepath)
        assert config.kh_anomalies == []

    def test_non_numeric_anomaly_line_breaks(self, tmp_path: Path) -> None:
        """Non-numeric anomaly data breaks the loop (lines 1627-1628)."""
        filepath = self._build_kh_file(
            tmp_path,
            [
                "2  / NEBK\n",
                "1.0  / FACT\n",
                "1DAY\n",
                "abc  def  ghi\n",  # non-numeric
                "1.0\n",
                "1  50.0\n",
            ],
        )
        config = GWMainFileReader().read(filepath)
        assert config.kh_anomalies == []


# ---------------------------------------------------------------------------
# 15. Initial heads parsing edge cases (lines 1651-1652, 1658, 1662,
#     1667-1668, 1671)
# ---------------------------------------------------------------------------


class TestInitialHeadsEdgeCases:
    """Cover edge cases in _read_initial_heads."""

    def _build_initial_heads_file(
        self, tmp_path: Path, heads_lines: list[str]
    ) -> Path:
        filepath = tmp_path / "gw_main.dat"
        lines = (
            _gw_main_header_lines()
            + _gw_main_output_files_empty()
            + [
                "0  / KDEB\n",
                "0  / NOUTH\n",
                "1.0  / FACTXY\n",
                "  / GWHYDOUTFL\n",
                "0  / NOUTF\n",
                "  / FCHYDOUTFL\n",
                # Minimal aquifer params
                "0  / NGROUP\n",
                "1.0 1.0 1.0 1.0 1.0 1.0\n",
                "1DAY\n",
                "1DAY\n",
                "1DAY\n",
                "1  10.0  0.001  0.15  0.01  5.0\n",
                "C  End aquifer params\n",
                # Kh anomaly (none)
                "0  / NEBK\n",
            ]
            + heads_lines
        )
        _write_file(filepath, lines)
        return filepath

    def test_non_float_facthp_returns_none(self, tmp_path: Path) -> None:
        """Non-float FACTHP returns None (lines 1651-1652)."""
        filepath = self._build_initial_heads_file(
            tmp_path,
            ["abc  / FACTHP\n"],  # non-float
        )
        config = GWMainFileReader().read(filepath)
        assert config.initial_heads is None

    def test_comment_in_head_data_skipped(self, tmp_path: Path) -> None:
        """Comment lines between head data rows are skipped (line 1658)."""
        filepath = self._build_initial_heads_file(
            tmp_path,
            [
                "1.0  / FACTHP\n",
                "1  50.0\n",
                "C  A comment between data rows\n",
                "2  55.0\n",
            ],
        )
        config = GWMainFileReader().read(filepath)
        assert config.initial_heads is not None
        assert config.initial_heads.shape == (2, 1)

    def test_short_head_line_breaks(self, tmp_path: Path) -> None:
        """Head line with < 2 parts breaks (line 1662)."""
        filepath = self._build_initial_heads_file(
            tmp_path,
            [
                "1.0  / FACTHP\n",
                "1  50.0\n",
                "2\n",  # only 1 part -- breaks
            ],
        )
        config = GWMainFileReader().read(filepath)
        assert config.initial_heads is not None
        assert config.initial_heads.shape == (1, 1)

    def test_non_numeric_head_data_breaks(self, tmp_path: Path) -> None:
        """Non-numeric head data breaks the loop (lines 1667-1668)."""
        filepath = self._build_initial_heads_file(
            tmp_path,
            [
                "1.0  / FACTHP\n",
                "abc  def\n",  # non-numeric
            ],
        )
        config = GWMainFileReader().read(filepath)
        # No valid rows => returns None (line 1671)
        assert config.initial_heads is None

    def test_no_head_rows_returns_none(self, tmp_path: Path) -> None:
        """No head data rows after FACTHP returns None (line 1671)."""
        filepath = self._build_initial_heads_file(
            tmp_path,
            ["1.0  / FACTHP\n"],  # no data rows follow
        )
        config = GWMainFileReader().read(filepath)
        assert config.initial_heads is None


# ---------------------------------------------------------------------------
# 16. Convenience function: read_gw_main_file (lines 1774-1775)
# ---------------------------------------------------------------------------


class TestReadGwMainFileConvenience:
    """Cover the read_gw_main_file convenience function."""

    def test_read_gw_main_file(self, tmp_path: Path) -> None:
        filepath = tmp_path / "gw_main.dat"
        lines = (
            _gw_main_header_lines()
            + _gw_main_output_files_empty()
            + _gw_main_tail_minimal()
        )
        _write_file(filepath, lines)

        config = read_gw_main_file(filepath)
        assert isinstance(config, GWMainFileConfig)
        assert config.version == "4.0"

    def test_read_gw_main_file_with_base_dir(self, tmp_path: Path) -> None:
        filepath = tmp_path / "gw_main.dat"
        lines = (
            [
                "C  GW Main\n",
                "# 4.0\n",
                "bc.dat  / BCFL\n",
                "  / TDFL\n",
                "  / PUMPFL\n",
                "  / SUBSFL\n",
                "  / OVRWRTFL\n",
                "1.0\n",
                "FT\n",
                "1.0\n",
                "TAF\n",
                "1.0\n",
                "FT/DAY\n",
            ]
            + _gw_main_output_files_empty()
            + _gw_main_tail_minimal()
        )
        _write_file(filepath, lines)

        custom_base = tmp_path / "custom_base"
        custom_base.mkdir()
        config = read_gw_main_file(filepath, base_dir=custom_base)
        assert config.bc_file is not None
        assert str(custom_base) in str(config.bc_file)


# ---------------------------------------------------------------------------
# 17. Convenience function: read_subsidence (line 1746 analog)
# ---------------------------------------------------------------------------


class TestReadSubsidenceConvenience:
    """Cover the read_subsidence convenience function."""

    def test_read_subsidence_function(self, tmp_path: Path) -> None:
        filepath = tmp_path / "subsidence.dat"
        filepath.write_text(
            "C  Subsidence file\n"
            "2  / N_SUBSIDENCE\n"
            "1  1  1e-5  1e-4  80.0\n"
            "2  1  2e-5  2e-4  75.0\n"
        )
        result = read_subsidence(filepath)
        assert len(result) == 2
        assert result[0].element == 1
        assert result[1].element == 2
