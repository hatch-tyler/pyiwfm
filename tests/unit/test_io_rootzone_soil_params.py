"""Tests for the rootzone soil parameter reader bug fix.

The bug: C2VSimFG showed 32,536 soil parameter rows instead of 32,537.

Original cause: ``_next_data_or_empty`` stripped inline comments with a regex
that treated ``#`` as a comment delimiter, truncating data lines.

Second cause (off-by-one): pyiwfm unconditionally read a ``FinalMoistureOutFile``
line that does **not exist** in IWFM v4.12.  This consumed the FACTK value as a
file path, cascaded through subsequent fields, and ultimately ate the first soil
parameter row — yielding N-1 rows instead of N.

Fix: ``FinalMoistureOutFile`` is only read for versions < 4.12.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

from pyiwfm.io.rootzone import (
    ElementSoilParamRow,
    RootZoneMainFileConfig,
    RootZoneMainFileReader,
)


def _make_config(version: str = "4.12") -> RootZoneMainFileConfig:
    return RootZoneMainFileConfig(version=version)


def _make_soil_line_v412(elem_id: int) -> str:
    """Generate a v4.12 soil param row (16 columns)."""
    return (
        f"  {elem_id}  0.20  0.35  0.45  0.10  1.0  2.0  1  1  1"
        f"  1.0  1  1  1  1  1\n"
    )


class TestSoilParamsRead:
    """Test that _read_element_soil_params reads all expected rows."""

    def test_reads_all_n_elements(self) -> None:
        """Verify all N elements are read when n_elements is specified."""
        n = 10
        lines = "".join(_make_soil_line_v412(i + 1) for i in range(n))
        f = io.StringIO(lines)
        reader = RootZoneMainFileReader.__new__(RootZoneMainFileReader)
        reader._line_num = 0
        config = _make_config("4.12")
        rows = reader._read_element_soil_params(f, config, n_elements=n)
        assert len(rows) == n
        assert rows[0].element_id == 1
        assert rows[-1].element_id == n

    def test_inline_slash_comment_not_lost(self) -> None:
        """Inline '/ description' must NOT cause the row to be lost."""
        lines = (
            "  1  0.20  0.35  0.45  0.10  1.0  2.0  1  1  1"
            "  1.0  1  1  1  1  1  / soil type A\n"
            "  2  0.20  0.35  0.45  0.10  1.0  2.0  1  1  1"
            "  1.0  1  1  1  1  1  / soil type B\n"
        )
        f = io.StringIO(lines)
        reader = RootZoneMainFileReader.__new__(RootZoneMainFileReader)
        reader._line_num = 0
        config = _make_config("4.12")
        rows = reader._read_element_soil_params(f, config, n_elements=2)
        assert len(rows) == 2

    def test_inline_hash_treated_as_data(self) -> None:
        """'#' is NOT a comment delimiter in IWFM — must not truncate."""
        # Row with '#' after the 16 values (shouldn't affect parsing)
        lines = (
            "  1  0.20  0.35  0.45  0.10  1.0  2.0  1  1  1"
            "  1.0  1  1  1  1  1  # annotation\n"
        )
        f = io.StringIO(lines)
        reader = RootZoneMainFileReader.__new__(RootZoneMainFileReader)
        reader._line_num = 0
        config = _make_config("4.12")
        rows = reader._read_element_soil_params(f, config, n_elements=1)
        assert len(rows) == 1
        assert rows[0].element_id == 1

    def test_extra_tokens_after_16_ignored(self) -> None:
        """Extra tokens beyond min_cols should be ignored (Fortran-like)."""
        lines = (
            "  1  0.20  0.35  0.45  0.10  1.0  2.0  1  1  1"
            "  1.0  1  1  1  1  1  99  extra  tokens\n"
        )
        f = io.StringIO(lines)
        reader = RootZoneMainFileReader.__new__(RootZoneMainFileReader)
        reader._line_num = 0
        config = _make_config("4.12")
        rows = reader._read_element_soil_params(f, config, n_elements=1)
        assert len(rows) == 1
        assert rows[0].element_id == 1

    def test_skips_comment_lines(self) -> None:
        """Comment lines within the block should be skipped."""
        lines = (
            _make_soil_line_v412(1)
            + "C this is a comment\n"
            + "* another comment\n"
            + _make_soil_line_v412(2)
        )
        f = io.StringIO(lines)
        reader = RootZoneMainFileReader.__new__(RootZoneMainFileReader)
        reader._line_num = 0
        config = _make_config("4.12")
        rows = reader._read_element_soil_params(f, config, n_elements=2)
        assert len(rows) == 2

    def test_count_mismatch_warning(self) -> None:
        """Warning is logged when actual count differs from expected."""
        lines = _make_soil_line_v412(1) + _make_soil_line_v412(2)
        f = io.StringIO(lines)
        reader = RootZoneMainFileReader.__new__(RootZoneMainFileReader)
        reader._line_num = 0
        config = _make_config("4.12")
        with patch("pyiwfm.io.rootzone.logger") as mock_logger:
            rows = reader._read_element_soil_params(
                f, config, n_elements=5
            )
        assert len(rows) == 2
        mock_logger.warning.assert_called_once()

    def test_v41_reads_correctly(self) -> None:
        """v4.1 format (13 columns) reads correctly."""
        lines = (
            "  1  0.20  0.35  0.45  0.10  1.0  1  1  1  1.0  1  1  1\n"
            "  2  0.20  0.35  0.45  0.10  1.0  1  1  1  1.0  1  1  1\n"
        )
        f = io.StringIO(lines)
        reader = RootZoneMainFileReader.__new__(RootZoneMainFileReader)
        reader._line_num = 0
        config = _make_config("4.1")
        rows = reader._read_element_soil_params(f, config, n_elements=2)
        assert len(rows) == 2

    def test_v40_reads_correctly(self) -> None:
        """v4.0 format (12 columns) reads correctly."""
        lines = (
            "  1  0.20  0.35  0.45  0.10  1.0  1  1  1.0  1  1  1\n"
            "  2  0.20  0.35  0.45  0.10  1.0  1  1  1.0  1  1  1\n"
        )
        f = io.StringIO(lines)
        reader = RootZoneMainFileReader.__new__(RootZoneMainFileReader)
        reader._line_num = 0
        config = _make_config("4.0")
        rows = reader._read_element_soil_params(f, config, n_elements=2)
        assert len(rows) == 2


# ── Helpers for full-file read() tests ───────────────────────────────

def _v412_main_file(n_elements: int = 3) -> str:
    """Build a minimal v4.12 rootzone main file (no FMFL line).

    v4.12 read order after zone budget files:
      LU area scaling → FACTK → FACTEXDTH → TUNITK → SurfFlowDest → Soil params
    """
    lines = [
        "#4.12",
        "C Root Zone Main File",
        "  0.001                                        / RZCONV",
        "  150                                          / RZITERMX",
        "  0.0833333                                    / FACTCN",
        "  0                                            / GWUPTK",
        "                                               / AGNPFL",
        "                                               / PFL",
        "                                               / URBFL",
        "                                               / NVRVFL",
        "                                               / RFFL",
        "                                               / RUFL",
        "                                               / IPFL",
        "                                               / MSRCFL",
        "                                               / AGWDFL",
        "                                               / LWUBUDFL",
        "                                               / RZBUDFL",
        "                                               / ZLWUBUDFL",
        "                                               / ZRZBUDFL",
        "                                               / ARSCLFL",
        # NOTE: no FMFL line in v4.12!
        "  1.0                                          / FACTK",
        "  1.0                                          / FACTEXDTH",
        "  1DAY                                         / TUNITK",
        "  SurfFlowDest.dat                             / DESTFL",
        "C  IE  WP  FC  TN  LAMBDA  K  KPonded  RHC  CPRISE  IRNE  FRNE  IMSRC  ICDSTAG  ICDSTURBIN  ICDSTURBOUT  ICDSTNVRV",
    ]
    for i in range(1, n_elements + 1):
        lines.append(
            f"  {i}  0.20  0.35  0.45  0.10  2.60  -1.0  2  0.0  1  1.0  0  1  1  1  1"
        )
    return "\n".join(lines) + "\n"


def _v411_main_file(n_elements: int = 3) -> str:
    """Build a minimal v4.11 rootzone main file (with FMFL line).

    v4.11 read order after zone budget files:
      FinalMoistureOutFile → FACTK → FACTEXDTH → TUNITK → Soil params
    (No ARSCLFL in v4.11 — that was introduced in v4.12.)
    """
    lines = [
        "#4.11",
        "C Root Zone Main File",
        "  0.001                                        / RZCONV",
        "  150                                          / RZITERMX",
        "  0.0833333                                    / FACTCN",
        "  0                                            / GWUPTK",
        "                                               / AGNPFL",
        "                                               / PFL",
        "                                               / URBFL",
        "                                               / NVRVFL",
        "                                               / RFFL",
        "                                               / RUFL",
        "                                               / IPFL",
        "                                               / MSRCFL",
        "                                               / AGWDFL",
        "                                               / LWUBUDFL",
        "                                               / RZBUDFL",
        "                                               / ZLWUBUDFL",
        "                                               / ZRZBUDFL",
        "  FinalMoisture.bin                            / FMFL",
        "  1.0                                          / FACTK",
        "  1.0                                          / FACTEXDTH",
        "  1DAY                                         / TUNITK",
        # NOTE: no DESTFL line in v4.11!
        "C  IE  WP  FC  TN  LAMBDA  K  RHC  CPRISE  IRNE  FRNE  IMSRC  IDEST_TYPE  IDEST",
    ]
    for i in range(1, n_elements + 1):
        lines.append(
            f"  {i}  0.20  0.35  0.45  0.10  2.60  2  0.0  1  1.0  0  1  0"
        )
    return "\n".join(lines) + "\n"


class TestV412NoFinalMoistureFile:
    """v4.12 must NOT read a FinalMoistureOutFile line."""

    def test_v412_reads_all_soil_params(self, tmp_path: Path) -> None:
        """v4.12 file without FMFL must read exactly N soil param rows."""
        n = 5
        rz_file = tmp_path / "RootZone_MAIN.dat"
        rz_file.write_text(_v412_main_file(n_elements=n))

        reader = RootZoneMainFileReader()
        config = reader.read(rz_file, base_dir=tmp_path, n_elements=n)

        assert config.version == "4.12"
        assert len(config.element_soil_params) == n
        assert config.element_soil_params[0].element_id == 1
        assert config.element_soil_params[-1].element_id == n

    def test_v412_no_final_moisture_set(self, tmp_path: Path) -> None:
        """v4.12 must NOT set final_moisture_file."""
        rz_file = tmp_path / "RootZone_MAIN.dat"
        rz_file.write_text(_v412_main_file(n_elements=2))

        reader = RootZoneMainFileReader()
        config = reader.read(rz_file, base_dir=tmp_path, n_elements=2)

        assert config.final_moisture_file is None

    def test_v412_correct_soil_factors(self, tmp_path: Path) -> None:
        """v4.12 must parse FACTK, FACTEXDTH, TUNITK, and DESTFL correctly."""
        rz_file = tmp_path / "RootZone_MAIN.dat"
        rz_file.write_text(_v412_main_file(n_elements=1))

        reader = RootZoneMainFileReader()
        config = reader.read(rz_file, base_dir=tmp_path, n_elements=1)

        assert config.k_factor == 1.0
        assert config.k_exdth_factor == 1.0
        assert config.k_time_unit == "1DAY"
        assert config.surface_flow_dest_file is not None


class TestV411WithFinalMoistureFile:
    """v4.11 must read the FinalMoistureOutFile line."""

    def test_v411_reads_all_soil_params(self, tmp_path: Path) -> None:
        """v4.11 file with FMFL must read exactly N soil param rows."""
        n = 5
        rz_file = tmp_path / "RootZone_MAIN.dat"
        rz_file.write_text(_v411_main_file(n_elements=n))

        reader = RootZoneMainFileReader()
        config = reader.read(rz_file, base_dir=tmp_path, n_elements=n)

        assert config.version == "4.11"
        assert len(config.element_soil_params) == n
        assert config.element_soil_params[0].element_id == 1
        assert config.element_soil_params[-1].element_id == n

    def test_v411_has_final_moisture_file(self, tmp_path: Path) -> None:
        """v4.11 must set final_moisture_file from the FMFL line."""
        rz_file = tmp_path / "RootZone_MAIN.dat"
        rz_file.write_text(_v411_main_file(n_elements=1))

        reader = RootZoneMainFileReader()
        config = reader.read(rz_file, base_dir=tmp_path, n_elements=1)

        assert config.final_moisture_file is not None
        assert "FinalMoisture" in str(config.final_moisture_file)

    def test_v411_no_surface_flow_dest(self, tmp_path: Path) -> None:
        """v4.11 must NOT set surface_flow_dest_file (v4.12+ only)."""
        rz_file = tmp_path / "RootZone_MAIN.dat"
        rz_file.write_text(_v411_main_file(n_elements=1))

        reader = RootZoneMainFileReader()
        config = reader.read(rz_file, base_dir=tmp_path, n_elements=1)

        assert config.surface_flow_dest_file is None

    def test_v411_correct_soil_factors(self, tmp_path: Path) -> None:
        """v4.11 must parse FACTK, FACTEXDTH, and TUNITK correctly."""
        rz_file = tmp_path / "RootZone_MAIN.dat"
        rz_file.write_text(_v411_main_file(n_elements=1))

        reader = RootZoneMainFileReader()
        config = reader.read(rz_file, base_dir=tmp_path, n_elements=1)

        assert config.k_factor == 1.0
        assert config.k_exdth_factor == 1.0
        assert config.k_time_unit == "1DAY"
