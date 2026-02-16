"""Tests for the rootzone soil parameter reader bug fix.

The bug: C2VSimFG showed 32,536 soil parameter rows instead of 32,537
because ``_next_data_or_empty`` stripped inline comments with a regex
that treated ``#`` as a comment delimiter (``\\s+[#/]``), which could
truncate data lines and cause them to be skipped as "short".

The fix: use ``next_data_line`` (raw line, Fortran-like) so that the
caller can split and take exactly N tokens — matching Fortran's
free-format ``READ``.
"""

from __future__ import annotations

import io
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
