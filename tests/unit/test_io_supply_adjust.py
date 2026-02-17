"""Tests for supply adjustment reader and writer.

Covers:
- read_supply_adjustment
- write_supply_adjustment
- Roundtrip (write -> read)
- _is_fortran_comment helper
- SupplyAdjustment dataclass
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from pyiwfm.io.supply_adjust import (
    SupplyAdjustment,
    _is_fortran_comment,
    read_supply_adjustment,
    write_supply_adjustment,
)

# =============================================================================
# Helper function tests
# =============================================================================


class TestIsFortranComment:
    """Tests for _is_fortran_comment."""

    def test_c_uppercase(self) -> None:
        assert _is_fortran_comment("C This is a comment") is True

    def test_c_lowercase(self) -> None:
        assert _is_fortran_comment("c lowercase comment") is True

    def test_asterisk(self) -> None:
        assert _is_fortran_comment("* asterisk comment") is True

    def test_empty_line_not_comment(self) -> None:
        """Empty/blank lines are NOT comments (caller decides handling)."""
        assert _is_fortran_comment("") is False
        assert _is_fortran_comment("   ") is False

    def test_hash_not_line_comment(self) -> None:
        """Hash at start is NOT a line comment - it's inline only."""
        assert _is_fortran_comment("# Not a line comment") is False

    def test_data_line(self) -> None:
        assert _is_fortran_comment("    10    / NCOLADJ") is False

    def test_blank_dssfl_line(self) -> None:
        """Blank DSSFL line with inline comment should NOT be a comment."""
        assert _is_fortran_comment("                   / DSSFL") is False


# =============================================================================
# SupplyAdjustment dataclass tests
# =============================================================================


class TestSupplyAdjustmentDataclass:
    """Tests for SupplyAdjustment dataclass."""

    def test_defaults(self) -> None:
        sa = SupplyAdjustment()
        assert sa.n_columns == 0
        assert sa.nsp == 1
        assert sa.nfq == 0
        assert sa.dss_file == ""
        assert sa.times == []
        assert sa.values == []
        assert sa.header_lines == []

    def test_with_data(self) -> None:
        sa = SupplyAdjustment(
            n_columns=3,
            nsp=1,
            nfq=0,
            times=[datetime(2020, 11, 1), datetime(2020, 12, 1)],
            values=[[0, 10, 1], [0, 0, 1]],
        )
        assert sa.n_columns == 3
        assert len(sa.times) == 2
        assert sa.values[0] == [0, 10, 1]


# =============================================================================
# Reader tests
# =============================================================================


class TestReadSupplyAdjustment:
    """Tests for read_supply_adjustment."""

    def _write_file(self, path: Path, content: str) -> Path:
        filepath = path / "supply_adj.dat"
        filepath.write_text(content)
        return filepath

    def test_read_basic(self, tmp_path: Path) -> None:
        """Read basic supply adjustment file with inline data."""
        content = (
            "C*************************************************************\n"
            "C                     SUPPLY ADJUSTMENT SPECIFICATIONS\n"
            "C*************************************************************\n"
            "C\n"
            "C   NCOLADJ:  Number of columns\n"
            "C   NSPADJ :  Update frequency\n"
            "C   NFQADJ :  Repetition frequency\n"
            "C   DSSFL  :  DSS filename\n"
            "C\n"
            "          3                                     / NCOLADJ\n"
            "          1                                     / NSPADJ\n"
            "          0                                     / NFQADJ\n"
            "                                                / DSSFL\n"
            "C*************************************************************\n"
            "C                    Supply Adjustment Specifications Data\n"
            "C*************************************************************\n"
            "    10/31/1973_24:00\t00\t10\t01\n"
            "    11/30/1973_24:00\t00\t00\t01\n"
        )
        filepath = self._write_file(tmp_path, content)
        result = read_supply_adjustment(filepath)

        assert result.n_columns == 3
        assert result.nsp == 1
        assert result.nfq == 0
        assert result.dss_file == ""
        assert len(result.times) == 2
        assert result.times[0] == datetime(1973, 11, 1)  # 10/31_24:00 -> Nov 1
        assert result.values[0] == [0, 10, 1]
        assert result.values[1] == [0, 0, 1]

    def test_read_preserves_header(self, tmp_path: Path) -> None:
        """Verify header comment lines are preserved."""
        content = (
            "C Line 1\n"
            "C Line 2\n"
            "* Line 3\n"
            "          2                                     / NCOLADJ\n"
            "          1                                     / NSPADJ\n"
            "          0                                     / NFQADJ\n"
            "                                                / DSSFL\n"
        )
        filepath = self._write_file(tmp_path, content)
        result = read_supply_adjustment(filepath)

        assert len(result.header_lines) == 3
        assert result.header_lines[0] == "C Line 1"
        assert result.header_lines[2] == "* Line 3"

    def test_read_with_dss_file(self, tmp_path: Path) -> None:
        """Read file that specifies a DSS file (no inline data)."""
        content = (
            "C Supply adjustment via DSS\n"
            "          2                                     / NCOLADJ\n"
            "          1                                     / NSPADJ\n"
            "          0                                     / NFQADJ\n"
            "    supply_adj.dss                              / DSSFL\n"
        )
        filepath = self._write_file(tmp_path, content)
        result = read_supply_adjustment(filepath)

        assert result.n_columns == 2
        assert result.dss_file == "supply_adj.dss"
        assert len(result.times) == 0  # no inline data

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Verify FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            read_supply_adjustment(tmp_path / "nonexistent.dat")

    def test_read_no_data_rows(self, tmp_path: Path) -> None:
        """Read file with parameters but no data rows."""
        content = (
            "          5                                     / NCOLADJ\n"
            "          1                                     / NSPADJ\n"
            "          0                                     / NFQADJ\n"
            "                                                / DSSFL\n"
        )
        filepath = self._write_file(tmp_path, content)
        result = read_supply_adjustment(filepath)

        assert result.n_columns == 5
        assert len(result.times) == 0


# =============================================================================
# Writer tests
# =============================================================================


class TestWriteSupplyAdjustment:
    """Tests for write_supply_adjustment."""

    def test_write_basic(self, tmp_path: Path) -> None:
        """Write basic supply adjustment file."""
        data = SupplyAdjustment(
            n_columns=2,
            nsp=1,
            nfq=0,
            times=[datetime(1973, 11, 1), datetime(1973, 12, 1)],
            values=[[10, 0], [0, 1]],
        )
        filepath = tmp_path / "output.dat"
        result_path = write_supply_adjustment(data, filepath)

        assert result_path.exists()
        content = result_path.read_text()
        assert "NCOLADJ" in content
        assert "NSPADJ" in content
        assert "10/31/1973_24:00" in content
        assert "11/30/1973_24:00" in content

    def test_write_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Verify that parent directories are created."""
        data = SupplyAdjustment(n_columns=1, nsp=1, nfq=0)
        filepath = tmp_path / "sub" / "dir" / "output.dat"
        result_path = write_supply_adjustment(data, filepath)
        assert result_path.exists()

    def test_write_with_dss(self, tmp_path: Path) -> None:
        """Write file with DSS reference (no data rows)."""
        data = SupplyAdjustment(
            n_columns=3,
            nsp=1,
            nfq=0,
            dss_file="supply.dss",
        )
        filepath = tmp_path / "output.dat"
        write_supply_adjustment(data, filepath)

        content = filepath.read_text()
        assert "supply.dss" in content

    def test_write_zero_padded_codes(self, tmp_path: Path) -> None:
        """Verify adjustment codes are zero-padded to 2 digits."""
        data = SupplyAdjustment(
            n_columns=3,
            nsp=1,
            nfq=0,
            times=[datetime(1973, 11, 1)],
            values=[[0, 1, 10]],
        )
        filepath = tmp_path / "output.dat"
        write_supply_adjustment(data, filepath)

        content = filepath.read_text()
        # Should have 00, 01, 10 (zero-padded)
        assert "\t00" in content
        assert "\t01" in content
        assert "\t10" in content


# =============================================================================
# Roundtrip tests
# =============================================================================


class TestSupplyAdjustmentRoundtrip:
    """Tests for write -> read roundtrip."""

    def test_roundtrip_basic(self, tmp_path: Path) -> None:
        """Write and re-read, verify data matches."""
        original = SupplyAdjustment(
            n_columns=3,
            nsp=1,
            nfq=0,
            times=[
                datetime(1973, 11, 1),
                datetime(1973, 12, 1),
                datetime(1974, 1, 1),
            ],
            values=[
                [0, 10, 1],
                [0, 0, 1],
                [10, 10, 0],
            ],
        )

        filepath = tmp_path / "roundtrip.dat"
        write_supply_adjustment(original, filepath)
        result = read_supply_adjustment(filepath)

        assert result.n_columns == original.n_columns
        assert result.nsp == original.nsp
        assert result.nfq == original.nfq
        assert result.dss_file == ""
        assert len(result.times) == len(original.times)

        for i in range(len(original.times)):
            assert result.times[i] == original.times[i]
            assert result.values[i] == original.values[i]

    def test_roundtrip_single_column(self, tmp_path: Path) -> None:
        """Roundtrip with a single adjustment column."""
        original = SupplyAdjustment(
            n_columns=1,
            nsp=1,
            nfq=0,
            times=[datetime(2020, 11, 1)],
            values=[[10]],
        )

        filepath = tmp_path / "roundtrip_single.dat"
        write_supply_adjustment(original, filepath)
        result = read_supply_adjustment(filepath)

        assert result.n_columns == 1
        assert result.values[0] == [10]
