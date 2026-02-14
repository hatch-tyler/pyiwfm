"""Coverage tests for binary.py Fortran unformatted I/O.

Exercises FortranBinaryWriter / FortranBinaryReader roundtrips
for every scalar and array type, skip_records, at_eof,
get_position, seek_to_position, peek_record_size,
read_mixed_record, read_character_record, and truncated-file error.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from pyiwfm.io.binary import FortranBinaryReader, FortranBinaryWriter
from pyiwfm.core.exceptions import FileFormatError


# ---------------------------------------------------------------------------
# Scalar roundtrip tests
# ---------------------------------------------------------------------------


class TestScalarRoundtrips:
    """Test write-then-read roundtrips for every scalar type."""

    def test_int_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "int.bin"
        with FortranBinaryWriter(path) as w:
            w.write_int(42)
        with FortranBinaryReader(path) as r:
            assert r.read_int() == 42

    def test_float_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "float.bin"
        with FortranBinaryWriter(path) as w:
            w.write_float(3.14)
        with FortranBinaryReader(path) as r:
            assert r.read_float() == pytest.approx(3.14, rel=1e-5)

    def test_double_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "double.bin"
        with FortranBinaryWriter(path) as w:
            w.write_double(2.718281828459045)
        with FortranBinaryReader(path) as r:
            assert r.read_double() == pytest.approx(2.718281828459045)


# ---------------------------------------------------------------------------
# Array roundtrip tests
# ---------------------------------------------------------------------------


class TestArrayRoundtrips:
    """Test write-then-read roundtrips for every array type."""

    def test_int_array_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "int_arr.bin"
        arr = np.array([10, 20, 30, 40], dtype=np.int32)
        with FortranBinaryWriter(path) as w:
            w.write_int_array(arr)
        with FortranBinaryReader(path) as r:
            result = r.read_int_array()
        np.testing.assert_array_equal(result, arr)

    def test_float_array_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "float_arr.bin"
        arr = np.array([1.5, 2.5, 3.5], dtype=np.float32)
        with FortranBinaryWriter(path) as w:
            w.write_float_array(arr)
        with FortranBinaryReader(path) as r:
            result = r.read_float_array()
        np.testing.assert_allclose(result, arr, rtol=1e-5)

    def test_double_array_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "double_arr.bin"
        arr = np.array([100.1, 200.2, 300.3])
        with FortranBinaryWriter(path) as w:
            w.write_double_array(arr)
        with FortranBinaryReader(path) as r:
            result = r.read_double_array()
        np.testing.assert_allclose(result, arr)


# ---------------------------------------------------------------------------
# String roundtrip
# ---------------------------------------------------------------------------


class TestStringRoundtrip:
    """Test write-then-read roundtrips for strings."""

    def test_string_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "string.bin"
        with FortranBinaryWriter(path) as w:
            w.write_string("IWFM_MODEL")
        with FortranBinaryReader(path) as r:
            assert r.read_string() == "IWFM_MODEL"

    def test_string_with_fixed_length(self, tmp_path: Path) -> None:
        path = tmp_path / "string_fixed.bin"
        with FortranBinaryWriter(path) as w:
            w.write_string("AB", length=10)
        with FortranBinaryReader(path) as r:
            raw = r.read_record()
        assert len(raw) == 10
        assert raw[:2] == b"AB"

    def test_read_string_truncated_length(self, tmp_path: Path) -> None:
        """read_string(length=N) truncates to N characters."""
        path = tmp_path / "string_trunc.bin"
        with FortranBinaryWriter(path) as w:
            w.write_string("Hello World")
        with FortranBinaryReader(path) as r:
            assert r.read_string(length=5) == "Hello"


# ---------------------------------------------------------------------------
# skip_records / at_eof / position methods
# ---------------------------------------------------------------------------


class TestNavigationMethods:
    """Test skip_records, at_eof, get_position, seek_to_position."""

    def test_skip_records(self, tmp_path: Path) -> None:
        """skip_records skips the correct number of records."""
        path = tmp_path / "skip.bin"
        with FortranBinaryWriter(path) as w:
            w.write_int(1)
            w.write_int(2)
            w.write_int(3)
        with FortranBinaryReader(path) as r:
            r.skip_records(2)
            assert r.read_int() == 3

    def test_at_eof_true(self, tmp_path: Path) -> None:
        """at_eof returns True when positioned at end of file."""
        path = tmp_path / "eof.bin"
        with FortranBinaryWriter(path) as w:
            w.write_int(99)
        with FortranBinaryReader(path) as r:
            r.read_int()
            assert r.at_eof() is True

    def test_at_eof_false(self, tmp_path: Path) -> None:
        """at_eof returns False when data remains."""
        path = tmp_path / "noteof.bin"
        with FortranBinaryWriter(path) as w:
            w.write_int(1)
            w.write_int(2)
        with FortranBinaryReader(path) as r:
            r.read_int()
            assert r.at_eof() is False

    def test_get_and_seek_position(self, tmp_path: Path) -> None:
        """get_position / seek_to_position allow re-reading a record."""
        path = tmp_path / "pos.bin"
        with FortranBinaryWriter(path) as w:
            w.write_int(111)
            w.write_int(222)
        with FortranBinaryReader(path) as r:
            pos = r.get_position()
            first_read = r.read_int()
            r.seek_to_position(pos)
            second_read = r.read_int()
        assert first_read == second_read == 111

    def test_peek_record_size(self, tmp_path: Path) -> None:
        """peek_record_size returns the byte size without advancing."""
        path = tmp_path / "peek.bin"
        with FortranBinaryWriter(path) as w:
            w.write_double(1.0)  # 8-byte record
        with FortranBinaryReader(path) as r:
            size = r.peek_record_size()
            assert size == 8
            # File position should not have advanced
            assert r.read_double() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# read_mixed_record and read_character_record
# ---------------------------------------------------------------------------


class TestMixedAndCharacterRecords:
    """Test read_mixed_record and read_character_record."""

    def test_read_mixed_record(self, tmp_path: Path) -> None:
        """Read a record containing int + double array + string."""
        path = tmp_path / "mixed.bin"
        # Manually construct a mixed record
        with open(path, "wb") as f:
            # Build record bytes: 1 int32 + 2 float64s + 10-char string
            rec = b""
            rec += struct.pack("<i", 42)
            rec += struct.pack("<2d", 1.5, 2.5)
            rec += b"HELLO     "  # 10 chars padded
            marker = struct.pack("<i", len(rec))
            f.write(marker + rec + marker)

        with FortranBinaryReader(path) as r:
            result = r.read_mixed_record([
                ("i4", 1),
                ("f8", 2),
                ("S10", 1),
            ])
        assert result[0] == 42
        np.testing.assert_allclose(result[1], [1.5, 2.5])
        assert result[2] == "HELLO"

    def test_read_character_record(self, tmp_path: Path) -> None:
        """read_character_record reads exactly N chars."""
        path = tmp_path / "char.bin"
        with FortranBinaryWriter(path) as w:
            w.write_string("ABCDEFGHIJ", length=20)
        with FortranBinaryReader(path) as r:
            result = r.read_character_record(5)
        assert result == "ABCDE"


# ---------------------------------------------------------------------------
# Error conditions
# ---------------------------------------------------------------------------


class TestErrorConditions:
    """Tests for error paths."""

    def test_read_truncated_file_raises(self, tmp_path: Path) -> None:
        """Reading from a truncated file raises FileFormatError."""
        path = tmp_path / "truncated.bin"
        with open(path, "wb") as f:
            # Write marker claiming 100 bytes, but only provide 4
            f.write(struct.pack("<i", 100))
            f.write(struct.pack("<i", 0))
        with FortranBinaryReader(path) as r:
            with pytest.raises(FileFormatError, match="Incomplete record"):
                r.read_record()

    def test_skip_records_past_eof_raises(self, tmp_path: Path) -> None:
        """Skipping past end of file raises EOFError."""
        path = tmp_path / "short.bin"
        with FortranBinaryWriter(path) as w:
            w.write_int(1)
        with FortranBinaryReader(path) as r:
            with pytest.raises(EOFError):
                r.skip_records(5)

    def test_reader_not_opened(self, tmp_path: Path) -> None:
        """Operations on an unopened reader raise RuntimeError."""
        path = tmp_path / "dummy.bin"
        path.write_bytes(b"")
        reader = FortranBinaryReader(path)
        with pytest.raises(RuntimeError, match="File not open"):
            reader.read_record()
        with pytest.raises(RuntimeError, match="File not open"):
            reader.skip_records(1)
        with pytest.raises(RuntimeError, match="File not open"):
            reader.at_eof()
        with pytest.raises(RuntimeError, match="File not open"):
            reader.get_position()
        with pytest.raises(RuntimeError, match="File not open"):
            reader.seek_to_position(0)
        with pytest.raises(RuntimeError, match="File not open"):
            reader.peek_record_size()

    def test_writer_not_opened(self, tmp_path: Path) -> None:
        """Writing to an unopened writer raises RuntimeError."""
        path = tmp_path / "dummy.bin"
        writer = FortranBinaryWriter(path)
        with pytest.raises(RuntimeError, match="File not open"):
            writer.write_record(b"data")
