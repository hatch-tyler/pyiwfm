"""Unit tests for binary I/O handlers."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from pyiwfm.core.exceptions import FileFormatError
from pyiwfm.io.binary import (
    FortranBinaryReader,
    FortranBinaryWriter,
    StreamAccessBinaryReader,
)


class TestFortranBinaryReader:
    """Tests for FortranBinaryReader."""

    def test_read_int(self, tmp_path: Path) -> None:
        """Test reading a single integer."""
        filepath = tmp_path / "test.bin"

        # Write a Fortran-style integer record
        with open(filepath, "wb") as f:
            value = 42
            data = struct.pack("<i", value)
            marker = struct.pack("<i", len(data))
            f.write(marker + data + marker)

        with FortranBinaryReader(filepath) as reader:
            result = reader.read_int()

        assert result == 42

    def test_read_int_array(self, tmp_path: Path) -> None:
        """Test reading an integer array."""
        filepath = tmp_path / "test.bin"

        values = [1, 2, 3, 4, 5]
        with open(filepath, "wb") as f:
            data = struct.pack("<5i", *values)
            marker = struct.pack("<i", len(data))
            f.write(marker + data + marker)

        with FortranBinaryReader(filepath) as reader:
            result = reader.read_int_array()

        np.testing.assert_array_equal(result, values)

    def test_read_double_array(self, tmp_path: Path) -> None:
        """Test reading a double array."""
        filepath = tmp_path / "test.bin"

        values = [1.1, 2.2, 3.3]
        with open(filepath, "wb") as f:
            data = struct.pack("<3d", *values)
            marker = struct.pack("<i", len(data))
            f.write(marker + data + marker)

        with FortranBinaryReader(filepath) as reader:
            result = reader.read_double_array()

        np.testing.assert_allclose(result, values)

    def test_read_multiple_records(self, tmp_path: Path) -> None:
        """Test reading multiple records."""
        filepath = tmp_path / "test.bin"

        with open(filepath, "wb") as f:
            # Record 1: integer
            data1 = struct.pack("<i", 100)
            marker1 = struct.pack("<i", len(data1))
            f.write(marker1 + data1 + marker1)

            # Record 2: double array
            data2 = struct.pack("<3d", 1.0, 2.0, 3.0)
            marker2 = struct.pack("<i", len(data2))
            f.write(marker2 + data2 + marker2)

        with FortranBinaryReader(filepath) as reader:
            n = reader.read_int()
            arr = reader.read_double_array()

        assert n == 100
        np.testing.assert_allclose(arr, [1.0, 2.0, 3.0])

    def test_read_mismatched_markers(self, tmp_path: Path) -> None:
        """Test error on mismatched record markers."""
        filepath = tmp_path / "test.bin"

        with open(filepath, "wb") as f:
            data = struct.pack("<i", 42)
            leading_marker = struct.pack("<i", len(data))
            trailing_marker = struct.pack("<i", len(data) + 1)  # Wrong!
            f.write(leading_marker + data + trailing_marker)

        with FortranBinaryReader(filepath) as reader:
            with pytest.raises(FileFormatError, match="marker mismatch"):
                reader.read_int()


class TestFortranBinaryWriter:
    """Tests for FortranBinaryWriter."""

    def test_write_int(self, tmp_path: Path) -> None:
        """Test writing a single integer."""
        filepath = tmp_path / "test.bin"

        with FortranBinaryWriter(filepath) as writer:
            writer.write_int(42)

        # Verify by reading back
        with FortranBinaryReader(filepath) as reader:
            result = reader.read_int()

        assert result == 42

    def test_write_double_array(self, tmp_path: Path) -> None:
        """Test writing a double array."""
        filepath = tmp_path / "test.bin"

        values = np.array([1.1, 2.2, 3.3])
        with FortranBinaryWriter(filepath) as writer:
            writer.write_double_array(values)

        with FortranBinaryReader(filepath) as reader:
            result = reader.read_double_array()

        np.testing.assert_allclose(result, values)

    def test_roundtrip_multiple_records(self, tmp_path: Path) -> None:
        """Test writing and reading multiple records."""
        filepath = tmp_path / "test.bin"

        with FortranBinaryWriter(filepath) as writer:
            writer.write_int(10)
            writer.write_int(20)
            writer.write_double_array(np.array([1.0, 2.0, 3.0]))
            writer.write_int_array(np.array([100, 200, 300], dtype=np.int32))

        with FortranBinaryReader(filepath) as reader:
            n1 = reader.read_int()
            n2 = reader.read_int()
            arr_d = reader.read_double_array()
            arr_i = reader.read_int_array()

        assert n1 == 10
        assert n2 == 20
        np.testing.assert_allclose(arr_d, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(arr_i, [100, 200, 300])


class TestStreamAccessBinaryReader:
    """Tests for StreamAccessBinaryReader (IWFM ACCESS='STREAM' format)."""

    def test_read_int(self, tmp_path: Path) -> None:
        """Test reading a single 4-byte int."""
        filepath = tmp_path / "test.bin"
        with open(filepath, "wb") as f:
            f.write(struct.pack("<i", 42))
        with StreamAccessBinaryReader(filepath) as reader:
            assert reader.read_int() == 42

    def test_read_double(self, tmp_path: Path) -> None:
        """Test reading a single 8-byte double."""
        filepath = tmp_path / "test.bin"
        with open(filepath, "wb") as f:
            f.write(struct.pack("<d", 3.14159))
        with StreamAccessBinaryReader(filepath) as reader:
            assert reader.read_double() == pytest.approx(3.14159)

    def test_read_ints(self, tmp_path: Path) -> None:
        """Test reading an array of ints."""
        filepath = tmp_path / "test.bin"
        values = [10, 20, 30, 40, 50]
        with open(filepath, "wb") as f:
            f.write(struct.pack("<5i", *values))
        with StreamAccessBinaryReader(filepath) as reader:
            result = reader.read_ints(5)
        np.testing.assert_array_equal(result, values)

    def test_read_doubles(self, tmp_path: Path) -> None:
        """Test reading an array of doubles."""
        filepath = tmp_path / "test.bin"
        values = [1.1, 2.2, 3.3]
        with open(filepath, "wb") as f:
            f.write(struct.pack("<3d", *values))
        with StreamAccessBinaryReader(filepath) as reader:
            result = reader.read_doubles(3)
        np.testing.assert_allclose(result, values)

    def test_read_logical_true(self, tmp_path: Path) -> None:
        """Test reading a Fortran LOGICAL (non-zero = True)."""
        filepath = tmp_path / "test.bin"
        with open(filepath, "wb") as f:
            f.write(struct.pack("<i", 1))
        with StreamAccessBinaryReader(filepath) as reader:
            assert reader.read_logical() is True

    def test_read_logical_false(self, tmp_path: Path) -> None:
        """Test reading a Fortran LOGICAL (zero = False)."""
        filepath = tmp_path / "test.bin"
        with open(filepath, "wb") as f:
            f.write(struct.pack("<i", 0))
        with StreamAccessBinaryReader(filepath) as reader:
            assert reader.read_logical() is False

    def test_read_logicals(self, tmp_path: Path) -> None:
        """Test reading an array of Fortran LOGICALs."""
        filepath = tmp_path / "test.bin"
        with open(filepath, "wb") as f:
            f.write(struct.pack("<4i", 1, 0, 1, 0))
        with StreamAccessBinaryReader(filepath) as reader:
            result = reader.read_logicals(4)
        np.testing.assert_array_equal(result, [True, False, True, False])

    def test_read_string(self, tmp_path: Path) -> None:
        """Test reading a fixed-length string."""
        filepath = tmp_path / "test.bin"
        text = "Hello"
        padded = text.ljust(20)
        with open(filepath, "wb") as f:
            f.write(padded.encode("ascii"))
        with StreamAccessBinaryReader(filepath) as reader:
            result = reader.read_string(20)
        assert result == "Hello"

    def test_at_eof(self, tmp_path: Path) -> None:
        """Test EOF detection."""
        filepath = tmp_path / "test.bin"
        with open(filepath, "wb") as f:
            f.write(struct.pack("<i", 99))
        with StreamAccessBinaryReader(filepath) as reader:
            assert not reader.at_eof()
            reader.read_int()
            assert reader.at_eof()

    def test_get_position(self, tmp_path: Path) -> None:
        """Test position tracking."""
        filepath = tmp_path / "test.bin"
        with open(filepath, "wb") as f:
            f.write(struct.pack("<2i", 1, 2))
        with StreamAccessBinaryReader(filepath) as reader:
            assert reader.get_position() == 0
            reader.read_int()
            assert reader.get_position() == 4
            reader.read_int()
            assert reader.get_position() == 8

    def test_read_multiple_types(self, tmp_path: Path) -> None:
        """Test reading mixed types in sequence (no record markers)."""
        filepath = tmp_path / "test.bin"
        with open(filepath, "wb") as f:
            f.write(struct.pack("<i", 5))  # int
            f.write(struct.pack("<d", 2.718))  # double
            f.write(struct.pack("<3i", 1, 2, 3))  # 3 ints
        with StreamAccessBinaryReader(filepath) as reader:
            assert reader.read_int() == 5
            assert reader.read_double() == pytest.approx(2.718)
            np.testing.assert_array_equal(reader.read_ints(3), [1, 2, 3])

    def test_big_endian(self, tmp_path: Path) -> None:
        """Test big-endian reading."""
        filepath = tmp_path / "test.bin"
        with open(filepath, "wb") as f:
            f.write(struct.pack(">i", 42))
            f.write(struct.pack(">d", 1.5))
        with StreamAccessBinaryReader(filepath, endian=">") as reader:
            assert reader.read_int() == 42
            assert reader.read_double() == pytest.approx(1.5)

    def test_eof_raises(self, tmp_path: Path) -> None:
        """Test that reading past EOF raises an error."""
        filepath = tmp_path / "test.bin"
        filepath.write_bytes(b"")
        with StreamAccessBinaryReader(filepath) as reader:
            with pytest.raises(EOFError):
                reader.read_int()


class TestBinaryEndianness:
    """Tests for endianness handling."""

    def test_big_endian_roundtrip(self, tmp_path: Path) -> None:
        """Test big-endian write/read."""
        filepath = tmp_path / "big_endian.bin"

        with FortranBinaryWriter(filepath, endian=">") as writer:
            writer.write_int(12345)
            writer.write_double_array(np.array([1.5, 2.5, 3.5]))

        with FortranBinaryReader(filepath, endian=">") as reader:
            n = reader.read_int()
            arr = reader.read_double_array()

        assert n == 12345
        np.testing.assert_allclose(arr, [1.5, 2.5, 3.5])


# =============================================================================
# Additional coverage tests
# =============================================================================


class TestFortranBinaryReaderEdgeCases:
    """Additional edge-case tests for FortranBinaryReader."""

    def test_read_record_file_not_open(self, tmp_path: Path) -> None:
        """Reading from a reader that is not opened raises RuntimeError."""
        filepath = tmp_path / "test.bin"
        filepath.write_bytes(b"")
        reader = FortranBinaryReader(filepath)
        # Do NOT use 'with', so _file stays None
        with pytest.raises(RuntimeError, match="File not open"):
            reader.read_record()

    def test_read_record_eof(self, tmp_path: Path) -> None:
        """Reading from an empty file raises EOFError."""
        filepath = tmp_path / "empty.bin"
        filepath.write_bytes(b"")
        with FortranBinaryReader(filepath) as reader:
            with pytest.raises(EOFError):
                reader.read_record()

    def test_read_record_incomplete_data(self, tmp_path: Path) -> None:
        """Incomplete record (marker says 100 bytes, only 4 available) raises FileFormatError."""
        filepath = tmp_path / "incomplete.bin"
        with open(filepath, "wb") as f:
            # marker says 100 bytes
            f.write(struct.pack("<i", 100))
            # only 4 bytes of data
            f.write(struct.pack("<i", 42))
        with FortranBinaryReader(filepath) as reader:
            with pytest.raises(FileFormatError, match="Incomplete record"):
                reader.read_record()

    def test_read_float(self, tmp_path: Path) -> None:
        """Test reading a single float record."""
        filepath = tmp_path / "float.bin"
        with FortranBinaryWriter(filepath) as writer:
            writer.write_float(3.14)
        with FortranBinaryReader(filepath) as reader:
            result = reader.read_float()
        assert result == pytest.approx(3.14, rel=1e-5)

    def test_read_float_array(self, tmp_path: Path) -> None:
        """Test reading a float array record."""
        filepath = tmp_path / "float_arr.bin"
        values = np.array([1.0, 2.5, 3.5], dtype=np.float32)
        with FortranBinaryWriter(filepath) as writer:
            writer.write_float_array(values)
        with FortranBinaryReader(filepath) as reader:
            result = reader.read_float_array()
        np.testing.assert_allclose(result, values, rtol=1e-5)

    def test_read_double(self, tmp_path: Path) -> None:
        """Test reading a single double record."""
        filepath = tmp_path / "double.bin"
        with FortranBinaryWriter(filepath) as writer:
            writer.write_double(2.718281828)
        with FortranBinaryReader(filepath) as reader:
            result = reader.read_double()
        assert result == pytest.approx(2.718281828)

    def test_read_string(self, tmp_path: Path) -> None:
        """Test reading a string record."""
        filepath = tmp_path / "string.bin"
        with FortranBinaryWriter(filepath) as writer:
            writer.write_string("Hello")
        with FortranBinaryReader(filepath) as reader:
            result = reader.read_string()
        assert result == "Hello"

    def test_read_string_with_length(self, tmp_path: Path) -> None:
        """Test reading a string record truncated to specified length."""
        filepath = tmp_path / "string_len.bin"
        with FortranBinaryWriter(filepath) as writer:
            writer.write_string("Hello World", length=20)
        with FortranBinaryReader(filepath) as reader:
            result = reader.read_string(length=5)
        assert result == "Hello"

    def test_read_eof_marker(self, tmp_path: Path) -> None:
        """Reading past the last record raises EOFError."""
        filepath = tmp_path / "single_record.bin"
        with FortranBinaryWriter(filepath) as writer:
            writer.write_int(42)
        with FortranBinaryReader(filepath) as reader:
            reader.read_int()
            with pytest.raises(EOFError):
                reader.read_int()


class TestFortranBinaryWriterEdgeCases:
    """Additional edge-case tests for FortranBinaryWriter."""

    def test_write_record_file_not_open(self, tmp_path: Path) -> None:
        """Writing to a writer that is not opened raises RuntimeError."""
        filepath = tmp_path / "test.bin"
        writer = FortranBinaryWriter(filepath)
        with pytest.raises(RuntimeError, match="File not open"):
            writer.write_record(b"data")

    def test_write_float(self, tmp_path: Path) -> None:
        """Test writing a single float record."""
        filepath = tmp_path / "wfloat.bin"
        with FortranBinaryWriter(filepath) as writer:
            writer.write_float(1.5)
        with FortranBinaryReader(filepath) as reader:
            result = reader.read_float()
        assert result == pytest.approx(1.5, rel=1e-5)

    def test_write_float_array(self, tmp_path: Path) -> None:
        """Test writing a float array record."""
        filepath = tmp_path / "wfloat_arr.bin"
        values = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        with FortranBinaryWriter(filepath) as writer:
            writer.write_float_array(values)
        with FortranBinaryReader(filepath) as reader:
            result = reader.read_float_array()
        np.testing.assert_allclose(result, values, rtol=1e-5)

    def test_write_double(self, tmp_path: Path) -> None:
        """Test writing a single double record."""
        filepath = tmp_path / "wdouble.bin"
        with FortranBinaryWriter(filepath) as writer:
            writer.write_double(9.81)
        with FortranBinaryReader(filepath) as reader:
            result = reader.read_double()
        assert result == pytest.approx(9.81)

    def test_write_string(self, tmp_path: Path) -> None:
        """Test writing a string record without length padding."""
        filepath = tmp_path / "wstring.bin"
        with FortranBinaryWriter(filepath) as writer:
            writer.write_string("IWFM")
        with FortranBinaryReader(filepath) as reader:
            result = reader.read_string()
        assert result == "IWFM"

    def test_write_string_with_length(self, tmp_path: Path) -> None:
        """Test writing a string record padded/truncated to a length."""
        filepath = tmp_path / "wstring_len.bin"
        with FortranBinaryWriter(filepath) as writer:
            writer.write_string("Hi", length=10)
        with FortranBinaryReader(filepath) as reader:
            result = reader.read_string()
        assert result == "Hi"
        # The string should have been padded to 10 chars
        with FortranBinaryReader(filepath) as reader:
            raw = reader.read_record()
        assert len(raw) == 10

    def test_write_string_truncated(self, tmp_path: Path) -> None:
        """Test that writing a long string truncates to specified length."""
        filepath = tmp_path / "wstring_trunc.bin"
        with FortranBinaryWriter(filepath) as writer:
            writer.write_string("LongStringHere", length=4)
        with FortranBinaryReader(filepath) as reader:
            result = reader.read_string()
        assert result == "Long"

    def test_write_creates_parent_directory(self, tmp_path: Path) -> None:
        """Writer creates parent directories if they don't exist."""
        filepath = tmp_path / "subdir" / "nested" / "test.bin"
        with FortranBinaryWriter(filepath) as writer:
            writer.write_int(1)
        assert filepath.exists()
