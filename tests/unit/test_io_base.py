"""Unit tests for base I/O classes.

Tests:
- FileInfo dataclass
- BaseReader
- BaseWriter
- BinaryReader
- BinaryWriter
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import pytest

from pyiwfm.io.base import (
    FileInfo,
    BaseReader,
    BaseWriter,
    BinaryReader,
    BinaryWriter,
)


# =============================================================================
# Concrete implementations for testing abstract classes
# =============================================================================


class ConcreteReader(BaseReader):
    """Concrete reader implementation for testing."""

    @property
    def format(self) -> str:
        return "test"

    def read(self) -> Any:
        return self.filepath.read_text()


class ConcreteWriter(BaseWriter):
    """Concrete writer implementation for testing."""

    @property
    def format(self) -> str:
        return "test"

    def write(self, data: Any) -> None:
        self._ensure_parent_exists()
        self.filepath.write_text(str(data))


class ConcreteBinaryReader(BinaryReader):
    """Concrete binary reader implementation for testing."""

    def read(self) -> bytes:
        with open(self.filepath, "rb") as f:
            return self._read_fortran_record(f)


class ConcreteBinaryWriter(BinaryWriter):
    """Concrete binary writer implementation for testing."""

    def write(self, data: bytes) -> None:
        self._ensure_parent_exists()
        with open(self.filepath, "wb") as f:
            self._write_fortran_record(f, data)


# =============================================================================
# Test FileInfo
# =============================================================================


class TestFileInfo:
    """Tests for FileInfo dataclass."""

    def test_basic_creation(self, tmp_path: Path) -> None:
        """Test basic FileInfo creation."""
        info = FileInfo(
            path=tmp_path / "test.dat",
            format="ascii",
        )

        assert info.path == tmp_path / "test.dat"
        assert info.format == "ascii"
        assert info.version is None
        assert info.description == ""
        assert info.metadata == {}

    def test_full_creation(self, tmp_path: Path) -> None:
        """Test FileInfo with all fields."""
        info = FileInfo(
            path=tmp_path / "test.hdf5",
            format="hdf5",
            version="2025.0",
            description="Test file",
            metadata={"author": "Test", "created": "2024-01-01"},
        )

        assert info.path == tmp_path / "test.hdf5"
        assert info.format == "hdf5"
        assert info.version == "2025.0"
        assert info.description == "Test file"
        assert info.metadata["author"] == "Test"

    def test_metadata_default_factory(self) -> None:
        """Test that metadata uses independent default dict."""
        info1 = FileInfo(path=Path("a.dat"), format="ascii")
        info2 = FileInfo(path=Path("b.dat"), format="ascii")

        info1.metadata["key"] = "value"

        assert "key" not in info2.metadata


# =============================================================================
# Test BaseReader
# =============================================================================


class TestBaseReader:
    """Tests for BaseReader class."""

    def test_init_with_existing_file(self, tmp_path: Path) -> None:
        """Test initialization with existing file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        reader = ConcreteReader(test_file)

        assert reader.filepath == test_file

    def test_init_with_string_path(self, tmp_path: Path) -> None:
        """Test initialization with string path."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        reader = ConcreteReader(str(test_file))

        assert reader.filepath == test_file

    def test_init_file_not_found(self, tmp_path: Path) -> None:
        """Test initialization with non-existent file."""
        nonexistent = tmp_path / "nonexistent.txt"

        with pytest.raises(FileNotFoundError, match="File not found"):
            ConcreteReader(nonexistent)

    def test_init_path_is_directory(self, tmp_path: Path) -> None:
        """Test initialization with directory path."""
        with pytest.raises(ValueError, match="not a file"):
            ConcreteReader(tmp_path)

    def test_format_property(self, tmp_path: Path) -> None:
        """Test format property."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        reader = ConcreteReader(test_file)

        assert reader.format == "test"

    def test_read_method(self, tmp_path: Path) -> None:
        """Test read method."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        reader = ConcreteReader(test_file)
        content = reader.read()

        assert content == "hello world"


# =============================================================================
# Test BaseWriter
# =============================================================================


class TestBaseWriter:
    """Tests for BaseWriter class."""

    def test_init(self, tmp_path: Path) -> None:
        """Test initialization."""
        output_file = tmp_path / "output.txt"

        writer = ConcreteWriter(output_file)

        assert writer.filepath == output_file

    def test_init_with_string_path(self, tmp_path: Path) -> None:
        """Test initialization with string path."""
        output_file = tmp_path / "output.txt"

        writer = ConcreteWriter(str(output_file))

        assert writer.filepath == output_file

    def test_ensure_parent_exists(self, tmp_path: Path) -> None:
        """Test parent directory creation."""
        deep_path = tmp_path / "a" / "b" / "c" / "output.txt"

        writer = ConcreteWriter(deep_path)
        writer.write("test content")

        assert deep_path.exists()
        assert deep_path.read_text() == "test content"

    def test_format_property(self, tmp_path: Path) -> None:
        """Test format property."""
        writer = ConcreteWriter(tmp_path / "output.txt")

        assert writer.format == "test"

    def test_write_method(self, tmp_path: Path) -> None:
        """Test write method."""
        output_file = tmp_path / "output.txt"

        writer = ConcreteWriter(output_file)
        writer.write("hello world")

        assert output_file.read_text() == "hello world"


# =============================================================================
# Test BinaryReader
# =============================================================================


class TestBinaryReader:
    """Tests for BinaryReader class."""

    def test_format_property(self, tmp_path: Path) -> None:
        """Test format property returns 'binary'."""
        test_file = tmp_path / "test.bin"
        # Write a simple fortran record
        with open(test_file, "wb") as f:
            data = b"test"
            marker = struct.pack("<i", len(data))
            f.write(marker + data + marker)

        reader = ConcreteBinaryReader(test_file)

        assert reader.format == "binary"

    def test_init_endian_default(self, tmp_path: Path) -> None:
        """Test default endianness."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"\x00")

        reader = ConcreteBinaryReader(test_file)

        assert reader.endian == "<"  # Little-endian default

    def test_init_endian_big(self, tmp_path: Path) -> None:
        """Test big-endian setting."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"\x00")

        reader = ConcreteBinaryReader(test_file, endian=">")

        assert reader.endian == ">"

    def test_read_fortran_record(self, tmp_path: Path) -> None:
        """Test reading a Fortran unformatted record."""
        test_file = tmp_path / "test.bin"
        data = b"Hello Fortran!"

        # Write fortran-style record (4-byte marker, data, 4-byte marker)
        with open(test_file, "wb") as f:
            marker = struct.pack("<i", len(data))
            f.write(marker + data + marker)

        reader = ConcreteBinaryReader(test_file)
        result = reader.read()

        assert result == data

    def test_read_fortran_record_big_endian(self, tmp_path: Path) -> None:
        """Test reading big-endian Fortran record."""
        test_file = tmp_path / "test.bin"
        data = b"Big endian test"

        # Write big-endian fortran-style record
        with open(test_file, "wb") as f:
            marker = struct.pack(">i", len(data))
            f.write(marker + data + marker)

        reader = ConcreteBinaryReader(test_file, endian=">")
        result = reader.read()

        assert result == data

    def test_read_fortran_record_eof_marker(self, tmp_path: Path) -> None:
        """Test EOF while reading marker."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"\x00\x00")  # Only 2 bytes, not 4

        reader = ConcreteBinaryReader(test_file)

        with pytest.raises(EOFError, match="record marker"):
            reader.read()

    def test_read_fortran_record_eof_data(self, tmp_path: Path) -> None:
        """Test EOF while reading data."""
        test_file = tmp_path / "test.bin"
        # Marker says 100 bytes but only 5 bytes of data
        marker = struct.pack("<i", 100)
        with open(test_file, "wb") as f:
            f.write(marker + b"short")

        reader = ConcreteBinaryReader(test_file)

        with pytest.raises(EOFError, match="record data"):
            reader.read()

    def test_read_fortran_record_marker_mismatch(self, tmp_path: Path) -> None:
        """Test record marker mismatch error."""
        test_file = tmp_path / "test.bin"
        data = b"test data"

        # Write mismatched markers
        with open(test_file, "wb") as f:
            f.write(struct.pack("<i", len(data)))
            f.write(data)
            f.write(struct.pack("<i", len(data) + 1))  # Wrong trailing marker

        reader = ConcreteBinaryReader(test_file)

        with pytest.raises(ValueError, match="marker mismatch"):
            reader.read()


# =============================================================================
# Test BinaryWriter
# =============================================================================


class TestBinaryWriter:
    """Tests for BinaryWriter class."""

    def test_format_property(self, tmp_path: Path) -> None:
        """Test format property returns 'binary'."""
        writer = ConcreteBinaryWriter(tmp_path / "test.bin")

        assert writer.format == "binary"

    def test_init_endian_default(self, tmp_path: Path) -> None:
        """Test default endianness."""
        writer = ConcreteBinaryWriter(tmp_path / "test.bin")

        assert writer.endian == "<"

    def test_init_endian_big(self, tmp_path: Path) -> None:
        """Test big-endian setting."""
        writer = ConcreteBinaryWriter(tmp_path / "test.bin", endian=">")

        assert writer.endian == ">"

    def test_write_fortran_record(self, tmp_path: Path) -> None:
        """Test writing a Fortran unformatted record."""
        test_file = tmp_path / "test.bin"
        data = b"Hello Fortran!"

        writer = ConcreteBinaryWriter(test_file)
        writer.write(data)

        # Read and verify
        with open(test_file, "rb") as f:
            content = f.read()

        expected_marker = struct.pack("<i", len(data))
        expected = expected_marker + data + expected_marker

        assert content == expected

    def test_write_fortran_record_big_endian(self, tmp_path: Path) -> None:
        """Test writing big-endian Fortran record."""
        test_file = tmp_path / "test.bin"
        data = b"Big endian test"

        writer = ConcreteBinaryWriter(test_file, endian=">")
        writer.write(data)

        # Read and verify
        with open(test_file, "rb") as f:
            content = f.read()

        expected_marker = struct.pack(">i", len(data))
        expected = expected_marker + data + expected_marker

        assert content == expected


# =============================================================================
# Test Roundtrip
# =============================================================================


class TestBinaryRoundtrip:
    """Tests for binary read-write roundtrip."""

    def test_roundtrip_simple(self, tmp_path: Path) -> None:
        """Test writing and reading back simple data."""
        test_file = tmp_path / "test.bin"
        original_data = b"Simple test data for roundtrip"

        # Write
        writer = ConcreteBinaryWriter(test_file)
        writer.write(original_data)

        # Read back
        reader = ConcreteBinaryReader(test_file)
        read_data = reader.read()

        assert read_data == original_data

    def test_roundtrip_big_endian(self, tmp_path: Path) -> None:
        """Test roundtrip with big-endian format."""
        test_file = tmp_path / "test.bin"
        original_data = b"Big endian roundtrip test"

        # Write
        writer = ConcreteBinaryWriter(test_file, endian=">")
        writer.write(original_data)

        # Read back
        reader = ConcreteBinaryReader(test_file, endian=">")
        read_data = reader.read()

        assert read_data == original_data

    def test_roundtrip_binary_data(self, tmp_path: Path) -> None:
        """Test roundtrip with binary (non-text) data."""
        test_file = tmp_path / "test.bin"
        # Create binary data with various byte values
        original_data = bytes(range(256))

        # Write
        writer = ConcreteBinaryWriter(test_file)
        writer.write(original_data)

        # Read back
        reader = ConcreteBinaryReader(test_file)
        read_data = reader.read()

        assert read_data == original_data
