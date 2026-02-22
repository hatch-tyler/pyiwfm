"""
Binary file I/O handlers for IWFM model files.

This module provides classes for reading and writing IWFM binary files:
- ``FortranBinaryReader`` / ``FortranBinaryWriter``: Fortran unformatted
  sequential-access files (record markers around each write).
- ``StreamAccessBinaryReader``: Raw byte-stream files matching IWFM's
  ``ACCESS='STREAM'`` (no record markers; caller supplies counts).
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import BinaryIO

import numpy as np
from numpy.typing import NDArray

from pyiwfm.core.exceptions import FileFormatError
from pyiwfm.core.mesh import AppGrid
from pyiwfm.core.stratigraphy import Stratigraphy


class FortranBinaryReader:
    """
    Reader for Fortran unformatted binary files.

    Handles the record markers that Fortran writes before and after each record.
    """

    def __init__(self, filepath: Path | str, endian: str = "<") -> None:
        """
        Initialize the reader.

        Args:
            filepath: Path to the binary file
            endian: Byte order ('<' = little-endian, '>' = big-endian)
        """
        self.filepath = Path(filepath)
        self.endian = endian
        self._file: BinaryIO | None = None

    def __enter__(self) -> FortranBinaryReader:
        self._file = open(self.filepath, "rb")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        if self._file:
            self._file.close()

    def read_record(self) -> bytes:
        """
        Read a single Fortran record.

        Returns:
            Record data as bytes
        """
        if self._file is None:
            raise RuntimeError("File not open")

        # Read leading record marker (4 bytes)
        marker_data = self._file.read(4)
        if len(marker_data) < 4:
            raise EOFError("End of file reached")

        record_length = struct.unpack(f"{self.endian}i", marker_data)[0]

        # Read record data
        data = self._file.read(record_length)
        if len(data) < record_length:
            raise FileFormatError(
                f"Incomplete record: expected {record_length} bytes, got {len(data)}"
            )

        # Read trailing record marker
        trailing_marker = self._file.read(4)
        trailing_length = struct.unpack(f"{self.endian}i", trailing_marker)[0]

        if trailing_length != record_length:
            raise FileFormatError(f"Record marker mismatch: {record_length} != {trailing_length}")

        return data

    def read_int(self) -> int:
        """Read a single integer record."""
        data = self.read_record()
        result: int = struct.unpack(f"{self.endian}i", data)[0]
        return result

    def read_int_array(self) -> NDArray[np.int32]:
        """Read an integer array record."""
        data = self.read_record()
        n = len(data) // 4
        return np.frombuffer(data, dtype=f"{self.endian}i4", count=n)

    def read_float(self) -> float:
        """Read a single float record (real*4)."""
        data = self.read_record()
        result: float = struct.unpack(f"{self.endian}f", data)[0]
        return result

    def read_float_array(self) -> NDArray[np.float32]:
        """Read a float array record (real*4)."""
        data = self.read_record()
        n = len(data) // 4
        return np.frombuffer(data, dtype=f"{self.endian}f4", count=n)

    def read_double(self) -> float:
        """Read a single double record (real*8)."""
        data = self.read_record()
        result: float = struct.unpack(f"{self.endian}d", data)[0]
        return result

    def read_double_array(self) -> NDArray[np.float64]:
        """Read a double array record (real*8)."""
        data = self.read_record()
        n = len(data) // 8
        return np.frombuffer(data, dtype=f"{self.endian}f8", count=n)

    def read_string(self, length: int | None = None) -> str:
        """Read a string record."""
        data = self.read_record()
        if length:
            data = data[:length]
        return data.decode("ascii").strip()

    def skip_records(self, n: int) -> None:
        """
        Skip n records without reading their data.

        Args:
            n: Number of records to skip
        """
        if self._file is None:
            raise RuntimeError("File not open")

        for _ in range(n):
            # Read leading record marker
            marker_data = self._file.read(4)
            if len(marker_data) < 4:
                raise EOFError("End of file reached while skipping records")

            record_length = struct.unpack(f"{self.endian}i", marker_data)[0]

            # Skip record data and trailing marker
            self._file.seek(record_length + 4, 1)  # 1 = SEEK_CUR

    def get_position(self) -> int:
        """
        Get current file position.

        Returns:
            Current byte offset in the file
        """
        if self._file is None:
            raise RuntimeError("File not open")
        return self._file.tell()

    def seek_to_position(self, pos: int) -> None:
        """
        Seek to a specific position in the file.

        Args:
            pos: Byte offset to seek to
        """
        if self._file is None:
            raise RuntimeError("File not open")
        self._file.seek(pos)

    def read_mixed_record(self, dtype_spec: list[tuple[str, int]]) -> tuple:
        """
        Read a record containing mixed data types.

        Args:
            dtype_spec: List of (dtype, count) tuples specifying the record format.
                        dtype can be: 'i4' (int32), 'i8' (int64), 'f4' (float32),
                        'f8' (float64), 'S' followed by length (e.g., 'S30' for 30-char string)

        Returns:
            Tuple of values read from the record

        Example:
            >>> reader.read_mixed_record([('i4', 1), ('f8', 3), ('S30', 1)])
            (42, array([1.0, 2.0, 3.0]), 'SOME_STRING')
        """
        data = self.read_record()
        offset = 0
        results = []

        for dtype, count in dtype_spec:
            if dtype.startswith("S"):
                # String type
                str_len = int(dtype[1:])
                total_len = str_len * count
                str_data = data[offset : offset + total_len]
                if count == 1:
                    results.append(str_data.decode("ascii", errors="replace").strip())
                else:
                    strings = []
                    for i in range(count):
                        s = (
                            str_data[i * str_len : (i + 1) * str_len]
                            .decode("ascii", errors="replace")
                            .strip()
                        )
                        strings.append(s)
                    results.append(strings)  # type: ignore[arg-type]
                offset += total_len
            else:
                # Numeric type
                np_dtype = f"{self.endian}{dtype}"
                item_size = np.dtype(np_dtype).itemsize
                total_size = item_size * count
                arr_data = data[offset : offset + total_size]
                arr = np.frombuffer(arr_data, dtype=np_dtype, count=count)
                if count == 1:
                    results.append(arr[0].item())
                else:
                    results.append(arr.copy())
                offset += total_size

        return tuple(results)

    def read_character_record(self, length: int) -> str:
        """
        Read a character record of specified length.

        Unlike read_string, this reads exactly 'length' bytes from the record
        without assuming the entire record is the string.

        Args:
            length: Number of characters to read

        Returns:
            The string value
        """
        data = self.read_record()
        return data[:length].decode("ascii", errors="replace").strip()

    def peek_record_size(self) -> int:
        """
        Peek at the next record size without advancing the file position.

        Returns:
            Size of the next record in bytes
        """
        if self._file is None:
            raise RuntimeError("File not open")

        pos = self._file.tell()
        marker_data = self._file.read(4)
        if len(marker_data) < 4:
            raise EOFError("End of file reached")

        record_length: int = struct.unpack(f"{self.endian}i", marker_data)[0]
        self._file.seek(pos)  # Restore position
        return record_length

    def at_eof(self) -> bool:
        """
        Check if we're at the end of the file.

        Returns:
            True if at end of file, False otherwise
        """
        if self._file is None:
            raise RuntimeError("File not open")

        pos = self._file.tell()
        data = self._file.read(1)
        if not data:
            return True
        self._file.seek(pos)
        return False


class FortranBinaryWriter:
    """
    Writer for Fortran unformatted binary files.

    Writes record markers before and after each record.
    """

    def __init__(self, filepath: Path | str, endian: str = "<") -> None:
        """
        Initialize the writer.

        Args:
            filepath: Path to the output file
            endian: Byte order ('<' = little-endian, '>' = big-endian)
        """
        self.filepath = Path(filepath)
        self.endian = endian
        self._file: BinaryIO | None = None

    def __enter__(self) -> FortranBinaryWriter:
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.filepath, "wb")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        if self._file:
            self._file.close()

    def write_record(self, data: bytes) -> None:
        """
        Write a single Fortran record with markers.

        Args:
            data: Record data as bytes
        """
        if self._file is None:
            raise RuntimeError("File not open")

        record_length = len(data)
        marker = struct.pack(f"{self.endian}i", record_length)

        self._file.write(marker)
        self._file.write(data)
        self._file.write(marker)

    def write_int(self, value: int) -> None:
        """Write a single integer record."""
        data = struct.pack(f"{self.endian}i", value)
        self.write_record(data)

    def write_int_array(self, arr: NDArray[np.int32]) -> None:
        """Write an integer array record."""
        arr_le = arr.astype(f"{self.endian}i4")
        self.write_record(arr_le.tobytes())

    def write_float(self, value: float) -> None:
        """Write a single float record (real*4)."""
        data = struct.pack(f"{self.endian}f", value)
        self.write_record(data)

    def write_float_array(self, arr: NDArray[np.float32]) -> None:
        """Write a float array record (real*4)."""
        arr_le = arr.astype(f"{self.endian}f4")
        self.write_record(arr_le.tobytes())

    def write_double(self, value: float) -> None:
        """Write a single double record (real*8)."""
        data = struct.pack(f"{self.endian}d", value)
        self.write_record(data)

    def write_double_array(self, arr: NDArray[np.float64]) -> None:
        """Write a double array record (real*8)."""
        arr_le = arr.astype(f"{self.endian}f8")
        self.write_record(arr_le.tobytes())

    def write_string(self, s: str, length: int | None = None) -> None:
        """Write a string record."""
        if length:
            s = s.ljust(length)[:length]
        data = s.encode("ascii")
        self.write_record(data)


class StreamAccessBinaryReader:
    """Reader for IWFM stream-access binary files (``ACCESS='STREAM'``).

    Unlike :class:`FortranBinaryReader`, IWFM preprocessor binary output
    is written with ``ACCESS='STREAM'`` which produces raw bytes without
    the 4-byte record-length markers that Fortran sequential writes add.
    The caller must supply explicit counts for every array read.
    """

    def __init__(self, filepath: Path | str, endian: str = "<") -> None:
        self.filepath = Path(filepath)
        self.endian = endian
        self._file: BinaryIO | None = None

    # -- context manager ---------------------------------------------------
    def __enter__(self) -> StreamAccessBinaryReader:
        self._file = open(self.filepath, "rb")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        if self._file:
            self._file.close()

    # -- helpers -----------------------------------------------------------
    def _read_bytes(self, n: int) -> bytes:
        if self._file is None:
            raise RuntimeError("File not open")
        data = self._file.read(n)
        if len(data) < n:
            raise EOFError(
                f"Expected {n} bytes at offset {self._file.tell() - len(data)}, got {len(data)}"
            )
        return data

    # -- scalar reads ------------------------------------------------------
    def read_int(self) -> int:
        """Read a single 4-byte integer."""
        result: int = struct.unpack(f"{self.endian}i", self._read_bytes(4))[0]
        return result

    def read_double(self) -> float:
        """Read a single 8-byte float (REAL*8)."""
        result: float = struct.unpack(f"{self.endian}d", self._read_bytes(8))[0]
        return result

    # -- array reads -------------------------------------------------------
    def read_ints(self, n: int) -> NDArray[np.int32]:
        """Read *n* consecutive 4-byte integers."""
        if n <= 0:
            return np.array([], dtype=np.int32)
        data = self._read_bytes(4 * n)
        return np.frombuffer(data, dtype=f"{self.endian}i4", count=n).copy()

    def read_doubles(self, n: int) -> NDArray[np.float64]:
        """Read *n* consecutive 8-byte floats."""
        if n <= 0:
            return np.array([], dtype=np.float64)
        data = self._read_bytes(8 * n)
        return np.frombuffer(data, dtype=f"{self.endian}f8", count=n).copy()

    # -- logical -----------------------------------------------------------
    def read_logical(self) -> bool:
        """Read a single Fortran LOGICAL (4-byte int, non-zero = True)."""
        return self.read_int() != 0

    def read_logicals(self, n: int) -> NDArray[np.bool_]:
        """Read *n* consecutive Fortran LOGICALs."""
        if n <= 0:
            return np.array([], dtype=np.bool_)
        raw = self.read_ints(n)
        return raw.astype(np.bool_)

    # -- string ------------------------------------------------------------
    def read_string(self, length: int) -> str:
        """Read a fixed-length ASCII string, right-stripped."""
        data = self._read_bytes(length)
        return data.decode("ascii", errors="replace").rstrip()

    # -- position ----------------------------------------------------------
    def get_position(self) -> int:
        if self._file is None:
            raise RuntimeError("File not open")
        return self._file.tell()

    def at_eof(self) -> bool:
        if self._file is None:
            raise RuntimeError("File not open")
        pos = self._file.tell()
        data = self._file.read(1)
        if not data:
            return True
        self._file.seek(pos)
        return False


def write_binary_mesh(filepath: Path | str, grid: AppGrid, endian: str = "<") -> None:
    """
    Write mesh data to a binary file.

    Args:
        filepath: Path to the output file
        grid: AppGrid instance to write
        endian: Byte order
    """
    with FortranBinaryWriter(filepath, endian) as f:
        f.write_int(grid.n_nodes)
        f.write_int(grid.n_elements)

        # Write coordinates
        f.write_double_array(grid.x)
        f.write_double_array(grid.y)

        # Build vertex array (0-based indices)
        n_elem = grid.n_elements
        vertex = np.zeros((n_elem, 4), dtype=np.int32)
        subregions = np.zeros(n_elem, dtype=np.int32)

        for i, elem_id in enumerate(sorted(grid.elements.keys())):
            elem = grid.elements[elem_id]
            for j, vid in enumerate(elem.vertices):
                vertex[i, j] = vid - 1  # Convert to 0-based
            # Leave 4th vertex as 0 for triangles
            subregions[i] = elem.subregion

        f.write_int_array(vertex.flatten())
        f.write_int_array(subregions)


def write_binary_stratigraphy(filepath: Path | str, strat: Stratigraphy, endian: str = "<") -> None:
    """
    Write stratigraphy data to a binary file.

    Args:
        filepath: Path to the output file
        strat: Stratigraphy instance to write
        endian: Byte order
    """
    with FortranBinaryWriter(filepath, endian) as f:
        f.write_int(strat.n_nodes)
        f.write_int(strat.n_layers)

        f.write_double_array(strat.gs_elev)
        f.write_double_array(strat.top_elev.flatten())
        f.write_double_array(strat.bottom_elev.flatten())
        f.write_int_array(strat.active_node.astype(np.int32).flatten())
