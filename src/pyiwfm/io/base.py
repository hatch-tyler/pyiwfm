"""
Base classes for IWFM file I/O.

This module provides abstract base classes for reading and writing
IWFM model files in various formats, including support for comment
preservation during round-trip operations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

from pyiwfm.io.iwfm_writer import ensure_parent_dir

if TYPE_CHECKING:
    from pyiwfm.core.mesh import AppGrid
    from pyiwfm.core.model import IWFMModel
    from pyiwfm.core.stratigraphy import Stratigraphy
    from pyiwfm.io.comment_metadata import CommentMetadata


@dataclass
class FileInfo:
    """Information about an IWFM file."""

    path: Path
    format: str  # 'ascii', 'binary', 'hdf5', 'dss'
    version: str | None = None
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseReader(ABC):
    """Abstract base class for IWFM file readers."""

    def __init__(self, filepath: Path | str) -> None:
        """
        Initialize the reader.

        Args:
            filepath: Path to the file to read
        """
        self.filepath = Path(filepath)
        self._validate_file()

    def _validate_file(self) -> None:
        """Validate that the file exists and is readable."""
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")
        if not self.filepath.is_file():
            raise ValueError(f"Path is not a file: {self.filepath}")

    @abstractmethod
    def read(self) -> Any:
        """
        Read the file and return the parsed data.

        Returns:
            Parsed data (type depends on subclass)
        """
        pass

    @property
    @abstractmethod
    def format(self) -> str:
        """Return the file format identifier."""
        pass


class BaseWriter(ABC):
    """Abstract base class for IWFM file writers."""

    def __init__(self, filepath: Path | str) -> None:
        """
        Initialize the writer.

        Args:
            filepath: Path to the output file
        """
        self.filepath = Path(filepath)

    def _ensure_parent_exists(self) -> None:
        """Ensure the parent directory exists."""
        ensure_parent_dir(self.filepath)

    @abstractmethod
    def write(self, data: Any) -> None:
        """
        Write data to the file.

        Args:
            data: Data to write (type depends on subclass)
        """
        pass

    @property
    @abstractmethod
    def format(self) -> str:
        """Return the file format identifier."""
        pass


class ModelReader(BaseReader):
    """Abstract base class for reading complete IWFM models."""

    @abstractmethod
    def read(self) -> IWFMModel:
        """
        Read the model from file(s).

        Returns:
            Complete IWFMModel instance
        """
        pass

    @abstractmethod
    def read_mesh(self) -> AppGrid:
        """
        Read only the mesh from the model files.

        Returns:
            AppGrid instance
        """
        pass

    @abstractmethod
    def read_stratigraphy(self) -> Stratigraphy:
        """
        Read only the stratigraphy from the model files.

        Returns:
            Stratigraphy instance
        """
        pass


class ModelWriter(BaseWriter):
    """Abstract base class for writing complete IWFM models."""

    @abstractmethod
    def write(self, model: IWFMModel) -> None:
        """
        Write the model to file(s).

        Args:
            model: IWFMModel instance to write
        """
        pass

    @abstractmethod
    def write_mesh(self, mesh: AppGrid) -> None:
        """
        Write only the mesh.

        Args:
            mesh: AppGrid instance to write
        """
        pass

    @abstractmethod
    def write_stratigraphy(self, stratigraphy: Stratigraphy) -> None:
        """
        Write only the stratigraphy.

        Args:
            stratigraphy: Stratigraphy instance to write
        """
        pass


class BinaryReader(BaseReader):
    """Base class for reading IWFM binary files."""

    # Fortran record markers are 4 bytes
    RECORD_MARKER_SIZE = 4

    def __init__(self, filepath: Path | str, endian: str = "<") -> None:
        """
        Initialize the binary reader.

        Args:
            filepath: Path to the binary file
            endian: Byte order ('<' = little-endian, '>' = big-endian)
        """
        super().__init__(filepath)
        self.endian = endian

    @property
    def format(self) -> str:
        return "binary"

    def _read_fortran_record(self, f: BinaryIO) -> bytes:
        """
        Read a Fortran unformatted record.

        Fortran unformatted files have record markers (4-byte integers)
        at the start and end of each record indicating the record length.

        Args:
            f: Binary file object

        Returns:
            Record data as bytes
        """
        import struct

        # Read leading record marker
        marker_data = f.read(self.RECORD_MARKER_SIZE)
        if len(marker_data) < self.RECORD_MARKER_SIZE:
            raise EOFError("Unexpected end of file reading record marker")

        record_length = struct.unpack(f"{self.endian}i", marker_data)[0]

        # Read record data
        data = f.read(record_length)
        if len(data) < record_length:
            raise EOFError("Unexpected end of file reading record data")

        # Read trailing record marker
        trailing_marker = f.read(self.RECORD_MARKER_SIZE)
        trailing_length = struct.unpack(f"{self.endian}i", trailing_marker)[0]

        if trailing_length != record_length:
            raise ValueError(f"Record marker mismatch: {record_length} != {trailing_length}")

        return data


class BinaryWriter(BaseWriter):
    """Base class for writing IWFM binary files."""

    RECORD_MARKER_SIZE = 4

    def __init__(self, filepath: Path | str, endian: str = "<") -> None:
        """
        Initialize the binary writer.

        Args:
            filepath: Path to the output file
            endian: Byte order ('<' = little-endian, '>' = big-endian)
        """
        super().__init__(filepath)
        self.endian = endian

    @property
    def format(self) -> str:
        return "binary"

    def _write_fortran_record(self, f: BinaryIO, data: bytes) -> None:
        """
        Write a Fortran unformatted record.

        Args:
            f: Binary file object
            data: Record data as bytes
        """
        import struct

        record_length = len(data)
        marker = struct.pack(f"{self.endian}i", record_length)

        f.write(marker)
        f.write(data)
        f.write(marker)


# =============================================================================
# Comment-Aware Reader/Writer Base Classes
# =============================================================================


class CommentAwareReader(BaseReader):
    """Base class for readers that preserve comments.

    This class extends BaseReader to extract and preserve comments
    from IWFM input files during reading. The extracted comment
    metadata can be used later for round-trip preservation.

    Example:
        >>> reader = MyCommentAwareReader("Preprocessor.in", preserve_comments=True)
        >>> data = reader.read()
        >>> metadata = reader.comment_metadata
        >>> metadata.save_for_file("Preprocessor.in")

    Attributes:
        preserve_comments: Whether to extract and store comments.
        _comment_metadata: Extracted comment metadata (lazy-loaded).
    """

    def __init__(
        self,
        filepath: Path | str,
        preserve_comments: bool = True,
    ) -> None:
        """
        Initialize the comment-aware reader.

        Args:
            filepath: Path to the file to read.
            preserve_comments: If True, extract and store comments.
        """
        super().__init__(filepath)
        self.preserve_comments = preserve_comments
        self._comment_metadata: CommentMetadata | None = None

    @property
    def comment_metadata(self) -> CommentMetadata | None:
        """Get extracted comment metadata.

        Returns None if preserve_comments is False or if
        comments have not been extracted yet.
        """
        return self._comment_metadata

    def extract_comments(self) -> CommentMetadata:
        """Extract comments from the file.

        This method can be called explicitly to extract comments
        without reading the full file content.

        Returns:
            CommentMetadata containing all extracted comments.
        """
        from pyiwfm.io.comment_extractor import CommentExtractor

        extractor = CommentExtractor()
        self._comment_metadata = extractor.extract(self.filepath)
        return self._comment_metadata

    def _ensure_comments_extracted(self) -> None:
        """Ensure comments have been extracted if preservation is enabled."""
        if self.preserve_comments and self._comment_metadata is None:
            self.extract_comments()


class CommentAwareWriter(BaseWriter):
    """Base class for writers that can restore preserved comments.

    This class extends BaseWriter to support injecting preserved
    comments into output files, enabling round-trip preservation
    of user comments.

    Example:
        >>> # Load metadata from sidecar file
        >>> metadata = CommentMetadata.load_for_file("Preprocessor.in")
        >>> writer = MyCommentAwareWriter("output/Preprocessor.in", metadata)
        >>> writer.write(data)

    Attributes:
        comment_metadata: CommentMetadata to use for restoration.
        use_templates_for_missing: If True, use template defaults when
            no preserved comments exist.
    """

    def __init__(
        self,
        filepath: Path | str,
        comment_metadata: CommentMetadata | None = None,
        use_templates_for_missing: bool = True,
    ) -> None:
        """
        Initialize the comment-aware writer.

        Args:
            filepath: Path to the output file.
            comment_metadata: CommentMetadata with preserved comments.
            use_templates_for_missing: If True, use default templates
                when no preserved comments exist.
        """
        super().__init__(filepath)
        self.comment_metadata = comment_metadata
        self.use_templates_for_missing = use_templates_for_missing

    def has_preserved_comments(self) -> bool:
        """Check if preserved comments are available."""
        return self.comment_metadata is not None and self.comment_metadata.has_comments()

    def get_comment_writer(self) -> CommentWriter:
        """Get a CommentWriter configured with our metadata.

        Returns:
            CommentWriter instance for restoring comments.
        """
        from pyiwfm.io.comment_writer import CommentWriter

        return CommentWriter(
            self.comment_metadata,
            use_fallback=self.use_templates_for_missing,
        )

    def save_comment_metadata(self) -> Path | None:
        """Save comment metadata as a sidecar file.

        Returns:
            Path to saved sidecar file, or None if no metadata.
        """
        if self.comment_metadata is not None:
            return self.comment_metadata.save_for_file(self.filepath)
        return None


# Type alias for import convenience
from pyiwfm.io.comment_writer import CommentWriter  # noqa: E402
