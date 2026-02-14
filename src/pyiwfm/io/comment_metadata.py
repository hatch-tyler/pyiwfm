"""
Comment metadata storage for IWFM input files.

This module provides data classes for storing and serializing comments
extracted from IWFM input files, enabling round-trip preservation of
user-defined comments when reading and writing models.

The comment metadata is stored in JSON sidecar files (.iwfm_comments.json)
alongside the original input files.

Classes:
    PreserveMode: Enum for comment preservation levels.
    SectionComments: Comments associated with a specific file section.
    CommentMetadata: Complete comment metadata for an IWFM file.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PreserveMode(Enum):
    """Comment preservation level.

    Attributes:
        NONE: Do not preserve any comments.
        HEADERS: Preserve only file header blocks.
        FULL: Preserve all comments including inline comments.
    """

    NONE = "none"
    HEADERS = "headers"
    FULL = "full"


@dataclass
class SectionComments:
    """Comments associated with a specific section of an IWFM file.

    A section is a logical grouping within an IWFM file, such as "NODES",
    "ELEMENTS", "STRATIGRAPHY", etc. Each section can have:
    - Header comments that appear before the section data
    - Inline comments on data lines
    - Data comments keyed by element ID
    - Trailing comments after the section data

    Attributes:
        section_name: Name/identifier of the section.
        header_comments: Comment lines appearing before section data.
        inline_comments: Map of line key to inline comment text.
            Keys are typically field names or column headers.
        data_comments: Map of data identifier to comment text.
            Keys use format "type:id" (e.g., "node:157", "elem:42").
        trailing_comments: Comment lines appearing after section data.
    """

    section_name: str
    header_comments: list[str] = field(default_factory=list)
    inline_comments: dict[str, str] = field(default_factory=dict)
    data_comments: dict[str, str] = field(default_factory=dict)
    trailing_comments: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "section_name": self.section_name,
            "header_comments": self.header_comments,
            "inline_comments": self.inline_comments,
            "data_comments": self.data_comments,
            "trailing_comments": self.trailing_comments,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SectionComments:
        """Create from dictionary (JSON deserialization)."""
        return cls(
            section_name=data.get("section_name", ""),
            header_comments=data.get("header_comments", []),
            inline_comments=data.get("inline_comments", {}),
            data_comments=data.get("data_comments", {}),
            trailing_comments=data.get("trailing_comments", []),
        )

    def has_comments(self) -> bool:
        """Check if this section has any preserved comments."""
        return bool(
            self.header_comments
            or self.inline_comments
            or self.data_comments
            or self.trailing_comments
        )

    def get_data_comment(self, data_type: str, data_id: int | str) -> str | None:
        """Get comment for a specific data item.

        Args:
            data_type: Type of data (e.g., "node", "elem", "reach").
            data_id: ID of the data item.

        Returns:
            Comment text if found, None otherwise.
        """
        key = f"{data_type}:{data_id}"
        return self.data_comments.get(key)

    def set_data_comment(
        self, data_type: str, data_id: int | str, comment: str
    ) -> None:
        """Set comment for a specific data item.

        Args:
            data_type: Type of data (e.g., "node", "elem", "reach").
            data_id: ID of the data item.
            comment: Comment text.
        """
        key = f"{data_type}:{data_id}"
        self.data_comments[key] = comment


@dataclass
class CommentMetadata:
    """Complete comment metadata for an IWFM input file.

    This class stores all extracted comments from an IWFM input file,
    organized by section. It supports JSON serialization for storage
    in sidecar files.

    Attributes:
        version: Metadata format version (for future compatibility).
        source_file: Name of the original source file.
        iwfm_version: IWFM version string if detected.
        preserve_mode: Level of comment preservation.
        header_block: File-level header comment lines.
        sections: Dictionary mapping section names to SectionComments.
        file_metadata: Additional key-value metadata from the file.
    """

    version: str = "1.0"
    source_file: str = ""
    iwfm_version: str = ""
    preserve_mode: PreserveMode = PreserveMode.FULL
    header_block: list[str] = field(default_factory=list)
    sections: dict[str, SectionComments] = field(default_factory=dict)
    file_metadata: dict[str, Any] = field(default_factory=dict)

    # Standard sidecar file suffix
    SIDECAR_SUFFIX = ".iwfm_comments.json"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "version": self.version,
            "source_file": self.source_file,
            "iwfm_version": self.iwfm_version,
            "preserve_mode": self.preserve_mode.value,
            "header_block": self.header_block,
            "sections": {
                name: section.to_dict() for name, section in self.sections.items()
            },
            "file_metadata": self.file_metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CommentMetadata:
        """Create from dictionary (JSON deserialization)."""
        # Parse preserve_mode enum
        mode_str = data.get("preserve_mode", "full")
        try:
            preserve_mode = PreserveMode(mode_str)
        except ValueError:
            preserve_mode = PreserveMode.FULL

        # Parse sections
        sections_data = data.get("sections", {})
        sections = {
            name: SectionComments.from_dict(section_data)
            for name, section_data in sections_data.items()
        }

        return cls(
            version=data.get("version", "1.0"),
            source_file=data.get("source_file", ""),
            iwfm_version=data.get("iwfm_version", ""),
            preserve_mode=preserve_mode,
            header_block=data.get("header_block", []),
            sections=sections,
            file_metadata=data.get("file_metadata", {}),
        )

    def save(self, path: Path | str) -> None:
        """Save comment metadata to a JSON file.

        Args:
            path: Path to the JSON file to create.

        Raises:
            IOError: If the file cannot be written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.debug(f"Saved comment metadata to {path}")
        except IOError as e:
            logger.error(f"Failed to save comment metadata to {path}: {e}")
            raise

    @classmethod
    def load(cls, path: Path | str) -> CommentMetadata | None:
        """Load comment metadata from a JSON file.

        Args:
            path: Path to the JSON file to read.

        Returns:
            CommentMetadata instance if file exists and is valid,
            None otherwise.
        """
        path = Path(path)
        if not path.exists():
            logger.debug(f"No comment metadata file at {path}")
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except (IOError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load comment metadata from {path}: {e}")
            return None

    @classmethod
    def sidecar_path(cls, source_path: Path | str) -> Path:
        """Get the sidecar file path for a source file.

        The sidecar file is stored alongside the source file with
        the suffix '.iwfm_comments.json'.

        Args:
            source_path: Path to the IWFM input file.

        Returns:
            Path to the corresponding sidecar file.
        """
        source_path = Path(source_path)
        return source_path.parent / (source_path.name + cls.SIDECAR_SUFFIX)

    @classmethod
    def load_for_file(cls, source_path: Path | str) -> CommentMetadata | None:
        """Load comment metadata for an IWFM input file.

        Looks for a sidecar file alongside the source file.

        Args:
            source_path: Path to the IWFM input file.

        Returns:
            CommentMetadata if sidecar exists and is valid, None otherwise.
        """
        sidecar = cls.sidecar_path(source_path)
        return cls.load(sidecar)

    def save_for_file(self, source_path: Path | str) -> Path:
        """Save comment metadata as a sidecar file.

        Args:
            source_path: Path to the IWFM input file.

        Returns:
            Path to the saved sidecar file.
        """
        sidecar = self.sidecar_path(source_path)
        self.save(sidecar)
        return sidecar

    def get_section(self, section_name: str) -> SectionComments | None:
        """Get comments for a specific section.

        Args:
            section_name: Name of the section.

        Returns:
            SectionComments if found, None otherwise.
        """
        return self.sections.get(section_name)

    def get_or_create_section(self, section_name: str) -> SectionComments:
        """Get or create comments for a specific section.

        Args:
            section_name: Name of the section.

        Returns:
            SectionComments instance (existing or newly created).
        """
        if section_name not in self.sections:
            self.sections[section_name] = SectionComments(section_name=section_name)
        return self.sections[section_name]

    def has_comments(self) -> bool:
        """Check if any comments are stored."""
        if self.header_block:
            return True
        return any(section.has_comments() for section in self.sections.values())

    def merge(self, other: CommentMetadata) -> None:
        """Merge comments from another metadata instance.

        Used when combining comments from multiple source files
        during model loading.

        Args:
            other: Another CommentMetadata instance to merge from.
        """
        # Merge header blocks (avoid duplicates)
        for line in other.header_block:
            if line not in self.header_block:
                self.header_block.append(line)

        # Merge sections
        for name, section in other.sections.items():
            if name in self.sections:
                # Merge section contents
                existing = self.sections[name]
                for line in section.header_comments:
                    if line not in existing.header_comments:
                        existing.header_comments.append(line)
                existing.inline_comments.update(section.inline_comments)
                existing.data_comments.update(section.data_comments)
                for line in section.trailing_comments:
                    if line not in existing.trailing_comments:
                        existing.trailing_comments.append(line)
            else:
                self.sections[name] = section

        # Merge file metadata
        self.file_metadata.update(other.file_metadata)


@dataclass
class FileCommentMetadata:
    """Container for comment metadata across multiple IWFM files.

    When loading a complete IWFM model, this class holds the comment
    metadata for all component files, keyed by file type.

    Attributes:
        files: Dictionary mapping file type to CommentMetadata.
            Keys are standardized names like "preprocessor_main",
            "gw_main", "stream_main", etc.
    """

    files: dict[str, CommentMetadata] = field(default_factory=dict)

    def get(self, file_type: str) -> CommentMetadata | None:
        """Get comment metadata for a file type."""
        return self.files.get(file_type)

    def set(self, file_type: str, metadata: CommentMetadata) -> None:
        """Set comment metadata for a file type."""
        self.files[file_type] = metadata

    def save_all(self, output_dir: Path, file_paths: dict[str, Path]) -> None:
        """Save all comment metadata as sidecar files.

        Args:
            output_dir: Base output directory.
            file_paths: Mapping of file type to output path.
        """
        for file_type, metadata in self.files.items():
            if file_type in file_paths:
                metadata.save_for_file(file_paths[file_type])

    @classmethod
    def load_all(cls, file_paths: dict[str, Path]) -> FileCommentMetadata:
        """Load comment metadata for all files.

        Args:
            file_paths: Mapping of file type to source path.

        Returns:
            FileCommentMetadata with loaded metadata for each file.
        """
        result = cls()
        for file_type, path in file_paths.items():
            metadata = CommentMetadata.load_for_file(path)
            if metadata is not None:
                result.files[file_type] = metadata
        return result

    def has_comments(self) -> bool:
        """Check if any file has preserved comments."""
        return any(meta.has_comments() for meta in self.files.values())
