"""
Comment restoration for IWFM input files.

This module provides tools for restoring preserved comments when writing
IWFM input files, enabling round-trip preservation of user-defined comments.

Classes:
    CommentWriter: Main class for restoring comments during file writing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TextIO

from pyiwfm.io.comment_metadata import (
    CommentMetadata,
    PreserveMode,
    SectionComments,
)

logger = logging.getLogger(__name__)


class CommentWriter:
    """Restore comments when writing IWFM input files.

    This class provides methods for injecting preserved comments into
    generated IWFM output files. It works with CommentMetadata extracted
    by CommentExtractor.

    The writer supports:
    - Restoring file header blocks
    - Adding section header comments
    - Appending inline comments to data lines
    - Inserting trailing comments after sections

    Example:
        >>> from pyiwfm.io.comment_metadata import CommentMetadata
        >>> metadata = CommentMetadata.load_for_file("Preprocessor.in")
        >>> writer = CommentWriter(metadata)
        >>> header = writer.restore_header()
        >>> print(header)

    Attributes:
        metadata: The CommentMetadata to use for restoration.
        use_fallback: If True, use default formatting when no comments exist.
    """

    # Default banner line format
    DEFAULT_BANNER = "C" + "*" * 78

    # Default section divider
    DEFAULT_DIVIDER = "C" + "-" * 78

    def __init__(
        self,
        metadata: CommentMetadata | None = None,
        use_fallback: bool = True,
    ) -> None:
        """Initialize the comment writer.

        Args:
            metadata: CommentMetadata to use for restoration.
                If None, no comments will be restored.
            use_fallback: If True, use default formatting when
                no preserved comments exist for a section.
        """
        self.metadata = metadata
        self.use_fallback = use_fallback

    def restore_header(self, fallback_lines: list[str] | None = None) -> str:
        """Restore the file header block.

        Args:
            fallback_lines: Lines to use if no header is preserved.

        Returns:
            Header text with newlines, or empty string if no header.
        """
        if self.metadata and self.metadata.header_block:
            return "\n".join(self.metadata.header_block) + "\n"

        if self.use_fallback and fallback_lines:
            return "\n".join(fallback_lines) + "\n"

        return ""

    def restore_section_header(
        self,
        section_name: str,
        fallback_lines: list[str] | None = None,
    ) -> str:
        """Restore header comments for a section.

        Args:
            section_name: Name of the section.
            fallback_lines: Lines to use if no comments are preserved.

        Returns:
            Section header text with newlines.
        """
        section = self._get_section(section_name)
        if section and section.header_comments:
            return "\n".join(section.header_comments) + "\n"

        if self.use_fallback and fallback_lines:
            return "\n".join(fallback_lines) + "\n"

        return ""

    def restore_section_trailing(
        self,
        section_name: str,
        fallback_lines: list[str] | None = None,
    ) -> str:
        """Restore trailing comments for a section.

        Args:
            section_name: Name of the section.
            fallback_lines: Lines to use if no comments are preserved.

        Returns:
            Trailing comment text with newlines.
        """
        section = self._get_section(section_name)
        if section and section.trailing_comments:
            return "\n".join(section.trailing_comments) + "\n"

        if self.use_fallback and fallback_lines:
            return "\n".join(fallback_lines) + "\n"

        return ""

    def format_data_with_comment(
        self,
        data: str,
        section_name: str,
        key: str,
        fallback_comment: str | None = None,
    ) -> str:
        """Format a data line with its preserved inline comment.

        Args:
            data: Data portion of the line.
            section_name: Name of the section containing this data.
            key: Key for looking up the comment (keyword or "line:N").
            fallback_comment: Comment to use if none preserved.

        Returns:
            Formatted line with inline comment if available.
        """
        comment = self._get_inline_comment(section_name, key)
        if comment is None:
            comment = fallback_comment

        if comment:
            # Ensure proper spacing
            if not comment.startswith(" "):
                comment = " " + comment
            return f"{data}{comment}"
        return data

    def format_value_with_keyword(
        self,
        value: str,
        keyword: str,
        section_name: str | None = None,
        width: int = 20,
    ) -> str:
        """Format a value line with / KEYWORD suffix.

        Tries to restore the original comment format if preserved,
        otherwise uses the standard / KEYWORD format.

        Args:
            value: Value to format.
            keyword: IWFM keyword (e.g., "NNODES", "NLAYERS").
            section_name: Section to look up comments from.
            width: Width for value field.

        Returns:
            Formatted line with keyword suffix.
        """
        # Check for preserved comment
        if section_name:
            preserved = self._get_inline_comment(section_name, keyword)
            if preserved:
                return f"{value:<{width}}{preserved}"

        # Use standard format
        return f"{value:<{width}} / {keyword}"

    def get_data_comment(
        self,
        section_name: str,
        data_type: str,
        data_id: int | str,
    ) -> str | None:
        """Get preserved comment for a data item.

        Args:
            section_name: Section containing the data.
            data_type: Type of data (e.g., "node", "elem").
            data_id: ID of the data item.

        Returns:
            Comment text if preserved, None otherwise.
        """
        section = self._get_section(section_name)
        if section:
            return section.get_data_comment(data_type, data_id)
        return None

    def write_header_to_file(
        self,
        file: TextIO,
        fallback_lines: list[str] | None = None,
    ) -> None:
        """Write header block to a file.

        Args:
            file: File object to write to.
            fallback_lines: Lines to use if no header preserved.
        """
        header = self.restore_header(fallback_lines)
        if header:
            file.write(header)

    def write_section_header_to_file(
        self,
        file: TextIO,
        section_name: str,
        fallback_lines: list[str] | None = None,
    ) -> None:
        """Write section header comments to a file.

        Args:
            file: File object to write to.
            section_name: Name of the section.
            fallback_lines: Lines to use if no comments preserved.
        """
        header = self.restore_section_header(section_name, fallback_lines)
        if header:
            file.write(header)

    def write_section_trailing_to_file(
        self,
        file: TextIO,
        section_name: str,
        fallback_lines: list[str] | None = None,
    ) -> None:
        """Write section trailing comments to a file.

        Args:
            file: File object to write to.
            section_name: Name of the section.
            fallback_lines: Lines to use if no comments preserved.
        """
        trailing = self.restore_section_trailing(section_name, fallback_lines)
        if trailing:
            file.write(trailing)

    def write_data_line(
        self,
        file: TextIO,
        data: str,
        section_name: str,
        key: str,
        fallback_comment: str | None = None,
    ) -> None:
        """Write a data line with optional inline comment.

        Args:
            file: File object to write to.
            data: Data portion of the line.
            section_name: Section containing this data.
            key: Key for comment lookup.
            fallback_comment: Comment to use if none preserved.
        """
        line = self.format_data_with_comment(
            data, section_name, key, fallback_comment
        )
        file.write(line + "\n")

    def _get_section(self, section_name: str) -> SectionComments | None:
        """Get section comments from metadata."""
        if self.metadata:
            return self.metadata.get_section(section_name)
        return None

    def _get_inline_comment(self, section_name: str, key: str) -> str | None:
        """Get inline comment for a key in a section."""
        section = self._get_section(section_name)
        if section:
            return section.inline_comments.get(key)
        return None

    def has_preserved_comments(self) -> bool:
        """Check if any comments are available for restoration."""
        return self.metadata is not None and self.metadata.has_comments()

    def has_section_comments(self, section_name: str) -> bool:
        """Check if a section has preserved comments."""
        section = self._get_section(section_name)
        return section is not None and section.has_comments()


class CommentInjector:
    """Inject comments into template-rendered content.

    This class post-processes template output to inject preserved
    comments, allowing templates to generate structure while
    preserving user comments.

    Example:
        >>> injector = CommentInjector(metadata)
        >>> output = injector.inject_header(rendered_content)
    """

    def __init__(self, metadata: CommentMetadata | None = None) -> None:
        """Initialize the comment injector.

        Args:
            metadata: CommentMetadata containing preserved comments.
        """
        self.metadata = metadata
        self.writer = CommentWriter(metadata)

    def inject_header(
        self,
        content: str,
        header_marker: str = "C*****",
    ) -> str:
        """Inject preserved header into content.

        Replaces the template-generated header (identified by marker)
        with the preserved header if available.

        Args:
            content: Template-rendered content.
            header_marker: Marker identifying the header start.

        Returns:
            Content with injected header.
        """
        if not self.metadata or not self.metadata.header_block:
            return content

        # Find header end in content
        lines = content.split("\n")
        header_end = 0
        in_header = False

        for i, line in enumerate(lines):
            if line.startswith(header_marker):
                in_header = True
            elif in_header and not line.startswith("C") and line.strip():
                header_end = i
                break

        if header_end > 0:
            # Replace header
            preserved_header = "\n".join(self.metadata.header_block)
            return preserved_header + "\n" + "\n".join(lines[header_end:])

        return content

    def inject_section_comments(
        self,
        content: str,
        section_name: str,
        section_marker: str,
    ) -> str:
        """Inject section comments into content.

        Finds the section by marker and injects preserved comments.

        Args:
            content: Template-rendered content.
            section_name: Name of the section.
            section_marker: Text that identifies section start.

        Returns:
            Content with injected section comments.
        """
        section = self.writer._get_section(section_name)
        if not section or not section.header_comments:
            return content

        # Find section marker
        marker_pos = content.find(section_marker)
        if marker_pos == -1:
            return content

        # Find line start
        line_start = content.rfind("\n", 0, marker_pos)
        if line_start == -1:
            line_start = 0
        else:
            line_start += 1

        # Insert section header comments before the marker line
        section_header = "\n".join(section.header_comments) + "\n"
        return content[:line_start] + section_header + content[line_start:]

    def process_content(
        self,
        content: str,
        sections: dict[str, str] | None = None,
    ) -> str:
        """Process content with all comment injections.

        Args:
            content: Template-rendered content.
            sections: Dict mapping section names to their markers.

        Returns:
            Fully processed content with all comments injected.
        """
        # Inject header
        result = self.inject_header(content)

        # Inject section comments
        if sections:
            for section_name, marker in sections.items():
                result = self.inject_section_comments(result, section_name, marker)

        return result
