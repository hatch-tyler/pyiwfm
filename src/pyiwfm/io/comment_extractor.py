"""
Comment extraction from IWFM input files.

This module provides tools for extracting comments from IWFM input files,
preserving them for later restoration during file writing.

IWFM supports several comment styles:
- Full-line comments starting with 'C', 'c', or '*' in column 1
- Inline comments starting with '/' preceded by whitespace
- Header blocks using banner patterns (C***...*)

Classes:
    LineType: Enum for classifying input lines.
    CommentExtractor: Main class for extracting comments from files.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Iterator, Sequence

from pyiwfm.io.comment_metadata import (
    CommentMetadata,
    PreserveMode,
    SectionComments,
)

logger = logging.getLogger(__name__)


class LineType(Enum):
    """Classification of IWFM input file lines."""

    BLANK = auto()
    FULL_LINE_COMMENT = auto()
    INLINE_COMMENT = auto()
    DATA = auto()
    SECTION_HEADER = auto()
    BANNER = auto()


@dataclass
class ParsedLine:
    """A parsed line from an IWFM input file.

    Attributes:
        line_number: 1-based line number in the file.
        raw: Original line text including newline.
        stripped: Line with trailing whitespace removed.
        line_type: Classification of the line.
        data_part: Data portion of the line (for DATA and INLINE_COMMENT).
        comment_part: Comment portion of the line (for INLINE_COMMENT).
        keyword: Keyword from / KEYWORD suffix (e.g., "NNODES").
    """

    line_number: int
    raw: str
    stripped: str
    line_type: LineType
    data_part: str = ""
    comment_part: str = ""
    keyword: str = ""


class CommentExtractor:
    """Extract comments from IWFM input files.

    This class parses IWFM input files and extracts all comment information,
    organizing it into a CommentMetadata structure that can be used to
    preserve comments during round-trip file operations.

    IWFM comment conventions (from IWFM source code):
    - Full-line comment indicators: 'C', 'c', '*' in column 1
    - Inline comment indicator: '/' preceded by whitespace
    - Banner patterns: Lines of C***** or C----- used as section dividers

    Example:
        >>> extractor = CommentExtractor()
        >>> metadata = extractor.extract(Path("Preprocessor.in"))
        >>> metadata.save_for_file(Path("Preprocessor.in"))

    Attributes:
        FULL_LINE_CHARS: Characters that indicate a full-line comment.
        INLINE_CHAR: Character that indicates an inline comment.
        BANNER_PATTERN: Regex pattern for banner lines.
        KEYWORD_PATTERN: Regex pattern for / KEYWORD suffixes.
    """

    # Comment indicators from IWFM source
    # See: Class_AsciiFileType.f90 line 178: f_cCommentIndicators = 'Cc*'
    FULL_LINE_CHARS = frozenset(("C", "c", "*"))

    # Inline comment indicator
    # See: GeneralUtilities.f90 line 188: f_cInlineCommentChar = '/'
    INLINE_CHAR = "/"

    # Pattern for banner lines (C***... or C---...)
    BANNER_PATTERN = re.compile(r"^[Cc]\s*[\*\-=]{10,}")

    # Pattern for keyword descriptions (/ KEYWORD or / N: KEYWORD)
    KEYWORD_PATTERN = re.compile(r"\s*/\s*(\w+)(?:\s|$)")
    NUMBERED_KEYWORD_PATTERN = re.compile(r"/\s*\d+:\s*(.+?)(?:\s*\(|$)")

    # Pattern for section identification
    SECTION_PATTERNS = {
        "NODES": re.compile(r"NNODES|node.*coord", re.IGNORECASE),
        "ELEMENTS": re.compile(r"NELEM|element.*config", re.IGNORECASE),
        "STRATIGRAPHY": re.compile(r"NLAYERS|stratigraph", re.IGNORECASE),
        "STREAMS": re.compile(r"NREACH|stream.*config", re.IGNORECASE),
        "LAKES": re.compile(r"NLAKES|lake.*config", re.IGNORECASE),
        "GROUNDWATER": re.compile(r"groundwater|aquifer", re.IGNORECASE),
        "ROOTZONE": re.compile(r"root.*zone|land.*use", re.IGNORECASE),
        "SIMULATION": re.compile(r"simulation|time.*step", re.IGNORECASE),
    }

    def __init__(self, preserve_mode: PreserveMode = PreserveMode.FULL) -> None:
        """Initialize the comment extractor.

        Args:
            preserve_mode: Level of comment preservation.
        """
        self.preserve_mode = preserve_mode

    def extract(self, filepath: Path | str) -> CommentMetadata:
        """Extract comments from an IWFM input file.

        Args:
            filepath: Path to the IWFM input file.

        Returns:
            CommentMetadata containing all extracted comments.

        Raises:
            FileNotFoundError: If the file does not exist.
            IOError: If the file cannot be read.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        logger.info(f"Extracting comments from {filepath}")

        # Initialize metadata
        metadata = CommentMetadata(
            source_file=filepath.name,
            preserve_mode=self.preserve_mode,
        )

        # Read and parse file
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        parsed_lines = list(self._parse_lines(lines))

        # Extract header block (lines before first data)
        header_end = self._find_header_end(parsed_lines)
        for i in range(header_end):
            pl = parsed_lines[i]
            if pl.line_type in (
                LineType.FULL_LINE_COMMENT,
                LineType.BANNER,
                LineType.BLANK,
            ):
                metadata.header_block.append(pl.stripped)

        # Extract sections
        if self.preserve_mode in (PreserveMode.FULL, PreserveMode.HEADERS):
            self._extract_sections(parsed_lines, header_end, metadata)

        # Try to detect IWFM version from header
        metadata.iwfm_version = self._detect_version(metadata.header_block)

        logger.info(
            f"Extracted {len(metadata.header_block)} header lines, "
            f"{len(metadata.sections)} sections from {filepath.name}"
        )

        return metadata

    def _parse_lines(self, lines: Sequence[str]) -> Iterator[ParsedLine]:
        """Parse lines and classify each one.

        Args:
            lines: Sequence of raw line strings.

        Yields:
            ParsedLine for each input line.
        """
        for i, raw_line in enumerate(lines, start=1):
            stripped = raw_line.rstrip()
            yield self._classify_line(i, raw_line, stripped)

    def _classify_line(
        self, line_number: int, raw: str, stripped: str
    ) -> ParsedLine:
        """Classify a single line.

        Args:
            line_number: 1-based line number.
            raw: Original line text.
            stripped: Line with trailing whitespace removed.

        Returns:
            ParsedLine with classification and parsed components.
        """
        # Empty or whitespace-only line
        if not stripped:
            return ParsedLine(
                line_number=line_number,
                raw=raw,
                stripped=stripped,
                line_type=LineType.BLANK,
            )

        # Check first character for full-line comment
        first_char = stripped[0]
        if first_char in self.FULL_LINE_CHARS:
            # Check if it's a banner line
            if self.BANNER_PATTERN.match(stripped):
                return ParsedLine(
                    line_number=line_number,
                    raw=raw,
                    stripped=stripped,
                    line_type=LineType.BANNER,
                    comment_part=stripped,
                )
            return ParsedLine(
                line_number=line_number,
                raw=raw,
                stripped=stripped,
                line_type=LineType.FULL_LINE_COMMENT,
                comment_part=stripped,
            )

        # Check for inline comment
        inline_result = self._extract_inline_comment(stripped)
        if inline_result[1] is not None:
            # Has inline comment
            data_part, comment_part = inline_result
            keyword = self._extract_keyword(comment_part)
            return ParsedLine(
                line_number=line_number,
                raw=raw,
                stripped=stripped,
                line_type=LineType.INLINE_COMMENT,
                data_part=data_part,
                comment_part=comment_part,
                keyword=keyword,
            )

        # Plain data line
        return ParsedLine(
            line_number=line_number,
            raw=raw,
            stripped=stripped,
            line_type=LineType.DATA,
            data_part=stripped,
        )

    @staticmethod
    def _is_dss_pathname(text: str) -> bool:
        """Check if text starting with '/' is a DSS pathname.

        DSS pathnames have format /A/B/C/D/E/F/ with 6+ slashes
        in a contiguous non-whitespace token.

        Args:
            text: Text starting from the '/' character.

        Returns:
            True if the token is a DSS pathname.
        """
        if not text or text[0] != "/":
            return False
        slash_count = 0
        for ch in text:
            if ch == " ":
                break
            if ch == "/":
                slash_count += 1
        return slash_count >= 6

    def _extract_inline_comment(self, line: str) -> tuple[str, str | None]:
        """Extract inline comment from a line.

        IWFM inline comments start with '/' preceded by whitespace.
        The '/' must not be the first character (that would be in the
        data portion). DSS pathnames (/A/B/C/D/E/F/) are excluded.

        Args:
            line: Line text to parse.

        Returns:
            Tuple of (data_part, comment_part). comment_part is None
            if no inline comment is present.
        """
        # Find '/' preceded by whitespace
        # Must handle dates (09/30/1990), file paths, and DSS pathnames
        pos = 1  # Start after first char (first char is data)
        while pos < len(line):
            slash_pos = line.find(self.INLINE_CHAR, pos)
            if slash_pos == -1:
                break

            # Rule 1: Must be preceded by whitespace
            if slash_pos > 0 and line[slash_pos - 1].isspace():
                # Rule 2: Must NOT be a DSS pathname
                if not self._is_dss_pathname(line[slash_pos:]):
                    data_part = line[:slash_pos].rstrip()
                    comment_part = line[slash_pos:]
                    return data_part, comment_part

            pos = slash_pos + 1

        return line, None

    def _extract_keyword(self, comment: str) -> str:
        """Extract keyword from inline comment.

        IWFM uses / KEYWORD or / N: DESCRIPTION patterns.

        Args:
            comment: Comment text (starting with /).

        Returns:
            Extracted keyword or empty string.
        """
        # Try / KEYWORD pattern first
        match = self.KEYWORD_PATTERN.search(comment)
        if match:
            return match.group(1)

        # Try / N: DESCRIPTION pattern
        match = self.NUMBERED_KEYWORD_PATTERN.search(comment)
        if match:
            return match.group(1).strip()

        return ""

    def _find_header_end(self, parsed_lines: list[ParsedLine]) -> int:
        """Find where the header block ends.

        The header ends at the first non-comment, non-blank line,
        or at the first line that looks like a section start.

        Args:
            parsed_lines: List of parsed lines.

        Returns:
            Index of the first non-header line.
        """
        for i, pl in enumerate(parsed_lines):
            if pl.line_type in (LineType.DATA, LineType.INLINE_COMMENT):
                return i
        return len(parsed_lines)

    def _extract_sections(
        self,
        parsed_lines: list[ParsedLine],
        start_idx: int,
        metadata: CommentMetadata,
    ) -> None:
        """Extract section-level comments.

        Identifies sections based on keywords and comment patterns,
        then extracts comments within each section.

        Args:
            parsed_lines: List of parsed lines.
            start_idx: Index to start processing from.
            metadata: CommentMetadata to populate.
        """
        current_section: SectionComments | None = None
        current_section_name = "MAIN"
        pending_comments: list[str] = []
        data_index = 0  # Counter for data lines within section

        for pl in parsed_lines[start_idx:]:
            # Check if this starts a new section
            new_section = self._detect_section_start(pl)
            if new_section and new_section != current_section_name:
                # Save pending comments to previous section
                if current_section and pending_comments:
                    current_section.trailing_comments.extend(pending_comments)
                    pending_comments = []

                # Start new section
                current_section_name = new_section
                current_section = metadata.get_or_create_section(new_section)
                data_index = 0

            # Process line based on type
            if pl.line_type == LineType.BLANK:
                # Preserve blank lines in context
                if current_section and data_index > 0:
                    pending_comments.append("")
                continue

            elif pl.line_type in (LineType.FULL_LINE_COMMENT, LineType.BANNER):
                # Full line comment
                if current_section:
                    if data_index == 0:
                        # Before any data - section header comment
                        current_section.header_comments.append(pl.stripped)
                    else:
                        # After data - pending trailing comment
                        pending_comments.append(pl.stripped)

            elif pl.line_type == LineType.INLINE_COMMENT:
                # Data with inline comment
                if current_section and self.preserve_mode == PreserveMode.FULL:
                    # Flush pending comments
                    if pending_comments:
                        current_section.trailing_comments.extend(pending_comments)
                        pending_comments = []

                    # Store inline comment keyed by keyword or line index
                    if pl.keyword:
                        current_section.inline_comments[pl.keyword] = pl.comment_part
                    else:
                        # Try to extract data ID
                        data_id = self._extract_data_id(pl.data_part, data_index)
                        if data_id:
                            key = f"line:{data_id}"
                            current_section.inline_comments[key] = pl.comment_part

                data_index += 1

            elif pl.line_type == LineType.DATA:
                # Plain data line - flush pending comments
                if pending_comments and current_section:
                    current_section.trailing_comments.extend(pending_comments)
                    pending_comments = []
                data_index += 1

        # Save final pending comments
        if current_section and pending_comments:
            current_section.trailing_comments.extend(pending_comments)

    def _detect_section_start(self, pl: ParsedLine) -> str | None:
        """Detect if a line indicates the start of a new section.

        Args:
            pl: Parsed line to check.

        Returns:
            Section name if detected, None otherwise.
        """
        # Check keyword first
        if pl.keyword:
            keyword_upper = pl.keyword.upper()
            for section_name, pattern in self.SECTION_PATTERNS.items():
                if pattern.search(keyword_upper):
                    return section_name

        # Check comment content for section markers
        if pl.line_type in (LineType.FULL_LINE_COMMENT, LineType.BANNER):
            for section_name, pattern in self.SECTION_PATTERNS.items():
                if pattern.search(pl.comment_part):
                    return section_name

        return None

    def _extract_data_id(self, data_part: str, line_index: int) -> str:
        """Extract a data ID from the data portion of a line.

        Tries to find an integer ID at the start of the data.

        Args:
            data_part: Data portion of the line.
            line_index: 0-based index of data line within section.

        Returns:
            Extracted ID or line index as fallback.
        """
        # Try to extract leading integer
        parts = data_part.split()
        if parts:
            try:
                return str(int(float(parts[0])))
            except (ValueError, IndexError):
                pass
        return str(line_index)

    def _detect_version(self, header_lines: list[str]) -> str:
        """Detect IWFM version from header comments.

        Args:
            header_lines: Header comment lines.

        Returns:
            Version string if found, empty string otherwise.
        """
        version_pattern = re.compile(
            r"(?:IWFM|Version|ver\.?)\s*(\d+\.?\d*\.?\d*)", re.IGNORECASE
        )
        for line in header_lines:
            match = version_pattern.search(line)
            if match:
                return match.group(1)
        return ""


def extract_comments(filepath: Path | str) -> CommentMetadata:
    """Convenience function to extract comments from an IWFM file.

    Args:
        filepath: Path to the IWFM input file.

    Returns:
        CommentMetadata containing all extracted comments.

    Example:
        >>> metadata = extract_comments("Preprocessor.in")
        >>> print(len(metadata.header_block))
        15
    """
    extractor = CommentExtractor()
    return extractor.extract(filepath)


def extract_and_save_comments(filepath: Path | str) -> Path:
    """Extract comments and save to a sidecar file.

    Args:
        filepath: Path to the IWFM input file.

    Returns:
        Path to the saved sidecar file.

    Example:
        >>> sidecar = extract_and_save_comments("Preprocessor.in")
        >>> print(sidecar)
        Preprocessor.in.iwfm_comments.json
    """
    metadata = extract_comments(filepath)
    return metadata.save_for_file(filepath)
