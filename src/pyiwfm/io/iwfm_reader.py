"""
Unified IWFM file line-reading utilities.

Matches the I/O approach in IWFM Fortran source code
(GeneralUtilities.f90, Class_AsciiFileType.f90).

Every ``io/`` reader should import helpers from this module rather than
defining its own copy.
"""

from __future__ import annotations

from typing import TextIO

from pyiwfm.core.exceptions import FileFormatError

# Full-line comment characters — matches IWFM SkipComment()
# (C, c, * only; NOT ! or #)
COMMENT_CHARS = ("C", "c", "*")


def is_comment_line(line: str) -> bool:
    """Check if line is an IWFM full-line comment or blank.

    Matches IWFM's ``SkipComment()`` which skips lines starting with
    ``C``, ``c``, or ``*`` in **column 1** (the first character of
    the raw line).  Lines with leading whitespace are data lines
    even if the first non-space character is ``C``.

    A Windows drive-letter prefix (``C:\\``) is **not** treated as a
    comment.
    """
    if not line or not line.strip():
        return True
    # Column-1 check: comment char must be the very first character
    ch = line[0]
    if ch not in COMMENT_CHARS:
        return False
    # Handle Windows drive letter: "C:\..." is NOT a comment
    if ch in ("C", "c") and len(line) > 1 and line[1] == ":":
        return False
    return True


def strip_inline_comment(line: str) -> tuple[str, str]:
    """Strip inline comment from an IWFM data line.

    Returns ``(value, description)``.

    Matches IWFM's ``FindInlineCommentPosition()`` +
    ``StripTextUntilCharacter()`` (GeneralUtilities.f90:716-782).
    Only ``/`` preceded by whitespace is treated as a comment
    delimiter.  ``#`` is **not** a comment character in IWFM.
    """
    for i, ch in enumerate(line):
        if ch != "/":
            continue
        # Must be preceded by whitespace
        if i == 0 or not line[i - 1].isspace():
            continue
        return line[:i].strip(), line[i + 1 :].strip()
    return line.strip(), ""


def next_data_value(f: TextIO, line_counter: list[int] | None = None) -> str:
    """Read next non-comment line and strip inline comment.

    Use for **scalar** or **string** data (file paths, parameters)
    where inline ``/ description`` should be removed.

    Matches IWFM pattern: ``SkipComment()`` + ``READ`` +
    ``StripTextUntilCharacter()``.
    """
    for line in f:
        if line_counter is not None:
            line_counter[0] += 1
        if is_comment_line(line):
            continue
        value, _ = strip_inline_comment(line)
        return value
    return ""


def next_data_line(f: TextIO, line_counter: list[int] | None = None) -> str:
    """Read next non-comment line **without** stripping inline comments.

    Use for **numeric data rows** (arrays, tables) where Fortran's
    free-format ``READ`` handles inline comments by reading exactly
    *N* values and ignoring the rest of the line.

    The caller should ``split()`` and take the first *N* tokens.
    """
    for line in f:
        if line_counter is not None:
            line_counter[0] += 1
        if is_comment_line(line):
            continue
        return line.strip()
    return ""


class LineBuffer:
    """Read-ahead buffer for positional-sequential IWFM files.

    Supports pushing a line back so that a tabular reader can stop
    at a boundary and leave the next line available for the caller.
    """

    __slots__ = ("_lines", "_pos")

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self._pos = 0

    @property
    def line_num(self) -> int:
        return self._pos

    def next_line(self) -> str | None:
        """Return the next raw line, or ``None`` at EOF."""
        if self._pos >= len(self._lines):
            return None
        line = self._lines[self._pos]
        self._pos += 1
        return line

    def pushback(self) -> None:
        """Push the last-read line back (decrement position)."""
        if self._pos > 0:
            self._pos -= 1

    def next_data(self) -> str:
        """Return next non-comment data value (inline comment stripped).

        Raises :class:`FileFormatError` on EOF.
        """
        while self._pos < len(self._lines):
            line = self._lines[self._pos]
            self._pos += 1
            if is_comment_line(line):
                continue
            value, _ = strip_inline_comment(line)
            if value:
                return value
        raise FileFormatError(
            "Unexpected end of file", line_number=self._pos
        )

    def next_data_or_empty(self) -> str:
        """Return next data value, or ``""`` at EOF.

        Skips comment lines (``C``/``c``/``*``) but stops at (and
        returns) the first non-comment line even if it is blank.
        """
        while self._pos < len(self._lines):
            line = self._lines[self._pos]
            self._pos += 1
            if line and line[0] in COMMENT_CHARS:
                continue
            value, _ = strip_inline_comment(line)
            return value
        return ""

    def next_raw_data(self) -> str:
        """Return next non-comment raw line (no inline comment stripping).

        Use for numeric data rows where the caller splits and takes
        the first *N* tokens.
        """
        while self._pos < len(self._lines):
            line = self._lines[self._pos]
            self._pos += 1
            if is_comment_line(line):
                continue
            stripped = line.strip()
            if stripped:
                return stripped
        return ""
