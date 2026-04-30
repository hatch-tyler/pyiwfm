"""ASCII file-format primitives used by IWFM.

This package collects the helpers for IWFM's plain-text input and output
files into one place:

- :mod:`pyiwfm.io.ascii.reader` — line-reading helpers (``parse_int``,
  ``parse_float``, ``next_data_value``, ``next_data_line``,
  ``next_data_or_empty``, ``is_comment_line``, ``strip_inline_comment``,
  ``resolve_path``, ``parse_version``, ``version_ge``, ``ReaderMixin``,
  ``LineBuffer``).

- :mod:`pyiwfm.io.ascii.writer` — line-writing helpers (``write_comment``,
  ``write_value``, ``ensure_parent_dir``, ``open_iwfm_file``,
  ``write_element_group``).

- :mod:`pyiwfm.io.ascii.comment_extractor`,
  :mod:`pyiwfm.io.ascii.comment_metadata`,
  :mod:`pyiwfm.io.ascii.comment_writer` — round-trip comment preservation
  for IWFM ASCII files.

The package re-exports the public symbols from each submodule. The v1.x
paths ``pyiwfm.io.iwfm_reader``, ``pyiwfm.io.iwfm_writer``,
``pyiwfm.io.comment_extractor``, ``pyiwfm.io.comment_metadata``, and
``pyiwfm.io.comment_writer`` are gone in v2.0; use
``from pyiwfm.io.ascii import X`` instead. See
``docs/MIGRATION_v1_to_v2.md`` § 10.
"""

from __future__ import annotations

from pyiwfm.io.ascii.comment_extractor import (
    CommentExtractor,
    LineType,
    ParsedLine,
    extract_and_save_comments,
    extract_comments,
)
from pyiwfm.io.ascii.comment_metadata import (
    CommentMetadata,
    FileCommentMetadata,
    PreserveMode,
    SectionComments,
)
from pyiwfm.io.ascii.comment_writer import (
    CommentInjector,
    CommentWriter,
)
from pyiwfm.io.ascii.reader import (
    COMMENT_CHARS,
    LineBuffer,
    ReaderMixin,
    is_comment_line,
    next_data_line,
    next_data_or_empty,
    next_data_value,
    parse_float,
    parse_int,
    parse_version,
    resolve_path,
    strip_inline_comment,
    version_ge,
)
from pyiwfm.io.ascii.writer import (
    ensure_parent_dir,
    open_iwfm_file,
    write_comment,
    write_element_group,
    write_value,
)

__all__ = [
    # reader.py
    "COMMENT_CHARS",
    "LineBuffer",
    "ReaderMixin",
    "is_comment_line",
    "next_data_line",
    "next_data_or_empty",
    "next_data_value",
    "parse_float",
    "parse_int",
    "parse_version",
    "resolve_path",
    "strip_inline_comment",
    "version_ge",
    # writer.py
    "ensure_parent_dir",
    "open_iwfm_file",
    "write_comment",
    "write_element_group",
    "write_value",
    # comment_extractor.py
    "CommentExtractor",
    "LineType",
    "ParsedLine",
    "extract_and_save_comments",
    "extract_comments",
    # comment_metadata.py
    "CommentMetadata",
    "FileCommentMetadata",
    "PreserveMode",
    "SectionComments",
    # comment_writer.py
    "CommentInjector",
    "CommentWriter",
]
