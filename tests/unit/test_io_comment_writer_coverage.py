"""Coverage tests for comment_writer.py module.

Tests CommentWriter (header, section, inline-comment restoration)
and CommentInjector (header and section comment injection into
template-rendered content).
"""

from __future__ import annotations

import io

import pytest

from pyiwfm.io.comment_metadata import (
    CommentMetadata,
    PreserveMode,
    SectionComments,
)
from pyiwfm.io.comment_writer import (
    CommentInjector,
    CommentWriter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def metadata_with_comments() -> CommentMetadata:
    """CommentMetadata with a header block and one section."""
    meta = CommentMetadata(
        source_file="test.in",
        iwfm_version="2025",
        preserve_mode=PreserveMode.FULL,
        header_block=[
            "C******* HEADER *******",
            "C  Preserved header comment line 1",
            "C  Preserved header comment line 2",
            "C******* END *******",
        ],
    )
    meta.sections["NODES"] = SectionComments(
        section_name="NODES",
        header_comments=["C  Node section header", "C-----------"],
        inline_comments={"NNODES": "/ NNODES", "FACTXY": "/ FACTXY"},
        data_comments={"node:1": "SW corner"},
        trailing_comments=["C  End of nodes", "C-----------"],
    )
    return meta


@pytest.fixture
def empty_metadata() -> CommentMetadata:
    """CommentMetadata with no comments at all."""
    return CommentMetadata(source_file="empty.in")


# ---------------------------------------------------------------------------
# CommentWriter tests
# ---------------------------------------------------------------------------


class TestCommentWriterNoMetadata:
    """CommentWriter behaviour when metadata is None."""

    def test_restore_header_returns_empty(self) -> None:
        writer = CommentWriter(None, use_fallback=False)
        assert writer.restore_header() == ""

    def test_restore_header_fallback(self) -> None:
        writer = CommentWriter(None, use_fallback=True)
        result = writer.restore_header(fallback_lines=["C  Fallback line"])
        assert "Fallback line" in result
        assert result.endswith("\n")

    def test_has_preserved_comments_false(self) -> None:
        writer = CommentWriter(None)
        assert writer.has_preserved_comments() is False

    def test_has_section_comments_false(self) -> None:
        writer = CommentWriter(None)
        assert writer.has_section_comments("ANY") is False


class TestCommentWriterWithMetadata:
    """CommentWriter behaviour with populated metadata."""

    def test_restore_header(self, metadata_with_comments: CommentMetadata) -> None:
        writer = CommentWriter(metadata_with_comments)
        header = writer.restore_header()
        assert "Preserved header comment line 1" in header
        assert "Preserved header comment line 2" in header
        assert header.endswith("\n")

    def test_restore_header_ignores_fallback_when_present(
        self, metadata_with_comments: CommentMetadata
    ) -> None:
        writer = CommentWriter(metadata_with_comments)
        header = writer.restore_header(fallback_lines=["C  SHOULD NOT APPEAR"])
        assert "SHOULD NOT APPEAR" not in header
        assert "Preserved header" in header

    def test_restore_section_header(
        self, metadata_with_comments: CommentMetadata
    ) -> None:
        writer = CommentWriter(metadata_with_comments)
        section_hdr = writer.restore_section_header("NODES")
        assert "Node section header" in section_hdr

    def test_restore_section_header_missing_section_uses_fallback(
        self, metadata_with_comments: CommentMetadata
    ) -> None:
        writer = CommentWriter(metadata_with_comments, use_fallback=True)
        result = writer.restore_section_header(
            "NONEXISTENT", fallback_lines=["C  Fallback section header"]
        )
        assert "Fallback section header" in result

    def test_restore_section_header_missing_no_fallback(
        self, metadata_with_comments: CommentMetadata
    ) -> None:
        writer = CommentWriter(metadata_with_comments, use_fallback=False)
        result = writer.restore_section_header("NONEXISTENT")
        assert result == ""

    def test_restore_section_trailing(
        self, metadata_with_comments: CommentMetadata
    ) -> None:
        writer = CommentWriter(metadata_with_comments)
        trailing = writer.restore_section_trailing("NODES")
        assert "End of nodes" in trailing

    def test_restore_section_trailing_fallback(
        self, metadata_with_comments: CommentMetadata
    ) -> None:
        writer = CommentWriter(metadata_with_comments, use_fallback=True)
        result = writer.restore_section_trailing(
            "MISSING", fallback_lines=["C  Trailing fallback"]
        )
        assert "Trailing fallback" in result

    def test_format_data_with_comment_preserved(
        self, metadata_with_comments: CommentMetadata
    ) -> None:
        writer = CommentWriter(metadata_with_comments)
        line = writer.format_data_with_comment("100", "NODES", "NNODES")
        assert "100" in line
        assert "NNODES" in line

    def test_format_data_with_comment_fallback(
        self, metadata_with_comments: CommentMetadata
    ) -> None:
        writer = CommentWriter(metadata_with_comments)
        line = writer.format_data_with_comment(
            "200", "NODES", "MISSING_KEY", fallback_comment="/ FB"
        )
        assert "200" in line
        assert "FB" in line

    def test_format_data_with_comment_no_comment(
        self, empty_metadata: CommentMetadata
    ) -> None:
        writer = CommentWriter(empty_metadata)
        line = writer.format_data_with_comment("300", "NODES", "KEY")
        assert line == "300"

    def test_format_value_with_keyword_preserved(
        self, metadata_with_comments: CommentMetadata
    ) -> None:
        writer = CommentWriter(metadata_with_comments)
        line = writer.format_value_with_keyword("100", "NNODES", section_name="NODES")
        assert "100" in line
        assert "NNODES" in line

    def test_format_value_with_keyword_default_format(self) -> None:
        writer = CommentWriter(None)
        line = writer.format_value_with_keyword("42", "NLAYERS", width=15)
        assert "42" in line
        assert "/ NLAYERS" in line
        # Value field should be padded to width 15
        assert line.startswith("42")

    def test_has_preserved_comments_true(
        self, metadata_with_comments: CommentMetadata
    ) -> None:
        writer = CommentWriter(metadata_with_comments)
        assert writer.has_preserved_comments() is True

    def test_has_section_comments_true(
        self, metadata_with_comments: CommentMetadata
    ) -> None:
        writer = CommentWriter(metadata_with_comments)
        assert writer.has_section_comments("NODES") is True

    def test_has_section_comments_false_for_missing(
        self, metadata_with_comments: CommentMetadata
    ) -> None:
        writer = CommentWriter(metadata_with_comments)
        assert writer.has_section_comments("NONEXISTENT") is False

    def test_get_data_comment(
        self, metadata_with_comments: CommentMetadata
    ) -> None:
        writer = CommentWriter(metadata_with_comments)
        comment = writer.get_data_comment("NODES", "node", 1)
        assert comment == "SW corner"

    def test_get_data_comment_missing(
        self, metadata_with_comments: CommentMetadata
    ) -> None:
        writer = CommentWriter(metadata_with_comments)
        comment = writer.get_data_comment("NODES", "node", 999)
        assert comment is None


class TestCommentWriterFileIO:
    """Tests for write_*_to_file convenience methods."""

    def test_write_header_to_file(
        self, metadata_with_comments: CommentMetadata
    ) -> None:
        writer = CommentWriter(metadata_with_comments)
        buf = io.StringIO()
        writer.write_header_to_file(buf)
        content = buf.getvalue()
        assert "Preserved header" in content

    def test_write_section_header_to_file(
        self, metadata_with_comments: CommentMetadata
    ) -> None:
        writer = CommentWriter(metadata_with_comments)
        buf = io.StringIO()
        writer.write_section_header_to_file(buf, "NODES")
        assert "Node section header" in buf.getvalue()

    def test_write_section_trailing_to_file(
        self, metadata_with_comments: CommentMetadata
    ) -> None:
        writer = CommentWriter(metadata_with_comments)
        buf = io.StringIO()
        writer.write_section_trailing_to_file(buf, "NODES")
        assert "End of nodes" in buf.getvalue()

    def test_write_data_line(
        self, metadata_with_comments: CommentMetadata
    ) -> None:
        writer = CommentWriter(metadata_with_comments)
        buf = io.StringIO()
        writer.write_data_line(buf, "100", "NODES", "NNODES")
        assert buf.getvalue().endswith("\n")
        assert "100" in buf.getvalue()


# ---------------------------------------------------------------------------
# CommentInjector tests
# ---------------------------------------------------------------------------


class TestCommentInjector:
    """Tests for CommentInjector class."""

    def test_inject_header_no_metadata(self) -> None:
        injector = CommentInjector(None)
        content = "C***** old header\nC  line\n100\n"
        assert injector.inject_header(content) == content

    def test_inject_header_replaces(
        self, metadata_with_comments: CommentMetadata
    ) -> None:
        injector = CommentInjector(metadata_with_comments)
        content = (
            "C***** template header\n"
            "C  template line\n"
            "100  / NNODES\n"
            "1.0  / FACTXY\n"
        )
        result = injector.inject_header(content)
        assert "Preserved header comment line 1" in result
        assert "template header" not in result
        # Data lines should be preserved
        assert "100" in result

    def test_inject_header_no_marker_found(
        self, metadata_with_comments: CommentMetadata
    ) -> None:
        """When the header marker is absent, content is returned unchanged."""
        injector = CommentInjector(metadata_with_comments)
        content = "100  / NNODES\n"
        result = injector.inject_header(content, header_marker="C*****")
        assert result == content

    def test_inject_section_comments(
        self, metadata_with_comments: CommentMetadata
    ) -> None:
        injector = CommentInjector(metadata_with_comments)
        content = "C  NODES section\n100  / NNODES\n"
        result = injector.inject_section_comments(content, "NODES", "C  NODES")
        assert "Node section header" in result

    def test_inject_section_comments_no_marker(
        self, metadata_with_comments: CommentMetadata
    ) -> None:
        """When the marker is absent, content is returned unchanged."""
        injector = CommentInjector(metadata_with_comments)
        content = "100\n200\n"
        result = injector.inject_section_comments(
            content, "NODES", "NONEXISTENT_MARKER"
        )
        assert result == content

    def test_inject_section_comments_no_section(
        self, metadata_with_comments: CommentMetadata
    ) -> None:
        """When the section does not exist in metadata, content is unchanged."""
        injector = CommentInjector(metadata_with_comments)
        content = "C  ELEMENTS\n50\n"
        result = injector.inject_section_comments(
            content, "MISSING_SECTION", "C  ELEMENTS"
        )
        assert result == content

    def test_process_content_header_and_sections(
        self, metadata_with_comments: CommentMetadata
    ) -> None:
        injector = CommentInjector(metadata_with_comments)
        content = (
            "C***** template header\n"
            "C  old\n"
            "100  / NNODES\n"
            "C  NODES section\n"
            "data line\n"
        )
        result = injector.process_content(
            content, sections={"NODES": "C  NODES"}
        )
        assert "Preserved header" in result
        assert "Node section header" in result

    def test_process_content_no_sections_arg(
        self, metadata_with_comments: CommentMetadata
    ) -> None:
        injector = CommentInjector(metadata_with_comments)
        content = (
            "C***** template header\n"
            "C  old\n"
            "100  / data\n"
        )
        result = injector.process_content(content, sections=None)
        # Header injection still happens
        assert "Preserved header" in result
