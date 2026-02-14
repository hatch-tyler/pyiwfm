"""Coverage tests for CommentAwareReader and CommentAwareWriter in io/base.py.

Covers missing lines: 305-307, 316, 327-331, 335-336, 373-375, 379, 390-392, 403-405

Tests:
- CommentAwareReader __init__ with preserve_comments True/False (lines 305-307)
- CommentAwareReader comment_metadata property returns None (line 316)
- CommentAwareReader extract_comments with mocked CommentExtractor (lines 327-331)
- CommentAwareReader _ensure_comments_extracted (lines 335-336)
- CommentAwareWriter __init__ with metadata and template flag (lines 373-375)
- CommentAwareWriter has_preserved_comments (line 379)
- CommentAwareWriter get_comment_writer (lines 390-392)
- CommentAwareWriter save_comment_metadata (lines 403-405)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from pyiwfm.io.base import CommentAwareReader, CommentAwareWriter
from pyiwfm.io.comment_metadata import CommentMetadata, PreserveMode, SectionComments
from pyiwfm.io.comment_writer import CommentWriter


# =============================================================================
# Concrete implementations for testing abstract classes
# =============================================================================


class _ConcreteCommentReader(CommentAwareReader):
    """Concrete CommentAwareReader for testing."""

    @property
    def format(self) -> str:
        return "test"

    def read(self) -> Any:
        self._ensure_comments_extracted()
        return self.filepath.read_text()


class _ConcreteCommentWriter(CommentAwareWriter):
    """Concrete CommentAwareWriter for testing."""

    @property
    def format(self) -> str:
        return "test"

    def write(self, data: Any) -> None:
        self._ensure_parent_exists()
        self.filepath.write_text(str(data))


# =============================================================================
# Test CommentAwareReader __init__
# =============================================================================


class TestCommentAwareReaderInit:
    """Test CommentAwareReader initialization."""

    def test_comment_aware_reader_init_preserve_true(self, tmp_path: Path) -> None:
        """Init with preserve_comments=True."""
        test_file = tmp_path / "test.in"
        test_file.write_text("C  comment\n100\n")

        reader = _ConcreteCommentReader(test_file, preserve_comments=True)

        assert reader.filepath == test_file
        assert reader.preserve_comments is True
        assert reader._comment_metadata is None

    def test_comment_aware_reader_init_preserve_false(self, tmp_path: Path) -> None:
        """Init with preserve_comments=False."""
        test_file = tmp_path / "test.in"
        test_file.write_text("C  comment\n100\n")

        reader = _ConcreteCommentReader(test_file, preserve_comments=False)

        assert reader.preserve_comments is False
        assert reader._comment_metadata is None

    def test_comment_aware_reader_init_default_preserve(self, tmp_path: Path) -> None:
        """Init with default preserve_comments (should be True)."""
        test_file = tmp_path / "test.in"
        test_file.write_text("C  comment\n100\n")

        reader = _ConcreteCommentReader(test_file)

        assert reader.preserve_comments is True


# =============================================================================
# Test CommentAwareReader comment_metadata property
# =============================================================================


class TestCommentAwareReaderProperty:
    """Test comment_metadata property."""

    def test_comment_metadata_property_returns_none_initially(
        self, tmp_path: Path
    ) -> None:
        """comment_metadata property returns None before extraction."""
        test_file = tmp_path / "test.in"
        test_file.write_text("C  comment\n100\n")

        reader = _ConcreteCommentReader(test_file, preserve_comments=True)

        assert reader.comment_metadata is None

    def test_comment_metadata_property_after_extraction(
        self, tmp_path: Path
    ) -> None:
        """comment_metadata property returns metadata after extract_comments()."""
        test_file = tmp_path / "test.in"
        test_file.write_text("C  comment\n100\n")

        reader = _ConcreteCommentReader(test_file, preserve_comments=True)

        mock_metadata = CommentMetadata(source_file="test.in")
        with patch(
            "pyiwfm.io.comment_extractor.CommentExtractor"
        ) as MockExtractor:
            instance = MockExtractor.return_value
            instance.extract.return_value = mock_metadata

            reader.extract_comments()

        assert reader.comment_metadata is mock_metadata


# =============================================================================
# Test CommentAwareReader extract_comments
# =============================================================================


class TestCommentAwareReaderExtract:
    """Test extract_comments method with mocked CommentExtractor."""

    def test_extract_comments(self, tmp_path: Path) -> None:
        """Call extract_comments() -> CommentExtractor.extract() invoked."""
        test_file = tmp_path / "test.in"
        test_file.write_text("C  header\n100 / NNODES\n")

        reader = _ConcreteCommentReader(test_file)

        mock_metadata = CommentMetadata(source_file="test.in")
        with patch(
            "pyiwfm.io.comment_extractor.CommentExtractor"
        ) as MockExtractor:
            instance = MockExtractor.return_value
            instance.extract.return_value = mock_metadata

            result = reader.extract_comments()

        MockExtractor.assert_called_once()
        instance.extract.assert_called_once_with(test_file)
        assert result is mock_metadata
        assert reader._comment_metadata is mock_metadata

    def test_extract_comments_stores_metadata(self, tmp_path: Path) -> None:
        """extract_comments stores result in _comment_metadata."""
        test_file = tmp_path / "test.in"
        test_file.write_text("data\n")

        reader = _ConcreteCommentReader(test_file)

        mock_metadata = CommentMetadata(
            source_file="test.in",
            header_block=["C  preserved header"],
        )
        with patch(
            "pyiwfm.io.comment_extractor.CommentExtractor"
        ) as MockExtractor:
            instance = MockExtractor.return_value
            instance.extract.return_value = mock_metadata

            reader.extract_comments()

        assert reader.comment_metadata is not None
        assert reader.comment_metadata.header_block == ["C  preserved header"]


# =============================================================================
# Test _ensure_comments_extracted
# =============================================================================


class TestEnsureCommentsExtracted:
    """Test _ensure_comments_extracted triggers extraction when needed."""

    def test_ensure_comments_extracted_with_preserve_true(
        self, tmp_path: Path
    ) -> None:
        """preserve_comments=True and no metadata -> triggers extraction."""
        test_file = tmp_path / "test.in"
        test_file.write_text("C  comment\n100\n")

        reader = _ConcreteCommentReader(test_file, preserve_comments=True)

        mock_metadata = CommentMetadata(source_file="test.in")
        with patch(
            "pyiwfm.io.comment_extractor.CommentExtractor"
        ) as MockExtractor:
            instance = MockExtractor.return_value
            instance.extract.return_value = mock_metadata

            reader._ensure_comments_extracted()

        instance.extract.assert_called_once_with(test_file)
        assert reader._comment_metadata is mock_metadata

    def test_ensure_comments_extracted_with_preserve_false(
        self, tmp_path: Path
    ) -> None:
        """preserve_comments=False -> does NOT trigger extraction."""
        test_file = tmp_path / "test.in"
        test_file.write_text("C  comment\n100\n")

        reader = _ConcreteCommentReader(test_file, preserve_comments=False)

        # No patch needed since CommentExtractor should never be imported
        reader._ensure_comments_extracted()

        assert reader._comment_metadata is None

    def test_ensure_comments_extracted_already_extracted(
        self, tmp_path: Path
    ) -> None:
        """Already extracted -> does NOT extract again."""
        test_file = tmp_path / "test.in"
        test_file.write_text("data\n")

        reader = _ConcreteCommentReader(test_file, preserve_comments=True)

        # Simulate prior extraction
        existing_metadata = CommentMetadata(source_file="test.in")
        reader._comment_metadata = existing_metadata

        with patch(
            "pyiwfm.io.comment_extractor.CommentExtractor"
        ) as MockExtractor:
            reader._ensure_comments_extracted()

        MockExtractor.assert_not_called()
        assert reader._comment_metadata is existing_metadata

    def test_read_triggers_ensure_extraction(self, tmp_path: Path) -> None:
        """read() calls _ensure_comments_extracted which triggers extraction."""
        test_file = tmp_path / "test.in"
        test_file.write_text("C  comment\nfile data\n")

        reader = _ConcreteCommentReader(test_file, preserve_comments=True)

        mock_metadata = CommentMetadata(source_file="test.in")
        with patch(
            "pyiwfm.io.comment_extractor.CommentExtractor"
        ) as MockExtractor:
            instance = MockExtractor.return_value
            instance.extract.return_value = mock_metadata

            content = reader.read()

        assert content == "C  comment\nfile data\n"
        assert reader._comment_metadata is mock_metadata


# =============================================================================
# Test CommentAwareWriter __init__
# =============================================================================


class TestCommentAwareWriterInit:
    """Test CommentAwareWriter initialization."""

    def test_comment_aware_writer_init_with_metadata(self, tmp_path: Path) -> None:
        """Init with metadata and default template flag."""
        metadata = CommentMetadata(source_file="test.in")
        output = tmp_path / "output.in"

        writer = _ConcreteCommentWriter(output, comment_metadata=metadata)

        assert writer.filepath == output
        assert writer.comment_metadata is metadata
        assert writer.use_templates_for_missing is True

    def test_comment_aware_writer_init_no_metadata(self, tmp_path: Path) -> None:
        """Init with no metadata."""
        output = tmp_path / "output.in"

        writer = _ConcreteCommentWriter(output)

        assert writer.comment_metadata is None
        assert writer.use_templates_for_missing is True

    def test_comment_aware_writer_init_template_false(self, tmp_path: Path) -> None:
        """Init with use_templates_for_missing=False."""
        output = tmp_path / "output.in"

        writer = _ConcreteCommentWriter(
            output,
            comment_metadata=None,
            use_templates_for_missing=False,
        )

        assert writer.use_templates_for_missing is False


# =============================================================================
# Test has_preserved_comments
# =============================================================================


class TestHasPreservedComments:
    """Test has_preserved_comments method."""

    def test_has_preserved_comments_none_metadata(self, tmp_path: Path) -> None:
        """With None metadata -> returns False."""
        writer = _ConcreteCommentWriter(tmp_path / "out.in")
        assert writer.has_preserved_comments() is False

    def test_has_preserved_comments_empty_metadata(self, tmp_path: Path) -> None:
        """With empty metadata (no comments) -> returns False."""
        metadata = CommentMetadata(source_file="test.in")
        writer = _ConcreteCommentWriter(
            tmp_path / "out.in", comment_metadata=metadata
        )
        assert writer.has_preserved_comments() is False

    def test_has_preserved_comments_with_header(self, tmp_path: Path) -> None:
        """With metadata containing header block -> returns True."""
        metadata = CommentMetadata(
            source_file="test.in",
            header_block=["C  preserved header"],
        )
        writer = _ConcreteCommentWriter(
            tmp_path / "out.in", comment_metadata=metadata
        )
        assert writer.has_preserved_comments() is True

    def test_has_preserved_comments_with_section(self, tmp_path: Path) -> None:
        """With metadata containing section comments -> returns True."""
        metadata = CommentMetadata(source_file="test.in")
        metadata.sections["NODES"] = SectionComments(
            section_name="NODES",
            header_comments=["C  node section"],
        )
        writer = _ConcreteCommentWriter(
            tmp_path / "out.in", comment_metadata=metadata
        )
        assert writer.has_preserved_comments() is True


# =============================================================================
# Test get_comment_writer
# =============================================================================


class TestGetCommentWriter:
    """Test get_comment_writer method."""

    def test_get_comment_writer_with_metadata(self, tmp_path: Path) -> None:
        """get_comment_writer returns CommentWriter with correct params."""
        metadata = CommentMetadata(
            source_file="test.in",
            header_block=["C  header"],
        )
        writer = _ConcreteCommentWriter(
            tmp_path / "out.in",
            comment_metadata=metadata,
            use_templates_for_missing=True,
        )

        cw = writer.get_comment_writer()

        assert isinstance(cw, CommentWriter)
        assert cw.metadata is metadata
        assert cw.use_fallback is True

    def test_get_comment_writer_no_fallback(self, tmp_path: Path) -> None:
        """get_comment_writer with use_templates_for_missing=False."""
        metadata = CommentMetadata(source_file="test.in")
        writer = _ConcreteCommentWriter(
            tmp_path / "out.in",
            comment_metadata=metadata,
            use_templates_for_missing=False,
        )

        cw = writer.get_comment_writer()

        assert isinstance(cw, CommentWriter)
        assert cw.use_fallback is False

    def test_get_comment_writer_none_metadata(self, tmp_path: Path) -> None:
        """get_comment_writer with None metadata."""
        writer = _ConcreteCommentWriter(tmp_path / "out.in")

        cw = writer.get_comment_writer()

        assert isinstance(cw, CommentWriter)
        assert cw.metadata is None


# =============================================================================
# Test save_comment_metadata
# =============================================================================


class TestSaveCommentMetadata:
    """Test save_comment_metadata method."""

    def test_save_comment_metadata_with_metadata(self, tmp_path: Path) -> None:
        """With metadata -> saves sidecar file and returns path."""
        metadata = CommentMetadata(
            source_file="test.in",
            header_block=["C  header"],
        )
        output_file = tmp_path / "test.in"
        output_file.write_text("placeholder")

        writer = _ConcreteCommentWriter(output_file, comment_metadata=metadata)

        sidecar = writer.save_comment_metadata()

        assert sidecar is not None
        assert sidecar.exists()
        assert sidecar.name == "test.in.iwfm_comments.json"

    def test_save_comment_metadata_none_returns_none(self, tmp_path: Path) -> None:
        """With None metadata -> returns None."""
        writer = _ConcreteCommentWriter(tmp_path / "test.in")

        result = writer.save_comment_metadata()

        assert result is None

    def test_save_comment_metadata_roundtrip(self, tmp_path: Path) -> None:
        """Save metadata, then verify it can be loaded back."""
        metadata = CommentMetadata(
            source_file="test.in",
            iwfm_version="2025",
            header_block=["C  test header"],
        )
        metadata.sections["SEC1"] = SectionComments(
            section_name="SEC1",
            header_comments=["C  section header"],
        )

        output_file = tmp_path / "test.in"
        output_file.write_text("placeholder")

        writer = _ConcreteCommentWriter(output_file, comment_metadata=metadata)
        sidecar = writer.save_comment_metadata()

        # Load it back
        loaded = CommentMetadata.load(sidecar)
        assert loaded is not None
        assert loaded.source_file == "test.in"
        assert loaded.iwfm_version == "2025"
        assert "SEC1" in loaded.sections
        assert loaded.sections["SEC1"].header_comments == ["C  section header"]
