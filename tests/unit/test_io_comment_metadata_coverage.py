"""Coverage tests for io/comment_metadata.py module.

Covers missing lines: 177-178, 213-215, 237-239, 326->325, 333-341, 379-381, 393-398

Tests:
- from_dict with invalid preserve_mode (lines 177-178)
- save IOError path (lines 213-215)
- load with invalid JSON (lines 237-239)
- merge with duplicate headers (line 326->325)
- merge with overlapping sections (lines 333-341)
- FileCommentMetadata.save_all (lines 379-381)
- FileCommentMetadata.load_all (lines 393-398)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from pyiwfm.io.comment_metadata import (
    CommentMetadata,
    FileCommentMetadata,
    PreserveMode,
    SectionComments,
)

# =============================================================================
# Test from_dict with invalid preserve_mode
# =============================================================================


class TestFromDictInvalidPreserveMode:
    """Test CommentMetadata.from_dict() with invalid preserve_mode value."""

    def test_from_dict_invalid_preserve_mode(self) -> None:
        """Pass invalid preserve_mode string -> defaults to PreserveMode.FULL."""
        data = {
            "version": "1.0",
            "source_file": "test.in",
            "iwfm_version": "2025",
            "preserve_mode": "INVALID_MODE",
            "header_block": [],
            "sections": {},
            "file_metadata": {},
        }
        meta = CommentMetadata.from_dict(data)
        assert meta.preserve_mode == PreserveMode.FULL

    def test_from_dict_empty_preserve_mode(self) -> None:
        """Pass empty preserve_mode string -> defaults to FULL."""
        data = {
            "preserve_mode": "",
        }
        meta = CommentMetadata.from_dict(data)
        assert meta.preserve_mode == PreserveMode.FULL

    def test_from_dict_numeric_preserve_mode(self) -> None:
        """Pass numeric preserve_mode -> defaults to FULL."""
        data = {
            "preserve_mode": "42",
        }
        meta = CommentMetadata.from_dict(data)
        assert meta.preserve_mode == PreserveMode.FULL


# =============================================================================
# Test save IOError
# =============================================================================


class TestSaveIOError:
    """Test CommentMetadata.save() IOError handling."""

    def test_save_ioerror(self, tmp_path: Path) -> None:
        """Mock open() to raise IOError -> verify IOError is re-raised."""
        meta = CommentMetadata(source_file="test.in")
        output = tmp_path / "output.json"

        with patch("builtins.open", side_effect=OSError("disk full")):
            with pytest.raises(IOError, match="disk full"):
                meta.save(output)

    def test_save_to_nonexistent_deep_path_succeeds(self, tmp_path: Path) -> None:
        """Save to a deeply nested path that doesn't exist yet -> should succeed."""
        meta = CommentMetadata(source_file="test.in")
        output = tmp_path / "a" / "b" / "c" / "metadata.json"
        meta.save(output)
        assert output.exists()

        # Verify contents
        with open(output, encoding="utf-8") as f:
            data = json.load(f)
        assert data["source_file"] == "test.in"


# =============================================================================
# Test load with invalid JSON
# =============================================================================


class TestLoadInvalidJSON:
    """Test CommentMetadata.load() with malformed JSON."""

    def test_load_invalid_json(self, tmp_path: Path) -> None:
        """Write malformed JSON -> load() returns None."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{invalid json content!!!", encoding="utf-8")

        result = CommentMetadata.load(bad_file)
        assert result is None

    def test_load_truncated_json(self, tmp_path: Path) -> None:
        """Write truncated JSON -> load() returns None."""
        bad_file = tmp_path / "truncated.json"
        bad_file.write_text('{"version": "1.0", "source_file":', encoding="utf-8")

        result = CommentMetadata.load(bad_file)
        assert result is None

    def test_load_empty_file(self, tmp_path: Path) -> None:
        """Write empty file -> load() returns None (JSONDecodeError)."""
        empty_file = tmp_path / "empty.json"
        empty_file.write_text("", encoding="utf-8")

        result = CommentMetadata.load(empty_file)
        assert result is None


# =============================================================================
# Test merge with duplicate headers
# =============================================================================


class TestMergeDuplicateHeaders:
    """Test CommentMetadata.merge() with overlapping header lines."""

    def test_merge_duplicate_headers(self) -> None:
        """Merge two metadata with overlapping headers -> verify deduplication."""
        meta1 = CommentMetadata(
            header_block=["C  Line A", "C  Line B", "C  Line C"],
        )
        meta2 = CommentMetadata(
            header_block=["C  Line B", "C  Line C", "C  Line D"],
        )

        meta1.merge(meta2)

        # "Line B" and "Line C" are duplicates and should not be added again
        assert meta1.header_block == [
            "C  Line A",
            "C  Line B",
            "C  Line C",
            "C  Line D",
        ]

    def test_merge_identical_headers(self) -> None:
        """Merge with identical headers -> no duplicates added."""
        meta1 = CommentMetadata(header_block=["C  Same header"])
        meta2 = CommentMetadata(header_block=["C  Same header"])

        meta1.merge(meta2)

        assert meta1.header_block == ["C  Same header"]


# =============================================================================
# Test merge with existing (overlapping) sections
# =============================================================================


class TestMergeExistingSection:
    """Test CommentMetadata.merge() with same section names."""

    def test_merge_existing_section(self) -> None:
        """Merge metadata with same section names -> inline/data/trailing merged."""
        meta1 = CommentMetadata()
        meta1.sections["NODES"] = SectionComments(
            section_name="NODES",
            header_comments=["C  header 1"],
            inline_comments={"NNODES": "/ node count"},
            data_comments={"node:1": "first"},
            trailing_comments=["C  trailing 1"],
        )

        meta2 = CommentMetadata()
        meta2.sections["NODES"] = SectionComments(
            section_name="NODES",
            header_comments=["C  header 1", "C  header 2"],  # header 1 is duplicate
            inline_comments={"FACTXY": "/ scale factor"},  # new inline key
            data_comments={"node:1": "overwritten", "node:2": "second"},
            trailing_comments=["C  trailing 1", "C  trailing 2"],  # trailing 1 is dup
        )

        meta1.merge(meta2)

        merged_section = meta1.sections["NODES"]

        # Header comments: "C  header 1" is duplicate -> should appear once, "C  header 2" added
        assert merged_section.header_comments == ["C  header 1", "C  header 2"]

        # Inline comments: both keys present (update merges)
        assert merged_section.inline_comments["NNODES"] == "/ node count"
        assert merged_section.inline_comments["FACTXY"] == "/ scale factor"

        # Data comments: "node:1" overwritten by meta2, "node:2" added
        assert merged_section.data_comments["node:1"] == "overwritten"
        assert merged_section.data_comments["node:2"] == "second"

        # Trailing comments: "C  trailing 1" is dup, "C  trailing 2" is new
        assert merged_section.trailing_comments == ["C  trailing 1", "C  trailing 2"]

    def test_merge_new_section(self) -> None:
        """Merge with a section that doesn't exist in target -> added as-is."""
        meta1 = CommentMetadata()
        meta1.sections["SEC1"] = SectionComments(section_name="SEC1")

        meta2 = CommentMetadata()
        meta2.sections["SEC2"] = SectionComments(
            section_name="SEC2",
            header_comments=["C  sec2 header"],
        )

        meta1.merge(meta2)

        assert "SEC1" in meta1.sections
        assert "SEC2" in meta1.sections
        assert meta1.sections["SEC2"].header_comments == ["C  sec2 header"]

    def test_merge_file_metadata(self) -> None:
        """Merge file_metadata dicts."""
        meta1 = CommentMetadata(file_metadata={"key1": "val1"})
        meta2 = CommentMetadata(file_metadata={"key2": "val2", "key1": "overwritten"})

        meta1.merge(meta2)

        assert meta1.file_metadata["key1"] == "overwritten"
        assert meta1.file_metadata["key2"] == "val2"


# =============================================================================
# Test FileCommentMetadata.save_all
# =============================================================================


class TestSaveAll:
    """Test FileCommentMetadata.save_all() with file_paths dict."""

    def test_save_all(self, tmp_path: Path) -> None:
        """save_all writes sidecar files for each matching file type."""
        fcm = FileCommentMetadata()
        meta_pp = CommentMetadata(source_file="preproc.in")
        meta_gw = CommentMetadata(source_file="gw.in")
        fcm.set("preprocessor_main", meta_pp)
        fcm.set("gw_main", meta_gw)

        file_paths = {
            "preprocessor_main": tmp_path / "preproc.in",
            "gw_main": tmp_path / "gw.in",
        }

        # Create the source files so paths are valid
        for p in file_paths.values():
            p.write_text("placeholder")

        fcm.save_all(tmp_path, file_paths)

        # Verify sidecar files exist
        sidecar_pp = tmp_path / "preproc.in.iwfm_comments.json"
        sidecar_gw = tmp_path / "gw.in.iwfm_comments.json"
        assert sidecar_pp.exists()
        assert sidecar_gw.exists()

        # Verify content
        with open(sidecar_pp, encoding="utf-8") as f:
            data = json.load(f)
        assert data["source_file"] == "preproc.in"

    def test_save_all_missing_file_type(self, tmp_path: Path) -> None:
        """save_all skips file types not present in file_paths."""
        fcm = FileCommentMetadata()
        meta = CommentMetadata(source_file="test.in")
        fcm.set("extra_type", meta)

        # file_paths does NOT include "extra_type"
        file_paths = {
            "preprocessor_main": tmp_path / "preproc.in",
        }
        (tmp_path / "preproc.in").write_text("placeholder")

        # Should not raise, just skip non-matching types
        fcm.save_all(tmp_path, file_paths)

        # No sidecar for extra_type since it wasn't in file_paths
        # and no sidecar for preprocessor_main since it wasn't in fcm.files
        assert not (tmp_path / "preproc.in.iwfm_comments.json").exists()


# =============================================================================
# Test FileCommentMetadata.load_all
# =============================================================================


class TestLoadAll:
    """Test FileCommentMetadata.load_all() with sidecar files."""

    def test_load_all(self, tmp_path: Path) -> None:
        """load_all reads sidecar files for each file path."""
        # Create source files and sidecar files
        pp_path = tmp_path / "preproc.in"
        gw_path = tmp_path / "gw.in"
        pp_path.write_text("placeholder")
        gw_path.write_text("placeholder")

        # Create sidecar files by saving metadata
        meta_pp = CommentMetadata(source_file="preproc.in")
        meta_pp.save_for_file(pp_path)

        meta_gw = CommentMetadata(
            source_file="gw.in",
            header_block=["C  GW header"],
        )
        meta_gw.save_for_file(gw_path)

        file_paths = {
            "preprocessor_main": pp_path,
            "gw_main": gw_path,
        }

        result = FileCommentMetadata.load_all(file_paths)

        assert "preprocessor_main" in result.files
        assert "gw_main" in result.files
        assert result.files["preprocessor_main"].source_file == "preproc.in"
        assert result.files["gw_main"].source_file == "gw.in"
        assert result.files["gw_main"].header_block == ["C  GW header"]

    def test_load_all_missing_sidecar(self, tmp_path: Path) -> None:
        """load_all skips file types with no sidecar file."""
        pp_path = tmp_path / "preproc.in"
        pp_path.write_text("placeholder")

        # Create sidecar for preproc only
        meta_pp = CommentMetadata(source_file="preproc.in")
        meta_pp.save_for_file(pp_path)

        gw_path = tmp_path / "gw.in"
        gw_path.write_text("placeholder")
        # No sidecar for gw

        file_paths = {
            "preprocessor_main": pp_path,
            "gw_main": gw_path,
        }

        result = FileCommentMetadata.load_all(file_paths)

        assert "preprocessor_main" in result.files
        assert "gw_main" not in result.files

    def test_load_all_empty_dict(self) -> None:
        """load_all with empty file_paths returns empty FileCommentMetadata."""
        result = FileCommentMetadata.load_all({})
        assert result.files == {}
