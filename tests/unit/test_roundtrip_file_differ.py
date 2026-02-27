"""Tests for roundtrip/file_differ.py.

Covers:
- _float_equal(): exact match, tolerance, non-numeric
- _lines_match(): identical, whitespace, float tolerance, token count
- _extract_data_lines(): comment stripping, data preservation
- diff_iwfm_files(): byte-identical, data-identical, missing, max_diffs
- diff_all_files(): auto-discovery, file_mapping, no common files
- InputDiffResult: properties and summary()
"""

from __future__ import annotations

from pathlib import Path

from pyiwfm.roundtrip.file_differ import (
    FileDiffResult,
    InputDiffResult,
    _extract_data_lines,
    _float_equal,
    _lines_match,
    diff_all_files,
    diff_iwfm_files,
)

# ---------------------------------------------------------------------------
# _float_equal
# ---------------------------------------------------------------------------


class TestFloatEqual:
    def test_exact_match(self) -> None:
        assert _float_equal("1.23456", "1.23456")

    def test_within_tolerance(self) -> None:
        assert _float_equal("1.0000001", "1.0000002", rtol=1e-6)

    def test_outside_tolerance(self) -> None:
        assert not _float_equal("1.0", "2.0", rtol=1e-6)

    def test_zero_zero(self) -> None:
        assert _float_equal("0.0", "0.0")

    def test_non_numeric_returns_false(self) -> None:
        # _float_equal returns False for non-numeric tokens
        assert not _float_equal("abc", "abc")

    def test_non_numeric_different(self) -> None:
        assert not _float_equal("abc", "def")

    def test_one_numeric_one_not(self) -> None:
        assert not _float_equal("1.0", "abc")

    def test_denom_zero_both_zero(self) -> None:
        # Both exactly zero: denom=0 branch
        assert _float_equal("0", "0.0")


# ---------------------------------------------------------------------------
# _lines_match
# ---------------------------------------------------------------------------


class TestLinesMatch:
    def test_identical(self) -> None:
        assert _lines_match("1  2  3.0  HELLO", "1  2  3.0  HELLO")

    def test_whitespace_normalization(self) -> None:
        assert _lines_match("1   2   3.0", "1 2 3.0")

    def test_float_tolerance(self) -> None:
        assert _lines_match("1  2.00000001  3", "1  2.00000002  3")

    def test_different_token_count(self) -> None:
        assert not _lines_match("1  2  3", "1  2")

    def test_different_values(self) -> None:
        assert not _lines_match("1  2  3", "1  2  99")


# ---------------------------------------------------------------------------
# _extract_data_lines
# ---------------------------------------------------------------------------


class TestExtractDataLines:
    def test_strips_comment_lines(self, tmp_path: Path) -> None:
        f = tmp_path / "test.dat"
        f.write_text("C This is a comment\n* Another comment\n1 2 3\n4 5 6\n")
        lines = _extract_data_lines(f)
        assert len(lines) == 2
        assert "1" in lines[0]

    def test_strips_blank_lines(self, tmp_path: Path) -> None:
        f = tmp_path / "test.dat"
        f.write_text("1 2 3\n\n4 5 6\n")
        lines = _extract_data_lines(f)
        assert len(lines) == 2

    def test_preserves_data(self, tmp_path: Path) -> None:
        f = tmp_path / "test.dat"
        f.write_text("10  20  30.5\n")
        lines = _extract_data_lines(f)
        assert len(lines) == 1
        assert "10" in lines[0]
        assert "30.5" in lines[0]

    def test_strips_inline_comments(self, tmp_path: Path) -> None:
        f = tmp_path / "test.dat"
        f.write_text("1 2 3 / description\n")
        lines = _extract_data_lines(f)
        assert len(lines) == 1
        # Inline comment should be stripped
        assert "/" not in lines[0] or "description" not in lines[0]


# ---------------------------------------------------------------------------
# diff_iwfm_files
# ---------------------------------------------------------------------------


class TestDiffIwfmFiles:
    def test_byte_identical(self, tmp_path: Path) -> None:
        content = "C comment\n1 2 3.0\n4 5 6.0\n"
        orig = tmp_path / "orig.dat"
        writ = tmp_path / "writ.dat"
        orig.write_text(content)
        writ.write_text(content)

        result = diff_iwfm_files(orig, writ, file_key="test")
        assert result.identical
        assert result.data_identical

    def test_data_identical_format_differs(self, tmp_path: Path) -> None:
        orig = tmp_path / "orig.dat"
        writ = tmp_path / "writ.dat"
        orig.write_text("C comment1\n1  2  3.0\n")
        writ.write_text("C different comment\n1 2 3.0\n")

        result = diff_iwfm_files(orig, writ, file_key="test")
        assert not result.identical
        assert result.data_identical

    def test_missing_original(self, tmp_path: Path) -> None:
        writ = tmp_path / "writ.dat"
        writ.write_text("1 2 3\n")

        result = diff_iwfm_files(tmp_path / "missing.dat", writ, file_key="test")
        assert not result.identical
        assert not result.data_identical

    def test_missing_written(self, tmp_path: Path) -> None:
        orig = tmp_path / "orig.dat"
        orig.write_text("1 2 3\n")

        result = diff_iwfm_files(orig, tmp_path / "missing.dat", file_key="test")
        assert not result.identical
        assert not result.data_identical

    def test_line_count_mismatch(self, tmp_path: Path) -> None:
        orig = tmp_path / "orig.dat"
        writ = tmp_path / "writ.dat"
        orig.write_text("1 2 3\n4 5 6\n")
        writ.write_text("1 2 3\n")

        result = diff_iwfm_files(orig, writ, file_key="test")
        assert not result.data_identical

    def test_max_diffs_truncation(self, tmp_path: Path) -> None:
        orig = tmp_path / "orig.dat"
        writ = tmp_path / "writ.dat"
        orig_lines = "".join(f"{i} 0 0\n" for i in range(20))
        writ_lines = "".join(f"{i} 99 99\n" for i in range(20))
        orig.write_text(orig_lines)
        writ.write_text(writ_lines)

        result = diff_iwfm_files(orig, writ, max_diffs=5)
        assert len(result.differences) <= 6  # max_diffs + potential truncation msg

    def test_file_key_stored(self, tmp_path: Path) -> None:
        f = tmp_path / "test.dat"
        f.write_text("1 2 3\n")
        result = diff_iwfm_files(f, f, file_key="my_key")
        assert result.file_key == "my_key"


# ---------------------------------------------------------------------------
# diff_all_files
# ---------------------------------------------------------------------------


class TestDiffAllFiles:
    def test_auto_discovery(self, tmp_path: Path) -> None:
        orig_dir = tmp_path / "original"
        writ_dir = tmp_path / "written"
        orig_dir.mkdir()
        writ_dir.mkdir()

        (orig_dir / "test.dat").write_text("1 2 3\n")
        (writ_dir / "test.dat").write_text("1 2 3\n")

        result = diff_all_files(orig_dir, writ_dir)
        assert result.files_compared >= 1

    def test_explicit_file_mapping(self, tmp_path: Path) -> None:
        orig_dir = tmp_path / "original"
        writ_dir = tmp_path / "written"
        orig_dir.mkdir()
        writ_dir.mkdir()

        orig_f = orig_dir / "a.dat"
        writ_f = writ_dir / "b.dat"
        orig_f.write_text("1 2 3\n")
        writ_f.write_text("1 2 3\n")

        mapping = {"mapped": (orig_f, writ_f)}
        result = diff_all_files(orig_dir, writ_dir, file_mapping=mapping)
        assert "mapped" in result.file_diffs

    def test_no_common_files(self, tmp_path: Path) -> None:
        orig_dir = tmp_path / "original"
        writ_dir = tmp_path / "written"
        orig_dir.mkdir()
        writ_dir.mkdir()

        (orig_dir / "a.dat").write_text("1\n")
        (writ_dir / "b.txt").write_text("2\n")

        result = diff_all_files(orig_dir, writ_dir)
        assert result.files_compared == 0

    def test_discovers_in_and_tab_files(self, tmp_path: Path) -> None:
        orig_dir = tmp_path / "original"
        writ_dir = tmp_path / "written"
        orig_dir.mkdir()
        writ_dir.mkdir()

        for ext in [".dat", ".in", ".tab"]:
            (orig_dir / f"test{ext}").write_text("1 2\n")
            (writ_dir / f"test{ext}").write_text("1 2\n")

        result = diff_all_files(orig_dir, writ_dir)
        assert result.files_compared >= 3


# ---------------------------------------------------------------------------
# InputDiffResult
# ---------------------------------------------------------------------------


class TestInputDiffResult:
    def test_empty_result(self) -> None:
        r = InputDiffResult()
        assert r.files_compared == 0
        assert r.files_identical == 0
        assert r.files_data_identical == 0

    def test_properties(self) -> None:
        diffs = {
            "a": FileDiffResult(file_key="a", identical=True, data_identical=True),
            "b": FileDiffResult(file_key="b", identical=False, data_identical=True),
            "c": FileDiffResult(file_key="c", identical=False, data_identical=False),
        }
        r = InputDiffResult(file_diffs=diffs)
        assert r.files_compared == 3
        assert r.files_identical == 1
        assert r.files_data_identical == 2

    def test_summary_format(self) -> None:
        diffs = {
            "a": FileDiffResult(file_key="a", identical=True, data_identical=True),
        }
        r = InputDiffResult(file_diffs=diffs)
        s = r.summary()
        assert isinstance(s, str)
        assert "1" in s  # Should contain at least the count


# ---------------------------------------------------------------------------
# FileDiffResult
# ---------------------------------------------------------------------------


class TestFileDiffResult:
    def test_defaults(self) -> None:
        r = FileDiffResult()
        assert r.file_key == ""
        assert not r.identical
        assert not r.data_identical
        assert r.n_data_lines_original == 0
        assert r.differences == []
