"""Tests for the unified IWFM line-reading module."""

from __future__ import annotations

import io

import pytest

from pyiwfm.core.exceptions import FileFormatError
from pyiwfm.io.iwfm_reader import (
    LineBuffer,
    is_comment_line,
    next_data_line,
    next_data_value,
    strip_inline_comment,
)

# ── is_comment_line ─────────────────────────────────────────────────


class TestIsCommentLine:
    def test_uppercase_c(self) -> None:
        assert is_comment_line("C this is a comment") is True

    def test_lowercase_c(self) -> None:
        assert is_comment_line("c this is a comment") is True

    def test_asterisk(self) -> None:
        assert is_comment_line("* this is a comment") is True

    def test_blank_line(self) -> None:
        assert is_comment_line("") is True
        assert is_comment_line("   ") is True
        assert is_comment_line("\n") is True

    def test_data_line(self) -> None:
        assert is_comment_line("1234  0.5  0.3") is False

    def test_windows_drive_letter(self) -> None:
        assert is_comment_line(r"C:\model\area.dss") is False
        assert is_comment_line(r"c:\model\area.dss") is False

    def test_hash_not_comment(self) -> None:
        assert is_comment_line("# this is NOT a comment in IWFM") is False

    def test_bang_not_comment(self) -> None:
        assert is_comment_line("! this is NOT a comment in IWFM") is False

    def test_leading_whitespace_with_data(self) -> None:
        assert is_comment_line("  100  200  300") is False

    def test_indented_c_not_comment(self) -> None:
        # Column-1 rule: indented 'C' is NOT a comment
        assert is_comment_line("  C indented") is False


# ── strip_inline_comment ────────────────────────────────────────────


class TestStripInlineComment:
    def test_basic_slash_comment(self) -> None:
        val, desc = strip_inline_comment("42 / NGROUP")
        assert val == "42"
        assert desc == "NGROUP"

    def test_no_comment(self) -> None:
        val, desc = strip_inline_comment("100  200  300")
        assert val == "100  200  300"
        assert desc == ""

    def test_slash_in_date_not_stripped(self) -> None:
        val, desc = strip_inline_comment("01/31/2000_24:00  1.5  2.5")
        assert val == "01/31/2000_24:00  1.5  2.5"
        assert desc == ""

    def test_date_with_trailing_comment(self) -> None:
        val, desc = strip_inline_comment("01/31/2000_24:00  1.5 / comment")
        assert val == "01/31/2000_24:00  1.5"
        assert desc == "comment"

    def test_dss_path_not_stripped(self) -> None:
        # DSS paths like /A/B/C/ have no whitespace before slashes
        val, desc = strip_inline_comment("/A/B/C/D/E/F/")
        assert val == "/A/B/C/D/E/F/"
        assert desc == ""

    def test_hash_not_treated_as_comment(self) -> None:
        val, desc = strip_inline_comment("42 # this stays")
        assert val == "42 # this stays"
        assert desc == ""

    def test_multiple_slash_comments(self) -> None:
        val, desc = strip_inline_comment("42 / first / second")
        assert val == "42"
        assert desc == "first / second"

    def test_slash_at_start_not_stripped(self) -> None:
        # '/' at position 0 has no preceding whitespace — not a comment
        val, desc = strip_inline_comment("/ all comment")
        assert val == "/ all comment"
        assert desc == ""


# ── next_data_value ─────────────────────────────────────────────────


class TestNextDataValue:
    def test_skips_comments(self) -> None:
        text = "C comment\n* another\n42 / NGROUP\n"
        f = io.StringIO(text)
        assert next_data_value(f) == "42"

    def test_strips_inline_comment(self) -> None:
        f = io.StringIO("  path/to/file.dat  / description\n")
        assert next_data_value(f) == "path/to/file.dat"

    def test_returns_empty_at_eof(self) -> None:
        f = io.StringIO("C only comments\n")
        assert next_data_value(f) == ""

    def test_line_counter(self) -> None:
        text = "C skip\nC skip\n42 / val\n"
        counter = [0]
        f = io.StringIO(text)
        next_data_value(f, line_counter=counter)
        assert counter[0] == 3

    def test_preserves_windows_path(self) -> None:
        f = io.StringIO(r"C:\model\area.dss" + "\n")
        assert next_data_value(f) == r"C:\model\area.dss"


# ── next_data_line ──────────────────────────────────────────────────


class TestNextDataLine:
    def test_returns_raw_line(self) -> None:
        text = "C comment\n1  2  3 / desc\n"
        f = io.StringIO(text)
        assert next_data_line(f) == "1  2  3 / desc"

    def test_does_not_strip_inline_comment(self) -> None:
        f = io.StringIO("100  200 / some note\n")
        result = next_data_line(f)
        assert "/ some note" in result

    def test_returns_empty_at_eof(self) -> None:
        f = io.StringIO("* comment only\n")
        assert next_data_line(f) == ""

    def test_line_counter(self) -> None:
        text = "* skip\n10 20 30\n"
        counter = [0]
        f = io.StringIO(text)
        next_data_line(f, line_counter=counter)
        assert counter[0] == 2


# ── LineBuffer ──────────────────────────────────────────────────────


class TestLineBuffer:
    def test_next_line(self) -> None:
        buf = LineBuffer(["a\n", "b\n", "c\n"])
        assert buf.next_line() == "a\n"
        assert buf.next_line() == "b\n"
        assert buf.next_line() == "c\n"
        assert buf.next_line() is None

    def test_pushback(self) -> None:
        buf = LineBuffer(["a\n", "b\n"])
        assert buf.next_line() == "a\n"
        buf.pushback()
        assert buf.next_line() == "a\n"

    def test_line_num(self) -> None:
        buf = LineBuffer(["a\n", "b\n"])
        assert buf.line_num == 0
        buf.next_line()
        assert buf.line_num == 1

    def test_next_data_skips_comments(self) -> None:
        lines = ["C comment\n", "* another\n", "42 / NGROUP\n"]
        buf = LineBuffer(lines)
        assert buf.next_data() == "42"

    def test_next_data_raises_on_eof(self) -> None:
        buf = LineBuffer(["C only comment\n"])
        with pytest.raises(FileFormatError):
            buf.next_data()

    def test_next_data_or_empty(self) -> None:
        lines = ["C skip\n", "42 / val\n"]
        buf = LineBuffer(lines)
        assert buf.next_data_or_empty() == "42"

    def test_next_data_or_empty_at_eof(self) -> None:
        buf = LineBuffer(["C comment\n"])
        assert buf.next_data_or_empty() == ""

    def test_next_raw_data(self) -> None:
        lines = ["C comment\n", "1  2  3 / desc\n"]
        buf = LineBuffer(lines)
        assert buf.next_raw_data() == "1  2  3 / desc"

    def test_next_raw_data_at_eof(self) -> None:
        buf = LineBuffer(["* comment\n"])
        assert buf.next_raw_data() == ""

    def test_pushback_at_zero(self) -> None:
        buf = LineBuffer(["a\n"])
        buf.pushback()  # Should not go negative
        assert buf.line_num == 0
        assert buf.next_line() == "a\n"
