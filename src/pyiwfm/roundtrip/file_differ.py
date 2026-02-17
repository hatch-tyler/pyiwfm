"""IWFM input file comparison with comment stripping and float tolerance."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pyiwfm.io.iwfm_reader import (
    is_comment_line,
    strip_inline_comment,
)


@dataclass
class FileDiffResult:
    """Result of comparing a single IWFM file pair.

    Attributes
    ----------
    file_key : str
        Identifier for this file (e.g. 'preprocessor_main').
    original : Path
        Path to the original file.
    written : Path
        Path to the written file.
    identical : bool
        True if files are byte-identical.
    data_identical : bool
        True if data content matches (ignoring comments/formatting).
    n_data_lines_original : int
        Number of data lines in the original file.
    n_data_lines_written : int
        Number of data lines in the written file.
    differences : list[str]
        Human-readable list of first N differences found.
    """

    file_key: str = ""
    original: Path = field(default_factory=lambda: Path("."))
    written: Path = field(default_factory=lambda: Path("."))
    identical: bool = False
    data_identical: bool = False
    n_data_lines_original: int = 0
    n_data_lines_written: int = 0
    differences: list[str] = field(default_factory=list)


@dataclass
class InputDiffResult:
    """Aggregate result of comparing all input files.

    Attributes
    ----------
    file_diffs : dict[str, FileDiffResult]
        Per-file diff results.
    """

    file_diffs: dict[str, FileDiffResult] = field(default_factory=dict)

    @property
    def files_compared(self) -> int:
        """Number of files compared."""
        return len(self.file_diffs)

    @property
    def files_identical(self) -> int:
        """Number of byte-identical files."""
        return sum(1 for d in self.file_diffs.values() if d.identical)

    @property
    def files_data_identical(self) -> int:
        """Number of data-identical files (ignoring comments)."""
        return sum(1 for d in self.file_diffs.values() if d.data_identical)

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Input File Comparison: {self.files_compared} files",
            f"  Byte-identical: {self.files_identical}",
            f"  Data-identical: {self.files_data_identical}",
            f"  With differences: {self.files_compared - self.files_data_identical}",
        ]
        for key, diff in self.file_diffs.items():
            status = "MATCH" if diff.data_identical else "DIFF"
            lines.append(f"  [{status}] {key}")
            for d in diff.differences[:3]:
                lines.append(f"    {d}")
        return "\n".join(lines)


def _extract_data_lines(filepath: Path) -> list[str]:
    """Extract data-only lines from an IWFM file.

    Strips full-line comments, inline comments, and normalizes whitespace.

    Parameters
    ----------
    filepath : Path
        Path to the IWFM file.

    Returns
    -------
    list[str]
        List of normalized data lines.
    """
    lines: list[str] = []
    text = filepath.read_text(errors="replace")

    for raw_line in text.splitlines():
        if is_comment_line(raw_line):
            continue
        value, _ = strip_inline_comment(raw_line)
        stripped = value.strip()
        if stripped:
            lines.append(stripped)

    return lines


def _float_equal(a: str, b: str, rtol: float = 1e-6) -> bool:
    """Check if two tokens represent the same float value.

    Parameters
    ----------
    a : str
        First token.
    b : str
        Second token.
    rtol : float
        Relative tolerance.

    Returns
    -------
    bool
        True if both parse as floats within tolerance.
    """
    try:
        fa = float(a)
        fb = float(b)
    except ValueError:
        return False

    if fa == fb:
        return True
    if fa == 0.0 and fb == 0.0:
        return True
    denom = max(abs(fa), abs(fb))
    if denom == 0:
        return True
    return abs(fa - fb) / denom <= rtol


def _lines_match(line_a: str, line_b: str) -> bool:
    """Compare two data lines with float tolerance.

    Splits each line into tokens, compares string-equal first,
    then tries float comparison for mismatches.

    Parameters
    ----------
    line_a : str
        First data line.
    line_b : str
        Second data line.

    Returns
    -------
    bool
        True if lines match within tolerance.
    """
    tokens_a = line_a.split()
    tokens_b = line_b.split()

    if len(tokens_a) != len(tokens_b):
        return False

    for ta, tb in zip(tokens_a, tokens_b, strict=True):
        if ta == tb:
            continue
        if not _float_equal(ta, tb):
            return False

    return True


def diff_iwfm_files(
    original: Path,
    written: Path,
    file_key: str = "",
    max_diffs: int = 10,
) -> FileDiffResult:
    """Compare an original IWFM file against a written copy.

    Strips comments and compares data content with float tolerance
    for formatting differences.

    Parameters
    ----------
    original : Path
        Path to the original file.
    written : Path
        Path to the written file.
    file_key : str
        Identifier for this file pair.
    max_diffs : int
        Maximum number of differences to report.

    Returns
    -------
    FileDiffResult
        Comparison result.
    """
    result = FileDiffResult(file_key=file_key, original=original, written=written)

    if not original.exists():
        result.differences.append(f"Original file missing: {original}")
        return result

    if not written.exists():
        result.differences.append(f"Written file missing: {written}")
        return result

    # Check byte-identical first
    orig_bytes = original.read_bytes()
    written_bytes = written.read_bytes()
    if orig_bytes == written_bytes:
        result.identical = True
        result.data_identical = True
        return result

    # Extract and compare data lines
    orig_lines = _extract_data_lines(original)
    written_lines = _extract_data_lines(written)
    result.n_data_lines_original = len(orig_lines)
    result.n_data_lines_written = len(written_lines)

    if len(orig_lines) != len(written_lines):
        result.differences.append(f"Line count differs: {len(orig_lines)} vs {len(written_lines)}")

    # Compare line by line
    n_diffs = 0
    max_lines = max(len(orig_lines), len(written_lines))
    all_match = True

    for i in range(max_lines):
        if i >= len(orig_lines):
            all_match = False
            if n_diffs < max_diffs:
                result.differences.append(
                    f"Line {i + 1}: extra in written: {written_lines[i][:80]}"
                )
                n_diffs += 1
        elif i >= len(written_lines):
            all_match = False
            if n_diffs < max_diffs:
                result.differences.append(f"Line {i + 1}: missing in written: {orig_lines[i][:80]}")
                n_diffs += 1
        elif not _lines_match(orig_lines[i], written_lines[i]):
            all_match = False
            if n_diffs < max_diffs:
                result.differences.append(
                    f"Line {i + 1}:\n"
                    f"  orig:    {orig_lines[i][:100]}\n"
                    f"  written: {written_lines[i][:100]}"
                )
                n_diffs += 1

    result.data_identical = all_match
    return result


def diff_all_files(
    original_dir: Path,
    written_dir: Path,
    file_mapping: dict[str, tuple[Path, Path]] | None = None,
    max_diffs_per_file: int = 10,
) -> InputDiffResult:
    """Compare all IWFM input files between two directories.

    If file_mapping is not provided, discovers files by matching
    relative paths between the two directories.

    Parameters
    ----------
    original_dir : Path
        Directory with original model files.
    written_dir : Path
        Directory with written model files.
    file_mapping : dict[str, tuple[Path, Path]] | None
        Explicit mapping of file_key -> (original, written) paths.
    max_diffs_per_file : int
        Maximum differences to report per file.

    Returns
    -------
    InputDiffResult
        Aggregate comparison result.
    """
    result = InputDiffResult()

    if file_mapping:
        for key, (orig, writ) in file_mapping.items():
            result.file_diffs[key] = diff_iwfm_files(orig, writ, key, max_diffs_per_file)
        return result

    # Auto-discover: find common files by relative path
    orig_files: dict[str, Path] = {}
    for f in original_dir.rglob("*"):
        if f.is_file() and f.suffix in (".dat", ".in", ".tab"):
            rel = f.relative_to(original_dir)
            orig_files[str(rel)] = f

    for rel_str, orig_path in sorted(orig_files.items()):
        writ_path = written_dir / rel_str
        if writ_path.exists():
            result.file_diffs[rel_str] = diff_iwfm_files(
                orig_path, writ_path, rel_str, max_diffs_per_file
            )

    return result
