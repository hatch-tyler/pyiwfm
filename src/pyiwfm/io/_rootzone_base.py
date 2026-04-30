"""Shared base class for IWFM v5+ rootzone sub-file readers.

The three v5+ rootzone variant readers
(:class:`~pyiwfm.io.rootzone_native.NativeRiparianReader`,
:class:`~pyiwfm.io.rootzone_nonponded.NonPondedCropReader`,
:class:`~pyiwfm.io.rootzone_urban.UrbanLandUseReader`) all parse a
positional-sequential text file with the same scaffolding shape:

1. Open the file, build a :class:`~pyiwfm.io.ascii.reader.LineBuffer`.
2. Read a sequence of scalar fields (factors, file paths, optional
   integers).
3. Read one or more tabular sections, each delimited either by an
   exact subregion-count or by a comment line.
4. Return a populated dataclass config.

In v1.x each reader had its own copies of ``_resolve`` (3×, byte-identical),
``_read_rows`` (3×, only differing in whether ``n_expected`` could be
overridden per call), and inline scalar-parsing patterns
(``try: float(val); except ValueError: raise FileFormatError(...)``).
v2.0 PR 6 extracts that scaffolding here.

Concrete readers inherit from :class:`_RootzoneReaderBase` and implement
their own :meth:`read` orchestrator that calls into the shared helpers.
The v4.x readers (in ``rootzone_v4x.py``) already use a separate
``_V4xReaderBase`` shipped before v2.0 — that base remains; this one
covers the v5+ readers.
"""

from __future__ import annotations

from pathlib import Path

from pyiwfm.core.exceptions import FileFormatError
from pyiwfm.io.ascii.reader import LineBuffer as _LineBuffer
from pyiwfm.io.ascii.reader import is_comment_line as _is_comment_line
from pyiwfm.io.ascii.reader import strip_inline_comment as _strip_comment


class _RootzoneReaderBase:
    """Shared scaffolding for v5+ rootzone variant readers.

    Subclasses set ``self._n_sub`` (and optionally ``self._n_elem``)
    in their constructor and implement a public ``read()`` method
    that orchestrates the per-variant scalar-then-tabular parse using
    the helpers below.

    Args:
        n_subregions: Number of subregions in the model. If provided,
            ``_read_rows`` reads exactly this many rows per section
            (skipping comments within); otherwise sections are
            delimited by comment lines.
        n_elements: Number of elements in the model. Some variants
            (urban) read per-element rows in addition to per-subregion
            rows.
    """

    def __init__(
        self,
        n_subregions: int | None = None,
        n_elements: int | None = None,
    ) -> None:
        self._n_sub = n_subregions
        self._n_elem = n_elements

    # ── path helpers ──────────────────────────────────────────────────

    @staticmethod
    def _resolve(base_dir: Path, filepath: str) -> Path:
        """Resolve a relative sub-file path against ``base_dir``."""
        p = Path(filepath.strip())
        if p.is_absolute():
            return p
        return base_dir / p

    # ── tabular-data helpers ──────────────────────────────────────────

    def _read_rows(
        self,
        buf: _LineBuffer,
        min_cols: int,
        n_expected: int | None = None,
    ) -> list[list[str]]:
        """Read tabular data rows from the buffer.

        If ``n_expected`` is provided (or ``self._n_sub`` is set when
        ``n_expected`` is None), reads exactly that many data rows
        (skipping comments within). Otherwise reads data rows until a
        comment line or a line with fewer than ``min_cols`` columns is
        encountered, pushing that line back so a subsequent section can
        consume it.

        Args:
            buf: Line buffer positioned at the start of a tabular section.
            min_cols: Minimum number of whitespace-separated columns
                expected on a data row. A row with fewer columns terminates
                the section.
            n_expected: Override for the expected row count. Defaults to
                ``self._n_sub`` (typical for per-subregion sections); pass
                an explicit value (e.g. ``self._n_elem``) for per-element
                sections in mixed readers like urban.

        Returns:
            List of split-value lists (one per row).
        """
        if n_expected is None:
            n_expected = self._n_sub

        rows: list[list[str]] = []
        started = False

        while True:
            raw = buf.next_line()
            if raw is None:
                break

            if _is_comment_line(raw):
                if n_expected is not None:
                    # Known count: skip comments within the section
                    continue
                if started:
                    # Unknown count: comment after data = section end
                    buf.pushback()
                    break
                continue

            value, _ = _strip_comment(raw)
            parts = value.split()

            if len(parts) < min_cols:
                buf.pushback()
                break

            rows.append(parts)
            started = True

            if n_expected is not None and len(rows) >= n_expected:
                break

        return rows

    # ── scalar-parsing helpers ────────────────────────────────────────

    @staticmethod
    def _parse_float(value: str, label: str, line_num: int) -> float:
        """Parse a string as a float, raising :class:`FileFormatError` on failure.

        Replaces the v1.x inline ``try: float(...); except ValueError:
        raise FileFormatError(...)`` blocks that appeared at every scalar
        read site (~6 places per reader × 3 readers).
        """
        try:
            return float(value)
        except ValueError as exc:
            raise FileFormatError(
                f"Invalid {label}: '{value}'",
                line_number=line_num,
            ) from exc

    @staticmethod
    def _parse_int(value: str, label: str, line_num: int) -> int:
        """Parse a string as an int, raising :class:`FileFormatError` on failure."""
        try:
            return int(value)
        except ValueError as exc:
            raise FileFormatError(
                f"Invalid {label}: '{value}'",
                line_number=line_num,
            ) from exc
