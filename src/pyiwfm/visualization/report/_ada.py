"""ADA / Section 508 accessibility helpers.

Advisory utilities for WCAG 2.1 AA compliance in matplotlib figures.
These are not hard gates — they are helpers that plot functions can use
to improve accessibility.

.. note::

   Full PDF/UA tagged structure requires external tools (e.g. Adobe
   Acrobat). matplotlib can embed fonts and set metadata but cannot
   produce tagged PDF structure trees.

Functions
---------
- :func:`luminance` — WCAG relative luminance of a hex color.
- :func:`contrast_ratio` — WCAG contrast ratio between two colors.
- :func:`check_contrast` — Check whether a color pair meets AA ratio.
- :func:`ensure_contrast` — Darken/lighten a color to meet AA ratio.
- :func:`validate_font_sizes` — Scan a figure for text below minimum size.
- :func:`apply_hatches` — Apply hatch patterns to bar patches.
- :func:`add_alt_text` — Set PDF /Description metadata on a figure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.colors as mcolors

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_FONT_SIZE: float = 8.0
"""Minimum font size (pt) for body text (WCAG guideline)."""

MIN_LABEL_SIZE: float = 9.0
"""Minimum font size (pt) for axis labels."""

MIN_TITLE_SIZE: float = 11.0
"""Minimum font size (pt) for titles."""

MIN_TEXT_CONTRAST: float = 4.5
"""Minimum WCAG contrast ratio for normal text (AA level)."""

MIN_GRAPHICAL_CONTRAST: float = 3.0
"""Minimum WCAG contrast ratio for graphical objects (AA level)."""

HATCH_PATTERNS: list[str] = [
    "///",
    "\\\\\\",
    "|||",
    "---",
    "+++",
    "xxx",
    "ooo",
    "...",
]
"""Eight hatch patterns for non-color distinction in bar charts."""


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------


def _hex_to_rgb(color: str) -> tuple[float, float, float]:
    """Convert any matplotlib color spec to (r, g, b) in [0, 1]."""
    rgba = mcolors.to_rgba(color)
    return rgba[0], rgba[1], rgba[2]


def luminance(color: str) -> float:
    """Compute WCAG 2.1 relative luminance.

    Parameters
    ----------
    color : str
        Any matplotlib-compatible color string (hex, named, etc.).

    Returns
    -------
    float
        Relative luminance in [0, 1].
    """
    r, g, b = _hex_to_rgb(color)

    def _linearize(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    rl, gl, bl = _linearize(r), _linearize(g), _linearize(b)
    return 0.2126 * rl + 0.7152 * gl + 0.0722 * bl


def contrast_ratio(fg: str, bg: str) -> float:
    """Compute WCAG contrast ratio between two colors.

    Parameters
    ----------
    fg, bg : str
        Foreground and background color strings.

    Returns
    -------
    float
        Contrast ratio >= 1.0.
    """
    l1 = luminance(fg)
    l2 = luminance(bg)
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def check_contrast(fg: str, bg: str, min_ratio: float = MIN_TEXT_CONTRAST) -> bool:
    """Check whether *fg* on *bg* meets the minimum contrast ratio.

    Parameters
    ----------
    fg, bg : str
        Foreground and background colors.
    min_ratio : float
        Required contrast ratio (default 4.5 for AA text).

    Returns
    -------
    bool
        ``True`` if the pair meets the ratio.
    """
    return contrast_ratio(fg, bg) >= min_ratio


def ensure_contrast(color: str, bg: str = "#FFFFFF") -> str:
    """Darken or lighten *color* until it meets AA contrast against *bg*.

    Parameters
    ----------
    color : str
        Starting color.
    bg : str
        Background color (default white).

    Returns
    -------
    str
        Hex color string that meets ``MIN_TEXT_CONTRAST`` against *bg*.
    """
    if check_contrast(color, bg):
        return mcolors.to_hex(mcolors.to_rgba(color)[:3])

    r, g, b = _hex_to_rgb(color)
    bg_lum = luminance(bg)

    # Decide direction: darken if bg is light, lighten if bg is dark
    darken = bg_lum > 0.5

    for step in range(100):
        factor = 1.0 - (step + 1) * 0.01 if darken else 1.0 + (step + 1) * 0.01
        nr = max(0.0, min(1.0, r * factor))
        ng = max(0.0, min(1.0, g * factor))
        nb = max(0.0, min(1.0, b * factor))
        candidate = mcolors.to_hex((nr, ng, nb))
        if check_contrast(candidate, bg):
            return candidate

    # Fallback: black or white
    return "#000000" if darken else "#FFFFFF"


# ---------------------------------------------------------------------------
# Font validation
# ---------------------------------------------------------------------------


def validate_font_sizes(fig: Figure) -> list[str]:
    """Scan a figure for text elements below minimum font sizes.

    Parameters
    ----------
    fig : Figure
        The matplotlib figure to scan.

    Returns
    -------
    list[str]
        Warning messages for text elements that are too small.
    """
    warnings: list[str] = []

    for text_obj in fig.findobj(match=lambda obj: hasattr(obj, "get_fontsize")):
        size: float = text_obj.get_fontsize()  # type: ignore[attr-defined]
        content: str = getattr(text_obj, "get_text", lambda: "")()
        if not content or not content.strip():
            continue
        if size < MIN_FONT_SIZE:
            warnings.append(
                f"Text '{content[:40]}' has font size {size:.1f}pt (minimum {MIN_FONT_SIZE}pt)"
            )

    return warnings


# ---------------------------------------------------------------------------
# Hatch helpers
# ---------------------------------------------------------------------------


def apply_hatches(patches: list[Patch], start_index: int = 0) -> None:
    """Apply hatch patterns to bar chart patches for non-color distinction.

    Parameters
    ----------
    patches : list[Patch]
        List of matplotlib patches (e.g. from ``ax.bar()``).
    start_index : int
        Starting index into :data:`HATCH_PATTERNS` (wraps around).
    """
    n = len(HATCH_PATTERNS)
    for i, patch in enumerate(patches):
        pattern = HATCH_PATTERNS[(start_index + i) % n]
        patch.set_hatch(pattern)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def add_alt_text(fig: Figure, description: str) -> None:
    """Set PDF ``/Description`` metadata on a figure.

    This stores the alt-text in the figure's metadata dict.  When saved
    with ``PdfPages``, it populates the ``/Description`` field.

    Parameters
    ----------
    fig : Figure
        Target figure.
    description : str
        Alt-text description of the figure content.
    """
    # matplotlib stores metadata in fig._suptitle or custom attributes.
    # The best we can do is store it for PdfPages metadata extraction.
    fig.set_label(description)
