"""Publication-quality USGS-style report framework.

Provides reusable building blocks for multi-page PDF reports with
consistent page layout, headers/footers, ADA-compliant defaults,
and minimalist Tufte-style tables.

Public API
----------
Layout & Builder:
    :class:`ReportBuilder` — Orchestrator for multi-page PDFs.
    :class:`ReportConfig` — Report-level settings.
    :class:`ReportPage` — Single page with printable-area coordinates.
    :class:`PageMargins` — Margin/header/footer dimensions.

Tables:
    :class:`TableStyle` — Table visual configuration.
    :func:`draw_table` — Minimalist table renderer.
    :func:`draw_stats_panel` — Two-column statistics panel.

Title Page:
    :func:`draw_title_page` — Title page with optional logo.

ADA / Accessibility:
    :func:`check_contrast` — WCAG AA contrast check.
    :func:`ensure_contrast` — Adjust color for contrast.
    :func:`validate_font_sizes` — Scan for text below minimum size.
    :data:`HATCH_PATTERNS` — Hatch patterns for non-color distinction.
    :func:`apply_hatches` — Apply hatch patterns to patches.
    :func:`add_alt_text` — Set PDF description metadata.
"""

from __future__ import annotations

from pyiwfm.visualization.report._ada import (
    HATCH_PATTERNS,
    add_alt_text,
    apply_hatches,
    check_contrast,
    contrast_ratio,
    ensure_contrast,
    luminance,
    validate_font_sizes,
)
from pyiwfm.visualization.report._page_layout import (
    PageMargins,
    ReportConfig,
    ReportPage,
)
from pyiwfm.visualization.report._report_builder import ReportBuilder
from pyiwfm.visualization.report._table import (
    TableStyle,
    draw_stats_panel,
    draw_table,
)
from pyiwfm.visualization.report._title_page import draw_title_page

__all__ = [
    # Layout & Builder
    "ReportBuilder",
    "ReportConfig",
    "ReportPage",
    "PageMargins",
    # Tables
    "TableStyle",
    "draw_table",
    "draw_stats_panel",
    # Title page
    "draw_title_page",
    # ADA / Accessibility
    "luminance",
    "contrast_ratio",
    "check_contrast",
    "ensure_contrast",
    "validate_font_sizes",
    "HATCH_PATTERNS",
    "apply_hatches",
    "add_alt_text",
]
