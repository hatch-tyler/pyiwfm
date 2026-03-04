"""Unit tests for the report page layout and ADA helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

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
from pyiwfm.visualization.report._table import (
    TableStyle,
    draw_stats_panel,
    draw_table,
)

# ---------------------------------------------------------------------------
# PageMargins
# ---------------------------------------------------------------------------


class TestPageMargins:
    """Tests for PageMargins dataclass."""

    def test_defaults(self) -> None:
        m = PageMargins()
        assert m.left == 0.85
        assert m.right == 0.65
        assert m.top == 0.90
        assert m.bottom == 0.75
        assert m.header == 0.50
        assert m.footer == 0.45

    def test_printable_width(self) -> None:
        m = PageMargins()
        expected = 8.5 - 0.85 - 0.65
        assert abs(m.printable_width - expected) < 1e-6

    def test_printable_height(self) -> None:
        m = PageMargins()
        expected = 11.0 - 0.90 - 0.75 - 0.50 - 0.45
        assert abs(m.printable_height - expected) < 1e-6

    def test_printable_fractions(self) -> None:
        m = PageMargins()
        assert 0 < m.printable_left < 1
        assert 0 < m.printable_bottom < 1
        assert 0 < m.printable_width_frac < 1
        assert 0 < m.printable_height_frac < 1

    def test_custom_margins(self) -> None:
        m = PageMargins(left=1.0, right=1.0, top=1.0, bottom=1.0)
        assert m.printable_width == 8.5 - 2.0


# ---------------------------------------------------------------------------
# ReportConfig
# ---------------------------------------------------------------------------


class TestReportConfig:
    """Tests for ReportConfig dataclass."""

    def test_defaults(self) -> None:
        cfg = ReportConfig()
        assert cfg.title == ""
        assert cfg.author == "pyiwfm"
        assert cfg.dpi == 300
        assert cfg.show_page_numbers is True

    def test_effective_date(self) -> None:
        from datetime import datetime

        cfg = ReportConfig()
        dt = cfg.effective_date()
        assert isinstance(dt, datetime)

    def test_effective_header(self) -> None:
        cfg = ReportConfig(title="My Report")
        assert cfg.effective_header() == "My Report"

        cfg2 = ReportConfig(title="My Report", header_text="Custom Header")
        assert cfg2.effective_header() == "Custom Header"

    def test_effective_footer_left(self) -> None:
        cfg = ReportConfig(footer_left="Custom Footer")
        assert cfg.effective_footer_left() == "Custom Footer"

        cfg2 = ReportConfig()
        footer = cfg2.effective_footer_left()
        assert len(footer) > 0  # Should be a date string


# ---------------------------------------------------------------------------
# ReportPage
# ---------------------------------------------------------------------------


class TestReportPage:
    """Tests for ReportPage coordinate translation and rendering."""

    def test_creation(self) -> None:
        cfg = ReportConfig(title="Test")
        page = ReportPage(cfg, page_number=1)
        assert page.page_number == 1
        assert page.fig is not None
        plt.close(page.fig)

    def test_to_fig_coords(self) -> None:
        cfg = ReportConfig()
        page = ReportPage(cfg, page_number=1)

        # Full printable area
        fl, fb, fw, fh = page._to_fig_coords(0, 0, 1, 1)
        m = cfg.margins
        assert abs(fl - m.printable_left) < 1e-6
        assert abs(fb - m.printable_bottom) < 1e-6
        assert abs(fw - m.printable_width_frac) < 1e-6
        assert abs(fh - m.printable_height_frac) < 1e-6
        plt.close(page.fig)

    def test_to_fig_coords_half(self) -> None:
        cfg = ReportConfig()
        page = ReportPage(cfg, page_number=1)

        # Half printable area
        fl, fb, fw, fh = page._to_fig_coords(0, 0, 0.5, 0.5)
        m = cfg.margins
        assert abs(fw - m.printable_width_frac * 0.5) < 1e-6
        assert abs(fh - m.printable_height_frac * 0.5) < 1e-6
        plt.close(page.fig)

    def test_add_axes(self) -> None:
        cfg = ReportConfig()
        page = ReportPage(cfg, page_number=1)
        ax = page.add_axes(0.1, 0.1, 0.8, 0.8)
        assert ax is not None
        plt.close(page.fig)

    def test_add_gridspec(self) -> None:
        cfg = ReportConfig()
        page = ReportPage(cfg, page_number=1)
        gs = page.add_gridspec(2, 2)
        assert gs is not None
        # Create a subplot to verify gridspec works
        ax = page.fig.add_subplot(gs[0, 0])
        assert ax is not None
        plt.close(page.fig)

    def test_header_footer_drawing(self) -> None:
        cfg = ReportConfig(title="Test Report", show_page_numbers=True)
        page = ReportPage(cfg, page_number=3, total_pages=10, section_name="Results")
        page._draw_header()
        page._draw_footer()
        # Check that text objects were added
        texts = [t.get_text() for t in page.fig.texts]
        assert any("Test Report" in t for t in texts)
        assert any("Results" in t for t in texts)
        assert any("Page 3 of 10" in t for t in texts)
        plt.close(page.fig)

    def test_header_without_section(self) -> None:
        cfg = ReportConfig(title="Test")
        page = ReportPage(cfg, page_number=1, section_name="")
        page._draw_header()
        texts = [t.get_text() for t in page.fig.texts]
        assert any("Test" in t for t in texts)
        plt.close(page.fig)

    def test_footer_without_total_pages(self) -> None:
        cfg = ReportConfig(show_page_numbers=True)
        page = ReportPage(cfg, page_number=5, total_pages=None)
        page._draw_footer()
        texts = [t.get_text() for t in page.fig.texts]
        assert any("Page 5" in t for t in texts)
        plt.close(page.fig)

    def test_footer_no_page_numbers(self) -> None:
        cfg = ReportConfig(show_page_numbers=False)
        page = ReportPage(cfg, page_number=1)
        page._draw_footer()
        texts = [t.get_text() for t in page.fig.texts]
        assert not any("Page" in t for t in texts)
        plt.close(page.fig)


# ---------------------------------------------------------------------------
# ADA helpers
# ---------------------------------------------------------------------------


class TestLuminance:
    """Tests for WCAG luminance calculation."""

    def test_black(self) -> None:
        assert abs(luminance("#000000")) < 0.001

    def test_white(self) -> None:
        assert abs(luminance("#FFFFFF") - 1.0) < 0.001

    def test_mid_gray(self) -> None:
        lum = luminance("#808080")
        assert 0.1 < lum < 0.5


class TestContrastRatio:
    """Tests for WCAG contrast ratio."""

    def test_black_on_white(self) -> None:
        ratio = contrast_ratio("#000000", "#FFFFFF")
        assert ratio > 20.0  # Should be 21:1

    def test_same_color(self) -> None:
        ratio = contrast_ratio("#FF0000", "#FF0000")
        assert abs(ratio - 1.0) < 0.01

    def test_symmetry(self) -> None:
        r1 = contrast_ratio("#0072B2", "#FFFFFF")
        r2 = contrast_ratio("#FFFFFF", "#0072B2")
        assert abs(r1 - r2) < 0.01


class TestCheckContrast:
    """Tests for check_contrast function."""

    def test_black_on_white_passes(self) -> None:
        assert check_contrast("#000000", "#FFFFFF") is True

    def test_light_on_white_fails(self) -> None:
        assert check_contrast("#EEEEEE", "#FFFFFF") is False

    def test_custom_ratio(self) -> None:
        assert check_contrast("#000000", "#FFFFFF", min_ratio=21.0) is True


class TestEnsureContrast:
    """Tests for ensure_contrast function."""

    def test_already_compliant(self) -> None:
        result = ensure_contrast("#000000", "#FFFFFF")
        assert check_contrast(result, "#FFFFFF")

    def test_light_color_darkened(self) -> None:
        result = ensure_contrast("#CCCCCC", "#FFFFFF")
        assert check_contrast(result, "#FFFFFF")

    def test_dark_color_on_dark_bg(self) -> None:
        result = ensure_contrast("#333333", "#000000")
        assert check_contrast(result, "#000000")


class TestValidateFontSizes:
    """Tests for validate_font_sizes function."""

    def test_no_warnings_for_large_text(self) -> None:
        fig, ax = plt.subplots()
        ax.set_title("Big Title", fontsize=14)
        ax.set_xlabel("X Label", fontsize=10)
        warnings = validate_font_sizes(fig)
        assert len(warnings) == 0
        plt.close(fig)

    def test_warning_for_small_text(self) -> None:
        fig = plt.figure()
        fig.text(0.5, 0.5, "Tiny text", fontsize=5)
        warnings = validate_font_sizes(fig)
        assert len(warnings) > 0
        assert "5.0pt" in warnings[0]
        plt.close(fig)

    def test_empty_text_ignored(self) -> None:
        fig = plt.figure()
        fig.text(0.5, 0.5, "", fontsize=4)
        warnings = validate_font_sizes(fig)
        assert len(warnings) == 0
        plt.close(fig)


class TestHatchPatterns:
    """Tests for HATCH_PATTERNS constant."""

    def test_length(self) -> None:
        assert len(HATCH_PATTERNS) == 8

    def test_types(self) -> None:
        for p in HATCH_PATTERNS:
            assert isinstance(p, str)
            assert len(p) > 0


class TestApplyHatches:
    """Tests for apply_hatches function."""

    def test_applies_patterns(self) -> None:
        fig, ax = plt.subplots()
        bars = ax.bar([1, 2, 3], [4, 5, 6])
        patches = list(bars)
        apply_hatches(patches)
        for patch in patches:
            assert patch.get_hatch() is not None
        plt.close(fig)

    def test_wraps_around(self) -> None:
        fig, ax = plt.subplots()
        bars = ax.bar(range(10), range(10))
        patches = list(bars)
        apply_hatches(patches)
        # 10 patches with 8 patterns: should wrap
        assert patches[8].get_hatch() == patches[0].get_hatch()
        plt.close(fig)

    def test_start_index(self) -> None:
        fig, ax = plt.subplots()
        bars = ax.bar([1, 2], [3, 4])
        patches = list(bars)
        apply_hatches(patches, start_index=2)
        assert patches[0].get_hatch() == HATCH_PATTERNS[2]
        plt.close(fig)


class TestAddAltText:
    """Tests for add_alt_text function."""

    def test_sets_label(self) -> None:
        fig = plt.figure()
        add_alt_text(fig, "A scatter plot of observed vs simulated values")
        assert fig.get_label() == "A scatter plot of observed vs simulated values"
        plt.close(fig)


# ---------------------------------------------------------------------------
# Table renderer
# ---------------------------------------------------------------------------


class TestDrawTable:
    """Tests for draw_table function."""

    def test_basic_table(self) -> None:
        fig, ax = plt.subplots()
        data = [["A", "1.23"], ["B", "4.56"]]
        draw_table(ax, data, col_labels=["Name", "Value"])
        # Axes should be off
        assert not ax.axison
        # Check text was added
        texts = [t.get_text() for t in ax.texts]
        assert "Name" in texts
        assert "Value" in texts
        assert "A" in texts
        assert "1.23" in texts
        plt.close(fig)

    def test_empty_data(self) -> None:
        fig, ax = plt.subplots()
        draw_table(ax, [])
        assert not ax.axison
        plt.close(fig)

    def test_no_labels(self) -> None:
        fig, ax = plt.subplots()
        data = [["X", "Y"]]
        draw_table(ax, data)
        texts = [t.get_text() for t in ax.texts]
        assert "X" in texts
        plt.close(fig)

    def test_summary_row(self) -> None:
        fig, ax = plt.subplots()
        data = [["A", "1"], ["B", "2"], ["Total", "3"]]
        draw_table(ax, data, col_labels=["Name", "Val"], summary_row_start=2)
        plt.close(fig)

    def test_zebra_shading(self) -> None:
        fig, ax = plt.subplots()
        style = TableStyle(zebra_shade="#f9f9f9")
        data = [["A", "1"], ["B", "2"], ["C", "3"]]
        draw_table(ax, data, style=style)
        plt.close(fig)

    def test_figure_ref(self) -> None:
        fig, ax = plt.subplots()
        data = [["A", "1"]]
        draw_table(ax, data, figure_ref="Table 1")
        texts = [t.get_text() for t in ax.texts]
        assert "Table 1" in texts
        plt.close(fig)

    def test_custom_widths_and_alignments(self) -> None:
        fig, ax = plt.subplots()
        data = [["Name", "123", "456"]]
        draw_table(
            ax,
            data,
            col_labels=["Col1", "Col2", "Col3"],
            col_widths=[0.5, 0.25, 0.25],
            col_alignments=["left", "center", "right"],
        )
        plt.close(fig)


class TestDrawStatsPanel:
    """Tests for draw_stats_panel function."""

    def test_basic_panel(self) -> None:
        fig, ax = plt.subplots()
        stats = {"RMSE": 12.34, "NSE": 0.876, "N": 100}
        draw_stats_panel(ax, stats, title="Statistics")
        texts = [t.get_text() for t in ax.texts]
        assert "Statistics" in texts
        assert "RMSE" in texts
        plt.close(fig)

    def test_empty_stats(self) -> None:
        fig, ax = plt.subplots()
        draw_stats_panel(ax, {})
        plt.close(fig)

    def test_string_values(self) -> None:
        fig, ax = plt.subplots()
        stats = {"Rating": "Good", "Count": "42"}
        draw_stats_panel(ax, stats)
        texts = [t.get_text() for t in ax.texts]
        assert "Good" in texts
        plt.close(fig)


class TestTableStyle:
    """Tests for TableStyle dataclass."""

    def test_defaults(self) -> None:
        s = TableStyle()
        assert s.fontsize == 8.0
        assert s.header_fontweight == "bold"
        assert s.show_header_rule is True
        assert s.show_footer_rule is True
        assert s.zebra_shade is None

    def test_custom(self) -> None:
        s = TableStyle(fontsize=10.0, zebra_shade="#f0f0f0")
        assert s.fontsize == 10.0
        assert s.zebra_shade == "#f0f0f0"
