"""Title page generator for multi-page reports.

Renders a centered title page with optional logo, title, subtitle,
horizontal rule, date/author, model info, and summary statistics.
The title page skips the normal header/footer.

Functions
---------
- :func:`draw_title_page` — Render the title page on a :class:`ReportPage`.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from pyiwfm.visualization.report._page_layout import ReportConfig, ReportPage

_PAGE_WIDTH = 8.5
_PAGE_HEIGHT = 11.0


def draw_title_page(
    page: ReportPage,
    config: ReportConfig,
    summary_stats: dict[str, Any] | None = None,
    model_info: dict[str, Any] | None = None,
) -> None:
    """Render a title page.

    Layout (top to bottom):

    1. Optional logo (centered, max 1.5" tall)
    2. Title (18pt bold, centered)
    3. Subtitle (12pt, centered)
    4. Horizontal rule
    5. Date and author
    6. Model info block
    7. Summary statistics block

    Parameters
    ----------
    page : ReportPage
        The page to render on (title page skips header/footer).
    config : ReportConfig
        Report configuration with title, subtitle, etc.
    summary_stats : dict[str, Any] | None
        Optional summary statistics (e.g. ``{"GW RMSE (ft)": 12.3}``).
    model_info : dict[str, Any] | None
        Optional model info (e.g. ``{"Nodes": 30000, "Elements": 20000}``).
    """
    fig = page.fig
    y_cursor = 0.82  # Start from upper portion of page

    # 1. Logo
    if config.logo_path is not None and config.logo_path.exists():
        try:
            logo_img = plt.imread(str(config.logo_path))
            # Max 1.5 inches tall, centered
            logo_height_frac = 1.5 / _PAGE_HEIGHT
            logo_ax = fig.add_axes((0.3, y_cursor - logo_height_frac, 0.4, logo_height_frac))
            logo_ax.imshow(logo_img, aspect="equal")
            logo_ax.set_axis_off()
            y_cursor -= logo_height_frac + 0.03
        except Exception:
            pass  # Skip logo on read error

    # 2. Title
    if config.title:
        fig.text(
            0.5,
            y_cursor,
            config.title,
            fontsize=18,
            fontweight="bold",
            ha="center",
            va="top",
        )
        y_cursor -= 0.05

    # 3. Subtitle
    if config.subtitle:
        fig.text(
            0.5,
            y_cursor,
            config.subtitle,
            fontsize=12,
            ha="center",
            va="top",
            color="#444444",
        )
        y_cursor -= 0.04

    # 4. Horizontal rule
    y_cursor -= 0.01
    from matplotlib.lines import Line2D

    line = Line2D(
        [0.2, 0.8],
        [y_cursor, y_cursor],
        transform=fig.transFigure,
        color="#999999",
        linewidth=0.8,
        clip_on=False,
    )
    fig.add_artist(line)
    y_cursor -= 0.03

    # 5. Date and author
    date_str = config.effective_date().strftime("%B %d, %Y")
    fig.text(
        0.5,
        y_cursor,
        f"{config.author}  |  {date_str}",
        fontsize=10,
        ha="center",
        va="top",
        color="#666666",
    )
    y_cursor -= 0.05

    # 6. Model info block
    if model_info:
        fig.text(
            0.5,
            y_cursor,
            "Model Summary",
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="top",
        )
        y_cursor -= 0.03

        for key, val in model_info.items():
            fig.text(
                0.35,
                y_cursor,
                f"{key}:",
                fontsize=9,
                ha="right",
                va="top",
                color="#333333",
            )
            fig.text(
                0.37,
                y_cursor,
                str(val),
                fontsize=9,
                ha="left",
                va="top",
                color="#333333",
            )
            y_cursor -= 0.022
        y_cursor -= 0.02

    # 7. Summary statistics block
    if summary_stats:
        fig.text(
            0.5,
            y_cursor,
            "Summary Statistics",
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="top",
        )
        y_cursor -= 0.03

        for key, val in summary_stats.items():
            if isinstance(val, float):
                val_str = f"{val:.3f}"
            else:
                val_str = str(val)
            fig.text(
                0.35,
                y_cursor,
                f"{key}:",
                fontsize=9,
                ha="right",
                va="top",
                color="#333333",
            )
            fig.text(
                0.37,
                y_cursor,
                val_str,
                fontsize=9,
                ha="left",
                va="top",
                color="#333333",
            )
            y_cursor -= 0.022


def save_title_page(
    pdf: PdfPages,
    config: ReportConfig,
    summary_stats: dict[str, Any] | None = None,
    model_info: dict[str, Any] | None = None,
) -> None:
    """Create and save a title page directly to a PdfPages.

    This is a convenience wrapper that creates a :class:`ReportPage`,
    draws the title page content, and saves it without header/footer.

    Parameters
    ----------
    pdf : PdfPages
        Open PDF handle.
    config : ReportConfig
        Report configuration.
    summary_stats, model_info : dict | None
        Passed to :func:`draw_title_page`.
    """
    page = ReportPage(config, page_number=0, section_name="")
    draw_title_page(page, config, summary_stats, model_info)
    # Title page: save directly without header/footer
    pdf.savefig(page.fig, dpi=config.dpi)
    plt.close(page.fig)
