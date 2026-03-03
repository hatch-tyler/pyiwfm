"""Page layout abstractions for multi-page PDF reports.

Provides :class:`ReportPage`, which wraps a matplotlib ``Figure`` and offers
a fractional coordinate system within the printable area (0,0 = bottom-left
of printable zone, 1,1 = top-right).  Content authors never need to think
about margins, headers, or footers.

Classes
-------
- :class:`PageMargins` — Margin/header/footer dimensions (inches).
- :class:`ReportConfig` — Report-level settings (title, author, DPI, etc.).
- :class:`ReportPage` — Single letter-size page with header, footer, and
  coordinate helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

# US Letter dimensions
_PAGE_WIDTH = 8.5
_PAGE_HEIGHT = 11.0


@dataclass
class PageMargins:
    """Margin dimensions in inches.

    Attributes
    ----------
    left : float
        Distance from left page edge to printable area.
    right : float
        Distance from right page edge to printable area.
    top : float
        Distance from top of header zone to top of printable area.
    bottom : float
        Distance from bottom of footer zone to bottom of printable area.
    header : float
        Height of the header zone (above printable area).
    footer : float
        Height of the footer zone (below printable area).
    """

    left: float = 0.85
    right: float = 0.65
    top: float = 0.90
    bottom: float = 0.75
    header: float = 0.50
    footer: float = 0.45

    @property
    def printable_width(self) -> float:
        """Width of printable area in inches."""
        return _PAGE_WIDTH - self.left - self.right

    @property
    def printable_height(self) -> float:
        """Height of printable area in inches."""
        return _PAGE_HEIGHT - self.top - self.bottom - self.header - self.footer

    @property
    def printable_left(self) -> float:
        """Left edge of printable area as fraction of page width."""
        return self.left / _PAGE_WIDTH

    @property
    def printable_bottom(self) -> float:
        """Bottom edge of printable area as fraction of page height."""
        return (self.bottom + self.footer) / _PAGE_HEIGHT

    @property
    def printable_width_frac(self) -> float:
        """Width of printable area as fraction of page width."""
        return self.printable_width / _PAGE_WIDTH

    @property
    def printable_height_frac(self) -> float:
        """Height of printable area as fraction of page height."""
        return self.printable_height / _PAGE_HEIGHT


@dataclass
class ReportConfig:
    """Report-level configuration.

    Attributes
    ----------
    title : str
        Main report title.
    subtitle : str
        Optional subtitle shown below the title.
    author : str
        Author name for PDF metadata.
    date : datetime | None
        Report date; defaults to ``datetime.now()`` at build time.
    logo_path : Path | None
        Optional path to a logo image (PNG/JPG) for the title page.
    margins : PageMargins
        Page margin configuration.
    dpi : int
        Output resolution in dots per inch.
    header_text : str | None
        Override for left header text (defaults to *title*).
    footer_left : str | None
        Left footer text (defaults to formatted *date*).
    show_page_numbers : bool
        Whether to render "Page X of Y" in the footer.
    units_length : str
        Length unit label (e.g. ``"ft"``).
    units_volume : str
        Volume unit label (e.g. ``"AF"``).
    """

    title: str = ""
    subtitle: str = ""
    author: str = "pyiwfm"
    date: datetime | None = None
    logo_path: Path | None = None
    margins: PageMargins = field(default_factory=PageMargins)
    dpi: int = 300
    header_text: str | None = None
    footer_left: str | None = None
    show_page_numbers: bool = True
    units_length: str = "ft"
    units_volume: str = "AF"

    def effective_date(self) -> datetime:
        """Return *date* or ``datetime.now()``."""
        return self.date if self.date is not None else datetime.now()

    def effective_header(self) -> str:
        """Return *header_text* or *title*."""
        return self.header_text if self.header_text is not None else self.title

    def effective_footer_left(self) -> str:
        """Return *footer_left* or formatted *date*."""
        if self.footer_left is not None:
            return self.footer_left
        return self.effective_date().strftime("%B %d, %Y")


class ReportPage:
    """Single letter-size page with header, footer, and printable-area coords.

    Parameters
    ----------
    config : ReportConfig
        Report-level configuration.
    page_number : int
        1-based page number.
    total_pages : int | None
        Total page count (for "Page X of Y").
    section_name : str
        Right-header text for this page's section.
    """

    def __init__(
        self,
        config: ReportConfig,
        page_number: int,
        total_pages: int | None = None,
        section_name: str = "",
    ) -> None:
        self.config = config
        self.page_number = page_number
        self.total_pages = total_pages
        self.section_name = section_name
        self._fig: Figure = plt.figure(figsize=(_PAGE_WIDTH, _PAGE_HEIGHT))
        self._margins = config.margins

    @property
    def fig(self) -> Figure:
        """The underlying matplotlib ``Figure``."""
        return self._fig

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _to_fig_coords(
        self,
        left: float,
        bottom: float,
        width: float,
        height: float,
    ) -> tuple[float, float, float, float]:
        """Convert printable-area fractions to figure fractions.

        Parameters
        ----------
        left, bottom, width, height : float
            Fractions of the printable area (0–1).

        Returns
        -------
        tuple[float, float, float, float]
            ``(fig_left, fig_bottom, fig_width, fig_height)`` in figure
            fraction coordinates suitable for ``fig.add_axes()``.
        """
        m = self._margins
        fig_left = m.printable_left + left * m.printable_width_frac
        fig_bottom = m.printable_bottom + bottom * m.printable_height_frac
        fig_width = width * m.printable_width_frac
        fig_height = height * m.printable_height_frac
        return fig_left, fig_bottom, fig_width, fig_height

    def add_axes(
        self,
        left: float,
        bottom: float,
        width: float,
        height: float,
        **kwargs: Any,
    ) -> plt.Axes:
        """Add an axes using printable-area fractions.

        Parameters
        ----------
        left, bottom, width, height : float
            Fractions of the printable area (each 0–1).
        **kwargs
            Passed to ``fig.add_axes()``.
        """
        rect = self._to_fig_coords(left, bottom, width, height)
        return self._fig.add_axes(rect, **kwargs)

    def add_gridspec(
        self,
        nrows: int,
        ncols: int,
        **kwargs: Any,
    ) -> GridSpec:
        """Add a :class:`~matplotlib.gridspec.GridSpec` confined to the printable area.

        Parameters
        ----------
        nrows, ncols : int
            Grid dimensions.
        **kwargs
            Passed to ``GridSpec`` (e.g. ``wspace``, ``hspace``).
        """
        m = self._margins
        gs = GridSpec(
            nrows,
            ncols,
            figure=self._fig,
            left=m.printable_left,
            right=m.printable_left + m.printable_width_frac,
            bottom=m.printable_bottom,
            top=m.printable_bottom + m.printable_height_frac,
            **kwargs,
        )
        return gs

    # ------------------------------------------------------------------
    # Header / Footer
    # ------------------------------------------------------------------

    def _draw_header(self) -> None:
        """Draw the header: title left, section right, thin rule."""
        m = self._margins
        header_y = 1.0 - (m.top / _PAGE_HEIGHT) + 0.01

        header_left = self.config.effective_header()
        if header_left:
            self._fig.text(
                m.printable_left,
                header_y,
                header_left,
                fontsize=8,
                fontweight="bold",
                va="bottom",
                ha="left",
            )
        if self.section_name:
            self._fig.text(
                m.printable_left + m.printable_width_frac,
                header_y,
                self.section_name,
                fontsize=8,
                va="bottom",
                ha="right",
                style="italic",
            )

        # Thin rule below header
        rule_y = header_y - 0.005
        from matplotlib.lines import Line2D

        line = Line2D(
            [m.printable_left, m.printable_left + m.printable_width_frac],
            [rule_y, rule_y],
            transform=self._fig.transFigure,
            color="#999999",
            linewidth=0.6,
            clip_on=False,
        )
        self._fig.add_artist(line)

    def _draw_footer(self) -> None:
        """Draw the footer: date left, page number center, thin rule."""
        m = self._margins
        footer_y = m.bottom / _PAGE_HEIGHT - 0.005

        # Thin rule above footer
        rule_y = footer_y + 0.015
        from matplotlib.lines import Line2D

        line = Line2D(
            [m.printable_left, m.printable_left + m.printable_width_frac],
            [rule_y, rule_y],
            transform=self._fig.transFigure,
            color="#999999",
            linewidth=0.6,
            clip_on=False,
        )
        self._fig.add_artist(line)

        # Date left
        footer_left_text = self.config.effective_footer_left()
        if footer_left_text:
            self._fig.text(
                m.printable_left,
                footer_y,
                footer_left_text,
                fontsize=7,
                va="top",
                ha="left",
                color="#666666",
            )

        # Page number center
        if self.config.show_page_numbers:
            if self.total_pages is not None:
                page_text = f"Page {self.page_number} of {self.total_pages}"
            else:
                page_text = f"Page {self.page_number}"
            center_x = m.printable_left + m.printable_width_frac / 2
            self._fig.text(
                center_x,
                footer_y,
                page_text,
                fontsize=7,
                va="top",
                ha="center",
                color="#666666",
            )

    def save_to_pdf(self, pdf: PdfPages) -> None:
        """Draw header/footer and save the page to the PDF.

        Parameters
        ----------
        pdf : PdfPages
            Open :class:`~matplotlib.backends.backend_pdf.PdfPages` handle.
        """
        self._draw_header()
        self._draw_footer()
        pdf.savefig(self._fig, dpi=self.config.dpi)
        plt.close(self._fig)
