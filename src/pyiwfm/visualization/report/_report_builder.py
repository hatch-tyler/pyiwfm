"""ReportBuilder — orchestrator for multi-page PDF reports.

Provides a two-pass build pipeline: first count pages, then render with
"Page X of Y" in footers.  Uses :class:`ReportPage` for consistent
header/footer layout and the ``pyiwfm-report`` style.

Classes
-------
- :class:`ReportBuilder` — Main orchestrator.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from pyiwfm.visualization._plot_utils import REPORT_STYLE
from pyiwfm.visualization.report._page_layout import ReportConfig, ReportPage
from pyiwfm.visualization.report._title_page import draw_title_page


class _Section:
    """Internal representation of a registered report section."""

    __slots__ = ("render_func", "name", "n_pages", "rasterize", "raster_dpi")

    def __init__(
        self,
        render_func: Callable[..., None],
        name: str,
        n_pages: int,
        rasterize: bool,
        raster_dpi: int,
    ) -> None:
        self.render_func = render_func
        self.name = name
        self.n_pages = n_pages
        self.rasterize = rasterize
        self.raster_dpi = raster_dpi


class ReportBuilder:
    """Orchestrator for building multi-page PDF reports.

    Parameters
    ----------
    config : ReportConfig
        Report-level configuration.
    output_path : Path | str
        Destination PDF file path.

    Examples
    --------
    >>> builder = ReportBuilder(ReportConfig(title="My Report"), "report.pdf")
    >>> builder.add_title_page()
    >>> builder.add_page(my_render_function, name="Results")
    >>> builder.build()
    """

    def __init__(self, config: ReportConfig, output_path: Path | str) -> None:
        self.config = config
        self.output_path = Path(output_path)
        self._has_title_page = False
        self._title_page_kwargs: dict[str, Any] = {}
        self._sections: list[_Section] = []

    def add_title_page(
        self,
        summary_stats: dict[str, Any] | None = None,
        model_info: dict[str, Any] | None = None,
    ) -> None:
        """Register a title page.

        Parameters
        ----------
        summary_stats : dict | None
            Summary statistics for the title page.
        model_info : dict | None
            Model info for the title page.
        """
        self._has_title_page = True
        self._title_page_kwargs = {
            "summary_stats": summary_stats,
            "model_info": model_info,
        }

    def add_page(
        self,
        render_func: Callable[[ReportPage], None],
        *,
        name: str = "",
        rasterize: bool = False,
        raster_dpi: int = 100,
    ) -> None:
        """Register a single content page.

        Parameters
        ----------
        render_func : Callable[[ReportPage], None]
            Function that receives a :class:`ReportPage` and draws content.
        name : str
            Section name shown in the right header.
        rasterize : bool
            Whether to rasterize the page for smaller file size.
        raster_dpi : int
            DPI for rasterized output.
        """
        self._sections.append(_Section(render_func, name, 1, rasterize, raster_dpi))

    def add_pages(
        self,
        render_func: Callable[[ReportPage, int], None],
        n_pages: int,
        *,
        name: str = "",
        rasterize: bool = False,
        raster_dpi: int = 100,
    ) -> None:
        """Register a multi-page section.

        Parameters
        ----------
        render_func : Callable[[ReportPage, int], None]
            Function receiving ``(page, page_index_within_section)``.
        n_pages : int
            Number of pages in this section.
        name : str
            Section name for headers.
        rasterize : bool
            Whether to rasterize pages.
        raster_dpi : int
            DPI for rasterized output.
        """
        self._sections.append(_Section(render_func, name, n_pages, rasterize, raster_dpi))

    def add_hydrograph_pages(
        self,
        items: list[Any],
        render_func: Callable[[ReportPage, list[Any]], None],
        items_per_page: int = 6,
        *,
        name: str = "",
        rasterize: bool = False,
        raster_dpi: int = 100,
    ) -> None:
        """Register paginated panel pages (e.g. 6 hydrographs per page).

        Parameters
        ----------
        items : list
            All items to paginate.
        render_func : Callable[[ReportPage, list], None]
            Function receiving ``(page, items_for_this_page)``.
        items_per_page : int
            Number of items per page.
        name : str
            Section name for headers.
        rasterize : bool
            Whether to rasterize pages.
        raster_dpi : int
            DPI for rasterized output.
        """
        n_pages = max(1, math.ceil(len(items) / items_per_page))

        def _paginated_render(page: ReportPage, page_idx: int) -> None:
            start = page_idx * items_per_page
            end = min(start + items_per_page, len(items))
            render_func(page, items[start:end])

        self._sections.append(_Section(_paginated_render, name, n_pages, rasterize, raster_dpi))

    def _count_pages(self) -> int:
        """Count total pages including title page."""
        total = 1 if self._has_title_page else 0
        for section in self._sections:
            total += section.n_pages
        return max(total, 1)

    def build(self, progress: bool = True) -> Path:
        """Build the PDF report.

        Parameters
        ----------
        progress : bool
            If ``True``, print progress to stdout.

        Returns
        -------
        Path
            Path to the generated PDF file.
        """
        total_pages = self._count_pages()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with plt.style.context(REPORT_STYLE):
            with PdfPages(
                self.output_path,
                metadata={
                    "Title": self.config.title,
                    "Author": self.config.author,
                    "CreationDate": self.config.effective_date(),
                },
            ) as pdf:
                page_num = 0

                # Title page (no header/footer)
                if self._has_title_page:
                    page_num += 1
                    if progress:
                        print(f"  Page {page_num}/{total_pages}: Title page")
                    page = ReportPage(
                        self.config,
                        page_number=page_num,
                        total_pages=total_pages,
                        section_name="",
                    )
                    draw_title_page(page, self.config, **self._title_page_kwargs)
                    pdf.savefig(page.fig, dpi=self.config.dpi)
                    plt.close(page.fig)

                # Content sections
                for section in self._sections:
                    for sec_page_idx in range(section.n_pages):
                        page_num += 1
                        if progress:
                            label = section.name or "Content"
                            if section.n_pages > 1:
                                label += f" ({sec_page_idx + 1}/{section.n_pages})"
                            print(f"  Page {page_num}/{total_pages}: {label}")

                        page = ReportPage(
                            self.config,
                            page_number=page_num,
                            total_pages=total_pages,
                            section_name=section.name,
                        )

                        if section.rasterize:
                            page.fig.set_rasterized(True)

                        # Call render function
                        if section.n_pages == 1:
                            section.render_func(page)
                        else:
                            section.render_func(page, sec_page_idx)

                        page.save_to_pdf(pdf)

                # If no pages were added, create a blank page
                if page_num == 0:
                    page = ReportPage(
                        self.config,
                        page_number=1,
                        total_pages=1,
                    )
                    page.fig.text(
                        0.5,
                        0.5,
                        "Empty Report",
                        ha="center",
                        va="center",
                        fontsize=14,
                    )
                    page.save_to_pdf(pdf)

        if progress:
            print(f"  Report saved to {self.output_path}")

        return self.output_path
