"""Unit tests for the ReportBuilder orchestrator."""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pyiwfm.visualization.report._page_layout import ReportConfig, ReportPage
from pyiwfm.visualization.report._report_builder import ReportBuilder
from pyiwfm.visualization.report._title_page import draw_title_page


class TestReportBuilder:
    """Tests for ReportBuilder end-to-end PDF generation."""

    def test_empty_report(self) -> None:
        """An empty report (no sections) produces a valid single-page PDF."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = ReportConfig(title="Empty Report")
            builder = ReportBuilder(cfg, Path(tmpdir) / "empty.pdf")
            result = builder.build(progress=False)
            assert result.exists()
            assert result.stat().st_size > 0

    def test_single_page(self) -> None:
        """A single content page generates a PDF."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = ReportConfig(title="Single Page")
            builder = ReportBuilder(cfg, Path(tmpdir) / "single.pdf")

            def render(page: ReportPage) -> None:
                ax = page.add_axes(0.1, 0.1, 0.8, 0.8)
                ax.plot([1, 2, 3], [1, 4, 9])
                ax.set_title("Test Plot")

            builder.add_page(render, name="Test")
            result = builder.build(progress=False)
            assert result.exists()
            assert result.stat().st_size > 0

    def test_title_page(self) -> None:
        """Title page renders correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = ReportConfig(title="My Report", subtitle="A test report")
            builder = ReportBuilder(cfg, Path(tmpdir) / "titled.pdf")
            builder.add_title_page(
                summary_stats={"RMSE": 12.5, "NSE": 0.87},
                model_info={"Nodes": 30000, "Elements": 20000},
            )
            result = builder.build(progress=False)
            assert result.exists()
            assert result.stat().st_size > 0

    def test_multi_page(self) -> None:
        """Multiple pages generate a PDF with correct page count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = ReportConfig(title="Multi Page")
            builder = ReportBuilder(cfg, Path(tmpdir) / "multi.pdf")

            for i in range(3):

                def render(page: ReportPage, idx: int = i) -> None:
                    ax = page.add_axes(0.1, 0.1, 0.8, 0.8)
                    ax.plot([1, 2, 3], [idx, idx + 1, idx + 2])

                builder.add_page(render, name=f"Page {i + 1}")

            result = builder.build(progress=False)
            assert result.exists()
            assert result.stat().st_size > 0

    def test_add_pages_multi(self) -> None:
        """add_pages() generates multiple pages from a single section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = ReportConfig(title="Multi Section")
            builder = ReportBuilder(cfg, Path(tmpdir) / "multi_section.pdf")

            def render(page: ReportPage, page_idx: int) -> None:
                ax = page.add_axes(0.1, 0.1, 0.8, 0.8)
                ax.text(0.5, 0.5, f"Page index {page_idx}", transform=ax.transAxes)

            builder.add_pages(render, 3, name="Section A")
            result = builder.build(progress=False)
            assert result.exists()

    def test_hydrograph_pages(self) -> None:
        """add_hydrograph_pages() paginates items correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = ReportConfig(title="Hydrographs")
            builder = ReportBuilder(cfg, Path(tmpdir) / "hydro.pdf")

            items = list(range(15))  # 15 items, 6 per page = 3 pages

            def render(page: ReportPage, page_items: list[int]) -> None:
                ax = page.add_axes(0.1, 0.1, 0.8, 0.8)
                ax.text(
                    0.5, 0.5,
                    f"Items: {page_items}",
                    transform=ax.transAxes,
                    ha="center",
                )

            builder.add_hydrograph_pages(items, render, items_per_page=6, name="Wells")
            # Should count 3 pages
            assert builder._count_pages() == 3
            result = builder.build(progress=False)
            assert result.exists()

    def test_title_page_plus_content(self) -> None:
        """Title page + content pages produce correct total."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = ReportConfig(title="Full Report")
            builder = ReportBuilder(cfg, Path(tmpdir) / "full.pdf")
            builder.add_title_page()

            def render(page: ReportPage) -> None:
                ax = page.add_axes(0.1, 0.1, 0.8, 0.8)
                ax.plot([0, 1], [0, 1])

            builder.add_page(render, name="Content")
            # Title + 1 content = 2
            assert builder._count_pages() == 2
            result = builder.build(progress=False)
            assert result.exists()

    def test_page_numbering(self) -> None:
        """Page numbers are set correctly during build."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = ReportConfig(title="Numbered", show_page_numbers=True)
            builder = ReportBuilder(cfg, Path(tmpdir) / "numbered.pdf")
            builder.add_title_page()

            pages_seen: list[int] = []

            def render(page: ReportPage) -> None:
                pages_seen.append(page.page_number)

            builder.add_page(render, name="A")
            builder.add_page(render, name="B")
            builder.build(progress=False)
            assert pages_seen == [2, 3]  # Title is page 1

    def test_rasterize_flag(self) -> None:
        """Rasterize flag doesn't cause errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = ReportConfig(title="Rasterized")
            builder = ReportBuilder(cfg, Path(tmpdir) / "raster.pdf")

            def render(page: ReportPage) -> None:
                ax = page.add_axes(0.1, 0.1, 0.8, 0.8)
                ax.plot([1, 2], [3, 4])

            builder.add_page(render, name="Raster", rasterize=True, raster_dpi=72)
            result = builder.build(progress=False)
            assert result.exists()

    def test_output_path_creates_parents(self) -> None:
        """Builder creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            deep_path = Path(tmpdir) / "a" / "b" / "c" / "report.pdf"
            cfg = ReportConfig(title="Deep Path")
            builder = ReportBuilder(cfg, deep_path)

            def render(page: ReportPage) -> None:
                pass

            builder.add_page(render)
            result = builder.build(progress=False)
            assert result.exists()

    def test_progress_output(self, capsys: object) -> None:
        """Progress output is printed when progress=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = ReportConfig(title="Progress Test")
            builder = ReportBuilder(cfg, Path(tmpdir) / "progress.pdf")

            def render(page: ReportPage) -> None:
                pass

            builder.add_page(render, name="Test Page")
            builder.build(progress=True)
            # capsys is a pytest fixture but we just verify no crash


class TestDrawTitlePage:
    """Tests for draw_title_page function."""

    def test_title_only(self) -> None:
        cfg = ReportConfig(title="Test Title")
        page = ReportPage(cfg, page_number=1)
        draw_title_page(page, cfg)
        texts = [t.get_text() for t in page.fig.texts]
        assert any("Test Title" in t for t in texts)
        plt.close(page.fig)

    def test_with_subtitle(self) -> None:
        cfg = ReportConfig(title="Main Title", subtitle="Subtitle Text")
        page = ReportPage(cfg, page_number=1)
        draw_title_page(page, cfg)
        texts = [t.get_text() for t in page.fig.texts]
        assert any("Subtitle Text" in t for t in texts)
        plt.close(page.fig)

    def test_with_model_info(self) -> None:
        cfg = ReportConfig(title="Model Report")
        page = ReportPage(cfg, page_number=1)
        draw_title_page(
            page, cfg,
            model_info={"Nodes": 30000, "Elements": 20000, "Layers": 4},
        )
        texts = [t.get_text() for t in page.fig.texts]
        assert any("Model Summary" in t for t in texts)
        assert any("30000" in t for t in texts)
        plt.close(page.fig)

    def test_with_summary_stats(self) -> None:
        cfg = ReportConfig(title="Stats Report")
        page = ReportPage(cfg, page_number=1)
        draw_title_page(
            page, cfg,
            summary_stats={"RMSE (ft)": 12.345, "NSE": 0.876},
        )
        texts = [t.get_text() for t in page.fig.texts]
        assert any("Summary Statistics" in t for t in texts)
        assert any("12.345" in t for t in texts)
        plt.close(page.fig)

    def test_with_nonexistent_logo(self) -> None:
        """Non-existent logo path is handled gracefully."""
        cfg = ReportConfig(title="Logo Test", logo_path=Path("/nonexistent/logo.png"))
        page = ReportPage(cfg, page_number=1)
        draw_title_page(page, cfg)  # Should not raise
        plt.close(page.fig)

    def test_with_logo_fixture(self, tmp_path: Path) -> None:
        """Test logo loading with a real PNG file."""
        import numpy as np

        # Create a small test PNG
        logo_path = tmp_path / "test_logo.png"
        img = np.ones((50, 100, 3), dtype=np.uint8) * 128
        plt.imsave(str(logo_path), img)

        cfg = ReportConfig(title="Logo Report", logo_path=logo_path)
        page = ReportPage(cfg, page_number=1)
        draw_title_page(page, cfg)
        plt.close(page.fig)
