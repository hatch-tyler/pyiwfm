"""Unit tests for comparison report generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pyiwfm.core.mesh import AppGrid, Node, Element
from pyiwfm.core.stratigraphy import Stratigraphy
from pyiwfm.comparison.differ import ModelDiffer, MeshDiff, ModelDiff
from pyiwfm.comparison.metrics import ComparisonMetrics, TimeSeriesComparison
from pyiwfm.comparison.report import (
    ReportGenerator,
    TextReport,
    JsonReport,
    HtmlReport,
    ComparisonReport,
)


@pytest.fixture
def simple_grid() -> AppGrid:
    """Create a simple 2x2 quad mesh for testing."""
    nodes = {
        1: Node(id=1, x=0.0, y=0.0, is_boundary=True),
        2: Node(id=2, x=100.0, y=0.0, is_boundary=True),
        3: Node(id=3, x=200.0, y=0.0, is_boundary=True),
        4: Node(id=4, x=0.0, y=100.0, is_boundary=True),
        5: Node(id=5, x=100.0, y=100.0, is_boundary=False),
        6: Node(id=6, x=200.0, y=100.0, is_boundary=True),
        7: Node(id=7, x=0.0, y=200.0, is_boundary=True),
        8: Node(id=8, x=100.0, y=200.0, is_boundary=True),
        9: Node(id=9, x=200.0, y=200.0, is_boundary=True),
    }
    elements = {
        1: Element(id=1, vertices=(1, 2, 5, 4), subregion=1),
        2: Element(id=2, vertices=(2, 3, 6, 5), subregion=1),
        3: Element(id=3, vertices=(4, 5, 8, 7), subregion=2),
        4: Element(id=4, vertices=(5, 6, 9, 8), subregion=2),
    }
    grid = AppGrid(nodes=nodes, elements=elements)
    grid.compute_connectivity()
    return grid


@pytest.fixture
def modified_grid() -> AppGrid:
    """Create a modified version of the simple grid."""
    nodes = {
        1: Node(id=1, x=0.0, y=0.0, is_boundary=True),
        2: Node(id=2, x=100.0, y=0.0, is_boundary=True),
        3: Node(id=3, x=200.0, y=0.0, is_boundary=True),
        4: Node(id=4, x=0.0, y=100.0, is_boundary=True),
        5: Node(id=5, x=105.0, y=105.0, is_boundary=False),  # Modified
        6: Node(id=6, x=200.0, y=100.0, is_boundary=True),
        7: Node(id=7, x=0.0, y=200.0, is_boundary=True),
        8: Node(id=8, x=100.0, y=200.0, is_boundary=True),
        9: Node(id=9, x=200.0, y=200.0, is_boundary=True),
        10: Node(id=10, x=250.0, y=100.0, is_boundary=True),  # Added
    }
    elements = {
        1: Element(id=1, vertices=(1, 2, 5, 4), subregion=1),
        2: Element(id=2, vertices=(2, 3, 6, 5), subregion=2),  # Modified
        3: Element(id=3, vertices=(4, 5, 8, 7), subregion=2),
        4: Element(id=4, vertices=(5, 6, 9, 8), subregion=2),
        5: Element(id=5, vertices=(3, 10, 6), subregion=1),  # Added
    }
    grid = AppGrid(nodes=nodes, elements=elements)
    grid.compute_connectivity()
    return grid


@pytest.fixture
def sample_metrics() -> ComparisonMetrics:
    """Create sample comparison metrics."""
    observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    simulated = np.array([1.1, 2.2, 3.1, 4.2, 5.1])
    return ComparisonMetrics.compute(observed, simulated)


@pytest.fixture
def sample_timeseries() -> TimeSeriesComparison:
    """Create sample time series comparison."""
    times = np.arange(10)
    observed = np.sin(times * 0.5) + 10
    simulated = np.sin(times * 0.5 + 0.1) + 10
    return TimeSeriesComparison(
        times=times,
        observed=observed,
        simulated=simulated,
    )


class TestTextReport:
    """Tests for text report generation."""

    def test_text_report_creation(self) -> None:
        """Test creating a text report."""
        report = TextReport()
        assert report is not None

    def test_text_report_from_model_diff(
        self, simple_grid: AppGrid, modified_grid: AppGrid
    ) -> None:
        """Test generating text report from model diff."""
        mesh_diff = MeshDiff.compare(simple_grid, modified_grid)
        model_diff = ModelDiff(mesh_diff=mesh_diff)

        report = TextReport()
        content = report.generate(model_diff)

        assert isinstance(content, str)
        assert len(content) > 0
        assert "Model" in content or "Diff" in content

    def test_text_report_from_metrics(
        self, sample_metrics: ComparisonMetrics
    ) -> None:
        """Test generating text report from metrics."""
        report = TextReport()
        content = report.generate_metrics_report(sample_metrics)

        assert isinstance(content, str)
        assert "RMSE" in content
        assert "MAE" in content

    def test_text_report_save(
        self,
        simple_grid: AppGrid,
        modified_grid: AppGrid,
        tmp_path: Path,
    ) -> None:
        """Test saving text report to file."""
        mesh_diff = MeshDiff.compare(simple_grid, modified_grid)
        model_diff = ModelDiff(mesh_diff=mesh_diff)

        report = TextReport()
        output_file = tmp_path / "report.txt"
        report.save(model_diff, output_file)

        assert output_file.exists()
        content = output_file.read_text()
        assert len(content) > 0


class TestJsonReport:
    """Tests for JSON report generation."""

    def test_json_report_creation(self) -> None:
        """Test creating a JSON report."""
        report = JsonReport()
        assert report is not None

    def test_json_report_from_model_diff(
        self, simple_grid: AppGrid, modified_grid: AppGrid
    ) -> None:
        """Test generating JSON report from model diff."""
        mesh_diff = MeshDiff.compare(simple_grid, modified_grid)
        model_diff = ModelDiff(mesh_diff=mesh_diff)

        report = JsonReport()
        content = report.generate(model_diff)

        # Should be valid JSON
        data = json.loads(content)
        assert isinstance(data, dict)

    def test_json_report_from_metrics(
        self, sample_metrics: ComparisonMetrics
    ) -> None:
        """Test generating JSON report from metrics."""
        report = JsonReport()
        content = report.generate_metrics_report(sample_metrics)

        data = json.loads(content)
        assert "rmse" in data
        assert "mae" in data

    def test_json_report_save(
        self,
        simple_grid: AppGrid,
        modified_grid: AppGrid,
        tmp_path: Path,
    ) -> None:
        """Test saving JSON report to file."""
        mesh_diff = MeshDiff.compare(simple_grid, modified_grid)
        model_diff = ModelDiff(mesh_diff=mesh_diff)

        report = JsonReport()
        output_file = tmp_path / "report.json"
        report.save(model_diff, output_file)

        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert isinstance(data, dict)


class TestHtmlReport:
    """Tests for HTML report generation."""

    def test_html_report_creation(self) -> None:
        """Test creating an HTML report."""
        report = HtmlReport()
        assert report is not None

    def test_html_report_from_model_diff(
        self, simple_grid: AppGrid, modified_grid: AppGrid
    ) -> None:
        """Test generating HTML report from model diff."""
        mesh_diff = MeshDiff.compare(simple_grid, modified_grid)
        model_diff = ModelDiff(mesh_diff=mesh_diff)

        report = HtmlReport()
        content = report.generate(model_diff)

        assert isinstance(content, str)
        assert "<html>" in content or "<!DOCTYPE" in content

    def test_html_report_from_metrics(
        self, sample_metrics: ComparisonMetrics
    ) -> None:
        """Test generating HTML report from metrics."""
        report = HtmlReport()
        content = report.generate_metrics_report(sample_metrics)

        assert "<html>" in content or "<!DOCTYPE" in content
        assert "RMSE" in content

    def test_html_report_save(
        self,
        simple_grid: AppGrid,
        modified_grid: AppGrid,
        tmp_path: Path,
    ) -> None:
        """Test saving HTML report to file."""
        mesh_diff = MeshDiff.compare(simple_grid, modified_grid)
        model_diff = ModelDiff(mesh_diff=mesh_diff)

        report = HtmlReport()
        output_file = tmp_path / "report.html"
        report.save(model_diff, output_file)

        assert output_file.exists()
        content = output_file.read_text()
        assert "<html>" in content or "<!DOCTYPE" in content


class TestReportGenerator:
    """Tests for ReportGenerator class."""

    def test_generator_creation(self) -> None:
        """Test creating a ReportGenerator."""
        generator = ReportGenerator()
        assert generator is not None

    def test_generate_text_report(
        self, simple_grid: AppGrid, modified_grid: AppGrid
    ) -> None:
        """Test generating text report through generator."""
        mesh_diff = MeshDiff.compare(simple_grid, modified_grid)
        model_diff = ModelDiff(mesh_diff=mesh_diff)

        generator = ReportGenerator()
        content = generator.generate(model_diff, format="text")

        assert isinstance(content, str)

    def test_generate_json_report(
        self, simple_grid: AppGrid, modified_grid: AppGrid
    ) -> None:
        """Test generating JSON report through generator."""
        mesh_diff = MeshDiff.compare(simple_grid, modified_grid)
        model_diff = ModelDiff(mesh_diff=mesh_diff)

        generator = ReportGenerator()
        content = generator.generate(model_diff, format="json")

        data = json.loads(content)
        assert isinstance(data, dict)

    def test_generate_html_report(
        self, simple_grid: AppGrid, modified_grid: AppGrid
    ) -> None:
        """Test generating HTML report through generator."""
        mesh_diff = MeshDiff.compare(simple_grid, modified_grid)
        model_diff = ModelDiff(mesh_diff=mesh_diff)

        generator = ReportGenerator()
        content = generator.generate(model_diff, format="html")

        assert "<html>" in content or "<!DOCTYPE" in content

    def test_generate_invalid_format(
        self, simple_grid: AppGrid, modified_grid: AppGrid
    ) -> None:
        """Test error handling for invalid format."""
        mesh_diff = MeshDiff.compare(simple_grid, modified_grid)
        model_diff = ModelDiff(mesh_diff=mesh_diff)

        generator = ReportGenerator()
        with pytest.raises(ValueError, match="Unknown format"):
            generator.generate(model_diff, format="invalid")


class TestComparisonReport:
    """Tests for ComparisonReport container."""

    def test_comparison_report_creation(
        self,
        simple_grid: AppGrid,
        modified_grid: AppGrid,
        sample_metrics: ComparisonMetrics,
    ) -> None:
        """Test creating a ComparisonReport."""
        mesh_diff = MeshDiff.compare(simple_grid, modified_grid)
        model_diff = ModelDiff(mesh_diff=mesh_diff)

        report = ComparisonReport(
            title="Test Comparison",
            model_diff=model_diff,
            head_metrics=sample_metrics,
        )

        assert report.title == "Test Comparison"
        assert report.model_diff is not None
        assert report.head_metrics is not None

    def test_comparison_report_to_text(
        self,
        simple_grid: AppGrid,
        modified_grid: AppGrid,
        sample_metrics: ComparisonMetrics,
    ) -> None:
        """Test converting ComparisonReport to text."""
        mesh_diff = MeshDiff.compare(simple_grid, modified_grid)
        model_diff = ModelDiff(mesh_diff=mesh_diff)

        report = ComparisonReport(
            title="Test Comparison",
            model_diff=model_diff,
            head_metrics=sample_metrics,
        )

        text = report.to_text()
        assert "Test Comparison" in text

    def test_comparison_report_to_json(
        self,
        simple_grid: AppGrid,
        modified_grid: AppGrid,
        sample_metrics: ComparisonMetrics,
    ) -> None:
        """Test converting ComparisonReport to JSON."""
        mesh_diff = MeshDiff.compare(simple_grid, modified_grid)
        model_diff = ModelDiff(mesh_diff=mesh_diff)

        report = ComparisonReport(
            title="Test Comparison",
            model_diff=model_diff,
            head_metrics=sample_metrics,
        )

        json_str = report.to_json()
        data = json.loads(json_str)
        assert data["title"] == "Test Comparison"

    def test_comparison_report_to_html(
        self,
        simple_grid: AppGrid,
        modified_grid: AppGrid,
        sample_metrics: ComparisonMetrics,
    ) -> None:
        """Test converting ComparisonReport to HTML."""
        mesh_diff = MeshDiff.compare(simple_grid, modified_grid)
        model_diff = ModelDiff(mesh_diff=mesh_diff)

        report = ComparisonReport(
            title="Test Comparison",
            model_diff=model_diff,
            head_metrics=sample_metrics,
        )

        html = report.to_html()
        assert "<html>" in html or "<!DOCTYPE" in html
        assert "Test Comparison" in html


# ---------------------------------------------------------------------------
# Additional tests for increased coverage
# ---------------------------------------------------------------------------


class TestTextReportExtended:
    """Extended tests for TextReport covering more branches."""

    def test_text_report_identical_models(self, simple_grid: AppGrid) -> None:
        """Test text report when models are identical (no changes)."""
        mesh_diff = MeshDiff.compare(simple_grid, simple_grid)
        model_diff = ModelDiff(mesh_diff=mesh_diff)
        assert model_diff.is_identical

        report = TextReport()
        content = report.generate(model_diff)
        assert "identical" in content.lower()

    def test_text_report_with_stratigraphy_diff(
        self, simple_grid: AppGrid, modified_grid: AppGrid
    ) -> None:
        """Test text report includes stratigraphy changes section."""
        from pyiwfm.comparison.differ import StratigraphyDiff, DiffItem, DiffType

        mesh_diff = MeshDiff.compare(simple_grid, modified_grid)
        strat_diff = StratigraphyDiff(items=[
            DiffItem(
                path="stratigraphy.gs_elev[0]",
                diff_type=DiffType.MODIFIED,
                old_value=100.0,
                new_value=101.0,
            ),
            DiffItem(
                path="stratigraphy.gs_elev[1]",
                diff_type=DiffType.MODIFIED,
                old_value=200.0,
                new_value=202.0,
            ),
        ])
        model_diff = ModelDiff(mesh_diff=mesh_diff, stratigraphy_diff=strat_diff)

        report = TextReport()
        content = report.generate(model_diff)
        assert "Stratigraphy Changes" in content
        assert "stratigraphy.gs_elev[0]" in content

    def test_text_report_truncation_mesh(self) -> None:
        """Test text report truncates mesh diff items beyond 50."""
        from pyiwfm.comparison.differ import DiffItem, DiffType

        items = [
            DiffItem(
                path=f"mesh.nodes.{i}.x",
                diff_type=DiffType.MODIFIED,
                old_value=float(i),
                new_value=float(i + 1),
            )
            for i in range(60)
        ]
        mesh_diff = MeshDiff(items=items, nodes_modified=60)
        model_diff = ModelDiff(mesh_diff=mesh_diff)

        report = TextReport()
        content = report.generate(model_diff)
        assert "... and 10 more items" in content

    def test_text_report_truncation_stratigraphy(self) -> None:
        """Test text report truncates stratigraphy diff items beyond 50."""
        from pyiwfm.comparison.differ import StratigraphyDiff, DiffItem, DiffType

        items = [
            DiffItem(
                path=f"stratigraphy.gs_elev[{i}]",
                diff_type=DiffType.MODIFIED,
                old_value=float(i),
                new_value=float(i + 0.5),
            )
            for i in range(55)
        ]
        strat_diff = StratigraphyDiff(items=items)
        model_diff = ModelDiff(stratigraphy_diff=strat_diff)

        report = TextReport()
        content = report.generate(model_diff)
        assert "Stratigraphy Changes" in content
        assert "... and 5 more items" in content

    def test_text_report_no_diffs(self) -> None:
        """Test text report with a ModelDiff that has no mesh or strat diff."""
        model_diff = ModelDiff()
        assert model_diff.is_identical

        report = TextReport()
        content = report.generate(model_diff)
        assert "identical" in content.lower()

    def test_text_report_metrics_report_contains_header(
        self, sample_metrics: ComparisonMetrics
    ) -> None:
        """Test text metrics report has expected header."""
        report = TextReport()
        content = report.generate_metrics_report(sample_metrics)
        assert "Comparison Metrics Report" in content
        assert "Generated:" in content


class TestJsonReportExtended:
    """Extended tests for JsonReport covering more branches."""

    def test_json_report_custom_indent(
        self, simple_grid: AppGrid, modified_grid: AppGrid
    ) -> None:
        """Test JsonReport with custom indent level."""
        report = JsonReport(indent=4)
        assert report.indent == 4

        mesh_diff = MeshDiff.compare(simple_grid, modified_grid)
        model_diff = ModelDiff(mesh_diff=mesh_diff)
        content = report.generate(model_diff)
        data = json.loads(content)
        assert data["report_type"] == "model_comparison"

    def test_json_report_structure_fields(
        self, simple_grid: AppGrid, modified_grid: AppGrid
    ) -> None:
        """Test JSON report has all expected top-level fields."""
        mesh_diff = MeshDiff.compare(simple_grid, modified_grid)
        model_diff = ModelDiff(mesh_diff=mesh_diff)

        report = JsonReport()
        content = report.generate(model_diff)
        data = json.loads(content)

        assert "report_type" in data
        assert "generated" in data
        assert "summary" in data
        assert "diff" in data
        assert "is_identical" in data["summary"]
        assert "statistics" in data["summary"]

    def test_json_report_metrics_has_rating(
        self, sample_metrics: ComparisonMetrics
    ) -> None:
        """Test JSON metrics report includes the rating field."""
        report = JsonReport()
        content = report.generate_metrics_report(sample_metrics)
        data = json.loads(content)

        assert "report_type" in data
        assert data["report_type"] == "comparison_metrics"
        assert "rating" in data
        assert "generated" in data
        assert "rmse" in data
        assert "mae" in data

    def test_json_report_identical_model(self, simple_grid: AppGrid) -> None:
        """Test JSON report when models are identical."""
        mesh_diff = MeshDiff.compare(simple_grid, simple_grid)
        model_diff = ModelDiff(mesh_diff=mesh_diff)

        report = JsonReport()
        content = report.generate(model_diff)
        data = json.loads(content)

        assert data["summary"]["is_identical"] is True
        assert data["summary"]["statistics"]["total_changes"] == 0


class TestHtmlReportExtended:
    """Extended tests for HtmlReport covering more branches."""

    def test_html_report_custom_title(self) -> None:
        """Test HtmlReport with custom title."""
        report = HtmlReport(title="Custom Title")
        assert report.title == "Custom Title"

    def test_html_report_identical_models(self, simple_grid: AppGrid) -> None:
        """Test HTML report includes 'identical' note when models match."""
        mesh_diff = MeshDiff.compare(simple_grid, simple_grid)
        model_diff = ModelDiff(mesh_diff=mesh_diff)
        assert model_diff.is_identical

        report = HtmlReport()
        content = report.generate(model_diff)
        assert "Models are identical" in content

    def test_html_report_with_stratigraphy_diff(
        self, simple_grid: AppGrid, modified_grid: AppGrid
    ) -> None:
        """Test HTML report renders stratigraphy changes table."""
        from pyiwfm.comparison.differ import StratigraphyDiff, DiffItem, DiffType

        mesh_diff = MeshDiff.compare(simple_grid, modified_grid)
        strat_diff = StratigraphyDiff(items=[
            DiffItem(
                path="stratigraphy.gs_elev[0]",
                diff_type=DiffType.MODIFIED,
                old_value=100.0,
                new_value=101.0,
            ),
        ])
        model_diff = ModelDiff(mesh_diff=mesh_diff, stratigraphy_diff=strat_diff)

        report = HtmlReport()
        content = report.generate(model_diff)
        assert "Stratigraphy Changes" in content
        assert "stratigraphy.gs_elev[0]" in content
        assert "<table>" in content

    def test_html_report_mesh_truncation(self) -> None:
        """Test HTML report truncates mesh diff items beyond 100."""
        from pyiwfm.comparison.differ import DiffItem, DiffType

        items = [
            DiffItem(
                path=f"mesh.nodes.{i}.x",
                diff_type=DiffType.MODIFIED,
                old_value=float(i),
                new_value=float(i + 1),
            )
            for i in range(110)
        ]
        mesh_diff = MeshDiff(items=items, nodes_modified=110)
        model_diff = ModelDiff(mesh_diff=mesh_diff)

        report = HtmlReport()
        content = report.generate(model_diff)
        assert "... and 10 more items" in content

    def test_html_report_stratigraphy_truncation(self) -> None:
        """Test HTML report truncates stratigraphy diff items beyond 100."""
        from pyiwfm.comparison.differ import StratigraphyDiff, DiffItem, DiffType

        items = [
            DiffItem(
                path=f"stratigraphy.gs_elev[{i}]",
                diff_type=DiffType.MODIFIED,
                old_value=float(i),
                new_value=float(i + 0.5),
            )
            for i in range(120)
        ]
        strat_diff = StratigraphyDiff(items=items)
        model_diff = ModelDiff(stratigraphy_diff=strat_diff)

        report = HtmlReport()
        content = report.generate(model_diff)
        assert "Stratigraphy Changes" in content
        assert "... and 20 more items" in content

    def test_html_report_metrics_rating_colors(self) -> None:
        """Test HTML metrics report uses correct rating color for each level."""
        # Excellent: NSE >= 0.90
        m_excellent = ComparisonMetrics(
            rmse=0.01, mae=0.01, mbe=0.0, nash_sutcliffe=0.95,
            percent_bias=0.1, correlation=0.99, max_error=0.02, n_points=100,
        )
        report = HtmlReport()
        content = report.generate_metrics_report(m_excellent)
        assert "EXCELLENT" in content
        assert "#28a745" in content

        # Good: NSE >= 0.65
        m_good = ComparisonMetrics(
            rmse=0.5, mae=0.4, mbe=0.1, nash_sutcliffe=0.75,
            percent_bias=2.0, correlation=0.9, max_error=1.0, n_points=100,
        )
        content_good = report.generate_metrics_report(m_good)
        assert "GOOD" in content_good
        assert "#5cb85c" in content_good

        # Fair: NSE >= 0.50
        m_fair = ComparisonMetrics(
            rmse=1.0, mae=0.8, mbe=0.3, nash_sutcliffe=0.55,
            percent_bias=5.0, correlation=0.8, max_error=2.0, n_points=100,
        )
        content_fair = report.generate_metrics_report(m_fair)
        assert "FAIR" in content_fair
        assert "#f0ad4e" in content_fair

        # Poor: NSE < 0.50
        m_poor = ComparisonMetrics(
            rmse=5.0, mae=4.0, mbe=2.0, nash_sutcliffe=0.2,
            percent_bias=20.0, correlation=0.5, max_error=10.0, n_points=100,
        )
        content_poor = report.generate_metrics_report(m_poor)
        assert "POOR" in content_poor
        assert "#d9534f" in content_poor

    def test_html_report_metrics_table_values(
        self, sample_metrics: ComparisonMetrics
    ) -> None:
        """Test HTML metrics report contains all metric values in table."""
        report = HtmlReport()
        content = report.generate_metrics_report(sample_metrics)
        assert "RMSE" in content
        assert "MAE" in content
        assert "MBE" in content
        assert "Nash-Sutcliffe" in content
        assert "Percent Bias" in content
        assert "Correlation" in content
        assert "Max Error" in content
        assert "N Points" in content

    def test_html_report_custom_title_in_output(
        self, simple_grid: AppGrid, modified_grid: AppGrid
    ) -> None:
        """Test that a custom title appears in the HTML output."""
        mesh_diff = MeshDiff.compare(simple_grid, modified_grid)
        model_diff = ModelDiff(mesh_diff=mesh_diff)

        report = HtmlReport(title="My Custom Report")
        content = report.generate(model_diff)
        assert "My Custom Report" in content


class TestReportGeneratorExtended:
    """Extended tests for ReportGenerator covering metrics and save."""

    def test_generate_metrics_text(
        self, sample_metrics: ComparisonMetrics
    ) -> None:
        """Test generating text metrics report through generator."""
        generator = ReportGenerator()
        content = generator.generate_metrics(sample_metrics, format="text")
        assert isinstance(content, str)
        assert "RMSE" in content

    def test_generate_metrics_json(
        self, sample_metrics: ComparisonMetrics
    ) -> None:
        """Test generating JSON metrics report through generator."""
        generator = ReportGenerator()
        content = generator.generate_metrics(sample_metrics, format="json")
        data = json.loads(content)
        assert "rmse" in data

    def test_generate_metrics_html(
        self, sample_metrics: ComparisonMetrics
    ) -> None:
        """Test generating HTML metrics report through generator."""
        generator = ReportGenerator()
        content = generator.generate_metrics(sample_metrics, format="html")
        assert "<!DOCTYPE html>" in content
        assert "RMSE" in content

    def test_generate_metrics_invalid_format(
        self, sample_metrics: ComparisonMetrics
    ) -> None:
        """Test error handling for invalid metrics format."""
        generator = ReportGenerator()
        with pytest.raises(ValueError, match="Unknown format"):
            generator.generate_metrics(sample_metrics, format="csv")

    def test_save_auto_detect_txt(
        self, simple_grid: AppGrid, modified_grid: AppGrid, tmp_path: Path
    ) -> None:
        """Test ReportGenerator.save auto-detects .txt format."""
        mesh_diff = MeshDiff.compare(simple_grid, modified_grid)
        model_diff = ModelDiff(mesh_diff=mesh_diff)

        generator = ReportGenerator()
        output_file = tmp_path / "report.txt"
        generator.save(model_diff, output_file)

        assert output_file.exists()
        content = output_file.read_text()
        assert "Model" in content or "Report" in content

    def test_save_auto_detect_json(
        self, simple_grid: AppGrid, modified_grid: AppGrid, tmp_path: Path
    ) -> None:
        """Test ReportGenerator.save auto-detects .json format."""
        mesh_diff = MeshDiff.compare(simple_grid, modified_grid)
        model_diff = ModelDiff(mesh_diff=mesh_diff)

        generator = ReportGenerator()
        output_file = tmp_path / "report.json"
        generator.save(model_diff, output_file)

        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert isinstance(data, dict)
        assert "report_type" in data

    def test_save_auto_detect_html(
        self, simple_grid: AppGrid, modified_grid: AppGrid, tmp_path: Path
    ) -> None:
        """Test ReportGenerator.save auto-detects .html format."""
        mesh_diff = MeshDiff.compare(simple_grid, modified_grid)
        model_diff = ModelDiff(mesh_diff=mesh_diff)

        generator = ReportGenerator()
        output_file = tmp_path / "report.html"
        generator.save(model_diff, output_file)

        assert output_file.exists()
        content = output_file.read_text()
        assert "<!DOCTYPE html>" in content

    def test_save_auto_detect_htm(
        self, simple_grid: AppGrid, modified_grid: AppGrid, tmp_path: Path
    ) -> None:
        """Test ReportGenerator.save auto-detects .htm format as HTML."""
        mesh_diff = MeshDiff.compare(simple_grid, modified_grid)
        model_diff = ModelDiff(mesh_diff=mesh_diff)

        generator = ReportGenerator()
        output_file = tmp_path / "report.htm"
        generator.save(model_diff, output_file)

        assert output_file.exists()
        content = output_file.read_text()
        assert "<!DOCTYPE html>" in content

    def test_save_auto_detect_unknown_ext(
        self, simple_grid: AppGrid, modified_grid: AppGrid, tmp_path: Path
    ) -> None:
        """Test ReportGenerator.save defaults to text for unknown extensions."""
        mesh_diff = MeshDiff.compare(simple_grid, modified_grid)
        model_diff = ModelDiff(mesh_diff=mesh_diff)

        generator = ReportGenerator()
        output_file = tmp_path / "report.xyz"
        generator.save(model_diff, output_file)

        assert output_file.exists()
        content = output_file.read_text()
        # Text report uses '=' separators
        assert "=" * 60 in content

    def test_save_explicit_format_overrides_extension(
        self, simple_grid: AppGrid, modified_grid: AppGrid, tmp_path: Path
    ) -> None:
        """Test ReportGenerator.save uses explicit format over extension."""
        mesh_diff = MeshDiff.compare(simple_grid, modified_grid)
        model_diff = ModelDiff(mesh_diff=mesh_diff)

        generator = ReportGenerator()
        # File has .txt extension but we request JSON format
        output_file = tmp_path / "report.txt"
        generator.save(model_diff, output_file, format="json")

        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert isinstance(data, dict)


class TestComparisonReportExtended:
    """Extended tests for ComparisonReport covering all branches."""

    def test_report_minimal_no_diffs(self) -> None:
        """Test ComparisonReport with no model_diff or metrics."""
        report = ComparisonReport(title="Empty Report")
        assert report.title == "Empty Report"
        assert report.model_diff is None
        assert report.head_metrics is None
        assert report.flow_metrics is None
        assert report.description == ""
        assert report.metadata == {}

    def test_to_text_minimal(self) -> None:
        """Test to_text with only title (no model_diff, no metrics)."""
        report = ComparisonReport(title="Minimal Report")
        text = report.to_text()
        assert "Minimal Report" in text
        # Should not have model diff or metrics sections
        assert "MODEL DIFFERENCES" not in text
        assert "HEAD COMPARISON" not in text
        assert "FLOW COMPARISON" not in text

    def test_to_text_with_description(self) -> None:
        """Test to_text includes description when provided."""
        report = ComparisonReport(
            title="Report With Description",
            description="This is a detailed description of the comparison.",
        )
        text = report.to_text()
        assert "Report With Description" in text
        assert "This is a detailed description of the comparison." in text

    def test_to_text_with_flow_metrics(
        self, sample_metrics: ComparisonMetrics
    ) -> None:
        """Test to_text includes flow metrics section."""
        report = ComparisonReport(
            title="Flow Report",
            flow_metrics=sample_metrics,
        )
        text = report.to_text()
        assert "FLOW COMPARISON METRICS" in text

    def test_to_text_with_all_sections(
        self,
        simple_grid: AppGrid,
        modified_grid: AppGrid,
        sample_metrics: ComparisonMetrics,
    ) -> None:
        """Test to_text with model_diff, head_metrics, and flow_metrics."""
        mesh_diff = MeshDiff.compare(simple_grid, modified_grid)
        model_diff = ModelDiff(mesh_diff=mesh_diff)

        report = ComparisonReport(
            title="Full Report",
            description="All sections present.",
            model_diff=model_diff,
            head_metrics=sample_metrics,
            flow_metrics=sample_metrics,
        )
        text = report.to_text()
        assert "Full Report" in text
        assert "All sections present." in text
        assert "MODEL DIFFERENCES" in text
        assert "HEAD COMPARISON METRICS" in text
        assert "FLOW COMPARISON METRICS" in text

    def test_to_json_minimal(self) -> None:
        """Test to_json with only title."""
        report = ComparisonReport(title="JSON Minimal")
        json_str = report.to_json()
        data = json.loads(json_str)
        assert data["title"] == "JSON Minimal"
        assert "model_diff" not in data
        assert "head_metrics" not in data
        assert "flow_metrics" not in data
        # metadata should not appear when empty
        assert "metadata" not in data

    def test_to_json_with_metadata(self) -> None:
        """Test to_json includes metadata when provided."""
        report = ComparisonReport(
            title="Meta Report",
            metadata={"author": "test", "version": "1.0"},
        )
        json_str = report.to_json()
        data = json.loads(json_str)
        assert "metadata" in data
        assert data["metadata"]["author"] == "test"
        assert data["metadata"]["version"] == "1.0"

    def test_to_json_with_flow_metrics(
        self, sample_metrics: ComparisonMetrics
    ) -> None:
        """Test to_json includes flow_metrics section."""
        report = ComparisonReport(
            title="Flow JSON",
            flow_metrics=sample_metrics,
        )
        json_str = report.to_json()
        data = json.loads(json_str)
        assert "flow_metrics" in data
        assert "rmse" in data["flow_metrics"]

    def test_to_json_with_model_diff(
        self, simple_grid: AppGrid, modified_grid: AppGrid
    ) -> None:
        """Test to_json includes model_diff section."""
        mesh_diff = MeshDiff.compare(simple_grid, modified_grid)
        model_diff = ModelDiff(mesh_diff=mesh_diff)

        report = ComparisonReport(
            title="Diff JSON",
            model_diff=model_diff,
        )
        json_str = report.to_json()
        data = json.loads(json_str)
        assert "model_diff" in data
        assert "is_identical" in data["model_diff"]
        assert "statistics" in data["model_diff"]

    def test_to_html_minimal(self) -> None:
        """Test to_html with only title."""
        report = ComparisonReport(title="HTML Minimal")
        html = report.to_html()
        assert "<!DOCTYPE html>" in html
        assert "HTML Minimal" in html
        assert "Model Differences" not in html
        assert "Head Comparison" not in html

    def test_to_html_with_description(self) -> None:
        """Test to_html includes description paragraph."""
        report = ComparisonReport(
            title="HTML Desc",
            description="A description for HTML.",
        )
        html = report.to_html()
        assert "A description for HTML." in html

    def test_to_html_with_flow_metrics(
        self, sample_metrics: ComparisonMetrics
    ) -> None:
        """Test to_html includes flow metrics section."""
        report = ComparisonReport(
            title="HTML Flow",
            flow_metrics=sample_metrics,
        )
        html = report.to_html()
        assert "Flow Comparison Metrics" in html
        assert "RMSE" in html
        assert "Nash-Sutcliffe" in html
        assert "Rating" in html

    def test_to_html_with_model_diff(
        self, simple_grid: AppGrid, modified_grid: AppGrid
    ) -> None:
        """Test to_html includes model diff section with stats."""
        mesh_diff = MeshDiff.compare(simple_grid, modified_grid)
        model_diff = ModelDiff(mesh_diff=mesh_diff)

        report = ComparisonReport(
            title="HTML Diff",
            model_diff=model_diff,
        )
        html = report.to_html()
        assert "Model Differences" in html
        assert "Total Changes" in html
        assert "Added" in html
        assert "Removed" in html
        assert "Modified" in html

    def test_to_html_with_all_sections(
        self,
        simple_grid: AppGrid,
        modified_grid: AppGrid,
        sample_metrics: ComparisonMetrics,
    ) -> None:
        """Test to_html with all sections populated."""
        mesh_diff = MeshDiff.compare(simple_grid, modified_grid)
        model_diff = ModelDiff(mesh_diff=mesh_diff)

        report = ComparisonReport(
            title="Full HTML Report",
            description="Full report with all sections.",
            model_diff=model_diff,
            head_metrics=sample_metrics,
            flow_metrics=sample_metrics,
            metadata={"source": "unit test"},
        )
        html = report.to_html()
        assert "Full HTML Report" in html
        assert "Full report with all sections." in html
        assert "Model Differences" in html
        assert "Head Comparison Metrics" in html
        assert "Flow Comparison Metrics" in html

    def test_save_auto_detect_txt(self, tmp_path: Path) -> None:
        """Test ComparisonReport.save auto-detects .txt format."""
        report = ComparisonReport(title="Save TXT")
        output_file = tmp_path / "cr.txt"
        report.save(output_file)

        assert output_file.exists()
        content = output_file.read_text()
        assert "Save TXT" in content
        # Text format uses '=' separators
        assert "=" * 60 in content

    def test_save_auto_detect_json(self, tmp_path: Path) -> None:
        """Test ComparisonReport.save auto-detects .json format."""
        report = ComparisonReport(title="Save JSON")
        output_file = tmp_path / "cr.json"
        report.save(output_file)

        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert data["title"] == "Save JSON"

    def test_save_auto_detect_html(self, tmp_path: Path) -> None:
        """Test ComparisonReport.save auto-detects .html format."""
        report = ComparisonReport(title="Save HTML")
        output_file = tmp_path / "cr.html"
        report.save(output_file)

        assert output_file.exists()
        content = output_file.read_text()
        assert "<!DOCTYPE html>" in content
        assert "Save HTML" in content

    def test_save_auto_detect_htm(self, tmp_path: Path) -> None:
        """Test ComparisonReport.save auto-detects .htm format as HTML."""
        report = ComparisonReport(title="Save HTM")
        output_file = tmp_path / "cr.htm"
        report.save(output_file)

        assert output_file.exists()
        content = output_file.read_text()
        assert "<!DOCTYPE html>" in content

    def test_save_auto_detect_unknown_ext(self, tmp_path: Path) -> None:
        """Test ComparisonReport.save defaults to text for unknown ext."""
        report = ComparisonReport(title="Save Unknown")
        output_file = tmp_path / "cr.dat"
        report.save(output_file)

        assert output_file.exists()
        content = output_file.read_text()
        assert "Save Unknown" in content
        assert "=" * 60 in content

    def test_save_explicit_format(self, tmp_path: Path) -> None:
        """Test ComparisonReport.save uses explicit format over extension."""
        report = ComparisonReport(title="Explicit Format")
        # File has .txt extension but we specify JSON format
        output_file = tmp_path / "cr.txt"
        report.save(output_file, format="json")

        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert data["title"] == "Explicit Format"

    def test_save_explicit_html_format(self, tmp_path: Path) -> None:
        """Test ComparisonReport.save with explicit html format."""
        report = ComparisonReport(title="Explicit HTML")
        output_file = tmp_path / "cr.dat"
        report.save(output_file, format="html")

        assert output_file.exists()
        content = output_file.read_text()
        assert "<!DOCTYPE html>" in content

    def test_save_explicit_text_format(self, tmp_path: Path) -> None:
        """Test ComparisonReport.save with explicit text format."""
        report = ComparisonReport(title="Explicit Text")
        output_file = tmp_path / "cr.json"
        report.save(output_file, format="text")

        assert output_file.exists()
        content = output_file.read_text()
        assert "Explicit Text" in content
        assert "=" * 60 in content
