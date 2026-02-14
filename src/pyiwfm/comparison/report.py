"""
Report generation for model comparisons.

This module provides classes for generating comparison reports
in various formats (text, JSON, HTML).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from pyiwfm.comparison.differ import ModelDiff
    from pyiwfm.comparison.metrics import ComparisonMetrics, TimeSeriesComparison


class BaseReport(ABC):
    """Abstract base class for report generators."""

    @abstractmethod
    def generate(self, model_diff: "ModelDiff") -> str:
        """
        Generate report content from model diff.

        Args:
            model_diff: Model difference object

        Returns:
            Report content as string
        """
        pass

    @abstractmethod
    def generate_metrics_report(self, metrics: "ComparisonMetrics") -> str:
        """
        Generate report content from metrics.

        Args:
            metrics: Comparison metrics object

        Returns:
            Report content as string
        """
        pass

    def save(self, model_diff: "ModelDiff", output_path: Path | str) -> None:
        """
        Save report to file.

        Args:
            model_diff: Model difference object
            output_path: Output file path
        """
        content = self.generate(model_diff)
        Path(output_path).write_text(content, encoding="utf-8")


class TextReport(BaseReport):
    """Generate plain text reports."""

    def generate(self, model_diff: "ModelDiff") -> str:
        """Generate text report from model diff."""
        lines = [
            "=" * 60,
            "Model Comparison Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            "",
        ]

        # Summary
        lines.append(model_diff.summary())
        lines.append("")

        # Detailed changes
        if model_diff.mesh_diff and not model_diff.mesh_diff.is_identical:
            lines.append("-" * 40)
            lines.append("Mesh Changes (Detailed)")
            lines.append("-" * 40)

            for item in model_diff.mesh_diff.items[:50]:  # Limit to 50 items
                lines.append(str(item))

            if len(model_diff.mesh_diff.items) > 50:
                lines.append(f"... and {len(model_diff.mesh_diff.items) - 50} more items")

        if model_diff.stratigraphy_diff and not model_diff.stratigraphy_diff.is_identical:
            lines.append("")
            lines.append("-" * 40)
            lines.append("Stratigraphy Changes (Detailed)")
            lines.append("-" * 40)

            for item in model_diff.stratigraphy_diff.items[:50]:
                lines.append(str(item))

            if len(model_diff.stratigraphy_diff.items) > 50:
                lines.append(
                    f"... and {len(model_diff.stratigraphy_diff.items) - 50} more items"
                )

        return "\n".join(lines)

    def generate_metrics_report(self, metrics: "ComparisonMetrics") -> str:
        """Generate text report from metrics."""
        lines = [
            "=" * 60,
            "Comparison Metrics Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            "",
            metrics.summary(),
        ]
        return "\n".join(lines)


class JsonReport(BaseReport):
    """Generate JSON reports."""

    def __init__(self, indent: int = 2) -> None:
        """
        Initialize JSON report generator.

        Args:
            indent: JSON indentation level
        """
        self.indent = indent

    def generate(self, model_diff: "ModelDiff") -> str:
        """Generate JSON report from model diff."""
        data = {
            "report_type": "model_comparison",
            "generated": datetime.now().isoformat(),
            "summary": {
                "is_identical": model_diff.is_identical,
                "statistics": model_diff.statistics(),
            },
            "diff": model_diff.to_dict(),
        }
        return json.dumps(data, indent=self.indent, default=str)

    def generate_metrics_report(self, metrics: "ComparisonMetrics") -> str:
        """Generate JSON report from metrics."""
        data = {
            "report_type": "comparison_metrics",
            "generated": datetime.now().isoformat(),
            "rating": metrics.rating(),
            **metrics.to_dict(),
        }
        return json.dumps(data, indent=self.indent)


class HtmlReport(BaseReport):
    """Generate HTML reports."""

    def __init__(self, title: str = "Model Comparison Report") -> None:
        """
        Initialize HTML report generator.

        Args:
            title: HTML page title
        """
        self.title = title

    def generate(self, model_diff: "ModelDiff") -> str:
        """Generate HTML report from model diff."""
        stats = model_diff.statistics()

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{self.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 1px solid #ccc; }}
        .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
        .stat {{ display: inline-block; margin: 10px 20px; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #007bff; }}
        .stat-label {{ font-size: 12px; color: #666; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #f2f2f2; }}
        .added {{ color: green; }}
        .removed {{ color: red; }}
        .modified {{ color: orange; }}
        .identical {{ color: green; font-weight: bold; }}
        .timestamp {{ color: #999; font-size: 12px; }}
    </style>
</head>
<body>
    <h1>{self.title}</h1>
    <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <div class="summary">
        <h2>Summary</h2>
        {"<p class='identical'>Models are identical.</p>" if model_diff.is_identical else ""}
        <div class="stat">
            <div class="stat-value">{stats['total_changes']}</div>
            <div class="stat-label">Total Changes</div>
        </div>
        <div class="stat">
            <div class="stat-value added">+{stats['added']}</div>
            <div class="stat-label">Added</div>
        </div>
        <div class="stat">
            <div class="stat-value removed">-{stats['removed']}</div>
            <div class="stat-label">Removed</div>
        </div>
        <div class="stat">
            <div class="stat-value modified">~{stats['modified']}</div>
            <div class="stat-label">Modified</div>
        </div>
    </div>
"""

        # Add mesh diff table
        if model_diff.mesh_diff and not model_diff.mesh_diff.is_identical:
            html += """
    <h2>Mesh Changes</h2>
    <table>
        <tr>
            <th>Type</th>
            <th>Path</th>
            <th>Old Value</th>
            <th>New Value</th>
        </tr>
"""
            for item in model_diff.mesh_diff.items[:100]:
                type_class = item.diff_type.value
                html += f"""        <tr>
            <td class="{type_class}">{item.diff_type.value}</td>
            <td>{item.path}</td>
            <td>{item.old_value}</td>
            <td>{item.new_value}</td>
        </tr>
"""
            if len(model_diff.mesh_diff.items) > 100:
                html += f"""        <tr>
            <td colspan="4">... and {len(model_diff.mesh_diff.items) - 100} more items</td>
        </tr>
"""
            html += "    </table>\n"

        # Add stratigraphy diff table
        if model_diff.stratigraphy_diff and not model_diff.stratigraphy_diff.is_identical:
            html += """
    <h2>Stratigraphy Changes</h2>
    <table>
        <tr>
            <th>Type</th>
            <th>Path</th>
            <th>Old Value</th>
            <th>New Value</th>
        </tr>
"""
            for item in model_diff.stratigraphy_diff.items[:100]:
                type_class = item.diff_type.value
                html += f"""        <tr>
            <td class="{type_class}">{item.diff_type.value}</td>
            <td>{item.path}</td>
            <td>{item.old_value}</td>
            <td>{item.new_value}</td>
        </tr>
"""
            if len(model_diff.stratigraphy_diff.items) > 100:
                html += f"""        <tr>
            <td colspan="4">... and {len(model_diff.stratigraphy_diff.items) - 100} more items</td>
        </tr>
"""
            html += "    </table>\n"

        html += """
</body>
</html>"""
        return html

    def generate_metrics_report(self, metrics: "ComparisonMetrics") -> str:
        """Generate HTML report from metrics."""
        rating = metrics.rating()
        rating_colors = {
            "excellent": "#28a745",
            "good": "#5cb85c",
            "fair": "#f0ad4e",
            "poor": "#d9534f",
        }
        rating_color = rating_colors.get(rating, "#666")

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Comparison Metrics Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .rating {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            background: {rating_color};
            color: white;
            font-weight: bold;
        }}
        table {{ border-collapse: collapse; width: 50%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background: #f2f2f2; width: 40%; }}
        .timestamp {{ color: #999; font-size: 12px; }}
    </style>
</head>
<body>
    <h1>Comparison Metrics Report</h1>
    <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <p>Overall Rating: <span class="rating">{rating.upper()}</span></p>

    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>RMSE</td><td>{metrics.rmse:.6f}</td></tr>
        <tr><td>MAE</td><td>{metrics.mae:.6f}</td></tr>
        <tr><td>MBE</td><td>{metrics.mbe:.6f}</td></tr>
        <tr><td>Nash-Sutcliffe</td><td>{metrics.nash_sutcliffe:.6f}</td></tr>
        <tr><td>Percent Bias</td><td>{metrics.percent_bias:.2f}%</td></tr>
        <tr><td>Correlation</td><td>{metrics.correlation:.6f}</td></tr>
        <tr><td>Max Error</td><td>{metrics.max_error:.6f}</td></tr>
        <tr><td>N Points</td><td>{metrics.n_points}</td></tr>
    </table>
</body>
</html>"""
        return html


class ReportGenerator:
    """
    Factory class for generating reports in various formats.

    Provides a unified interface for generating reports in
    text, JSON, or HTML format.
    """

    def __init__(self) -> None:
        """Initialize the report generator."""
        self._generators = {
            "text": TextReport(),
            "json": JsonReport(),
            "html": HtmlReport(),
        }

    def generate(
        self,
        model_diff: "ModelDiff",
        format: Literal["text", "json", "html"] = "text",
    ) -> str:
        """
        Generate a report in the specified format.

        Args:
            model_diff: Model difference object
            format: Output format ('text', 'json', 'html')

        Returns:
            Report content as string

        Raises:
            ValueError: If format is not recognized
        """
        if format not in self._generators:
            raise ValueError(f"Unknown format: {format}. Use 'text', 'json', or 'html'.")

        return self._generators[format].generate(model_diff)

    def generate_metrics(
        self,
        metrics: "ComparisonMetrics",
        format: Literal["text", "json", "html"] = "text",
    ) -> str:
        """
        Generate a metrics report in the specified format.

        Args:
            metrics: Comparison metrics object
            format: Output format

        Returns:
            Report content as string
        """
        if format not in self._generators:
            raise ValueError(f"Unknown format: {format}")

        return self._generators[format].generate_metrics_report(metrics)

    def save(
        self,
        model_diff: "ModelDiff",
        output_path: Path | str,
        format: Literal["text", "json", "html"] | None = None,
    ) -> None:
        """
        Save report to file.

        Args:
            model_diff: Model difference object
            output_path: Output file path
            format: Output format (auto-detected from extension if None)
        """
        output_path = Path(output_path)

        # Auto-detect format from extension
        if format is None:
            ext = output_path.suffix.lower()
            format_map = {".txt": "text", ".json": "json", ".html": "html", ".htm": "html"}
            format = format_map.get(ext, "text")

        content = self.generate(model_diff, format=format)
        output_path.write_text(content, encoding="utf-8")


@dataclass
class ComparisonReport:
    """
    Container for a complete comparison report.

    Combines model diff, metrics, and metadata into
    a single report object.

    Attributes:
        title: Report title
        model_diff: Model difference (optional)
        head_metrics: Head comparison metrics (optional)
        flow_metrics: Flow comparison metrics (optional)
        description: Report description
        metadata: Additional metadata
    """

    title: str
    model_diff: "ModelDiff | None" = None
    head_metrics: "ComparisonMetrics | None" = None
    flow_metrics: "ComparisonMetrics | None" = None
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_text(self) -> str:
        """Convert report to text format."""
        lines = [
            "=" * 60,
            self.title,
            "=" * 60,
            "",
        ]

        if self.description:
            lines.append(self.description)
            lines.append("")

        if self.model_diff:
            lines.append("MODEL DIFFERENCES")
            lines.append("-" * 40)
            lines.append(self.model_diff.summary())
            lines.append("")

        if self.head_metrics:
            lines.append("HEAD COMPARISON METRICS")
            lines.append("-" * 40)
            lines.append(self.head_metrics.summary())
            lines.append("")

        if self.flow_metrics:
            lines.append("FLOW COMPARISON METRICS")
            lines.append("-" * 40)
            lines.append(self.flow_metrics.summary())
            lines.append("")

        return "\n".join(lines)

    def to_json(self) -> str:
        """Convert report to JSON format."""
        data: dict[str, Any] = {
            "title": self.title,
            "generated": datetime.now().isoformat(),
            "description": self.description,
        }

        if self.model_diff:
            data["model_diff"] = {
                "is_identical": self.model_diff.is_identical,
                "statistics": self.model_diff.statistics(),
            }

        if self.head_metrics:
            data["head_metrics"] = self.head_metrics.to_dict()

        if self.flow_metrics:
            data["flow_metrics"] = self.flow_metrics.to_dict()

        if self.metadata:
            data["metadata"] = self.metadata

        return json.dumps(data, indent=2, default=str)

    def to_html(self) -> str:
        """Convert report to HTML format."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{self.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; max-width: 1200px; }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #666; }}
        .section {{ background: #f9f9f9; padding: 15px; margin: 15px 0; border-radius: 5px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; }}
        .metric-box {{ background: white; padding: 15px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #007bff; }}
        .metric-label {{ font-size: 12px; color: #666; margin-top: 5px; }}
        .timestamp {{ color: #999; font-size: 12px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>{self.title}</h1>
    <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
"""

        if self.description:
            html += f"    <p>{self.description}</p>\n"

        if self.model_diff:
            stats = self.model_diff.statistics()
            html += f"""
    <div class="section">
        <h2>Model Differences</h2>
        <div class="metric-grid">
            <div class="metric-box">
                <div class="metric-value">{stats['total_changes']}</div>
                <div class="metric-label">Total Changes</div>
            </div>
            <div class="metric-box">
                <div class="metric-value" style="color: green;">+{stats['added']}</div>
                <div class="metric-label">Added</div>
            </div>
            <div class="metric-box">
                <div class="metric-value" style="color: red;">-{stats['removed']}</div>
                <div class="metric-label">Removed</div>
            </div>
            <div class="metric-box">
                <div class="metric-value" style="color: orange;">~{stats['modified']}</div>
                <div class="metric-label">Modified</div>
            </div>
        </div>
    </div>
"""

        if self.head_metrics:
            html += f"""
    <div class="section">
        <h2>Head Comparison Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>RMSE</td><td>{self.head_metrics.rmse:.4f}</td></tr>
            <tr><td>MAE</td><td>{self.head_metrics.mae:.4f}</td></tr>
            <tr><td>Nash-Sutcliffe</td><td>{self.head_metrics.nash_sutcliffe:.4f}</td></tr>
            <tr><td>Correlation</td><td>{self.head_metrics.correlation:.4f}</td></tr>
            <tr><td>Rating</td><td>{self.head_metrics.rating()}</td></tr>
        </table>
    </div>
"""

        if self.flow_metrics:
            html += f"""
    <div class="section">
        <h2>Flow Comparison Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>RMSE</td><td>{self.flow_metrics.rmse:.4f}</td></tr>
            <tr><td>MAE</td><td>{self.flow_metrics.mae:.4f}</td></tr>
            <tr><td>Nash-Sutcliffe</td><td>{self.flow_metrics.nash_sutcliffe:.4f}</td></tr>
            <tr><td>Correlation</td><td>{self.flow_metrics.correlation:.4f}</td></tr>
            <tr><td>Rating</td><td>{self.flow_metrics.rating()}</td></tr>
        </table>
    </div>
"""

        html += """
</body>
</html>"""
        return html

    def save(
        self,
        output_path: Path | str,
        format: Literal["text", "json", "html"] | None = None,
    ) -> None:
        """
        Save report to file.

        Args:
            output_path: Output file path
            format: Output format (auto-detected if None)
        """
        output_path = Path(output_path)

        if format is None:
            ext = output_path.suffix.lower()
            format_map = {".txt": "text", ".json": "json", ".html": "html", ".htm": "html"}
            format = format_map.get(ext, "text")

        if format == "text":
            content = self.to_text()
        elif format == "json":
            content = self.to_json()
        else:
            content = self.to_html()

        output_path.write_text(content, encoding="utf-8")
