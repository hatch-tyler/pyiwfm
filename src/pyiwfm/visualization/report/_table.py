"""Minimalist Tufte-style table renderer for matplotlib.

USGS/Tufte conventions: no vertical rules, thin horizontal rules only at
header and footer, right-aligned numbers, optional zebra shading.

Classes
-------
- :class:`TableStyle` — Visual configuration for tables.

Functions
---------
- :func:`draw_table` — Render a data table on a matplotlib axes.
- :func:`draw_stats_panel` — Two-column statistics panel (metric | value).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


@dataclass
class TableStyle:
    """Visual configuration for minimalist tables.

    Attributes
    ----------
    fontsize : float
        Font size for table cells.
    header_fontweight : str
        Font weight for header row.
    rule_color : str
        Color for horizontal rules.
    rule_width : float
        Line width for horizontal rules.
    show_header_rule : bool
        Draw a rule below the header row.
    show_footer_rule : bool
        Draw a rule below the last data row.
    zebra_shade : str | None
        Background color for alternating rows (``None`` = no zebra).
    """

    fontsize: float = 8.0
    header_fontweight: str = "bold"
    rule_color: str = "#999999"
    rule_width: float = 0.6
    show_header_rule: bool = True
    show_footer_rule: bool = True
    zebra_shade: str | None = None


def _default_alignments(ncols: int) -> list[str]:
    """Default alignment: left for first column, right for the rest."""
    if ncols <= 1:
        return ["left"] * ncols
    return ["left"] + ["right"] * (ncols - 1)


def _default_widths(ncols: int) -> list[float]:
    """Equal-width columns summing to 1.0."""
    if ncols == 0:
        return []
    w = 1.0 / ncols
    return [w] * ncols


def draw_table(
    ax: Axes,
    data: list[list[Any]],
    col_labels: list[str] | None = None,
    col_widths: list[float] | None = None,
    col_alignments: list[str] | None = None,
    summary_row_start: int | None = None,
    style: TableStyle | None = None,
    figure_ref: str | None = None,
) -> None:
    """Render a minimalist table on an axes.

    The axes is turned off (no spines, ticks, labels).  The table is drawn
    using ``ax.text()`` calls positioned in axes-fraction coordinates.

    Parameters
    ----------
    ax : Axes
        Target axes (will be cleared and turned off).
    data : list[list[Any]]
        Row-major data (each inner list is one row).
    col_labels : list[str] | None
        Header labels.  If ``None``, no header row is drawn.
    col_widths : list[float] | None
        Relative column widths (should sum to ~1.0).  Defaults to equal.
    col_alignments : list[str] | None
        Per-column alignment (``"left"``, ``"center"``, ``"right"``).
    summary_row_start : int | None
        Index into *data* where summary rows begin (drawn with bold weight).
    style : TableStyle | None
        Visual configuration.  Uses defaults if ``None``.
    figure_ref : str | None
        Optional figure reference label placed above the table.
    """
    if style is None:
        style = TableStyle()

    ax.set_axis_off()

    if not data and not col_labels:
        return

    ncols = len(data[0]) if data else (len(col_labels) if col_labels else 0)
    if ncols == 0:
        return

    if col_widths is None:
        col_widths = _default_widths(ncols)
    if col_alignments is None:
        col_alignments = _default_alignments(ncols)

    nrows = len(data)
    has_header = col_labels is not None
    total_rows = nrows + (1 if has_header else 0)

    # Vertical spacing
    top_margin = 0.95
    bottom_margin = 0.05
    if figure_ref:
        top_margin = 0.90
    available = top_margin - bottom_margin
    if total_rows > 0:
        row_height = min(available / total_rows, 0.08)
    else:
        row_height = 0.04

    # Compute column x-positions
    left_margin = 0.02
    right_margin = 0.98
    table_width = right_margin - left_margin
    col_starts: list[float] = []
    cumulative = 0.0
    for w in col_widths:
        col_starts.append(left_margin + cumulative * table_width)
        cumulative += w

    def _text_x(col_idx: int) -> float:
        align = col_alignments[col_idx] if col_idx < len(col_alignments) else "left"
        start = col_starts[col_idx]
        end = col_starts[col_idx + 1] if col_idx + 1 < len(col_starts) else right_margin
        if align == "right":
            return end - 0.01
        if align == "center":
            return (start + end) / 2
        return start + 0.01

    # Figure reference
    if figure_ref:
        ax.text(
            0.5,
            0.97,
            figure_ref,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=style.fontsize + 1,
            fontweight="bold",
        )

    row_idx = 0

    # Header
    if has_header and col_labels is not None:
        y = top_margin - row_idx * row_height
        for ci, label in enumerate(col_labels):
            align = col_alignments[ci] if ci < len(col_alignments) else "left"
            ax.text(
                _text_x(ci),
                y,
                str(label),
                transform=ax.transAxes,
                ha=align,
                va="baseline",
                fontsize=style.fontsize,
                fontweight=style.header_fontweight,
            )
        # Header rule
        if style.show_header_rule:
            rule_y = y - row_height * 0.85
            line = Line2D(
                [left_margin - 0.02, right_margin + 0.02],
                [rule_y, rule_y],
                transform=ax.transAxes,
                color=style.rule_color,
                linewidth=style.rule_width,
                clip_on=False,
            )
            ax.add_artist(line)
        row_idx += 1

    # Data rows
    for di, row in enumerate(data):
        y = top_margin - row_idx * row_height
        is_summary = summary_row_start is not None and di >= summary_row_start

        # Zebra shading
        if style.zebra_shade and di % 2 == 1:
            rect = Rectangle(
                (left_margin - 0.02, y - row_height * 0.5),
                right_margin - left_margin + 0.04,
                row_height,
                transform=ax.transAxes,
                facecolor=style.zebra_shade,
                edgecolor="none",
                clip_on=False,
            )
            ax.add_patch(rect)

        for ci, cell in enumerate(row):
            align = col_alignments[ci] if ci < len(col_alignments) else "left"
            weight = "bold" if is_summary else "normal"
            ax.text(
                _text_x(ci),
                y,
                str(cell),
                transform=ax.transAxes,
                ha=align,
                va="top",
                fontsize=style.fontsize,
                fontweight=weight,
            )
        row_idx += 1

    # Footer rule
    if style.show_footer_rule and nrows > 0:
        y = top_margin - row_idx * row_height + row_height * 0.4
        footer_line = Line2D(
            [left_margin - 0.02, right_margin + 0.02],
            [y, y],
            transform=ax.transAxes,
            color=style.rule_color,
            linewidth=style.rule_width,
            clip_on=False,
        )
        ax.add_artist(footer_line)


def draw_stats_panel(
    ax: Axes,
    stats: Mapping[str, str | float | int],
    title: str | None = None,
    style: TableStyle | None = None,
) -> None:
    """Render a two-column statistics panel (metric name | value).

    Parameters
    ----------
    ax : Axes
        Target axes (will be cleared and turned off).
    stats : dict[str, str | float | int]
        Mapping of metric name to value.
    title : str | None
        Optional panel title.
    style : TableStyle | None
        Visual configuration.
    """
    if style is None:
        style = TableStyle()

    # Convert to table data
    data: list[list[str]] = []
    for key, val in stats.items():
        if isinstance(val, float):
            data.append([key, f"{val:.4f}"])
        else:
            data.append([key, str(val)])

    col_labels = None
    figure_ref = title

    draw_table(
        ax,
        data,
        col_labels=col_labels,
        col_widths=[0.55, 0.45],
        col_alignments=["left", "right"],
        style=style,
        figure_ref=figure_ref,
    )
