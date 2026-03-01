"""Multi-page calibration report PDF generator.

Generates a filtered, multi-page PDF report from a residuals DataFrame.
Each combination of filter criteria (subregion, layer, date range,
screen type) produces its own page(s) with 1:1 plots, CDF plots,
optional spatial bias maps, and a statistics summary panel.

Uses the ``pyiwfm-publication.mplstyle`` stylesheet for 300 DPI,
serif fonts, and journal-ready defaults.

Classes
-------
- :class:`CalibrationReportConfig` — Configuration for report contents.

Functions
---------
- :func:`generate_calibration_report` — Generate the PDF report.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from pyiwfm.calibration.residuals import filter_residuals  # noqa: E402
from pyiwfm.visualization._plot_utils import PUBLICATION_STYLE  # noqa: E402
from pyiwfm.visualization.plot_calibration import (  # noqa: E402
    plot_one_to_one,
    plot_residual_cdf,
)

if TYPE_CHECKING:
    from pyiwfm.core.mesh import AppGrid

# US Letter dimensions in inches
_PAGE_WIDTH = 8.5
_PAGE_HEIGHT = 11.0


@dataclass
class CalibrationReportConfig:
    """Configuration for the calibration report.

    Attributes
    ----------
    filter_by_subregion : bool
        Generate per-subregion pages.
    filter_by_layer : bool
        Generate per-layer pages.
    filter_by_date_range : list[tuple[datetime, datetime]] | None
        Optional list of date ranges to generate pages for.
    filter_by_screen_type : bool
        Generate per-screen-type pages.
    include_one_to_one : bool
        Include 1:1 scatter plots on each page.
    include_cdf : bool
        Include cumulative frequency (CDF) plots on each page.
    include_spatial_bias : bool
        Include spatial bias maps on each page (requires *grid*
        and ``x``, ``y`` columns in the residuals DataFrame).
    units : str
        Unit label for axes and statistics (e.g. ``"ft"``, ``"m"``).
    dpi : int
        Output resolution in dots per inch.
    """

    filter_by_subregion: bool = True
    filter_by_layer: bool = True
    filter_by_date_range: list[tuple[datetime, datetime]] | None = None
    filter_by_screen_type: bool = True
    include_one_to_one: bool = True
    include_cdf: bool = True
    include_spatial_bias: bool = True
    units: str = ""
    dpi: int = 300


def _format_stats_text(
    obs: NDArray[np.float64],
    sim: NDArray[np.float64],
    res: NDArray[np.float64],
    units: str,
) -> str:
    """Build a multi-line statistics summary string."""
    from pyiwfm.comparison.metrics import (
        correlation_coefficient,
        index_of_agreement,
        nash_sutcliffe,
        rmse,
    )

    n = len(obs)
    u = f" {units}" if units else ""
    lines = [f"N = {n:,}"]
    if n > 0:
        lines.append(f"Mean Bias  = {np.mean(res):>10.2f}{u}")
        lines.append(f"RMSE       = {rmse(obs, sim):>10.2f}{u}")
        if n > 1:
            lines.append(f"Std(resid) = {np.std(res, ddof=1):>10.2f}{u}")
            lines.append(f"NSE        = {nash_sutcliffe(obs, sim):>10.3f}")
            lines.append(f"d-index    = {index_of_agreement(obs, sim):>10.3f}")
            lines.append(f"r          = {correlation_coefficient(obs, sim):>10.3f}")
    return "\n".join(lines)


def _add_page(
    pdf: PdfPages,
    df: pd.DataFrame,
    title: str,
    config: CalibrationReportConfig,
    grid: AppGrid | None,
) -> None:
    """Render a single letter-size page of the report."""
    has_oto = config.include_one_to_one
    has_cdf = config.include_cdf
    has_spatial = (
        config.include_spatial_bias and grid is not None and "x" in df.columns and "y" in df.columns
    )

    plot_kinds: list[str] = []
    if has_oto:
        plot_kinds.append("oto")
    if has_cdf:
        plot_kinds.append("cdf")
    if has_spatial:
        plot_kinds.append("spatial")

    n = len(plot_kinds)
    if n == 0 or len(df) == 0:
        return

    obs = df["observed"].to_numpy(dtype=np.float64)
    sim = df["simulated"].to_numpy(dtype=np.float64)
    residuals = df["residual"].to_numpy(dtype=np.float64)

    page_title = f"{title}  (N = {len(df):,})"

    # Create letter-size figure (constrained_layout from publication style)
    fig = plt.figure(figsize=(_PAGE_WIDTH, _PAGE_HEIGHT))

    # Layout: plots on top row(s), statistics panel at bottom-right or below
    if n == 3:
        # 2x2: top = 1:1 + CDF, bottom-left = spatial, bottom-right = stats
        gs = fig.add_gridspec(2, 2)
        plot_axes = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]),
        ]
        ax_stats = fig.add_subplot(gs[1, 1])
    elif n == 2:
        # Top row: two plots; bottom: stats spanning full width
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])
        plot_axes = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
        ]
        ax_stats = fig.add_subplot(gs[1, :])
    else:
        # Single plot on top, stats below
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
        plot_axes = [fig.add_subplot(gs[0, 0])]
        ax_stats = fig.add_subplot(gs[1, 0])

    fig.suptitle(page_title, fontsize=13, fontweight="bold")

    # Render plots
    for kind, ax in zip(plot_kinds, plot_axes, strict=True):
        if kind == "oto":
            plot_one_to_one(
                obs,
                sim,
                ax=ax,
                title="Observed vs Simulated",
                units=config.units,
                show_metrics=False,
            )
        elif kind == "cdf":
            plot_residual_cdf(residuals, ax=ax, title="Residual CDF")
        elif kind == "spatial":
            from pyiwfm.calibration.residuals import mean_residuals
            from pyiwfm.visualization.plot_calibration import plot_spatial_bias

            assert grid is not None  # guaranteed by has_spatial check
            mean_df = mean_residuals(df)
            well_xy = df.groupby("well_id")[["x", "y"]].first().reset_index()
            merged = mean_df.merge(well_xy, on="well_id")
            plot_spatial_bias(
                grid,
                merged["x"].to_numpy(dtype=np.float64),
                merged["y"].to_numpy(dtype=np.float64),
                merged["mean_residual"].to_numpy(dtype=np.float64),
                ax=ax,
                title="Spatial Bias",
                units=config.units,
            )

    # Statistics summary panel
    stats_text = _format_stats_text(obs, sim, residuals, config.units)
    ax_stats.set_axis_off()
    ax_stats.text(
        0.5,
        0.5,
        stats_text,
        transform=ax_stats.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        family="monospace",
        bbox={
            "boxstyle": "round,pad=0.6",
            "facecolor": "#f5f5f5",
            "edgecolor": "lightgray",
            "alpha": 0.9,
        },
    )

    pdf.savefig(fig, dpi=config.dpi)
    plt.close(fig)


def generate_calibration_report(
    residuals_df: pd.DataFrame,
    output_path: str | Path,
    config: CalibrationReportConfig | None = None,
    grid: AppGrid | None = None,
) -> Path:
    """Generate a multi-page calibration report PDF.

    Parameters
    ----------
    residuals_df : pd.DataFrame
        Residuals DataFrame (output of
        :func:`~pyiwfm.calibration.residuals.compute_residuals`).
        Expected columns: ``well_id``, ``datetime``, ``observed``,
        ``simulated``, ``residual``.  Optional metadata: ``layer``,
        ``subregion``, ``screen_type``, ``x``, ``y``.
    output_path : str or Path
        Destination PDF path.
    config : CalibrationReportConfig | None
        Report configuration.  Uses defaults if ``None``.
    grid : AppGrid | None
        Model mesh, required for spatial bias plots.

    Returns
    -------
    Path
        Path to the written PDF.
    """
    if config is None:
        config = CalibrationReportConfig()

    output_path = Path(output_path)

    with plt.style.context(PUBLICATION_STYLE):
        with PdfPages(
            output_path,
            metadata={
                "Title": "Calibration Report",
                "Author": "pyiwfm",
                "CreationDate": datetime.now(),
            },
        ) as pdf:
            # Empty DataFrame: write a blank page so the PDF is valid
            if len(residuals_df) == 0:
                fig, ax = plt.subplots(figsize=(_PAGE_WIDTH, _PAGE_HEIGHT))
                ax.text(
                    0.5,
                    0.5,
                    "No observations",
                    ha="center",
                    va="center",
                    fontsize=14,
                )
                ax.set_axis_off()
                pdf.savefig(fig, dpi=config.dpi)
                plt.close(fig)
                return output_path

            _add_page(pdf, residuals_df, "All Observations", config, grid)

            # Per-subregion pages
            if config.filter_by_subregion and "subregion" in residuals_df.columns:
                for sr in sorted(residuals_df["subregion"].dropna().unique()):
                    subset = filter_residuals(residuals_df, subregions=[sr])
                    _add_page(pdf, subset, f"Subregion {sr}", config, grid)

            # Per-layer pages
            if config.filter_by_layer and "layer" in residuals_df.columns:
                for lyr in sorted(residuals_df["layer"].dropna().unique()):
                    subset = filter_residuals(residuals_df, layers=[lyr])
                    _add_page(pdf, subset, f"Layer {lyr}", config, grid)

            # Per-date-range pages
            if config.filter_by_date_range:
                for start, end in config.filter_by_date_range:
                    subset = filter_residuals(residuals_df, date_range=(start, end))
                    label = f"{start:%Y-%m-%d} to {end:%Y-%m-%d}"
                    _add_page(pdf, subset, f"Period: {label}", config, grid)

            # Per-screen-type pages
            if config.filter_by_screen_type and "screen_type" in residuals_df.columns:
                for st in sorted(residuals_df["screen_type"].dropna().unique()):
                    subset = filter_residuals(residuals_df, screen_types=[st])
                    _add_page(pdf, subset, f"Screen Type: {st}", config, grid)

    return output_path
