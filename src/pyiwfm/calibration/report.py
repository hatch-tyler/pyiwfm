"""Multi-page calibration report PDF generator.

Generates a filtered, multi-page PDF report from a residuals DataFrame.
Each combination of filter criteria (subregion, layer, date range,
screen type) produces its own page(s) with 1:1 plots, CDF plots,
optional spatial bias maps, and a statistics summary panel.

Uses the :class:`~pyiwfm.visualization.report.ReportBuilder` framework
for consistent page layout, headers/footers, and publication quality.

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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from pyiwfm.calibration.residuals import filter_residuals  # noqa: E402
from pyiwfm.visualization.report._page_layout import ReportConfig, ReportPage  # noqa: E402
from pyiwfm.visualization.report._report_builder import ReportBuilder  # noqa: E402
from pyiwfm.visualization.report._table import draw_stats_panel  # noqa: E402

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


def _format_stats(
    obs: NDArray[np.float64],
    sim: NDArray[np.float64],
    res: NDArray[np.float64],
    units: str,
) -> dict[str, str]:
    """Build a statistics dict for the stats panel."""
    from pyiwfm.comparison.metrics import (
        correlation_coefficient,
        index_of_agreement,
        nash_sutcliffe,
        rmse,
    )

    n = len(obs)
    u = f" {units}" if units else ""
    stats: dict[str, str] = {"N": f"{n:,}"}
    if n > 0:
        stats[f"Mean Bias{u}"] = f"{np.mean(res):.2f}"
        stats[f"RMSE{u}"] = f"{rmse(obs, sim):.2f}"
        if n > 1:
            stats[f"Std(resid){u}"] = f"{np.std(res, ddof=1):.2f}"
            stats["NSE"] = f"{nash_sutcliffe(obs, sim):.3f}"
            stats["d-index"] = f"{index_of_agreement(obs, sim):.3f}"
            stats["r"] = f"{correlation_coefficient(obs, sim):.3f}"
    return stats


def _render_page(
    page: ReportPage,
    df: pd.DataFrame,
    title: str,
    config: CalibrationReportConfig,
    grid: AppGrid | None,
) -> None:
    """Render a single calibration page using ReportPage."""
    from pyiwfm.visualization.plot_calibration import (
        plot_one_to_one,
        plot_residual_cdf,
    )

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

    # Layout using printable-area gridspec
    if n == 3:
        gs = page.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        plot_axes = [
            page.fig.add_subplot(gs[0, 0]),
            page.fig.add_subplot(gs[0, 1]),
            page.fig.add_subplot(gs[1, 0]),
        ]
        ax_stats = page.fig.add_subplot(gs[1, 1])
    elif n == 2:
        gs = page.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.3, wspace=0.3)
        plot_axes = [
            page.fig.add_subplot(gs[0, 0]),
            page.fig.add_subplot(gs[0, 1]),
        ]
        ax_stats = page.fig.add_subplot(gs[1, :])
    else:
        gs = page.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
        plot_axes = [page.fig.add_subplot(gs[0, 0])]
        ax_stats = page.fig.add_subplot(gs[1, 0])

    page.fig.suptitle(
        page_title,
        fontsize=11,
        fontweight="bold",
        y=page.config.margins.printable_bottom + page.config.margins.printable_height_frac + 0.01,
    )

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

            assert grid is not None
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

    # Statistics panel
    stats = _format_stats(obs, sim, residuals, config.units)
    draw_stats_panel(ax_stats, stats, title="Calibration Statistics")


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

    report_config = ReportConfig(
        title="Calibration Report",
        author="pyiwfm",
        dpi=config.dpi,
        units_length=config.units if config.units else "ft",
    )
    builder = ReportBuilder(report_config, output_path)

    # Empty DataFrame: single blank page
    if len(residuals_df) == 0:

        def _empty_page(page: ReportPage) -> None:
            ax = page.add_axes(0.1, 0.3, 0.8, 0.4)
            ax.text(
                0.5,
                0.5,
                "No observations",
                ha="center",
                va="center",
                fontsize=14,
                transform=ax.transAxes,
            )
            ax.set_axis_off()

        builder.add_page(_empty_page, name="Empty")
        return builder.build(progress=False)

    # All observations page
    def _all_obs(page: ReportPage) -> None:
        _render_page(page, residuals_df, "All Observations", config, grid)

    builder.add_page(_all_obs, name="All Observations")

    # Per-subregion pages
    if config.filter_by_subregion and "subregion" in residuals_df.columns:
        for sr in sorted(residuals_df["subregion"].dropna().unique()):
            subset = filter_residuals(residuals_df, subregions=[sr])

            def _sr_page(page: ReportPage, _sr: object = sr, _sub: pd.DataFrame = subset) -> None:
                _render_page(page, _sub, f"Subregion {_sr}", config, grid)

            builder.add_page(_sr_page, name=f"Subregion {sr}")

    # Per-layer pages
    if config.filter_by_layer and "layer" in residuals_df.columns:
        for lyr in sorted(residuals_df["layer"].dropna().unique()):
            subset = filter_residuals(residuals_df, layers=[lyr])

            def _lyr_page(
                page: ReportPage, _lyr: object = lyr, _sub: pd.DataFrame = subset
            ) -> None:
                _render_page(page, _sub, f"Layer {_lyr}", config, grid)

            builder.add_page(_lyr_page, name=f"Layer {lyr}")

    # Per-date-range pages
    if config.filter_by_date_range:
        for start, end in config.filter_by_date_range:
            subset = filter_residuals(residuals_df, date_range=(start, end))
            label = f"{start:%Y-%m-%d} to {end:%Y-%m-%d}"

            def _dr_page(
                page: ReportPage, _label: str = label, _sub: pd.DataFrame = subset
            ) -> None:
                _render_page(page, _sub, f"Period: {_label}", config, grid)

            builder.add_page(_dr_page, name=f"Period: {label}")

    # Per-screen-type pages
    if config.filter_by_screen_type and "screen_type" in residuals_df.columns:
        for st in sorted(residuals_df["screen_type"].dropna().unique()):
            subset = filter_residuals(residuals_df, screen_types=[st])

            def _st_page(page: ReportPage, _st: object = st, _sub: pd.DataFrame = subset) -> None:
                _render_page(page, _sub, f"Screen Type: {_st}", config, grid)

            builder.add_page(_st_page, name=f"Screen Type: {st}")

    return builder.build(progress=False)
