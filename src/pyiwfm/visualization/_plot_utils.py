"""Shared plotting utilities for pyiwfm visualization modules."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, TypeVar

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for saving

from collections.abc import Callable  # noqa: E402

import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

# ---------------------------------------------------------------------------
# Style paths and decorator
# ---------------------------------------------------------------------------

_STYLES_DIR = Path(__file__).parent / "styles"
SPATIAL_STYLE = str(_STYLES_DIR / "pyiwfm-spatial.mplstyle")
CHART_STYLE = str(_STYLES_DIR / "pyiwfm-chart.mplstyle")
PUBLICATION_STYLE = str(_STYLES_DIR / "pyiwfm-publication.mplstyle")
REPORT_STYLE = str(_STYLES_DIR / "pyiwfm-report.mplstyle")

# ---------------------------------------------------------------------------
# Colorblind-safe palettes
# ---------------------------------------------------------------------------

# Okabe-Ito colorblind-safe palette (8 colors)
OKABE_ITO = [
    "#0072B2",  # Blue
    "#E69F00",  # Orange
    "#009E73",  # Bluish Green
    "#D55E00",  # Vermillion
    "#CC79A7",  # Reddish Purple
    "#56B4E9",  # Sky Blue
    "#F0E442",  # Yellow
    "#000000",  # Black
]

# Budget-specific palettes (diverging: blues for inflows, reds for outflows)
BUDGET_INFLOW_COLORS = [
    "#0072B2",  # Blue (Okabe-Ito)
    "#56B4E9",  # Sky Blue
    "#009E73",  # Bluish Green
    "#4393c3",  # Steel Blue
    "#67a9cf",  # Light Steel
    "#92c5de",  # Powder Blue
    "#a6dba0",  # Sage Green
    "#7eb5d6",  # Muted Blue (was Ice Blue #d1e5f0)
]
BUDGET_OUTFLOW_COLORS = [
    "#D55E00",  # Vermillion (Okabe-Ito)
    "#E69F00",  # Orange
    "#CC79A7",  # Reddish Purple
    "#d6604d",  # Salmon
    "#f4a582",  # Peach
    "#fdb863",  # Light Orange
    "#e8a94f",  # Gold (was Cream #fee0b6)
    "#c51b7d",  # Magenta
]

# Unified budget color palette — named components
BUDGET_COLORS: dict[str, str] = {
    # GW inflows (blues/greens)
    "DEEP_PERC": "#4393c3",
    "RECHARGE": "#2166ac",
    "BOUNDARY_INFLOW": "#92c5de",
    "SUBSIDENCE": "#7eb5d6",
    "SUBSURF_IRRIGATION": "#67a9cf",
    "GW_RETURN_FLOW": "#1b7837",
    "NET_SUBSURF_INFLOW": "#a6dba0",
    # GW outflows (reds/oranges)
    "PUMPING": "#d6604d",
    "TILE_DRAINS": "#b2182b",
    "FLOW_TO_ROOTZONE": "#f4a582",
    # GW sign-dependent
    "GAIN_FROM_STRM": "#74add1",
    "GAIN_FROM_LAKE": "#abd9e9",
    # Stream inflows (blues/greens)
    "TRIB_INFLOW": "#2166ac",
    "TILE_DRN": "#4393c3",
    "GW_RTRN_FLOW": "#1b7837",
    "RUNOFF": "#67a9cf",
    "RETURN_FLOW": "#92c5de",
    "DIV_SPILL": "#d1e5f0",
    "POND_DRN": "#a6dba0",
    # Stream outflows (reds/oranges)
    "GAIN_FRM_GW_OUTM": "#d6604d",
    "RIPARIAN_ET": "#b2182b",
    "SURFACE_EVAP": "#f4a582",
    "DIVERSION": "#e08214",
    "BYPASS": "#fdb863",
    # Stream sign-dependent
    "GAIN_FRM_GW_INM": "#74add1",
    "GAIN_FRM_LAKE": "#abd9e9",
    # Root Zone (shared across AG/URB/NRV after prefix strip)
    "PRECIP": "#2166ac",
    "PRM_H2O": "#4393c3",
    "SR_INFLOW": "#67a9cf",
    "RE-USE": "#92c5de",
    "GW_INFLOW": "#1b7837",
    "OTHER_INFLOW": "#a6dba0",
    "OTHER_INFLO": "#a6dba0",
    "GAIN_EXP": "#d1e5f0",
    "ET": "#b2182b",
    "PERC": "#d6604d",
    "DRAIN": "#f4a582",
    "NT_RTRN_FLOW": "#e08214",
    "NT_RTRN_FLO": "#e08214",
    "STRM_ET": "#fdb863",
    # LWU supply (blues)
    "DELIVERY": "#4393c3",
    # LWU demand (reds)
    "SUP_REQ": "#b2182b",
    "ETAW": "#d6604d",
    "EFF_PRECIP": "#f4a582",
    "ET_GW": "#e08214",
    "ET_OTH": "#fdb863",
    "SHORTAGE": "#fee0b6",
    # Storage
    "Storage Change": "#ffd700",
}

_F = TypeVar("_F", bound=Callable[..., Any])


def _with_style(style_path: str) -> Callable[[_F], _F]:
    """Decorator that wraps a plotting function in ``plt.style.context``."""

    def decorator(func: _F) -> _F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with plt.style.context(style_path):
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _outlined_text(
    ax: Axes,
    x: float,
    y: float,
    text: str,
    fontsize: float = 7,
    color: str = "white",
    **kwargs: Any,
) -> Any:
    """White text with thin dark outline — readable on any background."""
    return ax.text(
        x,
        y,
        text,
        fontsize=fontsize,
        color=color,
        fontweight="bold",
        ha="center",
        va="center",
        path_effects=[pe.withStroke(linewidth=2, foreground="#333333")],
        **kwargs,
    )


def _format_thousands(ax: Axes) -> None:
    """Apply thousands-separator formatting to both axes."""
    from matplotlib.ticker import FuncFormatter

    def _thousands_fmt(x: float, pos: int) -> str:
        return f"{x:,.0f}"

    ax.xaxis.set_major_formatter(FuncFormatter(_thousands_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(_thousands_fmt))


def _rotate_date_labels(ax: Axes) -> None:
    """Rotate date x-tick labels (replacement for fig.autofmt_xdate())."""
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment("right")
