"""Shared plotting utilities for pyiwfm visualization modules."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, TypeVar

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for saving

from collections.abc import Callable  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

# ---------------------------------------------------------------------------
# Style paths and decorator
# ---------------------------------------------------------------------------

_STYLES_DIR = Path(__file__).parent / "styles"
SPATIAL_STYLE = str(_STYLES_DIR / "pyiwfm-spatial.mplstyle")
CHART_STYLE = str(_STYLES_DIR / "pyiwfm-chart.mplstyle")
PUBLICATION_STYLE = str(_STYLES_DIR / "pyiwfm-publication.mplstyle")

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
