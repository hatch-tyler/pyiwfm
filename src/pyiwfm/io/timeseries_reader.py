"""
Convenience wrapper for reading IWFM time-series data files.

Provides a simple ``read_iwfm_timeseries()`` function that returns
structured data from any IWFM ASCII time-series file (precipitation,
evapotranspiration, return flow fractions, crop coefficients, etc.).

Delegates to :class:`~pyiwfm.io.timeseries_ascii.TimeSeriesReader`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass
class IWFMTimeSeriesData:
    """Parsed contents of an IWFM time-series data file.

    Attributes:
        n_columns: Number of data columns (NDATA).
        dates: List of datetime objects for each timestep.
        data: Array of shape ``(n_timesteps, n_columns)``.
        factor: Unit conversion factor (FACTOR).
        time_unit: Time unit string from file header, if present.
    """

    n_columns: int = 0
    dates: list[datetime] = field(default_factory=list)
    data: NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 0), dtype=np.float64))
    factor: float = 1.0
    time_unit: str = ""


def read_iwfm_timeseries(filepath: Path | str) -> IWFMTimeSeriesData:
    """Read an IWFM ASCII time-series file.

    Args:
        filepath: Path to the time-series file.

    Returns:
        Structured time-series data with dates, values, and metadata.
    """
    from pyiwfm.io.timeseries_ascii import TimeSeriesReader

    reader = TimeSeriesReader()
    times, values, config = reader.read(filepath)

    return IWFMTimeSeriesData(
        n_columns=config.n_columns,
        dates=times,
        data=values,
        factor=config.factor,
        time_unit=getattr(config, "time_unit", ""),
    )
