"""
Time series classes for IWFM model data.

This module provides classes for representing time steps and time series
data used throughout IWFM models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Iterator, Sequence

import numpy as np
from numpy.typing import NDArray


class TimeUnit(Enum):
    """Time units supported by IWFM."""

    MINUTE = "MIN"
    HOUR = "HOUR"
    DAY = "DAY"
    WEEK = "WEEK"
    MONTH = "MON"
    YEAR = "YEAR"

    @classmethod
    def from_string(cls, s: str) -> "TimeUnit":
        """Parse a time unit from string.

        Accepts plain unit names (``DAY``, ``MON``) as well as IWFM-style
        count+unit tokens like ``1MON``, ``1DAY``, ``1HOUR``.
        """
        s_upper = s.upper().strip()

        # Strip leading digits (IWFM often writes "1MON", "1DAY", etc.)
        stripped = s_upper.lstrip("0123456789")
        if not stripped:
            stripped = s_upper

        mapping = {
            "MIN": cls.MINUTE,
            "MINUTE": cls.MINUTE,
            "MINUTES": cls.MINUTE,
            "HOUR": cls.HOUR,
            "HR": cls.HOUR,
            "HOURS": cls.HOUR,
            "DAY": cls.DAY,
            "DAYS": cls.DAY,
            "WEEK": cls.WEEK,
            "WEEKS": cls.WEEK,
            "MON": cls.MONTH,
            "MONTH": cls.MONTH,
            "MONTHS": cls.MONTH,
            "YEAR": cls.YEAR,
            "YR": cls.YEAR,
            "YEARS": cls.YEAR,
        }
        if stripped in mapping:
            return mapping[stripped]
        if s_upper in mapping:
            return mapping[s_upper]
        raise ValueError(f"Unknown time unit: '{s}'")

    def to_timedelta(self, n: int = 1) -> timedelta:
        """
        Convert to a timedelta.

        Note: Month and year are approximated.
        """
        if self == TimeUnit.MINUTE:
            return timedelta(minutes=n)
        elif self == TimeUnit.HOUR:
            return timedelta(hours=n)
        elif self == TimeUnit.DAY:
            return timedelta(days=n)
        elif self == TimeUnit.WEEK:
            return timedelta(weeks=n)
        elif self == TimeUnit.MONTH:
            return timedelta(days=30 * n)  # Approximation
        elif self == TimeUnit.YEAR:
            return timedelta(days=365 * n)  # Approximation
        else:
            raise ValueError(f"Unknown time unit: {self}")


@dataclass
class TimeStep:
    """
    A single time step in an IWFM simulation.

    Attributes:
        start: Start datetime of the time step
        end: End datetime of the time step
        index: Time step index (0-based)
    """

    start: datetime
    end: datetime
    index: int = 0

    @property
    def duration(self) -> timedelta:
        """Return the duration of this time step."""
        return self.end - self.start

    @property
    def midpoint(self) -> datetime:
        """Return the midpoint datetime of this time step."""
        return self.start + self.duration / 2

    def __repr__(self) -> str:
        return f"TimeStep({self.start.isoformat()} to {self.end.isoformat()})"


@dataclass
class SimulationPeriod:
    """
    Defines the simulation time period for an IWFM model.

    Attributes:
        start: Simulation start datetime
        end: Simulation end datetime
        time_step_length: Length of each time step
        time_step_unit: Unit of the time step length
    """

    start: datetime
    end: datetime
    time_step_length: int
    time_step_unit: TimeUnit

    @property
    def duration(self) -> timedelta:
        """Return total simulation duration."""
        return self.end - self.start

    @property
    def time_step_delta(self) -> timedelta:
        """Return the time step as a timedelta."""
        return self.time_step_unit.to_timedelta(self.time_step_length)

    @property
    def n_time_steps(self) -> int:
        """Return approximate number of time steps."""
        delta = self.time_step_delta
        if delta.total_seconds() == 0:
            return 0
        return int(self.duration.total_seconds() / delta.total_seconds())

    def iter_time_steps(self) -> Iterator[TimeStep]:
        """Iterate over all time steps in the simulation period."""
        delta = self.time_step_delta
        current = self.start
        index = 0

        while current < self.end:
            next_time = current + delta
            if next_time > self.end:
                next_time = self.end
            yield TimeStep(start=current, end=next_time, index=index)
            current = next_time
            index += 1

    def get_time_step(self, index: int) -> TimeStep:
        """Get a specific time step by index."""
        delta = self.time_step_delta
        start = self.start + delta * index
        end = start + delta
        if end > self.end:
            end = self.end
        return TimeStep(start=start, end=end, index=index)

    def __repr__(self) -> str:
        return (
            f"SimulationPeriod({self.start.isoformat()} to {self.end.isoformat()}, "
            f"dt={self.time_step_length} {self.time_step_unit.value})"
        )


@dataclass
class TimeSeries:
    """
    A time series of values.

    Attributes:
        times: Array of datetime objects or timestamps
        values: Array of values (can be 1D or 2D)
        name: Name/identifier for this time series
        units: Units of the values
        location: Location identifier (e.g., node ID, element ID)
    """

    times: NDArray[np.datetime64]
    values: NDArray[np.float64]
    name: str = ""
    units: str = ""
    location: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate time series data."""
        if len(self.times) != self.values.shape[0]:
            raise ValueError(
                f"Time array length ({len(self.times)}) doesn't match "
                f"value array first dimension ({self.values.shape[0]})"
            )

    @property
    def n_times(self) -> int:
        """Return number of time points."""
        return len(self.times)

    @property
    def start_time(self) -> np.datetime64:
        """Return start time."""
        return self.times[0]

    @property
    def end_time(self) -> np.datetime64:
        """Return end time."""
        return self.times[-1]

    @classmethod
    def from_datetimes(
        cls,
        times: Sequence[datetime],
        values: NDArray[np.float64],
        **kwargs,
    ) -> "TimeSeries":
        """
        Create a TimeSeries from Python datetime objects.

        Args:
            times: Sequence of datetime objects
            values: Array of values
            **kwargs: Additional attributes (name, units, location, metadata)
        """
        np_times = np.array(times, dtype="datetime64[s]")
        return cls(times=np_times, values=values, **kwargs)

    def to_dataframe(self):
        """
        Convert to a pandas DataFrame.

        Returns:
            DataFrame with times as index
        """
        import pandas as pd

        if self.values.ndim == 1:
            return pd.DataFrame(
                {self.name or "value": self.values},
                index=pd.DatetimeIndex(self.times),
            )
        else:
            columns = [f"{self.name}_{i}" for i in range(self.values.shape[1])]
            return pd.DataFrame(
                self.values,
                index=pd.DatetimeIndex(self.times),
                columns=columns,
            )

    def resample(self, freq: str) -> "TimeSeries":
        """
        Resample the time series to a new frequency.

        Args:
            freq: Pandas frequency string (e.g., 'D' for daily, 'M' for monthly)

        Returns:
            New resampled TimeSeries
        """
        df = self.to_dataframe()
        resampled = df.resample(freq).mean()

        return TimeSeries(
            times=np.array(resampled.index.values, dtype="datetime64[s]"),
            values=resampled.values,
            name=self.name,
            units=self.units,
            location=self.location,
            metadata=self.metadata.copy(),
        )

    def slice_time(
        self, start: datetime | None = None, end: datetime | None = None
    ) -> "TimeSeries":
        """
        Slice the time series to a time range.

        Args:
            start: Start datetime (inclusive)
            end: End datetime (inclusive)

        Returns:
            New sliced TimeSeries
        """
        mask = np.ones(len(self.times), dtype=bool)

        if start is not None:
            start_np = np.datetime64(start)
            mask &= self.times >= start_np

        if end is not None:
            end_np = np.datetime64(end)
            mask &= self.times <= end_np

        return TimeSeries(
            times=self.times[mask],
            values=self.values[mask],
            name=self.name,
            units=self.units,
            location=self.location,
            metadata=self.metadata.copy(),
        )

    def __getitem__(self, key: int | slice) -> NDArray[np.float64]:
        """Get values by index."""
        return self.values[key]

    def __len__(self) -> int:
        return len(self.times)

    def __repr__(self) -> str:
        return (
            f"TimeSeries(name='{self.name}', n_times={self.n_times}, "
            f"start={self.start_time}, end={self.end_time})"
        )


@dataclass
class TimeSeriesCollection:
    """
    A collection of related time series (e.g., heads at multiple nodes).

    Attributes:
        series: Dictionary mapping location ID to TimeSeries
        name: Name of the collection
        variable: Variable name (e.g., 'head', 'flow')
    """

    series: dict[str, TimeSeries] = field(default_factory=dict)
    name: str = ""
    variable: str = ""

    def add(self, ts: TimeSeries) -> None:
        """Add a time series to the collection."""
        key = ts.location or ts.name
        self.series[key] = ts

    def get(self, location: str) -> TimeSeries | None:
        """Get a time series by location."""
        return self.series.get(location)

    @property
    def locations(self) -> list[str]:
        """Return list of all locations."""
        return list(self.series.keys())

    def __len__(self) -> int:
        return len(self.series)

    def __iter__(self) -> Iterator[TimeSeries]:
        return iter(self.series.values())

    def __getitem__(self, key: str) -> TimeSeries:
        return self.series[key]

    def __repr__(self) -> str:
        return f"TimeSeriesCollection(name='{self.name}', n_series={len(self)})"
