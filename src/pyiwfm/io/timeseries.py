"""
Unified Time Series I/O Infrastructure for IWFM.

This module provides a unified interface for reading and writing time series data
across all supported formats (ASCII, DSS, HDF5). It enables format-agnostic
handling of IWFM time series data with automatic format detection and conversion.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

from pyiwfm.core.timeseries import TimeSeries, TimeSeriesCollection


class TimeSeriesFileType(Enum):
    """Supported time series file formats."""
    ASCII = "ascii"
    DSS = "dss"
    HDF5 = "hdf5"
    BINARY = "binary"


class TimeUnit(Enum):
    """Standard time units for IWFM."""
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


@dataclass
class TimeSeriesMetadata:
    """
    Metadata for a time series file.

    Attributes:
        file_type: Type of the source file
        n_columns: Number of data columns
        column_ids: Identifier for each column (location/entity ID)
        variable_name: Name of the variable (e.g., "Pumping", "Head")
        data_unit: Physical units of the data
        time_unit: Time unit for the series
        conversion_factor: Factor applied during read
        is_rate_data: Whether data represents rates (flow/time)
        recycling_interval: Number of months to recycle (0 = no recycling)
        start_time: Start of time series
        end_time: End of time series
        n_timesteps: Number of timesteps
        source_path: Original file path
    """
    file_type: TimeSeriesFileType = TimeSeriesFileType.ASCII
    n_columns: int = 0
    column_ids: list[str | int] = field(default_factory=list)
    variable_name: str = ""
    data_unit: str = ""
    time_unit: str = ""
    conversion_factor: float = 1.0
    is_rate_data: bool = False
    recycling_interval: int = 0  # 0=no recycle, 12=yearly recycle, etc.
    start_time: datetime | None = None
    end_time: datetime | None = None
    n_timesteps: int = 0
    source_path: Path | None = None


@dataclass
class UnifiedTimeSeriesConfig:
    """
    Configuration for unified time series reading.

    Attributes:
        filepath: Path to the time series file
        file_type: Type of file (auto-detected if None)
        n_columns: Expected number of columns (for validation)
        column_ids: Entity IDs for each column
        data_unit: Expected data unit
        time_unit: Expected time unit
        conversion_factor: Factor to apply to values
        is_rate_data: Whether to normalize by timestep
        recycling_interval: Recycle period (0 = no recycling)
        start_filter: Only read data after this time
        end_filter: Only read data before this time
        dss_pathname: DSS pathname pattern (for DSS files)
        hdf5_dataset: HDF5 dataset path (for HDF5 files)
    """
    filepath: Path
    file_type: TimeSeriesFileType | None = None
    n_columns: int | None = None
    column_ids: list[int] | None = None
    data_unit: str = ""
    time_unit: str = ""
    conversion_factor: float = 1.0
    is_rate_data: bool = False
    recycling_interval: int = 0
    start_filter: datetime | None = None
    end_filter: datetime | None = None
    # Format-specific options
    dss_pathname: str | None = None  # For DSS files
    hdf5_dataset: str | None = None  # For HDF5 files


class BaseTimeSeriesReader(ABC):
    """Abstract base class for time series readers."""

    @abstractmethod
    def read(
        self,
        filepath: Path | str,
        **kwargs: Any,
    ) -> tuple[NDArray[np.datetime64], NDArray[np.float64], TimeSeriesMetadata]:
        """
        Read time series data from file.

        Args:
            filepath: Path to the file
            **kwargs: Format-specific options

        Returns:
            Tuple of (times, values, metadata)
        """
        ...

    @abstractmethod
    def read_metadata(self, filepath: Path | str, **kwargs: Any) -> TimeSeriesMetadata:
        """
        Read only metadata without loading full data.

        Args:
            filepath: Path to the file
            **kwargs: Format-specific options

        Returns:
            TimeSeriesMetadata
        """
        ...


class AsciiTimeSeriesAdapter(BaseTimeSeriesReader):
    """Adapter for ASCII time series files."""

    def read(
        self,
        filepath: Path | str,
        **kwargs: Any,
    ) -> tuple[NDArray[np.datetime64], NDArray[np.float64], TimeSeriesMetadata]:
        """Read ASCII time series file."""
        from pyiwfm.io.timeseries_ascii import TimeSeriesReader

        reader = TimeSeriesReader()
        times, values, config = reader.read(filepath)

        # Convert to numpy datetime64
        np_times = np.array(times, dtype="datetime64[s]")

        metadata = TimeSeriesMetadata(
            file_type=TimeSeriesFileType.ASCII,
            n_columns=config.n_columns,
            column_ids=config.column_ids,
            conversion_factor=config.factor,
            source_path=Path(filepath),
            n_timesteps=len(times),
            start_time=times[0] if times else None,
            end_time=times[-1] if times else None,
        )

        return np_times, values, metadata

    def read_metadata(self, filepath: Path | str, **kwargs: Any) -> TimeSeriesMetadata:
        """Read ASCII file metadata (requires reading header)."""
        # For ASCII, we need to read at least the header to get metadata
        filepath = Path(filepath)

        n_columns = 0
        factor = 1.0
        n_lines = 0
        first_time = None
        last_time = None

        from pyiwfm.io.timeseries_ascii import (
            IWFM_TIMESTAMP_LENGTH,
            _is_comment_line,
            parse_iwfm_timestamp,
        )

        with open(filepath) as f:
            # Read NDATA
            for line in f:
                if _is_comment_line(line):
                    continue
                parts = line.split("/")
                n_columns = int(parts[0].strip())
                break

            # Read FACTOR
            for line in f:
                if _is_comment_line(line):
                    continue
                parts = line.split("/")
                factor = float(parts[0].strip())
                break

            # Count data lines and get first/last timestamps
            for line in f:
                if _is_comment_line(line):
                    continue
                line = line.strip()
                if not line:
                    continue

                ts_str = line[:IWFM_TIMESTAMP_LENGTH].strip()
                dt = parse_iwfm_timestamp(ts_str)

                if first_time is None:
                    first_time = dt
                last_time = dt
                n_lines += 1

        return TimeSeriesMetadata(
            file_type=TimeSeriesFileType.ASCII,
            n_columns=n_columns,
            column_ids=list(range(1, n_columns + 1)),
            conversion_factor=factor,
            source_path=filepath,
            n_timesteps=n_lines,
            start_time=first_time,
            end_time=last_time,
        )


class DssTimeSeriesAdapter(BaseTimeSeriesReader):
    """Adapter for HEC-DSS time series files."""

    def read(
        self,
        filepath: Path | str,
        **kwargs: Any,
    ) -> tuple[NDArray[np.datetime64], NDArray[np.float64], TimeSeriesMetadata]:
        """Read DSS time series file."""
        try:
            from pyiwfm.io.dss import DSSTimeSeriesReader
        except ImportError as e:
            raise ImportError(
                "DSS support requires the bundled HEC-DSS library. "
                "Check that pyiwfm.io.dss is importable."
            ) from e

        pathname = kwargs.get("pathname", "")
        reader = DSSTimeSeriesReader()
        times, values = reader.read(filepath, pathname)

        metadata = TimeSeriesMetadata(
            file_type=TimeSeriesFileType.DSS,
            n_columns=values.shape[1] if values.ndim > 1 else 1,
            source_path=Path(filepath),
            n_timesteps=len(times),
            start_time=times[0].astype(datetime) if len(times) > 0 else None,
            end_time=times[-1].astype(datetime) if len(times) > 0 else None,
        )

        return times, values, metadata

    def read_metadata(self, filepath: Path | str, **kwargs: Any) -> TimeSeriesMetadata:
        """Read DSS file metadata."""
        try:
            from pyiwfm.io.dss import DSSFile
        except ImportError as e:
            raise ImportError(
                "DSS support requires the bundled HEC-DSS library. "
                "Check that pyiwfm.io.dss is importable."
            ) from e

        pathname = kwargs.get("pathname", "")

        with DSSFile(filepath, "r") as dss:
            catalog = dss.catalog()
            n_records = len(catalog)

        return TimeSeriesMetadata(
            file_type=TimeSeriesFileType.DSS,
            n_columns=n_records,
            source_path=Path(filepath),
        )


class Hdf5TimeSeriesAdapter(BaseTimeSeriesReader):
    """Adapter for HDF5 time series files."""

    def read(
        self,
        filepath: Path | str,
        **kwargs: Any,
    ) -> tuple[NDArray[np.datetime64], NDArray[np.float64], TimeSeriesMetadata]:
        """Read HDF5 time series file."""
        dataset_path = kwargs.get("dataset", "/timeseries/data")
        time_path = kwargs.get("time_dataset", "/timeseries/time")

        with h5py.File(filepath, "r") as f:
            if time_path in f:
                times = np.array(f[time_path][:])
                # Convert to datetime64 if necessary
                if times.dtype.kind in ("i", "f"):
                    # Assume Julian dates or epoch seconds
                    times = np.array(times, dtype="datetime64[s]")
            else:
                times = np.array([], dtype="datetime64[s]")

            if dataset_path in f:
                values = np.array(f[dataset_path][:])
            else:
                values = np.array([])

        metadata = TimeSeriesMetadata(
            file_type=TimeSeriesFileType.HDF5,
            n_columns=values.shape[1] if values.ndim > 1 else 1,
            source_path=Path(filepath),
            n_timesteps=len(times),
        )

        return times, values, metadata

    def read_metadata(self, filepath: Path | str, **kwargs: Any) -> TimeSeriesMetadata:
        """Read HDF5 file metadata."""
        dataset_path = kwargs.get("dataset", "/timeseries/data")

        with h5py.File(filepath, "r") as f:
            if dataset_path in f:
                shape = f[dataset_path].shape
                n_columns = shape[1] if len(shape) > 1 else 1
                n_timesteps = shape[0]
            else:
                n_columns = 0
                n_timesteps = 0

        return TimeSeriesMetadata(
            file_type=TimeSeriesFileType.HDF5,
            n_columns=n_columns,
            source_path=Path(filepath),
            n_timesteps=n_timesteps,
        )


class UnifiedTimeSeriesReader:
    """
    Unified reader for all time series formats.

    This class provides a single interface for reading time series data
    from ASCII, DSS, and HDF5 files with automatic format detection and
    consistent output format.

    Example:
        >>> reader = UnifiedTimeSeriesReader()
        >>> config = UnifiedTimeSeriesConfig(filepath=Path("pumping.dat"))
        >>> times, values, metadata = reader.read(config)

        >>> # Or with auto-detection
        >>> times, values, metadata = reader.read_file("pumping.dat")
    """

    def __init__(self) -> None:
        """Initialize the unified reader with format adapters."""
        self._adapters: dict[TimeSeriesFileType, BaseTimeSeriesReader] = {
            TimeSeriesFileType.ASCII: AsciiTimeSeriesAdapter(),
        }

        # HDF5 is always available (h5py is a core dependency)
        self._adapters[TimeSeriesFileType.HDF5] = Hdf5TimeSeriesAdapter()

        # DSS adapter: bundled ctypes library may fail to load on some platforms
        try:
            from pyiwfm.io.dss import HAS_DSS_LIBRARY
            if HAS_DSS_LIBRARY:
                self._adapters[TimeSeriesFileType.DSS] = DssTimeSeriesAdapter()
        except ImportError:
            pass

    def read(
        self, config: UnifiedTimeSeriesConfig
    ) -> tuple[NDArray[np.datetime64], NDArray[np.float64], TimeSeriesMetadata]:
        """
        Read time series data using configuration.

        Args:
            config: UnifiedTimeSeriesConfig with file path and options

        Returns:
            Tuple of (times, values, metadata)
        """
        file_type = config.file_type or self._detect_format(config.filepath)
        adapter = self._get_adapter(file_type)

        # Build kwargs for adapter
        kwargs: dict[str, Any] = {}
        if config.dss_pathname:
            kwargs["pathname"] = config.dss_pathname
        if config.hdf5_dataset:
            kwargs["dataset"] = config.hdf5_dataset

        times, values, metadata = adapter.read(config.filepath, **kwargs)

        # Apply conversion factor
        if config.conversion_factor != 1.0:
            values = values * config.conversion_factor
            metadata.conversion_factor = config.conversion_factor

        # Apply time filter
        if config.start_filter or config.end_filter:
            times, values = self._apply_time_filter(
                times, values, config.start_filter, config.end_filter
            )
            metadata.n_timesteps = len(times)

        # Handle recycling
        if config.recycling_interval > 0:
            metadata.recycling_interval = config.recycling_interval

        return times, values, metadata

    def read_file(
        self,
        filepath: Path | str,
        file_type: TimeSeriesFileType | None = None,
        **kwargs: Any,
    ) -> tuple[NDArray[np.datetime64], NDArray[np.float64], TimeSeriesMetadata]:
        """
        Convenience method to read a time series file.

        Args:
            filepath: Path to the file
            file_type: File type (auto-detected if None)
            **kwargs: Additional options passed to adapter

        Returns:
            Tuple of (times, values, metadata)
        """
        filepath = Path(filepath)
        file_type = file_type or self._detect_format(filepath)
        adapter = self._get_adapter(file_type)
        return adapter.read(filepath, **kwargs)

    def read_to_collection(
        self,
        filepath: Path | str,
        column_ids: list[str] | None = None,
        variable: str = "",
        file_type: TimeSeriesFileType | None = None,
        **kwargs: Any,
    ) -> TimeSeriesCollection:
        """
        Read file and return as TimeSeriesCollection.

        Args:
            filepath: Path to the file
            column_ids: Optional column identifiers
            variable: Variable name for the collection
            file_type: File type (auto-detected if None)
            **kwargs: Additional options

        Returns:
            TimeSeriesCollection
        """
        times, values, metadata = self.read_file(filepath, file_type, **kwargs)

        if column_ids is None:
            column_ids = [str(cid) for cid in metadata.column_ids]

        # Ensure values is 2D
        if values.ndim == 1:
            values = values.reshape(-1, 1)

        collection = TimeSeriesCollection(variable=variable)

        for i, col_id in enumerate(column_ids):
            if i < values.shape[1]:
                ts = TimeSeries(
                    times=times,
                    values=values[:, i],
                    location=col_id,
                )
                collection.add(ts)

        return collection

    def read_metadata(
        self,
        filepath: Path | str,
        file_type: TimeSeriesFileType | None = None,
        **kwargs: Any,
    ) -> TimeSeriesMetadata:
        """
        Read only metadata without loading full data.

        Args:
            filepath: Path to the file
            file_type: File type (auto-detected if None)
            **kwargs: Additional options

        Returns:
            TimeSeriesMetadata
        """
        filepath = Path(filepath)
        file_type = file_type or self._detect_format(filepath)
        adapter = self._get_adapter(file_type)
        return adapter.read_metadata(filepath, **kwargs)

    def _detect_format(self, filepath: Path) -> TimeSeriesFileType:
        """Detect file format from extension."""
        suffix = filepath.suffix.lower()

        if suffix in (".dss",):
            return TimeSeriesFileType.DSS
        elif suffix in (".hdf", ".h5", ".hdf5"):
            return TimeSeriesFileType.HDF5
        elif suffix in (".bin",):
            return TimeSeriesFileType.BINARY
        else:
            # Default to ASCII for .dat, .txt, .in, etc.
            return TimeSeriesFileType.ASCII

    def _get_adapter(self, file_type: TimeSeriesFileType) -> BaseTimeSeriesReader:
        """Get adapter for file type."""
        if file_type not in self._adapters:
            raise ValueError(
                f"No adapter available for {file_type.value}. "
                f"Available formats: {list(self._adapters.keys())}"
            )
        return self._adapters[file_type]

    def _apply_time_filter(
        self,
        times: NDArray[np.datetime64],
        values: NDArray[np.float64],
        start: datetime | None,
        end: datetime | None,
    ) -> tuple[NDArray[np.datetime64], NDArray[np.float64]]:
        """Apply time filter to data."""
        mask = np.ones(len(times), dtype=bool)

        if start:
            start_np = np.datetime64(start)
            mask &= times >= start_np

        if end:
            end_np = np.datetime64(end)
            mask &= times <= end_np

        return times[mask], values[mask] if values.ndim == 1 else values[mask, :]


class RecyclingTimeSeriesReader:
    """
    Reader that handles IWFM time series recycling.

    IWFM supports repeating time series patterns. For example, monthly
    data for a single year can be recycled across the entire simulation period.
    """

    def __init__(self, base_reader: UnifiedTimeSeriesReader | None = None) -> None:
        """Initialize with optional base reader."""
        self._reader = base_reader or UnifiedTimeSeriesReader()

    def read_with_recycling(
        self,
        filepath: Path | str,
        target_times: NDArray[np.datetime64],
        recycling_interval: int = 12,
        **kwargs: Any,
    ) -> NDArray[np.float64]:
        """
        Read time series and recycle to match target times.

        Args:
            filepath: Source file path
            target_times: Target timestamps to generate values for
            recycling_interval: Number of months to recycle (12 = yearly)
            **kwargs: Additional read options

        Returns:
            Values array matching target_times
        """
        times, values, _ = self._reader.read_file(filepath, **kwargs)

        if len(times) == 0:
            return np.zeros((len(target_times), values.shape[1] if values.ndim > 1 else 1))

        # Build recycled values
        result = np.zeros((len(target_times), values.shape[1] if values.ndim > 1 else 1))

        # Convert times to month indices
        source_months = np.array([
            (t.astype("datetime64[M]") - t.astype("datetime64[Y]")).astype(int)
            for t in times
        ])

        for i, target_time in enumerate(target_times):
            target_month = (
                target_time.astype("datetime64[M]") - target_time.astype("datetime64[Y]")
            ).astype(int)

            # Find matching source index
            source_idx = np.where(source_months == target_month % recycling_interval)[0]
            if len(source_idx) > 0:
                result[i] = values[source_idx[0]] if values.ndim == 1 else values[source_idx[0], :]

        return result


# Convenience functions

def detect_timeseries_format(filepath: Path | str) -> TimeSeriesFileType:
    """
    Detect time series file format.

    Args:
        filepath: Path to file

    Returns:
        TimeSeriesFileType enum value
    """
    reader = UnifiedTimeSeriesReader()
    return reader._detect_format(Path(filepath))


def read_timeseries_unified(
    filepath: Path | str,
    file_type: TimeSeriesFileType | None = None,
    **kwargs: Any,
) -> tuple[NDArray[np.datetime64], NDArray[np.float64], TimeSeriesMetadata]:
    """
    Read time series from any supported format.

    Args:
        filepath: Path to file
        file_type: Optional explicit file type
        **kwargs: Format-specific options

    Returns:
        Tuple of (times, values, metadata)
    """
    reader = UnifiedTimeSeriesReader()
    return reader.read_file(filepath, file_type, **kwargs)


def get_timeseries_metadata(
    filepath: Path | str,
    file_type: TimeSeriesFileType | None = None,
    **kwargs: Any,
) -> TimeSeriesMetadata:
    """
    Get time series metadata without reading full data.

    Args:
        filepath: Path to file
        file_type: Optional explicit file type
        **kwargs: Format-specific options

    Returns:
        TimeSeriesMetadata
    """
    reader = UnifiedTimeSeriesReader()
    return reader.read_metadata(filepath, file_type, **kwargs)
