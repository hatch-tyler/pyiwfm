"""Time-series readers, writers, and lazy loaders.

The v1.x flat modules ``pyiwfm.io.timeseries`` (the unified reader),
``pyiwfm.io.timeseries_ascii`` (the ASCII reader/writer pair),
``pyiwfm.io.timeseries_writer`` (the IWFM time-series data writer),
``pyiwfm.io.timeseries_io`` (lazy loaders + cache), and
``pyiwfm.io.timeseries_reader`` (the thin facade) are now collapsed
into one subpackage:

- :mod:`pyiwfm.io.timeseries.reader` — was ``timeseries.py``
  (``UnifiedTimeSeriesReader``, ``TimeSeriesFileType``, ``TimeUnit``,
  format adapters).
- :mod:`pyiwfm.io.timeseries.ascii` — was ``timeseries_ascii.py``
  (``TimeSeriesReader``, ``TimeSeriesWriter``, IWFM datetime helpers).
- :mod:`pyiwfm.io.timeseries.writer` — was ``timeseries_writer.py``
  (``IWFMTimeSeriesDataWriter`` + ``make_*_ts_config`` helpers).
- :mod:`pyiwfm.io.timeseries.lazy` — was ``timeseries_io.py``
  (``LazyNodalLoader``, ``LazyTabularLoader``, ``TimeSeriesCache``).
- :mod:`pyiwfm.io.timeseries.compat` — was ``timeseries_reader.py``
  (``IWFMTimeSeriesData``, ``read_iwfm_timeseries``).

The package re-exports every public symbol. The four v1.x sibling
paths are gone in v2.0; use ``from pyiwfm.io.timeseries import X``
instead. See ``docs/MIGRATION_v1_to_v2.md`` § 10.
"""

from __future__ import annotations

from pyiwfm.io.timeseries.ascii import (
    TimeSeriesFileConfig,
    TimeSeriesReader,
    TimeSeriesWriter,
    format_iwfm_timestamp,
    iwfm_date_to_iso,
    parse_iwfm_datetime,
    parse_iwfm_timestamp,
    read_timeseries,
    write_timeseries,
)
from pyiwfm.io.timeseries.compat import (
    IWFMTimeSeriesData,
    read_iwfm_timeseries,
)
from pyiwfm.io.timeseries.lazy import (
    LazyNodalLoader,
    LazyTabularLoader,
    TimeSeriesCache,
)
from pyiwfm.io.timeseries.reader import (
    AsciiTimeSeriesAdapter,
    BaseTimeSeriesReader,
    DssTimeSeriesAdapter,
    Hdf5TimeSeriesAdapter,
    RecyclingTimeSeriesReader,
    TimeSeriesFileType,
    TimeSeriesMetadata,
    TimeUnit,
    UnifiedTimeSeriesConfig,
    UnifiedTimeSeriesReader,
    detect_timeseries_format,
    get_timeseries_metadata,
    read_timeseries_unified,
)
from pyiwfm.io.timeseries.writer import (
    DSSPathItem,
    IWFMTimeSeriesDataWriter,
    TimeSeriesDataConfig,
    make_ag_water_demand_ts_config,
    make_crop_coeff_ts_config,
    make_diversion_ts_config,
    make_et_ts_config,
    make_irig_period_ts_config,
    make_max_lake_elev_ts_config,
    make_precip_ts_config,
    make_pumping_ts_config,
    make_return_flow_ts_config,
    make_reuse_ts_config,
    make_stream_inflow_ts_config,
    make_stream_surface_area_ts_config,
)

__all__ = [
    # reader.py (unified format)
    "AsciiTimeSeriesAdapter",
    "BaseTimeSeriesReader",
    "DssTimeSeriesAdapter",
    "Hdf5TimeSeriesAdapter",
    "RecyclingTimeSeriesReader",
    "TimeSeriesFileType",
    "TimeSeriesMetadata",
    "TimeUnit",
    "UnifiedTimeSeriesConfig",
    "UnifiedTimeSeriesReader",
    "detect_timeseries_format",
    "get_timeseries_metadata",
    "read_timeseries_unified",
    # ascii.py
    "TimeSeriesFileConfig",
    "TimeSeriesReader",
    "TimeSeriesWriter",
    "format_iwfm_timestamp",
    "iwfm_date_to_iso",
    "parse_iwfm_datetime",
    "parse_iwfm_timestamp",
    "read_timeseries",
    "write_timeseries",
    # writer.py (data-config helpers)
    "DSSPathItem",
    "IWFMTimeSeriesDataWriter",
    "TimeSeriesDataConfig",
    "make_ag_water_demand_ts_config",
    "make_crop_coeff_ts_config",
    "make_diversion_ts_config",
    "make_et_ts_config",
    "make_irig_period_ts_config",
    "make_max_lake_elev_ts_config",
    "make_precip_ts_config",
    "make_pumping_ts_config",
    "make_return_flow_ts_config",
    "make_reuse_ts_config",
    "make_stream_inflow_ts_config",
    "make_stream_surface_area_ts_config",
    # lazy.py
    "LazyNodalLoader",
    "LazyTabularLoader",
    "TimeSeriesCache",
    # compat.py
    "IWFMTimeSeriesData",
    "read_iwfm_timeseries",
]
