"""
Generic IWFM time series data file writer.

All IWFM time series files (pumping, inflows, ET, precip, diversions,
crop coefficients, return flow fractions, reuse fractions, irrigation
periods, ag water demand, max lake elevation, stream surface area, etc.)
share a common structure:

    - Comment header
    - NCOL, FACT, [TUNIT], NSP, NFQ, DSSFL
    - Optional column mapping section
    - Date-indexed data rows  *or*  DSS pathnames

This module provides a single writer + dataclass that handles all variants.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import IO, Any

from numpy.typing import NDArray

from pyiwfm.templates.engine import TemplateEngine

logger = logging.getLogger(__name__)


@dataclass
class DSSPathItem:
    """A single DSS pathname entry for column mapping."""

    index: int
    path: str


@dataclass
class TimeSeriesDataConfig:
    """Configuration for a generic IWFM time series data file.

    This covers all IWFM TS file types: pumping, inflows, ET, precip,
    diversions, crop coefficients, return flow fractions, reuse fractions,
    irrigation periods, ag water demand, max lake elevation, stream
    surface area, generic moisture, etc.
    """

    title: str = ""
    ncol: int = 0
    factor: float = 1.0
    time_unit: str = ""
    has_time_unit: bool = False
    nsp: int = 1
    nfq: int = 0
    dss_file: str = ""

    # Tags (vary by file type)
    ncol_tag: str = "NCOL"
    factor_tag: str = "FACT"
    time_unit_tag: str = "TUNIT"
    nsp_tag: str = "NSP"
    nfq_tag: str = "NFQ"

    # Description lines for header comments
    description_lines: list[str] = field(default_factory=list)

    # Column mapping (list of pre-formatted strings)
    column_mapping: list[str] = field(default_factory=list)
    column_header: str = ""

    # Data
    dates: list[str] | None = None
    data: NDArray | None = None
    data_header: str = "Time Series Data"
    data_fmt: str = "%14.6f"

    # DSS mode
    dss_paths: list[DSSPathItem] = field(default_factory=list)

    @property
    def use_dss(self) -> bool:
        """Whether to write DSS pathname references instead of inline data."""
        return bool(self.dss_paths)


class IWFMTimeSeriesDataWriter:
    """Generic writer for IWFM time series data files.

    Handles: TSPumping, StreamInflow, DiversionData, Precip, ET,
    CropCoeff, IrrigPeriod, ReturnFlowFrac, ReuseFrac, AgWaterDemand,
    MaxLakeElev, StreamSurfaceArea, GenericMoisture, etc.

    Example
    -------
    >>> from pyiwfm.io.timeseries_writer import (
    ...     IWFMTimeSeriesDataWriter, TimeSeriesDataConfig,
    ... )
    >>> config = TimeSeriesDataConfig(
    ...     title="Pumping Time Series",
    ...     ncol=3,
    ...     factor=1.0,
    ...     nsp=1,
    ...     nfq=0,
    ...     ncol_tag="NCOLPUMP",
    ...     factor_tag="FACTPUMP",
    ...     nsp_tag="NSPPUMP",
    ...     nfq_tag="NFQPUMP",
    ...     dates=["10/01/1990_24:00", "10/02/1990_24:00"],
    ...     data=np.array([[100.0, 200.0, 300.0],
    ...                    [110.0, 210.0, 310.0]]),
    ... )
    >>> writer = IWFMTimeSeriesDataWriter()
    >>> writer.write(config, Path("TSPumping.dat"))
    """

    def __init__(self, engine: TemplateEngine | None = None) -> None:
        self._engine = engine or TemplateEngine()

    def write(self, config: TimeSeriesDataConfig, filepath: Path) -> Path:
        """Write a complete IWFM time series data file.

        Parameters
        ----------
        config : TimeSeriesDataConfig
            File specification
        filepath : Path
            Output file path

        Returns
        -------
        Path
            Path to written file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build description lines if not provided
        desc_lines = config.description_lines or [
            f"{config.ncol_tag:<12}: Number of data columns",
            f"{config.factor_tag:<12}: Conversion factor for data",
            f"{config.nsp_tag:<12}: Update frequency",
            f"{config.nfq_tag:<12}: Repetition frequency",
            "DSSFL     : DSS filename (blank = inline data)",
        ]

        context = {
            "title": config.title or "IWFM Time Series Data",
            "generation_time": generation_time,
            "description_lines": desc_lines,
            "ncol": config.ncol,
            "factor": config.factor,
            "has_time_unit": config.has_time_unit,
            "time_unit": config.time_unit,
            "nsp": config.nsp,
            "nfq": config.nfq,
            "dss_file": config.dss_file,
            "ncol_tag": config.ncol_tag,
            "factor_tag": config.factor_tag,
            "time_unit_tag": config.time_unit_tag,
            "nsp_tag": config.nsp_tag,
            "nfq_tag": config.nfq_tag,
            "column_mapping": config.column_mapping,
            "column_header": config.column_header,
            "use_dss": config.use_dss,
            "dss_paths": config.dss_paths,
            "data_header": config.data_header,
        }

        # Render header via template
        header = self._engine.render_template("timeseries/timeseries_data.j2", **context)

        with open(filepath, "w") as f:
            f.write(header)

            # Write data rows (date + values)
            if not config.use_dss and config.dates is not None and config.data is not None:
                self._write_data_rows(f, config.dates, config.data, config.data_fmt)

        logger.info("Wrote time series data file: %s", filepath)
        return filepath

    def write_dss_mode(self, config: TimeSeriesDataConfig, filepath: Path) -> Path:
        """Write with DSS pathname references instead of inline data.

        This is a convenience wrapper that ensures use_dss is True.

        Parameters
        ----------
        config : TimeSeriesDataConfig
            File specification (must have dss_paths populated)
        filepath : Path
            Output file path

        Returns
        -------
        Path
            Path to written file
        """
        if not config.dss_paths:
            raise ValueError("dss_paths must be populated for DSS mode")
        return self.write(config, filepath)

    @staticmethod
    def _write_data_rows(
        f: IO[str],
        dates: list[str],
        data: NDArray,
        fmt: str = "%14.6f",
    ) -> None:
        """Write date-indexed data rows efficiently using numpy.

        Parameters
        ----------
        f : file object
            Open file for writing
        dates : list[str]
            IWFM timestamp strings (e.g. "10/01/1990_24:00")
        data : NDArray
            Data array, shape (n_times,) or (n_times, n_cols)
        fmt : str
            Printf-style format for data values
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_times = len(dates)
        n_cols = data.shape[1]

        for i in range(n_times):
            # Pad date to 16 chars for alignment
            date_str = dates[i].ljust(16)
            values_str = " ".join(fmt % data[i, j] for j in range(n_cols))
            f.write(f"    {date_str} {values_str}\n")


# ---------------------------------------------------------------------------
# Factory helpers for common IWFM time series file types
# ---------------------------------------------------------------------------


def make_pumping_ts_config(
    ncol: int,
    factor: float = 1.0,
    nsp: int = 1,
    nfq: int = 0,
    **kwargs: Any,
) -> TimeSeriesDataConfig:
    """Create a TimeSeriesDataConfig pre-configured for pumping TS files."""
    return TimeSeriesDataConfig(
        title="Pumping Time Series Data",
        ncol=ncol,
        factor=factor,
        nsp=nsp,
        nfq=nfq,
        ncol_tag="NCOLPUMP",
        factor_tag="FACTPUMP",
        nsp_tag="NSPPUMP",
        nfq_tag="NFQPUMP",
        data_header="Pumping Data (Date  Col1  Col2  ...)",
        **kwargs,
    )


def make_stream_inflow_ts_config(
    ncol: int,
    factor: float = 1.0,
    nsp: int = 1,
    nfq: int = 0,
    **kwargs: Any,
) -> TimeSeriesDataConfig:
    """Create a TimeSeriesDataConfig pre-configured for stream inflow TS."""
    return TimeSeriesDataConfig(
        title="Stream Inflow Time Series Data",
        ncol=ncol,
        factor=factor,
        nsp=nsp,
        nfq=nfq,
        ncol_tag="NCOLSTRM",
        factor_tag="FACTSTRM",
        nsp_tag="NSPSTRM",
        nfq_tag="NFQSTRM",
        data_header="Stream Inflow Data (Date  Col1  Col2  ...)",
        **kwargs,
    )


def make_diversion_ts_config(
    ncol: int,
    factor: float = 1.0,
    nsp: int = 1,
    nfq: int = 0,
    **kwargs: Any,
) -> TimeSeriesDataConfig:
    """Create a TimeSeriesDataConfig pre-configured for diversion data TS."""
    return TimeSeriesDataConfig(
        title="Diversion Data Time Series",
        ncol=ncol,
        factor=factor,
        nsp=nsp,
        nfq=nfq,
        ncol_tag="NCOLDV",
        factor_tag="FACTDV",
        nsp_tag="NSPDV",
        nfq_tag="NFQDV",
        data_header="Diversion Data (Date  Col1  Col2  ...)",
        **kwargs,
    )


def make_precip_ts_config(
    ncol: int,
    factor: float = 1.0,
    nsp: int = 1,
    nfq: int = 0,
    **kwargs: Any,
) -> TimeSeriesDataConfig:
    """Create a TimeSeriesDataConfig pre-configured for precipitation TS."""
    return TimeSeriesDataConfig(
        title="Precipitation Time Series Data",
        ncol=ncol,
        factor=factor,
        nsp=nsp,
        nfq=nfq,
        ncol_tag="NRAIN",
        factor_tag="FACTRN",
        nsp_tag="NSPRN",
        nfq_tag="NFQRN",
        data_header="Precipitation Data (Date  Col1  Col2  ...)",
        **kwargs,
    )


def make_et_ts_config(
    ncol: int,
    factor: float = 1.0,
    nsp: int = 1,
    nfq: int = 0,
    **kwargs: Any,
) -> TimeSeriesDataConfig:
    """Create a TimeSeriesDataConfig pre-configured for ET TS files."""
    return TimeSeriesDataConfig(
        title="Evapotranspiration Time Series Data",
        ncol=ncol,
        factor=factor,
        nsp=nsp,
        nfq=nfq,
        ncol_tag="NCOLET",
        factor_tag="FACTET",
        nsp_tag="NSPET",
        nfq_tag="NFQET",
        data_header="ET Data (Date  Col1  Col2  ...)",
        **kwargs,
    )


def make_crop_coeff_ts_config(
    ncol: int,
    nsp: int = 1,
    nfq: int = 0,
    **kwargs: Any,
) -> TimeSeriesDataConfig:
    """Create a TimeSeriesDataConfig pre-configured for crop coeff TS."""
    return TimeSeriesDataConfig(
        title="Crop Coefficient Time Series Data",
        ncol=ncol,
        factor=1.0,
        nsp=nsp,
        nfq=nfq,
        ncol_tag="NCFF",
        factor_tag="FACTCFF",
        nsp_tag="NSPCFF",
        nfq_tag="NFQCFF",
        data_header="Crop Coefficient Data (Date  Col1  Col2  ...)",
        **kwargs,
    )


def make_return_flow_ts_config(
    ncol: int,
    nsp: int = 1,
    nfq: int = 0,
    **kwargs: Any,
) -> TimeSeriesDataConfig:
    """Create a TimeSeriesDataConfig for return flow fraction TS."""
    return TimeSeriesDataConfig(
        title="Return Flow Fraction Time Series Data",
        ncol=ncol,
        factor=1.0,
        nsp=nsp,
        nfq=nfq,
        ncol_tag="NCOLRT",
        factor_tag="FACTRT",
        nsp_tag="NSPRT",
        nfq_tag="NFQRT",
        data_header="Return Flow Fractions (Date  Col1  Col2  ...)",
        **kwargs,
    )


def make_reuse_ts_config(
    ncol: int,
    nsp: int = 1,
    nfq: int = 0,
    **kwargs: Any,
) -> TimeSeriesDataConfig:
    """Create a TimeSeriesDataConfig for reuse fraction TS."""
    return TimeSeriesDataConfig(
        title="Reuse Fraction Time Series Data",
        ncol=ncol,
        factor=1.0,
        nsp=nsp,
        nfq=nfq,
        ncol_tag="NCOLRUF",
        factor_tag="FACTRUF",
        nsp_tag="NSPRUF",
        nfq_tag="NFQRUF",
        data_header="Reuse Fractions (Date  Col1  Col2  ...)",
        **kwargs,
    )


def make_irig_period_ts_config(
    ncol: int,
    nsp: int = 1,
    nfq: int = 0,
    **kwargs: Any,
) -> TimeSeriesDataConfig:
    """Create a TimeSeriesDataConfig for irrigation period TS."""
    return TimeSeriesDataConfig(
        title="Irrigation Period Time Series Data",
        ncol=ncol,
        factor=1.0,
        nsp=nsp,
        nfq=nfq,
        ncol_tag="NCOLIP",
        factor_tag="FACTIP",
        nsp_tag="NSPIP",
        nfq_tag="NFQIP",
        data_header="Irrigation Period Data (Date  Col1  Col2  ...)",
        **kwargs,
    )


def make_ag_water_demand_ts_config(
    ncol: int,
    factor: float = 1.0,
    nsp: int = 1,
    nfq: int = 0,
    **kwargs: Any,
) -> TimeSeriesDataConfig:
    """Create a TimeSeriesDataConfig for ag water demand TS."""
    return TimeSeriesDataConfig(
        title="Agricultural Water Demand Time Series Data",
        ncol=ncol,
        factor=factor,
        nsp=nsp,
        nfq=nfq,
        ncol_tag="NDMAG",
        factor_tag="FACTDMAG",
        nsp_tag="NSPDMAG",
        nfq_tag="NFQDMAG",
        data_header="Ag Water Demand Data (Date  Col1  Col2  ...)",
        **kwargs,
    )


def make_max_lake_elev_ts_config(
    ncol: int,
    factor: float = 1.0,
    nsp: int = 1,
    nfq: int = 0,
    **kwargs: Any,
) -> TimeSeriesDataConfig:
    """Create a TimeSeriesDataConfig for max lake elevation TS."""
    return TimeSeriesDataConfig(
        title="Maximum Lake Elevation Time Series Data",
        ncol=ncol,
        factor=factor,
        nsp=nsp,
        nfq=nfq,
        ncol_tag="NCOLLK",
        factor_tag="FACTLK",
        nsp_tag="NSPLK",
        nfq_tag="NFQLK",
        data_header="Max Lake Elevation Data (Date  Col1  Col2  ...)",
        **kwargs,
    )


def make_stream_surface_area_ts_config(
    ncol: int,
    factor: float = 1.0,
    has_time_unit: bool = True,
    time_unit: str = "1DAY",
    nsp: int = 1,
    nfq: int = 0,
    **kwargs: Any,
) -> TimeSeriesDataConfig:
    """Create a TimeSeriesDataConfig for stream surface area TS."""
    return TimeSeriesDataConfig(
        title="Stream Surface Area Time Series Data",
        ncol=ncol,
        factor=factor,
        has_time_unit=has_time_unit,
        time_unit=time_unit,
        nsp=nsp,
        nfq=nfq,
        ncol_tag="NCOLSA",
        factor_tag="FACTSA",
        time_unit_tag="TUNITSA",
        nsp_tag="NSPSA",
        nfq_tag="NFQSA",
        data_header="Stream Surface Area Data (Date  Col1  Col2  ...)",
        **kwargs,
    )
