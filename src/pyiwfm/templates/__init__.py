"""Jinja2 template engine for IWFM file generation."""

from __future__ import annotations

from pyiwfm.templates.engine import (
    IWFMFileWriter,
    TemplateEngine,
)
from pyiwfm.templates.filters import (
    dss_date_part,
    dss_interval,
    dss_pathname,
    dss_timeseries_ref,
    fortran_float,
    fortran_int,
    fortran_scientific,
    iwfm_array_row,
    iwfm_comment,
    iwfm_data_row,
    iwfm_date,
    iwfm_path,
    iwfm_time_unit,
    iwfm_timestamp,
    iwfm_value,
    pad_left,
    pad_right,
    register_all_filters,
    timeseries_ref,
)

__all__ = [
    # Engine
    "TemplateEngine",
    "IWFMFileWriter",
    # Number formatting
    "fortran_float",
    "fortran_int",
    "fortran_scientific",
    # IWFM formatting
    "iwfm_comment",
    "iwfm_value",
    "iwfm_path",
    # Time formatting
    "iwfm_timestamp",
    "iwfm_date",
    "iwfm_time_unit",
    # DSS formatting
    "dss_pathname",
    "dss_date_part",
    "dss_interval",
    # Array formatting
    "iwfm_array_row",
    "iwfm_data_row",
    # String formatting
    "pad_right",
    "pad_left",
    # References
    "timeseries_ref",
    "dss_timeseries_ref",
    # Registration
    "register_all_filters",
]
