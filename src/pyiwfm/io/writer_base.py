"""
Enhanced base writer classes for IWFM file generation.

This module provides abstract base classes for writing IWFM files,
supporting both text and DSS output formats, with optional comment
preservation for round-trip operations.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

import numpy as np
from numpy.typing import NDArray

from pyiwfm.io.config import OutputFormat, TimeSeriesOutputConfig
from pyiwfm.io.iwfm_writer import ensure_parent_dir
from pyiwfm.templates.engine import TemplateEngine

if TYPE_CHECKING:
    from pyiwfm.core.model import IWFMModel
    from pyiwfm.io.comment_metadata import CommentMetadata
    from pyiwfm.io.comment_writer import CommentWriter

logger = logging.getLogger(__name__)


# =============================================================================
# DSS Support (Optional)
# =============================================================================

try:
    from pyiwfm.io.dss import DSSFile, write_timeseries_to_dss  # noqa: F401

    HAS_DSS = True
except ImportError:
    HAS_DSS = False


def _check_dss() -> None:
    """Check if DSS support is available."""
    if not HAS_DSS:
        raise ImportError(
            "DSS support requires the bundled HEC-DSS library (pyiwfm.io.dss). "
            "The library may not be available on this platform."
        )


# =============================================================================
# Template Writer Base Class
# =============================================================================


class TemplateWriter(ABC):
    """
    Base class for template-based IWFM file writers.

    Uses Jinja2 templates for file headers and structure,
    with numpy for efficient large data array output.

    Supports optional comment preservation via CommentMetadata.
    When comment_metadata is provided, preserved comments from
    the original file will be injected into the output.
    """

    def __init__(
        self,
        output_dir: Path | str,
        template_engine: TemplateEngine | None = None,
        comment_metadata: CommentMetadata | None = None,
    ) -> None:
        """
        Initialize the template writer.

        Args:
            output_dir: Directory for output files
            template_engine: Optional TemplateEngine instance
            comment_metadata: Optional CommentMetadata for comment preservation
        """
        self.output_dir = Path(output_dir)
        self._engine = template_engine or TemplateEngine()
        self.comment_metadata = comment_metadata
        self._comment_writer: CommentWriter | None = None

    def _ensure_dir(self, path: Path) -> None:
        """Ensure parent directory exists."""
        ensure_parent_dir(path)

    @property
    def comment_writer(self) -> CommentWriter:
        """Get or create the comment writer.

        Returns:
            CommentWriter instance configured with our comment_metadata.
        """
        if self._comment_writer is None:
            from pyiwfm.io.comment_writer import CommentWriter

            self._comment_writer = CommentWriter(
                self.comment_metadata,
                use_fallback=True,
            )
        return self._comment_writer

    def has_preserved_comments(self) -> bool:
        """Check if preserved comments are available."""
        return self.comment_metadata is not None and self.comment_metadata.has_comments()

    def render_header(self, template_name: str, **context: Any) -> str:
        """
        Render a template header.

        Args:
            template_name: Name of the template file
            **context: Template context variables

        Returns:
            Rendered header string
        """
        return self._engine.render_template(template_name, **context)

    def render_header_with_comments(
        self,
        template_name: str,
        section_name: str | None = None,
        **context: Any,
    ) -> str:
        """
        Render a template header with preserved comments.

        If comment_metadata is available and contains preserved comments,
        those comments are used instead of or in addition to the template.

        Args:
            template_name: Name of the template file (fallback)
            section_name: Section name for looking up comments
            **context: Template context variables

        Returns:
            Header string with preserved or template-based comments
        """
        # Check for preserved header
        if self.has_preserved_comments():
            preserved_header = self.comment_writer.restore_header()
            if preserved_header:
                return preserved_header

        # Fall back to template
        return self._engine.render_template(template_name, **context)

    def render_string(self, template_str: str, **context: Any) -> str:
        """
        Render a template from a string.

        Args:
            template_str: Template string
            **context: Template context variables

        Returns:
            Rendered string
        """
        return self._engine.render_string(template_str, **context)

    def write_data_block(
        self,
        file: TextIO,
        data: NDArray,
        fmt: str | Sequence[str] = "%14.6f",
        header_comment: str | None = None,
    ) -> None:
        """
        Write a numpy array as a formatted data block.

        Args:
            file: Open file object
            data: Data array (1D or 2D)
            fmt: Format string(s) for values
            header_comment: Optional comment line before data
        """
        if header_comment:
            file.write(f"C  {header_comment}\n")

        if isinstance(fmt, str):
            np.savetxt(file, data, fmt=fmt)
        else:
            # Multiple format strings - join them
            fmt_str = " ".join(fmt)
            np.savetxt(file, data, fmt=fmt_str)

    def write_indexed_data(
        self,
        file: TextIO,
        ids: NDArray[np.int32],
        data: NDArray,
        id_fmt: str = "%5d",
        data_fmt: str = "%14.6f",
    ) -> None:
        """
        Write indexed data (ID column + data columns).

        Args:
            file: Open file object
            ids: Row IDs
            data: Data values (can be 1D or 2D)
            id_fmt: Format for ID column
            data_fmt: Format for data columns
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_cols = data.shape[1]
        fmt = id_fmt + " " + " ".join([data_fmt] * n_cols)
        combined = np.column_stack([ids.astype(float), data])
        np.savetxt(file, combined, fmt=fmt)

    @abstractmethod
    def write(self, data: Any) -> None:
        """
        Write data to file(s).

        Args:
            data: Data to write (type depends on subclass)
        """
        pass

    @property
    @abstractmethod
    def format(self) -> str:
        """Return the file format identifier."""
        pass


# =============================================================================
# Time Series Writer with DSS Support
# =============================================================================


@dataclass
class TimeSeriesSpec:
    """Specification for a time series to write."""

    name: str
    dates: Sequence[datetime] | NDArray
    values: Sequence[float] | NDArray
    units: str = ""
    location: str = ""
    parameter: str = ""
    interval: str = "1DAY"


class TimeSeriesWriter:
    """
    Writer for time series data with text/DSS format support.

    Can write time series to:
    - ASCII text files (one or more columns)
    - HEC-DSS files (requires heclib)
    """

    def __init__(
        self,
        output_config: TimeSeriesOutputConfig,
        output_dir: Path | str,
    ) -> None:
        """
        Initialize the time series writer.

        Args:
            output_config: Configuration for output format
            output_dir: Directory for output files
        """
        self.config = output_config
        self.output_dir = Path(output_dir)
        self._dss_file = None

    def write_timeseries(
        self,
        ts_spec: TimeSeriesSpec,
        text_file: str | Path | None = None,
    ) -> None:
        """
        Write a single time series.

        Args:
            ts_spec: Time series specification
            text_file: Text file name (required if format includes TEXT)
        """
        if self.config.format in (OutputFormat.TEXT, OutputFormat.BOTH):
            if text_file is None:
                raise ValueError("text_file required for TEXT output format")
            self._write_text_timeseries(ts_spec, text_file)

        if self.config.format in (OutputFormat.DSS, OutputFormat.BOTH):
            _check_dss()
            self._write_dss_timeseries(ts_spec)

    def write_timeseries_table(
        self,
        dates: Sequence[datetime] | NDArray,
        columns: dict[str, NDArray],
        text_file: str | Path,
        header_lines: list[str] | None = None,
    ) -> None:
        """
        Write multiple time series columns to a single text file.

        Args:
            dates: Date/time values
            columns: Dict mapping column name to values
            text_file: Output file name
            header_lines: Optional header comment lines
        """
        if self.config.format not in (OutputFormat.TEXT, OutputFormat.BOTH):
            return

        output_path = self.output_dir / text_file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            # Write header
            if header_lines:
                for line in header_lines:
                    f.write(f"C  {line}\n")
                f.write("C\n")

            # Write column headers
            col_names = list(columns.keys())
            f.write(f"C  {'DATE/TIME':<21}")
            for name in col_names:
                f.write(f" {name:>14}")
            f.write("\n")

            # Write data rows
            from pyiwfm.io.timeseries_ascii import format_iwfm_timestamp

            for i, dt in enumerate(dates):
                ts_str = format_iwfm_timestamp(dt)
                f.write(f"{ts_str:<21}")
                for name in col_names:
                    f.write(f" {columns[name][i]:>14.6f}")
                f.write("\n")

        logger.info(f"Wrote time series table to {output_path}")

    def _write_text_timeseries(
        self,
        ts_spec: TimeSeriesSpec,
        text_file: str | Path,
    ) -> None:
        """Write time series to text file."""
        output_path = self.output_dir / text_file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            # Header
            f.write(f"C  Time series: {ts_spec.name}\n")
            if ts_spec.units:
                f.write(f"C  Units: {ts_spec.units}\n")
            if ts_spec.location:
                f.write(f"C  Location: {ts_spec.location}\n")
            f.write("C\n")
            f.write(f"C  {'DATE/TIME':<21} {'VALUE':>14}\n")

            # Data
            from pyiwfm.io.timeseries_ascii import format_iwfm_timestamp

            for dt, val in zip(ts_spec.dates, ts_spec.values, strict=False):
                ts_str = format_iwfm_timestamp(dt)
                f.write(f"{ts_str:<21} {val:>14.6f}\n")

        logger.info(f"Wrote time series to {output_path}")

    def _write_dss_timeseries(self, ts_spec: TimeSeriesSpec) -> None:
        """Write time series to DSS file."""
        if not self.config.dss_file:
            raise ValueError("DSS file not configured")

        dss_path = self.output_dir / self.config.dss_file

        # Build DSS pathname
        a_part = self.config.dss_a_part or ""
        b_part = ts_spec.location or ts_spec.name
        c_part = ts_spec.parameter or "VALUE"
        d_part = ""  # Date range - DSS fills this in
        e_part = ts_spec.interval
        f_part = self.config.dss_f_part or "PYIWFM"

        pathname = f"/{a_part}/{b_part}/{c_part}/{d_part}/{e_part}/{f_part}/"

        # Convert dates to DSS format
        dates = ts_spec.dates
        values = np.asarray(ts_spec.values)

        from pyiwfm.core.timeseries import TimeSeries

        ts_obj = TimeSeries(
            times=np.asarray(dates, dtype="datetime64[ns]"),
            values=values,
        )
        write_timeseries_to_dss(  # type: ignore[arg-type]
            str(dss_path),
            ts_obj,
            pathname,
            units=ts_spec.units,
        )

        logger.info(f"Wrote time series to DSS: {pathname}")

    def close(self) -> None:
        """Close any open DSS files."""
        if self._dss_file is not None:
            self._dss_file.close()
            self._dss_file = None


# =============================================================================
# Component Writer Base Class
# =============================================================================


class ComponentWriter(TemplateWriter):
    """
    Base class for IWFM component writers (GW, Streams, Lakes, etc.).

    Provides common functionality for writing component input files
    with support for both text and DSS time series formats.

    Supports optional comment preservation via CommentMetadata.
    """

    def __init__(
        self,
        output_dir: Path | str,
        ts_config: TimeSeriesOutputConfig | None = None,
        template_engine: TemplateEngine | None = None,
        comment_metadata: CommentMetadata | None = None,
    ) -> None:
        """
        Initialize the component writer.

        Args:
            output_dir: Directory for output files
            ts_config: Time series output configuration
            template_engine: Optional TemplateEngine instance
            comment_metadata: Optional CommentMetadata for comment preservation
        """
        super().__init__(output_dir, template_engine, comment_metadata)
        self.ts_config = ts_config or TimeSeriesOutputConfig()
        self._ts_writer: TimeSeriesWriter | None = None

    @property
    def ts_writer(self) -> TimeSeriesWriter:
        """Get or create the time series writer."""
        if self._ts_writer is None:
            self._ts_writer = TimeSeriesWriter(self.ts_config, self.output_dir)
        return self._ts_writer

    def write_component_header(
        self,
        file: TextIO,
        component_name: str,
        version: str = "",
        description: str = "",
    ) -> None:
        """
        Write a standard component file header.

        Args:
            file: Open file object
            component_name: Name of the component
            version: Optional version string
            description: Optional description
        """
        file.write(f"C  {component_name}\n")
        if version:
            file.write(f"C  Version: {version}\n")
        if description:
            file.write(f"C  {description}\n")
        file.write(f"C  Generated by pyiwfm on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write("C\n")

    def write_file_reference(
        self,
        file: TextIO,
        ref_path: Path | str | None,
        description: str = "",
    ) -> None:
        """
        Write a file path reference.

        Args:
            file: Open file object
            ref_path: Path to referenced file (None for blank)
            description: Description of the file
        """
        if ref_path is None or str(ref_path).strip() == "":
            path_str = ""
        else:
            path_str = str(ref_path).replace("\\", "/")

        if description:
            file.write(f"{path_str:<60}  / {description}\n")
        else:
            file.write(f"{path_str}\n")

    def write_value_line(
        self,
        file: TextIO,
        value: Any,
        description: str = "",
        width: int = 20,
    ) -> None:
        """
        Write a value with optional description.

        Args:
            file: Open file object
            value: Value to write
            description: Description of the value
            width: Width for value field
        """
        if isinstance(value, float):
            value_str = f"{value:.6f}"
        else:
            value_str = str(value)

        if description:
            file.write(f"{value_str:<{width}}  / {description}\n")
        else:
            file.write(f"{value_str}\n")

    @property
    def format(self) -> str:
        return "iwfm_component"


# =============================================================================
# Model Writer Base Class
# =============================================================================


class IWFMModelWriter(ABC):
    """
    Abstract base class for writing complete IWFM models.

    Coordinates all component writers to produce a complete,
    valid IWFM input file set.

    Supports optional comment preservation via FileCommentMetadata.
    """

    def __init__(
        self,
        model: IWFMModel,
        output_dir: Path | str,
        ts_format: OutputFormat = OutputFormat.TEXT,
        template_engine: TemplateEngine | None = None,
        comment_metadata: CommentMetadata | None = None,
    ) -> None:
        """
        Initialize the model writer.

        Args:
            model: IWFMModel instance to write
            output_dir: Directory for output files
            ts_format: Output format for time series (TEXT, DSS, or BOTH)
            template_engine: Optional TemplateEngine instance
            comment_metadata: Optional CommentMetadata for comment preservation
        """
        self.model = model
        self.output_dir = Path(output_dir)
        self.ts_format = ts_format
        self._engine = template_engine or TemplateEngine()
        self.comment_metadata = comment_metadata

    @abstractmethod
    def write_preprocessor(self) -> dict[str, Path]:
        """
        Write preprocessor files.

        Returns:
            Dict mapping file type to output path
        """
        pass

    @abstractmethod
    def write_simulation(self) -> dict[str, Path]:
        """
        Write simulation files.

        Returns:
            Dict mapping file type to output path
        """
        pass

    def write_all(self) -> dict[str, Path]:
        """
        Write complete model file set.

        Returns:
            Dict mapping file type to output path
        """
        results = {}

        # Preprocessor files
        pp_results = self.write_preprocessor()
        results.update({f"preprocessor_{k}": v for k, v in pp_results.items()})

        # Simulation files
        sim_results = self.write_simulation()
        results.update({f"simulation_{k}": v for k, v in sim_results.items()})

        return results

    def _ensure_directories(self) -> None:
        """Create output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
